//! Scalar reference of the Gemma 4 graph (`reference.cpp`; llama.cpp
//! 391fac16 `src/models/gemma4.cpp`). Weights are dequantized on the fly,
//! every reduction accumulates in `f64`, and every layer keeps full K/V for
//! all positions, so it only suits short prefixes. It pins operator
//! semantics: scaled embeddings, weighted q/k norms, the unweighted V norm and
//! K=V on global layers, NEOX rope with per-layer bases and frequency
//! divisors, window-masked attention with scale 1, sandwich norms, GeGLU,
//! routed experts (26B-A4B), the layer output scale, and the final logit
//! softcap. Matrix products run on all hardware threads (OpenMP in C++).

use gufo_formats::GgmlType;
use gufo_hip::F16;

use crate::dequant::dequantize_row;
use crate::tokenizer::TokenId;
use crate::weights::{GgufShard, LayerWeights, ModelWeights, TensorRef};

/// How K/V and activations are rounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    /// K/V stay exact.
    Float32,
    /// Stored K/V round through binary16, as the GPU cache does.
    HalfKv,
    /// Additionally every input of a quantized projection rounds to Q8_1
    /// blocks (32 values, binary16 scale), the activation format of
    /// llama.cpp's and Gufo's integer GEMM kernels; it separates operator
    /// semantics from activation rounding.
    Q8Activations,
}

/// Rows `[row, row + count)` of one forward hold an image: `embedding`
/// (`[count][hidden]`, the encoder output) replaces their token embeddings
/// unscaled, and in sliding layers each of them sees every key of its image
/// (the window still bounds older keys); global layers stay causal.
#[derive(Debug, Clone, Copy)]
pub struct ReferenceImage<'e> {
    pub row: u32,
    pub count: u32,
    pub embedding: &'e [f32],
}

/// Round-to-nearest-even through binary16, including subnormals.
fn round_half(x: f32) -> f32 {
    F16::from_f32(x).to_f32()
}

/// y = x * rsqrt(mean(x^2) + eps) * w; `w` empty means an unweighted norm.
fn rms_norm(x: &[f32], w: &[f32], eps: f32, y: &mut [f32]) {
    let n = x.len();
    let sum: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let scale = 1.0 / (sum / n as f64 + f64::from(eps)).sqrt();
    for i in 0..n {
        let v = f64::from(x[i]) * scale;
        y[i] = if w.is_empty() {
            v as f32
        } else {
            (v * f64::from(w[i])) as f32
        };
    }
}

/// In-place form of [`rms_norm`].
fn rms_norm_in_place(x: &mut [f32], w: &[f32], eps: f32) {
    let copy = x.to_vec();
    rms_norm(&copy, w, eps, x);
}

/// NEOX rotation of one head: pair (i, i + dim/2) turns by
/// pos * theta^(-2i/dim) / factor\[i\].
fn rope(head: &mut [f32], position: u32, theta: f32, factors: Option<&[f32]>) {
    let dim = head.len();
    let half = dim / 2;
    for i in 0..half {
        let mut angle = f64::from(position) * f64::from(theta).powf(-2.0 * i as f64 / dim as f64);
        if let Some(factors) = factors {
            angle /= f64::from(factors[i]);
        }
        let (s, c) = angle.sin_cos();
        let a = f64::from(head[i]);
        let b = f64::from(head[i + half]);
        head[i] = (a * c - b * s) as f32;
        head[i + half] = (a * s + b * c) as f32;
    }
}

fn gelu_tanh(x: f64) -> f64 {
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

/// Q8_1 round trip: per 32 values d = amax / 127 (stored as binary16) and
/// q = round(x / d), as ggml's quantize_row_q8_1.
fn round_q8(x: &[f32], y: &mut [f32]) {
    for (xb, yb) in x.chunks(32).zip(y.chunks_mut(32)) {
        let amax = xb.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_half = round_half(d);
        for (o, &v) in yb.iter_mut().zip(xb) {
            *o = (v * id).round() * d_half;
        }
    }
}

/// Runs `f(begin, end)` over `[0, count)` on all hardware threads; the
/// results are returned in order.
fn parallel<T: Send>(count: usize, f: impl Fn(usize, usize) -> T + Sync) -> Vec<T> {
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(count)
        .max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                let f = &f;
                scope.spawn(move || f(count * w / workers, count * (w + 1) / workers))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    })
}

/// The scalar Gemma 4 graph over one model (`Reference`).
pub struct Reference<'s, 'a> {
    weights: &'s ModelWeights,
    shards: &'s [GgufShard<'a>],
    storage: Storage,
    position: u32,
    /// Per layer, `[position][kv_heads * head_dim]`.
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    trace: bool,
    layer_trace: Vec<f32>,
}

impl<'s, 'a> Reference<'s, 'a> {
    pub fn new(weights: &'s ModelWeights, shards: &'s [GgufShard<'a>], storage: Storage) -> Self {
        let mut reference = Self {
            weights,
            shards,
            storage,
            position: 0,
            keys: Vec::new(),
            values: Vec::new(),
            trace: false,
            layer_trace: Vec::new(),
        };
        reference.reset();
        reference
    }

    pub fn reset(&mut self) {
        let layers = self.weights.config.num_layers as usize;
        self.position = 0;
        self.keys = vec![Vec::new(); layers];
        self.values = vec![Vec::new(); layers];
        self.layer_trace.clear();
    }

    pub fn position(&self) -> u32 {
        self.position
    }

    /// When enabled, [`Self::forward`] records the residual stream after
    /// every layer: `[layer][row][hidden]` for the rows of the latest call.
    pub fn set_trace(&mut self, enabled: bool) {
        self.trace = enabled;
    }

    pub fn layer_trace(&self) -> &[f32] {
        &self.layer_trace
    }

    fn vector(&self, t: &TensorRef) -> Vec<f32> {
        t.f32_values(self.shards)
    }

    /// out\[r\]\[o\] = sum_c W\[o\]\[c\] * x\[r\]\[c\] for `rows` input rows,
    /// on all hardware threads.
    fn mat_mul(&self, weight: &TensorRef, x_in: &[f32], rows: usize) -> Result<Vec<f32>, String> {
        self.mat_mul_on(weight, x_in, rows, true)
    }

    /// [`Self::mat_mul`], on the calling thread unless `threaded`. Every
    /// output is the same sum either way.
    fn mat_mul_on(
        &self,
        weight: &TensorRef,
        x_in: &[f32],
        rows: usize,
        threaded: bool,
    ) -> Result<Vec<f32>, String> {
        let cols = weight.cols as usize;
        let rounded;
        let x = if self.storage == Storage::Q8Activations && weight.ty != GgmlType::F32 {
            let mut r = vec![0.0f32; rows * cols];
            for (src, dst) in x_in.chunks(cols).zip(r.chunks_mut(cols)) {
                round_q8(src, dst);
            }
            rounded = r;
            &rounded[..]
        } else {
            x_in
        };
        let outputs = weight.rows as usize;
        let row_bytes = weight.row_bytes();
        let data = weight.bytes(self.shards);
        let product = |begin: usize, end: usize| -> Option<Vec<f32>> {
            let mut w = vec![0.0f32; cols];
            let mut part = vec![0.0f32; (end - begin) * rows];
            for o in begin..end {
                dequantize_row(weight.ty, &data[o * row_bytes..(o + 1) * row_bytes], &mut w)?;
                for r in 0..rows {
                    let xr = &x[r * cols..(r + 1) * cols];
                    let mut acc = 0.0f64;
                    for c in 0..cols {
                        acc += f64::from(w[c]) * f64::from(xr[c]);
                    }
                    part[(o - begin) * rows + r] = acc as f32;
                }
            }
            Some(part)
        };
        let parts = if threaded {
            parallel(outputs, product)
        } else {
            vec![product(0, outputs)]
        };
        let mut out = vec![0.0f32; rows * outputs];
        let mut o = 0;
        for part in parts {
            let part = part.ok_or("gemma4 reference: unsupported projection weight type")?;
            let count = part.len() / rows.max(1);
            for i in 0..count {
                for r in 0..rows {
                    out[r * outputs + o + i] = part[i * rows + r];
                }
            }
            o += count;
        }
        Ok(out)
    }

    /// `ends` (sliding layers, may be empty): per row, the exclusive key end
    /// when it exceeds the row's own position + 1.
    fn attention(&self, layer: u32, q: &[f32], rows: usize, ends: &[u32]) -> Vec<f32> {
        let c = &self.weights.config;
        let dim = c.head_dim(layer) as usize;
        let heads = c.num_heads as usize;
        let kv_heads = c.kv_heads[layer as usize] as usize;
        let group = heads / kv_heads;
        let kv_stride = kv_heads * dim;
        let keys = &self.keys[layer as usize];
        let values = &self.values[layer as usize];
        let first = self.position;
        let sliding = c.is_sliding(layer);
        let jobs = rows * heads;
        let parts = parallel(jobs, |begin, end| {
            let mut part = vec![0.0f32; (end - begin) * dim];
            for job in begin..end {
                let (r, h) = (job / heads, job % heads);
                let pos = first + r as u32;
                let lo = if sliding && pos + 1 > c.sliding_window {
                    pos + 1 - c.sliding_window
                } else {
                    0
                };
                let hi = if ends.is_empty() {
                    pos + 1
                } else {
                    (pos + 1).max(ends[r])
                };
                let qh = &q[(r * heads + h) * dim..][..dim];
                let kvh = h / group;
                let mut scores = vec![0.0f64; (hi - lo) as usize];
                let mut max_score = f64::NEG_INFINITY;
                for p in lo..hi {
                    let k = &keys[p as usize * kv_stride + kvh * dim..][..dim];
                    let mut s = 0.0f64;
                    for d in 0..dim {
                        s += f64::from(qh[d]) * f64::from(k[d]);
                    }
                    scores[(p - lo) as usize] = s; // attention scale is 1
                    max_score = max_score.max(s);
                }
                let mut total = 0.0f64;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    total += *s;
                }
                let mut acc = vec![0.0f64; dim];
                for p in lo..hi {
                    let v = &values[p as usize * kv_stride + kvh * dim..][..dim];
                    let weight = scores[(p - lo) as usize] / total;
                    for d in 0..dim {
                        acc[d] += weight * f64::from(v[d]);
                    }
                }
                let o = &mut part[(job - begin) * dim..][..dim];
                for d in 0..dim {
                    o[d] = acc[d] as f32;
                }
            }
            part
        });
        parts.concat()
    }

    /// Routed expert mixture of `rows` attention residual rows (26B-A4B):
    /// router, top-k softmax weights, per-expert scale, GeGLU experts.
    fn experts(&self, w: &LayerWeights, x: &[f32], rows: usize) -> Result<Vec<f32>, String> {
        let c = &self.weights.config;
        let d = c.hidden_size as usize;
        let experts = c.num_experts as usize;
        let width = c.expert_ffn_size as usize;
        let eps = c.rms_eps;
        // Router input: rms(x) / sqrt(hidden) * scale, then F32 logits.
        let scale = self.vector(&w.router_scale);
        let inv_root = 1.0f32 / (d as f32).sqrt();
        let mut routed = vec![0.0f32; rows * d];
        for r in 0..rows {
            let row = &mut routed[r * d..(r + 1) * d];
            rms_norm(&x[r * d..(r + 1) * d], &[], eps, row);
            for i in 0..d {
                row[i] = row[i] * inv_root * scale[i];
            }
        }
        let logits = self.mat_mul(&w.router, &routed, rows)?;
        // Expert input.
        let pre_norm = self.vector(&w.pre_ffn_norm_2);
        let mut h = vec![0.0f32; rows * d];
        for r in 0..rows {
            rms_norm(
                &x[r * d..(r + 1) * d],
                &pre_norm,
                eps,
                &mut h[r * d..(r + 1) * d],
            );
        }
        let expert_scale = self.vector(&w.down_exps_scale);
        let mut out = vec![0.0f32; rows * d];
        for r in 0..rows {
            // Softmax over every expert, the top experts_used by probability
            // (ties to the lower index), renormalized over the chosen ones.
            let l = &logits[r * experts..(r + 1) * experts];
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap_or(std::cmp::Ordering::Equal));
            let top = f64::from(l[order[0]]);
            let used = c.experts_used as usize;
            let total: f64 = order[..used]
                .iter()
                .map(|&e| (f64::from(l[e]) - top).exp())
                .sum();
            // The chosen experts run on their own threads; their outputs add
            // up in rank order, as in C++.
            let hr = &h[r * d..(r + 1) * d];
            let downs = std::thread::scope(|scope| {
                let handles: Vec<_> = order[..used]
                    .iter()
                    .map(|&e| {
                        scope.spawn(move || -> Result<Vec<f32>, String> {
                            let gate_up = self.mat_mul_on(
                                &w.gate_up_exps.expert_view(e as u64),
                                hr,
                                1,
                                false,
                            )?;
                            let act: Vec<f32> = (0..width)
                                .map(|i| {
                                    (gelu_tanh(f64::from(gate_up[i]))
                                        * f64::from(gate_up[width + i]))
                                        as f32
                                })
                                .collect();
                            self.mat_mul_on(&w.down_exps.expert_view(e as u64), &act, 1, false)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                    .collect::<Vec<_>>()
            });
            for (&e, down) in order[..used].iter().zip(downs) {
                let down = down?;
                let weight = (f64::from(l[e]) - top).exp() / total * f64::from(expert_scale[e]);
                for i in 0..d {
                    out[r * d + i] =
                        (f64::from(out[r * d + i]) + weight * f64::from(down[i])) as f32;
                }
            }
        }
        Ok(out)
    }

    /// Appends `tokens` after the current position. Returns the softcapped
    /// `[tokens][vocab]` logits when `want_logits`, and the post-norm
    /// `[tokens][hidden]` rows fed to the LM head and to MTP. Processing a
    /// prefix in several calls gives identical results.
    pub fn forward(
        &mut self,
        tokens: &[TokenId],
        images: &[ReferenceImage<'_>],
        want_logits: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let weights = self.weights;
        let c = &weights.config;
        let n = tokens.len();
        let d = c.hidden_size as usize;
        let eps = c.rms_eps;
        let mut x = vec![0.0f32; n * d];
        let mut h = vec![0.0f32; n * d];

        // Scaled token embeddings (the scale is applied in float, as
        // llama.cpp).
        let embed_scale = (d as f32).sqrt();
        let embed_row = weights.token_embd.row_bytes();
        let embed = weights.token_embd.bytes(self.shards);
        for (r, &token) in tokens.iter().enumerate() {
            let token = usize::try_from(token)
                .ok()
                .filter(|&t| t < weights.vocab_size as usize)
                .ok_or("gemma4 reference: token outside the vocabulary")?;
            let row = &mut x[r * d..(r + 1) * d];
            dequantize_row(
                weights.token_embd.ty,
                &embed[token * embed_row..(token + 1) * embed_row],
                row,
            )
            .ok_or("gemma4 reference: unsupported token embedding type")?;
            for v in row {
                *v *= embed_scale;
            }
        }
        // Image rows take the encoder output unscaled and, in sliding layers,
        // see their whole image.
        let mut ends = Vec::new();
        for image in images {
            let (row, count) = (image.row as usize, image.count as usize);
            if count == 0 || row + count > n || image.embedding.len() < count * d {
                return Err("gemma4 reference: invalid image rows".into());
            }
            x[row * d..(row + count) * d].copy_from_slice(&image.embedding[..count * d]);
            ends.resize(n, 0);
            ends[row..row + count].fill(self.position + image.row + image.count);
        }
        if self.trace {
            self.layer_trace = vec![0.0; c.num_layers as usize * n * d];
        }

        for l in 0..c.num_layers {
            let w = &weights.layers[l as usize];
            let dim = c.head_dim(l) as usize;
            let heads = c.num_heads as usize;
            let kv_heads = c.kv_heads[l as usize] as usize;
            let factors = (!c.is_sliding(l)).then_some(weights.rope_factors.values.as_slice());
            let theta = c.rope_theta(l);

            let attn_norm = self.vector(&w.attn_norm);
            for r in 0..n {
                rms_norm(
                    &x[r * d..(r + 1) * d],
                    &attn_norm,
                    eps,
                    &mut h[r * d..(r + 1) * d],
                );
            }
            let mut q = self.mat_mul(&w.attn_q, &h, n)?;
            let mut k = self.mat_mul(&w.attn_k, &h, n)?;
            let mut v = if w.attn_v.is_empty() {
                k.clone() // global layers: V is the raw K projection
            } else {
                self.mat_mul(&w.attn_v, &h, n)?
            };
            let q_norm = self.vector(&w.attn_q_norm);
            let k_norm = self.vector(&w.attn_k_norm);
            for r in 0..n {
                let pos = self.position + r as u32;
                for hh in 0..heads {
                    let head = &mut q[(r * heads + hh) * dim..][..dim];
                    rms_norm_in_place(head, &q_norm, eps);
                    rope(head, pos, theta, factors);
                }
                for hh in 0..kv_heads {
                    let kh = &mut k[(r * kv_heads + hh) * dim..][..dim];
                    rms_norm_in_place(kh, &k_norm, eps);
                    rope(kh, pos, theta, factors);
                    rms_norm_in_place(&mut v[(r * kv_heads + hh) * dim..][..dim], &[], eps);
                }
            }
            if self.storage != Storage::Float32 {
                for value in k.iter_mut().chain(v.iter_mut()) {
                    *value = round_half(*value);
                }
            }
            self.keys[l as usize].extend_from_slice(&k);
            self.values[l as usize].extend_from_slice(&v);

            let attn = self.attention(l, &q, n, if c.is_sliding(l) { &ends } else { &[] });
            let mut o = self.mat_mul(&w.attn_output, &attn, n)?;
            let post_attn = self.vector(&w.post_attn_norm);
            let ffn_norm = self.vector(&w.ffn_norm);
            for r in 0..n {
                let row = &mut o[r * d..(r + 1) * d];
                rms_norm_in_place(row, &post_attn, eps);
                for i in 0..d {
                    x[r * d + i] += row[i];
                }
                rms_norm(
                    &x[r * d..(r + 1) * d],
                    &ffn_norm,
                    eps,
                    &mut h[r * d..(r + 1) * d],
                );
            }

            let mut gate = self.mat_mul(&w.ffn_gate, &h, n)?;
            let up = self.mat_mul(&w.ffn_up, &h, n)?;
            for (g, &u) in gate.iter_mut().zip(&up) {
                *g = (gelu_tanh(f64::from(*g)) * f64::from(u)) as f32;
            }
            let mut f = self.mat_mul(&w.ffn_down, &gate, n)?;
            if c.has_experts() {
                // The dense MLP and the routed experts both read the
                // attention residual x; their separately normed outputs add
                // up to f.
                let mut moe = self.experts(w, &x, n)?;
                let norm_1 = self.vector(&w.post_ffn_norm_1);
                let norm_2 = self.vector(&w.post_ffn_norm_2);
                for r in 0..n {
                    let fr = &mut f[r * d..(r + 1) * d];
                    rms_norm_in_place(fr, &norm_1, eps);
                    let mr = &mut moe[r * d..(r + 1) * d];
                    rms_norm_in_place(mr, &norm_2, eps);
                    for i in 0..d {
                        fr[i] += mr[i];
                    }
                }
            }
            let post_ffn = self.vector(&w.post_ffn_norm);
            for r in 0..n {
                let fr = &mut f[r * d..(r + 1) * d];
                rms_norm_in_place(fr, &post_ffn, eps);
                for i in 0..d {
                    x[r * d + i] = (x[r * d + i] + fr[i]) * w.output_scale;
                }
            }
            if self.trace {
                self.layer_trace[l as usize * n * d..(l as usize + 1) * n * d].copy_from_slice(&x);
            }
        }

        let output_norm = self.vector(&weights.output_norm);
        for r in 0..n {
            rms_norm(
                &x[r * d..(r + 1) * d],
                &output_norm,
                eps,
                &mut h[r * d..(r + 1) * d],
            );
        }
        let mut logits = Vec::new();
        if want_logits {
            logits = self.mat_mul(&weights.output, &h, n)?;
            let cap = f64::from(c.final_logit_softcap);
            if cap > 0.0 {
                for value in &mut logits {
                    *value = (cap * (f64::from(*value) / cap).tanh()) as f32;
                }
            }
        }
        self.position += n as u32;
        Ok((logits, h))
    }
}
