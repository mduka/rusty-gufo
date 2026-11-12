//! Scalar reference of the Gemma 4 vision tower and embedder
//! (`vision/reference.cpp`; transformers `Gemma4VisionModel` +
//! `Gemma4MultimodalEmbedder`; llama.cpp `clip_graph_gemma4v` with
//! `clip.use_gelu`). Weights are widened to F32 and every reduction
//! accumulates in `f64`; attention is quadratic, so it only suits small
//! images. Output rows are row-major over the pooled grid.

use gufo_core::image::Image;

use crate::vision_prompt::{PATCH_SIZE, POOL_SIZE, SIDE_MULTIPLE};
use crate::vision_weights::{
    EPS, HEAD_DIM, HEADS, HIDDEN, LAYERS, PATCH_VALUES, POSITIONS, ROPE_THETA, VisionTensor,
    VisionWeights,
};

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

/// out\[r\]\[o\] = sum_i W\[o\]\[i\] * x\[r\]\[i\] over a widened `[out][in]`
/// matrix.
fn mat_mul_widened(w: &[f32], input: usize, x: &[f32], rows: usize) -> Vec<f32> {
    let out_width = w.len() / input;
    let parts = parallel(out_width, |begin, end| {
        let mut part = vec![0.0f32; (end - begin) * rows];
        for o in begin..end {
            let wo = &w[o * input..(o + 1) * input];
            for r in 0..rows {
                let xr = &x[r * input..(r + 1) * input];
                let mut sum = 0.0f64;
                for i in 0..input {
                    sum += f64::from(wo[i]) * f64::from(xr[i]);
                }
                part[(o - begin) * rows + r] = sum as f32;
            }
        }
        part
    });
    let mut out = vec![0.0f32; rows * out_width];
    let mut o = 0;
    for part in parts {
        let count = part.len() / rows.max(1);
        for i in 0..count {
            for r in 0..rows {
                out[r * out_width + o + i] = part[i * rows + r];
            }
        }
        o += count;
    }
    out
}

fn mat_mul(tensor: &VisionTensor, file: &[u8], x: &[f32], rows: usize) -> Vec<f32> {
    mat_mul_widened(&tensor.widen(file), tensor.dims[0] as usize, x, rows)
}

/// RMS-normalizes each `width` group; `weight` may be empty.
fn rms_norm(x: &mut [f32], width: usize, weight: &[f32]) {
    for row in x.chunks_mut(width) {
        let sum: f64 = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let scale = 1.0 / (sum / width as f64 + f64::from(EPS)).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            let w = if weight.is_empty() {
                1.0
            } else {
                f64::from(weight[i])
            };
            *v = (f64::from(*v) * scale * w) as f32;
        }
    }
}

/// NEOX rotation of each half of every head: dims \[0,36) by column,
/// \[36,72) by row, each as pairs (i, i+18).
fn rope(x: &mut [f32], count: usize, columns: u32) {
    let half = (HEAD_DIM / 2) as usize;
    let pairs = half / 2;
    for t in 0..count {
        let position = [f64::from(t as u32 % columns), f64::from(t as u32 / columns)];
        for h in 0..HEADS as usize {
            for (part, &pos) in position.iter().enumerate() {
                let base = t * HIDDEN as usize + h * HEAD_DIM as usize + part * half;
                for i in 0..pairs {
                    let angle = pos * f64::from(ROPE_THETA).powf(-2.0 * i as f64 / half as f64);
                    let (s, c) = (angle.sin(), angle.cos());
                    let a = f64::from(x[base + i]);
                    let b = f64::from(x[base + i + pairs]);
                    x[base + i] = (a * c - b * s) as f32;
                    x[base + i + pairs] = (b * c + a * s) as f32;
                }
            }
        }
    }
}

/// Sees each named stage's rows (`vision::Reference::Observer`).
pub type Observer<'o> = &'o mut dyn FnMut(&str, &[f32]);

/// The scalar vision tower over one sidecar (`vision::Reference`).
pub struct VisionReference<'w, 'f> {
    weights: &'w VisionWeights,
    file: &'f [u8],
}

impl<'w, 'f> VisionReference<'w, 'f> {
    /// `file` is the mapped sidecar the weights were resolved from.
    pub fn new(weights: &'w VisionWeights, file: &'f [u8]) -> Self {
        Self { weights, file }
    }

    /// `image` must already be resized (sides multiples of 48). The
    /// observer sees the stages "patch" (patch embedding plus positions),
    /// "layer<i>" (residual after block i), "pooled" (3x3 mean times
    /// sqrt(1152)) and "embedding".
    pub fn encode(
        &self,
        image: &Image,
        mut observer: Option<Observer<'_>>,
    ) -> Result<Vec<f32>, String> {
        if image.width == 0
            || image.height == 0
            || !image.width.is_multiple_of(SIDE_MULTIPLE)
            || !image.height.is_multiple_of(SIDE_MULTIPLE)
            || image.pixels.len() != image.width as usize * image.height as usize * 3
        {
            return Err("vision reference needs a resized RGB image".into());
        }
        let columns = image.width / PATCH_SIZE;
        let patch_rows = image.height / PATCH_SIZE;
        let count = columns as usize * patch_rows as usize;
        if columns > POSITIONS || patch_rows > POSITIONS {
            return Err("image exceeds the position tables".into());
        }
        let mut observe = |stage: &str, data: &[f32]| {
            if let Some(observer) = observer.as_deref_mut() {
                observer(stage, data);
            }
        };
        let (hidden, patch, values) = (HIDDEN as usize, PATCH_SIZE as usize, PATCH_VALUES as usize);
        let file = self.file;
        let w = self.weights;

        // Patches in 2*(v/255)-1, ordered channel, row, column like the
        // kernel.
        let mut patches = vec![0.0f32; count * values];
        for t in 0..count {
            let px = (t % columns as usize) * patch;
            let py = (t / columns as usize) * patch;
            for c in 0..3 {
                for y in 0..patch {
                    for x in 0..patch {
                        let value =
                            image.pixels[((py + y) * image.width as usize + px + x) * 3 + c];
                        patches[t * values + (c * patch + y) * patch + x] =
                            2.0 * (f32::from(value) / 255.0) - 1.0;
                    }
                }
            }
        }
        let mut h = mat_mul_widened(&w.patch.widen(file), values, &patches, count);
        let position = w.position.widen(file);
        let positions = POSITIONS as usize;
        for t in 0..count {
            let (x, y) = (t % columns as usize, t / columns as usize);
            for d in 0..hidden {
                h[t * hidden + d] +=
                    position[x * hidden + d] + position[(positions + y) * hidden + d];
            }
        }
        observe("patch", &h);

        for (l, layer) in w.layers.iter().enumerate().take(LAYERS as usize) {
            let mut a = h.clone();
            rms_norm(&mut a, hidden, &layer.ln1.widen(file));
            let mut q = mat_mul(&layer.q, file, &a, count);
            let mut k = mat_mul(&layer.k, file, &a, count);
            let mut v = mat_mul(&layer.v, file, &a, count);
            let head_dim = HEAD_DIM as usize;
            rms_norm(&mut q, head_dim, &layer.q_norm.widen(file));
            rms_norm(&mut k, head_dim, &layer.k_norm.widen(file));
            rms_norm(&mut v, head_dim, &[]);
            rope(&mut q, count, columns);
            rope(&mut k, count, columns);
            let heads = HEADS as usize;
            let parts = parallel(count * heads, |begin, end| {
                let mut part = vec![0.0f32; (end - begin) * head_dim];
                let mut scores = vec![0.0f64; count];
                for job in begin..end {
                    let (t, head) = (job / heads, job % heads);
                    let qt = &q[t * hidden + head * head_dim..][..head_dim];
                    let mut maximum = f64::NEG_INFINITY;
                    for (s, score) in scores.iter_mut().enumerate() {
                        let ks = &k[s * hidden + head * head_dim..][..head_dim];
                        let mut dot = 0.0f64;
                        for d in 0..head_dim {
                            dot += f64::from(qt[d]) * f64::from(ks[d]);
                        }
                        *score = dot; // scale 1.0
                        maximum = maximum.max(dot);
                    }
                    let mut sum = 0.0f64;
                    for score in &mut scores {
                        *score = (*score - maximum).exp();
                        sum += *score;
                    }
                    for d in 0..head_dim {
                        let mut value = 0.0f64;
                        for (s, &score) in scores.iter().enumerate() {
                            value += score * f64::from(v[s * hidden + head * head_dim + d]);
                        }
                        part[(job - begin) * head_dim + d] = (value / sum) as f32;
                    }
                }
                part
            });
            // Jobs run over (token, head), so the concatenated parts are the
            // `[token][hidden]` rows.
            let attention = parts.concat();
            let mut o = mat_mul(&layer.out, file, &attention, count);
            rms_norm(&mut o, hidden, &layer.attn_post_norm.widen(file));
            for (hv, ov) in h.iter_mut().zip(&o) {
                *hv += ov;
            }
            let mut b = h.clone();
            rms_norm(&mut b, hidden, &layer.ln2.widen(file));
            let mut gate = mat_mul(&layer.gate, file, &b, count);
            let up = mat_mul(&layer.up, file, &b, count);
            for (g, &u) in gate.iter_mut().zip(&up) {
                let x = f64::from(*g);
                let gelu =
                    0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044715 * x * x * x)).tanh());
                *g = (gelu * f64::from(u)) as f32;
            }
            let mut down = mat_mul(&layer.down, file, &gate, count);
            rms_norm(&mut down, hidden, &layer.ffn_post_norm.widen(file));
            for (hv, dv) in h.iter_mut().zip(&down) {
                *hv += dv;
            }
            observe(&format!("layer{l}"), &h);
        }

        let pool = POOL_SIZE as usize;
        let pooled_columns = columns as usize / pool;
        let pooled_rows = patch_rows as usize / pool;
        let rows = pooled_columns * pooled_rows;
        let mut pooled = vec![0.0f32; rows * hidden];
        for r in 0..rows {
            let px = (r % pooled_columns) * pool;
            let py = (r / pooled_columns) * pool;
            for d in 0..hidden {
                let mut sum = 0.0f64;
                for y in 0..pool {
                    for x in 0..pool {
                        sum += f64::from(h[((py + y) * columns as usize + px + x) * hidden + d]);
                    }
                }
                pooled[r * hidden + d] =
                    (sum / (pool * pool) as f64 * (hidden as f64).sqrt()) as f32;
            }
        }
        observe("pooled", &pooled);
        let bias = w.std_bias.widen(file);
        let scale = w.std_scale.widen(file);
        for (i, value) in pooled.iter_mut().enumerate() {
            *value = (*value - bias[i % hidden]) * scale[i % hidden];
        }
        rms_norm(&mut pooled, hidden, &[]);
        let embedding = mat_mul(&w.projection, file, &pooled, rows);
        observe("embedding", &embedding);
        Ok(embedding)
    }
}
