//! Single-token, float32, scalar reference of the Nemotron 3.5 Bolt trunk
//! (`reference.cpp`). It pins the semantics of every operator the GPU
//! runtime must reproduce: Mamba-2 (causal conv, selective scan, gated
//! grouped RMSNorm), NoPE GQA, and LatentMoE with sigmoid routing, a
//! selection-only bias and ReLU^2 experts. The operator order follows
//! llama.cpp's nemotron_h graph. Speed is irrelevant here.

// The loops index several arrays by head, channel and state as the C++
// does, which keeps the port checkable against it.
#![allow(clippy::needless_range_loop)]

use crate::config::{BlockType, Config};
use crate::cpu_ops::{dequantize_row, mat_vec, relu2, rms_norm, sigmoid, silu, softplus};
use crate::weights::{GgufShard, LayerWeights, ModelWeights};

/// llama.cpp clamps the routing-weight sum from below before normalizing.
const MIN_ROUTING_WEIGHT_SUM: f32 = 1.0 / 16384.0;

#[derive(Default)]
struct MambaState {
    /// `[kernel-1][channels]` raw x|B|C, oldest first
    conv: Vec<f32>,
    /// `[heads][head_dim][state_size]`
    ssm: Vec<f32>,
}

#[derive(Default)]
struct AttentionState {
    /// `[pos][kv_heads][head_dim]`
    k: Vec<f32>,
    /// `[pos][kv_heads][head_dim]`
    v: Vec<f32>,
}

/// The F32 vectors of one block, read once.
#[derive(Default)]
struct LayerVectors {
    norm: Vec<f32>,
    conv1d: Vec<f32>,
    conv1d_b: Vec<f32>,
    dt_b: Vec<f32>,
    a: Vec<f32>,
    d: Vec<f32>,
    ssm_norm: Vec<f32>,
    router_bias: Vec<f32>,
}

pub struct ReferenceModel<'w, 'a> {
    w: &'w ModelWeights,
    shards: &'w [GgufShard<'a>],
    max_context: u32,
    position: u32,
    mamba: Vec<MambaState>,
    attention: Vec<AttentionState>,
    vectors: Vec<LayerVectors>,
    output_norm: Vec<f32>,
}

impl<'w, 'a> ReferenceModel<'w, 'a> {
    pub fn new(weights: &'w ModelWeights, shards: &'w [GgufShard<'a>], max_context: u32) -> Self {
        let c = &weights.config;
        let f32s = |t: &crate::weights::TensorRef| {
            if t.is_empty() {
                Vec::new()
            } else {
                t.f32_values(shards)
            }
        };
        let vectors = weights
            .layers
            .iter()
            .map(|l| LayerVectors {
                norm: f32s(&l.norm),
                conv1d: f32s(&l.ssm_conv1d),
                conv1d_b: f32s(&l.ssm_conv1d_b),
                dt_b: f32s(&l.ssm_dt_b),
                a: f32s(&l.ssm_a),
                d: f32s(&l.ssm_d),
                ssm_norm: f32s(&l.ssm_norm),
                router_bias: f32s(&l.router_bias),
            })
            .collect();
        let mut model = Self {
            w: weights,
            shards,
            max_context,
            position: 0,
            mamba: (0..c.num_layers).map(|_| MambaState::default()).collect(),
            attention: (0..c.num_layers)
                .map(|_| AttentionState::default())
                .collect(),
            vectors,
            output_norm: f32s(&weights.output_norm),
        };
        model.reset();
        model
    }

    fn config(&self) -> &'w Config {
        &self.w.config
    }

    pub fn position(&self) -> u32 {
        self.position
    }

    pub fn reset(&mut self) {
        self.position = 0;
        let c = self.config();
        for il in 0..c.num_layers as usize {
            match c.blocks[il] {
                BlockType::Mamba => {
                    let s = &mut self.mamba[il];
                    s.conv = vec![
                        0.0;
                        (c.ssm_conv_kernel as usize - 1) * c.ssm_conv_channels() as usize
                    ];
                    s.ssm = vec![
                        0.0;
                        c.ssm_heads as usize
                            * c.ssm_head_dim() as usize
                            * c.ssm_state_size as usize
                    ];
                }
                BlockType::Attention => {
                    self.attention[il].k.clear();
                    self.attention[il].v.clear();
                }
                BlockType::Moe => {}
            }
        }
    }

    /// Runs one token at the next position; `logits` (vocab floats)
    /// receives the output distribution when non-empty (`Step`).
    pub fn step(&mut self, token: i32, logits: &mut [f32]) -> Result<(), String> {
        let c = self.config();
        if token < 0 || token as u32 >= c.vocab_size {
            return Err("token outside the vocabulary".into());
        }
        if self.position >= self.max_context {
            return Err("reference context is full".into());
        }
        if !logits.is_empty() && logits.len() != c.vocab_size as usize {
            return Err("logits span must hold the vocabulary".into());
        }
        let hidden = c.hidden_size as usize;
        let mut x = vec![0.0f32; hidden];
        dequantize_row(&self.w.token_embd, self.shards, 0, token as u64, &mut x);
        let mut h = vec![0.0f32; hidden];
        let mut out = vec![0.0f32; hidden];
        let w = self.w;
        for il in 0..c.num_layers as usize {
            let l = &w.layers[il];
            h.copy_from_slice(&x);
            rms_norm(&mut h, &self.vectors[il].norm, c.rms_eps);
            match l.ty {
                BlockType::Mamba => self.mamba(il, l, &h, &mut out),
                BlockType::Attention => self.attention(il, l, &h, &mut out),
                BlockType::Moe => self.moe(il, l, &h, &mut out),
            }
            for (xi, &oi) in x.iter_mut().zip(&out) {
                *xi += oi;
            }
        }
        self.position += 1;
        if !logits.is_empty() {
            rms_norm(&mut x, &self.output_norm, c.rms_eps);
            mat_vec(&self.w.output, self.shards, 0, &x, logits);
        }
        Ok(())
    }

    fn mamba(&mut self, il: usize, l: &LayerWeights, x: &[f32], out: &mut [f32]) {
        let c = self.config();
        let inner = c.ssm_inner_size as usize;
        let channels = c.ssm_conv_channels() as usize;
        let heads = c.ssm_heads as usize;
        let head_dim = c.ssm_head_dim() as usize;
        let state = c.ssm_state_size as usize;
        let groups = c.ssm_groups as usize;
        let kernel = c.ssm_conv_kernel as usize;
        let v = &self.vectors[il];
        let s = &mut self.mamba[il];

        let mut zxbcdt = vec![0.0f32; c.ssm_in_projection() as usize];
        mat_vec(&l.ssm_in, self.shards, 0, x, &mut zxbcdt);
        let z = &zxbcdt[..inner];
        let xbc = &zxbcdt[inner..inner + channels];
        let dt = &zxbcdt[inner + channels..];

        // Causal depthwise convolution over the kernel-1 previous raw inputs
        // and this one; tap 0 multiplies the oldest input. The state keeps
        // raw inputs.
        let mut conv = vec![0.0f32; channels];
        let w = &v.conv1d;
        for ch in 0..channels {
            let mut acc = 0.0f32;
            for k in 0..kernel - 1 {
                acc += s.conv[k * channels + ch] * w[ch * kernel + k];
            }
            acc += xbc[ch] * w[ch * kernel + kernel - 1];
            conv[ch] = silu(acc + v.conv1d_b[ch]);
        }
        for k in 0..kernel.saturating_sub(2) {
            s.conv
                .copy_within((k + 1) * channels..(k + 2) * channels, k * channels);
        }
        s.conv[(kernel - 2) * channels..(kernel - 1) * channels].copy_from_slice(xbc);

        // Selective scan with a scalar decay per head. Heads map onto B/C
        // groups as repeat_interleave: group = head / (heads / groups).
        let xs = &conv[..inner];
        let b = &conv[inner..inner + groups * state];
        let cm = &conv[inner + groups * state..];
        let mut y = vec![0.0f32; inner];
        for hh in 0..heads {
            let dt_sp = softplus(dt[hh] + v.dt_b[hh]);
            let da = (dt_sp * v.a[hh]).exp();
            let g = hh / (heads / groups);
            let bg = &b[g * state..(g + 1) * state];
            let cg = &cm[g * state..(g + 1) * state];
            for i in 0..head_dim {
                let ii = hh * head_dim + i;
                let x_dt = xs[ii] * dt_sp;
                let st = &mut s.ssm[ii * state..(ii + 1) * state];
                let mut sum = 0.0f64;
                for n in 0..state {
                    st[n] = st[n] * da + bg[n] * x_dt;
                    sum += f64::from(st[n]) * f64::from(cg[n]);
                }
                y[ii] = sum as f32 + xs[ii] * v.d[hh];
            }
        }

        // Gate with silu(z), then RMS-normalize each group of inner/groups.
        for (yi, &zi) in y.iter_mut().zip(z) {
            *yi *= silu(zi);
        }
        let group_size = inner / groups;
        for g in 0..groups {
            rms_norm(
                &mut y[g * group_size..(g + 1) * group_size],
                &v.ssm_norm[g * group_size..(g + 1) * group_size],
                c.rms_eps,
            );
        }
        mat_vec(&l.ssm_out, self.shards, 0, &y, out);
    }

    fn attention(&mut self, il: usize, l: &LayerWeights, x: &[f32], out: &mut [f32]) {
        let c = self.config();
        let heads = c.num_heads as usize;
        let kv_heads = c.num_kv_heads as usize;
        let hd = c.head_dim as usize;
        let kv_dim = kv_heads * hd;

        let mut q = vec![0.0f32; heads * hd];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        mat_vec(&l.attn_q, self.shards, 0, x, &mut q);
        mat_vec(&l.attn_k, self.shards, 0, x, &mut k);
        mat_vec(&l.attn_v, self.shards, 0, x, &mut v);
        let s = &mut self.attention[il];
        s.k.extend_from_slice(&k);
        s.v.extend_from_slice(&v);

        // No positional encoding: position enters only through causality and
        // the Mamba state.
        let positions = s.k.len() / kv_dim;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let mut ctx = vec![0.0f32; heads * hd];
        let mut scores = vec![0.0f64; positions];
        for hh in 0..heads {
            let kvh = hh / (heads / kv_heads);
            let qh = &q[hh * hd..(hh + 1) * hd];
            let mut max_score = f64::NEG_INFINITY;
            for (t, score) in scores.iter_mut().enumerate() {
                let kt = &s.k[t * kv_dim + kvh * hd..t * kv_dim + kvh * hd + hd];
                let mut dot = 0.0f64;
                for i in 0..hd {
                    dot += f64::from(qh[i]) * f64::from(kt[i]);
                }
                *score = dot * f64::from(scale);
                max_score = max_score.max(*score);
            }
            let mut denom = 0.0f64;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                denom += *score;
            }
            let ch = &mut ctx[hh * hd..(hh + 1) * hd];
            for (i, value) in ch.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for (t, &score) in scores.iter().enumerate() {
                    acc += score * f64::from(s.v[t * kv_dim + kvh * hd + i]);
                }
                *value = (acc / denom) as f32;
            }
        }
        mat_vec(&l.attn_out, self.shards, 0, &ctx, out);
    }

    fn moe(&mut self, il: usize, l: &LayerWeights, x: &[f32], out: &mut [f32]) {
        let c = self.config();
        let n = l.num_experts as usize;
        let used = c.experts_used[il] as usize;
        let ff = c.expert_ff[il] as usize;
        let latent = c.latent_size as usize;

        // Sigmoid scores; the bias only steers selection, weights use raw
        // scores.
        let mut probs = vec![0.0f32; n];
        mat_vec(&l.router, self.shards, 0, x, &mut probs);
        for p in &mut probs {
            *p = sigmoid(*p);
        }
        let bias = &self.vectors[il].router_bias;
        let mut order: Vec<u32> = (0..n as u32).collect();
        // The C++ partial_sort's comparator orders every pair (ties by
        // index), so a full sort gives the same first `used`.
        order.sort_by(|&a, &b| {
            let sa = probs[a as usize] + bias[a as usize];
            let sb = probs[b as usize] + bias[b as usize];
            if sa > sb || (sa == sb && a < b) {
                std::cmp::Ordering::Less
            } else if sb > sa || (sa == sb && b < a) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        let mut weights: Vec<f32> = order[..used].iter().map(|&e| probs[e as usize]).collect();
        let mut sum = 0.0f32;
        for &wgt in &weights {
            sum += wgt;
        }
        for wgt in &mut weights {
            if c.expert_weights_norm {
                *wgt /= sum.max(MIN_ROUTING_WEIGHT_SUM);
            }
            *wgt *= c.expert_weights_scale;
        }

        // Routed experts run in the latent width; their weighted sum is
        // projected back once.
        let mut lat = vec![0.0f32; latent];
        mat_vec(&l.latent_down, self.shards, 0, x, &mut lat);
        let mut acc = vec![0.0f32; latent];
        let mut up = vec![0.0f32; ff];
        let mut down = vec![0.0f32; latent];
        for (&expert, &wgt) in order[..used].iter().zip(&weights) {
            mat_vec(&l.up_exps, self.shards, u64::from(expert), &lat, &mut up);
            for u in &mut up {
                *u = relu2(*u);
            }
            mat_vec(&l.down_exps, self.shards, u64::from(expert), &up, &mut down);
            for (a, &d) in acc.iter_mut().zip(&down) {
                *a += wgt * d;
            }
        }
        mat_vec(&l.latent_up, self.shards, 0, &acc, out);

        let mut shared_up = vec![0.0f32; c.shared_expert_ff as usize];
        let mut shared = vec![0.0f32; c.hidden_size as usize];
        mat_vec(&l.up_shexp, self.shards, 0, x, &mut shared_up);
        for u in &mut shared_up {
            *u = relu2(*u);
        }
        mat_vec(&l.down_shexp, self.shards, 0, &shared_up, &mut shared);
        for (o, &s) in out.iter_mut().zip(&shared) {
            *o += s;
        }
    }
}
