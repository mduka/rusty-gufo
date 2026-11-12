//! f32 CPU implementation of the text embedding path
//! (`reference/cpu_encoder.cpp`). It is the independent oracle for the GPU
//! encoder: slow, single-threaded, and written to read like the model
//! definition.

use gufo_embedding_gemma2::config::{EMBED_SCALE, LayerType, TextConfig};
use gufo_embedding_gemma2::loader::{TEXT_TENSOR_PREFIX, TensorStore};

fn bf16_to_f32(bytes: &[u8]) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16)
}

fn weight(store: &TensorStore, name: &str) -> Vec<f32> {
    let tensor = store
        .find(name)
        .unwrap_or_else(|| panic!("embedding_gemma2 reference: missing {name}"));
    store
        .bytes(tensor)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16_to_f32(pair))
        .collect()
}

fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// `y[t, o] = sum_i x[t, i] * w[o, i]`
fn linear(x: &[f32], tokens: usize, input: usize, w: &[f32], output: usize) -> Vec<f32> {
    let mut y = vec![0f32; tokens * output];
    for t in 0..tokens {
        let row = &x[t * input..(t + 1) * input];
        for o in 0..output {
            let weights = &w[o * input..(o + 1) * input];
            let mut sum = 0f32;
            for i in 0..input {
                sum += row[i] * weights[i];
            }
            y[t * output + o] = sum;
        }
    }
    y
}

/// Normalises each `width`-wide row of `x` in place. No `weight` is the
/// unscaled norm the value projection uses.
fn rms_norm(x: &mut [f32], width: usize, weight: Option<&[f32]>, eps: f32) {
    for row in x.chunks_exact_mut(width) {
        let mut squares = 0f32;
        for &value in row.iter() {
            squares += value * value;
        }
        let scale = (squares / width as f32 + eps).powf(-0.5);
        for (i, value) in row.iter_mut().enumerate() {
            *value = *value * scale * weight.map_or(1.0, |w| w[i]);
        }
    }
}

/// Rotates the two halves of each head (rotate_half convention).
fn rope(x: &mut [f32], tokens: usize, heads: usize, head_dim: usize, theta: f32) {
    let half = head_dim / 2;
    let inverse: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
        .collect();
    for t in 0..tokens {
        for h in 0..heads {
            let head = &mut x[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim];
            for i in 0..half {
                let angle = t as f32 * inverse[i];
                let (c, s) = (angle.cos(), angle.sin());
                let (a, b) = (head[i], head[i + half]);
                head[i] = a * c - b * s;
                head[i + half] = b * c + a * s;
            }
        }
    }
}

pub struct CpuEncoder<'a> {
    config: &'a TextConfig,
    store: &'a TensorStore,
}

impl<'a> CpuEncoder<'a> {
    pub fn new(config: &'a TextConfig, store: &'a TensorStore) -> Self {
        Self { config, store }
    }

    /// Runs the backbone over one sequence and returns the projected token
    /// states, row-major `[tokens, embedding_dim]`.
    pub fn encode_tokens(&self, token_ids: &[u32]) -> Vec<f32> {
        let c = self.config;
        let store = self.store;
        let tokens = token_ids.len();
        let hidden = c.hidden_size as usize;
        let per_layer = c.hidden_size_per_layer_input as usize;
        let layers = c.num_hidden_layers as usize;
        let prefix = TEXT_TENSOR_PREFIX;
        let eps = c.rms_norm_eps;
        if tokens == 0 {
            return Vec::new();
        }
        let table = store
            .find(&format!("{prefix}embed_tokens.weight"))
            .expect("embedding_gemma2 reference: missing embeddings");
        let table = store.bytes(table);
        let mut state = vec![0f32; tokens * hidden];
        for (t, &id) in token_ids.iter().enumerate() {
            assert!(id < c.vocab_size, "embedding_gemma2 reference: token id");
            for i in 0..hidden {
                let at = (id as usize * hidden + i) * 2;
                state[t * hidden + i] = bf16_to_f32(&table[at..at + 2]) * EMBED_SCALE;
            }
        }

        // Per-layer inputs derive from the scaled embeddings alone.
        let mut ple = linear(
            &state,
            tokens,
            hidden,
            &weight(
                store,
                &format!("{prefix}ple.per_layer_model_projection.weight"),
            ),
            layers * per_layer,
        );
        let ple_scale = 1.0f32 / (hidden as f32).sqrt();
        for value in &mut ple {
            *value *= ple_scale;
        }
        rms_norm(
            &mut ple,
            per_layer,
            Some(&weight(
                store,
                &format!("{prefix}ple.per_layer_projection_norm.weight"),
            )),
            eps,
        );

        for (l, layer) in c.layers.iter().enumerate() {
            let p = format!("{prefix}layers.{l}.");
            let w = |name: &str| weight(store, &format!("{p}{name}"));
            let heads = c.num_attention_heads as usize;
            let kv_heads = layer.num_key_value_heads as usize;
            let head_dim = layer.head_dim as usize;

            // Attention block.
            let mut normed = state.clone();
            rms_norm(&mut normed, hidden, Some(&w("input_layernorm.weight")), eps);
            let mut q = linear(
                &normed,
                tokens,
                hidden,
                &w("self_attn.q_proj.weight"),
                heads * head_dim,
            );
            let mut k = linear(
                &normed,
                tokens,
                hidden,
                &w("self_attn.k_proj.weight"),
                kv_heads * head_dim,
            );
            let mut v = linear(
                &normed,
                tokens,
                hidden,
                &w("self_attn.v_proj.weight"),
                kv_heads * head_dim,
            );
            rms_norm(&mut q, head_dim, Some(&w("self_attn.q_norm.weight")), eps);
            rms_norm(&mut k, head_dim, Some(&w("self_attn.k_norm.weight")), eps);
            rms_norm(&mut v, head_dim, None, eps);
            rope(&mut q, tokens, heads, head_dim, layer.rope_theta);
            rope(&mut k, tokens, kv_heads, head_dim, layer.rope_theta);

            // The attention scale is 1.0, and the query and key norms
            // replace it.
            let group = heads / kv_heads;
            let windowed = layer.layer_type == LayerType::SlidingAttention;
            let mut context = vec![0f32; tokens * heads * head_dim];
            let mut scores = vec![0f32; tokens];
            for t in 0..tokens {
                for h in 0..heads {
                    let query = &q[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim];
                    let kv = h / group;
                    let mut maximum = f32::NEG_INFINITY;
                    for s in 0..tokens {
                        let distance = s.abs_diff(t);
                        if windowed && distance > c.sliding_window as usize {
                            scores[s] = f32::NEG_INFINITY;
                            continue;
                        }
                        let key =
                            &k[(s * kv_heads + kv) * head_dim..(s * kv_heads + kv + 1) * head_dim];
                        let mut dot = 0f32;
                        for i in 0..head_dim {
                            dot += query[i] * key[i];
                        }
                        scores[s] = dot;
                        maximum = maximum.max(dot);
                    }
                    let mut total = 0f32;
                    for score in scores.iter_mut() {
                        *score = if score.is_infinite() {
                            0.0
                        } else {
                            (*score - maximum).exp()
                        };
                        total += *score;
                    }
                    let out =
                        &mut context[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim];
                    for s in 0..tokens {
                        if scores[s] == 0.0 {
                            continue;
                        }
                        let weight = scores[s] / total;
                        let value =
                            &v[(s * kv_heads + kv) * head_dim..(s * kv_heads + kv + 1) * head_dim];
                        for i in 0..head_dim {
                            out[i] += weight * value[i];
                        }
                    }
                }
            }
            let mut attended = linear(
                &context,
                tokens,
                heads * head_dim,
                &w("self_attn.o_proj.weight"),
                hidden,
            );
            rms_norm(
                &mut attended,
                hidden,
                Some(&w("post_attention_layernorm.weight")),
                eps,
            );
            for (s, a) in state.iter_mut().zip(&attended) {
                *s += a;
            }

            // Feed-forward block.
            let mut normed = state.clone();
            rms_norm(
                &mut normed,
                hidden,
                Some(&w("pre_feedforward_layernorm.weight")),
                eps,
            );
            let inner = c.intermediate_size as usize;
            let mut gate = linear(&normed, tokens, hidden, &w("mlp.gate_proj.weight"), inner);
            let up = linear(&normed, tokens, hidden, &w("mlp.up_proj.weight"), inner);
            for (g, u) in gate.iter_mut().zip(&up) {
                *g = gelu_tanh(*g) * u;
            }
            let mut down = linear(&gate, tokens, inner, &w("mlp.down_proj.weight"), hidden);
            rms_norm(
                &mut down,
                hidden,
                Some(&w("post_feedforward_layernorm.weight")),
                eps,
            );
            for (s, d) in state.iter_mut().zip(&down) {
                *s += d;
            }

            // Per-layer input block. It gates a bottleneck through the
            // layer's slice of the per-layer inputs.
            let mut gated = linear(
                &state,
                tokens,
                hidden,
                &w("ple_block.per_layer_input_gate.weight"),
                per_layer,
            );
            for t in 0..tokens {
                for i in 0..per_layer {
                    gated[t * per_layer + i] =
                        gelu_tanh(gated[t * per_layer + i]) * ple[(t * layers + l) * per_layer + i];
                }
            }
            let mut mixed = linear(
                &gated,
                tokens,
                per_layer,
                &w("ple_block.per_layer_projection.weight"),
                hidden,
            );
            rms_norm(
                &mut mixed,
                hidden,
                Some(&w("ple_block.post_per_layer_input_norm.weight")),
                eps,
            );
            let scalar = w("layer_scalar")[0];
            for (s, m) in state.iter_mut().zip(&mixed) {
                *s = (*s + m) * scalar;
            }
        }

        rms_norm(
            &mut state,
            hidden,
            Some(&weight(store, &format!("{prefix}norm.weight"))),
            eps,
        );
        linear(
            &state,
            tokens,
            hidden,
            &weight(store, &format!("{prefix}embedding_projection.weight")),
            c.embedding_dim as usize,
        )
    }

    /// Mean-pools the projected token states and L2-normalises the result.
    pub fn embed(&self, token_ids: &[u32]) -> Vec<f32> {
        let states = self.encode_tokens(token_ids);
        let width = self.config.embedding_dim as usize;
        let mut pooled = vec![0f32; width];
        for row in states.chunks_exact(width) {
            for (p, s) in pooled.iter_mut().zip(row) {
                *p += s;
            }
        }
        let squares: f32 = pooled.iter().fold(0.0, |sum, v| sum + v * v);
        let norm = squares.sqrt();
        if norm > 0.0 {
            for value in &mut pooled {
                *value /= norm;
            }
        }
        pooled
    }
}
