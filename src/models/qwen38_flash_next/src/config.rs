//! Architecture of a `qwen4exp` artifact, a port of `Config::FromGguf`
//! (`config.cpp`). Every value comes from the file; the fixed checks at the
//! end bound what the kernels were written and validated for
//! (Qwen3.8-Flash-Next, 48 trunk layers).

use gufo_core::gguf::{Gguf, Value};

use crate::rope_scaling::{RopeScaling, scaled_context_length};

pub const MAX_PLE_HEADS: usize = 16;
pub const MAX_PLE_NGRAM: usize = 3;

const ARCHITECTURE: &[u8] = b"qwen4exp";

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Trunk layers (the MTP block excluded).
    pub num_layers: u32,
    /// `block_count`, the MTP block included.
    pub num_layers_all: u32,
    pub hidden_size: u32,
    /// The embedding row count; set by the weight binding.
    pub vocab_size: u32,
    pub context_length: u32,
    pub rms_eps: f32,

    /// Hyper-connections: `hc_count` residual streams of `hidden_size`.
    pub hc_count: u32,
    pub hc_low_rank: u32,

    /// Gated GQA attention on every `full_attention_interval`-th layer.
    pub full_attention_interval: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub rope_theta: f32,
    pub rope_sections: [u32; 4],

    /// Sparse attention indexer (block top-k selection).
    pub indexer_heads: u32,
    pub indexer_head_dim: u32,
    pub indexer_top_k: u32,
    /// Tokens per scored block.
    pub compress_ratio: u32,

    /// Gated DeltaNet.
    pub ssm_conv_kernel: u32,
    pub ssm_head_dim: u32,
    pub ssm_num_k_heads: u32,
    pub ssm_num_v_heads: u32,
    pub ssm_inner_size: u32,

    /// Mixture of experts with one always-on shared expert.
    pub num_experts: u32,
    pub num_experts_used: u32,
    pub expert_ff: u32,
    pub shared_expert_ff: u32,

    /// The linear layer the PLE n-gram embedding is injected into.
    pub ple_layer: Option<u32>,
    pub ple_ngram_size: u32,
    pub ple_heads_per_ngram: u32,
    pub ple_heads: u32,
    pub ple_head_dim: u32,
    pub ple_conv_kernel: u32,
    /// Resets the n-gram window.
    pub ple_eos_token: u32,
    pub ple_multipliers: [u64; MAX_PLE_NGRAM],
    pub ple_head_offsets: [u32; MAX_PLE_HEADS],
    pub ple_head_vocab: [u32; MAX_PLE_HEADS],
    /// max(offset + vocab) over heads.
    pub ple_rows: u64,

    /// Speculative MTP blocks (`nextn`), present in the draft sidecar only.
    pub nextn_layers: u32,

    /// Static YaRN, set by `ModelOptions::rope_scaling`; no GGUF key sets
    /// it. The native `context_length` is never rewritten.
    pub rope_scaling: RopeScaling,
}

impl Config {
    /// The longest session the rope configuration supports
    /// (`MaxContextLength`).
    pub fn max_context_length(&self) -> u32 {
        scaled_context_length(&self.rope_scaling, self.context_length)
    }

    /// The `u32` words of a sparse-mask row over `context` tokens: one bit
    /// per `compress_ratio` block.
    pub fn sparse_mask_words(&self, context: u32) -> u32 {
        context.div_ceil(self.compress_ratio).div_ceil(32)
    }

    pub fn hc_dim(&self) -> u32 {
        self.hc_count * self.hidden_size
    }

    pub fn ssm_key_dim(&self) -> u32 {
        self.ssm_num_k_heads * self.ssm_head_dim
    }

    pub fn ssm_value_dim(&self) -> u32 {
        self.ssm_num_v_heads * self.ssm_head_dim
    }

    /// Channels of the q|k|v projection and its causal convolution.
    pub fn ssm_conv_channels(&self) -> u32 {
        2 * self.ssm_key_dim() + self.ssm_value_dim()
    }

    pub fn attention_q_dim(&self) -> u32 {
        self.num_heads * self.head_dim
    }

    pub fn attention_kv_dim(&self) -> u32 {
        self.num_kv_heads * self.head_dim
    }

    pub fn ple_embedding_dim(&self) -> u32 {
        self.ple_heads * self.ple_head_dim
    }

    /// Tokens of PLE conv history per sequence: (kernel - 1) * dilation.
    pub fn ple_conv_history(&self) -> u32 {
        self.ple_conv_kernel.saturating_sub(1) * self.ple_ngram_size
    }

    pub fn is_linear_layer(&self, layer: u32) -> bool {
        layer < self.num_layers && !(layer + 1).is_multiple_of(self.full_attention_interval)
    }

    pub fn is_ple_layer(&self, layer: u32) -> bool {
        self.ple_layer == Some(layer)
    }

    /// The draft block runs with the trunk's constants and shares the
    /// vocabulary weights; PLE and SSM parameters do not apply to it.
    pub fn mtp_matches(&self, trunk: &Config) -> bool {
        self.nextn_layers == 1
            && self.num_layers == trunk.num_layers
            && self.hidden_size == trunk.hidden_size
            && self.context_length == trunk.context_length
            && self.rms_eps == trunk.rms_eps
            && self.hc_count == trunk.hc_count
            && self.hc_low_rank == trunk.hc_low_rank
            && self.num_heads == trunk.num_heads
            && self.num_kv_heads == trunk.num_kv_heads
            && self.head_dim == trunk.head_dim
            && self.rotary_dim == trunk.rotary_dim
            && self.rope_theta == trunk.rope_theta
            && self.rope_sections == trunk.rope_sections
            && self.indexer_heads == trunk.indexer_heads
            && self.indexer_head_dim == trunk.indexer_head_dim
            && self.indexer_top_k == trunk.indexer_top_k
            && self.compress_ratio == trunk.compress_ratio
            && self.num_experts == trunk.num_experts
            && self.num_experts_used == trunk.num_experts_used
            && self.expert_ff == trunk.expert_ff
            && self.shared_expert_ff == trunk.shared_expert_ff
    }

    /// Reads and validates the `qwen4exp.*` metadata. `require_trunk`
    /// rejects draft-only sidecars; the MTP sidecar is read without it.
    pub fn from_gguf(gguf: &Gguf<'_>, require_trunk: bool) -> Result<Self, String> {
        if gguf.string("general.architecture") != Some(ARCHITECTURE) {
            return Err("GGUF architecture is not qwen4exp".into());
        }
        let mut r = Reader { gguf, error: None };
        let mut c = Config::empty();
        c.num_layers_all = r.u32("block_count", None);
        c.nextn_layers = r.u32("nextn_predict_layers", Some(0));
        if r.error.is_none() && c.nextn_layers >= c.num_layers_all {
            return Err("nextn_predict_layers must be below block_count".into());
        }
        c.num_layers = c.num_layers_all.saturating_sub(c.nextn_layers);
        c.hidden_size = r.u32("embedding_length", None);
        c.context_length = r.u32("context_length", None);
        c.rms_eps = r.f32("attention.layer_norm_rms_epsilon", 1e-6);

        c.hc_count = r.u32("hyper_connection.count", None);
        c.hc_low_rank = r.u32("hyper_connection.low_rank", None);

        c.full_attention_interval = r.u32("full_attention_interval", None);
        c.num_heads = r.u32("attention.head_count", None);
        c.num_kv_heads = r.u32("attention.head_count_kv", None);
        c.head_dim = r.u32("attention.key_length", None);
        if r.u32("attention.value_length", Some(c.head_dim)) != c.head_dim {
            r.fail("attention value length must match the supported key length");
        }
        c.rotary_dim = r.u32("rope.dimension_count", None);
        c.rope_theta = r.f32("rope.freq_base", 10_000_000.0);
        let sections = r.u64_array("rope.dimension_sections", true);
        if sections.len() != 4 || sections[3] != 0 {
            r.fail("rope.dimension_sections must contain three axes and a zero pad");
        } else {
            let mut sum = 0u64;
            for (i, &section) in sections[..3].iter().enumerate() {
                if section == 0 || section > u64::from(c.rotary_dim / 2) {
                    r.fail("invalid mRoPE axis dimension");
                    break;
                }
                sum += section;
                c.rope_sections[i] = section as u32;
            }
            if sum != u64::from(c.rotary_dim / 2) {
                r.fail("mRoPE sections must sum to half the rotary dimension");
            }
        }

        c.indexer_heads = r.u32("attention.indexer.head_count", None);
        c.indexer_head_dim = r.u32("attention.indexer.key_length", None);
        c.indexer_top_k = r.u32("attention.indexer.top_k", None);
        // One ratio per layer, zero on linear layers; every attention layer
        // of this model shares one value, which is all the runtime runs.
        let ratios = r.u64_array("attention.compress_ratios", true);
        if ratios.len() != c.num_layers_all as usize || c.full_attention_interval == 0 {
            r.fail("attention.compress_ratios must describe every layer");
        } else {
            for (i, &ratio) in ratios.iter().enumerate() {
                let layer = i as u32;
                if layer >= c.num_layers {
                    // Older converters append zero for nextn layers although
                    // their attention exports indexer weights; the MTP block
                    // uses the same ratio-four indexer.
                    if ratio != 0 && ratio != 4 {
                        r.fail("unsupported MTP attention compression ratio");
                    }
                    continue;
                }
                if c.is_linear_layer(layer) {
                    if ratio != 0 {
                        r.fail("linear layers cannot use attention compression");
                    }
                } else if ratio == 0 || ratio > 4096 {
                    r.fail("attention compression must fit the 4096-row index ring");
                } else if c.compress_ratio == 0 {
                    c.compress_ratio = ratio as u32;
                } else if u64::from(c.compress_ratio) != ratio {
                    r.fail("attention.compress_ratios differ between attention layers");
                }
            }
        }

        c.ssm_conv_kernel = r.u32("ssm.conv_kernel", None);
        c.ssm_head_dim = r.u32("ssm.state_size", None);
        c.ssm_num_k_heads = r.u32("ssm.group_count", None);
        c.ssm_num_v_heads = r.u32("ssm.time_step_rank", None);
        c.ssm_inner_size = r.u32("ssm.inner_size", None);

        c.num_experts = r.u32("expert_count", None);
        c.num_experts_used = r.u32("expert_used_count", None);
        c.expert_ff = r.u32("expert_feed_forward_length", None);
        c.shared_expert_ff = r.u32("expert_shared_feed_forward_length", None);

        let layers = r.u64_array("ple.layers", false);
        if layers.len() > 1 {
            r.fail("only one PLE layer is supported");
        } else if let Some(&layer) = layers.first() {
            if layer >= u64::from(c.num_layers) {
                return Err(r
                    .error
                    .unwrap_or_else(|| "PLE layer lies outside the trunk".into()));
            }
            c.ple_layer = Some(layer as u32);
            c.ple_ngram_size = r.u32("ple.ngram_size", None);
            c.ple_heads_per_ngram = r.u32("ple.heads_per_ngram", None);
            c.ple_conv_kernel = r.u32("ple.conv_kernel", None);
            c.ple_eos_token = r.u32("ple.eos_token_id", None);
            c.ple_head_dim = r.u32("embedding_length_per_layer_input", None);
            let ngram = c.ple_ngram_size as usize;
            if !(2..=MAX_PLE_NGRAM).contains(&ngram)
                || c.ple_heads_per_ngram == 0
                || c.ple_heads_per_ngram as usize > MAX_PLE_HEADS / (ngram - 1)
                || c.ple_head_dim == 0
                || c.ple_conv_kernel == 0
            {
                r.fail("unsupported PLE n-gram geometry");
            } else {
                c.ple_heads = (c.ple_ngram_size - 1) * c.ple_heads_per_ngram;
                let multipliers = r.u64_array("ple.layer_multipliers", true);
                let offsets = r.u64_array("ple.head_offsets", true);
                let vocab = r.u64_array("ple.head_vocab_sizes", true);
                let heads = c.ple_heads as usize;
                if multipliers.len() != ngram || offsets.len() != heads || vocab.len() != heads {
                    r.fail("PLE hash tables must match the head count");
                } else {
                    c.ple_multipliers[..ngram].copy_from_slice(&multipliers);
                    for h in 0..heads {
                        let max = u64::from(u32::MAX);
                        if vocab[h] == 0 || vocab[h] > max || offsets[h] > max - vocab[h] {
                            r.fail("PLE head range does not fit a 32-bit row index");
                            break;
                        }
                        c.ple_head_offsets[h] = offsets[h] as u32;
                        c.ple_head_vocab[h] = vocab[h] as u32;
                        c.ple_rows = c.ple_rows.max(offsets[h] + vocab[h]);
                    }
                }
            }
        }

        if let Some(error) = r.error {
            return Err(error);
        }
        c.check(require_trunk)?;
        Ok(c)
    }

    fn empty() -> Self {
        Self {
            num_layers: 0,
            num_layers_all: 0,
            hidden_size: 0,
            vocab_size: 0,
            context_length: 0,
            rms_eps: 1e-6,
            hc_count: 0,
            hc_low_rank: 0,
            full_attention_interval: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            rotary_dim: 0,
            rope_theta: 0.0,
            rope_sections: [0; 4],
            indexer_heads: 0,
            indexer_head_dim: 0,
            indexer_top_k: 0,
            compress_ratio: 0,
            ssm_conv_kernel: 0,
            ssm_head_dim: 0,
            ssm_num_k_heads: 0,
            ssm_num_v_heads: 0,
            ssm_inner_size: 0,
            num_experts: 0,
            num_experts_used: 0,
            expert_ff: 0,
            shared_expert_ff: 0,
            ple_layer: None,
            ple_ngram_size: 0,
            ple_heads_per_ngram: 0,
            ple_heads: 0,
            ple_head_dim: 0,
            ple_conv_kernel: 0,
            ple_eos_token: 0,
            ple_multipliers: [0; MAX_PLE_NGRAM],
            ple_head_offsets: [0; MAX_PLE_HEADS],
            ple_head_vocab: [0; MAX_PLE_HEADS],
            ple_rows: 0,
            nextn_layers: 0,
            rope_scaling: RopeScaling::default(),
        }
    }

    /// The runtime's structural limits, then the one geometry its kernels
    /// are specialised for.
    fn check(&self, require_trunk: bool) -> Result<(), String> {
        let c = self;
        let shape_ok = c.context_length != 0
            && c.context_length <= i32::MAX as u32
            && c.rms_eps.is_finite()
            && c.rms_eps > 0.0
            && c.rope_theta.is_finite()
            && c.rope_theta > 0.0
            && c.hidden_size != 0
            && c.hc_count >= 2
            && c.hc_low_rank != 0
            && c.full_attention_interval != 0
            && c.num_heads != 0
            && c.num_kv_heads != 0
            && c.num_heads.is_multiple_of(c.num_kv_heads)
            && c.head_dim != 0
            && c.rotary_dim != 0
            && c.rotary_dim.is_multiple_of(2)
            && c.rotary_dim <= c.head_dim
            && c.indexer_heads != 0
            && c.indexer_head_dim != 0
            && c.rotary_dim <= c.indexer_head_dim
            && c.compress_ratio != 0
            && c.indexer_top_k != 0
            && c.indexer_top_k.is_multiple_of(c.compress_ratio)
            && c.ssm_conv_kernel != 0
            && c.ssm_head_dim != 0
            && c.ssm_num_k_heads != 0
            && c.ssm_num_v_heads != 0
            && c.ssm_num_v_heads.is_multiple_of(c.ssm_num_k_heads)
            && c.ssm_inner_size == c.ssm_value_dim()
            && c.num_experts != 0
            && c.num_experts_used != 0
            && c.num_experts_used <= c.num_experts
            && c.expert_ff != 0
            && c.shared_expert_ff != 0;
        if !shape_ok {
            return Err("qwen4exp metadata describes an unsupported shape".into());
        }
        // These dimensions select the specialised attention, indexer,
        // DeltaNet and expert kernels; divisibility alone does not make a
        // shape executable.
        let ple_ok = c.ple_layer.is_none()
            || (c.ple_ngram_size == 3
                && c.ple_heads_per_ngram == 8
                && c.ple_head_dim == 160
                && c.ple_conv_kernel == 4);
        let geometry_ok = c.num_layers == 48
            && c.nextn_layers <= 1
            && c.hidden_size == 2560
            && c.hc_count == 4
            && c.hc_low_rank == 320
            && c.full_attention_interval == 4
            && c.num_heads == 24
            && c.num_kv_heads == 2
            && c.head_dim == 256
            && c.rotary_dim == 64
            && c.rope_sections == [11, 11, 10, 0]
            && c.indexer_heads == 4
            && c.indexer_head_dim == 128
            && c.indexer_top_k == 2048
            && c.compress_ratio == 4
            && c.ssm_conv_kernel == 4
            && c.ssm_head_dim == 128
            && c.ssm_num_k_heads == 16
            && c.ssm_num_v_heads == 48
            && c.num_experts == 512
            && c.num_experts_used == 10
            && c.expert_ff == 640
            && c.shared_expert_ff == 640
            && ple_ok;
        if !geometry_ok {
            return Err("unsupported Qwen3.8-Flash-Next kernel geometry".into());
        }
        if require_trunk {
            if c.num_layers == 0 || !c.num_layers.is_multiple_of(c.full_attention_interval) {
                return Err("qwen4exp trunk layer count is not a multiple of the interval".into());
            }
            if let Some(layer) = c.ple_layer
                && !c.is_linear_layer(layer)
            {
                return Err("PLE layer must be a linear attention layer".into());
            }
        }
        Ok(())
    }
}

/// Metadata of the `qwen4exp.` namespace; the first failure is kept.
struct Reader<'g, 'a> {
    gguf: &'g Gguf<'a>,
    error: Option<String>,
}

impl Reader<'_, '_> {
    fn fail(&mut self, message: &str) {
        self.error.get_or_insert_with(|| message.to_string());
    }

    /// A required key (`fallback` None) or an optional one; a present key of
    /// the wrong type fails either way.
    fn u32(&mut self, suffix: &str, fallback: Option<u32>) -> u32 {
        let key = format!("qwen4exp.{suffix}");
        if let Some(value) = self.gguf.u32(&key) {
            return value;
        }
        if fallback.is_none() || self.gguf.metadata(&key).is_some() {
            self.fail(&format!("missing or invalid GGUF integer {key}"));
        }
        fallback.unwrap_or(0)
    }

    fn f32(&mut self, suffix: &str, fallback: f32) -> f32 {
        let key = format!("qwen4exp.{suffix}");
        match self.gguf.f32(&key) {
            Some(value) if value.is_finite() => value,
            _ => {
                self.fail(&format!("missing or invalid GGUF float {key}"));
                fallback
            }
        }
    }

    /// Integer arrays of any width; GGUF writers pick the narrowest type.
    fn u64_array(&mut self, suffix: &str, required: bool) -> Vec<u64> {
        let key = format!("qwen4exp.{suffix}");
        match self.gguf.metadata(&key) {
            None => {
                if required {
                    self.fail(&format!("missing GGUF array {key}"));
                }
                Vec::new()
            }
            Some(Value::UIntArray(values)) => values.clone(),
            Some(Value::IntArray(values)) => {
                if values.iter().any(|&v| v < 0) {
                    self.fail(&format!("negative GGUF array value {key}"));
                    return Vec::new();
                }
                values.iter().map(|&v| v as u64).collect()
            }
            Some(_) => {
                self.fail(&format!("GGUF array must contain integers: {key}"));
                Vec::new()
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;

    /// The Flash-Next geometry with made-up PLE hash constants.
    pub(crate) fn flash_next() -> Config {
        let mut c = Config::empty();
        c.num_layers = 48;
        c.num_layers_all = 49;
        c.nextn_layers = 1;
        c.hidden_size = 2560;
        c.vocab_size = 248_320;
        c.context_length = 262_144;
        c.hc_count = 4;
        c.hc_low_rank = 320;
        c.full_attention_interval = 4;
        c.num_heads = 24;
        c.num_kv_heads = 2;
        c.head_dim = 256;
        c.rotary_dim = 64;
        c.rope_theta = 1e7;
        c.rope_sections = [11, 11, 10, 0];
        c.indexer_heads = 4;
        c.indexer_head_dim = 128;
        c.indexer_top_k = 2048;
        c.compress_ratio = 4;
        c.ssm_conv_kernel = 4;
        c.ssm_head_dim = 128;
        c.ssm_num_k_heads = 16;
        c.ssm_num_v_heads = 48;
        c.ssm_inner_size = 6144;
        c.num_experts = 512;
        c.num_experts_used = 10;
        c.expert_ff = 640;
        c.shared_expert_ff = 640;
        c.ple_layer = Some(1);
        c.ple_ngram_size = 3;
        c.ple_heads_per_ngram = 8;
        c.ple_heads = 16;
        c.ple_head_dim = 160;
        c.ple_conv_kernel = 4;
        c.ple_multipliers = [
            0x9E37_79B9_7F4A_7C15,
            0xC2B2_AE3D_27D4_EB4F,
            0x1656_67B1_9E37_79F9,
        ];
        for h in 0..16 {
            c.ple_head_vocab[h] = 1_000_003 + h as u32 * 17;
            c.ple_head_offsets[h] = h as u32 * 2_000_000;
        }
        c.ple_rows = 32_000_000;
        c
    }

    #[test]
    fn the_test_geometry_passes_the_runtime_checks() {
        flash_next().check(true).unwrap();
    }
}

/// The metadata checks of `tests/models/qwen38_flash_next/config_test.cpp`.
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[derive(Clone)]
    enum Field {
        U32(u32),
        F32(f32),
        Str(&'static str),
        U64s(Vec<u64>),
        I64s(Vec<i64>),
        F64s(Vec<f64>),
    }

    type Metadata = BTreeMap<String, Field>;

    /// A metadata-only GGUF: no model is needed to test the loader.
    fn parse(fields: &Metadata, trunk: bool) -> Result<Config, String> {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&(fields.len() as u64).to_le_bytes());
        let text = |b: &mut Vec<u8>, s: &[u8]| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s);
        };
        for (key, value) in fields {
            text(&mut b, key.as_bytes());
            match value {
                Field::U32(v) => {
                    b.extend_from_slice(&4u32.to_le_bytes());
                    b.extend_from_slice(&v.to_le_bytes());
                }
                Field::F32(v) => {
                    b.extend_from_slice(&6u32.to_le_bytes());
                    b.extend_from_slice(&v.to_le_bytes());
                }
                Field::Str(v) => {
                    b.extend_from_slice(&8u32.to_le_bytes());
                    text(&mut b, v.as_bytes());
                }
                Field::U64s(v) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&10u32.to_le_bytes());
                    b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    v.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
                }
                Field::I64s(v) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&11u32.to_le_bytes());
                    b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    v.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
                }
                Field::F64s(v) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&12u32.to_le_bytes());
                    b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    v.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
                }
            }
        }
        b.resize(b.len().next_multiple_of(32), 0);
        let gguf = Gguf::parse(&b).map_err(|e| format!("GGUF: {e}"))?;
        Config::from_gguf(&gguf, trunk)
    }

    fn valid() -> Metadata {
        let mut fields = Metadata::new();
        fields.insert("general.architecture".into(), Field::Str("qwen4exp"));
        for (key, value) in [
            ("block_count", 48),
            ("embedding_length", 2560),
            ("context_length", 262_144),
            ("hyper_connection.count", 4),
            ("hyper_connection.low_rank", 320),
            ("full_attention_interval", 4),
            ("attention.head_count", 24),
            ("attention.head_count_kv", 2),
            ("attention.key_length", 256),
            ("rope.dimension_count", 64),
            ("attention.indexer.head_count", 4),
            ("attention.indexer.key_length", 128),
            ("attention.indexer.top_k", 2048),
            ("ssm.conv_kernel", 4),
            ("ssm.state_size", 128),
            ("ssm.group_count", 16),
            ("ssm.time_step_rank", 48),
            ("ssm.inner_size", 6144),
            ("expert_count", 512),
            ("expert_used_count", 10),
            ("expert_feed_forward_length", 640),
            ("expert_shared_feed_forward_length", 640),
        ] {
            fields.insert(format!("qwen4exp.{key}"), Field::U32(value));
        }
        fields.insert(
            "qwen4exp.attention.layer_norm_rms_epsilon".into(),
            Field::F32(1e-6),
        );
        fields.insert("qwen4exp.rope.freq_base".into(), Field::F32(1e7));
        fields.insert(
            "qwen4exp.rope.dimension_sections".into(),
            Field::U64s(vec![11, 11, 10, 0]),
        );
        let ratios = (0..48).map(|i| if i % 4 == 3 { 4 } else { 0 }).collect();
        fields.insert(
            "qwen4exp.attention.compress_ratios".into(),
            Field::U64s(ratios),
        );
        fields
    }

    fn ratios(fields: &mut Metadata) -> &mut Vec<u64> {
        match fields.get_mut("qwen4exp.attention.compress_ratios") {
            Some(Field::U64s(values)) => values,
            _ => unreachable!(),
        }
    }

    #[test]
    fn the_mtp_sidecar_must_share_the_trunk_constants() {
        let mut trunk = tests_support::flash_next();
        let mut sidecar = trunk.clone();
        sidecar.nextn_layers = 1;
        assert!(
            sidecar.mtp_matches(&trunk),
            "matching MTP constants rejected"
        );
        let integers: [fn(&mut Config) -> &mut u32; 5] = [
            |c| &mut c.hc_low_rank,
            |c| &mut c.context_length,
            |c| &mut c.rotary_dim,
            |c| &mut c.num_experts_used,
            |c| &mut c.shared_expert_ff,
        ];
        for member in integers {
            let mut wrong = sidecar.clone();
            *member(&mut wrong) += 1;
            assert!(
                !wrong.mtp_matches(&trunk),
                "dimension-compatible MTP with different semantics accepted"
            );
        }
        let floats: [fn(&mut Config) -> &mut f32; 2] = [|c| &mut c.rms_eps, |c| &mut c.rope_theta];
        for member in floats {
            let mut wrong = sidecar.clone();
            *member(&mut wrong) *= 2.0;
            assert!(
                !wrong.mtp_matches(&trunk),
                "MTP numerical constant mismatch accepted"
            );
        }
        let mut wrong = sidecar.clone();
        wrong.rope_sections.swap(0, 2);
        assert!(
            !wrong.mtp_matches(&trunk),
            "MTP RoPE partition mismatch accepted"
        );
        sidecar.ple_layer = None;
        trunk.ple_layer = Some(1);
        assert!(
            sidecar.mtp_matches(&trunk),
            "trunk-only PLE incorrectly required in MTP"
        );
    }

    #[test]
    fn malformed_metadata_is_refused() {
        let valid = valid();
        parse(&valid, true).unwrap();
        for key in [
            "qwen4exp.rope.dimension_sections",
            "qwen4exp.attention.compress_ratios",
        ] {
            let mut wrong = valid.clone();
            wrong.remove(key);
            assert!(parse(&wrong, true).is_err(), "missing {key}");
            for value in [
                Field::U32(4),
                Field::Str("4"),
                Field::U64s(Vec::new()),
                Field::I64s(vec![-1]),
                Field::F64s(vec![4.0]),
                Field::F64s(vec![f64::NAN]),
            ] {
                let mut wrong = valid.clone();
                wrong.insert(key.into(), value);
                assert!(parse(&wrong, true).is_err(), "{key}");
            }
        }
        for sections in [
            vec![11, 11, 10],
            vec![11, 11, 10, 0, 0],
            vec![11, 11, 10, 1],
            vec![11, 11, 11, 0],
            vec![10, 11, 11, 0],
            vec![0, 16, 16, 0],
            vec![u64::MAX, 11, 10, 0],
        ] {
            let mut wrong = valid.clone();
            wrong.insert(
                "qwen4exp.rope.dimension_sections".into(),
                Field::U64s(sections.clone()),
            );
            assert!(parse(&wrong, true).is_err(), "sections {sections:?}");
        }
        for (index, value) in [(0, 4), (3, 0), (7, 8), (3, 4097), (3, u64::MAX)] {
            let mut wrong = valid.clone();
            ratios(&mut wrong)[index] = value;
            assert!(parse(&wrong, true).is_err(), "ratio {index} = {value}");
        }
        for (key, value) in [
            ("attention.indexer.key_length", 32),
            ("attention.indexer.head_count", 0),
            ("attention.indexer.head_count", 8),
            ("attention.indexer.key_length", 256),
            ("attention.key_length", 128),
            ("attention.value_length", 128),
            ("embedding_length", 5120),
            ("ssm.state_size", 64),
            ("expert_count", 2048),
            ("expert_used_count", 11),
            ("attention.indexer.top_k", 2047),
            ("nextn_predict_layers", 48),
        ] {
            let mut wrong = valid.clone();
            wrong.insert(format!("qwen4exp.{key}"), Field::U32(value));
            assert!(parse(&wrong, true).is_err(), "{key} = {value}");
        }
        for value in [
            Field::Str("4"),
            Field::I64s(vec![-1]),
            Field::U64s(vec![48]),
        ] {
            let mut wrong = valid.clone();
            wrong.insert("qwen4exp.ple.layers".into(), value);
            assert!(parse(&wrong, true).is_err(), "ple.layers");
        }
        let mut sidecar = valid.clone();
        sidecar.insert("qwen4exp.block_count".into(), Field::U32(49));
        sidecar.insert("qwen4exp.nextn_predict_layers".into(), Field::U32(1));
        ratios(&mut sidecar).push(0);
        let mut sparse = sidecar.clone();
        *ratios(&mut sparse).last_mut().unwrap() = 4;
        parse(&sparse, false).unwrap();
        *ratios(&mut sparse).last_mut().unwrap() = 8;
        assert!(parse(&sparse, false).is_err(), "MTP ratio 8 accepted");
        let (draft, trunk) = (
            parse(&sidecar, false).unwrap(),
            parse(&valid, true).unwrap(),
        );
        assert!(draft.mtp_matches(&trunk));
    }
}
