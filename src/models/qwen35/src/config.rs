//! Architecture from GGUF metadata, a port of `GgufReader::ExtractModelConfig`
//! (`src/core/gguf_reader.cpp`) and `core::ModelConfig`.

use gufo_core::gguf::Gguf;

/// Hidden, per-head and SSM query/key normalisation epsilon (`kNormEps`).
pub const NORM_EPS: f32 = 1e-6;

#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub architecture: String,
    /// `general.name`.
    pub model_name: String,
    /// Main transformer layers, without the MTP block.
    pub num_layers: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub full_attention_interval: u32,
    pub mtp_num_layers: u32,
    pub ssm_conv_kernel: u32,
    pub ssm_state_size: u32,
    pub ssm_group_count: u32,
    pub ssm_time_step_rank: u32,
    pub ssm_inner_size: u32,
    pub rotary_dim: u32,
    pub rope_theta: f32,
}

impl Default for ModelConfig {
    /// The C++ defaults for keys a GGUF file omits.
    fn default() -> Self {
        Self {
            architecture: "qwen35".into(),
            model_name: "qwen3.5-4b-text".into(),
            num_layers: 36,
            hidden_size: 2560,
            intermediate_size: 9728,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            vocab_size: 248_320,
            context_length: 32_768,
            full_attention_interval: 4,
            mtp_num_layers: 1,
            ssm_conv_kernel: 4,
            ssm_state_size: 128,
            ssm_group_count: 16,
            ssm_time_step_rank: 32,
            ssm_inner_size: 4096,
            rotary_dim: 64,
            rope_theta: 10_000_000.0,
        }
    }
}

impl ModelConfig {
    pub fn from_gguf(gguf: &Gguf<'_>) -> Result<Self, String> {
        let mut c = Self {
            architecture: gguf
                .string("general.architecture")
                .map(|a| String::from_utf8_lossy(a).into_owned())
                .unwrap_or_else(|| "qwen".into()),
            ..Self::default()
        };
        let key = |name: &str| format!("{}.{name}", c.architecture);
        let read = |name: &str, field: &mut u32| {
            if let Some(v) = gguf.u32(&key(name)) {
                *field = v;
            }
        };
        let total_layers = gguf.u32(&key("block_count")).unwrap_or(c.num_layers);
        read("embedding_length", &mut c.hidden_size);
        read("feed_forward_length", &mut c.intermediate_size);
        read("attention.head_count", &mut c.num_attention_heads);
        read("attention.head_count_kv", &mut c.num_key_value_heads);
        match gguf.u32(&key("attention.key_length")) {
            Some(v) => c.head_dim = v,
            None if c.num_attention_heads > 0 => {
                c.head_dim = c.hidden_size / c.num_attention_heads;
            }
            None => {}
        }
        read("context_length", &mut c.context_length);
        read("full_attention_interval", &mut c.full_attention_interval);
        read("ssm.conv_kernel", &mut c.ssm_conv_kernel);
        read("ssm.state_size", &mut c.ssm_state_size);
        read("ssm.group_count", &mut c.ssm_group_count);
        read("ssm.time_step_rank", &mut c.ssm_time_step_rank);
        read("ssm.inner_size", &mut c.ssm_inner_size);
        if let Some(v) = gguf.f32(&key("rope.freq_base")) {
            c.rope_theta = v;
        }
        c.rotary_dim = gguf.u32(&key("rope.dimension_count")).unwrap_or(c.head_dim);

        // MTP layers count towards block_count but are not part of the stack.
        let names = || gguf.tensors().iter().map(|t| t.name);
        let embedded_mtp = gguf.u32(&key("nextn_predict_layers"));
        let has_mtp =
            names().any(|n| n.starts_with("mtp.") || n.contains(".mtp.") || n.contains(".nextn."));
        c.mtp_num_layers = embedded_mtp.unwrap_or(u32::from(has_mtp));
        c.num_layers = total_layers
            .checked_sub(embedded_mtp.unwrap_or(0))
            .ok_or("MTP layer count exceeds qwen block count")?;

        if names().any(|n| {
            n.starts_with("model.visual") || n.starts_with("visual.") || n.starts_with("v.")
        }) {
            return Err(
                "Model contains vision encoder tensors; text-only contract violation".into(),
            );
        }
        if let Some(name) = gguf.string("general.name") {
            c.model_name = String::from_utf8_lossy(name).into_owned();
        }
        if let Some(embedding) = gguf.tensor("token_embd.weight")
            && c.hidden_size > 0
            && embedding
                .elements()
                .is_multiple_of(u64::from(c.hidden_size))
        {
            c.vocab_size = (embedding.elements() / u64::from(c.hidden_size)) as u32;
        }
        if !c.is_valid_qwen() {
            return Err(format!(
                "Model config failed Qwen structural validation (head_dim={}, layers={})",
                c.head_dim, c.num_layers
            ));
        }
        Ok(c)
    }

    /// The Qwen3.5/3.8 block structure the kernels assume.
    fn is_valid_qwen(&self) -> bool {
        self.head_dim > 0
            && self.num_layers > 0
            && self.hidden_size > 0
            && self.intermediate_size > 0
            && self.num_attention_heads > 0
            && self.num_key_value_heads > 0
            && self.vocab_size > 0
            && self.full_attention_interval == 4
            && self.num_layers.is_multiple_of(self.full_attention_interval)
            && self.rotary_dim > 0
            && self.rotary_dim <= self.head_dim
            && self.rotary_dim.is_multiple_of(2)
            && self.ssm_conv_kernel == 4
            && self.ssm_state_size == 128
            && self.ssm_group_count > 0
            && self.ssm_time_step_rank > 0
            && self.ssm_inner_size > 0
            && self.ssm_inner_size.is_multiple_of(self.ssm_time_step_rank)
            && self.ssm_value_size() == 128
    }

    pub fn attention_size(&self) -> u32 {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_size(&self) -> u32 {
        self.num_key_value_heads * self.head_dim
    }

    /// Every `full_attention_interval`-th layer, counting from 1.
    pub fn is_full_attention(&self, layer: u32) -> bool {
        (layer + 1).is_multiple_of(self.full_attention_interval)
    }

    pub fn full_attention_layer_count(&self) -> u32 {
        self.num_layers / self.full_attention_interval
    }

    pub fn ssm_layer_count(&self) -> u32 {
        self.num_layers - self.full_attention_layer_count()
    }

    /// Dense index of a recurrent layer among the recurrent layers.
    pub fn ssm_layer_index(&self, layer: u32) -> u32 {
        layer - layer / self.full_attention_interval
    }

    pub fn ssm_value_size(&self) -> u32 {
        self.ssm_inner_size / self.ssm_time_step_rank
    }

    pub fn ssm_qkv_size(&self) -> u32 {
        2 * self.ssm_group_count * self.ssm_state_size + self.ssm_inner_size
    }
}

/// Whether the GGUF is a Qwen3.8 artifact, whose chat template must be a
/// pinned one: named so, or shaped like the 27B.
pub fn is_qwen38_artifact(gguf: &Gguf<'_>) -> bool {
    gufo_qwen::chat_template::is_qwen38_named(gguf)
        || ModelConfig::from_gguf(gguf).is_ok_and(|config| {
            config.num_layers == 64 && config.hidden_size == 5120 && config.vocab_size == 248_320
        })
}
