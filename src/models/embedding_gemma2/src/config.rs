//! `config.json` of `google/embeddinggemma-2` (`config.cpp`): the text
//! backbone's geometry and the check that only the released BF16
//! configuration runs.

use std::path::Path;

use gufo_core::json::{self, Value};

/// Square root of the hidden size, rounded to BF16 (sqrt(512) = 22.6274
/// becomes 22.625) the way the checkpoint's BF16 execution does. The device
/// kernels also round the scaled embedding to BF16 as that execution does;
/// the CPU reference applies the same constant but keeps the product in f32.
pub const EMBED_SCALE: f32 = 22.625;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

impl LayerType {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "sliding_attention" => Some(Self::SlidingAttention),
            "full_attention" => Some(Self::FullAttention),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::FullAttention => "full_attention",
            Self::SlidingAttention => "sliding_attention",
        }
    }
}

/// Attention geometry of one decoder layer. Full-attention layers use a
/// wider head and fewer key/value heads than sliding-window layers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerGeometry {
    pub layer_type: LayerType,
    pub head_dim: u32,
    pub num_key_value_heads: u32,
    pub rope_theta: f32,
}

impl Default for LayerGeometry {
    fn default() -> Self {
        Self {
            layer_type: LayerType::SlidingAttention,
            head_dim: 256,
            num_key_value_heads: 2,
            rope_theta: 10000.0,
        }
    }
}

/// The text backbone. The released checkpoint also carries vision and audio
/// towers; this implementation reads none of their tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct TextConfig {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub hidden_size_per_layer_input: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub embedding_dim: u32,
    /// Inclusive radius. A sliding layer attends to |query - key| <= radius,
    /// which is 1,025 keys for 512. Hugging Face's
    /// sliding_window_bidirectional_overlay in transformers/masking_utils.py
    /// returns abs(q_idx - kv_idx) <= sliding_window with
    /// config.sliding_window.
    pub sliding_window: u32,
    pub rms_norm_eps: f32,
    pub layers: Vec<LayerGeometry>,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 262144,
            hidden_size: 512,
            hidden_size_per_layer_input: 512,
            intermediate_size: 2048,
            num_hidden_layers: 24,
            num_attention_heads: 4,
            embedding_dim: 768,
            sliding_window: 512,
            rms_norm_eps: 1.0e-6,
            layers: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelConfig {
    pub model_type: String,
    pub architecture: String,
    pub dtype: String,
    pub text_model_type: String,
    pub hidden_activation: String,
    pub text: TextConfig,
}

/// A number member stored into `output`, truncated to 32 bits like C++.
fn parse_u32(object: &Value, name: &str, output: &mut u32) {
    if let Some(value @ Value::Number(_)) = object.find(name) {
        *output = value.as_size(*output as usize) as u32;
    }
}

fn parse_f32(object: &Value, name: &str, output: &mut f32) {
    if let Some(value @ Value::Number(_)) = object.find(name) {
        *output = value.as_double(f64::from(*output)) as f32;
    }
}

/// `std::from_chars` into `uint32_t` over the whole key ("05" is 5).
fn parse_layer_index(key: &str) -> Option<u32> {
    if key.is_empty() || !key.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    key.parse().ok()
}

fn parse_text_config(value: &Value, output: &mut TextConfig) -> bool {
    parse_u32(value, "vocab_size", &mut output.vocab_size);
    parse_u32(value, "hidden_size", &mut output.hidden_size);
    parse_u32(
        value,
        "hidden_size_per_layer_input",
        &mut output.hidden_size_per_layer_input,
    );
    parse_u32(value, "intermediate_size", &mut output.intermediate_size);
    parse_u32(value, "num_hidden_layers", &mut output.num_hidden_layers);
    parse_u32(
        value,
        "num_attention_heads",
        &mut output.num_attention_heads,
    );
    parse_u32(value, "embedding_dim", &mut output.embedding_dim);
    parse_u32(value, "sliding_window", &mut output.sliding_window);
    parse_f32(value, "rms_norm_eps", &mut output.rms_norm_eps);

    let mut head_dim = 256;
    let mut num_key_value_heads = 2;
    parse_u32(value, "head_dim", &mut head_dim);
    parse_u32(value, "num_key_value_heads", &mut num_key_value_heads);

    let Some(types) = value
        .find("layer_types")
        .filter(|t| t.is_array() && t.len() == output.num_hidden_layers as usize)
    else {
        return false;
    };
    output.layers = vec![LayerGeometry::default(); output.num_hidden_layers as usize];
    for (layer, name) in output.layers.iter_mut().zip(types.items()) {
        let Some(layer_type) = name
            .is_string()
            .then(|| LayerType::parse(name.str()))
            .flatten()
        else {
            return false;
        };
        layer.layer_type = layer_type;
        layer.head_dim = head_dim;
        layer.num_key_value_heads = num_key_value_heads;
    }

    // The checkpoint keys overrides by layer index, zero-padded as in "05".
    if let Some(overrides) = value.find("per_layer_config").filter(|o| o.is_object()) {
        for (key, layer) in overrides.members() {
            let Some(index) =
                parse_layer_index(key).filter(|&i| (i as usize) < output.layers.len())
            else {
                return false;
            };
            let geometry = &mut output.layers[index as usize];
            parse_u32(layer, "head_dim", &mut geometry.head_dim);
            parse_u32(
                layer,
                "num_key_value_heads",
                &mut geometry.num_key_value_heads,
            );
        }
    }

    let Some(rope) = value.find("rope_parameters").filter(|r| r.is_object()) else {
        return false;
    };
    for layer in &mut output.layers {
        let Some(parameters) = rope
            .find(layer.layer_type.name())
            .filter(|p| p.is_object() && p.member_str("rope_type", "") == "default")
        else {
            return false;
        };
        parse_f32(parameters, "rope_theta", &mut layer.rope_theta);
    }
    true
}

impl ModelConfig {
    /// `ParseModelConfig`: `None` for malformed JSON, a missing text backbone
    /// or inconsistent layer tables.
    pub fn parse(json_contents: &[u8]) -> Option<Self> {
        let root = json::parse(json_contents).ok()?;
        if !root.is_object() {
            return None;
        }
        let mut config = Self {
            model_type: root.member_str("model_type", ""),
            dtype: root.member_str("dtype", ""),
            ..Self::default()
        };
        if let Some(Value::String(first)) =
            root.find("architectures").and_then(|a| a.items().first())
        {
            config.architecture = first.clone();
        }
        let text = root.find("text_config").filter(|t| t.is_object())?;
        config.text_model_type = text.member_str("model_type", "");
        config.hidden_activation = text.member_str("hidden_activation", "");
        parse_text_config(text, &mut config.text).then_some(config)
    }

    /// `LoadModelConfigFromPath`: `<model_dir>/config.json`.
    pub fn load(model_dir: &Path) -> Option<Self> {
        let contents = std::fs::read(model_dir.join("config.json")).ok()?;
        Self::parse(&contents)
    }

    /// Accepts only google/embeddinggemma-2. It rejects unsupported geometry
    /// instead of running it with the wrong kernels.
    pub fn is_supported(&self) -> bool {
        let text = &self.text;
        if self.model_type != "embedding_gemma2"
            || self.architecture != "EmbeddingGemma2Model"
            || self.text_model_type != "embedding_gemma2_text"
            || self.dtype != "bfloat16"
            || self.hidden_activation != "gelu_pytorch_tanh"
            || text.vocab_size != 262144
            || text.hidden_size != 512
            || text.hidden_size_per_layer_input != 512
            || text.intermediate_size != 2048
            || text.num_hidden_layers != 24
            || text.num_attention_heads != 4
            || text.embedding_dim != 768
            || text.sliding_window != 512
            || text.layers.len() != 24
        {
            return false;
        }
        text.layers.iter().all(|layer| {
            let full = layer.layer_type == LayerType::FullAttention;
            layer.head_dim == if full { 512 } else { 256 }
                && layer.num_key_value_heads == if full { 1 } else { 2 }
                && layer.rope_theta == if full { 1_000_000.0 } else { 10_000.0 }
        })
    }
}
