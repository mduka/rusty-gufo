//! `config.json` of the checkpoint. Only Qwen3-ASR-1.7B in BF16 is accepted;
//! the 0.6B and forced-aligner checkpoints are outside this implementation.

use std::path::Path;

use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct AudioEncoderConfig {
    pub num_mel_bins: u32,
    pub d_model: u32,
    pub encoder_layers: u32,
    pub encoder_attention_heads: u32,
    pub encoder_ffn_dim: u32,
    pub max_source_positions: u32,
    pub n_window: u32,
    pub n_window_infer: u32,
    pub conv_chunksize: u32,
    pub downsample_hidden_size: u32,
    pub output_dim: u32,
}

impl Default for AudioEncoderConfig {
    fn default() -> Self {
        Self {
            num_mel_bins: 128,
            d_model: 1024,
            encoder_layers: 24,
            encoder_attention_heads: 16,
            encoder_ffn_dim: 4096,
            max_source_positions: 1500,
            n_window: 50,
            n_window_infer: 800,
            conv_chunksize: 500,
            downsample_hidden_size: 480,
            output_dim: 2048,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextConfig {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub max_position_embeddings: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub mrope_section: Vec<u32>,
    pub mrope_interleaved: bool,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 151_936,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 65_536,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            mrope_section: vec![24, 20, 20],
            mrope_interleaved: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub model_type: String,
    pub architecture: String,
    pub dtype: String,
    pub audio: AudioEncoderConfig,
    pub text: TextConfig,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_token_id: u32,
    pub supported_languages: Vec<String>,
}

impl ModelConfig {
    /// Parses `config.json`; absent numeric fields keep the 1.7B defaults.
    pub fn parse(json: &str) -> Result<Self, String> {
        let root: Value =
            serde_json::from_str(json).map_err(|e| format!("invalid config.json: {e}"))?;
        let root = root.as_object().ok_or("config.json is not an object")?;
        let string =
            |value: Option<&Value>| value.and_then(Value::as_str).unwrap_or("").to_string();
        let thinker = root
            .get("thinker_config")
            .and_then(Value::as_object)
            .ok_or("config.json has no thinker_config")?;
        let audio = thinker
            .get("audio_config")
            .and_then(Value::as_object)
            .ok_or("config.json has no audio_config")?;
        let text = thinker
            .get("text_config")
            .and_then(Value::as_object)
            .ok_or("config.json has no text_config")?;

        let mut config = Self {
            model_type: string(root.get("model_type")),
            architecture: root
                .get("architectures")
                .and_then(Value::as_array)
                .and_then(|a| a.first())
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            dtype: string(thinker.get("dtype")),
            audio: AudioEncoderConfig::default(),
            text: TextConfig::default(),
            audio_start_token_id: 151_669,
            audio_end_token_id: 151_670,
            audio_token_id: 151_676,
            supported_languages: root
                .get("support_languages")
                .and_then(Value::as_array)
                .map(|languages| {
                    languages
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
        };
        let u32_field = |object: &serde_json::Map<String, Value>, name: &str, out: &mut u32| {
            if let Some(value) = object.get(name).and_then(number_u32) {
                *out = value;
            }
        };
        let f32_field = |object: &serde_json::Map<String, Value>, name: &str, out: &mut f32| {
            if let Some(value) = object.get(name).and_then(Value::as_f64) {
                *out = value as f32;
            }
        };
        u32_field(
            thinker,
            "audio_start_token_id",
            &mut config.audio_start_token_id,
        );
        u32_field(
            thinker,
            "audio_end_token_id",
            &mut config.audio_end_token_id,
        );
        u32_field(thinker, "audio_token_id", &mut config.audio_token_id);

        let a = &mut config.audio;
        u32_field(audio, "num_mel_bins", &mut a.num_mel_bins);
        u32_field(audio, "d_model", &mut a.d_model);
        u32_field(audio, "encoder_layers", &mut a.encoder_layers);
        u32_field(
            audio,
            "encoder_attention_heads",
            &mut a.encoder_attention_heads,
        );
        u32_field(audio, "encoder_ffn_dim", &mut a.encoder_ffn_dim);
        u32_field(audio, "max_source_positions", &mut a.max_source_positions);
        u32_field(audio, "n_window", &mut a.n_window);
        u32_field(audio, "n_window_infer", &mut a.n_window_infer);
        u32_field(audio, "conv_chunksize", &mut a.conv_chunksize);
        u32_field(
            audio,
            "downsample_hidden_size",
            &mut a.downsample_hidden_size,
        );
        u32_field(audio, "output_dim", &mut a.output_dim);

        let t = &mut config.text;
        u32_field(text, "vocab_size", &mut t.vocab_size);
        u32_field(text, "hidden_size", &mut t.hidden_size);
        u32_field(text, "intermediate_size", &mut t.intermediate_size);
        u32_field(text, "num_hidden_layers", &mut t.num_hidden_layers);
        u32_field(text, "num_attention_heads", &mut t.num_attention_heads);
        u32_field(text, "num_key_value_heads", &mut t.num_key_value_heads);
        u32_field(text, "head_dim", &mut t.head_dim);
        u32_field(
            text,
            "max_position_embeddings",
            &mut t.max_position_embeddings,
        );
        f32_field(text, "rms_norm_eps", &mut t.rms_norm_eps);
        f32_field(text, "rope_theta", &mut t.rope_theta);
        if let Some(rope) = text.get("rope_scaling").and_then(Value::as_object) {
            let section: Vec<u32> = rope
                .get("mrope_section")
                .and_then(Value::as_array)
                .map(|items| items.iter().filter_map(number_u32).collect())
                .unwrap_or_default();
            if !section.is_empty() {
                t.mrope_section = section;
            }
            if let Some(interleaved) = rope
                .get("mrope_interleaved")
                .and_then(Value::as_bool)
                .or_else(|| rope.get("interleaved").and_then(Value::as_bool))
            {
                t.mrope_interleaved = interleaved;
            }
        }
        Ok(config)
    }

    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let path = model_dir.join("config.json");
        let json = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Self::parse(&json)
    }

    /// Whether this is the Qwen3-ASR-1.7B BF16 checkpoint the engine supports.
    pub fn is_supported(&self) -> bool {
        let (a, t) = (&self.audio, &self.text);
        self.model_type == "qwen3_asr"
            && self.architecture == "Qwen3ASRForConditionalGeneration"
            && self.dtype == "bfloat16"
            && *a == AudioEncoderConfig::default()
            && t.vocab_size == 151_936
            && t.hidden_size == 2048
            && t.intermediate_size == 6144
            && t.num_hidden_layers == 28
            && t.num_attention_heads == 16
            && t.num_key_value_heads == 8
            && t.head_dim == 128
            && t.max_position_embeddings == 65_536
            && t.mrope_section == [24, 20, 20]
            && t.mrope_interleaved
            && self.audio_start_token_id == 151_669
            && self.audio_end_token_id == 151_670
            && self.audio_token_id == 151_676
    }
}

/// A JSON number as `u32`, truncated like the C++ `static_cast`.
fn number_u32(value: &Value) -> Option<u32> {
    value
        .as_u64()
        .or_else(|| value.as_f64().filter(|v| *v >= 0.0).map(|v| v as u64))
        .map(|v| v as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"{
        "model_type": "qwen3_asr",
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "support_languages": ["Chinese", "English"],
        "thinker_config": {
            "dtype": "bfloat16",
            "audio_config": {},
            "text_config": {"rope_scaling": {"mrope_section": [24, 20, 20], "interleaved": true}}
        }
    }"#;

    #[test]
    fn defaults_describe_the_supported_checkpoint() {
        let config = ModelConfig::parse(MINIMAL).unwrap();
        assert!(config.is_supported());
        assert_eq!(config.supported_languages, ["Chinese", "English"]);
        assert_eq!(config.text.rms_norm_eps, 1.0e-6);
    }

    #[test]
    fn other_geometries_and_types_are_rejected() {
        let small = MINIMAL.replace(
            r#""audio_config": {}"#,
            r#""audio_config": {"d_model": 896}"#,
        );
        assert!(!ModelConfig::parse(&small).unwrap().is_supported());
        let half = MINIMAL.replace("bfloat16", "float16");
        assert!(!ModelConfig::parse(&half).unwrap().is_supported());
        assert!(ModelConfig::parse(r#"{"model_type": "qwen3_asr"}"#).is_err());
        assert!(ModelConfig::parse("[]").is_err());
    }
}
