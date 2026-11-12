//! The fields this crate reads from k2-fsa/OmniVoice's three configuration
//! files, with the released checkpoint's values (`config_samples.hpp`).

#![allow(dead_code)]

use std::path::PathBuf;

pub const MODEL_CONFIG: &str = r#"{
  "architectures": ["OmniVoice"],
  "model_type": "omnivoice",
  "audio_mask_id": 1024,
  "audio_vocab_size": 1025,
  "num_audio_codebook": 8,
  "llm_config": {
    "model_type": "qwen3",
    "attention_bias": false,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "max_position_embeddings": 40960,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
    "vocab_size": 151676
  }
}"#;

pub const CODEC_CONFIG: &str = r#"{
  "model_type": "higgs_audio_v2_tokenizer",
  "codebook_dim": 64,
  "codebook_size": 1024,
  "downsample_factor": 320,
  "sample_rate": 24000,
  "semantic_sample_rate": 16000,
  "target_bandwidths": [0.5, 1, 1.5, 2],
  "acoustic_model_config": {
    "model_type": "dac",
    "decoder_hidden_size": 1024,
    "downsampling_ratios": [8, 5, 4, 2, 3],
    "encoder_hidden_size": 64,
    "hidden_size": 256,
    "hop_length": 960,
    "sampling_rate": 16000,
    "upsampling_ratios": [8, 5, 4, 2, 3]
  },
  "semantic_model_config": {
    "model_type": "hubert",
    "conv_bias": false,
    "conv_dim": [512, 512, 512, 512, 512, 512, 512],
    "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
    "conv_stride": [5, 2, 2, 2, 2, 2, 2],
    "do_stable_layer_norm": false,
    "feat_extract_activation": "gelu",
    "feat_extract_norm": "group",
    "feat_proj_layer_norm": true,
    "hidden_act": "gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-05,
    "num_attention_heads": 12,
    "num_conv_pos_embedding_groups": 16,
    "num_conv_pos_embeddings": 128,
    "num_hidden_layers": 12
  }
}"#;

pub const PREPROCESSOR_CONFIG: &str = r#"{
  "feature_extractor_type": "DacFeatureExtractor",
  "hop_length": 960,
  "sampling_rate": 24000
}"#;

/// `text` with its single occurrence of `from` replaced by `to`.
pub fn replace(text: &str, from: &str, to: &str) -> String {
    assert_eq!(
        text.matches(from).count(),
        1,
        "sample edit must match once: {from}"
    );
    text.replacen(from, to, 1)
}

/// `tests/fixtures/<name>`.
pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// A golden fixture's JSON.
pub fn golden(name: &str) -> gufo_core::json::Value {
    let text = std::fs::read(fixture(name)).unwrap_or_else(|error| panic!("{name}: {error}"));
    gufo_core::json::parse(&text).unwrap_or_else(|error| panic!("{name}: {error}"))
}

/// `GUFO_OMNIVOICE_MODEL`: the model tests fail (rather than skip) without
/// it, since a missing model is not a pass.
pub fn model_root() -> PathBuf {
    std::env::var_os("GUFO_OMNIVOICE_MODEL")
        .filter(|root| !root.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("GUFO_OMNIVOICE_MODEL must name the OmniVoice model directory"))
}

/// A path unique to this process under the temporary directory.
pub fn temp_path(test: &str, name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("omnivoice_{test}_{}_{name}", std::process::id()))
}
