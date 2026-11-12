//! `config.json` parsing and the supported-configuration check
//! (`tests/models/embedding_gemma2/config_test.cpp`): no model files needed.

#![allow(clippy::unwrap_used)]

use gufo_embedding_gemma2::config::{LayerType, ModelConfig};

/// config.json of google/embeddinggemma-2 revision 914f7f89 without the
/// vision and audio towers, which this implementation does not read.
const CONFIG: &str = r#"
{
 "architectures": [
  "EmbeddingGemma2Model"
 ],
 "audio_token_id": 258881,
 "boa_token_id": 256000,
 "boi_token_id": 255999,
 "dtype": "bfloat16",
 "eoa_token_index": 258883,
 "eoi_token_id": 258882,
 "image_token_id": 258880,
 "initializer_range": 0.02,
 "model_type": "embedding_gemma2",
 "text_config": {
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 2,
  "dtype": "bfloat16",
  "embedding_dim": 768,
  "eos_token_id": 1,
  "head_dim": 256,
  "hidden_activation": "gelu_pytorch_tanh",
  "hidden_size": 512,
  "hidden_size_per_layer_input": 512,
  "initializer_range": 0.02,
  "intermediate_size": 2048,
  "layer_types": [
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "full_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "full_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "full_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "full_attention"
  ],
  "max_position_embeddings": 262144,
  "model_type": "embedding_gemma2_text",
  "num_attention_heads": 4,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "pad_token_id": 0,
  "per_layer_config": {
   "05": {
    "head_dim": 512,
    "num_key_value_heads": 1
   },
   "11": {
    "head_dim": 512,
    "num_key_value_heads": 1
   },
   "17": {
    "head_dim": 512,
    "num_key_value_heads": 1
   },
   "23": {
    "head_dim": 512,
    "num_key_value_heads": 1
   }
  },
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
   "full_attention": {
    "rope_theta": 1000000.0,
    "rope_type": "default"
   },
   "sliding_attention": {
    "rope_theta": 10000.0,
    "rope_type": "default"
   }
  },
  "sliding_window": 512,
  "vocab_size": 262144
 },
 "transformers_version": "5.18.0.dev0",
 "video_token_id": 258884,
 "vision_soft_tokens_per_image": 280
}
"#;

fn replace(text: &str, from: &str, to: &str) -> String {
    assert!(text.contains(from), "test edit must match the fixture");
    text.replacen(from, to, 1)
}

#[test]
fn the_released_configuration_is_supported() {
    let config =
        ModelConfig::parse(CONFIG.as_bytes()).expect("the released configuration must parse");
    assert!(
        config.is_supported(),
        "the released configuration must be supported"
    );
    let text = &config.text;
    assert_eq!(text.layers.len(), 24, "24 layers");
    assert_eq!(text.sliding_window, 512, "sliding window radius is 512");
    assert_eq!(text.embedding_dim, 768, "embedding dimension is 768");
    for (index, layer) in text.layers.iter().enumerate() {
        let full = index % 6 == 5;
        assert_eq!(
            layer.layer_type == LayerType::FullAttention,
            full,
            "every sixth layer is a full-attention layer"
        );
        assert_eq!(
            layer.head_dim,
            if full { 512 } else { 256 },
            "full-attention layers use a 512-wide head"
        );
        assert_eq!(
            layer.num_key_value_heads,
            if full { 1 } else { 2 },
            "full-attention layers use one key/value head"
        );
        assert_eq!(
            layer.rope_theta,
            if full { 1_000_000.0 } else { 10_000.0 },
            "rope base depends on the layer type"
        );
    }
}

#[test]
fn the_released_vision_geometry_is_supported() {
    let vision_config = replace(
        CONFIG,
        "\"vision_soft_tokens_per_image\": 280",
        r#""vision_soft_tokens_per_image": 280,
         "vision_config": {
           "model_type": "gemma4_vision", "dtype": "bfloat16",
           "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 768,
           "intermediate_size": 3072, "num_hidden_layers": 16,
           "num_attention_heads": 12, "num_key_value_heads": 12,
           "head_dim": 64, "patch_size": 16, "pooling_kernel_size": 3,
           "position_embedding_size": 10240, "standardize": false,
           "use_clipped_linears": false, "attention_bias": false,
           "rms_norm_eps": 1e-6,
           "rope_parameters": {"rope_type": "axial", "rope_theta": 100.0}
         }"#,
    );
    let vision = ModelConfig::parse(vision_config.as_bytes()).unwrap();
    assert!(
        vision.supported_vision,
        "released vision geometry is supported"
    );
    assert!(
        vision.supported_video,
        "released video placeholder is supported"
    );
    let wrong_video = ModelConfig::parse(
        replace(
            &vision_config,
            "\"video_token_id\": 258884",
            "\"video_token_id\": 42",
        )
        .as_bytes(),
    )
    .unwrap();
    assert!(
        wrong_video.supported_vision && !wrong_video.supported_video,
        "unsupported video token preserves image support"
    );
    for changed in [
        replace(&vision_config, "\"head_dim\": 64", "\"head_dim\": 128"),
        replace(
            &vision_config,
            "\"pooling_kernel_size\": 3",
            "\"pooling_kernel_size\": 2",
        ),
        replace(
            &vision_config,
            "\"standardize\": false",
            "\"standardize\": true",
        ),
        replace(
            &vision_config,
            "\"image_token_id\": 258880",
            "\"image_token_id\": 42",
        ),
    ] {
        let unsupported = ModelConfig::parse(changed.as_bytes()).unwrap();
        assert!(
            !unsupported.supported_vision && unsupported.is_supported(),
            "unsupported vision refuses images while retaining text support"
        );
    }
}

#[test]
fn unsupported_or_inconsistent_configurations_are_refused() {
    assert!(
        ModelConfig::parse(b"{}").is_none(),
        "a config without a text backbone is rejected"
    );
    let wrong_width = ModelConfig::parse(
        replace(CONFIG, "\"hidden_size\": 512", "\"hidden_size\": 768").as_bytes(),
    )
    .unwrap();
    assert!(
        !wrong_width.is_supported(),
        "a different width is unsupported"
    );
    let short_layers = ModelConfig::parse(
        replace(
            CONFIG,
            "\"num_hidden_layers\": 24",
            "\"num_hidden_layers\": 23",
        )
        .as_bytes(),
    );
    assert!(
        short_layers.is_none(),
        "layer_types must match num_hidden_layers"
    );
}
