//! The configuration cases of the C++ extension's `config_test.cpp`.

#![allow(clippy::unwrap_used)]

mod config_samples;

use config_samples::{CODEC_CONFIG, MODEL_CONFIG, PREPROCESSOR_CONFIG, replace};
use gufo_omnivoice::config::{is_supported, parse_config};

fn parse(
    model: &str,
    codec: &str,
    preprocessor: &str,
) -> Result<gufo_omnivoice::config::Config, String> {
    parse_config(model.as_bytes(), codec.as_bytes(), preprocessor.as_bytes())
}

#[test]
fn released_checkpoint() {
    let config = parse(MODEL_CONFIG, CODEC_CONFIG, PREPROCESSOR_CONFIG).unwrap();
    let backbone = &config.backbone;
    assert_eq!(
        (
            backbone.vocab_size,
            backbone.hidden_size,
            backbone.intermediate_size,
            backbone.layers,
            backbone.heads,
            backbone.kv_heads,
            backbone.head_dim,
            backbone.max_positions
        ),
        (151676, 1024, 3072, 28, 16, 8, 128, 40960),
        "backbone shape"
    );
    assert!(
        backbone.rms_norm_eps == 1e-6 && backbone.rope_theta == 1e6,
        "backbone norm epsilon and RoPE base"
    );
    assert_eq!(
        (
            config.audio_codebooks,
            config.audio_vocab_size,
            config.audio_mask_id
        ),
        (8, 1025, 1024),
        "audio vocabulary"
    );
    let codec = &config.codec;
    assert_eq!(
        (codec.sample_rate, codec.frame_samples, codec.frame_rate()),
        (24000, 960, 25),
        "codec frame rate"
    );
    assert_eq!(
        (codec.codebooks, codec.codebook_size, codec.codebook_dim),
        (8, 1024, 64),
        "codec codebooks"
    );
    let semantic = &codec.semantic;
    assert!(
        codec.semantic_stride == 2
            && semantic.sample_rate == 16000
            && semantic.frame_samples == 320
            && semantic.layers == 12
            && semantic.heads == 12
            && semantic.position_kernel == 128
            && semantic.position_groups == 16
            && semantic.conv_strides.len() == 7,
        "semantic encoder"
    );
    let acoustic = &codec.acoustic;
    assert!(
        acoustic.encoder_channels == 64
            && acoustic.decoder_channels == 1024
            && acoustic.latent_channels == 256
            && acoustic.upsampling_ratios == [8, 5, 4, 2, 3],
        "acoustic codec"
    );
    is_supported(&config).unwrap();
}

fn expect_rejected(name: &str, model: &str, codec: &str, preprocessor: &str, reason: &str) {
    let error = parse(model, codec, preprocessor).expect_err(name);
    assert!(
        error.contains(reason),
        "{name} failed for another reason: {error}"
    );
}

#[test]
fn inconsistent_configurations() {
    let (model, codec, preprocessor) = (MODEL_CONFIG, CODEC_CONFIG, PREPROCESSOR_CONFIG);
    expect_rejected(
        "another model type",
        &replace(
            model,
            r#""model_type": "omnivoice""#,
            r#""model_type": "qwen3_tts""#,
        ),
        codec,
        preprocessor,
        "model_type",
    );
    expect_rejected(
        "a codebook count mismatch",
        &replace(
            model,
            r#""num_audio_codebook": 8"#,
            r#""num_audio_codebook": 9"#,
        ),
        codec,
        preprocessor,
        "num_audio_codebook",
    );
    expect_rejected(
        "a mask id inside the codebook",
        &replace(model, r#""audio_mask_id": 1024"#, r#""audio_mask_id": 0"#),
        codec,
        preprocessor,
        "mask",
    );
    expect_rejected(
        "a missing RoPE base",
        &replace(model, r#""rope_theta": 1000000, "#, ""),
        codec,
        preprocessor,
        "rope_theta",
    );
    expect_rejected(
        "grouped heads that don't divide",
        &replace(
            model,
            r#""num_key_value_heads": 8"#,
            r#""num_key_value_heads": 6"#,
        ),
        codec,
        preprocessor,
        "KV heads",
    );
    expect_rejected(
        "a hop the strides don't reach",
        model,
        &replace(codec, r#""hop_length": 960"#, r#""hop_length": 480"#),
        preprocessor,
        "hop_length",
    );
    expect_rejected(
        "a layer-norm HuBERT variant",
        model,
        &replace(
            codec,
            r#""feat_extract_norm": "group""#,
            r#""feat_extract_norm": "layer""#,
        ),
        preprocessor,
        "feat_extract_norm",
    );
    expect_rejected(
        "a bandwidth between codebook counts",
        model,
        &replace(codec, "[0.5, 1, 1.5, 2]", "[0.5, 1, 1.5, 2.1]"),
        preprocessor,
        "fractional",
    );
    expect_rejected("malformed JSON", model, codec, "{", "preprocessor_config");
}

#[test]
fn unsupported_shapes() {
    let config = parse(
        &replace(MODEL_CONFIG, r#""head_dim": 128"#, r#""head_dim": 64"#),
        CODEC_CONFIG,
        PREPROCESSOR_CONFIG,
    )
    .unwrap();
    let error = is_supported(&config).unwrap_err();
    assert!(error.contains("head size"), "{error}");
}
