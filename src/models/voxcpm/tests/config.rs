//! The C++ extension's `config_test.cpp`: the released configuration, the
//! defaults of absent keys and the rejected variants.

#![allow(clippy::unwrap_used)]

mod config_samples;

use config_samples::{MODEL_CONFIG, replace};
use gufo_voxcpm::config::{is_supported, parse_config};

#[test]
fn released_config() {
    let config = parse_config(MODEL_CONFIG).unwrap();
    let lm = &config.lm;
    assert!(
        lm.hidden == 2048
            && lm.intermediate == 6144
            && lm.layers == 28
            && lm.heads == 16
            && lm.kv_heads == 2
            && lm.head_dim == 128
            && lm.vocab == 73448
            && lm.rope
            && !lm.use_mup,
        "text LM shape"
    );
    assert!((lm.rms_eps - 1e-5).abs() < 1e-12, "RMS epsilon");
    let residual = &config.residual_lm;
    assert!(
        residual.hidden == 2048 && residual.layers == 8 && residual.vocab == 0 && !residual.rope,
        "residual LM shape"
    );
    for stack in [&config.encoder, &config.dit] {
        assert!(
            stack.hidden == 1024
                && stack.intermediate == 4096
                && stack.heads == 16
                && stack.kv_heads == 2
                && stack.head_dim == 128
                && stack.layers == 12
                && stack.vocab == 0
                && stack.rope,
            "encoder and DiT shapes"
        );
    }
    assert!(
        config.rope.dim == 128
            && config.rope.factors().len() == 64
            && std::ptr::eq(config.rope.factors(), config.rope.short_factor.as_slice())
            && (config.rope.scaling() - 1.0).abs() < 1e-12,
        "LongRoPE uses the short factors unscaled"
    );
    assert!(
        config.patch_size == 4
            && config.feat_dim == 64
            && config.fsq_latent_dim == 512
            && config.fsq_scale == 9.0
            && config.max_length == 8192,
        "patching, FSQ and cache length"
    );
    let vae = &config.vae;
    assert!(
        vae.hop() == 640
            && vae.decode_hop() == 1920
            && vae.sample_rate == 16000
            && vae.out_sample_rate == 48000
            && vae.decoder_dim == 2048
            && vae.rate_bucket() == 3,
        "AudioVAE rates"
    );
    is_supported(&config).unwrap();
}

#[test]
fn defaults_and_errors() {
    // Without audio_vae_config, AudioVAEConfigV2's defaults apply.
    let start = MODEL_CONFIG.find("\"audio_vae_config\"").unwrap();
    let end = start + MODEL_CONFIG[start..].find('}').unwrap();
    let mut text = MODEL_CONFIG.to_owned();
    text.replace_range(start..end + 2, "");
    let config = parse_config(&text).unwrap();
    assert!(
        config.vae.decoder_rates.len() == 6 && config.vae.rate_bucket() == 3,
        "AudioVAE defaults"
    );
    assert!(
        parse_config(&replace(MODEL_CONFIG, "\"voxcpm2\"", "\"voxcpm\"")).is_err(),
        "other architectures are rejected"
    );
    assert!(
        parse_config(&replace(MODEL_CONFIG, "\"lm_config\"", "\"lm\"")).is_err(),
        "a missing lm_config is rejected"
    );
    assert!(
        parse_config(&replace(
            MODEL_CONFIG,
            "\"dit_config\": {",
            "\"dit_config\": {\"dit_mean_mode\": true,"
        ))
        .is_err(),
        "mean-mode DiTs are rejected"
    );
    assert!(
        parse_config(&replace(
            MODEL_CONFIG,
            "\"audio_vae_config\": {",
            "\"audio_vae_config\": {\"use_noise_block\": true,"
        ))
        .is_err(),
        "noise blocks are rejected"
    );
    let other_heads = parse_config(&replace(
        MODEL_CONFIG,
        "\"kv_channels\": 128",
        "\"kv_channels\": 64",
    ))
    .unwrap();
    assert!(
        is_supported(&other_heads).is_err(),
        "head sizes other than 128 are unsupported"
    );
}
