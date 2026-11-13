//! The C++ extension's `host_test.cpp`: configuration, language prompts and
//! transcript cleaning; with `--features models`, the probe and
//! configuration of `GUFO_NEMOTRON_ASR_MODEL`. (Its resampler checks have no
//! counterpart: the resampler serves only the C++ live sessions.)

#![allow(clippy::unwrap_used)]

mod support;

use gufo_nemotron_asr::config::{
    Config, chunk_frames, first_chunk_frames, is_supported, parse_config,
};
use gufo_nemotron_asr::language::{language_name, resolve_language, tag_locale};
use gufo_nemotron_asr::tokenizer::clean_text;

const CONFIG: &str = r#"{
  "architectures": ["Nemotron3_5AsrForRNNT"],
  "blank_token_id": 13087, "decoder_hidden_size": 640, "durations": [],
  "encoder_config": {
    "attention_bias": false, "conv_kernel_size": 9, "convolution_bias": false,
    "default_num_lookahead_tokens": 3, "hidden_act": "silu", "hidden_size": 1024,
    "intermediate_size": 4096, "max_position_embeddings": 5000,
    "model_type": "nemotron_asr_streaming_encoder", "num_attention_heads": 8,
    "num_hidden_layers": 24, "num_key_value_heads": 8, "num_mel_bins": 128,
    "scale_input": false, "sliding_window": 57, "subsampling_conv_channels": 256,
    "subsampling_conv_kernel_size": 3, "subsampling_conv_stride": 2,
    "subsampling_factor": 8, "supported_num_lookahead_tokens": [3, 0, 6, 13]
  },
  "hidden_act": "relu", "max_symbols_per_step": 10, "model_type": "nemotron3_5_asr",
  "num_decoder_layers": 2, "num_prompts": 128, "prompt_intermediate_size": 2048,
  "vocab_size": 13088, "default_prompt_id": 101
}"#;

const PROCESSOR: &str = r#"{
  "default_num_lookahead_tokens": 3,
  "feature_extractor": {"feature_size": 128, "hop_length": 160, "n_fft": 512,
    "preemphasis": 0.97, "sampling_rate": 16000, "win_length": 400},
  "num_prompts": 128,
  "prompt_dictionary": {"auto": 101, "de": 9, "de-DE": 9, "en": 0, "en-US": 0,
    "en-GB": 1, "ja-JA": 10, "ja-JP": 10, "zh-CN": 4, "zh-ZH": 4, "zh-TW": 5,
    "nb": 103, "nb-NO": 103, "no": 27, "no-NO": 27}
}"#;

#[test]
fn sample_configuration() {
    let config = parse_config(CONFIG, PROCESSOR).unwrap();
    is_supported(&config).unwrap();
    assert_eq!(config.encoder.head_dim(), 128, "head dimension");
    assert_eq!(config.encoder.left_context(), 56, "left context");
    assert_eq!(config.encoder.default_lookahead, 3, "default lookahead");
    assert_eq!(first_chunk_frames(&config, 13), 105, "first chunk frames");
    assert_eq!(chunk_frames(&config, 13), 112, "chunk frames");
    assert_eq!(
        first_chunk_frames(&config, 0),
        1,
        "first chunk, lookahead 0"
    );
    assert!(parse_config("{}", PROCESSOR).is_err(), "empty config fails");

    // The kernels' limits.
    let refuses = |changed: &Config, reason: &str| {
        is_supported(changed).is_err_and(|failure| failure.contains(reason))
    };
    let mut changed = config.clone();
    changed.encoder.subsampling_channels = 48;
    assert!(
        refuses(&changed, "multiple of 32"),
        "subsampling channels % 32"
    );
    let mut changed = config.clone();
    changed.prompt_hidden = 1000;
    assert!(
        refuses(&changed, "prompt projector"),
        "prompt projector width % 32"
    );
    let mut changed = config.clone();
    changed.encoder.sliding_window = 253; // left 252: whole chunks of 1 to 14
    assert!(
        refuses(&changed, "attend to at most"),
        "attention window limit"
    );
    let mut changed = config.clone();
    changed.encoder.lookaheads.push(27); // chunks of 28 frames
    if let Err(error) = is_supported(&changed) {
        panic!("chunks beyond 16 frames are supported: {error}");
    }
    assert!(parse_config(CONFIG, "{").is_err(), "bad JSON fails");
}

#[test]
fn language_prompts() {
    let config = parse_config(CONFIG, PROCESSOR).unwrap();
    let prompt = |language: &str| resolve_language(&config, Some(language)).ok();
    assert_eq!(prompt("de"), Some(9), "de");
    assert_eq!(prompt("DE_de"), Some(9), "DE_de");
    assert_eq!(prompt("German"), Some(9), "German");
    assert_eq!(prompt("german"), Some(9), "german");
    assert_eq!(prompt("ja"), Some(10), "ja (shared prompt)");
    assert_eq!(prompt("zh"), Some(4), "zh (its country's locale)");
    assert_eq!(prompt("ZH-tw"), Some(5), "zh-TW");
    assert_eq!(prompt("Norwegian"), Some(103), "Norwegian");
    assert_eq!(prompt("auto"), Some(101), "auto");
    assert_eq!(prompt(""), Some(101), "empty");
    assert_eq!(prompt("xx"), None, "unknown language");
    assert_eq!(
        resolve_language(&config, Some("xx")).unwrap_err(),
        "Nemotron ASR does not support language \"xx\"; use a code such as \"de\", a locale \
         such as \"de-DE\", an English name or \"auto\""
    );
    assert_eq!(resolve_language(&config, None), Ok(101), "no language");
    assert_eq!(language_name("de-DE"), "German", "German name");
    assert_eq!(language_name("xx-YY"), "xx-YY", "unnamed locale");
    assert_eq!(tag_locale("<de-DE>").as_deref(), Some("de-DE"), "tag");
    assert_eq!(tag_locale("<unk>"), None, "unk is no tag");
    assert_eq!(tag_locale("<pad>"), None, "pad is no tag");
}

#[test]
fn transcript_cleaning() {
    assert_eq!(clean_text(" Heute ist.  Wir "), "Heute ist. Wir", "clean");
    assert_eq!(clean_text(""), "", "clean empty");
    assert_eq!(clean_text(" \n "), "", "clean spaces");
    // Appending to the raw text only appends to its transcript.
    let raw = " A b.  ";
    let longer = format!("{raw}C");
    assert!(clean_text(&longer).starts_with(&clean_text(raw)), "prefix");
}

#[cfg(feature = "models")]
#[test]
fn released_model() {
    use gufo_nemotron_asr::config::load_config;
    use gufo_nemotron_asr::probe::{probe, resolve_artifacts};

    let root = support::model_root();
    let config = load_config(&root).unwrap();
    is_supported(&config).unwrap();
    assert_eq!(config.encoder.default_lookahead, 3, "default lookahead 3");
    assert_eq!(config.encoder.lookaheads.len(), 4, "four lookaheads");
    assert_eq!(resolve_language(&config, Some("de")), Ok(9), "de prompt");
    assert_eq!(
        resolve_language(&config, Some("zh")),
        Ok(4),
        "zh takes zh-ZH, which shares zh-CN's prompt"
    );
    assert!(probe(&root), "probe accepts the model");
    resolve_artifacts(&root).unwrap();
    let other = std::env::temp_dir().join(format!("nemotron_asr_probe_{}", std::process::id()));
    std::fs::create_dir_all(&other).unwrap();
    std::fs::write(other.join("config.json"), r#"{"model_type": "parakeet"}"#).unwrap();
    std::fs::write(other.join("processor_config.json"), "{}").unwrap();
    let rejected = !probe(&other);
    std::fs::remove_dir_all(&other).unwrap();
    assert!(rejected, "probe rejects other models");
}
