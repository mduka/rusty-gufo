//! The cases of the C++ unit tests (`tests/models/qwen3_tts/config_test.cpp`,
//! `prompt_test.cpp`, `audio_test.cpp`), without a device or checkpoint.

#![allow(clippy::unwrap_used)]

use gufo_qwen3_tts::audio::{AudioBuffer, resample_mono};
use gufo_qwen3_tts::config::{ModelConfig, ModelVariant};
use gufo_qwen3_tts::prompt::{
    IclPromptInput, NonIclPromptInput, build_icl_prompt, build_non_icl_prompt,
};

#[test]
fn configuration_parses_and_qualifies_variants() {
    let config = ModelConfig::parse(
        r#"{
        "model_type": "qwen3_tts",
        "tokenizer_type": "qwen3_tts_tokenizer_12hz",
        "tts_model_size": "1b7",
        "tts_model_type": "custom_voice",
        "im_start_token_id": 151644,
        "im_end_token_id": 151645,
        "tts_pad_token_id": 151671,
        "tts_bos_token_id": 151672,
        "tts_eos_token_id": 151673,
        "talker_config": {
          "hidden_size": 2048,
          "num_hidden_layers": 28,
          "max_position_embeddings": 32768,
          "spk_id": {"vivian": 3065, "eric": 3062},
          "codec_language_id": {"english": 2050},
          "spk_is_dialect": {"vivian": false, "eric": "sichuan_dialect"},
          "rope_scaling": {"mrope_section": [24, 20, 20], "interleaved": true},
          "code_predictor_config": {
            "hidden_size": 1024,
            "num_hidden_layers": 5,
            "max_position_embeddings": 65536
          }
        }
      }"#,
    )
    .expect("valid configuration parses");
    assert!(
        config.model_type == "qwen3_tts"
            && config.tts_model_type == "custom_voice"
            && config.variant == ModelVariant::CustomVoice,
        "model identity is preserved"
    );
    assert_eq!(config.talker.spk_id["vivian"], 3065, "speaker token");
    assert_eq!(
        config.talker.spk_dialect["vivian"], None,
        "non-dialect speaker"
    );
    assert_eq!(
        config.talker.spk_dialect["eric"].as_deref(),
        Some("sichuan_dialect"),
        "dialect name"
    );
    assert_eq!(config.talker.mrope_section, [24, 20, 20], "mRoPE sections");
    assert_eq!(
        config.code_predictor.max_position_embeddings, 65536,
        "code predictor dimensions are independent"
    );
    assert!(
        ModelConfig::parse("{").is_none(),
        "malformed JSON fails closed"
    );

    let parse = |model_size: &str, tokenizer: &str, model_type: &str, extra: &str| {
        ModelConfig::parse(&format!(
            r#"{{"model_type": "qwen3_tts", "tokenizer_type": "{tokenizer}",
                "tts_model_size": "{model_size}", "tts_model_type": "{model_type}"{extra}}}"#
        ))
        .unwrap()
    };
    let voice_design = parse("1b7", "qwen3_tts_tokenizer_12hz", "voice_design", "");
    assert!(voice_design.variant == ModelVariant::VoiceDesign && voice_design.is_supported());
    let base = parse(
        "1b7",
        "qwen3_tts_tokenizer_12hz",
        "base",
        r#", "speaker_encoder_config": {"enc_dim": 2048, "sample_rate": 24000}"#,
    );
    assert!(
        base.variant == ModelVariant::Base
            && base
                .speaker_encoder
                .is_some_and(|s| s.embedding_dim == 2048)
            && base.is_supported(),
        "1.7B Base speaker encoder is parsed and supported"
    );
    let no_speaker = parse("1b7", "qwen3_tts_tokenizer_12hz", "base", "");
    assert!(!no_speaker.is_supported(), "Base needs its speaker encoder");
    assert!(
        !parse("0b6", "qwen3_tts_tokenizer_12hz", "voice_design", "").is_supported(),
        "0.6B variants stay excluded"
    );
    assert!(
        !parse("1b7", "qwen3_tts_tokenizer_25hz", "voice_design", "").is_supported(),
        "unqualified 25 Hz variants stay excluded"
    );
}

#[test]
fn prompts_compose_their_rows() {
    let instruction = [1.0, 2.0];
    let role = [3.0, 4.0];
    // Language, codec pad, codec BOS.
    let codec = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let text = [5.0, 6.0];
    let tts_bos = [100.0, 200.0];
    let tts_eos = [300.0, 400.0];
    let tts_pad = [7.0, 8.0];
    let codec_pad = [9.0, 10.0];
    let non_icl = build_non_icl_prompt(&NonIclPromptInput {
        hidden_size: 2,
        instruction: &instruction,
        role: &role,
        codec: &codec,
        text: &text,
        tts_bos: &tts_bos,
        tts_eos: &tts_eos,
        tts_pad: &tts_pad,
        codec_pad: &codec_pad,
    })
    .unwrap();
    assert_eq!(non_icl.tokens, 7, "non-ICL prompt row count");
    assert_eq!(non_icl.trailing_text, tts_pad, "non-ICL trailing text");
    assert_eq!(
        non_icl.embeddings,
        [
            1.0, 2.0, 3.0, 4.0, 17.0, 28.0, 130.0, 240.0, 14.0, 16.0, 308.0, 410.0, 57.0, 68.0
        ],
        "non-ICL row composition"
    );

    let icl = build_icl_prompt(&IclPromptInput {
        hidden_size: 2,
        role: &role,
        codec: &codec,
        combined_text: &[1.0, 2.0, 3.0, 4.0],
        reference_codec: &[11.0, 12.0, 13.0, 14.0],
        tts_bos: &tts_bos,
        tts_eos: &tts_eos,
        tts_pad: &tts_pad,
    })
    .unwrap();
    assert_eq!(icl.tokens, 5, "ICL prompt row count");
    assert_eq!(icl.trailing_text, tts_eos, "ICL keeps unmatched text rows");
    assert_eq!(
        icl.embeddings,
        [3.0, 4.0, 17.0, 28.0, 130.0, 240.0, 12.0, 14.0, 16.0, 18.0],
        "ICL row composition"
    );
}

fn sine(sample_rate: u32, frequency: f64, frames: usize) -> AudioBuffer {
    AudioBuffer {
        sample_rate,
        channels: 1,
        samples: (0..frames)
            .map(|frame| {
                (2.0 * std::f64::consts::PI * frequency * frame as f64 / f64::from(sample_rate))
                    .sin() as f32
            })
            .collect(),
    }
}

fn rms(samples: &[f32]) -> f64 {
    (samples
        .iter()
        .map(|&s| f64::from(s) * f64::from(s))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt()
}

#[test]
fn resampling_mixes_mono_and_filters_aliases() {
    let stereo = AudioBuffer {
        sample_rate: 24000,
        channels: 2,
        samples: vec![1.0, -1.0, 0.5, 0.25],
    };
    assert_eq!(
        resample_mono(&stereo, 24000).unwrap(),
        [0.0, 0.375],
        "same-rate stereo input is mixed without resampling drift"
    );
    let trim = 64;
    let passband = resample_mono(&sine(48000, 3000.0, 48000), 24000).unwrap();
    let stopband = resample_mono(&sine(48000, 18000.0, 48000), 24000).unwrap();
    assert_eq!(passband.len(), 24000, "downsample frame count is exact");
    let pass = rms(&passband[trim..passband.len() - trim]);
    assert!(
        pass > 0.68 && pass < 0.73,
        "speech-band energy is preserved: {pass}"
    );
    let stop = rms(&stopband[trim..stopband.len() - trim]);
    assert!(stop < 0.01, "above-Nyquist energy is filtered: {stop}");
}

#[test]
fn a_missing_checkpoint_reports_the_tokenizer_error() {
    let error = gufo_qwen3_tts::synthesis::Synthesizer::new(
        std::path::Path::new("/nonexistent/qwen3-tts"),
        4096,
        ModelVariant::CustomVoice,
    )
    .err()
    .expect("missing model");
    assert_eq!(
        error,
        "Qwen3-TTS tokenizer load failed: cannot open /nonexistent/qwen3-tts/vocab.json"
    );
}
