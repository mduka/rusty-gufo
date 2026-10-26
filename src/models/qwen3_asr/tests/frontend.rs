//! The CPU frontend against the C++ engine on the fixture:
//! cargo test -p gufo-qwen3-asr --features gpu --test frontend
//!
//! Hashes come from the C++ baseline (ROCm 10.0.0) run on
//! the same WAV and checkpoint; equal hashes mean bit-identical stages.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use gufo_qwen3_asr::{audio, prompt, tokenizer::Tokenizer};

#[test]
fn stages_are_bit_identical_to_the_cpp_frontend() {
    let decoded = audio::decode_wav(&common::fixture_wav()).unwrap();
    assert_eq!((decoded.sample_rate, decoded.channels), (48_000, 1));
    let waveform = audio::resample_mono_16k(&decoded).unwrap();
    assert_eq!(waveform.len(), 240_820);
    assert_eq!(common::fnv_f32(&waveform), 0xa2a9_fc0f_3367_b113);
    let features = audio::log_mel_features(&waveform).unwrap();
    assert_eq!(features.frames, 1505);
    assert_eq!(common::fnv_f32(&features.values), 0x0d8f_ce1a_a831_edcb);

    let tokenizer = Tokenizer::load(&common::model_root()).unwrap();
    let tokens = audio::audio_embedding_token_count(features.frames).unwrap();
    assert_eq!(tokens, 196);
    let ids = prompt::build_prompt(&tokenizer, "", None, tokens).unwrap();
    assert_eq!(ids.len(), 211);
    assert_eq!(common::fnv_u32(&ids), 0x5760_cff2_3061_4bac);
}

#[test]
fn encoding_matches_the_cpp_tokenizer() {
    let tokenizer = Tokenizer::load(&common::model_root()).unwrap();
    let ids = tokenizer
        .encode("Hello, world! Grüße <|im_start|>x<asr_text>  y\n")
        .unwrap();
    assert_eq!(
        ids,
        [
            9707, 11, 1879, 0, 94063, 23455, 220, 151_644, 87, 151_704, 256, 88, 198
        ]
    );
    let text = tokenizer.decode(&ids, true);
    assert_eq!(text, "Hello, world! Grüße x<asr_text>  y\n".as_bytes());
    let forced = prompt::build_prompt(&tokenizer, "Names: Qwen", Some("English"), 2).unwrap();
    let text = tokenizer.decode(&forced, false);
    assert!(text.starts_with(b"<|im_start|>system\nNames: Qwen<|im_end|>"));
    assert!(text.ends_with(b"<|im_start|>assistant\nlanguage English<asr_text>"));
}
