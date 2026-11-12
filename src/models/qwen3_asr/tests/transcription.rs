//! End-to-end transcription on the device:
//! cargo test -p gufo-qwen3-asr --features gpu --test transcription
//!
//! The parity gate: the generated token IDs equal the reference fixture, and every
//! parity boundary is bit-identical to the C++ engine (hashes of its traces).

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use gufo_qwen3_asr::{Audio, Request, Transcriber, audio, prompt};
use serde_json::Value;

/// One loaded model for the tests of this binary, one test at a time, so
/// the cancellation test can count its polls.
fn transcriber() -> std::sync::MutexGuard<'static, Transcriber> {
    static MODEL: OnceLock<Mutex<Transcriber>> = OnceLock::new();
    MODEL
        .get_or_init(|| Mutex::new(Transcriber::load(&common::model_root(), 1024).unwrap()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/qwen3-asr-en.json")).unwrap()
}

fn request(wav: &[u8]) -> Request<'_> {
    Request {
        audio: Audio::Wav(wav),
        context: "",
        language: None,
        max_new_tokens: 256,
    }
}

#[test]
fn transcript_and_token_ids_match_the_reference() {
    let expected = fixture();
    let wav = common::fixture_wav();
    let model = transcriber();
    let result = model.transcribe(&request(&wav), &|| false, None).unwrap();
    let ids: Vec<u32> = expected["generated_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(result.generated_ids, ids);
    assert_eq!(result.text, expected["text"].as_str().unwrap());
    assert_eq!(result.language, expected["language"].as_str().unwrap());
    for (field, value) in [
        ("audio_samples", result.audio_samples),
        ("mel_frames", result.mel_frames),
        ("audio_tokens", result.audio_tokens),
        ("prompt_tokens", result.prompt_tokens),
    ] {
        assert_eq!(value as u64, expected[field].as_u64().unwrap(), "{field}");
    }
    assert_eq!((result.chunks, result.sample_rate), (1, 16_000));

    // A second request on the same state reproduces the first.
    let again = model.transcribe(&request(&wav), &|| false, None).unwrap();
    assert_eq!(again.generated_ids, result.generated_ids);
}

#[test]
fn parity_boundaries_are_bit_identical_to_the_cpp_engine() {
    let wav = common::fixture_wav();
    let waveform = audio::resample_mono_16k(&audio::decode_wav(&wav).unwrap()).unwrap();
    let features = audio::log_mel_features(&waveform).unwrap();
    let model = transcriber();
    let tokens = audio::audio_embedding_token_count(features.frames).unwrap();
    let ids = prompt::build_prompt(model.tokenizer(), "", None, tokens).unwrap();
    let (encoder, decoder) = model.trace(&features, &ids).unwrap();
    assert_eq!(common::fnv_f32(&encoder.frontend), 0x0e1d_ebd9_7885_d2b3);
    assert_eq!(common::fnv_f32(&encoder.layer0), 0x4f95_1983_5b07_917f);
    assert_eq!(common::fnv_f32(&encoder.output), 0x2836_edf2_1ec8_4b48);
    assert_eq!(common::fnv_f32(&decoder.layer0), 0x999d_de79_1e42_a1a3);
    assert_eq!(common::fnv_f32(&decoder.logits), 0xc6bb_f198_5780_34cf);
}

#[test]
fn partial_transcripts_stream_and_cancellation_stops_the_request() {
    let wav = common::fixture_wav();
    let model = transcriber();
    let mut updates = Vec::new();
    let mut record = |text: &str| {
        updates.push(text.to_string());
        true
    };
    let result = model
        .transcribe(&request(&wav), &|| false, Some(&mut record))
        .unwrap();
    assert!(updates.len() > 10, "{updates:?}");
    assert!(
        updates.windows(2).all(|w| w[1].starts_with(&w[0])),
        "{updates:?}"
    );
    assert_eq!(updates.last().unwrap(), &result.text);

    let mut seen = 0;
    let mut stop_after_three = |_: &str| {
        seen += 1;
        seen < 3
    };
    let error = model
        .transcribe(&request(&wav), &|| false, Some(&mut stop_after_three))
        .unwrap_err();
    assert!(error.contains("cancelled"), "{error}");

    // Two admissions (frontend and device) and four phase checks precede
    // the encoder.
    let polls = Cell::new(0);
    let cancel_in_encoder = || {
        polls.set(polls.get() + 1);
        polls.get() > 6
    };
    let error = model
        .transcribe(&request(&wav), &cancel_in_encoder, None)
        .unwrap_err();
    assert!(error.contains("cancelled"), "{error}");

    // The model recovers from both cancellations.
    let recovered = model.transcribe(&request(&wav), &|| false, None).unwrap();
    assert_eq!(recovered.generated_ids, result.generated_ids);
}

#[test]
fn a_forced_language_and_context_change_the_prompt_only() {
    let wav = common::fixture_wav();
    let model = transcriber();
    let forced = Request {
        language: Some("en"),
        context: "Podcast about music.",
        ..request(&wav)
    };
    let result = model.transcribe(&forced, &|| false, None).unwrap();
    assert_eq!(result.language, "English");
    assert!(result.text.starts_with("Uh huh."), "{}", result.text);
    assert!(result.prompt_tokens > 211);
    let error = model
        .transcribe(
            &Request {
                language: Some("xx"),
                ..request(&wav)
            },
            &|| false,
            None,
        )
        .unwrap_err();
    assert!(error.contains("unsupported"), "{error}");
}

#[test]
fn long_audio_is_split_into_chunks_within_the_context() {
    let wav = common::fixture_wav();
    let waveform = audio::resample_mono_16k(&audio::decode_wav(&wav).unwrap()).unwrap();
    // Four copies (60 s) with a 256-token capacity need several chunks.
    let long: Vec<f32> = waveform
        .iter()
        .copied()
        .cycle()
        .take(waveform.len() * 4)
        .collect();
    let small = Request {
        audio: Audio::Pcm16k(&long),
        max_new_tokens: 64,
        ..request(&wav)
    };
    let model = Transcriber::load(&common::model_root(), 256).unwrap();
    let result = model.transcribe(&small, &|| false, None).unwrap();
    assert!(result.chunks > 1, "{} chunks", result.chunks);
    assert_eq!(result.audio_samples, long.len());
    assert!(result.text.contains("Oh yeah"), "{}", result.text);

    // A shorter request that arrives while the long one runs takes the
    // device between its chunks and finishes first, and both transcripts
    // equal the serial ones.
    let short = Request {
        max_new_tokens: 64,
        ..request(&wav)
    };
    let alone = model.transcribe(&short, &|| false, None).unwrap();
    assert!(alone.chunks < result.chunks);
    let started = AtomicBool::new(false);
    let (long_result, short_result) = std::thread::scope(|scope| {
        let long = scope.spawn(|| {
            let mut first = |_: &str| {
                started.store(true, Ordering::Release);
                true
            };
            let transcript = model.transcribe(&small, &|| false, Some(&mut first));
            (transcript, Instant::now())
        });
        while !started.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        let transcript = model.transcribe(&short, &|| false, None);
        let short_done = Instant::now();
        let (transcript_long, long_done) = long.join().unwrap();
        assert!(short_done < long_done);
        (transcript_long.unwrap(), transcript.unwrap())
    });
    assert_eq!(long_result.generated_ids, result.generated_ids);
    assert_eq!(short_result.generated_ids, alone.generated_ids);
    assert_eq!(short_result.text, alone.text);
}
