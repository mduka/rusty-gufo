//! Image, audio and video embeddings on the GPU against the reference
//! model's vectors (`image_golden_test.cpp`, `audio_golden_hip_test.cpp`,
//! the video cases of `tests/functional/embedding_video.py`). The inputs are
//! synthetic images, 16-kHz waveforms and the encoded video fixtures, whose
//! frames the host tests check bit for bit; the reference vectors come from
//! transformers:
//! cargo test --profile gpu-test -p gufo-embedding-gemma2 --features gpu,models --test media

#![cfg(all(feature = "gpu", feature = "models"))]
#![allow(clippy::unwrap_used)]

mod common;

use gufo_core::json::{self, Value};
use gufo_embedding_gemma2::audio_decoder::decode_audio;
use gufo_embedding_gemma2::audio_processor::{SequenceAudio, process_audio};
use gufo_embedding_gemma2::config::{AUDIO_TOKEN_ID, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID};
use gufo_embedding_gemma2::image_processor::{DEFAULT_IMAGE_SOFT_TOKENS, process_image};
use gufo_embedding_gemma2::video_processor::{VideoOptions, process_video};
use gufo_embedding_gemma2::{EncoderRuntime, EncoderSequence, SequenceImage};

fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    json::parse(&std::fs::read(path).unwrap()).unwrap()
}

/// The waveforms of `audio_golden_hip_test.cpp`; "speech" is the
/// functional suite's 16-kHz PCM recording.
fn waveform(count: usize, kind: &str) -> Vec<f32> {
    if kind == "speech" {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../tests/functional/fixtures/audio/clear-english-voice.wav"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("speech fixture {path}: {e}"));
        return decode_audio(&bytes, None).unwrap();
    }
    (0..count)
        .map(|i| match kind {
            "triangle" => {
                let phase = (i % 80) as i32;
                (phase.min(80 - phase) - 20) as f32 / 64.0
            }
            "music" => {
                let period = if (i / 2000) % 2 == 0 { 64 } else { 80 };
                let phase = (i % period) as i32;
                (phase.min(period as i32 - phase) - period as i32 / 4) as f32 / 64.0
            }
            "environment" if i % 4000 < 800 => {
                ((((i * 73 + i * i * 19) % 1024) as i32) - 512) as f32 / 1024.0
            }
            "noise" => ((((i * 73 + i * i * 19) % 1024) as i32) - 512) as f32 / 1024.0,
            _ => 0.0,
        })
        .collect()
}

/// The sequence of a fixture case: its token IDs with the features of its
/// clips and images at their placeholders.
fn sequence(case: &Value) -> EncoderSequence {
    let tokens: Vec<u32> = case
        .find("token_ids")
        .unwrap()
        .items()
        .iter()
        .map(|id| id.as_size(0) as u32)
        .collect();
    let mut sequence = EncoderSequence::text(tokens);
    let mut cursor = 0;
    for clip in case.find("clips").map_or(&[][..], Value::items) {
        let features = process_audio(
            &waveform(clip.member_size("samples", 0), &clip.member_str("kind", "")),
            0,
            None,
        )
        .unwrap();
        while cursor < sequence.tokens.len() && sequence.tokens[cursor] != AUDIO_TOKEN_ID {
            cursor += 1;
        }
        let count = features.soft_tokens();
        sequence.audio.push(SequenceAudio {
            offset: cursor,
            features,
        });
        cursor += count;
    }
    cursor = 0;
    let budget = case.member_size("max_soft_tokens", DEFAULT_IMAGE_SOFT_TOKENS as usize) as u32;
    for picture in case.find("images").map_or(&[][..], Value::items) {
        let patches = process_image(&common::make_image(picture), budget).unwrap();
        while cursor < sequence.tokens.len() && sequence.tokens[cursor] != IMAGE_TOKEN_ID {
            cursor += 1;
        }
        let count = patches.soft_tokens();
        sequence.images.push(SequenceImage {
            offset: cursor,
            patches,
            token_id: IMAGE_TOKEN_ID,
        });
        cursor += count;
    }
    sequence
}

fn cosine_with(vector: &[f32], expected: &Value) -> f64 {
    let (mut dot, mut norm, mut reference_norm) = (0.0, 0.0, 0.0);
    for (&value, reference) in vector.iter().zip(expected.items()) {
        let reference = reference.as_double(0.0);
        dot += f64::from(value) * reference;
        norm += f64::from(value) * f64::from(value);
        reference_norm += reference * reference;
    }
    dot / (norm * reference_norm).sqrt()
}

#[test]
fn image_and_audio_embeddings_match_the_reference_model() {
    let root = common::checkpoint("model.safetensors");
    let runtime = EncoderRuntime::create(&root).unwrap_or_else(|e| panic!("{e}"));
    for (file, threshold) in [
        ("image_golden.json", 0.9999),
        ("audio_embedding_golden.json", 0.999),
    ] {
        for case in fixture(file).find("cases").unwrap().items() {
            let name = case.member_str("name", "");
            let vectors = runtime
                .embed_multimodal(vec![sequence(case)], None)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(vectors[0].len(), 768, "{name} width");
            assert!(vectors[0].iter().all(|v| v.is_finite()), "{name} finite");
            let cosine = cosine_with(&vectors[0], case.find("embedding").unwrap());
            println!("{name} embedding cosine={cosine}");
            assert!(cosine > threshold, "{name}: cosine {cosine}");
        }
    }
    assert!(!runtime.device_lost());
}

#[test]
fn video_embeddings_match_the_reference_model() {
    let root = common::checkpoint("model.safetensors");
    let runtime = EncoderRuntime::create(&root).unwrap_or_else(|e| panic!("{e}"));
    for case in fixture("video_golden.json").find("cases").unwrap().items() {
        let name = case.member_str("name", "");
        // Its image and audio parts are the functional suite's inputs.
        if name == "mixed" {
            continue;
        }
        let settings = case.find("options").unwrap();
        let options = VideoOptions {
            fps: settings.member_double("fps", 1.0),
            timestamps: settings
                .find("add_timestamps")
                .is_some_and(|v| v.as_bool(false)),
            max_frames: settings.member_size("max_frames", 32),
            truncate_overflow: settings.member_str("overflow_strategy", "") == "truncate",
        };
        let tokens: Vec<u32> = case
            .find("token_ids")
            .unwrap()
            .items()
            .iter()
            .map(|id| id.as_size(0) as u32)
            .collect();
        let mut sequence = EncoderSequence::text(tokens);
        let (mut cursor, mut sampled) = (0, 0);
        for video in case.find("videos").unwrap().items() {
            let path = common::fixture(&video.member_str("path", ""));
            let bytes = std::fs::read(&path).unwrap();
            for frame in process_video(&bytes, &options, None, 64 - sampled).unwrap() {
                while sequence.tokens[cursor] != VIDEO_TOKEN_ID {
                    cursor += 1;
                }
                let count = frame.patches.soft_tokens();
                sequence.images.push(SequenceImage {
                    offset: cursor,
                    patches: frame.patches,
                    token_id: VIDEO_TOKEN_ID,
                });
                cursor += count;
                sampled += 1;
            }
        }
        let vectors = runtime
            .embed_multimodal(vec![sequence], None)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(vectors[0].iter().all(|v| v.is_finite()), "{name} finite");
        let cosine = cosine_with(&vectors[0], case.find("embedding").unwrap());
        println!("{name} video embedding cosine={cosine}");
        assert!(cosine >= 0.9999, "{name}: cosine {cosine}");
    }
    assert!(!runtime.device_lost());
}
