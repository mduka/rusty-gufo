//! The Rust synthesis against the C++ engine: every request of [`cases`]
//! must produce the codec codes, samples and streamed chunks the C++ engine
//! produced (`tests/fixtures/qwen3-tts-parity.json`, FNV-1a over the
//! little-endian bits), buffered and streamed, for every variant.
//! cargo test -p gufo-qwen3-tts --features gpu,models --test parity
//!
//! On ROCm 10.0 the fixture was written from the C++ engine through its
//! former C ABI (the C++ runner replaced [`run_rust`] in this file, run with
//! the same cases). ROCm 10.1 changes the codec decoder's samples, not the
//! codes; `record_fixture` wrote them from this engine, whose CustomVoice
//! and VoiceDesign audio equals that of C++ v0.23.0 on ROCm 10.1 in the A/B
//! runs (docs/RUST.md):
//! cargo test -p gufo-qwen3-tts --features gpu,models --test parity -- --ignored record_fixture

#![cfg(all(feature = "gpu", feature = "models"))]
#![allow(clippy::unwrap_used)]

mod common;

use gufo_core::json;
use gufo_qwen3_tts::config::ModelVariant;
use gufo_qwen3_tts::sampling::SamplingOptions;
use gufo_qwen3_tts::synthesis::{SynthesisRequest, Synthesizer};

const CONTEXT: usize = 4096;

/// Codes, samples and the streamed chunks of one engine.
#[derive(Debug, PartialEq)]
struct Outcome {
    codes: Vec<i32>,
    samples: Vec<u32>,
    chunks: Vec<Vec<u32>>,
}

fn bits(samples: &[f32]) -> Vec<u32> {
    samples.iter().map(|s| s.to_bits()).collect()
}

fn run_rust(variant: ModelVariant, requests: &[(SynthesisRequest, bool)]) -> Vec<Outcome> {
    let root = common::model_root(variant);
    let mut synthesizer = Synthesizer::new(&root, CONTEXT, variant).unwrap();
    requests
        .iter()
        .map(|(request, streamed)| {
            let mut chunks = Vec::new();
            let mut sink = |samples: &[f32]| {
                chunks.push(bits(samples));
                true
            };
            let result = if *streamed {
                synthesizer
                    .generate(request, &|| false, Some(&mut sink))
                    .unwrap()
            } else {
                synthesizer.generate(request, &|| false, None).unwrap()
            };
            Outcome {
                codes: result.codes,
                samples: bits(&result.samples),
                chunks,
            }
        })
        .collect()
}

fn greedy() -> SamplingOptions {
    SamplingOptions {
        sample: false,
        predictor_sample: false,
        ..SamplingOptions::default()
    }
}

/// A named request of one variant; `streamed` passes an audio sink.
struct Case {
    name: &'static str,
    variant: ModelVariant,
    request: SynthesisRequest,
    streamed: bool,
}

fn cases() -> Vec<Case> {
    let custom = SynthesisRequest {
        text: "Hello from Gufo. The quick brown fox jumps over the lazy dog.".into(),
        speaker: "Vivian".into(),
        max_new_tokens: 48,
        ..SynthesisRequest::default()
    };
    let long = SynthesisRequest {
        text: "When the first light reaches the harbor, speak gently, with quiet confidence and a \
               warm, resonant tone. The boats are still resting beside the old wooden pier, and \
               the town is slowly waking to another bright morning. Gulls circle above the \
               masts while the fishermen check their nets and the bakers open their doors."
            .into(),
        speaker: "Vivian".into(),
        max_new_tokens: 420,
        sampling: greedy(),
        ..SynthesisRequest::default()
    };
    let (audio, transcript) = common::reference();
    let base = SynthesisRequest {
        text: "The boats are still resting beside the old wooden pier.".into(),
        reference_audio: audio,
        reference_text: transcript,
        max_new_tokens: 40,
        ..SynthesisRequest::default()
    };
    let case = |name, variant, request, streamed| Case {
        name,
        variant,
        request,
        streamed,
    };
    vec![
        case(
            "custom_greedy",
            ModelVariant::CustomVoice,
            SynthesisRequest {
                sampling: greedy(),
                ..custom.clone()
            },
            false,
        ),
        case(
            "custom_sampled",
            ModelVariant::CustomVoice,
            custom.clone(),
            false,
        ),
        case(
            "custom_auto_300",
            ModelVariant::CustomVoice,
            SynthesisRequest {
                max_new_tokens: 300,
                language: "auto".into(),
                ..custom.clone()
            },
            false,
        ),
        case(
            "custom_instruct_streamed",
            ModelVariant::CustomVoice,
            SynthesisRequest {
                sampling: SamplingOptions {
                    seed: 7,
                    top_k: 20,
                    top_p: 0.85,
                    temperature: 0.7,
                    predictor_top_k: 30,
                    predictor_top_p: 0.9,
                    predictor_temperature: 0.8,
                    ..SamplingOptions::default()
                },
                instruct: "Speak slowly and calmly.".into(),
                ..custom
            },
            true,
        ),
        // More than 300 frames: the offline decoder's chunk restart with 25
        // frames of context, and the incremental decoder's rebuild of it.
        case(
            "custom_long",
            ModelVariant::CustomVoice,
            long.clone(),
            false,
        ),
        case(
            "custom_long_streamed",
            ModelVariant::CustomVoice,
            long,
            true,
        ),
        case(
            "voice_design_streamed",
            ModelVariant::VoiceDesign,
            SynthesisRequest {
                text: "When the first light reaches the harbor, speak gently.".into(),
                instruct: "A warm, resonant female voice with quiet confidence.".into(),
                max_new_tokens: 64,
                ..SynthesisRequest::default()
            },
            true,
        ),
        case("base_icl", ModelVariant::Base, base.clone(), false),
        // The same reference again: cached encoder features and the
        // retained decoder prefix.
        case("base_icl_streamed", ModelVariant::Base, base.clone(), true),
        case(
            "base_speaker_only",
            ModelVariant::Base,
            SynthesisRequest {
                speaker_embedding_only: true,
                ..base
            },
            false,
        ),
    ]
}

fn fnv(words: &[u32]) -> u64 {
    words
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
}

/// The comparable digest of an outcome.
fn digest(outcome: &Outcome) -> json::Value {
    let codes: Vec<u32> = outcome.codes.iter().map(|&c| c as u32).collect();
    json::Value::object()
        .with("frames", codes.len() / 16)
        .with("codes_fnv", format!("{:016x}", fnv(&codes)))
        .with("samples", outcome.samples.len())
        .with("samples_fnv", format!("{:016x}", fnv(&outcome.samples)))
        .with(
            "chunks",
            outcome
                .chunks
                .iter()
                .map(|chunk| json::Value::from(format!("{:016x}", fnv(chunk))))
                .collect::<Vec<_>>(),
        )
}

fn fixture_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3-tts-parity.json")
}

/// Runs every case of `variant` on the Rust engine.
fn run(variant: ModelVariant) -> Vec<(&'static str, Outcome)> {
    let cases: Vec<Case> = cases()
        .into_iter()
        .filter(|c| c.variant == variant)
        .collect();
    let requests: Vec<(SynthesisRequest, bool)> = cases
        .iter()
        .map(|c| (c.request.clone(), c.streamed))
        .collect();
    cases
        .iter()
        .map(|c| c.name)
        .zip(run_rust(variant, &requests))
        .collect()
}

const VARIANTS: [ModelVariant; 3] = [
    ModelVariant::CustomVoice,
    ModelVariant::VoiceDesign,
    ModelVariant::Base,
];

/// Writes the fixture from this engine (see the module documentation).
#[test]
#[ignore = "records the fixture; run alone"]
fn record_fixture() {
    let mut fixture = json::Value::object().with(
        "description",
        "Qwen3-TTS outcomes for the cases of tests/parity.rs: frame count and FNV-1a of the \
         codes, sample count and FNV-1a of the sample bits, FNV-1a of every streamed chunk. \
         Written from the Rust engine on ROCm 10.1, whose CustomVoice and VoiceDesign audio \
         equals C++ v0.23.0's there; on ROCm 10.0 the C++ engine (upstream f783fed) wrote the \
         same codes.",
    );
    for variant in VARIANTS {
        for (name, outcome) in run(variant) {
            fixture = fixture.with(name, digest(&outcome));
        }
    }
    std::fs::write(fixture_path(), fixture.dump().unwrap() + "\n").unwrap();
}

#[test]
fn every_case_matches_the_cpp_engine() {
    let fixture = json::parse_str(&std::fs::read_to_string(fixture_path()).unwrap()).unwrap();
    for variant in VARIANTS {
        for (name, outcome) in run(variant) {
            let expected = fixture
                .find(name)
                .unwrap_or_else(|| panic!("{name}: no fixture"));
            assert_eq!(&digest(&outcome), expected, "{name}");
            assert!(outcome.codes.len() >= 16, "{name}: no codes");
            if name == "custom_long" {
                assert!(outcome.codes.len() / 16 > 300, "{name}: too short");
            }
        }
    }
}
