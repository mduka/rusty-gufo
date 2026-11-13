//! The GPU model (`tests/model_test.cpp`), against two references:
//!
//! - transformers in float32 (`golden.safetensors`, `golden.json`, from
//!   `tools/reference/golden.py`) within the C++ test's tolerances: the
//!   subsampling, encoder and projection outputs of the first three chunks,
//!   the prediction network's start state and the joint's logits, and the
//!   token IDs of every golden run with the audio pushed in random pieces;
//! - the C++ extension bit for bit (`cxx_reference.safetensors`, from
//!   `tools/reference/pack_cxx_reference.py`): the same encoder outputs, and
//!   the token IDs and the prediction network's final output of every
//!   golden run with each clip pushed at once.
//!
//! cargo test --profile gpu-test -p gufo-nemotron-asr --features gpu,models --test model

#![cfg(all(feature = "gpu", feature = "models"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::sync::OnceLock;

use gufo_nemotron_asr::language::resolve_language;
use gufo_nemotron_asr::model::{EncoderTrace, MAX_PASS_FRAMES, Model, Utterance};
use gufo_nemotron_asr::safetensors::{DType, SafetensorsFile};

fn model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| {
        Model::load(&support::model_root()).unwrap_or_else(|error| panic!("{error}"))
    })
}

fn cxx() -> &'static SafetensorsFile {
    static CXX: OnceLock<SafetensorsFile> = OnceLock::new();
    CXX.get_or_init(|| {
        SafetensorsFile::open(&support::fixture("cxx_reference.safetensors")).unwrap()
    })
}

fn cxx_ids(name: &str) -> Vec<u32> {
    let tensor = cxx()
        .find(name)
        .unwrap_or_else(|| panic!("cxx_reference lacks {name}"));
    assert_eq!(tensor.dtype, DType::I32);
    tensor
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| i32::from_le_bytes(b) as u32)
        .collect()
}

/// Bit equality with the C++ output `name`.
fn assert_bits(name: &str, actual: &[f32]) {
    let expected = support::floats(cxx(), name);
    assert_eq!(actual.len(), expected.len(), "{name}: length");
    let first = actual
        .iter()
        .zip(&expected)
        .position(|(a, e)| a.to_bits() != e.to_bits());
    if let Some(index) = first {
        let d = support::compare(actual, &expected);
        panic!(
            "{name} differs from C++ at {index}: {} against {} (max abs {} of {})",
            actual[index], expected[index], d.max_abs, d.max_ref
        );
    }
}

fn language(clip: &str) -> &'static str {
    if clip == "english" { "en-US" } else { "de-DE" }
}

#[test]
fn encoder_matches_transformers_and_cxx() {
    let model = model();
    let golden = SafetensorsFile::open(&support::fixture("golden.safetensors")).unwrap();
    let bins = model.config().encoder.mel_bins;
    for clip in ["english", "german"] {
        let prompt = resolve_language(model.config(), Some(language(clip))).unwrap();
        let features = support::floats(&golden, &format!("{clip}.features"));
        let mut utterance = Utterance::new(model, 13, prompt, MAX_PASS_FRAMES).unwrap();
        // The prediction network's matrices are float16 (float32 sums).
        let start = utterance.decoder_output().unwrap();
        support::assert_close(
            &format!("{clip} decoder start"),
            &start,
            &support::floats(&golden, &format!("{clip}.decoder.start")),
            5e-4,
        );
        assert_bits(&format!("{clip}.decoder.start"), &start);
        let mut at = 0;
        for chunk in 0..3 {
            let frames = if chunk == 0 { 105 } else { 112 };
            let mut trace = EncoderTrace::default();
            utterance
                .encode_features(
                    &features[at * bins..(at + frames) * bins],
                    Some(&mut trace),
                    false,
                )
                .unwrap();
            at += frames;
            // Matrices and the subsampling's activations are float16 (float32
            // sums); single projected values move by up to 0.5 % of the
            // range, the joint's logits much less.
            for (part, values, tolerance) in [
                ("subsampled", &trace.subsampled, 2e-3),
                ("encoder", &trace.hidden, 2e-3),
                ("projected", &trace.projected, 1e-2),
            ] {
                let name = format!("{clip}.{part}.{chunk}");
                support::assert_close(&name, values, &support::floats(&golden, &name), tolerance);
                assert_bits(&name, values);
            }
            if chunk == 0 {
                let logits = utterance.joint_logits(0).unwrap();
                let name = format!("{clip}.joint.start");
                support::assert_close(&name, &logits, &support::floats(&golden, &name), 2e-3);
                assert_bits(&name, &logits);
            }
        }
    }
}

/// Every golden run with the audio pushed in random pieces (eagerly): the
/// token IDs, transcript and counts of transformers' offline transcription.
#[test]
fn transcriptions_match_transformers_in_any_split() {
    let model = model();
    let golden = support::golden_json();
    let mut seed = 1;
    for (clip, entry) in golden.find("clips").unwrap().members() {
        let (audio, _) = support::read_wav(&format!("{clip}.wav"));
        for run in entry.find("runs").unwrap().items() {
            let lookahead = run.member_size("lookahead", 0);
            let prompt = run.member_size("prompt", 0);
            let mut utterance = Utterance::new(model, lookahead, prompt, MAX_PASS_FRAMES).unwrap();
            let mut splits = support::Splits::new(seed);
            seed += 1;
            let mut at = 0;
            while at < audio.len() {
                let n = (audio.len() - at).min(splits.next(9000));
                utterance.push(&audio[at..at + n], true).unwrap();
                at += n;
            }
            utterance.finish().unwrap();
            let label = format!(
                "{clip} {} lookahead {lookahead}",
                run.member_str("language", "")
            );
            assert_eq!(
                utterance.chunks(),
                run.member_size("chunks", 0),
                "{label}: chunk count"
            );
            assert_eq!(
                utterance.encoder_frames(),
                run.member_size("encoder_frames", 0),
                "{label}: encoder frames"
            );
            assert_eq!(
                utterance.text().ids(),
                support::ids(run.find("ids").unwrap()),
                "{label}: token IDs"
            );
            assert_eq!(
                utterance.text().text(),
                run.member_str("text", ""),
                "{label}: transcript"
            );
            println!(
                "{label}: {} (encoder {:.1} ms, decoder {:.1} ms)",
                utterance.text().text(),
                utterance.encoder_ms(),
                utterance.decoder_ms()
            );
        }
    }
}

/// Every golden run with the clip pushed at once and finished, as the C++
/// dump ran it: token IDs and the prediction network's final output equal
/// C++ bit for bit.
#[test]
fn transcriptions_match_cxx_bit_for_bit() {
    let model = model();
    let golden = support::golden_json();
    for (clip, entry) in golden.find("clips").unwrap().members() {
        let (audio, _) = support::read_wav(&format!("{clip}.wav"));
        for run in entry.find("runs").unwrap().items() {
            let lookahead = run.member_size("lookahead", 0);
            let prompt = run.member_size("prompt", 0);
            let mut utterance = Utterance::new(model, lookahead, prompt, MAX_PASS_FRAMES).unwrap();
            utterance.push(&audio, false).unwrap();
            utterance.finish().unwrap();
            let name = format!("{clip}.{}.{lookahead}", run.member_str("language", ""));
            assert_eq!(
                utterance.text().ids(),
                cxx_ids(&format!("{name}.ids")),
                "{name}: token IDs"
            );
            assert_bits(
                &format!("{name}.decoder.end"),
                &utterance.decoder_output().unwrap(),
            );
        }
    }
}
