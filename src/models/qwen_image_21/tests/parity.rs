//! Bit-exact parity with the C++ engine on the cases of
//! `tests/fixtures/cases.json`: the token IDs of every tokenizer string, and
//! per generation the SHA-256 of every observed tensor (in observation
//! order) and of the image, against `tests/fixtures/qwen-image-parity.json`.
//!
//! The fixture was recorded from the C++ engine (with edit noise bound to
//! the references) with
//! `tests/reference/cpp_reference.cpp` (`record_cpp_fixture`, ignored, reads its
//! output directory from `QWEN_IMAGE_CPP_DUMPS`). With the dumps present,
//! `compare_with_cpp_dumps` (ignored) reports the first diverging tensor
//! numerically.
//!
//! cargo test --profile gpu-test -p gufo-qwen-image-21 --features gpu,models --test parity

#![cfg(all(feature = "gpu", feature = "models"))]
#![allow(clippy::unwrap_used)]

mod common;

use std::path::PathBuf;

use gufo_core::json::{self, Value};
use gufo_qwen_image_21::{Model, Observer};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Every observed tensor of one generation, in order.
struct Recorder {
    all: bool,
    tensors: Vec<(String, usize, usize, Vec<f32>)>,
}

impl Observer for Recorder {
    fn wants(&self, name: &str) -> bool {
        self.all || common::coarse(name)
    }

    fn write(&mut self, name: &str, values: &[f32], rows: usize, cols: usize) {
        // As tests/reference/cpp_reference.cpp names them: `vision.output~2` for the
        // second reference image.
        let occurrence = 1 + self
            .tensors
            .iter()
            .filter(|t| t.0.split('~').next() == Some(name))
            .count();
        let name = if occurrence == 1 {
            name.to_owned()
        } else {
            format!("{name}~{occurrence}")
        };
        self.tensors.push((name, rows, cols, values.to_vec()));
    }
}

/// One generation's outcome as the fixture records it.
struct Outcome {
    tensors: Vec<(String, usize, usize, String)>,
    image: String,
    size: (i32, i32),
}

fn run_rust(model: &Model, case: &common::Case) -> (Outcome, Recorder) {
    let mut recorder = Recorder {
        all: case.all,
        tensors: Vec::new(),
    };
    let generation = model
        .generate(&case.request, None, Some(&mut recorder))
        .unwrap();
    let outcome = Outcome {
        tensors: recorder
            .tensors
            .iter()
            .map(|(name, rows, cols, values)| (name.clone(), *rows, *cols, hex(&f32_bytes(values))))
            .collect(),
        image: hex(&generation.image.rgba),
        size: (generation.image.width, generation.image.height),
    };
    (outcome, recorder)
}

fn cpp_dumps() -> PathBuf {
    std::env::var_os("QWEN_IMAGE_CPP_DUMPS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../target/image-ref/cpp")
        })
}

#[test]
#[ignore = "writes the fixture from the C++ dumps of tests/reference/cpp_reference.cpp"]
fn record_cpp_fixture() {
    let dumps = cpp_dumps();
    let tokens: Vec<Value> = std::fs::read_to_string(dumps.join("tokens.txt"))
        .unwrap()
        .lines()
        .map(|line| {
            Value::from(
                line.split_whitespace()
                    .map(|id| Value::from(id.parse::<f64>().unwrap()))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let mut generations = Vec::new();
    for case in common::generations() {
        let directory = dumps.join(&case.name);
        let shapes = std::fs::read_to_string(directory.join("tensors.txt")).unwrap();
        let mut tensors = Vec::new();
        for line in shapes.lines() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let bytes = std::fs::read(directory.join(format!("{}.f32", fields[0]))).unwrap();
            tensors.push(Value::from(vec![
                Value::from(fields[0]),
                Value::from(fields[1].parse::<f64>().unwrap()),
                Value::from(fields[2].parse::<f64>().unwrap()),
                Value::from(hex(&bytes)),
            ]));
        }
        let image = std::fs::read(directory.join("image.rgba")).unwrap();
        generations.push(
            Value::object()
                .with("name", case.name.as_str())
                .with("image", hex(&image))
                .with("tensors", Value::from(tensors)),
        );
    }
    let fixture = Value::object()
        .with("tokens", Value::from(tokens))
        .with("generations", Value::from(generations));
    std::fs::write(
        common::fixture("qwen-image-parity.json"),
        fixture.dump().unwrap() + "\n",
    )
    .unwrap();
}

#[test]
fn every_case_matches_the_cpp_engine() {
    let fixture =
        json::parse(&std::fs::read(common::fixture("qwen-image-parity.json")).unwrap()).unwrap();
    let model = Model::load(&common::model_root()).unwrap();
    let cases = common::cases();
    let texts = cases.find("tokenizer").unwrap().items();
    let expected_tokens = fixture.find("tokens").unwrap().items();
    assert_eq!(texts.len(), expected_tokens.len());
    for (text, expected) in texts.iter().zip(expected_tokens) {
        let ids: Vec<f64> = model
            .tokenize(text.str())
            .iter()
            .map(|&id| f64::from(id))
            .collect();
        let want: Vec<f64> = expected
            .items()
            .iter()
            .map(|id| id.as_double(-1.0))
            .collect();
        assert_eq!(ids, want, "tokens of {:?}", text.str());
    }
    let expected = fixture.find("generations").unwrap().items();
    let mut failures = Vec::new();
    for (case, want) in common::generations().iter().zip(expected) {
        assert_eq!(case.name, want.member_str("name", ""));
        let (outcome, _) = run_rust(&model, case);
        assert_eq!(outcome.size, (case.request.width, case.request.height));
        let want_tensors = want.find("tensors").unwrap().items();
        let names: Vec<String> = outcome.tensors.iter().map(|t| t.0.clone()).collect();
        let want_names: Vec<String> = want_tensors
            .iter()
            .map(|t| t.items()[0].str().to_owned())
            .collect();
        assert_eq!(names, want_names, "observed tensors of {}", case.name);
        for (got, want) in outcome.tensors.iter().zip(want_tensors) {
            let w = want.items();
            let shape = (w[1].as_double(0.0) as usize, w[2].as_double(0.0) as usize);
            assert_eq!((got.1, got.2), shape, "shape of {} in {}", got.0, case.name);
            if got.3 != w[3].str() {
                failures.push(format!("{}: first different tensor {}", case.name, got.0));
                break;
            }
        }
        if outcome.image != want.member_str("image", "") {
            failures.push(format!("{}: image differs", case.name));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Numerical differences against the raw C++ dumps, for debugging.
#[test]
#[ignore = "needs the C++ dumps of tests/reference/cpp_reference.cpp"]
fn compare_with_cpp_dumps() {
    let dumps = cpp_dumps();
    let model = Model::load(&common::model_root()).unwrap();
    let only = std::env::var("QWEN_IMAGE_CASE").ok();
    for case in common::generations() {
        if only.as_ref().is_some_and(|name| *name != case.name) {
            continue;
        }
        let (outcome, recorder) = run_rust(&model, &case);
        let directory = dumps.join(&case.name);
        let mut reported = 0;
        for (name, _, _, values) in &recorder.tensors {
            let bytes = std::fs::read(directory.join(format!("{name}.f32"))).unwrap();
            let reference: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let differing = values
                .iter()
                .zip(&reference)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            if differing > 0 && reported < 8 {
                let max = values
                    .iter()
                    .zip(&reference)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let first = values
                    .iter()
                    .zip(&reference)
                    .position(|(a, b)| a.to_bits() != b.to_bits())
                    .unwrap();
                eprintln!(
                    "{}: {name}: {differing}/{} differ, max |d| {max:e}, first at {first} ({} vs {})",
                    case.name,
                    values.len(),
                    values[first],
                    reference[first]
                );
                reported += 1;
            }
        }
        let image = std::fs::read(directory.join("image.rgba")).unwrap();
        let same = outcome.image == hex(&image);
        eprintln!(
            "{}: image {}",
            case.name,
            if same { "identical" } else { "differs" }
        );
    }
}
