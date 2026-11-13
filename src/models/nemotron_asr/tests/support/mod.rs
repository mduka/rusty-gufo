//! Helpers the tests share (`tests/test_support.hpp`): fixtures, WAV
//! decoding, golden values, split sizes and tolerance checks.

#![allow(dead_code)]

use std::path::PathBuf;

use gufo_core::json::Value;
use gufo_nemotron_asr::safetensors::SafetensorsFile;

/// A file of `tests/fixtures`.
pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// `GUFO_NEMOTRON_ASR_MODEL`, the Hugging Face directory: the model tests
/// fail (rather than skip) without it, since a missing model is not a pass.
pub fn model_root() -> PathBuf {
    std::env::var_os("GUFO_NEMOTRON_ASR_MODEL")
        .filter(|root| !root.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!("GUFO_NEMOTRON_ASR_MODEL must name the Nemotron 3.5 ASR model directory")
        })
}

/// Mono float samples and the rate of a 16-bit PCM WAV fixture, as Gufo's
/// audio decoder gives them (`value / 32768`).
pub fn read_wav(name: &str) -> (Vec<f32>, u32) {
    let bytes = std::fs::read(fixture(name)).unwrap_or_else(|error| panic!("{name}: {error}"));
    assert!(
        bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "{name} is a RIFF WAV"
    );
    let word = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
    let dword =
        |at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    let (mut rate, mut data) = (0, None);
    let mut offset = 12;
    while offset + 8 <= bytes.len() {
        let size = dword(offset + 4) as usize;
        let body = offset + 8;
        match &bytes[offset..offset + 4] {
            b"fmt " => {
                assert_eq!(word(body), 1, "{name} is PCM");
                assert_eq!(word(body + 2), 1, "{name} is mono");
                assert_eq!(word(body + 14), 16, "{name} has 16-bit samples");
                rate = dword(body + 4);
            }
            b"data" => data = Some(&bytes[body..body + size]),
            _ => {}
        }
        offset = body + size + (size & 1);
    }
    let data = data.unwrap_or_else(|| panic!("{name} has no data chunk"));
    let samples = data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from(i16::from_le_bytes(*pair)) / 32768.0)
        .collect();
    (samples, rate)
}

/// The values of an F32 tensor of `golden.safetensors`.
pub fn floats(golden: &SafetensorsFile, name: &str) -> Vec<f32> {
    golden
        .find(name)
        .ok_or_else(|| format!("golden lacks {name}"))
        .and_then(|tensor| tensor.f32_values())
        .unwrap_or_else(|error| panic!("{error}"))
}

/// `tests/fixtures/golden.json`.
pub fn golden_json() -> Value {
    let text = std::fs::read(fixture("golden.json")).expect("golden.json");
    gufo_core::json::parse(&text).expect("golden.json parses")
}

pub fn ids(array: &Value) -> Vec<u32> {
    array
        .items()
        .iter()
        .map(|item| item.as_double(0.0) as u32)
        .collect()
}

/// Deterministic pseudo-random split sizes (`Splits`).
pub struct Splits(u32);

impl Splits {
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }

    /// 1 to `limit`.
    pub fn next(&mut self, limit: usize) -> usize {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        1 + (self.0 >> 8) as usize % limit
    }
}

/// Max and mean absolute difference and the largest reference magnitude.
pub struct Difference {
    pub max_abs: f64,
    pub max_ref: f64,
    pub mean_abs: f64,
}

pub fn compare(actual: &[f32], expected: &[f32]) -> Difference {
    if actual.len() != expected.len() {
        return Difference {
            max_abs: f64::INFINITY,
            max_ref: 0.0,
            mean_abs: 0.0,
        };
    }
    let mut d = Difference {
        max_abs: 0.0,
        max_ref: 0.0,
        mean_abs: 0.0,
    };
    for (a, e) in actual.iter().zip(expected) {
        let error = (f64::from(*a) - f64::from(*e)).abs();
        d.max_abs = d.max_abs.max(error);
        d.max_ref = d.max_ref.max(f64::from(*e).abs());
        d.mean_abs += error;
    }
    d.mean_abs /= actual.len().max(1) as f64;
    d
}

/// Checks max |a - e| <= relative * max |e| (`ExpectClose`).
pub fn assert_close(what: &str, actual: &[f32], expected: &[f32], relative: f64) {
    let d = compare(actual, expected);
    println!(
        "{what}: max abs {} (of {}), mean abs {}",
        d.max_abs, d.max_ref, d.mean_abs
    );
    assert!(
        d.max_abs <= relative * d.max_ref,
        "{what} differs from the reference: max abs {} of {}",
        d.max_abs,
        d.max_ref
    );
}
