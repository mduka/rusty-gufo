//! Checkpoint and fixture locations shared by the model tests. The tests fail
//! (rather than skip) when these are missing: a missing model is not a pass.

#![allow(dead_code, clippy::unwrap_used)]

use std::path::PathBuf;

/// `QWEN3_ASR_MODEL_ROOT`, else the pinned snapshot in the Hugging Face cache.
pub fn model_root() -> PathBuf {
    std::env::var_os("QWEN3_ASR_MODEL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(
                ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/\
                 7278e1e70fe206f11671096ffdd38061171dd6e5",
            )
        })
}

/// The official English example (SHA-256 f9b4440a…).
pub fn fixture_wav() -> Vec<u8> {
    let path = std::env::var_os("QWEN3_ASR_TEST_WAV")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap())
                .join(".cache/gufo/fixtures/qwen3-asr-en.wav")
        });
    std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {}: {e}", path.display()))
}

/// FNV-1a over the little-endian bytes of `values`, as computed over the
/// C++ engine's stage dumps.
pub fn fnv<T: Copy>(values: &[T], bytes: impl Fn(T) -> [u8; 4]) -> u64 {
    values
        .iter()
        .flat_map(|&v| bytes(v))
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
}

pub fn fnv_f32(values: &[f32]) -> u64 {
    fnv(values, f32::to_le_bytes)
}

pub fn fnv_u32(values: &[u32]) -> u64 {
    fnv(values, u32::to_le_bytes)
}
