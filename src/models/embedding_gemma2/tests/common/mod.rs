//! Checkpoint and fixture locations shared by the model tests. The tests fail
//! (rather than skip) when the checkpoint is missing: a missing model is not
//! a pass.

#![allow(dead_code, clippy::unwrap_used)]

pub mod reference;

use std::path::PathBuf;

/// `EMBEDDING_GEMMA2_MODEL_ROOT`, else the pinned snapshot in the Hugging
/// Face cache.
pub fn model_root() -> PathBuf {
    std::env::var_os("EMBEDDING_GEMMA2_MODEL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(
                ".cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/\
                 914f7f89142e33e77833254d9c9b90c3cef7303b",
            )
        })
}

/// The model root, which must hold `file`; panics with the remedy otherwise.
pub fn checkpoint(file: &str) -> PathBuf {
    let root = model_root();
    assert!(
        root.join(file).is_file(),
        "EmbeddingGemma 2 checkpoint missing: {} has no {file}; download \
         google/embeddinggemma-2 at revision 914f7f89142e33e77833254d9c9b90c3cef7303b or set \
         EMBEDDING_GEMMA2_MODEL_ROOT",
        root.display()
    );
    root
}

pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    dot / (na * nb).sqrt()
}
