//! The checkpoint's text backbone through the safetensors loader:
//! cargo test -p gufo-embedding-gemma2 --features models --test loader

#![cfg(feature = "models")]
#![allow(clippy::unwrap_used)]

mod common;

use gufo_embedding_gemma2::loader::{
    DType, expected_text_tensors, load_model_directory, looks_like_embedding_gemma2,
};

#[test]
fn the_text_backbone_loads_with_exactly_its_tensors() {
    let root = common::checkpoint("model.safetensors");
    assert!(
        looks_like_embedding_gemma2(&root),
        "model configuration must be recognized"
    );
    let loaded = load_model_directory(&root).unwrap_or_else(|e| panic!("{e}"));
    let expected = expected_text_tensors(&loaded.config.text);
    assert_eq!(
        loaded.store.len(),
        expected.len(),
        "the store must hold exactly the text-backbone tensors"
    );
    let embedding = loaded
        .store
        .find("language_model.embed_tokens.weight")
        .unwrap();
    assert!(
        embedding.dtype == DType::Bf16 && embedding.byte_count == 262_144 * 512 * 2,
        "the token embedding must be BF16 [262144, 512]"
    );
    assert!(
        loaded
            .store
            .find("vision_tower.patch_embedder.input_proj.weight")
            .is_none(),
        "vision tensors must not be indexed"
    );
}

#[test]
fn other_directories_are_refused() {
    let directory = tempfile::tempdir().unwrap();
    assert!(!looks_like_embedding_gemma2(directory.path()));
    let error = load_model_directory(directory.path()).err().unwrap();
    assert!(error.contains("cannot parse"), "{error}");
}
