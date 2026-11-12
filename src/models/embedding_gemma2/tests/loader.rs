//! The checkpoint's text backbone through the safetensors loader:
//! cargo test -p gufo-embedding-gemma2 --features models --test loader

#![cfg(feature = "models")]
#![allow(clippy::unwrap_used)]

mod common;

use gufo_embedding_gemma2::loader::{
    DType, Towers, expected_audio_tensors, expected_text_tensors, expected_vision_tensors,
    load_model_directory, looks_like_embedding_gemma2,
};

#[test]
fn the_text_backbone_loads_with_exactly_its_tensors() {
    let root = common::checkpoint("model.safetensors");
    assert!(
        looks_like_embedding_gemma2(&root),
        "model configuration must be recognized"
    );
    let loaded = load_model_directory(&root, Towers::default()).unwrap_or_else(|e| panic!("{e}"));
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
    assert!(
        loaded
            .store
            .find("audio_tower.output_proj.weight")
            .is_none(),
        "text loading must not index audio weights"
    );
    let vision = Towers {
        vision: true,
        audio: false,
    };
    let multimodal = load_model_directory(&root, vision).unwrap_or_else(|e| panic!("{e}"));
    assert!(multimodal.config.supported_vision);
    assert_eq!(
        multimodal.store.len(),
        expected.len() + expected_vision_tensors().len(),
        "the multimodal store holds text and vision only"
    );
    assert!(
        multimodal
            .store
            .find("vision_tower.patch_embedder.input_proj.weight")
            .is_some()
    );
    let audio = Towers {
        vision: false,
        audio: true,
    };
    let audio = load_model_directory(&root, audio).unwrap_or_else(|e| panic!("{e}"));
    assert!(audio.config.supported_audio);
    assert_eq!(
        audio.store.len(),
        expected.len() + expected_audio_tensors().len(),
        "an audio load indexes exactly text and audio tensors"
    );
    assert!(audio.store.find("audio_tower.output_proj.weight").is_some());
}

#[test]
fn other_directories_are_refused() {
    let directory = tempfile::tempdir().unwrap();
    assert!(!looks_like_embedding_gemma2(directory.path()));
    let error = load_model_directory(directory.path(), Towers::default())
        .err()
        .unwrap();
    assert!(error.contains("cannot parse"), "{error}");
}
