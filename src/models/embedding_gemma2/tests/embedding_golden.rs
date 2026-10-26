//! GPU embeddings of text against sentence-transformers output for
//! google/embeddinggemma-2. The f32 reference fixture comes from
//! `tools/models/embedding_gemma2/reference_embeddings.py`. Token IDs must
//! match exactly, and embeddings must agree within the model's own BF16
//! noise:
//! cargo test --profile gpu-test -p gufo-embedding-gemma2 --features gpu,models --test embedding_golden

#![cfg(all(feature = "gpu", feature = "models"))]
#![allow(clippy::unwrap_used)]

mod common;

use common::cosine;
use gufo_core::json;
use gufo_embedding_gemma2::{EncoderRuntime, Tokenizer};

/// The checkpoint's own BF16 execution is within 0.99987 of f32 on these
/// probes. This implementation keeps activations in f32 between operations
/// and measures 0.99997 or better, so this bound leaves room for GPU
/// variation but catches a wrong operation.
const MINIMUM_COSINE: f64 = 0.9999;

fn truncated(vector: &[f32], dimensions: usize) -> Vec<f32> {
    let prefix = &vector[..dimensions];
    let norm = prefix
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    prefix
        .iter()
        .map(|&v| (f64::from(v) / norm) as f32)
        .collect()
}

#[test]
fn embeddings_match_sentence_transformers() {
    let root = common::checkpoint("model.safetensors");
    let tokenizer = Tokenizer::load(&root).unwrap_or_else(|e| panic!("{e}"));
    let runtime = EncoderRuntime::create(&root).unwrap_or_else(|e| panic!("{e}"));
    let fixture =
        json::parse(&std::fs::read(common::fixture("embedding_golden.json")).unwrap()).unwrap();
    let cases = fixture.find("cases").unwrap().items();
    assert!(!cases.is_empty(), "fixture must hold cases");

    let mut sequences = Vec::new();
    let mut expected = Vec::new();
    let mut names = Vec::new();
    for entry in cases {
        let name = entry.member_str("name", "");
        let text = entry.member_str("text", "");
        let ids: Vec<u32> = entry
            .find("token_ids")
            .unwrap()
            .items()
            .iter()
            .map(|id| id.as_size(0) as u32)
            .collect();
        assert_eq!(
            tokenizer.encode(text.as_bytes(), true),
            ids,
            "token ids must match Hugging Face for {name}"
        );
        let vector: Vec<f32> = entry
            .find("embedding")
            .unwrap()
            .items()
            .iter()
            .map(|value| value.as_double(0.0) as f32)
            .collect();
        assert_eq!(vector.len(), 768, "reference width");
        sequences.push(ids);
        expected.push(vector);
        names.push(name);
    }

    let actual = runtime.embed(&sequences, None).unwrap();
    assert_eq!(actual.len(), expected.len(), "one embedding per case");
    let mut worst = 1.0f64;
    let mut worst_truncated = 1.0f64;
    for i in 0..actual.len() {
        let full = cosine(&actual[i], &expected[i]);
        worst = worst.min(full);
        println!(
            "{} ({} tokens): cosine {full:.8}",
            names[i],
            sequences[i].len()
        );
        assert!(
            full >= MINIMUM_COSINE,
            "embedding differs from the reference for {}",
            names[i]
        );
        for dimensions in [128, 256, 512] {
            worst_truncated = worst_truncated.min(cosine(
                &truncated(&actual[i], dimensions),
                &truncated(&expected[i], dimensions),
            ));
        }
    }
    println!("worst full-width cosine {worst:.8}, worst truncated cosine {worst_truncated:.8}");
    assert!(
        worst_truncated >= MINIMUM_COSINE,
        "truncated embeddings differ from the reference"
    );
}
