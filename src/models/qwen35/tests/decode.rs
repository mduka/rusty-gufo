//! The Rust forward pass against the C++ engine's `ForwardToken`, bit for bit:
//! cargo test --release -p gufo-qwen35 --features gpu --test decode
//!
//! Needs the Qwen3.8 27B GGUF (`GUFO_QWEN27B_MODEL`, else the pinned file in
//! the Hugging Face cache) and fails without it. The fixture comes from
//! `tests/reference/cpp_decode_dump.cpp decode`: every prompt token and 140 greedy steps
//! through `ForwardToken`, crossing into split-K attention at position 127.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use gufo_qwen35::Engine;
use serde_json::Value;

fn model() -> PathBuf {
    std::env::var_os("GUFO_QWEN27B_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(
                ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/\
                 4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf",
            )
        })
}

fn fnv1a64(values: &[f32]) -> String {
    let hash = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        });
    format!("{hash:016x}")
}

fn u32s(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}

/// Feeds the prompt token by token, then checks `steps` greedy steps.
fn check(engine: &mut Engine, fixture: &Value, steps: usize) {
    let prompt = u32s(&fixture["prompt"]);
    let ids = u32s(&fixture["decode"]["ids"]);
    let digests = fixture["decode"]["logits_fnv1a64"].as_array().unwrap();
    engine.reset().unwrap();
    let mut next = None;
    for (pos, &token) in (0u32..).zip(&prompt) {
        next = engine
            .forward_token(token, pos, pos as usize + 1 == prompt.len())
            .unwrap();
    }
    for (step, pos) in (0..=steps).zip(prompt.len() as u32..) {
        let token = next.unwrap();
        assert_eq!(token, ids[step], "token of step {step}");
        let digest = fnv1a64(&engine.logits().unwrap());
        assert_eq!(
            digest,
            digests[step].as_str().unwrap(),
            "logits of step {step}"
        );
        next = engine.forward_token(token, pos, true).unwrap();
    }
}

#[test]
fn decode_matches_the_cpp_engine_and_reset_starts_a_new_sequence() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/qwen3.8-27b-decode.json")).unwrap();
    let mut engine = Engine::load(&model(), 4096).unwrap();
    assert_eq!(engine.config().num_layers, 64);
    check(&mut engine, &fixture, 140);
    // The same engine again: nothing of the first sequence may leak.
    check(&mut engine, &fixture, 8);
    // Below position 127 the step is one replayed graph.
    assert_eq!(engine.captured_graphs(), 1);
    // With target-layer taps the step copies hidden rows inside its graph.
    engine
        .set_hidden_capture(true, &[5, 19, 33, 47, 61])
        .unwrap();
    check(&mut engine, &fixture, 8);
    assert_eq!(engine.captured_graphs(), 2);
    assert_eq!(engine.last_hidden().unwrap().len(), 5 * 5120);
    assert_eq!(engine.graph_mismatches(), 0);
}
