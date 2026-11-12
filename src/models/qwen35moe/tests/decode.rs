//! The Rust forward pass against the C++ reference's `ForwardToken`, bit for
//! bit (`decode` fixtures):
//! cargo test --profile gpu-test -p gufo-qwen35moe --features gpu --test decode
//!
//! Needs `GUFO_QWEN35MOE_MODEL` (UD-Q4_K_M) and `GUFO_QWEN35MOE_Q8_MODEL`
//! (Q8_0) and fails without them. The haiku prompt token by token, then 140
//! greedy steps, crossing into split-K attention at position 127; a second
//! sequence on the same engine, and one with target-layer taps, repeat the
//! first steps.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use common::{Q8_0, Quant, SERIAL, UD_Q4_K_M, fnv1a64, u32s};
use gufo_qwen35moe::Engine;
use serde_json::Value;

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

fn decode_matches(quant: Quant) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = quant.fixture("decode");
    let max_context = u32::try_from(fixture["max_context"].as_u64().unwrap()).unwrap();
    let mut engine = Engine::load(&quant.model(), max_context).unwrap();
    assert_eq!(engine.config().num_layers, 40);
    let steps = fixture["decode"]["steps"].as_u64().unwrap() as usize;
    check(&mut engine, &fixture, steps);
    // The same engine again: nothing of the first sequence may leak.
    check(&mut engine, &fixture, 8);
    // Taps copy hidden rows without touching the arithmetic.
    engine
        .set_hidden_capture(true, &[1, 9, 17, 25, 33, 37, 38, 39])
        .unwrap();
    check(&mut engine, &fixture, 8);
    assert_eq!(engine.last_hidden().unwrap().len(), 8 * 2048);
}

#[test]
fn qwen36_35b_a3b_ud_q4_k_m() {
    decode_matches(UD_Q4_K_M);
}

#[test]
fn qwen36_35b_a3b_q8_0() {
    decode_matches(Q8_0);
}
