//! Speculative verification on the target side, without a draft model:
//! cargo test --release -p gufo-qwen35 --features gpu --test verify
//!
//! A scripted draft proposes the known greedy continuation of the decode
//! fixture with deliberate mistakes. Verification, rollback and replay of the
//! recurrent state must be lossless: the emitted tokens equal the
//! autoregressive ones, and a decode step afterwards gives the autoregressive
//! logits bit for bit.

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

#[test]
fn rolled_back_verification_is_lossless() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/qwen3.8-27b-decode.json")).unwrap();
    let prompt = u32s(&fixture["prompt"]);
    let ids = u32s(&fixture["decode"]["ids"]);
    let digests = fixture["decode"]["logits_fnv1a64"].as_array().unwrap();
    let mut engine = Engine::load(&model(), 4096).unwrap();
    engine
        .set_hidden_capture(true, &[5, 19, 33, 47, 61])
        .unwrap();

    // The prompt token by token, as in the fixture, then speculative steps.
    engine.reset().unwrap();
    let mut current = 0;
    for (pos, &token) in (0u32..).zip(&prompt) {
        if let Some(next) = engine
            .forward_token(token, pos, pos as usize + 1 == prompt.len())
            .unwrap()
        {
            current = next;
        }
    }
    assert_eq!(current, ids[0]);
    let (mut pos, mut emitted) = (prompt.len() as u32, 1usize);
    let target = 120;
    for step in 0.. {
        if emitted >= target {
            break;
        }
        // Up to seven drafts: the true continuation, with one wrong token at
        // a varying place in two of three steps.
        let count = (1 + step % 7).min(target - emitted);
        let mut proposal: Vec<u32> = ids[emitted..emitted + count].to_vec();
        if step % 3 != 2 {
            let wrong = step % count;
            proposal[wrong] = proposal[wrong].wrapping_add(1) % 248_000;
        }
        let mut inputs = vec![current];
        inputs.extend(&proposal);
        engine.save_state(pos).unwrap();
        let predictions = engine.verify(&inputs, pos).unwrap();
        assert_eq!(engine.verification_hidden().len(), inputs.len() * 5 * 5120);
        let accepted = predictions
            .iter()
            .zip(&proposal)
            .take_while(|(p, d)| p == d)
            .count();
        let correction = predictions[accepted];
        engine.finish_verification().unwrap();
        if accepted + 1 < inputs.len() {
            engine.restore_state().unwrap();
            engine.commit_verified(&inputs[..=accepted], pos).unwrap();
        }
        let mut out = proposal[..accepted].to_vec();
        out.push(correction);
        assert_eq!(out, ids[emitted..emitted + out.len()], "step {step}");
        pos += out.len() as u32;
        emitted += out.len();
        current = correction;
    }
    // The state equals the autoregressive one: the next step's logits too.
    let step = emitted - 1;
    let next = engine.forward_token(current, pos, true).unwrap();
    assert_eq!(next, Some(ids[step + 1]));
    assert_eq!(
        fnv1a64(&engine.logits().unwrap()),
        digests[step + 1].as_str().unwrap()
    );
    assert_eq!(engine.last_hidden().unwrap().len(), 5 * 5120);
    assert_eq!(engine.graph_mismatches(), 0);
}
