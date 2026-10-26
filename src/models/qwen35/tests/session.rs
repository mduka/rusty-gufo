//! Snapshots and retained sequences against the C++ engine:
//! cargo test --release -p gufo-qwen35 --features gpu --test session
//!
//! Needs the Qwen3.8 27B GGUF (`GUFO_QWEN27B_MODEL`, else the pinned file in
//! the Hugging Face cache) and fails without it. The session fixture comes
//! from `tests/reference/cpp_decode_dump.cpp session`; the snapshot test reuses the
//! prefill fixture.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use gufo_qwen35::Engine;
use gufo_qwen35::engine::RecurrentCapture;
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

/// The prefill fixture's chat prompt around `body` synthetic tokens.
fn synthetic(prefill: &Value, body: usize) -> Vec<u32> {
    let body = (0..body as u32).map(|i| 1000 + (i * 7919 + 13) % 50_000);
    u32s(&prefill["head"])
        .into_iter()
        .chain(body)
        .chain(u32s(&prefill["tail"]))
        .collect()
}

#[test]
fn snapshots_and_retained_sequences_match_the_cpp_engine() {
    let prefill: Value =
        serde_json::from_str(include_str!("fixtures/qwen3.8-27b-prefill.json")).unwrap();
    let decode: Value =
        serde_json::from_str(include_str!("fixtures/qwen3.8-27b-decode.json")).unwrap();
    let session: Value =
        serde_json::from_str(include_str!("fixtures/qwen3.8-27b-session.json")).unwrap();
    let max_context = u32::try_from(prefill["max_context"].as_u64().unwrap()).unwrap();
    let mut engine = Engine::load(&model(), max_context).unwrap();

    // A snapshot after the 353-token prefill, restored after another
    // sequence ran, continues with the C++ decode steps of that prefill.
    let case = &prefill["prefill"][2];
    let prompt = synthetic(&prefill, case["body"].as_u64().unwrap() as usize);
    let ids = u32s(&case["ids"]);
    let digests = case["logits_fnv1a64"].as_array().unwrap();
    engine.reset().unwrap();
    assert_eq!(
        engine.forward_prompt(&prompt, 0, true).unwrap(),
        Some(ids[0])
    );
    let snapshot = engine.save_snapshot(prompt.len() as u32).unwrap();
    assert_eq!(snapshot.valid_context() as usize, prompt.len());
    // The same capture into recurrent storage allocated ahead (what a
    // prefill hides for the serving runner) restores alike.
    let ahead = {
        let (model, session) = engine.parts_mut();
        let storage = RecurrentCapture::allocate(model.stream(), session.recurrent_lens()).unwrap();
        session
            .capture_with(model.stream(), prompt.len() as u32, Some(storage))
            .unwrap()
    };
    engine
        .generate(&synthetic(&prefill, 1129), 4, |_| true)
        .unwrap();
    for (round, snapshot) in [&snapshot, &ahead].into_iter().enumerate() {
        engine.restore_snapshot(snapshot).unwrap();
        let mut token = ids[0];
        for (step, pos) in (1..ids.len()).zip(prompt.len() as u32..) {
            token = engine.forward_token(token, pos, true).unwrap().unwrap();
            assert_eq!(token, ids[step], "round {round}: token of step {step}");
            assert_eq!(
                fnv1a64(&engine.logits().unwrap()),
                digests[step].as_str().unwrap(),
                "round {round}: logits of step {step}"
            );
        }
    }

    // A second chat turn over the retained first one.
    let suffix = u32s(&session["suffix"]);
    for case in session["cases"].as_array().unwrap() {
        let prompt = match case["body"].as_u64() {
            Some(body) => synthetic(&prefill, body as usize),
            None => u32s(&decode["prompt"]),
        };
        let max_new = case["max_new_tokens"].as_u64().unwrap() as usize;
        let first = engine.generate(&prompt, max_new, |_| true).unwrap();
        assert_eq!(first, u32s(&case["first"]), "{}", case["prompt"]);
        let mut next_prompt = prompt.clone();
        next_prompt.extend(&first);
        let retained = next_prompt.len();
        next_prompt.extend(&suffix);
        let second = engine
            .generate_from_prefix(&next_prompt, retained, max_new, |_| true)
            .unwrap();
        assert_eq!(
            second,
            u32s(&case["second"]),
            "{}: second turn",
            case["prompt"]
        );
    }
    assert_eq!(engine.graph_mismatches(), 0);
}
