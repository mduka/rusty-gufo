//! The dense MTP head against the C++ engine, bit for bit:
//! cargo test --profile gpu-test -p gufo-qwen35 --features gpu --test mtp
//!
//! Needs the Qwen3.8 27B GGUF (`GUFO_QWEN27B_MODEL`, else the pinned file in
//! the Hugging Face cache) and the test head of `tests/reference/mtp_test_head.py`
//! (`GUFO_QWEN27B_MTP`, else `target/qwen-mtp/qwen36-mtp-head.gguf`), and
//! fails without them. The fixtures come from `tests/reference/cpp_decode_dump.cpp`:
//! `mtp-head`, the prompt's final hidden rows, then the head primed with
//! them and 32 greedy head steps from the target's first token; `mtp`, the
//! verifier as `gufo prompt --speculative mtp` drives it (7 draft tokens,
//! adaptive length), whose every step must match.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use gufo_qwen35::mtp::{MtpDraft, MtpHead, MtpModel};
use gufo_qwen35::{Engine, Speculative};
use serde_json::Value;

/// One target model on the device at a time.
static SERIAL: Mutex<()> = Mutex::new(());

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

fn head() -> PathBuf {
    std::env::var_os("GUFO_QWEN27B_MTP")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../target/qwen-mtp/qwen36-mtp-head.gguf")
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
fn head_steps_match_the_cpp_engine() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/qwen3.8-27b-mtp.json")).unwrap();
    let decode: Value =
        serde_json::from_str(include_str!("fixtures/qwen3.8-27b-decode.json")).unwrap();
    let prompt = u32s(&decode["prompt"]);
    let expected = &fixture["head"];
    let ids = u32s(&expected["ids"]);
    let max_context = u32::try_from(fixture["max_context"].as_u64().unwrap()).unwrap();

    let mut engine = Engine::load(&model(), max_context).unwrap();
    engine.set_hidden_capture(true, &[]).unwrap();
    let first = engine.forward_prompt(&prompt, 0, true).unwrap().unwrap();
    assert_eq!(first, ids[0], "target token after the prompt");
    let hidden = engine.prompt_hidden().unwrap().to_vec();
    assert_eq!(
        fnv1a64(&hidden),
        expected["hidden_fnv1a64"].as_str().unwrap(),
        "final hidden rows of the prompt"
    );

    let model = engine.model();
    let stream = model.stream();
    let mtp = MtpModel::load(stream, &head(), model.weights()).unwrap();
    let mut head = MtpHead::new(stream, &mtp, max_context).unwrap();
    // Input token t + 1 pairs with target row t, as when priming.
    let width = hidden.len() / prompt.len();
    let row = |r: usize| &hidden[r * width..(r + 1) * width];
    for r in 0..prompt.len() - 1 {
        head.forward_target_hidden(stream, &mtp, None, prompt[r + 1], row(r), r as u32, false)
            .unwrap();
    }
    let mut token = first;
    let digests = expected["logits_fnv1a64"].as_array().unwrap();
    let positions = prompt.len() as u32 - 1..;
    for (position, (step, digest)) in positions.zip(digests.iter().enumerate()) {
        token = if step == 0 {
            head.forward_target_hidden(
                stream,
                &mtp,
                None,
                token,
                row(prompt.len() - 1),
                position,
                true,
            )
        } else {
            head.forward_feedback(stream, &mtp, None, token, position, true)
        }
        .unwrap();
        assert_eq!(token, ids[step + 1], "token of step {step}");
        assert_eq!(
            fnv1a64(&head.logits(stream).unwrap()),
            digest.as_str().unwrap(),
            "logits of step {step}"
        );
    }
}

#[test]
fn speculative_steps_match_the_cpp_engine() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/qwen3.8-27b-mtp.json")).unwrap();
    let decode: Value =
        serde_json::from_str(include_str!("fixtures/qwen3.8-27b-decode.json")).unwrap();
    let prompt = u32s(&decode["prompt"]);
    let expected = &fixture["speculative"];
    let total = expected["tokens"].as_u64().unwrap() as usize;
    let max_context = u32::try_from(fixture["max_context"].as_u64().unwrap()).unwrap();

    let mut engine = Engine::load(&model(), max_context).unwrap();
    let stream = engine.model().stream();
    let mtp = Arc::new(MtpModel::load(stream, &head(), engine.weights()).unwrap());
    let draft = MtpDraft::new(stream, &mtp, max_context, 7).unwrap();
    let mut speculative = Speculative::mtp(mtp, draft, 1, 7);
    let mut current = speculative.prime(&mut engine, &prompt, true).unwrap();
    let mut position = prompt.len() as u32;
    let mut ids = Vec::new();
    for (index, want) in expected["steps"].as_array().unwrap().iter().enumerate() {
        let step = speculative
            .verify_step(
                &mut engine,
                &[current],
                current,
                position,
                999_999,
                (total - ids.len()) as u32,
            )
            .unwrap();
        let mut got = vec![step.drafted as u32, step.accepted as u32];
        got.extend(&step.emitted);
        assert_eq!(got, u32s(want), "step {index}");
        for &token in &step.emitted {
            if ids.len() < total {
                ids.push(token);
                position += 1;
            }
        }
        current = step.next_token;
    }
    assert_eq!(ids, u32s(&expected["ids"]));
}
