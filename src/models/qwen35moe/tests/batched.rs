//! Batched decode of several sessions against the C++ reference, bit for bit
//! (`batched` fixtures):
//! cargo test --profile gpu-test -p gufo-qwen35moe --features gpu --test batched
//!
//! Needs `GUFO_QWEN35MOE_MODEL` (UD-Q4_K_M) and `GUFO_QWEN35MOE_Q8_MODEL`
//! (Q8_0) and fails without them. Four sessions of one model (haiku, and the
//! synthetic bodies 7, 100 and 341) at different positions: each alone,
//! a prefill and twelve greedy `forward_token` steps (`serial`); then the
//! first two, three and four together, each prefilled, then twelve
//! `forward_token_batch` steps (`ForwardTokenBatch`, the serve runner's
//! `AdvanceBatch`). Tokens and the FNV-1a digest of every session's logits
//! match the C++ traces, and the batched traces equal the serial ones. A
//! second pass of width 2 with target-layer taps checks that the tap copies
//! leave the arithmetic alone.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use common::{Q8_0, Quant, SERIAL, UD_Q4_K_M, case_prompt, fnv1a64, u32s};
use gufo_qwen35moe::{Model, Session};
use serde_json::{Value, json};

/// The fixture's session prompts, in order.
fn prompts(prefill: &Value, fixture: &Value) -> Vec<(String, Vec<u32>)> {
    fixture["sessions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|session| {
            let name = session["name"].as_str().unwrap().to_string();
            let case = match name.strip_prefix("body") {
                Some(body) => json!({ "body": body.parse::<u64>().unwrap() }),
                None => json!({}),
            };
            let prompt = case_prompt(prefill, &case);
            assert_eq!(prompt.len() as u64, session["prompt_len"].as_u64().unwrap());
            (name, prompt)
        })
        .collect()
}

/// The first divergence of `ids` and `digests` from a fixture trace.
fn divergence(entry: &Value, ids: &[u32], digests: &[String]) -> Option<String> {
    let expected = u32s(&entry["ids"]);
    let expected_digests = entry["logits_fnv1a64"].as_array().unwrap();
    if ids.len() != expected.len() {
        return Some(format!("{} entries, C++ {}", ids.len(), expected.len()));
    }
    for (step, (id, digest)) in ids.iter().zip(digests).enumerate() {
        if *id != expected[step] {
            return Some(format!("step {step}: token {id}, C++ {}", expected[step]));
        }
        if digest != expected_digests[step].as_str().unwrap() {
            return Some(format!(
                "step {step}: logits {digest}, C++ {}",
                expected_digests[step]
            ));
        }
    }
    None
}

/// Next tokens, positions, and per session the trace's tokens and digests.
type Primed = (Vec<u32>, Vec<u32>, Vec<Vec<u32>>, Vec<Vec<String>>);

/// Prefills each session from a fresh state; the first trace entry.
fn prime(model: &mut Model, sessions: &mut [Session], prompts: &[(String, Vec<u32>)]) -> Primed {
    let width = sessions.len();
    let (mut next, mut positions) = (Vec::new(), Vec::new());
    let (mut ids, mut digests) = (vec![Vec::new(); width], vec![Vec::new(); width]);
    for (index, st) in sessions.iter_mut().enumerate() {
        let prompt = &prompts[index].1;
        model.reset(st).unwrap();
        let token = model.forward_prompt(st, prompt, 0, true).unwrap().unwrap();
        next.push(token);
        positions.push(prompt.len() as u32);
        ids[index].push(token);
        digests[index].push(fnv1a64(&model.logits(st).unwrap()));
    }
    (next, positions, ids, digests)
}

/// `steps` batched decode steps over all `sessions`.
fn batched_steps(
    model: &mut Model,
    sessions: &mut [Session],
    prompts: &[(String, Vec<u32>)],
    steps: usize,
) -> (Vec<Vec<u32>>, Vec<Vec<String>>) {
    let (mut next, mut positions, mut ids, mut digests) = prime(model, sessions, prompts);
    for _ in 0..steps {
        let mut items: Vec<(&mut Session, u32, u32)> = sessions
            .iter_mut()
            .zip(next.iter().zip(&positions))
            .map(|(st, (&token, &position))| (st, token, position))
            .collect();
        let out = model.forward_token_batch(&mut items).unwrap();
        drop(items);
        for (index, st) in sessions.iter().enumerate() {
            next[index] = out[index];
            positions[index] += 1;
            ids[index].push(out[index]);
            digests[index].push(fnv1a64(&model.logits(st).unwrap()));
        }
    }
    (ids, digests)
}

fn batched_decode_matches(quant: Quant) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let prefill = quant.fixture("prefill");
    let fixture = quant.fixture("batched");
    let max_context = u32::try_from(fixture["max_context"].as_u64().unwrap()).unwrap();
    let steps = fixture["steps"].as_u64().unwrap() as usize;
    let prompts = prompts(&prefill, &fixture);
    let mut model = Model::load(&quant.model(), max_context).unwrap();
    let mut sessions: Vec<Session> = (0..prompts.len())
        .map(|_| model.new_session().unwrap())
        .collect();
    let mut failures = Vec::new();

    // Each session alone on session 0, as the probe's main executor.
    let mut serial = Vec::new();
    for ((name, prompt), entry) in prompts.iter().zip(fixture["serial"].as_array().unwrap()) {
        assert_eq!(entry["session"].as_str().unwrap(), name);
        let st = &mut sessions[0];
        model.reset(st).unwrap();
        let mut token = model.forward_prompt(st, prompt, 0, true).unwrap().unwrap();
        let (mut ids, mut digests) = (Vec::new(), Vec::new());
        for pos in prompt.len() as u32..=(prompt.len() + steps) as u32 {
            ids.push(token);
            digests.push(fnv1a64(&model.logits(st).unwrap()));
            if ids.len() <= steps {
                token = model.forward_token(st, token, pos, true).unwrap().unwrap();
            }
        }
        if let Some(divergence) = divergence(entry, &ids, &digests) {
            failures.push(format!("{}: serial {name}: {divergence}", quant.set));
        }
        serial.push((ids, digests));
    }

    for entry in fixture["batched"].as_array().unwrap() {
        let width = entry["width"].as_u64().unwrap() as usize;
        assert_eq!(entry["matches_serial"], Value::Bool(true));
        let (ids, digests) = batched_steps(&mut model, &mut sessions[..width], &prompts, steps);
        for (index, row) in entry["sessions"].as_array().unwrap().iter().enumerate() {
            let name = &prompts[index].0;
            assert_eq!(row["session"].as_str().unwrap(), name);
            if let Some(divergence) = divergence(row, &ids[index], &digests[index]) {
                failures.push(format!("{}: width {width} {name}: {divergence}", quant.set));
            }
            if (&ids[index], &digests[index]) != (&serial[index].0, &serial[index].1) {
                failures.push(format!(
                    "{}: width {width} {name}: differs from serial",
                    quant.set
                ));
            }
        }
    }

    // Taps copy hidden rows without touching the arithmetic.
    let taps = [1, 9, 17, 25, 33, 37, 38, 39];
    for st in &mut sessions[..2] {
        st.set_hidden_capture(model.config(), true, &taps).unwrap();
    }
    let (ids, digests) = batched_steps(&mut model, &mut sessions[..2], &prompts, 4);
    for index in 0..2 {
        if ids[index][..] != serial[index].0[..5] || digests[index][..] != serial[index].1[..5] {
            failures.push(format!(
                "{}: width 2 with taps, {}: differs from serial",
                quant.set, prompts[index].0
            ));
        }
        assert_eq!(model.last_hidden(&sessions[index]).unwrap().len(), 8 * 2048);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn qwen36_35b_a3b_ud_q4_k_m() {
    batched_decode_matches(UD_Q4_K_M);
}

#[test]
fn qwen36_35b_a3b_q8_0() {
    batched_decode_matches(Q8_0);
}
