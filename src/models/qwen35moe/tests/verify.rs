//! Speculative verification on the target side, without a draft model:
//! cargo test --profile gpu-test -p gufo-qwen35moe --features gpu --test verify
//!
//! Needs `GUFO_QWEN35MOE_MODEL` (UD-Q4_K_M) and `GUFO_QWEN35MOE_Q8_MODEL`
//! (Q8_0) and fails without them. A scripted draft proposes the known greedy
//! continuation of the `decode` fixture, with deliberate mistakes. Session A
//! verifies alone, ordinary blocks and wide chains of 16, 32 and 64 rows (the
//! replay ring grows from 16 to 64 slots) and a decode step captured into the
//! grown ring and replayed after a rollback; then A and a fresh session B
//! verify together (`ForwardVerificationBatch`, rings of 64 and 16 slots).
//! Every verified row whose inputs follow the greedy history has the logits
//! of the C++ `ForwardToken` at that position, bit for bit; rollback and
//! replay of the recurrent state are lossless, so the emitted tokens equal
//! the autoregressive ones and a decode step afterwards gives the C++
//! logits.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use common::{Q8_0, Quant, SERIAL, UD_Q4_K_M, fnv1a64, u32s};
use gufo_qwen35moe::{Model, Session, VerifyRequest};
use serde_json::Value;

const TAPS: [u32; 8] = [1, 9, 17, 25, 33, 37, 38, 39];

/// The decode fixture's greedy continuation and its logits digests.
struct Greedy {
    ids: Vec<u32>,
    digests: Vec<String>,
}

/// One session's place in the greedy continuation: `current` is
/// `ids[emitted - 1]`, to be run at `position`.
struct Cursor {
    emitted: usize,
    position: u32,
    current: u32,
}

/// The prompt token by token, as the fixture feeds it.
fn start(model: &mut Model, st: &mut Session, prompt: &[u32], greedy: &Greedy) -> Cursor {
    model.reset(st).unwrap();
    st.set_hidden_capture(model.config(), true, &TAPS).unwrap();
    let mut current = None;
    for (pos, &token) in (0u32..).zip(prompt) {
        current = model
            .forward_token(st, token, pos, pos as usize + 1 == prompt.len())
            .unwrap();
    }
    assert_eq!(current, Some(greedy.ids[0]));
    Cursor {
        emitted: 1,
        position: prompt.len() as u32,
        current: greedy.ids[0],
    }
}

/// The inputs of a step drafting `count` tokens, the first draft at index
/// `wrong` (if any) off by one.
fn inputs(greedy: &Greedy, cursor: &Cursor, count: usize, wrong: Option<usize>) -> Vec<u32> {
    let mut inputs = vec![cursor.current];
    inputs.extend(&greedy.ids[cursor.emitted..cursor.emitted + count]);
    if let Some(wrong) = wrong {
        inputs[1 + wrong] = (inputs[1 + wrong] + 1) % 248_000;
    }
    inputs
}

/// Checks a verified chunk, then commits its agreeing prefix and the
/// correction as the serving verifier does (`ProcessVerificationChunk`).
#[allow(clippy::too_many_arguments)]
fn finish(
    model: &mut Model,
    st: &mut Session,
    greedy: &Greedy,
    cursor: &mut Cursor,
    inputs: &[u32],
    predictions: &[u32],
    logits: &[f32],
    what: &str,
    failures: &mut Vec<String>,
) {
    let vocab = model.config().vocab_size as usize;
    assert_eq!(predictions.len(), inputs.len());
    assert_eq!(
        st.verification_hidden().len(),
        inputs.len() * TAPS.len() * 2048
    );
    let accepted = predictions
        .iter()
        .zip(&inputs[1..])
        .take_while(|(p, d)| p == d)
        .count();
    // Rows 0..=accepted follow the greedy history: decode-equivalent logits.
    for row in 0..=accepted {
        let digest = fnv1a64(&logits[row * vocab..(row + 1) * vocab]);
        if digest != greedy.digests[cursor.emitted + row] {
            failures.push(format!(
                "{what}: row {row} at {}: logits {digest}, C++ {}",
                cursor.position + row as u32,
                greedy.digests[cursor.emitted + row]
            ));
            break;
        }
    }
    let correction = predictions[accepted];
    model.finish_verification(st).unwrap();
    if accepted + 1 < inputs.len() {
        model.restore_state(st).unwrap();
        model
            .commit_verified(st, &inputs[..=accepted], cursor.position)
            .unwrap();
    }
    let mut out = inputs[1..=accepted].to_vec();
    out.push(correction);
    let expected = &greedy.ids[cursor.emitted..cursor.emitted + out.len()];
    if out != expected {
        failures.push(format!("{what}: emitted {out:?}, greedy {expected:?}"));
    }
    cursor.position += out.len() as u32;
    cursor.emitted += out.len();
    cursor.current = correction;
}

/// One single-session step drafting `count` tokens.
fn single(
    model: &mut Model,
    st: &mut Session,
    greedy: &Greedy,
    cursor: &mut Cursor,
    (count, wrong): (usize, Option<usize>),
    failures: &mut Vec<String>,
) {
    let inputs = inputs(greedy, cursor, count, wrong);
    model.save_state(st, cursor.position).unwrap();
    let verified = model
        .verify_batch(&mut [VerifyRequest {
            session: &mut *st,
            tokens: &inputs,
            position: cursor.position,
            capture_logits: true,
        }])
        .unwrap()
        .remove(0);
    let what = format!("single {} rows at {}", inputs.len(), cursor.position);
    finish(
        model,
        st,
        greedy,
        cursor,
        &inputs,
        &verified.tokens,
        &verified.logits,
        &what,
        failures,
    );
}

/// A step of two sessions verified in one pass.
fn pair(
    model: &mut Model,
    (a, b): (&mut Session, &mut Session),
    greedy: &Greedy,
    (ca, cb): (&mut Cursor, &mut Cursor),
    (sa, sb): ((usize, Option<usize>), (usize, Option<usize>)),
    failures: &mut Vec<String>,
) {
    let (ia, ib) = (
        inputs(greedy, ca, sa.0, sa.1),
        inputs(greedy, cb, sb.0, sb.1),
    );
    model.save_state(a, ca.position).unwrap();
    model.save_state(b, cb.position).unwrap();
    let mut verified = model
        .verify_batch(&mut [
            VerifyRequest {
                session: &mut *a,
                tokens: &ia,
                position: ca.position,
                capture_logits: true,
            },
            VerifyRequest {
                session: &mut *b,
                tokens: &ib,
                position: cb.position,
                capture_logits: true,
            },
        ])
        .unwrap();
    let (vb, va) = (verified.remove(1), verified.remove(0));
    let what = format!("pair A {} rows at {}", ia.len(), ca.position);
    finish(
        model, a, greedy, ca, &ia, &va.tokens, &va.logits, &what, failures,
    );
    let what = format!("pair B {} rows at {}", ib.len(), cb.position);
    finish(
        model, b, greedy, cb, &ib, &vb.tokens, &vb.logits, &what, failures,
    );
}

/// A decode step under replay capture, rolled back and replayed
/// (`VerifySequentialStep`): the decode records its recurrence inputs into
/// the ring at the ring's capacity, and the replay reads them back.
fn captured_decode(
    model: &mut Model,
    st: &mut Session,
    greedy: &Greedy,
    cursor: &mut Cursor,
    what: &str,
    failures: &mut Vec<String>,
) {
    model.save_state(st, cursor.position).unwrap();
    let next = model
        .forward_token(st, cursor.current, cursor.position, true)
        .unwrap();
    let digest = fnv1a64(&model.logits(st).unwrap());
    if next != Some(greedy.ids[cursor.emitted]) || digest != greedy.digests[cursor.emitted] {
        failures.push(format!("{what}: captured decode differs from C++"));
    }
    model.finish_verification(st).unwrap();
    model.restore_state(st).unwrap();
    model
        .commit_verified(st, &[cursor.current], cursor.position)
        .unwrap();
    cursor.emitted += 1;
    cursor.position += 1;
    cursor.current = greedy.ids[cursor.emitted - 1];
}

/// A decode step from the cursor gives the fixture's next token and logits.
fn decode_matches(
    model: &mut Model,
    st: &mut Session,
    greedy: &Greedy,
    cursor: &Cursor,
    what: &str,
    failures: &mut Vec<String>,
) {
    let next = model
        .forward_token(st, cursor.current, cursor.position, true)
        .unwrap();
    let digest = fnv1a64(&model.logits(st).unwrap());
    if next != Some(greedy.ids[cursor.emitted]) || digest != greedy.digests[cursor.emitted] {
        failures.push(format!(
            "{what}: decode after {} tokens: {next:?} {digest}, C++ {} {}",
            cursor.emitted, greedy.ids[cursor.emitted], greedy.digests[cursor.emitted]
        ));
    }
}

fn rolled_back_verification_is_lossless(quant: Quant) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = quant.fixture("decode");
    let max_context = u32::try_from(fixture["max_context"].as_u64().unwrap()).unwrap();
    let prompt = u32s(&fixture["prompt"]);
    let greedy = Greedy {
        ids: u32s(&fixture["decode"]["ids"]),
        digests: fixture["decode"]["logits_fnv1a64"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_str().unwrap().to_string())
            .collect::<Vec<_>>(),
    };
    assert_eq!(fixture["decode"]["ids"], Value::from(greedy.ids.clone()));
    let mut model = Model::load(&quant.model(), max_context).unwrap();
    let (mut a, mut b) = (model.new_session().unwrap(), model.new_session().unwrap());
    let mut failures = Vec::new();

    // A alone: ordinary blocks, then wide chains that grow the replay ring
    // (16, 32, 64 rows), rolled back inside and beyond the old capacity, and
    // a decode step captured into and replayed from the 64-slot ring.
    let e = quant.set;
    let mut ca = start(&mut model, &mut a, &prompt, &greedy);
    for script in [
        (3, Some(1)),
        (7, None),
        (15, Some(11)),
        (5, Some(0)),
        (31, None),
        (63, Some(40)),
    ] {
        single(&mut model, &mut a, &greedy, &mut ca, script, &mut failures);
    }
    assert_eq!(a.replay_capacity(), 64);
    let what = format!("{e} A");
    captured_decode(&mut model, &mut a, &greedy, &mut ca, &what, &mut failures);
    for script in [(2, Some(1)), (7, Some(6))] {
        single(&mut model, &mut a, &greedy, &mut ca, script, &mut failures);
    }
    decode_matches(
        &mut model,
        &mut a,
        &greedy,
        &ca,
        &format!("{e} A"),
        &mut failures,
    );
    ca.emitted += 1;
    ca.position += 1;
    ca.current = greedy.ids[ca.emitted - 1];

    // A and a fresh B together: up to eight rows each, at other positions.
    let mut cb = start(&mut model, &mut b, &prompt, &greedy);
    for (sa, sb) in [
        ((7, Some(3)), (7, None)),
        ((1, None), (4, Some(2))),
        ((6, None), (7, Some(0))),
        ((3, Some(2)), (5, None)),
    ] {
        pair(
            &mut model,
            (&mut a, &mut b),
            &greedy,
            (&mut ca, &mut cb),
            (sa, sb),
            &mut failures,
        );
    }
    decode_matches(
        &mut model,
        &mut a,
        &greedy,
        &ca,
        &format!("{e} A"),
        &mut failures,
    );
    decode_matches(
        &mut model,
        &mut b,
        &greedy,
        &cb,
        &format!("{e} B"),
        &mut failures,
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn qwen36_35b_a3b_ud_q4_k_m() {
    rolled_back_verification_is_lossless(UD_Q4_K_M);
}

#[test]
fn qwen36_35b_a3b_q8_0() {
    rolled_back_verification_is_lossless(Q8_0);
}
