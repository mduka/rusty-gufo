//! The in-pass prefill checkpoint (`CheckPrefillCheckpoint` of `snapshot_test.cpp`): a pass that captures a
//! stable boundary near its end leaves the same logits and sampled
//! continuation as two passes split there, its checkpoint has the size its
//! admission claim named and the bytes of a session synced to exactly the
//! boundary, and a restored checkpoint resumes exactly. Autoregressive and
//! MTP sessions, cold and warm, a boundary in the first or the second pass,
//! and one at the end of the prompt:
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next --features gpu --test checkpoint
//!
//! The overflow and short cases of the C++ check run with `--ignored`.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use common::{mtp_path, trunk_path};
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_qwen38_flash_next::engine::{Model, ModelOptions, Session, SpeculativeStats};
use gufo_qwen38_flash_next::executor::SessionMode;

const CONTEXT: u32 = 8192;

fn load() -> Model {
    Model::load(
        &trunk_path(),
        ModelOptions {
            max_context: CONTEXT,
            mtp_model_path: Some(mtp_path()),
            max_draft_tokens: 7,
            ..ModelOptions::default()
        },
    )
    .unwrap()
}

/// The 4095-token prompt of the snapshot check.
fn prompt(model: &Model) -> Vec<i32> {
    let pattern = model.tokenize(
        "The quick brown fox jumps over the lazy dog. \
         Strix Halo executes this deterministic benchmark sequence. ",
    );
    (0..4095).map(|i| pattern[i % pattern.len()]).collect()
}

fn decode(model: &mut Model, session: &mut Session, count: usize) -> (Vec<i32>, SpeculativeStats) {
    let config = SamplingConfig {
        temperature: 0.8,
        top_k: 20,
        top_p: 0.95,
        seed: 73,
        ..SamplingConfig::default()
    };
    let history: Vec<u32> = session.tokens().iter().map(|&t| t as u32).collect();
    let mut sampler = SamplerState::new(config, &history).unwrap();
    let before = session.statistics();
    let mut tokens = Vec::new();
    while tokens.len() < count {
        let step = model
            .decode_step(session, 8.min(count - tokens.len()), &mut sampler, false)
            .unwrap();
        assert!(!step.tokens.is_empty(), "empty decode step");
        tokens.extend_from_slice(&step.tokens);
    }
    let after = session.statistics();
    (
        tokens,
        SpeculativeStats {
            cycles: after.cycles - before.cycles,
            drafted: after.drafted - before.drafted,
            accepted: after.accepted - before.accepted,
            small_head_cycles: after.small_head_cycles - before.small_head_cycles,
            lookup: after.lookup - before.lookup,
            lookup_accepted: after.lookup_accepted - before.lookup_accepted,
        },
    )
}

fn bits(logits: &[f32]) -> Vec<u32> {
    logits.iter().map(|v| v.to_bits()).collect()
}

/// `CheckPrefillCheckpoint`: `warm` tokens synced first, the boundary
/// `suffix` tokens before the end.
fn check(model: &mut Model, prompt: &[i32], mode: SessionMode, warm: usize, suffix: usize) {
    let what = format!(
        "mode {mode:?} tokens {} warm {warm} suffix {suffix}",
        prompt.len()
    );
    let mut source = model.create_session(mode, CONTEXT).unwrap();
    let mut whole = model.create_session(mode, CONTEXT).unwrap();
    let mut split = model.create_session(mode, CONTEXT).unwrap();
    if warm != 0 {
        for session in [&mut source, &mut whole, &mut split] {
            model.sync(session, &prompt[..warm]).unwrap();
        }
    }
    let boundary = prompt.len() - suffix;
    let claim = model.prefill_checkpoint_bytes(&source, boundary as u32);
    let (captured, capture_ms) = model.sync_through(&mut source, prompt, boundary).unwrap();
    assert!(
        captured.is_borrowed(),
        "{what}: the checkpoint was not borrowed"
    );
    let checkpoint = captured.to_bytes().unwrap();
    assert!(
        capture_ms.is_finite() && capture_ms > 0.0,
        "{what}: in-pass checkpoint capture time was not reported"
    );
    assert_eq!(
        checkpoint.len() as u64,
        claim,
        "{what}: in-pass checkpoint differs from its admission claim"
    );
    model.sync(&mut whole, &prompt[..boundary]).unwrap();
    model.sync(&mut whole, prompt).unwrap();
    assert!(
        bits(source.logits()) == bits(whole.logits()),
        "{what}: in-pass checkpoint changed split-pass logits"
    );
    let continued = decode(model, &mut source, 8);
    let expected = decode(model, &mut whole, 8);
    assert!(
        continued.0 == expected.0
            && continued.1.drafted == expected.1.drafted
            && continued.1.accepted == expected.1.accepted,
        "{what}: in-pass sampled continuation differs"
    );
    assert!(
        bits(source.logits()) == bits(whole.logits()),
        "{what}: in-pass sampled continuation changed logits"
    );
    model.sync(&mut split, &prompt[..boundary]).unwrap();
    let ending = model.save_snapshot(&split).unwrap();
    assert!(
        checkpoint == ending,
        "{what}: in-pass boundary differs from a pass ending there"
    );
    model.reset(&mut source);
    model.restore(&mut source, &captured).unwrap();
    assert!(
        model.save_snapshot(&source).unwrap() == checkpoint,
        "{what}: in-pass checkpoint did not survive source reset"
    );
    let mut raw = model.create_session(mode, CONTEXT).unwrap();
    model.restore_snapshot(&mut raw, &checkpoint).unwrap();
    model.sync(&mut raw, prompt).unwrap();
    model.sync(&mut source, prompt).unwrap();
    let resumed = model.save_snapshot(&source).unwrap();
    assert!(
        model.save_snapshot(&raw).unwrap() == resumed,
        "{what}: borrowed and serialized boundaries resumed differently"
    );
    model.sync(&mut split, prompt).unwrap();
    assert!(
        model.save_snapshot(&split).unwrap() == resumed,
        "{what}: restored boundary did not resume exactly"
    );
    println!("in-pass checkpoint {what}: restore exact, capture {capture_ms:.1} ms");
}

#[test]
fn in_pass_checkpoint_equals_a_pass_ending_there() {
    let mut model = load();
    let prompt = prompt(&model);
    let mut warm = prompt.clone();
    warm.extend_from_slice(&prompt[..33]);
    let mut overflow = prompt.clone();
    overflow.extend_from_slice(&prompt[..2051]);
    for mode in [SessionMode::Autoregressive, SessionMode::Speculative] {
        // The boundary in one pass across the indexer budget, cold and warm.
        check(&mut model, &prompt, mode, 0, 7);
        check(&mut model, &warm, mode, prompt.len(), 7);
    }
    // The boundary in the second of two passes, one pass of 2,055 tokens,
    // and a boundary at the prompt's end.
    check(&mut model, &overflow, SessionMode::Autoregressive, 0, 7);
    check(&mut model, &warm[..2055], SessionMode::Speculative, 0, 7);
    check(&mut model, &prompt, SessionMode::Autoregressive, 0, 0);
}

#[test]
#[ignore = "QUALITY check on the Flash-Next artifact"]
fn in_pass_checkpoint_covers_every_shape() {
    let mut model = load();
    let prompt = prompt(&model);
    let mut overflow = prompt.clone();
    overflow.extend_from_slice(&prompt[..2051]);
    let mut warm = prompt.clone();
    warm.extend_from_slice(&prompt[..33]);
    for mode in [SessionMode::Autoregressive, SessionMode::Speculative] {
        check(&mut model, &prompt, mode, 0, 7);
        check(&mut model, &warm, mode, prompt.len(), 7);
        check(&mut model, &overflow, mode, prompt.len(), 7);
        check(&mut model, &prompt[..2040], mode, 0, 7);
        check(&mut model, &prompt, mode, 0, 0);
        check(&mut model, &warm[..2055], mode, 0, 7);
    }
}
