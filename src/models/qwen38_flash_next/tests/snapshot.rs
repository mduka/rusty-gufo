//! The snapshot QUALITY check of Qwen3.8 Flash-Next
//! (`tests/models/qwen38_flash_next/snapshot_test.cpp`) on the Rust engine:
//! a 4095-token context (a partial prefill batch and unpooled indexer rows
//! at the ring's end) restores into a fresh, a used and a persisted session
//! with the same sampled MTP continuation, as do a mid-decode snapshot and
//! one taken before a deferred residual; an extended context keeps its
//! prefix, and older payload versions, truncated payloads and too small
//! contexts are refused without disturbing the session:
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next --features gpu \
//!   --test snapshot -- --ignored --nocapture

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use std::time::Instant;

use common::{mtp_path, trunk_path};
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_qwen38_flash_next::engine::{
    Model, ModelOptions, SNAPSHOT_PAYLOAD_VERSION, Session, SpeculativeStats,
};
use gufo_qwen38_flash_next::executor::SessionMode;

const CONTEXT: u32 = 8192;
const TOKENS: usize = 48;

struct Decoded {
    tokens: Vec<i32>,
    stats: SpeculativeStats,
}

fn decode_with(
    model: &mut Model,
    session: &mut Session,
    count: usize,
    sampler: &mut SamplerState,
    budget: usize,
) -> Decoded {
    let before = session.statistics();
    let mut tokens = Vec::new();
    while tokens.len() < count {
        let step = model
            .decode_step(session, budget.min(count - tokens.len()), sampler, false)
            .unwrap();
        assert!(!step.tokens.is_empty(), "empty decode step");
        tokens.extend_from_slice(&step.tokens);
    }
    let after = session.statistics();
    Decoded {
        tokens,
        stats: SpeculativeStats {
            cycles: after.cycles - before.cycles,
            drafted: after.drafted - before.drafted,
            accepted: after.accepted - before.accepted,
            small_head_cycles: after.small_head_cycles - before.small_head_cycles,
            lookup: after.lookup - before.lookup,
            lookup_accepted: after.lookup_accepted - before.lookup_accepted,
        },
    }
}

/// Decodes `count` tokens past EOS with a fresh seeded sampler.
fn decode(
    model: &mut Model,
    session: &mut Session,
    count: usize,
    config: &SamplingConfig,
    budget: usize,
) -> Decoded {
    let history: Vec<u32> = session.tokens().iter().map(|&t| t as u32).collect();
    let mut sampler = SamplerState::new(config.clone(), &history).unwrap();
    decode_with(model, session, count, &mut sampler, budget)
}

fn same(expected: &Decoded, actual: &Decoded, what: &str) {
    assert_eq!(expected.tokens, actual.tokens, "{what}: tokens differ");
    assert!(
        expected.stats.drafted == actual.stats.drafted
            && expected.stats.accepted == actual.stats.accepted,
        "{what}: draft acceptance differs"
    );
}

fn bits(logits: &[f32]) -> Vec<u32> {
    logits.iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "QUALITY check on the Flash-Next artifact"]
fn snapshots_round_trip_exactly() {
    let mut model = Model::load(
        &trunk_path(),
        ModelOptions {
            max_context: CONTEXT,
            mtp_model_path: Some(mtp_path()),
            max_draft_tokens: 7,
            ..ModelOptions::default()
        },
    )
    .unwrap();
    let mode = if model.has_mtp() {
        SessionMode::Speculative
    } else {
        SessionMode::Autoregressive
    };
    let pattern = model.tokenize(
        "The quick brown fox jumps over the lazy dog. \
         Strix Halo executes this deterministic benchmark sequence. ",
    );
    // A partial prefill batch and three unpooled raw indexer rows at the end
    // of the ring; continuing (and speculative rollback) crosses its wrap.
    let prompt: Vec<i32> = (0..4095).map(|i| pattern[i % pattern.len()]).collect();
    let config = SamplingConfig {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.9,
        seed: 7,
        ..SamplingConfig::default()
    };
    let mut origin = model.create_session(mode, CONTEXT).unwrap();
    model.sync(&mut origin, &prompt).unwrap();
    let prompt_logits = origin.logits().to_vec();
    let start = Instant::now();
    let at_prompt = model.save_snapshot(&origin).unwrap();
    let save_ms = start.elapsed().as_secs_f64() * 1e3;
    assert_eq!(
        at_prompt.len() as u64,
        model.snapshot_bytes(&origin),
        "snapshot size differs from the estimate"
    );
    println!(
        "snapshot tokens={} bytes={} save_ms={save_ms:.1}",
        prompt.len(),
        at_prompt.len()
    );
    let expected = decode(&mut model, &mut origin, TOKENS, &config, 8);
    assert!(expected.stats.drafted > 0, "MTP did not draft");

    // Restore into a fresh session.
    let mut restored = model.create_session(mode, CONTEXT).unwrap();
    let start = Instant::now();
    model.restore_snapshot(&mut restored, &at_prompt).unwrap();
    println!("restore_ms={:.1}", start.elapsed().as_secs_f64() * 1e3);
    assert_eq!(restored.tokens(), &prompt[..], "restored tokens differ");
    assert_eq!(
        restored.position() as usize,
        prompt.len(),
        "restored position"
    );
    assert_eq!(
        bits(&prompt_logits),
        bits(restored.logits()),
        "restored logits differ"
    );
    same(
        &expected,
        &decode(&mut model, &mut restored, TOKENS, &config, 8),
        "fresh session restore",
    );

    // The persistent byte form restores the same way.
    let bytes = at_prompt.clone();
    let mut persisted = model.create_session(mode, CONTEXT).unwrap();
    model.restore_snapshot(&mut persisted, &bytes).unwrap();
    same(
        &expected,
        &decode(&mut model, &mut persisted, TOKENS, &config, 8),
        "persistent restore",
    );

    // Restoring over a session that already decoded discards its state.
    model.restore_snapshot(&mut origin, &at_prompt).unwrap();
    same(
        &expected,
        &decode(&mut model, &mut origin, TOKENS, &config, 8),
        "restore over a used session",
    );

    // Single-token decoding from a restored session also matches.
    model.restore_snapshot(&mut origin, &at_prompt).unwrap();
    let single = decode(&mut model, &mut origin, 16, &config, 1);
    model.restore_snapshot(&mut restored, &at_prompt).unwrap();
    assert_eq!(
        single.tokens,
        decode(&mut model, &mut restored, 16, &config, 1).tokens,
        "single-token decode differs after restore"
    );

    // A mid-decode snapshot carries the draft block's caught-up state and the
    // partial batch of kept trunk rows; the controller state is part of a
    // stochastic continuation's replay.
    model.restore_snapshot(&mut origin, &at_prompt).unwrap();
    let first_half = decode(&mut model, &mut origin, 16, &config, 8);
    let mid = model.save_snapshot(&origin).unwrap();
    let second_half = decode(&mut model, &mut origin, TOKENS, &config, 8);
    model.restore_snapshot(&mut restored, &mid).unwrap();
    assert_eq!(
        restored.position() as usize,
        prompt.len() + first_half.tokens.len(),
        "mid-decode position"
    );
    same(
        &second_half,
        &decode(&mut model, &mut restored, TOKENS, &config, 8),
        "mid-decode restore",
    );

    // A rejected proposal captured before its residual was evaluated: the
    // context snapshot and the request-owned sampler replay together.
    let mut pending = model.create_session(mode, CONTEXT).unwrap();
    let short = &prompt[..16];
    model.sync(&mut pending, short).unwrap();
    let history: Vec<u32> = short.iter().map(|&t| t as u32).collect();
    let mut sampler = SamplerState::new(
        SamplingConfig {
            temperature: 2.0,
            seed: 73,
            ..SamplingConfig::default()
        },
        &history,
    )
    .unwrap();
    let mut rejected = false;
    for _ in 0..32 {
        let step = model
            .decode_step(&mut pending, 2, &mut sampler, false)
            .unwrap();
        if step.tokens.len() == 1 {
            rejected = true;
            break;
        }
    }
    assert!(
        rejected,
        "snapshot test did not exercise a deferred residual"
    );
    let residual = model.save_snapshot(&pending).unwrap();
    let mut replay_sampler = sampler.clone();
    let residual_expected = decode_with(&mut model, &mut pending, 16, &mut sampler, 8);
    model.restore_snapshot(&mut restored, &residual).unwrap();
    same(
        &residual_expected,
        &decode_with(&mut model, &mut restored, 16, &mut replay_sampler, 8),
        "deferred residual restore",
    );
    assert_eq!(
        sampler.rng_state(),
        replay_sampler.rng_state(),
        "deferred residual restore changed RNG"
    );

    // Extending the restored context keeps the prefix.
    model.restore_snapshot(&mut restored, &at_prompt).unwrap();
    let mut extended = prompt.clone();
    extended.extend_from_slice(&expected.tokens);
    model.sync(&mut restored, &extended).unwrap();
    model.restore_snapshot(&mut origin, &at_prompt).unwrap();
    model.sync(&mut origin, &extended).unwrap();
    assert_eq!(
        bits(origin.logits()),
        bits(restored.logits()),
        "extension after restore differs"
    );

    // Cached state from older prefill arithmetic must be rebuilt.
    let mut incompatible = bytes.clone();
    incompatible[8..12].copy_from_slice(&(SNAPSHOT_PAYLOAD_VERSION - 1).to_le_bytes());
    assert!(
        model
            .restore_snapshot(&mut restored, &incompatible)
            .is_err(),
        "incompatible snapshot version accepted"
    );
    assert_eq!(
        restored.position() as usize,
        extended.len(),
        "incompatible snapshot disturbed the session"
    );
    // Rejections: a truncated payload, and a context too small to hold it.
    assert!(
        model
            .restore_snapshot(&mut restored, &bytes[..bytes.len() / 2])
            .is_err(),
        "truncated payload accepted"
    );
    assert_eq!(
        restored.position() as usize,
        extended.len(),
        "rejected payload disturbed the session"
    );
    let mut small = model.create_session(mode, 2048).unwrap();
    assert!(
        model.restore_snapshot(&mut small, &at_prompt).is_err(),
        "oversized snapshot accepted"
    );
    println!("snapshot round trips exact=1");
}
