//! MTP drafts over the request vocabulary: greedy
//! speculative decoding through the small head gives the autoregressive
//! tokens, English text stays on the small head while text that keeps
//! bringing new tokens above the prefix falls back to the full one, sampled
//! default-preset continuations replay exactly after a snapshot (in RAM and
//! serialized; the table rebuilds from the restored tokens), and a batch
//! that mixes both heads decodes what its sessions decode alone, and a
//! greedy request decoding alone on two sessions uses the small head:
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next --features gpu --test draft_vocab

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use common::{mtp_path, trunk_path};
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_qwen38_flash_next::draft_vocab::DRAFT_VOCAB;
use gufo_qwen38_flash_next::engine::{DecodeRequest, Model, ModelOptions, Session};
use gufo_qwen38_flash_next::executor::SessionMode;

const CONTEXT: u32 = 4096;
const ENGLISH: &str = "Write a detailed story about a traveller finding a lost map in a \
                       library. Include a puzzle and its solution.";
const THAI: &str = "เล่าเรื่องสั้นภาษาไทยเกี่ยวกับแมวสีส้มที่พบแผนที่\
                    ในห้องสมุดและออกเดินทางตามหาเมืองที่หายไป";

fn load(concurrency: u32) -> Model {
    Model::load(
        &trunk_path(),
        ModelOptions {
            max_context: CONTEXT,
            mtp_model_path: Some(mtp_path()),
            decode_concurrency: concurrency,
            ..ModelOptions::default()
        },
    )
    .unwrap()
}

/// What a session decoded: its tokens, tokens per step, the sampler's RNG
/// state and the draft statistics of those steps.
#[derive(Debug, PartialEq)]
struct Replay {
    tokens: Vec<i32>,
    widths: Vec<usize>,
    rng: u64,
    drafted: u64,
    accepted: u64,
    small_head_cycles: u64,
}

fn decode(
    model: &mut Model,
    session: &mut Session,
    sampler: &mut SamplerState,
    count: usize,
) -> Replay {
    let before = session.statistics();
    let (mut tokens, mut widths) = (Vec::new(), Vec::new());
    while tokens.len() < count {
        let step = model
            .decode_step(session, count - tokens.len(), sampler, false)
            .unwrap();
        assert!(!step.tokens.is_empty(), "the decode made no progress");
        widths.push(step.tokens.len());
        tokens.extend_from_slice(&step.tokens);
    }
    let after = session.statistics();
    Replay {
        tokens,
        widths,
        rng: sampler.rng_state(),
        drafted: after.drafted - before.drafted,
        accepted: after.accepted - before.accepted,
        small_head_cycles: after.small_head_cycles - before.small_head_cycles,
    }
}

fn synced(model: &mut Model, mode: SessionMode, prompt: &[i32]) -> Session {
    let mut session = model.create_session(mode, CONTEXT).unwrap();
    model.sync(&mut session, prompt).unwrap();
    session
}

fn greedy() -> SamplerState {
    SamplerState::new(SamplingConfig::default(), &[]).unwrap()
}

/// Greedy MTP decoding verifies every draft over the full vocabulary: the
/// small head changes proposals, never tokens.
fn greedy_output_is_unchanged(model: &mut Model, english: &[i32], thai: &[i32]) {
    for (name, prompt) in [("english", english), ("thai", thai)] {
        let mut ar = synced(model, SessionMode::Autoregressive, prompt);
        let expected = decode(model, &mut ar, &mut greedy(), 96);
        let mut mtp = synced(model, SessionMode::Speculative, prompt);
        let drafted = decode(model, &mut mtp, &mut greedy(), 96);
        assert_eq!(
            drafted.tokens, expected.tokens,
            "{name}: speculative greedy tokens differ from autoregressive ones"
        );
        let cycles = drafted.widths.len() as u64;
        println!(
            "{name}: {} cycles, {} on the small head, {} of {} drafts accepted",
            cycles, drafted.small_head_cycles, drafted.accepted, drafted.drafted
        );
        if name == "english" {
            assert!(
                drafted.small_head_cycles * 2 > cycles,
                "english text left the small head"
            );
        }
    }
}

/// A sampled continuation after a snapshot replays the uninterrupted one
/// exactly, from a RAM snapshot and from its bytes: the restored session
/// rebuilds the draft table from its tokens.
fn snapshots_replay_drafts(model: &mut Model, english: &[i32], thai: &[i32]) {
    let presets = [
        (
            "thinking-default",
            SamplingConfig {
                temperature: 1.0,
                top_k: 20,
                top_p: 0.95,
                seed: 1,
                ..SamplingConfig::default()
            },
        ),
        (
            "nonthinking-default",
            SamplingConfig {
                temperature: 0.7,
                top_k: 20,
                top_p: 0.8,
                seed: 1,
                presence_penalty: 1.5,
                ..SamplingConfig::default()
            },
        ),
    ];
    for (preset, config) in presets {
        for (name, prompt) in [("english", english), ("thai", thai)] {
            let mut live = synced(model, SessionMode::Speculative, prompt);
            let history: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();
            let mut sampler = SamplerState::new(config.clone(), &history).unwrap();
            let trained = decode(model, &mut live, &mut sampler, 128);
            assert!(trained.drafted > 0, "{preset} {name}: no drafts");
            let snapshot = model.snapshot(&live).unwrap();
            let bytes = model.save_snapshot(&live).unwrap();
            let saved = sampler.clone();
            let expected = decode(model, &mut live, &mut sampler, 64);
            let logits = live.logits().to_vec();
            for serialized in [false, true] {
                let mut restored = model
                    .create_session(SessionMode::Speculative, CONTEXT)
                    .unwrap();
                if serialized {
                    model.restore_snapshot(&mut restored, &bytes).unwrap();
                } else {
                    model.restore(&mut restored, &snapshot).unwrap();
                }
                let mut replay = saved.clone();
                assert_eq!(
                    decode(model, &mut restored, &mut replay, 64),
                    expected,
                    "{preset} {name}: restored drafts differ (serialized: {serialized})"
                );
                assert!(
                    restored.logits() == logits.as_slice(),
                    "{preset} {name}: restored target logits differ"
                );
            }
            println!(
                "{preset} {name}: {} small-head cycles of {}, replay exact",
                expected.small_head_cycles,
                expected.widths.len()
            );
        }
    }
}

/// Batches whose sessions use different heads decode what each session
/// decodes alone (greedy, so the batch policy changes rounds, not tokens).
fn mixed_batches_decode_alone(english: &[i32], thai: &[i32]) {
    let prompts = [english, thai, english, thai];
    let mut alone = Vec::new();
    {
        let mut model = load(1);
        for prompt in prompts {
            let mut session = synced(&mut model, SessionMode::Speculative, prompt);
            alone.push(decode(&mut model, &mut session, &mut greedy(), 48).tokens);
        }
    }
    let mut model = load(prompts.len() as u32);
    let mut sessions: Vec<Session> = prompts
        .iter()
        .map(|p| synced(&mut model, SessionMode::Speculative, p))
        .collect();
    let mut samplers: Vec<SamplerState> = prompts.iter().map(|_| greedy()).collect();
    let mut tokens = vec![Vec::new(); prompts.len()];
    while tokens.iter().any(|t| t.len() < 48) {
        let mut requests: Vec<DecodeRequest<'_>> = sessions
            .iter_mut()
            .zip(samplers.iter_mut())
            .map(|(session, sampler)| DecodeRequest {
                session,
                max_tokens: 8,
                sampler,
                stop_at_eos: false,
            })
            .collect();
        for (i, result) in model.decode_batch(&mut requests).into_iter().enumerate() {
            tokens[i].extend_from_slice(&result.unwrap().tokens);
        }
    }
    for (i, (batched, alone)) in tokens.iter().zip(&alone).enumerate() {
        assert_eq!(
            &batched[..48],
            &alone[..],
            "session {i}: batched tokens differ"
        );
    }
    let small: Vec<u64> = sessions
        .iter()
        .map(|s| s.statistics().small_head_cycles)
        .collect();
    println!("mixed batch: small-head cycles per session {small:?}, tokens as alone");
}

/// On two sessions a greedy request decoding alone drafts through the small
/// head as on one; beside a sampled peer, in an unplanned batch, it keeps the
/// full head, and its text is the same either way.
fn lone_greedy_uses_the_small_head(english: &[i32]) {
    let mut model = load(2);
    let mut alone = synced(&mut model, SessionMode::Speculative, english);
    let before = alone.statistics();
    let solo = decode(&mut model, &mut alone, &mut greedy(), 48);
    let lookup = alone.statistics().lookup - before.lookup;
    assert!(
        solo.drafted > lookup && solo.small_head_cycles > 0,
        "a lone greedy request kept the full head on two sessions"
    );
    let mut session = synced(&mut model, SessionMode::Speculative, english);
    let mut peer = synced(&mut model, SessionMode::Speculative, english);
    let history: Vec<u32> = english.iter().map(|&t| t as u32).collect();
    let mut sampler = greedy();
    let mut peer_sampler = SamplerState::new(
        SamplingConfig {
            temperature: 0.8,
            seed: 1,
            ..SamplingConfig::default()
        },
        &history,
    )
    .unwrap();
    let before = session.statistics();
    let mut tokens = Vec::new();
    while tokens.len() < solo.tokens.len() {
        let mut requests = vec![
            DecodeRequest {
                session: &mut session,
                max_tokens: 8,
                sampler: &mut sampler,
                stop_at_eos: false,
            },
            DecodeRequest {
                session: &mut peer,
                max_tokens: 8,
                sampler: &mut peer_sampler,
                stop_at_eos: false,
            },
        ];
        let result = model.decode_batch(&mut requests).remove(0).unwrap();
        assert!(!result.tokens.is_empty(), "the mixed step made no progress");
        tokens.extend_from_slice(&result.tokens);
    }
    let after = session.statistics();
    assert!(
        after.drafted - before.drafted > after.lookup - before.lookup
            && after.small_head_cycles == before.small_head_cycles,
        "greedy reached the small head beside a sampled peer"
    );
    assert_eq!(
        &tokens[..solo.tokens.len()],
        &solo.tokens[..],
        "the small head changed greedy text"
    );
    println!(
        "lone greedy on two sessions: {} small-head cycles of {}, text exact beside a sampled peer",
        solo.small_head_cycles,
        solo.widths.len()
    );
}

#[test]
fn drafts_over_the_request_vocabulary() {
    if !trunk_path().exists() || !mtp_path().exists() {
        panic!("the Flash-Next artifacts are missing");
    }
    let (english, thai) = {
        let mut model = load(1);
        let english = model.tokenize(ENGLISH);
        let thai = model.tokenize(THAI);
        assert!(
            thai.iter().any(|&t| t as u32 >= DRAFT_VOCAB),
            "the Thai prompt needs tokens above the draft prefix"
        );
        greedy_output_is_unchanged(&mut model, &english, &thai);
        snapshots_replay_drafts(&mut model, &english, &thai);
        (english, thai)
    };
    mixed_batches_decode_alone(&english, &thai);
    lone_greedy_uses_the_small_head(&english);
}
