//! The session engine against recorded sessions: prompt sync, MTP decode
//! steps (greedy with device verification, and sampled with rejection) and
//! snapshots, one session and batches of several, in the layout of
//! `tests/reference/cpp_session_dump.cpp`: the tokens and
//! tokens per step, the last logits bit for bit, the draft statistics and
//! the snapshot bytes after the prompt and after decoding:
//! cargo test -p gufo-qwen38-flash-next --features gpu --test session
//!
//! `GUFO_FLASH_NEXT_FIXTURES` names the directory with `prompt*.bin` and the
//! tool's output directories.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use std::path::{Path, PathBuf};

use common::{mtp_path, trunk_path};
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_qwen38_flash_next::engine::{
    DecodeRequest, Model, ModelOptions, SNAPSHOT_PAYLOAD_VERSION, Session,
};
use gufo_qwen38_flash_next::executor::SessionMode;
use gufo_qwen38_flash_next::lookup::{LookupState, NO_STATE};

fn fixtures() -> PathBuf {
    std::env::var_os("GUFO_FLASH_NEXT_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../target/flash-next"),
        PathBuf::from,
    )
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}; run tests/reference/cpp_session_dump.cpp",
            path.display()
        )
    })
}

fn words(path: &Path) -> Vec<u32> {
    read(path)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

/// A C++ fixture snapshot as this engine writes it: the payload version
/// rose from 16 to 17 for the draft head, whose tables are not part of the
/// payload, and to 18 with the prompt-lookup state after the header.
fn cpp_snapshot(path: &Path) -> Vec<u8> {
    let mut bytes = read(path);
    if u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) == SNAPSHOT_PAYLOAD_VERSION {
        return bytes;
    }
    assert_eq!(
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        16,
        "{}: fixture payload version",
        path.display()
    );
    bytes[8..12].copy_from_slice(&SNAPSHOT_PAYLOAD_VERSION.to_le_bytes());
    // Version 18 appends the prompt-lookup state to
    // the 120-byte header: no resume point and no request boundary.
    bytes.splice(120..120, [0xff; 8]);
    bytes
}

/// Names the first differing byte of two snapshots.
fn same_bytes(what: &str, rust: &[u8], cpp: &[u8]) {
    if rust == cpp {
        return;
    }
    let first = rust.iter().zip(cpp).position(|(a, b)| a != b);
    let count = rust.iter().zip(cpp).filter(|(a, b)| a != b).count();
    panic!(
        "{what}: {} vs {} bytes, {count} differ, first at {first:?}",
        rust.len(),
        cpp.len()
    );
}

pub fn load(concurrency: u32) -> Model {
    Model::load(
        &trunk_path(),
        ModelOptions {
            max_context: 4096,
            mtp_model_path: Some(mtp_path()),
            decode_concurrency: concurrency,
            ..ModelOptions::default()
        },
    )
    .unwrap()
}

pub fn sampler(temperature: f32, seed: i64) -> SamplerState {
    let config = if temperature > 0.0 {
        SamplingConfig {
            temperature,
            top_k: 40,
            seed,
            ..SamplingConfig::default()
        }
    } else {
        SamplingConfig::default()
    };
    SamplerState::new(config, &[]).unwrap()
}

/// What a session produced, compared with fixture session `i` of `dir`.
pub struct Outcome<'a> {
    pub tokens: Vec<u32>,
    pub steps: Vec<u32>,
    pub session: &'a Session,
}

pub fn compare(model: &Model, dir: &Path, i: usize, out: &Outcome<'_>) {
    let name = |what: &str| dir.join(format!("{what}_{i}.bin"));
    let label = format!("{} session {i}", dir.display());
    assert_eq!(out.tokens, words(&name("tokens")), "{label}: tokens");
    assert_eq!(out.steps, words(&name("steps")), "{label}: tokens per step");
    let logits: Vec<u32> = out.session.logits().iter().map(|v| v.to_bits()).collect();
    assert!(
        logits == words(&name("logits")),
        "{label}: last logits differ"
    );
    let stats = out.session.statistics();
    let expected: Vec<u64> = read(&name("stats"))
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| u64::from_le_bytes(*b))
        .collect();
    assert_eq!(
        vec![stats.cycles, stats.drafted, stats.accepted],
        expected,
        "{label}: cycles, drafted, accepted"
    );
    // The C++ fixtures predate the prompt-lookup state; after decoding
    // Rust's holds the request boundary, which a later step bounds matches by.
    let rust = model.save_snapshot(out.session).unwrap();
    let mut cpp = cpp_snapshot(&name("snapshot"));
    let lookup = LookupState::read(&rust[120..128]);
    assert!(
        lookup.resume == NO_STATE && lookup.request_start != NO_STATE,
        "{label}: prompt-lookup state after decoding"
    );
    cpp[120..128].copy_from_slice(&rust[120..128]);
    same_bytes(&format!("{label}: snapshot"), &rust, &cpp);
}

fn run_single(
    model: &mut Model,
    name: &str,
    prompt: &str,
    steps: usize,
    temperature: f32,
    seed: i64,
) {
    let dir = fixtures().join(name);
    let prompt: Vec<i32> = words(&fixtures().join(prompt))
        .into_iter()
        .map(|t| t as i32)
        .collect();
    let mut session = model
        .create_session(SessionMode::Speculative, 4096)
        .unwrap();
    model.sync(&mut session, &prompt).unwrap();
    let rust = model.save_snapshot(&session).unwrap();
    same_bytes(
        &format!("{name}: snapshot after the prompt"),
        &rust,
        &cpp_snapshot(&dir.join("sync_0.bin")),
    );
    let mut sampler = sampler(temperature, seed);
    let (mut tokens, mut counts) = (Vec::new(), Vec::new());
    for _ in 0..steps {
        let result = model
            .decode_step(&mut session, 8, &mut sampler, false)
            .unwrap();
        counts.push(result.tokens.len() as u32);
        tokens.extend(result.tokens.iter().map(|&t| t as u32));
    }
    compare(
        model,
        &dir,
        0,
        &Outcome {
            tokens,
            steps: counts,
            session: &session,
        },
    );
    // The final snapshot restores into a fresh session and decodes on as
    // the original does.
    let snapshot = model.save_snapshot(&session).unwrap();
    let mut restored = model
        .create_session(SessionMode::Speculative, 4096)
        .unwrap();
    model.restore_snapshot(&mut restored, &snapshot).unwrap();
    same_bytes(
        &format!("{name}: restored snapshot"),
        &model.save_snapshot(&restored).unwrap(),
        &snapshot,
    );
    let mut a = sampler.clone();
    let first = model.decode_step(&mut session, 8, &mut a, false).unwrap();
    let second = model
        .decode_step(&mut restored, 8, &mut sampler, false)
        .unwrap();
    assert_eq!(first, second, "{name}: decode after restore");
}

/// `k` sessions on the first `len - 7 i` prompt tokens decoded together for
/// `steps` rounds, as the tool does. Sampled batches follow each session's
/// own length policy and replay exactly; greedy batches choose their width
/// from measured cycle times, which changes the rounds but not the tokens.
fn run_batch(model: &mut Model, name: &str, k: usize, steps: usize, temperature: f32, seed: i64) {
    let dir = fixtures().join(name);
    let prompt: Vec<i32> = words(&fixtures().join("prompt50.bin"))
        .into_iter()
        .map(|t| t as i32)
        .collect();
    let mut sessions = Vec::new();
    let mut samplers = Vec::new();
    for i in 0..k {
        let mut session = model
            .create_session(SessionMode::Speculative, 4096)
            .unwrap();
        model
            .sync(&mut session, &prompt[..prompt.len() - 7 * i])
            .unwrap();
        let rust = model.save_snapshot(&session).unwrap();
        same_bytes(
            &format!("{name} session {i}: snapshot after the prompt"),
            &rust,
            &cpp_snapshot(&dir.join(format!("sync_{i}.bin"))),
        );
        sessions.push(session);
        samplers.push(sampler(temperature, seed + i as i64));
    }
    let mut tokens = vec![Vec::new(); k];
    let mut counts = vec![Vec::new(); k];
    for _ in 0..steps {
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
        let results = model.decode_batch(&mut requests);
        for (i, result) in results.into_iter().enumerate() {
            let result = result.unwrap();
            counts[i].push(result.tokens.len() as u32);
            tokens[i].extend(result.tokens.iter().map(|&t| t as u32));
        }
    }
    for i in 0..k {
        if temperature > 0.0 {
            compare(
                model,
                &dir,
                i,
                &Outcome {
                    tokens: tokens[i].clone(),
                    steps: counts[i].clone(),
                    session: &sessions[i],
                },
            );
        } else {
            let path = dir.join(format!("tokens_{i}.bin"));
            let expected = words(&path);
            let n = expected.len().min(tokens[i].len());
            assert_eq!(tokens[i][..n], expected[..n], "{name} session {i}: tokens");
        }
    }
}

#[test]
fn sessions_match_the_cpp_engine() {
    if !trunk_path().exists() || !mtp_path().exists() {
        panic!("the Flash-Next artifacts are missing");
    }
    {
        let mut model = load(1);
        // Sparse attention in the trunk and the draft block, greedy drafts
        // verified on the device.
        run_single(
            &mut model,
            "cpp-session-greedy2600",
            "prompt2600.bin",
            24,
            0.0,
            0,
        );
        run_single(
            &mut model,
            "cpp-session-sampled50",
            "prompt50.bin",
            24,
            0.8,
            42,
        );
    }
    // The policy's cost model takes the configured concurrency: one model
    // per batch size, one at a time.
    {
        let mut model = load(4);
        run_batch(&mut model, "cpp-batch-sampled50", 4, 16, 0.8, 7);
    }
    let mut model = load(3);
    run_batch(&mut model, "cpp-batch-greedy50", 3, 16, 0.0, 0);
}
