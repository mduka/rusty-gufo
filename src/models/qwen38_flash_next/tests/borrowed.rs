//! Borrowed snapshots (`CheckBorrowedSnapshots` and
//! `CheckCheckpointRetention` of `snapshot_test.cpp`): a snapshot that
//! leaves its append-only rows in the session restores, forks and
//! materializes the bytes of a host snapshot taken at the same point, through
//! rewinds and branches that overwrite those rows, a reader copying while the
//! session appends or decodes, a reset and the session's destruction; a
//! dropped deep snapshot frees what only it needed; and without device
//! headroom (`CheckDeviceExhaustion`) snapshots fall back to host bytes; a
//! capture for the disk holds exactly its payload in host memory:
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next --features gpu --test borrowed

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use common::{mtp_path, trunk_path};
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_qwen38_flash_next::engine::{Model, ModelOptions, Session};
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

fn prompt(model: &Model) -> Vec<i32> {
    let pattern = model.tokenize(
        "The quick brown fox jumps over the lazy dog. \
         Strix Halo executes this deterministic benchmark sequence. ",
    );
    (0..4095).map(|i| pattern[i % pattern.len()]).collect()
}

fn decode(model: &mut Model, session: &mut Session, count: usize) {
    let config = SamplingConfig {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.9,
        seed: 7,
        ..SamplingConfig::default()
    };
    let history: Vec<u32> = session.tokens().iter().map(|&t| t as u32).collect();
    let mut sampler = SamplerState::new(config, &history).unwrap();
    let mut decoded = 0;
    while decoded < count {
        let step = model
            .decode_step(session, 8.min(count - decoded), &mut sampler, false)
            .unwrap();
        assert!(!step.tokens.is_empty(), "empty decode step");
        decoded += step.tokens.len();
    }
}

fn joined(parts: &[&[i32]]) -> Vec<i32> {
    parts.concat()
}

fn borrowed_snapshots(model: &mut Model, prompt: &[i32], mode: SessionMode) {
    let mut source = model.create_session(mode, CONTEXT).unwrap();
    model.sync(&mut source, prompt).unwrap();
    let frontier_bytes = model.save_snapshot(&source).unwrap();
    let frontier = model.snapshot(&source).unwrap();
    assert!(
        frontier.is_borrowed() && frontier.borrows_from(&source),
        "{mode:?}: the snapshot did not borrow its session"
    );
    assert!(
        frontier.to_bytes().unwrap() == frontier_bytes,
        "{mode:?}: a borrowed snapshot materializes other bytes than a host one"
    );
    model.reset(&mut source);
    model.restore(&mut source, &frontier).unwrap();
    assert!(
        model.save_snapshot(&source).unwrap() == frontier_bytes,
        "{mode:?}: rewind after logical reset changed checkpoint rows"
    );

    let boundary_prompt = joined(&[prompt, &prompt[..9]]);
    model.sync(&mut source, &boundary_prompt).unwrap();
    let boundary_bytes = model.save_snapshot(&source).unwrap();
    let boundary = model.snapshot(&source).unwrap();
    model
        .sync(&mut source, &joined(&[&boundary_prompt, &prompt[9..16]]))
        .unwrap();

    // Rewind without reading either borrowed payload: the boundary's suffix
    // is copied out, overwritten by another branch, and restored over it.
    model.restore(&mut source, &frontier).unwrap();
    let branch = joined(&[prompt, &prompt[20..36]]);
    model.sync(&mut source, &branch).unwrap();
    let other_branch = model.snapshot(&source).unwrap();
    let other_bytes = model.save_snapshot(&source).unwrap();
    let mut fork = model.create_session(mode, CONTEXT).unwrap();
    model.restore(&mut fork, &boundary).unwrap();
    assert!(
        model.save_snapshot(&fork).unwrap() == boundary_bytes,
        "{mode:?}: foreign fork lost its live prefix or detached tail"
    );
    let fork_boundary = model.snapshot(&fork).unwrap();
    // Retained rows of a fork belong to it: resetting and overwriting that
    // prefix must protect them even without another capture.
    model.reset(&mut fork);
    model.sync(&mut fork, &prompt[20..52]).unwrap();
    assert!(
        fork_boundary.to_bytes().unwrap() == boundary_bytes,
        "{mode:?}: reset after a fork overwrote retained checkpoint rows"
    );
    model.restore(&mut fork, &fork_boundary).unwrap();
    assert!(
        model.save_snapshot(&fork).unwrap() == boundary_bytes,
        "{mode:?}: the fork's checkpoint did not restore its protected suffix"
    );
    model.restore(&mut source, &boundary).unwrap();
    assert!(
        model.save_snapshot(&source).unwrap() == boundary_bytes,
        "{mode:?}: borrowed restore differs from complete snapshot"
    );
    model.restore(&mut source, &other_branch).unwrap();
    assert!(
        model.save_snapshot(&source).unwrap() == other_bytes,
        "{mode:?}: second branch lost its overwritten rows"
    );

    // A byte reader copies committed rows while the session appends.
    let longer = joined(&[&branch, &prompt[36..52]]);
    let read = std::thread::scope(|scope| {
        let reader = scope.spawn(|| frontier.to_bytes().unwrap());
        model.sync(&mut source, &longer).unwrap();
        reader.join().unwrap()
    });
    assert!(
        read == frontier_bytes,
        "{mode:?}: concurrent append changed a borrowed snapshot"
    );
    model.reset(&mut source);
    assert!(
        boundary.to_bytes().unwrap() == boundary_bytes,
        "{mode:?}: reset discarded borrowed checkpoint rows"
    );
    assert!(
        other_branch.to_bytes().unwrap() == other_bytes,
        "{mode:?}: reset discarded a detached branch"
    );

    model.sync(&mut source, prompt).unwrap();
    let overwritten = model.snapshot(&source).unwrap();
    model.reset(&mut source);
    model.sync(&mut source, &prompt[20..52]).unwrap();
    assert!(
        overwritten.to_bytes().unwrap() == frontier_bytes,
        "{mode:?}: the first write after reset overwrote a borrowed checkpoint"
    );
    model.restore(&mut source, &overwritten).unwrap();
    let at_decode = model.snapshot(&source).unwrap();
    let decode_bytes = model.save_snapshot(&source).unwrap();
    let read = std::thread::scope(|scope| {
        let reader = scope.spawn(|| at_decode.to_bytes().unwrap());
        decode(model, &mut source, 16);
        reader.join().unwrap()
    });
    assert!(
        read == decode_bytes,
        "{mode:?}: concurrent decode changed checkpoint bytes"
    );

    let survivor = model.snapshot(&source).unwrap();
    let survivor_bytes = model.save_snapshot(&source).unwrap();
    drop(source);
    assert!(
        survivor.to_bytes().unwrap() == survivor_bytes,
        "{mode:?}: source destruction discarded borrowed rows"
    );
    model.restore(&mut fork, &survivor).unwrap();
    assert!(
        model.save_snapshot(&fork).unwrap() == survivor_bytes,
        "{mode:?}: foreign restore after source destruction lost checkpoint rows"
    );
    println!("borrowed snapshots {mode:?}: rewind, branches, readers, reset, destruction pass");
}

fn retention(model: &mut Model, prompt: &[i32], mode: SessionMode) {
    let mut source = model.create_session(mode, CONTEXT).unwrap();
    model.sync(&mut source, &prompt[..32]).unwrap();
    let expected = model.save_snapshot(&source).unwrap();
    let small = model.snapshot(&source).unwrap();
    assert!(
        small.borrows_from(&source),
        "{mode:?}: checkpoint lost its source identity"
    );
    let before = small.device_bytes();
    model.sync(&mut source, prompt).unwrap();
    let large = model.snapshot(&source).unwrap();
    model.reset(&mut source);
    model.sync(&mut source, &prompt[..16]).unwrap();
    assert!(
        large.device_bytes() > before + (80 << 20),
        "{mode:?}: deep checkpoint did not preserve its overwritten rows"
    );
    drop(large);
    // A short prefix keeps only its own rows.
    let after = small.device_bytes();
    assert!(
        after <= before + (8 << 20),
        "{mode:?}: short checkpoint retained the discarded deep allocation"
    );
    model.restore(&mut source, &small).unwrap();
    assert!(
        model.save_snapshot(&source).unwrap() == expected,
        "{mode:?}: discarding a deep checkpoint changed its shared short prefix"
    );
    println!(
        "checkpoint retention {mode:?}: additional {} bytes, short prefix exact",
        after - before
    );
}

/// Without device headroom, captures and rewinds keep snapshots in their
/// host payloads instead of failing the request.
fn exhaustion(model: &mut Model, prompt: &[i32], mode: SessionMode) {
    struct Uncap;
    impl Drop for Uncap {
        fn drop(&mut self) {
            gufo_hip::cap_reported_device_free(usize::MAX);
        }
    }
    let _uncap = Uncap;
    let mut source = model.create_session(mode, CONTEXT).unwrap();
    model.sync(&mut source, prompt).unwrap();
    // Read the bytes through another capture: materializing this one would
    // move it to the host before the rewind.
    let expected = model.save_snapshot(&source).unwrap();
    let borrowed = model.snapshot(&source).unwrap();
    assert!(
        borrowed.device_bytes() != 0,
        "{mode:?}: the capture did not borrow"
    );
    gufo_hip::cap_reported_device_free(0);
    model.reset(&mut source);
    model.sync(&mut source, &prompt[20..52]).unwrap();
    assert_eq!(
        borrowed.device_bytes(),
        0,
        "{mode:?}: an exhausted rewind kept checkpoint rows on the device"
    );
    assert!(
        borrowed.to_bytes().unwrap() == expected,
        "{mode:?}: an exhausted rewind lost overwritten checkpoint rows"
    );
    let owned = model.snapshot(&source).unwrap();
    assert!(
        !owned.is_borrowed() && owned.device_bytes() == 0,
        "{mode:?}: an exhausted capture allocated device storage"
    );
    let branch = owned.to_bytes().unwrap();
    model.restore(&mut source, &borrowed).unwrap();
    assert!(
        model.save_snapshot(&source).unwrap() == expected,
        "{mode:?}: an exhausted restore differs from its checkpoint"
    );
    model.restore(&mut source, &owned).unwrap();
    assert!(
        model.save_snapshot(&source).unwrap() == branch,
        "{mode:?}: a host capture differs after restore"
    );
    println!("device exhaustion {mode:?}: host fallback exact");
}

/// A capture for the disk (`SnapshotForPersistence`) holds the bytes of a
/// host snapshot and no more host memory than they need.
fn persistence(model: &mut Model, prompt: &[i32], mode: SessionMode) {
    let mut source = model.create_session(mode, CONTEXT).unwrap();
    model.sync(&mut source, prompt).unwrap();
    let expected = model.save_snapshot(&source).unwrap();
    let host = model.snapshotter().unwrap().capture_host(&source).unwrap();
    assert!(
        !host.is_borrowed() && host.device_bytes() == 0,
        "{mode:?}: a persistence capture borrowed device rows"
    );
    assert_eq!(
        host.host_capacity(),
        host.len(),
        "{mode:?}: a persistence capture holds more host memory than its payload"
    );
    assert!(
        host.to_bytes().unwrap() == expected,
        "{mode:?}: a persistence capture differs from the host snapshot"
    );
    let mut restored = model.create_session(mode, CONTEXT).unwrap();
    model.restore(&mut restored, &host).unwrap();
    assert!(
        model.save_snapshot(&restored).unwrap() == expected,
        "{mode:?}: a persistence capture restores differently"
    );
    println!("persistence {mode:?}: host capture exact, no spare capacity");
}

#[test]
fn borrowed_snapshots_restore_like_host_snapshots() {
    let mut model = load();
    let prompt = prompt(&model);
    for mode in [SessionMode::Autoregressive, SessionMode::Speculative] {
        borrowed_snapshots(&mut model, &prompt, mode);
        retention(&mut model, &prompt, mode);
        exhaustion(&mut model, &prompt, mode);
        persistence(&mut model, &prompt, mode);
    }
}
