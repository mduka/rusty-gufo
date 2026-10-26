//! WMMA prefill attention restricted to its last query tiles (upstream
//! `wmma_prefill_ops_test.cpp`): cargo test --profile gpu-test -p
//! gufo-qwen-kernels --features gpu --test attention
//!
//! The final prefill layer keeps only the tile holding its last token. The
//! retained rows must equal the full call's bytes at ragged depths and tails,
//! with the canonical FP16 cache and every bounded head-group workspace, and
//! the skipped rows stay untouched.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{F16, Launch, Stream};
use gufo_qwen_kernels as k;

const HEADS: usize = 24;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;

fn next_random(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn floats(count: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| ((next_random(&mut state) & 0xFFFF) as i32 - 32768) as f32 / 32768.0 * scale)
        .collect()
}

/// FP16 values of magnitude 1/8 to 1.
fn halves(count: usize, seed: u32) -> Vec<F16> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            let r = next_random(&mut state);
            F16((r & 0x8000) as u16 | (0x3000 + (r >> 16) % 0x0C00) as u16)
        })
        .collect()
}

/// One prompt chunk of `batch` rows at `start_pos` over layer 1 of a two-layer
/// cache, with packed K/V workspaces of `workspace` floats each.
fn first_tile_matches(stream: &Stream, start_pos: usize, batch: usize, workspace: usize) {
    let max_context = start_pos + batch + 5;
    let layer = 1usize;
    let rows_q = batch * HEADS * HEAD_DIM;
    let rows_kv = batch * KV_HEADS * HEAD_DIM;
    let cache = (layer + 1) * max_context * KV_HEADS * HEAD_DIM;
    let seed = (start_pos * 31 + batch) as u32 | 1;
    let upload_f32 = |values: &[f32]| {
        let mut buffer = stream.alloc::<f32>(values.len()).unwrap();
        stream.upload(values, buffer.as_mut_slice()).unwrap();
        buffer
    };
    let q = upload_f32(&floats(rows_q, seed, 0.25));
    let gate = upload_f32(&floats(rows_q, seed ^ 0x55, 4.0));
    let kv_rows = upload_f32(&floats(rows_kv, seed ^ 0x77, 1.0));
    let mut k_cache = stream.alloc::<F16>(cache).unwrap();
    stream
        .upload(&halves(cache, seed ^ 0x99), k_cache.as_mut_slice())
        .unwrap();
    let mut v_cache = stream.alloc::<F16>(cache).unwrap();
    stream
        .upload(&halves(cache, seed ^ 0xBB), v_cache.as_mut_slice())
        .unwrap();
    let mut packed_k = stream.alloc::<f32>(workspace.max(1)).unwrap();
    let mut packed_v = stream.alloc::<f32>(workspace.max(1)).unwrap();
    let mut out = stream.alloc::<f32>(rows_q).unwrap();
    let mut run = |out: &mut gufo_hip::DeviceBuffer<f32>, first_query: usize| {
        stream.fill_bytes(out.as_mut_slice(), 0xA5).unwrap();
        let launched = k::wmma_attention(
            stream,
            q.as_slice(),
            kv_rows.as_slice(),
            kv_rows.as_slice(),
            gate.as_slice(),
            k_cache.as_slice(),
            v_cache.as_slice(),
            out.as_mut_slice(),
            layer as u32,
            start_pos as u32,
            batch,
            max_context as u32,
            HEADS as u32,
            KV_HEADS as u32,
            HEAD_DIM as u32,
            packed_k.slice_mut(0..workspace),
            workspace,
            packed_v.slice_mut(0..workspace),
            workspace,
            first_query,
        )
        .unwrap();
        assert_eq!(launched, Launch::Launched, "the 27B shape runs on WMMA");
        let mut host = vec![0f32; rows_q];
        stream.download(out.as_slice(), &mut host).unwrap();
        host
    };
    let full = run(&mut out, 0);
    assert!(full.iter().all(|v| v.is_finite()));
    let first_query = (batch - 1) / 32 * 32;
    let tile = run(&mut out, first_query);
    let first = first_query * HEADS * HEAD_DIM;
    assert!(
        tile[..first].iter().all(|v| v.to_bits() == 0xA5A5_A5A5),
        "start {start_pos} batch {batch}: skipped rows were written"
    );
    assert!(
        tile[first..]
            .iter()
            .zip(&full[first..])
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "start {start_pos} batch {batch} workspace {workspace}: retained rows changed"
    );
}

#[test]
fn the_last_query_tile_keeps_the_full_arithmetic() {
    let stream = Stream::new().unwrap();
    // Short chunks at ragged depths and tails, canonical cache reads.
    for (start_pos, batch) in [(0, 33), (7, 70), (300, 95), (1025, 64)] {
        first_tile_matches(&stream, start_pos, batch, 0);
    }
    // From 1024 rows the KV heads are packed, as many as the workspace holds:
    // one, two, or all four heads of the visible context.
    for (start_pos, batch) in [(0, 1100), (30, 1030), (2048, 1025)] {
        let head = (start_pos + batch) * HEAD_DIM;
        for heads in [1, 2, KV_HEADS] {
            first_tile_matches(&stream, start_pos, batch, heads * head);
        }
    }
}
