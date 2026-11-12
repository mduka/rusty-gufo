//! WMMA prefill attention: cargo test --profile gpu-test -p gufo-qwen-kernels
//! --features gpu --test attention
//!
//! For every instantiated head shape (24/4 of the Qwen3.8 27B, 16/4 and 8/2
//! of the dense Qwen3.5 models):
//!
//! - restricted to its last query tiles (upstream `wmma_prefill_ops_test.cpp`):
//!   the final prefill layer keeps only the tile holding its last token. The
//!   retained rows must equal the full call's bytes at ragged depths and
//!   tails, with the canonical FP16 cache and every bounded head-group
//!   workspace, and the skipped rows stay untouched;
//! - against an FP64 CPU reference of the gated causal attention over the FP16
//!   cache, on rows at tile edges and the last row of each chunk, with every
//!   output row written (NaN-filled before the call).

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{DeviceBuffer, DeviceCopy, F16, Launch, Stream};
use gufo_qwen_kernels as k;

const HEAD_DIM: usize = 256;

/// The instantiated shapes: (query heads, KV heads).
const SHAPES: [(usize, usize); 3] = [(24, 4), (16, 4), (8, 2)];

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

/// Standard normal values (Box-Muller over the xorshift stream).
fn normals(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = || ((next_random(&mut state) >> 8) as f64 + 0.5) / f64::from(1u32 << 24);
    (0..count)
        .map(|_| {
            let (u, v) = (uniform(), uniform());
            ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
        })
        .collect()
}

fn upload<T: DeviceCopy>(stream: &Stream, values: &[T]) -> DeviceBuffer<T> {
    let mut buffer = stream.alloc::<T>(values.len()).unwrap();
    stream.upload(values, buffer.as_mut_slice()).unwrap();
    buffer
}

/// Workspace floats one packed KV head of `length` positions takes: FP16 keys,
/// and values padded to whole 16-key tiles.
fn packed_head_floats(length: usize) -> usize {
    length.next_multiple_of(16) * HEAD_DIM / 2
}

/// One prompt chunk of `batch` rows at `start_pos` over layer 1 of a two-layer
/// cache, with packed K/V workspaces of `workspace` floats each.
fn first_tile_matches(
    stream: &Stream,
    (heads, kv_heads): (usize, usize),
    start_pos: usize,
    batch: usize,
    workspace: usize,
) {
    let max_context = start_pos + batch + 5;
    let layer = 1usize;
    let rows_q = batch * heads * HEAD_DIM;
    let rows_kv = batch * kv_heads * HEAD_DIM;
    let cache = (layer + 1) * max_context * kv_heads * HEAD_DIM;
    let seed = (start_pos * 31 + batch + heads) as u32 | 1;
    let q = upload(stream, &floats(rows_q, seed, 0.25));
    let gate = upload(stream, &floats(rows_q, seed ^ 0x55, 4.0));
    let kv_rows = upload(stream, &floats(rows_kv, seed ^ 0x77, 1.0));
    let k_cache = upload(stream, &halves(cache, seed ^ 0x99));
    let v_cache = upload(stream, &halves(cache, seed ^ 0xBB));
    let mut packed_k = stream.alloc::<f32>(workspace.max(1)).unwrap();
    let mut packed_v = stream.alloc::<f32>(workspace.max(1)).unwrap();
    let mut out = stream.alloc::<f32>(rows_q).unwrap();
    let mut run = |out: &mut DeviceBuffer<f32>, first_query: usize| {
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
            heads as u32,
            kv_heads as u32,
            HEAD_DIM as u32,
            packed_k.slice_mut(0..workspace),
            workspace,
            packed_v.slice_mut(0..workspace),
            workspace,
            first_query,
        )
        .unwrap();
        assert_eq!(
            launched,
            Launch::Launched,
            "{heads}/{kv_heads} heads run on WMMA"
        );
        let mut host = vec![0f32; rows_q];
        stream.download(out.as_slice(), &mut host).unwrap();
        host
    };
    let full = run(&mut out, 0);
    assert!(full.iter().all(|v| v.is_finite()));
    let first_query = (batch - 1) / 32 * 32;
    let tile = run(&mut out, first_query);
    let first = first_query * heads * HEAD_DIM;
    assert!(
        tile[..first].iter().all(|v| v.to_bits() == 0xA5A5_A5A5),
        "{heads}/{kv_heads} start {start_pos} batch {batch}: skipped rows were written"
    );
    assert!(
        tile[first..]
            .iter()
            .zip(&full[first..])
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{heads}/{kv_heads} start {start_pos} batch {batch} workspace {workspace}: retained rows \
         changed"
    );
}

#[test]
fn the_last_query_tile_keeps_the_full_arithmetic() {
    let stream = Stream::new().unwrap();
    for shape in SHAPES {
        // Short chunks at ragged depths and tails, canonical cache reads.
        for (start_pos, batch) in [(0, 33), (7, 70), (300, 95), (1025, 64)] {
            first_tile_matches(&stream, shape, start_pos, batch, 0);
        }
        // From 1024 rows the KV heads are packed, as many as the workspace
        // holds: one, two, or all heads of the visible context.
        for (start_pos, batch) in [(0, 1100), (30, 1030), (2048, 1025)] {
            let head = packed_head_floats(start_pos + batch);
            for packed_heads in [1, 2, shape.1] {
                first_tile_matches(&stream, shape, start_pos, batch, packed_heads * head);
            }
        }
    }
}

/// The gated causal attention of query row `row` (all heads) in FP64:
/// softmax(q . k / 16) v over cache positions `0..=start_pos + row`, times
/// sigmoid(gate). `cache` holds one layer, `[position][kv head][dim]`.
#[allow(clippy::too_many_arguments)]
fn reference_row(
    (heads, kv_heads): (usize, usize),
    q: &[f32],
    gate: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    start_pos: usize,
    row: usize,
) -> Vec<f64> {
    let visible = start_pos + row + 1;
    let group = heads / kv_heads;
    let mut out = vec![0f64; heads * HEAD_DIM];
    let mut scores = vec![0f64; visible];
    for head in 0..heads {
        let kv = head / group;
        let query = &q[(row * heads + head) * HEAD_DIM..][..HEAD_DIM];
        for (p, score) in scores.iter_mut().enumerate() {
            let key = &k_cache[(p * kv_heads + kv) * HEAD_DIM..][..HEAD_DIM];
            let dot: f64 = query
                .iter()
                .zip(key)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            *score = dot / 16.0;
        }
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = scores
            .iter_mut()
            .map(|s| {
                *s = (*s - max).exp();
                *s
            })
            .sum();
        let o = &mut out[head * HEAD_DIM..][..HEAD_DIM];
        for (p, &weight) in scores.iter().enumerate() {
            let value = &v_cache[(p * kv_heads + kv) * HEAD_DIM..][..HEAD_DIM];
            for (o, &v) in o.iter_mut().zip(value) {
                *o += weight / sum * f64::from(v);
            }
        }
        let g = &gate[(row * heads + head) * HEAD_DIM..][..HEAD_DIM];
        for (o, &g) in o.iter_mut().zip(g) {
            *o /= 1.0 + (-f64::from(g)).exp();
        }
    }
    out
}

/// One chunk against the FP64 reference. Queries and keys carry the Qwen3.5
/// QK-norm scale (RMS 1.5, scores of standard deviation about 2.25),
/// values and gates are standard normal. Measured on such data (and on twice
/// the query scale) from 512 to 32K visible positions, the WMMA output's
/// relative L2 error is 2e-4 to 6.3e-4 and its largest error 1.4% of the
/// output RMS: FP16 queries and probabilities. The bounds leave a margin of
/// three and two; a wrong head, key or causal mask is off by the output's
/// own size.
fn matches_fp64_reference(
    stream: &Stream,
    (heads, kv_heads): (usize, usize),
    start_pos: usize,
    batch: usize,
    workspace: usize,
) {
    let max_context = start_pos + batch;
    let rows_q = batch * heads * HEAD_DIM;
    let cache = max_context * kv_heads * HEAD_DIM;
    let seed = (start_pos * 131 + batch * 7 + heads) as u32 | 1;
    let q: Vec<f32> = normals(rows_q, seed).iter().map(|x| x * 1.5).collect();
    let gate = normals(rows_q, seed ^ 0x5555);
    let k_half: Vec<F16> = normals(cache, seed ^ 0x7777)
        .iter()
        .map(|x| F16::from_f32(x * 1.5))
        .collect();
    let v_half: Vec<F16> = normals(cache, seed ^ 0x9999)
        .iter()
        .map(|&x| F16::from_f32(x))
        .collect();
    let d_q = upload(stream, &q);
    let d_gate = upload(stream, &gate);
    // Read only by the KV write, which the caller already did.
    let kv_rows = stream.alloc::<f32>(batch * kv_heads * HEAD_DIM).unwrap();
    let k_cache = upload(stream, &k_half);
    let v_cache = upload(stream, &v_half);
    let mut packed_k = stream.alloc::<f32>(workspace.max(1)).unwrap();
    let mut packed_v = stream.alloc::<f32>(workspace.max(1)).unwrap();
    let mut out = stream.alloc::<f32>(rows_q).unwrap();
    // NaN everywhere: a row or head the kernel misses stays NaN.
    stream.fill_bytes(out.as_mut_slice(), 0xFF).unwrap();
    let launched = k::wmma_attention(
        stream,
        d_q.as_slice(),
        kv_rows.as_slice(),
        kv_rows.as_slice(),
        d_gate.as_slice(),
        k_cache.as_slice(),
        v_cache.as_slice(),
        out.as_mut_slice(),
        0,
        start_pos as u32,
        batch,
        max_context as u32,
        heads as u32,
        kv_heads as u32,
        HEAD_DIM as u32,
        packed_k.slice_mut(0..workspace),
        workspace,
        packed_v.slice_mut(0..workspace),
        workspace,
        0,
    )
    .unwrap();
    assert_eq!(
        launched,
        Launch::Launched,
        "{heads}/{kv_heads} heads run on WMMA"
    );
    let mut host = vec![0f32; rows_q];
    stream.download(out.as_slice(), &mut host).unwrap();
    let case = format!("{heads}/{kv_heads} start {start_pos} batch {batch} workspace {workspace}");
    assert!(
        host.iter().all(|v| v.is_finite()),
        "{case}: an output was not written"
    );

    let k_cache: Vec<f32> = k_half.iter().map(|h| h.to_f32()).collect();
    let v_cache: Vec<f32> = v_half.iter().map(|h| h.to_f32()).collect();
    let rows = [
        0,
        1,
        31,
        32,
        63,
        64,
        batch / 2,
        batch - 33,
        batch - 32,
        batch - 1,
    ];
    let (mut err2, mut ref2, mut max_err) = (0f64, 0f64, 0f64);
    for row in rows {
        let expected = reference_row(
            (heads, kv_heads),
            &q,
            &gate,
            &k_cache,
            &v_cache,
            start_pos,
            row,
        );
        let got = &host[row * heads * HEAD_DIM..][..heads * HEAD_DIM];
        for (&g, &e) in got.iter().zip(&expected) {
            let d = f64::from(g) - e;
            err2 += d * d;
            ref2 += e * e;
            max_err = max_err.max(d.abs());
        }
    }
    let n = (rows.len() * heads * HEAD_DIM) as f64;
    let rel_l2 = (err2 / ref2).sqrt();
    let max_rel_rms = max_err / (ref2 / n).sqrt();
    assert!(
        rel_l2 < 2e-3 && max_rel_rms < 3e-2,
        "{case}: relative L2 error {rel_l2:.2e}, largest error {max_rel_rms:.2e} of the RMS"
    );
}

#[test]
fn every_shape_matches_an_fp64_reference() {
    let stream = Stream::new().unwrap();
    for shape in SHAPES {
        // Canonical cache reads below 1024 rows, at and away from depth 0.
        matches_fp64_reference(&stream, shape, 0, 512, 0);
        matches_fp64_reference(&stream, shape, 1500, 300, 0);
        // Packed heads from 1024 rows: all of them, and one per pass.
        matches_fp64_reference(&stream, shape, 0, 2048, shape.1 * packed_head_floats(2048));
        matches_fp64_reference(&stream, shape, 2048, 2048, packed_head_floats(4096));
    }
}
