//! Small-batch verification projections against the decode GEMV, byte for
//! byte (upstream `TestSmallBatchExactness` of `q4kxl_quant_ops_test.cpp`):
//! cargo test --profile gpu-test -p gufo-qwen-kernels --features gpu --test small_batch
//!
//! The 27B verifies several drafted rows with one pass whose per-row
//! arithmetic must equal decoding each row alone. This covers the five-token
//! IQ4_XS FFN schedule, the four-token Q4_K row-first FMA schedule and their
//! neighbouring widths on the real gate/up and down geometries.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{GgmlType, QuantSlice, Stream};
use gufo_qwen_kernels as k;

fn next_random(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Random blocks of `rows` x `columns` weights whose FP16 scales lie in
/// [0.5, 1.0), so that every product stays comfortably in range
/// (`MakeWeights`).
fn weights(format: GgmlType, rows: usize, columns: usize, seed: u32) -> Vec<u8> {
    let bytes = format.encoded_bytes((rows * columns) as u64).unwrap() as usize;
    let (block_elements, block_bytes) = format.block();
    assert_eq!(bytes, rows * columns / block_elements * block_bytes);
    let mut state = seed;
    let mut out: Vec<u8> = (0..bytes)
        .map(|_| (next_random(&mut state) & 0xFF) as u8)
        .collect();
    let mut tame = |at: &mut [u8]| {
        let bits = (0x3800 | (next_random(&mut state) & 0x03FF)) as u16;
        at[..2].copy_from_slice(&bits.to_le_bytes());
    };
    for block in out.chunks_exact_mut(block_bytes) {
        match format {
            GgmlType::Q4K | GgmlType::Q5K => {
                tame(&mut block[0..]);
                tame(&mut block[2..]);
            }
            GgmlType::Q6K => tame(&mut block[208..]),
            GgmlType::Q3K => tame(&mut block[108..]),
            GgmlType::Iq4Nl | GgmlType::Iq4Xs | GgmlType::Iq3S | GgmlType::Q8_0 => {
                tame(&mut block[0..]);
            }
            _ => {}
        }
    }
    out
}

/// Activations over three scales of magnitude, as verification rows carry.
fn activations(batch: usize, columns: usize) -> Vec<f32> {
    let mut state = 0x1357_9BDFu32;
    (0..batch * columns)
        .map(|_| {
            let unit = ((next_random(&mut state) & 0xFFFF) as i32 - 32768) as f32 / 32768.0;
            let exponent = (next_random(&mut state) % 12) as i32 - 6;
            unit * 2f32.powi(exponent)
        })
        .collect()
}

/// `batch` rows through the batched projection equal each row's decode GEMV,
/// and the projection writes no row beyond them.
fn exact(stream: &Stream, format: GgmlType, batch: usize, rows: usize, columns: usize) {
    const GUARD_ROWS: usize = 8;
    let host = weights(format, rows, columns, 0x2468_ACE0);
    let x = activations(batch, columns);
    let mut w = stream.alloc::<u8>(host.len()).unwrap();
    stream.upload(&host, w.as_mut_slice()).unwrap();
    let mut input = stream.alloc::<f32>(x.len()).unwrap();
    stream.upload(&x, input.as_mut_slice()).unwrap();
    let mut batched = stream.alloc::<f32>((batch + GUARD_ROWS) * rows).unwrap();
    stream.fill_bytes(batched.as_mut_slice(), 0xA5).unwrap();
    let mut single = stream.alloc::<f32>(rows).unwrap();
    let w_view = || QuantSlice::new(w.as_slice(), format);
    k::batched_quant_gemm_f32(
        stream,
        w_view(),
        input.as_slice(),
        batched.slice_mut(0..batch * rows),
        batch,
        rows,
        columns,
    )
    .unwrap();
    let mut got = vec![0f32; (batch + GUARD_ROWS) * rows];
    stream.download(batched.as_slice(), &mut got).unwrap();
    let mut mismatches = got[batch * rows..]
        .iter()
        .filter(|v| v.to_bits() != 0xA5A5_A5A5)
        .count();
    let mut expected = vec![0f32; rows];
    for token in 0..batch {
        k::gemv(
            stream,
            w_view(),
            input.slice(token * columns..(token + 1) * columns),
            single.as_mut_slice(),
            rows,
            columns,
        )
        .unwrap();
        stream.download(single.as_slice(), &mut expected).unwrap();
        mismatches += got[token * rows..(token + 1) * rows]
            .iter()
            .zip(&expected)
            .filter(|(a, b)| !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits())
            .count();
    }
    assert_eq!(
        mismatches, 0,
        "{format:?} batch {batch} {rows}x{columns}: rows differ from the decode GEMV"
    );
}

/// The 27B FFN projections: packed and single gate/up rows, and down.
const FFN: [(usize, usize); 3] = [(17408, 5120), (34816, 5120), (5120, 17408)];

/// Five live tokens reuse the eight-column distributed reduction; the
/// unused columns stay zero and write nothing.
#[test]
fn five_token_iq4_xs_projections_match_decode() {
    let stream = Stream::new().unwrap();
    for (rows, columns) in FFN {
        exact(&stream, GgmlType::Iq4Xs, 5, rows, columns);
    }
}

/// Four-row Q4_K verification converts each row's coefficients before its
/// dependent FMAs and keeps every accumulation; widths three and eight keep
/// their original schedules.
#[test]
fn four_token_q4_k_projections_match_decode() {
    let stream = Stream::new().unwrap();
    for batch in [4, 8] {
        for (rows, columns) in FFN {
            exact(&stream, GgmlType::Q4K, batch, rows, columns);
        }
    }
    exact(&stream, GgmlType::Q4K, 3, 17408, 5120);
}
