//! Batched Q8 projections keep each row's vector arithmetic: every row of a
//! 2- to N-row launch equals its single-row result bit for bit, the matrix
//! tiles of wide shapes included, and nothing is written past the output
//! (`CheckDecodeGrouping` of the C++
//! `tests/models/qwen38_flash_next/projection_ops_test.cpp`); Q8_0 weights
//! expanded to F32 equal their codes (`CheckExpandQ8_0`); prefill W8A8 rows
//! do not depend on the pass width (`CheckW8A8PassWidths`):
//! cargo test -p gufo-qwen38-flash-next-kernels --features gpu --test projection

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{GgmlType, QuantSlice, Stream};
use gufo_qwen38_flash_next_kernels as k;

/// The weight tail the launch contract requires.
const TAIL: usize = 4096;
const GUARD: u32 = 0xA5A5_A5A5;

/// A deterministic value in `[-scale, scale)` (`Uniform`).
fn uniform(seed: &mut u32, scale: f32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
}

/// The F16 bits of a normal `value` (truncated mantissa).
fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exponent), "{value} is not a normal F16");
    (((bits >> 16) & 0x8000) | ((exponent as u32) << 10) | ((bits >> 13) & 0x3ff)) as u16
}

/// Q8_0 rows of `m` x `k` random weights followed by the zero tail.
fn weights(m: usize, k: usize, mut seed: u32) -> Vec<u8> {
    let mut bytes = vec![0u8; m * (k / 32) * 34 + TAIL];
    for block in bytes[..m * (k / 32) * 34].chunks_mut(34) {
        let raw: Vec<f32> = (0..32).map(|_| uniform(&mut seed, 1.0)).collect();
        let max = raw.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-3);
        let bits = f16_bits(max / 127.0);
        let scale = {
            let e = i32::from((bits >> 10) & 0x1f) - 15;
            (1.0 + f32::from(bits & 0x3ff) / 1024.0) * 2f32.powi(e)
        };
        block[..2].copy_from_slice(&bits.to_le_bytes());
        for (q, v) in block[2..].iter_mut().zip(&raw) {
            *q = ((v / scale).round().clamp(-127.0, 127.0) as i8) as u8;
        }
    }
    bytes
}

fn check_decode_grouping(m: usize, k: usize) {
    let tokens = if m == 320 && k == 10240 {
        64
    } else if m >= 8192 && k == 2560 {
        48
    } else {
        32
    };
    let stream = Stream::new().unwrap();
    let host_w = weights(m, k, 11);
    let mut w = stream.alloc::<u8>(host_w.len()).unwrap();
    stream.upload(&host_w, w.as_mut_slice()).unwrap();
    let mut seed = 37;
    let x: Vec<f32> = (0..tokens * k).map(|_| uniform(&mut seed, 2.0)).collect();
    let mut dx = stream.alloc::<f32>(x.len()).unwrap();
    stream.upload(&x, dx.as_mut_slice()).unwrap();
    let mut q = stream.alloc::<u8>(k::q8_1_bytes(tokens, k)).unwrap();
    let quantized = k::mmq_quantize_q8_1(
        &stream,
        dx.as_slice(),
        q.as_mut_slice(),
        tokens as u32,
        k as u32,
    )
    .unwrap();
    assert_eq!(quantized, gufo_hip::Launch::Launched);
    let weights = || QuantSlice::new(w.as_slice(), GgmlType::Q8_0);
    let row = |t: usize| k::q8_1_bytes(t, k);

    let mut out = stream.alloc::<f32>(tokens * m).unwrap();
    for t in 0..tokens {
        let launched = k::mmq_q8_dense_vec(
            &stream,
            weights(),
            q.slice(row(t)..row(t + 1)),
            out.slice_mut(t * m..(t + 1) * m),
            m as u32,
            1,
            k as u32,
        )
        .unwrap();
        assert_eq!(launched, gufo_hip::Launch::Launched);
    }
    let mut single = vec![0f32; tokens * m];
    stream.download(out.as_slice(), &mut single).unwrap();

    for n in 2..=tokens {
        if m == 320 && k == 10240 && n > 32 && n % 8 != 0 {
            continue;
        }
        let mut batch = stream.alloc::<f32>(n * m + 4).unwrap();
        stream.fill_bytes(batch.as_mut_slice(), 0xA5).unwrap();
        // The input ends at its allocation, so an overread of a partial
        // final wave is not hidden by spare rows.
        let launched = k::mmq_q8_dense_vec(
            &stream,
            weights(),
            q.slice(row(tokens - n)..row(tokens)),
            batch.slice_mut(0..n * m),
            m as u32,
            n as u32,
            k as u32,
        )
        .unwrap();
        assert_eq!(launched, gufo_hip::Launch::Launched, "M={m} K={k} N={n}");
        let mut host = vec![0f32; n * m + 4];
        stream.download(batch.as_slice(), &mut host).unwrap();
        let expected = &single[(tokens - n) * m..];
        let first = host[..n * m]
            .iter()
            .zip(expected)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            first.is_none(),
            "Q8 dense grouping differs: M={m} K={k} N={n} at {first:?}"
        );
        assert!(
            host[n * m..].iter().all(|v| v.to_bits() == GUARD),
            "Q8 batch projection overwrote its guard: M={m} K={k} N={n}"
        );
    }
}

#[test]
fn batched_rows_keep_their_vector_arithmetic() {
    for (m, k) in [
        (64, 2560),
        (320, 10240),
        (1024, 2560),
        (10240, 320),
        (2561, 2560),
        (12289, 2560),
        (65537, 2560),
    ] {
        check_decode_grouping(m, k);
    }
}

/// Q8_0 inject and alpha/beta weights are expanded to F32 at load: every
/// value equals its block scale times its code, and nothing past `count` is
/// written (`CheckExpandQ8_0`).
fn check_expand_q8_0(m: usize, k: usize, seed: u32) {
    let stream = Stream::new().unwrap();
    let host_w = weights(m, k, seed);
    let count = m * k;
    let mut w = stream.alloc::<u8>(host_w.len()).unwrap();
    stream.upload(&host_w, w.as_mut_slice()).unwrap();
    let mut out = stream.alloc::<f32>(count + 64).unwrap();
    stream.fill_bytes(out.as_mut_slice(), 0x7F).unwrap();
    k::expand_q8_0(
        &stream,
        w.slice(0..count / 32 * 34),
        out.slice_mut(0..count),
        count,
    )
    .unwrap();
    let mut back = vec![0f32; count + 64];
    stream.download(out.as_slice(), &mut back).unwrap();
    for (block, values) in host_w[..count / 32 * 34]
        .chunks(34)
        .zip(back[..count].chunks(32))
    {
        let bits = u16::from_le_bytes([block[0], block[1]]);
        let e = i32::from((bits >> 10) & 0x1f) - 15;
        let scale = (1.0 + f32::from(bits & 0x3ff) / 1024.0) * 2f32.powi(e);
        for (&q, &value) in block[2..].iter().zip(values) {
            let want = scale * f32::from(q as i8);
            assert_eq!(value.to_bits(), want.to_bits(), "{m} x {k}");
        }
    }
    assert!(
        back[count..].iter().all(|v| v.to_bits() == 0x7F7F_7F7F),
        "Q8_0 expansion wrote past its count"
    );
}

#[test]
fn q8_0_weights_expand_to_their_codes() {
    // The hc_*_inject [4][10240] and stacked ssm_alpha/beta [96][2560]
    // shapes.
    check_expand_q8_0(4, 10240, 0xE8A4D1);
    check_expand_q8_0(96, 2560, 0xE8A4D2);
}

/// A prefill row's W8A8 projection does not depend on the pass width: a
/// prompt finished by a short pass after a checkpoint restore projects its
/// last rows with the 64-token tile (below 96 rows), a wide pass with the
/// 128-token tiles, their tail kernel or, from 1,024 rows of the 2560 x 6144
/// output projection, the wave64 kernel. Each short pass quantizes its own
/// rows, as its combine would (`CheckW8A8PassWidths`).
fn check_w8a8_pass_widths(m: usize, k: usize, wide: usize, seed: u32) {
    let stream = Stream::new().unwrap();
    let host = weights(m, k, seed);
    let mut state = seed ^ 0x9A55;
    let x: Vec<f32> = (0..wide * k).map(|_| uniform(&mut state, 2.0)).collect();
    let mut w = stream.alloc::<u8>(host.len()).unwrap();
    stream.upload(&host, w.as_mut_slice()).unwrap();
    let w = || QuantSlice::new(w.as_slice(), GgmlType::Q8_0);
    let mut input = stream.alloc::<f32>(x.len()).unwrap();
    stream.upload(&x, input.as_mut_slice()).unwrap();
    let tiled_bytes = |rows: usize| rows.div_ceil(128) * 8 * (k / 32) * 576;
    let mut tiled = stream.alloc::<u8>(tiled_bytes(wide)).unwrap();
    let mut out = stream.alloc::<f32>(wide * m).unwrap();
    k::quantize_q8_tiled(
        &stream,
        input.as_slice(),
        tiled.slice_mut(0..tiled_bytes(wide)),
        wide,
        k,
    )
    .unwrap();
    let launched = k::w8a8_gemm(
        &stream,
        w(),
        tiled.slice(0..tiled_bytes(wide)),
        out.as_mut_slice(),
        wide,
        m,
        k,
    )
    .unwrap();
    assert_eq!(
        launched,
        gufo_hip::Launch::Launched,
        "W8A8 wide pass rejected"
    );
    let mut reference = vec![0f32; wide * m];
    stream.download(out.as_slice(), &mut reference).unwrap();
    let mut failures = Vec::new();
    for rows in [1usize, 7, 9, 95, 96] {
        for first in [0, (wide - rows) / 2, wide - rows] {
            k::quantize_q8_tiled(
                &stream,
                input.slice(first * k..(first + rows) * k),
                tiled.slice_mut(0..tiled_bytes(rows)),
                rows,
                k,
            )
            .unwrap();
            let launched = k::w8a8_gemm(
                &stream,
                w(),
                tiled.slice(0..tiled_bytes(rows)),
                out.slice_mut(0..rows * m),
                rows,
                m,
                k,
            )
            .unwrap();
            assert_eq!(
                launched,
                gufo_hip::Launch::Launched,
                "W8A8 short pass rejected"
            );
            let mut actual = vec![0f32; rows * m];
            stream
                .download(out.slice(0..rows * m), &mut actual)
                .unwrap();
            let differ = (0..rows)
                .filter(|&r| {
                    actual[r * m..(r + 1) * m]
                        .iter()
                        .zip(&reference[(first + r) * m..(first + r + 1) * m])
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                })
                .count();
            if differ != 0 {
                failures.push(format!(
                    "{differ} of {rows} rows from {first} differ from the {wide}-row pass"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "W8A8 m={m} k={k}: {failures:#?}");
}

#[test]
fn w8a8_rows_do_not_depend_on_the_pass_width() {
    // The output projection takes the wave64 kernel from 1,024 rows; the
    // others keep the blocked kernel, past a W8A8 tail at 1549 rows.
    check_w8a8_pass_widths(2560, 6144, 1619, 0x5A_6144);
    check_w8a8_pass_widths(2560, 6144, 1549, 0x5B_6144);
    check_w8a8_pass_widths(320, 10240, 1549, 0x5A_0320);
    check_w8a8_pass_widths(640, 2560, 1619, 0x5A_0640);
}
