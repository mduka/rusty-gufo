//! `small_gemm` takes its load-ahead kernel for aligned rows and the scalar
//! kernel otherwise; both give the same outputs bit for bit, for every
//! weight format, batch width (grouped beyond eight rows) and a K that ends
//! inside a load-ahead group:
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next-kernels --features gpu --test small_gemm

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{GgmlType, QuantSlice, Stream};
use gufo_qwen38_flash_next_kernels as k;

fn next(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed
}

/// A deterministic value in `[-scale, scale)`.
fn uniform(seed: &mut u32, scale: f32) -> f32 {
    ((next(seed) >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
}

/// Row-major weights of `format` with finite values; row 0 is zero.
fn weights(format: GgmlType, m: usize, kk: usize, seed: &mut u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    for i in 0..m * kk {
        let zero = i < kk;
        match format {
            GgmlType::F32 => {
                let v = if zero { 0.0 } else { uniform(seed, 0.2) };
                bytes.extend(v.to_le_bytes());
            }
            GgmlType::F16 => {
                // Exponents 8..=14: magnitudes from 2^-7 to just below 1.
                let r = next(seed);
                let bits = if zero {
                    0
                } else {
                    ((r >> 31) << 15) as u16
                        | ((8 + (r >> 8) % 7) << 10) as u16
                        | (r & 0x3FF) as u16
                };
                bytes.extend(bits.to_le_bytes());
            }
            GgmlType::Bf16 => {
                let v = if zero { 0.0 } else { uniform(seed, 0.2) };
                bytes.extend(((v.to_bits() >> 16) as u16).to_le_bytes());
            }
            _ => unreachable!(),
        }
    }
    bytes
}

#[test]
fn aligned_and_scalar_kernels_agree() {
    let stream = Stream::new().unwrap();
    let mut seed = 23;
    let mut checked = 0usize;
    for format in [GgmlType::F16, GgmlType::Bf16, GgmlType::F32] {
        for (m, kk) in [(513, 2560), (96, 2560), (128, 2048), (7, 132), (33, 640)] {
            let host_w = weights(format, m, kk, &mut seed);
            let mut w = stream.alloc::<u8>(host_w.len()).unwrap();
            stream.upload(&host_w, w.as_mut_slice()).unwrap();
            for n in (1..=8).chain([11, 19]) {
                let x: Vec<f32> = (0..n * kk).map(|_| uniform(&mut seed, 3.0)).collect();
                let mut aligned = stream.alloc::<f32>(n * kk).unwrap();
                stream.upload(&x, aligned.as_mut_slice()).unwrap();
                // One float in: the rows lose their 16-byte alignment.
                let mut shifted = stream.alloc::<f32>(n * kk + 1).unwrap();
                stream.upload(&x, shifted.slice_mut(1..n * kk + 1)).unwrap();
                let mut outputs = Vec::new();
                for input in [aligned.as_slice(), shifted.slice(1..n * kk + 1)] {
                    let mut out = stream.alloc::<f32>(n * m).unwrap();
                    k::small_gemm(
                        &stream,
                        QuantSlice::new(w.as_slice(), format),
                        input,
                        out.as_mut_slice(),
                        n as u32,
                        m as u32,
                        kk as u32,
                    )
                    .unwrap();
                    let mut host = vec![0f32; n * m];
                    stream.download(out.as_slice(), &mut host).unwrap();
                    outputs.push(host.iter().map(|v| v.to_bits()).collect::<Vec<u32>>());
                }
                let first = outputs[0].iter().zip(&outputs[1]).position(|(a, b)| a != b);
                assert_eq!(
                    first, None,
                    "{format:?} m {m} k {kk} n {n}: the kernels differ"
                );
                assert!(
                    outputs[0]
                        .iter()
                        .enumerate()
                        .any(|(i, &b)| i % m != 0 && f32::from_bits(b) != 0.0),
                    "{format:?} m {m} k {kk} n {n}: degenerate output"
                );
                checked += n * m;
            }
        }
    }
    eprintln!("checked {checked} outputs");
}
