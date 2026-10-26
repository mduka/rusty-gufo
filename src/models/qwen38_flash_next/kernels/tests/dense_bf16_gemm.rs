//! `dense_bf16_gemm` sums each output in K order for every token: for the BF16 indexer projections (k = 2560, 64 to 512 rows)
//! a row's result does not depend on its position in the batch, so a prefill
//! chunk split anywhere gives the same scores bit for bit, and every output
//! follows a double-precision oracle:
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next-kernels --features gpu --test dense_bf16_gemm

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{Bf16, DeviceBuffer, GgmlType, Launch, QuantSlice, Stream};
use gufo_qwen38_flash_next_kernels as k;

const K: usize = 2560;

/// Deterministic BF16 values in `[-scale, scale]` (`dense_gemm_bench`).
fn bf16(count: usize, mut seed: u32, scale: f32) -> Vec<Bf16> {
    (0..count)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            Bf16::from_f32(scale * ((seed % 2001) as f32 - 1000.0) / 1000.0)
        })
        .collect()
}

#[test]
fn rows_do_not_depend_on_their_position_in_the_batch() {
    let stream = Stream::new().unwrap();
    // An unaligned cache boundary, a wide prefill split and a ragged chunk.
    let splits = [(136, 94), (2048, 1025), (3487, 777)];
    let max_tokens = splits.iter().map(|&(total, _)| total).max().unwrap();
    for m in [512, 256, 128, 64] {
        let host_w = bf16(m * K, 51 + m as u32, 0.05);
        let host_x = bf16(max_tokens * K, 52, 3.0);
        let mut w = stream.alloc::<Bf16>(host_w.len()).unwrap();
        stream.upload(&host_w, w.as_mut_slice()).unwrap();
        let mut x = stream.alloc::<Bf16>(host_x.len()).unwrap();
        stream.upload(&host_x, x.as_mut_slice()).unwrap();
        let mut out = stream.alloc::<f32>(max_tokens * m).unwrap();
        let weights = || QuantSlice::new(w.as_slice().cast::<u8>().unwrap(), GgmlType::Bf16);
        for (total, boundary) in splits {
            let run = |out: &mut DeviceBuffer<f32>, from: usize, to: usize| {
                let launched = k::dense_bf16_gemm(
                    &stream,
                    weights(),
                    x.slice(from * K..to * K),
                    out.slice_mut(from * m..to * m),
                    to - from,
                    m,
                    K,
                )
                .unwrap();
                assert_eq!(launched, Launch::Launched);
            };
            run(&mut out, 0, total);
            let mut bulk = vec![0f32; total * m];
            stream.download(out.slice(0..total * m), &mut bulk).unwrap();
            run(&mut out, 0, boundary);
            run(&mut out, boundary, total);
            let mut split = vec![0f32; total * m];
            stream
                .download(out.slice(0..total * m), &mut split)
                .unwrap();
            let differs = bulk
                .iter()
                .zip(&split)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert_eq!(
                differs, None,
                "m {m}: the projection changed across a split at {boundary} of {total}"
            );
            // Tokens at every position within the 128-token tiles, against
            // the products summed in double precision.
            for t in (0..total).step_by(37).chain([total - 1]) {
                for row in 0..m {
                    let (mut sum, mut magnitude) = (0f64, 0f64);
                    for c in 0..K {
                        let term = f64::from(host_w[row * K + c].to_f32())
                            * f64::from(host_x[t * K + c].to_f32());
                        sum += term;
                        magnitude += term.abs();
                    }
                    let got = f64::from(bulk[t * m + row]);
                    assert!(
                        (got - sum).abs() <= 1e-5 * magnitude,
                        "m {m} token {t} row {row}: {got} vs {sum}"
                    );
                }
            }
        }
    }
}
