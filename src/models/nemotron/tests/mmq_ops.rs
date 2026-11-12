//! The C++ extension's `mmq_ops_test.cpp`: the vendored quantized matrix
//! kernels (nb_mmq.h) against an independent CPU oracle, ggml's
//! dequantizers and double accumulation over the unquantized activations.
//!
//! The kernels quantize activations to Q8_1, so outputs differ from the
//! oracle by rounding noise. Each output's expected noise comes from its own
//! weights and activation scales:
//!   var = sum_b (d_b^2 / 12) * sum_{k in b} w_k^2      (code rounding)
//!       + sum_b (w_b . x_b)^2 * (2^-11)^2 / 3          (FP16 activation scale)
//! with d_b = max|x| / 127 over each 32-value block. Every output must lie
//! within MAX_Z standard deviations, the relative RMS error must stay at the
//! Q8_1 level, and outputs must correlate with the oracle.
//!
//! cargo test --profile gpu-test -p gufo-nemotron --features gpu --test mmq_ops

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod kernel_common;

use gufo_formats::GgmlType;
use gufo_formats::dequant::dequantize;
use gufo_hip::{DeviceSlice, DeviceSliceMut, Launch, LaunchError, Stream};
use gufo_nemotron::iq2xxs::dequantize_iq2xxs;
use gufo_nemotron::nvfp4::dequantize_nvfp4;
use gufo_nemotron_kernels as k;
use kernel_common::*;

const MAX_Z: f64 = 7.0;
const MAX_REL_RMS: f64 = 0.012;
const MIN_CORRELATION: f64 = 0.999;

/// An `mmq_dense_*` launcher.
type Dense = fn(
    &Stream,
    DeviceSlice<'_, u8>,
    DeviceSlice<'_, f32>,
    DeviceSliceMut<'_, f32>,
    usize,
    usize,
    usize,
) -> Result<Launch, LaunchError>;

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Q8_0,
    Iq2Xxs,
    Nvfp4,
}

impl Format {
    fn name(self) -> &'static str {
        match self {
            Self::Q8_0 => "Q8_0",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Nvfp4 => "NVFP4",
        }
    }

    fn row_bytes(self, k: usize) -> usize {
        match self {
            Self::Q8_0 => k / 32 * 34,
            Self::Iq2Xxs => k / 256 * 66,
            Self::Nvfp4 => k / 64 * 36,
        }
    }

    fn dense(self) -> Dense {
        match self {
            Self::Q8_0 => k::mmq_dense_q8_0,
            Self::Iq2Xxs => k::mmq_dense_iq2_xxs,
            Self::Nvfp4 => k::mmq_dense_nvfp4,
        }
    }

    fn dequantize_row(self, row: &[u8], out: &mut [f32]) {
        match self {
            Self::Q8_0 => dequantize(GgmlType::Q8_0, row, out).unwrap(),
            Self::Iq2Xxs => dequantize_iq2xxs(row, out),
            Self::Nvfp4 => dequantize_nvfp4(row, out),
        }
    }
}

/// Positive FP16 with a biased exponent in [e_lo, e_hi] and a random
/// mantissa.
fn random_half(rng: &mut Rng, e_lo: u32, e_hi: u32) -> u16 {
    let e = e_lo + rng.below(e_hi - e_lo + 1);
    ((e << 10) | rng.below(1024)) as u16
}

/// Random but valid blocks (the C++ test's `RandomWeights`).
fn random_weights(format: Format, rows: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut w = vec![0u8; rows * format.row_bytes(k)];
    let mut at = 0;
    while at < w.len() {
        match format {
            Format::Q8_0 => {
                w[at..at + 2].copy_from_slice(&random_half(rng, 5, 8).to_le_bytes());
                for i in 0..32 {
                    w[at + 2 + i] = (rng.below(255) as i32 - 127) as i8 as u8;
                }
                at += 34;
            }
            Format::Iq2Xxs => {
                w[at..at + 2].copy_from_slice(&random_half(rng, 4, 8).to_le_bytes());
                for i in (2..66).step_by(8) {
                    let bits = rng.next();
                    for b in 0..8 {
                        w[at + i + b] = (bits >> (8 * b)) as u8;
                    }
                }
                at += 66;
            }
            Format::Nvfp4 => {
                for sc in 0..4 {
                    w[at + sc] = (((4 + rng.below(6)) << 3) | rng.below(8)) as u8;
                }
                for i in (4..36).step_by(8) {
                    let bits = rng.next();
                    for b in 0..8 {
                        w[at + i + b] = (bits >> (8 * b)) as u8;
                    }
                }
                at += 36;
            }
        }
    }
    w
}

#[derive(Clone, Copy, Default)]
struct Oracle {
    value: f64,
    var: f64,
}

/// One output: dequantized weight row `w` against activation row `x`.
fn dot(w: &[f32], x: &[f32], k: usize) -> Oracle {
    let half_scale = (2f64).powi(-22) / 3.0;
    let mut o = Oracle::default();
    let mut abs_sum = 0f64;
    for b in 0..k / 32 {
        let amax = x[b * 32..b * 32 + 32]
            .iter()
            .fold(0f32, |m, v| m.max(v.abs()));
        let d = f64::from(amax) / 127.0;
        let (mut sum, mut ww) = (0f64, 0f64);
        for i in b * 32..b * 32 + 32 {
            let wi = f64::from(w[i]);
            sum += wi * f64::from(x[i]);
            ww += wi * wi;
            abs_sum += (wi * f64::from(x[i])).abs();
        }
        o.value += sum;
        o.var += d * d / 12.0 * ww + sum * sum * half_scale;
    }
    // Float accumulation on the GPU.
    let fp = 2e-6 * abs_sum;
    o.var += fp * fp;
    o
}

fn report(label: &str, out: &[f32], reference: &[Oracle]) -> bool {
    let n = out.len() as f64;
    let finite = out.iter().all(|v| v.is_finite());
    let mr = reference.iter().map(|o| o.value).sum::<f64>() / n;
    let mo = out.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let (mut se, mut sr, mut so, mut sro) = (0f64, 0f64, 0f64, 0f64);
    let (mut max_abs, mut max_z, mut rms_ref, mut rms_out) = (0f64, 0f64, 0f64, 0f64);
    for (o, r) in out.iter().zip(reference) {
        let (o, rv) = (f64::from(*o), r.value);
        let e = o - rv;
        max_abs = max_abs.max(e.abs());
        max_z = max_z.max(e.abs() / r.var.sqrt());
        se += e * e;
        sr += (rv - mr) * (rv - mr);
        so += (o - mo) * (o - mo);
        sro += (rv - mr) * (o - mo);
        rms_ref += rv * rv;
        rms_out += o * o;
    }
    let rms_ref = (rms_ref / n).sqrt();
    let rms_out = (rms_out / n).sqrt();
    let rel_rms = (se / n).sqrt() / rms_ref;
    let corr = sro / (sr * so).sqrt();
    let pass = finite
        && max_z <= MAX_Z
        && rel_rms <= MAX_REL_RMS
        && corr >= MIN_CORRELATION
        && rms_out >= 0.5 * rms_ref;
    println!(
        "{:<4} {label:<58} max|err| {max_abs:.3e} ({:.4} rms) max z {max_z:5.2} rel rms {rel_rms:.5} corr {corr:.7}",
        if pass { "ok" } else { "FAIL" },
        max_abs / rms_ref
    );
    pass
}

fn dense_case(
    s: &Stream,
    format: Format,
    m: usize,
    k: usize,
    ns: &[usize],
    rng: &mut Rng,
) -> usize {
    let w = random_weights(format, m, k, rng);
    let row_bytes = format.row_bytes(k);
    let dw = up(s, &w);
    let mut failures = 0;
    for &n in ns {
        let x: Vec<f32> = (0..n * k).map(|_| rng.normal()).collect();
        let mut reference = vec![Oracle::default(); m * n];
        std::thread::scope(|scope| {
            let workers = std::thread::available_parallelism()
                .map_or(1, usize::from)
                .min(m);
            let w = &w;
            let x = &x;
            let handles: Vec<_> = (0..workers)
                .map(|worker| {
                    scope.spawn(move || {
                        let mut wrow = vec![0f32; k];
                        let mut rows = Vec::new();
                        for row in (worker..m).step_by(workers) {
                            format.dequantize_row(
                                &w[row * row_bytes..(row + 1) * row_bytes],
                                &mut wrow,
                            );
                            for c in 0..n {
                                rows.push((c * m + row, dot(&wrow, &x[c * k..(c + 1) * k], k)));
                            }
                        }
                        rows
                    })
                })
                .collect();
            for handle in handles {
                for (at, oracle) in handle.join().unwrap() {
                    reference[at] = oracle;
                }
            }
        });
        let dx = up(s, &x);
        let mut out = s.alloc::<f32>(m * n).unwrap();
        s.fill_bytes(out.as_mut_slice(), 0xFF).unwrap();
        let label = format!(
            "dense {:<7} M={m:5} K={k:4} N={n:3} ({})",
            format.name(),
            if n <= 8 { "vector" } else { "tiled" }
        );
        match format.dense()(s, dw.as_slice(), dx.as_slice(), out.as_mut_slice(), m, n, k) {
            Ok(Launch::Launched) => {
                let got = down(s, &out, m * n);
                if !report(&label, &got, &reference) {
                    failures += 1;
                }
            }
            other => {
                println!("FAIL {label}: {other:?}");
                failures += 1;
            }
        }
    }
    failures
}

#[test]
fn quantized_products_stay_within_q8_1_noise() {
    let s = stream();
    let mut rng = Rng(0x6E65_6D6F_7472_6F6E);
    // Dense: N <= 8 takes the MMVQ vector kernels, larger N the MMQ tiles.
    let all_n = [1, 2, 5, 8, 33, 256];
    let mut failures = 0;
    failures += dense_case(&s, Format::Q8_0, 4096, 4096, &all_n, &mut rng);
    failures += dense_case(&s, Format::Q8_0, 18560, 4096, &[1, 5, 33], &mut rng);
    failures += dense_case(&s, Format::Q8_0, 4096, 8192, &[1, 8, 33], &mut rng);
    // M not a multiple of the 64-row tile exercises the bounds-checked MMQ.
    failures += dense_case(&s, Format::Q8_0, 1000, 2304, &[2, 33, 256], &mut rng);
    failures += dense_case(&s, Format::Nvfp4, 5376, 4096, &all_n, &mut rng);
    failures += dense_case(&s, Format::Nvfp4, 4096, 5376, &[1, 8, 256], &mut rng);
    failures += dense_case(&s, Format::Iq2Xxs, 1024, 4096, &all_n, &mut rng);
    failures += dense_case(&s, Format::Iq2Xxs, 256, 1024, &[1, 33], &mut rng);
    k::mmq_release_stream(&s).unwrap();
    assert_eq!(failures, 0, "quantized product cases failed");
}
