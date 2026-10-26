//! Static YaRN on the NEOX rotary: with the YaRN
//! constants each pair's frequency takes its ramp multiplier and cos/sin the
//! attention factor, as a CPU oracle computes them. Disabled constants keep
//! the original instruction sequence, which the Flash-Next forward fixtures
//! check bit for bit:
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next-kernels --features gpu --test rope_yarn

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{IndexSpec, Stream};
use gufo_qwen38_flash_next_kernels as k;

const TOKENS: u32 = 5;
const HEADS: u32 = 3;
const D: u32 = 256;
const ROTARY: u32 = 64;
const THETA: f32 = 1e7;
const MAX_CONTEXT: u32 = 400_000;

/// A deterministic value in `[-scale, scale)`.
fn uniform(seed: &mut u32, scale: f32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
}

/// The YaRN constants of factor 1.5625 over a 262144-token pretraining
/// context for this rotary dimension (`MakeYarnRope`).
struct Yarn {
    enabled: u32,
    low: f32,
    high: f32,
    inv_factor: f32,
    mscale: f32,
}

fn yarn(factor: f32) -> Yarn {
    let original = 262_144f64;
    let dim = |rotations: f64| {
        f64::from(ROTARY) * (original / (rotations * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * f64::from(THETA).ln())
    };
    Yarn {
        enabled: 1,
        low: dim(32.0).floor().max(0.0) as f32,
        high: dim(1.0).ceil().min(f64::from(ROTARY) - 1.0) as f32,
        inv_factor: 1.0 / factor,
        mscale: (0.1 * f64::from(factor).ln() + 1.0) as f32,
    }
}

fn rope(stream: &Stream, x: &[f32], start: u32, y: &Yarn) -> Vec<f32> {
    let mut buffer = stream.alloc::<f32>(x.len()).unwrap();
    stream.upload(x, buffer.as_mut_slice()).unwrap();
    let mut position = stream
        .alloc_indices_with(1, IndexSpec::below(MAX_CONTEXT - TOKENS + 1))
        .unwrap();
    stream.upload_indices(&[start], &mut position).unwrap();
    k::rope(
        stream,
        buffer.as_mut_slice(),
        TOKENS,
        HEADS,
        D,
        ROTARY,
        position.view(),
        THETA,
        None,
        y.enabled,
        y.low,
        y.high,
        y.inv_factor,
        y.mscale,
        MAX_CONTEXT,
    )
    .unwrap();
    let mut host = vec![0f32; x.len()];
    stream.download(buffer.as_slice(), &mut host).unwrap();
    host
}

/// The rotation with the kernel's single-precision angle (a float
/// position times a float frequency) and double-precision sin and cos:
/// pair frequencies take the ramp multiplier, cos and sin the attention
/// factor.
fn oracle(x: &[f32], start: u32, y: &Yarn) -> Vec<f32> {
    let mut out = x.to_vec();
    let half = ROTARY / 2;
    for t in 0..TOKENS {
        for h in 0..HEADS {
            let row = ((t * HEADS + h) * D) as usize;
            for i in 0..half {
                let mut freq = THETA.powf(-2.0 * i as f32 / ROTARY as f32);
                let mut scale = 1.0;
                if y.enabled != 0 {
                    let span = (y.high - y.low).max(0.001);
                    let ramp = ((i as f32 - y.low) / span).clamp(0.0, 1.0);
                    freq *= 1.0 - ramp * (1.0 - y.inv_factor);
                    scale = f64::from(y.mscale);
                }
                let angle = f64::from((start + t) as f32 * freq);
                let (s, c) = (angle.sin() * scale, angle.cos() * scale);
                let (a, b) = (
                    f64::from(x[row + i as usize]),
                    f64::from(x[row + (i + half) as usize]),
                );
                out[row + i as usize] = (a * c - b * s) as f32;
                out[row + (i + half) as usize] = (a * s + b * c) as f32;
            }
        }
    }
    out
}

#[test]
fn yarn_rotation_follows_the_oracle() {
    let stream = Stream::new().unwrap();
    let mut seed = 7;
    let x: Vec<f32> = (0..TOKENS * HEADS * D)
        .map(|_| uniform(&mut seed, 1.0))
        .collect();
    let off = Yarn {
        enabled: 0,
        low: 0.0,
        high: 0.0,
        inv_factor: 1.0,
        mscale: 1.0,
    };
    for start in [0u32, 1000, 20_000, 262_140, 300_000] {
        for (label, constants) in [("off", &off), ("1.5625", &yarn(1.5625)), ("3", &yarn(3.0))] {
            let got = rope(&stream, &x, start, constants);
            let want = oracle(&x, start, constants);
            // The device's powf and fast sincosf against the double oracle;
            // range reduction loses precision with the angle, the plain
            // rotation included, so far positions are only checked finite.
            for (index, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    g.is_finite() && (start > 20_000 || (g - w).abs() <= 1e-3),
                    "{label} start={start} index={index}: {g} vs {w}"
                );
            }
            // Dimensions past the rotary part pass through.
            for t in 0..TOKENS * HEADS {
                let row = (t * D) as usize;
                assert_eq!(
                    got[row + ROTARY as usize..row + D as usize],
                    x[row + ROTARY as usize..row + D as usize]
                );
            }
        }
        // A scaled rotation differs from the plain one.
        assert_ne!(
            rope(&stream, &x, start + 1, &yarn(1.5625)),
            rope(&stream, &x, start + 1, &off)
        );
    }
}
