//! The C++ extension's `vision_ops_test.cpp`: the vision encoder's kernels
//! (attention, LayerNorm, position embedding, 2x2 merge, elementwise ops) in
//! both operand formats against double CPU evaluations.
//!
//! cargo test --profile gpu-test -p gufo-nemotron --features gpu --test vision_ops

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod kernel_common;

use gufo_hip::{Bf16, F16, Stream};
use gufo_nemotron_kernels as k;
use gufo_nemotron_kernels::sizes::VISION_HEAD_DIM;
use kernel_common::*;

#[derive(Clone, Copy, PartialEq)]
enum Operand {
    Bf16,
    F16,
}

impl Operand {
    fn code(self) -> u32 {
        match self {
            Self::Bf16 => 0,
            Self::F16 => 1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
        }
    }

    fn encode(self, x: f32) -> F16 {
        match self {
            Self::Bf16 => F16(Bf16::from_f32(x).0),
            Self::F16 => F16::from_f32(x),
        }
    }

    fn decode(self, v: F16) -> f32 {
        match self {
            Self::Bf16 => Bf16(v.0).to_f32(),
            Self::F16 => v.to_f32(),
        }
    }

    /// Unit roundoff: the largest relative error of rounding to nearest.
    fn epsilon(self) -> f64 {
        match self {
            Self::Bf16 => 1.0 / 256.0,
            Self::F16 => 1.0 / 2048.0,
        }
    }
}

/// |got - want| <= tolerance * (|want| + floor) everywhere.
fn compare(what: &str, got: &[f32], want: &[f64], tolerance: f64, floor: f64) {
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - w).abs() / (w.abs() + floor))
        .fold(0f64, f64::max);
    println!("{what}: worst relative error {worst}");
    assert!(worst <= tolerance, "{what} exceeds {tolerance}: {worst}");
}

fn decoded(s: &Stream, out: &gufo_hip::DeviceBuffer<F16>, n: usize, op: Operand) -> Vec<f32> {
    down(s, out, n).into_iter().map(|v| op.decode(v)).collect()
}

fn attention(s: &Stream, rng: &mut Rng, op: Operand, tokens: usize) {
    let heads = 16;
    let d = VISION_HEAD_DIM;
    let width = heads * d;
    let qkv: Vec<F16> = rng
        .normals(tokens * 3 * width, 1.0)
        .into_iter()
        .map(|v| op.encode(v))
        .collect();
    let input = up(s, &qkv);
    let mut out = zeros::<F16>(s, tokens * width);
    let scale = 1.0 / (d as f32).sqrt();
    ok(
        k::vision_attention(
            s,
            input.as_slice(),
            out.as_mut_slice(),
            tokens,
            heads,
            scale,
            op.code(),
        ),
        "vision attention",
    );
    let got = decoded(s, &out, tokens * width, op);
    let at = |t: usize, part: usize, h: usize, e: usize| {
        f64::from(op.decode(qkv[t * 3 * width + part * width + h * d + e]))
    };
    let mut want = vec![0f64; got.len()];
    for h in 0..heads {
        for t in 0..tokens {
            let mut scores: Vec<f64> = (0..tokens)
                .map(|kk| {
                    (0..d)
                        .map(|e| at(t, 0, h, e) * at(kk, 1, h, e))
                        .sum::<f64>()
                        * f64::from(scale)
                })
                .collect();
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0f64;
            for v in &mut scores {
                *v = (*v - m).exp();
                sum += *v;
            }
            for e in 0..d {
                let o: f64 = (0..tokens).map(|kk| scores[kk] * at(kk, 2, h, e)).sum();
                want[t * width + h * d + e] = o / sum;
            }
        }
    }
    let scale_ref = want.iter().fold(0f64, |m, w| m.max(w.abs()));
    compare(
        &format!("attention {} tokens {tokens}", op.name()),
        &got,
        &want,
        4.0 * op.epsilon(),
        0.25 * scale_ref,
    );
}

fn layer_norm(s: &Stream, rng: &mut Rng, op: Operand) {
    let (rows, width) = (37, 1280);
    let x = rng.normals(rows * width, 3.0);
    let w = rng.normals(width, 1.0);
    let b = rng.normals(width, 0.1);
    let (dx, dw, db) = (up(s, &x), up(s, &w), up(s, &b));
    let mut out = zeros::<F16>(s, x.len());
    ok(
        k::vision_layer_norm(
            s,
            dx.as_slice(),
            dw.as_slice(),
            db.as_slice(),
            out.as_mut_slice(),
            rows,
            width,
            1e-6,
            op.code(),
        ),
        "vision layer norm",
    );
    let got = decoded(s, &out, x.len(), op);
    let mut want = vec![0f64; x.len()];
    for r in 0..rows {
        let row = &x[r * width..(r + 1) * width];
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / width as f64;
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>();
        let rstd = 1.0 / (var / width as f64 + 1e-6).sqrt();
        for c in 0..width {
            want[r * width + c] =
                (f64::from(row[c]) - mean) * rstd * f64::from(w[c]) + f64::from(b[c]);
        }
    }
    compare(
        &format!("layer norm {}", op.name()),
        &got,
        &want,
        op.epsilon(),
        1e-2,
    );
}

/// PyTorch's bilinear source index (half-pixel centers, clamped at zero).
fn source(scale: f32, dst: usize, size: usize) -> (usize, usize, f64) {
    let src = (scale * (dst as f32 + 0.5) - 0.5).max(0.0);
    let i0 = (src as usize).min(size - 1);
    let i1 = i0 + usize::from(i0 < size - 1);
    (i0, i1, (f64::from(src) - i0 as f64).clamp(0.0, 1.0))
}

fn embed(s: &Stream, rng: &mut Rng, grid_rows: usize, grid_cols: usize) {
    let (side, width, prefix) = (128, 64, 10);
    let tokens = prefix + grid_rows * grid_cols;
    let positions = rng.normals(side * side * width, 1.0);
    let prefix_rows = rng.normals(prefix * width, 1.0);
    let x = rng.normals(tokens * width, 1.0);
    let (dp, dprefix) = (up(s, &positions), up(s, &prefix_rows));
    let mut dx = up(s, &x);
    ok(
        k::vision_embed(
            s,
            dx.as_mut_slice(),
            dprefix.as_slice(),
            dp.as_slice(),
            prefix,
            grid_rows,
            grid_cols,
            side,
            width,
        ),
        "vision embed",
    );
    let got = down(s, &dx, x.len());
    let scale = side as f32 / grid_rows.max(grid_cols) as f32;
    let mut want = vec![0f64; got.len()];
    for t in 0..tokens {
        for c in 0..width {
            let i = t * width + c;
            if t < prefix {
                want[i] = f64::from(prefix_rows[i]);
                continue;
            }
            let p = t - prefix;
            let (y0, y1, ly1) = source(scale, p / grid_cols, side);
            let (x0, x1, lx1) = source(scale, p % grid_cols, side);
            let pos = |yy: usize, xx: usize| f64::from(positions[(yy * side + xx) * width + c]);
            want[i] = f64::from(x[i])
                + (1.0 - ly1) * ((1.0 - lx1) * pos(y0, x0) + lx1 * pos(y0, x1))
                + ly1 * ((1.0 - lx1) * pos(y1, x0) + lx1 * pos(y1, x1));
        }
    }
    compare(
        &format!("embed {grid_rows}x{grid_cols}"),
        &got,
        &want,
        1e-5,
        1.0,
    );
}

fn merge_norm(s: &Stream, rng: &mut Rng, op: Operand) {
    let (prefix, grid_rows, grid_cols, width) = (10, 6, 10, 1280);
    let merged_width = 4 * width;
    let x = rng.normals((prefix + grid_rows * grid_cols) * width, 2.0);
    let w = rng.normals(merged_width, 1.0);
    let (dx, dw) = (up(s, &x), up(s, &w));
    let merged = grid_rows * grid_cols / 4;
    let mut out = zeros::<F16>(s, merged * merged_width);
    ok(
        k::vision_merge_norm(
            s,
            dx.as_slice(),
            dw.as_slice(),
            out.as_mut_slice(),
            prefix,
            grid_rows,
            grid_cols,
            width,
            1e-5,
            op.code(),
        ),
        "vision merge norm",
    );
    let got = decoded(s, &out, merged * merged_width, op);
    let mut want = vec![0f64; got.len()];
    for m in 0..merged {
        let (i, j) = (m / (grid_cols / 2), m % (grid_cols / 2));
        let v: Vec<f64> = (0..merged_width)
            .map(|kk| {
                let q = kk / width;
                let patch = (2 * i + q / 2) * grid_cols + 2 * j + q % 2;
                f64::from(x[(prefix + patch) * width + kk % width])
            })
            .collect();
        let squares: f64 = v.iter().map(|a| a * a).sum();
        let r = 1.0 / (squares / merged_width as f64 + 1e-5).sqrt();
        for kk in 0..merged_width {
            want[m * merged_width + kk] = f64::from(w[kk]) * v[kk] * r;
        }
    }
    compare(
        &format!("merge norm {}", op.name()),
        &got,
        &want,
        op.epsilon(),
        1e-2,
    );
}

fn erf(x: f64) -> f64 {
    if x.abs() >= 6.0 {
        return x.signum();
    }
    if x.abs() < 3.0 {
        let mut term = x;
        let mut sum = x;
        let x2 = x * x;
        let mut n = 0.0;
        while term.abs() > 1e-17 * sum.abs().max(1e-300) {
            n += 1.0;
            term *= -x2 / n;
            sum += term / (2.0 * n + 1.0);
        }
        return sum * 2.0 / std::f64::consts::PI.sqrt();
    }
    // Continued fraction for erfc.
    let ax = x.abs();
    let mut f = 0.0;
    for k in (1..60).rev() {
        f = f64::from(k) / 2.0 / (ax + f);
    }
    let erfc = (-ax * ax).exp() / std::f64::consts::PI.sqrt() / (ax + f);
    x.signum() * (1.0 - erfc)
}

fn elementwise(s: &Stream, rng: &mut Rng, op: Operand) {
    let (rows, cols) = (19, 384);
    let n = rows * cols;
    let y = rng.normals(n, 2.0);
    let bias = rng.normals(cols, 0.5);
    let x = rng.normals(n, 1.0);
    let (dy, dbias) = (up(s, &y), up(s, &bias));
    let mut dx = up(s, &x);
    let mut out = zeros::<F16>(s, n);
    ok(
        k::vision_bias(
            s,
            dy.as_slice(),
            dbias.as_slice(),
            out.as_mut_slice(),
            rows,
            cols,
            op.code(),
        ),
        "vision bias",
    );
    let want: Vec<f64> = (0..n)
        .map(|i| f64::from(y[i]) + f64::from(bias[i % cols]))
        .collect();
    compare(
        &format!("bias {}", op.name()),
        &decoded(s, &out, n, op),
        &want,
        op.epsilon(),
        0.0,
    );
    ok(
        k::vision_bias_gelu(
            s,
            dy.as_slice(),
            dbias.as_slice(),
            out.as_mut_slice(),
            rows,
            cols,
            op.code(),
        ),
        "bias gelu",
    );
    let want: Vec<f64> = (0..n)
        .map(|i| {
            let v = f64::from(y[i]) + f64::from(bias[i % cols]);
            0.5 * v * (1.0 + erf(v / 2f64.sqrt()))
        })
        .collect();
    compare(
        &format!("bias gelu {}", op.name()),
        &decoded(s, &out, n, op),
        &want,
        op.epsilon(),
        1e-3,
    );
    ok(
        k::vision_relu2(s, dy.as_slice(), out.as_mut_slice(), n, op.code()),
        "relu2",
    );
    let want: Vec<f64> = y.iter().map(|&v| f64::from(v).max(0.0).powi(2)).collect();
    // FP16 squares below its normal range lose precision.
    compare(
        &format!("relu2 {}", op.name()),
        &decoded(s, &out, n, op),
        &want,
        op.epsilon(),
        1e-4,
    );
    ok(
        k::vision_residual(
            s,
            dx.as_mut_slice(),
            dy.as_slice(),
            dbias.as_slice(),
            rows,
            cols,
        ),
        "residual",
    );
    let want: Vec<f64> = (0..n)
        .map(|i| f64::from(x[i]) + f64::from(y[i]) + f64::from(bias[i % cols]))
        .collect();
    compare("residual", &down(s, &dx, n), &want, 1e-6, 1.0);
}

#[test]
fn vision_kernels() {
    let s = stream();
    let mut rng = Rng(20_261_109);
    for op in [Operand::Bf16, Operand::F16] {
        for tokens in [37, 150, 1034] {
            attention(&s, &mut rng, op, tokens);
        }
        layer_norm(&s, &mut rng, op);
        merge_norm(&s, &mut rng, op);
        elementwise(&s, &mut rng, op);
    }
    embed(&s, &mut rng, 32, 32);
    embed(&s, &mut rng, 68, 120);
    embed(&s, &mut rng, 14, 188);
    embed(&s, &mut rng, 128, 104);
}
