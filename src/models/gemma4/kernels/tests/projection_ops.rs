//! The extension's `projection_ops_test.cpp`: a tripwire over the projection
//! kernels at Gemma 4 shapes. Decode (one-row GEMV) and verification
//! (small-batch FP32 GEMM) agree bit for bit at every width, so greedy
//! speculation reproduces autoregressive decoding; both match the CPU
//! dequantized dot product, and the W8A8 prefill GEMM stays within its
//! activation-rounding envelope. The binary16 GEMV and prefill GEMM, the
//! WMMA projections (width invariant, also over the grouped layout), the
//! grouped and GeGLU prefill GEMMs, the BF16 rewrite and the drafter head
//! repack. (The C++ test's cold-weight timings are not repeated.)
//!
//! cargo test --profile gpu-test -p gufo-gemma4-kernels --features gpu --test projection_ops

#![cfg(feature = "gpu")]
#![allow(
    clippy::unwrap_used,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

mod common;

use common::*;
use gufo_gemma4_kernels as k;
use gufo_gemma4_kernels::sizes;
use gufo_hip::{F16, GgmlType, Launch};

const FORMATS: [Format; 5] = [Q4_K, Q5_K, Q6_K, Q8_0, Q4_0];

fn check_shape(f: &Format, m: usize, kk: usize, rng: &mut Rng) {
    let s = stream();
    let name = format!("{} {m}x{kk}", f.name);
    let w = random_matrix(f, m, kk, rng);
    const ROWS: usize = 16;
    const PREFILL_ROWS: usize = 40;
    let x = rng.normals(PREFILL_ROWS * kk, 1.0);
    let dw = up(&s, &w);
    let dx = up(&s, &x);
    let mut dy = zeros::<f32>(&s, PREFILL_ROWS * m);

    // Decode reference rows.
    for r in 0..ROWS {
        ok(
            k::gemv(
                &s,
                quant(&dw, f.ty),
                dx.slice(r * kk..(r + 1) * kk),
                dy.slice_mut(r * m..(r + 1) * m),
                m,
                kk,
            ),
            &name,
        );
    }
    let gemv = down(&s, &dy, ROWS * m);

    // Every verification width reproduces the decode rows bit for bit,
    // including the double-stage pass at one row (drafter-mode decode).
    let fill =
        |dy: &mut gufo_hip::DeviceBuffer<f32>| s.fill_bytes(dy.as_mut_slice(), 0xFF).unwrap();
    let double_stage = |dy: &mut gufo_hip::DeviceBuffer<f32>, width: usize| {
        k::kquant_small_batch_double_stage(
            &s,
            quant(&dw, f.ty),
            dx.slice(0..width * kk),
            dy.slice_mut(0..width * m),
            width,
            m,
            kk,
        )
    };
    let staged = matches!(
        f.ty,
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q4_0
    );
    if staged {
        fill(&mut dy);
        ok(double_stage(&mut dy, 1), &name);
        assert!(
            same_bits(&down(&s, &dy, m), &gemv[..m]),
            "{name}: one-row double stage differs from decode GEMV"
        );
    }
    for width in 2..=16 {
        fill(&mut dy);
        ok(
            k::batched_quant_gemm_f32(
                &s,
                quant(&dw, f.ty),
                dx.slice(0..width * kk),
                dy.slice_mut(0..width * m),
                width,
                m,
                kk,
            ),
            &name,
        );
        assert!(
            same_bits(&down(&s, &dy, width * m), &gemv[..width * m]),
            "{name}: width {width} differs from decode GEMV"
        );
        // Q8_0 past eight rows in one weight pass.
        if f.ty == GgmlType::Q8_0 && width > 8 {
            fill(&mut dy);
            ok(
                k::q8_0_small_batch_wide(
                    &s,
                    quant(&dw, f.ty),
                    dx.slice(0..width * kk),
                    dy.slice_mut(0..width * m),
                    width,
                    m,
                    kk,
                ),
                &format!("{name}: wide Q8_0 small batch"),
            );
            assert!(
                same_bits(&down(&s, &dy, width * m), &gemv[..width * m]),
                "{name}: wide Q8_0 width {width} differs from decode GEMV"
            );
        }
        // The double-stage configuration verification uses for Gemma shapes.
        if staged {
            fill(&mut dy);
            ok(double_stage(&mut dy, width), &name);
            assert!(
                same_bits(&down(&s, &dy, width * m), &gemv[..width * m]),
                "{name}: double-stage width {width} differs from decode GEMV"
            );
        }
    }

    // Decode against an FP64 dot of the CPU-dequantized row, on a row
    // sample; the error is bounded relative to sum |w x| (FP32 accumulation).
    let outputs = (0..m).step_by((m / 97).max(1));
    let worst = worst_dot_error(
        f,
        &w,
        kk,
        |r| x[r * kk..(r + 1) * kk].to_vec(),
        |r, o| gemv[r * m + o],
        (0..2).map(|r| (r, 0)),
        outputs.clone(),
    );
    assert!(worst < 2e-6, "{name}: decode error {worst}");

    // The Gemma autoregressive GEMV matches the FP64 dot (256-multiple
    // widths).
    if kk.is_multiple_of(256) {
        for r in 0..2 {
            ok(
                k::kquant_gemv(
                    &s,
                    quant(&dw, f.ty),
                    dx.slice(r * kk..(r + 1) * kk),
                    dy.slice_mut(r * m..(r + 1) * m),
                    m as u32,
                    kk as u32,
                ),
                &format!("{name}: Gemma GEMV"),
            );
        }
        let one = down(&s, &dy, 2 * m);
        let worst = worst_dot_error(
            f,
            &w,
            kk,
            |r| x[r * kk..(r + 1) * kk].to_vec(),
            |r, o| one[r * m + o],
            (0..2).map(|r| (r, 0)),
            outputs,
        );
        assert!(worst < 2e-6, "{name}: Gemma GEMV error {worst}");
    }

    // Prefill W8A8 stays within Q8_1 activation rounding.
    let mut dq = zeros::<u8>(&s, sizes::q8_activation_read_bytes(PREFILL_ROWS, kk));
    ok(
        k::quantize_q8_1_from_f32(
            &s,
            dx.as_slice(),
            dq.slice_mut(0..sizes::q8_activation_bytes(PREFILL_ROWS, kk)),
            PREFILL_ROWS,
            kk,
        ),
        "quantize",
    );
    ok(
        k::batched_quant_gemm_q8_1(
            &s,
            quant(&dw, f.ty),
            dq.as_slice(),
            dy.as_mut_slice(),
            PREFILL_ROWS,
            m,
            kk,
        ),
        &name,
    );
    let prefill = down(&s, &dy, PREFILL_ROWS * m);
    let (mut err2, mut ref2) = (0f64, 0f64);
    for i in 0..ROWS * m {
        assert!(prefill[i].is_finite(), "{name}: non-finite prefill");
        let d = f64::from(prefill[i]) - f64::from(gemv[i]);
        err2 += d * d;
        ref2 += f64::from(gemv[i]) * f64::from(gemv[i]);
    }
    let rel = (err2 / ref2).sqrt();
    assert!(rel < 2e-2, "{name}: prefill relative error {rel}");
}

#[test]
fn quantized_shapes() {
    let mut rng = Rng(11);
    // (M, K) of every target projection, plus the drafter's widths.
    let shapes = [
        (8192, 5376),
        (4096, 5376),
        (16384, 5376),
        (2048, 5376),
        (5376, 8192),
        (5376, 16384),
        (21504, 5376),
        (5376, 21504),
        (1024, 10752),
        (8192, 1024),
        (5376, 1024),
    ];
    for f in &FORMATS {
        for &(m, kk) in &shapes {
            check_shape(f, m, kk, &mut rng);
        }
    }
    // The MTP drafters' remaining one-row shapes, Q8_0 and Q4_0 (QAT).
    let draft_shapes = [
        (1024, 8192),
        (1024, 4096),
        (1024, 5632),
        (4096, 1024),
        (16384, 1024),
        (2816, 1024),
    ];
    for f in [&Q8_0, &Q4_0] {
        for &(m, kk) in &draft_shapes {
            check_shape(f, m, kk, &mut rng);
        }
    }
    // The 26B-A4B keeps attention, the dense MLP and the tied head in Q8_0
    // in every Unsloth quant; its drafter reads 1024-wide rows.
    let moe_shapes = [
        (4096, 2816),
        (2048, 2816),
        (8192, 2816),
        (1024, 2816),
        (2816, 4096),
        (2816, 8192),
        (2112, 2816),
        (2816, 2112),
        (1024, 5632),
        (2816, 1024),
        (4096, 1024),
        (1024, 4096),
    ];
    for &(m, kk) in &moe_shapes {
        check_shape(&Q8_0, m, kk, &mut rng);
    }
}

/// Binary16 weights: the small-batch GEMV rounds every width like one row
/// and matches the FP64 dot; the binary16 WMMA prefill GEMM stays within
/// activation rounding.
fn check_half_shape(m: usize, kk: usize, rng: &mut Rng) {
    let s = stream();
    let name = format!("F16 {m}x{kk}");
    const ROWS: usize = sizes::MAX_HALF_GEMV_ROWS;
    const PREFILL_ROWS: usize = 40;
    let w: Vec<F16> = (0..m * kk).map(|_| half(rng.normal() * 0.02)).collect();
    let x = rng.normals(PREFILL_ROWS * kk, 1.0);
    let dw = up(&s, &w);
    let dx = up(&s, &x);
    let mut dy = zeros::<f32>(&s, PREFILL_ROWS * m);
    for r in 0..ROWS {
        ok(
            k::half_gemv(
                &s,
                dw.as_slice(),
                dx.slice(r * kk..(r + 1) * kk),
                dy.slice_mut(r * m..(r + 1) * m),
                1,
                m as u32,
                kk as u32,
            ),
            &name,
        );
    }
    let one = down(&s, &dy, ROWS * m);
    for width in 2..=ROWS {
        s.fill_bytes(dy.as_mut_slice(), 0xFF).unwrap();
        ok(
            k::half_gemv(
                &s,
                dw.as_slice(),
                dx.slice(0..width * kk),
                dy.slice_mut(0..width * m),
                width as u32,
                m as u32,
                kk as u32,
            ),
            &name,
        );
        assert!(
            same_bits(&down(&s, &dy, width * m), &one[..width * m]),
            "{name}: width {width} differs from one row"
        );
    }
    let bytes: Vec<u8> = w.iter().flat_map(|h| h.0.to_le_bytes()).collect();
    let worst = worst_dot_error(
        &F16_FORMAT,
        &bytes,
        kk,
        |r| x[r * kk..(r + 1) * kk].to_vec(),
        |r, o| one[r * m + o],
        (0..2).map(|r| (r, 0)),
        (0..m).step_by((m / 97).max(1)),
    );
    assert!(worst < 2e-6, "{name}: GEMV error {worst}");

    // Prefill over binary16 activation rows.
    let xh: Vec<F16> = x.iter().map(|&v| half(v)).collect();
    let dxh = up(&s, &xh);
    let dwq = up(&s, &bytes);
    ok(
        k::unquantized_f16_gemm(
            &s,
            quant(&dwq, GgmlType::F16),
            dxh.as_slice(),
            dy.as_mut_slice(),
            PREFILL_ROWS,
            m,
            kk,
            1024,
        ),
        &name,
    );
    let prefill = down(&s, &dy, PREFILL_ROWS * m);
    let (mut err2, mut ref2) = (0f64, 0f64);
    for i in 0..ROWS * m {
        let d = f64::from(prefill[i]) - f64::from(one[i]);
        err2 += d * d;
        ref2 += f64::from(one[i]) * f64::from(one[i]);
    }
    let rel = (err2 / ref2).sqrt();
    assert!(rel < 2e-3, "{name}: prefill relative error {rel}");
}

#[test]
fn half_shapes() {
    let mut rng = Rng(12);
    // Binary16 projections of the UD-Q8_K_XL quants, fused where the loader
    // fuses them.
    for (m, kk) in [
        (8192, 5376),
        (4096, 5376),
        (18432, 5376),
        (43008, 5376),
        (5376, 21504),
        (9216, 2816),
        (2816, 8192),
        (4224, 2816),
        (2816, 2112),
    ] {
        check_half_shape(m, kk, &mut rng);
    }
}

fn group_weights(
    s: &gufo_hip::Stream,
    f: &Format,
    dw: &gufo_hip::DeviceBuffer<u8>,
    m: usize,
    kk: usize,
) -> gufo_hip::DeviceBuffer<u8> {
    let mut dg = zeros::<u8>(s, m * f.row_bytes(kk));
    let group = match f.ty {
        GgmlType::Q4K | GgmlType::Q4_0 => k::group_weights_144,
        GgmlType::Q5K => k::group_weights_176,
        GgmlType::Q6K => k::group_weights_210,
        _ => k::group_weights_272,
    };
    ok(
        group(s, quant(dw, f.ty), dg.as_mut_slice(), m as u32, kk as u32),
        "group weights",
    );
    dg
}

fn grouped_supported(f: &Format, m: usize, kk: usize) -> bool {
    kk.is_multiple_of(256)
        && (matches!(
            f.ty,
            GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q4_0
        ) || (f.ty == GgmlType::Q8_0 && m >= 2048 && kk >= 4096))
}

/// The WMMA projections: every width from 1 to 16 rounds a row identically,
/// and each output stays within FP32 rounding of the FP64 dot of the
/// dequantized row. Activation sub-blocks span zeros, tiny values and
/// magnitudes past the binary16 range.
fn check_wmma_shape(f: &Format, m: usize, kk: usize, rng: &mut Rng) {
    let s = stream();
    let name = format!("{} WMMA {m}x{kk}", f.name);
    const ROWS: usize = sizes::WMMA_GEMV_ROWS;
    let w = random_matrix(f, m, kk, rng);
    let mut x = vec![0f32; ROWS * kk];
    for b in 0..x.len() / 32 {
        let c = rng.below(10);
        let scale = match c {
            0 => 0.0,
            1 => 1e-6,
            2 => 3e4,
            _ => 1.0,
        };
        for j in 0..32 {
            x[b * 32 + j] = scale * rng.normal();
        }
    }
    let dw = up(&s, &w);
    let dx = up(&s, &x);
    let mut dy = zeros::<f32>(&s, ROWS * m);
    let mut scratch = zeros::<u8>(&s, sizes::wmma_gemv_scratch_bytes(kk));
    let mut launch = |w: &gufo_hip::DeviceBuffer<u8>,
                      first: usize,
                      width: usize,
                      grouped: bool,
                      dy: &mut gufo_hip::DeviceBuffer<f32>| {
        s.fill_bytes(dy.as_mut_slice(), 0xFF).unwrap();
        ok(
            k::wmma_gemv(
                &s,
                quant(w, f.ty),
                dx.slice(first * kk..(first + width) * kk),
                dy.slice_mut(0..width * m),
                width as u32,
                m as u32,
                kk as u32,
                scratch.as_mut_slice(),
                grouped,
            ),
            &name,
        );
        down(&s, dy, width * m)
    };
    let full = launch(&dw, 0, ROWS, false, &mut dy);
    for width in 1..ROWS {
        // Each width starts at a different row, so neighbours change too.
        let first = (width * 5) % (ROWS - width + 1);
        let part = launch(&dw, first, width, false, &mut dy);
        assert!(
            same_bits(&part, &full[first * m..(first + width) * m]),
            "{name}: width {width} differs"
        );
    }
    // The grouped layout (MTP targets) at every width, bit for bit.
    if grouped_supported(f, m, kk) {
        let dg = group_weights(&s, f, &dw, m, kk);
        for width in 1..=ROWS {
            let part = launch(&dg, 0, width, true, &mut dy);
            assert!(
                same_bits(&part, &full[..width * m]),
                "{name}: grouped width {width} differs"
            );
        }
        println!("{name}: grouped layout identical at every width");
    }
    let worst = worst_dot_error(
        f,
        &w,
        kk,
        |r| x[r * kk..(r + 1) * kk].to_vec(),
        |r, o| full[r * m + o],
        (0..ROWS).step_by(5).map(|r| (r, 0)),
        (0..m).step_by((m / 61).max(1)),
    );
    assert!(
        full.iter().all(|v| v.is_finite()),
        "{name}: non-finite output"
    );
    println!("{name}: width-invariant, error {worst}");
    assert!(worst < 1e-6, "{name}: error {worst}");
}

#[test]
fn wmma_shapes() {
    let mut rng = Rng(13);
    // The 31B's Q4_K shapes, a partial row block and the split reduction.
    for (m, kk) in [
        (8192, 5376),
        (4096, 5376),
        (18432, 5376),
        (43008, 5376),
        (5376, 8192),
        (5376, 16384),
        (5376, 21504),
        (4100, 5376),
        (4100, 21504),
    ] {
        check_wmma_shape(&Q4_K, m, kk, &mut rng);
    }
    // Its Q5_K / Q6_K layers and the QAT target's Q4_0, partial row blocks.
    check_wmma_shape(&Q5_K, 43008, 5376, &mut rng);
    check_wmma_shape(&Q5_K, 4100, 21504, &mut rng);
    check_wmma_shape(&Q6_K, 4096, 5376, &mut rng);
    check_wmma_shape(&Q6_K, 5376, 21504, &mut rng);
    check_wmma_shape(&Q6_K, 4100, 5376, &mut rng);
    check_wmma_shape(&Q4_0, 43008, 5376, &mut rng);
    check_wmma_shape(&Q4_0, 4100, 21504, &mut rng);
    // The UD-Q8_K_XL targets' Q8_0 projections.
    for (m, kk) in [
        (16384, 5376),
        (43008, 5376),
        (5376, 8192),
        (5376, 16384),
        (5376, 21504),
        (4100, 5376),
        (4100, 21504),
    ] {
        check_wmma_shape(&Q8_0, m, kk, &mut rng);
    }
    // The 26B-A4B's K-quant and Q4_0 quants over 2816-wide rows, up to the
    // vocabulary head.
    let moe_shapes = [
        (4096, 2816),
        (2048, 2816),
        (9216, 2816),
        (2816, 4096),
        (2816, 8192),
        (4224, 2816),
        (262_144, 2816),
    ];
    for f in [&Q4_K, &Q6_K, &Q4_0] {
        for &(m, kk) in &moe_shapes {
            check_wmma_shape(f, m, kk, &mut rng);
        }
    }
    // Q8_0 takes the WMMA projections from 2048 outputs and 4096 inputs.
    let s = stream();
    for &(m, kk) in &moe_shapes {
        if kk >= 4096 {
            check_wmma_shape(&Q8_0, m, kk, &mut rng);
        } else {
            let w = random_matrix(&Q8_0, m, kk, &mut rng);
            let dw = up(&s, &w);
            let dx = zeros::<f32>(&s, kk);
            let mut dy = zeros::<f32>(&s, m);
            let mut scratch = zeros::<u8>(&s, sizes::wmma_gemv_scratch_bytes(kk));
            let result = k::wmma_gemv(
                &s,
                quant(&dw, GgmlType::Q8_0),
                dx.as_slice(),
                dy.as_mut_slice(),
                1,
                m as u32,
                kk as u32,
                scratch.as_mut_slice(),
                false,
            );
            assert!(
                matches!(result, Ok(Launch::Unsupported)),
                "Q8_0 WMMA should decline {m}x{kk}: {result:?}"
            );
        }
    }
}

/// The prefill GEMM over grouped Q8_0 weights (MTP targets) equals the
/// row-order GEMM bit for bit, for every plan and a narrow batch; at the
/// 26B-A4B's attention outputs Gemma's dense GEMM gives the same sums.
fn check_grouped_half_gemm(m: usize, kk: usize, rng: &mut Rng) {
    let s = stream();
    let name = format!("Q8_0 grouped prefill {m}x{kk}");
    let w = random_matrix(&Q8_0, m, kk, rng);
    const BATCH: usize = 1100;
    let x: Vec<F16> = (0..BATCH * kk).map(|_| half(rng.normal())).collect();
    let dw = up(&s, &w);
    let dg = group_weights(&s, &Q8_0, &dw, m, kk);
    let dx = up(&s, &x);
    let mut dy = zeros::<f32>(&s, BATCH * m);
    let mut gemm = |w: &gufo_hip::DeviceBuffer<u8>, batch: usize, plan: u32, grouped: bool| {
        s.fill_bytes(dy.as_mut_slice(), 0xFF).unwrap();
        ok(
            k::dense_f16_gemm(
                &s,
                quant(w, GgmlType::Q8_0),
                dx.slice(0..batch * kk),
                dy.slice_mut(0..batch * m),
                batch,
                m,
                kk,
                plan,
                grouped,
            ),
            &name,
        );
        down(&s, &dy, batch * m)
    };
    for plan in 0..4u32 {
        for batch in [BATCH, 40] {
            let want = gemm(&dw, batch, plan, false);
            let got = gemm(&dg, batch, plan, true);
            assert!(
                same_bits(&want, &got),
                "{name}: plan {plan}, {batch} rows differ"
            );
        }
    }
    if m == 2816 {
        for batch in [BATCH, 40] {
            let want = gemm(&dw, batch, 0, false);
            for grouped in [false, true] {
                let mut out = filled::<f32>(&s, batch * m, 0xFF);
                ok(
                    k::dense_half_gemm(
                        &s,
                        quant(if grouped { &dg } else { &dw }, GgmlType::Q8_0),
                        dx.slice(0..batch * kk),
                        out.as_mut_slice(),
                        batch as u32,
                        m as u32,
                        kk as u32,
                        grouped,
                    ),
                    &name,
                );
                assert!(
                    same_bits(&down(&s, &out, batch * m), &want),
                    "{name}: Gemma GEMM differs at {batch} rows"
                );
            }
        }
    }
    println!("{name}: identical to the row order");
}

#[test]
fn grouped_half_gemm() {
    let mut rng = Rng(14);
    for (m, kk) in [
        (5376, 8192),
        (16384, 5376),
        (4100, 5376),
        (2816, 8192),
        (2816, 4096),
    ] {
        check_grouped_half_gemm(m, kk, &mut rng);
    }
}

/// The [gate | up] prefill GEMM with its GeGLU epilogue writes the binary16
/// rows of the FP32 GEMM followed by Gemma's GeGLU pass, bit for bit, in row
/// order and grouped.
fn check_geglu_half_gemm(f: &Format, m: usize, kk: usize, batch: usize, rng: &mut Rng) {
    let s = stream();
    let name = format!("{} GeGLU prefill {m}x{kk}, {batch} rows", f.name);
    let w = random_matrix(f, m, kk, rng);
    let x: Vec<F16> = (0..batch * kk).map(|_| half(rng.normal())).collect();
    let dw = up(&s, &w);
    let half_weights = f.ty == GgmlType::F16;
    let groups = !half_weights && grouped_supported(f, m, kk);
    let dx = up(&s, &x);
    let outputs = batch * (m / 2);
    let mut gu = zeros::<f32>(&s, batch * m);
    if half_weights {
        ok(
            k::unquantized_f16_gemm(
                &s,
                quant(&dw, f.ty),
                dx.as_slice(),
                gu.as_mut_slice(),
                batch,
                m,
                kk,
                1024,
            ),
            &name,
        );
    } else {
        ok(
            k::dense_f16_gemm(
                &s,
                quant(&dw, f.ty),
                dx.as_slice(),
                gu.as_mut_slice(),
                batch,
                m,
                kk,
                0,
                false,
            ),
            &name,
        );
    }
    let mut want = zeros::<F16>(&s, outputs);
    ok(
        k::geglu_packed(
            &s,
            gu.as_slice(),
            None,
            batch as u32,
            (m / 2) as u32,
            Some(want.as_mut_slice()),
        ),
        "geglu_packed",
    );
    let want = down(&s, &want, outputs);
    for grouped in [false, true] {
        if grouped && !groups {
            continue;
        }
        let mut got = filled::<F16>(&s, outputs, 0xFF);
        if half_weights {
            ok(
                k::dense_f16_geglu_gemm_half(
                    &s,
                    quant(&dw, f.ty),
                    dx.as_slice(),
                    got.as_mut_slice(),
                    batch,
                    m,
                    kk,
                ),
                &name,
            );
        } else {
            let dg;
            let wq = if grouped {
                dg = group_weights(&s, f, &dw, m, kk);
                quant(&dg, f.ty)
            } else {
                quant(&dw, f.ty)
            };
            ok(
                k::dense_f16_geglu_gemm(
                    &s,
                    wq,
                    dx.as_slice(),
                    got.as_mut_slice(),
                    batch,
                    m,
                    kk,
                    grouped,
                ),
                &name,
            );
        }
        let got = down(&s, &got, outputs);
        let differ = got.iter().zip(&want).filter(|(a, b)| a.0 != b.0).count();
        assert_eq!(
            differ,
            0,
            "{name}{}: {differ} of {outputs} values differ",
            if grouped { " grouped" } else { "" }
        );
    }
    println!("{name}: identical to the GEMM and GeGLU pass");
}

#[test]
fn geglu_half_gemm() {
    let mut rng = Rng(15);
    check_geglu_half_gemm(&Q8_0, 43008, 5376, 2048, &mut rng);
    check_geglu_half_gemm(&Q8_0, 43008, 5376, 300, &mut rng);
    check_geglu_half_gemm(&Q8_0, 4096, 4096, 129, &mut rng);
    check_geglu_half_gemm(&Q8_0, 4224, 2816, 2048, &mut rng);
    check_geglu_half_gemm(&F16_FORMAT, 43008, 5376, 2048, &mut rng);
    check_geglu_half_gemm(&F16_FORMAT, 4224, 2816, 300, &mut rng);
}

/// The load-time BF16 -> binary16 rewrite: exact inside the binary16 range,
/// round to nearest below it, and every value past it counted.
#[test]
fn bf16_conversion() {
    let s = stream();
    let mut rng = Rng(16);
    let mut bf16: Vec<u16> = (0..100_000).map(|_| rng.below(0x1_0000) as u16).collect();
    // 65280 (largest BF16 below the binary16 limit), 65536, and tiny values.
    bf16.extend([
        0x477F, 0xC77F, 0x4780, 0x3780, 0x3700, 0x0001, 0x0000, 0x8000,
    ]);
    let mut outside = 0u32;
    let want: Vec<u16> = bf16
        .iter()
        .map(|&b| {
            let value = f32::from_bits(u32::from(b) << 16);
            let h = F16::from_f32(value);
            if !h.to_f32().is_finite() {
                outside += 1;
            } else if value.abs() >= 2f32.powi(-17) || value == 0.0 {
                assert_eq!(
                    h.to_f32(),
                    value,
                    "BF16 value {value} is not a binary16 value"
                );
            }
            h.0
        })
        .collect();
    let bytes: Vec<u8> = bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut data = up(&s, &bytes);
    let mut count = zeros::<u32>(&s, 1);
    ok(
        k::convert_bf16_to_half(&s, data.as_mut_slice(), bf16.len(), count.as_mut_slice()),
        "convert",
    );
    let got = down(&s, &data, bytes.len());
    let counted = down(&s, &count, 1)[0];
    for i in 0..bf16.len() {
        let g = u16::from_le_bytes([got[2 * i], got[2 * i + 1]]);
        let nan = F16(g).to_f32().is_nan() && F16(want[i]).to_f32().is_nan();
        assert!(
            nan || g == want[i],
            "BF16 conversion differs at element {i}"
        );
    }
    assert_eq!(
        counted, outside,
        "BF16 conversion counted {counted} values outside binary16, expected {outside}"
    );
}

/// The drafter head repack: Q4_K rows reproduce the Q8_0 values within the
/// error of a 16-level fit per 32 values.
#[test]
fn repack() {
    let s = stream();
    let mut rng = Rng(17);
    const ROWS: usize = 257;
    const COLS: usize = 1024;
    let mut src = random_matrix(&Q8_0, ROWS, COLS, &mut rng);
    // Gaussian-like values, as in trained weights, instead of uniform bytes.
    for b in 0..ROWS * COLS / 32 {
        for i in 0..32 {
            src[b * 34 + 2 + i] = (rng.normal() * 30.0).clamp(-127.0, 127.0) as i8 as u8;
        }
    }
    let q4_bytes = ROWS * COLS / 256 * 144;
    let dsrc = up(&s, &src);
    let mut ddst = zeros::<u8>(&s, q4_bytes);
    ok(
        k::repack_q8_0_as_q4k(
            &s,
            quant(&dsrc, GgmlType::Q8_0),
            ddst.as_mut_slice(),
            ROWS as u32,
            COLS as u32,
        ),
        "repack",
    );
    let dst = down(&s, &ddst, q4_bytes);
    let (mut want, mut got) = (vec![0f32; COLS], vec![0f32; COLS]);
    let (mut error, mut energy) = (0f64, 0f64);
    for r in 0..ROWS {
        dequantize_row(&Q8_0, &src[r * COLS / 32 * 34..], &mut want);
        dequantize_row(&Q4_K, &dst[r * COLS / 256 * 144..], &mut got);
        for c in 0..COLS {
            let d = f64::from(got[c]) - f64::from(want[c]);
            error += d * d;
            energy += f64::from(want[c]) * f64::from(want[c]);
        }
    }
    let relative = (error / energy).sqrt();
    println!("draft head repack: relative RMS error {relative}");
    assert!(relative < 0.08, "Q8_0 -> Q4_K repack error {relative}");
}
