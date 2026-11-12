//! The C++ extension's `gemv_ops_test.cpp`: the IQ2_XXS vector products
//! (dense and routed), the NVFP4 prefill product and the routed IQ2_XXS
//! prefill product against ggml's dequantizers with double accumulation.
//! The vector products on BF16 activations must also equal the staged ones
//! bit for bit, round their BF16 stores exactly and group routed slots by
//! expert.
//!
//! cargo test --profile gpu-test -p gufo-nemotron --features gpu --test gemv_ops

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::too_many_arguments)]

mod kernel_common;

use gufo_hip::{Bf16, DeviceBuffer, F16, Stream};
use gufo_nemotron::iq2xxs::{dequantize_iq2xxs, iq2xxs_grid_bf16};
use gufo_nemotron::nvfp4::dequantize_nvfp4;
use gufo_nemotron_kernels as k;
use gufo_nemotron_kernels::sizes;
use kernel_common::*;

/// Accumulation-order noise allowed per output, relative to sum |w * x|.
const MAX_REL_ERROR: f64 = 2e-6;

/// Random valid IQ2_XXS rows: every grid index, sign index and group scale
/// is valid, so the payload is uniform random bits under an FP16 scale.
fn random_iq2xxs(rows: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
    let mut w = vec![0u8; rows * cols / 256 * 66];
    for block in w.chunks_mut(66) {
        let e = 4 + rng.below(5);
        let d = ((e << 10) | rng.below(1024)) as u16;
        block[..2].copy_from_slice(&d.to_le_bytes());
        for b in &mut block[2..] {
            *b = rng.next() as u8;
        }
    }
    w
}

/// Compares every output with the oracle: output row `s` of `rows` values is
/// `W[expert[s]] x[x_row[s]]`.
fn compare(
    label: &str,
    got: &[f32],
    w: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    expert: &[usize],
    x_row: &[usize],
    relu2: bool,
) {
    let row_bytes = cols / 256 * 66;
    let mut wrow = vec![0f32; cols];
    let mut worst = 0f64;
    for s in 0..expert.len() {
        for r in 0..rows {
            let at = (expert[s] * rows + r) * row_bytes;
            dequantize_iq2xxs(&w[at..at + row_bytes], &mut wrow);
            let (mut sum, mut mag) = (0f64, 0f64);
            for c in 0..cols {
                let p = f64::from(wrow[c]) * f64::from(bf16_round(x[x_row[s] * cols + c]));
                sum += p;
                mag += p.abs();
            }
            let (mut want, mut g) = (sum, f64::from(got[s * rows + r]));
            if relu2 {
                // Compare before the square, where the error is linear.
                want = want.max(0.0);
                g = g.max(0.0).sqrt();
            }
            worst = worst.max((g - want).abs() / mag.max(1e-30));
        }
    }
    println!("{label:<40} max rel error {worst:.2e}");
    assert!(worst < MAX_REL_ERROR, "{label}: {worst:.2e}");
}

fn grid(s: &Stream) -> DeviceBuffer<u32> {
    up(s, &iq2xxs_grid_bf16())
}

/// `x` on the device as BF16, through the products' own conversion.
fn to_bf16(s: &Stream, dx: &DeviceBuffer<f32>, n: usize) -> DeviceBuffer<Bf16> {
    let mut out = zeros::<Bf16>(s, n);
    ok(
        k::activations_to_bf16(s, dx.slice(0..n), out.as_mut_slice(), n),
        "activations_to_bf16",
    );
    out
}

/// Every stored BF16 value is the F32 result `got` rounded to nearest even.
fn check_bf16_store(s: &Stream, stored: &DeviceBuffer<Bf16>, got: &[f32]) {
    let stored = down(s, stored, got.len());
    let same = got
        .iter()
        .zip(&stored)
        .all(|(&g, b)| b.0 == (bf16_round(g).to_bits() >> 16) as u16);
    assert!(same, "BF16 stores round the F32 results");
}

/// `got` equals the staged kernels' result bit for bit.
fn check_same_as_staged(staged: &[f32], got: &[f32]) {
    let same = staged
        .iter()
        .zip(got)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(same, "BF16 kernels equal the staged kernels bit for bit");
}

/// The short routing groups every slot once, by expert, in slot order.
fn check_route(
    s: &Stream,
    order: &gufo_hip::DeviceIndices,
    tiles: &gufo_hip::DeviceIndices,
    num_tiles: &gufo_hip::DeviceIndices,
    ids: &[i32],
    experts: usize,
) {
    let count = down_indices(s, num_tiles, 1)[0];
    let tiles = down_indices(s, tiles, count.max(0) as usize * 4);
    let order = down_indices(s, order, ids.len());
    let mut seen = vec![0; ids.len()];
    let mut expert_tiles = vec![0; experts];
    let mut ok = count > 0;
    let mut next = 0;
    for t in tiles.chunks(4) {
        let (expert, first, n) = (t[0], t[1], t[2]);
        ok = ok && first == next && n > 0 && expert >= 0 && (expert as usize) < experts;
        if !ok {
            break;
        }
        expert_tiles[expert as usize] += 1;
        for j in 0..n {
            let slot = order[(first + j) as usize];
            ok = ok
                && slot >= 0
                && (slot as usize) < ids.len()
                && ids[slot as usize] == expert
                && (j == 0 || slot > order[(first + j - 1) as usize]);
            if ok {
                seen[slot as usize] += 1;
            }
        }
        next += n;
    }
    ok = ok && next as usize == ids.len();
    ok = ok && seen.iter().all(|&c| c == 1);
    ok = ok && expert_tiles.iter().all(|&c| c <= 1);
    assert!(ok, "short routing groups every slot once by expert");
}

/// With `bf16`, the activations go through `activations_to_bf16` and the
/// kernels that read them; a relu2 product also stores BF16 once, which
/// must round the F32 result exactly.
fn dense_case(
    s: &Stream,
    rows: usize,
    cols: usize,
    n: usize,
    relu2: bool,
    rng: &mut Rng,
    grid: &DeviceBuffer<u32>,
    bf16: bool,
) {
    let w = random_iq2xxs(rows, cols, rng);
    let x: Vec<f32> = (0..n * cols).map(|_| rng.uniform(-2.0, 2.0)).collect();
    let dw = up(s, &w);
    let dx = up(s, &x);
    let mut out = zeros::<f32>(s, n * rows);
    ok(
        k::iq2xxs_gemv(
            s,
            dw.as_slice(),
            grid.as_slice(),
            dx.as_slice(),
            out.as_mut_slice(),
            rows,
            cols,
            n,
            relu2,
            false,
        ),
        "dense gemv",
    );
    let mut got = down(s, &out, n * rows);
    if bf16 {
        let staged = got;
        let xb = to_bf16(s, &dx, n * cols);
        ok(
            k::iq2xxs_gemv_bf16(
                s,
                dw.as_slice(),
                grid.as_slice(),
                xb.as_slice(),
                out.as_mut_slice(),
                rows,
                cols,
                n,
                relu2,
                false,
            ),
            "dense gemv on BF16",
        );
        got = down(s, &out, n * rows);
        check_same_as_staged(&staged, &got);
        if relu2 {
            let mut stored = zeros::<Bf16>(s, n * rows);
            ok(
                k::iq2xxs_gemv_bf16_store_bf16(
                    s,
                    dw.as_slice(),
                    grid.as_slice(),
                    xb.as_slice(),
                    stored.as_mut_slice(),
                    rows,
                    cols,
                    n,
                    relu2,
                ),
                "dense gemv storing BF16",
            );
            check_bf16_store(s, &stored, &got);
        }
    }
    let label = format!(
        "dense {rows}x{cols}, {n} tokens{}{}",
        if relu2 { ", relu2" } else { "" },
        if bf16 { ", bf16" } else { "" }
    );
    compare(
        &label,
        &got,
        &w,
        rows,
        cols,
        &x,
        &vec![0; n],
        &(0..n).collect::<Vec<_>>(),
        relu2,
    );
}

/// Distinct experts per token, as top-k selects them; `skew` sends about a
/// quarter of the picks to expert 0 so that it takes several tiles.
fn distinct_experts(n: usize, used: usize, experts: usize, skew: bool, rng: &mut Rng) -> Vec<i32> {
    let mut ids = vec![0i32; n * used];
    for t in 0..n {
        for sl in 0..used {
            let e = loop {
                let e = if skew && rng.below(4) == 0 {
                    0
                } else {
                    rng.below(experts as u32) as i32
                };
                if !ids[t * used..t * used + sl].contains(&e) {
                    break e;
                }
            };
            ids[t * used + sl] = e;
        }
    }
    ids
}

/// With `bf16`, the slots run grouped by expert (`iq2xxs_route_short`) on
/// BF16 activations; few experts make tokens share them.
fn routed_case(
    s: &Stream,
    experts: usize,
    rows: usize,
    cols: usize,
    n: usize,
    used: usize,
    per_slot: bool,
    relu2: bool,
    rng: &mut Rng,
    grid: &DeviceBuffer<u32>,
    bf16: bool,
) {
    let w = random_iq2xxs(experts * rows, cols, rng);
    let x_rows = if per_slot { n * used } else { n };
    let x: Vec<f32> = (0..x_rows * cols).map(|_| rng.uniform(-2.0, 2.0)).collect();
    let ids = distinct_experts(n, used, experts, false, rng);
    let dw = up(s, &w);
    let dx = up(s, &x);
    let dids = indices(s, &ids, experts as u32);
    let mut out = zeros::<f32>(s, n * used * rows);
    let launch = if per_slot {
        k::iq2xxs_gemv_routed_per_slot
    } else {
        k::iq2xxs_gemv_routed
    };
    ok(
        launch(
            s,
            dw.as_slice(),
            grid.as_slice(),
            dx.as_slice(),
            dids.view(),
            out.as_mut_slice(),
            rows,
            cols,
            n,
            used,
            relu2,
            false,
            experts,
        ),
        "routed gemv",
    );
    let mut got = down(s, &out, n * used * rows);
    if bf16 {
        let staged = got;
        let slots = n * used;
        let xb = to_bf16(s, &dx, x_rows * cols);
        let mut order = index_out(s, slots, slots as u32);
        let mut tiles = index_out(s, slots * 4, (slots + experts + 1) as u32);
        let mut num_tiles = index_out(s, 1, slots as u32 + 1);
        ok(
            k::iq2xxs_route_short(
                s,
                dids.slice(0..slots),
                order.slice_mut(0..slots),
                tiles.slice_mut(0..slots * 4),
                num_tiles.slice_mut(0..1),
                slots,
                experts,
            ),
            "iq2xxs_route_short",
        );
        check_route(s, &order, &tiles, &num_tiles, &ids, experts);
        let launch = if per_slot {
            k::iq2xxs_gemv_tiles_per_slot
        } else {
            k::iq2xxs_gemv_tiles
        };
        ok(
            launch(
                s,
                dids.slice(0..slots),
                dw.as_slice(),
                grid.as_slice(),
                xb.as_slice(),
                out.as_mut_slice(),
                order.slice(0..slots),
                tiles.slice(0..slots * 4),
                num_tiles.slice(0..1),
                rows,
                cols,
                slots,
                used,
                relu2,
                false,
                experts,
            ),
            "routed gemv on BF16",
        );
        got = down(s, &out, slots * rows);
        check_same_as_staged(&staged, &got);
        if relu2 {
            assert!(!per_slot, "BF16 stores of per-slot activations are unused");
            let mut stored = zeros::<Bf16>(s, slots * rows);
            ok(
                k::iq2xxs_gemv_tiles_store_bf16(
                    s,
                    dids.slice(0..slots),
                    dw.as_slice(),
                    grid.as_slice(),
                    xb.as_slice(),
                    stored.as_mut_slice(),
                    order.slice(0..slots),
                    tiles.slice(0..slots * 4),
                    num_tiles.slice(0..1),
                    rows,
                    cols,
                    slots,
                    used,
                    relu2,
                    experts,
                ),
                "routed gemv storing BF16",
            );
            check_bf16_store(s, &stored, &got);
        }
    }
    let expert: Vec<usize> = ids.iter().map(|&e| e as usize).collect();
    let x_row: Vec<usize> = (0..ids.len())
        .map(|sl| if per_slot { sl } else { sl / used })
        .collect();
    let label = format!(
        "routed {experts} experts {rows}x{cols}, {n}x{used}{}{}{}",
        if per_slot { ", per slot" } else { "" },
        if relu2 { ", relu2" } else { "" },
        if bf16 { ", bf16" } else { "" }
    );
    compare(&label, &got, &w, rows, cols, &x, &expert, &x_row, relu2);
}

/// The NVFP4 prefill product on FP16 activations: decoded weights are exact
/// in FP16, so only F32 accumulation order remains.
fn nvfp4_gemm_case(s: &Stream, rows: usize, cols: usize, n: usize, rng: &mut Rng) {
    let mut w = vec![0u8; rows * cols / 64 * 36];
    for block in w.chunks_mut(36) {
        for scale in &mut block[..4] {
            // UE4M3 exponents 3..10, any mantissa.
            *scale = (((3 + rng.below(8)) << 3) | rng.below(8)) as u8;
        }
        for b in &mut block[4..] {
            *b = rng.next() as u8;
        }
    }
    let x16: Vec<F16> = (0..n * cols)
        .map(|_| half(rng.uniform(-3.0, 3.0)))
        .collect();
    let x: Vec<f32> = x16.iter().map(|h| h.to_f32()).collect();
    let dw = up(s, &w);
    let dx = up(s, &x16);
    let mut out = zeros::<f32>(s, n * rows);
    ok(
        k::nvfp4_gemm(
            s,
            dw.as_slice(),
            dx.as_slice(),
            out.as_mut_slice(),
            rows,
            cols,
            n,
        ),
        "nvfp4 gemm",
    );
    let got = down(s, &out, n * rows);
    let row_bytes = cols / 64 * 36;
    let mut wrow = vec![0f32; cols];
    let mut worst = 0f64;
    for r in 0..rows {
        dequantize_nvfp4(&w[r * row_bytes..(r + 1) * row_bytes], &mut wrow);
        for t in 0..n {
            let (mut sum, mut mag) = (0f64, 0f64);
            for c in 0..cols {
                let p = f64::from(wrow[c]) * f64::from(x[t * cols + c]);
                sum += p;
                mag += p.abs();
            }
            worst = worst.max((f64::from(got[t * rows + r]) - sum).abs() / mag.max(1e-30));
        }
    }
    let label = format!("nvfp4 gemm {rows}x{cols}, {n} tokens");
    println!("{label:<40} max rel error {worst:.2e}");
    assert!(worst < MAX_REL_ERROR, "{label}: {worst:.2e}");
}

/// The routed IQ2_XXS prefill product: exact weights, activations rounded to
/// BF16, F32 accumulation.
fn iq2xxs_gemm_case(
    s: &Stream,
    experts: usize,
    rows: usize,
    cols: usize,
    n: usize,
    used: usize,
    per_slot: bool,
    relu2: bool,
    rng: &mut Rng,
    grid: &DeviceBuffer<u32>,
) {
    let w = random_iq2xxs(experts * rows, cols, rng);
    let slots = n * used;
    let x_rows = if per_slot { slots } else { n };
    let mut x: Vec<f32> = (0..x_rows * cols).map(|_| rng.uniform(-2.0, 2.0)).collect();
    // Repeated rows land in other tile columns: their outputs must be equal.
    let copy_from = 3 % x_rows;
    let copy_to = [x_rows / 2, x_rows - 1];
    for &to in &copy_to {
        let row = x[copy_from * cols..(copy_from + 1) * cols].to_vec();
        x[to * cols..(to + 1) * cols].copy_from_slice(&row);
    }
    let ids = distinct_experts(n, used, experts, true, rng);
    let dw = up(s, &w);
    let dx = up(s, &x);
    let dids = indices(s, &ids, experts as u32);
    let max_tiles = sizes::iq2xxs_route_max_tiles(slots, experts);
    let mut cursor = zeros::<i32>(s, experts);
    let mut order = index_out(s, slots, slots as u32);
    let mut tiles = index_out(s, max_tiles * 4, (slots + experts + 33) as u32);
    let mut num_tiles = index_out(s, 1, (max_tiles + 1) as u32);
    ok(
        k::iq2xxs_route_slots(
            s,
            dids.view(),
            cursor.as_mut_slice(),
            order.view_mut(),
            tiles.view_mut(),
            num_tiles.view_mut(),
            slots,
            experts,
        ),
        "route slots",
    );
    let mut out = zeros::<f32>(s, slots * rows);
    let launch = if per_slot {
        k::iq2xxs_routed_gemm_per_slot
    } else {
        k::iq2xxs_routed_gemm
    };
    ok(
        launch(
            s,
            dids.view(),
            dw.as_slice(),
            grid.as_slice(),
            dx.as_slice(),
            out.as_mut_slice(),
            order.view(),
            tiles.view(),
            num_tiles.view(),
            rows,
            cols,
            relu2,
            slots,
            experts,
            used,
        ),
        "routed gemm",
    );
    let got = down(s, &out, slots * rows);
    let expert: Vec<usize> = ids.iter().map(|&e| e as usize).collect();
    let x_row: Vec<usize> = (0..slots)
        .map(|sl| if per_slot { sl } else { sl / used })
        .collect();
    let label = format!(
        "iq2 routed gemm {rows}x{cols}, {n}x{used}{}{}",
        if per_slot { ", per slot" } else { "" },
        if relu2 { ", relu2" } else { "" }
    );
    compare(&label, &got, &w, rows, cols, &x, &expert, &x_row, relu2);
    // Same input row and expert: bit-identical outputs.
    for sl in 0..slots {
        for o in 0..slots {
            if x_row[sl] != copy_from || expert[o] != expert[sl] || !copy_to.contains(&x_row[o]) {
                continue;
            }
            assert_eq!(
                got[sl * rows..(sl + 1) * rows],
                got[o * rows..(o + 1) * rows],
                "equal rows give equal outputs in any tile column"
            );
        }
    }
}

#[test]
fn vector_and_prefill_products() {
    let s = stream();
    let mut rng = Rng(20_261_029);
    let grid = grid(&s);
    // Rows of 4k super-blocks take whole 32-value groups per lane, others
    // eight values. 768 columns: one partial chunk; 2304: a full chunk and
    // one block; 3072: a full chunk and a half one.
    for n in [1, 2, 5, 8] {
        dense_case(&s, 64, 768, n, false, &mut rng, &grid, false);
        dense_case(&s, 32, 3072, n, false, &mut rng, &grid, false);
    }
    dense_case(&s, 96, 2304, 1, true, &mut rng, &grid, false);
    dense_case(&s, 32, 2304, 8, true, &mut rng, &grid, false);
    dense_case(&s, 64, 1024, 1, true, &mut rng, &grid, false);
    routed_case(&s, 12, 96, 512, 1, 9, false, true, &mut rng, &grid, false);
    routed_case(&s, 12, 64, 1024, 1, 9, false, true, &mut rng, &grid, false);
    routed_case(&s, 12, 64, 768, 1, 9, true, false, &mut rng, &grid, false);
    routed_case(&s, 7, 32, 2304, 3, 4, true, false, &mut rng, &grid, false);
    routed_case(&s, 7, 32, 2048, 2, 3, true, false, &mut rng, &grid, false);
    routed_case(&s, 7, 32, 256, 8, 2, false, true, &mut rng, &grid, false);

    // BF16 activations: exactly n rows, and routed slots grouped by expert,
    // several tokens sharing experts (up to eight slots per expert).
    for n in [2, 3, 5, 8] {
        dense_case(&s, 64, 768, n, false, &mut rng, &grid, true);
        dense_case(&s, 32, 3072, n, true, &mut rng, &grid, true);
    }
    dense_case(&s, 96, 1280, 7, true, &mut rng, &grid, true);
    dense_case(&s, 64, 4096, 8, true, &mut rng, &grid, true);
    dense_case(&s, 32, 5376, 4, false, &mut rng, &grid, true);
    routed_case(&s, 12, 96, 512, 2, 9, false, true, &mut rng, &grid, true);
    routed_case(&s, 12, 64, 1024, 8, 9, false, true, &mut rng, &grid, true);
    routed_case(&s, 7, 32, 2304, 3, 4, true, false, &mut rng, &grid, true);
    routed_case(&s, 5, 32, 768, 8, 3, true, false, &mut rng, &grid, true);
    routed_case(&s, 7, 32, 256, 6, 2, false, true, &mut rng, &grid, true);

    // Partial token tiles, several row tiles, K spanning many stages.
    nvfp4_gemm_case(&s, 256, 512, 9, &mut rng);
    nvfp4_gemm_case(&s, 128, 1024, 200, &mut rng);
    nvfp4_gemm_case(&s, 384, 192, 1, &mut rng);
    nvfp4_gemm_case(&s, 8192, 128, 300, &mut rng); // 256-token tiles

    // Experts over several tiles, partial tiles, K over several
    // super-blocks, per-token and per-slot activations.
    iq2xxs_gemm_case(&s, 7, 256, 512, 60, 3, false, true, &mut rng, &grid);
    iq2xxs_gemm_case(&s, 3, 256, 256, 5, 2, false, false, &mut rng, &grid);
    iq2xxs_gemm_case(&s, 7, 512, 768, 60, 3, true, false, &mut rng, &grid);
    iq2xxs_gemm_case(&s, 31, 256, 1024, 130, 9, false, true, &mut rng, &grid);
}
