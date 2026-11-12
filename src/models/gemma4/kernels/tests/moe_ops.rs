//! The extension's `moe_ops_test.cpp`: the routed expert kernels of the
//! 26B-A4B at its shapes. Routing against an FP64 host reference, every
//! routed weight format against FP64 dots of the CPU-dequantized rows, and
//! batch invariance (a row routed and projected alone equals the same row
//! inside a verification batch bit for bit, so greedy speculation reproduces
//! decode); the prefill GEMMs over the routing chain the executor builds on
//! the device (expert counts, compact buckets, tile maps), against each
//! other and the prefill copies; the dense blockwise kernels (Q4_0, Q5_0).
//!
//! Differences from the C++ test: the group tables, counts and tile maps come
//! from the device chain the launch contracts require (the C++ builds some on
//! the host; the values are the same), the one-launch routing of the decode
//! path is not declared (see launch.toml), and the timings are not repeated.
//!
//! cargo test --profile gpu-test -p gufo-gemma4-kernels --features gpu --test moe_ops

#![cfg(feature = "gpu")]
#![allow(
    clippy::unwrap_used,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]

mod common;

use common::*;
use gufo_gemma4_kernels as k;
use gufo_gemma4_kernels::sizes;
use gufo_hip::{DeviceBuffer, DeviceIndices, F16, GgmlType, IndexSpec, Stream};

const HIDDEN: usize = 2816;
const EXPERTS: usize = 128;
const USED: usize = 8;
const WIDTH: usize = 704;
const EPS: f32 = 1e-6;
const TILE: usize = sizes::ROUTED_HALF_TILE_ROWS;

fn ids_spec(experts: usize) -> IndexSpec {
    IndexSpec::below(experts as u32).signed()
}

fn groups_spec(experts: usize, slots: usize) -> IndexSpec {
    IndexSpec::below((experts + slots + 17) as u32).signed()
}

/// The group table of `ids` (expert_groups).
fn device_groups(s: &Stream, ids: &DeviceIndices, slots: usize, experts: usize) -> DeviceIndices {
    let mut groups = index_out(
        s,
        sizes::expert_group_table_len(experts),
        groups_spec(experts, slots),
    );
    ok(
        k::expert_groups(
            s,
            ids.slice(0..slots),
            groups.view_mut(),
            slots as u32,
            experts as u32,
        ),
        "expert_groups",
    );
    groups
}

struct Routing {
    ids: Vec<u32>,
    weights: Vec<f32>,
    groups: Vec<i32>,
    logits: Vec<f32>,
}

fn route(
    s: &Stream,
    x: gufo_hip::DeviceSlice<'_, f32>,
    router: &DeviceBuffer<f32>,
    expert_scale: &DeviceBuffer<f32>,
    rows: usize,
) -> Routing {
    let mut logits = zeros::<f32>(s, rows * EXPERTS);
    let mut ids = index_out(s, rows * USED, ids_spec(EXPERTS));
    let mut weights = zeros::<f32>(s, rows * USED);
    ok(
        k::moe_route(
            s,
            x,
            router.as_slice(),
            expert_scale.as_slice(),
            logits.as_mut_slice(),
            ids.view_mut(),
            weights.as_mut_slice(),
            rows as u32,
            HIDDEN as u32,
            EXPERTS as u32,
            USED as u32,
            EPS,
        ),
        "moe_route",
    );
    let groups = device_groups(s, &ids, rows * USED, EXPERTS);
    Routing {
        ids: down_indices(s, &ids, rows * USED),
        weights: down(s, &weights, rows * USED),
        groups: down_indices(s, &groups, sizes::expert_group_table_len(EXPERTS))
            .into_iter()
            .map(|v| v as i32)
            .collect(),
        logits: down(s, &logits, rows * EXPERTS),
    }
}

#[test]
fn routing() {
    let s = stream();
    let mut rng = Rng(26);
    const ROWS: usize = 16;
    let router = rng.normals(EXPERTS * HIDDEN, 0.05);
    let scale: Vec<f32> = rng
        .normals(HIDDEN, 1.0)
        .into_iter()
        .map(|v| 30.0 + v)
        .collect();
    let expert_scale: Vec<f32> = rng
        .normals(EXPERTS, 0.01)
        .into_iter()
        .map(|v| v + 1.0)
        .collect();
    let x = rng.normals(ROWS * HIDDEN, 3.0);
    let mut d_router = up(&s, &router);
    let d_scale = up(&s, &scale);
    ok(
        k::scale_router(
            &s,
            d_router.as_mut_slice(),
            d_scale.as_slice(),
            EXPERTS as u32,
            HIDDEN as u32,
        ),
        "scale_router",
    );
    let d_es = up(&s, &expert_scale);
    let d_x = up(&s, &x);
    let all = route(&s, d_x.as_slice(), &d_router, &d_es, ROWS);

    // FP64 reference routing.
    for r in 0..ROWS {
        let xr = &x[r * HIDDEN..(r + 1) * HIDDEN];
        let ss: f64 = xr.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let rs = 1.0 / (ss / HIDDEN as f64 + f64::from(EPS)).sqrt() / (HIDDEN as f64).sqrt();
        let logits: Vec<f64> = (0..EXPERTS)
            .map(|e| {
                let dot: f64 = (0..HIDDEN)
                    .map(|i| {
                        f64::from(router[e * HIDDEN + i]) * f64::from(scale[i]) * f64::from(xr[i])
                    })
                    .sum();
                dot * rs
            })
            .collect();
        for e in 0..EXPERTS {
            let got = f64::from(all.logits[r * EXPERTS + e]);
            assert!(
                (got - logits[e]).abs() < 1e-4 * (1.0 + logits[e].abs()),
                "router logit error"
            );
        }
        let mut order: Vec<usize> = (0..EXPERTS).collect();
        order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        let total: f64 = (0..USED)
            .map(|j| (logits[order[j]] - logits[order[0]]).exp())
            .sum();
        for j in 0..USED {
            let slot = r * USED + j;
            assert_eq!(
                all.ids[slot] as usize, order[j],
                "top-k expert differs from the reference"
            );
            let want = (logits[order[j]] - logits[order[0]]).exp() / total
                * f64::from(expert_scale[order[j]]);
            assert!(
                (f64::from(all.weights[slot]) - want).abs() < 1e-5,
                "mixture weight error"
            );
        }
    }
    // Group table: every selected expert once, ascending, slots ascending.
    let mut seen = 0usize;
    let mut previous = -1i32;
    for g in 0..all.groups[0] as usize {
        let group = &all.groups[1 + g * sizes::GROUP_INTS..];
        assert!(group[0] > previous && group[1] > 0, "group order");
        previous = group[0];
        for i in 0..group[1] as usize {
            assert!(
                all.ids[group[2 + i] as usize] == group[0] as u32
                    && (i == 0 || group[2 + i] > group[1 + i]),
                "group slots"
            );
            seen += 1;
        }
    }
    assert_eq!(seen, ROWS * USED, "groups do not cover every slot");

    // Batch invariance: each row routed alone gives identical bits.
    for r in (0..ROWS).step_by(5) {
        let one = route(
            &s,
            d_x.slice(r * HIDDEN..(r + 1) * HIDDEN),
            &d_router,
            &d_es,
            1,
        );
        assert!(
            same_bits(&one.logits, &all.logits[r * EXPERTS..(r + 1) * EXPERTS])
                && same_bits(&one.weights, &all.weights[r * USED..(r + 1) * USED])
                && one.ids == all.ids[r * USED..(r + 1) * USED],
            "routing depends on the batch"
        );
    }
    // The prefill form: raw dot products from a GEMM, scaled and selected.
    let mut raw = vec![0f32; ROWS * EXPERTS];
    for r in 0..ROWS {
        for e in 0..EXPERTS {
            let dot: f64 = (0..HIDDEN)
                .map(|i| {
                    f64::from(router[e * HIDDEN + i])
                        * f64::from(scale[i])
                        * f64::from(x[r * HIDDEN + i])
                })
                .sum();
            raw[r * EXPERTS + e] = dot as f32;
        }
    }
    let mut d_raw = up(&s, &raw);
    let mut ids = index_out(&s, ROWS * USED, ids_spec(EXPERTS));
    let mut weights = zeros::<f32>(&s, ROWS * USED);
    ok(
        k::moe_route_raw(
            &s,
            d_x.as_slice(),
            d_es.as_slice(),
            d_raw.as_mut_slice(),
            ids.view_mut(),
            weights.as_mut_slice(),
            ROWS as u32,
            HIDDEN as u32,
            EXPERTS as u32,
            USED as u32,
            EPS,
        ),
        "moe_route_raw",
    );
    assert_eq!(
        down_indices(&s, &ids, ROWS * USED),
        all.ids,
        "raw-logit routing selects other experts"
    );
    let w = down(&s, &weights, ROWS * USED);
    for i in 0..w.len() {
        assert!(
            (w[i] - all.weights[i]).abs() < 1e-5,
            "raw-logit mixture weights"
        );
    }
    // Expert counts derive from the IDs.
    let mut counts = index_out(&s, EXPERTS, IndexSpec::below((ROWS * USED + 1) as u32));
    ok(
        k::expert_counts(
            &s,
            ids.view(),
            counts.view_mut(),
            (ROWS * USED) as u32,
            EXPERTS as u32,
        ),
        "expert_counts",
    );
    let counts = down_indices(&s, &counts, EXPERTS);
    for e in 0..EXPERTS {
        let want = all.ids.iter().filter(|&&id| id as usize == e).count();
        assert_eq!(counts[e] as usize, want, "expert count");
    }
    println!("routing: {} expert groups for {ROWS} rows", all.groups[0]);
}

/// One projection shape: m outputs over k per expert; `row_div` maps a slot
/// to its activation row (USED for the gate/up input a row's slots share, 1
/// for the down projection's per-slot input).
fn check_projection(f: &Format, m: usize, kk: usize, row_div: usize, rng: &mut Rng) {
    let s = stream();
    let name = format!("{} {m}x{kk}", f.name);
    const ROWS: usize = 16;
    // Expert ids with repeats across rows, as in verification batches.
    let mut ids = vec![0u32; ROWS * USED];
    for r in 0..ROWS {
        let mut row: Vec<u32> = Vec::new();
        while row.len() < USED {
            let e = if r % 3 == 0 {
                rng.below(24)
            } else {
                rng.below(24) * 5
            };
            if !row.contains(&e) {
                row.push(e);
            }
        }
        ids[r * USED..(r + 1) * USED].copy_from_slice(&row);
    }
    let experts = *ids.iter().max().unwrap() as usize + 1;
    let w = random_matrix(f, experts * m, kk, rng);
    let x_rows = ROWS * USED / row_div;
    let x = rng.normals(x_rows * kk, 1.0);
    let dw = up(&s, &w);
    let dx = up(&s, &x);
    let out_width = |geglu: bool| if geglu { m / 2 } else { m };

    let run = |first_row: usize, rows: usize, single: bool, geglu: bool| -> Vec<f32> {
        let slots = rows * USED;
        let sub = indices_with(
            &s,
            &ids[first_row * USED..(first_row + rows) * USED],
            ids_spec(experts),
        );
        let groups = device_groups(&s, &sub, slots, experts);
        let width = out_width(geglu);
        let mut dy = filled::<f32>(&s, slots * width, 0xFF);
        let x0 = first_row * USED / row_div * kk;
        let xin = dx.slice(x0..x0 + slots.div_ceil(row_div) * kk);
        let launch = if geglu {
            k::routed_gemv_geglu
        } else {
            k::routed_gemv
        };
        ok(
            launch(
                &s,
                quant(&dw, f.ty),
                sub.view(),
                groups.view(),
                slots as u32,
                xin,
                row_div as u32,
                dy.as_mut_slice(),
                m as u32,
                kk as u32,
                single,
                slots as u32,
                experts as u32,
            ),
            &name,
        );
        down(&s, &dy, slots * width)
    };
    let all = run(0, ROWS, false, false);

    // Against FP64 dots of the dequantized rows (sampled).
    let worst = worst_dot_error(
        f,
        &w,
        kk,
        |slot| x[slot / row_div * kk..(slot / row_div + 1) * kk].to_vec(),
        |slot, o| all[slot * m + o],
        (0..ROWS * USED)
            .step_by(7)
            .map(|slot| (slot, ids[slot] as usize * m)),
        (0..m).step_by((m / 61).max(1)),
    );
    assert!(worst < 2e-6, "{name}: error {worst}");

    // A row alone, and inside narrower batches, matches the 16-row batch.
    for rows in [1usize, 2, 5, 8] {
        let mut first = 0;
        while first + rows <= ROWS {
            let part = run(first, rows, false, false);
            assert!(
                same_bits(&part, &all[first * USED * m..(first + rows) * USED * m]),
                "{name}: {rows}-row batch differs from the 16-row batch"
            );
            first += rows + 3;
        }
    }
    // One-row decode runs a single slot per pass.
    for first in (0..ROWS).step_by(5) {
        let part = run(first, 1, true, false);
        assert!(
            same_bits(&part, &all[first * USED * m..(first + 1) * USED * m]),
            "{name}: the one-slot pass differs from the 16-row batch"
        );
    }

    // The fused gate/up form: GeGLU of the same rows, batch invariant too.
    if row_div == USED {
        let width = m / 2;
        let fused = run(0, ROWS, false, true);
        let mut worst = 0f64;
        for slot in 0..ROWS * USED {
            for c in 0..width {
                let g = f64::from(all[slot * m + c]);
                let u = f64::from(all[slot * m + width + c]);
                let want = gelu_tanh(g) * u;
                // FP32 1 + tanh cancels for very negative gates: scale the
                // error by |gate * up|.
                worst = worst.max(
                    (f64::from(fused[slot * width + c]) - want).abs() / ((g * u).abs() + 1e-3),
                );
            }
        }
        assert!(worst < 1e-6, "{name}: GeGLU error {worst}");
        for rows in [1usize, 5] {
            let part = run(3, rows, false, true);
            assert!(
                same_bits(&part, &fused[3 * USED * width..(3 + rows) * USED * width]),
                "{name}: fused GeGLU depends on the batch"
            );
        }
        let single = run(3, 1, true, true);
        assert!(
            same_bits(&single, &fused[3 * USED * width..4 * USED * width]),
            "{name}: the one-slot fused GeGLU pass differs"
        );
    }
}

#[test]
fn routed_projections() {
    let mut rng = Rng(27);
    for f in [Q4_K, Q5_K, Q6_K, Q8_0, F16_FORMAT, Q4_0] {
        check_projection(&f, 2 * WIDTH, HIDDEN, USED, &mut rng);
    }
    for f in [Q5_1, Q8_0, F16_FORMAT, Q4_0, Q5_0] {
        check_projection(&f, HIDDEN, WIDTH, 1, &mut rng);
    }
}

#[test]
fn feed_forward_residual() {
    let s = stream();
    let mut rng = Rng(28);
    const ROWS: usize = 5;
    const SCALE: f32 = 0.25;
    let dense = rng.normals(ROWS * HIDDEN, 2.0);
    let experts = rng.normals(ROWS * USED * HIDDEN, 1.0);
    let weights = rng.normals(ROWS * USED, 0.3);
    let x = rng.normals(ROWS * HIDDEN, 5.0);
    let n1 = rng.normals(HIDDEN, 1.0);
    let n2 = rng.normals(HIDDEN, 1.0);
    let post = rng.normals(HIDDEN, 1.0);
    let next = rng.normals(HIDDEN, 1.0);
    let (dd, de, dw) = (up(&s, &dense), up(&s, &experts), up(&s, &weights));
    let mut dx = up(&s, &x);
    let (d1, d2, dp, dn) = (up(&s, &n1), up(&s, &n2), up(&s, &post), up(&s, &next));
    let mut dh = zeros::<f32>(&s, x.len());
    ok(
        k::moe_finish(
            &s,
            dd.as_slice(),
            de.as_slice(),
            dw.as_slice(),
            d1.as_slice(),
            d2.as_slice(),
            dp.as_slice(),
            SCALE,
            dx.as_mut_slice(),
            Some(dn.as_slice()),
            Some(dh.as_mut_slice()),
            ROWS as u32,
            HIDDEN as u32,
            USED as u32,
            EPS,
            None,
        ),
        "moe_finish",
    );
    let got_x = down(&s, &dx, x.len());
    let got_h = down(&s, &dh, x.len());
    let rms = |v: &[f64]| {
        1.0 / (v.iter().map(|e| e * e).sum::<f64>() / v.len() as f64 + f64::from(EPS)).sqrt()
    };
    let mut worst = 0f64;
    for r in 0..ROWS {
        let d: Vec<f64> = (0..HIDDEN)
            .map(|i| f64::from(dense[r * HIDDEN + i]))
            .collect();
        let mix: Vec<f64> = (0..HIDDEN)
            .map(|i| {
                (0..USED)
                    .map(|j| {
                        f64::from(weights[r * USED + j])
                            * f64::from(experts[(r * USED + j) * HIDDEN + i])
                    })
                    .sum()
            })
            .collect();
        let (rd, rm) = (rms(&d), rms(&mix));
        let f: Vec<f64> = (0..HIDDEN)
            .map(|i| d[i] * rd * f64::from(n1[i]) + mix[i] * rm * f64::from(n2[i]))
            .collect();
        let rf = rms(&f);
        let xr: Vec<f64> = (0..HIDDEN)
            .map(|i| {
                (f64::from(x[r * HIDDEN + i]) + f[i] * rf * f64::from(post[i])) * f64::from(SCALE)
            })
            .collect();
        let rx = rms(&xr);
        for i in 0..HIDDEN {
            let want_h = xr[i] * rx * f64::from(next[i]);
            worst = worst
                .max((f64::from(got_x[r * HIDDEN + i]) - xr[i]).abs() / (xr[i].abs() + 1e-2))
                .max((f64::from(got_h[r * HIDDEN + i]) - want_h).abs() / (want_h.abs() + 1e-2));
        }
    }
    assert!(worst < 1e-4, "feed-forward residual error {worst}");
}

/// The prefill routing the executor builds on the device: compact buckets
/// and tile maps of 112 rows and `rows_b` rows from the counts of `ids`.
struct Prefill {
    counts: DeviceIndices,
    bounds: DeviceIndices,
    rows_token: DeviceIndices,
    rows_slot: DeviceIndices,
    tiles: DeviceIndices,
    tiles_b: DeviceIndices,
    capacity: usize,
    capacity_b: usize,
    compact: usize,
    slots: usize,
}

fn prefill_routing(
    s: &Stream,
    ids: &[u32],
    tokens: usize,
    used: usize,
    experts: usize,
    rows_b: usize,
) -> Prefill {
    let slots = tokens * used;
    let d_ids = indices_with(s, ids, ids_spec(experts));
    let mut counts = index_out(s, experts, IndexSpec::below((slots + 1) as u32));
    ok(
        k::expert_counts(
            s,
            d_ids.view(),
            counts.view_mut(),
            slots as u32,
            experts as u32,
        ),
        "expert_counts",
    );
    let compact = sizes::routed_compact_rows(slots, experts);
    let mut bounds = index_out(
        s,
        experts + 1,
        IndexSpec::below((compact + 1) as u32).signed(),
    );
    let mut cursors = zeros::<i32>(s, experts);
    let mut rows_token = index_out(
        s,
        compact,
        IndexSpec::below(tokens as u32).signed().padded(),
    );
    let mut rows_slot = index_out(s, compact, IndexSpec::below(slots as u32).signed().padded());
    ok(
        k::routed_compact(
            s,
            d_ids.view(),
            counts.view(),
            bounds.view_mut(),
            cursors.as_mut_slice(),
            rows_token.view_mut(),
            rows_slot.view_mut(),
            tokens as u32,
            used as u32,
            experts as u32,
            slots as u32,
        ),
        "routed_compact",
    );
    let capacity = sizes::routed_tile_capacity(slots, experts, TILE);
    let capacity_b = sizes::routed_tile_capacity(slots, experts, rows_b);
    let tile_spec = IndexSpec::below(experts as u32).masked(0xFFFF);
    let mut tiles = index_out(s, capacity, tile_spec);
    let mut tiles_b = index_out(s, capacity_b, tile_spec);
    ok(
        k::build_routed_tiles(
            s,
            d_ids.view(),
            counts.view(),
            experts as u32,
            TILE as u32,
            capacity as u32,
            tiles.view_mut(),
            rows_b as u32,
            capacity_b as u32,
            tiles_b.view_mut(),
            slots as u32,
        ),
        "build_routed_tiles",
    );
    Prefill {
        counts,
        bounds,
        rows_token,
        rows_slot,
        tiles,
        tiles_b,
        capacity,
        capacity_b,
        compact,
        slots,
    }
}

fn halves(x: &[f32]) -> Vec<F16> {
    x.iter().map(|&v| half(v)).collect()
}

fn clamp_half(v: f32) -> F16 {
    half(v.clamp(-65504.0, 65504.0))
}

/// Which prefill copies a format takes (`PrefillLayoutSupports`).
fn compact_supported(f: &Format, kk: usize) -> bool {
    matches!(f.ty, GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K) && kk.is_multiple_of(256)
}

fn decoded_supported(f: &Format, kk: usize) -> bool {
    compact_supported(f, kk)
        || (matches!(f.ty, GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0)
            && kk.is_multiple_of(128))
}

/// ExpertFormat of a GGML type.
fn expert_format(ty: GgmlType) -> u32 {
    match ty {
        GgmlType::Q4K => 0,
        GgmlType::Q5K => 1,
        GgmlType::Q6K => 2,
        GgmlType::Q8_0 => 3,
        GgmlType::Q5_1 => 4,
        GgmlType::F16 => 5,
        GgmlType::Q4_0 => 6,
        GgmlType::Q5_0 => 7,
        other => panic!("{other:?}"),
    }
}

/// The K-quant routed prefill GEMM over binary16 rows in the routing layout
/// the executor builds, against FP64 dots of the dequantized rows and the
/// binary16 inputs; Q4_K, Q5_K, Q5_1 and Q8_0 also bit for bit against
/// half_gemm's RoutedF16Gemm. At the gate/up shape the fused GeGLU matches
/// the separate pass, and the compact copy of all experts its rows.
fn check_routed_prefill(f: &Format, m: usize, kk: usize, gate_up: bool, rng: &mut Rng) {
    let s = stream();
    // About 150 rows per expert: a full tile and a partly live one.
    const TOKENS: usize = 300;
    const EXPERTS_HERE: usize = 16;
    let mut ids = vec![0u32; TOKENS * USED];
    for t in 0..TOKENS {
        let mut all: Vec<u32> = (0..EXPERTS_HERE as u32).collect();
        rng.shuffle(&mut all);
        ids[t * USED..(t + 1) * USED].copy_from_slice(&all[..USED]);
    }
    let w = random_matrix(f, EXPERTS_HERE * m, kk, rng);
    let x = rng.normals(TOKENS * kk, 1.0);
    let xh = halves(&x);
    let down_kernel = matches!(f.ty, GgmlType::Q5_1 | GgmlType::Q8_0);
    let rows_b = if down_kernel { 64 } else { 48 };
    let p = prefill_routing(&s, &ids, TOKENS, USED, EXPERTS_HERE, rows_b);
    let dw = up(&s, &w);
    let dx = up(&s, &xh);
    let slots = p.slots;
    let mut dy = zeros::<f32>(&s, slots * m);
    ok(
        k::routed_half_gemm(
            &s,
            quant(&dw, f.ty),
            dx.as_slice(),
            p.counts.view(),
            p.tiles.view(),
            p.capacity as u32,
            p.bounds.view(),
            p.rows_token.view(),
            p.rows_slot.view(),
            dy.as_mut_slice(),
            m as u32,
            kk as u32,
            EXPERTS_HERE as u32,
            p.compact as u32,
            TOKENS as u32,
            slots as u32,
            slots as u32,
        ),
        &format!("routed prefill GEMM {}", f.name),
    );
    let got = down(&s, &dy, slots * m);
    let geglu = |w: gufo_hip::QuantSlice<'_>, out: &mut DeviceBuffer<F16>| {
        ok(
            k::routed_half_gemm_geglu(
                &s,
                w,
                dx.as_slice(),
                p.counts.view(),
                p.tiles.view(),
                p.capacity as u32,
                p.bounds.view(),
                p.rows_token.view(),
                p.rows_slot.view(),
                out.as_mut_slice(),
                m as u32,
                kk as u32,
                EXPERTS_HERE as u32,
                p.compact as u32,
                TOKENS as u32,
                slots as u32,
                slots as u32,
            ),
            "fused GeGLU routed prefill",
        );
    };
    // The compact prefill copy of all experts, bit for bit (GeGLU).
    if gate_up && compact_supported(f, kk) {
        let rows = EXPERTS_HERE * m;
        let stage = if f.ty == GgmlType::Q4K {
            sizes::PREFILL_COMPACT_Q4K_BYTES
        } else {
            sizes::PREFILL_COMPACT_Q6K_BYTES
        };
        let mut packed = zeros::<u8>(&s, sizes::prefill_compact_bytes(rows, kk, stage));
        let pack = if f.ty == GgmlType::Q4K {
            k::pack_prefill_weights_compact
        } else {
            k::pack_prefill_weights_compact_112
        };
        ok(
            pack(
                &s,
                quant(&dw, f.ty),
                packed.as_mut_slice(),
                rows as u32,
                kk as u32,
            ),
            "pack compact",
        );
        let mut want = zeros::<F16>(&s, slots * m / 2);
        let mut halfo = zeros::<F16>(&s, slots * m / 2);
        geglu(quant(&dw, f.ty), &mut want);
        let compact = match f.ty {
            GgmlType::Q4K => k::routed_half_gemm_compact_q4k,
            GgmlType::Q5K => k::routed_half_gemm_compact_q5k,
            _ => k::routed_half_gemm_compact_q6k,
        };
        ok(
            compact(
                &s,
                packed.as_slice(),
                dx.as_slice(),
                p.counts.view(),
                p.tiles.view(),
                p.capacity as u32,
                p.bounds.view(),
                p.rows_token.view(),
                p.rows_slot.view(),
                halfo.as_mut_slice(),
                m as u32,
                kk as u32,
                EXPERTS_HERE as u32,
                p.compact as u32,
                TOKENS as u32,
                slots as u32,
                slots as u32,
            ),
            "compact routed GeGLU",
        );
        assert!(
            same_halves(
                &down(&s, &want, slots * m / 2),
                &down(&s, &halfo, slots * m / 2)
            ),
            "{}: compact routed GeGLU differs",
            f.name
        );
        println!("routed prefill {}: compact copy identical", f.name);
    }
    // Gate/up with GeGLU in the epilogue: the bytes the separate binary16
    // pass makes of the binary16 [gate | up] rows.
    if gate_up {
        let width = m / 2;
        let mut gu = zeros::<F16>(&s, slots * m);
        let mut want = zeros::<F16>(&s, slots * width);
        let mut fused = zeros::<F16>(&s, slots * width);
        if matches!(f.ty, GgmlType::F16 | GgmlType::Q4_0) {
            // The FP32 rows above, rounded as the binary16 output would be.
            let rows: Vec<F16> = got.iter().map(|&v| clamp_half(v)).collect();
            s.upload(&rows, gu.as_mut_slice()).unwrap();
        } else if f.ty == GgmlType::Q8_0 {
            ok(
                k::routed_f16_gemm_half(
                    &s,
                    quant(&dw, f.ty),
                    dx.as_slice(),
                    p.counts.view(),
                    p.tiles_b.view(),
                    p.capacity_b as u32,
                    64,
                    p.bounds.view(),
                    p.rows_token.view(),
                    p.rows_slot.view(),
                    gu.as_mut_slice(),
                    m,
                    kk,
                    EXPERTS_HERE as u32,
                    p.compact,
                    TOKENS,
                    slots,
                    slots as u32,
                ),
                "RoutedF16Gemm",
            );
        } else {
            ok(
                k::routed_half_gemm_half(
                    &s,
                    quant(&dw, f.ty),
                    dx.as_slice(),
                    p.counts.view(),
                    p.tiles.view(),
                    p.capacity as u32,
                    p.bounds.view(),
                    p.rows_token.view(),
                    p.rows_slot.view(),
                    gu.as_mut_slice(),
                    m as u32,
                    kk as u32,
                    EXPERTS_HERE as u32,
                    p.compact as u32,
                    TOKENS as u32,
                    slots as u32,
                    slots as u32,
                ),
                "binary16 routed prefill",
            );
        }
        ok(
            k::geglu_packed_half(
                &s,
                gu.as_slice(),
                want.as_mut_slice(),
                slots as u32,
                width as u32,
            ),
            "geglu_packed_half",
        );
        geglu(quant(&dw, f.ty), &mut fused);
        assert!(
            same_halves(
                &down(&s, &want, slots * width),
                &down(&s, &fused, slots * width)
            ),
            "{}: fused GeGLU differs from the separate pass",
            f.name
        );
    }
    if !matches!(
        f.ty,
        GgmlType::Q6K | GgmlType::F16 | GgmlType::Q4_0 | GgmlType::Q5_0
    ) {
        // RoutedF16Gemm's widest tile per format: 48 rows (K-quants), 64
        // (Q5_1, Q8_0).
        let mut flash = zeros::<f32>(&s, slots * m);
        ok(
            k::routed_f16_gemm(
                &s,
                quant(&dw, f.ty),
                dx.as_slice(),
                p.counts.view(),
                p.tiles_b.view(),
                p.capacity_b as u32,
                rows_b as u32,
                p.bounds.view(),
                p.rows_token.view(),
                p.rows_slot.view(),
                flash.as_mut_slice(),
                m,
                kk,
                EXPERTS_HERE as u32,
                p.compact,
                TOKENS,
                slots,
                slots as u32,
            ),
            &format!("RoutedF16Gemm {}", f.name),
        );
        assert!(
            same_bits(&down(&s, &flash, slots * m), &got),
            "{} routed prefill differs from RoutedF16Gemm",
            f.name
        );
    }
    let xh_f32: Vec<f32> = xh.iter().map(|h| h.to_f32()).collect();
    let worst = worst_dot_error(
        f,
        &w,
        kk,
        |slot| xh_f32[slot / USED * kk..(slot / USED + 1) * kk].to_vec(),
        |slot, o| got[slot * m + o],
        (0..slots)
            .step_by(13)
            .map(|slot| (slot, ids[slot] as usize * m)),
        (0..m).step_by(37),
    );
    println!("routed prefill {}: error {worst}", f.name);
    assert!(worst < 2e-3, "routed prefill error {worst}");
}

#[test]
fn routed_prefill() {
    let mut rng = Rng(29);
    for f in [Q4_K, Q5_K, Q6_K, Q8_0, F16_FORMAT, Q4_0] {
        check_routed_prefill(&f, 2 * WIDTH, HIDDEN, true, &mut rng);
    }
    for f in [Q5_1, Q8_0, F16_FORMAT, Q4_0, Q5_0] {
        check_routed_prefill(&f, HIDDEN, WIDTH, false, &mut rng);
    }
}

/// The dense GEMM's results over every prefill copy and the grouped layout,
/// bit for bit: FP32 rows `want` and GeGLU rows `want_half`.
fn check_copies(
    s: &Stream,
    f: &Format,
    dw: &DeviceBuffer<u8>,
    dx: &DeviceBuffer<F16>,
    want: &[f32],
    want_half: &[F16],
    rows: usize,
    m: usize,
    kk: usize,
) {
    let compare = |out: &DeviceBuffer<f32>, out_half: &DeviceBuffer<F16>, what: &str| {
        assert!(
            same_bits(&down(s, out, rows * m), want),
            "{what}: differs from the dense GEMM"
        );
        assert!(
            same_halves(&down(s, out_half, rows * m / 2), want_half),
            "{what}: GeGLU differs from the dense GEMM"
        );
    };
    let name = format!("{} {m}x{kk} at {rows} rows", f.name);
    let format = expert_format(f.ty);
    if compact_supported(f, kk) {
        let stage = if f.ty == GgmlType::Q4K {
            sizes::PREFILL_COMPACT_Q4K_BYTES
        } else {
            sizes::PREFILL_COMPACT_Q6K_BYTES
        };
        let mut packed = zeros::<u8>(s, sizes::prefill_compact_bytes(m, kk, stage));
        let mut out = zeros::<f32>(s, rows * m);
        let mut out_half = zeros::<F16>(s, rows * m / 2);
        if f.ty == GgmlType::Q4K {
            ok(
                k::pack_prefill_weights_compact(
                    s,
                    quant(dw, f.ty),
                    packed.as_mut_slice(),
                    m as u32,
                    kk as u32,
                ),
                "pack",
            );
            ok(
                k::packed_prefill_gemm_compact(
                    s,
                    packed.as_slice(),
                    dx.as_slice(),
                    out.as_mut_slice(),
                    rows as u32,
                    m as u32,
                    kk as u32,
                    0,
                ),
                "packed",
            );
            ok(
                k::packed_prefill_gemm_compact_geglu(
                    s,
                    packed.as_slice(),
                    dx.as_slice(),
                    out_half.as_mut_slice(),
                    rows as u32,
                    m as u32,
                    kk as u32,
                    0,
                ),
                "packed geglu",
            );
        } else {
            ok(
                k::pack_prefill_weights_compact_112(
                    s,
                    quant(dw, f.ty),
                    packed.as_mut_slice(),
                    m as u32,
                    kk as u32,
                ),
                "pack",
            );
            ok(
                k::packed_prefill_gemm_compact_112(
                    s,
                    format,
                    packed.as_slice(),
                    dx.as_slice(),
                    out.as_mut_slice(),
                    rows as u32,
                    m as u32,
                    kk as u32,
                    0,
                ),
                "packed",
            );
            ok(
                k::packed_prefill_gemm_compact_112_geglu(
                    s,
                    format,
                    packed.as_slice(),
                    dx.as_slice(),
                    out_half.as_mut_slice(),
                    rows as u32,
                    m as u32,
                    kk as u32,
                    0,
                ),
                "packed geglu",
            );
        }
        compare(&out, &out_half, &format!("{name} compact"));
    }
    if decoded_supported(f, kk) {
        let mut packed = zeros::<F16>(s, sizes::prefill_decoded_values(m, kk));
        let mut out = zeros::<f32>(s, rows * m);
        let mut out_half = zeros::<F16>(s, rows * m / 2);
        ok(
            k::pack_prefill_weights_decoded(
                s,
                quant(dw, f.ty),
                packed.as_mut_slice(),
                m as u32,
                kk as u32,
            ),
            "pack",
        );
        ok(
            k::packed_prefill_gemm_decoded(
                s,
                format,
                packed.as_slice(),
                dx.as_slice(),
                out.as_mut_slice(),
                rows as u32,
                m as u32,
                kk as u32,
                0,
            ),
            "packed",
        );
        ok(
            k::packed_prefill_gemm_decoded_geglu(
                s,
                format,
                packed.as_slice(),
                dx.as_slice(),
                out_half.as_mut_slice(),
                rows as u32,
                m as u32,
                kk as u32,
                0,
            ),
            "packed geglu",
        );
        compare(&out, &out_half, &format!("{name} decoded"));
    }
    // The grouped layout (MTP targets), as WmmaGemvSupports admits it.
    let grouped = matches!(
        f.ty,
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q4_0
    ) || (f.ty == GgmlType::Q8_0 && m >= 2048 && kk >= 4096);
    if grouped && kk.is_multiple_of(256) {
        let mut dg = zeros::<u8>(s, m * f.row_bytes(kk));
        let group = match f.ty {
            GgmlType::Q4K | GgmlType::Q4_0 => k::group_weights_144,
            GgmlType::Q5K => k::group_weights_176,
            GgmlType::Q6K => k::group_weights_210,
            _ => k::group_weights_272,
        };
        ok(
            group(s, quant(dw, f.ty), dg.as_mut_slice(), m as u32, kk as u32),
            "group",
        );
        let mut out = zeros::<f32>(s, rows * m);
        let mut out_half = zeros::<F16>(s, rows * m / 2);
        ok(
            k::dense_half_gemm(
                s,
                quant(&dg, f.ty),
                dx.as_slice(),
                out.as_mut_slice(),
                rows as u32,
                m as u32,
                kk as u32,
                true,
            ),
            "grouped",
        );
        ok(
            k::dense_half_gemm_geglu(
                s,
                quant(&dg, f.ty),
                dx.as_slice(),
                out_half.as_mut_slice(),
                rows as u32,
                m as u32,
                kk as u32,
                true,
            ),
            "grouped geglu",
        );
        compare(&out, &out_half, &format!("{name} grouped"));
        println!("{name} grouped: identical");
    }
}

/// The dense binary16 prefill GEMM against the routed GEMM over an identity
/// routing, bit for bit, with FP32 and fused-GeGLU outputs, at row counts
/// that end inside a token tile and an m whose last row block is partial.
fn check_dense_prefill(f: &Format, m: usize, kk: usize, rng: &mut Rng) {
    let s = stream();
    let w = random_matrix(f, m, kk, rng);
    let dw = up(&s, &w);
    let row_counts: &[usize] = if f.ty == GgmlType::Q4K {
        &[1, 8, 9, 32, 33, 97, 111, 112, 113, 300, 511, 512, 513, 2048]
    } else {
        &[1, 97, 300, 513]
    };
    for &rows in row_counts {
        let xh = halves(&rng.normals(rows * kk, 1.0));
        let dx = up(&s, &xh);
        // One expert whose 16-padded bucket holds every row in order.
        let p = prefill_routing(&s, &vec![0u32; rows], rows, 1, 1, 64);
        let mut routed = zeros::<f32>(&s, rows * m);
        let mut routed_half = zeros::<F16>(&s, rows * m / 2);
        ok(
            k::routed_half_gemm(
                &s,
                quant(&dw, f.ty),
                dx.as_slice(),
                p.counts.view(),
                p.tiles.view(),
                p.capacity as u32,
                p.bounds.view(),
                p.rows_token.view(),
                p.rows_slot.view(),
                routed.as_mut_slice(),
                m as u32,
                kk as u32,
                1,
                p.compact as u32,
                rows as u32,
                rows as u32,
                rows as u32,
            ),
            "routed prefill",
        );
        ok(
            k::routed_half_gemm_geglu(
                &s,
                quant(&dw, f.ty),
                dx.as_slice(),
                p.counts.view(),
                p.tiles.view(),
                p.capacity as u32,
                p.bounds.view(),
                p.rows_token.view(),
                p.rows_slot.view(),
                routed_half.as_mut_slice(),
                m as u32,
                kk as u32,
                1,
                p.compact as u32,
                rows as u32,
                rows as u32,
                rows as u32,
            ),
            "routed prefill geglu",
        );
        let mut dense = zeros::<f32>(&s, rows * m);
        let mut dense_half = zeros::<F16>(&s, rows * m / 2);
        ok(
            k::dense_half_gemm(
                &s,
                quant(&dw, f.ty),
                dx.as_slice(),
                dense.as_mut_slice(),
                rows as u32,
                m as u32,
                kk as u32,
                false,
            ),
            "dense prefill",
        );
        ok(
            k::dense_half_gemm_geglu(
                &s,
                quant(&dw, f.ty),
                dx.as_slice(),
                dense_half.as_mut_slice(),
                rows as u32,
                m as u32,
                kk as u32,
                false,
            ),
            "dense prefill geglu",
        );
        let a = down(&s, &routed, rows * m);
        let b = down(&s, &dense, rows * m);
        assert!(
            same_bits(&a, &b),
            "{}: dense prefill differs from routed at {rows} rows",
            f.name
        );
        let ah = down(&s, &routed_half, rows * m / 2);
        let bh = down(&s, &dense_half, rows * m / 2);
        assert!(
            same_halves(&ah, &bh),
            "{}: dense GeGLU differs from routed at {rows} rows",
            f.name
        );
        check_copies(&s, f, &dw, &dx, &b, &bh, rows, m, kk);
    }
    println!("dense prefill {}: equals routed", f.name);
}

#[test]
fn dense_prefill() {
    let mut rng = Rng(30);
    for f in [Q4_K, Q5_K, Q6_K, Q8_0] {
        check_dense_prefill(&f, 2000, HIDDEN, &mut rng);
    }
    // Q6_K from 4096 rows takes 144-token tiles.
    check_dense_prefill(&Q6_K, 4100, HIDDEN, &mut rng);
}

/// The dense Q4_0 / Q5_0 prefill GEMM (the QAT targets, Q5_0 down
/// projections) against FP64 dots of the dequantized rows and the binary16
/// inputs; its fused GeGLU bit for bit against the separate pass over its
/// own binary16-rounded FP32 rows.
fn check_dense_blockwise(f: &Format, m: usize, kk: usize, rng: &mut Rng) {
    let s = stream();
    let name = format!("{} {m}x{kk}", f.name);
    let w = random_matrix(f, m, kk, rng);
    let dw = up(&s, &w);
    for rows in [1usize, 97, 300, 513] {
        let xh = halves(&rng.normals(rows * kk, 1.0));
        let dx = up(&s, &xh);
        let width = m / 2;
        let mut dy = zeros::<f32>(&s, rows * m);
        let mut fused = zeros::<F16>(&s, rows * width);
        ok(
            k::dense_half_gemm(
                &s,
                quant(&dw, f.ty),
                dx.as_slice(),
                dy.as_mut_slice(),
                rows as u32,
                m as u32,
                kk as u32,
                false,
            ),
            &name,
        );
        ok(
            k::dense_half_gemm_geglu(
                &s,
                quant(&dw, f.ty),
                dx.as_slice(),
                fused.as_mut_slice(),
                rows as u32,
                m as u32,
                kk as u32,
                false,
            ),
            &name,
        );
        let got = down(&s, &dy, rows * m);
        let xf: Vec<f32> = xh.iter().map(|h| h.to_f32()).collect();
        let worst = worst_dot_error(
            f,
            &w,
            kk,
            |r| xf[r * kk..(r + 1) * kk].to_vec(),
            |r, o| got[r * m + o],
            (0..rows).step_by(7).map(|r| (r, 0)),
            (0..m).step_by(13),
        );
        assert!(worst < 2e-3, "{name}: dense prefill error {worst}");
        let rounded: Vec<F16> = got.iter().map(|&v| clamp_half(v)).collect();
        let gu = up(&s, &rounded);
        let mut want = zeros::<F16>(&s, rows * width);
        ok(
            k::geglu_packed_half(
                &s,
                gu.as_slice(),
                want.as_mut_slice(),
                rows as u32,
                width as u32,
            ),
            "geglu_packed_half",
        );
        let b = down(&s, &fused, rows * width);
        assert!(
            same_halves(&down(&s, &want, rows * width), &b),
            "{name}: dense GeGLU differs from the separate pass at {rows} rows"
        );
        check_copies(&s, f, &dw, &dx, &got, &b, rows, m, kk);
        println!("dense prefill {name}, {rows} rows: error {worst}");
    }
}

/// The dense decode/verification GEMV against FP64 dots of the dequantized
/// rows; every row of a 1..15-row batch equals the same row of the 16-row
/// batch bit for bit.
fn check_dense_gemv(f: &Format, m: usize, kk: usize, rng: &mut Rng) {
    let s = stream();
    let name = format!("{} dense {m}x{kk}", f.name);
    const ROWS: usize = sizes::MAX_GROUP_SLOTS;
    let w = random_matrix(f, m, kk, rng);
    let x = rng.normals(ROWS * kk, 1.0);
    let dw = up(&s, &w);
    let dx = up(&s, &x);
    let run = |first: usize, rows: usize| {
        let mut dy = filled::<f32>(&s, rows * m, 0xFF);
        ok(
            k::dense_gemv(
                &s,
                quant(&dw, f.ty),
                dx.slice(first * kk..(first + rows) * kk),
                dy.as_mut_slice(),
                rows as u32,
                m as u32,
                kk as u32,
            ),
            &name,
        );
        down(&s, &dy, rows * m)
    };
    let all = run(0, ROWS);
    let worst = worst_dot_error(
        f,
        &w,
        kk,
        |r| x[r * kk..(r + 1) * kk].to_vec(),
        |r, o| all[r * m + o],
        (0..ROWS).step_by(3).map(|r| (r, 0)),
        (0..m).step_by(7),
    );
    assert!(worst < 2e-6, "{name}: error {worst}");
    for rows in 1..ROWS {
        let first = (rows * 5) % (ROWS - rows + 1);
        let part = run(first, rows);
        assert!(
            same_bits(&part, &all[first * m..(first + rows) * m]),
            "{name}: {rows}-row batch differs from the 16-row batch"
        );
    }
    println!("{name}: error {worst}");
}

#[test]
fn dense_blockwise() {
    let mut rng = Rng(31);
    for f in [Q4_0, Q5_0] {
        check_dense_blockwise(&f, 2000, HIDDEN, &mut rng);
        // The dense down projection [2816, 2112].
        check_dense_blockwise(&f, HIDDEN, 2112, &mut rng);
        check_dense_gemv(&f, HIDDEN, 2112, &mut rng);
    }
}
