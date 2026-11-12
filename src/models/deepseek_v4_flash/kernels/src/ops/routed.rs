//! The routed experts (`ds4_rocm_moe_launch.hip.hpp`), routed here as
//! `routed_moe_launch` chose: per-pair decode kernels for one token and
//! narrow batches, and for wider ones a route plan (the pairs sorted by
//! expert, in tiles) driving tile kernels, the vendored MMQ tier for
//! prompt-chunk IQ2_XXS gate/up, WMMA kernels for the experts with many
//! pairs and the Q2_K float-down route. The IQ2_XXS/Q2_K mix is the
//! artifact the fixtures cover; the Q4_K and all-Q2_K routes follow the C++
//! launcher without a model to compare against.
//!
//! The plan derives from one generation of the selection, and every
//! kernel that reads it requires that derivation; plan values are bounded
//! by the selection's capacity (`rows * 6` pairs), so the scratch views
//! cover the capacity too, as the workspaces are allocated.

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use gufo_hip::{F16 as Half, IndexSlice, Launch};

use crate::backend::{Backend, BatchMode, Error, Result};
use crate::generated;
use crate::tensor::{Input, Output};

use super::moe::{ExpertSelection, N_EXPERT, N_EXPERT_USED, RoutedExperts, RoutedMoe, Tiles};
use super::{F16, F32, ensure, launched, product, typed, typed_mut};

/// Bytes of a Q8_K block of 256 values.
const Q8K: u64 = 292;
/// Prompt-chunk width from which the MMQ tier owns routed IQ2 gate/up
/// (`DS4_ROCM_ROUTED_MMQ_ROWS`).
const ROUTED_MMQ_ROWS: u32 = 32;
/// Chunk width below which the column-tile cost model picks MMQ's tile
/// (`DS4_ROCM_ROUTED_TILE_MODEL_ROWS`): a forced-width sweep put the
/// vendored rule within 1.1% of the best width from here up, and 10 to 14%
/// short below. A fixed 48-column tile measured +3.4% at 1,024 and +1.1% at
/// 2,048, so it serves up to 2,048 and larger chunks keep the tier's rule.
const ROUTED_TILE_MODEL_ROWS: u32 = 1024;
/// Smallest row group of the wide Q2 down kernel
/// (`DS4_ROCM_WIDE_DOWN_MIN_TILE_M`).
const WIDE_DOWN_MIN_TILE_M: u32 = 32;
/// Pairs from which an IQ2 expert's gate/up runs on the WMMA hot list.
const IQ2_HOT: u32 = 8;

/// The routed expert formats (`routed_moe_build_plan`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum ExpertPath {
    /// Q4_K gate, up and down.
    Q4k,
    /// IQ2_XXS gate and up, Q2_K down: the Flash 0731 artifact.
    Iq2,
    /// Q2_K gate, up and down.
    Q2k,
}

/// Where the F16 mid rows of the hot experts live, if anywhere: behind the
/// F16 down rows in `down` (the MMQ route) or in `gate`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MidHalf {
    None,
    Down,
    Gate,
}

/// The weights of one routed call, resolved for all 256 experts.
struct ExpertWeights<'w> {
    gate: gufo_hip::DeviceSlice<'w, u8>,
    up: gufo_hip::DeviceSlice<'w, u8>,
    down: gufo_hip::DeviceSlice<'w, u8>,
}

/// The column-tile width of MMQ's routed gate/up that minimizes what the
/// pass issues (`ds4_mmq_routed_tile_cols_for_counts`): expert `e` with
/// `c_e` pairs costs `ceil(c_e / w)` tiles of `w` columns plus a weight
/// panel reload each, priced at 16 columns (fitted against a forced-width
/// sweep over 256 to 2,048 tokens; 0.85% mean shortfall).
fn routed_tile_cols(counts: &[u32]) -> u32 {
    const PANEL: u64 = 16;
    let mut best = (0u32, 0u64);
    for w in (16..=80).step_by(16) {
        let cost: u64 = counts
            .iter()
            .filter(|&&c| c != 0)
            .map(|&c| u64::from(c.div_ceil(w)) * (u64::from(w) + PANEL))
            .sum();
        if best.0 == 0 || cost < best.1 {
            best = (w, cost);
        }
    }
    best.0
}

/// The wide Q2 down kernel's (hot expert, row group) list
/// (`ds4_rocm_q2_down_tile_map`): 128-row groups first, then a 64-row tail
/// for each odd bucket when pairing; returns the paired span's length.
fn q2_down_tile_map(
    map: &mut Vec<u32>,
    counts: &[u32; 256],
    hot: &[u32],
    row_tiles: u32,
    pair_tiles: bool,
) -> u32 {
    map.clear();
    let rows = row_tiles * 16;
    for (h, &e) in (0u32..).zip(hot) {
        let groups = counts[e as usize].div_ceil(rows);
        let full = if pair_tiles { groups / 2 } else { groups };
        map.extend((0..full).map(|g| (h << 16) | g));
    }
    let wide = if pair_tiles { map.len() as u32 } else { 0 };
    if pair_tiles {
        for (h, &e) in (0u32..).zip(hot) {
            let groups = counts[e as usize].div_ceil(rows);
            if groups & 1 != 0 {
                map.push((h << 16) | (groups - 1));
            }
        }
    }
    wide
}

/// `routed_moe_launch`'s notices, printed once.
static MMQ_NOTICE: AtomicBool = AtomicBool::new(false);
static TILE_COLS_NOTICE: AtomicI32 = AtomicI32::new(0);

/// The views of a route plan for one call.
struct Plan<'p> {
    selected: IndexSlice<'p>,
    counts: IndexSlice<'p>,
    offsets: IndexSlice<'p>,
    sorted: IndexSlice<'p>,
}

impl ExpertSelection {
    fn plan_views(&self, n_tokens: u32) -> Plan<'_> {
        let pairs = (n_tokens * N_EXPERT_USED) as usize;
        Plan {
            selected: self.selected.slice(0..pairs),
            counts: self.plan.counts.view(),
            offsets: self.plan.offsets.view(),
            sorted: self.plan.sorted.view(),
        }
    }

    fn selected_view(&self, n_tokens: u32) -> IndexSlice<'_> {
        self.selected.slice(0..(n_tokens * N_EXPERT_USED) as usize)
    }
}

impl Tiles {
    fn views(&self) -> (IndexSlice<'_>, IndexSlice<'_>, IndexSlice<'_>) {
        (self.total.view(), self.experts.view(), self.starts.view())
    }
}

/// `ds4_rocm_expert_tile_capacity`: every live expert's first tile, then a
/// whole tile per further `tile_m` pairs, the exact worst case over 256
/// buckets.
fn tile_capacity(pairs: u32, tile_m: u32) -> u32 {
    let first = pairs.min(N_EXPERT);
    first + (pairs - first) / tile_m
}

impl Backend {
    /// `ds4_gpu_routed_moe_batch_tensor`: the six selected experts of each
    /// of `n_tokens` rows of `x`, weighted, into `t.out`, or left as F16
    /// rows in `t.down` when a deferred sum was requested and taken
    /// ([`Backend::set_routed_defer_sum`]).
    pub fn routed_moe_batch(
        &self,
        mut t: RoutedMoe<'_>,
        experts: &RoutedExperts,
        selection: &mut ExpertSelection,
        clamp: f32,
        x: &impl Input,
        n_tokens: u32,
    ) -> Result<()> {
        self.routed_moe(
            "routed moe batch",
            &mut t,
            experts,
            selection,
            clamp,
            x,
            n_tokens,
        )
    }

    /// `ds4_gpu_routed_moe_one_tensor`: the routed experts of one token.
    pub fn routed_moe_one(
        &self,
        mut t: RoutedMoe<'_>,
        experts: &RoutedExperts,
        selection: &mut ExpertSelection,
        clamp: f32,
        x: &impl Input,
    ) -> Result<()> {
        self.routed_moe("routed moe one", &mut t, experts, selection, clamp, x, 1)
    }

    /// `ds4_gpu_set_routed_defer_sum`: asks the next routed calls to leave
    /// the expert sum to the expansion; clears the taken flag. The request
    /// stays set until changed, as in C++.
    pub fn set_routed_defer_sum(&self, enabled: bool) {
        self.routed_defer.set((enabled, false));
    }

    /// `ds4_gpu_routed_sum_deferred`: whether a routed call since the last
    /// request deferred its sum. Only the F16 Q2 down route can defer, so
    /// the request is advisory.
    pub fn routed_sum_deferred(&self) -> bool {
        self.routed_defer.get().1
    }

    fn routed_moe(
        &self,
        op: &'static str,
        t: &mut RoutedMoe<'_>,
        e: &RoutedExperts,
        sel: &mut ExpertSelection,
        clamp: f32,
        x: &impl Input,
        n: u32,
    ) -> Result<()> {
        ensure(
            op,
            e.gate.model == e.up.model
                && e.gate.model == e.down.model
                && e.in_dim != 0
                && e.mid_dim != 0
                && e.out_dim != 0
                && e.in_dim.is_multiple_of(256)
                && e.mid_dim.is_multiple_of(256)
                && n != 0
                && n <= sel.rows,
        )?;
        let path = match (e.gate_type, e.down_type) {
            (12, 12) => ExpertPath::Q4k,
            (16, 10) => ExpertPath::Iq2,
            (10, 10) => ExpertPath::Q2k,
            _ => return Err(Error { op }),
        };
        debug_assert!(
            sel.written >= n,
            "experts read rows the router did not write"
        );
        let (_, gate_expert, _, down_expert) = e.strides().ok_or(Error { op })?;
        let n_expert = u64::from(N_EXPERT);
        let w = ExpertWeights {
            gate: self.weight_bytes(e.gate, product(op, &[n_expert, gate_expert])?)?,
            up: self.weight_bytes(e.up, product(op, &[n_expert, gate_expert])?)?,
            down: self.weight_bytes(e.down, product(op, &[n_expert, down_expert])?)?,
        };
        let x = typed::<f32>(op, x.input())?;
        match path {
            ExpertPath::Q2k => self.routed_q2k(op, t, e, sel, clamp, x, n, &w),
            _ => self.routed_quantized(op, path, t, e, sel, clamp, x, n, &w),
        }
    }

    /// The IQ2_XXS/Q2_K and Q4_K routes (`routed_moe_launch`'s main block).
    fn routed_quantized(
        &self,
        op: &'static str,
        path: ExpertPath,
        t: &mut RoutedMoe<'_>,
        e: &RoutedExperts,
        sel: &mut ExpertSelection,
        clamp: f32,
        x: gufo_hip::DeviceSlice<'_, f32>,
        n: u32,
        w: &ExpertWeights<'_>,
    ) -> Result<()> {
        let stream = self.stream();
        let (q4k, iq2) = (path == ExpertPath::Q4k, path == ExpertPath::Iq2);
        let (in_dim, mid_dim, out_dim) = (e.in_dim, e.mid_dim, e.out_dim);
        let rows_cap = sel.rows;
        let cap_pairs = u64::from(rows_cap) * u64::from(N_EXPERT_USED);
        let xq_blocks = in_dim / 256;
        // The Q8_K inputs live in `down`, the Q8_K mid rows in `gate`,
        // both at the selection's capacity.
        let xq_bytes = product(op, &[u64::from(rows_cap), u64::from(xq_blocks), Q8K])?;
        let midq_bytes = product(op, &[cap_pairs, u64::from(mid_dim / 256), Q8K])?;
        ensure(
            op,
            t.down.output().len() as u64 >= xq_bytes && t.gate.output().len() as u64 >= midq_bytes,
        )?;
        let pairs = n * N_EXPERT_USED;
        let mode = self.batch_mode();
        let verifier = mode == BatchMode::Verifier;
        let support = mode == BatchMode::Support;
        // A DSpark verification block is a handful of rows: the per-pair
        // decode route matches its work, and computes each row as ordinary
        // decode does.
        let small_batch = mode != BatchMode::Off && n > 1 && n <= 16;
        // Target verification combines at most 136 narrow rows; concurrent
        // DSpark support evaluates five rows per request, 10 to 40 at C2-C8.
        let grouped_gate = iq2 && n > 1 && ((verifier && n <= 136) || (support && n <= 40));
        let grouped_gate_only = grouped_gate && n > 6;
        let use_sorted = grouped_gate || (n > 1 && !small_batch && (!q4k || n >= 32));
        let tile_shift = if grouped_gate { 2 } else { 3 };
        // The vendored MMQ tier owns prompt-chunk IQ2 gate/up; never in a
        // DSpark route, whose rows must match ordinary decode.
        let use_mmq = self.projections.mmq_ready.get()
            && iq2
            && n >= ROUTED_MMQ_ROWS
            && mode == BatchMode::Off;
        let use_atomic_down = use_sorted && n >= 128;
        let use_gate_wide = !q4k && use_sorted && n >= 128;
        let use_down_tile16 = !q4k && use_atomic_down;
        let use_decode_lut = (n == 1 || (small_batch && !grouped_gate)) && xq_blocks <= 16;
        let use_direct_sum6 = n == 1 || verifier;

        if !use_mmq {
            self.routed_quantize_x(op, t, x, in_dim, n)?;
        }
        let mut counts = None;
        let mut tile_cap = 0;
        let mut tile16_cap = 0;
        if use_sorted {
            tile_cap = tile_capacity(pairs, 1 << tile_shift);
            tile16_cap = if use_down_tile16 {
                tile_capacity(pairs, 16)
            } else {
                0
            };
            self.routed_plan(
                op,
                sel,
                n,
                tile_shift,
                grouped_gate && pairs <= 48 && !use_down_tile16,
            )?;
            if use_down_tile16 {
                let plan = &mut sel.plan;
                let (total, experts, starts) = (
                    plan.tiles16.total.view_mut(),
                    plan.tiles16.experts.view_mut(),
                    plan.tiles16.starts.view_mut(),
                );
                launched(
                    op,
                    generated::moe_route_tiles(
                        stream,
                        sel.selected.slice(0..pairs as usize),
                        plan.counts.view(),
                        typed_mut::<i32>(op, plan.tile_offsets.output())?,
                        total,
                        experts,
                        starts,
                        4,
                        n,
                        rows_cap,
                    ),
                )?;
            }
        }
        // The counts reach the host only for the routes that read them (MMQ's
        // grid bounds, the IQ2 hot list, the Q2 float-down lists); C++ read
        // them back for every call outside the compact verifier route, and
        // short verification blocks waited for them on every layer unused.
        if use_mmq && use_sorted {
            counts = Some(self.routed_counts(op, sel)?);
        }

        let mut mmq_done = false;
        let mut mid_half = MidHalf::None;
        if use_mmq {
            (mmq_done, mid_half) =
                self.routed_mmq(op, t, e, sel, clamp, x, n, w, counts.as_ref())?;
            if !mmq_done {
                eprintln!("ds4: ROCm gate/up MMQ failed (tokens={n}); falling back");
                self.routed_quantize_x(op, t, x, in_dim, n)?;
            }
        }

        // The WMMA hot list of the IQ2 gate/up: experts with eight or more
        // pairs.
        let use_iq2_gate_wmma = !mmq_done
            && iq2
            && n >= IQ2_HOT
            && !grouped_gate_only
            && use_sorted
            && in_dim.is_multiple_of(16)
            && mid_dim.is_multiple_of(16);
        let (mut hot_count, mut hot_max) = (0u32, 0u32);
        if use_iq2_gate_wmma {
            // Read once for the hot list and the float-down lists, where C++
            // read the same values twice.
            let host = match counts {
                Some(c) => c,
                None => *counts.insert(self.routed_counts(op, sel)?),
            };
            let hot: Vec<u32> = (0..N_EXPERT)
                .filter(|&e| host[e as usize] >= IQ2_HOT)
                .collect();
            hot_count = hot.len() as u32;
            hot_max = hot.iter().map(|&e| host[e as usize]).max().unwrap_or(0);
            if hot_count != 0 {
                stream
                    .upload_indices(&hot, &mut sel.plan.hot)
                    .map_err(|_| Error { op })?;
            }
        }
        let gate_max = if hot_count != 0 { IQ2_HOT } else { 0 };
        let iq2_hot_mid_f16 = mid_half != MidHalf::None
            || (use_iq2_gate_wmma && hot_count != 0 && out_dim.is_multiple_of(2));
        if mid_half == MidHalf::None && iq2_hot_mid_f16 {
            mid_half = MidHalf::Gate;
        }
        let x_half_bytes = product(op, &[u64::from(rows_cap), u64::from(in_dim), F16])?;
        let use_x_f16 =
            use_iq2_gate_wmma && hot_count != 0 && t.up.output().len() as u64 >= x_half_bytes;
        if use_x_f16 {
            let dst = typed_mut::<Half>(op, t.up.output())?;
            let count = n as usize * in_dim as usize;
            launched(
                op,
                generated::convert_f32_to_f16(stream, dst, x.range(0..count), count),
            )?;
        }

        if !mmq_done {
            self.routed_gate_up(
                op,
                path,
                t,
                e,
                sel,
                clamp,
                n,
                w,
                use_sorted,
                tile_shift,
                tile_cap,
                use_gate_wide,
                use_decode_lut,
                gate_max,
            )?;
            if use_iq2_gate_wmma && hot_count != 0 {
                let plan = sel.plan_views(n);
                let up = t.up.output();
                let x_h = use_x_f16
                    .then(|| typed::<Half>(op, up.as_const()))
                    .transpose()?;
                if iq2_hot_mid_f16 {
                    launched(
                        op,
                        generated::moe_iq2_gate_up_hot_f16(
                            stream,
                            typed_mut::<Half>(op, t.gate.output())?,
                            w.gate,
                            w.up,
                            x,
                            x_h,
                            typed::<f32>(op, sel.weights.input())?,
                            plan.selected,
                            plan.counts,
                            plan.offsets,
                            plan.sorted,
                            sel.plan.hot.view(),
                            hot_count,
                            hot_max,
                            in_dim,
                            mid_dim,
                            clamp,
                            n,
                            rows_cap,
                        ),
                    )?;
                } else {
                    launched(
                        op,
                        generated::moe_iq2_gate_up_hot(
                            stream,
                            typed_mut::<f32>(op, t.mid.output())?,
                            w.gate,
                            w.up,
                            x,
                            x_h,
                            typed::<f32>(op, sel.weights.input())?,
                            plan.selected,
                            plan.counts,
                            plan.offsets,
                            plan.sorted,
                            sel.plan.hot.view(),
                            hot_count,
                            hot_max,
                            in_dim,
                            mid_dim,
                            clamp,
                            n,
                            rows_cap,
                        ),
                    )?;
                }
            }
        }

        let use_float_down =
            !verifier && iq2 && n > 1 && !use_direct_sum6 && !grouped_gate_only && use_sorted;
        if use_float_down {
            return self.routed_q2_float_down(
                op,
                t,
                e,
                sel,
                n,
                w,
                if iq2_hot_mid_f16 {
                    mid_half
                } else {
                    MidHalf::None
                },
                counts.as_ref(),
            );
        }
        // Q8_K mid rows in `gate`.
        launched(
            op,
            generated::moe_quantize_q8k(
                stream,
                t.gate.output(),
                typed::<f32>(op, t.mid.output().as_const())?
                    .range(0..pairs as usize * mid_dim as usize),
                mid_dim,
                pairs,
            ),
        )?;
        if use_direct_sum6 {
            // One kernel writes each row's sum.
            let midq = t.gate.output();
            let out = typed_mut::<f32>(op, t.out.output())?;
            let selected = sel.selected_view(n);
            return launched(
                op,
                if q4k {
                    generated::moe_q4k_down_sum6(
                        stream,
                        out,
                        w.down,
                        midq.as_const(),
                        selected,
                        mid_dim,
                        out_dim,
                        n,
                    )
                } else {
                    generated::moe_q2_down_sum6(
                        stream,
                        out,
                        w.down,
                        midq.as_const(),
                        selected,
                        mid_dim,
                        out_dim,
                        n,
                    )
                },
            );
        }
        if use_sorted {
            let (tiles, shift, cap) = if use_down_tile16 {
                (&sel.plan.tiles16, 4, tile16_cap)
            } else {
                (&sel.plan.tiles, tile_shift, tile_cap)
            };
            let (total, experts, starts) = tiles.views();
            let plan = sel.plan_views(n);
            let midq = t.gate.output();
            if use_atomic_down {
                let count = n as usize * out_dim as usize * F32 as usize;
                launched(op, generated::zero_bytes(stream, t.out.output(), count))?;
                let out = typed_mut::<f32>(op, t.out.output())?;
                launched(
                    op,
                    if q4k {
                        generated::moe_q4k_down_tiles_atomic(
                            stream,
                            out,
                            w.down,
                            midq.as_const(),
                            plan.selected,
                            plan.sorted,
                            plan.offsets,
                            plan.counts,
                            total,
                            experts,
                            starts,
                            mid_dim,
                            out_dim,
                            shift,
                            cap,
                            n,
                            rows_cap,
                        )
                    } else {
                        generated::moe_q2_down_tiles_atomic(
                            stream,
                            out,
                            w.down,
                            midq.as_const(),
                            plan.selected,
                            plan.sorted,
                            plan.offsets,
                            plan.counts,
                            total,
                            experts,
                            starts,
                            mid_dim,
                            out_dim,
                            shift,
                            cap,
                            use_down_tile16,
                            n,
                            rows_cap,
                        )
                    },
                )?;
                return Ok(());
            }
            let down = typed_mut::<f32>(op, t.down.output())?;
            launched(
                op,
                if q4k {
                    generated::moe_q4k_down_tiles(
                        stream,
                        down,
                        w.down,
                        midq.as_const(),
                        plan.selected,
                        plan.sorted,
                        plan.offsets,
                        plan.counts,
                        total,
                        experts,
                        starts,
                        mid_dim,
                        out_dim,
                        shift,
                        cap,
                        n,
                        rows_cap,
                    )
                } else {
                    generated::moe_q2_down_tiles(
                        stream,
                        down,
                        w.down,
                        midq.as_const(),
                        plan.selected,
                        plan.sorted,
                        plan.offsets,
                        plan.counts,
                        total,
                        experts,
                        starts,
                        mid_dim,
                        out_dim,
                        shift,
                        cap,
                        false,
                        n,
                        rows_cap,
                    )
                },
            )?;
        } else {
            // Per-pair down rows: the Q4_K route below 32 rows (the IQ2
            // route sorts every multi-row call).
            ensure(op, q4k)?;
            let midq = t.gate.output();
            launched(
                op,
                generated::moe_q4k_down_decode(
                    stream,
                    typed_mut::<f32>(op, t.down.output())?,
                    w.down,
                    midq.as_const(),
                    sel.selected_view(n),
                    mid_dim,
                    out_dim,
                    n,
                ),
            )?;
        }
        self.routed_sum(op, t, out_dim, n)
    }

    /// Q8_K rows of `x` into `down`.
    fn routed_quantize_x(
        &self,
        op: &'static str,
        t: &mut RoutedMoe<'_>,
        x: gufo_hip::DeviceSlice<'_, f32>,
        in_dim: u32,
        n: u32,
    ) -> Result<()> {
        launched(
            op,
            generated::moe_quantize_q8k(
                self.stream(),
                t.down.output(),
                x.range(0..n as usize * in_dim as usize),
                in_dim,
                n,
            ),
        )
    }

    /// The route plan: sorted pairs and tiles of `1 << tile_shift`, by the
    /// one-block builder for at most 48 pairs when `narrow`.
    fn routed_plan(
        &self,
        op: &'static str,
        sel: &mut ExpertSelection,
        n: u32,
        tile_shift: u32,
        narrow: bool,
    ) -> Result<()> {
        let stream = self.stream();
        let pairs = (n * N_EXPERT_USED) as usize;
        let rows = sel.rows;
        let plan = &mut sel.plan;
        let selected = sel.selected.slice(0..pairs);
        let tile_offsets = typed_mut::<i32>(op, plan.tile_offsets.output())?;
        let tiles = &mut plan.tiles;
        let (total, experts, starts) = (
            tiles.total.view_mut(),
            tiles.experts.view_mut(),
            tiles.starts.view_mut(),
        );
        if narrow {
            return launched(
                op,
                generated::moe_route_plan_narrow(
                    stream,
                    selected,
                    plan.counts.view_mut(),
                    plan.offsets.view_mut(),
                    plan.sorted.view_mut(),
                    tile_offsets,
                    total,
                    experts,
                    starts,
                    n,
                    tile_shift,
                    rows,
                ),
            );
        }
        launched(
            op,
            generated::moe_route_plan(
                stream,
                selected,
                plan.counts.view_mut(),
                plan.offsets.view_mut(),
                typed_mut::<i32>(op, plan.cursors.output())?,
                plan.sorted.view_mut(),
                tile_offsets,
                total,
                experts,
                starts,
                n,
                tile_shift,
                rows,
            ),
        )
    }

    /// The per-expert pair counts of the plan, read back.
    fn routed_counts(&self, op: &'static str, sel: &ExpertSelection) -> Result<[u32; 256]> {
        let mut host = [0u32; 256];
        self.stream()
            .download_blocking(sel.plan.counts.view().values(), &mut host)
            .map_err(|_| Error { op })?;
        Ok(host)
    }

    /// The MMQ gate/up of the IQ2 route: fused SwiGLU into F32 and F16
    /// mid rows when the F16 rows fit behind the F16 down rows in `down`,
    /// else gate and up rows and the SwiGLU kernel. Whether it ran, and
    /// where its F16 mid rows are.
    fn routed_mmq(
        &self,
        op: &'static str,
        t: &mut RoutedMoe<'_>,
        e: &RoutedExperts,
        sel: &ExpertSelection,
        clamp: f32,
        x: gufo_hip::DeviceSlice<'_, f32>,
        n: u32,
        w: &ExpertWeights<'_>,
        counts: Option<&[u32; 256]>,
    ) -> Result<(bool, MidHalf)> {
        let stream = self.stream();
        let (in_dim, mid_dim, out_dim) = (e.in_dim, e.mid_dim, e.out_dim);
        let cap_pairs = u64::from(sel.rows) * u64::from(N_EXPERT_USED);
        // The wide Q2 down kernel reads its activations as F16: the mirror
        // goes behind the F16 down rows, both at the selection's capacity.
        let down_h_bytes = product(op, &[cap_pairs, u64::from(out_dim), F16])?;
        let mid_h_bytes = product(op, &[cap_pairs, u64::from(mid_dim), F16])?;
        let has_mid_h = out_dim.is_multiple_of(2)
            && down_h_bytes
                .checked_add(mid_h_bytes)
                .is_some_and(|end| end <= t.down.output().len() as u64);
        let max_rows = counts.map_or(0, |c| c.iter().copied().max().unwrap_or(0));
        let tile_cols = match counts {
            Some(c) if n < ROUTED_TILE_MODEL_ROWS => routed_tile_cols(c),
            Some(_) if n <= 2048 => 48,
            _ => 0,
        };
        if tile_cols != 0
            && TILE_COLS_NOTICE.swap(tile_cols as i32, Ordering::Relaxed) != tile_cols as i32
        {
            let c = counts.map_or([0; 256], |c| *c);
            let live = c.iter().filter(|&&v| v != 0).count();
            eprintln!(
                "ds4: ROCm routed gate/up column tile {tile_cols} (tokens={n}, {live} live experts, max bucket {max_rows})"
            );
        }
        let selected = sel.selected_view(n);
        let weights = typed::<f32>(op, sel.weights.input())?;
        let mut fused = false;
        let mut done = false;
        if self.projections.gfx1151.get() && has_mid_h {
            let up_len = t.up.output().len();
            let down = t.down.output();
            let (down_h, mid_h) = down.split_at(down_h_bytes as usize);
            let discard = typed_mut::<f32>(op, down_h.range(0..4))?;
            let mid_h = typed_mut::<Half>(op, mid_h.range(0..mid_h_bytes as usize))?;
            let ran = generated::moe_iq2_mmq_gate_up_swiglu(
                stream,
                typed_mut::<f32>(op, t.gate.output())?,
                discard,
                typed_mut::<f32>(op, t.mid.output())?,
                mid_h,
                t.up.output(),
                up_len,
                w.gate,
                w.up,
                x,
                selected,
                weights,
                in_dim,
                mid_dim,
                n,
                max_rows,
                tile_cols,
                clamp,
            );
            fused = launched(op, ran)? == Launch::Launched;
            done = fused;
        }
        if !done {
            let down_len = t.down.output().len();
            let ran = generated::moe_iq2_mmq_gate_up(
                stream,
                typed_mut::<f32>(op, t.gate.output())?,
                typed_mut::<f32>(op, t.up.output())?,
                t.down.output(),
                down_len,
                w.gate,
                w.up,
                x,
                selected,
                in_dim,
                mid_dim,
                n,
                max_rows,
                tile_cols,
            );
            done = launched(op, ran)? == Launch::Launched;
            if done {
                let pairs_mid = n as usize * N_EXPERT_USED as usize * mid_dim as usize;
                let mid_h = if has_mid_h {
                    let down = t.down.output();
                    let (_, mid_h) = down.split_at(down_h_bytes as usize);
                    Some(typed_mut::<Half>(op, mid_h.range(0..mid_h_bytes as usize))?)
                } else {
                    None
                };
                let gate = t.gate.output();
                let up = t.up.output();
                launched(
                    op,
                    generated::moe_mmq_swiglu(
                        stream,
                        typed_mut::<f32>(op, t.mid.output())?,
                        mid_h,
                        typed::<f32>(op, gate.as_const())?.range(0..pairs_mid),
                        typed::<f32>(op, up.as_const())?.range(0..pairs_mid),
                        typed::<f32>(op, sel.weights.input())?,
                        mid_dim,
                        n,
                        clamp,
                    ),
                )?;
            }
        }
        if !done {
            return Ok((false, MidHalf::None));
        }
        if !MMQ_NOTICE.swap(true, Ordering::Relaxed) {
            eprintln!(
                "ds4: ROCm routed MoE using HIP MMQ gate/up{} with native Q2 down",
                if fused { " fused SwiGLU" } else { "" }
            );
        }
        Ok((
            true,
            if has_mid_h {
                MidHalf::Down
            } else {
                MidHalf::None
            },
        ))
    }

    /// The gate/up kernels of the quantized routes: expert tiles over the
    /// plan, else per-pair decode.
    fn routed_gate_up(
        &self,
        op: &'static str,
        path: ExpertPath,
        t: &mut RoutedMoe<'_>,
        e: &RoutedExperts,
        sel: &ExpertSelection,
        clamp: f32,
        n: u32,
        w: &ExpertWeights<'_>,
        use_sorted: bool,
        tile_shift: u32,
        tile_cap: u32,
        wide: bool,
        lut: bool,
        max_count: u32,
    ) -> Result<()> {
        let stream = self.stream();
        let (in_dim, mid_dim) = (e.in_dim, e.mid_dim);
        let rows_cap = sel.rows;
        let weights = typed::<f32>(op, sel.weights.input())?;
        let xq = t.down.output();
        let mid = typed_mut::<f32>(op, t.mid.output())?;
        if use_sorted {
            let plan = sel.plan_views(n);
            let (total, experts, starts) = sel.plan.tiles.views();
            return launched(
                op,
                match path {
                    ExpertPath::Q4k => generated::moe_q4k_gate_up_tiles(
                        stream,
                        mid,
                        w.gate,
                        w.up,
                        xq.as_const(),
                        plan.selected,
                        plan.sorted,
                        plan.offsets,
                        plan.counts,
                        total,
                        experts,
                        starts,
                        weights,
                        in_dim,
                        mid_dim,
                        tile_shift,
                        tile_cap,
                        clamp,
                        n,
                        rows_cap,
                    ),
                    _ if wide => generated::moe_iq2_gate_up_tiles_wide(
                        stream,
                        mid,
                        w.gate,
                        w.up,
                        xq.as_const(),
                        plan.selected,
                        plan.sorted,
                        plan.offsets,
                        plan.counts,
                        total,
                        experts,
                        starts,
                        weights,
                        in_dim,
                        mid_dim,
                        tile_cap,
                        max_count,
                        clamp,
                        tile_shift,
                        n,
                        rows_cap,
                    ),
                    _ => generated::moe_iq2_gate_up_tiles(
                        stream,
                        mid,
                        w.gate,
                        w.up,
                        xq.as_const(),
                        plan.selected,
                        plan.sorted,
                        plan.offsets,
                        plan.counts,
                        total,
                        experts,
                        starts,
                        weights,
                        in_dim,
                        mid_dim,
                        tile_shift,
                        tile_cap,
                        max_count,
                        clamp,
                        n,
                        rows_cap,
                    ),
                },
            );
        }
        let pairs = n as usize * N_EXPERT_USED as usize;
        let selected = sel.selected_view(n);
        let weights = weights.range(0..pairs);
        if path == ExpertPath::Q4k {
            return launched(
                op,
                generated::moe_q4k_gate_up_decode(
                    stream,
                    mid,
                    w.gate,
                    w.up,
                    xq.as_const(),
                    selected,
                    weights,
                    in_dim,
                    mid_dim,
                    n,
                    clamp,
                ),
            );
        }
        if lut {
            return launched(
                op,
                generated::moe_iq2_gate_up_lut(
                    stream,
                    mid,
                    w.gate,
                    w.up,
                    xq.as_const(),
                    selected,
                    weights,
                    in_dim,
                    mid_dim,
                    n,
                    clamp,
                ),
            );
        }
        launched(
            op,
            generated::moe_iq2_gate_up_qwarp(
                stream,
                typed_mut::<f32>(op, t.gate.output())?,
                typed_mut::<f32>(op, t.up.output())?,
                mid,
                w.gate,
                w.up,
                xq.as_const(),
                selected,
                weights,
                in_dim,
                mid_dim,
                n,
                clamp,
            ),
        )
    }

    /// The six pair-major F32 expert rows of each row summed into `out`.
    fn routed_sum(
        &self,
        op: &'static str,
        t: &mut RoutedMoe<'_>,
        out_dim: u32,
        n: u32,
    ) -> Result<()> {
        let count = n as usize * N_EXPERT_USED as usize * out_dim as usize;
        let down = t.down.output();
        launched(
            op,
            generated::moe_sum(
                self.stream(),
                typed_mut::<f32>(op, t.out.output())?,
                typed::<f32>(op, down.as_const())?.range(0..count),
                out_dim,
                n,
            ),
        )
    }

    /// The six pair-major F16 expert rows of each row summed into `out`.
    fn routed_sum_f16(
        &self,
        op: &'static str,
        t: &mut RoutedMoe<'_>,
        down_h: gufo_hip::DeviceSlice<'_, Half>,
        out_dim: u32,
        n: u32,
    ) -> Result<()> {
        let count = n as usize * N_EXPERT_USED as usize * out_dim as usize;
        launched(
            op,
            generated::moe_sum_f16(
                self.stream(),
                typed_mut::<f32>(op, t.out.output())?,
                down_h.range(0..count),
                out_dim,
                n,
            ),
        )
    }

    /// `routed_moe_q2_float_down_launch`: the Q2_K down of the IQ2 route
    /// from F32 mid rows (F16 for the hot experts when `mid_half` says
    /// where), into F16 down rows at even widths. Single-pair experts run
    /// on a 32-row kernel, the other cold experts on staged pair tiles and
    /// the hot ones (four pairs or more from 64 to 2,048 rows, else eight)
    /// on WMMA tiles; the active experts go to the device once, hot ones
    /// first.
    fn routed_q2_float_down(
        &self,
        op: &'static str,
        t: &mut RoutedMoe<'_>,
        e: &RoutedExperts,
        sel: &mut ExpertSelection,
        n: u32,
        w: &ExpertWeights<'_>,
        mid_half: MidHalf,
        counts: Option<&[u32; 256]>,
    ) -> Result<()> {
        let stream = self.stream();
        let (mid_dim, out_dim) = (e.mid_dim, e.out_dim);
        let rows_cap = sel.rows;
        let counts = match counts {
            Some(c) => *c,
            None => self.routed_counts(op, sel)?,
        };
        // Pairs a wave carries per dequantized weight row: at 32 tokens the
        // mean live bucket is below two, and eight stages more than it saves.
        let pair_shift = if n <= 128 { 1 } else { 2 };
        let use_f16_down = out_dim.is_multiple_of(2);
        // The scalar/WMMA crossover moves with the prompt width; short
        // seeds keep the established eight-row policy.
        let hot_threshold = if (64..=2048).contains(&n) { 4 } else { 8 };
        let use_wmma_hot = mid_dim.is_multiple_of(16) && out_dim.is_multiple_of(16);
        let mut active: Vec<u32> = Vec::with_capacity(256);
        if use_wmma_hot {
            active.extend((0..N_EXPERT).filter(|&e| counts[e as usize] >= hot_threshold));
        }
        let hot_count = active.len() as u32;
        let hot_max = active
            .iter()
            .map(|&e| counts[e as usize])
            .max()
            .unwrap_or(0);
        let scalar_max = if hot_count != 0 { hot_threshold } else { 0 };
        let cold = |c: u32| scalar_max == 0 || c < scalar_max;
        active.extend((0..N_EXPERT).filter(|&e| counts[e as usize] == 1 && cold(1)));
        let singleton_count = active.len() as u32 - hot_count;
        active.extend((0..N_EXPERT).filter(|&e| {
            let c = counts[e as usize];
            c > 1 && cold(c)
        }));
        let cold_multi_count = active.len() as u32 - hot_count - singleton_count;
        if !active.is_empty() {
            stream
                .upload_indices(&active, &mut sel.plan.hot)
                .map_err(|_| Error { op })?;
        }
        let range = |from: u32, len: u32| from as usize..(from + len) as usize;
        let batch = |t: &mut RoutedMoe<'_>,
                     list: std::ops::Range<usize>,
                     shift: u32,
                     rows_per_block: u32,
                     min_count: u32,
                     max_count: u32|
         -> Result<()> {
            let plan = sel.plan_views(n);
            let experts = sel.plan.hot.slice(list.clone());
            let n_experts = list.len() as u32;
            let mid = t.mid.output();
            let mid = typed::<f32>(op, mid.as_const())?;
            if use_f16_down {
                launched(
                    op,
                    generated::moe_q2_down_batch_f16(
                        stream,
                        typed_mut::<Half>(op, t.down.output())?,
                        w.down,
                        mid,
                        plan.selected,
                        plan.counts,
                        plan.offsets,
                        plan.sorted,
                        experts,
                        n_experts,
                        shift,
                        rows_per_block,
                        min_count,
                        max_count,
                        mid_dim,
                        out_dim,
                        n,
                        rows_cap,
                    ),
                )
            } else {
                launched(
                    op,
                    generated::moe_q2_down_batch(
                        stream,
                        typed_mut::<f32>(op, t.down.output())?,
                        w.down,
                        mid,
                        plan.selected,
                        plan.counts,
                        plan.offsets,
                        plan.sorted,
                        experts,
                        n_experts,
                        shift,
                        rows_per_block,
                        min_count,
                        max_count,
                        mid_dim,
                        out_dim,
                        n,
                        rows_cap,
                    ),
                )
            }
        };
        if singleton_count != 0 {
            batch(t, range(hot_count, singleton_count), 0, 32, 1, 2)?;
        }
        if cold_multi_count != 0 {
            batch(
                t,
                range(hot_count + singleton_count, cold_multi_count),
                pair_shift,
                16,
                2,
                scalar_max,
            )?;
        }
        if use_wmma_hot && hot_count != 0 {
            self.routed_q2_down_hot(
                op, t, e, sel, n, w, mid_half, &counts, &active, hot_count, hot_max,
            )?;
        }
        let (request, _) = self.routed_defer.get();
        if use_f16_down && request {
            // The per-expert F16 rows stay in `down` for the fused expansion.
            self.routed_defer.set((true, true));
            return Ok(());
        }
        if use_f16_down {
            let down = t.down.output();
            let down_h = typed::<Half>(op, down.as_const())?;
            let count = n as usize * N_EXPERT_USED as usize * out_dim as usize;
            let down_h = down_h.range(0..count);
            return launched(
                op,
                generated::moe_sum_f16(
                    stream,
                    typed_mut::<f32>(op, t.out.output())?,
                    down_h,
                    out_dim,
                    n,
                ),
            );
        }
        self.routed_sum(op, t, out_dim, n)
    }

    /// The WMMA down of the hot experts of the Q2 float-down route: four
    /// column fragments wide over a compacted (hot expert, row group) list
    /// when the F16 mid rows exist, else two.
    fn routed_q2_down_hot(
        &self,
        op: &'static str,
        t: &mut RoutedMoe<'_>,
        e: &RoutedExperts,
        sel: &mut ExpertSelection,
        n: u32,
        w: &ExpertWeights<'_>,
        mid_half: MidHalf,
        counts: &[u32; 256],
        active: &[u32],
        hot_count: u32,
        hot_max: u32,
    ) -> Result<()> {
        const NFRAG: u32 = 4;
        let stream = self.stream();
        let (mid_dim, out_dim) = (e.mid_dim, e.out_dim);
        let rows_cap = sel.rows;
        let cap_pairs = u64::from(rows_cap) * u64::from(N_EXPERT_USED);
        let use_f16_down = out_dim.is_multiple_of(2);
        let wide_mtiles = if n <= 128 { 2 } else { 4 };
        let down_h_bytes = product(op, &[cap_pairs, u64::from(out_dim), F16])? as usize;
        let mid_h_bytes = product(op, &[cap_pairs, u64::from(mid_dim), F16])? as usize;
        if use_f16_down && mid_half != MidHalf::None && out_dim.is_multiple_of(NFRAG * 16) {
            // A rectangular grid is sized by the largest bucket (near 3,500
            // rows at a 4,096-token chunk against a mean of 96), and every
            // idle block still reserves the whole LDS tile: compact the
            // groups that have rows.
            let cap = n * N_EXPERT_USED / WIDE_DOWN_MIN_TILE_M + N_EXPERT + 1;
            let mut map = Vec::with_capacity(cap as usize);
            let paired = q2_down_tile_map(
                &mut map,
                counts,
                &active[..hot_count as usize],
                wide_mtiles,
                n >= 1024,
            );
            let mapped = !map.is_empty() && map.len() <= cap as usize;
            if mapped {
                let dst = typed_mut::<u32>(op, sel.plan.tile_map.output())?.range(0..map.len());
                stream
                    .upload_blocking(&map, dst)
                    .map_err(|_| Error { op })?;
            }
            let (paired, wide_tiles) = if mapped {
                (paired, map.len() as u32 - paired)
            } else {
                (0, 0)
            };
            let launch = |t: &mut RoutedMoe<'_>,
                          mtiles: u32,
                          map_range: Option<std::ops::Range<usize>>|
             -> Result<()> {
                let plan = sel.plan_views(n);
                let tile_map = match &map_range {
                    Some(r) => Some(typed::<u32>(op, sel.plan.tile_map.input())?.range(r.clone())),
                    None => None,
                };
                let n_map = map_range.as_ref().map_or(0, |r| r.len() as u32);
                // The F16 mid rows: behind the F16 down rows, or in `gate`.
                let source;
                let down_h = match mid_half {
                    MidHalf::Down => {
                        let (down_h, mid_h) = t.down.output().split_at(down_h_bytes);
                        source = mid_h.range(0..mid_h_bytes);
                        typed_mut::<Half>(op, down_h)?
                    }
                    _ => {
                        source = t.gate.output();
                        typed_mut::<Half>(op, t.down.output())?
                    }
                };
                let mid_h = typed::<Half>(op, source.as_const())?;
                launched(
                    op,
                    generated::moe_q2_down_hot_wide(
                        stream,
                        down_h,
                        w.down,
                        mid_h,
                        plan.selected,
                        plan.counts,
                        plan.offsets,
                        plan.sorted,
                        sel.plan.hot.view(),
                        hot_count,
                        tile_map,
                        n_map,
                        hot_max,
                        mid_dim,
                        out_dim,
                        mtiles,
                        n,
                        rows_cap,
                    ),
                )
            };
            if paired != 0 {
                launch(t, 8, Some(0..paired as usize))?;
            }
            let rest = mapped.then(|| paired as usize..(paired + wide_tiles) as usize);
            if wide_mtiles == 2 {
                launch(t, 2, rest)?;
            } else if !mapped || wide_tiles != 0 {
                launch(t, 4, rest)?;
            }
            return Ok(());
        }
        let plan = sel.plan_views(n);
        let mid_h_src = mid_half != MidHalf::None;
        let gate_mid = mid_half == MidHalf::Gate;
        let (down, down_h, mid, mid_h);
        // F16 mid rows behind the F16 down rows share `down`.
        let down_bytes = t.down.output();
        let (down_part, mid_part) = if mid_half == MidHalf::Down {
            let (d, m) = down_bytes.split_at(down_h_bytes);
            (d, Some(m))
        } else {
            (down_bytes, None)
        };
        if use_f16_down {
            down = None;
            down_h = Some(typed_mut::<Half>(op, down_part)?);
        } else {
            down = Some(typed_mut::<f32>(op, down_part)?);
            down_h = None;
        }
        let gate_view;
        let mid_view;
        if mid_h_src {
            mid = None;
            mid_h = Some(match &mid_part {
                Some(m) => typed::<Half>(op, m.as_const())?.range(0..mid_h_bytes / 2),
                None => {
                    debug_assert!(gate_mid);
                    gate_view = t.gate.output();
                    typed::<Half>(op, gate_view.as_const())?
                }
            });
        } else {
            mid_view = t.mid.output();
            mid = Some(typed::<f32>(op, mid_view.as_const())?);
            mid_h = None;
        }
        let ran = generated::moe_q2_down_hot(
            stream,
            down,
            down_h,
            w.down,
            mid,
            mid_h,
            plan.selected,
            plan.counts,
            plan.offsets,
            plan.sorted,
            sel.plan.hot.view(),
            hot_count,
            hot_max,
            mid_dim,
            out_dim,
            4,
            n,
            rows_cap,
        );
        ensure(op, launched(op, ran)? == Launch::Launched)
    }

    /// The all-Q2_K routes: expert batches with WMMA hot lists from 32
    /// rows, per-row kernels below.
    fn routed_q2k(
        &self,
        op: &'static str,
        t: &mut RoutedMoe<'_>,
        e: &RoutedExperts,
        sel: &mut ExpertSelection,
        clamp: f32,
        x: gufo_hip::DeviceSlice<'_, f32>,
        n: u32,
        w: &ExpertWeights<'_>,
    ) -> Result<()> {
        let stream = self.stream();
        let (in_dim, mid_dim, out_dim) = (e.in_dim, e.mid_dim, e.out_dim);
        let pairs = n * N_EXPERT_USED;
        if n < 32 {
            let weights = typed::<f32>(op, sel.weights.input())?.range(0..pairs as usize);
            let selected = sel.selected_view(n);
            let mid = typed_mut::<f32>(op, t.mid.output())?;
            let xq_bytes = u64::from(n) * u64::from(in_dim / 256) * Q8K;
            if n == 1 && t.down.output().len() as u64 >= xq_bytes {
                self.routed_quantize_x(op, t, x, in_dim, n)?;
                let mid = typed_mut::<f32>(op, t.mid.output())?;
                let xq = t.down.output();
                launched(
                    op,
                    generated::moe_q2k_gate_up_decode(
                        stream,
                        mid,
                        w.gate,
                        w.up,
                        xq.as_const(),
                        selected,
                        weights,
                        in_dim,
                        mid_dim,
                        n,
                        clamp,
                    ),
                )?;
            } else {
                launched(
                    op,
                    generated::moe_q2k_gate_up_rows(
                        stream,
                        mid,
                        w.gate,
                        w.up,
                        x.range(0..n as usize * in_dim as usize),
                        selected,
                        weights,
                        in_dim,
                        mid_dim,
                        n,
                        clamp,
                    ),
                )?;
            }
            let midq_bytes = u64::from(pairs) * u64::from(mid_dim / 256) * Q8K;
            let mid_len = pairs as usize * mid_dim as usize;
            if n == 1 && t.down.output().len() as u64 >= midq_bytes {
                let mid = t.mid.output();
                launched(
                    op,
                    generated::moe_quantize_q8k(
                        stream,
                        t.down.output(),
                        typed::<f32>(op, mid.as_const())?.range(0..mid_len),
                        mid_dim,
                        pairs,
                    ),
                )?;
                let midq = t.down.output();
                return launched(
                    op,
                    generated::moe_q2_down_sum6(
                        stream,
                        typed_mut::<f32>(op, t.out.output())?,
                        w.down,
                        midq.as_const(),
                        selected,
                        mid_dim,
                        out_dim,
                        n,
                    ),
                );
            }
            let mid = t.mid.output();
            return launched(
                op,
                generated::moe_q2k_down_rows(
                    stream,
                    typed_mut::<f32>(op, t.out.output())?,
                    w.down,
                    typed::<f32>(op, mid.as_const())?.range(0..mid_len),
                    selected,
                    mid_dim,
                    out_dim,
                    n,
                ),
            );
        }

        let rows_cap = sel.rows;
        let cap_pairs = u64::from(rows_cap) * u64::from(N_EXPERT_USED);
        let wmma_hot =
            in_dim.is_multiple_of(16) && mid_dim.is_multiple_of(16) && out_dim.is_multiple_of(16);
        {
            let plan = &mut sel.plan;
            launched(
                op,
                generated::moe_route_sort(
                    stream,
                    sel.selected.slice(0..pairs as usize),
                    plan.counts.view_mut(),
                    plan.offsets.view_mut(),
                    typed_mut::<i32>(op, plan.cursors.output())?,
                    plan.sorted.view_mut(),
                    n,
                    rows_cap,
                ),
            )?;
        }
        // The F16 mid, down and input rows of the WMMA kernels, in the
        // shared scratch at the selection's capacity.
        let align = |v: u64| v.div_ceil(256) * 256;
        let mid_h_bytes = if wmma_hot {
            product(op, &[cap_pairs, u64::from(mid_dim), F16])?
        } else {
            0
        };
        let down_h_bytes = if wmma_hot {
            product(op, &[cap_pairs, u64::from(out_dim), F16])?
        } else {
            0
        };
        let x_h_bytes = if wmma_hot {
            product(op, &[u64::from(rows_cap), u64::from(in_dim), F16])?
        } else {
            0
        };
        let down_off = align(mid_h_bytes);
        let x_off = align(down_off + down_h_bytes);
        let total = align(x_off + x_h_bytes).max(256);
        let Some(mut scratch) = self.scratch(total, "routed_moe q2 expert batch buckets") else {
            return Err(Error { op });
        };
        let scratch = scratch.as_mut_slice().range(0..total as usize);
        let (mid_part, rest) = scratch.split_at(down_off as usize);
        let (down_part, x_part) = rest.split_at((x_off - down_off) as usize);
        let mut mid_h = typed_mut::<Half>(op, mid_part.range(0..mid_h_bytes as usize))?;
        let mut down_h = typed_mut::<Half>(op, down_part.range(0..down_h_bytes as usize))?;
        let mut x_h = typed_mut::<Half>(op, x_part.range(0..x_h_bytes as usize))?;
        if wmma_hot {
            let count = n as usize * in_dim as usize;
            launched(
                op,
                generated::convert_f32_to_f16(stream, x_h.reborrow(), x.range(0..count), count),
            )?;
        }
        const HOT: u32 = 8;
        const F16_LOW: u32 = 64;
        let (mut low, mut hot) = (Vec::new(), Vec::new());
        let counts = if wmma_hot {
            self.routed_counts(op, sel)?
        } else {
            [0; 256]
        };
        for e in 0..N_EXPERT {
            let c = counts[e as usize];
            if wmma_hot && c >= HOT {
                if c < F16_LOW {
                    low.push(e)
                } else {
                    hot.push(e)
                }
            }
        }
        let max_of = |list: &[u32]| list.iter().map(|&e| counts[e as usize]).max().unwrap_or(0);
        let (low_max, hot_max) = (max_of(&low), max_of(&hot));
        let scalar_max = if wmma_hot && !(low.is_empty() && hot.is_empty()) {
            HOT
        } else {
            0
        };
        let weights = typed::<f32>(op, sel.weights.input())?;
        {
            let plan = sel.plan_views(n);
            launched(
                op,
                generated::moe_q2k_gate_up_batch(
                    stream,
                    typed_mut::<f32>(op, t.mid.output())?,
                    w.gate,
                    w.up,
                    x,
                    weights,
                    plan.selected,
                    plan.counts,
                    plan.offsets,
                    plan.sorted,
                    scalar_max,
                    in_dim,
                    mid_dim,
                    clamp,
                    n,
                    rows_cap,
                ),
            )?;
        }
        if !low.is_empty() {
            stream
                .upload_indices(&low, &mut sel.plan.hot_low)
                .map_err(|_| Error { op })?;
        }
        if !hot.is_empty() {
            stream
                .upload_indices(&hot, &mut sel.plan.hot)
                .map_err(|_| Error { op })?;
        }
        for (list, mtiles, max) in [(&low, 4, low_max), (&hot, 8, hot_max)] {
            if list.is_empty() {
                continue;
            }
            let plan = sel.plan_views(n);
            let hot_experts = if mtiles == 4 {
                sel.plan.hot_low.view()
            } else {
                sel.plan.hot.view()
            };
            launched(
                op,
                generated::moe_q2k_gate_up_hot(
                    stream,
                    mid_h.reborrow(),
                    w.gate,
                    w.up,
                    x,
                    x_h.as_const(),
                    weights,
                    plan.selected,
                    plan.counts,
                    plan.offsets,
                    plan.sorted,
                    hot_experts,
                    list.len() as u32,
                    max,
                    mtiles,
                    in_dim,
                    mid_dim,
                    clamp,
                    n,
                    rows_cap,
                ),
            )?;
        }
        {
            let plan = sel.plan_views(n);
            let mid = t.mid.output();
            let mid = typed::<f32>(op, mid.as_const())?;
            let all = sel.plan.all.view();
            if wmma_hot {
                launched(
                    op,
                    generated::moe_q2_down_batch_f16(
                        stream,
                        down_h.reborrow(),
                        w.down,
                        mid,
                        plan.selected,
                        plan.counts,
                        plan.offsets,
                        plan.sorted,
                        all,
                        N_EXPERT,
                        2,
                        16,
                        1,
                        scalar_max,
                        mid_dim,
                        out_dim,
                        n,
                        rows_cap,
                    ),
                )?;
            } else {
                launched(
                    op,
                    generated::moe_q2_down_batch(
                        stream,
                        typed_mut::<f32>(op, t.down.output())?,
                        w.down,
                        mid,
                        plan.selected,
                        plan.counts,
                        plan.offsets,
                        plan.sorted,
                        all,
                        N_EXPERT,
                        2,
                        16,
                        1,
                        scalar_max,
                        mid_dim,
                        out_dim,
                        n,
                        rows_cap,
                    ),
                )?;
            }
        }
        for (list, mtiles, max) in [(&low, 4, low_max), (&hot, 8, hot_max)] {
            if list.is_empty() {
                continue;
            }
            let plan = sel.plan_views(n);
            let hot_experts = if mtiles == 4 {
                sel.plan.hot_low.view()
            } else {
                sel.plan.hot.view()
            };
            let ran = generated::moe_q2_down_hot(
                stream,
                None,
                Some(down_h.reborrow()),
                w.down,
                None,
                Some(mid_h.as_const()),
                plan.selected,
                plan.counts,
                plan.offsets,
                plan.sorted,
                hot_experts,
                list.len() as u32,
                max,
                mid_dim,
                out_dim,
                mtiles,
                n,
                rows_cap,
            );
            ensure(op, launched(op, ran)? == Launch::Launched)?;
        }
        if wmma_hot {
            return self.routed_sum_f16(op, t, down_h.as_const(), out_dim, n);
        }
        self.routed_sum(op, t, out_dim, n)
    }
}
