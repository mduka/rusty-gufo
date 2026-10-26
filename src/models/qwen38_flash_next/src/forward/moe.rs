//! The mixture of experts: routing, its hints, and the expert GEMMs.

use super::*;

impl Pass<'_> {
    /// Router, shared expert and routed experts over `mixed` into
    /// `block_out` (`Moe`).
    pub fn moe(&mut self, m: &Moe, n: u32, last_only: bool) -> Result<()> {
        let c = self.c().clone();
        let (r, w) = (self.row, self.w);
        let stream = self.ctx.stream;
        let used = c.num_experts_used;
        let slots = (n * used) as usize;
        // Router logits and the shared-expert gate come out of one GEMM.
        self.ctx.dense(
            &m.router,
            at(&self.s.mixed, r * w.hidden),
            at_mut(&mut self.s.router, r * w.router),
            n,
        )?;
        let ids_at = r * w.used;
        k::router_top_k(
            stream,
            at(&self.s.router, r * w.router),
            c.num_experts + 1,
            self.s.ids.slice_mut(ids_at..ids_at + slots),
            at_mut(&mut self.s.weights, r * w.used),
            n,
            c.num_experts,
            used,
        )
        .map_err(launch_error("router"))?;
        if self.ctx.route.expert_matrix_rows(n) {
            k::expert_counts(
                stream,
                self.s.ids.slice(ids_at..ids_at + slots),
                self.s.expert_counts.view_mut(),
                n,
                c.num_experts,
                used,
            )
            .map_err(launch_error("expert counts"))?;
            stream.download_async(self.s.expert_counts.view().values(), &mut self.host.counts)?;
        }
        // The shared expert does not depend on routing: queued after the
        // count download, before the host prepares the routed dispatch.
        self.ctx.route.shexp_half_ready = false;
        let shared_q8 = {
            let (shexp_gate, shexp_up) = (&mut self.s.shexp_gate, &mut self.s.shexp_up);
            self.ctx.gated_dense(
                &m.shexp_up,
                &m.shexp_gate,
                at(&self.s.mixed, r * w.hidden),
                at_mut(shexp_up, r * w.shared_ff),
                at_mut(shexp_gate, r * w.shared_ff),
                Some(at_mut(&mut self.s.shexp_half, r * w.shared_ff)),
                n,
                Some(&m.shexp_down),
            )?
        };
        if std::mem::replace(&mut self.ctx.route.shexp_half_ready, false) {
            let rows = m.shexp_down.rows as usize;
            launched(
                k::dense_f16_gemm(
                    stream,
                    m.shexp_down.view(),
                    at(&self.s.shexp_half, r * w.shared_ff).range(0..n as usize * w.shared_ff),
                    at_mut(&mut self.s.shexp_out, r * w.hidden).range(0..n as usize * rows),
                    n as usize,
                    rows,
                    m.shexp_down.cols as usize,
                ),
                "shared expert F16 GEMM failed",
            )?;
        } else {
            self.ctx.dense(
                &m.shexp_down,
                at(&self.s.shexp_up, r * w.shared_ff),
                at_mut(&mut self.s.shexp_out, r * w.hidden),
                n,
            )?;
        }
        if last_only
            && m.gate_exps.format() == GgmlType::Q8_0
            && m.up_exps.format() == GgmlType::Q8_0
            && m.down_exps.format() == GgmlType::Q8_0
        {
            // Catch-up keeps only the final row: its routed experts run with
            // the tiled arithmetic of a whole prompt batch.
            let (row, prefill) = (self.row, self.ctx.route.prefill);
            self.row = row + n as usize - 1;
            self.ctx.route.prefill = true;
            let result = self.moe_experts(m, 1, None);
            self.row = row;
            self.ctx.route.prefill = prefill;
            return result;
        }
        // The routed experts read the shared expert's Q8_1 rows of `mixed`.
        self.moe_experts(m, n, shared_q8)
    }

    /// Reads the counts of a tiled batch back and builds the routed tile
    /// maps (`RouteHints`). Tile heights are multiples of 16, so a bucket's
    /// tile count follows from its unpadded rows.
    pub(super) fn route_hints(&mut self, n: u32) -> Result<()> {
        let c = self.c().clone();
        let h = &mut self.ctx.route.hints;
        h.max_rows = 0;
        h.n_tiles = 0;
        h.tiles_64 = 0;
        h.pair_tiles = 0;
        h.pair_offset = 0;
        if !self.ctx.route.expert_matrix_rows(n) {
            return Ok(());
        }
        let counts: Vec<u32> = self.host.counts.host()?.to_vec();
        let tile_rows = routed_tile_rows((n * c.num_experts_used) as usize, &c);
        let tiles = self.host.tiles.host_mut()?;
        let mut n_tiles = 0usize;
        let mut append = |tiles: &mut [u32], rows: u32| {
            for (e, &count) in counts.iter().enumerate() {
                for j in 0..count.div_ceil(rows) {
                    tiles[n_tiles] = e as u32 | (j << 16);
                    n_tiles += 1;
                }
            }
            n_tiles as u32
        };
        let wide = self.ctx.wide_prefill(n);
        let h = &mut self.ctx.route.hints;
        h.tile_rows = tile_rows;
        h.max_rows = counts.iter().copied().max().unwrap_or(0).max(1);
        if wide {
            // Prefill: 64-token tiles for the down projection, then 128-token
            // tiles for the paired gate/up projections.
            h.tiles_64 = append(tiles, 64);
            h.pair_offset = h.tiles_64;
            h.pair_tiles = append(tiles, 128) - h.pair_offset;
        } else {
            h.n_tiles = append(tiles, tile_rows);
        }
        let n_tiles = (h.n_tiles + h.tiles_64 + h.pair_tiles) as usize;
        h.tile_cols = k::routed_tile_cols_for_counts(&counts);
        if n_tiles != 0 {
            self.ctx.stream.upload_indices_async(
                &mut self.host.tiles,
                n_tiles,
                &mut self.s.routed_tiles,
            )?;
        }
        Ok(())
    }

    /// Whether the routed experts take the F16 WMMA route, whose epilogue the
    /// following combine absorbs (`WmmaExperts`).
    pub fn wmma_experts(&self, m: &Moe, n: u32) -> bool {
        let c = self.c();
        self.ctx.route.expert_matrix_rows(n)
            && matches!(m.gate_exps.format(), GgmlType::Q4K | GgmlType::Q5K)
            && m.up_exps.format() == m.gate_exps.format()
            && matches!(m.down_exps.format(), GgmlType::Q5_1 | GgmlType::Q8_0)
            && c.hidden_size.is_multiple_of(256)
            && c.expert_ff.is_multiple_of(64)
    }

    /// The routed experts once router and shared expert are queued
    /// (`MoeExperts`); `x_q8` as for `Ctx::gated_experts`.
    pub fn moe_experts(&mut self, m: &Moe, n: u32, x_q8: Option<usize>) -> Result<()> {
        let c = self.c().clone();
        let (r, w) = (self.row, self.w);
        let stream = self.ctx.stream;
        let used = c.num_experts_used;
        let slots = (n * used) as usize;
        self.route_hints(n)?;
        // Tiled batches: assignments compacted by expert into 16-row padded
        // buckets, token rows narrowed to F16 once, then the F16 matrix-core
        // GEMM per (expert, row tile).
        let wmma = self.wmma_experts(m, n);
        let ids_at = r * w.used;
        if wmma {
            let max_slots = w.max_batch * w.used;
            let compact = k::routed_compact_rows(max_slots, c.num_experts as usize);
            k::routed_compact(
                stream,
                self.s.ids.slice(ids_at..ids_at + slots),
                self.s.expert_counts.view(),
                self.s.routed_bounds.view_mut(),
                self.s.routed_cursors.as_mut_slice(),
                self.s.rows_token.slice_mut(0..compact),
                self.s.rows_slot.slice_mut(0..compact),
                n,
                used,
                c.num_experts,
                max_slots as u32,
            )
            .map_err(launch_error("routed compaction"))?;
            // The router's F16 GEMM (or the mixer) usually left the rows in
            // x_half already.
            self.ctx
                .prepare_half(at(&self.s.mixed, r * w.hidden), n, w.hidden)?;
            let h = self.ctx.route.hints;
            let hidden = w.hidden;
            let ff = w.expert_ff;
            let tiles = |offset: u32, count: u32| (offset as usize, count as usize);
            let x_rows = w.max_batch;
            let (gated_ok, down_rows) = {
                let x_half = self.ctx.stage.x_half.slice(0..x_rows * hidden);
                let ([], [mut gate_e, up_half]) = self.s.phase.views(
                    [],
                    [self.s.gate_e.part(0..max_slots * ff), self.s.up_e.at(0)],
                );
                let up_half = up_half.cast::<F16>().ok_or("misaligned F16 view")?;
                let bounds = self.s.routed_bounds.view();
                let rows_token = self.s.rows_token.slice(0..compact);
                let rows_slot = self.s.rows_slot.slice(0..compact);
                // Prefill pairs the gate/up projections and applies SwiGLU
                // without materializing the gate; decode-time batches keep
                // separate projections.
                let gated = if h.tiles_64 != 0 {
                    let (at_tile, count) = tiles(h.pair_offset, h.pair_tiles);
                    launched(
                        k::routed_gated_f16_gemm(
                            stream,
                            m.gate_exps.view(),
                            m.up_exps.view(),
                            x_half,
                            self.s.routed_tiles.slice(at_tile..at_tile + count),
                            h.pair_tiles,
                            bounds,
                            rows_token,
                            rows_slot,
                            up_half.range(0..max_slots * ff),
                            ff,
                            hidden,
                            c.num_experts,
                            compact,
                            x_rows,
                            max_slots,
                        ),
                        "routed F16 gate/up GEMM failed",
                    )
                } else {
                    let count = h.n_tiles as usize;
                    launched(
                        k::routed_f16_gemm(
                            stream,
                            m.gate_exps.view(),
                            x_half,
                            self.s.routed_tiles.slice(0..count),
                            h.n_tiles,
                            h.tile_rows,
                            bounds,
                            rows_token,
                            rows_slot,
                            gate_e.reborrow(),
                            ff,
                            hidden,
                            c.num_experts,
                            compact,
                            x_rows,
                            max_slots,
                        ),
                        "routed F16 gate/up GEMM failed",
                    )
                    .and_then(|()| {
                        launched(
                            k::routed_f16_gemm_half(
                                stream,
                                m.up_exps.view(),
                                x_half,
                                self.s.routed_tiles.slice(0..count),
                                h.n_tiles,
                                h.tile_rows,
                                bounds,
                                rows_token,
                                rows_slot,
                                Some(gate_e.as_const()),
                                up_half.range(0..max_slots * ff),
                                ff,
                                hidden,
                                c.num_experts,
                                compact,
                                x_rows,
                                max_slots,
                            ),
                            "routed F16 gate/up GEMM failed",
                        )
                    })
                };
                (gated, compact)
            };
            gated_ok?;
            // Wide batches take the 64-token down tiles: they decode each
            // expert's weights over more rows, also where 48-token tiles
            // would be fewer.
            let (at_tile, count, tile_rows) = if h.tiles_64 != 0 {
                (0, h.tiles_64, 64)
            } else {
                (0, h.n_tiles, h.tile_rows)
            };
            let ([up_half], [down]) = self
                .s
                .phase
                .views([self.s.up_e.at(0)], [self.s.down_e.at(0)]);
            let up_half = up_half.cast::<F16>().ok_or("misaligned F16 view")?;
            let down = down.cast::<F16>().ok_or("misaligned F16 view")?;
            launched(
                k::routed_f16_gemm_half(
                    stream,
                    m.down_exps.view(),
                    up_half.range(0..max_slots * ff),
                    self.s.routed_tiles.slice(at_tile..at_tile + count as usize),
                    count,
                    tile_rows,
                    self.s.routed_bounds.view(),
                    self.s.rows_slot.slice(0..down_rows),
                    self.s.rows_slot.slice(0..down_rows),
                    None,
                    down.range(0..max_slots * hidden),
                    hidden,
                    ff,
                    c.num_experts,
                    down_rows,
                    max_slots,
                    max_slots,
                ),
                "routed F16 down GEMM failed",
            )?;
        } else {
            let ([], [gate_e, up_e]) = self.s.phase.views(
                [],
                [
                    self.s.gate_e.at(r * w.used * w.expert_ff),
                    self.s.up_e.at(r * w.used * w.expert_ff),
                ],
            );
            self.ctx.gated_experts(
                &m.gate_exps,
                &m.up_exps,
                at(&self.s.mixed, r * w.hidden),
                self.s.ids.slice(ids_at..ids_at + slots),
                gate_e,
                up_e,
                n,
                used,
                x_q8,
            )?;
            // The down projection sees one (token, slot) row per expert ID.
            let ([gate_e], [down_e]) = self.s.phase.views(
                [self.s.gate_e.at(r * w.used * w.expert_ff)],
                [self.s.down_e.at(r * w.used * w.hidden)],
            );
            self.ctx.experts(
                &m.down_exps,
                gate_e,
                self.s.ids.slice(ids_at..ids_at + slots),
                down_e,
                slots as u32,
                1,
                n,
            )?;
        }
        let gate = at(&self.s.router, r * w.router + c.num_experts as usize);
        let weights = at(&self.s.weights, r * w.used);
        let shared = at(&self.s.shexp_out, r * w.hidden);
        let out = at_mut(&mut self.s.block_out, r * w.hidden);
        let (e1, hidden) = (c.num_experts + 1, c.hidden_size);
        if wmma {
            // The combine that follows folds this epilogue in.
            self.ctx.route.moe_pending = true;
            return Ok(());
        }
        let down = self.s.phase.slice(self.s.down_e.at(r * w.used * w.hidden));
        if self.ctx.matrix_rows(n) {
            k::moe_epilogue_vec4(
                stream, down, weights, shared, gate, e1, out, n, used, hidden,
            )
        } else {
            k::moe_epilogue(
                stream, down, weights, shared, gate, e1, out, n, used, hidden,
            )
        }
        .map_err(launch_error("MoE epilogue"))
    }
}
