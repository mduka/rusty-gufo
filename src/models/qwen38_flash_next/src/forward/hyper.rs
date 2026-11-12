//! Hyper-connections: the mixer over the residual streams, its rows, and
//! the combine after each block.

use super::*;

impl Pass<'_> {
    /// The mixer of `m` over the residual rows at element `res_at`
    /// (`HcMix`). Without `normed` its grouped norm of res is computed here
    /// (F32); otherwise the previous combine produced it, as F16 plus tiled
    /// Q8 on the wide route or as F32 in `xn`. `with_inject` writes the
    /// inject logits (not for the head mixers). A decode batch's scalar
    /// epilogue also writes `mixed` as Q8_1 rows, which its projections read.
    /// `half_only` says every consumer reads the F16 rows: the fused
    /// wide projection then writes no F32 or Q8 copy of `mixed`; `no_f32`
    /// keeps its F16 and tiled Q8 copies but skips the F32 rows nothing
    /// reads.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_mix(
        &mut self,
        m: &Mixer,
        res_at: usize,
        normed: bool,
        with_inject: bool,
        n: u32,
        half_only: bool,
        no_f32: bool,
    ) -> Result<()> {
        let c = self.c().clone();
        let (r, w) = (self.row, self.w);
        let eps = c.rms_eps;
        let xn_q8 = self.ctx.route.xn_q8.take();
        if !normed {
            self.ctx.route.xn_half = false;
            k::rms_norm_rows(
                self.ctx.stream,
                at(&self.s.res, res_at),
                m.norm.view(),
                at_mut(&mut self.s.xn, r * w.hc),
                n,
                c.hc_dim(),
                c.hc_count,
                eps,
            )
            .map_err(launch_error("mixer norm"))?;
        }
        let xn_half = self.ctx.route.xn_half;
        let extras =
            n <= self.ctx.max_batch && c.hidden_size as usize <= self.ctx.model.max_half_cols;
        let fused_projection = xn_half
            && n >= 96
            && extras
            && c.hc_count == 4
            && c.hidden_size == 2560
            && c.hc_low_rank == 320
            && m.up.format() == GgmlType::Q8_0
            && m.up.rows == c.hc_dim()
            && m.up.cols == c.hc_low_rank;
        let q8t_bytes = k::q8_tiled_bytes(n as usize, w.hc);
        if fused_projection {
            // The up projection writes x_half; its input stays in hc_gate.
            launched(
                k::hc_down_f16_gemm(
                    self.ctx.stream,
                    m.down.view(),
                    at(&self.s.xn_q8t, 0).range(0..q8t_bytes),
                    half_mut(self.s.phase.slice_mut(self.s.hc_gate.at(r * w.hc)))?,
                    n,
                ),
                "fused HC down projection failed",
            )?;
        } else {
            let lo = at_mut(&mut self.s.lo, r * w.lo);
            if xn_half {
                launched(
                    k::w8a8_gemm(
                        self.ctx.stream,
                        m.down.view(),
                        at(&self.s.xn_q8t, 0)
                            .range(0..k::q8_tiled_bytes(n as usize, m.down.cols as usize)),
                        lo.range(0..n as usize * w.lo),
                        n as usize,
                        m.down.rows as usize,
                        m.down.cols as usize,
                    ),
                    "W8A8 mixer down projection failed",
                )?;
            } else if let Some(q) = self.ctx.held_q8(
                xn_q8.filter(|_| normed),
                at(&self.s.xn, r * w.hc),
                n,
                m.down.cols,
            ) {
                // The combine left the Q8_1 rows of the norm.
                self.ctx.dense_q(&m.down, &q, lo)?;
            } else {
                self.ctx.dense(&m.down, at(&self.s.xn, r * w.hc), lo, n)?;
            }
            let count = n as usize * w.lo;
            let scale = 1.0 / c.hc_count as f32;
            // The activation writes the up projection's Q8_1 rows.
            let rows = if self.ctx.q8_vector(&m.up, n) && m.up.cols as usize == w.lo {
                let stream = self.ctx.stream;
                let (slot, rows_q8) = self.ctx.claim_q8(n, m.up.cols);
                let launch = k::silu_scale_q8_1(
                    stream,
                    at_mut(&mut self.s.lo, r * w.lo).range(0..count),
                    scale,
                    rows_q8,
                    n,
                    m.up.cols,
                )
                .map_err(launch_error("mixer activation"))?;
                (launch == gufo_hip::Launch::Launched).then_some(slot)
            } else {
                None
            };
            if let Some(slot) = rows {
                let q = self
                    .ctx
                    .record_q8(slot, at(&self.s.lo, r * w.lo), n, m.up.cols);
                self.ctx.dense_q(
                    &m.up,
                    &q,
                    self.s.phase.slice_mut(self.s.hc_gate.at(r * w.hc)),
                )?;
            } else {
                k::silu_scale(
                    self.ctx.stream,
                    at_mut(&mut self.s.lo, r * w.lo).range(0..count),
                    scale,
                    count,
                )
                .map_err(launch_error("mixer activation"))?;
                self.ctx.dense(
                    &m.up,
                    at(&self.s.lo, r * w.lo),
                    self.s.phase.slice_mut(self.s.hc_gate.at(r * w.hc)),
                    n,
                )?;
            }
        }
        let inject_w = if with_inject { m.inject.as_ref() } else { None };
        let fused_inject = inject_w.and_then(|i| i.floats());
        // The previous combine already wrote these partials from the same
        // norm.
        let inject_fused = self.ctx.route.inject_fused.take();
        let inject_done = self.inject_at.is_none()
            && fused_inject.is_some_and(|w| inject_fused == Some(w.address()));
        // A separate inject projection can overwrite the cached half input:
        // keep the full mixed row for the following projection then.
        let half_only = half_only
            && fused_projection
            && extras
            && (inject_w.is_none() || fused_inject.is_some());
        let vectorized = xn_half
            || (self.ctx.matrix_rows(n) && c.hc_count == 4 && c.hidden_size.is_multiple_of(4));
        // `mixed` is rewritten: the input caches of it are stale.
        self.ctx.stage.q8t_src = None;
        self.ctx.stage.half_src = None;
        self.ctx.route.mixed_q8 = None;
        let hidden = c.hidden_size;
        let mixed_len = n as usize * w.hidden;
        let inject_at = self.inject_offset();
        let stream = self.ctx.stream;
        let rows_slot = if !xn_half && !vectorized && !self.ctx.matrix_rows(n) {
            self.mix_rows(fused_inject, inject_at, n, hidden, c.hc_count)?
        } else {
            None
        };
        if xn_half {
            let mixed_q8_bytes = k::q8_tiled_bytes(n as usize, w.hidden);
            let xn16 = at(&self.s.xn_half, r * w.hc);
            let mixed = at_mut(&mut self.s.mixed, r * w.hidden).range(0..mixed_len);
            let (half, q8) = if extras {
                (
                    Some(self.ctx.stage.x_half.slice_mut(0..mixed_len)),
                    (!half_only).then(|| self.ctx.stage.x_q8t.slice_mut(0..mixed_q8_bytes)),
                )
            } else {
                (None, None)
            };

            let inject = at_mut(&mut self.s.inject, inject_at);
            if fused_projection {
                let low_rank = self
                    .s
                    .phase
                    .slice(self.s.hc_gate.at(r * w.hc))
                    .cast::<F16>()
                    .ok_or("misaligned F16 view")?;
                let mixed = (!(half_only || no_f32)).then_some(mixed);
                let result = match fused_inject.filter(|_| !inject_done) {
                    Some(inject_w) => k::hc_mix_f16_gemm_inject(
                        stream,
                        m.up.view(),
                        low_rank,
                        xn16,
                        inject_w,
                        mixed,
                        half,
                        q8,
                        inject,
                        n,
                        hidden,
                        c.hc_low_rank,
                    ),
                    None => k::hc_mix_f16_gemm(
                        stream,
                        m.up.view(),
                        low_rank,
                        xn16,
                        mixed,
                        half,
                        q8,
                        n,
                        hidden,
                        c.hc_low_rank,
                    ),
                };
                launched(result, "fused HC projection failed")?;
            } else {
                let gate = self.s.phase.slice(self.s.hc_gate.at(r * w.hc));
                match fused_inject {
                    Some(inject_w) => k::hc_mix_epilogue_f16_inject(
                        stream, xn16, gate, inject_w, mixed, half, q8, inject, n, hidden,
                    ),
                    None => k::hc_mix_epilogue_f16(stream, xn16, gate, mixed, half, q8, n, hidden),
                }
                .map_err(launch_error("mixer epilogue"))?;
            }
            if extras {
                let address = self
                    .s
                    .mixed
                    .slice(r * w.hidden..self.s.mixed.len())
                    .address();
                self.ctx.stage.half_src = Some((address, n, w.hidden, false));
                if !half_only {
                    self.ctx.stage.q8t_src = Some((address, n, w.hidden));
                }
            }
        } else if rows_slot.is_some() {
            self.ctx.route.mixed_q8 = rows_slot;
        } else {
            let xn = at(&self.s.xn, r * w.hc);
            let gate = self.s.phase.slice(self.s.hc_gate.at(r * w.hc));
            let mixed = at_mut(&mut self.s.mixed, r * w.hidden);
            let inject = at_mut(&mut self.s.inject, inject_at);
            let streams = c.hc_count;
            match (vectorized, fused_inject) {
                (true, Some(inject_w)) => k::hc_mix_epilogue_vec4_inject(
                    stream, xn, gate, inject_w, mixed, inject, n, hidden, streams,
                ),
                (true, None) => {
                    k::hc_mix_epilogue_vec4(stream, xn, gate, mixed, n, hidden, streams)
                }
                (false, Some(inject_w)) => k::hc_mix_epilogue_inject(
                    stream, xn, gate, inject_w, mixed, inject, n, hidden, streams,
                ),
                (false, None) => k::hc_mix_epilogue(stream, xn, gate, mixed, n, hidden, streams),
            }
            .map_err(launch_error("mixer epilogue"))?;
        }
        self.ctx.route.inject_parts = match (fused_inject.is_some(), vectorized) {
            (true, true) => k::inject_parts_vec4(hidden),
            (true, false) => k::inject_parts(hidden),
            (false, _) => 1,
        };
        if let Some(inject_w) = inject_w
            && fused_inject.is_none()
        {
            // A quantized inject projection (the draft block's) reads an F32
            // norm; on the wide route the combine only produced F16.
            if xn_half {
                k::rms_norm_rows(
                    stream,
                    at(&self.s.res, res_at),
                    m.norm.view(),
                    at_mut(&mut self.s.xn, r * w.hc),
                    n,
                    c.hc_dim(),
                    c.hc_count,
                    eps,
                )
                .map_err(launch_error("mixer norm"))?;
            }
            self.ctx.dense(
                inject_w,
                at(&self.s.xn, r * w.hc),
                at_mut(&mut self.s.inject, inject_at),
                n,
            )?;
        }
        Ok(())
    }

    /// The decode mixer epilogue that also writes `mixed` as Q8_1 rows: their
    /// `x_q8` slot, or None (nothing launched) where the rows would be padded.
    pub(super) fn mix_rows(
        &mut self,
        inject_w: Option<DeviceSlice<'_, f32>>,
        inject_at: usize,
        n: u32,
        hidden: u32,
        streams: u32,
    ) -> Result<Option<usize>> {
        let (r, w) = (self.row, self.w);
        let stream = self.ctx.stream;
        let (slot, rows_q8) = self.ctx.claim_q8(n, hidden);
        let xn = at(&self.s.xn, r * w.hc);
        let gate = self.s.phase.slice(self.s.hc_gate.at(r * w.hc));
        let mixed = at_mut(&mut self.s.mixed, r * w.hidden);
        let result = match inject_w {
            Some(inject_w) => k::hc_mix_epilogue_inject_q8_1(
                stream,
                xn,
                gate,
                inject_w,
                mixed,
                at_mut(&mut self.s.inject, inject_at),
                rows_q8,
                n,
                hidden,
                streams,
            ),
            None => k::hc_mix_epilogue_q8_1(stream, xn, gate, mixed, rows_q8, n, hidden, streams),
        }
        .map_err(launch_error("mixer epilogue"))?;
        if result != gufo_hip::Launch::Launched {
            return Ok(None);
        }
        self.ctx
            .record_q8(slot, at(&self.s.mixed, r * w.hidden), n, hidden);
        Ok(Some(slot))
    }

    /// Residual update by the block output, plus the next mixer's grouped
    /// norm (`gamma`); wide batches write it as F16 and tiled Q8
    /// (`Combine`). A pending MoE epilogue folds in here. `down` is the next
    /// mixer's down projection when it reads this norm: its decode GEMV then
    /// gets the norm's Q8_1 rows from the combine. With the next mixer's F32
    /// `inject_w`, a wide combine that defers or applies a deferred update
    /// also writes that mixer's inject partials (`route.inject_fused`).
    pub fn combine(
        &mut self,
        gamma: Option<&DeviceTensor<f32>>,
        down: Option<&Matrix>,
        n: u32,
        inject_w: Option<DeviceSlice<'_, f32>>,
    ) -> Result<()> {
        let c = self.c().clone();
        let (r, w) = (self.row, self.w);
        let stream = self.ctx.stream;
        let xn_half = self.wide_mixer && self.ctx.matrix_rows(n) && gamma.is_some();
        self.ctx.route.xn_half = xn_half;
        self.ctx.route.xn_q8 = None;
        self.ctx.route.inject_fused = None;
        let parts = self.ctx.route.inject_parts;
        let inject_at = self.inject_offset();
        let inject_rows = k::inject_parts(c.hidden_size);
        let (hidden, streams, eps) = (c.hidden_size, c.hc_count, c.rms_eps);
        let q8t_bytes = k::q8_tiled_bytes(n as usize, w.hc);
        // The next mixer's inject pass reads only the F16 norm written here.
        let fused_inject = inject_w.filter(|_| xn_half && n >= 96);
        let defer = std::mem::take(&mut self.ctx.route.defer_residual);
        let mut deferred = std::mem::take(&mut self.ctx.route.res_deferred);
        let defer_at = r * streams as usize;
        if self.ctx.route.moe_pending {
            self.ctx.route.moe_pending = false;
            let used = c.num_experts_used;
            // The tiled expert path writes its F16 rows, slot-indexed within
            // the batch, from the start of down_e whatever the row offset.
            let down = self
                .s
                .phase
                .slice(self.s.down_e.at(0))
                .cast::<F16>()
                .ok_or("misaligned F16 view")?;
            let gate = at(&self.s.router, r * w.router + c.num_experts as usize);
            // Without a norm (the last layer's), the fused combine only
            // updates the residual.
            if self.wide_mixer && self.ctx.matrix_rows(n) {
                let (xn, xn_q8) = if xn_half {
                    (
                        Some(at_mut(&mut self.s.xn_half, r * w.hc)),
                        Some(at_mut(&mut self.s.xn_q8t, 0).range(0..q8t_bytes)),
                    )
                } else {
                    (None, None)
                };
                // The previous combine's deferred residual update comes first.
                let fused_w = fused_inject.filter(|_| deferred);
                let fused = if deferred {
                    k::hc_combine_moe_f16_prior(
                        stream,
                        at_mut(&mut self.s.res, r * w.hc),
                        down,
                        at(&self.s.weights, r * w.used),
                        at(&self.s.shexp_out, r * w.hidden),
                        gate,
                        c.num_experts + 1,
                        used,
                        at_mut(&mut self.s.inject, inject_at),
                        parts,
                        gamma.map(DeviceTensor::view),
                        xn,
                        xn_q8,
                        n,
                        hidden,
                        streams,
                        eps,
                        at(&self.s.block_out, r * w.hidden),
                        at(&self.s.hc_defer_w, defer_at),
                        fused_w,
                        inject_rows,
                    )
                } else {
                    k::hc_combine_moe_f16(
                        stream,
                        at_mut(&mut self.s.res, r * w.hc),
                        down,
                        at(&self.s.weights, r * w.used),
                        at(&self.s.shexp_out, r * w.hidden),
                        gate,
                        c.num_experts + 1,
                        used,
                        at(&self.s.inject, inject_at),
                        parts,
                        gamma.map(DeviceTensor::view),
                        xn,
                        xn_q8,
                        n,
                        hidden,
                        streams,
                        eps,
                    )
                }
                .map_err(launch_error("fused MoE combine"))?;
                if fused == gufo_hip::Launch::Launched {
                    self.ctx.route.inject_fused = fused_w.map(|w| w.address());
                    return Ok(());
                }
            }
            if deferred {
                // Settle the update before the epilogue rewrites block_out.
                k::hc_apply_deferred(
                    stream,
                    at_mut(&mut self.s.res, r * w.hc),
                    at(&self.s.block_out, r * w.hidden),
                    at(&self.s.hc_defer_w, defer_at),
                    n,
                    hidden,
                )
                .map_err(launch_error("deferred residual update"))?;
                deferred = false;
            }
            k::moe_epilogue_vec4_f16(
                stream,
                down,
                at(&self.s.weights, r * w.used),
                at(&self.s.shexp_out, r * w.hidden),
                gate,
                c.num_experts + 1,
                at_mut(&mut self.s.block_out, r * w.hidden),
                n,
                used,
                hidden,
            )
            .map_err(launch_error("MoE epilogue"))?;
        }
        if deferred {
            // Only a fused MoE combine follows a deferring one; settle the
            // update before anything else reads the residual.
            k::hc_apply_deferred(
                stream,
                at_mut(&mut self.s.res, r * w.hc),
                at(&self.s.block_out, r * w.hidden),
                at(&self.s.hc_defer_w, defer_at),
                n,
                hidden,
            )
            .map_err(launch_error("deferred residual update"))?;
        }
        if xn_half {
            let gamma = gamma.ok_or("combine without a norm")?;
            if defer && streams == 4 {
                // The FFN combine that follows rereads this block output
                // (42 MB at 4,096 tokens) instead of this combine writing the
                // residual and the next reading it back (168 MB each).
                let deferring = k::hc_combine_f16_deferred(
                    stream,
                    at_mut(&mut self.s.res, r * w.hc),
                    at(&self.s.block_out, r * w.hidden),
                    at_mut(&mut self.s.inject, inject_at),
                    parts,
                    gamma.view(),
                    at_mut(&mut self.s.xn_half, r * w.hc),
                    Some(at_mut(&mut self.s.xn_q8t, 0).range(0..q8t_bytes)),
                    n,
                    hidden,
                    streams,
                    eps,
                    at_mut(&mut self.s.hc_defer_w, defer_at),
                    fused_inject,
                    inject_rows,
                )
                .map_err(launch_error("deferred combine"))?;
                if deferring == gufo_hip::Launch::Launched {
                    self.ctx.route.res_deferred = true;
                    self.ctx.route.inject_fused = fused_inject.map(|w| w.address());
                    return Ok(());
                }
            }
            return k::hc_combine_f16(
                stream,
                at_mut(&mut self.s.res, r * w.hc),
                at(&self.s.block_out, r * w.hidden),
                at(&self.s.inject, inject_at),
                parts,
                gamma.view(),
                at_mut(&mut self.s.xn_half, r * w.hc),
                Some(at_mut(&mut self.s.xn_q8t, 0).range(0..q8t_bytes)),
                n,
                hidden,
                streams,
                eps,
            )
            .map_err(launch_error("combine"));
        }
        if let (Some(gamma), Some(down)) = (gamma, down)
            && self.ctx.q8_vector(down, n)
            && down.cols == streams * hidden
        {
            let (slot, rows_q8) = self.ctx.claim_q8(n, down.cols);
            let fused = k::hc_combine_q8_1(
                stream,
                at_mut(&mut self.s.res, r * w.hc),
                at(&self.s.block_out, r * w.hidden),
                at(&self.s.inject, inject_at),
                parts,
                gamma.view(),
                at_mut(&mut self.s.xn, r * w.hc),
                rows_q8,
                n,
                hidden,
                streams,
                eps,
            )
            .map_err(launch_error("combine"))?;
            if fused == gufo_hip::Launch::Launched {
                self.ctx
                    .record_q8(slot, at(&self.s.xn, r * w.hc), n, down.cols);
                self.ctx.route.xn_q8 = Some(slot);
                return Ok(());
            }
        }
        k::hc_combine(
            stream,
            at_mut(&mut self.s.res, r * w.hc),
            at(&self.s.block_out, r * w.hidden),
            at(&self.s.inject, inject_at),
            parts,
            gamma.map(DeviceTensor::view),
            at_mut(&mut self.s.xn, r * w.hc),
            n,
            hidden,
            streams,
            eps,
            false,
        )
        .map_err(launch_error("combine"))
    }
}
