//! Gated DeltaNet, the linear-attention mixer.

use super::*;

impl Pass<'_> {
    /// Gated DeltaNet for one linear layer over `mixed` into `block_out`
    /// (`LinearAttention`).
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &mut self,
        l: &Linear,
        state: &mut LinearState,
        rows: &mut [RollbackRow],
        layer: usize,
        n: u32,
        speculative: bool,
        projections_ready: bool,
        project_output: bool,
    ) -> Result<()> {
        let c = self.c().clone();
        let (r, w) = (self.row, self.w);
        let stream = self.ctx.stream;
        let channels = c.ssm_conv_channels();
        let mut qkv_at = r * w.channels;
        let mut z_at = w.max_batch * w.channels + r * w.value;
        let mut qkv_stride = channels;
        let mut z_stride = c.ssm_value_dim();
        let mut convolved = false;
        let checkpoint_tokens = self.checkpoint.as_ref().map_or(0, |c| c.tokens);
        if let Some(ssm_in) = &l.ssm_in {
            // Wide prefill convolves q|k|v in the projection's LDS tile; the
            // raw boundary rows stay available for the rolling history.
            if !projections_ready
                && !speculative
                && self.ctx.wide_prefill(n)
                && n <= self.ctx.max_batch
                && self.ctx.dense_f16_route(ssm_in, n)
            {
                self.ctx
                    .prepare_half(at(&self.s.mixed, r * w.hidden), n, ssm_in.cols as usize)?;
                let half = self
                    .ctx
                    .stage
                    .x_half
                    .slice(0..n as usize * ssm_in.cols as usize);
                let ([], [qkvz, conv_scratch]) = self
                    .s
                    .phase
                    .views([], [self.s.qkvz.at(r * w.qkvz), self.s.conv_scratch.at(0)]);
                let result = k::dense_f16_ssm_gemm(
                    stream,
                    ssm_in.view(),
                    half,
                    l.conv1d.view(),
                    state.conv_state.as_slice(),
                    qkvz,
                    conv_scratch,
                    n,
                    ssm_in.rows,
                    ssm_in.cols,
                    channels,
                    c.ssm_conv_kernel,
                    checkpoint_tokens,
                )
                .map_err(launch_error("fused SSM projection"))?;
                convolved = result == gufo_hip::Launch::Launched;
            }
            if !projections_ready && !convolved {
                self.ctx.dense(
                    ssm_in,
                    at(&self.s.mixed, r * w.hidden),
                    self.s.phase.slice_mut(self.s.qkvz.at(r * w.qkvz)),
                    n,
                )?;
            }
            qkv_at = r * w.qkvz;
            z_at = r * w.qkvz + channels as usize;
            qkv_stride = ssm_in.rows;
            z_stride = ssm_in.rows;
        } else if !projections_ready {
            let (qkv, gate) = (
                l.qkv.as_ref().ok_or("missing SSM projection")?,
                l.gate.as_ref().ok_or("missing SSM gate")?,
            );
            let x = at(&self.s.mixed, r * w.hidden);
            let q = self.ctx.quantize(x, n, c.hidden_size)?;
            let (qkv_part, z_part) = self
                .s
                .phase
                .slice_mut(self.s.qkvz.at(0))
                .split_at(w.max_batch * w.channels);
            self.ctx.dense_q(
                qkv,
                &q,
                qkv_part.range(r * w.channels..w.max_batch * w.channels),
            )?;
            let z_len = z_part.len();
            self.ctx
                .dense_q(gate, &q, z_part.range(r * w.value..z_len))?;
        }
        if !projections_ready {
            self.ctx.dense(
                &l.alpha_beta,
                at(&self.s.mixed, r * w.hidden),
                self.s
                    .phase
                    .slice_mut(self.s.alpha_beta.at(r * w.alpha_beta)),
                n,
            )?;
        }
        // One activation precision across prefill chunk boundaries.
        let tiled =
            self.ctx.matrix_rows(n) && l.out.format() == GgmlType::Q8_0 && n <= self.ctx.max_batch;
        let half_output = tiled && l.out.rows == 2560 && l.out.cols == 6144;
        // Wide prefill shifts successive F16 rows off the 12 KiB pitch; the
        // F32 scratch has room for the slightly padded rows.
        let half_stride = if half_output && n >= 1024 {
            l.out.cols + 32
        } else {
            0
        };
        let row_split = self.ctx.matrix_rows(n) && !speculative;
        let saved = if speculative {
            n.saturating_sub(1) as usize
        } else {
            0
        };
        let (first, rest) = rows[..saved]
            .split_first_mut()
            .map_or((None, &mut [][..]), |(f, r)| (Some(f), r));
        fn slices(
            row: &mut RollbackRow,
            layer: usize,
        ) -> (DeviceSliceMut<'_, f32>, DeviceSliceMut<'_, f32>) {
            let (conv_at, state_at) = row.layers[layer].expect("rollback row of a linear layer");
            let (conv, state) = row.buffer.as_mut_slice().split_at(state_at);
            (conv.range(conv_at..state_at), state)
        }
        let (conv_first, state_first) = match first {
            Some(row) => {
                let (conv, state) = slices(row, layer);
                (Some(conv), Some(state))
            }
            None => (None, None),
        };
        let mut conv_rest = Vec::with_capacity(rest.len());
        let mut state_rest = Vec::with_capacity(rest.len());
        for row in rest.iter_mut() {
            let (conv, state) = slices(row, layer);
            conv_rest.push(conv);
            state_rest.push(state);
        }
        let rest_count = conv_rest.len() as u32;
        // The recurrent state and history a pass ending at the checkpoint
        // leaves.
        let (capture_state, capture_history) = match &mut self.checkpoint {
            Some(checkpoint) => {
                let (history, state) = checkpoint
                    .storage
                    .linear_mut(layer)
                    .ok_or("checkpoint storage without this linear layer")?;
                (Some(state), Some(history))
            }
            None => (None, None),
        };
        let ([qkv, z, alpha_beta], [conv_scratch, qn, kn, raw, gdn_out]) = self.s.phase.views(
            [
                self.s.qkvz.at(qkv_at),
                self.s.qkvz.at(z_at),
                self.s.alpha_beta.at(r * w.alpha_beta),
            ],
            [
                self.s.conv_scratch.at(0),
                self.s.qn.at(r * w.key),
                self.s.kn.at(r * w.key),
                self.s.gdn_raw.at(r * w.value),
                self.s.gdn_out.at(r * w.value),
            ],
        );
        let (kh, vh, kernel, eps) = (
            c.ssm_num_k_heads,
            c.ssm_num_v_heads,
            c.ssm_conv_kernel,
            c.rms_eps,
        );
        let q8_bytes = k::q8_tiled_bytes(n as usize, w.value);
        // The decode output projection reads the epilogue's Q8_1 rows.
        let rows = project_output
            && !half_output
            && !tiled
            && self.ctx.q8_vector(&l.out, n)
            && l.out.cols as usize == w.value
            && l.out.cols.is_multiple_of(512);
        let mut rows_slot = None;
        let result = if half_stride != 0 {
            k::gated_delta_net_half_strided(
                stream,
                qkv,
                qkv_stride,
                z,
                z_stride,
                alpha_beta,
                l.conv1d.view(),
                l.a.view(),
                l.dt.view(),
                l.norm.view(),
                state.conv_state.as_mut_slice(),
                conv_scratch,
                qn,
                kn,
                raw,
                state.state.as_mut_slice(),
                half_mut(gdn_out)?,
                state_first,
                conv_first,
                &mut state_rest,
                &mut conv_rest,
                rest_count,
                n,
                kh,
                vh,
                kernel,
                row_split,
                convolved,
                eps,
                half_stride,
                capture_state,
                capture_history,
                checkpoint_tokens,
            )
        } else if half_output {
            k::gated_delta_net_half(
                stream,
                qkv,
                qkv_stride,
                z,
                z_stride,
                alpha_beta,
                l.conv1d.view(),
                l.a.view(),
                l.dt.view(),
                l.norm.view(),
                state.conv_state.as_mut_slice(),
                conv_scratch,
                qn,
                kn,
                raw,
                state.state.as_mut_slice(),
                half_mut(gdn_out)?,
                state_first,
                conv_first,
                &mut state_rest,
                &mut conv_rest,
                rest_count,
                n,
                kh,
                vh,
                kernel,
                row_split,
                convolved,
                eps,
                capture_state,
                capture_history,
                checkpoint_tokens,
            )
        } else if tiled {
            k::gated_delta_net_q8(
                stream,
                qkv,
                qkv_stride,
                z,
                z_stride,
                alpha_beta,
                l.conv1d.view(),
                l.a.view(),
                l.dt.view(),
                l.norm.view(),
                state.conv_state.as_mut_slice(),
                conv_scratch,
                qn,
                kn,
                raw,
                state.state.as_mut_slice(),
                self.ctx.stage.x_q8t.slice_mut(0..q8_bytes),
                state_first,
                conv_first,
                &mut state_rest,
                &mut conv_rest,
                rest_count,
                n,
                kh,
                vh,
                kernel,
                row_split,
                convolved,
                eps,
                capture_state,
                capture_history,
                checkpoint_tokens,
            )
        } else if rows {
            let (slot, rows_q8) = self.ctx.claim_q8(n, l.out.cols);
            rows_slot = Some(slot);
            k::gated_delta_net_q8_1(
                stream,
                qkv,
                qkv_stride,
                z,
                z_stride,
                alpha_beta,
                l.conv1d.view(),
                l.a.view(),
                l.dt.view(),
                l.norm.view(),
                state.conv_state.as_mut_slice(),
                conv_scratch,
                qn,
                kn,
                raw,
                state.state.as_mut_slice(),
                gdn_out,
                rows_q8,
                state_first,
                conv_first,
                &mut state_rest,
                &mut conv_rest,
                rest_count,
                n,
                kh,
                vh,
                kernel,
                row_split,
                convolved,
                eps,
                capture_state,
                capture_history,
                checkpoint_tokens,
            )
        } else {
            k::gated_delta_net(
                stream,
                qkv,
                qkv_stride,
                z,
                z_stride,
                alpha_beta,
                l.conv1d.view(),
                l.a.view(),
                l.dt.view(),
                l.norm.view(),
                state.conv_state.as_mut_slice(),
                conv_scratch,
                qn,
                kn,
                raw,
                state.state.as_mut_slice(),
                gdn_out,
                state_first,
                conv_first,
                &mut state_rest,
                &mut conv_rest,
                rest_count,
                n,
                kh,
                vh,
                kernel,
                row_split,
                convolved,
                eps,
                capture_state,
                capture_history,
                checkpoint_tokens,
            )
        };
        launched(result, "gated DeltaNet failed")?;
        if !project_output {
            return Ok(());
        }
        let out = at_mut(&mut self.s.block_out, r * w.hidden);
        if half_stride != 0 {
            let half = self
                .s
                .phase
                .slice(self.s.gdn_out.at(r * w.value))
                .cast::<F16>()
                .ok_or("misaligned F16 view")?;
            let stride = half_stride as usize;
            return launched(
                k::dense_f16_gemm_strided(
                    stream,
                    l.out.view(),
                    half.range(0..(n as usize - 1) * stride + l.out.cols as usize),
                    out.range(0..n as usize * w.hidden),
                    n as usize,
                    l.out.rows as usize,
                    l.out.cols as usize,
                    stride,
                ),
                "SSM output F16 GEMM failed",
            );
        }
        if half_output {
            let half = self
                .s
                .phase
                .slice(self.s.gdn_out.at(r * w.value))
                .cast::<F16>()
                .ok_or("misaligned F16 view")?;
            return launched(
                k::dense_f16_gemm(
                    stream,
                    l.out.view(),
                    half.range(0..n as usize * w.value),
                    out.range(0..n as usize * w.hidden),
                    n as usize,
                    l.out.rows as usize,
                    l.out.cols as usize,
                ),
                "SSM output F16 GEMM failed",
            );
        }
        if tiled {
            self.ctx.stage.q8t_src = None;
            return launched(
                k::w8a8_gemm(
                    stream,
                    l.out.view(),
                    self.ctx.stage.x_q8t.slice(0..q8_bytes),
                    out.range(0..n as usize * w.hidden),
                    n as usize,
                    l.out.rows as usize,
                    l.out.cols as usize,
                ),
                "W8A8 GEMM failed",
            );
        }
        if let Some(slot) = rows_slot {
            let q = self.ctx.record_q8(
                slot,
                self.s.phase.slice(self.s.gdn_out.at(r * w.value)),
                n,
                l.out.cols,
            );
            return self.ctx.dense_q(&l.out, &q, out);
        }
        self.ctx.dense(
            &l.out,
            self.s.phase.slice(self.s.gdn_out.at(r * w.value)),
            out,
            n,
        )
    }
}
