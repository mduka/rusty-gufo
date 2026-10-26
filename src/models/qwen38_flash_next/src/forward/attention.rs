//! Sparse attention: the selection and the attention over selected tiles.

use super::*;

/// The device control values of one attention launch.
pub(crate) struct AttentionControl<'a> {
    pub position: IndexSlice<'a>,
    pub first_block: IndexSlice<'a>,
    /// Host copy of the position, for the eager-only decisions.
    pub start_pos: u32,
    pub pool_grid: u32,
    pub max_context: u32,
    pub index_capacity: u32,
    pub sparse: bool,
    /// The session's image positions (`DeviceInput::rope`); `None` rotates
    /// by the physical position.
    pub rope: Option<&'a RowTable<i32, 3>>,
}

impl Pass<'_> {
    /// Gated attention with the sparse indexer over `mixed` into
    /// `block_out` (`Attention`).
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &mut self,
        a: &Attention,
        cache: &mut AttentionState,
        control: &AttentionControl<'_>,
        n: u32,
        last_only: bool,
        projections_ready: bool,
        project_output: bool,
    ) -> Result<()> {
        let c = self.c().clone();
        let (r, w) = (self.row, self.w);
        let stream = self.ctx.stream;
        let kv_row = c.attention_kv_dim();
        let (heads, kv_heads, d, rotary) = (c.num_heads, c.num_kv_heads, c.head_dim, c.rotary_dim);
        let (theta, eps, ratio) = (c.rope_theta, c.rms_eps, c.compress_ratio);
        let max_context = control.max_context;
        let pos = control.position;
        let mut prepared = false;
        if let Some(qkv) = &a.qkv {
            let fused_projection = !projections_ready
                && self.ctx.wide_prefill(n)
                && n <= self.ctx.max_batch
                && self.ctx.dense_f16_route(qkv, n)
                && qkv.rows == 13_312
                && qkv.cols == 2560
                && heads == 24
                && kv_heads == 2
                && d == 256
                && rotary == 64;
            if fused_projection {
                self.ctx
                    .prepare_half(at(&self.s.mixed, r * w.hidden), n, qkv.cols as usize)?;
                let half = self
                    .ctx
                    .stage
                    .x_half
                    .slice(0..n as usize * qkv.cols as usize);
                let ([], [q, gate]) = self
                    .s
                    .phase
                    .views([], [self.s.q.at(r * w.q), self.s.attn_gate.at(r * w.q)]);
                launched(
                    k::attention_f16_gemm(
                        stream,
                        qkv.view(),
                        half,
                        a.q_norm.view(),
                        a.k_norm.view(),
                        q,
                        gate,
                        cache.k_cache.as_mut_slice(),
                        cache.v_cache.as_mut_slice(),
                        n,
                        pos,
                        theta,
                        eps,
                        control.rope,
                        self.ctx.yarn.enabled,
                        self.ctx.yarn.low,
                        self.ctx.yarn.high,
                        self.ctx.yarn.inv_factor,
                        self.ctx.yarn.mscale,
                        max_context,
                    ),
                    "fused attention projection failed",
                )?;
                prepared = true;
            } else {
                if !projections_ready {
                    self.ctx.dense(
                        qkv,
                        at(&self.s.mixed, r * w.hidden),
                        self.s.phase.slice_mut(self.s.qg.at(r * w.qg)),
                        n,
                    )?;
                }
                let ([qg], [q, gate]) = self.s.phase.views(
                    [self.s.qg.at(r * w.qg)],
                    [self.s.q.at(r * w.q), self.s.attn_gate.at(r * w.q)],
                );
                let result = k::prepare_attention(
                    stream,
                    qg,
                    qkv.rows,
                    a.q_norm.view(),
                    a.k_norm.view(),
                    q,
                    gate,
                    cache.k_cache.as_mut_slice(),
                    cache.v_cache.as_mut_slice(),
                    n,
                    heads,
                    kv_heads,
                    d,
                    rotary,
                    pos,
                    theta,
                    eps,
                    control.rope,
                    self.ctx.route.prefill,
                    self.ctx.yarn.enabled,
                    self.ctx.yarn.low,
                    self.ctx.yarn.high,
                    self.ctx.yarn.inv_factor,
                    self.ctx.yarn.mscale,
                    max_context,
                )
                .map_err(launch_error("attention preparation"))?;
                prepared = result == gufo_hip::Launch::Launched;
                if !prepared {
                    let ([qg], [q, gate, keys, values]) = self.s.phase.views(
                        [self.s.qg.at(r * w.qg)],
                        [
                            self.s.q.at(r * w.q),
                            self.s.attn_gate.at(r * w.q),
                            self.s.k.at(r * w.kv),
                            self.s.v.at(r * w.kv),
                        ],
                    );
                    k::unpack_q_gate_kv(
                        stream, qg, qkv.rows, q, gate, keys, values, n, heads, d, kv_row,
                    )
                    .map_err(launch_error("attention unpack"))?;
                }
            }
        } else {
            let (q, kk, v) = (
                a.q.as_ref().ok_or("missing query projection")?,
                a.k.as_ref().ok_or("missing key projection")?,
                a.v.as_ref().ok_or("missing value projection")?,
            );
            if !projections_ready {
                let x = at(&self.s.mixed, r * w.hidden);
                let xq = self.ctx.quantize(x, n, c.hidden_size)?;
                self.ctx
                    .dense_q(q, &xq, self.s.phase.slice_mut(self.s.qg.at(r * w.qg)))?;
                self.ctx
                    .dense_q(kk, &xq, self.s.phase.slice_mut(self.s.k.at(r * w.kv)))?;
                self.ctx
                    .dense_q(v, &xq, self.s.phase.slice_mut(self.s.v.at(r * w.kv)))?;
            }
            let ([qg], [q, gate]) = self.s.phase.views(
                [self.s.qg.at(r * w.qg)],
                [self.s.q.at(r * w.q), self.s.attn_gate.at(r * w.q)],
            );
            k::unpack_q_gate(stream, qg, 2 * c.attention_q_dim(), q, gate, n, heads, d)
                .map_err(launch_error("attention unpack"))?;
        }
        if !prepared {
            let norm = |e| launch_error("head norm")(e);
            k::rms_norm_rows_in_place(
                stream,
                a.q_norm.view(),
                self.s.phase.slice_mut(self.s.q.at(r * w.q)),
                n * heads,
                d,
                1,
                eps,
            )
            .map_err(norm)?;
            k::rms_norm_rows_in_place(
                stream,
                a.k_norm.view(),
                self.s.phase.slice_mut(self.s.k.at(r * w.kv)),
                n * kv_heads,
                d,
                1,
                eps,
            )
            .map_err(norm)?;
            let rope = |e| launch_error("rotary embedding")(e);
            k::rope(
                stream,
                self.s.phase.slice_mut(self.s.q.at(r * w.q)),
                n,
                heads,
                d,
                rotary,
                pos,
                theta,
                control.rope,
                self.ctx.yarn.enabled,
                self.ctx.yarn.low,
                self.ctx.yarn.high,
                self.ctx.yarn.inv_factor,
                self.ctx.yarn.mscale,
                max_context,
            )
            .map_err(rope)?;
            k::rope(
                stream,
                self.s.phase.slice_mut(self.s.k.at(r * w.kv)),
                n,
                kv_heads,
                d,
                rotary,
                pos,
                theta,
                control.rope,
                self.ctx.yarn.enabled,
                self.ctx.yarn.low,
                self.ctx.yarn.high,
                self.ctx.yarn.inv_factor,
                self.ctx.yarn.mscale,
                max_context,
            )
            .map_err(rope)?;
            let store = |e| launch_error("cache store")(e);
            k::store_kv(
                stream,
                self.s.phase.slice(self.s.k.at(r * w.kv)),
                cache.k_cache.as_mut_slice(),
                n,
                kv_row,
                pos,
                max_context,
            )
            .map_err(store)?;
            k::store_kv(
                stream,
                self.s.phase.slice(self.s.v.at(r * w.kv)),
                cache.v_cache.as_mut_slice(),
                n,
                kv_row,
                pos,
                max_context,
            )
            .map_err(store)?;
        }
        // Raw indexer keys live until pooling, in a ring holding the backlog
        // before the budget is reached and one batch. Batched projections
        // keep each row's decode arithmetic for these keys as well.
        let (x, ik) = (
            at(&self.s.mixed, r * w.hidden),
            self.s.phase.slice_mut(self.s.ik.at(r * w.ik)),
        );
        if projections_ready {
            self.ctx.dense_batch(&a.indexer_k, x, ik, n)?;
        } else {
            self.ctx.dense(&a.indexer_k, x, ik, n)?;
        }
        k::store_rows(
            stream,
            self.s.phase.slice(self.s.ik.at(r * w.ik)),
            cache.index_k.as_mut_slice(),
            n,
            c.indexer_head_dim,
            pos,
            control.index_capacity,
            max_context,
        )
        .map_err(launch_error("indexer key store"))?;
        let mask_words = self.mask_words;
        if control.sparse {
            self.ctx.dense(
                &a.indexer_q,
                at(&self.s.mixed, r * w.hidden),
                self.s.phase.slice_mut(self.s.iq.at(r * w.iq)),
                n,
            )?;
            let (iheads, idim) = (c.indexer_heads, c.indexer_head_dim);
            k::rms_norm_rows_in_place(
                stream,
                a.indexer_q_norm.view(),
                self.s.phase.slice_mut(self.s.iq.at(r * w.iq)),
                n * iheads,
                idim,
                1,
                eps,
            )
            .map_err(launch_error("indexer norm"))?;
            k::rope(
                stream,
                self.s.phase.slice_mut(self.s.iq.at(r * w.iq)),
                n,
                iheads,
                idim,
                rotary,
                pos,
                theta,
                control.rope,
                self.ctx.yarn.enabled,
                self.ctx.yarn.low,
                self.ctx.yarn.high,
                self.ctx.yarn.inv_factor,
                self.ctx.yarn.mscale,
                max_context,
            )
            .map_err(launch_error("indexer rotary embedding"))?;
            k::pool_indexer_blocks(
                stream,
                cache.index_k.as_slice(),
                a.indexer_k_norm.view(),
                cache.block_k.as_mut_slice(),
                control.first_block,
                pos,
                n,
                control.pool_grid,
                ratio,
                idim,
                rotary,
                theta,
                eps,
                control.index_capacity,
                control.rope,
                self.ctx.yarn.enabled,
                self.ctx.yarn.low,
                self.ctx.yarn.high,
                self.ctx.yarn.inv_factor,
                self.ctx.yarn.mscale,
                max_context,
            )
            .map_err(launch_error("indexer pooling"))?;
            // Score rows align to full cache lines; the selector only
            // considers complete causal blocks. Wide prefill is never
            // captured, so its rows cover the populated context; graphs keep
            // the maximum stride for replay at later positions, and the score
            // scratch bounds the queries per launch.
            let score_context = if n > VEC_BATCH {
                control.start_pos + n
            } else {
                max_context
            };
            let blocks = score_context.div_ceil(ratio);
            let max_blocks = blocks.div_ceil(32) * 32;
            // Query rows are independent: a power-of-two chunk wasted up to
            // half the scratch and doubled the launches above 160K.
            let fit = (self.s.scores.len() / max_blocks as usize).max(1) as u32;
            let chunk = SELECT_CHUNK.min(fit);
            // Catch-up needs only the final tile's queries; cover its masks
            // even where the dense/sparse boundary shifts the four-query
            // sparse groups.
            let first_query = if last_only && n > 16 { n - 16 } else { 0 };
            let mut t0 = first_query;
            while t0 < n {
                let nn = chunk.min(n - t0);
                // Batches wider than VEC_BATCH are never captured, so the host
                // position bounds the scored range; a replayed graph must
                // cover max_blocks.
                let live = if n > VEC_BATCH {
                    (control.start_pos + t0 + nn) / ratio
                } else {
                    0
                };
                let ([iq], [mask]) = self.s.phase.views(
                    [self.s.iq.at((r + t0 as usize) * w.iq)],
                    [self.s.mask.at((r + t0 as usize) * w.mask)],
                );
                k::select_blocks(
                    stream,
                    iq,
                    cache.block_k.as_slice(),
                    words_mut(mask)?,
                    self.s.scores.as_mut_slice(),
                    nn,
                    pos,
                    t0,
                    iheads,
                    idim,
                    ratio,
                    c.indexer_top_k / ratio,
                    mask_words,
                    max_blocks,
                    live,
                    max_context,
                )
                .map_err(launch_error("block selection"))?;
                t0 += chunk;
            }
        }
        let ctx_len = n as usize * w.q;
        if last_only {
            self.ctx.stream.fill_bytes(
                self.s
                    .phase
                    .slice_mut(self.s.ctx.at(r * w.q))
                    .range(0..ctx_len),
                0,
            )?;
        }
        let checkpoint = self.checkpoint.as_ref().map_or(0, |c| c.tokens);
        if checkpoint != 0 && checkpoint < n {
            // Sparse tiles compact the union of their queries' selected
            // keys: the rows before the checkpoint keep the query grouping
            // of a pass ending there.
            if !self.attend(cache, control, 0, checkpoint, false)?
                || !self.attend(cache, control, checkpoint, n - checkpoint, false)?
            {
                return Err("checkpoint attention geometry is unsupported".into());
            }
            if !project_output {
                return Ok(());
            }
            return self.ctx.dense(
                &a.out,
                self.s.phase.slice(self.s.ctx.at(r * w.q)),
                at_mut(&mut self.s.block_out, r * w.hidden),
                n,
            );
        }
        // Wide batches run the fused WMMA kernel (never in a graph: the key
        // extent is a host value), the per-token kernel the rest.
        if self.ctx.matrix_rows(n) && self.attend(cache, control, 0, n, last_only)? {
            if !project_output {
                return Ok(());
            }
            return self.ctx.dense(
                &a.out,
                self.s.phase.slice(self.s.ctx.at(r * w.q)),
                at_mut(&mut self.s.block_out, r * w.hidden),
                n,
            );
        }
        // Narrow batches split each row's key tiles so a decode step at
        // depth fills the device.
        let split = !self.ctx.matrix_rows(n);
        let ([mask, q], [out]) = self.s.phase.views(
            [self.s.mask.at(r * w.mask), self.s.q.at(r * w.q)],
            [self.s.ctx.at(r * w.q)],
        );
        let mask = if control.sparse {
            Some(words(mask.range(0..n as usize * w.mask))?)
        } else {
            None
        };
        if split {
            k::attention_split(
                stream,
                q,
                cache.k_cache.as_slice(),
                cache.v_cache.as_slice(),
                mask,
                mask_words,
                out,
                self.s.attn_partials.as_mut_slice(),
                ATTN_SPLITS,
                n,
                pos,
                heads,
                kv_heads,
                d,
                ratio,
                self.s.attn_lists.as_mut_slice(),
                self.s.attn_counts.as_mut_slice(),
                crate::state::attention_list_capacity(&c),
                max_context,
            )
        } else {
            k::attention(
                stream,
                q,
                cache.k_cache.as_slice(),
                cache.v_cache.as_slice(),
                mask,
                mask_words,
                out,
                n,
                pos,
                heads,
                kv_heads,
                d,
                ratio,
                max_context,
            )
        }
        .map_err(launch_error("attention"))?;
        // The gate writes the output projection's Q8_1 rows.
        let rows = if project_output && self.ctx.q8_vector(&a.out, n) && a.out.cols as usize == w.q
        {
            let (slot, rows_q8) = self.ctx.claim_q8(n, a.out.cols);
            let ([gate], [out]) = self
                .s
                .phase
                .views([self.s.attn_gate.at(r * w.q)], [self.s.ctx.at(r * w.q)]);
            let launch = k::sigmoid_mul_q8_1(
                stream,
                out.range(0..ctx_len),
                gate.range(0..ctx_len),
                rows_q8,
                n,
                a.out.cols,
            )
            .map_err(launch_error("attention gate"))?;
            (launch == gufo_hip::Launch::Launched).then_some(slot)
        } else {
            None
        };
        if let Some(slot) = rows {
            let q = self.ctx.record_q8(
                slot,
                self.s.phase.slice(self.s.ctx.at(r * w.q)),
                n,
                a.out.cols,
            );
            return self
                .ctx
                .dense_q(&a.out, &q, at_mut(&mut self.s.block_out, r * w.hidden));
        }
        let ([gate], [out]) = self
            .s
            .phase
            .views([self.s.attn_gate.at(r * w.q)], [self.s.ctx.at(r * w.q)]);
        k::sigmoid_mul(
            stream,
            out.range(0..ctx_len),
            gate.range(0..ctx_len),
            ctx_len,
        )
        .map_err(launch_error("attention gate"))?;
        if !project_output {
            return Ok(());
        }
        self.ctx.dense(
            &a.out,
            self.s.phase.slice(self.s.ctx.at(r * w.q)),
            at_mut(&mut self.s.block_out, r * w.hidden),
            n,
        )
    }

    /// The fused WMMA attention of `rows` queries from `first`, gate
    /// included (`attend` in `Attention`). Queries before the token budget
    /// always take the dense tiles, also in a pass that crosses it, so each
    /// position's key sweep (and its rounding) does not depend on where the
    /// prefill chunks fall. The sparse part goes first:
    /// it is the launch that can be refused (false). `last` computes only
    /// the final tile, which lies in the sparse part when there is one.
    pub(super) fn attend(
        &mut self,
        cache: &AttentionState,
        control: &AttentionControl<'_>,
        first: u32,
        rows: u32,
        last: bool,
    ) -> Result<bool> {
        let c = self.c();
        let (heads, kv_heads, d, ratio) =
            (c.num_heads, c.num_kv_heads, c.head_dim, c.compress_ratio);
        let top_k = c.indexer_top_k;
        let pos = control.start_pos + first;
        let dense_rows = if control.sparse && pos < top_k {
            rows.min(top_k - pos)
        } else {
            0
        };
        let mut launch = |from: u32, count: u32, sparse: bool, last: bool| -> Result<bool> {
            let (r, w) = (self.row, self.w);
            let at_row = r + from as usize;
            let ([q, gate, mask], [out]) = self.s.phase.views(
                [
                    self.s.q.at(at_row * w.q),
                    self.s.attn_gate.at(at_row * w.q),
                    self.s.mask.at(at_row * w.mask),
                ],
                [self.s.ctx.at(at_row * w.q)],
            );
            let start = control.start_pos + from;
            let result = if sparse {
                k::wmma_causal_attention_sparse(
                    self.ctx.stream,
                    q,
                    gate,
                    cache.k_cache.as_slice(),
                    cache.v_cache.as_slice(),
                    words(mask.range(0..count as usize * w.mask))?,
                    self.mask_words,
                    out,
                    count,
                    start,
                    heads,
                    kv_heads,
                    d,
                    ratio,
                    last,
                    control.max_context,
                )
            } else {
                k::wmma_causal_attention(
                    self.ctx.stream,
                    q,
                    gate,
                    cache.k_cache.as_slice(),
                    cache.v_cache.as_slice(),
                    out,
                    count,
                    start,
                    heads,
                    kv_heads,
                    d,
                    ratio,
                    last,
                    control.max_context,
                )
            }
            .map_err(launch_error("causal attention"))?;
            Ok(result == gufo_hip::Launch::Launched)
        };
        if dense_rows != rows
            && !launch(first + dense_rows, rows - dense_rows, control.sparse, last)?
        {
            return Ok(false);
        }
        if dense_rows == 0 || (last && dense_rows != rows) {
            return Ok(true);
        }
        launch(first, dense_rows, false, last)
    }
}
