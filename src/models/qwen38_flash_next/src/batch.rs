//! Independent sessions decoded together, a port of `batch.cpp`: the trunk
//! and the draft block share their projections across the sessions of a
//! batch while each keeps its private recurrent state, caches, positions
//! and the decode arithmetic of its own rows (at most eight). Batches run
//! eagerly: their shape changes with every cohort.

use gufo_formats::GgmlType;
use gufo_hip::{DeviceCopy, DeviceSlice, DeviceSliceMut, DeviceTensor, Stream};
use gufo_qwen38_flash_next_kernels as k;

use crate::device::{Block, Linear, Matrix, Mixer, Moe};
use crate::draft_vocab::DRAFT_EXTRA_ROWS;
use crate::executor::{Core, Executor, download_logits, verification_logits};
use crate::forward::{AttentionControl, Pass, at, at_mut};
use crate::mtp::{MTP_CANDIDATES, MtpOutput};
use crate::ngram::hash_ngram_rows;
use crate::routes::{Ctx, Result, launch_error, launched};
use crate::state::{ROLLBACK_ROWS, Session};

/// Sessions a batch holds.
pub const BATCH_SESSIONS: usize = 8;
/// Rows each session brings at most; also the vector kernels' width.
const DECODE_ROWS: u32 = 8;

/// One session's chain in a trunk batch (`BatchItem`).
pub struct BatchItem<'a> {
    pub session: &'a mut Session,
    pub tokens: &'a [i32],
    /// Keeps rollback rows, as a verification forward does.
    pub speculative: bool,
}

/// One session's draft-block input (`MtpBatchItem`): kept trunk rows from
/// `hidden_row`, or with -1 (one token) the carried residual.
pub struct MtpBatchItem<'a> {
    pub session: &'a mut Session,
    pub tokens: &'a [i32],
    pub hidden_row: i32,
    /// Only the draft K/V is needed (no proposal follows): the residual is
    /// not kept.
    pub kv_only: bool,
}

/// One session's draft head (`MtpHeadItem`).
pub struct MtpHeadItem<'a> {
    pub session: &'a mut Session,
    pub output: MtpOutput<'a>,
}

/// Whether any session still runs; asks every one, as the C++ loop does.
/// Cancelled sessions keep their rows, so peers' arithmetic stays the same
/// (`AnyActive`).
fn any_active<'s>(sessions: impl Iterator<Item = &'s mut Session>) -> bool {
    sessions.fold(false, |active, s| active | !s.check_cancelled())
}

/// Disjoint views of `buffer` at ascending, non-overlapping element ranges.
fn split_rows<'a, T: DeviceCopy>(
    mut rest: DeviceSliceMut<'a, T>,
    ranges: &[(usize, usize)],
) -> Vec<DeviceSliceMut<'a, T>> {
    let mut out = Vec::with_capacity(ranges.len());
    let mut at = 0;
    for &(start, len) in ranges {
        let (_, tail) = rest.split_at(start - at);
        let (mine, tail) = tail.split_at(len);
        out.push(mine);
        rest = tail;
        at = start + len;
    }
    out
}

/// Moves `width` elements from `from * width` down to `to * width` (`to <
/// from`) within one buffer.
pub(crate) fn move_row(
    stream: &Stream,
    buffer: DeviceSliceMut<'_, f32>,
    width: usize,
    from: usize,
    to: usize,
) -> Result<()> {
    if from == to {
        return Ok(());
    }
    let (low, high) = buffer.split_at(from * width);
    stream.copy(
        high.as_const().range(0..width),
        low.range(to * width..(to + 1) * width),
    )?;
    Ok(())
}

impl Ctx<'_> {
    /// Quantizes all rows of independent requests once, padded to whole
    /// vector tiles, into the idle prefill staging (`QuantizeBatch`).
    fn quantize_batch(&mut self, x: DeviceSlice<'_, f32>, rows: u32, cols: u32) -> Result<()> {
        let padded = rows.div_ceil(DECODE_ROWS) * DECODE_ROWS;
        let bytes = k::q8_1_bytes(padded as usize, cols as usize);
        if bytes > self.stage.x_q8t.len() {
            return Err("batched projection exceeds activation scratch".into());
        }
        self.stage.q8t_src = None;
        let used = k::q8_1_bytes(rows as usize, cols as usize);
        if padded != rows {
            self.stream
                .fill_bytes(self.stage.x_q8t.slice_mut(used..bytes), 0)?;
        }
        launched(
            k::mmq_quantize_q8_1(
                self.stream,
                x.range(0..rows as usize * cols as usize),
                self.stage.x_q8t.slice_mut(0..used),
                rows,
                cols,
            ),
            "batched activation quantization failed",
        )
    }

    /// `out[t] = w . x[t]` over the rows of several requests, each keeping
    /// its decode arithmetic (`DenseBatch`).
    pub fn dense_batch(
        &mut self,
        w: &Matrix,
        x: DeviceSlice<'_, f32>,
        mut out: DeviceSliceMut<'_, f32>,
        rows: u32,
    ) -> Result<()> {
        if rows <= DECODE_ROWS {
            return self.dense(w, x, out, rows);
        }
        let (m, cols) = (w.rows as usize, w.cols as usize);
        if w.format() != GgmlType::Q8_0 {
            return k::small_gemm(
                self.stream,
                w.view(),
                x.range(0..rows as usize * cols),
                out.range(0..rows as usize * m),
                rows,
                w.rows,
                w.cols,
            )
            .map_err(launch_error("batched small GEMM"));
        }
        self.quantize_batch(x, rows, w.cols)?;
        let mut project = |ctx: &mut Self, first: u32, count: u32| -> Result<()> {
            let (a, b) = (
                k::q8_1_bytes(first as usize, cols),
                k::q8_1_bytes((first + count) as usize, cols),
            );
            launched(
                k::mmq_q8_dense_vec(
                    ctx.stream,
                    w.view(),
                    ctx.stage.x_q8t.slice(a..b),
                    out.reborrow()
                        .range(first as usize * m..(first + count) as usize * m),
                    w.rows,
                    count,
                    w.cols,
                ),
                "batched Q8 projection failed",
            )
        };
        if w.rows == 320 && w.cols == 10240 {
            let full = rows / DECODE_ROWS * DECODE_ROWS;
            project(self, 0, full)?;
            if full != rows {
                project(self, full, rows - full)?;
            }
            return Ok(());
        }
        // Three matrix tiles help 33-48 rows; four cost more than two 32-row
        // launches. Small outputs with long K sweeps stay at eight. The
        // matrix tiles keep each row's vector arithmetic.
        let matrix = (w.cols == 2560 && w.rows >= 1024) || (w.cols == 320 && w.rows == 10240);
        let chunk = if !matrix {
            DECODE_ROWS
        } else if w.cols == 2560 && w.rows >= 8192 && rows > 32 && rows <= 48 {
            48
        } else {
            32
        };
        // Whole 32-row groups in one launch: the matrix tiles keep each row's
        // arithmetic, and one launch reads the weights once.
        if chunk == 32 && rows >= 64 {
            let full = rows / 32 * 32;
            project(self, 0, full)?;
            if full != rows {
                project(self, full, rows - full)?;
            }
            return Ok(());
        }
        let mut r = 0;
        while r < rows {
            project(self, r, chunk.min(rows - r))?;
            r += chunk;
        }
        Ok(())
    }
}

impl Pass<'_> {
    /// The mixer of `m` over all rows with the scalar epilogue of a decode
    /// request (`HcMixBatch`); without `normed` its norm of `res` first.
    pub fn hc_mix_batch(
        &mut self,
        m: &Mixer,
        normed: bool,
        with_inject: bool,
        rows: u32,
    ) -> Result<()> {
        let c = self.ctx.config().clone();
        let w = self.w;
        let stream = self.ctx.stream;
        self.ctx.route.xn_half = false;
        if !normed {
            k::rms_norm_rows(
                stream,
                at(&self.s.res, 0),
                m.norm.view(),
                at_mut(&mut self.s.xn, 0),
                rows,
                c.hc_dim(),
                c.hc_count,
                c.rms_eps,
            )
            .map_err(launch_error("mixer norm"))?;
        }
        self.ctx
            .dense_batch(&m.down, at(&self.s.xn, 0), at_mut(&mut self.s.lo, 0), rows)?;
        let count = rows as usize * w.lo;
        k::silu_scale(
            stream,
            at_mut(&mut self.s.lo, 0).range(0..count),
            1.0 / c.hc_count as f32,
            count,
        )
        .map_err(launch_error("mixer activation"))?;
        self.ctx.dense_batch(
            &m.up,
            at(&self.s.lo, 0),
            self.s.phase.slice_mut(self.s.hc_gate.at(0)),
            rows,
        )?;
        let inject_w = if with_inject { m.inject.as_ref() } else { None };
        let fused = inject_w.and_then(|i| i.floats());
        self.ctx.stage.q8t_src = None;
        self.ctx.stage.half_src = None;
        self.ctx.route.mixed_q8 = None;
        // Every request holds at most eight rows: keep its scalar epilogue
        // even when the whole batch is wider.
        let (xn, gate) = (at(&self.s.xn, 0), self.s.phase.slice(self.s.hc_gate.at(0)));
        let mixed = at_mut(&mut self.s.mixed, 0);
        match fused {
            Some(iw) => k::hc_mix_epilogue_inject(
                stream,
                xn,
                gate,
                iw,
                mixed,
                at_mut(&mut self.s.inject, 0),
                rows,
                c.hidden_size,
                c.hc_count,
            ),
            None => k::hc_mix_epilogue(stream, xn, gate, mixed, rows, c.hidden_size, c.hc_count),
        }
        .map_err(launch_error("mixer epilogue"))?;
        self.ctx.route.inject_parts = if fused.is_some() {
            k::inject_parts(c.hidden_size)
        } else {
            1
        };
        if let Some(iw) = inject_w
            && fused.is_none()
        {
            self.ctx
                .dense_batch(iw, at(&self.s.xn, 0), at_mut(&mut self.s.inject, 0), rows)?;
        }
        Ok(())
    }

    /// Residual update with the decode arithmetic (`CombineBatch`).
    pub fn combine_batch(&mut self, gamma: Option<&DeviceTensor<f32>>, rows: u32) -> Result<()> {
        let c = self.ctx.config().clone();
        k::hc_combine(
            self.ctx.stream,
            at_mut(&mut self.s.res, 0),
            at(&self.s.block_out, 0),
            at(&self.s.inject, 0),
            self.ctx.route.inject_parts,
            gamma.map(DeviceTensor::view),
            at_mut(&mut self.s.xn, 0),
            rows,
            c.hidden_size,
            c.hc_count,
            c.rms_eps,
            true,
        )
        .map_err(launch_error("combine"))
    }

    /// The shared expert's gated projection over all rows into `shexp_up`
    /// (`GatedDenseBatch`).
    fn gated_dense_batch(&mut self, up: &Matrix, gate: &Matrix, rows: u32) -> Result<()> {
        let w = self.w;
        let q8 = up.format() == GgmlType::Q8_0
            && gate.format() == GgmlType::Q8_0
            && up.rows == gate.rows
            && up.cols == gate.cols;
        let cols = up.cols as usize;
        if q8 {
            self.ctx
                .quantize_batch(at(&self.s.mixed, 0), rows, up.cols)?;
        }
        let mut r = 0;
        while r < rows {
            let n = DECODE_ROWS.min(rows - r);
            let out_at = r as usize * up.rows as usize;
            if q8 {
                let (a, b) = (
                    k::q8_1_bytes(r as usize, cols),
                    k::q8_1_bytes((r + n) as usize, cols),
                );
                launched(
                    k::mmq_q8_gated_dense_vec(
                        self.ctx.stream,
                        up.view(),
                        gate.view(),
                        self.ctx.stage.x_q8t.slice(a..b),
                        at_mut(&mut self.s.shexp_up, out_at)
                            .range(0..n as usize * up.rows as usize),
                        up.rows,
                        n,
                        up.cols,
                    ),
                    "batched gated Q8 projection failed",
                )?;
            } else {
                self.ctx.stage.q8t_src = None;
                self.ctx.stage.half_src = None;
                let (shexp_gate, shexp_up) = (&mut self.s.shexp_gate, &mut self.s.shexp_up);
                self.ctx.gated_dense(
                    up,
                    gate,
                    at(&self.s.mixed, r as usize * cols),
                    at_mut(shexp_up, out_at),
                    at_mut(shexp_gate, r as usize * w.shared_ff),
                    None,
                    n,
                    None,
                )?;
                self.ctx.stage.q8t_src = None;
                self.ctx.stage.half_src = None;
            }
            r += DECODE_ROWS;
        }
        Ok(())
    }

    /// Router, shared and routed experts over all rows (`MoeBatch`).
    pub fn moe_batch(&mut self, m: &Moe, rows: u32) -> Result<()> {
        if rows <= DECODE_ROWS {
            return self.moe(m, rows, false);
        }
        let c = self.ctx.config().clone();
        let stream = self.ctx.stream;
        let used = c.num_experts_used;
        let slots = (rows * used) as usize;
        self.ctx.dense_batch(
            &m.router,
            at(&self.s.mixed, 0),
            at_mut(&mut self.s.router, 0),
            rows,
        )?;
        k::router_top_k(
            stream,
            at(&self.s.router, 0),
            c.num_experts + 1,
            self.s.ids.slice_mut(0..slots),
            at_mut(&mut self.s.weights, 0),
            rows,
            c.num_experts,
            used,
        )
        .map_err(launch_error("router"))?;
        self.gated_dense_batch(&m.shexp_up, &m.shexp_gate, rows)?;
        self.ctx.dense_batch(
            &m.shexp_down,
            at(&self.s.shexp_up, 0),
            at_mut(&mut self.s.shexp_out, 0),
            rows,
        )?;
        if m.gate_exps.format() == GgmlType::Q4K && m.up_exps.format() == GgmlType::Q4K {
            // Expert weights shared across requests, with the scalar Q8
            // activations and reduction of each.
            let (ff, hidden) = (c.expert_ff, c.hidden_size);
            let scratch_bytes = self.ctx.stage.mmq.len();
            launched(
                k::mmq_moe_gated_vec(
                    stream,
                    m.gate_exps.view(),
                    m.up_exps.view(),
                    at(&self.s.mixed, 0).range(0..rows as usize * hidden as usize),
                    self.s.ids.slice(0..slots),
                    self.s
                        .phase
                        .slice_mut(self.s.gate_e.at(0))
                        .range(0..slots * ff as usize),
                    ff,
                    hidden,
                    rows,
                    c.num_experts,
                    used,
                    self.ctx.stage.mmq.as_mut_slice(),
                    scratch_bytes,
                ),
                "batched routed vector projection failed",
            )?;
            let ([gate_e], [down_e]) = self
                .s
                .phase
                .views([self.s.gate_e.at(0)], [self.s.down_e.at(0)]);
            launched(
                k::mmq_moe_vec(
                    stream,
                    m.down_exps.view(),
                    gate_e.range(0..slots * ff as usize),
                    self.s.ids.slice(0..slots),
                    down_e.range(0..slots * hidden as usize),
                    hidden,
                    ff,
                    slots as u32,
                    c.num_experts,
                    1,
                    self.ctx.stage.mmq.as_mut_slice(),
                    scratch_bytes,
                ),
                "batched routed vector projection failed",
            )?;
            return k::moe_epilogue(
                stream,
                self.s.phase.slice(self.s.down_e.at(0)),
                at(&self.s.weights, 0),
                at(&self.s.shexp_out, 0),
                at(&self.s.router, c.num_experts as usize),
                c.num_experts + 1,
                at_mut(&mut self.s.block_out, 0),
                rows,
                used,
                hidden,
            )
            .map_err(launch_error("MoE epilogue"));
        }
        let mut r = 0;
        while r < rows {
            self.row = r as usize;
            self.ctx.stage.q8t_src = None;
            self.ctx.stage.half_src = None;
            self.ctx.route.xn_half = false;
            self.ctx.route.moe_pending = false;
            let result = self.moe_experts(m, DECODE_ROWS.min(rows - r), None);
            self.row = 0;
            result?;
            r += DECODE_ROWS;
        }
        self.ctx.stage.q8t_src = None;
        self.ctx.stage.half_src = None;
        self.ctx.route.xn_half = false;
        self.ctx.route.moe_pending = false;
        Ok(())
    }
}

/// Bytes of one layer's batched GatedDeltaNet descriptors (eight of them).
const GDN_SLOT_BYTES: usize = 8 * 200;

impl Core {
    fn ensure_batch_gdn(&mut self, stream: &Stream) -> Result<()> {
        if self.batch_gdn.is_none() {
            let bytes = self.model.config.num_layers as usize * GDN_SLOT_BYTES;
            let host = gufo_hip::HostMemory::pinned(bytes, true)?;
            self.batch_gdn = Some(host.into_device(stream).map_err(|(_, e)| e)?);
        }
        Ok(())
    }

    fn ensure_batch_logits(&mut self, stream: &Stream) -> Result<()> {
        if self.batch_logits.is_none() {
            let rows = BATCH_SESSIONS * DECODE_ROWS.min(self.options.max_logit_rows) as usize;
            self.batch_logits =
                Some(stream.alloc_persistent::<f32>(rows * self.model.config.vocab_size as usize)?);
        }
        Ok(())
    }
}

impl Executor {
    /// Packs independent decode chains for shared projections; every chain
    /// keeps its own state and the decode arithmetic (at most eight rows).
    /// Logit rows stay in batch order until [`Executor::select_batch_logits`]
    /// (`ForwardBatch`).
    pub fn forward_batch(
        &mut self,
        items: &mut [BatchItem<'_>],
    ) -> std::result::Result<(), String> {
        self.core.selected = None;
        let c = self.config().clone();
        let options = self.core.options;
        if items.is_empty() || items.len() > BATCH_SESSIONS {
            return Err("decode batch must contain 1..8 sessions".into());
        }
        // Snapshots borrowing rows the batch writes copy them out first.
        for item in items.iter() {
            let mtp_position = item.session.mtp.as_ref().map_or(0, |m| m.position);
            item.session.preserve(item.session.position, mtp_position)?;
        }
        let model = std::sync::Arc::clone(&self.core.model);
        if model
            .layers
            .iter()
            .any(|l| matches!(&l.block, Block::Attention(a) if a.qkv.is_none()))
        {
            return Err("batched decoding needs the stacked attention projection".into());
        }
        let mut offsets = [0u32; BATCH_SESSIONS];
        let mut sparse = [false; BATCH_SESSIONS];
        let mut complete = [0u32; BATCH_SESSIONS];
        let mut rows = 0u32;
        for (i, item) in items.iter().enumerate() {
            let n = item.tokens.len() as u32;
            let s = &item.session;
            if s.owner != self.id
                || n == 0
                || n > DECODE_ROWS
                || n > options.max_logit_rows
                || (item.speculative && n > options.max_speculative)
                || (s.mtp_enabled && n > options.max_speculative)
                || s.position + n > s.max_context
            {
                return Err("invalid session or chain in decode batch".into());
            }
            if item
                .tokens
                .iter()
                .any(|&t| t < 0 || t as u32 >= c.vocab_size)
            {
                return Err("token out of range".into());
            }
            offsets[i] = rows;
            rows += n;
            let end = s.position + n;
            sparse[i] = end > c.indexer_top_k;
            complete[i] = end / c.compress_ratio;
        }
        if rows > options.max_batch {
            return Err("decode batch exceeds executor capacity".into());
        }
        if !any_active(items.iter_mut().map(|i| &mut *i.session)) {
            return Ok(());
        }
        for item in items.iter_mut() {
            item.session.prepare_vision(item.tokens.len() as u32)?;
        }
        let limit = options.max_speculative.min(ROLLBACK_ROWS + 1);
        for item in items.iter_mut() {
            if item.speculative {
                item.session.ensure_rollback(
                    &self.stream,
                    &c,
                    item.tokens.len() as u32 - 1,
                    limit,
                )?;
            }
        }
        self.core
            .ensure_batch_logits(&self.stream)
            .map_err(|e| e.to_string())?;
        if self.core.ple_pending {
            self.core
                .pass(&self.stream, 0)
                .wait_ple()
                .map_err(|e| e.to_string())?;
        }
        if let Some(ngram) = &self.core.ngram {
            self.core.prefetch.finish(ngram);
        }
        self.core.batch_rows = 0;
        let heads = c.ple_heads as usize;
        {
            let host = self
                .core
                .host
                .tokens
                .host_mut()
                .map_err(|e| e.to_string())?;
            for (i, item) in items.iter_mut().enumerate() {
                let s = &mut *item.session;
                s.mutation_epoch += 1;
                s.spec_base = s.position;
                s.spec_tokens = if item.speculative {
                    item.tokens.len() as u32
                } else {
                    0
                };
                let at = offsets[i] as usize;
                for (slot, &t) in host[at..].iter_mut().zip(item.tokens) {
                    *slot = t as u32;
                }
                if c.ple_layer.is_some() {
                    for (t, token) in item.tokens.iter().enumerate() {
                        let row = (at + t) * heads;
                        hash_ngram_rows(
                            &c,
                            &mut s.ngram,
                            std::slice::from_ref(token),
                            &mut self.core.host.ple_rows[row..row + heads],
                        );
                        if item.speculative && t + 1 < item.tokens.len() {
                            s.ngram_snapshots[t] = s.ngram;
                        }
                    }
                }
            }
        }
        if c.ple_layer.is_some() {
            let ngram = self.core.ngram.as_ref().ok_or("n-gram table is not open")?;
            if !ngram.start_read(&self.core.host.ple_rows[..rows as usize * heads]) {
                return Err("batched n-gram read could not start".into());
            }
            self.core.ple_pending = true;
        }
        let body = BatchBody {
            offsets,
            sparse,
            complete,
            rows,
        };
        let result = self.core.forward_batch_body(&self.stream, items, &body);
        let drained = self.stream.synchronize();
        self.core.reset_route();
        result.map_err(|e| e.to_string())?;
        drained.map_err(|e| format!("batched forward: {e}"))?;
        for (i, item) in items.iter_mut().enumerate() {
            let s = &mut *item.session;
            if s.cancelled() {
                continue;
            }
            s.position += item.tokens.len() as u32;
            if sparse[i] {
                s.blocks = complete[i];
            }
        }
        self.core.batch_rows = rows;
        Ok(())
    }

    /// Makes rows `offset..offset + rows` of the last batch the verification
    /// rows and copies them into `logits` when given (`SelectBatchLogits`).
    pub fn select_batch_logits(
        &mut self,
        offset: u32,
        rows: u32,
        logits: Option<&mut [f32]>,
    ) -> std::result::Result<(), String> {
        let batch_rows = self.core.batch_rows;
        if rows == 0
            || rows > self.core.options.max_logit_rows
            || offset > batch_rows
            || rows > batch_rows - offset
        {
            return Err("logit rows outside the completed decode batch".into());
        }
        self.core.selected = Some(offset);
        if let Some(logits) = logits {
            let vocab = self.config().vocab_size as usize;
            let count = rows as usize * vocab;
            let Core {
                s,
                host,
                selected,
                batch_logits,
                ..
            } = &mut self.core;
            let source =
                verification_logits(*selected, batch_logits, &s.logits, vocab as u32, count);
            // The batch has completed: the download waits for nothing.
            download_logits(&self.stream, source, host, logits)
                .map_err(|e| format!("batched forward: {e}"))?;
        }
        Ok(())
    }

    /// Short predictor chains of independent sessions with shared
    /// projections and private caches, positions and carried hidden states
    /// (`MtpForwardBatch`).
    pub fn mtp_forward_batch(
        &mut self,
        items: &mut [MtpBatchItem<'_>],
    ) -> std::result::Result<(), String> {
        let c = self.config().clone();
        let options = self.core.options;
        if !self.has_mtp() || items.is_empty() || items.len() > BATCH_SESSIONS {
            return Err("invalid MTP body batch".into());
        }
        for item in items.iter() {
            let mtp_position = item.session.mtp.as_ref().map_or(0, |m| m.position);
            item.session.preserve(item.session.position, mtp_position)?;
        }
        let mut offsets = [0u32; BATCH_SESSIONS];
        let mut final_rows = [0u32; BATCH_SESSIONS];
        let mut rows = 0u32;
        for (i, item) in items.iter().enumerate() {
            let n = item.tokens.len() as u32;
            let s = &item.session;
            let position = s.mtp.as_ref().map_or(0, |m| m.position);
            if s.owner != self.id
                || !s.mtp_enabled
                || n == 0
                || n > DECODE_ROWS
                || (item.hidden_row < 0
                    && (n != 1
                        || !s.mtp.as_ref().is_some_and(|m| m.residual_valid)
                        || item.kv_only))
                || (item.hidden_row >= 0 && item.hidden_row as u32 + n > options.max_speculative)
                || position + n > s.max_context
            {
                return Err("invalid MTP body request".into());
            }
            if item
                .tokens
                .iter()
                .any(|&t| t < 0 || t as u32 >= c.vocab_size)
            {
                return Err("MTP token out of range".into());
            }
            offsets[i] = rows;
            rows += n;
            final_rows[i] = rows - 1;
        }
        if rows > options.max_batch {
            return Err("MTP body batch exceeds executor capacity".into());
        }
        if !any_active(items.iter_mut().map(|i| &mut *i.session)) {
            return Ok(());
        }
        if items.len() == 1 {
            let item = &mut items[0];
            return self.mtp_forward(
                item.session,
                item.tokens,
                item.hidden_row,
                MtpOutput {
                    kv_only: item.kv_only,
                    ..MtpOutput::default()
                },
            );
        }
        {
            let host = self
                .core
                .host
                .tokens
                .host_mut()
                .map_err(|e| e.to_string())?;
            for (i, item) in items.iter().enumerate() {
                for (slot, &t) in host[offsets[i] as usize..].iter_mut().zip(item.tokens) {
                    *slot = t as u32;
                }
            }
        }
        let result = self
            .core
            .mtp_batch_body(&self.stream, items, &offsets, &final_rows, rows);
        let drained = self.stream.synchronize();
        self.core.reset_route();
        result.map_err(|e| e.to_string())?;
        drained.map_err(|e| format!("batched forward: {e}"))?;
        for item in items.iter_mut() {
            let s = &mut *item.session;
            if s.cancelled() {
                continue;
            }
            let mtp = s.mtp.as_mut().ok_or("no MTP block loaded")?;
            mtp.position += item.tokens.len() as u32;
            mtp.residual_valid = !item.kv_only;
            if mtp.position > c.indexer_top_k {
                mtp.blocks = mtp.position / c.compress_ratio;
            }
        }
        Ok(())
    }

    /// Projects each session's carried draft residual with shared weights
    /// (`MtpHeads`).
    pub fn mtp_heads(&mut self, items: &mut [MtpHeadItem<'_>]) -> std::result::Result<(), String> {
        self.core.selected = None;
        let c = self.config().clone();
        if !self.has_mtp()
            || items.is_empty()
            || items.len() > BATCH_SESSIONS
            || items.len() > self.core.options.max_batch as usize
        {
            return Err("invalid MTP head batch".into());
        }
        for item in items.iter() {
            let s = &item.session;
            if s.owner != self.id
                || !s.mtp_enabled
                || s.mtp
                    .as_ref()
                    .is_none_or(|m| m.position == 0 || !m.residual_valid)
                || item.output.kv_only
                || (item.output.token.is_none() && item.output.candidates.is_none())
            {
                return Err("invalid MTP head request".into());
            }
        }
        if !any_active(items.iter_mut().map(|i| &mut *i.session)) {
            return Ok(());
        }
        let model = std::sync::Arc::clone(&self.core.model);
        let head = &model
            .mtp
            .as_ref()
            .and_then(|l| l.nextn.as_ref())
            .ok_or("MTP block without its projections")?
            .head;
        if items.len() > 1 {
            self.core
                .ensure_batch_logits(&self.stream)
                .map_err(|e| e.to_string())?;
        }
        let n = items.len();
        let hc = c.hc_dim() as usize;
        let cols = c.hidden_size as usize;
        let vocab = c.vocab_size as usize;
        let count = vocab.min(MTP_CANDIDATES);
        let single = n == 1;
        // Each request keeps the head it chose. A
        // small-head row's logits are the prefix rows, then the table rows,
        // `small` wide. When every row reads the small head they go through
        // one projection over the shared prefix, then one over each
        // request's own table. A mixed batch reads the full head anyway, so
        // every row goes through it once and each small-head row gathers its
        // table rows from its own full logits.
        let draft = model.draft.as_ref();
        let prefix = draft.map_or(0, |d| d.prefix.rows as usize);
        let small = prefix + DRAFT_EXTRA_ROWS as usize;
        let mut uses = [false; BATCH_SESSIONS];
        for (i, item) in items.iter().enumerate() {
            uses[i] = item.output.draft_subset && self.core.uses_draft_subset(item.session);
        }
        let mut subset_rows = uses.iter().filter(|&&u| u).count();
        // The prefix projection lands in the buffer's tail, then moves to
        // its rows.
        if subset_rows != 0
            && subset_rows * (small + prefix) + (n - subset_rows) * vocab > BATCH_SESSIONS * vocab
        {
            subset_rows = 0;
        }
        let mixed = subset_rows != 0 && subset_rows != n;
        let mut slot = [0usize; BATCH_SESSIONS];
        let (mut a, mut b) = (0, subset_rows);
        for (i, slot) in slot.iter_mut().enumerate().take(n) {
            let next = if subset_rows != 0 && uses[i] {
                &mut a
            } else {
                &mut b
            };
            *slot = *next;
            *next += 1;
        }
        let in_subset = |i: usize| slot[i] < subset_rows;
        let rows_of = |i: usize| if in_subset(i) { small } else { vocab };
        let logits_at = |i: usize| {
            if mixed || single {
                slot[i] * vocab
            } else if in_subset(i) {
                slot[i] * small
            } else {
                subset_rows * small + (slot[i] - subset_rows) * vocab
            }
        };
        let result = (|| -> Result<()> {
            let core = &mut self.core;
            let stream = &self.stream;
            core.reset_route();
            for (i, item) in items.iter().enumerate() {
                let mtp = item.session.mtp.as_ref().ok_or("no MTP block loaded")?;
                k::rms_norm_rows(
                    stream,
                    mtp.h.slice(0..hc),
                    head.norm.view(),
                    at_mut(&mut core.s.xn, slot[i] * hc),
                    1,
                    c.hc_dim(),
                    c.hc_count,
                    c.rms_eps,
                )
                .map_err(launch_error("draft head norm"))?;
            }
            if single {
                // One session keeps the single-row head (`MtpHead`).
                let mut pass = core.pass(stream, 0);
                pass.hc_mix(head, 0, true, false, 1, false, false)?;
                match (
                    draft,
                    items[0].session.mtp.as_ref().and_then(|m| m.draft.as_ref()),
                ) {
                    (Some(draft), Some(table)) if subset_rows == 1 => {
                        pass.draft_logits(draft, table, 0)?;
                    }
                    _ => pass.ctx.dense(
                        &model.output,
                        at(&pass.s.mixed, 0),
                        at_mut(&mut pass.s.logits, 0),
                        1,
                    )?,
                }
            } else {
                let mut logits = core.batch_logits.take().ok_or("batch logits missing")?;
                let mut pass = core.pass(stream, 0);
                let result = (|| -> Result<()> {
                    pass.hc_mix_batch(head, true, false, n as u32)?;
                    let tables = |i: usize| {
                        items[i]
                            .session
                            .mtp
                            .as_ref()
                            .and_then(|m| m.draft.as_ref())
                            .ok_or("session without a draft table")
                    };
                    if subset_rows == 0 || mixed {
                        pass.ctx.dense_batch(
                            &model.output,
                            at(&pass.s.mixed, 0),
                            logits.slice_mut(0..n * vocab),
                            n as u32,
                        )?;
                        for i in (0..n).filter(|&i| in_subset(i)) {
                            let at_row = logits_at(i);
                            k::gather_draft_logits(
                                stream,
                                logits.slice_mut(at_row..at_row + vocab),
                                tables(i)?.ids.view(),
                                prefix as u32,
                                DRAFT_EXTRA_ROWS,
                                vocab as u32,
                            )
                            .map_err(launch_error("draft logit gather"))?;
                        }
                        return Ok(());
                    }
                    let draft = draft.ok_or("no small draft head")?;
                    let staged = BATCH_SESSIONS * vocab - subset_rows * prefix;
                    pass.ctx.dense_batch(
                        &draft.prefix,
                        at(&pass.s.mixed, 0),
                        logits.slice_mut(staged..staged + subset_rows * prefix),
                        subset_rows as u32,
                    )?;
                    // Out of the tail into rows of the small head's width.
                    let (tail, rows) = (
                        staged..staged + subset_rows * prefix,
                        0..subset_rows * small,
                    );
                    let ([from], [to]) = logits.views([tail], [rows]);
                    stream.copy_2d(from, prefix, to, small, prefix, subset_rows)?;
                    for i in (0..n).filter(|&i| in_subset(i)) {
                        let q =
                            pass.ctx
                                .quantize(at(&pass.s.mixed, slot[i] * cols), 1, cols as u32)?;
                        let at_row = logits_at(i) + prefix;
                        crate::forward::draft_table_logits(
                            &mut pass.ctx,
                            tables(i)?,
                            &q,
                            vocab as u32,
                            logits.slice_mut(at_row..at_row + DRAFT_EXTRA_ROWS as usize),
                        )?;
                    }
                    Ok(())
                })();
                core.batch_logits = Some(logits);
                result?;
            }
            for (i, item) in items.iter_mut().enumerate() {
                if item.session.check_cancelled() {
                    continue;
                }
                let rows = rows_of(i);
                let at_row = logits_at(i);
                let logits = if single {
                    core.s.logits.slice(0..rows)
                } else {
                    core.batch_logits
                        .as_ref()
                        .ok_or("batch logits missing")?
                        .slice(at_row..at_row + rows)
                };
                let table = item
                    .session
                    .mtp
                    .as_ref()
                    .and_then(|m| m.draft.as_ref())
                    .filter(|_| in_subset(i));
                let remap = |ids: gufo_hip::DeviceSliceMut<'_, u32>, n: u32| -> Result<()> {
                    match table {
                        Some(table) => k::remap_draft_ids(
                            stream,
                            ids,
                            n,
                            prefix as u32,
                            table.ids.view(),
                            DRAFT_EXTRA_ROWS,
                            c.vocab_size,
                        )
                        .map_err(launch_error("draft ID remap")),
                        None => Ok(()),
                    }
                };
                if item.output.candidates.is_some() {
                    k::mtp_top_candidates(
                        stream,
                        logits,
                        core.s.mtp_ids.as_mut_slice(),
                        core.s.mtp_scratch_ids.as_mut_slice(),
                        rows as u32,
                        k::mtp_candidate_workspace(rows as u32),
                    )
                    .map_err(launch_error("draft candidates"))?;
                    remap(core.s.mtp_ids.slice_mut(0..count), count as u32)?;
                    stream.download_async(
                        core.s.mtp_ids.slice(0..MTP_CANDIDATES + count),
                        &mut core.host.draft_candidates[i],
                    )?;
                }
                if item.output.token.is_some() {
                    k::argmax(
                        stream,
                        logits,
                        core.s.mtp_argmax.slice_mut(0..64 * 8),
                        core.s.mtp_token.as_mut_slice(),
                        1,
                        rows as u32,
                    )
                    .map_err(launch_error("draft argmax"))?;
                    let token = core
                        .s
                        .mtp_token
                        .slice_mut(0..1)
                        .cast::<u32>()
                        .ok_or("misaligned draft token")?;
                    remap(token, 1)?;
                    stream.download_async(
                        core.s.mtp_token.slice(0..1),
                        &mut core.host.draft_tokens[i],
                    )?;
                }
            }
            Ok(())
        })();
        let drained = self.stream.synchronize();
        self.core.reset_route();
        result.map_err(|e| e.to_string())?;
        drained.map_err(|e| format!("batched forward: {e}"))?;
        for (i, item) in items.iter_mut().enumerate() {
            if item.session.cancelled() {
                continue;
            }
            if let Some(token) = item.output.token.as_deref_mut() {
                *token = self.core.host.draft_tokens[i]
                    .host()
                    .map_err(|e| e.to_string())?[0];
            }
            if let Some(candidates) = item.output.candidates.as_deref_mut() {
                let words = self.core.host.draft_candidates[i]
                    .host()
                    .map_err(|e| e.to_string())?;
                candidates.size = count;
                candidates.ids.copy_from_slice(&words[..MTP_CANDIDATES]);
                for (logit, &bits) in candidates.logits.iter_mut().zip(&words[MTP_CANDIDATES..]) {
                    *logit = f32::from_bits(bits);
                }
            }
        }
        Ok(())
    }
}

/// The per-session shape of one trunk batch.
struct BatchBody {
    offsets: [u32; BATCH_SESSIONS],
    sparse: [bool; BATCH_SESSIONS],
    complete: [u32; BATCH_SESSIONS],
    rows: u32,
}

impl Core {
    /// Everything a trunk batch enqueues after its host preparation.
    fn forward_batch_body(
        &mut self,
        stream: &Stream,
        items: &mut [BatchItem<'_>],
        b: &BatchBody,
    ) -> Result<()> {
        if !any_active(items.iter_mut().map(|i| &mut *i.session)) {
            return Ok(());
        }
        let c = self.model.config.clone();
        let model = std::sync::Arc::clone(&self.model);
        let rows = b.rows;
        self.reset_route();
        let batch_gdn = items.len() > 1 && c.ssm_head_dim == 128 && c.ssm_conv_kernel == 4;
        if batch_gdn {
            self.ensure_batch_gdn(stream)?;
        }
        for item in items.iter_mut() {
            let s = &mut *item.session;
            let n = item.tokens.len() as u32;
            s.control.set_position(stream, n, s.position)?;
            s.control.set_blocks(stream, s.blocks)?;
        }
        stream.upload_indices_async(&mut self.host.tokens, rows as usize, &mut self.s.tokens)?;
        let token_embd = &model.token_embd;
        k::embed_tokens(
            stream,
            token_embd.view(),
            self.s.tokens.slice(0..rows as usize),
            at_mut(&mut self.s.res, 0),
            rows,
            c.hidden_size,
            c.hc_count,
            token_embd.rows,
        )
        .map_err(launch_error("token embedding"))?;
        let hc_dim = (c.hidden_size * c.hc_count) as usize;
        for (item, &offset) in items.iter().zip(&b.offsets) {
            item.session
                .vision
                .inject(
                    stream,
                    at_mut(&mut self.s.res, offset as usize * hc_dim),
                    item.session.position,
                    item.tokens.len() as u32,
                    c.hidden_size,
                    c.hc_count,
                )
                .map_err(launch_error("image injection"))?;
        }
        let w = self.widths;
        let mut normed = false;
        for il in 0..c.num_layers as usize {
            if !any_active(items.iter_mut().map(|i| &mut *i.session)) {
                return Ok(());
            }
            let layer = &model.layers[il];
            if c.is_ple_layer(il as u32) {
                let p = layer.ple.as_ref().ok_or("PLE layer without its weights")?;
                {
                    let mut pass = self.pass(stream, 0);
                    pass.wait_ple()?;
                    pass.upload_ple(rows)?;
                }
                for (i, item) in items.iter_mut().enumerate() {
                    let s = &mut *item.session;
                    if s.cancelled() {
                        continue;
                    }
                    let history = s
                        .ple_history
                        .as_mut()
                        .ok_or("PLE layer without its history")?;
                    let n = item.tokens.len() as u32;
                    self.use_scratch();
                    let mut pass = self.pass(stream, b.offsets[i] as usize);
                    pass.ple(p, history, &mut s.rollback, n, item.speculative, true)?;
                }
            }
            self.use_scratch();
            let mut pass = self.pass(stream, 0);
            pass.hc_mix_batch(&layer.hc_attn, normed, true, rows)?;
            match &layer.block {
                Block::Linear(l) => {
                    if let Some(ssm_in) = &l.ssm_in {
                        pass.ctx.dense_batch(
                            ssm_in,
                            at(&pass.s.mixed, 0),
                            pass.s.phase.slice_mut(pass.s.qkvz.at(0)),
                            rows,
                        )?;
                    } else {
                        let (qkv, gate) = (
                            l.qkv.as_ref().ok_or("missing SSM projection")?,
                            l.gate.as_ref().ok_or("missing SSM gate")?,
                        );
                        let (qkv_part, z_part) = pass
                            .s
                            .phase
                            .slice_mut(pass.s.qkvz.at(0))
                            .split_at(w.max_batch * w.channels);
                        pass.ctx
                            .dense_batch(qkv, at(&pass.s.mixed, 0), qkv_part, rows)?;
                        pass.ctx
                            .dense_batch(gate, at(&pass.s.mixed, 0), z_part, rows)?;
                    }
                    pass.ctx.dense_batch(
                        &l.alpha_beta,
                        at(&pass.s.mixed, 0),
                        pass.s.phase.slice_mut(pass.s.alpha_beta.at(0)),
                        rows,
                    )?;
                }
                Block::Attention(a) => {
                    let qkv = a.qkv.as_ref().ok_or("missing attention projection")?;
                    pass.ctx.dense_batch(
                        qkv,
                        at(&pass.s.mixed, 0),
                        pass.s.phase.slice_mut(pass.s.qg.at(0)),
                        rows,
                    )?;
                }
            }
            match &layer.block {
                Block::Linear(l) if batch_gdn => self.gdn_batch(stream, items, b, l, il)?,
                Block::Linear(l) => {
                    for (i, item) in items.iter_mut().enumerate() {
                        let s = &mut *item.session;
                        let n = item.tokens.len() as u32;
                        let r = b.offsets[i] as usize;
                        if s.check_cancelled() {
                            stream.fill_bytes(
                                self.s
                                    .phase
                                    .slice_mut(self.s.gdn_out.at(r * w.value))
                                    .range(0..n as usize * w.value),
                                0,
                            )?;
                            continue;
                        }
                        self.use_scratch();
                        let state = s.linear[il].as_mut().ok_or("linear layer without state")?;
                        let mut pass = self.pass(stream, r);
                        pass.linear(
                            l,
                            state,
                            &mut s.rollback,
                            il,
                            n,
                            item.speculative,
                            true,
                            false,
                        )?;
                    }
                }
                Block::Attention(a) => {
                    for (i, item) in items.iter_mut().enumerate() {
                        let s = &mut *item.session;
                        let n = item.tokens.len() as u32;
                        let r = b.offsets[i] as usize;
                        if s.check_cancelled() {
                            stream.fill_bytes(
                                self.s
                                    .phase
                                    .slice_mut(self.s.ctx.at(r * w.q))
                                    .range(0..n as usize * w.q),
                                0,
                            )?;
                            continue;
                        }
                        let pool = if b.sparse[i] && b.complete[i] > s.blocks {
                            b.complete[i] - s.blocks
                        } else {
                            0
                        };
                        let (max_context, index_capacity, start_pos) =
                            (s.max_context, s.index_capacity, s.position);
                        let Session {
                            attention,
                            control,
                            vision,
                            ..
                        } = s;
                        let cache = attention[il]
                            .as_mut()
                            .ok_or("attention layer without caches")?;
                        let position = control.position(n).ok_or("position not uploaded")?;
                        let attention_control = AttentionControl {
                            position: position.view(),
                            first_block: control.blocks().view(),
                            start_pos,
                            pool_grid: pool,
                            max_context,
                            index_capacity,
                            sparse: b.sparse[i],
                            rope: vision.rope(),
                        };
                        self.use_scratch();
                        let mut pass = self.pass(stream, r);
                        pass.attention(a, cache, &attention_control, n, false, true, false)?;
                    }
                }
            }
            self.use_scratch();
            let mut pass = self.pass(stream, 0);
            match &layer.block {
                Block::Linear(l) => pass.ctx.dense_batch(
                    &l.out,
                    pass.s.phase.slice(pass.s.gdn_out.at(0)),
                    at_mut(&mut pass.s.block_out, 0),
                    rows,
                )?,
                Block::Attention(a) => pass.ctx.dense_batch(
                    &a.out,
                    pass.s.phase.slice(pass.s.ctx.at(0)),
                    at_mut(&mut pass.s.block_out, 0),
                    rows,
                )?,
            }
            pass.combine_batch(Some(&layer.hc_ffn.norm), rows)?;
            pass.hc_mix_batch(&layer.hc_ffn, true, true, rows)?;
            pass.moe_batch(&layer.moe, rows)?;
            let next_norm = if il + 1 < c.num_layers as usize {
                if c.is_ple_layer(il as u32 + 1) {
                    None
                } else {
                    Some(&model.layers[il + 1].hc_attn.norm)
                }
            } else {
                Some(&model.hc_head.norm)
            };
            self.use_scratch();
            self.pass(stream, 0).combine_batch(next_norm, rows)?;
            normed = next_norm.is_some();
        }
        let hc = w.hc;
        for (i, item) in items.iter_mut().enumerate() {
            let s = &mut *item.session;
            if s.cancelled() || !s.mtp_enabled {
                continue;
            }
            let n = item.tokens.len();
            let r = b.offsets[i] as usize;
            let mtp = s.mtp.as_mut().ok_or("no MTP block loaded")?;
            stream.copy(
                self.s.res.slice(r * hc..(r + n) * hc),
                mtp.target_hidden.slice_mut(0..n * hc),
            )?;
        }
        self.use_scratch();
        let vocab = c.vocab_size as usize;
        let mut logits = self.batch_logits.take().ok_or("batch logits missing")?;
        let mut pass = self.pass(stream, 0);
        let result = pass
            .hc_mix_batch(&model.hc_head, false, false, rows)
            .and_then(|()| {
                pass.ctx.dense_batch(
                    &model.output,
                    at(&pass.s.mixed, 0),
                    logits.slice_mut(0..rows as usize * vocab),
                    rows,
                )
            });
        self.batch_logits = Some(logits);
        result
    }

    /// The Gated DeltaNet of one linear layer for every active session in
    /// one launch per stage (`GatedDeltaNetBatch`).
    fn gdn_batch(
        &mut self,
        stream: &Stream,
        items: &mut [BatchItem<'_>],
        b: &BatchBody,
        l: &Linear,
        il: usize,
    ) -> Result<()> {
        let c = self.model.config.clone();
        let w = self.widths;
        let (channels, value) = (w.channels, w.value);
        let mut active = Vec::with_capacity(items.len());
        for (i, item) in items.iter_mut().enumerate() {
            let n = item.tokens.len();
            let r = b.offsets[i] as usize;
            if item.session.check_cancelled() {
                stream.fill_bytes(
                    self.s
                        .phase
                        .slice_mut(self.s.gdn_out.at(r * value))
                        .range(0..n * value),
                    0,
                )?;
            } else {
                active.push(i);
            }
        }
        if active.is_empty() {
            return Ok(());
        }
        let n_tokens: Vec<u32> = active
            .iter()
            .map(|&i| items[i].tokens.len() as u32)
            .collect();
        let starts: Vec<usize> = active.iter().map(|&i| b.offsets[i] as usize).collect();
        let span = |width: usize| -> Vec<(usize, usize)> {
            starts
                .iter()
                .zip(&n_tokens)
                .map(|(&r, &n)| (r * width, n as usize * width))
                .collect()
        };
        // The stacked projection interleaves [qkv | z] per row; separate
        // projections keep z behind the qkv rows of the whole batch.
        let (stride, qkv_stride, z_stride, z_at, z_width) = match &l.ssm_in {
            Some(ssm_in) => (w.qkvz, ssm_in.rows, ssm_in.rows, channels, w.qkvz),
            None => (
                channels,
                channels as u32,
                value as u32,
                w.max_batch * channels,
                value,
            ),
        };
        let ([qkvz, alpha_beta], [conv_scratch, qn, kn, raw, out]) = self.s.phase.views(
            [self.s.qkvz.at(0), self.s.alpha_beta.at(0)],
            [
                self.s.conv_scratch.at(0),
                self.s.qn.at(0),
                self.s.kn.at(0),
                self.s.gdn_raw.at(0),
                self.s.gdn_out.at(0),
            ],
        );
        let qkv: Vec<DeviceSlice<'_, f32>> = starts
            .iter()
            .zip(&n_tokens)
            .map(|(&r, &n)| {
                let at = r * stride;
                qkvz.range(at..at + (n as usize - 1) * qkv_stride as usize + channels)
            })
            .collect();
        let z: Vec<DeviceSlice<'_, f32>> = starts
            .iter()
            .zip(&n_tokens)
            .map(|(&r, &n)| {
                let at = z_at + r * z_width;
                qkvz.range(at..at + (n as usize - 1) * z_stride as usize + value)
            })
            .collect();
        let alpha_beta: Vec<DeviceSlice<'_, f32>> = span(w.alpha_beta)
            .into_iter()
            .map(|(at, len)| alpha_beta.range(at..at + len))
            .collect();
        let mut conv_scratch = split_rows(conv_scratch, &span(channels));
        let mut qn = split_rows(qn, &span(w.key));
        let mut kn = split_rows(kn, &span(w.key));
        let mut raw = split_rows(raw, &span(value));
        let mut out = split_rows(out, &span(value));
        let mut conv_state = Vec::with_capacity(active.len());
        let mut state = Vec::with_capacity(active.len());
        let mut state_rows: [Vec<Option<DeviceSliceMut<'_, f32>>>; 7] = Default::default();
        let mut conv_rows: [Vec<Option<DeviceSliceMut<'_, f32>>>; 7] = Default::default();
        let sessions = items
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| active.contains(i))
            .map(|(_, item)| &mut *item.session);
        for (slot, s) in sessions.enumerate() {
            let Session {
                linear,
                rollback,
                spec_tokens,
                ..
            } = s;
            let layer = linear[il].as_mut().ok_or("linear layer without state")?;
            conv_state.push(layer.conv_state.as_mut_slice());
            state.push(layer.state.as_mut_slice());
            // A verification forward keeps rows for its proper prefixes.
            let saved = if *spec_tokens != 0 {
                n_tokens[slot] as usize - 1
            } else {
                0
            };
            let mut rows = rollback.iter_mut();
            for r in 0..7 {
                match rows.next().filter(|_| r < saved) {
                    Some(row) => {
                        let (conv_at, state_at) =
                            row.layers[il].ok_or("rollback row of a linear layer")?;
                        let len = crate::state::gdn_rollback_row_floats(r as u32, &c);
                        let (conv, st) = row.buffer.as_mut_slice().split_at(state_at);
                        conv_rows[r].push(Some(conv.range(conv_at..state_at)));
                        state_rows[r].push(Some(st.range(0..len)));
                    }
                    None => {
                        conv_rows[r].push(None);
                        state_rows[r].push(None);
                    }
                }
            }
        }
        let [s0, s1, s2, s3, s4, s5, s6] = &mut state_rows;
        let [c0, c1, c2, c3, c4, c5, c6] = &mut conv_rows;
        // This layer's slot: the previous batch synchronised the stream, so
        // no queued kernel still reads it.
        let slot = self
            .batch_gdn
            .as_mut()
            .ok_or("batched GatedDeltaNet descriptors missing")?
            .slice_mut(il * GDN_SLOT_BYTES..(il + 1) * GDN_SLOT_BYTES);
        launched(
            k::gated_delta_net_batch(
                stream,
                slot,
                &qkv,
                qkv_stride,
                &z,
                z_stride,
                &alpha_beta,
                l.conv1d.view(),
                l.a.view(),
                l.dt.view(),
                l.norm.view(),
                &mut conv_state,
                &mut conv_scratch,
                &mut qn,
                &mut kn,
                &mut raw,
                &mut state,
                &mut out,
                s0,
                s1,
                s2,
                s3,
                s4,
                s5,
                s6,
                c0,
                c1,
                c2,
                c3,
                c4,
                c5,
                c6,
                &n_tokens,
                n_tokens.len() as u32,
                c.ssm_num_k_heads,
                c.ssm_num_v_heads,
                c.rms_eps,
            ),
            "batched GatedDeltaNet launch failed",
        )
    }

    /// Everything a draft batch enqueues after its host preparation.
    fn mtp_batch_body(
        &mut self,
        stream: &Stream,
        items: &mut [MtpBatchItem<'_>],
        offsets: &[u32; BATCH_SESSIONS],
        final_rows: &[u32; BATCH_SESSIONS],
        rows: u32,
    ) -> Result<()> {
        if !any_active(items.iter_mut().map(|i| &mut *i.session)) {
            return Ok(());
        }
        let c = self.model.config.clone();
        let model = std::sync::Arc::clone(&self.model);
        let l = model.mtp.as_ref().ok_or("no MTP block loaded")?;
        let nextn = l
            .nextn
            .as_ref()
            .ok_or("MTP block without its projections")?;
        let Block::Attention(a) = &l.block else {
            return Err("the MTP block has no attention".into());
        };
        let qkv = a.qkv.as_ref().ok_or("missing attention projection")?;
        let w = self.widths;
        let hc = w.hc;
        let max_speculative = self.options.max_speculative;
        // Only K/V for every session (a copy or an ordinary target step):
        // the packed projection's K/V rows keep its arithmetic.
        let kv = if items.iter().all(|i| i.kv_only) {
            Some(
                qkv.last_rows(2 * c.attention_kv_dim())
                    .ok_or("batched MTP cache projection failed")?,
            )
        } else {
            None
        };
        self.reset_route();
        stream.upload_indices_async(&mut self.host.tokens, rows as usize, &mut self.s.tokens)?;
        let token_embd = &model.token_embd;
        k::embed_tokens(
            stream,
            token_embd.view(),
            self.s.tokens.slice(0..rows as usize),
            at_mut(&mut self.s.mixed, 0),
            rows,
            c.hidden_size,
            1,
            token_embd.rows,
        )
        .map_err(launch_error("draft embedding"))?;
        k::rms_norm_rows_in_place(
            stream,
            nextn.enorm.view(),
            at_mut(&mut self.s.mixed, 0),
            rows,
            c.hidden_size,
            1,
            c.rms_eps,
        )
        .map_err(launch_error("draft embedding norm"))?;
        for (i, item) in items.iter_mut().enumerate() {
            let n = item.tokens.len() as u32;
            let s = &mut *item.session;
            let (position, blocks) = s.mtp.as_ref().map_or((0, 0), |m| (m.position, m.blocks));
            s.control.set_mtp_position(stream, n, position)?;
            s.control
                .set_hidden_row(stream, n, max_speculative, item.hidden_row)?;
            s.control.set_mtp_blocks(stream, blocks)?;
            let mtp = s.mtp.as_ref().ok_or("no MTP block loaded")?;
            let row = s
                .control
                .hidden_row(n, max_speculative)
                .ok_or("hidden row not uploaded")?;
            let base = mtp.target_hidden.slice(0..max_speculative as usize * hc);
            let alt = if item.hidden_row < 0 {
                mtp.h.slice(0..hc)
            } else {
                base.range(0..n as usize * hc)
            };
            k::mtp_hidden(
                stream,
                base,
                alt,
                row.view(),
                at_mut(&mut self.s.xn, offsets[i] as usize * hc),
                n,
                c.hc_dim(),
                max_speculative,
            )
            .map_err(launch_error("draft hidden input"))?;
        }
        k::rms_norm_rows_in_place(
            stream,
            nextn.hnorm.view(),
            at_mut(&mut self.s.xn, 0),
            rows,
            c.hc_dim(),
            1,
            c.rms_eps,
        )
        .map_err(launch_error("draft hidden norm"))?;
        self.use_scratch();
        {
            let mut pass = self.pass(stream, 0);
            // The persistent draft K/V uses the same projections for every
            // prompt split, catch-up length and number of sessions.
            pass.ctx.dense_batch(
                &nextn.fc_embedding,
                at(&pass.s.mixed, 0),
                pass.s.phase.slice_mut(pass.s.qkvz.at(0)),
                rows,
            )?;
            pass.ctx.dense_batch(
                &nextn.fc_hidden,
                at(&pass.s.xn, 0),
                at_mut(&mut pass.s.res, 0),
                rows * c.hc_count,
            )?;
            k::mtp_add_embedding(
                stream,
                pass.s.phase.slice(pass.s.qkvz.at(0)),
                at_mut(&mut pass.s.res, 0),
                rows,
                c.hidden_size,
                c.hc_count,
            )
            .map_err(launch_error("draft embedding sum"))?;
            pass.hc_mix_batch(&l.hc_attn, false, kv.is_none(), rows)?;
            match &kv {
                Some(kv) => {
                    pass.ctx.dense_batch(
                        kv,
                        at(&pass.s.mixed, 0),
                        pass.s.phase.slice_mut(pass.s.qg.at(0)),
                        rows,
                    )?;
                    pass.ctx.dense_batch(
                        &a.indexer_k,
                        at(&pass.s.mixed, 0),
                        pass.s.phase.slice_mut(pass.s.ik.at(0)),
                        rows,
                    )?;
                }
                None => pass.ctx.dense_batch(
                    qkv,
                    at(&pass.s.mixed, 0),
                    pass.s.phase.slice_mut(pass.s.qg.at(0)),
                    rows,
                )?,
            }
        }
        if let Some(kv) = &kv {
            // Stop before queries, attention, the output projection and the
            // FFN, whose residual would be discarded.
            let eps = c.rms_eps;
            for (i, item) in items.iter_mut().enumerate() {
                let n = item.tokens.len() as u32;
                let r = offsets[i] as usize;
                let s = &mut *item.session;
                if s.check_cancelled() {
                    continue;
                }
                s.mutation_epoch += 1;
                let (max_context, index_capacity) = (s.max_context, s.index_capacity);
                let Session {
                    mtp,
                    control,
                    vision,
                    ..
                } = s;
                let mtp = mtp.as_mut().ok_or("no MTP block loaded")?;
                mtp.residual_valid = false;
                let position = control
                    .mtp_position(n)
                    .ok_or("draft position not uploaded")?;
                let ([qg], [q, gate]) = self.s.phase.views(
                    [self.s.qg.at(r * kv.rows as usize)],
                    [self.s.q.part(0..0), self.s.attn_gate.part(0..0)],
                );
                let prepared = k::prepare_attention(
                    stream,
                    qg,
                    kv.rows,
                    a.k_norm.view(),
                    a.k_norm.view(),
                    q,
                    gate,
                    mtp.attention.k_cache.as_mut_slice(),
                    mtp.attention.v_cache.as_mut_slice(),
                    n,
                    0,
                    c.num_kv_heads,
                    c.head_dim,
                    c.rotary_dim,
                    position.view(),
                    c.rope_theta,
                    eps,
                    vision.rope(),
                    false,
                    self.yarn.enabled,
                    self.yarn.low,
                    self.yarn.high,
                    self.yarn.inv_factor,
                    self.yarn.mscale,
                    max_context,
                )
                .map_err(launch_error("batched MTP cache projection"))?;
                if prepared != gufo_hip::Launch::Launched {
                    return Err("batched MTP cache projection failed".into());
                }
                let at_ik = r * c.indexer_head_dim as usize;
                k::store_rows(
                    stream,
                    self.s.phase.slice(self.s.ik.at(at_ik)),
                    mtp.attention.index_k.as_mut_slice(),
                    n,
                    c.indexer_head_dim,
                    position.view(),
                    index_capacity,
                    max_context,
                )
                .map_err(launch_error("indexer key store"))?;
                if mtp.position + n > c.indexer_top_k {
                    let pool = (mtp.position + n) / c.compress_ratio - mtp.blocks;
                    k::pool_indexer_blocks(
                        stream,
                        mtp.attention.index_k.as_slice(),
                        a.indexer_k_norm.view(),
                        mtp.attention.block_k.as_mut_slice(),
                        control.mtp_blocks().view(),
                        position.view(),
                        n,
                        pool,
                        c.compress_ratio,
                        c.indexer_head_dim,
                        c.rotary_dim,
                        c.rope_theta,
                        eps,
                        index_capacity,
                        vision.rope(),
                        self.yarn.enabled,
                        self.yarn.low,
                        self.yarn.high,
                        self.yarn.inv_factor,
                        self.yarn.mscale,
                        max_context,
                    )
                    .map_err(launch_error("indexer pooling"))?;
                }
            }
            return Ok(());
        }
        for (i, item) in items.iter_mut().enumerate() {
            let n = item.tokens.len() as u32;
            let r = offsets[i] as usize;
            let s = &mut *item.session;
            if s.check_cancelled() {
                stream.fill_bytes(
                    self.s
                        .phase
                        .slice_mut(self.s.ctx.at(r * w.q))
                        .range(0..n as usize * w.q),
                    0,
                )?;
                continue;
            }
            s.mutation_epoch += 1;
            let (max_context, index_capacity) = (s.max_context, s.index_capacity);
            let Session {
                mtp,
                control,
                vision,
                ..
            } = s;
            let mtp = mtp.as_mut().ok_or("no MTP block loaded")?;
            mtp.residual_valid = false;
            let sparse = mtp.position + n > c.indexer_top_k;
            let complete = (mtp.position + n) / c.compress_ratio;
            let pool = if sparse {
                complete.saturating_sub(mtp.blocks)
            } else {
                0
            };
            let position = control
                .mtp_position(n)
                .ok_or("draft position not uploaded")?;
            let attention_control = AttentionControl {
                position: position.view(),
                first_block: control.mtp_blocks().view(),
                start_pos: mtp.position,
                pool_grid: pool,
                max_context,
                index_capacity,
                sparse,
                rope: vision.rope(),
            };
            self.use_scratch();
            let mut pass = self.pass(stream, r);
            pass.attention(
                a,
                &mut mtp.attention,
                &attention_control,
                n,
                false,
                true,
                false,
            )?;
        }
        // Attention kept every catch-up row; only each final residual feeds
        // the next proposal, so those rows move to the front before the
        // output projection and the FFN. A one-row tail in the eight-row
        // projection would round differently.
        let compact = rows > DECODE_ROWS && rows % DECODE_ROWS != 1;
        let parts = self.route.inject_parts as usize;
        let inject_width = c.hc_count as usize * parts;
        if compact {
            for (i, &from) in final_rows[..items.len()].iter().enumerate() {
                let from = from as usize;
                move_row(stream, self.s.res.as_mut_slice(), hc, from, i)?;
                move_row(stream, self.s.inject.as_mut_slice(), inject_width, from, i)?;
                let ([row], [compacted]) = self.s.phase.views(
                    [self.s.ctx.part(from * w.q..(from + 1) * w.q)],
                    [self.s.qg.part(i * w.q..(i + 1) * w.q)],
                );
                stream.copy(row, compacted)?;
            }
        }
        let tail_rows = if compact { items.len() as u32 } else { rows };
        self.use_scratch();
        {
            let mut pass = self.pass(stream, 0);
            let source = if compact {
                pass.s.phase.slice(pass.s.qg.at(0))
            } else {
                pass.s.phase.slice(pass.s.ctx.at(0))
            };
            pass.ctx
                .dense_batch(&a.out, source, at_mut(&mut pass.s.block_out, 0), tail_rows)?;
            pass.combine_batch(Some(&l.hc_ffn.norm), tail_rows)?;
            pass.hc_mix_batch(&l.hc_ffn, true, true, tail_rows)?;
        }
        if !any_active(items.iter_mut().map(|i| &mut *i.session)) {
            return Ok(());
        }
        {
            let mut pass = self.pass(stream, 0);
            pass.moe_batch(&l.moe, tail_rows)?;
            pass.combine_batch(None, tail_rows)?;
        }
        for (i, item) in items.iter_mut().enumerate() {
            let s = &mut *item.session;
            if s.check_cancelled() || item.kv_only {
                continue;
            }
            let row = if compact { i } else { final_rows[i] as usize };
            let mtp = s.mtp.as_mut().ok_or("no MTP block loaded")?;
            stream.copy(
                self.s.res.slice(row * hc..(row + 1) * hc),
                mtp.h.as_mut_slice(),
            )?;
        }
        Ok(())
    }
}
