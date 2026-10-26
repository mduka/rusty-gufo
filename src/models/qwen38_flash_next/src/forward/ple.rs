//! The per-layer embedding (PLE) injection, its upload and the draft
//! logits over the PLE table.

use super::*;

/// The logits of a session's draft table rows for the quantized row `q`,
/// empty slots at -infinity (the table half of `DraftLogits`).
pub(crate) fn draft_table_logits(
    ctx: &mut Ctx<'_>,
    table: &DraftTable,
    q: &Q8Input<'_>,
    vocab: u32,
    mut out: DeviceSliceMut<'_, f32>,
) -> Result<()> {
    let slot = q.slot.ok_or("draft head without Q8_1 rows")?;
    let bytes = k::q8_1_bytes(1, q.k as usize);
    launched(
        k::mmq_q8_dense_vec(
            ctx.stream,
            QuantSlice::new(table.rows.as_slice(), GgmlType::Q8_0),
            ctx.stage.x_q8[slot].slice(0..bytes),
            out.reborrow(),
            DRAFT_EXTRA_ROWS,
            1,
            q.k,
        ),
        "draft table GEMV failed",
    )?;
    k::mask_draft_logits(ctx.stream, out, table.ids.view(), DRAFT_EXTRA_ROWS, vocab)
        .map_err(launch_error("draft table mask"))
}

/// A mixer's F32 inject weights, which its combine can apply
/// (`InjectWeights`).
pub(crate) fn inject_weights(m: &Mixer) -> Option<DeviceSlice<'_, f32>> {
    m.inject.as_ref().and_then(Matrix::floats)
}

impl Pass<'_> {
    /// The PLE injection into the residual (`Ple`). `embeddings_ready`: the
    /// n-gram rows are in `ple_emb` already; otherwise this waits for the
    /// disk read and uploads them.
    pub fn ple(
        &mut self,
        p: &Ple,
        history: &mut DeviceBuffer<f32>,
        rows: &mut [RollbackRow],
        n: u32,
        speculative: bool,
        embeddings_ready: bool,
    ) -> Result<()> {
        let c = self.c().clone();
        let (r, w) = (self.row, self.w);
        let stream = self.ctx.stream;
        if !embeddings_ready {
            self.wait_ple()?;
            self.upload_ple(n)?;
            // The table is free again: read the named successor's rows
            // during this pass.
            if let Some(ngram) = self.ngram {
                self.prefetch.start(ngram);
            }
        }
        let hc_dim = c.hc_dim();
        let (streams, eps) = (c.hc_count, c.rms_eps);
        let emb = at(&self.s.ple_emb, r * w.ple);
        let q = self.ctx.quantize(emb, n, c.ple_embedding_dim())?;
        self.ctx.dense_q(
            &p.key,
            &q,
            self.s.phase.slice_mut(self.s.hc_gate.at(r * w.hc)),
        )?;
        self.ctx
            .dense_q(&p.value, &q, at_mut(&mut self.s.block_out, r * w.hidden))?;
        let norm = |e| launch_error("PLE norm")(e);
        k::rms_norm_rows_in_place(
            stream,
            p.norm_key.view(),
            self.s.phase.slice_mut(self.s.hc_gate.at(r * w.hc)),
            n,
            hc_dim,
            streams,
            eps,
        )
        .map_err(norm)?;
        k::rms_norm_rows(
            stream,
            at(&self.s.res, r * w.hc),
            p.norm_query.view(),
            at_mut(&mut self.s.xn, r * w.hc),
            n,
            hc_dim,
            streams,
            eps,
        )
        .map_err(norm)?;
        let ([key], [gated]) = self
            .s
            .phase
            .views([self.s.hc_gate.at(r * w.hc)], [self.s.qkvz.at(r * w.qkvz)]);
        k::ple_gate(
            stream,
            key,
            at(&self.s.xn, r * w.hc),
            at(&self.s.block_out, r * w.hidden),
            gated,
            n,
            c.hidden_size,
            streams,
        )
        .map_err(launch_error("PLE gate"))?;
        let ([gated], [conv]) = self
            .s
            .phase
            .views([self.s.qkvz.at(r * w.qkvz)], [self.s.hc_gate.at(r * w.hc)]);
        k::rms_norm_rows(
            stream,
            gated,
            p.norm_conv.view(),
            conv,
            n,
            hc_dim,
            streams,
            eps,
        )
        .map_err(norm)?;
        // Rows for the proper prefixes of a speculative batch only.
        let saved = if speculative {
            n.saturating_sub(1) as usize
        } else {
            0
        };
        let mut snapshots: Vec<DeviceSliceMut<'_, f32>> = rows[..saved]
            .iter_mut()
            .map(|row| {
                let at = row.ple.unwrap_or(0);
                let len = row.buffer.len();
                row.buffer.slice_mut(at..len)
            })
            .collect();
        // The checkpoint's history, before the convolution advances it.
        if let Some(checkpoint) = &mut self.checkpoint {
            let tokens = checkpoint.tokens;
            let destination = checkpoint
                .storage
                .ple_mut()
                .ok_or("checkpoint storage without PLE history")?;
            k::history_prefix(
                stream,
                self.s.phase.slice(self.s.hc_gate.at(r * w.hc)),
                hc_dim,
                history.as_slice(),
                destination,
                tokens,
                hc_dim,
                c.ple_conv_history(),
            )
            .map_err(launch_error("checkpoint PLE history"))?;
        }
        launched(
            k::ple_conv(
                stream,
                self.s.phase.slice(self.s.hc_gate.at(r * w.hc)),
                p.conv1d.view(),
                history.as_mut_slice(),
                self.s.ple_history_scratch.as_mut_slice(),
                at_mut(&mut self.s.xn, r * w.hc),
                &mut snapshots,
                saved as u32,
                n,
                hc_dim,
                c.ple_conv_kernel,
                c.ple_ngram_size,
            ),
            "PLE convolution failed",
        )?;
        let count = n as usize * w.hc;
        k::ple_inject(
            stream,
            at_mut(&mut self.s.res, r * w.hc).range(0..count),
            self.s
                .phase
                .slice(self.s.qkvz.at(r * w.qkvz))
                .range(0..count),
            at(&self.s.xn, r * w.hc).range(0..count),
            count,
        )
        .map_err(launch_error("PLE inject"))
    }

    /// Small-head draft logits of `mixed` row `row` into `logits` row
    /// `row` (rows of the small head's width): the prefix rows, then the
    /// session's table rows (`DraftLogits`).
    pub fn draft_logits(
        &mut self,
        draft: &DraftHead,
        table: &DraftTable,
        row: usize,
    ) -> Result<()> {
        let vocab = self.c().vocab_size;
        let cols = draft.prefix.cols;
        let prefix = draft.prefix.rows as usize;
        let at_row = row * (prefix + DRAFT_EXTRA_ROWS as usize);
        let q = self
            .ctx
            .quantize(at(&self.s.mixed, row * cols as usize), 1, cols)?;
        self.ctx.dense_q(
            &draft.prefix,
            &q,
            self.s.logits.slice_mut(at_row..at_row + prefix),
        )?;
        let end = at_row + prefix + DRAFT_EXTRA_ROWS as usize;
        draft_table_logits(
            &mut self.ctx,
            table,
            &q,
            vocab,
            self.s.logits.slice_mut(at_row + prefix..end),
        )
    }

    /// Waits for the n-gram rows (`WaitPle`).
    pub fn wait_ple(&mut self) -> Result<()> {
        let pending = std::mem::replace(self.ple_pending, false);
        let ngram = self.ngram.ok_or("n-gram table is not open")?;
        let ple = self
            .host
            .ple
            .as_mut()
            .ok_or("n-gram rows without a PLE layer")?;
        let prefetch = &mut *self.prefetch;
        let ok = if pending && std::mem::take(&mut prefetch.claimed) {
            // Prefetched rows: still in the table, or complete beside it.
            if std::mem::take(&mut prefetch.pending) {
                ngram.wait_read(ple.host_mut()?)
            } else if std::mem::take(&mut prefetch.ready) {
                let count = prefetch.tokens.len() * ngram.row_dim();
                ple.host_mut()?[..count].copy_from_slice(&prefetch.rows[..count]);
                true
            } else {
                false
            }
        } else {
            pending && ngram.wait_read(ple.host_mut()?)
        };
        if !ok {
            return Err("n-gram table read failed".into());
        }
        Ok(())
    }

    /// Enqueues the upload of `n` rows of n-gram embeddings.
    pub fn upload_ple(&mut self, n: u32) -> Result<()> {
        let count = n as usize * self.w.ple;
        let at = self.row * self.w.ple;
        let ple = self
            .host
            .ple
            .as_mut()
            .ok_or("n-gram rows without a PLE layer")?;
        self.ctx
            .stream
            .upload_async_front(ple, count, self.s.ple_emb.slice_mut(at..at + count))
            .map_err(|e| Error::Other(format!("n-gram upload: {e}")))
    }
}
