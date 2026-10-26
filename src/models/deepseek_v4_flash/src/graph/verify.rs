//! Speculative verification (the "DSpark Speculative Verification" part of
//! `rocm_graph.cpp`): a drafted block runs through the batched layers at its
//! position, which write the persistent caches, so the compressor frontiers
//! and row counts are saved before it and, for its first rows, after each
//! row; a rejected tail rolls back to a prefix or to the saved frontier.

use gufo_deepseek_v4_flash_kernels::{Backend, BatchMode, Error, Input, Result, TensorView};

use super::alloc::{BatchWorkspace, lent, lent_pair, rows_mut, verify_streams};
use super::decode::{Target, cached, cached_mut, present};
use super::prefill::{AttnBatch, FfnLayer, Slots};
use super::*;

fn attn_state_bytes(r: u32) -> u64 {
    let coff: u64 = if r == 4 { 2 } else { 1 };
    coff * u64::from(N_HEAD_DIM) * coff * u64::from(r) * F32
}

fn index_state_bytes(r: u32) -> u64 {
    let coff: u64 = if r == 4 { 2 } else { 1 };
    coff * u64::from(N_INDEXER_HEAD_DIM) * coff * u64::from(r) * F32
}

fn rows(t: &impl Input, row0: u32, n: u32, row_bytes: u64) -> Result<TensorView<'_>> {
    t.view(u64::from(row0) * row_bytes, u64::from(n) * row_bytes)
        .ok_or(Error { op: "verify view" })
}

/// One request's block of a batched verification (`ds4_rocm_verify_item`).
pub struct VerifyItem<'a> {
    pub graph: &'a mut Graph,
    /// The checkpoint followed by the drafted block.
    pub tokens: &'a [i32],
    pub start: u32,
    pub n_tokens: u32,
    /// Rows the request decides on (the others pad a group).
    pub logical_n_tokens: u32,
    /// The target argmax after each row.
    pub row_tops: Vec<i32>,
    /// The logits after the accepted prefix, which greedy requests receive.
    pub frontier_logits: Option<&'a mut [f32]>,
    /// Every row's logits, which sampled requests decide on.
    pub row_logits: Option<&'a mut [f32]>,
    /// Greedy items: the token the request stops at, or -1. The frontier
    /// gather stops before an accepted stop token as the host commit does.
    pub stop_token: i32,
}

/// Greedy acceptance of a verified DSpark block (`ds4_dspark_greedy_accept`):
/// `tokens[0]` is the anchor and `row_tops[i]` the target argmax after
/// `tokens[i]`. Accepts while the argmax reproduces the next token and stops
/// before an accepted `stop_token` (-1 for none), so frontier row
/// `accepted - 1` proposes it as the next anchor. The host commit and the
/// frontier gather both use it and must agree.
pub(crate) fn greedy_accept(
    row_tops: &[i32],
    tokens: &[i32],
    n_tokens: u32,
    stop_token: i32,
) -> u32 {
    let mut accepted = 1u32;
    while accepted < n_tokens
        && row_tops[accepted as usize - 1] == tokens[accepted as usize]
        && tokens[accepted as usize] != stop_token
    {
        accepted += 1;
    }
    accepted
}

impl Graph {
    /// `rocm_graph_capture_prefix_attn_state`: the attention compressor
    /// frontier after a verification block's row `slot`; nothing outside
    /// verification.
    pub(crate) fn capture_prefix_attn_state(
        &mut self,
        t: &Target<'_>,
        il: u32,
        slot: u32,
    ) -> Result<()> {
        if !self.spec_capture_prefixes {
            return Ok(());
        }
        let r = ratio(il);
        if r == 0 {
            return Ok(());
        }
        let refuse = Error {
            op: "prefix capture",
        };
        let spec = self.spec.as_mut().ok_or(refuse)?;
        let frontier = spec.attn[il as usize].as_mut().ok_or(refuse)?;
        if slot >= SPEC_PREFIX_SLOTS {
            return Err(refuse);
        }
        let cache = &self.layers[il as usize];
        let bytes = attn_state_bytes(r);
        let offset = u64::from(slot) * bytes;
        spec.prefix_n_comp[slot as usize][il as usize] = cache.n_comp;
        t.b.copy(
            &mut frontier.prefix_kv,
            offset,
            cached(&cache.attn_state_kv)?,
            0,
            bytes,
        )?;
        t.b.copy(
            &mut frontier.prefix_score,
            offset,
            cached(&cache.attn_state_score)?,
            0,
            bytes,
        )
    }

    /// `rocm_graph_capture_prefix_index_state`: the indexer compressor
    /// frontier after row `slot`.
    pub(crate) fn capture_prefix_index_state(
        &mut self,
        t: &Target<'_>,
        il: u32,
        slot: u32,
    ) -> Result<()> {
        if !self.spec_capture_prefixes || ratio(il) != 4 {
            return Ok(());
        }
        let refuse = Error {
            op: "prefix capture",
        };
        let spec = self.spec.as_mut().ok_or(refuse)?;
        let frontier = spec.index[il as usize].as_mut().ok_or(refuse)?;
        if slot >= SPEC_PREFIX_SLOTS {
            return Err(refuse);
        }
        let cache = &self.layers[il as usize];
        let bytes = index_state_bytes(4);
        let offset = u64::from(slot) * bytes;
        spec.prefix_n_index_comp[slot as usize][il as usize] = cache.n_index_comp;
        t.b.copy(
            &mut frontier.prefix_kv,
            offset,
            cached(&cache.index_state_kv)?,
            0,
            bytes,
        )?;
        t.b.copy(
            &mut frontier.prefix_score,
            offset,
            cached(&cache.index_state_score)?,
            0,
            bytes,
        )
    }

    /// `rocm_graph_spec_copy_frontier`: every compressor frontier into the
    /// saved copy, or back.
    fn spec_copy_frontier(&mut self, b: &Backend, save: bool) -> Result<()> {
        let refuse = Error {
            op: "spec frontier",
        };
        let spec = self.spec.as_mut().ok_or(refuse)?;
        for (il, cache) in (0..N_LAYER).zip(self.layers.iter_mut()) {
            let r = ratio(il);
            if r == 0 {
                continue;
            }
            let saved = spec.attn[il as usize].as_mut().ok_or(refuse)?;
            let (live_kv, live_score) = (
                cached_mut(&mut cache.attn_state_kv)?,
                cached_mut(&mut cache.attn_state_score)?,
            );
            let bytes = attn_state_bytes(r);
            if save {
                b.copy(&mut saved.saved_kv, 0, live_kv, 0, bytes)?;
                b.copy(&mut saved.saved_score, 0, live_score, 0, bytes)?;
            } else {
                b.copy(live_kv, 0, &saved.saved_kv, 0, bytes)?;
                b.copy(live_score, 0, &saved.saved_score, 0, bytes)?;
            }
            if r != 4 {
                continue;
            }
            let saved = spec.index[il as usize].as_mut().ok_or(refuse)?;
            let (live_kv, live_score) = (
                cached_mut(&mut cache.index_state_kv)?,
                cached_mut(&mut cache.index_state_score)?,
            );
            let bytes = index_state_bytes(r);
            if save {
                b.copy(&mut saved.saved_kv, 0, live_kv, 0, bytes)?;
                b.copy(&mut saved.saved_score, 0, live_score, 0, bytes)?;
            } else {
                b.copy(live_kv, 0, &saved.saved_kv, 0, bytes)?;
                b.copy(live_score, 0, &saved.saved_score, 0, bytes)?;
            }
        }
        Ok(())
    }

    /// `rocm_graph_spec_frontier_save`: the row counts and frontiers before
    /// a verification block.
    pub(crate) fn spec_frontier_save(&mut self, b: &Backend) -> Result<()> {
        let counts: Vec<(u32, u32)> = self
            .layers
            .iter()
            .map(|c| (c.n_comp, c.n_index_comp))
            .collect();
        let spec = self.spec.as_mut().ok_or(Error {
            op: "spec frontier",
        })?;
        if spec.rows_cap == 0 {
            return Err(Error {
                op: "spec frontier",
            });
        }
        for (il, (comp, index)) in counts.into_iter().enumerate() {
            spec.saved_n_comp[il] = comp;
            spec.saved_n_index_comp[il] = index;
        }
        let saved = b.commands(|| self.spec_copy_frontier(b, true));
        if let Some(spec) = &mut self.spec {
            spec.frontier_valid = saved.is_ok();
        }
        saved
    }

    /// `rocm_graph_spec_frontier_restore`: back to the state before the
    /// block.
    pub(crate) fn spec_frontier_restore(&mut self, b: &Backend) -> Result<()> {
        let spec = self.spec.as_ref().ok_or(Error {
            op: "spec frontier",
        })?;
        if !spec.frontier_valid {
            return Err(Error {
                op: "spec frontier",
            });
        }
        for (il, cache) in self.layers.iter_mut().enumerate() {
            cache.n_comp = spec.saved_n_comp[il];
            cache.n_index_comp = spec.saved_n_index_comp[il];
        }
        b.commands(|| self.spec_copy_frontier(b, false))
    }

    /// `rocm_graph_spec_frontier_commit_prefix`: the state after the
    /// block's first `prefix_len` rows (at most `SPEC_PREFIX_SLOTS`).
    pub(crate) fn spec_frontier_commit_prefix(
        &mut self,
        b: &Backend,
        prefix_len: u32,
    ) -> Result<()> {
        let refuse = Error { op: "spec prefix" };
        if prefix_len == 0 || prefix_len > SPEC_PREFIX_SLOTS {
            return Err(refuse);
        }
        let slot = prefix_len - 1;
        let spec = self.spec.as_ref().ok_or(refuse)?;
        // One wait covers the copies, as the block's own bracket would.
        b.commands(|| {
            for (il, cache) in (0..N_LAYER).zip(self.layers.iter_mut()) {
                let r = ratio(il);
                if r == 0 {
                    continue;
                }
                let frontier = spec.attn[il as usize].as_ref().ok_or(refuse)?;
                let bytes = attn_state_bytes(r);
                cache.n_comp = spec.prefix_n_comp[slot as usize][il as usize];
                b.copy(
                    cached_mut(&mut cache.attn_state_kv)?,
                    0,
                    &frontier.prefix_kv,
                    u64::from(slot) * bytes,
                    bytes,
                )?;
                b.copy(
                    cached_mut(&mut cache.attn_state_score)?,
                    0,
                    &frontier.prefix_score,
                    u64::from(slot) * bytes,
                    bytes,
                )?;
                if r != 4 {
                    continue;
                }
                let frontier = spec.index[il as usize].as_ref().ok_or(refuse)?;
                let bytes = index_state_bytes(r);
                cache.n_index_comp = spec.prefix_n_index_comp[slot as usize][il as usize];
                b.copy(
                    cached_mut(&mut cache.index_state_kv)?,
                    0,
                    &frontier.prefix_kv,
                    u64::from(slot) * bytes,
                    bytes,
                )?;
                b.copy(
                    cached_mut(&mut cache.index_state_score)?,
                    0,
                    &frontier.prefix_score,
                    u64::from(slot) * bytes,
                    bytes,
                )?;
            }
            Ok(())
        })
    }

    /// `rocm_graph_read_spec_logits_row`.
    pub(crate) fn read_spec_logits_row(
        &self,
        b: &Backend,
        row: u32,
        logits: &mut [f32],
    ) -> Result<()> {
        let spec = self.spec.as_ref().ok_or(Error { op: "spec logits" })?;
        if row >= spec.rows_cap {
            return Err(Error { op: "spec logits" });
        }
        let vocab = N_VOCAB as usize;
        let out = logits.get_mut(..vocab).ok_or(Error { op: "spec logits" })?;
        b.read_f32(&spec.logits, u64::from(row) * vocab as u64 * F32, out)
    }

    /// `rocm_graph_verify_suffix`: `tokens[start .. start + n_tokens]` in
    /// one batched pass through the graph's workspace `ws`, which writes the
    /// caches; the target argmax after each row. The caller rolls back what
    /// it rejects.
    pub(crate) fn verify_suffix(
        &mut self,
        t: &Target<'_>,
        ws: &mut BatchWorkspace,
        tokens: &[i32],
        start: u32,
        n_tokens: u32,
    ) -> Result<Vec<i32>> {
        let b = t.b;
        let rows_cap = self.spec.as_ref().map_or(0, |s| s.rows_cap);
        if n_tokens == 0
            || n_tokens > self.prefill_cap
            || n_tokens > rows_cap
            || start as usize > tokens.len()
            || n_tokens as usize > tokens.len() - start as usize
        {
            return Err(Error {
                op: "verify suffix",
            });
        }
        self.check_workspace(ws)?;
        let block: Vec<u8> = tokens[start as usize..(start + n_tokens) as usize]
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect();
        b.write(&mut ws.prefill_tokens, 0, &block)?;
        let embd = &t.weights.token_embd;
        b.embed_tokens_hc(
            &mut ws.hc[self.batch_cur_index],
            &ws.prefill_tokens,
            w(embd),
            embd.dim[1] as u32,
            n_tokens,
            N_EMBD,
            N_HC,
        )?;
        let vocab = t.weights.output.dim[1] as u32;
        // Verification-block kernel selection is scoped to this pass.
        let saved_capture = self.spec_capture_prefixes;
        self.spec_capture_prefixes = n_tokens > 1 && n_tokens <= SPEC_PREFIX_SLOTS + 2;
        b.set_batch_mode(BatchMode::Verifier);
        let result = (|| {
            b.begin_commands()?;
            for il in 0..N_LAYER {
                self.encode_layer_batch(t, ws, il, start, n_tokens)?;
            }
            self.encode_output_head_batch(t, ws, n_tokens)?;
            let spec = self.spec.as_mut().ok_or(Error {
                op: "verify suffix",
            })?;
            b.spec_row_argmax(&mut spec.row_tops, &spec.logits, vocab, n_tokens)?;
            b.end_commands()
        })();
        if result.is_err() {
            let _ = b.synchronize();
        }
        b.set_batch_mode(BatchMode::Off);
        self.spec_capture_prefixes = saved_capture;
        result?;
        let spec = self.spec.as_ref().ok_or(Error {
            op: "verify suffix",
        })?;
        let mut bytes = vec![0u8; n_tokens as usize * 4];
        b.read(&spec.row_tops, 0, &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| i32::from_le_bytes(c))
            .collect())
    }
}

/// `rocm_graph_distribute_verify_results`: each request's row argmax, and
/// the logits it decides on: every row for sampled requests, the row after
/// the accepted prefix for greedy ones.
fn distribute_verify_results(
    t: &Target<'_>,
    items: &mut [VerifyItem<'_>],
    offsets: &[u32],
    flat_tops: &[i32],
) -> Result<()> {
    let b = t.b;
    let vocab = t.weights.output.dim[1];
    let row_bytes = vocab * F32;
    let mut gather = true;
    for (index, item) in items.iter_mut().enumerate() {
        let (from, to) = (offsets[index] as usize, offsets[index + 1] as usize);
        item.row_tops = flat_tops[from..to].to_vec();
        gather = gather
            && item.frontier_logits.is_some()
            && item.logical_n_tokens != 0
            && item.logical_n_tokens <= item.n_tokens;
    }
    let mut spec = items[0].graph.spec.take().ok_or(Error {
        op: "verify results",
    })?;
    let result = (|| {
        if !gather {
            return b.commands(|| {
                for (index, item) in items.iter_mut().enumerate().skip(1) {
                    let target = item.graph.spec.as_mut().ok_or(Error {
                        op: "verify results",
                    })?;
                    b.copy(
                        &mut target.logits,
                        0,
                        &spec.logits,
                        u64::from(offsets[index]) * row_bytes,
                        u64::from(item.n_tokens) * row_bytes,
                    )?;
                }
                Ok(())
            });
        }
        // Sampled requests decide on the CPU and take every verified row.
        for (index, item) in items.iter_mut().enumerate() {
            if let Some(rows) = item.row_logits.as_deref_mut() {
                let n = item.n_tokens as usize * vocab as usize;
                let out = rows.get_mut(..n).ok_or(Error {
                    op: "verify results",
                })?;
                b.read_f32(&spec.logits, u64::from(offsets[index]) * row_bytes, out)?;
            }
        }
        let gathered = (|| {
            b.begin_commands()?;
            for (index, item) in items.iter().enumerate() {
                if item.row_logits.is_some() {
                    continue;
                }
                let accepted = greedy_accept(
                    &item.row_tops,
                    &item.tokens[item.start as usize..],
                    item.logical_n_tokens,
                    item.stop_token,
                );
                let source = offsets[index] + accepted - 1;
                b.copy(
                    &mut spec.frontier_logits,
                    index as u64 * row_bytes,
                    &spec.logits,
                    u64::from(source) * row_bytes,
                    row_bytes,
                )?;
            }
            b.end_commands()
        })();
        if gathered.is_err() {
            let _ = b.synchronize();
            return gathered;
        }
        let mut frontier = vec![0f32; items.len() * vocab as usize];
        b.read_f32(&spec.frontier_logits, 0, &mut frontier)?;
        for (index, item) in items.iter_mut().enumerate() {
            if item.row_logits.is_some() {
                continue;
            }
            if let Some(out) = item.frontier_logits.as_deref_mut() {
                let n = vocab as usize;
                out.get_mut(..n)
                    .ok_or(Error {
                        op: "verify results",
                    })?
                    .copy_from_slice(&frontier[index * n..(index + 1) * n]);
            }
        }
        Ok(())
    })();
    items[0].graph.spec = Some(spec);
    result
}

/// `rocm_graph_verify_sessions_batch`: one block per request. Attention
/// stays per session, on its own caches; the FFN and output head stream
/// their weights once over the flattened rows, one router group per
/// request. Pairs of short blocks also share the dense attention front.
/// The first request's workspace of `workspaces` holds the rows.
pub fn verify_sessions_batch(
    t: &Target<'_>,
    workspaces: &mut [BatchWorkspace],
    items: &mut [VerifyItem<'_>],
) -> Result<()> {
    let b = t.b;
    let count = items.len();
    let refuse = Error { op: "verify batch" };
    if !(2..=8).contains(&count) {
        return Err(refuse);
    }
    let coordinator_ws = lent(workspaces, &items[0].graph.workspace)?;
    let own_ws = items
        .iter()
        .map(|item| lent(workspaces, &item.graph.workspace))
        .collect::<Result<Vec<_>>>()?;
    let verify_rows_cap = match &workspaces[coordinator_ws].dspark_verify {
        Some((_, _, rows)) => *rows,
        None => return Err(refuse),
    };
    let mut offsets = vec![0u32; count + 1];
    let mut saved_capture = Vec::with_capacity(count);
    for (index, item) in items.iter().enumerate() {
        if item.n_tokens == 0
            || item.logical_n_tokens == 0
            || item.logical_n_tokens > item.n_tokens
            || item.start as usize > item.tokens.len()
            || item.n_tokens as usize > item.tokens.len() - item.start as usize
        {
            return Err(refuse);
        }
        offsets[index + 1] = offsets[index] + item.n_tokens;
        saved_capture.push(item.graph.spec_capture_prefixes);
    }
    let total = offsets[count];
    if total > items[0].graph.prefill_cap || total > verify_rows_cap {
        return Err(refuse);
    }
    let mut flat_tokens = Vec::with_capacity(total as usize);
    let mut prepared = 0;
    for (index, item) in items.iter_mut().enumerate() {
        let rows_cap = if index == 0 { total } else { item.n_tokens };
        if item.graph.spec_prepare(b, t.weights, rows_cap).is_err()
            || item.graph.spec_frontier_save(b).is_err()
        {
            break;
        }
        item.graph.spec_capture_prefixes =
            item.n_tokens > 1 && item.n_tokens <= SPEC_PREFIX_SLOTS + 2;
        item.graph.dspark_capture_reset();
        let start = item.start as usize;
        flat_tokens.extend_from_slice(&item.tokens[start..start + item.n_tokens as usize]);
        prepared += 1;
    }
    let restore = |items: &mut [VerifyItem<'_>], upto: usize| {
        for (item, &saved) in items.iter_mut().zip(&saved_capture).take(upto) {
            item.graph.spec_capture_prefixes = saved;
        }
    };
    if prepared != count {
        restore(items, prepared);
        return Err(refuse);
    }
    let hc_row = HC_DIM * F32;
    let combine_attn_front = count.is_multiple_of(2)
        && items
            .as_chunks::<2>()
            .0
            .iter()
            .all(|[a, b]| (1..=6).contains(&a.n_tokens) && (1..=6).contains(&b.n_tokens));
    let exact_attn_output_rows =
        (4..=8).contains(&total) || matches!(total, 10 | 12 | 16 | 18 | 24 | 32);
    let combine_attn_output = combine_attn_front && exact_attn_output_rows;
    let vocab = t.weights.output.dim[1] as u32;
    let bytes: Vec<u8> = flat_tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let result = (|| {
        let ws = &mut workspaces[coordinator_ws];
        b.write(&mut ws.prefill_tokens, 0, &bytes)?;
        let embd = &t.weights.token_embd;
        let (verify_cur_hc, _) = verify_streams(&mut ws.dspark_verify)?;
        b.embed_tokens_hc(
            verify_cur_hc,
            &ws.prefill_tokens,
            w(embd),
            embd.dim[1] as u32,
            total,
            N_EMBD,
            N_HC,
        )?;
        b.set_batch_mode(BatchMode::Verifier);
        let encoded = (|| {
            b.begin_commands()?;
            let cur = items[0].graph.batch_cur_index;
            for il in 0..N_LAYER {
                let layer = &t.weights.layers[il as usize];
                let q_rank = layer.attn_q_a.dim[1];
                let r = ratio(il);
                let comp_width = u64::from(if r == 4 { 2 } else { 1 } * N_HEAD_DIM);
                let combine_attn_f16 = combine_attn_front && r != 0;
                let combine_indexer_f16 = combine_attn_f16 && r == 4;
                // The coordinator's rows of request `index` stand in for the
                // session's batch tensors; views also confine compressor and
                // indexer scratch writes to the request's slice.
                if combine_attn_front {
                    for (index, item) in items.iter_mut().enumerate() {
                        let (row0, n) = (offsets[index], item.n_tokens);
                        let ws = &mut workspaces[coordinator_ws];
                        let (verify_cur_hc, _) = verify_streams(&mut ws.dspark_verify)?;
                        b.copy(
                            &mut ws.hc[cur],
                            u64::from(row0) * hc_row,
                            verify_cur_hc,
                            u64::from(row0) * hc_row,
                            u64::from(n) * hc_row,
                        )?;
                        let front = AttnBatch {
                            front_only: true,
                            project_qb: false,
                            project_q_a_kv: false,
                            ..AttnBatch::default()
                        };
                        item.graph.encode_layer_attention_batch(
                            t,
                            il,
                            item.start,
                            n,
                            front,
                            Slots::own(ws, cur).rows(row0, n, q_rank, comp_width)?,
                        )?;
                    }
                    let ws = &mut workspaces[coordinator_ws];
                    let n = u64::from(total);
                    let embd = u64::from(N_EMBD);
                    b.matmul_q8_0_group_pairs(
                        &mut ws.qr,
                        w(&layer.attn_q_a),
                        embd,
                        q_rank,
                        &ws.attn_norm,
                        n,
                        &offsets,
                    )?;
                    b.matmul_q8_0_group_pairs(
                        &mut ws.kv_raw,
                        w(&layer.attn_kv),
                        embd,
                        u64::from(N_HEAD_DIM),
                        &ws.attn_norm,
                        n,
                        &offsets,
                    )?;
                    b.qkv_rms_norm_rows(
                        &mut ws.qr_norm,
                        &ws.qr,
                        w(&layer.attn_q_a_norm),
                        q_rank as u32,
                        &mut ws.kv,
                        &ws.kv_raw,
                        w(&layer.attn_kv_a_norm),
                        N_HEAD_DIM,
                        total,
                        RMS_EPS,
                    )?;
                    let (mut q, _) = ws.stage.attention();
                    b.matmul_q8_0_group_pairs(
                        &mut q,
                        w(&layer.attn_q_b),
                        q_rank,
                        Q_DIM,
                        &ws.qr_norm,
                        n,
                        &offsets,
                    )?;
                    if combine_attn_f16 {
                        b.matmul_f16_group_pairs(
                            &mut ws.comp_kv,
                            w(present(&layer.attn_compressor_kv)?),
                            embd,
                            comp_width,
                            &ws.attn_norm,
                            n,
                            &offsets,
                        )?;
                        b.matmul_f16_group_pairs(
                            &mut ws.comp_sc,
                            w(present(&layer.attn_compressor_gate)?),
                            embd,
                            comp_width,
                            &ws.attn_norm,
                            n,
                            &offsets,
                        )?;
                    }
                    if combine_indexer_f16 {
                        b.matmul_f16_group_pairs(
                            &mut ws.indexer_q,
                            w(present(&layer.indexer_attn_q_b)?),
                            q_rank,
                            INDEXER_Q_DIM,
                            &ws.qr_norm,
                            n,
                            &offsets,
                        )?;
                        b.matmul_f16_group_pairs(
                            &mut ws.indexer_weights,
                            w(present(&layer.indexer_proj)?),
                            embd,
                            u64::from(N_INDEXER_HEAD),
                            &ws.attn_norm,
                            n,
                            &offsets,
                        )?;
                    }
                }
                for (index, item) in items.iter_mut().enumerate() {
                    let (row0, n) = (offsets[index], item.n_tokens);
                    let block_offset = u64::from(row0) * hc_row;
                    let block_bytes = u64::from(n) * hc_row;
                    // The request's own workspace, when it is not the
                    // coordinator's.
                    let (ws, mut own) = lent_pair(workspaces, coordinator_ws, own_ws[index]);
                    if combine_attn_front {
                        let rest = AttnBatch {
                            front_ready: true,
                            compressor_projection_ready: combine_attn_f16,
                            indexer_query_projection_ready: combine_indexer_f16,
                            project_output: !combine_attn_output,
                            ..AttnBatch::default()
                        };
                        let mut slots = Slots::own(ws, cur).rows(row0, n, q_rank, comp_width)?;
                        // The output projection writes the request's own
                        // workspace.
                        if let Some(own) = own.as_deref_mut() {
                            slots.attn_low = (&mut own.attn_low).into();
                            slots.attn_out = (&mut own.attn_out).into();
                        }
                        item.graph
                            .encode_layer_attention_batch(t, il, item.start, n, rest, slots)?;
                    } else {
                        let item_cur = item.graph.batch_cur_index;
                        let (verify_cur_hc, _) = verify_streams(&mut ws.dspark_verify)?;
                        let item_hc = match own.as_deref_mut() {
                            Some(own) => &mut own.hc[item_cur],
                            None => &mut ws.hc[item_cur],
                        };
                        b.copy(item_hc, 0, verify_cur_hc, block_offset, block_bytes)?;
                        let slots = match own.as_deref_mut() {
                            Some(own) => Slots::own(own, item_cur),
                            None => Slots::own(ws, item_cur),
                        };
                        item.graph.encode_layer_attention_batch(
                            t,
                            il,
                            item.start,
                            n,
                            AttnBatch::default(),
                            slots,
                        )?;
                    }
                    if !combine_attn_output {
                        let (_, verify_after_attn_hc) = verify_streams(&mut ws.dspark_verify)?;
                        let (source, source_offset) = match (combine_attn_front, own.as_deref()) {
                            (true, _) => (&ws.after_attn_hc, block_offset),
                            (false, Some(own)) => (&own.after_attn_hc, 0),
                            (false, None) => (&ws.after_attn_hc, 0),
                        };
                        b.copy(
                            verify_after_attn_hc,
                            block_offset,
                            source,
                            source_offset,
                            block_bytes,
                        )?;
                    }
                }
                let ws = &mut workspaces[coordinator_ws];
                if combine_attn_output {
                    let (_, heads) = ws.stage.attention();
                    b.attention_output_q8_exact_batch(
                        &mut ws.attn_out,
                        &mut ws.attn_low,
                        w(&layer.attn_output_a),
                        w(&layer.attn_output_b),
                        u64::from(GROUP_DIM),
                        u64::from(N_LORA_O),
                        N_OUT_GROUP,
                        u64::from(N_EMBD),
                        &heads,
                        total,
                    )?;
                    let (_, verify_after_attn_hc) = verify_streams(&mut ws.dspark_verify)?;
                    let mut after_attn_hc = rows_mut(verify_after_attn_hc, 0, total, HC_DIM)?;
                    let attn_out = rows(&ws.attn_out, 0, total, u64::from(N_EMBD) * F32)?;
                    let rows_hc = rows(&ws.hc[cur], 0, total, hc_row)?;
                    let hc_split = rows(&ws.hc_split, 0, total, MIX_HC * F32)?;
                    b.hc_expand_split(
                        &mut after_attn_hc,
                        &attn_out,
                        &rows_hc,
                        &hc_split,
                        N_EMBD,
                        N_HC,
                    )?;
                }
                let (_, verify_after_attn_hc) = verify_streams(&mut ws.dspark_verify)?;
                b.copy(
                    &mut ws.after_attn_hc,
                    0,
                    verify_after_attn_hc,
                    0,
                    u64::from(total) * hc_row,
                )?;
                let start = items[0].start;
                let coordinator = &mut *items[0].graph;
                coordinator.encode_layer_ffn_batch(
                    t,
                    ws,
                    FfnLayer::target(t, il),
                    start,
                    total,
                    Some(&offsets),
                )?;
                let next_hc = &ws.hc[1 - cur];
                for (index, item) in items.iter_mut().enumerate() {
                    let block_hc = rows(next_hc, offsets[index], item.n_tokens, hc_row)?;
                    item.graph
                        .dspark_capture_rows(b, il, &block_hc, item.n_tokens)?;
                }
                let (verify_cur_hc, _) = verify_streams(&mut ws.dspark_verify)?;
                b.copy(
                    verify_cur_hc,
                    0,
                    &ws.hc[1 - cur],
                    0,
                    u64::from(total) * hc_row,
                )?;
            }
            let ws = &mut workspaces[coordinator_ws];
            let (verify_cur_hc, _) = verify_streams(&mut ws.dspark_verify)?;
            b.copy(
                &mut ws.hc[cur],
                0,
                verify_cur_hc,
                0,
                u64::from(total) * hc_row,
            )?;
            let coordinator = &mut *items[0].graph;
            coordinator.encode_output_head_batch(t, ws, total)?;
            let spec = coordinator
                .spec
                .as_mut()
                .ok_or(Error { op: "verify batch" })?;
            b.spec_row_argmax(&mut spec.row_tops, &spec.logits, vocab, total)?;
            b.end_commands()
        })();
        if encoded.is_err() {
            let _ = b.synchronize();
        }
        b.set_batch_mode(BatchMode::Off);
        encoded
    })();
    restore(items, count);
    result?;
    let spec = items[0].graph.spec.as_ref().ok_or(refuse)?;
    let mut bytes = vec![0u8; total as usize * 4];
    b.read(&spec.row_tops, 0, &mut bytes)?;
    let flat_tops: Vec<i32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| i32::from_le_bytes(c))
        .collect();
    distribute_verify_results(t, items, &offsets, &flat_tops)
}

#[cfg(test)]
mod tests {
    use super::greedy_accept;

    /// The upstream #612 cycle: anchor 721, drafts 72048 and EOS, each
    /// reproduced by the target argmax of the preceding row.
    #[test]
    fn greedy_acceptance_stops_before_an_accepted_stop_token() {
        let eos = 1;
        let tokens = [721, 72048, eos];
        let tops = [72048, eos, 48];
        assert_eq!(greedy_accept(&tops, &tokens, 3, -1), 3, "reproduced drafts");
        let accepted = greedy_accept(&tops, &tokens, 3, eos);
        assert!(
            accepted == 2 && tops[accepted as usize - 1] == eos,
            "an accepted stop token stays the frontier row's next anchor"
        );
        assert!(
            greedy_accept(&[5, eos, 48], &tokens, 3, eos) == 1
                && greedy_accept(&tops, &tokens, 1, eos) == 1,
            "greedy acceptance always keeps the anchor"
        );
    }
}
