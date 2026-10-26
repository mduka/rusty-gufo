//! Multi-session decode (`rocm_graph_eval_sessions_batch`): two to eight
//! sessions advance one token together. The dense projections, the FFN and
//! the output head run once over the rows in the first session's batch
//! workspace; each row's stateful attention middle runs on its own session's
//! caches.

use gufo_deepseek_v4_flash_kernels::{BatchMode, DeviceTensor, Error, Input, Result, TensorView};

use super::alloc::{BatchWorkspace, lent, rows_mut};
use super::decode::{CoreRows, Target, present, read_logits};
use super::prefill::FfnLayer;
use super::*;
use crate::weights::Weights;

/// Rows of all verification blocks of one coordinator
/// (`DS4_SPEC_MAX_ROWS`: eight requests of a 16-token block plus the
/// correction row).
pub const SPEC_MAX_ROWS: u32 = 8 * (16 + 1);

/// A compressor frontier saved before a verification block, and the
/// frontiers after its first rows (`DS4_SPEC_PREFIX_SLOTS`).
pub(crate) struct SavedFrontier {
    pub saved_kv: DeviceTensor,
    pub saved_score: DeviceTensor,
    pub prefix_kv: DeviceTensor,
    pub prefix_score: DeviceTensor,
}

/// The verification state of a graph (`rocm_graph_spec_prepare`): logits
/// of up to `rows_cap` rows, their argmax, eight frontier rows, and per
/// compressed layer the attention (and in ratio-4 layers the indexer)
/// compressor frontiers.
pub(crate) struct SpecState {
    pub rows_cap: u32,
    pub logits: DeviceTensor,
    pub row_tops: DeviceTensor,
    pub frontier_logits: DeviceTensor,
    pub attn: Vec<Option<SavedFrontier>>,
    pub index: Vec<Option<SavedFrontier>>,
    /// The compressed row counts before the block, and after each of its
    /// first rows.
    pub saved_n_comp: Vec<u32>,
    pub saved_n_index_comp: Vec<u32>,
    pub prefix_n_comp: Vec<Vec<u32>>,
    pub prefix_n_index_comp: Vec<Vec<u32>>,
    /// The saved frontier is complete.
    pub frontier_valid: bool,
}

fn frontier(b: &gufo_deepseek_v4_flash_kernels::Backend, bytes: u64) -> Result<SavedFrontier> {
    let slots = u64::from(SPEC_PREFIX_SLOTS);
    Ok(SavedFrontier {
        saved_kv: b.alloc(bytes)?,
        saved_score: b.alloc(bytes)?,
        prefix_kv: b.alloc(slots * bytes)?,
        prefix_score: b.alloc(slots * bytes)?,
    })
}

impl SpecState {
    fn alloc(
        b: &gufo_deepseek_v4_flash_kernels::Backend,
        weights: &Weights,
        rows_cap: u32,
    ) -> Result<Self> {
        let vocab = weights.output.dim[1];
        let rows = u64::from(rows_cap);
        let mut state = Self {
            rows_cap,
            logits: b.alloc(rows * vocab * F32)?,
            row_tops: b.alloc(rows * 4)?,
            frontier_logits: b.alloc(rows.min(8) * vocab * F32)?,
            attn: Vec::with_capacity(N_LAYER as usize),
            index: Vec::with_capacity(N_LAYER as usize),
            saved_n_comp: vec![0; N_LAYER as usize],
            saved_n_index_comp: vec![0; N_LAYER as usize],
            prefix_n_comp: vec![vec![0; N_LAYER as usize]; SPEC_PREFIX_SLOTS as usize],
            prefix_n_index_comp: vec![vec![0; N_LAYER as usize]; SPEC_PREFIX_SLOTS as usize],
            frontier_valid: false,
        };
        for il in 0..N_LAYER {
            let r = ratio(il);
            let coff: u64 = if r == 4 { 2 } else { 1 };
            let attn = (r != 0)
                .then(|| frontier(b, coff * u64::from(N_HEAD_DIM) * coff * u64::from(r) * F32))
                .transpose()?;
            let index = (r == 4)
                .then(|| {
                    frontier(
                        b,
                        coff * u64::from(N_INDEXER_HEAD_DIM) * coff * u64::from(r) * F32,
                    )
                })
                .transpose()?;
            state.attn.push(attn);
            state.index.push(index);
        }
        Ok(state)
    }
}

/// One session of a multi-session decode step (`ds4_rocm_batch_item`).
pub struct BatchItem<'a> {
    pub graph: &'a mut Graph,
    pub token: i32,
    pub position: u32,
    pub logits: &'a mut [f32],
}

fn rows_view(t: &impl Input, row0: u32, rows: u32, row_floats: u64) -> Result<TensorView<'_>> {
    t.view(
        u64::from(row0) * row_floats * F32,
        u64::from(rows) * row_floats * F32,
    )
    .ok_or(Error { op: "batch view" })
}

impl Graph {
    /// `rocm_graph_spec_prepare`: verification state for `rows_cap` rows,
    /// kept when large enough, else replaced.
    pub(crate) fn spec_prepare(
        &mut self,
        b: &gufo_deepseek_v4_flash_kernels::Backend,
        weights: &Weights,
        rows_cap: u32,
    ) -> Result<()> {
        if rows_cap == 0 || rows_cap > SPEC_MAX_ROWS || rows_cap > self.prefill_cap {
            return Err(Error { op: "spec prepare" });
        }
        if self.spec.as_ref().is_some_and(|s| s.rows_cap >= rows_cap) {
            return Ok(());
        }
        // Replacing the state forgets its frontier and prefix captures
        // (`rocm_graph_spec_free`).
        self.spec = None;
        self.spec_capture_prefixes = false;
        self.spec = Some(SpecState::alloc(b, weights, rows_cap).inspect_err(|_| {
            eprintln!("ds4: ROCm failed to allocate DSpark verification state");
        })?);
        Ok(())
    }

    /// `rocm_graph_encode_output_head_batch`: the output head of the first
    /// `n_tokens` stream rows of `ws` into the verification logits.
    pub(crate) fn encode_output_head_batch(
        &mut self,
        t: &Target<'_>,
        ws: &mut BatchWorkspace,
        n_tokens: u32,
    ) -> Result<()> {
        let b = t.b;
        let spec = self.spec.as_mut().ok_or(Error {
            op: "output head batch",
        })?;
        if n_tokens == 0 || n_tokens > self.prefill_cap || n_tokens > spec.rows_cap {
            return Err(Error {
                op: "output head batch",
            });
        }
        let weights = t.weights;
        let hc = u64::from(N_HC);
        let mut output_pre = rows_mut(&mut ws.hc_mix, 0, n_tokens, hc)?;
        let mut output_weights = rows_mut(&mut ws.hc_split, 0, n_tokens, hc)?;
        let mut output_embd = rows_mut(&mut ws.ffn_cur, 0, n_tokens, u64::from(N_EMBD))?;
        let mut output_norm = rows_mut(&mut ws.ffn_norm, 0, n_tokens, u64::from(N_EMBD))?;
        let cur_hc = &ws.hc[self.batch_cur_index];
        let mut flat_hc = ws.stage.flat_hc();
        b.rms_norm_plain_rows(&mut flat_hc, cur_hc, HC_DIM as u32, n_tokens, RMS_EPS)?;
        b.matmul_f16(
            &mut output_pre,
            w(&weights.output_hc_fn),
            HC_DIM,
            hc,
            &flat_hc,
            u64::from(n_tokens),
        )?;
        b.output_hc_weights(
            &mut output_weights,
            &output_pre,
            w(&weights.output_hc_scale),
            w(&weights.output_hc_base),
            N_HC,
            HC_EPS,
        )?;
        b.hc_weighted_sum(&mut output_embd, cur_hc, &output_weights, N_EMBD, N_HC)?;
        b.rms_norm_weight_rows(
            &mut output_norm,
            &output_embd,
            w(&weights.output_norm),
            N_EMBD,
            n_tokens,
            RMS_EPS,
        )?;
        b.matmul_q8_0(
            &mut spec.logits,
            w(&weights.output),
            u64::from(N_EMBD),
            weights.output.dim[1],
            &output_norm,
            u64::from(n_tokens),
        )
    }
}

/// `rocm_graph_encode_sessions_attention_batch`: the attention half of
/// layer `il` for one row per session. The first session's workspace `ws`
/// holds the rows: the shared HC pre, norm and Q/KV projections, then each
/// row's stateful middle on its own session, then the shared output
/// projection and HC post.
fn encode_sessions_attention_batch(
    t: &Target<'_>,
    ws: &mut BatchWorkspace,
    items: &mut [BatchItem<'_>],
    il: u32,
) -> Result<()> {
    let b = t.b;
    let n_tokens = items.len() as u32;
    let coordinator = &items[0].graph;
    if n_tokens < 2
        || n_tokens > coordinator.prefill_cap
        || items.iter().any(|i| i.graph.raw_cap == 0)
    {
        return Err(Error {
            op: "sessions attention batch",
        });
    }
    b.clear_f16_input();
    let cur = coordinator.batch_cur_index;
    let cur_hc = &ws.hc[cur];
    let layer = &t.weights.layers[il as usize];
    let q_rank = layer.attn_q_a.dim[1];
    let n = u64::from(n_tokens);
    let embd = u64::from(N_EMBD);
    let mut hc_mix = rows_mut(&mut ws.hc_mix, 0, n_tokens, MIX_HC)?;
    let mut hc_split = rows_mut(&mut ws.hc_split, 0, n_tokens, MIX_HC)?;
    let mut attn_cur = rows_mut(&mut ws.attn_cur, 0, n_tokens, embd)?;

    let mut flat_hc = ws.stage.flat_hc();
    b.rms_norm_plain_rows(&mut flat_hc, cur_hc, HC_DIM as u32, n_tokens, RMS_EPS)?;
    b.matmul_f16(
        &mut hc_mix,
        w(&layer.hc_attn_fn),
        HC_DIM,
        MIX_HC,
        &flat_hc,
        n,
    )?;
    // The decode fusion maps one independent block to each row, so the
    // wider grid keeps the one-row reduction order.
    b.hc_split_weighted_sum_norm(
        &mut attn_cur,
        &mut ws.attn_norm,
        &mut hc_split,
        &hc_mix,
        cur_hc,
        w(&layer.hc_attn_scale),
        w(&layer.hc_attn_base),
        w(&layer.attn_norm),
        N_EMBD,
        N_HC,
        N_HC_SINKHORN_ITER,
        HC_EPS,
        RMS_EPS,
    )?;
    b.matmul_q8_0(
        &mut ws.qr,
        w(&layer.attn_q_a),
        embd,
        q_rank,
        &ws.attn_norm,
        n,
    )?;
    b.matmul_q8_0(
        &mut ws.kv_raw,
        w(&layer.attn_kv),
        embd,
        u64::from(N_HEAD_DIM),
        &ws.attn_norm,
        n,
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
        n_tokens,
        RMS_EPS,
    )?;
    let (mut q, mut heads) = ws.stage.attention();
    b.matmul_q8_0(&mut q, w(&layer.attn_q_b), q_rank, Q_DIM, &ws.qr_norm, n)?;
    // in place
    b.head_rms_norm(&mut q, n_tokens, N_HEAD, N_HEAD_DIM, RMS_EPS)?;
    // The compressor projections keep serial decode's per-row reduction
    // order while sharing both weight streams.
    let r = ratio(il);
    let comp_width = if r == 4 { 2 * N_HEAD_DIM } else { N_HEAD_DIM };
    if r != 0 {
        b.matmul_f16_pair_narrow(
            &mut ws.comp_kv,
            &mut ws.comp_sc,
            w(present(&layer.attn_compressor_kv)?),
            w(present(&layer.attn_compressor_gate)?),
            embd,
            u64::from(comp_width),
            &ws.attn_norm,
            n,
        )?;
    }
    for (row, item) in (0..n_tokens).zip(items.iter_mut()) {
        let comp = if r != 0 {
            Some((
                rows_mut(&mut ws.comp_kv, row, 1, u64::from(comp_width))?,
                rows_mut(&mut ws.comp_sc, row, 1, u64::from(comp_width))?,
            ))
        } else {
            None
        };
        let rows = CoreRows {
            attn_norm: rows_mut(&mut ws.attn_norm, row, 1, embd)?,
            qr_norm: rows_mut(&mut ws.qr_norm, row, 1, q_rank)?,
            q: rows_mut(&mut q, row, 1, Q_DIM)?,
            kv: rows_mut(&mut ws.kv, row, 1, u64::from(N_HEAD_DIM))?,
            heads: rows_mut(&mut heads, row, 1, Q_DIM)?,
            comp,
        };
        item.graph
            .attention_core(t, il, item.position, Some(rows))?;
    }
    b.attention_output_q8_batch(
        &mut ws.attn_out,
        &mut ws.attn_low,
        w(&layer.attn_output_a),
        w(&layer.attn_output_b),
        u64::from(GROUP_DIM),
        u64::from(N_LORA_O),
        N_OUT_GROUP,
        embd,
        &heads,
        n_tokens,
    )?;
    let mut after_attn_hc = rows_mut(&mut ws.after_attn_hc, 0, n_tokens, HC_DIM)?;
    b.hc_expand_split(
        &mut after_attn_hc,
        &ws.attn_out,
        cur_hc,
        &hc_split,
        N_EMBD,
        N_HC,
    )
}

/// `rocm_graph_eval_sessions_batch`: one token for each of two to eight
/// sessions, the logits of each into its item. The first item's graph
/// coordinates: its workspace of `workspaces` holds the rows and its
/// verification state the logits.
pub fn eval_sessions_batch(
    t: &Target<'_>,
    workspaces: &mut [BatchWorkspace],
    items: &mut [BatchItem<'_>],
) -> Result<()> {
    let b = t.b;
    let count = items.len();
    if !(2..=8).contains(&count)
        || count as u32 > items[0].graph.prefill_cap
        || items.iter().any(|i| i.graph.raw_cap == 0)
    {
        return Err(Error {
            op: "sessions batch",
        });
    }
    let coordinator = lent(workspaces, &items[0].graph.workspace)?;
    let ws = &mut workspaces[coordinator];
    let n_tokens = count as u32;
    items[0].graph.spec_prepare(b, t.weights, n_tokens)?;
    for item in items.iter_mut() {
        item.graph.dspark_capture_reset();
    }
    let tokens: Vec<u8> = items.iter().flat_map(|i| i.token.to_le_bytes()).collect();
    b.write(&mut ws.prefill_tokens, 0, &tokens)?;
    let hc_row = HC_DIM * F32;
    b.set_batch_mode(BatchMode::Verifier);
    let result = (|| {
        b.begin_commands()?;
        let embd = &t.weights.token_embd;
        b.embed_tokens_hc(
            &mut ws.hc[items[0].graph.batch_cur_index],
            &ws.prefill_tokens,
            w(embd),
            embd.dim[1] as u32,
            n_tokens,
            N_EMBD,
            N_HC,
        )?;
        for il in 0..N_LAYER {
            encode_sessions_attention_batch(t, ws, items, il)?;
            let position = items[0].position;
            let coordinator = &mut *items[0].graph;
            coordinator.encode_layer_ffn_batch(
                t,
                ws,
                FfnLayer::target(t, il),
                position,
                n_tokens,
                None,
            )?;
            coordinator.batch_cur_index = 1 - coordinator.batch_cur_index;
            let cur = coordinator.batch_cur_index;
            for (row, item) in (0..n_tokens).zip(items.iter_mut()) {
                let row_hc = rows_view(&ws.hc[cur], row, 1, HC_DIM)?;
                item.graph.dspark_capture_rows(b, il, &row_hc, 1)?;
            }
        }
        items[0].graph.encode_output_head_batch(t, ws, n_tokens)?;
        let cur = items[0].graph.batch_cur_index;
        for (row, item) in (0..n_tokens).zip(items.iter_mut()) {
            b.copy(
                &mut item.graph.d.cur_hc,
                0,
                &ws.hc[cur],
                u64::from(row) * hc_row,
                hc_row,
            )?;
        }
        // The logits rows go to the items, the coordinator's among them; the
        // copies follow the step in stream order, so one wait covers both.
        let spec = items[0].graph.spec.take().ok_or(Error {
            op: "sessions batch",
        })?;
        let read = (0..n_tokens)
            .zip(items.iter_mut())
            .try_for_each(|(row, item)| read_logits(b, &spec.logits, row, item.logits));
        items[0].graph.spec = Some(spec);
        read?;
        b.end_commands()
    })();
    if result.is_err() {
        let _ = b.synchronize();
    }
    b.set_batch_mode(BatchMode::Off);
    result
}
