//! Batched layers and prompt prefill (`rocm_graph_encode_layer_attention_batch`,
//! `rocm_graph_encode_layer_ffn_batch`, `rocm_graph_encode_layer_batch`,
//! `rocm_graph_prefill_layer_major`, `rocm_graph_prefill_chunked_range`).

use gufo_deepseek_v4_flash_kernels::{
    DeviceTensor, Error, HashTable, Input, Model, Output, Result, RoutedMoe, TensorViewMut,
};

use super::alloc::{BatchWorkspace, ScratchRows, rows_mut};
use super::decode::{
    Target, cached, cached_mut, check_strides, compressor, index_scale, matmul_plain, present,
    read_logits, routed_experts,
};
use super::*;
use crate::gguf::TensorInfo;
use crate::weights::LayerWeights;

/// The parts of a batched attention layer the caller wants run
/// (the defaulted parameters of the C++ encoder): verification and
/// multi-session decode run the front and the output separately.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttnBatch {
    pub front_ready: bool,
    pub front_only: bool,
    pub project_qb: bool,
    pub project_q_a_kv: bool,
    pub compressor_projection_ready: bool,
    pub indexer_query_projection_ready: bool,
    pub project_output: bool,
}

impl Default for AttnBatch {
    fn default() -> Self {
        Self {
            front_ready: false,
            front_only: false,
            project_qb: true,
            project_q_a_kv: true,
            compressor_projection_ready: false,
            indexer_query_projection_ready: false,
            project_output: true,
        }
    }
}

fn rows_view(
    t: &impl Input,
    rows: u64,
    row_floats: u64,
) -> Result<gufo_deepseek_v4_flash_kernels::TensorView<'_>> {
    t.view(0, rows * row_floats * F32)
        .ok_or(Error { op: "batch view" })
}

fn row_view(
    t: &impl Input,
    row: u32,
    row_floats: u64,
) -> Result<gufo_deepseek_v4_flash_kernels::TensorView<'_>> {
    t.view(u64::from(row) * row_floats * F32, row_floats * F32)
        .ok_or(Error { op: "row view" })
}

/// The weights the batched FFN half runs on: a target layer, or a DSpark
/// support stage, which C++ passes as another model's layer.
#[derive(Clone, Copy)]
pub(crate) struct FfnLayer<'a> {
    pub layer: &'a LayerWeights,
    pub model: Model,
    /// The checked hash-routing table of the layer, if it routes by token.
    pub hash: Option<&'a HashTable>,
}

impl<'a> FfnLayer<'a> {
    /// Target layer `il`; the first layers route by token hash.
    pub(crate) fn target(t: &Target<'a>, il: u32) -> Self {
        Self {
            layer: &t.weights.layers[il as usize],
            model: Model::Target,
            hash: (il < N_HASH_LAYER).then(|| &t.hash_tables[il as usize]),
        }
    }

    pub(crate) fn weight(&self, t: &TensorInfo) -> gufo_deepseek_v4_flash_kernels::Weight {
        gufo_deepseek_v4_flash_kernels::Weight {
            model: self.model,
            offset: t.offset,
        }
    }
}

/// The batch tensors the attention encoder reads and writes (the slots
/// C++ rebinds to row slices of a coordinator's buffers when it verifies
/// several sessions' blocks together), exclusively.
pub(crate) struct Slots<'a> {
    pub cur_hc: TensorViewMut<'a>,
    /// `flat_hc`, then `q` in the same storage.
    pub scratch: ScratchRows<'a>,
    pub hc_mix: TensorViewMut<'a>,
    pub hc_split: TensorViewMut<'a>,
    pub attn_cur: TensorViewMut<'a>,
    pub attn_norm: TensorViewMut<'a>,
    pub qr: TensorViewMut<'a>,
    pub qr_norm: TensorViewMut<'a>,
    pub kv_raw: TensorViewMut<'a>,
    pub kv: TensorViewMut<'a>,
    pub after_attn_hc: TensorViewMut<'a>,
    pub indexer_q: TensorViewMut<'a>,
    pub indexer_weights: TensorViewMut<'a>,
    pub heads: TensorViewMut<'a>,
    pub comp_kv: TensorViewMut<'a>,
    pub comp_sc: TensorViewMut<'a>,
    /// The output projection's tensors, always the session's own.
    pub attn_low: TensorViewMut<'a>,
    pub attn_out: TensorViewMut<'a>,
}

impl<'a> Slots<'a> {
    /// The workspace's own tensors, `cur` being the current stream buffer.
    pub(crate) fn own(ws: &'a mut BatchWorkspace, cur: usize) -> Self {
        let (scratch, heads) = ws.stage.split();
        Self {
            cur_hc: (&mut ws.hc[cur]).into(),
            scratch,
            hc_mix: (&mut ws.hc_mix).into(),
            hc_split: (&mut ws.hc_split).into(),
            attn_cur: (&mut ws.attn_cur).into(),
            attn_norm: (&mut ws.attn_norm).into(),
            qr: (&mut ws.qr).into(),
            qr_norm: (&mut ws.qr_norm).into(),
            kv_raw: (&mut ws.kv_raw).into(),
            kv: (&mut ws.kv).into(),
            after_attn_hc: (&mut ws.after_attn_hc).into(),
            indexer_q: (&mut ws.indexer_q).into(),
            indexer_weights: (&mut ws.indexer_weights).into(),
            heads,
            comp_kv: (&mut ws.comp_kv).into(),
            comp_sc: (&mut ws.comp_sc).into(),
            attn_low: (&mut ws.attn_low).into(),
            attn_out: (&mut ws.attn_out).into(),
        }
    }

    /// Rows `row0 .. row0 + n` of these whole slots, a coordinator's rows
    /// of one request; the output projection's tensors stay whole.
    pub(crate) fn rows(self, row0: u32, n: u32, q_rank: u64, comp_width: u64) -> Result<Self> {
        let embd = u64::from(N_EMBD);
        let head_dim = u64::from(N_HEAD_DIM);
        Ok(Self {
            cur_hc: rows_mut(self.cur_hc, row0, n, HC_DIM)?,
            scratch: self.scratch.rows(row0, n),
            hc_mix: rows_mut(self.hc_mix, row0, n, MIX_HC)?,
            hc_split: rows_mut(self.hc_split, row0, n, MIX_HC)?,
            attn_cur: rows_mut(self.attn_cur, row0, n, embd)?,
            attn_norm: rows_mut(self.attn_norm, row0, n, embd)?,
            qr: rows_mut(self.qr, row0, n, q_rank)?,
            qr_norm: rows_mut(self.qr_norm, row0, n, q_rank)?,
            kv_raw: rows_mut(self.kv_raw, row0, n, head_dim)?,
            kv: rows_mut(self.kv, row0, n, head_dim)?,
            after_attn_hc: rows_mut(self.after_attn_hc, row0, n, HC_DIM)?,
            indexer_q: rows_mut(self.indexer_q, row0, n, INDEXER_Q_DIM)?,
            indexer_weights: rows_mut(self.indexer_weights, row0, n, u64::from(N_INDEXER_HEAD))?,
            heads: rows_mut(self.heads, row0, n, Q_DIM)?,
            comp_kv: rows_mut(self.comp_kv, row0, n, comp_width)?,
            comp_sc: rows_mut(self.comp_sc, row0, n, comp_width)?,
            attn_low: self.attn_low,
            attn_out: self.attn_out,
        })
    }
}

impl Graph {
    /// `rocm_graph_refresh_ratio4_compressor_state`: the ratio-4 frontier
    /// of layer `il` (its indexer's when `index`) rebuilt from a chunk's last
    /// four rows of `attn_norm` with the small-batch projection into
    /// `comp_kv` and `comp_sc` (the chunk's matrix-matrix projection rounds
    /// differently).
    fn refresh_ratio4_state(
        &mut self,
        t: &Target<'_>,
        attn_norm: &impl Input,
        comp_kv: &mut TensorViewMut<'_>,
        comp_sc: &mut TensorViewMut<'_>,
        il: u32,
        index: bool,
        kv_weight: &TensorInfo,
        score_weight: &TensorInfo,
        ape: &TensorInfo,
        head_dim: u32,
        width: u32,
        pos0: u32,
        n_tokens: u32,
    ) -> Result<()> {
        if n_tokens < 4 {
            return Ok(());
        }
        let b = t.b;
        let embd = u64::from(N_EMBD);
        let cache = &mut self.layers[il as usize];
        let (state_kv, state_score) = if index {
            (
                cached_mut(&mut cache.index_state_kv)?,
                cached_mut(&mut cache.index_state_score)?,
            )
        } else {
            (
                cached_mut(&mut cache.attn_state_kv)?,
                cached_mut(&mut cache.attn_state_score)?,
            )
        };
        let tail = attn_norm
            .view(u64::from(n_tokens - 4) * embd * F32, 4 * embd * F32)
            .ok_or(Error { op: "ratio4 tail" })?;
        b.matmul_f16(comp_kv, w(kv_weight), embd, u64::from(width), &tail, 4)?;
        b.matmul_f16(comp_sc, w(score_weight), embd, u64::from(width), &tail, 4)?;
        b.compressor_prefill_state_ratio4(
            state_kv,
            state_score,
            comp_kv,
            comp_sc,
            w(ape),
            head_dim,
            pos0 + n_tokens - 4,
        )
    }

    /// `rocm_graph_encode_layer_attention_batch`: the attention half of
    /// layer `il` for `n_tokens` rows from `pos0` (HC pre and norm, Q and
    /// KV, cache and compression, attention, output projection, HC post) on
    /// the batch tensors `s`.
    pub(crate) fn encode_layer_attention_batch(
        &mut self,
        t: &Target<'_>,
        il: u32,
        pos0: u32,
        n_tokens: u32,
        o: AttnBatch,
        mut s: Slots<'_>,
    ) -> Result<()> {
        let b = t.b;
        if n_tokens == 0 || n_tokens > self.prefill_cap || (o.front_ready && o.front_only) {
            return Err(Error {
                op: "attention batch",
            });
        }
        self.reserve_compressed_position(b, il, u64::from(pos0) + u64::from(n_tokens))?;
        // A published F16 mirror belongs to the previous layer, whose buffers
        // this layer reuses.
        b.clear_f16_input();
        let layer = &t.weights.layers[il as usize];
        let q_rank = layer.attn_q_a.dim[1];
        let r = ratio(il);
        let compressed = r != 0;
        let zero_prefix = pos0 == 0;
        let rope = rope(il, pos0);
        let n = u64::from(n_tokens);
        let embd = u64::from(N_EMBD);
        let mut comp_counts = vec![0u32; if compressed { n_tokens as usize } else { 0 }];
        let mut index_counts = vec![0u32; if r == 4 { n_tokens as usize } else { 0 }];
        let batch_cur_hc = &s.cur_hc;

        if !o.front_ready {
            let mut hc_mix = rows_mut(&mut s.hc_mix, 0, n_tokens, MIX_HC)?;
            let mut hc_split = rows_mut(&mut s.hc_split, 0, n_tokens, MIX_HC)?;
            let mut attn_cur = rows_mut(&mut s.attn_cur, 0, n_tokens, embd)?;
            let mut hc_pre_fused = false;
            if n_tokens >= 128 {
                hc_pre_fused = true;
                let fused = b.hc_norm_mix_split_weighted_sum(
                    &mut attn_cur,
                    &mut hc_mix,
                    &mut hc_split,
                    batch_cur_hc,
                    w(&layer.hc_attn_fn),
                    w(&layer.hc_attn_scale),
                    w(&layer.hc_attn_base),
                    N_EMBD,
                    N_HC,
                    n_tokens,
                    N_HC_SINKHORN_ITER,
                    HC_EPS,
                    RMS_EPS,
                )?;
                if !fused {
                    return Err(Error { op: "hc norm mix" });
                }
            }
            if !hc_pre_fused {
                let mut flat_hc = s.scratch.flat_hc()?;
                b.rms_norm_plain_rows(
                    &mut flat_hc,
                    batch_cur_hc,
                    HC_DIM as u32,
                    n_tokens,
                    RMS_EPS,
                )?;
                b.matmul_f16(
                    &mut hc_mix,
                    w(&layer.hc_attn_fn),
                    HC_DIM,
                    MIX_HC,
                    &flat_hc,
                    n,
                )?;
                b.hc_split_weighted_sum(
                    &mut attn_cur,
                    &mut hc_split,
                    &hc_mix,
                    batch_cur_hc,
                    w(&layer.hc_attn_scale),
                    w(&layer.hc_attn_base),
                    N_EMBD,
                    N_HC,
                    N_HC_SINKHORN_ITER,
                    HC_EPS,
                )?;
            }
            b.rms_norm_weight_rows(
                &mut s.attn_norm,
                &s.attn_cur,
                w(&layer.attn_norm),
                N_EMBD,
                n_tokens,
                RMS_EPS,
            )?;
            // One F16 mirror of the normalized rows serves the layer's F16
            // projections; declining is not an error.
            if n_tokens >= 128 {
                let _ = b.publish_f16_input(&s.attn_norm, n * embd);
            }
            if o.project_q_a_kv {
                b.matmul_q8_0(&mut s.qr, w(&layer.attn_q_a), embd, q_rank, &s.attn_norm, n)?;
                b.matmul_q8_0(
                    &mut s.kv_raw,
                    w(&layer.attn_kv),
                    embd,
                    u64::from(N_HEAD_DIM),
                    &s.attn_norm,
                    n,
                )?;
                b.qkv_rms_norm_rows(
                    &mut s.qr_norm,
                    &s.qr,
                    w(&layer.attn_q_a_norm),
                    q_rank as u32,
                    &mut s.kv,
                    &s.kv_raw,
                    w(&layer.attn_kv_a_norm),
                    N_HEAD_DIM,
                    n_tokens,
                    RMS_EPS,
                )?;
            }
            if o.project_qb {
                let mut q = s.scratch.q()?;
                b.matmul_q8_0(&mut q, w(&layer.attn_q_b), q_rank, Q_DIM, &s.qr_norm, n)?;
            }
        }
        if o.front_only {
            return Ok(());
        }
        let mut q = s.scratch.q()?;
        // One pass over the query rows; falls back when the row does not fit.
        // in place
        let q_norm_rope_fused = n_tokens > 6
            && b.head_rms_norm_rope_tail(
                &mut q, n_tokens, N_HEAD, N_HEAD_DIM, false, rope, RMS_EPS,
            )
            .unwrap_or(false);
        if !q_norm_rope_fused {
            // in place
            b.head_rms_norm(&mut q, n_tokens, N_HEAD, N_HEAD_DIM, RMS_EPS)?;
            // in place
            b.rope_tail(&mut q, n_tokens, N_HEAD, N_HEAD_DIM, false, rope)?;
        }
        // in place
        b.rope_tail(&mut s.kv, n_tokens, N_HEAD_KV, N_HEAD_DIM, false, rope)?;
        // in place
        b.fp8_kv_quantize(&mut s.kv, n_tokens, N_HEAD_DIM, N_ROT)?;
        let raw_cap = self.raw_cap;
        if zero_prefix {
            b.store_raw_kv_batch(
                &mut self.layers[il as usize].raw,
                &s.kv,
                raw_cap,
                pos0,
                n_tokens,
                N_HEAD_DIM,
            )?;
        }
        let raw_batch_attention = zero_prefix && r == 0;
        let mut batch_attention_done = false;

        if raw_batch_attention {
            b.attention_prefill_raw_heads(
                &mut s.heads,
                w(&layer.attn_sinks),
                &q,
                &s.kv,
                n_tokens,
                self.raw_window,
                N_HEAD,
                N_HEAD_DIM,
            )?;
            batch_attention_done = true;
        } else if !zero_prefix && r == 0 && n_tokens <= raw_cap {
            // The ring holds the chunk and the previous window; attention reads
            // the window linearized from raw_start.
            let n_raw = self.raw_span_for_batch(pos0, n_tokens);
            let raw_start = self.raw_start_for_span(pos0 + n_tokens - 1, n_raw);
            let raw = &mut self.layers[il as usize].raw;
            b.store_raw_kv_batch(raw, &s.kv, raw_cap, pos0, n_tokens, N_HEAD_DIM)?;
            b.attention_decode_raw_batch_heads(
                &mut s.heads,
                w(&layer.attn_sinks),
                &q,
                &*raw,
                n_tokens,
                pos0,
                n_raw,
                raw_cap,
                raw_start,
                self.raw_window,
                N_HEAD,
                N_HEAD_DIM,
            )?;
            batch_attention_done = true;
        } else if r != 0 {
            let coff = if r == 4 { 2 } else { 1 };
            let comp_width = coff * N_HEAD_DIM;
            let attn_kv_w = present(&layer.attn_compressor_kv)?;
            let attn_gate_w = present(&layer.attn_compressor_gate)?;
            let attn_ape = present(&layer.attn_compressor_ape)?;
            let attn_c = compressor(&layer.attn_compressor_ape, &layer.attn_compressor_norm)?;
            if !o.compressor_projection_ready {
                b.matmul_f16(
                    &mut s.comp_kv,
                    w(attn_kv_w),
                    embd,
                    u64::from(comp_width),
                    &s.attn_norm,
                    n,
                )?;
                b.matmul_f16(
                    &mut s.comp_sc,
                    w(attn_gate_w),
                    embd,
                    u64::from(comp_width),
                    &s.attn_norm,
                    n,
                )?;
            }
            let hd = u64::from(N_HEAD_DIM);
            let n_comp;
            if zero_prefix {
                n_comp = n_tokens / r;
                if n_comp > self.layers[il as usize].comp_cap {
                    eprintln!(
                        "ds4: ROCm layer-major compressed KV cache capacity exceeded at layer {il}"
                    );
                    return Err(Error {
                        op: "compressed capacity",
                    });
                }
                let cache = &mut self.layers[il as usize];
                let view_rows = u64::from(n_comp.max(1));
                let mut target = cached_mut(&mut cache.attn_comp)?
                    .view_mut(0, view_rows * hd * F32)
                    .ok_or(Error { op: "comp target" })?;
                let mut mirror = match (&mut cache.attn_comp_f16, r == 4) {
                    (Some(m), true) => Some(
                        m.view_mut(0, view_rows * hd * 2)
                            .ok_or(Error { op: "comp mirror" })?,
                    ),
                    (None, true) => return Err(Error { op: "comp mirror" }),
                    _ => None,
                };
                // in place: the compressor state
                b.compressor_prefill(
                    &mut target,
                    cached_mut(&mut cache.attn_state_kv)?,
                    cached_mut(&mut cache.attn_state_score)?,
                    &s.comp_kv,
                    &s.comp_sc,
                    attn_c,
                    N_HEAD_DIM,
                    r,
                    pos0,
                    n_tokens,
                    rope,
                    true,
                    RMS_EPS,
                    mirror.as_mut().map(|m| m as &mut dyn Output),
                )?;
                if r == 4 {
                    self.refresh_ratio4_state(
                        t,
                        &s.attn_norm,
                        &mut s.comp_kv,
                        &mut s.comp_sc,
                        il,
                        false,
                        attn_kv_w,
                        attn_gate_w,
                        attn_ape,
                        N_HEAD_DIM,
                        comp_width,
                        pos0,
                        n_tokens,
                    )?;
                }
                self.layers[il as usize].n_comp = n_comp;
                for (t_, count) in comp_counts.iter_mut().enumerate() {
                    *count = (pos0 + t_ as u32 + 1) / r;
                }
            } else {
                // Verification may reject an aligned block partway; the bulk
                // compressor only saves its final state, so prefix captures
                // need the row loop.
                let aligned = !self.spec_capture_prefixes
                    && pos0.is_multiple_of(r)
                    && n_tokens.is_multiple_of(r);
                if aligned {
                    let before = self.layers[il as usize].n_comp;
                    let chunk = n_tokens / r;
                    if before + chunk > self.layers[il as usize].comp_cap {
                        eprintln!(
                            "ds4: ROCm graph compressed KV cache capacity exceeded at layer {il}"
                        );
                        return Err(Error {
                            op: "compressed capacity",
                        });
                    }
                    let cache = &mut self.layers[il as usize];
                    let view_rows = u64::from(chunk.max(1));
                    let mut target = cached_mut(&mut cache.attn_comp)?
                        .view_mut(u64::from(before) * hd * F32, view_rows * hd * F32)
                        .ok_or(Error { op: "comp target" })?;
                    let mut mirror = match (&mut cache.attn_comp_f16, r == 4) {
                        (Some(m), true) => Some(
                            m.view_mut(u64::from(before) * hd * 2, view_rows * hd * 2)
                                .ok_or(Error { op: "comp mirror" })?,
                        ),
                        (None, true) => return Err(Error { op: "comp mirror" }),
                        _ => None,
                    };
                    if r == 4 {
                        // in place: the compressor state
                        b.compressor_prefill_ratio4_replay(
                            &mut target,
                            cached_mut(&mut cache.attn_state_kv)?,
                            cached_mut(&mut cache.attn_state_score)?,
                            &s.comp_kv,
                            &s.comp_sc,
                            attn_c,
                            N_HEAD_DIM,
                            pos0,
                            n_tokens,
                            rope,
                            true,
                            RMS_EPS,
                            mirror.as_mut().map(|m| m as &mut dyn Output),
                        )?;
                    } else {
                        // in place: the compressor state
                        b.compressor_prefill(
                            &mut target,
                            cached_mut(&mut cache.attn_state_kv)?,
                            cached_mut(&mut cache.attn_state_score)?,
                            &s.comp_kv,
                            &s.comp_sc,
                            attn_c,
                            N_HEAD_DIM,
                            r,
                            pos0,
                            n_tokens,
                            rope,
                            true,
                            RMS_EPS,
                            mirror.as_mut().map(|m| m as &mut dyn Output),
                        )?;
                    }
                    if r == 4 {
                        self.refresh_ratio4_state(
                            t,
                            &s.attn_norm,
                            &mut s.comp_kv,
                            &mut s.comp_sc,
                            il,
                            false,
                            attn_kv_w,
                            attn_gate_w,
                            attn_ape,
                            N_HEAD_DIM,
                            comp_width,
                            pos0,
                            n_tokens,
                        )?;
                    }
                    self.layers[il as usize].n_comp = before + chunk;
                    for (t_, count) in comp_counts.iter_mut().enumerate() {
                        *count = (pos0 + t_ as u32 + 1) / r;
                    }
                } else {
                    for t_ in 0..n_tokens {
                        let pos = pos0 + t_;
                        let emit = (pos + 1).is_multiple_of(r);
                        if emit
                            && self.layers[il as usize].n_comp >= self.layers[il as usize].comp_cap
                        {
                            eprintln!(
                                "ds4: ROCm graph compressed KV cache capacity exceeded at layer {il}"
                            );
                            return Err(Error {
                                op: "compressed capacity",
                            });
                        }
                        let kv_row = row_view(&s.comp_kv, t_, u64::from(comp_width))?;
                        let sc_row = row_view(&s.comp_sc, t_, u64::from(comp_width))?;
                        let cache = &mut self.layers[il as usize];
                        let comp_row = cache.n_comp;
                        // in place: the compressor state
                        b.compressor_update(
                            &kv_row,
                            &sc_row,
                            cached_mut(&mut cache.attn_state_kv)?,
                            cached_mut(&mut cache.attn_state_score)?,
                            cached_mut(&mut cache.attn_comp)?,
                            attn_c,
                            N_HEAD_DIM,
                            r,
                            pos,
                            comp_row,
                            rope,
                            RMS_EPS,
                            false,
                        )?;
                        if emit {
                            self.layers[il as usize].quantize_comp_row(b, comp_row)?;
                            self.layers[il as usize].n_comp += 1;
                        }
                        comp_counts[t_ as usize] = self.layers[il as usize].n_comp;
                        if t_ + 1 < n_tokens && t_ < SPEC_PREFIX_SLOTS {
                            self.capture_prefix_attn_state(t, il, t_)?;
                        }
                    }
                }
                n_comp = self.layers[il as usize].n_comp;
            }

            if r == 4 {
                let index_width = coff * N_INDEXER_HEAD_DIM;
                let idx_kv_w = present(&layer.indexer_compressor_kv)?;
                let idx_gate_w = present(&layer.indexer_compressor_gate)?;
                let idx_ape = present(&layer.indexer_compressor_ape)?;
                let idx_c = compressor(
                    &layer.indexer_compressor_ape,
                    &layer.indexer_compressor_norm,
                )?;
                let idx_q_b = present(&layer.indexer_attn_q_b)?;
                let idx_proj = present(&layer.indexer_proj)?;
                b.matmul_f16(
                    &mut s.comp_kv,
                    w(idx_kv_w),
                    embd,
                    u64::from(index_width),
                    &s.attn_norm,
                    n,
                )?;
                b.matmul_f16(
                    &mut s.comp_sc,
                    w(idx_gate_w),
                    embd,
                    u64::from(index_width),
                    &s.attn_norm,
                    n,
                )?;
                if !o.indexer_query_projection_ready {
                    b.matmul_f16(
                        &mut s.indexer_q,
                        w(idx_q_b),
                        q_rank,
                        INDEXER_Q_DIM,
                        &s.qr_norm,
                        n,
                    )?;
                }
                // in place
                b.rope_tail(
                    &mut s.indexer_q,
                    n_tokens,
                    N_INDEXER_HEAD,
                    N_INDEXER_HEAD_DIM,
                    false,
                    rope,
                )?;
                // in place
                b.indexer_qat(
                    &mut s.indexer_q,
                    n_tokens * N_INDEXER_HEAD,
                    N_INDEXER_HEAD_DIM,
                )?;
                if !o.indexer_query_projection_ready {
                    b.matmul_f16(
                        &mut s.indexer_weights,
                        w(idx_proj),
                        embd,
                        u64::from(N_INDEXER_HEAD),
                        &s.attn_norm,
                        n,
                    )?;
                }
                let iw = u64::from(N_INDEXER_HEAD_DIM) * F32;
                if zero_prefix {
                    if n_comp > self.layers[il as usize].comp_cap {
                        eprintln!(
                            "ds4: ROCm layer-major indexer cache capacity exceeded at layer {il}"
                        );
                        return Err(Error {
                            op: "indexer capacity",
                        });
                    }
                    let cache = &mut self.layers[il as usize];
                    // in place: the compressor state
                    b.compressor_prefill(
                        cached_mut(&mut cache.index_comp)?,
                        cached_mut(&mut cache.index_state_kv)?,
                        cached_mut(&mut cache.index_state_score)?,
                        &s.comp_kv,
                        &s.comp_sc,
                        idx_c,
                        N_INDEXER_HEAD_DIM,
                        r,
                        pos0,
                        n_tokens,
                        rope,
                        false,
                        RMS_EPS,
                        None,
                    )?;
                    if n_comp != 0 {
                        // in place
                        b.indexer_qat(
                            cached_mut(&mut cache.index_comp)?,
                            n_comp,
                            N_INDEXER_HEAD_DIM,
                        )?;
                    }
                    self.refresh_ratio4_state(
                        t,
                        &s.attn_norm,
                        &mut s.comp_kv,
                        &mut s.comp_sc,
                        il,
                        true,
                        idx_kv_w,
                        idx_gate_w,
                        idx_ape,
                        N_INDEXER_HEAD_DIM,
                        index_width,
                        pos0,
                        n_tokens,
                    )?;
                    self.layers[il as usize].n_index_comp = n_comp;
                    for (t_, count) in index_counts.iter_mut().enumerate() {
                        *count = (pos0 + t_ as u32 + 1) / r;
                    }
                } else {
                    let aligned = !self.spec_capture_prefixes
                        && pos0.is_multiple_of(r)
                        && n_tokens.is_multiple_of(r);
                    if aligned {
                        let before = self.layers[il as usize].n_index_comp;
                        let chunk = n_tokens / r;
                        if before + chunk > self.layers[il as usize].comp_cap {
                            eprintln!(
                                "ds4: ROCm graph indexer compressed KV cache capacity exceeded at layer {il}"
                            );
                            return Err(Error {
                                op: "indexer capacity",
                            });
                        }
                        let cache = &mut self.layers[il as usize];
                        let mut view = cached_mut(&mut cache.index_comp)?
                            .view_mut(u64::from(before) * iw, u64::from(chunk) * iw)
                            .ok_or(Error { op: "index view" })?;
                        // in place: the compressor state
                        b.compressor_prefill_ratio4_replay(
                            &mut view,
                            cached_mut(&mut cache.index_state_kv)?,
                            cached_mut(&mut cache.index_state_score)?,
                            &s.comp_kv,
                            &s.comp_sc,
                            idx_c,
                            N_INDEXER_HEAD_DIM,
                            pos0,
                            n_tokens,
                            rope,
                            false,
                            RMS_EPS,
                            None,
                        )?;
                        if chunk != 0 {
                            // in place
                            b.indexer_qat(&mut view, chunk, N_INDEXER_HEAD_DIM)?;
                        }
                        self.refresh_ratio4_state(
                            t,
                            &s.attn_norm,
                            &mut s.comp_kv,
                            &mut s.comp_sc,
                            il,
                            true,
                            idx_kv_w,
                            idx_gate_w,
                            idx_ape,
                            N_INDEXER_HEAD_DIM,
                            index_width,
                            pos0,
                            n_tokens,
                        )?;
                        self.layers[il as usize].n_index_comp = before + chunk;
                        for (t_, count) in index_counts.iter_mut().enumerate() {
                            *count = (pos0 + t_ as u32 + 1) / r;
                        }
                    } else {
                        for t_ in 0..n_tokens {
                            let pos = pos0 + t_;
                            let emit = (pos + 1).is_multiple_of(r);
                            let cache = &mut self.layers[il as usize];
                            if emit && cache.n_index_comp >= cache.comp_cap {
                                eprintln!(
                                    "ds4: ROCm graph indexer compressed KV cache capacity exceeded at layer {il}"
                                );
                                return Err(Error {
                                    op: "indexer capacity",
                                });
                            }
                            let kv_row = row_view(&s.comp_kv, t_, u64::from(index_width))?;
                            let sc_row = row_view(&s.comp_sc, t_, u64::from(index_width))?;
                            let index_row = cache.n_index_comp;
                            // in place: the compressor state
                            b.compressor_update(
                                &kv_row,
                                &sc_row,
                                cached_mut(&mut cache.index_state_kv)?,
                                cached_mut(&mut cache.index_state_score)?,
                                cached_mut(&mut cache.index_comp)?,
                                idx_c,
                                N_INDEXER_HEAD_DIM,
                                r,
                                pos,
                                index_row,
                                rope,
                                RMS_EPS,
                                false,
                            )?;
                            if emit {
                                let mut row = cached_mut(&mut cache.index_comp)?
                                    .view_mut(u64::from(index_row) * iw, iw)
                                    .ok_or(Error { op: "index row" })?;
                                // in place
                                b.indexer_qat(&mut row, 1, N_INDEXER_HEAD_DIM)?;
                                self.layers[il as usize].n_index_comp += 1;
                            }
                            index_counts[t_ as usize] = self.layers[il as usize].n_index_comp;
                            if t_ + 1 < n_tokens && t_ < SPEC_PREFIX_SLOTS {
                                self.capture_prefix_index_state(t, il, t_)?;
                            }
                        }
                    }
                }
            }

            // A verifier block may cross the first sparse-indexer boundary;
            // its earlier rows keep the dense compressed attention order.
            let crosses_indexer_boundary = self.spec_capture_prefixes
                && r == 4
                && n_comp > N_INDEXER_TOP_K
                && (pos0 + 1) / r <= N_INDEXER_TOP_K;
            if !zero_prefix && n_tokens <= raw_cap && !crosses_indexer_boundary {
                let n_raw = self.raw_span_for_batch(pos0, n_tokens);
                let raw_start = self.raw_start_for_span(pos0 + n_tokens - 1, n_raw);
                b.store_raw_kv_batch(
                    &mut self.layers[il as usize].raw,
                    &s.kv,
                    raw_cap,
                    pos0,
                    n_tokens,
                    N_HEAD_DIM,
                )?;
                let mut use_indexed_comp = false;
                if r == 4 && n_comp > N_INDEXER_TOP_K {
                    self.reserve_indexer_scores(b, n_comp, n_tokens)?;
                    let cache = &self.layers[il as usize];
                    b.indexer_scores_decode_batch(
                        &mut self.d.indexer_scores,
                        &s.indexer_q,
                        &s.indexer_weights,
                        cached(&cache.index_comp)?,
                        n_comp,
                        n_tokens,
                        pos0,
                        N_INDEXER_HEAD,
                        N_INDEXER_HEAD_DIM,
                        r,
                        index_scale(),
                        Some(&mut s.heads),
                    )?;
                    b.indexer_topk(
                        &mut self.d.comp_selected,
                        &self.d.indexer_scores,
                        n_comp,
                        n_tokens,
                        N_INDEXER_TOP_K,
                    )?;
                    use_indexed_comp = true;
                }
                let cache = &self.layers[il as usize];
                if use_indexed_comp {
                    let (comp, comp_f16) = self.prefill_attn_comp_cache(il, n_tokens)?;
                    b.attention_indexed_mixed_batch_heads(
                        &mut s.heads,
                        w(&layer.attn_sinks),
                        &q,
                        &cache.raw,
                        comp,
                        comp_f16,
                        &self.d.comp_selected,
                        n_tokens,
                        pos0,
                        n_raw,
                        raw_cap,
                        raw_start,
                        n_comp,
                        N_INDEXER_TOP_K,
                        self.raw_window,
                        r,
                        N_HEAD,
                        N_HEAD_DIM,
                    )?;
                } else {
                    b.attention_decode_mixed_batch_heads(
                        &mut s.heads,
                        w(&layer.attn_sinks),
                        &q,
                        &cache.raw,
                        Some(cached(&cache.attn_comp)? as &dyn Input),
                        None,
                        n_tokens,
                        pos0,
                        n_raw,
                        raw_cap,
                        raw_start,
                        n_comp,
                        self.raw_window,
                        r,
                        N_HEAD,
                        N_HEAD_DIM,
                    )?;
                }
                batch_attention_done = true;
            }

            let topk_prefill_needed = r == 4 && n_comp > N_INDEXER_TOP_K;
            if zero_prefix && topk_prefill_needed && n_comp != 0 {
                self.reserve_indexer_scores(b, n_comp, n_tokens)?;
                let cache = &self.layers[il as usize];
                b.indexer_scores_prefill(
                    &mut self.d.indexer_scores,
                    &s.indexer_q,
                    &s.indexer_weights,
                    cached(&cache.index_comp)?,
                    n_comp,
                    n_tokens,
                    N_INDEXER_HEAD,
                    N_INDEXER_HEAD_DIM,
                    r,
                    index_scale(),
                    Some(&mut s.heads),
                )?;
                b.indexer_topk(
                    &mut self.d.comp_selected,
                    &self.d.indexer_scores,
                    n_comp,
                    n_tokens,
                    N_INDEXER_TOP_K,
                )?;
                let (comp, comp_f16) = self.prefill_attn_comp_cache(il, n_tokens)?;
                b.attention_indexed_mixed_batch_heads(
                    &mut s.heads,
                    w(&layer.attn_sinks),
                    &q,
                    &cache.raw,
                    comp,
                    comp_f16,
                    &self.d.comp_selected,
                    n_tokens,
                    pos0,
                    n_tokens,
                    raw_cap,
                    0,
                    n_comp,
                    N_INDEXER_TOP_K,
                    self.raw_window,
                    r,
                    N_HEAD,
                    N_HEAD_DIM,
                )?;
                batch_attention_done = true;
            }
            if zero_prefix && !topk_prefill_needed && n_comp != 0 {
                let cache = &self.layers[il as usize];
                b.attention_prefill_static_mixed_heads(
                    &mut s.heads,
                    w(&layer.attn_sinks),
                    &q,
                    &s.kv,
                    Some(cached(&cache.attn_comp)? as &dyn Input),
                    n_tokens,
                    n_comp,
                    self.raw_window,
                    r,
                    N_HEAD,
                    N_HEAD_DIM,
                )?;
                batch_attention_done = true;
            }
        }

        if !raw_batch_attention && !batch_attention_done {
            let mut raw_prefix_tokens = 0;
            if zero_prefix && r != 0 && n_tokens <= raw_cap {
                while raw_prefix_tokens < n_tokens && comp_counts[raw_prefix_tokens as usize] == 0 {
                    raw_prefix_tokens += 1;
                }
            }
            if raw_prefix_tokens != 0 {
                b.attention_prefill_raw_heads(
                    &mut s.heads,
                    w(&layer.attn_sinks),
                    &q,
                    &s.kv,
                    raw_prefix_tokens,
                    self.raw_window,
                    N_HEAD,
                    N_HEAD_DIM,
                )?;
            }
            for t_ in raw_prefix_tokens..n_tokens {
                let pos = pos0 + t_;
                let n_raw = self.raw_span_for_batch(pos, 1);
                let raw_start = self.raw_start_for_span(pos, n_raw);
                let cur_comp = comp_counts.get(t_ as usize).copied().unwrap_or(0);
                let cur_index = index_counts.get(t_ as usize).copied().unwrap_or(0);
                let mut n_selected = 0;
                if r == 4 && cur_comp > N_INDEXER_TOP_K {
                    let iq = row_view(&s.indexer_q, t_, INDEXER_Q_DIM)?;
                    let iwv = row_view(&s.indexer_weights, t_, u64::from(N_INDEXER_HEAD))?;
                    self.reserve_indexer_scores(b, cur_index, 1)?;
                    let cache = &self.layers[il as usize];
                    b.indexer_score_one(
                        &mut self.d.indexer_scores,
                        &iq,
                        &iwv,
                        cached(&cache.index_comp)?,
                        cur_index,
                        N_INDEXER_HEAD,
                        N_INDEXER_HEAD_DIM,
                        index_scale(),
                    )?;
                    b.indexer_topk(
                        &mut self.d.comp_selected,
                        &self.d.indexer_scores,
                        cur_index,
                        1,
                        N_INDEXER_TOP_K,
                    )?;
                    n_selected = N_INDEXER_TOP_K.min(cur_index);
                }
                let q_row = row_view(&q, t_, Q_DIM)?;
                let kv_row = row_view(&s.kv, t_, u64::from(N_HEAD_DIM))?;
                let mut heads_row = rows_mut(&mut s.heads, t_, 1, Q_DIM)?;
                let cache = &mut self.layers[il as usize];
                if !zero_prefix {
                    b.store_raw_kv(&mut cache.raw, &kv_row, raw_cap, pos % raw_cap, N_HEAD_DIM)?;
                }
                if n_selected != 0 {
                    b.attention_indexed_mixed_batch_heads(
                        &mut heads_row,
                        w(&layer.attn_sinks),
                        &q_row,
                        &cache.raw,
                        cached(&cache.attn_comp)?,
                        false,
                        &self.d.comp_selected,
                        1,
                        pos,
                        n_raw,
                        raw_cap,
                        raw_start,
                        cur_comp,
                        n_selected,
                        self.raw_window,
                        r,
                        N_HEAD,
                        N_HEAD_DIM,
                    )?;
                } else {
                    let comp = if cur_comp != 0 {
                        Some(cached(&cache.attn_comp)? as &dyn Input)
                    } else {
                        None
                    };
                    b.attention_decode_heads(
                        &mut heads_row,
                        w(&layer.attn_sinks),
                        &q_row,
                        &cache.raw,
                        n_raw,
                        raw_cap,
                        raw_start,
                        comp,
                        cur_comp,
                        None,
                        N_HEAD,
                        N_HEAD_DIM,
                    )?;
                }
            }
        }

        if !o.project_output {
            // in place
            return b.rope_tail(&mut s.heads, n_tokens, N_HEAD, N_HEAD_DIM, true, rope);
        }
        let mut after_attn_hc = rows_mut(&mut s.after_attn_hc, 0, n_tokens, HC_DIM)?;
        let hc_split = rows_view(&s.hc_split, n, MIX_HC)?;
        // A one-row batch (the multi-session route) keeps the decode output
        // fusion; the batch projection is sized for prompt rows.
        let mut output_hc_fused = false;
        if n_tokens == 1 {
            // in place
            b.rope_tail(&mut s.heads, 1, N_HEAD, N_HEAD_DIM, true, rope)?;
            b.attention_output_low_q8(
                &mut self.d.attn_low,
                w(&layer.attn_output_a),
                u64::from(GROUP_DIM),
                u64::from(N_LORA_O),
                N_OUT_GROUP,
                &s.heads,
            )?;
            b.matmul_q8_0_hc_expand(
                &mut after_attn_hc,
                &mut self.d.attn_out,
                w(&layer.attn_output_b),
                LOW_DIM,
                embd,
                &self.d.attn_low,
                batch_cur_hc,
                &hc_split,
                N_EMBD,
                N_HC,
            )?;
            output_hc_fused = true;
        }
        // The output projection folds in the inverse rotation when the shape
        // qualifies; otherwise the separate pass runs.
        let mut inv_rope_fused = false;
        if !output_hc_fused {
            inv_rope_fused = b.attention_output_q8_batch_inv_rope(
                &mut s.attn_out,
                &mut s.attn_low,
                w(&layer.attn_output_a),
                w(&layer.attn_output_b),
                u64::from(GROUP_DIM),
                u64::from(N_LORA_O),
                N_OUT_GROUP,
                embd,
                &s.heads,
                n_tokens,
                N_HEAD_DIM,
                rope,
            );
        }
        if !output_hc_fused && !inv_rope_fused {
            // in place
            b.rope_tail(&mut s.heads, n_tokens, N_HEAD, N_HEAD_DIM, true, rope)?;
            b.attention_output_q8_batch(
                &mut s.attn_out,
                &mut s.attn_low,
                w(&layer.attn_output_a),
                w(&layer.attn_output_b),
                u64::from(GROUP_DIM),
                u64::from(N_LORA_O),
                N_OUT_GROUP,
                embd,
                &s.heads,
                n_tokens,
            )?;
        }
        if !output_hc_fused {
            b.hc_expand_split(
                &mut after_attn_hc,
                &s.attn_out,
                batch_cur_hc,
                &hc_split,
                N_EMBD,
                N_HC,
            )?;
        }
        Ok(())
    }

    /// The compressed cache wide prefill attention reads: the F16 mirror
    /// from 128 rows when the layer keeps one
    /// (`rocm_graph_prefill_attn_comp_cache`).
    fn prefill_attn_comp_cache(&self, il: u32, n_tokens: u32) -> Result<(&DeviceTensor, bool)> {
        let cache = &self.layers[il as usize];
        match &cache.attn_comp_f16 {
            Some(mirror) if n_tokens >= 128 => Ok((mirror, true)),
            _ => Ok((cached(&cache.attn_comp)?, false)),
        }
    }

    /// `rocm_graph_encode_layer_ffn_batch`: the FFN half of layer `il` for
    /// `n_tokens` rows of `ws`, hash routing by its prompt tokens; router
    /// `groups` (verification blocks) are optional.
    pub(crate) fn encode_layer_ffn_batch(
        &mut self,
        t: &Target<'_>,
        ws: &mut BatchWorkspace,
        ffn: FfnLayer<'_>,
        pos0: u32,
        n_tokens: u32,
        groups: Option<&[u32]>,
    ) -> Result<()> {
        let _ = pos0;
        let b = t.b;
        if n_tokens == 0 || n_tokens > self.prefill_cap {
            return Err(Error { op: "ffn batch" });
        }
        if let Some(groups) = groups {
            let count = groups.len().saturating_sub(1);
            if count == 0
                || count > 8
                || groups[0] != 0
                || groups[count] != n_tokens
                || groups.windows(2).any(|g| g[1] <= g[0])
            {
                return Err(Error {
                    op: "ffn batch groups",
                });
            }
        }
        let layer = ffn.layer;
        let w = |t: &TensorInfo| ffn.weight(t);
        let shared_dim = layer.ffn_gate_shexp.dim[1];
        let n = u64::from(n_tokens);
        let embd = u64::from(N_EMBD);
        let tokens = &ws.prefill_tokens;
        let mut hc_mix = rows_mut(&mut ws.hc_mix, 0, n_tokens, MIX_HC)?;
        let mut hc_split = rows_mut(&mut ws.hc_split, 0, n_tokens, MIX_HC)?;
        let mut ffn_cur = rows_mut(&mut ws.ffn_cur, 0, n_tokens, embd)?;
        let mut hc_pre_fused = false;
        if n_tokens >= 128 {
            hc_pre_fused = true;
            let fused = b.hc_norm_mix_split_weighted_sum(
                &mut ffn_cur,
                &mut hc_mix,
                &mut hc_split,
                &ws.after_attn_hc,
                w(&layer.hc_ffn_fn),
                w(&layer.hc_ffn_scale),
                w(&layer.hc_ffn_base),
                N_EMBD,
                N_HC,
                n_tokens,
                N_HC_SINKHORN_ITER,
                HC_EPS,
                RMS_EPS,
            )?;
            if !fused {
                return Err(Error { op: "hc norm mix" });
            }
        }
        if !hc_pre_fused {
            let mut flat_hc = ws.stage.flat_hc();
            b.rms_norm_plain_rows(
                &mut flat_hc,
                &ws.after_attn_hc,
                HC_DIM as u32,
                n_tokens,
                RMS_EPS,
            )?;
            b.matmul_f16(
                &mut hc_mix,
                w(&layer.hc_ffn_fn),
                HC_DIM,
                MIX_HC,
                &flat_hc,
                n,
            )?;
            // The decode fusion is row-independent: exact to one launch per
            // row and without a separate normalization.
            b.hc_split_weighted_sum_norm(
                &mut ffn_cur,
                &mut ws.ffn_norm,
                &mut hc_split,
                &hc_mix,
                &ws.after_attn_hc,
                w(&layer.hc_ffn_scale),
                w(&layer.hc_ffn_base),
                w(&layer.ffn_norm),
                N_EMBD,
                N_HC,
                N_HC_SINKHORN_ITER,
                HC_EPS,
                RMS_EPS,
            )?;
        }
        if hc_pre_fused {
            b.rms_norm_weight_rows(
                &mut ws.ffn_norm,
                &ws.ffn_cur,
                w(&layer.ffn_norm),
                N_EMBD,
                n_tokens,
                RMS_EPS,
            )?;
        }
        let hash = ffn.hash;
        let bias = layer.ffn_exp_probs_b.as_ref().map(w);
        let expert = u64::from(N_EXPERT);
        match groups {
            Some(groups) => {
                for pair in groups.windows(2) {
                    let (row0, rows) = (pair[0], pair[1] - pair[0]);
                    let mut logits = rows_mut(&mut ws.router_logits, row0, rows, expert)?;
                    let mut probs = rows_mut(&mut ws.router_probs, row0, rows, expert)?;
                    let norm = ws
                        .ffn_norm
                        .view(u64::from(row0) * embd * F32, u64::from(rows) * embd * F32)
                        .ok_or(Error { op: "router view" })?;
                    matmul_plain(
                        b,
                        &mut logits,
                        &layer.ffn_gate_inp,
                        ffn.model,
                        embd,
                        expert,
                        &norm,
                        u64::from(rows),
                    )?;
                    b.router_select_batch(
                        &mut ws.router,
                        row0,
                        &mut probs,
                        &logits,
                        tokens,
                        row0,
                        bias,
                        hash,
                        rows,
                    )?;
                }
            }
            None => {
                matmul_plain(
                    b,
                    &mut ws.router_logits,
                    &layer.ffn_gate_inp,
                    ffn.model,
                    embd,
                    expert,
                    &ws.ffn_norm,
                    n,
                )?;
                b.router_select_batch(
                    &mut ws.router,
                    0,
                    &mut ws.router_probs,
                    &ws.router_logits,
                    tokens,
                    0,
                    bias,
                    hash,
                    n_tokens,
                )?;
            }
        }
        // The routed MoE may leave its F16 expert rows unsummed for the
        // expansion to fold in; advisory, read back below.
        b.set_routed_defer_sum(n_tokens >= 128);
        check_strides(layer);
        let mut routed = ws.stage.routed();
        b.routed_moe_batch(
            RoutedMoe {
                out: &mut ws.routed_out,
                gate: &mut routed.gate,
                up: &mut routed.up,
                mid: &mut routed.mid,
                down: &mut routed.down,
            },
            &routed_experts(layer, ffn.model),
            &mut ws.router,
            SWIGLU_CLAMP_EXP,
            &ws.ffn_norm,
            n_tokens,
        )?;
        let mut shared = ws.stage.shared();
        let group_count = groups.map_or(0, |g| g.len() - 1);
        if n_tokens <= 136 && (n_tokens <= 16 || group_count != 0) {
            b.shared_gate_up_swiglu_q8_0(
                &mut shared.gate,
                &mut shared.up,
                &mut shared.mid,
                w(&layer.ffn_gate_shexp),
                w(&layer.ffn_up_shexp),
                embd,
                shared_dim,
                &ws.ffn_norm,
                SWIGLU_CLAMP_EXP,
                n_tokens,
            )?;
        } else {
            b.matmul_q8_0(
                &mut shared.gate,
                w(&layer.ffn_gate_shexp),
                embd,
                shared_dim,
                &ws.ffn_norm,
                n,
            )?;
            b.matmul_q8_0(
                &mut shared.up,
                w(&layer.ffn_up_shexp),
                embd,
                shared_dim,
                &ws.ffn_norm,
                n,
            )?;
            b.swiglu(
                &mut shared.mid,
                &shared.gate,
                &shared.up,
                (n * shared_dim) as u32,
                SWIGLU_CLAMP_EXP,
                1.0,
            )?;
        }
        b.matmul_q8_0(
            &mut shared.out,
            w(&layer.ffn_down_shexp),
            shared_dim,
            embd,
            &shared.mid,
            n,
        )?;
        let deferred = b.routed_sum_deferred();
        b.set_routed_defer_sum(false);
        let mut next_hc = rows_mut(&mut ws.hc[1 - self.batch_cur_index], 0, n_tokens, HC_DIM)?;
        if deferred {
            b.hc_expand_add_split_moesum(
                &mut next_hc,
                &shared.routed_down,
                &shared.out,
                &ws.after_attn_hc,
                &hc_split,
                N_EMBD,
                N_HC,
                N_EXPERT_USED,
                n_tokens,
            )
            .inspect_err(|_| eprintln!("ds4: fused routed sum expansion rejected the shape"))
        } else {
            b.hc_expand_add_split(
                &mut next_hc,
                &ws.routed_out,
                &shared.out,
                &ws.after_attn_hc,
                &hc_split,
                N_EMBD,
                N_HC,
            )
        }
    }

    /// `rocm_graph_encode_layer_batch`: attention and FFN halves, then the
    /// stream swap and the DSpark capture of the layer's output.
    pub(crate) fn encode_layer_batch(
        &mut self,
        t: &Target<'_>,
        ws: &mut BatchWorkspace,
        il: u32,
        pos0: u32,
        n_tokens: u32,
    ) -> Result<()> {
        let own = Slots::own(ws, self.batch_cur_index);
        self.encode_layer_attention_batch(t, il, pos0, n_tokens, AttnBatch::default(), own)?;
        self.encode_layer_ffn_batch(t, ws, FfnLayer::target(t, il), pos0, n_tokens, None)?;
        self.batch_cur_index = 1 - self.batch_cur_index;
        self.dspark_capture_batch_layer(t, ws, il, pos0, n_tokens)
    }

    /// `rocm_graph_prefill_layer_major`: `prompt[start..start + n_tokens]`
    /// through every layer, then the head of the last row into `logits`.
    pub(crate) fn prefill_layer_major(
        &mut self,
        t: &Target<'_>,
        ws: &mut BatchWorkspace,
        prompt: &[i32],
        start: u32,
        n_tokens: u32,
        logits: Option<&mut [f32]>,
    ) -> Result<()> {
        let b = t.b;
        if n_tokens == 0
            || n_tokens > self.prefill_cap
            || start as usize > prompt.len()
            || n_tokens as usize > prompt.len() - start as usize
        {
            return Err(Error {
                op: "prefill range",
            });
        }
        // Grow before dispatch, leaving one raw window for decoding.
        for il in 0..N_LAYER {
            self.reserve_compressed_position(
                b,
                il,
                u64::from(start) + u64::from(n_tokens) + u64::from(N_SWA),
            )?;
        }
        let tokens: Vec<u8> = prompt[start as usize..(start + n_tokens) as usize]
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect();
        b.write(&mut ws.prefill_tokens, 0, &tokens)?;
        let result = (|| {
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
            b.begin_commands()?;
            for il in 0..N_LAYER {
                self.encode_layer_batch(t, ws, il, start, n_tokens)?;
            }
            if logits.is_some() {
                let last = row_view(&ws.hc[self.batch_cur_index], n_tokens - 1, HC_DIM)?;
                self.encode_output_head(t, Some(last))?;
            }
            b.end_commands()
        })();
        if result.is_err() {
            if b.synchronize().is_err() {
                eprintln!("ds4: ROCm synchronize after prefill failure also failed");
            }
            return result;
        }
        if let Some(logits) = logits {
            read_logits(b, &self.d.logits, 0, logits)?;
        }
        Ok(())
    }

    /// `rocm_graph_prefill_raw_swa`: a prompt that fits one chunk.
    fn prefill_raw_swa(
        &mut self,
        t: &Target<'_>,
        ws: &mut BatchWorkspace,
        prompt: &[i32],
        logits: Option<&mut [f32]>,
    ) -> Result<()> {
        let rows = prompt.len() as u32;
        if rows == 0 || rows > self.prefill_cap {
            return Err(Error { op: "prefill" });
        }
        self.dspark_resize_capture(t.b, rows)?;
        let mut result = self.prefill_layer_major(t, ws, prompt, 0, rows, logits);
        if result.is_ok() && self.dspark_capturing() {
            result = if self.dspark_capture_ready() {
                self.dspark_inject_rows(t, ws, 0, rows)
            } else {
                Err(Error {
                    op: "dspark capture",
                })
            };
        }
        // Back to one verification block once the prompt seeded the ring.
        if self.dspark_capturing() {
            let block = self.dspark_block_size();
            if let Err(error) = self.dspark_resize_capture(t.b, block + 1) {
                result = Err(error);
            }
        }
        result
    }

    /// `rocm_graph_prefill_chunked_range`: `prompt[start..start + n_tokens]`
    /// in chunks aligned to the absolute prefill-cap boundaries of a cold
    /// prompt, so compression windows finalize on the same schedule; `ws` is
    /// the graph's workspace.
    pub fn prefill_range(
        &mut self,
        t: &Target<'_>,
        ws: &mut BatchWorkspace,
        prompt: &[i32],
        start: u32,
        n_tokens: u32,
        mut logits: Option<&mut [f32]>,
    ) -> Result<()> {
        if n_tokens == 0
            || self.prefill_cap == 0
            || start as usize > prompt.len()
            || n_tokens as usize > prompt.len() - start as usize
        {
            return Err(Error {
                op: "prefill range",
            });
        }
        self.check_workspace(ws)?;
        let mut chunk_cap = self.prefill_cap;
        if start != 0 && chunk_cap > self.raw_cap {
            chunk_cap = self.raw_cap;
        }
        let end = start + n_tokens;
        let mut pos0 = start;
        while pos0 < end {
            let remaining = end - pos0;
            let mut local_cap = chunk_cap;
            if start != 0 {
                let rem = pos0 % self.prefill_cap;
                if rem != 0 {
                    local_cap = local_cap.min(self.prefill_cap - rem);
                }
            }
            let chunk = remaining.min(local_cap);
            let chunk_end = pos0 + chunk;
            let chunk_logits = if chunk_end == end {
                logits.take()
            } else {
                None
            };
            let mut result = self
                .dspark_resize_capture(t.b, chunk)
                .and_then(|()| self.prefill_layer_major(t, ws, prompt, pos0, chunk, chunk_logits));
            if result.is_ok() && self.dspark_capturing() {
                result = if self.dspark_capture_ready() {
                    self.dspark_inject_rows(t, ws, pos0, chunk)
                } else {
                    Err(Error {
                        op: "dspark capture",
                    })
                };
            }
            if let Err(error) = result {
                if self.dspark_capturing() {
                    let block = self.dspark_block_size();
                    let _ = self.dspark_resize_capture(t.b, block + 1);
                }
                if t.b.synchronize().is_err() {
                    eprintln!("ds4: ROCm synchronize after chunked prefill failure also failed");
                }
                return Err(error);
            }
            pos0 = chunk_end;
        }
        if self.dspark_capturing() {
            let block = self.dspark_block_size();
            self.dspark_resize_capture(t.b, block + 1)?;
        }
        Ok(())
    }

    /// `ds4_rocm_graph_prefill`: a whole prompt from position 0 through the
    /// graph's workspace `ws`.
    pub fn prefill(
        &mut self,
        t: &Target<'_>,
        ws: &mut BatchWorkspace,
        prompt: &[i32],
        logits: Option<&mut [f32]>,
    ) -> Result<()> {
        if prompt.is_empty() {
            return Err(Error { op: "prefill" });
        }
        self.check_workspace(ws)?;
        if (self.prefill_cap as usize) < prompt.len() {
            self.prefill_range(t, ws, prompt, 0, prompt.len() as u32, logits)
        } else {
            self.prefill_raw_swa(t, ws, prompt, logits)
        }
    }
}
