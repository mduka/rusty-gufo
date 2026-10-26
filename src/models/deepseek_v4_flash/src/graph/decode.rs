//! Single-token decode (`rocm_graph_encode_decode_layer`,
//! `rocm_graph_encode_output_head`, `rocm_graph_encode_token_raw_swa`,
//! `rocm_graph_eval_token_raw_swa`).

use gufo_deepseek_v4_flash_kernels::{
    Backend, CompressorWeights, DeviceTensor, Error, HashTable, Input, Output, Result,
    RoutedExperts, RoutedMoe, TensorView, TensorViewMut,
};

use super::alloc::LayerCache;
use super::*;
use crate::gguf::TensorInfo;
use crate::shape::tensor_type;
use crate::weights::{LayerWeights, Weights, routed_row_bytes};

/// What the encoders read besides the graph: the backend, the bound target
/// weights and the checked hash tables of the first layers.
pub struct Target<'a> {
    pub b: &'a Backend,
    pub weights: &'a Weights,
    pub hash_tables: &'a [HashTable],
    /// The support model, when the engine has one.
    pub dspark: Option<&'a crate::dspark_model::DsparkModel>,
}

/// `rocm_graph_matmul_plain_tensor`: the dense projections by weight type
/// (F16 and F32 in the target, Q8_0 for the support model's router).
pub(crate) fn matmul_plain(
    b: &Backend,
    out: &mut impl Output,
    weight: &TensorInfo,
    model: gufo_deepseek_v4_flash_kernels::Model,
    in_dim: u64,
    out_dim: u64,
    x: &impl Input,
    n_tok: u64,
) -> Result<()> {
    let w = gufo_deepseek_v4_flash_kernels::Weight {
        model,
        offset: weight.offset,
    };
    match weight.ty {
        tensor_type::F16 => b.matmul_f16(out, w, in_dim, out_dim, x, n_tok),
        tensor_type::F32 => b.matmul_f32(out, w, in_dim, out_dim, x, n_tok),
        tensor_type::Q8_0 => b.matmul_q8_0(out, w, in_dim, out_dim, x, n_tok),
        _ => {
            eprintln!(
                "ds4: ROCm plain matmul does not support {}",
                crate::gguf::type_name(weight.ty)
            );
            Err(Error { op: "plain matmul" })
        }
    }
}

/// The routed experts of a layer as the routed-MoE launcher takes them.
pub(crate) fn routed_experts(
    layer: &LayerWeights,
    model: gufo_deepseek_v4_flash_kernels::Model,
) -> RoutedExperts {
    let w = |t: &TensorInfo| gufo_deepseek_v4_flash_kernels::Weight {
        model,
        offset: t.offset,
    };
    RoutedExperts {
        gate: w(&layer.ffn_gate_exps),
        up: w(&layer.ffn_up_exps),
        down: w(&layer.ffn_down_exps),
        gate_type: layer.ffn_gate_exps.ty,
        down_type: layer.ffn_down_exps.ty,
        in_dim: layer.ffn_gate_exps.dim[0] as u32,
        mid_dim: layer.ffn_down_exps.dim[0] as u32,
        out_dim: layer.ffn_down_exps.dim[1] as u32,
    }
}

/// The strides the C++ graph passes, checked against the wrapper's.
pub(crate) fn check_strides(layer: &LayerWeights) {
    debug_assert_eq!(
        routed_experts(layer, gufo_deepseek_v4_flash_kernels::Model::Target)
            .strides()
            .map(|s| (s.0, s.2)),
        Some((
            routed_row_bytes(&layer.ffn_gate_exps),
            routed_row_bytes(&layer.ffn_down_exps)
        ))
    );
}

/// The compressor weights of an attention or indexer compressor.
pub(crate) fn compressor(
    ape: &Option<TensorInfo>,
    norm: &Option<TensorInfo>,
) -> Result<CompressorWeights> {
    match (ape, norm) {
        (Some(ape), Some(norm)) => Ok(CompressorWeights {
            ape: w(ape),
            norm: w(norm),
        }),
        _ => Err(Error {
            op: "compressor weights",
        }),
    }
}

/// A compressed layer's optional tensor, present by the layout checks.
pub(crate) fn present(t: &Option<TensorInfo>) -> Result<&TensorInfo> {
    t.as_ref().ok_or(Error {
        op: "compressed layer weights",
    })
}

pub(crate) fn cached(t: &Option<DeviceTensor>) -> Result<&DeviceTensor> {
    t.as_ref().ok_or(Error {
        op: "compressed layer cache",
    })
}

pub(crate) fn cached_mut(t: &mut Option<DeviceTensor>) -> Result<&mut DeviceTensor> {
    t.as_mut().ok_or(Error {
        op: "compressed layer cache",
    })
}

/// `1 / sqrt(indexer head dim * indexer heads)`, as C++ computes it in
/// single precision.
pub(crate) fn index_scale() -> f32 {
    1.0f32 / ((N_INDEXER_HEAD_DIM * N_INDEXER_HEAD) as f32).sqrt()
}

impl LayerCache {
    /// Rounds the new compressed row `row` through FP8, mirroring it into
    /// the F16 cache when the layer keeps one
    /// (`rocm_graph_quantize_attn_comp_row`).
    pub(crate) fn quantize_comp_row(&mut self, b: &Backend, row: u32) -> Result<()> {
        let hd = u64::from(N_HEAD_DIM);
        let comp = cached_mut(&mut self.attn_comp)?;
        let mut row_view = comp
            .view_mut(u64::from(row) * hd * F32, hd * F32)
            .ok_or(Error { op: "comp row" })?;
        match &mut self.attn_comp_f16 {
            Some(mirror) => {
                let mut mirror_row =
                    mirror
                        .view_mut(u64::from(row) * hd * 2, hd * 2)
                        .ok_or(Error {
                            op: "comp mirror row",
                        })?;
                // in place
                b.fp8_kv_quantize_mirror_f16(&mut row_view, &mut mirror_row, 1, N_HEAD_DIM, N_ROT)
            }
            // in place
            None => b.fp8_kv_quantize(&mut row_view, 1, N_HEAD_DIM, N_ROT),
        }
    }
}

/// The rows of a multi-session batch that [`Graph::attention_core`] works
/// on instead of the graph's decode tensors.
pub(crate) struct CoreRows<'a> {
    pub attn_norm: TensorViewMut<'a>,
    pub qr_norm: TensorViewMut<'a>,
    pub q: TensorViewMut<'a>,
    pub kv: TensorViewMut<'a>,
    pub heads: TensorViewMut<'a>,
    /// The compressor projections of the row, which the batch computed.
    pub comp: Option<(TensorViewMut<'a>, TensorViewMut<'a>)>,
}

impl Graph {
    /// `rocm_graph_encode_decode_layer`: one layer of a single-token decode
    /// at `pos`.
    pub(crate) fn encode_decode_layer(
        &mut self,
        t: &Target<'_>,
        il: u32,
        pos: u32,
        token: i32,
    ) -> Result<()> {
        let b = t.b;
        self.reserve_compressed_position(b, il, u64::from(pos) + 1)?;
        let layer = &t.weights.layers[il as usize];
        let q_rank = layer.attn_q_a.dim[1];
        let shared_dim = layer.ffn_gate_shexp.dim[1];
        let target = gufo_deepseek_v4_flash_kernels::Model::Target;
        let d = &mut self.d;

        b.rms_norm_plain(&mut d.flat_hc, &d.cur_hc, HC_DIM as u32, RMS_EPS)?;
        matmul_plain(
            b,
            &mut d.hc_mix,
            &layer.hc_attn_fn,
            target,
            HC_DIM,
            MIX_HC,
            &d.flat_hc,
            1,
        )?;
        b.hc_split_weighted_sum_norm(
            &mut d.attn_cur,
            &mut d.attn_norm,
            &mut d.hc_split,
            &d.hc_mix,
            &d.cur_hc,
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
            &mut d.qr,
            w(&layer.attn_q_a),
            u64::from(N_EMBD),
            q_rank,
            &d.attn_norm,
            1,
        )?;
        b.matmul_q8_0(
            &mut d.kv_raw,
            w(&layer.attn_kv),
            u64::from(N_EMBD),
            u64::from(N_HEAD_DIM),
            &d.attn_norm,
            1,
        )?;
        b.qkv_rms_norm_rows(
            &mut d.qr_norm,
            &d.qr,
            w(&layer.attn_q_a_norm),
            q_rank as u32,
            &mut d.kv,
            &d.kv_raw,
            w(&layer.attn_kv_a_norm),
            N_HEAD_DIM,
            1,
            RMS_EPS,
        )?;
        b.matmul_q8_0(&mut d.q, w(&layer.attn_q_b), q_rank, Q_DIM, &d.qr_norm, 1)?;
        // in place
        b.head_rms_norm(&mut d.q, 1, N_HEAD, N_HEAD_DIM, RMS_EPS)?;
        self.attention_core(t, il, pos, None)?;
        let d = &mut self.d;
        b.attention_output_low_q8(
            &mut d.attn_low,
            w(&layer.attn_output_a),
            u64::from(GROUP_DIM),
            u64::from(N_LORA_O),
            N_OUT_GROUP,
            &d.heads,
        )?;
        b.matmul_q8_0_hc_expand(
            &mut d.after_attn_hc,
            &mut d.attn_out,
            w(&layer.attn_output_b),
            LOW_DIM,
            u64::from(N_EMBD),
            &d.attn_low,
            &d.cur_hc,
            &d.hc_split,
            N_EMBD,
            N_HC,
        )?;

        b.rms_norm_plain(&mut d.flat_hc, &d.after_attn_hc, HC_DIM as u32, RMS_EPS)?;
        matmul_plain(
            b,
            &mut d.hc_mix,
            &layer.hc_ffn_fn,
            target,
            HC_DIM,
            MIX_HC,
            &d.flat_hc,
            1,
        )?;
        b.hc_split_weighted_sum_norm(
            &mut d.ffn_cur,
            &mut d.ffn_norm,
            &mut d.hc_split,
            &d.hc_mix,
            &d.after_attn_hc,
            w(&layer.hc_ffn_scale),
            w(&layer.hc_ffn_base),
            w(&layer.ffn_norm),
            N_EMBD,
            N_HC,
            N_HC_SINKHORN_ITER,
            HC_EPS,
            RMS_EPS,
        )?;
        check_strides(layer);
        matmul_plain(
            b,
            &mut d.router_logits,
            &layer.ffn_gate_inp,
            target,
            u64::from(N_EMBD),
            u64::from(N_EXPERT),
            &d.ffn_norm,
            1,
        )?;
        let hash = (il < N_HASH_LAYER).then(|| &t.hash_tables[il as usize]);
        b.router_select(
            &mut d.router,
            &mut d.router_probs,
            &d.router_logits,
            layer.ffn_exp_probs_b.as_ref().map(w),
            hash,
            token as u32,
        )?;
        b.routed_moe_one(
            RoutedMoe {
                out: &mut d.routed_out,
                gate: &mut d.routed_gate,
                up: &mut d.routed_up,
                mid: &mut d.routed_mid,
                down: &mut d.routed_down,
            },
            &routed_experts(layer, target),
            &mut d.router,
            SWIGLU_CLAMP_EXP,
            &d.ffn_norm,
        )?;
        b.shared_gate_up_swiglu_q8_0(
            &mut d.shared_gate,
            &mut d.shared_up,
            &mut d.shared_mid,
            w(&layer.ffn_gate_shexp),
            w(&layer.ffn_up_shexp),
            u64::from(N_EMBD),
            shared_dim,
            &d.ffn_norm,
            SWIGLU_CLAMP_EXP,
            1,
        )?;
        b.shared_down_hc_expand_q8_0(
            &mut d.after_ffn_hc,
            &mut d.shared_out,
            w(&layer.ffn_down_shexp),
            shared_dim,
            u64::from(N_EMBD),
            &d.shared_mid,
            &d.routed_out,
            &d.after_attn_hc,
            &d.hc_split,
            N_EMBD,
            N_HC,
        )
    }

    /// `rocm_graph_encode_session_attention_core`: the stateful middle of
    /// one decode row at `pos` (rotations, the raw cache store, compressor
    /// and indexer updates, attention, the inverse rotation of the heads)
    /// on the graph's decode tensors, or on `rows` of a multi-session batch,
    /// whose dense front end and output projection are shared.
    pub(crate) fn attention_core(
        &mut self,
        t: &Target<'_>,
        il: u32,
        pos: u32,
        rows: Option<CoreRows<'_>>,
    ) -> Result<()> {
        let b = t.b;
        if self.raw_cap == 0 {
            return Err(Error { op: "raw cache" });
        }
        self.reserve_compressed_position(b, il, u64::from(pos) + 1)?;
        let r = ratio(il);
        let emit = r != 0 && (pos + 1).is_multiple_of(r);
        // The score scratch the indexer below needs, grown before the rows
        // borrow the decode tensors; past the capacity the check below
        // refuses the row.
        if r == 4 {
            let cache = &self.layers[il as usize];
            let index_rows = cache.n_index_comp + u32::from(emit);
            if cache.n_comp + u32::from(emit) > N_INDEXER_TOP_K
                && index_rows > N_INDEXER_TOP_K
                && index_rows <= cache.comp_cap
            {
                self.reserve_indexer_scores(b, index_rows, 1)?;
            }
        }
        let raw_cap = self.raw_cap;
        let raw_window = self.raw_window;
        let raw_row = pos % raw_cap;
        let n_raw = self.raw_span_for_batch(pos, 1);
        let raw_start = self.raw_start_for_span(pos, n_raw);
        let Graph { d, layers, .. } = self;
        let mut rows = match rows {
            Some(rows) => rows,
            None => CoreRows {
                attn_norm: (&mut d.attn_norm).into(),
                qr_norm: (&mut d.qr_norm).into(),
                q: (&mut d.q).into(),
                kv: (&mut d.kv).into(),
                heads: (&mut d.heads).into(),
                comp: None,
            },
        };
        let layer = &t.weights.layers[il as usize];
        let q_rank = layer.attn_q_a.dim[1];
        let rope = rope(il, pos);
        let cache = &mut layers[il as usize];

        // in place
        b.rope_tail(&mut rows.q, 1, N_HEAD, N_HEAD_DIM, false, rope)?;
        // in place
        b.rope_tail(&mut rows.kv, 1, N_HEAD_KV, N_HEAD_DIM, false, rope)?;
        // in place, then stored
        b.kv_fp8_store_raw(
            &mut rows.kv,
            &mut cache.raw,
            raw_cap,
            raw_row,
            N_HEAD_DIM,
            N_ROT,
        )?;

        let mut n_comp = 0;
        let mut n_selected = 0;
        if r != 0 {
            let coff = if r == 4 { 2 } else { 1 };
            let comp_width = coff * N_HEAD_DIM;
            if emit && cache.n_comp >= cache.comp_cap {
                eprintln!("ds4: ROCm graph compressed KV cache capacity exceeded at layer {il}");
                return Err(Error {
                    op: "compressed capacity",
                });
            }
            let (comp_kv, comp_sc) = match &rows.comp {
                Some((kv, sc)) => (kv.as_view(), sc.as_view()),
                None => {
                    b.matmul_f16_pair(
                        &mut d.comp_kv_cur,
                        &mut d.comp_sc_cur,
                        w(present(&layer.attn_compressor_kv)?),
                        w(present(&layer.attn_compressor_gate)?),
                        u64::from(N_EMBD),
                        u64::from(comp_width),
                        &rows.attn_norm,
                        1,
                    )?;
                    (d.comp_kv_cur.whole(), d.comp_sc_cur.whole())
                }
            };
            let comp_row = cache.n_comp;
            // in place: the compressor state
            b.compressor_update(
                &comp_kv,
                &comp_sc,
                cached_mut(&mut cache.attn_state_kv)?,
                cached_mut(&mut cache.attn_state_score)?,
                cached_mut(&mut cache.attn_comp)?,
                compressor(&layer.attn_compressor_ape, &layer.attn_compressor_norm)?,
                N_HEAD_DIM,
                r,
                pos,
                comp_row,
                rope,
                RMS_EPS,
                false,
            )?;
            if emit {
                cache.quantize_comp_row(b, comp_row)?;
                cache.n_comp += 1;
            }
            if r == 4 {
                let index_width = coff * N_INDEXER_HEAD_DIM;
                if emit && cache.n_index_comp >= cache.comp_cap {
                    eprintln!(
                        "ds4: ROCm graph indexer compressed KV cache capacity exceeded at layer {il}"
                    );
                    return Err(Error {
                        op: "indexer capacity",
                    });
                }
                b.matmul_f16_pair(
                    &mut d.comp_kv_cur,
                    &mut d.comp_sc_cur,
                    w(present(&layer.indexer_compressor_kv)?),
                    w(present(&layer.indexer_compressor_gate)?),
                    u64::from(N_EMBD),
                    u64::from(index_width),
                    &rows.attn_norm,
                    1,
                )?;
                let index_row = cache.n_index_comp;
                // in place: the compressor state
                b.compressor_update(
                    &d.comp_kv_cur,
                    &d.comp_sc_cur,
                    cached_mut(&mut cache.index_state_kv)?,
                    cached_mut(&mut cache.index_state_score)?,
                    cached_mut(&mut cache.index_comp)?,
                    compressor(
                        &layer.indexer_compressor_ape,
                        &layer.indexer_compressor_norm,
                    )?,
                    N_INDEXER_HEAD_DIM,
                    r,
                    pos,
                    index_row,
                    rope,
                    RMS_EPS,
                    false,
                )?;
                if emit {
                    let width = u64::from(N_INDEXER_HEAD_DIM) * F32;
                    let mut row = cached_mut(&mut cache.index_comp)?
                        .view_mut(u64::from(index_row) * width, width)
                        .ok_or(Error { op: "index row" })?;
                    // in place
                    b.indexer_qat(&mut row, 1, N_INDEXER_HEAD_DIM)?;
                    cache.n_index_comp += 1;
                }
                let index_rows = cache.n_index_comp;
                if cache.n_comp > N_INDEXER_TOP_K && index_rows > N_INDEXER_TOP_K {
                    b.matmul_f16(
                        &mut d.indexer_q,
                        w(present(&layer.indexer_attn_q_b)?),
                        q_rank,
                        INDEXER_Q_DIM,
                        &rows.qr_norm,
                        1,
                    )?;
                    // in place
                    b.rope_tail(
                        &mut d.indexer_q,
                        1,
                        N_INDEXER_HEAD,
                        N_INDEXER_HEAD_DIM,
                        false,
                        rope,
                    )?;
                    // in place
                    b.indexer_qat(&mut d.indexer_q, N_INDEXER_HEAD, N_INDEXER_HEAD_DIM)?;
                    b.matmul_f16(
                        &mut d.indexer_weights,
                        w(present(&layer.indexer_proj)?),
                        u64::from(N_EMBD),
                        u64::from(N_INDEXER_HEAD),
                        &rows.attn_norm,
                        1,
                    )?;
                    // Reserved above.
                    b.indexer_score_one(
                        &mut d.indexer_scores,
                        &d.indexer_q,
                        &d.indexer_weights,
                        cached(&cache.index_comp)?,
                        index_rows,
                        N_INDEXER_HEAD,
                        N_INDEXER_HEAD_DIM,
                        index_scale(),
                    )?;
                    b.indexer_topk(
                        &mut d.comp_selected,
                        &d.indexer_scores,
                        index_rows,
                        1,
                        N_INDEXER_TOP_K,
                    )?;
                    n_selected = N_INDEXER_TOP_K.min(index_rows);
                }
            }
            n_comp = cache.n_comp;
        }

        if n_comp != 0 && n_selected != 0 {
            b.attention_indexed_mixed_batch_heads(
                &mut rows.heads,
                w(&layer.attn_sinks),
                &rows.q,
                &cache.raw,
                cached(&cache.attn_comp)?,
                false,
                &d.comp_selected,
                1,
                pos,
                n_raw,
                raw_cap,
                raw_start,
                n_comp,
                n_selected,
                raw_window,
                r,
                N_HEAD,
                N_HEAD_DIM,
            )?;
        } else {
            let comp = if n_comp != 0 {
                Some(cached(&cache.attn_comp)? as &dyn Input)
            } else {
                None
            };
            b.attention_decode_heads(
                &mut rows.heads,
                w(&layer.attn_sinks),
                &rows.q,
                &cache.raw,
                n_raw,
                raw_cap,
                raw_start,
                comp,
                n_comp,
                None,
                N_HEAD,
                N_HEAD_DIM,
            )?;
        }
        // in place
        b.rope_tail(&mut rows.heads, 1, N_HEAD, N_HEAD_DIM, true, rope)
    }

    /// `rocm_graph_encode_output_head`: the stream collapse, output norm and
    /// vocabulary projection of the streams `hc`, or of the decode streams,
    /// into `logits`.
    pub(crate) fn encode_output_head(
        &mut self,
        t: &Target<'_>,
        hc: Option<TensorView<'_>>,
    ) -> Result<()> {
        let b = t.b;
        let wt = t.weights;
        let d = &mut self.d;
        let hc = match hc {
            Some(hc) => hc,
            None => d.cur_hc.whole(),
        };
        b.rms_norm_plain(&mut d.flat_hc, &hc, HC_DIM as u32, RMS_EPS)?;
        b.matmul_f16(
            &mut d.output_pre,
            w(&wt.output_hc_fn),
            HC_DIM,
            u64::from(N_HC),
            &d.flat_hc,
            1,
        )?;
        b.output_hc_weights(
            &mut d.output_weights,
            &d.output_pre,
            w(&wt.output_hc_scale),
            w(&wt.output_hc_base),
            N_HC,
            HC_EPS,
        )?;
        b.hc_weighted_sum(&mut d.output_embd, &hc, &d.output_weights, N_EMBD, N_HC)?;
        b.rms_norm_weight(
            &mut d.output_norm,
            &d.output_embd,
            w(&wt.output_norm),
            N_EMBD,
            RMS_EPS,
        )?;
        b.matmul_q8_0(
            &mut d.logits,
            w(&wt.output),
            u64::from(N_EMBD),
            wt.output.dim[1],
            &d.output_norm,
            1,
        )
    }

    /// `rocm_graph_encode_token_raw_swa`: embeds `token`, runs every layer
    /// and the head when `need_logits`. The C++ graph waited for the device
    /// after the fourth layer (a command-buffer commit on Metal, a stall on
    /// HIP, whose launches start at once); the step runs through instead.
    pub(crate) fn encode_token(
        &mut self,
        t: &Target<'_>,
        token: i32,
        pos: u32,
        need_logits: bool,
    ) -> Result<()> {
        let b = t.b;
        if self.raw_cap == 0 {
            eprintln!("ds4: ROCm graph raw KV cache is not allocated");
            return Err(Error { op: "raw cache" });
        }
        let embd = &t.weights.token_embd;
        b.embed_token_hc(
            &mut self.d.cur_hc,
            w(embd),
            embd.dim[1] as u32,
            token as u32,
            N_EMBD,
            N_HC,
        )?;
        for il in 0..N_LAYER {
            let result = self.encode_decode_layer(t, il, pos, token);
            std::mem::swap(&mut self.d.cur_hc, &mut self.d.after_ffn_hc);
            result?;
            // Captured after the swap, so cur_hc holds this layer's output.
            self.dspark_capture_decode_layer(t, il)?;
        }
        if need_logits {
            self.encode_output_head(t, None)?;
        }
        Ok(())
    }

    /// `rocm_graph_eval_token_raw_swa`: one decode step, its logits read
    /// into `logits` when given.
    pub fn eval_token(
        &mut self,
        t: &Target<'_>,
        token: i32,
        pos: u32,
        logits: Option<&mut [f32]>,
    ) -> Result<()> {
        let b = t.b;
        let result = (|| {
            b.begin_commands()?;
            self.encode_token(t, token, pos, logits.is_some())?;
            // The copy follows the output head in stream order: one wait
            // covers the step and the copy.
            if let Some(logits) = logits {
                read_logits(b, &self.d.logits, 0, logits)?;
            }
            b.end_commands()
        })();
        if result.is_err() && b.synchronize().is_err() {
            eprintln!("ds4: ROCm synchronize after graph eval failure also failed");
        }
        result
    }
}

/// Reads the vocabulary row of `logits` into `out`.
pub(crate) fn read_logits(
    b: &Backend,
    logits: &impl Input,
    row: u32,
    out: &mut [f32],
) -> Result<()> {
    let vocab = N_VOCAB as usize;
    let out = out.get_mut(..vocab).ok_or(Error { op: "read logits" })?;
    b.read_f32(logits, u64::from(row) * vocab as u64 * F32, out)
}
