//! One chunk of prompt rows, a port of the extension's
//! `QwenGpuExecutor::ForwardPromptChunk` (`hip/prefill_chunk.cpp`) for an MoE
//! model of the formats the weights accept: quantized projections read the
//! tiled Q8_1 activation (`routes::reads_q8_activation`), BF16 ones the FP32
//! rows, and F32 ones (the UD files' DeltaNet alpha/beta) take hipBLAS. The
//! FFN is the routed MoE of [`crate::moe::prefill`]. The 27B's FP16 prefill
//! route never applies to an MoE shape and is not part of this port.

use gufo_hip::blas::Blas;
use gufo_hip::{
    Bf16, DeviceBuffer, DeviceCopy, DeviceSlice, DeviceSliceMut, GgmlType, HipError, Launch,
    LaunchError, PinnedDownload, QuantTensor, Stream,
};
use gufo_qwen35moe_kernels as k;
use gufo_qwen35moe_kernels::sizes;

use crate::arena::{Forward, Scratch, SessionBuffers};
use crate::config::NORM_EPS;
use crate::moe::{bf16_staging, bf16_wmma_projection, blas_gemm, f32_weights};
use crate::routes::{
    bf16_weights, fused_norm_quantize_supported, fused_ssm_epilogue_supported, reads_q8_activation,
};
use crate::weights::{Mixer, RecurrentWeights};

/// Why a prefill chunk stopped.
#[derive(Debug)]
pub(crate) enum PrefillError {
    Launch(LaunchError),
    /// Recording or querying a backpressure event.
    Device(HipError),
    /// The request's cancellation check fired between layers.
    Cancelled,
}

impl From<LaunchError> for PrefillError {
    fn from(error: LaunchError) -> Self {
        Self::Launch(error)
    }
}

impl From<HipError> for PrefillError {
    fn from(error: HipError) -> Self {
        Self::Device(error)
    }
}

/// `y[batch][m] = rows[batch][k] w^T` by the route C++ resolves
/// (`gemm_weight`): the tiled Q8_1 activation `q8` for the quantized
/// formats, the FP32 rows `fp32` for BF16 (the WMMA GEMM from 128 rows on an
/// MoE model, else the exact kernel) and F32 (hipBLAS).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm(
    s: &Stream,
    blas: &Blas,
    q8: &DeviceBuffer<u8>,
    batch: usize,
    w: &QuantTensor,
    fp32: DeviceSlice<'_, f32>,
    y: DeviceSliceMut<'_, f32>,
    m: usize,
    k: usize,
) -> Result<(), LaunchError> {
    if reads_q8_activation(w.format()) {
        return k::batched_quant_gemm_q8_1(
            s,
            w.view(),
            q8.slice(0..sizes::q8_activation_gemm_bytes(batch, k)),
            y,
            batch,
            m,
            k,
        );
    }
    match w.format() {
        GgmlType::Bf16
            if batch >= sizes::BF16_WMMA_MIN_PREFILL_BATCH && sizes::grouped_bf16_gemm(m, k, 1) =>
        {
            k::bf16_wmma_gemm(s, bf16_weights(w)?, fp32, y, batch, m, k)
        }
        GgmlType::Bf16 => k::exact_bf16_gemm_f32(s, bf16_weights(w)?, fp32, y, batch, m, k),
        GgmlType::F32 => blas_gemm(s, blas, w, fp32, y, batch, m, k),
        _ => Err(LaunchError::InvalidArgument {
            param: "weights",
            reason: "the Rust Qwen prefill takes Q8_0, K-quant, IQ, BF16 and F32 projections",
        }),
    }
}

/// The unfused RMSNorm (`LaunchBatchedRMSNorm` with its BF16 copy): FP32
/// rows in `x.normed`, BF16 rows in the staging scratch, and with
/// `quantize` the tiled Q8_1 activation of that copy.
fn unfused_norm(
    s: &Stream,
    x: &mut Scratch,
    weight: DeviceSlice<'_, f32>,
    batch: usize,
    hidden: usize,
    quantize: bool,
) -> Result<(), LaunchError> {
    let staging = bf16_staging(&mut x.scratch16, batch * hidden)?;
    k::batched_rms_norm_bf16(
        s,
        x.hidden.slice(0..batch * hidden),
        weight,
        x.normed.slice_mut(0..batch * hidden),
        staging,
        batch,
        hidden,
        NORM_EPS,
    )?;
    if quantize {
        let staging = bf16_staging(&mut x.scratch16, batch * hidden)?;
        k::quantize_q8_1(
            s,
            staging.as_const(),
            x.q8.slice_mut(0..sizes::q8_activation_bytes(batch, hidden)),
            batch,
            hidden,
        )?;
    }
    Ok(())
}

/// The tiled Q8_1 activation of the `x.ssm_out` rows (attention context or
/// gated recurrence output) a projection of format `format` reads:
/// quantized straight from FP32 for the Q8_1 routes, else through the BF16
/// staging copy (as C++, which quantizes for every format).
fn quantize_rows(
    s: &Stream,
    x: &mut Scratch,
    format: GgmlType,
    batch: usize,
    k: usize,
) -> Result<(), LaunchError> {
    let source = x.ssm_out.slice(0..batch * k);
    if reads_q8_activation(format) {
        return k::quantize_q8_1_from_f32(
            s,
            source,
            x.q8.slice_mut(0..sizes::q8_activation_bytes(batch, k)),
            batch,
            k,
        );
    }
    let staging = bf16_staging(&mut x.scratch16, batch * k)?;
    k::float_to_bf16(s, source, staging, batch * k)?;
    let staging = bf16_staging(&mut x.scratch16, batch * k)?;
    k::quantize_q8_1(
        s,
        staging.as_const(),
        x.q8.slice_mut(0..sizes::q8_activation_bytes(batch, k)),
        batch,
        k,
    )
}

/// GDN alpha and beta as one `[batch, 2 * rank]` BF16 WMMA GEMM when the two
/// F32 matrices are adjacent in the weight storage, the stacked rows then
/// split into the two control buffers (`LaunchStackedSsmControls`). Returns
/// false, launching nothing, when the layout, the chunk or the scratch does
/// not allow it; `tiled` reads the fused norm's tiled BF16 activation.
fn stacked_controls(
    s: &Stream,
    x: &mut Scratch,
    r: &RecurrentWeights,
    batch: usize,
    rank: usize,
    hidden: usize,
    tiled: bool,
) -> Result<bool, LaunchError> {
    if r.alpha.format() != GgmlType::F32 || r.beta.format() != GgmlType::F32 {
        return Ok(false);
    }
    let Some(joined) = r.alpha.join(&r.beta) else {
        return Ok(false);
    };
    // The BF16 weight scratch holds the converted weights, then the stacked
    // output 64-byte aligned.
    let rows = 2 * rank;
    let weight_bytes = rows * hidden * 2;
    let out_offset = weight_bytes.next_multiple_of(64);
    let out_bytes = batch * rows * 4;
    if x.packed.len() * 4 < out_offset + out_bytes
        || !bf16_wmma_projection(batch, rows, hidden, weight_bytes)
    {
        return Ok(false);
    }
    let invalid = || LaunchError::InvalidArgument {
        param: "packed",
        reason: "the weight scratch holds the BF16 controls",
    };
    let (base, stacked) = x.packed.as_mut_slice().split_at(out_offset / 4);
    let mut base = base
        .range(0..rows * hidden / 2)
        .cast::<Bf16>()
        .ok_or_else(invalid)?;
    let mut stacked = stacked.range(0..batch * rows);
    k::float_to_bf16(
        s,
        f32_weights(&joined)?.range(0..rows * hidden),
        base.reborrow(),
        rows * hidden,
    )?;
    if tiled {
        k::bf16_wmma_gemm_tiled(
            s,
            base.as_const(),
            x.moe.x_tiled.slice(0..batch * hidden),
            stacked.reborrow(),
            batch,
            rows,
            hidden,
        )?;
    } else {
        k::bf16_wmma_gemm(
            s,
            base.as_const(),
            x.normed.slice(0..batch * hidden),
            stacked.reborrow(),
            batch,
            rows,
            hidden,
        )?;
    }
    let stacked = stacked.as_const();
    s.copy_2d(
        stacked,
        rows,
        x.alpha.slice_mut(0..batch * rank),
        rank,
        rank,
        batch,
    )?;
    s.copy_2d(
        stacked.range(rank..batch * rows),
        rows,
        x.beta.slice_mut(0..batch * rank),
        rank,
        rank,
        batch,
    )?;
    Ok(true)
}

/// Runs `batch` prompt rows whose token IDs are in `x.tokens` at positions
/// `start_pos..`; with `compute_logits`, leaves the logits of the last row
/// and its greedy token in `st.logits` and `x.sampled`. `capture` downloads the
/// hidden rows after each tapped layer as `[row][tap][hidden]`, or without
/// taps each row's final hidden state, from element `offset` of its pinned
/// buffer and without waiting (the reader waits once). With a
/// cancellation check on `st`, each layer first passes
/// `SessionBuffers::prefill_layer_checkpoint`.
pub(crate) fn prefill_chunk(
    f: Forward<'_>,
    x: &mut Scratch,
    st: &mut SessionBuffers,
    batch: usize,
    start_pos: u32,
    compute_logits: bool,
    mut capture: Option<(&[u32], &mut PinnedDownload<'_, f32>, usize)>,
) -> Result<(), PrefillError> {
    // One recurrent-state pass per chunk.
    st.recurrent.begin_write();
    let (s, w, max_context) = (f.stream, f.weights, f.max_context);
    let c = &w.config;
    let hidden = c.hidden_size as usize;
    let attention = c.attention_size() as usize;
    let kv = c.kv_size() as usize;
    let qkv = c.ssm_qkv_size() as usize;
    let inner = c.ssm_inner_size as usize;
    let rank = c.ssm_time_step_rank as usize;
    let vocab = c.vocab_size as usize;
    let kv_plane = st.kv.len() / 2;
    let q8_bytes = |k: usize| sizes::q8_activation_bytes(batch, k);

    k::batched_embedding_lookup(
        s,
        w.token_embd.view(),
        x.tokens.slice(0..batch),
        x.hidden.slice_mut(0..batch * hidden),
        batch,
        hidden,
        vocab,
    )?;
    st.vision.inject(
        s,
        x.hidden.slice_mut(0..batch * hidden),
        start_pos,
        batch as u32,
        hidden as u32,
        1,
    )?;

    // `IsFusedRMSNormQuantizeQ8_1Supported` and the 2048-wide fused residual
    // norms (`IsFusedResidualRMSNormQuantizeQ8_1Supported`).
    let fusable_norm = fused_norm_quantize_supported(hidden);
    let fused_residual_norm = hidden == sizes::FUSED_RESIDUAL_NORM_DIM;
    for (l, layer) in (0u32..).zip(&w.layers) {
        st.prefill_layer_checkpoint(s, l)?;
        let reads_q8 = |w: &QuantTensor| reads_q8_activation(w.format());
        let mixer_inputs: [&QuantTensor; 4] = match &layer.mixer {
            Mixer::Attention(at) => [&at.q, &at.k, &at.v, &at.v],
            Mixer::Recurrent(r) => [&r.qkv, &r.gate, &r.alpha, &r.beta],
        };
        let recurrent = matches!(layer.mixer, Mixer::Recurrent(_));
        // Where every reader of a norm takes Q8_1, the norm writes it
        // directly. A DeltaNet layer always quantizes its normed row and the
        // MoE FFN always quantizes the FFN-normed row for the shared expert;
        // 2048-wide rows get that activation from the norm itself (with the
        // preceding residual add for the FFN norm), together with the FP32
        // rows, their BF16 copy and the tiled BF16 copy the router and the
        // stacked alpha/beta GEMMs read.
        let norm_feeds_q8_only = fusable_norm && mixer_inputs.iter().all(|w| reads_q8(w));
        let attn_norm_quantized = fused_residual_norm && !norm_feeds_q8_only && recurrent;
        let ffn_norm_quantized = fused_residual_norm;

        if norm_feeds_q8_only {
            k::batched_rms_norm_quantize_q8_1(
                s,
                x.hidden.slice(0..batch * hidden),
                layer.attn_norm.view(),
                x.q8.slice_mut(0..q8_bytes(hidden)),
                batch,
                hidden,
                NORM_EPS,
            )?;
        } else if attn_norm_quantized {
            let staging = bf16_staging(&mut x.scratch16, batch * hidden)?;
            k::batched_rms_norm_quantize_q8_1_bf16(
                s,
                x.hidden.slice(0..batch * hidden),
                layer.attn_norm.view(),
                x.normed.slice_mut(0..batch * hidden),
                staging,
                x.q8.slice_mut(0..q8_bytes(hidden)),
                batch,
                hidden,
                NORM_EPS,
                Some(x.moe.x_tiled.slice_mut(0..batch * hidden)),
            )?;
        } else {
            // A DeltaNet layer quantizes the BF16 copy always, an attention
            // layer when one of its projections reads it.
            let quantize = recurrent || mixer_inputs.iter().any(|w| reads_q8(w));
            unfused_norm(s, x, layer.attn_norm.view(), batch, hidden, quantize)?;
        }

        match &layer.mixer {
            Mixer::Attention(at) => {
                let attention_layer = l / c.full_attention_interval;
                let normed = x.normed.slice(0..batch * hidden);
                gemm(
                    s,
                    &x.blas,
                    &x.q8,
                    batch,
                    &at.q,
                    normed,
                    x.ssm_qkv.slice_mut(0..batch * 2 * attention),
                    2 * attention,
                    hidden,
                )?;
                gemm(
                    s,
                    &x.blas,
                    &x.q8,
                    batch,
                    &at.k,
                    normed,
                    x.k.slice_mut(0..batch * kv),
                    kv,
                    hidden,
                )?;
                gemm(
                    s,
                    &x.blas,
                    &x.q8,
                    batch,
                    &at.v,
                    normed,
                    x.v.slice_mut(0..batch * kv),
                    kv,
                    hidden,
                )?;
                k::batched_unpack_qg(
                    s,
                    x.ssm_qkv.slice(0..batch * 2 * attention),
                    x.q.slice_mut(0..batch * attention),
                    x.ssm_gate.slice_mut(0..batch * attention),
                    batch,
                    c.num_attention_heads,
                    c.head_dim,
                )?;
                let (k_cache, v_cache) = st.kv.as_mut_slice().split_at(kv_plane);
                k::batched_fused_qk_norm_rope_kv_write(
                    s,
                    x.v.slice(0..batch * kv),
                    at.q_norm.view(),
                    at.k_norm.view(),
                    x.q.slice_mut(0..batch * attention),
                    x.k.slice_mut(0..batch * kv),
                    k_cache,
                    v_cache,
                    attention_layer,
                    start_pos,
                    batch,
                    max_context,
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    c.head_dim,
                    c.rotary_dim,
                    c.rope_theta,
                    NORM_EPS,
                    st.vision.rope(),
                )?;
                // The packed KV workspaces are the full-batch FFN rows, as
                // in C++: their size decides how many heads a pass packs.
                let (packed_k_len, packed_v_len) = (x.ffn_gate.len(), x.ffn_up.len());
                let launched = k::wmma_attention(
                    s,
                    x.q.slice(0..batch * attention),
                    x.k.slice(0..batch * kv),
                    x.v.slice(0..batch * kv),
                    x.ssm_gate.slice(0..batch * attention),
                    st.kv.slice(0..kv_plane),
                    st.kv.slice(kv_plane..2 * kv_plane),
                    x.ssm_out.slice_mut(0..batch * attention),
                    attention_layer,
                    start_pos,
                    batch,
                    max_context,
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    c.head_dim,
                    x.ffn_gate.as_mut_slice(),
                    packed_k_len,
                    x.ffn_up.as_mut_slice(),
                    packed_v_len,
                    0,
                )?;
                if launched == Launch::Unsupported {
                    // Other head shapes: the baseline attention (the tiled
                    // fallback of C++ takes 24/4 heads only, which the WMMA
                    // kernel never declines with FP16 caches). The fused
                    // QK-norm kernel already wrote this chunk's K and V.
                    // Past the scores a workgroup holds in LDS, 256-element
                    // GQA heads take the bounded online arithmetic.
                    let (k_cache, v_cache) = st.kv.as_mut_slice().split_at(kv_plane);
                    let launch = if start_pos as usize + batch > 16_128 {
                        k::batched_attention_long
                    } else {
                        k::batched_attention
                    };
                    launch(
                        s,
                        x.q.slice(0..batch * attention),
                        x.k.slice(0..batch * kv),
                        x.v.slice(0..batch * kv),
                        x.ssm_gate.slice(0..batch * attention),
                        k_cache,
                        v_cache,
                        x.ssm_out.slice_mut(0..batch * attention),
                        attention_layer,
                        start_pos,
                        batch,
                        max_context,
                        c.num_attention_heads,
                        c.num_key_value_heads,
                        c.head_dim,
                        true,
                    )?;
                }
                quantize_rows(s, x, at.output.format(), batch, attention)?;
                gemm(
                    s,
                    &x.blas,
                    &x.q8,
                    batch,
                    &at.output,
                    x.ssm_out.slice(0..batch * attention),
                    x.attn_out.slice_mut(0..batch * hidden),
                    hidden,
                    attention,
                )?;
            }
            Mixer::Recurrent(r) => {
                // On the row-split route a Q8_0 qkv projection applies the
                // causal convolution in its own epilogue, so the FP32
                // projection is neither written nor read back (a prefill
                // never captures replay rows).
                let conv_fused = r.qkv.format() == GgmlType::Q8_0
                    && k::quant_gemm_ssm_conv(
                        s,
                        r.qkv.view(),
                        x.q8.slice(0..sizes::q8_activation_gemm_bytes(batch, hidden)),
                        x.ssm_qkv.slice_mut(0..batch * qkv),
                        r.conv1d.view(),
                        x.conv_out.slice_mut(0..batch * qkv),
                        batch,
                        qkv,
                        hidden,
                    )? == Launch::Launched;
                let normed = x.normed.slice(0..batch * hidden);
                if !conv_fused {
                    gemm(
                        s,
                        &x.blas,
                        &x.q8,
                        batch,
                        &r.qkv,
                        normed,
                        x.ssm_qkv.slice_mut(0..batch * qkv),
                        qkv,
                        hidden,
                    )?;
                }
                gemm(
                    s,
                    &x.blas,
                    &x.q8,
                    batch,
                    &r.gate,
                    normed,
                    x.ssm_gate.slice_mut(0..batch * inner),
                    inner,
                    hidden,
                )?;
                if !stacked_controls(s, x, r, batch, rank, hidden, attn_norm_quantized)? {
                    let normed = x.normed.slice(0..batch * hidden);
                    gemm(
                        s,
                        &x.blas,
                        &x.q8,
                        batch,
                        &r.alpha,
                        normed,
                        x.alpha.slice_mut(0..batch * rank),
                        rank,
                        hidden,
                    )?;
                    gemm(
                        s,
                        &x.blas,
                        &x.q8,
                        batch,
                        &r.beta,
                        normed,
                        x.beta.slice_mut(0..batch * rank),
                        rank,
                        hidden,
                    )?;
                }
                // The row-split epilogue writes the Q8_1 activation ssm_out
                // reads when the head shape allows it.
                let epilogue_q8 =
                    reads_q8(&r.out) && fused_ssm_epilogue_supported(c.ssm_value_size(), inner);
                let io = st.recurrent.io();
                let deltanet_in = match io.deltanet_in {
                    Some(state) => Some(state.cast::<u8>().ok_or(LaunchError::InvalidArgument {
                        param: "deltanet_in",
                        reason: "buffer does not hold whole, aligned values of the element type",
                    })?),
                    None => None,
                };
                let deltanet = cast(io.deltanet, "deltanet_state")?;
                let q8_out = epilogue_q8.then(|| x.q8.slice_mut(0..q8_bytes(inner)));
                let kq_scales = x
                    .ssm_kq_scales
                    .slice_mut(0..batch * c.ssm_group_count as usize * 3);
                let alpha_beta = x.ssm_alpha_beta.slice_mut(0..batch * rank * 2);
                if conv_fused {
                    k::batched_ssm_conv_recurrence_row_split_conv_fused(
                        s,
                        x.ssm_qkv.slice(0..batch * qkv),
                        r.conv1d.view(),
                        io.conv,
                        io.conv_in,
                        x.conv_out.slice_mut(0..batch * qkv),
                        deltanet,
                        deltanet_in,
                        x.alpha.slice(0..batch * rank),
                        x.beta.slice(0..batch * rank),
                        r.a.view(),
                        r.dt.view(),
                        r.norm.view(),
                        x.ssm_gate.slice(0..batch * inner),
                        x.ssm_out.slice_mut(0..batch * inner),
                        q8_out,
                        kq_scales,
                        alpha_beta,
                        c.ssm_layer_index(l),
                        batch,
                        qkv,
                        c.ssm_group_count,
                        c.ssm_time_step_rank,
                        c.ssm_state_size,
                        c.ssm_value_size(),
                    )?;
                } else {
                    k::batched_ssm_conv_recurrence_row_split(
                        s,
                        x.ssm_qkv.slice(0..batch * qkv),
                        r.conv1d.view(),
                        io.conv,
                        io.conv_in,
                        x.conv_out.slice_mut(0..batch * qkv),
                        deltanet,
                        deltanet_in,
                        x.alpha.slice(0..batch * rank),
                        x.beta.slice(0..batch * rank),
                        r.a.view(),
                        r.dt.view(),
                        r.norm.view(),
                        x.ssm_gate.slice(0..batch * inner),
                        x.ssm_out.slice_mut(0..batch * inner),
                        q8_out,
                        kq_scales,
                        alpha_beta,
                        c.ssm_layer_index(l),
                        batch,
                        qkv,
                        c.ssm_group_count,
                        c.ssm_time_step_rank,
                        c.ssm_state_size,
                        c.ssm_value_size(),
                        None,
                    )?;
                }
                if !epilogue_q8 {
                    quantize_rows(s, x, r.out.format(), batch, inner)?;
                }
                gemm(
                    s,
                    &x.blas,
                    &x.q8,
                    batch,
                    &r.out,
                    x.ssm_out.slice(0..batch * inner),
                    x.attn_out.slice_mut(0..batch * hidden),
                    hidden,
                    inner,
                )?;
            }
        }

        // The post-attention residual add folds into the FFN norm where the
        // fused kernel takes the width.
        if ffn_norm_quantized {
            let staging = bf16_staging(&mut x.scratch16, batch * hidden)?;
            k::batched_residual_rms_norm_quantize_q8_1_bf16(
                s,
                x.attn_out.slice(0..batch * hidden),
                layer.ffn_norm.view(),
                x.hidden.slice_mut(0..batch * hidden),
                x.normed.slice_mut(0..batch * hidden),
                staging,
                x.q8.slice_mut(0..q8_bytes(hidden)),
                batch,
                hidden,
                NORM_EPS,
                Some(x.moe.x_tiled.slice_mut(0..batch * hidden)),
            )?;
        } else {
            k::batched_residual_add(
                s,
                x.attn_out.slice(0..batch * hidden),
                x.hidden.slice_mut(0..batch * hidden),
                batch,
                hidden,
            )?;
            unfused_norm(s, x, layer.ffn_norm.view(), batch, hidden, false)?;
        }
        crate::moe::prefill(s, &layer.moe, c, x, batch, ffn_norm_quantized)?;

        if let Some((taps, download, offset)) = capture.as_mut()
            && let Some(tap) = taps.iter().position(|&t| t == l)
        {
            download.copy_2d(
                x.hidden.slice(0..batch * hidden),
                hidden,
                *offset + tap * hidden,
                taps.len() * hidden,
                hidden,
                batch,
            )?;
        }
    }
    // Without taps the capture is each row's final hidden state.
    if let Some((taps, download, offset)) = capture.as_mut()
        && taps.is_empty()
    {
        download.copy_2d(
            x.hidden.slice(0..batch * hidden),
            hidden,
            *offset,
            hidden,
            hidden,
            batch,
        )?;
    }

    if compute_logits {
        let last = (batch - 1) * hidden;
        k::rms_norm(
            s,
            x.hidden.slice(last..last + hidden),
            w.output_norm.view(),
            x.normed.slice_mut(0..hidden),
            hidden,
            NORM_EPS,
        )?;
        k::gemv(
            s,
            w.output.view(),
            x.normed.slice(0..hidden),
            st.logits.as_mut_slice(),
            vocab,
            hidden,
        )?;
        k::argmax(s, st.logits.as_slice(), x.sampled.as_mut_slice(), vocab)?;
    }
    Ok(())
}

/// `slice` as `U` values, for buffers the C++ chunk reuses under another
/// type (the FP32 recurrent state as bytes).
fn cast<'a, T: DeviceCopy, U: DeviceCopy>(
    slice: DeviceSliceMut<'a, T>,
    param: &'static str,
) -> Result<DeviceSliceMut<'a, U>, LaunchError> {
    slice.cast::<U>().ok_or(LaunchError::InvalidArgument {
        param,
        reason: "buffer does not hold whole, aligned values of the element type",
    })
}
