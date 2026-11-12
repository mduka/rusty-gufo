//! One chunk of prompt rows, a port of `QwenGpuExecutor::ForwardPromptChunk`
//! (`hip/prefill_chunk.cpp`) for the formats the weights accept, all of which
//! read the tiled Q8_1 activation (`routes::reads_q8_activation`). From 1024
//! rows of the 27B shape the chunk runs in FP16 (`routes::use_fp16_prefill`).
//!
//! Two simplifications give the same launches as the C++ routes: the dual
//! gate/up GEMM is two GEMMs in C++ as well, and the Q8_0-only fused SwiGLU
//! epilogue is bit-identical to the GEMM pair plus the SwiGLU quantizer used
//! here.

use gufo_hip::{
    Bf16, DeviceBuffer, DeviceCopy, DeviceSlice, DeviceSliceMut, F16, GgmlType, HipError, Launch,
    LaunchError, PinnedDownload, QuantTensor, Stream,
};
use gufo_qwen_kernels as k;

use crate::arena::{Forward, Scratch, SessionBuffers};
use crate::config::NORM_EPS;
use crate::routes::{
    bf16_weights, fused_norm_quantize_supported, fused_ssm_epilogue_supported,
    quantized_activation_bytes, reads_q8_activation, use_fp16_prefill,
};
use crate::weights::Mixer;

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

/// Bytes of the Q8_1 activation a Q8_0 GEMM over `[batch][k]` may read.
fn q8_gemm_input_bytes(batch: usize, k: usize) -> usize {
    batch.div_ceil(128) * 8 * (k / 32) * 640
}

/// What the projections of this chunk read: the tiled Q8_1 activation, or
/// the FP16 scratch in the half prefill.
struct Activation<'a> {
    half: bool,
    q8: &'a DeviceBuffer<u8>,
    scratch16: &'a DeviceBuffer<F16>,
    batch: usize,
}

impl<'a> Activation<'a> {
    fn new(
        half: bool,
        q8: &'a DeviceBuffer<u8>,
        scratch16: &'a DeviceBuffer<F16>,
        batch: usize,
    ) -> Self {
        Self {
            half,
            q8,
            scratch16,
            batch,
        }
    }

    /// `y[batch][m] = activation[batch][k] w^T` (`gemm_weight` of the C++
    /// chunk).
    fn gemm(
        &self,
        s: &Stream,
        w: &QuantTensor,
        y: DeviceSliceMut<'_, f32>,
        m: usize,
        k: usize,
    ) -> Result<(), LaunchError> {
        let batch = self.batch;
        if self.half {
            k::batched_quant_gemm_fp16(
                s,
                w.view(),
                self.scratch16.slice(0..batch * k),
                y,
                batch,
                m,
                k,
            )
        } else {
            k::batched_quant_gemm_q8_1(
                s,
                w.view(),
                self.q8.slice(0..q8_gemm_input_bytes(batch, k)),
                y,
                batch,
                m,
                k,
            )
        }
    }
}

/// `y[batch][m] = rows[batch][k] w^T` by the route C++ resolves
/// (`gemm_weight`): the FP16 or Q8_1 activation for the quantized formats,
/// the exact BF16 kernel over the FP32 rows `fp32` for BF16.
#[allow(clippy::too_many_arguments)]
fn project(
    s: &Stream,
    half: bool,
    q8: &DeviceBuffer<u8>,
    scratch16: &DeviceBuffer<F16>,
    batch: usize,
    w: &QuantTensor,
    fp32: DeviceSlice<'_, f32>,
    y: DeviceSliceMut<'_, f32>,
    m: usize,
    k: usize,
) -> Result<(), LaunchError> {
    if half || reads_q8_activation(w.format()) {
        return Activation::new(half, q8, scratch16, batch).gemm(s, w, y, m, k);
    }
    if w.format() == GgmlType::Bf16 {
        return k::exact_bf16_gemm_f32(s, bf16_weights(w)?, fp32, y, batch, m, k);
    }
    Err(LaunchError::InvalidArgument {
        param: "weights",
        reason: "the Rust Qwen prefill takes quantized and BF16 projections",
    })
}

/// The unfused RMSNorm (`LaunchBatchedRMSNorm`): FP32 rows in `x.normed`
/// and, when a projection reads the tiled Q8_1 activation, its BF16 copy
/// and the activation quantized from that copy, as C++ computes it.
fn unfused_norm(
    s: &Stream,
    x: &mut Scratch,
    weight: DeviceSlice<'_, f32>,
    batch: usize,
    hidden: usize,
    reads_q8: bool,
) -> Result<(), LaunchError> {
    if !reads_q8 {
        return k::batched_rms_norm(
            s,
            x.hidden.slice(0..batch * hidden),
            weight,
            x.normed.slice_mut(0..batch * hidden),
            batch,
            hidden,
            NORM_EPS,
        );
    }
    let staging = cast::<F16, Bf16>(x.scratch16.slice_mut(0..batch * hidden), "scratch16")?;
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
    let staging = cast::<F16, Bf16>(x.scratch16.slice_mut(0..batch * hidden), "scratch16")?;
    k::quantize_q8_1(
        s,
        staging.as_const(),
        x.q8.slice_mut(0..quantized_activation_bytes(batch, hidden)),
        batch,
        hidden,
    )
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
    let ffn = c.intermediate_size as usize;
    let vocab = c.vocab_size as usize;
    let kv_plane = st.kv.len() / 2;
    let half = use_fp16_prefill(w, batch);
    let q8_bytes = |k: usize| quantized_activation_bytes(batch, k);

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

    let fusable_norm = fused_norm_quantize_supported(hidden);
    for (l, layer) in (0u32..).zip(&w.layers) {
        st.prefill_layer_checkpoint(s, l)?;
        // Cache updates retain every row. The final layer's query and FFN
        // outputs only feed the last token unless this layer is captured
        // (without taps, every row's final hidden state is). Its original
        // 256-row FP16 GEMM tile stays, with the original tail mask and
        // FP16 rounding boundaries.
        let final_output_tile = half
            && l + 1 == c.num_layers
            && capture
                .as_ref()
                .is_none_or(|(taps, _, _)| !taps.is_empty() && !taps.contains(&l));
        let first_row = if final_output_tile {
            (batch - 1) / 256 * 256
        } else {
            0
        };
        let output_rows = batch - first_row;
        let output_hidden = first_row * hidden..batch * hidden;
        // A projection reads the tiled Q8_1 activation unless it is BF16 (or
        // the chunk runs in FP16).
        let reads_q8 = |w: &QuantTensor| !half && reads_q8_activation(w.format());
        let mixer_inputs: [&QuantTensor; 4] = match &layer.mixer {
            Mixer::Attention(at) => [&at.q, &at.k, &at.v, &at.v],
            Mixer::Recurrent(r) => [&r.qkv, &r.gate, &r.alpha, &r.beta],
        };
        // Where every reader of a norm takes Q8_1, the norm writes it
        // directly; otherwise the FP32 rows feed the BF16 projections and a
        // BF16 copy feeds the Q8_1 quantizer, which C++ also runs when no
        // projection reads its output (skipped here).
        let norm_feeds_q8_only = fusable_norm && mixer_inputs.iter().all(|w| reads_q8(w));
        let mixer_reads_q8 = mixer_inputs.iter().any(|w| reads_q8(w));
        let ffn_feeds_q8_only =
            fusable_norm && reads_q8(&layer.ffn_gate) && reads_q8(&layer.ffn_up);
        let ffn_reads_q8 = reads_q8(&layer.ffn_gate) || reads_q8(&layer.ffn_up);

        if half {
            k::batched_rms_norm_fp16(
                s,
                x.hidden.slice(0..batch * hidden),
                layer.attn_norm.view(),
                x.scratch16.slice_mut(0..batch * hidden),
                batch,
                hidden,
                NORM_EPS,
            )?;
        } else if norm_feeds_q8_only {
            k::batched_rms_norm_quantize_q8_1(
                s,
                x.hidden.slice(0..batch * hidden),
                layer.attn_norm.view(),
                x.q8.slice_mut(0..q8_bytes(hidden)),
                batch,
                hidden,
                NORM_EPS,
            )?;
        } else {
            unfused_norm(s, x, layer.attn_norm.view(), batch, hidden, mixer_reads_q8)?;
        }

        match &layer.mixer {
            Mixer::Attention(at) => {
                let attention_layer = l / c.full_attention_interval;
                project(
                    s,
                    half,
                    &x.q8,
                    &x.scratch16,
                    batch,
                    &at.q,
                    x.normed.slice(0..batch * hidden),
                    x.ssm_qkv.slice_mut(0..batch * 2 * attention),
                    2 * attention,
                    hidden,
                )?;
                project(
                    s,
                    half,
                    &x.q8,
                    &x.scratch16,
                    batch,
                    &at.k,
                    x.normed.slice(0..batch * hidden),
                    x.k.slice_mut(0..batch * kv),
                    kv,
                    hidden,
                )?;
                project(
                    s,
                    half,
                    &x.q8,
                    &x.scratch16,
                    batch,
                    &at.v,
                    x.normed.slice(0..batch * hidden),
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
                    first_row,
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
                if half {
                    k::float_to_fp16(
                        s,
                        x.ssm_out.slice(first_row * attention..batch * attention),
                        x.scratch16.slice_mut(0..output_rows * attention),
                        output_rows * attention,
                    )?;
                    k::batched_quant_gemm_residual_fp16(
                        s,
                        at.output.view(),
                        x.scratch16.slice(0..output_rows * attention),
                        x.hidden.slice_mut(output_hidden.clone()),
                        output_rows,
                        hidden,
                        attention,
                    )?;
                } else {
                    if reads_q8(&at.output) {
                        k::quantize_q8_1_from_f32(
                            s,
                            x.ssm_out.slice(0..batch * attention),
                            x.q8.slice_mut(0..q8_bytes(attention)),
                            batch,
                            attention,
                        )?;
                    }
                    project(
                        s,
                        half,
                        &x.q8,
                        &x.scratch16,
                        batch,
                        &at.output,
                        x.ssm_out.slice(0..batch * attention),
                        x.attn_out.slice_mut(0..batch * hidden),
                        hidden,
                        attention,
                    )?;
                }
            }
            Mixer::Recurrent(r) => {
                project(
                    s,
                    half,
                    &x.q8,
                    &x.scratch16,
                    batch,
                    &r.qkv,
                    x.normed.slice(0..batch * hidden),
                    x.ssm_qkv.slice_mut(0..batch * qkv),
                    qkv,
                    hidden,
                )?;
                project(
                    s,
                    half,
                    &x.q8,
                    &x.scratch16,
                    batch,
                    &r.gate,
                    x.normed.slice(0..batch * hidden),
                    x.ssm_gate.slice_mut(0..batch * inner),
                    inner,
                    hidden,
                )?;
                project(
                    s,
                    half,
                    &x.q8,
                    &x.scratch16,
                    batch,
                    &r.alpha,
                    x.normed.slice(0..batch * hidden),
                    x.alpha.slice_mut(0..batch * rank),
                    rank,
                    hidden,
                )?;
                project(
                    s,
                    half,
                    &x.q8,
                    &x.scratch16,
                    batch,
                    &r.beta,
                    x.normed.slice(0..batch * hidden),
                    x.beta.slice_mut(0..batch * rank),
                    rank,
                    hidden,
                )?;
                // The row-split epilogue writes what ssm_out reads: FP16 in
                // the half prefill, else the Q8_1 activation when ssm_out
                // reads it and the head shape allows it.
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
                k::batched_ssm_conv_recurrence_row_split(
                    s,
                    x.ssm_qkv.slice(0..batch * qkv),
                    r.conv1d.view(),
                    io.conv,
                    io.conv_in,
                    x.conv_out.slice_mut(0..batch * qkv),
                    cast(io.deltanet, "deltanet_state")?,
                    deltanet_in,
                    x.alpha.slice(0..batch * rank),
                    x.beta.slice(0..batch * rank),
                    r.a.view(),
                    r.dt.view(),
                    r.norm.view(),
                    x.ssm_gate.slice(0..batch * inner),
                    x.ssm_out.slice_mut(0..batch * inner),
                    epilogue_q8.then(|| x.q8.slice_mut(0..q8_bytes(inner))),
                    x.ssm_kq_scales
                        .slice_mut(0..batch * c.ssm_group_count as usize * 3),
                    x.ssm_alpha_beta.slice_mut(0..batch * rank * 2),
                    c.ssm_layer_index(l),
                    batch,
                    qkv,
                    c.ssm_group_count,
                    c.ssm_time_step_rank,
                    c.ssm_state_size,
                    c.ssm_value_size(),
                    half.then(|| x.scratch16.slice_mut(0..batch * inner)),
                )?;
                if half {
                    k::batched_quant_gemm_residual_fp16(
                        s,
                        r.out.view(),
                        x.scratch16.slice(0..batch * inner),
                        x.hidden.slice_mut(0..batch * hidden),
                        batch,
                        hidden,
                        inner,
                    )?;
                } else {
                    if reads_q8(&r.out) && !epilogue_q8 {
                        k::quantize_q8_1_from_f32(
                            s,
                            x.ssm_out.slice(0..batch * inner),
                            x.q8.slice_mut(0..q8_bytes(inner)),
                            batch,
                            inner,
                        )?;
                    }
                    project(
                        s,
                        half,
                        &x.q8,
                        &x.scratch16,
                        batch,
                        &r.out,
                        x.ssm_out.slice(0..batch * inner),
                        x.attn_out.slice_mut(0..batch * hidden),
                        hidden,
                        inner,
                    )?;
                }
            }
        }

        if half {
            // The final output tile's rows land compacted at the front of the
            // FP16 scratch and the FFN rows.
            let rows = output_rows;
            k::batched_rms_norm_fp16(
                s,
                x.hidden.slice(output_hidden.clone()),
                layer.ffn_norm.view(),
                x.scratch16.slice_mut(0..rows * hidden),
                rows,
                hidden,
                NORM_EPS,
            )?;
            // SwiGLU in FP16 lands in the up rows, read as halves by ffn_down.
            let up16 = cast::<f32, F16>(x.ffn_up.as_mut_slice(), "ffn_up")?;
            let paired = k::dual_quant_gemm_swiglu_fp16(
                s,
                layer.ffn_gate.view(),
                layer.ffn_up.view(),
                x.scratch16.slice(0..rows * hidden),
                up16.range(0..rows * ffn),
                rows,
                ffn,
                hidden,
            )?;
            if paired == Launch::Unsupported {
                k::batched_quant_gemm_fp16(
                    s,
                    layer.ffn_gate.view(),
                    x.scratch16.slice(0..rows * hidden),
                    x.ffn_gate.slice_mut(0..rows * ffn),
                    rows,
                    ffn,
                    hidden,
                )?;
                let up16 = cast::<f32, F16>(x.ffn_up.as_mut_slice(), "ffn_up")?;
                k::batched_quant_gemm_swiglu_fp16(
                    s,
                    layer.ffn_up.view(),
                    x.scratch16.slice(0..rows * hidden),
                    x.ffn_gate.slice(0..rows * ffn),
                    up16.range(0..rows * ffn),
                    rows,
                    ffn,
                    hidden,
                )?;
            }
            let up16 = cast::<f32, F16>(x.ffn_up.as_mut_slice(), "ffn_up")?;
            k::batched_quant_gemm_residual_fp16(
                s,
                layer.ffn_down.view(),
                up16.as_const().range(0..rows * ffn),
                x.hidden.slice_mut(output_hidden),
                rows,
                hidden,
                ffn,
            )?;
        } else {
            if ffn_feeds_q8_only {
                // The residual add folds into the norm.
                k::batched_residual_rms_norm_quantize_q8_1(
                    s,
                    x.attn_out.slice(0..batch * hidden),
                    layer.ffn_norm.view(),
                    x.hidden.slice_mut(0..batch * hidden),
                    x.q8.slice_mut(0..q8_bytes(hidden)),
                    batch,
                    hidden,
                    NORM_EPS,
                )?;
            } else {
                k::batched_residual_add(
                    s,
                    x.attn_out.slice(0..batch * hidden),
                    x.hidden.slice_mut(0..batch * hidden),
                    batch,
                    hidden,
                )?;
                unfused_norm(s, x, layer.ffn_norm.view(), batch, hidden, ffn_reads_q8)?;
            }
            // The dual and fused-epilogue routes of C++ give the same values
            // as these two projections and the SwiGLU quantizer.
            project(
                s,
                half,
                &x.q8,
                &x.scratch16,
                batch,
                &layer.ffn_gate,
                x.normed.slice(0..batch * hidden),
                x.ffn_gate.slice_mut(0..batch * ffn),
                ffn,
                hidden,
            )?;
            project(
                s,
                half,
                &x.q8,
                &x.scratch16,
                batch,
                &layer.ffn_up,
                x.normed.slice(0..batch * hidden),
                x.ffn_up.slice_mut(0..batch * ffn),
                ffn,
                hidden,
            )?;
            if reads_q8(&layer.ffn_down) {
                k::batched_swiglu_quantize_q8_1(
                    s,
                    x.ffn_gate.slice(0..batch * ffn),
                    x.ffn_up.slice(0..batch * ffn),
                    x.q8.slice_mut(0..q8_bytes(ffn)),
                    batch,
                    ffn,
                )?;
            } else {
                k::batched_swiglu(
                    s,
                    x.ffn_gate.slice(0..batch * ffn),
                    x.ffn_up.slice(0..batch * ffn),
                    x.ffn_act.slice_mut(0..batch * ffn),
                    batch * ffn,
                )?;
            }
            project(
                s,
                half,
                &x.q8,
                &x.scratch16,
                batch,
                &layer.ffn_down,
                x.ffn_act.slice(0..batch * ffn),
                x.ffn_out.slice_mut(0..batch * hidden),
                hidden,
                ffn,
            )?;
            k::batched_residual_add(
                s,
                x.ffn_out.slice(0..batch * hidden),
                x.hidden.slice_mut(0..batch * hidden),
                batch,
                hidden,
            )?;
        }
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
/// type (the FP32 recurrent state as bytes, the FFN up rows as halves).
fn cast<'a, T: DeviceCopy, U: DeviceCopy>(
    slice: DeviceSliceMut<'a, T>,
    param: &'static str,
) -> Result<DeviceSliceMut<'a, U>, LaunchError> {
    slice.cast::<U>().ok_or(LaunchError::InvalidArgument {
        param,
        reason: "buffer does not hold whole, aligned values of the element type",
    })
}
