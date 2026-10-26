//! Decode-equivalent verification of up to eight consecutive rows of one to
//! eight sessions, a port of `QwenGpuExecutor::ForwardVerificationBatch`
//! (`hip/batched_decode.cpp`) under the production policy. Every projection
//! uses the exact small-batch GEMM, which keeps the decode GEMV's per-output
//! arithmetic at any row count, and attention runs per session and the
//! recurrence of all sessions in one launch pair as in decode, so each row's
//! greedy token equals what autoregressive decoding would produce.
//!
//! C++ keeps the tapped rows and logits of one session in that session's
//! buffer; here they pass through the scratch and are copied to each session,
//! which moves no value.

use gufo_hip::{
    DeviceBuffer, DeviceSlice, DeviceSliceMut, GgmlType, Launch, LaunchError, QuantTensor, Stream,
};
use gufo_qwen_kernels as k;

use crate::arena::{Forward, MAX_VERIFY_ROWS, Scratch, SessionBuffers};
use crate::config::NORM_EPS;
use crate::routes::{bf16_weights, supports_exact_shared_projection};
use crate::weights::{LayerWeights, Mixer};

/// `y[rows][m] = x[rows][k] w^T` (`LaunchProjection`): the exact
/// shared-weight kernel of the quantized formats, the exact BF16 kernel, and
/// otherwise one decode GEMV per row.
pub(crate) fn project(
    s: &Stream,
    w: &QuantTensor,
    x: DeviceSlice<'_, f32>,
    mut y: DeviceSliceMut<'_, f32>,
    rows: usize,
    m: usize,
    k: usize,
) -> Result<(), LaunchError> {
    if supports_exact_shared_projection(w.format()) {
        return k::batched_quant_gemm_f32(s, w.view(), x, y, rows, m, k);
    }
    if w.format() == GgmlType::Bf16 {
        return k::exact_bf16_gemm_f32(s, bf16_weights(w)?, x, y, rows, m, k);
    }
    for row in 0..rows {
        k::gemv(
            s,
            w.view(),
            x.range(row * k..(row + 1) * k),
            y.reborrow().range(row * m..(row + 1) * m),
            m,
            k,
        )?;
    }
    Ok(())
}

/// Formats whose gate and up rows the C++ path projects as one matrix.
pub(crate) fn packs_ffn(format: GgmlType) -> bool {
    matches!(
        format,
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Iq4Xs | GgmlType::Q8_0
    )
}

/// One session of a verification batch.
pub(crate) struct Item<'a> {
    pub(crate) buffers: &'a mut SessionBuffers,
    /// Tokens of this session, at `position..`.
    pub(crate) rows: usize,
    pub(crate) position: u32,
    /// Its tapped hidden rows `[row][tap][hidden]` (or its hidden rows
    /// without taps), when the batch captures them.
    pub(crate) hidden: Option<&'a mut Vec<f32>>,
    /// Its logits rows, when requested.
    pub(crate) logits: Option<&'a mut Vec<f32>>,
}

fn invalid(param: &'static str, reason: &'static str) -> LaunchError {
    LaunchError::InvalidArgument { param, reason }
}

/// Runs the tokens in `x.tokens` of every item, row after row, whose
/// positions are in `x.positions`. Leaves each row's greedy token in
/// `x.sampled` and each item's logits rows in its `verification_logits`.
pub(crate) fn verify_batch(
    f: Forward<'_>,
    x: &mut Scratch,
    items: &mut [Item<'_>],
    taps: &[u32],
    capture_hidden: bool,
) -> Result<(), LaunchError> {
    let vocab = f.weights.config.vocab_size as usize;
    if items.is_empty()
        || items.len() > MAX_VERIFY_ROWS
        || items
            .iter()
            .any(|item| item.rows == 0 || item.rows > MAX_VERIFY_ROWS)
    {
        return Err(invalid(
            "items",
            "verification takes one to eight sessions of one to eight rows",
        ));
    }
    for item in items.iter_mut() {
        item.buffers.recurrent.begin_write();
    }
    if items
        .iter()
        .any(|item| item.buffers.verification_logits.is_none())
    {
        return Err(invalid(
            "verification_logits",
            "every session holds its logits rows",
        ));
    }
    let offsets: Vec<usize> = items
        .iter()
        .scan(0, |offset, item| {
            let start = *offset;
            *offset += item.rows;
            Some(start)
        })
        .collect();
    let n: usize = items.iter().map(|item| item.rows).sum();
    // The last FFN leaves the full-batch gate rows free before the
    // vocabulary projection: they hold the combined logits of several
    // sessions, the tapped rows stay in the verification buffer.
    let reuse_gate = items.len() > 1 && x.ffn_gate.len() >= n * vocab;
    // A single session computes in its own rows (so nothing is copied
    // there afterwards); several in the scratch buffer.
    let single = items.len() == 1;
    let taken = if single {
        items[0].buffers.verification_logits.take()
    } else {
        x.verification_logits.take()
    };
    let mut buffer = taken.ok_or(invalid(
        "verification_logits",
        "allocated before the first verification",
    ))?;
    let features = taps.len() * f.weights.config.hidden_size as usize;
    let needed = n * if capture_hidden {
        vocab.max(features)
    } else {
        vocab
    };
    let result = if buffer.len() < needed {
        Err(invalid(
            "verification_logits",
            "holds the rows and features of the verification",
        ))
    } else {
        run(
            f,
            x,
            &mut buffer,
            items,
            &offsets,
            taps,
            capture_hidden,
            reuse_gate,
        )
    };
    if single {
        items[0].buffers.verification_logits = Some(buffer);
    } else {
        x.verification_logits = Some(buffer);
    }
    result
}

/// The passes of [`verify_batch`] with the verification buffer taken out of
/// the scratch.
#[allow(clippy::too_many_arguments)]
fn run(
    f: Forward<'_>,
    x: &mut Scratch,
    feature_buffer: &mut DeviceBuffer<f32>,
    items: &mut [Item<'_>],
    offsets: &[usize],
    taps: &[u32],
    capture_hidden: bool,
    reuse_gate: bool,
) -> Result<(), LaunchError> {
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
    let features = taps.len() * hidden;
    let n: usize = items.iter().map(|item| item.rows).sum();

    k::batched_embedding_lookup(
        s,
        w.token_embd.view(),
        x.tokens.slice(0..n),
        x.hidden.slice_mut(0..n * hidden),
        n,
        hidden,
        vocab,
    )?;
    for (item, &offset) in items.iter().zip(offsets) {
        item.buffers.vision.inject(
            s,
            x.hidden
                .slice_mut(offset * hidden..(offset + item.rows) * hidden),
            item.position,
            item.rows as u32,
            hidden as u32,
            1,
        )?;
    }
    for (l, layer) in (0u32..).zip(&w.layers) {
        k::batched_rms_norm(
            s,
            x.hidden.slice(0..n * hidden),
            layer.attn_norm.view(),
            x.normed.slice_mut(0..n * hidden),
            n,
            hidden,
            NORM_EPS,
        )?;
        match &layer.mixer {
            Mixer::Attention(at) => {
                let input = x.normed.slice(0..n * hidden);
                project(
                    s,
                    &at.q,
                    input,
                    x.ssm_qkv.slice_mut(0..n * 2 * attention),
                    n,
                    2 * attention,
                    hidden,
                )?;
                project(s, &at.k, input, x.k.slice_mut(0..n * kv), n, kv, hidden)?;
                project(s, &at.v, input, x.v.slice_mut(0..n * kv), n, kv, hidden)?;
                k::batched_unpack_qg(
                    s,
                    x.ssm_qkv.slice(0..n * 2 * attention),
                    x.q.slice_mut(0..n * attention),
                    x.ssm_gate.slice_mut(0..n * attention),
                    n,
                    c.num_attention_heads,
                    c.head_dim,
                )?;
                let attention_layer = l / c.full_attention_interval;
                for (item, &offset) in items.iter_mut().zip(offsets) {
                    let (rows, start_pos) = (item.rows, item.position);
                    let (q, kr) = (offset * attention, offset * kv);
                    let kv_plane = item.buffers.kv.len() / 2;
                    let (k_cache, v_cache) = item.buffers.kv.as_mut_slice().split_at(kv_plane);
                    k::batched_fused_qk_norm_rope_kv_write(
                        s,
                        x.v.slice(kr..kr + rows * kv),
                        at.q_norm.view(),
                        at.k_norm.view(),
                        x.q.slice_mut(q..q + rows * attention),
                        x.k.slice_mut(kr..kr + rows * kv),
                        k_cache,
                        v_cache,
                        attention_layer,
                        start_pos,
                        rows,
                        max_context,
                        c.num_attention_heads,
                        c.num_key_value_heads,
                        c.head_dim,
                        c.rotary_dim,
                        c.rope_theta,
                        NORM_EPS,
                        item.buffers.vision.rope(),
                    )?;
                    // The weight scratch holds a split-K row per verified row
                    // when it is large enough, else the decode's single row.
                    let row_scratch = x.split_k.len();
                    let scratch = if x.packed.len() >= rows * row_scratch {
                        x.packed.slice_mut(0..rows * row_scratch)
                    } else {
                        x.split_k.as_mut_slice()
                    };
                    let scratch_len = scratch.len();
                    k::causal_decode_attention(
                        s,
                        x.q.slice(q..q + rows * attention),
                        x.ssm_gate.slice(q..q + rows * attention),
                        item.buffers.kv.slice(0..kv_plane),
                        item.buffers.kv.slice(kv_plane..2 * kv_plane),
                        x.ssm_out.slice_mut(q..q + rows * attention),
                        attention_layer,
                        start_pos,
                        rows,
                        max_context,
                        c.num_attention_heads,
                        c.num_key_value_heads,
                        c.head_dim,
                        scratch,
                        scratch_len,
                    )?;
                }
                project(
                    s,
                    &at.output,
                    x.ssm_out.slice(0..n * attention),
                    x.attn_out.slice_mut(0..n * hidden),
                    n,
                    hidden,
                    attention,
                )?;
            }
            Mixer::Recurrent(r) => {
                let input = x.normed.slice(0..n * hidden);
                project(
                    s,
                    &r.qkv,
                    input,
                    x.ssm_qkv.slice_mut(0..n * qkv),
                    n,
                    qkv,
                    hidden,
                )?;
                project(
                    s,
                    &r.gate,
                    input,
                    x.ssm_gate.slice_mut(0..n * inner),
                    n,
                    inner,
                    hidden,
                )?;
                // Adjacent Q8_0 alpha and beta form one 96-row matrix whose
                // rows interleave `[alpha | beta]` (`LaunchSsmControls`).
                let packed_controls = (rank == 48
                    && hidden == 5120
                    && r.alpha.format() == GgmlType::Q8_0
                    && x.packed.len() >= n * 2 * rank)
                    .then(|| r.alpha.join(&r.beta))
                    .flatten();
                let stride = if let Some(controls) = &packed_controls {
                    project(
                        s,
                        controls,
                        input,
                        x.packed.slice_mut(0..n * 2 * rank),
                        n,
                        2 * rank,
                        hidden,
                    )?;
                    2 * rank
                } else {
                    project(
                        s,
                        &r.alpha,
                        input,
                        x.alpha.slice_mut(0..n * rank),
                        n,
                        rank,
                        hidden,
                    )?;
                    project(
                        s,
                        &r.beta,
                        input,
                        x.beta.slice_mut(0..n * rank),
                        n,
                        rank,
                        hidden,
                    )?;
                    rank
                };
                if items.len() > 1 {
                    // One launch pair over every item's state and rows
                    // (`LaunchSSMConvRecurrenceBatch`).
                    let len = (n - 1) * stride + rank;
                    let (alpha, beta) = if packed_controls.is_some() {
                        (x.packed.slice(0..len), x.packed.slice(rank..rank + len))
                    } else {
                        (x.alpha.slice(0..len), x.beta.slice(0..len))
                    };
                    let count = items.len();
                    let mut conv = Vec::with_capacity(count);
                    let mut deltanet = Vec::with_capacity(count);
                    let mut conv_in = Vec::with_capacity(count);
                    let mut deltanet_in = Vec::with_capacity(count);
                    let (mut replay_qkv, mut replay_alpha, mut replay_beta) = (
                        Vec::with_capacity(count),
                        Vec::with_capacity(count),
                        Vec::with_capacity(count),
                    );
                    let (mut replay_position, mut replay_enabled) =
                        (Vec::with_capacity(count), Vec::with_capacity(count));
                    let mut row_offsets = Vec::with_capacity(count);
                    let mut row_counts = Vec::with_capacity(count);
                    for (item, &offset) in items.iter_mut().zip(offsets) {
                        let rows = item.rows;
                        row_offsets.push(offset as u32);
                        row_counts.push(rows as u32);
                        let b = &mut *item.buffers;
                        let io = b.recurrent.io();
                        conv.push(io.conv);
                        conv_in.push(io.conv_in);
                        deltanet.push(io.deltanet);
                        deltanet_in.push(io.deltanet_in);
                        match b.replay.as_mut() {
                            Some(ring) => {
                                replay_qkv.push(Some(ring.qkv.as_mut_slice()));
                                replay_alpha.push(Some(ring.alpha.as_mut_slice()));
                                replay_beta.push(Some(ring.beta.as_mut_slice()));
                                replay_position
                                    .push(Some(x.positions.slice(offset..offset + rows)));
                                replay_enabled.push(Some(ring.enabled.as_slice()));
                            }
                            None => {
                                replay_qkv.push(None);
                                replay_alpha.push(None);
                                replay_beta.push(None);
                                replay_position.push(None);
                                replay_enabled.push(None);
                            }
                        }
                    }
                    k::ssm_conv_recurrence_sequences(
                        s,
                        x.ssm_qkv.slice(0..n * qkv),
                        r.conv1d.view(),
                        x.conv_out.slice_mut(0..n * qkv),
                        alpha,
                        beta,
                        r.a.view(),
                        r.dt.view(),
                        Some(r.norm.view()),
                        Some(x.ssm_gate.slice(0..n * inner)),
                        Some(x.ssm_out.slice_mut(0..n * inner)),
                        &mut conv,
                        &conv_in,
                        &mut deltanet,
                        &deltanet_in,
                        &mut replay_qkv,
                        &mut replay_alpha,
                        &mut replay_beta,
                        &replay_position,
                        &replay_enabled,
                        &row_offsets,
                        &row_counts,
                        count as u32,
                        n as u32,
                        c.ssm_layer_index(l),
                        qkv,
                        c.ssm_group_count,
                        c.ssm_time_step_rank,
                        c.ssm_state_size,
                        c.ssm_value_size(),
                        stride,
                        inner,
                    )?;
                } else {
                    for (item, &offset) in items.iter_mut().zip(offsets) {
                        let rows = item.rows;
                        let control = offset * stride;
                        let control_len = (rows - 1) * stride + rank;
                        let (alpha, beta) = if packed_controls.is_some() {
                            (
                                x.packed.slice(control..control + control_len),
                                x.packed.slice(control + rank..control + rank + control_len),
                            )
                        } else {
                            (
                                x.alpha.slice(control..control + control_len),
                                x.beta.slice(control..control + control_len),
                            )
                        };
                        let b = &mut *item.buffers;
                        let (
                            replay_qkv,
                            replay_alpha,
                            replay_beta,
                            replay_position,
                            replay_enabled,
                        ) = match b.replay.as_mut() {
                            Some(ring) => (
                                Some(ring.qkv.as_mut_slice()),
                                Some(ring.alpha.as_mut_slice()),
                                Some(ring.beta.as_mut_slice()),
                                Some(x.positions.slice(offset..offset + rows)),
                                Some(ring.enabled.as_slice()),
                            ),
                            None => (None, None, None, None, None),
                        };
                        let io = b.recurrent.io();
                        k::ssm_conv_recurrence_rows(
                            s,
                            x.ssm_qkv.slice(offset * qkv..(offset + rows) * qkv),
                            r.conv1d.view(),
                            io.conv,
                            io.conv_in,
                            x.conv_out.slice_mut(offset * qkv..(offset + rows) * qkv),
                            io.deltanet,
                            io.deltanet_in,
                            alpha,
                            beta,
                            r.a.view(),
                            r.dt.view(),
                            Some(r.norm.view()),
                            Some(x.ssm_gate.slice(offset * inner..(offset + rows) * inner)),
                            Some(x.ssm_out.slice_mut(offset * inner..(offset + rows) * inner)),
                            c.ssm_layer_index(l),
                            qkv,
                            c.ssm_group_count,
                            c.ssm_time_step_rank,
                            c.ssm_state_size,
                            c.ssm_value_size(),
                            rows as u32,
                            stride,
                            inner,
                            replay_qkv,
                            replay_alpha,
                            replay_beta,
                            replay_position,
                            replay_enabled,
                        )?;
                    }
                }
                project(
                    s,
                    &r.out,
                    x.ssm_out.slice(0..n * inner),
                    x.attn_out.slice_mut(0..n * hidden),
                    n,
                    hidden,
                    inner,
                )?;
            }
        }
        k::batched_residual_add(
            s,
            x.attn_out.slice(0..n * hidden),
            x.hidden.slice_mut(0..n * hidden),
            n,
            hidden,
        )?;
        k::batched_rms_norm(
            s,
            x.hidden.slice(0..n * hidden),
            layer.ffn_norm.view(),
            x.normed.slice_mut(0..n * hidden),
            n,
            hidden,
            NORM_EPS,
        )?;
        ffn_activation(s, x, layer, n, ffn, hidden)?;
        project(
            s,
            &layer.ffn_down,
            x.ffn_act.slice(0..n * ffn),
            x.ffn_out.slice_mut(0..n * hidden),
            n,
            hidden,
            ffn,
        )?;
        k::batched_residual_add(
            s,
            x.ffn_out.slice(0..n * hidden),
            x.hidden.slice_mut(0..n * hidden),
            n,
            hidden,
        )?;
        if capture_hidden && let Some(tap) = taps.iter().position(|&t| t == l) {
            s.copy_2d(
                x.hidden.slice(0..n * hidden),
                hidden,
                feature_buffer.slice_mut(tap * hidden..n * features),
                features,
                hidden,
                n,
            )?;
        }
    }

    for (item, &offset) in items.iter_mut().zip(offsets) {
        let rows = item.rows;
        if capture_hidden && !taps.is_empty() {
            if let Some(host) = item.hidden.as_deref_mut() {
                host.resize(rows * features, 0.0);
                s.download(
                    feature_buffer.slice(offset * features..(offset + rows) * features),
                    host,
                )?;
            }
            // The last row's features follow the sequence for the draft.
            let last = (offset + rows - 1) * features;
            s.copy(
                feature_buffer.slice(last..last + features),
                item.buffers.target_features.slice_mut(0..features),
            )?;
        } else if capture_hidden && let Some(host) = item.hidden.as_deref_mut() {
            host.resize(rows * hidden, 0.0);
            s.download(
                x.hidden.slice(offset * hidden..(offset + rows) * hidden),
                host,
            )?;
        }
    }
    k::batched_rms_norm(
        s,
        x.hidden.slice(0..n * hidden),
        w.output_norm.view(),
        x.normed.slice_mut(0..n * hidden),
        n,
        hidden,
        NORM_EPS,
    )?;
    let logits = if reuse_gate {
        x.ffn_gate.slice_mut(0..n * vocab)
    } else {
        feature_buffer.slice_mut(0..n * vocab)
    };
    project(
        s,
        &w.output,
        x.normed.slice(0..n * hidden),
        logits,
        n,
        vocab,
        hidden,
    )?;
    let logits = if reuse_gate {
        x.ffn_gate.slice(0..n * vocab)
    } else {
        feature_buffer.slice(0..n * vocab)
    };
    k::batched_argmax(
        s,
        logits,
        x.sampled.slice_mut(0..n),
        n,
        vocab,
        x.ffn_out.slice_mut(0..n * hidden),
        n * hidden,
    )?;
    for (item, &offset) in items.iter_mut().zip(offsets) {
        let rows = logits.range(offset * vocab..(offset + item.rows) * vocab);
        if let Some(buffer) = item.buffers.verification_logits.as_mut() {
            s.copy(rows, buffer.slice_mut(0..item.rows * vocab))?;
        }
        if let Some(host) = item.logits.as_deref_mut() {
            host.resize(item.rows * vocab, 0.0);
            s.download(rows, host)?;
        }
    }
    Ok(())
}

/// `LaunchFfnActivation`: the fused packed kernel where it applies, else the
/// adjacent gate and up rows as one projection, else two.
pub(crate) fn ffn_activation(
    s: &Stream,
    x: &mut Scratch,
    layer: &LayerWeights,
    n: usize,
    ffn: usize,
    hidden: usize,
) -> Result<(), LaunchError> {
    let gate_up = (n >= 2 && ffn == 17408 && hidden == 5120 && packs_ffn(layer.ffn_gate.format()))
        .then(|| layer.ffn_gate.join(&layer.ffn_up))
        .flatten();
    let input = x.normed.slice(0..n * hidden);
    if let Some(gate_up) = &gate_up {
        let fused = k::packed_quant_swiglu_fp32(
            s,
            gate_up.view(),
            input,
            x.ffn_act.slice_mut(0..n * ffn),
            n,
            ffn,
            hidden,
        )?;
        if fused == Launch::Launched {
            return Ok(());
        }
        if x.packed.len() >= 2 * n * ffn {
            project(
                s,
                gate_up,
                input,
                x.packed.slice_mut(0..n * 2 * ffn),
                n,
                2 * ffn,
                hidden,
            )?;
            return k::packed_swiglu(
                s,
                x.packed.slice(0..n * 2 * ffn),
                x.ffn_act.slice_mut(0..n * ffn),
                n,
                ffn,
            );
        }
    }
    project(
        s,
        &layer.ffn_gate,
        input,
        x.ffn_gate.slice_mut(0..n * ffn),
        n,
        ffn,
        hidden,
    )?;
    project(
        s,
        &layer.ffn_up,
        input,
        x.ffn_up.slice_mut(0..n * ffn),
        n,
        ffn,
        hidden,
    )?;
    k::batched_swiglu(
        s,
        x.ffn_gate.slice(0..n * ffn),
        x.ffn_up.slice(0..n * ffn),
        x.ffn_act.slice_mut(0..n * ffn),
        n * ffn,
    )
}
