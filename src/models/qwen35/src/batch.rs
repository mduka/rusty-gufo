//! One decode step of two to eight sessions, a port of
//! `QwenGpuExecutor::ForwardTokenBatch` (`hip/batched_decode.cpp`) under the
//! production policy. The projections share weight loads through the exact
//! small-batch GEMM; per-head norm, RoPE and attention run per session on its
//! own cache, and the recurrence of all sessions runs in one launch pair over
//! their states, in the C++ order.

use gufo_hip::{DeviceBuffer, GgmlType, LaunchError, Stream};
use gufo_qwen_kernels as k;

use crate::arena::{Forward, MAX_VERIFY_ROWS, Scratch, SessionBuffers};
use crate::config::NORM_EPS;
use crate::decode::split_k_attention;
use crate::verify::{ffn_activation, project};
use crate::weights::Mixer;

/// One session of a batched step.
pub(crate) struct Row<'a> {
    pub(crate) buffers: &'a mut SessionBuffers,
    pub(crate) position: u32,
    /// Layers whose hidden row the draft reads, `[tap][hidden]`.
    pub(crate) taps: &'a [u32],
}

/// Runs the tokens already in `x.tokens` for `rows`, whose positions (for
/// the replay capture) are in `x.positions`; leaves each row's logits in
/// `x.verification_logits` and its greedy token in `x.sampled`.
pub(crate) fn decode_rows(
    f: Forward<'_>,
    s: &Stream,
    x: &mut Scratch,
    rows: &mut [Row<'_>],
) -> Result<(), LaunchError> {
    if !(2..=MAX_VERIFY_ROWS).contains(&rows.len()) {
        return Err(LaunchError::InvalidArgument {
            param: "rows",
            reason: "a batched decode step takes two to eight sessions",
        });
    }
    for row in rows.iter_mut() {
        row.buffers.recurrent.begin_write();
    }
    let mut buffer = x
        .verification_logits
        .take()
        .ok_or(LaunchError::InvalidArgument {
            param: "verification_logits",
            reason: "allocated before the first batched step",
        })?;
    let result = run(f, s, x, &mut buffer, rows);
    x.verification_logits = Some(buffer);
    result
}

/// The passes of [`decode_rows`] with the logits buffer taken out of the
/// scratch.
fn run(
    f: Forward<'_>,
    s: &Stream,
    x: &mut Scratch,
    logits: &mut DeviceBuffer<f32>,
    rows: &mut [Row<'_>],
) -> Result<(), LaunchError> {
    let (w, max_context) = (f.weights, f.max_context);
    let c = &w.config;
    let n = rows.len();
    let hidden = c.hidden_size as usize;
    let attention = c.attention_size() as usize;
    let kv = c.kv_size() as usize;
    let qkv = c.ssm_qkv_size() as usize;
    let inner = c.ssm_inner_size as usize;
    let rank = c.ssm_time_step_rank as usize;
    let ffn = c.intermediate_size as usize;
    let vocab = c.vocab_size as usize;

    k::batched_embedding_lookup(
        s,
        w.token_embd.view(),
        x.tokens.slice(0..n),
        x.hidden.slice_mut(0..n * hidden),
        n,
        hidden,
        vocab,
    )?;
    for (index, row) in rows.iter().enumerate() {
        row.buffers.vision.inject(
            s,
            x.hidden.slice_mut(index * hidden..(index + 1) * hidden),
            row.position,
            1,
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
                k::batched_per_head_rms_norm(
                    s,
                    at.q_norm.view(),
                    x.q.slice_mut(0..n * attention),
                    n,
                    c.num_attention_heads,
                    c.head_dim,
                    NORM_EPS,
                )?;
                k::batched_per_head_rms_norm(
                    s,
                    at.k_norm.view(),
                    x.k.slice_mut(0..n * kv),
                    n,
                    c.num_key_value_heads,
                    c.head_dim,
                    NORM_EPS,
                )?;
                let attention_layer = l / c.full_attention_interval;
                for (index, row) in rows.iter_mut().enumerate() {
                    let (q, kr) = (index * attention, index * kv);
                    k::rope(
                        s,
                        x.q.slice_mut(q..q + attention),
                        x.k.slice_mut(kr..kr + kv),
                        c.num_attention_heads,
                        c.num_key_value_heads,
                        c.head_dim,
                        c.rotary_dim,
                        row.position,
                        c.rope_theta,
                        row.buffers.vision.rope(),
                    )?;
                    let plane = row.buffers.kv.len() / 2;
                    let (k_cache, v_cache) = row.buffers.kv.as_mut_slice().split_at(plane);
                    let split = split_k_attention(c, row.position);
                    k::attention_write(
                        s,
                        x.q.slice(q..q + attention),
                        x.k.slice(kr..kr + kv),
                        x.v.slice(kr..kr + kv),
                        x.ssm_gate.slice(q..q + attention),
                        k_cache,
                        v_cache,
                        x.ssm_out.slice_mut(q..q + attention),
                        attention_layer,
                        row.position,
                        max_context,
                        c.num_attention_heads,
                        c.num_key_value_heads,
                        c.head_dim,
                        split.then(|| x.split_k.as_mut_slice()),
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
                if n > 1 {
                    // One launch pair for every session over its state
                    // (`LaunchSSMConvRecurrenceBatch`), as C++ runs it.
                    let len = (n - 1) * stride + rank;
                    let (alpha, beta) = if packed_controls.is_some() {
                        (x.packed.slice(0..len), x.packed.slice(rank..rank + len))
                    } else {
                        (x.alpha.slice(0..len), x.beta.slice(0..len))
                    };
                    let mut conv = Vec::with_capacity(n);
                    let mut deltanet = Vec::with_capacity(n);
                    let mut conv_in = Vec::with_capacity(n);
                    let mut deltanet_in = Vec::with_capacity(n);
                    let (mut replay_qkv, mut replay_alpha, mut replay_beta) = (
                        Vec::with_capacity(n),
                        Vec::with_capacity(n),
                        Vec::with_capacity(n),
                    );
                    let (mut replay_position, mut replay_enabled) =
                        (Vec::with_capacity(n), Vec::with_capacity(n));
                    for (index, row) in rows.iter_mut().enumerate() {
                        let b = &mut *row.buffers;
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
                                replay_position.push(Some(x.positions.slice(index..index + 1)));
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
                    let offsets: Vec<u32> = (0..n as u32).collect();
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
                        &offsets,
                        &vec![1; n],
                        n as u32,
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
                    for (index, row) in rows.iter_mut().enumerate() {
                        let control = index * stride;
                        let (alpha, beta) = if packed_controls.is_some() {
                            (
                                x.packed.slice(control..control + rank),
                                x.packed.slice(control + rank..control + 2 * rank),
                            )
                        } else {
                            (
                                x.alpha.slice(control..control + rank),
                                x.beta.slice(control..control + rank),
                            )
                        };
                        let b = &mut *row.buffers;
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
                                Some(x.positions.slice(index..index + 1)),
                                Some(ring.enabled.as_slice()),
                            ),
                            None => (None, None, None, None, None),
                        };
                        let io = b.recurrent.io();
                        k::ssm_conv_recurrence_rows(
                            s,
                            x.ssm_qkv.slice(index * qkv..(index + 1) * qkv),
                            r.conv1d.view(),
                            io.conv,
                            io.conv_in,
                            x.conv_out.slice_mut(index * qkv..(index + 1) * qkv),
                            io.deltanet,
                            io.deltanet_in,
                            alpha,
                            beta,
                            r.a.view(),
                            r.dt.view(),
                            Some(r.norm.view()),
                            Some(x.ssm_gate.slice(index * inner..(index + 1) * inner)),
                            Some(x.ssm_out.slice_mut(index * inner..(index + 1) * inner)),
                            c.ssm_layer_index(l),
                            qkv,
                            c.ssm_group_count,
                            c.ssm_time_step_rank,
                            c.ssm_state_size,
                            c.ssm_value_size(),
                            1,
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
        for (index, row) in rows.iter_mut().enumerate() {
            if let Some(tap) = row.taps.iter().position(|&t| t == l) {
                s.copy(
                    x.hidden.slice(index * hidden..(index + 1) * hidden),
                    row.buffers
                        .target_features
                        .slice_mut(tap * hidden..(tap + 1) * hidden),
                )?;
            }
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
    project(
        s,
        &w.output,
        x.normed.slice(0..n * hidden),
        logits.slice_mut(0..n * vocab),
        n,
        vocab,
        hidden,
    )?;
    k::batched_argmax(
        s,
        logits.slice(0..n * vocab),
        x.sampled.slice_mut(0..n),
        n,
        vocab,
        x.ffn_out.slice_mut(0..n * hidden),
        n * hidden,
    )?;
    for (index, row) in rows.iter_mut().enumerate() {
        s.copy(
            logits.slice(index * vocab..(index + 1) * vocab),
            row.buffers.logits.as_mut_slice(),
        )?;
    }
    Ok(())
}
