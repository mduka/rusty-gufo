//! One decode step of two to eight sessions, a port of the extension's
//! `QwenGpuExecutor::ForwardTokenBatch` (`hip/batched_decode.cpp`) under the
//! production policy. The projections share weight loads through the exact
//! small-batch GEMM or run the decode GEMV per row (`LaunchProjection`);
//! per-head norm, RoPE and attention run per session on its own cache, the
//! recurrence of all sessions runs in one launch pair over their states, and
//! the FFN is the routed MoE of [`crate::moe::rows`] (`LaunchMoeFfn`), in the
//! C++ order. Every row is bit-identical to the single-session decode step.

use gufo_hip::{DeviceBuffer, LaunchError, Stream};
use gufo_qwen35moe_kernels as k;

use crate::arena::{Forward, MAX_VERIFY_ROWS, REPLAY_CAPACITY, Scratch, SessionBuffers};
use crate::config::NORM_EPS;
use crate::decode::split_k_attention;
use crate::verify::{project, ssm_controls};
use crate::weights::Mixer;

/// One session of a batched step.
pub(crate) struct Row<'a> {
    pub(crate) buffers: &'a mut SessionBuffers,
    pub(crate) position: u32,
    /// Layers whose hidden row the draft reads, `[tap][hidden]`.
    pub(crate) taps: &'a [u32],
}

/// Runs the tokens already in `x.tokens` for `rows`, whose positions (for
/// the replay capture) are in `x.positions`; leaves each row's logits in its
/// session's logits, its greedy token in `x.sampled` and its final hidden
/// row in row `index` of `x.hidden`. `logits` holds `[row][vocab]` while
/// the step runs.
pub(crate) fn decode_rows(
    f: Forward<'_>,
    s: &Stream,
    x: &mut Scratch,
    logits: &mut DeviceBuffer<f32>,
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
    run(f, s, x, logits, rows)
}

/// The passes of [`decode_rows`].
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
                let (packed_controls, stride) = ssm_controls(s, x, r, n, rank, hidden)?;
                // One launch pair for every session over its state
                // (`LaunchSSMConvRecurrenceBatch`), as C++ runs it.
                let len = (n - 1) * stride + rank;
                let (alpha, beta) = if packed_controls {
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
                let (mut replay_position, mut replay_enabled, mut capacity) = (
                    Vec::with_capacity(n),
                    Vec::with_capacity(n),
                    Vec::with_capacity(n),
                );
                for (index, row) in rows.iter_mut().enumerate() {
                    let b = &mut *row.buffers;
                    let io = b.recurrent.io();
                    conv.push(io.conv);
                    conv_in.push(io.conv_in);
                    deltanet.push(io.deltanet);
                    deltanet_in.push(io.deltanet_in);
                    match b.replay.as_mut() {
                        Some(ring) => {
                            capacity.push(ring.capacity_u32());
                            replay_qkv.push(Some(ring.qkv.as_mut_slice()));
                            replay_alpha.push(Some(ring.alpha.as_mut_slice()));
                            replay_beta.push(Some(ring.beta.as_mut_slice()));
                            replay_position.push(Some(x.positions.slice(index..index + 1)));
                            replay_enabled.push(Some(ring.enabled.as_slice()));
                        }
                        None => {
                            capacity.push(REPLAY_CAPACITY as u32);
                            replay_qkv.push(None);
                            replay_alpha.push(None);
                            replay_beta.push(None);
                            replay_position.push(None);
                            replay_enabled.push(None);
                        }
                    }
                }
                let offsets: Vec<u32> = (0..n as u32).collect();
                k::ssm_conv_recurrence_sequences_ring(
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
                    &capacity,
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
        crate::moe::rows(
            s,
            &layer.moe,
            c,
            &mut x.moe,
            x.normed.slice(0..n * hidden),
            x.ffn_gate.as_mut_slice(),
            x.ffn_up.as_mut_slice(),
            x.ffn_out.slice_mut(0..n * hidden),
            n,
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
