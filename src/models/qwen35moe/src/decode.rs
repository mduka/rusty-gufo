//! The single-token forward pass, a port of the extension's
//! `ExecuteDecodeStep` (`hip/decode_step.cpp`) under the production policy:
//! fused QK-norm/RoPE/KV write, FP16 KV cache, FP32 recurrent state, the
//! routed MoE FFN of [`crate::moe::decode`], and on 2048-wide rows each
//! residual add fused with the RMSNorm that reads its sum (the MoE combine
//! included).

use gufo_hip::{LaunchError, Stream};
use gufo_qwen35moe_kernels as k;
use gufo_qwen35moe_kernels::sizes;

use crate::arena::{Forward, REPLAY_CAPACITY, Scratch, SessionBuffers};
use crate::config::{ModelConfig, NORM_EPS};
use crate::weights::Mixer;

/// From this sequence length decode attention splits the context
/// (`kSplitKDecodeAttentionMinContext`).
const SPLIT_K_MIN_CONTEXT: usize = 128;

/// Decode attention splits the context from 128 visible positions
/// (`IsSplitKDecodeAttentionSupported`).
pub(crate) fn split_k_attention(c: &ModelConfig, pos: u32) -> bool {
    pos as usize + 1 >= SPLIT_K_MIN_CONTEXT
        && c.num_attention_heads.is_multiple_of(c.num_key_value_heads)
        && c.head_dim == 256
}

/// One token through every layer (`ExecuteDecodeStep`), reading token and
/// position from `x.decode_input`. `split_k` carries the position
/// for split-K attention, the only launch that takes it as a scalar. The
/// hidden row after each layer in `taps` lands in `st.target_features`, and
/// with a replay ring the recurrence records its inputs there when enabled.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_step(
    f: Forward<'_>,
    s: &Stream,
    x: &mut Scratch,
    st: &mut SessionBuffers,
    compute_logits: bool,
    split_k: Option<u32>,
    taps: &[u32],
    pos: u32,
) -> Result<(), LaunchError> {
    let (w, max_context) = (f.weights, f.max_context);
    let c = &w.config;
    let hidden = c.hidden_size as usize;
    let attention = c.attention_size() as usize;
    let kv = c.kv_size() as usize;
    let qkv = c.ssm_qkv_size() as usize;
    let inner = c.ssm_inner_size as usize;
    let rank = c.ssm_time_step_rank as usize;
    let used = c.expert_used_count as usize;
    let vocab = c.vocab_size as usize;
    let kv_plane = st.kv.len() / 2;
    // Each residual add joins the RMSNorm that reads its sum; the arithmetic
    // is unchanged.
    let fuse_residual_norm = hidden == sizes::FUSED_RESIDUAL_NORM_DIM;

    k::embedding_lookup(
        s,
        w.token_embd.view(),
        x.decode_input.first(),
        x.hidden.slice_mut(0..hidden),
        hidden,
        vocab,
    )?;
    st.vision
        .inject(s, x.hidden.slice_mut(0..hidden), pos, 1, hidden as u32, 1)?;

    for (l, layer) in (0u32..).zip(&w.layers) {
        // Later layers get their attention norm from the previous layer's
        // fused residual add.
        if !fuse_residual_norm || l == 0 {
            k::rms_norm(
                s,
                x.hidden.slice(0..hidden),
                layer.attn_norm.view(),
                x.normed.slice_mut(0..hidden),
                hidden,
                NORM_EPS,
            )?;
        }
        match &layer.mixer {
            Mixer::Attention(at) => {
                let attention_layer = l / c.full_attention_interval;
                k::fused_qkv_projections(
                    s,
                    at.q.view(),
                    at.k.view(),
                    at.v.view(),
                    x.normed.slice(0..hidden),
                    x.ssm_qkv.slice_mut(0..2 * attention),
                    x.k.slice_mut(0..kv),
                    x.v.slice_mut(0..kv),
                    2 * attention,
                    kv,
                    hidden,
                )?;
                k::unpack_qg(
                    s,
                    x.ssm_qkv.slice(0..2 * attention),
                    x.q.slice_mut(0..attention),
                    x.ssm_gate.slice_mut(0..attention),
                    c.num_attention_heads,
                    c.head_dim,
                )?;
                let (k_cache, v_cache) = st.kv.as_mut_slice().split_at(kv_plane);
                k::fused_qk_norm_rope_kv_write(
                    s,
                    x.v.slice(0..kv),
                    at.q_norm.view(),
                    at.k_norm.view(),
                    x.q.slice_mut(0..attention),
                    x.k.slice_mut(0..kv),
                    k_cache,
                    v_cache,
                    attention_layer,
                    x.decode_input.second(),
                    max_context,
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    c.head_dim,
                    c.rotary_dim,
                    c.rope_theta,
                    NORM_EPS,
                    st.vision.rope(),
                )?;
                let (k_cache, v_cache) = st.kv.as_mut_slice().split_at(kv_plane);
                if let Some(pos) = split_k {
                    k::attention_split_k(
                        s,
                        x.q.slice(0..attention),
                        x.k.slice(0..kv),
                        x.v.slice(0..kv),
                        x.ssm_gate.slice(0..attention),
                        k_cache,
                        v_cache,
                        x.ssm_out.slice_mut(0..attention),
                        attention_layer,
                        pos,
                        max_context,
                        c.num_attention_heads,
                        c.num_key_value_heads,
                        c.head_dim,
                        x.split_k.as_mut_slice(),
                    )?;
                } else {
                    k::attention_graph(
                        s,
                        x.q.slice(0..attention),
                        x.k.slice(0..kv),
                        x.v.slice(0..kv),
                        x.ssm_gate.slice(0..attention),
                        k_cache,
                        v_cache,
                        x.ssm_out.slice_mut(0..attention),
                        attention_layer,
                        x.decode_input.second(),
                        max_context,
                        c.num_attention_heads,
                        c.num_key_value_heads,
                        c.head_dim,
                    )?;
                }
                k::gemv(
                    s,
                    at.output.view(),
                    x.ssm_out.slice(0..attention),
                    x.attn_out.slice_mut(0..hidden),
                    hidden,
                    attention,
                )?;
            }
            Mixer::Recurrent(r) => {
                // The ring's slots per layer (`GetSsmReplayCapture`): 16, or
                // more once a wide verification grew it.
                let (
                    replay_qkv,
                    replay_alpha,
                    replay_beta,
                    replay_position,
                    replay_enabled,
                    replay_capacity,
                ) = match st.replay.as_mut() {
                    Some(ring) => {
                        let capacity = ring.capacity_u32();
                        (
                            Some(ring.qkv.as_mut_slice()),
                            Some(ring.alpha.as_mut_slice()),
                            Some(ring.beta.as_mut_slice()),
                            Some(x.decode_input.second().values()),
                            Some(ring.enabled.as_slice()),
                            capacity,
                        )
                    }
                    None => (None, None, None, None, None, REPLAY_CAPACITY as u32),
                };
                k::fused_ssm_input_projections(
                    s,
                    r.qkv.view(),
                    r.gate.view(),
                    r.alpha.view(),
                    r.beta.view(),
                    x.normed.slice(0..hidden),
                    x.ssm_qkv.slice_mut(0..qkv),
                    x.ssm_gate.slice_mut(0..inner),
                    x.alpha.slice_mut(0..rank),
                    x.beta.slice_mut(0..rank),
                    hidden,
                    qkv,
                    inner,
                    rank,
                )?;
                let io = st.recurrent.io();
                k::ssm_conv_recurrence_ring(
                    s,
                    x.ssm_qkv.slice(0..qkv),
                    r.conv1d.view(),
                    io.conv,
                    io.conv_in,
                    x.conv_out.slice_mut(0..qkv),
                    io.deltanet,
                    io.deltanet_in,
                    x.alpha.slice(0..rank),
                    x.beta.slice(0..rank),
                    r.a.view(),
                    r.dt.view(),
                    r.norm.view(),
                    x.ssm_gate.slice(0..inner),
                    x.ssm_out.slice_mut(0..inner),
                    c.ssm_layer_index(l),
                    qkv,
                    c.ssm_group_count,
                    c.ssm_time_step_rank,
                    c.ssm_state_size,
                    c.ssm_value_size(),
                    replay_qkv,
                    replay_alpha,
                    replay_beta,
                    replay_position,
                    replay_enabled,
                    replay_capacity,
                )?;
                k::gemv(
                    s,
                    r.out.view(),
                    x.ssm_out.slice(0..inner),
                    x.attn_out.slice_mut(0..hidden),
                    hidden,
                    inner,
                )?;
            }
        }
        if fuse_residual_norm {
            k::fused_residual_rms_norm(
                s,
                x.attn_out.slice(0..hidden),
                layer.ffn_norm.view(),
                x.hidden.slice_mut(0..hidden),
                x.normed.slice_mut(0..hidden),
                hidden,
                NORM_EPS,
            )?;
        } else {
            k::residual_add(
                s,
                x.attn_out.slice(0..hidden),
                x.hidden.slice_mut(0..hidden),
                hidden,
            )?;
            k::rms_norm(
                s,
                x.hidden.slice(0..hidden),
                layer.ffn_norm.view(),
                x.normed.slice_mut(0..hidden),
                hidden,
                NORM_EPS,
            )?;
        }
        // The residual add after the FFN, with the next RMSNorm when fused:
        // the next layer's attention norm, or the output norm after the last
        // layer. The MoE combine then joins that residual add as well.
        let next_norm = match w.layers.get(l as usize + 1) {
            Some(next) => Some(&next.attn_norm),
            None => compute_logits.then_some(&w.output_norm),
        };
        let fuse_combine = fuse_residual_norm && next_norm.is_some();
        crate::moe::decode(
            s,
            &layer.moe,
            c,
            &mut x.moe,
            x.normed.slice(0..hidden),
            (!fuse_combine).then(|| x.ffn_out.slice_mut(0..hidden)),
        )?;
        match next_norm {
            Some(norm) if fuse_combine => {
                let m = &x.moe;
                k::fused_moe_residual_rms_norm(
                    s,
                    m.down_e.slice(0..used * hidden),
                    m.weights.slice(0..used),
                    m.shared_out.slice(0..hidden),
                    m.shared_gate.slice(0..1),
                    used as u32,
                    norm.view(),
                    x.hidden.slice_mut(0..hidden),
                    x.normed.slice_mut(0..hidden),
                    hidden,
                    NORM_EPS,
                )?;
            }
            _ => k::residual_add(
                s,
                x.ffn_out.slice(0..hidden),
                x.hidden.slice_mut(0..hidden),
                hidden,
            )?,
        }
        if let Some(tap) = taps.iter().position(|&t| t == l) {
            // The C++ tap memcpy (a graph node where a step is captured).
            s.graph_copy(
                x.hidden.slice(0..hidden),
                st.target_features
                    .slice_mut(tap * hidden..(tap + 1) * hidden),
            )?;
        }
    }

    if compute_logits {
        // The last fused residual add already applied the output norm.
        if !fuse_residual_norm {
            k::rms_norm(
                s,
                x.hidden.slice(0..hidden),
                w.output_norm.view(),
                x.normed.slice_mut(0..hidden),
                hidden,
                NORM_EPS,
            )?;
        }
        k::gemv(
            s,
            w.output.view(),
            x.normed.slice(0..hidden),
            st.logits.as_mut_slice(),
            vocab,
            hidden,
        )?;
        // The FFN output row is free again and holds the argmax tiles,
        // as in the C++ step: its length selects the tiled kernel.
        k::batched_argmax(
            s,
            st.logits.as_slice(),
            x.sampled.as_mut_slice(),
            1,
            vocab,
            x.ffn_out.slice_mut(0..hidden),
            hidden,
        )?;
    }
    Ok(())
}
