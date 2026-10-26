//! The single-token forward pass, a port of `ExecuteDecodeStep`
//! (`hip/decode_step.cpp`) under the production policy: fused
//! QK-norm/RoPE/KV write, FP16 KV cache, FP32 recurrent state.

use gufo_hip::{LaunchError, Stream};
use gufo_qwen_kernels as k;

use crate::arena::{Forward, Scratch, SessionBuffers};
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
/// for split-K attention, the only launch that takes it as a scalar; without
/// it the launches are the same at every position and can be replayed. The
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
    let ffn = c.intermediate_size as usize;
    let vocab = c.vocab_size as usize;
    let kv_plane = st.kv.len() / 2;

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
        k::rms_norm(
            s,
            x.hidden.slice(0..hidden),
            layer.attn_norm.view(),
            x.normed.slice_mut(0..hidden),
            hidden,
            NORM_EPS,
        )?;
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
                let (replay_qkv, replay_alpha, replay_beta, replay_position, replay_enabled) =
                    match st.replay.as_mut() {
                        Some(ring) => (
                            Some(ring.qkv.as_mut_slice()),
                            Some(ring.alpha.as_mut_slice()),
                            Some(ring.beta.as_mut_slice()),
                            Some(x.decode_input.second().values()),
                            Some(ring.enabled.as_slice()),
                        ),
                        None => (None, None, None, None, None),
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
                k::ssm_conv_recurrence(
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
        k::fused_swiglu_gemv(
            s,
            layer.ffn_gate.view(),
            layer.ffn_up.view(),
            x.normed.slice(0..hidden),
            x.ffn_act.slice_mut(0..ffn),
            ffn,
            hidden,
        )?;
        k::gemv(
            s,
            layer.ffn_down.view(),
            x.ffn_act.slice(0..ffn),
            x.ffn_out.slice_mut(0..hidden),
            hidden,
            ffn,
        )?;
        k::residual_add(
            s,
            x.ffn_out.slice(0..hidden),
            x.hidden.slice_mut(0..hidden),
            hidden,
        )?;
        if let Some(tap) = taps.iter().position(|&t| t == l) {
            // Part of the decode graph, as the C++ tap memcpy.
            s.graph_copy(
                x.hidden.slice(0..hidden),
                st.target_features
                    .slice_mut(tap * hidden..(tap + 1) * hidden),
            )?;
        }
    }

    if compute_logits {
        k::rms_norm(
            s,
            x.hidden.slice(0..hidden),
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
