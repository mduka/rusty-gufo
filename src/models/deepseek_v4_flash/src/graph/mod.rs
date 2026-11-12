//! The ROCm executor of the DS4 engine (`runtime/rocm_graph.cpp`): per
//! session one fixed set of decode tensors and the persistent KV state,
//! prompt chunks through a batch workspace that sessions share, and the
//! layer encoders that issue the C++ launch sequence.

// The encoders forward the launch parameters of the C++ functions.
#![allow(clippy::too_many_arguments)]

mod alloc;
mod batch;
mod decode;
mod dspark;
mod prefill;
mod snapshot;
mod verify;

pub(crate) use alloc::lent;
pub use alloc::{
    BatchWorkspace, Graph, WorkspaceLease, prefill_capacity, raw_cap_for_context,
    resume_prefill_min_tokens,
};
pub use batch::{BatchItem, eval_sessions_batch};
pub use decode::Target;
pub use dspark::{DraftItem, dspark_draft_head_batch};
pub use snapshot::{PAYLOAD_MAGIC, PAYLOAD_VERSION, Restored};
pub(crate) use verify::greedy_accept;
pub use verify::{VerifyItem, verify_sessions_batch};

use gufo_deepseek_v4_flash_kernels::{Rope, Weight};

use crate::gguf::TensorInfo;
use crate::shape::*;

/// The weight of a bound tensor in the target mapping.
pub(crate) fn w(t: &TensorInfo) -> Weight {
    Weight::target(t.offset)
}

/// The attention compression ratio of `layer` (`ds4_layer_compress_ratio`,
/// checked against the metadata at load).
pub(crate) fn ratio(layer: u32) -> u32 {
    expected_compress_ratio(layer)
}

/// The RoPE of `layer` at `pos0`: compressed layers use the YaRN-scaled
/// compressor base, the others the plain base (`layer_rope_freq_base`,
/// `layer_rope_freq_scale` and the factors every encoder derives).
pub(crate) fn rope(layer: u32, pos0: u32) -> Rope {
    let compressed = ratio(layer) != 0;
    let freq_base = if compressed && COMPRESS_ROPE_FREQ_BASE > 0.0 {
        COMPRESS_ROPE_FREQ_BASE
    } else {
        ROPE_FREQ_BASE
    };
    let freq_scale = if !compressed || ROPE_SCALE_FACTOR <= 0.0 {
        1.0
    } else {
        1.0 / ROPE_SCALE_FACTOR
    };
    let ext_factor = if compressed && ROPE_SCALE_FACTOR > 1.0 {
        1.0
    } else {
        0.0
    };
    let mut attn_factor = 1.0f32;
    if ext_factor != 0.0 && freq_scale > 0.0 {
        attn_factor /= 1.0 + 0.1 * (1.0f32 / freq_scale).ln();
    }
    Rope {
        n_rot: N_ROT,
        pos0,
        n_ctx_orig: if compressed { ROPE_ORIG_CTX as u32 } else { 0 },
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        beta_fast: ROPE_YARN_BETA_FAST,
        beta_slow: ROPE_YARN_BETA_SLOW,
    }
}

pub(crate) const HC_DIM: u64 = N_HC as u64 * N_EMBD as u64;
pub(crate) const MIX_HC: u64 = 2 * N_HC as u64 + N_HC as u64 * N_HC as u64;
pub(crate) const Q_DIM: u64 = N_HEAD as u64 * N_HEAD_DIM as u64;
pub(crate) const LOW_DIM: u64 = N_OUT_GROUP as u64 * N_LORA_O as u64;
pub(crate) const GROUP_DIM: u32 = N_HEAD_DIM * (N_HEAD / N_OUT_GROUP);
pub(crate) const INDEXER_Q_DIM: u64 = N_INDEXER_HEAD as u64 * N_INDEXER_HEAD_DIM as u64;
pub(crate) const F32: u64 = 4;
pub(crate) const NEG_INF: f32 = -1.0e30;
/// Prefix snapshots a verification block keeps for partial rollback
/// (`DS4_SPEC_PREFIX_SLOTS`).
pub(crate) const SPEC_PREFIX_SLOTS: u32 = 4;
