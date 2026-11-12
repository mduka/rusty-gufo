//! Nemotron-H kernels. The launchers are generated from `launch.toml`; the
//! generated FFI calls are the crate's only `unsafe`. [`sizes`] holds the
//! kernels' constants and the buffer sizes their callers allocate.

#![deny(unsafe_code)]

#[allow(unsafe_code)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/launch.rs"));
}

pub use generated::*;

/// The kernel headers' constants (`kernels.hpp`, `gemm.hpp`, `gemv.hpp`,
/// `sample.hpp`, `drafter_kernels.hpp`, `vision.hpp`) and the scratch sizes
/// the launchers take.
pub mod sizes {
    /// Bytes of one Q8_1 block of 32 values (`kQ8_1BlockBytes`).
    pub const Q8_1_BLOCK_BYTES: usize = 36;
    /// Most rows `router_logits` takes (`kRouterMaxTokens`).
    pub const ROUTER_MAX_TOKENS: usize = 8;
    /// Key splits per KV head of `attention_short` (`kAttentionSplits`).
    pub const ATTENTION_SPLITS: usize = 64;
    /// Most queries of `attention_short` (`kAttentionShortTokens`).
    pub const ATTENTION_SHORT_TOKENS: usize = 8;
    /// Weight rows per tile of `nvfp4_gemm` (`kNvfp4GemmRows`).
    pub const NVFP4_GEMM_ROWS: usize = 128;
    /// Weight rows and slots per tile of the routed IQ2_XXS product.
    pub const IQ2XXS_ROUTED_ROWS: usize = 256;
    pub const IQ2XXS_ROUTED_TOKENS: usize = 32;
    /// Most activation rows of the vector products (`kGemvMaxTokens`).
    pub const GEMV_MAX_TOKENS: usize = 8;
    /// Most slots `iq2xxs_route_short` groups (`kRouteShortSlots`).
    pub const ROUTE_SHORT_SLOTS: usize = 128;
    /// Candidates one logit head holds at most (`kLogitHeadCandidates`).
    pub const LOGIT_HEAD_CANDIDATES: usize = 1024;
    /// Bytes of a logit head record: F64 softmax sum, F32 maximum, i32
    /// count (`LogitHeadRecord`).
    pub const LOGIT_HEAD_RECORD_BYTES: usize = 16;
    /// Bytes of a logit head candidate: u32 token, F32 value.
    pub const LOGIT_HEAD_CANDIDATE_BYTES: usize = 8;
    /// Candidates per LiliCorr slot (`kLiliCorrTopK`).
    pub const LILICORR_TOP_K: usize = 8;
    /// Head width of the ViT's attention (`kVisionHeadDim`).
    pub const VISION_HEAD_DIM: usize = 80;

    /// `partial` floats of `attention_short` for `n_tokens` queries.
    pub fn attention_short_partial(n_tokens: usize, heads: usize) -> usize {
        n_tokens * heads * ATTENTION_SPLITS * (128 + 2)
    }

    /// Tiles a routing of `slots` slots over `experts` experts can need
    /// (`Iq2xxsRouteMaxTiles`); `tiles` holds four i32 per tile.
    pub fn iq2xxs_route_max_tiles(slots: usize, experts: usize) -> usize {
        slots / IQ2XXS_ROUTED_TOKENS + experts
    }

    /// Device scratch `logit_heads` needs for `rows` rows of `vocab` logits
    /// (`LogitHeadScratchBytes`).
    pub fn logit_head_scratch_bytes(rows: usize, vocab: usize) -> usize {
        const SPAN: usize = 256 * 16;
        const HISTOGRAM: usize = 512 + 1;
        let align = |n: usize| n.div_ceil(256) * 256;
        let parts = rows * vocab.div_ceil(SPAN);
        align(parts * 4)
            + align(parts * 8)
            + align(parts * HISTOGRAM * 4)
            + align(parts * HISTOGRAM * 4)
            + align(rows * 4)
            + align(rows * 4)
    }

    /// A logit head record's softmax sum, maximum and candidate count.
    pub fn logit_head_record(bytes: &[u8; LOGIT_HEAD_RECORD_BYTES]) -> (f64, f32, i32) {
        let (sum, rest) = bytes.split_at(8);
        let (maximum, count) = rest.split_at(4);
        let word = |b: &[u8]| [b[0], b[1], b[2], b[3]];
        let mut sum_bytes = [0u8; 8];
        sum_bytes.copy_from_slice(sum);
        (
            f64::from_le_bytes(sum_bytes),
            f32::from_le_bytes(word(maximum)),
            i32::from_le_bytes(word(count)),
        )
    }
}
