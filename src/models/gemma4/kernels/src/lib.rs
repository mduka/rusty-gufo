//! Gemma 4 kernels. The launchers are generated from `launch.toml`; the
//! generated FFI calls are the crate's only `unsafe`. [`sizes`] holds the
//! kernels' constants and the buffer sizes their callers allocate.

#![deny(unsafe_code)]

#[allow(unsafe_code)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/launch.rs"));
}

pub use generated::*;

/// The kernel headers' constants (`kernels.hpp`, `candidates.hpp`,
/// `gemv.hpp`, `half_gemm.hpp`, `moe.hpp`, `wmma_gemv.hpp`, the vision
/// kernels) and the buffer sizes the launch contracts take.
pub mod sizes {
    /// Rows the split (decode and verification) attention takes; more rows
    /// take the prefill kernels (`kSplitRows`).
    pub const SPLIT_ROWS: usize = 16;
    /// Most rotated pairs of a derived-key layer (`kMaxDerivedKeyPairs`); the
    /// attention launchers take exactly this many.
    pub const MAX_DERIVED_KEY_PAIRS: usize = 64;
    /// Layers `move_key` takes at most (`MoveKeyArgs::kMaxLayers`).
    pub const MOVE_KEY_MAX_LAYERS: usize = 96;
    /// Rows of `draft_concat` at most (`DraftConcatArgs::hidden`).
    pub const DRAFT_CONCAT_MAX_ROWS: usize = 8;
    /// Logits ranked per row by `logit_candidates`, and the most extra
    /// tokens it gathers (`kCandidateTop`, `kMaxCandidateExtra`).
    pub const CANDIDATE_TOP: usize = 64;
    pub const MAX_CANDIDATE_EXTRA: usize = 1024;
    /// `counts[r]` of a row that needs its full logits (`kCandidateOverflow`).
    pub const CANDIDATE_OVERFLOW: u32 = u32::MAX;
    /// Rows of the binary16 and WMMA small-batch projections
    /// (`kMaxHalfGemvRows`, `kWmmaGemvRows`).
    pub const MAX_HALF_GEMV_ROWS: usize = 16;
    pub const WMMA_GEMV_ROWS: usize = 16;
    /// Slots of one expert group, and ints per group in the group table
    /// (`kMaxGroupSlots`, `kGroupInts`).
    pub const MAX_GROUP_SLOTS: usize = 16;
    pub const GROUP_INTS: usize = 2 + MAX_GROUP_SLOTS;
    /// Router widths the routing kernel holds in registers
    /// (`kMaxRouterHidden`), and the most experts and slots per row it takes.
    pub const MAX_ROUTER_HIDDEN: usize = 3072;
    pub const MAX_EXPERTS: usize = 256;
    pub const MAX_EXPERTS_USED: usize = 8;
    /// Bucket rows per tile of `routed_half_gemm` (`kRoutedHalfTileRows`).
    pub const ROUTED_HALF_TILE_ROWS: usize = 112;
    /// Candidates `mtp_top_candidates` selects (`kMtpCandidates`).
    pub const MTP_CANDIDATES: usize = 64;
    /// Values per vocabulary tile of `batched_argmax` (`kArgmaxTile`).
    pub const ARGMAX_TILE: usize = 4096;
    /// Rows of the tiled Q8_1 activation per tile, and bytes per tile and
    /// 32 values (codes and scales, then the sums sidecar).
    pub const Q8_ACT_TILE_ROWS: usize = 16;
    pub const Q8_ACT_TILE_BYTES: usize = 576 + 64;
    /// Rows of the tiles the prefill GEMMs read.
    pub const Q8_ACT_GEMM_ROWS: usize = 128;
    /// Bytes per input value of a WMMA activation pack, and the split
    /// partials after it in a WMMA projection's scratch.
    pub const WMMA_PACK_BYTES_PER_VALUE: usize = 68;
    pub const WMMA_PARTIALS_BYTES: usize = 6 * 16 * 8192 * 4;
    /// Rows a prefill copy of dense weights pads to, the values per stage of
    /// a compact copy and its bytes per stage (Q4_K, and Q5_K/Q6_K).
    pub const PREFILL_ROW_PAD: usize = 16;
    pub const PREFILL_STAGE_VALUES: usize = 128;
    pub const PREFILL_COMPACT_Q4K_BYTES: usize = 80;
    pub const PREFILL_COMPACT_Q6K_BYTES: usize = 112;

    /// Vision tower geometry (`vision/weights.hpp`, `vision/prompt.hpp`,
    /// `vision/encoder.hip`).
    pub const VISION_HIDDEN: usize = 1152;
    pub const VISION_HEADS: usize = 16;
    pub const VISION_HEAD_DIM: usize = 72;
    /// Head width of the attention operands (zero padded).
    pub const VISION_PADDED_DIM: usize = 80;
    pub const VISION_FFN: usize = 4304;
    /// FFN width of the down projection's input (zero padded).
    pub const VISION_FFN_PADDED: usize = 4320;
    pub const VISION_LAYERS: usize = 27;
    /// Rows of each learned position table (x and y).
    pub const VISION_POSITIONS: usize = 10240;
    pub const VISION_EPS: f32 = 1e-6;
    pub const VISION_ROPE_THETA: f32 = 100.0;
    pub const VISION_PATCH_SIZE: usize = 16;
    pub const VISION_PATCH_VALUES: usize = 3 * 16 * 16;
    pub const VISION_POOL_SIZE: usize = 3;
    pub const VISION_SIDE_MULTIPLE: usize = VISION_PATCH_SIZE * VISION_POOL_SIZE;
    /// Rotary pairs per axis of a vision head.
    pub const VISION_ROPE_PAIRS: usize = VISION_HEAD_DIM / 4;
    /// Keys per attention tile; key sequences pad to whole tiles.
    pub const VISION_KEY_TILE: usize = 64;

    /// Keys per chunk of the split attention (`SplitChunk`).
    pub fn attention_chunk(head_dim: usize) -> usize {
        if head_dim <= 256 { 128 } else { 512 }
    }

    /// `partials` floats of `attention_ring` (head_dim 256) for `rows`
    /// rows whose window plus rows fit `partial_keys` keys.
    pub fn attention_ring_partial_floats(rows: usize, heads: usize, partial_keys: usize) -> usize {
        rows * heads * (partial_keys / 128 + 2) * (256 + 2)
    }

    /// `partials` floats of `attention_linear` and `attention_derived` over
    /// caches of `slots` keys.
    pub fn attention_linear_partial_floats(
        rows: usize,
        heads: usize,
        head_dim: usize,
        slots: usize,
    ) -> usize {
        let chunk = attention_chunk(head_dim);
        rows * heads * (slots.div_ceil(chunk) + 1) * (head_dim + 2)
    }

    /// Bytes the tiled Q8_1 activation of `rows` rows of `k` values takes
    /// (what the quantizers write; `QuantizedActivationBytes`).
    pub fn q8_activation_bytes(rows: usize, k: usize) -> usize {
        rows.div_ceil(Q8_ACT_TILE_ROWS) * (k / 32) * Q8_ACT_TILE_BYTES
    }

    /// Bytes a prefill GEMM reads of the tiled Q8_1 activation of `rows`
    /// rows (whole 128-row tiles).
    pub fn q8_activation_read_bytes(rows: usize, k: usize) -> usize {
        rows.div_ceil(Q8_ACT_GEMM_ROWS)
            * (Q8_ACT_GEMM_ROWS / Q8_ACT_TILE_ROWS)
            * (k / 32)
            * Q8_ACT_TILE_BYTES
    }

    /// Bytes of the activation pack of a `width`-wide projection input.
    pub fn wmma_pack_bytes(width: usize) -> usize {
        WMMA_PACK_BYTES_PER_VALUE * width
    }

    /// Bytes of a WMMA projection's scratch for inputs up to `max_k` values
    /// (`WmmaGemvScratchBytes`).
    pub fn wmma_gemv_scratch_bytes(max_k: usize) -> usize {
        wmma_pack_bytes(max_k) + WMMA_PARTIALS_BYTES
    }

    /// Bytes of `logit_candidates`' workspace for `rows` rows
    /// (`CandidateWorkBytes`).
    pub fn candidate_work_bytes(rows: usize) -> usize {
        let align = |n: usize| n.div_ceil(256) * 256;
        const BINS: usize = 4096;
        const BOUNDARY: usize = 2048;
        align(rows * BINS * 4)
            + align(rows * 2 * 4)
            + align(rows * (CANDIDATE_TOP + MAX_CANDIDATE_EXTRA) * 4)
            + align(rows * BOUNDARY * 8)
    }

    /// The token and logit of a `logit_candidates` record (a u32 pair).
    pub fn candidate(record: [u32; 2]) -> (u32, f32) {
        (record[0], f32::from_bits(record[1]))
    }

    /// Entries of `mtp_top_candidates`' `ids` and `scratch_ids`.
    pub fn mtp_candidate_workspace(vocab: usize) -> usize {
        (vocab.div_ceil(1024) + 2) * MTP_CANDIDATES
    }

    /// Floats of `batched_argmax`'s scratch that tile `batch` rows of
    /// `vocab` logits.
    pub fn argmax_scratch_floats(batch: usize, vocab: usize) -> usize {
        2 * batch * vocab.div_ceil(ARGMAX_TILE)
    }

    /// Entries of the group table of `expert_groups` over `experts` experts.
    pub fn expert_group_table_len(experts: usize) -> usize {
        1 + experts * GROUP_INTS
    }

    /// Compact rows of `routed_compact` (`RoutedCompactRows`).
    pub fn routed_compact_rows(slots: usize, experts: usize) -> usize {
        slots + experts * 15
    }

    /// Entries of a routed tile map of `rows`-row tiles
    /// (`RoutedTileCapacity`).
    pub fn routed_tile_capacity(slots: usize, experts: usize, rows: usize) -> usize {
        (slots + 15 * experts) / rows + experts
    }

    /// Binary16 values of a decoded prefill copy of [m][k] weights.
    pub fn prefill_decoded_values(m: usize, k: usize) -> usize {
        m.div_ceil(PREFILL_ROW_PAD) * PREFILL_ROW_PAD * k
    }

    /// Bytes of a compact prefill copy of [m][k] weights at `stage_bytes`
    /// per 128 values (`PREFILL_COMPACT_Q4K_BYTES` or `..._Q6K_BYTES`).
    pub fn prefill_compact_bytes(m: usize, k: usize, stage_bytes: usize) -> usize {
        m.div_ceil(PREFILL_ROW_PAD) * PREFILL_ROW_PAD * (k / PREFILL_STAGE_VALUES) * stage_bytes
    }
}
