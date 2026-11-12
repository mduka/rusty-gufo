//! Qwen3.6 35B-A3B (`qwen35moe`) kernels: model-private copies of the HIP
//! kernels of the strycast-qwen35moe-gufo-extension (`vendor/`, commit
//! 4882b7a), compiled with its C++ build's per-source flags. The launchers are
//! generated from `launch.toml`; the generated FFI calls are the crate's only
//! `unsafe`. [`sizes`] holds the kernels' constants and the buffer sizes their
//! callers allocate, [`blas`] the hipBLAS/hipBLASLt choices of the extension's
//! GEMM helpers, which gufo-hip's `Blas` and `BlasLt` carry out.

#![deny(unsafe_code)]

#[allow(unsafe_code)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/launch.rs"));
}

pub use generated::*;

/// The kernel headers' constants and the buffer sizes the launch contracts
/// take.
pub mod sizes {
    /// Bytes of one (position, KV head) row of the 8-bit KV cache: 256 int8
    /// values and 8 FP16 scales (`kv_q8::kRowBytes`); 256-wide heads only.
    pub const KV_Q8_ROW_BYTES: usize = 272;
    /// Head width of the 8-bit KV cache (`kv_q8::kHeadDim`).
    pub const KV_Q8_HEAD_DIM: usize = 256;
    /// Split-K partitions of decode attention
    /// (`kSplitKDecodeAttentionMaxSplits`).
    pub const DECODE_ATTENTION_MAX_SPLITS: usize = 32;
    /// SSM replay slots of an ordinary verification (`kSsmReplayCapacity`);
    /// the `_ring` launchers take wider rings (a power of two).
    pub const SSM_REPLAY_CAPACITY: usize = 16;
    /// Candidates per MTP draft row (`qwen38_flash_next::kMtpCandidates`).
    pub const MTP_CANDIDATES: usize = 64;
    /// Rows per tile of the tiled Q8_1 activation and bytes per tile and 32
    /// values (codes and scales, then the sums sidecar).
    pub const Q8_ACT_TILE_ROWS: usize = 16;
    pub const Q8_ACT_TILE_BYTES: usize = 576 + 64;
    /// Width of the fused residual/RMSNorm/quantize kernels
    /// (`IsFusedResidualRMSNormQuantizeQ8_1Supported`).
    pub const FUSED_RESIDUAL_NORM_DIM: usize = 2048;
    /// Floor of the routers' weight sum before renormalizing (2^-14).
    pub const MOE_WEIGHT_SUM_FLOOR: f32 = 1.0 / 16384.0;
    /// Row shift of a grouped BF16 tile entry (`kGroupedTileShift`): entries
    /// are `expert << 20 | first row`.
    pub const GROUPED_TILE_SHIFT: u32 = 20;
    /// Most experts the routers and grouped kernels take
    /// (`kGroupedMaxExperts`); the one-token fused router takes 256.
    pub const MAX_EXPERTS: usize = 1024;
    pub const MAX_FUSED_ROUTER_EXPERTS: usize = 256;

    /// `split_k_scratch` floats of one decode attention row
    /// (`DecodeAttentionScratchElements`).
    pub const fn decode_attention_scratch(num_heads: usize, head_dim: usize) -> usize {
        num_heads * DECODE_ATTENTION_MAX_SPLITS * (head_dim + 2)
    }

    /// Bytes of one layer of an 8-bit K or V plane (`kv_q8::LayerBytes`).
    pub const fn kv_q8_layer_bytes(max_context: usize, num_kv_heads: usize) -> usize {
        max_context * num_kv_heads * KV_Q8_ROW_BYTES
    }

    /// Bytes the tiled Q8_1 activation of `rows` rows of `k` values takes.
    pub const fn q8_activation_bytes(rows: usize, k: usize) -> usize {
        rows.div_ceil(Q8_ACT_TILE_ROWS) * (k / 32) * Q8_ACT_TILE_BYTES
    }

    /// Bytes a prefill GEMM reads of the tiled Q8_1 activation of `rows`
    /// rows (whole 128-row tiles).
    pub const fn q8_activation_gemm_bytes(rows: usize, k: usize) -> usize {
        rows.div_ceil(128) * 8 * (k / 32) * Q8_ACT_TILE_BYTES
    }

    const fn round_256(bytes: usize) -> usize {
        bytes.div_ceil(256) * 256
    }

    /// Bytes of the tier's Q8_1 rows of `n` rows of `k` values (rows padded
    /// to 512 values, 36 bytes per 32).
    pub const fn mmq_q8_1_bytes(n: usize, k: usize) -> usize {
        n * k.div_ceil(512) * 512 / 32 * 36
    }

    /// Scratch of `mmq_moe_vec`: the Q8_1 rows and the expert-ordered slot
    /// maps.
    pub const fn mmq_moe_vec_scratch(n_tokens: usize, k: usize) -> usize {
        round_256(mmq_q8_1_bytes(n_tokens, k) + 8 * n_tokens)
    }

    /// Scratch of `mmq_moe_gated_vec`: the Q8_1 rows and the larger of both
    /// expert group layouts.
    pub const fn mmq_moe_gated_vec_scratch(n_tokens: usize, k: usize, n_used: usize) -> usize {
        let groups = 4 + n_tokens * n_used * 72 + n_tokens * n_used * (n_tokens + 1) * 4;
        round_256(mmq_q8_1_bytes(n_tokens, k) + groups)
    }

    /// Compact rows of `routed_compact` for `slots` assignments
    /// (`RoutedCompactRows`: every bucket padded to 16 rows).
    pub const fn routed_compact_rows(slots: usize, n_experts: usize) -> usize {
        slots + n_experts * 15
    }

    /// IDs in each buffer of `mtp_top_candidates`
    /// (`MtpCandidateWorkspaceSize`).
    pub const fn mtp_candidate_workspace(vocab: usize) -> usize {
        let tiles = vocab.div_ceil(1024);
        (if tiles < 2 { 2 } else { tiles }) * MTP_CANDIDATES
    }

    /// Token rows per routed F16 tile (`RoutedTileRows`): the wide tile once
    /// the mean bucket fills most of it.
    pub const fn routed_tile_rows(slots: usize, n_experts: usize) -> u32 {
        if slots >= 16 * n_experts { 48 } else { 16 }
    }

    /// Smallest prefill chunk whose gate/up projections run as one paired
    /// launch (`kRoutedPairMinTokens`).
    pub const ROUTED_PAIR_MIN_TOKENS: usize = 1024;
    /// Rows per paired wide and narrow tile (`kPairRows`, `kNarrowRows`).
    pub const ROUTED_PAIR_ROWS: u32 = 128;
    pub const ROUTED_NARROW_ROWS: u32 = 64;
    /// Smallest prefill chunk that moves an MoE model's BF16 projections
    /// (and its F32 router) to the BF16 WMMA GEMM
    /// (`kBf16WmmaMinPrefillBatch`).
    pub const BF16_WMMA_MIN_PREFILL_BATCH: usize = 128;

    /// The GGUF type IDs the routed F16 GEMMs decode
    /// (`moe_f16::WeightType`).
    pub const fn routed_f16_type(ggml_type: u32) -> bool {
        matches!(ggml_type, 7 | 8 | 12 | 13 | 14)
    }

    /// Whether `routed_gated_f16_gemm` decodes a GGUF type
    /// (`RoutedGatedF16Supported`: Q4_K, Q5_K, Q6_K, Q8_0).
    pub const fn routed_gated_f16_type(ggml_type: u32) -> bool {
        matches!(ggml_type, 8 | 12 | 13 | 14)
    }

    /// Whether `routed_f16_gemm` takes 64- and 128-row tiles for a GGUF type
    /// (`RoutedF16WideTilesSupported`: Q5_1, Q6_K, Q8_0).
    pub const fn routed_f16_wide_tiles(ggml_type: u32) -> bool {
        matches!(ggml_type, 7 | 8 | 14)
    }

    /// Whether the grouped BF16 GEMMs take an `[m, k]` matrix of `n_experts`
    /// experts (`IsMoeGroupedBf16GemmSupported`).
    pub const fn grouped_bf16_gemm(m: usize, k: usize, n_experts: usize) -> bool {
        m != 0
            && k != 0
            && m.is_multiple_of(64)
            && k.is_multiple_of(64)
            && n_experts != 0
            && n_experts <= MAX_EXPERTS
    }

    /// Whether the fused one-token router takes these router and shared
    /// gate formats (GGUF IDs) and shape (`IsMoeRouterGateTopKSupported`).
    pub const fn router_gate_top_k(
        router_type: u32,
        shared_type: u32,
        n_experts: usize,
        k: usize,
        n_used: usize,
    ) -> bool {
        dense_router_type(router_type)
            && dense_router_type(shared_type)
            && n_experts <= MAX_FUSED_ROUTER_EXPERTS
            && k < 16384
            && n_used <= 32
    }

    /// F32 or BF16 (GGUF IDs 0 and 30).
    const fn dense_router_type(ggml_type: u32) -> bool {
        ggml_type == 0 || ggml_type == 30
    }
}

/// The GEMM helpers of the extension that call hipBLAS and hipBLASLt
/// (`hipblas_gemm.hip`, `hipblaslt_gemm.hip`): no device code of their own,
/// so the host runs them through gufo-hip's `Blas` and `BlasLt` with these
/// shapes and this plan policy, which keep the C++ choices.
pub mod blas {
    use gufo_hip::blas::{AlgoPolicy, Heuristic, Layout, MatmulShape, Op};

    /// `HipblasLtGemm::RunBf16`'s shape: `y[batch][m] = x[batch][k] a[m][k]^T`
    /// with BF16 `a` and `x` and F32 `y` (only TRANSA set).
    pub fn bf16_matmul_shape(batch: usize, m: usize, k: usize) -> MatmulShape {
        MatmulShape {
            trans_a: Some(Op::T),
            trans_b: None,
            a: Layout::new(k, m, k),
            b: Layout::new(k, batch, k),
            c: Layout::new(m, batch, m),
            batch: None,
        }
    }

    /// The C++ `HipblasLtGemm` plan choice for one shape (default options:
    /// no plan database, no workspace): 16 heuristic results, the rank
    /// `SelectHipblasLtHeuristicRank` prefers when it is usable, else the
    /// first usable one; solution names are resolved after installing.
    pub struct HipblasLtPlans {
        pub batch: usize,
        pub m: usize,
        pub k: usize,
    }

    const WIDE_FFN_THRESHOLD: usize = 3;
    const LARGE_BATCH: usize = 256;
    const LARGE_WIDE_RANK: usize = 4;
    const TALL_RANK: usize = 5;

    /// `SelectHipblasLtHeuristicRank`.
    pub fn heuristic_rank(batch: usize, m: usize, k: usize, result_count: usize) -> usize {
        if batch >= LARGE_BATCH {
            if m <= 1024 {
                return LARGE_WIDE_RANK;
            }
            return if k >= WIDE_FFN_THRESHOLD * m {
                TALL_RANK
            } else {
                LARGE_WIDE_RANK
            };
        }
        let tall_m_threshold = if batch >= 128 { k } else { 2 * k };
        if m > tall_m_threshold || k >= WIDE_FFN_THRESHOLD * m {
            return TALL_RANK;
        }
        result_count.saturating_sub(1)
    }

    impl AlgoPolicy for HipblasLtPlans {
        fn choose(&self, ranked: &[Heuristic], workspace_capacity: usize) -> Option<usize> {
            let preferred = heuristic_rank(self.batch, self.m, self.k, ranked.len());
            if ranked
                .get(preferred)
                .is_some_and(|h| h.is_usable(workspace_capacity))
            {
                return Some(preferred);
            }
            ranked.iter().position(|h| h.is_usable(workspace_capacity))
        }
    }

    /// `LaunchHipblasGEMM`/`LaunchHipblasGEMMBF16`: `hipblasGemmEx` with
    /// `(OP_T, OP_N)`, `m = M`, `n = batch`, `k = K`, leading dimensions
    /// `K`, `K`, `M`, F32 compute and `HIPBLAS_GEMM_DEFAULT` on a handle
    /// whose atomics mode the runner never sets (`Blas::with_default_atomics`).
    pub const GEMM_EX_OPS: (Op, Op) = (Op::T, Op::N);

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn rank_matches_the_cpp_selection() {
            // Large batches: wide (m <= 1024) and FFN-shaped tall.
            assert_eq!(heuristic_rank(512, 1024, 2048, 16), 4);
            assert_eq!(heuristic_rank(512, 2048, 8192, 16), 5);
            assert_eq!(heuristic_rank(512, 4096, 2048, 16), 4);
            // Smaller batches: tall m or wide k, else the last result.
            assert_eq!(heuristic_rank(64, 8192, 2048, 16), 5);
            assert_eq!(heuristic_rank(128, 2048, 2048, 16), 15);
            assert_eq!(heuristic_rank(64, 512, 2048, 7), 5);
            assert_eq!(heuristic_rank(64, 2048, 2048, 7), 6);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sizes::*;

    #[test]
    fn sizes_match_the_cpp_helpers() {
        assert_eq!(decode_attention_scratch(16, 256), 16 * 32 * 258);
        assert_eq!(kv_q8_layer_bytes(4, 2), 4 * 2 * 272);
        assert_eq!(q8_activation_bytes(17, 2048), 2 * 64 * 640);
        assert_eq!(mmq_q8_1_bytes(1, 640), 32 * 36);
        assert_eq!(mmq_moe_vec_scratch(1, 512), 768);
        assert_eq!(routed_compact_rows(2048, 256), 2048 + 3840);
        assert_eq!(mtp_candidate_workspace(248_320), 243 * 64);
        assert_eq!(mtp_candidate_workspace(50), 128);
        assert_eq!(routed_tile_rows(4096, 256), 48);
        assert!(routed_gated_f16_type(14) && !routed_gated_f16_type(7));
        assert!(routed_f16_wide_tiles(8) && !routed_f16_wide_tiles(12));
        assert!(grouped_bf16_gemm(512, 2048, 256) && !grouped_bf16_gemm(500, 2048, 256));
        assert!(router_gate_top_k(0, 0, 256, 2048, 8) && !router_gate_top_k(8, 0, 256, 2048, 8));
    }
}
