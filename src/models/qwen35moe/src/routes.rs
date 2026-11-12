//! Route decisions of the C++ prefill (`hip/prefill_chunk.cpp`) and the
//! support predicates it consults, as pure functions of shapes and formats
//! (the MoE kernels' own predicates are in `gufo_qwen35moe_kernels::sizes`).

use gufo_hip::{Bf16, DeviceSlice, GgmlType, LaunchError, QuantTensor};

/// Tokens per tile of the tiled Q8_1 activation layout.
const Q8_ACT_TILE_TOKENS: usize = 16;
/// Payload plus scales of one 16-token tile of 32 elements, and its
/// per-token activation sums (`kQ8ActTileBytes + kQ8ActSumTileBytes`).
const Q8_ACT_TILE_AND_SUM_BYTES: usize = 576 + 16 * 4;

/// Formats decoded in-kernel from packed form (`detail::IsNativeWmmaQuant`):
/// everything with a sub-16 decoder except Q8_0 (own kernels) and Q8_K.
pub fn is_native_wmma_quant(format: GgmlType) -> bool {
    matches!(
        format,
        GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Q3K
            | GgmlType::Iq4Nl
            | GgmlType::Iq4Xs
            | GgmlType::Iq3S
    )
}

/// Formats whose prefill projections read the tiled Q8_1 activation.
pub fn reads_q8_activation(format: GgmlType) -> bool {
    format == GgmlType::Q8_0 || is_native_wmma_quant(format)
}

/// Formats with the exact shared-weight batched GEMM over FP32 rows
/// (`SupportsExactSharedProjection`, Q8_K aside, which the Rust loader
/// rejects).
pub fn supports_exact_shared_projection(format: GgmlType) -> bool {
    reads_q8_activation(format)
}

/// The BF16 elements of a BF16 projection, for the exact BF16 kernels.
pub(crate) fn bf16_weights(w: &QuantTensor) -> Result<DeviceSlice<'_, Bf16>, LaunchError> {
    w.bytes()
        .view()
        .cast::<Bf16>()
        .ok_or(LaunchError::InvalidArgument {
            param: "weights",
            reason: "BF16 weights are not whole, aligned BF16 values",
        })
}

/// Bytes of the tiled Q8_1 activation of a `[batch, k]` tensor, sum sidecar
/// included (`QuantizedActivationBytes`).
pub fn quantized_activation_bytes(batch: usize, k: usize) -> usize {
    batch.div_ceil(Q8_ACT_TILE_TOKENS) * (k / 32) * Q8_ACT_TILE_AND_SUM_BYTES
}

/// The fused RMSNorm + Q8_1 kernel stages a row of whole 32-element blocks
/// in 32 KiB of LDS (`IsFusedRMSNormQuantizeQ8_1Supported`).
pub fn fused_norm_quantize_supported(dim: usize) -> bool {
    dim != 0 && dim.is_multiple_of(32) && dim * 4 <= 32_768
}

/// `IsFusedSSMEpilogueQuantizeQ8_1Supported`.
pub fn fused_ssm_epilogue_supported(val_dim: u32, inner_size: usize) -> bool {
    val_dim != 0 && val_dim <= 128 && val_dim.is_multiple_of(32) && inner_size.is_multiple_of(32)
}

/// `IsDeltaNetRowSplitSupported`: 128 x 128 state per head.
pub fn deltanet_row_split_supported(key_dim: u32, val_dim: u32) -> bool {
    key_dim == 128 && val_dim == 128
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_activation_size_rounds_tokens_to_whole_tiles() {
        // 16 tokens of 32 elements: one tile, 36 bytes per block row plus sums.
        assert_eq!(quantized_activation_bytes(16, 32), 640);
        assert_eq!(quantized_activation_bytes(1, 32), 640);
        assert_eq!(quantized_activation_bytes(17, 64), 2 * 2 * 640);
        assert_eq!(quantized_activation_bytes(20, 5120), 2 * 160 * 640);
    }

    #[test]
    fn predicates_accept_the_27b_shapes() {
        assert!(fused_norm_quantize_supported(5120));
        assert!(!fused_norm_quantize_supported(8200));
        assert!(!fused_norm_quantize_supported(16_384));
        assert!(fused_ssm_epilogue_supported(128, 6144));
        assert!(!fused_ssm_epilogue_supported(256, 6144));
        assert!(deltanet_row_split_supported(128, 128));
        assert!(reads_q8_activation(GgmlType::Iq3S));
        assert!(!reads_q8_activation(GgmlType::Q8K));
    }
}
