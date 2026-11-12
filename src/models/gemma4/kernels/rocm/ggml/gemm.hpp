// Excerpt of Gufo src/families/qwen/kernels/ops/gemm.hpp at upstream
// 1c75887b (gufo-org/gufo): the paths Gemma 4 dispatches for the formats its
// loader accepts, in the gemma4 extension namespace.
// Gemma 4 additions from the fork's feat/gemma (83b764b4) are applied.
#ifndef GUFO_MODELS_GEMMA4_HIP_GGML_GEMM_HPP_
#define GUFO_MODELS_GEMMA4_HIP_GGML_GEMM_HPP_

#include <cstddef>
#include <cstdint>
#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace gufo::models::gemma4::ggml {

/// y = A * x for one token: F32 and BF16 rows, and Q8_0, Q4_0, Q4_K, Q5_K and
/// Q6_K blocks decoded in the kernel (LaunchQ8KBlockGEMV). Other formats abort.
void LaunchGEMV(const void* A, core::GgmlType a_type, const float* x, float* y,
                std::size_t M, std::size_t K, hipStream_t stream = nullptr);
/// The block GEMV LaunchGEMV takes for quantized rows: one row per warp for
/// the K-quants, two for Q8_0 and Q4_0. Requires K to be a whole number of
/// blocks.
void LaunchQ8KBlockGEMV(const void* A, core::GgmlType type, const float* x,
                        float* y, std::size_t M, std::size_t K,
                        hipStream_t stream = nullptr);
namespace detail {
/// opt-q4kxl: the formats decoded in-kernel from packed form by the K-quant
/// kernels (the blocked WMMA GEMM at prefill batch and the exact
/// shared-weight kernel at draft width). Q8_0 keeps its own kernels.
[[nodiscard]] constexpr bool IsNativeWmmaQuant(core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
    case core::GgmlType::kQ4_0:
      return true;
    default:
      return false;
  }
}
}  // namespace detail
/// Directly computes Y[B, M] = X_fp32[B, K] * W_quant[M, K]^T using the same
/// quant-block dot product and FP32 activation precision as decode. Intended
/// for short verification batches where exact target-path numerics matter.
/// Groups of up to sixteen rows share weight loads while preserving the
/// isolated production GEMV arithmetic order.
void LaunchBatchedQuantGEMMFp32(core::GgmlType type, const void* w,
                                const float* fp32_x, float* y,
                                std::size_t batch, std::size_t m, std::size_t k,
                                hipStream_t stream = nullptr);
/// Q8_0 at 9..16 rows in one weight pass with the exact small-batch
/// arithmetic (every row equals LaunchGEMV bit for bit), at any shape;
/// LaunchBatchedQuantGEMMFp32 reads Q8_0 weights in eight-row passes. Returns
/// false without launching for other widths.
[[nodiscard]] bool LaunchQ8_0SmallBatchWide(const void* w, const float* fp32_x,
                                            float* y, std::size_t batch,
                                            std::size_t m, std::size_t k,
                                            hipStream_t stream = nullptr);
/// The K-quant (Q4_K/Q5_K/Q6_K) and Q4_0 small-batch kernel with 16 waves,
/// three rows per wave and two activation tiles per stage for 1..8 rows (one
/// past eight, up to 16). Like every exact small-batch configuration each row
/// equals LaunchGEMV bit for bit; this one suits long K. Returns false without
/// launching for other types, widths or index ranges.
[[nodiscard]] bool LaunchKQuantSmallBatchDoubleStage(
    core::GgmlType type, const void* w, const float* fp32_x, float* y,
    std::size_t batch, std::size_t m, std::size_t k,
    hipStream_t stream = nullptr);
/// Quantizes FP32 activation tensor X[B, K] to tiled block_q8_1 activation
/// blocks.
void LaunchQuantizeActivationQ8_1FromFp32(const float* fp32_x, void* q8_1_out,
                                          std::size_t batch, std::size_t k,
                                          hipStream_t stream = nullptr);
/// Byte size of the tiled Q8_1 activation buffer a [batch, k] activation needs
/// before LaunchBatchedQuantGEMMPreQuantized can consume it.
[[nodiscard]] std::size_t QuantizedActivationBytes(std::size_t batch,
                                                   std::size_t k);
/// Directly computes Y[B, M] = X_q8_1[B, K] * W_quant[M, K]^T using LDS-staged
/// Matrix Core (WMMA) operations with pre-quantized activation blocks.
void LaunchBatchedQuantGEMMPreQuantized(core::GgmlType type, const void* w,
                                        const void* q8_1_x, float* y,
                                        std::size_t batch, std::size_t m,
                                        std::size_t k,
                                        hipStream_t stream = nullptr);

}  // namespace gufo::models::gemma4::ggml

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_GEMMA4_HIP_GGML_GEMM_HPP_
