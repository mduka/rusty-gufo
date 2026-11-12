// Excerpt of the Gufo fork's feat/gemma (83b764b4)
// src/models/qwen38_flash_next/kernels/rocm/kernels.hpp: the binary16 dense
// and routed GEMMs and the MTP candidate selection, in the gemma4 extension
// namespace.
#ifndef GUFO_MODELS_GEMMA4_HIP_HALF_GEMM_HPP_
#define GUFO_MODELS_GEMMA4_HIP_HALF_GEMM_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::gemma4::rocm {

/// GGUF type ids of the routed expert formats RoutedF16Gemm decodes.
enum class WeightType : std::uint32_t {
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ4_K = 12,
  kQ5_K = 13,
};

/// Converts `count` floats to binary16 rows for the F16 GEMMs.
void NarrowActivations(const float* x, void* out, std::size_t count,
                       hipStream_t stream);

/// F16 activation rows [batch][k] against binary16 weights [m][k], F32
/// accumulation in two K16 chains. out is [batch][m]. Batches from
/// `wide_from` rows take 128 x 128 tiles instead of 64 x 64, with identical
/// results. Unsupported shapes launch nothing and return false.
bool UnquantizedF16Gemm(const void* w, const __half* x, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream, std::size_t wide_from = 1024);
/// Wide-batch tile plans a caller may choose for DenseF16Gemm; kAuto keeps
/// the default dispatch. The others apply from 1024 rows: eight row groups
/// (each weight fragment reused across the token tiles), optionally with the
/// two-block stage for long K, and optionally walking seven row tiles of a
/// token tile in launch order (row tiles divisible by seven; otherwise the
/// staged plan).
enum class DenseF16Plan : std::uint8_t {
  kAuto,
  kRowGroups8,
  kStagedRowGroups8,
  kStagedRowGroups8Walk7,
};
/// F16 activation rows [batch][k], Q8_0 weights dequantized to F16 in LDS,
/// F32 accumulation. out is [batch][m]. Every plan gives the same results.
/// `grouped`: Q8_0 weights in Gemma 4's grouped layout (wmma_gemv.hpp;
/// m > 512, k a multiple of 256), same results as the row order.
bool DenseF16Gemm(const void* w, const __half* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream,
                  DenseF16Plan plan = DenseF16Plan::kAuto,
                  bool grouped = false);
/// Gemma 4's fused Q8_0 [gate | up] (m rows: gate rows, then their up rows;
/// m a multiple of 64) with its GeGLU epilogue: `out` [batch][m / 2]
/// binary16 rows of gelu_tanh(gate) * up, the values Gemma's separate GeGLU
/// pass computes from DenseF16Gemm's FP32 rows. Needs 96 or more rows;
/// `grouped` as for DenseF16Gemm; `half_weights`: binary16 weights (as
/// UnquantizedF16Gemm reads them), row order only.
bool DenseF16GeGluGemm(const void* w, const __half* x, __half* out,
                       std::size_t batch, std::size_t m, std::size_t k,
                       hipStream_t stream, bool grouped = false,
                       bool half_weights = false);

/// Routed expert GEMMs. RoutedCompact sorts the (token, slot) assignments
/// by expert into `rows_token`/`rows_slot` (RoutedCompactRows(slots,
/// experts) entries, -1 for padding) with every bucket padded to 16 rows
/// (`pad_bounds`, experts + 1 entries) from the per-expert `counts`.
std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts);
void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream);

/// Routed expert GEMM on the F16 WMMA cores over F16 activation rows
/// `x` ([rows][k]): the weights are dequantized to F16 after the LDS read
/// and the whole K extent accumulates in F32. `tiles` (n_tiles entries)
/// packs each launched (expert, token tile) as expert | tile << 16, so empty
/// tiles cost nothing. Exactly one of `out` (F32) and `out_half` (F16)
/// receives the result.
/// `tile_rows` is the token rows per tile the map was built with: 16 or 48,
/// or 64 for Q5_1/Q8_0.
/// Q4_K and Q5_K need k % 256 == 0, Q5_1 and Q8_0 k % 64 == 0; other types
/// and tile widths return false.
bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   float* out, __half* out_half, std::size_t m, std::size_t k,
                   hipStream_t stream);

/// Number of IDs in each of the two selection buffers.
std::uint32_t MtpCandidateWorkspaceSize(std::uint32_t vocab);

/// Exact MTP top-64 selection, descending score with lowest-token-ID ties.
/// Nonfinite scores become -infinity. Both ID buffers have room for
/// MtpCandidateWorkspaceSize(vocab); scores receives min(vocab, 64) values.
/// Scores may reuse the ID workspace starting at offset 64.
/// All candidate operations are graph-safe.
void MtpTopCandidates(const float* logits, std::uint32_t* ids,
                      std::uint32_t* scratch_ids, float* scores,
                      std::uint32_t vocab, hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_HIP_HALF_GEMM_HPP_
