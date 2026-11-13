#ifndef GUFO_MODELS_VOXCPM_KERNELS_SKINNY_GEMM_HPP_
#define GUFO_MODELS_VOXCPM_KERNELS_SKINNY_GEMM_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::voxcpm::kernels {

/// Raw bfloat16 bits.
using Bf16 = std::uint16_t;

/// Rows up to which SkinnyGemm applies.
inline constexpr std::size_t kSkinnyRows = 32;

/// y[rows][out] (+)= x[rows][in] W^T for bf16 x and W ([out][in]) with
/// float32 sums, for the few rows of decoding with bf16 rows (the DiT's
/// bf16 matrices are tiles, wmma_products.hpp): each weight is read once.
/// Returns false (nothing launched) for more than kSkinnyRows rows or `in` not
/// a multiple of 8.
bool SkinnyGemm(const Bf16* x, const Bf16* w, float* y, std::size_t rows,
                std::size_t in, std::size_t out, bool accumulate,
                hipStream_t stream);

/// SkinnyGemm reading `vec` (1, 2 or 4; 4 for at most 16 rows) 16-byte
/// pieces of a weight row per lane and step (benchmarks).
bool SkinnyGemmWith(int vec, const Bf16* x, const Bf16* w, float* y,
                    std::size_t rows, std::size_t in, std::size_t out,
                    bool accumulate, hipStream_t stream);

/// Reads `pieces` 16-byte values with `blocks` blocks (benchmarks).
void ReadProbe(const Bf16* data, std::size_t pieces, unsigned blocks,
               unsigned* sink, hipStream_t stream);

}  // namespace gufo::models::voxcpm::kernels

#endif  // GUFO_MODELS_VOXCPM_KERNELS_SKINNY_GEMM_HPP_
