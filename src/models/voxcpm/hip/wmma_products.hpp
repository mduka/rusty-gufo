#ifndef GUFO_MODELS_VOXCPM_KERNELS_WMMA_PRODUCTS_HPP_
#define GUFO_MODELS_VOXCPM_KERNELS_WMMA_PRODUCTS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// Products of up to 32 activation rows on the matrix cores
// (v_wmma_f32_16x16x16_bf16) with weights in tiles: one wave per 16 weight
// rows and a range of the reduction, every weight read once, in contiguous
// pieces. They serve every product of GGML Q8_0 weights (decoding, the
// local encoder, the DiT) and the DiT's 22 rows (its CFG pair of 11) with
// bfloat16 weights.
//
// A tile holds 16 weight rows and 32 columns (one Q8_0 block of each row);
// the tiles of 16 rows follow each other along the columns, then the next
// 16 rows (rows beyond the matrix are zero):
//
// - Q8_0: 512 value bytes [16][32] (int8 q) and the 16 rows' float16
//   scales (kQ8TileBytes). The values stay exact; the products sum q * x in
//   float32 and scale per block.
// - bfloat16: [16][32] values (kBf16TileBytes).
namespace gufo::models::voxcpm::kernels {

/// Raw bfloat16 bits.
using Bf16 = std::uint16_t;

inline constexpr std::size_t kTileRows = 16;
inline constexpr std::size_t kTileColumns = 32;
inline constexpr std::size_t kQ8TileBytes = 16 * 32 + 16 * 2;
inline constexpr std::size_t kBf16TileBytes = 16 * 32 * 2;

[[nodiscard]] constexpr std::size_t Q8TilesBytes(std::size_t rows,
                                                 std::size_t in) {
  return (rows + kTileRows - 1) / kTileRows * (in / kTileColumns) *
         kQ8TileBytes;
}
[[nodiscard]] constexpr std::size_t Bf16TilesBytes(std::size_t rows,
                                                   std::size_t in) {
  return (rows + kTileRows - 1) / kTileRows * (in / kTileColumns) *
         kBf16TileBytes;
}

/// Repacks GGML Q8_0 blocks ([rows][in / 32] blocks of a float16 scale and
/// 32 int8 values, 34 bytes) into tiles at `output`, which must be zero
/// beyond the last row's tile rows.
void Q8_0ToTiles(const std::uint8_t* blocks, std::size_t rows, std::size_t in,
                 std::uint8_t* output, hipStream_t stream);

/// Repacks bfloat16 [rows][in] into tiles (zero beyond `rows` as above).
void Bf16ToTiles(const Bf16* weights, std::size_t rows, std::size_t in,
                 Bf16* output, hipStream_t stream);

/// Dequantizes Q8_0 tiles to bfloat16 [rows][in]: bf16(scale * q), the
/// rounding of a conversion at load (products of more than 128 rows).
void Q8TilesToBf16(const std::uint8_t* tiles, std::size_t rows, std::size_t in,
                   Bf16* output, hipStream_t stream);

/// Unpacks bfloat16 tiles to [rows][in].
void Bf16TilesToRows(const Bf16* tiles, std::size_t rows, std::size_t in,
                     Bf16* output, hipStream_t stream);

/// Device scratch of the split reduction: partial sums and one counter
/// per 16 weight rows (zero between products; the products leave them so).
struct SplitScratch {
  float* partials = nullptr;
  std::size_t partial_floats = 0;
  unsigned* counters = nullptr;
  std::size_t counter_count = 0;
};

/// How a product is split over waves.
struct WmmaPlan {
  std::size_t splits = 1;
  std::size_t blocks_per_split = 0;  // 32-column blocks of the reduction
  /// Scratch the product needs.
  std::size_t partial_floats = 0;
  std::size_t counters = 0;
};

/// The split of a product of `rows` (<= 32) rows over at least about
/// `target_waves` waves (0: the measured default for the row count).
[[nodiscard]] WmmaPlan PlanWmma(std::size_t rows, std::size_t in,
                                std::size_t out, std::size_t target_waves = 0);

/// y[rows][out] (+)= x[rows][in] W^T for bf16 x and Q8_0 tiles W, at most
/// 32 rows and `in` a multiple of 32. False (nothing launched) when the
/// shape doesn't suit it or the scratch is too small.
bool WmmaQ8(const Bf16* x, const std::uint8_t* w, float* y, std::size_t rows,
            std::size_t in, std::size_t out, bool accumulate,
            const WmmaPlan& plan, const SplitScratch& scratch,
            hipStream_t stream);

/// The same for bfloat16 tiles W.
bool WmmaBf16(const Bf16* x, const Bf16* w, float* y, std::size_t rows,
              std::size_t in, std::size_t out, bool accumulate,
              const WmmaPlan& plan, const SplitScratch& scratch,
              hipStream_t stream);

}  // namespace gufo::models::voxcpm::kernels

#endif  // GUFO_MODELS_VOXCPM_KERNELS_WMMA_PRODUCTS_HPP_
