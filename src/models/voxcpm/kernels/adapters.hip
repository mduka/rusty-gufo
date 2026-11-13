#include "src/models/voxcpm/kernels/adapters.hpp"

#include "src/models/voxcpm/hip/skinny_gemm.hpp"
#include "src/models/voxcpm/hip/wmma_products.hpp"

namespace gufo::models::voxcpm::kernels {

namespace {

/// The split scratch as WmmaQ8 and WmmaBf16 read it: an absent buffer holds
/// nothing, so a split product refuses it instead of writing through null.
SplitScratch Scratch(float* partials, std::size_t partial_floats,
                     std::uint32_t* counters, std::size_t counter_count) {
  return SplitScratch{partials, partials == nullptr ? 0 : partial_floats,
                      counters, counters == nullptr ? 0 : counter_count};
}

/// The plan WmmaQ8 and WmmaBf16 launch: they read the split only.
WmmaPlan Plan(std::size_t splits, std::size_t blocks_per_split) {
  WmmaPlan plan;
  plan.splits = splits;
  plan.blocks_per_split = blocks_per_split;
  return plan;
}

}  // namespace

bool SkinnyGemmChecked(const std::uint16_t* x, const std::uint16_t* w,
                       float* y, std::size_t rows, std::size_t in,
                       std::size_t out, bool accumulate, hipStream_t stream) {
  // SkinnyGemm sweeps up to 8 rows with in / 8 pieces per weight row; a
  // multiple of 2048 above 6144 takes more slices than one pass holds, and
  // its passes of 6144 columns overrun `in`.
  if (rows >= 1 && rows <= 8 && in > 6144 && in <= 8192 && in % 2048 == 0)
    return false;
  return SkinnyGemm(x, w, y, rows, in, out, accumulate, stream);
}

bool WmmaQ8Flat(const std::uint16_t* x, const std::uint8_t* w, float* y,
                std::size_t rows, std::size_t in, std::size_t out,
                bool accumulate, std::size_t splits,
                std::size_t blocks_per_split, float* partials,
                std::size_t partial_floats, std::uint32_t* counters,
                std::size_t counter_count, hipStream_t stream) {
  return WmmaQ8(x, w, y, rows, in, out, accumulate,
                Plan(splits, blocks_per_split),
                Scratch(partials, partial_floats, counters, counter_count),
                stream);
}

bool WmmaBf16Flat(const std::uint16_t* x, const std::uint16_t* w, float* y,
                  std::size_t rows, std::size_t in, std::size_t out,
                  bool accumulate, std::size_t splits,
                  std::size_t blocks_per_split, float* partials,
                  std::size_t partial_floats, std::uint32_t* counters,
                  std::size_t counter_count, hipStream_t stream) {
  return WmmaBf16(x, w, y, rows, in, out, accumulate,
                  Plan(splits, blocks_per_split),
                  Scratch(partials, partial_floats, counters, counter_count),
                  stream);
}

}  // namespace gufo::models::voxcpm::kernels
