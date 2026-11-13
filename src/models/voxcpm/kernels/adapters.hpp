// Flat forms of the VoxCPM2 launchers whose signatures the Rust launcher
// contract cannot describe directly (docs/development/DESIGN.md,
// "launch.toml"): the matrix-core products take their plan and split scratch
// as structs, and the skinny product refuses one shape whose sweep would read
// past its operands. Each adapter forwards after checking what its launcher
// assumes; it belongs to the trusted base like the launchers themselves. The
// kernels are the VoxCPM extension's (../hip, gufo-extensions/voxcpm 0941cd1).

#ifndef GUFO_MODELS_VOXCPM_KERNELS_ADAPTERS_HPP_
#define GUFO_MODELS_VOXCPM_KERNELS_ADAPTERS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::voxcpm::kernels {

/// SkinnyGemm, except for up to 8 rows of a reduction of more than 6144
/// (8192, a multiple of 2048 within its limit): there its register sweep
/// runs in passes of 6144 columns and the second pass would read past the
/// rows of `x` and `w`. Those shapes return false (nothing launched), so
/// the caller takes its hipBLASLt path; VoxCPM2's products reduce over at
/// most 6144.
bool SkinnyGemmChecked(const std::uint16_t* x, const std::uint16_t* w, float* y,
                       std::size_t rows, std::size_t in, std::size_t out,
                       bool accumulate, hipStream_t stream);

/// WmmaQ8 with the plan's split (`splits` waves of `blocks_per_split`
/// 32-column blocks) and the split scratch passed flat; absent scratch
/// (null) counts as empty, which a split product refuses.
bool WmmaQ8Flat(const std::uint16_t* x, const std::uint8_t* w, float* y,
                std::size_t rows, std::size_t in, std::size_t out,
                bool accumulate, std::size_t splits,
                std::size_t blocks_per_split, float* partials,
                std::size_t partial_floats, std::uint32_t* counters,
                std::size_t counter_count, hipStream_t stream);

/// WmmaBf16 in the flat form of WmmaQ8Flat.
bool WmmaBf16Flat(const std::uint16_t* x, const std::uint16_t* w, float* y,
                  std::size_t rows, std::size_t in, std::size_t out,
                  bool accumulate, std::size_t splits,
                  std::size_t blocks_per_split, float* partials,
                  std::size_t partial_floats, std::uint32_t* counters,
                  std::size_t counter_count, hipStream_t stream);

}  // namespace gufo::models::voxcpm::kernels

#endif  // GUFO_MODELS_VOXCPM_KERNELS_ADAPTERS_HPP_
