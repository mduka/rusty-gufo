// The constants the vision kernels take from the extension's
// vision/weights.hpp and vision/prompt.hpp (verbatim; the weight views and
// the prompt code are host code, ported to Rust).
#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_VISION_CONSTANTS_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_VISION_CONSTANTS_HPP_

#include <cstdint>

namespace gufo::models::gemma4::vision {

inline constexpr std::uint32_t kHidden = 1152;
inline constexpr std::uint32_t kHeads = 16;
inline constexpr std::uint32_t kHeadDim = 72;
inline constexpr std::uint32_t kFfn = 4304;
inline constexpr std::uint32_t kLayers = 27;
/// Rows of each learned position table (x and y).
inline constexpr std::uint32_t kPositions = 10240;
inline constexpr float kEps = 1e-6F;
inline constexpr float kRopeTheta = 100.0F;
/// Pixel values per patch, ordered channel, row, column.
inline constexpr std::uint32_t kPatchValues = 3 * 16 * 16;

inline constexpr std::uint32_t kPatchSize = 16;
/// Average-pooling kernel over the patch grid; one soft token per 3x3.
inline constexpr std::uint32_t kPoolSize = 3;
inline constexpr std::uint32_t kSideMultiple = kPatchSize * kPoolSize;

}  // namespace gufo::models::gemma4::vision

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_VISION_CONSTANTS_HPP_
