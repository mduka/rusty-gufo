// Flat forms of the Qwen-Image operators whose C++ signatures the Rust
// launcher contract cannot describe directly (docs/RUST.md, "launch.toml"):
// BF16 tensors as uint16 pointers, signed dimensions, the norm enum and the -1
// padding of the VAE down-sampler. Each adapter only forwards; it belongs to
// the trusted base like the launchers.

#ifndef GUFO_MODELS_QWEN_IMAGE_21_KERNELS_ADAPTERS_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_KERNELS_ADAPTERS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen_image_21::hip {

/// PackDense16: [rows][columns] to [columns / 16][rows][16].
void LaunchPackDenseFlat(const void* input, void* output, std::uint32_t rows,
                         std::uint32_t columns, hipStream_t stream);

/// DenseBf16 over packed weights and inputs; false for other shapes.
[[nodiscard]] bool LaunchDenseFlat(const void* weights, const void* input,
                                   void* output, std::uint32_t rows,
                                   std::uint32_t channels, std::uint32_t inner,
                                   hipStream_t stream);

/// DenseSwiGlu over packed weights and inputs; false for other shapes.
[[nodiscard]] bool LaunchDenseSwiGluFlat(const void* gate, const void* up,
                                         const void* input, void* output,
                                         std::uint32_t rows,
                                         std::uint32_t channels,
                                         std::uint32_t inner,
                                         hipStream_t stream);

/// Convolve3 with the total padding of both sides: 0 or 2 pad symmetrically,
/// 1 pads only the bottom and right (ZeroPad2d((0, 1, 0, 1))).
[[nodiscard]] bool LaunchConvolve3Flat(
    const void* input, const void* weights, const void* bias, void* output,
    std::uint32_t height, std::uint32_t width, std::uint32_t channels,
    std::uint32_t output_channels, std::uint32_t stride,
    std::uint32_t padding_total, std::uint32_t upscale, hipStream_t stream);

/// ExpandConvolution3 (the tiled 3x3 im2col); false for other geometries.
[[nodiscard]] bool LaunchExpandConvolution3Flat(
    const void* input, void* output, std::uint32_t height, std::uint32_t width,
    std::uint32_t channels, std::uint32_t stride, std::uint32_t padding,
    std::uint32_t upscale, std::uint32_t output_width, std::uint32_t start,
    std::uint32_t count, hipStream_t stream);

/// ReorderHeads from [row][kv_head][dim] to [head][row][dim], repeating each
/// KV head for its group of query heads.
void LaunchReorderHeadsFlat(const void* input, void* output, std::uint32_t rows,
                            std::uint32_t heads, std::uint32_t kv_heads,
                            std::uint32_t dim, hipStream_t stream);

/// The inverse ReorderHeads: [head][row][dim] back to [row][head][dim].
void LaunchRestoreHeadsFlat(const void* input, void* output, std::uint32_t rows,
                            std::uint32_t heads, std::uint32_t dim,
                            hipStream_t stream);

/// FusedAttention128 over head-major queries, keys and values.
void LaunchFusedAttention128Flat(const void* queries, const void* keys,
                                 const void* values, void* output,
                                 std::uint32_t heads, std::uint32_t rows,
                                 std::uint32_t count, std::uint32_t query_start,
                                 std::uint32_t query_count,
                                 const std::int32_t* limits,
                                 hipStream_t stream);

/// NormalizeRows with the norm kind as its integer value; false for an
/// unknown kind or a zero-centred RMSNorm without weight.
[[nodiscard]] bool LaunchNormalizeRowsFlat(const void* input, void* output,
                                           const void* weight, const void* bias,
                                           std::uint32_t rows,
                                           std::uint32_t columns,
                                           std::uint32_t kind, float epsilon,
                                           bool silu, hipStream_t stream);

/// NormalizeRotary128.
void LaunchNormalizeRotary128Flat(const void* input, void* output,
                                  const void* weight, const float* frequencies,
                                  std::uint32_t rows, std::uint32_t heads,
                                  float epsilon, hipStream_t stream);

}  // namespace gufo::models::qwen_image_21::hip

#endif
