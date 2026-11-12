#ifndef GUFO_MODELS_QWEN_VISION_KERNELS_HPP_
#define GUFO_MODELS_QWEN_VISION_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

/// Element-wise kernels of the Qwen3.8 ViT (FP32 math, BF16 rounding where
/// the reference rounds) and the language-side embedding injection. The
/// encoder host code runs them between hipBLAS(Lt) GEMMs.
namespace gufo::models::qwen::vision {

inline constexpr unsigned kHidden = 1152;
inline constexpr unsigned kHeads = 16;
inline constexpr unsigned kHeadDim = 72;
inline constexpr unsigned kFfn = 4304;
inline constexpr unsigned kMergeHidden = kHidden * 4;
inline constexpr unsigned kPatchDim = 3 * 2 * 16 * 16;

/// patches[count][kPatchDim] (BF16) from RGB8 pixels of a `width`-pixel
/// grid, patches ordered by 2x2 merge group, both temporal frames equal.
void LaunchPatchify(const std::uint8_t* pixels, void* patches, unsigned width,
                    unsigned grid_width, unsigned count, hipStream_t stream);
/// hidden += bias, then the bilinearly interpolated 48x48 position table.
void LaunchPatchPosition(float* hidden, const float* bias,
                         const float* positions, unsigned grid_height,
                         unsigned grid_width, unsigned count,
                         hipStream_t stream);
/// output[r] (BF16) = layernorm(input[r]) * weight + bias, eps 1e-6.
void LaunchLayerNorm(const float* input, void* output, const float* weight,
                     const float* bias, unsigned rows, unsigned width,
                     hipStream_t stream);
/// hidden = bf16(hidden + bf16(projection + bias)).
void LaunchBiasResidual(float* hidden, const float* projection,
                        const float* bias, unsigned size, unsigned width,
                        hipStream_t stream);
/// output (BF16) = gelu(bf16(input + bias)), tanh-approximate or exact.
void LaunchActivate(const float* input, const float* bias, void* output,
                    unsigned size, unsigned width, bool approximate,
                    hipStream_t stream);
/// Splits the fused QKV projection plus bias into head-major q, k and v
/// (BF16), with 2D rotary positions on q and k.
void LaunchQkvRope(const float* input, const float* bias, void* q, void* k,
                   void* v, unsigned count, unsigned grid_width,
                   hipStream_t stream);
/// probabilities[r] (BF16) = softmax(bf16(bf16(scores[r]) / sqrt(72))).
void LaunchSoftmax(const float* scores, void* probabilities, unsigned rows,
                   unsigned length, hipStream_t stream);
/// Rows begin..begin + rows of the token-major attention output (BF16) from
/// a head-major tile.
void LaunchAttentionRows(const float* input, void* output, unsigned begin,
                         unsigned rows, hipStream_t stream);
/// output = bf16(output + bias).
void LaunchFinish(float* output, const float* bias, unsigned size,
                  unsigned width, hipStream_t stream);
/// hidden[r][s] = embedding[r] for every one of `hc` residual streams.
void LaunchInjectRows(float* hidden, const float* embedding, std::uint32_t rows,
                      std::uint32_t width, std::uint32_t hc,
                      hipStream_t stream);

}  // namespace gufo::models::qwen::vision
#endif
