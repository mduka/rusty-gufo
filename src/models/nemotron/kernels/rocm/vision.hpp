#ifndef GUFO_MODELS_NEMOTRON_KERNELS_ROCM_VISION_HPP_
#define GUFO_MODELS_NEMOTRON_KERNELS_ROCM_VISION_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Kernels of the vision encoder: NVIDIA's C-RADIO ViT and the projector
/// into the trunk. Activations are row-major float32; operands of matrix
/// products are 16-bit values in the encoder's operand format, rounded to
/// nearest even. Every launch is asynchronous on `stream`.
namespace gufo::models::nemotron::rocm {

enum class VisionOperand : std::uint8_t { kBf16, kF16 };

/// Head width of the ViT's attention.
inline constexpr int kVisionHeadDim = 80;

/// out[i] = in[i] in the operand format.
hipError_t VisionToOperand(const float* in, std::uint16_t* out, std::size_t n,
                           VisionOperand op, hipStream_t stream);
/// BF16 values in the operand format: a copy for kBf16.
hipError_t VisionBf16ToOperand(const std::uint16_t* in, std::uint16_t* out,
                               std::size_t n, VisionOperand op,
                               hipStream_t stream);

/// The ViT's input rows. Rows 0..prefix_tokens-1 of x become the prefix
/// tokens; row prefix_tokens + p already holds patch p's projection (p
/// row-major over grid_rows x grid_cols) and gains its position: the
/// side x side table bilinearly resized (half-pixel centers) to the square
/// of the grid's longer side, cropped to the grid.
hipError_t VisionEmbed(float* x, const float* prefix, const float* positions,
                       int prefix_tokens, int grid_rows, int grid_cols,
                       int side, int width, hipStream_t stream);

/// LayerNorm of `rows` rows of `width` with affine w, b, into the operand
/// format.
hipError_t VisionLayerNorm(const float* x, const float* w, const float* b,
                           std::uint16_t* out, int rows, int width, float eps,
                           VisionOperand op, hipStream_t stream);

/// out = y + bias, in the operand format; `rows` rows of `cols`.
hipError_t VisionBias(const float* y, const float* bias, std::uint16_t* out,
                      int rows, int cols, VisionOperand op, hipStream_t stream);

/// out = gelu(y + bias) with the exact (erf) GELU, in the operand format.
hipError_t VisionBiasGelu(const float* y, const float* bias, std::uint16_t* out,
                          int rows, int cols, VisionOperand op,
                          hipStream_t stream);

/// x += y + bias; `rows` rows of `cols`.
hipError_t VisionResidual(float* x, const float* y, const float* bias, int rows,
                          int cols, hipStream_t stream);

/// Bidirectional multi-head attention of `tokens` rows of qkv (each row
/// q | k | v of heads x kVisionHeadDim) into out (heads x kVisionHeadDim per
/// row): softmax(q k^T * scale) v. Flash attention on the matrix units with
/// F32 scores, softmax and accumulation; the probabilities round to the
/// operand format.
hipError_t VisionAttention(const std::uint16_t* qkv, std::uint16_t* out,
                           int tokens, int heads, float scale, VisionOperand op,
                           hipStream_t stream);

/// The projector's input: patch features (rows prefix_tokens + p of x)
/// merged 2x2 into rows of 4 * width, channel (2 dy + dx) * width + c of the
/// patch at (2 i + dy, 2 j + dx), merged rows row-major over the halved
/// grid; RMS-normalized and scaled by w, in the operand format.
hipError_t VisionMergeNorm(const float* x, const float* w, std::uint16_t* out,
                           int prefix_tokens, int grid_rows, int grid_cols,
                           int width, float eps, VisionOperand op,
                           hipStream_t stream);

/// out = max(y, 0)^2 in the operand format.
hipError_t VisionRelu2(const float* y, std::uint16_t* out, std::size_t n,
                       VisionOperand op, hipStream_t stream);

}  // namespace gufo::models::nemotron::rocm

#endif  // GUFO_MODELS_NEMOTRON_KERNELS_ROCM_VISION_HPP_
