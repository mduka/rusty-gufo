// Flat launchers of the Gemma 4 vision kernels (adapters.hpp), with the grids
// and block sizes of the extension's vision/encoder.hip Encoder::Encode and
// Impl::UploadHalf. The kernels sit in an anonymous namespace, so they are
// included here, once.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/gemma4/kernels/rocm/adapters.hpp"
#include "src/models/gemma4/kernels/rocm/vision_kernels.hip.hpp"

namespace gufo::models::gemma4::rocm::flat {
namespace v = ::gufo::models::gemma4::vision;

bool VisionBf16ToHalf(void* data, std::size_t count, hipStream_t stream) {
  if (count == 0) {
    return false;
  }
  v::Bf16ToHalf<<<(count + v::kThreads - 1) / v::kThreads, v::kThreads, 0,
                  stream>>>(static_cast<std::uint16_t*>(data), count);
  return true;
}

bool VisionPatchify(const std::uint8_t* pixels, float* patches,
                    std::uint32_t width, std::uint32_t columns,
                    std::uint32_t count, hipStream_t stream) {
  v::Patchify<<<v::Blocks(std::size_t{count} * v::kPatchValues), v::kThreads, 0,
                stream>>>(pixels, patches, width, columns, count);
  return true;
}

bool VisionAddPositions(float* hidden, const float* table,
                        std::uint32_t columns, std::uint32_t count,
                        hipStream_t stream) {
  v::AddPositions<<<v::Blocks(std::size_t{count} * v::kHidden), v::kThreads, 0,
                    stream>>>(hidden, table, columns, count);
  return true;
}

bool VisionRmsNormHalf(const float* x, const float* weight, void* y,
                       std::uint32_t rows, std::uint32_t width,
                       hipStream_t stream) {
  v::RmsNormHalf<<<rows, v::kThreads, 0, stream>>>(
      x, weight, static_cast<v::Act*>(y), width);
  return true;
}

bool VisionQkvPost(const float* qkv, const float* q_norm, const float* k_norm,
                   const float* cos, const float* sin, void* q, void* k,
                   void* vt, std::uint32_t columns, std::uint32_t padded,
                   std::uint32_t count, hipStream_t stream) {
  v::QkvPost<<<count * v::kHeads, 96, 0, stream>>>(
      qkv, q_norm, k_norm, cos, sin, static_cast<_Float16*>(q),
      static_cast<_Float16*>(k), static_cast<_Float16*>(vt), columns, padded);
  return true;
}

bool VisionAttention(const void* q, const void* k, const void* vt, void* out,
                     std::uint32_t count, std::uint32_t padded,
                     hipStream_t stream) {
  const dim3 grid((padded / 16 + v::kAttentionWaves - 1) / v::kAttentionWaves,
                  v::kHeads);
  v::FusedAttention<<<grid, 32 * v::kAttentionWaves, 0, stream>>>(
      static_cast<const _Float16*>(q), static_cast<const _Float16*>(k),
      static_cast<const _Float16*>(vt), static_cast<v::Act*>(out), count,
      padded);
  return true;
}

bool VisionPostNormAdd(float* hidden, const float* o, const float* weight,
                       std::uint32_t rows, hipStream_t stream) {
  v::PostNormAdd<<<rows, v::kThreads, 0, stream>>>(hidden, o, weight);
  return true;
}

bool VisionGeGlu(const float* gate_up, void* out, std::uint32_t count,
                 hipStream_t stream) {
  v::GeGlu<<<v::Blocks(std::size_t{count} * v::kFfnPadded), v::kThreads, 0,
             stream>>>(gate_up, static_cast<v::Act*>(out), count);
  return true;
}

bool VisionPool(const float* hidden, float* pooled, std::uint32_t columns,
                std::uint32_t pooled_columns, std::uint32_t rows,
                hipStream_t stream) {
  v::Pool<<<v::Blocks(std::size_t{rows} * v::kHidden), v::kThreads, 0,
            stream>>>(hidden, pooled, columns, pooled_columns, rows);
  return true;
}

bool VisionStandardizeNorm(const float* pooled, const float* bias,
                           const float* scale, void* out, std::uint32_t rows,
                           hipStream_t stream) {
  v::StandardizeNorm<<<rows, v::kThreads, 0, stream>>>(
      pooled, bias, scale, static_cast<v::Act*>(out));
  return true;
}

}  // namespace gufo::models::gemma4::rocm::flat
