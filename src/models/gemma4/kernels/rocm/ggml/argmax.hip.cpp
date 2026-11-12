// Excerpt of Gufo src/families/qwen/kernels/sample.hip at upstream
// 1c75887b (gufo-org/gufo): the paths Gemma 4 dispatches for the formats its
// loader accepts, in the gemma4 extension namespace.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/gemma4/kernels/rocm/ggml/token.hpp"

namespace gufo::models::gemma4::ggml {
namespace {

constexpr std::size_t kArgmaxTile = 4096;

template<bool Partial = false, bool MapIndices = false>
__global__ void BatchedArgmaxKernel(const float* __restrict__ logits,
                                    std::uint32_t* __restrict__ out_tokens,
                                    std::size_t vocab_size,
                                    float* partial_values = nullptr,
                                    std::uint32_t* partial_tokens = nullptr) {
  const std::size_t row = Partial ? blockIdx.y : blockIdx.x;
  const std::size_t tid = threadIdx.x;
  const float* row_logits = logits + (row * vocab_size);
  const std::size_t begin = Partial ? blockIdx.x * kArgmaxTile : 0;
  const std::size_t end = Partial && begin + kArgmaxTile < vocab_size
                              ? begin + kArgmaxTile
                              : vocab_size;

  float max_val = -INFINITY;
  std::uint32_t max_idx = kInvalidSampleToken;
  for (std::size_t index = begin + tid; index < end; index += blockDim.x) {
    const float value = row_logits[index];
    if constexpr (MapIndices) {
      const auto candidate = partial_tokens[row * vocab_size + index];
      if (isfinite(value) &&
          (value > max_val || (value == max_val && candidate < max_idx))) {
        max_val = value;
        max_idx = candidate;
      }
    } else {
      if (isfinite(value) && value > max_val) {
        max_val = value;
        max_idx = static_cast<std::uint32_t>(index);
      }
    }
  }

  __shared__ float shared_max[256];
  __shared__ std::uint32_t shared_index[256];
  shared_max[tid] = max_val;
  shared_index[tid] = max_idx;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride && (shared_max[tid + stride] > shared_max[tid] ||
                         (shared_max[tid + stride] == shared_max[tid] &&
                          shared_index[tid + stride] < shared_index[tid]))) {
      shared_max[tid] = shared_max[tid + stride];
      shared_index[tid] = shared_index[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if constexpr (Partial) {
      const auto offset = row * gridDim.x + blockIdx.x;
      partial_values[offset] = shared_max[0];
      partial_tokens[offset] = shared_index[0];
    } else {
      out_tokens[row] = shared_index[0];
    }
  }
}

}  // namespace

void LaunchBatchedGPUArgmax(const float* logits, std::uint32_t* out_tokens,
                            std::size_t batch_size, std::size_t vocab_size,
                            std::span<float> scratch, hipStream_t stream) {
  if (batch_size == 0 || vocab_size == 0) {
    return;
  }
  const auto partials = (vocab_size - 1) / kArgmaxTile + 1;
  if (partials > 1 && batch_size <= scratch.size() / (2 * partials)) {
    // FFN output scratch is free after the head projection. Parallel vocabulary
    // tiles keep large rows from serializing their reads in one workgroup.
    auto* const indices = reinterpret_cast<std::uint32_t*>(
        scratch.data() + batch_size * partials);
    BatchedArgmaxKernel<true><<<dim3(static_cast<unsigned>(partials),
                                     static_cast<unsigned>(batch_size)),
                                256, 0, stream>>>(
        logits, out_tokens, vocab_size, scratch.data(), indices);
    BatchedArgmaxKernel<false, true>
        <<<static_cast<unsigned>(batch_size), 256, 0, stream>>>(
            scratch.data(), out_tokens, partials, nullptr, indices);
  } else {
    BatchedArgmaxKernel<>
        <<<static_cast<unsigned int>(batch_size), 256, 0, stream>>>(
            logits, out_tokens, vocab_size);
  }
}

}  // namespace gufo::models::gemma4::ggml
