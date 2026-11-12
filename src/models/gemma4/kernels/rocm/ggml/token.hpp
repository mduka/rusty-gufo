// Excerpt of Gufo src/families/qwen/kernels/ops/token.hpp at upstream
// 1c75887b (gufo-org/gufo): the paths Gemma 4 dispatches for the formats its
// loader accepts, in the gemma4 extension namespace.
#ifndef GUFO_MODELS_GEMMA4_HIP_GGML_TOKEN_HPP_
#define GUFO_MODELS_GEMMA4_HIP_GGML_TOKEN_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include "src/core/gguf_reader.hpp"
#include <span>
#include <stdexcept>

namespace gufo::models::gemma4::ggml {

inline constexpr std::uint32_t kInvalidSampleToken = UINT32_MAX;

inline std::uint32_t CheckedSampleToken(std::uint32_t token,
                                        std::size_t vocab_size) {
  if (token >= vocab_size)
    throw std::runtime_error("logit distribution contains no finite values");
  return token;
}

/// Computes one argmax per row of a [batch_size, vocab_size] logits matrix.
void LaunchBatchedGPUArgmax(const float* logits, std::uint32_t* out_tokens,
                            std::size_t batch_size, std::size_t vocab_size,
                            std::span<float> scratch,
                            hipStream_t stream = nullptr);

/// Batched Embedding lookup for B tokens
void LaunchBatchedEmbeddingLookup(const void* table, core::GgmlType type,
                                  const std::uint32_t* token_ids,
                                  float* out_hidden, std::size_t batch_size,
                                  std::size_t hidden_size,
                                  hipStream_t stream = nullptr);

}  // namespace gufo::models::gemma4::ggml

#endif  // GUFO_MODELS_GEMMA4_HIP_GGML_TOKEN_HPP_
