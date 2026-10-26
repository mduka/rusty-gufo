#include "src/models/embedding_gemma2/kernels/adapters.hpp"

#include "src/models/embedding_gemma2/kernels/hip/ops.hpp"

namespace gufo::models::embedding_gemma2::hip {

void LaunchQkvNormRopeFlat(const float* qkv, const void* query_norm_bf16,
                           const void* key_norm_bf16, void* query_bf16,
                           void* key_bf16, void* value_bf16,
                           const std::uint32_t* positions, std::size_t tokens,
                           std::uint32_t num_heads,
                           std::uint32_t num_key_value_heads,
                           std::uint32_t head_dim, const float* rope_cos,
                           const float* rope_sin, float epsilon,
                           hipStream_t stream) {
  const QkvLayout layout{num_heads, num_key_value_heads, head_dim};
  LaunchQkvNormRope(qkv, query_norm_bf16, key_norm_bf16, query_bf16, key_bf16,
                    value_bf16, positions, tokens, layout, rope_cos, rope_sin,
                    epsilon, stream);
}

bool LaunchAttentionFlat(const void* query_bf16, const void* key_bf16,
                         const void* value_bf16, void* context_bf16,
                         const std::uint32_t* sequence_begin,
                         const std::uint32_t* sequence_end, std::size_t tokens,
                         std::uint32_t num_heads,
                         std::uint32_t num_key_value_heads,
                         std::uint32_t head_dim, std::uint32_t window,
                         hipStream_t stream) {
  const QkvLayout layout{num_heads, num_key_value_heads, head_dim};
  return LaunchAttention(query_bf16, key_bf16, value_bf16, context_bf16,
                         sequence_begin, sequence_end, tokens, layout, window,
                         stream);
}

}  // namespace gufo::models::embedding_gemma2::hip
