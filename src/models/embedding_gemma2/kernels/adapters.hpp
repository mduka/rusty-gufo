// Flat forms of the EmbeddingGemma 2 launchers that take a `QkvLayout` by
// reference, which the Rust launcher contract cannot describe directly
// (docs/RUST.md, "launch.toml"). Each adapter only forwards; it
// belongs to the trusted base like the launchers.

#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_KERNELS_ADAPTERS_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_KERNELS_ADAPTERS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::embedding_gemma2::hip {

/// LaunchQkvNormRope with the layout as three scalars.
void LaunchQkvNormRopeFlat(const float* qkv, const void* query_norm_bf16,
                           const void* key_norm_bf16, void* query_bf16,
                           void* key_bf16, void* value_bf16,
                           const std::uint32_t* positions, std::size_t tokens,
                           std::uint32_t num_heads,
                           std::uint32_t num_key_value_heads,
                           std::uint32_t head_dim, const float* rope_cos,
                           const float* rope_sin, float epsilon,
                           hipStream_t stream);

/// LaunchAttention with the layout as three scalars.
[[nodiscard]] bool LaunchAttentionFlat(
    const void* query_bf16, const void* key_bf16, const void* value_bf16,
    void* context_bf16, const std::uint32_t* sequence_begin,
    const std::uint32_t* sequence_end, std::size_t tokens,
    std::uint32_t num_heads, std::uint32_t num_key_value_heads,
    std::uint32_t head_dim, std::uint32_t window, hipStream_t stream);

}  // namespace gufo::models::embedding_gemma2::hip

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_KERNELS_ADAPTERS_HPP_
