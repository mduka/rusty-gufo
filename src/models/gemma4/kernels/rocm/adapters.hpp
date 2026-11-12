// Flat forms of the Gemma 4 launchers for the Rust launcher contract
// (docs/development/DESIGN.md, "launch.toml"): argument structs, the activation packs
// (WmmaPackOut) and the format enums become pointers and scalars, binary16
// buffers `void*`, and every launcher takes `hipStream_t` last. Each adapter
// only forwards; like the launchers it belongs to the trusted base.
//
// Result: true when the work was queued; false when the adapter or the
// launcher rejected the geometry (nothing queued). Launcher exceptions reach
// the generated shim, which reports a failed launch.
//
// A `pack_scratch` is the scratch of a WMMA projection (LaunchWmmaGemv);
// adapters that fill its activation pack carve it with WmmaPackOf at the
// width the launcher writes (its first 68 bytes per value).
//
// Exceptions to "only forwards": ExpertCounts (a histogram of expert IDs,
// the counts TopKKernel accumulates beside them) and ExpertGroups (a copy of
// moe.hip.cpp's GroupsKernel) run kernels of adapters_routing.hip, so the
// group table and the counts derive from expert IDs in launches of their own
// (see launch.toml); LogitCandidates zeroes its workspace on every launch.

#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_ADAPTERS_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_ADAPTERS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gemma4::rocm::flat {

// --- kernels.hpp -------------------------------------------------------------

bool QkvPostInPlace(float* q, const float* k, const float* v,
                    const float* q_norm, const float* k_norm, float theta_scale,
                    const float* freq_factors, void* k_cache, void* v_cache,
                    std::uint32_t rows, std::uint32_t heads,
                    std::uint32_t kv_heads, std::uint32_t head_dim,
                    std::uint32_t first_position, std::uint32_t ring, float eps,
                    std::uint32_t rotated_pairs, std::uint32_t v_stride,
                    std::uint32_t siblings, std::uint32_t spare_key,
                    hipStream_t stream);
bool QkvPostFused(const float* q, const float* k, const float* v, float* q_out,
                  const float* q_norm, const float* k_norm, float theta_scale,
                  const float* freq_factors, void* k_cache, void* v_cache,
                  std::uint32_t rows, std::uint32_t heads,
                  std::uint32_t kv_heads, std::uint32_t head_dim,
                  std::uint32_t first_position, std::uint32_t ring, float eps,
                  std::uint32_t rotated_pairs, std::uint32_t row_stride,
                  std::uint32_t v_stride, std::uint32_t siblings,
                  std::uint32_t spare_key, hipStream_t stream);
bool QueryPost(float* q, const float* q_norm, float theta_scale,
               const float* freq_factors, std::uint32_t rows,
               std::uint32_t heads, std::uint32_t head_dim,
               std::uint32_t position, bool shared_position, float eps,
               const float* key_weight, std::uint32_t rotated_pairs,
               hipStream_t stream);
bool DraftConcat(const float* embed, const float* const* hidden, float* out,
                 float scale, std::uint32_t rows, std::uint32_t dim,
                 hipStream_t stream);
bool Attention(const float* q, const void* k_cache, const void* v_cache,
               float* out, float* partials, std::uint32_t rows,
               std::uint32_t heads, std::uint32_t kv_heads,
               std::uint32_t head_dim, std::uint32_t first_position,
               bool shared_position, std::uint32_t key_limit,
               std::uint32_t window, std::uint32_t ring,
               std::uint32_t rope_pairs, const std::uint32_t* key_ends,
               void* out_half, std::uint32_t siblings, std::uint32_t spare_key,
               void* pack_scratch, std::uint32_t pack_row0, hipStream_t stream);
bool PostAttentionNorm(const float* o, const float* post_norm, float* x,
                       const float* next_norm, float* h, std::uint32_t rows,
                       std::uint32_t dim, float eps, void* q8,
                       const float* second_norm, float* h2, void* h_half,
                       void* h2_half, void* pack_scratch, hipStream_t stream);
bool PostFeedForwardNorm(const float* f, const float* post_norm, float scale,
                         float* x, const float* next_norm, float* h,
                         std::uint32_t rows, std::uint32_t dim, float eps,
                         void* q8, void* h_half, void* pack_scratch,
                         hipStream_t stream);
bool GeGluPacked(const float* gu, float* out, std::uint32_t slots,
                 std::uint32_t width, void* out_half, void* pack_scratch,
                 hipStream_t stream);
bool MoveKey(void* const* k, void* const* v, const std::uint32_t* k_width,
             const std::uint32_t* v_width, const std::uint32_t* ring,
             std::uint32_t layers, std::uint32_t from, std::uint32_t to,
             hipStream_t stream);

// --- candidates.hpp ----------------------------------------------------------

/// `out` holds LogitCandidate records (u32 token, F32 value) as u32 pairs.
bool LogitCandidates(const float* logits, std::uint32_t rows,
                     std::uint32_t vocab, std::uint32_t top,
                     const std::uint32_t* extra, std::uint32_t n_extra,
                     void* work, std::uint32_t* counts, std::uint32_t* out,
                     hipStream_t stream);

// --- gemv.hpp ----------------------------------------------------------------

bool KQuantGemv(const void* w, gufo::core::GgmlType type, const float* x,
                float* y, std::uint32_t m, std::uint32_t k, hipStream_t stream);

// --- half_gemm.hpp -----------------------------------------------------------

bool RoutedF16Gemm(const void* w, gufo::core::GgmlType type, const void* x,
                   const std::uint32_t* counts, const std::uint32_t* tiles,
                   std::uint32_t n_tiles, std::uint32_t tile_rows,
                   const std::int32_t* pad_bounds, const std::int32_t* rows_in,
                   const std::int32_t* rows_out, float* out, void* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream);
bool UnquantizedF16Gemm(const void* w, const void* x, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        std::size_t wide_from, hipStream_t stream);
bool DenseF16Gemm(const void* w, const void* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, std::uint32_t plan,
                  bool grouped, hipStream_t stream);
bool DenseF16GeGluGemm(const void* w, const void* x, void* out,
                       std::size_t batch, std::size_t m, std::size_t k,
                       bool grouped, bool half_weights, hipStream_t stream);

// --- moe.hpp -----------------------------------------------------------------

bool MoeRoute(const float* x, const float* router, const float* expert_scale,
              float* logits, bool raw_logits, std::int32_t* ids, float* weights,
              std::uint32_t rows, std::uint32_t hidden, std::uint32_t experts,
              std::uint32_t used, float eps, hipStream_t stream);
bool ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t slots, std::uint32_t experts,
                  hipStream_t stream);
bool ExpertGroups(const std::int32_t* ids, std::int32_t* groups,
                  std::uint32_t slots, std::uint32_t experts,
                  hipStream_t stream);
bool RoutedGemv(const void* w, gufo::core::GgmlType type,
                const std::int32_t* ids, const std::int32_t* groups,
                std::uint32_t max_groups, const float* x, std::uint32_t x_div,
                float* y, std::uint32_t m, std::uint32_t k, bool geglu,
                bool single, hipStream_t stream);
bool DenseGemv(const void* w, gufo::core::GgmlType type, const float* x,
               float* y, std::uint32_t rows, std::uint32_t m, std::uint32_t k,
               hipStream_t stream);
bool PackPrefillWeights(const void* w, gufo::core::GgmlType type,
                        std::uint32_t layout, void* packed, std::uint32_t m,
                        std::uint32_t k, hipStream_t stream);
/// `format` and `layout`: ExpertFormat and PrefillLayout values (the
/// weights are a packed copy, no longer in their GGML format).
bool PackedPrefillGemm(std::uint32_t format, std::uint32_t layout,
                       const void* w, const void* x, float* out, void* out_half,
                       std::uint32_t rows, std::uint32_t m, std::uint32_t k,
                       bool geglu, std::uint32_t group, hipStream_t stream);
bool RoutedHalfGemm(const void* w, gufo::core::GgmlType type,
                    std::uint32_t layout, const void* x,
                    const std::uint32_t* counts, const std::uint32_t* tiles,
                    std::uint32_t n_tiles, const std::int32_t* pad_bounds,
                    const std::int32_t* rows_in, const std::int32_t* rows_out,
                    float* out, void* out_half, std::uint32_t m,
                    std::uint32_t k, bool geglu, hipStream_t stream);
bool DenseHalfGemm(const void* w, gufo::core::GgmlType type, const void* x,
                   float* out, void* out_half, std::uint32_t rows,
                   std::uint32_t m, std::uint32_t k, bool geglu, bool grouped,
                   hipStream_t stream);
bool BuildRoutedTiles(const std::int32_t* ids, const std::uint32_t* counts,
                      std::uint32_t experts, std::uint32_t rows_a,
                      std::uint32_t capacity_a, std::uint32_t* tiles_a,
                      std::uint32_t rows_b, std::uint32_t capacity_b,
                      std::uint32_t* tiles_b, hipStream_t stream);
bool MoeFinish(const float* dense, const float* experts, const float* weights,
               const float* norm1, const float* norm2, const float* post_norm,
               float scale, float* x, const float* next_norm, float* h,
               std::uint32_t rows, std::uint32_t hidden, std::uint32_t used,
               float eps, void* h_half, void* pack_scratch, hipStream_t stream);

// --- wmma_gemv.hpp -----------------------------------------------------------

bool WmmaGemv(const void* w, gufo::core::GgmlType type, const float* x,
              float* y, std::uint32_t rows, std::uint32_t m, std::uint32_t k,
              void* scratch, bool packed, bool grouped, hipStream_t stream);
bool GroupWeights(const void* src, gufo::core::GgmlType type, void* dst,
                  std::uint32_t m, std::uint32_t k, hipStream_t stream);

// --- ggml/token.hpp ----------------------------------------------------------

bool BatchedArgmax(const float* logits, std::uint32_t* out_tokens,
                   std::size_t batch, std::size_t vocab, float* scratch,
                   std::size_t scratch_len, hipStream_t stream);

// --- vision (adapters_vision.hip) --------------------------------------------

bool VisionBf16ToHalf(void* data, std::size_t count, hipStream_t stream);
bool VisionPatchify(const std::uint8_t* pixels, float* patches,
                    std::uint32_t width, std::uint32_t columns,
                    std::uint32_t count, hipStream_t stream);
bool VisionAddPositions(float* hidden, const float* table,
                        std::uint32_t columns, std::uint32_t count,
                        hipStream_t stream);
bool VisionRmsNormHalf(const float* x, const float* weight, void* y,
                       std::uint32_t rows, std::uint32_t width,
                       hipStream_t stream);
bool VisionQkvPost(const float* qkv, const float* q_norm, const float* k_norm,
                   const float* cos, const float* sin, void* q, void* k,
                   void* vt, std::uint32_t columns, std::uint32_t padded,
                   std::uint32_t count, hipStream_t stream);
bool VisionAttention(const void* q, const void* k, const void* vt, void* out,
                     std::uint32_t count, std::uint32_t padded,
                     hipStream_t stream);
bool VisionPostNormAdd(float* hidden, const float* o, const float* weight,
                       std::uint32_t rows, hipStream_t stream);
bool VisionGeGlu(const float* gate_up, void* out, std::uint32_t count,
                 hipStream_t stream);
bool VisionPool(const float* hidden, float* pooled, std::uint32_t columns,
                std::uint32_t pooled_columns, std::uint32_t rows,
                hipStream_t stream);
bool VisionStandardizeNorm(const float* pooled, const float* bias,
                           const float* scale, void* out, std::uint32_t rows,
                           hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm::flat

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_ADAPTERS_HPP_
