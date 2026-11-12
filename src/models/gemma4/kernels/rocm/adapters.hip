// Flat Gemma 4 launchers (adapters.hpp): forward to the extension's
// launchers.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "src/core/sampling.hpp"
#include "src/models/gemma4/kernels/rocm/adapters.hpp"
#include "src/models/gemma4/kernels/rocm/candidates.hpp"
#include "src/models/gemma4/kernels/rocm/gemv.hpp"
#include "src/models/gemma4/kernels/rocm/ggml/token.hpp"
#include "src/models/gemma4/kernels/rocm/half_gemm.hpp"
#include "src/models/gemma4/kernels/rocm/kernels.hpp"
#include "src/models/gemma4/kernels/rocm/moe.hpp"
#include "src/models/gemma4/kernels/rocm/wmma_gemv.hpp"

namespace gufo::models::gemma4::rocm::flat {
namespace {

using core::GgmlType;

bool SupportedHeadDim(std::uint32_t head_dim) {
  return head_dim == 256 || head_dim == 512;
}

/// The activation pack of a `width`-wide projection input in `scratch`.
WmmaPackOut PackIn(void* scratch, std::uint32_t width) {
  return scratch != nullptr ? WmmaPackOf(scratch, width) : WmmaPackOut{};
}

std::optional<ExpertFormat> ExpertFormatOf(GgmlType type) {
  switch (type) {
    case GgmlType::kQ4_K:
      return ExpertFormat::kQ4_K;
    case GgmlType::kQ5_K:
      return ExpertFormat::kQ5_K;
    case GgmlType::kQ6_K:
      return ExpertFormat::kQ6_K;
    case GgmlType::kQ8_0:
      return ExpertFormat::kQ8_0;
    case GgmlType::kQ5_1:
      return ExpertFormat::kQ5_1;
    case GgmlType::kF16:
      return ExpertFormat::kF16;
    case GgmlType::kQ4_0:
      return ExpertFormat::kQ4_0;
    case GgmlType::kQ5_0:
      return ExpertFormat::kQ5_0;
    default:
      return std::nullopt;
  }
}

std::optional<GemvFormat> GemvFormatOf(GgmlType type) {
  switch (type) {
    case GgmlType::kQ4_0:
      return GemvFormat::kQ4_0;
    case GgmlType::kQ4_K:
      return GemvFormat::kQ4_K;
    case GgmlType::kQ5_K:
      return GemvFormat::kQ5_K;
    case GgmlType::kQ6_K:
      return GemvFormat::kQ6_K;
    case GgmlType::kQ8_0:
      return GemvFormat::kQ8_0;
    default:
      return std::nullopt;
  }
}

std::optional<PrefillLayout> LayoutOf(std::uint32_t layout) {
  switch (layout) {
    case 0:
      return PrefillLayout::kNone;
    case 1:
      return PrefillLayout::kCompact;
    case 2:
      return PrefillLayout::kDecoded;
    default:
      return std::nullopt;
  }
}

}  // namespace

// --- kernels.hpp -------------------------------------------------------------

bool QkvPostInPlace(float* q, const float* k, const float* v,
                    const float* q_norm, const float* k_norm, float theta_scale,
                    const float* freq_factors, void* k_cache, void* v_cache,
                    std::uint32_t rows, std::uint32_t heads,
                    std::uint32_t kv_heads, std::uint32_t head_dim,
                    std::uint32_t first_position, std::uint32_t ring, float eps,
                    std::uint32_t rotated_pairs, std::uint32_t v_stride,
                    std::uint32_t siblings, std::uint32_t spare_key,
                    hipStream_t stream) {
  if (!SupportedHeadDim(head_dim)) {
    return false;
  }
  QkvPostArgs args{};
  args.q = q;
  args.k = k;
  args.v = v;
  args.q_norm = q_norm;
  args.k_norm = k_norm;
  args.theta_scale = theta_scale;
  args.freq_factors = freq_factors;
  args.k_cache = static_cast<std::uint16_t*>(k_cache);
  args.v_cache = static_cast<std::uint16_t*>(v_cache);
  args.rows = rows;
  args.heads = heads;
  args.kv_heads = kv_heads;
  args.head_dim = head_dim;
  args.first_position = first_position;
  args.ring = ring;
  args.eps = eps;
  args.rotated_pairs = rotated_pairs;
  args.row_stride = 0;
  args.v_stride = v_stride;
  args.q_out = nullptr;
  args.siblings = siblings;
  args.spare_key = spare_key;
  rocm::QkvPost(args, stream);
  return true;
}

bool QkvPostFused(const float* q, const float* k, const float* v, float* q_out,
                  const float* q_norm, const float* k_norm, float theta_scale,
                  const float* freq_factors, void* k_cache, void* v_cache,
                  std::uint32_t rows, std::uint32_t heads,
                  std::uint32_t kv_heads, std::uint32_t head_dim,
                  std::uint32_t first_position, std::uint32_t ring, float eps,
                  std::uint32_t rotated_pairs, std::uint32_t row_stride,
                  std::uint32_t v_stride, std::uint32_t siblings,
                  std::uint32_t spare_key, hipStream_t stream) {
  if (!SupportedHeadDim(head_dim)) {
    return false;
  }
  QkvPostArgs args{};
  // The kernel reads Q from here and writes it to q_out only.
  args.q = const_cast<float*>(q);
  args.k = k;
  args.v = v;
  args.q_norm = q_norm;
  args.k_norm = k_norm;
  args.theta_scale = theta_scale;
  args.freq_factors = freq_factors;
  args.k_cache = static_cast<std::uint16_t*>(k_cache);
  args.v_cache = static_cast<std::uint16_t*>(v_cache);
  args.rows = rows;
  args.heads = heads;
  args.kv_heads = kv_heads;
  args.head_dim = head_dim;
  args.first_position = first_position;
  args.ring = ring;
  args.eps = eps;
  args.rotated_pairs = rotated_pairs;
  args.row_stride = row_stride;
  args.v_stride = v_stride;
  args.q_out = q_out;
  args.siblings = siblings;
  args.spare_key = spare_key;
  rocm::QkvPost(args, stream);
  return true;
}

bool QueryPost(float* q, const float* q_norm, float theta_scale,
               const float* freq_factors, std::uint32_t rows,
               std::uint32_t heads, std::uint32_t head_dim,
               std::uint32_t position, bool shared_position, float eps,
               const float* key_weight, std::uint32_t rotated_pairs,
               hipStream_t stream) {
  if (!SupportedHeadDim(head_dim)) {
    return false;
  }
  rocm::QueryPost(q, q_norm, theta_scale, freq_factors, rows, heads, head_dim,
                  position, shared_position, eps, key_weight, rotated_pairs,
                  stream);
  return true;
}

bool DraftConcat(const float* embed, const float* const* hidden, float* out,
                 float scale, std::uint32_t rows, std::uint32_t dim,
                 hipStream_t stream) {
  DraftConcatArgs args{};
  if (rows > sizeof(args.hidden) / sizeof(args.hidden[0])) {
    return false;
  }
  args.embed = embed;
  for (std::uint32_t r = 0; r < rows; ++r) {
    args.hidden[r] = hidden[r];
  }
  args.out = out;
  args.scale = scale;
  args.rows = rows;
  args.dim = dim;
  rocm::DraftConcat(args, stream);
  return true;
}

bool Attention(const float* q, const void* k_cache, const void* v_cache,
               float* out, float* partials, std::uint32_t rows,
               std::uint32_t heads, std::uint32_t kv_heads,
               std::uint32_t head_dim, std::uint32_t first_position,
               bool shared_position, std::uint32_t key_limit,
               std::uint32_t window, std::uint32_t ring,
               std::uint32_t rope_pairs, const std::uint32_t* key_ends,
               void* out_half, std::uint32_t siblings, std::uint32_t spare_key,
               void* pack_scratch, std::uint32_t pack_row0,
               hipStream_t stream) {
  if (!SupportedHeadDim(head_dim)) {
    return false;
  }
  AttentionArgs args{};
  args.q = q;
  args.k_cache = static_cast<const std::uint16_t*>(k_cache);
  args.v_cache = static_cast<const std::uint16_t*>(v_cache);
  args.out = out;
  args.partials = partials;
  args.rows = rows;
  args.heads = heads;
  args.kv_heads = kv_heads;
  args.head_dim = head_dim;
  args.first_position = first_position;
  args.shared_position = shared_position;
  args.key_limit = key_limit;
  args.window = window;
  args.ring = ring;
  args.rope_pairs = rope_pairs;
  args.key_ends = key_ends;
  args.out_half = out_half;
  args.siblings = siblings;
  args.spare_key = spare_key;
  args.pack = PackIn(pack_scratch, heads * head_dim);
  args.pack_row0 = pack_row0;
  rocm::Attention(args, stream);
  return true;
}

bool PostAttentionNorm(const float* o, const float* post_norm, float* x,
                       const float* next_norm, float* h, std::uint32_t rows,
                       std::uint32_t dim, float eps, void* q8,
                       const float* second_norm, float* h2, void* h_half,
                       void* h2_half, void* pack_scratch, hipStream_t stream) {
  rocm::PostAttentionNorm(o, post_norm, x, next_norm, h, rows, dim, eps, stream,
                          q8, second_norm, h2, h_half, h2_half,
                          PackIn(pack_scratch, dim));
  return true;
}

bool PostFeedForwardNorm(const float* f, const float* post_norm, float scale,
                         float* x, const float* next_norm, float* h,
                         std::uint32_t rows, std::uint32_t dim, float eps,
                         void* q8, void* h_half, void* pack_scratch,
                         hipStream_t stream) {
  rocm::PostFeedForwardNorm(f, post_norm, scale, x, next_norm, h, rows, dim,
                            eps, stream, q8, h_half, PackIn(pack_scratch, dim));
  return true;
}

bool GeGluPacked(const float* gu, float* out, std::uint32_t slots,
                 std::uint32_t width, void* out_half, void* pack_scratch,
                 hipStream_t stream) {
  rocm::GeGluPacked(gu, out, slots, width, stream, out_half,
                    PackIn(pack_scratch, width));
  return true;
}

bool MoveKey(void* const* k, void* const* v, const std::uint32_t* k_width,
             const std::uint32_t* v_width, const std::uint32_t* ring,
             std::uint32_t layers, std::uint32_t from, std::uint32_t to,
             hipStream_t stream) {
  if (layers == 0 || layers > MoveKeyArgs::kMaxLayers) {
    return false;
  }
  MoveKeyArgs args{};
  for (std::uint32_t l = 0; l < layers; ++l) {
    args.k[l] = static_cast<std::uint16_t*>(k[l]);
    args.v[l] = static_cast<std::uint16_t*>(v[l]);
    args.k_width[l] = k_width[l];
    args.v_width[l] = v_width[l];
    args.ring[l] = ring[l];
  }
  args.layers = layers;
  args.from = from;
  args.to = to;
  rocm::MoveKey(args, stream);
  return true;
}

// --- candidates.hpp ----------------------------------------------------------

static_assert(sizeof(sampling::LogitCandidate) == 2 * sizeof(std::uint32_t) &&
              alignof(sampling::LogitCandidate) == alignof(std::uint32_t));

bool LogitCandidates(const float* logits, std::uint32_t rows,
                     std::uint32_t vocab, std::uint32_t top,
                     const std::uint32_t* extra, std::uint32_t n_extra,
                     void* work, std::uint32_t* counts, std::uint32_t* out,
                     hipStream_t stream) {
  if (rows == 0 || vocab == 0 || vocab % 4 != 0 || top == 0 ||
      top > kCandidateTop + kMaxCandidateExtra ||
      n_extra > kMaxCandidateExtra || (n_extra != 0 && extra == nullptr)) {
    return false;
  }
  // The kernels need a zeroed workspace (an unzeroed histogram could point
  // their writes outside it); zero it every launch.
  const CandidateWork zeroed = CandidateWorkAt(work, rows, stream);
  rocm::LogitCandidates(
      logits, rows, vocab, top, extra, n_extra, zeroed, counts,
      reinterpret_cast<sampling::LogitCandidate*>(out), stream);
  return true;
}

// --- gemv.hpp ----------------------------------------------------------------

bool KQuantGemv(const void* w, GgmlType type, const float* x, float* y,
                std::uint32_t m, std::uint32_t k, hipStream_t stream) {
  const auto format = GemvFormatOf(type);
  return format.has_value() && LaunchKQuantGemv(*format, w, x, y, m, k, stream);
}

// --- half_gemm.hpp -----------------------------------------------------------

bool RoutedF16Gemm(const void* w, GgmlType type, const void* x,
                   const std::uint32_t* counts, const std::uint32_t* tiles,
                   std::uint32_t n_tiles, std::uint32_t tile_rows,
                   const std::int32_t* pad_bounds, const std::int32_t* rows_in,
                   const std::int32_t* rows_out, float* out, void* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream) {
  // `counts` only ties `tiles` to its source in the contract; tile maps
  // derived from counts hold non-negative int32 values.
  (void)counts;
  // WeightType holds the GGML type IDs of the formats it takes.
  if (type != GgmlType::kQ4_K && type != GgmlType::kQ5_K &&
      type != GgmlType::kQ5_1 && type != GgmlType::kQ8_0) {
    return false;
  }
  return rocm::RoutedF16Gemm(
      w, static_cast<WeightType>(static_cast<std::uint32_t>(type)),
      static_cast<const __half*>(x),
      reinterpret_cast<const std::int32_t*>(tiles), n_tiles, tile_rows,
      pad_bounds, rows_in, rows_out, out, static_cast<__half*>(out_half), m, k,
      stream);
}

bool UnquantizedF16Gemm(const void* w, const void* x, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        std::size_t wide_from, hipStream_t stream) {
  return rocm::UnquantizedF16Gemm(w, static_cast<const __half*>(x), out, batch,
                                  m, k, stream, wide_from);
}

bool DenseF16Gemm(const void* w, const void* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, std::uint32_t plan,
                  bool grouped, hipStream_t stream) {
  if (plan > static_cast<std::uint32_t>(DenseF16Plan::kStagedRowGroups8Walk7)) {
    return false;
  }
  return rocm::DenseF16Gemm(w, static_cast<const __half*>(x), out, batch, m, k,
                            stream, static_cast<DenseF16Plan>(plan), grouped);
}

bool DenseF16GeGluGemm(const void* w, const void* x, void* out,
                       std::size_t batch, std::size_t m, std::size_t k,
                       bool grouped, bool half_weights, hipStream_t stream) {
  return rocm::DenseF16GeGluGemm(w, static_cast<const __half*>(x),
                                 static_cast<__half*>(out), batch, m, k, stream,
                                 grouped, half_weights);
}

// --- moe.hpp -----------------------------------------------------------------

bool MoeRoute(const float* x, const float* router, const float* expert_scale,
              float* logits, bool raw_logits, std::int32_t* ids, float* weights,
              std::uint32_t rows, std::uint32_t hidden, std::uint32_t experts,
              std::uint32_t used, float eps, hipStream_t stream) {
  MoeRouteArgs args{};
  args.x = x;
  args.router = router;
  args.expert_scale = expert_scale;
  args.logits = logits;
  args.raw_logits = raw_logits;
  args.ids = ids;
  args.weights = weights;
  args.counts = nullptr;
  args.groups = nullptr;
  args.rows = rows;
  args.hidden = hidden;
  args.experts = experts;
  args.used = used;
  args.eps = eps;
  args.sync = nullptr;
  rocm::MoeRoute(args, stream);
  return true;
}

bool RoutedGemv(const void* w, GgmlType type, const std::int32_t* ids,
                const std::int32_t* groups, std::uint32_t max_groups,
                const float* x, std::uint32_t x_div, float* y, std::uint32_t m,
                std::uint32_t k, bool geglu, bool single, hipStream_t stream) {
  // `ids` only ties `groups` to its source in the contract.
  (void)ids;
  const auto format = ExpertFormatOf(type);
  return format.has_value() &&
         LaunchRoutedGemv(*format, w, groups, max_groups, x, x_div, y, m, k,
                          stream, geglu, single);
}

bool DenseGemv(const void* w, GgmlType type, const float* x, float* y,
               std::uint32_t rows, std::uint32_t m, std::uint32_t k,
               hipStream_t stream) {
  const auto format = ExpertFormatOf(type);
  return format.has_value() &&
         LaunchDenseGemv(*format, w, x, y, rows, m, k, stream);
}

bool PackPrefillWeights(const void* w, GgmlType type, std::uint32_t layout,
                        void* packed, std::uint32_t m, std::uint32_t k,
                        hipStream_t stream) {
  const auto format = ExpertFormatOf(type);
  const auto prefill = LayoutOf(layout);
  return format.has_value() && prefill.has_value() &&
         rocm::PackPrefillWeights(*format, *prefill, w, packed, m, k, stream);
}

bool PackedPrefillGemm(std::uint32_t format, std::uint32_t layout,
                       const void* w, const void* x, float* out, void* out_half,
                       std::uint32_t rows, std::uint32_t m, std::uint32_t k,
                       bool geglu, std::uint32_t group, hipStream_t stream) {
  const auto prefill = LayoutOf(layout);
  if (format > static_cast<std::uint32_t>(ExpertFormat::kQ5_0) ||
      !prefill.has_value()) {
    return false;
  }
  return LaunchPackedPrefillGemm(static_cast<ExpertFormat>(format), *prefill, w,
                                 x, out, out_half, rows, m, k, stream, geglu,
                                 group);
}

bool RoutedHalfGemm(const void* w, GgmlType type, std::uint32_t layout,
                    const void* x, const std::uint32_t* counts,
                    const std::uint32_t* tiles, std::uint32_t n_tiles,
                    const std::int32_t* pad_bounds, const std::int32_t* rows_in,
                    const std::int32_t* rows_out, float* out, void* out_half,
                    std::uint32_t m, std::uint32_t k, bool geglu,
                    hipStream_t stream) {
  (void)counts;  // as in RoutedF16Gemm
  const auto format = ExpertFormatOf(type);
  const auto prefill = LayoutOf(layout);
  return format.has_value() && prefill.has_value() &&
         LaunchRoutedHalfGemm(
             *format, w, x, reinterpret_cast<const std::int32_t*>(tiles),
             n_tiles, kRoutedHalfTileRows, pad_bounds, rows_in, rows_out, out,
             out_half, m, k, stream, geglu, *prefill);
}

bool DenseHalfGemm(const void* w, GgmlType type, const void* x, float* out,
                   void* out_half, std::uint32_t rows, std::uint32_t m,
                   std::uint32_t k, bool geglu, bool grouped,
                   hipStream_t stream) {
  const auto format = ExpertFormatOf(type);
  return format.has_value() &&
         LaunchDenseHalfGemm(*format, w, x, out, out_half, rows, m, k, stream,
                             geglu, grouped);
}

bool BuildRoutedTiles(const std::int32_t* ids, const std::uint32_t* counts,
                      std::uint32_t experts, std::uint32_t rows_a,
                      std::uint32_t capacity_a, std::uint32_t* tiles_a,
                      std::uint32_t rows_b, std::uint32_t capacity_b,
                      std::uint32_t* tiles_b, hipStream_t stream) {
  // `ids` only ties `counts` to its source in the contract.
  (void)ids;
  if (experts > 256 || rows_a == 0 || rows_b == 0) {
    return false;
  }
  // The tile maps are int32: expert | tile << 16, tile below 0x8000.
  rocm::BuildRoutedTiles(counts, experts, rows_a, capacity_a,
                         reinterpret_cast<std::int32_t*>(tiles_a), rows_b,
                         capacity_b, reinterpret_cast<std::int32_t*>(tiles_b),
                         stream);
  return true;
}

bool MoeFinish(const float* dense, const float* experts, const float* weights,
               const float* norm1, const float* norm2, const float* post_norm,
               float scale, float* x, const float* next_norm, float* h,
               std::uint32_t rows, std::uint32_t hidden, std::uint32_t used,
               float eps, void* h_half, void* pack_scratch,
               hipStream_t stream) {
  MoeFinishArgs args{};
  args.dense = dense;
  args.experts = experts;
  args.weights = weights;
  args.norm1 = norm1;
  args.norm2 = norm2;
  args.post_norm = post_norm;
  args.scale = scale;
  args.x = x;
  args.next_norm = next_norm;
  args.h = h;
  args.rows = rows;
  args.hidden = hidden;
  args.used = used;
  args.eps = eps;
  args.h_half = h_half;
  args.pack = PackIn(pack_scratch, hidden);
  rocm::MoeFinish(args, stream);
  return true;
}

// --- wmma_gemv.hpp -----------------------------------------------------------

bool WmmaGemv(const void* w, GgmlType type, const float* x, float* y,
              std::uint32_t rows, std::uint32_t m, std::uint32_t k,
              void* scratch, bool packed, bool grouped, hipStream_t stream) {
  return LaunchWmmaGemv(type, w, x, y, rows, m, k, scratch, stream, packed,
                        grouped);
}

bool GroupWeights(const void* src, GgmlType type, void* dst, std::uint32_t m,
                  std::uint32_t k, hipStream_t stream) {
  return rocm::GroupWeights(type, src, dst, m, k, stream);
}

// --- ggml/token.hpp ----------------------------------------------------------

bool BatchedArgmax(const float* logits, std::uint32_t* out_tokens,
                   std::size_t batch, std::size_t vocab, float* scratch,
                   std::size_t scratch_len, hipStream_t stream) {
  ggml::LaunchBatchedGPUArgmax(logits, out_tokens, batch, vocab,
                               std::span<float>(scratch, scratch_len), stream);
  return true;
}

}  // namespace gufo::models::gemma4::rocm::flat
