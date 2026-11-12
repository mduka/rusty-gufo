#ifndef GUFO_MODELS_NEMOTRON_KERNELS_ROCM_GEMV_HPP_
#define GUFO_MODELS_NEMOTRON_KERNELS_ROCM_GEMV_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Model-private vector products for short chunks (decode and verification),
/// streaming each weight row once for every activation row of the chunk.
namespace gufo::models::nemotron::rocm {

/// Most activation rows the vector products take per weight matrix.
inline constexpr int kGemvMaxTokens = 8;

/// Words of the IQ2_XXS grid as BF16: 256 entries of four words, each word
/// two consecutive magnitudes (the lower one in the low half).
inline constexpr int kIq2xxsGridWords = 256 * 4;

/// Expands the GGUF IQ2_XXS grid (eight uint8 magnitudes per entry) into
/// BF16 pairs; every magnitude is exact in BF16.
void Iq2xxsGridBf16(const std::uint64_t* grid, std::uint32_t* words);

struct Iq2xxsRoute;

/// out = W x for IQ2_XXS weights. Activations are rounded to BF16, which
/// keeps 8 significant bits as Q8_1 does but without a shared block scale;
/// the signed grid values are exact in BF16 and products accumulate in F32.
///
/// With x_bf16, the activations arrive already rounded and the product
/// runs on kernels that read them through the caches instead of staging
/// them: a dense product takes exactly n_tokens rows, and a routed one with
/// `route` runs each expert once for all of its slots. Every output is
/// bit-identical to the one-token product of its activation row.
struct Iq2xxsGemvArgs {
  const void* weights{nullptr};        ///< [experts][rows][cols / 256 blocks]
  const std::uint32_t* grid{nullptr};  ///< kIq2xxsGridWords device words
  const float* x{nullptr};             ///< activation rows of `cols` floats
  /// [n_tokens][used] expert per slot; null for a dense product.
  const std::int32_t* ids{nullptr};
  int used{1};
  /// Routed products read one x row per slot instead of one per token.
  bool x_per_slot{false};
  /// [n_tokens][used][rows] when routed, else [n_tokens][rows].
  float* out{nullptr};
  int rows{0};  ///< Multiple of 32.
  int cols{0};  ///< Multiple of 256.
  int n_tokens{0};
  bool relu2{false};       ///< Stores max(out, 0)^2.
  bool accumulate{false};  ///< Adds into out instead of storing.
  /// BF16 activation rows in x's layout (rounded as the kernels round x).
  const void* x_bf16{nullptr};
  /// Routed products only, with x_bf16: the slots of ids grouped by expert
  /// (Iq2xxsRouteShort).
  const Iq2xxsRoute* route{nullptr};
  /// With x_bf16: stores BF16 rows here instead of F32 rows to out (no
  /// accumulate).
  void* out_bf16{nullptr};
};

hipError_t Iq2xxsGemv(const Iq2xxsGemvArgs& args, hipStream_t stream);

/// Most slots Iq2xxsRouteShort groups.
inline constexpr int kRouteShortSlots = 128;

/// Groups the at most kRouteShortSlots slots of a short chunk's routing (`ids`,
/// one expert per slot) by expert into `route`'s order, tiles and tile count,
/// for routed Iq2xxsGemv: one launch, no atomics, the same grouping on
/// every run. Each tile lists one expert's slots in slot order.
hipError_t Iq2xxsRouteShort(const std::int32_t* ids, int slots,
                            const Iq2xxsRoute& route, hipStream_t stream);

/// count floats (a multiple of 4) to BF16, rounded to nearest even.
hipError_t ToBf16(const float* x, void* out, std::size_t count,
                  hipStream_t stream);

}  // namespace gufo::models::nemotron::rocm

#endif  // GUFO_MODELS_NEMOTRON_KERNELS_ROCM_GEMV_HPP_
