// Flat forms of the Nemotron-H launchers for the Rust launcher contract
// (docs/development/DESIGN.md, "launch.toml"): every count is a std::size_t
// (checked against the kernels' `int` range here), argument structs and enums
// become scalars, FP16/BF16 operands `void*`, and the kernels' hipError_t
// becomes the contract's result. Each adapter only forwards; like the launchers
// it belongs to the trusted base.
//
// Result: true when the launcher returned hipSuccess; false when it rejected
// its arguments (hipErrorInvalidValue: nothing queued) or a count does not
// fit an `int`; any other HIP error throws, which the generated shim reports
// as a failed launch.
//
// The per-sequence (`...Rows`) launchers take their per-row buffers as host
// lists of device pointers, copied into the caller's `device_tables` (stream
// ordered, eight bytes per pointer) before the launch reads them there. The
// copy is pageable, so these launches can't be recorded in a graph.

#ifndef GUFO_MODELS_NEMOTRON_KERNELS_ROCM_ADAPTERS_HPP_
#define GUFO_MODELS_NEMOTRON_KERNELS_ROCM_ADAPTERS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::nemotron::rocm::flat {

// --- kernels.hpp -------------------------------------------------------------

bool EmbedQ8_0(const void* table, const std::int32_t* tokens, float* x,
               std::size_t n_tokens, std::size_t hidden, hipStream_t stream);
bool RmsNorm(const float* x, const float* w, float* out, std::size_t rows,
             std::size_t width, float eps, hipStream_t stream);
bool RmsNormF16(const float* x, const float* w, void* out, std::size_t rows,
                std::size_t width, float eps, hipStream_t stream);
bool RmsNormQ8(const float* x, const float* w, float* out, void* q8,
               std::size_t rows, std::size_t width, float eps,
               hipStream_t stream);
bool QuantizeQ8_1(const float* x, void* q8, std::size_t rows, std::size_t width,
                  hipStream_t stream);
bool AddScaled(float* x, const float* a, float scale, const float* b,
               std::size_t n, hipStream_t stream);
bool Relu2(float* x, std::size_t n, hipStream_t stream);
bool MambaConv(const float* proj, std::size_t stride, std::size_t xbc_offset,
               float in_scale, const float* weight, const float* bias,
               const float* state, float* out, std::size_t n_tokens,
               std::size_t channels, std::size_t kernel, hipStream_t stream);
bool MambaConvState(const float* proj, std::size_t stride,
                    std::size_t xbc_offset, float in_scale, float* state,
                    std::size_t n_tokens, std::size_t channels,
                    std::size_t kernel, hipStream_t stream);
bool MambaScan(const float* xbc, std::size_t channels, const float* proj,
               std::size_t stride, std::size_t dt_offset, float in_scale,
               const float* dt_bias, const float* a, const float* d,
               float* state, float* y, std::size_t n_tokens, std::size_t heads,
               std::size_t head_dim, std::size_t groups, bool write_state,
               hipStream_t stream);
bool MambaGateNormQ8(const float* y, const float* proj, std::size_t stride,
                     float in_scale, const float* weight, void* q8,
                     std::size_t n_tokens, std::size_t inner,
                     std::size_t groups, float eps, hipStream_t stream);
bool MambaGateNorm(float* y, const float* proj, std::size_t stride,
                   float in_scale, const float* weight, std::size_t n_tokens,
                   std::size_t inner, std::size_t groups, float eps,
                   hipStream_t stream);
bool MambaGateNormF16(float* y, const float* proj, std::size_t stride,
                      float in_scale, const float* weight, void* out,
                      std::size_t n_tokens, std::size_t inner,
                      std::size_t groups, float eps, hipStream_t stream);
bool KvAppend(const float* k, const float* v, void* k_cache, void* v_cache,
              const std::int32_t* pos, std::size_t n_tokens, std::size_t kv_dim,
              hipStream_t stream);
bool Attention(const float* q, const void* k_cache, const void* v_cache,
               float* out, const std::int32_t* pos, std::size_t n_tokens,
               std::size_t heads, std::size_t kv_heads, float scale,
               hipStream_t stream);
bool AttentionShort(const float* q, const void* k_cache, const void* v_cache,
                    float* out, const std::int32_t* pos, std::size_t n_tokens,
                    std::size_t heads, std::size_t kv_heads, float scale,
                    float* partial, hipStream_t stream);
/// MambaConvRows: one token of each of `rows` sequences; `states` holds each
/// row's history (`device_tables`: rows pointers).
bool MambaConvRows(const float* proj, std::size_t stride,
                   std::size_t xbc_offset, float in_scale, const float* weight,
                   const float* bias, float* const* states, void* device_tables,
                   float* out, std::size_t rows, std::size_t channels,
                   std::size_t kernel, hipStream_t stream);
/// MambaScan with row_states: one token of each of `rows` sequences, row r
/// scanning with states[r] (`device_tables`: rows pointers); writes the
/// states.
bool MambaScanRows(const float* xbc, std::size_t channels, const float* proj,
                   std::size_t stride, std::size_t dt_offset, float in_scale,
                   const float* dt_bias, const float* a, const float* d,
                   float* const* states, void* device_tables, float* y,
                   std::size_t rows, std::size_t heads, std::size_t head_dim,
                   std::size_t groups, hipStream_t stream);
/// KvAppendRows: row r's K and V at row_positions[r] of k_caches[r] /
/// v_caches[r] (`device_tables`: 2 * rows pointers, K first).
bool KvAppendRows(const float* k, const float* v, void* const* k_caches,
                  void* const* v_caches, void* device_tables,
                  const std::int32_t* row_positions, std::size_t rows,
                  std::size_t kv_dim, hipStream_t stream);
/// AttentionShortRows: row r's query at row_positions[r] over k_caches[r] /
/// v_caches[r] (`device_tables`: 2 * rows pointers, K first).
bool AttentionShortRows(const float* q, const void* const* k_caches,
                        const void* const* v_caches, void* device_tables,
                        float* out, const std::int32_t* row_positions,
                        std::size_t rows, std::size_t heads,
                        std::size_t kv_heads, float scale, float* partial,
                        hipStream_t stream);
bool RouterLogits(const float* w, const float* x, float* out, std::size_t rows,
                  std::size_t cols, std::size_t n_tokens, hipStream_t stream);
bool RouterTopK(const float* logits, const float* bias, std::size_t n_experts,
                std::size_t n_tokens, std::size_t k, bool normalize,
                float scale, std::int32_t* ids, float* weights,
                hipStream_t stream);
bool WeightedSlotSum(const float* slots, const float* weights, float* out,
                     std::size_t n_tokens, std::size_t k, std::size_t width,
                     hipStream_t stream);
bool WeightedSlotSumQ8(const float* slots, const float* weights, void* q8,
                       std::size_t n_tokens, std::size_t k, std::size_t width,
                       hipStream_t stream);

// --- gemm.hpp ----------------------------------------------------------------

bool Nvfp4Gemm(const void* weights, const void* x, float* out, std::size_t rows,
               std::size_t cols, std::size_t n_tokens, hipStream_t stream);
/// Iq2xxsRouteSlots with max_tiles = Iq2xxsRouteMaxTiles(slots, experts).
/// `tiles` holds four int32 per tile.
bool Iq2xxsRouteSlots(const std::int32_t* ids, std::int32_t* cursor,
                      std::int32_t* order, std::int32_t* tiles,
                      std::int32_t* num_tiles, std::size_t slots,
                      std::size_t experts, hipStream_t stream);
/// Iq2xxsRoutedGemm over the route of `ids` (which only ties the call to
/// the routing that wrote `order`, `tiles` and `num_tiles`); `x` holds one
/// row per token (slot / used) or, with `x_per_slot`, one per slot.
bool Iq2xxsRoutedGemm(const std::int32_t* ids, const void* weights,
                      const std::uint32_t* grid, const float* x, float* out,
                      const std::int32_t* order, const std::int32_t* tiles,
                      const std::int32_t* num_tiles, std::size_t rows,
                      std::size_t cols, bool relu2, std::size_t slots,
                      std::size_t experts, std::size_t used, bool x_per_slot,
                      hipStream_t stream);

// --- gemv.hpp ----------------------------------------------------------------

/// Dense Iq2xxsGemv (no ids).
bool Iq2xxsGemv(const void* weights, const std::uint32_t* grid, const float* x,
                float* out, std::size_t rows, std::size_t cols,
                std::size_t n_tokens, bool relu2, bool accumulate,
                hipStream_t stream);
/// Routed Iq2xxsGemv: `ids` [n_tokens][used] experts; `x` one row per token
/// (slot / used) or, with `x_per_slot`, one per slot.
bool Iq2xxsGemvRouted(const void* weights, const std::uint32_t* grid,
                      const float* x, const std::int32_t* ids, float* out,
                      std::size_t rows, std::size_t cols, std::size_t n_tokens,
                      std::size_t used, bool x_per_slot, bool relu2,
                      bool accumulate, hipStream_t stream);
/// gemv.hpp's ToBf16: `n` floats (a multiple of 4) to BF16.
bool ActivationsToBf16(const float* x, void* out, std::size_t n,
                       hipStream_t stream);
/// Dense Iq2xxsGemv on BF16 activation rows (`x_bf16`), storing F32 rows.
bool Iq2xxsGemvBf16(const void* weights, const std::uint32_t* grid,
                    const void* x_bf16, float* out, std::size_t rows,
                    std::size_t cols, std::size_t n_tokens, bool relu2,
                    bool accumulate, hipStream_t stream);
/// Dense Iq2xxsGemv on BF16 activation rows, storing BF16 rows.
bool Iq2xxsGemvBf16StoreBf16(const void* weights, const std::uint32_t* grid,
                             const void* x_bf16, void* out_bf16,
                             std::size_t rows, std::size_t cols,
                             std::size_t n_tokens, bool relu2,
                             hipStream_t stream);
/// Iq2xxsRouteShort with max_tiles = slots. `tiles` holds four int32 per
/// tile.
bool Iq2xxsRouteShort(const std::int32_t* ids, std::int32_t* order,
                      std::int32_t* tiles, std::int32_t* num_tiles,
                      std::size_t slots, hipStream_t stream);
/// Routed Iq2xxsGemv on BF16 activations over the short route of `ids`
/// (which only ties the call to the routing that wrote `order`, `tiles` and
/// `num_tiles`), storing F32 rows; `x_bf16` one row per token (slot / used)
/// or, with `x_per_slot`, one per slot.
bool Iq2xxsGemvTiles(const std::int32_t* ids, const void* weights,
                     const std::uint32_t* grid, const void* x_bf16, float* out,
                     const std::int32_t* order, const std::int32_t* tiles,
                     const std::int32_t* num_tiles, std::size_t rows,
                     std::size_t cols, std::size_t slots, std::size_t used,
                     bool x_per_slot, bool relu2, bool accumulate,
                     hipStream_t stream);
/// Iq2xxsGemvTiles storing BF16 rows.
bool Iq2xxsGemvTilesStoreBf16(const std::int32_t* ids, const void* weights,
                              const std::uint32_t* grid, const void* x_bf16,
                              void* out_bf16, const std::int32_t* order,
                              const std::int32_t* tiles,
                              const std::int32_t* num_tiles, std::size_t rows,
                              std::size_t cols, std::size_t slots,
                              std::size_t used, bool x_per_slot, bool relu2,
                              hipStream_t stream);

// --- sample.hpp --------------------------------------------------------------

bool LogitHeads(const float* logits, std::size_t rows, std::size_t vocab,
                float temperature, std::size_t keep, float top_p,
                std::size_t minimum, void* scratch, void* records,
                void* candidates, hipStream_t stream);

// --- drafter_kernels.hpp -----------------------------------------------------

bool ToBf16(const float* x, void* out, std::size_t n, hipStream_t stream);
bool GemvBf16(const void* w, const float* x, float* out, std::size_t rows,
              std::size_t cols, std::size_t n_tokens, hipStream_t stream);
/// `act`: 0 none, 1 SiLU.
bool ScaleBias(float* x, float scale, const float* bias, std::size_t rows,
               std::size_t cols, std::uint32_t act, hipStream_t stream);
bool SiluGate(const float* gate, float gate_scale, const float* up,
              float up_scale, float* out, std::size_t n, hipStream_t stream);
bool DflashConv(const float* x, const float* dynamic, const float* base,
                float* out, std::size_t n_tokens, std::size_t block,
                std::size_t hidden, std::size_t group, std::size_t kernel,
                std::size_t side, hipStream_t stream);
bool RopeNeox(float* x, const std::int32_t* first, std::size_t n_tokens,
              std::size_t heads, std::size_t head_dim, float theta,
              hipStream_t stream);
bool RingAppend(const float* k, const float* v, void* k_ring, void* v_ring,
                const std::int32_t* first, std::size_t n_tokens,
                std::size_t kv_dim, std::size_t ring, hipStream_t stream);
bool RowsToRing(const float* x, float* ring, const std::int32_t* first,
                std::size_t n_tokens, std::size_t width, std::size_t ring_rows,
                hipStream_t stream);
bool DrafterAttention(const float* q, const void* k_ring, const void* v_ring,
                      const float* sinks, float* out, const std::int32_t* first,
                      std::size_t n_tokens, std::size_t heads,
                      std::size_t kv_heads, std::size_t ring,
                      std::size_t window, float scale, hipStream_t stream);
bool TopK8(const float* logits, std::size_t rows, std::size_t vocab,
           std::int32_t* ids, float* values, float* lse, hipStream_t stream);
bool LiliCorrFeatures(const float* values, const float* lse, const float* w,
                      const float* b, float* out, std::size_t slots,
                      hipStream_t stream);
bool LiliCorrCombine(const float* token, const float* pass,
                     const float* feature, const float* slot_embd,
                     const float* rank_embd, float* x, std::size_t slots,
                     std::size_t width, hipStream_t stream);
bool NodeAttention(const float* qkv, const float* bias, float* out,
                   std::size_t nodes, std::size_t heads, float scale,
                   hipStream_t stream);
bool FactorInput(const float* x, const float* anchor, float* out,
                 std::size_t nodes, std::size_t width, hipStream_t stream);
bool BiasL2Norm(float* x, const float* bias, std::size_t rows,
                std::size_t width, float eps, hipStream_t stream);
bool LatticeScores(const float* in, const float* out, const float* anchor,
                   float* scores, std::size_t slots, std::size_t width,
                   float scale, hipStream_t stream);

// --- vision.hpp (op: 0 BF16, 1 FP16) -----------------------------------------

bool VisionToOperand(const float* in, void* out, std::size_t n,
                     std::uint32_t op, hipStream_t stream);
bool VisionBf16ToOperand(const void* in, void* out, std::size_t n,
                         std::uint32_t op, hipStream_t stream);
bool VisionEmbed(float* x, const float* prefix, const float* positions,
                 std::size_t prefix_tokens, std::size_t grid_rows,
                 std::size_t grid_cols, std::size_t side, std::size_t width,
                 hipStream_t stream);
bool VisionLayerNorm(const float* x, const float* w, const float* b, void* out,
                     std::size_t rows, std::size_t width, float eps,
                     std::uint32_t op, hipStream_t stream);
bool VisionBias(const float* y, const float* bias, void* out, std::size_t rows,
                std::size_t cols, std::uint32_t op, hipStream_t stream);
bool VisionBiasGelu(const float* y, const float* bias, void* out,
                    std::size_t rows, std::size_t cols, std::uint32_t op,
                    hipStream_t stream);
bool VisionResidual(float* x, const float* y, const float* bias,
                    std::size_t rows, std::size_t cols, hipStream_t stream);
bool VisionAttention(const void* qkv, void* out, std::size_t tokens,
                     std::size_t heads, float scale, std::uint32_t op,
                     hipStream_t stream);
bool VisionMergeNorm(const float* x, const float* w, void* out,
                     std::size_t prefix_tokens, std::size_t grid_rows,
                     std::size_t grid_cols, std::size_t width, float eps,
                     std::uint32_t op, hipStream_t stream);
bool VisionRelu2(const float* y, void* out, std::size_t n, std::uint32_t op,
                 hipStream_t stream);

// --- mmq/nb_mmq.h ------------------------------------------------------------

/// Reserves the tier's scratch arena of `stream` (nb_mmq_bind_stream), so
/// that a later product may be recorded by stream capture.
bool MmqBindStream(hipStream_t stream);
/// Frees the arena after the stream's queued work (nb_mmq_release_stream).
void MmqReleaseStream(hipStream_t stream);
/// nb_mmq_dense for one weight format.
bool MmqDenseQ8_0(const void* w, const float* x, float* out, std::size_t m,
                  std::size_t n, std::size_t k, hipStream_t stream);
bool MmqDenseIq2Xxs(const void* w, const float* x, float* out, std::size_t m,
                    std::size_t n, std::size_t k, hipStream_t stream);
bool MmqDenseNvfp4(const void* w, const float* x, float* out, std::size_t m,
                   std::size_t n, std::size_t k, hipStream_t stream);
/// nb_mmq_dense_q8_1 for one weight format, scale 1, no bias.
bool MmqDenseQ8_1Q8_0(const void* w, const void* xq, float* out, std::size_t m,
                      std::size_t n, std::size_t k, hipStream_t stream);
bool MmqDenseQ8_1Iq2Xxs(const void* w, const void* xq, float* out,
                        std::size_t m, std::size_t n, std::size_t k,
                        hipStream_t stream);
bool MmqDenseQ8_1Nvfp4(const void* w, const void* xq, float* out, std::size_t m,
                       std::size_t n, std::size_t k, hipStream_t stream);
/// nb_mmq_dense_q8_1 of one column with `scale` and the bias fused:
/// out = scale * W xq + out (the residual update; bias == out).
bool MmqResidualQ8_0(const void* w, const void* xq, float scale, float* out,
                     std::size_t m, std::size_t k, hipStream_t stream);
bool MmqResidualIq2Xxs(const void* w, const void* xq, float scale, float* out,
                       std::size_t m, std::size_t k, hipStream_t stream);
bool MmqResidualNvfp4(const void* w, const void* xq, float scale, float* out,
                      std::size_t m, std::size_t k, hipStream_t stream);

}  // namespace gufo::models::nemotron::rocm::flat

#endif  // GUFO_MODELS_NEMOTRON_KERNELS_ROCM_ADAPTERS_HPP_
