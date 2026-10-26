// Flat forms of Flash-Next launchers whose signatures the Rust launcher
// contract cannot describe directly (docs/RUST.md, "launch.toml"):
// structs passed by value, correlated optional buffers, status codes and the
// quantized GEMM tier's routing hints. Each adapter forwards after checking
// what its launcher assumes; it belongs to the trusted base like the
// launchers themselves.

#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ADAPTERS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ADAPTERS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// NarrowActivations to F16 rows (typed output).
void NarrowToHalf(const float* x, __half* out, std::size_t count,
                  hipStream_t stream);

/// PleConv with the rollback rows as a list: `snapshot_count` rows of
/// [(kernel - 1) * dilation][channels], none for a plain forward. False,
/// launching nothing, when the rows cannot hold every proper prefix.
[[nodiscard]] bool PleConvRows(const float* in, const float* w, float* history,
                               float* history_scratch, float* out,
                               float* const* snapshots,
                               std::uint32_t snapshot_count,
                               std::uint32_t n_tokens, std::uint32_t channels,
                               std::uint32_t kernel, std::uint32_t dilation,
                               hipStream_t stream);

/// GatedDeltaNet over 128-wide heads with the rollback rows split into the
/// first row (the full state, [v_heads][128][128], and the conv history,
/// [kernel - 1][channels]) and `rest` later rows of each list
/// (GdnRollbackRowFloats and [kernel - 1][channels]). Exactly one of `out`,
/// `out_q8` and `out_half` receives the rows; `out_rows` with `out` also
/// their Q8_1 rows (v_heads * 128 % 512 == 0). A nonzero
/// `out_half_stride` places the F16 rows that many elements apart. False,
/// launching nothing, when only one first row is given, rows come without a
/// first row, the rows cannot hold every proper prefix, or snapshots meet
/// `convolved`. A prefill checkpoint (`checkpoint_state` with
/// `checkpoint_history`, without rollback rows) receives the state and
/// history after the first `checkpoint_tokens` (1 to n_tokens) rows.
[[nodiscard]] bool GatedDeltaNetRows(
    const float* qkv, std::uint32_t qkv_stride, const float* z,
    std::uint32_t z_stride, const float* alpha_beta, const float* conv_w,
    const float* a, const float* dt, const float* norm_w, float* conv_state,
    float* conv_scratch, float* qn, float* kn, float* raw, float* state,
    float* out, void* out_q8, __half* out_half, void* out_rows,
    float* state_first, float* conv_first, float* const* state_rest,
    float* const* conv_rest, std::uint32_t rest, std::uint32_t n_tokens,
    std::uint32_t k_heads, std::uint32_t v_heads, std::uint32_t kernel,
    bool row_split, bool convolved, float eps, std::uint32_t out_half_stride,
    float* checkpoint_state, float* checkpoint_history,
    std::uint32_t checkpoint_tokens, hipStream_t stream);

/// HcMixEpilogue that also writes `mixed` as the tier's Q8_1 rows; false,
/// launching nothing, unless they are unpadded (hidden % 512 == 0).
[[nodiscard]] bool HcMixEpilogueQ8_1(const float* xn, const float* gate,
                                     const float* inject_w, float* mixed,
                                     float* inject, void* rows,
                                     std::uint32_t n_tokens,
                                     std::uint32_t hidden,
                                     std::uint32_t streams, hipStream_t stream);

/// HcCombineQ8_1; false, launching nothing, unless the norm's Q8_1 rows are
/// unpadded (streams * hidden % 512 == 0).
[[nodiscard]] bool HcCombineQ8_1Rows(
    float* res, const float* block_out, const float* inject,
    std::uint32_t inject_parts, const float* gamma, float* xn, void* xn_rows,
    std::uint32_t n_tokens, std::uint32_t hidden, std::uint32_t streams,
    float eps, hipStream_t stream);

/// HcCombineF16 without deferral.
void HcCombineF16Plain(float* res, const float* block_out, const float* inject,
                       std::uint32_t inject_parts, const float* gamma,
                       __half* xn, void* xn_q8, std::uint32_t n_tokens,
                       std::uint32_t hidden, std::uint32_t streams, float eps,
                       hipStream_t stream);

/// HcCombineMoeF16 without a deferred update or inject partials.
[[nodiscard]] bool HcCombineMoeF16Plain(
    float* res, const __half* expert_out, const float* weights,
    const float* shared_out, const float* gate, std::uint32_t gate_stride,
    std::uint32_t used, const float* inject, std::uint32_t inject_parts,
    const float* gamma, __half* xn, void* xn_q8, std::uint32_t n_tokens,
    std::uint32_t hidden, std::uint32_t streams, float eps, hipStream_t stream);

/// SiluScaleQ8_1 over `n` rows of `k`, each padded to the tier's width (its
/// padding blocks left as they are); false unless k % 32 == 0.
[[nodiscard]] bool SiluScaleQ8_1Rows(float* x, float scale, void* rows,
                                     std::uint32_t n, std::uint32_t k,
                                     hipStream_t stream);

/// SigmoidMulQ8_1 over `n` rows of `k`; false, launching nothing, unless
/// the rows are unpadded (k % 512 == 0).
[[nodiscard]] bool SigmoidMulQ8_1Rows(float* x, const float* g, void* rows,
                                      std::uint32_t n, std::uint32_t k,
                                      hipStream_t stream);

/// HcCombineF16 that defers its residual update into `defer_w` and, given
/// `inject_w`, replaces the `inject` logits it reads with the next mixer's
/// inject partials (every thread reads them before its first barrier).
/// False, launching nothing, unless four streams of at most 2,560 (a
/// multiple of 128) take the deferring kernel.
[[nodiscard]] bool HcCombineF16Deferred(
    float* res, const float* block_out, float* inject,
    std::uint32_t inject_parts, const float* gamma, __half* xn, void* xn_q8,
    std::uint32_t n_tokens, std::uint32_t hidden, std::uint32_t streams,
    float eps, float* defer_w, const float* inject_w, hipStream_t stream);

/// HcCombineMoeF16 that first applies a deferred combine's update
/// (`prior_out`, `prior_w`) and, given `inject_w`, replaces the `inject`
/// logits it reads with the next mixer's inject partials.
[[nodiscard]] bool HcCombineMoeF16Prior(
    float* res, const __half* expert_out, const float* weights,
    const float* shared_out, const float* gate, std::uint32_t gate_stride,
    std::uint32_t used, float* inject, std::uint32_t inject_parts,
    const float* gamma, __half* xn, void* xn_q8, std::uint32_t n_tokens,
    std::uint32_t hidden, std::uint32_t streams, float eps,
    const float* prior_out, const float* prior_w, const float* inject_w,
    hipStream_t stream);

/// DenseF16Gemm over F16 rows `input_stride` elements apart.
[[nodiscard]] bool DenseF16GemmStrided(const void* w, const __half* x,
                                       float* out, std::size_t batch,
                                       std::size_t m, std::size_t k,
                                       std::size_t input_stride,
                                       hipStream_t stream);

/// GatedDeltaNetBatch over `count` requests given as lists: request i has
/// `n_tokens[i]` (1-8) rows, and with `state_rows[0][i]` its rollback rows
/// `state_rows[r][i]` and `conv_rows[r][i]` for every r < n_tokens[i] - 1
/// (row 0 the full state). The descriptors go to `items_slot` (eight of 200
/// bytes): memory the host addresses at its device address and that no
/// queued work still reads.
/// False, launching nothing, when a request brings rows without the full
/// state, or too few rows.
[[nodiscard]] bool GatedDeltaNetBatchRows(
    void* items_slot, const float* const* qkv, std::uint32_t qkv_stride,
    const float* const* z, std::uint32_t z_stride,
    const float* const* alpha_beta, const float* conv_w, const float* a,
    const float* dt, const float* norm_w, float* const* conv_state,
    float* const* conv_scratch, float* const* qn, float* const* kn,
    float* const* raw, float* const* state, float* const* out,
    float* const* state_0, float* const* state_1, float* const* state_2,
    float* const* state_3, float* const* state_4, float* const* state_5,
    float* const* state_6, float* const* conv_0, float* const* conv_1,
    float* const* conv_2, float* const* conv_3, float* const* conv_4,
    float* const* conv_5, float* const* conv_6, const std::uint32_t* n_tokens,
    std::uint32_t count, std::uint32_t k_heads, std::uint32_t v_heads,
    float eps, hipStream_t stream);

/// RestoreGdnState from the first rollback row (the full state) and the
/// `rest` rows after it; false unless 1 <= keep <= rest + 1.
[[nodiscard]] bool RestoreGdnStateRows(
    float* state, const float* first, const float* const* rest,
    std::uint32_t rest_count, std::uint32_t keep, std::uint32_t k_heads,
    std::uint32_t v_heads, hipStream_t stream);

/// Argmax with its candidate scratch (n_tokens * kArgmaxParts candidates)
/// as untyped memory.
void ArgmaxFlat(const float* logits, void* scratch, std::int32_t* out,
                std::uint32_t n_tokens, std::uint32_t vocab,
                hipStream_t stream);

/// PenalizedArgmax over (token, generated count, repeated) triples, row r
/// ending at end_r (row 0 starting at 0), with its partial and output
/// candidates in untyped memory; false unless 1 <= rows <= 7.
bool PenalizedArgmaxFlat(const float* logits, const std::uint32_t* penalties,
                         std::uint32_t end_0, std::uint32_t end_1,
                         std::uint32_t end_2, std::uint32_t end_3,
                         std::uint32_t end_4, std::uint32_t end_5,
                         std::uint32_t end_6, float repeat, float frequency,
                         float presence, void* partial, void* out,
                         std::uint32_t rows, std::uint32_t vocab,
                         hipStream_t stream);

/// GatherArgmaxCandidates writing `rows` candidates to untyped memory.
void GatherArgmaxCandidatesFlat(const float* logits, const std::uint32_t* ids,
                                void* out, std::uint32_t rows,
                                std::uint32_t vocab, hipStream_t stream);

/// VerificationCandidates over penalty rows as in PenalizedArgmaxFlat,
/// writing its 8-byte candidates to untyped memory; false unless
/// 1 <= rows <= 7.
bool VerificationCandidatesFlat(
    const float* logits, const std::uint32_t* penalties, std::uint32_t end_0,
    std::uint32_t end_1, std::uint32_t end_2, std::uint32_t end_3,
    std::uint32_t end_4, std::uint32_t end_5, std::uint32_t end_6,
    std::uint32_t* ids, std::uint32_t* scratch_ids, void* out,
    std::uint32_t rows, std::uint32_t vocab, hipStream_t stream);

/// MtpTopCandidates with the scores packed behind the kMtpCandidates IDs in
/// `ids`, as the executor downloads them in one transfer.
void MtpTopCandidatesPacked(const float* logits, std::uint32_t* ids,
                            std::uint32_t* scratch_ids, std::uint32_t vocab,
                            hipStream_t stream);

// The quantized GEMM tier (mmq/qfn_mmq.h). Its entry points return a status;
// these return whether they launched. Routed entry points take the caller's
// scratch (256-byte aligned, `scratch_bytes` long) for the tier's stack
// allocations, so each stream's work and every graph that captures it keeps
// its own; a call that needs more scratch aborts.

/// Resolves the decode kernels the first token would otherwise load (the
/// tier's and the wide W8A8 GEMM), launching nothing, as the C++ executor
/// does before serving.
[[nodiscard]] bool PrepareDecodeKernels(hipStream_t stream);

/// Frees the tier's scratch arena of `stream`, which is about to be
/// destroyed, after its queued work (as the C++ executor's destructor).
void MmqReleaseStream(hipStream_t stream);

/// Q8_1 activations of `n` rows of `k` floats for the decode GEMVs.
[[nodiscard]] bool MmqQuantizeQ8_1(const float* x, void* x_q8, std::uint32_t n,
                                   std::uint32_t k, hipStream_t stream);

/// Q8_0 matrix-vector products over pre-quantized rows; the gated form
/// writes (up . x) * silu(gate . x).
[[nodiscard]] bool MmqQ8DenseVec(const void* w, const void* x_q8, float* out,
                                 std::uint32_t m, std::uint32_t n,
                                 std::uint32_t k, hipStream_t stream);
[[nodiscard]] bool MmqQ8GatedDenseVec(const void* up, const void* gate,
                                      const void* x_q8, float* out,
                                      std::uint32_t m, std::uint32_t n,
                                      std::uint32_t k, hipStream_t stream);

/// Routed vector projections (one or two weight sets) and the gated form.
[[nodiscard]] bool MmqMoeVec(WeightType type, const void* w, const float* x,
                             const std::int32_t* ids, float* out,
                             std::uint32_t m, std::uint32_t k,
                             std::uint32_t n_tokens, std::uint32_t n_experts,
                             std::uint32_t n_used, void* scratch,
                             std::size_t scratch_bytes, hipStream_t stream);
[[nodiscard]] bool MmqMoeVecPair(WeightType type, const void* w_a,
                                 const void* w_b, const float* x,
                                 const std::int32_t* ids, float* out_a,
                                 float* out_b, std::uint32_t m, std::uint32_t k,
                                 std::uint32_t n_tokens,
                                 std::uint32_t n_experts, std::uint32_t n_used,
                                 void* scratch, std::size_t scratch_bytes,
                                 hipStream_t stream);
[[nodiscard]] bool MmqMoeGatedVec(
    WeightType type, const void* gate, const void* up, const float* x,
    const std::int32_t* ids, float* out, std::uint32_t m, std::uint32_t k,
    std::uint32_t n_tokens, std::uint32_t n_experts, std::uint32_t n_used,
    void* scratch, std::size_t scratch_bytes, hipStream_t stream);
[[nodiscard]] bool MmqQ8MoeGatedVec(
    WeightType type, const void* gate, const void* up, const void* x_q8,
    const std::int32_t* ids, float* out, std::uint32_t m, std::uint32_t k,
    std::uint32_t n_tokens, std::uint32_t n_experts, std::uint32_t n_used,
    void* scratch, std::size_t scratch_bytes, hipStream_t stream);

/// Tiled routed projections with the column-grid hints the executor derives
/// from the expert counts (`max_rows` 0 and `tile_cols` 0: no hint).
[[nodiscard]] bool MmqMoeTiled(WeightType type, const void* w, const float* x,
                               const std::int32_t* ids, float* out,
                               std::uint32_t m, std::uint32_t k,
                               std::uint32_t n_tokens, std::uint32_t n_experts,
                               std::uint32_t n_used, std::uint32_t max_rows,
                               std::uint32_t tile_cols, void* scratch,
                               std::size_t scratch_bytes, hipStream_t stream);
[[nodiscard]] bool MmqQ4KMoePairTiled(
    const void* w_a, const void* w_b, const float* x, const std::int32_t* ids,
    float* out_a, float* out_b, std::uint32_t m, std::uint32_t k,
    std::uint32_t n_tokens, std::uint32_t n_experts, std::uint32_t n_used,
    std::uint32_t max_rows, std::uint32_t tile_cols, void* scratch,
    std::size_t scratch_bytes, hipStream_t stream);

/// The rope launches with their static YaRN constants as scalars
/// (`YarnRope`: enabled, low, high, inv_factor, mscale); disabled keeps the
/// original instruction sequence.
bool AttentionF16GemmYarn(
    const void* weights, const __half* input, const float* q_gamma,
    const float* k_gamma, float* query, float* gate, __half* keys,
    __half* values, std::uint32_t n_tokens, const std::uint32_t* position,
    float theta, float eps, hipStream_t stream,
    const qwen::vision::DeviceRope* rope, std::uint32_t yarn_enabled,
    float yarn_low, float yarn_high, float yarn_inv_factor, float yarn_mscale);
bool PrepareAttentionYarn(
    const float* packed, std::uint32_t stride, const float* q_gamma,
    const float* k_gamma, float* q, float* gate, __half* k_cache,
    __half* v_cache, std::uint32_t n_tokens, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t d, std::uint32_t rotary_dim,
    const std::uint32_t* start_pos, float theta, float eps, hipStream_t stream,
    const qwen::vision::DeviceRope* rope, bool prefill,
    std::uint32_t yarn_enabled, float yarn_low, float yarn_high,
    float yarn_inv_factor, float yarn_mscale);
void RopeYarn(float* x, std::uint32_t n_tokens, std::uint32_t heads,
              std::uint32_t d, std::uint32_t rotary_dim,
              const std::uint32_t* start_pos, float theta, hipStream_t stream,
              const qwen::vision::DeviceRope* rope, std::uint32_t yarn_enabled,
              float yarn_low, float yarn_high, float yarn_inv_factor,
              float yarn_mscale);
void PoolIndexerBlocksYarn(
    const float* raw_keys, const float* gamma, __half* blocks,
    const std::uint32_t* first_block, const std::uint32_t* start_pos,
    std::uint32_t n_tokens, std::uint32_t grid_blocks, std::uint32_t ratio,
    std::uint32_t dim, std::uint32_t rotary_dim, float theta, float eps,
    std::uint32_t capacity, hipStream_t stream,
    const qwen::vision::DeviceRope* rope, std::uint32_t yarn_enabled,
    float yarn_low, float yarn_high, float yarn_inv_factor, float yarn_mscale);

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ADAPTERS_HPP_
