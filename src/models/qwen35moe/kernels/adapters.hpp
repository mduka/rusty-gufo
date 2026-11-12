// Flat forms of the qwen35moe launchers whose signatures the Rust launcher
// contract cannot describe directly (docs/development/DESIGN.md,
// "launch.toml"): structs passed by value or reference, spans, enums of
// another width, parameters after the stream, the quantized GEMM tier's
// scratch and the GEMV route that would allocate it. Each adapter forwards
// after checking what its launcher assumes; it belongs to the trusted base
// like the launchers themselves. The kernels are the extension's
// (vendor/, strycast-qwen35moe-gufo-extension 4882b7a).

#ifndef GUFO_MODELS_QWEN35MOE_KERNELS_ADAPTERS_HPP_
#define GUFO_MODELS_QWEN35MOE_KERNELS_ADAPTERS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen35moe/kernels/vendor/family/hip_common/execution_policy.hpp"
#include "src/models/qwen35moe/kernels/vendor/family/hip_common/gemm_route.hpp"
#include "src/models/qwen35moe/kernels/vendor/family/hip_common/rope.hpp"

namespace gufo::models::qwen35moe::hip {

// --- GEMV
// ---------------------------------------------------------------------

/// LaunchGEMV in `mode`, except the route through the vendored vector kernel
/// (Ornith-1.5-122B's Q4_K 3072 x 8192 projections), whose activation
/// scratch would come from the tier's own arena: that route throws.
void LaunchGEMVChecked(const void* a, core::GgmlType a_type, const float* x,
                       float* y, std::size_t m, std::size_t k,
                       hipStream_t stream, models::qwen::QwenGemmMode mode);

/// LaunchGEMVRows in `mode` with the same exception as LaunchGEMVChecked.
void LaunchGEMVRowsChecked(const void* a, core::GgmlType a_type, const float* x,
                           float* y, std::size_t rows, std::size_t m,
                           std::size_t k, hipStream_t stream,
                           models::qwen::QwenGemmMode mode);

// --- Recurrent state
// ------------------------------------------------------------

/// LaunchBatchedSSMConvRecurrenceRowSplit. Here and in the SSM adapters
/// below, a pass forks from `*_in` into `conv_state`/`deltanet_state` when
/// they are given and updates those in place when they are null
/// (QwenRecurrentStateIo). With `conv_fused` the causal convolution of most
/// rows is already in `conv_out` (TryLaunchBatchedQuantGEMMSsmConv).
void LaunchBatchedSSMConvRecurrenceRowSplitFlat(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    const float* conv_in, float* conv_out, void* deltanet_state,
    const void* deltanet_in, const float* alpha_buf, const float* beta_buf,
    const float* ssm_a, const float* ssm_dt, const float* ssm_norm,
    const float* gate, float* out_buf, void* q8_out, float* kq_scales,
    float* alpha_beta, std::uint32_t layer_idx, std::size_t batch_size,
    std::size_t qkv_size, std::uint32_t num_key_heads, std::uint32_t num_heads,
    std::uint32_t key_dim, std::uint32_t val_dim, hipStream_t stream,
    QwenRecurrentStateStorage state_storage, void* fp16_out, bool conv_fused);

/// LaunchSSMConvRecurrence with FP32 recurrent state and the replay capture
/// as separate pointers (all null: no capture); the rings hold
/// `replay_capacity` slots per layer.
void LaunchSSMConvRecurrenceFlat(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    const float* conv_in, float* conv_out, float* deltanet_state,
    const float* deltanet_in, const float* alpha_buf, const float* beta_buf,
    const float* ssm_a, const float* ssm_dt, const float* ssm_norm,
    const float* gate, float* out_buf, std::uint32_t layer_idx,
    std::size_t qkv_size, std::uint32_t num_key_heads, std::uint32_t num_heads,
    std::uint32_t key_dim, std::uint32_t val_dim, float* replay_qkv,
    float* replay_alpha, float* replay_beta,
    const std::uint32_t* replay_position, const std::uint32_t* replay_enabled,
    std::uint32_t replay_capacity, hipStream_t stream);

/// LaunchSSMConvRecurrenceRows with FP32 recurrent state and the replay
/// capture as separate pointers: `rows` consecutive tokens of one sequence.
void LaunchSSMConvRecurrenceRowsFlat(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    const float* conv_in, float* conv_out, float* deltanet_state,
    const float* deltanet_in, const float* alpha_buf, const float* beta_buf,
    const float* ssm_a, const float* ssm_dt, const float* ssm_norm,
    const float* gate, float* out_buf, std::uint32_t layer_idx,
    std::size_t qkv_size, std::uint32_t num_key_heads, std::uint32_t num_heads,
    std::uint32_t key_dim, std::uint32_t val_dim, std::uint32_t rows,
    std::size_t projection_row_stride, std::size_t inner_row_stride,
    float* replay_qkv, float* replay_alpha, float* replay_beta,
    const std::uint32_t* replay_position, const std::uint32_t* replay_enabled,
    std::uint32_t replay_capacity, hipStream_t stream);

/// LaunchSSMConvRecurrenceBatch with FP32 recurrent state over flat lists:
/// sequence `i` has its conv and DeltaNet state, its replay capture (null
/// when absent; `replay_capacity[i]` slots per layer, all 16 when the list is
/// null) and rows `row_offset[i]..+rows[i]` of the shared inputs.
void LaunchSSMConvRecurrenceSequencesFlat(
    const float* qkv_in, const float* conv_weights, float* conv_out,
    const float* alpha_buf, const float* beta_buf, const float* ssm_a,
    const float* ssm_dt, const float* ssm_norm, const float* gate,
    float* out_buf, float* const* conv_state, const float* const* conv_in,
    float* const* deltanet_state, const float* const* deltanet_in,
    float* const* replay_qkv, float* const* replay_alpha,
    float* const* replay_beta, const std::uint32_t* const* replay_position,
    const std::uint32_t* const* replay_enabled,
    const std::uint32_t* replay_capacity, const std::uint32_t* row_offset,
    const std::uint32_t* rows, std::uint32_t count, std::uint32_t layer_idx,
    std::size_t qkv_size, std::uint32_t num_key_heads, std::uint32_t num_heads,
    std::uint32_t key_dim, std::uint32_t val_dim,
    std::size_t projection_row_stride, std::size_t inner_row_stride,
    hipStream_t stream);

/// LaunchCaptureBatchedSsmReplay with the capture as separate pointers.
void LaunchCaptureBatchedSsmReplayFlat(
    const float* qkv, const float* alpha, const float* beta, float* replay_qkv,
    float* replay_alpha, float* replay_beta, const std::uint32_t* enabled,
    std::uint32_t replay_capacity, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::size_t qkv_size,
    std::size_t time_step_rank, hipStream_t stream);

// --- Attention
// ------------------------------------------------------------------

/// PrepareBatchedGPUArgmax as a launcher: true once the batched argmax
/// kernels are loaded (nothing is launched).
[[nodiscard]] bool PrepareBatchedGPUArgmaxFlat(hipStream_t stream);

/// LaunchBatchedGPUArgmax with its scratch as pointer and length.
void LaunchBatchedGPUArgmaxFlat(const float* logits, std::uint32_t* out_tokens,
                                std::size_t batch_size, std::size_t vocab_size,
                                float* scratch, std::size_t scratch_len,
                                hipStream_t stream);

/// LaunchQwenWmmaAttention over the token-major cache after the fused
/// QK-norm kernel wrote this chunk (no FP32 cache, no LSE, key_begin 0,
/// skip_kv_write), with the packed K/V workspaces as pointer and length;
/// output rows before `first_query` are skipped. `kv_q8` reads the 8-bit
/// cache (kv_q8.hpp), which ignores the workspaces.
[[nodiscard]] bool LaunchQwenWmmaAttentionFlat(
    const float* q, const float* k, const float* v, const float* gate,
    const void* k_cache_f16, const void* v_cache_f16, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, float* packed_k,
    std::size_t packed_k_len, float* packed_v, std::size_t packed_v_len,
    std::size_t first_query, bool kv_q8, hipStream_t stream);

/// LaunchCausalDecodeAttention over the token-major cache (no FP32 cache)
/// with the split-K scratch as pointer and length; `kv_q8` reads the 8-bit
/// cache.
void LaunchCausalDecodeAttentionFlat(
    const float* q, const float* gate, const void* k_cache_f16,
    const void* v_cache_f16, float* out_context, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    float* scratch, std::size_t scratch_len, bool kv_q8, hipStream_t stream);

// --- Sampling
// -------------------------------------------------------------------

/// Greedy LaunchDFlashSelectorBatch for one sequence of `count` proposal
/// rows. The anchor token, which the selector reads as a codebook row, comes
/// from a checked index buffer and is copied in stream order into `tokens[0]`;
/// the proposals land in `tokens[1..=count]`. The request's numeric penalties
/// (kernels::DFlashSampling) evolve across the chain: `penalties` holds its
/// generated counts sorted by token (three words each: token, generated
/// count, repeated), `history` its repetition window; either may be null
/// when its count is zero.
void LaunchDFlashSelectorFlat(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    const std::uint32_t* anchor, std::uint32_t* tokens, float* confidences,
    float* partial_scores, std::uint32_t* partial_ids, std::uint32_t count,
    std::uint32_t vocab_size, std::uint32_t selector_rank,
    std::uint32_t selector_top_k, float repeat_penalty, float frequency_penalty,
    float presence_penalty, std::uint32_t repeat_last_n,
    const std::uint32_t* penalties, std::uint32_t penalty_count,
    const std::uint32_t* history, std::uint32_t history_count,
    hipStream_t stream);

/// LaunchDFlashSelectorBatch for one sequence of `count` proposal rows drawn
/// at `temperature` with one draw per row: like LaunchDFlashSelectorFlat,
/// plus each row's top-k candidates and their probabilities.
void LaunchDFlashSelectorSampledFlat(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    const std::uint32_t* anchor, std::uint32_t* tokens, float* confidences,
    float* partial_scores, std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::uint32_t count, float temperature, std::uint32_t vocab_size,
    std::uint32_t selector_rank, std::uint32_t selector_top_k,
    float repeat_penalty, float frequency_penalty, float presence_penalty,
    std::uint32_t repeat_last_n, const std::uint32_t* penalties,
    std::uint32_t penalty_count, const std::uint32_t* history,
    std::uint32_t history_count, hipStream_t stream);

/// LaunchDFlashSelectorFlat (greedy, `temperature` 0) or
/// LaunchDFlashSelectorSampledFlat over the draft vocabulary
/// (kernels::DFlashSelectorVocab): the logits rows hold the head's first
/// `prefix` IDs, `table_logits` [count][table_rows] the scores of the
/// request's table, whose IDs are `table_ids` (-1 for an empty slot, scored
/// -inf). The proposals and candidates are full vocabulary IDs.
void LaunchDFlashSelectorVocabFlat(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    const std::uint32_t* anchor, std::uint32_t* tokens, float* confidences,
    float* partial_scores, std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    const float* table_logits, const std::int32_t* table_ids,
    std::uint32_t table_rows, std::uint32_t prefix, std::uint32_t count,
    float temperature, std::uint32_t vocab_size, std::uint32_t selector_rank,
    std::uint32_t selector_top_k, float repeat_penalty, float frequency_penalty,
    float presence_penalty, std::uint32_t repeat_last_n,
    const std::uint32_t* penalties, std::uint32_t penalty_count,
    const std::uint32_t* history, std::uint32_t history_count,
    hipStream_t stream);

/// LaunchDFlashSelectorVocabFlat over logits rows `logits_stride` apart, of
/// which each row's first `prefix` columns are read: a batch whose other
/// requests draft over the full head projects every row over the whole
/// vocabulary (stride `vocab_size`), a batch that drafts only through draft
/// vocabularies over the prefix (stride `prefix`), as
/// QwenDFlashGpuExecutor::ForwardBlockBatch passes its stride.
void LaunchDFlashSelectorVocabStridedFlat(
    const float* logits, std::uint32_t logits_stride,
    const float* projected_hidden, const void* predecessor_codebook_bf16,
    const void* successor_codebook_bf16, const std::uint32_t* anchor,
    std::uint32_t* tokens, float* confidences, float* partial_scores,
    std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    const float* table_logits, const std::int32_t* table_ids,
    std::uint32_t table_rows, std::uint32_t prefix, std::uint32_t count,
    float temperature, std::uint32_t vocab_size, std::uint32_t selector_rank,
    std::uint32_t selector_top_k, float repeat_penalty, float frequency_penalty,
    float presence_penalty, std::uint32_t repeat_last_n,
    const std::uint32_t* penalties, std::uint32_t penalty_count,
    const std::uint32_t* history, std::uint32_t history_count,
    hipStream_t stream);

/// LaunchGPUSampling of one logits row with the sampling parameters as
/// scalars and the workspace as buffers. The penalties are already on the
/// device (three words each: token, generated count, repeated); the sort
/// storage must hold what GpuSamplingSortStorageBytes asks for, else the
/// launch fails.
void LaunchGPUSamplingFlat(const float* logits, std::uint32_t* out_token,
                           std::size_t vocab_size, float temperature,
                           std::int32_t top_k, float top_p, float min_p,
                           std::size_t min_keep, float repeat_penalty,
                           float frequency_penalty, float presence_penalty,
                           double uniform, const std::uint32_t* penalties,
                           std::size_t penalty_count, float* adjusted_logits,
                           float* sorted_logits, std::uint32_t* token_ids,
                           std::uint32_t* sorted_token_ids, void* sort_storage,
                           std::size_t sort_storage_bytes, hipStream_t stream);

/// LaunchGPUSpeculativeSampling of one verification row: accepts
/// `draft_token` with the lossless ratio or draws from the residual. The draft
/// candidates and penalties are already on the device (penalties as three
/// words each); the workspace buffers and sort storage as in
/// LaunchGPUSamplingFlat.
void LaunchGPUSpeculativeSamplingFlat(
    const float* logits, std::uint32_t* out_token, std::uint32_t* out_accepted,
    std::size_t vocab_size, float temperature, std::int32_t top_k, float top_p,
    float min_p, std::size_t min_keep, float repeat_penalty,
    float frequency_penalty, float presence_penalty, std::uint32_t draft_token,
    float draft_token_probability, const std::uint32_t* candidate_ids,
    const float* candidate_probabilities, std::size_t candidate_count,
    double acceptance_uniform, double residual_uniform,
    const std::uint32_t* penalties, std::size_t penalty_count,
    float* adjusted_logits, float* sorted_logits, std::uint32_t* token_ids,
    std::uint32_t* sorted_token_ids, void* sort_storage,
    std::size_t sort_storage_bytes, hipStream_t stream);

/// LaunchGPUSortLogits over caller buffers: the finite logits in descending
/// order with ascending IDs on ties into `sorted_logits`/`sorted_token_ids`,
/// or only their exact first `keep` entries when `keep` is nonzero. The sort
/// storage must hold what GpuSamplingSortStorageBytes asks for.
void LaunchGPUSortLogitsFlat(const float* logits, float* adjusted_logits,
                             float* sorted_logits, std::uint32_t* token_ids,
                             std::uint32_t* sorted_token_ids,
                             void* sort_storage, std::size_t sort_storage_bytes,
                             std::size_t vocab_size, std::size_t keep,
                             hipStream_t stream);

/// The 64 best logits of one row as IDs `ids[0..64)` and their scores
/// packed behind them (`ids + 64` as floats), the Flash-Next MTP candidate
/// selection (qwen38_flash_next::rocm::MtpTopCandidates).
void MtpTopCandidatesPacked(const float* logits, std::uint32_t* ids,
                            std::uint32_t* scratch_ids, std::uint32_t vocab,
                            hipStream_t stream);

// --- Routed experts
// ---------------------------------------------------------------

/// Copies `ids` with every negative entry (the routers' padding: a row whose
/// logits are not finite) replaced by expert 0, for the expert kernels that
/// index weights with every ID (the per-slot GEMVs and the grouped BF16
/// GEMM). Finite router rows hold no padding, so their IDs are unchanged.
void MoeIdsUnpad(const std::int32_t* ids, std::int32_t* out,
                 std::uint32_t count, hipStream_t stream);

/// LaunchMoeGroupSlots with the grouped scratch as three pointers.
void LaunchMoeGroupSlotsFlat(const std::int32_t* ids, std::uint32_t slots,
                             std::uint32_t n_experts,
                             std::int32_t* sorted_slots, std::int32_t* tiles,
                             std::int32_t* expert_bounds, hipStream_t stream);

/// LaunchMoeGroupedBf16Gemm over the grouping LaunchMoeGroupSlotsFlat
/// derived from `ids` (passed only to tie the grouping to its IDs).
void LaunchMoeGroupedBf16GemmFlat(
    const void* w, const float* x, float* out, std::size_t m, std::size_t k,
    std::uint32_t slots, std::uint32_t n_experts, std::uint32_t x_token_divisor,
    const std::int32_t* ids, const std::int32_t* sorted_slots,
    const std::int32_t* tiles, const std::int32_t* expert_bounds,
    hipStream_t stream);

/// moe_f16::RoutedF16Gemm with the GGML format (the routed kernels' type IDs
/// are the GGUF ones). False, launching nothing, for other formats.
[[nodiscard]] bool RoutedF16GemmFlat(
    const void* w, core::GgmlType type, const void* x,
    const std::int32_t* tiles, std::uint32_t n_tiles, std::uint32_t tile_rows,
    const std::int32_t* pad_bounds, const std::int32_t* rows_in,
    const std::int32_t* rows_out, const float* swiglu_gate, float* out,
    void* out_half, std::size_t m, std::size_t k, hipStream_t stream);

/// moe_f16::RoutedGatedF16Gemm with the GGML format.
[[nodiscard]] bool RoutedGatedF16GemmFlat(
    const void* gate, const void* up, core::GgmlType type, const void* x,
    const std::int32_t* tiles, std::uint32_t n_tiles, std::uint32_t tile_rows,
    const std::int32_t* pad_bounds, const std::int32_t* rows_in,
    const std::int32_t* rows_out, void* out, std::size_t m, std::size_t k,
    hipStream_t stream);

/// qwen38_flash_next::rocm::MoeEpilogueVec4F16 over untyped F16 rows.
void MoeEpilogueVec4F16Flat(const void* expert_out, const float* weights,
                            const float* shared_out, const float* gate,
                            std::uint32_t gate_stride, float* out,
                            std::uint32_t n_tokens, std::uint32_t k,
                            std::uint32_t dim, hipStream_t stream);

// --- Quantized GEMM tier (vendor/fnext/kernels/rocm/mmq/qfn_mmq.h)
// -----------------

/// Initialises the tier once per process (device context and pool) and
/// loads the dense decode functions; true when it is ready.
[[nodiscard]] bool MmqPrepare(hipStream_t stream);

/// Frees the tier's arena of `stream` after its queued work.
void MmqReleaseStream(hipStream_t stream);

/// q35_mmq_moe_vec (without a second weight set) with the tier's
/// allocations served from `scratch` (256-byte aligned). False when the tier
/// declines the shape or the scratch is unusable.
[[nodiscard]] bool MmqMoeVec(core::GgmlType type, const void* w, const float* x,
                             const std::int32_t* ids, float* out,
                             std::uint32_t m, std::uint32_t k,
                             std::uint32_t n_tokens, std::uint32_t n_experts,
                             std::uint32_t n_used, void* scratch,
                             std::size_t scratch_bytes, hipStream_t stream);

/// q35_mmq_moe_gated_vec with the tier's allocations served from `scratch`.
[[nodiscard]] bool MmqMoeGatedVec(
    core::GgmlType type, const void* gate, const void* up, const float* x,
    const std::int32_t* ids, float* out, std::uint32_t m, std::uint32_t k,
    std::uint32_t n_tokens, std::uint32_t n_experts, std::uint32_t n_used,
    void* scratch, std::size_t scratch_bytes, hipStream_t stream);

}  // namespace gufo::models::qwen35moe::hip

#endif  // GUFO_MODELS_QWEN35MOE_KERNELS_ADAPTERS_HPP_
