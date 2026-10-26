// Flat forms of Qwen launchers whose signatures the Rust launcher contract
// cannot describe directly (docs/RUST.md, "launch.toml"): structs
// passed by value, spans, and parameters after the stream. Each adapter only
// forwards; it belongs to the trusted base like the launchers themselves.

#ifndef GUFO_MODELS_QWEN_KERNELS_ADAPTERS_HPP_
#define GUFO_MODELS_QWEN_KERNELS_ADAPTERS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/execution_policy.hpp"

namespace gufo::hip {

/// LaunchBatchedSSMConvRecurrenceRowSplit. Here and in the SSM adapters
/// below, a pass forks from `*_in` into `conv_state`/`deltanet_state` when
/// they are given and updates those in place when they are null
/// (QwenRecurrentStateIo).
void LaunchBatchedSSMConvRecurrenceRowSplitFlat(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    const float* conv_in, float* conv_out, void* deltanet_state,
    const void* deltanet_in, const float* alpha_buf, const float* beta_buf,
    const float* ssm_a, const float* ssm_dt, const float* ssm_norm,
    const float* gate, float* out_buf, void* q8_out, float* kq_scales,
    float* alpha_beta, std::uint32_t layer_idx, std::size_t batch_size,
    std::size_t qkv_size, std::uint32_t num_key_heads, std::uint32_t num_heads,
    std::uint32_t key_dim, std::uint32_t val_dim, hipStream_t stream,
    QwenRecurrentStateStorage state_storage, void* fp16_out);

/// LaunchSSMConvRecurrence with FP32 recurrent state and the replay capture
/// as separate pointers (all null: no capture).
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
    hipStream_t stream);

/// PrepareBatchedGPUArgmax as a launcher: true once the batched argmax
/// kernels are loaded (nothing is launched).
[[nodiscard]] bool PrepareBatchedGPUArgmaxFlat(hipStream_t stream);

/// LaunchBatchedGPUArgmax with its scratch as pointer and length.
void LaunchBatchedGPUArgmaxFlat(const float* logits, std::uint32_t* out_tokens,
                                std::size_t batch_size, std::size_t vocab_size,
                                float* scratch, std::size_t scratch_len,
                                hipStream_t stream);

/// LaunchQwenWmmaAttention over the FP16 cache after the fused QK-norm kernel
/// wrote this chunk (no FP32 cache, no LSE, key_begin 0, skip_kv_write), with
/// the packed K/V workspaces as pointer and length; output rows before
/// `first_query` are skipped.
[[nodiscard]] bool LaunchQwenWmmaAttentionFlat(
    const float* q, const float* k, const float* v, const float* gate,
    const void* k_cache_f16, const void* v_cache_f16, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, float* packed_k,
    std::size_t packed_k_len, float* packed_v, std::size_t packed_v_len,
    std::size_t first_query, hipStream_t stream);

/// LaunchSSMConvRecurrenceBatch with FP32 recurrent state over flat lists:
/// sequence `i` has its conv and DeltaNet state, its replay capture (null
/// when absent) and rows `row_offset[i]..+rows[i]` of the shared inputs, as
/// batched decode and batched verification run them.
void LaunchSSMConvRecurrenceSequencesFlat(
    const float* qkv_in, const float* conv_weights, float* conv_out,
    const float* alpha_buf, const float* beta_buf, const float* ssm_a,
    const float* ssm_dt, const float* ssm_norm, const float* gate,
    float* out_buf, float* const* conv_state, const float* const* conv_in,
    float* const* deltanet_state, const float* const* deltanet_in,
    float* const* replay_qkv, float* const* replay_alpha,
    float* const* replay_beta, const std::uint32_t* const* replay_position,
    const std::uint32_t* const* replay_enabled, const std::uint32_t* row_offset,
    const std::uint32_t* rows, std::uint32_t count, std::uint32_t layer_idx,
    std::size_t qkv_size, std::uint32_t num_key_heads, std::uint32_t num_heads,
    std::uint32_t key_dim, std::uint32_t val_dim,
    std::size_t projection_row_stride, std::size_t inner_row_stride,
    hipStream_t stream);

/// LaunchSSMConvRecurrenceRows with FP32 recurrent state and the replay
/// capture as separate pointers: `rows` consecutive tokens of one sequence,
/// as the single-sequence verification (LaunchSSMConvRecurrenceBatch) and the
/// SSM replay run them.
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
    hipStream_t stream);

/// LaunchCausalDecodeAttention over the FP16 cache (no FP32 cache) with the
/// split-K scratch as pointer and length.
void LaunchCausalDecodeAttentionFlat(
    const float* q, const float* gate, const void* k_cache_f16,
    const void* v_cache_f16, float* out_context, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    float* scratch, std::size_t scratch_len, hipStream_t stream);

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

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_KERNELS_ADAPTERS_HPP_
