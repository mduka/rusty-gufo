#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>

#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/kernels/adapters.hpp"

namespace gufo::hip {

namespace {

/// The source and destination of one recurrent-state pass: a null `*_in`
/// updates `conv`/`recurrent` in place.
QwenRecurrentStateIo StateIo(float* conv, const float* conv_in,
                             void* recurrent, const void* recurrent_in) {
  return {conv_in != nullptr ? conv_in : conv, conv,
          recurrent_in != nullptr ? recurrent_in : recurrent, recurrent};
}

}  // namespace

void LaunchSSMConvRecurrenceFlat(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    const float* conv_in, float* conv_out, float* deltanet_state,
    const float* deltanet_in, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    float* replay_qkv, float* replay_alpha, float* replay_beta,
    const std::uint32_t* replay_position, const std::uint32_t* replay_enabled,
    hipStream_t stream) {
  LaunchSSMConvRecurrence(qkv_in, conv_weights,
                          StateIo(conv_state, conv_in, deltanet_state,
                                  deltanet_in),
                          conv_out, alpha_buf, beta_buf, ssm_a, ssm_dt,
                          ssm_norm, gate, out_buf, layer_idx, qkv_size,
                          num_key_heads, num_heads, key_dim, val_dim, stream,
                          SsmReplayCapture{.qkv = replay_qkv,
                                           .alpha = replay_alpha,
                                           .beta = replay_beta,
                                           .position = replay_position,
                                           .enabled = replay_enabled},
                          QwenRecurrentStateStorage::kFp32);
}

void LaunchBatchedSSMConvRecurrenceRowSplitFlat(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    const float* conv_in, float* conv_out, void* deltanet_state,
    const void* deltanet_in, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf, void* q8_out,
    float* kq_scales, float* alpha_beta, std::uint32_t layer_idx,
    std::size_t batch_size, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    hipStream_t stream, QwenRecurrentStateStorage state_storage,
    void* fp16_out) {
  LaunchBatchedSSMConvRecurrenceRowSplit(
      qkv_in, conv_weights,
      StateIo(conv_state, conv_in, deltanet_state, deltanet_in), conv_out,
      alpha_buf, beta_buf, ssm_a, ssm_dt, ssm_norm, gate, out_buf, q8_out,
      kq_scales, alpha_beta, layer_idx, batch_size, qkv_size, num_key_heads,
      num_heads, key_dim, val_dim, stream, state_storage, fp16_out);
}

bool PrepareBatchedGPUArgmaxFlat(hipStream_t stream) {
  (void)stream;
  return PrepareBatchedGPUArgmax() == hipSuccess;
}

void LaunchBatchedGPUArgmaxFlat(const float* logits, std::uint32_t* out_tokens,
                                std::size_t batch_size, std::size_t vocab_size,
                                float* scratch, std::size_t scratch_len,
                                hipStream_t stream) {
  LaunchBatchedGPUArgmax(logits, out_tokens, batch_size, vocab_size,
                         std::span<float>(scratch, scratch_len), stream);
}

bool LaunchQwenWmmaAttentionFlat(
    const float* q, const float* k, const float* v, const float* gate,
    const void* k_cache_f16, const void* v_cache_f16, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, float* packed_k,
    std::size_t packed_k_len, float* packed_v, std::size_t packed_v_len,
    std::size_t first_query, hipStream_t stream) {
  // With skip_kv_write the launcher only reads the FP16 planes.
  return LaunchQwenWmmaAttention(
      q, k, v, gate, nullptr, nullptr, const_cast<void*>(k_cache_f16),
      const_cast<void*>(v_cache_f16), out_context, layer_idx, start_pos,
      batch_size, max_context, num_heads, num_kv_heads, head_dim, stream,
      /*lse_out=*/nullptr, /*key_begin=*/0, /*skip_kv_write=*/true,
      std::span<float>(packed_k, packed_k_len),
      std::span<float>(packed_v, packed_v_len), first_query);
}

void LaunchSSMConvRecurrenceRowsFlat(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    const float* conv_in, float* conv_out, float* deltanet_state,
    const float* deltanet_in, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    std::uint32_t rows, std::size_t projection_row_stride,
    std::size_t inner_row_stride, float* replay_qkv, float* replay_alpha,
    float* replay_beta, const std::uint32_t* replay_position,
    const std::uint32_t* replay_enabled, hipStream_t stream) {
  LaunchSSMConvRecurrenceRows(
      qkv_in, conv_weights,
      StateIo(conv_state, conv_in, deltanet_state, deltanet_in), conv_out,
      alpha_buf, beta_buf, ssm_a, ssm_dt, ssm_norm, gate, out_buf, layer_idx,
      qkv_size,
      num_key_heads, num_heads, key_dim, val_dim, rows, projection_row_stride,
      inner_row_stride, stream,
      SsmReplayCapture{.qkv = replay_qkv,
                       .alpha = replay_alpha,
                       .beta = replay_beta,
                       .position = replay_position,
                       .enabled = replay_enabled},
      QwenRecurrentStateStorage::kFp32);
}

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
    hipStream_t stream) {
  std::array<SsmSequenceState, 8> sequences{};
  if (count > sequences.size())
    throw std::invalid_argument("SSM batch exceeds eight sequences");
  for (std::uint32_t i = 0; i < count; ++i) {
    sequences[i] = {
        .state = StateIo(conv_state[i], conv_in[i], deltanet_state[i],
                         deltanet_in[i]),
        .replay = {.qkv = replay_qkv[i],
                   .alpha = replay_alpha[i],
                   .beta = replay_beta[i],
                   .position = replay_position[i],
                   .enabled = replay_enabled[i]},
        .row_offset = row_offset[i],
        .rows = rows[i],
    };
  }
  LaunchSSMConvRecurrenceBatch(
      qkv_in, conv_weights, conv_out, alpha_buf, beta_buf, ssm_a, ssm_dt,
      ssm_norm, gate, out_buf, std::span(sequences).first(count), layer_idx,
      qkv_size, num_key_heads, num_heads, key_dim, val_dim,
      projection_row_stride, inner_row_stride, stream,
      QwenRecurrentStateStorage::kFp32);
}

void LaunchCausalDecodeAttentionFlat(
    const float* q, const float* gate, const void* k_cache_f16,
    const void* v_cache_f16, float* out_context, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    float* scratch, std::size_t scratch_len, hipStream_t stream) {
  // The launcher only reads the FP16 planes.
  LaunchCausalDecodeAttention(
      q, gate, nullptr, nullptr, const_cast<void*>(k_cache_f16),
      const_cast<void*>(v_cache_f16), out_context, layer_idx, start_pos,
      batch_size, max_context, num_heads, num_kv_heads, head_dim, stream,
      std::span<float>(scratch, scratch_len));
}

namespace {

/// The selector's numeric sampling controls over device penalty words.
kernels::DFlashSampling DraftSampling(float repeat_penalty,
                                      float frequency_penalty,
                                      float presence_penalty,
                                      std::uint32_t repeat_last_n,
                                      const std::uint32_t* penalties,
                                      std::uint32_t penalty_count,
                                      const std::uint32_t* history,
                                      std::uint32_t history_count) {
  static_assert(sizeof(sampling::TokenPenalty) == 3 * sizeof(std::uint32_t));
  return {.repeat_penalty = repeat_penalty,
          .frequency_penalty = frequency_penalty,
          .presence_penalty = presence_penalty,
          .repeat_last_n = repeat_last_n,
          .penalties =
              reinterpret_cast<const sampling::TokenPenalty*>(penalties),
          .penalty_count = penalties != nullptr ? penalty_count : 0U,
          .history = history,
          .history_count = history != nullptr ? history_count : 0U};
}

}  // namespace

void LaunchDFlashSelectorFlat(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    const std::uint32_t* anchor, std::uint32_t* tokens, float* confidences,
    float* partial_scores, std::uint32_t* partial_ids, std::uint32_t count,
    std::uint32_t vocab_size, std::uint32_t selector_rank,
    std::uint32_t selector_top_k, float repeat_penalty,
    float frequency_penalty, float presence_penalty,
    std::uint32_t repeat_last_n, const std::uint32_t* penalties,
    std::uint32_t penalty_count, const std::uint32_t* history,
    std::uint32_t history_count, hipStream_t stream) {
  if (hipMemcpyAsync(tokens, anchor, sizeof(std::uint32_t),
                     hipMemcpyDeviceToDevice, stream) != hipSuccess)
    return;  // The shim reports the error from hipGetLastError.
  const kernels::DFlashSelectorSequence sequence{
      .count = count,
      .temperature = 0.0F,
      .sampling = DraftSampling(repeat_penalty, frequency_penalty,
                                presence_penalty, repeat_last_n, penalties,
                                penalty_count, history, history_count)};
  kernels::LaunchDFlashSelectorBatch(
      logits, projected_hidden, predecessor_codebook_bf16,
      successor_codebook_bf16, tokens, confidences, partial_scores, partial_ids,
      /*uniforms=*/nullptr, /*candidate_ids=*/nullptr,
      /*candidate_probabilities=*/nullptr, std::span(&sequence, 1), vocab_size,
      selector_rank, selector_top_k, stream);
}

void LaunchGPUSamplingFlat(
    const float* logits, std::uint32_t* out_token, std::size_t vocab_size,
    float temperature, std::int32_t top_k, float top_p, float min_p,
    std::size_t min_keep, float repeat_penalty, float frequency_penalty,
    float presence_penalty, double uniform, const std::uint32_t* penalties,
    std::size_t penalty_count, float* adjusted_logits, float* sorted_logits,
    std::uint32_t* token_ids, std::uint32_t* sorted_token_ids,
    void* sort_storage, std::size_t sort_storage_bytes, hipStream_t stream) {
  static_assert(sizeof(sampling::TokenPenalty) == 3 * sizeof(std::uint32_t));
  GpuSamplingWorkspace workspace{};
  workspace.adjusted_logits = adjusted_logits;
  workspace.sorted_logits = sorted_logits;
  workspace.token_ids = token_ids;
  workspace.sorted_token_ids = sorted_token_ids;
  // Read only by the penalty kernel.
  workspace.penalties = reinterpret_cast<sampling::TokenPenalty*>(
      const_cast<std::uint32_t*>(penalties));
  workspace.sort_temp_storage = sort_storage;
  workspace.vocab_size = vocab_size;
  workspace.sort_temp_storage_bytes = GpuSamplingSortStorageBytes(workspace);
  if (workspace.sort_temp_storage_bytes > sort_storage_bytes)
    throw std::invalid_argument("GPU sampling sort storage is too small");
  workspace.penalty_capacity = penalty_count;
  LaunchGPUSamplingResident(logits, out_token, vocab_size,
                            GpuSamplingParameters{
                                .temperature = temperature,
                                .top_k = top_k,
                                .top_p = top_p,
                                .min_p = min_p,
                                .min_keep = min_keep,
                                .repeat_penalty = repeat_penalty,
                                .frequency_penalty = frequency_penalty,
                                .presence_penalty = presence_penalty,
                                .uniform = uniform,
                            },
                            penalty_count, &workspace, stream);
}

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
    std::size_t sort_storage_bytes, hipStream_t stream) {
  GpuSamplingWorkspace workspace{};
  workspace.adjusted_logits = adjusted_logits;
  workspace.sorted_logits = sorted_logits;
  workspace.token_ids = token_ids;
  workspace.sorted_token_ids = sorted_token_ids;
  // Read only by the kernels.
  workspace.penalties = reinterpret_cast<sampling::TokenPenalty*>(
      const_cast<std::uint32_t*>(penalties));
  workspace.draft_candidate_ids = const_cast<std::uint32_t*>(candidate_ids);
  workspace.draft_candidate_probabilities =
      const_cast<float*>(candidate_probabilities);
  workspace.sort_temp_storage = sort_storage;
  workspace.vocab_size = vocab_size;
  workspace.penalty_capacity = std::max(candidate_count, penalty_count);
  workspace.sort_temp_storage_bytes = GpuSamplingSortStorageBytes(workspace);
  if (workspace.sort_temp_storage_bytes > sort_storage_bytes)
    throw std::invalid_argument("GPU sampling sort storage is too small");
  LaunchGPUSpeculativeSamplingResident(
      logits, out_token, out_accepted, vocab_size,
      GpuSamplingParameters{
          .temperature = temperature,
          .top_k = top_k,
          .top_p = top_p,
          .min_p = min_p,
          .min_keep = min_keep,
          .repeat_penalty = repeat_penalty,
          .frequency_penalty = frequency_penalty,
          .presence_penalty = presence_penalty,
      },
      draft_token, draft_token_probability, candidate_count,
      acceptance_uniform, residual_uniform, penalty_count, &workspace, stream);
}

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
    std::uint32_t history_count, hipStream_t stream) {
  if (hipMemcpyAsync(tokens, anchor, sizeof(std::uint32_t),
                     hipMemcpyDeviceToDevice, stream) != hipSuccess)
    return;  // The shim reports the error from hipGetLastError.
  const kernels::DFlashSelectorSequence sequence{
      .count = count,
      .temperature = temperature,
      .sampling = DraftSampling(repeat_penalty, frequency_penalty,
                                presence_penalty, repeat_last_n, penalties,
                                penalty_count, history, history_count)};
  kernels::LaunchDFlashSelectorBatch(
      logits, projected_hidden, predecessor_codebook_bf16,
      successor_codebook_bf16, tokens, confidences, partial_scores, partial_ids,
      uniforms, candidate_ids, candidate_probabilities, std::span(&sequence, 1),
      vocab_size, selector_rank, selector_top_k, stream);
}

void LaunchGPUSortLogitsFlat(const float* logits, float* adjusted_logits,
                             float* sorted_logits, std::uint32_t* token_ids,
                             std::uint32_t* sorted_token_ids,
                             void* sort_storage,
                             std::size_t sort_storage_bytes,
                             std::size_t vocab_size, std::size_t keep,
                             hipStream_t stream) {
  GpuSamplingWorkspace workspace{};
  workspace.adjusted_logits = adjusted_logits;
  workspace.sorted_logits = sorted_logits;
  workspace.token_ids = token_ids;
  workspace.sorted_token_ids = sorted_token_ids;
  workspace.sort_temp_storage = sort_storage;
  workspace.vocab_size = vocab_size;
  workspace.sort_temp_storage_bytes = GpuSamplingSortStorageBytes(workspace);
  if (workspace.sort_temp_storage_bytes > sort_storage_bytes)
    throw std::invalid_argument("GPU candidate sort storage is too small");
  LaunchGPUSortLogits(logits, &workspace, stream, keep);
}

}  // namespace gufo::hip
