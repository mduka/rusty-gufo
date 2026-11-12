#include <algorithm>
#include <array>
#include <climits>
#include <span>
#include <stdexcept>

#include "src/models/qwen35moe/kernels/adapters.hpp"
#include "src/models/qwen35moe/kernels/vendor/family/kernels/dflash_kernels.hpp"
#include "src/models/qwen35moe/kernels/vendor/family/kernels/ops/attention.hpp"
#include "src/models/qwen35moe/kernels/vendor/family/kernels/ops/gemm.hpp"
#include "src/models/qwen35moe/kernels/vendor/family/kernels/ops/moe.hpp"
#include "src/models/qwen35moe/kernels/vendor/family/kernels/ops/routed_f16.hpp"
#include "src/models/qwen35moe/kernels/vendor/family/kernels/ops/ssm.hpp"
#include "src/models/qwen35moe/kernels/vendor/family/kernels/ops/token.hpp"
#include "src/models/qwen35moe/kernels/vendor/fnext/kernels/rocm/kernels.hpp"
#include "src/models/qwen35moe/kernels/vendor/fnext/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen35moe/kernels/vendor/fnext/mtp_sampling.hpp"

namespace gufo::models::qwen35moe::hip {

namespace {

namespace fnext = models::qwen38_flash_next;

/// The GEMV routes that would reach the tier's arena.
void RequireNoMmvq(core::GgmlType type, std::size_t m, std::size_t k,
                   models::qwen::QwenGemmMode mode) {
  const auto resolution = models::qwen::ResolveQwenGemmRoute(
      {.type = type, .batch_size = 1, .m = m, .k = k, .mode = mode});
  if (resolution.route == models::qwen::QwenGemmRoute::kHipQuantMmvq)
    throw std::invalid_argument(
        "the vendored vector GEMV route needs the tier's caller scratch");
}

/// The source and destination of one recurrent-state pass: a null `*_in`
/// updates `conv`/`recurrent` in place.
QwenRecurrentStateIo StateIo(float* conv, const float* conv_in, void* recurrent,
                             const void* recurrent_in) {
  return {conv_in != nullptr ? conv_in : conv, conv,
          recurrent_in != nullptr ? recurrent_in : recurrent, recurrent};
}

/// The replay capture of one pass; a ring of `capacity` slots per layer.
SsmReplayCapture Replay(float* qkv, float* alpha, float* beta,
                        const std::uint32_t* position,
                        const std::uint32_t* enabled, std::uint32_t capacity) {
  if (capacity == 0)
    throw std::invalid_argument("SSM replay ring without slots");
  return {.qkv = qkv,
          .alpha = alpha,
          .beta = beta,
          .position = position,
          .enabled = enabled,
          .capacity = capacity};
}

/// The selector's numeric sampling controls over device penalty words.
kernels::DFlashSampling DraftSampling(
    float repeat_penalty, float frequency_penalty, float presence_penalty,
    std::uint32_t repeat_last_n, const std::uint32_t* penalties,
    std::uint32_t penalty_count, const std::uint32_t* history,
    std::uint32_t history_count) {
  static_assert(sizeof(sampling::TokenPenalty) == 3 * sizeof(std::uint32_t));
  return {
      .repeat_penalty = repeat_penalty,
      .frequency_penalty = frequency_penalty,
      .presence_penalty = presence_penalty,
      .repeat_last_n = repeat_last_n,
      .penalties = reinterpret_cast<const sampling::TokenPenalty*>(penalties),
      .penalty_count = penalties != nullptr ? penalty_count : 0U,
      .history = history,
      .history_count = history != nullptr ? history_count : 0U};
}

/// Copies the checked anchor token into `tokens[0]` in stream order.
bool CopyAnchor(std::uint32_t* tokens, const std::uint32_t* anchor,
                hipStream_t stream) {
  // On failure the shim reports the error from hipGetLastError.
  return hipMemcpyAsync(tokens, anchor, sizeof(std::uint32_t),
                        hipMemcpyDeviceToDevice, stream) == hipSuccess;
}

/// The routed kernels' type IDs are the GGUF ones.
bool RoutedType(core::GgmlType type, moe_f16::WeightType* out) {
  switch (type) {
    case core::GgmlType::kQ5_1:
      *out = moe_f16::WeightType::kQ5_1;
      return true;
    case core::GgmlType::kQ8_0:
      *out = moe_f16::WeightType::kQ8_0;
      return true;
    case core::GgmlType::kQ4_K:
      *out = moe_f16::WeightType::kQ4_K;
      return true;
    case core::GgmlType::kQ5_K:
      *out = moe_f16::WeightType::kQ5_K;
      return true;
    case core::GgmlType::kQ6_K:
      *out = moe_f16::WeightType::kQ6_K;
      return true;
    default:
      return false;
  }
}

/// The tier takes `int` shapes.
bool FitsInt(std::uint32_t value) {
  return value <= static_cast<std::uint32_t>(INT_MAX);
}

/// Initialises the tier once per process (thread-safe static): the device
/// context and its pool, which calls from several threads then share.
bool MmqReady() {
  static const bool ready =
      q35_mmq_init(0) == 0 && q35_mmq_set_scratch(nullptr, 0) == 0;
  return ready;
}

/// Serves this thread's tier allocations from the caller's scratch for one
/// call.
class MmqScratch {
public:
  MmqScratch(void* base, std::size_t bytes)
      : valid(MmqReady() && base != nullptr &&
              reinterpret_cast<std::uintptr_t>(base) % 256 == 0 &&
              q35_mmq_set_scratch(base, bytes) == 0) {}
  ~MmqScratch() {
    if (valid) {
      (void)q35_mmq_set_scratch(nullptr, 0);
    }
  }
  MmqScratch(const MmqScratch&) = delete;
  MmqScratch& operator=(const MmqScratch&) = delete;

  const bool valid;
};

/// A routed vector projection's shape in the tier's `int`s.
bool MmqShape(std::uint32_t m, std::uint32_t k, std::uint32_t n_tokens,
              std::uint32_t n_experts, std::uint32_t n_used) {
  return FitsInt(m) && FitsInt(k) && FitsInt(n_tokens) && FitsInt(n_experts) &&
         FitsInt(n_used);
}

__global__ void MoeIdsUnpadKernel(const std::int32_t* __restrict__ ids,
                                  std::int32_t* __restrict__ out,
                                  std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const std::int32_t id = ids[i];
    out[i] = id < 0 ? 0 : id;
  }
}

}  // namespace

void LaunchGEMVChecked(const void* a, core::GgmlType a_type, const float* x,
                       float* y, std::size_t m, std::size_t k,
                       hipStream_t stream, models::qwen::QwenGemmMode mode) {
  RequireNoMmvq(a_type, m, k, mode);
  LaunchGEMV(a, a_type, x, y, m, k, stream, mode);
}

void LaunchGEMVRowsChecked(const void* a, core::GgmlType a_type, const float* x,
                           float* y, std::size_t rows, std::size_t m,
                           std::size_t k, hipStream_t stream,
                           models::qwen::QwenGemmMode mode) {
  RequireNoMmvq(a_type, m, k, mode);
  LaunchGEMVRows(a, a_type, x, y, rows, m, k, stream, mode);
}

void LaunchBatchedSSMConvRecurrenceRowSplitFlat(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    const float* conv_in, float* conv_out, void* deltanet_state,
    const void* deltanet_in, const float* alpha_buf, const float* beta_buf,
    const float* ssm_a, const float* ssm_dt, const float* ssm_norm,
    const float* gate, float* out_buf, void* q8_out, float* kq_scales,
    float* alpha_beta, std::uint32_t layer_idx, std::size_t batch_size,
    std::size_t qkv_size, std::uint32_t num_key_heads, std::uint32_t num_heads,
    std::uint32_t key_dim, std::uint32_t val_dim, hipStream_t stream,
    QwenRecurrentStateStorage state_storage, void* fp16_out, bool conv_fused) {
  LaunchBatchedSSMConvRecurrenceRowSplit(
      qkv_in, conv_weights,
      StateIo(conv_state, conv_in, deltanet_state, deltanet_in), conv_out,
      alpha_buf, beta_buf, ssm_a, ssm_dt, ssm_norm, gate, out_buf, q8_out,
      kq_scales, alpha_beta, layer_idx, batch_size, qkv_size, num_key_heads,
      num_heads, key_dim, val_dim, stream, state_storage, fp16_out, conv_fused);
}

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
    std::uint32_t replay_capacity, hipStream_t stream) {
  LaunchSSMConvRecurrence(
      qkv_in, conv_weights,
      StateIo(conv_state, conv_in, deltanet_state, deltanet_in), conv_out,
      alpha_buf, beta_buf, ssm_a, ssm_dt, ssm_norm, gate, out_buf, layer_idx,
      qkv_size, num_key_heads, num_heads, key_dim, val_dim, stream,
      Replay(replay_qkv, replay_alpha, replay_beta, replay_position,
             replay_enabled, replay_capacity),
      QwenRecurrentStateStorage::kFp32);
}

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
    std::uint32_t replay_capacity, hipStream_t stream) {
  LaunchSSMConvRecurrenceRows(
      qkv_in, conv_weights,
      StateIo(conv_state, conv_in, deltanet_state, deltanet_in), conv_out,
      alpha_buf, beta_buf, ssm_a, ssm_dt, ssm_norm, gate, out_buf, layer_idx,
      qkv_size, num_key_heads, num_heads, key_dim, val_dim, rows,
      projection_row_stride, inner_row_stride, stream,
      Replay(replay_qkv, replay_alpha, replay_beta, replay_position,
             replay_enabled, replay_capacity),
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
    const std::uint32_t* const* replay_enabled,
    const std::uint32_t* replay_capacity, const std::uint32_t* row_offset,
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
        .replay = Replay(replay_qkv[i], replay_alpha[i], replay_beta[i],
                         replay_position[i], replay_enabled[i],
                         replay_capacity != nullptr
                             ? replay_capacity[i]
                             : static_cast<std::uint32_t>(kSsmReplayCapacity)),
        .row_offset = row_offset[i],
        .rows = rows[i],
    };
  }
  LaunchSSMConvRecurrenceBatch(qkv_in, conv_weights, conv_out, alpha_buf,
                               beta_buf, ssm_a, ssm_dt, ssm_norm, gate, out_buf,
                               std::span(sequences).first(count), layer_idx,
                               qkv_size, num_key_heads, num_heads, key_dim,
                               val_dim, projection_row_stride, inner_row_stride,
                               stream, QwenRecurrentStateStorage::kFp32);
}

void LaunchCaptureBatchedSsmReplayFlat(
    const float* qkv, const float* alpha, const float* beta, float* replay_qkv,
    float* replay_alpha, float* replay_beta, const std::uint32_t* enabled,
    std::uint32_t replay_capacity, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::size_t qkv_size,
    std::size_t time_step_rank, hipStream_t stream) {
  LaunchCaptureBatchedSsmReplay(qkv, alpha, beta,
                                Replay(replay_qkv, replay_alpha, replay_beta,
                                       nullptr, enabled, replay_capacity),
                                layer_idx, start_pos, batch_size, qkv_size,
                                time_step_rank, stream);
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
    std::size_t first_query, bool kv_q8, hipStream_t stream) {
  // With skip_kv_write the launcher only reads the cache planes.
  return LaunchQwenWmmaAttention(
      q, k, v, gate, nullptr, nullptr, const_cast<void*>(k_cache_f16),
      const_cast<void*>(v_cache_f16), out_context, layer_idx, start_pos,
      batch_size, max_context, num_heads, num_kv_heads, head_dim, stream,
      /*lse_out=*/nullptr, /*key_begin=*/0, /*skip_kv_write=*/true,
      std::span<float>(packed_k, packed_k_len),
      std::span<float>(packed_v, packed_v_len), first_query, kv_q8);
}

void LaunchCausalDecodeAttentionFlat(
    const float* q, const float* gate, const void* k_cache_f16,
    const void* v_cache_f16, float* out_context, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    float* scratch, std::size_t scratch_len, bool kv_q8, hipStream_t stream) {
  // The launcher only reads the cache planes.
  LaunchCausalDecodeAttention(
      q, gate, nullptr, nullptr, const_cast<void*>(k_cache_f16),
      const_cast<void*>(v_cache_f16), out_context, layer_idx, start_pos,
      batch_size, max_context, num_heads, num_kv_heads, head_dim, stream,
      std::span<float>(scratch, scratch_len), kv_q8);
}

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
    hipStream_t stream) {
  if (!CopyAnchor(tokens, anchor, stream))
    return;
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
  if (!CopyAnchor(tokens, anchor, stream))
    return;
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
    hipStream_t stream) {
  if (prefix == 0)
    throw std::invalid_argument("draft vocabulary without a prefix");
  if (!CopyAnchor(tokens, anchor, stream))
    return;
  const kernels::DFlashSelectorSequence sequence{
      .count = count,
      .temperature = temperature,
      .sampling = DraftSampling(repeat_penalty, frequency_penalty,
                                presence_penalty, repeat_last_n, penalties,
                                penalty_count, history, history_count),
      .vocab = {.prefix = prefix,
                .logits = table_logits,
                .ids = table_ids,
                .rows = table_rows}};
  // The logits rows hold only the prefix columns.
  kernels::LaunchDFlashSelectorBatch(
      logits, projected_hidden, predecessor_codebook_bf16,
      successor_codebook_bf16, tokens, confidences, partial_scores, partial_ids,
      temperature > 0.0F ? uniforms : nullptr, candidate_ids,
      candidate_probabilities, std::span(&sequence, 1), vocab_size,
      selector_rank, selector_top_k, stream, prefix);
}

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
    hipStream_t stream) {
  if (prefix == 0 || logits_stride < prefix)
    throw std::invalid_argument("draft vocabulary without a prefix");
  if (!CopyAnchor(tokens, anchor, stream))
    return;
  const kernels::DFlashSelectorSequence sequence{
      .count = count,
      .temperature = temperature,
      .sampling = DraftSampling(repeat_penalty, frequency_penalty,
                                presence_penalty, repeat_last_n, penalties,
                                penalty_count, history, history_count),
      .vocab = {.prefix = prefix,
                .logits = table_logits,
                .ids = table_ids,
                .rows = table_rows}};
  kernels::LaunchDFlashSelectorBatch(
      logits, projected_hidden, predecessor_codebook_bf16,
      successor_codebook_bf16, tokens, confidences, partial_scores, partial_ids,
      temperature > 0.0F ? uniforms : nullptr, candidate_ids,
      candidate_probabilities, std::span(&sequence, 1), vocab_size,
      selector_rank, selector_top_k, stream, logits_stride);
}

void LaunchGPUSamplingFlat(const float* logits, std::uint32_t* out_token,
                           std::size_t vocab_size, float temperature,
                           std::int32_t top_k, float top_p, float min_p,
                           std::size_t min_keep, float repeat_penalty,
                           float frequency_penalty, float presence_penalty,
                           double uniform, const std::uint32_t* penalties,
                           std::size_t penalty_count, float* adjusted_logits,
                           float* sorted_logits, std::uint32_t* token_ids,
                           std::uint32_t* sorted_token_ids, void* sort_storage,
                           std::size_t sort_storage_bytes, hipStream_t stream) {
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
      draft_token, draft_token_probability, candidate_count, acceptance_uniform,
      residual_uniform, penalty_count, &workspace, stream);
}

void LaunchGPUSortLogitsFlat(const float* logits, float* adjusted_logits,
                             float* sorted_logits, std::uint32_t* token_ids,
                             std::uint32_t* sorted_token_ids,
                             void* sort_storage, std::size_t sort_storage_bytes,
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

void MtpTopCandidatesPacked(const float* logits, std::uint32_t* ids,
                            std::uint32_t* scratch_ids, std::uint32_t vocab,
                            hipStream_t stream) {
  fnext::rocm::MtpTopCandidates(
      logits, ids, scratch_ids,
      reinterpret_cast<float*>(ids + fnext::kMtpCandidates), vocab, stream);
}

void MoeIdsUnpad(const std::int32_t* ids, std::int32_t* out,
                 std::uint32_t count, hipStream_t stream) {
  if (count == 0)
    return;
  constexpr std::uint32_t kThreads = 256;
  MoeIdsUnpadKernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      ids, out, count);
}

void LaunchMoeGroupSlotsFlat(const std::int32_t* ids, std::uint32_t slots,
                             std::uint32_t n_experts,
                             std::int32_t* sorted_slots, std::int32_t* tiles,
                             std::int32_t* expert_bounds, hipStream_t stream) {
  LaunchMoeGroupSlots(ids, slots, n_experts,
                      {.sorted_slots = sorted_slots,
                       .tiles = tiles,
                       .expert_bounds = expert_bounds},
                      stream);
}

void LaunchMoeGroupedBf16GemmFlat(
    const void* w, const float* x, float* out, std::size_t m, std::size_t k,
    std::uint32_t slots, std::uint32_t n_experts, std::uint32_t x_token_divisor,
    const std::int32_t* ids, const std::int32_t* sorted_slots,
    const std::int32_t* tiles, const std::int32_t* expert_bounds,
    hipStream_t stream) {
  (void)ids;
  if (!IsMoeGroupedBf16GemmSupported(m, k, n_experts))
    throw std::invalid_argument("unsupported grouped BF16 expert GEMM shape");
  // The launch only reads the grouping.
  LaunchMoeGroupedBf16Gemm(
      w, x, out, m, k, slots, n_experts, x_token_divisor,
      {.sorted_slots = const_cast<std::int32_t*>(sorted_slots),
       .tiles = const_cast<std::int32_t*>(tiles),
       .expert_bounds = const_cast<std::int32_t*>(expert_bounds)},
      stream);
}

bool RoutedF16GemmFlat(const void* w, core::GgmlType type, const void* x,
                       const std::int32_t* tiles, std::uint32_t n_tiles,
                       std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                       const std::int32_t* rows_in,
                       const std::int32_t* rows_out, const float* swiglu_gate,
                       float* out, void* out_half, std::size_t m, std::size_t k,
                       hipStream_t stream) {
  moe_f16::WeightType routed{};
  return RoutedType(type, &routed) &&
         moe_f16::RoutedF16Gemm(w, routed, static_cast<const __half*>(x), tiles,
                                n_tiles, tile_rows, pad_bounds, rows_in,
                                rows_out, swiglu_gate, out,
                                static_cast<__half*>(out_half), m, k, stream);
}

bool RoutedGatedF16GemmFlat(const void* gate, const void* up,
                            core::GgmlType type, const void* x,
                            const std::int32_t* tiles, std::uint32_t n_tiles,
                            std::uint32_t tile_rows,
                            const std::int32_t* pad_bounds,
                            const std::int32_t* rows_in,
                            const std::int32_t* rows_out, void* out,
                            std::size_t m, std::size_t k, hipStream_t stream) {
  moe_f16::WeightType routed{};
  return RoutedType(type, &routed) &&
         moe_f16::RoutedGatedF16Supported(routed) &&
         moe_f16::RoutedGatedF16Gemm(
             gate, up, routed, static_cast<const __half*>(x), tiles, n_tiles,
             tile_rows, pad_bounds, rows_in, rows_out,
             static_cast<__half*>(out), m, k, stream);
}

void MoeEpilogueVec4F16Flat(const void* expert_out, const float* weights,
                            const float* shared_out, const float* gate,
                            std::uint32_t gate_stride, float* out,
                            std::uint32_t n_tokens, std::uint32_t k,
                            std::uint32_t dim, hipStream_t stream) {
  fnext::rocm::MoeEpilogueVec4F16(static_cast<const __half*>(expert_out),
                                  weights, shared_out, gate, gate_stride, out,
                                  n_tokens, k, dim, stream);
}

bool MmqPrepare(hipStream_t stream) {
  (void)stream;
  return MmqReady();
}

void MmqReleaseStream(hipStream_t stream) {
  q35_mmq_release_stream(stream);
}

bool MmqMoeVec(core::GgmlType type, const void* w, const float* x,
               const std::int32_t* ids, float* out, std::uint32_t m,
               std::uint32_t k, std::uint32_t n_tokens, std::uint32_t n_experts,
               std::uint32_t n_used, void* scratch, std::size_t scratch_bytes,
               hipStream_t stream) {
  if (!MmqShape(m, k, n_tokens, n_experts, n_used))
    return false;
  const MmqScratch scope(scratch, scratch_bytes);
  return scope.valid &&
         q35_mmq_moe_vec(static_cast<int>(type), w, x, ids, out,
                         static_cast<int>(m), static_cast<int>(k),
                         static_cast<int>(n_tokens),
                         static_cast<int>(n_experts), static_cast<int>(n_used),
                         stream, nullptr, nullptr) == 0;
}

bool MmqMoeGatedVec(core::GgmlType type, const void* gate, const void* up,
                    const float* x, const std::int32_t* ids, float* out,
                    std::uint32_t m, std::uint32_t k, std::uint32_t n_tokens,
                    std::uint32_t n_experts, std::uint32_t n_used,
                    void* scratch, std::size_t scratch_bytes,
                    hipStream_t stream) {
  if (!MmqShape(m, k, n_tokens, n_experts, n_used))
    return false;
  const MmqScratch scope(scratch, scratch_bytes);
  return scope.valid &&
         q35_mmq_moe_gated_vec(static_cast<int>(type), gate, up, x, ids, out,
                               static_cast<int>(m), static_cast<int>(k),
                               static_cast<int>(n_tokens),
                               static_cast<int>(n_experts),
                               static_cast<int>(n_used), stream) == 0;
}

}  // namespace gufo::models::qwen35moe::hip
