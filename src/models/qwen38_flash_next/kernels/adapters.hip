#include <algorithm>
#include <climits>
#include <iterator>

#include "src/models/qwen38_flash_next/kernels/adapters.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

constexpr std::uint32_t kRollbackCapacity =
    static_cast<std::uint32_t>(std::size(RollbackRows{}.rows));

/// The tier takes `int` shapes.
bool FitsInt(std::uint32_t value) {
  return value <= static_cast<std::uint32_t>(INT_MAX);
}

/// Initialises the tier once per process (thread-safe static): the device
/// context and its pool, which calls from several threads then share.
bool MmqReady() {
  static const bool ready =
      qfn_mmq_init(0) == 0 && qfn_mmq_set_scratch(nullptr, 0) == 0;
  return ready;
}

/// Whether the tier's Q8_1 rows of width k carry no padding, so a
/// producer's contiguous element order is the layout (the C++ executor's
/// UnpaddedQ8).
bool UnpaddedQ8(std::uint32_t k) {
  return k % 32 == 0 && FitsInt(k) &&
         qfn_mmq_q8_1_bytes(1, static_cast<int>(k)) == k / 32 * 36;
}

/// A routed projection's shape in the tier's `int`s.
struct Shape {
  bool valid;
  int m, k, tokens, experts, used;
  Shape(std::uint32_t m, std::uint32_t k, std::uint32_t tokens,
        std::uint32_t experts, std::uint32_t used)
      : valid(FitsInt(m) && FitsInt(k) && FitsInt(tokens) && FitsInt(experts) &&
              FitsInt(used)),
        m(static_cast<int>(m)),
        k(static_cast<int>(k)),
        tokens(static_cast<int>(tokens)),
        experts(static_cast<int>(experts)),
        used(static_cast<int>(used)) {}
};

/// Serves this thread's tier allocations from the caller's scratch for one
/// call.
class Scratch {
public:
  Scratch(void* base, std::size_t bytes)
      : valid(MmqReady() && base != nullptr &&
              reinterpret_cast<std::uintptr_t>(base) % 256 == 0 &&
              qfn_mmq_set_scratch(base, bytes) == 0) {}
  ~Scratch() {
    if (valid) {
      (void)qfn_mmq_set_scratch(nullptr, 0);
    }
  }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;

  const bool valid;
};

}  // namespace

void NarrowToHalf(const float* x, __half* out, std::size_t count,
                  hipStream_t stream) {
  NarrowActivations(x, out, false, count, stream);
}

bool PleConvRows(const float* in, const float* w, float* history,
                 float* history_scratch, float* out, float* const* snapshots,
                 std::uint32_t snapshot_count, std::uint32_t n_tokens,
                 std::uint32_t channels, std::uint32_t kernel,
                 std::uint32_t dilation, hipStream_t stream) {
  // Rows are written for every proper prefix: n_tokens - 1 of them.
  if (snapshot_count > kRollbackCapacity ||
      (snapshot_count != 0 && snapshot_count + 1 < n_tokens)) {
    return false;
  }
  RollbackRows rows;
  for (std::uint32_t i = 0; i < snapshot_count; ++i) {
    rows.rows[i] = snapshots[i];
  }
  PleConv(in, w, history, history_scratch, out, rows, n_tokens, channels,
          kernel, dilation, stream);
  return true;
}

bool GatedDeltaNetRows(
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
    std::uint32_t checkpoint_tokens, hipStream_t stream) {
  const bool snapshots = state_first != nullptr;
  const bool checkpoint = checkpoint_state != nullptr;
  if (checkpoint != (checkpoint_history != nullptr) ||
      checkpoint != (checkpoint_tokens != 0) ||
      (checkpoint && (snapshots || checkpoint_tokens > n_tokens))) {
    return false;
  }
  if (snapshots != (conv_first != nullptr) || (!snapshots && rest != 0) ||
      rest + 1 > kRollbackCapacity ||
      (snapshots && (convolved || rest + 2 < n_tokens))) {
    return false;
  }
  if (static_cast<int>(out != nullptr) + static_cast<int>(out_q8 != nullptr) +
          static_cast<int>(out_half != nullptr) !=
      1) {
    return false;
  }
  if (out_rows != nullptr && (out == nullptr || (v_heads * 128) % 512 != 0)) {
    return false;
  }
  if (out_half_stride != 0 &&
      (out_half == nullptr || out_half_stride < v_heads * 128)) {
    return false;
  }
  RollbackRows state_rows;
  RollbackRows conv_rows;
  if (snapshots) {
    state_rows.rows[0] = state_first;
    conv_rows.rows[0] = conv_first;
    for (std::uint32_t i = 0; i < rest; ++i) {
      state_rows.rows[i + 1] = state_rest[i];
      conv_rows.rows[i + 1] = conv_rest[i];
    }
  }
  GatedDeltaNet(qkv, qkv_stride, z, z_stride, alpha_beta, conv_w, a, dt, norm_w,
                conv_state, conv_scratch, qn, kn, raw, state, out, out_q8,
                state_rows, conv_rows, n_tokens, k_heads, v_heads, 128, kernel,
                row_split, convolved, eps, stream, out_half,
                GdnCheckpoint{checkpoint_state, checkpoint_history,
                              checkpoint_tokens},
                out_half_stride, out_rows);
  return true;
}

bool HcMixEpilogueQ8_1(const float* xn, const float* gate,
                       const float* inject_w, float* mixed, float* inject,
                       void* rows, std::uint32_t n_tokens, std::uint32_t hidden,
                       std::uint32_t streams, hipStream_t stream) {
  if (!UnpaddedQ8(hidden)) {
    return false;
  }
  HcMixEpilogue(xn, gate, inject_w, mixed, inject, n_tokens, hidden, streams,
                stream, rows);
  return true;
}

bool HcCombineQ8_1Rows(float* res, const float* block_out, const float* inject,
                       std::uint32_t inject_parts, const float* gamma, float* xn,
                       void* xn_rows, std::uint32_t n_tokens,
                       std::uint32_t hidden, std::uint32_t streams, float eps,
                       hipStream_t stream) {
  if (!UnpaddedQ8(hidden * streams)) {
    return false;
  }
  return HcCombineQ8_1(res, block_out, inject, inject_parts, gamma, xn,
                       xn_rows, n_tokens, hidden, streams, eps, stream);
}

void HcCombineF16Plain(float* res, const float* block_out, const float* inject,
                       std::uint32_t inject_parts, const float* gamma,
                       __half* xn, void* xn_q8, std::uint32_t n_tokens,
                       std::uint32_t hidden, std::uint32_t streams, float eps,
                       hipStream_t stream) {
  // Without `defer_w` the result only reports that nothing was deferred.
  (void)HcCombineF16(res, block_out, inject, inject_parts, gamma, xn, xn_q8,
                     n_tokens, hidden, streams, eps, stream);
}

bool HcCombineMoeF16Plain(float* res, const __half* expert_out,
                          const float* weights, const float* shared_out,
                          const float* gate, std::uint32_t gate_stride,
                          std::uint32_t used, const float* inject,
                          std::uint32_t inject_parts, const float* gamma,
                          __half* xn, void* xn_q8, std::uint32_t n_tokens,
                          std::uint32_t hidden, std::uint32_t streams,
                          float eps, hipStream_t stream) {
  return HcCombineMoeF16(res, expert_out, weights, shared_out, gate,
                         gate_stride, used, inject, inject_parts, gamma, xn,
                         xn_q8, n_tokens, hidden, streams, eps, stream);
}

bool SiluScaleQ8_1Rows(float* x, float scale, void* rows, std::uint32_t n,
                       std::uint32_t k, hipStream_t stream) {
  if (!FitsInt(k)) {
    return false;
  }
  // Rows padded to the tier's width, as the C++ executor's lo_q8_cols.
  const auto padded = static_cast<std::uint32_t>(
      qfn_mmq_q8_1_bytes(1, static_cast<int>(k)) / 36 * 32);
  return SiluScaleQ8_1(x, scale, n, k, padded, rows, stream);
}

bool SigmoidMulQ8_1Rows(float* x, const float* g, void* rows, std::uint32_t n,
                        std::uint32_t k, hipStream_t stream) {
  if (!UnpaddedQ8(k)) {
    return false;
  }
  return SigmoidMulQ8_1(x, g, static_cast<std::size_t>(n) * k, rows, stream);
}

bool HcCombineF16Deferred(float* res, const float* block_out, float* inject,
                          std::uint32_t inject_parts, const float* gamma,
                          __half* xn, void* xn_q8, std::uint32_t n_tokens,
                          std::uint32_t hidden, std::uint32_t streams,
                          float eps, float* defer_w, const float* inject_w,
                          hipStream_t stream) {
  // Launches nothing unless the four-stream kernel defers; the plain
  // combine is hc_combine_f16.
  if (streams != 4 || hidden % 128 != 0 || hidden > 2560 ||
      defer_w == nullptr || (inject_w != nullptr && gamma == nullptr)) {
    return false;
  }
  return HcCombineF16(res, block_out, inject, inject_parts, gamma, xn, xn_q8,
                      n_tokens, hidden, streams, eps, stream, defer_w,
                      inject_w != nullptr ? inject : nullptr, inject_w);
}

bool HcCombineMoeF16Prior(float* res, const __half* expert_out,
                          const float* weights, const float* shared_out,
                          const float* gate, std::uint32_t gate_stride,
                          std::uint32_t used, float* inject,
                          std::uint32_t inject_parts, const float* gamma,
                          __half* xn, void* xn_q8, std::uint32_t n_tokens,
                          std::uint32_t hidden, std::uint32_t streams,
                          float eps, const float* prior_out,
                          const float* prior_w, const float* inject_w,
                          hipStream_t stream) {
  return HcCombineMoeF16(res, expert_out, weights, shared_out, gate,
                         gate_stride, used, inject, inject_parts, gamma, xn,
                         xn_q8, n_tokens, hidden, streams, eps, stream,
                         prior_out, prior_w,
                         inject_w != nullptr ? inject : nullptr, inject_w);
}

bool DenseF16GemmStrided(const void* w, const __half* x, float* out,
                         std::size_t batch, std::size_t m, std::size_t k,
                         std::size_t input_stride, hipStream_t stream) {
  return DenseF16Gemm(w, x, out, batch, m, k, stream, input_stride);
}

bool GatedDeltaNetBatchRows(
    void* items_slot, const float* const* qkv, std::uint32_t qkv_stride, const float* const* z,
    std::uint32_t z_stride, const float* const* alpha_beta, const float* conv_w,
    const float* a, const float* dt, const float* norm_w,
    float* const* conv_state, float* const* conv_scratch, float* const* qn,
    float* const* kn, float* const* raw, float* const* state, float* const* out,
    float* const* state_0, float* const* state_1, float* const* state_2,
    float* const* state_3, float* const* state_4, float* const* state_5,
    float* const* state_6, float* const* conv_0, float* const* conv_1,
    float* const* conv_2, float* const* conv_3, float* const* conv_4,
    float* const* conv_5, float* const* conv_6, const std::uint32_t* n_tokens,
    std::uint32_t count, std::uint32_t k_heads, std::uint32_t v_heads,
    float eps, hipStream_t stream) {
  // The batch kernels keep up to eight tokens per thread in registers.
  constexpr std::uint32_t kMaxTokens = 8;
  if (count == 0 || count > 8) {
    return false;
  }
  float* const* state_rows[] = {state_0, state_1, state_2, state_3,
                                state_4, state_5, state_6};
  float* const* conv_rows[] = {conv_0, conv_1, conv_2, conv_3,
                               conv_4, conv_5, conv_6};
  static_assert(std::size(state_rows) == kRollbackCapacity);
  GdnBatchItem items[8]{};
  std::uint32_t max_tokens = 0;
  for (std::uint32_t i = 0; i < count; ++i) {
    const std::uint32_t n = n_tokens[i];
    if (n == 0 || n > kMaxTokens) {
      return false;
    }
    const bool rows = state_rows[0][i] != nullptr;
    for (std::uint32_t r = 0; r < kRollbackCapacity; ++r) {
      const bool state_row = state_rows[r][i] != nullptr;
      const bool conv_row = conv_rows[r][i] != nullptr;
      // Rows only with the full state, both lists alike, and every proper
      // prefix covered.
      if (state_row != conv_row || (state_row && !rows) ||
          (rows && r + 1 < n && !state_row)) {
        return false;
      }
    }
    auto& item = items[i];
    item = {qkv[i],
            z[i],
            alpha_beta[i],
            conv_state[i],
            conv_scratch[i],
            qn[i],
            kn[i],
            raw[i],
            state[i],
            out[i],
            {},
            {},
            n};
    for (std::uint32_t r = 0; r < kRollbackCapacity; ++r) {
      item.state_snapshots.rows[r] = state_rows[r][i];
      item.conv_snapshots.rows[r] = conv_rows[r][i];
    }
    max_tokens = std::max(max_tokens, n);
  }
  // The host addresses the slot at its device address (checked by the
  // launcher) and no queued work still reads it (the contract), so the
  // kernels read what the host writes here, as from the C++ executor's
  // batch_gdn_host_.
  static_assert(sizeof(GdnBatchItem) == 200);
  auto* slot = static_cast<GdnBatchItem*>(items_slot);
  std::copy_n(items, count, slot);
  return GatedDeltaNetBatch(slot, count, max_tokens, (1U << count) - 1,
                            qkv_stride, z_stride, conv_w, a, dt, norm_w,
                            k_heads, v_heads, eps, stream);
}

bool RestoreGdnStateRows(float* state, const float* first,
                         const float* const* rest, std::uint32_t rest_count,
                         std::uint32_t keep, std::uint32_t k_heads,
                         std::uint32_t v_heads, hipStream_t stream) {
  if (keep == 0 || keep > rest_count + 1 ||
      rest_count + 1 > kRollbackCapacity) {
    return false;
  }
  // The kernel only reads the rows.
  RollbackRows rows;
  rows.rows[0] = const_cast<float*>(first);
  for (std::uint32_t i = 0; i < rest_count; ++i) {
    rows.rows[i + 1] = const_cast<float*>(rest[i]);
  }
  RestoreGdnState(state, rows, keep, k_heads, v_heads, stream);
  return true;
}

void ArgmaxFlat(const float* logits, void* scratch, std::int32_t* out,
                std::uint32_t n_tokens, std::uint32_t vocab,
                hipStream_t stream) {
  Argmax(logits, static_cast<ArgmaxCandidate*>(scratch), out, n_tokens, vocab,
         stream);
}

namespace {

/// The kernels' penalty rows: the triples and each row's end (row 0 starts
/// at 0).
GreedyPenaltyRows PenaltyRows(const std::uint32_t* penalties,
                              const std::uint32_t (&ends)[7]) {
  static_assert(sizeof(sampling::TokenPenalty) == 3 * sizeof(std::uint32_t));
  GreedyPenaltyRows rows{};
  rows.penalties = reinterpret_cast<const sampling::TokenPenalty*>(penalties);
  std::copy(std::begin(ends), std::end(ends), rows.offsets + 1);
  return rows;
}

}  // namespace

bool PenalizedArgmaxFlat(const float* logits, const std::uint32_t* penalties,
                         std::uint32_t end_0, std::uint32_t end_1,
                         std::uint32_t end_2, std::uint32_t end_3,
                         std::uint32_t end_4, std::uint32_t end_5,
                         std::uint32_t end_6, float repeat, float frequency,
                         float presence, void* partial, void* out,
                         std::uint32_t rows, std::uint32_t vocab,
                         hipStream_t stream) {
  if (rows == 0 || rows > 7 || vocab == 0) {
    return false;
  }
  const std::uint32_t ends[7] = {end_0, end_1, end_2, end_3,
                                 end_4, end_5, end_6};
  PenalizedArgmax(logits, PenaltyRows(penalties, ends), repeat, frequency,
                  presence, static_cast<PenaltyArgmaxCandidate*>(partial),
                  static_cast<ArgmaxCandidate*>(out), rows, vocab, stream);
  return true;
}

void GatherArgmaxCandidatesFlat(const float* logits, const std::uint32_t* ids,
                                void* out, std::uint32_t rows,
                                std::uint32_t vocab, hipStream_t stream) {
  GatherArgmaxCandidates(logits, ids, static_cast<ArgmaxCandidate*>(out), rows,
                         vocab, stream);
}

bool VerificationCandidatesFlat(
    const float* logits, const std::uint32_t* penalties, std::uint32_t end_0,
    std::uint32_t end_1, std::uint32_t end_2, std::uint32_t end_3,
    std::uint32_t end_4, std::uint32_t end_5, std::uint32_t end_6,
    std::uint32_t* ids, std::uint32_t* scratch_ids, void* out,
    std::uint32_t rows, std::uint32_t vocab, hipStream_t stream) {
  if (rows == 0 || rows > 7 || vocab == 0) {
    return false;
  }
  const std::uint32_t ends[7] = {end_0, end_1, end_2, end_3,
                                 end_4, end_5, end_6};
  VerificationCandidates(logits, PenaltyRows(penalties, ends), ids,
                         scratch_ids,
                         static_cast<sampling::LogitCandidate*>(out), rows,
                         vocab, stream);
  return true;
}

void MtpTopCandidatesPacked(const float* logits, std::uint32_t* ids,
                            std::uint32_t* scratch_ids, std::uint32_t vocab,
                            hipStream_t stream) {
  MtpTopCandidates(logits, ids, scratch_ids,
                   reinterpret_cast<float*>(ids + kMtpCandidates), vocab,
                   stream);
}

bool PrepareDecodeKernels(hipStream_t stream) {
  (void)stream;
  return MmqReady() && PrepareW8A8GemmWave64() == hipSuccess;
}

void MmqReleaseStream(hipStream_t stream) {
  qfn_mmq_release_stream(stream);
}

bool MmqQuantizeQ8_1(const float* x, void* x_q8, std::uint32_t n,
                     std::uint32_t k, hipStream_t stream) {
  return MmqReady() && FitsInt(n) && FitsInt(k) &&
         qfn_mmq_quantize_q8_1(x, x_q8, static_cast<int>(n),
                               static_cast<int>(k), stream) == 0;
}

bool MmqQ8DenseVec(const void* w, const void* x_q8, float* out, std::uint32_t m,
                   std::uint32_t n, std::uint32_t k, hipStream_t stream) {
  return MmqReady() && FitsInt(m) && FitsInt(n) && FitsInt(k) &&
         qfn_mmq_q8_0_dense_vec_preq(w, nullptr, x_q8, out, static_cast<int>(m),
                                     static_cast<int>(n), static_cast<int>(k),
                                     stream) == 0;
}

bool MmqQ8GatedDenseVec(const void* up, const void* gate, const void* x_q8,
                        float* out, std::uint32_t m, std::uint32_t n,
                        std::uint32_t k, hipStream_t stream) {
  return MmqReady() && FitsInt(m) && FitsInt(n) && FitsInt(k) &&
         qfn_mmq_q8_0_dense_vec_preq(up, gate, x_q8, out, static_cast<int>(m),
                                     static_cast<int>(n), static_cast<int>(k),
                                     stream) == 0;
}

bool MmqMoeVec(WeightType type, const void* w, const float* x,
               const std::int32_t* ids, float* out, std::uint32_t m,
               std::uint32_t k, std::uint32_t n_tokens, std::uint32_t n_experts,
               std::uint32_t n_used, void* scratch, std::size_t scratch_bytes,
               hipStream_t stream) {
  const Shape shape(m, k, n_tokens, n_experts, n_used);
  const Scratch scope(scratch, scratch_bytes);
  return shape.valid && scope.valid &&
         qfn_mmq_moe_vec(static_cast<int>(type), w, x, ids, out, shape.m,
                         shape.k, shape.tokens, shape.experts, shape.used,
                         stream) == 0;
}

bool MmqMoeVecPair(WeightType type, const void* w_a, const void* w_b,
                   const float* x, const std::int32_t* ids, float* out_a,
                   float* out_b, std::uint32_t m, std::uint32_t k,
                   std::uint32_t n_tokens, std::uint32_t n_experts,
                   std::uint32_t n_used, void* scratch,
                   std::size_t scratch_bytes, hipStream_t stream) {
  const Shape shape(m, k, n_tokens, n_experts, n_used);
  const Scratch scope(scratch, scratch_bytes);
  return shape.valid && scope.valid &&
         qfn_mmq_moe_vec(static_cast<int>(type), w_a, x, ids, out_a, shape.m,
                         shape.k, shape.tokens, shape.experts, shape.used,
                         stream, w_b, out_b) == 0;
}

bool MmqMoeGatedVec(WeightType type, const void* gate, const void* up,
                    const float* x, const std::int32_t* ids, float* out,
                    std::uint32_t m, std::uint32_t k, std::uint32_t n_tokens,
                    std::uint32_t n_experts, std::uint32_t n_used,
                    void* scratch, std::size_t scratch_bytes,
                    hipStream_t stream) {
  const Shape shape(m, k, n_tokens, n_experts, n_used);
  const Scratch scope(scratch, scratch_bytes);
  return shape.valid && scope.valid &&
         qfn_mmq_moe_gated_vec(static_cast<int>(type), gate, up, x, ids, out,
                               shape.m, shape.k, shape.tokens, shape.experts,
                               shape.used, stream) == 0;
}

bool MmqQ8MoeGatedVec(WeightType type, const void* gate, const void* up,
                      const void* x_q8, const std::int32_t* ids, float* out,
                      std::uint32_t m, std::uint32_t k, std::uint32_t n_tokens,
                      std::uint32_t n_experts, std::uint32_t n_used,
                      void* scratch, std::size_t scratch_bytes,
                      hipStream_t stream) {
  const Shape shape(m, k, n_tokens, n_experts, n_used);
  const Scratch scope(scratch, scratch_bytes);
  return shape.valid && scope.valid &&
         qfn_mmq_moe_gated_vec_preq(static_cast<int>(type), gate, up, x_q8,
                                    ids, out, shape.m, shape.k, shape.tokens,
                                    shape.experts, shape.used, stream) == 0;
}

bool MmqMoeTiled(WeightType type, const void* w, const float* x,
                 const std::int32_t* ids, float* out, std::uint32_t m,
                 std::uint32_t k, std::uint32_t n_tokens,
                 std::uint32_t n_experts, std::uint32_t n_used,
                 std::uint32_t max_rows, std::uint32_t tile_cols, void* scratch,
                 std::size_t scratch_bytes, hipStream_t stream) {
  const Shape shape(m, k, n_tokens, n_experts, n_used);
  if (!shape.valid || !FitsInt(max_rows) || !FitsInt(tile_cols)) {
    return false;
  }
  const Scratch scope(scratch, scratch_bytes);
  if (!scope.valid) {
    return false;
  }
  qfn_mmq_set_routed_max_expert_rows(static_cast<int>(max_rows));
  qfn_mmq_set_routed_tile_cols(static_cast<int>(tile_cols));
  int status = -1;
  switch (type) {
    case WeightType::kQ4_K:
      status =
          qfn_mmq_q4_K_moe_raw(w, x, ids, out, shape.m, shape.k, shape.tokens,
                               shape.experts, shape.used, stream);
      break;
    case WeightType::kQ5_K:
      status =
          qfn_mmq_q5_K_moe_raw(w, x, ids, out, shape.m, shape.k, shape.tokens,
                               shape.experts, shape.used, stream);
      break;
    case WeightType::kQ5_1:
      status =
          qfn_mmq_q5_1_moe_raw(w, x, ids, out, shape.m, shape.k, shape.tokens,
                               shape.experts, shape.used, stream);
      break;
    case WeightType::kQ8_0:
      status =
          qfn_mmq_q8_0_moe_raw(w, x, ids, out, shape.m, shape.k, shape.tokens,
                               shape.experts, shape.used, stream);
      break;
    default:
      break;
  }
  return status == 0;
}

bool MmqQ4KMoePairTiled(const void* w_a, const void* w_b, const float* x,
                        const std::int32_t* ids, float* out_a, float* out_b,
                        std::uint32_t m, std::uint32_t k,
                        std::uint32_t n_tokens, std::uint32_t n_experts,
                        std::uint32_t n_used, std::uint32_t max_rows,
                        std::uint32_t tile_cols, void* scratch,
                        std::size_t scratch_bytes, hipStream_t stream) {
  const Shape shape(m, k, n_tokens, n_experts, n_used);
  if (!shape.valid || !FitsInt(max_rows) || !FitsInt(tile_cols)) {
    return false;
  }
  const Scratch scope(scratch, scratch_bytes);
  if (!scope.valid) {
    return false;
  }
  qfn_mmq_set_routed_max_expert_rows(static_cast<int>(max_rows));
  qfn_mmq_set_routed_tile_cols(static_cast<int>(tile_cols));
  return qfn_mmq_q4_K_moe_pair_unique(w_a, w_b, x, ids, out_a, out_b, shape.m,
                                      shape.k, shape.tokens, shape.experts,
                                      shape.used, stream) == 0;
}

namespace {

YarnRope Yarn(std::uint32_t enabled, float low, float high, float inv_factor,
              float mscale) {
  return {enabled, low, high, inv_factor, mscale};
}

}  // namespace

bool AttentionF16GemmYarn(const void* weights, const __half* input,
                          const float* q_gamma, const float* k_gamma,
                          float* query, float* gate, __half* keys,
                          __half* values, std::uint32_t n_tokens,
                          const std::uint32_t* position, float theta,
                          float eps, hipStream_t stream,
                          const qwen::vision::DeviceRope* rope,
                          std::uint32_t yarn_enabled, float yarn_low,
                          float yarn_high, float yarn_inv_factor,
                          float yarn_mscale) {
  return AttentionF16Gemm(
      weights, input, q_gamma, k_gamma, query, gate, keys, values, n_tokens,
      position, theta, eps, stream, rope,
      Yarn(yarn_enabled, yarn_low, yarn_high, yarn_inv_factor, yarn_mscale));
}

bool PrepareAttentionYarn(
    const float* packed, std::uint32_t stride, const float* q_gamma,
    const float* k_gamma, float* q, float* gate, __half* k_cache,
    __half* v_cache, std::uint32_t n_tokens, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t d, std::uint32_t rotary_dim,
    const std::uint32_t* start_pos, float theta, float eps, hipStream_t stream,
    const qwen::vision::DeviceRope* rope, bool prefill,
    std::uint32_t yarn_enabled, float yarn_low, float yarn_high,
    float yarn_inv_factor, float yarn_mscale) {
  return PrepareAttention(
      packed, stride, q_gamma, k_gamma, q, gate, k_cache, v_cache, n_tokens,
      heads, kv_heads, d, rotary_dim, start_pos, theta, eps, stream, rope,
      prefill,
      Yarn(yarn_enabled, yarn_low, yarn_high, yarn_inv_factor, yarn_mscale));
}

void RopeYarn(float* x, std::uint32_t n_tokens, std::uint32_t heads,
              std::uint32_t d, std::uint32_t rotary_dim,
              const std::uint32_t* start_pos, float theta, hipStream_t stream,
              const qwen::vision::DeviceRope* rope, std::uint32_t yarn_enabled,
              float yarn_low, float yarn_high, float yarn_inv_factor,
              float yarn_mscale) {
  Rope(x, n_tokens, heads, d, rotary_dim, start_pos, theta, stream, rope,
       Yarn(yarn_enabled, yarn_low, yarn_high, yarn_inv_factor, yarn_mscale));
}

void PoolIndexerBlocksYarn(
    const float* raw_keys, const float* gamma, __half* blocks,
    const std::uint32_t* first_block, const std::uint32_t* start_pos,
    std::uint32_t n_tokens, std::uint32_t grid_blocks, std::uint32_t ratio,
    std::uint32_t dim, std::uint32_t rotary_dim, float theta, float eps,
    std::uint32_t capacity, hipStream_t stream,
    const qwen::vision::DeviceRope* rope, std::uint32_t yarn_enabled,
    float yarn_low, float yarn_high, float yarn_inv_factor,
    float yarn_mscale) {
  PoolIndexerBlocks(
      raw_keys, gamma, blocks, first_block, start_pos, n_tokens, grid_blocks,
      ratio, dim, rotary_dim, theta, eps, capacity, stream, rope,
      Yarn(yarn_enabled, yarn_low, yarn_high, yarn_inv_factor, yarn_mscale));
}

}  // namespace gufo::models::qwen38_flash_next::rocm
