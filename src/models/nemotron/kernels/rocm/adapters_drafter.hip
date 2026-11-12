// The drafter's adapters: drafter_kernels.hpp and gemv.hpp both declare
// kGemvMaxTokens, so they take their own translation unit.
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <climits>
#include <stdexcept>
#include <string>

#include "src/models/nemotron/kernels/rocm/adapters.hpp"
#include "src/models/nemotron/kernels/rocm/drafter_kernels.hpp"

namespace gufo::models::nemotron::rocm::flat {
namespace {

/// The launchers' `int` counts.
bool Fits(std::size_t value) {
  return value <= static_cast<std::size_t>(INT_MAX);
}

template<typename... T>
bool Fit(T... values) {
  return (Fits(values) && ...);
}

int I(std::size_t value) {
  return static_cast<int>(value);
}

/// The launcher's result: rejected arguments queue nothing and read as
/// unsupported; other errors (a failed launch) throw.
bool Done(hipError_t status) {
  if (status == hipSuccess)
    return true;
  if (status == hipErrorInvalidValue)
    return false;
  throw std::runtime_error(std::string("Nemotron-H kernel launch failed: ") +
                           hipGetErrorString(status));
}

const __half* H(const void* p) {
  return static_cast<const __half*>(p);
}

__half* H(void* p) {
  return static_cast<__half*>(p);
}

}  // namespace

// --- drafter_kernels.hpp -----------------------------------------------------

bool ToBf16(const float* x, void* out, std::size_t n, hipStream_t stream) {
  return Done(rocm::ToBf16(x, static_cast<__hip_bfloat16*>(out), n, stream));
}

bool GemvBf16(const void* w, const float* x, float* out, std::size_t rows,
              std::size_t cols, std::size_t n_tokens, hipStream_t stream) {
  return Fit(rows, cols, n_tokens) &&
         Done(rocm::GemvBf16(static_cast<const __hip_bfloat16*>(w), x, out,
                             I(rows), I(cols), I(n_tokens), stream));
}

bool ScaleBias(float* x, float scale, const float* bias, std::size_t rows,
               std::size_t cols, std::uint32_t act, hipStream_t stream) {
  if (!Fit(rows, cols) || act > 1)
    return false;
  return Done(rocm::ScaleBias(x, scale, bias, I(rows), I(cols),
                              act == 0 ? Activation::kNone : Activation::kSilu,
                              stream));
}

bool SiluGate(const float* gate, float gate_scale, const float* up,
              float up_scale, float* out, std::size_t n, hipStream_t stream) {
  return Done(rocm::SiluGate(gate, gate_scale, up, up_scale, out, n, stream));
}

bool DflashConv(const float* x, const float* dynamic, const float* base,
                float* out, std::size_t n_tokens, std::size_t block,
                std::size_t hidden, std::size_t group, std::size_t kernel,
                std::size_t side, hipStream_t stream) {
  return Fit(n_tokens, block, hidden, group, kernel, side) &&
         Done(rocm::DflashConv(x, dynamic, base, out, I(n_tokens), I(block),
                               I(hidden), I(group), I(kernel), I(side),
                               stream));
}

bool RopeNeox(float* x, const std::int32_t* first, std::size_t n_tokens,
              std::size_t heads, std::size_t head_dim, float theta,
              hipStream_t stream) {
  return Fit(n_tokens, heads, head_dim) &&
         Done(rocm::RopeNeox(x, first, I(n_tokens), I(heads), I(head_dim),
                             theta, stream));
}

bool RingAppend(const float* k, const float* v, void* k_ring, void* v_ring,
                const std::int32_t* first, std::size_t n_tokens,
                std::size_t kv_dim, std::size_t ring, hipStream_t stream) {
  return Fit(n_tokens, kv_dim, ring) &&
         Done(rocm::RingAppend(k, v, H(k_ring), H(v_ring), first, I(n_tokens),
                               I(kv_dim), I(ring), stream));
}

bool RowsToRing(const float* x, float* ring, const std::int32_t* first,
                std::size_t n_tokens, std::size_t width, std::size_t ring_rows,
                hipStream_t stream) {
  return Fit(n_tokens, width, ring_rows) &&
         Done(rocm::RowsToRing(x, ring, first, I(n_tokens), I(width),
                               I(ring_rows), stream));
}

bool DrafterAttention(const float* q, const void* k_ring, const void* v_ring,
                      const float* sinks, float* out, const std::int32_t* first,
                      std::size_t n_tokens, std::size_t heads,
                      std::size_t kv_heads, std::size_t ring,
                      std::size_t window, float scale, hipStream_t stream) {
  return Fit(n_tokens, heads, kv_heads, ring, window) &&
         Done(rocm::DrafterAttention(q, H(k_ring), H(v_ring), sinks, out, first,
                                     I(n_tokens), I(heads), I(kv_heads),
                                     I(ring), I(window), scale, stream));
}

bool TopK8(const float* logits, std::size_t rows, std::size_t vocab,
           std::int32_t* ids, float* values, float* lse, hipStream_t stream) {
  return Fit(rows, vocab) &&
         Done(rocm::TopK8(logits, I(rows), I(vocab), ids, values, lse, stream));
}

bool LiliCorrFeatures(const float* values, const float* lse, const float* w,
                      const float* b, float* out, std::size_t slots,
                      hipStream_t stream) {
  return Fit(slots * kLiliCorrTopK) &&
         Done(rocm::LiliCorrFeatures(values, lse, w, b, out, I(slots), stream));
}

bool LiliCorrCombine(const float* token, const float* pass,
                     const float* feature, const float* slot_embd,
                     const float* rank_embd, float* x, std::size_t slots,
                     std::size_t width, hipStream_t stream) {
  return Fit(slots * kLiliCorrTopK, width) &&
         Done(rocm::LiliCorrCombine(token, pass, feature, slot_embd, rank_embd,
                                    x, I(slots), I(width), stream));
}

bool NodeAttention(const float* qkv, const float* bias, float* out,
                   std::size_t nodes, std::size_t heads, float scale,
                   hipStream_t stream) {
  return Fit(nodes, heads) &&
         Done(rocm::NodeAttention(qkv, bias, out, I(nodes), I(heads), scale,
                                  stream));
}

bool FactorInput(const float* x, const float* anchor, float* out,
                 std::size_t nodes, std::size_t width, hipStream_t stream) {
  return Fit(nodes, width) &&
         Done(rocm::FactorInput(x, anchor, out, I(nodes), I(width), stream));
}

bool BiasL2Norm(float* x, const float* bias, std::size_t rows,
                std::size_t width, float eps, hipStream_t stream) {
  return Fit(rows, width) &&
         Done(rocm::BiasL2Norm(x, bias, I(rows), I(width), eps, stream));
}

bool LatticeScores(const float* in, const float* out, const float* anchor,
                   float* scores, std::size_t slots, std::size_t width,
                   float scale, hipStream_t stream) {
  return Fit(slots, width) &&
         Done(rocm::LatticeScores(in, out, anchor, scores, I(slots), I(width),
                                  scale, stream));
}

}  // namespace gufo::models::nemotron::rocm::flat
