// The Gemma 4 vision tower's kernels: vision/encoder.hip of the extension,
// lines 1-419, verbatim except for the includes and without the host
// helpers that file's Encoder uses (Check, CheckBlas, Allocation). The
// Encoder itself (weight upload, the GEMMs, the layer loop) is host code,
// ported to Rust; adapters_vision.hip launches these kernels with that
// Encoder's grids. Included by exactly one translation unit.
#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_VISION_KERNELS_HIP_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_VISION_KERNELS_HIP_HPP_

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/gemma4/kernels/rocm/vision_constants.hpp"

namespace gufo::models::gemma4::vision {
namespace {
/// GEMM operand format: binary16 activations (and the BF16 weights converted
/// on upload, exact within its range), FP32 accumulation.
using Act = _Float16;

/// Binary16 of v, saturated to the finite range.
__device__ inline Act ToAct(float v) {
  return static_cast<Act>(fminf(fmaxf(v, -65504.0F), 65504.0F));
}

/// In place, BF16 weights to binary16 (values below 2^-24 flush to zero).
__global__ void Bf16ToHalf(std::uint16_t* data, std::size_t count) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float v = __uint_as_float(std::uint32_t{data[i]} << 16U);
  data[i] = __builtin_bit_cast(std::uint16_t, ToAct(v));
}
constexpr unsigned kThreads = 256;
constexpr unsigned kQkv = 3 * kHidden;
constexpr unsigned kGateUp = 2 * kFfn;
/// The down projection's K padded to whole 32-value blocks for the binary16
/// WMMA GEMM (zero weights and activations past kFfn).
constexpr unsigned kFfnPadded = (kFfn + 31) / 32 * 32;
/// Rotary pairs per axis: each half of a head rotates as NEOX pairs.
constexpr unsigned kRopePairs = kHeadDim / 4;
/// Attention operands pad each head to whole 16-wide WMMA steps and each
/// key sequence to whole kKeyTile tiles, with zeros.
constexpr unsigned kPaddedDim = 80;
constexpr unsigned kDimSteps = kPaddedDim / 16;
constexpr unsigned kAttentionWaves = 4;
/// Keys per softmax update; key sequences are padded to whole tiles.
constexpr unsigned kKeyTile = 64;

using Half16 = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using Float8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;

unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

/// Sum over a kThreads block; every thread receives the total.
__device__ float BlockSum(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  const float total = shared[0];
  __syncthreads();
  return total;
}

/// Patches in 2*(v/255)-1, each ordered channel, row, column like the
/// patch-embedding kernel.
__global__ void Patchify(const std::uint8_t* pixels, float* patches,
                         unsigned width, unsigned columns, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kPatchValues)
    return;
  const unsigned t = i / kPatchValues;
  const unsigned e = i % kPatchValues;
  const unsigned c = e / (kPatchSize * kPatchSize);
  const unsigned y = (e / kPatchSize) % kPatchSize;
  const unsigned x = e % kPatchSize;
  const unsigned px = (t % columns) * kPatchSize + x;
  const unsigned py = (t / columns) * kPatchSize + y;
  patches[i] =
      2.0F * (static_cast<float>(pixels[(py * width + px) * 3 + c]) / 255.0F) -
      1.0F;
}

__global__ void AddPositions(float* hidden, const float* table,
                             unsigned columns, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kHidden)
    return;
  const unsigned t = i / kHidden;
  const unsigned d = i % kHidden;
  hidden[i] += table[(t % columns) * kHidden + d] +
               table[(kPositions + t / columns) * kHidden + d];
}

/// y[row] = rms(x[row]) * weight (weight may be null) as binary16.
__global__ void RmsNormHalf(const float* x, const float* weight, Act* y,
                            unsigned width) {
  __shared__ float shared[kThreads];
  const float* row = x + std::size_t{blockIdx.x} * width;
  float sum = 0;
  for (unsigned j = threadIdx.x; j < width; j += kThreads)
    sum += row[j] * row[j];
  const float scale = rsqrtf(BlockSum(sum, shared) / width + kEps);
  for (unsigned j = threadIdx.x; j < width; j += kThreads) {
    y[std::size_t{blockIdx.x} * width + j] =
        ToAct(row[j] * scale * (weight != nullptr ? weight[j] : 1.0F));
  }
}

/// hidden[row] += rms(o[row]) * weight.
__global__ void PostNormAdd(float* hidden, const float* o,
                            const float* weight) {
  __shared__ float shared[kThreads];
  const std::size_t base = std::size_t{blockIdx.x} * kHidden;
  float sum = 0;
  for (unsigned j = threadIdx.x; j < kHidden; j += kThreads)
    sum += o[base + j] * o[base + j];
  const float scale = rsqrtf(BlockSum(sum, shared) / kHidden + kEps);
  for (unsigned j = threadIdx.x; j < kHidden; j += kThreads)
    hidden[base + j] += o[base + j] * scale * weight[j];
}

/// One block of three waves per token and head; wave 0, 1, 2 handles Q, K,
/// V. Each is RMS-normalized over the head (Q and K with their weights, V
/// without); Q and K are rotated. Q and K are written as binary16
/// [head][padded count][80], V transposed as [head][80][padded count], the
/// layouts FusedAttention reads. `cos`/`sin` hold [position][kRopePairs] for
/// the patch column (first half of the head) and row (second half).
__global__ void __launch_bounds__(96)
    QkvPost(const float* qkv, const float* q_norm, const float* k_norm,
            const float* cos, const float* sin, _Float16* q, _Float16* k,
            _Float16* vt, unsigned columns, unsigned padded) {
  __shared__ float values[3][kHeadDim];
  const unsigned token = blockIdx.x / kHeads;
  const unsigned head = blockIdx.x % kHeads;
  const unsigned part = threadIdx.x / 32;
  const unsigned lane = threadIdx.x % 32;
  const float* in =
      qkv + std::size_t{token} * kQkv + part * kHidden + head * kHeadDim;
  float x[3];
  float sum = 0;
#pragma unroll
  for (unsigned j = 0; j < 3; ++j) {
    const unsigned d = lane + 32 * j;
    x[j] = d < kHeadDim ? in[d] : 0.0F;
    sum += x[j] * x[j];
  }
#pragma unroll
  for (unsigned offset = 16; offset > 0; offset /= 2)
    sum += __shfl_xor(sum, offset, 32);
  const float scale = rsqrtf(sum / kHeadDim + kEps);
  const float* weight = part == 0 ? q_norm : part == 1 ? k_norm : nullptr;
#pragma unroll
  for (unsigned j = 0; j < 3; ++j) {
    const unsigned d = lane + 32 * j;
    if (d < kHeadDim)
      values[part][d] = x[j] * scale * (weight != nullptr ? weight[d] : 1.0F);
  }
  __syncthreads();
#pragma unroll
  for (unsigned j = 0; j < 3; ++j) {
    const unsigned d = lane + 32 * j;
    if (d >= kHeadDim)
      continue;
    if (part == 2) {
      vt[(std::size_t{head} * kPaddedDim + d) * padded + token] =
          static_cast<_Float16>(values[2][d]);
      continue;
    }
    const unsigned half = d / (kHeadDim / 2);
    const unsigned i = d % (kHeadDim / 2);
    const unsigned pair = i % kRopePairs;
    const unsigned position = half == 0 ? token % columns : token / columns;
    const float c = cos[position * kRopePairs + pair];
    const float s = sin[position * kRopePairs + pair];
    const unsigned first = half * (kHeadDim / 2) + pair;
    const float a = values[part][first];
    const float b = values[part][first + kRopePairs];
    const float rotated = i < kRopePairs ? a * c - b * s : b * c + a * s;
    (part == 0 ? q : k)[(std::size_t{head} * padded + token) * kPaddedDim + d] =
        static_cast<_Float16>(rotated);
  }
}

__device__ __forceinline__ Float8 Wmma(Half16 a, Half16 b, Float8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ Half16 LoadHalf16(const _Float16* p) {
  union {
    Half16 h;
    uint4 u[2];
  } value;
  value.u[0] = *reinterpret_cast<const uint4*>(p);
  value.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return value.h;
}

/// Bidirectional scale-1 attention with an online softmax. Each wave owns 16
/// queries of one head and walks 64-key tiles (four 16-key WMMA blocks per
/// softmax update): binary16 Q K^T on the matrix cores, FP32 softmax state,
/// binary16 probabilities times V accumulated in FP32. Fragment layout
/// (wave32 v_wmma_f32_16x16x16_f16): A holds row L%16 and 16 contiguous k, B
/// column L%16 and 16 contiguous k, and accumulator element i is row
/// 2i + L/16, column L%16. Writes token-major binary16 rows.
__global__ void __launch_bounds__(32 * kAttentionWaves)
    FusedAttention(const _Float16* q, const _Float16* k, const _Float16* vt,
                   Act* out, unsigned count, unsigned padded) {
  // Per-wave probability tiles: waves never exchange data.
  __shared__ __align__(16) _Float16 p_lds[kAttentionWaves][16][kKeyTile + 8];
  const unsigned lane = threadIdx.x % 32;
  const unsigned wave = threadIdx.x / 32;
  const unsigned head = blockIdx.y;
  const unsigned q0 = (blockIdx.x * kAttentionWaves + wave) * 16;
  if (q0 >= count)
    return;
  const unsigned row = lane % 16;
  const unsigned half = lane / 16;
  const _Float16* qh = q + std::size_t{head} * padded * kPaddedDim;
  const _Float16* kh = k + std::size_t{head} * padded * kPaddedDim;
  const _Float16* vh = vt + std::size_t{head} * kPaddedDim * padded;
  Half16 query[kDimSteps];
#pragma unroll
  for (unsigned s = 0; s < kDimSteps; ++s)
    query[s] = LoadHalf16(qh + std::size_t{q0 + row} * kPaddedDim + s * 16);
  Float8 acc[kDimSteps] = {};
  float m[8], l[8];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    m[i] = -INFINITY;
    l[i] = 0.0F;
  }
  constexpr unsigned kBlocks = kKeyTile / 16;
  for (unsigned key0 = 0; key0 < count; key0 += kKeyTile) {
    Float8 score[kBlocks] = {};
#pragma unroll
    for (unsigned b = 0; b < kBlocks; ++b) {
#pragma unroll
      for (unsigned s = 0; s < kDimSteps; ++s) {
        score[b] =
            Wmma(query[s],
                 LoadHalf16(kh + std::size_t{key0 + b * 16 + row} * kPaddedDim +
                            s * 16),
                 score[b]);
      }
    }
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      float x[kBlocks];
      float maximum = -INFINITY;
#pragma unroll
      for (unsigned b = 0; b < kBlocks; ++b) {
        x[b] = key0 + b * 16 + row < count ? score[b][i] : -INFINITY;
        maximum = fmaxf(maximum, x[b]);
      }
#pragma unroll
      for (unsigned offset = 8; offset > 0; offset /= 2)
        maximum = fmaxf(maximum, __shfl_xor(maximum, offset, 32));
      const float next = fmaxf(m[i], maximum);
      const float alpha = __expf(m[i] - next);
      float sum = 0.0F;
#pragma unroll
      for (unsigned b = 0; b < kBlocks; ++b) {
        const float p = __expf(x[b] - next);
        sum += p;
        p_lds[wave][2 * i + half][b * 16 + row] = static_cast<_Float16>(p);
      }
#pragma unroll
      for (unsigned offset = 8; offset > 0; offset /= 2)
        sum += __shfl_xor(sum, offset, 32);
      l[i] = l[i] * alpha + sum;
      m[i] = next;
#pragma unroll
      for (unsigned c = 0; c < kDimSteps; ++c)
        acc[c][i] *= alpha;
    }
    // A wave's LDS accesses complete in order; the barriers only keep the
    // compiler from moving them across the exchange.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
    for (unsigned b = 0; b < kBlocks; ++b) {
      const Half16 probabilities = LoadHalf16(&p_lds[wave][row][b * 16]);
#pragma unroll
      for (unsigned c = 0; c < kDimSteps; ++c) {
        acc[c] = Wmma(
            probabilities,
            LoadHalf16(vh + std::size_t{c * 16 + row} * padded + key0 + b * 16),
            acc[c]);
      }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
  }
#pragma unroll
  for (unsigned c = 0; c < kDimSteps; ++c) {
    const unsigned d = c * 16 + row;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      const unsigned token = q0 + 2 * i + half;
      if (token < count && d < kHeadDim) {
        out[std::size_t{token} * kHidden + head * kHeadDim + d] =
            ToAct(acc[c][i] / l[i]);
      }
    }
  }
}

/// gelu_tanh(gate) * up, from fused [count][gate | up] rows, as
/// [count][kFfnPadded] rows.
__global__ void GeGlu(const float* gate_up, Act* output, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kFfnPadded)
    return;
  const std::size_t row = i / kFfnPadded;
  const unsigned j = i % kFfnPadded;
  if (j >= kFfn) {
    output[i] = Act(0);
    return;
  }
  const float x = gate_up[row * kGateUp + j];
  const float up = gate_up[row * kGateUp + kFfn + j];
  const float gelu =
      0.5F * x *
      (1.0F + tanhf(0.7978845608028654F * (x + 0.044715F * x * x * x)));
  output[i] = ToAct(gelu * up);
}

/// 3x3 mean over the patch grid times sqrt(1152), row-major pooled cells.
__global__ void Pool(const float* hidden, float* pooled, unsigned columns,
                     unsigned pooled_columns, unsigned rows) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * kHidden)
    return;
  const unsigned r = i / kHidden;
  const unsigned d = i % kHidden;
  const unsigned px = (r % pooled_columns) * kPoolSize;
  const unsigned py = (r / pooled_columns) * kPoolSize;
  float sum = 0;
  for (unsigned y = 0; y < kPoolSize; ++y) {
    for (unsigned x = 0; x < kPoolSize; ++x)
      sum += hidden[(std::size_t{py + y} * columns + px + x) * kHidden + d];
  }
  pooled[i] = sum / (kPoolSize * kPoolSize) * sqrtf(float{kHidden});
}

/// rms((pooled - bias) * scale) without weight, as binary16.
__global__ void StandardizeNorm(const float* pooled, const float* bias,
                                const float* scale, Act* output) {
  __shared__ float shared[kThreads];
  __shared__ float values[kHidden];
  const std::size_t base = std::size_t{blockIdx.x} * kHidden;
  float sum = 0;
  for (unsigned j = threadIdx.x; j < kHidden; j += kThreads) {
    values[j] = (pooled[base + j] - bias[j]) * scale[j];
    sum += values[j] * values[j];
  }
  const float factor = rsqrtf(BlockSum(sum, shared) / kHidden + kEps);
  for (unsigned j = threadIdx.x; j < kHidden; j += kThreads)
    output[base + j] = ToAct(values[j] * factor);
}

}  // namespace

}  // namespace gufo::models::gemma4::vision

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_VISION_KERNELS_HIP_HPP_
