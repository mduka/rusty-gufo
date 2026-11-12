// The vision encoder's operators follow NVIDIA's C-RADIO and projector code
// (Apache-2.0, see licenses/nemotron-omni.txt); the position table's resize
// follows PyTorch's upsample_bilinear2d (BSD-3-Clause, licenses/pytorch.txt).
#include <hip/hip_fp16.h>

#include <cmath>

#include "src/models/nemotron/kernels/rocm/lanes.hip.hpp"
#include "src/models/nemotron/kernels/rocm/vision.hpp"

namespace gufo::models::nemotron::rocm {
namespace {

constexpr int kBlock = 256;
constexpr int kWave = 32;

template <VisionOperand O>
__device__ __forceinline__ std::uint16_t Encode(float x) {
  if constexpr (O == VisionOperand::kBf16) {
    const auto u = __builtin_bit_cast(std::uint32_t, x);
    if ((u & 0x7FFFFFFFU) > 0x7F800000U)  // NaN stays a quiet NaN
      return static_cast<std::uint16_t>((u >> 16) | 0x40U);
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
  } else {
    return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(x));
  }
}

__device__ __forceinline__ float Bf16ToFloat(std::uint16_t v) {
  return __builtin_bit_cast(float, static_cast<std::uint32_t>(v) << 16);
}

__device__ __forceinline__ float WaveSumShfl(float v) {
  for (int offset = kWave / 2; offset > 0; offset /= 2)
    v += __shfl_xor(v, offset, kWave);
  return v;
}

/// Sum over the block; every thread receives the result.
__device__ float BlockSum(float v, float* scratch) {
  v = WaveSumShfl(v);
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  __syncthreads();
  if (lane == 0)
    scratch[wave] = v;
  __syncthreads();
  float total = 0.0F;
  for (int i = 0; i < static_cast<int>(blockDim.x) / kWave; ++i)
    total += scratch[i];
  return total;
}

unsigned Blocks(std::size_t n) {
  return static_cast<unsigned>((n + kBlock - 1) / kBlock);
}

template <VisionOperand O>
__global__ void ToOperandKernel(const float* in, std::uint16_t* out,
                                std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(kBlock) + threadIdx.x;
  if (i < n)
    out[i] = Encode<O>(in[i]);
}

__global__ void Bf16ToF16Kernel(const std::uint16_t* in, std::uint16_t* out,
                                std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(kBlock) + threadIdx.x;
  if (i < n)
    out[i] = Encode<VisionOperand::kF16>(Bf16ToFloat(in[i]));
}

/// Source cell and weights of a half-pixel bilinear resize along one axis,
/// as PyTorch's upsample_bilinear2d computes them.
__device__ __forceinline__ void Bilinear(float scale, int dst, int size, int* i0,
                                         int* i1, float* l0, float* l1) {
  float src = scale * (static_cast<float>(dst) + 0.5F) - 0.5F;
  if (src < 0.0F)
    src = 0.0F;
  *i0 = min(static_cast<int>(src), size - 1);
  *i1 = *i0 + (*i0 < size - 1 ? 1 : 0);
  *l1 = fminf(fmaxf(src - static_cast<float>(*i0), 0.0F), 1.0F);
  *l0 = 1.0F - *l1;
}

__global__ void EmbedKernel(float* x, const float* prefix,
                            const float* positions, int prefix_tokens,
                            int grid_rows, int grid_cols, int side,
                            int width) {
  const int t = static_cast<int>(blockIdx.x);
  float* dst = x + static_cast<std::size_t>(t) * width;
  if (t < prefix_tokens) {
    for (int c = threadIdx.x; c < width; c += blockDim.x)
      dst[c] = prefix[static_cast<std::size_t>(t) * width + c];
    return;
  }
  const int p = t - prefix_tokens;
  const int r = p / grid_cols;
  const int col = p % grid_cols;
  // The table resized to the square of the longer side, then cropped.
  const float scale = static_cast<float>(side) /
                      static_cast<float>(max(grid_rows, grid_cols));
  int y0, y1, x0, x1;
  float ly0, ly1, lx0, lx1;
  Bilinear(scale, r, side, &y0, &y1, &ly0, &ly1);
  Bilinear(scale, col, side, &x0, &x1, &lx0, &lx1);
  const float* p00 = positions + (static_cast<std::size_t>(y0) * side + x0) * width;
  const float* p01 = positions + (static_cast<std::size_t>(y0) * side + x1) * width;
  const float* p10 = positions + (static_cast<std::size_t>(y1) * side + x0) * width;
  const float* p11 = positions + (static_cast<std::size_t>(y1) * side + x1) * width;
  for (int c = threadIdx.x; c < width; c += blockDim.x)
    dst[c] += ly0 * (lx0 * p00[c] + lx1 * p01[c]) +
              ly1 * (lx0 * p10[c] + lx1 * p11[c]);
}

template <VisionOperand O>
__global__ void LayerNormKernel(const float* x, const float* w, const float* b,
                                std::uint16_t* out, int width, float eps) {
  __shared__ float scratch[kBlock / kWave];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * width;
  const float* row = x + base;
  float sum = 0.0F;
  for (int i = threadIdx.x; i < width; i += blockDim.x)
    sum += row[i];
  const float mean = BlockSum(sum, scratch) / static_cast<float>(width);
  float squares = 0.0F;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float d = row[i] - mean;
    squares += d * d;
  }
  const float rstd =
      rsqrtf(BlockSum(squares, scratch) / static_cast<float>(width) + eps);
  for (int i = threadIdx.x; i < width; i += blockDim.x)
    out[base + i] = Encode<O>((row[i] - mean) * rstd * w[i] + b[i]);
}

template <VisionOperand O, bool kGelu>
__global__ void BiasKernel(const float* y, const float* bias,
                           std::uint16_t* out, std::size_t n, int cols) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(kBlock) + threadIdx.x;
  if (i >= n)
    return;
  float v = y[i] + bias[i % cols];
  if constexpr (kGelu)
    v = 0.5F * v * (1.0F + erff(v * 0.70710678118654752F));
  out[i] = Encode<O>(v);
}

__global__ void ResidualKernel(float* x, const float* y, const float* bias,
                               std::size_t n, int cols) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(kBlock) + threadIdx.x;
  if (i < n)
    x[i] += y[i] + bias[i % cols];
}

template <VisionOperand O>
__global__ void MergeNormKernel(const float* x, const float* w,
                                std::uint16_t* out, int prefix_tokens,
                                int grid_cols, int width, float eps) {
  __shared__ float scratch[kBlock / kWave];
  const int half_cols = grid_cols / 2;
  const int m = static_cast<int>(blockIdx.x);
  const int i = m / half_cols;
  const int j = m % half_cols;
  const int merged = 4 * width;
  const auto value = [&](int k) {
    const int q = k / width;
    const int patch = (2 * i + q / 2) * grid_cols + 2 * j + q % 2;
    return x[(static_cast<std::size_t>(prefix_tokens) + patch) * width +
             k % width];
  };
  float squares = 0.0F;
  for (int k = threadIdx.x; k < merged; k += blockDim.x) {
    const float v = value(k);
    squares += v * v;
  }
  const float r =
      rsqrtf(BlockSum(squares, scratch) / static_cast<float>(merged) + eps);
  std::uint16_t* dst = out + static_cast<std::size_t>(m) * merged;
  for (int k = threadIdx.x; k < merged; k += blockDim.x)
    dst[k] = Encode<O>(w[k] * (value(k) * r));
}

template <VisionOperand O>
__global__ void Relu2Kernel(const float* y, std::uint16_t* out, std::size_t n) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(kBlock) + threadIdx.x;
  if (i >= n)
    return;
  const float v = fmaxf(y[i], 0.0F);
  out[i] = Encode<O>(v * v);
}

// Flash attention of the ViT on the matrix units.
using V16u = unsigned short __attribute__((ext_vector_type(16)));
using V16h = _Float16 __attribute__((ext_vector_type(16)));
using V8f = float __attribute__((ext_vector_type(8)));

/// Keys per staged tile.
constexpr int kAttnKeys = 64;
/// Waves per block; each takes 16 query rows of one head.
constexpr int kAttnWaves = 8;
/// Values per staged row: payload plus bank padding.
constexpr int kKeyStride = kVisionHeadDim + 8;
constexpr int kVtStride = kAttnKeys + 8;

__device__ __forceinline__ V16u Load16(const std::uint16_t* p) {
  union {
    V16u v;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.v;
}

template <VisionOperand O>
__device__ __forceinline__ V8f Wmma(V16u a, V16u b, V8f c) {
  if constexpr (O == VisionOperand::kBf16)
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
  else
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
        __builtin_bit_cast(V16h, a), __builtin_bit_cast(V16h, b), c);
}

/// Max (kMax) or sum over the 16 lanes holding one row of a WMMA C tile.
template <bool kMax>
__device__ __forceinline__ float RowReduce16(float v) {
  if constexpr (kMax) {
    v = fmaxf(v, XorLane<1>(v));
    v = fmaxf(v, XorLane<2>(v));
    v = fmaxf(v, XorLane<4>(v));
    return fmaxf(v, XorLane<8>(v));
  } else {
    v = AddXorLane<1>(v);
    v = AddXorLane<2>(v);
    v = AddXorLane<4>(v);
    return AddXorLane<8>(v);
  }
}

/// Block (x, h) takes kAttnWaves * 16 query rows of head h; wave w its rows
/// q0 .. q0 + 15 as one WMMA row tile. K and V tiles of kAttnKeys keys are
/// staged once per block (V transposed); S = Q K^T and O += P V run on the
/// matrix units with F32 accumulation, the online softmax in F32 in the
/// exp2 domain. In a C tile, element r of a lane is row 2 r + lane / 16 and
/// column lane % 16.
template <VisionOperand O>
__global__ __launch_bounds__(kAttnWaves * kWave) void AttentionKernel(
    const std::uint16_t* qkv, std::uint16_t* out, int tokens, int heads,
    float scale_log2) {
  __shared__ alignas(16) std::uint16_t s_k[kAttnKeys][kKeyStride];
  __shared__ alignas(16) std::uint16_t s_vt[kVisionHeadDim][kVtStride];
  __shared__ alignas(16) std::uint16_t s_p[kAttnWaves][16][kVtStride];
  constexpr int kSteps = kVisionHeadDim / 16;
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWave;
  const int lane = tid % kWave;
  const int col = lane % 16;
  const int hi = lane / 16;
  const int h = static_cast<int>(blockIdx.y);
  const int width = heads * kVisionHeadDim;
  const std::size_t stride = 3 * static_cast<std::size_t>(width);
  const int q0 = static_cast<int>(blockIdx.x) * kAttnWaves * 16 + wave * 16;
  // Waves without rows only help stage the tiles.
  const bool active = q0 < tokens;

  V16u qf[kSteps];
  {
    const int row = min(q0 + col, tokens - 1);
    const std::uint16_t* src =
        qkv + static_cast<std::size_t>(row) * stride + h * kVisionHeadDim;
#pragma unroll
    for (int ks = 0; ks < kSteps; ++ks)
      qf[ks] = Load16(src + 16 * ks);
  }
  V8f o[kSteps] = {};
  float m[8];
  float l[8];  // Per-lane partial row sums, reduced at the end.
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    m[r] = -INFINITY;
    l[r] = 0.0F;
  }

  for (int k0 = 0; k0 < tokens; k0 += kAttnKeys) {
    __syncthreads();
    {
      // Each thread stages a quarter key row of K and V (20 values).
      const int key = tid / 4;
      const int d0 = (tid % 4) * 20;
      const bool live = k0 + key < tokens;
      const std::uint16_t* ks = qkv + static_cast<std::size_t>(k0 + key) * stride +
                                width + h * kVisionHeadDim + d0;
      uint2 kk[5];
      uint2 vv[5];
#pragma unroll
      for (int i = 0; i < 5; ++i) {
        kk[i] = live ? reinterpret_cast<const uint2*>(ks)[i] : make_uint2(0U, 0U);
        vv[i] = live ? reinterpret_cast<const uint2*>(ks + width)[i]
                     : make_uint2(0U, 0U);
      }
#pragma unroll
      for (int i = 0; i < 5; ++i)
        reinterpret_cast<uint2*>(&s_k[key][d0])[i] = kk[i];
      const auto* vh = reinterpret_cast<const std::uint16_t*>(vv);
#pragma unroll
      for (int e = 0; e < 20; ++e)
        s_vt[d0 + e][key] = vh[e];
    }
    __syncthreads();
    if (!active)
      continue;

    V8f s[kAttnKeys / 16];
#pragma unroll
    for (int j = 0; j < kAttnKeys / 16; ++j) {
      s[j] = V8f{};
#pragma unroll
      for (int ks = 0; ks < kSteps; ++ks)
        s[j] = Wmma<O>(qf[ks], Load16(&s_k[16 * j + col][16 * ks]), s[j]);
    }
#pragma unroll
    for (int j = 0; j < kAttnKeys / 16; ++j) {
      const bool live = k0 + 16 * j + col < tokens;
#pragma unroll
      for (int r = 0; r < 8; ++r)
        s[j][r] = live ? s[j][r] * scale_log2 : -INFINITY;
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      float tile_max = s[0][r];
#pragma unroll
      for (int j = 1; j < kAttnKeys / 16; ++j)
        tile_max = fmaxf(tile_max, s[j][r]);
      // Every tile holds at least one key, so the maximum is finite.
      const float new_m = fmaxf(m[r], RowReduce16<true>(tile_max));
      const float corr = exp2f(m[r] - new_m);
      m[r] = new_m;
      float sum = 0.0F;
#pragma unroll
      for (int j = 0; j < kAttnKeys / 16; ++j) {
        const float p = exp2f(s[j][r] - new_m);
        sum += p;
        s_p[wave][2 * r + hi][16 * j + col] = Encode<O>(p);
      }
      l[r] = l[r] * corr + sum;
#pragma unroll
      for (int dt = 0; dt < kSteps; ++dt)
        o[dt][r] *= corr;
    }
    __builtin_amdgcn_wave_barrier();
#pragma unroll
    for (int kk = 0; kk < kAttnKeys / 16; ++kk) {
      const V16u pf = Load16(&s_p[wave][col][16 * kk]);
#pragma unroll
      for (int dt = 0; dt < kSteps; ++dt)
        o[dt] = Wmma<O>(pf, Load16(&s_vt[16 * dt + col][16 * kk]), o[dt]);
    }
    __builtin_amdgcn_wave_barrier();
  }
  if (!active)
    return;
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const float inv = 1.0F / RowReduce16<false>(l[r]);
    const int row = q0 + 2 * r + hi;
    if (row >= tokens)
      continue;
    std::uint16_t* dst = out + static_cast<std::size_t>(row) * width +
                         h * kVisionHeadDim + col;
#pragma unroll
    for (int dt = 0; dt < kSteps; ++dt)
      dst[16 * dt] = Encode<O>(o[dt][r] * inv);
  }
}

}  // namespace

hipError_t VisionToOperand(const float* in, std::uint16_t* out, std::size_t n,
                           VisionOperand op, hipStream_t stream) {
  if (n == 0)
    return hipSuccess;
  if (op == VisionOperand::kBf16)
    ToOperandKernel<VisionOperand::kBf16><<<Blocks(n), kBlock, 0, stream>>>(
        in, out, n);
  else
    ToOperandKernel<VisionOperand::kF16><<<Blocks(n), kBlock, 0, stream>>>(
        in, out, n);
  return hipGetLastError();
}

hipError_t VisionBf16ToOperand(const std::uint16_t* in, std::uint16_t* out,
                               std::size_t n, VisionOperand op,
                               hipStream_t stream) {
  if (n == 0)
    return hipSuccess;
  if (op == VisionOperand::kBf16)
    return in == out ? hipSuccess
                     : hipMemcpyAsync(out, in, n * sizeof(std::uint16_t),
                                      hipMemcpyDeviceToDevice, stream);
  Bf16ToF16Kernel<<<Blocks(n), kBlock, 0, stream>>>(in, out, n);
  return hipGetLastError();
}

hipError_t VisionEmbed(float* x, const float* prefix, const float* positions,
                       int prefix_tokens, int grid_rows, int grid_cols,
                       int side, int width, hipStream_t stream) {
  const int tokens = prefix_tokens + grid_rows * grid_cols;
  EmbedKernel<<<tokens, kBlock, 0, stream>>>(x, prefix, positions,
                                             prefix_tokens, grid_rows,
                                             grid_cols, side, width);
  return hipGetLastError();
}

hipError_t VisionLayerNorm(const float* x, const float* w, const float* b,
                           std::uint16_t* out, int rows, int width, float eps,
                           VisionOperand op, hipStream_t stream) {
  if (op == VisionOperand::kBf16)
    LayerNormKernel<VisionOperand::kBf16><<<rows, kBlock, 0, stream>>>(
        x, w, b, out, width, eps);
  else
    LayerNormKernel<VisionOperand::kF16><<<rows, kBlock, 0, stream>>>(
        x, w, b, out, width, eps);
  return hipGetLastError();
}

namespace {
template <bool kGelu>
hipError_t LaunchBias(const float* y, const float* bias, std::uint16_t* out,
                      int rows, int cols, VisionOperand op,
                      hipStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  if (n == 0)
    return hipSuccess;
  if (op == VisionOperand::kBf16)
    BiasKernel<VisionOperand::kBf16, kGelu><<<Blocks(n), kBlock, 0, stream>>>(
        y, bias, out, n, cols);
  else
    BiasKernel<VisionOperand::kF16, kGelu><<<Blocks(n), kBlock, 0, stream>>>(
        y, bias, out, n, cols);
  return hipGetLastError();
}
}  // namespace

hipError_t VisionBias(const float* y, const float* bias, std::uint16_t* out,
                      int rows, int cols, VisionOperand op,
                      hipStream_t stream) {
  return LaunchBias<false>(y, bias, out, rows, cols, op, stream);
}

hipError_t VisionBiasGelu(const float* y, const float* bias,
                          std::uint16_t* out, int rows, int cols,
                          VisionOperand op, hipStream_t stream) {
  return LaunchBias<true>(y, bias, out, rows, cols, op, stream);
}

hipError_t VisionResidual(float* x, const float* y, const float* bias,
                          int rows, int cols, hipStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  if (n == 0)
    return hipSuccess;
  ResidualKernel<<<Blocks(n), kBlock, 0, stream>>>(x, y, bias, n, cols);
  return hipGetLastError();
}

hipError_t VisionAttention(const std::uint16_t* qkv, std::uint16_t* out,
                           int tokens, int heads, float scale,
                           VisionOperand op, hipStream_t stream) {
  if (tokens <= 0)
    return hipSuccess;
  constexpr float kLog2e = 1.4426950408889634F;
  const dim3 grid((tokens + kAttnWaves * 16 - 1) / (kAttnWaves * 16), heads);
  if (op == VisionOperand::kBf16)
    AttentionKernel<VisionOperand::kBf16>
        <<<grid, kAttnWaves * kWave, 0, stream>>>(qkv, out, tokens, heads,
                                                  scale * kLog2e);
  else
    AttentionKernel<VisionOperand::kF16>
        <<<grid, kAttnWaves * kWave, 0, stream>>>(qkv, out, tokens, heads,
                                                  scale * kLog2e);
  return hipGetLastError();
}

hipError_t VisionMergeNorm(const float* x, const float* w, std::uint16_t* out,
                           int prefix_tokens, int grid_rows, int grid_cols,
                           int width, float eps, VisionOperand op,
                           hipStream_t stream) {
  const int merged = (grid_rows / 2) * (grid_cols / 2);
  if (merged == 0)
    return hipSuccess;
  if (op == VisionOperand::kBf16)
    MergeNormKernel<VisionOperand::kBf16><<<merged, kBlock, 0, stream>>>(
        x, w, out, prefix_tokens, grid_cols, width, eps);
  else
    MergeNormKernel<VisionOperand::kF16><<<merged, kBlock, 0, stream>>>(
        x, w, out, prefix_tokens, grid_cols, width, eps);
  return hipGetLastError();
}

hipError_t VisionRelu2(const float* y, std::uint16_t* out, std::size_t n,
                       VisionOperand op, hipStream_t stream) {
  if (n == 0)
    return hipSuccess;
  if (op == VisionOperand::kBf16)
    Relu2Kernel<VisionOperand::kBf16><<<Blocks(n), kBlock, 0, stream>>>(y, out,
                                                                      n);
  else
    Relu2Kernel<VisionOperand::kF16><<<Blocks(n), kBlock, 0, stream>>>(y, out,
                                                                     n);
  return hipGetLastError();
}

}  // namespace gufo::models::nemotron::rocm
