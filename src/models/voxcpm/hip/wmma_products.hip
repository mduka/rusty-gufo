#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>

#include "device_math.hpp"
#include "wmma_products.hpp"

namespace gufo::models::voxcpm::kernels {
namespace {

using Fragment = unsigned short __attribute__((ext_vector_type(16)));
using Accumulator = float __attribute__((ext_vector_type(8)));

constexpr int kWaves = 4;
/// Fewest 32-column blocks per wave when splitting.
constexpr std::size_t kMinBlocks = 4;

__device__ inline Fragment Load16(const void* p) {
  union {
    Fragment v;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *static_cast<const uint4*>(p);
  cvt.u[1] = *(static_cast<const uint4*>(p) + 1);
  return cvt.v;
}

/// Bytes 2h and 2h + 1 of `word` (int8 q) as a packed bf16 pair, exactly
/// (at most 8 significant bits): sign extension, conversion to float32 and
/// its upper half.
__device__ inline std::uint32_t SignedPair(std::uint32_t word, int h) {
  const float lo = static_cast<float>(
      static_cast<std::int8_t>(static_cast<std::uint8_t>(word >> (16 * h))));
  const float hi = static_cast<float>(static_cast<std::int8_t>(
      static_cast<std::uint8_t>(word >> (16 * h + 8))));
  return __builtin_amdgcn_perm(__float_as_uint(hi), __float_as_uint(lo),
                               0x07060302U);
}

__device__ inline Fragment FromQ8(const uint4& bytes) {
  union {
    Fragment v;
    std::uint32_t w[8];
  } cvt;
  cvt.w[0] = SignedPair(bytes.x, 0);
  cvt.w[1] = SignedPair(bytes.x, 1);
  cvt.w[2] = SignedPair(bytes.y, 0);
  cvt.w[3] = SignedPair(bytes.y, 1);
  cvt.w[4] = SignedPair(bytes.z, 0);
  cvt.w[5] = SignedPair(bytes.z, 1);
  cvt.w[6] = SignedPair(bytes.w, 0);
  cvt.w[7] = SignedPair(bytes.w, 1);
  return cvt.v;
}

__device__ inline Accumulator Wmma(const Fragment& a, const Fragment& b,
                                   const Accumulator& c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

__global__ void Q8_0ToTilesKernel(const std::uint8_t* blocks, int rows,
                                  int in, std::uint8_t* output) {
  const int per_row = in / 32;
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * per_row)
    return;
  const int row = index / per_row;
  const int block = index % per_row;
  const std::uint8_t* source = blocks + static_cast<std::size_t>(index) * 34;
  std::uint8_t* tile =
      output +
      (static_cast<std::size_t>(row / 16) * per_row + block) * kQ8TileBytes;
  const int r = row % 16;
  tile[512 + 2 * r] = source[0];
  tile[512 + 2 * r + 1] = source[1];
  for (int i = 0; i < 32; ++i)
    tile[32 * r + i] = source[2 + i];
}

__global__ void Bf16ToTilesKernel(const Bf16* weights, int rows, int in,
                                  Bf16* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(rows) * in)
    return;
  const int row = static_cast<int>(index / in);
  const int k = static_cast<int>(index % in);
  const int per_row = in / 32;
  output[(static_cast<std::size_t>(row / 16) * per_row + k / 32) * 512 +
         (row % 16) * 32 + k % 32] = weights[index];
}

__global__ void Q8TilesToBf16Kernel(const std::uint8_t* tiles, int rows,
                                    int in, Bf16* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(rows) * in)
    return;
  const int row = static_cast<int>(index / in);
  const int k = static_cast<int>(index % in);
  const int per_row = in / 32;
  const std::uint8_t* tile =
      tiles +
      (static_cast<std::size_t>(row / 16) * per_row + k / 32) * kQ8TileBytes;
  const int r = row % 16;
  const auto scale_bits =
      static_cast<std::uint16_t>(tile[512 + 2 * r] | (tile[513 + 2 * r] << 8U));
  const float value =
      static_cast<float>(static_cast<std::int8_t>(tile[32 * r + k % 32]));
  output[index] = Bf16Bits(HalfToFloat(scale_bits) * value);
}

__global__ void Bf16TilesToRowsKernel(const Bf16* tiles, int rows, int in,
                                      Bf16* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(rows) * in)
    return;
  const int row = static_cast<int>(index / in);
  const int k = static_cast<int>(index % in);
  const int per_row = in / 32;
  output[index] =
      tiles[(static_cast<std::size_t>(row / 16) * per_row + k / 32) * 512 +
            (row % 16) * 32 + k % 32];
}

// One wave per 16 weight rows (output columns n) and a range of 32-column
// blocks of the reduction. A (16 x 16) is 16 activation rows, B the 16
// weight rows: lane L holds activation row L % 16 and weight row L % 16
// with 16 consecutive k, replicated in both half-waves; C element i of
// lane L is activation row 2i + L / 16 and weight row L % 16, so a weight
// row's Q8_0 scale is one value per lane. Activation rows beyond `m_size`
// read the last row; their results are dropped.
//
// With several splits, each wave stores its partial sums; the last wave of
// a tile to finish (an atomic counter) adds them in split order, so the
// result doesn't depend on timing, and resets the counter.
template<bool kQ8, int kMTiles>
__global__ __launch_bounds__(kWaves * 32) void WmmaKernel(
    const Bf16* x, const std::uint8_t* w, float* y, float* partials,
    unsigned* counters, int m_size, int k_size, int n_size, int splits,
    int blocks_per_split, bool accumulate) {
  constexpr std::size_t kTileBytes = kQ8 ? kQ8TileBytes : kBf16TileBytes;
  const int lane = threadIdx.x % 32;
  const int tiles = (n_size + 15) / 16;
  const int tile = static_cast<int>(blockIdx.x) * kWaves + threadIdx.x / 32;
  if (tile >= tiles)
    return;
  const int split = static_cast<int>(blockIdx.y);
  const int k_blocks = k_size / 32;
  const int first = split * blocks_per_split;
  const int last = min(first + blocks_per_split, k_blocks);
  const int column = lane % 16;
  const int n = tile * 16 + column;
  const Bf16* x_rows[kMTiles];
#pragma unroll
  for (int t = 0; t < kMTiles; ++t)
    x_rows[t] =
        x + static_cast<std::size_t>(min(t * 16 + column, m_size - 1)) * k_size;
  const std::uint8_t* tile_base =
      w + static_cast<std::size_t>(tile) * k_blocks * kTileBytes;
  Accumulator acc[kMTiles];
#pragma unroll
  for (int t = 0; t < kMTiles; ++t)
    acc[t] = Accumulator{};
  // The operands of one block: the lane's weight row (Q8_0: 32 bytes and
  // the scale; bf16: 64 bytes) and the activation fragments.
  struct Operands {
    uint4 w[kQ8 ? 2 : 4];
    std::uint16_t scale;
    Fragment a[kMTiles][2];
  };
  const auto load = [&](int b, Operands& o) {
    const std::uint8_t* chunk = tile_base + b * kTileBytes;
    if constexpr (kQ8) {
      o.w[0] = *reinterpret_cast<const uint4*>(chunk + column * 32);
      o.w[1] = *reinterpret_cast<const uint4*>(chunk + column * 32 + 16);
      o.scale =
          *reinterpret_cast<const std::uint16_t*>(chunk + 512 + column * 2);
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        o.w[i] = *reinterpret_cast<const uint4*>(chunk + column * 64 + 16 * i);
    }
#pragma unroll
    for (int t = 0; t < kMTiles; ++t) {
      o.a[t][0] = Load16(x_rows[t] + b * 32);
      o.a[t][1] = Load16(x_rows[t] + b * 32 + 16);
    }
  };
  const auto compute = [&](const Operands& o) {
    Fragment b0, b1;
    if constexpr (kQ8) {
      b0 = FromQ8(o.w[0]);
      b1 = FromQ8(o.w[1]);
    } else {
      b0 = Load16(&o.w[0]);
      b1 = Load16(&o.w[2]);
    }
#pragma unroll
    for (int t = 0; t < kMTiles; ++t) {
      if constexpr (kQ8) {
        Accumulator sum = Wmma(o.a[t][0], b0, Accumulator{});
        sum = Wmma(o.a[t][1], b1, sum);
        const float scale = HalfToFloat(o.scale);
#pragma unroll
        for (int i = 0; i < 8; ++i)
          acc[t][i] = fmaf(scale, sum[i], acc[t][i]);
      } else {
        acc[t] = Wmma(o.a[t][0], b0, acc[t]);
        acc[t] = Wmma(o.a[t][1], b1, acc[t]);
      }
    }
  };
  // Two blocks in flight: the next block's loads are issued before this
  // block's products (the loads complete in order, so both operands are
  // prefetched).
  Operands even, odd;
  if (first < last)
    load(first, even);
  for (int b = first; b < last; b += 2) {
    if (b + 1 < last)
      load(b + 1, odd);
    compute(even);
    if (b + 1 >= last)
      break;
    if (b + 2 < last)
      load(b + 2, even);
    compute(odd);
  }
  const int half = lane / 16;
  if (splits == 1) {
    if (n >= n_size)
      return;
#pragma unroll
    for (int t = 0; t < kMTiles; ++t) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int m = t * 16 + 2 * i + half;
        if (m < m_size) {
          float* target = y + static_cast<std::size_t>(m) * n_size + n;
          *target = accumulate ? *target + acc[t][i] : acc[t][i];
        }
      }
    }
    return;
  }
  if (n < n_size) {
#pragma unroll
    for (int t = 0; t < kMTiles; ++t) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int m = t * 16 + 2 * i + half;
        if (m < m_size)
          __hip_atomic_store(
              partials +
                  (static_cast<std::size_t>(split) * m_size + m) * n_size + n,
              acc[t][i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }
  __threadfence();
  unsigned previous = 0;
  if (lane == 0)
    previous = __hip_atomic_fetch_add(counters + tile, 1U, __ATOMIC_ACQ_REL,
                                      __HIP_MEMORY_SCOPE_AGENT);
  previous = __shfl(previous, 0, 32);
  if (previous + 1 != static_cast<unsigned>(splits))
    return;
  // Every lane acquires the counter the other waves released.
  (void)__hip_atomic_load(counters + tile, __ATOMIC_ACQUIRE,
                          __HIP_MEMORY_SCOPE_AGENT);
  if (n < n_size) {
#pragma unroll
    for (int t = 0; t < kMTiles; ++t) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int m = t * 16 + 2 * i + half;
        if (m >= m_size)
          continue;
        float total = 0.0F;
        for (int s = 0; s < splits; ++s)
          total += __hip_atomic_load(
              partials + (static_cast<std::size_t>(s) * m_size + m) * n_size +
                  n,
              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        float* target = y + static_cast<std::size_t>(m) * n_size + n;
        *target = accumulate ? *target + total : total;
      }
    }
  }
  if (lane == 0)
    __hip_atomic_store(counters + tile, 0U, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
}

template<bool kQ8>
bool Launch(const Bf16* x, const std::uint8_t* w, float* y, std::size_t rows,
            std::size_t in, std::size_t out, bool accumulate,
            const WmmaPlan& plan, const SplitScratch& scratch,
            hipStream_t stream) {
  const std::size_t tiles = (out + 15) / 16;
  // The plan must cover the reduction, and its scratch this product.
  if (rows == 0 || rows > 32 || in % 32 != 0 || plan.splits == 0 ||
      plan.blocks_per_split == 0 ||
      plan.splits * plan.blocks_per_split < in / 32)
    return false;
  if (plan.splits > 1 &&
      (scratch.partial_floats < plan.splits * rows * out ||
       scratch.counter_count < tiles))
    return false;
  const dim3 grid(static_cast<unsigned>((tiles + kWaves - 1) / kWaves),
                  static_cast<unsigned>(plan.splits));
  const auto launch = [&](auto kernel) {
    kernel<<<grid, kWaves * 32, 0, stream>>>(
        x, w, y, scratch.partials, scratch.counters, static_cast<int>(rows),
        static_cast<int>(in), static_cast<int>(out),
        static_cast<int>(plan.splits),
        static_cast<int>(plan.blocks_per_split), accumulate);
  };
  if (rows <= 16)
    launch(WmmaKernel<kQ8, 1>);
  else
    launch(WmmaKernel<kQ8, 2>);
  return true;
}

}  // namespace

void Q8_0ToTiles(const std::uint8_t* blocks, std::size_t rows, std::size_t in,
                 std::uint8_t* output, hipStream_t stream) {
  const std::size_t count = rows * (in / 32);
  if (count > 0)
    Q8_0ToTilesKernel<<<BlocksOf256(count), 256, 0, stream>>>(
        blocks, static_cast<int>(rows), static_cast<int>(in), output);
}

void Bf16ToTiles(const Bf16* weights, std::size_t rows, std::size_t in,
                 Bf16* output, hipStream_t stream) {
  const std::size_t count = rows * in;
  if (count > 0)
    Bf16ToTilesKernel<<<BlocksOf256(count), 256, 0, stream>>>(
        weights, static_cast<int>(rows), static_cast<int>(in), output);
}

void Q8TilesToBf16(const std::uint8_t* tiles, std::size_t rows,
                   std::size_t in, Bf16* output, hipStream_t stream) {
  const std::size_t count = rows * in;
  if (count > 0)
    Q8TilesToBf16Kernel<<<BlocksOf256(count), 256, 0, stream>>>(
        tiles, static_cast<int>(rows), static_cast<int>(in), output);
}

void Bf16TilesToRows(const Bf16* tiles, std::size_t rows, std::size_t in,
                     Bf16* output, hipStream_t stream) {
  const std::size_t count = rows * in;
  if (count > 0)
    Bf16TilesToRowsKernel<<<BlocksOf256(count), 256, 0, stream>>>(
        tiles, static_cast<int>(rows), static_cast<int>(in), output);
}

WmmaPlan PlanWmma(std::size_t rows, std::size_t in, std::size_t out,
                  std::size_t target_waves) {
  // Measured with tools/bench/gemm_bench.cpp: fewer, longer waves suit the
  // DiT's 22 rows (two activation tiles per wave), more suit 1-16 rows.
  if (target_waves == 0)
    target_waves = rows > 16 ? 160 : 640;
  WmmaPlan plan;
  const std::size_t tiles = (out + 15) / 16;
  const std::size_t k_blocks = in / 32;
  if (tiles == 0 || k_blocks == 0)
    return plan;
  std::size_t splits = (target_waves + tiles - 1) / tiles;
  splits = std::max<std::size_t>(
      1, std::min(splits, k_blocks / std::min(kMinBlocks, k_blocks)));
  plan.blocks_per_split = (k_blocks + splits - 1) / splits;
  plan.splits = (k_blocks + plan.blocks_per_split - 1) / plan.blocks_per_split;
  if (plan.splits > 1) {
    plan.partial_floats = plan.splits * rows * out;
    plan.counters = tiles;
  }
  return plan;
}

bool WmmaQ8(const Bf16* x, const std::uint8_t* w, float* y, std::size_t rows,
            std::size_t in, std::size_t out, bool accumulate,
            const WmmaPlan& plan, const SplitScratch& scratch,
            hipStream_t stream) {
  return Launch<true>(x, w, y, rows, in, out, accumulate, plan, scratch,
                      stream);
}

bool WmmaBf16(const Bf16* x, const Bf16* w, float* y, std::size_t rows,
              std::size_t in, std::size_t out, bool accumulate,
              const WmmaPlan& plan, const SplitScratch& scratch,
              hipStream_t stream) {
  return Launch<false>(x, reinterpret_cast<const std::uint8_t*>(w), y, rows,
                       in, out, accumulate, plan, scratch, stream);
}

}  // namespace gufo::models::voxcpm::kernels
