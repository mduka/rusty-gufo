// Matrix-core products, following the bfloat16/Q8_0 products of the VoxCPM
// extension by the same author.
#include "wmma.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>

#include "wmma_fragment.hpp"

namespace gufo::models::nemotron_asr::kernels {
namespace {

constexpr int kWaves = 4;
constexpr std::size_t kTileHalves = 16 * 32;
// Waves a product aims for: from 384 to 512 the encoder's products of 140
// and 224 rows took the least time, fewer waves leave SIMDs idle, more read
// the weights more often.
constexpr std::size_t kTargetWaves = 384;

__device__ inline Fragment FromWords(const uint4& lo, const uint4& hi) {
  union {
    Fragment v;
    uint4 u[2];
  } cvt;
  cvt.u[0] = lo;
  cvt.u[1] = hi;
  return cvt.v;
}

__global__ void HalfToTilesKernel(const std::uint16_t* weights, int rows,
                                  int in, std::uint16_t* tiles) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(rows) * in)
    return;
  const int row = static_cast<int>(index / in);
  const int k = static_cast<int>(index % in);
  const int per_row = in / 32;
  tiles[(static_cast<std::size_t>(row / 16) * per_row + k / 32) * kTileHalves +
        (row % 16) * 32 + k % 32] = weights[index];
}

// One wave per 16 weight rows (output columns n) and kTiles tiles of 16
// activation rows (the group blockIdx.y); the block's four waves share the
// activations, which they stage block by block of 32 columns in LDS
// (double-buffered). A (16 x 16) is 16 activation rows, B the 16 weight
// rows: lane L holds activation row L % 16 and weight row L % 16 with 16
// consecutive k, replicated in both half-waves; element i of lane L's sums
// is activation row 2i + L / 16 and weight row L % 16. Each output is one
// wave's sum over the whole reduction in order. Activation rows beyond
// `rows` repeat the last row; their results are dropped. The sums are
// scaled, added to y (`accumulate`), then get the bias and ReLU; kHalf
// stores them as float16 (saturated), without accumulation.
constexpr int kStride = 40;  // halves per LDS row: 32 and padding

template<int kTiles, bool kHalf>
__global__ __launch_bounds__(kWaves * 32) void WmmaKernel(
    const std::uint16_t* x, const std::uint16_t* w, void* y, int rows,
    int in, int out, bool accumulate, float scale, const float* bias,
    bool relu) {
  constexpr int kRows = kTiles * 16;
  // Each thread stages kChunks pieces of 16 bytes of a block's activations
  // ([kRows][32] halves = kRows * 4 pieces).
  constexpr int kChunks = (kRows * 4 + kWaves * 32 - 1) / (kWaves * 32);
  __shared__ __align__(16) std::uint16_t staged[2][kRows * kStride];
  const int lane = threadIdx.x % 32;
  const int tile = static_cast<int>(blockIdx.x) * kWaves + threadIdx.x / 32;
  const bool active = tile < out / 16;
  const int first_row = static_cast<int>(blockIdx.y) * kRows;
  const int k_blocks = in / 32;
  const int column = lane % 16;
  const int n = tile * 16 + column;
  const std::uint16_t* lane_weights =
      w + static_cast<std::size_t>(active ? tile : 0) * k_blocks * kTileHalves +
      column * 32;

  // Each thread's pieces of the activations and the lane's 64 weight bytes
  // of the next block, requested before this block's products. Without
  // branches around them the pieces stay in registers: threads beyond the
  // last piece repeat it (the same bytes to the same place), and the last
  // block is requested twice.
  int piece_offset[kChunks];
  int piece_lds[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    const int piece =
        min(static_cast<int>(threadIdx.x) + c * kWaves * 32, kRows * 4 - 1);
    piece_offset[c] =
        min(first_row + piece / 4, rows - 1) * in + (piece % 4) * 8;
    piece_lds[c] = (piece / 4) * kStride + (piece % 4) * 8;
  }
  const auto* weights4 = reinterpret_cast<const uint4*>(lane_weights);
  uint4 pieces[kChunks];
  uint4 next[4];

  Accumulator acc[kTiles];
#pragma unroll
  for (int t = 0; t < kTiles; ++t)
    acc[t] = Accumulator{};
#pragma unroll
  for (int c = 0; c < kChunks; ++c)
    pieces[c] = *reinterpret_cast<const uint4*>(x + piece_offset[c]);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    next[i] = weights4[i];
#pragma unroll
  for (int c = 0; c < kChunks; ++c)
    *reinterpret_cast<uint4*>(&staged[0][piece_lds[c]]) = pieces[c];
  __syncthreads();
  for (int b = 0; b < k_blocks; ++b) {
    const int buffer = b & 1;
    const Fragment b0 = FromWords(next[0], next[1]);
    const Fragment b1 = FromWords(next[2], next[3]);
    const int following = min(b + 1, k_blocks - 1);
#pragma unroll
    for (int c = 0; c < kChunks; ++c)
      pieces[c] =
          *reinterpret_cast<const uint4*>(x + piece_offset[c] + following * 32);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      next[i] = weights4[following * (kTileHalves / 8) + i];
#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
      const std::uint16_t* row = &staged[buffer][(t * 16 + column) * kStride];
      acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(LoadFragment(row),
                                                          b0, acc[t]);
      acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
          LoadFragment(row + 16), b1, acc[t]);
    }
#pragma unroll
    for (int c = 0; c < kChunks; ++c)
      *reinterpret_cast<uint4*>(&staged[buffer ^ 1][piece_lds[c]]) = pieces[c];
    __syncthreads();
  }
  if (!active)
    return;
  const int half = lane / 16;
  const float shift = bias != nullptr ? bias[n] : 0.0F;
#pragma unroll
  for (int t = 0; t < kTiles; ++t) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int m = first_row + t * 16 + 2 * i + half;
      if (m < rows) {
        const std::size_t at = static_cast<std::size_t>(m) * out + n;
        if constexpr (kHalf) {
          float value = scale * acc[t][i] + shift;
          if (relu)
            value = fmaxf(value, 0.0F);
          static_cast<__half*>(y)[at] = SaturatedHalf(value);
        } else {
          float* target = static_cast<float*>(y) + at;
          float value =
              accumulate ? fmaf(scale, acc[t][i], *target) : scale * acc[t][i];
          if (bias != nullptr)
            value += shift;
          if (relu)
            value = fmaxf(value, 0.0F);
          *target = value;
        }
      }
    }
  }
}

template<int kTiles, bool kHalf>
void Launch(const std::uint16_t* x, const std::uint16_t* w, void* y,
            std::size_t rows, std::size_t in, std::size_t out,
            bool accumulate, float scale, const float* bias, bool relu,
            hipStream_t stream) {
  const std::size_t tiles = out / 16;
  const std::size_t groups = (rows + kTiles * 16 - 1) / (kTiles * 16);
  const dim3 grid(static_cast<unsigned>((tiles + kWaves - 1) / kWaves),
                  static_cast<unsigned>(groups));
  WmmaKernel<kTiles, kHalf><<<grid, kWaves * 32, 0, stream>>>(
      x, w, y, static_cast<int>(rows), static_cast<int>(in),
      static_cast<int>(out), accumulate, scale, bias, relu);
}

template<bool kHalf>
bool Dispatch(const std::uint16_t* x, const std::uint16_t* tiles, void* y,
              std::size_t rows, std::size_t in, std::size_t out,
              bool accumulate, float scale, const float* bias, bool relu,
              hipStream_t stream) {
  if (rows == 0 || in % 32 != 0 || out % 16 != 0 || out == 0 ||
      (kHalf && accumulate))
    return false;
  // Activation tiles per wave: as many as keep about kTargetWaves waves
  // busy, at most 8 (registers).
  const std::size_t row_tiles = (rows + 15) / 16;
  const std::size_t weight_tiles = out / 16;
  std::size_t per_wave = std::clamp<std::size_t>(
      (row_tiles * weight_tiles + kTargetWaves - 1) / kTargetWaves, 1, 8);
  // Even groups: the fewest tiles per wave for the same number of groups.
  const std::size_t groups = (row_tiles + per_wave - 1) / per_wave;
  per_wave = (row_tiles + groups - 1) / groups;
  switch (per_wave) {
#define GUFO_NEMOTRON_ASR_WMMA(n)                                          \
  case n:                                                                  \
    Launch<n, kHalf>(x, tiles, y, rows, in, out, accumulate, scale, bias,  \
                     relu, stream);                                        \
    return true;
    GUFO_NEMOTRON_ASR_WMMA(1)
    GUFO_NEMOTRON_ASR_WMMA(2)
    GUFO_NEMOTRON_ASR_WMMA(3)
    GUFO_NEMOTRON_ASR_WMMA(4)
    GUFO_NEMOTRON_ASR_WMMA(5)
    GUFO_NEMOTRON_ASR_WMMA(6)
    GUFO_NEMOTRON_ASR_WMMA(7)
    GUFO_NEMOTRON_ASR_WMMA(8)
#undef GUFO_NEMOTRON_ASR_WMMA
    default:
      return false;
  }
}

}  // namespace

void HalfToTiles(const std::uint16_t* weights, std::size_t rows,
                 std::size_t in, std::uint16_t* tiles, hipStream_t stream) {
  const std::size_t count = rows * in;
  if (count > 0)
    HalfToTilesKernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0,
                        stream>>>(weights, static_cast<int>(rows),
                                  static_cast<int>(in), tiles);
}

bool WmmaLinear(const std::uint16_t* x, const std::uint16_t* tiles, float* y,
                std::size_t rows, std::size_t in, std::size_t out,
                bool accumulate, float scale, hipStream_t stream,
                const float* bias, bool relu) {
  return Dispatch<false>(x, tiles, y, rows, in, out, accumulate, scale, bias,
                         relu, stream);
}

bool WmmaLinearHalf(const std::uint16_t* x, const std::uint16_t* tiles,
                    std::uint16_t* y, std::size_t rows, std::size_t in,
                    std::size_t out, const float* bias, bool relu,
                    hipStream_t stream) {
  return Dispatch<true>(x, tiles, y, rows, in, out, false, 1.0F, bias, relu,
                        stream);
}

}  // namespace gufo::models::nemotron_asr::kernels
