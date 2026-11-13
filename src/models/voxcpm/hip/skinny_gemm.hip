#include <hip/hip_runtime.h>

#include <cstdint>

#include "skinny_gemm.hpp"

namespace gufo::models::voxcpm::kernels {
namespace {

constexpr int kWaves = 8;
constexpr int kThreads = kWaves * 32;

/// a.lo * b.lo + a.hi * b.hi + c for two packed bf16 pairs, with the
/// products and sums in float32 (v_dot2_f32_bf16).
__device__ inline float Dot2(std::uint32_t a, std::uint32_t b, float c) {
  using Pair = __bf16 __attribute__((ext_vector_type(2)));
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(Pair, a),
                                         __builtin_bit_cast(Pair, b), c,
                                         false);
}

/// The dot product of 8 packed bf16 values each, added to `c`.
__device__ inline float Dot8(const uint4& a, const uint4& b, float c) {
  c = Dot2(a.x, b.x, c);
  c = Dot2(a.y, b.y, c);
  c = Dot2(a.z, b.z, c);
  return Dot2(a.w, b.w, c);
}

__device__ inline float WaveSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2)
    value += __shfl_xor(value, offset, 32);
  return value;
}

// One wave per output column n (a weight row) for all `rows` activation
// rows: every weight is read once from memory. A lane reads kVec
// consecutive 16-byte pieces of the row per step (a chunk of 256 * kVec
// elements per wave); the activations of the chunk are staged in LDS for
// the block's eight waves.
template<int kMaxRows, int kVec>
__global__ __launch_bounds__(kThreads) void SkinnyGemmKernel(
    const Bf16* x, const Bf16* w, float* y, int rows, int k_size, int n_size,
    bool accumulate) {
  constexpr int kChunk = 32 * 8 * kVec;
  constexpr int kPieces = 32 * kVec;  // uint4 per row and chunk
  __shared__ uint4 staged[kMaxRows > 1 ? kMaxRows * kPieces : 1024];
  if constexpr (kMaxRows == 1) {
    for (int index = threadIdx.x; index < k_size / 8; index += kThreads)
      staged[index] = reinterpret_cast<const uint4*>(x)[index];
    __syncthreads();
  }
  const int lane = threadIdx.x % 32;
  const int wave = threadIdx.x / 32;
  const int n = min(static_cast<int>(blockIdx.x) * kWaves + wave, n_size - 1);
  float sums[kMaxRows];
#pragma unroll
  for (int m = 0; m < kMaxRows; ++m)
    sums[m] = 0.0F;
  const Bf16* weight_row = w + static_cast<std::size_t>(n) * k_size;
  for (int k0 = 0; k0 < k_size; k0 += kChunk) {
    if constexpr (kMaxRows > 1) {
      for (int index = threadIdx.x; index < rows * kPieces;
           index += kThreads) {
        const int m = index / kPieces;
        const int k = k0 + (index % kPieces) * 8;
        staged[index] =
            k < k_size
                ? *reinterpret_cast<const uint4*>(
                      x + static_cast<std::size_t>(m) * k_size + k)
                : make_uint4(0, 0, 0, 0);
      }
      __syncthreads();
    }
    uint4 packed[kVec];
#pragma unroll
    for (int v = 0; v < kVec; ++v) {
      const int k = k0 + (v * 32 + lane) * 8;
      packed[v] = k < k_size
                      ? *reinterpret_cast<const uint4*>(weight_row + k)
                      : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int v = 0; v < kVec; ++v) {
      const int k = k0 + (v * 32 + lane) * 8;
      if (k >= k_size)
        break;
#pragma unroll
      for (int m = 0; m < kMaxRows; ++m) {
        if (m < rows) {
          const uint4 values = kMaxRows > 1
                                   ? staged[m * kPieces + v * 32 + lane]
                                   : staged[k / 8];
          sums[m] = Dot8(packed[v], values, sums[m]);
        }
      }
    }
    if constexpr (kMaxRows > 1)
      __syncthreads();
  }
  const bool owner = static_cast<int>(blockIdx.x) * kWaves + wave < n_size;
#pragma unroll
  for (int m = 0; m < kMaxRows; ++m) {
    if (m < rows) {
      const float total = WaveSum(sums[m]);
      if (lane == 0 && owner) {
        float* target = y + static_cast<std::size_t>(m) * n_size + n;
        *target = accumulate ? *target + total : total;
      }
    }
  }
}

template<int kMaxRows, int kVec>
void Launch(const Bf16* x, const Bf16* w, float* y, std::size_t rows,
            std::size_t in, std::size_t out, bool accumulate,
            hipStream_t stream) {
  const dim3 blocks(static_cast<unsigned>((out + kWaves - 1) / kWaves));
  SkinnyGemmKernel<kMaxRows, kVec><<<blocks, kThreads, 0, stream>>>(
      x, w, y, static_cast<int>(rows), static_cast<int>(in),
      static_cast<int>(out), accumulate);
}

template<int kMaxRows>
bool LaunchWith(int vec, const Bf16* x, const Bf16* w, float* y,
                std::size_t rows, std::size_t in, std::size_t out,
                bool accumulate, hipStream_t stream) {
  switch (vec) {
    case 1:
      Launch<kMaxRows, 1>(x, w, y, rows, in, out, accumulate, stream);
      return true;
    case 2:
      Launch<kMaxRows, 2>(x, w, y, rows, in, out, accumulate, stream);
      return true;
    case 4:
      if constexpr (kMaxRows <= 16) {
        Launch<kMaxRows, 4>(x, w, y, rows, in, out, accumulate, stream);
        return true;
      }
      return false;
    default:
      return false;
  }
}

// Each block sweeps a contiguous block of kRows weight rows like a plain
// read (every step of the block reads 4 KB in one piece) and multiplies
// them with up to kMaxM activation rows, whose slices stay packed in
// registers. A weight row of P = K / 8 16-byte pieces is read by P threads
// (P <= 256, 256 / P rows per step) or by all 256 threads in kSlices steps
// (P = 256 * kSlices). x and w rows are `ldx` and `ldw` elements apart, so a
// product can run over K in several passes.
template<int kMaxM, int kRows, int kSlices>
__global__ __launch_bounds__(kThreads) void SweepGemmKernel(
    const Bf16* x, int ldx, const Bf16* w, int ldw, float* y, int m_size,
    int k_size, int n_size, bool accumulate) {
  __shared__ float partial[kWaves][kRows][kMaxM];
  const int pieces = k_size / 8;
  const int width = pieces < kThreads ? pieces : kThreads;  // threads/row
  const int per_step = kThreads / width;
  const int group = threadIdx.x / width;
  const int piece = threadIdx.x % width;
  uint4 xs[kMaxM][kSlices];
#pragma unroll
  for (int m = 0; m < kMaxM; ++m) {
    if (m < m_size) {
      const uint4* row = reinterpret_cast<const uint4*>(
          x + static_cast<std::size_t>(m) * ldx);
#pragma unroll
      for (int s = 0; s < kSlices; ++s)
        xs[m][s] = row[piece + s * kThreads];
    }
  }
  const int first = static_cast<int>(blockIdx.x) * kRows;
  const int steps = kRows / per_step;
  uint4 packed[kRows][kSlices];
#pragma unroll
  for (int i = 0; i < kRows; ++i) {
    if (i < steps) {
      const int row = min(first + i * per_step + group, n_size - 1);
      const uint4* row4 = reinterpret_cast<const uint4*>(
          w + static_cast<std::size_t>(row) * ldw);
#pragma unroll
      for (int s = 0; s < kSlices; ++s)
        packed[i][s] = row4[piece + s * kThreads];
    }
  }
  float sums[kRows][kMaxM];
#pragma unroll
  for (int i = 0; i < kRows; ++i)
#pragma unroll
    for (int m = 0; m < kMaxM; ++m)
      sums[i][m] = 0.0F;
#pragma unroll
  for (int i = 0; i < kRows; ++i) {
    if (i >= steps)
      continue;
#pragma unroll
    for (int s = 0; s < kSlices; ++s) {
#pragma unroll
      for (int m = 0; m < kMaxM; ++m) {
        if (m < m_size)
          sums[i][m] = Dot8(packed[i][s], xs[m][s], sums[i][m]);
      }
    }
  }
  // Sum each row over its `width` threads: lanes first, then waves.
  const int lanes = width < 32 ? width : 32;
  const int lane = threadIdx.x % 32;
  const int wave = threadIdx.x / 32;
#pragma unroll
  for (int i = 0; i < kRows; ++i) {
    if (i >= steps)
      continue;
    const int row = first + i * per_step + group;
#pragma unroll
    for (int m = 0; m < kMaxM; ++m) {
      if (m >= m_size)
        continue;
      float value = sums[i][m];
      for (int offset = lanes / 2; offset > 0; offset /= 2)
        value += __shfl_xor(value, offset, 32);
      if (width <= 32) {
        if (piece == 0 && row < n_size) {
          float* target = y + static_cast<std::size_t>(m) * n_size + row;
          *target = accumulate ? *target + value : value;
        }
      } else if (lane == 0) {
        partial[wave][i][m] = value;
      }
    }
  }
  if (width > 32) {
    __syncthreads();
    const int waves_per_row = width / 32;
    for (int index = threadIdx.x; index < kRows * m_size;
         index += kThreads) {
      const int m = index % m_size;
      const int r = index / m_size;  // row within the block
      const int i = r / per_step;
      const int g = r % per_step;
      float value = 0.0F;
      for (int v = 0; v < waves_per_row; ++v)
        value += partial[g * waves_per_row + v][i][m];
      const int row = first + r;
      if (row < n_size) {
        float* target = y + static_cast<std::size_t>(m) * n_size + row;
        *target = accumulate ? *target + value : value;
      }
    }
  }
}

template<int kMaxM, int kRows, int kSlices>
void LaunchSweep(const Bf16* x, int ldx, const Bf16* w, int ldw, float* y,
                 std::size_t rows, std::size_t in, std::size_t out,
                 bool accumulate, hipStream_t stream) {
  const dim3 blocks(static_cast<unsigned>((out + kRows - 1) / kRows));
  SweepGemmKernel<kMaxM, kRows, kSlices><<<blocks, kThreads, 0, stream>>>(
      x, ldx, w, ldw, y, static_cast<int>(rows), static_cast<int>(in),
      static_cast<int>(out), accumulate);
}

template<int kMaxM, int kRows>
bool SweepPass(int slices, const Bf16* x, int ldx, const Bf16* w, int ldw,
               float* y, std::size_t rows, std::size_t in, std::size_t out,
               bool accumulate, hipStream_t stream) {
  switch (slices) {
    case 1:
      LaunchSweep<kMaxM, kRows, 1>(x, ldx, w, ldw, y, rows, in, out,
                                   accumulate, stream);
      return true;
    case 2:
      if constexpr (kMaxM <= 8) {
        LaunchSweep<kMaxM, kRows, 2>(x, ldx, w, ldw, y, rows, in, out,
                                     accumulate, stream);
        return true;
      }
      return false;
    case 3:
      if constexpr (kMaxM <= 8) {
        LaunchSweep<kMaxM, kRows, 3>(x, ldx, w, ldw, y, rows, in, out,
                                     accumulate, stream);
        return true;
      }
      return false;
    default:
      return false;
  }
}

/// The sweep, in passes of 2048 K elements when the activation slices
/// would not fit the registers. False when the shape doesn't suit it.
template<int kMaxM, int kRows>
bool SweepFor(const Bf16* x, const Bf16* w, float* y, std::size_t rows,
              std::size_t in, std::size_t out, bool accumulate,
              hipStream_t stream) {
  const std::size_t pieces = in / 8;
  std::size_t slices = 0;
  if (pieces <= kThreads && kThreads % pieces == 0)
    slices = 1;
  else if (pieces % kThreads == 0)
    slices = pieces / kThreads;
  if (slices == 0)
    return false;
  const std::size_t per_step = pieces < kThreads ? kThreads / pieces : 1;
  if (per_step > kRows || kRows % per_step != 0)
    return false;
  const std::size_t budget = kMaxM <= 8 ? 3 : 1;  // slices per pass
  if (slices <= budget)
    return SweepPass<kMaxM, kRows>(static_cast<int>(slices), x,
                                   static_cast<int>(in), w,
                                   static_cast<int>(in), y, rows, in, out,
                                   accumulate, stream);
  const std::size_t pass = budget * kThreads * 8;
  for (std::size_t k = 0; k < in; k += pass)
    SweepPass<kMaxM, kRows>(static_cast<int>(budget), x + k,
                            static_cast<int>(in), w + k, static_cast<int>(in),
                            y, rows, pass, out, accumulate || k > 0, stream);
  return true;
}

bool Sweep(const Bf16* x, const Bf16* w, float* y, std::size_t rows,
           std::size_t in, std::size_t out, bool accumulate,
           hipStream_t stream) {
  if (rows == 1)
    return SweepFor<1, 4>(x, w, y, rows, in, out, accumulate, stream);
  if (rows <= 4)
    return SweepFor<4, 4>(x, w, y, rows, in, out, accumulate, stream);
  if (rows <= 8)
    return SweepFor<8, 4>(x, w, y, rows, in, out, accumulate, stream);
  if (rows <= 16)
    return SweepFor<16, 2>(x, w, y, rows, in, out, accumulate, stream);
  if (rows <= 24)
    return SweepFor<24, 2>(x, w, y, rows, in, out, accumulate, stream);
  return SweepFor<32, 2>(x, w, y, rows, in, out, accumulate, stream);
}

__global__ void ReadProbeKernel(const uint4* data, std::size_t count,
                                unsigned* sink) {
  unsigned value = 0;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const uint4 v = data[i];
    value ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (value == 0x12345678U)
    *sink = value;
}

}  // namespace

void ReadProbe(const Bf16* data, std::size_t pieces, unsigned blocks,
               unsigned* sink, hipStream_t stream) {
  ReadProbeKernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const uint4*>(data), pieces, sink);
}

bool SkinnyGemmWith(int vec, const Bf16* x, const Bf16* w, float* y,
                    std::size_t rows, std::size_t in, std::size_t out,
                    bool accumulate, hipStream_t stream) {
  if (rows == 0 || rows > kSkinnyRows || in % 8 != 0 || in > 8192)
    return false;
  if (vec == 0)
    return Sweep(x, w, y, rows, in, out, accumulate, stream);
  if (rows == 1)
    return LaunchWith<1>(vec, x, w, y, rows, in, out, accumulate, stream);
  if (rows <= 8)
    return LaunchWith<8>(vec, x, w, y, rows, in, out, accumulate, stream);
  if (rows <= 16)
    return LaunchWith<16>(vec, x, w, y, rows, in, out, accumulate, stream);
  if (rows <= 24)
    return LaunchWith<24>(vec, x, w, y, rows, in, out, accumulate, stream);
  return LaunchWith<32>(vec, x, w, y, rows, in, out, accumulate, stream);
}

bool SkinnyGemm(const Bf16* x, const Bf16* w, float* y, std::size_t rows,
                std::size_t in, std::size_t out, bool accumulate,
                hipStream_t stream) {
  // Fastest measured per row count (tools/bench/gemm_bench.cpp): the sweep
  // up to 8 rows, where the activation slices fit the registers, then a
  // wave per weight row.
  if (rows <= 8 && in <= 8192 &&
      SkinnyGemmWith(0, x, w, y, rows, in, out, accumulate, stream))
    return true;
  return SkinnyGemmWith(rows <= 16 ? 4 : 1, x, w, y, rows, in, out,
                        accumulate, stream);
}

}  // namespace gufo::models::voxcpm::kernels
