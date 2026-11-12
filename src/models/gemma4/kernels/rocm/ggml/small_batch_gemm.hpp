// Excerpt of Gufo src/families/qwen/kernels/small_batch_gemm.hpp at upstream
// 1c75887b (gufo-org/gufo): the paths Gemma 4 dispatches for the formats its
// loader accepts, in the gemma4 extension namespace.
#ifndef GUFO_MODELS_GEMMA4_HIP_GGML_SMALL_BATCH_GEMM_HPP_
#define GUFO_MODELS_GEMMA4_HIP_GGML_SMALL_BATCH_GEMM_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "src/models/gemma4/kernels/rocm/ggml/quant_ops.hpp"

namespace gufo::models::gemma4::ggml {

// Bound weight byte offsets and activation/output element offsets before
// selecting 32-bit device indexing. Division avoids overflow in the check.
// Batch >= 2 also leaves room for the rounded-up final row/tile indices.
constexpr bool FitsSmallBatch32BitIndices(std::size_t batch, std::size_t m,
                                          std::size_t k,
                                          std::size_t row_bytes) noexcept {
  constexpr auto limit = std::numeric_limits<std::uint32_t>::max();
  return batch >= 2 && batch <= 8 && m != 0 && k != 0 && row_bytes != 0 &&
         m <= limit / batch && k <= limit / batch && m <= limit / row_bytes;
}

// Immediate XOR masks avoid the lane-address calculation of __shfl_xor.
// Keep the descending butterfly order identical to scalar decoding.
template<int Offset = 16>
__device__ __forceinline__ float ReduceKQuantWave(float value) {
  const int other = __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, value),
                                                (Offset << 10) | 31);
  const float sum = value + __builtin_bit_cast(float, other);
  if constexpr (Offset > 1) {
    return ReduceKQuantWave<Offset / 2>(sum);
  } else {
    return sum;
  }
}

// These GEMMs exchange data through LDS only. Global inputs are read-only,
// and output stores follow the final tile. Complete LDS traffic and preserve
// compiler memory ordering without invalidating the global read cache.
__device__ __forceinline__ __attribute__((convergent)) void SyncKQuantTile() {
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_s_barrier();
  asm volatile("" ::: "memory");
}

// Exact FP32-activation Q8_0 kernel for decoding and speculative
// verification. Each lane visits its blocks in the same order at every batch
// width; tile geometry never changes arithmetic.
template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave, bool NarrowIndex = false,
         bool XorStage = false>
__launch_bounds__(WavesPerBlock * 32, 1) __global__
    void SmallBatchQ8_0ExactFp32VecGEMMKernel(const void* __restrict__ w,
                                              const float* __restrict__ x,
                                              float* __restrict__ y,
                                              std::size_t wide_m,
                                              std::size_t wide_k) {
  using Index = std::conditional_t<NarrowIndex, std::uint32_t, std::size_t>;
  const Index m = static_cast<Index>(wide_m);
  const Index k = static_cast<Index>(wide_k);
  constexpr Index kBlocksPerTile = 32;
  constexpr Index kVectorsPerBlock = kQ8_0BlockSize / 4;
  // Transpose the vector groups and XOR their block indices to fit sixteen
  // activation rows in 64 KiB of LDS without changing the dot-product order.
  constexpr Index kStride = kQ8_0BlockSize + (XorStage ? 0 : 4);
  constexpr Index kTileStride = kBlocksPerTile * kStride;
  __shared__ float staged_x[Batch * kTileStride];

  const Index lane = threadIdx.x & 31u;
  const Index warp_id = threadIdx.x >> 5u;
  const Index row_base =
      ((static_cast<std::size_t>(blockIdx.x) * WavesPerBlock) + warp_id) *
      RowsPerWave;
  const Index num_blocks = k / kQ8_0BlockSize;
  const auto* base = static_cast<const Q8_0Block*>(w);
  float sums[RowsPerWave][Batch] = {};

  for (Index tile_base = 0; tile_base < num_blocks;
       tile_base += kBlocksPerTile) {
    constexpr Index kTileVectors = Batch * kBlocksPerTile * kVectorsPerBlock;
    for (Index flat = threadIdx.x; flat < kTileVectors; flat += blockDim.x) {
      const Index token = flat / (kBlocksPerTile * kVectorsPerBlock);
      const Index within = flat % (kBlocksPerTile * kVectorsPerBlock);
      const Index block = within / kVectorsPerBlock;
      const Index vector = within % kVectorsPerBlock;
      const Index source_block = tile_base + block;
      float4 value = {0.0F, 0.0F, 0.0F, 0.0F};
      if (source_block < num_blocks) {
        value = *reinterpret_cast<const float4*>(
            x + (token * k) + (source_block * kQ8_0BlockSize) + (vector * 4));
      }
      *reinterpret_cast<float4*>(staged_x + (token * kTileStride) +
                                 (XorStage ? vector * 128 + (block ^ vector) * 4
                                           : block * kStride + vector * 4)) =
          value;
    }
    if constexpr (NarrowIndex) {
      SyncKQuantTile();
    } else {
      __syncthreads();
    }

    const Index block = tile_base + lane;
    if (block < num_blocks) {
      const Q8_0Block* rows[RowsPerWave];
      float scale[RowsPerWave];
#pragma unroll
      for (Index r = 0; r < RowsPerWave; ++r) {
        const Index row = row_base + r;
        const Index safe_row = row < m ? row : m - 1;
        rows[r] = &base[(safe_row * num_blocks) + block];
        scale[r] = __half2float(rows[r]->d);
      }
      float block_dots[RowsPerWave][Batch] = {};
#pragma unroll
      for (Index group = 0; group < kVectorsPerBlock; ++group) {
        std::uint32_t packed[RowsPerWave];
#pragma unroll
        for (Index r = 0; r < RowsPerWave; ++r) {
          __builtin_memcpy(&packed[r], rows[r]->qs + group * 4,
                           sizeof(std::uint32_t));
        }
#pragma unroll
        for (Index token = 0; token < Batch; ++token) {
          const float4 xv = *reinterpret_cast<const float4*>(
              staged_x + token * kTileStride +
              (XorStage ? group * 128 + (lane ^ group) * 4
                        : lane * kStride + group * 4));
#pragma unroll
          for (Index r = 0; r < RowsPerWave; ++r) {
            const std::uint32_t p = packed[r];
            block_dots[r][token] +=
                static_cast<float>(static_cast<std::int8_t>(p & 0xFFU)) * xv.x;
            block_dots[r][token] += static_cast<float>(static_cast<std::int8_t>(
                                        (p >> 8U) & 0xFFU)) *
                                    xv.y;
            block_dots[r][token] += static_cast<float>(static_cast<std::int8_t>(
                                        (p >> 16U) & 0xFFU)) *
                                    xv.z;
            block_dots[r][token] += static_cast<float>(static_cast<std::int8_t>(
                                        (p >> 24U) & 0xFFU)) *
                                    xv.w;
          }
        }
      }
#pragma unroll
      for (Index r = 0; r < RowsPerWave; ++r) {
#pragma unroll
        for (Index token = 0; token < Batch; ++token) {
          sums[r][token] += scale[r] * block_dots[r][token];
        }
      }
    }
    if constexpr (NarrowIndex) {
      SyncKQuantTile();
    } else {
      __syncthreads();
    }
  }

#pragma unroll
  for (Index r = 0; r < RowsPerWave; ++r) {
#pragma unroll
    for (Index token = 0; token < Batch; ++token) {
      if constexpr (NarrowIndex) {
        sums[r][token] = ReduceKQuantWave(sums[r][token]);
      } else {
        for (int offset = 16; offset > 0; offset >>= 1) {
          sums[r][token] += __shfl_xor(sums[r][token], offset);
        }
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (Index r = 0; r < RowsPerWave; ++r) {
      const Index row = row_base + r;
      if (row >= m) {
        continue;
      }
#pragma unroll
      for (Index token = 0; token < Batch; ++token) {
        y[(token * m) + row] = sums[r][token];
      }
    }
  }
}

// Exact FP32-activation kernel for the sub-16-decoded formats (Q4_K, Q5_K,
// Q6_K and Q4_0). Each lane decodes one sixteen-value sub-block per tile slot
// and visits its sub-blocks in the scalar GEMV's order at every batch width.
template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave, core::GgmlType WType,
         std::size_t TilesPerStage = 1, std::uint32_t MinWaves = 12,
         bool NarrowIndex = false>
__launch_bounds__(WavesPerBlock * 32, MinWaves) __global__
    void SmallBatchKQuantExactFp32GEMMKernel(const void* __restrict__ w,
                                             const float* __restrict__ x,
                                             float* __restrict__ y,
                                             std::size_t wide_m,
                                             std::size_t wide_k) {
  using Index = std::conditional_t<NarrowIndex, std::uint32_t, std::size_t>;
  const Index m = static_cast<Index>(wide_m);
  const Index k = static_cast<Index>(wide_k);
  constexpr Index kSubElems = 16;
  constexpr Index kSubsPerTile = 32 * TilesPerStage;
  constexpr Index kVectorsPerSub = kSubElems / 4;
  // Batch-eight Q5_K stages without per-sub-block padding.
  constexpr bool kCompact =
      WType == core::GgmlType::kQ5_K && Batch == 8 && TilesPerStage == 1;
  constexpr Index kStride = kSubElems + (kCompact ? 0 : 4);
  constexpr Index kTileStride = kSubsPerTile * kStride;
  constexpr bool kHasOffset =
      WType == core::GgmlType::kQ4_K || WType == core::GgmlType::kQ5_K;
  __shared__ float staged_x[Batch * kTileStride];
  // Affine formats use the same activation sum for every output row.
  // Compute it once during staging, in the decode GEMV's left-to-right order.
  __shared__ float staged_sums[kHasOffset ? Batch * kSubsPerTile : 1];

  const Index lane = threadIdx.x % 32u;
  const Index warp_id = threadIdx.x / 32u;
  const Index row_base =
      ((static_cast<std::size_t>(blockIdx.x) * WavesPerBlock) + warp_id) *
      RowsPerWave;
  const Index num_sub = k / kSubElems;
  const Index row_bytes = QuantRowBytes(WType, k);
  float sums[RowsPerWave][Batch] = {};

  for (Index tile_base = 0; tile_base < num_sub; tile_base += kSubsPerTile) {
    // One thread stages a whole (token, sub-block) so it can accumulate that
    // sub-block's activation sum sequentially while it has the values.
    for (Index flat = threadIdx.x; flat < Batch * kSubsPerTile;
         flat += blockDim.x) {
      const Index token = flat / kSubsPerTile;
      const Index sub = flat % kSubsPerTile;
      const Index source_sub = tile_base + sub;
      float* dst = staged_x + (token * kTileStride) + (sub * kStride);
      float total = 0.0F;
#pragma unroll
      for (Index vector = 0; vector < kVectorsPerSub; ++vector) {
        float4 value = {0.0F, 0.0F, 0.0F, 0.0F};
        if (source_sub < num_sub) {
          value = *reinterpret_cast<const float4*>(
              x + (token * k) + (source_sub * kSubElems) + (vector * 4));
        }
        // Compact staging removes per-sub-block padding. Both reads and
        // writes permute float4 groups; arithmetic is unchanged.
        const Index group = kCompact ? vector ^ ((sub >> 1U) & 3U) : vector;
        *reinterpret_cast<float4*>(dst + (group * 4)) = value;
        if constexpr (kHasOffset) {
          // Left to right, term by term, matching the GEMV's scalar loop.
          total += value.x;
          total += value.y;
          total += value.z;
          total += value.w;
        }
      }
      if constexpr (kHasOffset) {
        staged_sums[(token * kSubsPerTile) + sub] = total;
      }
    }
    SyncKQuantTile();

    for (Index tile = 0; tile < TilesPerStage; ++tile) {
      const Index slot = tile * 32 + lane;
      const Index sub = tile_base + slot;
      if (sub < num_sub) {
        QuantSub16 decoded[RowsPerWave];
#pragma unroll
        for (Index r = 0; r < RowsPerWave; ++r) {
          const Index row = row_base + r;
          const Index safe_row = row < m ? row : m - 1;
          DecodeQuantSub16<NarrowIndex>(
              WType,
              static_cast<const std::uint8_t*>(w) + (safe_row * row_bytes), sub,
              decoded[r]);
        }
        float dots[RowsPerWave][Batch] = {};
#pragma unroll
        for (Index group = 0; group < kVectorsPerSub; ++group) {
#pragma unroll
          for (Index token = 0; token < Batch; ++token) {
            const Index input_group =
                kCompact ? group ^ ((slot >> 1U) & 3U) : group;
            const float4 xv = *reinterpret_cast<const float4*>(
                staged_x + (token * kTileStride) + (slot * kStride) +
                (input_group * 4));
#pragma unroll
            for (Index r = 0; r < RowsPerWave; ++r) {
              const std::int8_t* q = decoded[r].q + (group * 4);
              dots[r][token] += static_cast<float>(q[0]) * xv.x;
              dots[r][token] += static_cast<float>(q[1]) * xv.y;
              dots[r][token] += static_cast<float>(q[2]) * xv.z;
              dots[r][token] += static_cast<float>(q[3]) * xv.w;
            }
          }
        }
#pragma unroll
        for (Index r = 0; r < RowsPerWave; ++r) {
#pragma unroll
          for (Index token = 0; token < Batch; ++token) {
            if constexpr (kHasOffset) {
              sums[r][token] += (decoded[r].scale * dots[r][token]) -
                                (decoded[r].offset *
                                 staged_sums[(token * kSubsPerTile) + slot]);
            } else {
              // Decode rounds the scaled dot before accumulating it. Keep
              // contraction disabled only here; the dot above still uses
              // its original FMAs. Symmetric formats need no activation
              // sums.
#pragma clang fp contract(off)
              const float contribution = decoded[r].scale * dots[r][token];
              sums[r][token] += contribution;
            }
          }
        }
      }
    }
    SyncKQuantTile();
  }

#pragma unroll
  for (Index r = 0; r < RowsPerWave; ++r) {
#pragma unroll
    for (Index token = 0; token < Batch; ++token) {
      sums[r][token] = ReduceKQuantWave(sums[r][token]);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (Index r = 0; r < RowsPerWave; ++r) {
      const Index row = row_base + r;
      if (row >= m) {
        continue;
      }
#pragma unroll
      for (Index token = 0; token < Batch; ++token) {
        y[(token * m) + row] = sums[r][token];
      }
    }
  }
}

}  // namespace gufo::models::gemma4::ggml

#endif  // GUFO_MODELS_GEMMA4_HIP_GGML_SMALL_BATCH_GEMM_HPP_
