// Excerpt of Gufo src/families/qwen/kernels/prefill_quant_gemm.hip at upstream
// 1c75887b (gufo-org/gufo): the paths Gemma 4 dispatches for the formats its
// loader accepts, in the gemma4 extension namespace.
// Gemma 4 additions from the fork's feat/gemma (83b764b4) are applied.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/gemma4/kernels/rocm/ggml/prefill_quant_gemm.hpp"

namespace gufo::models::gemma4::ggml {

__global__ void QuantizeActivationToQ8_1Kernel(const float* __restrict__ x,
                                               void* __restrict__ y,
                                               std::size_t batch,
                                               std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
  const std::size_t lane_id = threadIdx.x & 31u;

  const std::size_t total_blocks = batch * num_blocks;
  if (b_idx >= total_blocks) {
    return;
  }

  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const std::size_t elem_idx = (tok * k) + (blk * 32) + lane_id;

  const float val = x[elem_idx];
  float max_abs = fabsf(val);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }

  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const int8_t q = static_cast<int8_t>(roundf(val * id));

  StoreQ8ActLane(y, batch, num_blocks, tok, blk, lane_id, d, q);
}

/// Zeroes the tail of the last token tile so a batch that is not a multiple of
/// 16 cannot feed uninitialized scales into the GEMM epilogue.
__global__ void ZeroQ8ActTailKernel(void* __restrict__ y, std::size_t batch,
                                    std::size_t num_blocks) {
  const std::size_t tail = batch % kQ8ActTileTokens;
  if (tail == 0) {
    return;
  }
  const std::size_t tt = batch / kQ8ActTileTokens;
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t lanes = kQ8ActTileTokens - tail;
  if (idx >= num_blocks * lanes) {
    return;
  }
  const std::size_t kb = idx / lanes;
  const std::size_t tl = tail + (idx % lanes);
  std::int8_t* tile = Q8ActTile(y, num_blocks, tt, kb);
  reinterpret_cast<int32x4_t*>(tile + (tl * 16))[0] = {};
  reinterpret_cast<int32x4_t*>(tile + 256 + (tl * 16))[0] = {};
  *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
      0.0F;
  float* sums = Q8ActSumSidecar(y, batch, num_blocks * 32);
  sums[(((tt * num_blocks) + kb) * kQ8ActTileTokens) + tl] = 0.0F;
}

template<bool SubOpsel>
__device__ __forceinline__ int32x8_t Wmma_16x16x16_iu8(int32x4_t a, int32x4_t b,
                                                       int32x8_t c) {
#if defined(__has_builtin) && \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c,
                                                    SubOpsel);
#else
  return c;
#endif
}

// Two-dimensionally blocked W8A8 WMMA GEMM (opt-c163-blocked-w8a8).
//
// Traffic per multiply-accumulate is 34/(32*BN) bytes of weight and
// 36/(32*BM) bytes of activation. The previous 16-row macro tile therefore
// re-read the whole activation panel m/16 times (1088x, ~12.8 GB for the FFN
// shape at batch 2048), which dominated everything else. Blocking both
// dimensions at 128 cuts that to m/128 and lifts the kernel from 35% to 58% of
// the measured 55.07 TOPS WMMA int8 ceiling.
//
//   block = BM rows x BN tokens, BK 32-element K blocks per LDS stage
//   waves = WM row groups x WN token groups (WM*WN == 8)
//   wave  = (BM/WM) rows x (BN/WN) tokens
//
// Weights are read from the production row-major block_q8_0 layout, so no
// second copy of the model is needed; activations arrive in the tiled layout
// the quantize kernel emits. Accumulation order over K blocks is unchanged from
// the 16-row kernel, so results are bit-identical.
template<int BM, int BN, int BK, int WM, int WN>
__launch_bounds__(256) __global__
    void W8A8BlockedWmmaGEMMKernel(const void* __restrict__ w,
                                   const void* __restrict__ x_blocks,
                                   float* __restrict__ y, std::size_t batch,
                                   std::size_t m, std::size_t k) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert(BN / 16 <= 8,
                "the activation stage assigns one wave per token subtile");
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;

  __shared__ int32x4_t s_a[BK][kRowTiles][32];
  __shared__ float s_dw[BK][kRowTiles][16];
  __shared__ int32x4_t s_b[BK][kTokTiles][32];
  __shared__ float s_dx[BK][kTokTiles][16];

  const auto* w_blocks = static_cast<const Q8_0Block*>(w);
  const std::size_t num_blocks = k / 32;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;

  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;
  const std::size_t t_block = static_cast<std::size_t>(blockIdx.x) * BN;
  const std::size_t tt_block = t_block / kQ8ActTileTokens;

  float acc[kWaveRowTiles][kWaveTokTiles][8];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc[i][j][l] = 0.0F;
      }
    }
  }

  // Staging registers for the next K stage. Fetching one stage ahead lets the
  // latency of the scattered 34-byte block_q8_0 reads overlap the WMMA work
  // instead of gating the LDS fill.
  constexpr int kPrefetch = (BM * BK) / 256;
  int32x4_t r_q0[kPrefetch];
  int32x4_t r_q1[kPrefetch];
  float r_dw[kPrefetch];
  int32x4_t r_b[BK];
  float r_dx[BK];
  const int b_tile = wave_id;  // one wave per token subtile

  // opt-c179-gemm-addressing. An ISA dump of this kernel put only 64 of its
  // 2752 instructions in WMMA: ~309 were exec-mask manipulation
  // (`s_and_saveexec_b32` / `s_or_b32` / `s_and_b32`) from the per-thread
  // bounds guards below, and ~265 were 64-bit address arithmetic
  // (`v_mad_u64_u32`, `v_add_co_u32`) from recomputing `(r * num_blocks) + kb`
  // per stage. Both go away here:
  //
  //   * Every 64-bit base is hoisted out of the K loop, so a stage advance is a
  //     32-bit add.
  //   * Out-of-range rows and K blocks are CLAMPED rather than branched, and
  //     their *scale* is zeroed with a multiply. The epilogue then multiplies
  //     by dw = 0, so an out-of-range row contributes exactly zero with no
  //     divergence -- bit-identical to zero-filling the operands, because
  //     0 * dx * float(c) is 0 for any finite c.
  const int num_kb = static_cast<int>(num_blocks);
  const int m_i = static_cast<int>(m);
  const Q8_0Block* w_row[kPrefetch];
  float row_live[kPrefetch];
#pragma unroll
  for (int p = 0; p < kPrefetch; ++p) {
    const int r = static_cast<int>(r_block) + (((p * 256) + tid) / BK);
    const int r_clamped = (r < m_i) ? r : (m_i - 1);
    w_row[p] = w_blocks + (static_cast<std::size_t>(r_clamped) * num_blocks);
    row_live[p] = (r < m_i) ? 1.0F : 0.0F;
  }
  const auto* b_base = reinterpret_cast<const std::int8_t*>(x_blocks) +
                       ((tt_block + static_cast<std::size_t>(b_tile)) *
                        num_blocks * kQ8ActTileBytes);
  const bool b_live = b_tile < kTokTiles;

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
      const Q8_0Block& blk = w_row[p][kb_clamped];
      __builtin_memcpy(&r_q0[p], blk.qs + 0, 16);
      __builtin_memcpy(&r_q1[p], blk.qs + 16, 16);
      r_dw[p] = __half2float(blk.d) * ((kb < num_kb) ? row_live[p] : 0.0F);
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
        const auto* tile =
            b_base + (static_cast<std::size_t>(kb_clamped) * kQ8ActTileBytes);
        r_b[i] = reinterpret_cast<const int32x4_t*>(tile)[lane_id];
        r_dx[i] = ((kb < num_kb) ? 1.0F : 0.0F) *
                  reinterpret_cast<const float*>(
                      tile + kQ8ActScaleOffset)[lane_id & 15];
      }
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * 256) + tid;
      const int rr = idx / BK;
      const int kk = idx % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      s_a[kk][rs][rl] = r_q0[p];
      s_a[kk][rs][16 + rl] = r_q1[p];
      s_dw[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] = r_dw[p];
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile][lane_id] = r_b[i];
        if (lane_id < 16) {
          s_dx[i][b_tile][lane_id] = r_dx[i];
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }

    // Unroll the K blocks of a stage only if there are two. In a fully
    // unrolled BK=4 stage, the scheduler can move the LDS operand reads of all
    // blocks above the WMMAs of the first block. Clang 23 does this. Then the
    // kernel needs more than 256 VGPRs and spills.
#pragma unroll(BK <= 2 ? BK : 1)
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0[kWaveRowTiles];
      int32x4_t a1[kWaveRowTiles];
      float dw[kWaveRowTiles][8];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rs = (wave_row * kWaveRowTiles) + i;
        a0[i] = s_a[kb][rs][sub_lane];
        a1[i] = s_a[kb][rs][16 + sub_lane];
        const float4 lo =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][half_id * 8]);
        const float4 up =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][(half_id * 8) + 4]);
        dw[i][0] = lo.x;
        dw[i][1] = lo.y;
        dw[i][2] = lo.z;
        dw[i][3] = lo.w;
        dw[i][4] = up.x;
        dw[i][5] = up.y;
        dw[i][6] = up.z;
        dw[i][7] = up.w;
      }
      // The token-tile operands are read one tile at a time, inside the j loop,
      // rather than all kWaveTokTiles of them up front. Hoisting them held
      // 4 x (2 int32x4 + 1 float) = 36 extra VGPRs live across the whole WMMA
      // block, which is what pushed a BK=2 stage into 500 bytes/lane of
      // scratch.
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int ts = (wave_tok * kWaveTokTiles) + j;
        const int32x4_t b0 = s_b[kb][ts][sub_lane];
        const int32x4_t b1 = s_b[kb][ts][16 + sub_lane];
        const float dx = s_dx[kb][ts][sub_lane];
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = Wmma_16x16x16_iu8<true>(a0[i], b0, c);
          c = Wmma_16x16x16_iu8<true>(a1[i], b1, c);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[i][j][l] += (dw[i][l] * dx) * static_cast<float>(c[l]);
          }
        }
      }
      // Keep the LDS operand reads of the next K block below the WMMAs of this
      // K block. This prevents the same spill in an unrolled BK=2 stage. The
      // arithmetic order does not change.
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
  }

  // Transpose each 16x16 result tile through LDS so the global stores cover 16
  // consecutive rows of one token instead of 16 separate cache lines.
  __syncthreads();
  float* tile_scratch =
      reinterpret_cast<float*>(&s_a[0][0][0]) + (wave_id * 256);
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[i][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          r_block +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 =
          t_block +
          static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
      // One uniform branch per 16x16 tile instead of eight per-lane predicates,
      // which is where most of the kernel's exec-mask traffic came from.
      if (t0 + 16 <= batch && r0 + 16 <= m) {
        const std::size_t out_base = (t0 * m) + r0;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane_id;
          y[out_base + (static_cast<std::size_t>(flat >> 4) * m) +
            static_cast<std::size_t>(flat & 15)] = tile_scratch[flat];
        }
      } else {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane_id;
          const std::size_t tok = t0 + static_cast<std::size_t>(flat >> 4);
          const std::size_t r = r0 + static_cast<std::size_t>(flat & 15);
          if (tok < batch && r < m) {
            y[(tok * m) + r] = tile_scratch[flat];
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

template<typename Input>
__global__ void BatchedQuantGEMVKernel(core::GgmlType type,
                                       const void* __restrict__ w,
                                       const Input* __restrict__ x,
                                       float* __restrict__ y, std::size_t m,
                                       std::size_t k) {
  constexpr std::size_t kWarp = 32;
  const std::size_t warp_id = threadIdx.x / kWarp;
  const std::size_t lane = threadIdx.x % kWarp;
  const std::size_t warps_per_block = blockDim.x / kWarp;

  const std::size_t b = blockIdx.y;
  const std::size_t row = blockIdx.x * warps_per_block + warp_id;
  if (row >= m) {
    return;
  }

  const float dot =
      QuantWarpBlockDot<Input>(type, w, row, x + (b * k), k, lane, 0, 1);
  if (lane == 0) {
    y[(b * m) + row] = dot;
  }
}

namespace {

void LaunchZeroQ8ActTail(void* q8_1_out, std::size_t batch, std::size_t k,
                         hipStream_t stream) {
  const std::size_t tail = batch % kQ8ActTileTokens;
  if (tail == 0) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t work = num_blocks * (kQ8ActTileTokens - tail);
  ZeroQ8ActTailKernel<<<static_cast<unsigned int>((work + 255) / 256), 256, 0,
                        stream>>>(q8_1_out, batch, num_blocks);
}

}  // namespace

void LaunchQuantizeActivationQ8_1FromFp32(const float* fp32_x, void* q8_1_out,
                                          std::size_t batch, std::size_t k,
                                          hipStream_t stream) {
  if (batch == 0 || k == 0) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t total_x_blocks = batch * num_blocks;
  constexpr unsigned int quant_threads = 256;
  constexpr unsigned int warps_per_tb = quant_threads / 32;
  const std::size_t quant_grid =
      (total_x_blocks + warps_per_tb - 1) / warps_per_tb;
  QuantizeActivationToQ8_1Kernel<<<static_cast<unsigned int>(quant_grid),
                                   quant_threads, 0, stream>>>(fp32_x, q8_1_out,
                                                               batch, k);
  LaunchZeroQ8ActTail(q8_1_out, batch, k, stream);
}

std::size_t QuantizedActivationBytes(std::size_t batch, std::size_t k) {
  return Q8ActTiledBytes(batch, k) + Q8ActSumBytes(batch, k);
}

// opt-q4kxl: two-level dispatch for SmallBatchKQuantExactFp32GEMMKernel --
// exact batch width (a padded width would stage and multiply activations no
// output consumes) crossed with weight format.
// Sixteen warps per block, not eight: the staged activation tile is shared by
// every warp in the block, so doubling the warps halves the staging cost per
// output row. This is the same shape opt-c18x-q8-vec-smallbatch settled on for
// the Q8_0 vec kernel.
constexpr std::size_t kKQuantSmallBatchWarps = 16;
constexpr std::size_t kKQuantSmallBatchRowsPerWave = 2;

static_assert(FitsSmallBatch32BitIndices(8, 21504, 5376, 3024));
static_assert(FitsSmallBatch32BitIndices(8, 262144, 5376, 4410));
static_assert(FitsSmallBatch32BitIndices(2, 21504, 5376, 3024));
static_assert(FitsSmallBatch32BitIndices(2, 262144, 5376, 4410));
static_assert(!FitsSmallBatch32BitIndices(2, 1, 1ULL << 31, 1));
static_assert(!FitsSmallBatch32BitIndices(2, 1ULL << 31, 1, 1));
static_assert(!FitsSmallBatch32BitIndices(8, 1, 1ULL << 29, 1));
static_assert(!FitsSmallBatch32BitIndices(8, 1ULL << 29, 1, 1));
static_assert(!FitsSmallBatch32BitIndices(8, 65536, 131072, 90112));
static_assert(!FitsSmallBatch32BitIndices(8, 1, 1, 0));

template<std::size_t RowsPerWave, core::GgmlType WType>
void LaunchKQuantSmallBatchTyped(const void* w, const float* fp32_x, float* y,
                                 std::size_t batch, std::size_t m,
                                 std::size_t k, dim3 grid, dim3 block,
                                 hipStream_t stream) {
  const auto launch = [&]<std::size_t Batch>() {
    if constexpr (Batch >= 2 && (WType == core::GgmlType::kQ4_K ||
                                 WType == core::GgmlType::kQ5_K)) {
      if (FitsSmallBatch32BitIndices(Batch, m, k,
                                     quant::QuantizedRowBytes(WType, k))) {
        if constexpr (WType == core::GgmlType::kQ5_K && Batch >= 7) {
          // Q5_K from seven rows uses eight waves of four rows each.
          SmallBatchKQuantExactFp32GEMMKernel<8, Batch, 4, WType, 1, 4, true>
              <<<(m + 31) / 32, 256, 0, stream>>>(w, fp32_x, y, m, k);
        } else {
          SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, Batch,
                                              RowsPerWave, WType, 1, 12, true>
              <<<grid, block, 0, stream>>>(w, fp32_x, y, m, k);
        }
        return;
      }
    }
    SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, Batch,
                                        RowsPerWave, WType>
        <<<grid, block, 0, stream>>>(w, fp32_x, y, m, k);
  };
#define GUFO_KQUANT_SMALL_BATCH(N)   \
  case N:                            \
    launch.template operator()<N>(); \
    return;
  switch (batch) {
    GUFO_KQUANT_SMALL_BATCH(1)
    GUFO_KQUANT_SMALL_BATCH(2)
    GUFO_KQUANT_SMALL_BATCH(3)
    GUFO_KQUANT_SMALL_BATCH(4)
    GUFO_KQUANT_SMALL_BATCH(5)
    GUFO_KQUANT_SMALL_BATCH(6)
    GUFO_KQUANT_SMALL_BATCH(7)
    GUFO_KQUANT_SMALL_BATCH(8)
    default:
      return;
  }
#undef GUFO_KQUANT_SMALL_BATCH
}

void LaunchKQuantSmallBatchByType(core::GgmlType type, const void* w,
                                  const float* fp32_x, float* y,
                                  std::size_t batch, std::size_t m,
                                  std::size_t k, dim3 grid, dim3 block,
                                  hipStream_t stream) {
  constexpr std::size_t RowsPerWave = 3;
  switch (type) {
    case core::GgmlType::kQ4_K:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kQ4_K>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kQ5_K:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kQ5_K>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kQ6_K: {
      // The vocabulary projection reuses activations across three output
      // rows and two consecutive tiles, retaining the scalar sum order.
      if ((batch == 7 || batch == 8) && m >= 65536) {
        constexpr std::size_t kHeadRows = kKQuantSmallBatchWarps * 3;
        const auto launch_head = [&]<std::size_t Batch>() {
          const auto launch_indices = [&]<bool Narrow>() {
            SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, Batch,
                                                3, core::GgmlType::kQ6_K, 2, 1,
                                                Narrow>
                <<<(m + kHeadRows - 1) / kHeadRows, block, 0, stream>>>(
                    w, fp32_x, y, m, k);
          };
          if (FitsSmallBatch32BitIndices(
                  Batch, m, k,
                  quant::QuantizedRowBytes(core::GgmlType::kQ6_K, k))) {
            launch_indices.template operator()<true>();
          } else {
            launch_indices.template operator()<false>();
          }
        };
        switch (batch) {
          case 7:
            launch_head.template operator()<7>();
            break;
          case 8:
            launch_head.template operator()<8>();
            break;
          default:
            break;
        }
        return;
      }
      // Other shapes retain two rows to bound register pressure.
      constexpr std::size_t kRowsPerBlock =
          kKQuantSmallBatchWarps * kKQuantSmallBatchRowsPerWave;
      const dim3 q6_grid(
          static_cast<unsigned int>((m + kRowsPerBlock - 1) / kRowsPerBlock));
      LaunchKQuantSmallBatchTyped<kKQuantSmallBatchRowsPerWave,
                                  core::GgmlType::kQ6_K>(
          w, fp32_x, y, batch, m, k, q6_grid, block, stream);
    }
      return;
    case core::GgmlType::kQ4_0:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kQ4_0>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    default:
      return;
  }
}

// opt-q4kxl: tile-configuration dispatch for WKQuantA8BlockedWmmaGEMMKernel.
// The configurations mirror the Q8_0 launcher exactly so the two paths share
// the same occupancy tuning.
template<core::GgmlType WType>
void LaunchKQuantWmmaGEMMTyped(const void* w, const void* q8_1_x, float* y,
                               std::size_t batch, std::size_t m, std::size_t k,
                               hipStream_t stream) {
  if (batch <= 8) {
    constexpr int kBM = 128;
    const dim3 grid(1, static_cast<unsigned int>((m + kBM - 1) / kBM));
    if (m >= (3 * k)) {
      WKQuantA8BlockedWmmaGEMMKernel<kBM, 32, 4, 4, 2, WType>
          <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    } else {
      WKQuantA8BlockedWmmaGEMMKernel<kBM, 16, 4, 8, 1, WType>
          <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    }
    return;
  }
  if (batch >= 96) {
    constexpr int kBM = 128;
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    // WM=4/WN=2 (2 row subtiles and 4 token subtiles per wave). The mirror
    // shape WM=2/WN=4 halves the per-token activation sums the offset term
    // needs, but measured far worse -- pp2048 464 -> 372 -- because four row
    // subtiles per wave hold twice the A-fragment and scale state.
    WKQuantA8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2, WType>
        <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    return;
  }
  constexpr int kBM = 128;
  constexpr int kBN = 64;
  const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                  static_cast<unsigned int>((m + kBM - 1) / kBM));
  WKQuantA8BlockedWmmaGEMMKernel<kBM, kBN, 4, 8, 1, WType>
      <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
}

void LaunchKQuantWmmaGEMM(core::GgmlType type, const void* w,
                          const void* q8_1_x, float* y, std::size_t batch,
                          std::size_t m, std::size_t k, hipStream_t stream) {
  switch (type) {
    case core::GgmlType::kQ4_K:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kQ4_K>(w, q8_1_x, y, batch, m,
                                                       k, stream);
      return;
    case core::GgmlType::kQ5_K:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kQ5_K>(w, q8_1_x, y, batch, m,
                                                       k, stream);
      return;
    case core::GgmlType::kQ6_K:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kQ6_K>(w, q8_1_x, y, batch, m,
                                                       k, stream);
      return;
    case core::GgmlType::kQ4_0:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kQ4_0>(w, q8_1_x, y, batch, m,
                                                       k, stream);
      return;
    default:
      return;
  }
}

void LaunchBatchedQuantGEMMPreQuantized(core::GgmlType type, const void* w,
                                        const void* q8_1_x, float* y,
                                        std::size_t batch, std::size_t m,
                                        std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0) {
    return;
  }
  if (detail::TryLaunchQuantPrefillWave64(type, w, q8_1_x, y, batch, m, k,
                                          stream)) {
    return;
  }
  // opt-q4kxl: every K-quant format goes through the
  // generalized WMMA kernel. Without this the only prefill route for them was
  // BatchedQuantGEMVKernel, one warp per (row, token), which re-reads the whole
  // weight matrix once per token and is unusable at prefill batch.
  if (type != core::GgmlType::kQ8_0) {
    if (::gufo::quant::QuantizedBlockElements(type) != 0) {
      LaunchKQuantWmmaGEMM(type, w, q8_1_x, y, batch, m, k, stream);
    }
    return;
  }
  if (batch <= 8) {
    constexpr int kBM = 128;
    const dim3 grid(1, static_cast<unsigned int>((m + kBM - 1) / kBM));
    if (m >= (3 * k)) {
      W8A8BlockedWmmaGEMMKernel<kBM, 32, 4, 4, 2>
          <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    } else {
      W8A8BlockedWmmaGEMMKernel<kBM, 16, 4, 8, 1>
          <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    }
    return;
  }
  // A 128-token macro tile is the throughput configuration; short prompt chunks
  // would leave most of it idle, so they take the 64-token variant.
  if (batch >= 96) {
    constexpr int kBM = 128;
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    // Two K blocks per LDS stage, not four: the stage buffers drop from 36864
    // to 18432 bytes and occupancy rises from 6 to 8 waves per SIMD. This is
    // only reachable with the register pressure the addressing rewrite and the
    // one-token-tile-at-a-time compute loop freed up -- at BK=2 the previous
    // body needed 500 bytes/lane of scratch and ran 4.3x slower.
    W8A8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2>
        <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
  } else {
    constexpr int kBM = 128;
    constexpr int kBN = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    W8A8BlockedWmmaGEMMKernel<kBM, kBN, 4, 8, 1>
        <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
  }
}

bool LaunchKQuantSmallBatchDoubleStage(core::GgmlType type, const void* w,
                                       const float* fp32_x, float* y,
                                       std::size_t batch, std::size_t m,
                                       std::size_t k, hipStream_t stream) {
  if (batch < 1 || batch > 16 || m == 0 || k == 0 ||
      (type != core::GgmlType::kQ4_K && type != core::GgmlType::kQ5_K &&
       type != core::GgmlType::kQ6_K && type != core::GgmlType::kQ4_0) ||
      m > std::numeric_limits<std::uint32_t>::max() / batch ||
      k > std::numeric_limits<std::uint32_t>::max() / batch ||
      m > std::numeric_limits<std::uint32_t>::max() /
              quant::QuantizedRowBytes(type, k)) {
    // 32-bit device indexing, as FitsSmallBatch32BitIndices checks up to 8.
    return false;
  }
  constexpr std::uint32_t kWaves = 16;
  constexpr std::size_t kRows = 3;
  const dim3 grid(
      static_cast<unsigned>((m + kWaves * kRows - 1) / (kWaves * kRows)));
  const auto launch = [&]<core::GgmlType Type, std::size_t Batch>() {
    // Seven and eight rows are faster with the looser occupancy hint; past
    // eight, one activation tile per stage keeps the rows in shared memory.
    constexpr std::uint32_t kMinWaves = Batch >= 7 ? 8 : 12;
    constexpr std::size_t kTiles = Batch > 8 ? 1 : 2;
    SmallBatchKQuantExactFp32GEMMKernel<kWaves, Batch, kRows, Type, kTiles,
                                        kMinWaves, true>
        <<<grid, kWaves * 32, 0, stream>>>(w, fp32_x, y, m, k);
  };
  const auto typed = [&]<core::GgmlType Type>() {
    switch (batch) {
      case 1:
        launch.template operator()<Type, 1>();
        break;
      case 2:
        launch.template operator()<Type, 2>();
        break;
      case 3:
        launch.template operator()<Type, 3>();
        break;
      case 4:
        launch.template operator()<Type, 4>();
        break;
      case 5:
        launch.template operator()<Type, 5>();
        break;
      case 6:
        launch.template operator()<Type, 6>();
        break;
      case 7:
        launch.template operator()<Type, 7>();
        break;
      case 8:
        launch.template operator()<Type, 8>();
        break;
      case 9:
        launch.template operator()<Type, 9>();
        break;
      case 10:
        launch.template operator()<Type, 10>();
        break;
      case 11:
        launch.template operator()<Type, 11>();
        break;
      case 12:
        launch.template operator()<Type, 12>();
        break;
      case 13:
        launch.template operator()<Type, 13>();
        break;
      case 14:
        launch.template operator()<Type, 14>();
        break;
      case 15:
        launch.template operator()<Type, 15>();
        break;
      default:
        launch.template operator()<Type, 16>();
        break;
    }
  };
  switch (type) {
    case core::GgmlType::kQ4_K:
      typed.template operator()<core::GgmlType::kQ4_K>();
      break;
    case core::GgmlType::kQ5_K:
      typed.template operator()<core::GgmlType::kQ5_K>();
      break;
    case core::GgmlType::kQ4_0:
      typed.template operator()<core::GgmlType::kQ4_0>();
      break;
    default:
      typed.template operator()<core::GgmlType::kQ6_K>();
      break;
  }
  return true;
}

bool LaunchQ8_0SmallBatchWide(const void* w, const float* fp32_x, float* y,
                              std::size_t batch, std::size_t m, std::size_t k,
                              hipStream_t stream) {
  if (m == 0 || k == 0 || batch < 9 || batch > 16) {
    return false;
  }
  // The row groups of LaunchBatchedQuantGEMMFp32's exact-width route.
  const auto launch = [&]<std::size_t Batch>() {
    constexpr std::size_t rows = Batch == 11 || Batch == 13 || Batch == 14 ? 4
                                 : Batch == 12                             ? 2
                                                                           : 3;
    SmallBatchQ8_0ExactFp32VecGEMMKernel<16, Batch, rows, true, true>
        <<<(m + 16 * rows - 1) / (16 * rows), 512, 0, stream>>>(w, fp32_x, y, m,
                                                                k);
  };
  switch (batch) {
    case 9:
      launch.template operator()<9>();
      break;
    case 10:
      launch.template operator()<10>();
      break;
    case 11:
      launch.template operator()<11>();
      break;
    case 12:
      launch.template operator()<12>();
      break;
    case 13:
      launch.template operator()<13>();
      break;
    case 14:
      launch.template operator()<14>();
      break;
    case 15:
      launch.template operator()<15>();
      break;
    default:
      launch.template operator()<16>();
      break;
  }
  return true;
}

void LaunchBatchedQuantGEMMFp32(core::GgmlType type, const void* w,
                                const float* fp32_x, float* y,
                                std::size_t batch, std::size_t m, std::size_t k,
                                hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || type == core::GgmlType::kBF16 ||
      type == core::GgmlType::kF32) {
    return;
  }

  if (batch > 8) {
    // Cross-request verification can contain up to 64 rows: eight-row passes
    // with the exact small-batch arithmetic of the first eight.
    for (std::size_t offset = 0; offset < batch; offset += 8) {
      LaunchBatchedQuantGEMMFp32(type, w, fp32_x + offset * k, y + offset * m,
                                 std::min<std::size_t>(8, batch - offset), m, k,
                                 stream);
    }
    return;
  }

  constexpr std::size_t kWarp = 32;
  constexpr std::size_t kWarpsPerBlock = 8;
  const std::size_t num_blocks_x = (m + kWarpsPerBlock - 1) / kWarpsPerBlock;
  // opt-q4kxl: K-quant formats share one weight stream across the whole
  // draft block instead of re-reading it per token.
  if (::gufo::models::gemma4::ggml::detail::IsNativeWmmaQuant(type)) {
    // Exact width instantiations: a padded Batch would stage and multiply
    // activations that no output consumes.
    constexpr std::size_t kKWarpsPerBlock = kKQuantSmallBatchWarps;
    const dim3 block(static_cast<unsigned int>(kWarp * kKWarpsPerBlock));
    // Three rows with twelve waves is the retained geometry. Q6_K dispatches
    // two rows below because its extra scale state spills with three.
    constexpr std::size_t kRowsPerBlock = kKWarpsPerBlock * 3;
    const dim3 grid(
        static_cast<unsigned int>((m + kRowsPerBlock - 1) / kRowsPerBlock));
    LaunchKQuantSmallBatchByType(type, w, fp32_x, y, batch, m, k, grid, block,
                                 stream);
    return;
  }
  if (type == core::GgmlType::kQ8_0) {
    constexpr std::size_t kVecWarpsPerBlock = 16;
    constexpr std::size_t kVecRowsPerWave = 2;
    constexpr std::size_t kVecRowsPerBlock =
        kVecWarpsPerBlock * kVecRowsPerWave;
    const unsigned int vec_blocks = static_cast<unsigned int>(
        (m + kVecRowsPerBlock - 1) / kVecRowsPerBlock);
    const dim3 vec_block(static_cast<unsigned int>(kWarp * kVecWarpsPerBlock));
    switch (batch) {
      case 1:
        SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 1,
                                             kVecRowsPerWave>
            <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      case 2:
        SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 2,
                                             kVecRowsPerWave>
            <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      case 3:
        SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 3,
                                             kVecRowsPerWave>
            <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      case 4:
        SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 4,
                                             kVecRowsPerWave>
            <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      case 5:
        SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 5,
                                             kVecRowsPerWave>
            <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      case 6:
        SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 6,
                                             kVecRowsPerWave>
            <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      case 7:
        SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 7,
                                             kVecRowsPerWave>
            <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      default:
        SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 8,
                                             kVecRowsPerWave>
            <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
        return;
    }
  }

  const dim3 grid(static_cast<unsigned int>(num_blocks_x),
                  static_cast<unsigned int>(batch));
  const dim3 block(static_cast<unsigned int>(kWarp * kWarpsPerBlock));
  BatchedQuantGEMVKernel<float>
      <<<grid, block, 0, stream>>>(type, w, fp32_x, y, m, k);
}

}  // namespace gufo::models::gemma4::ggml
#endif  // defined(ENGINE_ENABLE_HIP)
