#include "src/models/nemotron/kernels/rocm/gemv.hpp"

#include <hip/hip_fp16.h>

#include <cstddef>

#include "src/models/nemotron/kernels/rocm/gemm.hpp"
#include "src/models/nemotron/kernels/rocm/iq2xxs.hip.hpp"
#include "src/models/nemotron/kernels/rocm/lanes.hip.hpp"

namespace gufo::models::nemotron::rocm {
namespace {


constexpr int kLanes = 32;
constexpr int kWaves = 8;
constexpr int kThreads = kLanes * kWaves;
/// Rows each wave streams; they share the activations held in registers.
constexpr int kRowsPerWave = 4;
constexpr int kRowsPerBlock = kWaves * kRowsPerWave;
/// IQ2_XXS super-blocks (256 columns) of activations staged per pass.
constexpr int kChunkBlocks = 8;

/// Each wave streams kRowsPerWave rows and keeps one activation slice per
/// token in registers for all of them. With kGroups, a lane owns a whole
/// 32-value group and a wave reads four consecutive super-blocks (264
/// contiguous bytes) of a row per step, which needs a multiple of four
/// super-blocks per row; otherwise a lane owns eight values, group lane / 4
/// and grid entry lane % 4, of one super-block per step. A routed block
/// serves one (token, slot) pair, a dense block every token.
template<int kTokens, bool kGroups>
__global__ __launch_bounds__(kThreads) void Iq2xxsGemvKernel(
    Iq2xxsGemvArgs a) {
  constexpr int kStepBlocks = kGroups ? kLanes / 8 : 1;
  // Several tokens stage fewer super-blocks, so more blocks fit per CU.
  constexpr int kChunk = kTokens > 1 ? kChunkBlocks / 2 : kChunkBlocks;
  static_assert(kChunk % kStepBlocks == 0);
  static_assert(kThreads == 256, "one grid entry per thread");
  __shared__ uint4 s_grid[256];
  __shared__ uint4 s_x[kTokens][kChunk * kLanes];
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kLanes;
  const int lane = tid % kLanes;
  const bool routed = a.ids != nullptr;
  const int slot = static_cast<int>(blockIdx.y);
  const int expert = routed ? a.ids[slot] : 0;
  const int blocks = a.cols / 256;
  const std::size_t row_bytes =
      static_cast<std::size_t>(blocks) * kIq2xxsBlockBytes;
  const auto* w = static_cast<const std::uint8_t*>(a.weights) +
                  static_cast<std::size_t>(expert) * a.rows * row_bytes;
  const int row0 =
      static_cast<int>(blockIdx.x) * kRowsPerBlock + wave * kRowsPerWave;
  const int x_row0 = routed ? (a.x_per_slot ? slot : slot / a.used) : 0;
  const int x_rows = routed ? 1 : a.n_tokens;
  s_grid[tid] = reinterpret_cast<const uint4*>(a.grid)[tid];
  float acc[kRowsPerWave][kTokens] = {};
  for (int b0 = 0; b0 < blocks; b0 += kChunk) {
    const int nb = min(kChunk, blocks - b0);
    __syncthreads();
    // Absent tokens stage zeros, so the loops below have fixed trip counts.
    for (int i = tid; i < kTokens * kChunk * kLanes; i += kThreads) {
      const int t = i / (kChunk * kLanes);
      const int v = i % (kChunk * kLanes);
      uint4 packed = make_uint4(0U, 0U, 0U, 0U);
      if (t < x_rows && v < nb * kLanes) {
        const float* src = a.x +
                           static_cast<std::size_t>(x_row0 + t) * a.cols +
                           static_cast<std::size_t>(b0) * 256 + v * 8;
        const float4 lo = *reinterpret_cast<const float4*>(src);
        const float4 hi = *reinterpret_cast<const float4*>(src + 4);
        packed = make_uint4(PackBf16(lo.x, lo.y), PackBf16(lo.z, lo.w),
                            PackBf16(hi.x, hi.y), PackBf16(hi.z, hi.w));
      }
      s_x[t][v] = packed;
    }
    __syncthreads();
    for (int s0 = 0; s0 < nb; s0 += kStepBlocks) {
      if constexpr (kGroups && kTokens == 1) {
        const int bl = s0 + lane / 8;
        const int group = lane % 8;
        uint4 xv[kTokens][4];
#pragma unroll
        for (int t = 0; t < kTokens; ++t)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            xv[t][e] = s_x[t][bl * kLanes + group * 4 + e];
#pragma unroll
        for (int r = 0; r < kRowsPerWave; ++r) {
          const std::uint8_t* block =
              w + static_cast<std::size_t>(row0 + r) * row_bytes +
              static_cast<std::size_t>(b0 + bl) * kIq2xxsBlockBytes;
          const auto* q16 =
              reinterpret_cast<const std::uint16_t*>(block + 2 + 8 * group);
          const std::uint32_t index =
              q16[0] | (static_cast<std::uint32_t>(q16[1]) << 16);
          const std::uint32_t aux =
              q16[2] | (static_cast<std::uint32_t>(q16[3]) << 16);
          const float db = GroupScale(block, aux);
          float dot[kTokens] = {};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const SignedGrid g(s_grid[(index >> (8 * e)) & 0xFFU],
                               (aux >> (7 * e)) & 127U);
#pragma unroll
            for (int t = 0; t < kTokens; ++t)
              dot[t] = g.Dot(xv[t][e], dot[t]);
          }
#pragma unroll
          for (int t = 0; t < kTokens; ++t)
            acc[r][t] = fmaf(db, dot[t], acc[r][t]);
        }
      } else if constexpr (kGroups) {
        const int bl = s0 + lane / 8;
        const int group = lane % 8;
        // Several tokens: per grid entry, every row against the tokens'
        // activations, so one entry's slice is in registers instead of
        // four. The sums keep the one-token order.
        std::uint32_t index[kRowsPerWave], aux[kRowsPerWave];
        float db[kRowsPerWave];
#pragma unroll
        for (int r = 0; r < kRowsPerWave; ++r) {
          const std::uint8_t* block =
              w + static_cast<std::size_t>(row0 + r) * row_bytes +
              static_cast<std::size_t>(b0 + bl) * kIq2xxsBlockBytes;
          const auto* q16 =
              reinterpret_cast<const std::uint16_t*>(block + 2 + 8 * group);
          index[r] = q16[0] | (static_cast<std::uint32_t>(q16[1]) << 16);
          aux[r] = q16[2] | (static_cast<std::uint32_t>(q16[3]) << 16);
          db[r] = GroupScale(block, aux[r]);
        }
        float dot[kRowsPerWave][kTokens] = {};
#pragma unroll 1
        for (int e = 0; e < 4; ++e) {
          uint4 xv[kTokens];
#pragma unroll
          for (int t = 0; t < kTokens; ++t)
            xv[t] = s_x[t][bl * kLanes + group * 4 + e];
#pragma unroll
          for (int r = 0; r < kRowsPerWave; ++r) {
            const SignedGrid g(s_grid[(index[r] >> (8 * e)) & 0xFFU],
                               (aux[r] >> (7 * e)) & 127U);
#pragma unroll
            for (int t = 0; t < kTokens; ++t)
              dot[r][t] = g.Dot(xv[t], dot[r][t]);
          }
        }
#pragma unroll
        for (int r = 0; r < kRowsPerWave; ++r)
#pragma unroll
          for (int t = 0; t < kTokens; ++t)
            acc[r][t] = fmaf(db[r], dot[r][t], acc[r][t]);
      } else {
        const int group = lane / 4;
        const int part = lane % 4;
        uint4 xv[kTokens];
#pragma unroll
        for (int t = 0; t < kTokens; ++t)
          xv[t] = s_x[t][s0 * kLanes + lane];
#pragma unroll
        for (int r = 0; r < kRowsPerWave; ++r) {
          const std::uint8_t* block =
              w + static_cast<std::size_t>(row0 + r) * row_bytes +
              static_cast<std::size_t>(b0 + s0) * kIq2xxsBlockBytes;
          const std::uint8_t* q = block + 2 + 8 * group;
          const auto* q16 = reinterpret_cast<const std::uint16_t*>(q);
          const std::uint32_t aux =
              q16[2] | (static_cast<std::uint32_t>(q16[3]) << 16);
          const float db = GroupScale(block, aux);
          const SignedGrid g(s_grid[q[part]], (aux >> (7 * part)) & 127U);
#pragma unroll
          for (int t = 0; t < kTokens; ++t)
            acc[r][t] = fmaf(db, g.Dot(xv[t], 0.0F), acc[r][t]);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRowsPerWave; ++r) {
#pragma unroll
    for (int t = 0; t < kTokens; ++t)
      acc[r][t] = WaveSum(acc[r][t]);
  }
  if (lane != 0)
    return;
#pragma unroll
  for (int t = 0; t < kTokens; ++t) {
    if (t >= x_rows)
      break;
    float* out = a.out + static_cast<std::size_t>(routed ? slot : t) * a.rows;
#pragma unroll
    for (int r = 0; r < kRowsPerWave; ++r) {
      float v = acc[r][t];
      if (a.relu2) {
        v = fmaxf(v, 0.0F);
        v *= v;
      }
      out[row0 + r] = a.accumulate ? out[row0 + r] + v : v;
    }
  }
}

template<int kTokens>
void LaunchIq2xxs(const Iq2xxsGemvArgs& args, dim3 grid, hipStream_t stream) {
  if ((args.cols / 256) % (kLanes / 8) == 0)
    Iq2xxsGemvKernel<kTokens, true><<<grid, kThreads, 0, stream>>>(args);
  else
    Iq2xxsGemvKernel<kTokens, false><<<grid, kThreads, 0, stream>>>(args);
}

/// Rows per wave of the BF16 products: two keep enough waves in flight
/// to hide the weight loads, which matters more here than sharing each
/// activation load over more rows.
constexpr int kBf16RowsPerWave = 2;
constexpr int kBf16RowsPerBlock = kWaves * kBf16RowsPerWave;

/// Where the BF16 products store: F32 rows, BF16 rows (for a product that
/// feeds another) or F32 sums into the existing rows.
struct Bf16Store {
  float* out;
  std::uint16_t* out_bf16;
  int rows;
  bool relu2;
  bool accumulate;
};

/// kTokens BF16 activation rows (`x[t]`, cols / 8 uint4 each) against the
/// wave's kBf16RowsPerWave weight rows from `w`; out row `out_rows[t]`
/// receives token t. Each token's sums run in the one-token kernel's order
/// (kGroups as there: a lane owns a 32-value group of one of four
/// super-blocks per step; otherwise eight values of one super-block), so
/// every output is bit-identical to a one-token product of its row.
template<int kTokens, bool kGroups>
__device__ void Bf16Rows(const std::uint8_t* w, std::size_t row_bytes,
                         int blocks, const uint4* const (&x)[kTokens],
                         const int (&out_rows)[kTokens], int row0,
                         const uint4* s_grid, const Bf16Store& store,
                         int lane) {
  constexpr int R = kBf16RowsPerWave;
  // Measured: unrolling twice overlaps the next step's loads up to seven
  // rows; with eight the compiler's schedule for it runs slower.
  constexpr int kUnroll = kTokens < 8 ? 2 : 1;
  float acc[R][kTokens] = {};
  if constexpr (kGroups) {
    const int group = lane % 8;
#pragma unroll kUnroll
    for (int s0 = 0; s0 < blocks; s0 += kLanes / 8) {
      const int b = s0 + lane / 8;
      std::uint32_t index[R], aux[R];
      float db[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const std::uint8_t* block =
            w + static_cast<std::size_t>(r) * row_bytes +
            static_cast<std::size_t>(b) * kIq2xxsBlockBytes;
        const auto* q16 =
            reinterpret_cast<const std::uint16_t*>(block + 2 + 8 * group);
        index[r] = q16[0] | (static_cast<std::uint32_t>(q16[1]) << 16);
        aux[r] = q16[2] | (static_cast<std::uint32_t>(q16[3]) << 16);
        db[r] = GroupScale(block, aux[r]);
      }
      float dot[R][kTokens] = {};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        uint4 xv[kTokens];
#pragma unroll
        for (int t = 0; t < kTokens; ++t)
          xv[t] = x[t][b * kLanes + group * 4 + e];
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const SignedGrid g(s_grid[(index[r] >> (8 * e)) & 0xFFU],
                             (aux[r] >> (7 * e)) & 127U);
#pragma unroll
          for (int t = 0; t < kTokens; ++t)
            dot[r][t] = g.Dot(xv[t], dot[r][t]);
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < kTokens; ++t)
          acc[r][t] = fmaf(db[r], dot[r][t], acc[r][t]);
    }
  } else {
    const int group = lane / 4;
    const int part = lane % 4;
#pragma unroll kUnroll
    for (int s = 0; s < blocks; ++s) {
      std::uint32_t idx[R], aux[R];
      float db[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const std::uint8_t* block =
            w + static_cast<std::size_t>(r) * row_bytes +
            static_cast<std::size_t>(s) * kIq2xxsBlockBytes;
        const std::uint8_t* q = block + 2 + 8 * group;
        const auto* q16 = reinterpret_cast<const std::uint16_t*>(q);
        idx[r] = q[part];
        aux[r] = q16[2] | (static_cast<std::uint32_t>(q16[3]) << 16);
        db[r] = GroupScale(block, aux[r]);
      }
      uint4 xv[kTokens];
#pragma unroll
      for (int t = 0; t < kTokens; ++t)
        xv[t] = x[t][s * kLanes + lane];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const SignedGrid g(s_grid[idx[r]], (aux[r] >> (7 * part)) & 127U);
#pragma unroll
        for (int t = 0; t < kTokens; ++t)
          acc[r][t] = fmaf(db[r], g.Dot(xv[t], 0.0F), acc[r][t]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int t = 0; t < kTokens; ++t)
      acc[r][t] = WaveSum(acc[r][t]);
  // Output (r, t) leaves from lane r * kTokens + t.
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int t = 0; t < kTokens; ++t) {
      if (lane != (r * kTokens + t) % kLanes)
        continue;
      float v = acc[r][t];
      if (store.relu2) {
        v = fmaxf(v, 0.0F);
        v *= v;
      }
      const std::size_t o =
          static_cast<std::size_t>(out_rows[t]) * store.rows + row0 + r;
      if (store.out_bf16 != nullptr)
        store.out_bf16[o] = static_cast<std::uint16_t>(Bf16Bits(v));
      else
        store.out[o] = store.accumulate ? store.out[o] + v : v;
    }
  }
}

Bf16Store StoreOf(const Iq2xxsGemvArgs& a) {
  return {.out = a.out,
          .out_bf16 = static_cast<std::uint16_t*>(a.out_bf16),
          .rows = a.rows,
          .relu2 = a.relu2,
          .accumulate = a.accumulate};
}

/// Dense product of exactly kTokens BF16 activation rows. Nothing waits at
/// a barrier after the grid is staged: the activations come through the
/// caches, which every wave of the grid shares.
template<int kTokens, bool kGroups>
__global__ __launch_bounds__(kThreads) void Iq2xxsGemvBf16Kernel(
    const std::uint8_t* weights, const uint4* grid, const uint4* xb, int rows,
    int cols, Bf16Store store) {
  __shared__ uint4 s_grid[256];
  const int tid = static_cast<int>(threadIdx.x);
  s_grid[tid] = grid[tid];
  __syncthreads();
  const int wave = tid / kLanes;
  const int row0 = static_cast<int>(blockIdx.x) * kBf16RowsPerBlock +
                   wave * kBf16RowsPerWave;
  if (row0 >= rows)
    return;
  const int blocks = cols / 256;
  const std::size_t row_bytes =
      static_cast<std::size_t>(blocks) * kIq2xxsBlockBytes;
  const uint4* x[kTokens];
  int out_rows[kTokens];
#pragma unroll
  for (int t = 0; t < kTokens; ++t) {
    x[t] = xb + static_cast<std::size_t>(t) * (cols / 8);
    out_rows[t] = t;
  }
  Bf16Rows<kTokens, kGroups>(
      weights + static_cast<std::size_t>(row0) * row_bytes, row_bytes, blocks,
      x, out_rows, row0, s_grid, store, tid % kLanes);
}

/// Routed product over the tiles of an expert routing: block (x, y) runs
/// row block x of tile y's expert for all of the tile's slots, so each
/// expert's weights stream once however many tokens chose it.
struct TileArgs {
  const std::uint8_t* weights;
  const uint4* grid;
  const uint4* xb;
  const std::int32_t* order;
  const int4* tiles;
  const std::int32_t* num_tiles;
  int rows;
  int cols;
  int used;
  bool x_per_slot;
  Bf16Store store;
};

/// Row passes per block of the routed product: an expert's matrices are
/// short (one to three steps per row at 1024 to 2560 columns), so a block
/// runs several row blocks to spread its grid staging and tile lookup
/// (measured: 8 passes beat 4, and 4 beat 1).
constexpr int kTilePasses = 8;

template<int kSlots, bool kGroups>
__device__ void TileSlots(const TileArgs& a, const std::uint8_t* w,
                          std::size_t row_bytes, int blocks, int first,
                          int row0, const uint4* s_grid, int lane) {
  const uint4* x[kSlots];
  int out_rows[kSlots];
#pragma unroll
  for (int j = 0; j < kSlots; ++j) {
    const int slot = a.order[first + j];
    const int x_row = a.x_per_slot ? slot : slot / a.used;
    x[j] = a.xb + static_cast<std::size_t>(x_row) * (a.cols / 8);
    out_rows[j] = slot;
  }
  for (int p = 0; p < kTilePasses; ++p) {
    const int r0 = row0 + p * kBf16RowsPerBlock;
    if (r0 >= a.rows)
      break;
    Bf16Rows<kSlots, kGroups>(
        w + static_cast<std::size_t>(p) * kBf16RowsPerBlock * row_bytes,
        row_bytes, blocks, x, out_rows, r0, s_grid, a.store, lane);
  }
}

template<bool kGroups>
__global__ __launch_bounds__(kThreads) void Iq2xxsGemvTilesKernel(TileArgs a) {
  const int tile_index = static_cast<int>(blockIdx.y);
  if (tile_index >= *a.num_tiles)
    return;
  __shared__ uint4 s_grid[256];
  const int tid = static_cast<int>(threadIdx.x);
  s_grid[tid] = a.grid[tid];
  __syncthreads();
  const int wave = tid / kLanes;
  const int lane = tid % kLanes;
  const int row0 =
      static_cast<int>(blockIdx.x) * kBf16RowsPerBlock * kTilePasses +
      wave * kBf16RowsPerWave;
  if (row0 >= a.rows)
    return;
  const int4 tile = a.tiles[tile_index];
  const int blocks = a.cols / 256;
  const std::size_t row_bytes =
      static_cast<std::size_t>(blocks) * kIq2xxsBlockBytes;
  const std::uint8_t* w =
      a.weights +
      (static_cast<std::size_t>(tile.x) * a.rows + row0) * row_bytes;
  // Short chunks give at most one slot per token and expert; longer tiles
  // run in pieces of kGemvMaxTokens.
  for (int done = 0; done < tile.z; done += kGemvMaxTokens) {
    const int first = tile.y + done;
    switch (min(kGemvMaxTokens, tile.z - done)) {
      case 1:
        TileSlots<1, kGroups>(a, w, row_bytes, blocks, first, row0, s_grid,
                                lane);
        break;
      case 2:
        TileSlots<2, kGroups>(a, w, row_bytes, blocks, first, row0, s_grid,
                                lane);
        break;
      case 3:
        TileSlots<3, kGroups>(a, w, row_bytes, blocks, first, row0, s_grid,
                                lane);
        break;
      case 4:
        TileSlots<4, kGroups>(a, w, row_bytes, blocks, first, row0, s_grid,
                                lane);
        break;
      case 5:
        TileSlots<5, kGroups>(a, w, row_bytes, blocks, first, row0, s_grid,
                                lane);
        break;
      case 6:
        TileSlots<6, kGroups>(a, w, row_bytes, blocks, first, row0, s_grid,
                                lane);
        break;
      case 7:
        TileSlots<7, kGroups>(a, w, row_bytes, blocks, first, row0, s_grid,
                                lane);
        break;
      default:
        TileSlots<8, kGroups>(a, w, row_bytes, blocks, first, row0, s_grid,
                                lane);
        break;
    }
  }
}

/// Groups a short chunk's slots by expert in one block: tiles in the order
/// of each expert's first slot, a tile's slots in slot order.
__global__ __launch_bounds__(kRouteShortSlots) void RouteShortKernel(
    const std::int32_t* ids, int slots, std::int32_t* order, int4* tiles,
    std::int32_t* num_tiles) {
  __shared__ int s_ids[kRouteShortSlots];
  __shared__ int s_lead_count[kRouteShortSlots];  ///< Count at a lead slot.
  const int s = static_cast<int>(threadIdx.x);
  if (s < slots)
    s_ids[s] = ids[s];
  __syncthreads();
  int lead = s;
  int rank = 0;
  int count = 0;
  if (s < slots) {
    const int e = s_ids[s];
    for (int j = 0; j < slots; ++j) {
      if (s_ids[j] != e)
        continue;
      ++count;
      if (j < s) {
        ++rank;
        lead = min(lead, j);
      }
    }
  }
  s_lead_count[s] = s < slots && rank == 0 ? count : 0;
  __syncthreads();
  if (s < slots) {
    int tile = 0;
    int offset = 0;
    for (int j = 0; j < lead; ++j) {
      tile += s_lead_count[j] != 0 ? 1 : 0;
      offset += s_lead_count[j];
    }
    order[offset + rank] = s;
    if (rank == 0)
      tiles[tile] = make_int4(s_ids[s], offset, count, 0);
  }
  if (s == 0) {
    int n = 0;
    for (int j = 0; j < slots; ++j)
      n += s_lead_count[j] != 0 ? 1 : 0;
    *num_tiles = n;
  }
}

__global__ void ToBf16Kernel(const float4* x, uint2* out, std::size_t n4) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n4) {
    const float4 v = x[i];
    out[i] = make_uint2(PackBf16(v.x, v.y), PackBf16(v.z, v.w));
  }
}

/// The one-token kernel's lane layout for these columns.
bool Groups(int cols) { return (cols / 256) % (kLanes / 8) == 0; }

template<int kTokens>
void LaunchBf16(const Iq2xxsGemvArgs& a, hipStream_t stream) {
  const auto blocks = static_cast<unsigned>(a.rows / kBf16RowsPerBlock);
  const auto* w = static_cast<const std::uint8_t*>(a.weights);
  const auto* grid = reinterpret_cast<const uint4*>(a.grid);
  const auto* xb = static_cast<const uint4*>(a.x_bf16);
  if (Groups(a.cols))
    Iq2xxsGemvBf16Kernel<kTokens, true><<<blocks, kThreads, 0, stream>>>(
        w, grid, xb, a.rows, a.cols, StoreOf(a));
  else
    Iq2xxsGemvBf16Kernel<kTokens, false><<<blocks, kThreads, 0, stream>>>(
        w, grid, xb, a.rows, a.cols, StoreOf(a));
}

}  // namespace

void Iq2xxsGridBf16(const std::uint64_t* grid, std::uint32_t* words) {
  for (int e = 0; e < 256; ++e) {
    for (int p = 0; p < 4; ++p) {
      // Magnitudes are small integers, so their BF16 encoding is the upper
      // half of the F32 one.
      std::uint32_t pair = 0;
      for (int h = 0; h < 2; ++h) {
        const auto m = static_cast<float>((grid[e] >> (8 * (2 * p + h))) & 0xFF);
        pair |= (__builtin_bit_cast(std::uint32_t, m) >> 16) << (16 * h);
      }
      words[e * 4 + p] = pair;
    }
  }
}

hipError_t Iq2xxsGemv(const Iq2xxsGemvArgs& args, hipStream_t stream) {
  if (args.rows <= 0 || args.rows % kRowsPerBlock != 0 || args.cols <= 0 ||
      args.cols % 256 != 0 || args.n_tokens < 1 ||
      args.n_tokens > kGemvMaxTokens || args.used < 1 ||
      (args.out_bf16 != nullptr && args.accumulate) ||
      (args.route != nullptr &&
       (args.ids == nullptr || args.x_bf16 == nullptr)))
    return hipErrorInvalidValue;
  const auto row_blocks = static_cast<unsigned>(args.rows / kRowsPerBlock);
  if (args.route != nullptr) {
    const TileArgs t{
        .weights = static_cast<const std::uint8_t*>(args.weights),
        .grid = reinterpret_cast<const uint4*>(args.grid),
        .xb = static_cast<const uint4*>(args.x_bf16),
        .order = args.route->order,
        .tiles = args.route->tiles,
        .num_tiles = args.route->num_tiles,
        .rows = args.rows,
        .cols = args.cols,
        .used = args.used,
        .x_per_slot = args.x_per_slot,
        .store = StoreOf(args),
    };
    // Every tile holds at least one slot.
    constexpr int kTileRows = kBf16RowsPerBlock * kTilePasses;
    const dim3 grid(
        static_cast<unsigned>((args.rows + kTileRows - 1) / kTileRows),
        static_cast<unsigned>(args.n_tokens * args.used));
    if (Groups(args.cols))
      Iq2xxsGemvTilesKernel<true><<<grid, kThreads, 0, stream>>>(t);
    else
      Iq2xxsGemvTilesKernel<false><<<grid, kThreads, 0, stream>>>(t);
    return hipGetLastError();
  }
  if (args.x_bf16 != nullptr && args.ids == nullptr) {
    switch (args.n_tokens) {
      case 1: LaunchBf16<1>(args, stream); break;
      case 2: LaunchBf16<2>(args, stream); break;
      case 3: LaunchBf16<3>(args, stream); break;
      case 4: LaunchBf16<4>(args, stream); break;
      case 5: LaunchBf16<5>(args, stream); break;
      case 6: LaunchBf16<6>(args, stream); break;
      case 7: LaunchBf16<7>(args, stream); break;
      default: LaunchBf16<8>(args, stream); break;
    }
    return hipGetLastError();
  }
  if (args.x_bf16 != nullptr || args.out_bf16 != nullptr)
    return hipErrorInvalidValue;
  if (args.ids != nullptr) {
    LaunchIq2xxs<1>(args,
                    dim3(row_blocks,
                         static_cast<unsigned>(args.n_tokens * args.used)),
                    stream);
  } else if (args.n_tokens == 1) {
    LaunchIq2xxs<1>(args, dim3(row_blocks), stream);
  } else {
    LaunchIq2xxs<kGemvMaxTokens>(args, dim3(row_blocks), stream);
  }
  return hipGetLastError();
}

hipError_t Iq2xxsRouteShort(const std::int32_t* ids, int slots,
                            const Iq2xxsRoute& route, hipStream_t stream) {
  if (slots <= 0 || slots > kRouteShortSlots || route.max_tiles < slots)
    return hipErrorInvalidValue;
  RouteShortKernel<<<1, kRouteShortSlots, 0, stream>>>(
      ids, slots, route.order, route.tiles, route.num_tiles);
  return hipGetLastError();
}

hipError_t ToBf16(const float* x, void* out, std::size_t count,
                  hipStream_t stream) {
  if (count % 4 != 0)
    return hipErrorInvalidValue;
  const std::size_t n4 = count / 4;
  ToBf16Kernel<<<static_cast<unsigned>((n4 + 255) / 256), 256, 0, stream>>>(
      reinterpret_cast<const float4*>(x), static_cast<uint2*>(out), n4);
  return hipGetLastError();
}

}  // namespace gufo::models::nemotron::rocm
