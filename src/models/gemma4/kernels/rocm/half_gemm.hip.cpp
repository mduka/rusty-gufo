// Excerpt of the Gufo fork's feat/gemma (83b764b4)
// src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp: the binary16
// dense and routed GEMMs, with Gemma 4's grouped Q8_0 layout, fused GeGLU
// epilogue and wide-batch plans, and the MTP candidate selection, in the
// gemma4 extension namespace.
#include "src/models/gemma4/kernels/rocm/half_gemm.hpp"

#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <hipcub/block/block_radix_sort.hpp>
#include <stdexcept>

#include "src/models/gemma4/kernels/rocm/mtp_candidates.hpp"

namespace gufo::models::gemma4::rocm {

namespace {

constexpr unsigned kThreads = 256;

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

constexpr unsigned kMtpCandidateTile = 1024;

// A token excluded from its tile's top Keep cannot enter the global top Keep.
// Reduce tiles repeatedly, retaining the original order of equal scores.
// Only token IDs need to survive between passes; the original logit buffer
// supplies scores and preserves the sign of zero in the final output.
template<unsigned Keep>
__global__ void MtpCandidateTileKernel(const float* logits,
                                       const std::uint32_t* input,
                                       std::uint32_t* output, float* scores,
                                       std::uint32_t size,
                                       std::uint32_t vocab) {
  using Sort = hipcub::BlockRadixSort<float, kThreads, 4, std::uint32_t>;
  __shared__ Sort::TempStorage scratch;
  float keys[4];
  std::uint32_t ids[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const std::size_t i =
        std::size_t(blockIdx.x) * kMtpCandidateTile + threadIdx.x * 4 + j;
    const auto id = i < size ? (input != nullptr ? input[i] : i) : UINT32_MAX;
    ids[j] = static_cast<std::uint32_t>(id);
    const float value = id < vocab ? logits[id] : -INFINITY;
    keys[j] = isfinite(value) ? (value == 0.0F ? 0.0F : value) : -INFINITY;
  }
  Sort(scratch).SortDescending(keys, ids);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const unsigned rank = threadIdx.x * 4 + j;
    if (rank < min(Keep, size)) {
      output[blockIdx.x * Keep + rank] = ids[j];
      if (scores != nullptr) {
        const float value = ids[j] < vocab ? logits[ids[j]] : -INFINITY;
        scores[rank] = isfinite(value) ? value : -INFINITY;
      }
    }
  }
}

inline unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

__global__ void NarrowKernel(const float* x, __half* out, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    out[i] = __half(x[i]);
  }
}

struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");

// Routed expert GEMMs: assignment rows are compacted by expert with every
// bucket padded to a 16-row tile (`pad_bounds`), so a token tile never
// straddles experts; `rows_out` maps a compact row to its (token, slot)
// output row or -1 for padding.
constexpr std::size_t kRoutedTileTokens = 16;

struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

struct Q5_1Block {
  __half d;
  __half m;
  std::uint32_t qh;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q5_1Block) == 24, "block_q5_1 must be 24 bytes");

// Routed F16 WMMA expert GEMM. The weights are dequantized to F16 right after
// the LDS read (each wave decodes only its own sixteen rows) and the
// activations are F16 rows, so the matrix core accumulates the whole K extent
// in F32 with no per-block work. The codes stay packed in LDS (4 bits for Q4_K,
// 4 + 1 for Q5_1), which keeps a two-K-block stage at 11-12 KB and five blocks
// resident per WGP.
//
// A code becomes a half through a byte permute into the mantissa of 1024.0
// (0x6400 | q = 1024 + q exactly for q < 32, the half's unit being 1 there),
// a packed subtract of 1024 (exact), then one packed FMA:
//
//     w = q * scale + bias
//
// with (scale, bias) = (d * sc, -dmin * mn) for Q4_K and (d, m) for Q5_1,
// staged per (row, K block) as a half2.
//
// grid (m / BM, tiles): `tiles[y]` packs the expert in the low 16 bits and
// the token macro tile index in the high 16, so no block is launched for an
// empty tile; the row blocks of one tile are consecutive in dispatch order so
// they share the tile's gathered activations through L2. Block (x, y)
// computes rows x*BM.. of the expert against its
// compact rows [pad_bounds[e] + j*BN, +BN) and scatters them to
// out[rows_out[c]][row] (F32, or F16 with the SwiGLU applied when `out_half`
// is given: the up projection then writes the down projection's input).
constexpr std::uint32_t kHalfMagic = 0x64646464U;  // 1024.0 high bytes

/// block_q5_K: the Q4_K header, 32 high-bit bytes (bit s of byte j is the
/// fifth bit of element j of K block s), then the Q4_K nibble layout.
constexpr std::size_t kQ5KBlockBytes = 176;

template<WeightType kType>
__device__ __forceinline__ std::size_t RoutedF16RowBytes(std::size_t k) {
  return kType == WeightType::kQ4_K   ? (k / 256) * sizeof(Q4KBlock)
         : kType == WeightType::kQ5_K ? (k / 256) * kQ5KBlockBytes
         : kType == WeightType::kQ5_1 ? (k / 32) * sizeof(Q5_1Block)
                                      : (k / 32) * sizeof(Q8_0Block);
}

/// Bit `s` of each of the four bytes of `w`, packed into bits 0-3.
__device__ __forceinline__ std::uint32_t GatherBit(std::uint32_t w, int s) {
  // 0x01020408 moves byte b's bit to bit 24 + b.
  return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U;
}

/// Four packed 5-bit codes: `nib` holds the 4-bit parts one per byte, `bits`
/// bits j..j+3 of the Q5_1 high-bit word, spread to bit 4 of each byte.
__device__ __forceinline__ std::uint32_t SpreadHighBits(std::uint32_t bits) {
  // 0x00204081 = 1 + 2^7 + 2^14 + 2^21: bit b of `bits` lands at 8b, every
  // cross term falls off the 0x01010101 mask.
  return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U;
}

/// Four halves from four code bytes: 1024 + q as F16, minus `magic` (1024,
/// or 1152 for a signed byte carried as q + 128), then the affine.
__device__ __forceinline__ void CodesToHalves(std::uint32_t codes,
                                              __half2 magic, __half2 scale2,
                                              __half2 bias2, __half2& lo,
                                              __half2& hi) {
  const std::uint32_t p0 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x01050004U);
  const std::uint32_t p1 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x03070206U);
  lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
  hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
}

template<WeightType kType, int BM, int BN, int BK>
__launch_bounds__(256) __global__
    void RoutedF16GEMMKernel(const void* __restrict__ w,
                             const __half* __restrict__ x,
                             const std::int32_t* __restrict__ tiles,
                             const std::int32_t* __restrict__ pad_bounds,
                             const std::int32_t* __restrict__ rows_in,
                             const std::int32_t* __restrict__ rows_out,
                             float* __restrict__ out,
                             __half* __restrict__ out_half, std::size_t m,
                             std::size_t k) {
  static_assert(BM == 128 || BM == 256, "eight waves, 16-row tiles");
  static_assert(BN % 16 == 0 && BN / 16 <= 8);
  static_assert(BK == 2, "one stage is one 32-byte Q4_K nibble group");
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = BM / 128;  // 16-row tiles per wave
  constexpr bool kQ5 = kType == WeightType::kQ5_1;
  constexpr bool kQ5K = kType == WeightType::kQ5_K;
  constexpr bool kQ8 = kType == WeightType::kQ8_0;
  // 16-byte code chunks per row and stage: Q4_K's nibble pair and Q5_1's
  // two nibble blocks are two, Q8_0's two byte blocks are four.
  constexpr int kChunks = kQ8 ? 2 * BK : BK;

  // Codes and affine parameters are staged in LDS, with nearby rows permuted
  // across the bank groups, beside the activation plane
  // [kb][16-element quarter][token][16 B] so a fragment read is 256
  // contiguous bytes; the epilogue reuses it all.
  // Q5_1 rows stage their blocks as stored: a row's BK blocks are 16-byte
  // aligned, so whole 16-byte chunks load coalesced and each wave reads its
  // rows' scale, high bits and codes from LDS.
  constexpr int kQ5Bytes = BK * 24;  // one row's blocks per stage
  constexpr int kCodeBytes = kQ5 ? BM * kQ5Bytes : BM * kChunks * 16;
  constexpr int kHighBytes = kQ5K ? BK * BM * 4 : 0;
  constexpr int kScaleBytes = kQ5 ? 0 : BK * BM * 4;
  // One slot of padding per activation quarter plane: the eight chunks of
  // a token then land on eight bank groups when they are written.
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  // The epilogue transposes one 16x16 tile per wave through the same
  // bytes (8 KB), which the narrow tile's stages do not reach.
  constexpr int kStageBytes = kCodeBytes + kHighBytes + kScaleBytes + kActBytes;
  constexpr int kEpilogueBytes = 8 * 1024;
  constexpr int kLdsBytes =
      kStageBytes > kEpilogueBytes ? kStageBytes : kEpilogueBytes;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_codes = reinterpret_cast<uint4*>(lds);
  auto* s_high = reinterpret_cast<std::uint32_t*>(lds + kCodeBytes);
  auto* s_scale =
      reinterpret_cast<std::uint32_t*>(lds + kCodeBytes + kHighBytes);
  auto* s_act =
      reinterpret_cast<uint4*>(lds + kCodeBytes + kHighBytes + kScaleBytes);

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  // A map built on the device pads its tail with tiles past every bucket.
  if (t_local >= bucket_rows) {
    return;
  }
  const int live_tok_tiles =
      std::min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = RoutedF16RowBytes<kType>(k);
  const auto* w_expert = static_cast<const std::uint8_t*>(w) +
                         static_cast<std::size_t>(expert) * m * row_bytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int r_block = static_cast<int>(blockIdx.x) * BM;

  // Thread tid loads chunk tid % 2 of row tid / 2 (+ 128 u).
  const int f_c = tid & 1;
  const std::uint8_t* f_ptr[kWaveRowTiles];
  bool f_live[kWaveRowTiles];
  uint4 f_header[kWaveRowTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r = r_block + (tid >> 1) + (u * 128);
    f_live[u] = r < m_i;
    f_ptr[u] = w_expert +
               static_cast<std::size_t>(f_live[u] ? r : (m_i - 1)) * row_bytes;
    f_header[u] = make_uint4(0u, 0u, 0u, 0u);
  }
  // The next stage's weights and activations, fetched one stage ahead. The
  // (scale, bias) pair is derived from the raw header word only when the
  // stage is committed, so nothing waits on the loads before the compute.
  uint4 f_codes[kWaveRowTiles];
  uint4 f_codes_hi[kWaveRowTiles];  ///< Q8_0: the block's second 16 codes
  uint4 f_qh[kWaveRowTiles][2];     ///< Q5_K: the superblock's high bits
  std::uint32_t f_high[kWaveRowTiles];
  std::uint32_t f_dm[kWaveRowTiles];  ///< Q8_0: d
  int f_sb32[kWaveRowTiles];          ///< Q4_K: the K block in its superblock
  constexpr int kActFetch = BN <= 64 ? 2 : 4;
  uint4 a_data[kActFetch];
  // Q5_1 chunk c is part c % kQ5Parts of row c / kQ5Parts.
  constexpr int kQ5Parts = kQ5Bytes / 16;
  constexpr int kQ5Chunks = BM * kQ5Parts;
  constexpr int kQ5Fetch = kQ5 ? (kQ5Chunks + 255) / 256 : 1;
  static_assert(!kQ5 || kQ5Bytes % 16 == 0);
  const std::uint8_t* q5_src[kQ5Fetch];
  uint4 f_q5[kQ5Fetch];
  if constexpr (kQ5) {
#pragma unroll
    for (int i = 0; i < kQ5Fetch; ++i) {
      const int chunk = tid + (i * 256);
      const int r = r_block + (chunk / kQ5Parts);
      q5_src[i] = chunk < kQ5Chunks
                      ? w_expert +
                            (static_cast<std::size_t>(r < m_i ? r : m_i - 1) *
                             row_bytes) +
                            ((chunk % kQ5Parts) * 16)
                      : nullptr;
    }
  }

  // Activation fetch: BN tokens x (BK * 64) bytes per stage in 16-byte
  // chunks, eight per token; each thread fetches consecutive 256-chunk
  // strides, up to four for a 128-token tile.
  constexpr int kActChunks = BN * BK * 4;
  static_assert(kActChunks <= kActFetch * 256);
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src = (chunk < kActChunks && c_row < bucket_rows)
                                 ? rows_in[bucket_begin + c_row]
                                 : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8)
                        : nullptr;
    // s_act[(kb * 4 + quarter) * kActStride + t]
    a_slot[i] = chunk < kActChunks ? (sub * kActStride) + t : -1;
  }

  const auto swizzle = [](int row, int c) {
    return (row * kChunks) +
           (c ^ (kChunks == 4 ? ((row >> 1) & 3) : ((row >> 2) & 1)));
  };

  const auto fetch_stage = [&](int kb0) {
    if constexpr (kQ5) {
#pragma unroll
      for (int i = 0; i < kQ5Fetch; ++i) {
        f_q5[i] = q5_src[i] != nullptr
                      ? *reinterpret_cast<const uint4*>(q5_src[i] + (kb0 * 24))
                      : make_uint4(0u, 0u, 0u, 0u);
      }
    }
#pragma unroll
    for (int u = 0; u < (kQ5 ? 0 : kWaveRowTiles); ++u) {
      if constexpr (kQ8) {
        // block_q8_0 is 34 bytes, so the code loads are 2-byte aligned.
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * 34);
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        __builtin_memcpy(&f_codes[u], blk + 2, 16);
        __builtin_memcpy(&f_codes_hi[u], blk + 18, 16);
      } else {
        constexpr int kBlockChunks = kQ5K ? 11 : 9;
        constexpr int kCodeChunk = kQ5K ? 3 : 1;
        const int block = kb0 / 8;
        const auto* blk =
            reinterpret_cast<const uint4*>(f_ptr[u]) + (block * kBlockChunks);
        // The K sweep enters a new superblock every eight Q8-sized blocks.
        if (kb0 % 8 == 0) {
          f_header[u] = blk[0];
          if constexpr (kQ5K) {
            f_qh[u][0] = blk[1];
            f_qh[u][1] = blk[2];
          }
        }
        const int sb32 = (kb0 % 8) + f_c;
        f_codes[u] = blk[kCodeChunk + (sb32 / 2) * 2 + f_c];
        f_sb32[u] = sb32;
        if constexpr (kQ5K) {
          // Bit sb32 of the 32 high-bit bytes, packed as the Q5_1 word.
          const std::uint32_t qh[8] = {f_qh[u][0].x, f_qh[u][0].y, f_qh[u][0].z,
                                       f_qh[u][0].w, f_qh[u][1].x, f_qh[u][1].y,
                                       f_qh[u][1].z, f_qh[u][1].w};
          std::uint32_t high = 0;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            high |= GatherBit(qh[i], sb32) << (4 * i);
          }
          f_high[u] = high;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      a_data[i] = a_src[i] != nullptr
                      ? *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32))
                      : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  const auto commit_stage = [&]() {
    if constexpr (kQ5) {
#pragma unroll
      for (int i = 0; i < kQ5Fetch; ++i) {
        if (q5_src[i] != nullptr)
          s_codes[tid + (i * 256)] = f_q5[i];
      }
    }
#pragma unroll
    for (int u = 0; u < (kQ5 ? 0 : kWaveRowTiles); ++u) {
      const int row = (tid >> 1) + (u * 128);
      std::uint32_t scale_bias = 0;
      if constexpr (kQ8) {
        s_codes[swizzle(row, 2 * f_c)] = f_codes[u];
        s_codes[swizzle(row, (2 * f_c) + 1)] = f_codes_hi[u];
        scale_bias = f_live[u] ? f_dm[u] : 0U;  // half2 (d, 0)
      } else {
        s_codes[swizzle(row, f_c)] = f_codes[u];
      }
      if constexpr (kQ5K) {
        s_high[(f_c * BM) + row] = f_high[u];
      }
      if constexpr (kQ8) {
      } else {
        const int sb32 = f_sb32[u];
        // The first four scale/min bytes live in y/z. Their remaining
        // low nibbles share w; the high two bits reuse those same y/z bytes.
        // Decode the fixed words directly, avoiding header-member selection.
        const uint4 h = f_header[u];
        const unsigned shift = 8U * (static_cast<unsigned>(sb32) & 3U);
        const std::uint32_t scale_byte = h.y >> shift;
        const std::uint32_t min_byte = h.z >> shift;
        std::uint32_t sc;
        std::uint32_t mn;
        if (sb32 < 4) {
          sc = scale_byte & 0x3FU;
          mn = min_byte & 0x3FU;
        } else {
          const std::uint32_t packed = h.w >> shift;
          sc = (packed & 0x0FU) | ((scale_byte & 0xC0U) >> 2U);
          mn = ((packed >> 4U) & 0x0FU) | ((min_byte & 0xC0U) >> 2U);
        }
        const __half2 dm = __builtin_bit_cast(__half2, f_header[u].x);
        const float scale =
            f_live[u] ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
        const float offset =
            f_live[u] ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
        scale_bias = __builtin_bit_cast(std::uint32_t,
                                        __floats2half2_rn(scale, -offset));
      }
      s_scale[(f_c * BM) + row] = scale_bias;
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_slot[i] >= 0) {
        s_act[a_slot[i]] = a_data[i];
      }
    }
  };

  v8f acc[kWaveRowTiles][kTokTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      acc[u][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  const __half2 magic =
      __floats2half2_rn(kQ8 ? -1152.0F : -1024.0F, kQ8 ? -1152.0F : -1024.0F);
  const auto compute_stage = [&]() {
    uint4 raw[kWaveRowTiles][BK];
    if constexpr (!kQ8 && !kQ5) {
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
#pragma unroll
        for (int c = 0; c < BK; ++c) {
          raw[u][c] = s_codes[swizzle(row, c)];
        }
      }
    }
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
      {
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
          const int row = (wave_id * 16) + (u * 128) + sub_lane;
          // Q5_1 block kb of the row: d | m, the high-bit word, 16 codes.
          uint2 q5_head{};
          uint2 q5_lo{};
          uint2 q5_hi{};
          std::uint32_t scale_bias;
          if constexpr (kQ5) {
            const std::uint8_t* blk = lds + (row * kQ5Bytes) + (kb * 24);
            q5_head = *reinterpret_cast<const uint2*>(blk);
            q5_lo = *reinterpret_cast<const uint2*>(blk + 8);
            q5_hi = *reinterpret_cast<const uint2*>(blk + 16);
            scale_bias = r_block + row < m_i ? q5_head.x : 0U;
          } else {
            scale_bias = s_scale[(kb * BM) + row];
          }
          const __half2 sb = __builtin_bit_cast(__half2, scale_bias);
          const __half2 scale2 = __low2half2(sb);
          const __half2 bias2 = __high2half2(sb);
          std::uint32_t nib[8];
          if constexpr (kQ8) {
            // Q8_0: the block's 32 signed bytes are chunks 2 kb and 2 kb + 1;
            // flipping the sign bit carries q + 128, which the 1152 magic
            // takes back out.
            const uint4 c0 = s_codes[swizzle(row, 2 * kb)];
            const uint4 c1 = s_codes[swizzle(row, (2 * kb) + 1)];
            const std::uint32_t words[8] = {c0.x, c0.y, c0.z, c0.w,
                                            c1.x, c1.y, c1.z, c1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              nib[i] = words[i] ^ 0x80808080U;
            }
          } else if constexpr (kQ5) {
            // Q5_1: elements 0-15 take the low nibbles of the 16 code
            // bytes, 16-31 the high, plus bit j of the high-bit word.
            const std::uint32_t high = q5_head.y;
            const std::uint32_t words[4] = {q5_lo.x, q5_lo.y, q5_hi.x, q5_hi.y};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              nib[i] = (words[i] & 0x0F0F0F0FU) |
                       SpreadHighBits((high >> (4 * i)) & 0xFU);
              nib[4 + i] = ((words[i] >> 4U) & 0x0F0F0F0FU) |
                           SpreadHighBits((high >> (16 + 4 * i)) & 0xFU);
            }
          } else {
            // Q4_K / Q5_K: elements 0-15 of K block kb0 + kb are the low
            // (kb = 0) or high (kb = 1) nibbles of chunk 0, elements 16-31
            // of chunk 1; Q5_K adds bit j of the staged high-bit word.
            const unsigned shift = 4U * static_cast<unsigned>(kb);
            const std::uint32_t words[8] = {
                raw[u][0].x, raw[u][0].y, raw[u][0].z, raw[u][0].w,
                raw[u][1].x, raw[u][1].y, raw[u][1].z, raw[u][1].w};
            const std::uint32_t high = kQ5K ? s_high[(kb * BM) + row] : 0U;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              nib[i] = (words[i] >> shift) & 0x0F0F0F0FU;
              if constexpr (kQ5K) {
                nib[i] |= SpreadHighBits((high >> (4 * i)) & 0xFU);
              }
            }
          }
          __half2 h[16];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            CodesToHalves(nib[i], magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
          }
          __builtin_memcpy(&a_lo[u], &h[0], 32);
          __builtin_memcpy(&a_hi[u], &h[8], 32);
        }
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if constexpr ((kQ5 || kQ8) && BN >= 48) {
          // Keep one token tile's LDS fragments live at a time. Hoisting
          // all eight tiles spills registers and defeats the wider tile's
          // reuse of each weight decode. This is a compiler barrier only.
          asm volatile("" ::: "memory");
        }
        // A short expert bucket has no output in the remaining token
        // tiles, so omit their WMMA work.
        if constexpr ((kQ5 || kQ8) && BN >= 48 && kTokTiles > 1) {
          if (j >= live_tok_tiles)
            continue;
        }
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          b[q] = frag[q * kActStride];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
          acc[u][j] = Wmma(a_lo[u], b_lo, acc[u][j]);
          acc[u][j] = Wmma(a_hi[u], b_hi, acc[u][j]);
        }
      }
    }
  };

  const auto run_stage = [&](int kb0) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    compute_stage();
    __syncthreads();
  };
  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    run_stage(kb0);
  }

  // One wave writes a complete 128-byte line of F16 output. Padding the
  // shared row by two floats also makes the accumulator scatter conflict-free.
  // Narrow buckets keep the lighter wave-local epilogue below.
  if constexpr (BN >= 48) {
    static_assert(kLdsBytes >= 16 * (BM + 2) * sizeof(float));
    if (out_half != nullptr) {
      constexpr unsigned stride = BM + 2;
      float* scratch = reinterpret_cast<float*>(lds);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
          for (int l = 0; l < 8; ++l)
            scratch[sub_lane * stride + wave_id * 16 + u * 128 + 2 * l +
                    half_id] = acc[u][j][l];
        }
        __syncthreads();
#pragma unroll
        for (int round = 0; round < 16 * BM / (256 * 2); ++round) {
          const unsigned flat = (round * 256 + tid) * 2;
          const unsigned tr = flat / BM, row = flat % BM;
          const unsigned t = t_local + j * 16 + tr, r = r_block + row;
          if (t < unsigned(bucket_rows) && r < m) {
            const int dst = rows_out[bucket_begin + t];
            if (dst >= 0) {
              float2 v =
                  *reinterpret_cast<const float2*>(scratch + tr * stride + row);
              const size_t o = size_t(dst) * m + r;
              if (m % 2 == 0 && r + 1 < m)
                *reinterpret_cast<__half2*>(out_half + o) =
                    __floats2half2_rn(v.x, v.y);
              else {
                out_half[o] = __float2half(v.x);
                if (r + 1 < m)
                  out_half[o + 1] = __float2half(v.y);
              }
            }
          }
        }
        __syncthreads();
      }
      return;
    }
  }
  // Transpose each 16x16 tile through LDS, then scatter the 16 rows of each
  // token to its output row.
  float* tile_scratch = reinterpret_cast<float*>(lds) + (wave_id * 256);
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r0 = r_block + (wave_id * 16) + (u * 128);
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[u][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const int t0 = t_local + (j * 16);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = (s * 32) + lane_id;
        const int t = t0 + (flat >> 4);
        const int r = r0 + (flat & 15);
        if (t < bucket_rows && r < m_i) {
          const std::int32_t dst = rows_out[bucket_begin + t];
          if (dst >= 0) {
            const std::size_t o = (static_cast<std::size_t>(dst) * m) +
                                  static_cast<std::size_t>(r);
            float v = tile_scratch[flat];
            if (out_half != nullptr) {
              out_half[o] = __float2half(v);
            } else {
              out[o] = v;
            }
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

/// Compacts the routed assignments by expert with 16-row padded buckets.
/// One block: exclusive scan of the padded counts into pad_bounds[0..E].
__global__ void RoutedPadBoundsKernel(const std::uint32_t* __restrict__ counts,
                                      std::int32_t* __restrict__ pad_bounds,
                                      std::int32_t* __restrict__ cursors,
                                      std::int32_t* __restrict__ rows_token,
                                      std::int32_t* __restrict__ rows_slot,
                                      std::uint32_t n_experts) {
  __shared__ std::int32_t padded[1024];
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    padded[e] = static_cast<std::int32_t>((counts[e] + 15u) / 16u * 16u);
    cursors[e] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::int32_t running = 0;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      pad_bounds[e] = running;
      running += padded[e];
    }
    pad_bounds[n_experts] = running;
  }
  __syncthreads();
  // Scatter overwrites every occupied row. Only the padding can retain a
  // previous batch's assignments; clear it while the bucket bounds are hot.
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    for (std::uint32_t row = counts[e];
         row < static_cast<std::uint32_t>(padded[e]); ++row) {
      const auto at = pad_bounds[e] + row;
      rows_token[at] = -1;
      rows_slot[at] = -1;
    }
  }
}

/// rows_token[c] / rows_slot[c] for every routed (token, slot); the order
/// inside a bucket is whatever the atomics produce, which changes nothing:
/// every output row is computed from its own inputs only.
__global__ void RoutedScatterKernel(const std::int32_t* __restrict__ ids,
                                    const std::int32_t* __restrict__ pad_bounds,
                                    std::int32_t* __restrict__ cursors,
                                    std::int32_t* __restrict__ rows_token,
                                    std::int32_t* __restrict__ rows_slot,
                                    std::uint32_t slots, std::uint32_t k) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= slots) {
    return;
  }
  const std::int32_t e = ids[slot];
  if (e < 0) {
    return;
  }
  const std::int32_t c = pad_bounds[e] + atomicAdd(&cursors[e], 1);
  rows_token[c] = static_cast<std::int32_t>(slot / k);
  rows_slot[c] = static_cast<std::int32_t>(slot);
}

}  // namespace

void NarrowActivations(const float* x, void* out, std::size_t count,
                       hipStream_t stream) {
  hipLaunchKernelGGL(NarrowKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, static_cast<__half*>(out), count);
}

std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts) {
  return slots + (n_experts * (kRoutedTileTokens - 1));
}

void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream) {
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  hipLaunchKernelGGL(RoutedPadBoundsKernel, dim3(1), dim3(1024), 0, stream,
                     counts, pad_bounds, cursors, rows_token, rows_slot,
                     n_experts);
  hipLaunchKernelGGL(RoutedScatterKernel, dim3(Blocks(slots)), dim3(kThreads),
                     0, stream, ids, pad_bounds, cursors, rows_token, rows_slot,
                     static_cast<std::uint32_t>(slots), k);
}

template<int BN>
bool LaunchRoutedF16(const void* w, WeightType type, const __half* x,
                     const std::int32_t* tiles, std::uint32_t n_tiles,
                     const std::int32_t* pad_bounds,
                     const std::int32_t* rows_in, const std::int32_t* rows_out,
                     float* out, __half* out_half, std::size_t m, std::size_t k,
                     hipStream_t stream) {
  constexpr int kBM = 128;
  constexpr int kBK = 2;
  const dim3 grid(static_cast<unsigned int>((m + kBM - 1) / kBM), n_tiles);
  switch (type) {
    case WeightType::kQ4_K:
      if constexpr (BN > 48) {
        return false;
      } else {
        hipLaunchKernelGGL(
            (RoutedF16GEMMKernel<WeightType::kQ4_K, kBM, BN, kBK>), grid,
            dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in,
            rows_out, out, out_half, m, k);
        return true;
      }
    case WeightType::kQ5_1:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ5_1, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, rows_in, rows_out, out, out_half, m, k);
      return true;
    case WeightType::kQ8_0:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ8_0, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, rows_in, rows_out, out, out_half, m, k);
      return true;
    case WeightType::kQ5_K:
      if constexpr (BN > 48) {
        return false;
      } else {
        hipLaunchKernelGGL(
            (RoutedF16GEMMKernel<WeightType::kQ5_K, kBM, BN, kBK>), grid,
            dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in,
            rows_out, out, out_half, m, k);
        return true;
      }
    default:
      return false;
  }
}

bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   float* out, __half* out_half, std::size_t m, std::size_t k,
                   hipStream_t stream) {
  const std::size_t block_elems =
      (type == WeightType::kQ4_K || type == WeightType::kQ5_K) ? 256 : 64;
  if (m == 0 || k == 0 || k % block_elems != 0 || n_tiles == 0 ||
      (out_half == nullptr) == (out == nullptr)) {
    return false;
  }
  switch (tile_rows) {
    case 16:
      return LaunchRoutedF16<16>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, out, out_half, m, k,
                                 stream);
    case 48:
      return LaunchRoutedF16<48>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, out, out_half, m, k,
                                 stream);
    case 64:
      return LaunchRoutedF16<64>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, out, out_half, m, k,
                                 stream);
    default:
      return false;
  }
}

namespace {

/// Gemma 4's gelu_tanh(x) * u: the expression and build flags of its
/// GeGluValue, out of line like it, so the fused [gate | up] epilogue rounds
/// as Gemma's separate GeGLU pass does.
__device__ __attribute__((noinline)) float GemmaGeGluValue(float x, float u) {
  constexpr float kSqrt2OverPi = 0.79788456080286535587989211986876F;
  constexpr float kCoefA = 0.044715F;
  const float g =
      0.5F * x * (1.0F + tanhf(kSqrt2OverPi * x * (1.0F + kCoefA * x * x)));
  return g * u;
}

/// Gemma 4's HalfOf: the binary16 store its GeGLU pass writes.
__device__ __forceinline__ __half GemmaHalfOf(float v) {
  asm volatile("" : "+v"(v));
  return __float2half(v);
}

/// Dense F16 WMMA GEMM over Q8_0 or F16 weights: block = BM rows x BN tokens,
/// BK 32-element K blocks per LDS stage, waves = WM row groups x WN token
/// groups. The codes are dequantized to F16 once per stage as they are
/// committed to LDS (magic-number F16 construction, exact for a Q8_0 code),
/// and the activations are F16 rows [batch][k], so the matrix cores
/// accumulate in F32 with no per-block scaling. F16 weights skip decoding
/// and use the same ordered K16 products. y is [batch][m]. kRowGroup walks
/// that many row tiles of a token tile in launch order.
/// kGrouped (Q8_0): Gemma 4's grouped layout (DenseF16Gemm), the same blocks
/// at other addresses. kGeGlu (DenseF16GeGluGemm): W is Gemma 4's fused
/// [gate | up]; a wave's first two row tiles take gate rows and its last two
/// the matching up rows, and the epilogue writes binary16 GeGLU rows of
/// width m / 2 into `out_half`.
template<int BM, int BN, int BK, int WM, int WN, int kRowGroup = 1,
         bool kHalfWeights = false, bool kGrouped = false, bool kGeGlu = false>
__launch_bounds__(256) __global__
    void DenseF16GEMMKernel(const void* __restrict__ w,
                            const __half* __restrict__ x, float* __restrict__ y,
                            std::size_t batch, std::size_t m, std::size_t k,
                            __half* out_half = nullptr) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;
  // Weight and activation K blocks staged per thread; a unit past the
  // stage's block count is idle.
  constexpr int kAUnits = BM * BK;
  constexpr int kBUnits = BN * BK;
  constexpr int kAPer = (kAUnits + 255) / 256;
  constexpr int kBPer = (kBUnits + 255) / 256;

  // One K block of one row or token is four 16-byte chunks; the chunks of
  // nearby rows are permuted so a fragment read (one row per lane, 64-byte
  // stride) covers all bank groups.
  constexpr int kLdsChunks = BK * (BM + BN) * 4;
  constexpr int kTransposeChunks = 8 * 16 * 36 * sizeof(float) / sizeof(uint4);
  constexpr int kLdsStorage =
      kLdsChunks < kTransposeChunks ? kTransposeChunks : kLdsChunks;
  __shared__ __attribute__((aligned(16))) uint4 s_lds[kLdsStorage];
  auto* s_a = reinterpret_cast<uint4(*)[BM][4]>(s_lds);
  auto* s_b = reinterpret_cast<uint4(*)[BN][4]>(s_lds + BK * BM * 4);
  const auto swizzle = [](int row, int c) { return c ^ ((row >> 1) & 3); };

  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const auto* w_bytes = static_cast<const std::uint8_t*>(w);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;
  // The tile grid's row count is divisible by kRowGroup; every dot product
  // keeps its K order.
  const unsigned row_group = blockIdx.y / kRowGroup;
  const unsigned within = (blockIdx.y % kRowGroup) * gridDim.x + blockIdx.x;
  const int r_block = (row_group * kRowGroup + within % kRowGroup) * BM;
  const int t_block = (within / kRowGroup) * BN;
  const auto token_of = [&](int local) { return t_block + local; };

  // Thread tid stages row (p*256+tid)/BK and K block (p*256+tid)%BK; rows
  // past m read the last row with a zero scale.
  static_assert(!kGrouped || !kHalfWeights);
  const std::uint8_t* a_ptr[kAPer];
  bool a_live[kAPer];
  // kGrouped: 128-row groups store 256-value super-blocks (eight blocks) of
  // all their rows back to back; a_ptr is the row's first super-block and
  // a_group the bytes between its super-blocks.
  [[maybe_unused]] std::size_t a_group[kGrouped ? kAPer : 1];
#pragma unroll
  for (int p = 0; p < kAPer; ++p) {
    const int idx = (p * 256) + tid;
    const int r = r_block + idx / BK;
    a_live[p] = idx < kAUnits && r < m_i;
    int weight_row = r;
    if constexpr (kGeGlu) {
      // Block row l: wave row group l / 64, gate rows for l % 64 < 32, then
      // the up rows of the same outputs.
      const int l = r - r_block;
      const int out = r_block / 2 + (l / 64) * 32 + l % 32;
      weight_row = l % 64 < 32 ? out : out + m_i / 2;
      a_live[p] = idx < kAUnits && out < m_i / 2;
    }
    if constexpr (kGrouped) {
      const auto row =
          static_cast<std::size_t>(a_live[p] ? weight_row : (m_i - 1));
      const std::size_t rows = min(
          std::size_t{128}, static_cast<std::size_t>(m_i) - row / 128 * 128);
      a_group[p] = rows * 272;
      a_ptr[p] =
          w_bytes +
          (row / 128 * 128 * static_cast<std::size_t>(num_kb / 8) + row % 128) *
              272;
    } else {
      a_ptr[p] = w_bytes +
                 static_cast<std::size_t>(a_live[p] ? weight_row : (m_i - 1)) *
                     static_cast<std::size_t>(num_kb) *
                     (kHalfWeights ? 64 : 34);
    }
  }
  const __half* b_ptr[kBPer];
#pragma unroll
  for (int p = 0; p < kBPer; ++p) {
    const int idx = (p * 256) + tid;
    const int t = token_of(idx / BK);
    b_ptr[p] = idx < kBUnits && t < static_cast<int>(batch)
                   ? x + (static_cast<std::size_t>(t) * k)
                   : nullptr;
  }
  uint4 a_codes[kAPer][2];
  uint4 a_half[kHalfWeights ? kAPer : 1][4];
  std::uint32_t a_d[kAPer];
  uint4 b_data[kBPer][4];

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const bool live = a_live[p] && kb < num_kb;
      if constexpr (kHalfWeights) {
        const auto* src = reinterpret_cast<const uint4*>(
            a_ptr[p] + static_cast<std::size_t>(kb) * 64);
#pragma unroll
        for (int c = 0; c < 4; ++c)
          a_half[p][c] = live ? src[c] : make_uint4(0u, 0u, 0u, 0u);
      } else {
        const auto kb_live = static_cast<std::size_t>(live ? kb : (num_kb - 1));
        const std::uint8_t* blk;
        if constexpr (kGrouped) {
          blk = a_ptr[p] + kb_live / 8 * a_group[p] + kb_live % 8 * 34;
        } else {
          blk = a_ptr[p] + kb_live * 34;
        }
        a_d[p] = live ? *reinterpret_cast<const std::uint16_t*>(blk) : 0U;
        __builtin_memcpy(&a_codes[p][0], blk + 2, 16);
        __builtin_memcpy(&a_codes[p][1], blk + 18, 16);
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      if (b_ptr[p] != nullptr && kb < num_kb) {
        const auto* src = reinterpret_cast<const uint4*>(b_ptr[p] + (kb * 32));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = src[c];
        }
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  const __half2 magic = __floats2half2_rn(-1152.0F, -1152.0F);
  const __half2 zero2 = __floats2half2_rn(0.0F, 0.0F);
  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kAUnits) {
        break;
      }
      const int row = idx / BK;
      const int kk = idx % BK;
      if constexpr (kHalfWeights) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          s_a[kk][row][swizzle(row, c)] = a_half[p][c];
      } else {
        // Q8_0 codes are signed; flipping the sign bit carries q + 128, which
        // the 1152 magic takes back out.
        const std::uint32_t words[8] = {
            a_codes[p][0].x, a_codes[p][0].y, a_codes[p][0].z, a_codes[p][0].w,
            a_codes[p][1].x, a_codes[p][1].y, a_codes[p][1].z, a_codes[p][1].w};
        const __half2 scale2 = __half2half2(
            __builtin_bit_cast(__half, static_cast<std::uint16_t>(a_d[p])));
        __half2 h[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          CodesToHalves(words[i] ^ 0x80808080U, magic, scale2, zero2, h[2 * i],
                        h[2 * i + 1]);
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          uint4 v;
          __builtin_memcpy(&v, &h[4 * c], 16);
          s_a[kk][row][swizzle(row, c)] = v;
        }
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kBUnits) {
        break;
      }
      const int t = idx / BK;
      const int kk = idx % BK;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        s_b[kk][t][swizzle(t, c)] = b_data[p][c];
      }
    }
  };

  v8f acc[kWaveRowTiles][kWaveTokTiles];
  // Separate K16 chains reduce FP32 accumulation error for the sensitive
  // unquantized projections, with the same order in every chunk.
  v8f acc_high[kHalfWeights ? kWaveRowTiles : 1]
              [kHalfWeights ? kWaveTokTiles : 1]{};
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      acc[i][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  // Full tiles stay free of per-K bounds branches.
  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    // Keeps the next stage's loads ahead of the matrix work: scheduled
    // freely, the compiler sinks them below it and the commit waits on
    // the full memory latency.
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int row = (((wave_row * kWaveRowTiles) + i) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_a[kb][row][swizzle(row, q)];
        }
        __builtin_memcpy(&a_lo[i], &c[0], 32);
        __builtin_memcpy(&a_hi[i], &c[2], 32);
      }
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int t = (((wave_tok * kWaveTokTiles) + j) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_b[kb][t][swizzle(t, q)];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &c[0], 32);
        __builtin_memcpy(&b_hi, &c[2], 32);
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          if constexpr (kHalfWeights) {
            acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
            acc_high[i][j] = Wmma(a_hi[i], b_hi, acc_high[i][j]);
          } else {
            acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
            acc[i][j] = Wmma(a_hi[i], b_hi, acc[i][j]);
          }
        }
      }
    }
    __syncthreads();
  }

  if constexpr (kHalfWeights) {
#pragma unroll
    for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j)
        acc[i][j] += acc_high[i][j];
    }
  }

  // Transpose the result through LDS, two row tiles at a time, so every
  // global store covers 32 consecutive rows of one token: a full 128-byte
  // line (half lines cost a read-modify-write on the fabric).
  if constexpr (kGeGlu) {
    static_assert(BM / WM == 64);
    static_assert(kWaveRowTiles == 4);
    // scratch[token][output] per wave: 16 tokens of the wave's 32 outputs.
    constexpr unsigned kStride = 36;
    static_assert(8 * 16 * kStride * sizeof(float) <= sizeof(s_lds));
    float* scratch = reinterpret_cast<float*>(s_lds) + wave_id * 16 * kStride;
    const std::size_t half_m = m / 2;
    const std::size_t r0 =
        static_cast<std::size_t>(r_block / 2) + wave_row * 32;
    // m / 2 is a multiple of 32: a wave's outputs are all live or none.
    if (r0 >= half_m) {
      return;
    }
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        scratch[sub_lane * kStride + 2 * l + half_id] =
            GemmaGeGluValue(acc[0][j][l], acc[2][j][l]);
        scratch[sub_lane * kStride + 16 + 2 * l + half_id] =
            GemmaGeGluValue(acc[1][j][l], acc[3][j][l]);
      }
      __builtin_amdgcn_wave_barrier();
      // Lane pair (2p, 2p + 1) stores token p's 32 outputs as binary16.
      const int tok_l = lane_id >> 1;
      const int col = (lane_id & 1) * 16;
      const std::size_t tok = static_cast<std::size_t>(
          token_of(((wave_tok * kWaveTokTiles) + j) * 16 + tok_l));
      if (tok < batch) {
        __half h[16];
#pragma unroll
        for (int v = 0; v < 16; ++v) {
          h[v] = GemmaHalfOf(scratch[tok_l * kStride + col + v]);
        }
        uint4 packed[2];
        __builtin_memcpy(packed, h, sizeof(packed));
        auto* dst = reinterpret_cast<uint4*>(out_half + tok * half_m + r0 +
                                             static_cast<std::size_t>(col));
        dst[0] = packed[0];
        dst[1] = packed[1];
      }
      __builtin_amdgcn_wave_barrier();
    }
    return;
  }

  // Four padding floats reduce scatter bank conflicts and retain float4
  // alignment for the output stores.
  constexpr unsigned kOutputStride = 36;
  static_assert(8 * 16 * kOutputStride * sizeof(float) <= sizeof(s_lds));
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  float* tile_scratch =
      reinterpret_cast<float*>(s_lds) + wave_id * 16 * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      // scratch[token][row], with 32 output rows per token.
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * kOutputStride) + (2 * l) + half_id] =
            acc[i][j][l];
        tile_scratch[(sub_lane * kOutputStride) + 16 + (2 * l) + half_id] =
            acc[i + 1][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          static_cast<std::size_t>(r_block) +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 = static_cast<std::size_t>(
          token_of(((wave_tok * kWaveTokTiles) + j) * 16));
      // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
      const int tok_l = lane_id >> 1;
      const int row_l = (lane_id & 1) * 16;
      const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
      const auto* src = reinterpret_cast<const float4*>(
          tile_scratch + (tok_l * kOutputStride) + row_l);
      if (tok < batch && r0 + 32 <= m && ((tok * m) % 4 == 0)) {
        auto* dst = reinterpret_cast<float4*>(y + (tok * m) + r0 +
                                              static_cast<std::size_t>(row_l));
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          dst[q] = src[q];
        }
      } else if (tok < batch) {
#pragma unroll
        for (int q = 0; q < 16; ++q) {
          const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
          if (r < m) {
            y[(tok * m) + r] =
                tile_scratch[(tok_l * kOutputStride) + row_l + q];
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

}  // namespace

bool UnquantizedF16Gemm(const void* w, const __half* x, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream, std::size_t wide_from) {
  if (m == 0 || batch == 0 || k == 0 || k % 32 != 0)
    return false;
  // Wide batches reuse each staged fragment across twice the rows and
  // tokens. Every output keeps its two K16 chains in K order, so the tile
  // choice never changes a result.
  if (batch >= wide_from) {
    hipLaunchKernelGGL((DenseF16GEMMKernel<128, 128, 2, 2, 4, 1, true>),
                       dim3((batch + 127) / 128, (m + 127) / 128),
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    return true;
  }
  hipLaunchKernelGGL((DenseF16GEMMKernel<64, 64, 2, 2, 4, 1, true>),
                     dim3((batch + 63) / 64, (m + 63) / 64), dim3(kThreads), 0,
                     stream, w, x, out, batch, m, k);
  return true;
}

bool DenseF16Gemm(const void* w, const __half* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream,
                  DenseF16Plan plan, bool grouped) {
  if (m == 0 || k == 0 || batch == 0 || k % 32 != 0) {
    return false;
  }
  if (grouped) {
    // Gemma 4's grouped Q8_0 weights: the same tiles as below for wide
    // projections, each block read from its grouped address.
    if (k % 256 != 0 || m <= 512)
      return false;
    if (batch < 96) {
      hipLaunchKernelGGL((DenseF16GEMMKernel<128, 64, 4, 4, 2, 1, false, true>),
                         dim3((batch + 63) / 64, (m + 127) / 128),
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
      return true;
    }
    const dim3 grid(static_cast<unsigned int>((batch + 127) / 128),
                    static_cast<unsigned int>((m + 255) / 256));
    const bool planned = plan != DenseF16Plan::kAuto && batch >= 1024;
    const auto launch = [&]<int BK, int WM, int WN, int kRowGroup>() {
      hipLaunchKernelGGL(
          (DenseF16GEMMKernel<256, 128, BK, WM, WN, kRowGroup, false, true>),
          grid, dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    };
    if (planned && plan == DenseF16Plan::kStagedRowGroups8Walk7 &&
        grid.y % 7 == 0) {
      launch.template operator()<2, 8, 1, 7>();
    } else if (planned && (plan == DenseF16Plan::kStagedRowGroups8 ||
                           plan == DenseF16Plan::kStagedRowGroups8Walk7)) {
      launch.template operator()<2, 8, 1, 1>();
    } else if (planned) {
      launch.template operator()<1, 8, 1, 1>();
    } else {
      launch.template operator()<1, 4, 2, 1>();
    }
    return true;
  }
  // A 128-token macro tile for wide batches, 64-row tiles for narrow
  // projections, 64 tokens below 96.
  constexpr int kBM = 128;
  if (m <= 512 && batch >= 96) {
    constexpr int kBN = 128;
    constexpr int kNarrowBM = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kNarrowBM - 1) / kNarrowBM));
    hipLaunchKernelGGL((DenseF16GEMMKernel<kNarrowBM, kBN, 2, 2, 4>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  } else if (batch >= 96) {
    // 64 x 64 wave tiles: half the LDS fragment bytes per matrix product
    // of the 32 x 64 tile (the F16 fragments are twice the int8 ones).
    constexpr int kWideBM = 256;
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kWideBM - 1) / kWideBM));
    const bool planned = plan != DenseF16Plan::kAuto && batch >= 1024;
    if (planned && plan == DenseF16Plan::kStagedRowGroups8Walk7 &&
        grid.y % 7 == 0) {
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 2, 8, 1, 7>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else if (planned && (plan == DenseF16Plan::kStagedRowGroups8 ||
                           plan == DenseF16Plan::kStagedRowGroups8Walk7)) {
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 2, 8, 1>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else if (planned) {
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 1, 8, 1>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else {
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 1, 4, 2>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    }
  } else {
    constexpr int kBN = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((DenseF16GEMMKernel<kBM, kBN, 4, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  }
  return true;
}

bool DenseF16GeGluGemm(const void* w, const __half* x, __half* out,
                       std::size_t batch, std::size_t m, std::size_t k,
                       hipStream_t stream, bool grouped, bool half_weights) {
  if (batch < 96 || m == 0 || m % 64 != 0 || k == 0 ||
      k % (grouped ? 256 : 32) != 0 || out == nullptr ||
      (grouped && half_weights)) {
    return false;
  }
  // DenseF16Gemm's wide plan for the Q8_0 [gate | up] shape (a block
  // computes 128 outputs, the last one fewer), UnquantizedF16Gemm's for
  // binary16 weights (64 outputs).
  const unsigned token_tiles = static_cast<unsigned int>((batch + 127) / 128);
  const auto launch =
      [&]<int BM, int BK, int WM, int WN, bool kGrouped, bool kHalf>() {
        const dim3 grid(token_tiles, static_cast<unsigned int>(
                                         (m / 2 + BM / 2 - 1) / (BM / 2)));
        hipLaunchKernelGGL(
            (DenseF16GEMMKernel<BM, 128, BK, WM, WN, 1, kHalf, kGrouped, true>),
            grid, dim3(kThreads), 0, stream, w, x, nullptr, batch, m, k, out);
      };
  if (half_weights) {
    launch.template operator()<128, 2, 2, 4, false, true>();
  } else if (grouped) {
    launch.template operator()<256, 1, 4, 2, true, false>();
  } else {
    launch.template operator()<256, 1, 4, 2, false, false>();
  }
  return true;
}

template<unsigned Keep>
void SelectMtpCandidates(const float* logits, std::uint32_t* ids,
                         std::uint32_t* scratch_ids, float* scores,
                         std::uint32_t vocab, hipStream_t stream) {
  if (logits == nullptr || ids == nullptr || scratch_ids == nullptr ||
      vocab == 0) {
    throw std::invalid_argument("invalid MTP candidate selection");
  }
  const auto tiles = [](std::uint32_t n) {
    return 1U + (n - 1U) / kMtpCandidateTile;
  };
  unsigned passes = 1;
  for (auto size = vocab; size > kMtpCandidateTile; size = tiles(size) * Keep) {
    ++passes;
  }
  // Choose the first buffer so the final pass always lands in `ids`.
  auto* destination = passes % 2 != 0 ? ids : scratch_ids;
  const std::uint32_t* source = nullptr;
  auto size = vocab;
  for (;;) {
    const auto blocks = tiles(size);
    hipLaunchKernelGGL((MtpCandidateTileKernel<Keep>), dim3(blocks),
                       dim3(kThreads), 0, stream, logits, source, destination,
                       blocks == 1 ? scores : nullptr, size, vocab);
    if (blocks == 1) {
      break;
    }
    size = blocks * Keep;
    source = destination;
    destination = destination == ids ? scratch_ids : ids;
  }
}

std::uint32_t MtpCandidateWorkspaceSize(std::uint32_t vocab) {
  const auto tiles = (vocab + 1023) / 1024;
  return (tiles > 2 ? tiles : 2) * kMtpCandidates;
}

void MtpTopCandidates(const float* logits, std::uint32_t* ids,
                      std::uint32_t* scratch_ids, float* scores,
                      std::uint32_t vocab, hipStream_t stream) {
  if (scores == nullptr)
    throw std::invalid_argument("invalid MTP candidate scores");
  SelectMtpCandidates<kMtpCandidates>(logits, ids, scratch_ids, scores, vocab,
                                      stream);
}

}  // namespace gufo::models::gemma4::rocm
