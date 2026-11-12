// Excerpt of Gufo src/families/qwen/hip_common/quant_ops.hpp at upstream
// 1c75887b (gufo-org/gufo): the paths Gemma 4 dispatches for the formats its
// loader accepts, in the gemma4 extension namespace.
// Gemma 4 additions from the fork's feat/gemma (83b764b4) are applied.
#ifndef GUFO_MODELS_GEMMA4_HIP_GGML_QUANT_OPS_HPP_
#define GUFO_MODELS_GEMMA4_HIP_GGML_QUANT_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>

namespace gufo::models::gemma4::ggml {

// Shared quantized block layouts + quant row-dot helpers for the decode and
// prefill GPU paths, matching the CPU oracles in ggml_dequant.cpp.

// block_q8_0 layout: {__half d; int8_t qs[32];}, 34 bytes, QK=32.
constexpr std::size_t kQ8_0BlockSize = 32;

struct Q8_0Block {
  __half d;
  std::int8_t qs[kQ8_0BlockSize];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");

// opt-r4-q5k-q6k: block_q5_K layout ({half d; half dmin; uint8 scales[12];
// uint8 qh[32]; uint8 qs[128]; }, 176 bytes, QK_K=256). The (scale, min)
// pair comes from GetQKScaleMin. Requires K % 256 == 0.
constexpr std::size_t kQ5KBlockSize = 256;

struct Q5KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q5KBlock) == 176, "block_q5_K must be 176 bytes");

// opt-r4-q5k-q6k: block_q6_K layout ({uint8 ql[128]; uint8 qh[64]; int8
// scales[16]; half d; }, 210 bytes, QK_K=256): packed 4-bit ql + 2-bit qh
// with one int8 scale per sixteen values. Requires K % 256 == 0.
constexpr std::size_t kQ6KBlockSize = 256;

struct Q6KBlock {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  __half d;
};
static_assert(sizeof(Q6KBlock) == 210, "block_q6_K must be 210 bytes");

// opt-q4kxl: the K-quants and Q4_0 are decoded in-kernel from their packed
// form rather than pre-dequantized to BF16.

// block_q4_K layout ({half d; half dmin; uint8 scales[12]; uint8 qs[128];},
// 144 bytes, QK_K=256). Eight 32-wide sub-blocks, each with a 6-bit scale and
// a 6-bit minimum: value = d*sc*q - dmin*m.
constexpr std::size_t kQ4KBlockSize = 256;

struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

// block_q4_0 layout ({half d; uint8 qs[16];}, 18 bytes, QK=32). Linear 4-bit
// values d * (q - 8): low nibbles hold elements 0..15, high nibbles 16..31.
constexpr std::size_t kQ4_0BlockSize = 32;

struct Q4_0Block {
  __half d;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q4_0Block) == 18, "block_q4_0 must be 18 bytes");

// opt-r2-q8_0: type-aware quant block geometry helpers so RowDot can own the
// per-type row stride (bytes per block and elements per block).
__device__ inline std::size_t QuantBlockBytes(core::GgmlType t) {
  switch (t) {
    case core::GgmlType::kQ8_0:
      return sizeof(Q8_0Block);
    case core::GgmlType::kQ5_K:
      return sizeof(Q5KBlock);
    case core::GgmlType::kQ6_K:
      return sizeof(Q6KBlock);
    case core::GgmlType::kQ4_K:
      return sizeof(Q4KBlock);
    case core::GgmlType::kQ4_0:
      return sizeof(Q4_0Block);
    default:
      return 0;
  }
}

__device__ inline std::size_t QuantBlockQK(core::GgmlType t) {
  switch (t) {
    case core::GgmlType::kQ8_0:
      return kQ8_0BlockSize;
    case core::GgmlType::kQ5_K:
      return kQ5KBlockSize;
    case core::GgmlType::kQ6_K:
      return kQ6KBlockSize;
    case core::GgmlType::kQ4_K:
      return kQ4KBlockSize;
    case core::GgmlType::kQ4_0:
      return kQ4_0BlockSize;
    default:
      return 0;
  }
}

// opt-r4-q5k-q6k: unpack the (scale, minimum) pair for Q5_K from the packed
// 12-byte scales array. index 0..7; mirrors CPU GetQ4ScaleMin in
// ggml_dequant.cpp (Q5_K uses the same scale/min encoding as Q4_K).
template<bool SharedScales = false>
__device__ inline void GetQKScaleMin(std::size_t index,
                                     const std::uint8_t* packed,
                                     std::uint8_t& sc,
                                     std::uint8_t& m) noexcept {
  // Batched projections amortize one extra byte load across token rows and
  // benefit from avoiding divergent execution of the two encodings. Scalar
  // GEMV keeps the fewer-load spelling.
  if constexpr (SharedScales) {
    const std::size_t base = index & 3U;
    const unsigned low = packed[base];
    const unsigned middle = packed[base + 4];
    const unsigned high = packed[base + 8];
    const unsigned upper_mask = 0U - static_cast<unsigned>(index >> 2U);
    const unsigned lower_scale = low & 63U;
    const unsigned upper_scale = (high & 15U) | ((low >> 6U) << 4U);
    const unsigned lower_minimum = middle & 63U;
    const unsigned upper_minimum = (high >> 4U) | ((middle >> 6U) << 4U);
    sc = static_cast<std::uint8_t>(lower_scale ^
                                   ((lower_scale ^ upper_scale) & upper_mask));
    m = static_cast<std::uint8_t>(
        lower_minimum ^ ((lower_minimum ^ upper_minimum) & upper_mask));
    return;
  }
  if (index < 4) {
    sc = packed[index] & 0x3FU;
    m = packed[index + 4] & 0x3FU;
    return;
  }
  sc = static_cast<std::uint8_t>((packed[index + 4] & 0x0FU) |
                                 ((packed[index - 4] >> 6U) << 4U));
  m = static_cast<std::uint8_t>((packed[index + 4] >> 4U) |
                                ((packed[index] >> 6U) << 4U));
}

/// Byte-wise subtraction of a small constant from four packed bytes.
/// Setting bit 7 of every byte first stops a borrow from crossing into the next
/// byte, and the final XOR removes it again, so each byte independently becomes
/// its two's-complement value. Every use here feeds a value that is known to
/// fit in int8 after the subtraction.
__device__ inline std::uint32_t SubBytes(std::uint32_t v,
                                         std::uint32_t bias) noexcept {
  return ((v | 0x80808080U) - bias) ^ 0x80808080U;
}

// opt-q4kxl: one 16-element sub-block decoded into int8 codes plus an affine
// pair, so that
//
//     w[j] = scale * code[j] - offset      for j in [0, 16)
//
// This is the single primitive the decode GEMV, the prefill WMMA GEMM and the
// embedding lookup all build on. Sixteen is the smallest scale group any of
// these formats uses (Q6_K carries one scale per sixteen elements, the
// rest per thirty-two or per super-block) AND it is exactly the K depth of one
// `wmma_16x16x16_iu8` fragment, so the GEMM never needs a scale that changes
// inside a fragment. `offset` is non-zero only for the two formats that encode
// an explicit minimum (Q4_K and Q5_K); everything else is symmetric.
//
// A 32-element variant was tried, sharing the super-block header (and the
// branchy scale/min extraction) between the two halves, which Q4_K and Q5_K
// decode identically. It left prefill unchanged and cost decode 13%
// (tg128 8.25 -> 7.18): holding thirty-two decoded bytes plus three scales
// pushed the GEMV kernels from 88 to 120-136 VGPRs, and those kernels are
// DRAM-bound, so the halved occupancy cost more memory-level parallelism than
// the saved header work returned. The decode path wants the small footprint
// more than it wants the shared header.
struct QuantSub16 {
  union {
    std::int8_t q[16];
    std::uint32_t w[4];
  };
  float scale;
  float offset;
};

/// Loads sixteen consecutive weight bytes as four 32-bit words. Quant blocks
/// are not word aligned (block_q4_K is 144 bytes, block_q6_K 210), so this must
/// go through memcpy rather than a reinterpreting load.
__device__ inline void LoadQuantWords16(const std::uint8_t* __restrict__ src,
                                        std::uint32_t out[4]) noexcept {
  __builtin_memcpy(out, src, 16);
}

/// Decodes the `sub16`-th 16-element sub-block of the row starting at `row`.
/// `row` must point at the first block of the row for `type`.
template<bool SharedScales = false>
__device__ inline void DecodeQuantSub16(core::GgmlType type,
                                        const void* __restrict__ row,
                                        std::size_t sub16,
                                        QuantSub16& out) noexcept {
  out.offset = 0.0F;
  switch (type) {
    case core::GgmlType::kQ8_0: {
      const auto& blk = static_cast<const Q8_0Block*>(row)[sub16 / 2];
      const std::size_t base = (sub16 % 2) * 16;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        out.q[j] = blk.qs[base + j];
      }
      out.scale = __half2float(blk.d);
      return;
    }
    case core::GgmlType::kQ4_K: {
      const auto& blk = static_cast<const Q4KBlock*>(row)[sub16 / 16];
      const std::size_t sb32 = (sub16 / 2) % 8;
      const std::size_t base = ((sb32 / 2) * 32) + ((sub16 % 2) * 16);
      const unsigned shift = 4U * static_cast<unsigned>(sb32 & 1U);
      std::uint32_t packed[4];
      LoadQuantWords16(blk.qs + base, packed);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        out.w[i] = (packed[i] >> shift) & 0x0F0F0F0FU;
      }
      std::uint8_t sc = 0;
      std::uint8_t m = 0;
      GetQKScaleMin<SharedScales>(sb32, blk.scales, sc, m);
      out.scale = __half2float(blk.d) * static_cast<float>(sc);
      out.offset = __half2float(blk.dmin) * static_cast<float>(m);
      return;
    }
    case core::GgmlType::kQ5_K: {
      const auto& blk = static_cast<const Q5KBlock*>(row)[sub16 / 16];
      const std::size_t sb32 = (sub16 / 2) % 8;
      const std::size_t lane0 = (sub16 % 2) * 16;
      const std::size_t base = ((sb32 / 2) * 32) + lane0;
      const unsigned shift = ((sb32 % 2) == 0) ? 0U : 4U;
      std::uint32_t packed[4];
      std::uint32_t high[4];
      LoadQuantWords16(blk.qs + base, packed);
      LoadQuantWords16(blk.qh + lane0, high);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const std::uint32_t low = (packed[i] >> shift) & 0x0F0F0F0FU;
        const std::uint32_t bit =
            (high[i] >> static_cast<unsigned>(sb32)) & 0x01010101U;
        out.w[i] = low | (bit << 4U);
      }
      std::uint8_t sc = 0;
      std::uint8_t m = 0;
      GetQKScaleMin<SharedScales>(sb32, blk.scales, sc, m);
      out.scale = __half2float(blk.d) * static_cast<float>(sc);
      out.offset = __half2float(blk.dmin) * static_cast<float>(m);
      return;
    }
    case core::GgmlType::kQ6_K: {
      const auto& blk = static_cast<const Q6KBlock*>(row)[sub16 / 16];
      const std::size_t sb32 = (sub16 / 2) % 8;
      const std::size_t half = sb32 / 4;
      const std::size_t segment = sb32 % 4;
      const std::size_t lane0 = (sub16 % 2) * 16;
      const std::size_t ql_base =
          (half * 64) + (((segment & 1U) != 0U) ? 32 : 0);
      const unsigned ql_shift = (segment >= 2) ? 4U : 0U;
      const unsigned qh_shift = 2U * static_cast<unsigned>(segment);
      std::uint32_t low_bits[4];
      std::uint32_t high_bits[4];
      LoadQuantWords16(blk.ql + ql_base + lane0, low_bits);
      LoadQuantWords16(blk.qh + (half * 32) + lane0, high_bits);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const std::uint32_t low = (low_bits[i] >> ql_shift) & 0x0F0F0F0FU;
        const std::uint32_t high = (high_bits[i] >> qh_shift) & 0x03030303U;
        // Each byte is 0..63 here, so the biased subtract lands in [-32, 31].
        out.w[i] = SubBytes(low | (high << 4U), 0x20202020U);
      }
      const std::size_t scale_index = (half * 8) + (segment * 2) + (sub16 % 2);
      out.scale =
          __half2float(blk.d) * static_cast<float>(blk.scales[scale_index]);
      return;
    }
    case core::GgmlType::kQ4_0: {
      const auto& blk = static_cast<const Q4_0Block*>(row)[sub16 / 2];
      const unsigned shift = 4U * static_cast<unsigned>(sub16 % 2);
      std::uint32_t packed[4];
      LoadQuantWords16(blk.qs, packed);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        // Each byte is 0..15 here, so the biased subtract lands in [-8, 7].
        out.w[i] = SubBytes((packed[i] >> shift) & 0x0F0F0F0FU, 0x08080808U);
      }
      out.scale = __half2float(blk.d);
      return;
    }
    default:
      break;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out.w[i] = 0;
  }
  out.scale = 0.0F;
}

/// Bytes occupied by one logical row of `k` elements in `type`.
__device__ inline std::size_t QuantRowBytes(core::GgmlType type,
                                            std::size_t k) noexcept {
  const std::size_t qk = QuantBlockQK(type);
  return (qk == 0) ? 0 : ((k / qk) * QuantBlockBytes(type));
}

/// Single-element dequantization for any DecodeQuantSub16-backed format.
/// `row` points at the first block of the row that contains `index`.
__device__ inline float QuantBlockElement(core::GgmlType type,
                                          const void* __restrict__ row,
                                          std::size_t index) noexcept {
  QuantSub16 sub;
  DecodeQuantSub16(type, row, index / 16, sub);
  return (sub.scale * static_cast<float>(sub.q[index % 16])) - sub.offset;
}

/// True when `type` is decoded through DecodeQuantSub16 rather than one of the
/// hand-rolled per-type branches in QuantWarpBlockDot.
__device__ inline bool IsSub16DecodedQuant(core::GgmlType t) noexcept {
  return t == core::GgmlType::kQ4_K || t == core::GgmlType::kQ5_K ||
         t == core::GgmlType::kQ6_K || t == core::GgmlType::kQ4_0;
}

/// Template argument of QuantWarpSub16Dot meaning "decode the runtime type".
inline constexpr core::GgmlType kRuntimeSub16Type = core::GgmlType::kF32;

/// The sub16 route of QuantWarpBlockDot: each lane owns the sub-blocks
/// lane_id, lane_id + 32, ... of [s0, s1), then the warp reduces. Type fixes
/// the format at compile time, or is kRuntimeSub16Type to decode `type`.
template<core::GgmlType Type, typename TX>
__device__ inline float QuantWarpSub16Dot(core::GgmlType type,
                                          const char* __restrict__ row,
                                          const TX* __restrict__ x,
                                          std::size_t s0, std::size_t s1,
                                          std::size_t lane_id) {
  const core::GgmlType decode_type = (Type == kRuntimeSub16Type) ? type : Type;
  float sumf = 0.0F;
  QuantSub16 sub;
  for (std::size_t sb = s0 + lane_id; sb < s1; sb += 32) {
    DecodeQuantSub16(decode_type, row, sb, sub);
    const TX* xb = x + (sb * 16);
    float dot = 0.0F;
    float xsum = 0.0F;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float xv = static_cast<float>(xb[j]);
      dot += static_cast<float>(sub.q[j]) * xv;
      xsum += xv;
    }
    sumf += (sub.scale * dot) - (sub.offset * xsum);
  }
  for (int off = 16; off > 0; off >>= 1) {
    sumf += __shfl_xor(sumf, off);
  }
  return sumf;
}

// opt-r7-decode-parallel: warp-parallel quant row-dot, templated on the input
// element type TX (float in decode, hip_bfloat16 for the prefill bf16 path).
// ONE warp cooperates: all 32 lanes process a slice of the row's quant blocks
// (b in [b0,b1)), each lane owning 8 elements per 256-wide block
// or 1 element per 32-wide block (Q8_0), then reduce within each
// block via __shfl_xor. Returns this warp's partial row-dot ONLY IN LANE 0
// (other lanes return 0), so it is a drop-in for both the one-warp-per-row
// kernels (a) and the one-block-per-row kernels (b) whose existing __shfl_xor
// / shared-memory reductions then broadcast/combine it correctly. With
// num_warps == 1 the warp processes the entire row. Mirrors Q8KBlockGEMVKernel
// exactly. The only TX-dependent code is the x reads, wrapped in
// static_cast<float> so TX=float is identity and TX=hip_bfloat16 converts.
template<typename TX>
__device__ __forceinline__ float QuantWarpBlockDot(
    core::GgmlType type, const void* __restrict__ base, std::size_t row_idx,
    const TX* __restrict__ x, std::size_t K, std::size_t lane_id,
    std::size_t warp_id, std::size_t num_warps) {
  const std::size_t qk = QuantBlockQK(type);
  const std::size_t block_bytes = QuantBlockBytes(type);
  const char* row =
      static_cast<const char*>(base) + (row_idx * (K / qk * block_bytes));
  const std::size_t num_blocks = K / qk;
  const std::size_t chunk = (num_blocks + num_warps - 1) / num_warps;
  const std::size_t b0 = warp_id * chunk;
  const std::size_t b1 = (b0 + chunk < num_blocks) ? (b0 + chunk) : num_blocks;
  float sumf = 0.0F;
  switch (type) {
    case core::GgmlType::kQ8_0: {
      const auto* row_blocks = reinterpret_cast<const Q8_0Block*>(row);
      for (std::size_t b = b0; b < b1; ++b) {
        const Q8_0Block& wblk = row_blocks[b];
        const float d_w = __half2float(wblk.d);
        const TX* xb = x + (b * kQ8_0BlockSize);
        float dot = static_cast<float>(wblk.qs[lane_id]) *
                    static_cast<float>(xb[lane_id]);
        for (int off = 16; off > 0; off >>= 1) {
          dot += __shfl_xor(dot, off);
        }
        sumf += d_w * dot;
      }
      break;
    }
    // opt-q4kxl: Q5_K and Q6_K take the sub16 route below, which unpacks a
    // whole 16-element group with word-wide bit operations instead of per
    // element.
    default: {
      // opt-q4kxl: generic route for every format decoded through
      // DecodeQuantSub16 (Q4_K, Q5_K, Q6_K, Q4_0). Instead of
      // splitting one 256-wide block across all 32 lanes eight elements at a
      // time, each lane owns whole 16-element sub-blocks. A sub-block is the
      // unit the decoder produces, so this pays the scale/nibble unpacking once
      // per sixteen values rather than once per value, and every lane's sixteen
      // weight bytes are contiguous.
      //
      // Sixteen and not thirty-two: these kernels are DRAM-bound, and the wider
      // decode raised them from 88 to 120-136 VGPRs, which cut occupancy and
      // measured 13% slower end to end.
      //
      // The K-quants get a copy of the loop specialized on the format, which
      // drops the per-sub-block type switch: the Q4_K_M GDN projections go
      // from 98 to 93 us per layer, with the same arithmetic.
      const std::size_t s0 = b0 * (qk / 16);
      const std::size_t s1 = b1 * (qk / 16);
      switch (type) {
        case core::GgmlType::kQ4_K:
          sumf = QuantWarpSub16Dot<core::GgmlType::kQ4_K>(type, row, x, s0, s1,
                                                          lane_id);
          break;
        case core::GgmlType::kQ5_K:
          sumf = QuantWarpSub16Dot<core::GgmlType::kQ5_K>(type, row, x, s0, s1,
                                                          lane_id);
          break;
        case core::GgmlType::kQ6_K:
          sumf = QuantWarpSub16Dot<core::GgmlType::kQ6_K>(type, row, x, s0, s1,
                                                          lane_id);
          break;
        default:
          if (IsSub16DecodedQuant(type)) {
            sumf = QuantWarpSub16Dot<kRuntimeSub16Type>(type, row, x, s0, s1,
                                                        lane_id);
          }
          break;
      }
      break;
    }
  }
  if (lane_id != 0)
    return 0.0F;
  return sumf;
}

}  // namespace gufo::models::gemma4::ggml

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_GEMMA4_HIP_GGML_QUANT_OPS_HPP_
