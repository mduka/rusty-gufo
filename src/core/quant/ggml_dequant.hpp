#ifndef GUFO_CORE_QUANT_GGML_DEQUANT_HPP_
#define GUFO_CORE_QUANT_GGML_DEQUANT_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::quant {

// The GGML quantized block layouts the HIP launchers and kernels share; the
// Rust side (gufo-formats) uses the same layouts. Field order and byte sizes
// match the HIP path (src/models/qwen/hip/quant_ops.hpp). Do NOT redefine
// block_* locally. Sizes are static_asserted.
#pragma pack(push, 1)
struct block_q4_K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};

struct block_q5_K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};

struct block_q6_K {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  std::uint16_t d;
};

struct block_q3_K {
  std::uint8_t hmask[32];
  std::uint8_t qs[64];
  std::uint8_t scales[12];
  std::uint16_t d;
};

struct block_q8_K {
  float d;
  std::int8_t qs[256];
  std::int16_t bsums[16];
};

// Q8_0: fp16 scale + 32 int8 quantized values (QK=32).
struct block_q8_0 {
  std::uint16_t d;
  std::int8_t qs[32];
};

// IQ4_NL: fp16 scale + 32 non-linear 4-bit codebook indices (QK=32). The
// codebook is kValuesIq4Nl; the index nibble order matches Q4_0 (low nibbles
// give the first 16 elements, high nibbles the second 16).
struct block_iq4_nl {
  std::uint16_t d;
  std::uint8_t qs[16];
};

// IQ4_XS: super-block of 256 with eight 6-bit sub-block scales split across
// scales_h (2 high bits each) and scales_l (4 low bits each), plus 128 bytes
// of packed non-linear 4-bit codebook indices.
struct block_iq4_xs {
  std::uint16_t d;
  std::uint16_t scales_h;
  std::uint8_t scales_l[4];
  std::uint8_t qs[128];
};

// IQ3_S: super-block of 256. Each group of 8 elements indexes the 512-entry
// kIq3sGrid table (8 low bits in qs, 1 high bit in qh) and carries a per-group
// sign byte; scales holds two 4-bit sub-block scales per byte.
struct block_iq3_s {
  std::uint16_t d;
  std::uint8_t qs[64];
  std::uint8_t qh[8];
  std::uint8_t signs[32];
  std::uint8_t scales[4];
};
#pragma pack(pop)

static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");
static_assert(sizeof(block_q5_K) == 176, "block_q5_K must be 176 bytes");
static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 bytes");
static_assert(sizeof(block_q3_K) == 110, "block_q3_K must be 110 bytes");
static_assert(sizeof(block_q8_K) == 292, "block_q8_K must be 292 bytes");
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");
static_assert(sizeof(block_iq4_nl) == 18, "block_iq4_nl must be 18 bytes");
static_assert(sizeof(block_iq4_xs) == 136, "block_iq4_xs must be 136 bytes");
static_assert(sizeof(block_iq3_s) == 110, "block_iq3_s must be 110 bytes");

/// Non-linear 4-bit codebook shared by IQ4_NL and IQ4_XS.
inline constexpr std::int8_t kValuesIq4Nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

/// Returns the logical elements represented by one supported quant block, or
/// zero for dense and unsupported storage types.
[[nodiscard]] constexpr std::size_t QuantizedBlockElements(
    core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kQ4_0:
    case core::GgmlType::kQ4_1:
    case core::GgmlType::kQ5_0:
    case core::GgmlType::kQ5_1:
    case core::GgmlType::kQ8_1:
    case core::GgmlType::kQ8_0:
    case core::GgmlType::kIQ4_NL:
      return 32;
    case core::GgmlType::kQ2_K:
    case core::GgmlType::kIQ2_XXS:
    case core::GgmlType::kIQ4_XS:
    case core::GgmlType::kIQ3_S:
    case core::GgmlType::kQ3_K:
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
    case core::GgmlType::kQ8_K:
      return 256;
    default:
      return 0;
  }
}

/// Returns the encoded byte count for one logical quantized row, or zero when
/// the type is not quantized or the element count is not block aligned.
[[nodiscard]] inline std::size_t QuantizedRowBytes(
    core::GgmlType type, std::size_t elements) noexcept {
  const std::size_t block_qk = QuantizedBlockElements(type);
  std::size_t block_bytes = 0;
  switch (type) {
    // GGML storage layouts also used by the DeepSeek runtime. Storage support
    // here does not imply that every model has a compute kernel for the type.
    case core::GgmlType::kQ4_0:
      block_bytes = 18;
      break;
    case core::GgmlType::kQ4_1:
      block_bytes = 20;
      break;
    case core::GgmlType::kQ5_0:
      block_bytes = 22;
      break;
    case core::GgmlType::kQ5_1:
      block_bytes = 24;
      break;
    case core::GgmlType::kQ8_1:
      block_bytes = 36;
      break;
    case core::GgmlType::kQ2_K:
      block_bytes = 84;
      break;
    case core::GgmlType::kIQ2_XXS:
      block_bytes = 66;
      break;
    case core::GgmlType::kQ3_K:
      block_bytes = sizeof(block_q3_K);
      break;
    case core::GgmlType::kQ4_K:
      block_bytes = sizeof(block_q4_K);
      break;
    case core::GgmlType::kQ5_K:
      block_bytes = sizeof(block_q5_K);
      break;
    case core::GgmlType::kQ6_K:
      block_bytes = sizeof(block_q6_K);
      break;
    case core::GgmlType::kQ8_K:
      block_bytes = sizeof(block_q8_K);
      break;
    case core::GgmlType::kQ8_0:
      block_bytes = sizeof(block_q8_0);
      break;
    case core::GgmlType::kIQ4_NL:
      block_bytes = sizeof(block_iq4_nl);
      break;
    case core::GgmlType::kIQ4_XS:
      block_bytes = sizeof(block_iq4_xs);
      break;
    case core::GgmlType::kIQ3_S:
      block_bytes = sizeof(block_iq3_s);
      break;
    default:
      return 0;
  }
  if ((elements % block_qk) != 0) {
    return 0;
  }
  const std::size_t blocks = elements / block_qk;
  if (blocks > std::numeric_limits<std::size_t>::max() / block_bytes) {
    return 0;
  }
  return blocks * block_bytes;
}

}  // namespace gufo::quant

#endif  // GUFO_CORE_QUANT_GGML_DEQUANT_HPP_
