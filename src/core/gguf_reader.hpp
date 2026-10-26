#ifndef GUFO_CORE_GGUF_READER_HPP_
#define GUFO_CORE_GGUF_READER_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace gufo::core {

// The GGML storage types the HIP launchers take (`format` parameters of
// launch.toml); GGUF reading itself is Rust (gufo-core `gguf`).

/// GGML tensor types used by supported model artifacts.
enum class GgmlType : std::uint16_t {
  kF32 = 0,
  kF16 = 1,
  kQ4_0 = 2,
  kQ4_1 = 3,
  kQ5_0 = 6,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ8_1 = 9,
  kQ2_K = 10,
  kQ3_K = 11,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
  kQ8_K = 15,
  kIQ2_XXS = 16,
  kIQ4_NL = 20,
  kIQ3_S = 21,
  kIQ4_XS = 23,
  kI32 = 26,
  kBF16 = 30,
};

}  // namespace gufo::core

#endif  // GUFO_CORE_GGUF_READER_HPP_
