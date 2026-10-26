#ifndef GUFO_CORE_SAMPLING_HPP_
#define GUFO_CORE_SAMPLING_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace gufo::sampling {

// The sampling types the HIP launchers share with the Rust engines
// (gufo-core `sampling`); the sampler itself is Rust.

using TokenId = std::uint32_t;

/// One token's penalty state, three 32-bit words as the kernels read it.
struct TokenPenalty {
  TokenId token{0};
  std::uint32_t generated_count{0};
  std::uint32_t repeated{0};
};

struct LogitCandidate {
  TokenId token;
  float value;
};

}  // namespace gufo::sampling

#endif  // GUFO_CORE_SAMPLING_HPP_
