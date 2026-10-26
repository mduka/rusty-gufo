#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_SAMPLING_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_SAMPLING_HPP_

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "src/core/sampling.hpp"

namespace gufo::models::qwen38_flash_next {

// Candidates per MTP draft row the kernels write; proposal sampling and
// verification are Rust (gufo-qwen38-flash-next `policy`).
inline constexpr std::size_t kMtpCandidates = 64;

}  // namespace gufo::models::qwen38_flash_next

#endif
