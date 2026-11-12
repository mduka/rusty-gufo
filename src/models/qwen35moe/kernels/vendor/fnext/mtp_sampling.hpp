#include "src/models/qwen35moe/kernels/vendor/ns.hpp"
#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_SAMPLING_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_SAMPLING_HPP_

#include <array>
#include <cmath>
#include <cstdint>
#include "src/core/sampling.hpp"
#include <span>
#include <stdexcept>

namespace gufo::models::qwen35moe::models::qwen38_flash_next {

enum class DraftDecision { kAccept, kReject, kStop };

inline constexpr std::size_t kMtpCandidates = 64;

struct MtpCandidateLogits {
  std::array<sampling::TokenId, kMtpCandidates> ids{};
  std::array<float, kMtpCandidates> logits{};
  std::size_t size{0};
};

// rusty-gufo: SampleMtpProposal and the verification helpers that
// followed here are host sampling (Gufo's SamplerState); the Rust
// engine implements them. Only the kernels' constants remain.

}  // namespace gufo::models::qwen35moe::models::qwen38_flash_next

#endif
