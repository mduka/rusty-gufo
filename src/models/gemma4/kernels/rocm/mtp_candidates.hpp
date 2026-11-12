// The constant half_gemm.hip.cpp takes from the extension's mtp_sampling.hpp
// (whose sampler code is host code, ported to Rust): MtpTopCandidates
// selects this many candidates per drafter row.
#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_MTP_CANDIDATES_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_MTP_CANDIDATES_HPP_

#include <cstddef>

namespace gufo::models::gemma4 {

inline constexpr std::size_t kMtpCandidates = 64;

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_MTP_CANDIDATES_HPP_
