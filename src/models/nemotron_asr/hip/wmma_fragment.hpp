#ifndef GUFO_MODELS_NEMOTRON_ASR_KERNELS_WMMA_FRAGMENT_HPP_
#define GUFO_MODELS_NEMOTRON_ASR_KERNELS_WMMA_FRAGMENT_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Operands of v_wmma_f32_16x16x16_f16 in wave32 (gfx11): lane L holds row
// L % 16 of A and column L % 16 of B with 16 consecutive k, replicated in
// both half-waves; element i of its sums is row 2i + L / 16, column L % 16.
namespace gufo::models::nemotron_asr::kernels {

using Fragment = _Float16 __attribute__((ext_vector_type(16)));
using Accumulator = float __attribute__((ext_vector_type(8)));

/// 16 float16 values from 16-byte aligned memory.
__device__ inline Fragment LoadFragment(const void* p) {
  union {
    Fragment v;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *static_cast<const uint4*>(p);
  cvt.u[1] = *(static_cast<const uint4*>(p) + 1);
  return cvt.v;
}

/// float16 of a float32, saturated at the largest finite value: activations
/// stored as float16 never become infinite.
__device__ inline __half SaturatedHalf(float value) {
  return __float2half(fminf(fmaxf(value, -65504.0F), 65504.0F));
}

}  // namespace gufo::models::nemotron_asr::kernels

#endif  // GUFO_MODELS_NEMOTRON_ASR_KERNELS_WMMA_FRAGMENT_HPP_
