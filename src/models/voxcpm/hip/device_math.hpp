#ifndef GUFO_MODELS_VOXCPM_KERNELS_DEVICE_MATH_HPP_
#define GUFO_MODELS_VOXCPM_KERNELS_DEVICE_MATH_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// Conversions shared by the device code (HIP sources only).
namespace gufo::models::voxcpm::kernels {

/// float32 to bfloat16 bits, rounded to nearest even; NaNs stay quiet NaNs.
__device__ inline std::uint16_t Bf16Bits(float value) {
  const std::uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0)
    return static_cast<std::uint16_t>((bits >> 16U) | 0x40U);
  const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

/// float16 bits to float32.
__device__ inline float HalfToFloat(std::uint16_t bits) {
  return static_cast<float>(__builtin_bit_cast(_Float16, bits));
}

/// Blocks of 256 threads covering `count` elements.
inline dim3 BlocksOf256(std::size_t count) {
  return dim3(static_cast<unsigned>((count + 255) / 256));
}

}  // namespace gufo::models::voxcpm::kernels

#endif  // GUFO_MODELS_VOXCPM_KERNELS_DEVICE_MATH_HPP_
