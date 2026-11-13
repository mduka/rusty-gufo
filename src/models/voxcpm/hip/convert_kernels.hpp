#ifndef GUFO_MODELS_VOXCPM_KERNELS_CONVERT_KERNELS_HPP_
#define GUFO_MODELS_VOXCPM_KERNELS_CONVERT_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// Load-time conversion of GGUF weights to the bfloat16 the kernels read.
namespace gufo::models::voxcpm::kernels {

/// output[i] = bf16(input[i]) for float16 input.
void HalfToBf16(const std::uint16_t* input, std::uint16_t* output,
                std::size_t count, hipStream_t stream);

/// output[i] = bf16(input[i]) for float32 input.
void FloatToBf16(const float* input, std::uint16_t* output, std::size_t count,
                 hipStream_t stream);

/// Dequantizes GGML Q8_0 (blocks of a float16 scale and 32 int8 values,
/// 34 bytes) to bfloat16: output[i] = bf16(scale * value).
void Q8_0ToBf16(const std::uint8_t* blocks, std::uint16_t* output,
                std::size_t count, hipStream_t stream);

}  // namespace gufo::models::voxcpm::kernels

#endif  // GUFO_MODELS_VOXCPM_KERNELS_CONVERT_KERNELS_HPP_
