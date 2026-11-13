#include <hip/hip_runtime.h>

#include <cstdint>

#include "convert_kernels.hpp"
#include "device_math.hpp"

namespace gufo::models::voxcpm::kernels {
namespace {

__global__ void HalfToBf16Kernel(const std::uint16_t* input,
                                 std::uint16_t* output, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = Bf16Bits(HalfToFloat(input[index]));
}

__global__ void FloatToBf16Kernel(const float* input, std::uint16_t* output,
                                  std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = Bf16Bits(input[index]);
}

__global__ void Q8_0ToBf16Kernel(const std::uint8_t* blocks,
                                 std::uint16_t* output, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const std::uint8_t* block = blocks + (index / 32) * 34;
  const std::uint16_t scale_bits =
      static_cast<std::uint16_t>(block[0] | (block[1] << 8U));
  const auto value = static_cast<std::int8_t>(block[2 + index % 32]);
  output[index] = Bf16Bits(HalfToFloat(scale_bits) * static_cast<float>(value));
}

}  // namespace

void HalfToBf16(const std::uint16_t* input, std::uint16_t* output,
                std::size_t count, hipStream_t stream) {
  if (count > 0)
    HalfToBf16Kernel<<<BlocksOf256(count), 256, 0, stream>>>(input, output,
                                                             count);
}

void FloatToBf16(const float* input, std::uint16_t* output, std::size_t count,
                 hipStream_t stream) {
  if (count > 0)
    FloatToBf16Kernel<<<BlocksOf256(count), 256, 0, stream>>>(input, output,
                                                              count);
}

void Q8_0ToBf16(const std::uint8_t* blocks, std::uint16_t* output,
                std::size_t count, hipStream_t stream) {
  if (count > 0)
    Q8_0ToBf16Kernel<<<BlocksOf256(count), 256, 0, stream>>>(blocks, output,
                                                             count);
}

}  // namespace gufo::models::voxcpm::kernels
