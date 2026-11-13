// Kernels of the AudioVAE V2 port (vae.cpp): operators of OpenBMB/VoxCPM's
// AudioVAE V2 (Apache-2.0, Copyright OpenBMB; see licenses/VoxCPM.txt) and
// the Snake activation of Descript's DAC (MIT, Copyright Descript; see
// licenses/DAC.txt), rewritten in HIP.

#include <hip/hip_runtime.h>

#include "vae_kernels.hpp"

namespace gufo::models::voxcpm::kernels {
namespace {

constexpr int kThreads = 256;

dim3 Blocks(std::size_t count) {
  return dim3(static_cast<unsigned>((count + kThreads - 1) / kThreads));
}

__global__ void SnakeKernel(const float* input, float* output,
                            std::size_t count, std::size_t channels,
                            const float* alpha, const float* scale,
                            const float* bias) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const std::size_t c = index % channels;
  float y = input[index];
  if (scale != nullptr)
    y = y * scale[c] + bias[c];
  const float a = alpha[c];
  const float s = sinf(a * y);
  output[index] = y + (1.0F / (a + 1e-9F)) * (s * s);
}

__global__ void DepthwiseConvKernel(const float* input, const float* weight,
                                    const float* bias, float* output,
                                    std::size_t count, std::size_t channels,
                                    std::size_t kernel,
                                    std::size_t dilation) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const std::size_t t = index / channels;
  const std::size_t c = index % channels;
  float sum = bias[c];
  for (std::size_t j = 0; j < kernel; ++j)
    sum += weight[c * kernel + j] * input[(t + j * dilation) * channels + c];
  output[index] = sum;
}

__global__ void ConvFromMonoKernel(const float* samples, const float* weight,
                                   const float* bias, float* output,
                                   std::size_t count, std::size_t channels,
                                   std::size_t kernel) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const std::size_t t = index / channels;
  const std::size_t c = index % channels;
  float sum = bias[c];
  for (std::size_t j = 0; j < kernel; ++j)
    sum += weight[c * kernel + j] * samples[t + j];
  output[index] = sum;
}

// One block per output frame.
__global__ void ConvToMonoKernel(const float* input, const float* weight,
                                 float bias, float* output,
                                 std::size_t channels, std::size_t kernel) {
  __shared__ float partial[kThreads];
  const std::size_t t = blockIdx.x;
  float sum = 0.0F;
  for (std::size_t index = threadIdx.x; index < channels * kernel;
       index += blockDim.x) {
    const std::size_t c = index / kernel;
    const std::size_t j = index % kernel;
    sum += weight[index] * input[(t + j) * channels + c];
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    output[t] = tanhf(partial[0] + bias);
}

__global__ void AddBiasF32Kernel(float* data, const float* bias,
                                 std::size_t count, std::size_t width) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    data[index] += bias[index % width];
}

}  // namespace

void Snake(const float* input, float* output, std::size_t frames,
           std::size_t channels, const float* alpha, const float* scale,
           const float* bias, hipStream_t stream) {
  const std::size_t count = frames * channels;
  if (count == 0)
    return;
  SnakeKernel<<<Blocks(count), kThreads, 0, stream>>>(
      input, output, count, channels, alpha, scale, bias);
}

void DepthwiseConv(const float* input, const float* weight, const float* bias,
                   float* output, std::size_t frames, std::size_t channels,
                   std::size_t kernel, std::size_t dilation,
                   hipStream_t stream) {
  const std::size_t count = frames * channels;
  if (count == 0)
    return;
  DepthwiseConvKernel<<<Blocks(count), kThreads, 0, stream>>>(
      input, weight, bias, output, count, channels, kernel, dilation);
}

void ConvFromMono(const float* samples, const float* weight, const float* bias,
                  float* output, std::size_t frames, std::size_t channels,
                  std::size_t kernel, hipStream_t stream) {
  const std::size_t count = frames * channels;
  if (count == 0)
    return;
  ConvFromMonoKernel<<<Blocks(count), kThreads, 0, stream>>>(
      samples, weight, bias, output, count, channels, kernel);
}

void ConvToMono(const float* input, const float* weight, float bias,
                float* output, std::size_t frames, std::size_t channels,
                std::size_t kernel, hipStream_t stream) {
  if (frames == 0)
    return;
  ConvToMonoKernel<<<static_cast<unsigned>(frames), kThreads, 0, stream>>>(
      input, weight, bias, output, channels, kernel);
}

void AddBiasF32(float* data, const float* bias, std::size_t rows,
                std::size_t width, hipStream_t stream) {
  const std::size_t count = rows * width;
  if (count == 0)
    return;
  AddBiasF32Kernel<<<Blocks(count), kThreads, 0, stream>>>(data, bias, count,
                                                           width);
}

}  // namespace gufo::models::voxcpm::kernels
