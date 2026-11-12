// Element-wise and layout kernels of the DAC decoder, ported from Hugging
// Face transformers (Apache-2.0, see licenses/transformers.txt).
#include "src/models/omnivoice/hip/codec_kernels.hpp"

namespace gufo::models::omnivoice::kernels {
namespace {

constexpr unsigned kThreads = 256;

__global__ void SnakeKernel(const float* input, std::size_t input_stride,
                            const float* alpha, float* output,
                            std::size_t length, std::size_t pad) {
  const std::size_t channel = blockIdx.y;
  const std::size_t width = length + 2 * pad;
  const std::size_t t = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (t >= width)
    return;
  float value = 0.0F;
  if (t >= pad && t < pad + length) {
    const float x = input[channel * input_stride + t - pad];
    const float a = alpha[channel];
    const float s = sinf(a * x);
    value = x + (1.0F / (a + 1e-9F)) * (s * s);
  }
  output[channel * width + t] = value;
}

__global__ void AddChannelBiasKernel(float* data, std::size_t stride,
                                     const float* bias, std::size_t length) {
  const std::size_t channel = blockIdx.y;
  const std::size_t t = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (t < length)
    data[channel * stride + t] += bias[channel];
}

__global__ void AddColumnBiasKernel(float* data, const float* bias,
                                    std::size_t columns) {
  float* row = data + blockIdx.x * columns;
  for (std::size_t j = threadIdx.x; j < columns; j += blockDim.x)
    row[j] += bias[j];
}

__global__ void Col2ImKernel(const float* columns, const float* bias,
                             float* output, std::size_t channels,
                             std::size_t input_length, std::size_t stride,
                             std::size_t kernel, std::size_t pad) {
  const std::size_t channel = blockIdx.y;
  const std::size_t length = input_length * stride;
  const std::size_t t = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (t >= length)
    return;
  float sum = bias[channel];
  // Taps k with t + pad - k = j * stride for 0 <= j < input_length.
  const std::size_t shifted = t + pad;
  for (std::size_t k = shifted % stride; k < kernel; k += stride) {
    if (k > shifted)
      break;
    const std::size_t j = (shifted - k) / stride;
    if (j < input_length)
      sum += columns[(k * channels + channel) * input_length + j];
  }
  output[channel * length + t] = sum;
}

__global__ void CodebookSumKernel(const float* table,
                                  const std::int32_t* codes, float* output,
                                  std::size_t codebooks, std::size_t entries,
                                  std::size_t frames, std::size_t width) {
  const std::size_t frame = blockIdx.x;
  for (std::size_t h = threadIdx.x; h < width; h += blockDim.x) {
    float sum = 0.0F;
    for (std::size_t c = 0; c < codebooks; ++c) {
      const auto code = static_cast<std::size_t>(codes[c * frames + frame]);
      sum += table[(c * entries + code) * width + h];
    }
    output[frame * width + h] = sum;
  }
}

unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

}  // namespace

void Snake(const float* input, std::size_t input_stride, const float* alpha,
           float* output, std::size_t channels, std::size_t length,
           std::size_t pad, hipStream_t stream) {
  if (channels == 0 || length == 0)
    return;
  const dim3 grid(Blocks(length + 2 * pad), static_cast<unsigned>(channels));
  SnakeKernel<<<grid, kThreads, 0, stream>>>(input, input_stride, alpha, output,
                                            length, pad);
}

void AddChannelBias(float* data, std::size_t stride, const float* bias,
                    std::size_t channels, std::size_t length,
                    hipStream_t stream) {
  if (channels == 0 || length == 0)
    return;
  const dim3 grid(Blocks(length), static_cast<unsigned>(channels));
  AddChannelBiasKernel<<<grid, kThreads, 0, stream>>>(data, stride, bias, length);
}

void AddColumnBias(float* data, const float* bias, std::size_t rows,
                   std::size_t columns, hipStream_t stream) {
  if (rows == 0)
    return;
  AddColumnBiasKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
      data, bias, columns);
}

void Col2Im(const float* columns, const float* bias, float* output,
            std::size_t channels, std::size_t input_length, std::size_t stride,
            std::size_t kernel, std::size_t pad, hipStream_t stream) {
  if (channels == 0 || input_length == 0)
    return;
  const dim3 grid(Blocks(input_length * stride), static_cast<unsigned>(channels));
  Col2ImKernel<<<grid, kThreads, 0, stream>>>(columns, bias, output, channels,
                                             input_length, stride, kernel, pad);
}

void CodebookSum(const float* table, const std::int32_t* codes, float* output,
                 std::size_t codebooks, std::size_t entries,
                 std::size_t frames, std::size_t width, hipStream_t stream) {
  if (frames == 0)
    return;
  CodebookSumKernel<<<static_cast<unsigned>(frames), kThreads, 0, stream>>>(
      table, codes, output, codebooks, entries, frames, width);
}

}  // namespace gufo::models::omnivoice::kernels
