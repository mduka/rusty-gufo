// Kernels of the HuBERT, semantic and DAC encoders and the RVQ, ported from
// Hugging Face transformers (Apache-2.0, see licenses/transformers.txt).
#include "src/models/omnivoice/hip/encoder_kernels.hpp"

#include <cfloat>

namespace gufo::models::omnivoice::kernels {
namespace {

constexpr unsigned kThreads = 256;

__device__ float BlockSum(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  const float total = scratch[0];
  __syncthreads();
  return total;
}

__device__ float BlockMax(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    __syncthreads();
  }
  const float total = scratch[0];
  __syncthreads();
  return total;
}

__device__ float GeluValue(float x) {
  return 0.5F * x * (1.0F + erff(x * 0.70710678118654752440F));
}

__global__ void Im2ColKernel(const float* input, std::size_t length,
                             std::size_t kernel, std::size_t stride,
                             std::size_t dilation, std::size_t pad,
                             std::size_t output_length, float* columns) {
  const std::size_t row = blockIdx.y;  // channel * kernel + tap
  const std::size_t channel = row / kernel;
  const std::size_t tap = row % kernel;
  const std::size_t t = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (t >= output_length)
    return;
  const long long position = static_cast<long long>(t * stride + tap * dilation) -
                             static_cast<long long>(pad);
  float value = 0.0F;
  if (position >= 0 && position < static_cast<long long>(length))
    value = input[channel * length + static_cast<std::size_t>(position)];
  columns[row * output_length + t] = value;
}

__global__ void GroupNormGeluKernel(float* data, const float* weight,
                                    const float* bias, std::size_t length,
                                    float epsilon) {
  __shared__ float scratch[kThreads];
  float* row = data + blockIdx.x * length;
  float sum = 0.0F;
  for (std::size_t t = threadIdx.x; t < length; t += blockDim.x)
    sum += row[t];
  const float mean = BlockSum(sum, scratch) / static_cast<float>(length);
  float squares = 0.0F;
  for (std::size_t t = threadIdx.x; t < length; t += blockDim.x) {
    const float centered = row[t] - mean;
    squares += centered * centered;
  }
  const float variance = BlockSum(squares, scratch) / static_cast<float>(length);
  const float inverse = rsqrtf(variance + epsilon);
  const float scale = weight[blockIdx.x];
  const float shift = bias[blockIdx.x];
  for (std::size_t t = threadIdx.x; t < length; t += blockDim.x)
    row[t] = GeluValue((row[t] - mean) * inverse * scale + shift);
}

__global__ void GeluKernel(float* data, std::size_t count) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count)
    data[i] = GeluValue(data[i]);
}

__global__ void EluKernel(const float* input, float* output, std::size_t count) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    const float x = input[i];
    output[i] = x > 0.0F ? x : expm1f(x);
  }
}

__global__ void TransposeKernel(const float* input, std::size_t rows,
                                std::size_t columns, float* output,
                                std::size_t output_stride) {
  __shared__ float tile[32][33];
  const std::size_t base_row = blockIdx.y * 32;
  const std::size_t base_column = blockIdx.x * 32;
  for (std::size_t y = threadIdx.y; y < 32; y += blockDim.y) {
    const std::size_t r = base_row + y;
    const std::size_t c = base_column + threadIdx.x;
    if (r < rows && c < columns)
      tile[y][threadIdx.x] = input[r * columns + c];
  }
  __syncthreads();
  for (std::size_t y = threadIdx.y; y < 32; y += blockDim.y) {
    const std::size_t c = base_column + y;
    const std::size_t r = base_row + threadIdx.x;
    if (r < rows && c < columns)
      output[c * output_stride + r] = tile[threadIdx.x][y];
  }
}

__global__ void LayerNormKernel(const float* input, const float* weight,
                                const float* bias, float* output,
                                std::size_t width, float epsilon) {
  __shared__ float scratch[kThreads];
  const float* in = input + blockIdx.x * width;
  float* out = output + blockIdx.x * width;
  float sum = 0.0F;
  for (std::size_t j = threadIdx.x; j < width; j += blockDim.x)
    sum += in[j];
  const float mean = BlockSum(sum, scratch) / static_cast<float>(width);
  float squares = 0.0F;
  for (std::size_t j = threadIdx.x; j < width; j += blockDim.x) {
    const float centered = in[j] - mean;
    squares += centered * centered;
  }
  const float inverse =
      rsqrtf(BlockSum(squares, scratch) / static_cast<float>(width) + epsilon);
  for (std::size_t j = threadIdx.x; j < width; j += blockDim.x)
    out[j] = (in[j] - mean) * inverse * weight[j] + bias[j];
}

__global__ void SoftmaxInPlaceKernel(float* scores, std::size_t columns,
                                     float scale) {
  __shared__ float scratch[kThreads];
  float* row = scores + blockIdx.x * columns;
  float maximum = -FLT_MAX;
  for (std::size_t j = threadIdx.x; j < columns; j += blockDim.x)
    maximum = fmaxf(maximum, row[j] * scale);
  maximum = BlockMax(maximum, scratch);
  float sum = 0.0F;
  for (std::size_t j = threadIdx.x; j < columns; j += blockDim.x)
    sum += expf(row[j] * scale - maximum);
  const float inverse = 1.0F / BlockSum(sum, scratch);
  for (std::size_t j = threadIdx.x; j < columns; j += blockDim.x)
    row[j] = expf(row[j] * scale - maximum) * inverse;
}

__global__ void AccumulateKernel(const float* source, float* target,
                                 std::size_t count) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count)
    target[i] += source[i];
}

__global__ void ScaleKernel(float* data, std::size_t count, float scale) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count)
    data[i] *= scale;
}

__global__ void NearestEntryKernel(const float* dots, const float* projected,
                                   const float* norms, std::int32_t* indices,
                                   std::size_t entries, std::size_t dim) {
  __shared__ float scratch[kThreads];
  __shared__ int positions[kThreads];
  const std::size_t row = blockIdx.x;
  const float* h = projected + row * dim;
  float squares = 0.0F;
  for (std::size_t d = threadIdx.x; d < dim; d += blockDim.x)
    squares += h[d] * h[d];
  const float norm = BlockSum(squares, scratch);
  float best = -FLT_MAX;
  int best_index = 0;
  const float* row_dots = dots + row * entries;
  for (std::size_t e = threadIdx.x; e < entries; e += blockDim.x) {
    const float distance = -((norm - 2.0F * row_dots[e]) + norms[e]);
    if (distance > best) {
      best = distance;
      best_index = static_cast<int>(e);
    }
  }
  scratch[threadIdx.x] = best;
  positions[threadIdx.x] = best_index;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const float other = scratch[threadIdx.x + stride];
      const int other_index = positions[threadIdx.x + stride];
      if (other > scratch[threadIdx.x] ||
          (other == scratch[threadIdx.x] && other_index < positions[threadIdx.x])) {
        scratch[threadIdx.x] = other;
        positions[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    indices[row] = positions[0];
}

__global__ void SubtractEntriesKernel(float* residual, const float* table,
                                      const std::int32_t* indices,
                                      std::size_t width) {
  float* row = residual + blockIdx.x * width;
  const float* entry = table + static_cast<std::size_t>(indices[blockIdx.x]) * width;
  for (std::size_t j = threadIdx.x; j < width; j += blockDim.x)
    row[j] -= entry[j];
}

__global__ void SquaredNormsKernel(const float* entries, float* norms,
                                   std::size_t dim) {
  __shared__ float scratch[kThreads];
  const float* entry = entries + blockIdx.x * dim;
  float squares = 0.0F;
  for (std::size_t d = threadIdx.x; d < dim; d += blockDim.x)
    squares += entry[d] * entry[d];
  const float total = BlockSum(squares, scratch);
  if (threadIdx.x == 0)
    norms[blockIdx.x] = total;
}

unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

}  // namespace

void Im2Col(const float* input, std::size_t channels, std::size_t length,
            std::size_t kernel, std::size_t stride, std::size_t dilation,
            std::size_t pad, std::size_t output_length, float* columns,
            hipStream_t stream) {
  if (channels == 0 || output_length == 0)
    return;
  const dim3 grid(Blocks(output_length), static_cast<unsigned>(channels * kernel));
  Im2ColKernel<<<grid, kThreads, 0, stream>>>(input, length, kernel, stride,
                                             dilation, pad, output_length,
                                             columns);
}

void GroupNormGelu(float* data, const float* weight, const float* bias,
                   std::size_t channels, std::size_t length, float epsilon,
                   hipStream_t stream) {
  if (channels == 0)
    return;
  GroupNormGeluKernel<<<static_cast<unsigned>(channels), kThreads, 0, stream>>>(
      data, weight, bias, length, epsilon);
}

void Gelu(float* data, std::size_t count, hipStream_t stream) {
  if (count == 0)
    return;
  GeluKernel<<<Blocks(count), kThreads, 0, stream>>>(data, count);
}

void Elu(const float* input, float* output, std::size_t count,
         hipStream_t stream) {
  if (count == 0)
    return;
  EluKernel<<<Blocks(count), kThreads, 0, stream>>>(input, output, count);
}

void Transpose(const float* input, std::size_t rows, std::size_t columns,
               float* output, std::size_t output_stride, hipStream_t stream) {
  if (rows == 0 || columns == 0)
    return;
  const dim3 grid(static_cast<unsigned>((columns + 31) / 32),
                  static_cast<unsigned>((rows + 31) / 32));
  TransposeKernel<<<grid, dim3(32, 8), 0, stream>>>(input, rows, columns, output,
                                                   output_stride);
}

void LayerNorm(const float* input, const float* weight, const float* bias,
               float* output, std::size_t rows, std::size_t width,
               float epsilon, hipStream_t stream) {
  if (rows == 0)
    return;
  LayerNormKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
      input, weight, bias, output, width, epsilon);
}

void SoftmaxInPlace(float* scores, std::size_t rows, std::size_t columns,
                    float scale, hipStream_t stream) {
  if (rows == 0)
    return;
  SoftmaxInPlaceKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
      scores, columns, scale);
}

void Accumulate(const float* source, float* target, std::size_t count,
                hipStream_t stream) {
  if (count == 0)
    return;
  AccumulateKernel<<<Blocks(count), kThreads, 0, stream>>>(source, target, count);
}

void Scale(float* data, std::size_t count, float scale, hipStream_t stream) {
  if (count == 0)
    return;
  ScaleKernel<<<Blocks(count), kThreads, 0, stream>>>(data, count, scale);
}

void NearestEntry(const float* dots, const float* projected,
                  const float* norms, std::int32_t* indices, std::size_t rows,
                  std::size_t entries, std::size_t dim, hipStream_t stream) {
  if (rows == 0)
    return;
  NearestEntryKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
      dots, projected, norms, indices, entries, dim);
}

void SubtractEntries(float* residual, const float* table,
                     const std::int32_t* indices, std::size_t rows,
                     std::size_t width, hipStream_t stream) {
  if (rows == 0)
    return;
  SubtractEntriesKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
      residual, table, indices, width);
}

void SquaredNorms(const float* entries, float* norms, std::size_t count,
                  std::size_t dim, hipStream_t stream) {
  if (count == 0)
    return;
  SquaredNormsKernel<<<static_cast<unsigned>(count), kThreads, 0, stream>>>(
      entries, norms, dim);
}

}  // namespace gufo::models::omnivoice::kernels
