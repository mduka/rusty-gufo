#include <hip/hip_bfloat16.h>

#include <cmath>

#include "src/models/qwen/vision/kernels.hpp"

namespace gufo::models::qwen::vision {
namespace {
using Bf16 = hip_bfloat16;
constexpr unsigned kThreads = 256;

__device__ __forceinline__ float Round(float x) {
  return static_cast<float>(Bf16(x));
}

__global__ void Patchify(const std::uint8_t* pixels, Bf16* patches,
                         unsigned width, unsigned grid_width, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kPatchDim)
    return;
  const unsigned patch = i / kPatchDim;
  const unsigned element = i % kPatchDim;
  const unsigned merge = patch / 4;
  const unsigned y = (merge / (grid_width / 2)) * 2 + (patch % 4) / 2;
  const unsigned x = (merge % (grid_width / 2)) * 2 + patch % 2;
  const unsigned channel = element / (2 * 16 * 16);
  const unsigned spatial = element % (16 * 16);
  const unsigned pixel =
      ((y * 16 + spatial / 16) * width + x * 16 + spatial % 16) * 3;
  // Match rescale_and_normalize: multiply by 1/255 then subtract/divide.
  patches[i] = Bf16(
      (static_cast<float>(pixels[pixel + channel]) * (1.0F / 255.0F) - 0.5F) /
      0.5F);
}

__global__ void PatchPosition(float* hidden, const float* bias,
                              const float* positions, unsigned grid_height,
                              unsigned grid_width, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kHidden)
    return;
  const unsigned patch = i / kHidden;
  const unsigned dim = i % kHidden;
  const unsigned merge = patch / 4;
  const unsigned y = (merge / (grid_width / 2)) * 2 + (patch % 4) / 2;
  const unsigned x = (merge % (grid_width / 2)) * 2 + patch % 2;
  const float fy = static_cast<float>(y) * (47.0F / (grid_height - 1));
  const float fx = static_cast<float>(x) * (47.0F / (grid_width - 1));
  const unsigned y0 = min(static_cast<unsigned>(fy), 47U);
  const unsigned x0 = min(static_cast<unsigned>(fx), 47U);
  const unsigned y1 = min(y0 + 1, 47U);
  const unsigned x1 = min(x0 + 1, 47U);
  const float wy = fy - y0;
  const float wx = fx - x0;
  // The reference sums four interpolation products in this order.
  float position =
      positions[(y0 * 48 + x0) * kHidden + dim] * ((1 - wy) * (1 - wx));
  position += positions[(y0 * 48 + x1) * kHidden + dim] * ((1 - wy) * wx);
  position += positions[(y1 * 48 + x0) * kHidden + dim] * (wy * (1 - wx));
  position += positions[(y1 * 48 + x1) * kHidden + dim] * (wy * wx);
  hidden[i] = Round(Round(Round(hidden[i]) + bias[dim]) + Round(position));
}

__global__ void LayerNorm(const float* input, Bf16* output, const float* weight,
                          const float* bias, unsigned width) {
  const unsigned row = blockIdx.x;
  __shared__ float sum[kThreads];
  float value = 0;
  for (unsigned j = threadIdx.x; j < width; j += kThreads)
    value += input[row * width + j];
  sum[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      sum[threadIdx.x] += sum[threadIdx.x + stride];
    __syncthreads();
  }
  const float mean = sum[0] / width;
  __syncthreads();
  value = 0;
  for (unsigned j = threadIdx.x; j < width; j += kThreads) {
    const float centered = input[row * width + j] - mean;
    value += centered * centered;
  }
  sum[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      sum[threadIdx.x] += sum[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(sum[0] / width + 1e-6F);
  for (unsigned j = threadIdx.x; j < width; j += kThreads) {
    output[row * width + j] =
        Bf16((input[row * width + j] - mean) * scale * weight[j] + bias[j]);
  }
}

__global__ void BiasResidual(float* hidden, const float* projection,
                             const float* bias, unsigned size, unsigned width) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    hidden[i] = Round(hidden[i] + Round(projection[i] + bias[i % width]));
}

__global__ void Activate(const float* input, const float* bias, Bf16* output,
                         unsigned size, unsigned width, bool approximate) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;
  const float x = Round(input[i] + bias[i % width]);
  const float gelu = approximate
                         ? 0.5F * x *
                               (1.0F + tanhf(0.7978845608028654F *
                                             (x + 0.044715F * x * x * x)))
                         : 0.5F * x * (1.0F + erff(x * 0.7071067811865475F));
  output[i] = Bf16(gelu);
}

__global__ void QkvRope(const float* input, const float* bias, Bf16* q, Bf16* k,
                        Bf16* v, unsigned count, unsigned grid_width) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kHidden)
    return;
  const unsigned token = i / kHidden;
  const unsigned head = (i % kHidden) / kHeadDim;
  const unsigned d = i % kHeadDim;
  const unsigned pair = d % (kHeadDim / 2);
  const unsigned merge = token / 4;
  const unsigned y = (merge / (grid_width / 2)) * 2 + (token % 4) / 2;
  const unsigned x = (merge % (grid_width / 2)) * 2 + token % 2;
  const unsigned position = pair < kHeadDim / 4 ? y : x;
  const float angle =
      position / powf(10000.0F, static_cast<float>(pair % (kHeadDim / 4)) /
                                    (kHeadDim / 4));
  const float cosine = cosf(angle);
  const float sine = sinf(angle);
  const unsigned first = head * kHeadDim + pair;
  const unsigned base = token * 3 * kHidden;
  const float q0 = Round(input[base + first] + bias[first]);
  const float q1 =
      Round(input[base + first + kHeadDim / 2] + bias[first + kHeadDim / 2]);
  const float k0 = Round(input[base + kHidden + first] + bias[kHidden + first]);
  const float k1 = Round(input[base + kHidden + first + kHeadDim / 2] +
                         bias[kHidden + first + kHeadDim / 2]);
  const unsigned target = (head * count + token) * kHeadDim + d;
  q[target] = Bf16(d < kHeadDim / 2 ? q0 * cosine - q1 * sine
                                    : q1 * cosine + q0 * sine);
  k[target] = Bf16(d < kHeadDim / 2 ? k0 * cosine - k1 * sine
                                    : k1 * cosine + k0 * sine);
  v[target] = Bf16(input[base + 2 * kHidden + head * kHeadDim + d] +
                   bias[2 * kHidden + head * kHeadDim + d]);
}

// Keep bounded rows in registers through normalization. Larger rows retain
// the streaming path; caching them would spill and reread private memory.
template<unsigned kValues>
__global__ void Softmax(const float* scores, Bf16* probabilities,
                        unsigned length) {
  const unsigned row = blockIdx.x;
  __shared__ float reduce[kThreads];
  float values[kValues != 0 ? kValues : 1];
  float maximum = -INFINITY;
  if constexpr (kValues != 0) {
#pragma unroll
    for (unsigned j = 0; j < kValues; ++j) {
      const unsigned i = threadIdx.x + j * kThreads;
      values[j] =
          i < length
              ? Round(Round(scores[row * length + i]) * 0.1178511301977579F)
              : -INFINITY;
      maximum = fmaxf(maximum, values[j]);
    }
  } else {
    for (unsigned i = threadIdx.x; i < length; i += kThreads) {
      maximum = fmaxf(maximum, Round(Round(scores[row * length + i]) *
                                     0.1178511301977579F));
    }
  }
  reduce[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      reduce[threadIdx.x] =
          fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
    __syncthreads();
  }
  maximum = reduce[0];
  __syncthreads();
  float sum = 0;
  if constexpr (kValues != 0) {
#pragma unroll
    for (unsigned j = 0; j < kValues; ++j) {
      if (threadIdx.x + j * kThreads < length) {
        values[j] = expf(values[j] - maximum);
        sum += values[j];
        // Limit overlapping exponential temporaries without changing their
        // order. Wider shapes otherwise spill registers on gfx1151.
        if constexpr (kValues > 16)
          __builtin_amdgcn_sched_barrier(0);
      }
    }
  } else {
    for (unsigned i = threadIdx.x; i < length; i += kThreads)
      sum += expf(Round(Round(scores[row * length + i]) * 0.1178511301977579F) -
                  maximum);
  }
  reduce[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    __syncthreads();
  }
  const float inverse = 1.0F / reduce[0];
  if constexpr (kValues != 0) {
#pragma unroll
    for (unsigned j = 0; j < kValues; ++j) {
      const unsigned i = threadIdx.x + j * kThreads;
      if (i < length)
        probabilities[row * length + i] = Bf16(values[j] * inverse);
    }
  } else {
    for (unsigned i = threadIdx.x; i < length; i += kThreads) {
      probabilities[row * length + i] = Bf16(
          expf(Round(Round(scores[row * length + i]) * 0.1178511301977579F) -
               maximum) *
          inverse);
    }
  }
}

__global__ void AttentionRows(const float* input, Bf16* output, unsigned begin,
                              unsigned rows) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * kHidden)
    return;
  const unsigned token = i / kHidden;
  const unsigned head = (i % kHidden) / kHeadDim;
  const unsigned d = i % kHeadDim;
  output[(begin + token) * kHidden + head * kHeadDim + d] =
      Bf16(input[(head * rows + token) * kHeadDim + d]);
}

__global__ void Finish(float* output, const float* bias, unsigned size,
                       unsigned width) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    output[i] = Round(output[i] + bias[i % width]);
}

__global__ void InjectRows(float* hidden, const float* embedding,
                           std::uint32_t rows, std::uint32_t width,
                           std::uint32_t hc) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * width * hc)
    return;
  const auto row = i / (width * hc);
  hidden[i] = embedding[row * width + i % width];
}

unsigned Blocks(unsigned size) {
  return (size + kThreads - 1) / kThreads;
}
}  // namespace

void LaunchPatchify(const std::uint8_t* pixels, void* patches, unsigned width,
                    unsigned grid_width, unsigned count, hipStream_t stream) {
  Patchify<<<Blocks(count * kPatchDim), kThreads, 0, stream>>>(
      pixels, static_cast<Bf16*>(patches), width, grid_width, count);
}

void LaunchPatchPosition(float* hidden, const float* bias,
                         const float* positions, unsigned grid_height,
                         unsigned grid_width, unsigned count,
                         hipStream_t stream) {
  PatchPosition<<<Blocks(count * kHidden), kThreads, 0, stream>>>(
      hidden, bias, positions, grid_height, grid_width, count);
}

void LaunchLayerNorm(const float* input, void* output, const float* weight,
                     const float* bias, unsigned rows, unsigned width,
                     hipStream_t stream) {
  LayerNorm<<<rows, kThreads, 0, stream>>>(input, static_cast<Bf16*>(output),
                                           weight, bias, width);
}

void LaunchBiasResidual(float* hidden, const float* projection,
                        const float* bias, unsigned size, unsigned width,
                        hipStream_t stream) {
  BiasResidual<<<Blocks(size), kThreads, 0, stream>>>(hidden, projection, bias,
                                                      size, width);
}

void LaunchActivate(const float* input, const float* bias, void* output,
                    unsigned size, unsigned width, bool approximate,
                    hipStream_t stream) {
  Activate<<<Blocks(size), kThreads, 0, stream>>>(
      input, bias, static_cast<Bf16*>(output), size, width, approximate);
}

void LaunchQkvRope(const float* input, const float* bias, void* q, void* k,
                   void* v, unsigned count, unsigned grid_width,
                   hipStream_t stream) {
  QkvRope<<<Blocks(count * kHidden), kThreads, 0, stream>>>(
      input, bias, static_cast<Bf16*>(q), static_cast<Bf16*>(k),
      static_cast<Bf16*>(v), count, grid_width);
}

void LaunchSoftmax(const float* scores, void* probabilities, unsigned rows,
                   unsigned length, hipStream_t stream) {
  auto* out = static_cast<Bf16*>(probabilities);
  if (length <= 1024)
    Softmax<4><<<rows, kThreads, 0, stream>>>(scores, out, length);
  else if (length <= 2048)
    Softmax<8><<<rows, kThreads, 0, stream>>>(scores, out, length);
  else if (length <= 4096)
    Softmax<16><<<rows, kThreads, 0, stream>>>(scores, out, length);
  else if (length <= 8192)
    Softmax<32><<<rows, kThreads, 0, stream>>>(scores, out, length);
  else
    Softmax<0><<<rows, kThreads, 0, stream>>>(scores, out, length);
}

void LaunchAttentionRows(const float* input, void* output, unsigned begin,
                         unsigned rows, hipStream_t stream) {
  AttentionRows<<<Blocks(rows * kHidden), kThreads, 0, stream>>>(
      input, static_cast<Bf16*>(output), begin, rows);
}

void LaunchFinish(float* output, const float* bias, unsigned size,
                  unsigned width, hipStream_t stream) {
  Finish<<<Blocks(size), kThreads, 0, stream>>>(output, bias, size, width);
}

void LaunchInjectRows(float* hidden, const float* embedding, std::uint32_t rows,
                      std::uint32_t width, std::uint32_t hc,
                      hipStream_t stream) {
  InjectRows<<<(rows * width * hc + 255) / 256, 256, 0, stream>>>(
      hidden, embedding, rows, width, hc);
}

}  // namespace gufo::models::qwen::vision
