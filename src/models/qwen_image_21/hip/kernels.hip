#include <hip/hip_bfloat16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen_image_21/hip/bf16.hpp"
#include "src/models/qwen_image_21/hip/ops.hpp"

namespace gufo::models::qwen_image_21::hip {
namespace {

using B = hip_bfloat16;

__device__ float Round(float x) {
  return static_cast<float>(Bf16(x));
}

template<bool Maximum>
__device__ float Reduce(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  if (tid < 128)
    shared[tid] = Maximum ? fmaxf(shared[tid], shared[tid + 128])
                          : shared[tid] + shared[tid + 128];
  __syncthreads();
  if (tid < 64)
    shared[tid] = Maximum ? fmaxf(shared[tid], shared[tid + 64])
                          : shared[tid] + shared[tid + 64];
  __syncthreads();
  value = tid < 32 ? (Maximum ? fmaxf(shared[tid], shared[tid + 32])
                              : shared[tid] + shared[tid + 32])
                   : 0;
  // Finish exactly the same descending tree within the first 32 lanes.
#pragma unroll
  for (int stride = 16; stride; stride /= 2) {
    const float peer = __shfl_down(value, stride, 32);
    if (tid < stride)
      value = Maximum ? fmaxf(value, peer) : value + peer;
  }
  if (tid == 0)
    shared[0] = value;
  __syncthreads();
  return shared[0];
}

__device__ float Sum(float value, float* shared) {
  return Reduce<false>(value, shared);
}

__global__ void FloatToBf16(const float* x, B* y, std::size_t n) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = Bf16(x[i]);
}

__global__ void Bf16ToFloat(const B* x, float* y, std::size_t n) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = float(x[i]);
}

template<bool Convolution>
__global__ void Bias(const float* x, const B* bias, B* y, int cols,
                     std::size_t n) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    // PyTorch/MIOpen adds convolution bias after rounding the BF16 product.
    // Linear/addmm instead includes bias before the output conversion.
    const float product = Convolution ? Round(x[i]) : x[i];
    y[i] = Bf16(product + float(bias[i % cols]));
  }
}

template<bool Silu>
__global__ void NormKernel(const B* x, B* y, const B* w, const B* bias,
                           int cols, int kind, float eps) {
  __shared__ float scratch[256];
  const auto offset = static_cast<std::size_t>(blockIdx.x) * cols;
  float mean = 0;
  if (kind == static_cast<int>(Norm::kLayer)) {
    float value = 0;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
      value += float(x[offset + c]);
    mean = Sum(value, scratch) / cols;
    __syncthreads();
  }
  float square = 0;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = float(x[offset + c]) - mean;
    square += v * v;
  }
  float sum = Sum(square, scratch);
  const float inv = kind == static_cast<int>(Norm::kVae)
                        ? 1.0F / fmaxf(sqrtf(sum), 1e-12F)
                        : rsqrtf(sum / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = (float(x[offset + c]) - mean) * inv;
    if (kind == static_cast<int>(Norm::kRms)) {
      v = Round(v);
      if (w)
        v *= float(w[c]);
    } else if (kind == static_cast<int>(Norm::kZeroRms)) {
      v *= 1.0F + float(w[c]);
    } else if (kind == static_cast<int>(Norm::kVae)) {
      v = Round(Round(v) * sqrtf(static_cast<float>(cols)));
      if (w)
        v = Round(v * float(w[c]));
    } else {
      if (w)
        v *= float(w[c]);
      if (bias)
        v += float(bias[c]);
    }
    if constexpr (Silu) {
      v = float(SiluBf16(Bf16(v)));
    }
    y[offset + c] = Bf16(v);
  }
}

// One wave owns each short row. Fold the eight original 32-lane
// partials in the same 256-thread reduction order before the wave reduction.
__device__ float WaveTree(float* p) {
#pragma unroll
  for (int j = 0; j < 4; j++)
    p[j] += p[j + 4];
#pragma unroll
  for (int j = 0; j < 2; j++)
    p[j] += p[j + 2];
  float sum = p[0] + p[1];
#pragma unroll
  for (int d = 16; d; d /= 2) {
    float v = __shfl_down(sum, d, 32);
    if (int(threadIdx.x % 32) < d)
      sum += v;
  }
  return __shfl(sum, 0, 32);
}
template<bool Silu>
__global__ void WaveNorm(const B* x, B* y, const B* w, const B* bias, int rows,
                         int cols, int kind, float eps) {
  int lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32;
  if (row >= rows)
    return;
  size_t offset = size_t(row) * cols;
  float mean = 0;
  if (kind == static_cast<int>(Norm::kLayer)) {
    float partial[8]{};
#pragma unroll
    for (int j = 0; j < 8; j++)
      for (int c = lane + j * 32; c < cols; c += 256)
        partial[j] += float(x[offset + c]);
    mean = WaveTree(partial) / cols;
  }
  float partial[8]{};
#pragma unroll
  for (int j = 0; j < 8; j++)
    for (int c = lane + j * 32; c < cols; c += 256) {
      float v = float(x[offset + c]) - mean;
      partial[j] += v * v;
    }
  float sum = WaveTree(partial);
  float inv = kind == static_cast<int>(Norm::kVae)
                  ? 1.0F / fmaxf(sqrtf(sum), 1e-12F)
                  : rsqrtf(sum / cols + eps);
  for (int c = lane; c < cols; c += 32) {
    float v = (float(x[offset + c]) - mean) * inv;
    if (kind == static_cast<int>(Norm::kRms)) {
      v = Round(v);
      if (w)
        v *= float(w[c]);
    } else if (kind == static_cast<int>(Norm::kZeroRms))
      v *= 1.0F + float(w[c]);
    else if (kind == static_cast<int>(Norm::kVae)) {
      v = Round(Round(v) * sqrtf(float(cols)));
      if (w)
        v = Round(v * float(w[c]));
    } else {
      if (w)
        v *= float(w[c]);
      if (bias)
        v += float(bias[c]);
    }
    if constexpr (Silu) {
      v = float(SiluBf16(Bf16(v)));
    }
    y[offset + c] = Bf16(v);
  }
}

// Preserve the RMS reduction and both BF16 normalization boundaries, then
// rotate adjacent channels while the normalized values remain in registers.
__global__ void RmsRotary128(const B* x, B* y, const B* weight,
                             const float* frequencies, int rows, int heads,
                             float epsilon) {
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * 8 + threadIdx.x / 32;
  if (row >= rows * heads)
    return;
  const auto offset = std::size_t(row) * 128;
  float values[4], partial[8]{};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    values[j] = float(x[offset + lane + j * 32]);
    partial[j] = values[j] * values[j];
  }
  const float inverse = rsqrtf(WaveTree(partial) / 128 + epsilon);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int channel = lane + j * 32;
    const float value =
        Round(Round(values[j] * inverse) * float(weight[channel]));
    const float other = __shfl_xor(value, 1, 32);
    const int pair = (row / heads) * 128 + (channel & ~1);
    const float cosine = frequencies[pair], sine = frequencies[pair + 1];
    y[offset + channel] = Bf16(lane % 2 ? value * cosine + other * sine
                                        : value * cosine - other * sine);
  }
}

__global__ void Elementwise(const B* x, const B* other, B* y, std::size_t n,
                            int op, float scalar) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const float a = float(x[i]);
  float value = a;
  if (op == 0)
    value = float(SiluBf16(x[i]));
  if (op == 1)
    value = 0.5F * a *
            (1.0F + tanhf(0.7978845608028654F * (a + 0.044715F * a * a * a)));
  if (op == 2)
    value = 0.5F * a * (1.0F + erff(a * 0.7071067811865475F));
  if (op == 4)
    value = float(SiluBf16(x[i])) * float(other[i]);
  if (op == 5)
    value = a + float(other[i]);
  // FlowMatchEuler keeps dt * model_output in BF16 (a zero-dimensional
  // FP32 dt does not promote the BF16 tensor), then adds to the FP32 sample.
  if (op == 6)
    value = a + Round(scalar * float(other[i]));
  y[i] = Bf16(value);
}

__global__ void PrepareModulation(const B* x, B* y, int count, bool gates) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const float value = float(x[i]);
    y[i] = Bf16(gates && ((i / 4096) % 2) ? tanhf(value) : 1.0F + value);
  }
}

// LayerNorm's original sum tree and BF16 boundary, followed by modulation.
// Cache each lane's input so neither normalization nor the fused scale needs
// another full activation read.
__global__ void NormModulation(const B* x, const B* mod, B* y, int mod_cols,
                               int part, int prefix) {
  constexpr int cols = 4096;
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  const auto offset = std::size_t(row) * cols;
  __shared__ float scratch[256];
  float values[16], sum = 0;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    values[i] = float(x[offset + lane + i * 256]);
    sum += values[i];
  }
  const float mean = Sum(sum, scratch) / cols;
  __syncthreads();
  float square = 0;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    values[i] -= mean;
    square += values[i] * values[i];
  }
  const float inv = rsqrtf(Sum(square, scratch) / cols + 1e-6F);
  const int mod_offset = (row < prefix ? mod_cols : 0) + part * cols;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int c = lane + i * 256;
    y[offset + c] = Bf16(Round(values[i] * inv) * float(mod[mod_offset + c]));
  }
}

__global__ void Modulation(const B* x, const B* mod, const B* residual, B* y,
                           int rows, int cols, int mod_cols, int part,
                           int prefix) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * cols)
    return;
  const int row = i / cols, c = i % cols;
  const float m = float(mod[(row < prefix ? mod_cols : 0) + part * cols + c]);
  float value = float(x[i]) * m;
  if (residual)
    value = float(residual[i]) + Round(value);
  y[i] = Bf16(value);
}

__global__ void Rotary(const B* x, B* y, const float* frequencies, int rows,
                       int heads, int dim, int mode) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * heads * dim / 2)
    return;
  const int half = dim / 2;
  const int pair = i % half;
  const int row = i / (half * heads);
  const auto base = (i / half) * dim;
  const auto left = base + (mode == 0 ? pair * 2 : pair);
  const auto right = mode == 0 ? left + 1 : left + half;
  float c = frequencies[(static_cast<std::size_t>(row) * half + pair) * 2];
  float s = frequencies[(static_cast<std::size_t>(row) * half + pair) * 2 + 1];
  const float a = float(x[left]), b = float(x[right]);
  if (mode == 2) {
    c = Round(c);
    s = Round(s);
    y[left] = Bf16(Round(a * c) - Round(b * s));
    y[right] = Bf16(Round(b * c) + Round(a * s));
  } else {
    y[left] = Bf16(a * c - b * s);
    y[right] = Bf16(b * c + a * s);
  }
}

__global__ void HeadLayout(const B* x, B* y, int rows, int heads, int kv_heads,
                           int dim, bool inverse) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * heads * dim)
    return;
  const int d = i % dim, row = (i / dim) % rows, head = i / (dim * rows);
  if (inverse)
    y[(static_cast<std::size_t>(row) * heads + head) * dim + d] = x[i];
  else
    y[i] = x[(static_cast<std::size_t>(row) * kv_heads +
              head / (heads / kv_heads)) *
                 dim +
             d];
}

template<bool Inverse>
__global__ void HeadLayout128(const B* x, B* y, int rows, int heads,
                              int kv_heads) {
  __shared__ __align__(16) B tile[8 * 8 * 128];
  const int row_start = blockIdx.y * 8, head_start = blockIdx.x * 8;
#pragma unroll
  for (int i = threadIdx.x * 8; i < 8 * 8 * 128; i += 256 * 8) {
    const int major = i / (8 * 128), minor = i / 128 % 8, d = i % 128;
    const int row = row_start + (Inverse ? minor : major);
    const int head = head_start + (Inverse ? major : minor);
    uint4 value{};
    if (row < rows && head < heads) {
      const auto index =
          Inverse ? (static_cast<std::size_t>(head) * rows + row) * 128 + d
                  : (static_cast<std::size_t>(row) * kv_heads +
                     head / (heads / kv_heads)) *
                            128 +
                        d;
      value = *reinterpret_cast<const uint4*>(x + index);
    }
    *reinterpret_cast<uint4*>(tile + i) = value;
  }
  __syncthreads();
#pragma unroll
  for (int i = threadIdx.x * 8; i < 8 * 8 * 128; i += 256 * 8) {
    const int major = i / (8 * 128), minor = i / 128 % 8, d = i % 128;
    const int row = row_start + (Inverse ? major : minor);
    const int head = head_start + (Inverse ? minor : major);
    if (row < rows && head < heads) {
      const auto value =
          *reinterpret_cast<const uint4*>(tile + (minor * 8 + major) * 128 + d);
      const auto index =
          Inverse ? (static_cast<std::size_t>(row) * heads + head) * 128 + d
                  : (static_cast<std::size_t>(head) * rows + row) * 128 + d;
      *reinterpret_cast<uint4*>(y + index) = value;
    }
  }
}

__global__ void Softmax(const float* scores, float* probabilities, int kv_rows,
                        int query_count, int query_start, const int* limits,
                        float scale) {
  __shared__ float scratch[256];
  const auto row = blockIdx.x;
  const auto offset = static_cast<std::size_t>(row) * kv_rows;
  const int limit = limits ? limits[query_start + row % query_count] : kv_rows;
  float maximum = -INFINITY;
  for (int i = threadIdx.x; i < limit; i += blockDim.x)
    maximum = fmaxf(maximum, scores[offset + i] * scale);
  maximum = Reduce<true>(maximum, scratch);
  __syncthreads();
  float sum = 0;
  for (int i = threadIdx.x; i < limit; i += blockDim.x)
    sum += expf(scores[offset + i] * scale - maximum);
  sum = Sum(sum, scratch);
  for (int i = threadIdx.x; i < kv_rows; i += blockDim.x)
    probabilities[offset + i] =
        i < limit ? expf(scores[offset + i] * scale - maximum) / sum : 0.0F;
}

// Reuse each scaled score and exponential for ordinary image contexts. Keep
// the same per-thread additions and 256-thread reduction as the long-row path.
template<int Capacity>
__global__ void SoftmaxCached(const float* scores, float* probabilities,
                              int kv_rows, int query_count, int query_start,
                              const int* limits, float scale) {
  __shared__ float scratch[256];
  float values[Capacity];
  const auto offset = static_cast<std::size_t>(blockIdx.x) * kv_rows;
  const int limit =
      limits ? limits[query_start + blockIdx.x % query_count] : kv_rows;
  float maximum = -INFINITY;
#pragma unroll
  for (int j = 0; j < Capacity; ++j) {
    const int index = threadIdx.x + j * 256;
    values[j] = index < limit ? scores[offset + index] * scale : -INFINITY;
    maximum = fmaxf(maximum, values[j]);
  }
  maximum = Reduce<true>(maximum, scratch);
  __syncthreads();
  float sum = 0;
#pragma unroll
  for (int j = 0; j < Capacity; ++j) {
    const int index = threadIdx.x + j * 256;
    values[j] = index < limit ? expf(values[j] - maximum) : 0.0F;
    if (index < limit)
      sum += values[j];
  }
  sum = Sum(sum, scratch);
#pragma unroll
  for (int j = 0; j < Capacity; ++j) {
    const int index = threadIdx.x + j * 256;
    if (index < kv_rows)
      probabilities[offset + index] = index < limit ? values[j] / sum : 0.0F;
  }
}

__global__ void CopyColumns(const B* x, B* y, int rows, int in_cols, int start,
                            int out_cols) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < static_cast<std::size_t>(rows) * out_cols)
    y[i] = x[(i / out_cols) * in_cols + start + i % out_cols];
}

__global__ void Embedding(const B* w, B* y, const std::uint32_t* ids, int rows,
                          int cols) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < static_cast<std::size_t>(rows) * cols)
    y[i] = w[static_cast<std::size_t>(ids[i / cols]) * cols + i % cols];
}

__global__ void Scatter(const B* values, B* y, const int* positions, int count,
                        int cols, bool add) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(count) * cols)
    return;
  const auto dst =
      static_cast<std::size_t>(positions[i / cols]) * cols + i % cols;
  y[dst] = add ? Bf16(float(y[dst]) + float(values[i])) : values[i];
}

__global__ void Im2col(const B* x, B* y, int height, int width, int channels,
                       int kernel, int stride, int padding, int upscale,
                       int out_width, int start, int count) {
  const int k = channels * kernel * kernel;
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(count) * k)
    return;
  const int pixel = start + i / k;
  const int kc = i % k;
  const int c = kc / (kernel * kernel);
  const int sy =
      (pixel / out_width) * stride - padding + (kc / kernel) % kernel;
  const int sx = (pixel % out_width) * stride - padding + kc % kernel;
  y[i] =
      (sy >= 0 && sx >= 0 && sy < height * upscale && sx < width * upscale)
          ? x[(static_cast<std::size_t>(sy / upscale) * width + sx / upscale) *
                  channels +
              c]
          : Bf16(0.0F);
}

__global__ void Shortcut(const B* x, B* y, int height, int width,
                         int in_channels, int out_channels, int temporal,
                         int spatial, bool up) {
  const int out_height = up ? height * spatial : height / spatial;
  const int out_width = up ? width * spatial : width / spatial;
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(out_height) * out_width * out_channels)
    return;
  const int c = i % out_channels;
  const int py = (i / out_channels) / out_width;
  const int px = (i / out_channels) % out_width;
  if (up) {
    const int repeats =
        out_channels * temporal * spatial * spatial / in_channels;
    const int expanded =
        ((c * temporal + temporal - 1) * spatial + py % spatial) * spatial +
        px % spatial;
    const int ic = expanded / repeats;
    y[i] = x[(static_cast<std::size_t>(py / spatial) * width + px / spatial) *
                 in_channels +
             ic];
  } else {
    const int group = in_channels * temporal * spatial * spatial / out_channels;
    float sum = 0;
    for (int j = 0; j < group; ++j) {
      const int index = c * group + j;
      const int ic = index / (temporal * spatial * spatial);
      const int time = (index / (spatial * spatial)) % temporal;
      if (time == temporal - 1) {
        const int iy = py * spatial + (index / spatial) % spatial;
        const int ix = px * spatial + index % spatial;
        sum += float(
            x[(static_cast<std::size_t>(iy) * width + ix) * in_channels + ic]);
      }
    }
    y[i] = Bf16(sum / group);
  }
}

__global__ void ScaleLatents(const B* x, B* y, const B* mean, const B* stddev,
                             std::size_t n, bool encode) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const int c = i % 64;
  const float v = float(x[i]), m = float(mean[c]), s = float(stddev[c]);
  y[i] = Bf16(encode ? Round(v - m) / s : Round(v * s) + m);
}

__global__ void Position(const B* x, const B* table, B* y, int height,
                         int width) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(height) * width * 1152)
    return;
  const int token = i / 1152, c = i % 1152;
  const int py = (token / 4 / (width / 2)) * 2 + (token / 2) % 2;
  const int px = ((token / 4) % (width / 2)) * 2 + token % 2;
  const float fy = static_cast<float>(py) * 47.0F / (height - 1);
  const float fx = static_cast<float>(px) * 47.0F / (width - 1);
  const int y0 = static_cast<int>(fy), x0 = static_cast<int>(fx);
  const int y1 = min(y0 + 1, 47), x1 = min(x0 + 1, 47);
  const float dy = fy - y0, dx = fx - x0;
  // Transformers 5.17 interpolates the BF16 position table with FP32
  // coefficients and accumulates in FP32 before one final BF16 conversion.
  const float p0 =
      float(table[(y0 * 48 + x0) * 1152 + c]) * ((1 - dy) * (1 - dx));
  const float p1 = float(table[(y0 * 48 + x1) * 1152 + c]) * ((1 - dy) * dx);
  const float p2 = float(table[(y1 * 48 + x0) * 1152 + c]) * (dy * (1 - dx));
  const float p3 = float(table[(y1 * 48 + x1) * 1152 + c]) * (dy * dx);
  y[i] = Bf16(float(x[i]) + Round(p0 + p1 + p2 + p3));
}

unsigned Blocks(std::size_t n) {
  return static_cast<unsigned>((n + 255) / 256);
}

}  // namespace

void NormalizeRows(const std::uint16_t* input, std::uint16_t* output,
                   const std::uint16_t* weight, const std::uint16_t* bias,
                   int rows, int cols, Norm kind, float epsilon,
                   hipStream_t stream, bool silu) {
  const auto* x = reinterpret_cast<const B*>(input);
  auto* y = reinterpret_cast<B*>(output);
  const auto* w = reinterpret_cast<const B*>(weight);
  const auto* b = reinterpret_cast<const B*>(bias);
  if (cols <= 1152) {
    if (silu)
      WaveNorm<true><<<(rows + 7) / 8, 256, 0, stream>>>(
          x, y, w, b, rows, cols, static_cast<int>(kind), epsilon);
    else
      WaveNorm<false><<<(rows + 7) / 8, 256, 0, stream>>>(
          x, y, w, b, rows, cols, static_cast<int>(kind), epsilon);
  } else {
    if (silu)
      NormKernel<true><<<rows, 256, 0, stream>>>(
          x, y, w, b, cols, static_cast<int>(kind), epsilon);
    else
      NormKernel<false><<<rows, 256, 0, stream>>>(
          x, y, w, b, cols, static_cast<int>(kind), epsilon);
  }
}

void NormalizeRotary128(const std::uint16_t* input, std::uint16_t* output,
                        const std::uint16_t* weight, const float* frequencies,
                        int rows, int heads, float epsilon,
                        hipStream_t stream) {
  RmsRotary128<<<(rows * heads + 7) / 8, 256, 0, stream>>>(
      reinterpret_cast<const B*>(input), reinterpret_cast<B*>(output),
      reinterpret_cast<const B*>(weight), frequencies, rows, heads, epsilon);
}

void ReorderHeads(const std::uint16_t* input, std::uint16_t* output, int rows,
                  int heads, int kv_heads, int dim, bool inverse,
                  hipStream_t stream) {
  const auto* x = reinterpret_cast<const B*>(input);
  auto* y = reinterpret_cast<B*>(output);
  if (dim == 128 && rows >= 128) {
    const dim3 grid((heads + 7) / 8, (rows + 7) / 8);
    if (inverse)
      HeadLayout128<true>
          <<<grid, 256, 0, stream>>>(x, y, rows, heads, kv_heads);
    else
      HeadLayout128<false>
          <<<grid, 256, 0, stream>>>(x, y, rows, heads, kv_heads);
  } else {
    HeadLayout<<<Blocks(static_cast<std::size_t>(rows) * heads * dim), 256, 0,
                 stream>>>(x, y, rows, heads, kv_heads, dim, inverse);
  }
}

void LaunchFloatToBf16(const float* input, void* output, std::size_t count,
                       hipStream_t stream) {
  FloatToBf16<<<Blocks(count), 256, 0, stream>>>(input, static_cast<B*>(output),
                                                 count);
}

void LaunchBf16ToFloat(const void* input, float* output, std::size_t count,
                       hipStream_t stream) {
  Bf16ToFloat<<<Blocks(count), 256, 0, stream>>>(static_cast<const B*>(input),
                                                 output, count);
}

void LaunchAddBias(const float* input, const void* bias, void* output,
                   std::uint32_t columns, std::size_t count, bool convolution,
                   hipStream_t stream) {
  const auto* b = static_cast<const B*>(bias);
  auto* y = static_cast<B*>(output);
  if (convolution)
    Bias<true><<<Blocks(count), 256, 0, stream>>>(
        input, b, y, static_cast<int>(columns), count);
  else
    Bias<false><<<Blocks(count), 256, 0, stream>>>(
        input, b, y, static_cast<int>(columns), count);
}

void LaunchActivate(const void* input, void* output, std::size_t count,
                    std::uint32_t activation, hipStream_t stream) {
  Elementwise<<<Blocks(count), 256, 0, stream>>>(
      static_cast<const B*>(input), nullptr, static_cast<B*>(output), count,
      static_cast<int>(activation), 0);
}

void LaunchSwiGlu(const void* gate, const void* up, void* output,
                  std::size_t count, hipStream_t stream) {
  Elementwise<<<Blocks(count), 256, 0, stream>>>(
      static_cast<const B*>(gate), static_cast<const B*>(up),
      static_cast<B*>(output), count, 4, 0);
}

void LaunchAdd(const void* left, const void* right, void* output,
               std::size_t count, hipStream_t stream) {
  Elementwise<<<Blocks(count), 256, 0, stream>>>(
      static_cast<const B*>(left), static_cast<const B*>(right),
      static_cast<B*>(output), count, 5, 0);
}

void LaunchEuler(const void* sample, const void* velocity, void* output,
                 std::size_t count, float dt, hipStream_t stream) {
  Elementwise<<<Blocks(count), 256, 0, stream>>>(
      static_cast<const B*>(sample), static_cast<const B*>(velocity),
      static_cast<B*>(output), count, 6, dt);
}

void LaunchPrepareModulation(const void* input, void* output,
                             std::uint32_t count, bool gates,
                             hipStream_t stream) {
  PrepareModulation<<<Blocks(count), 256, 0, stream>>>(
      static_cast<const B*>(input), static_cast<B*>(output),
      static_cast<int>(count), gates);
}

void LaunchNormalizeModulate(const void* input, const void* modulation,
                             void* output, std::uint32_t rows,
                             std::uint32_t modulation_columns,
                             std::uint32_t part, std::uint32_t prefix,
                             hipStream_t stream) {
  NormModulation<<<rows, 256, 0, stream>>>(
      static_cast<const B*>(input), static_cast<const B*>(modulation),
      static_cast<B*>(output), static_cast<int>(modulation_columns),
      static_cast<int>(part), static_cast<int>(prefix));
}

void LaunchModulate(const void* input, const void* modulation,
                    const void* residual, void* output, std::uint32_t rows,
                    std::uint32_t columns, std::uint32_t modulation_columns,
                    std::uint32_t part, std::uint32_t prefix,
                    hipStream_t stream) {
  Modulation<<<Blocks(static_cast<std::size_t>(rows) * columns), 256, 0,
               stream>>>(
      static_cast<const B*>(input), static_cast<const B*>(modulation),
      static_cast<const B*>(residual), static_cast<B*>(output),
      static_cast<int>(rows), static_cast<int>(columns),
      static_cast<int>(modulation_columns), static_cast<int>(part),
      static_cast<int>(prefix));
}

void LaunchRotary(const void* input, void* output, const float* frequencies,
                  std::uint32_t rows, std::uint32_t heads, std::uint32_t dim,
                  std::uint32_t mode, hipStream_t stream) {
  Rotary<<<Blocks(static_cast<std::size_t>(rows) * heads * dim / 2), 256, 0,
           stream>>>(static_cast<const B*>(input), static_cast<B*>(output),
                     frequencies, static_cast<int>(rows),
                     static_cast<int>(heads), static_cast<int>(dim),
                     static_cast<int>(mode));
}

void LaunchSoftmax(const float* scores, float* probabilities,
                   std::uint32_t heads, std::uint32_t queries,
                   std::uint32_t keys, std::uint32_t query_start,
                   const std::int32_t* limits, float scale,
                   hipStream_t stream) {
  const unsigned blocks = heads * queries;
  const int kv_rows = static_cast<int>(keys);
  const int count = static_cast<int>(queries);
  const int start = static_cast<int>(query_start);
  if (keys >= 4096 && keys <= 4352)
    SoftmaxCached<17><<<blocks, 256, 0, stream>>>(
        scores, probabilities, kv_rows, count, start, limits, scale);
  else if (keys > 4352 && keys <= 8448)
    SoftmaxCached<33><<<blocks, 256, 0, stream>>>(
        scores, probabilities, kv_rows, count, start, limits, scale);
  else
    Softmax<<<blocks, 256, 0, stream>>>(scores, probabilities, kv_rows, count,
                                        start, limits, scale);
}

void LaunchCopyColumns(const void* input, void* output, std::uint32_t rows,
                       std::uint32_t input_columns, std::uint32_t start,
                       std::uint32_t columns, hipStream_t stream) {
  CopyColumns<<<Blocks(static_cast<std::size_t>(rows) * columns), 256, 0,
                stream>>>(static_cast<const B*>(input), static_cast<B*>(output),
                          static_cast<int>(rows),
                          static_cast<int>(input_columns),
                          static_cast<int>(start), static_cast<int>(columns));
}

void LaunchEmbedding(const void* table, void* output, const std::uint32_t* ids,
                     std::uint32_t count, std::uint32_t columns,
                     hipStream_t stream) {
  Embedding<<<Blocks(static_cast<std::size_t>(count) * columns), 256, 0,
              stream>>>(static_cast<const B*>(table), static_cast<B*>(output),
                        ids, static_cast<int>(count),
                        static_cast<int>(columns));
}

void LaunchScatterRows(const void* values, void* output,
                       const std::int32_t* positions, std::uint32_t count,
                       std::uint32_t columns, bool add, hipStream_t stream) {
  Scatter<<<Blocks(static_cast<std::size_t>(count) * columns), 256, 0,
            stream>>>(static_cast<const B*>(values), static_cast<B*>(output),
                      positions, static_cast<int>(count),
                      static_cast<int>(columns), add);
}

void LaunchIm2col(const void* input, void* output, std::uint32_t height,
                  std::uint32_t width, std::uint32_t channels,
                  std::uint32_t kernel, std::uint32_t stride,
                  std::uint32_t padding, std::uint32_t upscale,
                  std::uint32_t output_width, std::uint32_t start,
                  std::uint32_t count, hipStream_t stream) {
  const auto size =
      static_cast<std::size_t>(count) * channels * kernel * kernel;
  Im2col<<<Blocks(size), 256, 0, stream>>>(
      static_cast<const B*>(input), static_cast<B*>(output),
      static_cast<int>(height), static_cast<int>(width),
      static_cast<int>(channels), static_cast<int>(kernel),
      static_cast<int>(stride), static_cast<int>(padding),
      static_cast<int>(upscale), static_cast<int>(output_width),
      static_cast<int>(start), static_cast<int>(count));
}

void LaunchDownShortcut(const void* input, void* output, std::uint32_t height,
                        std::uint32_t width, std::uint32_t input_channels,
                        std::uint32_t output_channels, std::uint32_t temporal,
                        std::uint32_t spatial, hipStream_t stream) {
  const auto size = static_cast<std::size_t>(height / spatial) *
                    (width / spatial) * output_channels;
  Shortcut<<<Blocks(size), 256, 0, stream>>>(
      static_cast<const B*>(input), static_cast<B*>(output),
      static_cast<int>(height), static_cast<int>(width),
      static_cast<int>(input_channels), static_cast<int>(output_channels),
      static_cast<int>(temporal), static_cast<int>(spatial), false);
}

void LaunchUpShortcut(const void* input, void* output, std::uint32_t height,
                      std::uint32_t width, std::uint32_t input_channels,
                      std::uint32_t output_channels, std::uint32_t temporal,
                      hipStream_t stream) {
  const auto size =
      static_cast<std::size_t>(height) * width * 4 * output_channels;
  Shortcut<<<Blocks(size), 256, 0, stream>>>(
      static_cast<const B*>(input), static_cast<B*>(output),
      static_cast<int>(height), static_cast<int>(width),
      static_cast<int>(input_channels), static_cast<int>(output_channels),
      static_cast<int>(temporal), 2, true);
}

void LaunchScaleLatents(const void* input, void* output, const void* mean,
                        const void* deviation, std::size_t count, bool encode,
                        hipStream_t stream) {
  ScaleLatents<<<Blocks(count), 256, 0, stream>>>(
      static_cast<const B*>(input), static_cast<B*>(output),
      static_cast<const B*>(mean), static_cast<const B*>(deviation), count,
      encode);
}

void LaunchVisionPosition(const void* input, const void* table, void* output,
                          std::uint32_t height, std::uint32_t width,
                          hipStream_t stream) {
  Position<<<Blocks(static_cast<std::size_t>(height) * width * 1152), 256, 0,
             stream>>>(static_cast<const B*>(input),
                       static_cast<const B*>(table), static_cast<B*>(output),
                       static_cast<int>(height), static_cast<int>(width));
}

}  // namespace gufo::models::qwen_image_21::hip
