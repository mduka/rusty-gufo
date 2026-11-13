// Kernels of the generator port (generator.cpp, transformer.cpp): operators
// of OpenBMB/VoxCPM's MiniCPM4 stacks, local encoder, local DiT and CFM
// sampler (Apache-2.0, Copyright OpenBMB; see licenses/VoxCPM.txt), rewritten
// in HIP.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "generator_kernels.hpp"

namespace gufo::models::voxcpm::kernels {
namespace {

constexpr int kThreads = 256;

__device__ inline float FromBf16(Bf16 bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
}

__device__ inline Bf16 ToBf16Bits(float value) {
  const std::uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0)
    return static_cast<Bf16>((bits >> 16U) | 0x40U);  // quiet NaN
  const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<Bf16>((bits + rounding) >> 16U);
}

__device__ inline float WaveSum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    value += __shfl_xor(value, offset);
  return value;
}

__device__ inline float WaveMax(float value) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    value = fmaxf(value, __shfl_xor(value, offset));
  return value;
}

/// Block-wide sum or max; `scratch` holds one float per wave.
template<bool kMax>
__device__ float BlockReduce(float value, float* scratch) {
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves = (blockDim.x + warpSize - 1) / warpSize;
  value = kMax ? WaveMax(value) : WaveSum(value);
  __syncthreads();
  if (lane == 0)
    scratch[wave] = value;
  __syncthreads();
  float result = scratch[0];
  for (int index = 1; index < waves; ++index)
    result = kMax ? fmaxf(result, scratch[index]) : result + scratch[index];
  return result;
}

dim3 Blocks(std::size_t count, int threads = kThreads) {
  return dim3(static_cast<unsigned>((count + threads - 1) / threads));
}

__global__ void EmbedKernel(const Bf16* table, const std::int32_t* ids,
                            const std::int32_t* text_mask, float* output,
                            std::size_t width, float scale) {
  const std::size_t row = blockIdx.x;
  const bool text = text_mask[row] != 0;
  const Bf16* source = table + static_cast<std::size_t>(ids[row]) * width;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x)
    output[row * width + i] = text ? FromBf16(source[i]) * scale : 0.0F;
}

__global__ void ScatterRowsKernel(const float* input,
                                  const std::int32_t* indices, float* output,
                                  std::size_t width) {
  const std::size_t row = blockIdx.x;
  const std::size_t target = static_cast<std::size_t>(indices[row]);
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x)
    output[target * width + i] = input[row * width + i];
}

__global__ void GatherRowsKernel(const float* input,
                                 const std::int32_t* indices, float* output,
                                 std::size_t width) {
  const std::size_t row = blockIdx.x;
  const std::size_t source = static_cast<std::size_t>(indices[row]);
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x)
    output[row * width + i] = input[source * width + i];
}

__global__ void RmsNormKernel(const float* input, const Bf16* weight,
                              Bf16* output, float* output_f32,
                              std::size_t width, float epsilon) {
  __shared__ float scratch[kThreads / 32];
  const std::size_t row = blockIdx.x;
  const float* x = input + row * width;
  float sum = 0.0F;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x)
    sum += x[i] * x[i];
  sum = BlockReduce<false>(sum, scratch);
  const float inverse = rsqrtf(sum / static_cast<float>(width) + epsilon);
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    const float value = x[i] * inverse * FromBf16(weight[i]);
    output[row * width + i] = ToBf16Bits(value);
    if (output_f32 != nullptr)
      output_f32[row * width + i] = value;
  }
}

__global__ void QkvRopeKernel(const float* qkv, const float* cos_sin,
                              const std::int32_t* positions,
                              const std::int32_t* cache_rows, float* q,
                              float* k_cache, float* v_cache,
                              std::size_t heads, std::size_t kv_heads,
                              std::size_t head_dim, std::size_t capacity) {
  const std::size_t row = blockIdx.x;
  const std::size_t half = head_dim / 2;
  const std::size_t width = (heads + 2 * kv_heads) * head_dim;
  const float* source = qkv + row * width;
  const float* table =
      cos_sin == nullptr
          ? nullptr
          : cos_sin + static_cast<std::size_t>(positions[row]) * head_dim;
  const std::size_t cache_row = static_cast<std::size_t>(cache_rows[row]);
  // Pairs (i, i + half) of q and k heads, then v.
  const std::size_t pairs = (heads + kv_heads) * half;
  for (std::size_t index = threadIdx.x; index < pairs; index += blockDim.x) {
    const std::size_t head = index / half;
    const std::size_t i = index % half;
    const float a = source[head * head_dim + i];
    const float b = source[head * head_dim + half + i];
    float first = a;
    float second = b;
    if (table != nullptr) {
      const float c = table[i];
      const float s = table[half + i];
      first = a * c - b * s;
      second = b * c + a * s;
    }
    if (head < heads) {
      float* target = q + (row * heads + head) * head_dim;
      target[i] = first;
      target[half + i] = second;
    } else {
      float* target =
          k_cache + ((head - heads) * capacity + cache_row) * head_dim;
      target[i] = first;
      target[half + i] = second;
    }
  }
  const float* v = source + (heads + kv_heads) * head_dim;
  for (std::size_t index = threadIdx.x; index < kv_heads * head_dim;
       index += blockDim.x) {
    const std::size_t head = index / head_dim;
    v_cache[(head * capacity + cache_row) * head_dim + index % head_dim] =
        v[index];
  }
}

// Four waves per (row, head) with head_dim 128: each lane holds four
// dimensions of q, of its keys and of the output; wave w takes keys
// w, w + 4, ... with an online softmax (running maximum and sum), and the
// four partial results are merged at the end.
constexpr int kAttentionWaves = 4;

__global__ __launch_bounds__(kAttentionWaves * 32) void AttentionKernel(
    const float* q, const float* k_cache, const float* v_cache, Bf16* output,
    std::size_t rows_per_sequence, std::size_t keys_per_sequence,
    std::size_t past, bool causal, std::size_t heads, std::size_t kv_heads,
    std::size_t capacity, float scale) {
  constexpr int kDim = 128;
  __shared__ float merge_max[kAttentionWaves];
  __shared__ float merge_sum[kAttentionWaves];
  __shared__ float4 merge_out[kAttentionWaves][32];
  const std::size_t row = blockIdx.x;
  const std::size_t head = blockIdx.y;
  const std::size_t sequence = row / rows_per_sequence;
  const std::size_t local = row % rows_per_sequence;
  const std::size_t count = past + (causal ? local + 1 : rows_per_sequence);
  const std::size_t first_key = sequence * keys_per_sequence;
  const std::size_t kv_head = head / (heads / kv_heads);
  const float4* keys = reinterpret_cast<const float4*>(
      k_cache + (kv_head * capacity + first_key) * kDim);
  const float4* values = reinterpret_cast<const float4*>(
      v_cache + (kv_head * capacity + first_key) * kDim);
  const int lane = threadIdx.x % 32;
  const int wave = threadIdx.x / 32;
  float4 query = reinterpret_cast<const float4*>(
      q + (row * heads + head) * kDim)[lane];
  query.x *= scale;
  query.y *= scale;
  query.z *= scale;
  query.w *= scale;
  float maximum = -INFINITY;
  float sum = 0.0F;
  float4 out = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  for (std::size_t key = wave; key < count; key += kAttentionWaves) {
    const float4 k = keys[key * 32 + lane];
    const float4 v = values[key * 32 + lane];
    float score = query.x * k.x + query.y * k.y + query.z * k.z +
                  query.w * k.w;
    score = WaveSum(score);
    const float next = fmaxf(maximum, score);
    const float decay = expf(maximum - next);
    const float weight = expf(score - next);
    sum = sum * decay + weight;
    out.x = out.x * decay + weight * v.x;
    out.y = out.y * decay + weight * v.y;
    out.z = out.z * decay + weight * v.z;
    out.w = out.w * decay + weight * v.w;
    maximum = next;
  }
  if (lane == 0) {
    merge_max[wave] = maximum;
    merge_sum[wave] = sum;
  }
  merge_out[wave][lane] = out;
  __syncthreads();
  if (wave != 0)
    return;
  float total_max = merge_max[0];
  for (int w = 1; w < kAttentionWaves; ++w)
    total_max = fmaxf(total_max, merge_max[w]);
  float total = 0.0F;
  float4 result = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  for (int w = 0; w < kAttentionWaves; ++w) {
    const float factor = expf(merge_max[w] - total_max);
    total += merge_sum[w] * factor;
    const float4 part = merge_out[w][lane];
    result.x += part.x * factor;
    result.y += part.y * factor;
    result.z += part.z * factor;
    result.w += part.w * factor;
  }
  const float inverse = 1.0F / total;
  Bf16* target = output + (row * heads + head) * kDim + lane * 4;
  target[0] = ToBf16Bits(result.x * inverse);
  target[1] = ToBf16Bits(result.y * inverse);
  target[2] = ToBf16Bits(result.z * inverse);
  target[3] = ToBf16Bits(result.w * inverse);
}

__global__ void SwiGluKernel(const float* gate_up, Bf16* output,
                             std::size_t rows, std::size_t width) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * width)
    return;
  const std::size_t row = index / width;
  const std::size_t i = index % width;
  const float gate = gate_up[row * 2 * width + i];
  const float up = gate_up[row * 2 * width + width + i];
  output[index] = ToBf16Bits(gate / (1.0F + expf(-gate)) * up);
}

__global__ void AddBiasKernel(float* data, const Bf16* bias, std::size_t rows,
                              std::size_t width) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < rows * width)
    data[index] += FromBf16(bias[index % width]);
}

__global__ void ToBf16Kernel(const float* input, Bf16* output,
                             std::size_t count, bool silu) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  float value = input[index];
  if (silu)
    value = value / (1.0F + expf(-value));
  output[index] = ToBf16Bits(value);
}

__global__ void ConcatRowsKernel(const float* a, const float* a_scale,
                                 std::size_t a_width, const float* b,
                                 const float* b_scale, std::size_t b_width,
                                 Bf16* output) {
  const std::size_t row = blockIdx.x;
  const std::size_t width = a_width + b_width;
  const float sa = a_scale == nullptr ? 1.0F : a_scale[row];
  const float sb = b_scale == nullptr ? 1.0F : b_scale[row];
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    const float value = i < a_width ? a[row * a_width + i] * sa
                                    : b[row * b_width + i - a_width] * sb;
    output[row * width + i] = ToBf16Bits(value);
  }
}

__global__ void QuantizeKernel(const float* input, Bf16* output,
                               std::size_t count, float scale) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = ToBf16Bits(rintf(tanhf(input[index]) * scale) / scale);
}

__global__ void SelectRowsKernel(const float* quantized, const float* plain,
                                 const std::int32_t* mask, float* output,
                                 std::size_t width) {
  const std::size_t row = blockIdx.x;
  const float* source = mask[row] != 0 ? quantized : plain;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x)
    output[row * width + i] = source[row * width + i];
}

__global__ void PrependSpecialKernel(const float* rows_in, const Bf16* special,
                                     float* output, std::size_t patch_size,
                                     std::size_t width) {
  const std::size_t patch = blockIdx.x;
  const std::size_t row = blockIdx.y;  // 0..patch_size
  float* target = output + (patch * (patch_size + 1) + row) * width;
  const float* source =
      row == 0 ? nullptr : rows_in + (patch * patch_size + row - 1) * width;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x)
    target[i] = source == nullptr ? FromBf16(special[i]) : source[i];
}

__global__ void AssembleDitKernel(const float* mu, const float* time,
                                  const float* cond, const float* x,
                                  float* output, std::size_t patch_size,
                                  std::size_t width) {
  const std::size_t sequence = blockIdx.x;  // 0 conditional, 1 not
  const std::size_t row = blockIdx.y;
  const std::size_t length = 3 + 2 * patch_size;
  float* target = output + (sequence * length + row) * width;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    float value;
    if (row < 2)
      value = sequence == 0 ? mu[row * width + i] : 0.0F;
    else if (row == 2)
      value = time[i];
    else if (row < 3 + patch_size)
      value = cond[(row - 3) * width + i];
    else
      value = x[(row - 3 - patch_size) * width + i];
    target[i] = value;
  }
}

__global__ void GuidedEulerKernel(const float* velocity, float* x,
                                  std::size_t count, float cfg, float dt) {
  __shared__ float scratch[kThreads / 32];
  const float* c = velocity;
  const float* u = velocity + count;
  float dot = 0.0F;
  float norm = 0.0F;
  for (std::size_t i = threadIdx.x; i < count; i += blockDim.x) {
    dot += c[i] * u[i];
    norm += u[i] * u[i];
  }
  dot = BlockReduce<false>(dot, scratch);
  norm = BlockReduce<false>(norm, scratch);
  const float s = dot / (norm + 1e-8F);
  for (std::size_t i = threadIdx.x; i < count; i += blockDim.x) {
    const float guided = u[i] * s + cfg * (c[i] - u[i] * s);
    x[i] -= dt * guided;
  }
}

}  // namespace

void Embed(const Bf16* table, const std::int32_t* ids,
           const std::int32_t* text_mask, float* output, std::size_t rows,
           std::size_t width, float scale, hipStream_t stream) {
  if (rows == 0)
    return;
  EmbedKernel<<<rows, kThreads, 0, stream>>>(table, ids, text_mask, output,
                                             width, scale);
}

void ScatterRows(const float* input, const std::int32_t* indices,
                 float* output, std::size_t rows, std::size_t width,
                 hipStream_t stream) {
  if (rows == 0)
    return;
  ScatterRowsKernel<<<rows, kThreads, 0, stream>>>(input, indices, output,
                                                   width);
}

void GatherRows(const float* input, const std::int32_t* indices,
                float* output, std::size_t rows, std::size_t width,
                hipStream_t stream) {
  if (rows == 0)
    return;
  GatherRowsKernel<<<rows, kThreads, 0, stream>>>(input, indices, output,
                                                  width);
}

void RmsNorm(const float* input, const Bf16* weight, Bf16* output,
             float* output_f32, std::size_t rows, std::size_t width,
             float epsilon, hipStream_t stream) {
  RmsNormKernel<<<rows, kThreads, 0, stream>>>(input, weight, output,
                                               output_f32, width, epsilon);
}

void QkvRope(const float* qkv, const float* cos_sin,
             const std::int32_t* positions, const std::int32_t* cache_rows,
             float* q, float* k_cache, float* v_cache, std::size_t rows,
             std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
             std::size_t capacity, hipStream_t stream) {
  QkvRopeKernel<<<rows, kThreads, 0, stream>>>(
      qkv, cos_sin, positions, cache_rows, q, k_cache, v_cache, heads,
      kv_heads, head_dim, capacity);
}

void Attention(const float* q, const float* k_cache, const float* v_cache,
               Bf16* output, std::size_t rows, std::size_t rows_per_sequence,
               std::size_t keys_per_sequence, std::size_t past, bool causal,
               std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
               std::size_t capacity, hipStream_t stream) {
  // head_dim is 128 (IsSupported).
  AttentionKernel<<<dim3(static_cast<unsigned>(rows),
                         static_cast<unsigned>(heads)),
                    kAttentionWaves * 32, 0, stream>>>(
      q, k_cache, v_cache, output, rows_per_sequence, keys_per_sequence, past,
      causal, heads, kv_heads, capacity,
      1.0F / std::sqrt(static_cast<float>(head_dim)));
}

void SwiGlu(const float* gate_up, Bf16* output, std::size_t rows,
            std::size_t width, hipStream_t stream) {
  SwiGluKernel<<<Blocks(rows * width), kThreads, 0, stream>>>(gate_up, output,
                                                              rows, width);
}

void AddBias(float* data, const Bf16* bias, std::size_t rows,
             std::size_t width, hipStream_t stream) {
  AddBiasKernel<<<Blocks(rows * width), kThreads, 0, stream>>>(data, bias,
                                                               rows, width);
}

void ToBf16(const float* input, Bf16* output, std::size_t count, bool silu,
            hipStream_t stream) {
  ToBf16Kernel<<<Blocks(count), kThreads, 0, stream>>>(input, output, count,
                                                       silu);
}

void ConcatRows(const float* a, const float* a_scale, std::size_t a_width,
                const float* b, const float* b_scale, std::size_t b_width,
                Bf16* output, std::size_t rows, hipStream_t stream) {
  ConcatRowsKernel<<<rows, kThreads, 0, stream>>>(a, a_scale, a_width, b,
                                                  b_scale, b_width, output);
}

void Quantize(const float* input, Bf16* output, std::size_t count, float scale,
              hipStream_t stream) {
  QuantizeKernel<<<Blocks(count), kThreads, 0, stream>>>(input, output, count,
                                                         scale);
}

void SelectRows(const float* quantized, const float* plain,
                const std::int32_t* mask, float* output, std::size_t rows,
                std::size_t width, hipStream_t stream) {
  SelectRowsKernel<<<rows, kThreads, 0, stream>>>(quantized, plain, mask,
                                                  output, width);
}

void PrependSpecial(const float* rows_in, const Bf16* special, float* output,
                    std::size_t patches, std::size_t patch_size,
                    std::size_t width, hipStream_t stream) {
  PrependSpecialKernel<<<dim3(static_cast<unsigned>(patches),
                              static_cast<unsigned>(patch_size + 1)),
                         kThreads, 0, stream>>>(rows_in, special, output,
                                                patch_size, width);
}

void AssembleDit(const float* mu, const float* time, const float* cond,
                 const float* x, float* output, std::size_t patch_size,
                 std::size_t width, hipStream_t stream) {
  AssembleDitKernel<<<dim3(2, static_cast<unsigned>(3 + 2 * patch_size)),
                      kThreads, 0, stream>>>(mu, time, cond, x, output,
                                             patch_size, width);
}

void GuidedEuler(const float* velocity, float* x, std::size_t count,
                 float cfg, float dt, hipStream_t stream) {
  GuidedEulerKernel<<<1, kThreads, 0, stream>>>(velocity, x, count, cfg, dt);
}

}  // namespace gufo::models::voxcpm::kernels
