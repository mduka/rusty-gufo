#include "src/models/omnivoice/hip/generator_kernels.hpp"

#include <cmath>
#include <cfloat>

namespace gufo::models::omnivoice::kernels {
namespace {

constexpr int kThreads = 256;

/// Sum over the block; every thread receives the total. `scratch` holds one
/// float per thread.
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

/// Maximum and its lowest index over the block.
__device__ void BlockArgMax(float* value, int* index, float* scratch,
                            int* indices) {
  scratch[threadIdx.x] = *value;
  indices[threadIdx.x] = *index;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const float other = scratch[threadIdx.x + stride];
      const int other_index = indices[threadIdx.x + stride];
      if (other > scratch[threadIdx.x] ||
          (other == scratch[threadIdx.x] && other_index < indices[threadIdx.x])) {
        scratch[threadIdx.x] = other;
        indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  *value = scratch[0];
  *index = indices[0];
  __syncthreads();
}

__global__ void EmbedKernel(const float* text, std::size_t text_rows,
                            const float* table, const std::int32_t* ids,
                            float* hidden, std::size_t rows, std::size_t width,
                            std::size_t codebooks, std::size_t vocab) {
  const std::size_t row = blockIdx.x;
  float* out = hidden + row * width;
  if (row < text_rows) {
    const float* in = text + row * width;
    for (std::size_t d = threadIdx.x; d < width; d += blockDim.x)
      out[d] = in[d];
    return;
  }
  for (std::size_t d = threadIdx.x; d < width; d += blockDim.x) {
    float sum = 0.0F;
    for (std::size_t c = 0; c < codebooks; ++c) {
      const std::size_t entry =
          c * vocab + static_cast<std::size_t>(ids[c * rows + row]);
      sum += table[entry * width + d];
    }
    out[d] = sum;
  }
}

__global__ void RmsNormKernel(const float* input, const float* weight,
                              Half* output, std::size_t width, float epsilon) {
  __shared__ float scratch[kThreads];
  const float* in = input + blockIdx.x * width;
  float squares = 0.0F;
  for (std::size_t d = threadIdx.x; d < width; d += blockDim.x)
    squares += in[d] * in[d];
  const float scale =
      rsqrtf(BlockSum(squares, scratch) / static_cast<float>(width) + epsilon);
  Half* out = output + blockIdx.x * width;
  for (std::size_t d = threadIdx.x; d < width; d += blockDim.x)
    out[d] = __float2half(weight[d] * (in[d] * scale));
}

/// One block per (row, head slot): slots [0, heads) are query heads,
/// [heads, heads + kv_heads) key heads, then value heads. Each thread owns
/// the rotary pair (i, i + head_dim / 2).
__global__ void QkNormRopeKernel(const float* qkv, const float* q_norm,
                                 const float* k_norm,
                                 const std::int32_t* positions, Half* q,
                                 Half* k, Half* v, std::size_t rows,
                                 std::size_t heads, std::size_t kv_heads,
                                 std::size_t head_dim, float epsilon,
                                 float rope_theta) {
  extern __shared__ float scratch[];
  const std::size_t row = blockIdx.x;
  const std::size_t slot = blockIdx.y;
  const std::size_t half = head_dim / 2;
  const std::size_t i = threadIdx.x;
  const std::size_t stride = (heads + 2 * kv_heads) * head_dim;
  const float* in = qkv + row * stride + slot * head_dim;
  const std::size_t group = heads / kv_heads;
  const float first = i < half ? in[i] : 0.0F;
  const float second = i < half ? in[i + half] : 0.0F;
  if (slot >= heads + kv_heads) {
    // Values: repeat for every query head of the group.
    if (i >= half)
      return;
    const std::size_t kv = slot - heads - kv_heads;
    for (std::size_t g = 0; g < group; ++g) {
      Half* out = v + ((kv * group + g) * rows + row) * head_dim;
      out[i] = __float2half(first);
      out[i + half] = __float2half(second);
    }
    return;
  }
  const bool is_query = slot < heads;
  const float* weight = is_query ? q_norm : k_norm;
  const float total = BlockSum(first * first + second * second, scratch);
  if (i >= half)
    return;
  const float scale = rsqrtf(total / static_cast<float>(head_dim) + epsilon);
  const float a = weight[i] * (first * scale);
  const float b = weight[i + half] * (second * scale);
  // Hugging Face's default RoPE: inv_freq = 1 / theta^(2i / d) in float32.
  const float inv_freq =
      1.0F / powf(rope_theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
  const float angle = static_cast<float>(positions[row]) * inv_freq;
  const float cosine = cosf(angle);
  const float sine = sinf(angle);
  const float rotated_first = a * cosine - b * sine;
  const float rotated_second = b * cosine + a * sine;
  if (is_query) {
    Half* out = q + (slot * rows + row) * head_dim;
    out[i] = __float2half(rotated_first);
    out[i + half] = __float2half(rotated_second);
    return;
  }
  const std::size_t kv = slot - heads;
  for (std::size_t g = 0; g < group; ++g) {
    Half* out = k + ((kv * group + g) * rows + row) * head_dim;
    out[i] = __float2half(rotated_first);
    out[i + half] = __float2half(rotated_second);
  }
}

__global__ void SoftmaxKernel(const float* scores, Half* output,
                              std::size_t columns, float scale) {
  __shared__ float scratch[kThreads];
  const float* in = scores + blockIdx.x * columns;
  float maximum = -FLT_MAX;
  for (std::size_t j = threadIdx.x; j < columns; j += blockDim.x)
    maximum = fmaxf(maximum, in[j] * scale);
  maximum = BlockMax(maximum, scratch);
  float sum = 0.0F;
  for (std::size_t j = threadIdx.x; j < columns; j += blockDim.x)
    sum += expf(in[j] * scale - maximum);
  const float inverse = 1.0F / BlockSum(sum, scratch);
  Half* out = output + blockIdx.x * columns;
  for (std::size_t j = threadIdx.x; j < columns; j += blockDim.x)
    out[j] = __float2half(expf(in[j] * scale - maximum) * inverse);
}

__global__ void MergeHeadsKernel(const float* input, Half* output,
                                 std::size_t rows, std::size_t heads,
                                 std::size_t head_dim) {
  const std::size_t row = blockIdx.x;
  const std::size_t width = heads * head_dim;
  for (std::size_t j = threadIdx.x; j < width; j += blockDim.x) {
    const std::size_t head = j / head_dim;
    const std::size_t d = j % head_dim;
    output[row * width + j] =
        __float2half(input[(head * rows + row) * head_dim + d]);
  }
}

__global__ void SwiGluKernel(const float* gate_up, Half* output,
                             std::size_t width) {
  const float* in = gate_up + blockIdx.x * 2 * width;
  Half* out = output + blockIdx.x * width;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    const float gate = in[i];
    const float silu = gate / (1.0F + expf(-gate));
    out[i] = __float2half(silu * in[width + i]);
  }
}

__global__ void GatherRowsKernel(const float* input,
                                 const std::int32_t* indices, float* output,
                                 std::size_t width) {
  const float* in = input + static_cast<std::size_t>(indices[blockIdx.x]) * width;
  float* out = output + blockIdx.x * width;
  for (std::size_t d = threadIdx.x; d < width; d += blockDim.x)
    out[d] = in[d];
}

/// One block per (frame, codebook).
__global__ void ScoreKernel(const float* logits, std::int32_t* prediction,
                            float* confidence, std::size_t frames,
                            std::size_t codebooks, std::size_t vocab,
                            std::uint32_t mask_id, float guidance) {
  __shared__ float scratch[kThreads];
  __shared__ int indices[kThreads];
  const std::size_t frame = blockIdx.x;
  const std::size_t codebook = blockIdx.y;
  const std::size_t width = codebooks * vocab;
  const float* conditional = logits + frame * width + codebook * vocab;
  const float* unconditional = logits + (frames + frame) * width + codebook * vocab;
  // log_softmax of both rows.
  float c_max = -FLT_MAX, u_max = -FLT_MAX;
  for (std::size_t j = threadIdx.x; j < vocab; j += blockDim.x) {
    c_max = fmaxf(c_max, conditional[j]);
    u_max = fmaxf(u_max, unconditional[j]);
  }
  c_max = BlockMax(c_max, scratch);
  u_max = BlockMax(u_max, scratch);
  float c_sum = 0.0F, u_sum = 0.0F;
  for (std::size_t j = threadIdx.x; j < vocab; j += blockDim.x) {
    c_sum += expf(conditional[j] - c_max);
    u_sum += expf(unconditional[j] - u_max);
  }
  const float c_lse = c_max + logf(BlockSum(c_sum, scratch));
  const float u_lse = u_max + logf(BlockSum(u_sum, scratch));
  // Guided log-probabilities, then their log_softmax.
  const auto guided = [&](std::size_t j) {
    const float c = conditional[j] - c_lse;
    const float u = unconditional[j] - u_lse;
    return c + guidance * (c - u);
  };
  float g_max = -FLT_MAX;
  for (std::size_t j = threadIdx.x; j < vocab; j += blockDim.x)
    g_max = fmaxf(g_max, guided(j));
  g_max = BlockMax(g_max, scratch);
  float g_sum = 0.0F;
  float best = -FLT_MAX;
  int best_index = static_cast<int>(vocab);
  for (std::size_t j = threadIdx.x; j < vocab; j += blockDim.x) {
    const float value = guided(j);
    g_sum += expf(value - g_max);
    if (j != mask_id && value > best) {
      best = value;
      best_index = static_cast<int>(j);
    }
  }
  const float g_lse = g_max + logf(BlockSum(g_sum, scratch));
  BlockArgMax(&best, &best_index, scratch, indices);
  if (threadIdx.x == 0) {
    prediction[codebook * frames + frame] = best_index;
    confidence[codebook * frames + frame] = best - g_lse;
  }
}

}  // namespace

void Embed(const float* text, std::size_t text_rows, const float* table,
           const std::int32_t* ids, float* hidden, std::size_t rows,
           std::size_t width, std::size_t codebooks, std::size_t vocab,
           hipStream_t stream) {
  if (rows == 0)
    return;
  EmbedKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
      text, text_rows, table, ids, hidden, rows, width, codebooks, vocab);
}

void RmsNorm(const float* input, const float* weight, Half* output,
             std::size_t rows, std::size_t width, float epsilon,
             hipStream_t stream) {
  if (rows == 0)
    return;
  RmsNormKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
      input, weight, output, width, epsilon);
}

void QkNormRope(const float* qkv, const float* q_norm, const float* k_norm,
                const std::int32_t* positions, Half* q, Half* k, Half* v,
                std::size_t rows, std::size_t heads, std::size_t kv_heads,
                std::size_t head_dim, float epsilon, float rope_theta,
                hipStream_t stream) {
  if (rows == 0)
    return;
  // A power of two of at least head_dim / 2 threads for the block sum.
  unsigned threads = 1;
  while (threads < head_dim / 2)
    threads *= 2;
  const dim3 grid(static_cast<unsigned>(rows),
                  static_cast<unsigned>(heads + 2 * kv_heads));
  QkNormRopeKernel<<<grid, threads, threads * sizeof(float), stream>>>(
      qkv, q_norm, k_norm, positions, q, k, v, rows, heads, kv_heads, head_dim,
      epsilon, rope_theta);
}

void Softmax(const float* scores, Half* output, std::size_t rows,
             std::size_t columns, float scale, hipStream_t stream) {
  if (rows == 0)
    return;
  SoftmaxKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
      scores, output, columns, scale);
}

void MergeHeads(const float* input, Half* output, std::size_t rows,
                std::size_t heads, std::size_t head_dim, hipStream_t stream) {
  if (rows == 0)
    return;
  MergeHeadsKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
      input, output, rows, heads, head_dim);
}

void SwiGlu(const float* gate_up, Half* output, std::size_t rows,
            std::size_t width, hipStream_t stream) {
  if (rows == 0)
    return;
  SwiGluKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
      gate_up, output, width);
}

void GatherRows(const float* input, const std::int32_t* indices, float* output,
                std::size_t rows, std::size_t width, hipStream_t stream) {
  if (rows == 0)
    return;
  GatherRowsKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
      input, indices, output, width);
}

void Score(const float* logits, std::int32_t* prediction, float* confidence,
           std::size_t frames, std::size_t codebooks, std::size_t vocab,
           std::uint32_t mask_id, float guidance, hipStream_t stream) {
  if (frames == 0)
    return;
  const dim3 grid(static_cast<unsigned>(frames), static_cast<unsigned>(codebooks));
  ScoreKernel<<<grid, kThreads, 0, stream>>>(logits, prediction, confidence,
                                            frames, codebooks, vocab, mask_id,
                                            guidance);
}

}  // namespace gufo::models::omnivoice::kernels
