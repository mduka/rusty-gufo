// Kernels of the encoder and transducer that model.cpp ports from
// transformers' Nemotron 3.5 ASR (Apache-2.0, see licenses/transformers.txt),
// itself a port of NVIDIA NeMo's FastConformer and RNN-T (Apache-2.0) whose
// relative-position attention NeMo adopts from ESPnet (Apache-2.0; both in
// licenses/NeMo.txt).
#include "kernels.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "wmma_fragment.hpp"

namespace gufo::models::nemotron_asr::kernels {
namespace {

constexpr int kGroup = 32;  // lanes that reduce together (wave32 or half a wave64)
constexpr int kHeadDim = 128;

__device__ float GroupSum(float value) {
  for (int offset = kGroup / 2; offset > 0; offset /= 2)
    value += __shfl_xor(value, offset, kGroup);
  return value;
}

__device__ float GroupMax(float value) {
  for (int offset = kGroup / 2; offset > 0; offset /= 2)
    value = fmaxf(value, __shfl_xor(value, offset, kGroup));
  return value;
}

__device__ float Sigmoid(float x) {
  return 1.0F / (1.0F + expf(-x));
}

/// Eight float16 values as floats.
__device__ void Unpack(const uint4& raw, float (&out)[8]) {
  const auto* pairs = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 pair = __half22float2(pairs[i]);
    out[2 * i] = pair.x;
    out[2 * i + 1] = pair.y;
  }
}

__device__ float Dot8(const float (&w)[8], const float4& a, const float4& b,
                      float sum) {
  sum = fmaf(w[0], a.x, sum);
  sum = fmaf(w[1], a.y, sum);
  sum = fmaf(w[2], a.z, sum);
  sum = fmaf(w[3], a.w, sum);
  sum = fmaf(w[4], b.x, sum);
  sum = fmaf(w[5], b.y, sum);
  sum = fmaf(w[6], b.z, sum);
  sum = fmaf(w[7], b.w, sum);
  return sum;
}

__device__ void Store(float* out, std::int64_t index, float value) {
  out[index] = value;
}

__device__ void Store(__half* out, std::int64_t index, float value) {
  out[index] = __float2half(value);
}

// One thread per channel, one block per output row and kStemFrequencies
// output frequencies: the block stages the three mel rows' columns it
// reads in LDS, each thread keeps its channel's nine weights.
constexpr int kStemFrequencies = 8;
constexpr int kStemColumns = 2 * kStemFrequencies + 1;

__global__ void SubsampleInputKernel(const float* mel, int bins,
                                     const float* weight, const float* bias,
                                     int channels, __half* out, int out_freq) {
  __shared__ float rows[3][kStemColumns];
  const int t = blockIdx.x;
  const int first = blockIdx.y * kStemFrequencies;
  for (int i = threadIdx.x; i < 3 * kStemColumns; i += blockDim.x) {
    const int dt = i / kStemColumns;
    const int in_f = 2 * first - 2 + i % kStemColumns;
    rows[dt][i % kStemColumns] =
        in_f >= 0 && in_f < bins
            ? mel[static_cast<std::int64_t>(2 * t + dt) * bins + in_f]
            : 0.0F;
  }
  __syncthreads();
  const int c = threadIdx.x;
  if (c >= channels)
    return;
  float w[9];
#pragma unroll
  for (int i = 0; i < 9; ++i)
    w[i] = weight[c * 9 + i];
  const float b = bias[c];
  for (int j = 0; j < kStemFrequencies; ++j) {
    const int f = first + j;
    if (f >= out_freq)
      break;
    // The same terms in the same order as a per-element loop that skips
    // the padding columns (whose zeros would add nothing).
    float sum = b;
#pragma unroll
    for (int dt = 0; dt < 3; ++dt)
#pragma unroll
      for (int df = 0; df < 3; ++df) {
        const int in_f = 2 * f + df - 2;
        if (in_f < 0 || in_f >= bins)
          continue;
        sum = fmaf(w[dt * 3 + df], rows[dt][2 * j + df], sum);
      }
    out[(static_cast<std::int64_t>(t) * out_freq + f) * channels + c] =
        SaturatedHalf(fmaxf(sum, 0.0F));
  }
}

// One thread per output row, frequency and pair of channels.
__global__ void SubsampleDepthwiseKernel(const __half2* in, int in_freq,
                                         const float* weight,
                                         const float* bias, int pairs,
                                         __half2* out, int out_rows,
                                         int out_freq) {
  const std::int64_t index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t total =
      static_cast<std::int64_t>(out_rows) * out_freq * pairs;
  if (index >= total)
    return;
  const int pair = static_cast<int>(index % pairs);
  const int f = static_cast<int>((index / pairs) % out_freq);
  const int t =
      static_cast<int>(index / (static_cast<std::int64_t>(pairs) * out_freq));
  const int c = 2 * pair;
  float sum0 = bias[c];
  float sum1 = bias[c + 1];
  for (int dt = 0; dt < 3; ++dt)
    for (int df = 0; df < 3; ++df) {
      const int in_f = 2 * f + df - 2;
      if (in_f < 0 || in_f >= in_freq)
        continue;
      const float2 value = __half22float2(
          in[(static_cast<std::int64_t>(2 * t + dt) * in_freq + in_f) * pairs +
             pair]);
      sum0 += weight[c * 9 + dt * 3 + df] * value.x;
      sum1 += weight[(c + 1) * 9 + dt * 3 + df] * value.y;
    }
  out[index] = __halves2half2(SaturatedHalf(sum0), SaturatedHalf(sum1));
}

// One block of 256 threads per row.
template<typename Out, bool kSilu>
__global__ void LayerNormKernel(const float* x, const float* weight,
                                const float* bias, Out* y, int cols) {
  __shared__ float partial[256 / kGroup];
  const float* row = x + static_cast<std::int64_t>(blockIdx.x) * cols;
  Out* out = y + static_cast<std::int64_t>(blockIdx.x) * cols;
  const int lane = threadIdx.x % kGroup;
  const int group = threadIdx.x / kGroup;
  const int groups = blockDim.x / kGroup;

  float sum = 0.0F;
  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    sum += row[c];
  sum = GroupSum(sum);
  if (lane == 0)
    partial[group] = sum;
  __syncthreads();
  float total = 0.0F;
  for (int g = 0; g < groups; ++g)
    total += partial[g];
  const float mean = total / static_cast<float>(cols);
  __syncthreads();

  float squares = 0.0F;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float d = row[c] - mean;
    squares += d * d;
  }
  squares = GroupSum(squares);
  if (lane == 0)
    partial[group] = squares;
  __syncthreads();
  float variance = 0.0F;
  for (int g = 0; g < groups; ++g)
    variance += partial[g];
  variance /= static_cast<float>(cols);
  const float inverse = rsqrtf(variance + 1e-5F);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float value = (row[c] - mean) * inverse * weight[c] + bias[c];
    if constexpr (kSilu)
      value *= Sigmoid(value);
    Store(out, c, value);
  }
}

__global__ void SiluHalfKernel(const float* x, __half* y, std::int64_t n) {
  const std::int64_t index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n)
    y[index] = __float2half(x[index] * Sigmoid(x[index]));
}

__global__ void ToHalfKernel(const float* x, __half* y, std::int64_t n) {
  const std::int64_t index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n)
    y[index] = __float2half(x[index]);
}

// Matrix-core attention: one block of four waves per (group of chunks,
// head, tile of 16 of the group's queries). A group is as many whole chunks
// as fit in 16 queries, or one chunk in tiles of 16; a block scores its
// queries against the union of their keys, and each query's softmax keeps
// its own chunk's window. The content terms (q + u) . k, the position terms
// (q + v) . pos[distance] and the probabilities times the values are
// products of float16 tiles summed in float32 (wmma_fragment.hpp: A holds
// queries or probabilities, B keys, distances or value dimensions); the
// scores and the softmax are float32. Rows and columns past the queries,
// keys or distances read clamped (finite) rows and are never used: the
// probabilities outside a query's window are zero.
constexpr int kAttentionWaves = 4;
constexpr int kAttentionThreads = kAttentionWaves * 32;
static_assert(kAttentionThreads == kHeadDim, "a thread per head dimension");
constexpr int kQueryStride = kHeadDim + 8;  // halves per staged query row

__host__ __device__ constexpr int Round16(int n) {
  return (n + 15) / 16 * 16;
}

__host__ __device__ constexpr int GroupChunks(int chunk) {
  return chunk >= 16 ? 1 : 16 / chunk;
}

struct AttentionLayout {
  int keys;       // score columns: keys of a group, padded to 16
  int distances;  // term columns: 16 queries + keys - 1, padded to 16
  // Byte offsets in LDS after the staged queries ([2][16][kQueryStride]).
  [[nodiscard]] __host__ __device__ int Scores() const {
    return 2 * 16 * kQueryStride * 2;
  }
  [[nodiscard]] __host__ __device__ int Terms() const {
    return Scores() + 16 * keys * 4;
  }
  [[nodiscard]] __host__ __device__ int Probabilities() const {
    return Terms() + 16 * distances * 4;
  }
  [[nodiscard]] __host__ __device__ int Inverse() const {
    return Probabilities() + 16 * (keys + 8) * 2;
  }
  [[nodiscard]] __host__ __device__ int Bytes() const {
    return Inverse() + 16 * 4;
  }
};

__global__ __launch_bounds__(kAttentionThreads) void RelPosAttentionKernel(
    const float* qkv, const float* cache_keys, const float* cache_values,
    const float* pos, const float* bias_u, const float* bias_v, __half* out,
    AttentionShape shape, AttentionLayout layout) {
  extern __shared__ __align__(16) unsigned char shared[];
  auto* query_u = reinterpret_cast<_Float16*>(shared);
  auto* query_v = query_u + 16 * kQueryStride;
  auto* scores = reinterpret_cast<float*>(shared + layout.Scores());
  auto* terms = reinterpret_cast<float*>(shared + layout.Terms());
  auto* probabilities =
      reinterpret_cast<_Float16*>(shared + layout.Probabilities());
  auto* inverse = reinterpret_cast<float*>(shared + layout.Inverse());
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int row = lane % 16;
  // Wave-uniform: scalar, so its guards do not split the wave.
  const int wave = __builtin_amdgcn_readfirstlane(tid / 32);
  const int width = shape.heads * kHeadDim;

  // The group's frames in this pass end at `end`: its keys are begin..end -
  // 1, this block's queries a0..a1 - 1 (absolute frames).
  const std::int64_t pass_end = shape.first_query + shape.queries;
  const int group_chunks = GroupChunks(shape.chunk);
  const std::int64_t chunk =
      shape.first_query / shape.chunk +
      static_cast<std::int64_t>(blockIdx.x) * group_chunks;
  std::int64_t end = (chunk + group_chunks) * shape.chunk;
  if (end > pass_end)
    end = pass_end;
  std::int64_t start = chunk * shape.chunk;
  if (start < shape.first_query)
    start = shape.first_query;
  const std::int64_t a0 = start + 16 * static_cast<std::int64_t>(blockIdx.z);
  if (a0 >= end)
    return;  // the chunk's last tile is shorter in this pass
  const std::int64_t a1 = a0 + 16 < end ? a0 + 16 : end;
  std::int64_t begin = chunk * shape.chunk - shape.left;
  if (begin < shape.first_key)
    begin = shape.first_key;
  const int queries = static_cast<int>(a1 - a0);
  const int count = static_cast<int>(end - begin);
  const int distances = queries + count - 1;
  // Distance of term column r: a0 - (end - 1) + r; query t and key j use
  // column t - j + count - 1.
  const std::int64_t nearest = a0 - (end - 1);
  const float* q_rows = qkv + (a0 - shape.first_query) * shape.q_stride;

  // Loads are issued in batches with clamped indices and selected results:
  // branches around them would make each wait for the one before.
  {
    const int column = h * kHeadDim + tid;
    const float u = bias_u[column];
    const float v = bias_v[column];
    float values[16];
#pragma unroll
    for (int t = 0; t < 16; ++t)
      values[t] =
          q_rows[static_cast<std::int64_t>(min(t, queries - 1)) *
                     shape.q_stride +
                 column];
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      const bool valid = t < queries;
      query_u[t * kQueryStride + tid] =
          valid ? static_cast<_Float16>(values[t] + u) : _Float16(0);
      query_v[t * kQueryStride + tid] =
          valid ? static_cast<_Float16>(values[t] + v) : _Float16(0);
    }
  }
  __syncthreads();

  // Scores: tiles of 16 keys, then of 16 distances, in turn over the waves.
  const int key_tiles = (count + 15) / 16;
  const int distance_tiles = (distances + 15) / 16;
  for (int tile = wave; tile < key_tiles + distance_tiles;
       tile += kAttentionWaves) {
    const bool is_key = tile < key_tiles;
    const int first = is_key ? tile * 16 : (tile - key_tiles) * 16;
    const int items = is_key ? count : distances;
    const int item = first + row;
    const int clamped = min(item, items - 1);
    const std::int64_t key = begin + clamped;
    // A group's distances reach past the table's by up to 16 - chunk frames
    // on each side, for keys outside every query's window (terms never
    // used): those read its first or last row.
    std::int64_t distance_row = shape.pos_zero + nearest + clamped;
    distance_row = distance_row < 0 ? 0 : distance_row;
    distance_row =
        distance_row < shape.pos_rows ? distance_row : shape.pos_rows - 1;
    const float* source =
        is_key ? (key < shape.first_query
                      ? cache_keys + (key - shape.first_key) * width
                      : qkv + (key - shape.first_query) * shape.q_stride +
                            width)
               : pos + distance_row * width;
    const auto* source4 =
        reinterpret_cast<const float4*>(source + h * kHeadDim);
    const _Float16* a_row = (is_key ? query_u : query_v) + row * kQueryStride;
    Accumulator acc = {};
#pragma unroll
    for (int part = 0; part < 2; ++part) {
      float4 raw[16];  // 64 dimensions: four steps of 16
#pragma unroll
      for (int i = 0; i < 16; ++i)
        raw[i] = source4[part * 16 + i];
#pragma unroll
      for (int step = 0; step < 4; ++step) {
        Fragment b;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float4 r = raw[step * 4 + i];
          b[4 * i] = static_cast<_Float16>(r.x);
          b[4 * i + 1] = static_cast<_Float16>(r.y);
          b[4 * i + 2] = static_cast<_Float16>(r.z);
          b[4 * i + 3] = static_cast<_Float16>(r.w);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            LoadFragment(a_row + (part * 4 + step) * 16), b, acc);
      }
    }
    float* target = is_key ? scores : terms;
    const int stride = is_key ? layout.keys : layout.distances;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      target[(2 * i + lane / 16) * stride + first + row] = acc[i];
  }
  __syncthreads();

  // Softmax per query row in float32 over its chunk's window; probabilities
  // as float16, zero outside the window and in the rows past the queries.
  for (int t = wave; t < 16; t += kAttentionWaves) {
    _Float16* p = probabilities + t * (layout.keys + 8);
    if (t >= queries) {
      for (int j = lane; j < layout.keys; j += kGroup)
        p[j] = _Float16(0);
      continue;
    }
    // The window of the query's chunk, in key columns.
    const std::int64_t own = (a0 + t) / shape.chunk;
    std::int64_t low = own * shape.chunk - shape.left;
    if (low < shape.first_key)
      low = shape.first_key;
    std::int64_t high = (own + 1) * shape.chunk;
    if (high > pass_end)
      high = pass_end;
    const int lo = static_cast<int>(low - begin);
    const int hi = static_cast<int>(high - begin);
    const float* score = scores + t * layout.keys;
    const float* term = terms + t * layout.distances + t + count - 1;
    float best = -INFINITY;
    for (int j = lo + lane; j < hi; j += kGroup)
      best = fmaxf(best, (score[j] + term[-j]) * shape.scale);
    best = GroupMax(best);
    float sum = 0.0F;
    for (int j = lane; j < layout.keys; j += kGroup) {
      float e = 0.0F;
      if (j >= lo && j < hi)
        e = expf((score[j] + term[-j]) * shape.scale - best);
      sum += e;
      p[j] = static_cast<_Float16>(e);
    }
    sum = GroupSum(sum);
    if (lane == 0)
      inverse[t] = 1.0F / sum;
  }
  __syncthreads();

  // Probabilities times values: each wave two tiles of 16 dimensions.
  const _Float16* a_row = probabilities + row * (layout.keys + 8);
  Accumulator acc[2] = {};
  for (int step = 0; step < key_tiles; ++step) {
    float raw[2][16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      const std::int64_t key = begin + min(step * 16 + e, count - 1);
      const float* values =
          key < shape.first_query
              ? cache_values + (key - shape.first_key) * width
              : qkv + (key - shape.first_query) * shape.q_stride + 2 * width;
#pragma unroll
      for (int d = 0; d < 2; ++d)
        raw[d][e] = values[h * kHeadDim + (wave + kAttentionWaves * d) * 16 +
                           row];
    }
    const Fragment a = LoadFragment(a_row + step * 16);
#pragma unroll
    for (int d = 0; d < 2; ++d) {
      Fragment b;
#pragma unroll
      for (int e = 0; e < 16; ++e)
        b[e] = static_cast<_Float16>(raw[d][e]);
      acc[d] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[d]);
    }
  }
#pragma unroll
  for (int d = 0; d < 2; ++d) {
    const int dim = h * kHeadDim + (wave + kAttentionWaves * d) * 16 + row;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int t = 2 * i + lane / 16;
      if (t < queries)
        out[(a0 - shape.first_query + t) * width + dim] =
            __float2half(acc[d][i] * inverse[t]);
    }
  }
}

// The last `keep` rows of [cache (cached rows) | the pass's rows], written
// over the cache: one thread per column walks the rows upwards, so every
// cache row is read before it is overwritten.
__global__ void UpdateKeyValueCacheKernel(float* keys, float* values,
                                          const float* qkv, int stride,
                                          int cached, int rows, int keep,
                                          int width) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= width)
    return;
  const int first = cached + rows - keep;
  for (int i = 0; i < keep; ++i) {
    const int at = first + i;
    float k, v;
    if (at < cached) {
      k = keys[static_cast<std::int64_t>(at) * width + c];
      v = values[static_cast<std::int64_t>(at) * width + c];
    } else {
      const float* row = qkv + static_cast<std::int64_t>(at - cached) * stride;
      k = row[width + c];
      v = row[2 * width + c];
    }
    keys[static_cast<std::int64_t>(i) * width + c] = k;
    values[static_cast<std::int64_t>(i) * width + c] = v;
  }
}

__device__ float GluAt(const float* in, const float* cache, int history,
                       int index, int channels, int c) {
  if (index < 0)
    return cache[static_cast<std::int64_t>(history + index) * channels + c];
  const float* row = in + static_cast<std::int64_t>(index) * 2 * channels;
  return row[c] * Sigmoid(row[channels + c]);
}

// out[r][c] = sum_j weight[c][j] * glu[r + j - (kernel - 1)][c], the GLU
// of `in` before row 0 taken from `cache` (kernel - 1 rows).
__global__ void GluDepthwiseKernel(const float* in, const float* cache,
                                   const float* weight, float* out, int rows,
                                   int channels, int kernel) {
  const std::int64_t index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::int64_t>(rows) * channels)
    return;
  const int r = static_cast<int>(index / channels);
  const int c = static_cast<int>(index % channels);
  const int history = kernel - 1;
  float sum = 0.0F;
  for (int j = 0; j < kernel; ++j)
    sum += weight[c * kernel + j] *
           GluAt(in, cache, history, r + j - history, channels, c);
  out[index] = sum;
}

// The conv cache becomes the last kernel - 1 GLU rows of [cache | pass].
__global__ void UpdateConvCacheKernel(const float* in, float* cache, int rows,
                                      int channels, int kernel) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;
  const int history = kernel - 1;
  for (int i = 0; i < history; ++i)
    cache[static_cast<std::int64_t>(i) * channels + c] =
        GluAt(in, cache, history, rows - history + i, channels, c);
}

// Eight rows per block of 256 threads: 32 lanes per row, lane l summing
// the float16 weights of columns 8l..8l + 7, 8(l + 32).. of concat(x1, x2)
// in that order.
__global__ void GemvKernel(const __half* weight, const float* x1, int k1,
                           const float* x2, int k2, const float* bias,
                           float* y, int rows) {
  const int row = blockIdx.x * (blockDim.x / kGroup) + threadIdx.x / kGroup;
  const int lane = threadIdx.x % kGroup;
  if (row >= rows)
    return;
  const int octets1 = k1 / 8;
  const int octets = octets1 + k2 / 8;
  const auto* w = reinterpret_cast<const uint4*>(
      weight + static_cast<std::int64_t>(row) * (k1 + k2));
  const auto* a = reinterpret_cast<const float4*>(x1);
  const auto* b = reinterpret_cast<const float4*>(x2);
  float sum = 0.0F;
#pragma unroll 4
  for (int o = lane; o < octets; o += kGroup) {
    float v[8];
    Unpack(w[o], v);
    const float4* x = o < octets1 ? a + 2 * o : b + 2 * (o - octets1);
    sum = Dot8(v, x[0], x[1], sum);
  }
  sum = GroupSum(sum);
  if (lane == 0)
    y[row] = sum + (bias != nullptr ? bias[row] : 0.0F);
}

__global__ void LstmCellKernel(const float* gates, float* h, float* c,
                               int hidden) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hidden)
    return;
  const float input = Sigmoid(gates[i]);
  const float forget = Sigmoid(gates[hidden + i]);
  const float cell = tanhf(gates[2 * hidden + i]);
  const float output = Sigmoid(gates[3 * hidden + i]);
  const float next = forget * c[i] + input * cell;
  c[i] = next;
  h[i] = output * tanhf(next);
}

__device__ unsigned long long Pack(float value, int index) {
  unsigned int bits = __float_as_uint(value);
  bits = (bits & 0x80000000U) != 0U ? ~bits : bits | 0x80000000U;
  return (static_cast<unsigned long long>(bits) << 32) |
         static_cast<unsigned long long>(0xFFFFFFFFU -
                                         static_cast<unsigned int>(index));
}

// relu(encoder[f] + decoder) as float16, the joint's input per frame.
__global__ void JointInputKernel(const float* encoder, const float* decoder,
                                 __half* out, int width, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = __float2half(fmaxf(encoder[i] + decoder[i % width], 0.0F));
}

// One block per frame: the packed arg max of logits + bias over the row.
__global__ void ArgmaxRowsKernel(const float* logits, const float* bias,
                                 int columns, unsigned long long* best) {
  __shared__ unsigned long long partial[256 / kGroup];
  const float* row = logits + static_cast<std::int64_t>(blockIdx.x) * columns;
  unsigned long long mine = 0;
  for (int c = threadIdx.x; c < columns; c += blockDim.x) {
    const unsigned long long packed = Pack(row[c] + bias[c], c);
    mine = packed > mine ? packed : mine;
  }
  for (int offset = kGroup / 2; offset > 0; offset /= 2) {
    const unsigned long long other = __shfl_xor(mine, offset, kGroup);
    mine = other > mine ? other : mine;
  }
  if (threadIdx.x % kGroup == 0)
    partial[threadIdx.x / kGroup] = mine;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int g = 1; g < static_cast<int>(blockDim.x) / kGroup; ++g)
      mine = partial[g] > mine ? partial[g] : mine;
    best[blockIdx.x] = mine;
  }
}

int Blocks(std::int64_t n, int threads) {
  return static_cast<int>((n + threads - 1) / threads);
}

}  // namespace

void SubsampleInput(const float* mel, int bins, const float* weight,
                    const float* bias, int channels, std::uint16_t* out,
                    int out_rows, int out_freq, hipStream_t stream) {
  if (out_rows == 0 || out_freq == 0)
    return;
  const dim3 grid(static_cast<unsigned>(out_rows),
                  static_cast<unsigned>((out_freq + kStemFrequencies - 1) /
                                        kStemFrequencies));
  SubsampleInputKernel<<<grid, channels, 0, stream>>>(
      mel, bins, weight, bias, channels, reinterpret_cast<__half*>(out),
      out_freq);
}

void SubsampleDepthwise(const std::uint16_t* in, int in_freq,
                        const float* weight, const float* bias, int channels,
                        std::uint16_t* out, int out_rows, int out_freq,
                        hipStream_t stream) {
  const int pairs = channels / 2;
  const std::int64_t n = static_cast<std::int64_t>(out_rows) * out_freq * pairs;
  if (n == 0)
    return;
  SubsampleDepthwiseKernel<<<Blocks(n, 256), 256, 0, stream>>>(
      reinterpret_cast<const __half2*>(in), in_freq, weight, bias, pairs,
      reinterpret_cast<__half2*>(out), out_rows, out_freq);
}

void LayerNorm(const float* x, const float* weight, const float* bias,
               float* y, int rows, int cols, hipStream_t stream) {
  if (rows == 0)
    return;
  LayerNormKernel<float, false><<<rows, 256, 0, stream>>>(x, weight, bias, y,
                                                          cols);
}

void LayerNormHalf(const float* x, const float* weight, const float* bias,
                   std::uint16_t* y, int rows, int cols, bool silu,
                   hipStream_t stream) {
  if (rows == 0)
    return;
  auto* out = reinterpret_cast<__half*>(y);
  if (silu)
    LayerNormKernel<__half, true><<<rows, 256, 0, stream>>>(x, weight, bias,
                                                            out, cols);
  else
    LayerNormKernel<__half, false><<<rows, 256, 0, stream>>>(x, weight, bias,
                                                             out, cols);
}

void SiluHalf(const float* x, std::uint16_t* y, std::int64_t n,
              hipStream_t stream) {
  if (n == 0)
    return;
  SiluHalfKernel<<<Blocks(n, 256), 256, 0, stream>>>(
      x, reinterpret_cast<__half*>(y), n);
}

void ToHalf(const float* x, std::uint16_t* y, std::int64_t n,
            hipStream_t stream) {
  if (n == 0)
    return;
  ToHalfKernel<<<Blocks(n, 256), 256, 0, stream>>>(
      x, reinterpret_cast<__half*>(y), n);
}

void RelPosAttentionHalf(const float* qkv, const float* cache_keys,
                         const float* cache_values, const float* pos,
                         const float* bias_u, const float* bias_v,
                         std::uint16_t* out, const AttentionShape& shape,
                         hipStream_t stream) {
  if (shape.queries == 0)
    return;
  // Unreachable for supported configurations (IsSupported) and the model's
  // table, which holds the distances of the widest lookahead's window.
  if (shape.left + shape.chunk > kMaxKeys)
    throw std::invalid_argument("attention window beyond kMaxKeys frames");
  if (shape.pos_zero < shape.chunk - 1 ||
      shape.pos_zero + shape.left + shape.chunk > shape.pos_rows)
    throw std::invalid_argument(
        "position table shorter than the attention window");
  const auto aligned = [](const void* p) {
    return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
  };
  if (shape.q_stride % 4 != 0 || !aligned(qkv) || !aligned(cache_keys) ||
      !aligned(cache_values) || !aligned(pos))
    throw std::invalid_argument("attention rows must be 16-byte aligned");
  const std::int64_t first = shape.first_query / shape.chunk;
  const std::int64_t last =
      (shape.first_query + shape.queries - 1) / shape.chunk;
  const int group = GroupChunks(shape.chunk) * shape.chunk;
  AttentionLayout layout{};
  layout.keys = Round16(shape.left + group);
  layout.distances = Round16(16 + shape.left + group - 1);
  const dim3 grid(
      static_cast<unsigned>((last - first) / GroupChunks(shape.chunk) + 1),
      static_cast<unsigned>(shape.heads),
      static_cast<unsigned>((group + 15) / 16));
  RelPosAttentionKernel<<<grid, kAttentionThreads,
                          static_cast<std::size_t>(layout.Bytes()), stream>>>(
      qkv, cache_keys, cache_values, pos, bias_u, bias_v,
      reinterpret_cast<__half*>(out), shape, layout);
}

void UpdateKeyValueCache(float* keys, float* values, const float* qkv,
                         int stride, int cached, int rows, int keep, int width,
                         hipStream_t stream) {
  if (keep == 0)
    return;
  UpdateKeyValueCacheKernel<<<Blocks(width, 256), 256, 0, stream>>>(
      keys, values, qkv, stride, cached, rows, keep, width);
}

void GluDepthwise(const float* in, const float* cache, const float* weight,
                  float* out, int rows, int channels, int kernel,
                  hipStream_t stream) {
  const std::int64_t n = static_cast<std::int64_t>(rows) * channels;
  if (n == 0)
    return;
  GluDepthwiseKernel<<<Blocks(n, 256), 256, 0, stream>>>(
      in, cache, weight, out, rows, channels, kernel);
}

void UpdateConvCache(const float* in, float* cache, int rows, int channels,
                     int kernel, hipStream_t stream) {
  UpdateConvCacheKernel<<<Blocks(channels, 256), 256, 0, stream>>>(
      in, cache, rows, channels, kernel);
}

void Gemv(const std::uint16_t* weight, const float* x1, int k1,
          const float* x2, int k2, const float* bias, float* y, int rows,
          hipStream_t stream) {
  constexpr int kThreads = 256;
  GemvKernel<<<Blocks(rows, kThreads / kGroup), kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(weight), x1, k1, x2, k2, bias, y, rows);
}

void LstmCell(const float* gates, float* h, float* c, int hidden,
              hipStream_t stream) {
  LstmCellKernel<<<Blocks(hidden, 256), 256, 0, stream>>>(gates, h, c,
                                                          hidden);
}

void JointInput(const float* encoder, const float* decoder,
                std::uint16_t* out, int frames, int width,
                hipStream_t stream) {
  const int count = frames * width;
  if (count == 0)
    return;
  JointInputKernel<<<Blocks(count, 256), 256, 0, stream>>>(
      encoder, decoder, reinterpret_cast<__half*>(out), width, count);
}

void ArgmaxRows(const float* logits, const float* bias, int rows, int columns,
                unsigned long long* best, hipStream_t stream) {
  if (rows == 0)
    return;
  ArgmaxRowsKernel<<<rows, 256, 0, stream>>>(logits, bias, columns, best);
}

}  // namespace gufo::models::nemotron_asr::kernels
