// Flat forms of the Nemotron ASR launchers (adapters.hpp).

#include "src/models/nemotron_asr/kernels/adapters.hpp"

#include <cstdint>

#include "src/models/nemotron_asr/hip/kernels.hpp"
#include "src/models/nemotron_asr/hip/wmma.hpp"

namespace gufo::models::nemotron_asr::kernels {

void ToHalfFlat(const float* x, std::uint16_t* y, std::size_t n,
                hipStream_t stream) {
  ToHalf(x, y, static_cast<std::int64_t>(n), stream);
}

void SiluHalfFlat(const float* x, std::uint16_t* y, std::size_t n,
                  hipStream_t stream) {
  SiluHalf(x, y, static_cast<std::int64_t>(n), stream);
}

void LayerNormInPlace(float* x, const float* weight, const float* bias,
                      int rows, int cols, hipStream_t stream) {
  LayerNorm(x, weight, bias, x, rows, cols, stream);
}

bool WmmaLinearFlat(const std::uint16_t* x, const std::uint16_t* tiles,
                    float* y, std::size_t rows, std::size_t in,
                    std::size_t out, bool accumulate, float scale,
                    const float* bias, bool relu, hipStream_t stream) {
  return WmmaLinear(x, tiles, y, rows, in, out, accumulate, scale, stream,
                    bias, relu);
}

bool RelPosAttentionHalfFlat(const float* qkv, const float* cache_keys,
                             const float* cache_values, const float* pos,
                             const float* bias_u, const float* bias_v,
                             std::uint16_t* out, int queries, int heads,
                             int q_stride, std::size_t first_query,
                             std::size_t first_key, int chunk, int left,
                             int pos_zero, int pos_rows, float scale,
                             hipStream_t stream) {
  const auto aligned = [](const void* p) {
    return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
  };
  if (!aligned(qkv) || !aligned(cache_keys) || !aligned(cache_values) ||
      !aligned(pos))
    return false;
  AttentionShape shape;
  shape.queries = queries;
  shape.heads = heads;
  shape.q_stride = q_stride;
  shape.first_query = static_cast<std::int64_t>(first_query);
  shape.first_key = static_cast<std::int64_t>(first_key);
  shape.chunk = chunk;
  shape.left = left;
  shape.pos_zero = pos_zero;
  shape.pos_rows = pos_rows;
  shape.scale = scale;
  RelPosAttentionHalf(qkv, cache_keys, cache_values, pos, bias_u, bias_v, out,
                      shape, stream);
  return true;
}

void UpdateKeyValueCacheFlat(float* keys, float* values, const float* qkv,
                             int stride, int cached, int rows, int keep,
                             int width, int capacity, hipStream_t stream) {
  (void)capacity;
  UpdateKeyValueCache(keys, values, qkv, stride, cached, rows, keep, width,
                      stream);
}

void ArgmaxRowsFlat(const float* logits, const float* bias, int rows,
                    int columns, std::uint64_t* best, hipStream_t stream) {
  static_assert(sizeof(std::uint64_t) == sizeof(unsigned long long));
  ArgmaxRows(logits, bias, rows, columns,
             reinterpret_cast<unsigned long long*>(best), stream);
}

}  // namespace gufo::models::nemotron_asr::kernels
