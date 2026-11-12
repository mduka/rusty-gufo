#include "src/models/qwen_image_21/hip/ops.hpp"
#include "src/models/qwen_image_21/kernels/adapters.hpp"

namespace gufo::models::qwen_image_21::hip {
namespace {

const std::uint16_t* Bf16(const void* pointer) {
  return static_cast<const std::uint16_t*>(pointer);
}

std::uint16_t* Bf16(void* pointer) {
  return static_cast<std::uint16_t*>(pointer);
}

int Int(std::uint32_t value) {
  return static_cast<int>(value);
}

}  // namespace

void LaunchPackDenseFlat(const void* input, void* output, std::uint32_t rows,
                         std::uint32_t columns, hipStream_t stream) {
  PackDense16(Bf16(input), Bf16(output), Int(rows), Int(columns), stream);
}

bool LaunchDenseFlat(const void* weights, const void* input, void* output,
                     std::uint32_t rows, std::uint32_t channels,
                     std::uint32_t inner, hipStream_t stream) {
  return DenseBf16(Bf16(weights), Bf16(input), Bf16(output), Int(rows),
                   Int(channels), Int(inner), stream);
}

bool LaunchDenseSwiGluFlat(const void* gate, const void* up, const void* input,
                           void* output, std::uint32_t rows,
                           std::uint32_t channels, std::uint32_t inner,
                           hipStream_t stream) {
  return DenseSwiGlu(Bf16(gate), Bf16(up), Bf16(input), Bf16(output), Int(rows),
                     Int(channels), Int(inner), stream);
}

bool LaunchConvolve3Flat(const void* input, const void* weights,
                         const void* bias, void* output, std::uint32_t height,
                         std::uint32_t width, std::uint32_t channels,
                         std::uint32_t output_channels, std::uint32_t stride,
                         std::uint32_t padding_total, std::uint32_t upscale,
                         hipStream_t stream) {
  if (padding_total > 2)
    return false;
  const int padding = padding_total == 1 ? -1 : Int(padding_total / 2);
  return Convolve3(Bf16(input), Bf16(weights), Bf16(bias), Bf16(output),
                   Int(height), Int(width), Int(channels), Int(output_channels),
                   stream, Int(stride), padding, Int(upscale));
}

bool LaunchExpandConvolution3Flat(const void* input, void* output,
                                  std::uint32_t height, std::uint32_t width,
                                  std::uint32_t channels, std::uint32_t stride,
                                  std::uint32_t padding, std::uint32_t upscale,
                                  std::uint32_t output_width,
                                  std::uint32_t start, std::uint32_t count,
                                  hipStream_t stream) {
  return ExpandConvolution3(Bf16(input), Bf16(output), Int(height), Int(width),
                            Int(channels), Int(stride), Int(padding),
                            Int(upscale), Int(output_width), Int(start),
                            Int(count), stream);
}

void LaunchReorderHeadsFlat(const void* input, void* output, std::uint32_t rows,
                            std::uint32_t heads, std::uint32_t kv_heads,
                            std::uint32_t dim, hipStream_t stream) {
  ReorderHeads(Bf16(input), Bf16(output), Int(rows), Int(heads), Int(kv_heads),
               Int(dim), false, stream);
}

void LaunchRestoreHeadsFlat(const void* input, void* output, std::uint32_t rows,
                            std::uint32_t heads, std::uint32_t dim,
                            hipStream_t stream) {
  ReorderHeads(Bf16(input), Bf16(output), Int(rows), Int(heads), Int(heads),
               Int(dim), true, stream);
}

void LaunchFusedAttention128Flat(const void* queries, const void* keys,
                                 const void* values, void* output,
                                 std::uint32_t heads, std::uint32_t rows,
                                 std::uint32_t count, std::uint32_t query_start,
                                 std::uint32_t query_count,
                                 const std::int32_t* limits,
                                 hipStream_t stream) {
  FusedAttention128(Bf16(queries), Bf16(keys), Bf16(values), Bf16(output),
                    Int(heads), Int(rows), Int(count), Int(query_start), stream,
                    Int(query_count), limits);
}

bool LaunchNormalizeRowsFlat(const void* input, void* output,
                             const void* weight, const void* bias,
                             std::uint32_t rows, std::uint32_t columns,
                             std::uint32_t kind, float epsilon, bool silu,
                             hipStream_t stream) {
  if (kind > static_cast<std::uint32_t>(Norm::kVae) ||
      (kind == static_cast<std::uint32_t>(Norm::kZeroRms) && !weight))
    return false;
  NormalizeRows(Bf16(input), Bf16(output), Bf16(weight), Bf16(bias), Int(rows),
                Int(columns), static_cast<Norm>(kind), epsilon, stream, silu);
  return true;
}

void LaunchNormalizeRotary128Flat(const void* input, void* output,
                                  const void* weight, const float* frequencies,
                                  std::uint32_t rows, std::uint32_t heads,
                                  float epsilon, hipStream_t stream) {
  NormalizeRotary128(Bf16(input), Bf16(output), Bf16(weight), frequencies,
                     Int(rows), Int(heads), epsilon, stream);
}

}  // namespace gufo::models::qwen_image_21::hip
