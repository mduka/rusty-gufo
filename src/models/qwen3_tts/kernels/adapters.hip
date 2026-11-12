#include "src/models/qwen3_tts/kernels/adapters.hpp"

#include <stdexcept>

#include "src/models/qwen3_tts/hip/encoder_ops.hpp"
#include "src/models/qwen3_tts/hip/talker_ops.hpp"

namespace gufo::models::qwen3_tts::hip {

bool LaunchBfloat16GemvGroupedFlat(
    const void* weights0, float* output0, std::size_t rows0,
    const void* weights1, float* output1, std::size_t rows1,
    const void* weights2, float* output2, std::size_t rows2,
    const void* inputs_bfloat16, std::size_t batch, std::size_t columns,
    hipStream_t stream) {
  const Bfloat16GemvGroup groups[kBfloat16GemvMaxGroups] = {
      {weights0, output0, rows0},
      {weights1, output1, rows1},
      {weights2, output2, rows2},
  };
  const std::size_t count = weights2 == nullptr ? 2 : 3;
  return LaunchBfloat16GemvGrouped(groups, count, inputs_bfloat16, batch,
                                   columns, stream);
}

void LaunchEncoderConv1dIm2ColFlat(
    const float* input, float* columns, std::size_t input_length,
    std::size_t input_channels, std::size_t output_length, std::size_t kernel,
    std::size_t stride, std::size_t dilation, std::size_t left_padding,
    std::uint32_t padding_mode, hipStream_t stream) {
  if (padding_mode > static_cast<std::uint32_t>(EncoderPaddingMode::kReflect))
    throw std::invalid_argument("unknown Qwen3-TTS encoder padding mode");
  LaunchEncoderConv1dIm2Col(input, columns, input_length, input_channels,
                            output_length, kernel, stride, dilation,
                            left_padding,
                            static_cast<EncoderPaddingMode>(padding_mode),
                            stream);
}

void LaunchBatchedCodecEmbeddingSumList(
    const void* const* embedding_tables_bfloat16, void* device_tables,
    const std::uint32_t* codes, const float* text_embedding, float* output,
    std::size_t frames, std::size_t groups, std::size_t hidden_size,
    hipStream_t stream) {
  // A pageable copy: the host list may be released once this returns.
  const hipError_t copied =
      hipMemcpyAsync(device_tables, embedding_tables_bfloat16,
                     groups * sizeof(const void*), hipMemcpyHostToDevice,
                     stream);
  if (copied != hipSuccess)
    throw std::runtime_error("cannot copy Qwen3-TTS codec table pointers");
  LaunchBatchedCodecEmbeddingSum(
      reinterpret_cast<const void* const*>(device_tables), codes,
      text_embedding, output, frames, groups, hidden_size, stream);
}

}  // namespace gufo::models::qwen3_tts::hip
