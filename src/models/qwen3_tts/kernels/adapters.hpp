// Flat forms of Qwen3-TTS launchers whose signatures the Rust launcher
// contract cannot describe directly (docs/RUST.md, "launch.toml"): an
// array of structs, an enum class, and a device array of table pointers. Each
// adapter only forwards; it belongs to the trusted base like the launchers.

#ifndef GUFO_MODELS_QWEN3_TTS_KERNELS_ADAPTERS_HPP_
#define GUFO_MODELS_QWEN3_TTS_KERNELS_ADAPTERS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen3_tts::hip {

/// LaunchBfloat16GemvGrouped over two or three projections given as separate
/// pointers; `weights2`/`output2` are null for two groups.
[[nodiscard]] bool LaunchBfloat16GemvGroupedFlat(
    const void* weights0, float* output0, std::size_t rows0,
    const void* weights1, float* output1, std::size_t rows1,
    const void* weights2, float* output2, std::size_t rows2,
    const void* inputs_bfloat16, std::size_t batch, std::size_t columns,
    hipStream_t stream);

/// LaunchEncoderConv1dIm2Col with the padding mode as its integer value.
void LaunchEncoderConv1dIm2ColFlat(
    const float* input, float* columns, std::size_t input_length,
    std::size_t input_channels, std::size_t output_length, std::size_t kernel,
    std::size_t stride, std::size_t dilation, std::size_t left_padding,
    std::uint32_t padding_mode, hipStream_t stream);

/// LaunchBatchedCodecEmbeddingSum with the table pointers as a host list:
/// they are copied into `device_tables` (one pointer per group, stream
/// ordered) before the launch reads them there.
void LaunchBatchedCodecEmbeddingSumList(
    const void* const* embedding_tables_bfloat16, void* device_tables,
    const std::uint32_t* codes, const float* text_embedding, float* output,
    std::size_t frames, std::size_t groups, std::size_t hidden_size,
    hipStream_t stream);

}  // namespace gufo::models::qwen3_tts::hip

#endif  // GUFO_MODELS_QWEN3_TTS_KERNELS_ADAPTERS_HPP_
