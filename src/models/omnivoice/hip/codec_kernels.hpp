#ifndef GUFO_MODELS_OMNIVOICE_KERNELS_CODEC_KERNELS_HPP_
#define GUFO_MODELS_OMNIVOICE_KERNELS_CODEC_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// Device kernels of the Higgs Audio v2 tokenizer's DAC decoder. Signals are
// channel-major float32 [channels][length] with a row stride.
namespace gufo::models::omnivoice::kernels {

/// output[c][pad + t] = x + sin(alpha[c] * x)^2 / (alpha[c] + 1e-9) for
/// x = input[c][t], with `pad` zeros on both sides of each output row
/// (output row stride length + 2 * pad).
void Snake(const float* input, std::size_t input_stride, const float* alpha,
           float* output, std::size_t channels, std::size_t length,
           std::size_t pad, hipStream_t stream);

/// data[c][t] += bias[c] for t < length, rows `stride` apart.
void AddChannelBias(float* data, std::size_t stride, const float* bias,
                    std::size_t channels, std::size_t length,
                    hipStream_t stream);

/// data[r][j] += bias[j] for rows of `columns`.
void AddColumnBias(float* data, const float* bias, std::size_t rows,
                   std::size_t columns, hipStream_t stream);

/// ConvTranspose1d's overlap-add: columns [kernel][channels][input_length]
/// (the weights times the input for every tap) into
/// output[c][t] = bias[c] + sum of columns[k][c][j] over t = j * stride + k
/// - pad, for t < input_length * stride.
void Col2Im(const float* columns, const float* bias, float* output,
            std::size_t channels, std::size_t input_length, std::size_t stride,
            std::size_t kernel, std::size_t pad, hipStream_t stream);

/// RVQ decode: output[t][h] = sum_c table[c][codes[c][t]][h], with `codes`
/// [codebooks][frames] and `table` [codebooks][entries][width].
void CodebookSum(const float* table, const std::int32_t* codes, float* output,
                 std::size_t codebooks, std::size_t entries, std::size_t frames,
                 std::size_t width, hipStream_t stream);

}  // namespace gufo::models::omnivoice::kernels

#endif  // GUFO_MODELS_OMNIVOICE_KERNELS_CODEC_KERNELS_HPP_
