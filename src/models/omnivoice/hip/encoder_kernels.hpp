#ifndef GUFO_MODELS_OMNIVOICE_KERNELS_ENCODER_KERNELS_HPP_
#define GUFO_MODELS_OMNIVOICE_KERNELS_ENCODER_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// Device kernels of the Higgs Audio v2 tokenizer's encoder (HuBERT, the
// semantic and DAC encoders and the RVQ), all float32.
namespace gufo::models::omnivoice::kernels {

/// columns[(c * kernel + k)][t] = input[c][t * stride + k * dilation - pad]
/// (zero outside the input) for t < output_length.
void Im2Col(const float* input, std::size_t channels, std::size_t length,
            std::size_t kernel, std::size_t stride, std::size_t dilation,
            std::size_t pad, std::size_t output_length, float* columns,
            hipStream_t stream);

/// Per channel of [channels][length]: normalize over time (GroupNorm with
/// one group per channel), scale, shift and apply the exact GELU.
void GroupNormGelu(float* data, const float* weight, const float* bias,
                   std::size_t channels, std::size_t length, float epsilon,
                   hipStream_t stream);

/// Exact (erf) GELU in place.
void Gelu(float* data, std::size_t count, hipStream_t stream);

/// ELU (alpha 1) from input to output; may alias.
void Elu(const float* input, float* output, std::size_t count,
         hipStream_t stream);

/// output[c][r] = input[r][c] for a [rows][columns] input; the output row
/// stride is `output_stride`.
void Transpose(const float* input, std::size_t rows, std::size_t columns,
               float* output, std::size_t output_stride, hipStream_t stream);

/// LayerNorm over rows of `width`; output may alias input.
void LayerNorm(const float* input, const float* weight, const float* bias,
               float* output, std::size_t rows, std::size_t width,
               float epsilon, hipStream_t stream);

/// Row-wise softmax of scores * scale in place.
void SoftmaxInPlace(float* scores, std::size_t rows, std::size_t columns,
                    float scale, hipStream_t stream);

/// target[i] += source[i].
void Accumulate(const float* source, float* target, std::size_t count,
                hipStream_t stream);

/// data[i] *= scale.
void Scale(float* data, std::size_t count, float scale, hipStream_t stream);

/// Nearest codebook entry per row by the reference's distance
/// -((|h|^2 - 2 h.e) + |e|^2): `dots` [rows][entries] holds h.e,
/// `projected` [rows][dim] holds h and `norms` [entries] |e|^2.
void NearestEntry(const float* dots, const float* projected, const float* norms,
                  std::int32_t* indices, std::size_t rows, std::size_t entries,
                  std::size_t dim, hipStream_t stream);

/// residual[r] -= table[indices[r]] for rows of `width`.
void SubtractEntries(float* residual, const float* table,
                     const std::int32_t* indices, std::size_t rows,
                     std::size_t width, hipStream_t stream);

/// norms[e] = sum of entries[e]^2 over `dim`.
void SquaredNorms(const float* entries, float* norms, std::size_t count,
                  std::size_t dim, hipStream_t stream);

}  // namespace gufo::models::omnivoice::kernels

#endif  // GUFO_MODELS_OMNIVOICE_KERNELS_ENCODER_KERNELS_HPP_
