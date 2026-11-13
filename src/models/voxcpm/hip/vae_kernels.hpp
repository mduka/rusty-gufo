#ifndef GUFO_MODELS_VOXCPM_KERNELS_VAE_KERNELS_HPP_
#define GUFO_MODELS_VOXCPM_KERNELS_VAE_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>

// Device kernels of AudioVAE V2 in float32. Activations are time-major
// [frames][channels]; a causal convolution reads `history` frames before
// the first output frame (the left padding or the streaming state).
namespace gufo::models::voxcpm::kernels {

/// output = y + sin(alpha * y)^2 / (alpha + 1e-9) with
/// y = input * scale + bias (scale and bias per channel, null for 1 and 0).
void Snake(const float* input, float* output, std::size_t frames,
           std::size_t channels, const float* alpha, const float* scale,
           const float* bias, hipStream_t stream);

/// Depthwise causal convolution: output[t][c] = bias[c] +
/// sum_j weight[c][j] * input[t + j * dilation][c], where `input` holds
/// (kernel - 1) * dilation history frames before the `frames` new ones.
void DepthwiseConv(const float* input, const float* weight, const float* bias,
                   float* output, std::size_t frames, std::size_t channels,
                   std::size_t kernel, std::size_t dilation,
                   hipStream_t stream);

/// One input channel to `channels`: output[t][c] = bias[c] +
/// sum_j weight[c][j] * samples[t + j] (kernel - 1 history samples).
void ConvFromMono(const float* samples, const float* weight, const float* bias,
                  float* output, std::size_t frames, std::size_t channels,
                  std::size_t kernel, hipStream_t stream);

/// `channels` to one output channel through tanh: output[t] =
/// tanh(bias + sum_c sum_j weight[c][j] * input[t + j][c]).
void ConvToMono(const float* input, const float* weight, float bias,
                float* output, std::size_t frames, std::size_t channels,
                std::size_t kernel, hipStream_t stream);

/// data[row][i] += bias[i].
void AddBiasF32(float* data, const float* bias, std::size_t rows,
                std::size_t width, hipStream_t stream);

}  // namespace gufo::models::voxcpm::kernels

#endif  // GUFO_MODELS_VOXCPM_KERNELS_VAE_KERNELS_HPP_
