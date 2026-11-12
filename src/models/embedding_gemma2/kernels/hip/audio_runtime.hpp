// Launchers of the EmbeddingGemma 2 audio tower's kernels
// (audio_runtime.hip). Buffers are float32; the weight of a norm may be null.

#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_KERNELS_HIP_AUDIO_RUNTIME_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_KERNELS_HIP_AUDIO_RUNTIME_HPP_

#include <hip/hip_runtime.h>

namespace gufo::models::embedding_gemma2::hip {

void LaunchAudioNorm(const float* x, const float* weight, float* y, int rows,
                     int width, bool centered, bool clipped,
                     hipStream_t stream);
void LaunchAudioNormInPlace(float* x, const float* weight, int rows, int width,
                            bool centered, bool clipped, hipStream_t stream);
void LaunchAudioActivate(float* x, int count, hipStream_t stream);
void LaunchAudioResidual(float* state, const float* update, int count,
                         float scale, hipStream_t stream);
void LaunchAudioClip(const float* x, float* y, int count, float low, float high,
                     hipStream_t stream);
void LaunchAudioClipInPlace(float* x, int count, float low, float high,
                            hipStream_t stream);
void LaunchAudioConv2d(const float* x, const float* w, float* y, int frames,
                       int bins, int in_channels, int out_channels,
                       int valid_frames, hipStream_t stream);
void LaunchAudioGlu(const float* x, float* y, int rows, hipStream_t stream);
void LaunchAudioDepthwise(const float* x, const float* w, float* y, int rows,
                          hipStream_t stream);
void LaunchAudioAttention(const float* q, const float* k, const float* v,
                          const float* relative, const float* dim_scale,
                          float* output, int rows, int valid,
                          hipStream_t stream);
void LaunchAudioBias(float* x, const float* bias, int rows, hipStream_t stream);

}  // namespace gufo::models::embedding_gemma2::hip

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_KERNELS_HIP_AUDIO_RUNTIME_HPP_
