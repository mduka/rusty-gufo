// The EmbeddingGemma 2 audio tower's kernels, verbatim from the C++
// audio_runtime.hip, whose host code (AudioRuntime) the Rust port carries
// (src/audio.rs). The launchers below are the port's: each issues one
// kernel with the grid the C++ host code uses.
#include "src/models/embedding_gemma2/kernels/hip/audio_runtime.hpp"

#include <cmath>

namespace gufo::models::embedding_gemma2::hip {
namespace {
__device__ float Sum(float v) {
  for (int d = 16; d; d >>= 1)
    v += __shfl_xor(v, d, 32);
  return v;
}
// One wave per row. LayerNorm is used only by the two subsampling blocks;
// all other norms use RMS and the checkpoint scale directly (no +1).
__global__ void Norm(const float* x, const float* weight, float* y, int width,
                     bool centered, bool clipped = false) {
  const int row = blockIdx.x, lane = threadIdx.x;
  float mean = 0, variance = 0;
  if (centered) {
    for (int c = lane; c < width; c += 32)
      mean += x[row * width + c];
    mean = Sum(mean) / width;
  }
  for (int c = lane; c < width; c += 32) {
    const float raw = x[row * width + c];
    const float v = (clipped ? fminf(fmaxf(raw, -1e10F), 1e10F) : raw) - mean;
    variance += v * v;
  }
  const float scale = rsqrtf(Sum(variance) / width + 1e-6F);
  for (int c = lane; c < width; c += 32) {
    const float raw = x[row * width + c];
    float v =
        ((clipped ? fminf(fmaxf(raw, -1e10F), 1e10F) : raw) - mean) * scale;
    if (weight)
      v *= weight[c];
    y[row * width + c] = centered ? fmaxf(v, 0.0F) : v;
  }
}
__global__ void Activate(float* x, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] = x[i] / (1 + expf(-x[i]));
}
__global__ void Residual(float* state, const float* update, int count,
                         float scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    state[i] += update[i] * scale;
}
__global__ void Clip(const float* x, float* y, int count, float low,
                     float high) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    y[i] = fminf(fmaxf(x[i], low), high);
}
// Subsample convolution in channels-last layout. Mask gates each input time
// row; layer1 uses the already subsampled [::2] input mask.
__global__ void Conv2d(const float* x, const float* w, float* y, int frames,
                       int bins, int in_channels, int out_channels,
                       int valid_frames) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_bins = (bins + 1) / 2, out_frames = (frames + 1) / 2;
  if (i >= out_frames * out_bins * out_channels)
    return;
  const int oc = i % out_channels;
  const int f = (i / out_channels) % out_bins, t = i / out_channels / out_bins;
  float sum = 0;
  for (int ic = 0; ic < in_channels; ++ic) {
    for (int dt = 0; dt < 3; ++dt) {
      const int time = t * 2 + dt - 1;
      if (time < 0 || time >= frames || time >= valid_frames)
        continue;
      for (int df = 0; df < 3; ++df) {
        const int bin = f * 2 + df - 1;
        if (bin >= 0 && bin < bins)
          sum += x[(time * bins + bin) * in_channels + ic] *
                 w[((oc * in_channels + ic) * 3 + dt) * 3 + df];
      }
    }
  }
  y[i] = sum;
}
__global__ void Glu(const float* x, float* y, int rows) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * 1024) {
    const int base = (i / 1024) * 2048 + i % 1024;
    y[i] = x[base] / (1 + expf(-x[base + 1024]));
  }
}
__global__ void Depthwise(const float* x, const float* w, float* y, int rows) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * 1024)
    return;
  const int t = i / 1024, c = i % 1024;
  float sum = 0;
  for (int tap = 0; tap < 5; ++tap)
    if (t + tap >= 4)
      sum += x[(t + tap - 4) * 1024 + c] * w[c * 5 + tap];
  y[i] = sum;
}
__global__ void Attention(const float* q, const float* k, const float* v,
                          const float* relative, const float* dim_scale,
                          float* output, int rows, int valid) {
  const int t = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
  const int chunk_start = (t / 12) * 12;
  float scores[24];
  float maximum = -1e9F;
  const float qscale = (1 / sqrtf(128.0F)) / logf(2.0F);
  const float kscale = logf(1 + expf(1.0F)) / logf(2.0F);
  float query[4];
  for (int j = 0; j < 4; ++j) {
    const int d = lane + j * 32;
    query[j] =
        q[t * 1024 + head * 128 + d] * qscale * log1pf(expf(dim_scale[d]));
  }
  for (int context = 0; context < 24; ++context) {
    const int key_t = chunk_start - 12 + context;
    float dot = 0;
    // _rel_shift flattens [12,25] then reshapes [12,24]. For
    // query position r the relative index is context-r. Only 0..12
    // survives padding; the causal 12-left mask ensures exactly that range.
    const int rel = context - t % 12;
    if (key_t >= 0 && key_t < rows) {
      for (int j = 0; j < 4; ++j) {
        const int d = head * 128 + lane + j * 32;
        dot += query[j] * k[key_t * 1024 + d] * kscale;
      }
    }
    if (rel >= 0 && rel <= 12) {
      for (int j = 0; j < 4; ++j)
        dot += query[j] * relative[rel * 1024 + head * 128 + lane + j * 32];
    }
    dot = Sum(dot);
    const bool allowed =
        key_t >= 0 && key_t < valid && key_t <= t && key_t > t - 12;
    scores[context] = allowed ? 50 * tanhf(dot / 50) : -1e9F;
    maximum = fmaxf(maximum, scores[context]);
  }
  float denom = 0;
  for (int j = 0; j < 24; ++j) {
    scores[j] = expf(scores[j] - maximum);
    denom += scores[j];
  }
  for (int j = 0; j < 4; ++j) {
    float sum = 0;
    for (int context = 0; context < 24; ++context) {
      const int key_t = chunk_start - 12 + context;
      if (key_t >= 0 && key_t < rows)
        sum += scores[context] / denom *
               v[key_t * 1024 + head * 128 + lane + j * 32];
    }
    output[t * 1024 + head * 128 + lane + j * 32] = sum;
  }
}
__global__ void Bias(float* x, const float* bias, int rows) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * 1536)
    x[i] += bias[i % 1536];
}
}  // namespace

namespace {
unsigned int Blocks(int count) {
  return static_cast<unsigned int>((count + 255) / 256);
}
}  // namespace

void LaunchAudioNorm(const float* x, const float* weight, float* y, int rows,
                     int width, bool centered, bool clipped,
                     hipStream_t stream) {
  Norm<<<rows, 32, 0, stream>>>(x, weight, y, width, centered, clipped);
}

void LaunchAudioNormInPlace(float* x, const float* weight, int rows,
                            int width, bool centered, bool clipped,
                            hipStream_t stream) {
  Norm<<<rows, 32, 0, stream>>>(x, weight, x, width, centered, clipped);
}

void LaunchAudioActivate(float* x, int count, hipStream_t stream) {
  Activate<<<Blocks(count), 256, 0, stream>>>(x, count);
}

void LaunchAudioResidual(float* state, const float* update, int count,
                         float scale, hipStream_t stream) {
  Residual<<<Blocks(count), 256, 0, stream>>>(state, update, count, scale);
}

void LaunchAudioClip(const float* x, float* y, int count, float low,
                     float high, hipStream_t stream) {
  Clip<<<Blocks(count), 256, 0, stream>>>(x, y, count, low, high);
}

void LaunchAudioClipInPlace(float* x, int count, float low, float high,
                            hipStream_t stream) {
  Clip<<<Blocks(count), 256, 0, stream>>>(x, x, count, low, high);
}

void LaunchAudioConv2d(const float* x, const float* w, float* y, int frames,
                       int bins, int in_channels, int out_channels,
                       int valid_frames, hipStream_t stream) {
  const int outputs = (frames + 1) / 2 * ((bins + 1) / 2) * out_channels;
  Conv2d<<<Blocks(outputs), 256, 0, stream>>>(
      x, w, y, frames, bins, in_channels, out_channels, valid_frames);
}

void LaunchAudioGlu(const float* x, float* y, int rows, hipStream_t stream) {
  Glu<<<Blocks(rows * 1024), 256, 0, stream>>>(x, y, rows);
}

void LaunchAudioDepthwise(const float* x, const float* w, float* y, int rows,
                          hipStream_t stream) {
  Depthwise<<<Blocks(rows * 1024), 256, 0, stream>>>(x, w, y, rows);
}

void LaunchAudioAttention(const float* q, const float* k, const float* v,
                          const float* relative, const float* dim_scale,
                          float* output, int rows, int valid,
                          hipStream_t stream) {
  Attention<<<dim3(rows, 8), 32, 0, stream>>>(q, k, v, relative, dim_scale,
                                               output, rows, valid);
}

void LaunchAudioBias(float* x, const float* bias, int rows,
                     hipStream_t stream) {
  Bias<<<Blocks(rows * 1536), 256, 0, stream>>>(x, bias, rows);
}

}  // namespace gufo::models::embedding_gemma2::hip
