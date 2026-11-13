#ifndef GUFO_MODELS_NEMOTRON_ASR_KERNELS_HPP_
#define GUFO_MODELS_NEMOTRON_ASR_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

// Kernels of the encoder and the transducer, row-major: float32 activations
// with float16 copies (std::uint16_t storage) as operands of the matrix
// products; the subsampling keeps float16 [time][frequency][channel].
namespace gufo::models::nemotron_asr::kernels {

/// The stem: a 3x3 stride-2 convolution of the single-channel mel
/// spectrogram [rows][bins] (frequency padded by 2 before and 1 after),
/// plus bias, then ReLU: out [out_rows][out_freq][channels] as float16.
void SubsampleInput(const float* mel, int bins, const float* weight,
                    const float* bias, int channels, std::uint16_t* out,
                    int out_rows, int out_freq, hipStream_t stream);

/// A depthwise 3x3 stride-2 convolution of float16 [rows][in_freq][channels]
/// plus bias (frequency padded like the stem), summed in float32: out
/// [out_rows][out_freq][channels] as float16; channels even.
void SubsampleDepthwise(const std::uint16_t* in, int in_freq,
                        const float* weight, const float* bias, int channels,
                        std::uint16_t* out, int out_rows, int out_freq,
                        hipStream_t stream);

/// y = LayerNorm(x) * weight + bias over rows of `cols` (eps 1e-5).
void LayerNorm(const float* x, const float* weight, const float* bias, float* y,
               int rows, int cols, hipStream_t stream);

/// LayerNorm as float16, followed by silu when `silu`.
void LayerNormHalf(const float* x, const float* weight, const float* bias,
                   std::uint16_t* y, int rows, int cols, bool silu,
                   hipStream_t stream);

/// y = silu(x) as float16 over n elements.
void SiluHalf(const float* x, std::uint16_t* y, std::int64_t n,
              hipStream_t stream);

/// y = x as float16 (round to nearest even) over n elements.
void ToHalf(const float* x, std::uint16_t* y, std::int64_t n,
            hipStream_t stream);

struct AttentionShape {
  int queries = 0;               // rows of q and out
  int heads = 0;                 // of 128 dimensions each
  int q_stride = 0;              // floats between query rows
  std::int64_t first_query = 0;  // absolute frame of query row 0
  std::int64_t first_key = 0;    // absolute frame of cache row 0
  int chunk = 0;                 // frames per attention chunk
  int left = 0;                  // frames attended before a query's chunk
  int pos_zero = 0;              // row of `pos` for distance 0
  int pos_rows = 0;              // rows of `pos`
  float scale = 0.0F;
};

/// Frames a query attends to at most (left context plus chunk).
inline constexpr int kMaxKeys = 256;

/// Transformer-XL attention with chunked limited context: a query attends
/// to the keys from `left` frames before its chunk to the chunk's end, with
/// scores ((q + u) . k + (q + v) . pos[query - key]) * scale. The pass's
/// queries, keys and values are `qkv` rows [q | k | v] (q_stride apart),
/// earlier keys and values the caches' rows (frames first_key to
/// first_query - 1). The products take float16 operands with float32 sums,
/// the softmax is float32; the output is float16. left + chunk at most
/// kMaxKeys. `pos` holds at least the distances of a query's window,
/// -(chunk - 1) to left + chunk - 1, at rows pos_zero + distance; no row
/// outside it is read. Rows are read as float4: qkv, the caches and pos
/// 16-byte aligned, q_stride a multiple of 4.
void RelPosAttentionHalf(const float* qkv, const float* cache_keys,
                         const float* cache_values, const float* pos,
                         const float* bias_u, const float* bias_v,
                         std::uint16_t* out, const AttentionShape& shape,
                         hipStream_t stream);

/// Keeps the last `keep` key and value rows of [cache (`cached` rows) |
/// the pass's `rows` qkv rows] in the caches.
void UpdateKeyValueCache(float* keys, float* values, const float* qkv,
                         int stride, int cached, int rows, int keep, int width,
                         hipStream_t stream);

/// The convolution module's GLU and causal depthwise convolution: `in` is
/// [rows][2 * channels], `cache` the GLU of the kernel - 1 rows before it.
void GluDepthwise(const float* in, const float* cache, const float* weight,
                  float* out, int rows, int channels, int kernel,
                  hipStream_t stream);

/// Replaces `cache` with the GLU of the last kernel - 1 rows of [cache | in].
void UpdateConvCache(const float* in, float* cache, int rows, int channels,
                     int kernel, hipStream_t stream);

/// y[r] = W[r] . concat(x1, x2) + bias[r] for `rows` float16 rows of
/// k1 + k2 values (k1 and k2 multiples of 8).
void Gemv(const std::uint16_t* weight, const float* x1, int k1, const float* x2,
          int k2, const float* bias, float* y, int rows, hipStream_t stream);

/// PyTorch's LSTM cell from gates [i | f | g | o] (each `hidden` wide).
void LstmCell(const float* gates, float* h, float* c, int hidden,
              hipStream_t stream);

/// Frames the joint takes per weight pass at most.
inline constexpr int kMaxJointFrames = 16;

/// The joint's input: out[f] = relu(encoder[f] + decoder) as float16 for
/// `frames` rows of `width`.
void JointInput(const float* encoder, const float* decoder, std::uint16_t* out,
                int frames, int width, hipStream_t stream);

/// best[r] = the packed arg max of logits[r] + bias over `columns`.
void ArgmaxRows(const float* logits, const float* bias, int rows, int columns,
                unsigned long long* best, hipStream_t stream);

/// The token of a packed arg max.
[[nodiscard]] inline std::uint32_t ArgmaxToken(unsigned long long packed) {
  return 0xFFFFFFFFU - static_cast<std::uint32_t>(packed & 0xFFFFFFFFULL);
}

}  // namespace gufo::models::nemotron_asr::kernels

#endif  // GUFO_MODELS_NEMOTRON_ASR_KERNELS_HPP_
