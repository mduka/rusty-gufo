#ifndef GUFO_MODELS_QWEN_IMAGE_21_HIP_OPS_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_HIP_OPS_HPP_

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen_image_21::hip {

enum class Norm { kLayer, kRms, kZeroRms, kVae };
enum class Activation { kSilu, kGeluTanh, kGelu, kIdentity };

// Model-private operators; callers supply validated dimensions and buffers.
bool DenseShape(int channels, int inner);
// Columns must be a multiple of 16.
void PackDense16(const std::uint16_t* input, std::uint16_t* output, int rows,
                 int cols, hipStream_t stream);
bool DenseBf16(const std::uint16_t* weights, const std::uint16_t* input,
               std::uint16_t* output, int rows, int channels, int inner,
               hipStream_t stream);
// All inputs and the output use PackDense16 layout.
bool DenseSwiGlu(const std::uint16_t* gate, const std::uint16_t* up,
                 const std::uint16_t* input, std::uint16_t* output, int rows,
                 int channels, int inner, hipStream_t stream);
bool Convolve3Shape(int channels, int output_channels);
// Weights use PackDense16 layout.
bool Convolve3(const std::uint16_t* input, const std::uint16_t* weights,
               const std::uint16_t* bias, std::uint16_t* output, int height,
               int width, int channels, int output_channels, hipStream_t stream,
               int stride = 1, int padding = 1, int upscale = 1);
bool ExpandConvolution3(const std::uint16_t* input, std::uint16_t* output,
                        int height, int width, int channels, int stride,
                        int padding, int upscale, int output_width, int start,
                        int count, hipStream_t stream);
void ReorderHeads(const std::uint16_t* input, std::uint16_t* output, int rows,
                  int heads, int kv_heads, int dim, bool inverse,
                  hipStream_t stream);
void FusedAttention128(const std::uint16_t* queries, const std::uint16_t* keys,
                       const std::uint16_t* values, std::uint16_t* output,
                       int heads, int rows, int count, int query_start,
                       hipStream_t stream, int query_count = 128,
                       const int* limits = nullptr);
void NormalizeRows(const std::uint16_t* input, std::uint16_t* output,
                   const std::uint16_t* weight, const std::uint16_t* bias,
                   int rows, int cols, Norm kind, float epsilon,
                   hipStream_t stream, bool silu = false);
void NormalizeRotary128(const std::uint16_t* input, std::uint16_t* output,
                        const std::uint16_t* weight, const float* frequencies,
                        int rows, int heads, float epsilon, hipStream_t stream);

// Launchers of the element-wise kernels in kernels.hip, one per operation of
// the runtime. BF16 tensors travel as void*; dimensions fit in int.
void LaunchFloatToBf16(const float* input, void* output, std::size_t count,
                       hipStream_t stream);
void LaunchBf16ToFloat(const void* input, float* output, std::size_t count,
                       hipStream_t stream);
// Adds a BF16 bias to an FP32 product; convolutions round the product first.
void LaunchAddBias(const float* input, const void* bias, void* output,
                   std::uint32_t columns, std::size_t count, bool convolution,
                   hipStream_t stream);
void LaunchActivate(const void* input, void* output, std::size_t count,
                    std::uint32_t activation, hipStream_t stream);
void LaunchSwiGlu(const void* gate, const void* up, void* output,
                  std::size_t count, hipStream_t stream);
void LaunchAdd(const void* left, const void* right, void* output,
               std::size_t count, hipStream_t stream);
void LaunchEuler(const void* sample, const void* velocity, void* output,
                 std::size_t count, float dt, hipStream_t stream);
void LaunchPrepareModulation(const void* input, void* output,
                             std::uint32_t count, bool gates,
                             hipStream_t stream);
// LayerNorm of 4096-column rows followed by the modulation scale; rows before
// `prefix` use the second modulation row.
void LaunchNormalizeModulate(const void* input, const void* modulation,
                             void* output, std::uint32_t rows,
                             std::uint32_t modulation_columns,
                             std::uint32_t part, std::uint32_t prefix,
                             hipStream_t stream);
void LaunchModulate(const void* input, const void* modulation,
                    const void* residual, void* output, std::uint32_t rows,
                    std::uint32_t columns, std::uint32_t modulation_columns,
                    std::uint32_t part, std::uint32_t prefix,
                    hipStream_t stream);
void LaunchRotary(const void* input, void* output, const float* frequencies,
                  std::uint32_t rows, std::uint32_t heads, std::uint32_t dim,
                  std::uint32_t mode, hipStream_t stream);
// Scaled softmax of `heads * queries` score rows of `keys` columns; query
// row r sees limits[query_start + r % queries] keys when limits are given.
void LaunchSoftmax(const float* scores, float* probabilities,
                   std::uint32_t heads, std::uint32_t queries,
                   std::uint32_t keys, std::uint32_t query_start,
                   const std::int32_t* limits, float scale, hipStream_t stream);
void LaunchCopyColumns(const void* input, void* output, std::uint32_t rows,
                       std::uint32_t input_columns, std::uint32_t start,
                       std::uint32_t columns, hipStream_t stream);
void LaunchEmbedding(const void* table, void* output, const std::uint32_t* ids,
                     std::uint32_t count, std::uint32_t columns,
                     hipStream_t stream);
void LaunchScatterRows(const void* values, void* output,
                       const std::int32_t* positions, std::uint32_t count,
                       std::uint32_t columns, bool add, hipStream_t stream);
void LaunchIm2col(const void* input, void* output, std::uint32_t height,
                  std::uint32_t width, std::uint32_t channels,
                  std::uint32_t kernel, std::uint32_t stride,
                  std::uint32_t padding, std::uint32_t upscale,
                  std::uint32_t output_width, std::uint32_t start,
                  std::uint32_t count, hipStream_t stream);
void LaunchDownShortcut(const void* input, void* output, std::uint32_t height,
                        std::uint32_t width, std::uint32_t input_channels,
                        std::uint32_t output_channels, std::uint32_t temporal,
                        std::uint32_t spatial, hipStream_t stream);
void LaunchUpShortcut(const void* input, void* output, std::uint32_t height,
                      std::uint32_t width, std::uint32_t input_channels,
                      std::uint32_t output_channels, std::uint32_t temporal,
                      hipStream_t stream);
void LaunchScaleLatents(const void* input, void* output, const void* mean,
                        const void* deviation, std::size_t count, bool encode,
                        hipStream_t stream);
void LaunchVisionPosition(const void* input, const void* table, void* output,
                          std::uint32_t height, std::uint32_t width,
                          hipStream_t stream);

}  // namespace gufo::models::qwen_image_21::hip
#endif
