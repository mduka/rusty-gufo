#ifndef GUFO_MODELS_NEMOTRON_ASR_KERNELS_WMMA_HPP_
#define GUFO_MODELS_NEMOTRON_ASR_KERNELS_WMMA_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// float16 products on the matrix cores (v_wmma_f32_16x16x16_f16): one wave
// per 16 weight rows and a group of activation rows, every output one
// wave's sum over the whole reduction in order. Weights are tiles of 16
// rows by 32 columns, the tiles of 16 rows following each other along the
// columns, then the next 16 rows.
namespace gufo::models::nemotron_asr::kernels {

/// Repacks float16 [rows][in] (rows a multiple of 16, in of 32) into tiles.
void HalfToTiles(const std::uint16_t* weights, std::size_t rows, std::size_t in,
                 std::uint16_t* tiles, hipStream_t stream);

/// y[rows][out] = scale * x[rows][in] W^T, plus y when `accumulate`, plus
/// bias[out] when given, then ReLU when `relu`, for float16 x and tiles W;
/// in a multiple of 32, out of 16. False (nothing launched) for other
/// shapes.
bool WmmaLinear(const std::uint16_t* x, const std::uint16_t* tiles, float* y,
                std::size_t rows, std::size_t in, std::size_t out,
                bool accumulate, float scale, hipStream_t stream,
                const float* bias = nullptr, bool relu = false);

/// WmmaLinear without scale and accumulation, stored as float16.
bool WmmaLinearHalf(const std::uint16_t* x, const std::uint16_t* tiles,
                    std::uint16_t* y, std::size_t rows, std::size_t in,
                    std::size_t out, const float* bias, bool relu,
                    hipStream_t stream);

}  // namespace gufo::models::nemotron_asr::kernels

#endif  // GUFO_MODELS_NEMOTRON_ASR_KERNELS_WMMA_HPP_
