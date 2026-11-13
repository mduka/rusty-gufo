// Flat forms of the Nemotron ASR launchers whose signatures the Rust
// launcher contract cannot describe directly (docs/development/DESIGN.md,
// "launch.toml"): 64-bit counts, the attention's shape struct, the arg
// max's `unsigned long long`, the in-place LayerNorm and the matrix-core
// product's stream before its defaulted bias. Each adapter
// forwards unchanged; the extra extents some take only bound the buffers
// for the generated checks, and the attention declines misaligned rows
// instead of throwing. They belong to the trusted base like the
// launchers themselves. The kernels are the Nemotron ASR extension's
// (../hip, gufo-extensions/nemotron_asr c256b69).

#ifndef GUFO_MODELS_NEMOTRON_ASR_KERNELS_ADAPTERS_HPP_
#define GUFO_MODELS_NEMOTRON_ASR_KERNELS_ADAPTERS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::nemotron_asr::kernels {

/// ToHalf over `n` elements.
void ToHalfFlat(const float* x, std::uint16_t* y, std::size_t n,
                hipStream_t stream);

/// SiluHalf over `n` elements.
void SiluHalfFlat(const float* x, std::uint16_t* y, std::size_t n,
                  hipStream_t stream);

/// LayerNorm with `x` as its own output (each row is read before the
/// thread that reads a column writes it).
void LayerNormInPlace(float* x, const float* weight, const float* bias,
                      int rows, int cols, hipStream_t stream);

/// WmmaLinear with the stream last.
bool WmmaLinearFlat(const std::uint16_t* x, const std::uint16_t* tiles,
                    float* y, std::size_t rows, std::size_t in, std::size_t out,
                    bool accumulate, float scale, const float* bias, bool relu,
                    hipStream_t stream);

/// RelPosAttentionHalf with the AttentionShape fields passed flat; false
/// (nothing launched) unless qkv, the caches and pos are 16-byte aligned,
/// which it throws for. The contract rules out its other throws.
bool RelPosAttentionHalfFlat(const float* qkv, const float* cache_keys,
                             const float* cache_values, const float* pos,
                             const float* bias_u, const float* bias_v,
                             std::uint16_t* out, int queries, int heads,
                             int q_stride, std::size_t first_query,
                             std::size_t first_key, int chunk, int left,
                             int pos_zero, int pos_rows, float scale,
                             hipStream_t stream);

/// UpdateKeyValueCache; `capacity` (the caches' rows) only bounds them.
void UpdateKeyValueCacheFlat(float* keys, float* values, const float* qkv,
                             int stride, int cached, int rows, int keep,
                             int width, int capacity, hipStream_t stream);

/// ArgmaxRows with the packed results as std::uint64_t.
void ArgmaxRowsFlat(const float* logits, const float* bias, int rows,
                    int columns, std::uint64_t* best, hipStream_t stream);

}  // namespace gufo::models::nemotron_asr::kernels

#endif  // GUFO_MODELS_NEMOTRON_ASR_KERNELS_ADAPTERS_HPP_
