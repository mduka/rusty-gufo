#ifndef GUFO_MODELS_VOXCPM_KERNELS_GENERATOR_KERNELS_HPP_
#define GUFO_MODELS_VOXCPM_KERNELS_GENERATOR_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// Device kernels of VoxCPM2's MiniCPM4 stacks and its generation step. The
// residual stream, norms, attention and every elementwise step run in
// float32; GEMM operands are bfloat16 (the checkpoint's weights) and GEMMs
// accumulate in float32.
namespace gufo::models::voxcpm::kernels {

/// Raw bfloat16 bits.
using Bf16 = std::uint16_t;

/// Rows with `text_mask[row]` take embedding[ids[row]] * scale; the others
/// are zero.
void Embed(const Bf16* table, const std::int32_t* ids,
           const std::int32_t* text_mask, float* output, std::size_t rows,
           std::size_t width, float scale, hipStream_t stream);

/// output[indices[i]] = input[i] for rows of `width` floats.
void ScatterRows(const float* input, const std::int32_t* indices, float* output,
                 std::size_t rows, std::size_t width, hipStream_t stream);

/// output[i] = input[indices[i]] for rows of `width` floats.
void GatherRows(const float* input, const std::int32_t* indices, float* output,
                std::size_t rows, std::size_t width, hipStream_t stream);

/// output = bf16(weight * x * rsqrt(mean(x^2) + epsilon)) per row; with
/// `output_f32`, also the float value.
void RmsNorm(const float* input, const Bf16* weight, Bf16* output,
             float* output_f32, std::size_t rows, std::size_t width,
             float epsilon, hipStream_t stream);

/// Splits the fused projection [q heads | k heads | v heads] of each row,
/// rotates q and k with the table's cos/sin at the row's position (no
/// rotation when `cos_sin` is null), writes q as [rows][heads][head_dim]
/// and k, v into the caches [kv_heads][capacity][head_dim] at
/// `cache_rows[row]`. `positions[row]` indexes the table.
void QkvRope(const float* qkv, const float* cos_sin,
             const std::int32_t* positions, const std::int32_t* cache_rows,
             float* q, float* k_cache, float* v_cache, std::size_t rows,
             std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
             std::size_t capacity, hipStream_t stream);

/// Attention of each q row over the cached keys: row r belongs to sequence
/// r / rows_per_sequence, whose keys start at that sequence's
/// `sequence * keys_per_sequence` cache row; it sees `past` earlier keys plus
/// either the sequence's rows up to itself (causal) or all of them. Writes
/// bf16 [rows][heads * head_dim].
void Attention(const float* q, const float* k_cache, const float* v_cache,
               Bf16* output, std::size_t rows, std::size_t rows_per_sequence,
               std::size_t keys_per_sequence, std::size_t past, bool causal,
               std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
               std::size_t capacity, hipStream_t stream);

/// output[row][i] = bf16(silu(gate_up[row][i]) * gate_up[row][width + i]).
void SwiGlu(const float* gate_up, Bf16* output, std::size_t rows,
            std::size_t width, hipStream_t stream);

/// data[row][i] += bias[i].
void AddBias(float* data, const Bf16* bias, std::size_t rows, std::size_t width,
             hipStream_t stream);

/// output = bf16(input), optionally through silu.
void ToBf16(const float* input, Bf16* output, std::size_t count, bool silu,
            hipStream_t stream);

/// Writes bf16 rows [a[row] * a_scale[row] | b[row] * b_scale[row]] of
/// widths a_width + b_width; null scales are 1.
void ConcatRows(const float* a, const float* a_scale, std::size_t a_width,
                const float* b, const float* b_scale, std::size_t b_width,
                Bf16* output, std::size_t rows, hipStream_t stream);

/// Scalar quantization: bf16(round(tanh(x) * scale) / scale), round half
/// to even like torch.round.
void Quantize(const float* input, Bf16* output, std::size_t count, float scale,
              hipStream_t stream);

/// output[row] = mask[row] ? quantized[row] : plain[row] (rows of `width`).
void SelectRows(const float* quantized, const float* plain,
                const std::int32_t* mask, float* output, std::size_t rows,
                std::size_t width, hipStream_t stream);

/// The local encoder's input: per patch, row 0 is `special`, rows 1..P are
/// the projected latent rows `rows_in[patch * P + i]`.
void PrependSpecial(const float* rows_in, const Bf16* special, float* output,
                    std::size_t patches, std::size_t patch_size,
                    std::size_t width, hipStream_t stream);

/// The DiT's input for the CFG pair of sequences of length 3 + 2P:
/// [mu_a, mu_b, time, cond rows, x rows]. The conditional sequence takes
/// `mu` (2 * width), the unconditional one zeros.
void AssembleDit(const float* mu, const float* time, const float* cond,
                 const float* x, float* output, std::size_t patch_size,
                 std::size_t width, hipStream_t stream);

/// CFG-Zero* guidance and the Euler step on one patch: with the conditional
/// and unconditional velocities c, u (`velocity` rows [2][count]),
///   s = <c, u> / (|u|^2 + 1e-8), v = u * s + cfg * (c - u * s),
///   x -= dt * v.
void GuidedEuler(const float* velocity, float* x, std::size_t count, float cfg,
                 float dt, hipStream_t stream);

}  // namespace gufo::models::voxcpm::kernels

#endif  // GUFO_MODELS_VOXCPM_KERNELS_GENERATOR_KERNELS_HPP_
