#ifndef GUFO_MODELS_OMNIVOICE_KERNELS_GENERATOR_KERNELS_HPP_
#define GUFO_MODELS_OMNIVOICE_KERNELS_GENERATOR_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// Device kernels of OmniVoice's bidirectional Qwen3 generator. Rows are
// token positions of the packed CFG batch: the conditional sequence's rows
// first, then the unconditional sequence's.
namespace gufo::models::omnivoice::kernels {

/// GEMM operand type: float16 like the reference's GPU inference, with
/// float32 accumulation, norms and residual stream.
using Half = __half;

/// Rows below `text_rows` copy `text` (one row each); the others sum the
/// audio embeddings of their codes: table[c * vocab + ids[c * rows + row]].
void Embed(const float* text, std::size_t text_rows, const float* table,
           const std::int32_t* ids, float* hidden, std::size_t rows,
           std::size_t width, std::size_t codebooks, std::size_t vocab,
           hipStream_t stream);

/// output = half(weight * x * rsqrt(mean(x^2) + epsilon)) per row.
void RmsNorm(const float* input, const float* weight, Half* output,
             std::size_t rows, std::size_t width, float epsilon,
             hipStream_t stream);

/// Splits the fused projection [q heads | k heads | v heads] per row,
/// applies the per-head RMSNorm and the rotary embedding at `positions[row]`
/// to q and k, and writes head-major half [heads][rows][head_dim] q, k and v.
/// k and v are repeated for each query head of their group.
void QkNormRope(const float* qkv, const float* q_norm, const float* k_norm,
                const std::int32_t* positions, Half* q, Half* k, Half* v,
                std::size_t rows, std::size_t heads, std::size_t kv_heads,
                std::size_t head_dim, float epsilon, float rope_theta,
                hipStream_t stream);

/// output[row] = half(softmax(scores[row] * scale)) for rows of `columns`.
void Softmax(const float* scores, Half* output, std::size_t rows,
             std::size_t columns, float scale, hipStream_t stream);

/// Head-major float [heads][rows][head_dim] to row-major half
/// [rows][heads * head_dim].
void MergeHeads(const float* input, Half* output, std::size_t rows,
                std::size_t heads, std::size_t head_dim, hipStream_t stream);

/// output[row][i] = half(silu(gate_up[row][i]) * gate_up[row][width + i]).
void SwiGlu(const float* gate_up, Half* output, std::size_t rows,
            std::size_t width, hipStream_t stream);

/// output[i] = input[indices[i]] for rows of `width` floats.
void GatherRows(const float* input, const std::int32_t* indices, float* output,
                std::size_t rows, std::size_t width, hipStream_t stream);

/// The unmasking step's scoring for `frames` target frames. logits rows
/// [0, frames) are conditional, [frames, 2 * frames) unconditional; each row
/// holds `codebooks` blocks of `vocab` logits. With guidance g:
///   lp = log_softmax(c + g * (c - u)) of the log-softmaxed rows c and u,
/// then the mask entry is excluded; writes argmax and max of lp to
/// prediction / confidence [codebook][frame].
void Score(const float* logits, std::int32_t* prediction, float* confidence,
           std::size_t frames, std::size_t codebooks, std::size_t vocab,
           std::uint32_t mask_id, float guidance, hipStream_t stream);

}  // namespace gufo::models::omnivoice::kernels

#endif  // GUFO_MODELS_OMNIVOICE_KERNELS_GENERATOR_KERNELS_HPP_
