// Definitions of launch.hpp; included last in backend.hip.cpp, where the
// kernels are visible.
#pragma once

#include "launch.hpp"

namespace gufo::models::deepseek_v4_flash {

void RmsNormPlainRows(float* out, const float* x, uint32_t n, uint32_t rows,
                      float eps, hipStream_t stream) {
  if (n == 0u || rows == 0u) return;
  // The hot caller is the 4-way hyper-connection row, 16,384 floats wide.
  if (n == 256u * 64u) {
    rms_norm_plain_regs_kernel<64u><<<rows, 256, 0, stream>>>(out, x, n, rows,
                                                               eps);
    return;
  }
  rms_norm_plain_kernel<<<rows, 256, 0, stream>>>(out, x, n, rows, eps);
}

void RmsNormWeightRows(float* out, const float* x, const float* w, uint32_t n,
                       uint32_t rows, float eps, hipStream_t stream) {
  if (n == 0u || rows == 0u) return;
  rms_norm_weight_kernel<<<rows, 256, 0, stream>>>(out, x, w, n, rows, eps);
}

void QkvRmsNormRows(float* q_out, const float* q, const float* q_w,
                    uint32_t q_n, float* kv_out, const float* kv,
                    const float* kv_w, uint32_t kv_n, uint32_t rows, float eps,
                    hipStream_t stream) {
  if ((q_n == 0u && kv_n == 0u) || rows == 0u) return;
  dim3 grid(rows, 2u, 1u);
  dsv4_qkv_rms_norm_rows_kernel<<<grid, 256, 0, stream>>>(
      q_out, q, q_w, q_n, kv_out, kv, kv_w, kv_n, rows, eps);
}

void HeadRmsNorm(float* x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
                 float eps, hipStream_t stream) {
  const uint32_t rows = n_tok * n_head;
  if (rows == 0u || head_dim == 0u) return;
  head_rms_norm_kernel<<<rows, 256, 0, stream>>>(x, n_tok, n_head, head_dim,
                                                 eps);
}

bool HeadRmsNormRopeTail(float* x, uint32_t n_tok, uint32_t n_head,
                         uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
                         uint32_t n_ctx_orig, bool inverse, float freq_base,
                         float freq_scale, float ext_factor, float attn_factor,
                         float beta_fast, float beta_slow, float eps,
                         hipStream_t stream) {
  const uint32_t rows = n_tok * n_head;
  if (rows == 0u) return true;
  // Restrict fusion to the qualified wide-prefill shapes.
  if (n_tok < DS4_ROCM_ATTENTION_WIDE_ROWS) return false;
  head_rms_norm_rope_tail_lds_kernel<<<rows, 256, 0, stream>>>(
      x, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse ? 1 : 0,
      freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
      eps);
  return true;
}

void RopeTail(float* x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
              uint32_t n_rot, uint32_t pos0, uint32_t pos_stride,
              uint32_t n_ctx_orig, bool inverse,
              float freq_base, float freq_scale, float ext_factor,
              float attn_factor, float beta_fast, float beta_slow,
              hipStream_t stream) {
  const uint32_t pairs = n_tok * n_head * (n_rot / 2);
  rope_tail_kernel<<<(pairs + 255) / 256, 256, 0, stream>>>(
      x, n_tok, n_head, head_dim, n_rot, pos0, pos_stride, n_ctx_orig,
      inverse ? 1 : 0,
      freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}

void HcWeightedSum(float* out, const float* residual_hc, const float* weights,
                   uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens,
                   hipStream_t stream) {
  hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256, 0,
                           stream>>>(out, residual_hc, weights, n_embd, n_hc,
                                     n_tokens, n_hc);
}

void HcSplitWeightedSum(float* out, float* split, const float* mix,
                        const float* residual_hc, const float* scale,
                        const float* base, uint32_t n_embd, uint32_t n_hc,
                        uint32_t n_rows, uint32_t sinkhorn_iters, float eps,
                        hipStream_t stream) {
  hc_split_weighted_sum_fused_kernel<<<n_rows, 256, 0, stream>>>(
      out, split, mix, residual_hc, scale, base, n_embd, n_hc, n_rows,
      sinkhorn_iters, eps);
}

bool HcNormMixSplitWeightedSum(float* out, float* mix, float* split,
                               const float* residual_hc, const __half* mix_w,
                               const float* scale, const float* base,
                               uint32_t n_embd, uint32_t n_rows,
                               uint32_t sinkhorn_iters, float eps,
                               float norm_eps, hipStream_t stream) {
  // Four tokens per workgroup, 256 lanes each, sixteen columns per lane (see
  // ds4_gpu_hc_norm_mix_split_weighted_sum_tensor).
  constexpr uint32_t TOK = 4u, TPT = 256u, COLS = 16u;
  if (n_embd != TPT * COLS) return false;
  // The weight is read eight halves at a time and the row in float4.
  if (((uintptr_t)mix_w & 15u) != 0u || ((uintptr_t)residual_hc & 15u) != 0u ||
      ((uintptr_t)out & 15u) != 0u) {
    return false;
  }
  hc4_norm_mix_split_weighted_sum_kernel<TOK, TPT, COLS>
      <<<(n_rows + TOK - 1u) / TOK, TOK * TPT, 0, stream>>>(
          out, mix, split, residual_hc, mix_w, scale, base, n_embd, n_rows,
          sinkhorn_iters, eps, norm_eps);
  return true;
}

void HcSplitWeightedSumNorm(float* out, float* norm_out, float* split,
                            const float* mix, const float* residual_hc,
                            const float* scale, const float* base,
                            const float* norm_w, uint32_t n_embd,
                            uint32_t n_hc, uint32_t n_rows,
                            uint32_t sinkhorn_iters, float eps, float norm_eps,
                            hipStream_t stream) {
  hc_split_weighted_sum_norm_fused_kernel<<<n_rows, 256, 0, stream>>>(
      out, norm_out, split, mix, residual_hc, scale, base, norm_w, n_embd,
      n_hc, n_rows, sinkhorn_iters, eps, norm_eps);
}

void OutputHcWeights(float* out, const float* pre, const float* scale,
                     const float* base, uint32_t n_hc, uint32_t n_tokens,
                     float eps, hipStream_t stream) {
  const uint64_t n = (uint64_t)n_tokens * n_hc;
  output_hc_weights_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      out, pre, scale, base, n_hc, n_tokens, eps);
}

void HcExpandSplit(float* out_hc, const float* block_out,
                   const float* residual_hc, const float* split,
                   uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens,
                   hipStream_t stream) {
  if (n_hc == 4u) {
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    if (hip_hc_vec4_ok(n_embd, out_hc, block_out, residual_hc)) {
      hc_expand4_vec4_kernel<<<((n >> 2u) + 255) / 256, 256, 0, stream>>>(
          out_hc, block_out, residual_hc, split, n_embd, n_tokens);
      return;
    }
    hc_expand4_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        out_hc, block_out, residual_hc, split, n_embd, n_tokens);
    return;
  }
  const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
  const uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
  hc_expand_kernel<<<(n_elem + 255) / 256, 256, 0, stream>>>(
      out_hc, block_out, block_out, residual_hc, split + n_hc,
      split + 2u * n_hc, n_embd, n_hc, n_tokens, mix_hc, mix_hc, 0);
}

void HcExpandAddSplit(float* out_hc, const float* block_out,
                      const float* block_add, const float* residual_hc,
                      const float* split, uint32_t n_embd, uint32_t n_hc,
                      uint32_t n_tokens, hipStream_t stream) {
  if (n_hc == 4u) {
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    hc_expand4_add_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        out_hc, block_out, block_add, residual_hc, split, n_embd, n_tokens);
    return;
  }
  const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
  const uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
  hc_expand_kernel<<<(n_elem + 255) / 256, 256, 0, stream>>>(
      out_hc, block_out, block_add, residual_hc, split + n_hc,
      split + 2u * n_hc, n_embd, n_hc, n_tokens, mix_hc, mix_hc, 1);
}

bool HcExpandAddSplitMoesum(float* out_hc, const __half* down_h,
                            const float* block_add, const float* residual_hc,
                            const float* split, uint32_t n_embd, uint32_t n_hc,
                            uint32_t n_expert, uint32_t n_tokens,
                            hipStream_t stream) {
  if (n_hc != 4u || n_embd == 0u || n_expert == 0u || n_tokens == 0u) {
    return false;
  }
  const uint64_t n = (uint64_t)n_tokens * n_embd;
  if (hip_hc_vec4_ok(n_embd, out_hc, block_add, residual_hc) &&
      ((uintptr_t)down_h & 7u) == 0u) {
    hc_expand4_add_moesum_vec4_kernel<<<((n >> 2u) + 255) / 256, 256, 0,
                                        stream>>>(out_hc, down_h, block_add,
                                                  residual_hc, split, n_embd,
                                                  n_expert, n_tokens);
    return true;
  }
  hc_expand4_add_moesum_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      out_hc, down_h, block_add, residual_hc, split, n_embd, n_expert,
      n_tokens);
  return true;
}

void SpecRowArgmax(int32_t* out_index, const float* logits, uint32_t vocab,
                   uint32_t n_rows, hipStream_t stream) {
  spec_row_argmax_kernel<<<n_rows, 256, 0, stream>>>(out_index, logits, vocab,
                                                     n_rows);
}

void DsparkCaptureFeatures(float* out, const float* hc,
                           uint32_t out_row_stride, uint32_t slot_offset,
                           uint32_t n_embd, uint32_t n_hc, uint32_t n_rows,
                           hipStream_t stream) {
  const uint64_t n = (uint64_t)n_embd * n_rows;
  dspark_capture_features_kernel<<<(n + 255u) / 256u, 256, 0, stream>>>(
      out + slot_offset, hc, out_row_stride, n_embd, n_hc, n_rows);
}

void EmbedTokenHc(float* out_hc, const __half* w, uint32_t /*n_vocab*/,
                  uint32_t token, uint32_t n_embd, uint32_t n_hc,
                  hipStream_t stream) {
  const uint64_t n = (uint64_t)n_embd * n_hc;
  embed_token_hc_kernel<<<(n + 255u) / 256u, 256, 0, stream>>>(
      out_hc, reinterpret_cast<const unsigned short*>(w), token, n_embd, n_hc);
}

void EmbedTokensHc(float* out_hc, const int32_t* tokens, const __half* w,
                   uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd,
                   uint32_t n_hc, hipStream_t stream) {
  const uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
  embed_tokens_hc_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      out_hc, tokens, w, n_vocab, n_tokens, n_embd, n_hc);
}

void Fp8KvQuantize(float* x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot,
                   hipStream_t stream) {
  if (n_tok == 0u || head_dim == 0u) return;
  const uint32_t n_nope = head_dim - n_rot;
  if (n_nope == 0) return;
  const uint32_t groups = (n_nope + 63u) / 64u;
  fp8_kv_quantize_kernel<<<dim3(n_tok, groups), 64, 0, stream>>>(
      x, nullptr, n_tok, head_dim, n_rot);
}

void Fp8KvQuantizeMirrorF16(float* x, __half* mirror_f16, uint32_t n_tok,
                            uint32_t head_dim, uint32_t n_rot,
                            hipStream_t stream) {
  if (n_tok == 0u || head_dim == 0u) return;
  const uint32_t groups = (head_dim + 63u) / 64u;
  fp8_kv_quantize_kernel<<<dim3(n_tok, groups), 64, 0, stream>>>(
      x, mirror_f16, n_tok, head_dim, n_rot);
}

void ConvertF32ToF16(__half* dst, const float* src, std::size_t count,
                     hipStream_t stream) {
  // hip_launch_f32_to_f16 on `stream`.
  if (count == 0u) return;
  if ((count & 3u) == 0u && ((uintptr_t)src & 15u) == 0u &&
      ((uintptr_t)dst & 7u) == 0u) {
    const uint64_t groups = count >> 2u;
    f32_to_f16_vec4_kernel<<<(groups + 255u) / 256u, 256, 0, stream>>>(
        dst, src, groups);
    return;
  }
  f32_to_f16_kernel<<<(count + 255u) / 256u, 256, 0, stream>>>(dst, src,
                                                               count);
}

void FillF32(float* x, float value, std::size_t count, hipStream_t stream) {
  if (count == 0) return;
  fill_f32_kernel<<<(count + 255u) / 256u, 256, 0, stream>>>(x, count, value);
}

void Swiglu(float* out, const float* gate, const float* up, uint32_t n,
            float clamp, float weight, hipStream_t stream) {
  if (n == 0u) return;
  swiglu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(out, gate, up, n, clamp,
                                                     weight);
}

void StoreRawKvBatch(float* raw_cache, const float* kv, uint32_t raw_cap,
                     uint32_t pos0, uint32_t n_tokens, uint32_t head_dim,
                     hipStream_t stream) {
  const uint64_t n = (uint64_t)n_tokens * head_dim;
  store_raw_kv_batch_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      raw_cache, kv, raw_cap, pos0, n_tokens, head_dim);
}

void IndexerQat(float* x, uint32_t n_rows, uint32_t head_dim,
                hipStream_t stream) {
  indexer_hadamard_fp4_kernel<<<n_rows, 128, 0, stream>>>(x, n_rows, head_dim);
}

// DSpark path selection for one drafted position. Selecting each position
// independently by argmax lets the block drift, because position t's best
// token given the block's own context is not the best token given what
// position t-1 actually emitted. The Markov head restores that coupling with
// a rank-256 bilinear term, and it stays on the device: the vocabulary row
// never moves to the host.
void DsparkCandidates(void* out, uint64_t* scratch, const float* logits,
                      std::size_t logits_stride, const float* hidden,
                      std::size_t hidden_stride, const int32_t* previous,
                      const void* w1, const void* w2, const void* confidence,
                      uint32_t vocab, uint32_t rank, uint32_t hidden_width,
                      uint32_t requests, hipStream_t stream) {
  const uint32_t blocks = (vocab + 255u) / 256u;
  dspark_markov_candidates_kernel<<<dim3(blocks, requests), 256, 0, stream>>>(
      scratch, logits, logits_stride, previous,
      static_cast<const unsigned char*>(w1),
      static_cast<const unsigned char*>(w2), vocab, rank);
  dspark_merge_candidates_kernel<<<requests, 256, 0, stream>>>(
      static_cast<ds4_dspark_candidates*>(out), scratch, blocks, hidden,
      hidden_stride, hidden_width, previous,
      static_cast<const unsigned char*>(w1),
      static_cast<const unsigned char*>(confidence), rank);
}

void DsparkMarkovArgmax(int32_t* out_index, uint64_t* scratch_key,
                        const float* logits_row, const void* w1_row,
                        const void* w2, uint32_t vocab, uint32_t markov_rank,
                        hipStream_t stream) {
  auto* key = reinterpret_cast<unsigned long long*>(scratch_key);
  (void)hipMemsetAsync(key, 0, sizeof(unsigned long long), stream);
  const uint32_t blocks = (vocab + 255u) / 256u;
  dspark_markov_argmax_kernel<<<blocks, 256, 0, stream>>>(
      key, logits_row, static_cast<const unsigned char*>(w1_row),
      static_cast<const unsigned char*>(w2), vocab, markov_rank / 32u);
  dspark_markov_key_decode_kernel<<<1, 1, 0, stream>>>(out_index, key);
}

bool DsparkMarkovArgmaxBatch(int32_t* out_index, uint64_t* scratch_key,
                             const float* logits_rows,
                             std::size_t logits_row_stride,
                             const int32_t* previous_tokens, const void* w1,
                             const void* w2, uint32_t vocab,
                             uint32_t markov_rank, uint32_t n_rows,
                             hipStream_t stream) {
  if (n_rows != 2u && n_rows != 4u && n_rows != 6u && n_rows != 8u) {
    return false;
  }
  auto* keys = reinterpret_cast<unsigned long long*>(scratch_key);
  (void)hipMemsetAsync(keys, 0, (size_t)n_rows * sizeof(unsigned long long),
                       stream);
  const uint32_t blocks = (vocab + 255u) / 256u;
  const uint32_t rank_blocks = markov_rank / 32u;
  const auto* w1b = static_cast<const unsigned char*>(w1);
  const auto* w2b = static_cast<const unsigned char*>(w2);
  switch (n_rows) {
    case 2u:
      dspark_markov_argmax_batch_kernel<2u><<<blocks, 256, 0, stream>>>(
          keys, logits_rows, logits_row_stride, previous_tokens, w1b, w2b,
          vocab, rank_blocks);
      break;
    case 4u:
      dspark_markov_argmax_batch_kernel<4u><<<blocks, 256, 0, stream>>>(
          keys, logits_rows, logits_row_stride, previous_tokens, w1b, w2b,
          vocab, rank_blocks);
      break;
    case 6u:
      dspark_markov_argmax_batch_kernel<6u><<<blocks, 256, 0, stream>>>(
          keys, logits_rows, logits_row_stride, previous_tokens, w1b, w2b,
          vocab, rank_blocks);
      break;
    default:
      dspark_markov_argmax_batch_kernel<8u><<<blocks, 256, 0, stream>>>(
          keys, logits_rows, logits_row_stride, previous_tokens, w1b, w2b,
          vocab, rank_blocks);
      break;
  }
  dspark_markov_key_decode_batch_kernel<<<1, 32, 0, stream>>>(out_index, keys,
                                                              n_rows);
  return true;
}

void AttentionDecodeHeads(float* heads, const float* sinks, const float* q,
                          const float* raw_kv, uint32_t n_raw,
                          uint32_t raw_cap, uint32_t raw_start,
                          const float* comp_kv, uint32_t n_comp,
                          const float* comp_mask, uint32_t n_head,
                          uint32_t head_dim, hipStream_t stream) {
  const uint32_t use_mask = comp_mask != nullptr ? 1u : 0u;
  const size_t shmem = (size_t)(n_raw + n_comp) * sizeof(float);
  attention_decode_mixed_one_fast_oldhip_kernel<<<n_head, 256, shmem,
                                                  stream>>>(
      heads, q, raw_kv, comp_kv, comp_mask, sinks, n_raw,
      raw_cap, raw_start, n_comp, use_mask, n_head, head_dim,
      (uint32_t)((head_dim & 3u) == 0u));
}

void AttentionNoncausalRawBatchHeads(float* heads, const float* sinks,
                                     const float* q, const float* raw_kv,
                                     uint32_t n_tokens, uint32_t n_raw,
                                     uint32_t raw_cap, uint32_t raw_start,
                                     uint32_t n_head, uint32_t head_dim,
                                     hipStream_t stream) {
  // The kernel stages one score per visible key in shared memory.
  const size_t shared_bytes = (size_t)n_raw * sizeof(float);
  dim3 grid(n_tokens, n_head, 1);
  attention_noncausal_raw_batch_heads_kernel<<<grid, 256, shared_bytes,
                                               stream>>>(
      heads, sinks, q, raw_kv, n_tokens, n_raw, raw_cap, raw_start, n_head,
      head_dim);
}

// The compressors' positional bias is F16 (the loader requires it); the
// kernels read it through their type switch.
constexpr uint32_t kCompressorApeF16 = 1u;

// `coff` (2 for ratio 4, else 1) sizes the extents only; the kernels derive
// it from the ratio.
void CompressorStore(const float* kv, const float* sc, float* state_kv,
                     float* state_score, const __half* ape, uint32_t /*coff*/,
                     uint32_t head_dim, uint32_t ratio, uint32_t pos0,
                     uint32_t n_tokens, hipStream_t stream) {
  const uint32_t coff = ratio == 4u ? 2u : 1u;
  const uint64_t n = (uint64_t)n_tokens * coff * head_dim;
  compressor_store_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      kv, sc, state_kv, state_score, ape, 0, kCompressorApeF16, head_dim, ratio,
      pos0, n_tokens);
}

void CompressorStateReset(float* state_kv, float* state_score,
                          std::size_t count, hipStream_t stream) {
  (void)hipMemsetAsync(state_kv, 0, count * sizeof(float), stream);
  fill_f32_kernel<<<(count + 255) / 256, 256, 0, stream>>>(state_score, count,
                                                            -INFINITY);
}

void CompressorSetRows(float* state_kv, float* state_score, const float* kv,
                       const float* sc, const __half* ape, uint32_t width,
                       uint32_t ratio, uint32_t pos0, uint32_t src0,
                       uint32_t dst0, uint32_t rows, hipStream_t stream) {
  const uint64_t n = (uint64_t)rows * width;
  compressor_set_rows_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      state_kv, state_score, kv, sc, ape, 0, kCompressorApeF16, width, ratio,
      pos0, src0, dst0, rows);
}

void CompressorPrefillPool(float* comp, const float* kv, const float* sc,
                           const float* state_kv, const float* state_score,
                           const __half* ape, uint32_t /*coff*/,
                           uint32_t head_dim, uint32_t ratio, uint32_t pos0,
                           uint32_t n_comp, bool replay, hipStream_t stream) {
  dim3 grid((head_dim + 255) / 256, n_comp, 1);
  compressor_prefill_pool_kernel<<<grid, 256, 0, stream>>>(
      comp, kv, sc, state_kv, state_score, ape, 0, kCompressorApeF16, head_dim,
      ratio, pos0, n_comp, replay ? 1u : 0u);
}

void CompressorUpdatePool(float* row, const float* state_kv,
                          const float* state_score, uint32_t /*coff*/,
                          uint32_t head_dim, uint32_t ratio,
                          hipStream_t stream) {
  compressor_update_pool_kernel<<<(head_dim + 255) / 256, 256, 0, stream>>>(
      row, state_kv, state_score, head_dim, ratio);
}

void CompressorShiftRatio4(float* state_kv, float* state_score, uint32_t width,
                           hipStream_t stream) {
  const uint64_t half = 4ull * width;
  compressor_shift_ratio4_kernel<<<(half + 255) / 256, 256, 0, stream>>>(
      state_kv, state_score, width);
}

// The shared-x group-pair kernels stage both rows of a pair in shared memory:
// 2 * in_dim floats, at most 64 KiB.
constexpr uint32_t kGroupPairSharedMaxInDim = 8192u;

void MatmulF16EqualGroupPairs(float* out, const __half* w, const float* x,
                              uint32_t in_dim, uint32_t out_dim,
                              uint32_t rows_per_group, uint32_t pairs,
                              hipStream_t stream) {
  if (in_dim <= kGroupPairSharedMaxInDim) {
    const uint32_t rows_per_block = 32u;
    const dim3 grid((out_dim + rows_per_block - 1u) / rows_per_block,
                    pairs * rows_per_group, 1u);
    matmul_f16_f32_equal_group_pairs_sharedx_warp_rows_w32_kernel<<<
        grid, rows_per_block * 32u, 2u * static_cast<size_t>(in_dim) * sizeof(float),
        stream>>>(out, w, x, in_dim, out_dim, rows_per_group);
    return;
  }
  const dim3 grid(out_dim, pairs * rows_per_group, 1u);
  matmul_f16_ordered_equal_group_pairs_exact_kernel<<<grid, 32u, 0, stream>>>(
      out, w, x, in_dim, out_dim, rows_per_group);
}

void MatmulF16RaggedGroupPair(float* out, const __half* w, const float* x,
                              uint32_t in_dim, uint32_t out_dim, uint32_t row0,
                              uint32_t rows0, uint32_t rows1,
                              hipStream_t stream) {
  const uint32_t row1 = row0 + rows0;
  const uint32_t pair_rows = rows0 > rows1 ? rows0 : rows1;
  if (in_dim <= kGroupPairSharedMaxInDim) {
    const uint32_t rows_per_block = 32u;
    const dim3 grid((out_dim + rows_per_block - 1u) / rows_per_block,
                    pair_rows, 1u);
    matmul_f16_f32_ragged_group_pair_sharedx_warp_rows_w32_kernel<<<
        grid, rows_per_block * 32u, 2u * static_cast<size_t>(in_dim) * sizeof(float),
        stream>>>(out, w, x, in_dim, out_dim, row0, row1, rows0, rows1);
    return;
  }
  const dim3 grid(out_dim, pair_rows, 1u);
  matmul_f16_ordered_ragged_group_pair_exact_kernel<<<grid, 32u, 0, stream>>>(
      out, w, x, in_dim, out_dim, row0, row1, rows0, rows1);
}

template <uint32_t Batch>
void LaunchF16PairNarrow(float* out0, float* out1, const __half* w0,
                         const __half* w1, const float* x, uint32_t in_dim,
                         uint32_t out_dim, hipStream_t stream) {
  constexpr uint32_t rows_per_block = 32u;
  const unsigned grid =
      (unsigned)(((uint64_t)out_dim + rows_per_block - 1u) / rows_per_block);
  matmul_f16_pair_batch_reuse_warp_rows_w32_kernel<Batch>
      <<<grid, rows_per_block * 32u, 0, stream>>>(out0, out1, w0, w1, x, in_dim,
                                                  out_dim, rows_per_block);
}

void MatmulF16PairNarrow(float* out0, float* out1, const __half* w0,
                         const __half* w1, const float* x, uint32_t in_dim,
                         uint32_t out_dim, uint32_t n_tok, hipStream_t stream) {
  switch (n_tok) {
    case 2u:
      LaunchF16PairNarrow<2u>(out0, out1, w0, w1, x, in_dim, out_dim, stream);
      break;
    case 3u:
      LaunchF16PairNarrow<3u>(out0, out1, w0, w1, x, in_dim, out_dim, stream);
      break;
    case 4u:
      LaunchF16PairNarrow<4u>(out0, out1, w0, w1, x, in_dim, out_dim, stream);
      break;
    case 5u:
      LaunchF16PairNarrow<5u>(out0, out1, w0, w1, x, in_dim, out_dim, stream);
      break;
    case 6u:
      LaunchF16PairNarrow<6u>(out0, out1, w0, w1, x, in_dim, out_dim, stream);
      break;
    case 7u:
      LaunchF16PairNarrow<7u>(out0, out1, w0, w1, x, in_dim, out_dim, stream);
      break;
    default:
      LaunchF16PairNarrow<8u>(out0, out1, w0, w1, x, in_dim, out_dim, stream);
      break;
  }
}

// Dense projections. Q8_0 weights are rows of 34-byte blocks (an F16 scale
// and 32 int8 values); `Preq` kernels read activations pre-quantized by
// QuantizeQ8Rows into the same blocks.

void QuantizeQ8Rows(void* xq, float* xscale, const float* x, uint32_t in_dim,
                    uint32_t n_tok, hipStream_t stream) {
  const uint64_t blocks = (in_dim + 31u) / 32u;
  dim3 grid((unsigned)blocks, n_tok, 1);
  quantize_q8_0_f32_kernel<<<grid, 32, 0, stream>>>(
      static_cast<int8_t*>(xq), xscale, x, in_dim, blocks);
}

void MatmulQ8Wmma(float* out, const void* w, const float* x, uint32_t in_dim,
                  uint32_t out_dim, uint32_t n_tok, hipStream_t stream) {
  const uint64_t blocks = in_dim / 32u;
  const dim3 grid((out_dim + 63u) / 64u, (n_tok + 63u) / 64u, 1u);
  matmul_q8_0_f32_batch_wmma_4w_kernel<<<grid, 128u, 0, stream>>>(
      out, static_cast<const unsigned char*>(w), x, n_tok, in_dim, out_dim,
      blocks * 34u);
}

template <uint32_t MaxT, bool Exact>
static void LaunchQ8PreqReuse(float* out, const unsigned char* w,
                              const int8_t* xq, const float* xscale,
                              uint64_t in_dim, uint64_t out_dim,
                              uint64_t blocks, uint32_t n_tok,
                              uint32_t rows_per_block, hipStream_t stream) {
  const unsigned grid =
      (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block);
  matmul_q8_0_preq_batch_reuse_w32_kernel<MaxT, Exact>
      <<<grid, rows_per_block * 32u, 0, stream>>>(
          out, w, xq, xscale, in_dim, out_dim, blocks, n_tok, rows_per_block);
}

bool MatmulQ8PreqReuse(float* out, const void* w, const void* xq,
                       const float* xscale, uint32_t in_dim, uint32_t out_dim,
                       uint32_t n_tok, uint32_t rows_per_block,
                       hipStream_t stream) {
  const auto* wb = static_cast<const unsigned char*>(w);
  const auto* q = static_cast<const int8_t*>(xq);
  const uint64_t blocks = (in_dim + 31u) / 32u;
  switch (n_tok) {
    case 2u:
      LaunchQ8PreqReuse<2u, true>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                  n_tok, rows_per_block, stream);
      break;
    case 3u:
      LaunchQ8PreqReuse<3u, true>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                  n_tok, rows_per_block, stream);
      break;
    case 4u:
      LaunchQ8PreqReuse<4u, true>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                  n_tok, rows_per_block, stream);
      break;
    case 5u:
      LaunchQ8PreqReuse<5u, true>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                  n_tok, rows_per_block, stream);
      break;
    case 6u:
      LaunchQ8PreqReuse<6u, true>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                  n_tok, rows_per_block, stream);
      break;
    case 7u:
      LaunchQ8PreqReuse<7u, true>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                  n_tok, rows_per_block, stream);
      break;
    case 8u:
      LaunchQ8PreqReuse<8u, true>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                  n_tok, rows_per_block, stream);
      break;
    case 10u:
      LaunchQ8PreqReuse<10u, true>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                   n_tok, rows_per_block, stream);
      break;
    case 12u:
      LaunchQ8PreqReuse<12u, true>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                   n_tok, rows_per_block, stream);
      break;
    case 16u:
      LaunchQ8PreqReuse<16u, true>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                   n_tok, rows_per_block, stream);
      break;
    case 18u:
      LaunchQ8PreqReuse<18u, true>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                   n_tok, rows_per_block, stream);
      break;
    case 24u:
      LaunchQ8PreqReuse<24u, true>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                   n_tok, rows_per_block, stream);
      break;
    case 32u:
      LaunchQ8PreqReuse<32u, true>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                   n_tok, rows_per_block, stream);
      break;
    default:
      // The inexact form holds up to 16 rows.
      if (n_tok > 16u) return false;
      LaunchQ8PreqReuse<16u, false>(out, wb, q, xscale, in_dim, out_dim,
                                    blocks, n_tok, rows_per_block, stream);
      break;
  }
  return true;
}

template <uint32_t TokTile>
static void LaunchQ8Sharedx(float* out, const unsigned char* w, const float* x,
                            uint32_t n_blocks, uint32_t out_dim,
                            uint32_t n_tok, uint64_t row_bytes,
                            hipStream_t stream) {
  constexpr uint32_t kRowsPerBlock = 32u;
  constexpr uint32_t kBlockTile = 16u;
  const dim3 grid((out_dim + kRowsPerBlock - 1u) / kRowsPerBlock,
                  (n_tok + TokTile - 1u) / TokTile, 1u);
  const size_t shmem = (size_t)TokTile * kBlockTile * 32u * sizeof(float);
  matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<TokTile,
                                                             kBlockTile>
      <<<grid, kRowsPerBlock * 32u, shmem, stream>>>(out, w, x, n_blocks,
                                                     out_dim, n_tok, row_bytes);
}

void MatmulQ8Sharedx(float* out, const void* w, const float* x,
                     uint32_t in_dim, uint32_t out_dim, uint32_t n_tok,
                     hipStream_t stream) {
  // The token tile is the unit of shared-activation staging and of the token
  // loop inside the kernel, so it follows the rows actually present: a
  // 32-wide tile on a six-row verification block does more than five times
  // the necessary work. Prompt chunks keep the 32-wide tile.
  const auto* wb = static_cast<const unsigned char*>(w);
  const uint32_t blocks = in_dim / 32u;
  const uint64_t row_bytes = (uint64_t)blocks * 34u;
  if (n_tok >= 32u) {
    LaunchQ8Sharedx<32u>(out, wb, x, blocks, out_dim, n_tok, row_bytes, stream);
  } else if (n_tok > 8u) {
    LaunchQ8Sharedx<16u>(out, wb, x, blocks, out_dim, n_tok, row_bytes, stream);
  } else if (n_tok > 4u) {
    LaunchQ8Sharedx<8u>(out, wb, x, blocks, out_dim, n_tok, row_bytes, stream);
  } else if (n_tok > 2u) {
    LaunchQ8Sharedx<4u>(out, wb, x, blocks, out_dim, n_tok, row_bytes, stream);
  } else {
    LaunchQ8Sharedx<2u>(out, wb, x, blocks, out_dim, n_tok, row_bytes, stream);
  }
}

void MatmulQ8Warp8(float* out, const void* w, const float* x, uint32_t in_dim,
                   uint32_t out_dim, uint32_t n_tok, hipStream_t stream) {
  const uint64_t blocks = (in_dim + 31u) / 32u;
  dim3 grid((out_dim + 7u) / 8u, n_tok, 1);
  matmul_q8_0_f32_batch_warp8_kernel<<<grid, 256, 0, stream>>>(
      out, static_cast<const unsigned char*>(w), x, in_dim, out_dim, n_tok,
      blocks);
}

void MatmulQ8PreqRows(float* out, const void* w, const void* xq,
                      const float* xscale, uint32_t in_dim, uint32_t out_dim,
                      hipStream_t stream) {
  const uint64_t blocks = (in_dim + 31u) / 32u;
  const uint32_t rows_per_block = 1u;
  matmul_q8_0_preq_rows_w32_kernel<<<(out_dim + rows_per_block - 1u) /
                                         rows_per_block,
                                     rows_per_block * 32u, 0, stream>>>(
      out, static_cast<const unsigned char*>(w),
      static_cast<const int8_t*>(xq), xscale, in_dim, out_dim, blocks,
      rows_per_block, 1);
}

template <uint32_t NT>
static void LaunchQ8PreqAllRows(float* out, const unsigned char* w,
                                const int8_t* xq, const float* xscale,
                                uint64_t in_dim, uint64_t out_dim,
                                uint64_t blocks, hipStream_t stream) {
  matmul_q8_0_preq_all_rows_exact_w32_kernel<NT>
      <<<(unsigned)out_dim, 32u, 0, stream>>>(out, w, xq, xscale, in_dim,
                                              out_dim, blocks, 1u);
}

bool MatmulQ8PreqAllRows(float* out, const void* w, const void* xq,
                         const float* xscale, uint32_t in_dim,
                         uint32_t out_dim, uint32_t n_tok, hipStream_t stream) {
  const auto* wb = static_cast<const unsigned char*>(w);
  const auto* q = static_cast<const int8_t*>(xq);
  const uint64_t blocks = in_dim / 32u;
  switch (n_tok) {
    case 6u:
      LaunchQ8PreqAllRows<6u>(out, wb, q, xscale, in_dim, out_dim, blocks,
                              stream);
      return true;
    case 8u:
      LaunchQ8PreqAllRows<8u>(out, wb, q, xscale, in_dim, out_dim, blocks,
                              stream);
      return true;
    case 12u:
      LaunchQ8PreqAllRows<12u>(out, wb, q, xscale, in_dim, out_dim, blocks,
                               stream);
      return true;
    case 16u:
      LaunchQ8PreqAllRows<16u>(out, wb, q, xscale, in_dim, out_dim, blocks,
                               stream);
      return true;
    case 18u:
      LaunchQ8PreqAllRows<18u>(out, wb, q, xscale, in_dim, out_dim, blocks,
                               stream);
      return true;
    case 24u:
      LaunchQ8PreqAllRows<24u>(out, wb, q, xscale, in_dim, out_dim, blocks,
                               stream);
      return true;
    case 32u:
      LaunchQ8PreqAllRows<32u>(out, wb, q, xscale, in_dim, out_dim, blocks,
                               stream);
      return true;
    default:
      return false;
  }
}

template <uint32_t NT>
static void LaunchQ8PreqEqualGroupPairs(float* out, const unsigned char* w,
                                        const int8_t* xq, const float* xscale,
                                        uint64_t in_dim, uint64_t out_dim,
                                        uint64_t blocks, uint32_t pairs,
                                        hipStream_t stream) {
  const dim3 grid((unsigned)out_dim, pairs, 1u);
  matmul_q8_0_preq_equal_group_pairs_w32_kernel<NT>
      <<<grid, 32u, 0, stream>>>(out, w, xq, xscale, in_dim, out_dim, blocks,
                                 1u);
}

void MatmulQ8PreqEqualGroupPairs(float* out, const void* w, const void* xq,
                                 const float* xscale, uint32_t in_dim,
                                 uint32_t out_dim, uint32_t rows_per_group,
                                 uint32_t pairs, hipStream_t stream) {
  const auto* wb = static_cast<const unsigned char*>(w);
  const auto* q = static_cast<const int8_t*>(xq);
  const uint64_t blocks = in_dim / 32u;
  switch (rows_per_group) {
    case 1u:
      LaunchQ8PreqEqualGroupPairs<1u>(out, wb, q, xscale, in_dim, out_dim,
                                      blocks, pairs, stream);
      break;
    case 2u:
      LaunchQ8PreqEqualGroupPairs<2u>(out, wb, q, xscale, in_dim, out_dim,
                                      blocks, pairs, stream);
      break;
    case 3u:
      LaunchQ8PreqEqualGroupPairs<3u>(out, wb, q, xscale, in_dim, out_dim,
                                      blocks, pairs, stream);
      break;
    case 4u:
      LaunchQ8PreqEqualGroupPairs<4u>(out, wb, q, xscale, in_dim, out_dim,
                                      blocks, pairs, stream);
      break;
    case 5u:
      LaunchQ8PreqEqualGroupPairs<5u>(out, wb, q, xscale, in_dim, out_dim,
                                      blocks, pairs, stream);
      break;
    default:
      LaunchQ8PreqEqualGroupPairs<6u>(out, wb, q, xscale, in_dim, out_dim,
                                      blocks, pairs, stream);
      break;
  }
}

template <uint32_t NT0, uint32_t NT1>
static void LaunchQ8PreqGroupPair(float* out, const unsigned char* w,
                                  const int8_t* xq, const float* xscale,
                                  uint64_t in_dim, uint64_t out_dim,
                                  uint64_t blocks, uint32_t row0, uint32_t row1,
                                  hipStream_t stream) {
  if constexpr (NT0 == NT1) {
    matmul_q8_0_preq_group_pair_w32_kernel<NT0>
        <<<(unsigned)out_dim, 32u, 0, stream>>>(out, w, xq, xscale, in_dim,
                                                out_dim, blocks, row0, row1,
                                                1u);
  } else {
    matmul_q8_0_preq_ragged_group_pair_w32_kernel<NT0, NT1>
        <<<(unsigned)out_dim, 32u, 0, stream>>>(out, w, xq, xscale, in_dim,
                                                out_dim, blocks, row0, row1,
                                                1u);
  }
}

template <uint32_t NT0>
static void DispatchQ8PreqGroupPair(float* out, const unsigned char* w,
                                    const int8_t* xq, const float* xscale,
                                    uint64_t in_dim, uint64_t out_dim,
                                    uint64_t blocks, uint32_t row0,
                                    uint32_t row1, uint32_t rows1,
                                    hipStream_t stream) {
  switch (rows1) {
    case 1u:
      LaunchQ8PreqGroupPair<NT0, 1u>(out, w, xq, xscale, in_dim, out_dim,
                                     blocks, row0, row1, stream);
      break;
    case 2u:
      LaunchQ8PreqGroupPair<NT0, 2u>(out, w, xq, xscale, in_dim, out_dim,
                                     blocks, row0, row1, stream);
      break;
    case 3u:
      LaunchQ8PreqGroupPair<NT0, 3u>(out, w, xq, xscale, in_dim, out_dim,
                                     blocks, row0, row1, stream);
      break;
    case 4u:
      LaunchQ8PreqGroupPair<NT0, 4u>(out, w, xq, xscale, in_dim, out_dim,
                                     blocks, row0, row1, stream);
      break;
    case 5u:
      LaunchQ8PreqGroupPair<NT0, 5u>(out, w, xq, xscale, in_dim, out_dim,
                                     blocks, row0, row1, stream);
      break;
    default:
      LaunchQ8PreqGroupPair<NT0, 6u>(out, w, xq, xscale, in_dim, out_dim,
                                     blocks, row0, row1, stream);
      break;
  }
}

void MatmulQ8PreqGroupPair(float* out, const void* w, const void* xq,
                           const float* xscale, uint32_t in_dim,
                           uint32_t out_dim, uint32_t row0, uint32_t rows0,
                           uint32_t rows1, hipStream_t stream) {
  const auto* wb = static_cast<const unsigned char*>(w);
  const auto* q = static_cast<const int8_t*>(xq);
  const uint64_t blocks = in_dim / 32u;
  const uint32_t row1 = row0 + rows0;
  switch (rows0) {
    case 1u:
      DispatchQ8PreqGroupPair<1u>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                  row0, row1, rows1, stream);
      break;
    case 2u:
      DispatchQ8PreqGroupPair<2u>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                  row0, row1, rows1, stream);
      break;
    case 3u:
      DispatchQ8PreqGroupPair<3u>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                  row0, row1, rows1, stream);
      break;
    case 4u:
      DispatchQ8PreqGroupPair<4u>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                  row0, row1, rows1, stream);
      break;
    case 5u:
      DispatchQ8PreqGroupPair<5u>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                  row0, row1, rows1, stream);
      break;
    default:
      DispatchQ8PreqGroupPair<6u>(out, wb, q, xscale, in_dim, out_dim, blocks,
                                  row0, row1, rows1, stream);
      break;
  }
}

void MatmulQ8PairPreqReuse(float* out0, float* out1, const void* w0,
                           const void* w1, const void* xq, const float* xscale,
                           uint32_t in_dim, uint32_t out_dim, uint32_t n_tok,
                           hipStream_t stream) {
  // The paired scalar projection rounds the activation product first, in
  // chunks of four rows; blockIdx.y picks the weight.
  const uint64_t blocks = (in_dim + 31u) / 32u;
  const auto* q = static_cast<const int8_t*>(xq);
  for (uint32_t first = 0; first < n_tok; first += 4u) {
    const uint32_t count = n_tok - first < 4u ? n_tok - first : 4u;
    matmul_q8_0_preq_batch_reuse_w32_kernel<4, false, true>
        <<<dim3(out_dim, 2), 32, 0, stream>>>(
            out0 + (uint64_t)first * out_dim,
            static_cast<const unsigned char*>(w0),
            q + (uint64_t)first * blocks * 32u, xscale + (uint64_t)first * blocks,
            in_dim, out_dim, blocks, count, 1u,
            static_cast<const unsigned char*>(w1),
            out1 + (uint64_t)first * out_dim);
  }
}

void MatmulQ8PairPreqRows(float* out0, float* out1, const void* w0,
                          const void* w1, const void* xq, const float* xscale,
                          uint32_t in_dim, uint32_t out0_dim, uint32_t out1_dim,
                          hipStream_t stream) {
  const uint64_t blocks = (in_dim + 31u) / 32u;
  const uint32_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
  matmul_q8_0_pair_preq_warp8_kernel<<<(max_out + 7u) / 8u, 256, 0, stream>>>(
      out0, out1, static_cast<const unsigned char*>(w0),
      static_cast<const unsigned char*>(w1), static_cast<const int8_t*>(xq),
      xscale, in_dim, out0_dim, out1_dim, blocks, 1);
}

void MatmulQ8HcExpandPreqRows(float* out_hc, float* block_out,
                              const float* block_add, const float* residual_hc,
                              const float* split, const void* w,
                              const void* xq, const float* xscale,
                              uint32_t in_dim, uint32_t n_embd, uint32_t n_hc,
                              hipStream_t stream) {
  // Without a block to add the kernel never reads `block_add`.
  const uint64_t blocks = (in_dim + 31u) / 32u;
  const uint32_t rows_per_block = 16u;
  matmul_q8_0_hc_expand_preq_rows_w32_kernel<<<(n_embd + rows_per_block - 1u) /
                                                   rows_per_block,
                                               rows_per_block * 32u, 0,
                                               stream>>>(
      out_hc, block_out, block_add != nullptr ? block_add : block_out,
      residual_hc, split, static_cast<const unsigned char*>(w),
      static_cast<const int8_t*>(xq), xscale, in_dim, n_embd, n_embd, n_hc,
      blocks, rows_per_block, block_add != nullptr ? 1 : 0, 1);
}

void DequantQ8ToF16(__half* out, const void* w, uint32_t in_dim,
                    uint32_t out_dim, hipStream_t stream) {
  const uint64_t blocks = (in_dim + 31u) / 32u;
  const uint64_t n = (uint64_t)in_dim * out_dim;
  dequant_q8_0_to_f16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      out, static_cast<const unsigned char*>(w), in_dim, out_dim, blocks);
}

bool MmqQ8Dense(float* out, const void* w, const float* x, uint32_t in_dim,
                uint32_t out_dim, uint32_t n_tok, hipStream_t stream) {
  return ds4_mmq_q8_0_dense(w, x, out, (int)out_dim, (int)n_tok, (int)in_dim,
                            stream) == 0;
}

void MatmulF16TinyBatch(float* out, const __half* w, const float* x,
                        uint32_t in_dim, uint32_t out_dim, uint32_t n_tok,
                        hipStream_t stream) {
  const dim3 grid(out_dim, n_tok, 1u);
  matmul_f16_tiny_batch_wave_kernel<<<grid, 32u, 0, stream>>>(out, w, x, in_dim,
                                                              out_dim, n_tok);
}

template <uint32_t Batch>
static void LaunchF16OrderedReuse(float* out, const __half* w, const float* x,
                                  uint32_t in_dim, uint32_t out_dim,
                                  hipStream_t stream) {
  matmul_f16_ordered_batch_reuse_kernel<Batch>
      <<<dim3(out_dim, 1u, 1u), 32u, 0, stream>>>(out, w, x, in_dim, out_dim);
}

void MatmulF16OrderedReuse(float* out, const __half* w, const float* x,
                           uint32_t in_dim, uint32_t out_dim, uint32_t n_tok,
                           hipStream_t stream) {
  switch (n_tok) {
    case 1u:
      LaunchF16OrderedReuse<1u>(out, w, x, in_dim, out_dim, stream);
      break;
    case 2u:
      LaunchF16OrderedReuse<2u>(out, w, x, in_dim, out_dim, stream);
      break;
    case 3u:
      LaunchF16OrderedReuse<3u>(out, w, x, in_dim, out_dim, stream);
      break;
    case 4u:
      LaunchF16OrderedReuse<4u>(out, w, x, in_dim, out_dim, stream);
      break;
    case 5u:
      LaunchF16OrderedReuse<5u>(out, w, x, in_dim, out_dim, stream);
      break;
    default:
      LaunchF16OrderedReuse<6u>(out, w, x, in_dim, out_dim, stream);
      break;
  }
}

void MatmulF16SharedxRows(float* out, const __half* w, const float* x,
                          uint32_t in_dim, uint32_t out_dim,
                          hipStream_t stream) {
  const uint32_t rows_per_block = 32u;
  matmul_f16_f32_sharedx_warp_rows_w32_kernel<<<
      (out_dim + rows_per_block - 1u) / rows_per_block, rows_per_block * 32u,
      (size_t)in_dim * sizeof(float), stream>>>(out, w, x, in_dim, out_dim);
}

void MatmulF16OrderedChunks(float* out, const __half* w, const float* x,
                            uint32_t in_dim, uint32_t out_dim, uint32_t n_tok,
                            hipStream_t stream) {
  dim3 grid(out_dim, n_tok, 1);
  matmul_f16_ordered_chunks_kernel<<<grid, 32, 0, stream>>>(out, w, x, in_dim,
                                                            out_dim, n_tok);
}

void MatmulF16PairSharedxRows(float* out0, float* out1, const __half* w0,
                              const __half* w1, const float* x, uint32_t in_dim,
                              uint32_t out_dim, hipStream_t stream) {
  const uint32_t rows_per_block = 32u;
  matmul_f16_pair_f32_sharedx_warp_rows_w32_kernel<<<
      (out_dim + rows_per_block - 1u) / rows_per_block, rows_per_block * 32u,
      (size_t)in_dim * sizeof(float), stream>>>(out0, out1, w0, w1, x, in_dim,
                                                out_dim);
}

void MatmulF16PairOrderedChunks(float* out0, float* out1, const __half* w0,
                                const __half* w1, const float* x,
                                uint32_t in_dim, uint32_t out_dim,
                                hipStream_t stream) {
  matmul_f16_pair_ordered_chunks_kernel<<<out_dim, 32, 0, stream>>>(
      out0, out1, w0, w1, x, in_dim, out_dim, out_dim);
}

void MatmulF32Plain(float* out, const float* w, const float* x, uint32_t in_dim,
                    uint32_t out_dim, uint32_t n_tok, hipStream_t stream) {
  dim3 grid(out_dim, n_tok, 1);
  matmul_f32_kernel<<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim,
                                              n_tok);
}

// Attention over the raw window ring and the compressed rows. Absent
// compressed rows pass the raw rows, which the kernels then never read.

void AttentionDecodeOnline(float* heads, const float* sinks, const float* q,
                           const float* raw_kv, const float* comp_kv,
                           uint32_t n_tokens, uint32_t pos0, uint32_t n_raw,
                           uint32_t raw_cap, uint32_t raw_start,
                           uint32_t n_comp, uint32_t window, uint32_t ratio,
                           uint32_t n_head, uint32_t head_dim,
                           hipStream_t stream) {
  dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
  attention_decode_mixed_heads8_online_kernel<<<grid, 256, 0, stream>>>(
      heads, sinks, q, raw_kv, comp_kv, n_tokens, pos0, n_raw, raw_cap,
      raw_start, n_comp, window, ratio, n_head, head_dim);
}

void AttentionMixedWmma(float* heads, const float* sinks, const float* q,
                        const float* raw_kv, const float* comp_kv,
                        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw,
                        uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp,
                        uint32_t window, uint32_t ratio, uint32_t n_head,
                        uint32_t head_dim, hipStream_t stream) {
  const dim3 grid(n_tokens, n_head / 32u, 1u);
  attention_mixed_heads32_wmma_kernel<false, false><<<grid, 1024, 0, stream>>>(
      heads, sinks, q, raw_kv, comp_kv, nullptr, nullptr, 0u, n_tokens, pos0,
      n_raw, raw_cap, raw_start, n_comp, 0u, window, ratio, n_head, head_dim);
}

void AttentionIndexedWmma(float* heads, const float* sinks, const float* q,
                          const float* raw_kv, const float* comp_kv,
                          const int32_t* topk, uint32_t n_tokens,
                          uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
                          uint32_t raw_start, uint32_t n_comp, uint32_t top_k,
                          uint32_t window, uint32_t ratio, uint32_t n_head,
                          uint32_t head_dim, hipStream_t stream) {
  const dim3 grid(n_tokens, n_head / 32u, 1u);
  attention_mixed_heads32_wmma_kernel<true, false><<<grid, 1024, 0, stream>>>(
      heads, sinks, q, raw_kv, comp_kv, topk, nullptr, 0u, n_tokens, pos0,
      n_raw, raw_cap, raw_start, n_comp, top_k, window, ratio, n_head,
      head_dim);
}

void AttentionIndexedWmmaF16(float* heads, const float* sinks, const float* q,
                             const float* raw_kv, const __half* comp_kv,
                             const int32_t* topk, uint32_t n_tokens,
                             uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
                             uint32_t raw_start, uint32_t n_comp,
                             uint32_t top_k, uint32_t window, uint32_t ratio,
                             uint32_t n_head, uint32_t head_dim,
                             hipStream_t stream) {
  const dim3 grid(n_tokens, n_head / 32u, 1u);
  attention_mixed_heads32_wmma_kernel<true, true><<<grid, 1024, 0, stream>>>(
      heads, sinks, q, raw_kv, comp_kv, topk, nullptr, 0u, n_tokens, pos0,
      n_raw, raw_cap, raw_start, n_comp, top_k, window, ratio, n_head,
      head_dim);
}

void AttentionDecodeMixed(float* heads, const float* sinks, const float* q,
                          const float* raw_kv, const float* comp_kv,
                          const float* comp_mask, uint32_t n_tokens,
                          uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
                          uint32_t raw_start, uint32_t n_comp, uint32_t window,
                          uint32_t ratio, uint32_t n_head, uint32_t head_dim,
                          hipStream_t stream) {
  dim3 grid(n_tokens, n_head, 1);
  attention_decode_mixed_kernel<<<grid, 256, 0, stream>>>(
      heads, sinks, q, raw_kv, comp_kv, comp_mask,
      comp_mask != nullptr ? 1u : 0u, n_tokens, pos0, n_raw, raw_cap, raw_start,
      n_comp, window, ratio, n_head, head_dim);
}

void AttentionDecodeIndexedOne(float* heads, const float* q,
                               const float* raw_kv, const float* comp_kv,
                               const int32_t* topk, const float* sinks,
                               uint32_t n_raw, uint32_t raw_cap,
                               uint32_t raw_start, uint32_t n_comp,
                               uint32_t top_k, uint32_t pos, uint32_t ratio,
                               uint32_t n_head, uint32_t head_dim,
                               hipStream_t stream) {
  // One score per raw row and selected compressed row in shared memory.
  const uint32_t rows = n_raw + (top_k < n_comp ? top_k : n_comp);
  const size_t shmem = (size_t)(rows ? rows : 1u) * sizeof(float);
  attention_decode_indexed_mixed_one_fast_oldhip_kernel<<<n_head, 256, shmem,
                                                          stream>>>(
      heads, q, raw_kv, comp_kv, topk, sinks, n_raw, raw_cap, raw_start, n_comp,
      top_k, pos, ratio, n_head, head_dim, (uint32_t)((head_dim & 3u) == 0u));
}

void IndexedTopkSort512(int32_t* dst, const int32_t* src, uint32_t n_tokens,
                        hipStream_t stream) {
  indexed_topk_sort_512_asc_kernel<<<n_tokens, 512, 0, stream>>>(dst, src,
                                                                 n_tokens);
}

void AttentionIndexedOnline(float* heads, const float* sinks, const float* q,
                            const float* raw_kv, const float* comp_kv,
                            const int32_t* topk, uint32_t n_tokens,
                            uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
                            uint32_t raw_start, uint32_t n_comp,
                            uint32_t top_k, uint32_t window, uint32_t ratio,
                            uint32_t n_head, uint32_t head_dim,
                            hipStream_t stream) {
  if (n_head <= 64u) {
    dim3 grid(n_tokens, (n_head + 31u) / 32u, 1);
    attention_indexed_mixed_heads8_online_kernel<8, 32>
        <<<grid, 1024, 0, stream>>>(heads, sinks, q, raw_kv, comp_kv, topk,
                                    n_tokens, pos0, n_raw, raw_cap, raw_start,
                                    n_comp, top_k, window, ratio, n_head,
                                    head_dim);
    return;
  }
  dim3 grid(n_tokens, (n_head + 15u) / 16u, 1);
  attention_indexed_mixed_heads8_online_kernel<8, 16><<<grid, 512, 0, stream>>>(
      heads, sinks, q, raw_kv, comp_kv, topk, n_tokens, pos0, n_raw, raw_cap,
      raw_start, n_comp, top_k, window, ratio, n_head, head_dim);
}

void AttentionIndexedMixed(float* heads, const float* sinks, const float* q,
                           const float* raw_kv, const float* comp_kv,
                           const int32_t* topk, uint32_t n_tokens,
                           uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
                           uint32_t raw_start, uint32_t n_comp, uint32_t top_k,
                           uint32_t window, uint32_t ratio, uint32_t n_head,
                           uint32_t head_dim, hipStream_t stream) {
  dim3 grid(n_tokens, n_head, 1);
  attention_indexed_mixed_kernel<<<grid, 256, 0, stream>>>(
      heads, sinks, q, raw_kv, comp_kv, topk, n_tokens, pos0, n_raw, raw_cap,
      raw_start, n_comp, top_k, window, ratio, n_head, head_dim);
}

void AttentionStaticOnline(float* heads, const float* sinks, const float* q,
                           const float* raw_kv, const float* comp_kv,
                           uint32_t n_tokens, uint32_t n_comp, uint32_t window,
                           uint32_t ratio, uint32_t n_head, uint32_t head_dim,
                           hipStream_t stream) {
  dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
  attention_static_mixed_heads8_online_kernel<<<grid, 256, 0, stream>>>(
      heads, sinks, q, raw_kv, comp_kv, n_tokens, n_comp, window, ratio, n_head,
      head_dim);
}

void AttentionPrefillRawSoftmax(float* scores, const float* sinks,
                                uint32_t n_tokens, uint32_t window,
                                uint32_t n_keys, uint32_t n_head,
                                hipStream_t stream) {
  dim3 grid(n_tokens, n_head, 1);
  attention_prefill_raw_softmax_kernel<<<grid, 256, 0, stream>>>(
      scores, sinks, n_tokens, window, n_keys);
}

void AttentionPrefillUnpackHeads(float* heads, const float* tmp,
                                 uint32_t n_tokens, uint32_t n_head,
                                 uint32_t head_dim, hipStream_t stream) {
  const uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
  attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      heads, tmp, n_tokens, n_head, head_dim);
}

bool AttentionPrefillRaw(float* heads, const float* sinks, const float* q,
                         const float* raw_kv, uint32_t n_tokens,
                         uint32_t window, uint32_t n_head, uint32_t head_dim,
                         hipStream_t stream) {
  // The kernel holds one score per visible row, at most 256.
  if (window > 256u || (window == 0u && n_tokens > 256u)) return false;
  dim3 grid(n_tokens, n_head, 1);
  attention_prefill_raw_kernel<<<grid, 128, 0, stream>>>(
      heads, sinks, q, raw_kv, n_tokens, window, n_head, head_dim);
  return true;
}

void AttentionPrefillPackMixedKv(float* dst, const float* raw_kv,
                                 const float* comp_kv, uint32_t n_tokens,
                                 uint32_t n_comp, uint32_t head_dim,
                                 hipStream_t stream) {
  const uint64_t count = (uint64_t)(n_tokens + n_comp) * head_dim;
  attention_prefill_pack_mixed_kv_kernel<<<(count + 255) / 256, 256, 0,
                                           stream>>>(dst, raw_kv, comp_kv,
                                                     n_tokens, n_comp,
                                                     head_dim);
}

void AttentionPrefillMixedSoftmax(float* scores, const float* sinks,
                                  const float* comp_mask, uint32_t n_tokens,
                                  uint32_t n_comp, uint32_t window,
                                  uint32_t ratio, uint32_t n_keys,
                                  uint32_t n_head, hipStream_t stream) {
  dim3 grid(n_tokens, n_head, 1);
  attention_prefill_mixed_softmax_kernel<<<grid, 256, 0, stream>>>(
      scores, sinks, comp_mask, comp_mask != nullptr ? 1u : 0u, n_tokens,
      n_comp, window, ratio, n_keys);
}

void AttentionPrefillMixedSoftmaxTile(float* scores, const float* sinks,
                                      const float* comp_mask,
                                      uint32_t raw_tokens, uint32_t tile_start,
                                      uint32_t tile_tokens, uint32_t n_comp,
                                      uint32_t window, uint32_t ratio,
                                      uint32_t n_keys, uint32_t n_head,
                                      hipStream_t stream) {
  dim3 grid(tile_tokens, n_head, 1);
  attention_prefill_mixed_softmax_tile_kernel<<<grid, 256, 0, stream>>>(
      scores, sinks, comp_mask, comp_mask != nullptr ? 1u : 0u, raw_tokens,
      tile_start, tile_tokens, n_comp, window, ratio, n_keys);
}

void AttentionPrefillMixed(float* heads, const float* sinks, const float* q,
                           const float* raw_kv, const float* comp_kv,
                           const float* comp_mask, uint32_t n_tokens,
                           uint32_t n_comp, uint32_t window, uint32_t ratio,
                           uint32_t n_head, uint32_t head_dim,
                           hipStream_t stream) {
  dim3 grid(n_tokens, n_head, 1);
  attention_prefill_mixed_kernel<<<grid, 256, 0, stream>>>(
      heads, sinks, q, raw_kv, comp_kv, comp_mask,
      comp_mask != nullptr ? 1u : 0u, n_tokens, n_comp, window, ratio, n_head,
      head_dim);
}

// The attention output projection: A maps each group of heads to a rank
// block (grouped Q8_0 rows), B the concatenated blocks to the embedding.

template <uint32_t NT>
static void LaunchAttnOutAPreqReuse(float* low, const unsigned char* w,
                                    const int8_t* xq, const float* xscale,
                                    uint32_t group_dim, uint32_t rank,
                                    uint32_t n_groups, uint32_t n_tokens,
                                    uint32_t rows_per_block,
                                    hipStream_t stream) {
  const uint64_t low_dim = (uint64_t)n_groups * rank;
  const uint64_t blocks = group_dim / 32u;
  const unsigned grid =
      (unsigned)((low_dim + rows_per_block - 1u) / rows_per_block);
  grouped_q8_0_a_preq_batch_reuse_w32_kernel<NT, true>
      <<<grid, rows_per_block * 32u, 0, stream>>>(low, w, xq, xscale,
                                                  group_dim, rank, n_groups,
                                                  n_tokens, blocks,
                                                  rows_per_block);
}

bool AttnOutAPreqReuse(float* low, const void* w, const void* xq,
                       const float* xscale, uint32_t group_dim, uint32_t rank,
                       uint32_t n_groups, uint32_t n_tokens,
                       uint32_t rows_per_block, hipStream_t stream) {
  const auto* wb = static_cast<const unsigned char*>(w);
  const auto* q = static_cast<const int8_t*>(xq);
  switch (n_tokens) {
#define DS4_ATTN_OUT_A_REUSE(NT)                                           \
  case NT:                                                                 \
    LaunchAttnOutAPreqReuse<NT>(low, wb, q, xscale, group_dim, rank,       \
                                n_groups, n_tokens, rows_per_block, stream); \
    return true;
    DS4_ATTN_OUT_A_REUSE(2u)
    DS4_ATTN_OUT_A_REUSE(3u)
    DS4_ATTN_OUT_A_REUSE(4u)
    DS4_ATTN_OUT_A_REUSE(5u)
    DS4_ATTN_OUT_A_REUSE(6u)
    DS4_ATTN_OUT_A_REUSE(7u)
    DS4_ATTN_OUT_A_REUSE(8u)
    DS4_ATTN_OUT_A_REUSE(10u)
    DS4_ATTN_OUT_A_REUSE(12u)
    DS4_ATTN_OUT_A_REUSE(16u)
    DS4_ATTN_OUT_A_REUSE(18u)
    DS4_ATTN_OUT_A_REUSE(24u)
    DS4_ATTN_OUT_A_REUSE(32u)
#undef DS4_ATTN_OUT_A_REUSE
    default:
      return false;
  }
}

void AttnOutASharedx(float* low, const void* w, const float* heads,
                     uint32_t group_dim, uint32_t rank, uint32_t n_groups,
                     uint32_t n_tokens, hipStream_t stream) {
  // The token tile follows the rows present, as in the dense Q8 path.
  const uint32_t blocks = group_dim / 32u;
  const uint32_t tile = n_tokens >= 32u ? 32u
                        : n_tokens > 8u ? 16u
                        : n_tokens > 4u ? 8u
                        : n_tokens > 2u ? 4u
                                        : 2u;
  const uint32_t rows_per_block = 32u;
  const uint32_t row_blocks = (rank + rows_per_block - 1u) / rows_per_block;
  const dim3 grid(n_groups * row_blocks, (n_tokens + tile - 1u) / tile, 1u);
  const auto* wb = static_cast<const unsigned char*>(w);
  const uint64_t row_bytes = (uint64_t)blocks * 34u;
  const size_t shmem = (size_t)tile * 16u * 32u * sizeof(float);
  switch (tile) {
#define DS4_ATTN_OUT_A_SHAREDX(TILE)                                         \
  case TILE:                                                                 \
    grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<TILE, 16u>           \
        <<<grid, rows_per_block * 32u, shmem, stream>>>(                     \
            low, wb, heads, n_tokens, n_groups, blocks, rank, row_bytes);    \
    break;
    DS4_ATTN_OUT_A_SHAREDX(2u)
    DS4_ATTN_OUT_A_SHAREDX(4u)
    DS4_ATTN_OUT_A_SHAREDX(8u)
    DS4_ATTN_OUT_A_SHAREDX(16u)
    DS4_ATTN_OUT_A_SHAREDX(32u)
#undef DS4_ATTN_OUT_A_SHAREDX
    default:
      break;
  }
}

void AttnOutAWarp8(float* low, const void* w, const float* heads,
                   uint32_t group_dim, uint32_t rank, uint32_t n_groups,
                   uint32_t n_tokens, hipStream_t stream) {
  const uint64_t low_dim = (uint64_t)n_groups * rank;
  const uint64_t blocks = (group_dim + 31u) / 32u;
  dim3 grid((unsigned)((low_dim + 7u) / 8u), n_tokens, 1);
  grouped_q8_0_a_f32_batch_warp8_kernel<<<grid, 256, 0, stream>>>(
      low, static_cast<const unsigned char*>(w), heads, group_dim, rank,
      n_groups, n_tokens, blocks);
}

void AttnOutAPreqWarp8(float* low, const void* w, const void* xq,
                       const float* xscale, uint32_t group_dim, uint32_t rank,
                       uint32_t n_groups, uint32_t n_tokens,
                       uint32_t rows_per_block, hipStream_t stream) {
  // The kernel reads its rows per block from the block width.
  const uint64_t low_dim = (uint64_t)n_groups * rank;
  const uint64_t blocks = (group_dim + 31u) / 32u;
  dim3 grid((unsigned)((low_dim + rows_per_block - 1u) / rows_per_block),
            n_tokens, 1);
  grouped_q8_0_a_preq_warp8_kernel<<<grid, rows_per_block * 32u, 0, stream>>>(
      low, static_cast<const unsigned char*>(w),
      static_cast<const int8_t*>(xq), xscale, group_dim, rank, n_groups,
      n_tokens, blocks, 1);
}

void PackGroupHeadsF16(__half* dst, const float* heads, uint32_t n_tokens,
                       uint32_t n_groups, uint32_t group_dim,
                       hipStream_t stream) {
  const uint64_t count = (uint64_t)n_groups * n_tokens * group_dim;
  if ((group_dim & 3u) == 0u && ((uintptr_t)heads & 15u) == 0u &&
      ((uintptr_t)dst & 7u) == 0u) {
    const uint64_t groups4 = count >> 2u;
    attention_pack_group_heads_f16_vec4_kernel<<<
        (uint32_t)((groups4 + 255u) / 256u), 256, 0, stream>>>(
        dst, heads, n_tokens, n_groups, group_dim);
    return;
  }
  attention_pack_group_heads_f16_kernel<<<(uint32_t)((count + 255u) / 256u),
                                          256, 0, stream>>>(
      dst, heads, n_tokens, n_groups, group_dim);
}

bool PackGroupHeadsRopeF16(__half* dst, const float* heads, uint32_t n_tokens,
                           uint32_t n_groups, uint32_t group_dim,
                           uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
                           uint32_t n_ctx_orig, float freq_base,
                           float freq_scale, float ext_factor,
                           float attn_factor, float beta_fast, float beta_slow,
                           hipStream_t stream) {
  // The pack walks each heads row as float4, so a rotated pair never
  // straddles a vector and the unrotated prefix is vector aligned.
  const ds4_attn_pack_rope rope = {
      head_dim,   n_rot,      pos0,        n_ctx_orig, /*inverse=*/1,
      freq_base,  freq_scale, ext_factor,  attn_factor, beta_fast,
      beta_slow,
  };
  const uint64_t count = (uint64_t)n_groups * n_tokens * group_dim;
  if (count == 0u || !ds4_attn_pack_rope_eligible(&rope, group_dim) ||
      (group_dim & 3u) != 0u || ((uintptr_t)heads & 15u) != 0u ||
      ((uintptr_t)dst & 7u) != 0u) {
    return false;
  }
  const uint64_t groups4 = count >> 2u;
  attention_pack_group_heads_rope_f16_vec4_kernel<<<
      (uint32_t)((groups4 + 255u) / 256u), 256, 0, stream>>>(
      dst, heads, n_tokens, n_groups, group_dim, head_dim, n_rot, pos0,
      n_ctx_orig, 1, freq_base, freq_scale, ext_factor, attn_factor, beta_fast,
      beta_slow);
  return true;
}

void AttnUnpackGroupLow(float* low, const float* packed, uint32_t n_tokens,
                        uint32_t n_groups, uint32_t rank, hipStream_t stream) {
  const uint64_t count = (uint64_t)n_groups * n_tokens * rank;
  attention_unpack_group_low_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
      low, packed, n_tokens, n_groups, rank);
}

void DequantQ8ToF16Transposed(__half* out, const void* w, uint32_t in_dim,
                              uint32_t out_dim, hipStream_t stream) {
  const uint64_t blocks = (in_dim + 31u) / 32u;
  const dim3 grid(
      (unsigned)((out_dim + DS4_Q8_T_TILE_ROW - 1u) / DS4_Q8_T_TILE_ROW),
      (unsigned)blocks, 1u);
  dequant_q8_0_to_f16_transpose_tiled_kernel<<<grid, 256, 0, stream>>>(
      out, static_cast<const unsigned char*>(w), in_dim, out_dim, blocks);
}

void GemmF16WmmaBatched(__half* out, const __half* weight, const __half* act,
                        uint32_t m, uint32_t n, uint32_t k, uint32_t ldc,
                        uint32_t batches, hipStream_t stream) {
  const dim3 grid(m / DS4_WMMA_BATCH_BM, n / DS4_WMMA_BATCH_BN, batches);
  ds4_gemm_f16_wmma_kernel<DS4_WMMA_BATCH_BM, DS4_WMMA_BATCH_BN,
                           DS4_WMMA_GEMM_BK, 2u, 2u, 1u, true, __half>
      <<<grid, 512u, 0, stream>>>(out, weight, act, m, n, k, ldc,
                                  (uint64_t)m * k, (uint64_t)n * k, m);
}

void GemmF16WmmaPadded(float* out, const __half* weight, const __half* act,
                       uint32_t m, uint32_t n_pad, uint32_t k,
                       hipStream_t stream) {
  // 128x128 below 1,024 columns, 256x128 from there; both walk K once in
  // ascending order, so the choice partitions the output only.
  if (n_pad >= DS4_WMMA_GEMM_WIDE_MIN_N && m % DS4_WMMA_GEMM_BM == 0u &&
      n_pad % DS4_WMMA_GEMM_BN == 0u) {
    const dim3 grid(m / DS4_WMMA_GEMM_BM, n_pad / DS4_WMMA_GEMM_BN, 1u);
    ds4_gemm_f16_wmma_kernel<DS4_WMMA_GEMM_BM, DS4_WMMA_GEMM_BN,
                             DS4_WMMA_GEMM_BK, 4u, 2u, 2u, false, float>
        <<<grid, 512u, 0, stream>>>(out, weight, act, m, n_pad, k, m, 0, 0, 0);
    return;
  }
  constexpr uint32_t kNarrowWmf = 2u;
  constexpr uint32_t kNarrowWnf = 2u;
  constexpr uint32_t kNarrowThreads =
      (DS4_WMMA_GEMM_NARROW_BM / 16u / kNarrowWmf) *
      (DS4_WMMA_GEMM_NARROW_BN / 16u / kNarrowWnf) * 32u;
  const dim3 grid(m / DS4_WMMA_GEMM_NARROW_BM, n_pad / DS4_WMMA_GEMM_NARROW_BN,
                  1u);
  ds4_gemm_f16_wmma_kernel<DS4_WMMA_GEMM_NARROW_BM, DS4_WMMA_GEMM_NARROW_BN,
                           DS4_WMMA_GEMM_BK, kNarrowWmf, kNarrowWnf, 8u, false,
                           float>
      <<<grid, kNarrowThreads, 0, stream>>>(out, weight, act, m, n_pad, k, m, 0,
                                            0, 0);
}

void ZeroBytes(void* x, std::size_t count, hipStream_t stream) {
  (void)hipMemsetAsync(x, 0, count, stream);
}

void IndexerScoreDirect(float* scores, const float* q, const float* weights,
                        const float* index_comp, uint32_t n_comp,
                        uint32_t n_tokens, uint32_t pos0, uint32_t ratio,
                        float scale, bool causal, hipStream_t stream) {
  indexer_score_one_direct_kernel<<<dim3(n_comp, n_tokens), 128, 0, stream>>>(
      scores, q, weights, index_comp, n_comp, pos0, ratio, scale,
      causal ? 1 : 0);
}

void IndexerPackKeys(__half* packed, const float* keys, uint32_t count,
                     hipStream_t stream) {
  const uint64_t padded = ((uint64_t(count) + 127u) / 128u) * 128u * 128u;
  indexer_pack_keys_kernel<<<(padded + 255u) / 256u, 256, 0, stream>>>(
      keys, packed, count, padded);
}

void IndexerPackQueries(__half* packed, const float* q, uint32_t tokens,
                        hipStream_t stream) {
  indexer_pack_queries_kernel<<<dim3(64, (tokens + 15u) / 16u), 256, 0,
                                stream>>>(q, packed, tokens);
}

void IndexerScoresWmmaPacked(float* scores, const __half* queries,
                             const float* weights, const __half* keys,
                             uint32_t n_comp, uint32_t n_tokens, uint32_t pos0,
                             uint32_t ratio, float scale, bool causal,
                             hipStream_t stream) {
  const dim3 grid((n_comp + 127u) / 128u, (n_tokens + 15u) / 16u, 1);
  indexer_scores_wmma128_kernel<true><<<grid, 256, 0, stream>>>(
      scores, queries, weights, keys, n_comp, n_tokens, pos0, 64u, 128u, ratio,
      scale, causal ? 1 : 0);
}

void IndexerScoresWmma(float* scores, const float* q, const float* weights,
                       const float* index_comp, uint32_t n_comp,
                       uint32_t n_tokens, uint32_t pos0, uint32_t ratio,
                       float scale, bool causal, hipStream_t stream) {
  const dim3 grid((n_comp + 127u) / 128u, (n_tokens + 15u) / 16u, 1);
  indexer_scores_wmma128_kernel<false><<<grid, 256, 0, stream>>>(
      scores, q, weights, index_comp, n_comp, n_tokens, pos0, 64u, 128u, ratio,
      scale, causal ? 1 : 0);
}

void IndexerScores(float* scores, const float* q, const float* weights,
                   const float* index_comp, uint32_t n_comp, uint32_t n_tokens,
                   uint32_t pos0, uint32_t n_head, uint32_t head_dim,
                   uint32_t ratio, float scale, bool causal,
                   hipStream_t stream) {
  indexer_scores_kernel<<<dim3(n_comp, n_tokens, 1), 256, 0, stream>>>(
      scores, q, weights, index_comp, n_comp, n_tokens, pos0, n_head, head_dim,
      ratio, scale, causal ? 1 : 0);
}

void IndexerTopkPartial(int32_t* selected, const float* scores,
                        uint32_t n_comp, uint32_t n_tokens,
                        hipStream_t stream) {
  auto* out = reinterpret_cast<uint32_t*>(selected);
  if (n_comp <= 8192u)
    indexer_partial_topk_kernel<8192>
        <<<n_tokens, 256, 0, stream>>>(out, scores, n_comp, n_tokens);
  else if (n_comp <= 16384u)
    indexer_partial_topk_kernel<16384>
        <<<n_tokens, 256, 0, stream>>>(out, scores, n_comp, n_tokens);
  else
    indexer_partial_topk_kernel<32768, 512>
        <<<n_tokens, 512, 0, stream>>>(out, scores, n_comp, n_tokens);
}

void IndexerTopk1024(int32_t* selected, const float* scores, uint32_t n_comp,
                     uint32_t n_tokens, uint32_t top_k, hipStream_t stream) {
  indexer_topk_1024_kernel<<<n_tokens, 1024, 0, stream>>>(
      reinterpret_cast<uint32_t*>(selected), scores, n_comp, n_tokens, top_k);
}

void IndexerTopk2048(int32_t* selected, const float* scores, uint32_t n_comp,
                     uint32_t n_tokens, uint32_t top_k, hipStream_t stream) {
  indexer_topk_pow2_kernel<2048><<<n_tokens, 1024, 0, stream>>>(
      reinterpret_cast<uint32_t*>(selected), scores, n_comp, n_tokens, top_k);
}

void IndexerTopk4096(int32_t* selected, const float* scores, uint32_t n_comp,
                     uint32_t n_tokens, uint32_t top_k, hipStream_t stream) {
  indexer_topk_pow2_kernel<4096><<<n_tokens, 1024, 0, stream>>>(
      reinterpret_cast<uint32_t*>(selected), scores, n_comp, n_tokens, top_k);
}

void IndexerTopk8192(int32_t* selected, const float* scores, uint32_t n_comp,
                     uint32_t n_tokens, uint32_t top_k, hipStream_t stream) {
  indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024, 0, stream>>>(
      reinterpret_cast<uint32_t*>(selected), scores, n_comp, n_tokens, top_k);
}

bool IndexerTopk8192Cub(int32_t* selected, const float* scores,
                        uint32_t n_comp, uint32_t n_tokens, uint32_t top_k,
                        hipStream_t stream) {
  using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
  constexpr int kSmem = (int)sizeof(typename TopkCubSort::TempStorage);
  // The opt-in limit and the kernel attribute do not change: settled once
  // instead of on every launch.
  static const bool ready = [] {
    int dev = 0;
    int max_optin_smem = 0;
    hipError_t err = hipGetDevice(&dev);
    if (err == hipSuccess)
      err = hipDeviceGetAttribute(
          &max_optin_smem, hipDeviceAttributeSharedMemPerBlockOptin, dev);
    if (err == hipSuccess && max_optin_smem >= kSmem)
      err = ds4_hip_set_dynamic_shared_memory(indexer_topk_8192_cub_kernel,
                                              kSmem);
    else if (err == hipSuccess)
      return false;
    if (err != hipSuccess) {
      (void)hipGetLastError();
      return false;
    }
    return true;
  }();
  if (!ready) return false;
  indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)kSmem, stream>>>(
      reinterpret_cast<uint32_t*>(selected), scores, n_comp, n_tokens, top_k);
  return true;
}

void IndexerTopkChunk(int32_t* candidates, const float* scores,
                      uint32_t n_comp, uint32_t n_tokens, uint32_t top_k,
                      uint32_t candidate_stride, hipStream_t stream) {
  const dim3 grid(n_tokens, (n_comp + 4095u) / 4096u, 1);
  indexer_topk_chunk_pow2_kernel<4096><<<grid, 1024, 0, stream>>>(
      reinterpret_cast<uint32_t*>(candidates), scores, n_comp, n_tokens, top_k,
      candidate_stride);
}

void IndexerTopkTreeMerge(int32_t* out, const int32_t* candidates,
                          const float* scores, uint32_t n_comp,
                          uint32_t n_tokens, uint32_t top_k, uint32_t n_sets,
                          uint32_t merge_group, uint32_t candidate_stride,
                          uint32_t out_stride, hipStream_t stream) {
  const dim3 grid(n_tokens, (n_sets + merge_group - 1u) / merge_group, 1);
  indexer_topk_tree_merge_pow2_kernel<4096><<<grid, 1024, 0, stream>>>(
      reinterpret_cast<uint32_t*>(out),
      reinterpret_cast<const uint32_t*>(candidates), scores, n_comp, n_tokens,
      top_k, n_sets, merge_group, candidate_stride, out_stride);
}

void IndexerTopkMerge(int32_t* selected, const int32_t* candidates,
                      const float* scores, uint32_t n_comp, uint32_t n_tokens,
                      uint32_t top_k, uint32_t candidate_count,
                      uint32_t candidate_stride, hipStream_t stream) {
  indexer_topk_merge_pow2_kernel<4096><<<n_tokens, 1024, 0, stream>>>(
      reinterpret_cast<uint32_t*>(selected),
      reinterpret_cast<const uint32_t*>(candidates), scores, n_comp, n_tokens,
      top_k, candidate_count, candidate_stride);
}

void IndexerTopkSerial(int32_t* selected, const float* scores, uint32_t n_comp,
                       uint32_t n_tokens, uint32_t top_k, hipStream_t stream) {
  indexer_topk_kernel<<<n_tokens, 1, 0, stream>>>(
      reinterpret_cast<uint32_t*>(selected), scores, n_comp, n_tokens, top_k);
}

void RouterSelect(int32_t* selected, float* weights, float* probs,
                  const float* bias, const int32_t* hash, const float* logits,
                  const int32_t* tokens, uint32_t token, uint32_t hash_rows,
                  uint32_t n_tokens, hipStream_t stream) {
  // Hash-routed layers take no bias.
  const bool hash_mode = hash != nullptr;
  const float* used_bias = hash_mode ? nullptr : bias;
  const dim3 block(32, 4, 1);
  router_select_warp_topk_kernel<<<(n_tokens + 3u) / 4u, block, 0, stream>>>(
      selected, weights, probs, used_bias, hash, logits, tokens,
      (int32_t)token, hash_rows, n_tokens, used_bias != nullptr ? 1 : 0,
      hash_mode ? 1 : 0);
}

// --- Routed experts: the IQ2_XXS gate/up and Q2_K down route ----------------

namespace {
// Bytes per 256 values of the routed expert formats.
constexpr uint64_t kIq2XxsBlock = 66u;
constexpr uint64_t kQ2KBlock = 84u;
constexpr uint32_t kUsed = DS4_ROCM_N_EXPERT_USED;
}  // namespace

void MoeQuantizeQ8K(void* xq, const float* x, uint32_t dim, uint32_t rows,
                    hipStream_t stream) {
  const dim3 grid(dim / ROCM_QK_K, rows, 1);
  q8_K_quantize_kernel<<<grid, 256, 0, stream>>>(
      static_cast<hip_block_q8_K*>(xq), x, dim, rows);
}

void MoeRoutePlan(const int32_t* selected, uint32_t* counts, uint32_t* offsets,
                  int32_t* cursors, uint32_t* sorted, int32_t* tile_offsets,
                  uint32_t* tile_total, uint32_t* tile_experts,
                  uint32_t* tile_starts, uint32_t n_tokens,
                  uint32_t tile_shift, hipStream_t stream) {
  const uint32_t pairs = n_tokens * kUsed;
  const uint32_t tile_m = 1u << tile_shift;
  (void)hipMemsetAsync(counts, 0, DS4_ROCM_N_EXPERT * sizeof(uint32_t),
                       stream);
  moe_count_sorted_pairs_kernel<<<(pairs + 255u) / 256u, 256, 0, stream>>>(
      counts, selected, pairs);
  moe_prefix_sorted_pairs_kernel<<<1, 1, 0, stream>>>(
      offsets, reinterpret_cast<uint32_t*>(cursors), counts);
  moe_scatter_sorted_pairs_deterministic_kernel<<<
      256u, 256u, 256u * sizeof(uint32_t), stream>>>(sorted, offsets, selected,
                                                     pairs);
  moe_build_expert_tile_offsets_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<uint32_t*>(tile_offsets), tile_total, counts, tile_m);
  moe_build_expert_tiles_kernel<<<1, 256, 0, stream>>>(
      tile_experts, tile_starts, reinterpret_cast<uint32_t*>(tile_offsets),
      counts, tile_m);
}

void MoeRoutePlanNarrow(const int32_t* selected, uint32_t* counts,
                        uint32_t* offsets, uint32_t* sorted,
                        int32_t* tile_offsets, uint32_t* tile_total,
                        uint32_t* tile_experts, uint32_t* tile_starts,
                        uint32_t n_tokens, uint32_t tile_shift,
                        hipStream_t stream) {
  moe_build_narrow_sorted_tiles_kernel<<<1, 256, 0, stream>>>(
      counts, offsets, sorted, reinterpret_cast<uint32_t*>(tile_offsets),
      tile_total, tile_experts, tile_starts, selected, n_tokens * kUsed,
      1u << tile_shift);
}

void MoeRouteTiles(const int32_t* selected, const uint32_t* counts,
                   int32_t* tile_offsets, uint32_t* tile_total,
                   uint32_t* tile_experts, uint32_t* tile_starts,
                   uint32_t tile_shift, hipStream_t stream) {
  (void)selected;
  const uint32_t tile_m = 1u << tile_shift;
  moe_build_expert_tile_offsets_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<uint32_t*>(tile_offsets), tile_total, counts, tile_m);
  moe_build_expert_tiles_kernel<<<1, 256, 0, stream>>>(
      tile_experts, tile_starts, reinterpret_cast<uint32_t*>(tile_offsets),
      counts, tile_m);
}

void MoeIq2GateUpLut(float* mid, const void* gate_w, const void* up_w,
                     const void* xq, const int32_t* selected,
                     const float* weights, uint32_t in_dim, uint32_t mid_dim,
                     uint32_t n_tokens, float clamp, hipStream_t stream) {
  const uint64_t row = (in_dim / ROCM_QK_K) * kIq2XxsBlock;
  const dim3 grid((mid_dim + 127u) / 128u, n_tokens * kUsed, 1);
  moe_gate_up_mid_decode_lut_qwarp32_kernel<<<grid, 256, 0, stream>>>(
      nullptr, nullptr, mid, static_cast<const char*>(gate_w),
      static_cast<const char*>(up_w), static_cast<const hip_block_q8_K*>(xq),
      selected, weights, row * mid_dim, row, in_dim / ROCM_QK_K, mid_dim,
      kUsed, 0u, clamp);
}

void MoeIq2GateUpQwarp(float* gate, float* up, float* mid, const void* gate_w,
                       const void* up_w, const void* xq,
                       const int32_t* selected, const float* weights,
                       uint32_t in_dim, uint32_t mid_dim, uint32_t n_tokens,
                       float clamp, hipStream_t stream) {
  const uint64_t row = (in_dim / ROCM_QK_K) * kIq2XxsBlock;
  const dim3 grid((mid_dim + 127u) / 128u, n_tokens * kUsed, 1);
  moe_gate_up_mid_qwarp32_kernel<<<grid, 256, 0, stream>>>(
      gate, up, mid, static_cast<const char*>(gate_w),
      static_cast<const char*>(up_w), static_cast<const hip_block_q8_K*>(xq),
      selected, weights, row * mid_dim, row, in_dim / ROCM_QK_K, mid_dim,
      kUsed, clamp);
}

void MoeIq2GateUpTiles(float* mid, const void* gate_w, const void* up_w,
                       const void* xq, const int32_t* selected,
                       const uint32_t* sorted, const uint32_t* offsets,
                       const uint32_t* counts, const uint32_t* tile_total,
                       const uint32_t* tile_experts,
                       const uint32_t* tile_starts, const float* weights,
                       uint32_t in_dim, uint32_t mid_dim, uint32_t tile_shift,
                       uint32_t tile_cap, uint32_t max_count, float clamp,
                       hipStream_t stream) {
  (void)selected;
  const uint64_t row = (in_dim / ROCM_QK_K) * kIq2XxsBlock;
  const dim3 grid((mid_dim + 31u) / 32u, tile_cap, 1);
  if (tile_shift == 3u) {
    moe_gate_up_mid_expert_tile8_row32_kernel<<<grid, 256, 0, stream>>>(
        nullptr, nullptr, mid, static_cast<const char*>(gate_w),
        static_cast<const char*>(up_w), static_cast<const hip_block_q8_K*>(xq),
        sorted, offsets, counts, tile_total, tile_experts, tile_starts,
        weights, row * mid_dim, row, in_dim / ROCM_QK_K, mid_dim, kUsed,
        max_count, 0u, clamp);
    return;
  }
  // Direct cached reads avoid 18,688 bytes of LDS and its copy/barrier.
  // Paired gfx1151 trials are faster through 136 rows and match every
  // scalar gate/up/activation result exactly.
  moe_gate_up_mid_expert_tile4_row32_kernel<<<grid, 256, 0, stream>>>(
      nullptr, nullptr, mid, static_cast<const char*>(gate_w),
      static_cast<const char*>(up_w), static_cast<const hip_block_q8_K*>(xq),
      sorted, offsets, counts, tile_total, tile_experts, tile_starts, weights,
      row * mid_dim, row, in_dim / ROCM_QK_K, mid_dim, kUsed, max_count, 0u,
      clamp);
}

void MoeIq2GateUpTilesWide(float* mid, const void* gate_w, const void* up_w,
                           const void* xq, const int32_t* selected,
                           const uint32_t* sorted, const uint32_t* offsets,
                           const uint32_t* counts, const uint32_t* tile_total,
                           const uint32_t* tile_experts,
                           const uint32_t* tile_starts, const float* weights,
                           uint32_t in_dim, uint32_t mid_dim,
                           uint32_t tile_cap, uint32_t max_count, float clamp,
                           hipStream_t stream) {
  (void)selected;
  const uint64_t row = (in_dim / ROCM_QK_K) * kIq2XxsBlock;
  const dim3 grid((mid_dim + 1023u) / 1024u, tile_cap, 1);
  moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<grid, 256, 0, stream>>>(
      nullptr, nullptr, mid, static_cast<const char*>(gate_w),
      static_cast<const char*>(up_w), static_cast<const hip_block_q8_K*>(xq),
      sorted, offsets, counts, tile_total, tile_experts, tile_starts, weights,
      row * mid_dim, row, in_dim / ROCM_QK_K, mid_dim, kUsed, max_count, 0u,
      clamp);
}

namespace {
template <bool OutF16>
void Iq2GateUpHot(float* mid, __half* mid_h, const void* gate_w,
                  const void* up_w, const float* x, const __half* x_h,
                  const float* weights, const uint32_t* counts,
                  const uint32_t* offsets, const uint32_t* sorted,
                  const uint32_t* hot_experts, uint32_t hot_count,
                  uint32_t hot_max, uint32_t in_dim, uint32_t mid_dim,
                  float clamp, hipStream_t stream) {
  constexpr uint32_t bm = 16u, bn = 16u, bk = 16u, mt = 4u;
  const uint64_t row = (in_dim / ROCM_QK_K) * kIq2XxsBlock;
  const dim3 block(32u * mt, 1u, 1u);
  const dim3 grid((mid_dim + 2u * bn - 1u) / (2u * bn),
                  (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
  const size_t shmem = (mt * bm * bk + 4u * bk * bn) * sizeof(__half) +
                       (4u * mt * bm * bn) * sizeof(float);
  if (x_h)
    moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4, 16, 16, 16, OutF16, true>
        <<<grid, block, shmem, stream>>>(
            mid, mid_h, static_cast<const char*>(gate_w),
            static_cast<const char*>(up_w), x, x_h, weights, counts, offsets,
            sorted, hot_experts, hot_count, in_dim, mid_dim, row * mid_dim, row,
            clamp);
  else
    moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4, 16, 16, 16, OutF16>
        <<<grid, block, shmem, stream>>>(
            mid, mid_h, static_cast<const char*>(gate_w),
            static_cast<const char*>(up_w), x, nullptr, weights, counts,
            offsets, sorted, hot_experts, hot_count, in_dim, mid_dim,
            row * mid_dim, row, clamp);
}
}  // namespace

void MoeIq2GateUpHot(float* mid, const void* gate_w, const void* up_w,
                     const float* x, const __half* x_h, const float* weights,
                     const int32_t* selected, const uint32_t* counts,
                     const uint32_t* offsets, const uint32_t* sorted,
                     const uint32_t* hot_experts, uint32_t hot_count,
                     uint32_t hot_max, uint32_t in_dim, uint32_t mid_dim,
                     float clamp, hipStream_t stream) {
  (void)selected;
  Iq2GateUpHot<false>(mid, nullptr, gate_w, up_w, x, x_h, weights, counts,
                      offsets, sorted, hot_experts, hot_count, hot_max, in_dim,
                      mid_dim, clamp, stream);
}

void MoeIq2GateUpHotF16(__half* mid_h, const void* gate_w, const void* up_w,
                        const float* x, const __half* x_h,
                        const float* weights, const int32_t* selected,
                        const uint32_t* counts, const uint32_t* offsets,
                        const uint32_t* sorted, const uint32_t* hot_experts,
                        uint32_t hot_count, uint32_t hot_max, uint32_t in_dim,
                        uint32_t mid_dim, float clamp, hipStream_t stream) {
  (void)selected;
  Iq2GateUpHot<true>(nullptr, mid_h, gate_w, up_w, x, x_h, weights, counts,
                     offsets, sorted, hot_experts, hot_count, hot_max, in_dim,
                     mid_dim, clamp, stream);
}

bool MoeIq2MmqGateUpSwiglu(float* gate, float* discard, float* mid,
                           __half* mid_h, void* q81_scratch,
                           std::size_t q81_bytes, const void* gate_w,
                           const void* up_w, const float* x,
                           const int32_t* selected, const float* weights,
                           uint32_t in_dim, uint32_t mid_dim,
                           uint32_t n_tokens, uint32_t max_rows,
                           uint32_t tile_cols, float clamp,
                           hipStream_t stream) {
  ds4_mmq_set_routed_max_expert_rows((int)max_rows);
  ds4_mmq_set_routed_tile_cols((int)tile_cols);
  ds4_mmq_set_aligned_q81_scratch(q81_scratch, q81_bytes);
  const int rc = ds4_mmq_iq2_xxs_moe_pair_token_bound_swiglu(
      gate_w, up_w, x, selected, weights, gate, discard, mid, mid_h,
      (int)mid_dim, (int)in_dim, (int)n_tokens, (int)DS4_ROCM_N_EXPERT,
      (int)kUsed, clamp, stream);
  ds4_mmq_set_aligned_q81_scratch(nullptr, 0);
  ds4_mmq_set_routed_max_expert_rows(0);
  ds4_mmq_set_routed_tile_cols(0);
  return rc == 0;
}

bool MoeIq2MmqGateUp(float* gate, float* up, void* q81_scratch,
                     std::size_t q81_bytes, const void* gate_w,
                     const void* up_w, const float* x, const int32_t* selected,
                     uint32_t in_dim, uint32_t mid_dim, uint32_t n_tokens,
                     uint32_t max_rows, uint32_t tile_cols,
                     hipStream_t stream) {
  ds4_mmq_set_routed_max_expert_rows((int)max_rows);
  ds4_mmq_set_routed_tile_cols((int)tile_cols);
  ds4_mmq_set_aligned_q81_scratch(q81_scratch, q81_bytes);
  const int rc = ds4_mmq_iq2_xxs_moe_pair_token_bound(
      gate_w, up_w, x, selected, gate, up, (int)mid_dim, (int)in_dim,
      (int)n_tokens, (int)DS4_ROCM_N_EXPERT, (int)kUsed, stream);
  ds4_mmq_set_aligned_q81_scratch(nullptr, 0);
  ds4_mmq_set_routed_max_expert_rows(0);
  ds4_mmq_set_routed_tile_cols(0);
  return rc == 0;
}

void MoeMmqSwiglu(float* mid, __half* mid_h, const float* gate,
                  const float* up, const float* weights, uint32_t mid_dim,
                  uint32_t n_tokens, float clamp, hipStream_t stream) {
  const uint64_t count = (uint64_t)n_tokens * kUsed * mid_dim;
  moe_mmq_swiglu_weighted_clamp_kernel<<<(uint32_t)((count + 255u) / 256u),
                                         256, 0, stream>>>(
      mid, mid_h, gate, up, weights, mid_dim, n_tokens, kUsed, clamp);
}

void MoeQ2DownSum6(float* out, const void* down_w, const void* midq,
                   const int32_t* selected, uint32_t mid_dim, uint32_t out_dim,
                   uint32_t n_tokens, hipStream_t stream) {
  const uint64_t row = (mid_dim / ROCM_QK_K) * kQ2KBlock;
  const dim3 grid((out_dim + 31u) / 32u, n_tokens, 1);
  moe_down_sum6_qwarp32_kernel<<<grid, 256, 0, stream>>>(
      out, static_cast<const char*>(down_w),
      static_cast<const hip_block_q8_K*>(midq), selected, row * out_dim, row,
      mid_dim / ROCM_QK_K, out_dim);
}

namespace {
void Q2DownTiles(float* out, const void* down_w, const void* midq,
                 const uint32_t* sorted, const uint32_t* offsets,
                 const uint32_t* counts, const uint32_t* tile_total,
                 const uint32_t* tile_experts, const uint32_t* tile_starts,
                 uint32_t mid_dim, uint32_t out_dim, uint32_t tile_shift,
                 uint32_t tile_cap, bool wide, uint32_t atomic,
                 hipStream_t stream) {
  const uint64_t row = (mid_dim / ROCM_QK_K) * kQ2KBlock;
  const char* w = static_cast<const char*>(down_w);
  const auto* q = static_cast<const hip_block_q8_K*>(midq);
  const uint32_t blocks = mid_dim / ROCM_QK_K;
  if (wide) {
    const dim3 grid((out_dim + 2047u) / 2048u, tile_cap, 1);
    moe_down_expert_tile16_row2048_kernel<<<grid, 256, 0, stream>>>(
        out, w, q, sorted, offsets, counts, tile_total, tile_experts,
        tile_starts, row * out_dim, row, blocks, out_dim, kUsed, atomic);
    return;
  }
  const dim3 grid((out_dim + 31u) / 32u, tile_cap, 1);
  if (tile_shift == 4u)
    moe_down_expert_tile16_row32_kernel<<<grid, 256, 0, stream>>>(
        out, w, q, sorted, offsets, counts, tile_total, tile_experts,
        tile_starts, row * out_dim, row, blocks, out_dim, kUsed, atomic);
  else if (tile_shift == 3u)
    moe_down_expert_tile8_row32_kernel<<<grid, 256, 0, stream>>>(
        out, w, q, sorted, offsets, counts, tile_total, tile_experts,
        tile_starts, row * out_dim, row, blocks, out_dim, kUsed, atomic);
  else
    moe_down_expert_tile4_row32_kernel<<<grid, 256, 0, stream>>>(
        out, w, q, sorted, offsets, counts, tile_total, tile_experts,
        tile_starts, row * out_dim, row, blocks, out_dim, kUsed, atomic);
}
}  // namespace

void MoeQ2DownTiles(float* down, const void* down_w, const void* midq,
                    const int32_t* selected, const uint32_t* sorted,
                    const uint32_t* offsets, const uint32_t* counts,
                    const uint32_t* tile_total, const uint32_t* tile_experts,
                    const uint32_t* tile_starts, uint32_t mid_dim,
                    uint32_t out_dim, uint32_t tile_shift, uint32_t tile_cap,
                    bool wide, hipStream_t stream) {
  (void)selected;
  Q2DownTiles(down, down_w, midq, sorted, offsets, counts, tile_total,
              tile_experts, tile_starts, mid_dim, out_dim, tile_shift,
              tile_cap, wide, 0u, stream);
}

void MoeQ2DownTilesAtomic(float* out, const void* down_w, const void* midq,
                          const int32_t* selected, const uint32_t* sorted,
                          const uint32_t* offsets, const uint32_t* counts,
                          const uint32_t* tile_total,
                          const uint32_t* tile_experts,
                          const uint32_t* tile_starts, uint32_t mid_dim,
                          uint32_t out_dim, uint32_t tile_shift,
                          uint32_t tile_cap, bool wide, hipStream_t stream) {
  (void)selected;
  Q2DownTiles(out, down_w, midq, sorted, offsets, counts, tile_total,
              tile_experts, tile_starts, mid_dim, out_dim, tile_shift,
              tile_cap, wide, 1u, stream);
}

void MoeSum(float* out, const float* down, uint32_t out_dim,
            uint32_t n_tokens, hipStream_t stream) {
  const uint64_t n = (uint64_t)n_tokens * out_dim;
  moe_sum_kernel<<<(n + 255u) / 256u, 256, 0, stream>>>(out, down, out_dim,
                                                       kUsed, n_tokens);
}

void MoeSumF16(float* out, const __half* down_h, uint32_t out_dim,
               uint32_t n_tokens, hipStream_t stream) {
  if ((out_dim & 1u) == 0u) {
    const uint64_t n2 = (uint64_t)n_tokens * (out_dim >> 1u);
    moe_sum_f16x2_kernel<<<(n2 + 255u) / 256u, 256, 0, stream>>>(
        out, down_h, out_dim, kUsed, n_tokens);
    return;
  }
  const uint64_t n = (uint64_t)n_tokens * out_dim;
  moe_sum_f16_kernel<<<(n + 255u) / 256u, 256, 0, stream>>>(
      out, down_h, out_dim, kUsed, n_tokens);
}

namespace {
template <uint32_t PairTile, bool OutF16>
void Q2DownBatch(float* down, __half* down_h, const char* w, const float* mid,
                 const uint32_t* counts, const uint32_t* offsets,
                 const uint32_t* sorted, uint32_t min_count, uint32_t max_count,
                 uint32_t mid_dim, uint32_t out_dim, uint64_t row,
                 const uint32_t* experts, uint32_t n_experts,
                 uint32_t rows_per_block, hipStream_t stream) {
  const dim3 grid((out_dim + rows_per_block - 1u) / rows_per_block, n_experts,
                  1u);
  moe_down_q2K_expert_batch_sharedmid_kernel<PairTile, false, OutF16>
      <<<grid, rows_per_block * 32u, (size_t)PairTile * 256u * sizeof(float),
         stream>>>(down, down_h, w, mid, nullptr, counts, offsets, sorted,
                   min_count, max_count, mid_dim, out_dim, row * out_dim, row,
                   0u, experts);
}

template <bool OutF16>
void Q2DownBatchTile(float* down, __half* down_h, const void* down_w,
                     const float* mid, const uint32_t* counts,
                     const uint32_t* offsets, const uint32_t* sorted,
                     const uint32_t* experts, uint32_t n_experts,
                     uint32_t pair_shift, uint32_t rows_per_block,
                     uint32_t min_count, uint32_t max_count, uint32_t mid_dim,
                     uint32_t out_dim, hipStream_t stream) {
  const uint64_t row = (mid_dim / ROCM_QK_K) * kQ2KBlock;
  const char* w = static_cast<const char*>(down_w);
  if (pair_shift == 0u)
    Q2DownBatch<1, OutF16>(down, down_h, w, mid, counts, offsets, sorted,
                           min_count, max_count, mid_dim, out_dim, row,
                           experts, n_experts, rows_per_block, stream);
  else if (pair_shift == 1u)
    Q2DownBatch<2, OutF16>(down, down_h, w, mid, counts, offsets, sorted,
                           min_count, max_count, mid_dim, out_dim, row,
                           experts, n_experts, rows_per_block, stream);
  else
    Q2DownBatch<4, OutF16>(down, down_h, w, mid, counts, offsets, sorted,
                           min_count, max_count, mid_dim, out_dim, row,
                           experts, n_experts, rows_per_block, stream);
}
}  // namespace

void MoeQ2DownBatch(float* down, const void* down_w, const float* mid,
                    const int32_t* selected, const uint32_t* counts,
                    const uint32_t* offsets, const uint32_t* sorted,
                    const uint32_t* experts, uint32_t n_experts,
                    uint32_t pair_shift, uint32_t rows_per_block,
                    uint32_t min_count, uint32_t max_count, uint32_t mid_dim,
                    uint32_t out_dim, hipStream_t stream) {
  (void)selected;
  Q2DownBatchTile<false>(down, nullptr, down_w, mid, counts, offsets, sorted,
                         experts, n_experts, pair_shift, rows_per_block,
                         min_count, max_count, mid_dim, out_dim, stream);
}

void MoeQ2DownBatchF16(__half* down_h, const void* down_w, const float* mid,
                       const int32_t* selected, const uint32_t* counts,
                       const uint32_t* offsets, const uint32_t* sorted,
                       const uint32_t* experts, uint32_t n_experts,
                       uint32_t pair_shift, uint32_t rows_per_block,
                       uint32_t min_count, uint32_t max_count,
                       uint32_t mid_dim, uint32_t out_dim,
                       hipStream_t stream) {
  (void)selected;
  Q2DownBatchTile<true>(nullptr, down_h, down_w, mid, counts, offsets, sorted,
                        experts, n_experts, pair_shift, rows_per_block,
                        min_count, max_count, mid_dim, out_dim, stream);
}

void MoeQ2DownHotWide(__half* down_h, const void* down_w, const __half* mid_h,
                      const int32_t* selected, const uint32_t* counts,
                      const uint32_t* offsets, const uint32_t* sorted,
                      const uint32_t* hot_experts, uint32_t hot_count,
                      const uint32_t* tile_map, uint32_t n_map,
                      uint32_t hot_max, uint32_t mid_dim, uint32_t out_dim,
                      uint32_t mtiles, hipStream_t stream) {
  (void)selected;
  constexpr uint32_t bm = 16u, bn = 16u, bk = 16u, nfrag = 4u;
  const uint64_t row = (mid_dim / ROCM_QK_K) * kQ2KBlock;
  const char* w = static_cast<const char*>(down_w);
  const uint32_t tile_m = mtiles * bm;
  const dim3 grid(out_dim / (nfrag * bn),
                  tile_map ? n_map : (hot_max + tile_m - 1u) / tile_m,
                  tile_map ? 1u : hot_count);
  if (mtiles == 8u)
    moe_down_q2K_hotlist_wmma_wide_kernel<8, 16, 16, 16, nfrag, true, true>
        <<<grid, 256, ds4_rocm_q2_down_wide_shmem(8, bm, bn, bk, nfrag),
           stream>>>(nullptr, down_h, w, nullptr, mid_h, counts, offsets,
                     sorted, hot_experts, hot_count, mid_dim, out_dim,
                     row * out_dim, row, 0u, tile_map);
  else if (mtiles == 4u)
    moe_down_q2K_hotlist_wmma_wide_kernel<4, 16, 16, 16, nfrag, true, true>
        <<<grid, dim3(32u * 4u, 1u, 1u),
           ds4_rocm_q2_down_wide_shmem(4, bm, bn, bk, nfrag), stream>>>(
            nullptr, down_h, w, nullptr, mid_h, counts, offsets, sorted,
            hot_experts, hot_count, mid_dim, out_dim, row * out_dim, row, 0u,
            tile_map);
  else
    moe_down_q2K_hotlist_wmma_wide_kernel<2, 16, 16, 16, nfrag, true, true>
        <<<grid, dim3(32u * 2u, 1u, 1u),
           ds4_rocm_q2_down_wide_shmem(2, bm, bn, bk, nfrag), stream>>>(
            nullptr, down_h, w, nullptr, mid_h, counts, offsets, sorted,
            hot_experts, hot_count, mid_dim, out_dim, row * out_dim, row, 0u,
            tile_map);
}

namespace {
template <uint32_t Mt, bool MidF16, bool OutF16>
void Q2DownHotTile(float* down, __half* down_h, const void* down_w,
                   const float* mid, const __half* mid_h,
                   const uint32_t* counts, const uint32_t* offsets,
                   const uint32_t* sorted, const uint32_t* hot_experts,
                   uint32_t hot_count, uint32_t hot_max, uint32_t mid_dim,
                   uint32_t out_dim, hipStream_t stream) {
  constexpr uint32_t bm = 16u, bn = 16u, bk = 16u;
  const uint64_t row = (mid_dim / ROCM_QK_K) * kQ2KBlock;
  const dim3 block(32u * Mt, 1u, 1u);
  const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                  (hot_max + Mt * bm - 1u) / (Mt * bm), hot_count);
  moe_down_q2K_hotlist_wmma_n2_kernel<Mt, 16, 16, 16, MidF16, OutF16>
      <<<grid, block, ds4_rocm_q2_down_wmma_shmem(Mt, bm, bn, bk), stream>>>(
          down, down_h, static_cast<const char*>(down_w), mid, mid_h, counts,
          offsets, sorted, hot_experts, hot_count, mid_dim, out_dim,
          row * out_dim, row);
}

template <bool MidF16, bool OutF16>
void Q2DownHot(float* down, __half* down_h, const void* down_w,
               const float* mid, const __half* mid_h, const uint32_t* counts,
               const uint32_t* offsets, const uint32_t* sorted,
               const uint32_t* hot_experts, uint32_t hot_count,
               uint32_t hot_max, uint32_t mid_dim, uint32_t out_dim,
               uint32_t mtiles, hipStream_t stream) {
  if (mtiles == 8u)
    Q2DownHotTile<8, MidF16, OutF16>(down, down_h, down_w, mid, mid_h, counts,
                                     offsets, sorted, hot_experts, hot_count,
                                     hot_max, mid_dim, out_dim, stream);
  else
    Q2DownHotTile<4, MidF16, OutF16>(down, down_h, down_w, mid, mid_h, counts,
                                     offsets, sorted, hot_experts, hot_count,
                                     hot_max, mid_dim, out_dim, stream);
}
}  // namespace

bool MoeQ2DownHot(float* down, __half* down_h, const void* down_w,
                  const float* mid, const __half* mid_h,
                  const int32_t* selected, const uint32_t* counts,
                  const uint32_t* offsets, const uint32_t* sorted,
                  const uint32_t* hot_experts, uint32_t hot_count,
                  uint32_t hot_max, uint32_t mid_dim, uint32_t out_dim,
                  uint32_t mtiles, hipStream_t stream) {
  (void)selected;
  // One output and one input, F32 or F16.
  if ((!down && !down_h) || (!mid && !mid_h)) return false;
  if (down_h && mid_h)
    Q2DownHot<true, true>(nullptr, down_h, down_w, nullptr, mid_h, counts,
                          offsets, sorted, hot_experts, hot_count, hot_max,
                          mid_dim, out_dim, mtiles, stream);
  else if (down_h)
    Q2DownHot<false, true>(nullptr, down_h, down_w, mid, nullptr, counts,
                           offsets, sorted, hot_experts, hot_count, hot_max,
                           mid_dim, out_dim, mtiles, stream);
  else if (mid_h)
    Q2DownHot<true, false>(down, nullptr, down_w, nullptr, mid_h, counts,
                           offsets, sorted, hot_experts, hot_count, hot_max,
                           mid_dim, out_dim, mtiles, stream);
  else
    Q2DownHot<false, false>(down, nullptr, down_w, mid, nullptr, counts,
                            offsets, sorted, hot_experts, hot_count, hot_max,
                            mid_dim, out_dim, mtiles, stream);
  return true;
}

// --- Routed experts: the Q4_K and the all-Q2_K routes -----------------------

namespace {
constexpr uint64_t kQ4KBlock = 144u;
}  // namespace

void MoeRouteSort(const int32_t* selected, uint32_t* counts,
                  uint32_t* offsets, int32_t* cursors, uint32_t* sorted,
                  uint32_t n_tokens, hipStream_t stream) {
  const uint32_t pairs = n_tokens * kUsed;
  (void)hipMemsetAsync(counts, 0, DS4_ROCM_N_EXPERT * sizeof(uint32_t),
                       stream);
  moe_count_sorted_pairs_kernel<<<(pairs + 255u) / 256u, 256, 0, stream>>>(
      counts, selected, pairs);
  moe_prefix_sorted_pairs_kernel<<<1, 1, 0, stream>>>(
      offsets, reinterpret_cast<uint32_t*>(cursors), counts);
  moe_scatter_sorted_pairs_deterministic_kernel<<<
      256u, 256u, 256u * sizeof(uint32_t), stream>>>(sorted, offsets, selected,
                                                     pairs);
}

void MoeQ4kGateUpTiles(float* mid, const void* gate_w, const void* up_w,
                       const void* xq, const int32_t* selected,
                       const uint32_t* sorted, const uint32_t* offsets,
                       const uint32_t* counts, const uint32_t* tile_total,
                       const uint32_t* tile_experts,
                       const uint32_t* tile_starts, const float* weights,
                       uint32_t in_dim, uint32_t mid_dim, uint32_t tile_shift,
                       uint32_t tile_cap, float clamp, hipStream_t stream) {
  (void)selected;
  const uint64_t row = (in_dim / ROCM_QK_K) * kQ4KBlock;
  const dim3 grid((mid_dim + 31u) / 32u, tile_cap, 1);
  if (tile_shift == 3u)
    moe_gate_up_mid_q4K_expert_tile8_row32_kernel<<<grid, 256, 0, stream>>>(
        nullptr, nullptr, mid, static_cast<const char*>(gate_w),
        static_cast<const char*>(up_w), static_cast<const hip_block_q8_K*>(xq),
        sorted, offsets, counts, tile_total, tile_experts, tile_starts,
        weights, row * mid_dim, row, in_dim / ROCM_QK_K, mid_dim, kUsed, 0u,
        0u, clamp);
  else
    moe_gate_up_mid_q4K_expert_tile4_row32_kernel<<<grid, 256, 0, stream>>>(
        nullptr, nullptr, mid, static_cast<const char*>(gate_w),
        static_cast<const char*>(up_w), static_cast<const hip_block_q8_K*>(xq),
        sorted, offsets, counts, tile_total, tile_experts, tile_starts,
        weights, row * mid_dim, row, in_dim / ROCM_QK_K, mid_dim, kUsed, 0u,
        0u, clamp);
}

void MoeQ4kGateUpDecode(float* mid, const void* gate_w, const void* up_w,
                        const void* xq, const int32_t* selected,
                        const float* weights, uint32_t in_dim,
                        uint32_t mid_dim, uint32_t n_tokens, float clamp,
                        hipStream_t stream) {
  const uint64_t row = (in_dim / ROCM_QK_K) * kQ4KBlock;
  const dim3 grid((mid_dim + 127u) / 128u, n_tokens * kUsed, 1);
  moe_gate_up_mid_decode_q4K_qwarp32_kernel<<<grid, 256, 0, stream>>>(
      nullptr, nullptr, mid, static_cast<const char*>(gate_w),
      static_cast<const char*>(up_w), static_cast<const hip_block_q8_K*>(xq),
      selected, weights, row * mid_dim, row, in_dim / ROCM_QK_K, mid_dim,
      kUsed, 0u, clamp);
}

void MoeQ4kDownSum6(float* out, const void* down_w, const void* midq,
                    const int32_t* selected, uint32_t mid_dim,
                    uint32_t out_dim, uint32_t n_tokens, hipStream_t stream) {
  const uint64_t row = (mid_dim / ROCM_QK_K) * kQ4KBlock;
  const dim3 grid((out_dim + 31u) / 32u, n_tokens, 1);
  moe_down_q4K_sum6_qwarp32_kernel<<<grid, 256, 0, stream>>>(
      out, static_cast<const char*>(down_w),
      static_cast<const hip_block_q8_K*>(midq), selected, row * out_dim, row,
      mid_dim / ROCM_QK_K, out_dim);
}

namespace {
void Q4kDownTiles(float* out, const void* down_w, const void* midq,
                  const uint32_t* sorted, const uint32_t* offsets,
                  const uint32_t* counts, const uint32_t* tile_total,
                  const uint32_t* tile_experts, const uint32_t* tile_starts,
                  uint32_t mid_dim, uint32_t out_dim, uint32_t tile_shift,
                  uint32_t tile_cap, uint32_t atomic, hipStream_t stream) {
  const uint64_t row = (mid_dim / ROCM_QK_K) * kQ4KBlock;
  const char* w = static_cast<const char*>(down_w);
  const auto* q = static_cast<const hip_block_q8_K*>(midq);
  const dim3 grid((out_dim + 31u) / 32u, tile_cap, 1);
  if (tile_shift == 3u)
    moe_down_q4K_expert_tile8_row32_kernel<<<grid, 256, 0, stream>>>(
        out, w, q, sorted, offsets, counts, tile_total, tile_experts,
        tile_starts, row * out_dim, row, mid_dim / ROCM_QK_K, out_dim, kUsed,
        atomic);
  else
    moe_down_q4K_expert_tile4_row32_kernel<<<grid, 256, 0, stream>>>(
        out, w, q, sorted, offsets, counts, tile_total, tile_experts,
        tile_starts, row * out_dim, row, mid_dim / ROCM_QK_K, out_dim, kUsed,
        atomic);
}
}  // namespace

void MoeQ4kDownTiles(float* down, const void* down_w, const void* midq,
                     const int32_t* selected, const uint32_t* sorted,
                     const uint32_t* offsets, const uint32_t* counts,
                     const uint32_t* tile_total, const uint32_t* tile_experts,
                     const uint32_t* tile_starts, uint32_t mid_dim,
                     uint32_t out_dim, uint32_t tile_shift, uint32_t tile_cap,
                     hipStream_t stream) {
  (void)selected;
  Q4kDownTiles(down, down_w, midq, sorted, offsets, counts, tile_total,
               tile_experts, tile_starts, mid_dim, out_dim, tile_shift,
               tile_cap, 0u, stream);
}

void MoeQ4kDownTilesAtomic(float* out, const void* down_w, const void* midq,
                           const int32_t* selected, const uint32_t* sorted,
                           const uint32_t* offsets, const uint32_t* counts,
                           const uint32_t* tile_total,
                           const uint32_t* tile_experts,
                           const uint32_t* tile_starts, uint32_t mid_dim,
                           uint32_t out_dim, uint32_t tile_shift,
                           uint32_t tile_cap, hipStream_t stream) {
  (void)selected;
  Q4kDownTiles(out, down_w, midq, sorted, offsets, counts, tile_total,
               tile_experts, tile_starts, mid_dim, out_dim, tile_shift,
               tile_cap, 1u, stream);
}

void MoeQ4kDownDecode(float* down, const void* down_w, const void* midq,
                      const int32_t* selected, uint32_t mid_dim,
                      uint32_t out_dim, uint32_t n_tokens,
                      hipStream_t stream) {
  const uint64_t row = (mid_dim / ROCM_QK_K) * kQ4KBlock;
  const dim3 grid((out_dim + 31u) / 32u, n_tokens * kUsed, 1);
  moe_down_q4K_qwarp32_kernel<<<grid, 256, 0, stream>>>(
      down, static_cast<const char*>(down_w),
      static_cast<const hip_block_q8_K*>(midq), selected, row * out_dim, row,
      mid_dim / ROCM_QK_K, out_dim, kUsed);
}

void MoeQ2kGateUpBatch(float* mid, const void* gate_w, const void* up_w,
                       const float* x, const float* weights,
                       const int32_t* selected, const uint32_t* counts,
                       const uint32_t* offsets, const uint32_t* sorted,
                       uint32_t max_count, uint32_t in_dim, uint32_t mid_dim,
                       float clamp, hipStream_t stream) {
  (void)selected;
  const uint64_t row = (in_dim / ROCM_QK_K) * kQ2KBlock;
  constexpr uint32_t rpb = 16u;
  const dim3 grid((mid_dim + rpb - 1u) / rpb, 256u, 1);
  moe_gate_up_mid_q2K_expert_batch_sharedx_kernel<4>
      <<<grid, rpb * 32u, 4u * 256u * sizeof(float), stream>>>(
          mid, nullptr, static_cast<const char*>(gate_w),
          static_cast<const char*>(up_w), x, weights, counts, offsets, sorted,
          1u, max_count, in_dim, mid_dim, row * mid_dim, row, clamp);
}

void MoeQ2kGateUpHot(__half* mid_h, const void* gate_w, const void* up_w,
                     const float* x, const __half* x_h, const float* weights,
                     const int32_t* selected, const uint32_t* counts,
                     const uint32_t* offsets, const uint32_t* sorted,
                     const uint32_t* hot_experts, uint32_t hot_count,
                     uint32_t hot_max, uint32_t mtiles, uint32_t in_dim,
                     uint32_t mid_dim, float clamp, hipStream_t stream) {
  (void)selected;
  constexpr uint32_t bm = 16u, bn = 16u, bk = 16u;
  const uint64_t row = (in_dim / ROCM_QK_K) * kQ2KBlock;
  const auto* gw = static_cast<const char*>(gate_w);
  const auto* uw = static_cast<const char*>(up_w);
  if (mtiles == 8u) {
    constexpr uint32_t mt = 8u;
    const dim3 grid((mid_dim + 2u * bn - 1u) / (2u * bn),
                    (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
    const size_t shmem = (mt * bm * bk + 4u * bk * bn) * sizeof(__half) +
                         (4u * mt * bm * bn) * sizeof(float);
    moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<8, 16, 16, 16, true, true>
        <<<grid, dim3(32u * mt, 1u, 1u), shmem, stream>>>(
            nullptr, mid_h, gw, uw, x, x_h, weights, counts, offsets, sorted,
            hot_experts, hot_count, in_dim, mid_dim, row * mid_dim, row, clamp);
    return;
  }
  constexpr uint32_t mt = 4u;
  const dim3 grid((mid_dim + 2u * bn - 1u) / (2u * bn),
                  (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
  const size_t shmem = (mt * bm * bk + 4u * bk * bn) * sizeof(__half) +
                       (4u * mt * bm * bn) * sizeof(float);
  moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<4, 16, 16, 16, true, true>
      <<<grid, dim3(32u * mt, 1u, 1u), shmem, stream>>>(
          nullptr, mid_h, gw, uw, x, x_h, weights, counts, offsets, sorted,
          hot_experts, hot_count, in_dim, mid_dim, row * mid_dim, row, clamp);
}

void MoeQ2kGateUpDecode(float* mid, const void* gate_w, const void* up_w,
                        const void* xq, const int32_t* selected,
                        const float* weights, uint32_t in_dim,
                        uint32_t mid_dim, uint32_t n_tokens, float clamp,
                        hipStream_t stream) {
  const uint64_t row = (in_dim / ROCM_QK_K) * kQ2KBlock;
  const dim3 grid((mid_dim + 255u) / 256u, n_tokens * kUsed, 1);
  moe_gate_up_mid_q2K_decode_q8_qwarp32_kernel<<<grid, 256u, 0, stream>>>(
      nullptr, nullptr, mid, static_cast<const char*>(gate_w),
      static_cast<const char*>(up_w), static_cast<const hip_block_q8_K*>(xq),
      selected, weights, row * mid_dim, row, in_dim / ROCM_QK_K, mid_dim,
      kUsed, 0u, clamp);
}

void MoeQ2kGateUpRows(float* mid, const void* gate_w, const void* up_w,
                      const float* x, const int32_t* selected,
                      const float* weights, uint32_t in_dim, uint32_t mid_dim,
                      uint32_t n_tokens, float clamp, hipStream_t stream) {
  const uint64_t row = (in_dim / ROCM_QK_K) * kQ2KBlock;
  const dim3 grid(mid_dim, n_tokens * kUsed, 1);
  moe_gate_up_mid_q2K_rows_rpb1_w32_kernel<<<grid, 32u, 0, stream>>>(
      nullptr, nullptr, mid, static_cast<const char*>(gate_w),
      static_cast<const char*>(up_w), x, selected, weights, row * mid_dim, row,
      in_dim, mid_dim, kUsed, clamp, 0);
}

void MoeQ2kDownRows(float* out, const void* down_w, const float* mid,
                    const int32_t* selected, uint32_t mid_dim,
                    uint32_t out_dim, uint32_t n_tokens, hipStream_t stream) {
  const uint64_t row = (mid_dim / ROCM_QK_K) * kQ2KBlock;
  const dim3 grid(out_dim, n_tokens, 1);
  moe_down_q2K_sum_rows_w32_kernel<<<grid, 32u, 0, stream>>>(
      out, static_cast<const char*>(down_w), mid, selected, n_tokens, mid_dim,
      out_dim, row * out_dim, row);
}

bool MmqInit(hipStream_t stream) {
  (void)stream;
  return ds4_mmq_init(0) == 0;
}

void MmqRelease(hipStream_t stream) {
  (void)stream;
  ds4_mmq_cleanup();
}

}  // namespace gufo::models::deepseek_v4_flash
