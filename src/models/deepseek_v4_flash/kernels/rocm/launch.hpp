// Launchers behind the generated Rust wrappers (launch.toml): the kernel
// choice and grid of the former ds4 ABI functions, on device pointers and an
// explicit stream, without the runtime's state.
#pragma once

#include <hip/hip_runtime.h>

#include <hip/hip_fp16.h>

#include <cstdint>

namespace gufo::models::deepseek_v4_flash {

void RmsNormPlainRows(float* out, const float* x, uint32_t n, uint32_t rows,
                      float eps, hipStream_t stream);
void RmsNormWeightRows(float* out, const float* x, const float* w, uint32_t n,
                       uint32_t rows, float eps, hipStream_t stream);
void QkvRmsNormRows(float* q_out, const float* q, const float* q_w,
                    uint32_t q_n, float* kv_out, const float* kv,
                    const float* kv_w, uint32_t kv_n, uint32_t rows, float eps,
                    hipStream_t stream);
void HeadRmsNorm(float* x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
                 float eps, hipStream_t stream);
bool HeadRmsNormRopeTail(float* x, uint32_t n_tok, uint32_t n_head,
                         uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
                         uint32_t n_ctx_orig, bool inverse, float freq_base,
                         float freq_scale, float ext_factor, float attn_factor,
                         float beta_fast, float beta_slow, float eps,
                         hipStream_t stream);
void RopeTail(float* x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
              uint32_t n_rot, uint32_t pos0, uint32_t pos_stride,
              uint32_t n_ctx_orig, bool inverse,
              float freq_base, float freq_scale, float ext_factor,
              float attn_factor, float beta_fast, float beta_slow,
              hipStream_t stream);

void HcWeightedSum(float* out, const float* residual_hc, const float* weights,
                   uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens,
                   hipStream_t stream);
void HcSplitWeightedSum(float* out, float* split, const float* mix,
                        const float* residual_hc, const float* scale,
                        const float* base, uint32_t n_embd, uint32_t n_hc,
                        uint32_t n_rows, uint32_t sinkhorn_iters, float eps,
                        hipStream_t stream);
bool HcNormMixSplitWeightedSum(float* out, float* mix, float* split,
                               const float* residual_hc, const __half* mix_w,
                               const float* scale, const float* base,
                               uint32_t n_embd, uint32_t n_rows,
                               uint32_t sinkhorn_iters, float eps,
                               float norm_eps, hipStream_t stream);
void HcSplitWeightedSumNorm(float* out, float* norm_out, float* split,
                            const float* mix, const float* residual_hc,
                            const float* scale, const float* base,
                            const float* norm_w, uint32_t n_embd,
                            uint32_t n_hc, uint32_t n_rows,
                            uint32_t sinkhorn_iters, float eps, float norm_eps,
                            hipStream_t stream);
void OutputHcWeights(float* out, const float* pre, const float* scale,
                     const float* base, uint32_t n_hc, uint32_t n_tokens,
                     float eps, hipStream_t stream);
void HcExpandSplit(float* out_hc, const float* block_out,
                   const float* residual_hc, const float* split,
                   uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens,
                   hipStream_t stream);
void HcExpandAddSplit(float* out_hc, const float* block_out,
                      const float* block_add, const float* residual_hc,
                      const float* split, uint32_t n_embd, uint32_t n_hc,
                      uint32_t n_tokens, hipStream_t stream);
bool HcExpandAddSplitMoesum(float* out_hc, const __half* down_h,
                            const float* block_add, const float* residual_hc,
                            const float* split, uint32_t n_embd, uint32_t n_hc,
                            uint32_t n_expert, uint32_t n_tokens,
                            hipStream_t stream);
void SpecRowArgmax(int32_t* out_index, const float* logits, uint32_t vocab,
                   uint32_t n_rows, hipStream_t stream);
void DsparkCaptureFeatures(float* out, const float* hc,
                           uint32_t out_row_stride, uint32_t slot_offset,
                           uint32_t n_embd, uint32_t n_hc, uint32_t n_rows,
                           hipStream_t stream);

void EmbedTokenHc(float* out_hc, const __half* w, uint32_t n_vocab,
                  uint32_t token, uint32_t n_embd, uint32_t n_hc,
                  hipStream_t stream);
void EmbedTokensHc(float* out_hc, const int32_t* tokens, const __half* w,
                   uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd,
                   uint32_t n_hc, hipStream_t stream);
void Fp8KvQuantize(float* x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot,
                   hipStream_t stream);
void Fp8KvQuantizeMirrorF16(float* x, __half* mirror_f16, uint32_t n_tok,
                            uint32_t head_dim, uint32_t n_rot,
                            hipStream_t stream);
void ConvertF32ToF16(__half* dst, const float* src, std::size_t count,
                     hipStream_t stream);
void FillF32(float* x, float value, std::size_t count, hipStream_t stream);
void Swiglu(float* out, const float* gate, const float* up, uint32_t n,
            float clamp, float weight, hipStream_t stream);
void StoreRawKvBatch(float* raw_cache, const float* kv, uint32_t raw_cap,
                     uint32_t pos0, uint32_t n_tokens, uint32_t head_dim,
                     hipStream_t stream);
void IndexerQat(float* x, uint32_t n_rows, uint32_t head_dim,
                hipStream_t stream);

void DsparkCandidates(void* out, uint64_t* scratch, const float* logits,
                      std::size_t logits_stride, const float* hidden,
                      std::size_t hidden_stride, const int32_t* previous,
                      const void* w1, const void* w2, const void* confidence,
                      uint32_t vocab, uint32_t rank, uint32_t hidden_width,
                      uint32_t requests, hipStream_t stream);
void DsparkMarkovArgmax(int32_t* out_index, uint64_t* scratch_key,
                        const float* logits_row, const void* w1_row,
                        const void* w2, uint32_t vocab, uint32_t markov_rank,
                        hipStream_t stream);
bool DsparkMarkovArgmaxBatch(int32_t* out_index, uint64_t* scratch_key,
                             const float* logits_rows,
                             std::size_t logits_row_stride,
                             const int32_t* previous_tokens, const void* w1,
                             const void* w2, uint32_t vocab,
                             uint32_t markov_rank, uint32_t n_rows,
                             hipStream_t stream);

void AttentionDecodeHeads(float* heads, const float* sinks, const float* q,
                          const float* raw_kv, uint32_t n_raw,
                          uint32_t raw_cap, uint32_t raw_start,
                          const float* comp_kv, uint32_t n_comp,
                          const float* comp_mask, uint32_t n_head,
                          uint32_t head_dim, hipStream_t stream);
void AttentionNoncausalRawBatchHeads(float* heads, const float* sinks,
                                     const float* q, const float* raw_kv,
                                     uint32_t n_tokens, uint32_t n_raw,
                                     uint32_t raw_cap, uint32_t raw_start,
                                     uint32_t n_head, uint32_t head_dim,
                                     hipStream_t stream);

void CompressorStore(const float* kv, const float* sc, float* state_kv,
                     float* state_score, const __half* ape, uint32_t coff,
                     uint32_t head_dim, uint32_t ratio, uint32_t pos0,
                     uint32_t n_tokens, hipStream_t stream);
void CompressorStateReset(float* state_kv, float* state_score,
                          std::size_t count, hipStream_t stream);
void CompressorSetRows(float* state_kv, float* state_score, const float* kv,
                       const float* sc, const __half* ape, uint32_t width,
                       uint32_t ratio, uint32_t pos0, uint32_t src0,
                       uint32_t dst0, uint32_t rows, hipStream_t stream);
void CompressorPrefillPool(float* comp, const float* kv, const float* sc,
                           const float* state_kv, const float* state_score,
                           const __half* ape, uint32_t coff, uint32_t head_dim,
                           uint32_t ratio, uint32_t pos0, uint32_t n_comp,
                           bool replay, hipStream_t stream);
void CompressorUpdatePool(float* row, const float* state_kv,
                          const float* state_score, uint32_t coff,
                          uint32_t head_dim, uint32_t ratio,
                          hipStream_t stream);
void CompressorShiftRatio4(float* state_kv, float* state_score, uint32_t width,
                           hipStream_t stream);

void MatmulF16EqualGroupPairs(float* out, const __half* w, const float* x,
                              uint32_t in_dim, uint32_t out_dim,
                              uint32_t rows_per_group, uint32_t pairs,
                              hipStream_t stream);
void MatmulF16RaggedGroupPair(float* out, const __half* w, const float* x,
                              uint32_t in_dim, uint32_t out_dim, uint32_t row0,
                              uint32_t rows0, uint32_t rows1,
                              hipStream_t stream);
void MatmulF16PairNarrow(float* out0, float* out1, const __half* w0,
                         const __half* w1, const float* x, uint32_t in_dim,
                         uint32_t out_dim, uint32_t n_tok, hipStream_t stream);

void QuantizeQ8Rows(void* xq, float* xscale, const float* x, uint32_t in_dim,
                    uint32_t n_tok, hipStream_t stream);
void MatmulQ8Wmma(float* out, const void* w, const float* x, uint32_t in_dim,
                  uint32_t out_dim, uint32_t n_tok, hipStream_t stream);
bool MatmulQ8PreqReuse(float* out, const void* w, const void* xq,
                       const float* xscale, uint32_t in_dim, uint32_t out_dim,
                       uint32_t n_tok, uint32_t rows_per_block,
                       hipStream_t stream);
void MatmulQ8Sharedx(float* out, const void* w, const float* x,
                     uint32_t in_dim, uint32_t out_dim, uint32_t n_tok,
                     hipStream_t stream);
void MatmulQ8Warp8(float* out, const void* w, const float* x, uint32_t in_dim,
                   uint32_t out_dim, uint32_t n_tok, hipStream_t stream);
void MatmulQ8PreqRows(float* out, const void* w, const void* xq,
                      const float* xscale, uint32_t in_dim, uint32_t out_dim,
                      hipStream_t stream);
bool MatmulQ8PreqAllRows(float* out, const void* w, const void* xq,
                         const float* xscale, uint32_t in_dim,
                         uint32_t out_dim, uint32_t n_tok, hipStream_t stream);
void MatmulQ8PreqEqualGroupPairs(float* out, const void* w, const void* xq,
                                 const float* xscale, uint32_t in_dim,
                                 uint32_t out_dim, uint32_t rows_per_group,
                                 uint32_t pairs, hipStream_t stream);
void MatmulQ8PreqGroupPair(float* out, const void* w, const void* xq,
                           const float* xscale, uint32_t in_dim,
                           uint32_t out_dim, uint32_t row0, uint32_t rows0,
                           uint32_t rows1, hipStream_t stream);
void MatmulQ8PairPreqReuse(float* out0, float* out1, const void* w0,
                           const void* w1, const void* xq, const float* xscale,
                           uint32_t in_dim, uint32_t out_dim, uint32_t n_tok,
                           hipStream_t stream);
void MatmulQ8PairPreqRows(float* out0, float* out1, const void* w0,
                          const void* w1, const void* xq, const float* xscale,
                          uint32_t in_dim, uint32_t out0_dim, uint32_t out1_dim,
                          hipStream_t stream);
void MatmulQ8HcExpandPreqRows(float* out_hc, float* block_out,
                              const float* block_add, const float* residual_hc,
                              const float* split, const void* w,
                              const void* xq, const float* xscale,
                              uint32_t in_dim, uint32_t n_embd, uint32_t n_hc,
                              hipStream_t stream);
void DequantQ8ToF16(__half* out, const void* w, uint32_t in_dim,
                    uint32_t out_dim, hipStream_t stream);
bool MmqQ8Dense(float* out, const void* w, const float* x, uint32_t in_dim,
                uint32_t out_dim, uint32_t n_tok, hipStream_t stream);
void MatmulF16TinyBatch(float* out, const __half* w, const float* x,
                        uint32_t in_dim, uint32_t out_dim, uint32_t n_tok,
                        hipStream_t stream);
void MatmulF16OrderedReuse(float* out, const __half* w, const float* x,
                           uint32_t in_dim, uint32_t out_dim, uint32_t n_tok,
                           hipStream_t stream);
void MatmulF16SharedxRows(float* out, const __half* w, const float* x,
                          uint32_t in_dim, uint32_t out_dim,
                          hipStream_t stream);
void MatmulF16OrderedChunks(float* out, const __half* w, const float* x,
                            uint32_t in_dim, uint32_t out_dim, uint32_t n_tok,
                            hipStream_t stream);
void MatmulF16PairSharedxRows(float* out0, float* out1, const __half* w0,
                              const __half* w1, const float* x, uint32_t in_dim,
                              uint32_t out_dim, hipStream_t stream);
void MatmulF16PairOrderedChunks(float* out0, float* out1, const __half* w0,
                                const __half* w1, const float* x,
                                uint32_t in_dim, uint32_t out_dim,
                                hipStream_t stream);
void MatmulF32Plain(float* out, const float* w, const float* x, uint32_t in_dim,
                    uint32_t out_dim, uint32_t n_tok, hipStream_t stream);

void AttentionDecodeOnline(float* heads, const float* sinks, const float* q,
                           const float* raw_kv, const float* comp_kv,
                           uint32_t n_tokens, uint32_t pos0, uint32_t n_raw,
                           uint32_t raw_cap, uint32_t raw_start,
                           uint32_t n_comp, uint32_t window, uint32_t ratio,
                           uint32_t n_head, uint32_t head_dim,
                           hipStream_t stream);
void AttentionMixedWmma(float* heads, const float* sinks, const float* q,
                        const float* raw_kv, const float* comp_kv,
                        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw,
                        uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp,
                        uint32_t window, uint32_t ratio, uint32_t n_head,
                        uint32_t head_dim, hipStream_t stream);
void AttentionIndexedWmma(float* heads, const float* sinks, const float* q,
                          const float* raw_kv, const float* comp_kv,
                          const int32_t* topk, uint32_t n_tokens,
                          uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
                          uint32_t raw_start, uint32_t n_comp, uint32_t top_k,
                          uint32_t window, uint32_t ratio, uint32_t n_head,
                          uint32_t head_dim, hipStream_t stream);
void AttentionIndexedWmmaF16(float* heads, const float* sinks, const float* q,
                             const float* raw_kv, const __half* comp_kv,
                             const int32_t* topk, uint32_t n_tokens,
                             uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
                             uint32_t raw_start, uint32_t n_comp,
                             uint32_t top_k, uint32_t window, uint32_t ratio,
                             uint32_t n_head, uint32_t head_dim,
                             hipStream_t stream);
void AttentionDecodeMixed(float* heads, const float* sinks, const float* q,
                          const float* raw_kv, const float* comp_kv,
                          const float* comp_mask, uint32_t n_tokens,
                          uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
                          uint32_t raw_start, uint32_t n_comp, uint32_t window,
                          uint32_t ratio, uint32_t n_head, uint32_t head_dim,
                          hipStream_t stream);
void AttentionDecodeIndexedOne(float* heads, const float* q,
                               const float* raw_kv, const float* comp_kv,
                               const int32_t* topk, const float* sinks,
                               uint32_t n_raw, uint32_t raw_cap,
                               uint32_t raw_start, uint32_t n_comp,
                               uint32_t top_k, uint32_t pos, uint32_t ratio,
                               uint32_t n_head, uint32_t head_dim,
                               hipStream_t stream);
void IndexedTopkSort512(int32_t* dst, const int32_t* src, uint32_t n_tokens,
                        hipStream_t stream);
void AttentionIndexedOnline(float* heads, const float* sinks, const float* q,
                            const float* raw_kv, const float* comp_kv,
                            const int32_t* topk, uint32_t n_tokens,
                            uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
                            uint32_t raw_start, uint32_t n_comp,
                            uint32_t top_k, uint32_t window, uint32_t ratio,
                            uint32_t n_head, uint32_t head_dim,
                            hipStream_t stream);
void AttentionIndexedMixed(float* heads, const float* sinks, const float* q,
                           const float* raw_kv, const float* comp_kv,
                           const int32_t* topk, uint32_t n_tokens,
                           uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
                           uint32_t raw_start, uint32_t n_comp, uint32_t top_k,
                           uint32_t window, uint32_t ratio, uint32_t n_head,
                           uint32_t head_dim, hipStream_t stream);
void AttentionStaticOnline(float* heads, const float* sinks, const float* q,
                           const float* raw_kv, const float* comp_kv,
                           uint32_t n_tokens, uint32_t n_comp, uint32_t window,
                           uint32_t ratio, uint32_t n_head, uint32_t head_dim,
                           hipStream_t stream);
void AttentionPrefillRawSoftmax(float* scores, const float* sinks,
                                uint32_t n_tokens, uint32_t window,
                                uint32_t n_keys, uint32_t n_head,
                                hipStream_t stream);
void AttentionPrefillUnpackHeads(float* heads, const float* tmp,
                                 uint32_t n_tokens, uint32_t n_head,
                                 uint32_t head_dim, hipStream_t stream);
bool AttentionPrefillRaw(float* heads, const float* sinks, const float* q,
                         const float* raw_kv, uint32_t n_tokens,
                         uint32_t window, uint32_t n_head, uint32_t head_dim,
                         hipStream_t stream);
void AttentionPrefillPackMixedKv(float* dst, const float* raw_kv,
                                 const float* comp_kv, uint32_t n_tokens,
                                 uint32_t n_comp, uint32_t head_dim,
                                 hipStream_t stream);
void AttentionPrefillMixedSoftmax(float* scores, const float* sinks,
                                  const float* comp_mask, uint32_t n_tokens,
                                  uint32_t n_comp, uint32_t window,
                                  uint32_t ratio, uint32_t n_keys,
                                  uint32_t n_head, hipStream_t stream);
void AttentionPrefillMixedSoftmaxTile(float* scores, const float* sinks,
                                      const float* comp_mask,
                                      uint32_t raw_tokens, uint32_t tile_start,
                                      uint32_t tile_tokens, uint32_t n_comp,
                                      uint32_t window, uint32_t ratio,
                                      uint32_t n_keys, uint32_t n_head,
                                      hipStream_t stream);
void AttentionPrefillMixed(float* heads, const float* sinks, const float* q,
                           const float* raw_kv, const float* comp_kv,
                           const float* comp_mask, uint32_t n_tokens,
                           uint32_t n_comp, uint32_t window, uint32_t ratio,
                           uint32_t n_head, uint32_t head_dim,
                           hipStream_t stream);

bool AttnOutAPreqReuse(float* low, const void* w, const void* xq,
                       const float* xscale, uint32_t group_dim, uint32_t rank,
                       uint32_t n_groups, uint32_t n_tokens,
                       uint32_t rows_per_block, hipStream_t stream);
void AttnOutASharedx(float* low, const void* w, const float* heads,
                     uint32_t group_dim, uint32_t rank, uint32_t n_groups,
                     uint32_t n_tokens, hipStream_t stream);
void AttnOutAWarp8(float* low, const void* w, const float* heads,
                   uint32_t group_dim, uint32_t rank, uint32_t n_groups,
                   uint32_t n_tokens, hipStream_t stream);
void AttnOutAPreqWarp8(float* low, const void* w, const void* xq,
                       const float* xscale, uint32_t group_dim, uint32_t rank,
                       uint32_t n_groups, uint32_t n_tokens,
                       uint32_t rows_per_block, hipStream_t stream);
void PackGroupHeadsF16(__half* dst, const float* heads, uint32_t n_tokens,
                       uint32_t n_groups, uint32_t group_dim,
                       hipStream_t stream);
bool PackGroupHeadsRopeF16(__half* dst, const float* heads, uint32_t n_tokens,
                           uint32_t n_groups, uint32_t group_dim,
                           uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
                           uint32_t n_ctx_orig, float freq_base,
                           float freq_scale, float ext_factor,
                           float attn_factor, float beta_fast, float beta_slow,
                           hipStream_t stream);
void AttnUnpackGroupLow(float* low, const float* packed, uint32_t n_tokens,
                        uint32_t n_groups, uint32_t rank, hipStream_t stream);
void DequantQ8ToF16Transposed(__half* out, const void* w, uint32_t in_dim,
                              uint32_t out_dim, hipStream_t stream);
void GemmF16WmmaBatched(__half* out, const __half* weight, const __half* act,
                        uint32_t m, uint32_t n, uint32_t k, uint32_t ldc,
                        uint32_t batches, hipStream_t stream);
void GemmF16WmmaPadded(float* out, const __half* weight, const __half* act,
                       uint32_t m, uint32_t n_pad, uint32_t k,
                       hipStream_t stream);
void ZeroBytes(void* x, std::size_t count, hipStream_t stream);
void IndexerScoreDirect(float* scores, const float* q, const float* weights,
                        const float* index_comp, uint32_t n_comp,
                        uint32_t n_tokens, uint32_t pos0, uint32_t ratio,
                        float scale, bool causal, hipStream_t stream);
void IndexerPackKeys(__half* packed, const float* keys, uint32_t count,
                     hipStream_t stream);
void IndexerPackQueries(__half* packed, const float* q, uint32_t tokens,
                        hipStream_t stream);
void IndexerScoresWmmaPacked(float* scores, const __half* queries,
                             const float* weights, const __half* keys,
                             uint32_t n_comp, uint32_t n_tokens, uint32_t pos0,
                             uint32_t ratio, float scale, bool causal,
                             hipStream_t stream);
void IndexerScoresWmma(float* scores, const float* q, const float* weights,
                       const float* index_comp, uint32_t n_comp,
                       uint32_t n_tokens, uint32_t pos0, uint32_t ratio,
                       float scale, bool causal, hipStream_t stream);
void IndexerScores(float* scores, const float* q, const float* weights,
                   const float* index_comp, uint32_t n_comp, uint32_t n_tokens,
                   uint32_t pos0, uint32_t n_head, uint32_t head_dim,
                   uint32_t ratio, float scale, bool causal,
                   hipStream_t stream);
void IndexerTopkPartial(int32_t* selected, const float* scores,
                        uint32_t n_comp, uint32_t n_tokens, hipStream_t stream);
void IndexerTopk1024(int32_t* selected, const float* scores, uint32_t n_comp,
                     uint32_t n_tokens, uint32_t top_k, hipStream_t stream);
void IndexerTopk2048(int32_t* selected, const float* scores, uint32_t n_comp,
                     uint32_t n_tokens, uint32_t top_k, hipStream_t stream);
void IndexerTopk4096(int32_t* selected, const float* scores, uint32_t n_comp,
                     uint32_t n_tokens, uint32_t top_k, hipStream_t stream);
void IndexerTopk8192(int32_t* selected, const float* scores, uint32_t n_comp,
                     uint32_t n_tokens, uint32_t top_k, hipStream_t stream);
bool IndexerTopk8192Cub(int32_t* selected, const float* scores,
                        uint32_t n_comp, uint32_t n_tokens, uint32_t top_k,
                        hipStream_t stream);
void IndexerTopkChunk(int32_t* candidates, const float* scores,
                      uint32_t n_comp, uint32_t n_tokens, uint32_t top_k,
                      uint32_t candidate_stride, hipStream_t stream);
void IndexerTopkTreeMerge(int32_t* out, const int32_t* candidates,
                          const float* scores, uint32_t n_comp,
                          uint32_t n_tokens, uint32_t top_k, uint32_t n_sets,
                          uint32_t merge_group, uint32_t candidate_stride,
                          uint32_t out_stride, hipStream_t stream);
void IndexerTopkMerge(int32_t* selected, const int32_t* candidates,
                      const float* scores, uint32_t n_comp, uint32_t n_tokens,
                      uint32_t top_k, uint32_t candidate_count,
                      uint32_t candidate_stride, hipStream_t stream);
void IndexerTopkSerial(int32_t* selected, const float* scores, uint32_t n_comp,
                       uint32_t n_tokens, uint32_t top_k, hipStream_t stream);
void RouterSelect(int32_t* selected, float* weights, float* probs,
                  const float* bias, const int32_t* hash, const float* logits,
                  const int32_t* tokens, uint32_t token, uint32_t hash_rows,
                  uint32_t n_tokens, hipStream_t stream);

void MoeQuantizeQ8K(void* xq, const float* x, uint32_t dim, uint32_t rows, hipStream_t stream);
void MoeRoutePlan(const int32_t* selected, uint32_t* counts, uint32_t* offsets, int32_t* cursors, uint32_t* sorted, int32_t* tile_offsets, uint32_t* tile_total, uint32_t* tile_experts, uint32_t* tile_starts, uint32_t n_tokens, uint32_t tile_shift, hipStream_t stream);
void MoeRoutePlanNarrow(const int32_t* selected, uint32_t* counts, uint32_t* offsets, uint32_t* sorted, int32_t* tile_offsets, uint32_t* tile_total, uint32_t* tile_experts, uint32_t* tile_starts, uint32_t n_tokens, uint32_t tile_shift, hipStream_t stream);
void MoeRouteTiles(const int32_t* selected, const uint32_t* counts, int32_t* tile_offsets, uint32_t* tile_total, uint32_t* tile_experts, uint32_t* tile_starts, uint32_t tile_shift, hipStream_t stream);
void MoeIq2GateUpLut(float* mid, const void* gate_w, const void* up_w, const void* xq, const int32_t* selected, const float* weights, uint32_t in_dim, uint32_t mid_dim, uint32_t n_tokens, float clamp, hipStream_t stream);
void MoeIq2GateUpQwarp(float* gate, float* up, float* mid, const void* gate_w, const void* up_w, const void* xq, const int32_t* selected, const float* weights, uint32_t in_dim, uint32_t mid_dim, uint32_t n_tokens, float clamp, hipStream_t stream);
void MoeIq2GateUpTiles(float* mid, const void* gate_w, const void* up_w, const void* xq, const int32_t* selected, const uint32_t* sorted, const uint32_t* offsets, const uint32_t* counts, const uint32_t* tile_total, const uint32_t* tile_experts, const uint32_t* tile_starts, const float* weights, uint32_t in_dim, uint32_t mid_dim, uint32_t tile_shift, uint32_t tile_cap, uint32_t max_count, float clamp, hipStream_t stream);
void MoeIq2GateUpTilesWide(float* mid, const void* gate_w, const void* up_w, const void* xq, const int32_t* selected, const uint32_t* sorted, const uint32_t* offsets, const uint32_t* counts, const uint32_t* tile_total, const uint32_t* tile_experts, const uint32_t* tile_starts, const float* weights, uint32_t in_dim, uint32_t mid_dim, uint32_t tile_cap, uint32_t max_count, float clamp, hipStream_t stream);
void MoeIq2GateUpHot(float* mid, const void* gate_w, const void* up_w, const float* x, const __half* x_h, const float* weights, const int32_t* selected, const uint32_t* counts, const uint32_t* offsets, const uint32_t* sorted, const uint32_t* hot_experts, uint32_t hot_count, uint32_t hot_max, uint32_t in_dim, uint32_t mid_dim, float clamp, hipStream_t stream);
void MoeIq2GateUpHotF16(__half* mid_h, const void* gate_w, const void* up_w, const float* x, const __half* x_h, const float* weights, const int32_t* selected, const uint32_t* counts, const uint32_t* offsets, const uint32_t* sorted, const uint32_t* hot_experts, uint32_t hot_count, uint32_t hot_max, uint32_t in_dim, uint32_t mid_dim, float clamp, hipStream_t stream);
bool MoeIq2MmqGateUpSwiglu(float* gate, float* discard, float* mid, __half* mid_h, void* q81_scratch, std::size_t q81_bytes, const void* gate_w, const void* up_w, const float* x, const int32_t* selected, const float* weights, uint32_t in_dim, uint32_t mid_dim, uint32_t n_tokens, uint32_t max_rows, uint32_t tile_cols, float clamp, hipStream_t stream);
bool MoeIq2MmqGateUp(float* gate, float* up, void* q81_scratch, std::size_t q81_bytes, const void* gate_w, const void* up_w, const float* x, const int32_t* selected, uint32_t in_dim, uint32_t mid_dim, uint32_t n_tokens, uint32_t max_rows, uint32_t tile_cols, hipStream_t stream);
void MoeMmqSwiglu(float* mid, __half* mid_h, const float* gate, const float* up, const float* weights, uint32_t mid_dim, uint32_t n_tokens, float clamp, hipStream_t stream);
void MoeQ2DownSum6(float* out, const void* down_w, const void* midq, const int32_t* selected, uint32_t mid_dim, uint32_t out_dim, uint32_t n_tokens, hipStream_t stream);
void MoeQ2DownTiles(float* down, const void* down_w, const void* midq, const int32_t* selected, const uint32_t* sorted, const uint32_t* offsets, const uint32_t* counts, const uint32_t* tile_total, const uint32_t* tile_experts, const uint32_t* tile_starts, uint32_t mid_dim, uint32_t out_dim, uint32_t tile_shift, uint32_t tile_cap, bool wide, hipStream_t stream);
void MoeQ2DownTilesAtomic(float* out, const void* down_w, const void* midq, const int32_t* selected, const uint32_t* sorted, const uint32_t* offsets, const uint32_t* counts, const uint32_t* tile_total, const uint32_t* tile_experts, const uint32_t* tile_starts, uint32_t mid_dim, uint32_t out_dim, uint32_t tile_shift, uint32_t tile_cap, bool wide, hipStream_t stream);
void MoeSum(float* out, const float* down, uint32_t out_dim, uint32_t n_tokens, hipStream_t stream);
void MoeSumF16(float* out, const __half* down_h, uint32_t out_dim, uint32_t n_tokens, hipStream_t stream);
void MoeQ2DownBatch(float* down, const void* down_w, const float* mid, const int32_t* selected, const uint32_t* counts, const uint32_t* offsets, const uint32_t* sorted, const uint32_t* experts, uint32_t n_experts, uint32_t pair_shift, uint32_t rows_per_block, uint32_t min_count, uint32_t max_count, uint32_t mid_dim, uint32_t out_dim, hipStream_t stream);
void MoeQ2DownBatchF16(__half* down_h, const void* down_w, const float* mid, const int32_t* selected, const uint32_t* counts, const uint32_t* offsets, const uint32_t* sorted, const uint32_t* experts, uint32_t n_experts, uint32_t pair_shift, uint32_t rows_per_block, uint32_t min_count, uint32_t max_count, uint32_t mid_dim, uint32_t out_dim, hipStream_t stream);
void MoeQ2DownHotWide(__half* down_h, const void* down_w, const __half* mid_h, const int32_t* selected, const uint32_t* counts, const uint32_t* offsets, const uint32_t* sorted, const uint32_t* hot_experts, uint32_t hot_count, const uint32_t* tile_map, uint32_t n_map, uint32_t hot_max, uint32_t mid_dim, uint32_t out_dim, uint32_t mtiles, hipStream_t stream);
bool MoeQ2DownHot(float* down, __half* down_h, const void* down_w, const float* mid, const __half* mid_h, const int32_t* selected, const uint32_t* counts, const uint32_t* offsets, const uint32_t* sorted, const uint32_t* hot_experts, uint32_t hot_count, uint32_t hot_max, uint32_t mid_dim, uint32_t out_dim, uint32_t mtiles, hipStream_t stream);
void MoeRouteSort(const int32_t* selected, uint32_t* counts, uint32_t* offsets, int32_t* cursors, uint32_t* sorted, uint32_t n_tokens, hipStream_t stream);
void MoeQ4kGateUpTiles(float* mid, const void* gate_w, const void* up_w, const void* xq, const int32_t* selected, const uint32_t* sorted, const uint32_t* offsets, const uint32_t* counts, const uint32_t* tile_total, const uint32_t* tile_experts, const uint32_t* tile_starts, const float* weights, uint32_t in_dim, uint32_t mid_dim, uint32_t tile_shift, uint32_t tile_cap, float clamp, hipStream_t stream);
void MoeQ4kGateUpDecode(float* mid, const void* gate_w, const void* up_w, const void* xq, const int32_t* selected, const float* weights, uint32_t in_dim, uint32_t mid_dim, uint32_t n_tokens, float clamp, hipStream_t stream);
void MoeQ4kDownSum6(float* out, const void* down_w, const void* midq, const int32_t* selected, uint32_t mid_dim, uint32_t out_dim, uint32_t n_tokens, hipStream_t stream);
void MoeQ4kDownTiles(float* down, const void* down_w, const void* midq, const int32_t* selected, const uint32_t* sorted, const uint32_t* offsets, const uint32_t* counts, const uint32_t* tile_total, const uint32_t* tile_experts, const uint32_t* tile_starts, uint32_t mid_dim, uint32_t out_dim, uint32_t tile_shift, uint32_t tile_cap, hipStream_t stream);
void MoeQ4kDownTilesAtomic(float* out, const void* down_w, const void* midq, const int32_t* selected, const uint32_t* sorted, const uint32_t* offsets, const uint32_t* counts, const uint32_t* tile_total, const uint32_t* tile_experts, const uint32_t* tile_starts, uint32_t mid_dim, uint32_t out_dim, uint32_t tile_shift, uint32_t tile_cap, hipStream_t stream);
void MoeQ4kDownDecode(float* down, const void* down_w, const void* midq, const int32_t* selected, uint32_t mid_dim, uint32_t out_dim, uint32_t n_tokens, hipStream_t stream);
void MoeQ2kGateUpBatch(float* mid, const void* gate_w, const void* up_w, const float* x, const float* weights, const int32_t* selected, const uint32_t* counts, const uint32_t* offsets, const uint32_t* sorted, uint32_t max_count, uint32_t in_dim, uint32_t mid_dim, float clamp, hipStream_t stream);
void MoeQ2kGateUpHot(__half* mid_h, const void* gate_w, const void* up_w, const float* x, const __half* x_h, const float* weights, const int32_t* selected, const uint32_t* counts, const uint32_t* offsets, const uint32_t* sorted, const uint32_t* hot_experts, uint32_t hot_count, uint32_t hot_max, uint32_t mtiles, uint32_t in_dim, uint32_t mid_dim, float clamp, hipStream_t stream);
void MoeQ2kGateUpDecode(float* mid, const void* gate_w, const void* up_w, const void* xq, const int32_t* selected, const float* weights, uint32_t in_dim, uint32_t mid_dim, uint32_t n_tokens, float clamp, hipStream_t stream);
void MoeQ2kGateUpRows(float* mid, const void* gate_w, const void* up_w, const float* x, const int32_t* selected, const float* weights, uint32_t in_dim, uint32_t mid_dim, uint32_t n_tokens, float clamp, hipStream_t stream);
void MoeQ2kDownRows(float* out, const void* down_w, const float* mid, const int32_t* selected, uint32_t mid_dim, uint32_t out_dim, uint32_t n_tokens, hipStream_t stream);
bool MmqInit(hipStream_t stream);
void MmqRelease(hipStream_t stream);

}  // namespace gufo::models::deepseek_v4_flash
