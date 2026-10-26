//! The compressed-row indexer (`ds4_rocm_indexer.hip.hpp`): scores, top-k
//! selection and the FP4 rounding of its keys, routed over the generated
//! launchers as the C++ launchers chose. Top-k rows may hold out-of-range
//! sentinels when scores are NaN; the attention kernels skip indices
//! outside the visible rows.

use gufo_hip::{DeviceSliceMut, F16 as Half, Launch};

use crate::backend::{Backend, BatchMode, Error, Result};
use crate::generated;
use crate::tensor::{Input, Output};

use super::{F16, I32, ensure, launched, product, typed, typed_mut};

/// The indexer shape of DS4 Flash, which the specialized score kernels
/// serve.
const HEADS: u32 = 64;
const HEAD_DIM: u32 = 128;
/// Scores per chunk of the top-k tree.
const TOPK_CHUNK: u32 = 4096;

impl Backend {
    /// `ds4_gpu_indexer_score_one_tensor`: one token against `n_comp`
    /// rows, without causal masking.
    pub fn indexer_score_one(
        &self,
        scores: &mut impl Output,
        q: &impl Input,
        weights: &impl Input,
        index_comp: &impl Input,
        n_comp: u32,
        n_head: u32,
        head_dim: u32,
        scale: f32,
    ) -> Result<()> {
        self.indexer_scores(
            "indexer score one",
            scores,
            q,
            weights,
            index_comp,
            [n_comp, 1, 0, n_head, head_dim, 1],
            scale,
            false,
            None,
        )
    }

    /// `ds4_gpu_indexer_scores_prefill_tensor`: a prompt chunk from
    /// position 0, causal at compression ratio `ratio`; `score_scratch`,
    /// when large enough, holds packed F16 keys and queries.
    pub fn indexer_scores_prefill(
        &self,
        scores: &mut impl Output,
        q: &impl Input,
        weights: &impl Input,
        index_comp: &impl Input,
        n_comp: u32,
        n_tokens: u32,
        n_head: u32,
        head_dim: u32,
        ratio: u32,
        scale: f32,
        score_scratch: Option<&mut impl Output>,
    ) -> Result<()> {
        const OP: &str = "indexer scores prefill";
        ensure(OP, ratio != 0)?;
        self.indexer_scores(
            OP,
            scores,
            q,
            weights,
            index_comp,
            [n_comp, n_tokens, 0, n_head, head_dim, ratio],
            scale,
            true,
            score_scratch.map(|s| s.output()),
        )
    }

    /// `ds4_gpu_indexer_scores_decode_batch_tensor`: `n_tokens` from
    /// `pos0`, causal.
    pub fn indexer_scores_decode_batch(
        &self,
        scores: &mut impl Output,
        q: &impl Input,
        weights: &impl Input,
        index_comp: &impl Input,
        n_comp: u32,
        n_tokens: u32,
        pos0: u32,
        n_head: u32,
        head_dim: u32,
        ratio: u32,
        scale: f32,
        score_scratch: Option<&mut impl Output>,
    ) -> Result<()> {
        const OP: &str = "indexer scores decode batch";
        ensure(OP, ratio != 0)?;
        self.indexer_scores(
            OP,
            scores,
            q,
            weights,
            index_comp,
            [n_comp, n_tokens, pos0, n_head, head_dim, ratio],
            scale,
            true,
            score_scratch.map(|s| s.output()),
        )
    }

    /// `indexer_scores_launch`: the per-score kernel for one-token decode
    /// and verification rows, rocWMMA tiles for prompt chunks (from packed
    /// F16 operands when `scratch` holds both), and the generic kernel for
    /// other indexer shapes. `shape` is `[n_comp, n_tokens, pos0, n_head,
    /// head_dim, ratio]`.
    fn indexer_scores(
        &self,
        op: &'static str,
        scores: &mut impl Output,
        q: &impl Input,
        weights: &impl Input,
        index_comp: &impl Input,
        shape: [u32; 6],
        scale: f32,
        causal: bool,
        scratch: Option<DeviceSliceMut<'_, u8>>,
    ) -> Result<()> {
        let [n_comp, n_tokens, pos0, n_head, head_dim, ratio] = shape;
        let scores = typed_mut::<f32>(op, scores.output())?;
        let (q, weights, index_comp) = (
            typed::<f32>(op, q.input())?,
            typed::<f32>(op, weights.input())?,
            typed::<f32>(op, index_comp.input())?,
        );
        let stream = self.stream();
        if n_head != HEADS || head_dim != HEAD_DIM {
            return launched(
                op,
                generated::indexer_scores(
                    stream, scores, q, weights, index_comp, n_comp, n_tokens, pos0, n_head,
                    head_dim, ratio, scale, causal,
                ),
            );
        }
        if n_tokens == 1 || self.batch_mode() == BatchMode::Verifier {
            return launched(
                op,
                generated::indexer_score_direct(
                    stream, scores, q, weights, index_comp, n_comp, n_tokens, pos0, ratio, scale,
                    causal,
                ),
            );
        }
        let keys = u64::from(n_comp).div_ceil(128) * 128 * 128;
        let queries = u64::from(n_tokens).div_ceil(16) * 16 * u64::from(HEADS * HEAD_DIM);
        let packed = (keys + queries) * F16;
        if let Some(scratch) = scratch.filter(|s| s.len() as u64 >= packed) {
            let scratch = typed_mut::<Half>(op, scratch.range(0..packed as usize))?;
            let (mut packed_keys, mut packed_queries) = scratch.split_at(keys as usize);
            launched(
                op,
                generated::indexer_pack_keys(stream, packed_keys.reborrow(), index_comp, n_comp),
            )?;
            launched(
                op,
                generated::indexer_pack_queries(stream, packed_queries.reborrow(), q, n_tokens),
            )?;
            return launched(
                op,
                generated::indexer_scores_wmma_packed(
                    stream,
                    scores,
                    packed_queries.as_const(),
                    weights,
                    packed_keys.as_const(),
                    n_comp,
                    n_tokens,
                    pos0,
                    ratio,
                    scale,
                    causal,
                ),
            );
        }
        launched(
            op,
            generated::indexer_scores_wmma(
                stream, scores, q, weights, index_comp, n_comp, n_tokens, pos0, ratio, scale,
                causal,
            ),
        )
    }

    /// `ds4_gpu_indexer_topk_tensor`: the `top_k` best of `n_comp` scores
    /// per token into `selected`: radix selection for 512 of up to 32,768
    /// scores, a bitonic sort while one block holds the row, else a tree of
    /// chunk selections in the scratch, and an insertion sort for other
    /// widths.
    pub fn indexer_topk(
        &self,
        selected: &mut impl Output,
        scores: &impl Input,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
    ) -> Result<()> {
        const OP: &str = "indexer topk";
        ensure(
            OP,
            n_comp != 0 && n_tokens != 0 && top_k != 0 && top_k <= n_comp,
        )?;
        let mut selected = typed_mut::<i32>(OP, selected.output())?;
        let scores = typed::<f32>(OP, scores.input())?;
        let stream = self.stream();
        // Partial selection also avoids sorting most scores in wide prefill
        // batches.
        if top_k == 512 && n_comp > 1024 && n_comp <= 32768 {
            return launched(
                OP,
                generated::indexer_topk_partial(stream, selected, scores, n_comp, n_tokens),
            );
        }
        let sort = match (top_k, n_comp) {
            (512 | 1024, ..=1024) => generated::indexer_topk_1024,
            (1024, ..=2048) => generated::indexer_topk_2048,
            (1024 | 2048, ..=4096) => generated::indexer_topk_4096,
            (1024 | 2048, ..=8192) => {
                let cub = generated::indexer_topk_8192_cub(
                    stream,
                    selected.reborrow(),
                    scores,
                    n_comp,
                    n_tokens,
                    top_k,
                );
                if launched(OP, cub)? == Launch::Launched {
                    return Ok(());
                }
                generated::indexer_topk_8192
            }
            (512 | 1024 | 2048, _) => {
                return self.indexer_topk_tree(OP, selected, scores, n_comp, n_tokens, top_k);
            }
            _ => generated::indexer_topk_serial,
        };
        launched(OP, sort(stream, selected, scores, n_comp, n_tokens, top_k))
    }

    /// The top-k tree: the best of every 4,096-score chunk, merged in
    /// groups until one block holds the remaining candidates.
    fn indexer_topk_tree(
        &self,
        op: &'static str,
        selected: DeviceSliceMut<'_, i32>,
        scores: gufo_hip::DeviceSlice<'_, f32>,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
    ) -> Result<()> {
        let n_chunks = n_comp.div_ceil(TOPK_CHUNK);
        let merge_group = TOPK_CHUNK / top_k;
        let candidate_stride =
            u32::try_from(u64::from(n_chunks) * u64::from(top_k)).map_err(|_| Error { op })?;
        let mut n_sets = n_chunks;
        let mut per_token = u64::from(candidate_stride);
        while n_sets > merge_group {
            n_sets = n_sets.div_ceil(merge_group);
            per_token += u64::from(n_sets) * u64::from(top_k);
        }
        let bytes = product(op, &[u64::from(n_tokens), per_token, I32])?;
        let Some(mut scratch) = self.scratch(bytes, "indexer topk tree") else {
            return Err(Error { op });
        };
        let all = typed_mut::<i32>(op, scratch.as_mut_slice().range(0..bytes as usize))?;
        let stream = self.stream();
        // Each level's candidate sets follow the previous level's.
        let width = |stride: u32| n_tokens as usize * stride as usize;
        let (mut cur, mut rest) = all.split_at(width(candidate_stride));
        launched(
            op,
            generated::indexer_topk_chunk(
                stream,
                cur.reborrow(),
                scores,
                n_comp,
                n_tokens,
                top_k,
                candidate_stride,
            ),
        )?;
        let (mut n_sets, mut cur_stride) = (n_chunks, candidate_stride);
        while n_sets > merge_group {
            let next_sets = n_sets.div_ceil(merge_group);
            let next_stride = next_sets * top_k;
            let (mut next, tail) = rest.split_at(width(next_stride));
            rest = tail;
            launched(
                op,
                generated::indexer_topk_tree_merge(
                    stream,
                    next.reborrow(),
                    cur.as_const(),
                    scores,
                    n_comp,
                    n_tokens,
                    top_k,
                    n_sets,
                    merge_group,
                    cur_stride,
                    next_stride,
                ),
            )?;
            cur = next;
            n_sets = next_sets;
            cur_stride = next_stride;
        }
        launched(
            op,
            generated::indexer_topk_merge(
                stream,
                selected,
                cur.as_const(),
                scores,
                n_comp,
                n_tokens,
                top_k,
                n_sets * top_k,
                cur_stride,
            ),
        )
    }

    /// The Hadamard FP4 rounding of `n_rows` rows of 128 values, in place.
    pub fn indexer_qat(&self, x: &mut impl Output, n_rows: u32, head_dim: u32) -> Result<()> {
        const OP: &str = "indexer qat";
        launched(
            OP,
            generated::indexer_qat(self.stream(), typed_mut(OP, x.output())?, n_rows, head_dim),
        )
    }
}
