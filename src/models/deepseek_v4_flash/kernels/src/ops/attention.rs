//! Raw KV stores and attention: the routing of
//! `ds4_rocm_attention_launch.hip.hpp` over the generated launchers of
//! `launch.toml`, hipBLAS and the projections' scratch and F16 weight
//! cache, the output projection included.

use std::cell::RefMut;
use std::sync::atomic::{AtomicBool, Ordering};

use gufo_hip::blas::{Op, Strides};
use gufo_hip::{DeviceBuffer, DeviceSlice, DeviceSliceMut, F16 as Half, Launch};

use crate::backend::{Backend, BatchMode, Error, Result};
use crate::generated;
use crate::tensor::{Input, Output};

use super::matmul::Cached;

/// Rows from which an attention batch is a prompt chunk for the rocWMMA
/// producer (`DS4_ROCM_ATTENTION_WIDE_ROWS`): it converts Q and KV to F16.
const ATTENTION_WIDE_ROWS: u32 = 96;

/// The first rocWMMA launch of each route says so, as the C++ runtime did.
static RESUMED_WMMA_NOTICE: AtomicBool = AtomicBool::new(false);
static PREFILL_WMMA_NOTICE: AtomicBool = AtomicBool::new(false);

use super::{F16, F32, I32, Rope, Weight, ensure, launched, need, product, typed, typed_mut};

impl Backend {
    /// `kv` (one row) into row `row % raw_cap` of the ring.
    pub fn store_raw_kv(
        &self,
        raw_cache: &mut impl Output,
        kv: &impl Input,
        raw_cap: u32,
        row: u32,
        head_dim: u32,
    ) -> Result<()> {
        self.store_raw_kv_batch(raw_cache, kv, raw_cap, row, 1, head_dim)
    }

    /// Rows `(pos0 + t) % raw_cap` of the ring. More tokens than ring rows
    /// would race on shared rows, so the wrapper refuses them.
    pub fn store_raw_kv_batch(
        &self,
        raw_cache: &mut impl Output,
        kv: &impl Input,
        raw_cap: u32,
        pos0: u32,
        n_tokens: u32,
        head_dim: u32,
    ) -> Result<()> {
        const OP: &str = "store raw kv batch";
        ensure(OP, n_tokens <= raw_cap)?;
        launched(
            OP,
            generated::store_raw_kv_batch(
                self.stream(),
                typed_mut(OP, raw_cache.output())?,
                typed(OP, kv.input())?,
                raw_cap,
                pos0,
                n_tokens,
                head_dim,
            ),
        )
    }

    /// Quantizes `kv` in place, then stores it, as the C++ launcher did;
    /// both steps are checked before the first launch.
    pub fn kv_fp8_store_raw(
        &self,
        kv: &mut (impl Output + Input),
        raw_cache: &mut impl Output,
        raw_cap: u32,
        row: u32,
        head_dim: u32,
        n_rot: u32,
    ) -> Result<()> {
        const OP: &str = "kv fp8 store raw";
        ensure(OP, raw_cap != 0 && n_rot <= head_dim)?;
        need(
            OP,
            raw_cache,
            &[u64::from(raw_cap), u64::from(head_dim), F32],
        )?;
        need(OP, kv, &[u64::from(head_dim), F32])?;
        self.fp8_kv_quantize(kv, 1, head_dim, n_rot)?;
        self.store_raw_kv(raw_cache, kv, raw_cap, row, head_dim)
    }

    /// One token over `n_raw` rows of the raw ring and `n_comp` compressed
    /// rows, optionally masked.
    pub fn attention_decode_heads(
        &self,
        heads: &mut impl Output,
        sinks_weight: Weight,
        q: &impl Input,
        raw_kv: &impl Input,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        comp_kv: Option<&dyn Input>,
        n_comp: u32,
        comp_mask: Option<&dyn Input>,
        n_head: u32,
        head_dim: u32,
    ) -> Result<()> {
        const OP: &str = "attention decode heads";
        ensure(OP, n_comp == 0 || comp_kv.is_some())?;
        let sinks = self.sinks(OP, sinks_weight, n_head)?;
        let raw_kv = typed(OP, raw_kv.input())?;
        // Absent compressed rows pass the raw rows, which the kernel never
        // reads then.
        let comp_kv = comp_kv
            .map(|c| typed(OP, c.input()))
            .transpose()?
            .unwrap_or(raw_kv);
        let comp_mask = comp_mask.map(|m| typed(OP, m.input())).transpose()?;
        launched(
            OP,
            generated::attention_decode_heads(
                self.stream(),
                typed_mut(OP, heads.output())?,
                sinks,
                typed(OP, q.input())?,
                raw_kv,
                n_raw,
                raw_cap,
                raw_start,
                comp_kv,
                n_comp,
                comp_mask,
                n_head,
                head_dim,
            ),
        )
    }

    /// Causal attention of `n_tokens` over their own raw rows (the first
    /// chunk of a prompt).
    pub fn attention_prefill_raw_heads(
        &self,
        heads: &mut impl Output,
        sinks_weight: Weight,
        q: &impl Input,
        raw_kv: &impl Input,
        n_tokens: u32,
        window: u32,
        n_head: u32,
        head_dim: u32,
    ) -> Result<()> {
        const OP: &str = "attention prefill raw heads";
        ensure(
            OP,
            n_tokens != 0 && n_head != 0 && head_dim != 0 && window <= 256,
        )?;
        let sinks = self.sinks(OP, sinks_weight, n_head)?;
        let (heads, q) = (typed_mut(OP, heads.output())?, typed(OP, q.input())?);
        let raw_kv = typed(OP, raw_kv.input())?;
        let stream = self.stream();
        if n_tokens > 1 && head_dim == 512 && (if window != 0 { window } else { n_tokens }) <= 768 {
            return launched(
                OP,
                generated::attention_static_online(
                    stream, heads, sinks, q, raw_kv, raw_kv, n_tokens, 0, window, 1, n_head,
                    head_dim,
                ),
            );
        }
        if n_tokens > 1 && head_dim == 512 {
            // Scores of every head through hipBLAS, then the values.
            let (n, keys, h, d) = (
                n_tokens as usize,
                n_tokens as usize,
                n_head as usize,
                head_dim as usize,
            );
            let score_bytes = product(
                OP,
                &[
                    u64::from(n_head),
                    u64::from(n_tokens),
                    u64::from(n_tokens),
                    F32,
                ],
            )?;
            let out_offset = score_bytes.next_multiple_of(256);
            let out_bytes = product(
                OP,
                &[
                    u64::from(n_head),
                    u64::from(n_tokens),
                    u64::from(head_dim),
                    F32,
                ],
            )?;
            let Some(mut scratch) = self.scratch(out_offset + out_bytes, "attention raw hipblas")
            else {
                return Err(Error { op: OP });
            };
            let (scores, out) = scratch
                .as_mut_slice()
                .range(0..(out_offset + out_bytes) as usize)
                .split_at(out_offset as usize);
            let mut scores = typed_mut::<f32>(OP, scores.range(0..score_bytes as usize))?;
            let mut out = typed_mut::<f32>(OP, out)?;
            let alpha = 1.0f32 / (head_dim as f32).sqrt();
            let shared = |stride_b, stride_c| Strides {
                count: h,
                a: 0,
                b: stride_b,
                c: stride_c,
            };
            self.sgemm_batched(
                OP,
                (Op::T, Op::N),
                (keys, n, d),
                alpha,
                (raw_kv, d),
                (q, h * d),
                (scores.reborrow(), keys),
                shared(d, keys * n),
            )?;
            launched(
                OP,
                generated::attention_prefill_raw_softmax(
                    stream,
                    scores.reborrow(),
                    sinks,
                    n_tokens,
                    window,
                    n_tokens,
                    n_head,
                ),
            )?;
            self.sgemm_batched(
                OP,
                (Op::N, Op::N),
                (d, n, keys),
                1.0,
                (raw_kv, d),
                (scores.as_const(), keys),
                (out.reborrow(), d),
                shared(keys * n, d * n),
            )?;
            return launched(
                OP,
                generated::attention_prefill_unpack_heads(
                    stream,
                    heads,
                    out.as_const(),
                    n_tokens,
                    n_head,
                    head_dim,
                ),
            );
        }
        let raw = generated::attention_prefill_raw(
            stream, heads, sinks, q, raw_kv, n_tokens, window, n_head, head_dim,
        );
        ensure(OP, launched(OP, raw)? == Launch::Launched)
    }

    /// `n_tokens` from `pos0` over the raw ring.
    pub fn attention_decode_raw_batch_heads(
        &self,
        heads: &mut impl Output,
        sinks_weight: Weight,
        q: &impl Input,
        raw_kv: &impl Input,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        window: u32,
        n_head: u32,
        head_dim: u32,
    ) -> Result<()> {
        self.attention_decode_batch(
            "attention decode raw batch heads",
            typed_mut("attention decode raw batch heads", heads.output())?,
            sinks_weight,
            q.input(),
            raw_kv.input(),
            None,
            None,
            n_tokens,
            pos0,
            n_raw,
            raw_cap,
            raw_start,
            0,
            window,
            1,
            n_head,
            head_dim,
        )
    }

    /// As the raw batch plus `n_comp` compressed rows of ratio `ratio`,
    /// optionally masked.
    pub fn attention_decode_mixed_batch_heads(
        &self,
        heads: &mut impl Output,
        sinks_weight: Weight,
        q: &impl Input,
        raw_kv: &impl Input,
        comp_kv: Option<&dyn Input>,
        comp_mask: Option<&dyn Input>,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) -> Result<()> {
        const OP: &str = "attention decode mixed batch heads";
        self.attention_decode_batch(
            OP,
            typed_mut(OP, heads.output())?,
            sinks_weight,
            q.input(),
            raw_kv.input(),
            comp_kv.map(|c| c.input()),
            comp_mask.map(|m| m.input()),
            n_tokens,
            pos0,
            n_raw,
            raw_cap,
            raw_start,
            n_comp,
            window,
            ratio,
            n_head,
            head_dim,
        )
    }

    /// `attention_decode_batch_launch`: decode attention of a batch of rows.
    fn attention_decode_batch(
        &self,
        op: &'static str,
        mut heads: DeviceSliceMut<'_, f32>,
        sinks_weight: Weight,
        q: DeviceSlice<'_, u8>,
        raw_kv: DeviceSlice<'_, u8>,
        comp_kv: Option<DeviceSlice<'_, u8>>,
        comp_mask: Option<DeviceSlice<'_, u8>>,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) -> Result<()> {
        ensure(
            op,
            n_tokens != 0 && n_raw != 0 && n_raw <= raw_cap && raw_start < raw_cap,
        )?;
        ensure(op, n_comp == 0 || (comp_kv.is_some() && ratio != 0))?;
        let sinks = self.sinks(op, sinks_weight, n_head)?;
        let (q, raw_kv) = (typed::<f32>(op, q)?, typed::<f32>(op, raw_kv)?);
        // Absent compressed rows pass the raw rows, which no kernel reads then.
        let comp = comp_kv
            .map(|c| typed::<f32>(op, c))
            .transpose()?
            .unwrap_or(raw_kv);
        let mask = comp_mask.map(|m| typed::<f32>(op, m)).transpose()?;
        let stream = self.stream();
        // A verification block replays decode per row, which keeps every row
        // bitwise equal to autoregressive decode.
        if self.batch_mode() == BatchMode::Verifier && n_tokens <= 6 {
            let first_raw_pos = pos0.wrapping_add(n_tokens).wrapping_sub(n_raw);
            let row = n_head as usize * head_dim as usize;
            for t in 0..n_tokens {
                let end = pos0.wrapping_add(t).wrapping_add(1);
                let first = if window != 0 && end > window {
                    first_raw_pos.max(end - window)
                } else {
                    first_raw_pos
                };
                let row_raw = end.wrapping_sub(first);
                let row_start = raw_start.wrapping_add(first.wrapping_sub(first_raw_pos)) % raw_cap;
                let row_comp = end
                    .checked_div(ratio)
                    .map_or(0, |visible| n_comp.min(visible));
                let (t, cols) = (t as usize, n_comp as usize);
                let row_mask = match mask {
                    Some(mask) => {
                        ensure(op, (t + 1) * cols <= mask.len())?;
                        Some(mask.range(t * cols..t * cols + row_comp as usize))
                    }
                    None => None,
                };
                ensure(op, (t + 1) * row <= heads.len() && (t + 1) * row <= q.len())?;
                launched(
                    op,
                    generated::attention_decode_heads(
                        stream,
                        heads.reborrow().range(t * row..(t + 1) * row),
                        sinks,
                        q.range(t * row..(t + 1) * row),
                        raw_kv,
                        row_raw,
                        raw_cap,
                        row_start,
                        comp,
                        row_comp,
                        row_mask,
                        n_head,
                        head_dim,
                    ),
                )?;
            }
            return Ok(());
        }
        let online = |heads: DeviceSliceMut<'_, f32>| {
            generated::attention_decode_online(
                stream, heads, sinks, q, raw_kv, comp, n_tokens, pos0, n_raw, raw_cap, raw_start,
                n_comp, window, ratio, n_head, head_dim,
            )
        };
        // The score table of the generic kernel holds 7936 compressed rows.
        if n_comp > 7936 {
            if mask.is_none() && head_dim == 512 {
                return launched(op, online(heads));
            }
            eprintln!("ds4: ROCm attention score buffer too small for {n_comp} compressed rows");
            return Err(Error { op });
        }
        // A prompt longer than one chunk resumes through this route: the
        // rocWMMA producer serves its ratio-128 layers at prompt-chunk width.
        if mask.is_none()
            && self.projections.gfx1151.get()
            && n_tokens >= ATTENTION_WIDE_ROWS
            && n_comp != 0
            && n_head == 64
            && head_dim == 512
            && window != 0
            && window <= 256
            && ratio != 0
        {
            launched(
                op,
                generated::attention_mixed_wmma(
                    stream, heads, sinks, q, raw_kv, comp, n_tokens, pos0, n_raw, raw_cap,
                    raw_start, n_comp, window, ratio, n_head, head_dim,
                ),
            )?;
            if !RESUMED_WMMA_NOTICE.swap(true, Ordering::Relaxed) {
                eprintln!("ds4: ROCm resumed mixed-window chunk using wave32 rocWMMA");
            }
            return Ok(());
        }
        if mask.is_none() && n_tokens > 1 && head_dim == 512 {
            return launched(op, online(heads));
        }
        launched(
            op,
            generated::attention_decode_mixed(
                stream, heads, sinks, q, raw_kv, comp, mask, n_tokens, pos0, n_raw, raw_cap,
                raw_start, n_comp, window, ratio, n_head, head_dim,
            ),
        )
    }

    /// The raw ring plus the compressed rows `topk` selects (`top_k` per
    /// token), from the F32 cache or its F16 mirror (`comp_kv_f16`, prompt
    /// chunks only). The kernels skip indices outside the visible rows.
    pub fn attention_indexed_mixed_batch_heads(
        &self,
        heads: &mut impl Output,
        sinks_weight: Weight,
        q: &impl Input,
        raw_kv: &impl Input,
        comp_kv: &impl Input,
        comp_kv_f16: bool,
        topk: &impl Input,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        top_k: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) -> Result<()> {
        const OP: &str = "attention indexed mixed batch heads";
        ensure(
            OP,
            n_tokens != 0 && n_raw != 0 && n_raw <= raw_cap && raw_start < raw_cap,
        )?;
        ensure(OP, n_comp != 0 && top_k != 0 && top_k <= 1024)?;
        let sinks = self.sinks(OP, sinks_weight, n_head)?;
        let mut heads = typed_mut::<f32>(OP, heads.output())?;
        let (q, raw_kv) = (
            typed::<f32>(OP, q.input())?,
            typed::<f32>(OP, raw_kv.input())?,
        );
        let mut topk = typed::<i32>(OP, topk.input())?;
        let stream = self.stream();
        let wmma =
            n_tokens >= 128 && n_head == 64 && head_dim == 512 && top_k == 512 && window <= 256;
        ensure(OP, !comp_kv_f16 || wmma)?;
        if comp_kv_f16 {
            let comp = typed::<Half>(OP, comp_kv.input())?;
            let sorted = self.sorted_topk(OP, topk, n_tokens)?;
            let sorted_len = n_tokens as usize * 512 * 4;
            return launched(
                OP,
                generated::attention_indexed_wmma_f16(
                    stream,
                    heads,
                    sinks,
                    q,
                    raw_kv,
                    comp,
                    typed(OP, sorted.as_slice().range(0..sorted_len))?,
                    n_tokens,
                    pos0,
                    n_raw,
                    raw_cap,
                    raw_start,
                    n_comp,
                    top_k,
                    window,
                    ratio,
                    n_head,
                    head_dim,
                ),
            );
        }
        let comp = typed::<f32>(OP, comp_kv.input())?;
        let one = |heads: DeviceSliceMut<'_, f32>,
                   q: DeviceSlice<'_, f32>,
                   topk: DeviceSlice<'_, i32>,
                   n_raw: u32,
                   raw_start: u32,
                   pos: u32| {
            generated::attention_decode_indexed_one(
                stream, heads, q, raw_kv, comp, topk, sinks, n_raw, raw_cap, raw_start, n_comp,
                top_k, pos, ratio, n_head, head_dim,
            )
        };
        if self.batch_mode() == BatchMode::Verifier && n_tokens > 1 && n_tokens <= 6 {
            let first_raw_pos = pos0.wrapping_add(n_tokens).wrapping_sub(n_raw);
            let row = n_head as usize * head_dim as usize;
            let k = top_k as usize;
            for t in 0..n_tokens {
                let end = pos0.wrapping_add(t).wrapping_add(1);
                let first = if window != 0 && end > window {
                    first_raw_pos.max(end - window)
                } else {
                    first_raw_pos
                };
                let row_raw = end.wrapping_sub(first);
                let row_start = raw_start.wrapping_add(first.wrapping_sub(first_raw_pos)) % raw_cap;
                let t_ = t as usize;
                ensure(
                    OP,
                    (t_ + 1) * row <= heads.len()
                        && (t_ + 1) * row <= q.len()
                        && (t_ + 1) * k <= topk.len(),
                )?;
                launched(
                    OP,
                    one(
                        heads.reborrow().range(t_ * row..(t_ + 1) * row),
                        q.range(t_ * row..(t_ + 1) * row),
                        topk.range(t_ * k..(t_ + 1) * k),
                        row_raw,
                        row_start,
                        pos0.wrapping_add(t),
                    ),
                )?;
            }
            return Ok(());
        }
        if n_tokens == 1 {
            let row = n_head as usize * head_dim as usize;
            ensure(
                OP,
                row <= heads.len() && row <= q.len() && top_k as usize <= topk.len(),
            )?;
            return launched(
                OP,
                one(
                    heads.range(0..row),
                    q.range(0..row),
                    topk.range(0..top_k as usize),
                    n_raw,
                    raw_start,
                    pos0,
                ),
            );
        }
        // The kernels visit the selected rows in ascending order.
        let sorted;
        if top_k == 512 {
            sorted = self.sorted_topk(OP, topk, n_tokens)?;
            topk = typed(OP, sorted.as_slice().range(0..n_tokens as usize * 512 * 4))?;
        }
        if head_dim == 512 {
            if wmma {
                return launched(
                    OP,
                    generated::attention_indexed_wmma(
                        stream, heads, sinks, q, raw_kv, comp, topk, n_tokens, pos0, n_raw,
                        raw_cap, raw_start, n_comp, top_k, window, ratio, n_head, head_dim,
                    ),
                );
            }
            return launched(
                OP,
                generated::attention_indexed_online(
                    stream, heads, sinks, q, raw_kv, comp, topk, n_tokens, pos0, n_raw, raw_cap,
                    raw_start, n_comp, top_k, window, ratio, n_head, head_dim,
                ),
            );
        }
        launched(
            OP,
            generated::attention_indexed_mixed(
                stream, heads, sinks, q, raw_kv, comp, topk, n_tokens, pos0, n_raw, raw_cap,
                raw_start, n_comp, top_k, window, ratio, n_head, head_dim,
            ),
        )
    }

    /// The top-k rows of 512 indices sorted ascending in the scratch.
    fn sorted_topk(
        &self,
        op: &'static str,
        topk: DeviceSlice<'_, i32>,
        n_tokens: u32,
    ) -> Result<RefMut<'_, DeviceBuffer<u8>>> {
        let bytes = product(op, &[u64::from(n_tokens), 512, I32])?;
        let Some(mut scratch) = self.scratch(bytes, "indexed attention topk sort") else {
            return Err(Error { op });
        };
        let dst = typed_mut::<i32>(op, scratch.as_mut_slice().range(0..bytes as usize))?;
        launched(
            op,
            generated::indexed_topk_sort_512(self.stream(), dst, topk, n_tokens),
        )?;
        Ok(scratch)
    }

    /// A prompt chunk over its own raw rows and `n_comp` compressed rows
    /// (`attention_prefill_mixed_launch`).
    pub fn attention_prefill_static_mixed_heads(
        &self,
        heads: &mut impl Output,
        sinks_weight: Weight,
        q: &impl Input,
        raw_kv: &impl Input,
        comp_kv: Option<&dyn Input>,
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) -> Result<()> {
        const OP: &str = "attention prefill static mixed heads";
        ensure(
            OP,
            n_tokens != 0 && ratio != 0 && n_head != 0 && head_dim != 0,
        )?;
        ensure(OP, n_comp == 0 || comp_kv.is_some())?;
        let sinks = self.sinks(OP, sinks_weight, n_head)?;
        let heads = typed_mut::<f32>(OP, heads.output())?;
        let (q, raw_kv) = (
            typed::<f32>(OP, q.input())?,
            typed::<f32>(OP, raw_kv.input())?,
        );
        let comp = comp_kv
            .map(|c| typed::<f32>(OP, c.input()))
            .transpose()?
            .unwrap_or(raw_kv);
        let stream = self.stream();
        // Mixed-window prompt chunks through the rocWMMA producer the indexed
        // layers use; it converts Q and KV to F16, so prompt-chunk width only.
        if self.projections.gfx1151.get()
            && n_tokens >= ATTENTION_WIDE_ROWS
            && n_comp != 0
            && n_head == 64
            && head_dim == 512
            && window != 0
            && window <= 256
        {
            launched(
                OP,
                generated::attention_mixed_wmma(
                    stream, heads, sinks, q, raw_kv, comp, n_tokens, 0, n_tokens, n_tokens, 0,
                    n_comp, window, ratio, n_head, head_dim,
                ),
            )?;
            if !PREFILL_WMMA_NOTICE.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "ds4: ROCm mixed-window prefill using wave32 rocWMMA (heads=32, rows=16, \
                     single-pass)"
                );
            }
            return Ok(());
        }
        let visible = if window != 0 { window } else { n_tokens };
        if n_tokens > 1 && head_dim == 512 && u64::from(visible) + u64::from(n_comp) <= 768 {
            return launched(
                OP,
                generated::attention_static_online(
                    stream, heads, sinks, q, raw_kv, comp, n_tokens, n_comp, window, ratio, n_head,
                    head_dim,
                ),
            );
        }
        if n_tokens > 1 && head_dim == 512 {
            let keys = n_tokens.checked_add(n_comp).ok_or(Error { op: OP })?;
            let layout = |tokens: u32| -> Result<(u64, u64, u64)> {
                let kv = product(OP, &[u64::from(keys), u64::from(head_dim), F32])?;
                let score_offset = kv.next_multiple_of(256);
                let scores = product(
                    OP,
                    &[u64::from(n_head), u64::from(tokens), u64::from(keys), F32],
                )?;
                let out_offset = score_offset + scores.next_multiple_of(256);
                let out = product(
                    OP,
                    &[
                        u64::from(n_head),
                        u64::from(tokens),
                        u64::from(head_dim),
                        F32,
                    ],
                )?;
                Ok((score_offset, out_offset, out_offset + out))
            };
            let (_, _, whole) = layout(n_tokens)?;
            let mut tile = n_tokens;
            let scratch = match self.scratch(whole, "attention mixed hipblas") {
                Some(scratch) => scratch,
                None => {
                    // Tiles of rows within 4 GiB of scratch.
                    while tile > 1 && layout(tile)?.2 > 4 << 30 {
                        tile = tile.div_ceil(2);
                    }
                    self.scratch(layout(tile)?.2, "attention mixed hipblas tiled")
                        .ok_or(Error { op: OP })?
                }
            };
            return self.attention_prefill_mixed_blas(
                OP,
                scratch,
                layout(tile)?,
                tile,
                heads,
                sinks,
                q,
                raw_kv,
                comp,
                n_tokens,
                n_comp,
                window,
                ratio,
                n_head,
                head_dim,
            );
        }
        let max_raw = if window != 0 && window < n_tokens {
            window
        } else {
            n_tokens
        };
        if u64::from(max_raw) + u64::from(n_comp) > 2048 {
            eprintln!(
                "ds4: ROCm attention mixed scalar fallback unsupported for {} scores (cap=2048, \
                 tokens={n_tokens}, comp={n_comp}, window={window})",
                u64::from(max_raw) + u64::from(n_comp)
            );
            return Err(Error { op: OP });
        }
        launched(
            OP,
            generated::attention_prefill_mixed(
                stream, heads, sinks, q, raw_kv, comp, None, n_tokens, n_comp, window, ratio,
                n_head, head_dim,
            ),
        )
    }

    /// The hipBLAS form of the mixed prefill: the packed keys, then per tile
    /// of `tile` rows the scores of every head, their softmax and the values.
    fn attention_prefill_mixed_blas(
        &self,
        op: &'static str,
        mut scratch: RefMut<'_, DeviceBuffer<u8>>,
        (score_offset, out_offset, bytes): (u64, u64, u64),
        tile: u32,
        mut heads: DeviceSliceMut<'_, f32>,
        sinks: DeviceSlice<'_, f32>,
        q: DeviceSlice<'_, f32>,
        raw_kv: DeviceSlice<'_, f32>,
        comp: DeviceSlice<'_, f32>,
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) -> Result<()> {
        let stream = self.stream();
        let keys = n_tokens + n_comp;
        let (kv, rest) = scratch
            .as_mut_slice()
            .range(0..bytes as usize)
            .split_at(score_offset as usize);
        let (scores, out) = rest.split_at((out_offset - score_offset) as usize);
        let mut kv = typed_mut::<f32>(op, kv.range(0..(keys as usize * head_dim as usize * 4)))?;
        let mut scores = typed_mut::<f32>(op, scores)?;
        let mut out = typed_mut::<f32>(op, out)?;
        launched(
            op,
            generated::attention_prefill_pack_mixed_kv(
                stream,
                kv.reborrow(),
                raw_kv,
                comp,
                n_tokens,
                n_comp,
                head_dim,
            ),
        )?;
        let kv = kv.as_const();
        let alpha = 1.0f32 / (head_dim as f32).sqrt();
        let (h, d, k) = (n_head as usize, head_dim as usize, keys as usize);
        let row = h * d;
        for t0 in (0..n_tokens).step_by(tile as usize) {
            let nt = tile.min(n_tokens - t0);
            let (n, t0) = (nt as usize, t0 as usize);
            ensure(
                op,
                (t0 + n) * row <= q.len() && (t0 + n) * row <= heads.len(),
            )?;
            let tile_scores = scores.reborrow().range(0..h * n * k);
            let shared = |stride_b, stride_c| Strides {
                count: h,
                a: 0,
                b: stride_b,
                c: stride_c,
            };
            let mut tile_scores = tile_scores;
            self.sgemm_batched(
                op,
                (Op::T, Op::N),
                (k, n, d),
                alpha,
                (kv, d),
                (q.range(t0 * row..(t0 + n) * row), row),
                (tile_scores.reborrow(), k),
                shared(d, k * n),
            )?;
            let softmax = if tile == n_tokens {
                generated::attention_prefill_mixed_softmax(
                    stream,
                    tile_scores.reborrow(),
                    sinks,
                    None,
                    n_tokens,
                    n_comp,
                    window,
                    ratio,
                    keys,
                    n_head,
                )
            } else {
                generated::attention_prefill_mixed_softmax_tile(
                    stream,
                    tile_scores.reborrow(),
                    sinks,
                    None,
                    n_tokens,
                    t0 as u32,
                    nt,
                    n_comp,
                    window,
                    ratio,
                    keys,
                    n_head,
                )
            };
            launched(op, softmax)?;
            let tile_out = out.reborrow().range(0..h * n * d);
            let mut tile_out = tile_out;
            self.sgemm_batched(
                op,
                (Op::N, Op::N),
                (d, n, k),
                1.0,
                (kv, d),
                (tile_scores.as_const(), k),
                (tile_out.reborrow(), d),
                shared(k * n, d * n),
            )?;
            launched(
                op,
                generated::attention_prefill_unpack_heads(
                    stream,
                    heads.reborrow().range(t0 * row..(t0 + n) * row),
                    tile_out.as_const(),
                    nt,
                    n_head,
                    head_dim,
                ),
            )?;
        }
        Ok(())
    }

    /// `hipblasSgemmStridedBatched` over the heads: `(trans_a, trans_b)`,
    /// `(m, n, k)`, operands with their leading dimensions; a failure is
    /// reported as the C++ runtime did.
    fn sgemm_batched(
        &self,
        op: &'static str,
        (trans_a, trans_b): (Op, Op),
        (m, n, k): (usize, usize, usize),
        alpha: f32,
        (a, lda): (DeviceSlice<'_, f32>, usize),
        (b, ldb): (DeviceSlice<'_, f32>, usize),
        (c, ldc): (DeviceSliceMut<'_, f32>, usize),
        strides: Strides,
    ) -> Result<()> {
        self.blas
            .sgemm_strided_batched(
                self.stream(),
                trans_a,
                trans_b,
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                c,
                ldc,
                strides,
            )
            .map_err(|error| {
                eprintln!("ds4: hipBLAS {op} gemm failed: {error}");
                Error { op }
            })
    }

    /// The sinks of `n_head` heads.
    fn sinks(&self, op: &'static str, sinks: Weight, n_head: u32) -> Result<DeviceSlice<'_, f32>> {
        typed(op, self.weight_bytes(sinks, u64::from(n_head) * F32)?)
    }

    /// The heads of `n_tokens` rows over the same `n_raw` window rows,
    /// without a causal mask.
    pub fn attention_noncausal_raw_batch_heads(
        &self,
        heads: &mut impl Output,
        sinks_weight: Weight,
        q: &impl Input,
        raw_kv: &impl Input,
        n_tokens: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_head: u32,
        head_dim: u32,
    ) -> Result<()> {
        const OP: &str = "attention noncausal raw batch heads";
        let sinks = typed(
            OP,
            self.weight_bytes(sinks_weight, product(OP, &[u64::from(n_head), F32])?)?,
        )?;
        launched(
            OP,
            generated::attention_noncausal_raw_batch_heads(
                self.stream(),
                typed_mut(OP, heads.output())?,
                sinks,
                typed(OP, q.input())?,
                typed(OP, raw_kv.input())?,
                n_tokens,
                n_raw,
                raw_cap,
                raw_start,
                n_head,
                head_dim,
            ),
        )
    }

    /// One token's A projection of the attention output: the grouped heads
    /// to `n_groups * rank` low values (`attention_output_low_q8`).
    pub fn attention_output_low_q8(
        &self,
        low: &mut impl Output,
        w_a: Weight,
        group_dim: u64,
        rank: u64,
        n_groups: u32,
        heads: &impl Input,
    ) -> Result<()> {
        const OP: &str = "attention output low q8";
        let (group_dim, rank) = (dim(OP, group_dim)?, dim(OP, rank)?);
        ensure(OP, group_dim != 0 && rank != 0 && n_groups != 0)?;
        let blocks = u64::from(group_dim.div_ceil(32));
        let a_bytes = product(OP, &[u64::from(n_groups), u64::from(rank), blocks, 34])?;
        let out_a = self.weight_bytes(w_a, a_bytes)?;
        // The split-K16x8 shape of the old-HIP decode path: the full-row
        // reduction is close but crosses FP8 KV midpoints downstream.
        let Some(mut scratch) = self.scratch(
            Self::quantized_bytes(OP, group_dim, n_groups)?,
            "attention output low q8 prequant",
        ) else {
            return Err(Error { op: OP });
        };
        let q = self.quantize(
            OP,
            &mut scratch,
            typed(OP, heads.input())?,
            group_dim,
            n_groups,
        )?;
        launched(
            OP,
            generated::attn_out_a_preq_warp8(
                self.stream(),
                typed_mut(OP, low.output())?,
                out_a,
                q.xq.as_const(),
                q.xscale.as_const(),
                group_dim,
                rank,
                n_groups,
                1,
                32,
            ),
        )
    }

    /// The attention output projection of `n_tokens` rows: A to the low
    /// values, then B to `out_dim` (`attention_output_q8_batch_launch`).
    pub fn attention_output_q8_batch(
        &self,
        out: &mut impl Output,
        low: &mut (impl Output + Input),
        w_a: Weight,
        w_b: Weight,
        group_dim: u64,
        rank: u64,
        n_groups: u32,
        out_dim: u64,
        heads: &impl Input,
        n_tokens: u32,
    ) -> Result<()> {
        const OP: &str = "attention output q8 batch";
        let done = self.attention_output(
            OP, out, low, w_a, w_b, group_dim, rank, n_groups, out_dim, heads, n_tokens, None,
        )?;
        ensure(OP, done)
    }

    /// [`attention_output_q8_batch`](Self::attention_output_q8_batch) with
    /// the inverse rotation of each head's tail folded into the F16 pack of
    /// prompt chunks; `false` where only the separate rotation serves, which
    /// the caller then launches before the plain projection (as in C++, a
    /// failure of this route also falls back).
    pub fn attention_output_q8_batch_inv_rope(
        &self,
        out: &mut impl Output,
        low: &mut (impl Output + Input),
        w_a: Weight,
        w_b: Weight,
        group_dim: u64,
        rank: u64,
        n_groups: u32,
        out_dim: u64,
        heads: &impl Input,
        n_tokens: u32,
        head_dim: u32,
        rope: Rope,
    ) -> bool {
        // Prompt-chunk width only: the backend builds with -ffast-math, and
        // folding the rotation changes the surrounding expression.
        if n_tokens < ATTENTION_WIDE_ROWS {
            return false;
        }
        let fused = self.attention_output(
            "attention output q8 batch inv rope",
            out,
            low,
            w_a,
            w_b,
            group_dim,
            rank,
            n_groups,
            out_dim,
            heads,
            n_tokens,
            Some((head_dim, rope)),
        );
        matches!(fused, Ok(true))
    }

    fn attention_output(
        &self,
        op: &'static str,
        out: &mut impl Output,
        low: &mut (impl Output + Input),
        w_a: Weight,
        w_b: Weight,
        group_dim: u64,
        rank: u64,
        n_groups: u32,
        out_dim: u64,
        heads: &impl Input,
        n_tokens: u32,
        rope: Option<(u32, Rope)>,
    ) -> Result<bool> {
        let (group_dim, rank, out_dim) = (dim(op, group_dim)?, dim(op, rank)?, dim(op, out_dim)?);
        if let Some((head_dim, rope)) = rope
            && !pack_rope_eligible(head_dim, rope.n_rot, group_dim)
        {
            return Ok(false);
        }
        ensure(
            op,
            group_dim != 0 && rank != 0 && n_groups != 0 && out_dim != 0 && n_tokens != 0,
        )?;
        let low_dim = rank.checked_mul(n_groups).ok_or(Error { op })?;
        let blocks_a = u64::from(group_dim.div_ceil(32));
        let a_bytes = product(op, &[u64::from(n_groups), u64::from(rank), blocks_a, 34])?;
        let out_a = self.weight_bytes(w_a, a_bytes)?;
        let blocks_b = u64::from(low_dim.div_ceil(32));
        self.weight_bytes(w_b, product(op, &[u64::from(out_dim), blocks_b, 34])?)?;
        let heads = typed::<f32>(op, heads.input())?;
        let stream = self.stream();
        // A verification block takes the grouped Q8 kernels: the hipBLAS
        // route's macro tile is far wider than its rows.
        let small = self.small_batch_rows();
        let blas_route = !(n_tokens > 1 && n_tokens <= small);
        if rope.is_some() && !blas_route {
            return Ok(false);
        }
        if !blas_route {
            if group_dim.is_multiple_of(32) && n_tokens <= 8 {
                let rows = n_tokens.checked_mul(n_groups).ok_or(Error { op })?;
                let Some(mut scratch) = self.scratch(
                    Self::quantized_bytes(op, group_dim, rows)?,
                    "attention output a session batch prequant",
                ) else {
                    return Err(Error { op });
                };
                let q = self.quantize(op, &mut scratch, heads, group_dim, rows)?;
                let reuse = generated::attn_out_a_preq_reuse(
                    stream,
                    typed_mut(op, low.output())?,
                    out_a,
                    q.xq.as_const(),
                    q.xscale.as_const(),
                    group_dim,
                    rank,
                    n_groups,
                    n_tokens,
                    if n_tokens == 2 { 2 } else { 4 },
                );
                ensure(op, launched(op, reuse)? == Launch::Launched)?;
            } else if group_dim.is_multiple_of(32) {
                launched(
                    op,
                    generated::attn_out_a_sharedx(
                        stream,
                        typed_mut(op, low.output())?,
                        out_a,
                        heads,
                        group_dim,
                        rank,
                        n_groups,
                        n_tokens,
                    ),
                )?;
            } else {
                launched(
                    op,
                    generated::attn_out_a_warp8(
                        stream,
                        typed_mut(op, low.output())?,
                        out_a,
                        heads,
                        group_dim,
                        rank,
                        n_groups,
                        n_tokens,
                    ),
                )?;
            }
            self.q8_projection(
                op,
                typed_mut(op, out.output())?,
                w_b,
                low_dim,
                out_dim,
                typed(op, low.input())?,
                n_tokens,
                Cached::AttentionOutput,
            )?;
            return Ok(true);
        }

        let mut a_f16 = if n_tokens >= 2 {
            self.f16_weight(w_a, group_dim, low_dim, Cached::AttentionOutput, false)?
        } else {
            None
        };
        if let Some(a_index) = a_f16 {
            if let Some(done) = self.attention_output_f16(
                op, out, a_index, w_b, group_dim, rank, n_groups, out_dim, heads, n_tokens, rope,
            )? {
                return Ok(done);
            }
            // Building the output-B cache may fail and release every cached
            // weight, A's included; the cache is then disabled.
            if self.projections.f16_weights.borrow().is_disabled() {
                a_f16 = None;
            }
        }
        if rope.is_some() {
            return Ok(false);
        }
        if let Some(a_index) = a_f16 {
            let heads_h = u64::from(n_groups) * u64::from(n_tokens) * u64::from(group_dim);
            let low_packed = u64::from(n_groups) * u64::from(n_tokens) * u64::from(rank);
            let low_offset = (heads_h * F16).next_multiple_of(256);
            let bytes = low_offset + low_packed * F32;
            let Some(mut scratch) = self.scratch(bytes, "attention output a hipblas") else {
                return Err(Error { op });
            };
            let (heads_h_bytes, low_bytes) = scratch
                .as_mut_slice()
                .range(0..bytes as usize)
                .split_at(low_offset as usize);
            let mut heads_h =
                typed_mut::<Half>(op, heads_h_bytes.range(0..(heads_h * F16) as usize))?;
            let mut packed = typed_mut::<f32>(op, low_bytes)?;
            launched(
                op,
                generated::pack_group_heads_f16(
                    stream,
                    heads_h.reborrow(),
                    heads,
                    n_tokens,
                    n_groups,
                    group_dim,
                ),
            )?;
            {
                let cache = self.projections.f16_weights.borrow();
                let a = cache.values(a_index).ok_or(Error { op })?;
                let (m, n, k) = (rank as usize, n_tokens as usize, group_dim as usize);
                self.blas
                    .gemm_strided_batched(
                        stream,
                        Op::T,
                        Op::N,
                        m,
                        n,
                        k,
                        a,
                        k,
                        heads_h.as_const(),
                        k,
                        packed.reborrow(),
                        m,
                        Strides {
                            count: n_groups as usize,
                            a: m * k,
                            b: n * k,
                            c: m * n,
                        },
                    )
                    .map_err(|error| {
                        eprintln!("ds4: hipBLAS attention output a gemm failed: {error}");
                        Error { op }
                    })?;
            }
            launched(
                op,
                generated::attn_unpack_group_low(
                    stream,
                    typed_mut(op, low.output())?,
                    packed.as_const(),
                    n_tokens,
                    n_groups,
                    rank,
                ),
            )?;
        } else {
            let rows = n_tokens.checked_mul(n_groups).ok_or(Error { op })?;
            let Some(mut scratch) = self.scratch(
                Self::quantized_bytes(op, group_dim, rows)?,
                "attention output a q8 prequant",
            ) else {
                return Err(Error { op });
            };
            let q = self.quantize(op, &mut scratch, heads, group_dim, rows)?;
            launched(
                op,
                generated::attn_out_a_preq_warp8(
                    stream,
                    typed_mut(op, low.output())?,
                    out_a,
                    q.xq.as_const(),
                    q.xscale.as_const(),
                    group_dim,
                    rank,
                    n_groups,
                    n_tokens,
                    8,
                ),
            )?;
        }
        let low_rows = typed(op, low.input())?;
        if self.q8_f16_gemm(
            op,
            typed_mut(op, out.output())?,
            w_b,
            low_dim,
            out_dim,
            low_rows,
            n_tokens,
            Cached::AttentionOutput,
        )? {
            return Ok(true);
        }
        self.q8_projection(
            op,
            typed_mut(op, out.output())?,
            w_b,
            low_dim,
            out_dim,
            low_rows,
            n_tokens,
            Cached::AttentionOutput,
        )?;
        Ok(true)
    }

    /// The packed F16 route: A batched over the groups into the interleaved
    /// low rows (each group's rank block at `g * rank` of a low row), B
    /// through the transposed weight's rocWMMA tile or hipBLAS. `None` when
    /// it declines (no B expansion, or BLAS failed and the caller falls
    /// back).
    fn attention_output_f16(
        &self,
        op: &'static str,
        out: &mut impl Output,
        a_index: usize,
        w_b: Weight,
        group_dim: u32,
        rank: u32,
        n_groups: u32,
        out_dim: u32,
        heads: DeviceSlice<'_, f32>,
        n_tokens: u32,
        rope: Option<(u32, Rope)>,
    ) -> Result<Option<bool>> {
        let stream = self.stream();
        let low_dim = rank * n_groups;
        let b_t = self.f16_weight(w_b, low_dim, out_dim, Cached::AttentionOutput, true)?;
        let b = match b_t {
            Some(_) => None,
            None => self.f16_weight(w_b, low_dim, out_dim, Cached::AttentionOutput, false)?,
        };
        if b_t.is_none() && b.is_none() {
            return Ok(None);
        }
        let gfx1151 = self.projections.gfx1151.get();
        let padded_eligible = |m: u32, n_pad: u32, k: u32| {
            gfx1151 && n_pad.is_multiple_of(128) && k.is_multiple_of(32) && m.is_multiple_of(128)
        };
        let mut out = typed_mut::<f32>(op, out.output())?;
        // The rocWMMA B tile covers whole column blocks: stage and compute a
        // width rounded up to one (zero padding, never read back), at
        // prompt-chunk width only.
        let mut pad_n = n_tokens.div_ceil(128) * 128;
        if b_t.is_none()
            || n_tokens < ATTENTION_WIDE_ROWS
            || !padded_eligible(out_dim, pad_n, low_dim)
            || (out.len() as u64) < u64::from(pad_n) * u64::from(out_dim)
        {
            pad_n = n_tokens;
        }
        let heads_h = u64::from(n_groups) * u64::from(n_tokens) * u64::from(group_dim);
        let low_h = u64::from(n_tokens) * u64::from(low_dim);
        let low_h_pad = u64::from(pad_n) * u64::from(low_dim);
        let low_offset = (heads_h * F16).next_multiple_of(256);
        let bytes = low_offset + low_h_pad * F16;
        let Some(mut scratch) = self.scratch(bytes, "attention output packed b hipblas") else {
            return Err(Error { op });
        };
        let (heads_bytes, low_bytes) = scratch
            .as_mut_slice()
            .range(0..bytes as usize)
            .split_at(low_offset as usize);
        let mut heads_h = typed_mut::<Half>(op, heads_bytes.range(0..(heads_h * F16) as usize))?;
        let mut low_h_all = low_bytes;
        if pad_n != n_tokens {
            let pad = low_h_all
                .reborrow()
                .range((low_h * F16) as usize..(low_h_pad * F16) as usize);
            launched(
                op,
                generated::zero_bytes(stream, pad, ((low_h_pad - low_h) * F16) as usize),
            )?;
        }
        let mut low_h = typed_mut::<Half>(op, low_h_all)?;
        match rope {
            Some((head_dim, rope)) => {
                let packed = generated::pack_group_heads_rope_f16(
                    stream,
                    heads_h.reborrow(),
                    heads,
                    n_tokens,
                    n_groups,
                    group_dim,
                    head_dim,
                    rope.n_rot,
                    rope.pos0,
                    rope.n_ctx_orig,
                    rope.freq_base,
                    rope.freq_scale,
                    rope.ext_factor,
                    rope.attn_factor,
                    rope.beta_fast,
                    rope.beta_slow,
                );
                if launched(op, packed)? != Launch::Launched {
                    return Ok(Some(false));
                }
            }
            None => launched(
                op,
                generated::pack_group_heads_f16(
                    stream,
                    heads_h.reborrow(),
                    heads,
                    n_tokens,
                    n_groups,
                    group_dim,
                ),
            )?,
        }
        let cache = self.projections.f16_weights.borrow();
        let a = cache.values(a_index).ok_or(Error { op })?;
        // A: rocBLAS reaches 2.7 TFLOP/s on this strided-batched shape; the
        // 128-row tile keeps each group's A panel resident across the sweep.
        let a_wmma = gfx1151
            && rank == 128
            && n_tokens.is_multiple_of(128)
            && group_dim.is_multiple_of(32)
            && n_groups > 0
            && n_groups <= u32::from(u16::MAX)
            && (n_tokens / 128) * n_groups >= 240;
        let a_done = if a_wmma {
            launched(
                op,
                generated::gemm_f16_wmma_batched(
                    stream,
                    low_h.reborrow(),
                    a,
                    heads_h.as_const(),
                    rank,
                    n_tokens,
                    group_dim,
                    low_dim,
                    n_groups,
                ),
            )?;
            Ok(())
        } else {
            let (m, n, k) = (rank as usize, n_tokens as usize, group_dim as usize);
            self.blas.gemm_strided_batched(
                stream,
                Op::T,
                Op::N,
                m,
                n,
                k,
                a,
                k,
                heads_h.as_const(),
                k,
                low_h.reborrow(),
                low_dim as usize,
                Strides {
                    count: n_groups as usize,
                    a: m * k,
                    b: n * k,
                    c: m,
                },
            )
        };
        if let Err(error) = a_done {
            eprintln!("ds4: hipBLAS attention output packed A failed: {error}; falling back");
            return Ok(None);
        }
        // B: the rocWMMA tile moves 3.22 GB against Tensile's 5.7 GB here
        // and needs the transposed cache (op N); hipBLASLt stays off for this
        // route, whose faster algorithm fails the retained trajectory.
        let (b_values, b_op, b_lda) = match b_t {
            Some(index) => (cache.values(index).ok_or(Error { op })?, Op::N, out_dim),
            None => (
                cache.values(b.ok_or(Error { op })?).ok_or(Error { op })?,
                Op::T,
                low_dim,
            ),
        };
        if b_t.is_some() && padded_eligible(out_dim, pad_n, low_dim) {
            launched(
                op,
                generated::gemm_f16_wmma_padded(
                    stream,
                    out.reborrow().range(0..pad_n as usize * out_dim as usize),
                    b_values,
                    low_h.as_const(),
                    out_dim,
                    pad_n,
                    low_dim,
                ),
            )?;
            return Ok(Some(true));
        }
        let (m, n, k) = (out_dim as usize, n_tokens as usize, low_dim as usize);
        let low_rows = low_h.as_const().range(0..n * k);
        match self.blas.gemm(
            stream,
            b_op,
            Op::N,
            m,
            n,
            k,
            b_values,
            b_lda as usize,
            low_rows,
            k,
            out,
            m,
        ) {
            Ok(()) => Ok(Some(true)),
            Err(error) => {
                eprintln!(
                    "ds4: hipBLAS attention output interleaved B failed: {error}; falling back"
                );
                Ok(None)
            }
        }
    }

    /// The exact multi-request verifier output projection: one accumulator
    /// per row in the narrow decode kernels, each Q8 block read once, so every
    /// row stays bit-identical to decode.
    pub fn attention_output_q8_exact_batch(
        &self,
        out: &mut impl Output,
        low: &mut (impl Output + Input),
        w_a: Weight,
        w_b: Weight,
        group_dim: u64,
        rank: u64,
        n_groups: u32,
        out_dim: u64,
        heads: &impl Input,
        n_tokens: u32,
    ) -> Result<()> {
        const OP: &str = "attention output q8 exact batch";
        let (group_dim, rank, out_dim) = (dim(OP, group_dim)?, dim(OP, rank)?, dim(OP, out_dim)?);
        ensure(
            OP,
            matches!(n_tokens, 4..=8 | 10 | 12 | 16 | 18 | 24 | 32)
                && group_dim != 0
                && group_dim.is_multiple_of(32)
                && rank != 0
                && n_groups != 0
                && out_dim != 0,
        )?;
        let low_dim = rank.checked_mul(n_groups).ok_or(Error { op: OP })?;
        let blocks_a = u64::from(group_dim / 32);
        let out_a = self.weight_bytes(
            w_a,
            product(OP, &[u64::from(n_groups), u64::from(rank), blocks_a, 34])?,
        )?;
        let blocks_b = u64::from(low_dim.div_ceil(32));
        let out_b = self.weight_bytes(w_b, product(OP, &[u64::from(out_dim), blocks_b, 34])?)?;
        let stream = self.stream();
        let rows = n_tokens.checked_mul(n_groups).ok_or(Error { op: OP })?;
        {
            let Some(mut scratch) = self.scratch(
                Self::quantized_bytes(OP, group_dim, rows)?,
                "attention output exact batch a prequant",
            ) else {
                return Err(Error { op: OP });
            };
            let q = self.quantize(OP, &mut scratch, typed(OP, heads.input())?, group_dim, rows)?;
            let reuse = generated::attn_out_a_preq_reuse(
                stream,
                typed_mut(OP, low.output())?,
                out_a,
                q.xq.as_const(),
                q.xscale.as_const(),
                group_dim,
                rank,
                n_groups,
                n_tokens,
                4,
            );
            ensure(OP, launched(OP, reuse)? == Launch::Launched)?;
        }
        let Some(mut scratch) = self.scratch(
            Self::quantized_bytes(OP, low_dim, n_tokens)?,
            "attention output exact batch b prequant",
        ) else {
            return Err(Error { op: OP });
        };
        let q = self.quantize(OP, &mut scratch, typed(OP, low.input())?, low_dim, n_tokens)?;
        let reuse = generated::matmul_q8_preq_reuse(
            stream,
            typed_mut(OP, out.output())?,
            out_b,
            q.xq.as_const(),
            q.xscale.as_const(),
            low_dim,
            out_dim,
            n_tokens,
            1,
        );
        ensure(OP, launched(OP, reuse)? == Launch::Launched)
    }
}

/// Whether the inverse rotation can fold into the F16 pack: it walks each
/// heads row as four-value vectors, so no rotated pair may straddle one
/// (`ds4_attn_pack_rope_eligible`).
fn pack_rope_eligible(head_dim: u32, n_rot: u32, group_dim: u32) -> bool {
    head_dim != 0
        && n_rot != 0
        && n_rot <= head_dim
        && head_dim.is_multiple_of(4)
        && n_rot.is_multiple_of(4)
        && (head_dim - n_rot).is_multiple_of(4)
        && group_dim != 0
        && group_dim.is_multiple_of(head_dim)
}

/// A dimension the launchers take in 32 bits.
fn dim(op: &'static str, value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error { op })
}
