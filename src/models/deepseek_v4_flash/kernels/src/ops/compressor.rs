//! The KV compressors, composed from the generated launchers of
//! `launch.toml` as the entry points of `ds4_rocm_compressor.hip.hpp`
//! composed its kernels: each layer of ratio 4 or 128 pools `ratio`
//! positions into one compressed row through a frontier state of
//! `coff * ratio` rows of `coff * head_dim` (`coff` is 2 for ratio 4, whose
//! windows overlap their predecessor, else 1).

use gufo_hip::{DeviceSlice, DeviceSliceMut};

use crate::backend::{Backend, Error, Result};
use crate::generated;
use crate::tensor::{Input, Output};

use super::{F16, F32, Rope, Weight, ensure, launched, product, typed, typed_mut};

/// The weights of one compressor: the positional bias (`ape`, F16 rows of
/// the state width, as the loader requires) and the F32 norm of the
/// compressed row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressorWeights {
    pub ape: Weight,
    pub norm: Weight,
}

/// `coff` after the checks every compressor op makes before its first
/// launch: a ratio the kernels' candidate arrays hold, a state width in 32
/// bits and an even rotated tail inside the head.
fn shape(op: &'static str, head_dim: u32, ratio: u32, rope: &Rope) -> Result<u32> {
    ensure(op, head_dim != 0 && (1..=128).contains(&ratio))?;
    let coff = if ratio == 4 { 2 } else { 1 };
    ensure(op, head_dim.checked_mul(coff).is_some())?;
    ensure(op, rope.n_rot <= head_dim && rope.n_rot.is_multiple_of(2))?;
    Ok(coff)
}

/// `bytes` at `offset` of `slice`, or an error outside it.
fn range_mut<'a>(
    op: &'static str,
    slice: DeviceSliceMut<'a, u8>,
    offset: u64,
    bytes: u64,
) -> Result<DeviceSliceMut<'a, u8>> {
    let end = offset
        .checked_add(bytes)
        .filter(|&end| end <= slice.len() as u64)
        .ok_or(Error { op })?;
    Ok(slice.range(offset as usize..end as usize))
}

impl Backend {
    /// The positional bias: `ratio` rows of `width` F16 values.
    fn ape(
        &self,
        op: &'static str,
        ape: Weight,
        ratio: u32,
        width: u32,
    ) -> Result<DeviceSlice<'_, gufo_hip::F16>> {
        let bytes = product(op, &[u64::from(ratio), u64::from(width), F16])?;
        typed(op, self.weight_bytes(ape, bytes)?)
    }

    /// An empty state of `count` values per tensor.
    fn reset_state(
        &self,
        op: &'static str,
        state_kv: &mut impl Output,
        state_score: &mut impl Output,
        count: u64,
    ) -> Result<()> {
        launched(
            op,
            generated::compressor_state_reset(
                self.stream(),
                typed_mut(op, state_kv.output())?,
                typed_mut(op, state_score.output())?,
                usize::try_from(count).map_err(|_| Error { op })?,
            ),
        )
    }

    /// Positions `src0 .. src0 + rows` of a chunk from `pos0` into state
    /// rows from `dst0`.
    fn set_state_rows(
        &self,
        op: &'static str,
        state_kv: &mut impl Output,
        state_score: &mut impl Output,
        kv: &impl Input,
        sc: &impl Input,
        ape: DeviceSlice<'_, gufo_hip::F16>,
        width: u32,
        ratio: u32,
        pos0: u32,
        src0: u32,
        dst0: u32,
        rows: u32,
    ) -> Result<()> {
        launched(
            op,
            generated::compressor_set_rows(
                self.stream(),
                typed_mut(op, state_kv.output())?,
                typed_mut(op, state_score.output())?,
                typed(op, kv.input())?,
                typed(op, sc.input())?,
                ape,
                width,
                ratio,
                pos0,
                src0,
                dst0,
                rows,
            ),
        )
    }

    /// The norm, the rotation (row `c` at position `pos0 + c * pos_stride`)
    /// and, with `quantize_fp8`, the FP8 rounding (into the F16 `mirror`
    /// too) of `n_comp` fresh compressed rows.
    fn finish_rows(
        &self,
        op: &'static str,
        mut comp: DeviceSliceMut<'_, f32>,
        mirror: Option<DeviceSliceMut<'_, gufo_hip::F16>>,
        norm: Weight,
        head_dim: u32,
        n_comp: u32,
        pos0: u32,
        pos_stride: u32,
        rope: &Rope,
        quantize_fp8: bool,
        rms_eps: f32,
    ) -> Result<()> {
        let norm = typed(
            op,
            self.weight_bytes(norm, product(op, &[u64::from(head_dim), F32])?)?,
        )?;
        launched(
            op,
            generated::rms_norm_weight_rows_in_place(
                self.stream(),
                comp.reborrow(),
                norm,
                head_dim,
                n_comp,
                rms_eps,
            ),
        )?;
        if rope.n_rot != 0 {
            launched(
                op,
                generated::rope_tail(
                    self.stream(),
                    comp.reborrow(),
                    n_comp,
                    1,
                    head_dim,
                    rope.n_rot,
                    pos0,
                    pos_stride,
                    rope.n_ctx_orig,
                    false,
                    rope.freq_base,
                    rope.freq_scale,
                    rope.ext_factor,
                    rope.attn_factor,
                    rope.beta_fast,
                    rope.beta_slow,
                ),
            )?;
        }
        match (quantize_fp8, mirror) {
            (false, None) => Ok(()),
            (false, Some(_)) => Err(Error { op }),
            (true, None) => launched(
                op,
                generated::fp8_kv_quantize(self.stream(), comp, n_comp, head_dim, rope.n_rot),
            ),
            (true, Some(mirror)) => launched(
                op,
                generated::fp8_kv_quantize_mirror_f16(
                    self.stream(),
                    comp,
                    mirror,
                    n_comp,
                    head_dim,
                    rope.n_rot,
                ),
            ),
        }
    }

    /// One position's KV and score into the state (unless
    /// `state_already_stored`), and at the end of each ratio window the
    /// compressed row `comp_row`.
    pub fn compressor_update(
        &self,
        kv_cur: &impl Input,
        sc_cur: &impl Input,
        state_kv: &mut (impl Output + Input),
        state_score: &mut (impl Output + Input),
        comp_cache: &mut impl Output,
        w: CompressorWeights,
        head_dim: u32,
        ratio: u32,
        pos: u32,
        comp_row: u32,
        rope: Rope,
        rms_eps: f32,
        state_already_stored: bool,
    ) -> Result<()> {
        const OP: &str = "compressor update";
        let coff = shape(OP, head_dim, ratio, &rope)?;
        if !state_already_stored {
            launched(
                OP,
                generated::compressor_store(
                    self.stream(),
                    typed(OP, kv_cur.input())?,
                    typed(OP, sc_cur.input())?,
                    typed_mut(OP, state_kv.output())?,
                    typed_mut(OP, state_score.output())?,
                    self.ape(OP, w.ape, ratio, coff * head_dim)?,
                    coff,
                    head_dim,
                    ratio,
                    pos,
                    1,
                ),
            )?;
        }
        // The window ends here; positions count in 32 bits, as in C++.
        if !pos.wrapping_add(1).is_multiple_of(ratio) {
            return Ok(());
        }
        let row_bytes = u64::from(head_dim) * F32;
        let offset = product(OP, &[u64::from(comp_row), row_bytes])?;
        let mut row = typed_mut(OP, range_mut(OP, comp_cache.output(), offset, row_bytes)?)?;
        launched(
            OP,
            generated::compressor_update_pool(
                self.stream(),
                row.reborrow(),
                typed(OP, state_kv.input())?,
                typed(OP, state_score.input())?,
                coff,
                head_dim,
                ratio,
            ),
        )?;
        self.finish_rows(
            OP,
            row,
            None,
            w.norm,
            head_dim,
            1,
            pos.wrapping_add(1).wrapping_sub(ratio),
            1,
            &rope,
            false,
            rms_eps,
        )?;
        if ratio == 4 {
            launched(
                OP,
                generated::compressor_shift_ratio4(
                    self.stream(),
                    typed_mut(OP, state_kv.output())?,
                    typed_mut(OP, state_score.output())?,
                    coff * head_dim,
                ),
            )?;
        }
        Ok(())
    }

    /// `n_tokens` from `pos0` into `n_tokens / ratio` compressed rows (and
    /// their F16 mirror) and the state of the remainder.
    pub fn compressor_prefill(
        &self,
        comp_cache: &mut impl Output,
        state_kv: &mut (impl Output + Input),
        state_score: &mut (impl Output + Input),
        kv: &impl Input,
        sc: &impl Input,
        w: CompressorWeights,
        head_dim: u32,
        ratio: u32,
        pos0: u32,
        n_tokens: u32,
        rope: Rope,
        quantize_fp8: bool,
        rms_eps: f32,
        comp_mirror_f16: Option<&mut dyn Output>,
    ) -> Result<()> {
        const OP: &str = "compressor prefill";
        let coff = shape(OP, head_dim, ratio, &rope)?;
        let n_comp = n_tokens / ratio;
        ensure(
            OP,
            n_tokens != 0 && (quantize_fp8 || n_comp == 0 || comp_mirror_f16.is_none()),
        )?;
        let width = coff * head_dim;
        let ape = self.ape(OP, w.ape, ratio, width)?;
        let cutoff = n_comp * ratio;
        let rem = n_tokens - cutoff;
        self.reset_state(
            OP,
            state_kv,
            state_score,
            u64::from(coff * ratio) * u64::from(width),
        )?;
        // Ratio 4 keeps the last whole window as the previous one, which the
        // next window overlaps.
        if ratio == 4 && cutoff >= ratio {
            self.set_state_rows(
                OP,
                state_kv,
                state_score,
                kv,
                sc,
                ape,
                width,
                ratio,
                pos0,
                cutoff - ratio,
                0,
                ratio,
            )?;
        }
        if rem != 0 {
            let dst0 = if ratio == 4 { ratio } else { 0 };
            self.set_state_rows(
                OP,
                state_kv,
                state_score,
                kv,
                sc,
                ape,
                width,
                ratio,
                pos0,
                cutoff,
                dst0,
                rem,
            )?;
        }
        if n_comp == 0 {
            return Ok(());
        }
        launched(
            OP,
            generated::compressor_prefill_pool(
                self.stream(),
                typed_mut(OP, comp_cache.output())?,
                typed(OP, kv.input())?,
                typed(OP, sc.input())?,
                typed(OP, state_kv.input())?,
                typed(OP, state_score.input())?,
                ape,
                coff,
                head_dim,
                ratio,
                pos0,
                n_comp,
                false,
            ),
        )?;
        let mirror = comp_mirror_f16
            .map(|mirror| typed_mut(OP, mirror.output()))
            .transpose()?;
        self.finish_rows(
            OP,
            typed_mut(OP, comp_cache.output())?,
            mirror,
            w.norm,
            head_dim,
            n_comp,
            pos0,
            ratio,
            &rope,
            quantize_fp8,
            rms_eps,
        )
    }

    /// The ratio-4 prefill of whole windows from an aligned `pos0`, whose
    /// first row overlaps the window in the state; the state then holds the
    /// chunk's last window.
    pub fn compressor_prefill_ratio4_replay(
        &self,
        comp_cache: &mut impl Output,
        state_kv: &mut (impl Output + Input),
        state_score: &mut (impl Output + Input),
        kv: &impl Input,
        sc: &impl Input,
        w: CompressorWeights,
        head_dim: u32,
        pos0: u32,
        n_tokens: u32,
        rope: Rope,
        quantize_fp8: bool,
        rms_eps: f32,
        comp_mirror_f16: Option<&mut dyn Output>,
    ) -> Result<()> {
        const OP: &str = "compressor prefill ratio4 replay";
        let coff = shape(OP, head_dim, 4, &rope)?;
        ensure(
            OP,
            n_tokens != 0 && n_tokens.is_multiple_of(4) && pos0.is_multiple_of(4),
        )?;
        ensure(OP, quantize_fp8 || comp_mirror_f16.is_none())?;
        let width = coff * head_dim;
        let ape = self.ape(OP, w.ape, 4, width)?;
        let n_comp = n_tokens / 4;
        launched(
            OP,
            generated::compressor_prefill_pool(
                self.stream(),
                typed_mut(OP, comp_cache.output())?,
                typed(OP, kv.input())?,
                typed(OP, sc.input())?,
                typed(OP, state_kv.input())?,
                typed(OP, state_score.input())?,
                ape,
                coff,
                head_dim,
                4,
                pos0,
                n_comp,
                true,
            ),
        )?;
        let mirror = comp_mirror_f16
            .map(|mirror| typed_mut(OP, mirror.output()))
            .transpose()?;
        self.finish_rows(
            OP,
            typed_mut(OP, comp_cache.output())?,
            mirror,
            w.norm,
            head_dim,
            n_comp,
            pos0,
            4,
            &rope,
            quantize_fp8,
            rms_eps,
        )?;
        self.reset_state(OP, state_kv, state_score, 8 * u64::from(width))?;
        self.set_state_rows(
            OP,
            state_kv,
            state_score,
            kv,
            sc,
            ape,
            width,
            4,
            pos0,
            n_tokens - 4,
            0,
            4,
        )
    }

    /// The ratio-4 state from the last four positions of a prompt, the first
    /// at `pos0`.
    pub fn compressor_prefill_state_ratio4(
        &self,
        state_kv: &mut impl Output,
        state_score: &mut impl Output,
        kv_tail: &impl Input,
        sc_tail: &impl Input,
        ape: Weight,
        head_dim: u32,
        pos0: u32,
    ) -> Result<()> {
        const OP: &str = "compressor prefill state ratio4";
        ensure(OP, head_dim != 0)?;
        let width = head_dim.checked_mul(2).ok_or(Error { op: OP })?;
        let ape = self.ape(OP, ape, 4, width)?;
        self.reset_state(OP, state_kv, state_score, 8 * u64::from(width))?;
        self.set_state_rows(
            OP,
            state_kv,
            state_score,
            kv_tail,
            sc_tail,
            ape,
            width,
            4,
            pos0,
            0,
            0,
            4,
        )
    }
}
