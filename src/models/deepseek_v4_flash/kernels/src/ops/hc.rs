//! Hyper-connection mixing, the output head's stream weights and the
//! verification argmax, through the generated launchers of `launch.toml`
//! (formerly `ds4_rocm_hc_output_launch.hip.hpp`).
//!
//! Several launchers took their row count from the output tensor's size,
//! which must be a whole number of rows; the wrappers derive it the same way
//! and pass it on.

use crate::backend::{Backend, Result};
use crate::generated;
use crate::tensor::{Input, Output, Tensor};

use super::{F32, Weight, ensure, launched, product, typed, typed_mut};

/// Rows of `row_floats` floats `out` holds exactly.
fn rows_of(op: &'static str, out: &Tensor, row_floats: u64) -> Result<u32> {
    let row = product(op, &[row_floats, F32])?;
    ensure(
        op,
        row != 0 && out.bytes() >= row && out.bytes().is_multiple_of(row),
    )?;
    let rows = out.bytes() / row;
    ensure(op, rows <= u64::from(u32::MAX))?;
    Ok(rows as u32)
}

fn same_model(op: &'static str, weights: &[Weight]) -> Result<()> {
    ensure(
        op,
        weights.windows(2).all(|w| w[0].model == w[1].model)
            && weights.iter().all(|w| w.offset % 4 == 0),
    )
}

/// The mix width `2 n_hc + n_hc^2`.
fn mix_width(n_hc: u32) -> u64 {
    let h = u64::from(n_hc);
    h * h + 2 * h
}

impl Backend {
    /// `F32` weights of `count` values at `w`.
    fn f32_weight(
        &self,
        op: &'static str,
        w: Weight,
        count: u64,
    ) -> Result<gufo_hip::DeviceSlice<'_, f32>> {
        typed(op, self.weight_bytes(w, product(op, &[count, F32])?)?)
    }

    /// `out[t] = sum_s weights[t][s] residual[t][s]`, one row per `n_embd`
    /// floats of `out`.
    pub fn hc_weighted_sum(
        &self,
        out: &mut impl Output,
        residual_hc: &impl Input,
        weights: &impl Input,
        n_embd: u32,
        n_hc: u32,
    ) -> Result<()> {
        const OP: &str = "hc weighted sum";
        let n_tokens = rows_of(OP, out, u64::from(n_embd))?;
        launched(
            OP,
            generated::hc_weighted_sum(
                self.stream(),
                typed_mut(OP, out.output())?,
                typed(OP, residual_hc.input())?,
                typed(OP, weights.input())?,
                n_embd,
                n_hc,
                n_tokens,
            ),
        )
    }

    /// The Sinkhorn split of `mix` and the weighted sum of the four
    /// streams.
    pub fn hc_split_weighted_sum(
        &self,
        out: &mut impl Output,
        split: &mut impl Output,
        mix: &impl Input,
        residual_hc: &impl Input,
        scale: Weight,
        base: Weight,
        n_embd: u32,
        n_hc: u32,
        sinkhorn_iters: u32,
        eps: f32,
    ) -> Result<()> {
        const OP: &str = "hc split weighted sum";
        same_model(OP, &[scale, base])?;
        let n_rows = rows_of(OP, out, u64::from(n_embd))?;
        let scale = self.f32_weight(OP, scale, 3)?;
        let base = self.f32_weight(OP, base, mix_width(n_hc))?;
        launched(
            OP,
            generated::hc_split_weighted_sum(
                self.stream(),
                typed_mut(OP, out.output())?,
                typed_mut(OP, split.output())?,
                typed(OP, mix.input())?,
                typed(OP, residual_hc.input())?,
                scale,
                base,
                n_embd,
                n_hc,
                n_rows,
                sinkhorn_iters,
                eps,
            ),
        )
    }

    /// The whole pre-block chain for four streams of 4,096 values. `false`
    /// when the launcher declines the shape or alignment, as in C++, where
    /// the caller runs the separate chain.
    pub fn hc_norm_mix_split_weighted_sum(
        &self,
        out: &mut impl Output,
        mix: &mut impl Output,
        split: &mut impl Output,
        residual_hc: &impl Input,
        mix_weight: Weight,
        scale: Weight,
        base: Weight,
        n_embd: u32,
        n_hc: u32,
        n_rows: u32,
        sinkhorn_iters: u32,
        eps: f32,
        norm_eps: f32,
    ) -> Result<bool> {
        const OP: &str = "hc norm mix split weighted sum";
        same_model(OP, &[mix_weight, scale, base])?;
        if n_hc != 4 || n_rows == 0 || n_embd != 4096 {
            return Ok(false);
        }
        let mix_w = typed::<gufo_hip::F16>(
            OP,
            self.weight_bytes(mix_weight, product(OP, &[24, 4, u64::from(n_embd), 2])?)?,
        )?;
        let scale = self.f32_weight(OP, scale, 3)?;
        let base = self.f32_weight(OP, base, 24)?;
        let launch = launched(
            OP,
            generated::hc_norm_mix_split_weighted_sum(
                self.stream(),
                typed_mut(OP, out.output())?,
                typed_mut(OP, mix.output())?,
                typed_mut(OP, split.output())?,
                typed(OP, residual_hc.input())?,
                mix_w,
                scale,
                base,
                n_embd,
                n_rows,
                sinkhorn_iters,
                eps,
                norm_eps,
            ),
        )?;
        Ok(launch == gufo_hip::Launch::Launched)
    }

    /// The split and weighted sum, and `norm_out = rmsnorm(out) *
    /// norm_weight`.
    pub fn hc_split_weighted_sum_norm(
        &self,
        out: &mut impl Output,
        norm_out: &mut impl Output,
        split: &mut impl Output,
        mix: &impl Input,
        residual_hc: &impl Input,
        scale: Weight,
        base: Weight,
        norm_weight: Weight,
        n_embd: u32,
        n_hc: u32,
        sinkhorn_iters: u32,
        eps: f32,
        norm_eps: f32,
    ) -> Result<()> {
        const OP: &str = "hc split weighted sum norm";
        same_model(OP, &[scale, base, norm_weight])?;
        let n_rows = rows_of(OP, out, u64::from(n_embd))?;
        let scale = self.f32_weight(OP, scale, 3)?;
        let base = self.f32_weight(OP, base, mix_width(n_hc))?;
        let norm_w = self.f32_weight(OP, norm_weight, u64::from(n_embd))?;
        launched(
            OP,
            generated::hc_split_weighted_sum_norm(
                self.stream(),
                typed_mut(OP, out.output())?,
                typed_mut(OP, norm_out.output())?,
                typed_mut(OP, split.output())?,
                typed(OP, mix.input())?,
                typed(OP, residual_hc.input())?,
                scale,
                base,
                norm_w,
                n_embd,
                n_hc,
                n_rows,
                sinkhorn_iters,
                eps,
                norm_eps,
            ),
        )
    }

    /// The output head's stream weights, one row of `n_hc` per token of
    /// `out`.
    pub fn output_hc_weights(
        &self,
        out: &mut impl Output,
        pre: &impl Input,
        scale: Weight,
        base: Weight,
        n_hc: u32,
        eps: f32,
    ) -> Result<()> {
        const OP: &str = "output hc weights";
        same_model(OP, &[scale, base])?;
        let n_tokens = rows_of(OP, out, u64::from(n_hc))?;
        // The kernel indexes its elements in 32 bits.
        ensure(
            OP,
            u64::from(n_tokens) * u64::from(n_hc) <= u64::from(u32::MAX),
        )?;
        let scale = self.f32_weight(OP, scale, 1)?;
        let base = self.f32_weight(OP, base, u64::from(n_hc))?;
        launched(
            OP,
            generated::output_hc_weights(
                self.stream(),
                typed_mut(OP, out.output())?,
                typed(OP, pre.input())?,
                scale,
                base,
                n_hc,
                n_tokens,
                eps,
            ),
        )
    }

    /// `out_hc = post * block_out + comb * residual`, one row per stream row
    /// of `out_hc`.
    pub fn hc_expand_split(
        &self,
        out_hc: &mut impl Output,
        block_out: &impl Input,
        residual_hc: &impl Input,
        split: &impl Input,
        n_embd: u32,
        n_hc: u32,
    ) -> Result<()> {
        const OP: &str = "hc expand split";
        let n_tokens = rows_of(
            OP,
            out_hc,
            product(OP, &[u64::from(n_hc), u64::from(n_embd)])?,
        )?;
        launched(
            OP,
            generated::hc_expand_split(
                self.stream(),
                typed_mut(OP, out_hc.output())?,
                typed(OP, block_out.input())?,
                typed(OP, residual_hc.input())?,
                typed(OP, split.input())?,
                n_embd,
                n_hc,
                n_tokens,
            ),
        )
    }

    /// The expansion of `block_out + block_add`.
    pub fn hc_expand_add_split(
        &self,
        out_hc: &mut impl Output,
        block_out: &impl Input,
        block_add: &impl Input,
        residual_hc: &impl Input,
        split: &impl Input,
        n_embd: u32,
        n_hc: u32,
    ) -> Result<()> {
        const OP: &str = "hc expand add split";
        let n_tokens = rows_of(
            OP,
            out_hc,
            product(OP, &[u64::from(n_hc), u64::from(n_embd)])?,
        )?;
        launched(
            OP,
            generated::hc_expand_add_split(
                self.stream(),
                typed_mut(OP, out_hc.output())?,
                typed(OP, block_out.input())?,
                typed(OP, block_add.input())?,
                typed(OP, residual_hc.input())?,
                typed(OP, split.input())?,
                n_embd,
                n_hc,
                n_tokens,
            ),
        )
    }

    /// The expansion of the sum of `n_expert` F16 expert rows per token plus
    /// `block_add`.
    pub fn hc_expand_add_split_moesum(
        &self,
        out_hc: &mut impl Output,
        down_h: &impl Input,
        block_add: &impl Input,
        residual_hc: &impl Input,
        split: &impl Input,
        n_embd: u32,
        n_hc: u32,
        n_expert: u32,
        n_tokens: u32,
    ) -> Result<()> {
        const OP: &str = "hc expand add split moesum";
        ensure(
            OP,
            n_hc == 4 && n_embd != 0 && n_expert != 0 && n_tokens != 0,
        )?;
        let launch = launched(
            OP,
            generated::hc_expand_add_split_moesum(
                self.stream(),
                typed_mut(OP, out_hc.output())?,
                typed(OP, down_h.input())?,
                typed(OP, block_add.input())?,
                typed(OP, residual_hc.input())?,
                typed(OP, split.input())?,
                n_embd,
                n_hc,
                n_expert,
                n_tokens,
            ),
        )?;
        ensure(OP, launch == gufo_hip::Launch::Launched)
    }

    /// The first maximum of each of `n_rows` rows of `vocab` logits.
    pub fn spec_row_argmax(
        &self,
        out_index: &mut impl Output,
        logits: &impl Input,
        vocab: u32,
        n_rows: u32,
    ) -> Result<()> {
        const OP: &str = "spec row argmax";
        launched(
            OP,
            generated::spec_row_argmax(
                self.stream(),
                typed_mut(OP, out_index.output())?,
                typed(OP, logits.input())?,
                vocab,
                n_rows,
            ),
        )
    }

    /// The mean of each row's streams into `out[t * out_row_stride +
    /// slot_offset ..][..n_embd]`.
    pub fn dspark_capture_features(
        &self,
        out: &mut impl Output,
        hc: &impl Input,
        out_row_stride: u32,
        slot_offset: u32,
        n_embd: u32,
        n_hc: u32,
        n_rows: u32,
    ) -> Result<()> {
        const OP: &str = "dspark capture features";
        launched(
            OP,
            generated::dspark_capture_features(
                self.stream(),
                typed_mut(OP, out.output())?,
                typed(OP, hc.input())?,
                out_row_stride,
                slot_offset,
                n_embd,
                n_hc,
                n_rows,
            ),
        )
    }
}
