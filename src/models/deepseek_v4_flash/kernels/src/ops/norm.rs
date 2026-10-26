//! Embeddings, norms, rotations, SwiGLU, conversions and the FP8 KV
//! rounding, through the generated launchers of `launch.toml`.

use crate::backend::{Backend, Result};
use crate::generated;
use crate::tensor::{Input, Output};

use super::{F16, F32, Rope, Weight, launched, product, typed, typed_mut};

impl Backend {
    /// The F16 embedding row of `token` into each of the `n_hc` streams.
    pub fn embed_token_hc(
        &self,
        out_hc: &mut impl Output,
        table: Weight,
        n_vocab: u32,
        token: u32,
        n_embd: u32,
        n_hc: u32,
    ) -> Result<()> {
        const OP: &str = "embed token hc";
        let table = typed(
            OP,
            self.weight_bytes(
                table,
                product(OP, &[u64::from(n_vocab), u64::from(n_embd), F16])?,
            )?,
        )?;
        launched(
            OP,
            generated::embed_token_hc(
                self.stream(),
                typed_mut(OP, out_hc.output())?,
                table,
                n_vocab,
                token,
                n_embd,
                n_hc,
            ),
        )
    }

    /// The embedding rows of `n_tokens` device token IDs, which the kernel
    /// clamps into the table.
    pub fn embed_tokens_hc(
        &self,
        out_hc: &mut impl Output,
        tokens: &impl Input,
        table: Weight,
        n_vocab: u32,
        n_tokens: u32,
        n_embd: u32,
        n_hc: u32,
    ) -> Result<()> {
        const OP: &str = "embed tokens hc";
        let table = typed(
            OP,
            self.weight_bytes(
                table,
                product(OP, &[u64::from(n_vocab), u64::from(n_embd), F16])?,
            )?,
        )?;
        launched(
            OP,
            generated::embed_tokens_hc(
                self.stream(),
                typed_mut(OP, out_hc.output())?,
                typed(OP, tokens.input())?,
                table,
                n_vocab,
                n_tokens,
                n_embd,
                n_hc,
            ),
        )
    }

    /// `rows` rows of `n` floats divided by their root mean square.
    pub fn rms_norm_plain_rows(
        &self,
        out: &mut impl Output,
        x: &impl Input,
        n: u32,
        rows: u32,
        eps: f32,
    ) -> Result<()> {
        const OP: &str = "rms norm plain rows";
        launched(
            OP,
            generated::rms_norm_plain_rows(
                self.stream(),
                typed_mut(OP, out.output())?,
                typed(OP, x.input())?,
                n,
                rows,
                eps,
            ),
        )
    }

    /// [`rms_norm_plain_rows`](Self::rms_norm_plain_rows) of `x` in place.
    pub fn rms_norm_plain_rows_in_place(
        &self,
        x: &mut impl Output,
        n: u32,
        rows: u32,
        eps: f32,
    ) -> Result<()> {
        const OP: &str = "rms norm plain rows in place";
        launched(
            OP,
            generated::rms_norm_plain_rows_in_place(
                self.stream(),
                typed_mut(OP, x.output())?,
                n,
                rows,
                eps,
            ),
        )
    }

    /// One row of [`rms_norm_plain_rows`](Self::rms_norm_plain_rows).
    pub fn rms_norm_plain(
        &self,
        out: &mut impl Output,
        x: &impl Input,
        n: u32,
        eps: f32,
    ) -> Result<()> {
        self.rms_norm_plain_rows(out, x, n, 1, eps)
    }

    /// [`rms_norm_plain_rows`](Self::rms_norm_plain_rows) times the F32
    /// weight row `w`.
    pub fn rms_norm_weight_rows(
        &self,
        out: &mut impl Output,
        x: &impl Input,
        w: Weight,
        n: u32,
        rows: u32,
        eps: f32,
    ) -> Result<()> {
        const OP: &str = "rms norm weight rows";
        let w = typed(OP, self.weight_bytes(w, u64::from(n) * F32)?)?;
        launched(
            OP,
            generated::rms_norm_weight_rows(
                self.stream(),
                typed_mut(OP, out.output())?,
                typed(OP, x.input())?,
                w,
                n,
                rows,
                eps,
            ),
        )
    }

    /// [`rms_norm_weight_rows`](Self::rms_norm_weight_rows) of `x` in
    /// place.
    pub fn rms_norm_weight_rows_in_place(
        &self,
        x: &mut impl Output,
        w: Weight,
        n: u32,
        rows: u32,
        eps: f32,
    ) -> Result<()> {
        const OP: &str = "rms norm weight rows in place";
        let w = typed(OP, self.weight_bytes(w, u64::from(n) * F32)?)?;
        launched(
            OP,
            generated::rms_norm_weight_rows_in_place(
                self.stream(),
                typed_mut(OP, x.output())?,
                w,
                n,
                rows,
                eps,
            ),
        )
    }

    /// One row of [`rms_norm_weight_rows`](Self::rms_norm_weight_rows).
    pub fn rms_norm_weight(
        &self,
        out: &mut impl Output,
        x: &impl Input,
        w: Weight,
        n: u32,
        eps: f32,
    ) -> Result<()> {
        self.rms_norm_weight_rows(out, x, w, n, 1, eps)
    }

    /// The query and KV norms of `rows` rows in one launch.
    pub fn qkv_rms_norm_rows(
        &self,
        q_out: &mut impl Output,
        q: &impl Input,
        q_weight: Weight,
        q_n: u32,
        kv_out: &mut impl Output,
        kv: &impl Input,
        kv_weight: Weight,
        kv_n: u32,
        rows: u32,
        eps: f32,
    ) -> Result<()> {
        const OP: &str = "qkv rms norm rows";
        let q_w = typed(OP, self.weight_bytes(q_weight, u64::from(q_n) * F32)?)?;
        let kv_w = typed(OP, self.weight_bytes(kv_weight, u64::from(kv_n) * F32)?)?;
        launched(
            OP,
            generated::qkv_rms_norm_rows(
                self.stream(),
                typed_mut(OP, q_out.output())?,
                typed(OP, q.input())?,
                q_w,
                q_n,
                typed_mut(OP, kv_out.output())?,
                typed(OP, kv.input())?,
                kv_w,
                kv_n,
                rows,
                eps,
            ),
        )
    }

    /// Per-head norm of `x` in place.
    pub fn head_rms_norm(
        &self,
        x: &mut impl Output,
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        eps: f32,
    ) -> Result<()> {
        const OP: &str = "head rms norm";
        launched(
            OP,
            generated::head_rms_norm(
                self.stream(),
                typed_mut(OP, x.output())?,
                n_tok,
                n_head,
                head_dim,
                eps,
            ),
        )
    }

    /// The head norm and the rotation of each head's last `n_rot` values,
    /// in place. `false` below the wide prefill width, as in C++, where the
    /// caller runs both separately.
    pub fn head_rms_norm_rope_tail(
        &self,
        x: &mut impl Output,
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        inverse: bool,
        rope: Rope,
        eps: f32,
    ) -> Result<bool> {
        const OP: &str = "head rms norm rope tail";
        let launch = launched(
            OP,
            generated::head_rms_norm_rope_tail(
                self.stream(),
                typed_mut(OP, x.output())?,
                n_tok,
                n_head,
                head_dim,
                rope.n_rot,
                rope.pos0,
                rope.n_ctx_orig,
                inverse,
                rope.freq_base,
                rope.freq_scale,
                rope.ext_factor,
                rope.attn_factor,
                rope.beta_fast,
                rope.beta_slow,
                eps,
            ),
        )?;
        Ok(launch == gufo_hip::Launch::Launched)
    }

    /// The rotation of each head's last `n_rot` values, in place.
    pub fn rope_tail(
        &self,
        x: &mut impl Output,
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        inverse: bool,
        rope: Rope,
    ) -> Result<()> {
        const OP: &str = "rope tail";
        launched(
            OP,
            generated::rope_tail(
                self.stream(),
                typed_mut(OP, x.output())?,
                n_tok,
                n_head,
                head_dim,
                rope.n_rot,
                rope.pos0,
                1,
                rope.n_ctx_orig,
                inverse,
                rope.freq_base,
                rope.freq_scale,
                rope.ext_factor,
                rope.attn_factor,
                rope.beta_fast,
                rope.beta_slow,
            ),
        )
    }

    /// `out = silu(clamp(gate)) * clamp(up) * weight` over `n` floats.
    pub fn swiglu(
        &self,
        out: &mut impl Output,
        gate: &impl Input,
        up: &impl Input,
        n: u32,
        clamp: f32,
        weight: f32,
    ) -> Result<()> {
        const OP: &str = "swiglu";
        launched(
            OP,
            generated::swiglu(
                self.stream(),
                typed_mut(OP, out.output())?,
                typed(OP, gate.input())?,
                typed(OP, up.input())?,
                n,
                clamp,
                weight,
            ),
        )
    }

    /// The first `count` values of `src` rounded to F16.
    pub fn convert_f32_to_f16(
        &self,
        dst: &mut impl Output,
        src: &impl Input,
        count: u64,
    ) -> Result<()> {
        const OP: &str = "convert f32 to f16";
        let count = usize::try_from(count).map_err(|_| crate::Error { op: OP })?;
        launched(
            OP,
            generated::convert_f32_to_f16(
                self.stream(),
                typed_mut(OP, dst.output())?,
                typed(OP, src.input())?,
                count,
            ),
        )
    }

    /// Rounds the non-rotated part of `n_tok` KV rows through FP8, in
    /// place.
    pub fn fp8_kv_quantize(
        &self,
        x: &mut impl Output,
        n_tok: u32,
        head_dim: u32,
        n_rot: u32,
    ) -> Result<()> {
        const OP: &str = "fp8 kv quantize";
        launched(
            OP,
            generated::fp8_kv_quantize(
                self.stream(),
                typed_mut(OP, x.output())?,
                n_tok,
                head_dim,
                n_rot,
            ),
        )
    }

    /// The rounding and an F16 copy of the rows.
    pub fn fp8_kv_quantize_mirror_f16(
        &self,
        x: &mut impl Output,
        mirror_f16: &mut impl Output,
        n_tok: u32,
        head_dim: u32,
        n_rot: u32,
    ) -> Result<()> {
        const OP: &str = "fp8 kv quantize mirror f16";
        launched(
            OP,
            generated::fp8_kv_quantize_mirror_f16(
                self.stream(),
                typed_mut(OP, x.output())?,
                typed_mut(OP, mirror_f16.output())?,
                n_tok,
                head_dim,
                n_rot,
            ),
        )
    }
}
