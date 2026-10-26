//! The projection routes of `executor.cpp`: activation quantization for
//! the decode GEMVs, the dense Q8_0 routes (Q8_1 vector kernels, the F16
//! WMMA GEMM, W8A8 over tiled Q8 rows), the 16-bit projections (small
//! kernels, the unquantized F16 GEMM, hipBLASLt) and the routed experts.
//! Which route runs decides the arithmetic, so every choice follows the C++
//! executor exactly.

use gufo_formats::GgmlType;
use gufo_hip::blas::{AlgoPolicy, BlasLt, Heuristic, Layout, MatmulShape, Op};
use gufo_hip::{Bf16, DeviceSlice, DeviceSliceMut, F16, IndexSlice, Launch, LaunchError, Stream};
use gufo_qwen38_flash_next_kernels as k;

use crate::config::Config;
use crate::device::{DeviceModel, Matrix};
use crate::state::{Stage, VEC_BATCH};

/// Wide dense Q8_0 projections take the F16 WMMA GEMM when its 256-row
/// tiles fill the device and the K sweep keeps the F16 rows cache-resident
/// (measured per shape in the C++ executor).
const DENSE_F16_MIN_ROWS: u32 = 2048;
const DENSE_F16_MAX_COLS: u32 = 2560;
/// Token rows per routed F16 expert tile: narrow for buckets of one or two
/// 16-row tiles, wide when the mean bucket fills most of one.
pub(crate) const ROUTED_TILE_ROWS_NARROW: u32 = 16;
pub(crate) const ROUTED_TILE_ROWS_WIDE: u32 = 48;

/// A forward's failure: a launch error kept intact (graph validation
/// recaptures on a mismatch), or a message.
#[derive(Debug)]
pub(crate) enum Error {
    Launch(LaunchError, &'static str),
    Other(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Launch(e, what) => write!(f, "{what}: {e}"),
            Self::Other(message) => f.write_str(message),
        }
    }
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::Other(message)
    }
}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self::Other(message.to_string())
    }
}

impl From<gufo_hip::HipError> for Error {
    fn from(e: gufo_hip::HipError) -> Self {
        Self::Launch(LaunchError::Hip(e), "HIP")
    }
}

impl From<LaunchError> for Error {
    fn from(e: LaunchError) -> Self {
        Self::Launch(e, "launch")
    }
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

pub(crate) fn launch_error(what: &'static str) -> impl Fn(LaunchError) -> Error {
    move |e| Error::Launch(e, what)
}

/// A launch whose `supported` result must be true.
pub(crate) fn launched(
    result: std::result::Result<Launch, LaunchError>,
    what: &'static str,
) -> Result<()> {
    match result {
        Ok(Launch::Launched) => Ok(()),
        Ok(Launch::Unsupported) => Err(Error::Other(what.to_string())),
        Err(e) => Err(Error::Launch(e, what)),
    }
}

/// The C++ hipBLASLt plan choice (`BlasLt::MakePlan`): 16 heuristic results
/// without workspace, the first valid, workspace-free one that
/// `matmulIsAlgoSupported` accepts, no solution-name lookup.
struct FlashNextPlans;

impl AlgoPolicy for FlashNextPlans {
    fn choose(&self, ranked: &[Heuristic], workspace_capacity: usize) -> Option<usize> {
        ranked
            .iter()
            .position(|h| h.is_usable(workspace_capacity) && h.workspace_bytes == 0)
    }

    fn verify_supported(&self) -> bool {
        true
    }

    fn resolve_names(&self) -> bool {
        false
    }
}

/// Routed tile hints derived from the expert counts of a tiled batch
/// (`RouteHints`), and the column-grid hints of the tiled tier.
#[derive(Default, Clone, Copy)]
pub(crate) struct Hints {
    /// 0: no readback yet.
    pub max_rows: u32,
    /// `tile_rows` tiles (narrow batches).
    pub n_tiles: u32,
    /// Wide batches' 64-token down tiles, first in the map.
    pub tiles_64: u32,
    /// Wide batches' 128-token gate/up tiles after them.
    pub pair_offset: u32,
    pub pair_tiles: u32,
    pub tile_rows: u32,
    pub tile_cols: u32,
}

/// State of the routes within one forward (the C++ executor's mutable
/// members).
#[derive(Default)]
pub(crate) struct Route {
    /// Prompt arithmetic at every chunk width (`prefill_phase`).
    pub prefill: bool,
    /// The last combine wrote the next mixer's norm as F16 and tiled Q8.
    pub xn_half: bool,
    /// The last combine also wrote its norm as Q8_1 rows into this `x_q8`
    /// slot, for the next mixer's decode GEMV.
    pub xn_q8: Option<usize>,
    /// The last mixer epilogue also wrote `mixed` as Q8_1 rows into this
    /// `x_q8` slot; every other write of `mixed` clears it.
    pub mixed_q8: Option<usize>,
    /// Partial sums per inject logit the last mixer left.
    pub inject_parts: u32,
    /// The MoE epilogue is left to the combine that follows.
    pub moe_pending: bool,
    /// Set before an attention-side combine whose FFN combine fuses the MoE
    /// epilogue: that combine may then leave its residual update to the
    /// next one (`res_deferred`), which rereads `block_out` and
    /// `hc_defer_w`.
    pub defer_residual: bool,
    pub res_deferred: bool,
    /// The address of the inject weights whose partials the last combine
    /// already wrote to `inject`: the following mixer skips its inject pass.
    pub inject_fused: Option<usize>,
    /// `shexp_half` holds the shared expert's F16 SwiGLU rows.
    pub shexp_half_ready: bool,
    pub hints: Hints,
}

impl Route {
    pub fn matrix_rows(&self, rows: u32) -> bool {
        self.prefill || rows > VEC_BATCH
    }

    pub fn expert_matrix_rows(&self, rows: u32) -> bool {
        self.prefill || rows > 4 * VEC_BATCH
    }

    /// Prefill passes wider than a vector batch take the fused projections
    /// and the paired routed expert tiles (upstream `WidePrefill`):
    /// they win from 9 rows up to the full chunk. Decode and verification
    /// batches keep their single-token arithmetic.
    pub fn wide_prefill(&self, rows: u32) -> bool {
        self.prefill && rows > VEC_BATCH
    }
}

/// An activation batch quantized once for the decode GEMVs: `slot` names the
/// Q8_1 staging buffer, `None` when the batch takes the tiled routes.
pub(crate) struct Q8Input<'a> {
    pub x: DeviceSlice<'a, f32>,
    pub slot: Option<usize>,
    pub n: u32,
    pub k: u32,
}

/// Everything the routes need besides the scratch they read and write.
pub(crate) struct Ctx<'e> {
    pub stream: &'e Stream,
    pub model: &'e DeviceModel,
    pub blaslt: &'e mut BlasLt,
    pub stage: &'e mut Stage,
    pub route: &'e mut Route,
    pub max_batch: u32,
    /// The YaRN constants of every rope launch.
    pub yarn: crate::rope_scaling::YarnRope,
}

fn cols(w: &Matrix) -> usize {
    w.cols as usize
}

fn rows(w: &Matrix) -> usize {
    w.rows as usize
}

impl Ctx<'_> {
    pub fn config(&self) -> &Config {
        &self.model.config
    }

    pub fn matrix_rows(&self, rows: u32) -> bool {
        self.route.matrix_rows(rows)
    }

    pub fn wide_prefill(&self, rows: u32) -> bool {
        self.route.wide_prefill(rows)
    }

    /// Quantizes `x` for the decode GEMVs; the tiled routes quantize per
    /// call. Two slots alternate, so an input stays valid across one other
    /// quantization.
    pub fn quantize<'a>(&mut self, x: DeviceSlice<'a, f32>, n: u32, k: u32) -> Result<Q8Input<'a>> {
        if self.matrix_rows(n) {
            return Ok(Q8Input {
                x,
                slot: None,
                n,
                k,
            });
        }
        if let Some(q) = self.held_q8(self.route.mixed_q8, x, n, k) {
            return Ok(q);
        }
        let stream = self.stream;
        let (slot, rows_q8) = self.claim_q8(n, k);
        launched(
            k::mmq_quantize_q8_1(stream, x.range(0..n as usize * k as usize), rows_q8, n, k),
            "activation quantization failed",
        )?;
        Ok(self.record_q8(slot, x, n, k))
    }

    /// Whether `w` over `n` rows takes the Q8_0 decode GEMV, which reads the
    /// Q8_1 rows of its input.
    pub fn q8_vector(&self, w: &Matrix, n: u32) -> bool {
        w.format() == GgmlType::Q8_0 && !self.matrix_rows(n)
    }

    /// The next `x_q8` slot, sized for `n` rows of `k` values, for a kernel
    /// that writes Q8_1 rows; it holds nothing until `record_q8`.
    pub fn claim_q8(&mut self, n: u32, k: u32) -> (usize, DeviceSliceMut<'_, u8>) {
        let slot = self.stage.q8_slot;
        self.stage.q8_slot ^= 1;
        self.stage.q8_src[slot] = None;
        let bytes = k::q8_1_bytes(n as usize, k as usize);
        (slot, self.stage.x_q8[slot].slice_mut(0..bytes))
    }

    /// Records that `slot` holds the Q8_1 rows of `n` rows of `k` values of
    /// `x`.
    pub fn record_q8<'a>(
        &mut self,
        slot: usize,
        x: DeviceSlice<'a, f32>,
        n: u32,
        k: u32,
    ) -> Q8Input<'a> {
        self.stage.q8_src[slot] = Some((x.address(), n, k));
        Q8Input {
            x,
            slot: Some(slot),
            n,
            k,
        }
    }

    /// The Q8_1 rows of `x` in `slot`, if it still holds them.
    pub fn held_q8<'a>(
        &self,
        slot: Option<usize>,
        x: DeviceSlice<'a, f32>,
        n: u32,
        k: u32,
    ) -> Option<Q8Input<'a>> {
        slot.filter(|&slot| self.stage.q8_src[slot] == Some((x.address(), n, k)))
            .map(|slot| Q8Input {
                x,
                slot: Some(slot),
                n,
                k,
            })
    }

    /// `out = w . x` over a quantized input (`Dense(w, Q8Input)`).
    pub fn dense_q(
        &mut self,
        w: &Matrix,
        q: &Q8Input<'_>,
        out: DeviceSliceMut<'_, f32>,
    ) -> Result<()> {
        if w.format() == GgmlType::Q8_0
            && let Some(slot) = q.slot
        {
            if w.cols != q.k {
                return Err("quantized input width mismatch".into());
            }
            let bytes = k::q8_1_bytes(q.n as usize, q.k as usize);
            let out = out.range(0..q.n as usize * rows(w));
            return launched(
                k::mmq_q8_dense_vec(
                    self.stream,
                    w.view(),
                    self.stage.x_q8[slot].slice(0..bytes),
                    out,
                    w.rows,
                    q.n,
                    w.cols,
                ),
                "Q8_0 GEMV failed",
            );
        }
        self.dense(w, q.x, out, q.n)
    }

    /// Whether a wide dense Q8_0 projection takes the F16 WMMA GEMM.
    pub fn dense_f16_route(&self, w: &Matrix, n: u32) -> bool {
        w.format() == GgmlType::Q8_0
            && self.matrix_rows(n)
            && w.rows >= DENSE_F16_MIN_ROWS
            && w.cols <= DENSE_F16_MAX_COLS
            && w.cols as usize <= self.model.max_half_cols
    }

    /// F16 rows of `x` in `x_half`, unless it holds them already.
    pub fn prepare_half(&mut self, x: DeviceSlice<'_, f32>, rows: u32, cols: usize) -> Result<()> {
        let key = (x.address(), rows, cols, false);
        if self.stage.half_src == Some(key) {
            return Ok(());
        }
        let count = rows as usize * cols;
        k::narrow_f16(
            self.stream,
            x.range(0..count),
            self.stage.x_half.slice_mut(0..count),
            count,
        )
        .map_err(launch_error("activation narrowing"))?;
        self.stage.half_src = Some(key);
        Ok(())
    }

    /// `out[t] = w . x[t]` for `n` rows (`Dense(w, x, out, n)`).
    pub fn dense(
        &mut self,
        w: &Matrix,
        x: DeviceSlice<'_, f32>,
        mut out: DeviceSliceMut<'_, f32>,
        n: u32,
    ) -> Result<()> {
        let (m, kk) = (rows(w), cols(w));
        if w.format() == GgmlType::Q8_0 {
            if !self.matrix_rows(n) {
                let q = self.quantize(x, n, w.cols)?;
                return self.dense_q(w, &q, out);
            }
            // Wide batches: the F16 route for the shapes it wins, else the
            // int8 WMMA GEMM over tiled Q8 rows. Both staging buffers hold
            // max_batch rows; a wider call (the draft block folds its
            // streams into rows) runs in pieces.
            let f16 = self.dense_f16_route(w, n);
            let piece = self.max_batch;
            let mut r0 = 0;
            while r0 < n {
                let rows_now = piece.min(n - r0);
                let src = x.range(r0 as usize * kk..(r0 + rows_now) as usize * kk);
                let dst = out
                    .reborrow()
                    .range(r0 as usize * m..(r0 + rows_now) as usize * m);
                if f16 {
                    self.prepare_half(src, rows_now, kk)?;
                    let half = self.stage.x_half.slice(0..rows_now as usize * kk);
                    launched(
                        k::dense_f16_gemm(
                            self.stream,
                            w.view(),
                            half,
                            dst,
                            rows_now as usize,
                            m,
                            kk,
                        ),
                        "dense F16 GEMM failed",
                    )?;
                } else {
                    let key = (src.address(), rows_now, kk);
                    let bytes = k::q8_tiled_bytes(rows_now as usize, kk);
                    if self.stage.q8t_src != Some(key) {
                        k::quantize_q8_tiled(
                            self.stream,
                            src,
                            self.stage.x_q8t.slice_mut(0..bytes),
                            rows_now as usize,
                            kk,
                        )
                        .map_err(launch_error("activation quantization"))?;
                        self.stage.q8t_src = Some(key);
                    }
                    launched(
                        k::w8a8_gemm(
                            self.stream,
                            w.view(),
                            self.stage.x_q8t.slice(0..bytes),
                            dst,
                            rows_now as usize,
                            m,
                            kk,
                        ),
                        "W8A8 GEMM failed",
                    )?;
                }
                r0 += piece;
            }
            return Ok(());
        }
        let x = x.range(0..n as usize * kk);
        let out = out.range(0..n as usize * m);
        if !self.matrix_rows(n) {
            return k::small_gemm(self.stream, w.view(), x, out, n, w.rows, w.cols)
                .map_err(launch_error("small projection"));
        }
        // Wide batches of the unquantized projections (router, alpha/beta,
        // indexer).
        let bf16 = match w.format() {
            GgmlType::Bf16 => true,
            GgmlType::F16 => false,
            _ => return Err("F32 projections of wide batches need hipBLAS SGEMM, which this port does not carry".into()),
        };
        let key = (x.address(), n, kk, bf16);
        let count = n as usize * kk;
        if self.stage.half_src != Some(key) {
            if bf16 {
                let target = self
                    .stage
                    .x_half
                    .slice_mut(0..count)
                    .cast::<Bf16>()
                    .ok_or("misaligned BF16 staging")?;
                k::narrow_bf16(self.stream, x, target, count)
                    .map_err(launch_error("activation narrowing"))?;
            } else {
                k::narrow_f16(self.stream, x, self.stage.x_half.slice_mut(0..count), count)
                    .map_err(launch_error("activation narrowing"))?;
            }
            self.stage.half_src = Some(key);
        }
        self.gemm_16(w, count, out, n, bf16)
    }

    /// Row-major `out[n][m] = input[n][k] . weights[m][k]^T` over F16/BF16
    /// rows in `x_half` (`BlasLt::Gemm`).
    fn gemm_16(
        &mut self,
        w: &Matrix,
        count: usize,
        out: DeviceSliceMut<'_, f32>,
        n: u32,
        bf16: bool,
    ) -> Result<()> {
        let (m, kk) = (rows(w), cols(w));
        // Library edge tiles change the accumulation order when a token moves
        // within a batch; the router and recurrent gates keep one K reduction.
        if !bf16 && kk == 2560 && (m == 96 || m == 513) {
            let half = self.stage.x_half.slice(0..count);
            return launched(
                k::unquantized_f16_gemm(self.stream, w.view(), half, out, n as usize, m, kk),
                "unquantized F16 GEMM failed",
            );
        }
        // The indexer query and key projections (and a TP2 rank's halves)
        // sum in K order for every token: the library's
        // MT32x32x64 kernel rotates each 32-token tile's K loop by the tile's
        // index, so a score depended on the token's position modulo 128 in
        // the chunk and a top-k near-tie could pick another key.
        if bf16 && kk == 2560 && matches!(m, 64 | 128 | 256 | 512) {
            let x = self
                .stage
                .x_half
                .slice(0..count)
                .cast::<Bf16>()
                .ok_or("misaligned BF16 rows")?;
            return launched(
                k::dense_bf16_gemm(self.stream, w.view(), x, out, n as usize, m, kk),
                "BF16 GEMM failed",
            );
        }
        let shape = MatmulShape {
            trans_a: Some(Op::T),
            trans_b: Some(Op::N),
            a: Layout::new(kk, m, kk),
            b: Layout::new(kk, n as usize, kk),
            c: Layout::new(m, n as usize, m),
            batch: None,
        };
        let weights = w.view().bytes().range(0..m * kk * 2);
        let result = if bf16 {
            let a = weights.cast::<Bf16>().ok_or("misaligned BF16 weights")?;
            let b = self
                .stage
                .x_half
                .slice(0..count)
                .cast::<Bf16>()
                .ok_or("misaligned BF16 rows")?;
            self.blaslt
                .matmul(self.stream, &shape, a, b, out, &FlashNextPlans)
        } else {
            let a = weights.cast::<F16>().ok_or("misaligned F16 weights")?;
            let b = self.stage.x_half.slice(0..count);
            self.blaslt
                .matmul(self.stream, &shape, a, b, out, &FlashNextPlans)
        };
        launched(
            result,
            "no workspace-free hipBLASLt kernel for the GEMM shape",
        )
    }

    /// `out = (up . x) * silu(gate . x)` (`GatedDense`); `gate_scratch` is the
    /// C++ `shexp_gate`. With `down`, a wide batch leaves only the staged
    /// input form of that projection (F16 rows in `shexp_half` or tiled Q8
    /// rows in `x_q8t`); `out` then holds the gate projection. Returns the
    /// `x_q8` slot that holds the Q8_1 rows of `x`, if it quantized them.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_dense(
        &mut self,
        up: &Matrix,
        gate: &Matrix,
        x: DeviceSlice<'_, f32>,
        mut out: DeviceSliceMut<'_, f32>,
        mut gate_scratch: DeviceSliceMut<'_, f32>,
        shexp_half: Option<DeviceSliceMut<'_, F16>>,
        n: u32,
        down: Option<&Matrix>,
    ) -> Result<Option<usize>> {
        // Decode and verification share one fused projection and activation.
        if !self.matrix_rows(n)
            && up.format() == GgmlType::Q8_0
            && gate.format() == GgmlType::Q8_0
            && up.rows == gate.rows
            && up.cols == gate.cols
        {
            let q = self.quantize(x, n, up.cols)?;
            let slot = q.slot.ok_or("gated projection without quantized rows")?;
            let bytes = k::q8_1_bytes(n as usize, up.cols as usize);
            let out = out.range(0..n as usize * rows(up));
            launched(
                k::mmq_q8_gated_dense_vec(
                    self.stream,
                    up.view(),
                    gate.view(),
                    self.stage.x_q8[slot].slice(0..bytes),
                    out,
                    up.rows,
                    n,
                    up.cols,
                ),
                "gated Q8_0 GEMV failed",
            )?;
            return Ok(Some(slot));
        }
        // Swiglu is in place over its first operand.
        self.dense(gate, x, out.reborrow(), n)?;
        self.dense(up, x, gate_scratch.reborrow(), n)?;
        let count = n as usize * rows(up);
        if let Some(down) = down
            && down.cols == up.rows
            && self.matrix_rows(n)
            && n <= self.max_batch
        {
            if self.dense_f16_route(down, n) {
                // A private F16 buffer: `x_half` keeps the routed experts'
                // token rows.
                let shexp_half = shexp_half.ok_or("shared expert F16 rows missing")?;
                k::swiglu_half(
                    self.stream,
                    out.as_const().range(0..count),
                    gate_scratch.as_const().range(0..count),
                    shexp_half.range(0..count),
                    count,
                )
                .map_err(launch_error("SwiGLU"))?;
                self.route.shexp_half_ready = true;
                return Ok(None);
            }
            if down.format() == GgmlType::Q8_0 {
                let bytes = k::q8_tiled_bytes(n as usize, rows(up));
                let result = k::swiglu_q8_tiled(
                    self.stream,
                    out.as_const().range(0..count),
                    gate_scratch.as_const().range(0..count),
                    self.stage.x_q8t.slice_mut(0..bytes),
                    n as usize,
                    rows(up),
                )
                .map_err(launch_error("SwiGLU"))?;
                if result == Launch::Launched {
                    self.stage.q8t_src = Some((out.address(), n, rows(up)));
                    return Ok(None);
                }
            }
        }
        k::swiglu(
            self.stream,
            out.range(0..count),
            gate_scratch.as_const().range(0..count),
            count,
        )
        .map_err(launch_error("SwiGLU"))?;
        Ok(None)
    }

    /// The column-grid hints of the tiled tier: the largest bucket and the
    /// tile width fitted to the counts, or the token count with a tile near
    /// twice the mean bucket before any readback (`RoutedHints`).
    pub fn routed_hints(&self, w: &Matrix, n: u32) -> (u32, u32) {
        let h = &self.route.hints;
        if h.max_rows > 0 {
            return (h.max_rows, h.tile_cols);
        }
        (
            n,
            routed_tile_cols(n, self.config().num_experts_used, w.experts),
        )
    }

    /// Routed projection `out[t][s] = W[ids[t][s]] . x[t]` (`Experts`).
    #[allow(clippy::too_many_arguments)]
    pub fn experts(
        &mut self,
        w: &Matrix,
        x: DeviceSlice<'_, f32>,
        ids: IndexSlice<'_>,
        out: DeviceSliceMut<'_, f32>,
        n_rows: u32,
        n_used: u32,
        n_tokens: u32,
    ) -> Result<()> {
        // The vector entries loop over column chunks, so the decode down
        // projection (top-k rows, one expert each) stays on them; verification
        // keeps the decode quantization even past 32 slot rows.
        let tiled = (self.route.prefill || n_rows > 4 * VEC_BATCH) && self.matrix_rows(n_tokens);
        let (m, kk) = (w.rows, w.cols);
        let x = x.range(0..n_rows as usize * cols(w));
        let out = out.range(0..(n_rows * n_used) as usize * rows(w));
        let scratch_bytes = self.stage.mmq.len();
        let result = if tiled {
            let (max_rows, tile_cols) = self.routed_hints(w, n_tokens);
            k::mmq_moe_tiled(
                self.stream,
                w.view(),
                x,
                ids,
                out,
                m,
                kk,
                n_rows,
                w.experts,
                n_used,
                max_rows,
                tile_cols,
                self.stage.mmq.as_mut_slice(),
                scratch_bytes,
            )
        } else {
            k::mmq_moe_vec(
                self.stream,
                w.view(),
                x,
                ids,
                out,
                m,
                kk,
                n_rows,
                w.experts,
                n_used,
                self.stage.mmq.as_mut_slice(),
                scratch_bytes,
            )
        };
        launched(result, "expert GEMM failed")
    }

    /// Routed gate/up projections followed by SwiGLU (`GatedExperts`); `up_e`
    /// takes the up projection when the two run separately. `x_q8` names an
    /// `x_q8` slot that may still hold the Q8_1 rows of `x` (the shared
    /// expert's): the vector projection then skips its own quantization.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_experts(
        &mut self,
        a: &Matrix,
        b: &Matrix,
        x: DeviceSlice<'_, f32>,
        ids: IndexSlice<'_>,
        mut out: DeviceSliceMut<'_, f32>,
        mut up_e: DeviceSliceMut<'_, f32>,
        n_tokens: u32,
        n_used: u32,
        x_q8: Option<usize>,
    ) -> Result<()> {
        let same_shape = a.format() == b.format()
            && a.rows == b.rows
            && a.cols == b.cols
            && a.experts == b.experts;
        let (m, kk) = (a.rows, a.cols);
        let count = (n_tokens * n_used) as usize * rows(a);
        let scratch_bytes = self.stage.mmq.len();
        let xs = x.range(0..n_tokens as usize * cols(a));
        if !self.matrix_rows(n_tokens)
            && n_used <= 32
            && same_shape
            && matches!(a.format(), GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q8_0)
        {
            let result = match self.held_q8(x_q8, xs, n_tokens, kk).and_then(|q| q.slot) {
                Some(slot) => k::mmq_q8_moe_gated_vec(
                    self.stream,
                    a.view(),
                    b.view(),
                    self.stage.x_q8[slot].slice(0..k::q8_1_bytes(n_tokens as usize, kk as usize)),
                    ids,
                    out.range(0..count),
                    m,
                    kk,
                    n_tokens,
                    a.experts,
                    n_used,
                    self.stage.mmq.as_mut_slice(),
                    scratch_bytes,
                ),
                None => k::mmq_moe_gated_vec(
                    self.stream,
                    a.view(),
                    b.view(),
                    xs,
                    ids,
                    out.range(0..count),
                    m,
                    kk,
                    n_tokens,
                    a.experts,
                    n_used,
                    self.stage.mmq.as_mut_slice(),
                    scratch_bytes,
                ),
            };
            return launched(result, "gated expert vector projection failed");
        }
        if !self.route.expert_matrix_rows(n_tokens) && same_shape {
            launched(
                k::mmq_moe_vec_pair(
                    self.stream,
                    a.view(),
                    b.view(),
                    xs,
                    ids,
                    out.reborrow().range(0..count),
                    up_e.reborrow().range(0..count),
                    m,
                    kk,
                    n_tokens,
                    a.experts,
                    n_used,
                    self.stage.mmq.as_mut_slice(),
                    scratch_bytes,
                ),
                "expert vector pair GEMM failed",
            )?;
        } else if same_shape && a.format() == GgmlType::Q4K {
            // The wide Q4_K path shares its gather and tiled quantization.
            let (max_rows, tile_cols) = self.routed_hints(a, n_tokens);
            launched(
                k::mmq_q4k_moe_pair_tiled(
                    self.stream,
                    a.view(),
                    b.view(),
                    xs,
                    ids,
                    out.reborrow().range(0..count),
                    up_e.reborrow().range(0..count),
                    m,
                    kk,
                    n_tokens,
                    a.experts,
                    n_used,
                    max_rows,
                    tile_cols,
                    self.stage.mmq.as_mut_slice(),
                    scratch_bytes,
                ),
                "expert pair GEMM failed",
            )?;
        } else {
            self.experts(a, x, ids, out.reborrow(), n_tokens, n_used, n_tokens)?;
            self.experts(b, x, ids, up_e.reborrow(), n_tokens, n_used, n_tokens)?;
        }
        k::swiglu(
            self.stream,
            out.range(0..count),
            up_e.as_const().range(0..count),
            count,
        )
        .map_err(launch_error("SwiGLU"))
    }
}

/// Column-tile width for the tiled routed GEMMs: the tile at or above twice
/// the mean bucket (`RoutedTileCols`).
pub(crate) fn routed_tile_cols(n_tokens: u32, n_used: u32, n_experts: u32) -> u32 {
    let mean = (n_tokens * n_used / n_experts.max(1)).max(1);
    let mut cols = 16;
    while cols < 80 {
        if cols >= 2 * mean {
            return cols;
        }
        cols += 16;
    }
    80
}

/// Token rows per routed F16 tile for `slots` assignments (`RoutedTileRows`).
pub(crate) fn routed_tile_rows(slots: usize, c: &Config) -> u32 {
    if slots >= 16 * c.num_experts as usize {
        ROUTED_TILE_ROWS_WIDE
    } else {
        ROUTED_TILE_ROWS_NARROW
    }
}
