//! Dense projections: the routing of `ds4_rocm_matmul.hip.hpp` over the
//! generated launchers of `launch.toml`, hipBLAS and hipBLASLt, with the
//! state the C++ runtime kept for them: one grow-only scratch, the F16
//! mirror of a published input and the lazy Q8_0 to F16 weight cache.
//!
//! The routes keep decode's reduction order wherever a verified row must
//! reproduce autoregressive decode (narrow batches replay decode kernels),
//! and take batched kernels, MMQ or hipBLAS(Lt) for prompt chunks.

use std::cell::{Cell, RefCell, RefMut};
use std::sync::atomic::{AtomicBool, Ordering};

use gufo_hip::blas::{AlgoPolicy, Heuristic, Layout, MatmulShape, Op};
use gufo_hip::{DeviceBuffer, DeviceSlice, DeviceSliceMut, F16 as Half, Launch};

use crate::backend::{Backend, BatchMode, Error, Model, Result};
use crate::generated;
use crate::tensor::{Input, Output};

use super::{F16, F32, Weight, ensure, launched, product, typed, typed_mut};

/// Rows from which a batch is a prompt chunk (`DS4_ROCM_WIDE_PREFILL_ROWS`):
/// MMQ for every shape, hipBLASLt, the profiled algorithm choice.
const WIDE_PREFILL_ROWS: u32 = 64;

/// The F16 cache's weight shapes (`in_dim`, `out_dim`) for projections
/// ([`Cached::Shape`]).
const F16_CACHE_SHAPES: [(u32, u32); 5] = [
    (4096, 2048),
    (2048, 4096),
    (4096, 1024),
    (4096, 512),
    (1024, 32768),
];

/// The first MMQ dense launch of the process says so, as the C++ runtime
/// did.
static MMQ_NOTICE: AtomicBool = AtomicBool::new(false);

/// `group_offsets` as the grouped launchers read it: `group_count + 1`
/// increasing token offsets from 0 to `n_tok`, an even count of 2 to 8
/// groups.
fn groups(op: &'static str, group_offsets: &[u32], n_tok: u64) -> Result<u32> {
    let count = group_offsets.len().saturating_sub(1);
    ensure(op, (2..=8).contains(&count) && count.is_multiple_of(2))?;
    ensure(
        op,
        group_offsets[0] == 0 && u64::from(group_offsets[count]) == n_tok,
    )?;
    ensure(op, group_offsets.windows(2).all(|w| w[0] < w[1]))?;
    Ok(count as u32)
}

/// A dimension the launchers take in 32 bits.
fn dim(op: &'static str, value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error { op })
}

/// The first `rows` rows of `width` values of `x`, or an error when `x` is
/// shorter.
fn rows<'a, T: gufo_hip::DeviceCopy>(
    op: &'static str,
    x: DeviceSlice<'a, T>,
    rows: u32,
    width: u32,
) -> Result<DeviceSlice<'a, T>> {
    let len = rows as usize * width as usize;
    ensure(op, len <= x.len())?;
    Ok(x.range(0..len))
}

/// [`rows`] of an output.
fn rows_mut<'a, T: gufo_hip::DeviceCopy>(
    op: &'static str,
    x: DeviceSliceMut<'a, T>,
    rows: u32,
    width: u32,
) -> Result<DeviceSliceMut<'a, T>> {
    let len = rows as usize * width as usize;
    ensure(op, len <= x.len())?;
    Ok(x.range(0..len))
}

/// DS4's hipBLASLt choice (`hipblaslt_gemm_plan_select`): on gfx1151 the
/// profiled candidate first (4 for prompt chunks, 5 and 6 for two
/// `in_dim == 4096` projections; narrow batches keep candidate 0, the
/// heuristic's own choice, which the retained trajectory was recorded
/// against), then every candidate in ranked order; the first usable one
/// that launches is installed, so timing never decides.
struct LtPolicy {
    preferred: Option<usize>,
}

impl LtPolicy {
    fn new(gfx1151: bool, out_dim: u32, n_tok: u32, in_dim: u32) -> Self {
        let preferred = if n_tok < WIDE_PREFILL_ROWS {
            0
        } else if in_dim == 4096 && out_dim == 256 {
            6
        } else if in_dim == 4096 && out_dim == 512 {
            5
        } else {
            4
        };
        Self {
            preferred: gfx1151.then_some(preferred),
        }
    }
}

impl AlgoPolicy for LtPolicy {
    fn choose(&self, _: &[Heuristic], _: usize) -> Option<usize> {
        None
    }

    fn resolve_names(&self) -> bool {
        false
    }

    fn trial_order(&self, ranked: &[Heuristic], _: usize) -> Option<Vec<usize>> {
        let preferred = self.preferred.filter(|&index| index < ranked.len());
        Some(preferred.into_iter().chain(0..ranked.len()).collect())
    }
}

/// The shape of `out (out_dim × n_tok) = Wᵀ x` for a row-major weight of
/// `out_dim` rows and `n_tok` rows of `x`, column-major for BLAS.
fn projection_shape(out_dim: u32, n_tok: u32, in_dim: u32) -> MatmulShape {
    let (out_dim, n_tok, in_dim) = (out_dim as usize, n_tok as usize, in_dim as usize);
    MatmulShape {
        trans_a: Some(Op::T),
        trans_b: Some(Op::N),
        a: Layout::new(in_dim, out_dim, in_dim),
        b: Layout::new(in_dim, n_tok, in_dim),
        c: Layout::new(out_dim, n_tok, out_dim),
        batch: None,
    }
}

/// Which weights the F16 cache admits (the labels of the C++ runtime):
/// projections of the listed shapes, or the attention outputs, whatever
/// their shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Cached {
    Shape,
    AttentionOutput,
}

/// One expanded weight of the F16 cache: `out_dim` rows of `in_dim` values,
/// or `in_dim` rows of `out_dim` when `transposed`.
struct F16Weight {
    model: Model,
    offset: u64,
    in_dim: u32,
    out_dim: u32,
    transposed: bool,
    values: DeviceBuffer<Half>,
}

/// The lazy Q8_0 to F16 weight cache (`hip_q8_f16_ptr`): weights of the
/// listed shapes, expanded on first use while device memory stays above a
/// reserve, all released and the cache disabled after a failure.
#[derive(Default)]
pub(crate) struct F16Weights {
    weights: Vec<F16Weight>,
    bytes: u64,
    disabled: bool,
    budget_notice: bool,
}

impl F16Weights {
    fn find(&self, w: Weight, in_dim: u32, out_dim: u32, transposed: bool) -> Option<usize> {
        self.weights.iter().position(|e| {
            e.model == w.model
                && e.offset == w.offset
                && e.in_dim == in_dim
                && e.out_dim == out_dim
                && e.transposed == transposed
        })
    }

    /// The values of entry `index`.
    pub(super) fn values(&self, index: usize) -> Option<DeviceSlice<'_, Half>> {
        self.weights.get(index).map(|e| e.values.as_slice())
    }

    pub(super) fn is_disabled(&self) -> bool {
        self.disabled
    }

    /// `hip_q8_f16_cache_has_budget`: free device memory stays above the
    /// reserve after `request` bytes.
    fn has_budget(&mut self, request: u64) -> bool {
        let (free, total) = match gufo_hip::device_memory() {
            Ok((free, total)) => (free as u64, total as u64),
            Err(error) => {
                eprintln!("ds4: ROCm q8 fp16 cache memory query failed: {error}; using q8 kernels");
                return false;
            }
        };
        // The cache only accelerates; keep room for BLAS workspaces,
        // transient buffers and driver bookkeeping.
        let reserve = if total >= 112 << 30 {
            512 << 20
        } else {
            (total / 20).max(4096 << 20)
        };
        if request > free || free - request < reserve {
            if !self.budget_notice {
                self.budget_notice = true;
                eprintln!(
                    "ds4: ROCm q8 fp16 cache budget exhausted; using q8 kernels (request={:.2} MiB \
                     cached={:.2} GiB free={:.2} GiB reserve={:.2} GiB total={:.2} GiB)",
                    request as f64 / 1048576.0,
                    self.bytes as f64 / 1073741824.0,
                    free as f64 / 1073741824.0,
                    reserve as f64 / 1073741824.0,
                    total as f64 / 1073741824.0,
                );
            }
            return false;
        }
        true
    }

    /// `hip_q8_f16_cache_disable_after_failure`.
    fn disable_after_failure(&mut self, what: &str, request: u64) {
        if !self.disabled {
            eprintln!(
                "ds4: ROCm q8 fp16 cache disabled after {what} (request={:.2} MiB cached={:.2} \
                 GiB); using q8 kernels",
                request as f64 / 1048576.0,
                self.bytes as f64 / 1073741824.0,
            );
        }
        self.disable();
    }

    /// Disables the cache and frees its weights; dropping them waits for the
    /// device.
    pub(crate) fn disable(&mut self) {
        self.disabled = true;
        self.weights.clear();
        self.bytes = 0;
    }
}

/// The F16 copy of one published F32 input (`hip_f16_input_publish`), which
/// several projections of a layer share instead of converting it each.
#[derive(Default)]
pub(crate) struct F16Input {
    /// Grow-only; kept when the input is cleared.
    mirror: Option<DeviceBuffer<Half>>,
    /// The published input's device address and length.
    source: Option<(usize, usize)>,
}

impl F16Input {
    fn lookup(&self, x: DeviceSlice<'_, f32>) -> Option<DeviceSlice<'_, Half>> {
        let mirror = self.mirror.as_ref()?;
        (self.source == Some((x.address(), x.len()))).then(|| mirror.as_slice().range(0..x.len()))
    }
}

/// The projections' runtime state, which the C++ runtime kept in globals.
#[derive(Default)]
pub(crate) struct Projections {
    /// `hip_tmp_alloc`: replaced by a larger exact allocation when needed.
    scratch: RefCell<Option<DeviceBuffer<u8>>>,
    pub(crate) f16_weights: RefCell<F16Weights>,
    f16_input: RefCell<F16Input>,
    /// MMQ initialised on a gfx1151 (`g_rocm_mmq_ready`).
    pub(crate) mmq_ready: Cell<bool>,
    /// The device is a gfx1151 (`g_rocm_gfx1151`).
    pub(crate) gfx1151: Cell<bool>,
}

/// Activations of `n_tok` rows quantized into Q8_0 blocks in the scratch:
/// the int8 values, then 16-aligned the F32 scales.
pub(super) struct Quantized<'a> {
    pub(super) xq: DeviceSliceMut<'a, u8>,
    pub(super) xscale: DeviceSliceMut<'a, f32>,
}

impl Backend {
    /// Rows below which dense projections replay decode kernels per row
    /// (`ds4_rocm_dense_small_batch_rows`): flattened verification waves hold
    /// up to eight draft blocks.
    pub(super) fn small_batch_rows(&self) -> u32 {
        match self.batch_mode() {
            BatchMode::Off => 0,
            BatchMode::Verifier => 136,
            BatchMode::Support => 24,
        }
    }

    /// The scratch of at least `bytes` (`hip_tmp_alloc`); `None` after a
    /// reported allocation failure, which some routes step over.
    pub(super) fn scratch(&self, bytes: u64, what: &str) -> Option<RefMut<'_, DeviceBuffer<u8>>> {
        let mut slot = self.projections.scratch.borrow_mut();
        if slot.as_ref().is_none_or(|s| (s.len() as u64) < bytes) {
            *slot = None;
            match usize::try_from(bytes)
                .ok()
                .map(|len| self.stream().alloc_persistent::<u8>(len))
            {
                Some(Ok(buffer)) => *slot = Some(buffer),
                Some(Err(error)) => {
                    eprintln!(
                        "ds4: ROCm temp alloc failed for {what} ({:.2} MiB): {error}",
                        bytes as f64 / 1048576.0
                    );
                    return None;
                }
                None => return None,
            }
        }
        RefMut::filter_map(slot, Option::as_mut).ok()
    }

    /// Quantizes `n_tok` rows of `x` into `scratch` as Q8_0 blocks.
    pub(super) fn quantize<'s>(
        &self,
        op: &'static str,
        scratch: &'s mut DeviceBuffer<u8>,
        x: DeviceSlice<'_, f32>,
        in_dim: u32,
        n_tok: u32,
    ) -> Result<Quantized<'s>> {
        let blocks = u64::from(in_dim.div_ceil(32));
        let xq_bytes = product(op, &[u64::from(n_tok), blocks, 32])?;
        let scale_offset = xq_bytes.next_multiple_of(16);
        let bytes = scale_offset + u64::from(n_tok) * blocks * F32;
        ensure(op, bytes <= scratch.len() as u64)?;
        let (xq, xscale) = scratch
            .as_mut_slice()
            .range(0..bytes as usize)
            .split_at(scale_offset as usize);
        let mut q = Quantized {
            xq: xq.range(0..xq_bytes as usize),
            xscale: typed_mut(op, xscale)?,
        };
        launched(
            op,
            generated::quantize_q8_rows(
                self.stream(),
                q.xq.reborrow(),
                q.xscale.reborrow(),
                x,
                in_dim,
                n_tok,
            ),
        )?;
        Ok(q)
    }

    /// Scratch bytes of [`quantize`](Self::quantize) for `n_tok` rows.
    pub(super) fn quantized_bytes(op: &'static str, in_dim: u32, n_tok: u32) -> Result<u64> {
        let blocks = u64::from(in_dim.div_ceil(32));
        let xq = product(op, &[u64::from(n_tok), blocks, 32])?;
        Ok(xq.next_multiple_of(16) + u64::from(n_tok) * blocks * F32)
    }

    /// The cached F16 expansion of a Q8_0 weight (transposed for the
    /// rocWMMA output tile), expanding it when admitted and the budget
    /// allows; its index in the cache.
    pub(super) fn f16_weight(
        &self,
        w: Weight,
        in_dim: u32,
        out_dim: u32,
        cached: Cached,
        transposed: bool,
    ) -> Result<Option<usize>> {
        const OP: &str = "q8 fp16 cache";
        let mut cache = self.projections.f16_weights.borrow_mut();
        if let Some(index) = cache.find(w, in_dim, out_dim, transposed) {
            return Ok(Some(index));
        }
        if cache.disabled
            || (cached == Cached::Shape && !F16_CACHE_SHAPES.contains(&(in_dim, out_dim)))
        {
            return Ok(None);
        }
        let q8_bytes = product(
            OP,
            &[u64::from(out_dim), u64::from(in_dim.div_ceil(32)), 34],
        )?;
        let q8 = self.weight_bytes(w, q8_bytes)?;
        let count = u64::from(in_dim) * u64::from(out_dim);
        let bytes = count * F16;
        if !cache.has_budget(bytes) {
            return Ok(None);
        }
        let (what, failure) = if transposed {
            ("transpose ", "transpose ")
        } else {
            ("", "")
        };
        let mut values = match self.stream().alloc_persistent::<Half>(count as usize) {
            Ok(values) => values,
            Err(error) => {
                eprintln!(
                    "ds4: ROCm q8 fp16 {what}cache alloc failed ({:.2} MiB): {error}",
                    bytes as f64 / 1048576.0
                );
                cache.disable_after_failure(&format!("{failure}allocation failure"), bytes);
                return Ok(None);
            }
        };
        let expanded = if transposed {
            generated::dequant_q8_to_f16_transposed(
                self.stream(),
                values.as_mut_slice(),
                q8,
                in_dim,
                out_dim,
            )
        } else {
            generated::dequant_q8_to_f16(self.stream(), values.as_mut_slice(), q8, in_dim, out_dim)
        };
        if let Err(error) = expanded {
            eprintln!("ds4: ROCm q8 fp16 {what}dequant launch failed: {error}");
            drop(values);
            let launch = if transposed {
                "transpose launch failure"
            } else {
                "dequant launch failure"
            };
            cache.disable_after_failure(launch, bytes);
            return Ok(None);
        }
        cache.weights.push(F16Weight {
            model: w.model,
            offset: w.offset,
            in_dim,
            out_dim,
            transposed,
            values,
        });
        cache.bytes += bytes;
        Ok(Some(cache.weights.len() - 1))
    }

    /// `out = Wᵀ x` of an F16 weight and F16 rows through hipBLASLt (prompt
    /// chunks) or hipBLAS; the hipBLAS status on failure.
    fn gemm_f16<C: gufo_hip::blas::BlasElement>(
        &self,
        mut out: DeviceSliceMut<'_, C>,
        w: DeviceSlice<'_, Half>,
        xh: DeviceSlice<'_, Half>,
        in_dim: u32,
        out_dim: u32,
        n_tok: u32,
    ) -> std::result::Result<(), gufo_hip::LaunchError> {
        if n_tok >= WIDE_PREFILL_ROWS
            && let Some(lt) = self.blaslt.borrow_mut().as_mut()
        {
            let policy = LtPolicy::new(self.projections.gfx1151.get(), out_dim, n_tok, in_dim);
            let shape = projection_shape(out_dim, n_tok, in_dim);
            // A failed hipBLASLt launch falls back to hipBLAS, as in C++.
            if let Ok(Launch::Launched) =
                lt.matmul(self.stream(), &shape, w, xh, out.reborrow(), &policy)
            {
                return Ok(());
            }
        }
        let (m, n, k) = (out_dim as usize, n_tok as usize, in_dim as usize);
        self.blas
            .gemm(self.stream(), Op::T, Op::N, m, n, k, w, k, xh, k, out, m)
    }

    /// The F16 rows of `x`: the published mirror, or a conversion into the
    /// scratch; `None` when the scratch cannot hold them.
    fn with_f16_rows<T>(
        &self,
        op: &'static str,
        x: DeviceSlice<'_, f32>,
        what: &str,
        run: impl FnOnce(DeviceSlice<'_, Half>) -> Result<T>,
    ) -> Result<Option<T>> {
        let input = self.projections.f16_input.borrow();
        if let Some(xh) = input.lookup(x) {
            return run(xh).map(Some);
        }
        drop(input);
        let Some(mut scratch) = self.scratch(x.len() as u64 * F16, what) else {
            return Ok(None);
        };
        let mut xh = typed_mut::<Half>(op, scratch.as_mut_slice().range(0..x.len() * 2))?;
        launched(
            op,
            generated::convert_f32_to_f16(self.stream(), xh.reborrow(), x, x.len()),
        )?;
        run(xh.as_const()).map(Some)
    }

    /// `hip_matmul_q8_0_tensor_f16_gemm`: a Q8_0 projection through its
    /// cached F16 expansion; `false` where the cache declines or BLAS fails,
    /// which disables the cache.
    pub(super) fn q8_f16_gemm(
        &self,
        op: &'static str,
        out: DeviceSliceMut<'_, f32>,
        w: Weight,
        in_dim: u32,
        out_dim: u32,
        x: DeviceSlice<'_, f32>,
        n_tok: u32,
        cached: Cached,
    ) -> Result<bool> {
        let Some(index) = self.f16_weight(w, in_dim, out_dim, cached, false)? else {
            return Ok(false);
        };
        let gemm = self.with_f16_rows(op, x, "q8 f16 gemm activations", |xh| {
            let cache = self.projections.f16_weights.borrow();
            let values = cache.weights[index].values.as_slice();
            Ok(self.gemm_f16(out, values, xh, in_dim, out_dim, n_tok))
        })?;
        match gemm {
            None => Ok(false),
            Some(Ok(())) => Ok(true),
            Some(Err(error)) => {
                eprintln!("ds4: hipBLAS q8 f16 matmul failed: {error}");
                let bytes = u64::from(in_dim) * u64::from(out_dim) * F16;
                self.projections
                    .f16_weights
                    .borrow_mut()
                    .disable_after_failure("hipBLAS f16 matmul failure", bytes);
                Ok(false)
            }
        }
    }

    /// `hip_matmul_q8_0_tensor_labeled`: `out[t] = W x[t]` with a Q8_0
    /// weight.
    pub(super) fn q8_projection(
        &self,
        op: &'static str,
        out: DeviceSliceMut<'_, f32>,
        w: Weight,
        in_dim: u32,
        out_dim: u32,
        x: DeviceSlice<'_, f32>,
        n_tok: u32,
        cached: Cached,
    ) -> Result<()> {
        ensure(op, in_dim != 0 && out_dim != 0 && n_tok != 0)?;
        let mut out = rows_mut(op, out, n_tok, out_dim)?;
        let x = rows(op, x, n_tok, in_dim)?;
        let small = self.small_batch_rows();
        // The shared expert's down projection: hipBLAS wins for prompt chunks
        // and loses badly for verification blocks (macro-tile padding).
        if n_tok > small
            && in_dim == 2048
            && out_dim == 4096
            && self.q8_f16_gemm(op, out.reborrow(), w, in_dim, out_dim, x, n_tok, cached)?
        {
            return Ok(());
        }
        let blocks = u64::from(in_dim.div_ceil(32));
        let weight = self.weight_bytes(w, product(op, &[u64::from(out_dim), blocks, 34])?)?;
        let whole_blocks = in_dim.is_multiple_of(32);
        // Dense MMQ for prompt chunks, never for verification or support
        // batches, so a verified row keeps decode's order. Below 64 rows only
        // the Q-B projection (out_dim 32768): it improved pp32 throughput
        // while keeping the sequential greedy choice; the other families
        // changed that choice or gained nothing.
        if n_tok >= 32
            && (n_tok >= WIDE_PREFILL_ROWS || out_dim == 32768)
            && self.batch_mode() == BatchMode::Off
            && self.projections.mmq_ready.get()
            && whole_blocks
            && i32::try_from(in_dim).is_ok()
            && i32::try_from(out_dim).is_ok()
            && i32::try_from(n_tok).is_ok()
            && u64::from(n_tok) * u64::from(out_dim) < 1 << 31
        {
            let dense = generated::mmq_q8_dense(
                self.stream(),
                out.reborrow(),
                weight,
                x,
                in_dim,
                out_dim,
                n_tok,
            );
            if launched(op, dense)? == Launch::Launched {
                if !MMQ_NOTICE.swap(true, Ordering::Relaxed) {
                    eprintln!("ds4: ROCm dense Q8 prefill using native HIP MMQ");
                }
                return Ok(());
            }
        }
        if n_tok > 1 {
            if whole_blocks && out_dim >= 1024 && n_tok >= 256 {
                return launched(
                    op,
                    generated::matmul_q8_wmma(
                        self.stream(),
                        out,
                        weight,
                        x,
                        in_dim,
                        out_dim,
                        n_tok,
                    ),
                );
            }
            // Verification-block width: quantize once and reuse each weight
            // block across the batch in the decode kernel's order, which moves
            // the weights once and keeps every row bitwise equal to decode.
            if whole_blocks
                && n_tok <= small
                && (n_tok <= 16 || self.batch_mode() == BatchMode::Verifier)
                && let Some(mut scratch) = self.scratch(
                    Self::quantized_bytes(op, in_dim, n_tok)?,
                    "q8_0 narrow batch prequant",
                )
            {
                let q = self.quantize(op, &mut scratch, x, in_dim, n_tok)?;
                let (xq, xscale) = (q.xq.as_const(), q.xscale.as_const());
                let (rows, row_blocks) = (out_dim as usize, blocks as usize);
                for first in (0..n_tok).step_by(16) {
                    let count = (n_tok - first).min(16);
                    let (t0, t1) = (first as usize, (first + count) as usize);
                    let reuse = launched(
                        op,
                        generated::matmul_q8_preq_reuse(
                            self.stream(),
                            out.reborrow().range(t0 * rows..t1 * rows),
                            weight,
                            xq.range(t0 * row_blocks * 32..t1 * row_blocks * 32),
                            xscale.range(t0 * row_blocks..t1 * row_blocks),
                            in_dim,
                            out_dim,
                            count,
                            if count == 2 { 2 } else { 1 },
                        ),
                    )?;
                    ensure(op, reuse == Launch::Launched)?;
                }
                return Ok(());
            }
            if whole_blocks {
                return launched(
                    op,
                    generated::matmul_q8_sharedx(
                        self.stream(),
                        out,
                        weight,
                        x,
                        in_dim,
                        out_dim,
                        n_tok,
                    ),
                );
            }
            return launched(
                op,
                generated::matmul_q8_warp8(self.stream(), out, weight, x, in_dim, out_dim, n_tok),
            );
        }
        let Some(mut scratch) =
            self.scratch(Self::quantized_bytes(op, in_dim, 1)?, "q8_0 prequant")
        else {
            return Err(Error { op });
        };
        let q = self.quantize(op, &mut scratch, x, in_dim, 1)?;
        launched(
            op,
            generated::matmul_q8_preq_rows(
                self.stream(),
                out,
                weight,
                q.xq.as_const(),
                q.xscale.as_const(),
                in_dim,
                out_dim,
            ),
        )
    }

    /// `ds4_gpu_matmul_f16_tensor`: `out[t] = W x[t]` with an F16 weight.
    fn f16_projection(
        &self,
        op: &'static str,
        out: DeviceSliceMut<'_, f32>,
        w: Weight,
        in_dim: u32,
        out_dim: u32,
        x: DeviceSlice<'_, f32>,
        n_tok: u32,
    ) -> Result<()> {
        ensure(op, in_dim != 0 && out_dim != 0 && n_tok != 0)?;
        let mut out = rows_mut(op, out, n_tok, out_dim)?;
        let x = rows(op, x, n_tok, in_dim)?;
        let bytes = product(op, &[u64::from(out_dim), u64::from(in_dim), F16])?;
        let weight = typed::<Half>(op, self.weight_bytes(w, bytes)?)?;
        let stream = self.stream();
        if self.batch_mode() == BatchMode::Support
            && (2..=8).contains(&n_tok)
            && in_dim == 16384
            && out_dim == 24
        {
            return launched(
                op,
                generated::matmul_f16_tiny_batch(stream, out, weight, x, in_dim, out_dim, n_tok),
            );
        }
        // The router projection (4096 to 256) is latency-bound; shared-x
        // serves the compressor and indexer projections, where reusing the
        // row across outputs wins.
        let router = in_dim == 4096 && out_dim == 256;
        let sharedx = !router && in_dim <= 8192;
        let small = self.small_batch_rows();
        if (2..=6).contains(&n_tok) && n_tok <= small && !sharedx {
            return launched(
                op,
                generated::matmul_f16_ordered_reuse(stream, out, weight, x, in_dim, out_dim, n_tok),
            );
        }
        if n_tok == 8 && n_tok <= small && !sharedx {
            let (rows, cols) = (out_dim as usize, in_dim as usize);
            for half in [0..4, 4..8] {
                launched(
                    op,
                    generated::matmul_f16_ordered_reuse(
                        stream,
                        out.reborrow().range(half.start * rows..half.end * rows),
                        weight,
                        x.range(half.start * cols..half.end * cols),
                        in_dim,
                        out_dim,
                        4,
                    ),
                )?;
            }
            return Ok(());
        }
        // A verification block replays the decode kernel per row: widening the
        // reduction across rows changes the accumulation order of the
        // 16384-wide hyper-connection projection, which flips the greedy
        // choice at the first verified row.
        if n_tok > 1 && n_tok <= small {
            let (rows, cols) = (out_dim as usize, in_dim as usize);
            for t in 0..n_tok as usize {
                self.f16_projection(
                    op,
                    out.reborrow().range(t * rows..(t + 1) * rows),
                    w,
                    in_dim,
                    out_dim,
                    x.range(t * cols..(t + 1) * cols),
                    1,
                )?;
            }
            return Ok(());
        }
        if n_tok > 1 {
            let gemm = self.with_f16_rows(op, x, "f16 gemm activations", |xh| {
                Ok(self.gemm_f16(out, weight, xh, in_dim, out_dim, n_tok))
            })?;
            return match gemm {
                Some(Ok(())) => Ok(()),
                Some(Err(error)) => {
                    eprintln!("ds4: hipBLAS f16 matmul failed: {error}");
                    Err(Error { op })
                }
                None => Err(Error { op }),
            };
        }
        if sharedx {
            return launched(
                op,
                generated::matmul_f16_sharedx_rows(stream, out, weight, x, in_dim, out_dim),
            );
        }
        // Four-wide loads keep the FMA sequence of each of the 32 chunks;
        // aligned chunks avoid the scalar peel.
        if in_dim.is_multiple_of(128) {
            return launched(
                op,
                generated::matmul_f16_ordered_reuse(stream, out, weight, x, in_dim, out_dim, 1),
            );
        }
        launched(
            op,
            generated::matmul_f16_ordered_chunks(stream, out, weight, x, in_dim, out_dim, 1),
        )
    }

    /// `hip_matmul_f16_f16_input_tensor`: an F16 projection of F16 rows
    /// through BLAS; `false` when BLAS fails.
    fn f16_half_projection(
        &self,
        op: &'static str,
        out: DeviceSliceMut<'_, f32>,
        w: Weight,
        in_dim: u32,
        out_dim: u32,
        xh: DeviceSlice<'_, Half>,
        n_tok: u32,
    ) -> Result<bool> {
        ensure(op, in_dim != 0 && out_dim != 0 && n_tok >= 2)?;
        let bytes = product(op, &[u64::from(out_dim), u64::from(in_dim), F16])?;
        let weight = typed::<Half>(op, self.weight_bytes(w, bytes)?)?;
        Ok(self
            .gemm_f16(out, weight, xh, in_dim, out_dim, n_tok)
            .is_ok())
    }

    /// `out[t] = W x[t]` with a Q8_0 weight of `out_dim` rows.
    pub fn matmul_q8_0(
        &self,
        out: &mut impl Output,
        w: Weight,
        in_dim: u64,
        out_dim: u64,
        x: &impl Input,
        n_tok: u64,
    ) -> Result<()> {
        const OP: &str = "matmul q8_0";
        self.q8_projection(
            OP,
            typed_mut(OP, out.output())?,
            w,
            dim(OP, in_dim)?,
            dim(OP, out_dim)?,
            typed(OP, x.input())?,
            dim(OP, n_tok)?,
            Cached::Shape,
        )
    }

    /// The projection of token groups of verification blocks
    /// (`group_offsets`, see [`groups`]), each pair of groups read together.
    pub fn matmul_q8_0_group_pairs(
        &self,
        out: &mut impl Output,
        w: Weight,
        in_dim: u64,
        out_dim: u64,
        x: &impl Input,
        n_tok: u64,
        group_offsets: &[u32],
    ) -> Result<()> {
        const OP: &str = "matmul q8_0 group pairs";
        ensure(OP, in_dim != 0 && in_dim.is_multiple_of(32) && out_dim != 0)?;
        let count = groups(OP, group_offsets, n_tok)?;
        let (in_dim, out_dim, n_tok) = (dim(OP, in_dim)?, dim(OP, out_dim)?, dim(OP, n_tok)?);
        let blocks = u64::from(in_dim / 32);
        let weight = self.weight_bytes(w, product(OP, &[u64::from(out_dim), blocks, 34])?)?;
        let sizes: Vec<u32> = group_offsets.windows(2).map(|g| g[1] - g[0]).collect();
        let equal = sizes[0] <= 6 && sizes.iter().all(|&rows| rows == sizes[0]);
        ensure(OP, equal || sizes.iter().all(|&rows| rows <= 6))?;
        let Some(mut scratch) = self.scratch(
            Self::quantized_bytes(OP, in_dim, n_tok)?,
            "q8_0 grouped-pair prequant",
        ) else {
            return Err(Error { op: OP });
        };
        let x = rows(OP, typed(OP, x.input())?, n_tok, in_dim)?;
        let q = self.quantize(OP, &mut scratch, x, in_dim, n_tok)?;
        let (xq, xscale) = (q.xq.as_const(), q.xscale.as_const());
        let stream = self.stream();
        if equal {
            let all_rows = generated::matmul_q8_preq_all_rows(
                stream,
                typed_mut(OP, out.output())?,
                weight,
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
            );
            if launched(OP, all_rows)? == Launch::Launched {
                return Ok(());
            }
            return launched(
                OP,
                generated::matmul_q8_preq_equal_group_pairs(
                    stream,
                    typed_mut(OP, out.output())?,
                    weight,
                    xq,
                    xscale,
                    in_dim,
                    out_dim,
                    sizes[0],
                    count / 2,
                ),
            );
        }
        for pair in 0..count as usize / 2 {
            launched(
                OP,
                generated::matmul_q8_preq_group_pair(
                    stream,
                    typed_mut(OP, out.output())?,
                    weight,
                    xq,
                    xscale,
                    in_dim,
                    out_dim,
                    group_offsets[2 * pair],
                    sizes[2 * pair],
                    sizes[2 * pair + 1],
                ),
            )?;
        }
        Ok(())
    }

    /// `out[t] = W x[t]` with an F16 weight.
    pub fn matmul_f16(
        &self,
        out: &mut impl Output,
        w: Weight,
        in_dim: u64,
        out_dim: u64,
        x: &impl Input,
        n_tok: u64,
    ) -> Result<()> {
        const OP: &str = "matmul f16";
        self.f16_projection(
            OP,
            typed_mut(OP, out.output())?,
            w,
            dim(OP, in_dim)?,
            dim(OP, out_dim)?,
            typed(OP, x.input())?,
            dim(OP, n_tok)?,
        )
    }

    /// `out[t] = W x[t]` with an F16 weight for groups of rows
    /// (`group_offsets`, 2 to 8 groups, in pairs), each pair's rows read
    /// together: one launch for equal groups of at most six rows, else one per
    /// pair.
    pub fn matmul_f16_group_pairs(
        &self,
        out: &mut impl Output,
        w: Weight,
        in_dim: u64,
        out_dim: u64,
        x: &impl Input,
        n_tok: u64,
        group_offsets: &[u32],
    ) -> Result<()> {
        const OP: &str = "matmul f16 group pairs";
        ensure(OP, w.offset.is_multiple_of(8))?;
        let count = groups(OP, group_offsets, n_tok)?;
        let w = typed(
            OP,
            self.weight_bytes(w, product(OP, &[out_dim, in_dim, F16])?)?,
        )?;
        let in_dim = dim(OP, in_dim)?;
        let out_dim = dim(OP, out_dim)?;
        let sizes: Vec<u32> = group_offsets.windows(2).map(|g| g[1] - g[0]).collect();
        if sizes[0] <= 6 && sizes.iter().all(|&rows| rows == sizes[0]) {
            return launched(
                OP,
                generated::matmul_f16_equal_group_pairs(
                    self.stream(),
                    typed_mut(OP, out.output())?,
                    w,
                    typed(OP, x.input())?,
                    in_dim,
                    out_dim,
                    sizes[0],
                    count / 2,
                ),
            );
        }
        ensure(OP, sizes.iter().all(|&rows| rows <= 6))?;
        for pair in 0..count as usize / 2 {
            launched(
                OP,
                generated::matmul_f16_ragged_group_pair(
                    self.stream(),
                    typed_mut(OP, out.output())?,
                    w,
                    typed(OP, x.input())?,
                    in_dim,
                    out_dim,
                    group_offsets[2 * pair],
                    sizes[2 * pair],
                    sizes[2 * pair + 1],
                ),
            )?;
        }
        Ok(())
    }

    /// Two F16 projections of one input of equal shape: prompt chunks
    /// convert the input once for both.
    pub fn matmul_f16_pair(
        &self,
        out_a: &mut impl Output,
        out_b: &mut impl Output,
        w_a: Weight,
        w_b: Weight,
        in_dim: u64,
        out_dim: u64,
        x: &impl Input,
        n_tok: u64,
    ) -> Result<()> {
        const OP: &str = "matmul f16 pair";
        let (in_dim, out_dim, n_tok) = (dim(OP, in_dim)?, dim(OP, out_dim)?, dim(OP, n_tok)?);
        ensure(OP, in_dim != 0 && out_dim != 0 && n_tok != 0)?;
        let x = rows(OP, typed::<f32>(OP, x.input())?, n_tok, in_dim)?;
        if n_tok >= 128 {
            // Both halves recompute below when either declines.
            let both = self.with_f16_rows_converted(OP, x, |xh| {
                Ok(self.f16_half_projection(
                    OP,
                    typed_mut(OP, out_a.output())?,
                    w_a,
                    in_dim,
                    out_dim,
                    xh,
                    n_tok,
                )? && self.f16_half_projection(
                    OP,
                    typed_mut(OP, out_b.output())?,
                    w_b,
                    in_dim,
                    out_dim,
                    xh,
                    n_tok,
                )?)
            })?;
            if both == Some(true) {
                return Ok(());
            }
        }
        if n_tok != 1 {
            self.f16_projection(
                OP,
                typed_mut(OP, out_a.output())?,
                w_a,
                in_dim,
                out_dim,
                x,
                n_tok,
            )?;
            return self.f16_projection(
                OP,
                typed_mut(OP, out_b.output())?,
                w_b,
                in_dim,
                out_dim,
                x,
                n_tok,
            );
        }
        let bytes = product(OP, &[u64::from(out_dim), u64::from(in_dim), F16])?;
        let wa = typed::<Half>(OP, self.weight_bytes(w_a, bytes)?)?;
        let wb = typed::<Half>(OP, self.weight_bytes(w_b, bytes)?)?;
        let (out_a, out_b) = (
            typed_mut(OP, out_a.output())?,
            typed_mut(OP, out_b.output())?,
        );
        if in_dim <= 8192 {
            return launched(
                OP,
                generated::matmul_f16_pair_sharedx_rows(
                    self.stream(),
                    out_a,
                    out_b,
                    wa,
                    wb,
                    x,
                    in_dim,
                    out_dim,
                ),
            );
        }
        launched(
            OP,
            generated::matmul_f16_pair_ordered_chunks(
                self.stream(),
                out_a,
                out_b,
                wa,
                wb,
                x,
                in_dim,
                out_dim,
            ),
        )
    }

    /// The F16 rows of `x` converted into the scratch (the pair never reads
    /// the published mirror); `None` when the scratch cannot hold them.
    fn with_f16_rows_converted<T>(
        &self,
        op: &'static str,
        x: DeviceSlice<'_, f32>,
        run: impl FnOnce(DeviceSlice<'_, Half>) -> Result<T>,
    ) -> Result<Option<T>> {
        let Some(mut scratch) = self.scratch(x.len() as u64 * F16, "f16 pair gemm activations")
        else {
            return Ok(None);
        };
        let mut xh = typed_mut::<Half>(op, scratch.as_mut_slice().range(0..x.len() * 2))?;
        launched(
            op,
            generated::convert_f32_to_f16(self.stream(), xh.reborrow(), x, x.len()),
        )?;
        run(xh.as_const()).map(Some)
    }

    /// Two F16 projections of the same 2 to 8 rows.
    pub fn matmul_f16_pair_narrow(
        &self,
        out_a: &mut impl Output,
        out_b: &mut impl Output,
        w_a: Weight,
        w_b: Weight,
        in_dim: u64,
        out_dim: u64,
        x: &impl Input,
        n_tok: u64,
    ) -> Result<()> {
        const OP: &str = "matmul f16 pair narrow";
        ensure(
            OP,
            w_a.offset.is_multiple_of(8) && w_b.offset.is_multiple_of(8),
        )?;
        let bytes = product(OP, &[out_dim, in_dim, F16])?;
        let w_a = typed(OP, self.weight_bytes(w_a, bytes)?)?;
        let w_b = typed(OP, self.weight_bytes(w_b, bytes)?)?;
        launched(
            OP,
            generated::matmul_f16_pair_narrow(
                self.stream(),
                typed_mut(OP, out_a.output())?,
                typed_mut(OP, out_b.output())?,
                w_a,
                w_b,
                typed(OP, x.input())?,
                dim(OP, in_dim)?,
                dim(OP, out_dim)?,
                dim(OP, n_tok)?,
            ),
        )
    }

    /// `out[t] = W x[t]` with an F32 weight: `hipblasSgemm` for batches.
    pub fn matmul_f32(
        &self,
        out: &mut impl Output,
        w: Weight,
        in_dim: u64,
        out_dim: u64,
        x: &impl Input,
        n_tok: u64,
    ) -> Result<()> {
        const OP: &str = "matmul f32";
        let (in_dim, out_dim, n_tok) = (dim(OP, in_dim)?, dim(OP, out_dim)?, dim(OP, n_tok)?);
        ensure(OP, in_dim != 0 && out_dim != 0 && n_tok != 0)?;
        let bytes = product(OP, &[u64::from(out_dim), u64::from(in_dim), F32])?;
        let weight = typed::<f32>(OP, self.weight_bytes(w, bytes)?)?;
        let (out, x) = (typed_mut(OP, out.output())?, typed(OP, x.input())?);
        if n_tok > 1 {
            let (m, n, k) = (out_dim as usize, n_tok as usize, in_dim as usize);
            return self
                .blas
                .sgemm(
                    self.stream(),
                    Op::T,
                    Op::N,
                    m,
                    n,
                    k,
                    1.0,
                    weight,
                    k,
                    x,
                    k,
                    out,
                    m,
                )
                .map_err(|error| {
                    eprintln!("ds4: hipBLAS f32 matmul failed: {error}");
                    Error { op: OP }
                });
        }
        launched(
            OP,
            generated::matmul_f32_plain(self.stream(), out, weight, x, in_dim, out_dim, 1),
        )
    }

    /// An F16 projection of an F16 input of at least two rows through BLAS;
    /// `false` when BLAS fails, where the caller converts instead.
    pub fn matmul_f16_f16_input(
        &self,
        out: &mut impl Output,
        w: Weight,
        in_dim: u64,
        out_dim: u64,
        x_h: &impl Input,
        n_tok: u64,
    ) -> Result<bool> {
        const OP: &str = "matmul f16 f16 input";
        ensure(OP, w.offset.is_multiple_of(8))?;
        self.f16_half_projection(
            OP,
            typed_mut(OP, out.output())?,
            w,
            dim(OP, in_dim)?,
            dim(OP, out_dim)?,
            typed(OP, x_h.input())?,
            dim(OP, n_tok)?,
        )
    }

    /// Publishes an F16 mirror of the first `count` floats of `x` for the
    /// projections of this layer to share (`hip_f16_input_publish`); `false`
    /// when none could be made (not an error). The caller clears it at each
    /// layer boundary ([`Backend::clear_f16_input`]).
    pub fn publish_f16_input(&self, x: &impl Input, count: u64) -> bool {
        let mut input = self.projections.f16_input.borrow_mut();
        input.source = None;
        let Some(x) = x.input().cast::<f32>() else {
            return false;
        };
        let Some(count) = usize::try_from(count)
            .ok()
            .filter(|&c| c != 0 && c <= x.len())
        else {
            return false;
        };
        let x = x.range(0..count);
        if input.mirror.as_ref().is_none_or(|m| m.len() < count) {
            input.mirror = None;
            match self.stream().alloc_persistent::<Half>(count) {
                Ok(mirror) => input.mirror = Some(mirror),
                Err(error) => {
                    eprintln!("ds4: ROCm f16 input mirror alloc failed: {error}");
                    return false;
                }
            }
        }
        let Some(mirror) = input.mirror.as_mut() else {
            return false;
        };
        let converted = generated::convert_f32_to_f16(
            self.stream(),
            mirror.as_mut_slice().range(0..count),
            x,
            count,
        );
        if launched("publish f16 input", converted).is_err() {
            return false;
        }
        input.source = Some((x.address(), count));
        true
    }

    /// Withdraws the published F16 input.
    pub fn clear_f16_input(&self) {
        self.projections.f16_input.borrow_mut().source = None;
    }

    /// Two Q8_0 projections of one input (`ds4_gpu_matmul_q8_0_pair_tensor`):
    /// one quantization and paired kernels for decode and verification rows.
    fn q8_pair(
        &self,
        op: &'static str,
        out0: DeviceSliceMut<'_, f32>,
        out1: DeviceSliceMut<'_, f32>,
        w0: Weight,
        w1: Weight,
        in_dim: u32,
        out_dim: u32,
        x: DeviceSlice<'_, f32>,
        n_tok: u32,
    ) -> Result<()> {
        ensure(op, in_dim != 0 && out_dim != 0 && n_tok != 0)?;
        let (mut out0, mut out1) = (
            rows_mut(op, out0, n_tok, out_dim)?,
            rows_mut(op, out1, n_tok, out_dim)?,
        );
        let x = rows(op, x, n_tok, in_dim)?;
        if n_tok > 1 && (n_tok > 136 || !in_dim.is_multiple_of(32)) {
            self.q8_projection(
                op,
                out0.reborrow(),
                w0,
                in_dim,
                out_dim,
                x,
                n_tok,
                Cached::Shape,
            )?;
            return self.q8_projection(op, out1, w1, in_dim, out_dim, x, n_tok, Cached::Shape);
        }
        let bytes = product(
            op,
            &[u64::from(out_dim), u64::from(in_dim.div_ceil(32)), 34],
        )?;
        let (q8_0, q8_1) = (self.weight_bytes(w0, bytes)?, self.weight_bytes(w1, bytes)?);
        let Some(mut scratch) = self.scratch(
            Self::quantized_bytes(op, in_dim, n_tok)?,
            "q8_0 pair prequant",
        ) else {
            return Err(Error { op });
        };
        let q = self.quantize(op, &mut scratch, x, in_dim, n_tok)?;
        let (xq, xscale) = (q.xq.as_const(), q.xscale.as_const());
        if n_tok > 1 {
            return launched(
                op,
                generated::matmul_q8_pair_preq_reuse(
                    self.stream(),
                    out0.reborrow(),
                    out1.reborrow(),
                    q8_0,
                    q8_1,
                    xq,
                    xscale,
                    in_dim,
                    out_dim,
                    n_tok,
                ),
            );
        }
        launched(
            op,
            generated::matmul_q8_pair_preq_rows(
                self.stream(),
                out0,
                out1,
                q8_0,
                q8_1,
                xq,
                xscale,
                in_dim,
                out_dim,
                out_dim,
            ),
        )
    }

    /// The shared expert's gate and up projections and
    /// `mid = swiglu(gate, up)`.
    pub fn shared_gate_up_swiglu_q8_0(
        &self,
        gate: &mut (impl Output + Input),
        up: &mut (impl Output + Input),
        mid: &mut impl Output,
        w_gate: Weight,
        w_up: Weight,
        in_dim: u64,
        out_dim: u64,
        x: &impl Input,
        clamp: f32,
        n_tok: u32,
    ) -> Result<()> {
        const OP: &str = "shared gate up swiglu q8_0";
        let n = u64::from(n_tok);
        let elements = dim(OP, product(OP, &[out_dim, n])?)?;
        self.q8_pair(
            OP,
            typed_mut(OP, gate.output())?,
            typed_mut(OP, up.output())?,
            w_gate,
            w_up,
            dim(OP, in_dim)?,
            dim(OP, out_dim)?,
            typed(OP, x.input())?,
            n_tok,
        )?;
        self.swiglu(mid, &*gate, &*up, elements, clamp, 1.0)
    }

    /// One row's Q8_0 projection into `block_out` (plus `block_add`),
    /// expanded into the hyper-connection streams.
    fn hc_expand(
        &self,
        op: &'static str,
        out_hc: &mut impl Output,
        block_out: &mut impl Output,
        w: Weight,
        in_dim: u64,
        out_dim: u64,
        x: &impl Input,
        block_add: Option<&dyn Input>,
        residual_hc: &impl Input,
        split: &impl Input,
        n_embd: u32,
        n_hc: u32,
    ) -> Result<()> {
        ensure(op, out_dim == u64::from(n_embd))?;
        let in_dim = dim(op, in_dim)?;
        ensure(op, in_dim != 0 && n_embd != 0)?;
        let bytes = product(op, &[u64::from(n_embd), u64::from(in_dim.div_ceil(32)), 34])?;
        let weight = self.weight_bytes(w, bytes)?;
        let Some(mut scratch) = self.scratch(
            Self::quantized_bytes(op, in_dim, 1)?,
            "q8_0 hc expand prequant",
        ) else {
            return Err(Error { op });
        };
        let q = self.quantize(op, &mut scratch, typed(op, x.input())?, in_dim, 1)?;
        let block_add = block_add.map(|add| typed(op, add.input())).transpose()?;
        launched(
            op,
            generated::matmul_q8_hc_expand_preq_rows(
                self.stream(),
                typed_mut(op, out_hc.output())?,
                typed_mut(op, block_out.output())?,
                block_add,
                typed(op, residual_hc.input())?,
                typed(op, split.input())?,
                weight,
                q.xq.as_const(),
                q.xscale.as_const(),
                in_dim,
                n_embd,
                n_hc,
            ),
        )
    }

    /// One token's projection `block_out = W x` expanded into the
    /// hyper-connection streams.
    pub fn matmul_q8_0_hc_expand(
        &self,
        out_hc: &mut impl Output,
        block_out: &mut impl Output,
        w: Weight,
        in_dim: u64,
        out_dim: u64,
        x: &impl Input,
        residual_hc: &impl Input,
        split: &impl Input,
        n_embd: u32,
        n_hc: u32,
    ) -> Result<()> {
        self.hc_expand(
            "matmul q8_0 hc expand",
            out_hc,
            block_out,
            w,
            in_dim,
            out_dim,
            x,
            None,
            residual_hc,
            split,
            n_embd,
            n_hc,
        )
    }

    /// The shared expert's down projection plus the routed output, expanded
    /// into the streams.
    pub fn shared_down_hc_expand_q8_0(
        &self,
        out_hc: &mut impl Output,
        shared_out: &mut impl Output,
        w: Weight,
        in_dim: u64,
        out_dim: u64,
        shared_mid: &impl Input,
        routed_out: &impl Input,
        residual_hc: &impl Input,
        split: &impl Input,
        n_embd: u32,
        n_hc: u32,
    ) -> Result<()> {
        self.hc_expand(
            "shared down hc expand q8_0",
            out_hc,
            shared_out,
            w,
            in_dim,
            out_dim,
            shared_mid,
            Some(routed_out),
            residual_hc,
            split,
            n_embd,
            n_hc,
        )
    }
}
