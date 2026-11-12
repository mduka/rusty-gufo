//! The routed-expert feed-forward of a `qwen35moe` layer: a softmax router
//! picks `expert_used_count` of `expert_count` SwiGLU experts per token, a
//! shared SwiGLU expert runs beside them under a sigmoid gate, and the
//! epilogue adds the weighted expert rows to the gated shared row. A port of
//! the extension's `ExecuteMoeDecodeStep` (`hip/decode_step.cpp`),
//! `LaunchMoeFfn` (`hip/batched_decode.cpp`) and `ExecuteMoePrefillChunk`
//! (`hip/prefill_chunk.cpp`), with their launches, arguments and order.
//!
//! Three entries share one [`MoeScratch`], each with the arithmetic of its
//! C++ caller:
//!
//! - [`decode`]: one token (`ExecuteMoeDecodeStep`). The fused router
//!   (router GEMV, shared-expert gate and top-k in one launch), the fused
//!   shared-expert SwiGLU GEMV, then the vendored MMQ vector kernels for the
//!   routed gate/up (SwiGLU fused) and down projections. Without `combine`
//!   the caller folds the epilogue into its residual add and next RMSNorm
//!   with `fused_moe_residual_rms_norm` over [`MoeScratch::down_e`],
//!   [`MoeScratch::weights`], [`MoeScratch::shared_out`] and
//!   [`MoeScratch::shared_gate`] (its first `expert_used_count` rows).
//! - [`rows`]: up to [`ROWS_MAX`] independent token rows with decode
//!   arithmetic (`LaunchMoeFfn`), for batched decode and verification: the
//!   router and the shared expert through the verification projection
//!   ([`crate::verify::project`]), the routed experts through the same MMQ
//!   kernels in slices of eight rows. Each row's output equals what the C++
//!   batched decode or verification computes for it.
//! - [`prefill`]: one prompt chunk (`ExecuteMoePrefillChunk`). The router on
//!   the BF16 WMMA GEMM from 128 rows (hipBLAS F32 below), the shared expert
//!   on the prefill GEMMs from the tiled Q8_1 activation, and the routed
//!   experts on the F16 WMMA GEMMs over a tile map built on the host from the
//!   per-expert counts; from 1024 rows gate and up run as one paired launch.
//!
//! How the forward passes call them (all on the model's stream, with the
//! model's [`Scratch`]):
//!
//! - decode (`crate::decode`): FFN-norm the row into `x.normed[..hidden]`,
//!   then `decode(s, &layer.moe, c, &mut x.moe, x.normed.slice(..hidden),
//!   out)`; with `out = Some(x.ffn_out[..hidden])` add `x.ffn_out` to the
//!   hidden row, with `None` finish with `k::fused_moe_residual_rms_norm`
//!   over `x.moe.down_e[..used * hidden]`, `x.moe.weights[..used]`,
//!   `x.moe.shared_out[..hidden]` and `x.moe.shared_gate[..1]` and the next
//!   norm's weight, as the C++ step does whenever a next norm exists.
//! - batched decode and verification rows (`crate::verify`): FFN-norm `n`
//!   rows into `x.normed`, then `rows(s, &layer.moe, c, &mut x.moe,
//!   x.normed.slice(..n * hidden), x.ffn_gate.as_mut_slice(),
//!   x.ffn_up.as_mut_slice(), x.ffn_out.slice_mut(..n * hidden), n)` and a
//!   `batched_residual_add` of `x.ffn_out`; rows keep their order, so
//!   sessions stack their rows as in `ForwardTokenBatch` and
//!   `ForwardVerificationBatch`.
//! - prefill (`crate::prefill`): after the FFN norm, `prefill(s,
//!   &layer.moe, c, x, batch, norm_quantized)`, which also adds the FFN
//!   output to `x.hidden`.
//!
//! Supported expert formats are those of the published Qwen3.6 35B-A3B
//! files: gate/up Q4_K, Q5_K or Q8_0 and down Q4_K, Q5_K, Q6_K or Q8_0, with
//! an F32 router and shared-expert gate. The C++ fallbacks for other formats
//! (per-slot GEMVs, the grouped BF16 GEMM) are not ported; their entries
//! return an error naming the format.

use gufo_hip::blas::{Blas, Op};
use gufo_hip::{
    Bf16, DeviceBuffer, DeviceIndices, DeviceSlice, DeviceSliceMut, F16, GgmlType, IndexSpec,
    Launch, LaunchError, PinnedBuffer, QuantTensor, Stream,
};
use gufo_qwen35moe_kernels as k;
use gufo_qwen35moe_kernels::sizes;

use crate::arena::{Scratch, alloc};
use crate::config::ModelConfig;
use crate::prefill::gemm;
use crate::routes::reads_q8_activation;
use crate::verify::project;
use crate::weights::MoeWeights;

/// Token rows one [`rows`] call takes: a wide single-session verification
/// chain (`kMaxWideVerificationRows`).
pub const ROWS_MAX: usize = 64;
/// Token rows per launch of the gated vector kernel (`MMVQ_MAX_BATCH_SIZE`).
const GATED_VEC_MAX_ROWS: usize = 8;

fn invalid(param: &'static str, reason: &'static str) -> LaunchError {
    LaunchError::InvalidArgument { param, reason }
}

fn unsupported(what: &'static str) -> LaunchError {
    LaunchError::InvalidArgument {
        param: "experts",
        reason: what,
    }
}

/// `Launch::Unsupported` from a launcher C++ requires to run.
fn launched(launch: Launch, what: &'static str) -> Result<(), LaunchError> {
    match launch {
        Launch::Launched => Ok(()),
        Launch::Unsupported => Err(unsupported(what)),
    }
}

/// The F32 values of an F32 weight, for the BLAS and BF16 conversions.
pub(crate) fn f32_weights(w: &QuantTensor) -> Result<DeviceSlice<'_, f32>, LaunchError> {
    if w.format() != GgmlType::F32 {
        return Err(invalid(
            "weights",
            "the hipBLAS F32 route takes F32 weights",
        ));
    }
    w.bytes().view().cast::<f32>().ok_or(invalid(
        "weights",
        "F32 weights are not whole, aligned floats",
    ))
}

/// `LaunchHipblasGEMM` with F32 weights: `y[batch][m] = x[batch][k] w[m][k]^T`
/// through `hipblasGemmEx` (OP_T, OP_N) on a handle with the default atomics
/// mode, F32 compute.
#[allow(clippy::too_many_arguments)]
pub(crate) fn blas_gemm(
    s: &Stream,
    blas: &Blas,
    w: &QuantTensor,
    x: DeviceSlice<'_, f32>,
    y: DeviceSliceMut<'_, f32>,
    batch: usize,
    m: usize,
    k: usize,
) -> Result<(), LaunchError> {
    blas.gemm::<f32, f32, f32>(s, Op::T, Op::N, m, batch, k, f32_weights(w)?, k, x, k, y, m)
}

/// Whether `rows` x `k` F32 weights of an MoE model take the BF16 WMMA GEMM
/// in this chunk, converted into `scratch_bytes` of scratch
/// (`CanUseBf16WmmaProjection`).
pub(crate) fn bf16_wmma_projection(
    batch: usize,
    rows: usize,
    k: usize,
    scratch_bytes: usize,
) -> bool {
    batch >= sizes::BF16_WMMA_MIN_PREFILL_BATCH
        && sizes::grouped_bf16_gemm(rows, k, 1)
        && scratch_bytes >= rows * k * 2
}

/// The routed F16 GEMMs' decode of an expert format with rows of `k` values
/// (`RoutedWeightType`): Q8_0 in whole 64-value blocks, the K-quants in
/// whole superblocks.
fn routed_weight_type(format: GgmlType, k: usize) -> Option<GgmlType> {
    match format {
        GgmlType::Q8_0 if k.is_multiple_of(64) => Some(format),
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K if k.is_multiple_of(256) => Some(format),
        _ => None,
    }
}

/// Formats of the MMQ gated vector kernel (`IsMoeMmqGatedVecType`).
fn mmq_gated_type(format: GgmlType) -> bool {
    matches!(format, GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q8_0)
}

/// Formats of the MMQ vector kernel (`IsMoeMmqVecType`).
fn mmq_vec_type(format: GgmlType) -> bool {
    matches!(
        format,
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q5_1 | GgmlType::Q8_0
    )
}

/// Row capacity of the routed buckets, each expert padded to 16 rows
/// (`RoutedRows`).
fn routed_rows(slots: usize, experts: usize) -> usize {
    sizes::routed_compact_rows(slots, experts)
}

/// Tile-map capacity: the 16-row map plus the paired map large chunks
/// append (`RoutedTiles`).
fn routed_tiles(slots: usize, experts: usize) -> usize {
    let rows = routed_rows(slots, experts);
    rows / 16 + rows / 64 + 2 * experts + 2
}

/// Routing and expert workspaces of the MoE FFN (`QwenMoeScratch`), sized
/// for `batch` token rows (the prefill batch). Slot buffers hold
/// `batch * expert_used_count` rows, token-major.
pub(crate) struct MoeScratch {
    /// Token rows the buffers hold.
    batch: usize,
    /// Completion counter of the one-token fused router; zero between
    /// launches.
    router_done: DeviceBuffer<u32>,
    /// `[token][expert]` router logits.
    router_logits: DeviceBuffer<f32>,
    /// The shared expert's gate logit per token.
    pub(crate) shared_gate: DeviceBuffer<f32>,
    /// `[token][slot]` expert IDs, padded (-1) for non-finite router rows.
    ids: DeviceIndices,
    /// `[token][slot]` renormalized router weights.
    pub(crate) weights: DeviceBuffer<f32>,
    /// `[slot][expert_ff]`: the routed SwiGLU activation (F32), or the gate
    /// rows of the unpaired prefill route.
    gate_e: DeviceBuffer<f32>,
    /// `[slot][expert_ff]`: the prefill's F16 SwiGLU activation.
    up_e: DeviceBuffer<f32>,
    /// `[slot][hidden]` routed expert outputs (F32 in decode, F16 in prefill).
    pub(crate) down_e: DeviceBuffer<f32>,
    /// `[token][hidden]` shared expert output.
    pub(crate) shared_out: DeviceBuffer<f32>,
    /// `[token][shared_ff]` shared expert activation of decode.
    shared_act: DeviceBuffer<f32>,
    /// Per-expert slot counts, derived from `ids`.
    counts: DeviceIndices,
    /// `[expert + 1]` starts of the 16-row padded buckets.
    bounds: DeviceIndices,
    cursors: DeviceBuffer<i32>,
    /// Token and slot of each compact bucket row (-1 for padding).
    rows_token: DeviceIndices,
    rows_slot: DeviceIndices,
    /// `expert | tile << 16` entries of the routed tile maps.
    tiles: DeviceIndices,
    /// F16 copy of the normed rows for the routed GEMMs.
    x_half: DeviceBuffer<F16>,
    /// The fused norms' tiled BF16 copy of their output, `[hidden / 64][batch][64]`.
    pub(crate) x_tiled: DeviceBuffer<Bf16>,
    /// Caller scratch of the MMQ vector kernels.
    mmq: DeviceBuffer<u8>,
    /// Pinned host side of the tile map (`RoutedHostState`).
    host_counts: PinnedBuffer<u32>,
    host_tiles: PinnedBuffer<u32>,
}

impl MoeScratch {
    pub(crate) fn new(stream: &Stream, c: &ModelConfig, batch: usize) -> Result<Self, String> {
        let experts = c.expert_count as usize;
        let used = c.expert_used_count as usize;
        let hidden = c.hidden_size as usize;
        let expert_ff = c.expert_ff_length as usize;
        let shared_ff = c.expert_shared_ff_length as usize;
        let slots = batch * used;
        let compact = routed_rows(slots, experts);
        let tile_capacity = routed_tiles(slots, experts);
        let error = |e| format!("Qwen MoE scratch: {e}");
        let indices = |len: usize, spec: IndexSpec| {
            stream
                .zeroed_indices_with(alloc::<u32>(stream, len.max(1))?, spec)
                .map_err(error)
        };
        let to_u32 = |n: usize| u32::try_from(n).map_err(|_| "Qwen MoE scratch exceeds u32");
        // The decode and row paths: the gated kernel's rows of eight tokens
        // and the down projection's one row per slot.
        let mmq = sizes::mmq_moe_gated_vec_scratch(GATED_VEC_MAX_ROWS, hidden, used)
            .max(sizes::mmq_moe_vec_scratch(ROWS_MAX * used, expert_ff))
            .max(sizes::mmq_moe_vec_scratch(used, expert_ff));
        let pinned = |len: usize| PinnedBuffer::<u32>::zeroed(len.max(1)).map_err(error);
        let mut router_done = alloc::<u32>(stream, 1)?;
        stream
            .fill_bytes(router_done.as_mut_slice(), 0)
            .map_err(error)?;
        Ok(Self {
            batch,
            router_done,
            router_logits: alloc(stream, batch * experts)?,
            shared_gate: alloc(stream, batch)?,
            ids: indices(slots, IndexSpec::below(to_u32(experts)?).padded().signed())?,
            weights: alloc(stream, slots)?,
            gate_e: alloc(stream, slots * expert_ff)?,
            up_e: alloc(stream, slots * expert_ff)?,
            down_e: alloc(stream, slots * hidden)?,
            shared_out: alloc(stream, batch * hidden)?,
            shared_act: alloc(stream, batch * shared_ff)?,
            counts: indices(experts, IndexSpec::below(to_u32(slots + 1)?))?,
            bounds: indices(experts + 1, IndexSpec::below(to_u32(compact + 1)?).signed())?,
            cursors: alloc(stream, experts)?,
            rows_token: indices(compact, IndexSpec::below(to_u32(batch)?).padded().signed())?,
            rows_slot: indices(compact, IndexSpec::below(to_u32(slots)?).padded().signed())?,
            tiles: indices(
                tile_capacity,
                IndexSpec::below(to_u32(experts)?).masked(0xFFFF).signed(),
            )?,
            x_half: alloc(stream, batch * hidden)?,
            x_tiled: alloc(stream, batch * hidden)?,
            mmq: alloc(stream, mmq)?,
            host_counts: pinned(experts)?,
            host_tiles: pinned(tile_capacity)?,
        })
    }
}

/// `x` (one normed row) through the routed FFN of one token
/// (`ExecuteMoeDecodeStep`). With `combine`, the epilogue writes the FFN
/// output row there; without, the caller combines `down_e`, `weights`,
/// `shared_out` and `shared_gate` itself (the fused residual RMSNorm).
pub(crate) fn decode(
    s: &Stream,
    w: &MoeWeights,
    c: &ModelConfig,
    m: &mut MoeScratch,
    x: DeviceSlice<'_, f32>,
    combine: Option<DeviceSliceMut<'_, f32>>,
) -> Result<(), LaunchError> {
    let hidden = c.hidden_size as usize;
    let experts = c.expert_count;
    let used = c.expert_used_count;
    let (e, u) = (experts as usize, used as usize);
    let shared_ff = c.expert_shared_ff_length as usize;
    if sizes::router_gate_top_k(
        w.router.format().id(),
        w.shared_gate.format().id(),
        e,
        hidden,
        u,
    ) {
        // Router, shared-expert gate and top-k in one launch, same
        // arithmetic.
        k::moe_router_gate_top_k(
            s,
            w.router.view(),
            w.shared_gate.view(),
            x,
            m.router_logits.slice_mut(0..e),
            m.shared_gate.slice_mut(0..1),
            m.ids.slice_mut(0..u),
            m.weights.slice_mut(0..u),
            experts,
            hidden,
            used,
            m.router_done.as_mut_slice(),
        )?;
    } else {
        k::gemv(
            s,
            w.router.view(),
            x,
            m.router_logits.slice_mut(0..e),
            e,
            hidden,
        )?;
        k::gemv(
            s,
            w.shared_gate.view(),
            x,
            m.shared_gate.slice_mut(0..1),
            1,
            hidden,
        )?;
        k::moe_router_top_k(
            s,
            m.router_logits.slice(0..e),
            experts,
            m.ids.slice_mut(0..u),
            m.weights.slice_mut(0..u),
            1,
            experts,
            used,
        )?;
    }

    // Shared expert: a dense SwiGLU FFN of width shared_ff.
    k::fused_swiglu_gemv(
        s,
        w.gate_shexp.view(),
        w.up_shexp.view(),
        x,
        m.shared_act.slice_mut(0..shared_ff),
        shared_ff,
        hidden,
    )?;
    k::gemv(
        s,
        w.down_shexp.view(),
        m.shared_act.slice(0..shared_ff),
        m.shared_out.slice_mut(0..hidden),
        hidden,
        shared_ff,
    )?;

    routed_vec(s, w, c, m, x, 0, 1)?;
    if let Some(out) = combine {
        k::moe_epilogue(
            s,
            m.down_e.slice(0..u * hidden),
            m.weights.slice(0..u),
            m.shared_out.slice(0..hidden),
            m.shared_gate.slice(0..1),
            out,
            1,
            used,
            hidden as u32,
        )?;
    }
    Ok(())
}

/// The routed gate/up (SwiGLU) and down projections of `n` token rows from
/// `first_row` on through the MMQ vector kernels, the gate/up launch in
/// slices of eight rows; `x` holds those rows.
fn routed_vec(
    s: &Stream,
    w: &MoeWeights,
    c: &ModelConfig,
    m: &mut MoeScratch,
    x: DeviceSlice<'_, f32>,
    first_row: usize,
    n: usize,
) -> Result<(), LaunchError> {
    let hidden = c.hidden_size as usize;
    let (experts, used) = (c.expert_count, c.expert_used_count);
    let u = used as usize;
    let expert_ff = c.expert_ff_length as usize;
    let gate = w.gate_exps.format();
    if gate != w.up_exps.format() || !mmq_gated_type(gate) {
        return Err(unsupported(
            "the per-slot SwiGLU GEMV of these gate/up expert formats is not ported yet",
        ));
    }
    if !mmq_vec_type(w.down_exps.format()) {
        return Err(unsupported(
            "the per-slot GEMV of this down expert format is not ported yet",
        ));
    }
    let scratch_bytes = m.mmq.len();
    let mut row = 0;
    while row < n {
        let rows = GATED_VEC_MAX_ROWS.min(n - row);
        let (slot, count) = ((first_row + row) * u, rows * u);
        launched(
            k::mmq_moe_gated_vec(
                s,
                w.gate_exps.view(),
                w.up_exps.view(),
                x.range(row * hidden..(row + rows) * hidden),
                m.ids.slice(slot..slot + count),
                m.gate_e
                    .slice_mut(slot * expert_ff..(slot + count) * expert_ff),
                expert_ff as u32,
                hidden as u32,
                rows as u32,
                experts,
                used,
                m.mmq.as_mut_slice(),
                scratch_bytes,
            )?,
            "MoE gated expert projection failed",
        )?;
        row += rows;
    }
    // Each (token, slot) pair carries its own activation row, so the down
    // projection runs as independent rows with one expert ID each.
    let (slot, slots) = (first_row * u, n * u);
    launched(
        k::mmq_moe_vec(
            s,
            w.down_exps.view(),
            m.gate_e.slice(slot * expert_ff..(slot + slots) * expert_ff),
            m.ids.slice(slot..slot + slots),
            m.down_e.slice_mut(slot * hidden..(slot + slots) * hidden),
            hidden as u32,
            expert_ff as u32,
            slots as u32,
            experts,
            1,
            m.mmq.as_mut_slice(),
            scratch_bytes,
        )?,
        "MoE down expert projection failed",
    )
}

/// The routed FFN of `n` independent token rows with decode arithmetic
/// (`LaunchMoeFfn`), for batched decode and verification: `normed` holds the
/// rows' FFN-normed inputs, `gate` and `up` (at least `n * shared_ff` each)
/// take the shared expert's projections, `out` the FFN output rows. Rows run
/// in the order given, so a verified row equals the same row of a batched
/// decode step.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rows(
    s: &Stream,
    w: &MoeWeights,
    c: &ModelConfig,
    m: &mut MoeScratch,
    normed: DeviceSlice<'_, f32>,
    mut gate: DeviceSliceMut<'_, f32>,
    mut up: DeviceSliceMut<'_, f32>,
    out: DeviceSliceMut<'_, f32>,
    n: usize,
) -> Result<(), LaunchError> {
    if n == 0 || n > ROWS_MAX || n > m.batch {
        return Err(invalid("rows", "the MoE rows take one to 64 token rows"));
    }
    let hidden = c.hidden_size as usize;
    let (experts, used) = (c.expert_count, c.expert_used_count);
    let (e, u) = (experts as usize, used as usize);
    let shared_ff = c.expert_shared_ff_length as usize;
    let input = normed.range(0..n * hidden);
    project(
        s,
        &w.router,
        input,
        m.router_logits.slice_mut(0..n * e),
        n,
        e,
        hidden,
    )?;
    project(
        s,
        &w.shared_gate,
        input,
        m.shared_gate.slice_mut(0..n),
        n,
        1,
        hidden,
    )?;
    k::moe_router_top_k(
        s,
        m.router_logits.slice(0..n * e),
        experts,
        m.ids.slice_mut(0..n * u),
        m.weights.slice_mut(0..n * u),
        n as u32,
        experts,
        used,
    )?;

    project(
        s,
        &w.gate_shexp,
        input,
        gate.reborrow().range(0..n * shared_ff),
        n,
        shared_ff,
        hidden,
    )?;
    project(
        s,
        &w.up_shexp,
        input,
        up.reborrow().range(0..n * shared_ff),
        n,
        shared_ff,
        hidden,
    )?;
    k::batched_swiglu(
        s,
        gate.as_const().range(0..n * shared_ff),
        up.as_const().range(0..n * shared_ff),
        m.shared_act.slice_mut(0..n * shared_ff),
        n * shared_ff,
    )?;
    project(
        s,
        &w.down_shexp,
        m.shared_act.slice(0..n * shared_ff),
        m.shared_out.slice_mut(0..n * hidden),
        n,
        hidden,
        shared_ff,
    )?;

    routed_vec(s, w, c, m, input, 0, n)?;
    k::moe_epilogue(
        s,
        m.down_e.slice(0..n * u * hidden),
        m.weights.slice(0..n * u),
        m.shared_out.slice(0..n * hidden),
        m.shared_gate.slice(0..n),
        out.range(0..n * hidden),
        n as u32,
        used,
        hidden as u32,
    )
}

/// The routed FFN of a prompt chunk of `batch` rows (`ExecuteMoePrefillChunk`)
/// with the FFN norm already applied: `x.normed` holds the FP32 rows and,
/// when `norm_quantized`, the fused norm also wrote the tiled Q8_1
/// activation (`x.q8`), the BF16 staging copy (`x.scratch16`) and the tiled
/// BF16 copy (`x.moe.x_tiled`); otherwise the staging copy holds the BF16
/// rows (`batched_rms_norm_bf16`). Adds the FFN output to `x.hidden`.
pub(crate) fn prefill(
    s: &Stream,
    w: &MoeWeights,
    c: &ModelConfig,
    x: &mut Scratch,
    batch: usize,
    norm_quantized: bool,
) -> Result<(), LaunchError> {
    let hidden = c.hidden_size as usize;
    let (experts, used) = (c.expert_count, c.expert_used_count);
    let (e, u) = (experts as usize, used as usize);
    let expert_ff = c.expert_ff_length as usize;
    let slots = batch * u;
    let max_batch = x.moe.batch;
    if batch == 0 || batch > max_batch {
        return Err(invalid("batch", "the MoE prefill takes one chunk of rows"));
    }
    let max_slots = max_batch * u;
    let compact = routed_rows(max_slots, e);

    // Router logits [B, E] and the shared-expert gate [B] from F32 weights:
    // large chunks round the router to BF16 for the WMMA GEMM, the one-row
    // gate and small chunks take hipBLAS FP32.
    if w.router.format() != GgmlType::F32 || w.shared_gate.format() != GgmlType::F32 {
        return Err(unsupported(
            "the prefill router takes F32 router and shared-gate weights",
        ));
    }
    let weight_scratch_bytes = x.packed.len() * 4;
    {
        let m = &mut x.moe;
        if bf16_wmma_projection(batch, e, hidden, weight_scratch_bytes) {
            let mut converted = x
                .packed
                .slice_mut(0..e * hidden / 2)
                .cast::<Bf16>()
                .ok_or(invalid("packed", "the weight scratch holds BF16 values"))?;
            k::float_to_bf16(s, f32_weights(&w.router)?, converted.reborrow(), e * hidden)?;
            if norm_quantized {
                k::bf16_wmma_gemm_tiled(
                    s,
                    converted.as_const(),
                    m.x_tiled.slice(0..batch * hidden),
                    m.router_logits.slice_mut(0..batch * e),
                    batch,
                    e,
                    hidden,
                )?;
            } else {
                k::bf16_wmma_gemm(
                    s,
                    converted.as_const(),
                    x.normed.slice(0..batch * hidden),
                    m.router_logits.slice_mut(0..batch * e),
                    batch,
                    e,
                    hidden,
                )?;
            }
        } else {
            blas_gemm(
                s,
                &x.blas,
                &w.router,
                x.normed.slice(0..batch * hidden),
                m.router_logits.slice_mut(0..batch * e),
                batch,
                e,
                hidden,
            )?;
        }
        blas_gemm(
            s,
            &x.blas,
            &w.shared_gate,
            x.normed.slice(0..batch * hidden),
            m.shared_gate.slice_mut(0..batch),
            batch,
            1,
            hidden,
        )?;
        k::moe_router_top_k(
            s,
            m.router_logits.slice(0..batch * e),
            experts,
            m.ids.slice_mut(0..slots),
            m.weights.slice_mut(0..slots),
            batch as u32,
            experts,
            used,
        )?;
    }

    let gate_type = routed_weight_type(w.gate_exps.format(), hidden);
    let up_type = routed_weight_type(w.up_exps.format(), hidden);
    let down_type = routed_weight_type(w.down_exps.format(), expert_ff);
    let (Some(gate_type), Some(up_type), Some(down_type)) = (gate_type, up_type, down_type) else {
        return Err(unsupported(
            "the BF16 grouped and per-slot expert routes of these formats are not ported yet",
        ));
    };
    // The per-expert counts come down now, so the shared expert below hides
    // the copy.
    {
        let m = &mut x.moe;
        k::expert_counts(
            s,
            m.ids.slice(0..slots),
            m.counts.view_mut(),
            batch as u32,
            experts,
            used,
        )?;
        s.download_async(m.counts.view().values(), &mut m.host_counts)?;
    }

    shared_expert(s, w, c, x, batch, norm_quantized)?;

    // The tile map from the real bucket sizes: one entry per (expert, token
    // tile) of its 16-padded bucket, `expert | tile << 16`. Large chunks
    // whose gate/up share a format pair them over a second, mixed map:
    // 128-row tiles for experts with more than 64 rows, one 64-row tile for
    // the others.
    let pair = batch >= sizes::ROUTED_PAIR_MIN_TOKENS
        && gate_type == up_type
        && sizes::routed_gated_f16_type(gate_type.id());
    let m = &mut x.moe;
    let counts: Vec<u32> = m.host_counts.host()?.to_vec();
    let tile_rows = sizes::routed_tile_rows(slots, e);
    let (pair_rows, narrow_rows) = (sizes::ROUTED_PAIR_ROWS, sizes::ROUTED_NARROW_ROWS);
    let padded = |count: u32| count.div_ceil(16) * 16;
    let mut map: Vec<u32> = Vec::new();
    for (expert, &count) in (0u32..).zip(&counts) {
        let mut j = 0;
        while j * tile_rows < padded(count) {
            map.push(expert | (j << 16));
            j += 1;
        }
    }
    let n_tiles = map.len();
    if pair {
        // Wide tiles first, the narrow ones after them.
        for (expert, &count) in (0u32..).zip(&counts) {
            let rows = padded(count);
            if rows > narrow_rows {
                map.extend((0..rows.div_ceil(pair_rows)).map(|j| expert | (j << 16)));
            }
        }
    }
    let n_pair = map.len() - n_tiles;
    if pair {
        for (expert, &count) in (0u32..).zip(&counts) {
            let rows = padded(count);
            if rows != 0 && rows <= narrow_rows {
                map.push(expert);
            }
        }
    }
    let n_narrow = map.len() - n_tiles - n_pair;
    m.host_tiles
        .host_mut()?
        .get_mut(..map.len())
        .ok_or(invalid("tiles", "the routed tile map fits its capacity"))?
        .copy_from_slice(&map);
    s.upload_indices_async(&mut m.host_tiles, n_tiles + n_pair + n_narrow, &mut m.tiles)?;
    k::routed_compact(
        s,
        m.ids.slice(0..slots),
        m.counts.view(),
        m.bounds.view_mut(),
        m.cursors.as_mut_slice(),
        m.rows_token.view_mut(),
        m.rows_slot.view_mut(),
        batch as u32,
        used,
        experts,
        max_slots as u32,
    )?;

    // Gate/up and SwiGLU: the activation lands as F16 rows in up_e.
    k::narrow_f16(
        s,
        x.normed.slice(0..batch * hidden),
        m.x_half.slice_mut(0..batch * hidden),
        batch * hidden,
    )?;
    let x_half = m.x_half.slice(0..max_batch * hidden);
    {
        let mut act_half = m
            .up_e
            .slice_mut(0..max_slots * expert_ff / 2)
            .cast::<F16>()
            .ok_or(invalid("up_e", "the expert scratch holds F16 values"))?;
        let bounds = m.bounds.view();
        let rows_token = m.rows_token.view();
        let rows_slot = m.rows_slot.view();
        if pair {
            let paired = |start: usize, count: usize, rows: u32, out: DeviceSliceMut<'_, F16>| {
                if count == 0 {
                    return Ok(());
                }
                launched(
                    k::routed_gated_f16_gemm(
                        s,
                        w.gate_exps.view(),
                        w.up_exps.view(),
                        x_half,
                        m.tiles.slice(start..start + count),
                        count as u32,
                        rows,
                        bounds,
                        rows_token,
                        rows_slot,
                        out,
                        expert_ff,
                        hidden,
                        experts,
                        compact,
                        max_batch,
                        max_slots,
                    )?,
                    "routed F16 expert gate/up GEMM failed",
                )
            };
            paired(n_tiles, n_pair, pair_rows, act_half.reborrow())?;
            paired(n_tiles + n_pair, n_narrow, narrow_rows, act_half.reborrow())?;
        } else {
            launched(
                k::routed_f16_gemm(
                    s,
                    w.gate_exps.view(),
                    x_half,
                    m.tiles.slice(0..n_tiles),
                    n_tiles as u32,
                    tile_rows,
                    bounds,
                    rows_token,
                    rows_slot,
                    m.gate_e.slice_mut(0..max_slots * expert_ff),
                    expert_ff,
                    hidden,
                    experts,
                    compact,
                    max_batch,
                    max_slots,
                )?,
                "routed F16 expert gate/up GEMM failed",
            )?;
            // up applies SwiGLU against the gate on its F16 output.
            launched(
                k::routed_f16_gemm_half(
                    s,
                    w.up_exps.view(),
                    x_half,
                    m.tiles.slice(0..n_tiles),
                    n_tiles as u32,
                    tile_rows,
                    bounds,
                    rows_token,
                    rows_slot,
                    Some(m.gate_e.slice(0..max_slots * expert_ff)),
                    act_half.reborrow(),
                    expert_ff,
                    hidden,
                    experts,
                    compact,
                    max_batch,
                    max_slots,
                )?,
                "routed F16 expert gate/up GEMM failed",
            )?;
        }
    }

    // The down projection reads one activation row per (token, slot). A
    // paired layer sends it over the mixed map it built for gate/up where
    // the format has 64- and 128-row tiles.
    let down_mixed = pair && sizes::routed_f16_wide_tiles(down_type.id());
    {
        let act_half = m
            .up_e
            .slice(0..max_slots * expert_ff / 2)
            .cast::<F16>()
            .ok_or(invalid("up_e", "the expert scratch holds F16 values"))?;
        let mut down_half = m
            .down_e
            .slice_mut(0..max_slots * hidden / 2)
            .cast::<F16>()
            .ok_or(invalid("down_e", "the expert scratch holds F16 values"))?;
        let bounds = m.bounds.view();
        let rows_slot = m.rows_slot.view();
        let mut down = |start: usize, count: usize, rows: u32| {
            if count == 0 {
                return Ok(());
            }
            launched(
                k::routed_f16_gemm_half(
                    s,
                    w.down_exps.view(),
                    act_half,
                    m.tiles.slice(start..start + count),
                    count as u32,
                    rows,
                    bounds,
                    rows_slot,
                    rows_slot,
                    None,
                    down_half.reborrow(),
                    hidden,
                    expert_ff,
                    experts,
                    compact,
                    max_slots,
                    max_slots,
                )?,
                "routed F16 expert down GEMM failed",
            )
        };
        if down_mixed {
            down(n_tiles, n_pair, pair_rows)?;
            down(n_tiles + n_pair, n_narrow, narrow_rows)?;
        } else {
            down(0, n_tiles, tile_rows)?;
        }
    }
    let down_half = m
        .down_e
        .slice(0..slots * hidden / 2)
        .cast::<F16>()
        .ok_or(invalid("down_e", "the expert scratch holds F16 values"))?;
    k::moe_epilogue_vec4_f16(
        s,
        down_half,
        m.weights.slice(0..slots),
        m.shared_out.slice(0..batch * hidden),
        m.shared_gate.slice(0..batch),
        1,
        x.ffn_out.slice_mut(0..batch * hidden),
        batch as u32,
        used,
        hidden as u32,
    )?;
    k::batched_residual_add(
        s,
        x.ffn_out.slice(0..batch * hidden),
        x.hidden.slice_mut(0..batch * hidden),
        batch,
        hidden,
    )
}

/// The shared expert of a prompt chunk through the dense prefill GEMMs
/// (width `shared_ff`) into `x.moe.shared_out`. The Q8_1 routes read the
/// tiled activation, which the fused FFN norm wrote, else quantized here
/// from the BF16 staging copy.
fn shared_expert(
    s: &Stream,
    w: &MoeWeights,
    c: &ModelConfig,
    x: &mut Scratch,
    batch: usize,
    norm_quantized: bool,
) -> Result<(), LaunchError> {
    let hidden = c.hidden_size as usize;
    let shared_ff = c.expert_shared_ff_length as usize;
    if !norm_quantized {
        let staging = bf16_staging(&mut x.scratch16, batch * hidden)?;
        k::quantize_q8_1(
            s,
            staging.as_const(),
            x.q8.slice_mut(0..sizes::q8_activation_bytes(batch, hidden)),
            batch,
            hidden,
        )?;
    }
    let input = x.normed.slice(0..batch * hidden);
    gemm(
        s,
        &x.blas,
        &x.q8,
        batch,
        &w.gate_shexp,
        input,
        x.ffn_gate.slice_mut(0..batch * shared_ff),
        shared_ff,
        hidden,
    )?;
    gemm(
        s,
        &x.blas,
        &x.q8,
        batch,
        &w.up_shexp,
        input,
        x.ffn_up.slice_mut(0..batch * shared_ff),
        shared_ff,
        hidden,
    )?;
    {
        let staging = bf16_staging(&mut x.scratch16, batch * shared_ff)?;
        k::batched_swiglu_bf16(
            s,
            x.ffn_gate.slice(0..batch * shared_ff),
            x.ffn_up.slice(0..batch * shared_ff),
            x.ffn_act.slice_mut(0..batch * shared_ff),
            staging,
            batch * shared_ff,
        )?;
    }
    if reads_q8_activation(w.down_shexp.format()) {
        let staging = bf16_staging(&mut x.scratch16, batch * shared_ff)?;
        k::quantize_q8_1(
            s,
            staging.as_const(),
            x.q8.slice_mut(0..sizes::q8_activation_bytes(batch, shared_ff)),
            batch,
            shared_ff,
        )?;
    }
    gemm(
        s,
        &x.blas,
        &x.q8,
        batch,
        &w.down_shexp,
        x.ffn_act.slice(0..batch * shared_ff),
        x.moe.shared_out.slice_mut(0..batch * hidden),
        hidden,
        shared_ff,
    )
}

/// The first `len` values of the BF16 staging scratch (the C++
/// `d_scratch_bf16`, an F16 buffer here).
pub(crate) fn bf16_staging(
    scratch16: &mut DeviceBuffer<F16>,
    len: usize,
) -> Result<DeviceSliceMut<'_, Bf16>, LaunchError> {
    scratch16.slice_mut(0..len).cast::<Bf16>().ok_or(invalid(
        "scratch16",
        "the staging scratch holds BF16 values",
    ))
}
