//! The Gemma 4 graph on device-resident weights (`hip/executor.cpp`):
//! sessions' KV caches, the forward pass over one or several sessions'
//! rows, and the frontier copies the engine keeps. Scratch is shared by
//! every session; calls are serialized on the model's stream.
//!
//! The kernels and their order are the C++'s, with one deviation that keeps
//! every sum: routing writes its group table and expert counts in launches
//! of their own (the kernels crate's `expert_groups` and `expert_counts`).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use gufo_gemma4_kernels as k;
use gufo_hip::blas::{Blas, Op};
use gufo_hip::{
    DeviceBuffer, DeviceCopy, DeviceIndices, DeviceSlice, DeviceSliceMut, DeviceTensor, Event, F16,
    GgmlType, HipError, IndexSpec, Launch, LaunchError, Stream,
};

use gufo_core::sampling::LogitCandidate;

use crate::config::Config;
use crate::device_model::{
    DeviceLayer, DeviceModel, ExpertFormat, PrefillLayout, Weight, expert_format_of,
    half_prefill_format, prefill_layout_supports, wmma_gemv_supports,
};
use crate::mtp_sampling::MtpCandidateLogits;

/// Rows the split (decode and verification) attention and the small-batch
/// projections take; more rows are a prefill (`kSplitRows`).
pub const SPLIT_ROWS: u32 = k::sizes::SPLIT_ROWS as u32;
/// Longest draft chain one cycle may request (`kMaxDraftTokens`).
pub const MAX_DRAFT_TOKENS: u32 = 15;
/// Sessions whose drafter chains share one forward per step
/// (`kMaxDraftSessions`).
pub const MAX_DRAFT_SESSIONS: u32 = 8;
/// Rows one expert group holds (`kMaxGroupSlots`).
const MAX_GROUP_SLOTS: u32 = k::sizes::MAX_GROUP_SLOTS as u32;
/// Token rows per tile of the routed binary16 GEMMs (`kRoutedTileRows`,
/// `kRoutedDownRows`, `kRoutedKQuantTileRows`).
const ROUTED_TILE_ROWS: u32 = 48;
const ROUTED_DOWN_ROWS: u32 = 64;
const ROUTED_KQUANT_TILE_ROWS: u32 = k::sizes::ROUTED_HALF_TILE_ROWS as u32;
/// Routing index tables kept per row count before they are dropped.
const ROUTING_TABLES: usize = 32;
/// Most router hidden width the routing kernel holds (`kMaxRouterHidden`).
const MAX_ROUTER_HIDDEN: u32 = k::sizes::MAX_ROUTER_HIDDEN as u32;

pub(crate) fn hip(error: HipError) -> String {
    error.to_string()
}

pub(crate) fn launch(error: LaunchError) -> String {
    error.to_string()
}

/// A launch the C++ makes unconditionally (or throws when unsupported).
fn launched(result: Result<Launch, LaunchError>, what: &str) -> Result<(), String> {
    match result.map_err(launch)? {
        Launch::Launched => Ok(()),
        Launch::Unsupported => Err(format!("gemma4 {what} shape unsupported")),
    }
}

/// A launch whose C++ launcher returns false for shapes or formats it does
/// not take: `Ok(false)` queued nothing, as when a contract precondition or
/// format list (the same conditions) refuses the launch.
fn attempt(result: Result<Launch, LaunchError>) -> Result<bool, String> {
    match result {
        Ok(Launch::Launched) => Ok(true),
        Ok(Launch::Unsupported) | Err(LaunchError::Precondition { .. }) => Ok(false),
        Err(LaunchError::InvalidArgument { reason, .. }) if reason == UNSUPPORTED_FORMAT => {
            Ok(false)
        }
        Err(error) => Err(launch(error)),
    }
}

/// The reason gufo-hip gives a launcher's format refusal.
const UNSUPPORTED_FORMAT: &str = "storage format not supported by this launcher";

fn unit(result: Result<(), LaunchError>) -> Result<(), String> {
    result.map_err(launch)
}

/// Ring slots for sliding layers given the largest forward (`RingSlots`).
pub fn ring_slots(config: &Config, max_rows: u32) -> u32 {
    (config.sliding_window + max_rows).div_ceil(256) * 256
}

/// One session's attention state (`KvCache`). Global layers keep every
/// position in a token-major binary16 cache; sliding layers keep a ring of
/// `ring` slots, at least window + the largest forward, so no forward
/// overwrites a key still inside a live window and rollback only moves the
/// frontier.
pub struct KvCache {
    pub max_context: u32,
    pub ring: u32,
    pub bytes: usize,
    /// Per layer, binary16: whole keys, or their rotated dims where keys are
    /// derived from V (`key_widths`).
    pub k: Vec<DeviceBuffer<F16>>,
    pub v: Vec<DeviceBuffer<F16>>,
    /// Slots of each layer's cache: `ring` for sliding layers, the context
    /// for global ones.
    pub slots: Vec<u32>,
    /// Target post-norm hidden state of the frontier token (MTP draft
    /// input).
    pub hidden: DeviceBuffer<f32>,
    /// `SPLIT_ROWS` slots of the same for lookahead rows a prefill computed
    /// past the frontier.
    pub ahead_hidden: DeviceBuffer<f32>,
    /// Logit row of the frontier token while the owning session keeps it on
    /// the device.
    pub logits: DeviceBuffer<f32>,
    /// Drafter head rows (`load_draft_table`) when the head is ordered.
    pub draft_table: Option<DeviceBuffer<u8>>,
    /// Positions `[written_from, written_to)` span every row forwards and
    /// key moves wrote since the owner last reset them.
    pub written_from: AtomicU32,
    pub written_to: AtomicU32,
}

/// Image soft-token rows of one forward: rows `[row, row + count)` take
/// `embedding` (`[count][hidden]` f32) unscaled in place of their token
/// embeddings and attend to each other bidirectionally in sliding layers
/// (`ImageRows`).
pub struct ImageRows {
    pub row: u32,
    pub count: u32,
    pub embedding: Arc<DeviceTensor<f32>>,
}

/// Consecutive rows of one session in a batched forward (`Segment`). The
/// last `siblings` rows are alternatives to the drafts at depths
/// 1..siblings: sibling i sits at position first_position + 1 + i, attends
/// what the chain's draft there attends, and keeps its K/V at the spare key
/// first_position + rows - siblings + i.
pub struct Segment<'a> {
    pub cache: &'a mut KvCache,
    pub first_position: u32,
    pub rows: u32,
    pub siblings: u32,
}

/// Which buffer holds a projection input staged for the prefill GEMMs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Staged {
    /// Small-batch widths read the FP32 rows.
    None,
    /// The tiled Q8_1 rows in `stage.q8`.
    Q8,
    /// Binary16 rows in `stage.x_half`.
    Half,
    /// The FP32 input buffer itself holds binary16 rows (GeGLU of a fused
    /// gate/up GEMM written into the free gate rows).
    Input,
}

/// The staging buffers of projection inputs.
struct Stage {
    q8: DeviceBuffer<u8>,
    x_half: DeviceBuffer<F16>,
}

/// Model-wide facts the projection choices read.
#[derive(Clone, Copy)]
struct Env<'a> {
    stream: &'a Stream,
    half_prefill: bool,
    has_draft: bool,
}

/// Decode projections on the routed GEMV's dense mode (`DenseGemvFormat`).
fn dense_gemv_format(ty: GgmlType, k: u32) -> Option<ExpertFormat> {
    if ty == GgmlType::Q5_0 {
        return Some(ExpertFormat::Q5_0);
    }
    if ty == GgmlType::Q4_0 && !k.is_multiple_of(256) {
        return Some(ExpertFormat::Q4_0);
    }
    None
}

/// Formats of the Gemma GEMV (`GemvFormatOf`).
fn gemv_format(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q4_0 | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0
    )
}

/// The formats of the dense binary16 prefill GEMM (`DenseHalfFormat`).
fn dense_half_format(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K
    )
}

/// `DenseF16Plan` of a Q8_0 projection of a Gemma width (`HalfPlan`):
/// kAuto 0, kRowGroups8 1, kStagedRowGroups8 2, kStagedRowGroups8Walk7 3.
fn half_plan(m: u32, k_: u32) -> u32 {
    if m == 5376 && k_ >= 8192 {
        return 3;
    }
    if m == 2816 && k_ == 2112 { 1 } else { 0 }
}

/// Whether a prefill of `rows` reads w's prefill copy (`PackedPrefill`).
fn packed_prefill(w: &Weight, rows: u32) -> bool {
    match w.prefill_layout {
        PrefillLayout::Compact => true,
        PrefillLayout::Decoded => rows >= 512,
        PrefillLayout::None => false,
    }
}

/// `WmmaProjection`.
fn wmma_projection(env: Env<'_>, w: &Weight, rows: u32) -> bool {
    w.ty != GgmlType::F16
        && (1..=SPLIT_ROWS).contains(&rows)
        && rows <= k::sizes::WMMA_GEMV_ROWS as u32
        && dense_gemv_format(w.ty, w.cols).is_none()
        && w.cols >= 2048
        && (env.has_draft || rows >= 6)
        && wmma_gemv_supports(w.ty, w.rows, w.cols)
}

/// The staged binary16 input of a prefill GEMM.
fn staged_half<'a>(
    stage: &'a Stage,
    x: DeviceSlice<'a, f32>,
    xq: Staged,
    count: usize,
) -> Result<DeviceSlice<'a, F16>, String> {
    match xq {
        Staged::Half => Ok(stage.x_half.slice(0..count)),
        Staged::Input => x
            .cast::<F16>()
            .map(|h| h.range(0..count))
            .ok_or_else(|| "gemma4 binary16 input alignment".to_string()),
        Staged::None | Staged::Q8 => Err("gemma4 prefill GEMM without binary16 input".into()),
    }
}

/// Launches a packed prefill GEMM (`LaunchPackedPrefillGemm`).
fn packed_prefill_gemm(
    s: &Stream,
    w: &Weight,
    x: DeviceSlice<'_, F16>,
    out: Option<DeviceSliceMut<'_, f32>>,
    out_half: Option<DeviceSliceMut<'_, F16>>,
    rows: u32,
    geglu: bool,
) -> Result<bool, String> {
    if !prefill_layout_supports(w.prefill_format, w.prefill_layout, w.cols) {
        return Ok(false);
    }
    let p = w.prefill_bytes()?;
    let (m, k_) = (w.rows, w.cols);
    let format = w.prefill_format as u32;
    match (w.prefill_layout, w.prefill_format, out, out_half) {
        (PrefillLayout::Decoded, _, Some(out), None) if !geglu => {
            let p = p.cast::<F16>().ok_or("prefill copy alignment")?;
            attempt(k::packed_prefill_gemm_decoded(
                s, format, p, x, out, rows, m, k_, 0,
            ))
        }
        (PrefillLayout::Decoded, _, None, Some(out)) if geglu => {
            let p = p.cast::<F16>().ok_or("prefill copy alignment")?;
            attempt(k::packed_prefill_gemm_decoded_geglu(
                s, format, p, x, out, rows, m, k_, 0,
            ))
        }
        (PrefillLayout::Compact, ExpertFormat::Q4K, Some(out), None) if !geglu => {
            attempt(k::packed_prefill_gemm_compact(s, p, x, out, rows, m, k_, 0))
        }
        (PrefillLayout::Compact, ExpertFormat::Q4K, None, Some(out)) if geglu => attempt(
            k::packed_prefill_gemm_compact_geglu(s, p, x, out, rows, m, k_, 0),
        ),
        (PrefillLayout::Compact, _, Some(out), None) if !geglu => attempt(
            k::packed_prefill_gemm_compact_112(s, format, p, x, out, rows, m, k_, 0),
        ),
        (PrefillLayout::Compact, _, None, Some(out)) if geglu => attempt(
            k::packed_prefill_gemm_compact_112_geglu(s, format, p, x, out, rows, m, k_, 0),
        ),
        _ => Ok(false),
    }
}

/// `y[rows][w.rows] = x[rows][w.cols] * W^T` (`Executor::Project`). `xq` is
/// x staged for the prefill GEMMs past the small-batch width; `packed` says
/// `wmma` already holds x's WMMA activation pack.
#[allow(clippy::too_many_arguments)]
fn project(
    env: Env<'_>,
    stage: &mut Stage,
    wmma: &mut DeviceBuffer<u8>,
    w: &Weight,
    x: DeviceSlice<'_, f32>,
    xq: Staged,
    rows: u32,
    mut y: DeviceSliceMut<'_, f32>,
    packed: bool,
) -> Result<(), String> {
    let s = env.stream;
    if w.ty == GgmlType::F16 {
        return project_half(env, stage, w, x, xq, rows, y);
    }
    let (m, k_) = (w.rows, w.cols);
    let (mu, ku, ru) = (m as usize, k_ as usize, rows as usize);
    if rows > SPLIT_ROWS {
        let mut xq = xq;
        if env.half_prefill {
            if packed_prefill(w, rows) {
                let xh = staged_half(stage, x, xq, ru * ku)?;
                if packed_prefill_gemm(s, w, xh, Some(y.reborrow()), None, rows, false)? {
                    return Ok(());
                }
            }
            // Binary16 activations: Q8_0, Q4_0 and K-quant weights decode to
            // binary16 in the WMMA GEMMs; other formats keep the W8A8 route
            // on their own Q8_1 rows. The 26B-A4B's Q8_0 attention outputs
            // fill the WGPs better on Gemma's 128-row blocks, with the same
            // sums.
            if w.ty == GgmlType::Q8_0 && m == 2816 && k_ >= 4096 {
                let xh = staged_half(stage, x, xq, ru * ku)?;
                if attempt(k::dense_half_gemm(
                    s,
                    w.q()?,
                    xh,
                    y.reborrow(),
                    rows,
                    m,
                    k_,
                    w.grouped,
                ))? {
                    return Ok(());
                }
            }
            if w.ty == GgmlType::Q8_0 {
                let xh = staged_half(stage, x, xq, ru * ku)?;
                let plan = half_plan(m, k_);
                if attempt(k::dense_f16_gemm(
                    s,
                    w.q()?,
                    xh,
                    y.reborrow(),
                    ru,
                    mu,
                    ku,
                    plan,
                    w.grouped,
                ))? {
                    return Ok(());
                }
            }
            if dense_half_format(w.ty) {
                let xh = staged_half(stage, x, xq, ru * ku)?;
                if attempt(k::dense_half_gemm(
                    s,
                    w.q()?,
                    xh,
                    y.reborrow(),
                    rows,
                    m,
                    k_,
                    w.grouped,
                ))? {
                    return Ok(());
                }
            }
            unit(k::quantize_q8_1_from_f32(
                s,
                x,
                stage.q8.as_mut_slice(),
                ru,
                ku,
            ))?;
            xq = Staged::Q8;
        }
        if w.grouped {
            return Err("gemma4 grouped weights reached a row-order GEMM".into());
        }
        if xq != Staged::Q8 {
            return Err("gemma4 prefill GEMM without Q8_1 input".into());
        }
        return unit(k::batched_quant_gemm_q8_1(
            s,
            w.q()?,
            stage.q8.as_slice(),
            y,
            ru,
            mu,
            ku,
        ));
    }
    // One kernel at every decode width, so verification rows round like
    // single tokens.
    if dense_gemv_format(w.ty, k_).is_some()
        && attempt(k::dense_gemv(s, w.q()?, x, y.reborrow(), rows, m, k_))?
    {
        return Ok(());
    }
    // Target projections take the WMMA kernel at every width when the
    // drafter is loaded, and batched autoregressive rows from six on.
    if wmma_projection(env, w, rows) {
        let done = if packed {
            attempt(k::wmma_gemv_packed(
                s,
                w.q()?,
                y.reborrow(),
                rows,
                m,
                k_,
                wmma.as_mut_slice(),
                w.grouped,
            ))?
        } else {
            attempt(k::wmma_gemv(
                s,
                w.q()?,
                x,
                y.reborrow(),
                rows,
                m,
                k_,
                wmma.as_mut_slice(),
                w.grouped,
            ))?
        };
        if done {
            return Ok(());
        }
    }
    // The device model groups only weights the WMMA projections take.
    if w.grouped {
        return Err("gemma4 grouped weights reached a row-order GEMV".into());
    }
    if rows > 1 {
        // Gemma's K-quant shapes lie outside the ggml dispatch's tuning; the
        // double-stage configuration is faster on them (the vocabulary head
        // keeps the default). All of them round like the one-row twins.
        let double_stage = if w.ty == GgmlType::Q4_0 {
            rows > 8
        } else {
            m < 65536
        };
        if double_stage
            && attempt(k::kquant_small_batch_double_stage(
                s,
                w.q()?,
                x,
                y.reborrow(),
                ru,
                mu,
                ku,
            ))?
        {
            return Ok(());
        }
        // Q8_0 past eight rows reads its weights once instead of per eight
        // rows.
        if w.ty == GgmlType::Q8_0
            && attempt(k::q8_0_small_batch_wide(
                s,
                w.q()?,
                x,
                y.reborrow(),
                ru,
                mu,
                ku,
            ))?
        {
            return Ok(());
        }
        return unit(k::batched_quant_gemm_f32(s, w.q()?, x, y, ru, mu, ku));
    }
    if env.has_draft {
        // Speculation verifies with the small-batch kernel, and single
        // tokens must round identically: its bit-identical one-row twins.
        let double_stage = w.ty == GgmlType::Q4K || (w.ty == GgmlType::Q5K && m > 4096);
        if double_stage
            && attempt(k::kquant_small_batch_double_stage(
                s,
                w.q()?,
                x,
                y.reborrow(),
                1,
                mu,
                ku,
            ))?
        {
            return Ok(());
        }
        if m <= 4096 && w.ty != GgmlType::Q4_0 {
            return unit(k::gemv(s, w.q()?, x, y, mu, ku));
        }
        return unit(k::batched_quant_gemm_f32(s, w.q()?, x, y, 1, mu, ku));
    }
    // Autoregressive decode: the Gemma GEMV serves every K-quant and Q4_0
    // projection, and Q8_0 past 2112 outputs and 1024 inputs.
    let gemma_gemv = w.ty != GgmlType::Q8_0 || (m > 2112 && k_ > 1024);
    if gemv_format(w.ty)
        && gemma_gemv
        && attempt(k::kquant_gemv(s, w.q()?, x, y.reborrow(), m, k_))?
    {
        return Ok(());
    }
    unit(k::gemv(s, w.q()?, x, y, mu, ku))
}

/// `Project` for binary16 weights: the batch-invariant small-batch GEMV up
/// to `SPLIT_ROWS` rows, the binary16 WMMA GEMM past that
/// (`ProjectHalf`).
fn project_half(
    env: Env<'_>,
    stage: &mut Stage,
    w: &Weight,
    x: DeviceSlice<'_, f32>,
    xq: Staged,
    rows: u32,
    y: DeviceSliceMut<'_, f32>,
) -> Result<(), String> {
    let s = env.stream;
    let (m, k_) = (w.rows, w.cols);
    if rows <= SPLIT_ROWS {
        return launched(
            k::half_gemv(s, w.f16()?, x, y, rows, m, k_),
            "binary16 projection",
        );
    }
    let count = rows as usize * k_ as usize;
    let mut xq = xq;
    if !env.half_prefill {
        unit(k::narrow_activations(
            s,
            x.range(0..count),
            stage.x_half.slice_mut(0..count),
            count,
        ))?;
        xq = Staged::Half;
    }
    let xh = staged_half(stage, x, xq, count)?;
    // Every Gemma projection is thousands of outputs wide: 128 x 128 tiles
    // from 128 rows.
    launched(
        k::unquantized_f16_gemm(
            s,
            w.q()?,
            xh,
            y,
            rows as usize,
            m as usize,
            k_ as usize,
            128,
        ),
        "binary16 prefill",
    )
}

/// Stages x for the prefill GEMMs (`Executor::Quantize`): binary16 rows on
/// models whose prefill runs with binary16 activations, else the tiled
/// Q8_1 encoding; nothing at small-batch widths.
fn quantize(
    env: Env<'_>,
    stage: &mut Stage,
    x: DeviceSlice<'_, f32>,
    rows: u32,
    cols: u32,
) -> Result<Staged, String> {
    if rows <= SPLIT_ROWS {
        return Ok(Staged::None);
    }
    let count = rows as usize * cols as usize;
    if env.half_prefill {
        unit(k::narrow_activations(
            env.stream,
            x.range(0..count),
            stage.x_half.slice_mut(0..count),
            count,
        ))?;
        return Ok(Staged::Half);
    }
    unit(k::quantize_q8_1_from_f32(
        env.stream,
        x,
        stage.q8.as_mut_slice(),
        rows as usize,
        cols as usize,
    ))?;
    Ok(Staged::Q8)
}

/// Whether layer l's experts take the binary16 prefill route
/// (`HalfExperts`).
fn routed_half_type(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q8_0 | GgmlType::Q5_1
    )
}

/// Gate/up formats the Gemma routed binary16 GEMM takes (`OwnGateUp`).
fn own_gate_up(format: Option<ExpertFormat>) -> bool {
    matches!(
        format,
        Some(
            ExpertFormat::Q4K
                | ExpertFormat::Q5K
                | ExpertFormat::Q6K
                | ExpertFormat::Q8_0
                | ExpertFormat::F16
                | ExpertFormat::Q4_0
        )
    )
}

/// Down formats the Gemma routed binary16 GEMM takes (`OwnDown`).
fn own_down(format: Option<ExpertFormat>) -> bool {
    matches!(
        format,
        Some(
            ExpertFormat::Q5_1
                | ExpertFormat::Q8_0
                | ExpertFormat::F16
                | ExpertFormat::Q4_0
                | ExpertFormat::Q5_0
        )
    )
}

fn half_experts(l: &DeviceLayer, c: &Config) -> bool {
    (routed_half_type(l.gate_up_exps.ty) || own_gate_up(expert_format_of(l.gate_up_exps.ty)))
        && (routed_half_type(l.down_exps.ty) || own_down(expert_format_of(l.down_exps.ty)))
        && c.hidden_size.is_multiple_of(256)
        && c.expert_ffn_size.is_multiple_of(64)
}

/// Index buffers of the prefill expert route for one row count.
struct Routing {
    counts: DeviceIndices,
    pad_bounds: DeviceIndices,
    rows_token: DeviceIndices,
    rows_slot: DeviceIndices,
    tiles_a: DeviceIndices,
    tiles_b: DeviceIndices,
}

/// Routed expert scratch (mixture-of-experts models).
struct MoeScratch {
    /// Binary16 expert input rows (prefill), written by the attention norm.
    x_half: DeviceBuffer<F16>,
    /// FP32 expert input rows.
    h: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    ids: DeviceIndices,
    weights: DeviceBuffer<f32>,
    /// gate/up rows: FP32, or binary16 in prefill.
    gu: DeviceBuffer<u8>,
    /// GeGLU rows: FP32, or binary16 in prefill.
    act: DeviceBuffer<u8>,
    out: DeviceBuffer<f32>,
    cursors: DeviceBuffer<i32>,
    /// Group tables by slot count.
    groups: HashMap<u32, DeviceIndices>,
    /// Prefill routes by row count.
    routing: HashMap<u32, Routing>,
}

/// Drafter scratch: one row per session drafting in the same step.
struct DraftScratch {
    tokens: DeviceIndices,
    embed: DeviceBuffer<f32>,
    concat: DeviceBuffer<f32>,
    x: DeviceBuffer<f32>,
    h: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    o: DeviceBuffer<f32>,
    /// Gate then up rows, or `[gate | up]` rows.
    gate: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    /// The drafter's state projected back to the target width.
    next: DeviceBuffer<f32>,
    /// Per row: the top-64 selection's workspace and scores.
    ids: Vec<DeviceIndices>,
    scratch_ids: Vec<DeviceBuffer<u32>>,
    scores: Vec<DeviceBuffer<f32>>,
}

/// `logit_candidates`' workspace and outputs.
struct CandidateScratch {
    work: DeviceBuffer<u8>,
    counts: DeviceBuffer<u32>,
    out: DeviceBuffer<u32>,
    extra: DeviceIndices,
}

/// Shared forward scratch.
struct Scratch {
    tokens: DeviceIndices,
    key_ends: DeviceBuffer<u32>,
    x: DeviceBuffer<f32>,
    h: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    /// K and V rows, or the `[q | k | v]` rows of a fused projection.
    kv: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    o: DeviceBuffer<f32>,
    /// Gate then up rows, or the `[gate | up]` rows of a fused projection.
    gate: DeviceBuffer<f32>,
    /// GeGLU rows of a fused gate/up projection.
    act: DeviceBuffer<f32>,
    hsel: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    partials: DeviceBuffer<f32>,
    stage: Stage,
    /// Activation pack and partial sums of the WMMA projections.
    wmma: DeviceBuffer<u8>,
    argmax: DeviceBuffer<u32>,
}

/// Runs the Gemma 4 graph (`Executor`).
pub struct Executor {
    stream: Stream,
    blas: Option<Blas>,
    max_rows: u32,
    max_logit_rows: u32,
    max_context: u32,
    ring: u32,
    key_widths: Vec<u32>,
    half_prefill: bool,
    has_draft: bool,
    vocab: u32,
    s: Scratch,
    moe: Option<MoeScratch>,
    draft: Option<DraftScratch>,
    candidates: Option<CandidateScratch>,
    logits_host: Vec<f32>,
    scratch_bytes: usize,
    tap: Option<LayerTapSink>,
    /// Experts beside the dense MLP at grouped widths (MoE models): their
    /// stream and its fork and join events.
    expert_stream: Option<(Stream, Event, Event)>,
}

/// Receives, after each layer of every forward, that layer's residual rows
/// (`layer`, `[count][hidden]` on the host, `count`): the target features a
/// DFlash drafter reads (`TapSink`).
pub type LayerTapSink = Box<dyn FnMut(u32, &[f32], u32) + Send>;

fn bytes_of<T: DeviceCopy>(buffer: &DeviceBuffer<T>) -> usize {
    buffer.len() * size_of::<T>()
}

fn alloc<T: DeviceCopy>(stream: &Stream, len: usize) -> Result<DeviceBuffer<T>, String> {
    stream
        .alloc_persistent::<T>(len.max(1))
        .map_err(|e| format!("gemma4 scratch allocation failed ({len} elements): {e}"))
}

impl Executor {
    /// Values per K cache row: the whole K, or only the rotated dims where
    /// one projection feeds K and V (`KeyWidths`).
    pub fn key_widths(model: &DeviceModel) -> Vec<u32> {
        let c = &model.config;
        (0..c.num_layers)
            .map(|l| {
                let pairs = model.global_rope_pairs;
                let derived = !c.is_sliding(l)
                    && c.head_dim(l) == 512
                    && model.layers[l as usize].attn_v.is_empty()
                    && pairs.is_multiple_of(16)
                    && pairs as usize <= k::sizes::MAX_DERIVED_KEY_PAIRS;
                if derived {
                    c.kv_heads[l as usize] * 2 * pairs
                } else {
                    c.kv_dim(l)
                }
            })
            .collect()
    }

    /// `max_rows` bounds one forward (prefill chunk); `max_logit_rows` how
    /// many rows of one forward may request logits.
    pub fn new(
        model: &DeviceModel,
        stream: Stream,
        max_rows: u32,
        max_logit_rows: u32,
        max_context: u32,
    ) -> Result<Executor, String> {
        let c = &model.config;
        if c.has_experts()
            && (!c.hidden_size.is_multiple_of(128)
                || c.hidden_size > MAX_ROUTER_HIDDEN
                || !c.num_experts.is_multiple_of(8))
        {
            return Err(format!(
                "gemma4 expert routing needs a hidden width that is a multiple of 128 up to \
                 {MAX_ROUTER_HIDDEN} and experts in groups of 8"
            ));
        }
        let ring = ring_slots(c, max_rows);
        let rows = max_rows as usize;
        let logit_rows = max_logit_rows as usize;
        let d = c.hidden_size as usize;
        let max_q = (0..c.num_layers).map(|l| c.q_dim(l)).max().unwrap_or(0) as usize;
        let max_kv = (0..c.num_layers).map(|l| c.kv_dim(l)).max().unwrap_or(0) as usize;
        let ffn = c.ffn_size as usize;
        let vocab = model.vocab_size as usize;
        let max_cols = model.max_cols;
        let split = (rows.min(SPLIT_ROWS as usize)).max(1);
        let heads = c.num_heads as usize;
        // Split attention partials: the ring's keys for sliding layers, the
        // context for global ones; drafter rows read the same layers.
        let partials = [
            k::sizes::attention_ring_partial_floats(split, heads, ring as usize),
            k::sizes::attention_linear_partial_floats(
                split,
                heads,
                c.head_dim_global as usize,
                max_context as usize,
            ),
            k::sizes::attention_linear_partial_floats(
                split,
                heads,
                c.head_dim_sliding as usize,
                max_context as usize,
            ),
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
        .max(MAX_DRAFT_SESSIONS as usize * heads * 1024);
        let q8_rows = rows.max(logit_rows);
        let q8_bytes = if q8_rows > SPLIT_ROWS as usize {
            k::sizes::q8_activation_read_bytes(q8_rows, max_cols)
        } else {
            0
        };
        let half_cols = if c.has_experts() {
            max_cols
        } else {
            model.max_half_cols
        };
        let x_half = if q8_rows > SPLIT_ROWS as usize {
            q8_rows * half_cols
        } else {
            0
        };
        let argmax_scratch = 2 * SPLIT_ROWS as usize * vocab.div_ceil(1024);
        let s = Scratch {
            tokens: stream.alloc_indices(rows, model.vocab_size).map_err(hip)?,
            key_ends: alloc(&stream, rows)?,
            x: alloc(&stream, rows * d)?,
            h: alloc(&stream, rows * d)?,
            q: alloc(&stream, rows * max_q)?,
            kv: alloc(&stream, rows * (max_q + 2 * max_kv))?,
            attn: alloc(&stream, rows * max_q)?,
            o: alloc(&stream, rows * d)?,
            gate: alloc(&stream, (rows * 2 * ffn).max(argmax_scratch))?,
            act: alloc(&stream, rows * ffn)?,
            hsel: alloc(&stream, logit_rows * d)?,
            logits: alloc(&stream, logit_rows * vocab)?,
            partials: alloc(&stream, partials)?,
            stage: Stage {
                q8: alloc(&stream, q8_bytes)?,
                x_half: alloc(&stream, x_half)?,
            },
            wmma: alloc(&stream, k::sizes::wmma_gemv_scratch_bytes(max_cols))?,
            argmax: alloc(&stream, SPLIT_ROWS as usize)?,
        };
        let mut moe = None;
        let mut blas = None;
        if c.has_experts() {
            let used = c.experts_used as usize;
            let experts = c.num_experts as usize;
            let slots = rows * used;
            let grouped = rows.min(MAX_GROUP_SLOTS as usize) * used;
            let width = c.expert_ffn_size as usize;
            moe = Some(MoeScratch {
                x_half: alloc(&stream, rows * d)?,
                h: alloc(&stream, rows * d)?,
                logits: alloc(&stream, rows * experts)?,
                ids: stream
                    .alloc_indices_with(slots, IndexSpec::below(c.num_experts).signed())
                    .map_err(hip)?,
                weights: alloc(&stream, slots)?,
                gu: alloc(
                    &stream,
                    (slots * 2 * width * 2).max(grouped * 2 * width * 4),
                )?,
                act: alloc(&stream, (slots * width * 2).max(grouped * width * 4))?,
                out: alloc(&stream, slots * d)?,
                cursors: alloc(&stream, experts)?,
                groups: HashMap::new(),
                routing: HashMap::new(),
            });
            // Prefill router logits as one GEMM, on hipBLAS's default
            // atomics mode like the C++ handle.
            blas = Some(Blas::with_default_atomics(&stream).map_err(hip)?);
        }
        let mut scratch_bytes = (s.tokens.len() + s.key_ends.len() + s.argmax.len()) * 4
            + [
                &s.x,
                &s.h,
                &s.q,
                &s.kv,
                &s.attn,
                &s.o,
                &s.gate,
                &s.act,
                &s.hsel,
                &s.logits,
                &s.partials,
            ]
            .into_iter()
            .map(bytes_of)
            .sum::<usize>()
            + bytes_of(&s.stage.q8)
            + bytes_of(&s.stage.x_half)
            + bytes_of(&s.wmma);
        if let Some(m) = &moe {
            scratch_bytes += bytes_of(&m.x_half)
                + bytes_of(&m.h)
                + bytes_of(&m.logits)
                + m.ids.len() * 4
                + bytes_of(&m.weights)
                + bytes_of(&m.gu)
                + bytes_of(&m.act)
                + bytes_of(&m.out)
                + bytes_of(&m.cursors);
        }
        let mut draft = None;
        if let Some(dm) = &model.draft {
            let dc = &dm.config;
            let n = MAX_DRAFT_SESSIONS as usize;
            let th = dc.target_hidden_size as usize;
            let hidden = dc.hidden_size as usize;
            let dq = (0..dc.num_layers).map(|l| dc.q_dim(l)).max().unwrap_or(0) as usize;
            let dffn = dc.ffn_size as usize;
            let workspace = k::sizes::mtp_candidate_workspace(vocab);
            let mut ids = Vec::with_capacity(n);
            let mut scratch_ids = Vec::with_capacity(n);
            let mut scores = Vec::with_capacity(n);
            for _ in 0..n {
                ids.push(
                    stream
                        .alloc_indices_with(workspace, IndexSpec::below(model.vocab_size).padded())
                        .map_err(hip)?,
                );
                scratch_ids.push(alloc(&stream, workspace)?);
                scores.push(alloc(&stream, k::sizes::MTP_CANDIDATES)?);
            }
            let d = DraftScratch {
                tokens: stream.alloc_indices(n, model.vocab_size).map_err(hip)?,
                embed: alloc(&stream, n * th)?,
                concat: alloc(&stream, n * 2 * th)?,
                x: alloc(&stream, n * hidden)?,
                h: alloc(&stream, n * hidden)?,
                q: alloc(&stream, n * dq)?,
                attn: alloc(&stream, n * dq)?,
                o: alloc(&stream, n * hidden)?,
                gate: alloc(&stream, n * 2 * dffn)?,
                act: alloc(&stream, n * dffn)?,
                logits: alloc(&stream, n * vocab)?,
                next: alloc(&stream, n * th)?,
                ids,
                scratch_ids,
                scores,
            };
            scratch_bytes += [
                &d.embed, &d.concat, &d.x, &d.h, &d.q, &d.attn, &d.o, &d.gate, &d.act, &d.logits,
                &d.next,
            ]
            .into_iter()
            .map(bytes_of)
            .sum::<usize>()
                + n * (workspace * 8 + k::sizes::MTP_CANDIDATES * 4);
            draft = Some(d);
        }
        let mut candidates = None;
        if vocab.is_multiple_of(4) {
            let split = SPLIT_ROWS as usize;
            let records = split * (k::sizes::CANDIDATE_TOP + 2 * k::sizes::MAX_CANDIDATE_EXTRA);
            let c = CandidateScratch {
                work: alloc(&stream, k::sizes::candidate_work_bytes(split))?,
                counts: alloc(&stream, split)?,
                out: alloc(&stream, 2 * records)?,
                extra: stream
                    .alloc_indices(k::sizes::MAX_CANDIDATE_EXTRA, model.vocab_size)
                    .map_err(hip)?,
            };
            scratch_bytes +=
                bytes_of(&c.work) + bytes_of(&c.counts) + bytes_of(&c.out) + c.extra.len() * 4;
            candidates = Some(c);
        }
        let expert_stream = if c.has_experts() {
            Some((
                Stream::new().map_err(hip)?,
                Event::new().map_err(hip)?,
                Event::new().map_err(hip)?,
            ))
        } else {
            None
        };
        Ok(Executor {
            expert_stream,
            tap: None,
            draft,
            candidates,
            scratch_bytes,
            key_widths: Self::key_widths(model),
            half_prefill: model.half_prefill,
            has_draft: model.has_draft(),
            vocab: model.vocab_size,
            logits_host: vec![0.0; logit_rows * vocab],
            stream,
            blas,
            max_rows,
            max_logit_rows,
            max_context,
            ring,
            s,
            moe,
        })
    }

    /// Sets or clears the layer tap (`SetTapSink`).
    pub fn set_tap_sink(&mut self, sink: Option<LayerTapSink>) {
        self.tap = sink;
    }

    /// Device bytes of the shared scratch.
    pub fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn max_rows(&self) -> u32 {
        self.max_rows
    }

    pub fn ring(&self) -> u32 {
        self.ring
    }

    pub fn widths(&self) -> &[u32] {
        &self.key_widths
    }

    /// Bytes of a cache's draft table (0 when the head is not ordered).
    pub fn draft_table_bytes(model: &DeviceModel) -> usize {
        if model.draft_head_tokens.is_empty() {
            0
        } else {
            crate::draft_vocab::DRAFT_TABLE_ROWS as usize * model.draft_row_bytes
        }
    }

    /// Per-session cache bytes (`CacheBytes`): sliding layers keep the ring.
    pub fn cache_bytes(
        c: &Config,
        max_context: u32,
        ring: u32,
        key_widths: &[u32],
        vocab: u32,
        draft_table_bytes: usize,
    ) -> usize {
        let align = |n: usize| n.div_ceil(256) * 256;
        let mut bytes = 0;
        for l in 0..c.num_layers {
            let slots = if c.is_sliding(l) { ring } else { max_context } as usize;
            bytes += align(slots * key_widths[l as usize] as usize * 2)
                + align(slots * c.kv_dim(l) as usize * 2);
        }
        bytes
            + align(c.hidden_size as usize * 4)
            + align(SPLIT_ROWS as usize * c.hidden_size as usize * 4)
            + align(vocab as usize * 4)
            + align(draft_table_bytes)
    }

    /// A session's cache of `max_context` positions (`CreateCache`).
    pub fn create_cache(&self, model: &DeviceModel, max_context: u32) -> Result<KvCache, String> {
        let c = &model.config;
        if max_context == 0 || max_context > self.max_context {
            return Err("session context exceeds the executor capacity".into());
        }
        let s = &self.stream;
        let mut k_ = Vec::new();
        let mut v = Vec::new();
        let mut slots = Vec::new();
        for l in 0..c.num_layers {
            let n = if c.is_sliding(l) {
                self.ring
            } else {
                max_context
            };
            k_.push(alloc::<F16>(
                s,
                n as usize * self.key_widths[l as usize] as usize,
            )?);
            v.push(alloc::<F16>(s, n as usize * c.kv_dim(l) as usize)?);
            slots.push(n);
        }
        let table = Self::draft_table_bytes(model);
        Ok(KvCache {
            max_context,
            ring: self.ring,
            bytes: Self::cache_bytes(
                c,
                max_context,
                self.ring,
                &self.key_widths,
                self.vocab,
                table,
            ),
            k: k_,
            v,
            slots,
            hidden: alloc(s, c.hidden_size as usize)?,
            ahead_hidden: alloc(s, SPLIT_ROWS as usize * c.hidden_size as usize)?,
            logits: alloc(s, self.vocab as usize)?,
            draft_table: if table != 0 {
                Some(alloc(s, table)?)
            } else {
                None
            },
            written_from: AtomicU32::new(0),
            written_to: AtomicU32::new(u32::MAX),
        })
    }

    /// One session's forward (`Forward(cache, tokens, ...)`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_one(
        &mut self,
        model: &DeviceModel,
        cache: &mut KvCache,
        tokens: &[i32],
        first_position: u32,
        logit_rows: &[u32],
        images: &[ImageRows],
        siblings: u32,
    ) -> Result<(), String> {
        let rows = u32::try_from(tokens.len()).map_err(|_| "gemma4 forward rows")?;
        let mut segments = [Segment {
            cache,
            first_position,
            rows,
            siblings,
        }];
        self.forward(model, &mut segments, tokens, logit_rows, images)
    }

    /// Evaluates the segments' rows (`tokens` in segment order) and writes
    /// softcapped logits of `logit_rows` (indices into `tokens`, in order)
    /// to the device logits buffer (`Executor::Forward`). Row-wise work runs
    /// once for all rows, attention and cache writes per segment.
    pub fn forward(
        &mut self,
        model: &DeviceModel,
        segments: &mut [Segment<'_>],
        tokens: &[i32],
        logit_rows: &[u32],
        images: &[ImageRows],
    ) -> Result<(), String> {
        let c = &model.config;
        let n = u32::try_from(tokens.len()).map_err(|_| "gemma4 forward rows")?;
        let mut total = 0u64;
        for seg in segments.iter() {
            if seg.rows == 0
                || seg.cache.ring != self.ring
                || u64::from(seg.first_position) + u64::from(seg.rows)
                    > u64::from(seg.cache.max_context)
            {
                return Err("gemma4 forward segment exceeds its cache".into());
            }
            // Siblings stand beside drafts: at most one per draft.
            if seg.siblings != 0
                && (2 * seg.siblings >= seg.rows || seg.rows > SPLIT_ROWS || !images.is_empty())
            {
                return Err("gemma4 forward siblings exceed the chain".into());
            }
            total += u64::from(seg.rows);
        }
        for seg in segments.iter_mut() {
            seg.cache
                .written_from
                .fetch_min(seg.first_position, Ordering::Relaxed);
            seg.cache
                .written_to
                .fetch_max(seg.first_position + seg.rows, Ordering::Relaxed);
        }
        // Several sessions share a forward only at decode widths, where every
        // projection keeps its single-session arithmetic.
        if n == 0
            || n > self.max_rows
            || total != u64::from(n)
            || (segments.len() > 1 && n > SPLIT_ROWS)
            || logit_rows.len() > self.max_logit_rows as usize
        {
            return Err("gemma4 forward exceeds its capacity".into());
        }
        let env = Env {
            stream: &self.stream,
            half_prefill: self.half_prefill,
            has_draft: self.has_draft,
        };
        let st = env.stream;
        let max_kv = (0..c.num_layers).map(|j| c.kv_dim(j)).max().unwrap_or(0) as usize;
        // V rows of separate projections after the widest K rows
        // (`l.v = l.k + rows * MaxKvDim`).
        let v_base = self.max_rows as usize * max_kv;
        let d = c.hidden_size;
        let du = d as usize;
        let nu = n as usize;
        let eps = c.rms_eps;
        let layers = &model.layers;
        let first_layer = layers.first().ok_or("gemma4 model without layers")?;
        let key_widths = &self.key_widths;
        let derived = |layer: u32| key_widths[layer as usize] != c.kv_dim(layer);
        let Scratch {
            tokens: token_ids,
            key_ends,
            x,
            h,
            q,
            kv,
            attn,
            o,
            gate,
            act,
            hsel,
            logits,
            partials,
            stage,
            wmma,
            ..
        } = &mut self.s;
        let mut moe = self.moe.as_mut();
        let blas = self.blas.as_ref();
        let mut tap = self.tap.as_mut();
        let side = self.expert_stream.as_ref();

        let ids: Vec<u32> = tokens
            .iter()
            .map(|&t| u32::try_from(t).map_err(|_| "gemma4 token outside the vocabulary"))
            .collect::<Result<_, _>>()?;
        st.upload_indices(&ids, token_ids).map_err(hip)?;
        unit(k::embedding_lookup(
            st,
            model.token_embd.q()?,
            token_ids.slice(0..nu),
            x.slice_mut(0..nu * du),
            nu,
            du,
            model.vocab_size as usize,
        ))?;
        unit(k::scale_rms_norm(
            st,
            x.slice_mut(0..nu * du),
            (d as f32).sqrt(),
            first_layer.attn_norm.f32()?,
            h.slice_mut(0..nu * du),
            n,
            d,
            eps,
        ))?;
        // Image rows take their embeddings unscaled; each image's rows see
        // all of its keys in sliding layers.
        let mut has_key_ends = false;
        if !images.is_empty() {
            if segments.len() != 1 {
                return Err("gemma4 images need a single-session forward".into());
            }
            let first_position = segments[0].first_position;
            let mut ends = vec![0u32; nu];
            let mut previous_end = 0u32;
            for image in images {
                if image.count == 0
                    || image.row < previous_end
                    || image.count > n - image.row
                    || image.embedding.len() < image.count as usize * du
                {
                    return Err("gemma4 image rows are out of order".into());
                }
                previous_end = image.row + image.count;
                let at = image.row as usize * du;
                let len = image.count as usize * du;
                st.copy(image.embedding.slice(0..len), x.slice_mut(at..at + len))
                    .map_err(hip)?;
                unit(k::rms_norm(
                    st,
                    x.slice(at..at + len),
                    Some(first_layer.attn_norm.f32()?),
                    h.slice_mut(at..at + len),
                    image.count,
                    d,
                    eps,
                ))?;
                for end in &mut ends[image.row as usize..previous_end as usize] {
                    *end = first_position + previous_end;
                }
            }
            st.upload(&ends, key_ends.slice_mut(0..nu)).map_err(hip)?;
            has_key_ends = true;
        }

        // Prefill (rows past the small-batch width) quantizes activations for
        // the W8A8 GEMMs; binary16 prefill stages each input before its
        // projections instead.
        let prefill = n > SPLIT_ROWS && !env.half_prefill;
        let half = n > SPLIT_ROWS && env.half_prefill;
        // Whether the previous layer's mixture wrote this layer's binary16
        // input, and whether h already holds its WMMA pack.
        let mut half_ready = false;
        let mut h_packed = false;
        let half_only =
            |ws: &[&Weight]| half && ws.iter().all(|w| w.is_empty() || half_prefill_format(w.ty));
        let qkv_half_only = |l: &DeviceLayer| {
            if !l.attn_qkv.is_empty() {
                half_only(&[&l.attn_qkv, &l.attn_v])
            } else {
                half_only(&[&l.attn_q, &l.attn_k, &l.attn_v])
            }
        };
        let fp32_input = |ws: &[&Weight]| prefill && ws.iter().any(|w| w.ty == GgmlType::F16);
        let qkv_fp32 =
            |l: &DeviceLayer| fp32_input(&[&l.attn_qkv, &l.attn_q, &l.attn_k, &l.attn_v]);
        let pack_width = |w: &Weight, rows: u32| wmma_projection(env, w, rows);

        for li in 0..c.num_layers {
            let l = &layers[li as usize];
            let sliding = c.is_sliding(li);
            let dim = c.head_dim(li);
            let q_dim = c.q_dim(li);
            let kv_dim = c.kv_dim(li);
            // Layer li > 0 of a prefill reads the rows the previous layer's
            // post-FFN norm wrote (Q8_1 or binary16).
            let hq = if li == 0 {
                quantize(env, stage, h.slice(0..nu * du), n, d)?
            } else if half_ready {
                Staged::Half
            } else if prefill && !qkv_fp32(l) {
                Staged::Q8
            } else {
                quantize(env, stage, h.slice(0..nu * du), n, d)?
            };
            // A fused projection leaves [q | k | v] or [q | k] rows in kv; at
            // WMMA decode widths [q | k] stays split.
            let wmma_qkv = n <= SPLIT_ROWS
                && (env.has_draft || n >= 6)
                && wmma_gemv_supports(l.attn_qkv.ty, l.attn_qkv.rows, l.attn_qkv.cols);
            let separate_v = !l.attn_v.is_empty() && l.attn_qkv.rows == q_dim + kv_dim;
            let qkv_stride = if l.attn_qkv.is_empty() || (separate_v && wmma_qkv) {
                0
            } else {
                l.attn_qkv.rows
            };
            let stride = qkv_stride as usize;
            // Offsets into kv of the q, k and v sources.
            let (q_off, k_off, v_off);
            if qkv_stride != 0 {
                project(
                    env,
                    stage,
                    wmma,
                    &l.attn_qkv,
                    h.slice(0..nu * du),
                    hq,
                    n,
                    kv.slice_mut(0..nu * stride),
                    h_packed,
                )?;
                q_off = 0;
                k_off = q_dim as usize;
                if separate_v {
                    v_off = nu * stride;
                    let packed = h_packed || pack_width(&l.attn_qkv, n);
                    let (_, v_rows) = kv.as_mut_slice().split_at(v_off);
                    project(
                        env,
                        stage,
                        wmma,
                        &l.attn_v,
                        h.slice(0..nu * du),
                        hq,
                        n,
                        v_rows.range(0..nu * kv_dim as usize),
                        packed,
                    )?;
                } else {
                    v_off = if l.attn_v.is_empty() {
                        k_off
                    } else {
                        k_off + kv_dim as usize
                    };
                }
            } else {
                project(
                    env,
                    stage,
                    wmma,
                    &l.attn_q,
                    h.slice(0..nu * du),
                    hq,
                    n,
                    q.slice_mut(0..nu * q_dim as usize),
                    h_packed,
                )?;
                let q_packed = h_packed || pack_width(&l.attn_q, n);
                let (k_rows, v_rows) = kv.as_mut_slice().split_at(v_base);
                project(
                    env,
                    stage,
                    wmma,
                    &l.attn_k,
                    h.slice(0..nu * du),
                    hq,
                    n,
                    k_rows.range(0..nu * kv_dim as usize),
                    q_packed,
                )?;
                q_off = 0;
                k_off = 0;
                if !l.attn_v.is_empty() {
                    let packed = q_packed || pack_width(&l.attn_k, n);
                    project(
                        env,
                        stage,
                        wmma,
                        &l.attn_v,
                        h.slice(0..nu * du),
                        hq,
                        n,
                        v_rows.range(0..nu * kv_dim as usize),
                        packed,
                    )?;
                    v_off = v_base;
                } else {
                    v_off = 0;
                }
            }
            let q_row = if qkv_stride != 0 {
                stride
            } else {
                q_dim as usize
            };
            let kv_row = if qkv_stride != 0 {
                stride
            } else {
                kv_dim as usize
            };
            let v_row = if qkv_stride != 0 && !separate_v {
                stride
            } else {
                kv_dim as usize
            };
            // At decode widths the attention merge packs its FP32 rows for a
            // WMMA output projection.
            let attn_pack =
                !(prefill || half_only(&[&l.attn_output])) && pack_width(&l.attn_output, n);
            let out_f32 = !half_only(&[&l.attn_output]);
            let theta_scale = c.rope_theta(li).powf(-2.0 / dim as f32);
            let freq = if sliding {
                None
            } else {
                Some(model.rope_factors.slice(0..(dim / 2) as usize))
            };
            let mut row0 = 0usize;
            for seg in segments.iter_mut() {
                let cache = &mut *seg.cache;
                let rows = seg.rows;
                let ring = if sliding {
                    cache.ring
                } else {
                    cache.slots[li as usize]
                };
                let spare_key = seg.first_position + seg.rows - seg.siblings;
                let rotated = if derived(li) {
                    model.global_rope_pairs
                } else {
                    0
                };
                let k_cache = cache.k[li as usize].as_mut_slice();
                let v_cache = cache.v[li as usize].as_mut_slice();
                let v_stride = if separate_v || qkv_stride == 0 {
                    kv_dim
                } else {
                    qkv_stride
                };
                if qkv_stride != 0 {
                    let src = kv.as_slice();
                    let qs = src.range(q_off + row0 * q_row..src.len());
                    let ks = src.range(k_off + row0 * kv_row..src.len());
                    let vs = src.range(v_off + row0 * v_row..src.len());
                    let q_out =
                        q.slice_mut(row0 * q_dim as usize..(row0 + rows as usize) * q_dim as usize);
                    let r = if rotated != 0 {
                        k::qkv_post_fused_derived(
                            st,
                            qs,
                            ks,
                            vs,
                            q_out,
                            l.attn_q_norm.f32()?,
                            l.attn_k_norm.f32()?,
                            theta_scale,
                            freq,
                            k_cache,
                            v_cache,
                            rows,
                            c.num_heads,
                            c.kv_heads[li as usize],
                            dim,
                            seg.first_position,
                            ring,
                            eps,
                            rotated,
                            qkv_stride,
                            v_stride,
                            seg.siblings,
                            spare_key,
                        )
                    } else {
                        k::qkv_post_fused(
                            st,
                            qs,
                            ks,
                            vs,
                            q_out,
                            l.attn_q_norm.f32()?,
                            l.attn_k_norm.f32()?,
                            theta_scale,
                            freq,
                            k_cache,
                            v_cache,
                            rows,
                            c.num_heads,
                            c.kv_heads[li as usize],
                            dim,
                            seg.first_position,
                            ring,
                            eps,
                            qkv_stride,
                            v_stride,
                            seg.siblings,
                            spare_key,
                        )
                    };
                    launched(r, "QKV post")?;
                } else {
                    let src = kv.as_slice();
                    let ks = src.range(k_off + row0 * kv_row..src.len());
                    let vs = src.range(v_off + row0 * v_row..src.len());
                    let q_io =
                        q.slice_mut(row0 * q_dim as usize..(row0 + rows as usize) * q_dim as usize);
                    let r = if rotated != 0 {
                        k::qkv_post_derived(
                            st,
                            q_io,
                            ks,
                            vs,
                            l.attn_q_norm.f32()?,
                            l.attn_k_norm.f32()?,
                            theta_scale,
                            freq,
                            k_cache,
                            v_cache,
                            rows,
                            c.num_heads,
                            c.kv_heads[li as usize],
                            dim,
                            seg.first_position,
                            ring,
                            eps,
                            rotated,
                            v_stride,
                            seg.siblings,
                            spare_key,
                        )
                    } else {
                        k::qkv_post(
                            st,
                            q_io,
                            ks,
                            vs,
                            l.attn_q_norm.f32()?,
                            l.attn_k_norm.f32()?,
                            theta_scale,
                            freq,
                            k_cache,
                            v_cache,
                            rows,
                            c.num_heads,
                            c.kv_heads[li as usize],
                            dim,
                            seg.first_position,
                            ring,
                            eps,
                            v_stride,
                            seg.siblings,
                            spare_key,
                        )
                    };
                    launched(r, "QKV post")?;
                }

                let qrange = row0 * q_dim as usize..(row0 + rows as usize) * q_dim as usize;
                let att = Attention {
                    q: q.slice(qrange.clone()),
                    k_cache: cache.k[li as usize].as_slice(),
                    v_cache: cache.v[li as usize].as_slice(),
                    out: out_f32.then(|| attn.slice_mut(qrange.clone())),
                    partials: partials.as_mut_slice(),
                    rows,
                    heads: c.num_heads,
                    kv_heads: c.kv_heads[li as usize],
                    head_dim: dim,
                    first_position: seg.first_position,
                    shared_position: false,
                    key_limit: u32::MAX,
                    window: if sliding { c.sliding_window } else { 0 },
                    ring: if sliding { cache.ring } else { 0 },
                    slots: cache.slots[li as usize],
                    rope_pairs: rotated,
                    key_ends: (sliding && has_key_ends)
                        .then(|| key_ends.slice(row0..row0 + rows as usize)),
                    out_half: half.then(|| stage.x_half.slice_mut(qrange.clone())),
                    siblings: seg.siblings,
                    spare_key,
                    pack: attn_pack.then(|| wmma.as_mut_slice()),
                    pack_row0: row0 as u32,
                };
                attention(st, att)?;
                row0 += rows as usize;
            }

            let attn_staged = if half {
                Staged::Half
            } else {
                quantize(env, stage, attn.slice(0..nu * q_dim as usize), n, q_dim)?
            };
            project(
                env,
                stage,
                wmma,
                &l.attn_output,
                attn.slice(0..nu * q_dim as usize),
                attn_staged,
                n,
                o.slice_mut(0..nu * du),
                attn_pack,
            )?;
            // Expert layers also normalize the attention residual for the
            // experts.
            let is_moe = !l.gate_up_exps.is_empty();
            let dense_half_only = if !l.ffn_gate_up.is_empty() {
                half_only(&[&l.ffn_gate_up])
            } else {
                half_only(&[&l.ffn_gate, &l.ffn_up])
            };
            let experts_half_only = half && is_moe && half_experts(l, c);
            let dense_fp32 = fp32_input(&[&l.ffn_gate_up, &l.ffn_gate, &l.ffn_up]);
            // At decode widths the norm packs h for a WMMA [gate | up] (or
            // gate).
            let ffn_first = if !l.ffn_gate_up.is_empty() {
                &l.ffn_gate_up
            } else {
                &l.ffn_gate
            };
            let ffn_pack = !(prefill || dense_half_only) && pack_width(ffn_first, n);
            let q8_out = prefill && !dense_fp32;
            {
                let h_out = (!dense_half_only).then(|| h.slice_mut(0..nu * du));
                let Stage { q8, x_half } = &mut *stage;
                let q8_slice = q8_out.then(|| q8.as_mut_slice());
                let h_half = half.then(|| x_half.slice_mut(0..nu * du));
                let pack = ffn_pack.then(|| wmma.as_mut_slice());
                if is_moe {
                    let m = moe
                        .as_deref_mut()
                        .ok_or("gemma4 expert layer without expert scratch")?;
                    let h2 = (!experts_half_only).then(|| m.h.slice_mut(0..nu * du));
                    let h2_half = half.then(|| m.x_half.slice_mut(0..nu * du));
                    launched(
                        k::post_attention_norm_moe(
                            st,
                            o.slice(0..nu * du),
                            l.post_attn_norm.f32()?,
                            x.slice_mut(0..nu * du),
                            l.ffn_norm.f32()?,
                            h_out,
                            n,
                            d,
                            eps,
                            q8_slice,
                            l.pre_ffn_norm_2.f32()?,
                            h2,
                            h_half,
                            h2_half,
                            pack,
                        ),
                        "post-attention norm",
                    )?;
                } else {
                    launched(
                        k::post_attention_norm(
                            st,
                            o.slice(0..nu * du),
                            l.post_attn_norm.f32()?,
                            x.slice_mut(0..nu * du),
                            l.ffn_norm.f32()?,
                            h_out,
                            n,
                            d,
                            eps,
                            q8_slice,
                            h_half,
                            pack,
                        ),
                        "post-attention norm",
                    )?;
                }
            }
            // At grouped widths the experts run beside the dense MLP on their
            // own stream: routing is latency bound, the projections
            // bandwidth bound. Their buffers are the side's until the join;
            // the residual and their input rows are only read.
            let mut fork = None;
            if is_moe
                && n <= MAX_GROUP_SLOTS
                && let Some((side_stream, fork_event, join_event)) = side
            {
                let m = moe
                    .as_deref_mut()
                    .ok_or("gemma4 expert layer without expert scratch")?;
                let slots = n * c.experts_used;
                ensure_groups(st, m, slots, c.num_experts)?;
                let f = st.fork(side_stream, fork_event, join_event).map_err(hip)?;
                f.lend(x.allocation()).map_err(hip)?;
                f.lend(m.h.allocation()).map_err(hip)?;
                for a in [
                    m.logits.allocation(),
                    m.weights.allocation(),
                    m.act.allocation(),
                    m.out.allocation(),
                ] {
                    f.hand_over(a).map_err(hip)?;
                }
                f.hand_over(m.ids.view_mut().allocation()).map_err(hip)?;
                let groups = m.groups.get_mut(&slots).ok_or("gemma4 expert groups")?;
                f.hand_over(groups.view_mut().allocation()).map_err(hip)?;
                let side_env = Env {
                    stream: side_stream,
                    ..env
                };
                experts(side_env, model, blas, m, l, x.slice(0..nu * du), n)?;
                fork = Some(f);
            }
            let fq = if q8_out {
                Staged::Q8
            } else if half {
                Staged::Half
            } else {
                quantize(env, stage, h.slice(0..nu * du), n, d)?
            };
            // The dense MLP. `down_in` names the FP32 rows the down
            // projection reads (gate or act), `gq` their staging.
            let mut gq = Staged::None;
            let mut down_from_act = false;
            let mut down_pack = false;
            let gu = &l.ffn_gate_up;
            let gate_up_half = dense_half_format(gu.ty);
            let gate_up_packed = packed_prefill(gu, n);
            let gate_up_q8 = gu.ty == GgmlType::Q8_0;
            let gate_up_f16 = gu.ty == GgmlType::F16;
            let ffn = c.ffn_size as usize;
            let mut fused_geglu = false;
            if !gu.is_empty()
                && half
                && (gate_up_half || gate_up_packed || gate_up_q8 || gate_up_f16)
                && half_only(&[&l.ffn_down])
            {
                // The GEMM writes GeGLU of each binary16 [gate | up] pair into
                // the (free) gate rows as the down projection's binary16
                // input.
                let fx = staged_half(stage, h.slice(0..nu * du), fq, nu * du)?;
                let out = gate
                    .as_mut_slice()
                    .cast::<F16>()
                    .ok_or("gate rows alignment")?
                    .range(0..nu * ffn);
                fused_geglu = (gate_up_packed
                    && packed_prefill_gemm(st, gu, fx, None, Some(out), n, true)?)
                    || (gate_up_half && {
                        let out = gate
                            .as_mut_slice()
                            .cast::<F16>()
                            .ok_or("gate rows alignment")?
                            .range(0..nu * ffn);
                        attempt(k::dense_half_gemm_geglu(
                            st,
                            gu.q()?,
                            fx,
                            out,
                            n,
                            gu.rows,
                            gu.cols,
                            gu.grouped,
                        ))?
                    })
                    || ((gate_up_q8 || gate_up_f16) && !gate_up_packed && {
                        let out = gate
                            .as_mut_slice()
                            .cast::<F16>()
                            .ok_or("gate rows alignment")?
                            .range(0..nu * ffn);
                        if gate_up_f16 {
                            attempt(k::dense_f16_geglu_gemm_half(
                                st,
                                gu.q()?,
                                fx,
                                out,
                                nu,
                                gu.rows as usize,
                                gu.cols as usize,
                            ))?
                        } else {
                            attempt(k::dense_f16_geglu_gemm(
                                st,
                                gu.q()?,
                                fx,
                                out,
                                nu,
                                gu.rows as usize,
                                gu.cols as usize,
                                gu.grouped,
                            ))?
                        }
                    });
                if fused_geglu {
                    gq = Staged::Input;
                }
            }
            if !fused_geglu {
                if !gu.is_empty() && !prefill {
                    // One projection writes [gate | up] rows; GeGLU packs its
                    // rows for a WMMA down projection.
                    project(
                        env,
                        stage,
                        wmma,
                        gu,
                        h.slice(0..nu * du),
                        fq,
                        n,
                        gate.slice_mut(0..nu * 2 * ffn),
                        ffn_pack,
                    )?;
                    let act_fp32 = !half_only(&[&l.ffn_down]);
                    down_pack = act_fp32 && pack_width(&l.ffn_down, n);
                    if down_pack {
                        launched(
                            k::geglu_packed_pack(
                                st,
                                gate.slice(0..nu * 2 * ffn),
                                act.slice_mut(0..nu * ffn),
                                n,
                                c.ffn_size,
                                wmma.as_mut_slice(),
                            ),
                            "GeGLU",
                        )?;
                    } else {
                        launched(
                            k::geglu_packed(
                                st,
                                gate.slice(0..nu * 2 * ffn),
                                act_fp32.then(|| act.slice_mut(0..nu * ffn)),
                                n,
                                c.ffn_size,
                                half.then(|| stage.x_half.slice_mut(0..nu * ffn)),
                            ),
                            "GeGLU",
                        )?;
                    }
                    down_from_act = true;
                    gq = if half {
                        Staged::Half
                    } else {
                        quantize(env, stage, act.slice(0..nu * ffn), n, c.ffn_size)?
                    };
                } else {
                    let up_off = nu * ffn;
                    {
                        let (g, u) = gate.as_mut_slice().split_at(up_off);
                        project(
                            env,
                            stage,
                            wmma,
                            &l.ffn_gate,
                            h.slice(0..nu * du),
                            fq,
                            n,
                            g,
                            ffn_pack,
                        )?;
                        let packed = ffn_pack || pack_width(&l.ffn_gate, n);
                        project(
                            env,
                            stage,
                            wmma,
                            &l.ffn_up,
                            h.slice(0..nu * du),
                            fq,
                            n,
                            u.range(0..nu * ffn),
                            packed,
                        )?;
                    }
                    // A binary16 down projection narrows the FP32 GeGLU rows
                    // itself.
                    if prefill && l.ffn_down.ty != GgmlType::F16 {
                        let g = gate.as_slice();
                        unit(k::geglu_quantize(
                            st,
                            g.range(0..nu * ffn),
                            g.range(up_off..up_off + nu * ffn),
                            stage.q8.as_mut_slice(),
                            n,
                            c.ffn_size,
                        ))?;
                        gq = Staged::Q8;
                    } else {
                        let (g, u) = gate.as_mut_slice().split_at(up_off);
                        unit(k::geglu_in_place(
                            st,
                            u.as_const().range(0..nu * ffn),
                            g,
                            nu * ffn,
                        ))?;
                        gq = quantize(env, stage, gate.slice(0..nu * ffn), n, c.ffn_size)?;
                    }
                }
            }
            if down_from_act {
                project(
                    env,
                    stage,
                    wmma,
                    &l.ffn_down,
                    act.slice(0..nu * ffn),
                    gq,
                    n,
                    o.slice_mut(0..nu * du),
                    down_pack,
                )?;
            } else {
                project(
                    env,
                    stage,
                    wmma,
                    &l.ffn_down,
                    gate.slice(0..gate.len()),
                    gq,
                    n,
                    o.slice_mut(0..nu * du),
                    down_pack,
                )?;
            }
            let last = li + 1 == c.num_layers;
            // At decode widths the layer's last norm packs the next layer's
            // input for its WMMA Q/K/V projections.
            let mut next_pack = false;
            if !last && !prefill {
                let next = &layers[li as usize + 1];
                if pack_width(&next.attn_q, n)
                    || (!next.attn_qkv.is_empty() && pack_width(&next.attn_qkv, n))
                {
                    next_pack = true;
                }
            }
            let next_norm = if last {
                model.output_norm.f32()?
            } else {
                layers[li as usize + 1].attn_norm.f32()?
            };
            if is_moe {
                let m = moe
                    .as_deref_mut()
                    .ok_or("gemma4 expert layer without expert scratch")?;
                match fork.take() {
                    Some(f) => f.join().map_err(hip)?,
                    None => experts(env, model, blas, m, l, x.slice(0..nu * du), n)?,
                }
                let write_h = !(half && !last && qkv_half_only(&layers[li as usize + 1]));
                let h_out = write_h.then(|| h.slice_mut(0..nu * du));
                let used = c.experts_used;
                let slots = nu * used as usize;
                if next_pack {
                    launched(
                        k::moe_finish_pack(
                            st,
                            o.slice(0..nu * du),
                            m.out.slice(0..slots * du),
                            m.weights.slice(0..slots),
                            l.post_ffn_norm_1.f32()?,
                            l.post_ffn_norm_2.f32()?,
                            l.post_ffn_norm.f32()?,
                            l.output_scale,
                            x.slice_mut(0..nu * du),
                            next_norm,
                            h_out,
                            n,
                            d,
                            used,
                            eps,
                            wmma.as_mut_slice(),
                        ),
                        "MoE finish",
                    )?;
                    half_ready = false;
                } else {
                    let h_half = (half && !last).then(|| stage.x_half.slice_mut(0..nu * du));
                    half_ready = h_half.is_some();
                    launched(
                        k::moe_finish(
                            st,
                            o.slice(0..nu * du),
                            m.out.slice(0..slots * du),
                            m.weights.slice(0..slots),
                            l.post_ffn_norm_1.f32()?,
                            l.post_ffn_norm_2.f32()?,
                            l.post_ffn_norm.f32()?,
                            l.output_scale,
                            x.slice_mut(0..nu * du),
                            Some(next_norm),
                            h_out,
                            n,
                            d,
                            used,
                            eps,
                            h_half,
                        ),
                        "MoE finish",
                    )?;
                }
            } else {
                // Binary16 prefill: the norm also writes the next layer's
                // binary16 input (and only that when every consumer reads
                // binary16).
                let next_half = half && !last;
                let write_h = !(next_half && qkv_half_only(&layers[li as usize + 1]));
                let h_out = write_h.then(|| h.slice_mut(0..nu * du));
                let q8_next = prefill && !last && !qkv_fp32(&layers[li as usize + 1]);
                let Stage { q8, x_half } = &mut *stage;
                launched(
                    k::post_feed_forward_norm(
                        st,
                        o.slice(0..nu * du),
                        l.post_ffn_norm.f32()?,
                        l.output_scale,
                        x.slice_mut(0..nu * du),
                        Some(next_norm),
                        h_out,
                        n,
                        d,
                        eps,
                        q8_next.then(|| q8.as_mut_slice()),
                        next_half.then(|| x_half.slice_mut(0..nu * du)),
                        next_pack.then(|| wmma.as_mut_slice()),
                    ),
                    "post-FFN norm",
                )?;
                half_ready = next_half;
            }
            h_packed = next_pack;
            if let Some(tap) = tap.as_mut() {
                let mut rows = vec![0.0f32; nu * du];
                st.download(x.slice(0..nu * du), &mut rows).map_err(hip)?;
                tap(li, &rows, n);
            }
        }

        let m = logit_rows.len();
        if m == 0 {
            return Ok(());
        }
        let mu = m as u32;
        let contiguous = logit_rows.windows(2).all(|w| w[1] == w[0] + 1);
        let first = logit_rows[0] as usize;
        let head_in = if contiguous {
            h.slice(first * du..(first + m) * du)
        } else {
            let mut index = st.alloc_indices(m, n).map_err(hip)?;
            st.upload_indices(logit_rows, &mut index).map_err(hip)?;
            unit(k::gather_rows(
                st,
                h.slice(0..nu * du),
                index.view(),
                hsel.slice_mut(0..m * du),
                mu,
                d,
                n,
            ))?;
            hsel.slice(0..m * du)
        };
        let vocab = model.vocab_size as usize;
        let staged = quantize(env, stage, head_in, mu, d)?;
        project(
            env,
            stage,
            wmma,
            &model.output,
            head_in,
            staged,
            mu,
            logits.slice_mut(0..m * vocab),
            false,
        )?;
        if c.final_logit_softcap > 0.0 {
            unit(k::softcap(
                st,
                logits.slice_mut(0..m * vocab),
                m * vocab,
                c.final_logit_softcap,
            ))?;
        }
        Ok(())
    }

    /// Keeps hidden row `row` of the last forward as the cache's frontier
    /// state (`CommitHidden`).
    pub fn commit_hidden(
        &mut self,
        model: &DeviceModel,
        cache: &mut KvCache,
        row: u32,
    ) -> Result<(), String> {
        let d = model.config.hidden_size as usize;
        let at = row as usize * d;
        self.stream
            .copy(self.s.h.slice(at..at + d), cache.hidden.slice_mut(0..d))
            .map_err(hip)
    }

    /// Copies row `row` of the last forward to stash slot `slot` of the
    /// cache's lookahead hidden states (`StashHidden`).
    pub fn stash_hidden(
        &mut self,
        model: &DeviceModel,
        cache: &mut KvCache,
        row: u32,
        slot: u32,
    ) -> Result<(), String> {
        if slot >= SPLIT_ROWS {
            return Err("gemma4 hidden stash slot".into());
        }
        let d = model.config.hidden_size as usize;
        let at = row as usize * d;
        let to = slot as usize * d;
        self.stream
            .copy(
                self.s.h.slice(at..at + d),
                cache.ahead_hidden.slice_mut(to..to + d),
            )
            .map_err(hip)
    }

    /// Copies stash slot `slot` back to the frontier state
    /// (`AdoptStashedHidden`).
    pub fn adopt_stashed_hidden(
        &mut self,
        model: &DeviceModel,
        cache: &mut KvCache,
        slot: u32,
    ) -> Result<(), String> {
        if slot >= SPLIT_ROWS {
            return Err("gemma4 hidden stash slot".into());
        }
        let d = model.config.hidden_size as usize;
        let at = slot as usize * d;
        let KvCache {
            hidden,
            ahead_hidden,
            ..
        } = cache;
        self.stream
            .copy(ahead_hidden.slice(at..at + d), hidden.slice_mut(0..d))
            .map_err(hip)
    }

    /// Keeps logit row `row` of the last forward in the cache
    /// (`CommitLogits`).
    pub fn commit_logits(&mut self, cache: &mut KvCache, row: u32) -> Result<(), String> {
        let vocab = self.vocab as usize;
        let at = row as usize * vocab;
        self.stream
            .copy(
                self.s.logits.slice(at..at + vocab),
                cache.logits.slice_mut(0..vocab),
            )
            .map_err(hip)
    }

    /// Copies the cache's logits to `out` (`CopyCacheLogits`).
    pub fn copy_cache_logits(&self, cache: &KvCache, out: &mut Vec<f32>) -> Result<(), String> {
        out.resize(self.vocab as usize, 0.0);
        self.stream
            .download(cache.logits.slice(0..self.vocab as usize), out)
            .map_err(hip)
    }

    /// The logits of the last forward on the host, `[rows][vocab]`
    /// (`ReadLogits`).
    pub fn read_logits(&mut self, rows: usize) -> Result<&[f32], String> {
        let count = rows * self.vocab as usize;
        if count > self.logits_host.len() {
            return Err("gemma4 logit read exceeds capacity".into());
        }
        if count != 0 {
            self.stream
                .download(
                    self.s.logits.slice(0..count),
                    &mut self.logits_host[..count],
                )
                .map_err(hip)?;
        }
        Ok(&self.logits_host[..count])
    }

    /// Argmax token id per row of the last forward's `rows` logit rows
    /// (`ReadArgmax`).
    pub fn read_argmax(&mut self, rows: usize) -> Result<Vec<u32>, String> {
        if rows == 0 || rows > SPLIT_ROWS as usize {
            return Err("gemma4 argmax rows exceed capacity".into());
        }
        let vocab = self.vocab as usize;
        let scratch = 2 * SPLIT_ROWS as usize * vocab.div_ceil(1024);
        let Scratch {
            logits,
            argmax,
            gate,
            ..
        } = &mut self.s;
        launched(
            k::batched_argmax(
                &self.stream,
                logits.slice(0..rows * vocab),
                argmax.slice_mut(0..rows),
                rows,
                vocab,
                gate.slice_mut(0..scratch),
                scratch,
            ),
            "argmax",
        )?;
        let mut out = vec![0u32; rows];
        self.stream
            .download(argmax.slice(0..rows), &mut out)
            .map_err(hip)?;
        Ok(out)
    }

    /// Copies the drafter head rows of `tokens` into table rows
    /// `[first, first + tokens.len())` of the cache's draft table
    /// (`LoadDraftTable`). Each row moves as its bytes viewed as floats: a
    /// copy, no arithmetic.
    pub fn load_draft_table(
        &mut self,
        model: &DeviceModel,
        cache: &mut KvCache,
        first: u32,
        tokens: &[i32],
    ) -> Result<(), String> {
        if tokens.is_empty() {
            return Ok(());
        }
        let rows = &model.draft_head_rows;
        let table = cache
            .draft_table
            .as_mut()
            .ok_or("gemma4 draft table exceeds its capacity")?;
        if first as usize + tokens.len() > crate::draft_vocab::DRAFT_TABLE_ROWS as usize {
            return Err("gemma4 draft table exceeds its capacity".into());
        }
        let head = &model
            .draft
            .as_ref()
            .ok_or("gemma4 model has no drafter")?
            .token_embd;
        let row_bytes = model.draft_row_bytes;
        let index: Vec<u32> = tokens
            .iter()
            .map(|&t| {
                usize::try_from(t)
                    .ok()
                    .and_then(|t| rows.get(t).copied())
                    .ok_or_else(|| "gemma4 draft table token".to_string())
            })
            .collect::<Result<_, _>>()?;
        let s = &self.stream;
        let mut indices = s.alloc_indices(index.len(), head.rows).map_err(hip)?;
        s.upload_indices(&index, &mut indices).map_err(hip)?;
        let dim = (row_bytes / 4) as u32;
        let src = head.bytes()?.cast::<f32>().ok_or("draft head alignment")?;
        let at = first as usize * row_bytes;
        let dst = table
            .slice_mut(at..at + tokens.len() * row_bytes)
            .cast::<f32>()
            .ok_or("draft table alignment")?;
        unit(k::gather_rows(
            s,
            src,
            indices.view(),
            dst,
            tokens.len() as u32,
            dim,
            head.rows,
        ))
    }

    /// Drafts up to each job's `steps` tokens with the MTP drafter after its
    /// token at its position (the committed frontier), reading the target's
    /// KV strictly before it and the cache's frontier hidden state
    /// (`DraftChains`). Every step is one drafter forward over all chains
    /// still running and one host wait; then `propose(job, candidates)`
    /// decides each running job's row, in job order. Returns each job's
    /// drafts.
    pub fn draft_chains(
        &mut self,
        model: &DeviceModel,
        jobs: &[DraftJob<'_>],
        propose: &mut dyn FnMut(usize, &MtpCandidateLogits) -> Result<DraftProposal, String>,
    ) -> Result<Vec<Vec<i32>>, String> {
        let n = jobs.len();
        let max_steps = jobs.iter().map(|j| j.steps).max().unwrap_or(0);
        if jobs.iter().any(|j| j.steps > MAX_DRAFT_TOKENS) {
            return Err("gemma4 draft chain exceeds its capacity".into());
        }
        let dm = model
            .draft
            .as_ref()
            .ok_or("gemma4 draft chain exceeds its capacity")?;
        if n == 0 || n > MAX_DRAFT_SESSIONS as usize {
            return Err("gemma4 draft chain exceeds its capacity".into());
        }
        let tc = &model.config;
        let c = &dm.config;
        let th = tc.hidden_size as usize;
        let hidden = c.hidden_size as usize;
        let hu = hidden as u32;
        let eps = c.rms_eps;
        let embed_scale = (th as f32).sqrt();
        let vocab = model.vocab_size as usize;
        let head_tokens = &model.draft_head_tokens;
        let head = &dm.token_embd;
        let head_rows = crate::draft_vocab::DRAFT_HEAD_ROWS;
        let small = n == 1
            && jobs[0].head.small
            && !head_tokens.is_empty()
            && gemv_format(head.ty)
            && head.rows > head_rows
            && jobs[0].head.table.len() <= crate::draft_vocab::DRAFT_TABLE_ROWS as usize;
        let table: &[i32] = if small { jobs[0].head.table } else { &[] };
        let env = Env {
            stream: &self.stream,
            half_prefill: self.half_prefill,
            has_draft: self.has_draft,
        };
        let s = env.stream;
        let key_widths = &self.key_widths;
        let derived = |layer: u32| key_widths[layer as usize] != tc.kv_dim(layer);
        let Scratch {
            partials,
            stage,
            wmma,
            ..
        } = &mut self.s;
        let d = self
            .draft
            .as_mut()
            .ok_or("gemma4 executor has no drafter scratch")?;
        // Row j drafts for job j every step; rows whose chain ended keep
        // being projected but are not attended, ranked or proposed.
        let mut active: Vec<bool> = jobs.iter().map(|j| j.steps > 0).collect();
        let mut tokens: Vec<u32> = jobs
            .iter()
            .map(|j| u32::try_from(j.token).map_err(|_| "gemma4 draft token"))
            .collect::<Result<_, _>>()?;
        let mut drafts: Vec<Vec<i32>> = jobs
            .iter()
            .map(|j| Vec::with_capacity(j.steps as usize))
            .collect();
        s.upload_indices(&tokens, &mut d.tokens).map_err(hip)?;
        let nu = n as u32;
        for step in 0..max_steps {
            if !active.iter().any(|&a| a) {
                break;
            }
            // [scaled target embedding of the token ; target-width hidden
            // state]
            unit(k::embedding_lookup(
                s,
                model.token_embd.q()?,
                d.tokens.slice(0..n),
                d.embed.slice_mut(0..n * th),
                n,
                th,
                vocab,
            ))?;
            {
                let hidden_rows: Vec<DeviceSlice<'_, f32>> = (0..n)
                    .map(|j| {
                        if step == 0 {
                            jobs[j].cache.hidden.slice(0..th)
                        } else {
                            d.next.slice(j * th..(j + 1) * th)
                        }
                    })
                    .collect();
                launched(
                    k::draft_concat(
                        s,
                        d.embed.slice(0..n * th),
                        &hidden_rows,
                        d.concat.slice_mut(0..2 * n * th),
                        embed_scale,
                        nu,
                        th as u32,
                    ),
                    "draft concat",
                )?;
            }
            project_draft(
                env,
                stage,
                wmma,
                &dm.pre_projection,
                d.concat.slice(0..2 * n * th),
                nu,
                d.x.slice_mut(0..n * hidden),
            )?;
            let first_norm = dm.layers.first().ok_or("gemma4 drafter without layers")?;
            unit(k::rms_norm(
                s,
                d.x.slice(0..n * hidden),
                Some(first_norm.attn_norm.f32()?),
                d.h.slice_mut(0..n * hidden),
                nu,
                hu,
                eps,
            ))?;
            for l in 0..c.num_layers {
                let layer = &dm.layers[l as usize];
                let sliding = c.is_sliding(l);
                let dim = c.head_dim(l);
                let q_dim = c.q_dim(l) as usize;
                let source = c.shared_kv_source(l, tc);
                project_draft(
                    env,
                    stage,
                    wmma,
                    &layer.attn_q,
                    d.h.slice(0..n * hidden),
                    nu,
                    d.q.slice_mut(0..n * q_dim),
                )?;
                for (j, job) in jobs.iter().enumerate() {
                    if !active[j] {
                        continue;
                    }
                    let cache = job.cache;
                    let rotated = if derived(source) {
                        model.global_rope_pairs
                    } else {
                        0
                    };
                    let freq = if sliding {
                        None
                    } else {
                        Some(dm.rope_factors.slice(0..(dim / 2) as usize))
                    };
                    launched(
                        k::query_post(
                            s,
                            d.q.slice_mut(j * q_dim..(j + 1) * q_dim),
                            Some(layer.attn_q_norm.f32()?),
                            c.rope_theta(l).powf(-2.0 / dim as f32),
                            freq,
                            1,
                            c.num_heads,
                            dim,
                            job.position,
                            true,
                            eps,
                            model.layers[source as usize].attn_k_norm.f32()?,
                            rotated,
                        ),
                        "draft query",
                    )?;
                    let att = Attention {
                        q: d.q.slice(j * q_dim..(j + 1) * q_dim),
                        k_cache: cache.k[source as usize].as_slice(),
                        v_cache: cache.v[source as usize].as_slice(),
                        out: Some(d.attn.slice_mut(j * q_dim..(j + 1) * q_dim)),
                        partials: partials.as_mut_slice(),
                        rows: 1,
                        heads: c.num_heads,
                        kv_heads: c.kv_heads[l as usize],
                        head_dim: dim,
                        first_position: job.position,
                        shared_position: true,
                        // Committed keys only.
                        key_limit: job.position,
                        window: if sliding { c.sliding_window } else { 0 },
                        ring: if sliding { cache.ring } else { 0 },
                        slots: cache.slots[source as usize],
                        rope_pairs: rotated,
                        key_ends: None,
                        out_half: None,
                        siblings: 0,
                        spare_key: 0,
                        pack: None,
                        pack_row0: 0,
                    };
                    attention(s, att)?;
                }
                project_draft(
                    env,
                    stage,
                    wmma,
                    &layer.attn_output,
                    d.attn.slice(0..n * q_dim),
                    nu,
                    d.o.slice_mut(0..n * hidden),
                )?;
                launched(
                    k::post_attention_norm(
                        s,
                        d.o.slice(0..n * hidden),
                        layer.post_attn_norm.f32()?,
                        d.x.slice_mut(0..n * hidden),
                        layer.ffn_norm.f32()?,
                        Some(d.h.slice_mut(0..n * hidden)),
                        nu,
                        hu,
                        eps,
                        None,
                        None,
                        None,
                    ),
                    "draft norm",
                )?;
                let ffn = c.ffn_size as usize;
                if !layer.ffn_gate_up.is_empty() {
                    project_draft(
                        env,
                        stage,
                        wmma,
                        &layer.ffn_gate_up,
                        d.h.slice(0..n * hidden),
                        nu,
                        d.gate.slice_mut(0..n * 2 * ffn),
                    )?;
                    launched(
                        k::geglu_packed(
                            s,
                            d.gate.slice(0..n * 2 * ffn),
                            Some(d.act.slice_mut(0..n * ffn)),
                            nu,
                            c.ffn_size,
                            None,
                        ),
                        "draft GeGLU",
                    )?;
                    project_draft(
                        env,
                        stage,
                        wmma,
                        &layer.ffn_down,
                        d.act.slice(0..n * ffn),
                        nu,
                        d.o.slice_mut(0..n * hidden),
                    )?;
                } else {
                    {
                        let (g, u) = d.gate.as_mut_slice().split_at(n * ffn);
                        project_draft(
                            env,
                            stage,
                            wmma,
                            &layer.ffn_gate,
                            d.h.slice(0..n * hidden),
                            nu,
                            g,
                        )?;
                        project_draft(
                            env,
                            stage,
                            wmma,
                            &layer.ffn_up,
                            d.h.slice(0..n * hidden),
                            nu,
                            u.range(0..n * ffn),
                        )?;
                    }
                    let (g, u) = d.gate.as_mut_slice().split_at(n * ffn);
                    unit(k::geglu_in_place(
                        s,
                        u.as_const().range(0..n * ffn),
                        g,
                        n * ffn,
                    ))?;
                    project_draft(
                        env,
                        stage,
                        wmma,
                        &layer.ffn_down,
                        d.gate.slice(0..n * ffn),
                        nu,
                        d.o.slice_mut(0..n * hidden),
                    )?;
                }
                let next = if l + 1 < c.num_layers {
                    dm.layers[l as usize + 1].attn_norm.f32()?
                } else {
                    dm.output_norm.f32()?
                };
                launched(
                    k::post_feed_forward_norm(
                        s,
                        d.o.slice(0..n * hidden),
                        layer.post_ffn_norm.f32()?,
                        layer.output_scale,
                        d.x.slice_mut(0..n * hidden),
                        Some(next),
                        Some(d.h.slice_mut(0..n * hidden)),
                        nu,
                        hu,
                        eps,
                        None,
                        None,
                        None,
                    ),
                    "draft norm",
                )?;
            }
            // h is the output-normalized state: vocabulary head and the
            // projection back to the target width for the next step. A lone
            // chain on the small head scores the leading ordered rows, then
            // its table rows right after them.
            let row_bytes = model.draft_row_bytes;
            let mut scored = vocab;
            let head_q = |rows: usize| -> Result<gufo_hip::QuantSlice<'_>, String> {
                Ok(gufo_hip::QuantSlice::new(
                    head.bytes()?.range(0..rows * row_bytes),
                    head.ty,
                ))
            };
            let mut done = false;
            if small
                && attempt(k::kquant_gemv(
                    s,
                    head_q(head_rows as usize)?,
                    d.h.slice(0..hidden),
                    d.logits.slice_mut(0..head_rows as usize),
                    head_rows,
                    head.cols,
                ))?
            {
                if !table.is_empty() {
                    let cache_table = jobs[0]
                        .cache
                        .draft_table
                        .as_ref()
                        .ok_or("gemma4 draft table projection failed")?;
                    let t = gufo_hip::QuantSlice::new(
                        cache_table.slice(0..table.len() * row_bytes),
                        head.ty,
                    );
                    let at = head_rows as usize;
                    if !attempt(k::kquant_gemv(
                        s,
                        t,
                        d.h.slice(0..hidden),
                        d.logits.slice_mut(at..at + table.len()),
                        table.len() as u32,
                        head.cols,
                    ))? {
                        return Err("gemma4 draft table projection failed".into());
                    }
                }
                scored = head_rows as usize + table.len();
                done = true;
            }
            if !done
                && !(n == 1
                    && gemv_format(head.ty)
                    && attempt(k::kquant_gemv(
                        s,
                        head.q()?,
                        d.h.slice(0..hidden),
                        d.logits.slice_mut(0..vocab),
                        head.rows,
                        head.cols,
                    ))?)
            {
                project(
                    env,
                    stage,
                    wmma,
                    head,
                    d.h.slice(0..n * hidden),
                    Staged::None,
                    nu,
                    d.logits.slice_mut(0..n * vocab),
                    false,
                )?;
            }
            // The host picks each proposal from its row's top-64
            // candidates; one wait serves every row.
            let count = scored.min(k::sizes::MTP_CANDIDATES);
            for (j, &running) in active.iter().enumerate() {
                if !running {
                    continue;
                }
                unit(k::mtp_top_candidates(
                    s,
                    d.logits.slice(j * vocab..j * vocab + scored),
                    d.ids[j].view_mut(),
                    d.scratch_ids[j].as_mut_slice(),
                    d.scores[j].as_mut_slice(),
                    scored as u32,
                ))?;
            }
            let mut any = false;
            for j in 0..n {
                if !active[j] {
                    continue;
                }
                let mut host = MtpCandidateLogits {
                    size: count,
                    ..MtpCandidateLogits::default()
                };
                s.download(
                    d.ids[j].view().values().range(0..count),
                    &mut host.ids[..count],
                )
                .map_err(hip)?;
                s.download(d.scores[j].slice(0..count), &mut host.logits[..count])
                    .map_err(hip)?;
                // Head rows back to token ids.
                if !head_tokens.is_empty() {
                    for id in &mut host.ids[..count] {
                        let row = *id;
                        let token = if row >= head_rows && scored != vocab {
                            table.get((row - head_rows) as usize).copied()
                        } else {
                            head_tokens.get(row as usize).copied()
                        };
                        *id = token
                            .and_then(|t| u32::try_from(t).ok())
                            .ok_or("gemma4 draft candidate outside the head")?;
                    }
                }
                let proposal = propose(j, &host)?;
                active[j] = false;
                let Some(token) = proposal.token else {
                    continue;
                };
                drafts[j].push(token);
                tokens[j] = u32::try_from(token).map_err(|_| "gemma4 draft token")?;
                if !proposal.last && step + 1 < jobs[j].steps {
                    active[j] = true;
                    any = true;
                }
            }
            if !any {
                break;
            }
            s.upload_indices(&tokens, &mut d.tokens).map_err(hip)?;
            project_draft(
                env,
                stage,
                wmma,
                &dm.post_projection,
                d.h.slice(0..n * hidden),
                nu,
                d.next.slice_mut(0..n * th),
            )?;
        }
        Ok(drafts)
    }

    /// Whether `read_candidates` is available (`CanReadCandidates`).
    pub fn can_read_candidates(&self) -> bool {
        self.candidates.is_some()
    }

    /// The `CANDIDATE_TOP + extra.len()` largest logits of each of the first
    /// `rows` logit rows of the last forward (at most `SPLIT_ROWS`) and the
    /// logits of `extra` (at most `MAX_CANDIDATE_EXTRA` tokens), on the
    /// host (`ReadCandidates`).
    pub fn read_candidates(&mut self, rows: usize, extra: &[u32]) -> Result<Candidates, String> {
        let c = self
            .candidates
            .as_mut()
            .ok_or("gemma4 candidate rows exceed capacity")?;
        if rows == 0 || rows > SPLIT_ROWS as usize || extra.len() > k::sizes::MAX_CANDIDATE_EXTRA {
            return Err("gemma4 candidate rows exceed capacity".into());
        }
        let s = &self.stream;
        let vocab = self.vocab;
        let n_extra = extra.len();
        let top = k::sizes::CANDIDATE_TOP + n_extra;
        let stride = top + n_extra;
        let logits = self.s.logits.slice(0..rows * vocab as usize);
        let out_len = 2 * rows * stride;
        if n_extra != 0 {
            s.upload_indices(extra, &mut c.extra).map_err(hip)?;
            launched(
                k::logit_candidates_extra(
                    s,
                    logits,
                    rows as u32,
                    vocab,
                    top as u32,
                    c.extra.slice(0..n_extra),
                    n_extra as u32,
                    c.work.as_mut_slice(),
                    c.counts.slice_mut(0..rows),
                    c.out.slice_mut(0..out_len),
                ),
                "logit candidates",
            )?;
        } else {
            launched(
                k::logit_candidates(
                    s,
                    logits,
                    rows as u32,
                    vocab,
                    top as u32,
                    c.work.as_mut_slice(),
                    c.counts.slice_mut(0..rows),
                    c.out.slice_mut(0..out_len),
                ),
                "logit candidates",
            )?;
        }
        let mut counts = vec![0u32; rows];
        s.download(c.counts.slice(0..rows), &mut counts)
            .map_err(hip)?;
        let mut raw = vec![0u32; out_len];
        s.download(c.out.slice(0..out_len), &mut raw).map_err(hip)?;
        let data = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| {
                let (token, value) = k::sizes::candidate(pair);
                LogitCandidate { token, value }
            })
            .collect();
        Ok(Candidates {
            counts,
            data,
            top,
            stride,
        })
    }

    /// Moves every layer's K/V of key `from` to key `to`: an accepted
    /// sibling's spare key into its position (`MoveKey`).
    pub fn move_key(
        &mut self,
        model: &DeviceModel,
        cache: &mut KvCache,
        from: u32,
        to: u32,
    ) -> Result<(), String> {
        let c = &model.config;
        if c.num_layers as usize > k::sizes::MOVE_KEY_MAX_LAYERS {
            return Err("gemma4 MoveKey layer count".into());
        }
        cache.written_from.fetch_min(to, Ordering::Relaxed);
        cache.written_to.fetch_max(to + 1, Ordering::Relaxed);
        let k_width = self.key_widths.clone();
        let v_width: Vec<u32> = (0..c.num_layers).map(|l| c.kv_dim(l)).collect();
        let ring: Vec<u32> = cache.slots.clone();
        let KvCache { k: ks, v: vs, .. } = cache;
        let mut k_slices: Vec<DeviceSliceMut<'_, F16>> =
            ks.iter_mut().map(DeviceBuffer::as_mut_slice).collect();
        let mut v_slices: Vec<DeviceSliceMut<'_, F16>> =
            vs.iter_mut().map(DeviceBuffer::as_mut_slice).collect();
        launched(
            k::move_key(
                &self.stream,
                &mut k_slices,
                &mut v_slices,
                &k_width,
                &v_width,
                &ring,
                c.num_layers,
                from,
                to,
            ),
            "MoveKey",
        )
    }
}

/// A step's draft token, or none to end the chain before it; `last` keeps
/// the token and ends the chain after it (`DraftProposal`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DraftProposal {
    pub token: Option<i32>,
    pub last: bool,
}

/// The drafter head rows a chain scores: the leading `DRAFT_HEAD_ROWS` rows
/// of the ordered head plus the session's table rows, which hold `table`'s
/// tokens; otherwise (or with more than one chain) the whole head
/// (`DraftHead`).
#[derive(Debug, Clone, Copy, Default)]
pub struct DraftHead<'a> {
    pub small: bool,
    pub table: &'a [i32],
}

/// One session's chain in `draft_chains` (`DraftJob`).
pub struct DraftJob<'a> {
    pub cache: &'a KvCache,
    pub token: i32,
    pub position: u32,
    pub steps: u32,
    pub head: DraftHead<'a>,
}

/// `logit_candidates` of the last forward's logit rows on the host
/// (`Executor::Candidates`): row r's top logits and the logits of the
/// extra tokens.
#[derive(Debug, Clone, Default)]
pub struct Candidates {
    pub counts: Vec<u32>,
    pub data: Vec<LogitCandidate>,
    pub top: usize,
    pub stride: usize,
}

impl Candidates {
    /// Whether `top(r)` holds row r's top logits; else it needs its full
    /// row.
    pub fn complete(&self, r: usize) -> bool {
        self.counts[r] != k::sizes::CANDIDATE_OVERFLOW
    }

    pub fn top(&self, r: usize) -> &[LogitCandidate] {
        &self.data[r * self.stride..r * self.stride + self.counts[r] as usize]
    }

    pub fn extra(&self, r: usize) -> &[LogitCandidate] {
        &self.data[r * self.stride + self.top..(r + 1) * self.stride]
    }
}

/// `Project` for drafter weights (`ProjectDraft`): a lone chain reads the
/// shapes the WMMA kernel would take with the Gemma GEMV.
#[allow(clippy::too_many_arguments)]
fn project_draft(
    env: Env<'_>,
    stage: &mut Stage,
    wmma: &mut DeviceBuffer<u8>,
    w: &Weight,
    x: DeviceSlice<'_, f32>,
    rows: u32,
    mut y: DeviceSliceMut<'_, f32>,
) -> Result<(), String> {
    if rows == 1
        && w.cols >= 2048
        && wmma_gemv_supports(w.ty, w.rows, w.cols)
        && gemv_format(w.ty)
        && attempt(k::kquant_gemv(
            env.stream,
            w.q()?,
            x,
            y.reborrow(),
            w.rows,
            w.cols,
        ))?
    {
        return Ok(());
    }
    project(env, stage, wmma, w, x, Staged::None, rows, y, false)
}

/// The arguments of one attention launch (`AttentionArgs`).
struct Attention<'a> {
    q: DeviceSlice<'a, f32>,
    k_cache: DeviceSlice<'a, F16>,
    v_cache: DeviceSlice<'a, F16>,
    out: Option<DeviceSliceMut<'a, f32>>,
    partials: DeviceSliceMut<'a, f32>,
    rows: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    first_position: u32,
    shared_position: bool,
    key_limit: u32,
    window: u32,
    /// Sliding layers' ring (0 for global layers).
    ring: u32,
    /// Slots of the layer's cache.
    slots: u32,
    rope_pairs: u32,
    key_ends: Option<DeviceSlice<'a, u32>>,
    out_half: Option<DeviceSliceMut<'a, F16>>,
    siblings: u32,
    spare_key: u32,
    pack: Option<DeviceSliceMut<'a, u8>>,
    pack_row0: u32,
}

/// `Attention`: the split kernels up to `SPLIT_ROWS` rows, the prefill
/// kernels past that; ring caches for sliding layers, linear caches with
/// whole or derived keys for global ones.
fn attention(s: &Stream, a: Attention<'_>) -> Result<(), String> {
    if a.siblings != 0 && (a.rows > SPLIT_ROWS || a.shared_position || 2 * a.siblings >= a.rows) {
        return Err("siblings need split attention beside drafts".into());
    }
    let split = a.rows <= SPLIT_ROWS;
    let what = "attention";
    if a.ring != 0 {
        if split {
            let out = a.out.ok_or("gemma4 split attention needs FP32 rows")?;
            return launched(
                k::attention_ring(
                    s,
                    a.q,
                    a.k_cache,
                    a.v_cache,
                    out,
                    a.partials,
                    a.rows,
                    a.heads,
                    a.kv_heads,
                    a.head_dim,
                    a.first_position,
                    a.shared_position,
                    a.key_limit,
                    a.window,
                    a.ring,
                    a.key_ends,
                    a.siblings,
                    a.spare_key,
                    a.pack,
                    a.pack_row0,
                    a.ring,
                ),
                what,
            );
        }
        return launched(
            k::attention_ring_prefill(
                s,
                a.q,
                a.k_cache,
                a.v_cache,
                a.out,
                a.rows,
                a.heads,
                a.kv_heads,
                a.head_dim,
                a.first_position,
                a.shared_position,
                a.key_limit,
                a.window,
                a.ring,
                a.key_ends,
                a.out_half,
            ),
            what,
        );
    }
    if a.rope_pairs != 0 {
        if split {
            let out = a.out.ok_or("gemma4 split attention needs FP32 rows")?;
            return launched(
                k::attention_derived(
                    s,
                    a.q,
                    a.k_cache,
                    a.v_cache,
                    out,
                    a.partials,
                    a.rows,
                    a.heads,
                    a.kv_heads,
                    a.head_dim,
                    a.first_position,
                    a.shared_position,
                    a.key_limit,
                    a.window,
                    a.rope_pairs,
                    a.siblings,
                    a.spare_key,
                    a.pack,
                    a.pack_row0,
                    a.slots,
                ),
                what,
            );
        }
        return launched(
            k::attention_derived_prefill(
                s,
                a.q,
                a.k_cache,
                a.v_cache,
                a.out,
                a.rows,
                a.heads,
                a.kv_heads,
                a.head_dim,
                a.first_position,
                a.shared_position,
                a.key_limit,
                a.window,
                a.rope_pairs,
                a.out_half,
                a.slots,
            ),
            what,
        );
    }
    if split {
        let out = a.out.ok_or("gemma4 split attention needs FP32 rows")?;
        return launched(
            k::attention_linear(
                s,
                a.q,
                a.k_cache,
                a.v_cache,
                out,
                a.partials,
                a.rows,
                a.heads,
                a.kv_heads,
                a.head_dim,
                a.first_position,
                a.shared_position,
                a.key_limit,
                a.window,
                a.siblings,
                a.spare_key,
                a.pack,
                a.pack_row0,
                a.slots,
            ),
            what,
        );
    }
    launched(
        k::attention_linear_prefill(
            s,
            a.q,
            a.k_cache,
            a.v_cache,
            a.out,
            a.rows,
            a.heads,
            a.kv_heads,
            a.head_dim,
            a.first_position,
            a.shared_position,
            a.key_limit,
            a.window,
            a.out_half,
            a.slots,
        ),
        what,
    )
}

/// Routed experts of layer `l` for the `n` rows of the attention residual
/// `x` and their expert input (`moe.h`, or binary16 `moe.x_half` in
/// prefill): routing weights in `moe.weights`, every slot's output in
/// `moe.out`, mixed by `moe_finish` (`Executor::Experts`).
fn experts(
    env: Env<'_>,
    model: &DeviceModel,
    blas: Option<&Blas>,
    m: &mut MoeScratch,
    l: &DeviceLayer,
    x: DeviceSlice<'_, f32>,
    n: u32,
) -> Result<(), String> {
    let s = env.stream;
    let c = &model.config;
    let d = c.hidden_size;
    let du = d as usize;
    let used = c.experts_used;
    let experts = c.num_experts;
    let width = c.expert_ffn_size;
    let grouped = n <= MAX_GROUP_SLOTS;
    let nu = n as usize;
    let slots = n * used;
    let su = slots as usize;
    if grouped {
        launched(
            k::moe_route(
                s,
                x,
                l.router.f32()?,
                l.down_exps_scale.f32()?,
                m.logits.slice_mut(0..nu * experts as usize),
                m.ids.slice_mut(0..su),
                m.weights.slice_mut(0..su),
                n,
                d,
                experts,
                used,
                c.rms_eps,
            ),
            "expert routing",
        )?;
    } else {
        // Prefill logits as one GEMM: logits[r][e] = router[e] . x[r].
        let blas = blas.ok_or("gemma4 router GEMM without hipBLAS")?;
        blas.sgemm(
            s,
            Op::T,
            Op::N,
            experts as usize,
            nu,
            du,
            1.0,
            l.router.f32()?,
            du,
            x,
            du,
            m.logits.slice_mut(0..nu * experts as usize),
            experts as usize,
        )
        .map_err(|e| format!("gemma4 router GEMM failed: {e}"))?;
        launched(
            k::moe_route_raw(
                s,
                x,
                l.down_exps_scale.f32()?,
                m.logits.slice_mut(0..nu * experts as usize),
                m.ids.slice_mut(0..su),
                m.weights.slice_mut(0..su),
                n,
                d,
                experts,
                used,
                c.rms_eps,
            ),
            "expert routing",
        )?;
    }
    if !grouped && prefill_experts(env, model, m, l, n)? {
        return Ok(());
    }
    // Grouped FP32 projections; a prefill whose formats lack the binary16
    // route runs them over 16-row pieces.
    let gate_up = expert_format_of(l.gate_up_exps.ty);
    let down = expert_format_of(l.down_exps.ty);
    if gate_up.is_none() || down.is_none() {
        return Err("gemma4 expert format unsupported".into());
    }
    let mut r0 = 0u32;
    while r0 < n {
        let rows = MAX_GROUP_SLOTS.min(n - r0);
        let piece = (r0 * used) as usize..((r0 + rows) * used) as usize;
        let piece_slots = rows * used;
        if !grouped {
            let xr = x.range(r0 as usize * du..(r0 + rows) as usize * du);
            let lr = r0 as usize * experts as usize..(r0 + rows) as usize * experts as usize;
            launched(
                k::moe_route(
                    s,
                    xr,
                    l.router.f32()?,
                    l.down_exps_scale.f32()?,
                    m.logits.slice_mut(lr),
                    m.ids.slice_mut(piece.clone()),
                    m.weights.slice_mut(piece.clone()),
                    rows,
                    d,
                    experts,
                    used,
                    c.rms_eps,
                ),
                "expert routing",
            )?;
        }
        ensure_groups(s, m, piece_slots, experts)?;
        let groups = m
            .groups
            .get_mut(&piece_slots)
            .ok_or("gemma4 expert groups")?;
        launched(
            k::expert_groups(
                s,
                m.ids.slice(piece.clone()),
                groups.view_mut(),
                piece_slots,
                experts,
            ),
            "expert groups",
        )?;
        let max_groups = experts.min(rows * used);
        let single = rows == 1;
        let act = m
            .act
            .as_mut_slice()
            .cast::<f32>()
            .ok_or("expert rows alignment")?
            .range(0..piece_slots as usize * width as usize);
        // The gate/up projection applies GeGLU in its epilogue.
        launched(
            k::routed_gemv_geglu(
                s,
                l.gate_up_exps.q()?,
                m.ids.slice(piece.clone()),
                groups.view(),
                max_groups,
                m.h.slice(r0 as usize * du..(r0 + rows) as usize * du),
                used,
                act,
                2 * width,
                d,
                single,
                piece_slots,
                experts,
            ),
            "expert gate/up",
        )?;
        let act = m
            .act
            .as_slice()
            .cast::<f32>()
            .ok_or("expert rows alignment")?
            .range(0..piece_slots as usize * width as usize);
        let out = (r0 * used) as usize * du..((r0 + rows) * used) as usize * du;
        launched(
            k::routed_gemv(
                s,
                l.down_exps.q()?,
                m.ids.slice(piece.clone()),
                groups.view(),
                max_groups,
                act,
                1,
                m.out.slice_mut(out),
                d,
                width,
                single,
                piece_slots,
                experts,
            ),
            "expert down",
        )?;
        r0 += rows;
    }
    Ok(())
}

/// The group table of `slots` expert slots, allocated on `stream` when
/// missing.
fn ensure_groups(
    stream: &Stream,
    m: &mut MoeScratch,
    slots: u32,
    experts: u32,
) -> Result<(), String> {
    if m.groups.contains_key(&slots) {
        return Ok(());
    }
    if m.groups.len() >= ROUTING_TABLES {
        m.groups.clear();
    }
    let table = stream
        .alloc_indices_with(
            k::sizes::expert_group_table_len(experts as usize),
            IndexSpec::below(experts + slots + 17).signed(),
        )
        .map_err(hip)?;
    m.groups.insert(slots, table);
    Ok(())
}

/// The prefill route of `experts`: binary16 expert GEMMs over rows
/// compacted by expert. `false` when a weight format lacks that route
/// (`PrefillExperts`).
fn prefill_experts(
    env: Env<'_>,
    model: &DeviceModel,
    m: &mut MoeScratch,
    l: &DeviceLayer,
    n: u32,
) -> Result<bool, String> {
    let s = env.stream;
    let c = &model.config;
    if !half_experts(l, c) {
        return Ok(false);
    }
    let d = c.hidden_size;
    let used = c.experts_used;
    let experts = c.num_experts;
    let width = c.expert_ffn_size;
    let slots = n * used;
    let su = slots as usize;
    let own_format = expert_format_of(l.gate_up_exps.ty);
    let own_up = own_gate_up(own_format);
    let gate_up_rows = if own_up {
        ROUTED_KQUANT_TILE_ROWS
    } else {
        ROUTED_TILE_ROWS
    };
    let tiles = k::sizes::routed_tile_capacity(su, experts as usize, gate_up_rows as usize) as u32;
    let down_format = expert_format_of(l.down_exps.ty);
    let own_dn = own_down(down_format);
    let down_rows = if own_dn {
        ROUTED_KQUANT_TILE_ROWS
    } else {
        ROUTED_DOWN_ROWS
    };
    let down_tiles =
        k::sizes::routed_tile_capacity(su, experts as usize, down_rows as usize) as u32;
    let compact = k::sizes::routed_compact_rows(su, experts as usize);
    // Prefill widths vary with every prompt: keep the tables of a few.
    if m.routing.len() >= ROUTING_TABLES && !m.routing.contains_key(&n) {
        m.routing.clear();
    }
    let routing = match m.routing.entry(n) {
        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
        std::collections::hash_map::Entry::Vacant(e) => e.insert(Routing {
            counts: s.alloc_indices(experts as usize, slots + 1).map_err(hip)?,
            pad_bounds: s
                .alloc_indices_with(
                    experts as usize + 1,
                    IndexSpec::below(compact as u32 + 1).signed(),
                )
                .map_err(hip)?,
            rows_token: s
                .alloc_indices_with(compact, IndexSpec::below(n).signed().padded())
                .map_err(hip)?,
            rows_slot: s
                .alloc_indices_with(compact, IndexSpec::below(slots).signed().padded())
                .map_err(hip)?,
            tiles_a: s
                .alloc_indices_with(tiles as usize, IndexSpec::below(experts).masked(0xFFFF))
                .map_err(hip)?,
            tiles_b: s
                .alloc_indices_with(
                    down_tiles as usize,
                    IndexSpec::below(experts).masked(0xFFFF),
                )
                .map_err(hip)?,
        }),
    };
    // The routing pass wrote the IDs; counts and tables derive from them.
    let ids = m.ids.slice(0..su);
    launched(
        k::expert_counts(s, ids, routing.counts.view_mut(), slots, experts),
        "expert counts",
    )?;
    let Routing {
        counts,
        pad_bounds,
        rows_token,
        rows_slot,
        tiles_a,
        tiles_b,
    } = routing;
    launched(
        k::build_routed_tiles(
            s,
            m.ids.slice(0..su),
            counts.view(),
            experts,
            gate_up_rows,
            tiles,
            tiles_a.view_mut(),
            down_rows,
            down_tiles,
            tiles_b.view_mut(),
            slots,
        ),
        "routed tiles",
    )?;
    unit(k::routed_compact(
        s,
        m.ids.slice(0..su),
        counts.view(),
        pad_bounds.view_mut(),
        m.cursors.as_mut_slice(),
        rows_token.view_mut(),
        rows_slot.view_mut(),
        n,
        used,
        experts,
        slots,
    ))?;
    // The attention norm wrote the binary16 expert input.
    let x_half = m.x_half.slice(0..n as usize * d as usize);
    let act_len = su * width as usize;
    let compact_rows = compact as u32;
    // The Gemma routed GEMM applies GeGLU in its epilogue; the F16 WMMA GEMM
    // writes [gate | up] for the separate pass.
    if own_up {
        let act = m
            .act
            .as_mut_slice()
            .cast::<F16>()
            .ok_or("expert rows alignment")?
            .range(0..act_len);
        let w = &l.gate_up_exps;
        let ok = match (w.prefill_layout, own_format) {
            (PrefillLayout::Compact, Some(ExpertFormat::Q4K)) => {
                attempt(k::routed_half_gemm_compact_q4k(
                    s,
                    w.prefill_bytes()?,
                    x_half,
                    counts.view(),
                    tiles_a.view(),
                    tiles,
                    pad_bounds.view(),
                    rows_token.view(),
                    rows_slot.view(),
                    act,
                    2 * width,
                    d,
                    experts,
                    compact_rows,
                    n,
                    slots,
                    slots,
                ))?
            }
            (PrefillLayout::Compact, Some(ExpertFormat::Q5K)) => {
                attempt(k::routed_half_gemm_compact_q5k(
                    s,
                    w.prefill_bytes()?,
                    x_half,
                    counts.view(),
                    tiles_a.view(),
                    tiles,
                    pad_bounds.view(),
                    rows_token.view(),
                    rows_slot.view(),
                    act,
                    2 * width,
                    d,
                    experts,
                    compact_rows,
                    n,
                    slots,
                    slots,
                ))?
            }
            (PrefillLayout::Compact, Some(ExpertFormat::Q6K)) => {
                attempt(k::routed_half_gemm_compact_q6k(
                    s,
                    w.prefill_bytes()?,
                    x_half,
                    counts.view(),
                    tiles_a.view(),
                    tiles,
                    pad_bounds.view(),
                    rows_token.view(),
                    rows_slot.view(),
                    act,
                    2 * width,
                    d,
                    experts,
                    compact_rows,
                    n,
                    slots,
                    slots,
                ))?
            }
            (PrefillLayout::Compact, _) => false,
            _ => attempt(k::routed_half_gemm_geglu(
                s,
                w.q()?,
                x_half,
                counts.view(),
                tiles_a.view(),
                tiles,
                pad_bounds.view(),
                rows_token.view(),
                rows_slot.view(),
                act,
                2 * width,
                d,
                experts,
                compact_rows,
                n,
                slots,
                slots,
            ))?,
        };
        if !ok {
            return Ok(false);
        }
    } else {
        let gu_len = su * 2 * width as usize;
        let gu =
            m.gu.as_mut_slice()
                .cast::<F16>()
                .ok_or("expert rows alignment")?
                .range(0..gu_len);
        if !attempt(k::routed_f16_gemm_half(
            s,
            l.gate_up_exps.q()?,
            x_half,
            counts.view(),
            tiles_a.view(),
            tiles,
            ROUTED_TILE_ROWS,
            pad_bounds.view(),
            rows_token.view(),
            rows_slot.view(),
            gu,
            2 * width as usize,
            d as usize,
            experts,
            compact,
            n as usize,
            su,
            slots,
        ))? {
            return Ok(false);
        }
        let gu =
            m.gu.as_slice()
                .cast::<F16>()
                .ok_or("expert rows alignment")?
                .range(0..gu_len);
        let act = m
            .act
            .as_mut_slice()
            .cast::<F16>()
            .ok_or("expert rows alignment")?
            .range(0..act_len);
        unit(k::geglu_packed_half(s, gu, act, slots, width))?;
    }
    let act = m
        .act
        .as_slice()
        .cast::<F16>()
        .ok_or("expert rows alignment")?
        .range(0..act_len);
    let out = m.out.slice_mut(0..su * d as usize);
    if own_dn {
        return attempt(k::routed_half_gemm(
            s,
            l.down_exps.q()?,
            act,
            counts.view(),
            tiles_b.view(),
            down_tiles,
            pad_bounds.view(),
            rows_slot.view(),
            rows_slot.view(),
            out,
            d,
            width,
            experts,
            compact_rows,
            slots,
            slots,
            slots,
        ));
    }
    attempt(k::routed_f16_gemm(
        s,
        l.down_exps.q()?,
        act,
        counts.view(),
        tiles_b.view(),
        down_tiles,
        ROUTED_DOWN_ROWS,
        pad_bounds.view(),
        rows_slot.view(),
        rows_slot.view(),
        out,
        d as usize,
        width as usize,
        experts,
        compact,
        su,
        su,
        slots,
    ))
}
