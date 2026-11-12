//! The target and its optional MTP drafter resident on the device
//! (`hip/device_model.cpp`). Weights keep their GGUF encoding, with a zeroed
//! tail every quantized kernel may over-read; projections that share an
//! input and a format are stacked into one matrix (`[q | k | v]`,
//! `[gate | up]`) with the separate tensors as views into it. After the
//! copies, BF16 projections become binary16, routers take their input scale,
//! the drafter's projections may be reordered into the grouped layout and
//! its vocabulary head requantized and sorted.
//!
//! The C++ rewrites some uploads in place; here immutable tensors are
//! replaced by new ones holding the same bytes the C++ ends with.

use std::sync::Arc;

use gufo_gemma4_kernels as k;
use gufo_hip::{
    DeviceBuffer, DeviceSlice, DeviceTensor, F16, GgmlType, HipError, Launch, LaunchError,
    MappedFile, Pending, Placement, QuantSlice, Stream, WeightLoader,
};

use crate::config::Config;
use crate::provider_options::PrefillWeights;
use crate::tokenizer::TokenId;
use crate::weights::{DraftWeights, LayerWeights, ModelWeights, TensorRef};

/// Zeroed bytes after every upload: the quantized GEMM kernels read whole
/// 256-value k-iterations and may over-read the last row (`kTailMargin`).
pub const TAIL_MARGIN: usize = 4096;

/// `PrefillLayout` (moe.hpp).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefillLayout {
    #[default]
    None = 0,
    Compact = 1,
    Decoded = 2,
}

/// `ExpertFormat` (moe.hpp); the discriminants are the kernels' values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpertFormat {
    #[default]
    Q4K = 0,
    Q5K = 1,
    Q6K = 2,
    Q8_0 = 3,
    Q5_1 = 4,
    F16 = 5,
    Q4_0 = 6,
    Q5_0 = 7,
}

/// The expert formats of the routed kernels (`ExpertFormatOf`).
pub fn expert_format_of(ty: GgmlType) -> Option<ExpertFormat> {
    match ty {
        GgmlType::Q4K => Some(ExpertFormat::Q4K),
        GgmlType::Q5K => Some(ExpertFormat::Q5K),
        GgmlType::Q6K => Some(ExpertFormat::Q6K),
        GgmlType::Q8_0 => Some(ExpertFormat::Q8_0),
        GgmlType::Q5_1 => Some(ExpertFormat::Q5_1),
        GgmlType::F16 => Some(ExpertFormat::F16),
        GgmlType::Q4_0 => Some(ExpertFormat::Q4_0),
        GgmlType::Q5_0 => Some(ExpertFormat::Q5_0),
        _ => None,
    }
}

/// Formats with a prefill copy (`PrefillFormat`).
fn prefill_format(ty: GgmlType) -> Option<ExpertFormat> {
    match ty {
        GgmlType::Q4K => Some(ExpertFormat::Q4K),
        GgmlType::Q5K => Some(ExpertFormat::Q5K),
        GgmlType::Q6K => Some(ExpertFormat::Q6K),
        GgmlType::Q4_0 => Some(ExpertFormat::Q4_0),
        GgmlType::Q5_0 => Some(ExpertFormat::Q5_0),
        GgmlType::Q8_0 => Some(ExpertFormat::Q8_0),
        _ => None,
    }
}

/// Values per stage of a prefill copy (`kStageK`).
const PREFILL_STAGE_K: u32 = 128;

/// `PrefillLayoutSupports`.
pub fn prefill_layout_supports(format: ExpertFormat, layout: PrefillLayout, k: u32) -> bool {
    let kquant = matches!(
        format,
        ExpertFormat::Q4K | ExpertFormat::Q5K | ExpertFormat::Q6K
    );
    match layout {
        PrefillLayout::Compact => kquant && k.is_multiple_of(256),
        PrefillLayout::Decoded => {
            (kquant && k.is_multiple_of(256))
                || (matches!(
                    format,
                    ExpertFormat::Q4_0 | ExpertFormat::Q5_0 | ExpertFormat::Q8_0
                ) && k.is_multiple_of(PREFILL_STAGE_K))
        }
        PrefillLayout::None => false,
    }
}

/// `PrefillWeightBytes`.
pub fn prefill_weight_bytes(format: ExpertFormat, layout: PrefillLayout, m: u32, k: u32) -> usize {
    let rows = (m as usize).div_ceil(16) * 16;
    if layout == PrefillLayout::Decoded {
        return rows * k as usize * 2;
    }
    let stride = if format == ExpertFormat::Q4K {
        k::sizes::PREFILL_COMPACT_Q4K_BYTES
    } else {
        k::sizes::PREFILL_COMPACT_Q6K_BYTES
    };
    rows * (k / PREFILL_STAGE_K) as usize * stride
}

/// `WmmaGemvSupports` (and `GroupedLayoutSupports`, the same predicate).
pub fn wmma_gemv_supports(ty: GgmlType, m: u32, k: u32) -> bool {
    matches!(
        ty,
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q4_0 | GgmlType::Q8_0
    ) && m > 0
        && k > 0
        && k.is_multiple_of(256)
        && (ty != GgmlType::Q8_0 || (m >= 2048 && k >= 4096))
}

/// Projection formats with a binary16-activation prefill GEMM
/// (`HalfPrefillFormat`).
pub fn half_prefill_format(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q4_0
            | GgmlType::Q5_0
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Q8_0
            | GgmlType::F16
    )
}

/// One weight resident on the device, still in its GGUF encoding
/// (`DeviceTensor`).
#[derive(Clone)]
pub struct Weight {
    pub data: Option<DeviceTensor<u8>>,
    /// Prefill copy (`PrefillWeights`); `data` stays the decode source.
    pub prefill: Option<DeviceTensor<u8>>,
    pub prefill_layout: PrefillLayout,
    pub prefill_format: ExpertFormat,
    /// `data` holds the grouped layout (wmma_gemv.hpp), not GGUF rows.
    pub grouped: bool,
    pub ty: GgmlType,
    pub cols: u32,
    pub rows: u32,
    /// Stacked `[rows][cols]` expert matrices.
    pub experts: u32,
    /// Where an upload's bytes land before `finish` resolves them.
    slot: Option<Slot>,
}

impl Default for Weight {
    fn default() -> Self {
        Self {
            data: None,
            prefill: None,
            prefill_layout: PrefillLayout::None,
            prefill_format: ExpertFormat::Q4K,
            grouped: false,
            ty: GgmlType::F32,
            cols: 0,
            rows: 0,
            experts: 1,
            slot: None,
        }
    }
}

fn absent() -> String {
    "gemma4 weight is not resident".to_string()
}

impl Weight {
    pub fn is_empty(&self) -> bool {
        self.data.is_none()
    }

    pub fn bytes(&self) -> Result<DeviceSlice<'_, u8>, String> {
        self.data
            .as_ref()
            .map(DeviceTensor::view)
            .ok_or_else(absent)
    }

    /// The weight in its block format.
    pub fn q(&self) -> Result<QuantSlice<'_>, String> {
        Ok(QuantSlice::new(self.bytes()?, self.ty))
    }

    /// The tensor as F32 values (norms, routers, scales).
    pub fn f32(&self) -> Result<DeviceSlice<'_, f32>, String> {
        self.bytes()?
            .cast::<f32>()
            .ok_or_else(|| "gemma4 weight is not F32".to_string())
    }

    /// Binary16 weights.
    pub fn f16(&self) -> Result<DeviceSlice<'_, F16>, String> {
        self.bytes()?
            .cast::<F16>()
            .ok_or_else(|| "gemma4 weight is not binary16".to_string())
    }

    pub fn prefill_bytes(&self) -> Result<DeviceSlice<'_, u8>, String> {
        self.prefill
            .as_ref()
            .map(DeviceTensor::view)
            .ok_or_else(absent)
    }

    fn size_bytes(&self) -> usize {
        self.data.as_ref().map_or(0, DeviceTensor::len)
    }
}

/// Bytes `[offset, offset + len)` of queued upload `index`.
#[derive(Clone, Copy)]
struct Slot {
    index: usize,
    offset: usize,
    len: usize,
}

/// `DeviceLayer`.
#[derive(Clone, Default)]
pub struct DeviceLayer {
    pub attn_norm: Weight,
    pub attn_q: Weight,
    pub attn_k: Weight,
    pub attn_v: Weight,
    pub attn_q_norm: Weight,
    pub attn_k_norm: Weight,
    pub attn_output: Weight,
    pub post_attn_norm: Weight,
    pub ffn_norm: Weight,
    pub ffn_gate: Weight,
    pub ffn_up: Weight,
    pub ffn_down: Weight,
    pub post_ffn_norm: Weight,
    pub output_scale: f32,
    /// `attn_q`, `attn_k` (and `attn_v`) back to back when they share a
    /// format: one projection whose rows are `[q | k | v]`, with the
    /// separate tensors above as views into it. Empty otherwise.
    pub attn_qkv: Weight,
    /// `ffn_gate` and `ffn_up` back to back (`[gate | up]` rows), likewise.
    pub ffn_gate_up: Weight,
    // Routed experts (`LayerWeights`); empty on dense models.
    pub router: Weight,
    pub router_scale: Weight,
    pub pre_ffn_norm_2: Weight,
    pub post_ffn_norm_1: Weight,
    pub post_ffn_norm_2: Weight,
    pub gate_up_exps: Weight,
    pub down_exps: Weight,
    pub down_exps_scale: Weight,
}

impl DeviceLayer {
    fn all_mut(&mut self) -> [&mut Weight; 23] {
        [
            &mut self.attn_norm,
            &mut self.attn_q,
            &mut self.attn_k,
            &mut self.attn_v,
            &mut self.attn_q_norm,
            &mut self.attn_k_norm,
            &mut self.attn_output,
            &mut self.post_attn_norm,
            &mut self.ffn_norm,
            &mut self.ffn_gate,
            &mut self.ffn_up,
            &mut self.ffn_down,
            &mut self.post_ffn_norm,
            &mut self.attn_qkv,
            &mut self.ffn_gate_up,
            &mut self.router,
            &mut self.router_scale,
            &mut self.pre_ffn_norm_2,
            &mut self.post_ffn_norm_1,
            &mut self.post_ffn_norm_2,
            &mut self.gate_up_exps,
            &mut self.down_exps,
            &mut self.down_exps_scale,
        ]
    }

    /// The dense projections forward passes read (fused ones and views).
    pub fn projections(&self) -> [&Weight; 9] {
        [
            &self.attn_qkv,
            &self.attn_q,
            &self.attn_k,
            &self.attn_v,
            &self.attn_output,
            &self.ffn_gate_up,
            &self.ffn_gate,
            &self.ffn_up,
            &self.ffn_down,
        ]
    }
}

/// `DeviceDraft`.
#[derive(Clone)]
pub struct DeviceDraft {
    pub config: Config,
    pub pre_projection: Weight,
    pub post_projection: Weight,
    pub token_embd: Weight,
    pub output_norm: Weight,
    pub rope_factors: DeviceTensor<f32>,
    pub layers: Vec<DeviceLayer>,
}

/// `DeviceModel`.
pub struct DeviceModel {
    pub config: Config,
    pub vocab_size: u32,
    pub token_embd: Weight,
    pub output: Weight,
    pub output_norm: Weight,
    /// Global-layer rope divisors, `[head_dim_global / 2]`.
    pub rope_factors: DeviceTensor<f32>,
    /// Leading global-layer rope pairs that turn at some position; the
    /// divisors of the others (1e30 in this checkpoint) leave them
    /// unrotated.
    pub global_rope_pairs: u32,
    pub layers: Vec<DeviceLayer>,
    pub draft: Option<DeviceDraft>,
    /// The token of each drafter head row when the rows were stored in the
    /// draft order (empty otherwise).
    pub draft_head_tokens: Vec<TokenId>,
    /// The drafter head row of each token; only with `draft_head_tokens`.
    pub draft_head_rows: Vec<u32>,
    /// Bytes of one drafter head row (a multiple of 16), with
    /// `draft_head_tokens`.
    pub draft_row_bytes: usize,
    pub resident_bytes: usize,
    /// Widest reduction dimension among the projections.
    pub max_cols: usize,
    /// Widest reduction among the binary16 dense projections (0 without
    /// any).
    pub max_half_cols: usize,
    /// Whether prefill runs on binary16 activations.
    pub half_prefill: bool,
}

fn hip(error: HipError) -> String {
    error.to_string()
}

fn launch(error: LaunchError) -> String {
    error.to_string()
}

/// Requires a launcher to have queued its work.
fn launched(result: Result<Launch, LaunchError>, what: &str) -> Result<(), String> {
    match result.map_err(launch)? {
        Launch::Launched => Ok(()),
        Launch::Unsupported => Err(format!("{what}: shape unsupported")),
    }
}

/// Weight memory: device memory, or coarse-grain managed memory when the
/// files do not fit the device heap above its dispatch reserve
/// (`WeightAllocator`).
struct Memory<'s> {
    stream: &'s Stream,
    managed: bool,
}

impl Memory<'_> {
    fn alloc<T: gufo_hip::DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<T>, String> {
        let buffer = if self.managed {
            self.stream.alloc_managed_coarse::<T>(len.max(1))
        } else {
            self.stream.alloc_persistent::<T>(len.max(1))
        };
        buffer.map_err(|e| format!("device allocation failed ({len} elements): {e}"))
    }

    /// `bytes` plus a zeroed tail.
    fn weight(&self, bytes: usize) -> Result<DeviceBuffer<u8>, String> {
        let mut buffer = self.alloc::<u8>(bytes + TAIL_MARGIN)?;
        self.stream
            .fill_bytes(buffer.slice_mut(bytes..bytes + TAIL_MARGIN), 0)
            .map_err(hip)?;
        Ok(buffer)
    }

    /// F32 values with a zeroed tail (`Uploader::Floats`).
    fn floats(&self, values: &[f32]) -> Result<DeviceTensor<f32>, String> {
        let tail = TAIL_MARGIN / 4;
        let mut buffer = self.alloc::<f32>(values.len() + tail)?;
        self.stream
            .upload(values, buffer.slice_mut(0..values.len()))
            .map_err(hip)?;
        self.stream
            .fill_bytes(buffer.slice_mut(values.len()..values.len() + tail), 0)
            .map_err(hip)?;
        let tensor = buffer.freeze(self.stream).map_err(hip)?;
        tensor
            .head(values.len())
            .ok_or_else(|| "rope factor upload".to_string())
    }
}

/// The copies of one upload (`Uploader`): tensors are queued on one loader
/// and resolved after it finished.
struct Uploader<'s> {
    loader: WeightLoader<'s>,
    files: &'s [Arc<MappedFile>],
    pending: Vec<Pending<u8>>,
    shard_base: usize,
    bytes: usize,
    max_cols: usize,
    /// Widest reduction of a binary16 dense projection.
    max_half_cols: usize,
}

impl<'s> Uploader<'s> {
    fn file(&self, t: &TensorRef) -> Result<&'s Arc<MappedFile>, String> {
        self.files
            .get(self.shard_base + t.shard as usize)
            .ok_or_else(|| format!("upload failed for {}", t.name))
    }

    /// Queues one tensor (`Copy`).
    fn copy(&mut self, t: &TensorRef) -> Result<Weight, String> {
        if t.is_empty() {
            return Ok(Weight::default());
        }
        let file = self.file(t)?;
        let size = t.size_bytes();
        let offset =
            usize::try_from(t.file_offset).map_err(|_| format!("upload failed for {}", t.name))?;
        let pending = self
            .loader
            .add_parts::<u8>(&[(file, offset, size)], TAIL_MARGIN)
            .map_err(|e| {
                format!(
                    "device allocation failed for {} ({size} bytes): {e}",
                    t.name
                )
            })?;
        self.pending.push(pending);
        self.bytes += size + TAIL_MARGIN;
        if t.ty != GgmlType::F32 {
            self.max_cols = self.max_cols.max(t.cols as usize);
        }
        Ok(Weight {
            ty: t.ty,
            cols: t.cols as u32,
            rows: t.rows as u32,
            experts: t.experts as u32,
            slot: Some(Slot {
                index: self.pending.len() - 1,
                offset: 0,
                len: size,
            }),
            ..Weight::default()
        })
    }

    /// Queues `parts` (same format and width) as one tensor, rows back to
    /// back, and sets `views` to each part (`Fuse`). Empty (nothing queued)
    /// when the parts differ.
    fn fuse(&mut self, parts: &[&TensorRef], views: &mut [&mut Weight]) -> Result<Weight, String> {
        let Some(first) = parts.first() else {
            return Ok(Weight::default());
        };
        let mut rows = 0u64;
        let mut ranges = Vec::with_capacity(parts.len());
        for t in parts {
            if t.is_empty()
                || t.ty != first.ty
                || t.cols != first.cols
                || t.experts != 1
                || t.ty == GgmlType::F32
            {
                return Ok(Weight::default());
            }
            let offset = usize::try_from(t.file_offset)
                .map_err(|_| format!("upload failed for {}", t.name))?;
            ranges.push((self.file(t)?, offset, t.size_bytes()));
            rows += t.rows;
        }
        let size: usize = ranges.iter().map(|r| r.2).sum();
        let pending = self
            .loader
            .add_parts::<u8>(&ranges, TAIL_MARGIN)
            .map_err(|e| format!("device allocation failed for {} (fused): {e}", first.name))?;
        self.pending.push(pending);
        let index = self.pending.len() - 1;
        self.bytes += size + TAIL_MARGIN;
        let mut at = 0;
        for (t, view) in parts.iter().zip(views.iter_mut()) {
            **view = Weight {
                ty: t.ty,
                cols: t.cols as u32,
                rows: t.rows as u32,
                slot: Some(Slot {
                    index,
                    offset: at,
                    len: t.size_bytes(),
                }),
                ..Weight::default()
            };
            at += t.size_bytes();
        }
        self.max_cols = self.max_cols.max(first.cols as usize);
        Ok(Weight {
            ty: first.ty,
            cols: first.cols as u32,
            rows: rows as u32,
            slot: Some(Slot {
                index,
                offset: 0,
                len: size,
            }),
            ..Weight::default()
        })
    }

    fn layer(&mut self, l: &LayerWeights) -> Result<DeviceLayer, String> {
        let mut d = DeviceLayer {
            attn_norm: self.copy(&l.attn_norm)?,
            ..DeviceLayer::default()
        };
        // Fused projections share one launch in decode and one GEMM in
        // prefill.
        if !l.attn_k.is_empty() {
            d.attn_qkv = if l.attn_v.is_empty() {
                self.fuse(&[&l.attn_q, &l.attn_k], &mut [&mut d.attn_q, &mut d.attn_k])?
            } else {
                self.fuse(
                    &[&l.attn_q, &l.attn_k, &l.attn_v],
                    &mut [&mut d.attn_q, &mut d.attn_k, &mut d.attn_v],
                )?
            };
            if d.attn_qkv.slot.is_none() && !l.attn_v.is_empty() {
                d.attn_qkv =
                    self.fuse(&[&l.attn_q, &l.attn_k], &mut [&mut d.attn_q, &mut d.attn_k])?;
                if d.attn_qkv.slot.is_some() {
                    d.attn_v = self.copy(&l.attn_v)?;
                }
            }
        }
        if d.attn_qkv.slot.is_none() {
            d.attn_q = self.copy(&l.attn_q)?;
            d.attn_k = self.copy(&l.attn_k)?;
            d.attn_v = self.copy(&l.attn_v)?;
        }
        d.attn_q_norm = self.copy(&l.attn_q_norm)?;
        d.attn_k_norm = self.copy(&l.attn_k_norm)?;
        d.attn_output = self.copy(&l.attn_output)?;
        d.post_attn_norm = self.copy(&l.post_attn_norm)?;
        d.ffn_norm = self.copy(&l.ffn_norm)?;
        d.ffn_gate_up = self.fuse(
            &[&l.ffn_gate, &l.ffn_up],
            &mut [&mut d.ffn_gate, &mut d.ffn_up],
        )?;
        if d.ffn_gate_up.slot.is_none() {
            d.ffn_gate = self.copy(&l.ffn_gate)?;
            d.ffn_up = self.copy(&l.ffn_up)?;
        }
        d.ffn_down = self.copy(&l.ffn_down)?;
        d.post_ffn_norm = self.copy(&l.post_ffn_norm)?;
        d.output_scale = l.output_scale;
        d.router = self.copy(&l.router)?;
        d.router_scale = self.copy(&l.router_scale)?;
        d.pre_ffn_norm_2 = self.copy(&l.pre_ffn_norm_2)?;
        d.post_ffn_norm_1 = self.copy(&l.post_ffn_norm_1)?;
        d.post_ffn_norm_2 = self.copy(&l.post_ffn_norm_2)?;
        d.gate_up_exps = self.copy(&l.gate_up_exps)?;
        d.down_exps = self.copy(&l.down_exps)?;
        d.down_exps_scale = self.copy(&l.down_exps_scale)?;
        // Binary16 kernels serve the BF16 projections and experts: they are
        // converted after the copies (`Half`), retyped here.
        for t in [
            &mut d.attn_qkv,
            &mut d.ffn_gate_up,
            &mut d.attn_q,
            &mut d.attn_k,
            &mut d.attn_v,
            &mut d.attn_output,
            &mut d.ffn_gate,
            &mut d.ffn_up,
            &mut d.ffn_down,
            &mut d.gate_up_exps,
            &mut d.down_exps,
        ] {
            if t.slot.is_some() && t.ty == GgmlType::Bf16 {
                t.ty = GgmlType::F16;
            }
        }
        for t in d.projections() {
            if t.slot.is_some() && t.ty == GgmlType::F16 && t.experts == 1 {
                self.max_half_cols = self.max_half_cols.max(t.cols as usize);
            }
        }
        Ok(d)
    }
}

/// Resolves a queued weight to its bytes in the loaded uploads.
fn resolve(w: &mut Weight, loaded: &[DeviceTensor<u8>]) -> Result<(), String> {
    if let Some(slot) = w.slot.take() {
        let tensor = loaded
            .get(slot.index)
            .and_then(|t| t.tail(slot.offset))
            .and_then(|t| t.head(slot.len))
            .ok_or_else(|| "weight upload index".to_string())?;
        w.data = Some(tensor);
    }
    Ok(())
}

/// Replaces `parent` by `replacement` (same layout) and re-points the views
/// into it at their old offsets.
fn replace_with_views(
    parent: &mut Weight,
    views: &mut [&mut Weight],
    replacement: DeviceTensor<u8>,
) -> Result<(), String> {
    let base = parent.data.as_ref().map_or(0, DeviceTensor::offset);
    for view in views.iter_mut() {
        if let Some(data) = view.data.as_ref() {
            let at = data.offset() - base;
            let len = data.len();
            view.data = Some(
                replacement
                    .tail(at)
                    .and_then(|t| t.head(len))
                    .ok_or_else(|| "weight view outside its tensor".to_string())?,
            );
        }
    }
    parent.data = Some(replacement);
    Ok(())
}

/// A new weight buffer holding `t`'s bytes and a zeroed tail.
fn duplicate(memory: &Memory<'_>, t: &Weight) -> Result<DeviceBuffer<u8>, String> {
    let bytes = t.bytes()?;
    let mut buffer = memory.weight(bytes.len())?;
    memory
        .stream
        .copy(bytes, buffer.slice_mut(0..bytes.len()))
        .map_err(hip)?;
    Ok(buffer)
}

/// The prefill copy of a dense projection with `k` inputs under `mode`
/// (`ChoosePrefillLayout`).
fn choose_prefill_layout(mode: PrefillWeights, format: ExpertFormat, k: u32) -> PrefillLayout {
    // Binary16 copies pay only on the 31B's 5376-wide inputs; Q8_0 takes
    // none (see the C++ for the measurements).
    let decoded = prefill_layout_supports(format, PrefillLayout::Decoded, k)
        && format != ExpertFormat::Q8_0
        && k > 4096
        && k < 8192;
    if mode == PrefillWeights::Decoded && decoded {
        return PrefillLayout::Decoded;
    }
    if mode != PrefillWeights::Off && prefill_layout_supports(format, PrefillLayout::Compact, k) {
        return PrefillLayout::Compact;
    }
    PrefillLayout::None
}

/// Packs a prefill copy of `t` (`m` rows) in `layout` (`PackPrefillWeights`).
fn pack_prefill(
    memory: &Memory<'_>,
    t: &mut Weight,
    format: ExpertFormat,
    layout: PrefillLayout,
    m: u32,
) -> Result<usize, String> {
    let s = memory.stream;
    let bytes = prefill_weight_bytes(format, layout, m, t.cols);
    let mut packed = memory.weight(bytes)?;
    let k_ = t.cols;
    let w = QuantSlice::new(t.bytes()?, t.ty);
    let what = "prefill weight packing";
    match (layout, format) {
        (PrefillLayout::Decoded, _) => {
            let out = packed
                .slice_mut(0..bytes)
                .cast::<F16>()
                .ok_or("prefill copy alignment")?;
            launched(k::pack_prefill_weights_decoded(s, w, out, m, k_), what)?;
        }
        (PrefillLayout::Compact, ExpertFormat::Q4K) => {
            launched(
                k::pack_prefill_weights_compact(s, w, packed.slice_mut(0..bytes), m, k_),
                what,
            )?;
        }
        (PrefillLayout::Compact, _) => {
            launched(
                k::pack_prefill_weights_compact_112(s, w, packed.slice_mut(0..bytes), m, k_),
                what,
            )?;
        }
        (PrefillLayout::None, _) => return Ok(0),
    }
    let tensor = packed.freeze(s).map_err(hip)?;
    t.prefill = Some(tensor.head(bytes).ok_or(what)?);
    t.prefill_layout = layout;
    t.prefill_format = format;
    Ok(bytes + TAIL_MARGIN)
}

/// Bytes of a grouped-layout format's matrix (256-value super-blocks).
fn grouped_bytes(t: &Weight) -> usize {
    let block = match t.ty {
        GgmlType::Q5K => 176,
        GgmlType::Q6K => 210,
        GgmlType::Q8_0 => 272,
        _ => 144,
    };
    t.rows as usize * (t.cols as usize / 256) * block
}

/// Reorders `t` into the grouped layout (`GroupWeights`), its views with it.
fn group(memory: &Memory<'_>, t: &mut Weight, views: &mut [&mut Weight]) -> Result<(), String> {
    let s = memory.stream;
    let bytes = grouped_bytes(t);
    let mut dst = memory.weight(bytes)?;
    let src = QuantSlice::new(t.bytes()?, t.ty);
    let out = dst.slice_mut(0..bytes);
    let what = "weight grouping";
    match t.ty {
        GgmlType::Q4K | GgmlType::Q4_0 => {
            launched(k::group_weights_144(s, src, out, t.rows, t.cols), what)?
        }
        GgmlType::Q5K => launched(k::group_weights_176(s, src, out, t.rows, t.cols), what)?,
        GgmlType::Q6K => launched(k::group_weights_210(s, src, out, t.rows, t.cols), what)?,
        GgmlType::Q8_0 => launched(k::group_weights_272(s, src, out, t.rows, t.cols), what)?,
        _ => return Err("grouping weights failed".into()),
    }
    let tensor = dst.freeze(s).map_err(hip)?;
    let tensor = tensor.head(bytes).ok_or(what)?;
    replace_with_views(t, views, tensor)?;
    t.grouped = true;
    for v in views.iter_mut() {
        v.grouped = !v.is_empty();
    }
    Ok(())
}

impl DeviceModel {
    /// Uploads the target and optionally its drafter (`DeviceModel::Upload`).
    /// `files` are the target's mapped shards, then the drafter's;
    /// `draft_order` the token of each head row when the drafter head is to
    /// be stored in that order.
    pub fn upload(
        stream: &Stream,
        weights: &ModelWeights,
        target_files: usize,
        files: &[Arc<MappedFile>],
        draft: Option<&DraftWeights>,
        draft_order: &[TokenId],
        prefill_weights: PrefillWeights,
    ) -> Result<DeviceModel, String> {
        let config = weights.config.clone();
        // The placement is decided once, from the mapped files: they hold
        // every tensor this upload copies.
        let file_bytes: usize = files.iter().map(|f| f.len()).sum();
        let (free, _) = gufo_hip::device_memory().map_err(hip)?;
        let memory = Memory {
            stream,
            managed: !gufo_hip::heap_fits_above_reserve(free as u64, file_bytes as u64),
        };
        let mut up = Uploader {
            loader: WeightLoader::new(stream, Placement::DirectCopy).managed_arenas(memory.managed),
            files,
            pending: Vec::new(),
            shard_base: 0,
            bytes: 0,
            max_cols: 0,
            max_half_cols: 0,
        };
        let mut token_embd = up.copy(&weights.token_embd)?;
        let mut output = if weights.tied_output() {
            Weight::default()
        } else {
            up.copy(&weights.output)?
        };
        let mut output_norm = up.copy(&weights.output_norm)?;
        let mut layers = Vec::with_capacity(weights.layers.len());
        for l in &weights.layers {
            layers.push(up.layer(l)?);
        }
        let mut draft_parts = None;
        if let Some(d) = draft {
            up.shard_base = target_files;
            let pre_projection = up.copy(&d.pre_projection)?;
            let post_projection = up.copy(&d.post_projection)?;
            let head = up.copy(&d.token_embd)?;
            let norm = up.copy(&d.output_norm)?;
            let mut draft_layers = Vec::with_capacity(d.layers.len());
            for l in &d.layers {
                draft_layers.push(up.layer(l)?);
            }
            draft_parts = Some((pre_projection, post_projection, head, norm, draft_layers));
        }
        let Uploader {
            loader,
            pending,
            bytes: mut resident_bytes,
            max_cols,
            max_half_cols: mut max_half,
            ..
        } = up;
        let loaded = loader
            .finish()
            .map_err(|e| format!("weight upload failed: {e}"))?;
        let loaded: Vec<DeviceTensor<u8>> = pending.into_iter().map(|p| loaded.get(p)).collect();
        resolve(&mut token_embd, &loaded)?;
        resolve(&mut output, &loaded)?;
        if weights.tied_output() {
            output = token_embd.clone();
        }
        resolve(&mut output_norm, &loaded)?;
        for layer in &mut layers {
            for w in layer.all_mut() {
                resolve(w, &loaded)?;
            }
        }
        let mut draft_model = None;
        if let (Some((mut pre, mut post, mut head, mut norm, mut draft_layers)), Some(d)) =
            (draft_parts, draft)
        {
            for w in [&mut pre, &mut post, &mut head, &mut norm] {
                resolve(w, &loaded)?;
            }
            for layer in &mut draft_layers {
                for w in layer.all_mut() {
                    resolve(w, &loaded)?;
                }
            }
            let rope = memory.floats(&d.rope_factors.values)?;
            resident_bytes += d.rope_factors.values.len() * 4 + TAIL_MARGIN;
            draft_model = Some(DeviceDraft {
                config: d.config.clone(),
                pre_projection: pre,
                post_projection: post,
                token_embd: head,
                output_norm: norm,
                rope_factors: rope,
                layers: draft_layers,
            });
        }
        let rope_factors = memory.floats(&weights.rope_factors.values)?;
        resident_bytes += weights.rope_factors.values.len() * 4 + TAIL_MARGIN;
        // A pair whose angle stays below 1e-12 rad at 2^24 positions rounds
        // to the identity in binary16 keys.
        let mut global_rope_pairs = 0u32;
        {
            let scale =
                f64::from(config.rope_theta_global).powf(-2.0 / f64::from(config.head_dim_global));
            for (i, &factor) in weights.rope_factors.values.iter().enumerate() {
                let angle = 16_777_216.0 * scale.powf(i as f64) / f64::from(factor);
                if angle >= 1e-12 {
                    global_rope_pairs = i as u32 + 1;
                }
            }
        }

        let mut half_prefill = config.has_experts();
        if !half_prefill {
            // A dense model takes binary16 prefill when every projection has
            // its GEMM; each projection input is then staged as binary16.
            let mut all = true;
            let mut widest = 0usize;
            for l in &layers {
                for t in l.projections() {
                    if !t.is_empty() {
                        all = all && half_prefill_format(t.ty);
                        widest = widest.max(t.cols as usize);
                    }
                }
            }
            half_prefill = all;
            if all {
                max_half = max_half.max(widest);
            }
        }
        if half_prefill && prefill_weights != PrefillWeights::Off {
            for l in &mut layers {
                let mut pack = |t: &mut Weight| -> Result<(), String> {
                    let Some(format) = prefill_format(t.ty) else {
                        return Ok(());
                    };
                    if t.is_empty() {
                        return Ok(());
                    }
                    let layout = choose_prefill_layout(prefill_weights, format, t.cols);
                    if layout != PrefillLayout::None {
                        resident_bytes += pack_prefill(&memory, t, format, layout, t.rows)?;
                    }
                    Ok(())
                };
                // The tensors prefill projects: fused ones, not their views.
                if !l.attn_qkv.is_empty() {
                    pack(&mut l.attn_qkv)?;
                    if l.attn_qkv.rows == l.attn_q.rows + l.attn_k.rows {
                        pack(&mut l.attn_v)?;
                    }
                } else {
                    pack(&mut l.attn_q)?;
                    pack(&mut l.attn_k)?;
                    pack(&mut l.attn_v)?;
                }
                pack(&mut l.attn_output)?;
                if !l.ffn_gate_up.is_empty() {
                    pack(&mut l.ffn_gate_up)?;
                } else {
                    pack(&mut l.ffn_gate)?;
                    pack(&mut l.ffn_up)?;
                }
                pack(&mut l.ffn_down)?;
                // Expert [gate | up] rows stack per expert: one compact
                // matrix, whole 16-row groups per expert.
                let e = &mut l.gate_up_exps;
                if let Some(format) = prefill_format(e.ty)
                    && !e.is_empty()
                    && e.rows % 16 == 0
                    && prefill_layout_supports(format, PrefillLayout::Compact, e.cols)
                {
                    let rows = e.rows * e.experts;
                    resident_bytes +=
                        pack_prefill(&memory, e, format, PrefillLayout::Compact, rows)?;
                }
            }
            stream
                .synchronize()
                .map_err(|_| "prefill weight packing failed".to_string())?;
        }
        // With a drafter every decode width runs the WMMA projections, which
        // read the grouped layout faster; only projections WmmaProjection
        // takes at every width are reordered, and a fused tensor only when
        // each view starts a 128-row group.
        if draft.is_some() && half_prefill {
            let eligible = |t: &Weight| {
                !t.is_empty()
                    && t.experts == 1
                    && t.cols >= 2048
                    && wmma_gemv_supports(t.ty, t.rows, t.cols)
            };
            let views_aligned =
                |views: &[&mut Weight]| views.iter().all(|v| v.is_empty() || v.rows % 128 == 0);
            for l in &mut layers {
                let DeviceLayer {
                    attn_qkv,
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    ffn_gate_up,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    ..
                } = l;
                let regroup = |t: &mut Weight, views: &mut [&mut Weight]| -> Result<(), String> {
                    if eligible(&*t) && views_aligned(&*views) {
                        group(&memory, t, views)?;
                    }
                    Ok(())
                };
                if !attn_qkv.is_empty() {
                    let separate_v = attn_qkv.rows == attn_q.rows + attn_k.rows;
                    if separate_v {
                        regroup(attn_qkv, &mut [attn_q, attn_k])?;
                        regroup(attn_v, &mut [])?;
                    } else {
                        regroup(attn_qkv, &mut [attn_q, attn_k, attn_v])?;
                    }
                } else {
                    regroup(attn_q, &mut [])?;
                    regroup(attn_k, &mut [])?;
                    regroup(attn_v, &mut [])?;
                }
                regroup(attn_output, &mut [])?;
                if !ffn_gate_up.is_empty() {
                    regroup(ffn_gate_up, &mut [ffn_gate, ffn_up])?;
                } else {
                    regroup(ffn_gate, &mut [])?;
                    regroup(ffn_up, &mut [])?;
                }
                regroup(ffn_down, &mut [])?;
            }
            stream
                .synchronize()
                .map_err(|_| "grouping weights failed".to_string())?;
        }
        // Every BF16 weight from 2^-17 to 65504 in magnitude is a binary16
        // value; one past the binary16 range would become infinite.
        let mut layer_sets: Vec<&mut Vec<DeviceLayer>> = vec![&mut layers];
        if let Some(d) = draft_model.as_mut() {
            layer_sets.push(&mut d.layers);
        }
        let originals: Vec<&[LayerWeights]> = std::iter::once(weights.layers.as_slice())
            .chain(draft.map(|d| d.layers.as_slice()))
            .collect();
        for (set, source) in layer_sets.into_iter().zip(originals) {
            for (l, w) in set.iter_mut().zip(source) {
                convert_halves(&memory, l, w)?;
            }
        }
        stream
            .synchronize()
            .map_err(|e| format!("weight upload failed: {e}"))?;
        // Routing reads the router with its input scale folded in.
        for l in &mut layers {
            if l.router.is_empty() {
                continue;
            }
            let mut scaled = duplicate(&memory, &l.router)?;
            let n = l.router.rows as usize * l.router.cols as usize;
            let out = scaled
                .slice_mut(0..n * 4)
                .cast::<f32>()
                .ok_or("router alignment")?;
            k::scale_router(
                stream,
                out,
                l.router_scale.f32()?,
                l.router.rows,
                l.router.cols,
            )
            .map_err(launch)?;
            let tensor = scaled.freeze(stream).map_err(hip)?;
            l.router.data = Some(tensor.head(n * 4).ok_or("router")?);
        }
        let mut draft_head_tokens = Vec::new();
        let mut draft_head_rows = Vec::new();
        let mut draft_row_bytes = 0usize;
        if let Some(d) = draft_model.as_mut() {
            // The drafter's vocabulary head only proposes tokens; read as
            // Q4_K it moves half the bytes per draft step.
            let head = &mut d.token_embd;
            if head.ty == GgmlType::Q8_0 && head.cols % 256 == 0 {
                let size = head.rows as usize * head.cols as usize / 256 * 144;
                let mut q4 = memory.weight(size)?;
                k::repack_q8_0_as_q4k(
                    stream,
                    head.q()?,
                    q4.slice_mut(0..size),
                    head.rows,
                    head.cols,
                )
                .map_err(launch)?;
                let tensor = q4.freeze(stream).map_err(hip)?;
                resident_bytes -= head.size_bytes() + TAIL_MARGIN;
                resident_bytes += size + TAIL_MARGIN;
                head.data = Some(tensor.head(size).ok_or("draft head")?);
                head.ty = GgmlType::Q4K;
            }
            // Rows in `draft_order`, so a draft step can score a leading
            // range.
            let row_bytes = crate::weights::format_of(head.ty);
            let row_bytes = if row_bytes.0 == 0 || !u64::from(head.cols).is_multiple_of(row_bytes.0)
            {
                0
            } else {
                (head.cols as u64 / row_bytes.0 * row_bytes.1) as usize
            };
            if draft_order.len() == head.rows as usize && row_bytes != 0 && row_bytes % 16 == 0 {
                let size = row_bytes * head.rows as usize;
                let mut ordered = memory.weight(size)?;
                let mut rows = vec![0u32; head.rows as usize];
                let mut order = Vec::with_capacity(draft_order.len());
                for (row, &token) in draft_order.iter().enumerate() {
                    let token = u32::try_from(token).map_err(|_| "draft order token")?;
                    *rows.get_mut(token as usize).ok_or("draft order token")? = row as u32;
                    order.push(token);
                }
                // Each row moves as its bytes viewed as floats: a copy, no
                // arithmetic.
                let mut index = stream.alloc_indices(order.len(), head.rows).map_err(hip)?;
                stream.upload_indices(&order, &mut index).map_err(hip)?;
                let dim = (row_bytes / 4) as u32;
                let src = head.bytes()?.cast::<f32>().ok_or("draft head alignment")?;
                let dst = ordered
                    .slice_mut(0..size)
                    .cast::<f32>()
                    .ok_or("draft head alignment")?;
                k::gather_rows(stream, src, index.view(), dst, head.rows, dim, head.rows)
                    .map_err(launch)?;
                let tensor = ordered.freeze(stream).map_err(hip)?;
                head.data = Some(tensor.head(size).ok_or("draft head")?);
                draft_head_tokens = draft_order.to_vec();
                draft_head_rows = rows;
                draft_row_bytes = row_bytes;
            }
        }
        stream
            .synchronize()
            .map_err(|_| "router scaling failed".to_string())?;
        Ok(DeviceModel {
            vocab_size: weights.vocab_size,
            config,
            token_embd,
            output,
            output_norm,
            rope_factors,
            global_rope_pairs,
            layers,
            draft: draft_model,
            draft_head_tokens,
            draft_head_rows,
            draft_row_bytes,
            resident_bytes,
            max_cols,
            max_half_cols: max_half,
            half_prefill,
        })
    }

    pub fn has_draft(&self) -> bool {
        self.draft.is_some()
    }
}

/// Rewrites the layer's BF16 projections and experts as binary16 (`Half`
/// and the conversion pass of `Upload`): fused tensors convert once with
/// their views.
fn convert_halves(
    memory: &Memory<'_>,
    d: &mut DeviceLayer,
    l: &LayerWeights,
) -> Result<(), String> {
    let bf16 = |t: &TensorRef| !t.is_empty() && t.ty == GgmlType::Bf16;
    let DeviceLayer {
        attn_qkv,
        attn_q,
        attn_k,
        attn_v,
        attn_output,
        ffn_gate_up,
        ffn_gate,
        ffn_up,
        ffn_down,
        gate_up_exps,
        down_exps,
        ..
    } = d;
    let convert = |t: &mut Weight, views: &mut [&mut Weight], name: &str| -> Result<(), String> {
        if t.is_empty() {
            return Ok(());
        }
        let s = memory.stream;
        let count = t.rows as usize * t.cols as usize * t.experts as usize;
        let mut buffer = duplicate(memory, t)?;
        let mut overflow = s.alloc::<u32>(1).map_err(hip)?;
        s.fill_bytes(overflow.as_mut_slice(), 0).map_err(hip)?;
        k::convert_bf16_to_half(
            s,
            buffer.slice_mut(0..count * 2),
            count,
            overflow.as_mut_slice(),
        )
        .map_err(launch)?;
        let mut counted = [0u32];
        s.download(overflow.as_slice(), &mut counted).map_err(hip)?;
        if counted[0] != 0 {
            return Err(format!(
                "tensor {name} has {} BF16 values outside the binary16 range",
                counted[0]
            ));
        }
        let tensor = buffer.freeze(s).map_err(hip)?;
        let tensor = tensor.head(count * 2).ok_or("binary16 conversion")?;
        replace_with_views(t, views, tensor)
    };
    // A fused tensor of BF16 parts was retyped at upload.
    if !attn_qkv.is_empty() && bf16(&l.attn_q) {
        convert(
            attn_qkv,
            &mut [&mut *attn_q, &mut *attn_k, &mut *attn_v],
            &l.attn_q.name,
        )?;
        // A separate V of a fused [q | k] converts on its own.
        if attn_qkv.rows == attn_q.rows + attn_k.rows && bf16(&l.attn_v) {
            convert(attn_v, &mut [], &l.attn_v.name)?;
        }
    } else {
        for (t, source) in [
            (attn_q, &l.attn_q),
            (attn_k, &l.attn_k),
            (attn_v, &l.attn_v),
        ] {
            if bf16(source) {
                convert(t, &mut [], &source.name)?;
            }
        }
    }
    if !ffn_gate_up.is_empty() && bf16(&l.ffn_gate) {
        convert(
            ffn_gate_up,
            &mut [&mut *ffn_gate, &mut *ffn_up],
            &l.ffn_gate.name,
        )?;
    } else {
        for (t, source) in [(ffn_gate, &l.ffn_gate), (ffn_up, &l.ffn_up)] {
            if bf16(source) {
                convert(t, &mut [], &source.name)?;
            }
        }
    }
    for (t, source) in [
        (attn_output, &l.attn_output),
        (ffn_down, &l.ffn_down),
        (gate_up_exps, &l.gate_up_exps),
        (down_exps, &l.down_exps),
    ] {
        if bf16(source) {
            convert(t, &mut [], &source.name)?;
        }
    }
    Ok(())
}
