//! NVIDIA's LiLiCorr DFlash drafter (`drafter.cpp`): the drafter weights on
//! the device, the per-sequence context (the drafter layers' K/V rings over
//! the trunk's context and the projected rows the LiLiCorr anchor reads),
//! the block-diffusion pass that drafts `block_size - 1` tokens and the
//! LiLiCorr head that scores their candidates jointly.
//!
//! A [`DraftContext`] owns no stream or BLAS handle: every call that
//! enqueues work takes the stream and handle of the session that owns the
//! context, and the context's buffers live on that stream. The quantized
//! products need `mmq_bind_stream` on the stream first (the session binds
//! it when it creates the stream).

use std::path::Path;
use std::sync::Arc;

use gufo_hip::blas::{Blas, Op};
use gufo_hip::{
    Bf16, DeviceBuffer, DeviceCopy, DeviceIndices, DeviceSlice, DeviceSliceMut, F16, GgmlType,
    HipError, Launch, LaunchError, MappedFile, PinnedBuffer, Stream,
};
use gufo_nemotron_kernels as k;
use gufo_nemotron_kernels::sizes::{GEMV_MAX_TOKENS, LILICORR_TOP_K};

use crate::config::{Config, DrafterConfig};
use crate::device_model::{DeviceDrafter, DeviceModel, DeviceWeight};
use crate::weights::{DrafterWeights, GgufShard};

/// Projected context rows kept for the LiLiCorr anchor, by position.
const ANCHOR_ROWS: usize = 16;

/// The bound of the position index buffers (`first`, `pos0`): the kernels
/// that index the rings by position bound it by a contract scalar
/// `positions` of at most 2^30 (their int position arithmetic), and each
/// launch passes `positions = POSITION_BOUND + n - 1`, so every position
/// below this bound is valid for blocks of up to 4096 rows.
pub const POSITION_BOUND: u32 = (1 << 30) - 4096;

/// `Activation::kNone` and `Activation::kSilu` of `ScaleBias`.
const ACT_NONE: u32 = 0;
const ACT_SILU: u32 = 1;

/// Rows of each layer's K and V ring: the window plus room for the blocks
/// drafted beyond it.
fn ring_rows(c: &DrafterConfig) -> usize {
    (c.sliding_window + 4 * c.block_size) as usize
}

fn hip(error: HipError) -> String {
    error.to_string()
}

/// A launch that must run: an unsupported geometry is an error.
fn ran(result: Result<Launch, LaunchError>, what: &str) -> Result<(), String> {
    match result {
        Ok(Launch::Launched) => Ok(()),
        Ok(Launch::Unsupported) => Err(format!("{what}: unsupported geometry")),
        Err(error) => Err(format!("{what}: {error}")),
    }
}

fn f32_of(w: &DeviceWeight) -> Result<DeviceSlice<'_, f32>, String> {
    w.f32()
        .ok_or_else(|| "drafter tensor is not F32".to_string())
}

/// Reallocates on `stream` when `count` exceeds the capacity.
fn alloc<T: DeviceCopy>(stream: &Stream, count: usize) -> Result<DeviceBuffer<T>, String> {
    stream
        .alloc::<T>(count.max(1))
        .map_err(|_| "could not allocate drafter memory".to_string())
}

/// The LiLiCorr DFlash drafter: its configuration and device weights. It
/// has no embedding or head of its own and uses the trunk's.
pub struct Drafter {
    weights: DrafterWeights,
    device: Arc<DeviceDrafter>,
    /// Keeps the mapping the weights were read from alive.
    _file: Arc<MappedFile>,
}

impl Drafter {
    /// Opens the drafter GGUF at `path` for the trunk `target` and uploads
    /// it (`Drafter::Load`).
    pub fn load(path: &Path, target: &Config) -> Result<Self, String> {
        let file =
            MappedFile::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
        let shard = GgufShard::parse(file.bytes())
            .map_err(|error| format!("open {}: {error}", path.display()))?;
        let weights = DrafterWeights::bind(&[shard], target)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let stream = Stream::new().map_err(hip)?;
        let device = DeviceDrafter::upload(&weights, std::slice::from_ref(&file), &stream)?;
        stream.synchronize().map_err(hip)?;
        Ok(Self {
            weights,
            device: Arc::new(device),
            _file: file,
        })
    }

    pub fn config(&self) -> &DrafterConfig {
        &self.weights.config
    }

    pub fn resident_bytes(&self) -> usize {
        self.device.resident_bytes
    }

    pub fn device(&self) -> &DeviceDrafter {
        &self.device
    }
}

/// The LiLiCorr lattice of one drafted block: per drafted slot its top
/// candidates in descending drafter logit, and the pairwise scores.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DraftLattice {
    pub slots: u32,
    pub top_k: u32,
    /// `[slots][top_k]`.
    pub candidates: Vec<i32>,
    /// `top_k` scores of the first slot's candidates against the anchor,
    /// then `[slots - 1][top_k previous][top_k next]` transition scores.
    pub scores: Vec<f32>,
}

impl DraftLattice {
    /// llama.cpp's greedy walk: the best first candidate, then at each slot
    /// the best successor of the previous choice; at most `max_tokens`.
    pub fn walk(&self, max_tokens: u32) -> Vec<i32> {
        self.walk_with(max_tokens, None)
    }

    /// `walk`, also giving each walked transition's softmax probability
    /// among its slot's candidates.
    pub fn walk_with_probabilities(
        &self,
        max_tokens: u32,
        probabilities: &mut Vec<f32>,
    ) -> Vec<i32> {
        self.walk_with(max_tokens, Some(probabilities))
    }

    fn walk_with(&self, max_tokens: u32, mut probabilities: Option<&mut Vec<f32>>) -> Vec<i32> {
        if let Some(p) = probabilities.as_deref_mut() {
            p.clear();
        }
        let top_k = self.top_k as usize;
        let n = max_tokens.min(self.slots) as usize;
        let mut out = Vec::with_capacity(n);
        let mut prev = 0usize;
        for s in 0..n {
            let start = if s == 0 {
                0
            } else {
                top_k + ((s - 1) * top_k + prev) * top_k
            };
            let row = &self.scores[start..start + top_k];
            // std::max_element: the first of equal maxima.
            let mut best = 0usize;
            for (index, &value) in row.iter().enumerate() {
                if value > row[best] {
                    best = index;
                }
            }
            prev = best;
            out.push(self.candidates[s * top_k + prev]);
            if let Some(p) = probabilities.as_deref_mut() {
                // F32 sums of std::exp(float) in candidate order.
                let mut sum = 0.0f32;
                for &value in row {
                    sum += (value - row[prev]).exp();
                }
                p.push(1.0 / sum);
            }
        }
        out
    }
}

/// Per-sequence drafter state on the device: the drafter layers' K/V rings
/// over the trunk's context, the projected context rows the LiLiCorr anchor
/// reads, and scratch. Work runs on the owner's stream.
pub struct DraftContext {
    config: DrafterConfig,
    weights: Arc<DeviceDrafter>,
    /// The trunk's embedding and head, which the drafter uses.
    token_embd: DeviceWeight,
    output: DeviceWeight,
    max_rows: usize,
    // Rings over the trunk context, per drafter layer.
    k_ring: Vec<DeviceBuffer<F16>>,
    v_ring: Vec<DeviceBuffer<F16>>,
    /// `[ANCHOR_ROWS][hidden]` projected context rows.
    anchors: DeviceBuffer<f32>,
    ring: usize,
    // Injection scratch, `max_rows` rows.
    /// Host-injected features only.
    features: Option<DeviceBuffer<f32>>,
    /// fc + enc_norm output.
    context: DeviceBuffer<f32>,
    context_bf16: DeviceBuffer<Bf16>,
    ck: DeviceBuffer<f32>,
    cv: DeviceBuffer<f32>,
    /// The injection's first position and its host staging.
    first: DeviceIndices,
    first_host: PinnedBuffer<u32>,
    /// The copy the in-place norms read (`RmsNorm(x, w, x)` of the C++).
    norm_scratch: DeviceBuffer<f32>,
    // Block scratch.
    tokens: DeviceIndices,
    tokens_host: PinnedBuffer<u32>,
    pos0: DeviceIndices,
    pos0_host: PinnedBuffer<u32>,
    x: DeviceBuffer<f32>,
    h: DeviceBuffer<f32>,
    hc: DeviceBuffer<f32>,
    dynamic: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    ctx: DeviceBuffer<f32>,
    o: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
    hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    // LiLiCorr scratch.
    cand: DeviceBuffer<i32>,
    /// `cand` as embedding indices: the top-k kernel's IDs are checked on
    /// the host before the embedding reads them.
    cand_indices: DeviceIndices,
    cand_host: PinnedBuffer<u32>,
    values: DeviceBuffer<f32>,
    lse: DeviceBuffer<f32>,
    feat5: DeviceBuffer<f32>,
    feat_up: DeviceBuffer<f32>,
    feat: DeviceBuffer<f32>,
    embd: DeviceBuffer<f32>,
    token: DeviceBuffer<f32>,
    pass: DeviceBuffer<f32>,
    nodes: DeviceBuffer<f32>,
    anchor: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    att: DeviceBuffer<f32>,
    proj: DeviceBuffer<f32>,
    mlp: DeviceBuffer<f32>,
    factor_in: DeviceBuffer<f32>,
    factors: DeviceBuffer<f32>,
    out_head: DeviceBuffer<f32>,
    in_head: DeviceBuffer<f32>,
    anchor_head: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    scores_len: usize,
}

/// NVFP4, Q8_0 or IQ2_XXS weights through the quantized kernels:
/// `out = x W^T` over `n` rows.
fn quant(
    stream: &Stream,
    w: &DeviceWeight,
    x: DeviceSlice<'_, f32>,
    out: DeviceSliceMut<'_, f32>,
    n: usize,
    what: &str,
) -> Result<(), String> {
    let (m, cols) = (w.rows, w.cols);
    let launched = match w.ty {
        GgmlType::Q8_0 => k::mmq_dense_q8_0(stream, w.bytes(), x, out, m, n, cols),
        GgmlType::Nvfp4 => k::mmq_dense_nvfp4(stream, w.bytes(), x, out, m, n, cols),
        GgmlType::Iq2Xxs => k::mmq_dense_iq2_xxs(stream, w.bytes(), x, out, m, n, cols),
        _ => return Err(format!("quantized matrix product failed: {what}")),
    };
    match launched {
        Ok(Launch::Launched) => Ok(()),
        _ => Err(format!("quantized matrix product failed: {what}")),
    }
}

/// F32 weights: `out = x W^T` over `n` rows (hipBLAS).
fn dense_f32(
    stream: &Stream,
    blas: &Blas,
    w: &DeviceWeight,
    x: DeviceSlice<'_, f32>,
    out: DeviceSliceMut<'_, f32>,
    n: usize,
) -> Result<(), String> {
    blas.sgemm(
        stream,
        Op::T,
        Op::N,
        w.rows,
        n,
        w.cols,
        1.0,
        f32_of(w)?,
        w.cols,
        x,
        w.cols,
        out,
        w.rows,
    )
    .map_err(|_| "F32 GEMM failed".to_string())
}

/// BF16 weights: a streaming kernel up to `GEMV_MAX_TOKENS` rows, hipBLAS
/// with BF16 activations beyond.
fn dense_bf16(
    stream: &Stream,
    blas: &Blas,
    w: &DeviceWeight,
    x: DeviceSlice<'_, f32>,
    out: DeviceSliceMut<'_, f32>,
    n: usize,
    x_bf16: Option<&mut DeviceBuffer<Bf16>>,
) -> Result<(), String> {
    let wb = w
        .bytes()
        .cast::<Bf16>()
        .ok_or_else(|| "drafter tensor is not BF16".to_string())?;
    if n <= GEMV_MAX_TOKENS {
        return ran(
            k::gemv_bf16(stream, wb, x, out, w.rows, w.cols, n),
            "gemv_bf16",
        );
    }
    let x_bf16 = x_bf16.ok_or_else(|| "BF16 GEMM failed".to_string())?;
    let count = n * w.cols;
    ran(
        k::to_bf16(stream, x, x_bf16.slice_mut(0..count), count),
        "to_bf16",
    )?;
    blas.gemm(
        stream,
        Op::T,
        Op::N,
        w.rows,
        n,
        w.cols,
        wb,
        w.cols,
        x_bf16.slice(0..count),
        w.cols,
        out,
        w.rows,
    )
    .map_err(|_| "BF16 GEMM failed".to_string())
}

/// `RmsNorm(x, w, x)`: normalizes `rows` rows of `width` in place by reading
/// a copy, which gives what the kernel gives in place.
#[allow(clippy::too_many_arguments)]
fn rms_norm_in_place(
    stream: &Stream,
    scratch: &mut DeviceBuffer<f32>,
    x: DeviceSliceMut<'_, f32>,
    w: DeviceSlice<'_, f32>,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<(), String> {
    let count = rows * width;
    stream
        .copy(x.as_const().range(0..count), scratch.slice_mut(0..count))
        .map_err(hip)?;
    ran(
        k::rms_norm(stream, scratch.slice(0..count), w, x, rows, width, eps),
        "rms_norm",
    )
}

impl DraftContext {
    /// The context of one session for `drafter` over `trunk`, with buffers
    /// on `stream`; `max_rows` bounds one inject call (`DraftContext::Create`).
    pub fn create(
        stream: &Stream,
        _blas: &Blas,
        drafter: &Drafter,
        trunk: &DeviceModel,
        max_rows: usize,
    ) -> Result<Self, String> {
        let c = drafter.config().clone();
        // The top-k and LiLiCorr kernels have a fixed candidate width.
        if c.lc_top_k as usize != LILICORR_TOP_K {
            return Err("the drafter's candidate count differs from the kernels'".into());
        }
        let max_rows = max_rows.max(1);
        let hidden = c.hidden_size as usize;
        let r = max_rows;
        let t = c.block_size as usize;
        let s = c.slots() as usize;
        let n = s * c.lc_top_k as usize;
        let l = c.lc_hidden as usize;
        let kv = c.kv_dim() as usize;
        let q_dim = c.q_dim() as usize;
        let ff = c.ff_size as usize;
        let vocab = trunk.output.rows;
        let embd_rows = u32::try_from(trunk.token_embd.rows)
            .map_err(|_| "could not allocate drafter memory".to_string())?;
        let ring = ring_rows(&c);
        let mut k_ring = Vec::with_capacity(c.num_layers as usize);
        let mut v_ring = Vec::with_capacity(c.num_layers as usize);
        for _ in 0..c.num_layers {
            k_ring.push(alloc::<F16>(stream, ring * kv)?);
            v_ring.push(alloc::<F16>(stream, ring * kv)?);
        }
        let indices = |len: usize, bound: u32| -> Result<DeviceIndices, String> {
            let mut spec = gufo_hip::IndexSpec::below(bound);
            spec.signed = true;
            stream
                .alloc_indices_with(len, spec)
                .map_err(|_| "could not allocate drafter memory".to_string())
        };
        let pinned = |len: usize| -> Result<PinnedBuffer<u32>, String> {
            PinnedBuffer::zeroed(len).map_err(|_| "could not allocate drafter memory".to_string())
        };
        // Mask tokens fill the block after its anchor.
        let mut tokens_host = pinned(t)?;
        tokens_host
            .host_mut()
            .map_err(|_| "could not initialize the drafter block".to_string())?
            .fill(c.mask_token);
        let scratch = (r * hidden).max(r * kv).max(t * q_dim).max(l);
        let scores_len = c.lc_top_k as usize + (s - 1) * (c.lc_top_k * c.lc_top_k) as usize;
        Ok(Self {
            weights: Arc::clone(&drafter.device),
            token_embd: trunk.token_embd.clone(),
            output: trunk.output.clone(),
            max_rows,
            k_ring,
            v_ring,
            anchors: alloc(stream, ANCHOR_ROWS * hidden)?,
            ring,
            features: None,
            context: alloc(stream, r * hidden)?,
            context_bf16: alloc(stream, r * hidden)?,
            ck: alloc(stream, r * kv)?,
            cv: alloc(stream, r * kv)?,
            first: indices(1, POSITION_BOUND)?,
            first_host: pinned(1)?,
            norm_scratch: alloc(stream, scratch)?,
            tokens: indices(t, embd_rows)?,
            tokens_host,
            pos0: indices(1, POSITION_BOUND)?,
            pos0_host: pinned(1)?,
            x: alloc(stream, t * hidden)?,
            h: alloc(stream, t * hidden)?,
            hc: alloc(stream, t * hidden)?,
            dynamic: alloc(stream, t * c.conv_projection() as usize)?,
            q: alloc(stream, t * q_dim)?,
            k: alloc(stream, t * kv)?,
            v: alloc(stream, t * kv)?,
            ctx: alloc(stream, t * q_dim)?,
            o: alloc(stream, t * hidden)?,
            gate: alloc(stream, t * ff)?,
            up: alloc(stream, t * ff)?,
            act: alloc(stream, t * ff)?,
            hidden: alloc(stream, t * hidden)?,
            logits: alloc(stream, s * vocab)?,
            cand: alloc(stream, n)?,
            cand_indices: indices(n, embd_rows)?,
            cand_host: pinned(n)?,
            values: alloc(stream, n)?,
            lse: alloc(stream, s)?,
            feat5: alloc(stream, n * 5)?,
            feat_up: alloc(stream, n * l)?,
            feat: alloc(stream, n * l)?,
            embd: alloc(stream, n * hidden)?,
            token: alloc(stream, n * l)?,
            pass: alloc(stream, s * l)?,
            nodes: alloc(stream, n * l)?,
            anchor: alloc(stream, l)?,
            normed: alloc(stream, n * l)?,
            qkv: alloc(stream, n * 3 * l)?,
            att: alloc(stream, n * l)?,
            proj: alloc(stream, n * l)?,
            mlp: alloc(stream, n * c.lc_mlp as usize)?,
            factor_in: alloc(stream, n * 3 * l)?,
            factors: alloc(stream, n * l)?,
            out_head: alloc(stream, n * c.lc_factor as usize)?,
            in_head: alloc(stream, n * c.lc_factor as usize)?,
            anchor_head: alloc(stream, c.lc_factor as usize)?,
            scores: alloc(stream, scores_len)?,
            scores_len,
            config: c,
        })
    }

    pub fn config(&self) -> &DrafterConfig {
        &self.config
    }

    /// Enqueues the projection of `n` rows of trunk features (device memory,
    /// `feature_size` floats each) at positions `first..` into the rings and
    /// the anchor rows (`DraftContext::Inject`).
    pub fn inject(
        &mut self,
        stream: &Stream,
        blas: &Blas,
        features: DeviceSlice<'_, f32>,
        first: u32,
        n: usize,
    ) -> Result<(), String> {
        if n == 0 {
            return Ok(());
        }
        if n > self.max_rows {
            return Err("drafter injection exceeds its row budget".into());
        }
        self.upload_position(stream, first, true)
            .map_err(|error| format!("drafter injection: {error}"))?;
        self.enqueue_inject(stream, blas, features, n)
            .map_err(|error| format!("drafter injection: {error}"))
    }

    /// Host features, for validation: uploads and injects them at `first`
    /// and waits (`DraftContext::InjectHost`).
    pub fn inject_host(
        &mut self,
        stream: &Stream,
        blas: &Blas,
        features: &[f32],
        first: u32,
    ) -> Result<(), String> {
        let width = self.config.feature_size() as usize;
        if !features.len().is_multiple_of(width) {
            return Err("feature rows must hold FeatureSize floats".into());
        }
        let n = features.len() / width;
        if self
            .features
            .as_ref()
            .is_none_or(|buffer| buffer.len() < features.len())
        {
            self.features = Some(
                stream
                    .alloc::<f32>(features.len().max(1))
                    .map_err(|_| "could not allocate the feature buffer".to_string())?,
            );
        }
        let mut buffer = self
            .features
            .take()
            .ok_or_else(|| "could not allocate the feature buffer".to_string())?;
        let result = (|| {
            stream
                .upload(features, buffer.slice_mut(0..features.len()))
                .map_err(|_| "could not upload features".to_string())?;
            self.inject(stream, blas, buffer.slice(0..features.len()), first, n)?;
            stream
                .synchronize()
                .map_err(|_| "drafter injection failed".to_string())
        })();
        self.features = Some(buffer);
        result
    }

    /// Writes `first` (or the block's `pos0`) to its device index buffer.
    fn upload_position(
        &mut self,
        stream: &Stream,
        position: u32,
        inject: bool,
    ) -> Result<(), String> {
        if position >= POSITION_BOUND {
            return Err("position outside the drafter's range".into());
        }
        let (host, device) = if inject {
            (&mut self.first_host, &mut self.first)
        } else {
            (&mut self.pos0_host, &mut self.pos0)
        };
        host.host_mut().map_err(hip)?[0] = position;
        stream.upload_indices_async(host, 1, device).map_err(hip)
    }

    fn enqueue_inject(
        &mut self,
        stream: &Stream,
        blas: &Blas,
        features: DeviceSlice<'_, f32>,
        n: usize,
    ) -> Result<(), String> {
        let c = &self.config;
        let w = &*self.weights;
        let hidden = c.hidden_size as usize;
        let kv = c.kv_dim() as usize;
        let kv_heads = c.num_kv_heads as usize;
        let head_dim = c.head_dim as usize;
        let positions = POSITION_BOUND as usize + n - 1;
        let first_i32 = self
            .first
            .view()
            .values()
            .cast::<i32>()
            .ok_or_else(|| "drafter position buffer".to_string())?;
        quant(
            stream,
            &w.fc,
            features,
            self.context.slice_mut(0..n * hidden),
            n,
            "fc",
        )?;
        ran(
            k::scale_bias(
                stream,
                self.context.slice_mut(0..n * hidden),
                w.fc.scale,
                None,
                n,
                hidden,
                ACT_NONE,
            ),
            "scale_bias",
        )?;
        rms_norm_in_place(
            stream,
            &mut self.norm_scratch,
            self.context.slice_mut(0..n * hidden),
            f32_of(&w.enc_norm)?,
            n,
            hidden,
            c.rms_eps,
        )?;
        ran(
            k::rows_to_ring(
                stream,
                self.context.slice(0..n * hidden),
                self.anchors.as_mut_slice(),
                self.first.view(),
                n,
                hidden,
                ANCHOR_ROWS,
                positions,
            ),
            "rows_to_ring",
        )?;
        for (il, l) in w.layers.iter().enumerate() {
            dense_bf16(
                stream,
                blas,
                &l.attn_k,
                self.context.slice(0..n * hidden),
                self.ck.slice_mut(0..n * kv),
                n,
                Some(&mut self.context_bf16),
            )?;
            dense_bf16(
                stream,
                blas,
                &l.attn_v,
                self.context.slice(0..n * hidden),
                self.cv.slice_mut(0..n * kv),
                n,
                Some(&mut self.context_bf16),
            )?;
            rms_norm_in_place(
                stream,
                &mut self.norm_scratch,
                self.ck.slice_mut(0..n * kv),
                f32_of(&l.attn_k_norm)?,
                n * kv_heads,
                head_dim,
                c.rms_eps,
            )?;
            ran(
                k::rope_neox(
                    stream,
                    self.ck.slice_mut(0..n * kv),
                    first_i32,
                    n,
                    kv_heads,
                    head_dim,
                    c.rope_theta,
                ),
                "rope_neox",
            )?;
            ran(
                k::ring_append(
                    stream,
                    self.ck.slice(0..n * kv),
                    self.cv.slice(0..n * kv),
                    self.k_ring[il].as_mut_slice(),
                    self.v_ring[il].as_mut_slice(),
                    self.first.view(),
                    n,
                    kv,
                    self.ring,
                    positions,
                ),
                "ring_append",
            )?;
        }
        Ok(())
    }

    /// Drafts the block that starts with `anchor` at position `pos0`
    /// (`pos0 >= 1`; the rings must hold the features before `pos0`) and
    /// waits for it (`DraftContext::Draft`). `logits`, when given, receives
    /// the drafted slots' logits (`slots * vocab`).
    pub fn draft(
        &mut self,
        stream: &Stream,
        blas: &Blas,
        anchor: i32,
        pos0: u32,
        logits: Option<&mut [f32]>,
    ) -> Result<DraftLattice, String> {
        let s = self.config.slots() as usize;
        let vocab = self.output.rows;
        if pos0 == 0 {
            return Err("drafting needs a context before the anchor".into());
        }
        if logits.as_ref().is_some_and(|l| l.len() != s * vocab) {
            return Err("draft logits must hold one row per slot".into());
        }
        self.enqueue_draft(stream, blas, anchor, pos0, logits)
            .map_err(|error| format!("draft: {error}"))
    }

    fn enqueue_draft(
        &mut self,
        stream: &Stream,
        blas: &Blas,
        anchor: i32,
        pos0: u32,
        logits: Option<&mut [f32]>,
    ) -> Result<DraftLattice, String> {
        let token =
            u32::try_from(anchor).map_err(|_| "anchor outside the vocabulary".to_string())?;
        self.tokens_host.host_mut().map_err(hip)?[0] = token;
        let t = self.config.block_size as usize;
        stream
            .upload_indices_async(&mut self.tokens_host, t, &mut self.tokens)
            .map_err(hip)?;
        self.upload_position(stream, pos0, false)?;
        let c = &self.config;
        let w = &*self.weights;
        let s = c.slots() as usize;
        let top_k = c.lc_top_k as usize;
        let n = s * top_k;
        let hidden = c.hidden_size as usize;
        let l_width = c.lc_hidden as usize;
        let vocab = self.output.rows;
        let embd_rows = self.token_embd.rows;
        let th = t * hidden;
        let heads = c.num_heads as usize;
        let kv_heads = c.num_kv_heads as usize;
        let hd = c.head_dim as usize;
        let kv = c.kv_dim() as usize;
        let q_dim = c.q_dim() as usize;
        let ff = c.ff_size as usize;
        let groups_kernel = c.conv_kernel as usize;
        let group = c.conv_group as usize;
        let conv = c.conv_projection() as usize;
        let positions = POSITION_BOUND as usize + t - 1;
        let pos0_i32 = self
            .pos0
            .view()
            .values()
            .cast::<i32>()
            .ok_or_else(|| "drafter position buffer".to_string())?;
        ran(
            k::embed_q8_0(
                stream,
                self.token_embd.bytes(),
                self.tokens.slice(0..t),
                self.x.slice_mut(0..th),
                t,
                hidden,
                embd_rows,
            ),
            "embed",
        )?;
        for (il, l) in w.layers.iter().enumerate() {
            // Attention with the dynamic convolution around it.
            ran(
                k::rms_norm(
                    stream,
                    self.x.slice(0..th),
                    f32_of(&l.attn_norm)?,
                    self.h.slice_mut(0..th),
                    t,
                    hidden,
                    c.rms_eps,
                ),
                "rms_norm",
            )?;
            dense_bf16(
                stream,
                blas,
                &l.attn_conv_proj,
                self.h.slice(0..th),
                self.dynamic.slice_mut(0..t * conv),
                t,
                None,
            )?;
            ran(
                k::dflash_conv(
                    stream,
                    self.h.slice(0..th),
                    self.dynamic.slice(0..t * conv),
                    f32_of(&l.attn_conv_base)?,
                    self.hc.slice_mut(0..th),
                    t,
                    t,
                    hidden,
                    group,
                    groups_kernel,
                    0,
                ),
                "dflash_conv",
            )?;
            dense_bf16(
                stream,
                blas,
                &l.attn_q,
                self.hc.slice(0..th),
                self.q.slice_mut(0..t * q_dim),
                t,
                None,
            )?;
            dense_bf16(
                stream,
                blas,
                &l.attn_k,
                self.hc.slice(0..th),
                self.k.slice_mut(0..t * kv),
                t,
                None,
            )?;
            dense_bf16(
                stream,
                blas,
                &l.attn_v,
                self.hc.slice(0..th),
                self.v.slice_mut(0..t * kv),
                t,
                None,
            )?;
            rms_norm_in_place(
                stream,
                &mut self.norm_scratch,
                self.q.slice_mut(0..t * q_dim),
                f32_of(&l.attn_q_norm)?,
                t * heads,
                hd,
                c.rms_eps,
            )?;
            rms_norm_in_place(
                stream,
                &mut self.norm_scratch,
                self.k.slice_mut(0..t * kv),
                f32_of(&l.attn_k_norm)?,
                t * kv_heads,
                hd,
                c.rms_eps,
            )?;
            ran(
                k::rope_neox(
                    stream,
                    self.q.slice_mut(0..t * q_dim),
                    pos0_i32,
                    t,
                    heads,
                    hd,
                    c.rope_theta,
                ),
                "rope_neox",
            )?;
            ran(
                k::rope_neox(
                    stream,
                    self.k.slice_mut(0..t * kv),
                    pos0_i32,
                    t,
                    kv_heads,
                    hd,
                    c.rope_theta,
                ),
                "rope_neox",
            )?;
            ran(
                k::ring_append(
                    stream,
                    self.k.slice(0..t * kv),
                    self.v.slice(0..t * kv),
                    self.k_ring[il].as_mut_slice(),
                    self.v_ring[il].as_mut_slice(),
                    self.pos0.view(),
                    t,
                    kv,
                    self.ring,
                    positions,
                ),
                "ring_append",
            )?;
            ran(
                k::drafter_attention(
                    stream,
                    self.q.slice(0..t * q_dim),
                    self.k_ring[il].as_slice(),
                    self.v_ring[il].as_slice(),
                    f32_of(&l.attn_sinks)?,
                    self.ctx.slice_mut(0..t * q_dim),
                    self.pos0.view(),
                    t,
                    heads,
                    kv_heads,
                    self.ring,
                    c.sliding_window as usize,
                    1.0 / (hd as f32).sqrt(),
                    positions,
                ),
                "drafter_attention",
            )?;
            dense_bf16(
                stream,
                blas,
                &l.attn_out,
                self.ctx.slice(0..t * q_dim),
                self.o.slice_mut(0..th),
                t,
                None,
            )?;
            ran(
                k::dflash_conv(
                    stream,
                    self.o.slice(0..th),
                    self.dynamic.slice(0..t * conv),
                    f32_of(&l.attn_conv_base)?,
                    self.hc.slice_mut(0..th),
                    t,
                    t,
                    hidden,
                    group,
                    groups_kernel,
                    1,
                ),
                "dflash_conv",
            )?;
            ran(
                k::add_scaled(
                    stream,
                    self.x.slice_mut(0..th),
                    self.hc.slice(0..th),
                    1.0,
                    None,
                    th,
                ),
                "add_scaled",
            )?;
            // SiLU-gated FFN with the dynamic convolution around it.
            ran(
                k::rms_norm(
                    stream,
                    self.x.slice(0..th),
                    f32_of(&l.ffn_norm)?,
                    self.h.slice_mut(0..th),
                    t,
                    hidden,
                    c.rms_eps,
                ),
                "rms_norm",
            )?;
            dense_bf16(
                stream,
                blas,
                &l.ffn_conv_proj,
                self.h.slice(0..th),
                self.dynamic.slice_mut(0..t * conv),
                t,
                None,
            )?;
            ran(
                k::dflash_conv(
                    stream,
                    self.h.slice(0..th),
                    self.dynamic.slice(0..t * conv),
                    f32_of(&l.ffn_conv_base)?,
                    self.hc.slice_mut(0..th),
                    t,
                    t,
                    hidden,
                    group,
                    groups_kernel,
                    0,
                ),
                "dflash_conv",
            )?;
            quant(
                stream,
                &l.ffn_gate,
                self.hc.slice(0..th),
                self.gate.slice_mut(0..t * ff),
                t,
                "ffn_gate",
            )?;
            quant(
                stream,
                &l.ffn_up,
                self.hc.slice(0..th),
                self.up.slice_mut(0..t * ff),
                t,
                "ffn_up",
            )?;
            ran(
                k::silu_gate(
                    stream,
                    self.gate.slice(0..t * ff),
                    l.ffn_gate.scale,
                    self.up.slice(0..t * ff),
                    l.ffn_up.scale,
                    self.act.slice_mut(0..t * ff),
                    t * ff,
                ),
                "silu_gate",
            )?;
            quant(
                stream,
                &l.ffn_down,
                self.act.slice(0..t * ff),
                self.o.slice_mut(0..th),
                t,
                "ffn_down",
            )?;
            ran(
                k::scale_bias(
                    stream,
                    self.o.slice_mut(0..th),
                    l.ffn_down.scale,
                    None,
                    t,
                    hidden,
                    ACT_NONE,
                ),
                "scale_bias",
            )?;
            ran(
                k::dflash_conv(
                    stream,
                    self.o.slice(0..th),
                    self.dynamic.slice(0..t * conv),
                    f32_of(&l.ffn_conv_base)?,
                    self.hc.slice_mut(0..th),
                    t,
                    t,
                    hidden,
                    group,
                    groups_kernel,
                    1,
                ),
                "dflash_conv",
            )?;
            ran(
                k::add_scaled(
                    stream,
                    self.x.slice_mut(0..th),
                    self.hc.slice(0..th),
                    1.0,
                    None,
                    th,
                ),
                "add_scaled",
            )?;
        }
        // The drafted slots (block rows 1..) through the trunk's head.
        ran(
            k::rms_norm(
                stream,
                self.x.slice(0..th),
                f32_of(&w.output_norm)?,
                self.hidden.slice_mut(0..th),
                t,
                hidden,
                c.rms_eps,
            ),
            "rms_norm",
        )?;
        let slots_hidden = hidden..hidden + s * hidden;
        quant(
            stream,
            &self.output,
            self.hidden.slice(slots_hidden.clone()),
            self.logits.slice_mut(0..s * vocab),
            s,
            "output",
        )?;

        // LiLiCorr: candidates, their features and states.
        ran(
            k::top_k8(
                stream,
                self.logits.slice(0..s * vocab),
                s,
                vocab,
                self.cand.slice_mut(0..n),
                self.values.slice_mut(0..n),
                self.lse.slice_mut(0..s),
            ),
            "top_k8",
        )?;
        ran(
            k::lilicorr_features(
                stream,
                self.values.slice(0..n),
                self.lse.slice(0..s),
                f32_of(&w.feature_norm)?,
                f32_of(&w.feature_norm_b)?,
                self.feat5.slice_mut(0..n * 5),
                s,
            ),
            "lilicorr_features",
        )?;
        dense_f32(
            stream,
            blas,
            &w.feature_up,
            self.feat5.slice(0..n * 5),
            self.feat_up.slice_mut(0..n * l_width),
            n,
        )?;
        ran(
            k::scale_bias(
                stream,
                self.feat_up.slice_mut(0..n * l_width),
                1.0,
                Some(f32_of(&w.feature_up_b)?),
                n,
                l_width,
                ACT_SILU,
            ),
            "scale_bias",
        )?;
        dense_f32(
            stream,
            blas,
            &w.feature_down,
            self.feat_up.slice(0..n * l_width),
            self.feat.slice_mut(0..n * l_width),
            n,
        )?;
        ran(
            k::scale_bias(
                stream,
                self.feat.slice_mut(0..n * l_width),
                1.0,
                Some(f32_of(&w.feature_down_b)?),
                n,
                l_width,
                ACT_NONE,
            ),
            "scale_bias",
        )?;
        // The candidates index the embedding: their IDs come back to the host
        // and return as checked indices (the C++ reads them on the device).
        let mut candidates = vec![0i32; n];
        stream
            .download_blocking(self.cand.slice(0..n), &mut candidates)
            .map_err(hip)?;
        {
            let host = self.cand_host.host_mut().map_err(hip)?;
            for (slot, &id) in host.iter_mut().zip(&candidates) {
                *slot = u32::try_from(id)
                    .map_err(|_| "candidate outside the vocabulary".to_string())?;
            }
        }
        stream
            .upload_indices_async(&mut self.cand_host, n, &mut self.cand_indices)
            .map_err(|_| "candidate outside the vocabulary".to_string())?;
        ran(
            k::embed_q8_0(
                stream,
                self.token_embd.bytes(),
                self.cand_indices.slice(0..n),
                self.embd.slice_mut(0..n * hidden),
                n,
                hidden,
                embd_rows,
            ),
            "embed",
        )?;
        quant(
            stream,
            &w.token_proj,
            self.embd.slice(0..n * hidden),
            self.token.slice_mut(0..n * l_width),
            n,
            "token_proj",
        )?;
        ran(
            k::scale_bias(
                stream,
                self.token.slice_mut(0..n * l_width),
                w.token_proj.scale,
                Some(f32_of(&w.token_proj_b)?),
                n,
                l_width,
                ACT_NONE,
            ),
            "scale_bias",
        )?;
        quant(
            stream,
            &w.pass_proj,
            self.hidden.slice(slots_hidden),
            self.pass.slice_mut(0..s * l_width),
            s,
            "pass_hidden_proj",
        )?;
        ran(
            k::scale_bias(
                stream,
                self.pass.slice_mut(0..s * l_width),
                w.pass_proj.scale,
                Some(f32_of(&w.pass_proj_b)?),
                s,
                l_width,
                ACT_NONE,
            ),
            "scale_bias",
        )?;
        ran(
            k::lilicorr_combine(
                stream,
                self.token.slice(0..n * l_width),
                self.pass.slice(0..s * l_width),
                self.feat.slice(0..n * l_width),
                f32_of(&w.slot_embd)?,
                f32_of(&w.rank_embd)?,
                self.nodes.slice_mut(0..n * l_width),
                s,
                l_width,
            ),
            "lilicorr_combine",
        )?;
        let anchor_row = ((pos0 as usize - 1) % ANCHOR_ROWS) * hidden;
        quant(
            stream,
            &w.context_proj,
            self.anchors.slice(anchor_row..anchor_row + hidden),
            self.anchor.slice_mut(0..l_width),
            1,
            "context_proj",
        )?;
        ran(
            k::scale_bias(
                stream,
                self.anchor.slice_mut(0..l_width),
                w.context_proj.scale,
                Some(f32_of(&w.context_proj_b)?),
                1,
                l_width,
                ACT_NONE,
            ),
            "scale_bias",
        )?;
        let nl = n * l_width;
        let lc_mlp = c.lc_mlp as usize;
        let lc_heads = c.lc_heads as usize;
        let node_scale = 1.0 / ((l_width / lc_heads) as f32).sqrt();
        for l in &w.lc_layers {
            ran(
                k::rms_norm(
                    stream,
                    self.nodes.slice(0..nl),
                    f32_of(&l.attn_norm)?,
                    self.normed.slice_mut(0..nl),
                    n,
                    l_width,
                    c.rms_eps,
                ),
                "rms_norm",
            )?;
            dense_f32(
                stream,
                blas,
                &l.in_proj,
                self.normed.slice(0..nl),
                self.qkv.slice_mut(0..3 * nl),
                n,
            )?;
            ran(
                k::scale_bias(
                    stream,
                    self.qkv.slice_mut(0..3 * nl),
                    1.0,
                    Some(f32_of(&l.in_proj_b)?),
                    n,
                    3 * l_width,
                    ACT_NONE,
                ),
                "scale_bias",
            )?;
            ran(
                k::node_attention(
                    stream,
                    self.qkv.slice(0..3 * nl),
                    f32_of(&w.attention_bias)?,
                    self.att.slice_mut(0..nl),
                    n,
                    lc_heads,
                    node_scale,
                ),
                "node_attention",
            )?;
            quant(
                stream,
                &l.out_proj,
                self.att.slice(0..nl),
                self.proj.slice_mut(0..nl),
                n,
                "lilicorr attn_output",
            )?;
            ran(
                k::scale_bias(
                    stream,
                    self.proj.slice_mut(0..nl),
                    l.out_proj.scale,
                    Some(f32_of(&l.out_proj_b)?),
                    n,
                    l_width,
                    ACT_NONE,
                ),
                "scale_bias",
            )?;
            ran(
                k::add_scaled(
                    stream,
                    self.nodes.slice_mut(0..nl),
                    self.proj.slice(0..nl),
                    1.0,
                    None,
                    nl,
                ),
                "add_scaled",
            )?;
            ran(
                k::rms_norm(
                    stream,
                    self.nodes.slice(0..nl),
                    f32_of(&l.mlp_norm)?,
                    self.normed.slice_mut(0..nl),
                    n,
                    l_width,
                    c.rms_eps,
                ),
                "rms_norm",
            )?;
            quant(
                stream,
                &l.mlp_up,
                self.normed.slice(0..nl),
                self.mlp.slice_mut(0..n * lc_mlp),
                n,
                "lilicorr ffn_up",
            )?;
            ran(
                k::scale_bias(
                    stream,
                    self.mlp.slice_mut(0..n * lc_mlp),
                    l.mlp_up.scale,
                    Some(f32_of(&l.mlp_up_b)?),
                    n,
                    lc_mlp,
                    ACT_SILU,
                ),
                "scale_bias",
            )?;
            quant(
                stream,
                &l.mlp_down,
                self.mlp.slice(0..n * lc_mlp),
                self.proj.slice_mut(0..nl),
                n,
                "lilicorr ffn_down",
            )?;
            ran(
                k::scale_bias(
                    stream,
                    self.proj.slice_mut(0..nl),
                    l.mlp_down.scale,
                    Some(f32_of(&l.mlp_down_b)?),
                    n,
                    l_width,
                    ACT_NONE,
                ),
                "scale_bias",
            )?;
            ran(
                k::add_scaled(
                    stream,
                    self.nodes.slice_mut(0..nl),
                    self.proj.slice(0..nl),
                    1.0,
                    None,
                    nl,
                ),
                "add_scaled",
            )?;
        }
        // Factorized pairwise scores.
        ran(
            k::rms_norm(
                stream,
                self.nodes.slice(0..nl),
                f32_of(&w.lc_output_norm)?,
                self.normed.slice_mut(0..nl),
                n,
                l_width,
                c.rms_eps,
            ),
            "rms_norm",
        )?;
        rms_norm_in_place(
            stream,
            &mut self.norm_scratch,
            self.anchor.slice_mut(0..l_width),
            f32_of(&w.anchor_norm)?,
            1,
            l_width,
            c.rms_eps,
        )?;
        ran(
            k::factor_input(
                stream,
                self.normed.slice(0..nl),
                self.anchor.slice(0..l_width),
                self.factor_in.slice_mut(0..3 * nl),
                n,
                l_width,
            ),
            "factor_input",
        )?;
        dense_f32(
            stream,
            blas,
            &w.factor_in,
            self.factor_in.slice(0..3 * nl),
            self.factors.slice_mut(0..nl),
            n,
        )?;
        ran(
            k::scale_bias(
                stream,
                self.factors.slice_mut(0..nl),
                1.0,
                Some(f32_of(&w.factor_in_b)?),
                n,
                l_width,
                ACT_SILU,
            ),
            "scale_bias",
        )?;
        let f = c.lc_factor as usize;
        dense_f32(
            stream,
            blas,
            &w.out_head,
            self.factors.slice(0..nl),
            self.out_head.slice_mut(0..n * f),
            n,
        )?;
        ran(
            k::bias_l2_norm(
                stream,
                self.out_head.slice_mut(0..n * f),
                f32_of(&w.out_head_b)?,
                n,
                f,
                c.lc_vector_eps,
            ),
            "bias_l2_norm",
        )?;
        dense_f32(
            stream,
            blas,
            &w.in_head,
            self.factors.slice(0..nl),
            self.in_head.slice_mut(0..n * f),
            n,
        )?;
        ran(
            k::bias_l2_norm(
                stream,
                self.in_head.slice_mut(0..n * f),
                f32_of(&w.in_head_b)?,
                n,
                f,
                c.lc_vector_eps,
            ),
            "bias_l2_norm",
        )?;
        dense_f32(
            stream,
            blas,
            &w.anchor_head,
            self.anchor.slice(0..l_width),
            self.anchor_head.slice_mut(0..f),
            1,
        )?;
        ran(
            k::bias_l2_norm(
                stream,
                self.anchor_head.slice_mut(0..f),
                f32_of(&w.anchor_head_b)?,
                1,
                f,
                c.lc_vector_eps,
            ),
            "bias_l2_norm",
        )?;
        ran(
            k::lattice_scores(
                stream,
                self.in_head.slice(0..n * f),
                self.out_head.slice(0..n * f),
                self.anchor_head.slice(0..f),
                self.scores.slice_mut(0..self.scores_len),
                s,
                f,
                c.lc_logit_scale,
            ),
            "lattice_scores",
        )?;
        let mut scores = vec![0f32; self.scores_len];
        stream
            .download_blocking(self.scores.slice(0..self.scores_len), &mut scores)
            .map_err(hip)?;
        if let Some(logits) = logits {
            stream
                .download_blocking(self.logits.slice(0..s * vocab), logits)
                .map_err(hip)?;
        }
        stream.synchronize().map_err(hip)?;
        Ok(DraftLattice {
            slots: s as u32,
            top_k: top_k as u32,
            candidates,
            scores,
        })
    }

    /// Ring and anchor contents, for session snapshots.
    pub fn state_bytes(&self) -> usize {
        Self::state_bytes_for(&self.config)
    }

    /// [`state_bytes`](Self::state_bytes) of any context of a drafter with
    /// this configuration.
    pub fn state_bytes_for(c: &DrafterConfig) -> usize {
        2 * c.num_layers as usize * ring_rows(c) * c.kv_dim() as usize * size_of::<F16>()
            + ANCHOR_ROWS * c.hidden_size as usize * size_of::<f32>()
    }

    /// The device buffers of that state, in `save_state` order.
    pub fn state_pieces(&self) -> Vec<DeviceSlice<'_, u8>> {
        let mut pieces = Vec::with_capacity(2 * self.k_ring.len() + 1);
        for (k, v) in self.k_ring.iter().zip(&self.v_ring) {
            pieces.extend(k.as_slice().cast::<u8>());
            pieces.extend(v.as_slice().cast::<u8>());
        }
        pieces.extend(self.anchors.as_slice().cast::<u8>());
        pieces
    }

    /// [`state_pieces`](Self::state_pieces) for writing, for restores that
    /// copy device state directly.
    pub fn state_pieces_mut(&mut self) -> Vec<DeviceSliceMut<'_, u8>> {
        let mut pieces = Vec::with_capacity(2 * self.k_ring.len() + 1);
        for (k, v) in self.k_ring.iter_mut().zip(self.v_ring.iter_mut()) {
            pieces.extend(k.as_mut_slice().cast::<u8>());
            pieces.extend(v.as_mut_slice().cast::<u8>());
        }
        pieces.extend(self.anchors.as_mut_slice().cast::<u8>());
        pieces
    }

    /// Copies the state to the host (`SaveState`).
    pub fn save_state(&self, stream: &Stream, out: &mut [u8]) -> Result<(), String> {
        if out.len() != self.state_bytes() {
            return Err("drafter state size mismatch".into());
        }
        let mut offset = 0;
        for piece in self.state_pieces() {
            let bytes = piece.len();
            stream
                .download_bytes(piece, &mut out[offset..offset + bytes])
                .map_err(|_| "could not save the drafter state".to_string())?;
            offset += bytes;
        }
        stream
            .synchronize()
            .map_err(|_| "could not save the drafter state".to_string())
    }

    /// Replaces the state from the host (`RestoreState`).
    pub fn restore_state(&mut self, stream: &Stream, input: &[u8]) -> Result<(), String> {
        if input.len() != self.state_bytes() {
            return Err("drafter state size mismatch".into());
        }
        let mut offset = 0;
        for piece in self.state_pieces_mut() {
            let bytes = piece.len();
            stream
                .upload_bytes(&input[offset..offset + bytes], piece)
                .map_err(|_| "could not restore the drafter state".to_string())?;
            offset += bytes;
        }
        stream
            .synchronize()
            .map_err(|_| "could not restore the drafter state".to_string())
    }
}
