//! The Qwen3.8 text model and the state of its sequences, a port of
//! `QwenGpuExecutor` (`hip/executor.cpp`, `hip/decode.cpp`, `hip/prefill.cpp`).
//! Launch order and arguments are those of the C++ engine, so logits and
//! token IDs match it bit for bit.
//!
//! A [`Model`] holds the weights, the stream and the activation scratch that
//! one operation at a time uses; each [`Session`] holds the caches and
//! recurrent state of one sequence, allocated on the model's stream, so a
//! batched step reads several sessions in one stream. [`Engine`] is a model
//! with one session.

use std::path::Path;
use std::sync::Arc;

use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_core::text::CancellationCheck;
use gufo_hip::{
    DeviceBuffer, DeviceSlice, DeviceSliceMut, DeviceTensor, Event, F16, GraphCache, LaunchError,
    PinnedBuffer, Stream,
};
use gufo_qwen_kernels as k;

use crate::arena::{
    Forward, MAX_PROMPT_BATCH, MAX_TARGET_TAPS, MAX_VERIFY_ROWS, REPLAY_CAPACITY, ReplayRing,
    Scratch, SessionBuffers, alloc,
};
use crate::batch::{Row, decode_rows};
use crate::config::ModelConfig;
use crate::decode::{decode_step, split_k_attention};
use crate::prefill::{PrefillError, prefill_chunk};
use crate::verify::{Item, verify_batch};
use crate::weights::{Mixer, Weights};
use gufo_qwen::vision::{Encoder, Prompt, RopeLayout, VisionInput};

/// Copies the valid KV rows of both planes between the cache (`max_context`
/// rows per attention layer) and a snapshot (`valid` rows per layer) with one
/// 2D copy per plane, like the C++ arena. `across` lets the source belong to
/// another stream (a capture on a transfer stream), waiting for the copy.
fn copy_kv_rows(
    s: &Stream,
    from: DeviceSlice<'_, F16>,
    from_pitch: usize,
    mut to: DeviceSliceMut<'_, F16>,
    to_pitch: usize,
    (layers, width, valid): (usize, usize, usize),
    across: bool,
) -> Result<(), gufo_hip::HipError> {
    let (from_plane, to_plane) = (layers * from_pitch, layers * to_pitch);
    for plane in 0..2 {
        let from = from.range(plane * from_plane..(plane + 1) * from_plane);
        let to = to
            .reborrow()
            .range(plane * to_plane..(plane + 1) * to_plane);
        if across {
            s.copy_2d_across(from, from_pitch, to, to_pitch, valid * width, layers)?;
        } else {
            s.copy_2d(from, from_pitch, to, to_pitch, valid * width, layers)?;
        }
    }
    Ok(())
}

/// Largest head dimension of the fused QK-norm/RoPE/KV kernel.
const FUSED_QK_NORM_MAX_HEAD_DIM: u32 = 256;

/// The shapes the Rust kernels and routes take: heads up to the fused
/// QK-norm kernel's width in whole groups per KV head, and the row-split
/// DeltaNet's 128 x 128 state. The C++ fallbacks for other shapes (unfused
/// QK-norm and RoPE, the plain recurrence) are not ported; no published
/// Qwen3.5 checkpoint needs them.
pub fn shape_supported(c: &ModelConfig) -> Result<(), String> {
    if c.head_dim > FUSED_QK_NORM_MAX_HEAD_DIM {
        return Err(format!(
            "head_dim {} exceeds the fused QK-norm kernel ({FUSED_QK_NORM_MAX_HEAD_DIM})",
            c.head_dim
        ));
    }
    if c.num_key_value_heads == 0 || !c.num_attention_heads.is_multiple_of(c.num_key_value_heads) {
        return Err("the query heads must form whole groups per KV head".into());
    }
    if !crate::routes::deltanet_row_split_supported(c.ssm_state_size, c.ssm_value_size()) {
        return Err("the Rust Qwen recurrence needs a 128 x 128 DeltaNet state".into());
    }
    Ok(())
}

fn launch(error: LaunchError) -> String {
    format!("Qwen forward pass: {error}")
}

/// A host draw under a JSON grammar, which the device sampler cannot apply.
fn host_sample(sampler: &mut SamplerState, logits: &[f32]) -> Result<u32, String> {
    sampler.sample(logits).map_err(|e| e.message().to_string())
}

/// The error of a cancelled prompt forward.
const PREFILL_CANCELLED: &str = "Qwen prefill cancelled";

fn prefill_error(error: PrefillError) -> String {
    match error {
        PrefillError::Launch(error) => launch(error),
        PrefillError::Device(error) => format!("Qwen prefill backpressure: {error}"),
        PrefillError::Cancelled => PREFILL_CANCELLED.into(),
    }
}

/// The Qwen3.8 text model on one device: weights, stream and the scratch of
/// the operation that runs.
pub struct Model {
    stream: Stream,
    weights: Weights,
    scratch: Scratch,
    max_context: u32,
    /// GPU sampling buffers, allocated with the first sampled token.
    sampling: Option<SamplingWorkspace>,
    /// `[token][tap][hidden]` of the last `forward_prompt` of a session with
    /// capture enabled, in its first `prompt_hidden_len` values. One buffer
    /// for every session: a draft reads it right after the prefill under the
    /// model lock, and its pages serve every later prompt (a buffer per
    /// session faulted hundreds of MB anew in each session's first request).
    /// Page-locked, so each layer's rows download without a wait and the
    /// reader waits once; it only grows, to the longest prompt, and is not
    /// zeroed again.
    prompt_hidden: Option<PinnedBuffer<f32>>,
    prompt_hidden_len: usize,
}

/// `GpuSamplingWorkspace`: the adjusted and sorted logits, their token IDs,
/// the penalty list (three words each), the sort storage and the drawn
/// token.
struct SamplingWorkspace {
    adjusted: DeviceBuffer<f32>,
    sorted: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    sorted_ids: DeviceBuffer<u32>,
    penalties: DeviceBuffer<u32>,
    sort: DeviceBuffer<u8>,
    token: DeviceBuffer<u32>,
    accepted: DeviceBuffer<u32>,
    /// Draft candidates of speculative sampling.
    candidate_ids: DeviceBuffer<u32>,
    candidate_probabilities: DeviceBuffer<f32>,
}

/// Where a sampled row lives.
enum LogitsRow {
    /// The session's last logits.
    Last,
    /// A row of the session's last verification, at this element offset.
    Verification(usize),
}

/// Bytes reserved for the logit sort; the launcher checks the actual need.
const SAMPLING_SORT_BYTES: usize = 16 << 20;

/// Host side of the saved recurrent state and its replay log
/// (`QwenGpuArena` fields).
#[derive(Default)]
struct ReplayState {
    has_saved: bool,
    saved_context: u32,
    active: bool,
    last_position: u32,
    captured: usize,
}

/// The state of one sequence on a model's stream.
pub struct Session {
    stream_id: u64,
    max_context: u32,
    /// Attention layers and KV width, for snapshots.
    layers: usize,
    width: usize,
    buffers: SessionBuffers,
    /// The decode step below the split-K context, which reads token and
    /// position from device memory (one graph, like the C++ executor). Its
    /// launches depend on the tapped layers, on whether a replay ring exists
    /// and on whether the session has an mRoPE table (created by its first
    /// image), so all three form the key, with the recurrent buffers the
    /// step reads and writes.
    graphs: GraphCache<(Vec<u32>, bool, bool, u8)>,
    /// The token after the retained sequence, left by the last generation.
    next_token: Option<u32>,
    /// Layers whose hidden rows the draft model reads (`SetPromptHiddenCapture`).
    taps: Vec<u32>,
    capture_hidden: bool,
    /// `[row][tap][hidden]` of the last verification.
    verification_hidden: Vec<f32>,
    replay: ReplayState,
    /// After `restore_state`: committed rows replay the recorded recurrence.
    replaying: bool,
    /// Rows of the last verification (`last_verification_rows_`).
    last_verification_rows: usize,
    /// Where the last forward's final row starts in the model's hidden
    /// scratch (`last_hidden_offset_`): 0 after a decode step, the chunk's
    /// last row after a prefill.
    last_hidden_offset: usize,
    /// Without taps, verified rows stay in the model's hidden scratch
    /// ([`Model::verification_rows`]) instead of `verification_hidden`.
    device_rows: bool,
    /// The scratch row of this session's first row in the last verification.
    verification_offset: usize,
    /// Recorded on the model's stream after a committed verification's
    /// replay, which ends without a host wait: a capture on another stream
    /// waits for it instead of relying on a later call to.
    settled: Event,
}

/// Device storage of a snapshot's recurrent state, allocated before its
/// capture ([`Session::capture_with`]): `hipMalloc` of its ~180 MB takes
/// about 9 ms and does not wait for queued work, so a prefill running
/// meanwhile hides it.
pub struct RecurrentCapture {
    conv: DeviceBuffer<f32>,
    deltanet: DeviceBuffer<f32>,
}

impl RecurrentCapture {
    /// Storage for `lens` (conv, DeltaNet) on `transfer`, the stream the
    /// capture copies on.
    pub fn allocate(transfer: &Stream, (conv, deltanet): (usize, usize)) -> Result<Self, String> {
        let error = |e| format!("Qwen snapshot: {e}");
        Ok(Self {
            conv: transfer.alloc_persistent(conv).map_err(error)?,
            deltanet: transfer.alloc_persistent(deltanet).map_err(error)?,
        })
    }

    fn fits(&self, transfer: &Stream, (conv, deltanet): (usize, usize)) -> bool {
        let home = Some(transfer.id());
        self.conv.len() == conv
            && self.deltanet.len() == deltanet
            && self.conv.as_slice().home_stream() == home
            && self.deltanet.as_slice().home_stream() == home
    }
}

/// One session's tokens in a verification batch.
pub struct VerifyRequest<'a> {
    pub session: &'a mut Session,
    pub tokens: &'a [u32],
    pub position: u32,
    /// Also return the rows' logits.
    pub capture_logits: bool,
}

/// What a verification returned for one session.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Verification {
    /// Each row's greedy next token.
    pub tokens: Vec<u32>,
    /// `[row][vocab]` when requested.
    pub logits: Vec<f32>,
}

/// Bytes of one image grid (`vision::ImageGrid`: offset, height, width).
const GRID_BYTES: usize = 12;

/// Device copy of a sequence state at `valid_context` positions
/// (`QwenGpuSnapshot`): the KV rows written so far and the recurrent state,
/// immutable and readable on every stream.
pub struct Snapshot {
    valid_context: u32,
    max_context: u32,
    /// K rows then V rows, each `[attention layer][valid_context][kv width]`.
    kv: DeviceTensor<F16>,
    conv: DeviceTensor<f32>,
    deltanet: DeviceTensor<f32>,
    /// The positions of the images the captured sequence consumed.
    vision_layout: RopeLayout,
}

impl Snapshot {
    /// The image grids of the captured sequence (none for text).
    pub fn vision_layout(&self) -> &RopeLayout {
        &self.vision_layout
    }

    pub fn valid_context(&self) -> u32 {
        self.valid_context
    }

    /// Payload bytes of the snapshot: the device state and its image grids
    /// (`QwenGpuSnapshot::PayloadBytes`).
    pub fn bytes(&self) -> usize {
        self.kv.len() * 2
            + (self.conv.len() + self.deltanet.len()) * 4
            + self.vision_layout.images.len() * GRID_BYTES
    }

    /// K rows then V rows, each `[attention layer][valid_context][kv width]`
    /// in FP16.
    pub fn kv(&self) -> &DeviceTensor<F16> {
        &self.kv
    }

    /// The convolution state (f32).
    pub fn conv(&self) -> &DeviceTensor<f32> {
        &self.conv
    }

    /// The DeltaNet state (f32).
    pub fn deltanet(&self) -> &DeviceTensor<f32> {
        &self.deltanet
    }
}

impl Session {
    /// The request's cancellation check, asked before prefill chunks and
    /// layers and while images encode (`SetCancellationCheck`).
    pub fn set_cancellation_check(&mut self, check: Option<CancellationCheck>) {
        self.buffers.cancel = check;
    }

    /// The image positions of the current request.
    pub fn vision_layout(&self) -> &RopeLayout {
        self.buffers.vision.layout()
    }

    fn check(&self, model: &Model) -> Result<(), String> {
        if self.stream_id != model.stream.id() {
            return Err("Qwen session belongs to another model".into());
        }
        Ok(())
    }

    /// Captures the first `valid_context` positions on `transfer`, which may
    /// be another stream than the model's: the session must be idle (its
    /// last operation finished), other sessions may run meanwhile
    /// (`SaveSnapshot` on a `SnapshotTransfer`).
    pub fn capture(&self, transfer: &Stream, valid_context: u32) -> Result<Snapshot, String> {
        self.capture_with(transfer, valid_context, None)
    }

    /// The conv and DeltaNet lengths a capture copies, for a
    /// [`RecurrentCapture`] allocated ahead.
    pub fn recurrent_lens(&self) -> (usize, usize) {
        let state = self.buffers.recurrent.live();
        (state.conv.len(), state.deltanet.len())
    }

    /// [`Session::capture`] into `recurrent` when it fits this session on
    /// `transfer`, else into new storage.
    pub fn capture_with(
        &self,
        transfer: &Stream,
        valid_context: u32,
        recurrent: Option<RecurrentCapture>,
    ) -> Result<Snapshot, String> {
        if valid_context > self.max_context {
            return Err("Qwen snapshot exceeds the context length".into());
        }
        let (layers, width, valid) = (self.layers, self.width, valid_context as usize);
        let error = |e| format!("Qwen snapshot: {e}");
        let across = transfer.id() != self.stream_id;
        if across {
            self.settled.synchronize().map_err(error)?;
        }
        let b = &self.buffers;
        let state = b.recurrent.live();
        let mut kv = transfer
            .alloc_persistent(2 * layers * valid * width)
            .map_err(error)?;
        let RecurrentCapture {
            mut conv,
            mut deltanet,
        } = match recurrent {
            Some(r) if r.fits(transfer, self.recurrent_lens()) => r,
            _ => RecurrentCapture::allocate(transfer, self.recurrent_lens())?,
        };
        copy_kv_rows(
            transfer,
            b.kv.as_slice(),
            self.max_context as usize * width,
            kv.as_mut_slice(),
            valid * width,
            (layers, width, valid),
            across,
        )
        .map_err(error)?;
        if across {
            transfer
                .copy_across(state.conv.as_slice(), conv.as_mut_slice())
                .map_err(error)?;
            transfer
                .copy_across(state.deltanet.as_slice(), deltanet.as_mut_slice())
                .map_err(error)?;
        } else {
            transfer
                .copy(state.conv.as_slice(), conv.as_mut_slice())
                .map_err(error)?;
            transfer
                .copy(state.deltanet.as_slice(), deltanet.as_mut_slice())
                .map_err(error)?;
        }
        Ok(Snapshot {
            valid_context,
            max_context: self.max_context,
            kv: kv.freeze(transfer).map_err(error)?,
            conv: conv.freeze(transfer).map_err(error)?,
            deltanet: deltanet.freeze(transfer).map_err(error)?,
            // Only images the captured context consumed; the attached
            // request may append more.
            vision_layout: b.vision.layout().prefix(valid_context),
        })
    }

    /// The logits of the last row that computed them, copied on `transfer`
    /// like [`Session::capture`] (`CopyLastLogits`).
    pub fn copy_logits(&self, transfer: &Stream, vocab: usize) -> Result<Vec<f32>, String> {
        let error = |e| format!("Qwen logits: {e}");
        let mut logits = vec![0f32; vocab];
        let source = self.buffers.logits.slice(0..vocab);
        if transfer.id() == self.stream_id {
            transfer.download(source, &mut logits).map_err(error)?;
        } else {
            self.settled.synchronize().map_err(error)?;
            let mut copy = transfer.alloc::<f32>(vocab).map_err(error)?;
            transfer
                .copy_across(source, copy.as_mut_slice())
                .map_err(error)?;
            transfer
                .download(copy.as_slice(), &mut logits)
                .map_err(error)?;
        }
        Ok(logits)
    }

    /// Captures the hidden rows after `layers` (zero-based, at most five) for
    /// a draft model: prompt rows in [`Model::prompt_hidden`], verified rows
    /// in [`Session::verification_hidden`], the last decode or verification
    /// row in the target features (`SetPromptHiddenCapture`). Without layers
    /// it captures each row's final hidden state, the MTP head's input.
    pub fn set_hidden_capture(
        &mut self,
        config: &ModelConfig,
        enabled: bool,
        layers: &[u32],
    ) -> Result<(), String> {
        if layers.len() > MAX_TARGET_TAPS || layers.iter().any(|&l| l >= config.num_layers) {
            return Err("target hidden-layer taps are out of range".into());
        }
        self.capture_hidden = enabled;
        self.verification_hidden.clear();
        self.taps = if enabled { layers.to_vec() } else { Vec::new() };
        Ok(())
    }

    /// Without taps, keeps verified rows on the device for a draft that
    /// copies them there (the MTP head) instead of downloading them.
    pub fn set_device_rows(&mut self, enabled: bool) {
        self.device_rows = enabled;
    }

    /// `[row][tap][hidden]` of the last verification with capture enabled.
    pub fn verification_hidden(&self) -> &[f32] {
        &self.verification_hidden
    }

    /// Graph replays whose launches no longer matched their capture; 0 unless
    /// a per-step value reached a kernel as a scalar.
    pub fn graph_mismatches(&self) -> u64 {
        self.graphs.mismatches()
    }

    /// Decode graphs captured so far (at most one per key).
    pub fn captured_graphs(&self) -> usize {
        self.graphs.len()
    }

    fn mark_replay_position(&mut self, position: u32) {
        let r = &mut self.replay;
        if r.active && position >= r.saved_context {
            r.last_position = r.last_position.max(position);
            r.captured = r.captured.max((position - r.saved_context) as usize + 1);
        }
    }

    fn can_replay(&self, position: u32) -> bool {
        let r = &self.replay;
        r.has_saved
            && self.buffers.replay.is_some()
            && r.captured > 0
            && r.captured <= REPLAY_CAPACITY
            && position >= r.saved_context
            && position <= r.last_position
    }
}

impl Model {
    /// Loads `path` and allocates the scratch for `max_context` positions.
    pub fn load(path: &Path, max_context: u32) -> Result<Self, String> {
        let stream = Stream::new().map_err(|e| format!("HIP stream: {e}"))?;
        let weights = Weights::load(&stream, path)?;
        Self::new(stream, weights, max_context)
    }

    pub fn new(stream: Stream, weights: Weights, max_context: u32) -> Result<Self, String> {
        let c = &weights.config;
        if max_context == 0 || max_context > c.context_length {
            return Err("Requested GPU context exceeds the model context length".into());
        }
        shape_supported(c)?;
        let scratch = Scratch::new(&stream, c, max_context)?;
        Ok(Self {
            stream,
            weights,
            scratch,
            max_context,
            sampling: None,
            prompt_hidden: None,
            prompt_hidden_len: 0,
        })
    }

    /// `[token][tap][hidden]` of the last [`Model::forward_prompt`], if its
    /// session captures hidden rows ([`Session::set_hidden_capture`]); empty
    /// otherwise.
    pub fn prompt_hidden(&self) -> Result<&[f32], String> {
        match &self.prompt_hidden {
            Some(buffer) => buffer
                .host()
                .map(|rows| &rows[..self.prompt_hidden_len])
                .map_err(|e| format!("Qwen prompt features: {e}")),
            None => Ok(&[]),
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.weights.config
    }

    /// The weights, for a draft model that shares the embedding and LM head.
    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    pub fn max_context(&self) -> u32 {
        self.max_context
    }

    /// The stream every session's work runs on.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// A fresh sequence state on this model's stream.
    pub fn new_session(&self) -> Result<Session, String> {
        let c = &self.weights.config;
        let mut session = Session {
            stream_id: self.stream.id(),
            max_context: self.max_context,
            layers: c.full_attention_layer_count() as usize,
            width: c.kv_size() as usize,
            buffers: SessionBuffers::new(&self.stream, c, self.max_context)?,
            graphs: GraphCache::new(),
            next_token: None,
            taps: Vec::new(),
            capture_hidden: false,
            verification_hidden: Vec::new(),
            replay: ReplayState::default(),
            replaying: false,
            last_verification_rows: 0,
            last_hidden_offset: 0,
            device_rows: false,
            verification_offset: 0,
            settled: Event::new().map_err(|e| format!("HIP event: {e}"))?,
        };
        self.reset(&mut session)?;
        Ok(session)
    }

    /// Starts a new sequence: clears the recurrent state and the retained
    /// frontier. KV rows are written before any attention reads them, so the
    /// cache is left as it is (like `QwenGpuArena::Reset`).
    pub fn reset(&self, st: &mut Session) -> Result<(), String> {
        st.check(self)?;
        st.next_token = None;
        st.replaying = false;
        st.last_hidden_offset = 0;
        let error = |e| format!("Qwen state reset: {e}");
        // Buffer 0 becomes the live state again.
        st.buffers.recurrent.reset();
        let state = st.buffers.recurrent.live_mut();
        self.stream
            .fill_bytes(state.conv.as_mut_slice(), 0)
            .map_err(error)?;
        self.stream
            .fill_bytes(state.deltanet.as_mut_slice(), 0)
            .map_err(error)?;
        self.disable_replay_capture(st)?;
        st.replay = ReplayState::default();
        Ok(())
    }

    /// Runs `token` at `pos`; with `compute_logits`, returns the greedy next
    /// token and leaves the logits in place for [`Model::logits`].
    pub fn forward_token(
        &mut self,
        st: &mut Session,
        token: u32,
        pos: u32,
        compute_logits: bool,
    ) -> Result<Option<u32>, String> {
        st.check(self)?;
        if pos >= self.max_context {
            return Err("token position exceeds the GPU context length".into());
        }
        let vocab = self.weights.config.vocab_size;
        if token >= vocab {
            return Err(format!("token {token} is outside the vocabulary"));
        }
        // A committed row after a rollback replays the recorded recurrence
        // instead of running the layers (its KV rows are already written).
        if st.replaying {
            if !compute_logits && st.can_replay(pos) {
                self.replay_recurrence(st, pos, 1)?;
                return Ok(None);
            }
            st.replaying = false;
            self.disable_replay_capture(st)?;
        }
        st.mark_replay_position(pos);
        st.last_hidden_offset = 0;
        st.buffers
            .vision
            .prepare(pos, 1, None)
            .map_err(|e| e.message().to_string())?;
        let error = |e| format!("Qwen decode input: {e}");
        self.scratch
            .stage_decode_input(&self.stream, token, pos)
            .map_err(error)?;
        let f = Forward {
            stream: &self.stream,
            weights: &self.weights,
            max_context: self.max_context,
        };
        let (s, x) = (&self.stream, &mut self.scratch);
        let (b, taps) = (&mut st.buffers, &st.taps);
        // Image rows inject their embeddings: only later positions replay
        // the graph (as the C++ executor).
        let after_images = pos >= b.vision.layout().prefix_length();
        b.recurrent.begin_write();
        if split_k_attention(&self.weights.config, pos) {
            decode_step(f, s, x, b, compute_logits, Some(pos), taps, pos).map_err(launch)?;
        } else if compute_logits && after_images {
            let key = (
                taps.clone(),
                b.replay.is_some(),
                b.vision.rope().is_some(),
                b.recurrent.identity(),
            );
            st.graphs
                .run(s, key, |s| decode_step(f, s, x, b, true, None, taps, pos))
                .map_err(launch)?;
        } else {
            decode_step(f, s, x, b, compute_logits, None, taps, pos).map_err(launch)?;
        }
        if !compute_logits {
            return Ok(None);
        }
        self.sampled_tokens(1, "sampled").map(|t| Some(t[0]))
    }

    /// Runs `tokens` at positions `start_pos..` in chunks of at most
    /// [`MAX_PROMPT_BATCH`] rows (`ForwardPromptBatch`); with
    /// `compute_logits`, returns the greedy token after the last one and
    /// leaves its logits for [`Model::logits`].
    pub fn forward_prompt(
        &mut self,
        st: &mut Session,
        tokens: &[u32],
        start_pos: u32,
        compute_logits: bool,
    ) -> Result<Option<u32>, String> {
        st.check(self)?;
        if st.buffers.cancelled() {
            return Err(PREFILL_CANCELLED.into());
        }
        if tokens.is_empty() {
            return Ok(None);
        }
        let end = start_pos as usize + tokens.len();
        if end > self.max_context as usize {
            return Err("prompt exceeds the GPU context length".into());
        }
        st.replaying = false;
        self.disable_replay_capture(st)?;
        // Without taps the capture holds each row's final hidden state.
        let hidden = self.weights.config.hidden_size as usize;
        let features = st.taps.len().max(1) * hidden;
        let error = |e| format!("Qwen prompt features: {e}");
        self.prompt_hidden_len = 0;
        let needed = tokens.len() * features;
        if st.capture_hidden && self.prompt_hidden.as_ref().is_none_or(|b| b.len() < needed) {
            self.prompt_hidden = None;
            self.prompt_hidden = Some(PinnedBuffer::zeroed(needed).map_err(error)?);
        }
        let max_batch = self.max_prompt_batch() as usize;
        let mut download = match self.prompt_hidden.as_mut() {
            Some(buffer) if st.capture_hidden => {
                Some(self.stream.begin_download(buffer).map_err(error)?)
            }
            _ => None,
        };
        for (index, chunk) in tokens.chunks(max_batch).enumerate() {
            if st.buffers.cancelled() {
                return Err(PREFILL_CANCELLED.into());
            }
            let offset = (index * max_batch) as u32;
            let last = offset as usize + chunk.len() == tokens.len();
            let cancel = st.buffers.cancel.clone();
            st.buffers
                .vision
                .prepare(
                    start_pos + offset,
                    chunk.len() as u32,
                    cancel.as_deref().map(|c| c as &dyn Fn() -> bool),
                )
                .map_err(|e| e.message().to_string())?;
            self.stream
                .upload_indices(chunk, &mut self.scratch.tokens)
                .map_err(|e| format!("Qwen prompt tokens: {e}"))?;
            let captured = self.prompt_hidden_len;
            let capture = download
                .as_mut()
                .map(|download| (st.taps.as_slice(), download, captured));
            let f = Forward {
                stream: &self.stream,
                weights: &self.weights,
                max_context: self.max_context,
            };
            prefill_chunk(
                f,
                &mut self.scratch,
                &mut st.buffers,
                chunk.len(),
                start_pos + offset,
                compute_logits && last,
                capture,
            )
            .map_err(prefill_error)?;
            if download.is_some() {
                self.prompt_hidden_len = captured + chunk.len() * features;
            }
            st.last_hidden_offset = (chunk.len() - 1) * hidden;
        }
        download
            .map(|download| download.finish())
            .transpose()
            .map_err(error)?;
        // The last chunk computed the logits.
        Ok(if compute_logits {
            Some(self.sampled_tokens(1, "sampled")?[0])
        } else {
            None
        })
    }

    /// One decode step of two to eight sessions, each running its token at
    /// its position (`ForwardTokenBatch`); returns each session's greedy
    /// next token and leaves its logits for [`Model::logits`].
    pub fn forward_token_batch(
        &mut self,
        items: &mut [(&mut Session, u32, u32)],
    ) -> Result<Vec<u32>, String> {
        let n = items.len();
        if !(2..=MAX_VERIFY_ROWS).contains(&n) {
            return Err("Qwen GPU decode batch requires two to eight sessions".into());
        }
        let vocab = self.weights.config.vocab_size;
        let mut capture_replay = false;
        for (st, token, position) in items.iter_mut() {
            st.check(self)?;
            if *position >= self.max_context {
                return Err("token position exceeds a batched Qwen session context".into());
            }
            if *token >= vocab {
                return Err(format!("token {token} is outside the vocabulary"));
            }
            if st.replaying {
                st.replaying = false;
                self.disable_replay_capture(st)?;
            }
            capture_replay |= st.replay.active;
            st.mark_replay_position(*position);
            st.buffers
                .vision
                .prepare(*position, 1, None)
                .map_err(|e| e.message().to_string())?;
        }
        let tokens: Vec<u32> = items.iter().map(|(_, token, _)| *token).collect();
        let positions: Vec<u32> = items.iter().map(|(_, _, position)| *position).collect();
        let error = |e| format!("Qwen batched decode input: {e}");
        let (s, w, x) = (&self.stream, &self.weights, &mut self.scratch);
        x.stage_tokens(s, &tokens).map_err(error)?;
        if capture_replay {
            x.stage_positions(s, &positions).map_err(error)?;
        }
        if x.verification_logits.is_none() {
            x.verification_logits = Some(alloc(
                s,
                MAX_VERIFY_ROWS * MAX_VERIFY_ROWS * w.config.vocab_size as usize,
            )?);
        }
        let mut rows: Vec<Row<'_>> = items
            .iter_mut()
            .map(|(st, _, position)| {
                let st = &mut **st;
                Row {
                    buffers: &mut st.buffers,
                    position: *position,
                    taps: &st.taps,
                }
            })
            .collect();
        let f = Forward {
            stream: s,
            weights: w,
            max_context: self.max_context,
        };
        decode_rows(f, s, x, &mut rows).map_err(launch)?;
        drop(rows);
        self.sampled_tokens(n, "sampled")
    }

    /// Rows one prefill chunk takes (`GetMaxPromptBatch`).
    pub fn max_prompt_batch(&self) -> u32 {
        self.max_context.min(MAX_PROMPT_BATCH)
    }

    /// Copies the state of the first `valid_context` positions on the
    /// model's stream (`SaveSnapshot`).
    pub fn save_snapshot(&self, st: &Session, valid_context: u32) -> Result<Snapshot, String> {
        st.check(self)?;
        st.capture(&self.stream, valid_context)
    }

    /// Returns to the state of `snapshot` (`RestoreSnapshot`); the retained
    /// frontier is cleared.
    pub fn restore_snapshot(&self, st: &mut Session, snapshot: &Snapshot) -> Result<(), String> {
        st.check(self)?;
        let (layers, width) = (st.layers, st.width);
        let valid = snapshot.valid_context as usize;
        if snapshot.max_context != self.max_context
            || snapshot.kv.len() != 2 * layers * valid * width
            || snapshot.conv.len() != st.buffers.recurrent.live().conv.len()
            || snapshot.deltanet.len() != st.buffers.recurrent.live().deltanet.len()
        {
            return Err("Qwen snapshot is incompatible with the engine".into());
        }
        // Like `RestoreSnapshot`: a fresh sequence state first.
        self.reset(st)?;
        let error = |e| format!("Qwen snapshot restore: {e}");
        let (s, b) = (&self.stream, &mut st.buffers);
        copy_kv_rows(
            s,
            snapshot.kv.view(),
            valid * width,
            b.kv.as_mut_slice(),
            self.max_context as usize * width,
            (layers, width, valid),
            false,
        )
        .map_err(error)?;
        let state = b.recurrent.live_mut();
        s.copy(snapshot.conv.view(), state.conv.as_mut_slice())
            .map_err(error)?;
        s.copy(snapshot.deltanet.view(), state.deltanet.as_mut_slice())
            .map_err(error)?;
        b.vision
            .restore_layout(&snapshot.vision_layout, s)
            .map_err(|e| e.message().to_string())?;
        s.synchronize().map_err(error)
    }

    /// Installs a request's image prompt (`None` for text) and the encoder
    /// on the session (`ConfigureVision`).
    pub fn configure_vision(
        &self,
        st: &mut Session,
        prompt: Option<Arc<Prompt>>,
        encoder: Option<Arc<Encoder>>,
    ) -> Result<(), String> {
        st.check(self)?;
        if let Some(prompt) = &prompt {
            prompt
                .rope
                .validate(self.max_context)
                .map_err(|e| e.message().to_string())?;
        }
        st.buffers
            .vision
            .configure(prompt, encoder, &self.stream)
            .map_err(|e| e.message().to_string())
    }

    /// Loads a compact target state (`GQKVSNP1` sections: the K and V rows of
    /// `valid_context` positions, FP16, and the FP32 convolution and DeltaNet
    /// state) into a fresh sequence (`RestoreCompactSnapshot`).
    #[allow(clippy::too_many_arguments)]
    pub fn restore_compact(
        &self,
        st: &mut Session,
        k: &[u8],
        v: &[u8],
        conv: &[u8],
        deltanet: &[u8],
        valid_context: u32,
        vision_layout: &RopeLayout,
    ) -> Result<(), String> {
        st.check(self)?;
        let (layers, width, valid) = (st.layers, st.width, valid_context as usize);
        let row = valid * width * 2;
        if valid_context > self.max_context
            || k.len() != layers * row
            || v.len() != k.len()
            || conv.len() != st.buffers.recurrent.live().conv.len() * 4
            || deltanet.len() != st.buffers.recurrent.live().deltanet.len() * 4
        {
            return Err("Qwen compact snapshot is incompatible with the arena".into());
        }
        self.reset(st)?;
        let error = |e| format!("Qwen compact snapshot restore: {e}");
        let s = &self.stream;
        let pitch = self.max_context as usize * width * 2;
        let b = &mut st.buffers;
        let plane = b.kv.len() / 2;
        for (index, rows) in [k, v].into_iter().enumerate() {
            let target =
                b.kv.slice_mut(index * plane..(index + 1) * plane)
                    .cast::<u8>()
                    .ok_or("Qwen KV cache is not byte addressable")?;
            s.upload_2d(rows, row, target, pitch, row, layers)
                .map_err(error)?;
        }
        let unaddressable = "Qwen recurrent state is not byte addressable";
        let state = b.recurrent.live_mut();
        let conv_bytes = state
            .conv
            .as_mut_slice()
            .cast::<u8>()
            .ok_or(unaddressable)?;
        s.upload(conv, conv_bytes).map_err(error)?;
        let deltanet_bytes = state
            .deltanet
            .as_mut_slice()
            .cast::<u8>()
            .ok_or(unaddressable)?;
        s.upload(deltanet, deltanet_bytes).map_err(error)?;
        b.vision
            .restore_layout(vision_layout, s)
            .map_err(|e| e.message().to_string())
    }

    /// The first `rows` tokens the last argmax wrote, checked against the
    /// vocabulary.
    fn sampled_tokens(&self, rows: usize, what: &str) -> Result<Vec<u32>, String> {
        let mut next = vec![0u32; rows];
        self.stream
            .download(self.scratch.sampled.slice(0..rows), &mut next)
            .map_err(|e| format!("Qwen {what} token: {e}"))?;
        let vocab = self.weights.config.vocab_size;
        if let Some(bad) = next.iter().find(|&&t| t >= vocab) {
            return Err(format!("{what} token {bad} is outside the vocabulary"));
        }
        Ok(next)
    }

    /// The final hidden rows of `st`'s last verification, `[row][hidden]`,
    /// with device rows enabled. They live in the model's scratch, so they
    /// must be copied before any session runs again.
    pub fn verification_rows(&self, st: &Session) -> Result<DeviceSlice<'_, f32>, String> {
        st.check(self)?;
        if !st.capture_hidden || !st.device_rows || !st.taps.is_empty() {
            return Err("verified rows are kept on the device only without taps".into());
        }
        let hidden = self.weights.config.hidden_size as usize;
        let start = st.verification_offset * hidden;
        Ok(self
            .scratch
            .hidden
            .slice(start..start + st.last_verification_rows * hidden))
    }

    /// The final hidden row of the last forward, on the device; like the
    /// verified rows it must be copied before any session runs again.
    pub fn final_row(&self, st: &Session) -> Result<DeviceSlice<'_, f32>, String> {
        st.check(self)?;
        let hidden = self.weights.config.hidden_size as usize;
        let start = st.last_hidden_offset;
        Ok(self.scratch.hidden.slice(start..start + hidden))
    }

    /// The tapped hidden rows `[tap][hidden]` after the last decode step or
    /// verified row, or without taps the final hidden row of the last
    /// forward (`CopyLastHidden`). The latter lives in the model's scratch,
    /// so it must be read before any session runs again.
    pub fn last_hidden(&self, st: &Session) -> Result<Vec<f32>, String> {
        st.check(self)?;
        if st.taps.is_empty() {
            let hidden = self.weights.config.hidden_size as usize;
            let start = st.last_hidden_offset;
            let mut row = vec![0f32; hidden];
            self.stream
                .download(self.scratch.hidden.slice(start..start + hidden), &mut row)
                .map_err(|e| format!("Qwen hidden row: {e}"))?;
            return Ok(row);
        }
        let mut rows = vec![0f32; st.taps.len() * self.weights.config.hidden_size as usize];
        self.stream
            .download(st.buffers.target_features.slice(0..rows.len()), &mut rows)
            .map_err(|e| format!("Qwen hidden rows: {e}"))?;
        Ok(rows)
    }

    /// Allocates the second buffer of the rollback pair now, so the first
    /// verification of `st` allocates nothing (`AllocateRecurrentSnapshot`).
    pub fn prepare_rollback(&self, st: &mut Session) -> Result<(), String> {
        st.check(self)?;
        st.buffers.recurrent.allocate_second(&self.stream)
    }

    /// Saves the recurrent state at `valid_context` and starts recording the
    /// recurrence inputs of the following rows, so a verification can be
    /// rolled back (`SaveState`). KV rows are append-only and stay.
    pub fn save_state(&self, st: &mut Session, valid_context: u32) -> Result<(), String> {
        st.check(self)?;
        if valid_context > self.max_context {
            return Err("saved GPU state exceeds the context length".into());
        }
        st.replaying = false;
        let error = |e| format!("Qwen saved state: {e}");
        let (s, b) = (&self.stream, &mut st.buffers);
        // The state is pinned where it is; the next pass writes the other
        // buffer.
        b.recurrent.save(s)?;
        st.replay = ReplayState {
            has_saved: true,
            saved_context: valid_context,
            active: false,
            last_position: valid_context,
            captured: 0,
        };
        // Begin the replay capture (`BeginSsmReplayCapture`); the flag word is
        // 0x01010101 as in C++.
        if b.replay.is_none() {
            b.replay = Some(ReplayRing::new(s, &self.weights.config)?);
        }
        if let Some(ring) = b.replay.as_mut() {
            s.fill_bytes(ring.enabled.as_mut_slice(), 1)
                .map_err(error)?;
        }
        st.replay.active = true;
        Ok(())
    }

    /// Returns to the saved recurrent state; the rows committed next replay
    /// their recorded inputs (`RestoreState`).
    pub fn restore_state(&self, st: &mut Session) -> Result<(), String> {
        st.check(self)?;
        if !st.replay.has_saved {
            return Err("GPU state has not been saved".into());
        }
        if st.replay.active {
            self.disable_replay_capture(st)?;
        }
        st.buffers.recurrent.restore();
        st.replaying = true;
        Ok(())
    }

    /// Ends the replay capture of a verification (`FinishVerification`).
    pub fn finish_verification(&self, st: &mut Session) -> Result<(), String> {
        st.check(self)?;
        self.disable_replay_capture(st)
    }

    /// Runs up to eight `tokens` at positions `start_pos..` with decode
    /// arithmetic and returns each row's greedy next token
    /// (`ForwardDecodeEquivalentVerificationChunk`). Writes their KV rows and
    /// advances the recurrent state over all of them.
    pub fn verify(
        &mut self,
        st: &mut Session,
        tokens: &[u32],
        start_pos: u32,
    ) -> Result<Vec<u32>, String> {
        let mut requests = [VerifyRequest {
            session: st,
            tokens,
            position: start_pos,
            capture_logits: false,
        }];
        let mut verified = self.verify_batch(&mut requests)?;
        Ok(std::mem::take(&mut verified[0].tokens))
    }

    /// Verifies the tokens of one to eight sessions in one pass
    /// (`ForwardVerificationBatch`); each session's rows run at its
    /// position. The sessions must capture the same target layers.
    pub fn verify_batch(
        &mut self,
        requests: &mut [VerifyRequest<'_>],
    ) -> Result<Vec<Verification>, String> {
        let first = requests
            .first()
            .ok_or("Qwen verification requires one to eight sessions")?;
        let (capture, taps) = (first.session.capture_hidden, first.session.taps.clone());
        if requests.len() > MAX_VERIFY_ROWS {
            return Err("Qwen verification requires one to eight sessions".into());
        }
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        for request in requests.iter_mut() {
            request.session.check(self)?;
            let rows = request.tokens.len();
            if request.session.capture_hidden != capture
                || request.session.taps != taps
                || rows == 0
                || rows > MAX_VERIFY_ROWS
            {
                return Err("Qwen verification has incompatible sessions or token counts".into());
            }
            if request.position as usize + rows > self.max_context as usize {
                return Err("Qwen verification exceeds a session context".into());
            }
            let st = &mut *request.session;
            st.replaying = false;
            st.last_verification_rows = rows;
            st.verification_hidden.clear();
            for row in 0..rows as u32 {
                st.mark_replay_position(request.position + row);
                positions.push(request.position + row);
            }
            st.buffers
                .vision
                .prepare(request.position, rows as u32, None)
                .map_err(|e| e.message().to_string())?;
            tokens.extend_from_slice(request.tokens);
        }
        let vocab = self.weights.config.vocab_size as usize;
        let error = |e| format!("Qwen verification input: {e}");
        let (s, w, x) = (&self.stream, &self.weights, &mut self.scratch);
        // One session verifies in its own logits rows, which then also hold
        // its captured features; several share the scratch buffer, which is
        // allocated with the first batch (`EnsureVerificationLogits`).
        let single = requests.len() == 1;
        if !single && x.verification_logits.is_none() {
            x.verification_logits = Some(alloc(s, MAX_VERIFY_ROWS * MAX_VERIFY_ROWS * vocab)?);
        }
        let row = if single && capture {
            vocab.max(taps.len() * w.config.hidden_size as usize)
        } else {
            vocab
        };
        for request in requests.iter_mut() {
            let b = &mut request.session.buffers;
            // Room for the longest verification at once: growing it later
            // would free and allocate within a request.
            let len = request.tokens.len().max(MAX_VERIFY_ROWS) * row;
            if b.verification_logits.as_ref().is_none_or(|l| l.len() < len) {
                b.verification_logits = None;
                b.verification_logits = Some(alloc(s, len)?);
            }
        }
        x.stage_tokens(s, &tokens).map_err(error)?;
        x.stage_positions(s, &positions).map_err(error)?;
        let mut logits: Vec<Vec<f32>> = vec![Vec::new(); requests.len()];
        let mut first_row = 0;
        let mut items: Vec<Item<'_>> = requests
            .iter_mut()
            .zip(logits.iter_mut())
            .map(|(request, logits)| {
                let st = &mut *request.session;
                st.verification_offset = first_row;
                first_row += request.tokens.len();
                let download = st.capture_hidden && !(st.device_rows && st.taps.is_empty());
                Item {
                    rows: request.tokens.len(),
                    position: request.position,
                    hidden: download.then_some(&mut st.verification_hidden),
                    logits: request.capture_logits.then_some(logits),
                    buffers: &mut st.buffers,
                }
            })
            .collect();
        let f = Forward {
            stream: s,
            weights: w,
            max_context: self.max_context,
        };
        verify_batch(f, x, &mut items, &taps, capture).map_err(launch)?;
        drop(items);
        let predictions = self.sampled_tokens(tokens.len(), "verified")?;
        let mut offset = 0;
        Ok(requests
            .iter()
            .zip(logits)
            .map(|(request, logits)| {
                let rows = request.tokens.len();
                offset += rows;
                Verification {
                    tokens: predictions[offset - rows..offset].to_vec(),
                    logits,
                }
            })
            .collect())
    }

    /// One token from row `row` of `st`'s last verification under
    /// `sampler`'s policy (`SampleVerificationLogits`); a JSON grammar
    /// samples on the host.
    pub fn sample_verification(
        &mut self,
        st: &Session,
        row: usize,
        sampler: &mut SamplerState,
    ) -> Result<u32, String> {
        let vocab = self.weights.config.vocab_size as usize;
        if row >= st.last_verification_rows || st.buffers.verification_logits.is_none() {
            return Err("Qwen verification logit row is unavailable".into());
        }
        if sampler
            .needs_constraint_mask()
            .map_err(|e| e.message().to_string())?
        {
            return host_sample(sampler, &self.verification_logits(st, row)?);
        }
        self.sample_logits(st, LogitsRow::Verification(row * vocab), sampler)
    }

    /// Row `row` of `st`'s last verification logits (`CopyVerificationLogits`).
    pub fn verification_logits(&self, st: &Session, row: usize) -> Result<Vec<f32>, String> {
        st.check(self)?;
        let vocab = self.weights.config.vocab_size as usize;
        let buffer = st
            .buffers
            .verification_logits
            .as_ref()
            .filter(|_| row < st.last_verification_rows)
            .ok_or("Qwen verification logit row is unavailable")?;
        let mut logits = vec![0f32; vocab];
        self.stream
            .download(buffer.slice(row * vocab..(row + 1) * vocab), &mut logits)
            .map_err(|e| format!("Qwen verification logits: {e}"))?;
        Ok(logits)
    }

    /// Commits the first `tokens` of a rolled-back verification at
    /// `start_pos..`: replays the recorded recurrence where possible and runs
    /// the rest as decode steps (`CommitVerificationChunk`).
    pub fn commit_verified(
        &mut self,
        st: &mut Session,
        tokens: &[u32],
        start_pos: u32,
    ) -> Result<(), String> {
        st.check(self)?;
        if tokens.is_empty() {
            return Ok(());
        }
        let committed = self.commit_rows(st, tokens, start_pos);
        self.stream
            .record(&st.settled)
            .map_err(|e| format!("Qwen replay: {e}"))?;
        committed
    }

    fn commit_rows(
        &mut self,
        st: &mut Session,
        tokens: &[u32],
        start_pos: u32,
    ) -> Result<(), String> {
        let mut replayed = 0;
        if st.replaying {
            while replayed < tokens.len() && st.can_replay(start_pos + replayed as u32) {
                replayed += 1;
            }
            if replayed != 0 {
                self.replay_recurrence(st, start_pos, replayed as u32)?;
            }
        }
        st.replaying = false;
        if replayed == tokens.len() {
            return Ok(());
        }
        self.disable_replay_capture(st)?;
        // Unrecorded rows use the decode arithmetic, without hidden capture.
        let capture = std::mem::replace(&mut st.capture_hidden, false);
        let result = (replayed..tokens.len()).try_for_each(|row| {
            self.forward_token(st, tokens[row], start_pos + row as u32, false)
                .map(|_| ())
        });
        st.capture_hidden = capture;
        result
    }

    fn disable_replay_capture(&self, st: &mut Session) -> Result<(), String> {
        if let Some(ring) = st.buffers.replay.as_mut() {
            self.stream
                .fill_bytes(ring.enabled.as_mut_slice(), 0)
                .map_err(|e| format!("Qwen replay capture: {e}"))?;
        }
        st.replay.active = false;
        Ok(())
    }

    /// Replays `count` recorded rows from `position` through every recurrent
    /// layer (`ReplaySsmState`): no output, only the conv and DeltaNet state,
    /// split where the ring wraps or the batch limit is reached.
    fn replay_recurrence(
        &mut self,
        st: &mut Session,
        position: u32,
        count: u32,
    ) -> Result<(), String> {
        let c = &self.weights.config;
        let (qkv, rank, inner) = (
            c.ssm_qkv_size() as usize,
            c.ssm_time_step_rank as usize,
            c.ssm_inner_size as usize,
        );
        let max_batch = self.max_prompt_batch();
        let (s, w, x, b) = (
            &self.stream,
            &self.weights,
            &mut self.scratch,
            &mut st.buffers,
        );
        let ring = b.replay.as_ref().ok_or("SSM replay has no recorded rows")?;
        // One pass: it reads the restored (pinned) state and leaves it intact
        // for a second restore.
        b.recurrent.begin_write();
        for (l, layer) in (0u32..).zip(&w.layers) {
            let Mixer::Recurrent(r) = &layer.mixer else {
                continue;
            };
            let ssm_layer = c.ssm_layer_index(l) as usize;
            let mut offset = 0;
            while offset < count {
                // A layer's later splits continue from what its first launch
                // wrote.
                let io = if offset == 0 {
                    b.recurrent.io()
                } else {
                    b.recurrent.continued()
                };
                let start = position + offset;
                let slot = start as usize % REPLAY_CAPACITY;
                let rows = (count - offset)
                    .min((REPLAY_CAPACITY - slot) as u32)
                    .min(max_batch);
                let base = ssm_layer * REPLAY_CAPACITY + slot;
                let n = rows as usize;
                k::ssm_conv_recurrence_rows(
                    s,
                    ring.qkv.slice(base * qkv..(base + n) * qkv),
                    r.conv1d.view(),
                    io.conv,
                    io.conv_in,
                    x.conv_out.slice_mut(0..n * qkv),
                    io.deltanet,
                    io.deltanet_in,
                    ring.alpha.slice(base * rank..(base + n) * rank),
                    ring.beta.slice(base * rank..(base + n) * rank),
                    r.a.view(),
                    r.dt.view(),
                    None,
                    None,
                    None,
                    ssm_layer as u32,
                    qkv,
                    c.ssm_group_count,
                    c.ssm_time_step_rank,
                    c.ssm_state_size,
                    c.ssm_value_size(),
                    rows,
                    rank,
                    inner,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .map_err(launch)?;
                offset += rows;
            }
        }
        Ok(())
    }

    /// One token from `st`'s last logits under `sampler`'s policy on the
    /// device (`SampleLastLogits`); `cached` logits replace the last ones
    /// first (`SampleCachedLogits`). A random policy consumes one draw. A
    /// grammar that restricts the next token samples on the host.
    pub fn sample(
        &mut self,
        st: &mut Session,
        cached: Option<&[f32]>,
        sampler: &mut SamplerState,
    ) -> Result<u32, String> {
        if sampler
            .needs_constraint_mask()
            .map_err(|e| e.message().to_string())?
        {
            if let Some(logits) = cached {
                if logits.len() != self.weights.config.vocab_size as usize {
                    return Err("cached Qwen frontier has the wrong vocabulary".into());
                }
                return host_sample(sampler, logits);
            }
            return host_sample(sampler, &self.logits(st)?);
        }
        if let Some(logits) = cached {
            st.check(self)?;
            if logits.len() != self.weights.config.vocab_size as usize {
                return Err("cached Qwen frontier has the wrong vocabulary".into());
            }
            self.stream
                .upload(logits, st.buffers.logits.as_mut_slice())
                .map_err(|e| format!("Qwen sampling input: {e}"))?;
        }
        self.sample_logits(st, LogitsRow::Last, sampler)
    }

    fn ensure_sampling(&mut self) -> Result<(), String> {
        if self.sampling.is_some() {
            return Ok(());
        }
        let vocab = self.weights.config.vocab_size as usize;
        let s = &self.stream;
        let capacity = self.max_context as usize;
        self.sampling = Some(SamplingWorkspace {
            adjusted: alloc(s, vocab)?,
            sorted: alloc(s, vocab)?,
            ids: alloc(s, vocab)?,
            sorted_ids: alloc(s, vocab)?,
            penalties: alloc(s, (capacity * 3).max(3))?,
            sort: alloc(s, SAMPLING_SORT_BYTES)?,
            token: alloc(s, 1)?,
            accepted: alloc(s, 1)?,
            candidate_ids: alloc(s, capacity)?,
            candidate_probabilities: alloc(s, capacity)?,
        });
        Ok(())
    }

    /// Lossless verification of a sampled draft token against row `row` of
    /// `st`'s last verification (`VerifySampledToken`): accepted with the
    /// ratio of the target and draft probabilities, else replaced by a draw
    /// from the residual. Both draws are taken; on acceptance the sampler
    /// returns to the state before the residual draw and accepts the token.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_sampled_token(
        &mut self,
        st: &Session,
        row: usize,
        draft_token: u32,
        candidate_ids: &[u32],
        candidate_probabilities: &[f32],
        draft_token_probability: f64,
        sampler: &mut SamplerState,
    ) -> Result<(u32, bool), String> {
        st.check(self)?;
        let vocab = self.weights.config.vocab_size as usize;
        if row >= st.last_verification_rows || st.buffers.verification_logits.is_none() {
            return Err("Qwen verification logit row is unavailable".into());
        }
        if candidate_ids.is_empty()
            || candidate_ids.len() != candidate_probabilities.len()
            || !draft_token_probability.is_finite()
            || draft_token_probability <= 0.0
            || draft_token_probability > 1.0
        {
            return Err("Qwen sampled verification proposal is malformed".into());
        }
        if sampler
            .needs_constraint_mask()
            .map_err(|e| e.message().to_string())?
        {
            // The target p follows the grammar; the draft's q may not.
            let message = |e: gufo_core::text::Error| e.message().to_string();
            let target = sampler
                .distribution(&self.verification_logits(st, row)?)
                .map_err(message)?;
            if sampler.uniform() * draft_token_probability < target.probability(draft_token) {
                sampler.accept(draft_token).map_err(message)?;
                return Ok((draft_token, true));
            }
            let token = target
                .sample_residual(
                    candidate_ids,
                    candidate_probabilities,
                    sampler.mutable_rng_state(),
                )
                .map_err(message)?;
            return Ok((token, false));
        }
        self.ensure_sampling()?;
        let error = |e| format!("Qwen speculative sampling: {e}");
        let config = sampler.config().clone();
        let acceptance_uniform = sampler.uniform();
        let residual_checkpoint = sampler.rng_state();
        let residual_uniform = sampler.uniform();
        let s = &self.stream;
        let Some(w) = self.sampling.as_mut() else {
            return Err("Qwen sampling workspace is missing".into());
        };
        let count = candidate_ids.len();
        let penalties: Vec<u32> = sampler
            .penalties()
            .iter()
            .flat_map(|p| [p.token, p.generated_count, p.repeated])
            .collect();
        if count > w.candidate_ids.len() || penalties.len() > w.penalties.len() {
            return Err("speculative sampling input exceeds the workspace".into());
        }
        s.upload(candidate_ids, w.candidate_ids.slice_mut(0..count))
            .map_err(error)?;
        s.upload(
            candidate_probabilities,
            w.candidate_probabilities.slice_mut(0..count),
        )
        .map_err(error)?;
        if !penalties.is_empty() {
            s.upload(&penalties, w.penalties.slice_mut(0..penalties.len()))
                .map_err(error)?;
        }
        let logits = st
            .buffers
            .verification_logits
            .as_ref()
            .ok_or("Qwen verification logit row is unavailable")?
            .slice(row * vocab..(row + 1) * vocab);
        k::gpu_speculative_sampling(
            s,
            logits,
            w.token.as_mut_slice(),
            w.accepted.as_mut_slice(),
            vocab,
            config.temperature,
            config.top_k,
            config.top_p,
            config.min_p,
            config.min_keep,
            config.repeat_penalty,
            config.frequency_penalty,
            config.presence_penalty,
            draft_token,
            draft_token_probability as f32,
            w.candidate_ids.slice(0..count),
            w.candidate_probabilities.slice(0..count),
            count,
            acceptance_uniform,
            residual_uniform,
            w.penalties.slice(0..penalties.len()),
            penalties.len() / 3,
            w.adjusted.as_mut_slice(),
            w.sorted.as_mut_slice(),
            w.ids.as_mut_slice(),
            w.sorted_ids.as_mut_slice(),
            w.sort.as_mut_slice(),
            SAMPLING_SORT_BYTES,
        )
        .map_err(launch)?;
        let (mut token, mut accepted) = ([0u32], [0u32]);
        s.download(w.token.as_slice(), &mut token).map_err(error)?;
        s.download(w.accepted.as_slice(), &mut accepted)
            .map_err(error)?;
        if token[0] as usize >= vocab || accepted[0] > 1 {
            return Err("target logit distribution contains no finite values".into());
        }
        let accepted = accepted[0] != 0;
        if accepted {
            sampler.set_rng_state(residual_checkpoint);
            sampler
                .accept(draft_token)
                .map_err(|e| e.message().to_string())?;
        }
        Ok((token[0], accepted))
    }

    fn sample_logits(
        &mut self,
        st: &Session,
        row: LogitsRow,
        sampler: &mut SamplerState,
    ) -> Result<u32, String> {
        st.check(self)?;
        let vocab = self.weights.config.vocab_size as usize;
        self.ensure_sampling()?;
        let error = |e| format!("Qwen sampling: {e}");
        let s = &self.stream;
        let Some(w) = self.sampling.as_mut() else {
            return Err("Qwen sampling workspace is missing".into());
        };
        let penalties: Vec<u32> = sampler
            .penalties()
            .iter()
            .flat_map(|p| [p.token, p.generated_count, p.repeated])
            .collect();
        if penalties.len() > w.penalties.len() {
            return Err("sampling penalties exceed the workspace".into());
        }
        if !penalties.is_empty() {
            s.upload(&penalties, w.penalties.slice_mut(0..penalties.len()))
                .map_err(error)?;
        }
        let config = sampler.config().clone();
        let uniform = if config.uses_random_sampling() {
            sampler.uniform()
        } else {
            0.0
        };
        let logits = match row {
            LogitsRow::Last => st.buffers.logits.as_slice(),
            LogitsRow::Verification(offset) => st
                .buffers
                .verification_logits
                .as_ref()
                .ok_or("Qwen verification logit row is unavailable")?
                .slice(offset..offset + vocab),
        };
        k::gpu_sampling(
            s,
            logits,
            w.token.as_mut_slice(),
            vocab,
            config.temperature,
            config.top_k,
            config.top_p,
            config.min_p,
            config.min_keep,
            config.repeat_penalty,
            config.frequency_penalty,
            config.presence_penalty,
            uniform,
            w.penalties.slice(0..penalties.len()),
            penalties.len() / 3,
            w.adjusted.as_mut_slice(),
            w.sorted.as_mut_slice(),
            w.ids.as_mut_slice(),
            w.sorted_ids.as_mut_slice(),
            w.sort.as_mut_slice(),
            SAMPLING_SORT_BYTES,
        )
        .map_err(launch)?;
        let mut token = [0u32];
        s.download(w.token.as_slice(), &mut token).map_err(error)?;
        if token[0] as usize >= vocab {
            return Err(format!(
                "sampled token {} is outside the vocabulary",
                token[0]
            ));
        }
        Ok(token[0])
    }

    /// Generates up to `max_new_tokens` after `prompt` under `sampling`
    /// (`QwenGpuExecutor::Generate`): a fresh sequence, the prompt as one
    /// prefill, then one step per token until a stop token. `on_token` sees
    /// each generated token and stops generation by returning `false`.
    pub fn generate(
        &mut self,
        st: &mut Session,
        prompt: &[u32],
        max_new_tokens: usize,
        sampling: &SamplingConfig,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, String> {
        self.reset(st)?;
        self.generate_from_prefix(st, prompt, 0, max_new_tokens, sampling, on_token)
    }

    /// Continues a retained sequence (`GenerateFromPrefix`): the session
    /// holds `prompt[..cached]`, for example the previous prompt and its
    /// generated tokens. Prefills the rest of `prompt` at its positions, or
    /// with nothing new continues from the retained frontier. A stop token
    /// ends generation without being fed, so a following turn starts with it.
    pub fn generate_from_prefix(
        &mut self,
        st: &mut Session,
        prompt: &[u32],
        cached: usize,
        max_new_tokens: usize,
        sampling: &SamplingConfig,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, String> {
        let mut output = Vec::new();
        if prompt.is_empty() {
            return Ok(output);
        }
        if cached > prompt.len() {
            return Err("cached Qwen prefix exceeds the rendered prompt".into());
        }
        let mut next = if cached < prompt.len() {
            let start = u32::try_from(cached).map_err(|_| "prefix exceeds u32")?;
            self.forward_prompt(st, &prompt[cached..], start, true)?
                .ok_or("prefill returned no token")?
        } else {
            st.next_token
                .ok_or("Qwen retained prefix has no next-token frontier")?
        };
        let greedy = sampling.can_use_unmodified_argmax();
        let mut sampler = SamplerState::new(sampling.clone(), prompt).map_err(|e| e.to_string())?;
        if !greedy {
            next = self.sample(st, None, &mut sampler)?;
        }
        let stops = self.weights.tokenizer.stop_tokens();
        let mut pos = prompt.len();
        while output.len() < max_new_tokens {
            if stops.contains(next) {
                break;
            }
            output.push(next);
            sampler.accept(next).map_err(|e| e.to_string())?;
            if !on_token(next) {
                break;
            }
            let position = u32::try_from(pos).map_err(|_| "position exceeds u32")?;
            next = self
                .forward_token(st, next, position, true)?
                .ok_or("decode returned no token")?;
            if !greedy {
                next = self.sample(st, None, &mut sampler)?;
            }
            pos += 1;
        }
        st.next_token = Some(next);
        Ok(output)
    }

    /// The logits of `st`'s last step that computed them.
    pub fn logits(&self, st: &Session) -> Result<Vec<f32>, String> {
        st.check(self)?;
        let mut logits = vec![0f32; self.weights.config.vocab_size as usize];
        self.stream
            .download(st.buffers.logits.as_slice(), &mut logits)
            .map_err(|e| format!("Qwen logits: {e}"))?;
        Ok(logits)
    }
}

/// The Qwen3.8 text model with the state of one sequence.
pub struct Engine {
    model: Model,
    session: Session,
}

impl Engine {
    /// Loads `path` and allocates the state for `max_context` positions.
    pub fn load(path: &Path, max_context: u32) -> Result<Self, String> {
        Self::from_model(Model::load(path, max_context)?)
    }

    pub fn new(stream: Stream, weights: Weights, max_context: u32) -> Result<Self, String> {
        Self::from_model(Model::new(stream, weights, max_context)?)
    }

    fn from_model(model: Model) -> Result<Self, String> {
        let session = model.new_session()?;
        Ok(Self { model, session })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The model and the sequence's state, for callers that drive both
    /// (the serving verifier, images).
    pub fn parts_mut(&mut self) -> (&mut Model, &mut Session) {
        (&mut self.model, &mut self.session)
    }

    pub fn config(&self) -> &ModelConfig {
        self.model.config()
    }

    /// The weights, for a draft model that shares the embedding and LM head.
    pub fn weights(&self) -> &Weights {
        self.model.weights()
    }

    pub fn max_context(&self) -> u32 {
        self.model.max_context
    }

    /// Starts a new sequence (`QwenGpuArena::Reset`).
    pub fn reset(&mut self) -> Result<(), String> {
        self.model.reset(&mut self.session)
    }

    /// See [`Model::forward_token`].
    pub fn forward_token(
        &mut self,
        token: u32,
        pos: u32,
        compute_logits: bool,
    ) -> Result<Option<u32>, String> {
        self.model
            .forward_token(&mut self.session, token, pos, compute_logits)
    }

    /// See [`Model::forward_prompt`].
    pub fn forward_prompt(
        &mut self,
        tokens: &[u32],
        start_pos: u32,
        compute_logits: bool,
    ) -> Result<Option<u32>, String> {
        self.model
            .forward_prompt(&mut self.session, tokens, start_pos, compute_logits)
    }

    /// Greedy generation from a fresh state (`QwenGpuExecutor::Generate`
    /// with greedy sampling): the prompt as one prefill, then one step per
    /// token until a stop token or `max_new_tokens`. `on_token` sees each
    /// generated token and stops generation by returning `false`.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, String> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }
        self.reset()?;
        self.generate_from_prefix(prompt, 0, max_new_tokens, on_token)
    }

    /// Continues a retained sequence (`GenerateFromPrefix`): the engine state
    /// holds `prompt[..cached]`, for example the previous prompt and its
    /// generated tokens. Prefills the rest of `prompt` at its positions, or
    /// with nothing new continues from the retained frontier. A stop token
    /// ends generation without being fed, so a following turn starts with it.
    pub fn generate_from_prefix(
        &mut self,
        prompt: &[u32],
        cached: usize,
        max_new_tokens: usize,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, String> {
        self.model.generate_from_prefix(
            &mut self.session,
            prompt,
            cached,
            max_new_tokens,
            &SamplingConfig::default(),
            on_token,
        )
    }

    /// Copies the state of the first `valid_context` positions
    /// (`SaveSnapshot`).
    pub fn save_snapshot(&mut self, valid_context: u32) -> Result<Snapshot, String> {
        self.model.save_snapshot(&self.session, valid_context)
    }

    /// Returns to the state of `snapshot` (`RestoreSnapshot`); the retained
    /// frontier is cleared.
    pub fn restore_snapshot(&mut self, snapshot: &Snapshot) -> Result<(), String> {
        self.model.restore_snapshot(&mut self.session, snapshot)
    }

    /// See [`Session::set_hidden_capture`].
    pub fn set_hidden_capture(&mut self, enabled: bool, layers: &[u32]) -> Result<(), String> {
        let config = &self.model.weights.config;
        self.session.set_hidden_capture(config, enabled, layers)
    }

    pub fn prompt_hidden(&self) -> Result<&[f32], String> {
        self.model.prompt_hidden()
    }

    pub fn verification_hidden(&self) -> &[f32] {
        self.session.verification_hidden()
    }

    /// See [`Model::last_hidden`].
    pub fn last_hidden(&self) -> Result<Vec<f32>, String> {
        self.model.last_hidden(&self.session)
    }

    /// See [`Model::save_state`].
    pub fn save_state(&mut self, valid_context: u32) -> Result<(), String> {
        self.model.save_state(&mut self.session, valid_context)
    }

    /// See [`Model::restore_state`].
    pub fn restore_state(&mut self) -> Result<(), String> {
        self.model.restore_state(&mut self.session)
    }

    /// See [`Model::finish_verification`].
    pub fn finish_verification(&mut self) -> Result<(), String> {
        self.model.finish_verification(&mut self.session)
    }

    /// See [`Model::verify`].
    pub fn verify(&mut self, tokens: &[u32], start_pos: u32) -> Result<Vec<u32>, String> {
        self.model.verify(&mut self.session, tokens, start_pos)
    }

    /// See [`Model::commit_verified`].
    pub fn commit_verified(&mut self, tokens: &[u32], start_pos: u32) -> Result<(), String> {
        self.model
            .commit_verified(&mut self.session, tokens, start_pos)
    }

    pub fn graph_mismatches(&self) -> u64 {
        self.session.graph_mismatches()
    }

    pub fn captured_graphs(&self) -> usize {
        self.session.captured_graphs()
    }

    /// The logits of the last step that computed them.
    pub fn logits(&self) -> Result<Vec<f32>, String> {
        self.model.logits(&self.session)
    }

    /// Samples the last logits under `sampler` (`SampleLastLogits`).
    pub fn sample(&mut self, sampler: &mut SamplerState) -> Result<u32, String> {
        self.model.sample(&mut self.session, None, sampler)
    }

    /// The session's image input, which the MTP head reads too.
    pub(crate) fn vision(&self) -> &VisionInput {
        &self.session.buffers.vision
    }

    /// See [`Session::set_device_rows`].
    pub fn set_device_rows(&mut self, enabled: bool) {
        self.session.set_device_rows(enabled);
    }

    /// See [`Model::verification_rows`].
    pub fn verification_rows(&self) -> Result<DeviceSlice<'_, f32>, String> {
        self.model.verification_rows(&self.session)
    }

    /// See [`Model::final_row`].
    pub fn final_row(&self) -> Result<DeviceSlice<'_, f32>, String> {
        self.model.final_row(&self.session)
    }
}
