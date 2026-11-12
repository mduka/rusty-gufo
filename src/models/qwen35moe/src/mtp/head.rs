//! The mutable state of one MTP sequence, a port of `QwenMtpGpuExecutor`
//! (`hip/mtp/executor.cpp`).
//!
//! One step fuses the normalised embedding of the input token with the
//! normalised hidden row of the target (or of the head's own previous step),
//! runs the block's attention over its FP32 KV cache and its routed FFN, and
//! scores the result with the target's LM head: all of it, or the small
//! draft head (the head's leading [`DRAFT_VOCAB`] rows read in place, then
//! the session's table of copied rows). The prompt primes the cache with a
//! projection-only batch that writes each row's K and V and nothing else.
//!
//! The C++ executor also fills an FP16 copy of the cache that nothing reads
//! while the FP32 cache is given; it is not kept here.

use gufo_hip::{
    DeviceBuffer, DeviceIndices, DeviceSlice, DeviceSliceMut, DeviceTensor, LaunchError,
    PinnedBuffer, QuantSlice, QuantTensor, Stream,
};
use gufo_qwen35moe_kernels as k;
use gufo_qwen35moe_kernels::sizes;

use crate::arena::alloc;
use crate::config::NORM_EPS;
use crate::moe::{self, MoeScratch};
use crate::mtp::sampling::Candidates;
use crate::mtp::vocab::{DRAFT_EXTRA_ROWS, DRAFT_VOCAB};
use crate::mtp::{MtpContext, MtpModel};

/// Rows per batched prime slice (`kMtpDefaultPrimeChunk`); `0` selects the
/// serial per-token prime.
pub const DEFAULT_PRIME_CHUNK: u32 = 1024;

/// Device slots of one batched draft chain (`kMtpMaxChainTokens`).
pub const MAX_CHAIN_TOKENS: usize = 8;

/// Rows from which a prime projection takes the WMMA GEMM
/// (`kMtpPrimeWmmaMinRows`).
const PRIME_WMMA_MIN_ROWS: usize = 128;

/// Rows per exact BF16 GEMM group below that (`kMtpPrimeExactRows`): its
/// accumulation order is the decode GEMV's and independent of the width.
const PRIME_EXACT_ROWS: usize = 16;

/// Slack past the draft table's rows (`MtpDraftTableBytes`).
const DRAFT_TABLE_SLACK: usize = 4096;

pub(crate) fn launch(error: LaunchError) -> String {
    format!("MTP launch failed: {error}")
}

fn copy_error(error: gufo_hip::HipError) -> String {
    format!("MTP hidden copy failed: {error}")
}

/// The head's input hidden row for one step.
#[derive(Clone, Copy)]
pub(crate) enum Input<'a> {
    /// A host row, copied in first (`ForwardTargetHidden`).
    Host(&'a [f32]),
    /// A row already on the device.
    Device(DeviceSlice<'a, f32>),
    /// The head's own previous output (`ForwardFeedback`).
    Feedback,
}

/// Where a step's greedy token goes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Output {
    /// No logits.
    None,
    /// Only the step's K/V cache row: nothing after the attention launch
    /// that writes it, so the step leaves no output row (the draft's
    /// replays, which only extend the cache; the C++ executor runs them as
    /// whole steps without logits, whose cache rows are the same).
    KvOnly,
    /// The step token, read back by the caller (`d_out_token_`).
    Token,
    /// A slot of the batched draft chain (`d_chain_tokens_`).
    Chain(usize),
}

/// The session's table of output-head rows behind the small head's prefix.
struct DraftRows {
    /// `DRAFT_EXTRA_ROWS` rows of the head's encoding, then the slack.
    rows: DeviceBuffer<u8>,
    ids: DeviceBuffer<i32>,
    /// Host mirror of `ids`, uploaded from the front.
    host_ids: PinnedBuffer<i32>,
    count: u32,
}

/// Batched-prime scratch for `rows` rows.
struct PrimeScratch {
    rows: usize,
    tokens: DeviceIndices,
    hidden: DeviceBuffer<f32>,
    embed: DeviceBuffer<f32>,
    fusion: DeviceBuffer<f32>,
    fusion_out: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
}

/// One sequence of the MTP block (`QwenMtpGpuExecutor`).
pub struct MtpHead {
    max_context: u32,
    next_position: u32,
    target_hidden: DeviceBuffer<f32>,
    embedding: DeviceBuffer<f32>,
    fusion: DeviceBuffer<f32>,
    hidden: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    qg: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    context: DeviceBuffer<f32>,
    attn_out: DeviceBuffer<f32>,
    moe: MoeScratch,
    ffn_out: DeviceBuffer<f32>,
    feedback: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    /// Candidate IDs, then their scores (as f32 bits) from
    /// [`sizes::MTP_CANDIDATES`] on.
    candidates: DeviceBuffer<u32>,
    candidate_scratch: DeviceBuffer<u32>,
    /// `[kv head][max_context][head_dim]`.
    k_cache: DeviceBuffer<f32>,
    v_cache: DeviceBuffer<f32>,
    split_k: DeviceBuffer<f32>,
    token: DeviceBuffer<u32>,
    chain: DeviceBuffer<u32>,
    draft: Option<DraftRows>,
    prime: Option<PrimeScratch>,
    scratch_bytes: usize,
}

/// The committed cache prefix of a head (`QwenMtpGpuSnapshot`): the key
/// plane, then the value plane, each packed per head.
pub struct HeadSnapshot {
    planes: Option<DeviceTensor<f32>>,
    valid_context: u32,
    max_context: u32,
    kv_width: u32,
    head_dim: u32,
    num_kv_heads: u32,
}

/// `GMTPGPU0`, the persistent form of a [`HeadSnapshot`].
const HEAD_MAGIC: [u8; 8] = *b"GMTPGPU0";
const HEAD_VERSION: u32 = 1;
pub(crate) const HEAD_HEADER_BYTES: usize = 64;

pub(crate) fn put_u32(destination: &mut [u8], offset: usize, value: u32) {
    destination[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

pub(crate) fn put_u64(destination: &mut [u8], offset: usize, value: u64) {
    destination[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

pub(crate) fn get_u32(source: &[u8], offset: usize) -> Option<u32> {
    let bytes = source.get(offset..offset + 4)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

pub(crate) fn get_u64(source: &[u8], offset: usize) -> Option<u64> {
    let bytes = source.get(offset..offset + 8)?;
    Some(u64::from_le_bytes(bytes.try_into().ok()?))
}

impl HeadSnapshot {
    /// The packed planes' bytes (`PayloadBytes`).
    pub fn payload_bytes(&self) -> usize {
        2 * self.num_kv_heads as usize
            * self.valid_context as usize
            * self.head_dim as usize
            * size_of::<f32>()
    }

    /// The draft context this snapshot restores (`ValidContext`).
    pub fn valid_context(&self) -> u32 {
        self.valid_context
    }

    /// `PersistentPayloadBytes`.
    pub fn persistent_bytes(&self) -> usize {
        HEAD_HEADER_BYTES + self.payload_bytes()
    }

    /// Writes the persistent form into `destination` of exactly
    /// [`HeadSnapshot::persistent_bytes`] (`SerializePersistent`).
    pub fn serialize(&self, stream: &Stream, destination: &mut [u8]) -> Result<usize, String> {
        let expected = self.persistent_bytes();
        if destination.len() != expected
            || self.head_dim == 0
            || self.num_kv_heads == 0
            || self.kv_width != self.num_kv_heads * self.head_dim
            || self.max_context == 0
            || self.valid_context > self.max_context
        {
            return Err("MTP GPU snapshot metadata is malformed".into());
        }
        let payload = self.payload_bytes();
        if payload != 0 && self.planes.is_none() {
            return Err("MTP GPU snapshot payload is malformed".into());
        }
        destination.fill(0);
        destination[..8].copy_from_slice(&HEAD_MAGIC);
        put_u32(destination, 8, HEAD_VERSION);
        put_u32(destination, 12, HEAD_HEADER_BYTES as u32);
        put_u32(destination, 16, self.kv_width);
        put_u32(destination, 20, self.head_dim);
        put_u32(destination, 24, self.num_kv_heads);
        put_u32(destination, 28, self.max_context);
        put_u32(destination, 32, self.valid_context);
        put_u32(destination, 36, 0);
        put_u64(
            destination,
            40,
            u64::from(self.valid_context) * u64::from(self.head_dim),
        );
        put_u64(destination, 48, payload as u64);
        put_u64(destination, 56, expected as u64);
        if let Some(planes) = &self.planes {
            stream
                .download_bytes(planes.view(), &mut destination[HEAD_HEADER_BYTES..])
                .map_err(|e| format!("MTP GPU snapshot download failed: {e}"))?;
        }
        Ok(expected)
    }
}

impl MtpHead {
    /// The state for `max_context` positions with batched-prime slices of
    /// `prime_chunk` rows (`QwenMtpGpuExecutor::Create`).
    pub fn new(ctx: &MtpContext<'_>, max_context: u32, prime_chunk: u32) -> Result<Self, String> {
        let (s, model) = (ctx.stream, ctx.model);
        let c = &model.config;
        if max_context == 0 || max_context > c.context_length {
            return Err("MTP GPU executor context is invalid".into());
        }
        let prime_chunk = prime_chunk.min(max_context) as usize;
        let hidden = c.hidden_size as usize;
        let attention = c.attention_size() as usize;
        let kv = c.kv_size() as usize;
        let vocab = c.vocab_size as usize;
        let cache = kv * max_context as usize;
        let workspace = sizes::mtp_candidate_workspace(vocab);
        let mut bytes = 0usize;
        let mut scratch = |len: usize, size: usize| {
            bytes += len * size;
            len
        };
        let f = size_of::<f32>();
        let mut head = Self {
            max_context,
            next_position: 0,
            target_hidden: alloc(s, scratch(hidden, f))?,
            embedding: alloc(s, scratch(hidden, f))?,
            fusion: alloc(s, scratch(2 * hidden, f))?,
            hidden: alloc(s, scratch(hidden, f))?,
            normed: alloc(s, scratch(hidden, f))?,
            qg: alloc(s, scratch(2 * attention, f))?,
            q: alloc(s, scratch(attention, f))?,
            k: alloc(s, scratch(kv, f))?,
            v: alloc(s, scratch(kv, f))?,
            gate: alloc(s, scratch(attention, f))?,
            context: alloc(s, scratch(attention, f))?,
            attn_out: alloc(s, scratch(hidden, f))?,
            moe: MoeScratch::new(s, c, 1)?,
            ffn_out: alloc(s, scratch(hidden, f))?,
            feedback: alloc(s, scratch(hidden, f))?,
            logits: alloc(s, scratch(vocab, f))?,
            candidates: alloc(s, scratch(workspace, 4))?,
            candidate_scratch: alloc(s, scratch(workspace, 4))?,
            // The attention cache is request state, not scratch.
            k_cache: alloc(s, cache)?,
            v_cache: alloc(s, cache)?,
            split_k: alloc(
                s,
                scratch(
                    sizes::decode_attention_scratch(
                        c.num_attention_heads as usize,
                        c.head_dim as usize,
                    ),
                    f,
                ),
            )?,
            token: alloc(s, scratch(1, 4))?,
            chain: alloc(s, scratch(MAX_CHAIN_TOKENS, 4))?,
            draft: None,
            prime: None,
            scratch_bytes: 0,
        };
        head.scratch_bytes = bytes;
        if prime_chunk != 0 {
            let rows = prime_chunk;
            let error = |e| format!("Qwen device buffer: {e}");
            head.prime = Some(PrimeScratch {
                rows,
                tokens: s
                    .zeroed_indices(alloc::<u32>(s, rows)?, c.vocab_size)
                    .map_err(error)?,
                hidden: alloc(s, rows * hidden)?,
                embed: alloc(s, rows * hidden)?,
                fusion: alloc(s, rows * 2 * hidden)?,
                fusion_out: alloc(s, rows * hidden)?,
                normed: alloc(s, rows * hidden)?,
                k: alloc(s, rows * kv)?,
                v: alloc(s, rows * kv)?,
            });
            head.scratch_bytes +=
                rows * 4 + rows * hidden * f * 4 + rows * 2 * hidden * f + rows * 2 * kv * f;
        }
        if model.has_draft_vocab() {
            let extra = DRAFT_EXTRA_ROWS as usize;
            let table = extra * model.output_row_bytes() + DRAFT_TABLE_SLACK;
            head.draft = Some(DraftRows {
                rows: alloc(s, table)?,
                ids: alloc(s, extra)?,
                host_ids: PinnedBuffer::zeroed(extra)
                    .map_err(|e| format!("MTP draft table: {e}"))?,
                count: 0,
            });
            head.scratch_bytes += table + extra * 4;
            // The table opens with the special tokens; every request keeps
            // them.
            head.append_draft_rows(ctx, model.draft_specials())?;
        }
        Ok(head)
    }

    /// Starts a new sequence; attention reads only the valid prefix, so the
    /// cache stays (`Reset`).
    pub fn reset(&mut self) {
        self.next_position = 0;
    }

    /// Returns to `position`, dropping the rows after it (`Rewind`).
    pub fn rewind(&mut self, position: u32) -> Result<(), String> {
        if position > self.next_position {
            return Err("MTP rewind position exceeds executed context".into());
        }
        self.next_position = position;
        Ok(())
    }

    pub fn next_position(&self) -> u32 {
        self.next_position
    }

    pub fn max_context(&self) -> u32 {
        self.max_context
    }

    /// Resident request state: both FP32 cache planes (`StateBytes`).
    pub fn state_bytes(&self) -> usize {
        (self.k_cache.len() + self.v_cache.len()) * size_of::<f32>()
    }

    /// Device bytes of the scratch buffers.
    pub fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }

    /// Bytes a snapshot of the current prefix holds (`SnapshotPayloadBytes`).
    pub fn snapshot_payload_bytes(&self, model: &MtpModel) -> usize {
        2 * model.config.kv_size() as usize * self.next_position as usize * size_of::<f32>()
    }

    /// One step on a host row (`ForwardTargetHidden`); with
    /// `compute_logits`, returns the head's greedy token (over the small
    /// head with `small_head`), else 0.
    pub fn forward_target_hidden(
        &mut self,
        ctx: &MtpContext<'_>,
        token: u32,
        target_hidden: &[f32],
        position: u32,
        compute_logits: bool,
        small_head: bool,
    ) -> Result<u32, String> {
        if target_hidden.len() != ctx.model.config.hidden_size as usize {
            return Err("MTP target hidden width is invalid".into());
        }
        self.run(
            ctx,
            token,
            Input::Host(target_hidden),
            position,
            compute_logits,
            small_head,
        )
    }

    /// [`MtpHead::forward_target_hidden`] on a row already on the device.
    pub fn forward_row(
        &mut self,
        ctx: &MtpContext<'_>,
        token: u32,
        row: DeviceSlice<'_, f32>,
        position: u32,
        compute_logits: bool,
        small_head: bool,
    ) -> Result<u32, String> {
        if row.len() != ctx.model.config.hidden_size as usize {
            return Err("MTP target hidden width is invalid".into());
        }
        self.run(
            ctx,
            token,
            Input::Device(row),
            position,
            compute_logits,
            small_head,
        )
    }

    /// The K/V cache row of one step on a device row and nothing else: what
    /// a step without logits leaves for later steps, as the draft's
    /// feedback replays use it. The step's output row is not computed, so
    /// no feedback step may follow it.
    pub fn replay_row(
        &mut self,
        ctx: &MtpContext<'_>,
        token: u32,
        row: DeviceSlice<'_, f32>,
        position: u32,
    ) -> Result<(), String> {
        if row.len() != ctx.model.config.hidden_size as usize {
            return Err("MTP target hidden width is invalid".into());
        }
        self.step(
            ctx,
            token,
            Input::Device(row),
            position,
            Output::KvOnly,
            false,
        )
    }

    /// One step on the head's previous output (`ForwardFeedback`).
    pub fn forward_feedback(
        &mut self,
        ctx: &MtpContext<'_>,
        token: u32,
        position: u32,
        compute_logits: bool,
        small_head: bool,
    ) -> Result<u32, String> {
        self.run(
            ctx,
            token,
            Input::Feedback,
            position,
            compute_logits,
            small_head,
        )
    }

    /// The logits of the last step that computed them (`CopyLastLogits`); a
    /// small-head step leaves them partial.
    pub fn logits(&self, stream: &Stream) -> Result<Vec<f32>, String> {
        let mut logits = vec![0f32; self.logits.len()];
        stream
            .download(self.logits.as_slice(), &mut logits)
            .map_err(|e| format!("MTP logit synchronization failed: {e}"))?;
        Ok(logits)
    }

    /// The head's last output row (`CopyLastHidden`).
    pub fn last_hidden(&self, stream: &Stream) -> Result<Vec<f32>, String> {
        let mut row = vec![0f32; self.feedback.len()];
        stream
            .download(self.feedback.as_slice(), &mut row)
            .map_err(|e| format!("MTP hidden synchronization failed: {e}"))?;
        Ok(row)
    }

    /// The exact top candidates of the full head's last logits, descending
    /// with lowest-ID ties (`TopCandidates`). Needs a full-head step.
    pub fn top_candidates(&mut self, ctx: &MtpContext<'_>) -> Result<Candidates, String> {
        let s = ctx.stream;
        let vocab = ctx.model.config.vocab_size;
        let workspace = self.candidates.len();
        k::mtp_top_candidates(
            s,
            self.logits.as_slice(),
            self.candidates.as_mut_slice(),
            self.candidate_scratch.as_mut_slice(),
            vocab,
            workspace as u32,
        )
        .map_err(launch)?;
        let n = sizes::MTP_CANDIDATES;
        let size = (vocab as usize).min(n);
        let mut packed = vec![0u32; n + size];
        s.download(self.candidates.slice(0..n + size), &mut packed)
            .map_err(|e| format!("MTP candidate synchronization failed: {e}"))?;
        let mut candidates = Candidates {
            ids: [0; sizes::MTP_CANDIDATES],
            logits: [0.0; sizes::MTP_CANDIDATES],
            size,
        };
        candidates.ids.copy_from_slice(&packed[..n]);
        for (logit, &bits) in candidates.logits.iter_mut().zip(&packed[n..]) {
            *logit = f32::from_bits(bits);
        }
        Ok(candidates)
    }

    /// `QwenMtpGpuExecutor::Run`: one step whose greedy token is read back.
    fn run(
        &mut self,
        ctx: &MtpContext<'_>,
        token: u32,
        input: Input<'_>,
        position: u32,
        compute_logits: bool,
        small_head: bool,
    ) -> Result<u32, String> {
        if position != self.next_position || position >= self.max_context {
            return Err("MTP GPU position is not sequential".into());
        }
        let output = if compute_logits {
            Output::Token
        } else {
            Output::None
        };
        self.step(ctx, token, input, position, output, small_head)?;
        if !compute_logits {
            return Ok(0);
        }
        self.read_token(ctx, Output::Token)
    }

    /// The greedy token of a step, checked against the vocabulary
    /// (`CheckedSampleToken`).
    pub(crate) fn read_token(&self, ctx: &MtpContext<'_>, output: Output) -> Result<u32, String> {
        let source = match output {
            Output::Chain(slot) => self.chain.slice(slot..slot + 1),
            _ => self.token.as_slice(),
        };
        let mut result = [0u32];
        ctx.stream
            .download_blocking(source, &mut result)
            .map_err(|_| "MTP GPU result synchronization failed".to_string())?;
        if result[0] >= ctx.model.config.vocab_size {
            return Err("logit distribution contains no finite values".into());
        }
        Ok(result[0])
    }

    /// One full draft step (`RunDraftForward`) on the host `token`.
    pub(crate) fn step(
        &mut self,
        ctx: &MtpContext<'_>,
        token: u32,
        input: Input<'_>,
        position: u32,
        output: Output,
        small_head: bool,
    ) -> Result<(), String> {
        if position != self.next_position || position >= self.max_context {
            return Err("MTP GPU position is not sequential".into());
        }
        if small_head && self.draft.is_none() {
            return Err("MTP GPU small draft head needs a draft table".into());
        }
        if let Output::Chain(slot) = output
            && slot >= MAX_CHAIN_TOKENS
        {
            return Err("MTP GPU chain slot is outside the budget".into());
        }
        let (s, m, vision) = (ctx.stream, ctx.model, ctx.vision);
        let c = &m.config;
        let hidden = c.hidden_size as usize;
        let attention = c.attention_size() as usize;
        let kv = c.kv_size() as usize;
        let vocab = c.vocab_size as usize;
        if let Input::Host(row) = input {
            s.upload(row, self.target_hidden.as_mut_slice())
                .map_err(copy_error)?;
        }

        k::embedding_lookup_token(
            s,
            m.token_embd.view(),
            token,
            self.embedding.as_mut_slice(),
            hidden,
            vocab,
        )
        .map_err(launch)?;
        if let Some(vision) = vision {
            vision
                .inject(
                    s,
                    self.embedding.as_mut_slice(),
                    position + 1,
                    1,
                    hidden as u32,
                    1,
                )
                .map_err(launch)?;
        }
        k::rms_norm(
            s,
            self.embedding.as_slice(),
            m.embedding_norm.view(),
            self.fusion.slice_mut(0..hidden),
            hidden,
            NORM_EPS,
        )
        .map_err(launch)?;
        let source = match input {
            Input::Host(_) => self.target_hidden.as_slice(),
            Input::Device(row) => row,
            Input::Feedback => self.feedback.as_slice(),
        };
        k::rms_norm(
            s,
            source,
            m.hidden_norm.view(),
            self.fusion.slice_mut(hidden..2 * hidden),
            hidden,
            NORM_EPS,
        )
        .map_err(launch)?;
        k::gemv_mtp(
            s,
            m.fusion.view(),
            self.fusion.as_slice(),
            self.hidden.as_mut_slice(),
            hidden,
            2 * hidden,
        )
        .map_err(launch)?;

        k::rms_norm(
            s,
            self.hidden.as_slice(),
            m.attn_norm.view(),
            self.normed.as_mut_slice(),
            hidden,
            NORM_EPS,
        )
        .map_err(launch)?;
        k::fused_qkv_projections(
            s,
            m.q.view(),
            m.k.view(),
            m.v.view(),
            self.normed.as_slice(),
            self.qg.as_mut_slice(),
            self.k.as_mut_slice(),
            self.v.as_mut_slice(),
            2 * attention,
            kv,
            hidden,
        )
        .map_err(launch)?;
        k::unpack_qg(
            s,
            self.qg.as_slice(),
            self.q.as_mut_slice(),
            self.gate.as_mut_slice(),
            c.num_attention_heads,
            c.head_dim,
        )
        .map_err(launch)?;
        k::per_head_rms_norm(
            s,
            m.q_norm.view(),
            self.q.as_mut_slice(),
            c.num_attention_heads,
            c.head_dim,
            NORM_EPS,
        )
        .map_err(launch)?;
        k::per_head_rms_norm(
            s,
            m.k_norm.view(),
            self.k.as_mut_slice(),
            c.num_key_value_heads,
            c.head_dim,
            NORM_EPS,
        )
        .map_err(launch)?;
        k::rope(
            s,
            self.q.as_mut_slice(),
            self.k.as_mut_slice(),
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            c.rotary_dim,
            position,
            c.rope_theta,
            vision.and_then(|v| v.rope()),
        )
        .map_err(launch)?;
        k::attention_write_fp32(
            s,
            self.q.as_slice(),
            self.k.as_slice(),
            self.v.as_slice(),
            self.gate.as_slice(),
            self.k_cache.as_mut_slice(),
            self.v_cache.as_mut_slice(),
            self.context.as_mut_slice(),
            0,
            position,
            self.max_context,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            self.split_k.as_mut_slice(),
        )
        .map_err(launch)?;
        if output == Output::KvOnly {
            self.next_position += 1;
            return Ok(());
        }
        k::gemv_mtp(
            s,
            m.output.view(),
            self.context.as_slice(),
            self.attn_out.as_mut_slice(),
            hidden,
            attention,
        )
        .map_err(launch)?;
        k::residual_add(
            s,
            self.attn_out.as_slice(),
            self.hidden.as_mut_slice(),
            hidden,
        )
        .map_err(launch)?;

        k::rms_norm(
            s,
            self.hidden.as_slice(),
            m.ffn_norm.view(),
            self.normed.as_mut_slice(),
            hidden,
            NORM_EPS,
        )
        .map_err(launch)?;
        // The block of a MoE target is a MoE block too; it runs the target
        // decode step's one-token kernels.
        moe::decode(
            s,
            &m.moe,
            c,
            &mut self.moe,
            self.normed.as_slice(),
            Some(self.ffn_out.as_mut_slice()),
        )
        .map_err(launch)?;
        k::residual_add(
            s,
            self.ffn_out.as_slice(),
            self.hidden.as_mut_slice(),
            hidden,
        )
        .map_err(launch)?;
        k::rms_norm(
            s,
            self.hidden.as_slice(),
            m.shared_head_norm.view(),
            self.feedback.as_mut_slice(),
            hidden,
            NORM_EPS,
        )
        .map_err(launch)?;
        self.next_position += 1;

        let out_token = match output {
            Output::None | Output::KvOnly => return Ok(()),
            Output::Token => self.token.as_mut_slice(),
            Output::Chain(slot) => self.chain.slice_mut(slot..slot + 1),
        };
        let (Some(draft), true) = (&self.draft, small_head) else {
            k::gemv_mtp(
                s,
                m.lm_head.view(),
                self.feedback.as_slice(),
                self.logits.as_mut_slice(),
                vocab,
                hidden,
            )
            .map_err(launch)?;
            return k::argmax(s, self.logits.as_slice(), out_token, vocab).map_err(launch);
        };
        // The small head: the prefix is the output head's leading rows, read
        // in place, and the table rows follow it in the logits. The remap
        // turns a table index back into its token ID.
        let prefix = DRAFT_VOCAB as usize;
        let rows = draft.count as usize;
        k::gemv_mtp(
            s,
            m.lm_head.view(),
            self.feedback.as_slice(),
            self.logits.slice_mut(0..prefix),
            prefix,
            hidden,
        )
        .map_err(launch)?;
        if rows != 0 {
            let row_bytes = m.output_row_bytes();
            k::gemv_mtp(
                s,
                QuantSlice::new(draft.rows.slice(0..rows * row_bytes), m.lm_head.format()),
                self.feedback.as_slice(),
                self.logits.slice_mut(prefix..prefix + rows),
                rows,
                hidden,
            )
            .map_err(launch)?;
        }
        let mut out_token = out_token;
        k::argmax(
            s,
            self.logits.slice(0..prefix + rows),
            out_token.reborrow(),
            prefix + rows,
        )
        .map_err(launch)?;
        k::remap_draft_token(
            s,
            out_token,
            DRAFT_VOCAB,
            draft.ids.slice(0..rows),
            rows as u32,
        )
        .map_err(launch)
    }

    /// Draft-table rows in use: the model's special tokens, then the rows
    /// appended since the last reset (`GetDraftRows`).
    pub fn draft_rows(&self) -> u32 {
        self.draft.as_ref().map_or(0, |d| d.count)
    }

    /// Keeps only the special-token rows (`ResetDraftRows`).
    pub fn reset_draft_rows(&mut self, ctx: &MtpContext<'_>) {
        if let Some(draft) = &mut self.draft {
            draft.count = ctx.model.draft_specials().len() as u32;
        }
    }

    /// Copies the output-head rows of `ids` into the table after the rows in
    /// use (`AppendDraftRows`).
    pub fn append_draft_rows(&mut self, ctx: &MtpContext<'_>, ids: &[i32]) -> Result<(), String> {
        if ids.is_empty() {
            return Ok(());
        }
        let s = ctx.stream;
        let m = ctx.model;
        let Some(draft) = &mut self.draft else {
            return Err("MTP GPU draft table append is invalid".into());
        };
        let count = draft.count as usize;
        if ids.len() > DRAFT_EXTRA_ROWS as usize - count {
            return Err("MTP GPU draft table append is invalid".into());
        }
        let vocab = m.config.vocab_size as usize;
        if ids.iter().any(|&id| id < 0 || id as usize >= vocab) {
            return Err("MTP GPU draft table token is out of range".into());
        }
        let row_bytes = m.output_row_bytes();
        let head = m.lm_head.bytes();
        let error = |e| format!("MTP GPU draft table copy failed: {e}");
        for (index, &id) in ids.iter().enumerate() {
            let source = id as usize * row_bytes;
            let target = (count + index) * row_bytes;
            s.copy(
                head.slice(source..source + row_bytes),
                draft.rows.slice_mut(target..target + row_bytes),
            )
            .map_err(error)?;
        }
        let end = count + ids.len();
        draft.host_ids.host_mut().map_err(error)?[count..end].copy_from_slice(ids);
        s.upload_async_front(&mut draft.host_ids, end, draft.ids.slice_mut(0..end))
            .map_err(error)?;
        draft.count = end as u32;
        Ok(())
    }

    /// Batched, projection-only prime (`PrimeTargetHiddenBatch`): writes the
    /// K and V rows of `successor_tokens.len()` positions from
    /// `start_position` on, from the target rows that own them;
    /// `successor_tokens[i]` follows the position of row `i`. `fill` copies
    /// rows `offset..offset + rows` into its staging slice. The cursor ends
    /// at `start_position + count`.
    pub(crate) fn prime_rows(
        &mut self,
        ctx: &MtpContext<'_>,
        start_position: u32,
        successor_tokens: &[u32],
        mut fill: impl FnMut(&Stream, usize, usize, DeviceSliceMut<'_, f32>) -> Result<(), String>,
    ) -> Result<(), String> {
        let count = successor_tokens.len();
        if count == 0 {
            return Ok(());
        }
        let Some(prime) = &mut self.prime else {
            return Err("MTP batched prime needs a non-zero chunk".into());
        };
        if start_position > self.max_context || count > (self.max_context - start_position) as usize
        {
            return Err("MTP prime exceeds the executor context".into());
        }
        let (s, m, vision) = (ctx.stream, ctx.model, ctx.vision);
        let c = &m.config;
        let hidden = c.hidden_size as usize;
        let kv = c.kv_size() as usize;
        let vocab = c.vocab_size as usize;
        let mut offset = 0;
        while offset < count {
            let rows = prime.rows.min(count - offset);
            let row_base = start_position + offset as u32;
            fill(s, offset, rows, prime.hidden.slice_mut(0..rows * hidden))?;
            s.upload_indices(&successor_tokens[offset..offset + rows], &mut prime.tokens)
                .map_err(|e| format!("MTP prime token copy failed: {e}"))?;
            k::batched_embedding_lookup(
                s,
                m.token_embd.view(),
                prime.tokens.slice(0..rows),
                prime.embed.slice_mut(0..rows * hidden),
                rows,
                hidden,
                vocab,
            )
            .map_err(launch)?;
            if let Some(vision) = vision {
                // The row at position `p` injects at `p + 1`, as a step does.
                vision
                    .inject(
                        s,
                        prime.embed.slice_mut(0..rows * hidden),
                        row_base + 1,
                        rows as u32,
                        hidden as u32,
                        1,
                    )
                    .map_err(launch)?;
            }
            let fusion_len = (rows - 1) * 2 * hidden + hidden;
            k::batched_rms_norm_strided(
                s,
                prime.embed.slice(0..rows * hidden),
                m.embedding_norm.view(),
                prime.fusion.slice_mut(0..fusion_len),
                2 * hidden,
                rows,
                hidden,
                NORM_EPS,
            )
            .map_err(launch)?;
            k::batched_rms_norm_strided(
                s,
                prime.hidden.slice(0..rows * hidden),
                m.hidden_norm.view(),
                prime.fusion.slice_mut(hidden..hidden + fusion_len),
                2 * hidden,
                rows,
                hidden,
                NORM_EPS,
            )
            .map_err(launch)?;
            prime_gemm(
                s,
                &m.fusion,
                prime.fusion.slice(0..rows * 2 * hidden),
                prime.fusion_out.slice_mut(0..rows * hidden),
                rows,
                hidden,
                2 * hidden,
            )?;
            k::batched_rms_norm(
                s,
                prime.fusion_out.slice(0..rows * hidden),
                m.attn_norm.view(),
                prime.normed.slice_mut(0..rows * hidden),
                rows,
                hidden,
                NORM_EPS,
            )
            .map_err(launch)?;
            prime_gemm(
                s,
                &m.k,
                prime.normed.slice(0..rows * hidden),
                prime.k.slice_mut(0..rows * kv),
                rows,
                kv,
                hidden,
            )?;
            prime_gemm(
                s,
                &m.v,
                prime.normed.slice(0..rows * hidden),
                prime.v.slice_mut(0..rows * kv),
                rows,
                kv,
                hidden,
            )?;
            // K/V write only: the normed K lands back in `prime.k`, which
            // nothing reads; attention reads the cache.
            k::mtp_prime_k_norm_rope_kv_write(
                s,
                prime.v.slice(0..rows * kv),
                m.k_norm.view(),
                prime.k.slice_mut(0..rows * kv),
                self.k_cache.as_mut_slice(),
                self.v_cache.as_mut_slice(),
                0,
                row_base,
                rows,
                self.max_context,
                c.num_key_value_heads,
                c.head_dim,
                c.rotary_dim,
                c.rope_theta,
                NORM_EPS,
                vision.and_then(|v| v.rope()),
            )
            .map_err(launch)?;
            offset += rows;
        }
        self.next_position = start_position + count as u32;
        Ok(())
    }

    /// [`MtpHead::prime_rows`] from host rows, `[row][hidden]`
    /// (`PrimeTargetHiddenBatch`).
    pub fn prime_batch(
        &mut self,
        ctx: &MtpContext<'_>,
        start_position: u32,
        successor_tokens: &[u32],
        hidden_rows: &[f32],
    ) -> Result<(), String> {
        let width = ctx.model.config.hidden_size as usize;
        if hidden_rows.len() != successor_tokens.len() * width {
            return Err("MTP prime hidden rows are the wrong shape".into());
        }
        self.prime_rows(
            ctx,
            start_position,
            successor_tokens,
            |s, offset, rows, staging| {
                s.upload(
                    &hidden_rows[offset * width..(offset + rows) * width],
                    staging,
                )
                .map_err(copy_error)
            },
        )
    }

    /// Captures the committed cache prefix, packed per head (`SaveSnapshot`),
    /// copied on `transfer` and waited for. The caller orders the head's
    /// queued work before it, as for every cross-stream copy.
    pub fn save_snapshot(
        &self,
        transfer: &Stream,
        model: &MtpModel,
    ) -> Result<HeadSnapshot, String> {
        let c = &model.config;
        let (head_dim, heads) = (c.head_dim as usize, c.num_key_value_heads as usize);
        let mut snapshot = HeadSnapshot {
            planes: None,
            valid_context: self.next_position,
            max_context: self.max_context,
            kv_width: (heads * head_dim) as u32,
            head_dim: head_dim as u32,
            num_kv_heads: heads as u32,
        };
        let row = self.next_position as usize * head_dim;
        let plane = heads * row;
        if plane == 0 {
            return Ok(snapshot);
        }
        let error = |e| format!("MTP snapshot copy failed: {e}");
        let mut planes = alloc::<f32>(transfer, 2 * plane)?;
        let stride = self.max_context as usize * head_dim;
        let mut copies = Vec::with_capacity(2 * heads);
        let mut rest = planes.as_mut_slice();
        for cache in [&self.k_cache, &self.v_cache] {
            for head in 0..heads {
                let (target, tail) = rest.split_at(row);
                rest = tail;
                let source = head * stride;
                copies.push((cache.slice(source..source + row), target));
            }
        }
        transfer.copy_across_all(&mut copies).map_err(error)?;
        drop(copies);
        snapshot.planes = Some(planes.freeze(transfer).map_err(error)?);
        Ok(snapshot)
    }

    /// Returns to a snapshot's prefix (`RestoreSnapshot`).
    pub fn restore_snapshot(
        &mut self,
        ctx: &MtpContext<'_>,
        snapshot: &HeadSnapshot,
    ) -> Result<(), String> {
        let c = &ctx.model.config;
        if snapshot.num_kv_heads != c.num_key_value_heads
            || snapshot.kv_width != c.kv_size()
            || snapshot.head_dim != c.head_dim
            || snapshot.max_context != self.max_context
            || snapshot.valid_context > self.max_context
        {
            return Err("MTP snapshot is incompatible with the executor".into());
        }
        if snapshot.payload_bytes() != 0 && snapshot.planes.is_none() {
            return Err("MTP snapshot payload is incomplete".into());
        }
        if let Some(planes) = &snapshot.planes {
            let s = ctx.stream;
            let (head_dim, heads) = (c.head_dim as usize, c.num_key_value_heads as usize);
            let row = snapshot.valid_context as usize * head_dim;
            let plane = heads * row;
            let stride = self.max_context as usize * head_dim;
            let error = |e| format!("MTP snapshot restore failed: {e}");
            for head in 0..heads {
                let target = head * stride;
                s.copy(
                    planes.slice(head * row..(head + 1) * row),
                    self.k_cache.slice_mut(target..target + row),
                )
                .map_err(error)?;
                s.copy(
                    planes.slice(plane + head * row..plane + (head + 1) * row),
                    self.v_cache.slice_mut(target..target + row),
                )
                .map_err(error)?;
            }
        }
        self.next_position = snapshot.valid_context;
        Ok(())
    }

    /// Restores a [`HeadSnapshot::serialize`] payload
    /// (`RestorePersistentSnapshot`).
    pub fn restore_persistent(
        &mut self,
        ctx: &MtpContext<'_>,
        payload: &[u8],
    ) -> Result<(), String> {
        let invalid = || "MTP GPU persistent header is invalid".to_string();
        if payload.len() < HEAD_HEADER_BYTES
            || payload[..8] != HEAD_MAGIC
            || get_u32(payload, 8) != Some(HEAD_VERSION)
            || get_u32(payload, 12) != Some(HEAD_HEADER_BYTES as u32)
            || get_u32(payload, 36) != Some(0)
        {
            return Err(invalid());
        }
        let field = |offset| get_u32(payload, offset).ok_or_else(invalid);
        let (kv_width, head_dim, heads) = (field(16)?, field(20)?, field(24)?);
        let (max_context, valid_context) = (field(28)?, field(32)?);
        let wide = |offset| {
            get_u64(payload, offset)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(invalid)
        };
        let (per_head, payload_bytes, total) = (wide(40)?, wide(48)?, wide(56)?);
        let c = &ctx.model.config;
        if head_dim == 0
            || heads == 0
            || head_dim != c.head_dim
            || heads != c.num_key_value_heads
            || kv_width != c.kv_size()
            || max_context != self.max_context
            || valid_context > self.max_context
            || per_head != valid_context as usize * head_dim as usize
            || total != payload.len()
            || payload_bytes != 2 * heads as usize * per_head * size_of::<f32>()
            || HEAD_HEADER_BYTES + payload_bytes != payload.len()
        {
            return Err("MTP GPU persistent metadata is incompatible".into());
        }
        let s = ctx.stream;
        let data = &payload[HEAD_HEADER_BYTES..];
        let row_bytes = per_head * size_of::<f32>();
        let plane = heads as usize * row_bytes;
        let stride = self.max_context as usize * head_dim as usize;
        let error = |e| format!("MTP snapshot restore failed: {e}");
        for head in 0..heads as usize {
            if per_head == 0 {
                continue;
            }
            let target = head * stride;
            s.upload_bytes(
                &data[head * row_bytes..(head + 1) * row_bytes],
                self.k_cache.slice_mut(target..target + per_head),
            )
            .map_err(error)?;
            s.upload_bytes(
                &data[plane + head * row_bytes..plane + (head + 1) * row_bytes],
                self.v_cache.slice_mut(target..target + per_head),
            )
            .map_err(error)?;
        }
        self.next_position = valid_context;
        Ok(())
    }
}

/// `out[rows][m] = x[rows][k] w[m][k]^T` over the BF16 block weights
/// (`PrimeGemm`): the WMMA GEMM from [`PRIME_WMMA_MIN_ROWS`] rows, else the
/// exact GEMM in groups of [`PRIME_EXACT_ROWS`].
fn prime_gemm(
    s: &Stream,
    weight: &QuantTensor,
    x: DeviceSlice<'_, f32>,
    mut out: DeviceSliceMut<'_, f32>,
    rows: usize,
    m: usize,
    k_dim: usize,
) -> Result<(), String> {
    let w = weight
        .bytes()
        .view()
        .cast::<gufo_hip::Bf16>()
        .filter(|_| weight.format() == gufo_hip::GgmlType::Bf16)
        .ok_or("MTP prime requires BF16 weights")?;
    if rows >= PRIME_WMMA_MIN_ROWS && sizes::grouped_bf16_gemm(m, k_dim, 1) {
        return k::bf16_wmma_gemm(s, w, x, out, rows, m, k_dim).map_err(launch);
    }
    let mut offset = 0;
    while offset < rows {
        let group = PRIME_EXACT_ROWS.min(rows - offset);
        k::exact_bf16_gemm_f32(
            s,
            w,
            x.range(offset * k_dim..(offset + group) * k_dim),
            out.reborrow().range(offset * m..(offset + group) * m),
            group,
            m,
            k_dim,
        )
        .map_err(launch)?;
        offset += group;
    }
    Ok(())
}
