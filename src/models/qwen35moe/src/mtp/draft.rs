//! The MTP draft of one sequence, a port of `QwenMtpGpuDraftBackend`
//! (`hip/mtp/backend.cpp`): primed with the target's prompt rows, it
//! proposes greedy blocks (over the small draft head when the session's
//! table fits) or sampled ones (from the full head's top candidates, with
//! the distribution they were drawn from), batches greedy proposals of
//! several sessions, and once a block is verified replays the committed rows
//! on the target's hidden states.
//!
//! The C++ backend keeps its target rows on the host and copies each one in;
//! these stay on the device, and verified and decoded rows are copied there
//! in the stream's order.

use gufo_hip::{DeviceBuffer, DeviceSlice, DeviceSliceMut, Stream};
use gufo_qwen::vision::VisionInput;

use crate::arena::alloc;
use crate::mtp::head::{
    DEFAULT_PRIME_CHUNK, HEAD_HEADER_BYTES, HeadSnapshot, Input, MAX_CHAIN_TOKENS, MtpHead, Output,
    get_u32, get_u64, put_u32, put_u64,
};
use crate::mtp::sampling::{proposal_sampler, sample_proposal};
use crate::mtp::vocab::{DRAFT_EXTRA_ROWS, DraftTable};
use crate::mtp::{MtpContext, MtpModel};

/// The draft configuration (`QwenMtpGpuDraftConfig`; its vision input is
/// the [`MtpContext`]'s).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpDraftConfig {
    pub max_context: u32,
    pub max_draft_tokens: u32,
    /// Rows per batched prime slice; `0` primes token by token.
    pub prime_chunk: u32,
}

impl Default for MtpDraftConfig {
    fn default() -> Self {
        Self {
            max_context: 4096,
            max_draft_tokens: 4,
            prime_chunk: DEFAULT_PRIME_CHUNK,
        }
    }
}

/// Target hidden rows handed to the draft: host rows (the prompt) or rows
/// still on the device (verified and decoded rows), `[row][hidden]`.
#[derive(Clone, Copy)]
pub enum Rows<'a> {
    Host(&'a [f32]),
    Device(DeviceSlice<'a, f32>),
}

impl Rows<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Host(rows) => rows.len(),
            Self::Device(rows) => rows.len(),
        }
    }

    /// Writes rows `first..first + count` of `width` elements into
    /// `destination`, in the stream's order. Rows of another stream (a draft
    /// on a stream of its own) are copied across and waited for: the caller
    /// has completed the target work that wrote them, as after a
    /// verification whose tokens it read.
    fn store(
        &self,
        stream: &Stream,
        first: usize,
        count: usize,
        width: usize,
        destination: DeviceSliceMut<'_, f32>,
    ) -> Result<(), String> {
        let error = |e| format!("MTP hidden copy failed: {e}");
        let range = first * width..(first + count) * width;
        match self {
            Self::Host(rows) => stream.upload(&rows[range], destination).map_err(error),
            Self::Device(rows) if rows.home_stream().is_some_and(|home| home != stream.id()) => {
                stream
                    .copy_across(rows.range(range), destination)
                    .map_err(error)
            }
            Self::Device(rows) => stream.copy(rows.range(range), destination).map_err(error),
        }
    }
}

/// A draft proposal (`DraftProposal`): the tokens after the anchor and, for
/// a sampled one, each token's candidate distribution, `[token][candidate]`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Proposal {
    pub start_pos: u32,
    pub tokens: Vec<u32>,
    pub candidates_per_token: usize,
    pub candidate_ids: Vec<u32>,
    pub candidate_probabilities: Vec<f32>,
}

/// One session's request in a proposal batch (`DraftProposalRequest`).
pub struct ProposalRequest<'a> {
    pub draft: &'a mut MtpDraft,
    /// The session's image input; the batch's context gives the stream and
    /// the block.
    pub vision: Option<&'a VisionInput>,
    /// The committed sequence through the anchor at `position`.
    pub tokens: &'a [u32],
    pub position: u32,
    pub max_tokens: u32,
    /// Temperature and RNG state of a sampled request.
    pub sampling: Option<(f32, &'a mut u64)>,
}

/// `ctx` with one session's image input.
fn session<'a>(ctx: &MtpContext<'a>, vision: Option<&'a VisionInput>) -> MtpContext<'a> {
    MtpContext {
        stream: ctx.stream,
        model: ctx.model,
        vision,
    }
}

/// The MTP draft of one sequence (`QwenMtpGpuDraftBackend`).
pub struct MtpDraft {
    head: MtpHead,
    config: MtpDraftConfig,
    /// Upper bound on the next proposals' length: the concurrent draft cap
    /// a runner sets before each step (`SetStepDraftCap`).
    step_draft_cap: u32,
    width: usize,
    /// The newest committed target row (`target_hidden_`).
    target_row: DeviceBuffer<f32>,
    /// The row a proposal started from (`proposal_target_hidden_`).
    proposal_row: DeviceBuffer<f32>,
    /// Rows committed during a proposal (`committed_target_hidden_`), at most
    /// a block plus its correction.
    committed: DeviceBuffer<f32>,
    committed_rows: usize,
    proposed: Vec<u32>,
    proposal_input: u32,
    proposal_checkpoint: u32,
    table: Option<DraftTable>,
    primed: bool,
    proposal_active: bool,
    last_error: String,
}

/// Draft state at a committed proposal boundary (`QwenMtpDraftSnapshot`):
/// the cache prefix and the committed target row that seeds the next
/// proposal.
pub struct MtpDraftSnapshot {
    gpu: HeadSnapshot,
    target_hidden: Vec<f32>,
    config: MtpDraftConfig,
    primed: bool,
}

/// `GMTDRF01`, the persistent form of an [`MtpDraftSnapshot`].
const DRAFT_MAGIC: [u8; 8] = *b"GMTDRF01";
const DRAFT_VERSION: u32 = 1;
const DRAFT_HEADER_BYTES: usize = 64;
const PRIMED_FLAG: u32 = 1;

impl MtpDraftSnapshot {
    /// `PayloadBytes`: the cache planes, the target row and the flag.
    pub fn payload_bytes(&self) -> usize {
        self.gpu.payload_bytes() + self.target_hidden.len() * 4 + 4
    }

    /// `PersistentPayloadBytes`.
    pub fn persistent_bytes(&self) -> usize {
        DRAFT_HEADER_BYTES + self.gpu.persistent_bytes() + self.target_hidden.len() * 4
    }

    /// Writes the persistent form into `destination` of exactly
    /// [`MtpDraftSnapshot::persistent_bytes`] (`SerializePersistent`).
    pub fn serialize(&self, stream: &Stream, destination: &mut [u8]) -> Result<usize, String> {
        let expected = self.persistent_bytes();
        if destination.len() != expected
            || self.primed == self.target_hidden.is_empty()
            || (!self.primed && self.gpu.valid_context() != 0)
        {
            return Err("MTP draft persistent destination is invalid".into());
        }
        let gpu_bytes = self.gpu.persistent_bytes();
        destination.fill(0);
        destination[..8].copy_from_slice(&DRAFT_MAGIC);
        put_u32(destination, 8, DRAFT_VERSION);
        put_u32(destination, 12, DRAFT_HEADER_BYTES as u32);
        put_u32(destination, 16, if self.primed { PRIMED_FLAG } else { 0 });
        put_u32(destination, 20, self.target_hidden.len() as u32);
        put_u32(destination, 24, self.gpu.valid_context());
        put_u32(destination, 28, self.config.max_context);
        put_u32(destination, 32, self.config.max_draft_tokens);
        put_u64(destination, 40, gpu_bytes as u64);
        put_u64(destination, 48, expected as u64);
        put_u64(destination, 56, self.target_hidden.len() as u64);
        let hidden_offset = DRAFT_HEADER_BYTES + gpu_bytes;
        let written = self
            .gpu
            .serialize(stream, &mut destination[DRAFT_HEADER_BYTES..hidden_offset])?;
        if written != gpu_bytes {
            return Err("MTP GPU persistent serializer returned the wrong byte count".into());
        }
        for (index, value) in self.target_hidden.iter().enumerate() {
            put_u32(destination, hidden_offset + index * 4, value.to_bits());
        }
        Ok(expected)
    }
}

impl MtpDraft {
    /// `QwenMtpGpuDraftBackend::Create` with the default prime chunk.
    pub fn new(
        stream: &Stream,
        model: &MtpModel,
        max_context: u32,
        max_draft_tokens: u32,
    ) -> Result<Self, String> {
        Self::with_config(
            &MtpContext {
                stream,
                model,
                vision: None,
            },
            MtpDraftConfig {
                max_context,
                max_draft_tokens,
                prime_chunk: DEFAULT_PRIME_CHUNK,
            },
        )
    }

    /// `QwenMtpGpuDraftBackend::Create`.
    pub fn with_config(ctx: &MtpContext<'_>, config: MtpDraftConfig) -> Result<Self, String> {
        if config.max_draft_tokens == 0 {
            return Err("MTP GPU draft configuration is invalid".into());
        }
        let s = ctx.stream;
        let width = ctx.model.config.hidden_size as usize;
        let head = MtpHead::new(ctx, config.max_context, config.prime_chunk)?;
        let mut target_row = alloc::<f32>(s, width)?;
        s.fill_bytes(target_row.as_mut_slice(), 0)
            .map_err(|e| format!("MTP hidden reset failed: {e}"))?;
        Ok(Self {
            head,
            config,
            step_draft_cap: u32::MAX,
            width,
            target_row,
            proposal_row: alloc(s, width)?,
            committed: alloc(s, (config.max_draft_tokens as usize + 1) * width)?,
            committed_rows: 0,
            proposed: Vec::new(),
            proposal_input: 0,
            proposal_checkpoint: 0,
            table: ctx
                .model
                .has_draft_vocab()
                .then(|| DraftTable::new(ctx.model.draft_specials().to_vec())),
            primed: false,
            proposal_active: false,
            last_error: String::new(),
        })
    }

    pub fn config(&self) -> MtpDraftConfig {
        self.config
    }

    pub fn max_draft_tokens(&self) -> u32 {
        self.config.max_draft_tokens
    }

    /// The head, for its cursor and memory.
    pub fn head(&self) -> &MtpHead {
        &self.head
    }

    /// Why the last table upkeep fell back to the full head, or a call
    /// failed (`GetLastError`).
    pub fn last_error(&self) -> &str {
        &self.last_error
    }

    /// Caps the next proposals' length (`SetStepDraftCap`); `u32::MAX`
    /// lifts the cap.
    pub fn set_step_draft_cap(&mut self, cap: u32) {
        self.step_draft_cap = cap;
    }

    pub fn is_primed(&self) -> bool {
        self.primed
    }

    /// `Reset`. The C++ backend zeroes its host row; nothing reads this one
    /// before priming writes it.
    pub fn reset(&mut self) {
        self.head.reset();
        self.proposed.clear();
        self.committed_rows = 0;
        self.proposal_input = 0;
        self.proposal_checkpoint = 0;
        self.primed = false;
        self.proposal_active = false;
        self.last_error.clear();
        if let Some(table) = &mut self.table {
            table.forget();
        }
    }

    /// A reused prefix can continue with other tokens than the ones scanned
    /// (`BeginRequest`).
    pub fn begin_request(&mut self) {
        if let Some(table) = &mut self.table {
            table.forget();
        }
    }

    /// Pairs each prompt token after the first with the target row before
    /// it and keeps the last row for the first proposal
    /// (`PrimeTargetContext`).
    pub fn prime(
        &mut self,
        ctx: &MtpContext<'_>,
        tokens: &[u32],
        hidden: &[f32],
    ) -> Result<(), String> {
        self.reset();
        let width = self.width;
        if tokens.is_empty() || hidden.len() != tokens.len() * width {
            self.last_error = "MTP prompt hidden-state shape is invalid".into();
            return Err(self.last_error.clone());
        }
        if let Err(error) = self.prime_rows(ctx, tokens, hidden) {
            self.reset();
            self.last_error = error.clone();
            return Err(error);
        }
        self.primed = true;
        Ok(())
    }

    fn prime_rows(
        &mut self,
        ctx: &MtpContext<'_>,
        tokens: &[u32],
        hidden: &[f32],
    ) -> Result<(), String> {
        let width = self.width;
        let rows = tokens.len() - 1;
        if self.config.prime_chunk != 0 {
            // Projection-only: the rows the serial loop writes (position `i`
            // takes token `i + 1` and the target row of `i`), without
            // attention, MoE or logits, which the draft never reads.
            self.head
                .prime_batch(ctx, 0, &tokens[1..], &hidden[..rows * width])?;
        } else {
            for index in 0..rows {
                self.head.forward_target_hidden(
                    ctx,
                    tokens[index + 1],
                    &hidden[index * width..(index + 1) * width],
                    index as u32,
                    false,
                    false,
                )?;
            }
        }
        Rows::Host(hidden).store(ctx.stream, rows, 1, width, self.target_row.as_mut_slice())
    }

    /// Appends committed tokens at `position` with their target rows
    /// (`AppendTargetContext`).
    pub fn append(
        &mut self,
        ctx: &MtpContext<'_>,
        tokens: &[u32],
        rows: Rows<'_>,
        position: u32,
    ) -> Result<(), String> {
        let width = self.width;
        if !self.primed
            || self.proposal_active
            || self.head.next_position() + 1 != position
            || rows.len() != tokens.len() * width
        {
            return Err("MTP draft cannot append the target context".into());
        }
        let Some(count) = tokens.len().checked_sub(1) else {
            return Ok(());
        };
        if self.config.prime_chunk != 0 {
            // The draft row at `position - 1 + j` takes token `j` and the
            // target row of `j - 1` (the newest committed row for `j == 0`):
            // what the serial loop below computes, projection-only.
            let target_row = &self.target_row;
            self.head
                .prime_rows(ctx, position - 1, tokens, |s, offset, n, mut staging| {
                    let mut first = 0;
                    if offset == 0 {
                        s.copy(target_row.as_slice(), staging.reborrow().range(0..width))
                            .map_err(|e| format!("MTP hidden copy failed: {e}"))?;
                        first = 1;
                    }
                    if first < n {
                        rows.store(
                            s,
                            offset + first - 1,
                            n - first,
                            width,
                            staging.range(first * width..n * width),
                        )?;
                    }
                    Ok(())
                })?;
            return rows.store(ctx.stream, count, 1, width, self.target_row.as_mut_slice());
        }
        for (row, &token) in tokens.iter().enumerate() {
            self.head.forward_row(
                ctx,
                token,
                self.target_row.as_slice(),
                position + row as u32 - 1,
                false,
                false,
            )?;
            rows.store(ctx.stream, row, 1, width, self.target_row.as_mut_slice())?;
        }
        Ok(())
    }

    /// The checks of a proposal and its length (`Propose`'s preamble).
    fn begin(&self, tokens: &[u32], position: u32, max_tokens: u32) -> Result<u32, String> {
        if !self.primed || tokens.is_empty() || position == 0 {
            return Err("MTP GPU draft backend is not primed".into());
        }
        if self.proposal_active {
            return Err("MTP GPU proposal feedback is pending".into());
        }
        if self.head.next_position() + 1 != position {
            return Err("MTP GPU draft position is inconsistent".into());
        }
        Ok(max_tokens
            .min(self.config.max_draft_tokens)
            .min(self.step_draft_cap))
    }

    /// Records the proposal's anchor and starting row.
    fn start(&mut self, ctx: &MtpContext<'_>, anchor: u32) -> Result<(), String> {
        self.proposal_checkpoint = self.head.next_position();
        self.proposal_input = anchor;
        ctx.stream
            .copy(self.target_row.as_slice(), self.proposal_row.as_mut_slice())
            .map_err(|e| format!("MTP hidden copy failed: {e}"))?;
        self.committed_rows = 0;
        self.proposed.clear();
        Ok(())
    }

    /// Brings the draft table up to `tokens` (the sequence through the
    /// anchor) and returns whether the proposal scores the small head
    /// (`SyncDraftVocab`). Table upkeep never fails a proposal: an
    /// inconsistent table drafts over the full head and rebuilds next time.
    fn sync_draft_vocab(&mut self, ctx: &MtpContext<'_>, tokens: &[u32]) -> Result<bool, String> {
        let Some(table) = &mut self.table else {
            return Ok(false);
        };
        let update = table.sync(tokens);
        if update.restart {
            self.head.reset_draft_rows(ctx);
        }
        // An append the table cannot hold falls back to the full head; a
        // failed copy leaves the device table partial, which the next sync
        // rebuilds.
        let room = DRAFT_EXTRA_ROWS - self.head.draft_rows();
        if update.added.len() > room as usize {
            table.forget();
            self.last_error = "MTP GPU draft table append is invalid".into();
            return Ok(false);
        }
        if let Err(error) = self.head.append_draft_rows(ctx, update.added) {
            table.forget();
            return Err(error);
        }
        if self.head.draft_rows() != table.rows() {
            table.forget();
            self.last_error = "MTP draft table disagrees with its device rows".into();
            return Ok(false);
        }
        Ok(tokens.last().is_some_and(|&anchor| table.fits(anchor)))
    }

    /// Up to `max_tokens` greedy drafts after the last token of `tokens` (the
    /// committed sequence through the anchor) at `position` (`Propose`).
    pub fn propose_greedy(
        &mut self,
        ctx: &MtpContext<'_>,
        tokens: &[u32],
        position: u32,
        max_tokens: u32,
    ) -> Result<Proposal, String> {
        let count = self.begin(tokens, position, max_tokens)?;
        let mut proposal = Proposal {
            start_pos: position,
            ..Proposal::default()
        };
        if count == 0 {
            return Ok(proposal);
        }
        let anchor = tokens[tokens.len() - 1];
        self.start(ctx, anchor)?;
        let small = self.sync_draft_vocab(ctx, tokens)?;
        let checkpoint = self.proposal_checkpoint;
        let mut token = self.head.forward_row(
            ctx,
            anchor,
            self.target_row.as_slice(),
            checkpoint,
            true,
            small,
        )?;
        self.proposed.push(token);
        for index in 1..count {
            token = self
                .head
                .forward_feedback(ctx, token, checkpoint + index, true, small)?;
            self.proposed.push(token);
        }
        proposal.tokens = self.proposed.clone();
        self.proposal_active = true;
        Ok(proposal)
    }

    /// Up to `max_tokens` drafts drawn from the full head's top candidates
    /// at `temperature`, each with the exact distribution it was drawn from,
    /// so the verifier's rejection sampling stays lossless
    /// (`ProposeSampled`).
    pub fn propose_sampled(
        &mut self,
        ctx: &MtpContext<'_>,
        tokens: &[u32],
        position: u32,
        max_tokens: u32,
        temperature: f32,
        rng: &mut u64,
    ) -> Result<Proposal, String> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err("MTP sampled proposal temperature must be finite and positive".into());
        }
        let count = self.begin(tokens, position, max_tokens)?;
        let mut proposal = Proposal {
            start_pos: position,
            ..Proposal::default()
        };
        if count == 0 {
            return Ok(proposal);
        }
        let anchor = tokens[tokens.len() - 1];
        self.start(ctx, anchor)?;
        let checkpoint = self.proposal_checkpoint;
        // q is the temperature softmax over the full head's top candidates.
        // Rows the distribution drops keep their IDs at probability zero, so
        // every row has the same width and distinct IDs.
        let sampler = proposal_sampler(temperature)?;
        let mut width = 0;
        for index in 0..count {
            if index == 0 {
                self.head.forward_row(
                    ctx,
                    anchor,
                    self.target_row.as_slice(),
                    checkpoint,
                    true,
                    false,
                )?;
            } else {
                let last = self.proposed[self.proposed.len() - 1];
                self.head
                    .forward_feedback(ctx, last, checkpoint + index, true, false)?;
            }
            let candidates = self.head.top_candidates(ctx)?;
            if width == 0 {
                width = candidates.size;
                proposal.candidates_per_token = width;
                proposal.candidate_ids.reserve(count as usize * width);
                proposal
                    .candidate_probabilities
                    .reserve(count as usize * width);
            } else if candidates.size != width {
                return Err("MTP candidate width changed within a proposal".into());
            }
            let drawn = sample_proposal(&candidates, &sampler, rng)?;
            let row = proposal.candidate_ids.len();
            proposal
                .candidate_ids
                .extend_from_slice(&drawn.ids[..drawn.size]);
            proposal
                .candidate_probabilities
                .extend_from_slice(&drawn.probabilities[..drawn.size]);
            for &id in &candidates.ids[..width] {
                if proposal.candidate_ids.len() >= row + width {
                    break;
                }
                if !proposal.candidate_ids[row..].contains(&id) {
                    proposal.candidate_ids.push(id);
                    proposal.candidate_probabilities.push(0.0);
                }
            }
            self.proposed.push(drawn.token);
        }
        proposal.tokens = self.proposed.clone();
        self.proposal_active = true;
        Ok(proposal)
    }

    /// A greedy or sampled proposal (`ProposeForRequest`).
    pub fn propose_for(
        &mut self,
        ctx: &MtpContext<'_>,
        tokens: &[u32],
        position: u32,
        max_tokens: u32,
        sampling: Option<(f32, &mut u64)>,
    ) -> Result<Proposal, String> {
        match sampling {
            Some((temperature, rng)) if temperature > 0.0 => {
                self.propose_sampled(ctx, tokens, position, max_tokens, temperature, rng)
            }
            _ => self.propose_greedy(ctx, tokens, position, max_tokens),
        }
    }

    /// Greedy proposals of two to eight sessions on one model and stream
    /// (`ProposeBatch`): validated before any state changes, each session's
    /// head chosen as its serial proposal would, every session's first step
    /// enqueued before the first read-back. Sampled or mixed cohorts, and
    /// other sizes, propose session by session. Each request reads its own
    /// image input; `ctx.vision` is not used.
    pub fn propose_batch(
        ctx: &MtpContext<'_>,
        requests: &mut [ProposalRequest<'_>],
    ) -> Result<Vec<Proposal>, String> {
        let greedy = requests
            .iter()
            .all(|r| r.sampling.as_ref().is_none_or(|(t, _)| *t == 0.0));
        if requests.len() < 2 || requests.len() > MAX_CHAIN_TOKENS || !greedy {
            return requests
                .iter_mut()
                .map(|r| {
                    r.draft.propose_for(
                        &session(ctx, r.vision),
                        r.tokens,
                        r.position,
                        r.max_tokens,
                        r.sampling.as_mut().map(|(t, rng)| (*t, &mut **rng)),
                    )
                })
                .collect();
        }
        // Reproduce the serial proposal's checks before any cache or cursor
        // changes. The requests borrow their sessions exclusively, so none
        // repeats.
        for request in requests.iter() {
            let draft = &request.draft;
            if !draft.primed
                || request.tokens.is_empty()
                || request.position == 0
                || draft.proposal_active
                || draft.head.next_position() + 1 != request.position
            {
                return Err("MTP proposal batch has an invalid request".into());
            }
        }
        let mut counts = Vec::with_capacity(requests.len());
        for request in requests.iter() {
            let draft = &request.draft;
            let count = request
                .max_tokens
                .min(draft.config.max_draft_tokens)
                .min(draft.step_draft_cap);
            if draft.head.next_position() + count > draft.head.max_context() {
                return Err("MTP proposal batch exceeds a session context limit".into());
            }
            counts.push(count);
        }
        // Every head is chosen, as serial proposals would from the same
        // history, before any step runs.
        let mut small = vec![false; requests.len()];
        for (index, request) in requests.iter_mut().enumerate() {
            if counts[index] != 0 {
                small[index] = request
                    .draft
                    .sync_draft_vocab(&session(ctx, request.vision), request.tokens)?;
            }
        }
        let checkpoints: Vec<u32> = requests
            .iter()
            .map(|r| r.draft.head.next_position())
            .collect();
        let result = Self::run_batch(ctx, requests, &counts, &small);
        if let Err(error) = result {
            // A failed batch leaves every session at its committed frontier.
            for (request, &checkpoint) in requests.iter_mut().zip(&checkpoints) {
                let draft = &mut *request.draft;
                if draft.head.next_position() != checkpoint
                    && let Err(rollback) = draft.head.rewind(checkpoint)
                {
                    draft.last_error = rollback;
                }
                draft.committed_rows = 0;
                draft.proposed.clear();
                draft.proposal_active = false;
            }
            return Err(error);
        }
        Ok(requests
            .iter()
            .zip(&counts)
            .map(|(request, &count)| Proposal {
                start_pos: request.position,
                tokens: if count == 0 {
                    Vec::new()
                } else {
                    request.draft.proposed.clone()
                },
                ..Proposal::default()
            })
            .collect())
    }

    /// The steps of a proposal batch: step `k` of every session, then the
    /// tokens of that step, so one read-back covers every session's step.
    fn run_batch(
        ctx: &MtpContext<'_>,
        requests: &mut [ProposalRequest<'_>],
        counts: &[u32],
        small: &[bool],
    ) -> Result<(), String> {
        for request in requests.iter_mut() {
            let anchor = request.tokens[request.tokens.len() - 1];
            request.draft.start(&session(ctx, request.vision), anchor)?;
        }
        let steps = counts.iter().copied().max().unwrap_or(0);
        for step in 0..steps {
            for (index, request) in requests.iter_mut().enumerate() {
                if step >= counts[index] {
                    continue;
                }
                let ctx = &session(ctx, request.vision);
                let draft = &mut *request.draft;
                let position = draft.proposal_checkpoint + step;
                let slot = Output::Chain(step as usize);
                if step == 0 {
                    draft.head.step(
                        ctx,
                        draft.proposal_input,
                        Input::Device(draft.target_row.as_slice()),
                        position,
                        slot,
                        small[index],
                    )?;
                } else {
                    let token = draft.proposed[step as usize - 1];
                    draft
                        .head
                        .step(ctx, token, Input::Feedback, position, slot, small[index])?;
                }
            }
            for (index, request) in requests.iter_mut().enumerate() {
                if step >= counts[index] {
                    continue;
                }
                let draft = &mut *request.draft;
                let token = draft
                    .head
                    .read_token(&session(ctx, request.vision), Output::Chain(step as usize))?;
                draft.proposed.push(token);
            }
        }
        for (request, &count) in requests.iter_mut().zip(counts) {
            request.draft.proposal_active = count != 0;
        }
        Ok(())
    }

    /// The target row paired with the next correction; during a proposal it
    /// also collects the committed rows (`UpdateTargetHidden`).
    pub fn update_hidden(&mut self, ctx: &MtpContext<'_>, row: Rows<'_>) -> Result<(), String> {
        let width = self.width;
        if row.len() != width {
            return Err("MTP target hidden-state shape is invalid".into());
        }
        if self.proposal_active {
            let slot = self.committed_rows;
            if (slot + 1) * width > self.committed.len() {
                return Err("MTP feedback has more committed rows than a block".into());
            }
            row.store(
                ctx.stream,
                0,
                1,
                width,
                self.committed.slice_mut(slot * width..(slot + 1) * width),
            )?;
            self.committed_rows += 1;
        }
        row.store(ctx.stream, 0, 1, width, self.target_row.as_mut_slice())
    }

    /// Replays the proposal's anchor and its `accepted` tokens on the
    /// committed target rows instead of the head's own (`AcceptFeedback`):
    /// input `t + 1` pairs with target row `t`, as in priming.
    pub fn accept(&mut self, ctx: &MtpContext<'_>, accepted: &[u32]) -> Result<(), String> {
        let width = self.width;
        if !self.proposal_active || accepted.len() > self.proposed.len() {
            return Err("MTP GPU proposal feedback is invalid".into());
        }
        if self.committed_rows != accepted.len() + 1 {
            return Err("MTP feedback requires each committed target row".into());
        }
        let checkpoint = self.proposal_checkpoint;
        self.head.rewind(checkpoint)?;
        // Only the cache rows of these steps are read later.
        self.head.replay_row(
            ctx,
            self.proposal_input,
            self.proposal_row.as_slice(),
            checkpoint,
        )?;
        for (index, &token) in accepted.iter().enumerate() {
            self.head.replay_row(
                ctx,
                token,
                self.committed.slice(index * width..(index + 1) * width),
                checkpoint + index as u32 + 1,
            )?;
        }
        self.proposal_active = false;
        self.proposed.clear();
        self.committed_rows = 0;
        Ok(())
    }

    /// Drops an uncommitted proposal so a cancelled verification resumes from
    /// `position` (`DiscardPendingTargetContext`); the cache trails the
    /// target by one row.
    pub fn discard_pending(&mut self, position: u32) -> Result<(), String> {
        if self.proposal_active {
            if self.proposal_checkpoint + 1 != position {
                return Err("MTP cancellation frontier does not match the draft".into());
            }
            self.head.rewind(self.proposal_checkpoint)?;
            self.proposal_active = false;
            self.proposed.clear();
            self.committed_rows = 0;
        }
        if self.head.next_position() + 1 != position {
            return Err("MTP cancelled draft is not at the committed frontier".into());
        }
        Ok(())
    }

    /// Bytes a snapshot holds, an exact upper bound (`SnapshotPayloadBytes`).
    pub fn snapshot_payload_bytes(&self, model: &MtpModel) -> usize {
        self.head.snapshot_payload_bytes(model) + self.width * 4 + 4
    }

    /// The state at a committed boundary (`Snapshot`), copied on `transfer`
    /// and waited for; the caller orders the session's queued work before
    /// it.
    pub fn snapshot(
        &self,
        transfer: &Stream,
        model: &MtpModel,
    ) -> Result<MtpDraftSnapshot, String> {
        if self.proposal_active {
            return Err("MTP snapshot requires a committed proposal boundary".into());
        }
        let mut bytes = vec![0u8; self.width * 4];
        transfer
            .download_bytes_across(self.target_row.as_slice(), &mut bytes)
            .map_err(|e| format!("MTP hidden synchronization failed: {e}"))?;
        let target_hidden = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        Ok(MtpDraftSnapshot {
            gpu: self.head.save_snapshot(transfer, model)?,
            target_hidden,
            config: self.config,
            primed: self.primed,
        })
    }

    /// Returns to a snapshot; the table rebuilds from the next proposal's
    /// sequence (`RestoreSnapshot`).
    pub fn restore(
        &mut self,
        ctx: &MtpContext<'_>,
        snapshot: &MtpDraftSnapshot,
    ) -> Result<(), String> {
        if snapshot.config.max_context != self.config.max_context
            || snapshot.config.max_draft_tokens != self.config.max_draft_tokens
            || snapshot.target_hidden.len() != self.width
            || snapshot.primed == snapshot.target_hidden.is_empty()
        {
            return Err("MTP draft snapshot has a malformed target hidden row".into());
        }
        self.head.restore_snapshot(ctx, &snapshot.gpu)?;
        ctx.stream
            .upload(&snapshot.target_hidden, self.target_row.as_mut_slice())
            .map_err(|e| format!("MTP hidden copy failed: {e}"))?;
        self.after_restore(snapshot.primed);
        Ok(())
    }

    /// Restores a [`MtpDraftSnapshot::serialize`] payload
    /// (`RestorePersistentSnapshot`).
    pub fn restore_persistent(
        &mut self,
        ctx: &MtpContext<'_>,
        payload: &[u8],
    ) -> Result<(), String> {
        let invalid = || "MTP draft persistent header is invalid".to_string();
        if payload.len() < DRAFT_HEADER_BYTES
            || payload[..8] != DRAFT_MAGIC
            || get_u32(payload, 8) != Some(DRAFT_VERSION)
            || get_u32(payload, 12) != Some(DRAFT_HEADER_BYTES as u32)
            || get_u32(payload, 36) != Some(0)
        {
            return Err(invalid());
        }
        let field = |offset| get_u32(payload, offset).ok_or_else(invalid);
        let wide = |offset| {
            get_u64(payload, offset)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(invalid)
        };
        let flags = field(16)?;
        let hidden_size = field(20)? as usize;
        let valid_context = field(24)?;
        let (max_context, max_draft_tokens) = (field(28)?, field(32)?);
        let (gpu_bytes, total, hidden_count) = (wide(40)?, wide(48)?, wide(56)?);
        let primed = flags & PRIMED_FLAG != 0;
        if flags & !PRIMED_FLAG != 0
            || hidden_size != self.width
            || max_context != self.config.max_context
            || max_draft_tokens != self.config.max_draft_tokens
            || valid_context > self.config.max_context
            || hidden_count != if primed { self.width } else { 0 }
            || (!primed && valid_context != 0)
            || total != payload.len()
        {
            return Err("MTP draft persistent metadata is incompatible".into());
        }
        let hidden_offset = DRAFT_HEADER_BYTES
            .checked_add(gpu_bytes)
            .filter(|&end| end <= payload.len())
            .ok_or("MTP draft persistent payload size is invalid")?;
        if hidden_offset + hidden_count * 4 != payload.len() || gpu_bytes < HEAD_HEADER_BYTES {
            return Err("MTP draft persistent payload size is invalid".into());
        }
        let target_hidden: Vec<f32> = payload[hidden_offset..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_bits(u32::from_le_bytes(b)))
            .collect();
        self.head
            .restore_persistent(ctx, &payload[DRAFT_HEADER_BYTES..hidden_offset])?;
        if self.head.next_position() != valid_context {
            return Err("MTP draft persistent context length is inconsistent".into());
        }
        let error = |e| format!("MTP hidden copy failed: {e}");
        if primed {
            ctx.stream
                .upload(&target_hidden, self.target_row.as_mut_slice())
                .map_err(error)?;
        } else {
            ctx.stream
                .fill_bytes(self.target_row.as_mut_slice(), 0)
                .map_err(error)?;
        }
        self.after_restore(primed);
        Ok(())
    }

    fn after_restore(&mut self, primed: bool) {
        if let Some(table) = &mut self.table {
            table.forget();
        }
        self.proposed.clear();
        self.committed_rows = 0;
        self.proposal_input = 0;
        self.proposal_checkpoint = 0;
        self.primed = primed;
        self.proposal_active = false;
    }
}
