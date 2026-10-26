//! The draft side of one serving session, a port of the bookkeeping of
//! `QwenDFlashGpuDraftBackend` (`hip/dflash/backend.cpp`): the history ring,
//! the length controller, the prompt-history lookup, committed target
//! features waiting for the next proposal, greedy and sampled proposals of
//! one session or a batch of them, snapshots and the persistent `GDFDRF01`
//! payload. The draft model and the pass scratch are shared
//! ([`DraftContext`]).

use gufo_core::prompt_lookup::LookupState;
use gufo_core::sampling;
use gufo_hip::{GgmlType, Stream};

use crate::dflash::executor::{
    self, BlockRequest, DFlashSnapshot, DraftPenalties, DraftScratch, DraftSession, Proposal,
    Selection,
};
use crate::dflash::lookup::{DraftLookup, LookupLimits};
use crate::dflash::model::DFlashModel;
use crate::dflash::policy::{DraftPolicy, LengthController, MAX_DRAFT_TOKENS};
use crate::persistent::{DraftGpuLayout, DraftHeader, DraftLimits};

/// What every draft call uses: the stream, the draft model and the scratch.
pub struct DraftContext<'a> {
    pub stream: &'a Stream,
    pub model: &'a DFlashModel,
    pub scratch: &'a mut DraftScratch,
}

/// The draft configuration of a runner (`QwenDFlashGpuDraftConfig`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftConfig {
    pub max_context: u32,
    pub max_draft_tokens: u32,
    pub policy: DraftPolicy,
}

impl DraftConfig {
    /// The limit the backend applies: the block size and seven drafts
    /// (`QwenDFlashGpuDraftBackend::Create`).
    pub fn for_model(
        model: &DFlashModel,
        max_context: u32,
        max_draft_tokens: u32,
        policy: DraftPolicy,
    ) -> Result<Self, String> {
        if max_draft_tokens == 0 || model.config.block_size < 2 {
            return Err("DFlash GPU draft configuration is invalid".into());
        }
        Ok(Self {
            max_context,
            max_draft_tokens: max_draft_tokens
                .min(model.config.block_size - 1)
                .min(MAX_DRAFT_TOKENS),
            policy,
        })
    }
}

/// One session's draft backend.
pub struct Draft {
    session: DraftSession,
    config: DraftConfig,
    controller: LengthController,
    lookup: DraftLookup,
    width: usize,
    /// Committed feature rows not yet injected, `[row][taps * hidden]`.
    pending: Vec<f32>,
    proposed: Vec<u32>,
    primed: bool,
    proposal_active: bool,
}

/// Draft state at a committed boundary (`QwenDFlashDraftSnapshot`).
pub struct DraftSnapshot {
    gpu: DFlashSnapshot,
    pending: Vec<f32>,
    controller: (f32, u32),
    lookup: LookupState,
    primed: bool,
    config: DraftConfig,
}

impl DraftSnapshot {
    /// `PayloadBytes`: the device rings, the pending features, the
    /// controller and the lookup state.
    pub fn payload_bytes(&self) -> usize {
        self.gpu.bytes() + self.pending.len() * 4 + 4 + 4 + LookupState::BYTES
    }
}

/// One session's request in a proposal batch (`DraftProposalRequest`).
pub struct ProposalRequest<'a> {
    pub draft: &'a mut Draft,
    /// The sequence through the anchor, its last token, at `position`. Only
    /// a complete history (`position + 1` tokens) can be copied from.
    pub tokens: &'a [u32],
    pub position: u32,
    pub max_tokens: u32,
    /// Temperature and RNG state of a sampled request.
    pub sampling: Option<(f32, &'a mut u64)>,
    /// The request's numeric penalties, which guide q.
    pub penalties: Option<&'a DraftPenalties>,
}

impl Draft {
    pub fn new(ctx: &DraftContext<'_>, config: DraftConfig) -> Result<Self, String> {
        let session = DraftSession::new(ctx.stream, ctx.model, config.max_context)?;
        // The tied output tensor distinguishes the Q8 target from Q4 for the
        // verification cost.
        let q8_target = ctx.model.output.format() == GgmlType::Q8_0;
        Ok(Self {
            session,
            config,
            controller: LengthController::new(config.policy, config.max_draft_tokens, q8_target),
            lookup: DraftLookup::default(),
            width: ctx.model.config.target_features(),
            pending: Vec::new(),
            proposed: Vec::new(),
            primed: false,
            proposal_active: false,
        })
    }

    pub fn config(&self) -> DraftConfig {
        self.config
    }

    /// Committed positions whose features the ring holds.
    pub fn injected(&self) -> u32 {
        self.session.injected()
    }

    /// `Reset`: a fresh draft.
    pub fn reset(&mut self) {
        self.session.reset();
        self.controller.reset();
        self.lookup.clear();
        self.pending.clear();
        self.proposed.clear();
        self.primed = false;
        self.proposal_active = false;
    }

    /// `BeginRequest`: the controller starts over, and a copy cannot resume
    /// the previous answer's.
    pub fn begin_request(&mut self) {
        self.controller.reset();
        self.lookup.begin_request();
    }

    /// Starts over a prefilled prompt of `rows` feature rows
    /// (`PrimeTargetContext`).
    pub fn prime(
        &mut self,
        ctx: &mut DraftContext<'_>,
        features: &[f32],
        rows: usize,
    ) -> Result<(), String> {
        self.reset();
        if rows == 0 || features.len() != rows * self.width {
            return Err("draft backend failed to prime target context".into());
        }
        let result = executor::inject(
            ctx.stream,
            ctx.model,
            ctx.scratch,
            &mut self.session,
            features,
            0,
            rows as u32,
        );
        if result.is_err() {
            self.reset();
        }
        result?;
        self.primed = true;
        Ok(())
    }

    /// Appends target rows at `position` after the pending ones
    /// (`AppendTargetContext`).
    pub fn append(
        &mut self,
        ctx: &mut DraftContext<'_>,
        features: &[f32],
        rows: usize,
        position: u32,
    ) -> Result<(), String> {
        if !self.primed || self.proposal_active || features.len() != rows * self.width {
            return Err("draft failed to append committed target token".into());
        }
        self.inject_pending(ctx, position)?;
        executor::inject(
            ctx.stream,
            ctx.model,
            ctx.scratch,
            &mut self.session,
            features,
            position,
            rows as u32,
        )
    }

    /// Queues the features of one committed row (`UpdateTargetHidden`).
    pub fn update_hidden(&mut self, row: &[f32]) -> Result<(), String> {
        if row.len() != self.width {
            return Err("DFlash committed target feature width is invalid".into());
        }
        self.pending.extend_from_slice(row);
        Ok(())
    }

    /// Rows pending before `position` (`PendingFeatureCount`).
    fn pending_rows(&self, position: u32) -> Result<u32, String> {
        let injected = self.session.injected();
        if position < injected {
            return Err("DFlash committed position precedes injected history".into());
        }
        if position > injected {
            let count = position - injected;
            if self.pending.len() != count as usize * self.width {
                return Err("DFlash committed target feature history is incomplete".into());
            }
            return Ok(count);
        }
        if !self.pending.is_empty() {
            return Err("DFlash has target features without a matching committed position".into());
        }
        Ok(0)
    }

    fn inject_pending(&mut self, ctx: &mut DraftContext<'_>, position: u32) -> Result<(), String> {
        let count = self.pending_rows(position)?;
        if count > 0 {
            // The features stay pending until the injection succeeds.
            let injected = self.session.injected();
            executor::inject(
                ctx.stream,
                ctx.model,
                ctx.scratch,
                &mut self.session,
                &self.pending,
                injected,
                count,
            )
            .map_err(|_| "DFlash committed target feature injection failed".to_string())?;
            self.pending.clear();
        }
        Ok(())
    }

    /// The lookup's view of this draft.
    fn lookup_limits<'a>(&self, model: &'a DFlashModel) -> LookupLimits<'a> {
        LookupLimits {
            neural_mean: self.controller.state().0,
            max_draft_tokens: self.config.max_draft_tokens,
            vocab_size: model.config.vocab_size,
            boundaries: &model.message_boundaries,
        }
    }

    /// A prompt-history copy, which then is the active proposal.
    fn try_lookup(
        &mut self,
        model: &DFlashModel,
        tokens: &[u32],
        position: u32,
        limit: u32,
        sampled: bool,
    ) -> Option<Proposal> {
        let limits = self.lookup_limits(model);
        let proposal = self
            .lookup
            .propose(tokens, position, limit, sampled, &limits)?;
        self.proposed = proposal.tokens.clone();
        self.proposal_active = true;
        Some(proposal)
    }

    /// Proposals after the last token of `tokens` at `position`, at most
    /// `max_tokens`, drawn at a temperature when `sampling` is given, under
    /// the numeric `penalties` of the request (`Propose`, `ProposeSampled`,
    /// `ProposeForRequest`). A complete history (`position + 1` tokens) may
    /// be copied from instead.
    pub fn propose(
        &mut self,
        ctx: &mut DraftContext<'_>,
        tokens: &[u32],
        position: u32,
        max_tokens: u32,
        sampling: Option<(f32, &mut u64)>,
        penalties: Option<&DraftPenalties>,
    ) -> Result<Proposal, String> {
        let Some(&anchor) = tokens.last() else {
            return Err("DFlash GPU draft backend is not primed".into());
        };
        if !self.primed || position == 0 {
            return Err("DFlash GPU draft backend is not primed".into());
        }
        if self.proposal_active {
            return Err("DFlash GPU proposal feedback is pending".into());
        }
        if let Some((temperature, _)) = &sampling
            && (!temperature.is_finite() || *temperature <= 0.0)
        {
            return Err("DFlash sampled proposal temperature must be finite and positive".into());
        }
        self.inject_pending(ctx, position)?;
        let budget = self.config.max_context.saturating_sub(position + 1);
        if let Some(copy) = self.try_lookup(
            ctx.model,
            tokens,
            position,
            max_tokens.min(budget),
            sampling.is_some(),
        ) {
            return Ok(copy);
        }
        let count = self.controller.choose(max_tokens.min(budget), position, 1);
        if count == 0 {
            return Ok(Proposal::default());
        }
        self.proposed.clear();
        let (temperature, uniforms) = match sampling {
            Some((temperature, rng)) => (
                temperature,
                (0..count).map(|_| sampling::uniform(rng) as f32).collect(),
            ),
            None => (0.0, Vec::new()),
        };
        let selection = if temperature > 0.0 {
            Selection::Sampled {
                temperature,
                uniforms: &uniforms,
            }
        } else {
            Selection::Greedy
        };
        let proposal = executor::forward_block(
            ctx.stream,
            ctx.model,
            ctx.scratch,
            &self.session,
            anchor,
            position,
            count,
            selection,
            penalties,
        )?;
        self.proposed = proposal.tokens.clone();
        self.proposal_active = !proposal.tokens.is_empty();
        Ok(proposal)
    }

    /// Proposals of several sessions (`ProposeBatch`): committed features
    /// are injected together, lengths are chosen per request (all greedy
    /// cohorts use their shared cost, Q8 cohorts one shared width), a
    /// request that copies its history skips the pass, the draws of the
    /// others are taken in request order, and their blocks run in one pass.
    pub fn propose_batch(
        ctx: &mut DraftContext<'_>,
        requests: &mut [ProposalRequest<'_>],
    ) -> Result<Vec<Proposal>, String> {
        if requests.len() < 2 || requests.len() > 8 {
            return requests
                .iter_mut()
                .map(|request| {
                    request.draft.propose(
                        ctx,
                        request.tokens,
                        request.position,
                        request.max_tokens,
                        request.sampling.as_mut().map(|(t, rng)| (*t, &mut **rng)),
                        request.penalties,
                    )
                })
                .collect();
        }
        // Validate every request before changing caches or drawing values.
        for request in requests.iter() {
            let temperature = request.sampling.as_ref().map_or(0.0, |(t, _)| *t);
            if !request.draft.primed
                || request.tokens.is_empty()
                || request.position == 0
                || request.draft.proposal_active
                || !temperature.is_finite()
                || temperature < 0.0
            {
                return Err("DFlash2 proposal batch has an invalid request".into());
            }
        }
        // The features stay pending until the injection succeeds.
        let mut batch: Vec<(&mut DraftSession, &[f32])> = Vec::new();
        for request in requests.iter_mut() {
            if request.draft.pending_rows(request.position)? > 0 {
                let draft = &mut *request.draft;
                batch.push((&mut draft.session, draft.pending.as_slice()));
            }
        }
        if !batch.is_empty() {
            executor::inject_batch(ctx.stream, ctx.model, ctx.scratch, &mut batch)?;
            drop(batch);
            for request in requests.iter_mut() {
                request.draft.pending.clear();
            }
        }
        let greedy = requests.iter().all(|r| r.sampling.is_none());
        let greedy_batch_size = if greedy { requests.len() } else { 1 };
        let budgets: Vec<u32> = requests
            .iter()
            .map(|r| {
                r.max_tokens
                    .min(r.draft.config.max_context.saturating_sub(r.position + 1))
            })
            .collect();
        let positions: Vec<u32> = requests.iter().map(|r| r.position).collect();
        let shared = if greedy_batch_size > 1 {
            let controllers: Vec<&LengthController> =
                requests.iter().map(|r| &r.draft.controller).collect();
            LengthController::choose_greedy_batch(&controllers, &budgets, &positions)
        } else {
            None
        };
        let mut proposals = vec![Proposal::default(); requests.len()];
        let mut uniforms: Vec<Vec<f32>> = vec![Vec::new(); requests.len()];
        let mut counts = vec![0u32; requests.len()];
        for (index, request) in requests.iter_mut().enumerate() {
            let count = match shared {
                Some(count) => count,
                None => request.draft.controller.choose(
                    budgets[index],
                    request.position,
                    greedy_batch_size,
                ),
            };
            if count == 0 {
                continue;
            }
            if let Some(copy) = request.draft.try_lookup(
                ctx.model,
                request.tokens,
                request.position,
                budgets[index],
                request.sampling.as_ref().is_some_and(|(t, _)| *t > 0.0),
            ) {
                proposals[index] = copy;
                continue;
            }
            counts[index] = count;
            request.draft.proposed.clear();
            if let Some((_, rng)) = request.sampling.as_mut() {
                uniforms[index] = (0..count).map(|_| sampling::uniform(rng) as f32).collect();
            }
        }
        let indices: Vec<usize> = (0..requests.len()).filter(|&i| counts[i] > 0).collect();
        if indices.is_empty() {
            return Ok(proposals);
        }
        let mut blocks: Vec<BlockRequest<'_>> = Vec::with_capacity(indices.len());
        let mut remaining: Vec<Option<&mut ProposalRequest<'_>>> =
            requests.iter_mut().map(Some).collect();
        for &index in &indices {
            let Some(request) = remaining[index].take() else {
                continue;
            };
            let selection = match request.sampling.as_ref() {
                Some((temperature, _)) => Selection::Sampled {
                    temperature: *temperature,
                    uniforms: &uniforms[index],
                },
                None => Selection::Greedy,
            };
            let anchor = *request
                .tokens
                .last()
                .ok_or("DFlash2 proposal batch has an invalid request")?;
            blocks.push(BlockRequest {
                session: &mut request.draft.session,
                anchor,
                position: request.position,
                count: counts[index],
                selection,
                penalties: request.penalties,
            });
        }
        let generated =
            executor::forward_block_batch(ctx.stream, ctx.model, ctx.scratch, &mut blocks)?;
        drop(blocks);
        drop(remaining);
        for (&index, proposal) in indices.iter().zip(generated) {
            let draft = &mut requests[index].draft;
            draft.proposed = proposal.tokens.clone();
            draft.proposal_active = !proposal.tokens.is_empty();
            proposals[index] = proposal;
        }
        Ok(proposals)
    }

    /// Feedback on the last proposal: its first `accepted` tokens were kept
    /// (`AcceptFeedback`).
    pub fn accept(&mut self, accepted: usize) -> Result<(), String> {
        if !self.proposal_active || accepted > self.proposed.len() {
            return Err("DFlash GPU proposal feedback is invalid".into());
        }
        // A copy's acceptance belongs to the lookup, not the neural estimate.
        if !self.lookup.observe(self.proposed.len(), accepted) {
            self.controller.observe(accepted, self.proposed.len());
        }
        self.proposal_active = false;
        self.proposed.clear();
        Ok(())
    }

    /// Drops the committed features of a cancelled verification
    /// (`DiscardPendingTargetContext`).
    pub fn discard_pending(&mut self, position: u32) -> Result<(), String> {
        if self.proposal_active || self.session.injected() != position {
            return Err("DFlash cancellation frontier is not retained".into());
        }
        self.pending.clear();
        Ok(())
    }

    /// Bytes a snapshot of this state takes (`SnapshotPayloadBytes`).
    pub fn snapshot_payload_bytes(&self, model: &DFlashModel) -> usize {
        let kv = model.config.kv_size();
        let slots = self.session.injected().min(self.session.history_capacity()) as usize * kv;
        2 * model.config.num_layers as usize * slots * 4
            + 4
            + 4
            + LookupState::BYTES
            + self.pending.len() * 4
    }

    /// Captures the draft on `transfer` (`Snapshot`).
    pub fn snapshot(
        &self,
        transfer: &Stream,
        model: &DFlashModel,
    ) -> Result<DraftSnapshot, String> {
        if self.proposal_active {
            return Err("DFlash snapshot requires a committed proposal boundary".into());
        }
        Ok(DraftSnapshot {
            gpu: self.session.capture(transfer, model)?,
            pending: self.pending.clone(),
            controller: self.controller.state(),
            lookup: self.lookup.save(),
            primed: self.primed,
            config: self.config,
        })
    }

    /// Returns to `snapshot` (`RestoreSnapshot`).
    pub fn restore(
        &mut self,
        ctx: &DraftContext<'_>,
        snapshot: &DraftSnapshot,
    ) -> Result<(), String> {
        if snapshot.config != self.config || !snapshot.pending.len().is_multiple_of(self.width) {
            return Err("DFlash draft snapshot has malformed pending target features".into());
        }
        let mut controller = self.controller.clone();
        controller.restore(snapshot.controller)?;
        let committed = snapshot.gpu.valid_context() as usize + snapshot.pending.len() / self.width;
        if !snapshot.lookup.valid(committed) {
            return Err("DFlash lookup snapshot has invalid positions".into());
        }
        self.session.restore(ctx.stream, ctx.model, &snapshot.gpu)?;
        self.controller = controller;
        self.lookup.restore(snapshot.lookup);
        self.pending = snapshot.pending.clone();
        self.proposed.clear();
        self.primed = snapshot.primed;
        self.proposal_active = false;
        Ok(())
    }

    fn gpu_layout(
        &self,
        model: &DFlashModel,
        valid_context: u32,
    ) -> Result<DraftGpuLayout, String> {
        DraftGpuLayout::new(
            model.config.num_layers,
            model.config.kv_size() as u32,
            self.config.max_context,
            valid_context,
            u64::from(self.session.history_capacity()),
        )
    }

    fn limits(&self) -> DraftLimits {
        DraftLimits {
            target_feature_width: self.width as u32,
            max_context: self.config.max_context,
            policy: self.config.policy as u32,
            max_draft_tokens: self.config.max_draft_tokens,
        }
    }
}

impl DraftSnapshot {
    /// The `GDFDRF01` header of this snapshot and its `GDFGPU01` layout.
    fn layouts(
        &self,
        model: &DFlashModel,
        history_capacity: u32,
    ) -> Result<(DraftHeader, DraftGpuLayout), String> {
        let gpu = DraftGpuLayout::new(
            model.config.num_layers,
            model.config.kv_size() as u32,
            self.config.max_context,
            self.gpu.valid_context(),
            u64::from(history_capacity),
        )?;
        let header = DraftHeader {
            primed: self.primed,
            target_feature_width: model.config.target_features() as u32,
            pending_features: self.pending.len(),
            gpu_bytes: gpu.total_bytes(),
            injected_context: self.gpu.valid_context(),
            max_context: self.config.max_context,
            controller_state: self.controller.0,
            policy: self.config.policy as u32,
            max_draft_tokens: self.config.max_draft_tokens,
            last_full_width: self.controller.1,
            lookup: self.lookup,
        };
        Ok((header, gpu))
    }

    /// `PersistentPayloadBytes`.
    pub fn persistent_bytes(
        &self,
        model: &DFlashModel,
        history_capacity: u32,
    ) -> Result<usize, String> {
        let (header, _) = self.layouts(model, history_capacity)?;
        header.total_bytes()
    }

    /// `SerializePersistent` into exactly sized `destination`, reading the
    /// frozen rings on `transfer`.
    pub fn serialize(
        &self,
        transfer: &Stream,
        model: &DFlashModel,
        history_capacity: u32,
        destination: &mut [u8],
    ) -> Result<(), String> {
        let (header, gpu) = self.layouts(model, history_capacity)?;
        header.write(destination)?;
        let base = header.gpu().start;
        let range = base..base + gpu.total_bytes();
        let gpu_bytes = &mut destination[range];
        gpu.write(gpu_bytes)?;
        let error = |e| format!("DFlash GPU persistent: {e}");
        transfer
            .download_bytes(self.gpu.k().view(), &mut gpu_bytes[gpu.k()])
            .map_err(error)?;
        transfer
            .download_bytes(self.gpu.v().view(), &mut gpu_bytes[gpu.v()])
            .map_err(error)?;
        for (chunk, value) in destination[header.pending()]
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(&self.pending)
        {
            *chunk = value.to_le_bytes();
        }
        Ok(())
    }
}

impl Draft {
    /// `RestorePersistentSnapshot` of a `GDFDRF01` payload.
    pub fn restore_persistent(
        &mut self,
        ctx: &DraftContext<'_>,
        payload: &[u8],
    ) -> Result<(), String> {
        let header = DraftHeader::parse(payload, self.limits())?;
        let mut controller = self.controller.clone();
        controller.restore((header.controller_state, header.last_full_width))?;
        let gpu_bytes = &payload[header.gpu()];
        let expected = self.gpu_layout(ctx.model, 0)?;
        let gpu = DraftGpuLayout::parse(gpu_bytes, &expected)?;
        let pending: Vec<f32> = payload[header.pending()]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_bits(u32::from_le_bytes(b)))
            .collect();
        self.session.restore_bytes(
            ctx.stream,
            ctx.model,
            &gpu_bytes[gpu.k()],
            &gpu_bytes[gpu.v()],
            gpu.valid_context,
        )?;
        if self.session.injected() != header.injected_context {
            return Err("DFlash draft persistent context length is inconsistent".into());
        }
        self.pending = pending;
        self.controller = controller;
        self.lookup.restore(header.lookup);
        self.proposed.clear();
        self.primed = header.primed;
        self.proposal_active = false;
        Ok(())
    }

    /// The history ring length, for the persistent layout.
    pub fn history_capacity(&self) -> u32 {
        self.session.history_capacity()
    }
}
