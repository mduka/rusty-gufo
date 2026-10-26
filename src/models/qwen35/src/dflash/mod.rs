//! DFlash2 speculative decoding: the draft model, its length policy, the
//! prompt-history lookup and the draft-side bookkeeping of
//! `QwenDFlashGpuDraftBackend` (`hip/dflash/backend.cpp`) for one greedy
//! session.
//!
//! Target features of committed rows arrive one row at a time
//! ([`DFlash::update_hidden`]) and wait on the host until the next proposal
//! injects them in one call, as in C++.

pub mod draft;
pub mod executor;
pub mod lookup;
pub mod model;
pub mod policy;

use std::path::Path;

use gufo_core::prompt_lookup::LookupState;
use gufo_hip::GgmlType;

use crate::weights::Weights;
use executor::{DFlashExecutor, DFlashSnapshot};
use lookup::{DraftLookup, LookupLimits};
use model::DFlashModel;
use policy::{DraftPolicy, LengthController, MAX_DRAFT_TOKENS};

pub struct DFlash {
    model: DFlashModel,
    executor: DFlashExecutor,
    controller: LengthController,
    lookup: DraftLookup,
    max_context: u32,
    max_draft_tokens: u32,
    policy: DraftPolicy,
    /// Committed feature rows not yet injected, `[row][taps * hidden]`.
    pending: Vec<f32>,
    proposed: Vec<u32>,
    primed: bool,
    proposal_active: bool,
}

/// Draft state at a committed boundary (`Snapshot` of the draft backend).
pub struct DFlashState {
    executor: DFlashSnapshot,
    pending: Vec<f32>,
    controller: (f32, u32),
    lookup: LookupState,
    primed: bool,
    config: (u32, u32, DraftPolicy),
}

impl DFlash {
    /// Loads the draft for `target` (`QwenDFlashGpuDraftBackend::CreateFromGguf`).
    pub fn load(
        path: &Path,
        target: &Weights,
        max_context: u32,
        max_draft_tokens: u32,
        policy: DraftPolicy,
    ) -> Result<Self, String> {
        let stream = gufo_hip::Stream::new().map_err(|e| format!("DFlash2 stream: {e}"))?;
        let model = DFlashModel::load(&stream, path, target)?;
        drop(stream);
        if max_draft_tokens == 0 || model.config.block_size < 2 {
            return Err("DFlash2 needs a positive draft length and a block of two".into());
        }
        let max_draft_tokens = max_draft_tokens
            .min(model.config.block_size - 1)
            .min(MAX_DRAFT_TOKENS);
        // The Q8 cost model applies to a Q8_0 LM head.
        let q8_target = model.output.format() == GgmlType::Q8_0;
        let executor = DFlashExecutor::new(&model, max_context)?;
        Ok(Self {
            controller: LengthController::new(policy, max_draft_tokens, q8_target),
            lookup: DraftLookup::default(),
            model,
            executor,
            max_context,
            max_draft_tokens,
            policy,
            pending: Vec::new(),
            proposed: Vec::new(),
            primed: false,
            proposal_active: false,
        })
    }

    pub fn model(&self) -> &DFlashModel {
        &self.model
    }

    /// Target layers whose outputs the draft reads.
    pub fn target_layers(&self) -> &[u32] {
        &self.model.config.target_layer_ids
    }

    pub fn max_draft_tokens(&self) -> u32 {
        self.max_draft_tokens
    }

    fn width(&self) -> usize {
        self.model.config.target_features()
    }

    pub fn reset(&mut self) {
        self.executor.reset();
        self.controller.reset();
        self.lookup.clear();
        self.pending.clear();
        self.proposed.clear();
        self.primed = false;
        self.proposal_active = false;
    }

    /// Starts the draft over a prefilled prompt (`PrimeTargetContext`).
    pub fn prime(&mut self, features: &[f32], rows: usize) -> Result<(), String> {
        self.reset();
        if features.len() != rows * self.width() {
            return Err("DFlash2 prompt features have the wrong shape".into());
        }
        let result = self.executor.inject(&self.model, features, 0, rows as u32);
        if result.is_err() {
            self.reset();
        }
        result?;
        self.primed = true;
        Ok(())
    }

    /// Appends target rows at `position` after the pending ones
    /// (`AppendTargetContext`).
    pub fn append(&mut self, features: &[f32], rows: usize, position: u32) -> Result<(), String> {
        if !self.primed || self.proposal_active || features.len() != rows * self.width() {
            return Err("DFlash2 cannot append target context now".into());
        }
        self.inject_pending(position)?;
        self.executor
            .inject(&self.model, features, position, rows as u32)
    }

    /// Queues the features of one committed row (`UpdateTargetHidden`).
    pub fn update_hidden(&mut self, row: &[f32]) -> Result<(), String> {
        if row.len() != self.width() {
            return Err("DFlash2 feature row has the wrong width".into());
        }
        self.pending.extend_from_slice(row);
        Ok(())
    }

    /// Rows pending before `position` (`PendingFeatureCount`).
    fn pending_rows(&self, position: u32) -> Result<u32, String> {
        let injected = self.executor.injected();
        if position < injected {
            return Err("DFlash2 position precedes the injected context".into());
        }
        let count = position - injected;
        if self.pending.len() != count as usize * self.width() {
            return Err("DFlash2 pending features do not match the position".into());
        }
        Ok(count)
    }

    fn inject_pending(&mut self, position: u32) -> Result<(), String> {
        let count = self.pending_rows(position)?;
        if count > 0 {
            let pending = std::mem::take(&mut self.pending);
            let injected = self.executor.injected();
            self.executor
                .inject(&self.model, &pending, injected, count)?;
        }
        Ok(())
    }

    /// Greedy proposals after the last token of `tokens` (the anchor) at
    /// `position`, at most `max_tokens` (`Propose`). A complete sequence
    /// (`position + 1` tokens) may be copied from instead.
    pub fn propose(
        &mut self,
        tokens: &[u32],
        position: u32,
        max_tokens: u32,
    ) -> Result<Vec<u32>, String> {
        let Some(&anchor) = tokens.last() else {
            return Err("DFlash2 cannot propose now".into());
        };
        if !self.primed || position == 0 || self.proposal_active {
            return Err("DFlash2 cannot propose now".into());
        }
        self.inject_pending(position)?;
        let budget = if position < self.max_context {
            self.max_context - position - 1
        } else {
            0
        };
        let limits = LookupLimits {
            neural_mean: self.controller.state().0,
            max_draft_tokens: self.max_draft_tokens,
            vocab_size: self.model.config.vocab_size,
            boundaries: &self.model.message_boundaries,
        };
        if let Some(copy) =
            self.lookup
                .propose(tokens, position, max_tokens.min(budget), false, &limits)
        {
            self.proposed = copy.tokens.clone();
            self.proposal_active = true;
            return Ok(copy.tokens);
        }
        let count = self.controller.choose(max_tokens.min(budget), position, 1);
        if count == 0 {
            return Ok(Vec::new());
        }
        let tokens = self
            .executor
            .forward_block(&self.model, anchor, position, count)?;
        self.proposed = tokens.clone();
        self.proposal_active = !tokens.is_empty();
        Ok(tokens)
    }

    /// Feedback on the last proposal: its first `accepted` tokens were kept
    /// (`AcceptFeedback`).
    pub fn accept(&mut self, accepted: usize) -> Result<(), String> {
        if !self.proposal_active || accepted > self.proposed.len() {
            return Err("DFlash2 feedback without a matching proposal".into());
        }
        if !self.lookup.observe(self.proposed.len(), accepted) {
            self.controller.observe(accepted, self.proposed.len());
        }
        self.proposal_active = false;
        self.proposed.clear();
        Ok(())
    }

    pub fn snapshot(&self) -> Result<DFlashState, String> {
        if self.proposal_active {
            return Err("DFlash2 snapshot during a proposal".into());
        }
        Ok(DFlashState {
            executor: self.executor.save_snapshot(&self.model)?,
            pending: self.pending.clone(),
            controller: self.controller.state(),
            lookup: self.lookup.save(),
            primed: self.primed,
            config: (self.max_context, self.max_draft_tokens, self.policy),
        })
    }

    pub fn restore(&mut self, state: &DFlashState) -> Result<(), String> {
        if state.config != (self.max_context, self.max_draft_tokens, self.policy) {
            return Err("DFlash2 snapshot was taken with another configuration".into());
        }
        let committed =
            state.executor.valid_context() as usize + state.pending.len() / self.width();
        if !state.lookup.valid(committed) {
            return Err("DFlash lookup snapshot has invalid positions".into());
        }
        self.controller.restore(state.controller)?;
        self.executor
            .restore_snapshot(&self.model, &state.executor)?;
        self.lookup.restore(state.lookup);
        self.pending = state.pending.clone();
        self.proposed.clear();
        self.primed = state.primed;
        self.proposal_active = false;
        Ok(())
    }
}
