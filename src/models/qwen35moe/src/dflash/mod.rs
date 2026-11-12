//! DFlash2 speculative decoding: the draft model, its length policy, the
//! prompt-history lookup, the draft vocabulary and the draft-side
//! bookkeeping of `QwenDFlashGpuDraftBackend` (`q35/model/hip/dflash` of the
//! qwen35moe extension). [`draft::Draft`] is one serving session's backend
//! over a shared model and scratch; [`DFlash`] is one greedy session with
//! its own model, stream and scratch, configured as `gufo prompt
//! --speculative dflash2` (no wide copies).
//!
//! Target features of committed rows arrive one row at a time
//! ([`DFlash::update_hidden`]) and wait on the host until the next proposal
//! injects them in one call, as in C++.

pub mod draft;
pub mod draft_vocab;
pub mod executor;
pub mod lookup;
pub mod model;
pub mod policy;

use std::path::Path;

use gufo_hip::Stream;

use crate::weights::Weights;
use draft::{Draft, DraftConfig, DraftContext, DraftSnapshot};
use executor::DraftScratch;
use model::DFlashModel;
use policy::DraftPolicy;

pub struct DFlash {
    stream: Stream,
    model: DFlashModel,
    scratch: DraftScratch,
    draft: Draft,
}

/// Draft state at a committed boundary (`Snapshot` of the draft backend).
pub struct DFlashState {
    draft: DraftSnapshot,
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
        let stream = Stream::new().map_err(|e| format!("DFlash2 stream: {e}"))?;
        let model = DFlashModel::load(&stream, path, target)?;
        let config = DraftConfig::for_model(&model, max_context, max_draft_tokens, policy)?;
        let mut scratch = DraftScratch::new(&stream, &model, max_context)?;
        let draft = Draft::new(
            &DraftContext {
                stream: &stream,
                model: &model,
                scratch: &mut scratch,
            },
            config,
        )?;
        Ok(Self {
            stream,
            model,
            scratch,
            draft,
        })
    }

    fn parts(&mut self) -> (DraftContext<'_>, &mut Draft) {
        (
            DraftContext {
                stream: &self.stream,
                model: &self.model,
                scratch: &mut self.scratch,
            },
            &mut self.draft,
        )
    }

    pub fn model(&self) -> &DFlashModel {
        &self.model
    }

    /// The session's backend.
    pub fn draft(&self) -> &Draft {
        &self.draft
    }

    /// Target layers whose outputs the draft reads.
    pub fn target_layers(&self) -> &[u32] {
        &self.model.config.target_layer_ids
    }

    pub fn max_draft_tokens(&self) -> u32 {
        self.draft.config().max_draft_tokens
    }

    pub fn reset(&mut self) {
        self.draft.reset();
    }

    /// Starts the draft over a prefilled prompt (`PrimeTargetContext`).
    pub fn prime(&mut self, features: &[f32], rows: usize) -> Result<(), String> {
        let (mut ctx, draft) = self.parts();
        draft.prime(&mut ctx, features, rows)
    }

    /// Appends target rows at `position` after the pending ones
    /// (`AppendTargetContext`).
    pub fn append(&mut self, features: &[f32], rows: usize, position: u32) -> Result<(), String> {
        let (mut ctx, draft) = self.parts();
        draft.append(&mut ctx, features, rows, position)
    }

    /// Queues the features of one committed row (`UpdateTargetHidden`).
    pub fn update_hidden(&mut self, row: &[f32]) -> Result<(), String> {
        self.draft.update_hidden(row)
    }

    /// Greedy proposals after the last token of `tokens` (the anchor) at
    /// `position`, at most `max_tokens` (`ProposeForRequest` of a greedy
    /// request). A complete sequence (`position + 1` tokens) may be copied
    /// from instead.
    pub fn propose(
        &mut self,
        tokens: &[u32],
        position: u32,
        max_tokens: u32,
    ) -> Result<Vec<u32>, String> {
        let (mut ctx, draft) = self.parts();
        draft
            .propose(&mut ctx, tokens, position, max_tokens, None, None)
            .map(|proposal| proposal.tokens)
    }

    /// Feedback on the last proposal: its first `accepted` tokens were kept
    /// (`AcceptFeedback`).
    pub fn accept(&mut self, accepted: usize) -> Result<(), String> {
        let (mut ctx, draft) = self.parts();
        draft.accept(&mut ctx, accepted)
    }

    pub fn snapshot(&self) -> Result<DFlashState, String> {
        Ok(DFlashState {
            draft: self.draft.snapshot(&self.stream, &self.model)?,
        })
    }

    pub fn restore(&mut self, state: &DFlashState) -> Result<(), String> {
        let (ctx, draft) = self.parts();
        draft.restore(&ctx, &state.draft)
    }
}
