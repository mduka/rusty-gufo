//! Prompt-history copies for the DFlash2 draft (`TryLookup` of
//! `QwenDFlashGpuDraftBackend`): where the generated text repeats an earlier
//! passage, the draft copies its continuation instead of running the neural
//! block. The index is Flash-Next's ([`gufo_core::prompt_lookup`]) with this
//! session's private request and resume state; the target still verifies
//! every copied token.
//!
//! A first match needs [`prompt_lookup::SAMPLED_MIN_MATCH`] tokens, or
//! [`prompt_lookup::MIN_MATCH`] for a greedy request whose neural acceptance
//! has fallen below its neutral estimate: parallel DFlash already predicts
//! short schema and reasoning fragments well. The decision uses only the
//! snapshotted controller state, no timings or new draws. A copy reports
//! point-mass q when sampled and draws nothing.
//!
//! A source whose continuation crosses message framing (the target
//! tokenizer's `<|im_start|>`, `<|im_end|>` and `<|endoftext|>`) is
//! rejected for an earlier match: the nearest repeat of a passage often ends
//! at a user message's closing tag and header, and copying them repeatedly
//! wastes verification. Thinking and tool tokens remain copyable.

use gufo_core::prompt_lookup::{self, LookupState, NO_STATE, PromptLookup};

use crate::dflash::executor::Proposal;

/// One session's copy state.
#[derive(Debug, Default)]
pub struct DraftLookup {
    index: PromptLookup<u32>,
    /// Committed index of the active copy's source (`lookup_start_`).
    start: usize,
    /// The active proposal is a copy (`lookup_proposal_`).
    active: bool,
}

/// What a copy depends on besides the tokens.
pub struct LookupLimits<'a> {
    /// The neural length controller's acceptance estimate.
    pub neural_mean: f32,
    /// The draft's maximum proposal length.
    pub max_draft_tokens: u32,
    pub vocab_size: u32,
    pub boundaries: &'a [u32],
}

impl DraftLookup {
    /// `Clear`: a fresh draft.
    pub fn clear(&mut self) {
        self.index.clear();
        self.active = false;
    }

    /// `BeginRequest`: nothing to resume, and the next step sets the
    /// request's boundary.
    pub fn begin_request(&mut self) {
        self.index.begin_request();
    }

    pub fn save(&self) -> LookupState {
        self.index.save()
    }

    /// Adopts a checked `state`; the index is rebuilt from the next tokens.
    pub fn restore(&mut self, state: LookupState) {
        self.index.restore(state);
        self.active = false;
    }

    /// Feedback on the last proposal: true when it was a copy, whose
    /// acceptance the lookup keeps instead of the neural controller.
    pub fn observe(&mut self, proposed: usize, accepted: usize) -> bool {
        let copied = self.active;
        if copied {
            self.index.observe(self.start, proposed, accepted);
        }
        self.active = false;
        copied
    }

    /// A copy of at most `limit` tokens after the anchor `tokens[position]`,
    /// when the committed `tokens[..position]` repeat a long enough passage.
    /// Callers that pass only the anchor never copy.
    pub fn propose(
        &mut self,
        tokens: &[u32],
        position: u32,
        limit: u32,
        sampled: bool,
        limits: &LookupLimits<'_>,
    ) -> Option<Proposal> {
        self.active = false;
        let position = position as usize;
        if limit == 0 || tokens.len() != position + 1 || !self.index.step(position) {
            return None;
        }
        let weak_neural = limits.neural_mean < 3.0f32.min(limits.max_draft_tokens as f32);
        let minimum = if !sampled && weak_neural {
            prompt_lookup::MIN_MATCH
        } else {
            prompt_lookup::SAMPLED_MIN_MATCH
        };
        let state = self.index.save();
        // As in C++, the generated length is the history beyond the request's
        // first step (that step set the boundary just above).
        if state.resume == NO_STATE
            && tokens.len().wrapping_sub(state.request_start as usize) < minimum
        {
            return None;
        }
        let committed = &tokens[..position];
        self.index.extend(committed, sampled);
        let cap = limit.min(limits.max_draft_tokens) as usize;
        let indexed = self.index.indexed_tokens();
        let boundary = |token: &u32| limits.boundaries.contains(token);
        let found = self.index.find_accepted(
            committed,
            &tokens[position..],
            minimum,
            |start| {
                let count = cap.min(indexed - start);
                !committed[start..start + count].iter().any(boundary)
            },
            false,
        );
        if found.length == 0 {
            return None;
        }
        // The source aligns with the context ending at the anchor, so its
        // continuation runs past the committed tokens into the anchor and the
        // copy itself: a short-period repetition fills the block instead of
        // proposing one period per verification.
        let mut copy: Vec<u32> = Vec::with_capacity(cap);
        let mut source = found.start;
        while copy.len() < cap {
            let token = if source <= position {
                tokens[source]
            } else {
                copy[source - position - 1]
            };
            if token >= limits.vocab_size || boundary(&token) {
                break;
            }
            copy.push(token);
            source += 1;
        }
        if copy.is_empty() {
            return None;
        }
        let mut proposal = Proposal {
            from_lookup: true,
            ..Proposal::default()
        };
        if sampled {
            // A deterministic copy is q(token) = 1, conditional only on the
            // prior history; the ordinary p/q verifier still accepts or
            // corrects every token.
            proposal.candidate_ids = copy.clone();
            proposal.candidate_probabilities = vec![1.0; copy.len()];
        }
        proposal.tokens = copy;
        self.start = found.start;
        self.active = true;
        Some(proposal)
    }
}
