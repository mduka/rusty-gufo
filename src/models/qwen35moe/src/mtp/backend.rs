//! The MTP draft as the serving verifier drives it: [`DraftBackend`] (Gufo's
//! `IDraftBackend`) over [`MtpDraft`], as `CreateQwenGpuSpeculativeVerifier`
//! binds `QwenMtpGpuDraftBackend` to the target.
//!
//! The draft runs on the target's stream and reads the target's final hidden
//! rows where the target keeps them: the prompt rows of the prompt capture
//! (host), the decoded and verified rows in the target's scratch (device), so
//! no row is downloaded. The C++ backend reads the session's image input
//! through a pointer; here the calls that see the session take its
//! embeddings and position table directly, and the others (proposals,
//! feedback, restores) a copy of its position table that each of those calls
//! brings up to date. Draft positions past the prompt read only that table,
//! never an image embedding.

use gufo_core::sampling::SamplerState;
use gufo_hip::Stream;
use gufo_qwen::vision::{RopeLayout, VisionInput};

use crate::dflash::executor::Proposal;
use crate::engine::{Model, Session};
use crate::mtp::draft::{MtpDraft, MtpDraftConfig, MtpDraftSnapshot, ProposalRequest, Rows};
use crate::mtp::{MtpContext, MtpModel};
use crate::verifier::{DraftBackend, Proposing, SpecTarget};

/// The target and the MTP block: what an MTP verification step runs on.
pub struct MtpExec<'a> {
    pub model: &'a mut Model,
    pub mtp: &'a MtpModel,
}

impl SpecTarget for MtpExec<'_> {
    fn target(&mut self) -> &mut Model {
        self.model
    }

    fn target_ref(&self) -> &Model {
        self.model
    }
}

/// One session's MTP draft (`QwenMtpGpuDraftBackend`).
pub struct MtpBackend {
    draft: MtpDraft,
    /// The session's image positions, for the calls that do not see the
    /// session.
    rope: VisionInput,
}

/// An [`MtpBackend`] at a committed boundary: the draft and the image
/// positions it drafts with.
pub struct MtpBackendSnapshot {
    pub draft: MtpDraftSnapshot,
    pub layout: RopeLayout,
}

fn err(message: &str) -> String {
    message.to_string()
}

impl MtpBackend {
    /// A draft of `config` on the target's stream
    /// (`QwenMtpGpuDraftBackend::Create`).
    pub fn new(stream: &Stream, model: &MtpModel, config: MtpDraftConfig) -> Result<Self, String> {
        let ctx = MtpContext {
            stream,
            model,
            vision: None,
        };
        Ok(Self {
            draft: MtpDraft::with_config(&ctx, config)?,
            rope: VisionInput::default(),
        })
    }

    pub fn draft(&self) -> &MtpDraft {
        &self.draft
    }

    pub fn draft_mut(&mut self) -> &mut MtpDraft {
        &mut self.draft
    }

    /// Copies the session's image positions.
    fn follow(&mut self, stream: &Stream, st: &Session) -> Result<(), String> {
        self.rope
            .restore_layout(st.vision_layout(), stream)
            .map_err(|e| e.message().to_string())
    }

    /// The context of a call that sees the session: its own image input.
    fn with_session<'a>(exec: &'a MtpExec<'_>, st: &'a Session) -> MtpContext<'a> {
        MtpContext {
            stream: exec.model.stream(),
            model: exec.mtp,
            vision: Some(st.vision()),
        }
    }

    /// `AppendTargetContext` of `tokens` at `position` with their rows.
    fn append(
        &mut self,
        exec: &MtpExec<'_>,
        st: &Session,
        tokens: &[u32],
        rows: Rows<'_>,
        position: u32,
        error: &str,
    ) -> Result<(), String> {
        self.follow(exec.model.stream(), st)?;
        self.draft
            .append(&Self::with_session(exec, st), tokens, rows, position)
            .map_err(|_| err(error))
    }

    /// The target's verified rows of `st`, one per verified token.
    fn verified<'a>(
        exec: &'a MtpExec<'_>,
        st: &Session,
    ) -> Result<gufo_hip::DeviceSlice<'a, f32>, String> {
        exec.model.verification_rows(st)
    }
}

/// The verifier's proposal from the draft's.
fn proposal(proposal: crate::mtp::Proposal) -> Proposal {
    Proposal {
        tokens: proposal.tokens,
        candidate_ids: proposal.candidate_ids,
        candidate_probabilities: proposal.candidate_probabilities,
        from_lookup: false,
    }
}

impl DraftBackend for MtpBackend {
    type Exec<'a> = MtpExec<'a>;
    /// Proposals follow neither penalties nor constraints: verification
    /// applies them.
    type Guide = ();
    type Snapshot = MtpBackendSnapshot;
    type Model = MtpModel;

    fn guide(_sampler: &SamplerState) {}

    fn supports_sampled_proposals(&self) -> bool {
        true
    }

    fn set_step_draft_cap(&mut self, cap: u32) {
        self.draft.set_step_draft_cap(cap);
    }

    fn reset(&mut self) {
        self.draft.reset();
    }

    fn begin_request(&mut self) {
        self.draft.begin_request();
    }

    /// The final hidden rows, kept on the device where verification leaves
    /// them.
    fn capture(exec: &mut MtpExec<'_>, st: &mut Session) -> Result<(), String> {
        st.set_device_rows(true);
        st.set_hidden_capture(exec.model.config(), true, &[])
    }

    fn prime(
        &mut self,
        exec: &mut MtpExec<'_>,
        st: &Session,
        prompt: &[u32],
        _first: u32,
    ) -> Result<(), String> {
        let hidden = exec.model.prompt_hidden()?;
        if hidden.is_empty() || !hidden.len().is_multiple_of(prompt.len()) {
            return Err("target executor did not capture complete prompt hidden states".into());
        }
        self.follow(exec.model.stream(), st)?;
        self.draft
            .prime(&Self::with_session(exec, st), prompt, hidden)
            .map_err(|_| err("draft backend failed to prime target context"))
    }

    fn append_prompt(
        &mut self,
        exec: &mut MtpExec<'_>,
        st: &Session,
        tokens: &[u32],
        _first: u32,
        position: u32,
    ) -> Result<(), String> {
        let hidden = exec.model.prompt_hidden()?;
        self.append(
            exec,
            st,
            tokens,
            Rows::Host(hidden),
            position,
            "draft failed to append target prompt features",
        )
    }

    fn append_decoded(
        &mut self,
        exec: &mut MtpExec<'_>,
        st: &Session,
        token: u32,
        _next: u32,
        position: u32,
    ) -> Result<(), String> {
        let row = exec.model.final_row(st)?;
        self.append(
            exec,
            st,
            &[token],
            Rows::Device(row),
            position,
            "draft failed to append committed target token",
        )
    }

    fn append_verified(
        &mut self,
        exec: &mut MtpExec<'_>,
        st: &Session,
        token: u32,
        _next: u32,
        position: u32,
    ) -> Result<(), String> {
        let width = exec.mtp.config().hidden_size as usize;
        let rows = Self::verified(exec, st)?;
        if rows.len() < width {
            return Err("target executor returned incomplete verification hidden states".into());
        }
        self.append(
            exec,
            st,
            &[token],
            Rows::Device(rows.range(0..width)),
            position,
            "draft failed to append committed target token",
        )
    }

    fn check_verified(exec: &MtpExec<'_>, st: &Session, rows: usize) -> Result<(), String> {
        let width = exec.mtp.config().hidden_size as usize;
        if Self::verified(exec, st)?.len() != rows * width {
            return Err("target executor returned incomplete verification hidden states".into());
        }
        Ok(())
    }

    fn update_verified(
        &mut self,
        exec: &mut MtpExec<'_>,
        st: &Session,
        row: usize,
    ) -> Result<(), String> {
        let width = exec.mtp.config().hidden_size as usize;
        let rows = Self::verified(exec, st)?;
        if rows.len() < (row + 1) * width {
            return Err("target executor returned incomplete verification hidden states".into());
        }
        self.follow(exec.model.stream(), st)?;
        self.draft.update_hidden(
            &Self::with_session(exec, st),
            Rows::Device(rows.range(row * width..(row + 1) * width)),
        )
    }

    fn propose(
        &mut self,
        exec: &mut MtpExec<'_>,
        tokens: &[u32],
        position: u32,
        max_tokens: u32,
        sampling: Option<(f32, &mut u64)>,
        _guide: Option<&()>,
    ) -> Result<Proposal, String> {
        let ctx = MtpContext {
            stream: exec.model.stream(),
            model: exec.mtp,
            vision: Some(&self.rope),
        };
        self.draft
            .propose_for(&ctx, tokens, position, max_tokens, sampling)
            .map(proposal)
    }

    fn propose_batch(
        exec: &mut MtpExec<'_>,
        requests: Vec<Proposing<'_, Self>>,
    ) -> Result<Vec<Proposal>, String> {
        let ctx = MtpContext {
            stream: exec.model.stream(),
            model: exec.mtp,
            vision: None,
        };
        let mut batch: Vec<ProposalRequest<'_>> = requests
            .into_iter()
            .map(|request| {
                let MtpBackend { draft, rope } = request.draft;
                ProposalRequest {
                    draft,
                    vision: Some(&*rope),
                    tokens: request.tokens,
                    position: request.position,
                    max_tokens: request.max_tokens,
                    sampling: request.sampling,
                }
            })
            .collect();
        Ok(MtpDraft::propose_batch(&ctx, &mut batch)?
            .into_iter()
            .map(proposal)
            .collect())
    }

    fn accept(
        &mut self,
        exec: &mut MtpExec<'_>,
        accepted: &[u32],
        _correction: u32,
    ) -> Result<(), String> {
        let ctx = MtpContext {
            stream: exec.model.stream(),
            model: exec.mtp,
            vision: Some(&self.rope),
        };
        self.draft.accept(&ctx, accepted)
    }

    fn snapshot_payload_bytes(&self, model: &MtpModel) -> usize {
        self.draft.snapshot_payload_bytes(model)
    }

    fn snapshot(&self, transfer: &Stream, model: &MtpModel) -> Result<MtpBackendSnapshot, String> {
        Ok(MtpBackendSnapshot {
            draft: self.draft.snapshot(transfer, model)?,
            layout: self.rope.layout().clone(),
        })
    }

    fn snapshot_bytes(snapshot: &MtpBackendSnapshot) -> usize {
        snapshot.draft.payload_bytes()
    }

    fn restore(
        &mut self,
        exec: &mut MtpExec<'_>,
        snapshot: &MtpBackendSnapshot,
    ) -> Result<(), String> {
        let stream = exec.model.stream();
        self.rope
            .restore_layout(&snapshot.layout, stream)
            .map_err(|e| e.message().to_string())?;
        let ctx = MtpContext {
            stream,
            model: exec.mtp,
            vision: Some(&self.rope),
        };
        self.draft.restore(&ctx, &snapshot.draft)
    }

    fn persistent_bytes(
        snapshot: &MtpBackendSnapshot,
        _model: &MtpModel,
        _history_capacity: u32,
    ) -> Result<usize, String> {
        Ok(snapshot.draft.persistent_bytes())
    }

    fn serialize(
        snapshot: &MtpBackendSnapshot,
        transfer: &Stream,
        _model: &MtpModel,
        _history_capacity: u32,
        destination: &mut [u8],
    ) -> Result<(), String> {
        snapshot.draft.serialize(transfer, destination).map(|_| ())
    }

    /// The payload holds no image positions: the draft keeps the last ones
    /// it saw until the next call that sees the session (the decode or
    /// prompt extension that follows a restore) copies the session's.
    fn restore_persistent(&mut self, exec: &mut MtpExec<'_>, payload: &[u8]) -> Result<(), String> {
        let ctx = MtpContext {
            stream: exec.model.stream(),
            model: exec.mtp,
            vision: Some(&self.rope),
        };
        self.draft.restore_persistent(&ctx, payload)
    }
}
