//! The native MTP draft of Qwen3.6 35B-A3B, a port of the extension's
//! `QwenMtpWeights::LoadFromGguf` (`mtp_reference.cpp`), `QwenMtpGpuModel`,
//! `QwenMtpGpuExecutor` and `QwenMtpGpuDraftBackend` (`hip/mtp/model.cpp`,
//! `hip/mtp/executor.cpp`, `hip/mtp/backend.cpp`) and the small draft head
//! (`mtp_draft_vocab.hpp`).
//!
//! One MoE block after the target's layers (`blk.40` with its `nextn` fusion
//! projection and norms) predicts the token after the next: it fuses the
//! normalised embedding of the input token with the normalised final hidden
//! row of the target (or of its own previous step), runs the block over its
//! own FP32 KV cache and scores the target's LM head, or the small draft
//! head: the head's first 98,304 rows plus a per-session table of the
//! special tokens and the rare tokens the session has used. Everything runs
//! on the target's stream ([`MtpContext`]), so verified target rows reach
//! the draft without a host copy.
//!
//! - [`MtpModel`]: the packed block ([`model`]).
//! - [`MtpHead`]: one sequence's cache and scratch, the forward step, the
//!   batched prompt prime, the draft table and snapshots ([`head`]).
//! - [`MtpDraft`]: the draft backend of one session: priming, greedy,
//!   sampled and batched proposals, feedback and snapshots ([`draft`]).
//! - [`MtpBackend`]: [`MtpDraft`] as the serving verifier's
//!   [`DraftBackend`](crate::verifier::DraftBackend) over [`MtpExec`]
//!   ([`backend`]): `Verifier<MtpBackend>`.

pub mod backend;
pub mod draft;
pub mod head;
pub mod model;
pub mod sampling;
pub mod vocab;

use gufo_hip::Stream;
use gufo_qwen::vision::VisionInput;

pub use backend::{MtpBackend, MtpBackendSnapshot, MtpExec};
pub use draft::{MtpDraft, MtpDraftConfig, MtpDraftSnapshot, Proposal, ProposalRequest, Rows};
pub use head::{DEFAULT_PRIME_CHUNK, HeadSnapshot, MAX_CHAIN_TOKENS, MtpHead};
pub use model::{MtpModel, open};

/// What every MTP call runs on: the target's stream, the packed block and
/// the target session's image input, which the block reads at its own
/// positions (the C++ draft's `vision_input`).
pub struct MtpContext<'a> {
    pub stream: &'a Stream,
    pub model: &'a MtpModel,
    pub vision: Option<&'a VisionInput>,
}
