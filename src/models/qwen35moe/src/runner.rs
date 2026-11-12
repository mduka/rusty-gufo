//! The Qwen3.6 35B-A3B serving runner, a port of the extension's
//! `QwenTextRunner` and `QwenTextRunnerState` (`q35/model/runner.cpp`),
//! autoregressive or with DFlash2 or MTP speculation: [`QwenRunner`] over a
//! [`ServeDraft`], the DFlash2 [`Draft`] (also the autoregressive runner) or
//! the [`MtpBackend`].
//!
//! It is the dense runner (`gufo-qwen35`) with the extension's differences:
//! the concurrent draft cap of a routed-expert target ([`concurrent_draft_cap`]),
//! seeded sampled requests capped for the configured sessions, DFlash2 wide
//! copies of up to 64 rows narrowed until the sessions fit, which cannot be
//! rewound, and no sampling preset for this shape. The
//! extension's state lives in Gufo main's component cache (`cache::Slot`,
//! prefill rollback and partial prefill retention); this runner keeps
//! rusty-gufo's snapshot pool and disk payload of the dense runner instead.
//!
//! Prompts follow the model's own chat template when it is the pinned
//! Qwen3.6 one ([`Dialect::Qwen36`]), a deliberate deviation from the
//! extension, which renders it with the Qwen3.8 template.
//!
//! Target sessions live on the model's stream, DFlash2 sessions and the
//! shared draft scratch on a draft stream, so a proposal overlaps the
//! target's replay; the MTP head runs on the target's stream. One operation
//! runs at a time under the model lock.
//! Snapshots are captured on a transfer stream while the model runs other
//! sessions, and are immutable afterwards.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;

use gufo_core::gguf::{Gguf, read_header};
use gufo_core::gguf_identity;
use gufo_core::json_constraint::{ConstraintVocabulary, ToolFormat};
use gufo_core::qwen_tokenizer::{QwenTokenizer, StopTokens};
use gufo_core::sampling::{SamplerState, SamplingConfig, TokenLogprobs, collect_logprobs};
use gufo_core::text::{
    self, ChatRequest, Error, InitialOutputState, ReasoningTags, Result, TextDecodeSelection,
    TextDecodeStep, TextExecutionPlan, TextExecutionPlanKind, TextModelRunner, TextPrefillStep,
    TextPreparedPrompt, TextRunnerAdvance, TextRunnerCapabilities, TextRunnerDecode,
    TextRunnerDescriptor, TextRunnerMeasuredResources, TextRunnerPersistenceDescriptor,
    TextRunnerResourceClaim, Token, default_decode_batch, default_decode_step,
};
use gufo_hip::{DeviceProbe, DeviceProbeStatus, Stream};

use crate::config::ModelConfig;
use crate::dflash::draft::{Draft, DraftConfig, DraftContext};
use crate::dflash::executor::{DraftScratch, DraftSession};
use crate::dflash::model::DFlashModel;
use crate::dflash::policy::DraftPolicy;
use crate::engine::{
    MAX_WIDE_VERIFY_ROWS, Model, RecurrentCapture, SSM_REPLAY_CAPACITY, Session, Snapshot,
    ssm_replay_bytes,
};
use crate::mtp::{DEFAULT_PRIME_CHUNK, MtpBackend, MtpDraftConfig, MtpExec, MtpModel};
use crate::persistent::{
    CompactLayout, KvStorage, MAX_RESUME_TOKENS, RUNNER_VERSION, RecurrentStorage, RunnerHeader,
    RunnerLimits,
};
use crate::verifier::{
    DraftBackend, SpecExec, SpecTarget, StepRequest, StepResult, Verifier, VerifierSnapshot,
};
use gufo_qwen::chat_template::{
    ChatTemplate, ChatTemplateOptions, Dialect, render_and_tokenize, render_request_for,
    request_options_for, request_tools,
};
use gufo_qwen::vision::{
    Encoder, ImageCache, ImageContext, ImageGrid, RopeLayout, count_request_tokens_with,
    prepare_request_with, projection_width,
};

/// `QwenStateAbi` for the production policy (FP16 KV, FP32 recurrent state).
const STATE_ABI: &str = "qwen-gfx1151-state-v2-fp16-kv";
const DFLASH_STATE_ABI: &str = "qwen-gfx1151-dflash-state-v3-fp16-kv";
const MTP_STATE_ABI: &str = "qwen-gfx1151-mtp-state-v1-fp16-kv";
/// The serving `SpeculativeOptions` the identity records.
const ROLLING_WINDOW: u32 = 16;
const TARGET_ACCEPTANCE: f32 = 0.70;
/// From this context decode attention splits (`kSplitKDecodeAttentionMinContext`)
/// into at most this many parts (`kSplitKDecodeAttentionMaxSplits`).
const SPLIT_K_MIN_CONTEXT: u32 = 128;
const SPLIT_K_MAX_SPLITS: u32 = 32;

fn engine(message: String) -> Error {
    Error::runtime(message)
}

/// The serving view of a device probe.
fn device_status(status: DeviceProbeStatus) -> text::DeviceStatus {
    match status {
        DeviceProbeStatus::Usable => text::DeviceStatus::Usable,
        DeviceProbeStatus::Pending => text::DeviceStatus::Pending,
        DeviceProbeStatus::Lost => text::DeviceStatus::Lost,
    }
}

/// `ForwardToken`'s check at the end of the context.
const TOKEN_BEYOND_CONTEXT: &str = "token position exceeds the GPU context length";
/// `ForwardVerificationBatch`'s check at the end of the context.
const VERIFICATION_BEYOND_CONTEXT: &str = "Qwen verification exceeds a session context";

/// `QwenCompatibilityIdentity`: what a persisted continuation depends on,
/// byte for byte as the C++ runner writes it.
fn compatibility_identity(
    artifact_fingerprint: &str,
    max_context: u32,
    dialect: Dialect,
    state_abi: &str,
    draft: Option<String>,
) -> Vec<u8> {
    let mut identity = format!(
        "schema=gufo-text-continuation-v2\n\
         model_kind=qwen3.8\n\
         artifact_id={}:{artifact_fingerprint}\n\
         tokenizer=embedded-in-artifact\n\
         chat_template={}\n\
         chat_template_reference_sha256={}\n\
         state_abi={state_abi}\n\
         payload_layout=qwen-gfx1151-live-prefix-v3\n\
         checkpoint_frontier=complete-prefix-v1\n\
         numerics=qwen-bf16-fp32-prefill-v1\n\
         rmsnorm=fused-square-sum-v1\n\
         prefill_attention=visible-causal-tail-v1\n\
         attention_split_min_context={SPLIT_K_MIN_CONTEXT}\n\
         attention_split_count={SPLIT_K_MAX_SPLITS}\n\
         kv_storage=fp16\n\
         recurrent_storage=fp32\n\
         context_tokens={max_context}\n\
         position_policy=absolute-v1\n\
         rope_window_policy=qwen-gguf-config-v1\n\
         adapters=none\n",
        gguf_identity::SCHEME,
        dialect.id(),
        dialect.reference_sha256(),
    );
    if let Some(draft) = draft {
        identity.push_str(&draft);
    }
    identity.into_bytes()
}

/// The identity lines of a DFlash2 draft.
fn dflash_identity(fingerprint: &str, options: &DFlashOptions) -> String {
    let policy = match options.policy {
        DraftPolicy::Fixed => "fixed",
        DraftPolicy::Adaptive => "adaptive",
    };
    format!(
        "draft_backend=dflash2-gfx1151-v1\n\
         draft_artifact_id={}:{fingerprint}\n\
         draft_state_layout=dflash-window-kv-and-frontier-v4\n\
         draft_policy={policy}-v3\n\
         draft_max_tokens={}\n\
         draft_min_tokens={}\n\
         draft_initial_tokens={}\n\
         draft_rolling_window={ROLLING_WINDOW}\n\
         draft_target_acceptance={}\n\
         draft_adaptive=false\n\
         draft_batched_verification=true\n\
         draft_target_verification=decode-equivalent-v1\n",
        gguf_identity::SCHEME,
        options.max_draft_tokens,
        options.min_draft_tokens,
        options.max_draft_tokens,
        max_digits10(TARGET_ACCEPTANCE),
    )
}

/// The identity lines of the MTP head (`QwenCompatibilityIdentity` with
/// `kMtp`): its full-context cache and greedy chain must not cross-reuse
/// DFlash2 entries.
fn mtp_identity(fingerprint: &str, options: &MtpOptions) -> String {
    format!(
        "draft_backend=qwen-mtp-gfx1151-v1\n\
         draft_artifact_id={}:{fingerprint}\n\
         draft_state_layout=qwen-mtp-full-kv-and-hidden-row-v1\n\
         draft_propose=mtp-chain-greedy-v1\n\
         draft_max_tokens={}\n\
         draft_min_tokens={}\n\
         draft_initial_tokens={}\n\
         draft_rolling_window={ROLLING_WINDOW}\n\
         draft_target_acceptance={}\n\
         draft_adaptive=true\n\
         draft_batched_verification=true\n\
         draft_target_verification=decode-equivalent-v1\n",
        gguf_identity::SCHEME,
        options.max_draft_tokens,
        options.min_draft_tokens,
        options.max_draft_tokens,
        max_digits10(TARGET_ACCEPTANCE),
    )
}

/// `value` as `std::ostream` prints it at `max_digits10` (nine significant
/// digits, `%g` style).
fn max_digits10(value: f32) -> String {
    let value = f64::from(value);
    if value == 0.0 {
        return "0".into();
    }
    let exponent = value.abs().log10().floor() as i32;
    if (-5..9).contains(&exponent) {
        let decimals = (8 - exponent).max(0) as usize;
        let text = format!("{value:.decimals$}");
        let text = if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.')
        } else {
            &text
        };
        return text.to_string();
    }
    let text = format!("{value:.8e}");
    let (mantissa, exponent) = text.split_once('e').unwrap_or((&text, "0"));
    let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
    let exponent: i32 = exponent.parse().unwrap_or(0);
    format!(
        "{mantissa}e{}{:02}",
        if exponent < 0 { '-' } else { '+' },
        exponent.abs()
    )
}

pub struct RunnerOptions {
    pub model_path: PathBuf,
    pub max_context: u32,
    /// Persistent snapshots for a disk cache: fingerprints the artifacts.
    pub persistence: bool,
    pub dflash: Option<DFlashOptions>,
    /// `--speculative mtp`: the embedded MTP head; only the
    /// [`MtpBackend`] runner takes it.
    pub mtp: Option<MtpOptions>,
    /// `--sessions`: a seeded sampled request drafts as if all of them
    /// decoded ([`QwenRunner::draft_cap`]).
    pub sessions: usize,
    /// `--mmproj`: the BF16 vision sidecar of the 35B-A3B; without it
    /// `mmproj-BF16.gguf` beside the model serves images when present.
    pub vision_model_path: Option<PathBuf>,
}

/// DFlash2 speculation (`TextSpeculativeConfig`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DFlashOptions {
    pub model_path: PathBuf,
    pub max_draft_tokens: u32,
    pub min_draft_tokens: u32,
    pub policy: DraftPolicy,
}

/// MTP speculation (`TextSpeculativeConfig` with `kMtp`): the GGUF holding
/// the `nextn` block, the target itself for Qwen3.6 35B-A3B.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpOptions {
    pub model_path: PathBuf,
    pub max_draft_tokens: u32,
    pub min_draft_tokens: u32,
}

/// Whether `path` is a `qwen35moe` artifact (`ProbeQwenGguf`), which this
/// runner claims; its loader reports what it cannot run.
pub fn is_qwen35moe(path: &Path) -> bool {
    let Ok((header, len)) = read_header(path) else {
        return false;
    };
    Gguf::parse_prefix(&header, len)
        .is_ok_and(|gguf| gguf.string("general.architecture") == Some(b"qwen35moe"))
}

/// Whether the runner takes the model at `path`: a `qwen35moe` artifact of
/// a shape and formats the Rust kernels carry, with a valid template.
pub fn supports(path: &Path) -> bool {
    let Ok((header, len)) = read_header(path) else {
        return false;
    };
    let Ok(gguf) = Gguf::parse_prefix(&header, len) else {
        return false;
    };
    let architecture = gguf.string("general.architecture");
    let Ok(config) = ModelConfig::from_gguf(&gguf) else {
        return false;
    };
    architecture == Some(b"qwen35moe")
        && crate::engine::shape_supported(&config).is_ok()
        && crate::weights::formats_supported(&gguf, config.num_layers)
        && ChatTemplate::validate_gguf(&gguf, crate::config::is_qwen38_artifact(&gguf)).is_ok()
}

/// The draft ceiling of a speculative step shared by `active` sessions
/// (`ConcurrentDraftCap`); `u32::MAX` leaves the draft uncapped. On a
/// routed-expert target each stacked verification row reads mostly other
/// experts, so batched verification grows with the rows and concurrent
/// sessions draft less: DFlash2 two tokens at two sessions and one from
/// three, MTP one at two and none from three. Dense targets keep the full
/// draft. The extension measured it on Qwen3.6-35B-A3B (gfx1151).
pub fn concurrent_draft_cap(active: usize, moe_target: bool, mtp_draft: bool) -> u32 {
    if active <= 1 || !moe_target {
        return u32::MAX;
    }
    match (active, mtp_draft) {
        (2, true) => 1,
        (2, false) => 2,
        (_, true) => 0,
        (_, false) => 1,
    }
}

/// Extra device bytes of a session whose wide copies verify up to `rows`
/// rows: its SSM replay ring grows from [`SSM_REPLAY_CAPACITY`] slots to
/// `rows.next_power_of_two()` (`QwenGpuArena::EnsureSsmReplayCapacity`).
fn wide_copy_bytes(config: &ModelConfig, rows: u32) -> usize {
    let ring = (rows as usize).next_power_of_two();
    if ring <= SSM_REPLAY_CAPACITY {
        return 0;
    }
    ssm_replay_bytes(config, ring) - ssm_replay_bytes(config, SSM_REPLAY_CAPACITY)
}

/// The widest copy the configured sessions fit (`LimitCopyRows` in
/// `CreateQwenTextRunner`): from [`MAX_WIDE_VERIFY_ROWS`] it halves while
/// `sessions` states with the larger ring exceed the free device memory,
/// down to an ordinary verification's rows. Without a free figure the
/// width stays.
fn copy_rows_for(
    config: &ModelConfig,
    sessions: usize,
    state_bytes: usize,
    free: Option<usize>,
) -> u32 {
    let mut rows = MAX_WIDE_VERIFY_ROWS as u32;
    while let Some(free) = free
        && rows as usize > SSM_REPLAY_CAPACITY
        && sessions.saturating_mul(state_bytes + wide_copy_bytes(config, rows)) > free
    {
        rows /= 2;
    }
    rows
}

static NEXT_RUNNER: AtomicU64 = AtomicU64::new(1);

/// What runs under the model lock: the target and the drafts' shared
/// resources.
struct Exec<D: ServeDraft> {
    model: Model,
    draft: Option<D::Locked>,
}

/// The draft scratch and the stream it and every draft session live on,
/// apart from the target's: a proposal overlaps the target's replay.
pub struct DraftExec {
    stream: Stream,
    scratch: DraftScratch,
}

/// The draft side of a DFlash2 runner. The model is immutable after load,
/// so captures read it without the lock.
pub struct DFlash {
    model: DFlashModel,
    config: DraftConfig,
    options: DFlashOptions,
    min_draft_tokens: u32,
    max_draft_tokens: u32,
    history_capacity: u32,
    /// Rows a lookup copy of a request decoding alone may verify
    /// (`copy_rows_`, [`DraftConfig::max_copy_rows`]); the ring it needs is
    /// claimed per session.
    copy_rows: u32,
}

/// The draft side of an MTP runner: the packed `nextn` block of the target
/// GGUF, on the target's stream.
pub struct Mtp {
    model: MtpModel,
    config: MtpDraftConfig,
    options: MtpOptions,
}

/// A draft the serving runner verifies with (`QwenDraftKind`): the DFlash2
/// [`Draft`], whose runner also serves without a draft, or the
/// [`MtpBackend`]. One runner serves one kind.
pub trait ServeDraft:
    DraftBackend<Snapshot: Send + Sync + 'static, Model: Send + Sync + 'static> + Send + 'static
{
    /// The draft's model and configuration, read without the model lock.
    type Shared: Send + Sync + 'static;
    /// What every session's draft shares under the model lock.
    type Locked: Send + 'static;
    /// The MTP head (`QwenDraftKind::kMtp`): its concurrent draft cap.
    const MTP: bool;
    /// `QwenStateAbi` of a runner with this draft.
    const STATE_ABI: &'static str;

    /// Checks `options`' draft before the target loads.
    fn check(options: &RunnerOptions) -> Result<()>;
    /// Loads the draft of `options`, `None` without one.
    fn load(model: &Model, options: &RunnerOptions)
    -> Result<Option<(Self::Shared, Self::Locked)>>;
    /// The draft GGUF a persistent cache fingerprints.
    fn artifact(shared: &Self::Shared) -> &Path;
    /// The identity lines of the draft with its artifact `fingerprint`.
    fn identity(shared: &Self::Shared, fingerprint: &str) -> String;
    /// Device bytes of one session's draft state.
    fn state_bytes(shared: &Self::Shared, max_context: u32) -> usize;
    /// Extra device bytes a session may grow to (the wide-copy ring).
    fn scratch_bytes(_shared: &Self::Shared, _config: &ModelConfig) -> usize {
        0
    }
    /// Sizes what depends on the configured sessions and the free memory
    /// once the per-session state is known.
    fn fit(
        _shared: &mut Self::Shared,
        _config: &ModelConfig,
        _sessions: usize,
        _state_bytes: usize,
    ) {
    }
    /// The calls' context.
    fn exec<'a>(
        model: &'a mut Model,
        shared: &'a Self::Shared,
        locked: &'a mut Self::Locked,
    ) -> Self::Exec<'a>;
    /// A new session's verifier.
    fn verifier(
        model: &Model,
        shared: &Self::Shared,
        locked: &mut Self::Locked,
    ) -> std::result::Result<Verifier<Self>, String>;
    /// The draft model snapshots read.
    fn model(shared: &Self::Shared) -> &Self::Model;
    /// The draft history a persistent payload holds, where there is one.
    fn history_capacity(_shared: &Self::Shared) -> u32 {
        0
    }
    /// `DiscardPendingTargetContext` of a cancelled verification.
    fn discard_pending(
        verifier: &mut Verifier<Self>,
        position: u32,
    ) -> std::result::Result<(), String>;
}

impl ServeDraft for Draft {
    type Shared = DFlash;
    type Locked = DraftExec;
    const MTP: bool = false;
    const STATE_ABI: &'static str = DFLASH_STATE_ABI;

    fn check(options: &RunnerOptions) -> Result<()> {
        if options.mtp.is_some() {
            return Err(Error::logic("the DFlash2 runner does not draft with MTP"));
        }
        let Some(dflash) = &options.dflash else {
            return Ok(());
        };
        if dflash.max_draft_tokens == 0
            || dflash.min_draft_tokens == 0
            || dflash.min_draft_tokens > dflash.max_draft_tokens
        {
            return Err(Error::invalid("HTTP speculative draft limits are invalid"));
        }
        if dflash.min_draft_tokens != 1 {
            return Err(Error::invalid(
                "DFlash2 requires --min-draft-tokens 1; bound blocks with --draft-tokens",
            ));
        }
        if dflash.model_path.as_os_str().is_empty() {
            return Err(Error::invalid(
                "DFlash HTTP decoding requires --dflash-model",
            ));
        }
        Ok(())
    }

    fn load(model: &Model, options: &RunnerOptions) -> Result<Option<(DFlash, DraftExec)>> {
        let Some(dflash) = &options.dflash else {
            return Ok(None);
        };
        let draft_model = DFlashModel::load(model.stream(), &dflash.model_path, model.weights())
            .map_err(|e| Error::runtime(format!("Failed to create DFlash model: {e}")))?;
        let draft_config = DraftConfig::for_model(
            &draft_model,
            options.max_context,
            dflash.max_draft_tokens,
            dflash.policy,
        )
        .map_err(|e| Error::runtime(format!("Failed to create DFlash session: {e}")))?;
        let stream = Stream::new().map_err(|e| engine(format!("HIP stream: {e}")))?;
        let scratch = DraftScratch::new(&stream, &draft_model, options.max_context)
            .map_err(|e| Error::runtime(format!("Failed to create DFlash session: {e}")))?;
        let history_capacity = options.max_context.min(draft_model.config.sliding_window);
        let max_draft_tokens = dflash.max_draft_tokens.max(1);
        Ok(Some((
            DFlash {
                model: draft_model,
                config: draft_config,
                options: dflash.clone(),
                min_draft_tokens: dflash.min_draft_tokens.clamp(1, max_draft_tokens),
                max_draft_tokens,
                history_capacity,
                copy_rows: 0,
            },
            DraftExec { stream, scratch },
        )))
    }

    fn artifact(shared: &DFlash) -> &Path {
        &shared.options.model_path
    }

    fn identity(shared: &DFlash, fingerprint: &str) -> String {
        dflash_identity(fingerprint, &shared.options)
    }

    /// The draft's window rings and draft table
    /// (`QwenDFlashGpuExecutor::EstimateMemoryUsage`).
    fn state_bytes(shared: &DFlash, max_context: u32) -> usize {
        DraftSession::device_bytes(&shared.model, max_context)
    }

    /// A wide copy grows the session's SSM replay ring to every one of its
    /// rows (`ResourceClaim`).
    fn scratch_bytes(shared: &DFlash, config: &ModelConfig) -> usize {
        wide_copy_bytes(config, shared.copy_rows)
    }

    /// Each session's wide copies need a larger SSM replay ring. Narrow the
    /// width until the configured sessions fit, rather than refusing a
    /// session count the ordinary verification width fits.
    fn fit(shared: &mut DFlash, config: &ModelConfig, sessions: usize, state_bytes: usize) {
        let free = gufo_hip::device_memory().ok().map(|(free, _)| free);
        shared.copy_rows = copy_rows_for(config, sessions, state_bytes, free);
        shared.config = shared.config.with_max_copy_rows(shared.copy_rows);
    }

    fn exec<'a>(
        model: &'a mut Model,
        shared: &'a DFlash,
        locked: &'a mut DraftExec,
    ) -> SpecExec<'a> {
        SpecExec {
            model,
            draft_model: &shared.model,
            draft_scratch: &mut locked.scratch,
            draft_stream: &locked.stream,
        }
    }

    /// The verifier admits a wide lookup copy; the draft keeps neural blocks
    /// and batched copies at `--draft-tokens` (`BindComponentState`).
    fn verifier(
        _model: &Model,
        shared: &DFlash,
        locked: &mut DraftExec,
    ) -> std::result::Result<Verifier<Draft>, String> {
        let ctx = DraftContext {
            stream: &locked.stream,
            model: &shared.model,
            scratch: &mut locked.scratch,
        };
        let max_draft_tokens = if shared.copy_rows > shared.max_draft_tokens + 1 {
            shared.copy_rows - 1
        } else {
            shared.max_draft_tokens
        };
        Verifier::new(
            &ctx,
            shared.config,
            shared.min_draft_tokens,
            max_draft_tokens,
        )
        .map_err(|e| format!("Failed to create DFlash session: {e}"))
    }

    fn model(shared: &DFlash) -> &DFlashModel {
        &shared.model
    }

    fn history_capacity(shared: &DFlash) -> u32 {
        shared.history_capacity
    }

    fn discard_pending(
        verifier: &mut Verifier<Draft>,
        position: u32,
    ) -> std::result::Result<(), String> {
        verifier.draft_mut().discard_pending(position)
    }
}

impl ServeDraft for MtpBackend {
    type Shared = Mtp;
    type Locked = ();
    const MTP: bool = true;
    const STATE_ABI: &'static str = MTP_STATE_ABI;

    fn check(options: &RunnerOptions) -> Result<()> {
        if options.dflash.is_some() {
            return Err(Error::logic("the MTP runner does not draft with DFlash2"));
        }
        let Some(mtp) = &options.mtp else {
            return Err(Error::logic("the MTP runner needs --speculative mtp"));
        };
        if mtp.max_draft_tokens == 0 {
            return Err(Error::invalid("HTTP speculative draft limits are invalid"));
        }
        if mtp.min_draft_tokens != 1 {
            return Err(Error::invalid(
                "MTP drafts a fixed proposal chain; --min-draft-tokens must be 1",
            ));
        }
        if mtp.model_path.as_os_str().is_empty() {
            return Err(Error::invalid("MTP HTTP decoding requires --mtp-model"));
        }
        Ok(())
    }

    fn load(model: &Model, options: &RunnerOptions) -> Result<Option<(Mtp, ())>> {
        let Some(mtp) = &options.mtp else {
            return Ok(None);
        };
        let mtp_model = MtpModel::load(model.stream(), &mtp.model_path, model.weights())
            .map_err(|e| Error::runtime(format!("Failed to create MTP model: {e}")))?;
        Ok(Some((
            Mtp {
                model: mtp_model,
                config: MtpDraftConfig {
                    max_context: options.max_context,
                    max_draft_tokens: mtp.max_draft_tokens,
                    prime_chunk: DEFAULT_PRIME_CHUNK,
                },
                options: mtp.clone(),
            },
            (),
        )))
    }

    fn artifact(shared: &Mtp) -> &Path {
        &shared.options.model_path
    }

    fn identity(shared: &Mtp, fingerprint: &str) -> String {
        mtp_identity(fingerprint, &shared.options)
    }

    /// The head's FP32 KV cache over the context, its prime chunk and step
    /// scratch (`QwenMtpGpuExecutor::EstimateMemoryUsage`, the dominant
    /// terms).
    fn state_bytes(shared: &Mtp, max_context: u32) -> usize {
        let c = &shared.model.config;
        let (hidden, kv) = (c.hidden_size as usize, c.kv_size() as usize);
        let prime = shared.config.prime_chunk.min(max_context) as usize;
        2 * kv * max_context as usize * 4
            + prime * (7 * hidden + 2 * kv) * 4
            + 3 * c.vocab_size as usize * 4
    }

    fn exec<'a>(model: &'a mut Model, shared: &'a Mtp, _locked: &'a mut ()) -> MtpExec<'a> {
        MtpExec {
            model,
            mtp: &shared.model,
        }
    }

    /// Adaptive draft length and batched verification, as
    /// `CreateQwenTextRunner` mirrors `gufo prompt --speculative mtp`.
    fn verifier(
        model: &Model,
        shared: &Mtp,
        _locked: &mut (),
    ) -> std::result::Result<Verifier<MtpBackend>, String> {
        let backend = MtpBackend::new(model.stream(), &shared.model, shared.config)
            .map_err(|e| format!("Failed to create MTP session: {e}"))?;
        Ok(Verifier::with_draft(
            backend,
            shared.options.min_draft_tokens,
            shared.options.max_draft_tokens,
            true,
        ))
    }

    fn model(shared: &Mtp) -> &MtpModel {
        &shared.model
    }

    fn discard_pending(
        verifier: &mut Verifier<MtpBackend>,
        position: u32,
    ) -> std::result::Result<(), String> {
        verifier.draft_mut().draft_mut().discard_pending(position)
    }
}

/// The serving runner; `D` is its draft's kind, [`Draft`] also without a
/// draft.
pub struct QwenRunner<D: ServeDraft = Draft> {
    id: u64,
    exec: Mutex<Exec<D>>,
    draft: Option<D::Shared>,
    /// Snapshot captures and persistent serialisation; shared with the
    /// threads that allocate a capture's storage ahead.
    transfer: Arc<Mutex<Stream>>,
    config: ModelConfig,
    tokenizer: QwenTokenizer,
    /// The chat template prompts follow: the model's own Qwen3.6 one when
    /// its GGUF carries it.
    dialect: Dialect,
    max_context: u32,
    max_prompt_batch: usize,
    /// The configured sessions (`session_count_`), for the draft cap of a
    /// seeded sampled request.
    sessions: usize,
    persistence: Option<TextRunnerPersistenceDescriptor>,
    state_bytes: usize,
    /// The 35B-A3B's vision encoder; hashes the sidecar for image prompts
    /// on HTTP threads.
    vision: Option<Arc<Encoder>>,
    /// Decoded and resized images of earlier requests.
    image_cache: ImageCache,
    /// Tells a lost GPU context from an ordinary failure.
    device_probe: DeviceProbe,
}

/// A request's sequence state and its frontier (`QwenTextRunnerState`).
pub struct QwenState<D: ServeDraft = Draft> {
    session: Session,
    position: usize,
    frontier: Option<u32>,
    frontier_logits: Vec<f32>,
    frontier_published: bool,
    resume_tokens: Vec<u32>,
    /// `Invalidate` resets the device state before the next use.
    reset_pending: bool,
    state_bytes: usize,
    verifier: Option<Verifier<D>>,
    /// The speculative sequence; its last token anchors a proposal.
    sequence: Vec<u32>,
    /// The frontier before an unfinished verification, for cancellation.
    rollback_position: Option<usize>,
    rollback_tokens: Vec<u32>,
    rollback_logits: Vec<f32>,
    /// False for an `ignore_eos` request: stop tokens are published and
    /// generation continues. Verification still ends its accepted run at
    /// every stop token, which clips the draft window there.
    stop_at_eos: bool,
    /// Storage of the next snapshot's recurrent state, allocated while the
    /// prefill that reaches it runs ([`TextModelRunner::prepare_snapshot`]);
    /// locked because a capture reads the state shared.
    prepared: Mutex<Option<JoinHandle<std::result::Result<RecurrentCapture, String>>>>,
}

impl<D: ServeDraft> QwenState<D> {
    /// The storage a [`TextModelRunner::prepare_snapshot`] allocated, once
    /// its allocation finished.
    fn take_prepared(&self) -> Option<RecurrentCapture> {
        let handle = self
            .prepared
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()?;
        handle.join().ok()?.ok()
    }

    /// Drops storage left unused once the state moves past the snapshot it
    /// was allocated for.
    fn discard_prepared(&mut self) {
        let prepared = self
            .prepared
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(handle) = prepared.take() {
            let _ = handle.join();
        }
    }

    fn set_frontier(&mut self, frontier: u32) {
        self.frontier = Some(frontier);
        self.frontier_logits.clear();
        self.frontier_published = false;
        self.resume_tokens.clear();
    }

    fn set_pending_token(&mut self, token: Option<u32>) {
        self.resume_tokens.clear();
        self.resume_tokens.extend(token);
    }

    fn verifier(&mut self) -> Result<&mut Verifier<D>> {
        self.verifier
            .as_mut()
            .ok_or_else(|| Error::logic("Qwen state has no speculative verifier"))
    }

    /// `VerificationRequest`: the step of this state, which may be rolled
    /// back to the current frontier until it completes.
    fn verification_request<'a>(
        &'a mut self,
        remaining: usize,
        sampler: &'a mut SamplerState,
        stops: StopTokens,
    ) -> Result<StepRequest<'a, D>> {
        let frontier = self
            .frontier
            .ok_or_else(|| Error::logic("Qwen speculative state has no frontier"))?;
        let position = u32::try_from(self.position)
            .map_err(|_| Error::length("token position exceeds the GPU context length"))?;
        self.rollback_position = Some(self.position);
        self.rollback_tokens.clear();
        self.rollback_tokens.push(frontier);
        self.rollback_logits = std::mem::take(&mut self.frontier_logits);
        let verifier = self
            .verifier
            .as_mut()
            .ok_or_else(|| Error::logic("Qwen state has no speculative verifier"))?;
        Ok(StepRequest {
            verifier,
            session: &mut self.session,
            sequence: &self.sequence,
            position,
            current_token: frontier,
            stops,
            max_emitted: u32::try_from(remaining).unwrap_or(u32::MAX),
            sampler,
        })
    }
}

impl<D: ServeDraft> text::TextRunnerState for QwenState<D> {
    fn invalidate(&mut self) {
        self.discard_prepared();
        if let Some(verifier) = self.verifier.as_mut() {
            verifier.reset();
        }
        self.reset_pending = true;
        self.sequence.clear();
        self.position = 0;
        self.frontier = None;
        self.frontier_logits.clear();
        self.frontier_published = false;
        self.resume_tokens.clear();
        self.rollback_position = None;
        self.rollback_tokens.clear();
        self.rollback_logits.clear();
    }

    fn set_stop_at_eos(&mut self, stop_at_eos: bool) -> Result<()> {
        self.stop_at_eos = stop_at_eos;
        Ok(())
    }

    fn set_cancellation_check(&mut self, check: Option<text::CancellationCheck>) {
        self.session.set_cancellation_check(check);
    }

    fn measured_resources(&self) -> TextRunnerMeasuredResources {
        TextRunnerMeasuredResources {
            per_request_state_bytes: Some(self.state_bytes),
            temporary_scratch_bytes: Some(0),
        }
    }
}

/// An immutable continuation.
pub struct QwenSnapshot<D: ServeDraft = Draft> {
    runner: u64,
    snapshot: Snapshot,
    position: usize,
    frontier: Option<u32>,
    frontier_logits: Vec<f32>,
    pending_tokens: Vec<u32>,
    verifier: Option<VerifierSnapshot<D>>,
}

impl<D: ServeDraft> text::TextRunnerSnapshot for QwenSnapshot<D> {
    fn payload_bytes(&self) -> usize {
        self.snapshot.bytes()
            + self.frontier_logits.len() * 4
            + self.pending_tokens.len() * 4
            + self
                .verifier
                .as_ref()
                .map_or(0, VerifierSnapshot::<D>::payload_bytes)
    }
}

impl<D: ServeDraft> QwenRunner<D> {
    /// The vision encoder, when the model serves images.
    pub fn vision_encoder(&self) -> Option<&Encoder> {
        self.vision.as_deref()
    }

    pub fn load(options: &RunnerOptions) -> Result<Self> {
        D::check(options)?;
        let dialect = read_header(&options.model_path)
            .ok()
            .and_then(|(header, len)| {
                Gguf::parse_prefix(&header, len)
                    .ok()
                    .map(|gguf| Dialect::of_gguf(&gguf))
            })
            .unwrap_or_default();
        let model = Model::load(&options.model_path, options.max_context).map_err(engine)?;
        let config = model.config().clone();
        // `CreateQwenTextRunner`: the trunk's width selects the projector;
        // without `--mmproj`, `mmproj-BF16.gguf` beside the model (or one
        // directory up) serves images when present.
        let vision = match projection_width(&config.architecture, config.hidden_size) {
            Some(width) => {
                let sidecar = options.vision_model_path.clone().unwrap_or_default();
                Encoder::open(&options.model_path, &sidecar, width)?
            }
            None if options.vision_model_path.is_some() => {
                return Err(Error::invalid(
                    "image input supports Qwen3.8-27B, Qwen3.6-35B-A3B and Flash-Next",
                ));
            }
            None => None,
        };
        let (draft, locked) = match D::load(&model, options)? {
            Some((shared, locked)) => (Some(shared), Some(locked)),
            None => (None, None),
        };
        let persistence = if options.persistence {
            let fingerprint =
                gguf_identity::identity_hex(std::slice::from_ref(&options.model_path))
                    .map_err(|e| Error::runtime(format!("Failed to fingerprint Qwen GGUF: {e}")))?;
            let draft_identity = match &draft {
                Some(shared) => {
                    let path = D::artifact(shared).to_path_buf();
                    let fingerprint = gguf_identity::identity_hex(std::slice::from_ref(&path))
                        .map_err(|e| {
                            Error::runtime(format!("Failed to fingerprint draft GGUF: {e}"))
                        })?;
                    Some(D::identity(shared, &fingerprint))
                }
                None => None,
            };
            let state_abi = if draft.is_some() {
                D::STATE_ABI
            } else {
                STATE_ABI
            };
            Some(TextRunnerPersistenceDescriptor {
                compatibility_identity: compatibility_identity(
                    &fingerprint,
                    options.max_context,
                    dialect,
                    state_abi,
                    draft_identity,
                ),
                payload_version: RUNNER_VERSION,
            })
        } else {
            None
        };
        let transfer = Stream::new().map_err(|e| engine(format!("HIP stream: {e}")))?;
        let tokenizer = model.weights().tokenizer.clone();
        let (layers, kv) = (
            config.full_attention_layer_count() as usize,
            config.kv_size() as usize,
        );
        let ssm = config.ssm_layer_count() as usize;
        let state_bytes = 2 * layers * kv * options.max_context as usize * 2
            + ssm * config.ssm_qkv_size() as usize * config.ssm_conv_kernel as usize * 4
            + ssm
                * config.ssm_time_step_rank as usize
                * config.ssm_state_size as usize
                * config.ssm_value_size() as usize
                * 4
            + config.vocab_size as usize * 4
            + crate::MAX_TARGET_TAPS * config.hidden_size as usize * 4
            + draft
                .as_ref()
                .map_or(0, |shared| D::state_bytes(shared, options.max_context));
        let mut draft = draft;
        if let Some(shared) = draft.as_mut() {
            D::fit(shared, &config, options.sessions.max(1), state_bytes);
        }
        Ok(Self {
            id: NEXT_RUNNER.fetch_add(1, Ordering::Relaxed),
            device_probe: DeviceProbe::new().map_err(|e| engine(format!("device probe: {e}")))?,
            max_prompt_batch: model.max_prompt_batch() as usize,
            exec: Mutex::new(Exec {
                model,
                draft: locked,
            }),
            draft,
            transfer: Arc::new(Mutex::new(transfer)),
            config,
            tokenizer,
            dialect,
            max_context: options.max_context,
            sessions: options.sessions.max(1),
            persistence,
            state_bytes,
            vision,
            image_cache: ImageCache::default(),
        })
    }

    /// The draft ceiling of `sampler`'s request while `active` requests
    /// verify (`DraftCap`). Greedy and unseeded sampled requests follow the
    /// requests decoding now; a seeded sampled request takes the ceiling of
    /// the configured sessions, since its drafts consume its RNG and it must
    /// draft the same at any load to replay the same text.
    fn draft_cap(&self, sampler: &SamplerState, active: usize) -> u32 {
        let config = sampler.config();
        let seeded = config.uses_random_sampling() && config.seed >= 0;
        concurrent_draft_cap(
            if seeded { self.sessions } else { active },
            self.config.is_moe(),
            D::MTP,
        )
    }

    fn exec(&self) -> MutexGuard<'_, Exec<D>> {
        self.exec
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The target and the draft's resources of a speculative step.
    fn spec<'a>(&'a self, exec: &'a mut Exec<D>) -> Result<D::Exec<'a>> {
        let Exec { model, draft } = exec;
        match (&self.draft, draft.as_mut()) {
            (Some(shared), Some(locked)) => Ok(D::exec(model, shared, locked)),
            _ => Err(Error::logic("Qwen state has no speculative verifier")),
        }
    }

    /// The template options of `request` in the model's dialect.
    fn request_options(&self, request: &ChatRequest) -> ChatTemplateOptions {
        request_options_for(self.dialect, request, self.max_context)
    }

    fn transfer(&self) -> MutexGuard<'_, Stream> {
        self.transfer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn stop_tokens(&self) -> StopTokens {
        self.tokenizer.stop_tokens()
    }

    /// Two to eight sessions per batched step (`MaximumDecodeBatchWidth`);
    /// speculation reserves eight token rows per session.
    fn maximum_decode_batch_width(&self) -> usize {
        let rows_per_session = if self.draft.is_some() { 8 } else { 1 };
        (self.max_context as usize / rows_per_session).clamp(1, 8)
    }

    fn reset_if_pending(&self, model: &Model, state: &mut QwenState<D>) -> Result<()> {
        if state.reset_pending {
            model.reset(&mut state.session).map_err(engine)?;
            state.reset_pending = false;
        }
        Ok(())
    }

    /// `SelectFrontier`: the argmax frontier, or a draw from the frontier
    /// logits under the request's policy.
    fn select_frontier(
        model: &mut Model,
        state: &mut QwenState<D>,
        sampler: &mut SamplerState,
    ) -> Result<u32> {
        let frontier = state
            .frontier
            .ok_or_else(|| Error::logic("Qwen state has no next-token frontier"))?;
        // Without a verifier the frontier is the unmodified argmax, which a
        // greedy request may take when the grammar admits it.
        if sampler.config().can_use_unmodified_argmax()
            || (state.verifier.is_none() && sampler.can_select_argmax(frontier, false)?)
        {
            return Ok(frontier);
        }
        let cached =
            (!state.frontier_logits.is_empty()).then_some(state.frontier_logits.as_slice());
        model
            .sample(&mut state.session, cached, sampler)
            .map_err(engine)
    }

    /// The log probabilities of `token` in the frontier logits
    /// (`CopyFrontierLogits`: the retained row, else the target's last) when
    /// the request asks for them.
    fn frontier_logprobs(
        model: &Model,
        state: &QwenState<D>,
        token: u32,
        sampler: &SamplerState,
    ) -> Result<Option<TokenLogprobs>> {
        let count = sampler.config().logprobs;
        if count.is_none() {
            return Ok(None);
        }
        if state.frontier_logits.is_empty() {
            let logits = model.logits(&state.session).map_err(engine)?;
            collect_logprobs(&logits, token, count)
        } else {
            collect_logprobs(&state.frontier_logits, token, count)
        }
    }

    /// `PrimeSpeculative`: a fresh target and draft over `prompt`.
    fn prime_speculative(
        &self,
        state: &mut QwenState<D>,
        prompt: &[u32],
        decode_ready: bool,
    ) -> Result<()> {
        let mut exec = self.exec();
        let mut spec = self.spec(&mut exec)?;
        let QwenState {
            verifier, session, ..
        } = state;
        let verifier = verifier
            .as_mut()
            .ok_or_else(|| Error::logic("Qwen state has no speculative verifier"))?;
        let frontier = verifier
            .prime(&mut spec, session, prompt, decode_ready)
            .map_err(engine)?;
        state.reset_pending = false;
        state.frontier = Some(frontier);
        state.frontier_logits.clear();
        if decode_ready {
            state.frontier_logits = spec.target_ref().logits(&state.session).map_err(engine)?;
        }
        state.sequence.clear();
        state.sequence.extend_from_slice(prompt);
        state.position = prompt.len();
        state.frontier_published = false;
        state.resume_tokens.clear();
        state.rollback_position = None;
        Ok(())
    }

    /// `ExtendSpeculative`: `suffix` after the retained prefix.
    fn extend_speculative(
        &self,
        state: &mut QwenState<D>,
        suffix: &[u32],
        decode_ready: bool,
    ) -> Result<()> {
        if state.verifier.is_none() || state.frontier.is_none() {
            return Err(Error::logic("Qwen speculative prefix is not initialized"));
        }
        if state.sequence.len() != state.position {
            return Err(Error::logic(
                "Qwen speculative sequence does not match retained position",
            ));
        }
        let position = u32::try_from(state.position)
            .map_err(|_| Error::length("prompt exceeds the GPU context length"))?;
        let mut exec = self.exec();
        let mut spec = self.spec(&mut exec)?;
        let QwenState {
            verifier, session, ..
        } = state;
        let verifier = verifier
            .as_mut()
            .ok_or_else(|| Error::logic("Qwen state has no speculative verifier"))?;
        let frontier = verifier
            .extend_prompt(&mut spec, session, suffix, position, decode_ready)
            .map_err(engine)?;
        state.frontier = Some(frontier);
        state.sequence.extend_from_slice(suffix);
        state.position += suffix.len();
        state.frontier_logits.clear();
        if decode_ready {
            state.frontier_logits = spec.target_ref().logits(&state.session).map_err(engine)?;
        }
        state.frontier_published = false;
        Ok(())
    }

    /// `AppendSpeculativeSelection`: publishes `token` with its log
    /// probabilities unless it stops.
    fn append_selection(
        &self,
        state: &mut QwenState<D>,
        token: u32,
        sampler: &mut SamplerState,
        result: &mut TextDecodeStep,
        logprobs: Option<TokenLogprobs>,
    ) -> Result<bool> {
        if state.stop_at_eos && self.stop_tokens().contains(token) {
            result.terminal_logprobs = logprobs;
            result.stop = true;
            return Ok(false);
        }
        result.selections.push(TextDecodeSelection {
            stop: false,
            token,
            piece: self.tokenizer.decode(&[token]),
            logprobs,
        });
        state.sequence.push(token);
        state.set_pending_token(Some(token));
        sampler.accept(token)?;
        Ok(true)
    }

    /// `PrepareSpeculativeDecode`: publishes the frontier once; whether a
    /// verification step follows within `max_tokens`.
    fn prepare_speculative_decode(
        &self,
        model: &mut Model,
        state: &mut QwenState<D>,
        max_tokens: usize,
        sampler: &mut SamplerState,
        result: &mut TextDecodeStep,
    ) -> Result<bool> {
        if state.verifier.is_none() || state.frontier.is_none() {
            return Err(Error::logic("Qwen speculative state has no frontier"));
        }
        if max_tokens == 0 {
            return Err(Error::invalid(
                "Qwen speculative decode budget must be at least one token",
            ));
        }
        state.resume_tokens.clear();
        if !state.frontier_published {
            if !sampler.config().can_use_unmodified_argmax() {
                state.frontier = Some(Self::select_frontier(model, state, sampler)?);
            }
            let frontier = state
                .frontier
                .ok_or_else(|| Error::logic("Qwen speculative state has no frontier"))?;
            let logprobs = Self::frontier_logprobs(model, state, frontier, sampler)?;
            if !self.append_selection(state, frontier, sampler, result, logprobs)? {
                return Ok(false);
            }
            state.frontier_published = true;
        }
        Ok(result.selections.len() < max_tokens)
    }

    /// `FinishSpeculativeDecode`: publishes the verified tokens up to a stop
    /// token and keeps the correction as the next frontier.
    fn finish_speculative_decode(
        &self,
        state: &mut QwenState<D>,
        verification: StepResult,
        sampler: &mut SamplerState,
        result: &mut TextDecodeStep,
    ) -> Result<()> {
        result.draft_rounds = usize::from(verification.draft_count > 0);
        result.draft_tokens = verification.draft_count;
        result.draft_accepted_tokens = verification.accepted_count;
        result.lookup_tokens = verification.lookup_tokens;
        result.lookup_accepted_tokens = verification.lookup_accepted_tokens;
        result.execution_plan = TextExecutionPlan {
            kind: if verification.physical_width > 1 {
                TextExecutionPlanKind::Batched
            } else {
                TextExecutionPlanKind::Serial
            },
            physical_width: verification.physical_width,
        };
        let mut logprobs = verification.logprobs.into_iter();
        for &token in &verification.emitted {
            // Each prediction consumes one input, including a terminal
            // prediction; the stop token stays the unpublished frontier.
            state.position += 1;
            if !self.append_selection(state, token, sampler, result, logprobs.next())? {
                break;
            }
            state.rollback_tokens.push(token);
        }
        state.frontier = Some(verification.next_token);
        state.frontier_logits = verification.next_token_logits;
        state.frontier_published = !result.stop;
        let pending = state.frontier.filter(|_| state.frontier_published);
        state.set_pending_token(pending);
        // A wide copy that kept more rows than the draft may leave pending
        // has injected them already: its state is final, like a target-only
        // step's.
        if verification.draft_count == 0 || verification.emitted.len() >= MAX_RESUME_TOKENS {
            state.rollback_position = None;
        }
        Ok(())
    }

    /// `DecodeSpeculative`: one verification step of a single state.
    fn decode_speculative(
        &self,
        state: &mut QwenState<D>,
        max_tokens: usize,
        sampler: &mut SamplerState,
    ) -> Result<TextDecodeStep> {
        let mut result = TextDecodeStep::default();
        let mut working = sampler.clone();
        let mut exec = self.exec();
        if self.prepare_speculative_decode(
            &mut exec.model,
            state,
            max_tokens,
            &mut working,
            &mut result,
        )? {
            // At the end of the context the step has no draft and its decode
            // fails as C++ `ForwardToken` does; below it, drafts fit.
            if state.position >= self.max_context as usize {
                return Err(Error::length(TOKEN_BEYOND_CONTEXT));
            }
            let remaining = max_tokens - result.selections.len();
            state
                .verifier()?
                .set_step_draft_cap(self.draft_cap(sampler, 1));
            let verification = {
                let mut spec = self.spec(&mut exec)?;
                let mut request =
                    state.verification_request(remaining, &mut working, self.stop_tokens())?;
                Verifier::<D>::verify_step(&mut spec, &mut request).map_err(engine)?
            };
            self.finish_speculative_decode(state, verification, &mut working, &mut result)?;
        }
        sampler.set_rng_state(working.rng_state());
        Ok(result)
    }

    /// `ResumeGeneratedToken`: a token the previous request published but
    /// did not feed runs with decode arithmetic before the new suffix.
    fn resume_generated_token(
        &self,
        state: &mut QwenState<D>,
        token: u32,
        decode_ready: bool,
    ) -> Result<bool> {
        let Some(&front) = state.resume_tokens.first() else {
            return Ok(false);
        };
        if front != token {
            state.resume_tokens.clear();
            return Ok(false);
        }
        state.resume_tokens.remove(0);
        let mut exec = self.exec();
        let position = u32::try_from(state.position)
            .map_err(|_| Error::length("token position exceeds the GPU context length"))?;
        let frontier = if state.verifier.is_some() {
            let mut spec = self.spec(&mut exec)?;
            let QwenState {
                verifier, session, ..
            } = &mut *state;
            let verifier = verifier
                .as_mut()
                .ok_or_else(|| Error::logic("Qwen state has no speculative verifier"))?;
            let frontier = verifier
                .advance_committed_token(&mut spec, session, token, position)
                .map_err(engine)?;
            state.sequence.push(token);
            frontier
        } else {
            exec.model
                .forward_token(&mut state.session, token, position, true)
                .map_err(engine)?
                .ok_or_else(|| Error::runtime("decode returned no token"))?
        };
        state.frontier = Some(frontier);
        state.position += 1;
        state.frontier_logits.clear();
        if decode_ready {
            state.frontier_logits = exec.model.logits(&state.session).map_err(engine)?;
        }
        Ok(true)
    }

    fn check_snapshot<'a>(&self, snapshot: &'a QwenSnapshot<D>) -> Result<&'a QwenSnapshot<D>> {
        if snapshot.runner != self.id || snapshot.frontier.is_none() {
            return Err(Error::invalid(
                "Qwen snapshot does not belong to this model",
            ));
        }
        if snapshot.verifier.is_some() != self.draft.is_some() {
            return Err(Error::invalid(
                "Qwen snapshot speculative mode does not match the destination",
            ));
        }
        Ok(snapshot)
    }

    /// The compact layout of `snapshot`'s target state with its image grids.
    fn compact_layout(&self, valid_context: u32, images: Vec<ImageGrid>) -> Result<CompactLayout> {
        let c = &self.config;
        let ssm = c.ssm_layer_count() as usize;
        CompactLayout::new(
            c.full_attention_layer_count(),
            c.kv_size(),
            self.max_context,
            valid_context,
            KvStorage::F16,
            RecurrentStorage::F32,
            ssm * c.ssm_qkv_size() as usize * c.ssm_conv_kernel as usize,
            ssm * c.ssm_time_step_rank as usize
                * c.ssm_state_size as usize
                * c.ssm_value_size() as usize,
            images,
        )
        .map_err(Error::invalid)
    }

    fn persistent_header(
        &self,
        snapshot: &QwenSnapshot<D>,
    ) -> Result<(RunnerHeader, CompactLayout)> {
        let invalid = || Error::invalid("Qwen persistent snapshot does not belong to this runner");
        let frontier = snapshot.frontier.ok_or_else(invalid)?;
        if snapshot.runner != self.id
            || snapshot.verifier.is_some() != self.draft.is_some()
            || snapshot.position != snapshot.snapshot.valid_context() as usize
            || snapshot.pending_tokens.len() > MAX_RESUME_TOKENS
            || snapshot.frontier_logits.len() != self.config.vocab_size as usize
        {
            return Err(invalid());
        }
        let compact = self.compact_layout(
            snapshot.snapshot.valid_context(),
            snapshot.snapshot.vision_layout().images.clone(),
        )?;
        let verifier_bytes = match (&snapshot.verifier, &self.draft) {
            (Some(verifier), Some(shared)) => verifier
                .persistent_bytes(D::model(shared), D::history_capacity(shared))
                .map_err(Error::invalid)?,
            _ => 0,
        };
        let header = RunnerHeader {
            position: snapshot.position as u64,
            frontier,
            speculative: snapshot.verifier.is_some(),
            pending_tokens: snapshot.pending_tokens.clone(),
            logits_count: snapshot.frontier_logits.len(),
            target_bytes: compact.total_bytes(),
            verifier_bytes,
        };
        Ok((header, compact))
    }
}

impl<D: ServeDraft> TextModelRunner for QwenRunner<D> {
    type State = QwenState<D>;
    type Snapshot = QwenSnapshot<D>;
    type Sampler = SamplerState;
    type PromptContext = ImageContext;

    fn descriptor(&self) -> TextRunnerDescriptor {
        let speculative = self.draft.is_some();
        let width = self.maximum_decode_batch_width();
        TextRunnerDescriptor {
            model_id: self.config.model_name.clone(),
            state_abi: if speculative { D::STATE_ABI } else { STATE_ABI }.into(),
            max_context: self.max_context,
            capabilities: TextRunnerCapabilities {
                incremental_prefill: true,
                snapshot: true,
                fork: true,
                final_token_advance_required: !speculative,
                incremental_text_is_exact: true,
                multi_token_decode: speculative,
                batched_multi_token_decode: speculative && width > 1,
                batched_multi_token_decode_max_width: if speculative { width } else { 0 },
                prefix_reuse: true,
                in_pass_checkpoint: false,
            },
            persistence: self.persistence.clone(),
        }
    }

    fn resource_claim(&self) -> TextRunnerResourceClaim {
        let free = gufo_hip::device_memory().ok().map(|(free, _)| free);
        // Snapshot buffers are device allocations, but on a unified-memory
        // APU they come from the same RAM as host allocations, and HIP's free
        // figure does not see host pressure: cap them by the host budget too.
        let host = text::host_snapshot_budget_bytes();
        TextRunnerResourceClaim {
            resident_weights_bytes: None,
            state_capacity_bytes: free,
            per_request_state_bytes: Some(self.state_bytes),
            temporary_scratch_bytes: Some(
                self.draft
                    .as_ref()
                    .map_or(0, |shared| D::scratch_bytes(shared, &self.config)),
            ),
            retained_snapshot_capacity_bytes: Some(free.map_or(host, |free| free.min(host))),
            retained_snapshot_ceiling_bytes: Some(
                free.map_or(text::host_snapshot_ceiling_bytes(), |free| {
                    free.min(text::host_snapshot_ceiling_bytes())
                }),
            ),
            requires_device_runtime_lock: true,
        }
    }

    fn supported_plans(&self) -> Vec<TextExecutionPlan> {
        std::iter::once(TextExecutionPlan::SERIAL)
            .chain((2..=self.maximum_decode_batch_width()).map(TextExecutionPlan::batched))
            .collect()
    }

    fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
        Ok(self.tokenizer.encode(text))
    }

    /// llama.cpp's Qwen3-Coder format forces `\n</think>` and also ends
    /// reasoning at a tool call.
    fn reasoning_budget_tags(&self) -> ReasoningTags {
        ReasoningTags {
            start: "<think>",
            forced_end: "\n</think>",
            ends: &["</think>", "<tool_call>"],
        }
    }

    fn render_and_tokenize(&self, request: &ChatRequest) -> Result<Option<Vec<Token>>> {
        Ok(render_and_tokenize(
            &self.tokenizer,
            &request.messages,
            request_tools(request),
            &self.request_options(request),
        )
        .ok())
    }

    fn render_prompt(&self, request: &ChatRequest) -> Result<Option<String>> {
        render_request_for(self.dialect, request, self.max_context)
            .map(Some)
            .map_err(Error::invalid)
    }

    fn prepare_prompt(
        &self,
        request: &ChatRequest,
    ) -> Result<Option<TextPreparedPrompt<ImageContext>>> {
        let (prepared, context) = prepare_request_with(
            &self.tokenizer,
            request,
            &self.request_options(request),
            self.vision.as_deref(),
            self.max_context,
            &self.image_cache,
        )?;
        Ok(Some(TextPreparedPrompt {
            tokens: prepared.tokens,
            context,
            cache_prefix_tokens: prepared.cache_prefix_tokens,
        }))
    }

    fn count_prompt_tokens(&self, request: &ChatRequest) -> Result<Option<usize>> {
        count_request_tokens_with(
            &self.tokenizer,
            request,
            &self.request_options(request),
            self.vision.as_deref(),
        )
    }

    fn set_prompt_context(
        &self,
        state: &mut QwenState<D>,
        context: Option<&Arc<ImageContext>>,
    ) -> Result<()> {
        let exec = self.exec();
        self.reset_if_pending(&exec.model, state)?;
        exec.model
            .configure_vision(
                &mut state.session,
                context.map(|c| Arc::clone(&c.prompt)),
                self.vision.clone(),
            )
            .map_err(Error::invalid)
    }

    fn tool_format(&self) -> ToolFormat {
        ToolFormat::Qwen
    }

    fn supports_images(&self) -> bool {
        self.vision.is_some()
    }

    fn poll_device(&self) -> text::DeviceStatus {
        device_status(self.device_probe.poll())
    }

    fn initial_output_state(&self, request: &ChatRequest) -> InitialOutputState {
        if self.request_options(request).enable_thinking {
            InitialOutputState::Reasoning
        } else {
            InitialOutputState::Content
        }
    }

    fn build_constraint_vocabulary(&self) -> Result<ConstraintVocabulary> {
        self.tokenizer.constraint_vocabulary()
    }

    fn decode(&self, tokens: &[Token]) -> Result<Vec<u8>> {
        Ok(self.tokenizer.decode(tokens))
    }

    fn create_state(&self) -> Result<QwenState<D>> {
        let mut exec = self.exec();
        let mut session = exec
            .model
            .new_session()
            .map_err(|e| Error::runtime(format!("Failed to create GPU session: {e}")))?;
        if self.draft.is_some() {
            // The rollback pair exists before readiness, as in C++
            // (`ContinuationAdapter::CreateSlot`): a request's first
            // verification then allocates nothing.
            exec.model
                .prepare_rollback(&mut session)
                .map_err(|e| Error::runtime(format!("Failed to create GPU session: {e}")))?;
        }
        let verifier = match (&self.draft, &mut *exec) {
            (
                Some(shared),
                Exec {
                    model,
                    draft: Some(locked),
                },
            ) => Some(D::verifier(model, shared, locked).map_err(Error::runtime)?),
            _ => None,
        };
        Ok(QwenState {
            session,
            position: 0,
            frontier: None,
            frontier_logits: Vec::new(),
            frontier_published: false,
            resume_tokens: Vec::with_capacity(MAX_RESUME_TOKENS),
            reset_pending: false,
            state_bytes: self.state_bytes,
            verifier,
            sequence: Vec::new(),
            rollback_position: None,
            rollback_tokens: Vec::with_capacity(MAX_RESUME_TOKENS),
            rollback_logits: Vec::new(),
            stop_at_eos: true,
            prepared: Mutex::new(None),
        })
    }

    fn create_sampler(&self, config: &SamplingConfig, prompt: &[Token]) -> Result<SamplerState> {
        SamplerState::new(config.clone(), prompt)
    }

    fn preview_first_token(
        &self,
        state: &mut QwenState<D>,
        sampler: &mut SamplerState,
    ) -> Result<Option<TextDecodeSelection>> {
        self.select_next(state, sampler).map(Some)
    }

    fn prepare_prefix_reuse(&self, state: &mut QwenState<D>, prefix: &[Token]) -> Result<()> {
        if state.position != prefix.len() || state.frontier.is_none() {
            return Err(Error::logic(
                "Qwen retained prefix does not match its target state",
            ));
        }
        if let Some(verifier) = state.verifier.as_mut() {
            verifier.begin_request();
            state.sequence.clear();
            state.sequence.extend_from_slice(prefix);
            // Verification's frontier may be a sampled draw. A new request
            // selects from its distribution again, including when it
            // switches to greedy or omits the published pending token.
            if !state.resume_tokens.is_empty() && !state.frontier_logits.is_empty() {
                let mut greedy = SamplerState::new(SamplingConfig::default(), &[])?;
                state.frontier = Some(greedy.sample(&state.frontier_logits)?);
            }
        }
        state.frontier_published = false;
        state.rollback_position = None;
        Ok(())
    }

    fn prefill(
        &self,
        state: &mut QwenState<D>,
        prompt: &[Token],
        offset: usize,
        max_input_tokens: usize,
    ) -> Result<TextPrefillStep> {
        if offset != state.position {
            return Err(Error::logic(
                "Qwen prefill offset does not match retained state",
            ));
        }
        if offset >= prompt.len() {
            return Err(Error::logic("Qwen prefill has no remaining input"));
        }
        let max_input_tokens = max_input_tokens.min(self.max_prompt_batch);
        if max_input_tokens != 0
            && offset != 0
            && self.resume_generated_token(state, prompt[offset], offset + 1 == prompt.len())?
        {
            return Ok(TextPrefillStep {
                consumed_tokens: 1,
                decode_ready: offset + 1 == prompt.len(),
            });
        }
        if state.verifier.is_some() {
            if offset == 0 {
                let consumed = max_input_tokens.min(prompt.len());
                if consumed == 0 {
                    return Err(Error::logic("Qwen prefill requires a nonzero token budget"));
                }
                self.prime_speculative(state, &prompt[..consumed], consumed == prompt.len())?;
                return Ok(TextPrefillStep {
                    consumed_tokens: consumed,
                    decode_ready: consumed == prompt.len(),
                });
            }
            let consumed = max_input_tokens.min(prompt.len() - offset);
            if consumed == 0 {
                return Err(Error::logic(
                    "Qwen speculative retained prefix has no suffix to prefill",
                ));
            }
            let decode_ready = offset + consumed == prompt.len();
            self.extend_speculative(state, &prompt[offset..offset + consumed], decode_ready)?;
            return Ok(TextPrefillStep {
                consumed_tokens: consumed,
                decode_ready,
            });
        }
        let consumed = max_input_tokens.min(prompt.len() - offset);
        let decode_ready = offset + consumed == prompt.len();
        let mut exec = self.exec();
        self.reset_if_pending(&exec.model, state)?;
        let start = u32::try_from(offset)
            .map_err(|_| Error::length("prompt exceeds the GPU context length"))?;
        let frontier = exec
            .model
            .forward_prompt(
                &mut state.session,
                &prompt[offset..offset + consumed],
                start,
                decode_ready,
            )
            .map_err(engine)?;
        state.position = offset + consumed;
        if decode_ready {
            state
                .set_frontier(frontier.ok_or_else(|| Error::runtime("prefill returned no token"))?);
        }
        Ok(TextPrefillStep {
            consumed_tokens: consumed,
            decode_ready,
        })
    }

    fn select_next(
        &self,
        state: &mut QwenState<D>,
        sampler: &mut SamplerState,
    ) -> Result<TextDecodeSelection> {
        let (token, logprobs) = {
            let model = &mut self.exec().model;
            let token = Self::select_frontier(model, state, sampler)?;
            (
                token,
                Self::frontier_logprobs(model, state, token, sampler)?,
            )
        };
        if state.stop_at_eos && self.stop_tokens().contains(token) {
            return Ok(TextDecodeSelection {
                stop: true,
                token,
                piece: Vec::new(),
                logprobs,
            });
        }
        // Keep the selected token apart from the argmax frontier: sampled
        // cancellation and prompt snapshot restore must not replace it.
        state.set_pending_token(Some(token));
        Ok(TextDecodeSelection {
            stop: false,
            token,
            piece: self.tokenizer.decode(&[token]),
            logprobs,
        })
    }

    fn advance(&self, state: &mut QwenState<D>, token: Token) -> Result<()> {
        if state.verifier.is_some() {
            return Err(Error::logic(
                "Qwen speculative decoding requires a multi-token decode step",
            ));
        }
        if state.frontier.is_none() {
            return Err(Error::logic("Qwen decode state has no retained frontier"));
        }
        if state.position >= self.max_context as usize {
            return Err(Error::length(TOKEN_BEYOND_CONTEXT));
        }
        let position =
            u32::try_from(state.position).map_err(|_| Error::length(TOKEN_BEYOND_CONTEXT))?;
        let frontier = self
            .exec()
            .model
            .forward_token(&mut state.session, token, position, true)
            .map_err(engine)?
            .ok_or_else(|| Error::runtime("decode returned no token"))?;
        state.position += 1;
        state.set_frontier(frontier);
        Ok(())
    }

    fn decode_step(
        &self,
        state: &mut QwenState<D>,
        max_tokens: usize,
        sampler: &mut SamplerState,
    ) -> Result<TextDecodeStep> {
        state.discard_prepared();
        if state.verifier.is_none() {
            return default_decode_step(self, state, max_tokens, sampler);
        }
        self.decode_speculative(state, max_tokens, sampler)
    }

    /// Verification steps of two to eight speculative states in shared
    /// passes (`DecodeBatch`).
    fn decode_batch(
        &self,
        decodes: &mut [TextRunnerDecode<'_, QwenState<D>, SamplerState>],
    ) -> Result<Vec<Result<TextDecodeStep>>> {
        for decode in decodes.iter_mut() {
            decode.state.discard_prepared();
        }
        if self.draft.is_none()
            || decodes.len() < 2
            || decodes.len() > self.maximum_decode_batch_width()
        {
            return Ok(default_decode_batch(self, decodes));
        }
        let mut steps: Vec<TextDecodeStep> =
            decodes.iter().map(|_| TextDecodeStep::default()).collect();
        let mut samplers: Vec<SamplerState> = decodes.iter().map(|d| d.sampler.clone()).collect();
        let mut ready = vec![false; decodes.len()];
        let mut exec = self.exec();
        for (index, decode) in decodes.iter_mut().enumerate() {
            ready[index] = self.prepare_speculative_decode(
                &mut exec.model,
                decode.state,
                decode.max_tokens,
                &mut samplers[index],
                &mut steps[index],
            )?;
        }
        // A single step verifies like `decode_step`; several fail in the
        // batched verification at the end of a context.
        let verifying = ready.iter().filter(|&&r| r).count();
        if decodes
            .iter()
            .zip(&ready)
            .any(|(d, &r)| r && d.state.position >= self.max_context as usize)
        {
            return Err(Error::length(if verifying == 1 {
                TOKEN_BEYOND_CONTEXT
            } else {
                VERIFICATION_BEYOND_CONTEXT
            }));
        }
        let verified = {
            let mut requests = Vec::with_capacity(decodes.len());
            for (((decode, sampler), step), &ready) in decodes
                .iter_mut()
                .zip(samplers.iter_mut())
                .zip(&steps)
                .zip(&ready)
            {
                if ready {
                    let cap = self.draft_cap(sampler, verifying);
                    decode.state.verifier()?.set_step_draft_cap(cap);
                    requests.push(decode.state.verification_request(
                        decode.max_tokens - step.selections.len(),
                        sampler,
                        self.stop_tokens(),
                    )?);
                }
            }
            let mut spec = self.spec(&mut exec)?;
            Verifier::<D>::verify_batch(&mut spec, &mut requests).map_err(engine)?
        };
        let mut verified = verified.into_iter();
        for (index, decode) in decodes.iter_mut().enumerate() {
            if !ready[index] {
                continue;
            }
            let verification = verified.next().ok_or_else(|| {
                Error::runtime("target returned an incomplete verification batch")
            })?;
            self.finish_speculative_decode(
                decode.state,
                verification,
                &mut samplers[index],
                &mut steps[index],
            )?;
        }
        for (decode, sampler) in decodes.iter_mut().zip(&samplers) {
            decode.sampler.set_rng_state(sampler.rng_state());
        }
        Ok(steps.into_iter().map(Ok).collect())
    }

    fn advance_batch(
        &self,
        advances: &mut [TextRunnerAdvance<'_, QwenState<D>>],
    ) -> Result<Vec<Result<()>>> {
        if self.draft.is_some() {
            return Err(Error::logic(
                "Qwen speculative sessions do not support batch advance",
            ));
        }
        if advances.len() < 2 || advances.len() > 8 {
            return Err(Error::invalid(
                "Qwen batched decode requires two to eight sessions",
            ));
        }
        let mut items = Vec::with_capacity(advances.len());
        for advance in advances.iter_mut() {
            let state = &mut *advance.state;
            if state.frontier.is_none() {
                return Err(Error::logic("Qwen batched state has no retained frontier"));
            }
            if state.position >= self.max_context as usize {
                return Err(Error::length(
                    "token position exceeds a batched Qwen session context",
                ));
            }
            let position = u32::try_from(state.position).map_err(|_| {
                Error::length("token position exceeds a batched Qwen session context")
            })?;
            items.push((&mut state.session, advance.token, position));
        }
        let frontiers = self
            .exec()
            .model
            .forward_token_batch(&mut items)
            .map_err(engine)?;
        drop(items);
        if frontiers.len() != advances.len() {
            return Err(Error::runtime(
                "Qwen batched decode returned an invalid frontier count",
            ));
        }
        for (advance, frontier) in advances.iter_mut().zip(frontiers) {
            advance.state.position += 1;
            advance.state.set_frontier(frontier);
        }
        Ok(advances.iter().map(|_| Ok(())).collect())
    }

    fn checkpoint_position(&self, state: &QwenState<D>) -> Result<usize> {
        Ok(state.position)
    }

    /// `PrepareCancellation`: returns to the frontier before an unfinished
    /// verification; its tokens become pending again.
    fn prepare_cancellation(&self, state: &mut QwenState<D>) -> Result<()> {
        let Some(rollback) = state.rollback_position else {
            return Ok(());
        };
        if state.rollback_tokens.is_empty()
            || state.rollback_tokens.len() > MAX_RESUME_TOKENS
            || state.rollback_logits.is_empty()
        {
            return Err(Error::logic(
                "Qwen cancelled verification has no exact frontier",
            ));
        }
        // Verification already saved the target state; its features are
        // still pending in the draft.
        let exec = self.exec();
        let position = u32::try_from(rollback)
            .map_err(|_| Error::length("token position exceeds the GPU context length"))?;
        D::discard_pending(state.verifier()?, position).map_err(engine)?;
        exec.model
            .restore_state(&mut state.session)
            .map_err(engine)?;
        state.position = rollback;
        state.sequence.resize(rollback, 0);
        state.frontier_logits = std::mem::take(&mut state.rollback_logits);
        let mut greedy = SamplerState::new(SamplingConfig::default(), &[])?;
        state.frontier = Some(greedy.sample(&state.frontier_logits)?);
        state.frontier_published = false;
        state.resume_tokens = std::mem::take(&mut state.rollback_tokens);
        state.rollback_position = None;
        Ok(())
    }

    fn snapshot_payload_bytes(&self, state: &QwenState<D>) -> Result<usize> {
        if state.position > self.max_context as usize {
            return Err(Error::length("Qwen snapshot exceeds the context length"));
        }
        let compact = self.compact_layout(state.position as u32, Vec::new())?;
        let logits = if state.frontier_logits.is_empty() {
            self.config.vocab_size as usize
        } else {
            state.frontier_logits.len()
        };
        let verifier = match (&state.verifier, &self.draft) {
            (Some(verifier), Some(shared)) => verifier.snapshot_payload_bytes(D::model(shared)),
            _ => 0,
        };
        // Device bytes of the live state, frontier logits, pending tokens,
        // the image grids, the verifier.
        Ok(compact.k().len() * 2
            + compact.conv().len()
            + compact.deltanet().len()
            + logits * 4
            + state.resume_tokens.len() * 4
            + state
                .session
                .vision_layout()
                .prefix(u32::try_from(state.position).unwrap_or(u32::MAX))
                .images
                .len()
                * 12
            + verifier)
    }

    /// Allocates the recurrent part of the snapshot the pool takes after the
    /// coming prefill on a thread of its own: `hipMalloc` of its ~180 MB
    /// took ~9 ms of the capture, and of the first token's latency.
    fn prepare_snapshot(&self, state: &mut QwenState<D>) -> Result<()> {
        let lens = state.session.recurrent_lens();
        let prepared = state
            .prepared
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        if prepared.is_some() {
            return Ok(());
        }
        let transfer = self.transfer.clone();
        let handle = std::thread::Builder::new()
            .name("gufo-snapshot-alloc".into())
            .spawn(move || {
                let transfer = transfer.lock().unwrap_or_else(PoisonError::into_inner);
                RecurrentCapture::allocate(&transfer, lens)
            })
            .map_err(|e| Error::runtime(format!("snapshot allocation thread: {e}")))?;
        *prepared = Some(handle);
        Ok(())
    }

    fn snapshot(&self, state: &QwenState<D>) -> Result<QwenSnapshot<D>> {
        let frontier = state
            .frontier
            .ok_or_else(|| Error::logic("Qwen state has no exact frontier to snapshot"))?;
        // Before locking the stream, which the allocating thread holds.
        let prepared = state.take_prepared();
        let transfer = self.transfer();
        let position = u32::try_from(state.position)
            .map_err(|_| Error::length("Qwen snapshot exceeds the context length"))?;
        let snapshot = state
            .session
            .capture_with(&transfer, position, prepared)
            .map_err(engine)?;
        let frontier_logits = if state.frontier_logits.is_empty() {
            state
                .session
                .copy_logits(&transfer, self.config.vocab_size as usize)
                .map_err(engine)?
        } else {
            state.frontier_logits.clone()
        };
        let verifier = match (&state.verifier, &self.draft) {
            (Some(verifier), Some(shared)) => Some(
                verifier
                    .snapshot(&transfer, D::model(shared))
                    .map_err(engine)?,
            ),
            _ => None,
        };
        Ok(QwenSnapshot {
            runner: self.id,
            snapshot,
            position: state.position,
            frontier: Some(frontier),
            frontier_logits,
            pending_tokens: state.resume_tokens.clone(),
            verifier,
        })
    }

    fn restore_or_fork(&self, state: &mut QwenState<D>, snapshot: &QwenSnapshot<D>) -> Result<()> {
        state.discard_prepared();
        let snapshot = self.check_snapshot(snapshot)?;
        let frontier = snapshot
            .frontier
            .ok_or_else(|| Error::invalid("Qwen snapshot does not belong to this model"))?;
        let mut exec = self.exec();
        exec.model
            .restore_snapshot(&mut state.session, &snapshot.snapshot)
            .map_err(engine)?;
        state.reset_pending = false;
        state.position = snapshot.position;
        state.set_frontier(frontier);
        state.frontier_logits = snapshot.frontier_logits.clone();
        state.resume_tokens = snapshot.pending_tokens.clone();
        state.rollback_position = None;
        if let Some(verifier_snapshot) = &snapshot.verifier {
            let mut spec = self.spec(&mut exec)?;
            let QwenState {
                verifier, session, ..
            } = state;
            verifier
                .as_mut()
                .ok_or_else(|| {
                    Error::invalid("cannot restore speculative state into a plain Qwen session")
                })?
                .restore(&mut spec, session, verifier_snapshot)
                .map_err(engine)?;
        }
        Ok(())
    }

    fn persistent_snapshot_payload_bytes(&self, snapshot: &QwenSnapshot<D>) -> Result<usize> {
        let (header, _) = self.persistent_header(snapshot)?;
        header.total_bytes().map_err(Error::invalid)
    }

    fn serialize_persistent_snapshot(
        &self,
        snapshot: &QwenSnapshot<D>,
        destination: &mut [u8],
    ) -> Result<usize> {
        let (header, compact) = self.persistent_header(snapshot)?;
        if destination.len() != header.total_bytes().map_err(Error::invalid)? {
            return Err(Error::invalid(
                "Qwen persistent snapshot destination size is invalid",
            ));
        }
        header.write(destination).map_err(Error::invalid)?;
        let target = header.target();
        let payload = &mut destination[target.clone()];
        compact.write(payload).map_err(Error::invalid)?;
        let transfer = self.transfer();
        let error = |e| Error::runtime(format!("Qwen compact snapshot: {e}"));
        let kv = snapshot.snapshot.kv();
        let plane = kv.len() / 2;
        transfer
            .download_bytes(kv.slice(0..plane), &mut payload[compact.k()])
            .map_err(error)?;
        transfer
            .download_bytes(kv.slice(plane..2 * plane), &mut payload[compact.v()])
            .map_err(error)?;
        transfer
            .download_bytes(
                snapshot.snapshot.conv().view(),
                &mut payload[compact.conv()],
            )
            .map_err(error)?;
        transfer
            .download_bytes(
                snapshot.snapshot.deltanet().view(),
                &mut payload[compact.deltanet()],
            )
            .map_err(error)?;
        for (chunk, value) in destination[header.logits()]
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(&snapshot.frontier_logits)
        {
            *chunk = value.to_le_bytes();
        }
        if let (Some(verifier), Some(shared)) = (&snapshot.verifier, &self.draft) {
            verifier
                .serialize(
                    &transfer,
                    D::model(shared),
                    D::history_capacity(shared),
                    &mut destination[header.verifier()],
                )
                .map_err(|e| Error::runtime(format!("Qwen verifier serializer: {e}")))?;
        }
        Ok(destination.len())
    }

    fn restore_persistent_snapshot(&self, state: &mut QwenState<D>, payload: &[u8]) -> Result<()> {
        state.discard_prepared();
        let header = RunnerHeader::parse(
            payload,
            RunnerLimits {
                max_context: self.max_context,
                vocab_size: self.config.vocab_size,
                speculative: self.draft.is_some(),
            },
        )
        .map_err(Error::invalid)?;
        let target = &payload[header.target()];
        let layout = CompactLayout::parse(target).map_err(Error::invalid)?;
        let expected = self.compact_layout(layout.valid_context, Vec::new())?;
        if layout.valid_context as u64 != header.position
            || layout.attention_layers != expected.attention_layers
            || layout.kv_width != expected.kv_width
            || layout.max_context != expected.max_context
            || layout.kv_storage != expected.kv_storage
            || layout.recurrent_storage != expected.recurrent_storage
            || layout.conv_elements != expected.conv_elements
            || layout.deltanet_elements != expected.deltanet_elements
        {
            return Err(Error::invalid(
                "Qwen compact snapshot is incompatible with the arena",
            ));
        }
        let logits: Vec<f32> = payload[header.logits()]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        let mut exec = self.exec();
        exec.model
            .restore_compact(
                &mut state.session,
                &target[layout.k()],
                &target[layout.v()],
                &target[layout.conv()],
                &target[layout.deltanet()],
                layout.valid_context,
                &RopeLayout {
                    images: layout.images.clone(),
                },
            )
            .map_err(engine)?;
        if header.speculative {
            let mut spec = self.spec(&mut exec)?;
            let QwenState {
                verifier, session, ..
            } = &mut *state;
            verifier
                .as_mut()
                .ok_or_else(|| {
                    Error::invalid(
                        "cannot restore speculative persistent state into a plain Qwen session",
                    )
                })?
                .restore_persistent(&mut spec, session, &payload[header.verifier()])
                .map_err(Error::invalid)?;
        }
        state.reset_pending = false;
        state.position = header.position as usize;
        state.set_frontier(header.frontier);
        state.frontier_logits = logits;
        state.resume_tokens = header.pending_tokens;
        state.rollback_position = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The extension's `ConcurrentDraftCap` table.
    #[test]
    fn concurrent_draft_caps_match_the_extension() {
        for mtp in [false, true] {
            assert_eq!(concurrent_draft_cap(0, true, mtp), u32::MAX);
            assert_eq!(concurrent_draft_cap(1, true, mtp), u32::MAX);
            for active in 2..=8 {
                assert_eq!(concurrent_draft_cap(active, false, mtp), u32::MAX);
            }
        }
        assert_eq!(concurrent_draft_cap(2, true, false), 2);
        assert_eq!(concurrent_draft_cap(2, true, true), 1);
        for active in 3..=8 {
            assert_eq!(concurrent_draft_cap(active, true, false), 1);
            assert_eq!(concurrent_draft_cap(active, true, true), 0);
        }
    }

    /// Wide copies narrow by halves until the sessions fit, never below an
    /// ordinary verification's rows; without a free figure they stay.
    #[test]
    fn copy_rows_narrow_until_sessions_fit() {
        let config = ModelConfig {
            num_layers: 40,
            ..ModelConfig::default()
        };
        assert_eq!(wide_copy_bytes(&config, SSM_REPLAY_CAPACITY as u32), 0);
        let wide = wide_copy_bytes(&config, 64);
        assert!(wide > wide_copy_bytes(&config, 32));
        assert_eq!(copy_rows_for(&config, 4, 1000, None), 64);
        assert_eq!(copy_rows_for(&config, 4, 1000, Some(4 * (1000 + wide))), 64);
        assert_eq!(
            copy_rows_for(&config, 4, 1000, Some(4 * (1000 + wide) - 1)),
            32
        );
        assert_eq!(
            copy_rows_for(&config, 4, 1000, Some(0)),
            SSM_REPLAY_CAPACITY as u32
        );
    }
}
