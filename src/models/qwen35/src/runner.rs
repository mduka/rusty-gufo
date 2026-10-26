//! The Qwen3.8 serving runner, a port of `QwenTextRunner` and
//! `QwenTextRunnerState` (`src/cli/serve/inference_backend.cpp`) for dense
//! Qwen, with the 27B's vision sidecar, autoregressive or with DFlash2
//! speculation.
//!
//! Target sessions live on the model's stream, draft sessions and the shared
//! draft scratch on a draft stream, so a proposal overlaps the target's
//! replay; one operation runs at a time under the model lock.
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
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_core::text::{
    self, ChatRequest, Error, InitialOutputState, ReasoningTags, Result, TextDecodeSelection,
    TextDecodeStep, TextExecutionPlan, TextExecutionPlanKind, TextModelRunner, TextPrefillStep,
    TextPreparedPrompt, TextRunnerAdvance, TextRunnerCapabilities, TextRunnerDecode,
    TextRunnerDescriptor, TextRunnerMeasuredResources, TextRunnerPersistenceDescriptor,
    TextRunnerResourceClaim, Token, default_decode_batch, default_decode_step,
};
use gufo_hip::{DeviceProbe, DeviceProbeStatus, Stream};

use crate::config::ModelConfig;
use crate::dflash::draft::{DraftConfig, DraftContext};
use crate::dflash::executor::DraftScratch;
use crate::dflash::model::DFlashModel;
use crate::dflash::policy::DraftPolicy;
use crate::engine::{Model, RecurrentCapture, Session, Snapshot};
use crate::persistent::{
    CompactLayout, KvStorage, MAX_RESUME_TOKENS, RUNNER_VERSION, RecurrentStorage, RunnerHeader,
    RunnerLimits,
};
use crate::verifier::{SpecExec, StepRequest, StepResult, Verifier, VerifierSnapshot};
use gufo_qwen::chat_template::{
    ChatTemplate, OFFICIAL_TEMPLATE_SHA256, render_and_tokenize, request_options, request_tools,
};
use gufo_qwen::vision::{
    Encoder, ImageCache, ImageContext, ImageGrid, RopeLayout, count_request_tokens, prepare_request,
};

/// `QwenStateAbi` for the production policy (FP16 KV, FP32 recurrent state).
const STATE_ABI: &str = "qwen-gfx1151-state-v2-fp16-kv";
const DFLASH_STATE_ABI: &str = "qwen-gfx1151-dflash-state-v3-fp16-kv";
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
    draft: Option<(&str, &DFlashOptions)>,
) -> Vec<u8> {
    let state_abi = if draft.is_some() {
        DFLASH_STATE_ABI
    } else {
        STATE_ABI
    };
    let mut identity = format!(
        "schema=gufo-text-continuation-v2\n\
         model_kind=qwen3.8\n\
         artifact_id={}:{artifact_fingerprint}\n\
         tokenizer=embedded-in-artifact\n\
         chat_template=qwen38-reasoning-compiled-v3\n\
         chat_template_reference_sha256={OFFICIAL_TEMPLATE_SHA256}\n\
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
        gguf_identity::SCHEME
    );
    if let Some((fingerprint, options)) = draft {
        let policy = match options.policy {
            DraftPolicy::Fixed => "fixed",
            DraftPolicy::Adaptive => "adaptive",
        };
        identity.push_str(&format!(
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
        ));
    }
    identity.into_bytes()
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
    /// `--mmproj`: the vision sidecar of the 27B; without it the canonical
    /// one next to the model serves images when present.
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

/// Whether the dense Qwen runner takes the model at `path`: a `qwen35`
/// artifact of a shape and formats the Rust kernels carry, with a pinned
/// template.
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
    architecture == Some(b"qwen35")
        && crate::engine::shape_supported(&config).is_ok()
        && crate::weights::formats_supported(&gguf, config.num_layers)
        && ChatTemplate::validate_gguf(&gguf, crate::config::is_qwen38_artifact(&gguf)).is_ok()
}

static NEXT_RUNNER: AtomicU64 = AtomicU64::new(1);

/// What runs under the model lock: the target and the shared draft.
struct Exec {
    model: Model,
    draft: Option<DraftExec>,
}

/// The draft scratch and the stream it and every draft session live on,
/// apart from the target's: a proposal overlaps the target's replay.
struct DraftExec {
    stream: Stream,
    scratch: DraftScratch,
}

/// The draft side of a DFlash2 runner. The model is immutable after load,
/// so captures read it without the lock.
struct DFlash {
    model: DFlashModel,
    config: DraftConfig,
    min_draft_tokens: u32,
    max_draft_tokens: u32,
    history_capacity: u32,
}

pub struct QwenRunner {
    id: u64,
    exec: Mutex<Exec>,
    dflash: Option<DFlash>,
    /// Snapshot captures and persistent serialisation; shared with the
    /// threads that allocate a capture's storage ahead.
    transfer: Arc<Mutex<Stream>>,
    config: ModelConfig,
    tokenizer: QwenTokenizer,
    max_context: u32,
    max_prompt_batch: usize,
    persistence: Option<TextRunnerPersistenceDescriptor>,
    state_bytes: usize,
    /// The 27B's vision encoder; hashes the sidecar for image prompts on
    /// HTTP threads.
    vision: Option<Arc<Encoder>>,
    /// Decoded and resized images of earlier requests.
    image_cache: ImageCache,
    /// Tells a lost GPU context from an ordinary failure.
    device_probe: DeviceProbe,
}

/// A request's sequence state and its frontier (`QwenTextRunnerState`).
pub struct QwenState {
    session: Session,
    position: usize,
    frontier: Option<u32>,
    frontier_logits: Vec<f32>,
    frontier_published: bool,
    resume_tokens: Vec<u32>,
    /// `Invalidate` resets the device state before the next use.
    reset_pending: bool,
    state_bytes: usize,
    verifier: Option<Verifier>,
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

impl QwenState {
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

    fn verifier(&mut self) -> Result<&mut Verifier> {
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
    ) -> Result<StepRequest<'a>> {
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

impl text::TextRunnerState for QwenState {
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

/// An immutable continuation (`QwenTextRunnerSnapshot`).
pub struct QwenSnapshot {
    runner: u64,
    snapshot: Snapshot,
    position: usize,
    frontier: Option<u32>,
    frontier_logits: Vec<f32>,
    pending_tokens: Vec<u32>,
    verifier: Option<VerifierSnapshot>,
}

impl text::TextRunnerSnapshot for QwenSnapshot {
    fn payload_bytes(&self) -> usize {
        self.snapshot.bytes()
            + self.frontier_logits.len() * 4
            + self.pending_tokens.len() * 4
            + self
                .verifier
                .as_ref()
                .map_or(0, VerifierSnapshot::payload_bytes)
    }
}

impl QwenRunner {
    /// The vision encoder, when the model serves images.
    pub fn vision_encoder(&self) -> Option<&Encoder> {
        self.vision.as_deref()
    }

    pub fn load(options: &RunnerOptions) -> Result<Self> {
        if let Some(dflash) = &options.dflash {
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
        }
        let model = Model::load(&options.model_path, options.max_context).map_err(engine)?;
        let config = model.config().clone();
        let vision = if config.hidden_size == 5120 {
            let sidecar = options.vision_model_path.clone().unwrap_or_default();
            Encoder::open(&options.model_path, &sidecar, 5120)?
        } else if options.vision_model_path.is_some() {
            return Err(Error::invalid(
                "image input supports Qwen3.8-27B and Flash-Next",
            ));
        } else {
            None
        };
        let persistence = if options.persistence {
            let fingerprint =
                gguf_identity::identity_hex(std::slice::from_ref(&options.model_path))
                    .map_err(|e| Error::runtime(format!("Failed to fingerprint Qwen GGUF: {e}")))?;
            let draft_fingerprint = match &options.dflash {
                Some(dflash) => Some(
                    gguf_identity::identity_hex(std::slice::from_ref(&dflash.model_path)).map_err(
                        |e| Error::runtime(format!("Failed to fingerprint DFlash GGUF: {e}")),
                    )?,
                ),
                None => None,
            };
            let draft = draft_fingerprint.as_deref().zip(options.dflash.as_ref());
            Some(TextRunnerPersistenceDescriptor {
                compatibility_identity: compatibility_identity(
                    &fingerprint,
                    options.max_context,
                    draft,
                ),
                payload_version: RUNNER_VERSION,
            })
        } else {
            None
        };
        let (dflash, draft) = match &options.dflash {
            Some(dflash) => {
                let draft_model =
                    DFlashModel::load(model.stream(), &dflash.model_path, model.weights())
                        .map_err(|e| {
                            Error::runtime(format!("Failed to create DFlash model: {e}"))
                        })?;
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
                (
                    Some(DFlash {
                        model: draft_model,
                        config: draft_config,
                        min_draft_tokens: dflash.min_draft_tokens.clamp(1, max_draft_tokens),
                        max_draft_tokens,
                        history_capacity,
                    }),
                    Some(DraftExec { stream, scratch }),
                )
            }
            None => (None, None),
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
            // The draft's window rings (`QwenDFlashGpuExecutor::EstimateMemoryUsage`).
            + dflash.as_ref().map_or(0, |d| {
                2 * d.model.config.num_layers as usize
                    * d.history_capacity as usize
                    * d.model.config.kv_size()
                    * 4
            });
        Ok(Self {
            id: NEXT_RUNNER.fetch_add(1, Ordering::Relaxed),
            device_probe: DeviceProbe::new().map_err(|e| engine(format!("device probe: {e}")))?,
            max_prompt_batch: model.max_prompt_batch() as usize,
            exec: Mutex::new(Exec { model, draft }),
            dflash,
            transfer: Arc::new(Mutex::new(transfer)),
            config,
            tokenizer,
            max_context: options.max_context,
            persistence,
            state_bytes,
            vision,
            image_cache: ImageCache::default(),
        })
    }

    fn exec(&self) -> MutexGuard<'_, Exec> {
        self.exec
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The target, draft and draft scratch of a speculative step.
    fn spec<'a>(&'a self, exec: &'a mut Exec) -> Result<SpecExec<'a>> {
        let Exec { model, draft } = exec;
        match (&self.dflash, draft.as_mut()) {
            (Some(dflash), Some(draft)) => Ok(SpecExec {
                model,
                draft_model: &dflash.model,
                draft_scratch: &mut draft.scratch,
                draft_stream: &draft.stream,
            }),
            _ => Err(Error::logic("Qwen state has no speculative verifier")),
        }
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
        let rows_per_session = if self.dflash.is_some() { 8 } else { 1 };
        (self.max_context as usize / rows_per_session).clamp(1, 8)
    }

    fn reset_if_pending(&self, model: &Model, state: &mut QwenState) -> Result<()> {
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
        state: &mut QwenState,
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

    /// `PrimeSpeculative`: a fresh target and draft over `prompt`.
    fn prime_speculative(
        &self,
        state: &mut QwenState,
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
            state.frontier_logits = spec.model.logits(&state.session).map_err(engine)?;
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
        state: &mut QwenState,
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
            state.frontier_logits = spec.model.logits(&state.session).map_err(engine)?;
        }
        state.frontier_published = false;
        Ok(())
    }

    /// `AppendSpeculativeSelection`: publishes `token` unless it stops.
    fn append_selection(
        &self,
        state: &mut QwenState,
        token: u32,
        sampler: &mut SamplerState,
        result: &mut TextDecodeStep,
    ) -> Result<bool> {
        if state.stop_at_eos && self.stop_tokens().contains(token) {
            result.stop = true;
            return Ok(false);
        }
        result.selections.push(TextDecodeSelection {
            stop: false,
            token,
            piece: self.tokenizer.decode(&[token]),
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
        state: &mut QwenState,
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
            if !self.append_selection(state, frontier, sampler, result)? {
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
        state: &mut QwenState,
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
        for &token in &verification.emitted {
            // Each prediction consumes one input, including a terminal
            // prediction; the stop token stays the unpublished frontier.
            state.position += 1;
            if !self.append_selection(state, token, sampler, result)? {
                break;
            }
            state.rollback_tokens.push(token);
        }
        state.frontier = Some(verification.next_token);
        state.frontier_logits = verification.next_token_logits;
        state.frontier_published = !result.stop;
        let pending = state.frontier.filter(|_| state.frontier_published);
        state.set_pending_token(pending);
        if verification.draft_count == 0 {
            state.rollback_position = None;
        }
        Ok(())
    }

    /// `DecodeSpeculative`: one verification step of a single state.
    fn decode_speculative(
        &self,
        state: &mut QwenState,
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
            let verification = {
                let mut spec = self.spec(&mut exec)?;
                let mut request =
                    state.verification_request(remaining, &mut working, self.stop_tokens())?;
                Verifier::verify_step(&mut spec, &mut request).map_err(engine)?
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
        state: &mut QwenState,
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

    fn check_snapshot<'a>(&self, snapshot: &'a QwenSnapshot) -> Result<&'a QwenSnapshot> {
        if snapshot.runner != self.id || snapshot.frontier.is_none() {
            return Err(Error::invalid(
                "Qwen snapshot does not belong to this model",
            ));
        }
        if snapshot.verifier.is_some() != self.dflash.is_some() {
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

    fn persistent_header(&self, snapshot: &QwenSnapshot) -> Result<(RunnerHeader, CompactLayout)> {
        let invalid = || Error::invalid("Qwen persistent snapshot does not belong to this runner");
        let frontier = snapshot.frontier.ok_or_else(invalid)?;
        if snapshot.runner != self.id
            || snapshot.verifier.is_some() != self.dflash.is_some()
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
        let verifier_bytes = match (&snapshot.verifier, &self.dflash) {
            (Some(verifier), Some(dflash)) => verifier
                .persistent_bytes(&dflash.model, dflash.history_capacity)
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

impl TextModelRunner for QwenRunner {
    type State = QwenState;
    type Snapshot = QwenSnapshot;
    type Sampler = SamplerState;
    type PromptContext = ImageContext;

    fn descriptor(&self) -> TextRunnerDescriptor {
        let speculative = self.dflash.is_some();
        let width = self.maximum_decode_batch_width();
        TextRunnerDescriptor {
            model_id: self.config.model_name.clone(),
            state_abi: if speculative {
                DFLASH_STATE_ABI
            } else {
                STATE_ABI
            }
            .into(),
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
            temporary_scratch_bytes: Some(0),
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
            &request_options(request, self.max_context),
        )
        .ok())
    }

    fn prepare_prompt(
        &self,
        request: &ChatRequest,
    ) -> Result<Option<TextPreparedPrompt<ImageContext>>> {
        let (prepared, context) = prepare_request(
            &self.tokenizer,
            request,
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
        count_request_tokens(
            &self.tokenizer,
            request,
            self.vision.as_deref(),
            self.max_context,
        )
    }

    fn set_prompt_context(
        &self,
        state: &mut QwenState,
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
        if request_options(request, self.max_context).enable_thinking {
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

    fn create_state(&self) -> Result<QwenState> {
        let mut exec = self.exec();
        let mut session = exec
            .model
            .new_session()
            .map_err(|e| Error::runtime(format!("Failed to create GPU session: {e}")))?;
        if self.dflash.is_some() {
            // The rollback pair exists before readiness, as in C++
            // (`ContinuationAdapter::CreateSlot`): a request's first
            // verification then allocates nothing.
            exec.model
                .prepare_rollback(&mut session)
                .map_err(|e| Error::runtime(format!("Failed to create GPU session: {e}")))?;
        }
        let verifier = match &self.dflash {
            Some(dflash) => {
                let spec = self.spec(&mut exec)?;
                let ctx = DraftContext {
                    stream: spec.draft_stream,
                    model: spec.draft_model,
                    scratch: spec.draft_scratch,
                };
                Some(
                    Verifier::new(
                        &ctx,
                        dflash.config,
                        dflash.min_draft_tokens,
                        dflash.max_draft_tokens,
                    )
                    .map_err(|e| Error::runtime(format!("Failed to create DFlash session: {e}")))?,
                )
            }
            None => None,
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
        state: &mut QwenState,
        sampler: &mut SamplerState,
    ) -> Result<Option<TextDecodeSelection>> {
        self.select_next(state, sampler).map(Some)
    }

    fn prepare_prefix_reuse(&self, state: &mut QwenState, prefix: &[Token]) -> Result<()> {
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
        state: &mut QwenState,
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
                    "Qwen DFlash retained prefix has no suffix to prefill",
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
        state: &mut QwenState,
        sampler: &mut SamplerState,
    ) -> Result<TextDecodeSelection> {
        let token = Self::select_frontier(&mut self.exec().model, state, sampler)?;
        if state.stop_at_eos && self.stop_tokens().contains(token) {
            return Ok(TextDecodeSelection {
                stop: true,
                token: 0,
                piece: Vec::new(),
            });
        }
        // Keep the selected token apart from the argmax frontier: sampled
        // cancellation and prompt snapshot restore must not replace it.
        state.set_pending_token(Some(token));
        Ok(TextDecodeSelection {
            stop: false,
            token,
            piece: self.tokenizer.decode(&[token]),
        })
    }

    fn advance(&self, state: &mut QwenState, token: Token) -> Result<()> {
        if state.verifier.is_some() {
            return Err(Error::logic(
                "Qwen DFlash decoding requires a multi-token decode step",
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
        state: &mut QwenState,
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
        decodes: &mut [TextRunnerDecode<'_, QwenState, SamplerState>],
    ) -> Result<Vec<Result<TextDecodeStep>>> {
        for decode in decodes.iter_mut() {
            decode.state.discard_prepared();
        }
        if self.dflash.is_none()
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
                    requests.push(decode.state.verification_request(
                        decode.max_tokens - step.selections.len(),
                        sampler,
                        self.stop_tokens(),
                    )?);
                }
            }
            let mut spec = self.spec(&mut exec)?;
            Verifier::verify_batch(&mut spec, &mut requests).map_err(engine)?
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
        advances: &mut [TextRunnerAdvance<'_, QwenState>],
    ) -> Result<Vec<Result<()>>> {
        if self.dflash.is_some() {
            return Err(Error::logic(
                "Qwen DFlash sessions do not support batch advance",
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

    fn checkpoint_position(&self, state: &QwenState) -> Result<usize> {
        Ok(state.position)
    }

    /// `PrepareCancellation`: returns to the frontier before an unfinished
    /// verification; its tokens become pending again.
    fn prepare_cancellation(&self, state: &mut QwenState) -> Result<()> {
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
        state
            .verifier()?
            .draft_mut()
            .discard_pending(position)
            .map_err(engine)?;
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

    fn snapshot_payload_bytes(&self, state: &QwenState) -> Result<usize> {
        if state.position > self.max_context as usize {
            return Err(Error::length("Qwen snapshot exceeds the context length"));
        }
        let compact = self.compact_layout(state.position as u32, Vec::new())?;
        let logits = if state.frontier_logits.is_empty() {
            self.config.vocab_size as usize
        } else {
            state.frontier_logits.len()
        };
        let verifier = match (&state.verifier, &self.dflash) {
            (Some(verifier), Some(dflash)) => verifier.snapshot_payload_bytes(&dflash.model),
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
    fn prepare_snapshot(&self, state: &mut QwenState) -> Result<()> {
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

    fn snapshot(&self, state: &QwenState) -> Result<QwenSnapshot> {
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
        let verifier = match (&state.verifier, &self.dflash) {
            (Some(verifier), Some(dflash)) => Some(
                verifier
                    .snapshot(&transfer, &dflash.model)
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

    fn restore_or_fork(&self, state: &mut QwenState, snapshot: &QwenSnapshot) -> Result<()> {
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

    fn persistent_snapshot_payload_bytes(&self, snapshot: &QwenSnapshot) -> Result<usize> {
        let (header, _) = self.persistent_header(snapshot)?;
        header.total_bytes().map_err(Error::invalid)
    }

    fn serialize_persistent_snapshot(
        &self,
        snapshot: &QwenSnapshot,
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
        if let (Some(verifier), Some(dflash)) = (&snapshot.verifier, &self.dflash) {
            verifier
                .serialize(
                    &transfer,
                    &dflash.model,
                    dflash.history_capacity,
                    &mut destination[header.verifier()],
                )
                .map_err(|e| Error::runtime(format!("Qwen verifier serializer: {e}")))?;
        }
        Ok(destination.len())
    }

    fn restore_persistent_snapshot(&self, state: &mut QwenState, payload: &[u8]) -> Result<()> {
        state.discard_prepared();
        let header = RunnerHeader::parse(
            payload,
            RunnerLimits {
                max_context: self.max_context,
                vocab_size: self.config.vocab_size,
                speculative: self.dflash.is_some(),
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
