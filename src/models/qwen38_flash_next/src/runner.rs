//! The serving runner of Qwen3.8-Flash-Next, a port of
//! `QwenFlashNextTextRunner` (`src/cli/serve/inference_backend.cpp`): one
//! engine session per request, prefill through prompt sync, MTP decode
//! steps (batched across requests), and host snapshots that restore in
//! memory and persist to disk.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use gufo_core::gguf::{Gguf, read_header};
use gufo_core::gguf_identity;
use gufo_core::json_constraint::{ConstraintVocabulary, ToolFormat};
use gufo_core::qwen_tokenizer::QwenTokenizer;
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_core::text::{
    self, CancellationCheck, ChatRequest, Error, InitialOutputState, ReasoningTags, Result,
    TextDecodeSelection, TextDecodeStep, TextExecutionPlan, TextModelRunner, TextPrefillStep,
    TextPrefillThrough, TextPreparedPrompt, TextRunnerAdvance, TextRunnerCapabilities,
    TextRunnerDecode, TextRunnerDescriptor, TextRunnerMeasuredResources,
    TextRunnerPersistenceDescriptor, TextRunnerResourceClaim, TextRunnerSnapshot, Token,
};
use gufo_hip::{DeviceProbe, DeviceProbeStatus};
use gufo_qwen::chat_template::{
    ChatTemplate, OFFICIAL_TEMPLATE_SHA256, is_qwen38_named, render_and_tokenize, request_options,
    request_tools,
};
use gufo_qwen::vision::{Encoder, ImageCache, ImageContext, count_request_tokens, prepare_request};

use crate::engine::{
    AdvanceRequest, DecodeRequest, Model, ModelOptions, SNAPSHOT_PAYLOAD_VERSION, Session,
    SessionSnapshot, Snapshotter,
};
use crate::executor::SessionMode;
use crate::rope_scaling::{resolve_rope_scaling, rope_position_policy};
use crate::weights::GgufFiles;

/// `kQwenFlashNextStateAbi`.
const STATE_ABI: &str = "qwen38-flash-next-rocm-session-v1";

static NEXT_RUNNER: AtomicU64 = AtomicU64::new(1);

pub struct RunnerOptions {
    pub model_path: PathBuf,
    pub max_context: u32,
    /// Persistent snapshots for a disk cache: fingerprints the artifacts.
    pub persistence: bool,
    pub mtp: Option<MtpOptions>,
    /// Serving sessions, the cost model's fixed concurrency (1..=8).
    pub sessions: usize,
    /// `--mmproj`: the vision sidecar; without it the canonical one next to
    /// the model serves images when present.
    pub vision_model_path: Option<PathBuf>,
    /// `--enable-yarn`: static YaRN, required for (and only accepted with) a
    /// `max_context` above the native one.
    pub enable_yarn: bool,
}

/// The native context of the Flash-Next artifact at `path`, read from its
/// header before any weights load; 0 when unknown.
pub fn native_context(path: &Path) -> u32 {
    let Ok((header, len)) = read_header(path) else {
        return 0;
    };
    Gguf::parse_prefix(&header, len)
        .ok()
        .and_then(|gguf| gguf.u32("qwen4exp.context_length"))
        .unwrap_or(0)
}

/// MTP speculation (`TextSpeculativeConfig` with the MTP backend).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpOptions {
    pub model_path: PathBuf,
    pub max_draft_tokens: u32,
}

/// Whether the model at `path` is a Flash-Next (`qwen4exp`) artifact; its
/// chat template is checked when the runner loads.
pub fn supports(path: &Path) -> bool {
    let Ok((header, len)) = read_header(path) else {
        return false;
    };
    Gguf::parse_prefix(&header, len)
        .is_ok_and(|gguf| gguf.string("general.architecture") == Some(b"qwen4exp"))
}

/// `QwenFlashNextCompatibilityIdentity`: what a persisted continuation
/// depends on, byte for byte as the C++ runner writes it.
fn compatibility_identity(
    artifact_fingerprint: &str,
    mtp: Option<(&str, u32)>,
    max_context: u32,
    decode_concurrency: u32,
    position_policy: &str,
) -> Vec<u8> {
    let scheme = gguf_identity::SCHEME;
    let mut identity = format!(
        "schema=gufo-text-continuation-v2\n\
         model_kind=qwen38-flash-next\n\
         artifact_id={scheme}:{artifact_fingerprint}\n\
         tokenizer=embedded-in-artifact\n\
         chat_template=qwen38-reasoning-compiled-v3\n\
         chat_template_reference_sha256={OFFICIAL_TEMPLATE_SHA256}\n\
         state_abi={STATE_ABI}\n\
         payload_layout=qfn-rocm-session-snapshot-v{SNAPSHOT_PAYLOAD_VERSION}\n\
         context_tokens={max_context}\n\
         position_policy={position_policy}\n\
         adapters=none\n"
    );
    if let Some((fingerprint, max_draft_tokens)) = mtp {
        identity.push_str(&format!(
            "draft_backend=qfn-mtp-v1\n\
             draft_artifact_id={scheme}:{fingerprint}\n\
             draft_max_tokens={max_draft_tokens}\n\
             draft_cost_concurrency={decode_concurrency}\n"
        ));
    }
    identity.into_bytes()
}

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

pub struct FlashNextRunner {
    id: u64,
    /// Tells a lost GPU context from an ordinary failure.
    device_probe: DeviceProbe,
    exec: Mutex<Model>,
    /// Captures without the model lock, on their own stream.
    snapshots: Snapshotter,
    tokenizer: QwenTokenizer,
    /// Hashes the sidecar for image prompts on HTTP threads.
    vision: Option<Arc<Encoder>>,
    /// Decoded and resized images of earlier requests.
    image_cache: ImageCache,
    name: String,
    max_context: u32,
    use_mtp: bool,
    prefill_capacity: usize,
    /// A remainder up to this long finishes in one pass.
    prefill_through_capacity: usize,
    resident_bytes: usize,
    state_bytes: usize,
    deferred_bytes: usize,
    persistence: Option<TextRunnerPersistenceDescriptor>,
}

/// A request's session and its checkpoint (`QwenFlashNextTextRunnerState`).
pub struct FlashNextState {
    session: Session,
    position: usize,
    /// `Invalidate` resets the device state before the next use.
    reset_pending: bool,
    /// The pending reset keeps the MTP scratch and graphs: the session was
    /// healthy when the cache reassigned it.
    reuse_mtp: bool,
    /// False for an `ignore_eos` request.
    stop_at_eos: bool,
}

/// A session snapshot, borrowed or in host bytes
/// (`QwenFlashNextTextRunnerSnapshot`).
pub struct FlashNextSnapshot {
    runner: u64,
    snapshot: SessionSnapshot,
    position: usize,
}

impl TextRunnerSnapshot for FlashNextSnapshot {
    fn payload_bytes(&self) -> usize {
        self.snapshot.len()
    }

    fn source(&self) -> Option<u64> {
        self.snapshot.source()
    }
}

impl text::TextRunnerState for FlashNextState {
    fn invalidate(&mut self) {
        self.session.set_cancellation_check(None);
        // A healthy slot is reassigned without rebuilding its MTP graphs
        // (`ContinuationAdapter::Invalidate`).
        if !self.reset_pending {
            self.reuse_mtp = self.session.is_valid();
        }
        self.reset_pending = true;
        self.position = 0;
    }

    fn set_cancellation_check(&mut self, check: Option<CancellationCheck>) {
        self.session.set_cancellation_check(check);
    }

    fn set_stop_at_eos(&mut self, stop_at_eos: bool) -> Result<()> {
        self.stop_at_eos = stop_at_eos;
        Ok(())
    }

    fn measured_resources(&self) -> TextRunnerMeasuredResources {
        TextRunnerMeasuredResources {
            per_request_state_bytes: Some(self.session.allocated_bytes()),
            temporary_scratch_bytes: Some(0),
        }
    }

    fn snapshot_owner(&self) -> Option<u64> {
        self.session.snapshot_owner()
    }
}

impl FlashNextRunner {
    /// The vision encoder, when the model serves images.
    pub fn vision_encoder(&self) -> Option<&Encoder> {
        self.vision.as_deref()
    }

    pub fn load(options: &RunnerOptions) -> Result<Self> {
        if options.sessions == 0 {
            return Err(Error::invalid("HTTP session count must be at least one"));
        }
        {
            let (header, len) = read_header(&options.model_path).map_err(engine)?;
            let gguf = Gguf::parse_prefix(&header, len).map_err(|e| engine(e.to_string()))?;
            ChatTemplate::validate_gguf(&gguf, is_qwen38_named(&gguf)).map_err(|e| {
                Error::invalid(format!("Unsupported Qwen3.8-Flash-Next chat template: {e}"))
            })?;
        }
        if let Some(mtp) = &options.mtp
            && mtp.max_draft_tokens == 0
        {
            return Err(Error::invalid(
                "Flash-Next MTP requires a positive draft limit and --min-draft-tokens 1",
            ));
        }
        let decode_concurrency = options.sessions.clamp(1, 8) as u32;
        // Static YaRN changes every request, so a context above the native
        // one needs --enable-yarn; checked before any weights load.
        let native = native_context(&options.model_path);
        let rope_scaling = resolve_rope_scaling(options.max_context, native, options.enable_yarn)
            .map_err(Error::invalid)?;
        let model = Model::load(
            &options.model_path,
            ModelOptions {
                rope_scaling,
                max_context: options.max_context,
                mtp_model_path: options.mtp.as_ref().map(|m| m.model_path.clone()),
                max_draft_tokens: options
                    .mtp
                    .as_ref()
                    .map_or(crate::policy::MAX_DRAFT_TOKENS, |m| m.max_draft_tokens),
                decode_concurrency,
                vision_model_path: options.vision_model_path.clone(),
            },
        )
        .map_err(|e| engine(format!("Failed to create Qwen3.8-Flash-Next model: {e}")))?;
        let use_mtp = options.mtp.is_some();
        let max_draft_tokens = options.mtp.as_ref().map_or(0, |m| m.max_draft_tokens);
        let persistence = if options.persistence {
            // A split artifact is identified by all its shards, in order.
            let fingerprint = |label: &str, path: &Path| {
                GgufFiles::open(path)
                    .and_then(|files| gguf_identity::identity_hex(&files.paths))
                    .map_err(|e| engine(format!("Failed to fingerprint {label} GGUF: {e}")))
            };
            let artifact = fingerprint("Qwen3.8-Flash-Next", &options.model_path)?;
            let draft = match &options.mtp {
                Some(mtp) => Some(fingerprint("Qwen3.8-Flash-Next MTP", &mtp.model_path)?),
                None => None,
            };
            Some(TextRunnerPersistenceDescriptor {
                compatibility_identity: compatibility_identity(
                    &artifact,
                    draft.as_deref().map(|f| (f, max_draft_tokens)),
                    options.max_context,
                    decode_concurrency,
                    &rope_position_policy(&model.config().rope_scaling),
                ),
                payload_version: SNAPSHOT_PAYLOAD_VERSION,
            })
        } else {
            None
        };
        let mode = if use_mtp {
            SessionMode::Speculative
        } else {
            SessionMode::Autoregressive
        };
        Ok(Self {
            id: NEXT_RUNNER.fetch_add(1, Ordering::Relaxed),
            device_probe: DeviceProbe::new().map_err(|e| engine(format!("device probe: {e}")))?,
            snapshots: model.snapshotter().map_err(engine)?,
            tokenizer: model.tokenizer().clone(),
            vision: model.vision_encoder().cloned(),
            image_cache: ImageCache::default(),
            name: model.name().to_string(),
            max_context: options.max_context,
            use_mtp,
            prefill_capacity: model.prefill_capacity() as usize,
            prefill_through_capacity: model.prefill_through_capacity() as usize,
            resident_bytes: model.resident_bytes(),
            state_bytes: model.session_bytes(mode, options.max_context),
            deferred_bytes: model.deferred_scratch_bytes(),
            persistence,
            exec: Mutex::new(model),
        })
    }

    fn exec(&self) -> MutexGuard<'_, Model> {
        self.exec.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn piece(&self, token: i32) -> Vec<u8> {
        self.tokenizer.decode(&[token as Token])
    }

    fn is_stop(&self, token: i32) -> bool {
        let stops = self.tokenizer.stop_tokens();
        u32::try_from(token).is_ok_and(|token| stops.contains(token))
    }

    /// Resets an invalidated state before it runs again.
    fn ready(&self, model: &mut Model, state: &mut FlashNextState) {
        if std::mem::take(&mut state.reset_pending) {
            if std::mem::take(&mut state.reuse_mtp) {
                model.reset_reusing_mtp(&mut state.session);
            } else {
                model.reset(&mut state.session);
            }
        }
    }

    fn check_snapshot<'a>(&self, snapshot: &'a FlashNextSnapshot) -> Result<&'a SessionSnapshot> {
        if snapshot.runner != self.id {
            return Err(Error::invalid(
                "Qwen3.8-Flash-Next snapshot does not belong to this model",
            ));
        }
        Ok(&snapshot.snapshot)
    }
}

impl TextModelRunner for FlashNextRunner {
    type State = FlashNextState;
    type Snapshot = FlashNextSnapshot;
    type Sampler = SamplerState;
    type PromptContext = ImageContext;

    fn descriptor(&self) -> TextRunnerDescriptor {
        TextRunnerDescriptor {
            model_id: self.name.clone(),
            state_abi: STATE_ABI.into(),
            max_context: self.max_context,
            capabilities: TextRunnerCapabilities {
                incremental_prefill: true,
                snapshot: true,
                fork: true,
                final_token_advance_required: false,
                incremental_text_is_exact: true,
                multi_token_decode: self.use_mtp,
                batched_multi_token_decode: self.use_mtp,
                batched_multi_token_decode_max_width: if self.use_mtp { 8 } else { 0 },
                prefix_reuse: true,
                in_pass_checkpoint: true,
            },
            persistence: self.persistence.clone(),
        }
    }

    fn resource_claim(&self) -> TextRunnerResourceClaim {
        let capacity = gufo_hip::device_memory()
            .ok()
            .map(|(free, _)| free.saturating_sub(self.deferred_bytes));
        // Snapshots live in host memory, not in the device state pool.
        TextRunnerResourceClaim {
            resident_weights_bytes: Some(self.resident_bytes),
            state_capacity_bytes: capacity,
            per_request_state_bytes: Some(self.state_bytes),
            temporary_scratch_bytes: Some(0),
            retained_snapshot_capacity_bytes: Some(text::host_snapshot_budget_bytes()),
            retained_snapshot_ceiling_bytes: Some(text::host_snapshot_ceiling_bytes()),
            requires_device_runtime_lock: true,
        }
    }

    fn supported_plans(&self) -> Vec<TextExecutionPlan> {
        std::iter::once(TextExecutionPlan::SERIAL)
            .chain((2..=8).map(TextExecutionPlan::batched))
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
        state: &mut FlashNextState,
        context: Option<&Arc<ImageContext>>,
    ) -> Result<()> {
        let mut model = self.exec();
        self.ready(&mut model, state);
        model
            .configure_vision(&mut state.session, context.map(|c| Arc::clone(&c.prompt)))
            .map_err(Error::invalid)
    }

    fn supports_images(&self) -> bool {
        self.vision.is_some()
    }

    fn poll_device(&self) -> text::DeviceStatus {
        device_status(self.device_probe.poll())
    }

    fn tool_format(&self) -> ToolFormat {
        ToolFormat::Qwen
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

    fn create_state(&self) -> Result<FlashNextState> {
        let mode = if self.use_mtp {
            SessionMode::Speculative
        } else {
            SessionMode::Autoregressive
        };
        let session = self
            .exec()
            .create_session(mode, self.max_context)
            .map_err(|e| engine(format!("Failed to create Qwen3.8-Flash-Next session: {e}")))?;
        Ok(FlashNextState {
            session,
            position: 0,
            reset_pending: false,
            reuse_mtp: false,
            stop_at_eos: true,
        })
    }

    fn create_sampler(&self, config: &SamplingConfig, prompt: &[Token]) -> Result<SamplerState> {
        SamplerState::new(config.clone(), prompt)
    }

    fn preview_first_token(
        &self,
        state: &mut FlashNextState,
        sampler: &mut SamplerState,
    ) -> Result<Option<TextDecodeSelection>> {
        self.select_next(state, sampler).map(Some)
    }

    fn prepare_prefix_reuse(&self, state: &mut FlashNextState, prefix: &[Token]) -> Result<()> {
        if state.position != prefix.len() {
            return Err(Error::logic(
                "Qwen3.8-Flash-Next reused prefix does not match checkpoint",
            ));
        }
        state.session.reset_draft_policy();
        Ok(())
    }

    fn prefill(
        &self,
        state: &mut FlashNextState,
        prompt: &[Token],
        offset: usize,
        max_input_tokens: usize,
    ) -> Result<TextPrefillStep> {
        if offset != state.position {
            return Err(Error::logic(
                "Qwen3.8-Flash-Next prefill offset does not match retained state",
            ));
        }
        if offset >= prompt.len() {
            return Err(Error::logic(
                "Qwen3.8-Flash-Next prefill has no remaining input",
            ));
        }
        let remaining = prompt.len() - offset;
        let consumed =
            max_input_tokens
                .min(remaining)
                .min(if remaining <= self.prefill_through_capacity {
                    self.prefill_through_capacity
                } else {
                    self.prefill_capacity
                });
        let next = offset + consumed;
        let prefix = engine_tokens(&prompt[..next])?;
        // The next step most likely takes the same budget: its n-gram rows
        // are read during this one and used only if it does.
        let after = prompt.len() - next;
        let following =
            max_input_tokens
                .min(after)
                .min(if after <= self.prefill_through_capacity {
                    self.prefill_through_capacity
                } else {
                    self.prefill_capacity
                });
        let following = engine_tokens(&prompt[next..next + following])?;
        let mut model = self.exec();
        self.ready(&mut model, state);
        if let Err(e) = model.sync_before(&mut state.session, &prefix, &following) {
            state.position = 0;
            return Err(engine(format!("Qwen3.8-Flash-Next prefill failed: {e}")));
        }
        state.position = next;
        Ok(TextPrefillStep {
            consumed_tokens: consumed,
            decode_ready: next == prompt.len(),
        })
    }

    fn prefill_checkpoint_bytes(
        &self,
        state: &FlashNextState,
        prompt: &[Token],
        offset: usize,
        max_input_tokens: usize,
        boundary: usize,
    ) -> Result<Option<usize>> {
        let consumed = max_input_tokens
            .min(prompt.len().saturating_sub(offset))
            .min(self.prefill_through_capacity);
        // The pass must reach past the boundary, which lies in its text
        // tail behind at least two of its rows.
        if boundary <= offset + 1
            || boundary >= offset + consumed
            || offset + consumed - boundary > 8
        {
            return Ok(None);
        }
        Ok(
            match self
                .snapshots
                .checkpoint_bytes(&state.session, boundary as u32)
            {
                0 => None,
                bytes => usize::try_from(bytes).ok(),
            },
        )
    }

    fn prefill_through(
        &self,
        state: &mut FlashNextState,
        prompt: &[Token],
        offset: usize,
        max_input_tokens: usize,
        boundary: usize,
    ) -> Result<TextPrefillThrough<FlashNextSnapshot>> {
        if offset != state.position {
            return Err(Error::logic(
                "in-pass checkpoint offset does not match state",
            ));
        }
        let consumed = max_input_tokens
            .min(prompt.len().saturating_sub(offset))
            .min(self.prefill_through_capacity);
        let next = offset + consumed;
        let prefix = engine_tokens(&prompt[..next])?;
        let mut model = self.exec();
        self.ready(&mut model, state);
        let (snapshot, checkpoint_ms) =
            match model.sync_through(&mut state.session, &prefix, boundary) {
                Ok(captured) => captured,
                Err(e) => {
                    state.position = 0;
                    return Err(engine(format!("in-pass checkpoint failed: {e}")));
                }
            };
        state.position = next;
        Ok(TextPrefillThrough {
            step: TextPrefillStep {
                consumed_tokens: consumed,
                decode_ready: next == prompt.len(),
            },
            checkpoint: FlashNextSnapshot {
                runner: self.id,
                snapshot,
                position: boundary,
            },
            checkpoint_ms,
        })
    }

    fn select_next(
        &self,
        state: &mut FlashNextState,
        sampler: &mut SamplerState,
    ) -> Result<TextDecodeSelection> {
        if state.position >= self.max_context as usize {
            return Ok(TextDecodeSelection {
                stop: true,
                ..TextDecodeSelection::default()
            });
        }
        let logits = state.session.logits();
        if logits.is_empty() {
            return Err(engine(
                "Qwen3.8-Flash-Next token selection has no logits".into(),
            ));
        }
        let token = sampler.sample(logits)? as i32;
        if state.stop_at_eos && self.is_stop(token) {
            return Ok(TextDecodeSelection {
                stop: true,
                ..TextDecodeSelection::default()
            });
        }
        Ok(TextDecodeSelection {
            stop: false,
            token: token as Token,
            piece: self.piece(token),
        })
    }

    fn advance(&self, state: &mut FlashNextState, token: Token) -> Result<()> {
        let token = engine_token(token)?;
        let mut model = self.exec();
        self.ready(&mut model, state);
        model
            .evaluate(&mut state.session, token)
            .map_err(|e| engine(format!("Qwen3.8-Flash-Next decode failed: {e}")))?;
        state.position += 1;
        Ok(())
    }

    fn decode_step(
        &self,
        state: &mut FlashNextState,
        max_tokens: usize,
        sampler: &mut SamplerState,
    ) -> Result<TextDecodeStep> {
        if !self.use_mtp || max_tokens == 1 {
            return text::default_decode_step(self, state, max_tokens, sampler);
        }
        if max_tokens == 0 {
            return Err(Error::invalid(
                "Qwen3.8-Flash-Next MTP decode budget must be at least one token",
            ));
        }
        if state.position >= self.max_context as usize {
            return Ok(TextDecodeStep {
                stop: true,
                ..TextDecodeStep::default()
            });
        }
        let before = state.session.statistics();
        let mut working = sampler.clone();
        // The engine bounds the physical block and stops at EOS; the whole
        // remaining budget lets its policy recognize a short tail.
        let decoded = {
            let mut model = self.exec();
            self.ready(&mut model, state);
            model
                .decode_step(
                    &mut state.session,
                    max_tokens,
                    &mut working,
                    state.stop_at_eos,
                )
                .map_err(|e| engine(format!("Qwen3.8-Flash-Next MTP decode failed: {e}")))?
        };
        // The pool accepts the returned tokens once; publish the RNG and a
        // residual draw so the next step keeps the rejection-conditioned
        // distribution.
        sampler.copy_draw_state_from(&working);
        state.position = state.session.position() as usize;
        let after = state.session.statistics();
        Ok(TextDecodeStep {
            selections: decoded
                .tokens
                .iter()
                .map(|&t| TextDecodeSelection {
                    stop: false,
                    token: t as Token,
                    piece: self.piece(t),
                })
                .collect(),
            draft_rounds: (after.cycles - before.cycles) as usize,
            draft_tokens: (after.drafted - before.drafted) as usize,
            draft_accepted_tokens: (after.accepted - before.accepted) as usize,
            lookup_tokens: (after.lookup - before.lookup) as usize,
            lookup_accepted_tokens: (after.lookup_accepted - before.lookup_accepted) as usize,
            stop: decoded.stop,
            execution_plan: TextExecutionPlan::SERIAL,
        })
    }

    fn decode_batch(
        &self,
        decodes: &mut [TextRunnerDecode<'_, FlashNextState, SamplerState>],
    ) -> Result<Vec<Result<TextDecodeStep>>> {
        if decodes.len() < 2 || !self.use_mtp {
            return Ok(text::default_decode_batch(self, decodes));
        }
        let mut samplers: Vec<SamplerState> = decodes.iter().map(|d| d.sampler.clone()).collect();
        let before: Vec<_> = decodes
            .iter()
            .map(|d| d.state.session.statistics())
            .collect();
        let results = {
            let mut model = self.exec();
            for d in decodes.iter_mut() {
                self.ready(&mut model, d.state);
            }
            let mut requests: Vec<DecodeRequest<'_>> = decodes
                .iter_mut()
                .zip(samplers.iter_mut())
                .map(|(d, sampler)| DecodeRequest {
                    session: &mut d.state.session,
                    max_tokens: d.max_tokens,
                    sampler,
                    stop_at_eos: d.state.stop_at_eos,
                })
                .collect();
            model.decode_batch(&mut requests)
        };
        let active = results
            .iter()
            .filter(|r| r.as_ref().is_ok_and(|d| !d.tokens.is_empty()))
            .count();
        Ok(decodes
            .iter_mut()
            .zip(results)
            .zip(samplers.iter().zip(before))
            .map(|((d, result), (sampler, before))| {
                let decoded = result.map_err(|e| engine(format!("Flash-Next MTP failed: {e}")))?;
                d.sampler.copy_draw_state_from(sampler);
                d.state.position = d.state.session.position() as usize;
                let after = d.state.session.statistics();
                Ok(TextDecodeStep {
                    execution_plan: if active > 1 && !decoded.tokens.is_empty() {
                        TextExecutionPlan::batched(active)
                    } else {
                        TextExecutionPlan::SERIAL
                    },
                    selections: decoded
                        .tokens
                        .iter()
                        .map(|&t| TextDecodeSelection {
                            stop: false,
                            token: t as Token,
                            piece: self.piece(t),
                        })
                        .collect(),
                    draft_rounds: (after.cycles - before.cycles) as usize,
                    draft_tokens: (after.drafted - before.drafted) as usize,
                    draft_accepted_tokens: (after.accepted - before.accepted) as usize,
                    lookup_tokens: (after.lookup - before.lookup) as usize,
                    lookup_accepted_tokens: (after.lookup_accepted - before.lookup_accepted)
                        as usize,
                    stop: decoded.stop,
                })
            })
            .collect())
    }

    fn advance_batch(
        &self,
        advances: &mut [TextRunnerAdvance<'_, FlashNextState>],
    ) -> Result<Vec<Result<()>>> {
        if advances.len() < 2 {
            return Ok(text::default_advance_batch(self, advances));
        }
        let tokens = advances
            .iter()
            .map(|a| engine_token(a.token))
            .collect::<Result<Vec<_>>>()?;
        let results = {
            let mut model = self.exec();
            for a in advances.iter_mut() {
                self.ready(&mut model, a.state);
            }
            let mut requests: Vec<AdvanceRequest<'_>> = advances
                .iter_mut()
                .zip(&tokens)
                .map(|(a, &token)| AdvanceRequest {
                    session: &mut a.state.session,
                    token,
                })
                .collect();
            model.evaluate_batch(&mut requests)
        };
        Ok(advances
            .iter_mut()
            .zip(results)
            .map(|(a, result)| {
                result.map_err(|e| engine(format!("Flash-Next advance failed: {e}")))?;
                a.state.position = a.state.session.position() as usize;
                Ok(())
            })
            .collect())
    }

    fn checkpoint_position(&self, state: &FlashNextState) -> Result<usize> {
        Ok(state.position)
    }

    fn snapshot_payload_bytes(&self, state: &FlashNextState) -> Result<usize> {
        match self.snapshots.bytes(&state.session) {
            0 => Err(Error::runtime(
                "Qwen3.8-Flash-Next snapshot size is unavailable",
            )),
            bytes => usize::try_from(bytes)
                .map_err(|_| Error::runtime("Qwen3.8-Flash-Next snapshot size is unavailable")),
        }
    }

    fn snapshot(&self, state: &FlashNextState) -> Result<FlashNextSnapshot> {
        let snapshot = self
            .snapshots
            .capture(&state.session)
            .map_err(|e| engine(format!("Qwen3.8-Flash-Next snapshot failed: {e}")))?;
        Ok(FlashNextSnapshot {
            runner: self.id,
            snapshot,
            position: state.position,
        })
    }

    /// Host bytes from the start: a disk export then adds no device copy
    /// while the session resumes (`SnapshotForPersistence`).
    fn snapshot_for_persistence(&self, state: &FlashNextState) -> Result<FlashNextSnapshot> {
        let snapshot = self
            .snapshots
            .capture_host(&state.session)
            .map_err(|e| engine(format!("Qwen3.8-Flash-Next snapshot failed: {e}")))?;
        Ok(FlashNextSnapshot {
            runner: self.id,
            snapshot,
            position: state.position,
        })
    }

    fn restore_or_fork(
        &self,
        state: &mut FlashNextState,
        snapshot: &FlashNextSnapshot,
    ) -> Result<()> {
        let session_snapshot = self.check_snapshot(snapshot)?;
        let mut model = self.exec();
        self.ready(&mut model, state);
        if let Err(e) = model.restore(&mut state.session, session_snapshot) {
            text::TextRunnerState::invalidate(state);
            return Err(engine(format!(
                "Qwen3.8-Flash-Next snapshot restore failed: {e}"
            )));
        }
        state.position = snapshot.position;
        Ok(())
    }

    fn persistent_snapshot_payload_bytes(&self, snapshot: &FlashNextSnapshot) -> Result<usize> {
        Ok(self.check_snapshot(snapshot)?.len())
    }

    fn serialize_persistent_snapshot(
        &self,
        snapshot: &FlashNextSnapshot,
        destination: &mut [u8],
    ) -> Result<usize> {
        let payload = self.check_snapshot(snapshot)?;
        if destination.len() != payload.len() {
            return Err(Error::invalid(
                "Qwen3.8-Flash-Next persistent snapshot serialization failed",
            ));
        }
        payload
            .copy_to(destination)
            .map_err(|e| engine(format!("Qwen3.8-Flash-Next snapshot materialize: {e}")))?;
        Ok(destination.len())
    }

    fn stream_persistent_snapshot(
        &self,
        snapshot: &FlashNextSnapshot,
        sink: &mut dyn FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        let payload = self
            .check_snapshot(snapshot)?
            .to_bytes()
            .map_err(|e| engine(format!("Qwen3.8-Flash-Next snapshot materialize: {e}")))?;
        sink(&payload)
    }

    fn restore_persistent_snapshot(
        &self,
        state: &mut FlashNextState,
        payload: &[u8],
    ) -> Result<()> {
        let mut model = self.exec();
        self.ready(&mut model, state);
        if let Err(e) = model.restore_snapshot(&mut state.session, payload) {
            text::TextRunnerState::invalidate(state);
            return Err(engine(format!(
                "Qwen3.8-Flash-Next persistent snapshot restore failed: {e}"
            )));
        }
        let position = state.session.position();
        if position == 0 || position > self.max_context {
            text::TextRunnerState::invalidate(state);
            return Err(engine(
                "Qwen3.8-Flash-Next persistent snapshot restored an invalid position".into(),
            ));
        }
        state.position = position as usize;
        Ok(())
    }
}

fn engine_token(token: Token) -> Result<i32> {
    i32::try_from(token)
        .map_err(|_| Error::invalid("Qwen3.8-Flash-Next token ID exceeds engine range"))
}

fn engine_tokens(tokens: &[Token]) -> Result<Vec<i32>> {
    tokens.iter().map(|&t| engine_token(t)).collect()
}
