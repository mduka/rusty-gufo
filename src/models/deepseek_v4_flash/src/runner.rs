//! The serving runner of DeepSeek V4 Flash, a port of `DeepSeekTextRunner`
//! (upstream `src/cli/serve/inference_backend.cpp`): one engine session
//! per request, prefill through checkpoint sync, DSpark speculative cycles
//! (batched across requests), and host snapshots that restore in memory and
//! persist to disk.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use gufo_core::gguf::{Gguf, read_header};
use gufo_core::gguf_identity;
use gufo_core::json;
use gufo_core::json_constraint::{ConstraintVocabulary, Piece, ToolFormat};
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_core::text::{
    self, CancellationCheck, ChatRequest, ChatRole, Error, InitialOutputState, NoPromptContext,
    Result, TextDecodeSelection, TextDecodeStep, TextExecutionPlan, TextModelRunner,
    TextPrefillStep, TextPreparedPrompt, TextRunnerAdvance, TextRunnerCapabilities,
    TextRunnerDecode, TextRunnerDescriptor, TextRunnerMeasuredResources,
    TextRunnerPersistenceDescriptor, TextRunnerResourceClaim, TextRunnerSnapshot, Token,
    ToolChoice,
};
use gufo_deepseek_v4_flash_kernels::SnapshotReader;
use gufo_hip::{DeviceProbe, DeviceProbeStatus};

use crate::chat_template::{
    self, CHAT_TEMPLATE_ID, ChatMessage, ChatTool, ENCODER_REFERENCE_SHA256, ToolArgument, ToolCall,
};
use crate::dspark_sampler::DsparkSampler;
use crate::engine::Engine;
use crate::graph::PAYLOAD_VERSION;
use crate::session::{DsparkItem, Session};
use crate::shape::MODEL_SHAPE_NAME;
use crate::tokenizer::Vocab;

/// `kDeepSeekStateAbi`.
const STATE_ABI: &str = "deepseek-v4-flash-gfx1151-state-v4";

static NEXT_RUNNER: AtomicU64 = AtomicU64::new(1);

pub struct RunnerOptions {
    pub model_path: PathBuf,
    pub max_context: u32,
    /// Persistent snapshots for a disk cache: fingerprints the artifacts.
    pub persistence: bool,
    /// The DSpark support model; `None` decodes autoregressively.
    pub dspark_model_path: Option<PathBuf>,
    /// At most this many support tokens per DSpark cycle; also part of the
    /// persistence identity without DSpark, as in the C++ runner.
    pub max_draft_tokens: u32,
}

/// Whether the model at `path` is a DeepSeek V4 Flash (`deepseek4`)
/// artifact; its chat template is checked when the runner loads.
pub fn supports(path: &Path) -> bool {
    let Ok((header, len)) = read_header(path) else {
        return false;
    };
    Gguf::parse_prefix(&header, len)
        .is_ok_and(|gguf| gguf.string("general.architecture") == Some(b"deepseek4"))
}

/// `DeepSeekCompatibilityIdentity`: what a persisted continuation depends
/// on, byte for byte as the C++ runner writes it.
fn compatibility_identity(
    artifact: &str,
    support: &str,
    max_context: u32,
    max_draft_tokens: u32,
) -> Vec<u8> {
    let scheme = gguf_identity::SCHEME;
    format!(
        "schema=gufo-text-continuation-v2\n\
         model_kind=deepseek4\n\
         artifact_id={scheme}:{artifact}\n\
         tokenizer=joyai-byte-bpe-v1\n\
         chat_template={CHAT_TEMPLATE_ID}\n\
         chat_template_reference_sha256={ENCODER_REFERENCE_SHA256}\n\
         state_abi={STATE_ABI}\n\
         payload_layout=ds4-rocm-v4-window128-continuation\n\
         numerics=ds4-scalar-verifier-v1\n\
         context_tokens={max_context}\n\
         position_policy=absolute-v1\n\
         rope_window_policy=deepseek4-compiled-v1\n\
         adapters=none\n\
         support_id={scheme}:{support}\n\
         max_draft_tokens={max_draft_tokens}\n\
         draft_policy=dspark-cost-v5-window128\n"
    )
    .into_bytes()
}

fn engine_error(message: String) -> Error {
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

/// The template role of a request message (`ChatRoleName`): developer
/// messages keep their own role.
fn role_name(role: ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::Developer => "developer",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
        ChatRole::Tool => "tool",
    }
}

pub struct DeepSeekRunner {
    id: u64,
    exec: Mutex<Engine>,
    /// Reads frozen sessions for snapshots without the engine lock.
    snapshots: SnapshotReader,
    vocab: Arc<Vocab>,
    max_context: u32,
    use_dspark: bool,
    max_draft_tokens: u32,
    persistence: Option<TextRunnerPersistenceDescriptor>,
    /// Tells a lost GPU context from an ordinary failure.
    device_probe: DeviceProbe,
}

/// A request's session and its checkpoint (`DeepSeekTextRunnerState`).
pub struct DeepSeekState {
    session: Session,
    position: usize,
    /// False for an `ignore_eos` request.
    stop_at_eos: bool,
}

/// A host payload of `SaveSnapshot` (`DeepSeekTextRunnerSnapshot`).
pub struct DeepSeekSnapshot {
    runner: u64,
    bytes: Vec<u8>,
    position: usize,
}

impl TextRunnerSnapshot for DeepSeekSnapshot {
    fn payload_bytes(&self) -> usize {
        self.bytes.len()
    }
}

impl text::TextRunnerState for DeepSeekState {
    fn invalidate(&mut self) {
        self.session.set_cancellation(None);
        self.session.invalidate();
        self.position = 0;
    }

    fn set_cancellation_check(&mut self, check: Option<CancellationCheck>) {
        self.session.set_cancellation(
            check.map(|check| Box::new(move || check()) as Box<dyn Fn() -> bool + Send>),
        );
    }

    fn set_stop_at_eos(&mut self, stop_at_eos: bool) -> Result<()> {
        self.stop_at_eos = stop_at_eos;
        Ok(())
    }

    fn measured_resources(&self) -> TextRunnerMeasuredResources {
        match usize::try_from(self.session.payload_bytes()) {
            Ok(bytes) if bytes != 0 => TextRunnerMeasuredResources {
                per_request_state_bytes: Some(bytes),
                temporary_scratch_bytes: None,
            },
            _ => TextRunnerMeasuredResources::default(),
        }
    }
}

impl DeepSeekRunner {
    /// The DeepSeek branch of `InferenceBackend::load`: the template check,
    /// the engine with its optional support model, and the artifact
    /// fingerprints a disk cache keys on.
    pub fn load(options: &RunnerOptions) -> Result<Self> {
        if options.max_context < 2 {
            return Err(Error::invalid(
                "DeepSeek V4 Flash context must be at least 2",
            ));
        }
        if options.max_context > i32::MAX as u32 {
            return Err(Error::invalid(
                "DeepSeek V4 Flash context exceeds engine limits",
            ));
        }
        let engine = Engine::open(&options.model_path, options.dspark_model_path.as_deref())
            .map_err(|_| {
                engine_error(
                    "Failed to create DeepSeek model: failed to load DeepSeek V4 Flash GGUF".into(),
                )
            })?;
        chat_template::validate_gguf_template(&engine.model)
            .map_err(|e| Error::invalid(format!("Unsupported DeepSeek chat template: {e}")))?;
        let use_dspark = options.dspark_model_path.is_some();
        let max_draft_tokens = options.max_draft_tokens.max(1);
        let persistence = if options.persistence {
            let fingerprint = |label: &str, path: &Path| {
                gguf_identity::identity_hex(&[path.to_path_buf()])
                    .map_err(|e| engine_error(format!("Failed to fingerprint {label} GGUF: {e}")))
            };
            let artifact = fingerprint("DeepSeek", &options.model_path)?;
            let support = match &options.dspark_model_path {
                Some(path) => fingerprint("DSpark", path)?,
                None => String::new(),
            };
            Some(TextRunnerPersistenceDescriptor {
                compatibility_identity: compatibility_identity(
                    &artifact,
                    &support,
                    options.max_context,
                    max_draft_tokens,
                ),
                payload_version: PAYLOAD_VERSION,
            })
        } else {
            None
        };
        Ok(Self {
            id: NEXT_RUNNER.fetch_add(1, Ordering::Relaxed),
            device_probe: DeviceProbe::new()
                .map_err(|e| engine_error(format!("device probe: {e}")))?,
            snapshots: engine.backend.snapshot_reader(),
            vocab: Arc::clone(&engine.vocab),
            max_context: options.max_context,
            use_dspark,
            max_draft_tokens,
            persistence,
            exec: Mutex::new(engine),
        })
    }

    fn exec(&self) -> MutexGuard<'_, Engine> {
        self.exec.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn piece(&self, token: i32) -> Vec<u8> {
        self.vocab.token_text(token)
    }

    fn is_stop(&self, token: i32) -> bool {
        self.vocab.is_stop(token)
    }

    fn check_snapshot<'a>(&self, snapshot: &'a DeepSeekSnapshot) -> Result<&'a [u8]> {
        if snapshot.runner != self.id {
            return Err(Error::invalid(
                "DeepSeek snapshot does not belong to this model",
            ));
        }
        Ok(&snapshot.bytes)
    }

    /// The emitted tokens of a DSpark cycle up to the first stop token.
    fn step(&self, emitted: &[i32], stop_at_eos: bool) -> TextDecodeStep {
        let mut step = TextDecodeStep::default();
        for &token in emitted {
            if stop_at_eos && self.is_stop(token) {
                step.stop = true;
                break;
            }
            step.selections.push(TextDecodeSelection {
                stop: false,
                token: token as Token,
                piece: self.piece(token),
            });
        }
        step
    }

    /// The conversation of a request (`RenderAndTokenize`).
    fn conversation(request: &ChatRequest) -> Result<(Vec<ChatMessage>, Vec<ChatTool>)> {
        let messages = request
            .messages
            .iter()
            .map(|message| ChatMessage {
                role: role_name(message.role).into(),
                // The template reads the server's instructions as content.
                content: format!("{}{}", message.content, message.framing_suffix),
                reasoning_content: message.thought.clone(),
                tool_call_id: message.tool_call_id.clone(),
                tool_calls: message
                    .tool_calls
                    .iter()
                    .map(|call| ToolCall {
                        name: call.name.clone(),
                        id: call.id.clone(),
                        arguments: call
                            .arguments
                            .iter()
                            .map(|a| ToolArgument {
                                name: a.name.clone(),
                                value: a.value.clone(),
                                is_string: a.is_string,
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect();
        let mut tools = Vec::new();
        if request.tool_choice != ToolChoice::None {
            for tool in &request.tools {
                let definition_json = if tool.definition_json.is_empty() {
                    String::new()
                } else {
                    let definition = json::parse(tool.definition_json.as_bytes())?;
                    definition
                        .find("function")
                        .cloned()
                        .unwrap_or_default()
                        .dump()?
                };
                tools.push(ChatTool {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters_json: tool.parameters_json.clone(),
                    definition_json,
                });
            }
        }
        Ok((messages, tools))
    }

    /// One DSpark cycle per request, the C++ `DecodeBatch` of a cohort.
    /// One DSpark step of `decodes` together; `None` when a request was
    /// cancelled, which the batch rejects before any session moves, so the
    /// others can step alone.
    fn try_dspark_cohort(
        &self,
        decodes: &mut [TextRunnerDecode<'_, DeepSeekState, SamplerState>],
    ) -> Result<Option<Vec<TextDecodeStep>>> {
        // Greedy and sampled requests share one cohort; requests without a
        // bridge keep the greedy verifier.
        let mut bridges: Vec<Option<DsparkSampler>> = decodes
            .iter()
            .map(|d| {
                (!d.sampler.config().can_use_unmodified_argmax())
                    .then(|| DsparkSampler::new(d.sampler))
            })
            .collect();
        let before: Vec<_> = decodes
            .iter()
            .map(|d| d.state.session.dspark_stats())
            .collect();
        let emitted = {
            let mut engine = self.exec();
            let mut items: Vec<DsparkItem<'_>> = decodes
                .iter_mut()
                .zip(bridges.iter_mut())
                .map(|(d, bridge)| DsparkItem {
                    stop_at_eos: d.state.stop_at_eos,
                    session: &mut d.state.session,
                    max_tokens: d.max_tokens,
                    max_draft_tokens: self.max_draft_tokens,
                    sampler: bridge.as_mut(),
                    emitted: Vec::new(),
                })
                .collect();
            match Session::dspark_step_batch(&mut engine, &mut items) {
                Ok(()) => {}
                Err(e) if e.interrupted => return Ok(None),
                Err(e) => {
                    return Err(engine_error(format!(
                        "DeepSeek DSpark batch decode failed: {e}"
                    )));
                }
            }
            items
                .into_iter()
                .map(|item| item.emitted)
                .collect::<Vec<_>>()
        };
        let width = decodes.len();
        decodes
            .iter_mut()
            .zip(emitted)
            .zip(before.into_iter().zip(bridges))
            .map(|((d, emitted), (before, bridge))| {
                if emitted.is_empty() {
                    return Err(engine_error(
                        "DeepSeek DSpark batch produced no tokens".into(),
                    ));
                }
                if let Some(bridge) = bridge {
                    d.sampler.set_rng_state(bridge.rng_state());
                }
                d.state.position = d.state.session.position();
                let mut step = self.step(&emitted, d.state.stop_at_eos);
                if width > 1 {
                    step.execution_plan = TextExecutionPlan::batched(width);
                }
                let after = d.state.session.dspark_stats();
                step.draft_rounds = (after.steps - before.steps) as usize;
                step.draft_tokens = (after.support_drafted - before.support_drafted) as usize;
                step.draft_accepted_tokens =
                    (after.support_accepted - before.support_accepted) as usize;
                Ok(step)
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }
}

impl TextModelRunner for DeepSeekRunner {
    type State = DeepSeekState;
    type Snapshot = DeepSeekSnapshot;
    type Sampler = SamplerState;
    type PromptContext = NoPromptContext;

    fn descriptor(&self) -> TextRunnerDescriptor {
        TextRunnerDescriptor {
            model_id: MODEL_SHAPE_NAME.into(),
            state_abi: STATE_ABI.into(),
            max_context: self.max_context,
            capabilities: TextRunnerCapabilities {
                incremental_prefill: true,
                snapshot: true,
                fork: true,
                final_token_advance_required: false,
                incremental_text_is_exact: true,
                multi_token_decode: self.use_dspark,
                batched_multi_token_decode: self.use_dspark,
                batched_multi_token_decode_max_width: 8,
                prefix_reuse: true,
                in_pass_checkpoint: false,
            },
            persistence: self.persistence.clone(),
        }
    }

    fn resource_claim(&self) -> TextRunnerResourceClaim {
        TextRunnerResourceClaim {
            resident_weights_bytes: None,
            state_capacity_bytes: gufo_hip::device_memory().ok().map(|(free, _)| free),
            per_request_state_bytes: None,
            temporary_scratch_bytes: None,
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
        let mut tokens = Vec::new();
        self.vocab.tokenize(text.as_bytes(), &mut tokens);
        runner_tokens(&tokens)
    }

    /// The template's special tokens by ID, such as `</think>`.
    fn tokenize_control(&self, text: &str) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        self.vocab
            .tokenize_rendered_chat(text.as_bytes(), &mut tokens);
        runner_tokens(&tokens)
    }

    fn render_and_tokenize(&self, request: &ChatRequest) -> Result<Option<Vec<Token>>> {
        let (messages, tools) = Self::conversation(request)?;
        let mut options = chat_template::resolve_chat_options(&request.reasoning);
        options.tools_present =
            !request.tools.is_empty() && request.tool_choice != ToolChoice::None;
        let rendered = chat_template::render_chat(&messages, &tools, &options);
        let mut tokens = Vec::new();
        self.vocab
            .tokenize_rendered_chat(rendered.as_bytes(), &mut tokens);
        if tokens.is_empty() {
            return Ok(None);
        }
        runner_tokens(&tokens).map(Some)
    }

    fn prepare_prompt(
        &self,
        request: &ChatRequest,
    ) -> Result<Option<TextPreparedPrompt<NoPromptContext>>> {
        if request.messages.iter().any(|m| !m.images.is_empty()) {
            return Err(Error::invalid("model does not support image input"));
        }
        let Some(tokens) = self.render_and_tokenize(request)? else {
            return Ok(None);
        };
        // DeepSeek joins adjacent user and tool messages into one user
        // block, so a new user turn after a discarded assistant can also
        // change the final token before the generation suffix: the stable
        // frontier comes from the renderer and tokenizer.
        let mut continuation = request.clone();
        continuation
            .messages
            .push(gufo_core::text::ChatMessage::new(ChatRole::User, ""));
        let cache_prefix_tokens = match self.render_and_tokenize(&continuation)? {
            Some(continued) => tokens
                .iter()
                .zip(&continued)
                .take_while(|(a, b)| a == b)
                .count(),
            None => 0,
        };
        Ok(Some(TextPreparedPrompt {
            tokens,
            context: None,
            cache_prefix_tokens,
        }))
    }

    fn poll_device(&self) -> text::DeviceStatus {
        device_status(self.device_probe.poll())
    }

    fn tool_format(&self) -> ToolFormat {
        ToolFormat::DeepSeek
    }

    fn initial_output_state(&self, request: &ChatRequest) -> InitialOutputState {
        if chat_template::resolve_chat_options(&request.reasoning).enable_thinking {
            InitialOutputState::Reasoning
        } else {
            InitialOutputState::Content
        }
    }

    fn build_constraint_vocabulary(&self) -> Result<ConstraintVocabulary> {
        ConstraintVocabulary::new(self.vocab.len() as u32, |id| {
            Ok(Piece {
                text: self.vocab.token_text(id as i32),
                stop: self.vocab.is_stop(id as i32),
            })
        })
    }

    fn decode(&self, tokens: &[Token]) -> Result<Vec<u8>> {
        let mut text = Vec::new();
        for &token in tokens {
            text.extend(self.vocab.token_text(engine_token(token)?));
        }
        Ok(text)
    }

    fn create_state(&self) -> Result<DeepSeekState> {
        let session = self
            .exec()
            .create_session(self.max_context, self.use_dspark)
            .map_err(|e| engine_error(format!("Failed to create DeepSeek session: {e}")))?;
        Ok(DeepSeekState {
            session,
            position: 0,
            stop_at_eos: true,
        })
    }

    fn create_sampler(&self, config: &SamplingConfig, prompt: &[Token]) -> Result<SamplerState> {
        SamplerState::new(config.clone(), prompt)
    }

    fn preview_first_token(
        &self,
        state: &mut DeepSeekState,
        sampler: &mut SamplerState,
    ) -> Result<Option<TextDecodeSelection>> {
        if state.position >= self.max_context as usize {
            return Ok(Some(TextDecodeSelection {
                stop: true,
                ..TextDecodeSelection::default()
            }));
        }
        let logits = state.session.logits().ok_or_else(|| {
            engine_error("DeepSeek token preview failed: session has no valid checkpoint".into())
        })?;
        let token = sampler.sample(logits)? as i32;
        Ok(Some(TextDecodeSelection {
            stop: state.stop_at_eos && self.is_stop(token),
            token: token as Token,
            piece: self.piece(token),
        }))
    }

    fn prepare_prefix_reuse(&self, state: &mut DeepSeekState, prefix: &[Token]) -> Result<()> {
        if state.position != prefix.len() {
            return Err(Error::logic(
                "DeepSeek reused prefix does not match checkpoint",
            ));
        }
        state.session.begin_request();
        Ok(())
    }

    fn prefill(
        &self,
        state: &mut DeepSeekState,
        prompt: &[Token],
        offset: usize,
        max_input_tokens: usize,
    ) -> Result<TextPrefillStep> {
        if offset != state.position {
            return Err(Error::logic(
                "DeepSeek prefill offset does not match retained state",
            ));
        }
        if offset >= prompt.len() {
            return Err(Error::logic("DeepSeek prefill has no remaining input"));
        }
        let consumed = max_input_tokens
            .min(prompt.len() - offset)
            .min(state.session.prefill_capacity() as usize);
        let next = offset + consumed;
        let prefix = engine_tokens(&prompt[..next])?;
        let mut engine = self.exec();
        state
            .session
            .sync(&mut engine, &prefix)
            .map_err(|e| engine_error(format!("DeepSeek prefill failed: {e}")))?;
        state.position = next;
        Ok(TextPrefillStep {
            consumed_tokens: consumed,
            decode_ready: next == prompt.len(),
        })
    }

    fn select_next(
        &self,
        state: &mut DeepSeekState,
        sampler: &mut SamplerState,
    ) -> Result<TextDecodeSelection> {
        if state.position >= self.max_context as usize {
            return Ok(TextDecodeSelection {
                stop: true,
                ..TextDecodeSelection::default()
            });
        }
        // A sampled DSpark cycle may have drawn the next token already.
        let mut token = state.session.take_pending_dspark_token();
        if token < 0 {
            let logits = state.session.logits().ok_or_else(|| {
                engine_error(
                    "DeepSeek token selection failed: session has no valid checkpoint".into(),
                )
            })?;
            token = sampler.sample(logits)? as i32;
        }
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

    fn advance(&self, state: &mut DeepSeekState, token: Token) -> Result<()> {
        let token = engine_token(token)?;
        let mut engine = self.exec();
        state
            .session
            .eval(&mut engine, token)
            .map_err(|e| engine_error(format!("DeepSeek decode failed: {e}")))?;
        state.position += 1;
        Ok(())
    }

    fn decode_step(
        &self,
        state: &mut DeepSeekState,
        max_tokens: usize,
        sampler: &mut SamplerState,
    ) -> Result<TextDecodeStep> {
        if !self.use_dspark {
            return text::default_decode_step(self, state, max_tokens, sampler);
        }
        if max_tokens == 0 {
            return Err(Error::invalid(
                "DeepSeek DSpark decode budget must be at least one token",
            ));
        }
        if state.position >= self.max_context as usize {
            return Ok(TextDecodeStep {
                stop: true,
                ..TextDecodeStep::default()
            });
        }
        let mut decodes = [TextRunnerDecode {
            state,
            max_tokens,
            sampler,
        }];
        let mut steps = self
            .try_dspark_cohort(&mut decodes)?
            .ok_or_else(|| engine_error("DeepSeek DSpark decode cancelled".into()))?;
        Ok(steps.remove(0))
    }

    fn decode_batch(
        &self,
        decodes: &mut [TextRunnerDecode<'_, DeepSeekState, SamplerState>],
    ) -> Result<Vec<Result<TextDecodeStep>>> {
        if !self.use_dspark || decodes.len() < 2 || decodes.len() > 8 {
            return Ok(text::default_decode_batch(self, decodes));
        }
        // Requests at the context limit stop; the others step together.
        let full: Vec<bool> = decodes
            .iter()
            .map(|d| d.state.position >= self.max_context as usize)
            .collect();
        let mut active: Vec<TextRunnerDecode<'_, DeepSeekState, SamplerState>> = decodes
            .iter_mut()
            .zip(&full)
            .filter(|(_, full)| !**full)
            .map(|(d, _)| TextRunnerDecode {
                state: &mut *d.state,
                max_tokens: d.max_tokens,
                sampler: &mut *d.sampler,
            })
            .collect();
        let mut active_steps = match active.len() {
            0 => Vec::new(),
            1 => {
                let first = &mut active[0];
                vec![self.decode_step(&mut *first.state, first.max_tokens, &mut *first.sampler)]
            }
            _ => match self.try_dspark_cohort(&mut active)? {
                Some(steps) => steps.into_iter().map(Ok).collect(),
                // Isolate a cancelled request instead of failing its peers.
                None => text::default_decode_batch(self, &mut active),
            },
        }
        .into_iter();
        Ok(full
            .iter()
            .map(|&full| {
                Ok(if full {
                    TextDecodeStep {
                        stop: true,
                        ..TextDecodeStep::default()
                    }
                } else {
                    active_steps
                        .next()
                        .unwrap_or_else(|| Ok(TextDecodeStep::default()))?
                })
            })
            .collect())
    }

    fn advance_batch(
        &self,
        advances: &mut [TextRunnerAdvance<'_, DeepSeekState>],
    ) -> Result<Vec<Result<()>>> {
        if advances.len() < 2 || advances.len() > 8 {
            return Err(Error::invalid(
                "DeepSeek batched decode requires two to eight sessions",
            ));
        }
        let tokens = advances
            .iter()
            .map(|a| engine_token(a.token))
            .collect::<Result<Vec<_>>>()?;
        let interrupted = {
            let mut engine = self.exec();
            let mut items: Vec<(&mut Session, i32)> = advances
                .iter_mut()
                .zip(&tokens)
                .map(|(a, &token)| (&mut a.state.session, token))
                .collect();
            match Session::eval_batch(&mut engine, &mut items) {
                Ok(()) => false,
                Err(e) if e.interrupted => true,
                Err(e) => return Err(engine_error(format!("DeepSeek batch decode failed: {e}"))),
            }
        };
        // A cancelled request is rejected before any session moves: advance
        // the others alone instead of failing every client of the batch.
        if interrupted {
            return Ok(text::default_advance_batch(self, advances));
        }
        for advance in advances.iter_mut() {
            advance.state.position += 1;
        }
        Ok(advances.iter().map(|_| Ok(())).collect())
    }

    fn checkpoint_position(&self, state: &DeepSeekState) -> Result<usize> {
        Ok(state.position)
    }

    fn snapshot_payload_bytes(&self, state: &DeepSeekState) -> Result<usize> {
        let session = &state.session;
        if self.use_dspark && session.dspark_stats().context_tokens as usize != session.position() {
            return Err(Error::runtime("DSpark prefix lacks complete support state"));
        }
        match usize::try_from(session.payload_bytes()) {
            Ok(bytes) if bytes != 0 => Ok(bytes),
            _ => Err(Error::runtime("DeepSeek snapshot size is unavailable")),
        }
    }

    fn snapshot(&self, state: &DeepSeekState) -> Result<DeepSeekSnapshot> {
        let bytes = state
            .session
            .save_snapshot_with(&self.snapshots)
            .map_err(|e| engine_error(format!("DeepSeek snapshot failed: {e}")))?;
        Ok(DeepSeekSnapshot {
            runner: self.id,
            bytes,
            position: state.position,
        })
    }

    fn restore_or_fork(
        &self,
        state: &mut DeepSeekState,
        snapshot: &DeepSeekSnapshot,
    ) -> Result<()> {
        let payload = self.check_snapshot(snapshot)?;
        let engine = self.exec();
        state
            .session
            .restore_snapshot(&engine, payload)
            .map_err(|e| engine_error(format!("DeepSeek snapshot restore failed: {e}")))?;
        state.position = snapshot.position;
        Ok(())
    }

    fn persistent_snapshot_payload_bytes(&self, snapshot: &DeepSeekSnapshot) -> Result<usize> {
        self.check_snapshot(snapshot).map(<[u8]>::len).map_err(|_| {
            Error::invalid("DeepSeek persistent snapshot does not belong to this model")
        })
    }

    fn serialize_persistent_snapshot(
        &self,
        snapshot: &DeepSeekSnapshot,
        destination: &mut [u8],
    ) -> Result<usize> {
        let payload = self.check_snapshot(snapshot)?;
        if destination.len() != payload.len() {
            return Err(Error::invalid(
                "DeepSeek persistent snapshot serialization failed",
            ));
        }
        destination.copy_from_slice(payload);
        Ok(destination.len())
    }

    fn stream_persistent_snapshot(
        &self,
        snapshot: &DeepSeekSnapshot,
        sink: &mut dyn FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        sink(self.check_snapshot(snapshot)?)
    }

    fn restore_persistent_snapshot(&self, state: &mut DeepSeekState, payload: &[u8]) -> Result<()> {
        let engine = self.exec();
        state
            .session
            .restore_snapshot(&engine, payload)
            .map_err(|e| {
                engine_error(format!("DeepSeek persistent snapshot restore failed: {e}"))
            })?;
        let position = state.session.position();
        if position == 0 || position > self.max_context as usize {
            state.session.invalidate();
            return Err(engine_error(
                "DeepSeek persistent snapshot restored an invalid position".into(),
            ));
        }
        state.position = position;
        Ok(())
    }
}

fn runner_tokens(tokens: &[i32]) -> Result<Vec<Token>> {
    tokens
        .iter()
        .map(|&t| {
            Token::try_from(t).map_err(|_| Error::invalid("DeepSeek token ID must not be negative"))
        })
        .collect()
}

fn engine_token(token: Token) -> Result<i32> {
    i32::try_from(token).map_err(|_| Error::invalid("DeepSeek token ID exceeds engine range"))
}

fn engine_tokens(tokens: &[Token]) -> Result<Vec<i32>> {
    tokens.iter().map(|&t| engine_token(t)).collect()
}
