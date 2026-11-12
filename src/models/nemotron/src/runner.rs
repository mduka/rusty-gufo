//! The serving runner of Nemotron 3.5 Bolt: the extension's runner
//! (`runner.cpp`) on the text runner API this server keeps, Gufo v0.28.1's
//! `TextModelRunner`, with images through NVIDIA's vision projector, the
//! device logit heads of top-k sampling, the separator after a reasoning
//! block the prompt closed, and NVIDIA's recommended sampling.
//!
//! One engine session per request: chunked prefill, one decode step per
//! token or one LiLiCorr draft-and-verify round, live-prefix reuse at the
//! exact frontier (Mamba states cannot rewind) and host snapshots for
//! earlier boundaries.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use gufo_core::json_constraint::{ConstraintVocabulary, Piece, ToolFormat};
use gufo_core::sampling::{SamplerState, SamplingConfig, TokenId, collect_logprobs};
use gufo_core::text::{
    self, ChatRequest, ChatRole, ContinuationInputPrefix, Error, InitialOutputState, PromptContext,
    ReasoningTags, Result, TextDecodeSelection, TextDecodeStep, TextExecutionPlan, TextModelRunner,
    TextPrefillStep, TextPreparedPrompt, TextRunnerAdvance, TextRunnerCapabilities,
    TextRunnerDecode, TextRunnerDescriptor, TextRunnerResourceClaim, TextRunnerSnapshot, Token,
    ToolChoice,
};
use gufo_hip::{DeviceProbe, DeviceProbeStatus};

use crate::chat_template::{
    self, ChatMessage, ChatTool, ChatToolArgument, ChatToolCall, ContentSpan,
};
use crate::engine::{
    InputRows, MAX_BATCH, Model, ModelOptions, PREFILL_CHUNK, Session, SessionSnapshot,
};
use crate::logit_head::{LogitHead, LogitHeadFilter, head_filter_for, sample_with_head};
use crate::speculative::{DraftHistory, Frontier, speculative_step};
use crate::vision_encoder::{VisionEmbedding, VisionEncoder, VisionEncoderOptions};
use crate::vision_gemm::VisionOperand;
use crate::vision_prompt::{
    ImageFraming, VisionInput, VisionPrompt, expand_images, find_image_framing,
};

/// The model ID clients name (the extension's descriptor).
pub const MODEL_ID: &str = "nemotron-3.5-bolt";

/// The layout of a session's state (`kStateAbi`).
const STATE_ABI: &str = "nemotron-h/state/v1;kv=f16;mamba=f32;chunk=2048";

/// The GGUF identity scheme the vision identity names
/// (`core::kGgufIdentityScheme`).
const GGUF_IDENTITY_SCHEME: &str = "gguf-sha256-v1";

// Concurrent requests either decode one token each in a shared pass, or
// draft in turn, one LiLiCorr round each, when their recent rounds emit
// tokens faster than that pass does. A round costs about 42 ms plus 2.9 ms
// per draft, a shared pass 27 ms plus 3 ms per further request: two
// requests writing code (6 of 7 drafts kept) gain from rounds, prose does
// not.
/// Most requests that draft in turn. Four share a pass at 111 tok/s or
/// more, which rounds of eight tokens in 62 ms barely beat (the extension
/// measured 1 % slower).
const MAX_DRAFT_IN_TURN: usize = 3;
/// Tokens of its answer a request decodes in shared passes before it may
/// draft in turn: a reply this short gains nothing from rounds, and peers
/// that arrive together still form a batch.
const SHARED_ANSWER_TOKENS: u32 = 16;
/// Weight of the newest measurement in a decayed mean.
const RATE_WEIGHT: f64 = 0.2;
/// Tokens per round toward which a state's mean drifts while it decodes in
/// shared passes, so that it tries a round again now and then: about the
/// rate of code.
const OPTIMISTIC_ROUND_TOKENS: f64 = 5.0;
const ROUND_TOKENS_DRIFT: f64 = 0.01;

/// A shared pass of `width` requests on Strix Halo before measurements, in
/// seconds.
fn prior_pass_seconds(width: usize) -> f64 {
    0.0273 + 0.003 * (width - 1) as f64
}

fn measure(mean: &mut f64, value: f64, first: bool) {
    *mean = if first {
        value
    } else {
        *mean + RATE_WEIGHT * (value - *mean)
    };
}

/// Decayed means of a state's recent LiLiCorr rounds: the tokens each
/// emitted and its wall time.
#[derive(Debug, Clone, Copy, Default)]
struct RoundRate {
    tokens: f64,
    seconds: f64,
    measured: bool,
}

static NEXT_RUNNER: AtomicU64 = AtomicU64::new(1);

pub struct RunnerOptions {
    pub model_path: PathBuf,
    /// Zero for the native context.
    pub max_context: u32,
    /// NVIDIA's LiLiCorr drafter; `None` decodes one token per step.
    pub draft_model_path: Option<PathBuf>,
    /// At most this many drafts per round, bounded by the drafter's slots.
    pub max_draft_tokens: u32,
    /// NVIDIA's vision projector (`--mmproj`); `None` refuses images.
    pub vision_model_path: Option<PathBuf>,
}

/// Whether the model at `path` is a Nemotron-H MoE (`nemotron_h_moe`)
/// artifact this runtime supports.
pub fn supports(path: &Path) -> bool {
    crate::probe::probe_bolt(path)
}

/// The prompt ends at the session's position; `closed_reasoning` when it
/// ends in "</think>". Rates of earlier requests describe other text.
fn start_answer(state: &mut NemotronState, closed_reasoning: bool) {
    state.leading_separator = closed_reasoning;
    state.round_rate = RoundRate::default();
    state.answer_start = state.session.position();
}

fn runtime(message: impl Into<String>) -> Error {
    Error::runtime(message.into())
}

/// The serving view of a device probe.
fn device_status(status: DeviceProbeStatus) -> text::DeviceStatus {
    match status {
        DeviceProbeStatus::Usable => text::DeviceStatus::Usable,
        DeviceProbeStatus::Pending => text::DeviceStatus::Pending,
        DeviceProbeStatus::Lost => text::DeviceStatus::Lost,
    }
}

fn has_images(request: &ChatRequest) -> bool {
    request.messages.iter().any(|m| !m.images.is_empty())
}

/// The length of the common prefix of two token sequences.
fn agree(a: &[TokenId], b: &[TokenId]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The images of a prompt with images, travelling with its request.
pub struct VisionContext {
    prompt: Arc<VisionPrompt>,
    cache_prefixes: Vec<ContinuationInputPrefix>,
}

impl PromptContext for VisionContext {
    fn cache_identity(&self) -> &[u8] {
        &self.prompt.cache_identity
    }

    fn cache_prefixes(&self) -> &[ContinuationInputPrefix] {
        &self.cache_prefixes
    }
}

/// One request's session plus the next-token distribution at its frontier.
pub struct NemotronState {
    session: Session,
    /// The host copy of the frontier's logits and, when the device
    /// summarized them, their head.
    frontier: Frontier,
    /// How this conversation's text has accepted drafts.
    draft_history: DraftHistory,
    /// What the current request's rounds have yielded, to choose between
    /// drafting in turn and decoding in a shared pass beside other requests.
    round_rate: RoundRate,
    /// The position where the current request's answer began.
    answer_start: u32,
    /// The host copy matches the session's frontier.
    logits_ready: bool,
    /// The head filter of the request decoding from this state, for the
    /// device summary of each next row.
    head_filter: Option<LogitHeadFilter>,
    /// The prompt closed reasoning itself ("<think></think>") and no answer
    /// text has been emitted yet: whitespace the model writes first is the
    /// separator after that block, framing rather than answer.
    leading_separator: bool,
    /// False for an `ignore_eos` request.
    stop_at_eos: bool,
    /// The current prompt's images, and the embeddings of those that
    /// prefill has reached but not yet passed, by image index.
    vision: Option<Arc<VisionPrompt>>,
    embeddings: BTreeMap<usize, Arc<VisionEmbedding>>,
}

impl text::TextRunnerState for NemotronState {
    fn invalidate(&mut self) {
        self.session.invalidate();
        self.logits_ready = false;
        self.frontier.head.candidates.clear();
    }

    fn set_stop_at_eos(&mut self, stop_at_eos: bool) -> Result<()> {
        self.stop_at_eos = stop_at_eos;
        Ok(())
    }
}

/// Host copy of a session, with its frontier distribution when it has one.
pub struct NemotronSnapshot {
    runner: u64,
    session: SessionSnapshot,
    /// Empty without a decode frontier.
    logits: Vec<f32>,
}

impl TextRunnerSnapshot for NemotronSnapshot {
    fn payload_bytes(&self) -> usize {
        self.session.bytes() + self.logits.len() * 4
    }
}

pub struct NemotronRunner {
    id: u64,
    model: Arc<Model>,
    vision: Option<Arc<VisionEncoder>>,
    framing: Option<ImageFraming>,
    /// What image embeddings depend on: the projector and the operand
    /// format.
    vision_identity: String,
    context: u32,
    draft_tokens: u32,
    /// "</think>", which ends the prompt when it closed reasoning itself.
    thought_close: TokenId,
    /// Decayed mean wall time of a shared decode pass, by width; zero
    /// before the first.
    pass_seconds: Mutex<[f64; MAX_BATCH + 1]>,
    /// Tells a lost GPU context from an ordinary failure.
    device_probe: DeviceProbe,
}

impl NemotronRunner {
    /// The provider's factory (`CreateNemotronTextRunner`): loads the
    /// trunk, the LiLiCorr drafter and the vision projector.
    pub fn load(options: &RunnerOptions) -> Result<Self> {
        let mut model = Model::load(
            &options.model_path,
            ModelOptions {
                max_context: options.max_context,
            },
        )
        .map_err(Error::runtime)?;
        let context = model.max_context();
        if context < 2 {
            return Err(Error::invalid(
                "Nemotron needs a context of at least two tokens",
            ));
        }
        if let Some(path) = &options.draft_model_path {
            model.load_drafter(path).map_err(Error::runtime)?;
        }
        let model = Arc::new(model);
        // NVIDIA's vision projector, with its matrix-product algorithms
        // chosen now rather than during the first images.
        let vision = match &options.vision_model_path {
            Some(path) => {
                let encoder =
                    VisionEncoder::load(path, model.config(), VisionEncoderOptions::default())
                        .map_err(Error::runtime)?;
                encoder.prepare().map_err(Error::runtime)?;
                Some(Arc::new(encoder))
            }
            None => None,
        };
        Self::new(model, context, options.max_draft_tokens, vision)
    }

    /// A runner over a loaded model (`CreateNemotronRunner`). `draft_tokens`
    /// bounds the drafts of one round; zero decodes one token per step.
    pub fn new(
        model: Arc<Model>,
        context: u32,
        draft_tokens: u32,
        vision: Option<Arc<VisionEncoder>>,
    ) -> Result<Self> {
        let draft_tokens = model
            .drafter()
            .map_or(0, |d| draft_tokens.min(d.config().slots()));
        let close = model.tokenizer().encode("</think>", true);
        let [thought_close] = close[..] else {
            return Err(Error::invalid("Nemotron vocabulary has no </think> token"));
        };
        let (framing, vision_identity) = match &vision {
            Some(encoder) => {
                let framing = find_image_framing(model.tokenizer()).map_err(Error::invalid)?;
                // Embeddings depend on the projector and on the operand
                // format.
                let operand = match encoder.operand() {
                    VisionOperand::F16 => ";operand=fp16",
                    VisionOperand::Bf16 => ";operand=bf16",
                };
                (
                    Some(framing),
                    format!("{GGUF_IDENTITY_SCHEME}:{}{operand}", encoder.identity()),
                )
            }
            None => (None, String::new()),
        };
        Ok(Self {
            id: NEXT_RUNNER.fetch_add(1, Ordering::Relaxed),
            device_probe: DeviceProbe::new().map_err(|e| runtime(format!("device probe: {e}")))?,
            model,
            vision,
            framing,
            vision_identity,
            context,
            draft_tokens,
            thought_close,
            pass_seconds: Mutex::new([0.0; MAX_BATCH + 1]),
        })
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    pub fn vision_encoder(&self) -> Option<&Arc<VisionEncoder>> {
        self.vision.as_ref()
    }

    fn is_stop(&self, token: TokenId) -> bool {
        self.model.tokenizer().is_stop(token)
    }

    /// The text of `token` in the answer. The host trims the separator after
    /// a "</think>" the model writes, but it has no state for a block the
    /// prompt closed, where the template continues
    /// "<think></think>\n<tool_call>". Without this, a reply's leading
    /// newline would reach clients as text, and a client sending it back
    /// would no longer match the generated tokens. The tokens themselves
    /// stay as the model wrote them.
    /// Whether the requests emit tokens faster drafting in turn than in a
    /// shared pass, by their recent rounds and the recent passes of this
    /// width.
    fn draft_in_turn(&self, decodes: &[TextRunnerDecode<'_, NemotronState, SamplerState>]) -> bool {
        if self.draft_tokens == 0 || decodes.len() > MAX_DRAFT_IN_TURN {
            return false;
        }
        let (mut tokens, mut seconds, mut unmeasured) = (0.0, 0.0, false);
        for decode in decodes {
            let state = &*decode.state;
            if decode.max_tokens <= 1
                || state.session.position()
                    < state.answer_start.saturating_add(SHARED_ANSWER_TOKENS)
            {
                return false;
            }
            let rate = state.round_rate;
            unmeasured |= !rate.measured;
            tokens += rate.tokens;
            seconds += rate.seconds;
        }
        // A request without rounds of its own drafts one to measure its text.
        if unmeasured {
            return true;
        }
        let measured = self
            .pass_seconds
            .lock()
            .map_or(0.0, |passes| passes[decodes.len()]);
        let pass = if measured > 0.0 {
            measured
        } else {
            prior_pass_seconds(decodes.len())
        };
        tokens * pass > decodes.len() as f64 * seconds
    }

    fn measure_pass(&self, width: usize, seconds: f64) {
        if let Ok(mut passes) = self.pass_seconds.lock() {
            let mean = &mut passes[width];
            let first = *mean == 0.0;
            measure(mean, seconds, first);
        }
    }

    fn piece(&self, state: &mut NemotronState, token: TokenId) -> Vec<u8> {
        let piece = self.model.tokenizer().token_bytes(token);
        if !state.leading_separator {
            return piece.to_vec();
        }
        let is_space = |b: &u8| matches!(b, b' ' | b'\t' | b'\r' | b'\n' | 0x0c | 0x0b);
        match piece.iter().position(|b| !is_space(b)) {
            None => Vec::new(),
            Some(first) => {
                state.leading_separator = false;
                piece[first..].to_vec()
            }
        }
    }

    /// After a restore or a failure the host copy of the frontier is stale.
    fn read_frontier(&self, state: &mut NemotronState) -> Result<()> {
        if state.logits_ready {
            return Ok(());
        }
        state
            .session
            .read_logits(&mut state.frontier.logits)
            .map_err(|e| runtime(format!("Nemotron frontier read failed: {e}")))?;
        state.frontier.head.candidates.clear();
        state.logits_ready = true;
        Ok(())
    }

    /// The conversation through the model's chat template (`Render`).
    fn render(
        &self,
        request: &ChatRequest,
        generation_prompt: bool,
    ) -> Result<(String, Vec<ContentSpan>)> {
        let mut messages = Vec::with_capacity(request.messages.len());
        for message in &request.messages {
            if !message.images.is_empty() && self.vision.is_none() {
                return Err(Error::invalid("model does not support image input"));
            }
            let role = match message.role {
                ChatRole::System | ChatRole::Developer => chat_template::ChatRole::System,
                ChatRole::User => chat_template::ChatRole::User,
                ChatRole::Assistant => chat_template::ChatRole::Assistant,
                ChatRole::Tool => chat_template::ChatRole::Tool,
            };
            messages.push(ChatMessage {
                role,
                content: format!("{}{}", message.content, message.framing_suffix),
                reasoning: message.thought.clone(),
                tool_calls: message
                    .tool_calls
                    .iter()
                    .map(|call| ChatToolCall {
                        name: call.name.clone(),
                        arguments: call
                            .arguments
                            .iter()
                            .map(|a| ChatToolArgument {
                                name: a.name.clone(),
                                value: a.value.clone(),
                                is_string: a.is_string,
                            })
                            .collect(),
                    })
                    .collect(),
                image_offsets: message.images.iter().map(|image| image.offset).collect(),
            });
        }
        let tools: Vec<ChatTool> = if request.tool_choice == ToolChoice::None {
            Vec::new()
        } else {
            request
                .tools
                .iter()
                .map(|tool| ChatTool {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters_json: tool.parameters_json.clone(),
                    definition_json: tool.definition_json.clone(),
                })
                .collect()
        };
        let mut options = chat_template::resolve_chat_options(&request.reasoning);
        options.add_generation_prompt = generation_prompt;
        chat_template::render_chat(&messages, &tools, &options).map_err(Error::invalid)
    }

    /// The rendered conversation's tokens.
    fn rendered_tokens(
        &self,
        request: &ChatRequest,
        generation_prompt: bool,
    ) -> Result<Vec<TokenId>> {
        let (rendered, spans) = self.render(request, generation_prompt)?;
        Ok(chat_template::tokenize_chat(
            self.model.tokenizer(),
            &rendered,
            &spans,
        ))
    }

    /// Where the prompt agrees with both ways a conversation goes on. A
    /// client that discards the reply appends its next user turn after the
    /// history, before the assistant opening. A follow-up re-renders the
    /// reply after it, with earlier reasoning as "<think></think>". In
    /// both, the template moves the reasoning effort hint to the new last
    /// user turn.
    fn stable_boundary(&self, request: &ChatRequest, rendered: &[TokenId]) -> Result<usize> {
        let history = self.rendered_tokens(request, false)?;
        let mut continuation = request.clone();
        continuation
            .messages
            .push(text::ChatMessage::new(ChatRole::Assistant, ""));
        continuation
            .messages
            .push(text::ChatMessage::new(ChatRole::User, ""));
        let continued = self.rendered_tokens(&continuation, true)?;
        Ok(agree(rendered, &history).min(agree(rendered, &continued)))
    }

    /// The rendered prompt with each image placeholder expanded into its
    /// span (`ExpandRequest`).
    fn expand_request(&self, request: &ChatRequest, preprocess: bool) -> Result<VisionPrompt> {
        let (Some(vision), Some(framing)) = (&self.vision, &self.framing) else {
            return Err(Error::invalid("model does not support image input"));
        };
        let rendered = self.rendered_tokens(request, true)?;
        let images: Vec<VisionInput> = request
            .messages
            .iter()
            .flat_map(|m| m.images.iter())
            .map(|image| VisionInput {
                bytes: Some(Arc::clone(&image.bytes)),
                low_detail: image.detail == text::ImageDetail::Low,
            })
            .collect();
        expand_images(
            vision.config(),
            framing,
            &rendered,
            &images,
            &self.vision_identity,
            preprocess,
        )
        .map_err(Error::invalid)
    }

    /// `PrepareImagePrompt`: the stable boundary as for text prompts, found
    /// among the rendered tokens and moved past the images expanded before
    /// it.
    fn prepare_image_prompt(
        &self,
        request: &ChatRequest,
    ) -> Result<TextPreparedPrompt<VisionContext>> {
        let prompt = Arc::new(self.expand_request(request, true)?);
        let rendered = self.rendered_tokens(request, true)?;
        let common = self.stable_boundary(request, &rendered)?;
        // A prefix ending at or before an image holds only the images
        // before it.
        let cache_prefixes = prompt
            .images
            .iter()
            .zip(&prompt.prefix_identities)
            .map(|(image, identity)| ContinuationInputPrefix {
                token_count: image.offset as usize,
                identity: identity.clone(),
            })
            .collect();
        let cache_prefix_tokens = if common == rendered.len() {
            0
        } else {
            prompt.expanded(common)
        };
        Ok(TextPreparedPrompt {
            tokens: prompt.tokens.clone(),
            context: Some(Arc::new(VisionContext {
                prompt,
                cache_prefixes,
            })),
            cache_prefix_tokens,
        })
    }

    fn select(
        &self,
        state: &mut NemotronState,
        sampler: &mut SamplerState,
    ) -> Result<TextDecodeSelection> {
        if state.session.position() >= self.context {
            return Ok(TextDecodeSelection {
                stop: true,
                ..TextDecodeSelection::default()
            });
        }
        if state.session.position() == 0 {
            return Err(Error::logic("Nemotron selection before its frontier"));
        }
        self.read_frontier(state)?;
        state.head_filter = head_filter_for(sampler.config());
        let token = sample_with_head(sampler, &state.frontier.logits, &state.frontier.head)?;
        let stop = state.stop_at_eos && self.is_stop(token);
        let logprobs = collect_logprobs(&state.frontier.logits, token, sampler.config().logprobs)?;
        Ok(TextDecodeSelection {
            stop,
            token,
            piece: if stop {
                Vec::new()
            } else {
                self.piece(state, token)
            },
            logprobs,
        })
    }

    fn check_token(&self, token: Token) -> Result<i32> {
        if token as usize >= self.model.tokenizer().vocab_size() {
            return Err(Error::invalid("Nemotron token ID exceeds the vocabulary"));
        }
        i32::try_from(token).map_err(|_| Error::invalid("Nemotron token ID exceeds the vocabulary"))
    }

    /// One token of each state in a shared pass (`AdvanceBatch`).
    fn advance_shared(&self, advances: &mut [TextRunnerAdvance<'_, NemotronState>]) -> Result<()> {
        let n = advances.len();
        if !(2..=MAX_BATCH).contains(&n) {
            return Err(Error::invalid(
                "Nemotron batched decode requires two to eight sessions",
            ));
        }
        let mut tokens = Vec::with_capacity(n);
        let mut filters = Vec::with_capacity(n);
        for advance in advances.iter() {
            tokens.push(self.check_token(advance.token)?);
            filters.push(advance.state.head_filter);
        }
        let mut heads = vec![LogitHead::default(); n];
        let result = {
            let mut sessions: Vec<&mut Session> = Vec::with_capacity(n);
            let mut logits: Vec<&mut [f32]> = Vec::with_capacity(n);
            for advance in advances.iter_mut() {
                let state = &mut *advance.state;
                sessions.push(&mut state.session);
                logits.push(&mut state.frontier.logits[..]);
            }
            Session::decode_batch(&mut sessions, &tokens, &mut logits, &filters, &mut heads)
        };
        if let Err(error) = result {
            for advance in advances.iter_mut() {
                advance.state.logits_ready = false;
            }
            return Err(runtime(format!("Nemotron batched decode failed: {error}")));
        }
        for (advance, head) in advances.iter_mut().zip(heads) {
            advance.state.frontier.head = head;
            advance.state.logits_ready = true;
        }
        Ok(())
    }
}

impl TextModelRunner for NemotronRunner {
    type State = NemotronState;
    type Snapshot = NemotronSnapshot;
    type Sampler = SamplerState;
    type PromptContext = VisionContext;

    fn descriptor(&self) -> TextRunnerDescriptor {
        let drafting = self.draft_tokens != 0;
        TextRunnerDescriptor {
            model_id: MODEL_ID.into(),
            // States of a drafting runner also carry the drafter's context.
            state_abi: format!(
                "{STATE_ABI}{}",
                if drafting { ";draft=lilicorr" } else { "" }
            ),
            max_context: self.context,
            capabilities: TextRunnerCapabilities {
                incremental_prefill: true,
                snapshot: true,
                fork: true,
                final_token_advance_required: true,
                incremental_text_is_exact: true,
                multi_token_decode: drafting,
                // Concurrent drafting requests decode one token each.
                batched_multi_token_decode: drafting,
                batched_multi_token_decode_max_width: if drafting { MAX_BATCH } else { 0 },
                prefix_reuse: true,
                in_pass_checkpoint: false,
            },
            persistence: None,
        }
    }

    fn resource_claim(&self) -> TextRunnerResourceClaim {
        TextRunnerResourceClaim {
            resident_weights_bytes: Some(
                self.model.resident_bytes()
                    + self.model.drafter().map_or(0, |d| d.resident_bytes())
                    + self.vision.as_ref().map_or(0, |v| v.resident_bytes()),
            ),
            state_capacity_bytes: gufo_hip::device_memory().ok().map(|(free, _)| free),
            per_request_state_bytes: Some(self.model.session_bytes(self.context)),
            temporary_scratch_bytes: self.vision.as_ref().map(|v| v.max_scratch_bytes()),
            retained_snapshot_capacity_bytes: Some(text::host_snapshot_budget_bytes()),
            retained_snapshot_ceiling_bytes: Some(text::host_snapshot_ceiling_bytes()),
            requires_device_runtime_lock: true,
        }
    }

    fn supported_plans(&self) -> Vec<TextExecutionPlan> {
        // Concurrent requests decode one token each in a shared pass;
        // drafting stays with a request decoding alone.
        std::iter::once(TextExecutionPlan::SERIAL)
            .chain((2..=MAX_BATCH).map(TextExecutionPlan::batched))
            .collect()
    }

    /// Raw prompts map control-token spellings to their special tokens, as
    /// the chat template's render does and as llama-server's `/completion`
    /// does.
    fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
        Ok(self.model.tokenizer().encode(text, true))
    }

    /// llama.cpp forces "\n</think>" for this template, the newline after
    /// "<think>" in the generation prompt counting toward the budget.
    fn reasoning_budget_tags(&self) -> ReasoningTags {
        ReasoningTags {
            start: "<think>",
            forced_end: "\n</think>",
            ends: &["</think>"],
        }
    }

    fn render_and_tokenize(&self, request: &ChatRequest) -> Result<Option<Vec<Token>>> {
        self.rendered_tokens(request, true).map(Some)
    }

    fn render_prompt(&self, request: &ChatRequest) -> Result<Option<String>> {
        // Images expand into embedding rows, which no text spells.
        if has_images(request) {
            return Ok(None);
        }
        self.render(request, true).map(|(text, _)| Some(text))
    }

    fn prepare_prompt(
        &self,
        request: &ChatRequest,
    ) -> Result<Option<TextPreparedPrompt<VisionContext>>> {
        if has_images(request) {
            return self.prepare_image_prompt(request).map(Some);
        }
        let tokens = self.rendered_tokens(request, true)?;
        let common = self.stable_boundary(request, &tokens)?;
        // Zero keeps the complete prompt as the stable checkpoint rather
        // than as an optional retry copy.
        let cache_prefix_tokens = if common == tokens.len() { 0 } else { common };
        Ok(Some(TextPreparedPrompt {
            tokens,
            context: None,
            cache_prefix_tokens,
        }))
    }

    fn count_prompt_tokens(&self, request: &ChatRequest) -> Result<Option<usize>> {
        if !has_images(request) {
            return self
                .render_and_tokenize(request)
                .map(|t| t.map(|t| t.len()));
        }
        // Sized as generation prepares it, without resizing the pixels.
        Ok(Some(self.expand_request(request, false)?.tokens.len()))
    }

    fn set_prompt_context(
        &self,
        state: &mut NemotronState,
        context: Option<&Arc<VisionContext>>,
    ) -> Result<()> {
        let prompt = context.map(|c| Arc::clone(&c.prompt));
        let same = match (&prompt, &state.vision) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        if !same {
            state.vision = prompt;
            state.embeddings.clear();
        }
        Ok(())
    }

    fn initial_output_state(&self, request: &ChatRequest) -> InitialOutputState {
        // Without thinking the prompt ends in "<think></think>".
        if chat_template::resolve_chat_options(&request.reasoning).enable_thinking {
            InitialOutputState::Reasoning
        } else {
            InitialOutputState::Content
        }
    }

    /// The template's Qwen3-Coder XML calls.
    fn tool_format(&self) -> ToolFormat {
        ToolFormat::Qwen
    }

    fn supports_images(&self) -> bool {
        self.vision.is_some()
    }

    fn poll_device(&self) -> text::DeviceStatus {
        device_status(self.device_probe.poll())
    }

    fn build_constraint_vocabulary(&self) -> Result<ConstraintVocabulary> {
        let tokenizer = self.model.tokenizer();
        ConstraintVocabulary::new(tokenizer.vocab_size() as u32, |id| {
            Ok(Piece {
                text: tokenizer.token_bytes(id).to_vec(),
                stop: tokenizer.is_stop(id),
            })
        })
    }

    fn decode(&self, tokens: &[Token]) -> Result<Vec<u8>> {
        let tokenizer = self.model.tokenizer();
        let mut text = Vec::new();
        for &token in tokens {
            if token as usize >= tokenizer.vocab_size() {
                return Err(Error::invalid("Nemotron token ID exceeds the vocabulary"));
            }
            text.extend_from_slice(tokenizer.token_bytes(token));
        }
        Ok(text)
    }

    fn create_state(&self) -> Result<NemotronState> {
        let session = self
            .model
            .create_session(self.context)
            .map_err(|e| runtime(format!("Nemotron session allocation failed: {e}")))?;
        Ok(NemotronState {
            session,
            frontier: Frontier {
                logits: vec![0.0; self.model.config().vocab_size as usize],
                head: LogitHead::default(),
            },
            draft_history: DraftHistory::default(),
            round_rate: RoundRate::default(),
            answer_start: 0,
            logits_ready: false,
            head_filter: None,
            leading_separator: false,
            stop_at_eos: true,
            vision: None,
            embeddings: BTreeMap::new(),
        })
    }

    fn create_sampler(&self, config: &SamplingConfig, prompt: &[Token]) -> Result<SamplerState> {
        SamplerState::new(config.clone(), prompt)
    }

    fn prepare_prefix_reuse(&self, state: &mut NemotronState, prefix: &[Token]) -> Result<()> {
        if state.session.position() as usize != prefix.len() {
            return Err(Error::logic(
                "Nemotron reused prefix does not match its state",
            ));
        }
        // A prompt served whole from the cache ends here; prefill overrides
        // it when more of the prompt follows.
        start_answer(state, prefix.last() == Some(&self.thought_close));
        Ok(())
    }

    fn prefill(
        &self,
        state: &mut NemotronState,
        prompt: &[Token],
        offset: usize,
        max_input_tokens: usize,
    ) -> Result<TextPrefillStep> {
        if offset != state.session.position() as usize {
            return Err(Error::logic("Nemotron prefill offset does not match state"));
        }
        if offset >= prompt.len() {
            return Err(Error::logic("Nemotron prefill has no remaining input"));
        }
        let consumed = max_input_tokens
            .min(prompt.len() - offset)
            .min(PREFILL_CHUNK);
        if consumed == 0 {
            return Err(Error::invalid("Nemotron prefill budget must be positive"));
        }
        let decode_ready = offset + consumed == prompt.len();
        let tokens = prompt[offset..offset + consumed]
            .iter()
            .map(|&t| self.check_token(t))
            .collect::<Result<Vec<i32>>>()?;
        state.frontier.head.candidates.clear();
        // The embeddings of the images this step reaches, encoded on first
        // use. Attention is causal, so a step may end inside an image.
        let mut reached: Vec<(u32, Arc<VisionEmbedding>)> = Vec::new();
        if let Some(vision) = state.vision.clone() {
            for (i, image) in vision.images.iter().enumerate() {
                let end = image.offset as usize + image.rows as usize;
                if end <= offset || image.offset as usize >= offset + consumed {
                    continue;
                }
                let embedding = match state.embeddings.get(&i) {
                    Some(embedding) => Arc::clone(embedding),
                    None => {
                        let (Some(encoder), Some(patches)) = (&self.vision, &image.patches) else {
                            return Err(Error::logic("Nemotron image prompt without its pixels"));
                        };
                        let embedding = encoder
                            .encode(patches, &image.grid, None)
                            .map_err(|e| runtime(format!("Nemotron {e}")))?;
                        state.embeddings.insert(i, Arc::clone(&embedding));
                        embedding
                    }
                };
                reached.push((image.offset, embedding));
            }
        }
        let rows: Vec<InputRows<'_>> = reached
            .iter()
            .map(|(position, embedding)| InputRows {
                position: *position,
                rows: embedding.data(),
            })
            .collect();
        let logits: &mut [f32] = if decode_ready {
            &mut state.frontier.logits
        } else {
            &mut []
        };
        let appended = state.session.append_with_rows(&tokens, logits, &rows);
        drop(rows);
        // Images this step finished are not read again unless a restore
        // returns before them, which encodes them anew.
        if let Some(vision) = &state.vision {
            state.embeddings.retain(|&i, _| {
                vision.images.get(i).is_some_and(|image| {
                    image.offset as usize + image.rows as usize > offset + consumed
                })
            });
        }
        if let Err(error) = appended {
            state.logits_ready = false;
            return Err(runtime(format!("Nemotron prefill failed: {error}")));
        }
        state.logits_ready = decode_ready;
        if decode_ready {
            start_answer(state, prompt.last() == Some(&self.thought_close));
        }
        Ok(TextPrefillStep {
            consumed_tokens: consumed,
            decode_ready,
        })
    }

    fn select_next(
        &self,
        state: &mut NemotronState,
        sampler: &mut SamplerState,
    ) -> Result<TextDecodeSelection> {
        self.select(state, sampler)
    }

    fn advance(&self, state: &mut NemotronState, token: Token) -> Result<()> {
        let one = [self.check_token(token)?];
        let frontier = &mut state.frontier;
        frontier.head.candidates.clear();
        let result = match &state.head_filter {
            Some(filter) => state.session.append_with_head(
                &one,
                &mut frontier.logits,
                filter,
                &mut frontier.head,
            ),
            None => state.session.append(&one, &mut frontier.logits),
        };
        if let Err(error) = result {
            state.logits_ready = false;
            return Err(runtime(format!("Nemotron decode failed: {error}")));
        }
        state.logits_ready = true;
        Ok(())
    }

    /// A LiLiCorr round: the drafter proposes, the trunk verifies them in
    /// one pass and the tokens the request's sampler agrees with are kept.
    /// Only the RNG and a pending draw from a rejected draft carry over.
    fn decode_step(
        &self,
        state: &mut NemotronState,
        max_tokens: usize,
        sampler: &mut SamplerState,
    ) -> Result<TextDecodeStep> {
        if self.draft_tokens == 0 || max_tokens <= 1 {
            return text::default_decode_step(self, state, max_tokens, sampler);
        }
        if state.session.position() >= self.context {
            return Ok(TextDecodeStep {
                stop: true,
                ..TextDecodeStep::default()
            });
        }
        if state.session.position() == 0 {
            return Err(Error::logic("Nemotron selection before its frontier"));
        }
        self.read_frontier(state)?;
        let budget = max_tokens.min(self.draft_tokens as usize + 1) as u32;
        let stop_at_eos = state.stop_at_eos;
        let tokenizer = self.model.tokenizer();
        let is_stop =
            |token: i32| stop_at_eos && u32::try_from(token).is_ok_and(|t| tokenizer.is_stop(t));
        let logprobs = sampler.config().logprobs;
        let mut working = sampler.clone();
        let start = Instant::now();
        let round = match speculative_step(
            &mut state.session,
            &mut working,
            &mut state.frontier,
            &mut state.draft_history,
            budget,
            &is_stop,
            logprobs.is_some(),
        ) {
            Ok(round) => round,
            Err(error) => {
                state.logits_ready = false;
                return Err(runtime(format!("Nemotron draft round failed: {error}")));
            }
        };
        if round.drafted != 0 {
            let rate = &mut state.round_rate;
            let first = !rate.measured;
            measure(&mut rate.tokens, round.tokens.len() as f64, first);
            measure(&mut rate.seconds, start.elapsed().as_secs_f64(), first);
            rate.measured = true;
        }
        sampler.copy_draw_state_from(&working);
        state.head_filter = head_filter_for(sampler.config());
        let vocab = state.frontier.logits.len();
        let mut step = TextDecodeStep::default();
        for (i, &token) in round.tokens.iter().enumerate() {
            let token = token as Token;
            let token_logprobs = if logprobs.is_some() {
                collect_logprobs(&round.rows[i * vocab..(i + 1) * vocab], token, logprobs)?
            } else {
                None
            };
            if round.stop && i + 1 == round.tokens.len() {
                step.stop = true;
                step.terminal_logprobs = token_logprobs;
                break;
            }
            let piece = self.piece(state, token);
            step.selections.push(TextDecodeSelection {
                stop: false,
                token,
                piece,
                logprobs: token_logprobs,
            });
        }
        step.draft_rounds = usize::from(round.drafted != 0);
        step.draft_tokens = round.drafted as usize;
        step.draft_accepted_tokens = round.accepted as usize;
        Ok(step)
    }

    /// Requests decoding together take one token each from a shared pass,
    /// or one LiLiCorr round each in turn when that is faster
    /// (`draft_in_turn`).
    fn decode_batch(
        &self,
        decodes: &mut [TextRunnerDecode<'_, NemotronState, SamplerState>],
    ) -> Result<Vec<Result<TextDecodeStep>>> {
        if decodes.len() < 2 || decodes.len() > MAX_BATCH {
            return Ok(text::default_decode_batch(self, decodes));
        }
        if self.draft_in_turn(decodes) {
            return Ok(decodes
                .iter_mut()
                .map(|decode| {
                    self.decode_step(decode.state, decode.max_tokens, decode.sampler)
                        .map(|mut step| {
                            step.execution_plan = TextExecutionPlan::SERIAL;
                            step
                        })
                })
                .collect());
        }
        let start = Instant::now();
        let mut steps: Vec<Result<TextDecodeStep>> = Vec::with_capacity(decodes.len());
        let mut advanced = Vec::new();
        for (i, decode) in decodes.iter_mut().enumerate() {
            match self.select(decode.state, decode.sampler) {
                Ok(selection) if selection.stop => steps.push(Ok(TextDecodeStep {
                    stop: true,
                    terminal_logprobs: selection.logprobs,
                    ..TextDecodeStep::default()
                })),
                Ok(selection) => {
                    advanced.push((i, selection.token));
                    steps.push(Ok(TextDecodeStep {
                        selections: vec![selection],
                        ..TextDecodeStep::default()
                    }));
                }
                Err(error) => steps.push(Err(error)),
            }
        }
        let result = {
            let mut advances: Vec<TextRunnerAdvance<'_, NemotronState>> =
                Vec::with_capacity(advanced.len());
            let mut next = advanced.iter().peekable();
            for (i, decode) in decodes.iter_mut().enumerate() {
                if let Some(&&(index, token)) = next.peek()
                    && index == i
                {
                    next.next();
                    advances.push(TextRunnerAdvance {
                        state: &mut *decode.state,
                        token,
                    });
                }
            }
            match advances.len() {
                0 => Ok(()),
                1 => {
                    let token = advances[0].token;
                    self.advance(advances[0].state, token)
                }
                _ => self.advance_shared(&mut advances),
            }
        };
        match result {
            Ok(()) if advanced.len() == decodes.len() => {
                self.measure_pass(advanced.len(), start.elapsed().as_secs_f64());
            }
            Ok(()) => {}
            Err(error) => {
                for &(i, _) in &advanced {
                    steps[i] = Err(error.clone());
                }
            }
        }
        for decode in decodes.iter_mut() {
            let rate = &mut decode.state.round_rate;
            if rate.measured && rate.tokens < OPTIMISTIC_ROUND_TOKENS {
                rate.tokens += ROUND_TOKENS_DRIFT * (OPTIMISTIC_ROUND_TOKENS - rate.tokens);
            }
        }
        let plan = if advanced.len() > 1 {
            TextExecutionPlan::batched(advanced.len())
        } else {
            TextExecutionPlan::SERIAL
        };
        for step in steps.iter_mut().flatten() {
            step.execution_plan = plan;
        }
        Ok(steps)
    }

    fn advance_batch(
        &self,
        advances: &mut [TextRunnerAdvance<'_, NemotronState>],
    ) -> Result<Vec<Result<()>>> {
        match advances.len() {
            0 => Ok(Vec::new()),
            1 => {
                let token = advances[0].token;
                Ok(vec![self.advance(advances[0].state, token)])
            }
            n if n > MAX_BATCH => Ok(text::default_advance_batch(self, advances)),
            n => {
                let result = self.advance_shared(advances);
                Ok((0..n).map(|_| result.clone()).collect())
            }
        }
    }

    fn checkpoint_position(&self, state: &NemotronState) -> Result<usize> {
        Ok(state.session.position() as usize)
    }

    fn snapshot_payload_bytes(&self, state: &NemotronState) -> Result<usize> {
        let frontier = if state.session.position() > 0 {
            state.frontier.logits.len() * 4
        } else {
            0
        };
        Ok(state.session.snapshot_bytes() + frontier)
    }

    fn snapshot(&self, state: &NemotronState) -> Result<NemotronSnapshot> {
        let session = state
            .session
            .save()
            .map_err(|e| runtime(format!("Nemotron snapshot failed: {e}")))?;
        // The frontier of a position that ends a prompt or a decoded token:
        // the host copy when current, else the session's device row.
        let logits = if state.session.position() == 0 {
            Vec::new()
        } else if state.logits_ready {
            state.frontier.logits.clone()
        } else {
            let mut logits = vec![0.0f32; state.frontier.logits.len()];
            state
                .session
                .read_logits(&mut logits)
                .map_err(|e| runtime(format!("Nemotron snapshot failed: {e}")))?;
            logits
        };
        Ok(NemotronSnapshot {
            runner: self.id,
            session,
            logits,
        })
    }

    fn restore_or_fork(
        &self,
        state: &mut NemotronState,
        snapshot: &NemotronSnapshot,
    ) -> Result<()> {
        if snapshot.runner != self.id {
            return Err(Error::invalid(
                "Nemotron snapshot does not belong to this model",
            ));
        }
        if let Err(error) = state.session.restore(&snapshot.session) {
            state.logits_ready = false;
            return Err(runtime(format!(
                "Nemotron snapshot restore failed: {error}"
            )));
        }
        state.frontier.head.candidates.clear();
        state.logits_ready = !snapshot.logits.is_empty();
        if state.logits_ready {
            state.frontier.logits.copy_from_slice(&snapshot.logits);
        }
        Ok(())
    }
}
