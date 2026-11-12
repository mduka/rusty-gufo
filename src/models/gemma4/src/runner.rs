//! The serving runner of Gemma 4 (the extension's `runner.cpp`) on the text
//! runner API this server keeps, Gufo v0.28.1's `TextModelRunner`, in the
//! shape of `Gemma4TextRunner` (`src/cli/serve/inference_backend.cpp` of
//! franzmoca/gufo's `feat/gemma` branch, which carries the extension onto
//! v0.28.1).
//!
//! One engine session per request: prefill syncs the session to each step's
//! prefix (the engine reuses what it holds), decoding runs MTP cycles (the
//! drafter, prompt lookup and siblings) or single tokens, and concurrent
//! requests decode in one verification forward. Snapshots share the
//! engine's position blocks; with the disk cache they persist under the
//! extension's compatibility identity.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use gufo_core::gguf_identity;
use gufo_core::json_constraint::{ConstraintVocabulary, Piece, ToolFormat};
use gufo_core::sampling::{SamplerState, SamplingConfig, collect_logprobs};
use gufo_core::text::{
    self, ChatRequest, ChatTool, ContinuationInputPrefix, Error, InitialOutputState, PromptContext,
    ReasoningMarkup, ReasoningTags, Result, TextDecodeSelection, TextDecodeStep, TextExecutionPlan,
    TextModelRunner, TextPrefillStep, TextPreparedPrompt, TextRunnerAdvance,
    TextRunnerCapabilities, TextRunnerDecode, TextRunnerDescriptor, TextRunnerMeasuredResources,
    TextRunnerPersistenceDescriptor, TextRunnerResourceClaim, TextRunnerSnapshot, Token,
    ToolChoice,
};
use gufo_hip::{DeviceProbe, DeviceProbeStatus, DeviceTensor};

use crate::chat_template::{
    self, CHANNEL_OPEN, ChatOptions, TEMPLATE_ID, THOUGHT_END, THOUGHT_START,
    UNSLOTH_TEMPLATE_SHA256,
};
use crate::draft_policy::{DraftCalibrationScope, DraftPolicy};
use crate::engine::{
    BatchDecode, DecodeResult, ImageSpan, Model, ModelOptions, Session, SessionSnapshot,
    SpeculativeStats,
};
use crate::provider_options::PrefillWeights;
use crate::tokenizer::TokenId;
use crate::vision_encoder::VisionEncoder;
use crate::vision_prompt::{self, Prompt as VisionPrompt};

/// Each request owns its KV cache (global rows plus the sliding-window ring)
/// and its snapshots (`kStateAbi`).
const STATE_ABI: &str = "gemma4-rocm-session-v1";

/// Most sessions one decode forward serves.
const MAX_BATCH: usize = 8;

static NEXT_RUNNER: AtomicU64 = AtomicU64::new(1);

/// What `CreateGemma4TextRunner` takes: the server's options with the
/// provider options parsed.
#[derive(Debug, Clone, PartialEq)]
pub struct RunnerOptions {
    pub model_path: PathBuf,
    /// Zero for the native context.
    pub max_context: u32,
    /// The `gemma4-assistant` drafter (`--speculative mtp --mtp-model`).
    pub mtp_model_path: Option<PathBuf>,
    pub draft_tokens: u32,
    pub min_draft_tokens: u32,
    pub draft_policy: DraftPolicy,
    pub draft_calibration: DraftCalibrationScope,
    pub draft_trace_path: Option<PathBuf>,
    pub prefill_weights: PrefillWeights,
    /// `--mmproj`; without it `mmproj-BF16.gguf` beside the model is used
    /// when present.
    pub vision_model_path: Option<PathBuf>,
    /// Soft tokens per image.
    pub image_tokens: u32,
    /// Fingerprint the artifacts for the disk cache.
    pub persistence: bool,
}

impl RunnerOptions {
    /// The engine's options.
    pub fn model_options(&self) -> ModelOptions {
        ModelOptions {
            max_context: self.max_context,
            mtp_model_path: self.mtp_model_path.clone(),
            draft_tokens: self.draft_tokens,
            draft_policy: self.draft_policy,
            min_draft_tokens: self.min_draft_tokens,
            draft_calibration: self.draft_calibration,
            draft_trace_path: self.draft_trace_path.clone(),
            prefill_weights: self.prefill_weights,
            ..ModelOptions::default()
        }
    }
}

/// Whether the model at `path` is a Gemma 4 target GGUF.
pub fn supports(path: &Path) -> bool {
    crate::probe::probe_gemma4(path)
}

fn runtime(message: impl Into<String>) -> Error {
    Error::runtime(message.into())
}

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

/// `EngineToken`.
fn engine_token(token: Token) -> Result<TokenId> {
    TokenId::try_from(token).map_err(|_| Error::invalid("Gemma 4 token ID exceeds engine range"))
}

fn engine_tokens(tokens: &[Token]) -> Result<Vec<TokenId>> {
    tokens.iter().map(|&t| engine_token(t)).collect()
}

fn runner_tokens(tokens: &[TokenId]) -> Result<Vec<Token>> {
    tokens
        .iter()
        .map(|&t| Token::try_from(t).map_err(|_| runtime("Gemma 4 produced a negative token ID")))
        .collect()
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The disk cache's compatibility identity (`CompatibilityIdentity`).
pub fn compatibility_identity(
    artifact_fingerprint: &str,
    mtp_fingerprint: Option<&str>,
    max_context: u32,
    draft_tokens: u32,
    ring: usize,
) -> Result<Vec<u8>> {
    if !is_sha256_hex(artifact_fingerprint) || mtp_fingerprint.is_some_and(|f| !is_sha256_hex(f)) {
        return Err(Error::invalid(
            "Gemma 4 disk cache requires artifact fingerprints",
        ));
    }
    let scheme = gguf_identity::SCHEME;
    let mut identity = format!(
        "schema=gufo-text-continuation-v2\nmodel_kind=gemma4\n\
         artifact_id={scheme}:{artifact_fingerprint}\ntokenizer=embedded-in-artifact\n\
         chat_template={TEMPLATE_ID}\nchat_template_reference_sha256={UNSLOTH_TEMPLATE_SHA256}\n\
         state_abi={STATE_ABI}\npayload_layout=gemma4-rocm-session-snapshot-v{}\n\
         kv_layout=f16-token-major-global+sliding-ring-logical\nsliding_ring_slots={ring}\n\
         context_tokens={max_context}\nposition_policy=absolute-v1\nadapters=none\n",
        Session::SNAPSHOT_PAYLOAD_VERSION
    );
    if let Some(mtp) = mtp_fingerprint {
        identity.push_str(&format!(
            "draft_backend=gemma4-assistant-mtp-v1\ndraft_artifact_id={scheme}:{mtp}\n\
             draft_tokens={draft_tokens}\n"
        ));
    }
    Ok(identity.into_bytes())
}

/// The tokens of `thought\n` after the channel opener: the host starts a
/// reasoning budget after `<|channel>`, so they would count against it;
/// zero when the tokenizer does not split the markers that way.
fn thought_header_tokens(open: &[TokenId], channel: &[TokenId]) -> usize {
    match channel {
        [opener] if open.len() > 1 && open[0] == *opener => open.len() - 1,
        _ => 0,
    }
}

/// `CoverThoughtHeader`: a request's budget widened by the thought header,
/// so a budget spent inside the header cannot leave unparsable markup.
fn cover_thought_header(config: &SamplingConfig, header: usize) -> SamplingConfig {
    let mut config = config.clone();
    if header != 0
        && let Some(budget) = &config.reasoning_budget
    {
        let mut widened = (**budget).clone();
        widened.tokens = widened.tokens.saturating_add(header);
        config.reasoning_budget = Some(Arc::new(widened));
    }
    config
}

/// Images of one request (`ImageContext`): they extend the cache identity
/// at their first token, so a prefix ending at or before an image holds only
/// the earlier images.
pub struct ImageContext {
    prompt: Arc<VisionPrompt>,
    cache_prefixes: Vec<ContinuationInputPrefix>,
}

impl ImageContext {
    fn new(prompt: VisionPrompt) -> Self {
        let cache_prefixes = prompt
            .images
            .iter()
            .map(|image| ContinuationInputPrefix {
                token_count: image.span.offset as usize,
                identity: prompt
                    .identity_for_prefix(image.span.offset as usize)
                    .to_vec(),
            })
            .collect();
        Self {
            prompt: Arc::new(prompt),
            cache_prefixes,
        }
    }
}

impl PromptContext for ImageContext {
    fn cache_identity(&self) -> &[u8] {
        &self.prompt.cache_identity
    }

    fn cache_prefixes(&self) -> &[ContinuationInputPrefix] {
        &self.cache_prefixes
    }

    /// `ImageContext::PrefillStop`: before an image or after its closing
    /// token.
    fn prefill_stop(&self, position: usize, up: bool) -> usize {
        crate::text_contract::prefill_stop(Some(&self.prompt), position, up)
    }
}

/// One request's session (`State`), with the images of its prompt and the
/// embeddings encoded for them.
pub struct Gemma4State {
    session: Session,
    stop_at_eos: bool,
    images: Option<Arc<VisionPrompt>>,
    embeddings: BTreeMap<usize, Arc<DeviceTensor<f32>>>,
}

impl Gemma4State {
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// The current request's images; their embeddings are encoded on first
    /// use (`SetImages`).
    fn set_images(&mut self, prompt: Option<Arc<VisionPrompt>>) {
        let same = match (&prompt, &self.images) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        if !same {
            self.images = prompt;
            self.embeddings.clear();
        }
    }
}

impl text::TextRunnerState for Gemma4State {
    fn invalidate(&mut self) {
        self.session.reset();
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
}

/// A session snapshot of one runner.
pub struct Gemma4Snapshot {
    runner: u64,
    session: SessionSnapshot,
}

impl TextRunnerSnapshot for Gemma4Snapshot {
    fn payload_bytes(&self) -> usize {
        usize::try_from(self.session.size_bytes()).unwrap_or(usize::MAX)
    }
}

/// The emitted tokens of a decode, up to a stop token whose logprobs become
/// the step's terminal entry (`AppendTokens`), with the cycle counts
/// (`Count`).
struct StepTokens<'a> {
    runner: &'a Gemma4Runner,
    stop_at_eos: bool,
}

impl StepTokens<'_> {
    fn step(
        &self,
        decoded: DecodeResult,
        before: &SpeculativeStats,
        after: &SpeculativeStats,
    ) -> Result<TextDecodeStep> {
        let mut step = TextDecodeStep::default();
        let mut logprobs = decoded.logprobs.into_iter();
        for &token in &decoded.tokens {
            let token_logprobs = logprobs.next();
            if self.stop_at_eos && self.runner.model.is_stop_token(token) {
                step.stop = true;
                step.terminal_logprobs = token_logprobs;
                break;
            }
            step.selections.push(TextDecodeSelection {
                stop: false,
                token: engine_to_runner(token)?,
                piece: self.runner.model.token_text(token),
                logprobs: token_logprobs,
            });
        }
        step.draft_rounds = (after.verified - before.verified) as usize;
        step.draft_tokens = (after.drafted - before.drafted) as usize;
        step.draft_accepted_tokens = (after.accepted - before.accepted) as usize;
        step.lookup_tokens = (after.copied - before.copied) as usize;
        step.lookup_accepted_tokens = (after.copied_accepted - before.copied_accepted) as usize;
        Ok(step)
    }
}

fn engine_to_runner(token: TokenId) -> Result<Token> {
    Token::try_from(token).map_err(|_| runtime("Gemma 4 produced a negative token ID"))
}

pub struct Gemma4Runner {
    id: u64,
    model: Arc<Model>,
    vision: Option<Arc<VisionEncoder>>,
    image_tokens: u32,
    max_context: u32,
    /// The tokens of `thought\n` after `<|channel>`.
    thought_header: usize,
    persistence: Option<TextRunnerPersistenceDescriptor>,
    device_probe: DeviceProbe,
}

impl Gemma4Runner {
    /// The provider's factory (`CreateGemma4TextRunner`): loads the target
    /// with its MTP drafter, opens the vision sidecar and, for the disk
    /// cache, fingerprints the artifacts.
    pub fn load(options: &RunnerOptions) -> Result<Self> {
        if !vision_prompt::is_soft_token_budget(options.image_tokens) {
            return Err(Error::invalid(
                "--image-tokens must be 70, 140, 280, 560 or 1120",
            ));
        }
        if options
            .mtp_model_path
            .as_ref()
            .is_some_and(|_| options.draft_tokens == 0)
        {
            return Err(Error::invalid(
                "Gemma 4 MTP requires --mtp-model and a positive draft limit",
            ));
        }
        let model = Model::load(&options.model_path, options.model_options())
            .map_err(|e| runtime(format!("Failed to create Gemma 4 model: {e}")))?;
        let vision = VisionEncoder::open(
            &options.model_path,
            options.vision_model_path.as_deref(),
            model.config().hidden_size,
        )
        .map_err(|e| runtime(format!("Gemma 4 vision sidecar: {e}")))?
        .map(Arc::new);
        let max_context = if options.max_context == 0 {
            model.max_context()
        } else {
            options.max_context
        };
        if max_context == 0 || max_context > model.max_context() {
            return Err(Error::invalid(
                "Gemma 4 HTTP sessions or context exceed the loaded model",
            ));
        }
        if options.mtp_model_path.is_some() != model.has_mtp() {
            return Err(Error::invalid(
                "Gemma 4 MTP requires a model loaded with its drafter",
            ));
        }
        let persistence = if options.persistence {
            let fingerprint = |label: &str, path: &Path| {
                gguf_identity::identity_hex(&[path.to_path_buf()])
                    .map_err(|e| runtime(format!("{label} fingerprinting failed: {e}")))
            };
            let artifact = fingerprint("Gemma 4", &options.model_path)?;
            let mtp = match &options.mtp_model_path {
                Some(path) => Some(fingerprint("Gemma 4 MTP", path)?),
                None => None,
            };
            Some(TextRunnerPersistenceDescriptor {
                compatibility_identity: compatibility_identity(
                    &artifact,
                    mtp.as_deref(),
                    max_context,
                    model.draft_tokens(),
                    model.session_ring_slots(),
                )?,
                payload_version: Session::SNAPSHOT_PAYLOAD_VERSION,
            })
        } else {
            None
        };
        Self::new(
            model,
            max_context,
            vision,
            options.image_tokens,
            persistence,
        )
    }

    /// A runner over a loaded model; the vision sidecar is hashed and
    /// uploaded now, so the first image request pays neither
    /// (`PrepareVision`).
    pub fn new(
        model: Arc<Model>,
        max_context: u32,
        vision: Option<Arc<VisionEncoder>>,
        image_tokens: u32,
        persistence: Option<TextRunnerPersistenceDescriptor>,
    ) -> Result<Self> {
        if let Some(encoder) = &vision {
            let _ = encoder.identity();
            encoder.upload().map_err(|e| {
                runtime(format!("could not upload the Gemma 4 vision sidecar: {e}"))
            })?;
        }
        let thought_header = thought_header_tokens(
            &model.tokenize(THOUGHT_START),
            &model.tokenize(CHANNEL_OPEN),
        );
        Ok(Self {
            id: NEXT_RUNNER.fetch_add(1, Ordering::Relaxed),
            device_probe: DeviceProbe::new().map_err(|e| runtime(format!("device probe: {e}")))?,
            model,
            vision,
            image_tokens,
            max_context,
            thought_header,
            persistence,
        })
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    pub fn vision_encoder(&self) -> Option<&Arc<VisionEncoder>> {
        self.vision.as_ref()
    }

    /// `Tools`: none when the request turned tools off.
    fn tools(request: &ChatRequest) -> &[ChatTool] {
        if request.tool_choice == ToolChoice::None {
            &[]
        } else {
            &request.tools
        }
    }

    /// `Options`.
    fn chat_options(request: &ChatRequest) -> ChatOptions {
        chat_template::resolve_chat_options(&request.reasoning)
    }

    /// The conversation through the chat template; images are rendered as
    /// placeholders.
    fn render(&self, request: &ChatRequest) -> Result<chat_template::RenderedPrompt> {
        let mut offsets = Vec::new();
        chat_template::render(
            &request.messages,
            Self::tools(request),
            &Self::chat_options(request),
            has_images(request).then_some(&mut offsets),
        )
        .map_err(|e| Error::invalid(format!("Gemma 4 chat template: {e}")))
    }

    fn require_vision(&self) -> Result<&Arc<VisionEncoder>> {
        self.vision
            .as_ref()
            .ok_or_else(|| Error::invalid("image input requires a matching --mmproj BF16 sidecar"))
    }

    /// The prompt with each image expanded into its soft tokens (`Prepare`).
    fn prepare_images(&self, request: &ChatRequest, max_context: u32) -> Result<VisionPrompt> {
        let encoder = self.require_vision()?;
        vision_prompt::prepare(
            self.model.tokenizer(),
            &request.messages,
            Self::tools(request),
            &Self::chat_options(request),
            encoder.identity(),
            max_context,
            self.image_tokens,
        )
    }

    /// `PrepareImagePrompt`.
    fn prepare_image_prompt(
        &self,
        request: &ChatRequest,
    ) -> Result<TextPreparedPrompt<ImageContext>> {
        let prompt = self.prepare_images(request, self.max_context)?;
        let tokens = runner_tokens(&prompt.tokens)?;
        let cache_prefix_tokens = prompt.stable_prefix_tokens;
        Ok(TextPreparedPrompt {
            tokens,
            context: Some(Arc::new(ImageContext::new(prompt))),
            cache_prefix_tokens,
        })
    }

    /// Where a step starting at `offset` with `max_input_tokens` ends
    /// (`StepEnd`): never inside an image or before its closing token; it
    /// stops before the image, or takes the whole image when it starts
    /// there.
    fn step_end(
        state: &Gemma4State,
        prompt_len: usize,
        offset: usize,
        max_input_tokens: usize,
    ) -> Result<usize> {
        if offset != state.session.position() as usize || offset >= prompt_len {
            return Err(Error::logic("Gemma 4 prefill offset does not match state"));
        }
        let mut end = offset + max_input_tokens.min(prompt_len - offset);
        if let Some(images) = &state.images {
            for image in &images.images {
                let first = image.span.offset as usize;
                let last = first + image.span.rows as usize; // closing token
                if end > first && end <= last {
                    end = if first > offset { first } else { last + 1 };
                }
            }
        }
        Ok(end)
    }

    /// Syncs the session to `prefix` with the images that end before it
    /// (`Sync`); the embeddings of images it must evaluate are encoded on
    /// first use.
    fn sync(&self, state: &mut Gemma4State, prefix: &[TokenId], end: usize) -> Result<()> {
        let Gemma4State {
            session,
            images,
            embeddings,
            ..
        } = state;
        let mut spans = Vec::new();
        if let Some(prompt) = images {
            for image in &prompt.images {
                if (image.span.offset as usize) + (image.span.rows as usize) < end {
                    spans.push(ImageSpan {
                        offset: image.span.offset,
                        rows: image.span.rows,
                        identity: image.prefix_identity,
                    });
                }
            }
        }
        let synced = {
            let cache = RefCell::new(&mut *embeddings);
            let embed = |index: usize| -> std::result::Result<Arc<DeviceTensor<f32>>, String> {
                let mut cache = cache.borrow_mut();
                if let Some(embedding) = cache.get(&index) {
                    return Ok(Arc::clone(embedding));
                }
                let encoder = self
                    .vision
                    .as_ref()
                    .ok_or("Gemma 4 image input has no encoder")?;
                let image = images
                    .as_ref()
                    .and_then(|prompt| prompt.images.get(index))
                    .ok_or("Gemma 4 image index outside the prompt")?;
                let embedding = encoder.encode(&image.pixels)?;
                cache.insert(index, Arc::clone(&embedding));
                Ok(embedding)
            };
            session.sync_with(prefix, &spans, &embed, &[], &[])
        };
        if let Err(error) = synced {
            session.reset();
            return Err(runtime(format!("Gemma 4 prefill failed: {error}")));
        }
        // Images the session passed are read again only after a restore
        // before them, which encodes them anew.
        if let Some(prompt) = images {
            embeddings.retain(|&i, _| {
                prompt.images.get(i).is_some_and(|image| {
                    image.span.offset as usize + image.span.rows as usize >= end
                })
            });
        }
        Ok(())
    }

    fn stop_step() -> TextDecodeStep {
        TextDecodeStep {
            stop: true,
            ..TextDecodeStep::default()
        }
    }
}

impl TextModelRunner for Gemma4Runner {
    type State = Gemma4State;
    type Snapshot = Gemma4Snapshot;
    type Sampler = SamplerState;
    type PromptContext = ImageContext;

    fn descriptor(&self) -> TextRunnerDescriptor {
        let mtp = self.model.has_mtp();
        TextRunnerDescriptor {
            model_id: self.model.model_name(),
            state_abi: STATE_ABI.into(),
            max_context: self.max_context,
            capabilities: TextRunnerCapabilities {
                incremental_prefill: true,
                snapshot: true,
                fork: true,
                final_token_advance_required: false,
                incremental_text_is_exact: true,
                multi_token_decode: mtp,
                batched_multi_token_decode: mtp,
                batched_multi_token_decode_max_width: MAX_BATCH,
                prefix_reuse: true,
                in_pass_checkpoint: false,
            },
            persistence: self.persistence.clone(),
        }
    }

    fn resource_claim(&self) -> TextRunnerResourceClaim {
        // Snapshots are device allocations carved from host RAM on the APU:
        // cap both limits by the host and by the device heap.
        let capacity = gufo_hip::device_memory().ok().map(|(free, _)| free);
        let cap = |bytes: usize| capacity.map_or(bytes, |free| bytes.min(free));
        TextRunnerResourceClaim {
            resident_weights_bytes: Some(
                self.model.resident_bytes()
                    + self.vision.as_ref().map_or(0, |v| v.resident_bytes()),
            ),
            state_capacity_bytes: capacity,
            per_request_state_bytes: Some(self.model.session_bytes(self.max_context)),
            temporary_scratch_bytes: Some(0),
            retained_snapshot_capacity_bytes: Some(cap(text::host_snapshot_budget_bytes())),
            retained_snapshot_ceiling_bytes: Some(cap(text::host_snapshot_ceiling_bytes())),
            requires_device_runtime_lock: true,
        }
    }

    fn supported_plans(&self) -> Vec<TextExecutionPlan> {
        // Up to eight sessions share one forward (decode widths stay exact).
        std::iter::once(TextExecutionPlan::SERIAL)
            .chain((2..=MAX_BATCH).map(TextExecutionPlan::batched))
            .collect()
    }

    /// Special tokens parsed, no BOS: rendered chat prompts carry their own
    /// `<bos>`.
    fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
        runner_tokens(&self.model.tokenize(text))
    }

    /// The budget opens on the channel token; without a header length it
    /// would cut the header, so Gemma then declares none.
    fn reasoning_budget_tags(&self) -> ReasoningTags {
        if self.thought_header == 0 {
            return ReasoningTags {
                start: "",
                forced_end: "",
                ends: &[],
            };
        }
        ReasoningTags {
            start: CHANNEL_OPEN,
            forced_end: THOUGHT_END,
            ends: &[THOUGHT_END],
        }
    }

    /// The thought channel the model opens itself (`ReasoningDelimiters`).
    fn reasoning_markup(&self) -> ReasoningMarkup {
        ReasoningMarkup {
            start: THOUGHT_START,
            end: THOUGHT_END,
            model_opens: true,
        }
    }

    fn render_and_tokenize(&self, request: &ChatRequest) -> Result<Option<Vec<Token>>> {
        Ok(self
            .prepare_prompt(request)?
            .map(|prepared| prepared.tokens))
    }

    fn render_prompt(&self, request: &ChatRequest) -> Result<Option<String>> {
        // Images expand into embedding rows, which no text spells.
        if has_images(request) {
            return Ok(None);
        }
        self.render(request).map(|rendered| Some(rendered.text))
    }

    fn prepare_prompt(
        &self,
        request: &ChatRequest,
    ) -> Result<Option<TextPreparedPrompt<ImageContext>>> {
        if has_images(request) {
            return self.prepare_image_prompt(request).map(Some);
        }
        let rendered = self.render(request)?;
        let tokens = self.tokenize(&rendered.text)?;
        // The stable prefix is shared by the next turn; the text after it
        // starts with a special token, so it tokenizes apart.
        let mut cache_prefix_tokens = 0;
        if rendered.stable_prefix_offset < rendered.text.len() {
            let suffix = self.tokenize(&rendered.text[rendered.stable_prefix_offset..])?;
            if suffix.len() < tokens.len() && tokens.ends_with(&suffix) {
                cache_prefix_tokens = tokens.len() - suffix.len();
            }
        }
        Ok(Some(TextPreparedPrompt {
            tokens,
            context: None,
            cache_prefix_tokens,
        }))
    }

    fn count_prompt_tokens(&self, request: &ChatRequest) -> Result<Option<usize>> {
        if !has_images(request) {
            return Ok(self.prepare_prompt(request)?.map(|p| p.tokens.len()));
        }
        // Generation's CPU image preparation, without encoding. No context
        // limit, so oversized prompts can still be sized.
        Ok(Some(self.prepare_images(request, u32::MAX)?.tokens.len()))
    }

    fn set_prompt_context(
        &self,
        state: &mut Gemma4State,
        context: Option<&Arc<ImageContext>>,
    ) -> Result<()> {
        state.set_images(context.map(|c| Arc::clone(&c.prompt)));
        Ok(())
    }

    fn initial_output_state(&self, request: &ChatRequest) -> InitialOutputState {
        let options = Self::chat_options(request);
        if !options.enable_thinking {
            // The prompt closes the thought channel; a repeated block still
            // parses.
            return InitialOutputState::Auto;
        }
        // After a tool response the prompt already opened the thought
        // channel.
        match self.render(request) {
            Ok(rendered) if rendered.text.ends_with(THOUGHT_START) => InitialOutputState::Reasoning,
            _ => InitialOutputState::Auto,
        }
    }

    /// `<|tool_call>call:NAME{...}<tool_call|>`.
    fn tool_format(&self) -> ToolFormat {
        ToolFormat::Gemma4
    }

    fn supports_images(&self) -> bool {
        self.vision.is_some()
    }

    fn poll_device(&self) -> text::DeviceStatus {
        device_status(self.device_probe.poll())
    }

    fn build_constraint_vocabulary(&self) -> Result<ConstraintVocabulary> {
        let model = &self.model;
        ConstraintVocabulary::new(model.vocab_size(), |id| {
            let token = engine_token(id)?;
            Ok(Piece {
                text: model.decode(&[token]),
                stop: model.is_stop_token(token),
            })
        })
    }

    fn decode(&self, tokens: &[Token]) -> Result<Vec<u8>> {
        let vocab = self.model.vocab_size();
        if tokens.iter().any(|&t| t >= vocab) {
            return Err(Error::invalid("Gemma 4 token ID exceeds the vocabulary"));
        }
        Ok(self.model.decode(&engine_tokens(tokens)?))
    }

    fn create_state(&self) -> Result<Gemma4State> {
        let session = self
            .model
            .create_session(self.max_context)
            .map_err(|e| runtime(format!("Gemma 4 session allocation failed: {e}")))?;
        Ok(Gemma4State {
            session,
            stop_at_eos: true,
            images: None,
            embeddings: BTreeMap::new(),
        })
    }

    /// The request's sampler, its reasoning budget widened by the thought
    /// header (`CoverThoughtHeader`, which the extension applies before the
    /// first step: the sampler has accepted no output then either).
    fn create_sampler(&self, config: &SamplingConfig, prompt: &[Token]) -> Result<SamplerState> {
        SamplerState::new(cover_thought_header(config, self.thought_header), prompt)
    }

    fn preview_first_token(
        &self,
        state: &mut Gemma4State,
        sampler: &mut SamplerState,
    ) -> Result<Option<TextDecodeSelection>> {
        self.select_next(state, sampler).map(Some)
    }

    fn prepare_prefix_reuse(&self, state: &mut Gemma4State, prefix: &[Token]) -> Result<()> {
        if state.session.position() as usize != prefix.len() {
            return Err(Error::logic(
                "Gemma 4 reused prefix does not match checkpoint",
            ));
        }
        // A fully reused prompt decodes without a sync; it still starts a
        // request.
        state.session.begin_request();
        Ok(())
    }

    fn prefill(
        &self,
        state: &mut Gemma4State,
        prompt: &[Token],
        offset: usize,
        max_input_tokens: usize,
    ) -> Result<TextPrefillStep> {
        let end = Self::step_end(state, prompt.len(), offset, max_input_tokens)?;
        // An image starting here is evaluated whole; the pool's step covers
        // it (`PromptContext::prefill_stop`).
        if end - offset > max_input_tokens {
            return Err(Error::logic(format!(
                "a Gemma 4 prefill step of {max_input_tokens} tokens ends inside an image of {} \
                 tokens",
                end - offset
            )));
        }
        let prefix = engine_tokens(&prompt[..end])?;
        self.sync(state, &prefix, end)?;
        Ok(TextPrefillStep {
            consumed_tokens: end - offset,
            decode_ready: end == prompt.len(),
        })
    }

    fn select_next(
        &self,
        state: &mut Gemma4State,
        sampler: &mut SamplerState,
    ) -> Result<TextDecodeSelection> {
        if state.session.position() >= self.max_context {
            return Ok(TextDecodeSelection {
                stop: true,
                ..TextDecodeSelection::default()
            });
        }
        let stop_at_eos = state.stop_at_eos;
        let logits = state
            .session
            .logits()
            .map_err(|e| runtime(format!("Gemma 4 token selection failed: {e}")))?;
        if logits.is_empty() {
            return Err(runtime("Gemma 4 token selection has no logits"));
        }
        let token = sampler.sample(logits)?;
        let logprobs = collect_logprobs(logits, token, sampler.config().logprobs)?;
        let engine = engine_token(token)?;
        if stop_at_eos && self.model.is_stop_token(engine) {
            return Ok(TextDecodeSelection {
                stop: true,
                token: 0,
                piece: Vec::new(),
                logprobs,
            });
        }
        Ok(TextDecodeSelection {
            stop: false,
            token,
            piece: self.model.token_text(engine),
            logprobs,
        })
    }

    fn advance(&self, state: &mut Gemma4State, token: Token) -> Result<()> {
        state
            .session
            .evaluate(engine_token(token)?)
            .map_err(|e| runtime(format!("Gemma 4 decode failed: {e}")))
    }

    /// One MTP cycle: drafts, verifies and emits the accepted prefix plus the
    /// target's own next sample. Only the RNG and a pending draw carry over
    /// to the request's sampler; serving accepts the committed tokens.
    fn decode_step(
        &self,
        state: &mut Gemma4State,
        max_tokens: usize,
        sampler: &mut SamplerState,
    ) -> Result<TextDecodeStep> {
        // Select-and-advance samples the frontier logits, which predict the
        // next token only while no emitted token is pending.
        if !self.model.has_mtp() || (max_tokens == 1 && !state.session.has_pending_token()) {
            return text::default_decode_step(self, state, max_tokens, sampler);
        }
        if state.session.position() >= self.max_context {
            return Ok(Self::stop_step());
        }
        let before = *state.session.statistics();
        let mut working = sampler.clone();
        let budget = max_tokens.min(self.model.draft_tokens() as usize + 1);
        let decoded = state
            .session
            .decode_step(budget, &mut working, state.stop_at_eos)
            .map_err(|e| runtime(format!("Gemma 4 MTP decode failed: {e}")))?;
        sampler.copy_draw_state_from(&working);
        let after = *state.session.statistics();
        StepTokens {
            runner: self,
            stop_at_eos: state.stop_at_eos,
        }
        .step(decoded, &before, &after)
    }

    /// Requests decoding together draft and verify in one forward of at most
    /// 16 rows; a one-token step evaluates its token alone.
    fn decode_batch(
        &self,
        decodes: &mut [TextRunnerDecode<'_, Gemma4State, SamplerState>],
    ) -> Result<Vec<Result<TextDecodeStep>>> {
        if decodes.len() < 2 || !self.model.has_mtp() {
            return Ok(text::default_decode_batch(self, decodes));
        }
        let mut steps: Vec<Option<Result<TextDecodeStep>>> =
            (0..decodes.len()).map(|_| None).collect();
        let mut index = Vec::new();
        for (i, decode) in decodes.iter_mut().enumerate() {
            if decode.max_tokens == 1 && !decode.state.session.has_pending_token() {
                steps[i] = Some(self.decode_step(decode.state, 1, decode.sampler));
            } else if decode.state.session.position() >= self.max_context {
                steps[i] = Some(Ok(Self::stop_step()));
            } else {
                index.push(i);
            }
        }
        let mut working: Vec<SamplerState> =
            index.iter().map(|&i| decodes[i].sampler.clone()).collect();
        let before: Vec<SpeculativeStats> = index
            .iter()
            .map(|&i| *decodes[i].state.session.statistics())
            .collect();
        let draft_limit = self.model.draft_tokens() as usize + 1;
        let outcomes: Vec<(DecodeResult, Option<String>, bool)> = {
            let mut batch: Vec<BatchDecode<'_>> = Vec::with_capacity(index.len());
            let mut next = index.iter().peekable();
            let mut samplers = working.iter_mut();
            for (i, decode) in decodes.iter_mut().enumerate() {
                if next.peek() != Some(&&i) {
                    continue;
                }
                next.next();
                let Some(sampler) = samplers.next() else {
                    break;
                };
                let stop_at_eos = decode.state.stop_at_eos;
                batch.push(BatchDecode {
                    session: &mut decode.state.session,
                    max_tokens: decode.max_tokens.min(draft_limit),
                    sampler,
                    result: DecodeResult::default(),
                    stop_at_eos,
                    error: None,
                });
            }
            if !batch.is_empty() {
                let _ = Session::decode_batch(&mut batch);
            }
            batch
                .into_iter()
                .map(|b| (b.result, b.error, b.stop_at_eos))
                .collect()
        };
        let plan = if index.len() > 1 {
            TextExecutionPlan::batched(index.len())
        } else {
            TextExecutionPlan::SERIAL
        };
        for (k, (decoded, error, stop_at_eos)) in outcomes.into_iter().enumerate() {
            let i = index[k];
            if let Some(error) = error {
                steps[i] = Some(Err(runtime(format!("Gemma 4 MTP decode failed: {error}"))));
                continue;
            }
            decodes[i].sampler.copy_draw_state_from(&working[k]);
            let after = *decodes[i].state.session.statistics();
            steps[i] = Some(
                StepTokens {
                    runner: self,
                    stop_at_eos,
                }
                .step(decoded, &before[k], &after)
                .map(|mut step| {
                    step.execution_plan = plan;
                    step
                }),
            );
        }
        Ok(steps
            .into_iter()
            .map(|step| step.unwrap_or_else(|| Err(Error::logic("Gemma 4 batch lost a step"))))
            .collect())
    }

    fn advance_batch(
        &self,
        advances: &mut [TextRunnerAdvance<'_, Gemma4State>],
    ) -> Result<Vec<Result<()>>> {
        if advances.len() < 2 {
            return Ok(text::default_advance_batch(self, advances));
        }
        let tokens = advances
            .iter()
            .map(|a| engine_token(a.token))
            .collect::<Result<Vec<_>>>()?;
        let result = {
            let mut sessions: Vec<&mut Session> =
                advances.iter_mut().map(|a| &mut a.state.session).collect();
            Session::evaluate_batch(&mut sessions, &tokens)
        };
        let result = result.map_err(|e| runtime(format!("Gemma 4 batched decode failed: {e}")));
        Ok((0..advances.len()).map(|_| result.clone()).collect())
    }

    fn checkpoint_position(&self, state: &Gemma4State) -> Result<usize> {
        Ok(state.session.position() as usize)
    }

    fn snapshot_payload_bytes(&self, state: &Gemma4State) -> Result<usize> {
        usize::try_from(state.session.snapshot_bytes())
            .map_err(|_| runtime("Gemma 4 snapshot exceeds the address space"))
    }

    fn snapshot(&self, state: &Gemma4State) -> Result<Gemma4Snapshot> {
        let session = state
            .session
            .save_snapshot()
            .map_err(|e| runtime(format!("Gemma 4 snapshot failed: {e}")))?;
        Ok(Gemma4Snapshot {
            runner: self.id,
            session,
        })
    }

    fn restore_or_fork(&self, state: &mut Gemma4State, snapshot: &Gemma4Snapshot) -> Result<()> {
        if snapshot.runner != self.id {
            return Err(Error::invalid(
                "Gemma 4 snapshot does not belong to this model",
            ));
        }
        state
            .session
            .restore_snapshot(&snapshot.session)
            .map_err(|e| runtime(format!("Gemma 4 snapshot restore failed: {e}")))
    }

    fn persistent_snapshot_payload_bytes(&self, snapshot: &Gemma4Snapshot) -> Result<usize> {
        if self.persistence.is_none() || snapshot.runner != self.id {
            return Err(Error::invalid(
                "Gemma 4 persistent snapshot does not belong to this model",
            ));
        }
        Ok(snapshot.payload_bytes())
    }

    fn serialize_persistent_snapshot(
        &self,
        snapshot: &Gemma4Snapshot,
        destination: &mut [u8],
    ) -> Result<usize> {
        if destination.len() != self.persistent_snapshot_payload_bytes(snapshot)? {
            return Err(Error::invalid(
                "Gemma 4 persistent snapshot serialization failed",
            ));
        }
        snapshot
            .session
            .copy_to(destination)
            .map_err(|e| runtime(format!("Gemma 4 persistent snapshot failed: {e}")))?;
        Ok(destination.len())
    }

    fn stream_persistent_snapshot(
        &self,
        snapshot: &Gemma4Snapshot,
        sink: &mut dyn FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        self.persistent_snapshot_payload_bytes(snapshot)?;
        let mut failure = None;
        let streamed = snapshot.session.stream(&mut |piece| {
            if failure.is_none()
                && let Err(error) = sink(piece)
            {
                failure = Some(error);
            }
        });
        if let Some(error) = failure {
            return Err(error);
        }
        streamed.map_err(|e| runtime(format!("Gemma 4 persistent snapshot failed: {e}")))
    }

    fn restore_persistent_snapshot(&self, state: &mut Gemma4State, payload: &[u8]) -> Result<()> {
        if self.persistence.is_none() {
            return Err(Error::invalid("Gemma 4 runner has no persistent snapshots"));
        }
        state
            .session
            .restore_payload(payload)
            .map_err(|e| runtime(format!("Gemma 4 persistent snapshot restore failed: {e}")))?;
        let position = state.session.position();
        if position == 0 || position > self.max_context {
            state.session.reset();
            return Err(runtime(
                "Gemma 4 persistent snapshot restored an invalid position",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gufo_core::sampling::ReasoningBudget;

    #[test]
    fn the_thought_header_follows_the_channel_opener() {
        // `<|channel>` `thought` `\n`.
        assert_eq!(thought_header_tokens(&[100, 7, 8], &[100]), 2);
        // A tokenizer that does not split the opener off has no header.
        assert_eq!(thought_header_tokens(&[101, 7], &[100]), 0);
        assert_eq!(thought_header_tokens(&[100], &[100]), 0);
        assert_eq!(thought_header_tokens(&[100, 7], &[100, 9]), 0);
    }

    #[test]
    fn budgets_are_widened_by_the_thought_header() {
        let config = SamplingConfig {
            reasoning_budget: Some(Arc::new(ReasoningBudget {
                tokens: 10,
                ..ReasoningBudget::default()
            })),
            ..SamplingConfig::default()
        };
        let widened = cover_thought_header(&config, 2);
        assert_eq!(
            widened.reasoning_budget.as_ref().map(|b| b.tokens),
            Some(12)
        );
        // The caller's budget is left alone.
        assert_eq!(config.reasoning_budget.as_ref().map(|b| b.tokens), Some(10));
        assert_eq!(
            cover_thought_header(&config, 0)
                .reasoning_budget
                .map(|b| b.tokens),
            Some(10)
        );
        assert!(
            cover_thought_header(&SamplingConfig::default(), 2)
                .reasoning_budget
                .is_none()
        );
    }

    #[test]
    fn the_compatibility_identity_names_the_artifacts() {
        let artifact = "a".repeat(64);
        let mtp = "b".repeat(64);
        let identity =
            compatibility_identity(&artifact, Some(&mtp), 8192, 15, 2560).expect("identity");
        let text = String::from_utf8(identity).expect("text");
        assert!(text.starts_with("schema=gufo-text-continuation-v2\nmodel_kind=gemma4\n"));
        assert!(text.contains(&format!("artifact_id=gguf-sha256-v1:{artifact}\n")));
        assert!(text.contains("sliding_ring_slots=2560\ncontext_tokens=8192\n"));
        assert!(text.contains(&format!(
            "draft_backend=gemma4-assistant-mtp-v1\ndraft_artifact_id=gguf-sha256-v1:{mtp}\n\
             draft_tokens=15\n"
        )));
        let plain = compatibility_identity(&artifact, None, 8192, 15, 2560).expect("identity");
        assert!(!String::from_utf8(plain).expect("text").contains("draft_"));
        assert!(compatibility_identity("not-a-digest", None, 8192, 15, 2560).is_err());
        assert!(compatibility_identity(&artifact, Some("x"), 8192, 15, 2560).is_err());
    }
}
