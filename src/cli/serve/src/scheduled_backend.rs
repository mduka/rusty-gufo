//! The serving backend over one runner's scheduler: the facade half of
//! `InferenceBackend` (`src/cli/serve/inference_backend.cpp`). It tokenizes
//! or prepares prompts on the calling HTTP thread, submits them to the
//! scheduler and hands the request back; model loading is the caller's.

use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use gufo_core::json;
use gufo_core::json_constraint::{
    ConstraintVocabulary, JsonConstraint, TokenConstraint, Tool, ToolFormat,
};
use gufo_core::sampling::{ReasoningBudget, SamplingConfig, SamplingOverrides, TextModelPreset};
use gufo_core::text::{
    CancellationCheck, ChatMessage, ChatRequest, ChatRole, Error, GenerationErrorCode,
    InitialOutputState, ReasoningMarkup, ReasoningOptions, Result, TextModelRunner, Token,
    ToolChoice,
};

use crate::backend::{
    GenerationRequest, GenerationResult, SamplingDefaults, SessionState, StreamCallbacks,
    TextGenerationBackend, TokenCallback,
};
use crate::logging;
use crate::pool::TextRunnerPool;
use crate::scheduler::{
    RequestMetadata, SchedulerRequest, TextGenerationScheduler, TextPrefillPolicy,
    TextSchedulerPolicy,
};

#[derive(Clone)]
struct Settings {
    model_id: String,
    extra_model_ids: Vec<String>,
    model_path: String,
    build_info: String,
    sampling_defaults: SamplingDefaults,
    reasoning_defaults: ReasoningOptions,
}

pub struct ScheduledBackend<R: TextModelRunner> {
    scheduler: Arc<TextGenerationScheduler<R>>,
    settings: RwLock<Settings>,
    constraints: Mutex<ConstraintBindings>,
}

/// The model vocabulary for grammar masks, built with the first structured
/// request, and the last grammars bound to it with their mask caches.
#[derive(Default)]
struct ConstraintBindings {
    vocabulary: Option<Arc<ConstraintVocabulary>>,
    bindings: Vec<Arc<TokenConstraint>>,
    /// Per token whether its text ends inside a UTF-8 character, for
    /// reasoning budgets.
    partial_utf8: Option<Arc<Vec<bool>>>,
    /// Per token whether its text is only whitespace, for budgets that wait
    /// for the output to open its reasoning.
    blank: Option<Arc<Vec<bool>>>,
}

/// A chat request prepared for constrained decoding.
struct Constrained {
    request: ChatRequest,
    sampling: SamplingConfig,
}

/// Builds the grammar-mask vocabulary before serving:
/// the first constrained request would otherwise build it inside its queue
/// time (about 30 ms for Flash-Next's 248,320 tokens). `None` when the runner
/// has no structured output; the first constrained request then reports why.
fn prepare_structured_output<R: TextModelRunner>(runner: &R) -> Option<Arc<ConstraintVocabulary>> {
    let resident = resident_kib();
    let started = Instant::now();
    let vocabulary = runner.build_constraint_vocabulary().ok()?;
    logging::info(
        "loader",
        &format!(
            "event=structured_output_prepared vocabulary_tokens={} prepare_ms={:.1} \
             rss_delta_kib={} {}",
            vocabulary.len(),
            started.elapsed().as_secs_f64() * 1e3,
            resident_kib() as i64 - resident as i64,
            logging::memory_status()
        ),
    );
    Some(Arc::new(vocabulary))
}

/// `VmRSS` of this process in KiB; 0 when unknown.
fn resident_kib() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("VmRSS:")?
                    .split_whitespace()
                    .next()?
                    .parse()
                    .ok()
            })
        })
        .unwrap_or(0)
}

/// Whether `request`'s output may open the model's reasoning itself in an
/// automatic output state ([`ReasoningMarkup::model_opens`]): not when the
/// request turned reasoning off.
fn model_opens_reasoning(markup: &ReasoningMarkup, request: &ChatRequest) -> bool {
    markup.model_opens && request.reasoning.enabled != Some(false)
}

/// Grammars kept bound, with their masks, for repeated schemas.
const BOUND_CONSTRAINTS: usize = 4;

/// Whether a tool definition declares `function.strict: true`.
fn strict_tool(definition_json: &str) -> Result<bool> {
    if definition_json.is_empty() {
        return Ok(false);
    }
    let definition = json::parse_str(definition_json)?;
    Ok(definition
        .find("function")
        .and_then(|function| function.find("strict"))
        .is_some_and(|strict| strict.as_bool(false)))
}

struct ScheduledGenerationRequest {
    request: SchedulerRequest,
    /// Where the output's reasoning ends and, for output that may open it
    /// itself, starts ([`ReasoningMarks`]).
    reasoning: ReasoningMarks,
    tool_format: Option<ToolFormat>,
    reasoning_markup: ReasoningMarkup,
}

impl GenerationRequest for ScheduledGenerationRequest {
    fn wait(&mut self, on_token: Option<TokenCallback<'_>>) -> Result<GenerationResult> {
        self.wait_streaming(StreamCallbacks {
            on_token,
            ..StreamCallbacks::default()
        })
    }

    fn wait_streaming(&mut self, callbacks: StreamCallbacks<'_>) -> Result<GenerationResult> {
        let mut result = self.request.wait_streaming(callbacks)?;
        count_reasoning(&mut result, &self.reasoning);
        Ok(result)
    }

    fn cancel(&self) {
        self.request.cancel();
    }

    fn tool_format(&self) -> Option<ToolFormat> {
        self.tool_format
    }

    fn reasoning_markup(&self) -> ReasoningMarkup {
        self.reasoning_markup
    }
}

/// The tokens of the reasoning delimiters a request's output is counted by:
/// `end` when the output starts in reasoning or may open it, and `start`
/// when the output opens it itself. Both empty: no reasoning.
#[derive(Debug, Clone, Default)]
struct ReasoningMarks {
    end: Vec<Token>,
    start: Vec<Token>,
}

/// The generated tokens before the closing reasoning delimiter, all of them
/// while it is missing; nothing without a reasoning start, or when output
/// that may open its reasoning did not.
fn count_reasoning(result: &mut GenerationResult, reasoning: &ReasoningMarks) {
    let end = &reasoning.end;
    if end.is_empty() || !result.tokens.starts_with(&reasoning.start) {
        return;
    }
    result.reasoning_tokens = result
        .tokens
        .windows(end.len())
        .position(|window| window == end.as_slice())
        .unwrap_or(result.tokens.len());
}

impl<R: TextModelRunner> ScheduledBackend<R> {
    /// The reasoning delimiters as the template renders them: the end when
    /// `request`'s output starts in reasoning, and also the start when the
    /// model may open its reasoning itself ([`ReasoningMarkup::model_opens`]).
    fn reasoning_marks(&self, request: &ChatRequest) -> Result<ReasoningMarks> {
        let runner = self.scheduler.runner();
        let markup = runner.reasoning_markup();
        match runner.initial_output_state(request) {
            InitialOutputState::Reasoning => Ok(ReasoningMarks {
                end: runner.tokenize_control(markup.end)?,
                start: Vec::new(),
            }),
            InitialOutputState::Auto if model_opens_reasoning(&markup, request) => {
                Ok(ReasoningMarks {
                    end: runner.tokenize_control(markup.end)?,
                    start: runner.tokenize_control(markup.start)?,
                })
            }
            _ => Ok(ReasoningMarks::default()),
        }
    }

    /// `sampling` with the request's reasoning budget when its output starts
    /// in reasoning; unchanged otherwise.
    fn with_reasoning_budget(
        &self,
        request: &ChatRequest,
        sampling: &SamplingConfig,
        prompt: &[Token],
    ) -> Result<Option<SamplingConfig>> {
        let Some(tokens) = request.reasoning.budget_tokens else {
            return Ok(None);
        };
        let runner = self.scheduler.runner();
        let opens = match runner.initial_output_state(request) {
            InitialOutputState::Reasoning => false,
            InitialOutputState::Auto
                if model_opens_reasoning(&runner.reasoning_markup(), request) =>
            {
                true
            }
            _ => return Ok(None),
        };
        let Some(budget) = self.reasoning_budget(tokens, prompt, opens)? else {
            return Ok(None);
        };
        Ok(Some(SamplingConfig {
            reasoning_budget: Some(budget),
            ..sampling.clone()
        }))
    }

    /// A budget of `tokens` for a prompt that opens reasoning, or with
    /// `opens` for output that opens it itself at the start tag
    /// (`BuildReasoningBudget`); `None` when the model's tags are not single
    /// tokens.
    fn reasoning_budget(
        &self,
        tokens: usize,
        prompt: &[Token],
        opens: bool,
    ) -> Result<Option<Arc<ReasoningBudget>>> {
        let runner = self.scheduler.runner();
        let tags = runner.reasoning_budget_tags();
        let start = runner.tokenize_control(tags.start)?;
        let forced = runner.tokenize_control(tags.forced_end)?;
        if start.len() != 1 || forced.is_empty() {
            return Ok(None);
        }
        let mut ends = Vec::with_capacity(tags.ends.len());
        for end in tags.ends {
            match runner.tokenize_control(end)?[..] {
                [token] => ends.push(token),
                _ => return Ok(None),
            }
        }
        if opens {
            // An automatic output opens its own block; an opener in the
            // prompt belongs to an earlier turn.
            return Ok(Some(Arc::new(ReasoningBudget {
                tokens,
                forced,
                ends,
                prompt: Vec::new(),
                partial_utf8: self.partial_utf8()?,
                opener: Some(start[0]),
                blank: self.blank_tokens()?,
            })));
        }
        // llama.cpp counts the generation prompt after the start tag, such as
        // the newline after Qwen's <think>.
        let opened = prompt
            .iter()
            .rposition(|&token| token == start[0])
            .map_or(&[][..], |at| &prompt[at + 1..]);
        Ok(Some(Arc::new(ReasoningBudget {
            tokens,
            forced,
            ends,
            prompt: opened.to_vec(),
            partial_utf8: self.partial_utf8()?,
            opener: None,
            blank: None,
        })))
    }

    /// Which tokens end inside a UTF-8 character, from the grammar-mask
    /// vocabulary; `None` when the runner cannot enumerate it.
    fn partial_utf8(&self) -> Result<Option<Arc<Vec<bool>>>> {
        let mut constraints = self.constraints.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(partial) = &constraints.partial_utf8 {
            return Ok(Some(partial.clone()));
        }
        let Some(vocabulary) = self.vocabulary(&mut constraints) else {
            return Ok(None);
        };
        let partial = Arc::new(vocabulary.partial_utf8());
        constraints.partial_utf8 = Some(partial.clone());
        Ok(Some(partial))
    }

    /// Which tokens are only whitespace, from the grammar-mask vocabulary;
    /// `None` when the runner cannot enumerate it.
    fn blank_tokens(&self) -> Result<Option<Arc<Vec<bool>>>> {
        let mut constraints = self.constraints.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(blank) = &constraints.blank {
            return Ok(Some(blank.clone()));
        }
        let Some(vocabulary) = self.vocabulary(&mut constraints) else {
            return Ok(None);
        };
        let blank = Arc::new(vocabulary.blank());
        constraints.blank = Some(blank.clone());
        Ok(Some(blank))
    }

    /// The grammar-mask vocabulary, built on first use; `None` when the
    /// runner cannot enumerate it.
    fn vocabulary(
        &self,
        constraints: &mut ConstraintBindings,
    ) -> Option<Arc<ConstraintVocabulary>> {
        if let Some(vocabulary) = &constraints.vocabulary {
            return Some(vocabulary.clone());
        }
        let vocabulary = Arc::new(self.scheduler.runner().build_constraint_vocabulary().ok()?);
        constraints.vocabulary = Some(vocabulary.clone());
        Some(vocabulary)
    }

    /// Starts the scheduler over `pool`; the model ID is the runner's until
    /// [`Self::set_model_ids`].
    pub fn new(
        pool: Arc<TextRunnerPool<R>>,
        prefill_policy: TextPrefillPolicy,
        scheduler_policy: TextSchedulerPolicy,
    ) -> Result<Self> {
        let model_id = pool.runner().descriptor().model_id;
        let scheduler = Arc::new(TextGenerationScheduler::new(
            pool,
            prefill_policy,
            scheduler_policy,
        )?);
        let constraints = ConstraintBindings {
            vocabulary: prepare_structured_output(&**scheduler.runner()),
            bindings: Vec::new(),
            partial_utf8: None,
            blank: None,
        };
        Ok(Self {
            scheduler,
            settings: RwLock::new(Settings {
                model_id,
                extra_model_ids: Vec::new(),
                model_path: String::new(),
                build_info: String::new(),
                sampling_defaults: SamplingDefaults::default(),
                reasoning_defaults: ReasoningOptions::default(),
            }),
            constraints: Mutex::new(constraints),
        })
    }

    /// `grammar` over the model vocabulary, reusing the binding (and its
    /// cached masks) of the same compiled grammar (`BindConstraint`).
    fn bind_constraint(&self, grammar: Arc<JsonConstraint>) -> Result<Arc<TokenConstraint>> {
        let mut constraints = self.constraints.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(binding) = constraints
            .bindings
            .iter()
            .find(|binding| Arc::ptr_eq(binding.grammar(), &grammar))
        {
            return Ok(binding.clone());
        }
        let vocabulary = match &constraints.vocabulary {
            Some(vocabulary) => vocabulary.clone(),
            None => {
                let vocabulary = Arc::new(self.scheduler.runner().build_constraint_vocabulary()?);
                constraints.vocabulary = Some(vocabulary.clone());
                vocabulary
            }
        };
        let binding = Arc::new(TokenConstraint::new(grammar, vocabulary));
        if constraints.bindings.len() >= BOUND_CONSTRAINTS {
            constraints.bindings.remove(0);
        }
        constraints.bindings.push(binding.clone());
        Ok(binding)
    }

    /// A structured request with its instructions in the system message,
    /// `sampling` bound to its grammar and the tool format it admits; `None`
    /// when neither a response format nor tool calls apply
    /// (`ConstrainChatRequest`). Every tool request is constrained: calls take
    /// the model's native syntax, or the JSON envelope when a schema cannot be
    /// spelled in native parameter tags.
    fn constrain(
        &self,
        request: &ChatRequest,
        sampling: &SamplingConfig,
    ) -> Result<Option<Constrained>> {
        let tools_apply = !request.tools.is_empty() && request.tool_choice != ToolChoice::None;
        let response_format = request.response_format.as_ref().map(|f| f.0.clone());
        if response_format.is_none() && !tools_apply {
            return Ok(None);
        }
        let mut instruction = response_format
            .as_ref()
            .map_or_else(String::new, |format| format.prompt().to_owned());
        let mut grammar = response_format.clone();
        if tools_apply {
            let required = request.tool_choice == ToolChoice::Required;
            // A native model keeps its call syntax for every schema: what the tags cannot carry is best effort.
            let format = self.scheduler.runner().tool_format();
            let mut parameters = Vec::with_capacity(request.tools.len());
            for tool in &request.tools {
                let enforce = strict_tool(&tool.definition_json)?;
                let schema = json::parse_str(&tool.parameters_json)?;
                parameters.push(
                    JsonConstraint::tool_parameters(&schema, enforce, format)?
                        .ok_or_else(|| Error::logic("tool parameters always compile"))?,
                );
            }
            let tools: Vec<Tool> = request
                .tools
                .iter()
                .map(|tool| tool.name.clone())
                .zip(parameters)
                .collect();
            grammar = JsonConstraint::with_tools(
                grammar,
                tools,
                required,
                response_format.is_none() && request.parallel_tool_calls,
                format,
            )?;
            if format == ToolFormat::Json {
                instruction.push_str(
                    "\nIf a tool is needed, respond using the JSON tool-call form \
                     <tool_call>{\"name\":\"function_name\",\"arguments\":{...}}</tool_call>. \
                     Tool arguments must follow the chosen function's schema.",
                );
            }
            if response_format.is_some() {
                instruction.push_str(" The JSON response schema applies to the final answer.");
            }
        }
        if !request.response_format_description.is_empty() {
            instruction.insert_str(0, &format!("{}\n\n", request.response_format_description));
        }
        let mut constrained = request.clone();
        // Native constraints follow the model's own template: they change no
        // prompt token and keep continuation checkpoints valid. Instructions
        // are the server's framing, not client content.
        if !instruction.is_empty() {
            match constrained.messages.first_mut() {
                Some(first) if matches!(first.role, ChatRole::System | ChatRole::Developer) => {
                    first.framing_suffix.push_str("\n\n");
                    first.framing_suffix.push_str(&instruction);
                }
                _ => constrained.messages.insert(
                    0,
                    ChatMessage {
                        framing_suffix: instruction,
                        ..ChatMessage::new(ChatRole::System, "")
                    },
                ),
            }
        }
        let mut grammar =
            grammar.ok_or_else(|| Error::logic("structured request has no grammar"))?;
        // The reasoning phase uses the model's delimiters, as the response
        // parsers do; output that may open it itself keeps that choice.
        let runner = self.scheduler.runner();
        let markup = runner.reasoning_markup();
        match runner.initial_output_state(request) {
            InitialOutputState::Reasoning => {
                grammar = JsonConstraint::with_reasoning_markup(grammar, markup.end, "")?;
            }
            InitialOutputState::Auto if model_opens_reasoning(&markup, request) => {
                grammar = JsonConstraint::with_reasoning_markup(grammar, markup.end, markup.start)?;
            }
            _ => {}
        }
        let mut sampling = sampling.clone();
        sampling.constraint = Some(self.bind_constraint(grammar)?);
        Ok(Some(Constrained {
            request: constrained,
            sampling,
        }))
    }

    pub fn scheduler(&self) -> &Arc<TextGenerationScheduler<R>> {
        &self.scheduler
    }

    fn settings(&self) -> Settings {
        self.settings
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn update(&self, change: impl FnOnce(&mut Settings)) {
        change(&mut self.settings.write().unwrap_or_else(|e| e.into_inner()));
    }

    /// Overrides the public model IDs; empty keeps the current one. The first
    /// name is primary, the rest are additional served names.
    pub fn set_model_ids(&self, mut model_ids: Vec<String>) {
        if model_ids.is_empty() {
            return;
        }
        let extra = model_ids.split_off(1);
        self.update(|settings| {
            settings.model_id = model_ids.remove(0);
            settings.extra_model_ids = extra;
        });
    }

    /// The served model file and build, as `/props` reports them.
    pub fn set_served_artifact(&self, model_path: &str, build_info: &str) {
        self.update(|settings| {
            settings.model_path = model_path.to_owned();
            settings.build_info = build_info.to_owned();
        });
    }

    /// The server defaults: `max_tokens`, and `sampling` for the fields in
    /// `supplied`; the others follow the model preset
    /// ([`Self::set_sampling_preset`]).
    pub fn set_sampling_defaults(
        &self,
        max_tokens: usize,
        sampling: &SamplingConfig,
        supplied: SamplingOverrides,
    ) -> Result<()> {
        sampling.validate()?;
        self.update(|settings| {
            let defaults = &mut settings.sampling_defaults;
            defaults.max_tokens = max_tokens;
            defaults.sampling = sampling.clone();
            defaults.supplied = supplied;
        });
        Ok(())
    }

    /// The model whose recommended sampling fills unsupplied fields.
    pub fn set_sampling_preset(&self, model: TextModelPreset) {
        self.update(|settings| settings.sampling_defaults.model = model);
    }

    pub fn set_reasoning_defaults(&self, reasoning: ReasoningOptions) {
        self.update(|settings| settings.reasoning_defaults = reasoning);
    }

    /// `GenerateScheduled`: an empty prompt or a request cancelled before
    /// submission returns without generating.
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &self,
        prompt: Vec<Token>,
        request_start: Instant,
        max_tokens: usize,
        sampling: &SamplingConfig,
        is_cancelled: Option<&CancellationCheck>,
        on_token: Option<TokenCallback<'_>>,
        client_id: &str,
        prompt_context: Option<Arc<R::PromptContext>>,
        cache_prompt: bool,
        cache_persist: bool,
        cache_prefix_tokens: usize,
        stop_sequences: &[String],
    ) -> Result<GenerationResult> {
        let mut result = GenerationResult {
            prompt_tokens: prompt.len(),
            client_id: if client_id.is_empty() {
                "anonymous".into()
            } else {
                client_id.to_owned()
            },
            ..GenerationResult::default()
        };
        if prompt.is_empty() {
            return Ok(result);
        }
        if is_cancelled.is_some_and(|check| check()) {
            result.cancelled = true;
            return Ok(result);
        }
        let mut request = self.scheduler.submit(
            prompt,
            max_tokens,
            sampling,
            is_cancelled.cloned(),
            on_token.is_some(),
            RequestMetadata {
                client_id: client_id.to_owned(),
                deadline: None,
                request_start,
                prompt_context,
                cache_prompt,
                cache_persist,
                cache_prefix_tokens,
                stop_sequences: stop_sequences.to_vec(),
                stop_at_eos: true,
                return_progress: false,
            },
        )?;
        request.wait(on_token)
    }
}

impl<R: TextModelRunner> TextGenerationBackend for ScheduledBackend<R> {
    fn model_id(&self) -> String {
        self.settings().model_id
    }

    fn model_ids(&self) -> Vec<String> {
        let settings = self.settings();
        let mut ids = vec![settings.model_id];
        ids.extend(settings.extra_model_ids);
        ids
    }

    fn ready(&self) -> bool {
        true
    }

    fn max_context(&self) -> u32 {
        self.scheduler.max_context()
    }

    fn session_states(&self) -> Vec<SessionState> {
        self.scheduler.session_states()
    }

    fn sampling_defaults(&self) -> SamplingDefaults {
        self.settings().sampling_defaults
    }

    fn reasoning_defaults(&self) -> ReasoningOptions {
        self.settings().reasoning_defaults
    }

    fn initial_output_state(&self, request: &ChatRequest) -> InitialOutputState {
        self.scheduler.runner().initial_output_state(request)
    }

    fn supports_images(&self) -> bool {
        self.scheduler.runner().supports_images()
    }

    fn model_path(&self) -> String {
        self.settings().model_path
    }

    fn build_info(&self) -> String {
        self.settings().build_info
    }

    fn device_lost(&self) -> bool {
        self.scheduler.device_lost()
    }

    fn fatal_failure(&self) -> Option<GenerationErrorCode> {
        self.scheduler.fatal_failure()
    }

    fn complete(
        &self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingConfig,
        is_cancelled: Option<&CancellationCheck>,
        on_token: Option<TokenCallback<'_>>,
        client_id: &str,
        stop_sequences: &[String],
        cache_prompt: bool,
        cache_persist: bool,
    ) -> Result<GenerationResult> {
        let request_start = Instant::now();
        let tokens = self.scheduler.runner().tokenize(prompt)?;
        self.generate(
            tokens,
            request_start,
            max_tokens,
            sampling,
            is_cancelled,
            on_token,
            client_id,
            None,
            cache_prompt,
            cache_persist,
            0,
            stop_sequences,
        )
    }

    fn chat(
        &self,
        request: &ChatRequest,
        max_tokens: usize,
        sampling: &SamplingConfig,
        is_cancelled: Option<&CancellationCheck>,
        on_token: Option<TokenCallback<'_>>,
    ) -> Result<GenerationResult> {
        let request_start = Instant::now();
        let constrained = self.constrain(request, sampling)?;
        let (request, sampling) = match &constrained {
            Some(constrained) => (&constrained.request, &constrained.sampling),
            None => (request, sampling),
        };
        let Some(prompt) = self.scheduler.runner().prepare_prompt(request)? else {
            return Ok(GenerationResult::default());
        };
        if prompt.tokens.is_empty() {
            return Ok(GenerationResult::default());
        }
        let budgeted = self.with_reasoning_budget(request, sampling, &prompt.tokens)?;
        let sampling = budgeted.as_ref().unwrap_or(sampling);
        let reasoning = self.reasoning_marks(request)?;
        let mut result = self.generate(
            prompt.tokens,
            request_start,
            max_tokens,
            sampling,
            is_cancelled,
            on_token,
            &request.client_id,
            prompt.context,
            request.cache_prompt,
            request.cache_persist,
            prompt.cache_prefix_tokens,
            &request.stop_sequences,
        )?;
        count_reasoning(&mut result, &reasoning);
        Ok(result)
    }

    fn start_chat(
        &self,
        request: &ChatRequest,
        max_tokens: usize,
        sampling: &SamplingConfig,
        is_cancelled: Option<CancellationCheck>,
        stream_output: bool,
    ) -> Result<Option<Box<dyn GenerationRequest>>> {
        let request_start = Instant::now();
        let constrained = self.constrain(request, sampling)?;
        let (request, sampling) = match &constrained {
            Some(constrained) => (&constrained.request, &constrained.sampling),
            None => (request, sampling),
        };
        let Some(prompt) = self.scheduler.runner().prepare_prompt(request)? else {
            return Ok(None);
        };
        if prompt.tokens.is_empty() {
            return Ok(None);
        }
        let budgeted = self.with_reasoning_budget(request, sampling, &prompt.tokens)?;
        let sampling = budgeted.as_ref().unwrap_or(sampling);
        let client_id = if request.client_id.is_empty() {
            "anonymous".to_owned()
        } else {
            request.client_id.clone()
        };
        let scheduled = self.scheduler.submit(
            prompt.tokens,
            max_tokens,
            sampling,
            is_cancelled,
            stream_output,
            RequestMetadata {
                client_id,
                deadline: None,
                request_start,
                prompt_context: prompt.context,
                cache_prompt: request.cache_prompt,
                cache_persist: request.cache_persist,
                cache_prefix_tokens: prompt.cache_prefix_tokens,
                stop_sequences: request.stop_sequences.clone(),
                stop_at_eos: true,
                return_progress: request.return_progress,
            },
        )?;
        Ok(Some(Box::new(ScheduledGenerationRequest {
            request: scheduled,
            reasoning: self.reasoning_marks(request)?,
            // Parsing needs the model's dialect even without tool
            // constraints, e.g. Qwen's separator after `</think>`.
            tool_format: Some(self.scheduler.runner().tool_format()),
            reasoning_markup: self.scheduler.runner().reasoning_markup(),
        })))
    }

    fn start_complete(
        &self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingConfig,
        is_cancelled: Option<CancellationCheck>,
        stream_output: bool,
        ignore_eos: bool,
        client_id: &str,
        stop_sequences: &[String],
        return_progress: bool,
        cache_prompt: bool,
        cache_persist: bool,
    ) -> Result<Option<Box<dyn GenerationRequest>>> {
        let request_start = Instant::now();
        let tokens = self.scheduler.runner().tokenize(prompt)?;
        let scheduled = self.scheduler.submit(
            tokens,
            max_tokens,
            sampling,
            is_cancelled,
            stream_output,
            RequestMetadata {
                client_id: if client_id.is_empty() {
                    "anonymous".to_owned()
                } else {
                    client_id.to_owned()
                },
                deadline: None,
                request_start,
                prompt_context: None,
                cache_prompt,
                cache_persist,
                cache_prefix_tokens: 0,
                stop_sequences: stop_sequences.to_vec(),
                stop_at_eos: !ignore_eos,
                return_progress,
            },
        )?;
        Ok(Some(Box::new(ScheduledGenerationRequest {
            request: scheduled,
            reasoning: ReasoningMarks::default(),
            tool_format: None,
            reasoning_markup: self.scheduler.runner().reasoning_markup(),
        })))
    }

    fn count_tokens(&self, text: &str) -> usize {
        // C++ lets a tokenizer failure escape; no route counts tokens.
        self.scheduler
            .runner()
            .tokenize(text)
            .map_or(0, |tokens| tokens.len())
    }

    fn count_chat_tokens(&self, request: &ChatRequest) -> Result<Option<usize>> {
        // The preparation of chat(): constrained tool or format requests add
        // prompt text before the template renders. Counting happens before
        // context admission, as in llama.cpp, so a client can size an
        // oversized prompt; generation still enforces the context.
        let constrained = self.constrain(request, &SamplingConfig::default())?;
        let request = constrained.as_ref().map_or(request, |c| &c.request);
        self.scheduler.runner().count_prompt_tokens(request)
    }

    fn render_chat_prompt(&self, request: &ChatRequest) -> Result<Option<String>> {
        // The same preparation as count_chat_tokens(), so the text is what
        // chat() would tokenize.
        let constrained = self.constrain(request, &SamplingConfig::default())?;
        let request = constrained.as_ref().map_or(request, |c| &c.request);
        self.scheduler.runner().render_prompt(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_ends_at_the_first_closing_delimiter() {
        let opened = |tokens: &[Token], end: &[Token], start: &[Token]| {
            let mut result = GenerationResult {
                tokens: tokens.to_vec(),
                ..GenerationResult::default()
            };
            let marks = ReasoningMarks {
                end: end.to_vec(),
                start: start.to_vec(),
            };
            count_reasoning(&mut result, &marks);
            result.reasoning_tokens
        };
        let count = |tokens: &[Token], end: &[Token]| opened(tokens, end, &[]);
        assert_eq!(count(&[1, 2, 7, 8, 3, 7, 8], &[7, 8]), 2);
        assert_eq!(count(&[1, 2, 7], &[7, 8]), 3, "unclosed reasoning");
        assert_eq!(count(&[7, 8, 1], &[7, 8]), 0);
        assert_eq!(count(&[1, 2], &[]), 0, "output without reasoning");
        // Output that may open its reasoning counts it only when it does,
        // its start tokens included.
        assert_eq!(opened(&[5, 6, 1, 7, 8, 2], &[7, 8], &[5, 6]), 3);
        assert_eq!(
            opened(&[1, 7, 8, 2], &[7, 8], &[5, 6]),
            0,
            "answered directly"
        );
        assert_eq!(
            opened(&[5, 6, 1], &[7, 8], &[5, 6]),
            3,
            "unclosed reasoning"
        );
    }
}
