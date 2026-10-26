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
    InitialOutputState, ReasoningOptions, Result, TextModelRunner, Token, ToolChoice,
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
    /// The tokens of `</think>` when the output starts in reasoning.
    reasoning_end: Vec<Token>,
    tool_format: Option<ToolFormat>,
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
        count_reasoning(&mut result, &self.reasoning_end);
        Ok(result)
    }

    fn cancel(&self) {
        self.request.cancel();
    }

    fn tool_format(&self) -> Option<ToolFormat> {
        self.tool_format
    }
}

/// The generated tokens before the closing reasoning delimiter, all of them
/// while it is missing; nothing without a reasoning start.
fn count_reasoning(result: &mut GenerationResult, reasoning_end: &[Token]) {
    if reasoning_end.is_empty() {
        return;
    }
    result.reasoning_tokens = result
        .tokens
        .windows(reasoning_end.len())
        .position(|window| window == reasoning_end)
        .unwrap_or(result.tokens.len());
}

impl<R: TextModelRunner> ScheduledBackend<R> {
    /// `</think>`, as the template renders it, when `request`'s output
    /// starts in reasoning.
    fn reasoning_end(&self, request: &ChatRequest) -> Result<Vec<Token>> {
        let runner = self.scheduler.runner();
        if runner.initial_output_state(request) == InitialOutputState::Reasoning {
            runner.tokenize_control("</think>")
        } else {
            Ok(Vec::new())
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
        if self.scheduler.runner().initial_output_state(request) != InitialOutputState::Reasoning {
            return Ok(None);
        }
        let Some(budget) = self.reasoning_budget(tokens, prompt)? else {
            return Ok(None);
        };
        Ok(Some(SamplingConfig {
            reasoning_budget: Some(budget),
            ..sampling.clone()
        }))
    }

    /// A budget of `tokens` for a prompt that opens reasoning
    /// (`BuildReasoningBudget`); `None` when the model's tags are not single
    /// tokens.
    fn reasoning_budget(
        &self,
        tokens: usize,
        prompt: &[Token],
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
        })))
    }

    /// Which tokens end inside a UTF-8 character, from the grammar-mask
    /// vocabulary; `None` when the runner cannot enumerate it.
    fn partial_utf8(&self) -> Result<Option<Arc<Vec<bool>>>> {
        let mut constraints = self.constraints.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(partial) = &constraints.partial_utf8 {
            return Ok(Some(partial.clone()));
        }
        let vocabulary = match &constraints.vocabulary {
            Some(vocabulary) => vocabulary.clone(),
            None => match self.scheduler.runner().build_constraint_vocabulary() {
                Ok(vocabulary) => {
                    let vocabulary = Arc::new(vocabulary);
                    constraints.vocabulary = Some(vocabulary.clone());
                    vocabulary
                }
                Err(_) => return Ok(None),
            },
        };
        let partial = Arc::new(vocabulary.partial_utf8());
        constraints.partial_utf8 = Some(partial.clone());
        Ok(Some(partial))
    }

    /// Starts the scheduler over `pool`; the model ID is the runner's until
    /// [`Self::set_model_id`].
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
        };
        Ok(Self {
            scheduler,
            settings: RwLock::new(Settings {
                model_id,
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
        if self.scheduler.runner().initial_output_state(request) == InitialOutputState::Reasoning {
            grammar = JsonConstraint::with_reasoning(grammar);
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

    /// Overrides the public model ID; empty keeps the current one.
    pub fn set_model_id(&self, model_id: &str) {
        if !model_id.is_empty() {
            self.update(|settings| settings.model_id = model_id.to_owned());
        }
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
        let reasoning_end = self.reasoning_end(request)?;
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
            prompt.cache_prefix_tokens,
            &request.stop_sequences,
        )?;
        count_reasoning(&mut result, &reasoning_end);
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
                cache_prefix_tokens: prompt.cache_prefix_tokens,
                stop_sequences: request.stop_sequences.clone(),
                stop_at_eos: true,
                return_progress: request.return_progress,
            },
        )?;
        Ok(Some(Box::new(ScheduledGenerationRequest {
            request: scheduled,
            reasoning_end: self.reasoning_end(request)?,
            // Parsing needs the model's dialect even without tool
            // constraints, e.g. Qwen's separator after `</think>`.
            tool_format: Some(self.scheduler.runner().tool_format()),
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
                cache_prefix_tokens: 0,
                stop_sequences: stop_sequences.to_vec(),
                stop_at_eos: !ignore_eos,
                return_progress,
            },
        )?;
        Ok(Some(Box::new(ScheduledGenerationRequest {
            request: scheduled,
            reasoning_end: Vec::new(),
            tool_format: None,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_ends_at_the_first_closing_delimiter() {
        let count = |tokens: &[Token], end: &[Token]| {
            let mut result = GenerationResult {
                tokens: tokens.to_vec(),
                ..GenerationResult::default()
            };
            count_reasoning(&mut result, end);
            result.reasoning_tokens
        };
        assert_eq!(count(&[1, 2, 7, 8, 3, 7, 8], &[7, 8]), 2);
        assert_eq!(count(&[1, 2, 7], &[7, 8]), 3, "unclosed reasoning");
        assert_eq!(count(&[7, 8, 1], &[7, 8]), 0);
        assert_eq!(count(&[1, 2], &[]), 0, "output without reasoning");
    }
}
