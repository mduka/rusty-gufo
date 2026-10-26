//! Port of `tests/cli/openai_chat_test.cpp`. The two cases that also render
//! the Qwen chat template keep their API assertions here; the template half
//! is in `gufo-qwen`'s `chat_template` tests.

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::Duration;

use gufo_core::json::{self, Value};
use gufo_core::json_constraint::JsonConstraint;
use gufo_core::sampling::SamplingConfig;
use gufo_core::text::{
    CancellationCheck, ChatRequest, Error, GenerationErrorCode, ReasoningEffort, ReasoningOptions,
    Result, ToolChoice,
};
use gufo_serve::backend::{
    FinishReason, GenerationRequest, GenerationResult, InitialOutputState, SamplingDefaults,
    TextGenerationBackend, TokenCallback, ToolFormat,
};
use gufo_serve::generation_metrics::prefill_tokens_per_second;
use gufo_serve::http::{HttpRequest, HttpResponse};
use gufo_serve::openai_chat::{
    ResponsesReasoningTarget, create_anthropic_message, create_openai_response, handle_openai_chat,
    parse_tool_controls,
};

struct Config {
    pieces: Vec<Vec<u8>>,
    cache_miss_reason: String,
    finish_reason: FinishReason,
    stop_sequence: String,
    block_after_first_piece: bool,
    defaults: SamplingDefaults,
    reasoning_defaults: ReasoningOptions,
    reject_on_start: Option<GenerationErrorCode>,
    /// The tool format generations report as admitted; `None` reports no
    /// format metadata.
    tool_format: Option<ToolFormat>,
    /// The phase the output starts in, instead of the one the request's
    /// reasoning selects.
    initial: Option<InitialOutputState>,
    image_support: bool,
}

#[derive(Default)]
struct Observed {
    last_request: ChatRequest,
    last_max_tokens: usize,
    last_sampling: SamplingConfig,
}

#[derive(Default)]
struct Gate {
    first_piece_emitted: bool,
    released: bool,
}

struct FakeBackend {
    this: Weak<FakeBackend>,
    config: Mutex<Config>,
    observed: Mutex<Observed>,
    completed: AtomicBool,
    chat_calls: AtomicI32,
    gate: Mutex<Gate>,
    condition: Condvar,
}

impl FakeBackend {
    fn new() -> Arc<Self> {
        Arc::new_cyclic(|this| Self {
            this: this.clone(),
            config: Mutex::new(Config {
                pieces: Vec::new(),
                cache_miss_reason: String::new(),
                finish_reason: FinishReason::Stop,
                stop_sequence: String::new(),
                block_after_first_piece: false,
                defaults: SamplingDefaults::default(),
                reasoning_defaults: ReasoningOptions::default(),
                reject_on_start: None,
                tool_format: None,
                initial: None,
                image_support: true,
            }),
            observed: Mutex::new(Observed::default()),
            completed: AtomicBool::new(false),
            chat_calls: AtomicI32::new(0),
            gate: Mutex::new(Gate::default()),
            condition: Condvar::new(),
        })
    }

    fn config(&self) -> MutexGuard<'_, Config> {
        self.config.lock().expect("config")
    }

    fn set_pieces(&self, pieces: &[&str]) {
        self.config().pieces = pieces.iter().map(|p| p.as_bytes().to_vec()).collect();
    }

    fn last_request(&self) -> ChatRequest {
        self.observed.lock().expect("observed").last_request.clone()
    }

    fn wait_for_first_piece(&self) -> bool {
        let gate = self.gate.lock().expect("gate");
        let (gate, _) = self
            .condition
            .wait_timeout_while(gate, Duration::from_secs(2), |g| !g.first_piece_emitted)
            .expect("gate");
        gate.first_piece_emitted
    }

    fn release(&self) {
        self.gate.lock().expect("gate").released = true;
        self.condition.notify_all();
    }
}

impl TextGenerationBackend for FakeBackend {
    fn model_id(&self) -> String {
        "test-model".into()
    }

    fn ready(&self) -> bool {
        true
    }

    fn sampling_defaults(&self) -> SamplingDefaults {
        self.config().defaults.clone()
    }

    fn reasoning_defaults(&self) -> ReasoningOptions {
        self.config().reasoning_defaults
    }

    fn supports_images(&self) -> bool {
        self.config().image_support
    }

    fn initial_output_state(&self, request: &ChatRequest) -> InitialOutputState {
        if let Some(initial) = self.config().initial {
            initial
        } else if request.reasoning.enabled.unwrap_or(false) {
            InitialOutputState::Reasoning
        } else {
            InitialOutputState::Content
        }
    }

    fn complete(
        &self,
        _: &str,
        _: usize,
        _: &SamplingConfig,
        _: Option<&CancellationCheck>,
        _: Option<&mut dyn FnMut(&[u8]) -> bool>,
        _: &str,
        _: &[String],
        _: bool,
    ) -> Result<GenerationResult> {
        Ok(GenerationResult::default())
    }

    fn chat(
        &self,
        request: &ChatRequest,
        max_tokens: usize,
        sampling: &SamplingConfig,
        is_cancelled: Option<&CancellationCheck>,
        mut on_token: Option<&mut dyn FnMut(&[u8]) -> bool>,
    ) -> Result<GenerationResult> {
        self.chat_calls.fetch_add(1, Ordering::SeqCst);
        {
            let mut observed = self.observed.lock().expect("observed");
            observed.last_request = request.clone();
            observed.last_max_tokens = max_tokens;
            observed.last_sampling = sampling.clone();
        }
        let (pieces, miss, finish, stop_sequence, block) = {
            let config = self.config();
            (
                config.pieces.clone(),
                config.cache_miss_reason.clone(),
                config.finish_reason,
                config.stop_sequence.clone(),
                config.block_after_first_piece,
            )
        };
        let mut result = GenerationResult {
            prompt_tokens: 7,
            cached_prompt_tokens: 5,
            prefill_tokens: 2,
            prefill_chunks: 1,
            queue_depth_at_submit: 3,
            client_queue_depth_at_submit: 1,
            resident_requests_at_admission: 2,
            requested_logical_concurrency: 4,
            physical_execution_width: 2,
            queue_ms: 1.25,
            prefill_ms: 2.5,
            decode_ms: 4.0,
            ttft_ms: 3.75,
            mean_inter_token_ms: 2.0,
            max_inter_token_ms: 2.5,
            execution_plan: "serial-fallback".into(),
            cache_hit: miss.is_empty(),
            cache_miss_reason: miss,
            ..GenerationResult::default()
        };
        if !result.cache_hit {
            result.cached_prompt_tokens = 0;
            result.prefill_tokens = result.prompt_tokens;
            result.cache_common_prefix_tokens = 2;
            result.cache_checkpoint_tokens = 5;
        }
        for piece in pieces {
            let cancelled = is_cancelled.is_some_and(|c| c());
            let refused = !cancelled && on_token.as_mut().is_some_and(|callback| !callback(&piece));
            if cancelled || refused {
                result.cancelled = true;
                result.finish_reason = FinishReason::Cancelled;
                return Ok(result);
            }
            result.text.extend_from_slice(&piece);
            result.tokens.push(result.tokens.len() as u32);
            if block && result.tokens.len() == 1 {
                self.gate.lock().expect("gate").first_piece_emitted = true;
                self.condition.notify_all();
                let gate = self.gate.lock().expect("gate");
                let _gate = self
                    .condition
                    .wait_while(gate, |g| !g.released)
                    .expect("gate");
            }
        }
        result.completion_tokens = result.tokens.len();
        result.finish_reason = finish;
        result.stop_sequence = stop_sequence;
        self.completed.store(true, Ordering::SeqCst);
        Ok(result)
    }

    fn start_chat(
        &self,
        request: &ChatRequest,
        max_tokens: usize,
        sampling: &SamplingConfig,
        is_cancelled: Option<CancellationCheck>,
        _: bool,
    ) -> Result<Option<Box<dyn GenerationRequest>>> {
        let (reject, tool_format) = {
            let config = self.config();
            (config.reject_on_start, config.tool_format)
        };
        if let Some(code) = reject {
            return Err(Error::generation(code, "injected admission rejection"));
        }
        let Some(tool_format) = tool_format else {
            return Ok(None);
        };
        Ok(Some(Box::new(FormattedRequest {
            backend: self.this.upgrade().expect("backend"),
            request: request.clone(),
            max_tokens,
            sampling: sampling.clone(),
            is_cancelled,
            cancelled: Arc::new(AtomicBool::new(false)),
            tool_format,
        })))
    }

    fn count_tokens(&self, text: &str) -> usize {
        text.len()
    }
}

/// A chat generation that reports the tool format it was admitted with.
struct FormattedRequest {
    backend: Arc<FakeBackend>,
    request: ChatRequest,
    max_tokens: usize,
    sampling: SamplingConfig,
    is_cancelled: Option<CancellationCheck>,
    cancelled: Arc<AtomicBool>,
    tool_format: ToolFormat,
}

impl GenerationRequest for FormattedRequest {
    fn wait(&mut self, on_token: Option<TokenCallback<'_>>) -> Result<GenerationResult> {
        let cancelled = self.cancelled.clone();
        let external = self.is_cancelled.clone();
        let check: CancellationCheck = Arc::new(move || {
            cancelled.load(Ordering::Acquire) || external.as_ref().is_some_and(|c| c())
        });
        self.backend.chat(
            &self.request,
            self.max_tokens,
            &self.sampling,
            Some(&check),
            on_token,
        )
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    fn tool_format(&self) -> Option<ToolFormat> {
        Some(self.tool_format)
    }
}

fn request(body: &str) -> HttpRequest {
    HttpRequest {
        method: "POST".into(),
        path: "/v1/chat/completions".into(),
        body: body.as_bytes().to_vec(),
        ..HttpRequest::default()
    }
}

fn chat(backend: &Arc<FakeBackend>, body: &str) -> HttpResponse {
    let dynamic: Arc<dyn TextGenerationBackend> = backend.clone();
    handle_openai_chat(&request(body), &dynamic).expect("handler")
}

/// `body` (a Chat Completions request) through Responses: its tools in the
/// flat Responses shape, with `reasoning` on or off.
fn responses(
    backend: &Arc<FakeBackend>,
    body: &Value,
    reasoning: bool,
    max_tokens: usize,
    stream: bool,
) -> HttpResponse {
    let mut body = body.clone();
    if let Some(tools) = body.find("tools") {
        let flat: Vec<Value> = tools
            .items()
            .iter()
            .map(|tool| {
                tool.find("function")
                    .expect("function")
                    .clone()
                    .with("type", "function")
            })
            .collect();
        body.set("tools", flat);
    }
    let mut chat = ChatRequest::default();
    chat.reasoning.enabled = Some(reasoning);
    parse_tool_controls(&body, &mut chat, true).expect("valid Responses tool controls");
    let dynamic: Arc<dyn TextGenerationBackend> = backend.clone();
    create_openai_response(
        &request(&body.dump().expect("dump")),
        &dynamic,
        &chat,
        max_tokens,
        &SamplingConfig::default(),
        stream,
        ResponsesReasoningTarget::default(),
    )
    .expect("response")
}

/// The JSON events of a streamed body, or its buffered JSON body.
fn events(response: &mut HttpResponse, stream: bool) -> Vec<Value> {
    if !stream {
        return vec![json::parse(&response.body).expect("body")];
    }
    drain(response)
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| !data.starts_with("[DONE]"))
        .map(|data| json::parse_str(data).expect("event"))
        .collect()
}

fn body_text(response: &HttpResponse) -> String {
    String::from_utf8_lossy(&response.body).into_owned()
}

/// Runs a streamed body, collecting everything it writes.
fn drain(response: &mut HttpResponse) -> String {
    let mut output = Vec::new();
    if let Some(body) = response.streaming_body.take() {
        body(&mut |chunk: &[u8]| {
            output.extend_from_slice(chunk);
            true
        })
        .expect("stream");
    }
    String::from_utf8_lossy(&output).into_owned()
}

fn stream_details(response: &HttpResponse) -> String {
    response
        .stream_log
        .as_ref()
        .map(|log| log.lock().expect("log").details.clone())
        .unwrap_or_default()
}

const HELLO: &str = r#"{"model":"test-model","messages":[{"role":"user","content":"hello"}]}"#;

/// `HELLO` with `fields` set.
fn hello_with(fields: &[(&str, &str)]) -> String {
    let mut body = json::parse_str(HELLO).expect("json");
    for (field, value) in fields {
        body.set(field, json::parse_str(value).expect("json"));
    }
    body.dump().expect("json")
}

#[test]
fn stop_sequences_and_default_fields() {
    for (stop, count) in [
        ("null", 0),
        ("[]", 0),
        ("\"END\"", 1),
        ("[\"END\",\"終\"]", 2),
    ] {
        let backend = FakeBackend::new();
        backend.set_pieces(&["hello"]);
        let response = chat(&backend, &hello_with(&[("stop", stop)]));
        assert_eq!(response.status, 200, "supported stop forms are accepted");
        let sequences = backend.last_request().stop_sequences;
        assert_eq!(sequences.len(), count, "stop sequences reach the backend");
        assert!(count == 0 || sequences[0] == "END");
    }
    for stream in ["false", "true"] {
        for stop in [
            "true",
            "123",
            "{}",
            "\"\"",
            "[\"\"]",
            "[\"END\",null]",
            "[\"1\",\"2\",\"3\",\"4\",\"5\"]",
        ] {
            let backend = FakeBackend::new();
            let response = chat(&backend, &hello_with(&[("stream", stream), ("stop", stop)]));
            assert!(
                response.status == 400
                    && backend.chat_calls.load(Ordering::SeqCst) == 0
                    && response.streaming_body.is_none(),
                "invalid stops fail before generation or streaming headers: {stop}"
            );
        }
    }
    let backend = FakeBackend::new();
    let long = format!("\"{}\"", "s".repeat(4097));
    assert_eq!(
        chat(&backend, &hello_with(&[("stop", &long)])).status,
        400,
        "oversized stop sequences are bounded"
    );
    {
        let mut config = backend.config();
        config.defaults.max_tokens = 37;
        config.defaults.sampling.temperature = 0.7;
        config.defaults.sampling.top_p = 0.9;
        config.defaults.sampling.seed = 123;
        config.defaults.sampling.frequency_penalty = 0.5;
        config.defaults.sampling.presence_penalty = 0.3;
    }
    let nulls: Vec<(&str, &str)> = [
        "tools",
        "tool_choice",
        "stream",
        "stream_options",
        "n",
        "max_tokens",
        "max_completion_tokens",
        "logprobs",
        "top_logprobs",
        "response_format",
        "modalities",
        "audio",
        "temperature",
        "top_p",
        "seed",
        "logit_bias",
        "frequency_penalty",
        "presence_penalty",
        "reasoning_effort",
    ]
    .iter()
    .map(|field| (*field, "null"))
    .collect();
    assert_eq!(
        chat(&backend, &hello_with(&nulls)).status,
        200,
        "nullable API defaults do not enable unsupported features"
    );
    let defaults = backend.config().defaults.clone();
    {
        let observed = backend.observed.lock().expect("observed");
        assert_eq!(
            observed.last_max_tokens, defaults.max_tokens,
            "null token limits retain the configured default"
        );
        let (actual, expected) = (observed.last_sampling.clone(), defaults.sampling.clone());
        assert!(
            actual.temperature == expected.temperature
                && actual.top_p == expected.top_p
                && actual.seed == expected.seed
                && actual.frequency_penalty == expected.frequency_penalty
                && actual.presence_penalty == expected.presence_penalty,
            "null sampling fields retain server defaults"
        );
    }
    let mut explicit = nulls.clone();
    explicit.extend([
        ("logprobs", "false"),
        ("logit_bias", "{}"),
        ("response_format", r#"{"type":"text"}"#),
        ("modalities", r#"["text"]"#),
    ]);
    assert_eq!(
        chat(&backend, &hello_with(&explicit)).status,
        200,
        "explicit text-only and disabled logprobs defaults work"
    );
    for (field, value) in [
        ("logprobs", "true"),
        ("top_logprobs", "3"),
        ("modalities", r#"["text","audio"]"#),
        ("audio", "{}"),
        ("response_format", r#"{"type":"text","unexpected":true}"#),
    ] {
        let before = backend.chat_calls.load(Ordering::SeqCst);
        assert!(
            chat(&backend, &hello_with(&[(field, value)])).status == 400
                && backend.chat_calls.load(Ordering::SeqCst) == before,
            "actual unsupported feature requests remain explicit errors: {field}"
        );
    }
}

/// Content and reasoning deltas of a streamed body, or of the buffered
/// message.
/// The assistant content of a buffered body or of an event stream's deltas.
fn output_content(output: &str, stream: bool) -> String {
    if !stream {
        return json::parse_str(output)
            .expect("json")
            .find("choices")
            .expect("choices")
            .items()[0]
            .find("message")
            .expect("message")
            .member_str("content", "");
    }
    let mut content = String::new();
    for data in output
        .split("\n\n")
        .filter_map(|line| line.strip_prefix("data: "))
    {
        if data == "[DONE]" {
            continue;
        }
        let event = json::parse_str(data).expect("event");
        for choice in event.find("choices").map(Value::items).unwrap_or_default() {
            if let Some(delta) = choice.find("delta") {
                content.push_str(&delta.member_str("content", ""));
            }
        }
    }
    content
}

fn answer_and_reasoning(response: &mut HttpResponse, stream: bool) -> (String, String) {
    let (mut answer, mut reasoning) = (String::new(), String::new());
    if stream {
        for line in drain(response).split("\n\n") {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                continue;
            }
            let event = json::parse_str(data).expect("event");
            for choice in event.find("choices").map(Value::items).unwrap_or_default() {
                if let Some(delta) = choice.find("delta") {
                    answer.push_str(&delta.member_str("content", ""));
                    reasoning.push_str(&delta.member_str("reasoning_content", ""));
                }
            }
        }
    } else {
        let response = json::parse_str(&body_text(response)).expect("json");
        let message = response.find("choices").expect("choices").items()[0]
            .find("message")
            .expect("message")
            .clone();
        answer = message.member_str("content", "");
        reasoning = message.member_str("reasoning_content", "");
    }
    (answer, reasoning)
}

/// Qwen writes a newline between parallel calls and may end with one; that
/// layout is not a reply, so the turn has null content, as with one call
/// (`TestParallelCallSeparatorIsNotContent`).
#[test]
fn parallel_call_separator_is_not_content() {
    let alpha = "<tool_call>\n<function=get_alpha>\n</function>\n</tool_call>";
    let beta = "<tool_call>\n<function=get_beta>\n</function>\n</tool_call>";
    for thinking in [false, true] {
        for stream in [false, true] {
            for constrained in [false, true] {
                for separator in ["\n", "\n\n", " \n"] {
                    let backend = FakeBackend::new();
                    if constrained {
                        backend.config().tool_format = Some(ToolFormat::Qwen);
                    }
                    let raw = format!(
                        "{}{alpha}{separator}{beta}\n",
                        if thinking { "Plan.</think>\n\n" } else { "" }
                    );
                    backend.config().pieces = raw.bytes().map(|b| vec![b]).collect();
                    let body = format!(
                        r#"{{"model":"test-model","messages":[{{"role":"user","content":"both"}}],
                        "parallel_tool_calls":true,"tool_choice":"auto","stream":{stream},
                        "chat_template_kwargs":{{"enable_thinking":{thinking}}},
                        "tools":[{{"type":"function","function":{{"name":"get_alpha",
                          "parameters":{{"type":"object","properties":{{}}}}}}}},
                         {{"type":"function","function":{{"name":"get_beta",
                          "parameters":{{"type":"object","properties":{{}}}}}}}}]}}"#
                    );
                    let mut response = chat(&backend, &body);
                    let (mut content, mut names, mut null_content) =
                        (String::new(), Vec::new(), false);
                    for event in events(&mut response, stream) {
                        for choice in event.find("choices").expect("choices").items() {
                            let part = choice
                                .find(if stream { "delta" } else { "message" })
                                .expect("part");
                            if !stream {
                                null_content = part.find("content").is_some_and(Value::is_null);
                            }
                            content += &part.member_str("content", "");
                            for call in part.find("tool_calls").map_or(&[][..], Value::items) {
                                let name = call
                                    .find("function")
                                    .map(|f| f.member_str("name", ""))
                                    .unwrap_or_default();
                                if !name.is_empty() {
                                    names.push(name);
                                }
                            }
                        }
                    }
                    let case = format!(
                        "thinking {thinking} stream {stream} constrained {constrained} {separator:?}"
                    );
                    assert_eq!(names, ["get_alpha", "get_beta"], "{case}");
                    assert_eq!(content, "", "{case}");
                    assert!(stream || null_content, "{case}");
                }
            }
        }
    }
}

#[test]
fn structured_response_format() {
    let backend = FakeBackend::new();
    backend.config().reasoning_defaults.enabled = Some(true);
    let mut body = json::parse_str(
        r#"{"model":"test-model","messages":[{"role":"user","content":"Return a value"}]}"#,
    )
    .expect("json");
    for format in [
        r#"{"type":"json_object"}"#,
        r#"{"type":"json_schema","json_schema":{"name":"Reply","strict":true,"schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}}}"#,
    ] {
        body.set("response_format", json::parse_str(format).expect("json"));
        let response = chat(&backend, &body.dump().expect("dump"));
        assert!(
            response.status == 200 && backend.last_request().response_format.is_some(),
            "compiled format reaches backend"
        );
        assert_eq!(
            backend.last_request().reasoning.enabled,
            Some(true),
            "structured output retains the model reasoning default"
        );
        let mut with_tools = body.clone();
        with_tools.set(
            "tools",
            json::parse_str(
                r#"[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]"#,
            )
            .expect("json"),
        );
        let accepted = chat(&backend, &with_tools.dump().expect("dump"));
        assert!(
            accepted.status == 200 && backend.last_request().tools.len() == 1,
            "structured requests keep their active tools"
        );
    }
    body.member_mut("response_format")
        .member_mut("json_schema")
        .set("description", "Use the requested labels.");
    assert!(
        chat(&backend, &body.dump().expect("dump")).status == 200
            && backend.last_request().response_format_description == "Use the requested labels.",
        "format description reaches the model request"
    );
    let literal = r#"{"text":"<think>literal</think> <tool_call>literal</tool_call> ┌"}"#;
    let bytes = literal.as_bytes();
    backend.config().pieces = vec![
        bytes[..bytes.len() - 4].to_vec(),
        bytes[bytes.len() - 4..bytes.len() - 3].to_vec(),
        bytes[bytes.len() - 3..].to_vec(),
    ];
    body.set(
        "response_format",
        json::parse_str(r#"{"type":"json_object"}"#).expect("json"),
    );
    body.set("reasoning_effort", "none");
    for stream in [false, true] {
        body.set("stream", stream);
        let mut response = chat(&backend, &body.dump().expect("dump"));
        let (answer, reasoning) = answer_and_reasoning(&mut response, stream);
        assert!(
            answer == literal && reasoning.is_empty(),
            "JSON strings do not activate output protocol markers (stream {stream})"
        );
    }
    for stream in [false, true] {
        body.set("stream", stream);
        body.set("reasoning_effort", "high");
        backend.set_pieces(&["Consider <tool_call> as text.", "</thi", "nk>", literal]);
        let mut reply = chat(&backend, &body.dump().expect("dump"));
        assert_eq!(reply.status, 200, "thinking and structured output coexist");
        let (answer, reasoning) = answer_and_reasoning(&mut reply, stream);
        assert!(
            answer == literal && reasoning == "Consider <tool_call> as text.",
            "only the reasoning delimiter changes structured-output phase (stream {stream})"
        );
    }
    body.set("stop", "END");
    assert_eq!(
        chat(&backend, &body.dump().expect("dump")).status,
        200,
        "structured requests accept explicit stops"
    );
    let mut body =
        json::parse_str(r#"{"model":"test-model","messages":[{"role":"user","content":"value"}]}"#)
            .expect("json");
    for invalid in [
        r#"{"type":"unknown"}"#,
        r#"{"type":"json_schema","json_schema":{"name":"bad name","schema":{}}}"#,
        r#"{"type":"json_schema","json_schema":{"name":"Reply","strict":1,"schema":{}}}"#,
        r#"{"type":"json_schema","json_schema":{"name":"Reply","schema":{"type":"object","properties":{},"additionalProperties":false,"not":{}}}}"#,
    ] {
        body.set("response_format", json::parse_str(invalid).expect("json"));
        body.set("stream", true);
        let calls = backend.chat_calls.load(Ordering::SeqCst);
        let rejected = chat(&backend, &body.dump().expect("dump"));
        assert!(
            rejected.status == 400
                && rejected.streaming_body.is_none()
                && backend.chat_calls.load(Ordering::SeqCst) == calls
                && body_text(&rejected).contains("invalid_response_format"),
            "invalid schema fails before stream headers or admission: {invalid}"
        );
    }
}

fn grammar_accepts(grammar: &JsonConstraint, text: &str) -> bool {
    let mut state = grammar.start().expect("start");
    for &byte in text.as_bytes() {
        state = grammar.advance(&state, byte).expect("advance");
        if state.is_empty() {
            return false;
        }
    }
    grammar.complete(&state)
}

/// As in llama.cpp, response schemas that omit additionalProperties are
/// admitted and keep declared objects closed; additionalProperties true or a
/// schema admits extra keys. Strict tool parameters keep requiring
/// explicitly closed objects.
#[test]
fn response_schemas_without_additional_properties() {
    let backend = FakeBackend::new();
    let mut open = json::parse_str(
        r#"{"model":"test-model","messages":[{"role":"user","content":"Who?"}],
            "response_format":{"type":"json_schema","json_schema":{
              "name":"person","strict":true,"schema":{"type":"object",
              "properties":{"name":{"type":"string"},
                "born":{"type":"integer","description":"birth year"}},
              "required":["name","born"]}}}}"#,
    )
    .expect("json");
    for stream in [false, true] {
        open.set("stream", stream);
        let response = chat(&backend, &open.dump().expect("dump"));
        assert_eq!(
            response.status, 200,
            "admitted without additionalProperties"
        );
        let grammar = backend.last_request().response_format.expect("format").0;
        assert!(
            grammar_accepts(&grammar, r#"{"name":"Nostradamus","born":1503}"#)
                && !grammar_accepts(
                    &grammar,
                    r#"{"name":"Nostradamus","born":1503,"note":"seer"}"#
                )
                && !grammar_accepts(&grammar, r#"{"name":"Nostradamus"}"#),
            "omitted additionalProperties keeps declared objects closed"
        );
    }
    open.set("stream", false);
    open.set(
        "tools",
        json::parse_str(
            r#"[{"type":"function","function":{"name":"f","strict":true,
                "parameters":{"type":"object","properties":{"x":{"type":"string"}},
                "required":["x"]}}}]"#,
        )
        .expect("json"),
    );
    let calls = backend.chat_calls.load(Ordering::SeqCst);
    let rejected = chat(&backend, &open.dump().expect("dump"));
    assert!(
        rejected.status == 400
            && backend.chat_calls.load(Ordering::SeqCst) == calls
            && body_text(&rejected).contains("invalid_tools"),
        "strict tools still require closed parameter objects"
    );
    let format = json::parse_str(
        r#"{"type":"json_schema","name":"person","strict":true,"schema":{"type":"object",
            "properties":{"name":{"type":"string"}},"required":["name"],
            "additionalProperties":{"type":"integer"}}}"#,
    )
    .expect("json");
    let grammar = gufo_serve::response_format::parse_response_format(Some(&format), true)
        .expect("format")
        .expect("grammar");
    assert!(
        grammar_accepts(&grammar, r#"{"name":"a","b":1}"#)
            && !grammar_accepts(&grammar, r#"{"name":"a","b":"1"}"#),
        "Responses text.format admits typed additional properties"
    );
}

#[test]
fn structured_tool_truncation() {
    let call = r#"<tool_call>{"name":"f","arguments":{}}</tool_call>"#;
    for stream in [false, true] {
        for length in 0..=call.len() {
            let backend = FakeBackend::new();
            backend.set_pieces(&[&call[..length]]);
            backend.config().finish_reason = FinishReason::Length;
            let mut body = json::parse_str(
                r#"{"model":"test-model","messages":[{"role":"user","content":"call f"}],
                "response_format":{"type":"json_object"},"reasoning_effort":"none",
                "tool_choice":"required",
                "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}"#,
            )
            .expect("json");
            body.set("stream", stream);
            let mut response = chat(&backend, &body.dump().expect("dump"));
            assert_eq!(
                response.status, 200,
                "truncated constrained tool is a length result"
            );
            if stream {
                let output = drain(&mut response);
                assert!(
                    output.contains(r#""finish_reason":"length""#)
                        && !output.contains(r#""error""#)
                        && !output.contains("<tool"),
                    "partial tool delimiters are not exposed as structured content ({length})"
                );
            } else {
                let output = json::parse_str(&body_text(&response)).expect("json");
                let choice = &output.find("choices").expect("choices").items()[0];
                let message = choice.find("message").expect("message");
                assert!(
                    choice.member_str("finish_reason", "") == "length"
                        && message.member_str("content", "").is_empty()
                        && message.contains("tool_calls") == (length == call.len()),
                    "buffered truncated tools retain the length contract ({length})"
                );
            }
        }
    }
}

#[test]
fn strict_tool_schema() {
    let mut body = json::parse_str(
        r#"{"model":"test-model","messages":[{"role":"user","content":"call f"}],
        "reasoning_effort":"none","parallel_tool_calls":false,
        "tools":[{"type":"function","function":{"name":"f","strict":true,
          "parameters":{"type":"object","properties":{"text":{"type":"string"}},
            "required":["text"],"additionalProperties":false}}}]}"#,
    )
    .expect("json");
    let call = r#"<tool_call>{"name":"f","arguments":{"text":"<think>x</think></tool_call>"}}</tool_call>"#;
    let bytewise = |prefix: &str, text: &str, suffix: &str| -> Vec<Vec<u8>> {
        let mut pieces = Vec::new();
        if !prefix.is_empty() {
            pieces.push(prefix.as_bytes().to_vec());
        }
        pieces.extend(text.bytes().map(|byte| vec![byte]));
        if !suffix.is_empty() {
            pieces.push(suffix.as_bytes().to_vec());
        }
        pieces
    };
    for stream in [false, true] {
        let backend = FakeBackend::new();
        backend.config().pieces = bytewise("Calling f. ", call, " After f.");
        body.set("stream", stream);
        body.set(
            "tool_choice",
            json::parse_str(r#"{"type":"function","function":{"name":"f"}}"#).expect("json"),
        );
        let mut response = chat(&backend, &body.dump().expect("dump"));
        let output = body_text(&response) + &drain(&mut response);
        let request = backend.last_request();
        assert!(
            response.status == 200
                && request.constrained_tools
                && request.response_format.is_none()
                && !request.parallel_tool_calls
                && request.tool_choice == ToolChoice::Required,
            "strict tool schemas and forced choice reach the runner independently of response_format"
        );
        assert!(
            output.contains(r#""finish_reason":"tool_calls""#)
                && output_content(&output, stream) == "Calling f.  After f."
                && output.contains(r#"<think>x</think></tool_call>"#)
                && !output.contains("reasoning_content"),
            "strict tool payload markers remain argument data (stream {stream}):\n{output}"
        );
    }
    body.set("tool_choice", "auto");
    body.set("parallel_tool_calls", true);
    for stream in [false, true] {
        let backend = FakeBackend::new();
        let raw = format!(" Before {call} between {call} after <");
        backend.config().pieces = bytewise("", &raw, "");
        body.set("stream", stream);
        let mut response = chat(&backend, &body.dump().expect("dump"));
        let (content, _) = answer_and_reasoning(&mut response, stream);
        assert_eq!(
            content, " Before  between  after <",
            "parallel strict calls preserve surrounding prose and whitespace (stream {stream})"
        );
    }
    body.set("parallel_tool_calls", false);
    for invalid in [
        r#"{"tool_choice":{"type":"function","function":{"name":"absent"}}}"#,
        r#"{"parallel_tool_calls":"false"}"#,
        r#"{"tools":[{"type":"function","function":{"name":"bad<name","parameters":{}}}]}"#,
        r#"{"tools":[{"type":"function","function":{"name":"f"}},{"type":"function","function":{"name":"f"}}]}"#,
        r#"{"tools":[{"type":"function","function":{"name":"bad","strict":true,"parameters":{"type":"object"}}}]}"#,
    ] {
        let backend = FakeBackend::new();
        let mut request = body.clone();
        for (key, value) in json::parse_str(invalid).expect("json").members() {
            request.set(key, value.clone());
        }
        assert!(
            chat(&backend, &request.dump().expect("dump")).status == 400
                && backend.chat_calls.load(Ordering::SeqCst) == 0,
            "invalid tool constraints fail before generation: {invalid}"
        );
    }
    body.set("tool_choice", "auto");
    for stream in [false, true] {
        let prose = FakeBackend::new();
        prose.set_pieces(&["A literal ", "<"]);
        body.set("stream", stream);
        let mut response = chat(&prose, &body.dump().expect("dump"));
        let output = body_text(&response) + &drain(&mut response);
        assert!(
            response.status == 200 && output_content(&output, stream) == "A literal <",
            "automatic strict tools preserve ordinary text ending with a partial marker"
        );
    }
    for length in 1..call.len() {
        let backend = FakeBackend::new();
        backend.set_pieces(&["Calling f. ", &call[..length]]);
        backend.config().finish_reason = FinishReason::Length;
        body.set("stream", false);
        let response = chat(&backend, &body.dump().expect("dump"));
        let output = body_text(&response);
        assert!(
            response.status == 200
                && output.contains(r#""finish_reason":"length""#)
                && !output.contains(r#""tool_calls""#),
            "incomplete strict calls never become API tool calls ({length})"
        );
    }
    let no_arguments = FakeBackend::new();
    no_arguments.set_pieces(&[r#"<tool_call>{"name":"f","arguments":{}}</tool_call>"#]);
    body.set(
        "tools",
        json::parse_str(r#"[{"type":"function","function":{"name":"f","strict":true}}]"#)
            .expect("json"),
    );
    assert!(
        chat(&no_arguments, &body.dump().expect("dump")).status == 200
            && no_arguments.last_request().constrained_tools,
        "omitted parameters define a strict function without arguments"
    );
    body.set(
        "tools",
        json::parse_str(
            r#"[{"type":"function","function":{"name":"f","parameters":{"type":"object"},"strict":false}}]"#,
        )
        .expect("json"),
    );
    body.set("parallel_tool_calls", false);
    let non_strict = FakeBackend::new();
    non_strict.set_pieces(&[r#"<tool_call>{"name":"f","arguments":{}}</tool_call>"#]);
    assert!(
        chat(&non_strict, &body.dump().expect("dump")).status == 200
            && non_strict.last_request().constrained_tools,
        "parallel_tool_calls false also constrains non-strict tools"
    );
}

#[test]
fn explicit_stop_output_framing() {
    for stream in [false, true] {
        for thinking in [false, true] {
            let backend = FakeBackend::new();
            // The scheduler already removed the stop marker: protocol framing
            // alone, including a required tool interrupted before its call.
            backend.set_pieces(&["safe"]);
            {
                let mut config = backend.config();
                config.finish_reason = FinishReason::StopSequence;
                config.stop_sequence = "END".into();
            }
            let body = hello_with(&[
                (
                    "tools",
                    r#"[{"type":"function","function":{"name":"f","parameters":{}}}]"#,
                ),
                ("tool_choice", "\"required\""),
                ("stop", "\"END\""),
                ("stream", if stream { "true" } else { "false" }),
                (
                    "chat_template_kwargs",
                    if thinking {
                        r#"{"enable_thinking":true}"#
                    } else {
                        r#"{"enable_thinking":false}"#
                    },
                ),
            ]);
            let mut response = chat(&backend, &body);
            assert_eq!(response.status, 200, "explicit stop completes successfully");
            let output = body_text(&response) + &drain(&mut response);
            assert!(
                output.contains("\"finish_reason\":\"stop\"")
                    && !output.contains("tool_choice_unsatisfied")
                    && !output.contains("END"),
                "explicit stop wins over required tools and hides its marker: {output}"
            );
            assert!(
                output.contains(if thinking {
                    "\"reasoning_content\":\"safe\""
                } else {
                    "\"content\":\"safe\""
                }),
                "stopped text retains reasoning/content framing: {output}"
            );
        }
    }
}

#[test]
fn stop_inside_tool_arguments() {
    for stream in [false, true] {
        for thinking in [false, true] {
            for partial in [
                "<tool_call><function=f><parameter=text>partial-",
                "<｜DSML｜tool_calls｜><｜DSML｜invoke name=\"f\">\
                 <｜DSML｜parameter name=\"text\" string=\"true\">partial-",
            ] {
                for earlier_call in [false, true] {
                    let backend = FakeBackend::new();
                    let raw = format!(
                        "{}{partial}",
                        if earlier_call {
                            "<tool_call><function=f></function></tool_call>"
                        } else {
                            ""
                        }
                    );
                    {
                        let mut config = backend.config();
                        config.finish_reason = FinishReason::StopSequence;
                        config.stop_sequence = "STOP".into();
                        config.pieces = vec![b"safe".to_vec()];
                        // Constrained reasoning closes before a tool starts.
                        if thinking {
                            config.pieces.push(b"</think>".to_vec());
                        }
                        // Splits inside XML delimiters and multibyte DSML tags.
                        config.pieces.extend(raw.bytes().map(|byte| vec![byte]));
                    }
                    let body = hello_with(&[
                        ("messages", r#"[{"role":"user","content":"call f"}]"#),
                        (
                            "tools",
                            r#"[{"type":"function","function":{"name":"f","parameters":{}}}]"#,
                        ),
                        ("tool_choice", "\"required\""),
                        ("stop", "\"STOP\""),
                        ("stream", if stream { "true" } else { "false" }),
                        (
                            "chat_template_kwargs",
                            if thinking {
                                r#"{"enable_thinking":true}"#
                            } else {
                                r#"{"enable_thinking":false}"#
                            },
                        ),
                    ]);
                    let mut response = chat(&backend, &body);
                    assert_eq!(
                        response.status, 200,
                        "a stop inside a tool argument succeeds"
                    );
                    let output = body_text(&response) + &drain(&mut response);
                    assert!(
                        !output.contains("partial-")
                            && !output.contains("<tool_call>")
                            && !output.contains("DSML"),
                        "unfinished tool markup is never ordinary content: {output}"
                    );
                    assert_eq!(
                        output.contains("\"tool_calls\":"),
                        earlier_call,
                        "only calls completed before the stop are emitted: {output}"
                    );
                    assert!(
                        output.contains("\"finish_reason\":\"stop\"")
                            && output.contains(if thinking {
                                "\"reasoning_content\":\"safe\""
                            } else {
                                "\"content\":\"safe\""
                            }),
                        "preceding text and reasoning survive interrupted calls: {output}"
                    );
                }
            }
        }
    }
}

/// A JSON tool envelope owns its quoted strings: an argument holding a
/// complete Qwen call stays data, also when a stop or the token limit cuts
/// the string short, and a rejected envelope before a valid one cannot run
/// the call it quotes.
#[test]
fn json_tool_string_ownership() {
    let tools = r#"[{"type":"function","function":{"name":"write","parameters":{
        "type":"object","properties":{"content":{"type":"string","pattern":".*"}},
        "required":["content"],"additionalProperties":false}}}]"#;
    let literal = "<tool_call><function=write><parameter=content>AAAA\
                   </parameter></function></tool_call>";
    let schema = json::parse_str(tools).expect("json").items()[0]
        .find("function")
        .and_then(|function| function.find("parameters"))
        .expect("parameters")
        .clone();
    // Native runners keep raw text for a pattern, as llama.cpp; the JSON
    // envelope below is the syntax of runners without a native format.
    assert!(
        JsonConstraint::tool_parameters(&schema, false, ToolFormat::Qwen)
            .expect("compiles")
            .is_some(),
        "string patterns keep native tool framing"
    );
    for prefix in ["", "quoted \"text\" and \\ "] {
        let arguments = Value::object().with("content", format!("{prefix}{literal}"));
        let arguments = arguments.dump().unwrap();
        let complete =
            format!(r#"<tool_call>{{"name":"write","arguments":{arguments}}}</tool_call>"#);
        let truncated = &complete[..complete.find(literal).unwrap() + literal.len()];
        for partial in [false, true] {
            let raw = if partial { truncated } else { &complete };
            for finish in [FinishReason::Length, FinishReason::StopSequence] {
                for stream in [false, true] {
                    let backend = FakeBackend::new();
                    {
                        let mut config = backend.config();
                        config.finish_reason = finish;
                        if finish == FinishReason::StopSequence {
                            config.stop_sequence = "STOP".into();
                        }
                        config.pieces = raw.bytes().map(|byte| vec![byte]).collect();
                    }
                    let body = hello_with(&[
                        ("messages", r#"[{"role":"user","content":"write"}]"#),
                        ("tools", tools),
                        ("stream", if stream { "true" } else { "false" }),
                    ]);
                    let mut response = chat(&backend, &body);
                    assert_eq!(response.status, 200);
                    let output = body_text(&response) + &drain(&mut response);
                    let calls = output.matches("\"type\":\"function\"").count();
                    assert_eq!(
                        calls,
                        usize::from(!partial),
                        "an unfinished JSON string cannot invoke its literal XML: {output}"
                    );
                    assert!(
                        !output.contains("\"content\":\"<") && !output.contains("AAAA\"}"),
                        "JSON tool framing stays out of content: {output}"
                    );
                    if !partial {
                        let expected = Value::from(arguments.clone()).dump().unwrap();
                        assert!(
                            output.contains(&format!("\"arguments\":{expected}")),
                            "a complete JSON call keeps its literal XML argument: {output}"
                        );
                    }
                }
            }
        }
        for rejected in [
            format!(r#"{{"name":"undeclared","arguments":{arguments}}}"#),
            format!(r#"{{"name":"","arguments":{arguments}}}"#),
            format!(
                r#"{{"name":"write","arguments":[{}]}}"#,
                Value::from(format!("{prefix}{literal}")).dump().unwrap()
            ),
        ] {
            let backend = FakeBackend::new();
            backend.set_pieces(&[&format!("<tool_call>{rejected}</tool_call>{complete}")]);
            let body = hello_with(&[
                ("messages", r#"[{"role":"user","content":"write"}]"#),
                ("tools", tools),
            ]);
            let response = chat(&backend, &body);
            assert_eq!(response.status, 200);
            let output = json::parse(&response.body).unwrap();
            let message = output.find("choices").unwrap().items()[0]
                .find("message")
                .unwrap()
                .clone();
            let calls = message.find("tool_calls").expect("tool_calls");
            assert_eq!(calls.items().len(), 1, "{}", body_text(&response));
            assert_eq!(
                calls.items()[0]
                    .find("function")
                    .unwrap()
                    .member_str("arguments", ""),
                arguments,
                "rejected JSON envelopes cannot invoke quoted XML during recovery"
            );
        }
    }
}

#[test]
fn streaming_is_live() {
    let backend = FakeBackend::new();
    // Without tools a literal marker is ordinary content.
    backend.set_pieces(&["Hel<tool_call>", "lo"]);
    backend.config().finish_reason = FinishReason::Length;
    backend.config().block_after_first_piece = true;
    let mut response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"hello"}],
            "max_tokens":2,"stream":true,"stream_options":{"include_usage":true}}"#,
    );
    assert_eq!(response.status, 200, "Streaming request is accepted");
    let body = response
        .streaming_body
        .take()
        .expect("Streaming request returns a streaming body");
    let output = Arc::new((Mutex::new(String::new()), Condvar::new()));
    let writer = {
        let output = output.clone();
        std::thread::spawn(move || {
            body(&mut |chunk: &[u8]| {
                output
                    .0
                    .lock()
                    .expect("output")
                    .push_str(&String::from_utf8_lossy(chunk));
                output.1.notify_all();
                true
            })
            .expect("stream");
        })
    };
    assert!(
        backend.wait_for_first_piece(),
        "Backend emits the first token"
    );
    {
        let text = output.0.lock().expect("output");
        let (_text, timeout) = output
            .1
            .wait_timeout_while(text, Duration::from_secs(2), |t| {
                !t.contains(r#""content":"Hel<tool_call>""#)
            })
            .expect("output");
        assert!(
            !timeout.timed_out(),
            "First content delta is written promptly"
        );
        assert!(
            !backend.completed.load(Ordering::SeqCst),
            "First content delta arrives before generation completes"
        );
    }
    backend.release();
    writer.join().expect("writer");
    let output = output.0.lock().expect("output").clone();
    for (needle, message) in [
        (
            r#""finish_reason":"length""#,
            "Token limit is reported as finish_reason length",
        ),
        (r#""prompt_tokens":7"#, "Usage is emitted when requested"),
        (
            r#""cached_tokens":5"#,
            "Usage reports transparently reused prompt tokens",
        ),
        (r#""prefill_tokens":2"#, "Usage reports actual prefill work"),
        (r#""prompt_n":2"#, "Timings exclude cached tokens"),
        (
            r#""prompt_tokens_per_second":800"#,
            "Usage throughput counts only tokens actually prefilled",
        ),
        (r#""prefill_ms":2.5"#, "Usage reports server prefill time"),
        (r#""decode_ms":4"#, "Usage reports server decode time"),
        (
            r#""requested_logical_concurrency":4"#,
            "Usage reports configured logical concurrency",
        ),
        (
            r#""execution_plan":"serial-fallback""#,
            "Usage reports the executed serving plan",
        ),
    ] {
        assert!(output.contains(needle), "{message}: {output}");
    }
    assert!(
        output.ends_with("data: [DONE]\n\n"),
        "Stream terminates with the OpenAI DONE sentinel"
    );
    let details = stream_details(&response);
    assert!(
        details.contains("cached_tokens=5") && details.contains("finish=length"),
        "Streaming completion retains request diagnostics"
    );
}

#[test]
fn cache_prompt_option() {
    for stream in [false, true] {
        for value in ["true", "false", "null", "0", "\"false\""] {
            let backend = FakeBackend::new();
            let mut response = chat(
                &backend,
                &format!(
                    r#"{{"model":"test-model","messages":[{{"role":"user","content":"hello"}}],"stream":{stream},"cache_prompt":{value}}}"#
                ),
            );
            // Null keeps the default, as in llama-server.
            let valid = value == "true" || value == "false" || value == "null";
            assert_eq!(
                response.status,
                if valid { 200 } else { 400 },
                "cache_prompt accepts only JSON booleans and null"
            );
            if valid {
                drain(&mut response);
                assert_eq!(
                    backend.last_request().cache_prompt,
                    value != "false",
                    "cache_prompt reaches both buffered and streaming backends"
                );
            }
        }
    }
}

#[test]
fn streaming_without_usage() {
    for options in ["", r#","stream_options":{"include_usage":false}"#] {
        let backend = FakeBackend::new();
        backend.set_pieces(&["ok"]);
        let mut response = chat(
            &backend,
            &format!(
                r#"{{"model":"test-model","messages":[{{"role":"user","content":"hello"}}],"stream":true{options}}}"#
            ),
        );
        assert!(
            response.status == 200 && response.streaming_body.is_some(),
            "Stream without usage is accepted"
        );
        let output = drain(&mut response);
        assert!(
            !output.contains(r#""usage":"#) && output.ends_with("data: [DONE]\n\n"),
            "Usage chunk is opt-in"
        );
        assert!(
            output.contains(r#""prompt_per_second":800"#) && output.contains(r#""cache_n":5"#),
            "Terminal timings survive omitted or disabled usage on cached turns"
        );
        assert!(
            stream_details(&response).contains("generated_tokens=1"),
            "Request diagnostics do not depend on client usage preference"
        );
    }
}

fn check_utf8(pieces: Vec<Vec<u8>>, expected: &str, reasoning: bool) {
    let backend = FakeBackend::new();
    backend.config().pieces = pieces;
    let mut body = json::parse_str(HELLO).expect("json");
    body.member_mut("chat_template_kwargs")
        .set("enable_thinking", reasoning);
    let field = if reasoning {
        "reasoning_content"
    } else {
        "content"
    };
    let complete = chat(&backend, &body.dump().expect("dump"));
    assert_eq!(complete.status, 200, "Unicode completion succeeds");
    let parsed = json::parse(&complete.body).expect("json");
    assert_eq!(
        parsed.find("choices").expect("choices").items()[0]
            .find("message")
            .expect("message")
            .member_str(field, ""),
        expected,
        "Non-streaming UTF-8 preserves scalars and replaces malformed bytes"
    );
    body.set("stream", true);
    let mut stream = chat(&backend, &body.dump().expect("dump"));
    assert!(
        stream.status == 200 && stream.streaming_body.is_some(),
        "Unicode stream succeeds"
    );
    let mut text = String::new();
    if let Some(write) = stream.streaming_body.take() {
        write(&mut |chunk: &[u8]| {
            if chunk == b"data: [DONE]\n\n" {
                return true;
            }
            assert!(chunk.starts_with(b"data: "), "SSE has a data prefix");
            let event = json::parse(&chunk[6..]).expect("event");
            for choice in event.find("choices").expect("choices").items() {
                if let Some(delta) = choice.find("delta") {
                    text.push_str(&delta.member_str(field, ""));
                }
            }
            assert!(
                expected.starts_with(&text),
                "Every SSE event ends at a complete Unicode scalar"
            );
            true
        })
        .expect("stream");
    }
    assert_eq!(text, expected, "Streaming and non-streaming UTF-8 agree");
}

#[test]
fn utf8_output() {
    let valid = "Aé中┌😀Z".as_bytes();
    let replacement = "\u{FFFD}";
    let valid_text = "Aé中┌😀Z";
    for reasoning in [false, true] {
        for split in 1..valid.len() {
            check_utf8(
                vec![valid[..split].to_vec(), valid[split..].to_vec()],
                valid_text,
                reasoning,
            );
        }
        check_utf8(
            valid.iter().map(|b| vec![*b]).collect(),
            valid_text,
            reasoning,
        );
        check_utf8(
            vec![b"\xE2\x82".to_vec(), b"X".to_vec()],
            &format!("{replacement}X"),
            reasoning,
        );
        check_utf8(
            vec![b"\xF0".to_vec(), b"\x9F".to_vec()],
            replacement,
            reasoning,
        );
        check_utf8(
            vec![b"\x80".to_vec(), b"ok".to_vec()],
            &format!("{replacement}ok"),
            reasoning,
        );
        check_utf8(
            vec![b"\xC0\xAF".to_vec()],
            &replacement.repeat(2),
            reasoning,
        );
        check_utf8(
            vec![b"\xED".to_vec(), b"\xA0\x80".to_vec()],
            &replacement.repeat(3),
            reasoning,
        );
        check_utf8(
            vec![b"\xF4\x90\x80\x80".to_vec()],
            &replacement.repeat(4),
            reasoning,
        );
    }
}

#[test]
fn cached_prefill_metrics() {
    let backend = FakeBackend::new();
    backend.set_pieces(&["ok"]);
    let response = chat(&backend, HELLO);
    assert_eq!(
        response.status, 200,
        "Cached non-streaming response succeeds"
    );
    let body = json::parse(&response.body).expect("json");
    let usage = body.find("usage").expect("usage");
    let timings = body.find("timings").expect("timings");
    assert!(
        body.find("metrics").is_none(),
        "One timing schema prevents proxies from subtracting cached tokens twice"
    );
    assert_eq!(
        usage.member_size("prompt_tokens", 0),
        7,
        "Token usage includes cached tokens"
    );
    assert!(
        timings.member_size("prompt_n", 0) == 2
            && timings.member_double("prompt_per_second", 0.0) == 800.0
            && timings.member_double("prompt_per_token_ms", 0.0) == 1.25,
        "Non-streaming timings report executed prefill work"
    );
    let mut cached = GenerationResult {
        prompt_tokens: 1024,
        cached_prompt_tokens: 1024,
        prefill_ms: 0.01,
        ..GenerationResult::default()
    };
    assert_eq!(
        prefill_tokens_per_second(&cached),
        0.0,
        "Full cache hits cannot report artificial prefill throughput"
    );
    cached.prefill_tokens = 10;
    cached.prefill_ms = 0.0;
    assert_eq!(
        prefill_tokens_per_second(&cached),
        0.0,
        "Untimed work does not divide by zero"
    );

    backend.config().cache_miss_reason = "prefix_changed".into();
    let miss = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"changed"}]}"#,
    );
    let miss_body = json::parse(&miss.body).expect("json");
    let metrics = miss_body
        .find("usage")
        .and_then(|u| u.find("gufo"))
        .expect("metrics");
    assert!(
        metrics.member_str("cache_miss_reason", "") == "prefix_changed"
            && metrics.member_size("cache_common_prefix_tokens", 0) == 2
            && metrics.member_size("cache_checkpoint_tokens", 0) == 5,
        "Usage explains cache misses without exposing prompt text"
    );
}

#[test]
fn tool_calls_are_structured() {
    let backend = FakeBackend::new();
    backend.set_pieces(&[
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>",
    ]);
    let response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"weather in Rome"}],
            "tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather",
              "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
            "tool_choice":"required","stream":false}"#,
    );
    let text = body_text(&response);
    assert_eq!(response.status, 200, "Tool request is accepted");
    assert!(
        text.contains(r#""finish_reason":"tool_calls""#),
        "Tool generation reports tool_calls finish reason"
    );
    assert!(
        text.contains(r#""name":"get_weather""#),
        "Tool name is translated to OpenAI format"
    );
    assert!(
        text.contains(r#"\"city\":\"Rome\""#),
        "Tool arguments are translated to JSON"
    );
    let last = backend.last_request();
    assert_eq!(last.tools.len(), 1, "Tool schema reaches the model backend");
    assert_eq!(
        last.tool_choice,
        ToolChoice::Required,
        "Required tool choice reaches the model backend"
    );
}

#[test]
fn tool_parameter_compatibility() {
    let cases = [
        (r#"{"name":"f"}"#, "{}"),
        (r#"{"name":"f","parameters":null}"#, "{}"),
        (r#"{"name":"f","parameters":{}}"#, "{}"),
        (r#"{"name":"f","parametersJsonSchema":null}"#, "{}"),
        (
            r#"{"name":"f","parametersJsonSchema":{"type":"object"}}"#,
            r#"{"type":"object"}"#,
        ),
        (
            r#"{"name":"f","parameters":null,"parametersJsonSchema":{"type":"object"}}"#,
            r#"{"type":"object"}"#,
        ),
        (
            r#"{"name":"f","parameters":{},"parametersJsonSchema":"ignored"}"#,
            "{}",
        ),
    ];
    for flat in [false, true] {
        for stream in [false, true] {
            for (function_json, expected_parameters) in cases {
                let backend = FakeBackend::new();
                backend.set_pieces(&["<tool_call>\n<function=f>\n</function>\n</tool_call>"]);
                let mut body = json::parse_str(
                    r#"{"model":"test-model","messages":[{"role":"user","content":"call f"}],
                        "tool_choice":"required","tools":[]}"#,
                )
                .expect("json");
                let mut definition = Value::object().with("type", "function");
                let function = json::parse_str(function_json).expect("json");
                if flat {
                    for (key, value) in function.members() {
                        definition.append_member(key.clone(), value.clone());
                    }
                } else {
                    definition.set("function", function);
                }
                body.member_mut("tools").push(definition);
                body.set("stream", stream);
                let mut response = chat(&backend, &body.dump().expect("dump"));
                assert_eq!(response.status, 200, "Compatible tool schema is accepted");
                let mut output = body_text(&response);
                if stream {
                    assert!(
                        response.streaming_body.is_some(),
                        "Compatible tool request supports streaming"
                    );
                    output += &drain(&mut response);
                }
                assert!(
                    output.contains(r#""name":"f""#)
                        && output.contains(r#""finish_reason":"tool_calls""#),
                    "No-argument function returns a structured tool call"
                );
                let last = backend.last_request();
                assert_eq!(last.tools.len(), 1, "Normalized tool reaches the backend");
                let tool = &last.tools[0];
                assert_eq!(
                    tool.parameters_json, expected_parameters,
                    "Missing/null schemas normalize and parameters take precedence"
                );
                let expected_definition = Value::object().with("type", "function").with(
                    "function",
                    Value::object().with("name", "f").with(
                        "parameters",
                        json::parse_str(expected_parameters).expect("json"),
                    ),
                );
                assert_eq!(
                    tool.definition_json,
                    expected_definition.dump().expect("dump"),
                    "Templates receive the normalized nested definition"
                );
            }
        }
    }
}

#[test]
fn invalid_tools_fail_before_generation() {
    let invalid = [
        "null",
        "42",
        r#"{"function":{"name":"f"}}"#,
        r#"{"type":42,"function":{"name":"f"}}"#,
        r#"{"type":"custom","custom":{"name":"shell"}}"#,
        r#"{"type":"function","function":null,"name":"f"}"#,
        r#"{"type":"function","function":[],"name":"f"}"#,
        r#"{"type":"function","function":{}}"#,
        r#"{"type":"function","name":""}"#,
        r#"{"type":"function","name":42}"#,
        r#"{"type":"function","name":"f","parameters":"bad"}"#,
        r#"{"type":"function","name":"f","parameters":[]}"#,
        r#"{"type":"function","name":"f","parameters":false}"#,
        r#"{"type":"function","name":"f","parametersJsonSchema":[]}"#,
    ];
    for stream in [false, true] {
        for valid_first in [false, true] {
            for entry in invalid {
                let backend = FakeBackend::new();
                let mut body = json::parse_str(
                    r#"{"model":"test-model","messages":[{"role":"user","content":"use tools"}],"tools":[]}"#,
                )
                .expect("json");
                if valid_first {
                    body.member_mut("tools").push(
                        json::parse_str(
                            r#"{"type":"function","function":{"name":"valid","parameters":{}}}"#,
                        )
                        .expect("json"),
                    );
                }
                body.member_mut("tools")
                    .push(json::parse_str(entry).expect("json"));
                body.set("stream", stream);
                let response = chat(&backend, &body.dump().expect("dump"));
                assert!(
                    response.status == 400
                        && body_text(&response).contains("invalid_tools")
                        && response.streaming_body.is_none()
                        && backend.chat_calls.load(Ordering::SeqCst) == 0,
                    "Invalid tools fail before generation, including mixed lists: {entry}"
                );
            }
        }
    }
}

/// Adds the tool calls of one message (`stream`: one chunk's deltas, merged
/// by index) to `calls` (`AppendToolDeltas`).
fn append_tool_deltas(calls: &mut Vec<Value>, deltas: &Value, stream: bool) {
    for delta in deltas.items() {
        if !stream {
            calls.push(delta.clone());
            continue;
        }
        let index = delta.member_size("index", 0);
        assert!(index <= calls.len(), "tool indexes have no gaps");
        if index == calls.len() {
            calls.push(Value::object().with("function", Value::object()));
        }
        let call = &mut calls[index];
        let id = delta.member_str("id", "");
        if !id.is_empty() {
            assert!(call.member_str("id", "").is_empty(), "tool ID is sent once");
            call.set("id", id);
            call.set("type", delta.member_str("type", ""));
        }
        if let Some(function) = delta.find("function") {
            for key in ["name", "arguments"] {
                let joined =
                    call.member_mut("function").member_str(key, "") + &function.member_str(key, "");
                call.member_mut("function").set(key, joined);
            }
        }
    }
}

/// Tool calls from a response: the message's list, or every streamed delta.
fn collect_calls(response: &mut HttpResponse, stream: bool, expect_calls: bool) -> Vec<Value> {
    let mut calls = Vec::new();
    if !stream {
        let output = json::parse(&response.body).expect("json");
        let message = output.find("choices").expect("choices").items()[0]
            .find("message")
            .expect("message")
            .clone();
        if let Some(found) = message.find("tool_calls") {
            calls.extend(found.items().iter().cloned());
        }
        return calls;
    }
    let output = drain(response);
    let mut cursor = 0;
    while let Some(found) = output[cursor..].find("data: ") {
        let begin = cursor + found + 6;
        let end = output[begin..]
            .find('\n')
            .map_or(output.len(), |e| begin + e);
        cursor = end;
        let payload = &output[begin..end];
        if payload == "[DONE]" {
            break;
        }
        let event = json::parse_str(payload).expect("event");
        let Some(choices) = event.find("choices").filter(|c| !c.is_empty()) else {
            continue;
        };
        let delta = choices.items()[0].find("delta");
        if let Some(found) = delta.and_then(|d| d.find("tool_calls")) {
            append_tool_deltas(&mut calls, found, true);
        }
        if expect_calls && let Some(delta) = delta {
            assert!(
                !delta.member_str("reasoning_content", "").contains('<'),
                "tool markers do not leak into reasoning deltas"
            );
        }
    }
    calls
}

#[test]
fn qwen_tool_boundaries_and_schema() {
    let schema = json::parse_str(
        r#"{"model":"test-model","messages":[{"role":"user","content":"use f"}],
            "tools":[{"type":"function","function":{"name":"f","parameters":{
              "type":"object","properties":{"text":{"type":"string"},"count":{"type":"integer"},
              "flag":{"type":"boolean"},"limit":{"type":["integer","null"]},
              "tags":{"type":"array"}}}}}]}"#,
    )
    .expect("json");
    let good = "<tool_call><function=f><parameter=text>42</parameter>\
                <parameter=count>42</parameter></function></tool_call>";
    let cases: Vec<(String, usize, &str)> = vec![
        (good.into(), 1, r#"{"text":"42","count":42}"#),
        (
            "<tool_call><function=f><parameter=text>literal </tool_call> and </function></parameter></function></tool_call>".into(),
            1,
            r#"{"text":"literal </tool_call> and </function>"}"#,
        ),
        (
            "<tool_call><function=f><parameter=text>ok</parameter><parameter=count>oops</parameter></function></tool_call>".into(),
            0,
            "",
        ),
        (
            "<tool_call><function=f><parameter=text>ok</parameter><parameter=count>42</function></tool_call>".into(),
            0,
            "",
        ),
        (
            format!("<tool_call><function=f><parameter=text>ok</parameter><parameter=count>42</function></tool_call>{good}"),
            1,
            r#"{"text":"42","count":42}"#,
        ),
        (
            format!("<tool_call><function=f><parameter=text>unclosed{good}"),
            1,
            r#"{"text":"42","count":42}"#,
        ),
        (
            "<tool_call><function=f><parameter=text>literal </think></parameter></function></tool_call>".into(),
            1,
            r#"{"text":"literal </think>"}"#,
        ),
        // One newline on each side of a value is template framing; a file's
        // final newline, indentation and an empty value survive.
        (
            "<tool_call>\n<function=f>\n<parameter=text>\n  line 1\nline 2\n\n</parameter>\n\
             <parameter=count>\n42\n</parameter>\n</function>\n</tool_call>"
                .into(),
            1,
            r#"{"text":"  line 1\nline 2\n","count":42}"#,
        ),
        (
            "<tool_call>\n<function=f>\n<parameter=text>\n\n</parameter>\n</function>\n</tool_call>"
                .into(),
            1,
            r#"{"text":""}"#,
        ),
        (
            "<tool_call>\r\n<function=f>\r\n<parameter=text>\r\n\r\nx\r\n</parameter>\r\n\
             </function>\r\n</tool_call>"
                .into(),
            1,
            r#"{"text":"\r\nx"}"#,
        ),
        // Python literals where the schema wants JSON; a string keeps them.
        (
            "<tool_call><function=f><parameter=flag>\nTrue\n</parameter>\
             <parameter=limit>None</parameter><parameter=tags>\
             [False, \"None\", \"a \\\" True\"]</parameter>\
             <parameter=text>True</parameter></function></tool_call>"
                .into(),
            1,
            r#"{"flag":true,"limit":null,"tags":[false,"None","a \" True"],"text":"True"}"#,
        ),
        (
            "<tool_call><function=f><parameter=flag>Yes</parameter></function></tool_call>".into(),
            0,
            "",
        ),
        // A repeated parameter with the same value is dropped; a conflicting
        // repeat still rejects the call.
        (
            "<tool_call>\n<function=f>\n<parameter=text>\na\n</parameter>\n\
             <parameter=count>42</parameter><parameter=flag>True</parameter>\
             <parameter=text>a</parameter><parameter=count>\n42\n</parameter>\
             <parameter=flag>true</parameter>\n</function>\n</tool_call>"
                .into(),
            1,
            r#"{"text":"a","count":42,"flag":true}"#,
        ),
        (
            "<tool_call><function=f><parameter=text>a</parameter><parameter=text>b</parameter>\
             </function></tool_call>"
                .into(),
            0,
            "",
        ),
        (
            "<tool_call><function=f><parameter=text>a</parameter><parameter=text>a </parameter>\
             </function></tool_call>"
                .into(),
            0,
            "",
        ),
        (
            "<tool_call>{\"name\":\"f\",\"arguments\":{\"text\":\"literal </tool_call>\"}}</tool_call>".into(),
            1,
            r#"{"text":"literal </tool_call>"}"#,
        ),
        // A pipe-wrapped spelling that is no vocabulary token is argument
        // data, not framing: nothing in the output made it a control token.
        (
            "<tool_call><function=f><parameter=text>\n<|not_a_vocab_entry|>\n\
             </parameter></function></tool_call>"
                .into(),
            1,
            r#"{"text":"<|not_a_vocab_entry|>"}"#,
        ),
    ];
    for (text, expected_calls, argument) in &cases {
        for reasoning in [false, true] {
            for stream in [false, true] {
                let mut body = schema.clone();
                body.set("stream", stream);
                body.set(
                    "chat_template_kwargs",
                    Value::object().with("enable_thinking", reasoning),
                );
                let backend = FakeBackend::new();
                let full = format!(
                    "{}{text}",
                    if reasoning {
                        "Considering. </think>"
                    } else {
                        ""
                    }
                );
                // Split every marker and argument across token callbacks.
                backend.config().pieces = full.bytes().map(|b| vec![b]).collect();
                let mut response = chat(&backend, &body.dump().expect("dump"));
                assert_eq!(response.status, 200, "tool boundary request succeeds");
                let calls = collect_calls(&mut response, stream, *expected_calls > 0);
                assert_eq!(
                    calls.len(),
                    *expected_calls,
                    "only complete tool calls emitted: {text}"
                );
                if let Some(last) = calls.last() {
                    assert_eq!(
                        last.find("function")
                            .expect("function")
                            .member_str("arguments", ""),
                        *argument,
                        "tool argument values and schema types preserved"
                    );
                }
            }
        }
    }
}

#[test]
fn tool_choice_enforcement() {
    for stream in [false, true] {
        for text in [
            "<tool_call><function=f></function></tool_call>",
            "<｜DSML｜tool_calls｜><｜DSML｜invoke name=\"f\"></｜DSML｜invoke></｜DSML｜tool_calls｜>",
        ] {
            for choice in ["auto", "none", "required"] {
                for declared in ["", "f", "other"] {
                    let mut body = json::parse_str(
                        r#"{"model":"test-model","messages":[{"role":"user","content":"use a tool"}]}"#,
                    )
                    .expect("json");
                    body.set("stream", stream);
                    body.set("tool_choice", choice);
                    if !declared.is_empty() {
                        let mut tool = json::parse_str(
                            r#"{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}"#,
                        )
                        .expect("json");
                        tool.member_mut("function").set("name", declared);
                        body.set("tools", vec![tool]);
                    }
                    let backend = FakeBackend::new();
                    backend.set_pieces(&[text]);
                    let mut response = chat(&backend, &body.dump().expect("dump"));
                    let required = choice == "required";
                    if required && declared.is_empty() {
                        assert_eq!(response.status, 400, "required needs declared tools");
                        continue;
                    }
                    let mut output = body_text(&response);
                    output += &drain(&mut response);
                    let allowed = declared == "f" && choice != "none";
                    assert_eq!(
                        output.contains("\"tool_calls\":"),
                        allowed,
                        "only declared, enabled tool calls can enter API output"
                    );
                    if required && !allowed {
                        assert!(
                            output.contains("tool_choice_unsatisfied")
                                && (stream || response.status == 502),
                            "required cannot silently return text"
                        );
                    }
                }
            }
        }
    }
    let backend = FakeBackend::new();
    backend.set_pieces(&["ordinary text"]);
    let response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"call f"}],"tool_choice":"required",
            "tools":[{"type":"function","function":{"name":"f","parameters":{}}}]}"#,
    );
    assert_eq!(
        response.status, 502,
        "ordinary text cannot fulfill required tool choice"
    );
}

#[test]
fn deepseek_tool_calls_are_structured() {
    let backend = FakeBackend::new();
    backend.set_pieces(&[
        "<｜DSML｜tool_calls｜>\n<｜DS｜invoke name=\"read\">\n<｜DS｜parameter name=\"path\" string=\"true\">/etc/hostname</｜DS｜parameter>\n</｜DS｜invoke>\n</｜DSML｜tool_calls｜>",
    ]);
    let response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"read the hostname"}],
            "tools":[{"type":"function","function":{"name":"read","description":"Read a file",
              "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],
            "stream":false}"#,
    );
    let text = body_text(&response);
    assert_eq!(response.status, 200, "DeepSeek tool request is accepted");
    assert!(
        text.contains(r#""finish_reason":"tool_calls""#),
        "Hybrid DeepSeek syntax reports tool_calls finish reason"
    );
    assert!(
        text.contains(r#""name":"read""#),
        "Hybrid DeepSeek tool name is translated"
    );
    assert!(
        text.contains(r#"\"path\":\"/etc/hostname\""#),
        "Hybrid DeepSeek tool arguments are translated"
    );
}

/// A call in the selected dialect is a call; another dialect's syntax under
/// a native grammar is visible content, while the JSON fallback, whose text
/// before `<tool_call>` is free, keeps recognizing native calls. Native calls
/// end the output in both formats; DeepSeek's "\n\n" before its call block
/// is framing.
#[test]
fn native_tool_dialect_selection() {
    let native_call = |format: ToolFormat, path: &str| match format {
        ToolFormat::DeepSeek => format!(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n\
             <｜DSML｜parameter name=\"path\" string=\"true\">{path}</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        ),
        _ => format!(
            "<tool_call>\n<function=read>\n<parameter=path>\n{path}\n</parameter>\n</function>\n</tool_call>"
        ),
    };
    let request_with = |schema: &Value, stream: bool| {
        let mut body = json::parse_str(
            r#"{"model":"test-model","reasoning_effort":"none",
            "messages":[{"role":"user","content":"Read fixture.xml."}],"tool_choice":"auto"}"#,
        )
        .expect("json");
        let function = Value::object()
            .with("name", "read")
            .with("strict", false)
            .with("parameters", schema.clone());
        body.set(
            "tools",
            vec![
                Value::object()
                    .with("type", "function")
                    .with("function", function),
            ],
        );
        body.set("stream", stream);
        body
    };
    let collect = |response: &mut HttpResponse, stream: bool, over_responses: bool| {
        let (mut content, mut arguments) = (String::new(), Vec::new());
        for event in events(response, stream) {
            if over_responses {
                let result = if stream {
                    event.find("response")
                } else {
                    Some(&event)
                };
                let Some(result) = result.filter(|r| r.member_str("status", "") == "completed")
                else {
                    continue;
                };
                for item in result.find("output").expect("output").items() {
                    match item.member_str("type", "").as_str() {
                        "function_call" => arguments.push(item.member_str("arguments", "")),
                        "message" => {
                            for part in item.find("content").expect("content").items() {
                                content += &part.member_str("text", "");
                            }
                        }
                        _ => {}
                    }
                }
                continue;
            }
            for choice in event.find("choices").map_or(&[][..], |c| c.items()) {
                let message = choice
                    .find(if stream { "delta" } else { "message" })
                    .expect("message");
                content += &message.member_str("content", "");
                for item in message.find("tool_calls").map_or(&[][..], |t| t.items()) {
                    // Streamed arguments arrive in pieces under a stable index.
                    let index = if stream {
                        item.member_size("index", 0)
                    } else {
                        arguments.len()
                    };
                    if index == arguments.len() {
                        arguments.push(String::new());
                    }
                    assert!(index < arguments.len(), "tool index is stable");
                    arguments[index] += &item
                        .find("function")
                        .expect("function")
                        .member_str("arguments", "");
                }
            }
        }
        (content, arguments)
    };
    // The JSON fallback leaves text before <tool_call> free: a native
    // DeepSeek call written there stays a call.
    let patterned = json::parse_str(
        r#"{"type":"object","properties":{"path":{"type":"string","pattern":"^[a-z.]+$"}},
        "required":["path"],"additionalProperties":false}"#,
    )
    .expect("json");
    // A native runner keeps DSML for this schema; the JSON-envelope runner
    // below is the remaining fallback syntax.
    assert!(
        JsonConstraint::tool_parameters(&patterned, false, ToolFormat::DeepSeek)
            .expect("compiles")
            .is_some(),
        "a patterned string keeps native tool framing"
    );
    for stream in [false, true] {
        let backend = FakeBackend::new();
        backend.config().tool_format = Some(ToolFormat::Json);
        backend.set_pieces(&[&native_call(ToolFormat::DeepSeek, "fixture.xml")]);
        let mut response = chat(
            &backend,
            &request_with(&patterned, stream).dump().expect("dump"),
        );
        assert_eq!(response.status, 200, "JSON fallback request succeeds");
        let (content, arguments) = collect(&mut response, stream, false);
        assert_eq!(
            arguments,
            [r#"{"path":"fixture.xml"}"#],
            "a native call written under the JSON fallback is returned"
        );
        assert!(
            !content.contains("DSML"),
            "the recognized native call is not content"
        );
    }
    let schema = json::parse_str(
        r#"{"type":"object","properties":{"path":{"type":"string"}},
        "required":["path"],"additionalProperties":false}"#,
    )
    .expect("json");
    for format in [ToolFormat::Qwen, ToolFormat::DeepSeek] {
        let parameters = JsonConstraint::tool_parameters(&schema, false, format)
            .expect("compiles")
            .expect("the file tool has a native wire format");
        let grammar = JsonConstraint::with_tools(
            None,
            vec![("read".into(), parameters)],
            false,
            true,
            format,
        )
        .expect("grammar")
        .expect("tools");
        let call = native_call(format, "fixture.xml");
        let foreign = if format == ToolFormat::DeepSeek {
            ToolFormat::Qwen
        } else {
            ToolFormat::DeepSeek
        };
        let prefixes = [
            "The XML root is `<tool_calls></tool_calls>`.\n".to_owned(),
            format!(
                "The foreign wrapper is `{}`.\n",
                if foreign == ToolFormat::Qwen {
                    "<tool_call>"
                } else {
                    "<｜DSML｜tool_calls>"
                }
            ),
            format!(
                "Example syntax:\n{}\nNow inspect the fixture.\n",
                native_call(foreign, "example.xml")
            ),
        ];
        // As in llama.cpp, calls end the output in both formats.
        for prefix in &prefixes {
            let raw = format!("{prefix}{call}");
            let mut state = grammar.start().expect("start");
            for byte in raw.bytes() {
                state = grammar.advance(&state, byte).expect("advance");
            }
            assert!(
                grammar.complete(&state),
                "foreign visible literals and the actual call satisfy the tool grammar: {raw:?}"
            );
            for over_responses in [false, true] {
                for stream in [false, true] {
                    let body = request_with(&schema, stream);
                    let backend = FakeBackend::new();
                    {
                        let mut config = backend.config();
                        config.tool_format = Some(format);
                        config.pieces = raw.bytes().map(|byte| vec![byte]).collect();
                    }
                    let mut response = if over_responses {
                        responses(&backend, &body, false, 256, stream)
                    } else {
                        chat(&backend, &body.dump().expect("dump"))
                    };
                    assert_eq!(response.status, 200, "native file tool request succeeds");
                    // A later backend configuration cannot change an admitted
                    // stream's format: the request holds its own.
                    backend.config().tool_format = None;
                    let (content, arguments) = collect(&mut response, stream, over_responses);
                    let label =
                        format!("{format:?} {raw:?} responses={over_responses} stream={stream}");
                    assert_eq!(
                        arguments,
                        [r#"{"path":"fixture.xml"}"#],
                        "only the selected dialect's call reaches the client: {label}"
                    );
                    assert_eq!(
                        &content, prefix,
                        "foreign tool syntax stays visible content: {label}"
                    );
                }
            }
        }
    }
    // DeepSeek's template writes "\n\n" before its call block: as in
    // llama.cpp's parser that separator is framing, and as content the
    // replayed history would carry it twice.
    let read = json::parse_str(
        r#"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}"#,
    )
    .expect("json");
    for prose in ["", "Reading the fixture."] {
        for stream in [false, true] {
            let backend = FakeBackend::new();
            let raw = format!(
                "{prose}\n\n{}",
                native_call(ToolFormat::DeepSeek, "fixture.xml")
            );
            {
                let mut config = backend.config();
                config.tool_format = Some(ToolFormat::DeepSeek);
                config.pieces = raw.bytes().map(|byte| vec![byte]).collect();
            }
            // `strict: false` is the default the upstream request leaves out.
            let body = request_with(&read, stream);
            let mut response = chat(&backend, &body.dump().expect("dump"));
            assert_eq!(response.status, 200);
            let (content, arguments) = collect(&mut response, stream, false);
            assert!(
                arguments.len() == 1 && content == prose,
                "the separator before a DeepSeek call block is not content: {content:?}"
            );
        }
    }
}

#[test]
fn deepseek_repeated_tool_parameters() {
    // String values keep their whitespace, so a framed copy conflicts.
    for (repeat, accepted) in [("/a", true), ("\n/a\n", false), ("/b", false)] {
        let backend = FakeBackend::new();
        let call = format!(
            "<｜DSML｜tool_calls｜><｜DSML｜invoke name=\"read\"><｜DSML｜parameter name=\"path\" \
             string=\"true\">/a</｜DSML｜parameter><｜DSML｜parameter name=\"path\" \
             string=\"true\">{repeat}</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls｜>"
        );
        backend.set_pieces(&[call.as_str()]);
        let response = chat(
            &backend,
            r#"{"model":"test-model","messages":[{"role":"user","content":"read"}],
                "tools":[{"type":"function","function":{"name":"read",
                  "parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]}"#,
        );
        let text = body_text(&response);
        assert_eq!(response.status, 200, "DeepSeek repeated parameter request");
        assert_eq!(
            text.contains(r#""arguments":"{\"path\":\"/a\"}""#),
            accepted,
            "identical DeepSeek repeats are dropped, conflicts rejected: {text}"
        );
        assert_eq!(
            text.contains(r#""finish_reason":"tool_calls""#),
            accepted,
            "a conflicting DeepSeek repeat is not a tool call"
        );
    }
}

#[test]
fn tool_name_characters() {
    let declare = |name: &str| {
        let mut body = json::parse_str(
            r#"{"model":"test-model","messages":[{"role":"user","content":"use it"}],"tools":[]}"#,
        )
        .expect("json");
        let mut tool = json::parse_str(
            r#"{"type":"function","function":{"name":"x","parameters":{"type":"object","properties":{}}}}"#,
        )
        .expect("json");
        tool.member_mut("function").set("name", name);
        body.member_mut("tools").push(tool);
        body.dump().expect("json")
    };
    let replay = |name: &str| {
        let mut call = json::parse_str(
            r#"{"id":"call_1","type":"function","function":{"name":"x","arguments":"{\"path\":\"a.txt\"}"}}"#,
        )
        .expect("json");
        call.member_mut("function").set("name", name);
        let mut assistant =
            json::parse_str(r#"{"role":"assistant","tool_calls":[]}"#).expect("json");
        assistant.member_mut("tool_calls").push(call);
        let mut body = json::parse_str(
            r#"{"model":"test-model","messages":[{"role":"user","content":"use it"}]}"#,
        )
        .expect("json");
        body.member_mut("messages").push(assistant);
        body.member_mut("messages").push(
            json::parse_str(r#"{"role":"tool","tool_call_id":"call_1","content":"done"}"#)
                .expect("json"),
        );
        body.dump().expect("json")
    };
    // Agent harnesses name bridged tools after their server; nothing between
    // the request and either renderer treats these characters as structure.
    let long = "n".repeat(64);
    for name in [
        "read_file",
        "read-file",
        "readFile",
        "server:tool",
        "fs/read",
        "github.create_issue",
        "tool@v1",
        "a.b.c~d+e",
        long.as_str(),
    ] {
        let declared = FakeBackend::new();
        let response = chat(&declared, &declare(name));
        let tools = declared.last_request().tools;
        assert!(
            response.status == 200 && tools.len() == 1 && tools[0].name == name,
            "a renderable tool name reaches the backend unchanged: {name}"
        );
        let replayed = FakeBackend::new();
        assert!(
            chat(&replayed, &replay(name)).status == 200
                && replayed.chat_calls.load(Ordering::SeqCst) == 1,
            "the same name is accepted when a message replays a call: {name}"
        );
    }
    // The renderers append a name verbatim, so a declared name that frames a
    // call is refused; so is non-ASCII. History records past output instead
    // of declaring a tool, so a replayed call keeps any non-empty name.
    let too_long = "n".repeat(65);
    for name in [
        "bad>name",
        "bad<name",
        "bad\"name",
        "bad\\name",
        "bad name",
        "bad\nname",
        "bad\x7fname",
        "outil_traçage",
        "reаd",
        "\u{200b}read",
        "",
        too_long.as_str(),
    ] {
        let declared = FakeBackend::new();
        let response = chat(&declared, &declare(name));
        assert!(
            response.status == 400
                && body_text(&response).contains("invalid_tools")
                && declared.chat_calls.load(Ordering::SeqCst) == 0,
            "an unrenderable declared name fails with invalid_tools: {name:?}"
        );
        let replayed = FakeBackend::new();
        let response = chat(&replayed, &replay(name));
        if name.is_empty() {
            assert!(
                response.status == 400
                    && body_text(&response).contains("invalid_messages")
                    && replayed.chat_calls.load(Ordering::SeqCst) == 0,
                "an empty replayed name fails with invalid_messages"
            );
        } else {
            assert!(
                response.status == 200
                    && replayed.last_request().messages[1].tool_calls[0].name == name,
                "a replayed name reaches the renderer unchanged: {name:?}"
            );
        }
    }
}

/// Malformed history fails before model work.
#[test]
fn malformed_historical_functions() {
    for source in [
        r#"{"arguments":"{}"}"#,
        r#"{"name":"","arguments":"{}"}"#,
        r#"{"name":null,"arguments":"{}"}"#,
        r#"{"name":42,"arguments":"{}"}"#,
        r#"{"name":"read"}"#,
        r#"{"name":"read","arguments":{}}"#,
        r#"{"name":"read","arguments":null}"#,
        r#"{"name":"read","arguments":"[]"}"#,
        r#"{"name":"read","arguments":"["}"#,
        r#"{"name":"read","arguments":"{\"x\":\"raw\nnewline\"}"}"#,
        r#"{"name":"read\u0000file","arguments":"{}"}"#,
    ] {
        let body = format!(
            r#"{{"model":"test-model","messages":[{{"role":"user","content":"continue"}},
                {{"role":"assistant","tool_calls":[{{"id":"call_1","type":"function","function":{source}}}]}}]}}"#
        );
        let backend = FakeBackend::new();
        let response = chat(&backend, &body);
        assert!(
            response.status == 400
                && body_text(&response).contains("invalid_messages")
                && backend.chat_calls.load(Ordering::SeqCst) == 0,
            "malformed history is rejected before model work: {source}"
        );
    }
}

/// Without tools, or with `tool_choice: none`, tool markup is ordinary text:
/// it ends no reasoning, hides no content and becomes no call.
#[test]
fn tool_markers_when_tools_are_disabled() {
    let thought = "Quoted <tool_call> and <｜DSML｜tool_calls> are reasoning data.";
    let answer = "Literal <tool_call><function=f></function></tool_call> end.";
    for stream in [false, true] {
        for declared in [false, true] {
            for phase in 0..3 {
                let incomplete = phase == 2;
                let expected_thought = if phase > 0 { thought } else { "" };
                let expected_answer = if incomplete { "" } else { answer };
                let raw = format!(
                    "{expected_thought}{}{expected_answer}",
                    if phase == 1 { "</think>" } else { "" }
                );
                let backend = FakeBackend::new();
                {
                    let mut config = backend.config();
                    config.finish_reason = if incomplete {
                        FinishReason::Length
                    } else {
                        FinishReason::Stop
                    };
                    config.pieces = raw.bytes().map(|byte| vec![byte]).collect();
                }
                let mut fields = vec![
                    (
                        "messages",
                        r#"[{"role":"user","content":"Explain syntax"}]"#,
                    ),
                    ("tool_choice", "\"none\""),
                    ("stream", if stream { "true" } else { "false" }),
                    (
                        "chat_template_kwargs",
                        if phase > 0 {
                            r#"{"enable_thinking":true}"#
                        } else {
                            r#"{"enable_thinking":false}"#
                        },
                    ),
                ];
                if declared {
                    fields.push((
                        "tools",
                        r#"[{"type":"function","function":{"name":"f","parameters":
                            {"type":"object","properties":{},"additionalProperties":false}}}]"#,
                    ));
                }
                let mut response = chat(&backend, &hello_with(&fields));
                assert_eq!(response.status, 200);
                let (text, reasoning) = answer_and_reasoning(&mut response, stream);
                assert_eq!(
                    (text.as_str(), reasoning.as_str()),
                    (expected_answer, expected_thought),
                    "stream={stream} declared={declared} phase={phase}"
                );
            }
        }
    }
}

/// Templates render typed arguments with Jinja `tojson`, so replayed history
/// spells them the same way and reuses the turn's checkpoint.
#[test]
fn historical_typed_arguments_use_tojson() {
    let backend = FakeBackend::new();
    let body = r#"{"model":"test-model","messages":[{"role":"user","content":"edit"},
        {"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"edit",
          "arguments":"{\"path\":\"a.txt\",\"edits\":[{\"oldText\":\"a\",\"newText\":\"b\"}],\"n\":2}"}}]},
        {"role":"tool","tool_call_id":"call_1","content":"ok"}]}"#;
    assert_eq!(chat(&backend, body).status, 200);
    let arguments = backend.last_request().messages[1].tool_calls[0]
        .arguments
        .clone();
    let values: Vec<(&str, &str, bool)> = arguments
        .iter()
        .map(|a| (a.name.as_str(), a.value.as_str(), a.is_string))
        .collect();
    assert_eq!(
        values,
        [
            ("path", "a.txt", true),
            ("edits", r#"[{"oldText": "a", "newText": "b"}]"#, false),
            ("n", "2", false),
        ]
    );
}

/// Qwen parameters whose union admits other types than strings try those
/// first, as llama.cpp's qwen3-coder parser does; plain strings keep quotes
/// and literal JSON text.
#[test]
fn qwen_union_tool_types() {
    let backend = FakeBackend::new();
    backend.set_pieces(&["<tool_call>\n<function=record>\n\
        <parameter=content>\n#include \"a.h\"\n42\n</parameter>\n\
        <parameter=provider>\nexa\n</parameter>\n\
        <parameter=args>\n{\"key\": 1}\n</parameter>\n\
        <parameter=label>\nnull\n</parameter>\n\
        <parameter=note>\nplain text\n</parameter>\n\
        <parameter=limit>\n3\n</parameter>\n\
        </function>\n</tool_call>"]);
    let body = r#"{"model":"test-model","messages":[{"role":"user","content":"call record"}],
        "tools":[{"type":"function","function":{"name":"record","parameters":{
          "type":"object","properties":{
            "content":{"type":"string"},
            "provider":{"anyOf":[{"type":"string","const":"brave"},{"type":"string","const":"exa"}]},
            "args":{"anyOf":[{"type":"string"},{"type":"object","additionalProperties":true}]},
            "label":{"anyOf":[{"type":"string"},{"type":"null"}]},
            "note":{"anyOf":[{"type":"string"},{"type":"null"}]},
            "limit":{"anyOf":[{"type":"integer"},{"type":"null"}]}},
          "required":["content"]}}}],
        "tool_choice":"auto"}"#;
    let response = chat(&backend, body);
    assert_eq!(response.status, 200);
    let output = json::parse(&response.body).unwrap();
    let arguments = output.find("choices").unwrap().items()[0]
        .find("message")
        .unwrap()
        .find("tool_calls")
        .unwrap()
        .items()[0]
        .find("function")
        .unwrap()
        .member_str("arguments", "");
    let expected = json::parse_str(
        r##"{"content":"#include \"a.h\"\n42","provider":"exa","args":{"key":1},
            "label":null,"note":"plain text","limit":3}"##,
    )
    .unwrap();
    assert_eq!(
        json::parse_str(&arguments).unwrap().dump().unwrap(),
        expected.dump().unwrap()
    );
}

#[test]
fn model_sampling_defaults() {
    use gufo_core::sampling::{SamplingOverrides, TextModelPreset};
    let backend = FakeBackend::new();
    {
        let mut config = backend.config();
        config.defaults.model = TextModelPreset::Qwen38;
        config.defaults.supplied = SamplingOverrides::default();
    }
    let send = |fields: &str| {
        let response = chat(
            &backend,
            &format!(
                r#"{{"model":"test-model","messages":[{{"role":"user","content":"hello"}}]{fields}}}"#
            ),
        );
        assert_eq!(response.status, 200, "model default request accepted");
        backend
            .observed
            .lock()
            .expect("observed")
            .last_sampling
            .clone()
    };
    let config = send("");
    assert!(
        config.temperature == 1.0
            && config.top_p == 0.95
            && config.top_k == 20
            && config.presence_penalty == 0.0,
        "Qwen defaults follow the thinking-on template: {config:?}"
    );
    let config = send(r#","chat_template_kwargs":{"enable_thinking":false}"#);
    assert!(
        config.temperature == 0.7 && config.top_p == 0.8 && config.presence_penalty == 1.5,
        "request thinking-off selects its model preset"
    );
    backend.config().reasoning_defaults.enabled = Some(false);
    let config = send(r#","temperature":null,"top_p":null,"presence_penalty":null"#);
    assert!(
        config.temperature == 0.7 && config.presence_penalty == 1.5,
        "nulls inherit the effective server thinking preset"
    );
    let config = send(r#","reasoning_effort":"high""#);
    assert!(
        config.temperature == 1.0 && config.presence_penalty == 0.0,
        "an effort enables thinking before sampling resolves"
    );
    {
        let mut config = backend.config();
        config.defaults.sampling.temperature = 0.2;
        config.defaults.sampling.top_k = 0;
        config.defaults.supplied.temperature = true;
        config.defaults.supplied.top_k = true;
    }
    let config = send("");
    assert!(
        config.temperature == 0.2
            && config.top_k == 0
            && config.top_p == 0.8
            && config.presence_penalty == 1.5,
        "a partial server override keeps the other model defaults"
    );
    let config = send(r#","temperature":0,"presence_penalty":0,"top_k":40"#);
    assert!(
        config.temperature == 0.0 && config.presence_penalty == 0.0 && config.top_k == 40,
        "an explicit request zero overrides server and model presets"
    );
    backend.config().defaults.supplied = SamplingOverrides::ALL;
    let config = send("");
    assert_eq!(
        config,
        backend.config().defaults.sampling,
        "fully configured defaults ignore the preset"
    );
}

#[test]
fn backend_sampling_defaults() {
    let backend = FakeBackend::new();
    backend.config().defaults = SamplingDefaults {
        max_tokens: 37,
        sampling: SamplingConfig::with_temperature(0.25),
        ..SamplingDefaults::default()
    };
    let response = chat(&backend, HELLO);
    assert_eq!(response.status, 200, "Defaulted request is accepted");
    {
        let observed = backend.observed.lock().expect("observed");
        assert_eq!(
            observed.last_max_tokens, 37,
            "Backend max-token default reaches generation"
        );
        assert!(
            observed.last_sampling.temperature > 0.24 && observed.last_sampling.temperature < 0.26,
            "Backend temperature default reaches generation"
        );
    }
    let response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"hello"}],"max_tokens":11,"temperature":0}"#,
    );
    assert_eq!(response.status, 200, "Sampling override is accepted");
    let observed = backend.observed.lock().expect("observed");
    assert_eq!(
        observed.last_max_tokens, 11,
        "Explicit max tokens override the backend default"
    );
    assert_eq!(
        observed.last_sampling.temperature, 0.0,
        "Explicit temperature overrides the backend default"
    );
}

#[test]
fn complete_tool_definitions_reach_backend() {
    let backend = FakeBackend::new();
    let response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"Emit a value."}],
            "tools":[{"type":"function","function":{"parameters":{"type":"object","additionalProperties":false},
              "strict":true,"description":"","name":"emit"},"vendor":{"version":2}}]}"#,
    );
    let last = backend.last_request();
    assert!(
        response.status == 200 && last.tools.len() == 1,
        "complete function definition reaches the backend"
    );
    assert_eq!(
        last.tools[0].definition_json,
        r#"{"type":"function","function":{"parameters":{"type":"object","additionalProperties":false},"strict":true,"description":"","name":"emit"},"vendor":{"version":2}}"#,
        "Valid nested definitions retain every field and its original order"
    );
}

#[test]
fn flat_tool_fields_reach_backend() {
    for strict in ["true", "false", "null"] {
        let backend = FakeBackend::new();
        let mut body = json::parse_str(
            r#"{"model":"test-model","messages":[{"role":"user","content":"use f"}],"tools":[]}"#,
        )
        .expect("json");
        let mut function = json::parse_str(
            r#"{"name":"f","description":"","parameters":{"type":"object","additionalProperties":false},"strict":null,"vendor":{"version":2}}"#,
        )
        .expect("json");
        function.set("strict", json::parse_str(strict).expect("json"));
        let flat = function.clone().with("type", "function");
        body.member_mut("tools").push(flat);
        let response = chat(&backend, &body.dump().expect("dump"));
        let flat_tools = backend.last_request().tools;
        assert!(
            response.status == 200 && flat_tools.len() == 1,
            "Flat function with additional fields is accepted"
        );
        let definition = json::parse_str(&flat_tools[0].definition_json).expect("json");
        let nested = definition.find("function").expect("function");
        assert!(
            nested.dump().expect("dump") == function.dump().expect("dump")
                && !definition.contains("strict")
                && !definition.contains("vendor"),
            "All flat function fields survive DS4's nested function extraction"
        );
        let canonical = Value::object()
            .with("type", "function")
            .with("function", function);
        body.set("tools", vec![canonical]);
        let nested_response = chat(&backend, &body.dump().expect("dump"));
        assert_eq!(nested_response.status, 200, "Nested equivalent is accepted");
        assert_eq!(
            backend.last_request().tools[0].definition_json,
            flat_tools[0].definition_json,
            "Flat and nested tools reach the template identically"
        );
    }
}

#[test]
fn all_sampling_controls_reach_backend() {
    let backend = FakeBackend::new();
    let response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"hello"}],"temperature":0.8,"top_k":40,
            "top_p":0.9,"min_p":0.05,"min_keep":3,"seed":123,"repeat_penalty":1.1,"repeat_last_n":32,
            "frequency_penalty":0.25,"presence_penalty":0.5}"#,
    );
    assert_eq!(
        response.status, 200,
        "Complete sampling request is accepted"
    );
    let sampling = backend
        .observed
        .lock()
        .expect("observed")
        .last_sampling
        .clone();
    assert!(
        sampling.temperature > 0.79 && sampling.temperature < 0.81,
        "temperature reaches backend"
    );
    assert_eq!(sampling.top_k, 40, "top-k reaches backend");
    assert!(
        sampling.top_p > 0.89 && sampling.top_p < 0.91,
        "top-p reaches backend"
    );
    assert!(
        sampling.min_p > 0.04 && sampling.min_p < 0.06,
        "min-p reaches backend"
    );
    assert_eq!(sampling.min_keep, 3, "min-keep reaches backend");
    assert_eq!(sampling.seed, 123, "seed reaches backend");
    assert!(
        sampling.repeat_penalty > 1.09 && sampling.repeat_penalty < 1.11,
        "repeat penalty reaches backend"
    );
    assert_eq!(sampling.repeat_last_n, 32, "repeat window reaches backend");
    assert!(
        sampling.frequency_penalty > 0.24 && sampling.frequency_penalty < 0.26,
        "frequency penalty reaches backend"
    );
    assert!(
        sampling.presence_penalty > 0.49 && sampling.presence_penalty < 0.51,
        "presence penalty reaches backend"
    );
}

#[test]
fn unsupported_sampling_controls_are_rejected() {
    let backend = FakeBackend::new();
    for choices in [0.0, 1.4, 2.0, 1e100] {
        let mut body = json::parse_str(HELLO).expect("json");
        body.set("n", choices);
        assert_eq!(
            chat(&backend, &body.dump().expect("dump")).status,
            400,
            "n must be exactly one, without truncation or overflow"
        );
    }
    for field in [
        "draft_temperature",
        "temperature_draft",
        "draft_top_k",
        "draft_top_p",
        "draft_min_p",
        "draft_seed",
        "draft_policy",
        "samplers",
        "typical_p",
        "tfs_z",
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "dynatemp_range",
        "dynatemp_exponent",
        "xtc_probability",
        "xtc_threshold",
        "dry_multiplier",
        "dry_base",
        "dry_allowed_length",
        "dry_penalty_last_n",
        "dry_sequence_breakers",
        "top_n_sigma",
        "logit_bias",
    ] {
        let mut body = json::parse_str(HELLO).expect("json");
        body.set(field, 0.8);
        let response = chat(&backend, &body.dump().expect("dump"));
        assert!(
            response.status == 400 && body_text(&response).contains("unsupported_sampling"),
            "unsupported sampling must not be silently ignored: {field}"
        );
    }
}

/// The API's sampling ranges are inclusive and checked before the value
/// narrows to `f32`.
#[test]
fn sampling_ranges() {
    for (field, minimum, maximum) in [
        ("temperature", 0.0, 2.0),
        ("top_p", 0.0, 1.0),
        ("min_p", 0.0, 1.0),
        ("frequency_penalty", -2.0, 2.0),
        ("presence_penalty", -2.0, 2.0),
    ] {
        for value in [minimum - 1e-8, minimum, maximum, maximum + 1e-8] {
            let backend = FakeBackend::new();
            backend.set_pieces(&["ok"]);
            let body = Value::object()
                .with("model", "test-model")
                .with(
                    "messages",
                    json::parse_str(r#"[{"role":"user","content":"hello"}]"#).unwrap(),
                )
                .with(field, value);
            let response = chat(&backend, &body.dump().unwrap());
            let rejected = value < minimum || value > maximum;
            assert_eq!(
                response.status,
                if rejected { 400 } else { 200 },
                "{field}={value}"
            );
            if rejected {
                assert!(body_text(&response).contains(&format!("invalid_{field}")));
            }
        }
    }
}

#[test]
fn assistant_reasoning_content_reaches_backend() {
    let backend = FakeBackend::new();
    backend.set_pieces(&["Blue"]);
    let response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"Name one color."},
            {"role":"assistant","reasoning_content":"I should answer concisely.","content":"Red"},
            {"role":"user","content":"Name another."}],"max_tokens":1}"#,
    );
    assert_eq!(
        response.status, 200,
        "Assistant reasoning history is accepted"
    );
    let last = backend.last_request();
    assert_eq!(
        last.messages.len(),
        3,
        "Complete reasoning history reaches the backend"
    );
    assert_eq!(
        last.messages[1].thought, "I should answer concisely.",
        "Assistant reasoning_content reaches the model template"
    );
    assert_eq!(
        last.messages[1].content, "Red",
        "Assistant visible content remains separate from reasoning"
    );
}

#[test]
fn pi_reasoning_controls_and_output_framing() {
    let backend = FakeBackend::new();
    backend.set_pieces(&["I should verify this.", "</think>\n\n", "Forty-two."]);
    let response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"What is six times seven?"}],
            "reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"high",
            "preserve_thinking":false}}"#,
    );
    assert_eq!(response.status, 200, "Pi reasoning request is accepted");
    let last = backend.last_request();
    assert_eq!(
        last.reasoning.enabled,
        Some(true),
        "Pi enable_thinking reaches the backend"
    );
    assert_eq!(
        last.reasoning.effort,
        Some(ReasoningEffort::High),
        "Pi reasoning effort reaches the backend"
    );
    assert_eq!(
        last.reasoning.preserve_thinking,
        Some(false),
        "Pi preservation control reaches the backend"
    );
    let text = body_text(&response);
    assert!(
        text.contains(r#""reasoning_content":"I should verify this.""#),
        "Prompt-opened reasoning is returned separately"
    );
    assert!(
        text.contains(r#""content":"Forty-two.""#),
        "Visible answer excludes reasoning"
    );
    for effort in ["low", "medium", "xhigh"] {
        let native = chat(
            &backend,
            &format!(
                r#"{{"model":"test-model","messages":[{{"role":"user","content":"hello"}}],"chat_template_kwargs":{{"reasoning_effort":"{effort}"}}}}"#
            ),
        );
        assert!(
            native.status == 200 && backend.last_request().reasoning.enabled == Some(true),
            "Native effort alone enables thinking"
        );
    }
    backend.set_pieces(&["Done."]);
    let disabled = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"hello"}],
            "chat_template_kwargs":{"enable_thinking":false,"reasoning_effort":"low"}}"#,
    );
    assert!(
        disabled.status == 200 && backend.last_request().reasoning.enabled == Some(false),
        "Template enable_thinking=false suppresses the configured effort"
    );
    assert!(
        body_text(&disabled).contains(r#""content":"Done.""#),
        "Explicit thinking-off produces visible content"
    );
    let vision_ids = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"hello"}],"chat_template_kwargs":{"add_vision_id":true}}"#,
    );
    assert!(
        vision_ids.status == 200 && backend.last_request().add_vision_id,
        "Official add_vision_id option reaches the model formatter"
    );
    let invalid_ids = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"hello"}],"chat_template_kwargs":{"add_vision_id":"true"}}"#,
    );
    assert_eq!(invalid_ids.status, 400, "add_vision_id requires a boolean");
}

#[test]
fn pi_native_deepseek_thinking_object() {
    let backend = FakeBackend::new();
    backend.set_pieces(&["Check.", "</think>", "Done."]);
    let response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"Check this."}],
            "thinking":{"type":"enabled"},"reasoning_effort":"high"}"#,
    );
    assert_eq!(
        response.status, 200,
        "Pi native DeepSeek thinking object is accepted"
    );
    let last = backend.last_request();
    assert_eq!(
        last.reasoning.enabled,
        Some(true),
        "Pi DeepSeek thinking.type enables reasoning"
    );
    assert_eq!(
        last.reasoning.effort,
        Some(ReasoningEffort::High),
        "Pi DeepSeek reasoning effort reaches the backend"
    );
    let disabled = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"Answer directly."}],"thinking":{"type":"disabled"}}"#,
    );
    assert_eq!(
        disabled.status, 200,
        "Pi native DeepSeek disabled thinking object is accepted"
    );
    assert_eq!(
        backend.last_request().reasoning.enabled,
        Some(false),
        "Pi DeepSeek thinking.type disables reasoning"
    );
}

#[test]
fn streaming_prompt_opened_reasoning() {
    let backend = FakeBackend::new();
    backend.set_pieces(&["Check", " carefully", "</thi", "nk>\n\n", "Done"]);
    let mut response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"Check it."}],"reasoning_effort":"xhigh","stream":true}"#,
    );
    assert_eq!(
        response.status, 200,
        "Streaming reasoning request is accepted"
    );
    let output = drain(&mut response);
    assert!(
        output.contains(r#""reasoning_content":"Check""#),
        "Streaming reasoning uses reasoning_content deltas"
    );
    assert!(
        output.contains(r#""content":"Done""#),
        "Streaming answer switches to content after think end"
    );
    assert!(
        !output.contains(r#""content":"Check""#),
        "Reasoning is never exposed as visible content"
    );
}

#[test]
fn conflicting_reasoning_controls_are_rejected() {
    let backend = FakeBackend::new();
    let response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"hello"}],"reasoning_effort":"high",
            "chat_template_kwargs":{"enable_thinking":false}}"#,
    );
    assert_eq!(
        response.status, 400,
        "Conflicting reasoning controls are rejected"
    );
    assert!(
        body_text(&response).contains("invalid_reasoning"),
        "Reasoning conflict has a stable error code"
    );
    assert_eq!(
        backend.chat_calls.load(Ordering::SeqCst),
        0,
        "Invalid reasoning request never reaches generation"
    );
}

#[test]
fn wrong_model_is_rejected() {
    let backend = FakeBackend::new();
    let response = chat(
        &backend,
        r#"{"model":"wrong-model","messages":[{"role":"user","content":"hello"}]}"#,
    );
    assert_eq!(response.status, 404, "Unknown model is rejected");
    assert!(
        body_text(&response).contains("model_not_found"),
        "Unknown model returns a stable error code"
    );
    assert_eq!(
        backend.chat_calls.load(Ordering::SeqCst),
        0,
        "Rejected request never reaches the backend"
    );
}

#[test]
fn client_identity_reaches_backend() {
    let backend = FakeBackend::new();
    backend.set_pieces(&["ok"]);
    let dynamic: Arc<dyn TextGenerationBackend> = backend.clone();
    for header in ["pi-agent-2", "contains spaces", "rotated"] {
        let mut req = request(HELLO);
        req.headers = vec![("X-Client-ID".into(), header.into())];
        req.client_id = "192.0.2.7".into();
        let response = handle_openai_chat(&req, &dynamic).expect("handler");
        assert!(
            response.status == 200 && backend.last_request().client_id == req.client_id,
            "untrusted headers cannot change the transport identity"
        );
    }
}

#[test]
fn streaming_overload_is_rejected_before_headers() {
    let backend = FakeBackend::new();
    backend.config().reject_on_start = Some(GenerationErrorCode::QueueFull);
    let response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
    );
    assert_eq!(
        response.status, 429,
        "streaming overload is rejected synchronously"
    );
    assert!(
        response.streaming_body.is_none(),
        "overloaded stream does not commit successful SSE headers"
    );
    assert!(
        body_text(&response).contains("queue_full"),
        "overload response carries a stable retry code"
    );
    assert!(
        response
            .headers
            .iter()
            .any(|(n, v)| n == "Retry-After" && v == "1"),
        "retryable overload advertises Retry-After"
    );
}

#[test]
fn image_parts_retain_order_and_identity() {
    let backend = FakeBackend::new();
    backend.set_pieces(&["ok"]);
    let response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":[
            {"type":"text","text":"left"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,AQID","detail":"auto"}},
            {"type":"text","text":"right"},
            {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,BAUG"}}]}]}"#,
    );
    assert_eq!(
        response.status, 200,
        "image content parts reach the backend"
    );
    let message = backend.last_request().messages[0].clone();
    assert!(
        message.content == "leftright" && message.images.len() == 2,
        "images do not become text placeholders before model preparation"
    );
    assert!(
        message.images[0].offset == 4 && message.images[1].offset == 9,
        "image placement relative to text is preserved"
    );
    assert!(
        *message.images[0].bytes == [1, 2, 3] && *message.images[1].bytes == [4, 5, 6],
        "each image retains its own decoded transport bytes"
    );
    for body in [
        r#"{"messages":[{"role":"assistant","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}]}]}"#,
        r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID","detail":"ultra"}}]}]}"#,
        r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID","detail":1}}]}]}"#,
        r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"file:///tmp/image.png"}}]}]}"#,
    ] {
        // The C++ test omits "model", which is rejected first; the images
        // are then never read. Keep the request as written.
        assert_eq!(
            chat(&backend, body).status,
            400,
            "unsupported image role, policy and URL are rejected"
        );
    }
    // Codex sends "high" from view_image; every standard value behaves as
    // auto.
    for detail in ["auto", "low", "high", "original"] {
        let body = format!(
            r#"{{"model":"test-model","messages":[{{"role":"user","content":[
            {{"type":"image_url","image_url":{{"url":"data:image/png;base64,AQID","detail":"{detail}"}}}}]}}]}}"#
        );
        assert!(
            chat(&backend, &body).status == 200
                && backend.last_request().messages[0].images.len() == 1,
            "standard image detail {detail} is accepted as auto"
        );
    }
}

/// No image count: every image part within the byte budget reaches the
/// model, in one message or split across messages.
#[test]
fn image_count_and_byte_budget() {
    let part =
        json::parse_str(r#"{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}"#)
            .expect("json");
    for count in [17usize, 257] {
        for split in [false, true] {
            let backend = FakeBackend::new();
            let message =
                |parts: Vec<Value>| Value::object().with("role", "user").with("content", parts);
            let messages = if split {
                vec![message(vec![part.clone()]); count]
            } else {
                vec![message(vec![part.clone(); count])]
            };
            let body = Value::object()
                .with("model", "test-model")
                .with("messages", messages);
            let response = chat(&backend, &body.dump().expect("dump"));
            let observed: usize = backend
                .last_request()
                .messages
                .iter()
                .map(|m| m.images.len())
                .sum();
            assert_eq!(
                (response.status, observed),
                (200, count),
                "count {count} split {split}"
            );
        }
    }
}

/// Argument text that looks like markup of every dialect, with quotes,
/// escapes, non-ASCII and a trailing CR.
const LITERAL_ARGUMENT: &str = concat!(
    " <think>literal</think><tool_call></tool_call>",
    "<tool_call>{\"name\":\"f\",\"arguments\":{\"text\":\"nested\"}}</tool_call>",
    "<｜DSML｜tool_calls><｜DSML｜invoke name=\"f\"></｜DSML｜invoke></｜DSML｜tool_calls>",
    "</function></｜DSML｜invoke> \"42\" \\\nπ🦉\n \r"
);

/// One call to `f` carrying [`LITERAL_ARGUMENT`] in each native syntax.
fn literal_calls() -> [String; 3] {
    let arguments = json::Value::object()
        .with("text", LITERAL_ARGUMENT)
        .dump()
        .expect("dump");
    [
        format!(
            "<tool_call>\n<function=f>\n<parameter=text>\n{LITERAL_ARGUMENT}\n</parameter>\n\
             </function>\n</tool_call>"
        ),
        format!(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"text\" \
             string=\"true\">{LITERAL_ARGUMENT}</｜DSML｜parameter>\n</｜DSML｜invoke>\n\
             </｜DSML｜tool_calls>"
        ),
        format!("<tool_call>{{\"name\":\"f\",\"arguments\":{arguments}}}</tool_call>"),
    ]
}

/// Native and JSON calls keep their exact arguments in every mode and over
/// both APIs, and their tags never reach the reasoning; automatic
/// non-strict tools are constrained too.
#[test]
fn native_tool_transports() {
    let expected = json::Value::object()
        .with("text", LITERAL_ARGUMENT)
        .dump()
        .expect("dump");
    for call in literal_calls() {
        for thinking in [false, true] {
            for stream in [false, true] {
                for parallel in [false, true] {
                    for over_responses in [false, true] {
                        for constrained in [false, true] {
                            let backend = FakeBackend::new();
                            if constrained {
                                backend.config().tool_format =
                                    Some(if call.starts_with("<｜DSML｜tool_calls>") {
                                        ToolFormat::DeepSeek
                                    } else {
                                        ToolFormat::Qwen
                                    });
                            }
                            let raw = format!(
                                "{}{call} \n",
                                if thinking { "Reasoning.</think>" } else { "" }
                            );
                            backend.config().pieces = raw.bytes().map(|byte| vec![byte]).collect();
                            let mut body = json::parse_str(
                                r#"{"model":"test-model","messages":[{"role":"user","content":"call f"}],
                                "tools":[{"type":"function","function":{"name":"f","strict":false,
                                  "parameters":{"type":"object","properties":{"text":{"type":"string"}},
                                  "required":["text"],"additionalProperties":false}}}]}"#,
                            )
                            .expect("json");
                            body.set("stream", stream);
                            body.set("parallel_tool_calls", parallel);
                            body.set("tool_choice", if constrained { "required" } else { "auto" });
                            body.set(
                                "chat_template_kwargs",
                                Value::object().with("enable_thinking", thinking),
                            );
                            let mut response = if over_responses {
                                responses(&backend, &body, thinking, 256, stream)
                            } else {
                                chat(&backend, &body.dump().expect("dump"))
                            };
                            assert_eq!(response.status, 200, "native tool request succeeds");
                            let (mut arguments, mut reasoning, mut calls) =
                                (String::new(), String::new(), 0);
                            for event in events(&mut response, stream) {
                                if over_responses {
                                    if event.member_str("type", "")
                                        == "response.function_call_arguments.done"
                                    {
                                        assert!(
                                            event.member_str("name", "") == "f"
                                                && event.member_str("arguments", "") == expected,
                                            "Responses argument completion names the function"
                                        );
                                    }
                                    let result = if stream {
                                        event.find("response")
                                    } else {
                                        Some(&event)
                                    };
                                    if let Some(result) =
                                        result.filter(|r| r.member_str("status", "") == "completed")
                                    {
                                        for item in result.find("output").expect("output").items() {
                                            if item.member_str("type", "") == "function_call" {
                                                calls += 1;
                                                arguments = item.member_str("arguments", "");
                                            }
                                        }
                                    }
                                    continue;
                                }
                                for choice in event.find("choices").map_or(&[][..], |c| c.items()) {
                                    let Some(message) =
                                        choice.find(if stream { "delta" } else { "message" })
                                    else {
                                        continue;
                                    };
                                    reasoning += &message.member_str("reasoning_content", "");
                                    for tool in
                                        message.find("tool_calls").map_or(&[][..], |t| t.items())
                                    {
                                        calls += usize::from(
                                            !stream || !tool.member_str("id", "").is_empty(),
                                        );
                                        arguments += &tool
                                            .find("function")
                                            .expect("function")
                                            .member_str("arguments", "");
                                    }
                                }
                            }
                            assert!(
                                calls == 1 && arguments == expected,
                                "arguments survive: {call:?} thinking={thinking} stream={stream} \
                                 parallel={parallel} responses={over_responses} \
                                 constrained={constrained}: {calls} {arguments}"
                            );
                            assert!(
                                backend.last_request().constrained_tools,
                                "automatic non-strict tools also bind decoding constraints"
                            );
                            assert!(
                                !reasoning.contains("literal"),
                                "argument tags never enter the reasoning"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// A truncated call is never recovered from a nested literal call: only a
/// complete call with its full arguments is reported.
#[test]
fn native_tool_truncation_keeps_complete_arguments_only() {
    let expected = json::Value::object()
        .with("text", LITERAL_ARGUMENT)
        .dump()
        .expect("dump");
    for call in literal_calls() {
        for finish in [FinishReason::Length, FinishReason::StopSequence] {
            let boundaries: Vec<usize> = (0..call.len())
                .filter(|&length| call.is_char_boundary(length))
                .collect();
            for length in boundaries {
                let backend = FakeBackend::new();
                backend.set_pieces(&[&call[..length]]);
                backend.config().finish_reason = finish;
                let response = chat(
                    &backend,
                    r#"{"model":"test-model","messages":[{"role":"user","content":"call f"}],
                       "tool_choice":"required","tools":[{"type":"function","function":{
                         "name":"f","parameters":{}}}]}"#,
                );
                assert_eq!(
                    response.status, 200,
                    "tool truncation terminates successfully"
                );
                let output = json::parse(&response.body).expect("body");
                let message = output.find("choices").expect("choices").items()[0]
                    .find("message")
                    .expect("message");
                if let Some(emitted) = message.find("tool_calls") {
                    assert!(
                        emitted.len() == 1
                            && emitted.items()[0]
                                .find("function")
                                .expect("function")
                                .member_str("arguments", "")
                                == expected,
                        "truncation at {length} of {call:?} keeps complete arguments only"
                    );
                }
            }
        }
    }
}

/// Argument types follow `$ref` properties and a referenced root schema
///.
#[test]
fn native_referenced_argument_types() {
    let arguments = json::parse_str(r#"{"n":42,"b":true,"a":[1],"o":{"x":2},"s":"42","z":null}"#)
        .expect("args");
    let schema = json::parse_str(
        r##"{"type":"object","properties":{
          "n":{"$ref":"#/$defs/number~1type~0"},
          "b":{"$ref":"#/$defs/boolean"},"a":{"$ref":"#/$defs/array"},
          "o":{"$ref":"#/$defs/object"},"s":{"$ref":"#/$defs/string"},
          "z":{"$ref":"#/$defs/null"}},
          "required":["n","b","a","o","s","z"],"additionalProperties":false,
          "$defs":{"number/type~":{"type":"integer"},"boolean":{"type":"boolean"},
            "array":{"type":"array","items":{"type":"integer"}},
            "object":{"type":"object","properties":{"x":{"type":"integer"}},
                      "required":["x"],"additionalProperties":false},
            "string":{"type":"string"},"null":{"type":"null"}}}"##,
    )
    .expect("schema");
    for root_reference in [false, true] {
        let parameters = if root_reference {
            let mut object = json::Value::object();
            for (key, value) in schema.members() {
                if key != "$defs" {
                    object.set(key, value.clone());
                }
            }
            let mut definitions = schema.find("$defs").expect("defs").clone();
            definitions.set("Arguments", object);
            json::Value::object()
                .with("$ref", "#/$defs/Arguments")
                .with("$defs", definitions)
        } else {
            schema.clone()
        };
        let mut call = "<tool_call>\n<function=f>\n".to_owned();
        for (name, value) in arguments.members() {
            let text = if value.is_string() {
                value.str().to_owned()
            } else {
                value.dump().expect("dump")
            };
            call += &format!("<parameter={name}>\n{text}\n</parameter>\n");
        }
        call += "</function>\n</tool_call>";
        let backend = FakeBackend::new();
        backend.set_pieces(&[call.as_str()]);
        let body = json::Value::object()
            .with("model", "test-model")
            .with(
                "messages",
                vec![
                    json::Value::object()
                        .with("role", "user")
                        .with("content", "call f"),
                ],
            )
            .with(
                "tools",
                vec![
                    json::Value::object().with("type", "function").with(
                        "function",
                        json::Value::object()
                            .with("name", "f")
                            .with("strict", true)
                            .with("parameters", parameters),
                    ),
                ],
            )
            .with("tool_choice", "required");
        let response = chat(&backend, &body.dump().expect("dump"));
        assert_eq!(
            response.status,
            200,
            "referenced native tool succeeds: {}",
            body_text(&response)
        );
        let output = json::parse(&response.body).expect("body");
        let encoded = output.find("choices").expect("choices").items()[0]
            .find("message")
            .and_then(|m| m.find("tool_calls"))
            .expect("tool calls")
            .items()[0]
            .find("function")
            .expect("function")
            .member_str("arguments", "");
        assert_eq!(
            encoded,
            arguments.dump().expect("dump"),
            "native argument types follow property and root references ({root_reference})"
        );
    }
}

/// The arguments of a Qwen chat response: every streamed delta's, or the
/// first buffered call's, with the visible content.
fn qwen_call_arguments(response: &mut HttpResponse, stream: bool) -> (String, String) {
    let (mut arguments, mut content) = (String::new(), String::new());
    for event in events(response, stream) {
        for choice in event.find("choices").map_or(&[][..], |c| c.items()) {
            let Some(message) = choice.find(if stream { "delta" } else { "message" }) else {
                continue;
            };
            content += &message.member_str("content", "");
            for call in message.find("tool_calls").map_or(&[][..], |c| c.items()) {
                arguments += &call
                    .find("function")
                    .expect("function")
                    .member_str("arguments", "");
                if !stream {
                    break;
                }
            }
        }
    }
    (arguments, content)
}

/// Schemas that used to force a JSON envelope stay native (upstream #438).
/// Their raw values are typed as llama.cpp's qwen3-coder parser types them:
/// a schema that admits other JSON types tries them first, an exact string
/// stays text.
#[test]
fn native_argument_typing_matrix() {
    let cases: &[(&str, &str, &str)] = &[
        (r#"{"type":"string","pattern":"^[0-9]+$"}"#, "42", r#""42""#),
        (
            r#"{"type":"string","format":"uri"}"#,
            "https://a.b",
            r#""https://a.b""#,
        ),
        (
            r#"{"type":"string","default":"x","examples":["y"]}"#,
            "true",
            r#""true""#,
        ),
        (
            r#"{"oneOf":[{"type":"string"},{"type":"integer"}]}"#,
            "42",
            "42",
        ),
        (
            r#"{"oneOf":[{"type":"string"},{"type":"integer"}]}"#,
            "forty",
            r#""forty""#,
        ),
        (
            r#"{"allOf":[{"type":"string"},{"minLength":1}]}"#,
            "42",
            r#""42""#,
        ),
        (
            r#"{"allOf":[{"type":"integer"},{"minimum":1}]}"#,
            "42",
            "42",
        ),
        (r#"{"const":"42"}"#, "42", r#""42""#),
        (r#"{"enum":["true","false"]}"#, "true", r#""true""#),
        (r#"{"enum":["null","42"]}"#, "null", r#""null""#),
        (
            r#"{"anyOf":[{"const":"42"},{"type":"null"}]}"#,
            "42",
            r#""42""#,
        ),
        (r#"{"enum":["42",17]}"#, "17", "17"),
        (r#"{"type":"boolean"}"#, "True", "true"),
        (r#"{"minLength":1}"#, "42", r#""42""#),
        (r#"{"pattern":".*"}"#, "true", r#""true""#),
        (r#"{"not":{"type":"null"}}"#, "plain", r#""plain""#),
        (r#"{"type":["string","null"]}"#, "null", "null"),
        (
            r#"{"anyOf":[{"type":"integer"},{"type":"null"}]}"#,
            "null",
            "null",
        ),
        (
            r#"{"type":"integer","exclusiveMinimum":0,"maximum":9007199254740991}"#,
            "240000",
            "240000",
        ),
        (
            r#"{"type":"object","properties":{"x":{"type":"integer"}}}"#,
            r#"{"x": 1}"#,
            r#"{"x":1}"#,
        ),
        (
            r#"{"type":"array","items":{"type":"object"}}"#,
            r#"[{"k": "v"}]"#,
            r#"[{"k":"v"}]"#,
        ),
        // A recursive schema the grammar cannot enforce keeps its JSON value.
        (r##"{"$ref":"#"}"##, "5", "5"),
        (
            r#"{"type":"string"}"#,
            "line 1\n<b>\"q\"</b>",
            r#""line 1\n<b>\"q\"</b>""#,
        ),
        (
            r#"{"type":"string"}"#,
            "line\n</parameter> is literal\nend",
            r#""line\n</parameter> is literal\nend""#,
        ),
        (
            r#"{"type":"string"}"#,
            "line\n</parameter> <parameter=other>literal\nend",
            r#""line\n</parameter> <parameter=other>literal\nend""#,
        ),
        (
            r#"{"type":"string"}"#,
            "line\n</parameter> </function>literal\nend",
            r#""line\n</parameter> </function>literal\nend""#,
        ),
    ];
    let canonical = |text: &str| {
        json::parse_str(text)
            .ok()
            .and_then(|value| value.dump().ok())
    };
    for &(schema, raw, expected) in cases {
        for root in [
            r#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object"}"#,
            r#"{"type":"object","additionalProperties":true}"#,
        ] {
            for stream in [false, true] {
                let parameters = json::parse_str(root)
                    .expect("root")
                    .with(
                        "properties",
                        Value::object().with("v", json::parse_str(schema).expect("schema")),
                    )
                    .with("required", vec![Value::from("v")]);
                let function = Value::object()
                    .with("name", "record")
                    .with("parameters", parameters);
                let body = json::parse_str(
                    r#"{"model":"test-model","messages":[{"role":"user","content":"call record"}],
                    "tool_choice":"required"}"#,
                )
                .expect("json")
                .with(
                    "tools",
                    vec![
                        Value::object()
                            .with("type", "function")
                            .with("function", function),
                    ],
                )
                .with("stream", stream);
                let backend = FakeBackend::new();
                backend.config().tool_format = Some(ToolFormat::Qwen);
                let call = format!(
                    "<tool_call>\n<function=record>\n<parameter=v>\n{raw}\n</parameter>\n\
                     </function>\n</tool_call>"
                );
                // A stream arrives byte by byte, past every partial argument.
                backend.config().pieces = if stream {
                    bytewise(&call)
                } else {
                    whole(&call)
                };
                let mut response = chat(&backend, &body.dump().expect("dump"));
                let label = format!("schema={schema} raw={raw:?} root={root} stream={stream}");
                assert_eq!(
                    response.status,
                    200,
                    "native typed call succeeds: {label}: {}",
                    body_text(&response)
                );
                let (arguments, content) = qwen_call_arguments(&mut response, stream);
                assert!(
                    !arguments.is_empty()
                        && canonical(&arguments) == canonical(&format!(r#"{{"v":{expected}}}"#)),
                    "native arguments are typed as llama.cpp types them: {label}: {arguments}"
                );
                assert!(
                    content.is_empty(),
                    "a native call leaves no visible framing: {label}: {content:?}"
                );
            }
        }
    }
    // Parseable JSON and Python-literal recovery must both respect a known
    // argument type. A malformed call must not invoke a tool with another
    // type.
    for (kind, raw) in [
        ("integer", "true"),
        ("integer", "True"),
        ("boolean", "1"),
        ("array", "{}"),
        ("object", "null"),
    ] {
        let parameters =
            json::parse_str(r#"{"type":"object","required":["v"],"additionalProperties":false}"#)
                .expect("json")
                .with(
                    "properties",
                    Value::object().with("v", Value::object().with("type", kind)),
                );
        let function = Value::object()
            .with("name", "record")
            .with("parameters", parameters);
        let body = json::parse_str(
            r#"{"model":"test-model","messages":[{"role":"user","content":"call record"}]}"#,
        )
        .expect("json")
        .with(
            "tools",
            vec![
                Value::object()
                    .with("type", "function")
                    .with("function", function),
            ],
        );
        let backend = FakeBackend::new();
        backend.config().tool_format = Some(ToolFormat::Qwen);
        backend.set_pieces(&[&format!(
            "<tool_call>\n<function=record>\n<parameter=v>\n{raw}\n</parameter>\n\
             </function>\n</tool_call>"
        )]);
        let response = chat(&backend, &body.dump().expect("dump"));
        assert_eq!(
            response.status, 200,
            "invalid native output is handled safely"
        );
        let output = json::parse(&response.body).expect("json");
        let message = output.find("choices").expect("choices").items()[0]
            .find("message")
            .expect("message");
        assert!(
            !message.contains("tool_calls"),
            "a value with the wrong declared type is not invoked: {kind} {raw:?}: {}",
            body_text(&response)
        );
    }
    // A declared name with surrounding spaces remains exact. llama.cpp's
    // JSON mapper trims it despite matching the exact key in its PEG grammar.
    for stream in [false, true] {
        let body = json::parse_str(
            r#"{"model":"test-model","messages":[{"role":"user","content":"call record"}],
            "tools":[{"type":"function","function":{"name":"record","strict":true,
              "parameters":{"type":"object","properties":{" value ":{"type":"string"}},
              "required":[" value "],"additionalProperties":false}}}],
            "tool_choice":"required"}"#,
        )
        .expect("json")
        .with("stream", stream);
        let backend = FakeBackend::new();
        backend.config().tool_format = Some(ToolFormat::Qwen);
        backend.set_pieces(&["<tool_call>\n<function=record>\n<parameter= value >\nx\n\
             </parameter>\n</function>\n</tool_call>"]);
        let mut response = chat(&backend, &body.dump().expect("dump"));
        assert_eq!(response.status, 200, "spaced parameter call succeeds");
        let (arguments, _) = qwen_call_arguments(&mut response, stream);
        assert_eq!(
            arguments, r#"{" value ":"x"}"#,
            "a declared spaced name keeps its exact spelling (stream={stream})"
        );
    }
}

/// The content deltas of a streamed chat response, and its tool calls.
fn streamed_content(output: &str) -> (String, usize) {
    let (mut content, mut calls) = (String::new(), 0);
    for data in output
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| !data.starts_with("[DONE]"))
    {
        let event = json::parse_str(data).expect("event");
        for choice in event.find("choices").map_or(&[][..], |c| c.items()) {
            if let Some(delta) = choice.find("delta") {
                content += &delta.member_str("content", "");
                calls += delta.find("tool_calls").map_or(0, json::Value::len);
            }
        }
    }
    (content, calls)
}

/// Prose that names the dialect's tags in quotes is content, never a call
///.
#[test]
fn prose_about_the_dialect_is_not_a_call() {
    let message = concat!(
        "Got it — no comment posted. Here's the explanation, just for you.\n\n",
        "## Why it was a bug\n\n",
        "**The setup.** The Qwen chat model was trained to signal tool calls with special ",
        "markup: `<tool_call>`. The server's job is to act as a **translator**: it watches ",
        "the raw token stream, converts that markup into the structured tool_calls field of ",
        "the OpenAI API, and makes sure the *rest* of the markup never appears as text.\n\n",
        "1. **Closing tags escaped into content.** When the call parsed, gufo removed the ",
        "opening envelope but let the tail — `</parameter>`, `</invoke>`, `</function>` — ",
        "escape into content.\n\n",
        "The client repeats `<invoke name=\"terminal\">` in history, so the model mimics that ",
        "dialect. Quoted, the shape is:\n\n",
        "```\n<invoke name=\"f\">\n<parameter name=\"text\">x</parameter>\n</invoke>\n```\n\n",
        "The same holds for `<|tool_call|>`, `<tool_call>` and for `<function=f>`. Nothing ",
        "above is a call being made, and none of it may be cut."
    );
    for stream in [false, true] {
        for bytewise in [false, true] {
            let backend = FakeBackend::new();
            let pieces: Vec<String> = if bytewise {
                message.chars().map(String::from).collect()
            } else {
                vec![message.to_owned()]
            };
            let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
            backend.set_pieces(&pieces);
            let mut response = chat(
                &backend,
                &format!(
                    r#"{{"model":"test-model","messages":[{{"role":"user","content":"why was it a bug"}}],
                    "tools":[{{"type":"function","function":{{"name":"f","parameters":{{
                      "type":"object","properties":{{"text":{{"type":"string"}}}},
                      "required":["text"]}}}}}}],"tool_choice":"auto","stream":{stream}}}"#
                ),
            );
            assert_eq!(
                response.status, 200,
                "an answer about the dialect is served"
            );
            let (content, calls) = if stream {
                streamed_content(&drain(&mut response))
            } else {
                let output = json::parse(&response.body).expect("body");
                let message = output.find("choices").expect("choices").items()[0]
                    .find("message")
                    .expect("message")
                    .clone();
                (
                    message.member_str("content", ""),
                    message.find("tool_calls").map_or(0, json::Value::len),
                )
            };
            assert_eq!(calls, 0, "prose that names the dialect is not a call");
            assert_eq!(
                content.trim(),
                message.trim(),
                "an answer about the dialect survives in full (stream={stream}, bytewise={bytewise})"
            );
        }
    }
}

/// Bracket-dense content is ordinary text in both transports.
#[test]
fn bracket_dense_content() {
    let dense = "<a>".repeat(2000);
    for tail in ["", "<|not_a_vocab|>", "<|not_a_vocab_entry"] {
        let backend = FakeBackend::new();
        let mut pieces: Vec<String> = dense
            .as_bytes()
            .chunks(90)
            .map(|chunk| String::from_utf8(chunk.to_vec()).expect("ascii"))
            .collect();
        if !tail.is_empty() {
            pieces.push(tail.to_owned());
        }
        let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
        backend.set_pieces(&pieces);
        let body = r#"{"model":"test-model","messages":[{"role":"user","content":"echo it"}],
            "reasoning_effort":"none","tools":[{"type":"function","function":{"name":"f",
            "parameters":{"type":"object"}}}]"#;
        let buffered = chat(&backend, &format!("{body}}}"));
        let output = json::parse(&buffered.body).expect("body");
        assert_eq!(
            output.find("choices").expect("choices").items()[0]
                .find("message")
                .expect("message")
                .member_str("content", ""),
            format!("{dense}{tail}"),
            "bracket-dense prose and its tail survive verbatim"
        );
        let mut streamed = chat(&backend, &format!(r#"{body},"stream":true}}"#));
        assert_eq!(
            streamed_content(&drain(&mut streamed)).0,
            format!("{dense}{tail}"),
            "the streaming path retains bracket-dense prose and its tail"
        );
    }
}

/// Comparison prose streams before generation completes.
#[test]
fn less_than_prose_streams_before_completion() {
    let backend = FakeBackend::new();
    backend.set_pieces(&["3 < 5 and x < y", " are comparisons."]);
    backend.config().block_after_first_piece = true;
    let mut response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"user","content":"explain"}],
            "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
            "stream":true,"reasoning_effort":"none"}"#,
    );
    let body = response.streaming_body.take().expect("stream");
    let output = Arc::new((Mutex::new(String::new()), Condvar::new()));
    let writer = {
        let output = output.clone();
        std::thread::spawn(move || {
            body(&mut |chunk: &[u8]| {
                output
                    .0
                    .lock()
                    .expect("output")
                    .push_str(&String::from_utf8_lossy(chunk));
                output.1.notify_all();
                true
            })
            .expect("stream");
        })
    };
    assert!(
        backend.wait_for_first_piece(),
        "backend reaches the comparison prose"
    );
    let arrived = {
        let text = output.0.lock().expect("output");
        let (_text, timeout) = output
            .1
            .wait_timeout_while(text, Duration::from_secs(2), |text| {
                streamed_content(text).0 != "3 < 5 and x < y"
            })
            .expect("output");
        !timeout.timed_out()
    };
    let completed = backend.completed.load(Ordering::SeqCst);
    backend.release();
    writer.join().expect("writer");
    assert!(
        arrived && !completed,
        "all comparison prose streams while generation is still blocked"
    );
    let output = output.0.lock().expect("output").clone();
    assert_eq!(
        streamed_content(&output).0,
        "3 < 5 and x < y are comparisons.",
        "comparison content stays exact"
    );
}

/// With thinking off the output starts as content and reasoning tags the
/// client asked for stay data; an automatic start still recognizes a
/// leading reasoning block, also split across tokens.
#[test]
fn initial_output_phases() {
    for (text, automatic) in [
        ("<think>literal example</think>", false),
        (" \n<think>literal example</think>\nanswer", false),
        ("<think>unfinished example", false),
        ("<think>Check carefully.</think>\nAnswer.", true),
    ] {
        for stream in [false, true] {
            let backend = FakeBackend::new();
            if automatic {
                backend.config().initial = Some(InitialOutputState::Auto);
            }
            let pieces: Vec<String> = text.chars().map(String::from).collect();
            let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
            backend.set_pieces(&pieces);
            let effort = if automatic {
                ""
            } else {
                r#""reasoning_effort":"none","#
            };
            let mut response = chat(
                &backend,
                &format!(
                    r#"{{"model":"test-model",{effort}"stream":{stream},
                       "messages":[{{"role":"user","content":"Repeat it."}}]}}"#
                ),
            );
            assert_eq!(response.status, 200, "{text:?}");
            let (content, reasoning) = if stream {
                let output = drain(&mut response);
                let (mut content, mut reasoning) = (String::new(), String::new());
                for data in output
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter(|data| !data.starts_with("[DONE]"))
                {
                    let event = json::parse_str(data).expect("event");
                    for choice in event.find("choices").map_or(&[][..], |c| c.items()) {
                        if let Some(delta) = choice.find("delta") {
                            content += &delta.member_str("content", "");
                            reasoning += &delta.member_str("reasoning_content", "");
                        }
                    }
                }
                (content, reasoning)
            } else {
                let output = json::parse(&response.body).expect("body");
                let message = output.find("choices").expect("choices").items()[0]
                    .find("message")
                    .expect("message")
                    .clone();
                (
                    message.member_str("content", ""),
                    message.member_str("reasoning_content", ""),
                )
            };
            if automatic {
                assert_eq!(
                    (content.as_str(), reasoning.as_str()),
                    ("Answer.", "Check carefully."),
                    "stream={stream}"
                );
            } else {
                assert_eq!(content, text, "literal tags stay content (stream={stream})");
                assert!(reasoning.is_empty(), "{text:?} stream={stream}");
            }
        }
    }
}

/// Only the separator after `</think>` is framing: Qwen's whitespace before
/// the answer goes, in plain, tool and JSON answers, buffered and streamed,
/// split per byte or not; whitespace inside the answer and an answer without
/// reasoning stay exact.
#[test]
fn reasoning_answer_separator() {
    for thinking in [false, true] {
        for stream in [false, true] {
            for split in [false, true] {
                for mode in 0..3 {
                    for separator in ["", "\n\n", "\r\n\r\n", " \t\n\n\x0c\x0b"] {
                        for truncated in [false, true] {
                            let answer = if truncated {
                                ""
                            } else if mode == 2 {
                                r#"{"answer":9,"text":"\n\nliteral"}"#
                            } else {
                                "9\n\nNext paragraph.\n    Indented line."
                            };
                            let raw = format!(
                                "{}{separator}{answer}",
                                if thinking { "Check</think>" } else { "" }
                            );
                            let backend = FakeBackend::new();
                            if truncated {
                                backend.config().finish_reason = FinishReason::Length;
                            }
                            backend.config().tool_format = Some(ToolFormat::Qwen);
                            if split {
                                let pieces: Vec<String> =
                                    raw.bytes().map(|b| (b as char).to_string()).collect();
                                let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
                                backend.set_pieces(&pieces);
                            } else {
                                backend.set_pieces(&[&raw]);
                            }
                            let mut body = json::parse_str(
                                r#"{"model":"test-model","temperature":0,
                                   "messages":[{"role":"user","content":"4+5?"}]}"#,
                            )
                            .expect("body");
                            body = body
                                .with("stream", stream)
                                .with("reasoning_effort", if thinking { "low" } else { "none" });
                            if mode == 1 {
                                body = body
                                    .with(
                                        "tools",
                                        json::parse_str(
                                            r#"[{"type":"function","function":{
                                               "name":"unused","parameters":{"type":"object",
                                               "properties":{},"additionalProperties":false}}}]"#,
                                        )
                                        .expect("tools"),
                                    )
                                    .with("tool_choice", "auto");
                            } else if mode == 2 {
                                body = body.with(
                                    "response_format",
                                    json::parse_str(
                                        r#"{"type":"json_schema","json_schema":{"name":"answer",
                                           "strict":true,"schema":{"type":"object","properties":{
                                           "answer":{"type":"integer"},"text":{"type":"string"}},
                                           "required":["answer","text"],
                                           "additionalProperties":false}}}"#,
                                    )
                                    .expect("format"),
                                );
                            }
                            let mut response = chat(&backend, &body.dump().expect("dump"));
                            assert_eq!(
                                response.status, 200,
                                "reasoning separator request succeeds"
                            );
                            let content = if stream {
                                let output = drain(&mut response);
                                let mut content = String::new();
                                for data in output
                                    .lines()
                                    .filter_map(|line| line.strip_prefix("data: "))
                                    .filter(|data| !data.starts_with("[DONE]"))
                                {
                                    let event = json::parse_str(data).expect("event");
                                    for choice in
                                        event.find("choices").map_or(&[][..], |c| c.items())
                                    {
                                        if let Some(delta) = choice.find("delta") {
                                            content += &delta.member_str("content", "");
                                        }
                                    }
                                }
                                content
                            } else {
                                let output = json::parse(&response.body).expect("body");
                                output.find("choices").expect("choices").items()[0]
                                    .find("message")
                                    .expect("message")
                                    .member_str("content", "")
                            };
                            let expected = if thinking {
                                answer.to_owned()
                            } else {
                                format!("{separator}{answer}")
                            };
                            assert_eq!(
                                content, expected,
                                "only the reasoning separator is removed; answer whitespace \
                                 survives: thinking={thinking} stream={stream} split={split} \
                                 mode={mode} truncated={truncated} separator={separator:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Constrained reasoning ends only at `</think>`: quoted tool openers in it
/// are reasoning data, and the call after it keeps its exact file data
///.
#[test]
fn tool_markers_inside_constrained_reasoning() {
    let reasoning = "Confirmed garbage on the docstring line. The broken line is:\n\
        `    \"\"\"The one BPA resource; find_resources maps logical id to a \
        Mapping.\"\"\"}}]}}</tool_call>\n\n<tool_call>('tool', '{`\n\
        A quoted <tool_call> is file data, not the end of reasoning.\n\
        Also <｜DSML｜tool_calls> is quoted data. Preserve oldText exactly.";
    let replacement = Value::object()
        .with(
            "oldText",
            "    \"\"\"The one BPA resource; find_resources maps logical id to a \
             Mapping.\"\"\"}}]}}</tool_call>\n\n<tool_call>('tool', '{",
        )
        .with(
            "newText",
            "    \"\"\"The one BPA resource; find_resources maps logical id to a Mapping.\"\"\"",
        );
    let arguments = Value::object()
        .with("path", "tests/unit/shared/test_vpc_block_public_access.py")
        .with("edits", vec![replacement]);
    let call = format!(
        "<tool_call>\n<function=edit>\n<parameter=path>\n{}\n</parameter>\n<parameter=edits>\n{}\n\
         </parameter>\n</function>\n</tool_call>",
        arguments.member_str("path", ""),
        arguments
            .find("edits")
            .expect("edits")
            .dump()
            .expect("dump")
    );
    let prose = "\nVerify the file, then run mypy and tests.";
    // The reasoning separator is framing; newlines inside the edit and the
    // prose following its call remain data in both APIs and transports
    //.
    let complete = format!("</think>\n\n{call}{prose}");
    let expected_arguments = arguments.dump().expect("dump");
    for over_responses in [false, true] {
        for stream in [false, true] {
            for bytewise in [false, true] {
                for strict in [false, true] {
                    for tail in [complete.as_str(), "", "</thi"] {
                        let truncated = tail != complete;
                        let expected_reasoning =
                            format!("{reasoning}{}", if truncated { tail } else { "" });
                        let expected_content = if truncated { "" } else { prose };
                        let mut body = json::parse_str(
                            r#"{"model":"test-model","messages":[{"role":"user","content":"fix the docstring"}],
                            "chat_template_kwargs":{"enable_thinking":true},
                            "parallel_tool_calls":false,"tool_choice":"auto",
                            "tools":[{"type":"function","function":{"name":"edit",
                              "parameters":{"type":"object","properties":{
                                "path":{"type":"string"},"edits":{"type":"array","items":{
                                  "type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},
                                  "required":["oldText","newText"],"additionalProperties":false}}},
                                "required":["path","edits"],"additionalProperties":false}}}]}"#,
                        )
                        .expect("json");
                        body.set("stream", stream);
                        let mut tool = body.find("tools").expect("tools").items()[0].clone();
                        let function = tool.find("function").expect("function").clone();
                        tool.set("function", function.with("strict", strict));
                        body.set("tools", vec![tool]);
                        let backend = FakeBackend::new();
                        let raw = format!("{reasoning}{tail}");
                        {
                            let mut config = backend.config();
                            config.finish_reason = if truncated {
                                FinishReason::Length
                            } else {
                                FinishReason::Stop
                            };
                            config.pieces = if bytewise {
                                raw.bytes().map(|byte| vec![byte]).collect()
                            } else {
                                vec![raw.into_bytes()]
                            };
                        }
                        let mut response = if over_responses {
                            responses(&backend, &body, true, 4096, stream)
                        } else {
                            chat(&backend, &body.dump().expect("dump"))
                        };
                        assert_eq!(response.status, 200, "reasoning marker fixture succeeds");
                        let (mut content, mut thought, mut emitted_arguments) =
                            (String::new(), String::new(), String::new());
                        let (mut streamed_content, mut streamed_thought) =
                            (String::new(), String::new());
                        let mut calls = 0;
                        let mut terminal = false;
                        for event in events(&mut response, stream) {
                            if over_responses {
                                match event.member_str("type", "").as_str() {
                                    "response.output_text.delta" => {
                                        streamed_content += &event.member_str("delta", "");
                                    }
                                    "response.reasoning_summary_text.delta" => {
                                        streamed_thought += &event.member_str("delta", "");
                                    }
                                    _ => {}
                                }
                                // Terminal items are authoritative in both modes.
                                let result = if stream {
                                    event.find("response")
                                } else {
                                    Some(&event)
                                };
                                let status = if truncated { "incomplete" } else { "completed" };
                                let Some(result) =
                                    result.filter(|r| r.member_str("status", "") == status)
                                else {
                                    continue;
                                };
                                terminal = true;
                                for item in result.find("output").expect("output").items() {
                                    match item.member_str("type", "").as_str() {
                                        "function_call" => {
                                            calls += 1;
                                            emitted_arguments = item.member_str("arguments", "");
                                        }
                                        kind @ ("reasoning" | "message") => {
                                            // Reasoning in upstream's summary.
                                            let parts = if kind == "message" {
                                                "content"
                                            } else {
                                                "summary"
                                            };
                                            for part in
                                                item.find(parts).map_or(&[][..], |c| c.items())
                                            {
                                                let target = if kind == "message" {
                                                    &mut content
                                                } else {
                                                    &mut thought
                                                };
                                                *target += &part.member_str("text", "");
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                                continue;
                            }
                            for choice in event.find("choices").map_or(&[][..], |c| c.items()) {
                                let finish = choice.member_str("finish_reason", "");
                                if !finish.is_empty() {
                                    terminal = true;
                                    assert_eq!(
                                        finish,
                                        if truncated { "length" } else { "tool_calls" },
                                        "reasoning limits keep the correct finish reason"
                                    );
                                }
                                let Some(message) =
                                    choice.find(if stream { "delta" } else { "message" })
                                else {
                                    continue;
                                };
                                thought += &message.member_str("reasoning_content", "");
                                content += &message.member_str("content", "");
                                for tool in
                                    message.find("tool_calls").map_or(&[][..], |t| t.items())
                                {
                                    calls += 1;
                                    emitted_arguments += &tool
                                        .find("function")
                                        .expect("function")
                                        .member_str("arguments", "");
                                }
                            }
                        }
                        let label = format!(
                            "responses={over_responses} stream={stream} bytewise={bytewise} \
                             strict={strict} tail={tail:?}"
                        );
                        assert!(terminal, "a terminal result: {label}");
                        assert_eq!(
                            thought, expected_reasoning,
                            "literal tool openers cannot end constrained reasoning: {label}"
                        );
                        assert_eq!(
                            content, expected_content,
                            "reasoning text and closing think tags never leak as content: {label}"
                        );
                        if truncated {
                            assert!(calls == 0 && emitted_arguments.is_empty(), "{label}");
                        } else {
                            assert!(
                                calls == 1 && emitted_arguments == expected_arguments,
                                "the edit keeps exact file data: {label}: {emitted_arguments}"
                            );
                        }
                        if over_responses && stream {
                            assert!(
                                streamed_content == expected_content
                                    && streamed_thought == expected_reasoning,
                                "Responses deltas agree with the buffered phases: {label}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// A native call header outside quotations ends constrained reasoning
/// without `</think>`, as in llama.cpp's native parsers; a quoted header
/// stays reasoning, and only a complete call is invoked.
#[test]
fn native_tool_implicit_reasoning_end() {
    for format in [ToolFormat::Qwen, ToolFormat::DeepSeek] {
        let qwen = format == ToolFormat::Qwen;
        let call = if qwen {
            "<tool_call>\n<function=read>\n<parameter=path>\nfile.txt\n\
             </parameter>\n</function>\n</tool_call>"
        } else {
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n\
             <｜DSML｜parameter name=\"path\" string=\"true\">file.txt\
             </｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        };
        for over_responses in [false, true] {
            for stream in [false, true] {
                for quoted in [false, true] {
                    for interrupted in [false, true] {
                        let thought = if quoted {
                            format!("Example:\n```xml\n{call}\n```\nRead the file.\n\n")
                        } else {
                            "Read the file.\n\n".to_owned()
                        };
                        let emitted = if interrupted {
                            &call[..call.find("file.txt").expect("path") + 4]
                        } else {
                            call
                        };
                        let text =
                            format!("{thought}{}{emitted}", if quoted { "</think>" } else { "" });
                        // llama.cpp treats the two newlines before a DeepSeek
                        // call as TC_SEPARATOR, including when the model
                        // omitted </think>.
                        let expected_thought = if qwen || quoted {
                            thought.as_str()
                        } else {
                            &thought[..thought.len() - 2]
                        };
                        // Responses reasoning is trimmed like the chat
                        // completion's (docs/RUST.md, differences).
                        let expected_thought = if over_responses {
                            expected_thought.trim()
                        } else {
                            expected_thought
                        };
                        let mut body = json::parse_str(
                            r#"{"model":"test-model","messages":[{"role":"user","content":"read the file"}],
                            "reasoning_effort":"low","tool_choice":"required","parallel_tool_calls":false,
                            "tools":[{"type":"function","function":{"name":"read","parameters":{
                              "type":"object","properties":{"path":{"type":"string"}},
                              "required":["path"],"additionalProperties":false}}}]}"#,
                        )
                        .expect("json");
                        body.set("stream", stream);
                        let backend = FakeBackend::new();
                        {
                            let mut config = backend.config();
                            config.tool_format = Some(format);
                            config.finish_reason = if interrupted {
                                FinishReason::Length
                            } else {
                                FinishReason::Stop
                            };
                            config.pieces = text.bytes().map(|byte| vec![byte]).collect();
                        }
                        let mut response = if over_responses {
                            responses(&backend, &body, true, 512, stream)
                        } else {
                            chat(&backend, &body.dump().expect("dump"))
                        };
                        assert_eq!(
                            response.status, 200,
                            "implicit native tool request succeeds"
                        );
                        let mut result = None;
                        let (mut streamed_reasoning, mut streamed_content) =
                            (String::new(), String::new());
                        let (mut streamed_arguments, mut finish) = (String::new(), String::new());
                        for event in events(&mut response, stream) {
                            if !stream {
                                result = Some(event);
                            } else if over_responses {
                                match event.member_str("type", "").as_str() {
                                    "response.reasoning_summary_text.delta" => {
                                        streamed_reasoning += &event.member_str("delta", "");
                                    }
                                    "response.output_text.delta" => {
                                        streamed_content += &event.member_str("delta", "");
                                    }
                                    "response.completed" | "response.incomplete" => {
                                        result = event.find("response").cloned();
                                    }
                                    _ => {}
                                }
                            } else {
                                for choice in event.find("choices").map_or(&[][..], |c| c.items()) {
                                    let reason = choice.member_str("finish_reason", "");
                                    if !reason.is_empty() {
                                        finish = reason;
                                    }
                                    if let Some(delta) = choice.find("delta") {
                                        streamed_reasoning +=
                                            &delta.member_str("reasoning_content", "");
                                        streamed_content += &delta.member_str("content", "");
                                        for tool in
                                            delta.find("tool_calls").map_or(&[][..], |t| t.items())
                                        {
                                            streamed_arguments += &tool
                                                .find("function")
                                                .expect("function")
                                                .member_str("arguments", "");
                                        }
                                    }
                                }
                            }
                        }
                        let (mut reasoning, mut content, mut calls) =
                            (String::new(), String::new(), 0);
                        if over_responses {
                            let result = result.expect("terminal response");
                            for item in result.find("output").expect("output").items() {
                                let kind = item.member_str("type", "");
                                if kind == "function_call" {
                                    calls +=
                                        usize::from(item.member_str("status", "") == "completed");
                                    streamed_arguments = item.member_str("arguments", "");
                                }
                                // Reasoning items carry upstream's summary
                                // parts by default, messages content parts.
                                let parts = if kind == "reasoning" {
                                    "summary"
                                } else {
                                    "content"
                                };
                                for part in item.find(parts).map_or(&[][..], |c| c.items()) {
                                    let target = if kind == "reasoning" {
                                        &mut reasoning
                                    } else {
                                        &mut content
                                    };
                                    *target += &part.member_str("text", "");
                                }
                            }
                        } else if !stream {
                            let result = result.expect("body");
                            let message = result.find("choices").expect("choices").items()[0]
                                .find("message")
                                .expect("message");
                            reasoning = message.member_str("reasoning_content", "");
                            content = message.member_str("content", "");
                            calls = message
                                .find("tool_calls")
                                .map_or(0, |tools| tools.items().len());
                        } else {
                            reasoning = streamed_reasoning.clone();
                            content = streamed_content.clone();
                            calls = usize::from(finish == "tool_calls");
                        }
                        let label = format!(
                            "{format:?} responses={over_responses} stream={stream} \
                             quoted={quoted} interrupted={interrupted}"
                        );
                        assert_eq!(
                            calls,
                            usize::from(!interrupted),
                            "only a complete unquoted native call is invoked: {label}"
                        );
                        if stream && interrupted {
                            assert_eq!(
                                streamed_arguments, r#"{"path":"file"#,
                                "an interrupted call retains only its unfinished prefix: {label}"
                            );
                        }
                        assert!(
                            content.is_empty() && reasoning == expected_thought,
                            "implicit call markup is not leaked into either output channel: \
                             {label}: reasoning={reasoning:?} content={content:?}"
                        );
                        if stream {
                            assert!(
                                streamed_content.is_empty()
                                    && streamed_reasoning == expected_thought,
                                "implicit reasoning boundary is correct during streaming: \
                                 {label}: reasoning={streamed_reasoning:?} \
                                 content={streamed_content:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Framing goes only with request context: closers echoed directly after a
/// call and client envelopes naming a declared tool; other XML, quoted
/// documentation and DeepSeek content outside its call block stay content
///.
#[test]
fn tool_closing_framing() {
    let call = "<tool_call><function=f><parameter=text>\n42\n</parameter></function></tool_call>";
    let inline_call =
        "<tool_call><function=f><parameter=text>42</parameter></function></tool_call>";
    let envelope = r#"<invoke name="f"><parameter name="text">x</parameter></invoke>"#;
    let raw_xml = r#"<invoke name="documentation"><parameter name="text">x</parameter></invoke>"#;
    let arg = r#"{"text":"42"}"#;
    let case = |text: String, calls: usize, argument: &str, content: String| {
        (text, calls, argument.to_owned(), content)
    };
    let qwen_cases = vec![
        case(format!("I'll update `config.py\n{call}"), 1, arg, "I'll update `config.py\n".into()),
        case(format!("Let`s write it.\n{call}"), 1, arg, "Let`s write it.\n".into()),
        case(
            format!("looks like `{call}` and no call."),
            0,
            "",
            format!("looks like `{call}` and no call."),
        ),
        case(
            format!("looks like ``{call}`` and no call."),
            0,
            "",
            format!("looks like ``{call}`` and no call."),
        ),
        case(format!("looks like `{call}"), 1, arg, "looks like `".into()),
        case(format!("looks like ``{call}"), 1, arg, "looks like ``".into()),
        case(format!("`example\n\n{call}"), 1, arg, "`example\n\n".into()),
        case(
            format!("looks like `literal <think>\n{call}`"),
            0,
            "",
            format!("looks like `literal <think>\n{call}`"),
        ),
        case(format!("{call}\n</invoke>\n</parameter>\n</function>\n"), 1, arg, String::new()),
        case(format!("{call}\n</function>\n{call}"), 2, arg, String::new()),
        case(format!("{call}\n{envelope}"), 1, arg, String::new()),
        case(format!("{call}\n</invoke>\n<|im_end|>"), 1, arg, "<|im_end|>".into()),
        case(format!("</invoke>\n{call}"), 1, arg, "</invoke>\n".into()),
        case("</invoke>".into(), 0, "", "</invoke>".into()),
        case("Text </parameter>".into(), 0, "", "Text </parameter>".into()),
        case("The token is <|im_end|>".into(), 0, "", "The token is <|im_end|>".into()),
        case("<parameter>x</parameter>".into(), 0, "", "<parameter>x</parameter>".into()),
        case(raw_xml.into(), 0, "", raw_xml.into()),
        case(format!("Raw XML: {raw_xml}"), 0, "", format!("Raw XML: {raw_xml}")),
        case(envelope.into(), 0, "", String::new()),
        case(format!("Planning.\n{envelope}"), 0, "", "Planning.".into()),
        case(r#"<invoke name="f"><parameter name="text">x"#.into(), 0, "", String::new()),
        case(
            "Planning.\n<parameter=text>\nx\n</parameter>\n</function>\n</tool_call>".into(),
            0,
            "",
            "Planning.\n<parameter=text>\nx\n</parameter>\n</function>\n</tool_call>".into(),
        ),
        case(
            "<function=f><parameter=text>x</parameter></function>".into(),
            0,
            "",
            "<function=f><parameter=text>x</parameter></function>".into(),
        ),
        case(format!("```xml\n{envelope}\n```"), 0, "", format!("```xml\n{envelope}\n```")),
        case(format!("`{envelope}` and prose"), 0, "", format!("`{envelope}` and prose")),
        case(format!("```python\nprint(1)\n{call}"), 1, arg, "```python\nprint(1)\n".into()),
        case(format!("```xml\n{call}\n```"), 0, "", format!("```xml\n{call}\n```")),
        case(format!("```xml\n{call}\n```\n{call}"), 1, arg, format!("```xml\n{call}\n```\n")),
        case(format!("`{inline_call}`"), 0, "", format!("`{inline_call}`")),
        case(format!("``{inline_call}``"), 0, "", format!("``{inline_call}``")),
        case(
            "```xml\n<tool_call><function=unknown></function></tool_call>".into(),
            0,
            "",
            "```xml\n<tool_call><function=unknown></function></tool_call>".into(),
        ),
        case(
            "```xml\n<tool_call><function=f></function></tool_call>".into(),
            0,
            "",
            "```xml\n<tool_call><function=f></function></tool_call>".into(),
        ),
        case(
            "```xml\n<tool_call><function=f><parameter=extra>x</parameter></function></tool_call>"
                .into(),
            0,
            "",
            "```xml\n<tool_call><function=f><parameter=extra>x</parameter></function></tool_call>"
                .into(),
        ),
        case(
            "<tool_call><function=f><parameter=text>\nEOS = \"<|im_end|>\"\n</parameter></function></tool_call>"
                .into(),
            1,
            r#"{"text":"EOS = \"<|im_end|>\""}"#,
            String::new(),
        ),
        case(
            "<tool_call><function=f><parameter=text>\n<|im_start|><|endoftext|><|not_a_vocab_entry|>\n\
             </parameter></function></tool_call>"
                .into(),
            1,
            r#"{"text":"<|im_start|><|endoftext|><|not_a_vocab_entry|>"}"#,
            String::new(),
        ),
        case(
            "<tool_call><function=f><parameter=text>\n</invoke>\n</parameter></function></tool_call>"
                .into(),
            1,
            r#"{"text":"</invoke>"}"#,
            String::new(),
        ),
    ];
    // As in llama.cpp, DeepSeek output is content outside its native block,
    // which ends the output: client envelopes and closers are never removed.
    let dsml = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n\
        <｜DSML｜parameter name=\"text\" string=\"true\">42</｜DSML｜parameter>\n\
        </｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    let deepseek_cases = vec![
        case(envelope.into(), 0, "", envelope.into()),
        case(
            format!("Planning.\n{envelope}"),
            0,
            "",
            format!("Planning.\n{envelope}"),
        ),
        case(
            r#"<invoke name="f"><parameter name="text">x"#.into(),
            0,
            "",
            r#"<invoke name="f"><parameter name="text">x"#.into(),
        ),
        case(
            "</invoke>\n</parameter>".into(),
            0,
            "",
            "</invoke>\n</parameter>".into(),
        ),
        case(call.into(), 0, "", call.into()),
        case(format!("</invoke>{dsml}"), 1, arg, "</invoke>".into()),
        case(
            format!("Planning.\n{envelope}{dsml}"),
            1,
            arg,
            format!("Planning.\n{envelope}"),
        ),
    ];
    let mut body = json::parse_str(
        r#"{"model":"test-model","messages":[{"role":"user","content":"use f"}],
        "tools":[{"type":"function","function":{"name":"f","parameters":{
          "type":"object","properties":{"text":{"type":"string"}},
          "required":["text"],"additionalProperties":false}}}],"tool_choice":"auto"}"#,
    )
    .expect("json");
    for (format, cases) in [
        (ToolFormat::Qwen, &qwen_cases),
        (ToolFormat::DeepSeek, &deepseek_cases),
    ] {
        for (text, expected_calls, argument, expected_content) in cases {
            for stream in [false, true] {
                for bytewise in [false, true] {
                    body.set("stream", stream);
                    let backend = FakeBackend::new();
                    {
                        let mut config = backend.config();
                        config.tool_format = Some(format);
                        config.pieces = if bytewise {
                            text.bytes().map(|byte| vec![byte]).collect()
                        } else {
                            vec![text.clone().into_bytes()]
                        };
                    }
                    let mut response = chat(&backend, &body.dump().expect("dump"));
                    assert_eq!(response.status, 200, "framing request succeeds");
                    let (mut calls, mut content, mut finish) =
                        (Vec::new(), String::new(), String::new());
                    for event in events(&mut response, stream) {
                        for choice in event.find("choices").map_or(&[][..], |c| c.items()) {
                            finish += &choice.member_str("finish_reason", "");
                            if let Some(message) =
                                choice.find(if stream { "delta" } else { "message" })
                            {
                                content += &message.member_str("content", "");
                                if let Some(found) = message.find("tool_calls") {
                                    append_tool_deltas(&mut calls, found, stream);
                                }
                            }
                        }
                    }
                    let label = format!("{format:?} {text:?} stream={stream} bytewise={bytewise}");
                    assert_eq!(
                        calls.len(),
                        *expected_calls,
                        "only complete declared calls: {label}"
                    );
                    assert_eq!(
                        &content, expected_content,
                        "framing is told from content by request context: {label}"
                    );
                    assert_eq!(
                        finish,
                        if *expected_calls > 0 {
                            "tool_calls"
                        } else {
                            "stop"
                        },
                        "finish reason agrees with the calls: {label}"
                    );
                    if let Some(last) = calls.last() {
                        assert_eq!(
                            &last
                                .find("function")
                                .expect("function")
                                .member_str("arguments", ""),
                            argument,
                            "literal arguments survive exactly: {label}"
                        );
                    }
                }
            }
        }
    }
}

/// One output of [`native_tool_text_outside_envelopes`].
struct EnvelopeFixture {
    format: ToolFormat,
    raw: String,
    text: String,
    paths: Vec<String>,
    complete: bool,
    /// Spelled in the native grammar's canonical form.
    canonical: bool,
    finish: FinishReason,
    required: bool,
    admitted_format: Option<ToolFormat>,
}

impl EnvelopeFixture {
    fn new(format: ToolFormat, raw: String, text: String, paths: &[&str]) -> Self {
        Self {
            format,
            raw,
            text,
            paths: paths.iter().map(|path| (*path).to_owned()).collect(),
            complete: true,
            canonical: true,
            finish: FinishReason::Stop,
            required: false,
            admitted_format: None,
        }
    }
}

/// Text outside complete call blocks is content; calls inside them keep
/// their literal arguments; a native grammar owns only its canonical
/// envelope, and its calls end the output.
#[test]
fn native_tool_text_outside_envelopes() {
    use ToolFormat::{DeepSeek, Json, Qwen};
    let schema = json::parse_str(
        r#"{"type":"object","properties":{"path":{"type":"string"}},
        "required":["path"],"additionalProperties":false}"#,
    )
    .expect("json");
    let invoke = |path: &str| {
        format!(
            "\n<｜DSML｜invoke name=\"read\">\n<｜DSML｜parameter name=\"path\" string=\"true\">{path}\
             </｜DSML｜parameter>\n</｜DSML｜invoke>"
        )
    };
    let envelope = |calls: &str| format!("<｜DSML｜tool_calls>{calls}\n</｜DSML｜tool_calls>");
    let suffix = "\nNo edits are requested.";
    let first = envelope(&invoke("fixture.txt"));
    let legacy = format!("<tool_calls>{}\n</tool_calls>", invoke("example.txt"));
    let literal = "Keep </｜DSML｜tool_calls> as argument data.";
    let fixture = EnvelopeFixture::new;
    let fixtures = vec![
        fixture(
            Qwen,
            format!(
                "<tool_call>\n<function=read>\n<parameter=path>\nfixture.txt\n</parameter>\n\
                 </function>\n</tool_call>{suffix}"
            ),
            suffix.into(),
            &["fixture.txt"],
        ),
        fixture(
            Json,
            format!(
                r#"<tool_call>{{"name":"read","arguments":{{"path":"fixture.txt"}}}}</tool_call>{suffix}"#
            ),
            suffix.into(),
            &["fixture.txt"],
        ),
        fixture(
            DeepSeek,
            format!("{first}{suffix}"),
            suffix.into(),
            &["fixture.txt"],
        ),
        // Native framing may be part of the admitted opener.
        EnvelopeFixture {
            admitted_format: Some(DeepSeek),
            ..fixture(
                DeepSeek,
                format!("\n\n{first}{suffix}"),
                suffix.into(),
                &["fixture.txt"],
            )
        },
        fixture(
            DeepSeek,
            format!(
                "Before.\n{}{suffix}",
                envelope(&(invoke("fixture.txt") + &invoke("second.txt")))
            ),
            format!("Before.\n{suffix}"),
            &["fixture.txt", "second.txt"],
        ),
        fixture(
            DeepSeek,
            format!(
                "{first}\nBetween.\n{}{suffix}",
                envelope(&invoke("second.txt"))
            ),
            format!("\nBetween.\n{suffix}"),
            &["fixture.txt", "second.txt"],
        ),
        fixture(
            DeepSeek,
            format!("{}{suffix}", envelope(&invoke(literal))),
            suffix.into(),
            &[literal],
        ),
        // Interrupted output may keep a complete invocation before its outer
        // list closes, but exposes no unfinished markup.
        EnvelopeFixture {
            complete: false,
            ..fixture(
                DeepSeek,
                format!(
                    "{first}\nBetween.\n<｜DSML｜tool_calls>{}",
                    invoke("second.txt")
                ),
                "\nBetween.\n".into(),
                &["fixture.txt", "second.txt"],
            )
        },
        EnvelopeFixture {
            complete: false,
            ..fixture(
                DeepSeek,
                format!(
                    "{first}\nBetween.\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n\
                     <｜DSML｜parameter name=\"path\" string=\"true\">partial"
                ),
                "\nBetween.\n".into(),
                &["fixture.txt"],
            )
        },
        // A compatibility spelling is accepted by extraction, not generated by
        // the canonical native grammar.
        EnvelopeFixture {
            canonical: false,
            ..fixture(
                DeepSeek,
                format!(
                    "<｜DSML｜tool_calls｜>{}\n</｜DSML｜tool_calls｜>{suffix}",
                    invoke("fixture.txt")
                ),
                suffix.into(),
                &["fixture.txt"],
            )
        },
        // A partial marker is legal text in auto mode; required tools hold it
        // back when a length limit or an explicit stop interrupts.
        EnvelopeFixture {
            finish: FinishReason::Length,
            ..fixture(
                DeepSeek,
                format!("{first}\nBetween.\n<｜DSML｜tool_calls"),
                "\nBetween.\n<｜DSML｜tool_calls".into(),
                &["fixture.txt"],
            )
        },
        EnvelopeFixture {
            finish: FinishReason::Length,
            required: true,
            ..fixture(
                DeepSeek,
                format!("{first}\nBetween.\n<｜DSML｜tool_calls"),
                "\nBetween.\n".into(),
                &["fixture.txt"],
            )
        },
        EnvelopeFixture {
            finish: FinishReason::StopSequence,
            required: true,
            ..fixture(
                DeepSeek,
                format!("{first}\nBetween.\n<｜DSML｜tool_calls"),
                "\nBetween.\n".into(),
                &["fixture.txt"],
            )
        },
        // Only an outer envelope starts DSML calls: bare invocation examples
        // outside it are text, even before another real envelope.
        fixture(
            DeepSeek,
            format!("{first}{}{suffix}", invoke("example.txt")),
            format!("{}{suffix}", invoke("example.txt")),
            &["fixture.txt"],
        ),
        fixture(
            DeepSeek,
            format!(
                "{first}{}{}{suffix}",
                invoke("example.txt"),
                envelope(&invoke("second.txt"))
            ),
            format!("{}{suffix}", invoke("example.txt")),
            &["fixture.txt", "second.txt"],
        ),
        EnvelopeFixture {
            admitted_format: Some(DeepSeek),
            ..fixture(
                DeepSeek,
                format!("{first}{suffix}"),
                suffix.into(),
                &["fixture.txt"],
            )
        },
        // A known native grammar owns only its canonical envelope: foreign
        // wrapper examples stay prose, native-looking invocations included.
        EnvelopeFixture {
            admitted_format: Some(DeepSeek),
            ..fixture(
                DeepSeek,
                format!("{first}\n<tool_calls></tool_calls>{suffix}"),
                format!("\n<tool_calls></tool_calls>{suffix}"),
                &["fixture.txt"],
            )
        },
        EnvelopeFixture {
            admitted_format: Some(DeepSeek),
            ..fixture(
                DeepSeek,
                format!("{first}{legacy}{suffix}"),
                format!("{legacy}{suffix}"),
                &["fixture.txt"],
            )
        },
        // Unknown metadata and the JSON fallback keep every native opener:
        // both wrapped calls belong to the tool output.
        fixture(
            DeepSeek,
            format!("{first}{legacy}{suffix}"),
            suffix.into(),
            &["fixture.txt", "example.txt"],
        ),
        EnvelopeFixture {
            admitted_format: Some(Json),
            ..fixture(
                Json,
                format!("{first}{legacy}{suffix}"),
                suffix.into(),
                &["fixture.txt", "example.txt"],
            )
        },
    ];
    for fixture in &fixtures {
        let mut parameters_schema = schema.clone();
        if fixture.format == Json {
            parameters_schema.set(
                "enum",
                json::parse_str(r#"[{"path":"fixture.txt"},{"path":"example.txt"}]"#)
                    .expect("json"),
            );
            // A native runner keeps DSML for finite objects too; this fixture
            // covers runners whose only call syntax is the JSON envelope.
            assert!(
                JsonConstraint::tool_parameters(&parameters_schema, false, DeepSeek)
                    .expect("compiles")
                    .is_some(),
                "finite object parameters keep native tool framing"
            );
        }
        if fixture.canonical {
            let parameters =
                JsonConstraint::tool_parameters(&parameters_schema, false, fixture.format)
                    .expect("compiles")
                    .expect("native tool schema is supported");
            let grammar = JsonConstraint::with_tools(
                None,
                vec![("read".into(), parameters)],
                fixture.required,
                true,
                fixture.format,
            )
            .expect("grammar")
            .expect("tools");
            // As in llama.cpp, native calls end the output (Qwen may add the
            // whitespace llama.cpp's `space` admits). Text after them is
            // reachable only without that grammar, e.g. the JSON envelope
            // runners use.
            let mut admitted = fixture.raw.as_str();
            if fixture.format == DeepSeek {
                const CLOSE: &str = "\n</｜DSML｜tool_calls>";
                if let Some(close) = admitted.find(CLOSE) {
                    admitted = &admitted[..close + CLOSE.len()];
                }
            } else if fixture.format == Qwen {
                const CLOSE: &str = "</tool_call>";
                if let Some(close) = admitted.find(CLOSE) {
                    let end = close + CLOSE.len();
                    let newlines = admitted[end..].bytes().take_while(|&b| b == b'\n').count();
                    admitted = &admitted[..end + newlines];
                }
            }
            let mut state = grammar.start().expect("start");
            for byte in admitted.bytes() {
                state = grammar.advance(&state, byte).expect("advance");
                assert!(
                    !state.is_empty(),
                    "the grammar admits the fixture: {:?}",
                    fixture.raw
                );
            }
            assert_eq!(
                grammar.complete(&state),
                fixture.complete || admitted.len() < fixture.raw.len(),
                "only a complete fixture may end normally: {:?}",
                fixture.raw
            );
            if admitted.len() < fixture.raw.len() {
                assert!(
                    grammar
                        .advance(&state, fixture.raw.as_bytes()[admitted.len()])
                        .expect("advance")
                        .is_empty(),
                    "native output ends after its calls"
                );
            }
        }
        for over_responses in [false, true] {
            for stream in [false, true] {
                for fragmented in [false, true] {
                    let function = Value::object()
                        .with("name", "read")
                        .with("strict", false)
                        .with("parameters", parameters_schema.clone());
                    let mut body = json::parse_str(
                        r#"{"model":"test-model","reasoning_effort":"none",
                        "messages":[{"role":"user","content":"Read the fixtures."}]}"#,
                    )
                    .expect("json");
                    body.set(
                        "tools",
                        vec![
                            Value::object()
                                .with("type", "function")
                                .with("function", function),
                        ],
                    );
                    body.set("stream", stream);
                    body.set(
                        "tool_choice",
                        if fixture.required { "required" } else { "auto" },
                    );
                    let backend = FakeBackend::new();
                    {
                        let mut config = backend.config();
                        config.tool_format = fixture.admitted_format;
                        config.finish_reason = if fixture.complete {
                            fixture.finish
                        } else {
                            FinishReason::Length
                        };
                        config.pieces = if fragmented {
                            fixture.raw.bytes().map(|byte| vec![byte]).collect()
                        } else {
                            vec![fixture.raw.clone().into_bytes()]
                        };
                    }
                    let mut response = if over_responses {
                        responses(&backend, &body, false, 512, stream)
                    } else {
                        chat(&backend, &body.dump().expect("dump"))
                    };
                    assert_eq!(response.status, 200, "native tool request succeeds");
                    backend.config().tool_format = None;
                    let (mut text, mut deltas, mut paths) =
                        (String::new(), String::new(), Vec::new());
                    let mut chat_calls = Vec::new();
                    let mut call = |function: &Value| {
                        assert_eq!(
                            function.member_str("name", ""),
                            "read",
                            "only the declared call"
                        );
                        let arguments = json::parse_str(&function.member_str("arguments", ""))
                            .expect("arguments");
                        assert_eq!(arguments.len(), 1, "the call has exactly its path argument");
                        paths.push(arguments.member_str("path", ""));
                    };
                    for event in events(&mut response, stream) {
                        if over_responses {
                            let kind = event.member_str("type", "");
                            if kind == "response.output_text.delta" {
                                deltas += &event.member_str("delta", "");
                            }
                            let result = if stream {
                                event.find("response")
                            } else {
                                Some(&event)
                            };
                            let Some(result) = result.filter(|_| {
                                !stream
                                    || kind == "response.completed"
                                    || kind == "response.incomplete"
                            }) else {
                                continue;
                            };
                            for item in result.find("output").expect("output").items() {
                                match item.member_str("type", "").as_str() {
                                    "function_call" => call(item),
                                    "message" => {
                                        for part in item.find("content").expect("content").items() {
                                            text += &part.member_str("text", "");
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            continue;
                        }
                        for choice in event.find("choices").map_or(&[][..], |c| c.items()) {
                            let message = choice
                                .find(if stream { "delta" } else { "message" })
                                .expect("message");
                            text += &message.member_str("content", "");
                            if let Some(found) = message.find("tool_calls") {
                                append_tool_deltas(&mut chat_calls, found, stream);
                            }
                        }
                    }
                    for tool in &chat_calls {
                        call(tool.find("function").expect("function"));
                    }
                    let label = format!(
                        "{:?} {:?} responses={over_responses} stream={stream} fragmented={fragmented}",
                        fixture.format, fixture.raw
                    );
                    assert!(
                        backend.last_request().constrained_tools
                            && backend.completed.load(Ordering::SeqCst),
                        "the tool request is constrained: {label}"
                    );
                    assert_eq!(
                        paths, fixture.paths,
                        "complete calls keep their arguments: {label}"
                    );
                    assert_eq!(
                        text, fixture.text,
                        "text outside complete call blocks survives: {label}"
                    );
                    if over_responses && stream {
                        assert_eq!(
                            deltas, text,
                            "Responses text deltas agree with the output: {label}"
                        );
                    }
                }
            }
        }
    }
}

/// Arguments matched by `patternProperties` keep their JSON types through
/// DeepSeek calls over both APIs; Qwen tags carry no type flag, so only
/// declared parameters are generated there.
#[test]
fn wildcard_tool_types() {
    let schema = json::parse_str(
        r#"{"type":"object","properties":{"value":{"type":"string"}},"required":["value"],
        "patternProperties":{"^x_":{"type":"integer"}}}"#,
    )
    .expect("json");
    let arguments = json::parse_str(
        r#"{"value":"alpha","x_n":1,"x_b":true,"x_z":null,"x_a":[1],"x_o":{"n":1},"x_s":"1"}"#,
    )
    .expect("json");
    let raw_value = |value: &Value| match value {
        Value::String(text) => text.clone(),
        other => other.dump().expect("dump"),
    };
    for format in [ToolFormat::Qwen, ToolFormat::DeepSeek] {
        let parameters = JsonConstraint::tool_parameters(&schema, false, format)
            .expect("compiles")
            .expect("wildcard tools keep native framing");
        let grammar = JsonConstraint::with_tools(
            None,
            vec![("record".into(), parameters)],
            true,
            false,
            format,
        )
        .expect("grammar")
        .expect("tools");
        // Qwen tags carry no type flag, so as in llama.cpp only declared
        // parameters are generated; DeepSeek's flag keeps typed wildcards.
        let expected = if format == ToolFormat::Qwen {
            r#"{"value":"alpha"}"#.to_owned()
        } else {
            arguments.dump().expect("dump")
        };
        let call = if format == ToolFormat::Qwen {
            let mut extras = String::new();
            for (name, value) in arguments.members() {
                if name != "value" {
                    extras += &format!("<parameter={name}>\n{}\n</parameter>\n", raw_value(value));
                }
            }
            let head = "<tool_call>\n<function=record>\n<parameter=value>\nalpha\n</parameter>\n";
            let tail = "</function>\n</tool_call>";
            let mut rejected = grammar.start().expect("start");
            for byte in format!("{head}{extras}{tail}").bytes() {
                rejected = grammar.advance(&rejected, byte).expect("advance");
            }
            assert!(
                rejected.is_empty() || !grammar.complete(&rejected),
                "undeclared Qwen wildcards are not generated"
            );
            format!("{head}{tail}")
        } else {
            let mut call = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n".to_owned();
            for (name, value) in arguments.members() {
                call += &format!(
                    "<｜DSML｜parameter name=\"{name}\" string=\"{}\">{}</｜DSML｜parameter>\n",
                    value.is_string(),
                    raw_value(value)
                );
            }
            call + "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        };
        let mut state = grammar.start().expect("start");
        for byte in call.bytes() {
            state = grammar.advance(&state, byte).expect("advance");
        }
        assert!(
            grammar.complete(&state),
            "the round-trip call is admissible: {format:?}"
        );
        for over_responses in [false, true] {
            let mut body = json::parse_str(
                r#"{"model":"test-model","messages":[{"role":"user","content":"call record"}],
                "tool_choice":"auto"}"#,
            )
            .expect("json");
            let function = Value::object()
                .with("name", "record")
                .with("strict", false)
                .with("parameters", schema.clone());
            body.set(
                "tools",
                vec![
                    Value::object()
                        .with("type", "function")
                        .with("function", function),
                ],
            );
            let backend = FakeBackend::new();
            backend.set_pieces(&[&call]);
            let response = if over_responses {
                responses(&backend, &body, false, 256, false)
            } else {
                chat(&backend, &body.dump().expect("dump"))
            };
            assert_eq!(response.status, 200, "wildcard call succeeds");
            let output = json::parse(&response.body).expect("json");
            let encoded = if over_responses {
                output
                    .find("output")
                    .expect("output")
                    .items()
                    .last()
                    .expect("item")
                    .member_str("arguments", "")
            } else {
                output.find("choices").expect("choices").items()[0]
                    .find("message")
                    .and_then(|message| message.find("tool_calls"))
                    .expect("tool_calls")
                    .items()[0]
                    .find("function")
                    .expect("function")
                    .member_str("arguments", "")
            };
            assert_eq!(
                encoded, expected,
                "wildcard JSON types survive: {format:?} responses={over_responses}"
            );
        }
    }
}

/// A text-only model refuses images before reading them: an unresolvable
/// HTTPS host would fail differently if it were fetched, and a data URL
/// differently if it were decoded.
#[test]
fn text_only_backend_refuses_images_before_reading() {
    let backend = FakeBackend::new();
    backend.config().image_support = false;
    for url in [
        "https://image-fetch-must-not-happen.invalid/x.png",
        "data:image/png;base64,!!!not-base64!!!",
    ] {
        let body = format!(
            r#"{{"model":"test-model","messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"{url}"}}}}]}}]}}"#
        );
        let response = chat(&backend, &body);
        assert!(
            response.status == 400 && body_text(&response).contains("does not support image input"),
            "a text-only model refuses images before reading them: {url}"
        );
    }
}

/// Validation errors describe the accepted values and name the field that
/// carried the bad one.
#[test]
fn validation_errors_name_the_field() {
    let base = r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#;
    for (field, value, message) in [
        ("model", "5", "'model' must be a string"),
        (
            "reasoning_effort",
            r#""bogus""#,
            "reasoning_effort must be none, minimal",
        ),
        ("max_tokens", "0", "'max_tokens' must be a positive integer"),
        (
            "max_completion_tokens",
            "0",
            "'max_completion_tokens' must be a positive integer",
        ),
    ] {
        let backend = FakeBackend::new();
        let mut invalid = json::parse_str(base).expect("json");
        invalid.set(field, json::parse_str(value).expect("value"));
        let response = chat(&backend, &invalid.dump().expect("dump"));
        assert!(
            response.status == 400 && body_text(&response).contains(message),
            "{field}: {}",
            body_text(&response)
        );
    }
    // As in llama-server, a null cache_prompt keeps the default.
    let backend = FakeBackend::new();
    let mut unset = json::parse_str(base).expect("json");
    unset.set("cache_prompt", Value::Null);
    assert_eq!(chat(&backend, &unset.dump().expect("dump")).status, 200);
    assert!(backend.last_request().cache_prompt);
}

/// What a tool boundary fixture produced through one API (`ToolBoundaryOutput`).
#[derive(Debug, Default)]
struct ToolBoundaryOutput {
    content: String,
    reasoning: String,
    finish: String,
    arguments: Vec<String>,
    names: Vec<String>,
    failed: bool,
}

/// `pieces` through Chat (`api` 0), Responses (1) or Messages (2) with the
/// `record` and `noop` tools (`ToolBoundaryRequest`). Streamed events are
/// checked for ownership too: a delta never precedes its header or follows
/// its block's close.
fn tool_boundary_request(
    pieces: &[Vec<u8>],
    format: ToolFormat,
    api: u8,
    stream: bool,
    initial: InitialOutputState,
    finish: FinishReason,
) -> ToolBoundaryOutput {
    let backend = FakeBackend::new();
    {
        let mut config = backend.config();
        config.pieces = pieces.to_vec();
        config.tool_format = Some(format);
        config.initial = Some(initial);
        config.finish_reason = finish;
        config.stop_sequence = "HALT".into();
    }
    let mut body = json::parse_str(
        r#"{"model":"test-model","messages":[{"role":"user","content":"test framing"}],
        "tool_choice":"auto","tools":[
          {"type":"function","function":{"name":"record","parameters":{
            "type":"object","properties":{"text":{"type":"string"},
              "data":{"type":"object"}},"additionalProperties":false}}},
          {"type":"function","function":{"name":"noop","parameters":{
            "type":"object","properties":{},"additionalProperties":false}}}]}"#,
    )
    .expect("json");
    body.set("stream", stream);
    body.set(
        "reasoning_effort",
        if initial == InitialOutputState::Content {
            "none"
        } else {
            "low"
        },
    );
    let mut response = match api {
        0 => chat(&backend, &body.dump().expect("dump")),
        1 => responses(&backend, &body, false, 4096, stream),
        _ => {
            let mut flat = body.clone();
            let tools: Vec<Value> = body
                .find("tools")
                .expect("tools")
                .items()
                .iter()
                .map(|tool| {
                    tool.find("function")
                        .expect("function")
                        .clone()
                        .with("type", "function")
                })
                .collect();
            flat.set("tools", tools);
            let mut chat = ChatRequest::default();
            parse_tool_controls(&flat, &mut chat, true).expect("tool controls");
            let dynamic: Arc<dyn TextGenerationBackend> = backend.clone();
            create_anthropic_message(
                &request("{}"),
                &dynamic,
                &chat,
                4096,
                &SamplingConfig::default(),
                stream,
            )
            .expect("message")
        }
    };
    assert_eq!(response.status, 200, "tag boundary request is accepted");
    let mut out = ToolBoundaryOutput::default();
    let mut chat_calls = Vec::new();
    let mut open = std::collections::BTreeMap::<usize, bool>::new();
    let mut tools = std::collections::BTreeMap::<usize, usize>::new();
    let mut terminal = false;
    for event in events(&mut response, stream) {
        let kind = event.member_str("type", "");
        if event.find("error").is_some_and(|e| !e.is_null()) || kind == "response.failed" {
            out.failed = true;
            continue;
        }
        if api == 0 {
            for choice in event.find("choices").map_or(&[][..], |c| c.items()) {
                let message = choice
                    .find(if stream { "delta" } else { "message" })
                    .expect("message");
                out.content += &message.member_str("content", "");
                out.reasoning += &message.member_str("reasoning_content", "");
                if let Some(calls) = message.find("tool_calls") {
                    assert!(!terminal, "tool deltas precede the terminal event");
                    append_tool_deltas(&mut chat_calls, calls, stream);
                }
                let reason = choice.member_str("finish_reason", "");
                if !reason.is_empty() {
                    assert!(!terminal, "one terminal choice");
                    terminal = true;
                    out.finish = reason;
                }
            }
            continue;
        }
        if !stream {
            for value in event
                .find(if api == 1 { "output" } else { "content" })
                .expect("output")
                .items()
            {
                match value.member_str("type", "").as_str() {
                    "function_call" => {
                        out.names.push(value.member_str("name", ""));
                        out.arguments.push(value.member_str("arguments", ""));
                    }
                    "tool_use" => {
                        out.names.push(value.member_str("name", ""));
                        out.arguments
                            .push(value.find("input").expect("input").dump().expect("dump"));
                    }
                    "thinking" => out.reasoning += &value.member_str("thinking", ""),
                    "text" => out.content += &value.member_str("text", ""),
                    // Reasoning items carry upstream's summary parts by
                    // default, messages content parts.
                    kind => {
                        let (target, parts) = if kind == "reasoning" {
                            (&mut out.reasoning, "summary")
                        } else {
                            (&mut out.content, "content")
                        };
                        for part in value.find(parts).map_or(&[][..], |c| c.items()) {
                            *target += &part.member_str("text", "");
                        }
                    }
                }
            }
            out.finish = event.member_str(if api == 1 { "status" } else { "stop_reason" }, "");
            terminal = true;
            continue;
        }
        let index = event.member_size(if api == 1 { "output_index" } else { "index" }, 0);
        match kind.as_str() {
            "response.output_item.added" | "content_block_start" => {
                assert!(
                    !terminal && !open.contains_key(&index),
                    "each output block starts exactly once: {event:?}"
                );
                open.insert(index, true);
                let value = event
                    .find(if api == 1 { "item" } else { "content_block" })
                    .expect("block");
                if matches!(
                    value.member_str("type", "").as_str(),
                    "function_call" | "tool_use"
                ) {
                    assert!(!tools.contains_key(&index), "tool header occurs once");
                    let id = value.member_str("call_id", &value.member_str("id", ""));
                    assert!(!id.is_empty(), "tool header has an ID");
                    tools.insert(index, out.arguments.len());
                    out.arguments.push(String::new());
                    out.names.push(value.member_str("name", ""));
                }
            }
            "response.output_item.done" | "content_block_stop" => {
                assert!(
                    open.get(&index) == Some(&true),
                    "each output block closes exactly once"
                );
                if api == 1
                    && let Some(&tool) = tools.get(&index)
                {
                    assert_eq!(
                        event
                            .find("item")
                            .expect("item")
                            .member_str("arguments", ""),
                        out.arguments[tool],
                        "done tool item equals its accumulated deltas"
                    );
                }
                open.insert(index, false);
            }
            "response.function_call_arguments.delta" | "content_block_delta" => {
                assert!(
                    !terminal && open.get(&index) == Some(&true),
                    "deltas belong to an open block"
                );
                let delta = event.find("delta").expect("delta");
                if api == 1 || delta.member_str("type", "") == "input_json_delta" {
                    let tool = *tools
                        .get(&index)
                        .expect("arguments follow their tool header");
                    out.arguments[tool] += &if api == 1 {
                        event.member_str("delta", "")
                    } else {
                        delta.member_str("partial_json", "")
                    };
                } else if delta.member_str("type", "") == "thinking_delta" {
                    out.reasoning += &delta.member_str("thinking", "");
                } else if delta.member_str("type", "") == "text_delta" {
                    out.content += &delta.member_str("text", "");
                }
            }
            "response.function_call_arguments.done" => {
                assert!(
                    open.get(&index) == Some(&true)
                        && tools.get(&index).is_some_and(
                            |&tool| event.member_str("arguments", "") == out.arguments[tool]
                        ),
                    "argument done event matches its live tool"
                );
            }
            "response.output_text.delta" => out.content += &event.member_str("delta", ""),
            "response.reasoning_summary_text.delta" => {
                out.reasoning += &event.member_str("delta", "")
            }
            "response.completed" | "response.incomplete" | "message_delta" => {
                assert!(!terminal, "one terminal event");
                terminal = true;
                assert!(
                    open.values().all(|active| !active),
                    "all blocks close before the terminal event"
                );
                out.finish = if api == 1 {
                    event
                        .find("response")
                        .expect("response")
                        .member_str("status", "")
                } else {
                    event
                        .find("delta")
                        .expect("delta")
                        .member_str("stop_reason", "")
                };
            }
            _ => {}
        }
    }
    for call in &chat_calls {
        assert!(
            !call.member_str("id", "").is_empty(),
            "Chat tools have stable identities"
        );
        let function = call.find("function").expect("function");
        out.names.push(function.member_str("name", ""));
        out.arguments.push(function.member_str("arguments", ""));
    }
    assert!(
        terminal || out.failed,
        "stream terminates or reports its error"
    );
    out
}

fn whole(raw: &str) -> Vec<Vec<u8>> {
    vec![raw.as_bytes().to_vec()]
}

fn bytewise(raw: &str) -> Vec<Vec<u8>> {
    raw.bytes().map(|byte| vec![byte]).collect()
}

fn split_at(raw: &str, cut: usize) -> Vec<Vec<u8>> {
    vec![
        raw.as_bytes()[..cut].to_vec(),
        raw.as_bytes()[cut..].to_vec(),
    ]
}

fn is_json(text: &str) -> bool {
    json::parse_str(text).is_ok()
}

/// Every split of the reasoning tags, of a call's tags and values, and every
/// interruption keeps the phases and calls the buffered parse reports
/// (upstream `0377339`).
#[test]
fn tool_tag_boundaries() {
    use FinishReason as Finish;
    use InitialOutputState as Initial;
    let request = tool_boundary_request;
    for format in [ToolFormat::Qwen, ToolFormat::DeepSeek] {
        let qwen = format == ToolFormat::Qwen;
        for initial in [Initial::Reasoning, Initial::Auto] {
            for finish in [Finish::Length, Finish::StopSequence] {
                let phase = "<think>Check.</think>\n";
                for api in 0..3 {
                    for cut in 0..=phase.len() {
                        let raw = &phase[..cut];
                        let buffered = request(&whole(raw), format, api, false, initial, finish);
                        let streamed = request(&bytewise(raw), format, api, true, initial, finish);
                        assert!(
                            !streamed.failed
                                && !buffered.failed
                                && streamed.arguments.is_empty()
                                && streamed.content == buffered.content
                                && streamed.reasoning == buffered.reasoning
                                && !streamed.reasoning.contains("<think>")
                                && !streamed.reasoning.contains("</think>"),
                            "an interrupted thinking tag keeps the admitted phase without a \
                             tool: api={api} raw={raw:?} {streamed:?} {buffered:?}"
                        );
                    }
                }
            }
        }
        for frame in ["\n", "\r\n"] {
            if !qwen && frame == "\r\n" {
                continue;
            }
            let opening = if qwen {
                format!("<tool_call>\n<function=record>\n<parameter=text>{frame}")
            } else {
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n\
                 <｜DSML｜parameter name=\"text\" string=\"true\">"
                    .to_owned()
            };
            let closing = if qwen {
                format!("{frame}</parameter>{frame}</function>\n</tool_call>")
            } else {
                "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>".to_owned()
            };
            for value in [
                "",
                "\r",
                "\n",
                " \t\r\n",
                "café 🍵 \"quoted\" \\",
                "literal <think></think> <tool_call></tool_call> <|im_end|>",
                "line\n</parameter> is data\n</parameter>\rX\nlast",
            ] {
                let raw = format!("{opening}{value}{closing}");
                let expected = Value::object().with("text", value).dump().expect("dump");
                for api in 0..3 {
                    let buffered = request(
                        &whole(&raw),
                        format,
                        api,
                        false,
                        Initial::Content,
                        Finish::Stop,
                    );
                    assert!(
                        !buffered.failed
                            && buffered.arguments == [expected.clone()]
                            && buffered.content.is_empty()
                            && buffered.reasoning.is_empty(),
                        "the buffered fixture keeps the literal value: api={api} \
                         raw={raw:?} {buffered:?}"
                    );
                    // Every two-piece split, and a stream split at every byte.
                    for cut in 0..=raw.len() + 1 {
                        if cut <= raw.len() && !raw.is_char_boundary(cut) {
                            continue;
                        }
                        let pieces = if cut > raw.len() {
                            bytewise(&raw)
                        } else {
                            split_at(&raw, cut)
                        };
                        let streamed =
                            request(&pieces, format, api, true, Initial::Content, Finish::Stop);
                        assert!(
                            !streamed.failed
                                && streamed.arguments == buffered.arguments
                                && streamed.names == buffered.names
                                && streamed.content.is_empty()
                                && streamed.reasoning.is_empty(),
                            "every tag split keeps values and hides native framing: \
                             api={api} cut={cut} raw={raw:?} {streamed:?}"
                        );
                    }
                }
            }
            let call = format!("{opening}value{closing}");
            let noop = if qwen {
                "<tool_call>\n<function=noop>\n</function>\n</tool_call>"
            } else {
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"noop\">\n\
                 </｜DSML｜invoke>\n</｜DSML｜tool_calls>"
            };
            for initial in [Initial::Content, Initial::Reasoning, Initial::Auto] {
                for implicit in [false, true] {
                    if implicit && initial != Initial::Reasoning {
                        continue;
                    }
                    let prefix = if initial == Initial::Content {
                        String::new()
                    } else if implicit {
                        "Check tags.\n\n".to_owned()
                    } else {
                        format!(
                            "{}Check tags.</think>\n",
                            if initial == Initial::Auto {
                                "<think>"
                            } else {
                                ""
                            }
                        )
                    };
                    let raw = format!("{prefix}{call}\n{noop}");
                    for api in 0..3 {
                        let buffered =
                            request(&whole(&raw), format, api, false, initial, Finish::Stop);
                        for cut in (0..=raw.len()).filter(|&cut| raw.is_char_boundary(cut)) {
                            let streamed = request(
                                &split_at(&raw, cut),
                                format,
                                api,
                                true,
                                initial,
                                Finish::Stop,
                            );
                            assert!(
                                !streamed.failed
                                    && !buffered.failed
                                    && streamed.arguments == buffered.arguments
                                    && streamed.names == ["record", "noop"]
                                    && streamed.reasoning.trim() == buffered.reasoning.trim()
                                    && streamed.reasoning.trim()
                                        == if initial == Initial::Content {
                                            ""
                                        } else {
                                            "Check tags."
                                        }
                                    && streamed.content == buffered.content
                                    && streamed.content.trim().is_empty(),
                                "reasoning boundaries and successive calls agree across \
                                 transports: api={api} cut={cut} initial={initial:?} \
                                 implicit={implicit} {streamed:?} {buffered:?}"
                            );
                        }
                    }
                }
            }
            for api in 0..3 {
                for enabled in [false, true] {
                    let raw = format!("<think>Check tags.</think>\n{call}");
                    let initial = if enabled {
                        Initial::Reasoning
                    } else {
                        Initial::Content
                    };
                    for stream in [false, true] {
                        let output =
                            request(&bytewise(&raw), format, api, stream, initial, Finish::Stop);
                        assert!(
                            !output.failed
                                && output.reasoning.trim()
                                    == if enabled { "Check tags." } else { "" }
                                && output.content
                                    == if enabled {
                                        ""
                                    } else {
                                        "<think>Check tags.</think>\n"
                                    }
                                && output.arguments == [r#"{"text":"value"}"#],
                            "an opening think tag follows the admitted initial phase: \
                             api={api} enabled={enabled} stream={stream} {output:?}"
                        );
                    }
                }
            }
            // A stop or the limit at every byte of both calls: a partial JSON
            // prefix is never repaired into a call.
            for finish in [Finish::Length, Finish::StopSequence] {
                for api in 0..3 {
                    for cut in (0..=call.len()).filter(|&cut| call.is_char_boundary(cut)) {
                        let raw = format!("Check.</think>\n{noop}\n{}", &call[..cut]);
                        let buffered =
                            request(&whole(&raw), format, api, false, Initial::Reasoning, finish);
                        let streamed = request(
                            &bytewise(&raw),
                            format,
                            api,
                            true,
                            Initial::Reasoning,
                            finish,
                        );
                        assert!(
                            !streamed.failed
                                && !buffered.failed
                                && streamed.arguments.len() >= buffered.arguments.len()
                                && streamed.arguments.len() <= buffered.arguments.len() + 1
                                && streamed.content == buffered.content
                                && streamed.reasoning.trim() == "Check.",
                            "interrupted tags keep the preceding calls and reasoning: \
                             api={api} cut={cut} {streamed:?} {buffered:?}"
                        );
                        assert_eq!(
                            streamed.arguments[..buffered.arguments.len()],
                            buffered.arguments[..],
                            "calls completed before the interruption stay unchanged"
                        );
                        if streamed.arguments.len() > buffered.arguments.len() {
                            assert!(
                                !is_json(streamed.arguments.last().expect("prefix")),
                                "an interrupted prefix is not executable JSON"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Quoted examples, backticks in arguments and duplicate parameters while
/// streaming (upstream `0377339`).
#[test]
fn streaming_quoted_and_malformed_tools() {
    let request = |pieces: &[Vec<u8>], format, api, stream, initial| {
        tool_boundary_request(pieces, format, api, stream, initial, FinishReason::Stop)
    };
    for format in [ToolFormat::Qwen, ToolFormat::DeepSeek] {
        let qwen = format == ToolFormat::Qwen;
        let opening = if qwen {
            "<tool_call>\n<function=record>\n"
        } else {
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n"
        };
        let parameter = |value: &str| {
            if qwen {
                format!("<parameter=text>\n{value}\n</parameter>\n")
            } else {
                format!(
                    "<｜DSML｜parameter name=\"text\" string=\"true\">{value}</｜DSML｜parameter>\n"
                )
            }
        };
        let closing = if qwen {
            "</function>\n</tool_call>"
        } else {
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        };
        let call = format!("{opening}{}{closing}", parameter("value"));
        for api in 0..3 {
            let data = r#"{"literal":"</parameter></｜DSML｜parameter>","phase":"<think></think>","nested":"<tool_call><function=noop>"}"#;
            let raw = if qwen {
                format!("{opening}<parameter=data>\n{data}\n</parameter>\n{closing}")
            } else {
                format!(
                    "{opening}<｜DSML｜parameter name=\"data\" string=\"false\">{data}\
                     </｜DSML｜parameter>\n{closing}"
                )
            };
            let output = request(
                &bytewise(&raw),
                format,
                api,
                true,
                InitialOutputState::Content,
            );
            let expected = Value::object()
                .with("data", json::parse_str(data).expect("json"))
                .dump()
                .expect("dump");
            assert!(
                !output.failed
                    && output.arguments == [expected]
                    && output.content.is_empty()
                    && output.reasoning.is_empty(),
                "JSON string delimiters inside typed parameters stay argument data: \
                 api={api} {output:?}"
            );
            let raw = format!(
                "{opening}{}{closing}\n{opening}{}{closing}",
                parameter("unmatched ` in data"),
                parameter("later ` in data")
            );
            for stream in [false, true] {
                let output = request(
                    &bytewise(&raw),
                    format,
                    api,
                    stream,
                    InitialOutputState::Content,
                );
                assert!(
                    !output.failed
                        && output.content.trim().is_empty()
                        && output.reasoning.is_empty()
                        && output.arguments
                            == [
                                r#"{"text":"unmatched ` in data"}"#,
                                r#"{"text":"later ` in data"}"#,
                            ],
                    "backticks inside arguments cannot quote a later call: api={api} \
                     stream={stream} {output:?}"
                );
            }
            for quote in ["`", "```\n"] {
                let example = format!("Example: {quote}{call}{quote}\n");
                for reasoning in [false, true] {
                    let raw = format!(
                        "{example}{}{call}",
                        if reasoning { "</think>\n" } else { "" }
                    );
                    let initial = if reasoning {
                        InitialOutputState::Reasoning
                    } else {
                        InitialOutputState::Content
                    };
                    let buffered = request(&whole(&raw), format, api, false, initial);
                    let streamed = request(&bytewise(&raw), format, api, true, initial);
                    assert!(
                        !streamed.failed
                            && !buffered.failed
                            && streamed.names == ["record"]
                            && streamed.arguments == [r#"{"text":"value"}"#]
                            && streamed.content == buffered.content
                            && streamed.reasoning.trim() == buffered.reasoning.trim(),
                        "closing quotes never turn a streamed example into a phantom call: \
                         api={api} quote={quote:?} reasoning={reasoning} {streamed:?} {buffered:?}"
                    );
                }
            }
            // Duplicate parameters keep their semantics; once conflicting
            // bytes have streamed, the stream fails instead of completing an
            // unsafe call.
            for conflicting in [false, true] {
                let raw = format!(
                    "{opening}{}{}{closing}",
                    parameter("value"),
                    parameter(if conflicting { "changed" } else { "value" })
                );
                let streamed = request(
                    &bytewise(&raw),
                    format,
                    api,
                    true,
                    InitialOutputState::Content,
                );
                if conflicting {
                    assert!(
                        streamed.failed
                            && streamed.finish.is_empty()
                            && !streamed.arguments.is_empty()
                            && !is_json(streamed.arguments.last().expect("prefix")),
                        "conflicting parameters fail without a terminal event or \
                         executable JSON: api={api} {streamed:?}"
                    );
                } else {
                    assert!(
                        !streamed.failed && streamed.arguments == [r#"{"text":"value"}"#],
                        "identical duplicate parameters stream once: api={api} {streamed:?}"
                    );
                }
            }
        }
    }
}

/// A parameter the schema rejects ends a streamed call's arguments: neither
/// it nor a later unfinished value is published.
#[test]
fn streaming_stops_at_rejected_parameter() {
    for format in [ToolFormat::Qwen, ToolFormat::DeepSeek] {
        let raw = if format == ToolFormat::Qwen {
            "<tool_call>\n<function=record>\n<parameter=data>\nnot json\n\
             </parameter>\n<parameter=text>\nlater value\n</parameter>\n\
             </function>\n</tool_call>"
        } else {
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n\
             <｜DSML｜parameter name=\"data\" string=\"false\">[]</｜DSML｜parameter>\n\
             <｜DSML｜parameter name=\"text\" string=\"true\">later value</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        };
        for pieces in [whole(raw), bytewise(raw)] {
            for api in 0..3 {
                let output = tool_boundary_request(
                    &pieces,
                    format,
                    api,
                    true,
                    InitialOutputState::Content,
                    FinishReason::Stop,
                );
                assert!(
                    output.failed,
                    "a rejected streamed call fails safely: {output:?}"
                );
                for arguments in &output.arguments {
                    assert!(
                        !arguments.contains("data") && !arguments.contains("later"),
                        "neither the rejected parameter nor a later one streams: {arguments}"
                    );
                }
            }
        }
    }
}

/// A non-string DSML parameter that does not parse as its declared type
/// rejects the call, as for Qwen arguments; strings and undeclared
/// parameters stay as the dialect wrote them.
#[test]
fn deepseek_schema_checked_arguments() {
    let call = |name: &str, string: &str, value: &str| {
        format!(
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"run\"><｜DSML｜parameter name=\"{name}\" \
             string=\"{string}\">{value}</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"
        )
    };
    for (text, accepted, arguments) in [
        (call("tags", "false", "[1,2]"), true, r#"{"tags":[1,2]}"#),
        (call("tags", "false", "oops"), false, ""),
        (call("tags", "false", r#"{"a":1}"#), false, ""),
        (call("tags", "true", "oops"), true, r#"{"tags":"oops"}"#),
        // The dialect's string attribute decides; a declared string is not
        // retyped from the schema.
        (call("tags", "true", "[1]"), true, r#"{"tags":"[1]"}"#),
        // An undeclared parameter still has to satisfy the dialect's JSON
        // claim.
        (call("unknown", "false", "oops"), false, ""),
        (call("unknown", "false", "[1]"), true, r#"{"unknown":[1]}"#),
        (
            call("unknown", "true", "oops"),
            true,
            r#"{"unknown":"oops"}"#,
        ),
    ] {
        let backend = FakeBackend::new();
        backend.set_pieces(&[&text]);
        let response = chat(
            &backend,
            r#"{"model":"test-model","messages":[{"role":"user","content":"run"}],
            "tools":[{"type":"function","function":{"name":"run",
            "parameters":{"type":"object","properties":{"tags":{"type":"array"}}}}}]}"#,
        );
        assert_eq!(response.status, 200, "DeepSeek schema request is accepted");
        let output = json::parse(&response.body).expect("json");
        let message = output.find("choices").expect("choices").items()[0]
            .find("message")
            .expect("message")
            .clone();
        let calls = message
            .find("tool_calls")
            .map_or(0, |calls| calls.items().len());
        assert_eq!(
            calls == 1,
            accepted,
            "declared DSML parameter types: {text}"
        );
        if accepted {
            let function = message.find("tool_calls").expect("calls").items()[0]
                .find("function")
                .expect("function")
                .clone();
            let parsed = json::parse_str(&function.member_str("arguments", "")).expect("arguments");
            assert_eq!(parsed.dump().expect("dump"), arguments, "{text}");
        }
    }
}

/// Native tool arguments reach every streaming API while the model still
/// writes them, under one header; a disconnect mid-call cancels the
/// generation (upstream `0377339`).
#[test]
fn incremental_tool_arguments() {
    let value = "first line\nliteral </parameter> is data\n\
                 <tool_call> and <|im_end|> stay literal; \"quoted\" \\ café";
    for format in [ToolFormat::Qwen, ToolFormat::DeepSeek] {
        let qwen = format == ToolFormat::Qwen;
        for api in 0..3u8 {
            for reasoning in [false, true] {
                let opening = if qwen {
                    "<tool_call>\n<function=record>\n<parameter=text>\n"
                } else {
                    "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n\
                     <｜DSML｜parameter name=\"text\" string=\"true\">"
                };
                let closing = if qwen {
                    "\n</parameter>\n</function>\n</tool_call>"
                } else {
                    "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
                };
                let mut body = json::parse_str(
                    r#"{"model":"test-model","stream":true,
                    "messages":[{"role":"user","content":"record the literal text"}],
                    "tools":[{"type":"function","function":{"name":"record",
                      "parameters":{"type":"object","properties":{"text":{"type":"string"}},
                      "required":["text"],"additionalProperties":false}}}],
                    "tool_choice":"required"}"#,
                )
                .expect("json");
                if reasoning {
                    body.set("reasoning_effort", "low");
                }
                let start = |backend: &Arc<FakeBackend>| match api {
                    0 => chat(backend, &body.dump().expect("dump")),
                    1 => responses(backend, &body, reasoning, 512, true),
                    _ => {
                        let mut flat = body.clone();
                        let tools: Vec<Value> = body
                            .find("tools")
                            .expect("tools")
                            .items()
                            .iter()
                            .map(|tool| {
                                tool.find("function")
                                    .expect("function")
                                    .clone()
                                    .with("type", "function")
                            })
                            .collect();
                        flat.set("tools", tools);
                        let mut chat = ChatRequest::default();
                        chat.reasoning.enabled = Some(reasoning);
                        parse_tool_controls(&flat, &mut chat, true).expect("tool controls");
                        let dynamic: Arc<dyn TextGenerationBackend> = backend.clone();
                        create_anthropic_message(
                            &request("{}"),
                            &dynamic,
                            &chat,
                            512,
                            &SamplingConfig::default(),
                            true,
                        )
                        .expect("message")
                    }
                };
                let backend = FakeBackend::new();
                {
                    let mut config = backend.config();
                    config.tool_format = Some(format);
                    config.block_after_first_piece = true;
                    let mut pieces = vec![
                        format!(
                            "{}{opening}{}",
                            if reasoning {
                                "Check briefly.</think>\n"
                            } else {
                                ""
                            },
                            &value[..10]
                        )
                        .into_bytes(),
                    ];
                    // Every byte boundary, UTF-8 and escaped JSON included.
                    pieces.extend(
                        format!("{}{closing}", &value[10..])
                            .bytes()
                            .map(|b| vec![b]),
                    );
                    config.pieces = pieces;
                }
                #[derive(Default)]
                struct Seen {
                    arguments: String,
                    name: String,
                    id: String,
                    content: String,
                    fragments: usize,
                    headers: usize,
                    failed: bool,
                }
                let seen = Arc::new((Mutex::new(Seen::default()), Condvar::new()));
                let mut response = start(&backend);
                let body_fn = response.streaming_body.take().expect("stream");
                let writer = {
                    let seen = seen.clone();
                    std::thread::spawn(move || {
                        let mut pending = String::new();
                        body_fn(&mut |chunk: &[u8]| {
                            pending.push_str(&String::from_utf8_lossy(chunk));
                            while let Some(end) = pending.find("\n\n") {
                                let block: String = pending.drain(..end + 2).collect();
                                let Some(data) =
                                    block.lines().find_map(|l| l.strip_prefix("data: "))
                                else {
                                    continue;
                                };
                                if data.starts_with("[DONE]") {
                                    continue;
                                }
                                let event = json::parse_str(data).expect("event");
                                let (lock, condition) = &*seen;
                                let mut seen = lock.lock().expect("seen");
                                seen.failed |= event.contains("error")
                                    || event.member_str("type", "") == "response.failed";
                                let append = |seen: &mut Seen, piece: String| {
                                    if !piece.is_empty() {
                                        seen.arguments += &piece;
                                        seen.fragments += 1;
                                    }
                                };
                                let header = |seen: &mut Seen, item: &Value| {
                                    seen.name = item.member_str("name", "");
                                    seen.id =
                                        item.member_str("call_id", &item.member_str("id", ""));
                                    seen.headers += 1;
                                };
                                let kind = event.member_str("type", "");
                                if api == 0 {
                                    for choice in
                                        event.find("choices").map_or(&[][..], |c| c.items())
                                    {
                                        let delta = choice.find("delta").expect("delta");
                                        seen.content += &delta.member_str("content", "");
                                        for call in
                                            delta.find("tool_calls").map_or(&[][..], |c| c.items())
                                        {
                                            assert_eq!(
                                                call.member_size("index", 9),
                                                0,
                                                "deltas keep their tool index"
                                            );
                                            let function = call.find("function").expect("function");
                                            if !call.member_str("id", "").is_empty() {
                                                seen.id = call.member_str("id", "");
                                                seen.name = function.member_str("name", "");
                                                seen.headers += 1;
                                            }
                                            append(&mut seen, function.member_str("arguments", ""));
                                        }
                                    }
                                } else if api == 1 {
                                    if kind == "response.output_item.added"
                                        && event.find("item").expect("item").member_str("type", "")
                                            == "function_call"
                                    {
                                        header(&mut seen, event.find("item").expect("item"));
                                    }
                                    if kind == "response.function_call_arguments.delta" {
                                        append(&mut seen, event.member_str("delta", ""));
                                    }
                                    if kind == "response.output_text.delta" {
                                        seen.content += &event.member_str("delta", "");
                                    }
                                } else {
                                    if kind == "content_block_start"
                                        && event
                                            .find("content_block")
                                            .expect("block")
                                            .member_str("type", "")
                                            == "tool_use"
                                    {
                                        header(
                                            &mut seen,
                                            event.find("content_block").expect("block"),
                                        );
                                    }
                                    if let Some(delta) =
                                        event.find("delta").filter(|d| d.is_object())
                                    {
                                        match delta.member_str("type", "").as_str() {
                                            "input_json_delta" => append(
                                                &mut seen,
                                                delta.member_str("partial_json", ""),
                                            ),
                                            "text_delta" => {
                                                seen.content += &delta.member_str("text", "")
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                condition.notify_all();
                            }
                            true
                        })
                        .expect("stream");
                    })
                };
                assert!(
                    backend.wait_for_first_piece(),
                    "the model reaches the partial call"
                );
                let live = {
                    let (lock, condition) = &*seen;
                    let guard = lock.lock().expect("seen");
                    let (guard, _) = condition
                        .wait_timeout_while(guard, Duration::from_secs(2), |seen| {
                            !seen.arguments.contains("first line")
                        })
                        .expect("seen");
                    assert!(
                        !backend.completed.load(Ordering::SeqCst),
                        "generation is still blocked"
                    );
                    guard.arguments.contains("first line")
                };
                backend.release();
                writer.join().expect("writer");
                let seen = seen.0.lock().expect("seen");
                let label = format!("{format:?} api={api} reasoning={reasoning}");
                assert!(
                    live,
                    "argument bytes arrive before generation completes: {label}"
                );
                assert!(
                    !seen.failed,
                    "the fragmented tool stream completes: {label}"
                );
                assert!(
                    seen.headers == 1 && !seen.id.is_empty() && seen.name == "record",
                    "one stable tool header: {label}"
                );
                assert!(
                    seen.fragments > 2,
                    "arguments arrive incrementally: {label}"
                );
                assert_eq!(
                    json::parse_str(&seen.arguments)
                        .expect("arguments")
                        .member_str("text", ""),
                    value,
                    "the fragments keep the literal bytes: {label}"
                );
                assert!(
                    seen.content.is_empty(),
                    "tool markup stays out of the content: {label}"
                );
                // A disconnect while the backend still writes arguments
                // cancels it, through every serializer.
                let backend = FakeBackend::new();
                {
                    let mut config = backend.config();
                    config.tool_format = Some(format);
                    config.pieces = format!(
                        "{}{opening}{value}{closing}",
                        if reasoning {
                            "Check briefly.</think>\n"
                        } else {
                            ""
                        }
                    )
                    .bytes()
                    .map(|b| vec![b])
                    .collect();
                }
                let mut response = start(&backend);
                let mut disconnected = false;
                if let Some(body_fn) = response.streaming_body.take() {
                    body_fn(&mut |chunk: &[u8]| {
                        let text = String::from_utf8_lossy(chunk);
                        let Some(data) = text.lines().find_map(|l| l.strip_prefix("data: ")) else {
                            return true;
                        };
                        if data.starts_with("[DONE]") {
                            return true;
                        }
                        let event = json::parse_str(data).expect("event");
                        for choice in event.find("choices").map_or(&[][..], |c| c.items()) {
                            disconnected |= choice
                                .find("delta")
                                .is_some_and(|d| d.contains("tool_calls"));
                        }
                        disconnected |= event.member_str("type", "")
                            == "response.function_call_arguments.delta";
                        disconnected |= event
                            .find("delta")
                            .filter(|d| d.is_object())
                            .is_some_and(|d| d.member_str("type", "") == "input_json_delta");
                        !disconnected
                    })
                    .expect("stream");
                }
                assert!(
                    disconnected && !backend.completed.load(Ordering::SeqCst),
                    "a disconnect during a partial tool cancels the generation: {label}"
                );
            }
        }
    }
}

/// Chat tool messages keep their images at their text offsets; other roles
/// still reject images.
#[test]
fn tool_images_retain_order_and_identity() {
    let backend = FakeBackend::new();
    backend.set_pieces(&["ok"]);
    let response = chat(
        &backend,
        r#"{"model":"test-model","messages":[{"role":"tool","tool_call_id":"call_picture",
        "content":[
          {"type":"text","text":"left "},
          {"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}},
          {"type":"text","text":" right"},
          {"type":"image_url","image_url":{"url":"data:image/png;base64,BAUG"}}]}]}"#,
    );
    assert_eq!(response.status, 200, "{}", body_text(&response));
    let message = backend.last_request().messages[0].clone();
    assert!(
        message.role == gufo_core::text::ChatRole::Tool
            && message.tool_call_id == "call_picture"
            && message.content == "left  right"
            && message.images.len() == 2
            && message.images[0].offset == 5
            && message.images[1].offset == 11
    );
    for role in ["assistant", "system", "developer"] {
        let body = format!(
            r#"{{"model":"test-model","messages":[{{"role":"{role}","content":[
            {{"type":"image_url","image_url":{{"url":"data:image/png;base64,AQID"}}}}]}}]}}"#
        );
        assert_eq!(chat(&backend, &body).status, 400, "{role}");
    }
    for image in [
        r#"{"url":17}"#,
        r#"{"url":"data:image/png;base64,AQID","detail":"ultra"}"#,
        r#"{"url":"file:///tmp/image.png"}"#,
        r#"{"url":"data:image/png;base64,A==="}"#,
    ] {
        let body = format!(
            r#"{{"model":"test-model","messages":[{{"role":"tool","content":[
            {{"type":"image_url","image_url":{image}}}]}}]}}"#
        );
        assert_eq!(chat(&backend, &body).status, 400, "{image}");
    }
}
