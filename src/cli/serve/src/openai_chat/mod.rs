//! The supported OpenAI Chat Completions subset
//! (`src/cli/serve/openai_chat.cpp`): request validation, reasoning
//! controls, Qwen and DSML tool calls, and streaming. Streamed responses
//! consume scheduler-published pieces inside `HttpResponse::streaming_body`;
//! socket writes never own or execute model state.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use gufo_core::json::{self, Value};
use gufo_core::json_constraint::JsonConstraint;
use gufo_core::sampling::SamplingConfig;
use gufo_core::text::{
    ChatMessage, ChatRequest, ChatRole, ChatTool, Error, ErrorKind, GenerationErrorCode,
    OutputGrammar, ReasoningEffort, ReasoningOptions, Result, ToolArgument, ToolCall, ToolChoice,
};
use gufo_core::utf8::Utf8Decoder;

use crate::backend::{
    FinishReason, GenerationRequest, GenerationResult, InitialOutputState, PromptProgress,
    StreamCallbacks, TextGenerationBackend, ToolFormat, start_chat,
};
use crate::generation_metrics::{
    generation_log_details, generation_timings, prefill_tokens_per_second, prompt_progress_json,
    server_timing,
};
use crate::http::{BodyWriter, HttpRequest, HttpResponse, StreamLog};
use crate::image_url::{ImageReadBudget, read_image_url};
use crate::logging;
use crate::metrics::record_server_metrics;
use crate::quote_tracker::QuoteTracker;
use crate::response_format::{parse_response_format, response_format_description};
use crate::sampling_request::parse_sampling_config;
use crate::stop_sequences::{StopSequenceFormat, parse_stop_sequences};
use crate::util::{random_id, unix_time};

mod anthropic;
mod generation;
mod markers;
mod request;
mod responses;
mod stream;
mod tool_calls;

pub(crate) use anthropic::*;
pub(crate) use generation::*;
use markers::*;
pub use request::*;
use responses::*;
pub(crate) use stream::*;
use tool_calls::*;

struct ParsedChatRequest {
    chat: ChatRequest,
    model: String,
    max_tokens: usize,
    sampling: SamplingConfig,
    stream: bool,
    include_usage: bool,
}

#[derive(Debug, Clone)]
struct ParsedToolCall {
    id: String,
    name: String,
    arguments: Vec<ToolArgument>,
    /// An append-only JSON prefix of an unfinished call while streaming;
    /// `None` on complete calls.
    partial_arguments: Option<String>,
}

#[derive(Debug, Default)]
struct ParsedGeneration {
    text: String,
    reasoning_content: String,
    tool_calls: Vec<ParsedToolCall>,
    /// Tool markup is withheld from the content, also an unfinished call.
    hide_tool_markup: bool,
}

/// Every known tool marker: the JSON fallback, whose grammar leaves text
/// before `<tool_call>` free, where models still write their native syntax,
/// and backends that do not report the admitted format.
const TOOL_MARKERS: [&str; 7] = [
    "<tool_call>",
    "<｜DSML｜tool_calls｜>",
    "<｜DSML｜tool_calls>",
    "<DSML｜tool_calls｜>",
    "<DSML｜tool_calls>",
    "<tool_calls｜>",
    "<tool_calls>",
];

fn dump(value: &Value) -> Vec<u8> {
    value.dump().unwrap_or_default().into_bytes()
}

/// The request errors of this API (`Error`), always `invalid_request_error`.
fn error(status: u16, reason: &str, message: &str, code: &str) -> HttpResponse {
    let body = Value::object().with(
        "error",
        Value::object()
            .with("message", message)
            .with("type", "invalid_request_error")
            .with("code", code),
    );
    HttpResponse::new(status, reason, dump(&body))
}

fn status_reason(status: u16) -> &'static str {
    match status {
        408 => "Request Timeout",
        429 => "Too Many Requests",
        503 => "Service Unavailable",
        502 => "Bad Gateway",
        _ => "Internal Server Error",
    }
}

fn generation_error(code: GenerationErrorCode, message: &str) -> HttpResponse {
    let body = Value::object().with(
        "error",
        Value::object()
            .with("message", message)
            .with("type", "server_error")
            .with("code", code.stable_code()),
    );
    let mut response = HttpResponse::new(
        code.http_status(),
        status_reason(code.http_status()),
        dump(&body),
    );
    if code.retryable() {
        response = response.with_header("Retry-After", "1");
    }
    response
}

/// The output of a chat compatibility API (`ResponsesOutput`,
/// `AnthropicOutput`): one document when buffered, its SSE events when
/// streamed. Both pass the Chat reasoning, UTF-8 and tool filters, so the
/// two transports carry the same items.
trait CompatibilityOutput {
    /// The events that commit the deferred headers.
    fn begin(&mut self) -> bool;
    /// Prompt progress before the first output.
    fn progress(&mut self, progress: &PromptProgress) -> bool;
    fn append(&mut self, piece: &str, reasoning: bool) -> bool;
    fn tool(&mut self, call: &ToolDelta<'_>) -> bool;
    /// Closes the open items and reports the final document.
    fn complete(&mut self, result: &GenerationResult) -> Value;
    fn fail(&mut self, message: &str) -> bool;
    fn connected(&self) -> bool;
}

/// Builds the output of one request, with the stream's writer or none.
type MakeOutput = for<'w> fn(
    &str,
    Option<&'w mut BodyWriter<'w>>,
    &ChatRequest,
) -> Box<dyn CompatibilityOutput + 'w>;

/// The request semantics of the compatibility routes where llama.cpp and
/// Gufo differ (`--api-compat`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApiCompat {
    /// Gufo's: Messages thinking without `budget_tokens` has no budget, a
    /// Responses `reasoning` item restores only reasoning the server emits
    /// (encrypted reasoning is refused), and unknown Responses `reasoning`
    /// members are refused.
    #[default]
    Gufo,
    /// llama.cpp's: enabled Messages thinking defaults to a 10000-token
    /// budget, a Responses `reasoning` item without readable text is
    /// ignored, and the `reasoning` object takes only known members with
    /// checked values while unknown ones pass.
    LlamaCpp,
}

impl ApiCompat {
    /// The option value: `gufo` or `llama.cpp`.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "gufo" => Some(Self::Gufo),
            "llama.cpp" => Some(Self::LlamaCpp),
            _ => None,
        }
    }
}

/// Where a Responses `reasoning` item carries the model's raw reasoning
/// (`--responses-reasoning-target`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResponsesReasoningTarget {
    /// Upstream gufo's: one `summary_text` part in `summary` and the
    /// `response.reasoning_summary_*` events.
    #[default]
    Summary,
    /// OpenAI's and llama.cpp's: one `reasoning_text` part in `content` and
    /// the `response.reasoning_text.*` events.
    ReasoningText,
    /// Open Responses': the `reasoning_text` part in `content` and the
    /// `response.reasoning.*` events.
    Reasoning,
}

impl ResponsesReasoningTarget {
    /// The option value: `summary`, `reasoning-text` or `reasoning`.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "summary" => Some(Self::Summary),
            "reasoning-text" => Some(Self::ReasoningText),
            "reasoning" => Some(Self::Reasoning),
            _ => None,
        }
    }

    /// The streamed text's event: `delta` or `done`.
    fn text_event(self, kind: &str) -> String {
        match self {
            Self::Summary => format!("response.reasoning_summary_text.{kind}"),
            Self::ReasoningText => format!("response.reasoning_text.{kind}"),
            Self::Reasoning => format!("response.reasoning.{kind}"),
        }
    }
}

/// One Responses or Messages generation once validated
/// (`CreateCompatibilityResponse`): through the Chat reasoning, UTF-8 and
/// tool filters, buffered or as the SSE events of `make`'s output.
fn create_compatibility_response(
    request: &HttpRequest,
    backend: &Arc<dyn TextGenerationBackend>,
    chat: &ChatRequest,
    max_tokens: usize,
    sampling: &SamplingConfig,
    stream: bool,
    make: MakeOutput,
) -> Result<HttpResponse> {
    let initial = backend.initial_output_state(chat);
    let structured = chat.response_format.is_some();
    let tool_only = chat.constrained_tools && !structured;
    let return_progress = chat.return_progress;
    let model = backend.model_id();
    let mut generation = start_chat(
        backend,
        chat,
        max_tokens,
        sampling,
        request.is_cancelled.clone(),
        stream,
    )?;
    let syntax = tool_syntax(chat, generation.tool_format());
    if !stream {
        let result = generation.wait(None)?;
        let output = RefCell::new(make(&model, None, chat));
        let mut filter = StreamingTextFilter::structured(
            initial,
            |text: &str, reasoning: bool| output.borrow_mut().append(text, reasoning),
            structured,
            tool_only,
            syntax,
            &chat.tools,
        );
        filter.push(&result.text, false);
        filter.push(&[], true);
        let interrupted = result.finish_reason != FinishReason::Stop;
        let generated = parse_request_generation(
            chat,
            syntax,
            &filter.raw.clone(),
            initial,
            !interrupted,
            false,
        )?;
        StreamingToolCalls::default().finish(
            &generated.tool_calls,
            interrupted,
            &mut |call: &ToolDelta<'_>| output.borrow_mut().tool(call),
        )?;
        filter.finish(
            generated.hide_tool_markup,
            &generated.text,
            &generated.reasoning_content,
        );
        drop(filter);
        let response = output.into_inner().complete(&result);
        record_server_metrics(&result);
        let mut http = HttpResponse::new(200, "OK", dump(&response))
            .with_header("Server-Timing", server_timing(&result));
        http.log_details = generation_log_details(&result);
        return Ok(http);
    }
    let stream_log = Arc::new(Mutex::new(StreamLog::default()));
    let log = stream_log.clone();
    let chat = chat.clone();
    let body: crate::http::StreamingBody = Box::new(move |writer| {
        let output = RefCell::new(make(&model, Some(writer), &chat));
        let started = std::cell::Cell::new(false);
        // The lifecycle events commit the deferred headers, once the request
        // starts or produces anything; failures before them are HTTP errors.
        let begin = || {
            if !started.replace(true) {
                return output.borrow_mut().begin();
            }
            output.borrow().connected()
        };
        if return_progress && !begin() {
            generation.cancel();
            return Ok(());
        }
        let mut filter = StreamingTextFilter::structured(
            initial,
            |text: &str, reasoning: bool| output.borrow_mut().append(text, reasoning),
            structured,
            tool_only,
            syntax,
            &chat.tools,
        );
        let mut tools = StreamingToolCalls::default();
        let mut emit_tool = |call: &ToolDelta<'_>| output.borrow_mut().tool(call);
        let mut on_progress =
            |progress: &PromptProgress| begin() && output.borrow_mut().progress(progress);
        let mut on_start = || begin();
        let outcome = (|| -> Result<()> {
            let result = generation.wait_streaming(StreamCallbacks {
                on_token: Some(&mut |piece: &[u8]| {
                    if !begin() || !filter.push(piece, false) {
                        return false;
                    }
                    !filter.tool_mode
                        || syntax.markers.len() > 2
                        || tools.update(
                            &parse_request_generation(
                                &chat,
                                syntax,
                                &filter.raw,
                                initial,
                                false,
                                true,
                            )
                            .map(|generated| generated.tool_calls)
                            .unwrap_or_default(),
                            &mut emit_tool,
                        )
                }),
                on_progress: if return_progress {
                    Some(&mut on_progress)
                } else {
                    None
                },
                on_start: Some(&mut on_start),
            })?;
            if let Ok(mut log) = log.lock() {
                log.details = generation_log_details(&result);
            }
            record_server_metrics(&result);
            if result.cancelled || !begin() || !filter.push(&[], true) {
                return Ok(());
            }
            let interrupted = result.finish_reason != FinishReason::Stop;
            let generated = parse_request_generation(
                &chat,
                syntax,
                &filter.raw.clone(),
                initial,
                !interrupted,
                false,
            )?;
            if !tools.finish(&generated.tool_calls, interrupted, &mut emit_tool)?
                || !filter.finish(
                    generated.hide_tool_markup,
                    &generated.text,
                    &generated.reasoning_content,
                )
            {
                return Ok(());
            }
            output.borrow_mut().complete(&result);
            Ok(())
        })();
        drop(filter);
        if let Err(failure) = outcome {
            if !started.get() {
                return Err(failure);
            }
            let (message, code) = stream_failure(&failure);
            let sent = output.borrow_mut().fail(&message);
            if let Ok(mut log) = log.lock() {
                log.error_code = code.to_owned();
                log.error_event_sent = sent;
            }
        }
        if !output.borrow().connected() {
            generation.cancel();
        }
        Ok(())
    });
    let mut response = HttpResponse::new(200, "OK", Vec::new())
        .with_header("Content-Type", "text/event-stream")
        .with_header("Cache-Control", "no-cache")
        .with_header("X-Accel-Buffering", "no");
    response.streaming_body = Some(body);
    response.defer_stream_headers = !return_progress;
    response.stream_log = Some(stream_log);
    Ok(response)
}

/// `POST /v1/responses` once validated: one response, or its semantic SSE
/// events (`CreateOpenAiResponse`), the reasoning where `target` puts it.
pub fn create_openai_response(
    request: &HttpRequest,
    backend: &Arc<dyn TextGenerationBackend>,
    chat: &ChatRequest,
    max_tokens: usize,
    sampling: &SamplingConfig,
    stream: bool,
    target: ResponsesReasoningTarget,
) -> Result<HttpResponse> {
    // A function per target: the streamed body outlives this call.
    let make: MakeOutput = match target {
        ResponsesReasoningTarget::Summary => |model, writer, chat| {
            Box::new(ResponsesOutput::new(
                model,
                writer,
                chat,
                ResponsesReasoningTarget::Summary,
            ))
        },
        ResponsesReasoningTarget::ReasoningText => |model, writer, chat| {
            Box::new(ResponsesOutput::new(
                model,
                writer,
                chat,
                ResponsesReasoningTarget::ReasoningText,
            ))
        },
        ResponsesReasoningTarget::Reasoning => |model, writer, chat| {
            Box::new(ResponsesOutput::new(
                model,
                writer,
                chat,
                ResponsesReasoningTarget::Reasoning,
            ))
        },
    };
    create_compatibility_response(request, backend, chat, max_tokens, sampling, stream, make)
}

/// `POST /v1/messages` once validated: one message of content blocks, or
/// Anthropic's SSE events (`CreateAnthropicMessage`).
pub fn create_anthropic_message(
    request: &HttpRequest,
    backend: &Arc<dyn TextGenerationBackend>,
    chat: &ChatRequest,
    max_tokens: usize,
    sampling: &SamplingConfig,
    stream: bool,
) -> Result<HttpResponse> {
    create_compatibility_response(
        request,
        backend,
        chat,
        max_tokens,
        sampling,
        stream,
        anthropic_output,
    )
}

pub fn handle_openai_chat(
    request: &HttpRequest,
    backend: &Arc<dyn TextGenerationBackend>,
) -> Result<HttpResponse> {
    let defaults = backend.sampling_defaults();
    let mut parsed = ParsedChatRequest {
        chat: ChatRequest::default(),
        model: String::new(),
        max_tokens: defaults.max_tokens,
        sampling: defaults.sampling,
        stream: false,
        include_usage: false,
    };
    if let Some(response) = parse_request(request, backend, &mut parsed) {
        return Ok(response);
    }
    let initial = backend.initial_output_state(&parsed.chat);
    let outcome = start_chat(
        backend,
        &parsed.chat,
        parsed.max_tokens,
        &parsed.sampling,
        request.is_cancelled.clone(),
        parsed.stream,
    )
    .and_then(|generation| {
        if parsed.stream {
            Ok(streaming_response(parsed, backend, generation, initial))
        } else {
            non_streaming_response(&parsed, backend, generation, initial)
        }
    });
    match outcome {
        Err(failure) => match failure.kind() {
            ErrorKind::Generation(code) => Ok(generation_error(code, failure.message())),
            ErrorKind::InvalidArgument => Ok(error(
                400,
                "Bad Request",
                failure.message(),
                "invalid_prompt",
            )),
            ErrorKind::Length => Ok(error(
                400,
                "Bad Request",
                failure.message(),
                "context_length_exceeded",
            )),
            _ => Err(failure),
        },
        ok => ok,
    }
}
