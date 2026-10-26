//! Responses as they stream: the text filter that holds back tool-call
//! markers, streamed tool-call deltas, and the non-streaming and streaming
//! Chat Completions responses.

use super::*;

pub(crate) fn sse(value: &Value) -> Vec<u8> {
    format!("data: {}\n\n", value.dump().unwrap_or_default()).into_bytes()
}

pub(super) fn base_chunk(id: &str, created: u64, model: &str) -> Value {
    Value::object()
        .with("id", id)
        .with("object", "chat.completion.chunk")
        .with("created", created)
        .with("model", model)
}

pub(super) fn choice_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish: Option<&str>,
) -> Value {
    let choice = Value::object()
        .with("index", 0)
        .with("delta", delta)
        .with("finish_reason", finish.map_or(Value::Null, Value::from));
    base_chunk(id, created, model).with("choices", vec![choice])
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FilterState {
    Initial,
    Thinking,
    Content,
}

/// Splits streamed text into reasoning and content deltas, holding back
/// possible partial markers and hiding text once a tool marker appears.
/// Emits a delta: the text and whether it is reasoning.
pub(super) type EmitPiece<'a> = Box<dyn FnMut(&str, bool) -> bool + 'a>;

pub(super) struct StreamingTextFilter<'a> {
    pub(super) emit: EmitPiece<'a>,
    pub(super) decoder: Utf8Decoder,
    pub(super) raw: String,
    pub(super) pending: String,
    pub(super) hidden: String,
    pub(super) state: FilterState,
    pub(super) tool_mode: bool,
    /// The separator after `</think>` is dropped, also across pieces.
    pub(super) trim_reasoning_separator: bool,
    /// Qwen's separator is any whitespace ([`after_reasoning_separator`]).
    pub(super) qwen: bool,
    /// Structured output: after the reasoning, content passes unparsed.
    pub(super) raw_content: bool,
    /// Constrained tool calls without a response format: prose streams,
    /// calls are withheld.
    pub(super) tool_only: bool,
    /// The tool markup withheld from the content.
    pub(super) markers: &'static [&'static str],
    pub(super) closers: &'static [&'static str],
    /// The declared tools, whose client envelopes are framing.
    pub(super) tools: &'a [ChatTool],
    pub(super) structured_started: bool,
    pub(super) emitted_content_bytes: usize,
    pub(super) emitted_reasoning_bytes: usize,
    /// The quoting of `raw`; `pending` is always a suffix of it.
    pub(super) quotes: QuoteTracker,
    /// A reasoning phase the prompt opened may still begin with its own
    /// `<think>`, which the parsed output drops too.
    pub(super) reasoning_start: bool,
}

impl<'a> StreamingTextFilter<'a> {
    /// `structured` passes the answer after the reasoning unparsed;
    /// `tool_only` withholds constrained tool calls but streams prose; only
    /// `syntax` delimits tool markup ([`tool_syntax`]).
    pub(super) fn structured(
        initial: InitialOutputState,
        emit: impl FnMut(&str, bool) -> bool + 'a,
        structured: bool,
        tool_only: bool,
        syntax: ToolSyntax,
        tools: &'a [ChatTool],
    ) -> Self {
        Self {
            emit: Box::new(emit),
            decoder: Utf8Decoder::default(),
            raw: String::new(),
            pending: String::new(),
            hidden: String::new(),
            state: match initial {
                InitialOutputState::Reasoning => FilterState::Thinking,
                InitialOutputState::Content => FilterState::Content,
                InitialOutputState::Auto => FilterState::Initial,
            },
            tool_mode: false,
            trim_reasoning_separator: false,
            qwen: syntax.qwen,
            raw_content: structured || tool_only,
            tool_only,
            markers: syntax.markers,
            closers: syntax.closers,
            tools,
            structured_started: false,
            emitted_content_bytes: 0,
            emitted_reasoning_bytes: 0,
            quotes: QuoteTracker::default(),
            reasoning_start: initial == InitialOutputState::Reasoning,
        }
    }

    /// Structured content: a leading tool call is withheld whole; other JSON
    /// streams as it comes (`StructuredContent`).
    pub(super) fn structured_content(&mut self) -> bool {
        if self.tool_only {
            let offset = self.pending_offset();
            if let Some(found) = earliest_marker(&self.raw, self.markers, &self.quotes, offset)
                .map(|(found, _)| found)
            {
                let prose = content_before(
                    &self.raw,
                    offset,
                    found,
                    self.closers,
                    &self.quotes,
                    self.tools,
                    false,
                );
                if !prose.is_empty() && !(self.emit)(prose, false) {
                    return false;
                }
                self.emitted_content_bytes += prose.len();
                self.hidden = self.pending[found - offset..].to_owned();
                self.pending.clear();
                self.tool_mode = true;
                return true;
            }
            let ready = self.pending.len() - self.held_content(offset);
            if ready > 0 {
                let text = content_before(
                    &self.raw,
                    offset,
                    offset + ready,
                    self.closers,
                    &self.quotes,
                    self.tools,
                    false,
                );
                if !text.is_empty() && !(self.emit)(text, false) {
                    return false;
                }
                self.emitted_content_bytes += text.len();
            }
            self.pending.drain(..ready);
            return true;
        }
        if !self.structured_started {
            let content = trim(&self.pending);
            let content_offset = self.pending_offset() + offset_in(&self.pending, content);
            let at_marker = earliest_marker(&self.raw, self.markers, &self.quotes, content_offset)
                .is_some_and(|(p, _)| p == content_offset);
            if content.is_empty() || (tool_marker_prefix(content, self.markers) && !at_marker) {
                return true;
            }
            self.structured_started = true;
            if at_marker {
                self.tool_mode = true;
                self.hidden = std::mem::take(&mut self.pending);
                return true;
            }
        }
        let text = std::mem::take(&mut self.pending);
        text.is_empty() || (self.emit)(&text, false)
    }

    pub(super) fn push(&mut self, bytes: &[u8], last: bool) -> bool {
        let piece = self.decoder.push(bytes, last);
        self.raw.push_str(&piece);
        self.quotes.append(&piece);
        if self.tool_mode {
            self.hidden.push_str(&piece);
            return true;
        }
        self.pending.push_str(&piece);
        if self.reasoning_start {
            // Without exposing a split opener as thought.
            const THINK_START: &str = "<think>";
            if !last
                && self.pending.len() < THINK_START.len()
                && THINK_START.starts_with(self.pending.as_str())
            {
                return true;
            }
            if self.pending.starts_with(THINK_START) {
                self.pending.drain(..THINK_START.len());
            }
            self.reasoning_start = false;
        }
        if self.raw_content && self.state == FilterState::Content && !self.trim_reasoning_separator
        {
            return self.structured_content();
        }
        if self.state == FilterState::Initial {
            const THINK_START: &str = "<think>";
            let view = self.pending.trim_start_matches(is_space);
            if view.is_empty() {
                return true;
            }
            // A piece may carry the marker and what follows it: a buffered
            // text, or several tokens accepted at once.
            if let Some(rest) = view.strip_prefix(THINK_START) {
                self.pending = rest.to_owned();
                self.state = FilterState::Thinking;
            } else if THINK_START.starts_with(view) {
                return true;
            } else {
                self.state = FilterState::Content;
            }
        }
        if self.state == FilterState::Thinking {
            const THINK_END: &str = "</think>";
            let offset = self.pending_offset();
            let end = self.pending.find(THINK_END);
            // Constrained reasoning ends at `</think>` or a native call
            // header outside quotations; bare markers in it are data.
            let thinking_markers = if self.raw_content {
                reasoning_tool_markers(self.markers)
            } else {
                self.markers
            };
            let marker = if self.raw_content {
                reasoning_tool_marker(&self.raw, thinking_markers, &self.quotes, offset)
            } else {
                earliest_marker(&self.raw, thinking_markers, &self.quotes, offset)
                    .map(|(found, _)| found)
            }
            .map(|found| found - offset);
            if let Some(marker) = marker
                && end.is_none_or(|e| marker < e)
            {
                if marker > 0 && !(self.emit)(&self.pending[..marker], true) {
                    return false;
                }
                self.emitted_reasoning_bytes += marker;
                self.hidden = self.pending[marker..].to_owned();
                self.pending.clear();
                self.tool_mode = true;
                return true;
            }
            match end {
                Some(end) => {
                    if end > 0 && !(self.emit)(&self.pending[..end], true) {
                        return false;
                    }
                    self.emitted_reasoning_bytes += end;
                    self.pending.drain(..end + THINK_END.len());
                    self.state = FilterState::Content;
                    // The phase transition starts a fresh quote scope.
                    self.quotes.reset(&self.raw, offset + end + THINK_END.len());
                    self.trim_reasoning_separator = true;
                }
                None => {
                    let mut held = held_marker_prefix(
                        &self.raw,
                        self.raw.len(),
                        thinking_markers,
                        &self.quotes,
                    )
                    .min(self.pending.len());
                    let bytes = self.pending.as_bytes();
                    for length in (1..=bytes.len().min(THINK_END.len() - 1)).rev() {
                        if THINK_END
                            .as_bytes()
                            .starts_with(&bytes[bytes.len() - length..])
                        {
                            held = held.max(length);
                            break;
                        }
                    }
                    let ready = self.pending.len() - held;
                    if ready > 0 && !(self.emit)(&self.pending[..ready], true) {
                        return false;
                    }
                    self.emitted_reasoning_bytes += ready;
                    self.pending.drain(..ready);
                    return true;
                }
            }
        }
        if self.state == FilterState::Content {
            if self.trim_reasoning_separator {
                let content = after_reasoning_separator(&self.pending, self.qwen).len();
                if content == 0 {
                    self.pending.clear();
                    return true;
                }
                self.pending.drain(..self.pending.len() - content);
                self.trim_reasoning_separator = false;
            }
            if self.raw_content {
                return self.structured_content();
            }
            let offset = self.pending_offset();
            if let Some(found) = earliest_marker(&self.raw, self.markers, &self.quotes, offset)
                .map(|(found, _)| found)
            {
                let prose = content_before(
                    &self.raw,
                    offset,
                    found,
                    self.closers,
                    &self.quotes,
                    self.tools,
                    false,
                );
                if !prose.is_empty() && !(self.emit)(prose, false) {
                    return false;
                }
                self.hidden = self.pending[found - offset..].to_owned();
                self.pending.clear();
                self.tool_mode = true;
                return true;
            }
            let ready = self.pending.len() - self.held_content(offset);
            if ready > 0 && !(self.emit)(&self.pending[..ready], false) {
                return false;
            }
            self.pending.drain(..ready);
        }
        true
    }

    /// Where `pending` begins in `raw`.
    pub(super) fn pending_offset(&self) -> usize {
        self.raw.len() - self.pending.len()
    }

    /// The bytes at the end of `pending` (which begins at `offset` in `raw`)
    /// that may still open a call or a client envelope.
    pub(super) fn held_content(&self, offset: usize) -> usize {
        let marker = held_marker_prefix(&self.raw, self.raw.len(), self.markers, &self.quotes);
        let framing = framing_hold(
            &self.raw,
            self.raw.len(),
            offset,
            self.closers,
            &self.quotes,
            self.tools,
        );
        marker.max(framing).min(self.pending.len())
    }

    /// `content` is the parsed answer; `tool_only` streams its prose between
    /// or after complete calls without their markup.
    /// `reasoning` is the parsed reasoning: a quoted example taken for a
    /// call inside the reasoning returns there.
    pub(super) fn finish(
        &mut self,
        hide_tool_markup: bool,
        content: &str,
        reasoning: &str,
    ) -> bool {
        if self.tool_mode && self.state == FilterState::Thinking {
            let tail = reasoning
                .get(self.emitted_reasoning_bytes.min(reasoning.len())..)
                .unwrap_or("");
            return (tail.is_empty() || (self.emit)(tail, true))
                && (content.is_empty() || (self.emit)(content, false));
        }
        if self.tool_only && self.tool_mode && hide_tool_markup {
            let tail = content.get(self.emitted_content_bytes..).unwrap_or("");
            return tail.is_empty() || (self.emit)(tail, false);
        }
        if self.raw_content && !self.structured_started && hide_tool_markup {
            self.pending.clear();
            return true;
        }
        if !self.tool_mode {
            if !self.pending.is_empty() {
                // With tools declared, the closing tags a model repeats after
                // a call are markup, never assistant prose a client stores.
                let reasoning = self.state == FilterState::Thinking;
                let offset = self.pending_offset();
                let text = if self.state != FilterState::Content {
                    &self.pending
                } else {
                    content_before(
                        &self.raw,
                        offset,
                        self.raw.len(),
                        self.closers,
                        &self.quotes,
                        self.tools,
                        false,
                    )
                };
                let emitted = (self.emit)(text, reasoning);
                self.pending.clear();
                return emitted;
            }
            return true;
        }
        if !hide_tool_markup && !self.hidden.is_empty() {
            let hidden = self.hidden.clone();
            return (self.emit)(&hidden, false);
        }
        true
    }
}

/// One published change of a streamed tool call (`ToolDelta`): the first
/// carries the call's `id` and `name`, later ones leave them empty; each
/// carries the new suffix of its JSON arguments, and `complete` once they
/// are final.
pub(super) struct ToolDelta<'c> {
    pub(super) index: usize,
    pub(super) id: &'c str,
    pub(super) name: &'c str,
    pub(super) arguments: &'c str,
    pub(super) complete: bool,
}

pub(super) struct SentToolCall {
    pub(super) id: String,
    pub(super) name: String,
    pub(super) arguments: String,
    pub(super) complete: bool,
}

/// The tool calls a stream has published (`StreamingToolCalls`, upstream
/// `0377339`): as llama.cpp's `task_result_state`, IDs are kept by call
/// index and only the newly parsed argument suffix is published. The final
/// parse stays authoritative.
#[derive(Default)]
pub(super) struct StreamingToolCalls {
    pub(super) sent: Vec<SentToolCall>,
}

impl StreamingToolCalls {
    /// Publishes what `calls`, a parse of the output so far, add.
    pub(super) fn update(
        &mut self,
        calls: &[ParsedToolCall],
        emit: &mut dyn FnMut(&ToolDelta<'_>) -> bool,
    ) -> bool {
        for (index, call) in calls.iter().enumerate() {
            let arguments = call
                .partial_arguments
                .clone()
                .unwrap_or_else(|| arguments_json(&call.arguments));
            let added = index == self.sent.len();
            if added {
                self.sent.push(SentToolCall {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: String::new(),
                    complete: false,
                });
            }
            let previous = &mut self.sent[index];
            // A still unfinished quotation or value can make a partial parse
            // disappear for a while; only the final parse accepts the turn.
            if previous.name != call.name || !arguments.starts_with(previous.arguments.as_str()) {
                continue;
            }
            let complete = call.partial_arguments.is_none();
            let delta = &arguments[previous.arguments.len()..];
            if (added || !delta.is_empty() || (complete && !previous.complete))
                && !emit(&ToolDelta {
                    index,
                    id: if added { &previous.id } else { "" },
                    name: if added { &previous.name } else { "" },
                    arguments: delta,
                    complete,
                })
            {
                return false;
            }
            previous.arguments = arguments;
            previous.complete = complete;
        }
        true
    }

    /// Publishes the rest of the final `calls`; every published call must
    /// have completed as the final parse has it, except an unfinished one
    /// that a stop sequence or the token limit cut (`interrupted`).
    pub(super) fn finish(
        &mut self,
        calls: &[ParsedToolCall],
        interrupted: bool,
        emit: &mut dyn FnMut(&ToolDelta<'_>) -> bool,
    ) -> Result<bool> {
        if !self.update(calls, emit) {
            return Ok(false);
        }
        for (index, sent) in self.sent.iter().enumerate() {
            let safe = match calls.get(index) {
                None => interrupted && !sent.complete,
                Some(call) => {
                    sent.complete
                        && sent.name == call.name
                        && sent.arguments == arguments_json(&call.arguments)
                }
            };
            if !safe {
                return Err(Error::runtime("streamed tool call did not complete safely"));
            }
        }
        Ok(true)
    }
}

/// The message and stable code a stream ends with when generation fails
/// after its head: the failure's own message, which the
/// log keeps too, beside the client's stable code.
pub(crate) fn stream_failure(failure: &Error) -> (String, &'static str) {
    let code = failure
        .generation_code()
        .map_or("generation_failed", |code| code.stable_code());
    if failure.generation_code().is_none() {
        logging::error("chat", failure.message());
    }
    let message = if failure.message().is_empty() {
        "generation failed".to_owned()
    } else {
        failure.message().to_owned()
    };
    (message, code)
}

pub(super) fn non_streaming_response(
    request: &ParsedChatRequest,
    backend: &Arc<dyn TextGenerationBackend>,
    mut generation: Box<dyn GenerationRequest>,
    initial: InitialOutputState,
) -> Result<HttpResponse> {
    let syntax = tool_syntax(&request.chat, generation.tool_format());
    let result = generation.wait(None)?;
    let text = Utf8Decoder::default().push(&result.text, true);
    let generated = parse_request_generation(
        &request.chat,
        syntax,
        &text,
        initial,
        result.finish_reason == FinishReason::Stop,
        false,
    )?;
    let mut message = Value::object().with("role", "assistant");
    if !generated.reasoning_content.is_empty() {
        message.set("reasoning_content", generated.reasoning_content.as_str());
    }
    if generated.text.is_empty() && !generated.tool_calls.is_empty() {
        message.set("content", Value::Null);
    } else {
        message.set("content", generated.text.as_str());
    }
    if !generated.tool_calls.is_empty() {
        message.set("tool_calls", tool_calls_json(&generated.tool_calls));
    }
    let choice = Value::object()
        .with("index", 0)
        .with("message", message)
        .with(
            "finish_reason",
            finish_reason(&result, !generated.tool_calls.is_empty()),
        );
    let response = Value::object()
        .with("id", random_id("chatcmpl-", 20))
        .with("object", "chat.completion")
        .with("created", unix_time())
        .with("model", backend.model_id())
        .with("choices", vec![choice])
        .with("usage", usage(&result))
        .with("timings", generation_timings(&result));
    record_server_metrics(&result);
    let mut output = HttpResponse::new(200, "OK", dump(&response))
        .with_header("Server-Timing", server_timing(&result));
    output.log_details = generation_log_details(&result);
    Ok(output)
}

pub(super) fn streaming_response(
    request: ParsedChatRequest,
    backend: &Arc<dyn TextGenerationBackend>,
    mut generation: Box<dyn GenerationRequest>,
    initial: InitialOutputState,
) -> HttpResponse {
    let id = random_id("chatcmpl-", 20);
    let created = unix_time();
    let model = backend.model_id();
    let syntax = tool_syntax(&request.chat, generation.tool_format());
    let stream_log = Arc::new(Mutex::new(StreamLog::default()));
    let log = stream_log.clone();
    let return_progress = request.chat.return_progress;
    let body: crate::http::StreamingBody = Box::new(move |writer| {
        let chat = &request.chat;
        let connected = std::cell::Cell::new(true);
        let started = std::cell::Cell::new(false);
        let writer = std::cell::RefCell::new(writer);
        let write = |bytes: &[u8]| (writer.borrow_mut())(bytes);
        // The role chunk commits the deferred headers, once the request
        // starts or produces anything; failures before it are HTTP errors.
        let begin = || {
            if !started.replace(true) {
                let role_delta = Value::object().with("role", "assistant");
                connected.set(write(&sse(&choice_chunk(
                    &id, created, &model, role_delta, None,
                ))));
            }
            connected.get()
        };
        if chat.return_progress && !begin() {
            generation.cancel();
            return Ok(());
        }
        let mut filter = StreamingTextFilter::structured(
            initial,
            |text: &str, reasoning: bool| {
                if text.is_empty() {
                    return true;
                }
                let delta = if reasoning {
                    Value::object().with("reasoning_content", text)
                } else {
                    Value::object().with("content", text)
                };
                let sent = write(&sse(&choice_chunk(&id, created, &model, delta, None)));
                connected.set(sent);
                sent
            },
            chat.response_format.is_some(),
            chat.constrained_tools && chat.response_format.is_none(),
            syntax,
            &chat.tools,
        );
        let mut tools = StreamingToolCalls::default();
        let mut emit_tool = |call: &ToolDelta<'_>| {
            // Completion without more arguments needs no Chat delta.
            if call.id.is_empty() && call.arguments.is_empty() {
                return connected.get();
            }
            let mut item = Value::object().with("index", call.index);
            let mut function = Value::object();
            if !call.id.is_empty() {
                item.set("id", call.id);
                item.set("type", "function");
                function.set("name", call.name);
            }
            function.set("arguments", call.arguments);
            item.set("function", function);
            let delta = Value::object().with("tool_calls", vec![item]);
            let sent = write(&sse(&choice_chunk(&id, created, &model, delta, None)));
            connected.set(sent);
            sent
        };
        let mut on_progress = |progress: &PromptProgress| {
            let chunk = choice_chunk(&id, created, &model, Value::object(), None)
                .with("prompt_progress", prompt_progress_json(progress));
            let sent = begin() && write(&sse(&chunk));
            connected.set(sent);
            sent
        };
        let mut on_start = || begin();
        let outcome = (|| -> Result<()> {
            let result = generation.wait_streaming(StreamCallbacks {
                on_token: Some(&mut |piece: &[u8]| {
                    if !begin() || !filter.push(piece, false) {
                        return false;
                    }
                    // Native calls stream their arguments as they parse.
                    !filter.tool_mode
                        || syntax.markers.len() > 2
                        || tools.update(
                            &parse_request_generation(
                                chat,
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
                on_progress: if chat.return_progress {
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
            if !begin() || result.cancelled {
                return Ok(());
            }
            if !filter.push(&[], true) {
                return Ok(());
            }
            let interrupted = result.finish_reason != FinishReason::Stop;
            let generated = parse_request_generation(
                chat,
                syntax,
                &filter.raw.clone(),
                initial,
                !interrupted,
                false,
            )?;
            if !tools.finish(&generated.tool_calls, interrupted, &mut emit_tool)? {
                return Ok(());
            }
            if !filter.finish(
                generated.hide_tool_markup,
                &generated.text,
                &generated.reasoning_content,
            ) {
                return Ok(());
            }
            drop(filter);
            // llama.cpp reports timings on the terminal choice regardless of
            // the optional usage chunk; proxies need them either way.
            let terminal = choice_chunk(
                &id,
                created,
                &model,
                Value::object(),
                Some(finish_reason(&result, !generated.tool_calls.is_empty())),
            )
            .with("timings", generation_timings(&result));
            if !write(&sse(&terminal)) {
                return Ok(());
            }
            if request.include_usage {
                let usage_chunk = base_chunk(&id, created, &model)
                    .with("choices", Value::array())
                    .with("usage", usage(&result))
                    .with("timings", generation_timings(&result));
                if !write(&sse(&usage_chunk)) {
                    return Ok(());
                }
            }
            write(b"data: [DONE]\n\n");
            Ok(())
        })();
        if let Err(failure) = outcome {
            if !started.get() {
                return Err(failure);
            }
            let (message, code) = stream_failure(&failure);
            if let Ok(mut log) = log.lock() {
                log.error_code = code.to_owned();
            }
            let error = Value::object().with(
                "error",
                Value::object()
                    .with("message", message)
                    .with("type", "server_error")
                    .with("code", code),
            );
            let sent = write(&sse(&error));
            if let Ok(mut log) = log.lock() {
                log.error_event_sent = sent;
            }
            write(b"data: [DONE]\n\n");
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
    response
}
