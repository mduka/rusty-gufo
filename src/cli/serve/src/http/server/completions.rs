//! The OpenAI completions and Responses routes and the Anthropic Messages
//! routes.

use super::*;

pub(super) fn openai_completions(
    request: &HttpRequest,
    backend: &Arc<dyn TextGenerationBackend>,
) -> Result<HttpResponse> {
    compatibility_errors((|| {
        let body = try_response!(parse_body(request));
        let (max_tokens, sampling) = try_response!(read_compatibility_options(
            &body,
            backend,
            "max_tokens",
            &["stop", "stream", "stream_options", "ignore_eos"],
            backend.reasoning_defaults().enabled,
        ));
        let stream = body.find("stream").is_some_and(|s| s.as_bool(false));
        let mut include_usage = false;
        if let Some(options) = body.find("stream_options") {
            if !stream || !options.is_object() {
                return Ok(invalid_compatibility_request(
                    "'stream_options' requires streaming and must be an object",
                ));
            }
            for (key, value) in options.members() {
                let Value::Bool(value) = value else {
                    return Ok(invalid_compatibility_request(
                        "only boolean 'stream_options.include_usage' is supported",
                    ));
                };
                if key != "include_usage" {
                    return Ok(invalid_compatibility_request(
                        "only boolean 'stream_options.include_usage' is supported",
                    ));
                }
                include_usage = *value;
            }
        }
        let ignore_eos = match body.find("ignore_eos") {
            None => false,
            Some(Value::Bool(value)) => *value,
            Some(_) => {
                return Ok(invalid_compatibility_request(
                    "'ignore_eos' must be a boolean",
                ));
            }
        };
        let return_progress = match body.find("return_progress") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(value)) => *value,
            Some(_) => {
                return Ok(invalid_compatibility_request(
                    "'return_progress' must be a boolean",
                ));
            }
        };
        let stop_sequences =
            match parse_stop_sequences(body.find("stop"), StopSequenceFormat::OpenAi) {
                Ok(sequences) => sequences,
                Err(message) => return Ok(invalid_compatibility_request(&message)),
            };
        let cache_prompt = try_response!(read_cache_prompt(&body));
        let Some(Value::String(prompt)) = body.find("prompt") else {
            return Ok(invalid_compatibility_request(
                "'prompt' must be a single string",
            ));
        };
        if prompt.is_empty() {
            return Ok(err(
                400,
                "Bad Request",
                "'prompt' is required",
                "invalid_request_error",
                "missing_prompt",
            ));
        }
        let mut generation = start_complete(
            backend,
            prompt,
            max_tokens,
            &sampling,
            request.is_cancelled.clone(),
            stream,
            ignore_eos,
            &request.client_id,
            &stop_sequences,
            return_progress,
            cache_prompt,
        )?;
        let id = random_id("cmpl-", 12);
        let created = unix_time();
        let model = backend.model_id();
        if stream {
            return Ok(streaming_completion(
                generation,
                id,
                created,
                model,
                include_usage,
                return_progress,
            ));
        }
        let result = generation.wait(None)?;
        let choice = Value::object()
            .with("text", text_of(&result))
            .with("index", 0)
            .with("logprobs", Value::Null)
            .with(
                "finish_reason",
                if result.finish_reason == FinishReason::Length {
                    "length"
                } else {
                    "stop"
                },
            );
        let response = Value::object()
            .with("id", id)
            .with("object", "text_completion")
            .with("created", created)
            .with("model", model)
            .with("choices", vec![choice])
            .with("usage", usage_json(&result))
            .with("timings", generation_timings(&result));
        Ok(with_timing(ok(&response), &result))
    })())
}

/// Raw Completions over SSE: empty-text `prompt_progress` chunks when asked,
/// one `text_completion` chunk per piece, a terminal chunk with the finish
/// reason, optionally a usage chunk, then `[DONE]`.
pub(super) fn streaming_completion(
    mut generation: Box<dyn GenerationRequest>,
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    return_progress: bool,
) -> HttpResponse {
    let stream_log = Arc::new(Mutex::new(StreamLog::default()));
    let log = stream_log.clone();
    let body: crate::http::StreamingBody = Box::new(move |writer| {
        let chunk = |text: &str, finish_reason: Option<&str>, usage: Option<Value>| {
            let base = Value::object()
                .with("id", id.as_str())
                .with("object", "text_completion")
                .with("created", created)
                .with("model", model.as_str());
            match usage {
                Some(usage) => base
                    .with("choices", Vec::<Value>::new())
                    .with("usage", usage),
                None => base.with(
                    "choices",
                    vec![
                        Value::object()
                            .with("text", text)
                            .with("index", 0)
                            .with("logprobs", Value::Null)
                            .with(
                                "finish_reason",
                                finish_reason.map_or(Value::Null, Value::from),
                            ),
                    ],
                ),
            }
        };
        let mut decoder = Utf8Decoder::default();
        let connected = std::cell::Cell::new(true);
        // Any write commits the deferred headers; after that, failures are
        // SSE errors.
        let started = std::cell::Cell::new(return_progress);
        let writer = std::cell::RefCell::new(writer);
        let write = |bytes: &[u8]| {
            started.set(true);
            let sent = connected.get() && (writer.borrow_mut())(bytes);
            connected.set(sent);
            sent
        };
        let mut on_progress = |progress: &PromptProgress| {
            write(&sse(
                &chunk("", None, None).with("prompt_progress", prompt_progress_json(progress))
            ))
        };
        // Admitted or long queued: commit the headers.
        let mut on_start = || write(b"");
        let outcome = generation.wait_streaming(StreamCallbacks {
            on_token: Some(&mut |piece: &[u8]| {
                write(&sse(&chunk(&decoder.push(piece, false), None, None)))
            }),
            on_progress: if return_progress {
                Some(&mut on_progress)
            } else {
                None
            },
            on_start: Some(&mut on_start),
        });
        let connected = connected.get();
        let writer = writer.into_inner();
        let result = match outcome {
            Ok(result) => result,
            Err(failure) if !started.get() => return Err(failure),
            Err(failure) => {
                let (message, code) = crate::openai_chat::stream_failure(&failure);
                let event = Value::object().with(
                    "error",
                    Value::object()
                        .with("message", message)
                        .with("type", "server_error")
                        .with("code", code),
                );
                let sent = writer(&sse(&event));
                if let Ok(mut log) = log.lock() {
                    log.error_code = code.to_owned();
                    log.error_event_sent = sent;
                }
                writer(b"data: [DONE]\n\n");
                return Ok(());
            }
        };
        if let Ok(mut log) = log.lock() {
            log.details = generation_log_details(&result);
        }
        record_server_metrics(&result);
        if !connected || result.cancelled {
            return Ok(());
        }
        let finish = if result.finish_reason == FinishReason::Length {
            "length"
        } else {
            "stop"
        };
        // llama.cpp reports timings on the terminal chunk regardless of the
        // optional usage chunk.
        let terminal = chunk(&decoder.push(&[], true), Some(finish), None)
            .with("timings", generation_timings(&result));
        if !writer(&sse(&terminal)) {
            return Ok(());
        }
        if include_usage && !writer(&sse(&chunk("", None, Some(usage_json(&result))))) {
            return Ok(());
        }
        writer(b"data: [DONE]\n\n");
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

pub(super) fn openai_responses(
    request: &HttpRequest,
    backend: &Arc<dyn TextGenerationBackend>,
    semantics: Semantics,
) -> Result<HttpResponse> {
    let target = semantics.target;
    compatibility_errors((|| {
        let body = try_response!(parse_body(request));
        let reasoning = match read_reasoning(&body, ChatApi::Responses, semantics.compat) {
            Ok(reasoning) => with_reasoning_defaults(reasoning, backend),
            Err(message) => return Ok(invalid_compatibility_request(&message)),
        };
        let (max_tokens, sampling) = try_response!(read_compatibility_options(
            &body,
            backend,
            "max_output_tokens",
            &[
                "reasoning",
                "chat_template_kwargs",
                "text",
                "stream",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                // A standard Responses field (encrypted reasoning, for
                // example): no server-side conversation is kept, so it is
                // accepted and ignored.
                "include",
            ],
            reasoning.enabled,
        ));
        let mut messages = Vec::new();
        if let Some(instructions) = body.find("instructions") {
            let Value::String(instructions) = instructions else {
                return Ok(invalid_compatibility_request(
                    "'instructions' must be a string",
                ));
            };
            messages.push(ChatMessage::new(ChatRole::System, instructions.clone()));
        }
        match body.find("input") {
            Some(Value::String(input)) if !input.is_empty() => {
                messages.push(ChatMessage::new(ChatRole::User, input.clone()));
            }
            input => {
                let mut parse_error = None;
                if !read_text_messages(
                    input,
                    ChatApi::Responses,
                    semantics,
                    backend.supports_images(),
                    &mut messages,
                    &mut parse_error,
                ) {
                    return Ok(invalid_compatibility_request(
                        parse_error.as_deref().unwrap_or(
                            "'input' must be nonempty text, or messages with text and images, \
                         function calls and their outputs",
                        ),
                    ));
                }
            }
        }
        let mut chat = ChatRequest::new(messages);
        chat.client_id = request.client_id.clone();
        chat.reasoning = reasoning;
        if let Err((message, code)) = parse_response_tool_controls(&body, &mut chat) {
            return Ok(err(
                400,
                "Bad Request",
                &message,
                "invalid_request_error",
                code,
            ));
        }
        if let Err(message) = parse_openai_response_format(&body, &mut chat) {
            return Ok(err(
                400,
                "Bad Request",
                &message,
                "invalid_request_error",
                "invalid_response_format",
            ));
        }
        match body.find("cache_prompt") {
            None | Some(Value::Null) => {}
            Some(Value::Bool(cache_prompt)) => chat.cache_prompt = *cache_prompt,
            Some(_) => {
                return Ok(err(
                    400,
                    "Bad Request",
                    "'cache_prompt' must be a boolean",
                    "invalid_request_error",
                    "invalid_cache_prompt",
                ));
            }
        }
        match body.find("return_progress") {
            None | Some(Value::Null) => {}
            Some(Value::Bool(return_progress)) => chat.return_progress = *return_progress,
            Some(_) => {
                return Ok(err(
                    400,
                    "Bad Request",
                    "'return_progress' must be a boolean",
                    "invalid_request_error",
                    "invalid_return_progress",
                ));
            }
        }
        let stream = body
            .find("stream")
            .is_some_and(|stream| stream.as_bool(false));
        crate::openai_chat::create_openai_response(
            request, backend, &chat, max_tokens, &sampling, stream, target,
        )
    })())
}

/// The request fields Messages and `count_tokens` share
/// (`ReadAnthropicChat`): sampling and reasoning controls, tools, the system
/// text and the turns. The error is the response.
#[allow(clippy::result_large_err)]
pub(super) fn read_anthropic_chat(
    body: &Value,
    backend: &Arc<dyn TextGenerationBackend>,
    compat: ApiCompat,
) -> std::result::Result<(ChatRequest, usize, SamplingConfig), HttpResponse> {
    let reasoning = match read_reasoning(body, ChatApi::Anthropic, compat) {
        Ok(reasoning) => with_reasoning_defaults(reasoning, backend),
        Err(message) => return Err(invalid_compatibility_request(&message)),
    };
    let (max_tokens, sampling) = read_compatibility_options(
        body,
        backend,
        "max_tokens",
        &[
            "stream",
            "thinking",
            "output_config",
            "chat_template_kwargs",
            "stop_sequences",
            "tools",
            "tool_choice",
        ],
        reasoning.enabled,
    )?;
    let mut chat = ChatRequest::new(Vec::new());
    chat.reasoning = reasoning;
    if let Err((message, code)) = parse_anthropic_tool_controls(body, &mut chat) {
        return Err(err(
            400,
            "Bad Request",
            &message,
            "invalid_request_error",
            code,
        ));
    }
    let mut messages = Vec::new();
    if let Some(system) = body.find("system") {
        let mut text = String::new();
        if !read_text_content(Some(system), &mut text) {
            return Err(invalid_compatibility_request(
                "'system' must contain only text",
            ));
        }
        messages.push(ChatMessage::new(ChatRole::System, text));
    }
    let mut parse_error = None;
    if !read_text_messages(
        body.find("messages"),
        ChatApi::Anthropic,
        Semantics {
            compat,
            ..Semantics::default()
        },
        backend.supports_images(),
        &mut messages,
        &mut parse_error,
    ) {
        return Err(invalid_compatibility_request(
            parse_error.as_deref().unwrap_or(
                "'messages' must contain text, image, thinking, tool_use or tool_result blocks",
            ),
        ));
    }
    chat.messages = messages;
    chat.stop_sequences =
        parse_stop_sequences(body.find("stop_sequences"), StopSequenceFormat::Anthropic)
            .map_err(|message| invalid_compatibility_request(&message))?;
    Ok((chat, max_tokens, sampling))
}

pub(super) fn anthropic_messages(
    request: &HttpRequest,
    backend: &Arc<dyn TextGenerationBackend>,
    compat: ApiCompat,
) -> Result<HttpResponse> {
    compatibility_errors((|| {
        let body = try_response!(parse_body(request));
        let (mut chat, max_tokens, sampling) =
            try_response!(read_anthropic_chat(&body, backend, compat));
        chat.client_id = request.client_id.clone();
        let stream = body
            .find("stream")
            .is_some_and(|stream| stream.as_bool(false));
        crate::openai_chat::create_anthropic_message(
            request, backend, &chat, max_tokens, &sampling, stream,
        )
    })())
}

/// `POST /v1/messages/count_tokens` (`AnthropicCountTokens`): the prompt size a Messages request would have, cached tokens
/// included, without generating or touching the conversation cache.
pub(super) fn anthropic_count_tokens(
    request: &HttpRequest,
    backend: &Arc<dyn TextGenerationBackend>,
    compat: ApiCompat,
) -> Result<HttpResponse> {
    compatibility_errors((|| {
        let body = try_response!(parse_body(request));
        let (chat, _, _) = try_response!(read_anthropic_chat(&body, backend, compat));
        if body
            .find("stream")
            .is_some_and(|stream| stream.as_bool(false))
        {
            return Ok(invalid_compatibility_request("'stream' must be false"));
        }
        match backend.count_chat_tokens(&chat)? {
            Some(tokens) => Ok(ok(&Value::object().with("input_tokens", tokens))),
            None => not_implemented(request, backend),
        }
    })())
}
