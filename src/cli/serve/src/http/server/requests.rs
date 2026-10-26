//! Reading the compatibility APIs' requests: messages and their content,
//! reasoning items and options, Anthropic turns, and the request body.

use super::*;

pub(super) fn text_of(result: &GenerationResult) -> String {
    Utf8Decoder::default().push(&result.text, true)
}

pub(super) fn not_implemented(
    _: &HttpRequest,
    _: &Arc<dyn TextGenerationBackend>,
) -> Result<HttpResponse> {
    Ok(err(
        501,
        "Not Implemented",
        "endpoint not implemented on this text model",
        "server_error",
        "not_implemented",
    ))
}

pub(super) fn role_from(role: &str) -> ChatRole {
    match role {
        "system" => ChatRole::System,
        "developer" => ChatRole::Developer,
        "assistant" => ChatRole::Assistant,
        "tool" => ChatRole::Tool,
        _ => ChatRole::User,
    }
}

pub(super) fn read_text_content(content: Option<&Value>, out: &mut String) -> bool {
    match content {
        Some(Value::String(text)) => {
            *out = text.clone();
            true
        }
        Some(Value::Array(parts)) => {
            for part in parts {
                let kind = part.member_str("type", "");
                match part.find("text") {
                    Some(Value::String(text))
                        if part.is_object()
                            && matches!(kind.as_str(), "text" | "input_text" | "output_text") =>
                    {
                        out.push_str(text);
                    }
                    _ => return false,
                }
            }
            true
        }
        _ => false,
    }
}

/// The chat compatibility APIs: they differ in their reasoning controls and
/// in how a client returns earlier reasoning (llama.cpp's conversions).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ChatApi {
    Anthropic,
    Responses,
}

/// What a compatibility route accepts where llama.cpp and Gufo differ
/// (`--api-compat`), and where the Responses route puts the raw reasoning.
#[derive(Clone, Copy, Default)]
pub(super) struct Semantics {
    pub(super) compat: ApiCompat,
    pub(super) target: ResponsesReasoningTarget,
}

/// The text of a Responses `reasoning` item: its `reasoning_text` parts,
/// else its summary. Gufo restores only reasoning the server emits: a
/// `summary` array, and `content` only when the reasoning target writes it;
/// encrypted reasoning cannot be restored and is refused. llama.cpp ignores
/// an item with neither (only `encrypted_content`).
pub(super) fn reasoning_item_text(item: &Value, semantics: Semantics) -> Option<String> {
    let present = |field| item.find(field).is_some_and(|value| !value.is_null());
    if semantics.compat == ApiCompat::Gufo {
        let content = semantics.target != ResponsesReasoningTarget::Summary && present("content");
        let summary = item.find("summary").is_some_and(Value::is_array);
        if present("encrypted_content") || (present("content") && !content) || !(summary || content)
        {
            return None;
        }
    }
    for (field, kind) in [("content", "reasoning_text"), ("summary", "summary_text")] {
        let Some(parts) = item.find(field).filter(|parts| !parts.is_null()) else {
            continue;
        };
        let Value::Array(parts) = parts else {
            return None;
        };
        let mut text = String::new();
        for part in parts {
            match part.find("text") {
                Some(Value::String(piece)) if part.member_str("type", kind) == kind => {
                    text.push_str(piece);
                }
                _ => return None,
            }
        }
        if !text.is_empty() {
            return Some(text);
        }
    }
    Some(String::new())
}

/// The content of an Anthropic assistant message: text blocks and the
/// `thinking` blocks of earlier reasoning; redacted thinking is dropped.
pub(super) fn read_anthropic_assistant(content: Option<&Value>, message: &mut ChatMessage) -> bool {
    let Some(Value::Array(blocks)) = content else {
        return read_text_content(content, &mut message.content);
    };
    for block in blocks {
        match (block.member_str("type", "").as_str(), block.find("text")) {
            ("text", Some(Value::String(text))) => message.content.push_str(text),
            ("thinking", _) => match block.find("thinking") {
                Some(Value::String(thinking)) => message.thought.push_str(thinking),
                _ => return false,
            },
            ("redacted_thinking", _) => {}
            _ => return false,
        }
    }
    true
}

/// The content of an Anthropic user message: text, and image blocks at the
/// current end of the text. An image's error is the
/// `Err` message; other invalid content is `Err(None)`.
pub(super) fn read_anthropic_user(
    content: Option<&Value>,
    message: &mut ChatMessage,
    budget: &mut ImageReadBudget,
) -> std::result::Result<(), Option<String>> {
    let Some(Value::Array(blocks)) = content else {
        return read_text_content(content, &mut message.content)
            .then_some(())
            .ok_or(None);
    };
    for block in blocks {
        let kind = block.member_str("type", "");
        match block.find("text") {
            _ if block.is_object() && kind == "image" => {
                parse_anthropic_image(block, message, budget).map_err(Some)?;
            }
            Some(Value::String(text))
                if block.is_object()
                    && matches!(kind.as_str(), "text" | "input_text" | "output_text") =>
            {
                message.content.push_str(text);
            }
            _ => return Err(None),
        }
    }
    Ok(())
}

/// The text messages of `input`. Earlier reasoning becomes the thought of
/// its assistant message: Anthropic `thinking` blocks, and Responses
/// `reasoning` items, which join the assistant message around them.
/// `parse_error` receives why a Responses message item failed, such as an
/// unsupported image data URL.
///
/// Responses replays one assistant turn as several reasoning, message and
/// function_call items; a system/developer item after an assistant message
/// waits until the next item that starts a new message, so it does not split
/// that turn.
pub(super) fn read_text_messages(
    input: Option<&Value>,
    api: ChatApi,
    semantics: Semantics,
    images_supported: bool,
    messages: &mut Vec<ChatMessage>,
    parse_error: &mut Option<String>,
) -> bool {
    let Some(Value::Array(items)) = input else {
        return false;
    };
    if items.is_empty() {
        return false;
    }
    // Whether the last message holds only a reasoning item so far.
    let mut reasoning_only = false;
    let mut budget = ImageReadBudget {
        accepts_images: images_supported,
        ..ImageReadBudget::default()
    };
    let mut deferred = Vec::new();
    let push = |messages: &mut Vec<ChatMessage>, deferred: &mut Vec<_>, message| {
        messages.append(deferred);
        messages.push(message);
    };
    for item in items {
        // A Messages turn with tool blocks: its calls, or the results of the
        // previous calls.
        if api == ChatApi::Anthropic
            && let Some(Value::Array(parts)) = item.find("content")
            && parts.iter().any(|part| {
                part.is_object()
                    && matches!(
                        part.member_str("type", "").as_str(),
                        "tool_use" | "tool_result"
                    )
            })
        {
            if let Err(error) = parse_anthropic_tool_message(item, messages, &mut budget) {
                *parse_error = Some(error).filter(|error| !error.is_empty());
                return false;
            }
            reasoning_only = false;
            continue;
        }
        if api == ChatApi::Responses
            && item.is_object()
            && item.member_str("type", "") == "reasoning"
        {
            let Some(thought) = reasoning_item_text(item, semantics) else {
                return false;
            };
            if thought.is_empty() {
                continue;
            }
            match messages.last_mut() {
                Some(last) if last.role == ChatRole::Assistant => last.thought = thought,
                _ => {
                    let mut message = ChatMessage::new(ChatRole::Assistant, "");
                    message.thought = thought;
                    push(messages, &mut deferred, message);
                    reasoning_only = true;
                    continue;
                }
            }
            reasoning_only = false;
            continue;
        }
        // Function calls and their outputs, merged as llama.cpp does: a call
        // joins a preceding assistant message.
        if api == ChatApi::Responses && item.is_object() {
            match item.member_str("type", "").as_str() {
                "function_call" | "custom_tool_call" => {
                    let call = match parse_response_function_call(item) {
                        Ok(call) => call,
                        Err(error) => {
                            *parse_error = Some(error);
                            return false;
                        }
                    };
                    match messages.last_mut() {
                        Some(last) if last.role == ChatRole::Assistant => {
                            last.tool_calls.push(call);
                        }
                        _ => {
                            let mut message = ChatMessage::new(ChatRole::Assistant, "");
                            message.tool_calls.push(call);
                            push(messages, &mut deferred, message);
                        }
                    }
                    reasoning_only = false;
                    continue;
                }
                "function_call_output" | "custom_tool_call_output" => {
                    let message = match parse_response_function_output(item, &mut budget) {
                        Ok(message) => message,
                        Err(error) => {
                            *parse_error = Some(error);
                            return false;
                        }
                    };
                    push(messages, &mut deferred, message);
                    reasoning_only = false;
                    continue;
                }
                _ => {}
            }
        }
        let role = item.member_str("role", "");
        if !item.is_object()
            || !matches!(role.as_str(), "user" | "assistant" | "system" | "developer")
            || item.member_str("type", "message") != "message"
            || item.contains("tool_calls")
        {
            return false;
        }
        let mut message = ChatMessage::new(role_from(&role), "");
        let read = if api == ChatApi::Anthropic && message.role == ChatRole::Assistant {
            read_anthropic_assistant(item.find("content"), &mut message)
        } else if api == ChatApi::Anthropic && message.role == ChatRole::User {
            match read_anthropic_user(item.find("content"), &mut message, &mut budget) {
                Ok(()) => true,
                Err(error) => {
                    *parse_error = error;
                    false
                }
            }
        } else if api == ChatApi::Responses {
            match parse_openai_response_message(item, &mut message, &mut budget) {
                Ok(()) => true,
                Err(error) => {
                    *parse_error = Some(error).filter(|error| !error.is_empty());
                    false
                }
            }
        } else {
            read_text_content(item.find("content"), &mut message.content)
        };
        if !read {
            return false;
        }
        if api == ChatApi::Responses
            && matches!(message.role, ChatRole::System | ChatRole::Developer)
            && messages
                .last()
                .is_some_and(|last| last.role == ChatRole::Assistant)
        {
            deferred.push(message);
            continue;
        }
        match messages.last_mut() {
            Some(last) if reasoning_only && message.role == ChatRole::Assistant => {
                last.content = message.content;
            }
            _ => push(messages, &mut deferred, message),
        }
        reasoning_only = false;
    }
    messages.append(&mut deferred);
    true
}

/// The reasoning controls of a chat compatibility route, and
/// `chat_template_kwargs` as in llama.cpp. Anthropic's `thinking` enables
/// or disables thinking (`budget_tokens` is checked, not enforced); the
/// Responses `reasoning.effort` selects the effort, "none" disables it, and
/// summaries are never produced.
pub(super) fn read_reasoning(
    body: &Value,
    api: ChatApi,
    compat: ApiCompat,
) -> std::result::Result<ReasoningOptions, String> {
    let mut options = ReasoningOptions::default();
    match api {
        ChatApi::Anthropic => {
            if let Some(thinking) = body.find("thinking") {
                if !thinking.is_object() {
                    return Err("'thinking' must be an object".into());
                }
                // Adaptive leaves the choice to the server's default, as in
                // llama.cpp.
                options.enabled = match thinking.find("type") {
                    Some(Value::String(kind)) if kind == "enabled" => Some(true),
                    Some(Value::String(kind)) if kind == "disabled" => Some(false),
                    Some(Value::String(kind)) if kind == "adaptive" => None,
                    _ => {
                        return Err("'thinking.type' must be enabled, adaptive, or disabled".into());
                    }
                };
                // Thinking returns for every display: replayed blocks restore
                // the cached thought.
                if let Some(display) = thinking.find("display")
                    && !matches!(display, Value::String(d)
                        if matches!(d.as_str(), "summarized" | "omitted" | "updates"))
                {
                    return Err("'thinking.display' must be summarized, omitted, or updates".into());
                }
                // With enabled thinking, budget_tokens caps the reasoning.
                // Without it Gufo keeps the server's budget; llama.cpp's
                // Anthropic conversion uses 10000.
                let budget = match thinking.find("budget_tokens") {
                    None if compat == ApiCompat::LlamaCpp => Some(10_000.0),
                    None => None,
                    Some(budget) if budget.is_number() => Some(budget.as_double(0.0)),
                    Some(_) => Some(0.0),
                };
                if let Some(budget) = budget {
                    if !budget.is_finite()
                        || budget < 1.0
                        || budget > f64::from(i32::MAX)
                        || budget.floor() != budget
                    {
                        return Err(
                            "'thinking.budget_tokens' must be a positive integer at most \
                                2147483647"
                                .into(),
                        );
                    }
                    if options.enabled == Some(true) {
                        options.budget_tokens = Some(budget as usize);
                    }
                }
            }
            if let Some(config) = body.find("output_config").filter(|c| !c.is_null()) {
                if !config.is_object() {
                    return Err("'output_config' must be an object".into());
                }
                for (key, value) in config.members() {
                    if key != "effort" {
                        return Err(format!("unsupported output_config member: {key}"));
                    }
                    if value.is_null() {
                        continue;
                    }
                    // Anthropic's effort never selects thinking, and it has
                    // no minimal effort.
                    let effort = match value {
                        Value::String(effort) if effort != "minimal" => {
                            parse_reasoning_effort_name(effort)
                        }
                        _ => None,
                    };
                    options.effort = Some(effort.ok_or(
                        "'output_config.effort' must be low, medium, high, xhigh, or max",
                    )?);
                }
            }
        }
        ChatApi::Responses => {
            if let Some(reasoning) = body.find("reasoning").filter(|r| !r.is_null()) {
                if !reasoning.is_object() {
                    return Err("'reasoning' must be an object".into());
                }
                // Gufo takes `effort` and accepts any `summary` (local
                // reasoning is always returned); other members are refused.
                if compat == ApiCompat::Gufo
                    && let Some((key, _)) = reasoning
                        .members()
                        .iter()
                        .find(|(key, _)| !matches!(key.as_str(), "effort" | "summary"))
                {
                    return Err(format!("unsupported reasoning member: {key}"));
                }
                if let Some(effort) = reasoning.find("effort").filter(|e| !e.is_null()) {
                    let parsed = match effort {
                        Value::String(effort)
                            if effort == "none"
                                || (compat == ApiCompat::Gufo && effort == "off") =>
                        {
                            None
                        }
                        Value::String(effort) => Some(parse_reasoning_effort_name(effort).ok_or(
                            "'reasoning.effort' must be none, minimal, low, medium, high, xhigh, \
                             or max",
                        )?),
                        _ => return Err("'reasoning.effort' must be a string".into()),
                    };
                    options.enabled = Some(parsed.is_some());
                    options.effort = parsed;
                }
                if let Some(summary) = reasoning
                    .find("summary")
                    .filter(|s| compat == ApiCompat::LlamaCpp && !s.is_null())
                {
                    match summary {
                        Value::String(summary)
                            if matches!(summary.as_str(), "auto" | "concise" | "detailed") => {}
                        _ => {
                            return Err(
                                "'reasoning.summary' must be auto, concise, or detailed".into()
                            );
                        }
                    }
                }
            }
        }
    }
    if api == ChatApi::Responses {
        parse_reasoning_budget(body, &mut options)?;
    }
    if let Some(kwargs) = body.find("chat_template_kwargs") {
        let controls = Value::object().with("chat_template_kwargs", kwargs.clone());
        parse_reasoning_options(&controls, &mut options)?;
    }
    Ok(options)
}

/// Request reasoning controls over the server's thinking defaults.
pub(super) fn with_reasoning_defaults(
    reasoning: ReasoningOptions,
    backend: &Arc<dyn TextGenerationBackend>,
) -> ReasoningOptions {
    let defaults = backend.reasoning_defaults();
    ReasoningOptions {
        enabled: reasoning.enabled.or(defaults.enabled),
        effort: reasoning.effort.or(defaults.effort),
        preserve_thinking: reasoning.preserve_thinking.or(defaults.preserve_thinking),
        budget_tokens: reasoning.budget_tokens.or(defaults.budget_tokens),
    }
}

pub(super) fn invalid_compatibility_request(message: &str) -> HttpResponse {
    err(
        400,
        "Bad Request",
        message,
        "invalid_request_error",
        "invalid_request",
    )
}

/// Compatibility routes implement a synchronous text subset; options are
/// validated before dispatch so a client never gets an answer to another
/// request. The error is the response (a cold path, not worth boxing).
#[allow(clippy::result_large_err)]
pub(super) fn read_compatibility_options(
    body: &Value,
    backend: &Arc<dyn TextGenerationBackend>,
    token_field: &str,
    accepted: &[&str],
    thinking: Option<bool>,
) -> std::result::Result<(usize, SamplingConfig), HttpResponse> {
    if !body.is_object() {
        return Err(invalid_compatibility_request(
            "request body must be an object",
        ));
    }
    if let Some(model) = body.find("model") {
        let Value::String(model) = model else {
            return Err(invalid_compatibility_request("'model' must be a string"));
        };
        if *model != backend.model_id() {
            return Err(err(
                404,
                "Not Found",
                "requested model is not loaded",
                "invalid_request_error",
                "model_not_found",
            ));
        }
    }
    for field in ["stream", "echo", "store", "background"] {
        let Some(value) = body.find(field) else {
            continue;
        };
        if accepted.contains(&field) {
            // As on Chat Completions, null leaves the option off.
            if !value.is_null() && !value.is_bool() {
                return Err(invalid_compatibility_request(&format!(
                    "'{field}' must be a boolean"
                )));
            }
            continue;
        }
        if !value.is_bool() || value.as_bool(false) {
            return Err(invalid_compatibility_request(&format!(
                "'{field}' must be false"
            )));
        }
    }
    for field in ["n", "best_of"] {
        if let Some(value) = body.find(field)
            && (!value.is_number() || value.as_double(0.0) != 1.0)
        {
            return Err(invalid_compatibility_request(&format!(
                "only '{field}=1' is supported"
            )));
        }
    }
    for field in [
        "stream_options",
        "ignore_eos",
        "stop",
        "stop_sequences",
        "logprobs",
        "top_logprobs",
        "suffix",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "output_config",
        "text",
        "reasoning",
        "reasoning_effort",
        "thinking",
        "chat_template_kwargs",
        "previous_response_id",
        "conversation",
        "include",
        "truncation",
        "modalities",
        "audio",
    ] {
        if body.contains(field) && !accepted.contains(&field) {
            return Err(invalid_compatibility_request(&format!(
                "request field '{field}' is not supported on this endpoint"
            )));
        }
    }
    for field in [
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "n_predict",
    ] {
        if field != token_field && body.contains(field) {
            return Err(invalid_compatibility_request(&format!(
                "use '{token_field}' on this endpoint"
            )));
        }
    }
    let defaults = backend.sampling_defaults();
    let mut max_tokens = defaults.max_tokens;
    read_integer(body, token_field, 1, u64::from(u32::MAX), &mut max_tokens)
        .map_err(|e| invalid_compatibility_request(&e.message))?;
    let sampling = parse_sampling_config(body, &defaults.resolve(thinking)).map_err(|e| {
        err(
            400,
            "Bad Request",
            &e.message,
            "invalid_request_error",
            &e.code,
        )
    })?;
    Ok((max_tokens, sampling))
}

/// `cache_prompt` of a raw completion: `false` skips the cache lookup for
/// this request; absent or null keeps the default, `true`.
#[allow(clippy::result_large_err)]
pub(super) fn read_cache_prompt(body: &Value) -> std::result::Result<bool, HttpResponse> {
    match body.find("cache_prompt") {
        None | Some(Value::Null) => Ok(true),
        Some(Value::Bool(cache_prompt)) => Ok(*cache_prompt),
        Some(_) => Err(invalid_compatibility_request(
            "'cache_prompt' must be a boolean",
        )),
    }
}

#[allow(clippy::result_large_err)]
pub(super) fn parse_body(request: &HttpRequest) -> std::result::Result<Value, HttpResponse> {
    json::parse(&request.body).map_err(|e| {
        err(
            400,
            "Bad Request",
            e.message(),
            "invalid_request_error",
            "parse_error",
        )
    })
}

/// Maps `length_error` and `invalid_argument` of a compatibility route to
/// 400; other failures propagate to the connection.
pub(super) fn compatibility_errors(result: Result<HttpResponse>) -> Result<HttpResponse> {
    match result {
        Err(error) if error.kind() == TextErrorKind::Length => Ok(err(
            400,
            "Bad Request",
            error.message(),
            "invalid_request_error",
            "context_length_exceeded",
        )),
        Err(error) if error.kind() == TextErrorKind::InvalidArgument => Ok(err(
            400,
            "Bad Request",
            error.message(),
            "invalid_request_error",
            "invalid_prompt",
        )),
        other => other,
    }
}
