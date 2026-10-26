//! Chat request parsing: roles, content parts, historical and Responses
//! function calls, tools, `tool_choice` and the reasoning controls.

use super::*;

pub(super) fn is_known_role(role: &str) -> bool {
    matches!(role, "system" | "developer" | "user" | "assistant" | "tool")
}

pub(super) fn parse_role(role: &str) -> ChatRole {
    match role {
        "system" => ChatRole::System,
        "developer" => ChatRole::Developer,
        "assistant" => ChatRole::Assistant,
        "tool" => ChatRole::Tool,
        _ => ChatRole::User,
    }
}

pub(super) fn parse_content(
    content: Option<&Value>,
    message: &mut ChatMessage,
    budget: &mut ImageReadBudget,
) -> std::result::Result<(), String> {
    let parts = match content {
        None | Some(Value::Null) => return Ok(()),
        Some(Value::String(text)) => {
            message.content = text.clone();
            return Ok(());
        }
        Some(Value::Array(parts)) => parts,
        Some(_) => {
            return Err("message content must be a string, null, or content-part array".into());
        }
    };
    for part in parts {
        if !part.is_object() {
            return Err("message content parts must be objects".into());
        }
        let kind = part.member_str("type", "text");
        if kind == "image_url" {
            let image = part.find("image_url");
            let url = image.filter(|i| i.is_object()).and_then(|i| i.find("url"));
            // Tool results keep their images too.
            let url = match url {
                Some(Value::String(url))
                    if matches!(message.role, ChatRole::User | ChatRole::Tool) =>
                {
                    url
                }
                _ => {
                    return Err("image_url requires a user or tool message and a string URL".into());
                }
            };
            // Resolution is model-owned. The standard detail values all
            // behave as 'auto', since clients such as Codex always send one;
            // anything else is still a malformed request.
            if let Some(detail) = image.and_then(|i| i.find("detail"))
                && (!detail.is_string()
                    || !matches!(detail.str(), "auto" | "low" | "high" | "original"))
            {
                return Err("image_url.detail must be 'auto', 'low', 'high' or 'original'".into());
            }
            let bytes = read_image_url(url, budget).map_err(|e| e.message().to_owned())?;
            message.images.push(gufo_core::text::ImagePart {
                offset: message.content.len(),
                bytes: Arc::new(bytes),
            });
            continue;
        }
        if kind != "text" && kind != "input_text" {
            return Err("message content parts must use text or image_url".into());
        }
        match part.find("text") {
            Some(Value::String(text)) => message.content.push_str(text),
            _ => return Err("text message content parts require a string 'text'".into()),
        }
    }
    Ok(())
}

pub(super) fn parse_arguments(
    arguments: &str,
    out: &mut Vec<ToolArgument>,
) -> std::result::Result<(), String> {
    let parsed = json::parse_str(arguments)
        .map_err(|e| format!("tool arguments are not valid JSON: {}", e.message()))?;
    if !parsed.is_object() {
        return Err("tool arguments must encode a JSON object".into());
    }
    // Typed values render as the reference templates' `tojson` does (and
    // llama.cpp's Jinja runtime), so a replayed call matches the tokens the
    // model generated and its continuation checkpoint is reused.
    for (name, value) in parsed.members() {
        out.push(ToolArgument {
            name: name.clone(),
            value: match value {
                Value::String(s) => s.clone(),
                other => other.tojson().map_err(|e| e.message().to_owned())?,
            },
            is_string: value.is_string(),
        });
    }
    Ok(())
}

/// A replayed call (`ParseHistoricalFunction`). History records what the
/// model wrote rather than declaring a tool it may call now, so any
/// non-empty name passes, as llama.cpp keeps them; only declared tools
/// follow the name rule.
pub(super) fn parse_historical_function(
    function: &Value,
    call: &mut ToolCall,
) -> std::result::Result<(), String> {
    let (Some(Value::String(name)), Some(Value::String(arguments))) =
        (function.find("name"), function.find("arguments"))
    else {
        return Err(HISTORICAL_FUNCTION_RULE.into());
    };
    if name.is_empty() {
        return Err(HISTORICAL_FUNCTION_RULE.into());
    }
    if name.contains('\0') {
        return Err("historical function names cannot contain NUL".into());
    }
    call.name = name.clone();
    parse_arguments(arguments, &mut call.arguments)
}

/// A Responses `function_call` or `custom_tool_call` input item: a call the
/// model made earlier, held to the same rules as a replayed Chat call. A
/// custom call's free-form `input` stays literal argument data in the
/// model's native tool history; this does not enable
/// custom tool generation.
pub(crate) fn parse_response_function_call(item: &Value) -> std::result::Result<ToolCall, String> {
    let id = item.member_str("call_id", "");
    if id.is_empty() {
        return Err("function items require a nonempty call_id".into());
    }
    let mut call = ToolCall {
        id,
        ..ToolCall::default()
    };
    if item.member_str("type", "") == "custom_tool_call" {
        let Some(input @ Value::String(_)) = item.find("input") else {
            return Err("historical custom tool calls require string input".into());
        };
        let arguments = Value::object().with("input", input.clone());
        let function = item
            .clone()
            .with("arguments", arguments.dump().unwrap_or_default());
        parse_historical_function(&function, &mut call)?;
    } else {
        parse_historical_function(item, &mut call)?;
    }
    Ok(call)
}

/// A Responses `function_call_output` or `custom_tool_call_output` input
/// item as a tool message: text, or an array of `input_text` and
/// `input_image` parts whose images stay in the tool result.
pub(crate) fn parse_response_function_output(
    item: &Value,
    budget: &mut ImageReadBudget,
) -> std::result::Result<ChatMessage, String> {
    let id = item.member_str("call_id", "");
    if id.is_empty() {
        return Err("function items require a nonempty call_id".into());
    }
    let output = match item.find("output") {
        Some(output @ (Value::String(_) | Value::Array(_))) => output.clone(),
        _ => return Err("function output requires a string or content array".into()),
    };
    let converted = Value::object()
        .with("role", "tool")
        .with("tool_call_id", id)
        .with("content", output);
    let mut message = ChatMessage::default();
    parse_openai_response_message(&converted, &mut message, budget)?;
    Ok(message)
}

pub(super) const HISTORICAL_FUNCTION_RULE: &str =
    "historical function calls require a non-empty name and string arguments";

pub(super) fn parse_message(
    value: &Value,
    message: &mut ChatMessage,
    budget: &mut ImageReadBudget,
) -> std::result::Result<(), String> {
    if !value.is_object() {
        return Err("each message must be an object".into());
    }
    let role = value.member_str("role", "");
    if !is_known_role(&role) {
        return Err("message role must be system, developer, user, assistant, or tool".into());
    }
    message.role = parse_role(&role);
    message.name = value.member_str("name", "");
    message.tool_call_id = value.member_str("tool_call_id", "");
    parse_content(value.find("content"), message, budget)?;
    if let Some(reasoning) = value.find("reasoning_content").filter(|r| !r.is_null()) {
        let Value::String(reasoning) = reasoning else {
            return Err(
                "'reasoning_content' is only valid as a string on assistant messages".into(),
            );
        };
        if message.role != ChatRole::Assistant {
            return Err(
                "'reasoning_content' is only valid as a string on assistant messages".into(),
            );
        }
        message.thought = reasoning.clone();
    }
    let Some(tool_calls) = value.find("tool_calls") else {
        return Ok(());
    };
    if message.role != ChatRole::Assistant || !tool_calls.is_array() {
        return Err("'tool_calls' is only valid as an array on assistant messages".into());
    }
    for item in tool_calls.items() {
        if !item.is_object() || item.member_str("type", "function") != "function" {
            return Err("only function tool calls are supported".into());
        }
        let function = match item.find("function") {
            Some(function) if function.is_object() => function,
            _ => return Err("assistant tool calls require a function object".into()),
        };
        let mut call = ToolCall {
            id: item.member_str("id", ""),
            name: String::new(),
            arguments: Vec::new(),
        };
        parse_historical_function(function, &mut call)?;
        message.tool_calls.push(call);
    }
    Ok(())
}

/// A tool name reaches the Qwen and DeepSeek renderers unescaped, inside
/// `<function=NAME>` and `name="NAME"`, so the characters that frame a call
/// are excluded; the dots, colons and slashes of bridged agent tool names are
/// data. Non-ASCII bytes are excluded too: names go into prompts and logs,
/// where confusable and invisible characters buy a client nothing.
pub(super) const TOOL_NAME_RULE: &str = "function names require 1-64 printable ASCII characters other than \
                              spaces, '<', '>', '\"' and '\\'";

pub(super) fn renderable_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|c| c > 0x20 && c < 0x7f && !matches!(c, b'<' | b'>' | b'"' | b'\\'))
}

/// The function tools of `tools`. With `allow_non_function` (Responses), a
/// `namespace` entry flattens into its functions and hosted tool types are
/// skipped.
pub(super) fn parse_tools(
    tools: Option<&Value>,
    output: &mut Vec<ChatTool>,
    allow_non_function: bool,
) -> std::result::Result<(), String> {
    let Some(tools) = tools.filter(|t| !t.is_null()) else {
        return Ok(());
    };
    if !tools.is_array() {
        return Err("'tools' must be an array".into());
    }
    for item in tools.items() {
        if !item.is_object() {
            return Err("'tools' entries must be objects".into());
        }
        let kind = item.member_str("type", "");
        if allow_non_function && kind == "namespace" {
            // A namespace only groups client-executed function tools; calls
            // replay by the plain function name. The uniqueness check below
            // rejects ambiguous namespaces.
            let nested = item
                .find("tools")
                .filter(|nested| nested.is_array())
                .ok_or("namespace tools require a tools array")?;
            parse_tools(Some(nested), output, allow_non_function)?;
            continue;
        }
        if kind != "function" {
            // Hosted tool types (web_search, file_search, code_interpreter,
            // mcp, ...) only the provider can execute; Chat Completions
            // declares only functions.
            if allow_non_function {
                continue;
            }
            return Err("only function tools are supported".into());
        }
        let function = item.find("function");
        if function.is_some_and(|f| !f.is_object()) {
            return Err("'function' must be an object".into());
        }
        let source = function.unwrap_or(item);
        if source
            .find("strict")
            .is_some_and(|strict| !strict.is_null() && !matches!(strict, Value::Bool(_)))
        {
            return Err("function strict must be a boolean or null".into());
        }
        // OpenAI uses "parameters"; some agent clients send
        // parametersJsonSchema.
        let parameters = source
            .find("parameters")
            .filter(|p| !p.is_null())
            .or_else(|| source.find("parametersJsonSchema"));
        let name = source.member_str("name", "");
        let description = source.member_str("description", "");
        if !renderable_tool_name(&name) {
            return Err(TOOL_NAME_RULE.into());
        }
        if output.iter().any(|previous| previous.name == name) {
            return Err("function names must be unique".into());
        }
        if parameters.is_some_and(|p| !p.is_null() && !p.is_object()) {
            return Err("function tools require an object parameters schema".into());
        }
        // Preserve nested definitions and their field order; flat tools move
        // the complete function body (including strict) under "function".
        let mut function_object = Value::object();
        for (key, value) in source.members() {
            if key == "parametersJsonSchema" || (function.is_none() && key == "type") {
                continue;
            }
            function_object.append_member(key.clone(), value.clone());
        }
        function_object.set(
            "parameters",
            match parameters {
                Some(p) if p.is_object() => p.clone(),
                _ => Value::object(),
            },
        );
        if function_object
            .find("strict")
            .is_some_and(|strict| strict.as_bool(false))
        {
            // Omitting parameters defines a function with no arguments.
            if parameters.is_none_or(Value::is_null) {
                function_object.set(
                    "parameters",
                    json::parse_str(
                        r#"{"type":"object","properties":{},"required":[],"additionalProperties":false}"#,
                    )
                    .map_err(|e| e.message().to_owned())?,
                );
            }
            if let Some(schema) = function_object.find("parameters") {
                JsonConstraint::compile(schema, true).map_err(|e| e.message().to_owned())?;
            }
        }
        let parameters_json = function_object
            .find("parameters")
            .map(|p| p.dump())
            .transpose()
            .map_err(|e| e.message().to_owned())?
            .unwrap_or_default();
        let mut definition = if function.is_some() {
            item.clone()
        } else {
            Value::object()
        };
        definition.set("type", "function");
        definition.set("function", function_object);
        // Counted per flattened function, whatever the namespace order.
        if output.len() >= 128 {
            return Err("'tools' supports at most 128 functions".into());
        }
        output.push(ChatTool {
            name,
            description,
            parameters_json,
            definition_json: definition.dump().map_err(|e| e.message().to_owned())?,
        });
    }
    Ok(())
}

pub(super) fn parse_tool_choice(
    value: Option<&Value>,
    chat: &mut ChatRequest,
) -> std::result::Result<(), String> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(());
    };
    if value.is_string() {
        chat.forced_tool_name.clear();
    }
    chat.tool_choice = match value {
        Value::String(choice) if choice == "auto" => ToolChoice::Auto,
        Value::String(choice) if choice == "none" => ToolChoice::None,
        Value::String(choice) if choice == "required" => ToolChoice::Required,
        // A named function, nested (Chat) or flat (Responses): only it is
        // offered, and it must be called.
        Value::Object(_) if value.member_str("type", "") == "function" => {
            let name = value.find("function").map_or_else(
                || value.member_str("name", ""),
                |function| function.member_str("name", ""),
            );
            let tools = &mut chat.tools;
            let Some(index) = tools.iter().position(|tool| tool.name == name) else {
                return Err("'tool_choice' must name a declared function".into());
            };
            let tool = tools.swap_remove(index);
            *tools = vec![tool];
            chat.forced_tool_name = name;
            ToolChoice::Required
        }
        _ => {
            return Err(
                "'tool_choice' must be auto, none, required, or a declared function".into(),
            );
        }
    };
    Ok(())
}

/// `tools`, `tool_choice` and `parallel_tool_calls`, shared by Chat
/// Completions and Responses (whose `parallel_tool_calls` may be null).
/// Every tool request is decoded under its grammar; a function without
/// `strict` keeps non-strict parameters, as in llama.cpp.
pub fn parse_tool_controls(
    body: &Value,
    chat: &mut ChatRequest,
    nullable_parallel: bool,
) -> std::result::Result<(), (String, &'static str)> {
    tool_controls(body, chat, nullable_parallel, false)
}

/// The Responses tool controls: hosted tool types are skipped, namespaces
/// flatten into their functions, and each namespaced function keeps its
/// namespace for its calls.
pub fn parse_response_tool_controls(
    body: &Value,
    chat: &mut ChatRequest,
) -> std::result::Result<(), (String, &'static str)> {
    tool_controls(body, chat, true, true)?;
    if let Some(tools) = body.find("tools").filter(|tools| tools.is_array()) {
        for item in tools.items() {
            let name = item.member_str("name", "");
            if item.member_str("type", "") != "namespace" || name.is_empty() {
                continue;
            }
            for tool in item.find("tools").into_iter().flat_map(Value::items) {
                if tool.member_str("type", "") == "function" {
                    chat.tool_namespaces
                        .insert(tool.member_str("name", ""), name.clone());
                }
            }
        }
    }
    Ok(())
}

pub(super) fn tool_controls(
    body: &Value,
    chat: &mut ChatRequest,
    nullable_parallel: bool,
    allow_non_function: bool,
) -> std::result::Result<(), (String, &'static str)> {
    parse_tools(body.find("tools"), &mut chat.tools, allow_non_function)
        .and_then(|()| parse_tool_choice(body.find("tool_choice"), chat))
        .map_err(|message| (message, "invalid_tools"))?;
    if chat.tool_choice == ToolChoice::Required && chat.tools.is_empty() {
        return Err((
            "'tool_choice' cannot be required without tools".into(),
            "invalid_tool_choice",
        ));
    }
    match body.find("parallel_tool_calls") {
        None => {}
        Some(Value::Null) if nullable_parallel => {}
        Some(Value::Bool(parallel)) => chat.parallel_tool_calls = *parallel,
        Some(_) => {
            return Err((
                "'parallel_tool_calls' must be a boolean".into(),
                "invalid_tools",
            ));
        }
    }
    // A forced function executes exactly once.
    if body.find("tool_choice").is_some_and(Value::is_object) {
        chat.parallel_tool_calls = false;
    }
    // Every tool request is decoded under its grammar.
    chat.constrained_tools = !chat.tools.is_empty() && chat.tool_choice != ToolChoice::None;
    Ok(())
}

pub(crate) fn parse_reasoning_effort_name(value: &str) -> Option<ReasoningEffort> {
    Some(match value {
        "minimal" => ReasoningEffort::Minimal,
        "low" => ReasoningEffort::Low,
        "medium" => ReasoningEffort::Medium,
        "high" => ReasoningEffort::High,
        "xhigh" => ReasoningEffort::XHigh,
        "max" => ReasoningEffort::Max,
        _ => return None,
    })
}

pub(super) fn assign_reasoning_enabled(
    options: &mut ReasoningOptions,
    enabled: bool,
) -> std::result::Result<(), String> {
    if options.enabled.is_some_and(|e| e != enabled) {
        return Err("reasoning controls disagree about whether thinking is enabled".into());
    }
    options.enabled = Some(enabled);
    Ok(())
}

pub(super) fn assign_reasoning_effort(
    options: &mut ReasoningOptions,
    value: &str,
    enable_thinking: bool,
) -> std::result::Result<(), String> {
    if value == "off" || value == "none" {
        return assign_reasoning_enabled(options, false);
    }
    let Some(effort) = parse_reasoning_effort_name(value) else {
        return Err(
            "reasoning_effort must be none, minimal, low, medium, high, xhigh, or max".into(),
        );
    };
    if options.effort.is_some_and(|e| e != effort) {
        return Err("top-level and chat_template_kwargs reasoning_effort disagree".into());
    }
    options.effort = Some(effort);
    if enable_thinking {
        assign_reasoning_enabled(options, true)?;
    }
    Ok(())
}

/// llama.cpp's `reasoning_budget_tokens`, or its older spelling
/// `thinking_budget_tokens` (`ParseReasoningBudget`):
/// a non-null canonical value takes precedence, and -1 keeps the server
/// default.
pub(crate) fn parse_reasoning_budget(
    body: &Value,
    options: &mut ReasoningOptions,
) -> std::result::Result<(), String> {
    let present = |field: &'static str| {
        body.find(field)
            .filter(|value| !value.is_null())
            .map(|value| (field, value))
    };
    let Some((field, budget)) =
        present("reasoning_budget_tokens").or_else(|| present("thinking_budget_tokens"))
    else {
        return Ok(());
    };
    let value = if budget.is_number() {
        budget.as_double(-2.0)
    } else {
        -2.0
    };
    if !value.is_finite() || value < -1.0 || value > f64::from(i32::MAX) || value.floor() != value {
        return Err(format!(
            "'{field}' must be an integer between -1 and 2147483647"
        ));
    }
    if value >= 0.0 {
        options.budget_tokens = Some(value as usize);
    }
    Ok(())
}

pub(crate) fn parse_reasoning_options(
    body: &Value,
    options: &mut ReasoningOptions,
) -> std::result::Result<(), String> {
    if let Some(thinking) = body.find("thinking") {
        if !thinking.is_object() {
            return Err("'thinking' must be an object".into());
        }
        let value = match thinking.find("type") {
            Some(Value::String(value)) if value == "enabled" || value == "disabled" => value,
            _ => return Err("'thinking.type' must be enabled or disabled".into()),
        };
        assign_reasoning_enabled(options, value == "enabled")?;
    }
    if let Some(effort) = body.find("reasoning_effort").filter(|e| !e.is_null()) {
        let Value::String(effort) = effort else {
            return Err("'reasoning_effort' must be a string".into());
        };
        assign_reasoning_effort(options, effort, true)?;
    }
    let Some(kwargs) = body.find("chat_template_kwargs") else {
        return Ok(());
    };
    if !kwargs.is_object() {
        return Err("'chat_template_kwargs' must be an object".into());
    }
    if let Some(enabled) = kwargs.find("enable_thinking") {
        let Value::Bool(enabled) = enabled else {
            return Err("'chat_template_kwargs.enable_thinking' must be a boolean".into());
        };
        assign_reasoning_enabled(options, *enabled)?;
    }
    if let Some(mode) = kwargs.find("thinking_mode") {
        let Value::String(value) = mode else {
            return Err("'chat_template_kwargs.thinking_mode' must be a string".into());
        };
        if value != "auto" {
            let enabled = value == "thinking" || value == "on";
            if !enabled && !matches!(value.as_str(), "chat" | "off" | "none") {
                return Err(
                    "'chat_template_kwargs.thinking_mode' must be auto, thinking, or chat".into(),
                );
            }
            assign_reasoning_enabled(options, enabled)?;
        }
    }
    if let Some(effort) = kwargs.find("reasoning_effort") {
        let Value::String(effort) = effort else {
            return Err("'chat_template_kwargs.reasoning_effort' must be a string".into());
        };
        let enable = options.enabled.unwrap_or(true);
        assign_reasoning_effort(options, effort, enable)?;
    }
    if let Some(preserve) = kwargs.find("preserve_thinking") {
        let Value::Bool(preserve) = preserve else {
            return Err("'chat_template_kwargs.preserve_thinking' must be a boolean".into());
        };
        options.preserve_thinking = Some(*preserve);
    }
    Ok(())
}

pub(super) fn parse_request(
    request: &HttpRequest,
    backend: &Arc<dyn TextGenerationBackend>,
    output: &mut ParsedChatRequest,
) -> Option<HttpResponse> {
    let body = match json::parse(&request.body) {
        Ok(body) => body,
        Err(e) => return Some(error(400, "Bad Request", e.message(), "parse_error")),
    };
    if !body.is_object() {
        return Some(error(
            400,
            "Bad Request",
            "request body must be a JSON object",
            "invalid_body",
        ));
    }
    if body
        .find("model")
        .is_some_and(|model| !model.is_null() && !model.is_string())
    {
        return Some(error(
            400,
            "Bad Request",
            "'model' must be a string",
            "invalid_model",
        ));
    }
    output.model = body.member_str("model", "");
    if output.model.is_empty() {
        return Some(error(
            400,
            "Bad Request",
            "'model' is required",
            "missing_model",
        ));
    }
    if output.model != backend.model_id() {
        return Some(error(
            404,
            "Not Found",
            &format!("model '{}' is not served by this process", output.model),
            "model_not_found",
        ));
    }
    let format = body.find("response_format");
    match parse_response_format(format, false) {
        Ok(grammar) => {
            if grammar.is_some() {
                output.chat.response_format_description =
                    response_format_description(format, false);
            }
            output.chat.response_format = grammar.map(OutputGrammar);
        }
        Err(message) => {
            return Some(error(
                400,
                "Bad Request",
                &message,
                "invalid_response_format",
            ));
        }
    }
    output.chat.client_id = request.client_id.clone();
    // As in llama-server, null keeps the default.
    if let Some(cache_prompt) = body.find("cache_prompt").filter(|value| !value.is_null()) {
        let Value::Bool(cache_prompt) = cache_prompt else {
            return Some(error(
                400,
                "Bad Request",
                "'cache_prompt' must be a boolean",
                "invalid_cache_prompt",
            ));
        };
        output.chat.cache_prompt = *cache_prompt;
    }
    match body.find("return_progress") {
        None | Some(Value::Null) => {}
        Some(Value::Bool(return_progress)) => output.chat.return_progress = *return_progress,
        Some(_) => {
            return Some(error(
                400,
                "Bad Request",
                "'return_progress' must be a boolean",
                "invalid_return_progress",
            ));
        }
    }
    match parse_stop_sequences(body.find("stop"), StopSequenceFormat::OpenAi) {
        Ok(sequences) => output.chat.stop_sequences = sequences,
        Err(message) => return Some(error(400, "Bad Request", &message, "invalid_stop")),
    }
    let messages = match body.find("messages") {
        Some(Value::Array(messages)) if !messages.is_empty() => messages,
        _ => {
            return Some(error(
                400,
                "Bad Request",
                "'messages' must be a non-empty array",
                "missing_messages",
            ));
        }
    };
    let mut budget = ImageReadBudget {
        accepts_images: backend.supports_images(),
        ..ImageReadBudget::default()
    };
    for item in messages {
        let mut message = ChatMessage::default();
        if let Err(message) = parse_message(item, &mut message, &mut budget) {
            return Some(error(400, "Bad Request", &message, "invalid_messages"));
        }
        output.chat.messages.push(message);
    }
    if let Err((message, code)) = parse_tool_controls(&body, &mut output.chat, false) {
        return Some(error(400, "Bad Request", &message, code));
    }
    if let Err(message) = parse_reasoning_options(&body, &mut output.chat.reasoning)
        .and_then(|()| parse_reasoning_budget(&body, &mut output.chat.reasoning))
    {
        return Some(error(400, "Bad Request", &message, "invalid_reasoning"));
    }
    if let Some(vision_id) = body
        .find("chat_template_kwargs")
        .and_then(|k| k.find("add_vision_id"))
    {
        let Value::Bool(vision_id) = vision_id else {
            return Some(error(
                400,
                "Bad Request",
                "'chat_template_kwargs.add_vision_id' must be a boolean",
                "invalid_template_options",
            ));
        };
        output.chat.add_vision_id = *vision_id;
    }
    let defaults = backend.reasoning_defaults();
    let reasoning = &mut output.chat.reasoning;
    reasoning.enabled = reasoning.enabled.or(defaults.enabled);
    reasoning.effort = reasoning.effort.or(defaults.effort);
    reasoning.preserve_thinking = reasoning.preserve_thinking.or(defaults.preserve_thinking);
    reasoning.budget_tokens = reasoning.budget_tokens.or(defaults.budget_tokens);

    // Null OpenAI fields keep the server defaults.
    let present = |field: &str| body.find(field).filter(|value| !value.is_null());
    if let Some(stream) = present("stream") {
        let Value::Bool(stream) = stream else {
            return Some(error(
                400,
                "Bad Request",
                "'stream' must be a boolean",
                "invalid_stream",
            ));
        };
        output.stream = *stream;
    }
    if let Some(options) = present("stream_options") {
        if !options.is_object() {
            return Some(error(
                400,
                "Bad Request",
                "'stream_options' must be an object",
                "invalid_stream_options",
            ));
        }
        if let Some(include_usage) = options.find("include_usage") {
            let Value::Bool(include_usage) = include_usage else {
                return Some(error(
                    400,
                    "Bad Request",
                    "'stream_options.include_usage' must be a boolean",
                    "invalid_stream_options",
                ));
            };
            output.include_usage = *include_usage;
        }
    }
    let max_tokens = present("max_completion_tokens")
        .map(|value| ("max_completion_tokens", value))
        .or_else(|| present("max_tokens").map(|value| ("max_tokens", value)));
    if let Some((field, max_tokens)) = max_tokens {
        let value = max_tokens.as_double(0.0);
        if !max_tokens.is_number()
            || !value.is_finite()
            || value.floor() != value
            || value < 1.0
            || value > f64::from(u32::MAX)
        {
            return Some(error(
                400,
                "Bad Request",
                &format!("'{field}' must be a positive integer"),
                "invalid_max_tokens",
            ));
        }
        output.max_tokens = max_tokens.as_size(0);
    }
    // Unset fields follow the model's recommendation for the request's
    // effective thinking mode.
    let base = backend
        .sampling_defaults()
        .resolve(output.chat.reasoning.enabled);
    let sampling = match parse_sampling_config(&body, &base) {
        Ok(sampling) => sampling,
        Err(e) => return Some(error(400, "Bad Request", &e.message, &e.code)),
    };
    output.sampling = sampling;
    if let Some(choices) = present("n")
        && (!choices.is_number() || choices.as_double(0.0) != 1.0)
    {
        return Some(error(
            400,
            "Bad Request",
            "only n=1 is supported",
            "unsupported_n",
        ));
    }
    // Raw Completions owns the fixed-length benchmark contract; refusing the
    // field keeps a harness from measuring silently shortened runs.
    if present("ignore_eos").is_some() {
        return Some(error(
            400,
            "Bad Request",
            "request field 'ignore_eos' is not supported on this endpoint",
            "unsupported_field",
        ));
    }
    for unsupported in ["logprobs", "top_logprobs", "modalities", "audio"] {
        let Some(value) = present(unsupported) else {
            continue;
        };
        // The values that ask for what the server does anyway.
        let default = match unsupported {
            "logprobs" => matches!(value, Value::Bool(false)),
            "modalities" => {
                value.is_array()
                    && value.len() == 1
                    && value.items()[0].is_string()
                    && value.items()[0].str() == "text"
            }
            _ => false,
        };
        if !default {
            return Some(error(
                400,
                "Bad Request",
                &format!("request field '{unsupported}' is not implemented"),
                "unsupported_field",
            ));
        }
    }
    None
}
