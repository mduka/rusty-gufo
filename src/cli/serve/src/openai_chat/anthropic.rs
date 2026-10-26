//! The Anthropic Messages API: its tool controls, images and tool messages
//! in requests, and its output events.

use super::*;

/// Messages `tools` and `tool_choice` as the Chat tool declarations, so
/// validation, schema constraints and tool framing stay on the one Chat
/// tool path (`ParseAnthropicToolControls`). Only
/// custom tools with an `input_schema` are supported.
pub(crate) fn parse_anthropic_tool_controls(
    body: &Value,
    chat: &mut ChatRequest,
) -> std::result::Result<(), (String, &'static str)> {
    let tools = body.find("tools");
    let choice = body.find("tool_choice");
    if tools.is_none() && choice.is_none() {
        return Ok(());
    }
    let mut converted = Value::object();
    if let Some(tools) = tools {
        let Value::Array(tools) = tools else {
            return Err(("'tools' must be an array".into(), "invalid_tools"));
        };
        let mut functions = Vec::with_capacity(tools.len());
        for tool in tools {
            let Some(schema) = tool
                .find("input_schema")
                .filter(|_| tool.is_object() && tool.member_str("type", "custom") == "custom")
            else {
                return Err((
                    "only custom tools with an input_schema are supported".into(),
                    "invalid_tools",
                ));
            };
            let mut function = Value::object();
            for key in ["name", "description"] {
                if let Some(value) = tool.find(key) {
                    function.set(key, value.clone());
                }
            }
            function.set("parameters", schema.clone());
            functions.push(
                Value::object()
                    .with("type", "function")
                    .with("function", function),
            );
        }
        converted.set("tools", functions);
    }
    if let Some(choice) = choice {
        let kind = if choice.is_object() {
            choice.member_str("type", "")
        } else {
            String::new()
        };
        match kind.as_str() {
            "auto" | "none" => converted.set("tool_choice", kind.as_str()),
            "any" => converted.set("tool_choice", "required"),
            "tool" => {
                let mut function = Value::object();
                if let Some(name) = choice.find("name") {
                    function.set("name", name.clone());
                }
                converted.set(
                    "tool_choice",
                    Value::object()
                        .with("type", "function")
                        .with("function", function),
                );
            }
            _ => {
                return Err((
                    "'tool_choice.type' must be auto, any, tool or none".into(),
                    "invalid_tool_choice",
                ));
            }
        }
        if let Some(single) = choice.find("disable_parallel_tool_use") {
            let Value::Bool(single) = single else {
                return Err((
                    "'disable_parallel_tool_use' must be a boolean".into(),
                    "invalid_tools",
                ));
            };
            if kind != "tool" {
                converted.set("parallel_tool_calls", !*single);
            }
        }
    }
    parse_tool_controls(&converted, chat, false)
}

/// One Messages image block, with a `base64` or HTTPS `url` source, at the
/// current end of `message`'s text (`ParseAnthropicImage`).
pub(crate) fn parse_anthropic_image(
    block: &Value,
    message: &mut ChatMessage,
    budget: &mut ImageReadBudget,
) -> std::result::Result<(), String> {
    let source = block.find("source").filter(|source| source.is_object());
    let url = match source
        .map(|source| source.member_str("type", ""))
        .as_deref()
    {
        Some("base64") => {
            let source = source.expect("an object source");
            let (Some(Value::String(media)), Some(Value::String(data))) =
                (source.find("media_type"), source.find("data"))
            else {
                return Err("base64 image sources require string media_type and data".into());
            };
            format!("data:{media};base64,{data}")
        }
        Some("url") => {
            let Some(Value::String(url)) = source.and_then(|source| source.find("url")) else {
                return Err("url image sources require a string url".into());
            };
            url.clone()
        }
        _ => return Err("image blocks require a base64 or url source".into()),
    };
    let bytes = read_image_url(&url, budget).map_err(|e| e.message().to_owned())?;
    message.images.push(gufo_core::text::ImagePart {
        offset: message.content.len(),
        bytes: Arc::new(bytes),
    });
    Ok(())
}

/// One Messages turn with `tool_use` or `tool_result` blocks
/// (`ParseAnthropicToolMessage`). An
/// assistant turn becomes one message with its thinking, text and calls;
/// in a user turn each `tool_result` becomes a tool message with its text
/// and images, and the other blocks follow as user messages.
pub(crate) fn parse_anthropic_tool_message(
    item: &Value,
    messages: &mut Vec<ChatMessage>,
    budget: &mut ImageReadBudget,
) -> std::result::Result<(), String> {
    let role = item.member_str("role", "");
    let content = match item.find("content") {
        Some(Value::Array(content))
            if item.is_object() && matches!(role.as_str(), "user" | "assistant") =>
        {
            content
        }
        _ => return Err("tool blocks require a user or assistant content array".into()),
    };
    let read_text = |part: &Value, out: &mut String| match part.find("text") {
        Some(Value::String(text)) if part.member_str("type", "") == "text" => {
            out.push_str(text);
            Ok(())
        }
        _ => Err("unsupported Messages content block".to_owned()),
    };
    if role == "assistant" {
        let mut message = ChatMessage::new(ChatRole::Assistant, "");
        for part in content {
            let kind = if part.is_object() {
                part.member_str("type", "")
            } else {
                String::new()
            };
            match kind.as_str() {
                "thinking" => {
                    let Some(Value::String(thinking)) = part.find("thinking") else {
                        return Err("thinking blocks require thinking text".into());
                    };
                    message.thought.push_str(thinking);
                }
                // As in a text-only turn, redacted thinking has no text.
                "redacted_thinking" => {}
                "tool_use" => {
                    let id = part.member_str("id", "");
                    let input = part.find("input").filter(|input| input.is_object());
                    let Some(input) = input.filter(|_| !id.is_empty()) else {
                        return Err(
                            "tool_use blocks require a non-empty id and an object input".into()
                        );
                    };
                    let mut function =
                        Value::object().with("arguments", input.dump().unwrap_or_default());
                    if let Some(name) = part.find("name") {
                        function.set("name", name.clone());
                    }
                    let mut call = ToolCall {
                        id,
                        ..ToolCall::default()
                    };
                    parse_historical_function(&function, &mut call)?;
                    message.tool_calls.push(call);
                }
                _ => read_text(part, &mut message.content)?,
            }
        }
        messages.push(message);
        return Ok(());
    }
    // User text and tool results may carry images too.
    let mut read_part = |part: &Value, target: &mut ChatMessage| {
        if part.is_object() && part.member_str("type", "") == "image" {
            parse_anthropic_image(part, target, budget)
        } else {
            read_text(part, &mut target.content)
        }
    };
    // One user turn carries the results of the previous calls, then any text.
    let mut text: Option<ChatMessage> = None;
    for part in content {
        if !part.is_object() || part.member_str("type", "") != "tool_result" {
            let target = text.get_or_insert_with(|| ChatMessage::new(ChatRole::User, ""));
            read_part(part, target)?;
            continue;
        }
        messages.extend(text.take());
        let id = part.member_str("tool_use_id", "");
        if id.is_empty()
            || part
                .find("is_error")
                .is_some_and(|failed| !failed.is_bool())
        {
            return Err("tool_result blocks require a non-empty tool_use_id".into());
        }
        let mut result = ChatMessage::new(ChatRole::Tool, "");
        result.tool_call_id = id;
        match part.find("content") {
            None => {}
            Some(Value::String(output)) => result.content = output.clone(),
            Some(Value::Array(blocks)) => {
                for block in blocks {
                    read_part(block, &mut result)?;
                }
            }
            Some(_) => {
                return Err("tool_result content must be a string, text or image blocks".into());
            }
        }
        messages.push(result);
    }
    messages.extend(text);
    Ok(())
}

/// One Messages output as Anthropic content blocks (`AnthropicOutput`): streamed as `message_start`,
/// `content_block_start`/`_delta`/`_stop` and `message_delta`/`_stop`.
pub(super) struct AnthropicOutput<'w> {
    pub(super) writer: Option<&'w mut BodyWriter<'w>>,
    pub(super) message: Value,
    /// The open `tool_use` block, or null.
    pub(super) tool: Value,
    pub(super) tool_arguments: String,
    pub(super) text: String,
    /// Trailing reasoning whitespace, emitted only if more reasoning follows.
    pub(super) held: String,
    pub(super) index: usize,
    pub(super) active: bool,
    pub(super) reasoning: bool,
    pub(super) tool_use: bool,
    pub(super) connected: bool,
}

pub(super) fn anthropic_output<'w>(
    model: &str,
    writer: Option<&'w mut BodyWriter<'w>>,
    _: &ChatRequest,
) -> Box<dyn CompatibilityOutput + 'w> {
    Box::new(AnthropicOutput::new(model, writer))
}

impl<'w> AnthropicOutput<'w> {
    pub(super) fn new(model: &str, writer: Option<&'w mut BodyWriter<'w>>) -> Self {
        // Prompt accounting is final only with the result; `message_delta`
        // carries the complete usage.
        let message = Value::object()
            .with("id", random_id("msg_", 12))
            .with("type", "message")
            .with("role", "assistant")
            .with("model", model)
            .with("content", Value::array())
            .with("stop_reason", Value::Null)
            .with("stop_sequence", Value::Null)
            .with(
                "usage",
                Value::object()
                    .with("input_tokens", 0)
                    .with("output_tokens", 0),
            );
        Self {
            writer,
            message,
            tool: Value::Null,
            tool_arguments: String::new(),
            text: String::new(),
            held: String::new(),
            index: 0,
            active: false,
            reasoning: false,
            tool_use: false,
            connected: true,
        }
    }

    pub(super) fn event(kind: &str) -> Value {
        Value::object().with("type", kind)
    }

    pub(super) fn block(&self) -> Value {
        let block = Value::object()
            .with("type", if self.reasoning { "thinking" } else { "text" })
            .with(
                if self.reasoning { "thinking" } else { "text" },
                self.text.as_str(),
            );
        // Local reasoning is not signed; clients replay the block unchanged.
        if self.reasoning {
            block.with("signature", "")
        } else {
            block
        }
    }

    pub(super) fn block_start(&self, block: Value) -> Value {
        Self::event("content_block_start")
            .with("index", self.index)
            .with("content_block", block)
    }

    pub(super) fn block_delta(&self, delta: Value) -> Value {
        Self::event("content_block_delta")
            .with("index", self.index)
            .with("delta", delta)
    }

    pub(super) fn block_stop(&mut self) -> bool {
        let event = Self::event("content_block_stop").with("index", self.index);
        self.index += 1;
        self.emit(event)
    }

    pub(super) fn close_block(&mut self) -> bool {
        self.held.clear();
        if !self.active {
            return self.connected;
        }
        let block = self.block();
        self.message.member_mut("content").push(block);
        self.active = false;
        self.block_stop()
    }

    /// Closes the open `tool_use` block; only a `complete` one has its input
    /// and makes the turn a tool use.
    pub(super) fn close_tool(&mut self, complete: bool) -> bool {
        if self.tool.is_null() {
            return self.connected;
        }
        let mut tool = std::mem::replace(&mut self.tool, Value::Null);
        if complete {
            tool.set(
                "input",
                json::parse_str(&self.tool_arguments).unwrap_or_else(|_| Value::object()),
            );
            self.tool_use = true;
        }
        self.message.member_mut("content").push(tool);
        self.block_stop()
    }

    pub(super) fn emit(&mut self, event: Value) -> bool {
        let Some(writer) = self.writer.as_mut() else {
            return true;
        };
        if !self.connected {
            return false;
        }
        let mut bytes = format!("event: {}\n", event.member_str("type", "")).into_bytes();
        bytes.extend(sse(&event));
        self.connected = writer(&bytes);
        self.connected
    }
}

impl CompatibilityOutput for AnthropicOutput<'_> {
    fn begin(&mut self) -> bool {
        let event = Self::event("message_start").with("message", self.message.clone());
        self.emit(event)
    }

    fn progress(&mut self, _: &PromptProgress) -> bool {
        self.connected
    }

    fn append(&mut self, piece: &str, reasoning: bool) -> bool {
        if !piece.is_empty() && !self.close_tool(false) {
            return false;
        }
        // The preceding block's held whitespace goes before this block's
        // first piece is collected, or closing it would drop that piece's.
        if (!self.active || self.reasoning != reasoning) && !self.close_block() {
            return false;
        }
        let joined;
        let mut text = piece;
        if reasoning {
            // Thinking blocks carry the reasoning trimmed as Chat reports
            // it: leading whitespace goes, trailing whitespace waits until
            // more reasoning follows.
            if !(self.active && self.reasoning) {
                text = text.trim_start_matches(is_space);
            }
            let body = text.trim_end_matches(is_space).len();
            if body == 0 {
                self.held.push_str(text);
                return true;
            }
            joined = std::mem::take(&mut self.held) + &text[..body];
            self.held.push_str(&text[body..]);
            text = &joined;
        }
        if text.is_empty() {
            return true;
        }
        if !self.active || self.reasoning != reasoning {
            self.reasoning = reasoning;
            self.active = true;
            self.text.clear();
            let start = self.block_start(self.block());
            if !self.emit(start) {
                return false;
            }
        }
        self.text.push_str(text);
        let delta = if reasoning {
            Value::object()
                .with("type", "thinking_delta")
                .with("thinking", text)
        } else {
            Value::object()
                .with("type", "text_delta")
                .with("text", text)
        };
        let event = self.block_delta(delta);
        self.emit(event)
    }

    fn tool(&mut self, call: &ToolDelta<'_>) -> bool {
        if !call.id.is_empty() {
            if !self.close_block() {
                return false;
            }
            self.tool = Value::object()
                .with("type", "tool_use")
                .with("id", call.id)
                .with("name", call.name)
                .with("input", Value::object());
            self.tool_arguments.clear();
            let start = self.block_start(self.tool.clone());
            if !self.emit(start) {
                return false;
            }
        }
        self.tool_arguments.push_str(call.arguments);
        if !call.arguments.is_empty() {
            let delta = self.block_delta(
                Value::object()
                    .with("type", "input_json_delta")
                    .with("partial_json", call.arguments),
            );
            if !self.emit(delta) {
                return false;
            }
        }
        !call.complete || self.close_tool(true)
    }

    fn complete(&mut self, result: &GenerationResult) -> Value {
        self.close_tool(false);
        self.close_block();
        // An empty turn still has one (empty) text block.
        if self.message.find("content").is_some_and(Value::is_empty) {
            self.text.clear();
            self.reasoning = false;
            self.active = true;
            let start = self.block_start(self.block());
            self.emit(start);
            self.close_block();
        }
        // Chat's finish_reason precedence: a truncated or stopped turn is
        // reported as such even when complete calls precede the cut.
        let stop_reason = match result.finish_reason {
            FinishReason::StopSequence => "stop_sequence",
            FinishReason::Length => "max_tokens",
            _ if self.tool_use => "tool_use",
            _ => "end_turn",
        };
        let stop_sequence = if result.finish_reason == FinishReason::StopSequence {
            Value::from(result.stop_sequence.as_str())
        } else {
            Value::Null
        };
        // Anthropic reports cache reads apart: input_tokens counts only the
        // uncached prompt, so the three input fields add up to the prompt
        //.
        let usage = Value::object()
            .with(
                "input_tokens",
                result
                    .prompt_tokens
                    .saturating_sub(result.cached_prompt_tokens),
            )
            .with("output_tokens", result.completion_tokens)
            .with("cache_creation_input_tokens", 0)
            .with("cache_read_input_tokens", result.cached_prompt_tokens);
        self.message.set("stop_reason", stop_reason);
        self.message.set("stop_sequence", stop_sequence.clone());
        self.message.set("usage", usage.clone());
        self.message.set("timings", generation_timings(result));
        let delta = Self::event("message_delta")
            .with(
                "delta",
                Value::object()
                    .with("stop_reason", stop_reason)
                    .with("stop_sequence", stop_sequence),
            )
            .with("usage", usage);
        if self.emit(delta) {
            self.emit(Self::event("message_stop"));
        }
        self.message.clone()
    }

    fn fail(&mut self, message: &str) -> bool {
        let event = Self::event("error").with(
            "error",
            Value::object()
                .with("type", "api_error")
                .with("message", message),
        );
        self.emit(event)
    }

    fn connected(&self) -> bool {
        self.connected
    }
}
