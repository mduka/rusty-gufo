//! The `/v1/responses` output: items, content parts and their events.

use super::*;

/// One Responses output, buffered or streamed (`ResponsesOutput`). The
/// reasoning is a `reasoning` item with one text part where `target` puts
/// it; the answer a `message` item with one `output_text` part. Events add
/// the indices and sequence numbers the OpenAI SDK validates.
pub(super) struct ResponsesOutput<'w> {
    pub(super) writer: Option<&'w mut BodyWriter<'w>>,
    /// The namespace of each function flattened out of one, for its calls.
    pub(super) namespaces: std::collections::BTreeMap<String, String>,
    pub(super) response: Value,
    pub(super) item: Value,
    pub(super) text: String,
    /// The open function call item, or null.
    pub(super) tool: Value,
    pub(super) tool_arguments: String,
    /// Reasoning whitespace not yet followed by text: the reasoning comes
    /// trimmed, as in the parsed chat reasoning.
    pub(super) held: String,
    pub(super) sequence: usize,
    pub(super) output_index: usize,
    pub(super) active: bool,
    pub(super) reasoning: bool,
    pub(super) connected: bool,
    /// A streamed call that never completed, cut by a stop or the limit.
    pub(super) unfinished_tool: bool,
    pub(super) target: ResponsesReasoningTarget,
}

impl<'w> ResponsesOutput<'w> {
    pub(super) fn new(
        model: &str,
        writer: Option<&'w mut BodyWriter<'w>>,
        chat: &ChatRequest,
        target: ResponsesReasoningTarget,
    ) -> Self {
        let tool_choice = if !chat.forced_tool_name.is_empty() {
            Value::object()
                .with("type", "function")
                .with("name", chat.forced_tool_name.as_str())
        } else if chat.tools.is_empty() || chat.tool_choice == ToolChoice::None {
            Value::from("none")
        } else if chat.tool_choice == ToolChoice::Required {
            Value::from("required")
        } else {
            Value::from("auto")
        };
        // The declared functions in the flat Responses shape.
        let tools = chat
            .tools
            .iter()
            .filter_map(|tool| {
                let definition = json::parse_str(&tool.definition_json).ok()?;
                Some(
                    definition
                        .find("function")?
                        .clone()
                        .with("type", "function"),
                )
            })
            .collect::<Vec<_>>();
        let response = Value::object()
            .with("id", random_id("resp_", 12))
            .with("object", "response")
            .with("created_at", unix_time())
            .with("model", model)
            .with("status", "in_progress")
            .with("error", Value::Null)
            .with("incomplete_details", Value::Null)
            .with("usage", Value::Null)
            .with("output", Value::array())
            .with("store", false)
            .with(
                "parallel_tool_calls",
                !chat.tools.is_empty() && chat.parallel_tool_calls,
            )
            .with("tool_choice", tool_choice)
            .with("tools", Value::from(tools));
        Self {
            writer,
            namespaces: chat.tool_namespaces.clone(),
            response,
            item: Value::Null,
            text: String::new(),
            tool: Value::Null,
            tool_arguments: String::new(),
            held: String::new(),
            sequence: 0,
            output_index: 0,
            active: false,
            reasoning: false,
            connected: true,
            unfinished_tool: false,
            target,
        }
    }

    /// Whether the open item is reasoning carried in `summary`.
    pub(super) fn in_summary(&self) -> bool {
        self.reasoning && self.target == ResponsesReasoningTarget::Summary
    }

    /// Closes the open function call item with `status`.
    pub(super) fn close_tool(&mut self, status: &str) -> bool {
        if self.tool.is_null() {
            return self.connected;
        }
        self.unfinished_tool |= status == "incomplete";
        let mut item = std::mem::replace(&mut self.tool, Value::Null);
        // llama.cpp sends only the deltas; the OpenAI protocol (and the SDK
        // event types) close the arguments before the item.
        let done = self
            .indexed("response.function_call_arguments.done")
            .with("item_id", item.member_str("id", ""))
            .with("name", item.member_str("name", ""))
            .with("arguments", self.tool_arguments.as_str());
        if !self.emit(done) {
            return false;
        }
        item.set("arguments", self.tool_arguments.as_str());
        item.set("status", status);
        let closed = self
            .indexed("response.output_item.done")
            .with("item", item.clone());
        self.response.member_mut("output").push(item);
        self.output_index += 1;
        self.emit(closed)
    }

    pub(super) fn indexed(&self, kind: &str) -> Value {
        Value::object()
            .with("type", kind)
            .with("output_index", self.output_index)
    }

    pub(super) fn part_event(&self, kind: &str) -> Value {
        self.indexed(kind)
            .with("item_id", self.item.member_str("id", ""))
            .with(
                if self.in_summary() {
                    "summary_index"
                } else {
                    "content_index"
                },
                0,
            )
    }

    /// The open part's `added` or `done` event.
    pub(super) fn part_lifecycle(&self, kind: &str) -> Value {
        let family = if self.in_summary() {
            "response.reasoning_summary_part"
        } else {
            "response.content_part"
        };
        self.part_event(&format!("{family}.{kind}"))
            .with("part", self.part())
    }

    /// The open part's text event: `delta` or `done` with `text` under
    /// `key`.
    pub(super) fn text_event(&self, kind: &str, key: &str, text: &str) -> Value {
        if self.reasoning {
            self.part_event(&self.target.text_event(kind))
                .with(key, text)
        } else {
            self.part_event(&format!("response.output_text.{kind}"))
                .with(key, text)
                .with("logprobs", Value::array())
        }
    }

    pub(super) fn part(&self) -> Value {
        if self.in_summary() {
            Value::object()
                .with("type", "summary_text")
                .with("text", self.text.as_str())
        } else if self.reasoning {
            Value::object()
                .with("type", "reasoning_text")
                .with("text", self.text.as_str())
        } else {
            Value::object()
                .with("type", "output_text")
                .with("annotations", Value::array())
                .with("logprobs", Value::array())
                .with("text", self.text.as_str())
        }
    }

    pub(super) fn close_item(&mut self, status: &str) -> bool {
        if !self.active {
            return self.connected;
        }
        self.held.clear();
        let done = self.text_event("done", "text", &self.text);
        if !self.emit(done) {
            return false;
        }
        if !self.emit(self.part_lifecycle("done")) {
            return false;
        }
        let parts = if self.in_summary() {
            "summary"
        } else {
            "content"
        };
        let part = self.part();
        self.item.member_mut(parts).push(part);
        self.item.set("status", status);
        let item_done = self
            .indexed("response.output_item.done")
            .with("item", self.item.clone());
        self.response.member_mut("output").push(self.item.clone());
        self.active = false;
        self.output_index += 1;
        self.emit(item_done)
    }

    pub(super) fn lifecycle(&mut self, kind: &str) -> bool {
        let event = Value::object()
            .with("type", kind)
            .with("response", self.response.clone());
        self.emit(event)
    }

    pub(super) fn emit(&mut self, mut event: Value) -> bool {
        let Some(writer) = self.writer.as_mut() else {
            return true;
        };
        if !self.connected {
            return false;
        }
        event.set("sequence_number", self.sequence);
        self.sequence += 1;
        let mut bytes = format!("event: {}\n", event.member_str("type", "")).into_bytes();
        bytes.extend(sse(&event));
        self.connected = writer(&bytes);
        self.connected
    }
}

impl CompatibilityOutput for ResponsesOutput<'_> {
    fn begin(&mut self) -> bool {
        self.lifecycle("response.created") && self.lifecycle("response.in_progress")
    }

    /// Prompt progress before the first output, as `response.in_progress`.
    fn progress(&mut self, progress: &PromptProgress) -> bool {
        let event = Value::object()
            .with("type", "response.in_progress")
            .with("response", self.response.clone())
            .with("prompt_progress", prompt_progress_json(progress));
        self.emit(event)
    }

    fn append(&mut self, piece: &str, reasoning: bool) -> bool {
        let joined;
        let mut text = piece;
        if reasoning {
            if !(self.active && self.reasoning) {
                text = text.trim_start_matches(is_space);
            }
            let body = text.trim_end_matches(is_space).len();
            if body == 0 {
                if self.active && self.reasoning {
                    self.held.push_str(text);
                }
                return true;
            }
            joined = std::mem::take(&mut self.held) + &text[..body];
            self.held.push_str(&text[body..]);
            text = &joined;
        }
        if text.is_empty() {
            return true;
        }
        if !self.close_tool("incomplete") {
            return false;
        }
        if !self.active || self.reasoning != reasoning {
            if !self.close_item("completed") {
                return false;
            }
            self.reasoning = reasoning;
            self.active = true;
            self.text.clear();
            let id = random_id(if reasoning { "rs_" } else { "msg_" }, 12);
            self.item = if self.in_summary() {
                // Upstream's item: the summary alone.
                Value::object()
                    .with("id", id)
                    .with("type", "reasoning")
                    .with("status", "in_progress")
                    .with("summary", Value::array())
            } else if reasoning {
                Value::object()
                    .with("id", id)
                    .with("type", "reasoning")
                    .with("summary", Value::array())
                    .with("content", Value::array())
                    .with("encrypted_content", "")
                    .with("status", "in_progress")
            } else {
                Value::object()
                    .with("id", id)
                    .with("type", "message")
                    .with("role", "assistant")
                    .with("content", Value::array())
                    .with("status", "in_progress")
            };
            let added = self
                .indexed("response.output_item.added")
                .with("item", self.item.clone());
            if !self.emit(added) {
                return false;
            }
            if !self.emit(self.part_lifecycle("added")) {
                return false;
            }
        }
        self.text.push_str(text);
        let delta = self.text_event("delta", "delta", text);
        self.emit(delta)
    }

    /// A function call item in llama.cpp's events: added in progress, its
    /// arguments as they parse, then done.
    fn tool(&mut self, call: &ToolDelta<'_>) -> bool {
        if !call.id.is_empty() {
            // Reasoning that resumes after the call (a quoted example taken
            // for one) keeps its separator.
            let held = std::mem::take(&mut self.held);
            if !self.close_item("completed") {
                return false;
            }
            self.held = held;
            let mut item = Value::object()
                .with("id", random_id("fc_", 20))
                .with("type", "function_call")
                .with("status", "in_progress")
                .with("arguments", "")
                .with("call_id", call.id)
                .with("name", call.name);
            // Clients route namespaced calls by namespace and name.
            if let Some(namespace) = self.namespaces.get(call.name) {
                item.set("namespace", namespace.as_str());
            }
            self.tool_arguments.clear();
            let added = self
                .indexed("response.output_item.added")
                .with("item", item.clone());
            self.tool = item;
            if !self.emit(added) {
                return false;
            }
        }
        self.tool_arguments.push_str(call.arguments);
        if !call.arguments.is_empty() {
            let delta = self
                .indexed("response.function_call_arguments.delta")
                .with("item_id", self.tool.member_str("id", ""))
                .with("delta", call.arguments);
            if !self.emit(delta) {
                return false;
            }
        }
        !call.complete || self.close_tool("completed")
    }

    fn complete(&mut self, result: &GenerationResult) -> Value {
        let limited = result.finish_reason == FinishReason::Length;
        self.close_tool("incomplete");
        // A length limit cuts the last item short.
        self.close_item(if limited { "incomplete" } else { "completed" });
        let incomplete = limited || self.unfinished_tool;
        self.response.set(
            "status",
            if incomplete {
                "incomplete"
            } else {
                "completed"
            },
        );
        if limited {
            self.response.set(
                "incomplete_details",
                Value::object().with("reason", "max_output_tokens"),
            );
        } else if self.unfinished_tool {
            self.response.set("incomplete_details", Value::object());
        } else {
            self.response.set("completed_at", unix_time());
        }
        self.response.set(
            "usage",
            Value::object()
                .with("input_tokens", result.prompt_tokens)
                .with("output_tokens", result.completion_tokens)
                .with(
                    "total_tokens",
                    result.prompt_tokens + result.completion_tokens,
                )
                .with(
                    "input_tokens_details",
                    Value::object()
                        .with("cached_tokens", result.cached_prompt_tokens)
                        .with("cache_write_tokens", result.prefill_tokens),
                )
                .with(
                    "output_tokens_details",
                    Value::object().with("reasoning_tokens", result.reasoning_tokens),
                ),
        );
        self.response.set("timings", generation_timings(result));
        self.lifecycle(if incomplete {
            "response.incomplete"
        } else {
            "response.completed"
        });
        self.response.clone()
    }

    fn fail(&mut self, message: &str) -> bool {
        self.response.set("status", "failed");
        // Responses defines a closed error-code enum; the specific runtime
        // code stays in the request log.
        self.response.set(
            "error",
            Value::object()
                .with("code", "server_error")
                .with("message", message),
        );
        self.lifecycle("response.failed")
    }

    fn connected(&self) -> bool {
        self.connected
    }
}
