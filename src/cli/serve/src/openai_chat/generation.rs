//! A finished generation as the API's message: content, reasoning, tool
//! calls and structured output, the finish reason and usage.

use super::*;

/// A Responses message item as a chat message: `input_text`/`output_text`
/// parts become text, `input_image` parts image URLs (HTTPS or data URLs)
/// read under `budget` (`ParseOpenAiResponseMessage`).
pub(crate) fn parse_openai_response_message(
    item: &Value,
    message: &mut ChatMessage,
    budget: &mut ImageReadBudget,
) -> std::result::Result<(), String> {
    let mut converted = item.clone();
    if let Some(Value::Array(parts)) = item.find("content") {
        let mut content = Vec::with_capacity(parts.len());
        for part in parts {
            let mut value = part.clone();
            match part.member_str("type", "").as_str() {
                "input_image" => {
                    let Some(url) = part.find("image_url").filter(|url| url.is_string()) else {
                        return Err("input_image requires an image_url".into());
                    };
                    if part.contains("file_id") {
                        return Err("input_image requires an image_url".into());
                    }
                    let mut image_url = Value::object().with("url", url.clone());
                    if let Some(detail) = part.find("detail") {
                        image_url.set("detail", detail.clone());
                    }
                    value = Value::object()
                        .with("type", "image_url")
                        .with("image_url", image_url);
                }
                "input_text" | "output_text" => value.set("type", "text"),
                "text" => {}
                _ => return Err("unsupported Responses input content type".into()),
            }
            content.push(value);
        }
        converted.set("content", Value::Array(content));
    }
    parse_message(&converted, message, budget)
}

/// The structured output of a Responses request: `text.format` as a
/// grammar and its description (`ParseOpenAiResponseControls`; this server
/// reads `reasoning` with the other compatibility controls).
pub(crate) fn parse_openai_response_format(
    body: &Value,
    chat: &mut ChatRequest,
) -> std::result::Result<(), String> {
    let Some(text) = body.find("text").filter(|text| !text.is_null()) else {
        return Ok(());
    };
    if !text.is_object() {
        return Err("'text' must be an object".into());
    }
    for (key, value) in text.members() {
        // Verbosity has no native equivalent: accepted and ignored.
        if key == "verbosity" {
            continue;
        }
        if key != "format" {
            return Err(format!("unsupported text member: {key}"));
        }
        let grammar = parse_response_format(Some(value), true)?;
        if grammar.is_some() {
            chat.response_format_description = response_format_description(Some(value), true);
        }
        chat.response_format = grammar.map(OutputGrammar);
    }
    Ok(())
}

/// `enforce_required` is false when a stop sequence or the token limit cut
/// the generation short of a required tool call. Only `syntax.markers` begin
/// tool markup ([`tool_syntax`]); without them reasoning ends at `</think>`
/// alone.
pub(super) fn parse_generation(
    raw: &str,
    initial_output_state: InitialOutputState,
    tools: &[ChatTool],
    choice: ToolChoice,
    enforce_required: bool,
    syntax: ToolSyntax,
    partial: bool,
) -> Result<ParsedGeneration> {
    let ToolSyntax {
        markers,
        closers,
        qwen,
    } = syntax;
    const THINK_START: &str = "<think>";
    const THINK_END: &str = "</think>";
    let mut parsed = ParsedGeneration::default();
    let mut content = raw;
    // A call taken to end the reasoning implicitly, inside an unclosed
    // quote: its reasoning context, kept in case the call is invalid.
    let mut reasoning_call_context: Option<&str> = None;
    let unclosed_marker = |text: &str, marker: usize| {
        let mut quotes = QuoteTracker::default();
        quotes.reset(text, 0);
        quotes.unclosed_at(marker)
    };
    if initial_output_state == InitialOutputState::Reasoning {
        content = content.strip_prefix(THINK_START).unwrap_or(content);
        let mut think_end = content.find(THINK_END);
        let marker = earliest_marker_in(content, markers).map(|(p, _)| p);
        if marker.is_some_and(|m| think_end.is_none_or(|e| m < e)) {
            think_end = None;
        }
        match think_end {
            None => {
                parsed.reasoning_content =
                    trim(&content[..marker.unwrap_or(content.len())]).to_owned();
                let Some(marker) = marker else {
                    if enforce_required && choice == ToolChoice::Required {
                        return Err(tool_choice_unsatisfied());
                    }
                    return Ok(parsed);
                };
                if unclosed_marker(content, marker) {
                    reasoning_call_context = Some(content);
                }
                parsed.text = content[marker..].to_owned();
            }
            Some(end) => {
                parsed.reasoning_content = trim(&content[..end]).to_owned();
                parsed.text =
                    after_reasoning_separator(&content[end + THINK_END.len()..], qwen).to_owned();
            }
        }
    } else if initial_output_state == InitialOutputState::Auto
        && let Some(think_start) = content
            .find(|c: char| !matches!(c, ' ' | '\t' | '\r' | '\n'))
            .filter(|&leading| content[leading..].starts_with(THINK_START))
    {
        // Only the initial phase of an automatic start has reasoning markup
        // semantics: a literal tag inside an argument or quoted text, or any
        // tag while thinking is off, is never stripped.
        let think_content_start = think_start + THINK_START.len();
        let mut think_end = content[think_content_start..]
            .find(THINK_END)
            .map(|e| e + think_content_start);
        let marker = earliest_marker_in(&content[think_content_start..], markers).map(|(p, _)| p);
        if let Some(marker) = marker
            && think_end.is_none_or(|e| think_content_start + marker < e)
        {
            think_end = None;
        }
        match think_end {
            Some(end) => {
                parsed.reasoning_content = trim(&content[think_content_start..end]).to_owned();
                let remaining = after_reasoning_separator(&content[end + THINK_END.len()..], qwen);
                parsed.text = if think_start > 0 {
                    format!("{}{}", &content[..think_start], remaining)
                } else {
                    remaining.to_owned()
                };
            }
            None => {
                let remaining = &content[think_content_start..];
                let marker = earliest_marker_in(remaining, markers).map(|(p, _)| p);
                parsed.reasoning_content =
                    trim(&remaining[..marker.unwrap_or(remaining.len())]).to_owned();
                parsed.text = content[..think_start].to_owned();
                if let Some(marker) = marker {
                    if unclosed_marker(remaining, marker) {
                        reasoning_call_context = Some(remaining);
                    }
                    parsed.text.push_str(&remaining[marker..]);
                }
            }
        }
    } else {
        parsed.text = content.to_owned();
    }

    let recognize_tools = !tools.is_empty() && choice != ToolChoice::None;
    let mut quotes = QuoteTracker::default();
    quotes.reset(&parsed.text, 0);
    let marker = earliest_marker(&parsed.text, markers, &quotes, 0).map(|(marker, _)| marker);
    if let Some(marker) = marker.filter(|_| recognize_tools) {
        // While streaming, a call in an unfinished quotation is undecided.
        if partial && (reasoning_call_context.is_some() || quotes.unclosed_at(marker)) {
            return Ok(parsed);
        }
        let mut spans = Vec::new();
        // The outer envelope selects the dialect, as in llama.cpp's model
        // parsers: scanning both would run a call quoted in an argument.
        // Qwen calls see the whole text, so their quoting context counts.
        if parsed.text[marker..].starts_with("<tool_call>") {
            let text = parsed.text.clone();
            parse_qwen_calls(
                &text,
                tools,
                &mut parsed.tool_calls,
                None,
                reasoning_call_context.is_some(),
                partial,
            );
        } else {
            parse_dsml_calls(
                &parsed.text[marker..],
                markers,
                tools,
                &mut parsed.tool_calls,
                Some(&mut spans),
                partial,
            );
        }
        parsed
            .tool_calls
            .retain(|call| tools.iter().any(|tool| tool.name == call.name));
        if parsed.tool_calls.is_empty() && choice != ToolChoice::Required {
            // The implicit end of the reasoning was an invalid quoted
            // example: it stays reasoning, with a later explicit boundary.
            if let Some(context) = reasoning_call_context {
                let end = context.find(THINK_END);
                let mut kept = match end {
                    Some(end) => parse_generation(
                        &context[end + THINK_END.len()..],
                        InitialOutputState::Content,
                        tools,
                        choice,
                        enforce_required,
                        syntax,
                        partial,
                    )?,
                    None => ParsedGeneration::default(),
                };
                kept.reasoning_content = context[..end.unwrap_or(context.len())].to_owned();
                return Ok(kept);
            }
            // A marker in an unclosed quote without a valid call is prose.
            if quotes.unclosed_at(marker) {
                return Ok(parsed);
            }
        }
        // A stop sequence can cut a call before its closing tags: complete
        // calls stay, an unfinished one is not content either.
        if !parsed.tool_calls.is_empty() || !enforce_required {
            // Text outside complete call blocks is content; framing echoed
            // directly after a call goes, as for Qwen calls.
            let full = std::mem::take(&mut parsed.text);
            let content = |begin: usize, end: usize, after_call: bool| {
                content_before(&full, begin, end, closers, &quotes, tools, after_call)
            };
            let mut text = content(0, marker, false).to_owned();
            let mut cursor = marker;
            for &(begin, end) in &spans {
                text.push_str(content(cursor, marker + begin, cursor > marker));
                cursor = marker + end;
            }
            if !spans.is_empty() {
                let mut unfinished =
                    earliest_marker(&full, markers, &quotes, cursor).map(|(found, _)| found);
                if unfinished.is_none() && !enforce_required && choice == ToolChoice::Required {
                    let held = held_marker_prefix(&full, full.len(), markers, &quotes);
                    if held > 0 {
                        unfinished = Some(full.len() - held);
                    }
                }
                text.push_str(content(cursor, unfinished.unwrap_or(full.len()), true));
            }
            parsed.text = text;
            parsed.hide_tool_markup = true;
        }
    } else if recognize_tools {
        // Without a marker the text is content, but the framing it carries
        // still goes: the streamed response drops it as the turn ends, and
        // both transports report the same content.
        parsed.text = content_before(
            &parsed.text,
            0,
            parsed.text.len(),
            closers,
            &quotes,
            tools,
            false,
        )
        .to_owned();
    }
    if enforce_required && choice == ToolChoice::Required && parsed.tool_calls.is_empty() {
        return Err(tool_choice_unsatisfied());
    }
    Ok(parsed)
}

/// The output of a structured request: reasoning keeps its markup
/// semantics, while tool markers inside the JSON answer are data. Constrained
/// tool calls are parsed only where the grammar allows them; `tool_only`
/// requests keep prose between or after complete calls
/// (`ParseStructuredGeneration`).
#[allow(clippy::too_many_arguments)]
pub(super) fn parse_structured_generation(
    raw: &str,
    initial: InitialOutputState,
    tools: &[ChatTool],
    choice: ToolChoice,
    enforce_required: bool,
    tool_only: bool,
    syntax: ToolSyntax,
    partial: bool,
) -> Result<ParsedGeneration> {
    const TOOL_START: &str = "<tool_call>";
    let ToolSyntax {
        markers,
        closers,
        qwen,
    } = syntax;
    let earliest = |text: &str| earliest_marker_in(text, markers).map(|(position, _)| position);
    let mut parsed = ParsedGeneration::default();
    let mut raw = raw;
    let mut initial = initial;
    // An automatic start opens reasoning with a leading `<think>`.
    if initial == InitialOutputState::Auto
        && let Some(leading) = raw.find(|c: char| !matches!(c, ' ' | '\t' | '\r' | '\n'))
        && raw[leading..].starts_with("<think>")
    {
        raw = &raw[leading..];
        initial = InitialOutputState::Reasoning;
    }
    if initial == InitialOutputState::Reasoning {
        raw = raw.strip_prefix("<think>").unwrap_or(raw);
        // Constrained reasoning ends at `</think>` or at a native call
        // header outside quotations; quoted markers are
        // reasoning data.
        let end = raw.find("</think>");
        let mut quotes = QuoteTracker::default();
        quotes.reset(raw, 0);
        let call = if !tools.is_empty() && choice != ToolChoice::None {
            reasoning_tool_marker(raw, reasoning_tool_markers(markers), &quotes, 0)
        } else {
            None
        };
        if let Some(call) = call.filter(|&call| end.is_none_or(|end| call < end)) {
            if partial && quotes.unclosed_at(call) {
                return Ok(parsed);
            }
            parsed.reasoning_content = raw[..call].to_owned();
            raw = &raw[call..];
        } else {
            parsed.reasoning_content = raw[..end.unwrap_or(raw.len())].to_owned();
            let Some(end) = end else {
                return Ok(parsed);
            };
            raw = after_reasoning_separator(&raw[end + "</think>".len()..], qwen);
        }
    }
    let content = if tool_only { raw } else { trim(raw) };
    let mut quotes = QuoteTracker::default();
    quotes.reset(content, 0);
    // A native call in another dialect is parsed as free-form markup.
    if let Some(marker) = earliest(content)
        && !tools.is_empty()
        && choice != ToolChoice::None
        && (tool_only || marker == 0)
        && !content[marker..].starts_with(TOOL_START)
    {
        let mut native = parse_generation(
            content,
            InitialOutputState::Content,
            tools,
            choice,
            enforce_required,
            syntax,
            partial,
        )?;
        native.reasoning_content = parsed.reasoning_content;
        return Ok(native);
    }
    let mut marker = if tool_only {
        earliest(content)
    } else {
        Some(0)
    };
    if tool_only && choice == ToolChoice::Required && marker.is_none() {
        // A required call cut short keeps its partial opening marker hidden.
        let held = held_marker_prefix(content, content.len(), markers, &quotes);
        if held > 0 {
            marker = Some(content.len() - held);
        }
    }
    if let Some(marker) = marker
        && !tools.is_empty()
        && choice != ToolChoice::None
        && !content.is_empty()
        && (tool_only || content.starts_with(TOOL_START) || tool_marker_prefix(content, markers))
    {
        let mut call = ParsedGeneration::default();
        let mut spans = Vec::new();
        parse_qwen_calls(
            content,
            tools,
            &mut call.tool_calls,
            tool_only.then_some(&mut spans),
            false,
            partial,
        );
        if tool_only {
            let prose = |begin: usize, end: usize| {
                content_before(content, begin, end, closers, &quotes, tools, begin > 0)
            };
            let mut cursor = 0;
            for &(begin, end) in &spans {
                call.text.push_str(prose(cursor, begin));
                cursor = end;
            }
            let tail = &content[cursor..];
            let mut unfinished =
                earliest_marker(content, markers, &quotes, cursor).map(|(p, _)| p - cursor);
            if unfinished.is_none() && !enforce_required && choice == ToolChoice::Required {
                let held = held_marker_prefix(content, content.len(), markers, &quotes);
                if held > 0 {
                    unfinished = Some(tail.len() - held);
                }
            }
            let cut = if spans.is_empty() {
                Some(marker)
            } else {
                unfinished
            };
            // The model repeats the framing of the call it just made: that
            // echo is markup, not prose a client replays on the next turn.
            call.text.push_str(prose(
                cursor,
                cut.map_or(content.len(), |cut| (cursor + cut).min(content.len())),
            ));
        }
        // A marker in an unclosed quote without a valid call is prose.
        if call.tool_calls.is_empty()
            && choice != ToolChoice::Required
            && quotes.unclosed_at(marker)
        {
            parsed.text = raw.to_owned();
            return Ok(parsed);
        }
        call.hide_tool_markup = true;
        if call.tool_calls.is_empty() && enforce_required && choice == ToolChoice::Required {
            return Err(Error::generation(
                GenerationErrorCode::ToolChoiceUnsatisfied,
                "model did not complete a declared tool call",
            ));
        }
        call.reasoning_content = parsed.reasoning_content;
        return Ok(call);
    }
    // Markers in JSON strings are data; only the initial reasoning phase has
    // markup semantics.
    if enforce_required && choice == ToolChoice::Required {
        return Err(Error::generation(
            GenerationErrorCode::ToolChoiceUnsatisfied,
            "model did not complete a declared tool call",
        ));
    }
    // A constrained turn without a call is still reported without framing:
    // the streamed response drops it as the turn ends, and both transports
    // report the same content.
    parsed.text = if tool_only {
        content_before(raw, 0, raw.len(), closers, &quotes, tools, false).to_owned()
    } else {
        raw.to_owned()
    };
    Ok(parsed)
}

/// Stops (stop sequences included) do not guarantee complete JSON; token
/// limits say `length` even after complete tool calls.
pub(super) fn finish_reason(result: &GenerationResult, has_tool_calls: bool) -> &'static str {
    if result.finish_reason == FinishReason::StopSequence {
        "stop"
    } else if result.finish_reason == FinishReason::Length {
        "length"
    } else if has_tool_calls {
        "tool_calls"
    } else {
        "stop"
    }
}

/// How a chat request's output is parsed (`StreamingGeneration`):
/// structured requests follow the grammar, others the free-form tool and
/// reasoning markup. `partial` reads a streamed prefix, returning the
/// unfinished call too; only a native grammar establishes argument framing
/// before the value completes, so recovery from unconstrained or JSON
/// fallback text still needs a complete call.
pub(super) fn parse_request_generation(
    chat: &ChatRequest,
    syntax: ToolSyntax,
    raw: &str,
    initial: InitialOutputState,
    enforce_required: bool,
    partial: bool,
) -> Result<ParsedGeneration> {
    let partial = partial && syntax.markers.len() <= 2;
    if chat.response_format.is_some() || chat.constrained_tools {
        parse_structured_generation(
            raw,
            initial,
            &chat.tools,
            chat.tool_choice,
            enforce_required,
            chat.constrained_tools && chat.response_format.is_none(),
            syntax,
            partial,
        )
    } else {
        parse_generation(
            raw,
            initial,
            &chat.tools,
            chat.tool_choice,
            enforce_required,
            syntax,
            partial,
        )
    }
}

pub(super) fn usage(result: &GenerationResult) -> Value {
    let predicted_per_second = if result.decode_ms > 0.0 && result.completion_tokens > 0 {
        result.completion_tokens as f64 / (result.decode_ms / 1000.0)
    } else {
        0.0
    };
    let mut metrics = Value::object().with("cache_hit", result.cache_hit);
    if !result.cache_miss_reason.is_empty() {
        metrics.set("cache_miss_reason", result.cache_miss_reason.as_str());
        metrics.set(
            "cache_common_prefix_tokens",
            result.cache_common_prefix_tokens,
        );
        metrics.set("cache_checkpoint_tokens", result.cache_checkpoint_tokens);
    }
    let metrics = metrics
        .with("cache_restore_bytes", result.cache_restore_bytes)
        .with("cache_snapshot_bytes", result.cache_snapshot_bytes)
        .with("cache_disk_queued_bytes", result.cache_disk_queued_bytes)
        .with("cache_shared_bytes", result.cache_shared_bytes)
        .with("cache_restore_ms", result.cache_restore_ms)
        .with("cache_snapshot_ms", result.cache_snapshot_ms)
        .with("cache_disk_enqueue_ms", result.cache_disk_enqueue_ms)
        .with("cache_disk_hit", result.cache_disk_hit)
        .with(
            "cache_shared_prefix_snapshots",
            result.cache_shared_prefix_snapshots,
        )
        .with(
            "cache_shared_prefix_bytes",
            result.cache_shared_prefix_bytes,
        )
        .with("cache_shared_prefix_ms", result.cache_shared_prefix_ms)
        .with("draft_rounds", result.draft_rounds)
        .with("prefill_tokens", result.prefill_tokens)
        .with("prefill_chunks", result.prefill_chunks)
        .with(
            "active_decode_prefill_chunks",
            result.active_decode_prefill_chunks,
        )
        .with("max_prefill_chunk_tokens", result.max_prefill_chunk_tokens)
        .with("queue_depth_at_submit", result.queue_depth_at_submit)
        .with("shared_prefix_wait_ms", result.shared_prefix_wait_ms)
        .with(
            "client_queue_depth_at_submit",
            result.client_queue_depth_at_submit,
        )
        .with(
            "resident_requests_at_admission",
            result.resident_requests_at_admission,
        )
        .with(
            "requested_logical_concurrency",
            result.requested_logical_concurrency,
        )
        .with("physical_execution_width", result.physical_execution_width)
        .with("queue_ms", result.queue_ms)
        .with("prefill_ms", result.prefill_ms)
        .with("decode_ms", result.decode_ms)
        .with("ttft_ms", result.ttft_ms)
        .with("mean_inter_token_ms", result.mean_inter_token_ms)
        .with("max_inter_token_ms", result.max_inter_token_ms)
        .with("execution_plan", result.execution_plan.as_str());
    Value::object()
        .with("prompt_tokens", result.prompt_tokens)
        .with("completion_tokens", result.completion_tokens)
        .with(
            "total_tokens",
            result.prompt_tokens + result.completion_tokens,
        )
        .with(
            "prompt_tokens_details",
            Value::object().with("cached_tokens", result.cached_prompt_tokens),
        )
        .with("cached_tokens", result.cached_prompt_tokens)
        .with(
            "prompt_tokens_per_second",
            prefill_tokens_per_second(result),
        )
        .with("completion_tokens_per_second", predicted_per_second)
        .with("draft_tokens", result.draft_tokens)
        .with("draft_tokens_accepted", result.draft_accepted_tokens)
        .with("gufo", metrics)
}

pub(super) fn tool_calls_json(calls: &[ParsedToolCall]) -> Value {
    Value::from(
        calls
            .iter()
            .map(|call| {
                Value::object()
                    .with("id", call.id.as_str())
                    .with("type", "function")
                    .with(
                        "function",
                        Value::object()
                            .with("name", call.name.as_str())
                            .with("arguments", arguments_json(&call.arguments)),
                    )
            })
            .collect::<Vec<_>>(),
    )
}
