//! The pinned DeepSeek V4 Flash 0731 conversation format
//! (`chat_template.cpp`): system text, user and tool blocks joined into one
//! user turn, assistant turns with their reasoning and DSML tool calls, the
//! tool schemas, and the reasoning-effort prompts.

use std::collections::HashMap;

use gufo_core::text::{ReasoningEffort, ReasoningOptions};
use sha2::{Digest, Sha256};

use crate::gguf::Model;

const BOS: &str = "<｜begin▁of▁sentence｜>";
const EOS: &str = "<｜end▁of▁sentence｜>";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";

const HIGH_REASONING_PROMPT: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n\
You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n\
Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

const MAX_REASONING_PROMPT: &str = "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n\
You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n\
Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

/// `ChatTemplateId`.
pub const CHAT_TEMPLATE_ID: &str = "deepseek-v4-flash-0731-compiled-v3";
/// `ArtifactTemplateSha256`: the embedded template this formatter implements.
pub const ARTIFACT_TEMPLATE_SHA256: &str =
    "872492071c22c8d2025238120309ffbddddb666b49f4433f55c19b69bf51af27";
/// `EncoderReferenceSha256`.
pub const ENCODER_REFERENCE_SHA256: &str =
    "abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6";

/// A tool call argument of an assistant turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolArgument {
    pub name: String,
    pub value: String,
    pub is_string: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Vec<ToolArgument>,
    pub id: String,
}

/// A message with its template role name: system, developer, user,
/// assistant, tool or function.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub reasoning_content: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTool {
    pub name: String,
    pub description: String,
    pub parameters_json: String,
    /// The complete function definition, including optional fields such as
    /// `strict`.
    pub definition_json: String,
}

/// The literal encoder options; [`resolve_chat_options`] applies the
/// user-facing defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatTemplateOptions {
    pub enable_thinking: bool,
    pub reasoning_effort: ReasoningEffort,
    pub preserve_thinking: bool,
    pub tools_present: bool,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            enable_thinking: false,
            reasoning_effort: ReasoningEffort::Low,
            preserve_thinking: false,
            tools_present: false,
        }
    }
}

/// `ResolveDeepSeekChatOptions`: the user-facing defaults follow
/// DeepSeek's API (thinking enabled, high effort); explicit options override
/// them, and the checkpoint encoder itself has no default thinking mode.
pub fn resolve_chat_options(reasoning: &ReasoningOptions) -> ChatTemplateOptions {
    ChatTemplateOptions {
        enable_thinking: reasoning.enabled.unwrap_or(true),
        reasoning_effort: reasoning.effort.unwrap_or(ReasoningEffort::High),
        preserve_thinking: reasoning.preserve_thinking.unwrap_or(false),
        ..ChatTemplateOptions::default()
    }
}

#[derive(Clone, Copy)]
enum NativeEffort {
    Low,
    High,
    Max,
}

fn map_effort(effort: ReasoningEffort) -> NativeEffort {
    match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low => NativeEffort::Low,
        ReasoningEffort::Medium | ReasoningEffort::High | ReasoningEffort::XHigh => {
            NativeEffort::High
        }
        ReasoningEffort::Max => NativeEffort::Max,
    }
}

/// `DeepSeekReasoningEffortName`: the native effort after mapping.
pub fn reasoning_effort_name(effort: ReasoningEffort) -> &'static str {
    match map_effort(effort) {
        NativeEffort::Low => "low",
        NativeEffort::High => "high",
        NativeEffort::Max => "max",
    }
}

/// `GenerationPrompt`.
pub fn generation_prompt(enable_thinking: bool) -> &'static str {
    if enable_thinking {
        "<｜Assistant｜><think>"
    } else {
        "<｜Assistant｜></think>"
    }
}

/// `AppendJsonString`: quotes and escapes like the reference encoder
/// (other control characters and non-ASCII pass through).
fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `PythonJsonSpacing`: a space after every `:` and `,` outside strings, as
/// Python's `json.dumps` writes.
fn python_json_spacing(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + value.len() / 8);
    let (mut in_string, mut escaped) = (false, false);
    for c in value.chars() {
        out.push(c);
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
        } else if c == ':' || c == ',' {
            out.push(' ');
        }
    }
    out
}

fn push_tools_prompt(out: &mut String, tools: &[ChatTool]) {
    if tools.is_empty() {
        return;
    }
    out.push_str(
        "## Tools\n\n\
You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n\
<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"$TOOL_NAME\">\n\
<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n\
...\n\
</｜DSML｜invoke>\n\
<｜DSML｜invoke name=\"$TOOL_NAME2\">\n\
...\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>\n\n\
String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n\
If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n\
Otherwise, output directly after </think> with tool calls or final response.\n\n\
### Available Tool Schemas\n\n",
    );
    for (index, tool) in tools.iter().enumerate() {
        if !tool.definition_json.is_empty() {
            out.push_str(&python_json_spacing(&tool.definition_json));
        } else {
            out.push_str("{\"name\": ");
            push_json_string(out, &tool.name);
            out.push_str(", \"description\": ");
            push_json_string(out, &tool.description);
            out.push_str(", \"parameters\": ");
            if tool.parameters_json.is_empty() {
                out.push_str("{}");
            } else {
                out.push_str(&python_json_spacing(&tool.parameters_json));
            }
            out.push('}');
        }
        if index + 1 < tools.len() {
            out.push('\n');
        }
    }
    out.push_str("\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n");
}

fn push_tool_calls(out: &mut String, calls: &[ToolCall]) {
    if calls.is_empty() {
        return;
    }
    out.push_str("\n\n<｜DSML｜tool_calls>\n");
    for (call_index, call) in calls.iter().enumerate() {
        out.push_str("<｜DSML｜invoke name=\"");
        out.push_str(&call.name);
        out.push_str("\">\n");
        for (argument_index, argument) in call.arguments.iter().enumerate() {
            out.push_str("<｜DSML｜parameter name=\"");
            out.push_str(&argument.name);
            out.push_str("\" string=\"");
            out.push_str(if argument.is_string { "true" } else { "false" });
            out.push_str("\">");
            out.push_str(&argument.value);
            out.push_str("</｜DSML｜parameter>");
            if argument_index + 1 < call.arguments.len() {
                out.push('\n');
            }
        }
        out.push_str("\n</｜DSML｜invoke>");
        if call_index + 1 < calls.len() {
            out.push('\n');
        }
    }
    out.push_str("\n</｜DSML｜tool_calls>");
}

fn is_user_block(role: &str) -> bool {
    matches!(role, "user" | "tool" | "function")
}

/// `RenderChat`: the conversation, tool schemas and generation prompt.
pub fn render_chat(
    messages: &[ChatMessage],
    tools: &[ChatTool],
    options: &ChatTemplateOptions,
) -> String {
    let mut out = String::from(BOS);
    if options.enable_thinking {
        match map_effort(options.reasoning_effort) {
            NativeEffort::Low => {}
            NativeEffort::High => out.push_str(HIGH_REASONING_PROMPT),
            NativeEffort::Max => out.push_str(MAX_REASONING_PROMPT),
        }
    }
    let last_user_index = messages
        .iter()
        .rposition(|m| m.role == "user" || m.role == "developer")
        .unwrap_or(messages.len());
    let preserve_thinking = options.preserve_thinking || options.tools_present || !tools.is_empty();
    let mut tool_host_index = messages.len();
    if !tools.is_empty() {
        if let Some(index) = messages
            .iter()
            .position(|m| m.role == "system" || m.role == "developer")
        {
            tool_host_index = index;
        } else {
            push_tools_prompt(&mut out, tools);
        }
    }
    let mut tool_call_order: HashMap<&str, usize> = HashMap::new();
    let mut index = 0;
    while index < messages.len() {
        let message = &messages[index];
        let role = message.role.as_str();
        match role {
            "system" => {
                out.push_str(&message.content);
                if index == tool_host_index {
                    if !message.content.is_empty() {
                        out.push_str("\n\n");
                    }
                    push_tools_prompt(&mut out, tools);
                }
            }
            "developer" => {
                if !(options.enable_thinking && !preserve_thinking && index < last_user_index) {
                    out.push_str(USER);
                    out.push_str(&message.content);
                    if index == tool_host_index {
                        if !message.content.is_empty() {
                            out.push_str("\n\n");
                        }
                        push_tools_prompt(&mut out, tools);
                    }
                }
            }
            _ if is_user_block(role) => {
                let end = messages[index..]
                    .iter()
                    .position(|m| !is_user_block(&m.role))
                    .map_or(messages.len(), |n| index + n);
                // `sort_tool_results_by_call_order` of the reference encoder:
                // only the tool slots reorder, interleaved user text stays;
                // unknown IDs rank zero and ties keep arrival order.
                let mut results: Vec<&ChatMessage> = messages[index..end]
                    .iter()
                    .filter(|m| m.role != "user")
                    .collect();
                results.sort_by_key(|m| {
                    tool_call_order
                        .get(m.tool_call_id.as_str())
                        .copied()
                        .unwrap_or(0)
                });
                let mut results = results.into_iter();
                out.push_str(USER);
                for (block, next) in messages[index..end].iter().enumerate() {
                    if block > 0 {
                        out.push_str("\n\n");
                    }
                    if next.role == "user" {
                        out.push_str(&next.content);
                    } else if let Some(result) = results.next() {
                        out.push_str("<tool_result>");
                        out.push_str(&result.content);
                        out.push_str("</tool_result>");
                    }
                }
                index = end;
                continue;
            }
            "assistant" => {
                if !message.tool_calls.is_empty() {
                    tool_call_order.clear();
                    for (call, tool_call) in message.tool_calls.iter().enumerate() {
                        if !tool_call.id.is_empty() {
                            tool_call_order.insert(&tool_call.id, call);
                        }
                    }
                }
                out.push_str(ASSISTANT);
                if options.enable_thinking && (preserve_thinking || index > last_user_index) {
                    out.push_str(THINK_START);
                    out.push_str(&message.reasoning_content);
                    out.push_str(THINK_END);
                } else {
                    out.push_str(THINK_END);
                }
                out.push_str(&message.content);
                push_tool_calls(&mut out, &message.tool_calls);
                out.push_str(EOS);
            }
            _ => {}
        }
        index += 1;
    }
    if let Some(last) = messages.last()
        && matches!(
            last.role.as_str(),
            "user" | "developer" | "tool" | "function"
        )
    {
        out.push_str(generation_prompt(options.enable_thinking));
    }
    out
}

/// `ValidateGgufTemplate`: refuses an artifact whose embedded template is
/// not the pinned converter version this formatter implements.
pub fn validate_gguf_template(model: &Model) -> Result<(), String> {
    if model.get_string("general.architecture") != Some(b"deepseek4".as_slice()) {
        return Err("DeepSeek V4 Flash artifact has unexpected architecture".into());
    }
    let wanted = b"DeepSeek V4 Flash";
    if !model
        .get_string("general.name")
        .is_some_and(|name| name.windows(wanted.len()).any(|w| w == wanted))
    {
        return Err("DeepSeek artifact is not the supported V4 Flash revision".into());
    }
    let template = model
        .get_string("tokenizer.chat_template")
        .filter(|t| !t.is_empty())
        .ok_or("DeepSeek GGUF is missing tokenizer.chat_template metadata")?;
    let actual: String = Sha256::digest(template)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if actual != ARTIFACT_TEMPLATE_SHA256 {
        return Err(format!(
            "DeepSeek GGUF chat template SHA-256 is not the pinned 0731 artifact version: {actual}"
        ));
    }
    Ok(())
}
