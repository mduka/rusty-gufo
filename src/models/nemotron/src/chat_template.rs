//! The artifact's chat template (n5h_5p5_nanov2) and its tokenization
//! (`chat_template.cpp`): reasoning modes, the Qwen3-Coder XML tool format
//! and the client-text ranges that keep a literal control token as text.

use std::ops::Range;

use gufo_core::json::{self, Value};
use gufo_core::text::{ReasoningEffort, ReasoningOptions};

use crate::tokenizer::{TokenId, Tokenizer};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>\n";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const EMPTY_THINK: &str = "<think></think>";

const TOOL_INSTRUCTIONS: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the \
following format with NO suffix:\n\n<tool_call>\n\
<function=example_function_name>\n<parameter=example_parameter_1>\n\
value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the \
value for the second parameter\nthat can span\nmultiple lines\n\
</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- \
Function calls MUST follow the specified format: an inner \
<function=...></function> block must be nested within \
<tool_call></tool_call> XML tags\n- Required parameters MUST be \
specified\n- You may provide optional reasoning for your function call \
in natural language BEFORE the function call, but NOT after\n- If there \
is no function call available, answer the question like normal with \
your current knowledge and do not tell the user about function calls\n\
</IMPORTANT>";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ChatRole {
    System,
    #[default]
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatToolArgument {
    pub name: String,
    /// The string itself, or the argument's JSON text (`tojson` spacing).
    pub value: String,
    pub is_string: bool,
}

impl Default for ChatToolArgument {
    fn default() -> Self {
        Self {
            name: String::new(),
            value: String::new(),
            is_string: true,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatToolCall {
    pub name: String,
    pub arguments: Vec<ChatToolArgument>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    /// Assistant reasoning_content.
    pub reasoning: String,
    pub tool_calls: Vec<ChatToolCall>,
    /// Byte offsets into `content` where images sit, ascending (user and
    /// tool messages). Each renders as the template's `<image>`, labeled
    /// `<image N>` when the message holds several.
    pub image_offsets: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTool {
    pub name: String,
    pub description: String,
    /// JSON schema object; its properties, required list and other keys
    /// render as the template's XML-like tool block.
    pub parameters_json: String,
    /// Complete tool object when supplied, for keys beyond the four standard
    /// ones; may be empty.
    pub definition_json: String,
}

impl Default for ChatTool {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            parameters_json: "{}".into(),
            definition_json: String::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatOptions {
    pub enable_thinking: bool,
    /// Appends the template's "{reasoning effort: efficient}" hint.
    pub medium_effort: bool,
    pub truncate_history_thinking: bool,
    pub add_generation_prompt: bool,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            enable_thinking: true,
            medium_effort: false,
            truncate_history_thinking: true,
            add_generation_prompt: true,
        }
    }
}

/// Thinking on unless disabled; minimal, low and medium effort select the
/// template's efficient mode (`ResolveChatOptions`).
pub fn resolve_chat_options(reasoning: &ReasoningOptions) -> ChatOptions {
    let mut options = ChatOptions {
        enable_thinking: reasoning.enabled.unwrap_or(true),
        ..ChatOptions::default()
    };
    if matches!(
        reasoning.effort,
        Some(ReasoningEffort::Minimal | ReasoningEffort::Low | ReasoningEffort::Medium)
    ) {
        options.medium_effort = true;
    }
    if let Some(preserve) = reasoning.preserve_thinking {
        options.truncate_history_thinking = !preserve;
    }
    options
}

/// A byte range of a rendered prompt that holds client text (message
/// content, reasoning, tool-call arguments) rather than template framing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ContentSpan {
    pub offset: usize,
    pub size: usize,
}

/// Python's repr of a float: shortest round-trip digits, positional between
/// 1e-4 and 1e16, otherwise exponent form with a signed two-digit exponent.
fn python_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.into();
    }
    let sci = format!("{value:e}");
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let (negative, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, mantissa),
    };
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    let mut out = if negative {
        "-".to_string()
    } else {
        String::new()
    };
    if !(-4..16).contains(&exponent) {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push_str(&format!(
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        ));
        return out;
    }
    let point = exponent + 1; // Digits before the decimal point.
    if point <= 0 {
        out.push_str("0.");
        out.push_str(&"0".repeat((-point) as usize));
        out.push_str(&digits);
    } else if point as usize >= digits.len() {
        out.push_str(&digits);
        out.push_str(&"0".repeat(point as usize - digits.len()));
        out.push_str(".0");
    } else {
        out.push_str(&digits[..point as usize]);
        out.push('.');
        out.push_str(&digits[point as usize..]);
    }
    out
}

fn python_number(value: f64) -> String {
    const EXACT_INTEGERS: f64 = 9_007_199_254_740_992.0; // 2^53
    if value.trunc() == value
        && value.abs() <= EXACT_INTEGERS
        && !(value == 0.0 && value.is_sign_negative())
    {
        return (value as i64).to_string();
    }
    python_float(value)
}

fn python_quote(s: &str) -> String {
    let has_single = s.contains('\'');
    let has_double = s.contains('"');
    let quote = if has_single && !has_double {
        b'"'
    } else {
        b'\''
    };
    let mut out = vec![quote];
    for &c in s.as_bytes() {
        match c {
            b'\\' => out.extend_from_slice(b"\\\\"),
            _ if c == quote => {
                out.push(b'\\');
                out.push(c);
            }
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            _ if c < 0x20 || c == 0x7F => out.extend_from_slice(format!("\\x{c:02x}").as_bytes()),
            _ => out.push(c),
        }
    }
    out.push(quote);
    // Only ASCII bytes were inserted around the UTF-8 input.
    String::from_utf8(out).unwrap_or_default()
}

fn python_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => python_number(*n),
        Value::String(s) => python_quote(s),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(python_repr).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(members) => {
            let parts: Vec<String> = members
                .iter()
                .map(|(key, item)| format!("{}: {}", python_quote(key), python_repr(item)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// Jinja's `| string` filter: Python str().
fn python_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => python_repr(v),
    }
}

fn is_container(v: &Value) -> bool {
    v.is_object() || v.is_array()
}

fn tojson(v: &Value) -> Result<String, String> {
    v.tojson().map_err(|e| e.to_string())
}

fn parse(text: &str) -> Result<Value, String> {
    json::parse_str(text).map_err(|e| e.to_string())
}

fn render_extra_keys(out: &mut String, dict: &Value, handled: &[&str]) -> Result<(), String> {
    let Value::Object(members) = dict else {
        return Ok(());
    };
    for (key, value) in members {
        if handled.contains(&key.as_str()) {
            continue;
        }
        out.push_str(&format!("\n<{key}>"));
        out.push_str(&if is_container(value) {
            tojson(value)?
        } else {
            python_str(value)
        });
        out.push_str(&format!("</{key}>"));
    }
    Ok(())
}

/// The function object the template iterates: `tool.function` when
/// present.
fn tool_function(tool: &ChatTool) -> Result<Value, String> {
    if !tool.definition_json.is_empty() {
        let definition = parse(&tool.definition_json)?;
        if let Some(function) = definition.find("function") {
            return Ok(function.clone());
        }
        return Ok(definition);
    }
    let mut function = Value::object();
    function.set("name", tool.name.as_str());
    if !tool.description.is_empty() {
        function.set("description", tool.description.as_str());
    }
    function.set(
        "parameters",
        parse(if tool.parameters_json.is_empty() {
            "{}"
        } else {
            &tool.parameters_json
        })?,
    );
    Ok(function)
}

fn render_tools(out: &mut String, tools: &[ChatTool]) -> Result<(), String> {
    out.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for t in tools {
        let tool = tool_function(t)?;
        out.push_str("\n<function>\n<name>");
        if let Some(name) = tool.find("name") {
            out.push_str(&python_str(name));
        }
        out.push_str("</name>");
        if let Some(description) = tool.find("description") {
            out.push_str("\n<description>");
            out.push_str(python_strip(&python_str(description)));
            out.push_str("</description>");
        }
        out.push_str("\n<parameters>");
        let parameters = tool.find("parameters");
        if let Some(parameters) = parameters.filter(|p| p.is_object())
            && let Some(Value::Object(properties)) = parameters.find("properties")
        {
            for (param, fields) in properties {
                out.push_str(&format!("\n<parameter>\n<name>{param}</name>"));
                if let Some(ty) = fields.find("type") {
                    out.push_str(&format!("\n<type>{}</type>", python_str(ty)));
                }
                if let Some(description) = fields.find("description") {
                    out.push_str("\n<description>");
                    out.push_str(python_strip(&python_str(description)));
                    out.push_str("</description>");
                }
                if let Some(values) = fields.find("enum") {
                    out.push_str(&format!("\n<enum>{}</enum>", tojson(values)?));
                }
                render_extra_keys(out, fields, &["name", "type", "description", "enum"])?;
                out.push_str("\n</parameter>");
            }
        }
        if let Some(parameters) = parameters {
            render_extra_keys(out, parameters, &["type", "properties", "required"])?;
            if let Some(required) = parameters.find("required") {
                out.push_str(&format!("\n<required>{}</required>", tojson(required)?));
            }
        }
        out.push_str("\n</parameters>");
        render_extra_keys(out, &tool, &["type", "name", "description", "parameters"])?;
        out.push_str("\n</function>");
    }
    out.push_str(TOOL_INSTRUCTIONS);
    Ok(())
}

/// The template's `args_value | string` for scalars; containers keep
/// tojson.
fn argument_text(argument: &ChatToolArgument) -> Result<String, String> {
    if argument.is_string {
        return Ok(argument.value.clone());
    }
    let value = parse(&argument.value)?;
    if is_container(&value) {
        tojson(&value)
    } else {
        Ok(python_str(&value))
    }
}

/// The range of `text` after the last `token`, or all of it.
fn after_last(text: &str, range: Range<usize>, token: &str) -> Range<usize> {
    match text[range.clone()].rfind(token) {
        Some(pos) => range.start + pos + token.len()..range.end,
        None => range,
    }
}

/// The range of `text` before the first `token`, or all of it.
fn before_first(text: &str, range: Range<usize>, token: &str) -> Range<usize> {
    match text[range.clone()].find(token) {
        Some(pos) => range.start..range.start + pos,
        None => range,
    }
}

/// [`python_strip`] of `text[range]`, as a range of `text`.
fn strip_range(text: &str, range: Range<usize>) -> Range<usize> {
    let slice = &text[range.clone()];
    let stripped = python_strip(slice);
    let start = range.start + (stripped.as_ptr() as usize - slice.as_ptr() as usize);
    start..start + stripped.len()
}

/// Rendered text with the ranges of client text in it.
#[derive(Default)]
struct Rendered {
    text: String,
    spans: Vec<ContentSpan>,
}

impl Rendered {
    fn framing(&mut self, s: &str) {
        self.text.push_str(s);
    }

    fn content(&mut self, s: &str) {
        if !s.is_empty() {
            self.spans.push(ContentSpan {
                offset: self.text.len(),
                size: s.len(),
            });
        }
        self.text.push_str(s);
    }

    /// Appends `source.text[slice]` with its client ranges.
    fn append(&mut self, source: &Rendered, slice: Range<usize>) {
        let (begin, end) = (slice.start, slice.end);
        for span in &source.spans {
            let from = span.offset.max(begin);
            let to = (span.offset + span.size).min(end);
            if from < to {
                self.spans.push(ContentSpan {
                    offset: self.text.len() + from - begin,
                    size: to - from,
                });
            }
        }
        self.text.push_str(&source.text[slice]);
    }

    fn all(&self) -> Range<usize> {
        0..self.text.len()
    }
}

/// A user, system or tool message's content: the template's render_content
/// puts `<image>` where each image part sits and, when the message holds
/// several, labels them "<image 1>", "<image 2>", ... Client text that
/// spells `<image>` stays text ([`tokenize_chat`]), so the template's rule
/// for prompts that place images by that spelling does not apply.
fn render_content(out: &mut Rendered, m: &ChatMessage) {
    let content = m.content.as_str();
    let count = m.image_offsets.len();
    let mut at = 0;
    for (k, &offset) in m.image_offsets.iter().enumerate() {
        let offset = offset.clamp(at, content.len());
        out.content(content.get(at..offset).unwrap_or(""));
        if count > 1 {
            out.framing(&format!("<image {}>", k + 1));
        }
        out.framing("<image>");
        at = offset;
    }
    out.content(content.get(at..).unwrap_or(""));
}

fn render_assistant(out: &mut Rendered, m: &ChatMessage, truncate: bool) -> Result<(), String> {
    let mut content = Rendered::default();
    if !python_strip(&m.reasoning).is_empty() {
        content.framing(THINK_OPEN);
        content.framing("\n");
        content.content(&m.reasoning);
        content.framing(THINK_CLOSE);
    } else if !m.content.contains(THINK_OPEN) && !m.content.contains(THINK_CLOSE) {
        content.framing(EMPTY_THINK);
    }
    content.content(&m.content);
    let text = content.text.as_str();
    out.framing(IM_START);
    out.framing("assistant\n");
    if !m.tool_calls.is_empty() {
        if !python_strip(text).is_empty() {
            if !truncate {
                out.append(&content, strip_range(text, content.all()));
                out.framing("\n");
            } else {
                let mut c = content.all();
                if text.contains(THINK_CLOSE) {
                    c = after_last(text, c, THINK_CLOSE);
                } else if text.contains(THINK_OPEN) {
                    c = before_first(text, c, THINK_OPEN);
                }
                out.framing(EMPTY_THINK);
                out.append(&content, c);
                out.framing("\n");
            }
        } else {
            out.framing(EMPTY_THINK);
        }
        for call in &m.tool_calls {
            out.framing(&format!("<tool_call>\n<function={}>\n", call.name));
            for argument in &call.arguments {
                out.framing(&format!("<parameter={}>\n", argument.name));
                out.content(&argument_text(argument)?);
                out.framing("\n</parameter>\n");
            }
            out.framing("</function>\n</tool_call>\n");
        }
        out.framing(IM_END);
        return Ok(());
    }
    if !truncate {
        out.append(&content, strip_range(text, content.all()));
        out.framing(IM_END);
        return Ok(());
    }
    let mut c = Rendered::default();
    if text.contains(THINK_OPEN) && text.contains(THINK_CLOSE) {
        c.framing(EMPTY_THINK);
        c.append(&content, after_last(text, content.all(), THINK_CLOSE));
    } else {
        c.append(&content, content.all());
    }
    let stripped = strip_range(&c.text, c.all());
    out.append(&c, stripped);
    out.framing(IM_END);
    Ok(())
}

fn role_name(role: ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
        ChatRole::Tool => "tool",
    }
}

/// Python `str.strip()`, as Jinja's `trim` filter applies it: Unicode
/// White_Space and the four ASCII information separators (`PythonStrip`).
pub fn python_strip(value: &str) -> &str {
    let whitespace = |c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c);
    value.trim_matches(whitespace)
}

/// Renders the artifact's chat template (n5h_5p5_nanov2), with the client
/// text's ranges in ascending order (`RenderChat`). Fails for malformed tool
/// schemas.
pub fn render_chat(
    messages: &[ChatMessage],
    tools: &[ChatTool],
    options: &ChatOptions,
) -> Result<(String, Vec<ContentSpan>), String> {
    let mut out = Rendered::default();
    let mut system_message = "";
    let mut list = messages;
    if let Some(first) = messages.first()
        && first.role == ChatRole::System
    {
        system_message = &first.content;
        list = &messages[1..];
    }
    let last_user = list.iter().rposition(|m| m.role == ChatRole::User);

    // The template always defines system_message, so every prompt opens with
    // a (possibly empty) system turn.
    out.framing(IM_START);
    out.framing("system\n");
    out.content(system_message);
    if !tools.is_empty() {
        if !system_message.is_empty() {
            out.framing("\n\n");
        }
        render_tools(&mut out.text, tools)?;
    }
    out.framing(IM_END);

    for (i, m) in list.iter().enumerate() {
        let truncate = options.truncate_history_thinking && last_user.is_some_and(|u| i < u);
        match m.role {
            ChatRole::Assistant => render_assistant(&mut out, m, truncate)?,
            ChatRole::User | ChatRole::System => {
                out.framing(IM_START);
                out.framing(role_name(m.role));
                out.framing("\n");
                render_content(&mut out, m);
                if m.role == ChatRole::User && last_user == Some(i) && options.medium_effort {
                    out.framing("\n\n{reasoning effort: efficient}");
                }
                out.framing(IM_END);
            }
            ChatRole::Tool => {
                if i > 0 && list[i - 1].role != ChatRole::Tool {
                    out.framing(IM_START);
                    out.framing("user\n");
                }
                out.framing("<tool_response>\n");
                render_content(&mut out, m);
                out.framing("\n</tool_response>\n");
                let last = i + 1 == list.len();
                if last || list[i + 1].role != ChatRole::Tool {
                    out.framing(IM_END);
                }
            }
        }
    }
    if options.add_generation_prompt {
        out.framing(IM_START);
        out.framing(if options.enable_thinking {
            "assistant\n<think>\n"
        } else {
            "assistant\n<think></think>"
        });
    }
    Ok((out.text, out.spans))
}

/// Tokenizes a rendered prompt with its framing's control tokens
/// (`TokenizeChat`). Client text that spells a control token (a literal
/// `<|im_end|>`) is encoded as text; only such ranges are encoded apart, so
/// every other prompt keeps the token IDs of encoding the whole text, as
/// llama.cpp does.
pub fn tokenize_chat(
    tokenizer: &Tokenizer,
    rendered: &str,
    content_spans: &[ContentSpan],
) -> Vec<TokenId> {
    let bytes = rendered.as_bytes();
    let mut tokens = Vec::new();
    let mut append = |text: &[u8], parse_control: bool| {
        tokens.extend(tokenizer.encode(text, parse_control));
    };
    let mut cursor = 0;
    for span in content_spans {
        let content = &bytes[span.offset..span.offset + span.size];
        if !tokenizer.spells_control(content) {
            continue;
        }
        append(&bytes[cursor..span.offset], true);
        append(content, false);
        cursor = span.offset + span.size;
    }
    append(&bytes[cursor..], true);
    tokens
}

#[cfg(test)]
mod tests {
    use super::{python_float, python_number, python_quote};

    #[test]
    fn python_reprs() {
        assert_eq!(python_number(3.0), "3");
        assert_eq!(python_number(-0.0), "-0.0");
        assert_eq!(python_float(0.5), "0.5");
        assert_eq!(python_float(1e-5), "1e-05");
        assert_eq!(python_float(1.5e16), "1.5e+16");
        assert_eq!(python_float(123.25), "123.25");
        assert_eq!(python_float(0.0001), "0.0001");
        assert_eq!(python_float(1e22), "1e+22");
        assert_eq!(python_quote("it's"), "\"it's\"");
        assert_eq!(python_quote("a\nb\u{7f}"), "'a\\nb\\x7f'");
    }
}
