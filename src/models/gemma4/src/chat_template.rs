//! The compiled renderer of Unsloth's Gemma 4 chat template
//! (`chat_template.cpp`). The GGUF template is only accepted when its
//! SHA-256 matches a qualified revision.
//!
//! Known, documented approximations of the Jinja source: message content
//! parts arrive flattened by the HTTP layer, JSON numbers render as Python
//! ints when integral (Python would print 5.0 for a float literal), text
//! parts between two images are trimmed as one segment, a string `const` in a
//! tool schema is shown as the equivalent one-value `enum` (the Jinja source
//! drops `const`), and a replayed thought ends directly at `<channel|>`. The
//! Jinja source adds a newline there that Gemma does not generate, so a
//! replayed tool turn could not reuse its cached tokens.

use gufo_core::gguf::Gguf;
use gufo_core::json::{self, Value};
use gufo_core::text::{
    ChatMessage, ChatRole, ChatTool, ReasoningEffort, ReasoningOptions, ToolCall,
};
use sha2::{Digest, Sha256};

/// Reasoning channel markup produced by the model.
pub const CHANNEL_OPEN: &str = "<|channel>";
pub const THOUGHT_START: &str = "<|channel>thought\n";
pub const THOUGHT_END: &str = "<channel|>";
pub const TOOL_CALL_START: &str = "<|tool_call>";
pub const TOOL_CALL_END: &str = "<tool_call|>";
/// Rendered in place of each user image; the prompt builder replaces it with
/// the image's soft tokens. [`render`] reports the byte offset of each one it
/// emits through `image_offsets`; without that output images are rejected.
pub const IMAGE_PLACEHOLDER: &str = "<|image|>";
/// String delimiter of the tool-call argument syntax.
pub const STRING_QUOTE: &str = "<|\"|>";

/// SHA-256 of the qualified Unsloth template revision.
pub const UNSLOTH_TEMPLATE_SHA256: &str =
    "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b";
/// The template's identity.
pub const TEMPLATE_ID: &str = "gemma4-unsloth-845f1ee4";

const Q: &str = STRING_QUOTE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatOptions {
    pub add_generation_prompt: bool,
    /// Gemma 4 thinking is opt-in: the template default is off.
    pub enable_thinking: bool,
    pub preserve_thinking: bool,
    /// Serving only; the official template adds nothing here. With thinking
    /// off and a tool response last, close an empty thought block in the
    /// generation prompt: after an image result the model otherwise writes
    /// the block without its opening token, and it reaches the reply as text.
    pub close_thought_after_tool: bool,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: false,
            preserve_thinking: false,
            close_thought_after_tool: false,
        }
    }
}

/// Resolves provider-neutral controls against the Gemma 4 template defaults
/// (`ResolveChatOptions`). Effort levels have no Gemma equivalent; any level
/// other than minimal enables thinking when it was not set explicitly.
pub fn resolve_chat_options(reasoning: &ReasoningOptions) -> ChatOptions {
    let mut options = ChatOptions::default();
    if let Some(enabled) = reasoning.enabled {
        options.enable_thinking = enabled;
    } else if let Some(effort) = reasoning.effort {
        options.enable_thinking = effort != ReasoningEffort::Minimal;
    }
    options.preserve_thinking = reasoning.preserve_thinking.unwrap_or(false);
    options.close_thought_after_tool = true;
    options
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RenderedPrompt {
    pub text: String,
    /// Byte offset where the generation prompt starts; equals `text.len()`
    /// without one. Everything before it is the stable conversation prefix.
    pub generation_prompt_offset: usize,
    /// Byte offset of the prefix the next request is expected to share: the
    /// generation prompt, or, when the prompt ends with a text-only user turn
    /// after a tool result or another user turn, the end of the turn before
    /// it. Agent clients send per-turn context there and replace it on the
    /// next request. A tool result stays inside its model turn, which the
    /// next request continues, so the boundary precedes that turn's closing
    /// marker.
    pub stable_prefix_offset: usize,
}

/// Whether the GGUF's `tokenizer.chat_template` is the qualified revision
/// (`ValidateGgufTemplate`).
pub fn validate_gguf_template(gguf: &Gguf<'_>) -> Result<(), String> {
    let source = gguf
        .string("tokenizer.chat_template")
        .ok_or("gemma4 GGUF has no tokenizer.chat_template")?;
    let digest = sha256_hex(source);
    if digest != UNSLOTH_TEMPLATE_SHA256 {
        return Err(format!(
            "unqualified gemma4 chat template (sha256 {digest})"
        ));
    }
    Ok(())
}

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ---------------------------------------------------------------------------
// Python/Jinja value semantics over gufo_core::json. `None` is Jinja's
// Undefined; a JSON null is Python's None.
// ---------------------------------------------------------------------------

fn get<'v>(object: Option<&'v Value>, key: &str) -> Option<&'v Value> {
    object.filter(|v| v.is_object()).and_then(|v| v.find(key))
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => *n != 0.0,
        Some(Value::String(s)) => !s.is_empty(),
        Some(other) => !other.is_empty(),
    }
}

/// Python's repr/str of a float ('short' style): shortest round-trip digits,
/// scientific notation when the decimal exponent is below -4 or at least 16.
fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let sci = format!("{:e}", x.abs()); // d[.ddd]e[-]x
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let mut out = String::from(if x.is_sign_negative() { "-" } else { "" });
    if !(-4..16).contains(&exponent) {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push_str(&format!(
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.unsigned_abs()
        ));
        return out;
    }
    if exponent < 0 {
        out.push_str("0.");
        out.push_str(&"0".repeat((-exponent - 1) as usize));
        out.push_str(&digits);
        return out;
    }
    let int_digits = exponent as usize + 1;
    if digits.len() <= int_digits {
        out.push_str(&digits);
        out.push_str(&"0".repeat(int_digits - digits.len()));
        out.push_str(".0");
        return out;
    }
    out.push_str(&digits[..int_digits]);
    out.push('.');
    out.push_str(&digits[int_digits..]);
    out
}

fn py_number(x: f64) -> String {
    // gufo_core::json keeps numbers as doubles; integral values print as
    // ints.
    if x.is_finite() && x.floor() == x && x.abs() < 9.0e15 {
        return (x as i64).to_string();
    }
    py_float(x)
}

fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python repr(), used for containers and their members.
fn py_repr(v: &Value) -> String {
    match v {
        Value::String(s) => py_str_repr(s),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(members) => {
            let parts: Vec<String> = members
                .iter()
                .map(|(key, value)| format!("{}: {}", py_str_repr(key), py_repr(value)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        other => py_str(other),
    }
}

/// Python str(), which Jinja uses for `{{ value }}`.
fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => py_number(*n),
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// `{{ x }}` of a possibly undefined value.
fn text(v: Option<&Value>) -> String {
    v.map(py_str).unwrap_or_default()
}

/// Jinja `| upper` of a possibly undefined value.
fn upper(v: Option<&Value>) -> String {
    text(v).to_uppercase()
}

/// Jinja `dictsort`: stable, case-insensitive key order.
fn dict_sort(object: &Value) -> Vec<(&String, &Value)> {
    let mut items: Vec<(String, &String, &Value)> = object
        .members()
        .iter()
        .map(|(key, value)| (key.to_lowercase(), key, value))
        .collect();
    items.sort_by(|a, b| a.0.cmp(&b.0));
    items
        .into_iter()
        .map(|(_, key, value)| (key, value))
        .collect()
}

/// Iteration of a Jinja value: list items, mapping keys or string chars.
fn iterate(v: &Value) -> Vec<Value> {
    match v {
        Value::Array(items) => items.clone(),
        Value::Object(members) => members
            .iter()
            .map(|(key, _)| Value::String(key.clone()))
            .collect(),
        Value::String(s) => s.chars().map(|c| Value::String(c.to_string())).collect(),
        _ => Vec::new(),
    }
}

fn is_python_space(c: char) -> bool {
    let c = c as u32;
    (0x09..=0x0D).contains(&c)
        || (0x1C..=0x20).contains(&c)
        || c == 0x85
        || c == 0xA0
        || c == 0x1680
        || (0x2000..=0x200A).contains(&c)
        || c == 0x2028
        || c == 0x2029
        || c == 0x202F
        || c == 0x205F
        || c == 0x3000
}

/// Python `str.strip()` with no arguments.
pub fn py_trim(s: &str) -> &str {
    s.trim_matches(is_python_space)
}

fn strip_thinking(text: &str) -> String {
    let mut result = String::new();
    let mut rest = text;
    loop {
        let split = rest.find(THOUGHT_END);
        let part = &rest[..split.unwrap_or(rest.len())];
        result.push_str(&part[..part.find(CHANNEL_OPEN).unwrap_or(part.len())]);
        let Some(split) = split else {
            break;
        };
        rest = &rest[split + THOUGHT_END.len()..];
    }
    py_trim(&result).to_owned()
}

// ---------------------------------------------------------------------------
// Template macros.
// ---------------------------------------------------------------------------

fn format_argument(v: &Value, escape_keys: bool, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::String(s) => {
            out.push_str(Q);
            out.push_str(s);
            out.push_str(Q);
        }
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Object(_) => {
            out.push('{');
            for (i, (key, value)) in dict_sort(v).into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                if escape_keys {
                    out.push_str(Q);
                    out.push_str(key);
                    out.push_str(Q);
                } else {
                    out.push_str(key);
                }
                out.push(':');
                format_argument(value, escape_keys, out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                format_argument(item, escape_keys, out);
            }
            out.push(']');
        }
        Value::Number(_) => out.push_str(&py_str(v)),
    }
}

fn quoted_list(list: &Value, out: &mut String) {
    for (i, item) in iterate(list).iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(Q);
        out.push_str(&py_str(item));
        out.push_str(Q);
    }
}

/// The template shows a string's allowed values only through `enum` and
/// drops `const`. A single-value enum is the same constraint in the form it
/// shows; without it Gemma 4 does not know the required value (gufo tool
/// grammar).
fn show_constants(schema: &Value) -> Value {
    if !schema.is_object() {
        return schema.clone();
    }
    let mut shown = Value::object();
    for (key, value) in schema.members() {
        if key == "properties" && value.is_object() {
            let mut properties = Value::object();
            for (name, property) in value.members() {
                properties.append_member(name.clone(), show_constants(property));
            }
            shown.append_member(key.clone(), properties);
        } else if key == "items" || key == "parameters" {
            shown.append_member(key.clone(), show_constants(value));
        } else {
            shown.append_member(key.clone(), value.clone());
        }
    }
    if let Some(constant @ Value::String(_)) = schema.find("const")
        && !schema.contains("enum")
        && upper(schema.find("type")) == "STRING"
    {
        shown.set("enum", Value::Array(vec![constant.clone()]));
    }
    shown
}

fn format_parameters(properties: &Value, filter_keys: bool, out: &mut String) {
    const STANDARD_KEYS: [&str; 5] = ["description", "type", "properties", "required", "nullable"];
    let mut found_first = false;
    for (key, value) in dict_sort(properties) {
        if filter_keys && STANDARD_KEYS.contains(&key.as_str()) {
            continue;
        }
        let mut add_comma = false;
        let comma = |add_comma: &mut bool, out: &mut String| {
            if *add_comma {
                out.push(',');
            } else {
                *add_comma = true;
            }
        };
        if found_first {
            out.push(',');
        }
        found_first = true;
        out.push_str(key);
        out.push_str(":{");
        if let Some(description) = get(Some(value), "description")
            && truthy(Some(description))
        {
            out.push_str("description:");
            out.push_str(Q);
            out.push_str(&py_str(description));
            out.push_str(Q);
            add_comma = true;
        }
        let ty = upper(get(Some(value), "type"));
        if ty == "STRING" {
            if let Some(values) = get(Some(value), "enum")
                && truthy(Some(values))
            {
                comma(&mut add_comma, out);
                out.push_str("enum:");
                format_argument(values, true, out);
            }
        } else if ty == "ARRAY"
            && let Some(items) = get(Some(value), "items")
            && items.is_object()
            && truthy(Some(items))
        {
            comma(&mut add_comma, out);
            out.push_str("items:{");
            let mut items_first = false;
            for (item_key, item_value) in dict_sort(items) {
                if item_value.is_null() {
                    continue;
                }
                if items_first {
                    out.push(',');
                }
                items_first = true;
                match item_key.as_str() {
                    "properties" => {
                        out.push_str("properties:{");
                        if item_value.is_object() {
                            format_parameters(item_value, false, out);
                        }
                        out.push('}');
                    }
                    "required" => {
                        out.push_str("required:[");
                        quoted_list(item_value, out);
                        out.push(']');
                    }
                    "type" => {
                        out.push_str("type:");
                        if let Value::String(s) = item_value {
                            format_argument(&Value::String(s.to_uppercase()), true, out);
                        } else {
                            let upper: Vec<Value> = iterate(item_value)
                                .iter()
                                .map(|t| Value::String(py_str(t).to_uppercase()))
                                .collect();
                            format_argument(&Value::Array(upper), true, out);
                        }
                    }
                    _ => {
                        out.push_str(item_key);
                        out.push(':');
                        format_argument(item_value, true, out);
                    }
                }
            }
            out.push('}');
        }
        if truthy(get(Some(value), "nullable")) {
            comma(&mut add_comma, out);
            out.push_str("nullable:true");
        }
        if ty == "OBJECT" {
            match get(Some(value), "properties") {
                Some(nested) if nested.is_object() => {
                    comma(&mut add_comma, out);
                    out.push_str("properties:{");
                    format_parameters(nested, false, out);
                    out.push('}');
                }
                _ if value.is_object() => {
                    comma(&mut add_comma, out);
                    out.push_str("properties:{");
                    format_parameters(value, true, out);
                    out.push('}');
                }
                _ => {}
            }
            if let Some(required) = get(Some(value), "required")
                && truthy(Some(required))
            {
                comma(&mut add_comma, out);
                out.push_str("required:[");
                quoted_list(required, out);
                out.push(']');
            }
        }
        comma(&mut add_comma, out);
        out.push_str("type:");
        out.push_str(Q);
        out.push_str(&ty);
        out.push_str(Q);
        out.push('}');
    }
}

fn format_function_declaration(tool: &Value) -> String {
    let function = get(Some(tool), "function");
    let mut out = format!(
        "declaration:{}{{description:{Q}{}{Q}",
        text(get(function, "name")),
        text(get(function, "description"))
    );
    if let Some(params) = get(function, "parameters")
        && truthy(Some(params))
    {
        out.push_str(",parameters:{");
        if let Some(properties) = get(Some(params), "properties")
            && truthy(Some(properties))
            && properties.is_object()
        {
            out.push_str("properties:{");
            format_parameters(properties, false, &mut out);
            out.push_str("},");
        }
        if let Some(required) = get(Some(params), "required")
            && truthy(Some(required))
        {
            out.push_str("required:[");
            quoted_list(required, &mut out);
            out.push_str("],");
        }
        if let Some(ty) = get(Some(params), "type")
            && truthy(Some(ty))
        {
            out.push_str(&format!("type:{Q}{}{Q}}}", upper(Some(ty))));
        }
    }
    if let Some(function) = function.filter(|f| f.is_object())
        && function.contains("response")
    {
        let response = function.find("response");
        out.push_str(",response:{");
        if let Some(description) = get(response, "description")
            && truthy(Some(description))
        {
            out.push_str(&format!("description:{Q}{}{Q},", py_str(description)));
        }
        if upper(get(response, "type")) == "OBJECT" {
            out.push_str(&format!("type:{Q}OBJECT{Q}}}"));
        }
    }
    out.push('}');
    out
}

fn tool_response_block(name: &str, response: &Value) -> String {
    let mut out = String::from("<|tool_response>");
    if response.is_object() {
        out.push_str(&format!("response:{name}{{"));
        for (i, (key, value)) in dict_sort(response).into_iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(key);
            out.push(':');
            format_argument(value, false, &mut out);
        }
        out.push('}');
    } else {
        out.push_str(&format!("response:{name}{{value:"));
        format_argument(response, false, &mut out);
        out.push('}');
    }
    out.push_str("<tool_response|>");
    out
}

fn role_name(role: ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::Developer => "developer",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
        ChatRole::Tool => "tool",
    }
}

fn arguments_object(call: &ToolCall) -> Value {
    let mut object = Value::object();
    for argument in &call.arguments {
        let value = if argument.is_string {
            Value::String(argument.value.clone())
        } else {
            json::parse_str(&argument.value)
                .unwrap_or_else(|_| Value::String(argument.value.clone()))
        };
        object.set(&argument.name, value);
    }
    object
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MessageType {
    None,
    Think,
    Tool,
    ToolCall,
    ToolResponse,
}

/// Renders a conversation (`ChatTemplate::Render`). `image_offsets`
/// receives the byte offset of each image placeholder; without it images are
/// rejected.
pub fn render(
    messages: &[ChatMessage],
    tools: &[ChatTool],
    options: &ChatOptions,
    mut image_offsets: Option<&mut Vec<usize>>,
) -> Result<RenderedPrompt, String> {
    if let Some(offsets) = image_offsets.as_deref_mut() {
        offsets.clear();
    }
    // Tool results render only after the assistant turn that made the calls.
    let mut tool_results = false;
    for message in messages {
        if message.role != ChatRole::Tool {
            tool_results = message.role == ChatRole::Assistant && !message.tool_calls.is_empty();
        }
        if message.images.is_empty() {
            continue;
        }
        if image_offsets.is_none() {
            return Err("gemma4 image input requires a vision sidecar".into());
        }
        if message.role == ChatRole::Tool && !tool_results {
            return Err("tool result images require a preceding tool call".into());
        }
        let mut previous = 0;
        for image in &message.images {
            if !matches!(message.role, ChatRole::User | ChatRole::Tool)
                || image.offset < previous
                || image.offset > message.content.len()
                || !message.content.is_char_boundary(image.offset)
            {
                return Err("invalid user image content".into());
            }
            previous = image.offset;
        }
    }
    // "call:NAME{" and "declaration:NAME{" end a function name at its first
    // brace, so a name holding one cannot be framed.
    const BRACE_NAME: &str = "gemma4 function names cannot contain '{'";
    if messages
        .iter()
        .any(|m| m.name.contains('{') || m.tool_calls.iter().any(|c| c.name.contains('{')))
    {
        return Err(BRACE_NAME.into());
    }
    let mut tool_data = Vec::with_capacity(tools.len());
    for tool in tools {
        if tool.name.contains('{') {
            return Err(BRACE_NAME.into());
        }
        let definition = if tool.definition_json.is_empty() {
            "{}"
        } else {
            tool.definition_json.as_str()
        };
        let mut data = json::parse_str(definition)
            .map_err(|e| format!("invalid tool definition: {}", e.message()))?;
        if !data.is_object() {
            return Err("tool definitions must be objects".into());
        }
        if let Some(function) = data.find("function").filter(|f| f.is_object()) {
            let shown = show_constants(function);
            data.set("function", shown);
        }
        tool_data.push(data);
    }

    let mut out = String::from("<bos>");
    let mut prev_type = MessageType::None;
    let mut prev_non_tool_role = "";
    let first_is_system = messages
        .first()
        .is_some_and(|m| matches!(m.role, ChatRole::System | ChatRole::Developer));
    let mut lp = messages;
    if options.enable_thinking || !tools.is_empty() || first_is_system {
        out.push_str("<|turn>system\n");
        if options.enable_thinking {
            out.push_str("<|think|>\n");
            prev_type = MessageType::Think;
        }
        if first_is_system {
            // Server-authored instructions follow the client's system text.
            let system = format!("{}{}", messages[0].content, messages[0].framing_suffix);
            out.push_str(py_trim(&system));
            lp = &messages[1..];
        }
        for tool in &tool_data {
            out.push_str("<|tool>");
            out.push_str(py_trim(&format_function_declaration(tool)));
            out.push_str("<tool|>");
            prev_type = MessageType::Tool;
        }
        out.push_str("<turn|>\n");
    }

    let last_user = lp.iter().rposition(|m| m.role == ChatRole::User);

    // Per-turn context an agent replaces on every request follows a tool
    // result or another user turn, never an assistant reply (that is
    // ordinary chat, which the next request keeps).
    let final_user_turn = last_user.is_some_and(|last| {
        last >= 1
            && last + 1 == lp.len()
            && lp[last].images.is_empty()
            && matches!(lp[last - 1].role, ChatRole::Tool | ChatRole::User)
    });
    let mut seen_assistant = false;
    let mut final_user_start = None;
    let mut turn_end = 0;

    for (index, message) in lp.iter().enumerate() {
        if message.role == ChatRole::Tool {
            continue;
        }
        if message.role == ChatRole::Assistant {
            seen_assistant = true;
        } else if final_user_turn && seen_assistant && Some(index) == last_user {
            final_user_start = Some(if lp[index - 1].role == ChatRole::Tool {
                turn_end
            } else {
                out.len()
            });
        }
        prev_type = MessageType::None;
        let raw_role = role_name(message.role);
        let role = if message.role == ChatRole::Assistant {
            "model"
        } else {
            raw_role
        };
        let continue_turn = role == "model" && prev_non_tool_role == "assistant";
        if !continue_turn {
            out.push_str("<|turn>");
            out.push_str(role);
            out.push('\n');
        }

        let thinking_gate = last_user.is_none_or(|last| index > last)
            || (options.preserve_thinking && !message.tool_calls.is_empty());
        if !message.thought.is_empty() && thinking_gate {
            out.push_str(THOUGHT_START);
            out.push_str(&message.thought);
            out.push_str(THOUGHT_END);
        }

        if !message.tool_calls.is_empty() {
            for call in &message.tool_calls {
                out.push_str(TOOL_CALL_START);
                out.push_str("call:");
                out.push_str(&call.name);
                out.push('{');
                let arguments = arguments_object(call);
                for (i, (key, value)) in dict_sort(&arguments).into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(key);
                    out.push(':');
                    format_argument(value, false, &mut out);
                }
                out.push('}');
                out.push_str(TOOL_CALL_END);
            }
            prev_type = MessageType::ToolCall;
        }

        let mut tool_responses = false;
        if !message.tool_calls.is_empty() {
            for follow in lp[index + 1..]
                .iter()
                .take_while(|m| m.role == ChatRole::Tool)
            {
                let mut name = if follow.name.is_empty() {
                    "unknown"
                } else {
                    follow.name.as_str()
                };
                for call in &message.tool_calls {
                    if call.id == follow.tool_call_id {
                        name = &call.name;
                    }
                }
                // The text parts form the response; its images follow it in
                // order.
                out.push_str(&tool_response_block(
                    name,
                    &Value::String(follow.content.clone()),
                ));
                for _ in &follow.images {
                    if let Some(offsets) = image_offsets.as_deref_mut() {
                        offsets.push(out.len());
                    }
                    out.push_str(IMAGE_PLACEHOLDER);
                }
                tool_responses = true;
                prev_type = MessageType::ToolResponse;
            }
        }

        let content = if !message.images.is_empty() {
            // A content-part list: every text part is trimmed on its own and
            // each image renders as a placeholder. Adjacent flattened text
            // parts form one segment.
            let mut content = String::new();
            let mut cursor = 0;
            for image in &message.images {
                content.push_str(py_trim(&message.content[cursor..image.offset]));
                if let Some(offsets) = image_offsets.as_deref_mut() {
                    offsets.push(out.len() + content.len());
                }
                content.push_str(IMAGE_PLACEHOLDER);
                cursor = image.offset;
            }
            content.push_str(py_trim(&message.content[cursor..]));
            content
        } else if role == "model" {
            strip_thinking(&message.content)
        } else {
            py_trim(&format!("{}{}", message.content, message.framing_suffix)).to_owned()
        };
        out.push_str(&content);
        let has_content = !py_trim(&content).is_empty();

        let next = lp[index + 1..].iter().find(|m| m.role != ChatRole::Tool);
        let next_role = next.map_or("", |m| role_name(m.role));
        let continues_into_next = role == "model"
            && next_role == "assistant"
            && (message.tool_calls.is_empty() || tool_responses);
        turn_end = out.len();
        if prev_type == MessageType::ToolCall && !tool_responses {
            out.push_str("<|tool_response>");
        } else if continues_into_next {
        } else if !(tool_responses && !has_content && next.is_none()) {
            out.push_str("<turn|>\n");
        }
        prev_non_tool_role = raw_role;
    }

    let generation_prompt_offset = out.len();
    let stable_prefix_offset = final_user_start.unwrap_or(out.len());
    if options.add_generation_prompt {
        if prev_type != MessageType::ToolResponse && prev_type != MessageType::ToolCall {
            out.push_str("<|turn>model\n");
            if !options.enable_thinking {
                out.push_str(THOUGHT_START);
                out.push_str(THOUGHT_END);
            }
        } else if prev_type == MessageType::ToolResponse {
            if options.enable_thinking {
                out.push_str(THOUGHT_START);
            } else if options.close_thought_after_tool {
                out.push_str(THOUGHT_START);
                out.push_str(THOUGHT_END);
            }
        }
    }
    Ok(RenderedPrompt {
        text: out,
        generation_prompt_offset,
        stable_prefix_offset,
    })
}
