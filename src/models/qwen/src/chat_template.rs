//! The Qwen chat formatter, a port of `QwenChatTemplate`
//! (`src/models/qwen/chat_template.cpp`): the official Qwen3.8 template
//! compiled by hand, with its reasoning, tool and image framing, and the
//! pinned template hashes a Qwen3.8 artifact must carry.

use gufo_core::bpe::TokenId;
use gufo_core::gguf::Gguf;
use gufo_core::qwen_tokenizer::QwenTokenizer;
use gufo_core::text::{
    ChatMessage, ChatRequest, ChatRole, ChatTool, Error, ReasoningEffort, ReasoningOptions,
    ToolCall, ToolChoice,
};
use sha2::{Digest, Sha256};

/// SHA-256 of the upstream Qwen3.8 `chat_template.jinja`.
pub const OFFICIAL_TEMPLATE_SHA256: &str =
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041";
/// SHA-256 of the template embedded in the Unsloth GGUF artifacts.
pub const UNSLOTH_ARTIFACT_TEMPLATE_SHA256: &str =
    "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce";

const DEFAULT_CHATML_TEMPLATE: &str = "{% for message in messages %}{{'<|im_start|>' + \
     message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}{% endfor %}{% if \
     add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

const LOW_REASONING_INSTRUCTION: &str = "Reasoning effort is set to low. Keep your thinking \
     brief and focused, moving directly to the conclusion without unnecessary elaboration.";

const XHIGH_REASONING_INSTRUCTION: &str = "Reasoning effort is set to xhigh. Please think \
     carefully through the task, validate key assumptions, consider plausible alternatives, \
     and prioritize correctness, consistency, and clarity in the final answer.";

const TOOLS_INSTRUCTION: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in \
     the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n\
     <parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\n\
     This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n\
     </function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the \
     specified format: an inner <function=...></function> block must be nested within \
     <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may \
     provide optional reasoning for your function call in natural language BEFORE the \
     function call, but NOT after\n- If there is no function call available, answer the \
     question like normal with your current knowledge and do not tell the user about \
     function calls\n</IMPORTANT>";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QwenReasoningEffort {
    Low,
    Medium,
    XHigh,
}

/// Formatting options for rendering a conversation into a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatTemplateOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub reasoning_effort: QwenReasoningEffort,
    pub preserve_thinking: bool,
    pub add_vision_id: bool,
    /// Upper bound of the rendered prompt in bytes.
    pub max_output_bytes: usize,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: true,
            reasoning_effort: QwenReasoningEffort::XHigh,
            preserve_thinking: true,
            add_vision_id: false,
            max_output_bytes: 1024 * 1024,
        }
    }
}

/// Resolves CLI and API controls against the official Qwen3.8 template
/// defaults. Provider-neutral minimal, high and max map to the native low,
/// xhigh and xhigh.
pub fn resolve_options(reasoning: &ReasoningOptions, add_vision_id: bool) -> ChatTemplateOptions {
    let defaults = ChatTemplateOptions::default();
    ChatTemplateOptions {
        add_vision_id,
        enable_thinking: reasoning.enabled.unwrap_or(defaults.enable_thinking),
        preserve_thinking: reasoning
            .preserve_thinking
            .unwrap_or(defaults.preserve_thinking),
        reasoning_effort: match reasoning.effort.unwrap_or(ReasoningEffort::XHigh) {
            ReasoningEffort::Minimal | ReasoningEffort::Low => QwenReasoningEffort::Low,
            ReasoningEffort::Medium => QwenReasoningEffort::Medium,
            ReasoningEffort::High | ReasoningEffort::XHigh | ReasoningEffort::Max => {
                QwenReasoningEffort::XHigh
            }
        },
        ..defaults
    }
}

/// The longest Qwen3.8 vocabulary entry in bytes (Flash-Next, 248,320
/// tokens): a prompt of N tokens renders to at most N times this.
pub const MAX_RENDERED_BYTES_PER_TOKEN: usize = 128;

/// The rendered-prompt bound of a session of `context_tokens`: every prompt
/// that fits the context passes, never below the 1 MiB default.
pub fn rendered_prompt_bound_bytes(context_tokens: u32) -> usize {
    (1usize << 20).max(context_tokens as usize * MAX_RENDERED_BYTES_PER_TOKEN)
}

/// The options a serving request selects in a session of `max_context`
/// tokens (`QwenChatOptions` in `src/cli/serve/inference_backend.cpp`).
pub fn request_options(request: &ChatRequest, max_context: u32) -> ChatTemplateOptions {
    ChatTemplateOptions {
        max_output_bytes: rendered_prompt_bound_bytes(max_context),
        ..resolve_options(&request.reasoning, request.add_vision_id)
    }
}

/// The tools a serving request offers the model: none with
/// `tool_choice: none`.
pub fn request_tools(request: &ChatRequest) -> &[ChatTool] {
    if request.tool_choice == ToolChoice::None {
        &[]
    } else {
        &request.tools
    }
}

/// The suffix that opens a new assistant turn, outside the stable
/// conversation.
pub fn generation_prompt(enable_thinking: bool) -> &'static str {
    if enable_thinking {
        "<|im_start|>assistant\n<think>\n"
    } else {
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Profile {
    LegacyChatMl,
    Qwen38Reasoning,
}

/// A model's embedded template string and its classification. Rendering
/// does not interpret the string: the Qwen3.8 formatter is compiled in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplate {
    template: Vec<u8>,
    sha256: String,
    profile: Profile,
}

/// A byte range of a rendered prompt that came from a message.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ContentSpan {
    pub offset: usize,
    pub size: usize,
}

/// A rendered prompt and the byte offset of every `<|image_pad|>`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Rendered {
    pub text: String,
    pub image_offsets: Vec<usize>,
    /// Message text, in order: tokenized as text, never as framing, so a
    /// control token a client spells stays the characters it wrote
    /// ([`encode_rendered`]).
    pub content_spans: Vec<ContentSpan>,
    /// The bytes an interrupted assistant followed by a user turn leaves
    /// unchanged: before the generation prompt, before the current tool
    /// cycle whose reasoning a new user turn removes, or before a final user
    /// turn after a tool result or another user message, which agent clients
    /// replace with each request.
    pub stable_prefix_bytes: usize,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_qwen38_reasoning_template(template: &[u8]) -> bool {
    let hash = sha256_hex(template);
    hash == OFFICIAL_TEMPLATE_SHA256 || hash == UNSLOTH_ARTIFACT_TEMPLATE_SHA256
}

/// Whether the GGUF names a Qwen3.8 model. Architecture crates add their own
/// shape check (`gufo_qwen35::config::is_qwen38_artifact`).
pub fn is_qwen38_named(gguf: &Gguf<'_>) -> bool {
    gguf.string("general.name")
        .is_some_and(|name| name.windows(7).any(|w| w == b"Qwen3.8"))
}

impl ChatTemplate {
    /// The template from `tokenizer.chat_template`. A Qwen3.8 artifact
    /// (`qwen38_artifact`, as the architecture recognizes one) must carry one
    /// of the pinned templates; other models without one fall back to the
    /// default ChatML template.
    pub fn from_gguf(gguf: &Gguf<'_>, qwen38_artifact: bool) -> Result<Self, String> {
        let template = gguf
            .string("tokenizer.chat_template")
            .filter(|t| !t.is_empty());
        let Some(template) = template else {
            if qwen38_artifact {
                return Err("Qwen3.8 GGUF is missing tokenizer.chat_template metadata".into());
            }
            return Ok(Self::default_template());
        };
        let profile = if is_qwen38_reasoning_template(template) {
            Profile::Qwen38Reasoning
        } else {
            Profile::LegacyChatMl
        };
        if profile != Profile::Qwen38Reasoning && qwen38_artifact {
            return Err(format!(
                "Qwen3.8 GGUF chat template SHA-256 is not a recognized pinned version: {}",
                sha256_hex(template)
            ));
        }
        Ok(Self {
            template: template.to_vec(),
            sha256: sha256_hex(template),
            profile,
        })
    }

    /// Checks that a Qwen3.8 artifact carries a pinned template; other Qwen
    /// revisions keep the legacy formatter.
    pub fn validate_gguf(gguf: &Gguf<'_>, qwen38_artifact: bool) -> Result<(), String> {
        if !qwen38_artifact {
            return Ok(());
        }
        Self::from_gguf(gguf, true).map(|_| ())
    }

    /// The default ChatML template.
    pub fn default_template() -> Self {
        Self::from_raw(DEFAULT_CHATML_TEMPLATE.as_bytes())
    }

    /// A template string classified by its hash.
    pub fn from_raw(template: &[u8]) -> Self {
        if template.is_empty() {
            return Self::default_template();
        }
        Self {
            template: template.to_vec(),
            sha256: sha256_hex(template),
            profile: if is_qwen38_reasoning_template(template) {
                Profile::Qwen38Reasoning
            } else {
                Profile::LegacyChatMl
            },
        }
    }

    pub fn template(&self) -> &[u8] {
        &self.template
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// The compiled formatter version, part of the cache identity.
    pub fn id(&self) -> &'static str {
        match self.profile {
            Profile::LegacyChatMl => "qwen-chatml-compiled-v1",
            Profile::Qwen38Reasoning => "qwen38-reasoning-compiled-v3",
        }
    }
}

/// Python `str.strip()`: Unicode White_Space plus the four ASCII
/// information separators.
fn trim(value: &str) -> &str {
    value.trim_matches(is_strip)
}

fn is_strip(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Inserts the space Python's `json.dumps` puts after `:` and `,` outside
/// strings.
fn python_json_spacing(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + value.len() / 8);
    let mut in_string = false;
    let mut escaped = false;
    for character in value.chars() {
        output.push(character);
        if in_string {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }
        if character == '"' {
            in_string = true;
        } else if character == ':' || character == ',' {
            output.push(' ');
        }
    }
    output
}

fn append_reasoning_instruction(output: &mut String, options: &ChatTemplateOptions) {
    if !options.enable_thinking {
        return;
    }
    match options.reasoning_effort {
        QwenReasoningEffort::Low => output.push_str(LOW_REASONING_INSTRUCTION),
        QwenReasoningEffort::Medium => {}
        QwenReasoningEffort::XHigh => output.push_str(XHIGH_REASONING_INSTRUCTION),
    }
}

/// A JSON string that escapes only quotes, backslashes and the five named
/// control characters, as the C++ formatter does.
fn append_json_string(output: &mut String, value: &str) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{8}' => output.push_str("\\b"),
            '\u{c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            other => output.push(other),
        }
    }
    output.push('"');
}

fn append_tools_prompt(output: &mut String, tools: &[ChatTool]) {
    if tools.is_empty() {
        return;
    }
    if !output.is_empty() {
        output.push_str("\n\n");
    }
    output.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for tool in tools {
        if !tool.definition_json.is_empty() {
            output.push('\n');
            output.push_str(&python_json_spacing(&tool.definition_json));
            continue;
        }
        output.push_str("\n{\"type\": \"function\", \"function\": {\"name\": ");
        append_json_string(output, &tool.name);
        output.push_str(", \"description\": ");
        append_json_string(output, &tool.description);
        output.push_str(", \"parameters\": ");
        if tool.parameters_json.is_empty() {
            output.push_str("{}");
        } else {
            output.push_str(&python_json_spacing(&tool.parameters_json));
        }
        output.push_str("}}");
    }
    output.push_str(TOOLS_INSTRUCTION);
}

/// Appends message text to `output`, recording its span.
fn append_content(output: &mut String, spans: &mut Vec<ContentSpan>, text: &str) {
    let offset = output.len();
    output.push_str(text);
    if !text.is_empty() {
        spans.push(ContentSpan {
            offset,
            size: text.len(),
        });
    }
}

/// Appends a message's trimmed content with its images in place
/// (`AppendMessageContent`): images stay inside their
/// own message, tool results included. Only the inserted vision markers are
/// framing; whitespace between text and images stays significant, and the
/// image offsets follow the trim. Whether any content was written.
fn append_message_content(
    output: &mut String,
    message: &ChatMessage,
    options: &ChatTemplateOptions,
    image_count: &mut usize,
    image_offsets: &mut Vec<usize>,
    content_spans: &mut Vec<ContentSpan>,
) -> bool {
    if message.images.is_empty() {
        let content = trim(&message.content);
        let content_offset = output.len();
        output.push_str(content);
        if !content.is_empty() {
            content_spans.push(ContentSpan {
                offset: content_offset,
                size: content.len(),
            });
        }
        return !content.is_empty();
    }
    let mut rendered = String::new();
    let mut spans = Vec::new();
    let mut offsets = Vec::new();
    let mut cursor = 0;
    for image in &message.images {
        append_content(
            &mut rendered,
            &mut spans,
            &message.content[cursor..image.offset],
        );
        *image_count += 1;
        if options.add_vision_id {
            rendered.push_str(&format!("Picture {image_count}: "));
        }
        rendered.push_str("<|vision_start|>");
        offsets.push(rendered.len());
        rendered.push_str("<|image_pad|><|vision_end|>");
        cursor = image.offset;
    }
    append_content(&mut rendered, &mut spans, &message.content[cursor..]);
    let content = trim(&rendered);
    let removed = rendered.len() - rendered.trim_start_matches(is_strip).len();
    let content_offset = output.len();
    for offset in &offsets {
        image_offsets.push(content_offset + offset - removed);
    }
    output.push_str(content);
    // The trim removed `removed` bytes before the first span; spans cut by
    // the trim keep their remaining part.
    let end = content_offset + content.len();
    for span in &spans {
        let begin = (content_offset + span.offset)
            .saturating_sub(removed)
            .max(content_offset);
        let stop = (content_offset + span.offset + span.size)
            .saturating_sub(removed)
            .min(end);
        if stop > begin {
            content_spans.push(ContentSpan {
                offset: begin,
                size: stop - begin,
            });
        }
    }
    !content.is_empty()
}

fn append_tool_calls(output: &mut String, calls: &[ToolCall], spans: &mut Vec<ContentSpan>) {
    for (index, call) in calls.iter().enumerate() {
        if index > 0 {
            output.push('\n');
        }
        output.push_str("<tool_call>\n<function=");
        output.push_str(&call.name);
        output.push_str(">\n");
        for argument in &call.arguments {
            output.push_str("<parameter=");
            output.push_str(&argument.name);
            output.push_str(">\n");
            append_content(output, spans, &argument.value);
            output.push_str("\n</parameter>\n");
        }
        output.push_str("</function>\n</tool_call>");
    }
}

fn is_system(role: ChatRole) -> bool {
    matches!(role, ChatRole::System | ChatRole::Developer)
}

/// Renders messages into the Qwen3.8 prompt. Image offsets must lie on
/// character boundaries of their message, in order.
pub fn render(
    messages: &[ChatMessage],
    tools: &[ChatTool],
    options: &ChatTemplateOptions,
) -> Result<Rendered, String> {
    if messages.is_empty() {
        return Err("No messages provided".into());
    }
    // The reference template accepts one leading system turn, but OpenAI
    // clients such as Codex send system/developer messages mid-conversation.
    // Hoist them, in order, into that turn.
    let hoisted: Vec<ChatMessage>;
    let messages = match messages.iter().position(|m| !is_system(m.role)) {
        Some(first) if messages[first..].iter().any(|m| is_system(m.role)) => {
            hoisted = messages
                .iter()
                .filter(|m| is_system(m.role))
                .chain(messages.iter().filter(|m| !is_system(m.role)))
                .cloned()
                .collect();
            &hoisted[..]
        }
        _ => messages,
    };

    let mut estimated_len = 0usize;
    for message in messages {
        estimated_len += message.content.len()
            + message.framing_suffix.len()
            + message.thought.len()
            + 32
            + message.images.len() * 64;
        let mut previous = 0;
        for image in &message.images {
            if !matches!(message.role, ChatRole::User | ChatRole::Tool)
                || image.offset < previous
                || !message.content.is_char_boundary(image.offset)
            {
                return Err("invalid user or tool image content".into());
            }
            previous = image.offset;
        }
        for call in &message.tool_calls {
            estimated_len += call.name.len() + 64;
            for argument in &call.arguments {
                estimated_len += argument.name.len() + argument.value.len() + 40;
            }
        }
    }
    for tool in tools {
        estimated_len += tool.name.len()
            + tool.description.len()
            + tool.parameters_json.len()
            + 2 * tool.definition_json.len()
            + 96;
    }
    if options.add_generation_prompt {
        estimated_len += 64;
    }
    estimated_len += XHIGH_REASONING_INSTRUCTION.len() + 64;
    if estimated_len > options.max_output_bytes || estimated_len > i32::MAX as usize {
        return Err(format!(
            "Rendered prompt estimated length ({estimated_len} bytes) exceeds maximum bound ({} \
             bytes)",
            options.max_output_bytes
        ));
    }

    let mut output = String::with_capacity(estimated_len);
    let mut image_offsets = Vec::new();
    let mut content_spans = Vec::new();

    let mut index = 0;
    let mut system_content = String::new();
    let mut system_content_spans = Vec::new();
    while index < messages.len() && is_system(messages[index].role) {
        let message = &messages[index];
        // The client's text is content; the server's suffix is framing.
        let untrimmed = format!("{}{}", message.content, message.framing_suffix);
        let content = trim(&untrimmed);
        if !content.is_empty() {
            if !system_content.is_empty() {
                system_content.push('\n');
            }
            let begin = untrimmed.len() - untrimmed.trim_start_matches(is_strip).len();
            let client = message
                .content
                .len()
                .min(begin + content.len())
                .saturating_sub(begin);
            append_content(
                &mut system_content,
                &mut system_content_spans,
                &content[..client],
            );
            system_content.push_str(&content[client..]);
        }
        index += 1;
    }

    let mut system_prefix = String::new();
    append_reasoning_instruction(&mut system_prefix, options);
    append_tools_prompt(&mut system_prefix, tools);
    if !system_prefix.is_empty() && !system_content.is_empty() {
        system_prefix.push_str("\n\n");
    }
    let system_content_offset = system_prefix.len();
    system_prefix.push_str(&system_content);
    if !system_prefix.is_empty() {
        output.push_str("<|im_start|>system\n");
        let prefix_offset = output.len();
        output.push_str(&system_prefix);
        output.push_str("<|im_end|>\n");
        content_spans.extend(system_content_spans.iter().map(|span| ContentSpan {
            offset: prefix_offset + system_content_offset + span.offset,
            size: span.size,
        }));
    }

    // The last user message that is not a wrapped tool response.
    let last_user_index = messages
        .iter()
        .rposition(|message| {
            if message.role != ChatRole::User {
                return false;
            }
            let content = trim(&message.content);
            !(message.images.is_empty()
                && content.starts_with("<tool_response>")
                && content.ends_with("</tool_response>"))
        })
        .ok_or("No user query found in messages.")?;

    let mut image_count = 0usize;
    let mut mutable_reasoning = None;
    // Agent clients may end every request with a user message the next
    // request replaces (per-turn runtime context) instead of keeping: a
    // checkpoint at the end of the prompt then never prefixes the next
    // request. Such context follows a tool result or another user message,
    // not an assistant reply, so only then the checkpoint stays before the
    // final text-only user turn; a client that keeps it prefills it again.
    let final_user_turn = last_user_index + 1 == messages.len()
        && messages.len() >= 2
        && messages[last_user_index].images.is_empty()
        && matches!(
            messages[messages.len() - 2].role,
            ChatRole::Tool | ChatRole::User
        );
    let mut seen_assistant = false;
    let mut final_user_start = None;
    while index < messages.len() {
        let message = &messages[index];
        if message.role == ChatRole::Tool {
            output.push_str("<|im_start|>user\n");
            while index < messages.len() && messages[index].role == ChatRole::Tool {
                output.push_str("<tool_response>\n");
                append_message_content(
                    &mut output,
                    &messages[index],
                    options,
                    &mut image_count,
                    &mut image_offsets,
                    &mut content_spans,
                );
                output.push_str("\n</tool_response>");
                index += 1;
                if index < messages.len() && messages[index].role == ChatRole::Tool {
                    output.push('\n');
                }
            }
            output.push_str("<|im_end|>\n");
            continue;
        }
        // Current tool-cycle assistants keep their reasoning even when older
        // reasoning is dropped; a new user turn removes it, empty <think>
        // framing included. Checkpoint before the first affected assistant.
        if !options.preserve_thinking
            && message.role == ChatRole::Assistant
            && index > last_user_index
            && mutable_reasoning.is_none()
        {
            mutable_reasoning = Some(output.len());
        }
        if message.role == ChatRole::Assistant {
            seen_assistant = true;
        } else if final_user_turn && seen_assistant && index == last_user_index {
            final_user_start = Some(output.len());
        }
        output.push_str("<|im_start|>");
        output.push_str(message.role.as_str());
        output.push('\n');

        if message.role == ChatRole::Assistant
            && (options.preserve_thinking || index > last_user_index)
        {
            output.push_str("<think>\n");
            append_content(&mut output, &mut content_spans, trim(&message.thought));
            output.push_str("\n</think>\n\n");
        }
        let has_content = append_message_content(
            &mut output,
            message,
            options,
            &mut image_count,
            &mut image_offsets,
            &mut content_spans,
        );
        if message.role == ChatRole::Assistant && !message.tool_calls.is_empty() {
            if has_content {
                output.push_str("\n\n");
            }
            append_tool_calls(&mut output, &message.tool_calls, &mut content_spans);
        }
        output.push_str("<|im_end|>\n");

        if output.len() > options.max_output_bytes {
            return Err("Prompt exceeded max output bytes during message rendering".into());
        }
        index += 1;
    }

    let stable_prefix_bytes = mutable_reasoning
        .unwrap_or(output.len())
        .min(final_user_start.unwrap_or(output.len()));
    if options.add_generation_prompt {
        output.push_str(generation_prompt(options.enable_thinking));
    }
    if output.len() > options.max_output_bytes {
        return Err("Prompt exceeded max output bytes after generation prompt".into());
    }
    Ok(Rendered {
        text: output,
        image_offsets,
        content_spans,
        stable_prefix_bytes,
    })
}

/// The tokens of `rendered[begin..end]`: message content that spells an
/// added token is read as text, so a control token a client spelled stays
/// the characters it wrote; everything else, framing and ordinary content,
/// keeps the ordinary reading. Only the spelling spans change, so plain
/// prompts tokenize exactly as before and a slice starting on a token
/// boundary reads like the same bytes of the whole prompt (upstream decides
/// per slice, which breaks the stable prefix once an earlier message spells
/// a token).
pub fn encode_rendered(
    tokenizer: &QwenTokenizer,
    rendered: &str,
    begin: usize,
    end: usize,
    spans: &[ContentSpan],
) -> Vec<TokenId> {
    debug_assert!(
        spans
            .windows(2)
            .all(|pair| pair[0].offset + pair[0].size <= pair[1].offset)
    );
    let mut tokens = Vec::new();
    let mut cursor = begin;
    for span in spans {
        let span_end = span.offset + span.size;
        let (from, to) = (span.offset.max(begin), span_end.min(end));
        // The whole span decides, so every slice reads it alike.
        if from >= to || !tokenizer.spells_special_token(&rendered[span.offset..span_end]) {
            continue;
        }
        if from > cursor {
            tokens.extend(tokenizer.encode(&rendered[cursor..from]));
        }
        tokens.extend(tokenizer.encode_text(&rendered[from..to]));
        cursor = to;
    }
    if cursor < end {
        tokens.extend(tokenizer.encode(&rendered[cursor..end]));
    }
    tokens
}

/// Renders and tokenizes a text-only conversation; added tokens in the
/// rendered prompt are recognised.
pub fn render_and_tokenize(
    tokenizer: &QwenTokenizer,
    messages: &[ChatMessage],
    tools: &[ChatTool],
    options: &ChatTemplateOptions,
) -> Result<Vec<TokenId>, String> {
    if messages.iter().any(|message| !message.images.is_empty()) {
        return Err("image messages require the model's vision prompt preparation".into());
    }
    let rendered = render(messages, tools, options)?;
    Ok(encode_rendered(
        tokenizer,
        &rendered.text,
        0,
        rendered.text.len(),
        &rendered.content_spans,
    ))
}

/// A serving prompt and its snapshot boundary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PreparedPrompt {
    pub tokens: Vec<TokenId>,
    /// The tokens an interrupted assistant followed by a user turn leaves
    /// unchanged ([`Rendered::stable_prefix_bytes`]); zero uses the complete
    /// prompt.
    pub cache_prefix_tokens: usize,
}

/// `PrepareQwenPrompt` (`src/cli/serve/inference_backend.cpp`) for a model
/// without a vision encoder.
pub fn prepare_prompt(
    tokenizer: &QwenTokenizer,
    request: &ChatRequest,
    max_context: u32,
) -> gufo_core::text::Result<PreparedPrompt> {
    let options = request_options(request, max_context);
    let rendered =
        render(&request.messages, request_tools(request), &options).map_err(Error::invalid)?;
    if request
        .messages
        .iter()
        .any(|message| !message.images.is_empty())
    {
        return Err(Error::invalid(
            "image input requires the model's matching mmproj",
        ));
    }
    let tokens = encode_rendered(
        tokenizer,
        &rendered.text,
        0,
        rendered.text.len(),
        &rendered.content_spans,
    );
    if tokens.len() > max_context as usize {
        return Err(crate::vision::context_exceeded(max_context));
    }
    let cache_prefix_tokens = stable_prefix_tokens(tokenizer, &rendered, &tokens)?;
    Ok(PreparedPrompt {
        tokens,
        cache_prefix_tokens,
    })
}

/// The tokens of `rendered.stable_prefix_bytes`. Agent clients may drop an
/// interrupted assistant and append another user turn, with or without
/// thinking and tools. With thinking the generation prompt ends in
/// `<think>\n`; a later turn replaying this assistant with empty reasoning
/// renders `<think>\n\n`, which BPE merges into one token, so the full
/// prompt is no longer a prefix while the tokens before the suffix are. The
/// boundary starts a special token after every image, so only the text
/// suffix is encoded again.
pub(crate) fn stable_prefix_tokens(
    tokenizer: &QwenTokenizer,
    rendered: &Rendered,
    tokens: &[TokenId],
) -> gufo_core::text::Result<usize> {
    let suffix = encode_rendered(
        tokenizer,
        &rendered.text,
        rendered.stable_prefix_bytes,
        rendered.text.len(),
        &rendered.content_spans,
    );
    if !tokens.ends_with(&suffix) {
        return Err(Error::logic("Qwen stable prefix is not a token boundary"));
    }
    Ok(tokens.len() - suffix.len())
}
