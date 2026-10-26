//! Structured chat input (`tokenization::ChatMessage`, `ChatTool` and the
//! server's `ChatRequest`), shared by the HTTP layer and every model's chat
//! template.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::json_constraint::JsonConstraint;

/// Provider-neutral reasoning levels accepted by the HTTP and CLI surfaces;
/// model formatters map them onto their own effort sets
/// (`core/reasoning.hpp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ReasoningOptions {
    pub enabled: Option<bool>,
    pub effort: Option<ReasoningEffort>,
    pub preserve_thinking: Option<bool>,
    /// Reasoning tokens before the end of thinking is forced; `None` is unlimited.
    pub budget_tokens: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ChatRole {
    System,
    Developer,
    #[default]
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    /// The template role name; developer messages render as system.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System | Self::Developer => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// An image inserted before byte `offset` of the message content. Equal
/// offsets keep input order; image-only messages use offset zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImagePart {
    pub offset: usize,
    /// The encoded PNG or JPEG bytes.
    pub bytes: Arc<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolArgument {
    pub name: String,
    pub value: String,
    pub is_string: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Vec<ToolArgument>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    /// Optional function or tool name.
    pub name: String,
    /// Optional reasoning that preceded the content.
    pub thought: String,
    pub tool_call_id: String,
    pub images: Vec<ImagePart>,
    pub tool_calls: Vec<ToolCall>,
    /// Server-authored instructions after a system message's content: chat
    /// template framing, never supplied by a client.
    pub framing_suffix: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTool {
    pub name: String,
    pub description: String,
    pub parameters_json: String,
    /// The complete validated HTTP tool object, preserving field order and
    /// extensions.
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
}

/// The compiled grammar a structured answer follows; requests compare it by
/// identity.
#[derive(Debug, Clone)]
pub struct OutputGrammar(pub Arc<JsonConstraint>);

impl PartialEq for OutputGrammar {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for OutputGrammar {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ChatTool>,
    pub client_id: String,
    pub tool_choice: ToolChoice,
    /// The function a `tool_choice` object named; Responses echoes it.
    pub forced_tool_name: String,
    /// Tool calls follow their argument schemas even without `strict`
    /// functions (a forced or single call).
    pub constrained_tools: bool,
    pub parallel_tool_calls: bool,
    pub reasoning: ReasoningOptions,
    pub add_vision_id: bool,
    /// Bypasses prompt reuse for this request; its completed state may still
    /// be retained.
    pub cache_prompt: bool,
    /// Streams llama-server `prompt_progress` before the first token; never
    /// changes the prompt or its cache identity.
    pub return_progress: bool,
    /// Output sequences that end generation, excluded from the output.
    pub stop_sequences: Vec<String>,
    /// `response_format` (`text.format` in Responses) as a grammar.
    pub response_format: Option<OutputGrammar>,
    pub response_format_description: String,
    /// The Responses `namespace` of each function flattened out of one,
    /// echoed on its calls.
    pub tool_namespaces: BTreeMap<String, String>,
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            tools: Vec::new(),
            client_id: "anonymous".into(),
            tool_choice: ToolChoice::Auto,
            forced_tool_name: String::new(),
            constrained_tools: false,
            parallel_tool_calls: true,
            reasoning: ReasoningOptions::default(),
            add_vision_id: false,
            cache_prompt: true,
            return_progress: false,
            stop_sequences: Vec::new(),
            response_format: None,
            response_format_description: String::new(),
            tool_namespaces: BTreeMap::new(),
        }
    }
}

impl ChatRequest {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            ..Self::default()
        }
    }
}
