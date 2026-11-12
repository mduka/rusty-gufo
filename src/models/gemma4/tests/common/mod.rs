//! Shared helpers of the Gemma 4 tests: fixtures, the released files and
//! in-memory GGUF images (the extension's `gguf_builder.hpp` and
//! `template_cases.hpp`).

#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use gufo_core::json::Value;
use gufo_core::text::{
    ChatMessage, ChatRole, ChatTool, ImageDetail, ImagePart, ToolArgument, ToolCall,
};
use gufo_formats::GgmlType;
use gufo_gemma4::chat_template::ChatOptions;

/// `tests/fixtures/<name>`.
pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// A fixture's JSON.
pub fn read_json(name: &str) -> Value {
    let text = std::fs::read(fixture(name)).unwrap_or_else(|error| panic!("{name}: {error}"));
    gufo_core::json::parse(&text).unwrap_or_else(|error| panic!("{name}: {error}"))
}

/// A path from the environment; the model tests fail rather than skip
/// without it, since a missing model is not a pass.
pub fn env_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must name a Gemma 4 GGUF"))
}

/// A path from the environment, when set.
pub fn optional_env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
}

// ---------------------------------------------------------------------------
// In-memory GGUF v3 images (`gguf_builder.hpp`).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Meta {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    BoolArray(Vec<bool>),
    I32Array(Vec<i32>),
    StrArray(Vec<String>),
}

pub type Metadata = BTreeMap<String, Meta>;

#[derive(Debug, Clone, PartialEq)]
pub struct TensorSpec {
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    pub data: Vec<u8>,
}

pub type Tensors = BTreeMap<String, TensorSpec>;

fn ggml_id(ty: GgmlType) -> u32 {
    (0..64)
        .find(|&id| GgmlType::from_id(id) == Some(ty))
        .unwrap_or_else(|| panic!("no GGUF id for {ty:?}"))
}

/// Serializes a GGUF v3 file in memory, keys and tensors in name order.
pub fn gguf_image(metadata: &Metadata, tensors: &Tensors) -> Vec<u8> {
    let mut out = Vec::new();
    let text = |out: &mut Vec<u8>, s: &str| {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    };
    let align = |out: &mut Vec<u8>| out.resize(out.len().div_ceil(32) * 32, 0);
    out.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        text(&mut out, key);
        match value {
            Meta::U32(v) => {
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Meta::F32(v) => {
                out.extend_from_slice(&6u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Meta::Bool(v) => {
                out.extend_from_slice(&7u32.to_le_bytes());
                out.push(u8::from(*v));
            }
            Meta::Str(v) => {
                out.extend_from_slice(&8u32.to_le_bytes());
                text(&mut out, v);
            }
            Meta::BoolArray(values) => {
                out.extend_from_slice(&9u32.to_le_bytes());
                out.extend_from_slice(&7u32.to_le_bytes());
                out.extend_from_slice(&(values.len() as u64).to_le_bytes());
                out.extend(values.iter().map(|&v| u8::from(v)));
            }
            Meta::I32Array(values) => {
                out.extend_from_slice(&9u32.to_le_bytes());
                out.extend_from_slice(&5u32.to_le_bytes());
                out.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for v in values {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            Meta::StrArray(values) => {
                out.extend_from_slice(&9u32.to_le_bytes());
                out.extend_from_slice(&8u32.to_le_bytes());
                out.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for v in values {
                    text(&mut out, v);
                }
            }
        }
    }
    let mut offset = 0u64;
    for (name, spec) in tensors {
        text(&mut out, name);
        out.extend_from_slice(&(spec.dims.len() as u32).to_le_bytes());
        for d in &spec.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&ggml_id(spec.ty).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += (spec.data.len() as u64).div_ceil(32) * 32;
    }
    align(&mut out);
    for spec in tensors.values() {
        out.extend_from_slice(&spec.data);
        align(&mut out);
    }
    if tensors.is_empty() {
        align(&mut out);
    }
    out
}

// ---------------------------------------------------------------------------
// Template cases (`template_cases.hpp`).
// ---------------------------------------------------------------------------

pub struct TemplateCase {
    pub name: String,
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ChatTool>,
    pub options: ChatOptions,
}

fn parse_role(role: &str) -> ChatRole {
    match role {
        "system" => ChatRole::System,
        "developer" => ChatRole::Developer,
        "assistant" => ChatRole::Assistant,
        "tool" => ChatRole::Tool,
        _ => ChatRole::User,
    }
}

fn dump(value: &Value) -> String {
    value.dump().expect("finite JSON")
}

/// Converts the shared fixture into the server's message types, the same way
/// the OpenAI adapter does (tool arguments as name/value pairs).
pub fn load_template_cases() -> Vec<TemplateCase> {
    let spec = read_json("template_cases.json");
    let mut tools = BTreeMap::new();
    for (name, tool) in spec.find("tools").expect("tools").members() {
        let function = tool.find("function").expect("function");
        tools.insert(
            name.clone(),
            ChatTool {
                name: function.member_str("name", ""),
                description: function.member_str("description", ""),
                parameters_json: function.find("parameters").map_or("{}".into(), dump),
                definition_json: dump(tool),
            },
        );
    }
    let mut cases = Vec::new();
    for (name, c) in spec.find("cases").expect("cases").members() {
        let mut options = ChatOptions::default();
        if let Some(v) = c.find("add_generation_prompt") {
            options.add_generation_prompt = v.as_bool(false);
        }
        if let Some(v) = c.find("enable_thinking") {
            options.enable_thinking = v.as_bool(false);
        }
        if let Some(v) = c.find("preserve_thinking") {
            options.preserve_thinking = v.as_bool(false);
        }
        let case_tools = c
            .find("tools")
            .map(|list| {
                list.items()
                    .iter()
                    .map(|tool| tools[tool.str()].clone())
                    .collect()
            })
            .unwrap_or_default();
        let mut messages = Vec::new();
        for m in c.find("messages").expect("messages").items() {
            let mut message = ChatMessage::new(parse_role(&m.member_str("role", "")), "");
            match m.find("content") {
                Some(Value::Array(parts)) => {
                    // Content parts, flattened as the OpenAI adapter does:
                    // text appends, an image records its byte offset.
                    for part in parts {
                        if part.member_str("type", "") == "image_url" {
                            message.images.push(ImagePart {
                                offset: message.content.len(),
                                bytes: Arc::new(Vec::new()),
                                detail: ImageDetail::Auto,
                            });
                        } else {
                            message.content.push_str(&part.member_str("text", ""));
                        }
                    }
                }
                _ => message.content = m.member_str("content", ""),
            }
            message.name = m.member_str("name", "");
            message.thought = m.member_str("reasoning_content", "");
            message.tool_call_id = m.member_str("tool_call_id", "");
            if let Some(calls) = m.find("tool_calls") {
                for call in calls.items() {
                    message.tool_calls.push(ToolCall {
                        id: call.member_str("id", ""),
                        name: call.member_str("name", ""),
                        arguments: call
                            .find("arguments")
                            .expect("arguments")
                            .members()
                            .iter()
                            .map(|(arg, value)| ToolArgument {
                                name: arg.clone(),
                                value: if value.is_string() {
                                    value.str().to_owned()
                                } else {
                                    dump(value)
                                },
                                is_string: value.is_string(),
                            })
                            .collect(),
                    });
                }
            }
            messages.push(message);
        }
        cases.push(TemplateCase {
            name: name.clone(),
            messages,
            tools: case_tools,
            options,
        });
    }
    cases
}
