//! The Qwen3.6 dialect against its template: every case of
//! `fixtures/qwen36_chat_template.json`, which
//! `reference/qwen36_template_goldens.py` renders from the Qwen3.6-35B-A3B
//! GGUF template (`reference/qwen36_chat_template.jinja`) with transformers'
//! Jinja settings, renders byte for byte with `Dialect::Qwen36`. The Qwen3.8
//! dialect keeps its own goldens (`tests/chat_template.rs`).

#![allow(clippy::unwrap_used)]

use gufo_core::text::{
    ChatMessage, ChatRole, ChatTool, ReasoningEffort, ReasoningOptions, ToolArgument, ToolCall,
};
use gufo_qwen::chat_template::{
    Dialect, QWEN36_TEMPLATE_SHA256, render, resolve_dialect_options, resolve_options,
};
use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/qwen36_chat_template.json");
const TEMPLATE: &str = include_str!("../reference/qwen36_chat_template.jinja");

fn role(name: &str) -> ChatRole {
    match name {
        "system" => ChatRole::System,
        "developer" => ChatRole::Developer,
        "assistant" => ChatRole::Assistant,
        "tool" => ChatRole::Tool,
        _ => ChatRole::User,
    }
}

fn message(value: &Value) -> ChatMessage {
    let mut message = ChatMessage::new(
        role(value["role"].as_str().unwrap()),
        value["content"].as_str().unwrap(),
    );
    message.thought = value["thought"].as_str().unwrap().to_string();
    message.tool_calls = value["tool_calls"]
        .as_array()
        .unwrap()
        .iter()
        .map(|call| ToolCall {
            id: call["id"].as_str().unwrap().into(),
            name: call["name"].as_str().unwrap().into(),
            arguments: call["arguments"]
                .as_array()
                .unwrap()
                .iter()
                .map(|argument| ToolArgument {
                    name: argument[0].as_str().unwrap().into(),
                    value: argument[1].as_str().unwrap().into(),
                    is_string: argument[2].as_bool().unwrap(),
                })
                .collect(),
        })
        .collect();
    message
}

/// The request controls of a case's template arguments.
fn reasoning(kwargs: &Value) -> ReasoningOptions {
    ReasoningOptions {
        enabled: kwargs.get("enable_thinking").and_then(Value::as_bool),
        preserve_thinking: kwargs.get("preserve_thinking").and_then(Value::as_bool),
        effort: kwargs
            .get("reasoning_effort")
            .and_then(Value::as_str)
            .map(|effort| match effort {
                "low" => ReasoningEffort::Low,
                "medium" => ReasoningEffort::Medium,
                _ => ReasoningEffort::XHigh,
            }),
        ..ReasoningOptions::default()
    }
}

#[test]
fn qwen36_dialect_matches_its_template() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    assert_eq!(
        fixture["template_sha256"].as_str().unwrap(),
        QWEN36_TEMPLATE_SHA256
    );
    assert_eq!(
        Dialect::of_template(TEMPLATE.as_bytes()),
        Dialect::Qwen36,
        "the stored template is the pinned one"
    );
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let messages: Vec<ChatMessage> = case["chat"]
            .as_array()
            .unwrap()
            .iter()
            .map(message)
            .collect();
        let tools: Vec<ChatTool> = case["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|definition| ChatTool {
                definition_json: definition.as_str().unwrap().into(),
                ..ChatTool::default()
            })
            .collect();
        let options = resolve_dialect_options(Dialect::Qwen36, &reasoning(&case["kwargs"]), false);
        let rendered = render(&messages, &tools, &options).unwrap();
        if rendered.text != case["text"].as_str().unwrap() {
            failures.push(format!(
                "{name}:\n--- Rust\n{}\n--- template\n{}",
                rendered.text,
                case["text"].as_str().unwrap()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The dialect's defaults: no effort instruction, earlier reasoning dropped
/// unless asked for; the Qwen3.8 defaults stay.
#[test]
fn dialects_resolve_their_own_defaults() {
    let default = ReasoningOptions::default();
    let qwen36 = resolve_dialect_options(Dialect::Qwen36, &default, false);
    assert!(qwen36.enable_thinking && !qwen36.preserve_thinking);
    assert_eq!(qwen36.dialect, Dialect::Qwen36);
    let qwen38 = resolve_options(&default, false);
    assert!(qwen38.enable_thinking && qwen38.preserve_thinking);
    assert_eq!(qwen38.dialect, Dialect::Qwen38);
    assert_eq!(
        Dialect::of_template(b"{{ messages }}"),
        Dialect::Qwen38,
        "an unknown template keeps the Qwen3.8 formatter"
    );
    // Every effort renders the same prompt in Qwen3.6.
    let haiku = [ChatMessage::new(ChatRole::User, "Write a haiku.")];
    let texts: Vec<String> = [
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::Max,
    ]
    .into_iter()
    .map(|effort| {
        let options = resolve_dialect_options(
            Dialect::Qwen36,
            &ReasoningOptions {
                effort: Some(effort),
                ..ReasoningOptions::default()
            },
            false,
        );
        render(&haiku, &[], &options).unwrap().text
    })
    .collect();
    assert!(texts.windows(2).all(|pair| pair[0] == pair[1]));
    assert!(!texts[0].contains("Reasoning effort"));
}
