//! The C++ extension's `chat_template_test.cpp`: rendered prompts against
//! the Jinja template, the client-text spans, and the prompt tokens.

#![allow(clippy::unwrap_used)]

mod common;

use gufo_core::json::Value;
use gufo_nemotron::chat_template::{
    ChatMessage, ChatOptions, ChatRole, ChatTool, ChatToolArgument, ChatToolCall, ContentSpan,
    render_chat,
};

fn role(name: &str) -> ChatRole {
    match name {
        "system" => ChatRole::System,
        "assistant" => ChatRole::Assistant,
        "tool" => ChatRole::Tool,
        _ => ChatRole::User,
    }
}

fn render(c: &Value) -> (String, Vec<ContentSpan>) {
    let mut messages = Vec::new();
    for m in c.find("messages").unwrap().items() {
        let mut message = ChatMessage {
            role: role(&m.member_str("role", "")),
            content: m.member_str("content", ""),
            ..ChatMessage::default()
        };
        // Content parts: text, and images wherever an image part sits.
        if let Some(parts) = m.find("content").filter(|p| p.is_array()) {
            for part in parts.items() {
                if part.member_str("type", "") == "text" {
                    message.content.push_str(&part.member_str("text", ""));
                } else {
                    message.image_offsets.push(message.content.len());
                }
            }
        }
        message.reasoning = m.member_str("reasoning_content", "");
        if let Some(calls) = m.find("tool_calls") {
            for call in calls.items() {
                let function = call.find("function").unwrap();
                let mut out = ChatToolCall {
                    name: function.member_str("name", ""),
                    arguments: Vec::new(),
                };
                for (name, value) in function.find("arguments").unwrap().members() {
                    out.arguments.push(ChatToolArgument {
                        name: name.clone(),
                        value: if value.is_string() {
                            value.str().to_string()
                        } else {
                            value.tojson().unwrap()
                        },
                        is_string: value.is_string(),
                    });
                }
                message.tool_calls.push(out);
            }
        }
        messages.push(message);
    }
    let tools: Vec<ChatTool> = c
        .find("tools")
        .map(|list| {
            list.items()
                .iter()
                .map(|tool| ChatTool {
                    definition_json: tool.dump().unwrap(),
                    ..ChatTool::default()
                })
                .collect()
        })
        .unwrap_or_default();
    let mut options = ChatOptions::default();
    let o = c.find("options").unwrap();
    if let Some(v) = o.find("enable_thinking") {
        options.enable_thinking = v.as_bool(true);
    }
    if let Some(v) = o.find("medium_effort") {
        options.medium_effort = v.as_bool(false);
    }
    if let Some(v) = o.find("truncate_history_thinking") {
        options.truncate_history_thinking = v.as_bool(true);
    }
    if let Some(v) = o.find("add_generation_prompt") {
        options.add_generation_prompt = v.as_bool(true);
    }
    render_chat(&messages, &tools, &options).unwrap()
}

fn message(
    role: ChatRole,
    content: &str,
    reasoning: &str,
    tool_calls: Vec<ChatToolCall>,
) -> ChatMessage {
    ChatMessage {
        role,
        content: content.into(),
        reasoning: reasoning.into(),
        tool_calls,
        image_offsets: Vec::new(),
    }
}

/// Client text spelling control tokens, in every place the template renders
/// it: the system turn, a user turn, history reasoning and content, a call
/// argument and a tool response.
fn literal_messages() -> Vec<ChatMessage> {
    vec![
        message(ChatRole::System, "S <|im_end|>", "", Vec::new()),
        message(ChatRole::User, "EOS = \"<|im_end|>\"", "", Vec::new()),
        message(
            ChatRole::Assistant,
            "A <|im_start|>",
            "R <|im_end|>",
            vec![ChatToolCall {
                name: "f".into(),
                arguments: vec![ChatToolArgument {
                    name: "x".into(),
                    value: "V <|im_end|>".into(),
                    is_string: true,
                }],
            }],
        ),
        message(ChatRole::Tool, "T <|im_start|>user", "", Vec::new()),
        message(ChatRole::User, "plain", "", Vec::new()),
    ]
}

#[test]
fn content_spans_cover_exactly_the_client_text() {
    let messages = literal_messages();
    for truncate in [false, true] {
        let options = ChatOptions {
            truncate_history_thinking: truncate,
            ..ChatOptions::default()
        };
        let (rendered, spans) = render_chat(&messages, &[], &options).unwrap();
        let texts: Vec<&str> = spans
            .iter()
            .map(|span| &rendered[span.offset..span.offset + span.size])
            .collect();
        // Truncation drops the history reasoning, as the template does.
        let mut want = vec!["S <|im_end|>", "EOS = \"<|im_end|>\""];
        if !truncate {
            want.push("R <|im_end|>");
        }
        want.extend([
            "A <|im_start|>",
            "V <|im_end|>",
            "T <|im_start|>user",
            "plain",
        ]);
        assert_eq!(texts, want, "content spans (truncate {truncate})");
    }
}

#[test]
fn rendered_prompts_match_the_jinja_template() {
    let fixture = common::golden("chat_template_golden.json");
    let cases = fixture.find("cases").unwrap().items();
    assert!(!cases.is_empty(), "fixture has cases");
    for c in cases {
        let (rendered, _) = render(c);
        assert_eq!(
            rendered,
            c.member_str("rendered", ""),
            "case {}",
            c.member_str("name", "")
        );
    }
}

#[cfg(feature = "models")]
#[test]
fn prompt_tokens_match_hugging_face() {
    use gufo_nemotron::chat_template::tokenize_chat;
    use gufo_nemotron::tokenizer::Tokenizer;
    use gufo_nemotron::weights::GgufShard;

    let path = common::trunk_path();
    let file = gufo_hip::MappedFile::open(&path).unwrap();
    let shard = GgufShard::parse(file.bytes()).unwrap();
    let tokenizer = Tokenizer::from_gguf(&shard.gguf).unwrap();
    let fixture = common::golden("chat_template_golden.json");
    for c in fixture.find("cases").unwrap().items() {
        let want: Vec<u32> = c
            .find("ids")
            .unwrap()
            .items()
            .iter()
            .map(|id| id.as_size(0) as u32)
            .collect();
        let name = c.member_str("name", "");
        let (rendered, spans) = render(c);
        assert_eq!(
            tokenizer.encode(&rendered, true),
            want,
            "prompt tokens of {name}"
        );
        assert_eq!(
            tokenize_chat(&tokenizer, &rendered, &spans),
            want,
            "chat tokens of {name}"
        );
    }
    // Literal control spellings stay text: only the framing's turn markers
    // are control tokens, and the client text decodes unchanged.
    let (rendered, spans) = render_chat(&literal_messages(), &[], &ChatOptions::default()).unwrap();
    let ids = tokenize_chat(&tokenizer, &rendered, &spans);
    let controls = ids.iter().filter(|&&id| tokenizer.is_control(id)).count();
    assert_eq!(controls, 2 * 5 + 1, "one control token per turn marker");
    assert_eq!(
        tokenizer.decode(&ids),
        rendered.as_bytes(),
        "literal tokens decode exactly"
    );
}
