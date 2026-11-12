//! The extension's `template_test.cpp`: every golden case renders
//! byte-identically to the Jinja source (apart from the documented end of a
//! replayed thought), plus option resolution, rejections and the serving
//! additions.

#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use gufo_core::text::{
    ChatMessage, ChatRole, ChatTool, ImageDetail, ImagePart, ReasoningEffort, ReasoningOptions,
    ToolArgument, ToolCall,
};
use gufo_gemma4::chat_template::{
    ChatOptions, IMAGE_PLACEHOLDER, THOUGHT_END, THOUGHT_START, UNSLOTH_TEMPLATE_SHA256, render,
    resolve_chat_options, sha256_hex,
};

/// The renderer ends a replayed thought at `<channel|>`, where the Jinja
/// source adds a newline Gemma does not generate. Restores it for Jinja
/// parity.
fn with_jinja_thought_newlines(mut text: String, messages: &[ChatMessage]) -> String {
    for message in messages {
        if message.thought.is_empty() {
            continue;
        }
        let replayed = format!("{THOUGHT_START}{}{THOUGHT_END}", message.thought);
        if let Some(at) = text.find(&replayed) {
            text.insert(at + THOUGHT_START.len() + message.thought.len(), '\n');
        }
    }
    text
}

fn image(offset: usize) -> ImagePart {
    ImagePart {
        offset,
        bytes: Arc::new(Vec::new()),
        detail: ImageDetail::Auto,
    }
}

fn call(id: &str, name: &str, arguments: &[(&str, &str, bool)]) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: arguments
            .iter()
            .map(|&(name, value, is_string)| ToolArgument {
                name: name.into(),
                value: value.into(),
                is_string,
            })
            .collect(),
    }
}

fn tool(name: &str, definition: &str) -> ChatTool {
    ChatTool {
        name: name.into(),
        definition_json: definition.into(),
        ..ChatTool::default()
    }
}

#[test]
fn goldens() {
    let goldens = common::read_json("chat_template_hf_goldens.json");
    let gemma = goldens
        .find("gemma4")
        .expect("goldens lack a gemma4 section");
    assert_eq!(
        gemma.member_str("template_sha256", ""),
        UNSLOTH_TEMPLATE_SHA256,
        "goldens were rendered from another template"
    );
    let expected = gemma.find("cases").unwrap();
    let cases = common::load_template_cases();
    assert_eq!(cases.len(), expected.len(), "case/golden count mismatch");
    for c in &cases {
        let mut images = Vec::new();
        let rendered = render(&c.messages, &c.tools, &c.options, Some(&mut images))
            .unwrap_or_else(|e| panic!("{}: {e}", c.name));
        let expected_images: usize = c.messages.iter().map(|m| m.images.len()).sum();
        assert_eq!(images.len(), expected_images, "{}: image count", c.name);
        for &offset in &images {
            assert!(
                rendered.text[offset..].starts_with(IMAGE_PLACEHOLDER),
                "{}: image offset is not a placeholder",
                c.name
            );
        }
        let golden = expected
            .find(&c.name)
            .unwrap_or_else(|| panic!("{}: no golden", c.name));
        let text = with_jinja_thought_newlines(rendered.text.clone(), &c.messages);
        assert_eq!(
            sha256_hex(text.as_bytes()),
            golden.member_str("rendered_sha256", ""),
            "{}: rendering differs from Jinja:\n{}",
            c.name,
            rendered.text
        );

        // The generation prompt is a pure suffix of the stable conversation.
        let without = ChatOptions {
            add_generation_prompt: false,
            ..c.options
        };
        let prefix = render(&c.messages, &c.tools, &without, Some(&mut images)).unwrap();
        assert_eq!(
            prefix.text,
            rendered.text[..rendered.generation_prompt_offset],
            "{}: generation prompt is not a suffix",
            c.name
        );
    }
}

#[test]
fn options() {
    let mut reasoning = ReasoningOptions::default();
    assert!(
        !resolve_chat_options(&reasoning).enable_thinking,
        "thinking is off by default"
    );
    reasoning.effort = Some(ReasoningEffort::High);
    assert!(
        resolve_chat_options(&reasoning).enable_thinking,
        "an effort level enables thinking"
    );
    reasoning.effort = Some(ReasoningEffort::Minimal);
    assert!(
        !resolve_chat_options(&reasoning).enable_thinking,
        "minimal effort keeps thinking off"
    );
    reasoning.enabled = Some(true);
    assert!(
        resolve_chat_options(&reasoning).enable_thinking,
        "explicit enable wins"
    );
}

#[test]
fn rejections() {
    let defaults = ChatOptions::default();
    let mut with_image = ChatMessage::default();
    with_image.images.push(image(0));
    let error = render(&[with_image], &[], &defaults, None)
        .expect_err("image input accepted without image offsets");
    assert!(!error.is_empty());
    // The template renders tool results only after the turn that made the
    // calls; an image in any other tool message would be dropped.
    let mut orphan = ChatMessage::new(ChatRole::Tool, "chart");
    orphan.images.push(image(0));
    let orphaned = [ChatMessage::new(ChatRole::User, "hi"), orphan];
    let mut offsets = Vec::new();
    assert!(
        render(&orphaned, &[], &defaults, Some(&mut offsets)).is_err(),
        "orphan tool result image accepted"
    );
    let mut assistant = ChatMessage::new(ChatRole::Assistant, "");
    assistant.images.push(image(0));
    assert!(
        render(&[assistant], &[], &defaults, Some(&mut offsets)).is_err(),
        "assistant image accepted"
    );
    assert!(
        render(&[], &[tool("", "[1]")], &defaults, None).is_err(),
        "non-object tool accepted"
    );
    assert!(
        render(&[], &[tool("a{b", "")], &defaults, None).is_err(),
        "braced tool name accepted"
    );
    let mut braced_call = ChatMessage::new(ChatRole::Assistant, "");
    braced_call.tool_calls.push(call("c", "x{y", &[]));
    assert!(
        render(&[braced_call], &[], &defaults, None).is_err(),
        "braced replayed call accepted"
    );
    let dotted = tool(
        "github.create_issue",
        r#"{"type":"function","function":{"name":"github.create_issue"}}"#,
    );
    let rendered = render(&[], &[dotted], &defaults, None).unwrap();
    assert!(
        rendered.text.contains("declaration:github.create_issue{"),
        "dotted tool name rendered"
    );
}

#[test]
fn shown_constants() {
    let record = tool(
        "record",
        r#"{"type":"function","function":{"name":"record",
    "parameters":{"type":"object","properties":{
      "value":{"type":"string","const":"alpha"},
      "items":{"type":"array","items":{"type":"object","properties":{
        "kind":{"type":"string","const":"x"}}}},
      "n":{"type":"integer","const":3}},"required":["value"]}}}"#,
    );
    let rendered = render(&[], &[record], &ChatOptions::default(), None).unwrap();
    let text = &rendered.text;
    assert!(
        text.contains("value:{enum:[<|\"|>alpha<|\"|>]")
            && text.contains("kind:{enum:[<|\"|>x<|\"|>]")
            && text.contains("n:{type:"),
        "a string const is shown as a one-value enum: {text}"
    );
}

/// A replayed thought renders as Gemma generates it, so a continuation's
/// prompt reproduces the reasoning and call tokens of the previous turn.
#[test]
fn replayed_thought() {
    let mut assistant = ChatMessage::new(ChatRole::Assistant, "");
    assistant.thought = "Read the file first.".into();
    assistant
        .tool_calls
        .push(call("c", "read", &[("path", "a.py", true)]));
    let mut result = ChatMessage::new(ChatRole::Tool, "x = 1");
    result.tool_call_id = "c".into();
    let messages = [
        ChatMessage::new(ChatRole::User, "Read a.py."),
        assistant,
        result,
    ];
    let options = ChatOptions {
        enable_thinking: true,
        ..ChatOptions::default()
    };
    let rendered = render(&messages, &[], &options, None).unwrap();
    assert!(
        rendered
            .text
            .contains("<|channel>thought\nRead the file first.<channel|><|tool_call>call:read{"),
        "a replayed thought ends at <channel|>: {}",
        rendered.text
    );
}

/// Server-authored framing such as a response-format instruction renders
/// after the client's system text, including in a system turn it creates.
#[test]
fn framing_suffix() {
    let mut system = ChatMessage::new(ChatRole::System, "Be brief.");
    system.framing_suffix = "\n\nAnswer in JSON.".into();
    let rendered = render(
        &[system, ChatMessage::new(ChatRole::User, "Hi")],
        &[],
        &ChatOptions::default(),
        None,
    )
    .unwrap();
    assert!(
        rendered
            .text
            .contains("<|turn>system\nBe brief.\n\nAnswer in JSON.<turn|>"),
        "framing follows system text: {}",
        rendered.text
    );
    let mut framing = ChatMessage::new(ChatRole::System, "");
    framing.framing_suffix = "Answer in JSON.".into();
    let rendered = render(
        &[framing, ChatMessage::new(ChatRole::User, "Hi")],
        &[],
        &ChatOptions::default(),
        None,
    )
    .unwrap();
    assert!(
        rendered
            .text
            .contains("<|turn>system\nAnswer in JSON.<turn|>"),
        "framing renders in a created system turn: {}",
        rendered.text
    );
}

fn tool_history() -> (ChatMessage, ChatMessage) {
    let mut assistant = ChatMessage::new(ChatRole::Assistant, "");
    assistant
        .tool_calls
        .push(call("c", "read", &[("path", "a.py", true)]));
    let mut result = ChatMessage::new(ChatRole::Tool, "x = 1");
    result.tool_call_id = "c".into();
    (assistant, result)
}

/// Serving closes the thought block after a tool response when thinking is
/// off; the default rendering stays the official template's.
#[test]
fn closed_thought_after_tool() {
    let (assistant, result) = tool_history();
    let history = [
        ChatMessage::new(ChatRole::User, "Start the task."),
        assistant,
        result,
    ];
    let suffix = |options: &ChatOptions| {
        let rendered = render(&history, &[], options, None).unwrap();
        rendered.text[rendered.generation_prompt_offset..].to_owned()
    };
    let mut reasoning = ReasoningOptions::default();
    assert!(
        suffix(&ChatOptions::default()).is_empty(),
        "the official template adds nothing"
    );
    assert_eq!(
        suffix(&resolve_chat_options(&reasoning)),
        "<|channel>thought\n<channel|>",
        "serving closes the thought block"
    );
    reasoning.enabled = Some(true);
    assert_eq!(
        suffix(&resolve_chat_options(&reasoning)),
        "<|channel>thought\n",
        "thinking opens the thought block"
    );
}

/// Agent clients may end each request with user context that the next
/// request replaces. The stable boundary then precedes that turn, so it
/// prefixes the next request; ordinary chat keeps it at the generation
/// prompt.
#[test]
fn replaced_final_user_turn() {
    let render_one = |messages: &[ChatMessage]| {
        let rendered = render(messages, &[], &ChatOptions::default(), None).unwrap();
        assert!(
            rendered.stable_prefix_offset <= rendered.generation_prompt_offset,
            "boundary precedes the generation prompt"
        );
        (rendered.text, rendered.stable_prefix_offset)
    };
    let prefixes = |text: &str, first: &str, stable: usize| text.starts_with(&first[..stable]);
    let (assistant, result) = tool_history();
    let history = vec![
        ChatMessage::new(ChatRole::System, "You are an agent."),
        ChatMessage::new(ChatRole::User, "Start the task."),
        assistant,
        result,
    ];

    // After a tool result the model turn stays open on the next request.
    let mut first = history.clone();
    first.push(ChatMessage::new(ChatRole::User, "Runtime context, turn 1."));
    let mut second = history.clone();
    second.push(ChatMessage::new(ChatRole::Assistant, "Done."));
    second.push(ChatMessage::new(ChatRole::User, "Runtime context, turn 2."));
    let (first_text, first_stable) = render_one(&first);
    let (second_text, second_stable) = render_one(&second);
    assert!(
        first_text[first_stable..].starts_with("<turn|>")
            && first_text[..first_stable].ends_with("<tool_response|>"),
        "boundary closes the tool result: {first_text}"
    );
    assert!(
        prefixes(&second_text, &first_text, first_stable),
        "boundary prefixes a request that replaces the final user turn"
    );
    assert!(
        second_stable > first_stable,
        "boundary advances with the history"
    );

    // A client that keeps the user turn still finds the boundary as a
    // prefix; a user turn after an assistant reply is ordinary chat.
    let mut kept = first.clone();
    kept.push(ChatMessage::new(ChatRole::Assistant, "Done."));
    kept.push(ChatMessage::new(ChatRole::User, "Next."));
    let (kept_text, kept_stable) = render_one(&kept);
    assert!(
        prefixes(&kept_text, &first_text, first_stable),
        "boundary prefixes an appended conversation"
    );
    assert_eq!(
        &kept_text[kept_stable..],
        "<|turn>model\n<|channel>thought\n<channel|>",
        "ordinary chat keeps the boundary at the generation prompt"
    );

    // Context sent as a second user message after the real query.
    let mut query = second.clone();
    query.last_mut().unwrap().content = "Real question.".into();
    query.push(ChatMessage::new(ChatRole::User, "Runtime context, turn 2."));
    let (query_text, query_stable) = render_one(&query);
    assert!(
        query_text[query_stable..].starts_with("<|turn>user\n")
            && query_text[query_stable..].contains("Runtime context")
            && !query_text[query_stable..].contains("Real question"),
        "boundary precedes trailing context after the real query"
    );

    // The opening user turn has no earlier assistant: nothing to replace.
    let (opening_text, opening_stable) = render_one(&[ChatMessage::new(ChatRole::User, "Hello")]);
    assert_eq!(
        &opening_text[opening_stable..],
        "<|turn>model\n<|channel>thought\n<channel|>",
        "opening turn keeps the boundary at the generation prompt"
    );
}
