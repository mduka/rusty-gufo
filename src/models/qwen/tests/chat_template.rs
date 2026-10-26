//! The Qwen chat formatter: every case of the C++
//! `tests/models/qwen/tokenization/chat_template_test.cpp`, and with the
//! `models` feature the Hugging Face token goldens of
//! `chat_template_hf_token_golden_test.cpp` on the Qwen3.8 27B GGUF
//! (`GUFO_QWEN27B_MODEL`, else the pinned file in the Hugging Face cache).

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use gufo_core::gguf::Gguf;
use gufo_core::qwen_tokenizer::QwenTokenizer;
use gufo_core::text::{
    ChatMessage, ChatRequest, ChatRole, ChatTool, ImagePart, ReasoningEffort, ReasoningOptions,
    ToolArgument, ToolCall,
};
use gufo_qwen::chat_template::{
    self, ChatTemplate, ChatTemplateOptions, OFFICIAL_TEMPLATE_SHA256, Profile,
    QwenReasoningEffort, encode_rendered, is_qwen38_named, prepare_prompt, render,
    render_and_tokenize, resolve_options,
};
use gufo_qwen::vision;
use serde_json::Value;
use sha2::{Digest, Sha256};

const GOLDENS: &str = include_str!("../../../../tests/fixtures/chat_template_hf_goldens.json");
const REFERENCE_TEMPLATE: &str = include_str!("../reference/chat_template.jinja");

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn message(role: ChatRole, content: &str) -> ChatMessage {
    ChatMessage::new(role, content)
}

fn with_thought(role: ChatRole, content: &str, thought: &str) -> ChatMessage {
    ChatMessage {
        thought: thought.into(),
        ..message(role, content)
    }
}

fn render_text(messages: &[ChatMessage], options: &ChatTemplateOptions) -> String {
    render(messages, &[], options).unwrap().text
}

fn no_thinking() -> ChatTemplateOptions {
    ChatTemplateOptions {
        enable_thinking: false,
        ..ChatTemplateOptions::default()
    }
}

/// A GGUF with string (type 8) and string-array (type 9) metadata.
fn gguf_image(strings: &[(&str, &[u8])], arrays: &[(&str, Vec<Vec<u8>>)]) -> Vec<u8> {
    fn string(out: &mut Vec<u8>, value: &[u8]) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value);
    }
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&((strings.len() + arrays.len()) as u64).to_le_bytes());
    for (key, value) in strings {
        string(&mut out, key.as_bytes());
        out.extend_from_slice(&8u32.to_le_bytes());
        string(&mut out, value);
    }
    for (key, values) in arrays {
        string(&mut out, key.as_bytes());
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&8u32.to_le_bytes());
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            string(&mut out, value);
        }
    }
    out.resize(out.len().next_multiple_of(32), 0);
    out
}

#[test]
fn basic_chat_rendering() {
    let messages = [
        message(ChatRole::System, "You are a concise assistant."),
        message(ChatRole::User, "What is 2+2?"),
    ];
    assert_eq!(
        render_text(&messages, &no_thinking()),
        "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n\
         <|im_start|>user\nWhat is 2+2?<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[test]
fn thinking_framing() {
    let messages = [
        message(ChatRole::User, "Solve this equation."),
        with_thought(
            ChatRole::Assistant,
            "The roots are 2 and 3.",
            "First let's factor the polynomial.",
        ),
    ];
    let options = ChatTemplateOptions {
        reasoning_effort: QwenReasoningEffort::Medium,
        ..ChatTemplateOptions::default()
    };
    assert_eq!(
        render_text(&messages, &options),
        "<|im_start|>user\nSolve this equation.<|im_end|>\n\
         <|im_start|>assistant\n<think>\nFirst let's factor the polynomial.\n</think>\n\n\
         The roots are 2 and 3.<|im_end|>\n\
         <|im_start|>assistant\n<think>\n"
    );
}

#[test]
fn historical_thinking_does_not_enable_new_thinking() {
    let messages = [
        message(ChatRole::User, "Name one color."),
        with_thought(
            ChatRole::Assistant,
            "Red",
            "I should answer with one color.",
        ),
        message(ChatRole::User, "Name another."),
    ];
    assert_eq!(
        render_text(&messages, &no_thinking()),
        "<|im_start|>user\nName one color.<|im_end|>\n\
         <|im_start|>assistant\n<think>\nI should answer with one color.\n</think>\n\n\
         Red<|im_end|>\n<|im_start|>user\nName another.<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[test]
fn reasoning_effort_and_preservation() {
    let messages = [
        message(ChatRole::System, "System rules."),
        message(ChatRole::Developer, "Developer rules."),
        message(ChatRole::User, "First question."),
        with_thought(
            ChatRole::Assistant,
            "First answer.",
            "Old private reasoning.",
        ),
        message(ChatRole::User, "Second question."),
    ];
    let options = ChatTemplateOptions {
        reasoning_effort: QwenReasoningEffort::Low,
        preserve_thinking: false,
        ..ChatTemplateOptions::default()
    };
    let rendered = render_text(&messages, &options);
    assert!(rendered.contains("Reasoning effort is set to low."));
    assert!(rendered.contains("System rules.\nDeveloper rules."));
    assert!(!rendered.contains("Old private reasoning."));
    assert!(rendered.contains("First answer."));
    assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n"));
}

#[test]
fn bounded_output_limit() {
    let messages = [message(ChatRole::User, &"A".repeat(2000))];
    let options = ChatTemplateOptions {
        max_output_bytes: 100,
        ..ChatTemplateOptions::default()
    };
    let error = render(&messages, &[], &options).unwrap_err();
    assert!(error.contains("exceeds maximum bound"), "{error}");
}

#[test]
fn rendered_prompt_bound_follows_the_context() {
    use gufo_qwen::chat_template::rendered_prompt_bound_bytes;
    const MIB: usize = 1 << 20;
    assert_eq!(
        rendered_prompt_bound_bytes(0),
        MIB,
        "zero context keeps 1 MiB"
    );
    assert_eq!(
        rendered_prompt_bound_bytes(4096),
        MIB,
        "small contexts keep the floor"
    );
    assert_eq!(rendered_prompt_bound_bytes(262_144), 262_144 * 128);
    assert_eq!(rendered_prompt_bound_bytes(409_600), 409_600 * 128);
    // gufo #285: about 1.05 MB rendered, far below a 262,144-token context.
    let messages = [message(ChatRole::User, &"a".repeat(1_100_000))];
    let mut options = ChatTemplateOptions::default();
    assert!(
        render(&messages, &[], &options).is_err(),
        "the 1 MiB default refuses"
    );
    options.max_output_bytes = rendered_prompt_bound_bytes(262_144);
    assert!(
        render(&messages, &[], &options).is_ok(),
        "a context-sized bound admits a prompt the context can hold"
    );
}

#[test]
fn gguf_template_extraction() {
    let reference = REFERENCE_TEMPLATE
        .strip_suffix('\n')
        .unwrap_or(REFERENCE_TEMPLATE);
    let bytes = gguf_image(
        &[
            ("general.name", b"Qwen3.8-27B"),
            ("tokenizer.chat_template", reference.as_bytes()),
        ],
        &[],
    );
    let gguf = Gguf::parse(&bytes).unwrap();
    let template = ChatTemplate::from_gguf(&gguf, is_qwen38_named(&gguf)).unwrap();
    assert!(
        String::from_utf8_lossy(template.template()).contains("preserve_thinking"),
        "the template string is the GGUF's"
    );
    assert_eq!(template.profile(), Profile::Qwen38Reasoning);
    assert_eq!(template.id(), "qwen38-reasoning-compiled-v3");
    assert_eq!(template.sha256(), OFFICIAL_TEMPLATE_SHA256);
    assert!(ChatTemplate::validate_gguf(&gguf, is_qwen38_named(&gguf)).is_ok());

    let lookalike = gguf_image(
        &[
            ("general.name", b"Qwen3.8-27B"),
            (
                "tokenizer.chat_template",
                b"{% set enable_thinking = true %}{% set preserve_thinking = true %}\
                  <|im_start|><think>",
            ),
        ],
        &[],
    );
    let gguf = Gguf::parse(&lookalike).unwrap();
    let error = ChatTemplate::from_gguf(&gguf, is_qwen38_named(&gguf)).unwrap_err();
    assert!(error.starts_with("Qwen3.8 GGUF chat template SHA-256 is not a recognized pinned"));
    assert!(ChatTemplate::validate_gguf(&gguf, is_qwen38_named(&gguf)).is_err());

    let missing = gguf_image(&[("general.name", b"Qwen3.8-27B")], &[]);
    assert_eq!(
        ChatTemplate::from_gguf(&Gguf::parse(&missing).unwrap(), true).unwrap_err(),
        "Qwen3.8 GGUF is missing tokenizer.chat_template metadata"
    );
    // Other models fall back to ChatML.
    let other = gguf_image(&[("general.name", b"Qwen3.5-4B")], &[]);
    let template = ChatTemplate::from_gguf(&Gguf::parse(&other).unwrap(), false).unwrap();
    assert_eq!(template.profile(), Profile::LegacyChatMl);
    assert_eq!(template.id(), "qwen-chatml-compiled-v1");
    assert_eq!(template, ChatTemplate::default_template());
}

fn golden(name: &str) -> Value {
    let fixture: Value = serde_json::from_str(GOLDENS).unwrap();
    fixture["qwen"]["cases"][name].clone()
}

fn check_golden(
    messages: &[ChatMessage],
    tools: &[ChatTool],
    options: &ChatTemplateOptions,
    name: &str,
) {
    let rendered = render(messages, tools, options).unwrap();
    assert_eq!(
        sha256(rendered.text.as_bytes()),
        golden(name)["rendered_sha256"].as_str().unwrap(),
        "{name}: rendered bytes match the pinned Hugging Face golden"
    );
}

fn base() -> Vec<ChatMessage> {
    vec![message(ChatRole::User, "Name one color.")]
}

fn history() -> Vec<ChatMessage> {
    vec![
        message(ChatRole::User, "Name one color."),
        with_thought(
            ChatRole::Assistant,
            "Red",
            "I should answer with one color.",
        ),
        message(ChatRole::User, "Name another."),
    ]
}

fn weather_tool(description: &str) -> ChatTool {
    ChatTool {
        name: "get_weather".into(),
        description: description.into(),
        parameters_json:
            r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#
                .into(),
        definition_json: String::new(),
    }
}

fn weather_call(id: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: "get_weather".into(),
        arguments: vec![ToolArgument {
            name: "city".into(),
            value: "Rome".into(),
            is_string: true,
        }],
    }
}

fn image_part(offset: usize) -> ImagePart {
    ImagePart {
        offset,
        bytes: Arc::new(vec![0]),
    }
}

#[test]
fn hugging_face_rendered_goldens() {
    let mut options = ChatTemplateOptions::default();
    check_golden(&base(), &[], &options, "thinking_xhigh");
    check_golden(
        &base(),
        &[],
        &resolve_options(&ReasoningOptions::default(), false),
        "thinking_xhigh",
    );
    options.enable_thinking = false;
    check_golden(&base(), &[], &options, "chat");

    options.enable_thinking = true;
    options.reasoning_effort = QwenReasoningEffort::Low;
    check_golden(&base(), &[], &options, "thinking_low");
    options.reasoning_effort = QwenReasoningEffort::Medium;
    check_golden(&base(), &[], &options, "thinking_medium");
    options.reasoning_effort = QwenReasoningEffort::XHigh;
    check_golden(&base(), &[], &options, "thinking_xhigh");
    for (effort, expected) in [
        (ReasoningEffort::Low, QwenReasoningEffort::Low),
        (ReasoningEffort::Medium, QwenReasoningEffort::Medium),
        (ReasoningEffort::XHigh, QwenReasoningEffort::XHigh),
    ] {
        let resolved = resolve_options(
            &ReasoningOptions {
                effort: Some(effort),
                ..ReasoningOptions::default()
            },
            false,
        );
        options.reasoning_effort = expected;
        assert!(resolved.enable_thinking && resolved.preserve_thinking);
        assert_eq!(
            render(&base(), &[], &resolved),
            render(&base(), &[], &options)
        );
    }
    check_golden(
        &base(),
        &[],
        &resolve_options(
            &ReasoningOptions {
                enabled: Some(false),
                effort: Some(ReasoningEffort::Low),
                preserve_thinking: None,
                budget_tokens: None,
            },
            false,
        ),
        "chat",
    );

    options.reasoning_effort = QwenReasoningEffort::Medium;
    options.preserve_thinking = false;
    check_golden(&history(), &[], &options, "history_drop");
    options.enable_thinking = false;
    options.preserve_thinking = true;
    check_golden(&history(), &[], &options, "history_preserve");

    let tools = [weather_tool("Get weather")];
    let mut system_tools = vec![message(ChatRole::System, "Be concise."), base().remove(0)];
    check_golden(
        &system_tools,
        &tools,
        &ChatTemplateOptions::default(),
        "system_tools",
    );
    system_tools.insert(1, message(ChatRole::Developer, "Use metric units."));
    check_golden(
        &system_tools,
        &tools,
        &ChatTemplateOptions::default(),
        "developer_tools",
    );

    let mut vision = vec![
        message(ChatRole::User, "\u{2003} before  after \u{a0}"),
        message(ChatRole::Assistant, "Done."),
        message(ChatRole::User, " \t \n"),
    ];
    vision[0].images.push(image_part("\u{2003} before ".len()));
    vision[2].images.push(image_part(2));
    options = no_thinking();
    check_golden(&vision, &[], &options, "vision_whitespace");
    options.add_vision_id = true;
    check_golden(&vision, &[], &options, "vision_ids");
    let rendered = render(&vision, &[], &options).unwrap();
    assert_eq!(rendered.image_offsets.len(), 2);
    for offset in rendered.image_offsets {
        assert_eq!(&rendered.text[offset..offset + 13], "<|image_pad|>");
    }

    let mut call = with_thought(ChatRole::Assistant, "", "Use the tool.");
    call.tool_calls.push(weather_call(""));
    options = ChatTemplateOptions {
        reasoning_effort: QwenReasoningEffort::Medium,
        preserve_thinking: false,
        ..ChatTemplateOptions::default()
    };
    check_golden(
        &[
            base().remove(0),
            call,
            message(ChatRole::User, "<tool_response>21 C</tool_response>"),
        ],
        &[],
        &options,
        "tool_loop_preserve",
    );
    // The reference template rejects late system/developer messages; they
    // join the leading system turn in their order.
    for role in [ChatRole::System, ChatRole::Developer] {
        let h = history();
        let late = [
            message(ChatRole::System, "Be concise."),
            h[0].clone(),
            h[1].clone(),
            message(role, "Late instructions"),
            h[2].clone(),
        ];
        let leading = [
            late[0].clone(),
            late[3].clone(),
            late[1].clone(),
            late[2].clone(),
            late[4].clone(),
        ];
        let rendered = render(&late, &tools, &options).unwrap();
        assert_eq!(rendered, render(&leading, &tools, &options).unwrap());
        assert!(
            rendered
                .text
                .contains("Be concise.\nLate instructions<|im_end|>"),
            "{}",
            rendered.text
        );
    }
    for role in [
        ChatRole::System,
        ChatRole::Developer,
        ChatRole::Assistant,
        ChatRole::Tool,
    ] {
        assert_eq!(
            render(
                &[message(role, "No user query")],
                &[],
                &ChatTemplateOptions::default()
            )
            .unwrap_err(),
            "No user query found in messages."
        );
    }
    assert!(
        render(
            &[message(
                ChatRole::User,
                "<tool_response>result</tool_response>"
            )],
            &[],
            &ChatTemplateOptions::default()
        )
        .is_err(),
        "a wrapped tool response is not a user query"
    );
}

/// GPT-2's byte-to-character map of byte-level BPE vocabularies.
fn byte_level(byte: u8) -> char {
    let direct = |b: u8| (b'!'..=b'~').contains(&b) || (0xa1..=0xac).contains(&b) || b >= 0xae;
    if direct(byte) {
        return char::from(byte);
    }
    let shifted = (0..byte).filter(|&b| !direct(b)).count() as u32;
    char::from_u32(256 + shifted).unwrap()
}

/// Single-byte tokens plus the four framing tokens.
fn byte_tokenizer() -> QwenTokenizer {
    let mut tokens: Vec<Vec<u8>> = (0..=255u8)
        .map(|b| byte_level(b).to_string().into_bytes())
        .collect();
    for special in ["<|im_start|>", "<|im_end|>", "<think>", "</think>"] {
        tokens.push(special.as_bytes().to_vec());
    }
    let bytes = gguf_image(&[], &[("tokenizer.ggml.tokens", tokens)]);
    QwenTokenizer::from_gguf(&Gguf::parse(&bytes).unwrap()).unwrap()
}

#[test]
fn render_and_tokenize_round_trips() {
    let tokenizer = byte_tokenizer();

    let ids = render_and_tokenize(
        &tokenizer,
        &[message(ChatRole::User, "Hello")],
        &[],
        &no_thinking(),
    )
    .unwrap();
    assert_eq!(ids[0], 256, "<|im_start|> is one added token");
    assert_eq!(
        tokenizer.decode(&ids),
        b"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );

    let mut image = message(ChatRole::User, "Look");
    image.images.push(image_part(0));
    assert_eq!(
        render_and_tokenize(&tokenizer, &[image], &[], &no_thinking()).unwrap_err(),
        "image messages require the model's vision prompt preparation"
    );
}

/// Serving checkpoints a thinking prompt for the next turn. When that turn
/// replays the assistant without reasoning, the generation suffix
/// `<think>\n` renders as `<think>\n\n`, which BPE merges into one token:
/// the full prompt stops being a token prefix, the prompt before the
/// generation suffix does not.
#[test]
fn empty_reasoning_replay_changes_the_thinking_suffix_tokens() {
    let newline = byte_level(b'\n').to_string();
    let mut tokens: Vec<Vec<u8>> = (0..=255u8)
        .map(|b| byte_level(b).to_string().into_bytes())
        .collect();
    for special in ["<|im_start|>", "<|im_end|>", "<think>", "</think>"] {
        tokens.push(special.as_bytes().to_vec());
    }
    tokens.push(format!("{newline}{newline}").into_bytes());
    let bytes = gguf_image(
        &[],
        &[
            ("tokenizer.ggml.tokens", tokens),
            (
                "tokenizer.ggml.merges",
                vec![format!("{newline} {newline}").into_bytes()],
            ),
        ],
    );
    let tokenizer = QwenTokenizer::from_gguf(&Gguf::parse(&bytes).unwrap()).unwrap();
    let options = ChatTemplateOptions::default();
    assert!(options.enable_thinking && options.preserve_thinking);

    let prompt = render_and_tokenize(
        &tokenizer,
        &[message(ChatRole::User, "Hello")],
        &[],
        &options,
    )
    .unwrap();
    let suffix = tokenizer.encode(chat_template::generation_prompt(true));
    assert!(
        prompt.len() > suffix.len() && prompt.ends_with(&suffix),
        "the first turn ends with the thinking generation suffix"
    );
    let next_turn = |thought: &str| {
        let messages = [
            message(ChatRole::User, "Hello"),
            with_thought(ChatRole::Assistant, "Hi", thought),
            message(ChatRole::User, "Again"),
        ];
        render_and_tokenize(&tokenizer, &messages, &[], &options).unwrap()
    };
    let stable = &prompt[..prompt.len() - suffix.len()];

    let dropped = next_turn("");
    assert!(
        !dropped.starts_with(&prompt),
        "empty reasoning replay merges the suffix newline"
    );
    assert!(
        dropped.starts_with(stable),
        "the prompt before the generation suffix survives empty reasoning"
    );
    assert!(
        next_turn("Greeting.").starts_with(&prompt),
        "replayed reasoning keeps the full prompt as a prefix"
    );
}

/// A 64×64 PNG, the fixture of `TestToolReasoningCheckpoint`.
const FIXTURE_PNG: &[u8] =
    b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x40\x00\x00\
    \x00\x40\x08\x02\x00\x00\x00\x25\x0b\xe6\x89\x00\x00\x00\x62\x49\x44\x41\x54\x78\x9c\xed\
    \xcf\x31\x0d\x00\x20\x00\xc0\x30\x40\x03\xfe\x35\x22\x01\x11\x1c\x0d\xc9\xaa\x60\x9b\x67\
    \xef\xf1\xb3\xa5\x03\x5e\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\
    \x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\
    \x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\x35\xa0\
    \x35\xa0\x5d\x08\xbb\x01\x98\x4a\x4b\x8b\x55\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\
    \x82";

/// With reasoning dropped (`preserve_thinking=false`), a new user turn
/// removes the reasoning of the preceding tool cycle: the stable prefix ends
/// before that cycle's first assistant, after every image, and survives the
/// next user turn in every thinking mode (`TestToolReasoningCheckpoint`).
#[test]
fn tool_reasoning_checkpoint_survives_the_next_user_turn() {
    let mut tokens: Vec<Vec<u8>> = (0..=255u8)
        .map(|b| byte_level(b).to_string().into_bytes())
        .collect();
    for special in [
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<|vision_start|>",
        "<|vision_end|>",
    ] {
        tokens.push(special.as_bytes().to_vec());
    }
    let bytes = gguf_image(&[], &[("tokenizer.ggml.tokens", tokens)]);
    let tokenizer = QwenTokenizer::from_gguf(&Gguf::parse(&bytes).unwrap()).unwrap();
    for images in [false, true] {
        for thinking in [false, true] {
            for preserve in [false, true] {
                for thought in ["", "Read the fixture."] {
                    let mut user = message(ChatRole::User, "Read the fixture.");
                    if images {
                        user.images.push(ImagePart {
                            offset: 0,
                            bytes: Arc::new(FIXTURE_PNG.to_vec()),
                        });
                    }
                    let mut assistant = with_thought(ChatRole::Assistant, "", thought);
                    assistant.tool_calls.push(ToolCall {
                        id: "call".into(),
                        name: "read_fixture".into(),
                        arguments: Vec::new(),
                    });
                    let mut messages = vec![
                        user,
                        assistant,
                        message(ChatRole::Tool, "The fixture is ready."),
                    ];
                    let options = ChatTemplateOptions {
                        enable_thinking: thinking,
                        preserve_thinking: preserve,
                        ..ChatTemplateOptions::default()
                    };
                    let case = format!(
                        "images={images} thinking={thinking} preserve={preserve} thought={thought:?}"
                    );
                    let prepare = |messages: &[ChatMessage]| {
                        vision::prepare(&tokenizer, messages, &[], &options, "fixture", 4096, None)
                            .unwrap()
                    };
                    let prompt = prepare(&messages);
                    messages.push(message(ChatRole::User, "."));
                    let continued = prepare(&messages);
                    let stable = &prompt.tokens[..prompt.stable_prefix_tokens];
                    assert!(continued.tokens.starts_with(stable), "{case}");
                    if images {
                        assert!(
                            prompt.stable_prefix_tokens >= prompt.rope.prefix_length() as usize,
                            "{case}: the boundary follows every image"
                        );
                    }
                    messages.pop();
                    let rendered = render(&messages, &[], &options).unwrap();
                    if preserve {
                        assert_eq!(
                            rendered.stable_prefix_bytes,
                            rendered.text.len() - chat_template::generation_prompt(thinking).len(),
                            "{case}"
                        );
                    } else {
                        assert_eq!(
                            Some(rendered.stable_prefix_bytes),
                            rendered.text.find("<|im_start|>assistant\n"),
                            "{case}"
                        );
                        assert!(
                            prompt.stable_prefix_tokens < continued.stable_prefix_tokens,
                            "{case}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn prepared_prompts_mark_the_stable_prefix() {
    let tokenizer = byte_tokenizer();
    let request = |reasoning: ReasoningOptions, tools: Vec<ChatTool>| ChatRequest {
        tools,
        reasoning,
        ..ChatRequest::new(vec![message(ChatRole::User, "Hi")])
    };
    let no_thinking = ReasoningOptions {
        enabled: Some(false),
        ..ReasoningOptions::default()
    };
    let suffix = tokenizer
        .encode(chat_template::generation_prompt(false))
        .len();
    // With retained reasoning and no tools too: the next turn may replay
    // the assistant differently.
    for prepared in [
        prepare_prompt(&tokenizer, &request(no_thinking, Vec::new()), 4096).unwrap(),
        prepare_prompt(
            &tokenizer,
            &request(
                ReasoningOptions {
                    preserve_thinking: Some(false),
                    ..no_thinking
                },
                Vec::new(),
            ),
            4096,
        )
        .unwrap(),
        prepare_prompt(
            &tokenizer,
            &request(no_thinking, vec![weather_tool("Get weather")]),
            4096,
        )
        .unwrap(),
    ] {
        assert_eq!(prepared.cache_prefix_tokens, prepared.tokens.len() - suffix);
    }

    let length = prepare_prompt(&tokenizer, &request(no_thinking, Vec::new()), 8).unwrap_err();
    assert_eq!(
        (length.kind(), length.message()),
        (
            gufo_core::text::ErrorKind::Length,
            "prompt exceeds the 8-token context; increase --context or shorten the conversation"
        )
    );
    let mut image = request(no_thinking, Vec::new());
    image.messages[0].images.push(image_part(0));
    assert_eq!(
        prepare_prompt(&tokenizer, &image, 4096)
            .unwrap_err()
            .message(),
        "image input requires the model's matching mmproj"
    );
    image.messages[0].images[0].offset = 5;
    assert_eq!(
        prepare_prompt(&tokenizer, &image, 4096)
            .unwrap_err()
            .message(),
        "invalid user or tool image content"
    );
}

#[test]
fn chat_corpus_conformance() {
    let generation = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    assert_eq!(
        render_text(
            &[message(ChatRole::User, "Hello, Strix Halo!")],
            &no_thinking()
        ),
        format!("<|im_start|>user\nHello, Strix Halo!<|im_end|>\n{generation}")
    );
    assert_eq!(
        render_text(
            &[
                message(ChatRole::User, "Fetch weather."),
                message(ChatRole::Tool, r#"{"temp": 22, "city": "Rome"}"#),
            ],
            &no_thinking()
        ),
        format!(
            "<|im_start|>user\nFetch weather.<|im_end|>\n<|im_start|>user\n<tool_response>\n\
             {{\"temp\": 22, \"city\": \"Rome\"}}\n</tool_response><|im_end|>\n{generation}"
        )
    );
    assert_eq!(
        render_text(&[message(ChatRole::User, "你好，世界！🚀")], &no_thinking()),
        format!("<|im_start|>user\n你好，世界！🚀<|im_end|>\n{generation}")
    );
}

#[test]
fn tool_rendering() {
    let mut assistant = message(ChatRole::Assistant, "");
    assistant.tool_calls.push(weather_call("call_weather"));
    let messages = [
        message(ChatRole::System, "Be concise."),
        message(ChatRole::User, "What is the weather?"),
        assistant,
    ];
    // `tool_choice: required` adds no instruction: the grammar enforces the
    // call, and switching it keeps the cached prefix.
    let rendered = render(
        &messages,
        &[weather_tool("Return current weather")],
        &ChatTemplateOptions::default(),
    )
    .unwrap()
    .text;
    assert!(rendered.contains("# Tools"));
    assert!(rendered.contains(r#""name": "get_weather""#));
    assert!(rendered.contains("<function=get_weather>"));
    assert!(rendered.contains("<parameter=city>\nRome\n</parameter>"));
    assert!(!rendered.contains("You must call at least one available function"));
}

#[test]
fn tool_replay_preserves_generated_prefix() {
    let mut assistant = with_thought(ChatRole::Assistant, "", "I should read the requested file.");
    assistant.tool_calls.push(ToolCall {
        id: "call_read".into(),
        name: "read".into(),
        arguments: vec![ToolArgument {
            name: "path".into(),
            value: "/etc/hostname".into(),
            is_string: true,
        }],
    });
    let options = ChatTemplateOptions {
        add_generation_prompt: false,
        enable_thinking: false,
        ..ChatTemplateOptions::default()
    };
    assert_eq!(
        render_text(
            &[message(ChatRole::User, "Read the file."), assistant],
            &options
        ),
        "<|im_start|>user\nRead the file.<|im_end|>\n<|im_start|>assistant\n\
         <think>\nI should read the requested file.\n</think>\n\n\
         <tool_call>\n<function=read>\n<parameter=path>\n/etc/hostname\n</parameter>\n\
         </function>\n</tool_call><|im_end|>\n"
    );
}

#[test]
fn image_offsets_off_a_character_boundary_are_rejected() {
    let mut split = message(ChatRole::User, "é");
    split.images.push(image_part(1));
    assert_eq!(
        render(&[split], &[], &ChatTemplateOptions::default()).unwrap_err(),
        "invalid user or tool image content"
    );
    assert_eq!(
        chat_template::generation_prompt(true),
        "<|im_start|>assistant\n<think>\n"
    );
}

#[cfg(feature = "models")]
mod hugging_face_tokens {
    use std::path::PathBuf;

    use gufo_core::gguf::read_header;

    use super::*;

    fn model() -> PathBuf {
        std::env::var_os("GUFO_QWEN27B_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(std::env::var_os("HOME").unwrap()).join(
                    ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/\
                     4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf",
                )
            })
    }

    fn token_sha256(ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        sha256(&bytes)
    }

    fn check(
        tokenizer: &QwenTokenizer,
        messages: &[ChatMessage],
        tools: &[ChatTool],
        options: &ChatTemplateOptions,
        name: &str,
    ) {
        check_golden(messages, tools, options, name);
        let ids = render_and_tokenize(tokenizer, messages, tools, options).unwrap();
        let expected = golden(name);
        assert_eq!(
            ids.len() as u64,
            expected["token_count"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(
            token_sha256(&ids),
            expected["token_ids_le_u32_sha256"].as_str().unwrap(),
            "{name}"
        );
    }

    #[test]
    fn prompts_tokenize_like_hugging_face() {
        let (header, len) = read_header(&model()).unwrap();
        let gguf = Gguf::parse_prefix(&header, len).unwrap();
        assert_eq!(gguf.string("tokenizer.ggml.pre"), Some(&b"qwen35"[..]));
        ChatTemplate::validate_gguf(&gguf, is_qwen38_named(&gguf)).unwrap();
        let template = ChatTemplate::from_gguf(&gguf, is_qwen38_named(&gguf)).unwrap();
        assert_eq!(template.profile(), Profile::Qwen38Reasoning);
        let tokenizer = QwenTokenizer::from_gguf(&gguf).unwrap();

        let fixture: Value = serde_json::from_str(GOLDENS).unwrap();
        for example in fixture["qwen"]["tokenization"].as_array().unwrap() {
            let ids = tokenizer.encode(example["text"].as_str().unwrap());
            let name = example["name"].as_str().unwrap();
            assert_eq!(
                ids.len() as u64,
                example["token_count"].as_u64().unwrap(),
                "{name}"
            );
            assert_eq!(
                token_sha256(&ids),
                example["token_ids_le_u32_sha256"].as_str().unwrap(),
                "{name}"
            );
        }

        let tools = [weather_tool("Get weather")];
        check(
            &tokenizer,
            &base(),
            &[],
            &ChatTemplateOptions::default(),
            "thinking_xhigh",
        );
        let mut options = no_thinking();
        check(&tokenizer, &base(), &[], &options, "chat");
        options.enable_thinking = true;
        options.reasoning_effort = QwenReasoningEffort::Low;
        check(&tokenizer, &base(), &[], &options, "thinking_low");
        options.reasoning_effort = QwenReasoningEffort::Medium;
        check(&tokenizer, &base(), &[], &options, "thinking_medium");
        options.reasoning_effort = QwenReasoningEffort::XHigh;
        check(&tokenizer, &base(), &[], &options, "thinking_xhigh");
        options.reasoning_effort = QwenReasoningEffort::Medium;
        options.preserve_thinking = false;
        check(&tokenizer, &history(), &[], &options, "history_drop");
        options.enable_thinking = false;
        options.preserve_thinking = true;
        check(&tokenizer, &history(), &[], &options, "history_preserve");
        check(&tokenizer, &base(), &tools, &no_thinking(), "tools");
        let mut system_tools = vec![message(ChatRole::System, "Be concise."), base().remove(0)];
        let defaults = ChatTemplateOptions::default();
        check(&tokenizer, &system_tools, &tools, &defaults, "system_tools");
        system_tools.insert(1, message(ChatRole::Developer, "Use metric units."));
        check(
            &tokenizer,
            &system_tools,
            &tools,
            &defaults,
            "developer_tools",
        );
    }
}

/// Single-byte tokens plus `specials`, numbered from 256.
fn special_tokenizer(specials: &[&str]) -> QwenTokenizer {
    let mut tokens: Vec<Vec<u8>> = (0..=255u8)
        .map(|b| byte_level(b).to_string().into_bytes())
        .collect();
    tokens.extend(specials.iter().map(|special| special.as_bytes().to_vec()));
    let bytes = gguf_image(&[], &[("tokenizer.ggml.tokens", tokens)]);
    QwenTokenizer::from_gguf(&Gguf::parse(&bytes).unwrap()).unwrap()
}

const CONTENT_SPECIALS: [&str; 5] = [
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<|endoftext|>",
];

/// Message text is text: a client that spells a vocabulary token does not
/// inject that control token, which would end a turn in the middle of the
/// conversation; its characters still reach the model and the framing stays
/// framing (#383).
#[test]
fn content_spelling_a_token_is_not_parsed_as_one() {
    let tokenizer = special_tokenizer(&CONTENT_SPECIALS);
    let text = "the vocabulary token `<|endoftext|>` is written here as prose, and the \
                thinking phase ends with `</think>`";
    let ids = render_and_tokenize(
        &tokenizer,
        &[message(ChatRole::User, text)],
        &[],
        &no_thinking(),
    )
    .unwrap();
    // The template never writes this token: only message text could.
    let injected = tokenizer.special("<|endoftext|>").unwrap();
    assert!(
        !ids.contains(&injected),
        "content spelling a control token does not inject it"
    );
    assert!(
        ids.contains(&tokenizer.special("<|im_start|>").unwrap()),
        "template framing still tokenizes as control tokens"
    );
    assert!(
        String::from_utf8(tokenizer.decode(&ids))
            .unwrap()
            .contains(text),
        "the literal characters of the message reach the model"
    );
}

/// Replayed tool arguments, string or JSON, are data like the tool result
///.
#[test]
fn tool_replay_arguments_are_content() {
    let tokenizer =
        special_tokenizer(&["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"]);
    let options = ChatTemplateOptions {
        add_generation_prompt: false,
        ..no_thinking()
    };
    for is_string in [true, false] {
        let value = if is_string {
            "EOS = \"<|im_end|>\""
        } else {
            r#"{"text":"<|im_end|><tool_call></tool_call>"}"#
        };
        let mut assistant = message(ChatRole::Assistant, "");
        assistant.tool_calls.push(ToolCall {
            id: "call_write".into(),
            name: "write".into(),
            arguments: vec![ToolArgument {
                name: "content".into(),
                value: value.into(),
                is_string,
            }],
        });
        let messages = [
            message(ChatRole::User, "Write this file."),
            assistant,
            message(ChatRole::Tool, value),
        ];
        let rendered = render(&messages, &[], &options).unwrap();
        let text = &rendered.text;
        let argument = text.find(value).unwrap();
        let result = argument + value.len() + text[argument + value.len()..].find(value).unwrap();
        let encode = |begin: usize| {
            encode_rendered(
                &tokenizer,
                text,
                begin,
                begin + value.len(),
                &rendered.content_spans,
            )
        };
        assert_eq!(encode(argument), encode(result), "{value}");
        assert_eq!(
            encode(argument),
            tokenizer.encode_text(value),
            "replayed arguments match the tool result's text, literal EOS included"
        );
        let direct = render_and_tokenize(&tokenizer, &messages, &[], &options).unwrap();
        let prepared =
            vision::prepare(&tokenizer, &messages, &[], &options, "", 1024, None).unwrap();
        assert_eq!(
            prepared.tokens, direct,
            "serving and direct tokenization agree on replayed arguments"
        );
        assert_eq!(
            tokenizer.decode(&direct),
            text.as_bytes(),
            "replay preserves the rendered conversation exactly"
        );
        let count = |id: u32| direct.iter().filter(|&&token| token == id).count();
        assert_eq!(count(257), 3, "only message boundaries end a turn");
        assert_eq!(
            (count(258), count(259)),
            (1, 1),
            "native call tags stay framing while argument spellings stay text"
        );
    }
}

/// Serving prepares every request through the vision path, text-only ones
/// included.
#[test]
fn vision_preparation_reads_content_as_text() {
    let tokenizer = special_tokenizer(&CONTENT_SPECIALS);
    let text = "Literal <|endoftext|> text";
    let messages = [message(ChatRole::User, text)];
    let injected = tokenizer.special("<|endoftext|>").unwrap();
    let direct = render_and_tokenize(&tokenizer, &messages, &[], &no_thinking()).unwrap();
    assert!(
        !direct.contains(&injected),
        "the template path injects nothing"
    );
    let prepared =
        vision::prepare(&tokenizer, &messages, &[], &no_thinking(), "", 512, None).unwrap();
    assert_eq!(prepared.tokens, direct, "both entry points agree");
    assert!(
        String::from_utf8(tokenizer.decode(&prepared.tokens))
            .unwrap()
            .contains(text),
        "the literal characters reach the model"
    );
    assert!(
        prepared
            .tokens
            .contains(&tokenizer.special("<|im_start|>").unwrap()),
        "template framing still tokenizes as control tokens"
    );
}

/// Content around a picture reads as text in every slice the preparation
/// encodes.
#[test]
fn encode_rendered_reads_image_content_as_text() {
    let mut specials = CONTENT_SPECIALS.to_vec();
    specials.push("<|image_pad|>");
    let tokenizer = special_tokenizer(&specials);
    let text = "Literal <|endoftext|> text";
    let mut user = message(ChatRole::User, text);
    user.images.push(image_part(0));
    let rendered = render(&[user], &[], &no_thinking()).unwrap();
    let image = rendered.image_offsets[0];
    assert!(rendered.text[image..].starts_with("<|image_pad|>"));
    let encode = |begin: usize, end: usize| {
        encode_rendered(
            &tokenizer,
            &rendered.text,
            begin,
            end,
            &rendered.content_spans,
        )
    };
    let injected = tokenizer.special("<|endoftext|>").unwrap();
    let whole = encode(0, rendered.text.len());
    for slice in [
        &whole,
        &encode(0, image),
        &encode(image + "<|image_pad|>".len(), rendered.text.len()),
    ] {
        assert!(
            !slice.contains(&injected),
            "no slice injects a spelled token"
        );
    }
    assert!(whole.contains(&tokenizer.special("<|im_start|>").unwrap()));
    assert!(
        String::from_utf8(tokenizer.decode(&whole))
            .unwrap()
            .contains(text)
    );
}

/// The server's instructions after a system message render as before but
/// are framing: only the client's part is content.
#[test]
fn framing_suffix_follows_system_content() {
    for (content, suffix) in [
        ("Be brief.", "\n\nUse the JSON form."),
        ("  Be brief.  ", "\n\nUse the JSON form.  "),
        ("", "Use the JSON form."),
        ("   ", "\n\nUse the JSON form."),
    ] {
        let framed = ChatMessage {
            framing_suffix: suffix.into(),
            ..message(ChatRole::System, content)
        };
        let user = message(ChatRole::User, "Hi");
        let rendered = render(&[framed, user.clone()], &[], &no_thinking()).unwrap();
        let joined = render(
            &[
                message(ChatRole::System, &format!("{content}{suffix}")),
                user,
            ],
            &[],
            &no_thinking(),
        )
        .unwrap();
        assert_eq!(rendered.text, joined.text, "{content:?}");
        let system: Vec<&str> = rendered
            .content_spans
            .iter()
            .map(|span| &rendered.text[span.offset..span.offset + span.size])
            .filter(|span| *span != "Hi")
            .collect();
        let client = content.trim_start();
        let expected: Vec<&str> = if client.is_empty() {
            Vec::new()
        } else {
            vec![client]
        };
        assert_eq!(system, expected, "{content:?}");
    }
}

/// A spelled token before the stable boundary must not change how the
/// mutable suffix tokenizes: with a `.\n` merge, reasoning ending in a full
/// stop reads the same in the whole prompt and in the suffix.
#[test]
fn spelled_token_before_the_stable_boundary_keeps_the_suffix() {
    let newline = byte_level(b'\n');
    let mut tokens: Vec<Vec<u8>> = (0..=255u8)
        .map(|b| byte_level(b).to_string().into_bytes())
        .collect();
    tokens.extend(CONTENT_SPECIALS.iter().map(|s| s.as_bytes().to_vec()));
    tokens.push(format!(".{newline}").into_bytes());
    let bytes = gguf_image(
        &[],
        &[
            ("tokenizer.ggml.tokens", tokens),
            (
                "tokenizer.ggml.merges",
                vec![format!(". {newline}").into_bytes()],
            ),
        ],
    );
    let tokenizer = QwenTokenizer::from_gguf(&Gguf::parse(&bytes).unwrap()).unwrap();
    let merged = tokenizer.encode(".\n");
    assert_eq!(
        merged.len(),
        1,
        "the fixture merges a full stop and newline"
    );
    let options = ChatTemplateOptions {
        preserve_thinking: false,
        ..ChatTemplateOptions::default()
    };
    for spelled in ["Explain <|endoftext|>.", "Explain the stop."] {
        let mut assistant = with_thought(ChatRole::Assistant, "", "Let me check.");
        assistant.tool_calls.push(ToolCall {
            id: "call_read".into(),
            name: "read".into(),
            arguments: vec![ToolArgument {
                name: "path".into(),
                value: "a.txt".into(),
                is_string: true,
            }],
        });
        let messages = [
            message(ChatRole::User, spelled),
            assistant,
            message(ChatRole::Tool, "Done."),
        ];
        let prompt = vision::prepare(&tokenizer, &messages, &[], &options, "", 4096, None)
            .unwrap_or_else(|error| panic!("{spelled:?}: {error}"));
        assert!(
            prompt.tokens[prompt.stable_prefix_tokens..].contains(&merged[0]),
            "{spelled:?}: ordinary reasoning keeps its ordinary tokens"
        );
    }
}

/// Agent clients end every request with context the next request replaces:
/// the stable boundary then starts that final user turn, after a tool
/// result or another user message only.
#[test]
fn stable_boundary_precedes_a_replaced_final_user_turn() {
    let options = ChatTemplateOptions::default();
    let render = |messages: &[ChatMessage]| {
        let rendered = render(messages, &[], &options).unwrap();
        (rendered.text, rendered.stable_prefix_bytes)
    };
    let history = vec![
        message(ChatRole::System, "You are an agent."),
        message(ChatRole::User, "Start the task."),
        message(ChatRole::Assistant, "Reading."),
        message(ChatRole::Tool, "file contents"),
    ];
    let with = |extra: &[ChatMessage]| {
        let mut messages = history.clone();
        messages.extend_from_slice(extra);
        messages
    };
    let first = with(&[message(ChatRole::User, "Runtime context, turn 1.")]);
    let second = with(&[
        message(ChatRole::Assistant, "Done."),
        message(ChatRole::User, "Runtime context, turn 2."),
    ]);
    let (first_text, first_stable) = render(&first);
    let (second_text, second_stable) = render(&second);
    assert!(
        first_stable < first_text.len() && first_text[first_stable..].starts_with("<|im_start|>"),
        "the boundary starts the final user turn"
    );
    assert!(
        second_text.starts_with(&first_text[..first_stable]),
        "the boundary prefixes a request that replaces the final user turn"
    );
    assert!(
        second_stable > first_stable,
        "the boundary advances with the history"
    );

    // A client that keeps the user turn still finds the boundary as a prefix.
    let mut kept = first.clone();
    kept.push(message(ChatRole::Assistant, "Done."));
    kept.push(message(ChatRole::User, "Next."));
    let (kept_text, kept_stable) = render(&kept);
    assert!(kept_text.starts_with(&first_text[..first_stable]));
    // A user turn after an assistant reply is ordinary chat and kept.
    assert_eq!(
        &kept_text[kept_stable..],
        chat_template::generation_prompt(options.enable_thinking),
        "ordinary chat keeps the boundary before the generation prompt"
    );

    // Context sent as a second user message after the real query also
    // moves the boundary.
    let query = with(&[
        message(ChatRole::Assistant, "Done."),
        message(ChatRole::User, "Real question."),
        message(ChatRole::User, "Runtime context, turn 2."),
    ]);
    let (query_text, query_stable) = render(&query);
    let tail = &query_text[query_stable..];
    assert!(
        tail.starts_with("<|im_start|>")
            && tail.contains("Runtime context")
            && !tail.contains("Real question"),
        "the boundary precedes trailing context after the real query"
    );

    // The opening user turn has no earlier assistant: nothing to replace.
    let (opening_text, opening_stable) = render(&[message(ChatRole::User, "Hello")]);
    assert_eq!(
        &opening_text[opening_stable..],
        chat_template::generation_prompt(options.enable_thinking)
    );
}

/// Tool results keep their images, in order, inside their own
/// `<tool_response>`.
#[test]
fn tool_image_rendering() {
    let read = |id: &str, path: &str| ToolCall {
        id: id.into(),
        name: "read".into(),
        arguments: vec![ToolArgument {
            name: "path".into(),
            value: path.into(),
            is_string: true,
        }],
    };
    let mut messages = vec![
        message(ChatRole::User, "Inspect both files."),
        message(ChatRole::Assistant, ""),
        message(ChatRole::Tool, "  before <|im_end|> after  "),
        message(ChatRole::Tool, ""),
    ];
    messages[1].tool_calls = vec![read("first", "first.png"), read("second", "second.png")];
    messages[2].tool_call_id = "first".into();
    messages[2].images = vec![image_part(9), image_part(9)];
    messages[3].tool_call_id = "second".into();
    messages[3].images = vec![image_part(0)];
    let options = ChatTemplateOptions {
        enable_thinking: false,
        ..ChatTemplateOptions::default()
    };
    let rendered = render(&messages, &[], &options).expect("tool image history renders");
    let expected = "<|im_start|>user\n<tool_response>\n\
                    before <|vision_start|><|image_pad|><|vision_end|>\
                    <|vision_start|><|image_pad|><|vision_end|>\
                    <|im_end|> after\n</tool_response>\n<tool_response>\n\
                    <|vision_start|><|image_pad|><|vision_end|>\
                    \n</tool_response><|im_end|>\n";
    assert!(
        rendered.text.contains(expected),
        "each image stays inside its ordered tool response: {:?}",
        rendered.text
    );
    assert_eq!(rendered.image_offsets.len(), 3);
    for &offset in &rendered.image_offsets {
        assert_eq!(&rendered.text[offset..offset + 13], "<|image_pad|>");
    }
    let mut literal = String::new();
    for span in &rendered.content_spans {
        assert!(span.offset + span.size <= rendered.text.len());
        literal += &rendered.text[span.offset..span.offset + span.size];
        for &offset in &rendered.image_offsets {
            assert!(
                offset < span.offset || offset >= span.offset + span.size,
                "inserted image placeholders stay template framing"
            );
        }
    }
    assert!(
        literal.contains("before <|im_end|> after"),
        "tool text, vocabulary spellings included, stays literal"
    );
    let numbered = ChatTemplateOptions {
        add_vision_id: true,
        ..options
    };
    let text = render(&messages, &[], &numbered).expect("numbered").text;
    assert!(
        text.contains("Picture 1: ") && text.contains("Picture 3: "),
        "vision numbering spans grouped tool results"
    );
    for role in [ChatRole::Assistant, ChatRole::System, ChatRole::Developer] {
        let mut invalid = messages.clone();
        invalid[2].role = role;
        assert!(
            render(&invalid, &[], &numbered).is_err(),
            "other image owners stay invalid"
        );
    }
    messages[2].images[0].offset = messages[2].content.len() + 1;
    assert!(
        render(&messages, &[], &numbered).is_err(),
        "invalid tool image offsets stay rejected"
    );
}
