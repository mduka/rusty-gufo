//! The chat template against the C++ formatter and engine
//! (`tests/fixtures/chat_template.json`, recorded by
//! `tests/reference/record_chat_template.py` with `tests/reference/ds4_ref_dump.cpp render`):
//! 22 conversations covering system and developer messages, thinking with
//! each effort, dropped and preserved reasoning, tool schemas with every
//! host, required calls, full definitions, assistant tool calls with
//! results in a different order, the function role, a trailing assistant
//! turn, and escapes, control characters and non-ASCII text. The rendered
//! text must be equal; the token IDs need the model file.
//!
//! Every case of the C++ `tests/models/deepseek_v4_flash/chat_template_test.cpp`
//! follows: the chat and thinking prefixes with the serving defaults, the
//! historical reasoning policy, the effort mapping, joined tool results, the
//! official DSML tool loop, parallel results restored to call order, full
//! tool definitions, the pinned Hugging Face digests
//! (`tests/fixtures/chat_template_hf_goldens.json`) and the pinned artifact
//! template check. With the `models` feature, `token_goldens.cpp`: the token
//! IDs of the Hugging Face golden conversations with the model's tokenizer.

#![allow(clippy::unwrap_used)]

mod common;

use gufo_core::json;
use gufo_core::text::{ReasoningEffort, ReasoningOptions};
use gufo_deepseek_v4_flash::chat_template::{
    ARTIFACT_TEMPLATE_SHA256, ChatMessage, ChatTemplateOptions, ChatTool, ToolArgument, ToolCall,
    reasoning_effort_name, render_chat, resolve_chat_options, validate_gguf_template,
};
use gufo_deepseek_v4_flash::gguf::Model;
use sha2::{Digest, Sha256};

const HF_GOLDENS: &[u8] =
    include_bytes!("../../../../tests/fixtures/chat_template_hf_goldens.json");
const ARTIFACT_TEMPLATE: &[u8] = include_bytes!("../reference/artifact_chat_template.jinja");

fn member(value: &json::Value, key: &str) -> String {
    value
        .find(key)
        .map_or(String::new(), |v| v.str().to_owned())
}

fn flag(value: Option<&json::Value>, key: &str) -> bool {
    value
        .and_then(|v| v.find(key))
        .is_some_and(|v| matches!(v, json::Value::Bool(true)))
}

/// The conversation of a fixture case, as `ds4_ref_dump render` builds it.
fn conversation(case: &json::Value) -> (Vec<ChatMessage>, Vec<ChatTool>, ChatTemplateOptions) {
    let items = |key: &str| case.find(key).map_or(Vec::new(), |v| v.items().to_vec());
    let messages = items("messages")
        .iter()
        .map(|m| ChatMessage {
            role: member(m, "role"),
            content: member(m, "content"),
            reasoning_content: member(m, "reasoning"),
            tool_call_id: member(m, "tool_call_id"),
            tool_calls: m.find("tool_calls").map_or(Vec::new(), |calls| {
                calls
                    .items()
                    .iter()
                    .map(|c| ToolCall {
                        name: member(c, "name"),
                        id: member(c, "id"),
                        arguments: c.find("arguments").map_or(Vec::new(), |arguments| {
                            arguments
                                .items()
                                .iter()
                                .map(|a| ToolArgument {
                                    name: member(a, "name"),
                                    value: member(a, "value"),
                                    is_string: !matches!(
                                        a.find("string"),
                                        Some(json::Value::Bool(false))
                                    ),
                                })
                                .collect()
                        }),
                    })
                    .collect()
            }),
        })
        .collect();
    let tools = items("tools")
        .iter()
        .map(|t| ChatTool {
            name: member(t, "name"),
            description: member(t, "description"),
            parameters_json: t
                .find("parameters")
                .map_or("{}".to_owned(), |v| v.str().to_owned()),
            definition_json: member(t, "definition"),
        })
        .collect();
    let o = case.find("options");
    let effort = match o.map(|o| member(o, "effort")).as_deref() {
        Some("minimal") => ReasoningEffort::Minimal,
        Some("medium") => ReasoningEffort::Medium,
        Some("high") => ReasoningEffort::High,
        Some("xhigh") => ReasoningEffort::XHigh,
        Some("max") => ReasoningEffort::Max,
        _ => ReasoningEffort::Low,
    };
    let options = ChatTemplateOptions {
        enable_thinking: flag(o, "thinking"),
        reasoning_effort: effort,
        preserve_thinking: flag(o, "preserve"),
        tools_present: flag(o, "tools_present"),
    };
    (messages, tools, options)
}

#[test]
fn rendering_matches_the_cpp_formatter() {
    let fixture = json::parse(include_bytes!("fixtures/chat_template.json")).unwrap();
    for case in fixture.find("cases").unwrap().items() {
        let (messages, tools, options) = conversation(case);
        assert_eq!(
            render_chat(&messages, &tools, &options),
            case.find("rendered").unwrap().str(),
            "{}",
            member(case, "name")
        );
    }
}

#[cfg(feature = "models")]
#[test]
fn chat_tokens_match_the_cpp_engine() {
    use gufo_deepseek_v4_flash::tokenizer::Vocab;

    let fixture = json::parse(include_bytes!("fixtures/chat_template.json")).unwrap();
    let model = Model::open(&common::model_path()).unwrap();
    validate_gguf_template(&model).unwrap();
    let vocab = Vocab::load(&model).unwrap();
    for case in fixture.find("cases").unwrap().items() {
        let (messages, tools, options) = conversation(case);
        let mut tokens = Vec::new();
        vocab.tokenize_rendered_chat(
            render_chat(&messages, &tools, &options).as_bytes(),
            &mut tokens,
        );
        let name = member(case, "name");
        assert_eq!(
            tokens.len(),
            case.member_size("token_count", 0),
            "{name}: token count"
        );
        assert_eq!(
            common::fnv(&common::token_bytes(&tokens)),
            member(case, "tokens"),
            "{name}: tokens"
        );
    }
}

#[test]
fn requests_default_to_thinking_at_high_effort() {
    let resolved = resolve_chat_options(&ReasoningOptions::default());
    assert!(resolved.enable_thinking && !resolved.preserve_thinking);
    assert_eq!(resolved.reasoning_effort, ReasoningEffort::High);
    let explicit = resolve_chat_options(&ReasoningOptions {
        enabled: Some(false),
        effort: Some(ReasoningEffort::Minimal),
        preserve_thinking: Some(true),
        budget_tokens: None,
    });
    assert!(!explicit.enable_thinking && explicit.preserve_thinking);
    assert_eq!(explicit.reasoning_effort, ReasoningEffort::Minimal);
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `GoldenCase(fixture, "deepseek", name)` of the Hugging Face goldens.
fn hf_golden(name: &str) -> json::Value {
    json::parse(HF_GOLDENS)
        .unwrap()
        .find("deepseek")
        .and_then(|model| model.find("cases"))
        .and_then(|cases| cases.find(name))
        .unwrap_or_else(|| panic!("the golden fixture has no deepseek case {name}"))
        .clone()
}

fn message(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: content.into(),
        ..ChatMessage::default()
    }
}

fn with_reasoning(role: &str, content: &str, reasoning: &str) -> ChatMessage {
    ChatMessage {
        reasoning_content: reasoning.into(),
        ..message(role, content)
    }
}

fn tool_result(content: &str, tool_call_id: &str) -> ChatMessage {
    ChatMessage {
        tool_call_id: tool_call_id.into(),
        ..message("tool", content)
    }
}

/// A string argument, the C++ default.
fn argument(name: &str, value: &str) -> ToolArgument {
    ToolArgument {
        name: name.into(),
        value: value.into(),
        is_string: true,
    }
}

fn thinking(effort: ReasoningEffort) -> ChatTemplateOptions {
    ChatTemplateOptions {
        enable_thinking: true,
        reasoning_effort: effort,
        ..ChatTemplateOptions::default()
    }
}

fn render(messages: &[ChatMessage], options: &ChatTemplateOptions) -> String {
    render_chat(messages, &[], options)
}

/// A `get_weather` call with its result and a follow-up request.
fn weather_conversation() -> (Vec<ChatMessage>, Vec<ChatTool>) {
    let assistant = ChatMessage {
        tool_calls: vec![ToolCall {
            name: "get_weather".into(),
            arguments: vec![argument("city", "Rome")],
            id: String::new(),
        }],
        ..with_reasoning("assistant", "", "Use the tool.")
    };
    let messages = vec![
        message("system", "Be concise."),
        message("user", "Weather in Rome?"),
        assistant,
        message("tool", "{\"temperature\":21}"),
        message("user", "Summarize."),
    ];
    let tools = vec![ChatTool {
        name: "get_weather".into(),
        description: "Get weather".into(),
        parameters_json:
            r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#
                .into(),
        definition_json: String::new(),
    }];
    (messages, tools)
}

/// Two parallel calls whose results arrive in reverse order around user text.
fn parallel_conversation() -> Vec<ChatMessage> {
    let read = |path: &str| ToolCall {
        name: "read".into(),
        arguments: vec![argument("path", path)],
        id: format!("call_{path}"),
    };
    vec![
        message("user", "Read A and B."),
        ChatMessage {
            tool_calls: vec![read("A"), read("B")],
            ..message("assistant", "")
        },
        tool_result("Result B", "call_B"),
        message("user", "Compare them."),
        tool_result("Result A", "call_A"),
    ]
}

/// A conversation of `chat_template_hf_goldens.json`.
struct HfCase {
    name: &'static str,
    messages: Vec<ChatMessage>,
    tools: Vec<ChatTool>,
    options: ChatTemplateOptions,
}

/// The Hugging Face golden conversations, as `token_goldens.cpp` builds them.
fn hf_cases() -> Vec<HfCase> {
    let base = || {
        vec![
            message("system", "Be concise."),
            message("user", "Name one color."),
        ]
    };
    let history = || {
        vec![
            message("user", "Name one color."),
            with_reasoning("assistant", "Red", "I should answer with one color."),
            message("user", "Name another."),
        ]
    };
    let preserve = |preserve_thinking| ChatTemplateOptions {
        preserve_thinking,
        ..thinking(ReasoningEffort::Low)
    };
    let (weather, weather_tools) = weather_conversation();
    let case = |name, messages, tools, options| HfCase {
        name,
        messages,
        tools,
        options,
    };
    vec![
        case("chat", base(), vec![], ChatTemplateOptions::default()),
        case(
            "thinking_low",
            base(),
            vec![],
            thinking(ReasoningEffort::Low),
        ),
        case(
            "thinking_high",
            base(),
            vec![],
            thinking(ReasoningEffort::High),
        ),
        case(
            "thinking_max",
            base(),
            vec![],
            thinking(ReasoningEffort::Max),
        ),
        case("history_drop", history(), vec![], preserve(false)),
        case("history_preserve", history(), vec![], preserve(true)),
        case(
            "tools",
            weather,
            weather_tools,
            thinking(ReasoningEffort::Low),
        ),
        case(
            "parallel_tools",
            parallel_conversation(),
            vec![],
            ChatTemplateOptions::default(),
        ),
    ]
}

/// `TestChatAndThinkingPrefixes`, with the serving defaults of
/// `ResolveDeepSeekChatOptions`.
#[test]
fn chat_and_thinking_prefixes() {
    let messages = [message("system", "Be concise."), message("user", "Answer.")];
    let chat = render(&messages, &ChatTemplateOptions::default());
    assert_eq!(
        chat, "<｜begin▁of▁sentence｜>Be concise.<｜User｜>Answer.<｜Assistant｜></think>",
        "Chat mode uses the no-thinking assistant prefix"
    );
    let thinking = render(&messages, &thinking(ReasoningEffort::High));
    assert!(
        thinking.contains("Reasoning Effort: Absolute maximum"),
        "High effort uses the pinned DeepSeek instruction"
    );
    assert!(
        thinking.ends_with("<｜Assistant｜><think>"),
        "Thinking mode starts inside a reasoning block"
    );

    let defaults = resolve_chat_options(&ReasoningOptions::default());
    assert!(
        defaults.enable_thinking && defaults.reasoning_effort == ReasoningEffort::High,
        "User-facing DeepSeek defaults match the API's thinking/high"
    );
    assert_eq!(
        render(&messages, &defaults),
        thinking,
        "Default serving prompt matches the official high-effort encoding"
    );
    let off = resolve_chat_options(&ReasoningOptions {
        enabled: Some(false),
        ..ReasoningOptions::default()
    });
    assert_eq!(
        render(&messages, &off),
        chat,
        "Explicit thinking off overrides the serving default"
    );
    let low = resolve_chat_options(&ReasoningOptions {
        effort: Some(ReasoningEffort::Low),
        preserve_thinking: Some(true),
        ..ReasoningOptions::default()
    });
    assert!(
        low.enable_thinking
            && low.reasoning_effort == ReasoningEffort::Low
            && low.preserve_thinking,
        "Explicit effort and history policy override serving defaults"
    );
}

/// `TestHistoricalThinkingPolicy`.
#[test]
fn historical_thinking_policy() {
    let messages = [
        message("user", "First."),
        with_reasoning("assistant", "Visible answer.", "Private reasoning."),
        message("user", "Second."),
    ];
    let mut options = thinking(ReasoningEffort::Low);
    let dropped = render(&messages, &options);
    assert!(
        !dropped.contains("Private reasoning."),
        "DeepSeek drops old thinking by default"
    );
    assert!(
        dropped.contains("<｜Assistant｜></think>Visible answer."),
        "Dropping thought preserves visible assistant content"
    );

    options.preserve_thinking = true;
    let preserved = render(&messages, &options);
    assert!(
        preserved.contains("<｜Assistant｜><think>Private reasoning.</think>Visible answer."),
        "Explicit preservation replays historical reasoning"
    );

    options.enable_thinking = false;
    let chat_mode = render(&messages, &options);
    assert!(
        !chat_mode.contains("Private reasoning."),
        "Standard DeepSeek chat mode never replays reasoning history"
    );
    assert!(
        chat_mode.contains("<｜Assistant｜></think>Visible answer."),
        "Chat mode reconstructs assistant history into visible content"
    );

    options.enable_thinking = true;
    options.preserve_thinking = false;
    options.tools_present = true;
    let tool_loop = render(&messages, &options);
    assert!(
        tool_loop.contains("Private reasoning."),
        "Tool conversations force reasoning preservation"
    );
}

/// `TestEffortMappingAndToolResults` of the C++ tests:
/// xhigh maps to high like DeepSeek's API.
#[test]
fn effort_mapping_and_tool_results() {
    assert_eq!(
        reasoning_effort_name(ReasoningEffort::Minimal),
        "low",
        "minimal maps to DeepSeek low"
    );
    assert_eq!(
        reasoning_effort_name(ReasoningEffort::Medium),
        "high",
        "medium maps to DeepSeek high"
    );
    assert_eq!(
        reasoning_effort_name(ReasoningEffort::XHigh),
        "high",
        "xhigh maps to DeepSeek high like the API"
    );
    assert_eq!(
        reasoning_effort_name(ReasoningEffort::Max),
        "max",
        "max selects DeepSeek's highest effort"
    );

    let messages = [message("tool", "one"), message("tool", "two")];
    assert!(
        render(&messages, &ChatTemplateOptions::default())
            .contains("<｜User｜><tool_result>one</tool_result>\n\n<tool_result>two</tool_result>"),
        "Consecutive tool results share one user turn"
    );
}

/// `TestOfficialDsmlToolLoop`.
#[test]
fn official_dsml_tool_loop() {
    let (messages, tools) = weather_conversation();
    let rendered = render_chat(&messages, &tools, &thinking(ReasoningEffort::Low));
    assert!(
        rendered.contains("<｜DSML｜tool_calls>"),
        "DeepSeek uses the official DSML tool-calls token"
    );
    assert!(
        rendered.contains("<｜DSML｜invoke name=\"get_weather\">"),
        "DeepSeek uses the official DSML invoke token"
    );
    assert!(
        rendered
            .contains("<｜DSML｜parameter name=\"city\" string=\"true\">Rome</｜DSML｜parameter>"),
        "DeepSeek uses official typed DSML parameters"
    );
    assert!(
        !rendered.contains("<｜DS｜"),
        "Legacy nonstandard DS tool markers are absent"
    );
    assert_eq!(
        sha256(rendered.as_bytes()),
        "cd89794650716ce13bac7ac2392db6a06231ce9927ceb1d712f7c399a0a2fde6",
        "DeepSeek tool loop matches the pinned official encoder golden"
    );
}

/// `TestParallelToolOrder`.
#[test]
fn parallel_tool_results_follow_call_order() {
    let options = ChatTemplateOptions::default();
    let mut messages = parallel_conversation();
    let reverse = render(&messages, &options);
    assert_eq!(
        sha256(reverse.as_bytes()),
        hf_golden("parallel_tools").member_str("rendered_sha256", ""),
        "Parallel tool rendering matches the pinned official encoder"
    );
    assert!(
        reverse.contains(
            "<tool_result>Result A</tool_result>\n\nCompare them.\n\n<tool_result>Result B</tool_result>"
        ),
        "IDs restore call order without moving interleaved user text"
    );
    messages.swap(2, 4);
    assert_eq!(
        render(&messages, &options),
        reverse,
        "Parallel result arrival order does not change the prompt"
    );
    messages[2].tool_call_id.clear();
    messages[4].tool_call_id.clear();
    messages.swap(2, 4);
    assert!(
        render(&messages, &options).contains(
            "<tool_result>Result B</tool_result>\n\nCompare them.\n\n<tool_result>Result A</tool_result>"
        ),
        "Missing IDs retain stable arrival order, as the official encoder does"
    );
}

/// The `main` case of the C++ test: a complete tool definition keeps its
/// optional fields.
#[test]
fn complete_function_definitions_survive() {
    let tools = [ChatTool {
        name: "emit".into(),
        description: String::new(),
        parameters_json: "{}".into(),
        definition_json: r#"{"name":"emit","strict":true,"parameters":{"type":"object","additionalProperties":false},"vendor":2}"#.into(),
    }];
    let rendered = render_chat(
        &[message("user", "Emit.")],
        &tools,
        &ChatTemplateOptions::default(),
    );
    assert!(
        rendered.contains(r#"{"name": "emit", "strict": true, "parameters": {"type": "object", "additionalProperties": false}, "vendor": 2}"#),
        "complete function definition survives DeepSeek rendering"
    );
}

/// `TestHuggingFaceRenderedGoldens`: the plain and history conversations
/// against their pinned Hugging Face digests.
#[test]
fn rendering_matches_the_hugging_face_goldens() {
    let pinned = [
        (
            "chat",
            "d3953e69808de7acaa73269d1fbd13a930a4ee743b65ec2883b91d206ceb4121",
        ),
        (
            "thinking_low",
            "302d80db67339dedc8489476bde608f250f13f2b7bb89d6c4467538e057e0dba",
        ),
        (
            "thinking_high",
            "4e0a1a55cd4c39b593c49f3ec8384e32cde119bff97416a611d7ab459624aaa9",
        ),
        (
            "thinking_max",
            "b61a0552e96f0458cada39d46bd11a15adc206f0cd3a1dfea096687fe3fb4968",
        ),
        (
            "history_drop",
            "df3c2370812a2f993fca67aee691ccf730d57f24f9207cb8e5b1be38ba916e88",
        ),
        (
            "history_preserve",
            "2fc631d9339a08cd4fffd643eae2b58426fb0d709e69daaaffb978d55b6906d3",
        ),
    ];
    let cases = hf_cases();
    for (name, digest) in pinned {
        let case = cases.iter().find(|case| case.name == name).unwrap();
        assert_eq!(
            sha256(render_chat(&case.messages, &case.tools, &case.options).as_bytes()),
            digest,
            "{name}: DeepSeek rendered bytes match the pinned Hugging Face golden"
        );
    }
}

/// A GGUF v3 of string metadata (`GgufTemplateBuilder`).
fn gguf_image(strings: &[(&str, &[u8])]) -> Vec<u8> {
    fn string(out: &mut Vec<u8>, value: &[u8]) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value);
    }
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&(strings.len() as u64).to_le_bytes());
    for (key, value) in strings {
        string(&mut out, key.as_bytes());
        out.extend_from_slice(&8u32.to_le_bytes());
        string(&mut out, value);
    }
    out.resize(out.len().next_multiple_of(32), 0);
    out
}

/// Validates the template of a DeepSeek V4 Flash GGUF holding `template`;
/// the reader maps files, so the image goes through a temporary file.
fn validate_template(label: &str, template: &[u8]) -> Result<(), String> {
    let path = std::env::temp_dir().join(format!(
        "gufo-ds4-chat-template-{label}-{}.gguf",
        std::process::id()
    ));
    std::fs::write(
        &path,
        gguf_image(&[
            ("general.architecture", b"deepseek4"),
            ("general.name", b"DeepSeek V4 Flash"),
            ("tokenizer.chat_template", template),
        ]),
    )
    .unwrap();
    let model = Model::open(&path);
    std::fs::remove_file(&path).unwrap();
    let model = model.unwrap_or_else(|error| panic!("DeepSeek {label} GGUF opens: {error}"));
    validate_gguf_template(&model)
}

/// `TestPinnedArtifactTemplateValidation`.
#[test]
fn pinned_artifact_template_validation() {
    assert!(
        !ARTIFACT_TEMPLATE.is_empty(),
        "Pinned DeepSeek artifact template is readable"
    );
    assert_eq!(
        sha256(ARTIFACT_TEMPLATE),
        ARTIFACT_TEMPLATE_SHA256,
        "Versioned DeepSeek Jinja has the pinned artifact hash"
    );
    assert_eq!(
        validate_template("pinned", ARTIFACT_TEMPLATE),
        Ok(()),
        "Pinned DeepSeek artifact template is accepted"
    );
    let lookalike = [ARTIFACT_TEMPLATE, b" "].concat();
    let error = validate_template("lookalike", &lookalike).unwrap_err();
    assert!(
        error.starts_with("DeepSeek GGUF chat template SHA-256 is not the pinned 0731 artifact"),
        "DeepSeek look-alike template is rejected by hash: {error}"
    );
}

/// `CheckTokenGoldens` of `ds4_quality_test`: the rendered bytes and token
/// IDs of the Hugging Face golden conversations with the model's tokenizer.
#[cfg(feature = "models")]
#[test]
fn chat_tokens_match_the_hugging_face_goldens() {
    use gufo_deepseek_v4_flash::tokenizer::Vocab;

    let model = Model::open(&common::model_path()).unwrap();
    let vocab = Vocab::load(&model).unwrap();
    for case in hf_cases() {
        let expected = hf_golden(case.name);
        let name = case.name;
        let rendered = render_chat(&case.messages, &case.tools, &case.options);
        assert_eq!(
            sha256(rendered.as_bytes()),
            expected.member_str("rendered_sha256", ""),
            "{name}: DeepSeek rendered bytes match Hugging Face"
        );
        let mut tokens = Vec::new();
        vocab.tokenize_rendered_chat(rendered.as_bytes(), &mut tokens);
        assert_eq!(
            tokens.len(),
            expected.member_size("token_count", usize::MAX),
            "{name}: DeepSeek token count matches Hugging Face"
        );
        assert_eq!(
            sha256(&common::token_bytes(&tokens)),
            expected.member_str("token_ids_le_u32_sha256", ""),
            "{name}: DeepSeek token IDs match Hugging Face"
        );
    }
}
