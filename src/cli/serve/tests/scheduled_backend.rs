//! `ScheduledBackend`, the facade half of the C++ `InferenceBackend`, over a
//! runner that echoes its prompt.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use gufo_core::json_constraint::{ConstraintVocabulary, JsonConstraint, Piece, ToolFormat};
use gufo_core::sampling::{ReasoningBudget, SamplingConfig, SamplingOverrides, TextModelPreset};
use gufo_core::text::{
    CancellationCheck, ChatMessage, ChatRequest, ChatRole, ChatTool, ErrorKind, InitialOutputState,
    NoPromptContext, OutputGrammar, ReasoningOptions, ReasoningTags, Result, Sampler,
    TextDecodeSelection, TextExecutionPlan, TextModelRunner, TextPrefillStep, TextRunnerDescriptor,
    TextRunnerResourceClaim, TextRunnerSnapshot, TextRunnerState, Token, ToolChoice,
};
use gufo_serve::backend::{self, FinishReason, TextGenerationBackend};
use gufo_serve::pool::TextRunnerPool;
use gufo_serve::scheduled_backend::ScheduledBackend;
use gufo_serve::scheduler::{TextPrefillPolicy, TextSchedulerPolicy};

/// Generates its prompt back, one byte per token, then stops. The prompt is
/// the messages' content with their server framing.
struct EchoRunner {
    tool_format: ToolFormat,
    /// Output starts in reasoning, and `<think>`, `</think>` and `\n</think>`
    /// are control tokens 300, 301 and [10, 301].
    reasoning: bool,
    /// The reasoning budget of the last sampler created.
    budget: Mutex<Option<Arc<ReasoningBudget>>>,
}

impl Default for EchoRunner {
    fn default() -> Self {
        Self {
            tool_format: ToolFormat::Json,
            reasoning: false,
            budget: Mutex::new(None),
        }
    }
}

#[derive(Default)]
struct EchoState {
    prompt: Vec<Token>,
    generated: usize,
}

impl TextRunnerState for EchoState {
    fn invalidate(&mut self) {
        *self = Self::default();
    }
}

struct NoSnapshot;

impl TextRunnerSnapshot for NoSnapshot {
    fn payload_bytes(&self) -> usize {
        0
    }
}

#[derive(Clone)]
struct NoSampler;

impl Sampler for NoSampler {
    fn accept(&mut self, _: Token) {}
}

impl TextModelRunner for EchoRunner {
    type State = EchoState;
    type Snapshot = NoSnapshot;
    type Sampler = NoSampler;
    type PromptContext = NoPromptContext;

    fn descriptor(&self) -> TextRunnerDescriptor {
        TextRunnerDescriptor {
            model_id: "echo".into(),
            state_abi: "echo-v1".into(),
            max_context: 1024,
            ..TextRunnerDescriptor::default()
        }
    }

    fn resource_claim(&self) -> TextRunnerResourceClaim {
        TextRunnerResourceClaim::default()
    }

    fn supported_plans(&self) -> Vec<TextExecutionPlan> {
        vec![TextExecutionPlan::SERIAL]
    }

    fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
        Ok(text.bytes().map(Token::from).collect())
    }

    fn tokenize_control(&self, text: &str) -> Result<Vec<Token>> {
        Ok(match text {
            "<think>" if self.reasoning => vec![300],
            "</think>" if self.reasoning => vec![301],
            "\n</think>" if self.reasoning => vec![10, 301],
            _ => self.tokenize(text)?,
        })
    }

    fn reasoning_budget_tags(&self) -> ReasoningTags {
        ReasoningTags {
            forced_end: "\n</think>",
            ..ReasoningTags::default()
        }
    }

    fn initial_output_state(&self, _: &ChatRequest) -> InitialOutputState {
        if self.reasoning {
            InitialOutputState::Reasoning
        } else {
            InitialOutputState::Content
        }
    }

    fn render_and_tokenize(&self, request: &ChatRequest) -> Result<Option<Vec<Token>>> {
        if request.messages.is_empty() {
            return Ok(None);
        }
        let text: String = request
            .messages
            .iter()
            .map(|m| format!("{}{}", m.content, m.framing_suffix))
            .collect();
        self.tokenize(&text).map(Some)
    }

    fn tool_format(&self) -> ToolFormat {
        self.tool_format
    }

    fn build_constraint_vocabulary(&self) -> Result<ConstraintVocabulary> {
        ConstraintVocabulary::new(256, |token| {
            Ok(Piece {
                text: vec![token as u8],
                stop: false,
            })
        })
    }

    fn decode(&self, tokens: &[Token]) -> Result<Vec<u8>> {
        Ok(tokens.iter().map(|&t| t as u8).collect())
    }

    fn create_state(&self) -> Result<EchoState> {
        Ok(EchoState::default())
    }

    fn create_sampler(&self, config: &SamplingConfig, _: &[Token]) -> Result<NoSampler> {
        *self.budget.lock().expect("budget") = config.reasoning_budget.clone();
        Ok(NoSampler)
    }

    fn prefill(
        &self,
        state: &mut EchoState,
        prompt: &[Token],
        offset: usize,
        _: usize,
    ) -> Result<TextPrefillStep> {
        state.prompt = prompt.to_vec();
        state.generated = 0;
        Ok(TextPrefillStep {
            consumed_tokens: prompt.len() - offset,
            decode_ready: true,
        })
    }

    fn select_next(&self, state: &mut EchoState, _: &mut NoSampler) -> Result<TextDecodeSelection> {
        let Some(&token) = state.prompt.get(state.generated) else {
            return Ok(TextDecodeSelection {
                stop: true,
                ..TextDecodeSelection::default()
            });
        };
        Ok(TextDecodeSelection {
            stop: false,
            token,
            piece: vec![token as u8],
        })
    }

    fn advance(&self, state: &mut EchoState, _: Token) -> Result<()> {
        state.generated += 1;
        Ok(())
    }

    fn checkpoint_position(&self, state: &EchoState) -> Result<usize> {
        Ok(state.prompt.len() + state.generated)
    }
}

fn backend() -> ScheduledBackend<EchoRunner> {
    backend_over(EchoRunner::default())
}

fn backend_over(runner: EchoRunner) -> ScheduledBackend<EchoRunner> {
    let pool = Arc::new(TextRunnerPool::new(Arc::new(runner), 1, None).expect("pool"));
    ScheduledBackend::new(
        pool,
        TextPrefillPolicy::default(),
        TextSchedulerPolicy::default(),
    )
    .expect("backend")
}

fn chat(text: &str) -> ChatRequest {
    ChatRequest {
        client_id: "client".into(),
        ..ChatRequest::new(vec![ChatMessage::new(ChatRole::User, text)])
    }
}

#[test]
fn settings_follow_the_cpp_setters() {
    let backend = backend();
    assert!(backend.ready());
    assert_eq!(backend.model_id(), "echo");
    backend.set_model_id("");
    assert_eq!(backend.model_id(), "echo");
    backend.set_model_id("served");
    assert_eq!(backend.model_id(), "served");

    let sampling = SamplingConfig::with_temperature(0.5);
    backend
        .set_sampling_defaults(64, &sampling, SamplingOverrides::ALL)
        .expect("valid");
    assert_eq!(backend.sampling_defaults().max_tokens, 64);
    assert_eq!(backend.sampling_defaults().sampling, sampling);
    assert_eq!(backend.sampling_defaults().resolve(Some(false)), sampling);
    // A model preset fills the fields the operator left unset.
    backend.set_sampling_preset(TextModelPreset::Qwen38);
    backend
        .set_sampling_defaults(
            64,
            &sampling,
            SamplingOverrides {
                temperature: true,
                ..SamplingOverrides::default()
            },
        )
        .expect("valid");
    let direct = backend.sampling_defaults().resolve(Some(false));
    assert_eq!(
        (direct.temperature, direct.top_p, direct.top_k),
        (0.5, 0.8, 20)
    );
    let thinking = backend.sampling_defaults().resolve(None);
    assert_eq!((thinking.temperature, thinking.top_p), (0.5, 0.95));
    backend.set_sampling_preset(TextModelPreset::Unspecified);
    let invalid = SamplingConfig {
        top_p: 2.0,
        ..SamplingConfig::default()
    };
    let error = backend
        .set_sampling_defaults(8, &invalid, SamplingOverrides::ALL)
        .expect_err("invalid sampling");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(backend.sampling_defaults().max_tokens, 64);

    let reasoning = ReasoningOptions {
        enabled: Some(false),
        ..ReasoningOptions::default()
    };
    backend.set_reasoning_defaults(reasoning);
    assert_eq!(backend.reasoning_defaults(), reasoning);
    assert_eq!(backend.count_tokens("four"), 4);
}

#[test]
fn completion_and_chat_run_through_the_scheduler() {
    let backend = backend();
    let mut pieces = Vec::new();
    let mut on_token = |piece: &[u8]| {
        pieces.extend_from_slice(piece);
        true
    };
    let result = backend
        .complete(
            "abc",
            16,
            &SamplingConfig::default(),
            None,
            Some(&mut on_token),
            "",
            &[],
            true,
        )
        .expect("complete");
    assert_eq!(result.text, b"abc");
    assert_eq!(pieces, b"abc");
    assert_eq!(result.prompt_tokens, 3);
    assert_eq!(result.completion_tokens, 3);
    assert_eq!(result.client_id, "anonymous");
    assert_eq!(result.finish_reason, FinishReason::Stop);

    let result = backend
        .chat(&chat("hello"), 2, &SamplingConfig::default(), None, None)
        .expect("chat");
    assert_eq!(result.text, b"he");
    assert_eq!(result.client_id, "client");
    assert_eq!(result.finish_reason, FinishReason::Length);
}

#[test]
fn prompts_without_tokens_return_empty_results() {
    let backend = backend();
    let result = backend
        .chat(
            &ChatRequest::default(),
            4,
            &SamplingConfig::default(),
            None,
            None,
        )
        .expect("chat");
    assert_eq!(result.prompt_tokens, 0);
    assert!(result.text.is_empty());
    let result = backend
        .complete("", 4, &SamplingConfig::default(), None, None, "", &[], true)
        .expect("complete");
    assert_eq!(result.prompt_tokens, 0);

    // Without a prepared prompt, start_chat defers to a blocking chat.
    assert!(
        backend
            .start_chat(
                &ChatRequest::default(),
                4,
                &SamplingConfig::default(),
                None,
                false
            )
            .expect("start")
            .is_none()
    );
}

#[test]
fn cancellation_before_submission_skips_generation() {
    let backend = backend();
    let cancelled = Arc::new(AtomicBool::new(true));
    let flag = cancelled.clone();
    let check: CancellationCheck = Arc::new(move || flag.load(Ordering::Acquire));
    let result = backend
        .complete(
            "abc",
            4,
            &SamplingConfig::default(),
            Some(&check),
            None,
            "c",
            &[],
            true,
        )
        .expect("complete");
    assert!(result.cancelled);
    assert_eq!(result.completion_tokens, 0);
    assert_eq!(result.prompt_tokens, 3);
    assert_eq!(result.client_id, "c");
}

#[test]
fn started_chats_are_scheduled_requests() {
    let backend: Arc<dyn TextGenerationBackend> = Arc::new(backend());
    let mut request = backend::start_chat(
        &backend,
        &chat("xyz"),
        8,
        &SamplingConfig::default(),
        None,
        true,
    )
    .expect("start");
    let mut pieces = Vec::new();
    let mut on_token = |piece: &[u8]| {
        pieces.push(piece.to_vec());
        true
    };
    let result = request.wait(Some(&mut on_token)).expect("wait");
    assert_eq!(result.text, b"xyz");
    assert_eq!(pieces, [b"x".to_vec(), b"y".to_vec(), b"z".to_vec()]);
    assert_eq!(result.client_id, "client");
}

/// Native tool constraints follow the model's template and add no prompt
/// text. Only runners without a native call syntax use the JSON envelope,
/// whose instruction is server framing beside the client's unchanged
/// content; a native runner never switches syntax for a schema (upstream
/// `ConstrainChatRequest`).
#[test]
fn tool_constraints_select_their_format_and_framing() {
    let schema = r#"{"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}"#;
    for expected in [ToolFormat::Qwen, ToolFormat::Json] {
        let backend: Arc<dyn TextGenerationBackend> = Arc::new(backend_over(EchoRunner {
            tool_format: expected,
            ..EchoRunner::default()
        }));
        for role in [ChatRole::User, ChatRole::System, ChatRole::Developer] {
            for client in ["Be concise.  ", "Client <tool_call>example</tool_call>"] {
                let mut request = chat("");
                request.messages.clear();
                if role != ChatRole::User {
                    request.messages.push(ChatMessage::new(role, client));
                }
                request.messages.push(ChatMessage::new(
                    ChatRole::User,
                    if role == ChatRole::User {
                        client
                    } else {
                        "Call record."
                    },
                ));
                request.tool_choice = ToolChoice::Required;
                request.tools = vec![ChatTool {
                    name: "record".into(),
                    parameters_json: schema.into(),
                    ..ChatTool::default()
                }];
                let mut generation = backend::start_chat(
                    &backend,
                    &request,
                    4096,
                    &SamplingConfig::default(),
                    None,
                    false,
                )
                .expect("start");
                assert_eq!(generation.tool_format(), Some(expected));
                let prompt =
                    String::from_utf8(generation.wait(None).expect("wait").text).expect("utf8");
                let client_text: String = request
                    .messages
                    .iter()
                    .map(|m| m.content.as_str())
                    .collect();
                if expected == ToolFormat::Json {
                    let instruction = prompt
                        .strip_prefix(if role == ChatRole::User { "" } else { client })
                        .expect("the client content stays first and unchanged");
                    assert!(
                        instruction.contains("<tool_call>{\"name\"")
                            && instruction.ends_with(if role == ChatRole::User {
                                client
                            } else {
                                "Call record."
                            }),
                        "the JSON instruction is framing after the system content: {prompt:?}"
                    );
                } else {
                    assert_eq!(prompt, client_text, "native constraints add no prompt text");
                }
            }
        }
    }
}

/// Plain and JSON answers keep the model's output dialect: parsing needs it
/// without tool constraints, e.g. for Qwen's separator after `</think>`
///.
#[test]
fn plain_and_json_answers_report_the_native_dialect() {
    for format in [ToolFormat::Qwen, ToolFormat::DeepSeek] {
        let backend: Arc<dyn TextGenerationBackend> = Arc::new(backend_over(EchoRunner {
            tool_format: format,
            ..EchoRunner::default()
        }));
        for json in [false, true] {
            let mut request = chat("answer");
            if json {
                request.response_format = Some(OutputGrammar(
                    JsonConstraint::compile(
                        &gufo_core::json::parse_str(
                            r#"{"type":"object","properties":{},"additionalProperties":false}"#,
                        )
                        .expect("schema"),
                        false,
                    )
                    .expect("grammar"),
                ));
            }
            let generation = backend::start_chat(
                &backend,
                &request,
                4096,
                &SamplingConfig::default(),
                None,
                false,
            )
            .expect("start");
            assert_eq!(generation.tool_format(), Some(format), "json={json}");
        }
    }
}

/// Schemas native tags cannot enforce exactly, beside an ordinary neighbor,
/// under every tool choice: the request stays native as in llama.cpp, so the
/// prompt is the client's own and needs no extra prefill.
#[test]
fn native_runners_keep_native_calls_for_every_schema() {
    let backend: Arc<dyn TextGenerationBackend> = Arc::new(backend_over(EchoRunner {
        tool_format: ToolFormat::Qwen,
        ..EchoRunner::default()
    }));
    let schemas = [
        r#"{"type":"object","properties":{"text":{"type":"string","const":"\n</parameter>\n"}},"required":["text"],"additionalProperties":false}"#,
        r#"{"type":"object","properties":{"text":{"type":"string"}},"patternProperties":{"^x_":{"type":"integer"}},"required":["text"]}"#,
        r#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"text":{"type":"string"}},"required":["text"]}"#,
        r#"{"type":"object","properties":{"date":{"type":"string","pattern":"^[0-9]{4}$"}}}"#,
        r#"{"type":"object","properties":{"v":{"oneOf":[{"type":"string"},{"type":"integer"}]}}}"#,
        r#"{"type":"object","properties":{"v":{"allOf":[{"type":"string"},{"minLength":1}]}}}"#,
        r#"{"type":"object","properties":{"v":{"not":{"type":"null"}}}}"#,
        r#"{"type":"object","properties":{"v":{"type":"string"}},"additionalProperties":true}"#,
        r#"{"type":"object","properties":{"o":{"type":"object","properties":{"x":{"type":"integer"}}}}}"#,
    ];
    for schema in schemas {
        for choice in [ToolChoice::Auto, ToolChoice::Required] {
            for strict in [false, true] {
                let mut request = chat("");
                request.messages = vec![
                    ChatMessage::new(ChatRole::System, "Be concise."),
                    ChatMessage::new(ChatRole::User, "Call record."),
                ];
                request.reasoning.enabled = Some(false);
                request.tool_choice = choice;
                request.tools = vec![
                    ChatTool {
                        name: "bash".into(),
                        parameters_json: r#"{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}"#.into(),
                        ..ChatTool::default()
                    },
                    ChatTool {
                        name: "record".into(),
                        parameters_json: schema.into(),
                        definition_json: format!(
                            r#"{{"type":"function","function":{{"name":"record","strict":{strict},"parameters":{schema}}}}}"#
                        ),
                        ..ChatTool::default()
                    },
                ];
                let generation = backend::start_chat(
                    &backend,
                    &request,
                    4096,
                    &SamplingConfig::default(),
                    None,
                    false,
                );
                let Ok(mut generation) = generation else {
                    // An impossible strict schema is rejected before
                    // generation.
                    assert!(strict, "only strict schemas may be rejected: {schema}");
                    continue;
                };
                assert_eq!(
                    generation.tool_format(),
                    Some(ToolFormat::Qwen),
                    "a native runner keeps native calls: {schema}"
                );
                let prompt =
                    String::from_utf8(generation.wait(None).expect("wait").text).expect("utf8");
                assert_eq!(
                    prompt, "Be concise.Call record.",
                    "a native runner adds no tool instruction: {schema}"
                );
            }
        }
    }
}

/// A request whose output starts in reasoning samples under its budget:
/// the runner's tags as control tokens, the prompt after the last
/// `<think>`, and the vocabulary's partial UTF-8 tokens. Other requests sample without one.
#[test]
fn reasoning_budgets_reach_the_sampler() {
    let runner = Arc::new(EchoRunner {
        reasoning: true,
        ..EchoRunner::default()
    });
    let pool = Arc::new(TextRunnerPool::new(runner.clone(), 1, None).expect("pool"));
    let backend = ScheduledBackend::new(
        pool,
        TextPrefillPolicy::default(),
        TextSchedulerPolicy::default(),
    )
    .expect("backend");
    let mut request = chat("a<think>b");
    request.reasoning.budget_tokens = Some(7);
    backend
        .chat(&request, 4, &SamplingConfig::default(), None, None)
        .expect("chat");
    let budget = runner
        .budget
        .lock()
        .expect("budget")
        .clone()
        .expect("a budget");
    assert_eq!(budget.tokens, 7);
    assert_eq!(budget.forced, [10, 301]);
    assert_eq!(budget.ends, [301]);
    // The echo prompt spells <think> bytewise: no start token, no prompt
    // reasoning.
    assert!(budget.prompt.is_empty());
    assert_eq!(budget.partial_utf8.as_ref().map(|p| p.len()), Some(256));
    assert!(
        budget
            .partial_utf8
            .as_ref()
            .is_some_and(|p| p[0xC3] && !p[b'a' as usize])
    );
    request.reasoning.budget_tokens = None;
    backend
        .chat(&request, 4, &SamplingConfig::default(), None, None)
        .expect("chat");
    assert!(runner.budget.lock().expect("budget").is_none());
}
