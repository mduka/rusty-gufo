//! Port of `tests/cli/http_server_test.cpp` (the request-logging case is in
//! `http_logging.rs`, which captures the process-wide log).

mod common;
mod http_fixture;

use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use gufo_core::json;
use gufo_core::sampling::SamplingConfig;
use gufo_core::text::{
    ChatRequest, ChatRole, DEVICE_LOST_MESSAGE, GenerationErrorCode, ReasoningEffort,
    ReasoningOptions, SCHEDULER_FAILED_MESSAGE, ToolChoice,
};
use gufo_serve::backend::{
    InitialOutputState, PromptProgress, SessionState, TextGenerationBackend, ToolFormat,
};
use gufo_serve::http::server::{HttpServer, HttpServices, LoadingResponder, is_listen_address};
use gufo_serve::http::{HttpRequest, HttpResponse, HttpServerOptions};
use gufo_serve::openai_chat::{
    ApiCompat, ResponsesReasoningTarget, create_openai_response, parse_tool_controls,
};

use http_fixture::{RunningServer, body_of, expect_status};

#[test]
fn authorization() {
    let secured = RunningServer::new(HttpServerOptions {
        api_key: "test-secret".into(),
        ..HttpServerOptions::default()
    });
    for path in [
        "/health",
        "/ready",
        "/v1/models",
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/messages",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/video/generations",
        "/echo",
    ] {
        expect_status(
            &secured.send(&format!("POST {path} HTTP/1.1\r\n\r\n"), false),
            401,
        );
    }
    for header in [
        "",
        "Authorization: Bearer wrong\r\n",
        "Authorization: Basic test-secret\r\n",
        "Authorization: Bearer test-secret\r\nAuthorization: Bearer test-secret\r\n",
    ] {
        let response = secured.send(&format!("GET /v1/models HTTP/1.1\r\n{header}\r\n"), false);
        expect_status(&response, 401);
        assert!(response.contains("WWW-Authenticate: Bearer"));
        assert!(!response.contains("test-secret"));
    }
    assert_eq!(secured.backend.calls.load(Ordering::SeqCst), 0);
    expect_status(
        &secured.send(
            "GET /v1/models HTTP/1.1\r\naUtHoRiZaTiOn: bEaReR  test-secret\r\n\r\n",
            false,
        ),
        200,
    );
    expect_status(
        &secured.send("OPTIONS /v1/chat/completions HTTP/1.1\r\n\r\n", false),
        204,
    );
    let body = r#"{"prompt":"hi","max_tokens":1}"#;
    expect_status(
        &secured.send(
            &format!(
                "POST /v1/completions HTTP/1.1\r\nAuthorization: Bearer test-secret\r\n\
                 Content-Length: {}\r\n\r\n{body}",
                body.len()
            ),
            false,
        ),
        200,
    );
    assert_eq!(secured.backend.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn framing_and_metrics() {
    let server = RunningServer::new(HttpServerOptions {
        max_request_body_bytes: 8192,
        ..HttpServerOptions::default()
    });
    expect_status(&server.send("GET /health HTTP/1.1\r\n\r\n", false), 200);
    for header in [
        "Content-Length: nope",
        "Content-Length: -1",
        "Content-Length: 4junk",
        "Content-Length: 18446744073709551616",
        "Content-Length: 4\r\nContent-Length: 3",
        "Transfer-Encoding: chunked",
        "broken-header",
        "Transfer-Encoding : chunked",
        "Content-Length\t: 5",
        "X-Test: one\r\n folded",
        "Bad Name: x",
    ] {
        expect_status(
            &server.send(&format!("POST /echo HTTP/1.1\r\n{header}\r\n\r\n"), false),
            400,
        );
    }
    expect_status(&server.send("POST /echo\r\n\r\n", false), 400);
    expect_status(
        &server.send("POST /echo HTTP/1.1 extra\r\n\r\n", false),
        400,
    );
    expect_status(
        &server.send("POST /echo HTTP/1.1\r\nContent-Length: 4\r\n\r\nx", true),
        400,
    );
    expect_status(
        &server.send("POST /echo HTTP/1.1\r\nContent-Length: 8193\r\n\r\n", false),
        413,
    );
    let payload = "x".repeat(5000);
    let echo = server.send(
        &format!(
            "POST /echo HTTP/1.1\r\nContent-Length: 5000\r\nContent-Length: 5000\r\n\r\n{payload}"
        ),
        false,
    );
    expect_status(&echo, 200);
    assert_eq!(body_of(&echo), payload);

    let body = r#"{"prompt":"hi","n_predict":1}"#;
    let response = server.send(
        &format!(
            "POST /completion HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ),
        false,
    );
    expect_status(&response, 200);
    let parsed = json::parse_str(body_of(&response)).expect("json");
    let timings = parsed.find("timings").expect("timings");
    assert_eq!(timings.member_double("prompt_n", 0.0), 2.0);
    assert_eq!(timings.member_double("cache_n", 0.0), 8.0);
    assert_eq!(timings.member_double("prompt_per_second", 0.0), 500.0);
    assert_eq!(timings.member_double("prompt_per_token_ms", 0.0), 2.0);
}

/// curl sends `Expect: 100-continue` before a body over 1 MiB and waits for
/// the interim response; without it every large upload stalls for a second
///.
#[test]
fn expect_continue() {
    let server = RunningServer::new(HttpServerOptions {
        max_request_body_bytes: 8192,
        ..HttpServerOptions::default()
    });
    let interim = b"HTTP/1.1 100 Continue\r\n\r\n";
    for expect in ["Expect: 100-continue", "expect: 100-Continue"] {
        let mut stream = server.connect();
        stream
            .write_all(
                format!("POST /echo HTTP/1.1\r\n{expect}\r\nContent-Length: 5\r\n\r\n").as_bytes(),
            )
            .expect("send head");
        // The interim response arrives before any body byte is sent.
        let mut response = vec![0u8; interim.len()];
        stream.read_exact(&mut response).expect("interim response");
        assert_eq!(response, interim, "{expect}");
        stream.write_all(b"hello").expect("send body");
        let mut rest = Vec::new();
        let _ = stream.read_to_end(&mut rest);
        let rest = String::from_utf8_lossy(&rest);
        expect_status(&rest, 200);
        assert_eq!(body_of(&rest), "hello");
    }
    // An oversized body gets its final status instead of the interim one.
    expect_status(
        &server.send(
            "POST /echo HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 8193\r\n\r\n",
            false,
        ),
        413,
    );
}

#[test]
fn compatibility_requests() {
    let server = RunningServer::new(HttpServerOptions::default());
    let response_body = |response: &str| {
        expect_status(response, 200);
        json::parse_str(body_of(response)).expect("json")
    };
    let endpoints = [
        ("/v1/completions", r#"{"prompt":"hi"}"#, "max_tokens"),
        ("/v1/responses", r#"{"input":"hi"}"#, "max_output_tokens"),
        (
            "/v1/messages",
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
            "max_tokens",
        ),
        ("/completion", r#"{"prompt":"hi"}"#, "n_predict"),
    ];
    for (path, input, limit) in endpoints {
        // Without a limit the server default applies: until EOS or the
        // context is full.
        response_body(&server.post(path, input));
        assert_eq!(server.backend.last_call().max_tokens, 0, "{path}");
        let mut body = json::parse_str(input).expect("json");
        body.set("model", "test");
        body.set(limit, 1);
        body.set("temperature", 0.6);
        body.set("top_k", 40);
        body.set("top_p", 0.9);
        body.set("seed", 123);
        body.set("repeat_penalty", 1.1);
        let dumped = body.dump().expect("dump");
        let output = response_body(&server.post(path, &dumped));
        let last = server.backend.last_call();
        for failure in [1, 2] {
            server.backend.failure.store(failure, Ordering::SeqCst);
            let rejected = server.post(path, &dumped);
            expect_status(&rejected, 400);
            assert!(rejected.contains(if failure == 1 {
                "context_length_exceeded"
            } else {
                "invalid_prompt"
            }));
        }
        server.backend.failure.store(0, Ordering::SeqCst);
        assert_eq!(last.client_id, "127.0.0.1");
        assert!(
            last.max_tokens == 1
                && last.sampling.temperature == 0.6f32
                && last.sampling.top_k == 40
                && last.sampling.top_p == 0.9f32
                && last.sampling.seed == 123
                && last.sampling.repeat_penalty == 1.1f32
        );
        match path {
            "/v1/responses" => {
                assert_eq!(output.member_str("status", ""), "incomplete");
                assert!(
                    output.member_size("created_at", 0) > 0
                        && output.find("completed_at").is_none()
                        && output.find("error") == Some(&json::Value::Null),
                    "an incomplete response has no completion time"
                );
                let items = output.find("output").expect("output").items();
                assert!(
                    items
                        .last()
                        .is_some_and(|item| item.member_str("status", "") == "incomplete"),
                    "the limit cuts the last item short"
                );
                assert_eq!(
                    output
                        .find("incomplete_details")
                        .expect("details")
                        .member_str("reason", ""),
                    "max_output_tokens"
                );
                let usage = output.find("usage").expect("usage");
                assert_eq!(
                    usage
                        .find("input_tokens_details")
                        .expect("details")
                        .member_size("cache_write_tokens", 0),
                    2,
                    "new prompt-cache rows"
                );
                assert!(
                    usage
                        .find("output_tokens_details")
                        .expect("details")
                        .contains("reasoning_tokens")
                );
                assert!(
                    output.find("parallel_tool_calls") == Some(&json::Value::Bool(false))
                        && output.member_str("tool_choice", "") == "none"
                        && output.find("tools").is_some_and(|tools| tools.is_empty()),
                    "Responses declares the tool policy the SDK requires"
                );
            }
            "/v1/messages" => assert_eq!(output.member_str("stop_reason", ""), "max_tokens"),
            "/completion" => {
                assert!(output.find("stopped_length").expect("field").as_bool(false));
                assert!(!output.find("stopped_eos").expect("field").as_bool(true));
            }
            _ => assert_eq!(
                output.find("choices").expect("choices").items()[0].member_str("finish_reason", ""),
                "length"
            ),
        }
        let calls = server.backend.calls.load(Ordering::SeqCst);
        for value in ["0", "-1", "1.5", "1e100", "\"1\"", "null"] {
            let mut invalid = body.clone();
            invalid.set(limit, json::parse_str(value).expect("json"));
            expect_status(&server.post(path, &invalid.dump().expect("dump")), 400);
        }
        for field in [
            "stream",
            "echo",
            "store",
            "background",
            "tools",
            "stop",
            "reasoning",
            "output_config",
            "logit_bias",
            "ignore_eos",
        ] {
            if (path == "/v1/completions" && (field == "stream" || field == "ignore_eos"))
                || (matches!(path, "/v1/responses" | "/v1/messages") && field == "stream")
            {
                continue;
            }
            let mut invalid = body.clone();
            invalid.set(field, true);
            expect_status(&server.post(path, &invalid.dump().expect("dump")), 400);
        }
        let mut invalid = body.clone();
        invalid.set("n", 1.4);
        expect_status(&server.post(path, &invalid.dump().expect("dump")), 400);
        let mut invalid = body.clone();
        invalid.set("model", "wrong");
        expect_status(&server.post(path, &invalid.dump().expect("dump")), 404);
        expect_status(&server.post(path, "[]"), 400);
        expect_status(&server.post(path, "{"), 400);
        assert_eq!(server.backend.calls.load(Ordering::SeqCst), calls);
    }
    let calls = server.backend.calls.load(Ordering::SeqCst);
    expect_status(
        &server.post("/v1/completions", r#"{"prompt":["one","two"]}"#),
        400,
    );
    expect_status(
        &server.post(
            "/v1/responses",
            r#"{"input":[{"role":"assistant","content":[
                {"type":"input_text","text":"describe"},
                {"type":"input_image","image_url":"data:image/png;base64,AA=="}]}]}"#,
        ),
        400,
    );
    expect_status(
        &server.post(
            "/v1/messages",
            r#"{"messages":[{"role":"tool","content":"hi"}]}"#,
        ),
        400,
    );
    expect_status(&server.post("/v1/messages", r#"{"messages":[]}"#), 400);
    expect_status(
        &server.post("/infill", r#"{"input_prefix":"one","input_suffix":"two"}"#),
        501,
    );
    expect_status(
        &server.post(
            "/v1/messages/count_tokens",
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        ),
        501,
    );
    assert_eq!(server.backend.calls.load(Ordering::SeqCst), calls);

    // Images and a JSON Schema answer on the Responses route.
    server.backend.image_support.store(true, Ordering::SeqCst);
    response_body(&server.post(
        "/v1/responses",
        r#"{"input":[{"role":"user","content":[
            {"type":"input_text","text":"describe"},
            {"type":"input_image","image_url":"data:image/png;base64,AA=="}]}],
            "reasoning":{"effort":"none"},
            "text":{"format":{"type":"json_schema","name":"answer","strict":true,
              "schema":{"type":"object","properties":{"score":{"type":"integer","minimum":1,"maximum":5}},
              "required":["score"],"additionalProperties":false}}}}"#,
    ));
    let structured = server.backend.last_call().chat;
    let last = structured.messages.last().expect("message");
    assert!(
        structured.response_format.is_some()
            && structured.reasoning.enabled == Some(false)
            && last.images.len() == 1
            && last.images[0].offset == 8
    );
    // Codex's view_image sends detail "high"; it is served as automatic
    // detail, and an unknown value is still rejected.
    let calls = server.backend.calls.load(Ordering::SeqCst);
    expect_status(
        &server.post(
            "/v1/responses",
            r#"{"input":[{"role":"user","content":[
                {"type":"input_image","image_url":"data:image/png;base64,AA==","detail":"ultra"}]}]}"#,
        ),
        400,
    );
    assert_eq!(server.backend.calls.load(Ordering::SeqCst), calls);
    response_body(&server.post(
        "/v1/responses",
        r#"{"input":[{"role":"user","content":[
            {"type":"input_text","text":"describe"},
            {"type":"input_image","image_url":"data:image/png;base64,AA==","detail":"high"}]}]}"#,
    ));
    let chat = server.backend.last_call().chat;
    assert_eq!(chat.messages.last().expect("message").images.len(), 1);

    let response = response_body(&server.post(
        "/v1/responses",
        r#"{"instructions":"Be concise.","input":[{"role":"user","content":[
            {"type":"input_text","text":"hi"}]}],"max_output_tokens":2,"store":false}"#,
    ));
    assert_eq!(response.member_str("status", ""), "completed");
    let messages = server.backend.last_call().chat.messages;
    assert_eq!(messages.len(), 2);
    assert!(
        messages[0].role == ChatRole::System
            && messages[0].content == "Be concise."
            && messages[1].content == "hi"
    );
    let anthropic = response_body(&server.post(
        "/v1/messages",
        r#"{"system":[{"type":"text","text":"Be concise."}],
            "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
            "max_tokens":2}"#,
    ));
    assert_eq!(anthropic.member_str("stop_reason", ""), "end_turn");
    assert_eq!(
        server.backend.last_call().chat.messages[0].content,
        "Be concise."
    );
}

/// The reasoning item `target` gives `text`: upstream's summary alone, or one
/// `reasoning_text` part in `content` beside an empty summary.
fn assert_reasoning_item(item: &json::Value, target: ResponsesReasoningTarget, text: &str) {
    assert!(item.member_str("id", "").starts_with("rs_"), "{target:?}");
    assert_eq!(item.member_str("type", ""), "reasoning");
    let summary = item.find("summary").expect("summary").items();
    let (parts, kind) = if target == ResponsesReasoningTarget::Summary {
        assert!(
            item.find("content").is_none() && item.find("encrypted_content").is_none(),
            "upstream's item has no other fields: {item:?}"
        );
        (summary, "summary_text")
    } else {
        assert!(summary.is_empty(), "{target:?}");
        assert_eq!(item.member_str("encrypted_content", "x"), "");
        (
            item.find("content").expect("content").items(),
            "reasoning_text",
        )
    };
    assert_eq!(parts.len(), 1, "{target:?}");
    assert_eq!(parts[0].member_str("type", ""), kind);
    assert_eq!(parts[0].member_str("text", ""), text);
}

/// Reasoning in the Anthropic and Responses adapters: a thinking block or a
/// reasoning item (upstream's summary by default) before the answer, the
/// controls that switch thinking, and earlier reasoning in the history.
#[test]
fn compatibility_reasoning() {
    let server = RunningServer::new(HttpServerOptions::default());
    let response_body = |response: &str| {
        expect_status(response, 200);
        json::parse_str(body_of(response)).expect("json")
    };
    let anthropic = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
    let responses = r#"{"input":"hi"}"#;
    let kinds = |items: &[json::Value]| -> Vec<String> {
        items
            .iter()
            .map(|item| item.member_str("type", ""))
            .collect()
    };

    // A template that opens the reasoning block, and one the model opens.
    for (initial, output) in [
        (
            InitialOutputState::Reasoning,
            &b"plan\n</think>\n\nanswer"[..],
        ),
        (InitialOutputState::Auto, &b"<think>plan</think>answer"[..]),
    ] {
        server.backend.set_initial_output_state(initial);
        server.backend.set_output(output.to_vec());
        let message = response_body(&server.post("/v1/messages", anthropic));
        let content = message.find("content").expect("content").items();
        assert_eq!(kinds(content), ["thinking", "text"]);
        assert_eq!(content[0].member_str("thinking", ""), "plan");
        assert_eq!(content[0].member_str("signature", "x"), "");
        assert_eq!(content[1].member_str("text", ""), "answer");
        let response = response_body(&server.post("/v1/responses", responses));
        let output = response.find("output").expect("output").items();
        assert_eq!(kinds(output), ["reasoning", "message"]);
        assert_reasoning_item(&output[0], ResponsesReasoningTarget::Summary, "plan");
        let text = &output[1].find("content").expect("content").items()[0];
        assert_eq!(text.member_str("text", ""), "answer");
        // Fields the OpenAI SDK's strict validation requires.
        assert!(
            output
                .iter()
                .all(|item| item.member_str("status", "") == "completed")
        );
        assert!(text.find("annotations").is_some_and(|a| a.is_empty()));
        let created = response.member_size("created_at", 0);
        assert!(created > 0 && response.member_size("completed_at", 0) >= created);
    }
    // A limit inside the reasoning leaves no answer block or message.
    server
        .backend
        .set_initial_output_state(InitialOutputState::Reasoning);
    server.backend.set_output(b"still thinking".to_vec());
    let message = response_body(&server.post("/v1/messages", anthropic));
    assert_eq!(
        kinds(message.find("content").expect("content").items()),
        ["thinking"]
    );
    let response = response_body(&server.post("/v1/responses", responses));
    assert_eq!(
        kinds(response.find("output").expect("output").items()),
        ["reasoning"]
    );
    server
        .backend
        .set_initial_output_state(InitialOutputState::Auto);
    server.backend.set_output(b"ok".to_vec());

    let reasoning_of = |path: &str, body: &str| {
        response_body(&server.post(path, body));
        server.backend.last_call().chat.reasoning
    };
    let messages = r#""messages":[{"role":"user","content":"hi"}]"#;
    let cases: [(&str, String, Option<bool>, Option<ReasoningEffort>); 11] = [
        ("/v1/messages", format!("{{{messages}}}"), None, None),
        (
            "/v1/messages",
            format!(r#"{{{messages},"thinking":{{"type":"disabled"}}}}"#),
            Some(false),
            None,
        ),
        (
            "/v1/messages",
            format!(r#"{{{messages},"thinking":{{"type":"enabled","budget_tokens":2048}}}}"#),
            Some(true),
            None,
        ),
        (
            "/v1/messages",
            format!(r#"{{{messages},"thinking":{{"type":"adaptive"}}}}"#),
            None,
            None,
        ),
        // The fields Claude Code sends: adaptive keeps the server's default
        // and the effort applies while thinking is on.
        (
            "/v1/messages",
            format!(
                r#"{{{messages},"thinking":{{"type":"adaptive","display":"omitted"}},
                    "output_config":{{"effort":"xhigh"}}}}"#
            ),
            None,
            Some(ReasoningEffort::XHigh),
        ),
        // Effort never enables thinking.
        (
            "/v1/messages",
            format!(
                r#"{{{messages},"thinking":{{"type":"disabled"}},
                    "output_config":{{"effort":"low"}}}}"#
            ),
            Some(false),
            Some(ReasoningEffort::Low),
        ),
        (
            "/v1/messages",
            format!(r#"{{{messages},"chat_template_kwargs":{{"enable_thinking":false}}}}"#),
            Some(false),
            None,
        ),
        ("/v1/responses", responses.to_owned(), None, None),
        (
            "/v1/responses",
            r#"{"input":"hi","reasoning":{"effort":"low","summary":"auto"}}"#.to_owned(),
            Some(true),
            Some(ReasoningEffort::Low),
        ),
        (
            "/v1/responses",
            r#"{"input":"hi","reasoning":{"effort":"none"}}"#.to_owned(),
            Some(false),
            None,
        ),
        (
            "/v1/responses",
            r#"{"input":"hi","reasoning":{"effort":"xhigh"},
                "chat_template_kwargs":{"preserve_thinking":false}}"#
                .to_owned(),
            Some(true),
            Some(ReasoningEffort::XHigh),
        ),
    ];
    for (path, body, enabled, effort) in cases {
        let reasoning = reasoning_of(path, &body);
        assert_eq!(
            (reasoning.enabled, reasoning.effort),
            (enabled, effort),
            "{body}"
        );
    }
    assert_eq!(
        reasoning_of(
            "/v1/responses",
            r#"{"input":"hi","chat_template_kwargs":{"preserve_thinking":false}}"#
        )
        .preserve_thinking,
        Some(false)
    );

    let calls = server.backend.calls.load(Ordering::SeqCst);
    for (path, body) in [
        ("/v1/messages", format!(r#"{{{messages},"thinking":true}}"#)),
        (
            "/v1/messages",
            format!(r#"{{{messages},"thinking":{{"type":"on"}}}}"#),
        ),
        (
            "/v1/messages",
            format!(r#"{{{messages},"thinking":{{"type":"enabled","budget_tokens":0}}}}"#),
        ),
        (
            "/v1/messages",
            format!(r#"{{{messages},"thinking":{{"type":"enabled","budget_tokens":"x"}}}}"#),
        ),
        (
            "/v1/messages",
            format!(
                r#"{{{messages},"thinking":{{"type":"enabled"}},
                    "chat_template_kwargs":{{"enable_thinking":false}}}}"#
            ),
        ),
        (
            "/v1/messages",
            format!(r#"{{{messages},"reasoning":{{"effort":"low"}}}}"#),
        ),
        (
            "/v1/messages",
            format!(r#"{{{messages},"reasoning_effort":"low"}}"#),
        ),
        (
            "/v1/messages",
            format!(r#"{{{messages},"thinking":{{"type":"adaptive","display":"full"}}}}"#),
        ),
        (
            "/v1/messages",
            format!(r#"{{{messages},"output_config":{{"effort":"adaptive"}}}}"#),
        ),
        (
            "/v1/messages",
            format!(r#"{{{messages},"output_config":{{"effort":"minimal"}}}}"#),
        ),
        (
            "/v1/messages",
            format!(r#"{{{messages},"output_config":"high"}}"#),
        ),
        (
            "/v1/messages",
            format!(r#"{{{messages},"output_config":{{"format":{{"type":"json_schema"}}}}}}"#),
        ),
        (
            "/v1/messages",
            r#"{"messages":[{"role":"user","content":[
                {"type":"thinking","thinking":"no","signature":""}]}]}"#
                .to_owned(),
        ),
        (
            "/v1/messages",
            r#"{"messages":[{"role":"assistant","content":[{"type":"thinking"}]}]}"#.to_owned(),
        ),
        (
            "/v1/responses",
            r#"{"input":"hi","reasoning":"low"}"#.to_owned(),
        ),
        (
            "/v1/responses",
            r#"{"input":"hi","reasoning":{"effort":"bogus"}}"#.to_owned(),
        ),
        (
            "/v1/responses",
            r#"{"input":"hi","reasoning":{"effort":1}}"#.to_owned(),
        ),
        (
            "/v1/responses",
            r#"{"input":"hi","reasoning":{"effort":"low","verbosity":"high"}}"#.to_owned(),
        ),
        (
            "/v1/responses",
            r#"{"input":"hi","thinking":{"type":"enabled"}}"#.to_owned(),
        ),
        (
            "/v1/responses",
            r#"{"input":"hi","chat_template_kwargs":{"enable_thinking":"no"}}"#.to_owned(),
        ),
        (
            "/v1/responses",
            r#"{"input":[{"role":"user","content":"hi"},
                {"type":"reasoning","summary":[],"content":"no"}]}"#
                .to_owned(),
        ),
    ] {
        expect_status(&server.post(path, &body), 400);
    }
    assert_eq!(server.backend.calls.load(Ordering::SeqCst), calls);

    // Earlier reasoning returns as the thought of its assistant message.
    response_body(&server.post(
        "/v1/messages",
        r#"{"messages":[{"role":"user","content":"hi"},
            {"role":"assistant","content":[
                {"type":"thinking","thinking":"plan","signature":""},
                {"type":"redacted_thinking","data":"x"},
                {"type":"text","text":"answer"}]},
            {"role":"user","content":"more"}]}"#,
    ));
    let history = server.backend.last_call().chat.messages;
    assert_eq!(history.len(), 3);
    assert_eq!(
        (history[1].thought.as_str(), history[1].content.as_str()),
        ("plan", "answer")
    );
    // Gufo restores the summary it emits, and refuses encrypted reasoning
    // and reasoning content its target does not write.
    response_body(&server.post(
        "/v1/responses",
        r#"{"input":[{"role":"user","content":"hi"},
            {"type":"reasoning","summary":[{"type":"summary_text","text":"brief"}],
             "encrypted_content":null,"content":null},
            {"role":"assistant","content":"reply"},
            {"role":"user","content":"more"}]}"#,
    ));
    let history = server.backend.last_call().chat.messages;
    assert_eq!(
        (history[1].thought.as_str(), history[1].content.as_str()),
        ("brief", "reply")
    );
    let replay = r#"{"input":[{"role":"user","content":"hi"},
        {"type":"reasoning","id":"rs_1","summary":[],
         "content":[{"type":"reasoning_text","text":"plan"}]},
        {"type":"message","role":"assistant",
         "content":[{"type":"output_text","text":"answer"}]},
        {"role":"user","content":"more"},
        {"type":"reasoning","summary":[{"type":"summary_text","text":"brief"}]},
        {"role":"assistant","content":"reply"},
        {"type":"reasoning","summary":[],"encrypted_content":"opaque"},
        {"role":"user","content":"last"}]}"#;
    let calls = server.backend.calls.load(Ordering::SeqCst);
    for body in [
        replay,
        r#"{"input":[{"role":"user","content":"hi"},
            {"type":"reasoning","summary":[],"encrypted_content":"opaque"},
            {"role":"user","content":"last"}]}"#,
        r#"{"input":[{"role":"user","content":"hi"},
            {"type":"reasoning","content":[{"type":"reasoning_text","text":"plan"}]},
            {"role":"user","content":"last"}]}"#,
    ] {
        expect_status(&server.post("/v1/responses", body), 400);
    }
    assert_eq!(server.backend.calls.load(Ordering::SeqCst), calls);
    // A reasoning-text target restores its own content parts.
    let text = RunningServer::new(HttpServerOptions {
        responses_reasoning_target: ResponsesReasoningTarget::ReasoningText,
        ..HttpServerOptions::default()
    });
    expect_status(
        &text.post(
            "/v1/responses",
            r#"{"input":[{"role":"user","content":"hi"},
                {"type":"reasoning","summary":[],
                 "content":[{"type":"reasoning_text","text":"plan"}]},
                {"role":"assistant","content":"answer"},
                {"role":"user","content":"more"}]}"#,
        ),
        200,
    );
    assert_eq!(text.backend.last_call().chat.messages[1].thought, "plan");

    // llama.cpp reads reasoning content and summaries, ignores encrypted
    // reasoning, checks summary values and passes unknown members.
    let llama = RunningServer::new(HttpServerOptions {
        api_compat: ApiCompat::LlamaCpp,
        ..HttpServerOptions::default()
    });
    expect_status(&llama.post("/v1/responses", replay), 200);
    let history = llama.backend.last_call().chat.messages;
    let turns: Vec<(ChatRole, &str, &str)> = history
        .iter()
        .map(|m| (m.role, m.thought.as_str(), m.content.as_str()))
        .collect();
    assert_eq!(
        turns,
        [
            (ChatRole::User, "", "hi"),
            (ChatRole::Assistant, "plan", "answer"),
            (ChatRole::User, "", "more"),
            (ChatRole::Assistant, "brief", "reply"),
            (ChatRole::User, "", "last"),
        ]
    );
    expect_status(
        &llama.post(
            "/v1/responses",
            r#"{"input":"hi","reasoning":{"summary":"long"}}"#,
        ),
        400,
    );
    expect_status(
        &llama.post(
            "/v1/responses",
            r#"{"input":"hi","reasoning":{"effort":"low","verbosity":"high"}}"#,
        ),
        200,
    );
    // Gufo accepts any summary request: local reasoning is always returned.
    expect_status(
        &server.post(
            "/v1/responses",
            r#"{"input":"hi","reasoning":{"summary":"long"}}"#,
        ),
        200,
    );
}

/// Codex re-sends developer items mid-conversation: they stay in place, and
/// one inside a replayed reasoning/function_call group waits for the group
/// to end instead of splitting the assistant turn.
#[test]
fn responses_system_items_keep_their_turns() {
    let server = RunningServer::new(HttpServerOptions::default());
    let response_body = |response: &str| expect_status(response, 200);
    response_body(&server.post(
        "/v1/responses",
        r#"{"instructions":"Base rules.","input":[
            {"role":"user","content":"first"},
            {"type":"message","role":"assistant","content":[
              {"type":"output_text","text":"noted","annotations":[]}]},
            {"type":"message","role":"developer","content":[
              {"type":"input_text","text":"Compacted state"}]},
            {"role":"user","content":"second"}]}"#,
    ));
    let turns: Vec<(ChatRole, String)> = server
        .backend
        .last_call()
        .chat
        .messages
        .into_iter()
        .map(|m| (m.role, m.content))
        .collect();
    assert_eq!(
        turns,
        [
            (ChatRole::System, "Base rules.".to_owned()),
            (ChatRole::User, "first".to_owned()),
            (ChatRole::Assistant, "noted".to_owned()),
            (ChatRole::Developer, "Compacted state".to_owned()),
            (ChatRole::User, "second".to_owned()),
        ]
    );

    let items = [
        r#"{"role":"user","content":"Check state."}"#,
        r#"{"type":"reasoning","summary":[{"type":"summary_text","text":"I will inspect."}]}"#,
        r#"{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"key\":\"state\"}"}"#,
        r#"{"type":"function_call_output","call_id":"call_1","output":"OK"}"#,
    ];
    let developer = r#"{"role":"developer","content":"Follow policy."}"#;
    for position in [2, 3] {
        let mut input: Vec<&str> = items.to_vec();
        input.insert(position, developer);
        response_body(&server.post(
            "/v1/responses",
            &format!(r#"{{"input":[{}]}}"#, input.join(",")),
        ));
        let grouped = server.backend.last_call().chat.messages;
        assert_eq!(grouped.len(), 4, "{position}: {grouped:?}");
        assert_eq!(grouped[0].role, ChatRole::User);
        assert_eq!(
            (
                grouped[1].role,
                grouped[1].content.as_str(),
                grouped[1].thought.as_str()
            ),
            (ChatRole::Assistant, "", "I will inspect.")
        );
        assert_eq!(grouped[1].tool_calls.len(), 1);
        assert_eq!(grouped[1].tool_calls[0].id, "call_1");
        assert_eq!(
            (grouped[2].role, grouped[2].content.as_str()),
            (ChatRole::Developer, "Follow policy.")
        );
        assert_eq!(
            (
                grouped[3].role,
                grouped[3].tool_call_id.as_str(),
                grouped[3].content.as_str()
            ),
            (ChatRole::Tool, "call_1", "OK")
        );
    }

    // Chat Completions keeps them in place; the model template decides.
    response_body(&server.post(
        "/v1/chat/completions",
        r#"{"model":"test","messages":[{"role":"user","content":"first"},
            {"role":"assistant","content":"noted"},
            {"role":"developer","content":"compacted state"},
            {"role":"user","content":"second"}]}"#,
    ));
    let roles: Vec<ChatRole> = server
        .backend
        .last_call()
        .chat
        .messages
        .iter()
        .map(|m| m.role)
        .collect();
    assert_eq!(
        roles,
        [
            ChatRole::User,
            ChatRole::Assistant,
            ChatRole::Developer,
            ChatRole::User
        ]
    );
}

/// Standard Responses fields with no native effect are accepted: hosted tool
/// types are skipped, namespaces flatten into their functions (whose calls
/// echo the namespace), and `include`, `reasoning.summary` and
/// `text.verbosity` are ignored; malformed controls still fail (`TestResponsesClientCompatTolerances`).
#[test]
fn responses_client_compat_tolerances() {
    let server = RunningServer::new(HttpServerOptions::default());
    let post = |body: &str| server.post("/v1/responses", body);
    let tool_names = || -> Vec<String> {
        server
            .backend
            .last_call()
            .chat
            .tools
            .into_iter()
            .map(|t| t.name)
            .collect()
    };
    expect_status(
        &post(
            r#"{"input":"hi","tools":[
            {"type":"function","name":"exec","parameters":{"type":"object",
              "properties":{"cmd":{"type":"string"}},"required":["cmd"]}},
            {"type":"web_search","external_web_access":false},
            {"type":"namespace","name":"agents","tools":[{"type":"function","name":"spawn"}]},
            {"type":"code_interpreter"}]}"#,
        ),
        200,
    );
    assert_eq!(tool_names(), ["exec", "spawn"]);
    // A namespace is a client-side grouping: its functions keep their schema
    // and strictness.
    expect_status(
        &post(
            r#"{"input":"hi","tools":[{"type":"namespace","name":"crm",
            "description":"Local customer tools","tools":[{"type":"function","name":"lookup",
            "parameters":{"type":"object","properties":{},"required":[],
            "additionalProperties":false},"strict":true}]}]}"#,
        ),
        200,
    );
    let tools = server.backend.last_call().chat.tools;
    assert_eq!(tools.len(), 1);
    assert!(tools[0].definition_json.contains(r#""strict":true"#));
    for rejected in [
        // Ambiguous names across a namespace.
        r#"{"input":"hi","tools":[{"type":"function","name":"lookup"},
            {"type":"namespace","name":"crm","tools":[{"type":"function","name":"lookup"}]}]}"#,
        r#"{"input":"hi","tools":[{"type":"namespace","name":"crm"}]}"#,
        r#"{"input":"hi","tools":[{"type":"function","function":null,"name":"f"}]}"#,
        r#"{"input":"hi","text":{"bogus":1}}"#,
    ] {
        let response = post(rejected);
        assert!(
            response.starts_with("HTTP/1.1 400"),
            "{rejected}\n{response}"
        );
    }
    // The 128-function cap counts flattened functions, in either order.
    let nested = format!(
        r#"{{"type":"namespace","name":"crm","tools":[{}]}}"#,
        (0..128)
            .map(|i| format!(r#"{{"type":"function","name":"f{i}"}}"#))
            .collect::<Vec<_>>()
            .join(",")
    );
    let outside = r#"{"type":"function","name":"outside"}"#;
    expect_status(
        &post(&format!(r#"{{"input":"hi","tools":[{nested}]}}"#)),
        200,
    );
    assert_eq!(tool_names().len(), 128);
    for tools in [format!("{outside},{nested}"), format!("{nested},{outside}")] {
        expect_status(
            &post(&format!(r#"{{"input":"hi","tools":[{tools}]}}"#)),
            400,
        );
    }
    // reasoning.summary is ignored while effort applies; text.verbosity is
    // ignored while text.format applies.
    expect_status(
        &post(r#"{"input":"hi","reasoning":{"effort":"low","summary":"auto"}}"#),
        200,
    );
    assert_eq!(
        server.backend.last_call().chat.reasoning.effort,
        Some(ReasoningEffort::Low)
    );
    expect_status(&post(r#"{"input":"hi","text":{"verbosity":"low"}}"#), 200);
    assert!(server.backend.last_call().chat.response_format.is_none());
    expect_status(
        &post(r#"{"input":"hi","text":{"format":{"type":"json_object"},"verbosity":"low"}}"#),
        200,
    );
    assert!(server.backend.last_call().chat.response_format.is_some());

    // A Codex request.
    expect_status(
        &post(
            r#"{"model":"test","instructions":"You are a coding agent.","input":[
              {"type":"message","role":"developer","content":[
                {"type":"input_text","text":"AGENTS instructions"}]},
              {"type":"message","role":"user","content":[
                {"type":"input_text","text":"say hi"}]}],
            "reasoning":{"effort":"medium","summary":"auto"},
            "text":{"verbosity":"low"},
            "tool_choice":"auto","parallel_tool_calls":true,
            "store":false,"stream":false,
            "include":["reasoning.encrypted_content"],
            "prompt_cache_key":"cache-1","client_metadata":{"thread_id":"t-1"},
            "tools":[
              {"type":"function","name":"exec_command","strict":false,
               "parameters":{"type":"object",
                 "properties":{"cmd":{"type":"string"}},"required":["cmd"]}},
              {"type":"namespace","name":"multi_agent_v1","tools":[
                {"type":"function","name":"close_agent","strict":false,
                 "parameters":{"type":"object","properties":{}}}]},
              {"type":"web_search","external_web_access":false}]}"#,
        ),
        200,
    );
    let call = server.backend.last_call();
    assert_eq!(tool_names(), ["exec_command", "close_agent"]);
    assert_eq!(call.chat.reasoning.effort, Some(ReasoningEffort::Medium));
    let turns: Vec<(ChatRole, String)> = call
        .chat
        .messages
        .into_iter()
        .map(|m| (m.role, m.content))
        .collect();
    assert_eq!(
        turns,
        [
            (ChatRole::System, "You are a coding agent.".to_owned()),
            (ChatRole::Developer, "AGENTS instructions".to_owned()),
            (ChatRole::User, "say hi".to_owned()),
        ]
    );
    // The include leniency is Responses-only.
    expect_status(
        &server.post(
            "/v1/messages",
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"include":["x"]}"#,
        ),
        400,
    );

    // Calls of a flattened function carry its namespace; top-level ones none.
    for stream in [false, true] {
        let raw = r#"<tool_call>{"name":"close_agent","arguments":{}}</tool_call><tool_call>{"name":"exec","arguments":{}}</tool_call>"#;
        *server.backend.pieces.lock().expect("pieces") =
            raw.chars().map(|c| c.to_string().into_bytes()).collect();
        let response = post(&format!(
            r#"{{"input":"go","stream":{stream},"tools":[
              {{"type":"function","name":"exec","strict":false,
               "parameters":{{"type":"object","properties":{{}}}}}},
              {{"type":"namespace","name":"multi_agent_v1","tools":[
                {{"type":"function","name":"close_agent","strict":false,
                 "parameters":{{"type":"object","properties":{{}}}}}}]}}]}}"#
        ));
        expect_status(&response, 200);
        let (calls, _) = responses_calls(&response, stream);
        let namespaces: Vec<(String, String)> = calls
            .iter()
            .map(|c| (c.member_str("name", ""), c.member_str("namespace", "-")))
            .collect();
        assert_eq!(
            namespaces,
            [
                ("close_agent".to_owned(), "multi_agent_v1".to_owned()),
                ("exec".to_owned(), "-".to_owned()),
            ],
            "stream={stream}"
        );
    }
}

/// Responses accepts more than sixteen images across messages: the context,
/// not a count, bounds them.
#[test]
fn responses_take_more_than_sixteen_images() {
    let server = RunningServer::new(HttpServerOptions::default());
    server.backend.image_support.store(true, Ordering::SeqCst);
    let message = r#"{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AQID"}]}"#;
    let body = format!(r#"{{"input":[{}]}}"#, vec![message; 17].join(","));
    expect_status(&server.post("/v1/responses", &body), 200);
    let images: usize = server
        .backend
        .last_call()
        .chat
        .messages
        .iter()
        .map(|m| m.images.len())
        .sum();
    assert_eq!(images, 17);
}

#[test]
fn compatibility_sampling_presets() {
    use gufo_core::sampling::{SamplingOverrides, TextModelPreset};
    let server = RunningServer::new(HttpServerOptions::default());
    for (path, body) in [
        ("/v1/completions", r#"{"prompt":"hi"}"#),
        ("/completion", r#"{"prompt":"hi"}"#),
        ("/v1/responses", r#"{"input":"hi"}"#),
        (
            "/v1/messages",
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}"#,
        ),
    ] {
        {
            let mut defaults = server.backend.defaults.lock().unwrap();
            *defaults = Default::default();
            defaults.model = TextModelPreset::Qwen38;
            defaults.supplied = SamplingOverrides::default();
        }
        server.backend.reasoning.lock().unwrap().enabled = Some(false);
        expect_status(&server.post(path, body), 200);
        let preset = server.backend.last_call().sampling;
        assert!(
            preset.temperature == 0.7
                && preset.top_p == 0.8
                && preset.top_k == 20
                && preset.presence_penalty == 1.5,
            "{path}: {preset:?}"
        );
        {
            let mut defaults = server.backend.defaults.lock().unwrap();
            defaults.sampling.top_k = 0;
            defaults.supplied.top_k = true;
        }
        let mut overriding = json::parse_str(body).expect("json");
        overriding.set("temperature", 0);
        overriding.set("presence_penalty", 0);
        expect_status(&server.post(path, &overriding.dump().expect("dump")), 200);
        let overridden = server.backend.last_call().sampling;
        assert!(
            overridden.temperature == 0.0
                && overridden.top_k == 0
                && overridden.top_p == 0.8
                && overridden.presence_penalty == 0.0,
            "{path}: {overridden:?}"
        );
    }
    *server.backend.defaults.lock().unwrap() = Default::default();
    *server.backend.reasoning.lock().unwrap() = Default::default();
}

#[test]
fn response_sampling_defaults() {
    use gufo_core::sampling::{SamplingOverrides, TextModelPreset};
    let server = RunningServer::new(HttpServerOptions::default());
    server.backend.defaults.lock().unwrap().model = TextModelPreset::Qwen38;
    for server_thinking in [false, true] {
        server.backend.reasoning.lock().unwrap().enabled = Some(server_thinking);
        for effort in [
            "null",
            "\"none\"",
            "\"minimal\"",
            "\"low\"",
            "\"medium\"",
            "\"high\"",
            "\"xhigh\"",
            "\"max\"",
        ] {
            let mut body = json::parse_str(
                r#"{"input":"hello","max_output_tokens":1,"temperature":null,"top_p":null,"presence_penalty":null,"reasoning":{}}"#,
            )
            .expect("json");
            body.member_mut("reasoning")
                .set("effort", json::parse_str(effort).expect("json"));
            let thinking = if effort == "null" {
                server_thinking
            } else {
                effort != "\"none\""
            };
            {
                let mut defaults = server.backend.defaults.lock().unwrap();
                defaults.sampling = Default::default();
                defaults.supplied = SamplingOverrides::default();
            }
            expect_status(
                &server.post("/v1/responses", &body.dump().expect("dump")),
                200,
            );
            let call = server.backend.last_call();
            assert_eq!(call.chat.reasoning.enabled, Some(thinking), "{effort}");
            let (temperature, top_p, presence) = if thinking {
                (1.0, 0.95, 0.0)
            } else {
                (0.7, 0.8, 1.5)
            };
            assert!(
                call.sampling.temperature == temperature
                    && call.sampling.top_p == top_p
                    && call.sampling.presence_penalty == presence
                    && call.sampling.top_k == 20,
                "{effort} thinking={thinking}: {:?}",
                call.sampling
            );
            // Explicit CLI values survive reasoning changes and SDK nulls.
            {
                let mut defaults = server.backend.defaults.lock().unwrap();
                defaults.sampling.temperature = 0.25;
                defaults.sampling.top_k = 0;
                defaults.supplied = SamplingOverrides {
                    temperature: true,
                    top_k: true,
                    ..SamplingOverrides::default()
                };
            }
            expect_status(
                &server.post("/v1/responses", &body.dump().expect("dump")),
                200,
            );
            let call = server.backend.last_call();
            assert!(
                call.sampling.temperature == 0.25
                    && call.sampling.top_k == 0
                    && call.sampling.top_p == top_p
            );
            // A request zero overrides both CLI and model values.
            body.set("temperature", 0);
            body.set("presence_penalty", 0);
            body.set("top_k", 3);
            expect_status(
                &server.post("/v1/responses", &body.dump().expect("dump")),
                200,
            );
            let call = server.backend.last_call();
            assert!(
                call.sampling.temperature == 0.0
                    && call.sampling.top_k == 3
                    && call.sampling.presence_penalty == 0.0
            );
        }
    }
    // Responses keep the request-owned cache control of Chat Completions.
    let mut body = json::parse_str(r#"{"input":"hello","max_output_tokens":1}"#).expect("json");
    expect_status(
        &server.post("/v1/responses", &body.dump().expect("dump")),
        200,
    );
    assert!(server.backend.last_call().chat.cache_prompt);
    for cache in [false, true] {
        body.set("cache_prompt", cache);
        expect_status(
            &server.post("/v1/responses", &body.dump().expect("dump")),
            200,
        );
        assert_eq!(server.backend.last_call().chat.cache_prompt, cache);
    }
    // As in llama-server, null keeps the default.
    body.set("cache_prompt", json::Value::Null);
    expect_status(
        &server.post("/v1/responses", &body.dump().expect("dump")),
        200,
    );
    assert!(server.backend.last_call().chat.cache_prompt);
    for invalid in ["0", "\"false\""] {
        body.set("cache_prompt", json::parse_str(invalid).expect("json"));
        let calls = server.backend.calls.load(Ordering::SeqCst);
        expect_status(
            &server.post("/v1/responses", &body.dump().expect("dump")),
            400,
        );
        assert_eq!(server.backend.calls.load(Ordering::SeqCst), calls);
    }
}

#[test]
fn raw_completion_streaming() {
    let server = RunningServer::new(HttpServerOptions::default());
    let models = server.send("GET /v1/models HTTP/1.1\r\n\r\n", false);
    expect_status(&models, 200);
    let listing = json::parse_str(body_of(&models)).expect("json");
    assert_eq!(
        listing.find("data").expect("data").items()[0].member_size("context_length", 0),
        65536
    );

    let response = server.post(
        "/v1/completions",
        r#"{"prompt":"hello","max_tokens":256,"stream":true,"stream_options":{"include_usage":true},"ignore_eos":true}"#,
    );
    expect_status(&response, 200);
    assert!(response.contains("text/event-stream"));
    let events: Vec<json::Value> = body_of(&response)
        .split("data: ")
        .filter_map(|event| event.split("\n\n").next())
        .filter(|data| data.starts_with('{'))
        .map(|data| json::parse_str(data).expect("event"))
        .collect();
    assert_eq!(events.len(), 3, "{response}");
    let choice = |event: &json::Value| event.find("choices").expect("choices").items().to_vec();
    let content = &events[0];
    assert_eq!(content.member_str("object", ""), "text_completion");
    assert_eq!(choice(content).len(), 1);
    assert_eq!(choice(content)[0].member_str("text", ""), "ok");
    assert!(
        choice(content)[0]
            .find("finish_reason")
            .expect("finish")
            .is_null()
    );
    assert!(content.find("usage").is_none());
    let terminal = &events[1];
    assert!(choice(terminal)[0].member_str("text", "x").is_empty());
    assert_eq!(choice(terminal)[0].member_str("finish_reason", ""), "stop");
    assert!(terminal.find("usage").is_none());
    let usage = &events[2];
    assert!(choice(usage).is_empty());
    assert_eq!(
        usage
            .find("usage")
            .expect("usage")
            .member_size("completion_tokens", 0),
        1
    );
    assert_eq!(
        usage
            .find("usage")
            .expect("usage")
            .member_size("cached_tokens", 0),
        8
    );
    assert!(response.contains("data: [DONE]\n\n"));
    assert!(server.backend.last_ignore_eos.load(Ordering::SeqCst));
    assert_eq!(server.backend.last_call().max_tokens, 256);

    // The terminal chunk carries llama.cpp's timings without a usage chunk.
    let without_usage = server.post("/v1/completions", r#"{"prompt":"hello","stream":true}"#);
    expect_status(&without_usage, 200);
    assert!(without_usage.contains("\"timings\":"), "{without_usage}");
    assert!(!without_usage.contains("\"usage\":"), "{without_usage}");

    // Before the stream starts a failure is a JSON error; after a token it
    // is a terminal SSE error. The failure's own message reaches the client.
    for failure in [1, 5, 6] {
        server.backend.failure.store(failure, Ordering::SeqCst);
        let message = if failure == 5 {
            "generation failed"
        } else {
            "context exceeded"
        };
        for (path, body) in [
            (
                "/v1/completions",
                r#"{"prompt":"hello","stream":true,"stream_options":{"include_usage":true}}"#,
            ),
            (
                "/v1/chat/completions",
                r#"{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
            ),
            ("/v1/responses", r#"{"input":"hello","stream":true}"#),
        ] {
            let failed = server.post(path, body);
            expect_status(&failed, if failure == 6 { 200 } else { 500 });
            assert!(
                failed.contains(&format!("\"message\":\"{message}\"")),
                "{path} {failure}: {failed}"
            );
            if failure != 6 {
                assert!(
                    failed.contains("Content-Type: application/json"),
                    "{failed}"
                );
                assert!(!failed.contains("data: "), "{failed}");
                assert!(failed.contains("\"code\":\"server_exception\""), "{failed}");
                continue;
            }
            let responses = path == "/v1/responses";
            assert!(
                failed.contains(if responses {
                    "\"code\":\"server_error\""
                } else {
                    "\"code\":\"generation_failed\""
                }),
                "{failed}"
            );
            assert!(
                failed.contains(if responses {
                    "event: response.failed"
                } else {
                    "data: [DONE]\n\n"
                }),
                "{failed}"
            );
            assert!(!failed.contains("event: response.completed"), "{failed}");
            assert!(failed.ends_with("0\r\n\r\n"), "{failed}");
        }
    }
    server.backend.failure.store(0, Ordering::SeqCst);

    for body in [
        r#"{"prompt":"hello","stream_options":{"include_usage":true}}"#,
        r#"{"prompt":"hello","stream":true,"stream_options":{"include_usage":"true"}}"#,
        r#"{"prompt":"hello","stream":true,"stream_options":{"other":true}}"#,
        r#"{"prompt":"hello","ignore_eos":1}"#,
        r#"{"prompt":"hello","stream":true,"text":{"format":{"type":"json_object"}}}"#,
        r#"{"prompt":"hello","stream":true,"reasoning":{"effort":"none"}}"#,
    ] {
        expect_status(&server.post("/v1/completions", body), 400);
    }
    // Raw Completions owns the fixed-length contract; the other text routes
    // refuse the field instead of generating a shorter run.
    expect_status(
        &server.post(
            "/v1/chat/completions",
            r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"ignore_eos":true}"#,
        ),
        400,
    );
}

#[test]
fn compatibility_stop_sequences() {
    let server = RunningServer::new(HttpServerOptions::default());
    *server.backend.forced_stop_sequence.lock().unwrap() = "END".into();
    for (path, body) in [
        ("/v1/completions", r#"{"prompt":"hello","stop":"END"}"#),
        ("/completion", r#"{"prompt":"hello","stop":["END"]}"#),
        (
            "/v1/messages",
            r#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":32,"stop_sequences":["END"]}"#,
        ),
    ] {
        let response = server.post(path, body);
        expect_status(&response, 200);
        let call = server.backend.last_call();
        let sequences = if path == "/v1/messages" {
            call.chat.stop_sequences
        } else {
            call.stop_sequences
        };
        assert_eq!(sequences, ["END"], "{path}");
        let output = json::parse_str(body_of(&response)).expect("json");
        match path {
            "/v1/messages" => {
                assert_eq!(output.member_str("stop_reason", ""), "stop_sequence");
                assert_eq!(output.member_str("stop_sequence", ""), "END");
            }
            "/completion" => {
                assert!(output.find("stopped_word").unwrap().as_bool(false));
                assert!(!output.find("stopped_eos").unwrap().as_bool(true));
                assert_eq!(output.member_str("stopping_word", ""), "END");
            }
            _ => assert_eq!(
                output.find("choices").unwrap().items()[0].member_str("finish_reason", ""),
                "stop"
            ),
        }
    }
    let before = server.backend.calls.load(Ordering::SeqCst);
    let messages = |stops: &str, limit: usize| {
        format!(
            r#"{{"messages":[{{"role":"user","content":"hello"}}],"max_tokens":{limit},"stop_sequences":{stops}}}"#
        )
    };
    for stop in ["null", "\"END\"", "[null]", "[\"\"]", "true"] {
        expect_status(&server.post("/v1/messages", &messages(stop, 32)), 400);
    }
    expect_status(
        &server.post("/v1/responses", r#"{"input":"hi","stop":"END"}"#),
        400,
    );
    let many: Vec<String> = (0..65).map(|i| format!("\"{i}\"")).collect();
    expect_status(
        &server.post(
            "/v1/messages",
            &messages(&format!("[{}]", many.join(",")), 32),
        ),
        400,
    );
    let long = format!("\"{}\"", "x".repeat(4096));
    let long = vec![long; 5].join(",");
    expect_status(
        &server.post("/v1/messages", &messages(&format!("[{long}]"), 32)),
        400,
    );
    assert_eq!(server.backend.calls.load(Ordering::SeqCst), before);
    server.backend.forced_stop_sequence.lock().unwrap().clear();
    for limit in [1, 32] {
        let response = server.post("/v1/messages", &messages("[]", limit));
        expect_status(&response, 200);
        let output = json::parse_str(body_of(&response)).expect("json");
        assert_eq!(
            output.member_str("stop_reason", ""),
            if limit == 1 { "max_tokens" } else { "end_turn" }
        );
        assert!(output.find("stop_sequence").unwrap().is_null());
    }
}

#[test]
fn compatibility_thinking_defaults() {
    let server = RunningServer::new(HttpServerOptions::default());
    for enabled in [false, true] {
        let defaults = ReasoningOptions {
            enabled: Some(enabled),
            effort: Some(ReasoningEffort::High),
            preserve_thinking: Some(true),
            budget_tokens: None,
        };
        *server.backend.reasoning.lock().unwrap() = defaults;
        for (path, body) in [
            ("/v1/responses", r#"{"input":"hello"}"#),
            (
                "/v1/messages",
                r#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":32}"#,
            ),
        ] {
            expect_status(&server.post(path, body), 200);
            assert_eq!(
                server.backend.last_call().chat.reasoning,
                defaults,
                "{path}"
            );
        }
    }
}

#[test]
fn invalid_bind_settings() {
    for port in [-1, 65536] {
        let server = HttpServer::new(
            "127.0.0.1",
            port,
            None,
            HttpServices::default(),
            HttpServerOptions::default(),
        )
        .expect("server");
        let error = server.start().expect_err("invalid port");
        assert!(!error.is_empty());
    }
    let invalid = HttpServer::new(
        "bad.address",
        0,
        None,
        HttpServices::default(),
        HttpServerOptions::default(),
    )
    .expect("server");
    assert!(!invalid.start().expect_err("invalid host").is_empty());
}

/// Numeric IPv4 and IPv6 listen addresses; `::` is dual-stack and an IPv4
/// peer keeps its IPv4 client ID (`TestListenAddresses`).
#[test]
fn listen_addresses() {
    for host in ["127.0.0.1", "0.0.0.0", "::", "::1", "fe80::1%lo"] {
        assert!(is_listen_address(host), "{host}");
    }
    for host in ["", "localhost", "127.1", "[::1]", "bad.address"] {
        assert!(!is_listen_address(host), "{host}");
    }
    if std::net::TcpListener::bind("[::1]:0").is_err() {
        println!("IPv6 loopback unavailable; skipping IPv6 listener checks.");
        return;
    }
    let peers = |host: &str, clients: &[&str]| -> Vec<String> {
        let server = RunningServer::on_host(host);
        server.server.add(
            "GET",
            "/peer",
            Arc::new(|request, _| {
                Ok(HttpResponse::new(
                    200,
                    "OK",
                    request.client_id.clone().into_bytes(),
                ))
            }),
        );
        clients
            .iter()
            .map(|client| {
                let address: std::net::IpAddr = client.parse().expect("client address");
                let mut stream =
                    std::net::TcpStream::connect((address, server.server.port())).expect("connect");
                stream
                    .write_all(b"GET /peer HTTP/1.1\r\n\r\n")
                    .expect("send");
                let mut response = Vec::new();
                let _ = stream.read_to_end(&mut response);
                let response = String::from_utf8_lossy(&response).into_owned();
                expect_status(&response, 200);
                body_of(&response).to_owned()
            })
            .collect()
    };
    assert_eq!(peers("::1", &["::1"]), ["::1"]);
    assert_eq!(peers("::", &["::1", "127.0.0.1"]), ["::1", "127.0.0.1"]);
}

/// A free port on the loopback, released for the code under test.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("free port")
        .port()
}

/// Sends `parts` with pauses between them and reads to EOF.
fn send_parts(port: u16, parts: &[&str]) -> String {
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .expect("timeout");
    for (index, part) in parts.iter().enumerate() {
        if index > 0 {
            std::thread::sleep(Duration::from_millis(50));
        }
        stream.write_all(part.as_bytes()).expect("send");
    }
    let mut response = Vec::new();
    let _ = stream.read_to_end(&mut response);
    String::from_utf8_lossy(&response).into_owned()
}

fn expect_loading(response: &str) {
    expect_status(response, 503);
    assert!(response.contains("\r\nRetry-After: 1\r\n"), "{response}");
    assert_eq!(
        body_of(response),
        r#"{"error":{"message":"Loading model","type":"unavailable_error","code":"loading"}}"#
    );
}

/// The port answers 503 `Loading model` while the model loads, then hands
/// over to the server (`TestLoadingResponder`).
#[test]
fn loading_responder() {
    // Nothing to hold for an ephemeral port or an address the server
    // rejects.
    assert!(!LoadingResponder::start("127.0.0.1", 0).listening());
    assert!(!LoadingResponder::start("127.0.0.1", 65536).listening());
    assert!(!LoadingResponder::start("localhost", i32::from(free_port())).listening());

    let port = free_port();
    let mut loading = LoadingResponder::start("127.0.0.1", i32::from(port));
    assert!(loading.listening());
    // Health and readiness included: a 200 would read as ready.
    for path in [
        "/health",
        "/v1/health",
        "/healthz",
        "/ready",
        "/v1/ready",
        "/readyz",
        "/v1/models",
    ] {
        expect_loading(&send_parts(
            port,
            &[&format!("GET {path} HTTP/1.1\r\n\r\n")],
        ));
    }
    expect_loading(&send_parts(
        port,
        &["POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}"],
    ));
    // A request head split across segments is answered after it completes.
    expect_loading(&send_parts(port, &["GET /hea", "lth HTTP/1.1\r\n", "\r\n"]));
    // The port is held: the server cannot bind it until stop.
    let early = HttpServer::new(
        "127.0.0.1",
        i32::from(port),
        None,
        HttpServices::default(),
        HttpServerOptions::default(),
    )
    .expect("server");
    assert!(early.start().is_err());
    // A client that connects and sends nothing delays neither the next
    // client nor stop beyond the bounded request wait.
    let idle = std::net::TcpStream::connect(("127.0.0.1", port)).expect("idle");
    let started = Instant::now();
    expect_loading(&send_parts(port, &["GET /health HTTP/1.1\r\n\r\n"]));
    assert!(started.elapsed() < Duration::from_secs(2));
    let idle_during_stop = std::net::TcpStream::connect(("127.0.0.1", port)).expect("idle");
    let started = Instant::now();
    loading.stop();
    assert!(started.elapsed() < Duration::from_secs(2));
    assert!(!loading.listening());
    drop((idle, idle_during_stop));

    // Handoff: the server binds the same address and answers health.
    let server = Arc::new(
        HttpServer::new(
            "127.0.0.1",
            i32::from(port),
            None,
            HttpServices::default(),
            HttpServerOptions::default(),
        )
        .expect("server"),
    );
    server.start().expect("handoff");
    let running = server.clone();
    let worker = std::thread::spawn(move || running.run(false));
    let health = send_parts(port, &["GET /health HTTP/1.1\r\n\r\n"]);
    expect_status(&health, 200);
    assert!(health.ends_with(r#"{"status":"ok"}"#), "{health}");
    server.stop();
    worker.join().expect("accept loop");

    // A port already in use leaves the responder idle.
    let owner = RunningServer::new(HttpServerOptions::default());
    assert!(!LoadingResponder::start("127.0.0.1", i32::from(owner.server.port())).listening());
}

#[test]
fn compatibility_utf8() {
    let server = RunningServer::new(HttpServerOptions::default());
    server
        .backend
        .set_output(b"\xc3\xa9\xe4\xb8\xad\xf0\x9f\x98\x80\xE2\x94!\xF0\x9F".to_vec());
    let expected = "é中😀\u{FFFD}!\u{FFFD}";
    for (path, input) in [
        ("/v1/completions", r#"{"prompt":"hi"}"#),
        ("/v1/responses", r#"{"input":"hi"}"#),
        (
            "/v1/messages",
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        ),
        ("/completion", r#"{"prompt":"hi"}"#),
    ] {
        let response = server.post(path, input);
        expect_status(&response, 200);
        let output = json::parse_str(body_of(&response)).expect("json");
        let text = match path {
            "/v1/completions" => {
                output.find("choices").expect("choices").items()[0].member_str("text", "")
            }
            "/v1/responses" => output.find("output").expect("output").items()[0]
                .find("content")
                .expect("content")
                .items()[0]
                .member_str("text", ""),
            "/v1/messages" => {
                output.find("content").expect("content").items()[0].member_str("text", "")
            }
            _ => output.member_str("content", ""),
        };
        assert_eq!(text, expected, "{path}");
    }
}

#[test]
fn peer_disconnect() {
    let server = RunningServer::new(HttpServerOptions::default());
    server
        .backend
        .wait_for_disconnect
        .store(true, Ordering::SeqCst);
    let mut stream = server.connect();
    let body = r#"{"prompt":"hi","max_tokens":128}"#;
    let request = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).expect("send");
    assert!(
        server
            .backend
            .entered
            .try_acquire_for(Duration::from_secs(2))
    );
    drop(stream);
    assert!(
        server
            .backend
            .finished
            .try_acquire_for(Duration::from_secs(2))
    );
    assert!(server.backend.disconnected.load(Ordering::SeqCst));
}

#[test]
fn streaming_framing() {
    let server = RunningServer::new(HttpServerOptions::default());
    let chunks = "3\r\na\0b\r\n3\r\nend\r\n";
    // A failed stream stays incomplete unless its error event was sent.
    for body in ["", "fail", "reported"] {
        let response = server.post("/stream", body);
        expect_status(&response, 200);
        assert!(response.contains("Transfer-Encoding: chunked\r\n"));
        assert_eq!(
            body_of(&response),
            format!("{chunks}{}", if body == "fail" { "" } else { "0\r\n\r\n" })
        );
    }
    let thrown = server.post("/stream-error", "");
    assert_eq!(body_of(&thrown), "b\r\nfirst chunk\r\n");
    // HTTP/1.0 clients keep close-delimited framing.
    let legacy = server.send("POST /stream HTTP/1.0\r\n\r\n", false);
    assert!(!legacy.contains("Transfer-Encoding:"));
    assert_eq!(body_of(&legacy), "a\0bend");
}

/// SIGINT and SIGTERM stop an idle and a busy server through normal
/// cleanup. Signals go to a child process running
/// [`signal_shutdown_child`], never to the test runner itself.
#[test]
fn signal_shutdown() {
    let executable = std::env::current_exe().expect("test executable");
    for signal in ["INT", "TERM"] {
        for active in [false, true] {
            let status = std::process::Command::new(&executable)
                .args(["--exact", "signal_shutdown_child", "--ignored", "--quiet"])
                .env("GUFO_SIGNAL_CHILD", format!("{signal},{active}"))
                .status()
                .expect("child");
            assert!(status.success(), "SIG{signal} active={active}: {status}");
        }
    }
}

#[test]
#[ignore = "the child half of signal_shutdown"]
fn signal_shutdown_child() {
    let Ok(role) = std::env::var("GUFO_SIGNAL_CHILD") else {
        return;
    };
    let (signal, active) = role.split_once(',').expect("role");
    let signal = match signal {
        "INT" => signal_hook::consts::SIGINT,
        _ => signal_hook::consts::SIGTERM,
    };
    let server = RunningServer::with_signals(HttpServerOptions::default());
    // An answered request proves that `run` installed its handlers.
    expect_status(&server.post("/echo", "ready"), 200);
    let mut client = None;
    if active == "true" {
        server
            .backend
            .wait_for_disconnect
            .store(true, Ordering::SeqCst);
        let mut stream = server.connect();
        let body = r#"{"prompt":"hello"}"#;
        stream
            .write_all(
                format!(
                    "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .expect("send");
        assert!(
            server
                .backend
                .entered
                .try_acquire_for(Duration::from_secs(2))
        );
        client = Some(stream);
    }
    signal_hook::low_level::raise(signal).expect("raise");
    assert!(
        server.run_finished.try_acquire_for(Duration::from_secs(2)),
        "the accept loop returned"
    );
    if client.is_some() {
        assert!(
            server.backend.disconnected.load(Ordering::SeqCst),
            "the active request was cancelled"
        );
    }
}

#[test]
fn connection_limit_answers_503() {
    let server = RunningServer::new(HttpServerOptions {
        max_connections: 1,
        ..HttpServerOptions::default()
    });
    server
        .backend
        .wait_for_disconnect
        .store(true, Ordering::SeqCst);
    let mut held = server.connect();
    let body = r#"{"prompt":"hi","max_tokens":1}"#;
    held.write_all(
        format!(
            "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    )
    .expect("send");
    assert!(
        server
            .backend
            .entered
            .try_acquire_for(Duration::from_secs(2))
    );
    let refused = server.send("GET /health HTTP/1.1\r\n\r\n", false);
    expect_status(&refused, 503);
    assert!(refused.contains("Retry-After: 1"));
    assert!(refused.contains("\"code\":\"overloaded\""));
    drop(held);
    assert!(
        server
            .backend
            .finished
            .try_acquire_for(Duration::from_secs(2))
    );
}

/// The server's streamed Responses events, kind and data.
fn responses_events(server: &RunningServer, body: &str) -> Vec<(String, json::Value)> {
    let response = server.post("/v1/responses", body);
    expect_status(&response, 200);
    assert!(response.contains("text/event-stream"));
    body_of(&response)
        .split("event: ")
        .skip(1)
        .map(|event| {
            let (kind, rest) = event.split_once('\n').expect("event line");
            let data = rest.strip_prefix("data: ").expect("data line");
            let data = data.split("\n\n").next().expect("event end");
            (kind.to_owned(), json::parse_str(data).expect("event json"))
        })
        .collect()
}

/// Streamed Responses: the reasoning events of each
/// `--responses-reasoning-target` (upstream's summary by default, OpenAI's
/// and llama.cpp's `reasoning_text`, Open Responses' `reasoning`), with the
/// sequence numbers and indices the OpenAI SDK validates, the items of the
/// buffered response, and a terminal event for limits and failures.
#[test]
fn responses_streaming() {
    let items = |response: &json::Value| -> Vec<String> {
        response
            .find("output")
            .expect("output")
            .items()
            .iter()
            .map(|item| {
                let mut item = item.clone();
                item.set("id", "");
                item.dump().expect("dump")
            })
            .collect()
    };
    for (target, part, text) in [
        (
            ResponsesReasoningTarget::Summary,
            "response.reasoning_summary_part",
            "response.reasoning_summary_text",
        ),
        (
            ResponsesReasoningTarget::ReasoningText,
            "response.content_part",
            "response.reasoning_text",
        ),
        (
            ResponsesReasoningTarget::Reasoning,
            "response.content_part",
            "response.reasoning",
        ),
    ] {
        let server = RunningServer::new(HttpServerOptions {
            responses_reasoning_target: target,
            ..HttpServerOptions::default()
        });
        // The whole text arrives as one piece, the model's `<think>` included.
        for (initial, output) in [
            (
                InitialOutputState::Reasoning,
                &b"plan\n</think>\n\nanswer"[..],
            ),
            (InitialOutputState::Auto, &b"<think>plan</think>answer"[..]),
        ] {
            server.backend.set_initial_output_state(initial);
            server.backend.set_output(output.to_vec());
            let buffered =
                json::parse_str(body_of(&server.post("/v1/responses", r#"{"input":"hi"}"#)))
                    .expect("json");
            assert_reasoning_item(
                &buffered.find("output").expect("output").items()[0],
                target,
                "plan",
            );
            let streamed = responses_events(&server, r#"{"input":"hi","stream":true}"#);
            let kinds: Vec<&str> = streamed.iter().map(|(kind, _)| kind.as_str()).collect();
            let reasoning = [
                format!("{part}.added"),
                format!("{text}.delta"),
                format!("{text}.done"),
                format!("{part}.done"),
            ];
            assert_eq!(
                kinds,
                [
                    "response.created",
                    "response.in_progress",
                    "response.output_item.added",
                    reasoning[0].as_str(),
                    reasoning[1].as_str(),
                    reasoning[2].as_str(),
                    reasoning[3].as_str(),
                    "response.output_item.done",
                    "response.output_item.added",
                    "response.content_part.added",
                    "response.output_text.delta",
                    "response.output_text.done",
                    "response.content_part.done",
                    "response.output_item.done",
                    "response.completed",
                ],
                "{target:?}"
            );
            for (index, (kind, event)) in streamed.iter().enumerate() {
                assert_eq!(&event.member_str("type", ""), kind);
                assert_eq!(event.member_size("sequence_number", usize::MAX), index);
            }
            let (delta, answer) = (&streamed[4].1, &streamed[10].1);
            assert_eq!(delta.member_str("delta", ""), "plan");
            assert_eq!(delta.member_size("output_index", 9), 0);
            // Summary parts are indexed apart from content parts.
            let index = if target == ResponsesReasoningTarget::Summary {
                "summary_index"
            } else {
                "content_index"
            };
            for event in &streamed[3..7] {
                assert_eq!(event.1.member_size(index, 9), 0, "{target:?} {}", event.0);
            }
            assert_eq!(streamed[5].1.member_str("text", ""), "plan");
            assert_eq!(answer.member_str("delta", ""), "answer");
            assert_eq!(answer.member_size("output_index", 9), 1);
            assert_eq!(answer.member_size("content_index", 9), 0);
            assert!(answer.find("logprobs").is_some_and(|l| l.is_empty()));
            let created = streamed[0].1.find("response").expect("response");
            assert_eq!(created.member_str("status", ""), "in_progress");
            assert!(created.find("output").is_some_and(|o| o.is_empty()));
            let completed = streamed[14].1.find("response").expect("response");
            assert_eq!(completed.member_str("status", ""), "completed");
            assert_eq!(items(completed), items(&buffered));
            assert_eq!(completed.find("usage"), buffered.find("usage"));
        }
    }

    let server = RunningServer::new(HttpServerOptions::default());
    let events = |body: &str| responses_events(&server, body);
    // The limit cuts the last item short.
    let limited = events(r#"{"input":"hi","stream":true,"max_output_tokens":1}"#);
    let (kind, last) = limited.last().expect("events");
    assert_eq!(kind, "response.incomplete");
    let response = last.find("response").expect("response");
    assert_eq!(response.member_str("status", ""), "incomplete");
    assert_eq!(
        response
            .find("incomplete_details")
            .expect("details")
            .member_str("reason", ""),
        "max_output_tokens"
    );
    let output = response.find("output").expect("output").items();
    assert_eq!(
        output.last().expect("item").member_str("status", ""),
        "incomplete"
    );

    // A failure after the head ends the stream with response.failed.
    server.backend.failure.store(6, Ordering::SeqCst);
    let failed = events(r#"{"input":"hi","stream":true}"#);
    server.backend.failure.store(0, Ordering::SeqCst);
    let (kind, last) = failed.last().expect("events");
    assert_eq!(kind, "response.failed");
    let response = last.find("response").expect("response");
    assert_eq!(response.member_str("status", ""), "failed");
    assert_eq!(
        response
            .find("error")
            .expect("error")
            .member_str("code", ""),
        "server_error"
    );

    // Chat streaming splits a piece that carries the model's `<think>`.
    server
        .backend
        .set_initial_output_state(InitialOutputState::Auto);
    server
        .backend
        .set_output(b"<think>plan</think>answer".to_vec());
    let chat = server.post(
        "/v1/chat/completions",
        r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
    );
    assert!(chat.contains(r#""reasoning_content":"plan""#), "{chat}");
    assert!(chat.contains(r#""content":"answer""#), "{chat}");
}

/// Idle SSE streams carry `: ping` comments, on chunked and HTTP/1.0
/// streams alike.
#[test]
fn sse_heartbeat() {
    let server = RunningServer::new(HttpServerOptions {
        sse_heartbeat_interval: Duration::from_millis(20),
        ..HttpServerOptions::default()
    });
    let response = server.post("/sse-idle", "");
    expect_status(&response, 200);
    let first = response.find("data: first\n\n").expect("first event");
    let ping = first + response[first..].find(": ping\n\n").expect("ping");
    let second_ping = ping
        + 1
        + response[ping + 1..]
            .find(": ping\n\n")
            .expect("second ping");
    let last = second_ping
        + response[second_ping..]
            .find("data: last\n\n")
            .expect("last event");
    assert!(response[last..].contains("data: [DONE]\n\n"));
    assert!(response.ends_with("0\r\n\r\n"), "{response}");

    let legacy = server.send("POST /sse-idle HTTP/1.0\r\n\r\n", false);
    assert!(!legacy.contains("Transfer-Encoding:"), "{legacy}");
    assert!(legacy.contains(": ping\n\n"), "{legacy}");
    assert!(legacy.ends_with("data: [DONE]\n\n"), "{legacy}");

    let disabled = RunningServer::new(HttpServerOptions {
        sse_heartbeat_interval: Duration::ZERO,
        ..HttpServerOptions::default()
    });
    assert!(!disabled.post("/sse-idle", "").contains(": ping"));
}

/// Streams that finish or fail while the heartbeat enters its wait end at
/// once, not at its deadline: clients time out after three seconds
///.
#[test]
fn sse_heartbeat_shutdown() {
    let server = RunningServer::new(HttpServerOptions {
        sse_heartbeat_interval: Duration::from_secs(30),
        ..HttpServerOptions::default()
    });
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                for request in 0..16 {
                    let fail = request % 2 != 0;
                    let response = server.post("/sse-finish", if fail { "fail" } else { "" });
                    expect_status(&response, 200);
                    assert!(response.contains("data: first\n\n"));
                    assert!(!response.contains(": ping"));
                    assert_eq!(response.ends_with("0\r\n\r\n"), !fail, "{response}");
                    assert_eq!(response.contains("data: [DONE]\n\n"), !fail);
                }
            });
        }
    });
}

/// llama-server slots list one entry per session, and the metrics report
/// in-flight context and every request counter.
/// `/props` reports what the server runs, as llama-server's does, so a client
/// can record which model, build and settings answered it (`TestLlamaProps`).
#[test]
fn llama_props() {
    let server = RunningServer::new(HttpServerOptions::default());
    let props = |path: &str| {
        let response = server.send(&format!("GET {path} HTTP/1.1\r\n\r\n"), false);
        expect_status(&response, 200);
        json::parse_str(body_of(&response)).expect("json")
    };

    // Nothing is invented for what the backend does not report.
    let bare = props("/props");
    assert_eq!(bare.member_str("model", ""), "test");
    assert_eq!(bare.member_str("model_alias", ""), "test");
    for absent in ["model_path", "build_info", "total_slots"] {
        assert!(bare.find(absent).is_none(), "{absent}");
    }
    let generation = bare
        .find("default_generation_settings")
        .expect("generation");
    assert_eq!(generation.member_double("n_ctx", 0.0), 65536.0);
    let params = generation.find("params").expect("params");
    assert!(params.find("temperature").is_some());
    assert_eq!(params.member_double("n_predict", 0.0), -1.0);
    assert_eq!(params.member_double("max_tokens", 0.0), -1.0);
    let modalities = bare.find("modalities").expect("modalities");
    assert!(!modalities.find("vision").expect("vision").as_bool(true));

    *server.backend.served_path.lock().unwrap() = "/models/model-00001-of-00002.gguf".into();
    *server.backend.served_build.lock().unwrap() = "0.2.0 (abc1234)".into();
    server
        .backend
        .sessions
        .lock()
        .unwrap()
        .resize(2, SessionState::default());
    server.backend.image_support.store(true, Ordering::SeqCst);
    server.backend.defaults.lock().unwrap().max_tokens = 512;
    let full = props("/props");
    assert_eq!(
        full.member_str("model_path", ""),
        "/models/model-00001-of-00002.gguf"
    );
    assert_eq!(full.member_str("build_info", ""), "0.2.0 (abc1234)");
    assert_eq!(full.member_double("total_slots", 0.0), 2.0);
    let params = full
        .find("default_generation_settings")
        .and_then(|generation| generation.find("params"))
        .expect("params");
    assert_eq!(params.member_double("n_predict", 0.0), 512.0);
    let modalities = full.find("modalities").expect("modalities");
    assert!(modalities.find("vision").expect("vision").as_bool(false));

    // ?model= is accepted and not required, as outside llama-server router
    // mode.
    let named = props("/props?model=alias");
    assert_eq!(named.member_str("model", ""), "alias");
    assert_eq!(named.member_str("model_alias", ""), "test");
    assert_eq!(named.member_str("build_info", ""), "0.2.0 (abc1234)");
}

#[test]
fn llama_slots_and_metrics() {
    let server = RunningServer::new(HttpServerOptions::default());
    let get = |path: &str| {
        let response = server.send(&format!("GET {path} HTTP/1.1\r\n\r\n"), false);
        expect_status(&response, 200);
        body_of(&response).to_owned()
    };
    let expect_slot = |slot: &json::Value, id: usize, processing: bool| {
        let field = |key: &str| slot.find(key).unwrap_or_else(|| panic!("{key}"));
        assert_eq!(slot.member_size("id", 99), id);
        assert_eq!(slot.member_size("n_ctx", 0), 65536);
        assert!(field("speculative").is_bool());
        assert_eq!(field("is_processing").as_bool(!processing), processing);
        assert_eq!(
            field("state").as_double(-1.0),
            f64::from(u8::from(processing))
        );
        assert_eq!(
            field("task_id").as_double(0.0),
            field("id_task").as_double(1.0)
        );
        for key in [
            "n_prompt_tokens",
            "n_prompt_tokens_cache",
            "n_prompt_tokens_processed",
        ] {
            assert!(field(key).is_number(), "{key}");
        }
        assert!(field("prompt").is_string() && slot.member_str("prompt", "x").is_empty());
        assert_eq!(slot.member_str("model", ""), "test");
        let next = field("next_token");
        assert!(next.is_array() && next.len() == 1);
        let token = &next.items()[0];
        assert_eq!(
            token.find("has_next_token").map(|v| v.as_bool(!processing)),
            Some(processing)
        );
        assert!(token.find("has_new_line").is_some_and(json::Value::is_bool));
        assert!(token.find("n_remain").is_some_and(json::Value::is_number));
        assert!(token.find("n_decoded").is_some_and(json::Value::is_number));
    };

    // Backends without a session pool report one idle slot.
    let slots = json::parse_str(&get("/slots")).expect("slots");
    assert!(slots.is_array() && slots.len() == 1);
    let fallback = &slots.items()[0];
    expect_slot(fallback, 0, false);
    assert_eq!(fallback.member_double("id_task", 0.0), -1.0);
    assert_eq!(fallback.member_size("n_prompt_tokens", 99), 0);
    assert_eq!(
        fallback.find("next_token").expect("next").items()[0].member_double("n_remain", 0.0),
        -1.0
    );
    assert!(get("/metrics").contains("llamacpp:kv_cache_usage_ratio 0\n"));

    *server.backend.sessions.lock().expect("sessions") = vec![
        SessionState::default(),
        SessionState {
            processing: true,
            speculative: true,
            request_id: 42,
            prompt_tokens: 10,
            cached_prompt_tokens: 8,
            processed_prompt_tokens: 2,
            generated_tokens: 3,
            remaining_tokens: 5,
        },
    ];
    let slots = json::parse_str(&get("/v1/slots")).expect("slots");
    assert_eq!(slots.len(), 2);
    expect_slot(&slots.items()[0], 0, false);
    let busy = &slots.items()[1];
    expect_slot(busy, 1, true);
    assert!(
        busy.find("speculative").is_some_and(|v| v.as_bool(false))
            && busy.member_size("id_task", 0) == 42
            && busy.member_size("n_prompt_tokens", 0) == 10
            && busy.member_size("n_prompt_tokens_cache", 0) == 8
            && busy.member_size("n_prompt_tokens_processed", 0) == 2
    );
    let token = &busy.find("next_token").expect("next").items()[0];
    assert!(token.member_size("n_remain", 0) == 5 && token.member_size("n_decoded", 0) == 3);

    let metrics = get("/metrics");
    let ratio = metrics
        .lines()
        .find_map(|line| line.strip_prefix("llamacpp:kv_cache_usage_ratio "))
        .expect("ratio");
    assert_eq!(
        ratio.parse::<f64>().expect("number"),
        13.0 / (2.0 * 65536.0)
    );
    for counter in [
        "prompt_tokens_total",
        "prompt_tokens_cached_total",
        "prompt_seconds_total",
        "tokens_predicted_total",
        "tokens_predicted_seconds_total",
        "n_tokens_max",
        "spec_decode_num_drafts_total",
        "spec_decode_num_draft_tokens_total",
        "spec_decode_num_accepted_tokens_total",
    ] {
        assert!(
            metrics.contains(&format!("# HELP llamacpp:{counter} ")),
            "{counter}"
        );
        assert!(
            metrics.contains(&format!("# TYPE llamacpp:{counter} counter\n")),
            "{counter}"
        );
    }
    assert!(metrics.contains("# TYPE llamacpp:kv_cache_usage_ratio gauge\n"));
    assert!(metrics.contains("# HELP gufo_device_lost_total "));
    assert!(metrics.contains("# TYPE gufo_device_lost_total counter\n"));
    assert!(metrics.contains("gufo_device_lost_total 0\n"));
}

/// Streamed `return_progress` requests carry llama-server `prompt_progress`
/// before their first token.
#[test]
fn raw_completion_prompt_progress() {
    let server = RunningServer::new(HttpServerOptions::default());
    *server.backend.progress.lock().expect("progress") = vec![PromptProgress {
        total: 4,
        cache: 1,
        processed: 4,
        time_ms: 2,
    }];
    let response = server.post(
        "/v1/completions",
        r#"{"prompt":"hello","stream":true,"return_progress":true}"#,
    );
    expect_status(&response, 200);
    let progress = response
        .find(concat!(
            r#""choices":[{"text":"","index":0,"logprobs":null,"finish_reason":null}],"#,
            r#""prompt_progress":{"total":4,"cache":1,"processed":4,"time_ms":2}}"#
        ))
        .expect("progress chunk");
    assert!(progress < response.find(r#""text":"ok""#).expect("text"));

    for body in [
        r#"{"prompt":"hello","stream":true}"#,
        r#"{"prompt":"hello","stream":true,"return_progress":null}"#,
        r#"{"prompt":"hello","return_progress":true}"#,
    ] {
        let response = server.post("/v1/completions", body);
        expect_status(&response, 200);
        assert!(!response.contains("prompt_progress"), "{body}");
    }
    expect_status(
        &server.post(
            "/v1/completions",
            r#"{"prompt":"hello","return_progress":1}"#,
        ),
        400,
    );
}

#[test]
fn chat_prompt_progress() {
    let server = RunningServer::new(HttpServerOptions::default());
    *server.backend.progress.lock().expect("progress") = vec![
        PromptProgress {
            total: 7,
            cache: 5,
            processed: 5,
            time_ms: 0,
        },
        PromptProgress {
            total: 7,
            cache: 5,
            processed: 7,
            time_ms: 3,
        },
    ];
    let stream = |options: &str| {
        let response = server.post(
            "/v1/chat/completions",
            &format!(
                r#"{{"model":"test","messages":[{{"role":"user","content":"hello"}}],"stream":true{options}}}"#
            ),
        );
        expect_status(&response, 200);
        response
    };
    let output = stream(r#","return_progress":true"#);
    let first = output.find(concat!(
        r#""delta":{},"finish_reason":null}],"#,
        r#""prompt_progress":{"total":7,"cache":5,"processed":5,"time_ms":0}}"#
    ));
    let last = output.find(concat!(
        r#""delta":{},"finish_reason":null}],"#,
        r#""prompt_progress":{"total":7,"cache":5,"processed":7,"time_ms":3}}"#
    ));
    let content = output.find(r#""content":"ok""#);
    assert!(
        matches!((first, last, content), (Some(first), Some(last), Some(content)) if first < last && last < content),
        "empty-delta prompt progress chunks precede streamed content: {output}"
    );
    for options in [
        "",
        r#","return_progress":false"#,
        r#","return_progress":null"#,
    ] {
        assert!(
            !stream(options).contains("prompt_progress"),
            "prompt progress is opt-in"
        );
    }
    let invalid = server.post(
        "/v1/chat/completions",
        r#"{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true,"return_progress":"yes"}"#,
    );
    expect_status(&invalid, 400);
    assert!(invalid.contains("invalid_return_progress"));
}

#[test]
fn responses_prompt_progress() {
    let server = RunningServer::new(HttpServerOptions::default());
    *server.backend.progress.lock().expect("progress") = vec![
        PromptProgress {
            total: 7,
            cache: 5,
            processed: 5,
            time_ms: 0,
        },
        PromptProgress {
            total: 7,
            cache: 5,
            processed: 7,
            time_ms: 3,
        },
    ];
    for (setting, expected) in [("true", 2), ("false", 0), ("null", 0)] {
        let response = server.post(
            "/v1/responses",
            &format!(
                r#"{{"model":"test","input":"hello","stream":true,"return_progress":{setting}}}"#
            ),
        );
        expect_status(&response, 200);
        let (mut sequence, mut progress_count, mut content_seen) = (0, 0, false);
        for data in body_of(&response)
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
        {
            let event = json::parse_str(data).expect("event");
            assert_eq!(
                event.member_size("sequence_number", usize::MAX),
                sequence,
                "progress preserves contiguous Responses sequence numbers"
            );
            sequence += 1;
            if let Some(progress) = event.find("prompt_progress") {
                assert!(
                    !content_seen
                        && event.member_str("type", "") == "response.in_progress"
                        && event
                            .find("response")
                            .is_some_and(|r| r.member_str("status", "") == "in_progress"),
                    "Responses progress precedes content in an in_progress event"
                );
                assert_eq!(
                    progress.member_size("processed", 0),
                    if progress_count == 0 { 5 } else { 7 }
                );
                progress_count += 1;
            }
            content_seen |= event.member_str("type", "") == "response.output_text.delta";
        }
        assert!(
            content_seen && progress_count == expected,
            "Responses progress is opt-in and preserves content ({setting})"
        );
    }
    expect_status(
        &server.post(
            "/v1/responses",
            r#"{"model":"test","input":"hello","stream":true,"return_progress":1}"#,
        ),
        400,
    );
}

/// `/v1/models` advertises image input only while a vision encoder is
/// loaded, under an unchanged model ID (upstream `5451525`).
#[test]
fn model_input_modalities() {
    let server = RunningServer::new(HttpServerOptions::default());
    for images in [false, true, false] {
        server.backend.image_support.store(images, Ordering::SeqCst);
        let response = server.send("GET /v1/models HTTP/1.1\r\n\r\n", false);
        expect_status(&response, 200);
        let listing = json::parse_str(body_of(&response)).expect("listing");
        assert_eq!(listing.member_str("object", ""), "list");
        let models = listing.find("data").expect("data").items();
        assert_eq!(models.len(), 1);
        let model = &models[0];
        assert_eq!(model.member_str("id", ""), "test");
        assert_eq!(model.member_str("owned_by", ""), "gufo");
        assert_eq!(model.member_size("context_length", 0), 65536);
        let modalities = model
            .find("architecture")
            .and_then(|architecture| architecture.find("input_modalities"))
            .expect("input_modalities");
        assert_eq!(
            modalities.dump().expect("dump"),
            if images {
                r#"["text","image"]"#
            } else {
                r#"["text"]"#
            }
        );
    }
}

/// A lost device answers 503 `device_lost` on health, readiness and every
/// generation route without reaching the backend, and runs the hook once
///.
#[test]
fn device_loss() {
    let hook_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counting = |calls: &std::sync::Arc<std::sync::atomic::AtomicUsize>| {
        let calls = calls.clone();
        move |code| {
            assert_eq!(code, GenerationErrorCode::DeviceLost);
            calls.fetch_add(1, Ordering::SeqCst);
        }
    };
    let server = RunningServer::new(HttpServerOptions::default());
    server.server.on_fatal_failure(counting(&hook_calls));
    let chat = r#"{"model":"test","messages":[{"role":"user","content":"hi"}]}"#;

    // A failure while the device still accepts work keeps the old contract.
    server.backend.failure.store(4, Ordering::SeqCst);
    let failed = server.post("/v1/completions", r#"{"prompt":"hello"}"#);
    expect_status(&failed, 500);
    assert!(failed.contains(r#""code":"server_exception""#), "{failed}");
    expect_status(&server.send("GET /health HTTP/1.1\r\n\r\n", false), 200);
    expect_status(&server.send("GET /ready HTTP/1.1\r\n\r\n", false), 200);
    assert_eq!(hook_calls.load(Ordering::SeqCst), 0);

    // A stream has committed its status; the terminal event names the loss.
    server.backend.failure.store(7, Ordering::SeqCst);
    let stream = server.post("/v1/completions", r#"{"prompt":"hello","stream":true}"#);
    expect_status(&stream, 200);
    assert!(stream.contains(r#""code":"device_lost""#), "{stream}");
    assert!(stream.contains(DEVICE_LOST_MESSAGE), "{stream}");
    assert_eq!(hook_calls.load(Ordering::SeqCst), 1);

    server.backend.failure.store(0, Ordering::SeqCst);
    for path in [
        "/health",
        "/v1/health",
        "/healthz",
        "/ready",
        "/v1/ready",
        "/readyz",
    ] {
        let response = server.send(&format!("GET {path} HTTP/1.1\r\n\r\n"), false);
        expect_status(&response, 503);
        assert!(response.contains(r#""status":"device_lost""#), "{path}");
        assert!(response.contains(r#""code":"device_lost""#), "{path}");
    }
    let calls = server.backend.calls.load(Ordering::SeqCst);
    for (path, body) in [
        ("/v1/chat/completions", chat),
        ("/v1/completions", r#"{"prompt":"hello","stream":true}"#),
        ("/v1/embeddings", r#"{"input":"hello"}"#),
    ] {
        let response = server.post(path, body);
        expect_status(&response, 503);
        assert!(response.contains(r#""code":"device_lost""#), "{path}");
        assert!(!response.contains("Retry-After"), "{path}");
    }
    assert_eq!(server.backend.calls.load(Ordering::SeqCst), calls);
    // Routes that never reach the device keep answering.
    expect_status(&server.send("GET /v1/models HTTP/1.1\r\n\r\n", false), 200);
    expect_status(&server.send("GET /metrics HTTP/1.1\r\n\r\n", false), 200);
    assert_eq!(hook_calls.load(Ordering::SeqCst), 1);

    // Before a response starts, the failing request itself gets the 503.
    let first_hook_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let first = RunningServer::new(HttpServerOptions::default());
    first.server.on_fatal_failure(counting(&first_hook_calls));
    first.backend.failure.store(3, Ordering::SeqCst);
    for (path, body) in [
        ("/v1/completions", r#"{"prompt":"hello"}"#),
        ("/v1/chat/completions", chat),
    ] {
        let response = first.post(path, body);
        expect_status(&response, 503);
        assert!(response.contains(r#""code":"device_lost""#), "{path}");
        assert!(response.contains(DEVICE_LOST_MESSAGE), "{path}");
        assert!(!response.contains("Retry-After"), "{path}");
    }
    assert_eq!(first.backend.calls.load(Ordering::SeqCst), 1);
    assert_eq!(first_hook_calls.load(Ordering::SeqCst), 1);

    // A stream that fails before it starts gets the 503 too.
    for (path, body) in [
        ("/v1/completions", r#"{"prompt":"hello","stream":true}"#),
        (
            "/v1/chat/completions",
            r#"{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
        ),
        ("/v1/responses", r#"{"input":"hello","stream":true}"#),
    ] {
        let early = RunningServer::new(HttpServerOptions::default());
        early.backend.failure.store(3, Ordering::SeqCst);
        let response = early.post(path, body);
        expect_status(&response, 503);
        assert!(response.contains(r#""code":"device_lost""#), "{path}");
        assert!(!response.contains("data: "), "{path}");
    }
}

/// A panicked scheduler answers like a lost device, with its own code:
/// 503 `scheduler_failed` on health, readiness and generation without
/// reaching the backend, and the hook runs once with that code.
#[test]
fn scheduler_failure() {
    let hook_codes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let server = RunningServer::new(HttpServerOptions::default());
    {
        let hook_codes = hook_codes.clone();
        server.server.on_fatal_failure(move |code| {
            hook_codes.lock().expect("codes").push(code);
        });
    }
    expect_status(&server.send("GET /health HTTP/1.1\r\n\r\n", false), 200);
    server
        .backend
        .scheduler_failed
        .store(true, Ordering::SeqCst);
    for path in ["/health", "/ready"] {
        let response = server.send(&format!("GET {path} HTTP/1.1\r\n\r\n"), false);
        expect_status(&response, 503);
        assert!(
            response.contains(r#""status":"scheduler_failed""#),
            "{path}"
        );
        assert!(response.contains(SCHEDULER_FAILED_MESSAGE), "{path}");
    }
    let calls = server.backend.calls.load(Ordering::SeqCst);
    let response = server.post(
        "/v1/chat/completions",
        r#"{"model":"test","messages":[{"role":"user","content":"hi"}]}"#,
    );
    expect_status(&response, 503);
    assert!(
        response.contains(r#""code":"scheduler_failed""#),
        "{response}"
    );
    assert!(!response.contains("Retry-After"), "{response}");
    assert_eq!(server.backend.calls.load(Ordering::SeqCst), calls);
    assert_eq!(
        *hook_codes.lock().expect("codes"),
        [GenerationErrorCode::SchedulerFailed]
    );
}

/// The listener observes a loss while a worker blocks writing to a client
/// that never reads.
#[test]
fn device_loss_while_writer_blocked() {
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Arc, mpsc};
    let hook_calls = Arc::new(AtomicUsize::new(0));
    let (hook_called, hook_seen) = mpsc::channel();
    let server = RunningServer::new(HttpServerOptions {
        sse_heartbeat_interval: Duration::ZERO,
        ..HttpServerOptions::default()
    });
    {
        let hook_calls = hook_calls.clone();
        let hook_called = std::sync::Mutex::new(hook_called);
        server.server.on_fatal_failure(move |_| {
            hook_calls.fetch_add(1, Ordering::SeqCst);
            let _ = hook_called.lock().expect("hook").send(());
        });
    }
    let (writing, write_started) = mpsc::channel();
    let write_finished = Arc::new(AtomicBool::new(false));
    {
        let writing = std::sync::Mutex::new(writing);
        let write_finished = write_finished.clone();
        server.server.add(
            "POST",
            "/blocked",
            Arc::new(move |_, _| {
                let writing = writing.lock().expect("writing").clone();
                let write_finished = write_finished.clone();
                Ok(gufo_serve::http::HttpResponse {
                    headers: vec![("Content-Type".into(), "text/event-stream".into())],
                    streaming_body: Some(Box::new(move |write| {
                        let chunk = vec![b'x'; 8 * 1024 * 1024];
                        let _ = writing.send(());
                        let _ = write(&chunk);
                        write_finished.store(true, Ordering::SeqCst);
                        Ok(())
                    })),
                    ..gufo_serve::http::HttpResponse::default()
                })
            }),
        );
    }
    let mut client = server.connect();
    rustix::net::sockopt::set_socket_recv_buffer_size(&client, 1024).expect("receive buffer");
    client
        .write_all(b"POST /blocked HTTP/1.1\r\n\r\n")
        .expect("send");
    write_started
        .recv_timeout(Duration::from_secs(2))
        .expect("the stream starts writing");
    // The peer never drains the response: no request completes to notice the
    // loss, so the listener has to.
    server.backend.lost.store(true, Ordering::SeqCst);
    hook_seen
        .recv_timeout(Duration::from_secs(2))
        .expect("the listener reports the loss");
    assert!(hook_calls.load(Ordering::SeqCst) == 1 && !write_finished.load(Ordering::SeqCst));
    let _ = client.shutdown(std::net::Shutdown::Both);
}

const TEXT_STREAMS: [(&str, &str); 3] = [
    ("/v1/completions", r#"{"prompt":"hello","stream":true}"#),
    (
        "/v1/chat/completions",
        r#"{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}"#,
    ),
    ("/v1/responses", r#"{"input":"hello","stream":true}"#),
];

/// Text streams commit their headers when they start, so an early failure
/// keeps its status and no heartbeat precedes it.
#[test]
fn deferred_stream_headers() {
    let server = RunningServer::new(HttpServerOptions {
        sse_heartbeat_interval: Duration::from_millis(5),
        ..HttpServerOptions::default()
    });
    server.server.add(
        "POST",
        "/early-failure",
        std::sync::Arc::new(|_, _| {
            Ok(gufo_serve::http::HttpResponse {
                headers: vec![("Content-Type".into(), "text/event-stream".into())],
                streaming_body: Some(Box::new(|_| {
                    std::thread::sleep(Duration::from_millis(30));
                    Err(gufo_core::text::Error::runtime("early generation failure"))
                })),
                defer_stream_headers: true,
                ..gufo_serve::http::HttpResponse::default()
            })
        }),
    );
    let failed = server.post("/early-failure", "");
    expect_status(&failed, 500);
    assert!(!failed.contains(": ping"), "{failed}");
    // Successful termination without a token still commits the stream.
    server.backend.failure.store(8, Ordering::SeqCst);
    server.backend.set_output(Vec::new());
    for (path, body) in TEXT_STREAMS {
        let empty = server.post(path, body);
        expect_status(&empty, 200);
        assert!(empty.ends_with("0\r\n\r\n"), "{path}: {empty}");
    }
    // Opt-in progress starts the stream before any token exists.
    *server.backend.progress.lock().expect("progress") = vec![PromptProgress {
        total: 100,
        processed: 10,
        ..PromptProgress::default()
    }];
    server.backend.failure.store(4, Ordering::SeqCst);
    let progress = server.post(
        "/v1/completions",
        r#"{"prompt":"hello","stream":true,"return_progress":true}"#,
    );
    expect_status(&progress, 200);
    assert!(progress.contains("prompt_progress"), "{progress}");
    assert!(
        progress.contains(r#""code":"generation_failed""#),
        "{progress}"
    );
    server.backend.progress.lock().expect("progress").clear();
    for (path, body) in [
        (
            "/v1/completions",
            r#"{"prompt":"hello","stream":true,"return_progress":true}"#,
        ),
        (
            "/v1/chat/completions",
            r#"{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true,"return_progress":true}"#,
        ),
        (
            "/v1/responses",
            r#"{"input":"hello","stream":true,"return_progress":true}"#,
        ),
    ] {
        let failure = server.post(path, body);
        expect_status(&failure, 200);
        assert!(
            failure.contains(if path == "/v1/responses" {
                r#""code":"server_error""#
            } else {
                r#""code":"generation_failed""#
            }),
            "{path}: {failure}"
        );
        assert!(failure.ends_with("0\r\n\r\n"), "{path}");
    }
}

/// Admission commits the headers, so heartbeats cover the prefill, and a
/// later failure is a terminal SSE error.
#[test]
fn admitted_stream_headers() {
    let server = RunningServer::new(HttpServerOptions {
        sse_heartbeat_interval: Duration::from_millis(5),
        ..HttpServerOptions::default()
    });
    server.backend.admit.store(true, Ordering::SeqCst);
    *server.backend.token_delay.lock().expect("delay") = Duration::from_millis(40);
    for (path, body) in TEXT_STREAMS {
        let response = server.post(path, body);
        expect_status(&response, 200);
        let ping = response.find(": ping\n\n").expect("ping");
        // Quoted, so a random response ID cannot match.
        assert!(ping < response.find("\"ok\"").expect("text"), "{path}");
        assert!(response.ends_with("0\r\n\r\n"), "{path}");
    }
    for (path, body) in TEXT_STREAMS {
        let lost = RunningServer::new(HttpServerOptions::default());
        lost.backend.admit.store(true, Ordering::SeqCst);
        lost.backend.failure.store(3, Ordering::SeqCst);
        let failed = lost.post(path, body);
        expect_status(&failed, 200);
        assert!(failed.contains(DEVICE_LOST_MESSAGE), "{path}: {failed}");
        assert!(failed.contains("data: "), "{path}");
        assert!(failed.ends_with("0\r\n\r\n"), "{path}");
    }
}

const LITERAL_ARGUMENT: &str = concat!(
    " <think>literal</think><tool_call></tool_call>",
    "<tool_call>{\"name\":\"f\",\"arguments\":{\"text\":\"nested\"}}</tool_call>",
    "<｜DSML｜tool_calls><｜DSML｜invoke name=\"f\"></｜DSML｜invoke></｜DSML｜tool_calls>",
    "</function></｜DSML｜invoke> \"42\" \\\nπ🦉\n \r"
);

/// The function call items of a Responses result, buffered or streamed.
fn responses_calls(response: &str, stream: bool) -> (Vec<json::Value>, Vec<json::Value>) {
    let body = body_of(response);
    if !stream {
        let result = json::parse_str(body).expect("response");
        let calls = result
            .find("output")
            .expect("output")
            .items()
            .iter()
            .filter(|item| item.member_str("type", "") == "function_call")
            .cloned()
            .collect();
        return (calls, vec![result]);
    }
    let events: Vec<json::Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| json::parse_str(data).expect("event"))
        .collect();
    let calls = events
        .iter()
        .filter(|event| event.member_str("type", "") == "response.completed")
        .flat_map(|event| {
            event
                .find("response")
                .and_then(|r| r.find("output"))
                .map_or(&[][..], |o| o.items())
                .iter()
                .filter(|item| item.member_str("type", "") == "function_call")
                .cloned()
                .collect::<Vec<_>>()
        })
        .collect();
    (calls, events)
}

/// Responses declares tools and returns native calls as function_call
/// items, streamed in llama.cpp's events.
#[test]
fn responses_native_tools() {
    let server = RunningServer::new(HttpServerOptions::default());
    let expected = json::Value::object()
        .with("text", LITERAL_ARGUMENT)
        .dump()
        .expect("dump");
    let calls = [
        format!(
            "<tool_call>\n<function=f>\n<parameter=text>\n{LITERAL_ARGUMENT}\n</parameter>\n\
             </function>\n</tool_call>"
        ),
        format!(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"text\" \
             string=\"true\">{LITERAL_ARGUMENT}</｜DSML｜parameter>\n</｜DSML｜invoke>\n\
             </｜DSML｜tool_calls>"
        ),
        format!("<tool_call>{{\"name\":\"f\",\"arguments\":{expected}}}</tool_call>"),
    ];
    for call in &calls {
        for thinking in [false, true] {
            for stream in [false, true] {
                for parallel in [false, true] {
                    for choice in ["required", "auto"] {
                        server.backend.set_initial_output_state(if thinking {
                            InitialOutputState::Reasoning
                        } else {
                            InitialOutputState::Content
                        });
                        let raw = format!(
                            "{}{call} \n",
                            if thinking { "Reasoning.</think>" } else { "" }
                        );
                        *server.backend.pieces.lock().expect("pieces") =
                            raw.chars().map(|c| c.to_string().into_bytes()).collect();
                        let body = format!(
                            r#"{{"model":"test","input":"call f","stream":{stream},
                            "tools":[{{"type":"function","name":"f","strict":false,
                              "parameters":{{"type":"object","properties":{{"text":{{"type":"string"}}}},
                              "required":["text"],"additionalProperties":false}}}}],
                            "tool_choice":"{choice}","parallel_tool_calls":{parallel}}}"#
                        );
                        let response = server.post("/v1/responses", &body);
                        expect_status(&response, 200);
                        let (items, events) = responses_calls(&response, stream);
                        assert!(
                            items.len() == 1
                                && items[0].member_str("arguments", "") == expected
                                && items[0].member_str("name", "") == "f"
                                && items[0].member_str("call_id", "").starts_with("call_"),
                            "native arguments survive Responses: {call:?} thinking={thinking} \
                             stream={stream} parallel={parallel} {choice}: {items:?}"
                        );
                        if stream {
                            let kinds: Vec<String> =
                                events.iter().map(|e| e.member_str("type", "")).collect();
                            let added = kinds
                                .iter()
                                .position(|k| k == "response.output_item.added")
                                .expect("added");
                            let delta = kinds
                                .iter()
                                .position(|k| k == "response.function_call_arguments.delta")
                                .expect("delta");
                            let done = kinds
                                .iter()
                                .position(|k| k == "response.function_call_arguments.done")
                                .expect("done");
                            let closed = kinds
                                .iter()
                                .rposition(|k| k == "response.output_item.done")
                                .expect("item done");
                            assert!(
                                added < delta && delta < done && done < closed,
                                "the OpenAI event order: {kinds:?}"
                            );
                            assert!(
                                events[delta].member_str("delta", "") == expected,
                                "the arguments arrive as one delta"
                            );
                            assert!(
                                events[done].member_str("arguments", "") == expected
                                    && events[done].member_str("name", "") == "f"
                                    && events[done].member_str("item_id", "")
                                        == events[delta].member_str("item_id", ""),
                                "the done event repeats the arguments and names the function"
                            );
                        }
                        let reasoning = response.contains("literal</think>")
                            && events.iter().any(|event| {
                                reasoning_delta(&event.member_str("type", ""))
                                    && event.member_str("delta", "").contains("literal")
                            });
                        assert!(!reasoning, "argument tags never enter the reasoning");
                    }
                }
            }
        }
    }
    server.backend.pieces.lock().expect("pieces").clear();
}

/// Named function choices are echoed, parallel calls may be null, CRLF
/// framing keeps a literal CR, and an explicit choice clears the name
///.
#[test]
fn responses_tool_metadata_and_framing() {
    let server = RunningServer::new(HttpServerOptions::default());
    *server.backend.pieces.lock().expect("pieces") = vec![
        b"<tool_call>\n<function=f>\n<parameter=text>\r\nliteral\r\r\n</parameter>\n</function>\n</tool_call>"
            .to_vec(),
    ];
    let body = r#"{"model":"test","input":"call f",
        "tools":[{"type":"function","name":"f","strict":true,
          "parameters":{"type":"object","properties":{"text":{"type":"string"}},
          "required":["text"],"additionalProperties":false}}],
        "tool_choice":{"type":"function","name":"f"},"parallel_tool_calls":null"#;
    for stream in [false, true] {
        let response = server.post("/v1/responses", &format!("{body},\"stream\":{stream}}}"));
        expect_status(&response, 200);
        let (items, events) = responses_calls(&response, stream);
        let results: Vec<&json::Value> = if stream {
            events.iter().filter_map(|e| e.find("response")).collect()
        } else {
            events.iter().collect()
        };
        assert!(!results.is_empty());
        for result in results {
            let choice = result.find("tool_choice").expect("tool_choice");
            assert!(
                choice.member_str("type", "") == "function" && choice.member_str("name", "") == "f",
                "Responses echoes named function choices in every lifecycle event"
            );
            assert!(
                !result
                    .find("parallel_tool_calls")
                    .expect("parallel")
                    .as_bool(true)
            );
            assert_eq!(
                result.find("tools").expect("tools").items()[0].member_str("name", ""),
                "f"
            );
        }
        let arguments = json::parse_str(&items[0].member_str("arguments", "")).expect("arguments");
        assert_eq!(
            arguments.member_str("text", ""),
            "literal\r",
            "CRLF framing retains a literal trailing carriage return"
        );
    }
    server.backend.pieces.lock().expect("pieces").clear();
    // Null keeps the default for Responses; Chat still requires a boolean.
    let auto = server.post(
        "/v1/responses",
        r#"{"model":"test","input":"hi","tool_choice":"auto","parallel_tool_calls":null,
           "tools":[{"type":"function","name":"f","parameters":{}}]}"#,
    );
    expect_status(&auto, 200);
    let result = json::parse_str(body_of(&auto)).expect("response");
    assert!(
        result
            .find("parallel_tool_calls")
            .expect("parallel")
            .as_bool(false)
    );
    assert_eq!(result.member_str("tool_choice", ""), "auto");
    expect_status(
        &server.post(
            "/v1/chat/completions",
            r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"parallel_tool_calls":null,
               "tools":[{"type":"function","function":{"name":"f","parameters":{}}}]}"#,
        ),
        400,
    );
}

/// Earlier function calls and their outputs replay as an assistant tool
/// call and a tool message, merged as llama.cpp does.
#[test]
fn responses_function_call_history() {
    let server = RunningServer::new(HttpServerOptions::default());
    let response = server.post(
        "/v1/responses",
        r#"{"model":"test","tools":[{"type":"function","name":"f","parameters":{}}],
           "input":[{"role":"user","content":"call f twice"},
             {"type":"message","role":"assistant","content":[{"type":"output_text","text":"Calling."}]},
             {"type":"function_call","call_id":"call_1","name":"f","arguments":"{\"n\":1}"},
             {"type":"function_call","call_id":"call_2","name":"f","arguments":"{\"n\":2}"},
             {"type":"function_call_output","call_id":"call_1","output":"one"},
             {"type":"function_call_output","call_id":"call_2",
              "output":[{"type":"input_text","text":"t"},{"type":"input_text","text":"wo"}]}]}"#,
    );
    expect_status(&response, 200);
    let messages = server.backend.last_call().chat.messages;
    assert_eq!(messages.len(), 4, "{messages:?}");
    assert!(messages[1].role == ChatRole::Assistant && messages[1].content == "Calling.");
    let calls = &messages[1].tool_calls;
    assert!(
        calls.len() == 2
            && calls[0].id == "call_1"
            && calls[1].id == "call_2"
            && calls[1].arguments[0].value == "2",
        "consecutive calls join the assistant message: {calls:?}"
    );
    assert!(
        messages[2].role == ChatRole::Tool
            && messages[2].tool_call_id == "call_1"
            && messages[2].content == "one"
    );
    assert!(messages[3].tool_call_id == "call_2" && messages[3].content == "two");
    for item in [
        r#"{"type":"function_call","call_id":"","name":"f","arguments":"{}"}"#,
        r#"{"type":"function_call","call_id":"c","name":"f","arguments":"[]"}"#,
        r#"{"type":"function_call_output","call_id":"c","output":1}"#,
    ] {
        expect_status(
            &server.post(
                "/v1/responses",
                &format!(r#"{{"model":"test","input":[{item}]}}"#),
            ),
            400,
        );
    }
}

/// Content and function calls of a Responses result, with the streamed
/// content and reasoning deltas.
struct ResponsesParts {
    calls: Vec<json::Value>,
    content: String,
    reasoning: String,
    streamed_content: String,
    streamed_reasoning: String,
}

/// A streamed reasoning delta of any `--responses-reasoning-target`.
fn reasoning_delta(kind: &str) -> bool {
    matches!(
        kind,
        "response.reasoning_summary_text.delta"
            | "response.reasoning_text.delta"
            | "response.reasoning.delta"
    )
}

fn responses_parts(response: &str, stream: bool) -> ResponsesParts {
    let mut parts = ResponsesParts {
        calls: Vec::new(),
        content: String::new(),
        reasoning: String::new(),
        streamed_content: String::new(),
        streamed_reasoning: String::new(),
    };
    let body = body_of(response);
    let output = if stream {
        let mut completed = None;
        for data in body.lines().filter_map(|line| line.strip_prefix("data: ")) {
            let event = json::parse_str(data).expect("event");
            match event.member_str("type", "").as_str() {
                "response.output_text.delta" => {
                    parts.streamed_content += &event.member_str("delta", "");
                }
                kind if reasoning_delta(kind) => {
                    parts.streamed_reasoning += &event.member_str("delta", "");
                }
                "response.completed" => completed = event.find("response").cloned(),
                _ => {}
            }
        }
        completed.expect("completed")
    } else {
        json::parse_str(body).expect("response")
    };
    for item in output.find("output").expect("output").items() {
        match item.member_str("type", "").as_str() {
            "function_call" => parts.calls.push(item.clone()),
            kind @ ("message" | "reasoning") => {
                let parts_of = |field| item.find(field).map_or(&[][..], |p| p.items());
                for part in parts_of("summary").iter().chain(parts_of("content")) {
                    let target = if kind == "message" {
                        &mut parts.content
                    } else {
                        &mut parts.reasoning
                    };
                    *target += &part.member_str("text", "");
                }
            }
            _ => {}
        }
    }
    parts
}

const QUOTE_TOOLS: &str = r#""tools":[{"type":"function","name":"f","parameters":{"type":"object",
    "properties":{"text":{"type":"string","const":"42"}},"required":["text"],
    "additionalProperties":false}}]"#;

/// A Responses request over [`QUOTE_TOOLS`] whose output follows the
/// constrained or the free-form tool parser.
fn quote_chat(constrained: bool) -> ChatRequest {
    let mut chat = ChatRequest::default();
    let controls = json::parse_str(&format!("{{{QUOTE_TOOLS}}}")).expect("json");
    parse_tool_controls(&controls, &mut chat, true).expect("valid tool controls");
    chat.constrained_tools = constrained;
    chat
}

/// The response `create_openai_response` gives `chat` on `server`'s
/// backend, as HTTP response text for [`responses_parts`].
fn direct_response(server: &RunningServer, chat: &ChatRequest, stream: bool) -> String {
    let backend: Arc<dyn TextGenerationBackend> = server.backend.clone();
    let mut response = create_openai_response(
        &HttpRequest::default(),
        &backend,
        chat,
        512,
        &SamplingConfig::default(),
        stream,
        ResponsesReasoningTarget::default(),
    )
    .expect("response");
    let mut body = response.body.clone();
    if let Some(streaming) = response.streaming_body.take() {
        streaming(&mut |chunk: &[u8]| {
            body.extend_from_slice(chunk);
            true
        })
        .expect("stream");
    }
    format!("HTTP/1.1 200 OK\r\n\r\n{}", String::from_utf8_lossy(&body))
}

/// Markers in completed quotes are prose; after an unclosed quote only a
/// schema-valid declared call counts, in both parsers.
#[test]
fn unclosed_quote_fallback_checks_schema() {
    let server = RunningServer::new(HttpServerOptions::default());
    server
        .backend
        .set_initial_output_state(InitialOutputState::Content);
    let prefix = "```python\nprint(1)\n";
    let valid =
        "<tool_call>\n<function=f>\n<parameter=text>42</parameter>\n</function>\n</tool_call>";
    let wrong = "<tool_call><function=f><parameter=text>43</parameter></function></tool_call>";
    let missing = "<tool_call><function=f></function></tool_call>";
    let cases = [
        (format!("{prefix}{valid}"), true, prefix.to_owned()),
        (
            format!("{prefix}{wrong}"),
            false,
            format!("{prefix}{wrong}"),
        ),
        (
            format!("{prefix}{missing}"),
            false,
            format!("{prefix}{missing}"),
        ),
        (
            format!("looks like `{valid}`"),
            false,
            format!("looks like `{valid}`"),
        ),
        (
            format!("looks like ``{valid}``"),
            false,
            format!("looks like ``{valid}``"),
        ),
        (
            format!("looks like `{valid}"),
            true,
            "looks like `".to_owned(),
        ),
        (
            format!("I'll update `config.py\n{valid}"),
            true,
            "I'll update `config.py\n".to_owned(),
        ),
        (
            format!("Let`s write it.\n{valid}"),
            true,
            "Let`s write it.\n".to_owned(),
        ),
        (
            format!("looks like ``{valid}"),
            true,
            "looks like ``".to_owned(),
        ),
        (
            format!("looks like `{wrong}"),
            false,
            format!("looks like `{wrong}"),
        ),
        (
            format!("looks like `{missing}"),
            false,
            format!("looks like `{missing}"),
        ),
        (
            format!("looks like `{wrong}\n\n"),
            false,
            format!("looks like `{wrong}\n\n"),
        ),
    ];
    *server.backend.tool_format.lock().expect("tool format") = Some(ToolFormat::Qwen);
    for (text, call, content) in &cases {
        for constrained in [false, true] {
            for stream in [false, true] {
                for bytewise in [false, true] {
                    *server.backend.pieces.lock().expect("pieces") = if bytewise {
                        text.chars().map(|c| c.to_string().into_bytes()).collect()
                    } else {
                        vec![text.clone().into_bytes()]
                    };
                    let response = direct_response(&server, &quote_chat(constrained), stream);
                    let parts = responses_parts(&response, stream);
                    let label = format!(
                        "{text:?} constrained={constrained} stream={stream} bytewise={bytewise}"
                    );
                    assert_eq!(parts.calls.len(), usize::from(*call), "{label}");
                    for item in &parts.calls {
                        assert_eq!(
                            item.member_str("arguments", ""),
                            r#"{"text":"42"}"#,
                            "{label}"
                        );
                    }
                    assert_eq!(&parts.content, content, "{label}");
                    if stream {
                        assert_eq!(parts.streamed_content, parts.content, "{label}");
                    }
                }
            }
        }
    }
    server.backend.pieces.lock().expect("pieces").clear();
}

/// A call that implicitly ends the reasoning inside an unclosed quote counts
/// only when schema-valid; otherwise it stays reasoning.
#[test]
fn unfinished_inline_reasoning_fallback() {
    let server = RunningServer::new(HttpServerOptions::default());
    server
        .backend
        .set_initial_output_state(InitialOutputState::Reasoning);
    let prefix = "Check `foo then\n";
    let valid = "<tool_call><function=f><parameter=text>42</parameter></function></tool_call>";
    let wrong = "<tool_call><function=f><parameter=text>43</parameter></function></tool_call>";
    let missing = "<tool_call><function=f></function></tool_call>";
    let cases = [
        (format!("{prefix}{valid}"), true, prefix.to_owned(), ""),
        (
            format!("{prefix}{valid}</think>"),
            true,
            prefix.to_owned(),
            "",
        ),
        (
            format!("{prefix}{wrong}"),
            false,
            format!("{prefix}{wrong}"),
            "",
        ),
        (
            format!("{prefix}{missing}"),
            false,
            format!("{prefix}{missing}"),
            "",
        ),
        (
            format!("{prefix}{wrong}</think>Answer"),
            false,
            format!("{prefix}{wrong}"),
            "Answer",
        ),
        (
            format!("{prefix}{valid}`</think>Answer"),
            false,
            format!("{prefix}{valid}`"),
            "Answer",
        ),
        (
            format!("{prefix}{valid}`</think>{valid}"),
            true,
            format!("{prefix}{valid}`"),
            "",
        ),
    ];
    // The free-form parser's implicit transition before `</think>`:
    // constrained reasoning requires the explicit delimiter.
    *server.backend.tool_format.lock().expect("tool format") = Some(ToolFormat::Qwen);
    for (text, call, reasoning, content) in &cases {
        for stream in [false, true] {
            for bytewise in [false, true] {
                *server.backend.pieces.lock().expect("pieces") = if bytewise {
                    text.chars().map(|c| c.to_string().into_bytes()).collect()
                } else {
                    vec![text.clone().into_bytes()]
                };
                let response = direct_response(&server, &quote_chat(false), stream);
                let parts = responses_parts(&response, stream);
                let label = format!("{text:?} stream={stream} bytewise={bytewise}");
                assert_eq!(parts.calls.len(), usize::from(*call), "{label}");
                assert_eq!(parts.content, *content, "{label}");
                assert_eq!(parts.reasoning.trim(), reasoning.trim(), "{label}");
                if stream {
                    assert_eq!(parts.streamed_content, parts.content, "{label}");
                    assert_eq!(
                        parts.streamed_reasoning.trim(),
                        parts.reasoning.trim(),
                        "{label}"
                    );
                }
            }
        }
    }
    server.backend.pieces.lock().expect("pieces").clear();
}

/// Thinking returns for every display, and adaptive follows the server's
/// default.
#[test]
fn messages_return_thinking_for_every_display() {
    let server = RunningServer::new(HttpServerOptions::default());
    let response_body = |response: &str| {
        expect_status(response, 200);
        json::parse_str(body_of(response)).expect("body")
    };
    server
        .backend
        .set_initial_output_state(InitialOutputState::Auto);
    server
        .backend
        .set_output(b"<think>plan</think>answer".to_vec());
    for server_thinking in [false, true] {
        *server.backend.reasoning.lock().expect("reasoning") = ReasoningOptions {
            enabled: Some(server_thinking),
            effort: Some(ReasoningEffort::Low),
            ..ReasoningOptions::default()
        };
        let message = response_body(&server.post(
            "/v1/messages",
            r#"{"max_tokens":256,"thinking":{"type":"adaptive","display":"omitted"},
                "output_config":{"effort":"xhigh"},
                "messages":[{"role":"user","content":"hi"}]}"#,
        ));
        if server_thinking {
            assert_eq!(
                message.find("content").expect("content").items()[0].member_str("thinking", ""),
                "plan"
            );
        }
        let reasoning = server.backend.last_call().chat.reasoning;
        assert_eq!(reasoning.enabled, Some(server_thinking));
        assert_eq!(reasoning.effort, Some(ReasoningEffort::XHigh));
    }
    *server.backend.reasoning.lock().expect("reasoning") = ReasoningOptions::default();
    let message = response_body(&server.post(
        "/v1/messages",
        r#"{"messages":[{"role":"user","content":"hi"}],
            "thinking":{"type":"enabled","display":"updates"}}"#,
    ));
    assert_eq!(
        message.find("content").expect("content").items()[0].member_str("thinking", ""),
        "plan"
    );
}

/// A rejected image data URL names its problem instead of the generic input
/// error, before any generation.
#[test]
fn responses_name_the_rejected_image() {
    let server = RunningServer::new(HttpServerOptions::default());
    server.backend.image_support.store(true, Ordering::SeqCst);
    let calls = server.backend.calls.load(Ordering::SeqCst);
    for (url, expected) in [
        ("data:image/gif;base64,AA==", "image/gif"),
        ("data:image/png,AA==", "base64"),
        ("data:image/png;base64,A===", "base64"),
    ] {
        let body = format!(
            r#"{{"input":[{{"role":"user","content":[{{"type":"input_image","image_url":{}}},
                {{"type":"input_text","text":"describe"}}]}}]}}"#,
            json::Value::from(url).dump().expect("url")
        );
        let response = server.post("/v1/responses", &body);
        expect_status(&response, 400);
        let error = json::parse_str(body_of(&response)).expect("body");
        let error = error.find("error").expect("error");
        assert_eq!(error.member_str("code", ""), "invalid_request");
        assert!(
            error.member_str("message", "").contains(expected),
            "{url}: {}",
            error.member_str("message", "")
        );
    }
    assert_eq!(server.backend.calls.load(Ordering::SeqCst), calls);
}

/// With thinking off, reasoning tags the client asked for come back as one
/// text block.
#[test]
fn messages_keep_literal_reasoning_tags_without_thinking() {
    let server = RunningServer::new(HttpServerOptions::default());
    server
        .backend
        .set_initial_output_state(InitialOutputState::Content);
    let literal = "<think>literal example</think>";
    server.backend.set_output(literal.as_bytes().to_vec());
    let response = server.post(
        "/v1/messages",
        r#"{"messages":[{"role":"user","content":
            "Copy this XML exactly: <think>literal example</think>"}],
            "thinking":{"type":"disabled"}}"#,
    );
    expect_status(&response, 200);
    let message = json::parse_str(body_of(&response)).expect("body");
    assert_eq!(
        server.backend.last_call().chat.reasoning.enabled,
        Some(false)
    );
    let blocks = message.find("content").expect("content").items();
    assert!(
        blocks.len() == 1
            && blocks[0].member_str("type", "") == "text"
            && blocks[0].member_str("text", "") == literal,
        "disabled thinking keeps literal tags as one text block"
    );
}

/// A text-only model refuses image blocks before any image is read, so a
/// client cannot make the server fetch a URL for a request that cannot
/// succeed.
#[test]
fn text_only_backend_refuses_images_before_reading() {
    let server = RunningServer::new(HttpServerOptions::default());
    server.backend.image_support.store(false, Ordering::SeqCst);
    let calls = server.backend.calls.load(Ordering::SeqCst);
    // An unresolvable host or invalid base64 would produce a different error
    // if the image were fetched or decoded.
    for url in [
        "https://image-fetch-must-not-happen.invalid/x.png",
        "data:image/png;base64,!!!not-base64!!!",
    ] {
        let body = format!(
            r#"{{"input":[{{"role":"user","content":[{{"type":"input_image","image_url":"{url}"}}]}}]}}"#
        );
        let rejected = server.post("/v1/responses", &body);
        expect_status(&rejected, 400);
        assert!(
            rejected.contains("does not support image input"),
            "{rejected}"
        );
    }
    assert_eq!(server.backend.calls.load(Ordering::SeqCst), calls);
}

/// The raw completion routes read `cache_prompt` like Chat Completions and
/// Responses; null keeps the default. A null `stream` leaves streaming off,
/// and any other non-boolean is refused.
#[test]
fn raw_completions_honour_cache_prompt_and_null_stream() {
    let server = RunningServer::new(HttpServerOptions::default());
    for (path, body) in [
        ("/v1/completions", r#"{"prompt":"hello""#),
        ("/completion", r#"{"prompt":"hello""#),
    ] {
        for (value, expected) in [("false", false), ("true", true), ("null", true)] {
            let request = format!(r#"{body},"cache_prompt":{value}}}"#);
            expect_status(&server.post(path, &request), 200);
            assert_eq!(
                server.backend.last_cache_prompt.load(Ordering::SeqCst),
                expected,
                "{path} {value}"
            );
        }
        let calls = server.backend.calls.load(Ordering::SeqCst);
        let rejected = server.post(path, &format!(r#"{body},"cache_prompt":"no"}}"#));
        expect_status(&rejected, 400);
        assert!(
            rejected.contains("'cache_prompt' must be a boolean"),
            "{rejected}"
        );
        assert_eq!(server.backend.calls.load(Ordering::SeqCst), calls);
    }
    let rejected = server.post("/v1/completions", r#"{"prompt":"hello","stream":"true"}"#);
    expect_status(&rejected, 400);
    assert!(
        rejected.contains("'stream' must be a boolean"),
        "{rejected}"
    );
    let buffered = server.post("/v1/completions", r#"{"prompt":"hello","stream":null}"#);
    expect_status(&buffered, 200);
    assert!(!buffered.contains("text/event-stream"), "{buffered}");
}

/// The `data:` payloads of an SSE response on the wire.
fn sse_events(wire: &str) -> Vec<json::Value> {
    let mut events = Vec::new();
    let mut cursor = 0;
    while let Some(found) = wire[cursor..].find("data: ") {
        let begin = cursor + found + 6;
        let end = wire[begin..]
            .find("\n\n")
            .map_or(wire.len(), |end| begin + end);
        if !wire[begin..end].starts_with("[DONE]") {
            events.push(json::parse_str(&wire[begin..end]).expect("event"));
        }
        cursor = end;
    }
    events
}

/// Messages tools map onto the Chat tool path: declarations, tool_choice,
/// tool_use output and replayed tool_use/tool_result history; streamed
/// Messages use Anthropic's SSE events over the same filters.
#[test]
fn messages_tools_and_streaming() {
    let server = RunningServer::new(HttpServerOptions::default());
    let response_body = |response: &str| {
        expect_status(response, 200);
        json::parse_str(body_of(response)).expect("json")
    };
    let weather = r#"[{"name":"get_weather","description":"Get weather","input_schema":{
        "type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#;
    let call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>";
    server.backend.set_output(call.into());
    let called = response_body(&server.post(
        "/v1/messages",
        &format!(
            r#"{{"max_tokens":64,"tool_choice":{{"type":"any"}},
            "messages":[{{"role":"user","content":"weather in Rome?"}}],"tools":{weather}}}"#
        ),
    ));
    assert_eq!(called.member_str("stop_reason", ""), "tool_use");
    let blocks = called.find("content").expect("content").items();
    assert!(
        blocks.len() == 1
            && blocks[0].member_str("type", "") == "tool_use"
            && !blocks[0].member_str("id", "").is_empty()
            && blocks[0].member_str("name", "") == "get_weather"
            && blocks[0]
                .find("input")
                .expect("input")
                .member_str("city", "")
                == "Rome",
        "{called:?}"
    );
    // input_tokens excludes cache reads: together they cover the prompt.
    let usage = called.find("usage").expect("usage");
    assert!(
        usage.member_size("input_tokens", 0) == 2
            && usage.member_size("cache_read_input_tokens", 0) == 8,
        "{usage:?}"
    );
    let declared = server.backend.last_call().chat;
    assert!(
        declared.tools.len() == 1
            && declared.tools[0].name == "get_weather"
            && declared.tools[0].description == "Get weather"
            && declared.tool_choice == ToolChoice::Required
            && declared.constrained_tools
    );
    let simple = r#""messages":[{"role":"user","content":"hi"}]"#;
    response_body(&server.post(
        "/v1/messages",
        &format!(
            r#"{{{simple},"tool_choice":{{"type":"tool","name":"get_weather"}},"tools":{weather}}}"#
        ),
    ));
    let forced = server.backend.last_call().chat;
    assert!(forced.forced_tool_name == "get_weather" && !forced.parallel_tool_calls);
    response_body(&server.post(
        "/v1/messages",
        &format!(
            r#"{{{simple},"tool_choice":{{"type":"auto","disable_parallel_tool_use":true}},
            "tools":{weather}}}"#
        ),
    ));
    let single = server.backend.last_call().chat;
    assert!(single.tool_choice == ToolChoice::Auto && !single.parallel_tool_calls);
    response_body(&server.post(
        "/v1/messages",
        &format!(r#"{{{simple},"tool_choice":{{"type":"none"}},"tools":{weather}}}"#),
    ));
    let none = server.backend.last_call().chat;
    assert!(none.tool_choice == ToolChoice::None && !none.constrained_tools);

    server.backend.set_output(b"It is sunny.".to_vec());
    let answered = response_body(&server.post(
        "/v1/messages",
        &format!(
            r#"{{"tools":{weather},"messages":[
            {{"role":"user","content":"weather in Rome?"}},
            {{"role":"assistant","content":[
              {{"type":"thinking","thinking":"look it up","signature":""}},
              {{"type":"text","text":"Checking."}},
              {{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{{"city":"Rome"}}}}]}},
            {{"role":"user","content":[
              {{"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"}},
              {{"type":"tool_result","tool_use_id":"toolu_2","is_error":true,
               "content":[{{"type":"text","text":"timeout"}}]}},
              {{"type":"text","text":"thanks"}}]}}]}}"#
        ),
    ));
    assert!(
        answered.member_str("stop_reason", "") == "end_turn"
            && answered.find("content").expect("content").items()[0].member_str("text", "")
                == "It is sunny."
    );
    let history = server.backend.last_call().chat.messages;
    assert_eq!(history.len(), 5);
    assert!(
        history[1].role == ChatRole::Assistant
            && history[1].thought == "look it up"
            && history[1].content == "Checking."
            && history[1].tool_calls.len() == 1
            && history[1].tool_calls[0].id == "toolu_1"
            && history[1].tool_calls[0].name == "get_weather"
            && history[1].tool_calls[0].arguments.len() == 1
            && history[1].tool_calls[0].arguments[0].name == "city"
            && history[1].tool_calls[0].arguments[0].value == "Rome"
    );
    assert!(
        history[2].role == ChatRole::Tool
            && history[2].tool_call_id == "toolu_1"
            && history[2].content == "sunny"
    );
    assert!(
        history[3].role == ChatRole::Tool
            && history[3].tool_call_id == "toolu_2"
            && history[3].content == "timeout"
    );
    assert!(history[4].role == ChatRole::User && history[4].content == "thanks");

    let calls = server.backend.calls.load(Ordering::SeqCst);
    for invalid in [
        format!(r#"{{{simple},"tools":{{}}}}"#),
        format!(r#"{{{simple},"tools":[{{"type":"web_search_20250305","name":"web_search"}}]}}"#),
        format!(r#"{{{simple},"tools":[{{"name":"f"}}]}}"#),
        format!(r#"{{{simple},"tool_choice":{{"type":"any"}}}}"#),
        format!(r#"{{{simple},"tool_choice":"auto","tools":{weather}}}"#),
        format!(
            r#"{{{simple},"tool_choice":{{"type":"auto","disable_parallel_tool_use":1}},
            "tools":{weather}}}"#
        ),
        r#"{"messages":[{"role":"assistant","content":[{"type":"tool_use",
            "id":"toolu_1","name":"f","input":"{}"}]}]}"#
            .to_owned(),
        r#"{"messages":[{"role":"assistant","content":[{"type":"tool_use",
            "name":"f","input":{}}]}]}"#
            .to_owned(),
        r#"{"messages":[{"role":"user","content":[{"type":"tool_result","content":"x"}]}]}"#
            .to_owned(),
        r#"{"messages":[{"role":"user","content":[{"type":"tool_result",
            "tool_use_id":"toolu_1","content":[{"type":"image"}]}]}]}"#
            .to_owned(),
    ] {
        expect_status(&server.post("/v1/messages", &invalid), 400);
    }
    assert_eq!(server.backend.calls.load(Ordering::SeqCst), calls);
    // A turn the limit cut reports the cut, as Chat's finish_reason does,
    // even after a complete call.
    server.backend.set_output(call.into());
    let truncated = response_body(&server.post(
        "/v1/messages",
        &format!(
            r#"{{"max_tokens":1,"tool_choice":{{"type":"auto"}},
            "messages":[{{"role":"user","content":"weather in Rome?"}}],"tools":{weather}}}"#
        ),
    ));
    assert_eq!(truncated.member_str("stop_reason", ""), "max_tokens");
    // Without declared tools, call markup stays visible text.
    server.backend.set_output(b"<tool_call>leak".to_vec());
    let undeclared = response_body(&server.post("/v1/messages", &format!("{{{simple}}}")));
    let content = undeclared.find("content").expect("content").items();
    assert!(
        undeclared.member_str("stop_reason", "") == "end_turn"
            && content.len() == 1
            && content[0].member_str("text", "") == "<tool_call>leak"
    );

    // Streamed Messages.
    server
        .backend
        .set_output(b"<think>plan</think>answer".to_vec());
    let streamed = server.post(
        "/v1/messages",
        &format!(
            r#"{{"stream":true,"max_tokens":64,"thinking":{{"type":"enabled","budget_tokens":1024}},
            {simple}}}"#
        ),
    );
    expect_status(&streamed, 200);
    assert!(streamed.contains("text/event-stream") && streamed.contains("event: message_start\n"));
    let events = sse_events(&streamed);
    let types: Vec<String> = events.iter().map(|e| e.member_str("type", "")).collect();
    let (mut thought, mut answer) = (String::new(), String::new());
    for event in &events {
        if let Some(delta) = event.find("delta") {
            match delta.member_str("type", "").as_str() {
                "thinking_delta" => thought += &delta.member_str("thinking", ""),
                "text_delta" => answer += &delta.member_str("text", ""),
                _ => {}
            }
        }
    }
    assert!(thought == "plan" && answer == "answer", "{events:?}");
    assert!(types.first().is_some_and(|t| t == "message_start"));
    assert!(types.last().is_some_and(|t| t == "message_stop"));
    let first_block = events[1].find("content_block").expect("block");
    assert!(
        first_block.member_str("type", "") == "thinking" && events[1].member_size("index", 9) == 0
    );
    let done = &events[events.len() - 2];
    let usage = done.find("usage").expect("usage");
    assert!(
        done.member_str("type", "") == "message_delta"
            && done
                .find("delta")
                .expect("delta")
                .member_str("stop_reason", "")
                == "end_turn"
            && usage.member_size("input_tokens", 0) == 2
            && usage.member_size("cache_read_input_tokens", 0) == 8
    );
    assert_eq!(
        types.iter().filter(|t| *t == "content_block_start").count(),
        2
    );
    assert_eq!(
        types.iter().filter(|t| *t == "content_block_stop").count(),
        2
    );
    // Thinking blocks keep the trimmed reasoning, buffered and streamed.
    server
        .backend
        .set_output(b"<think>\nplan\n\nmore\n</think>\n\nanswer".to_vec());
    for stream in [false, true] {
        let framed = server.post(
            "/v1/messages",
            &format!(
                r#"{{"max_tokens":64,"stream":{stream},
                "thinking":{{"type":"enabled","budget_tokens":1024}},{simple}}}"#
            ),
        );
        expect_status(&framed, 200);
        let thought = if stream {
            sse_events(&framed)
                .iter()
                .filter_map(|event| event.find("delta"))
                .filter(|delta| delta.member_str("type", "") == "thinking_delta")
                .map(|delta| delta.member_str("thinking", ""))
                .collect::<String>()
        } else {
            response_body(&framed)
                .find("content")
                .expect("content")
                .items()[0]
                .member_str("thinking", "")
        };
        assert_eq!(thought, "plan\n\nmore", "stream={stream}");
    }
    server.backend.set_output(call.into());
    let streamed_tool = server.post(
        "/v1/messages",
        &format!(
            r#"{{"stream":true,"tool_choice":{{"type":"any"}},
            "messages":[{{"role":"user","content":"weather in Rome?"}}],"tools":{weather}}}"#
        ),
    );
    expect_status(&streamed_tool, 200);
    assert!(!streamed_tool.contains("<tool_call>"));
    let (mut started, mut partial_json, mut stop_reason) = (false, String::new(), String::new());
    for event in sse_events(&streamed_tool) {
        if let Some(block) = event.find("content_block") {
            started |= block.member_str("type", "") == "tool_use"
                && block.member_str("name", "") == "get_weather";
        }
        if let Some(delta) = event.find("delta") {
            if delta.member_str("type", "") == "input_json_delta" {
                partial_json += &delta.member_str("partial_json", "");
            }
            if event.member_str("type", "") == "message_delta" {
                stop_reason = delta.member_str("stop_reason", "");
            }
        }
    }
    assert!(
        started
            && stop_reason == "tool_use"
            && json::parse_str(&partial_json)
                .expect("arguments")
                .member_str("city", "")
                == "Rome"
    );
    // The fixture fails after the headers.
    server.backend.failure.store(6, Ordering::SeqCst);
    let failed = server.post("/v1/messages", &format!(r#"{{"stream":true,{simple}}}"#));
    expect_status(&failed, 200);
    let events = sse_events(&failed);
    let last = events.last().expect("event");
    assert!(
        last.member_str("type", "") == "error"
            && last.find("error").expect("error").member_str("type", "") == "api_error"
            && !failed.contains("event: message_stop")
    );
    server.backend.failure.store(0, Ordering::SeqCst);
    server.backend.set_output(b"ok".to_vec());
}

/// Messages image blocks in user turns and tool results, and
/// `count_tokens` over the same parsed request.
#[test]
fn messages_images_and_count_tokens() {
    let server = RunningServer::new(HttpServerOptions::default());
    let response_body = |response: &str| {
        expect_status(response, 200);
        json::parse_str(body_of(response)).expect("json")
    };
    server.backend.image_support.store(true, Ordering::SeqCst);
    server.backend.set_output(b"seen".to_vec());
    response_body(&server.post(
        "/v1/messages",
        r#"{"max_tokens":8,"messages":[{"role":"user","content":[
            {"type":"text","text":"look: "},
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQID"}},
            {"type":"text","text":"what is it?"}]}]}"#,
    ));
    let sent = server
        .backend
        .last_call()
        .chat
        .messages
        .last()
        .cloned()
        .expect("message");
    assert!(
        sent.role == ChatRole::User
            && sent.content == "look: what is it?"
            && sent.images.len() == 1
            && sent.images[0].offset == 6
            && *sent.images[0].bytes == [1, 2, 3]
    );
    response_body(&server.post(
        "/v1/messages",
        r#"{"max_tokens":8,"messages":[
            {"role":"user","content":"read it"},
            {"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"read","input":{}}]},
            {"role":"user","content":[
              {"type":"tool_result","tool_use_id":"toolu_1","content":[
                {"type":"text","text":"swatch.png"},
                {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"BAUG"}}]},
              {"type":"text","text":"which color?"}]}]}"#,
    ));
    let sent = server.backend.last_call().chat.messages;
    assert!(
        sent.len() == 4
            && sent[2].role == ChatRole::Tool
            && sent[2].tool_call_id == "toolu_1"
            && sent[2].content == "swatch.png"
            && sent[2].images.len() == 1
            && sent[2].images[0].offset == 10
            && *sent[2].images[0].bytes == [4, 5, 6]
            && sent[3].role == ChatRole::User
            && sent[3].images.is_empty()
            && sent[3].content == "which color?"
    );
    let streamed = server.post(
        "/v1/messages",
        r#"{"stream":true,"max_tokens":8,"messages":[{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQID"}},
            {"type":"text","text":"what is it?"}]}]}"#,
    );
    expect_status(&streamed, 200);
    assert!(streamed.contains("event: message_start\n"));
    let sent = server
        .backend
        .last_call()
        .chat
        .messages
        .last()
        .cloned()
        .expect("message");
    assert!(sent.content == "what is it?" && sent.images.len() == 1 && sent.images[0].offset == 0);
    // Unsupported spellings stay errors that name the problem.
    let calls = server.backend.calls.load(Ordering::SeqCst);
    for (body, expected) in [
        (
            r#"{"messages":[{"role":"user","content":[{"type":"image",
                "source":{"type":"base64","media_type":"image/gif","data":"AA=="}}]}]}"#,
            "image/gif",
        ),
        (
            r#"{"messages":[{"role":"user","content":[{"type":"image",
                "source":{"type":"url","url":"file:///tmp/x.png"}}]}]}"#,
            "HTTPS",
        ),
        (
            r#"{"messages":[{"role":"user","content":[{"type":"image",
                "source":{"type":"file","file_id":"f"}}]}]}"#,
            "base64 or url source",
        ),
        (
            r#"{"messages":[{"role":"assistant","content":[{"type":"image",
                "source":{"type":"base64","media_type":"image/png","data":"AQID"}}]}]}"#,
            "'messages' must contain",
        ),
    ] {
        let rejected = server.post("/v1/messages", body);
        expect_status(&rejected, 400);
        let error = json::parse_str(body_of(&rejected)).expect("json");
        let message = error
            .find("error")
            .expect("error")
            .member_str("message", "");
        assert!(message.contains(expected), "{message}");
    }
    assert_eq!(server.backend.calls.load(Ordering::SeqCst), calls);

    // A backend that can count reports the prompt of the request Messages
    // would generate from, without generating.
    *server.backend.chat_token_count.lock().expect("count") = Some(1234);
    let counted = response_body(&server.post(
        "/v1/messages/count_tokens",
        r#"{"system":"Be brief.","messages":[{"role":"user","content":[
            {"type":"text","text":"what is it?"}]}],
            "tools":[{"name":"look","input_schema":{"type":"object"}}]}"#,
    ));
    assert_eq!(counted.member_size("input_tokens", 0), 1234);
    assert_eq!(*server.backend.counted.lock().expect("counted"), (2, 1));
    assert_eq!(server.backend.calls.load(Ordering::SeqCst), calls);
    for invalid in [
        r#"{"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        r#"{"messages":[]}"#,
        r#"{"messages":[{"role":"user","content":"hi"}],"thinking":true}"#,
    ] {
        expect_status(&server.post("/v1/messages/count_tokens", invalid), 400);
    }
    expect_status(
        &server.post(
            "/v1/messages/count_tokens",
            r#"{"model":"other","messages":[{"role":"user","content":"hi"}]}"#,
        ),
        404,
    );
    *server.backend.chat_token_count.lock().expect("count") = None;
}

/// Responses tool outputs keep their images inside the tool result; custom
/// tool calls replay their free-form input as literal argument data
///.
#[test]
fn responses_tool_images() {
    let server = RunningServer::new(HttpServerOptions::default());
    server.backend.image_support.store(true, Ordering::SeqCst);
    server.backend.set_output(b"red".to_vec());
    for custom in [false, true] {
        for image_only in [false, true] {
            for stream in [false, true] {
                let call = json::parse_str(if custom {
                    r#"{"type":"custom_tool_call","call_id":"picture","name":"read_file",
                    "input":"swatch.png\nliteral <|im_end|>"}"#
                } else {
                    r#"{"type":"function_call","call_id":"picture","name":"read_file",
                    "arguments":"{\"path\":\"swatch.png\"}"}"#
                })
                .expect("call");
                let mut parts = Vec::new();
                if !image_only {
                    parts.push(
                        json::Value::object()
                            .with("type", "input_text")
                            .with("text", "pixels:"),
                    );
                }
                parts.push(
                    json::Value::object()
                        .with("type", "input_image")
                        .with("image_url", "data:image/png;base64,AQID"),
                );
                let output = json::Value::object()
                    .with(
                        "type",
                        if custom {
                            "custom_tool_call_output"
                        } else {
                            "function_call_output"
                        },
                    )
                    .with("call_id", "picture")
                    .with("output", parts);
                let body = json::Value::object()
                    .with(
                        "input",
                        vec![
                            json::parse_str(r#"{"role":"user","content":"read the image"}"#)
                                .expect("user"),
                            call,
                            output,
                        ],
                    )
                    .with("stream", stream);
                let wire = server.post("/v1/responses", &body.dump().expect("dump"));
                expect_status(&wire, 200);
                let messages = server.backend.last_call().chat.messages;
                assert!(
                    messages.len() == 3
                        && messages[1].role == ChatRole::Assistant
                        && messages[1].tool_calls.len() == 1
                        && messages[1].tool_calls[0].id == "picture"
                        && messages[1].tool_calls[0].name == "read_file",
                    "{messages:?}"
                );
                if custom {
                    let input = &messages[1].tool_calls[0].arguments;
                    assert!(
                        input.len() == 1
                            && input[0].name == "input"
                            && input[0].value == "swatch.png\nliteral <|im_end|>"
                            && input[0].is_string
                    );
                }
                let result = &messages[2];
                assert!(
                    result.role == ChatRole::Tool
                        && result.tool_call_id == "picture"
                        && result.images.len() == 1
                        && result.content == if image_only { "" } else { "pixels:" }
                        && result.images[0].offset == if image_only { 0 } else { 7 }
                        && *result.images[0].bytes == [1, 2, 3],
                    "{result:?}"
                );
                if stream {
                    assert!(wire.contains("response.completed"));
                }
            }
        }
    }
    let calls = server.backend.calls.load(Ordering::SeqCst);
    for item in [
        r#"{"type":"custom_tool_call","call_id":"picture","name":"read_file","input":3}"#,
        r#"{"type":"custom_tool_call","name":"read_file","input":"file.png"}"#,
        r#"{"type":"custom_tool_call_output","output":[]}"#,
        r#"{"type":"custom_tool_call_output","call_id":"picture","output":3}"#,
        r#"{"type":"function_call_output","call_id":"picture","output":[{"type":"input_file","file_id":"x"}]}"#,
        r#"{"type":"custom_tool_call_output","call_id":"picture","output":[{"type":"input_image","file_id":"x"}]}"#,
    ] {
        let body = format!(r#"{{"input":[{item}]}}"#);
        expect_status(&server.post("/v1/responses", &body), 400);
    }
    assert_eq!(server.backend.calls.load(Ordering::SeqCst), calls);
}

/// `--reasoning-budget` is the server default on Chat, Responses and
/// Messages; requests override it: Chat and Responses with llama.cpp's
/// `reasoning_budget_tokens` or `thinking_budget_tokens`, Messages with
/// `thinking.budget_tokens` while thinking is enabled. Without it Gufo keeps
/// the server's budget and `--api-compat llama.cpp` takes llama.cpp's 10000.
#[test]
fn reasoning_budgets() {
    let server = RunningServer::new(HttpServerOptions::default());
    let budget = || server.backend.last_call().chat.reasoning.budget_tokens;
    let ok = |path: &str, body: &str| expect_status(&server.post(path, body), 200);
    let hi = r#""messages":[{"role":"user","content":"hi"}]"#;
    ok(
        "/v1/messages",
        &format!(r#"{{{hi},"thinking":{{"type":"enabled","budget_tokens":1024}}}}"#),
    );
    assert_eq!(budget(), Some(1024));
    ok(
        "/v1/messages",
        &format!(r#"{{{hi},"thinking":{{"type":"enabled"}}}}"#),
    );
    assert_eq!(budget(), None, "the server's budget");
    let llama = RunningServer::new(HttpServerOptions {
        api_compat: ApiCompat::LlamaCpp,
        ..HttpServerOptions::default()
    });
    expect_status(
        &llama.post(
            "/v1/messages",
            &format!(r#"{{{hi},"thinking":{{"type":"enabled"}}}}"#),
        ),
        200,
    );
    assert_eq!(
        llama.backend.last_call().chat.reasoning.budget_tokens,
        Some(10_000),
        "llama.cpp's Anthropic default"
    );
    ok(
        "/v1/messages",
        &format!(r#"{{{hi},"thinking":{{"type":"disabled"}}}}"#),
    );
    assert_eq!(budget(), None);

    server
        .backend
        .reasoning
        .lock()
        .expect("reasoning")
        .budget_tokens = Some(50);
    ok("/v1/responses", r#"{"input":"hi"}"#);
    assert_eq!(budget(), Some(50));
    ok(
        "/v1/chat/completions",
        &format!(r#"{{"model":"test",{hi}}}"#),
    );
    assert_eq!(budget(), Some(50));
    ok(
        "/v1/messages",
        &format!(r#"{{{hi},"thinking":{{"type":"enabled","budget_tokens":8}}}}"#),
    );
    assert_eq!(budget(), Some(8));
    // As in llama.cpp, adaptive thinking takes no budget_tokens.
    ok(
        "/v1/messages",
        &format!(r#"{{{hi},"thinking":{{"type":"adaptive","budget_tokens":8}}}}"#),
    );
    assert_eq!(budget(), Some(50));
    ok(
        "/v1/chat/completions",
        &format!(r#"{{"model":"test",{hi},"thinking_budget_tokens":0}}"#),
    );
    assert_eq!(budget(), Some(0), "0 ends reasoning at once");
    ok(
        "/v1/chat/completions",
        &format!(r#"{{"model":"test",{hi},"thinking_budget_tokens":-1}}"#),
    );
    assert_eq!(budget(), Some(50), "-1 keeps the server default");
    ok(
        "/v1/responses",
        r#"{"input":"hi","thinking_budget_tokens":12}"#,
    );
    assert_eq!(budget(), Some(12));
    for path in ["/v1/chat/completions", "/v1/responses"] {
        let input = json::parse_str(if path == "/v1/responses" {
            r#"{"input":"hi"}"#
        } else {
            r#"{"model":"test","messages":[{"role":"user","content":"hi"}]}"#
        })
        .expect("json");
        for field in ["reasoning_budget_tokens", "thinking_budget_tokens"] {
            for (value, expected) in [
                ("-1", 50),
                ("0", 0),
                ("16", 16),
                ("2147483647", 2_147_483_647),
            ] {
                let body = input
                    .clone()
                    .with(field, json::parse_str(value).expect("value"));
                ok(path, &body.dump().expect("dump"));
                assert_eq!(budget(), Some(expected), "{path} {field}={value}");
            }
            for invalid in ["-2", "1.5", "\"8\"", "2147483648", "1e30"] {
                let body = input
                    .clone()
                    .with(field, json::parse_str(invalid).expect("value"));
                let rejected = server.post(path, &body.dump().expect("dump"));
                expect_status(&rejected, 400);
                assert!(rejected.contains(field), "{path} {field}={invalid}");
            }
        }
        let body = input
            .clone()
            .with("reasoning_budget_tokens", 4)
            .with("thinking_budget_tokens", 16);
        ok(path, &body.dump().expect("dump"));
        assert_eq!(budget(), Some(4), "the canonical field takes precedence");
        ok(
            path,
            &body
                .clone()
                .with("reasoning_budget_tokens", -1)
                .dump()
                .expect("dump"),
        );
        assert_eq!(budget(), Some(50));
        ok(
            path,
            &body
                .clone()
                .with("reasoning_budget_tokens", json::Value::Null)
                .dump()
                .expect("dump"),
        );
        assert_eq!(
            budget(),
            Some(16),
            "a null canonical field defers to the old spelling"
        );
    }
    for invalid in ["2147483648", "18446744073709551616", "1e30", "0"] {
        let rejected = server.post(
            "/v1/messages",
            &format!(r#"{{{hi},"thinking":{{"type":"enabled","budget_tokens":{invalid}}}}}"#),
        );
        expect_status(&rejected, 400);
        assert!(rejected.contains("thinking.budget_tokens"), "{invalid}");
    }
    server
        .backend
        .reasoning
        .lock()
        .expect("reasoning")
        .budget_tokens = None;
}
