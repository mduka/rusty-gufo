//! `gufo eval`, a port of `src/cli/eval/eval.cpp`: the pinned Antirez DS4
//! capability questions against an OpenAI-compatible server, one request
//! after the other, with a sanitized JSON report checkpointed after every
//! case.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use gufo_core::json::{self, Value};
use gufo_eval::dataset::{self, EvalSuite};
use gufo_eval::extract::{self, ExtractionStatus};
use gufo_eval::http::{self, HttpClient, HttpResult};

use crate::REVISION;
use crate::arg_parser::ArgParser;

const MAXIMUM_QUESTIONS: usize = 75;
const MAXIMUM_COMPLETION_TOKENS: usize = 16_000;
const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8080/v1";

struct Options {
    base_url: String,
    questions: usize,
    greedy: bool,
    output: PathBuf,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.into(),
            questions: MAXIMUM_QUESTIONS,
            greedy: false,
            output: PathBuf::new(),
        }
    }
}

#[derive(Default)]
struct ModelInfo {
    id: String,
    owned_by: String,
    artifact_id: String,
    revision: String,
}

struct ParsedCompletion {
    content: String,
    reasoning_content: String,
    finish_reason: String,
    usage: Value,
    timings: Value,
    metrics: Value,
}

fn environment(name: &str) -> String {
    std::env::var_os(name)
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Replaces every occurrence of `needle`, counting them (`ReplaceAll`).
fn replace_all(text: &mut String, needle: &str, replacement: &str, count: &mut usize) {
    if needle.is_empty() {
        return;
    }
    let matches = text.matches(needle).count();
    if matches > 0 {
        *text = text.replace(needle, replacement);
        *count += matches;
    }
}

fn is_word(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// `std::isspace` in the C locale.
fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// The end of `(?:\.\d{1,3}){groups}\b` at `at`: every group but the last
/// is followed by the next group's dot, and the last by a word boundary.
fn dotted_groups(text: &[u8], mut at: usize, groups: usize) -> Option<usize> {
    for group in 0..groups {
        if text.get(at) != Some(&b'.') {
            return None;
        }
        at += 1;
        let digits = text[at..]
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        if !(1..=3).contains(&digits) {
            return None;
        }
        at += digits;
        let last = group + 1 == groups;
        if last && text.get(at).is_some_and(|&byte| is_word(byte)) {
            return None;
        }
    }
    Some(at)
}

/// The private IPv4 patterns of the C++ report, in their order:
/// `\b127(?:\.\d{1,3}){3}\b`, `\b10(?:\.\d{1,3}){3}\b`,
/// `\b192\.168(?:\.\d{1,3}){2}\b` and
/// `\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b`.
#[derive(Clone, Copy)]
enum PrivateRange {
    Loopback,
    Ten,
    OneNineTwo,
    OneSevenTwo,
}

/// An address of `range` at `at` after a word boundary; returns its end.
fn private_address(text: &[u8], at: usize, range: PrivateRange) -> Option<usize> {
    if at > 0 && is_word(text[at - 1]) {
        return None;
    }
    let rest = &text[at..];
    match range {
        PrivateRange::Loopback => rest
            .starts_with(b"127")
            .then(|| dotted_groups(text, at + 3, 3))
            .flatten(),
        PrivateRange::Ten => rest
            .starts_with(b"10")
            .then(|| dotted_groups(text, at + 2, 3))
            .flatten(),
        PrivateRange::OneNineTwo => rest
            .starts_with(b"192.168")
            .then(|| dotted_groups(text, at + 7, 2))
            .flatten(),
        PrivateRange::OneSevenTwo => {
            let second = matches!(
                rest.get(..6)?,
                [b'1', b'7', b'2', b'.', b'1', b'6'..=b'9']
                    | [b'1', b'7', b'2', b'.', b'2', b'0'..=b'9']
                    | [b'1', b'7', b'2', b'.', b'3', b'0' | b'1']
            );
            second.then(|| dotted_groups(text, at + 6, 2)).flatten()
        }
    }
}

/// `prefix[^/\s]+/` at `at`; returns its end.
fn user_directory(text: &[u8], at: usize, prefix: &[u8]) -> Option<usize> {
    if !text[at..].starts_with(prefix) {
        return None;
    }
    let start = at + prefix.len();
    let name = text[start..]
        .iter()
        .take_while(|&&byte| byte != b'/' && !is_space(byte))
        .count();
    (name > 0 && text.get(start + name) == Some(&b'/')).then_some(start + name + 1)
}

/// `std::regex_replace` of one pattern: leftmost matches, scanning on after
/// each; whether anything was replaced.
fn replace_matches(
    text: &mut String,
    matcher: impl Fn(&[u8], usize) -> Option<usize>,
    replacement: &str,
) -> bool {
    let bytes = text.as_bytes();
    let mut output = String::with_capacity(text.len());
    let (mut copied, mut at, mut replaced) = (0, 0, false);
    while at < bytes.len() {
        match matcher(bytes, at) {
            Some(end) => {
                output.push_str(&text[copied..at]);
                output.push_str(replacement);
                (copied, at, replaced) = (end, end, true);
            }
            None => at += 1,
        }
    }
    if replaced {
        output.push_str(&text[copied..]);
        *text = output;
    }
    replaced
}

struct SanitizedText {
    text: String,
    redactions: usize,
}

/// `SanitizeText`: the endpoint, the credential and the home directory,
/// then private addresses and user directories.
fn sanitize_text(text: &str, base_url: &str, bearer_token: &str) -> SanitizedText {
    let mut result = SanitizedText {
        text: text.to_owned(),
        redactions: 0,
    };
    replace_all(
        &mut result.text,
        base_url,
        "<endpoint>",
        &mut result.redactions,
    );
    replace_all(
        &mut result.text,
        bearer_token,
        "<credential>",
        &mut result.redactions,
    );
    replace_all(
        &mut result.text,
        &environment("HOME"),
        "<home>",
        &mut result.redactions,
    );
    // The C++ patterns, applied one after the other: four private address
    // ranges, then `/home/<user>/` and `/Users/<user>/`. Matches start and
    // end at ASCII bytes, so the text stays UTF-8.
    for range in [
        PrivateRange::Loopback,
        PrivateRange::Ten,
        PrivateRange::OneNineTwo,
        PrivateRange::OneSevenTwo,
    ] {
        let matcher = |text: &[u8], at: usize| private_address(text, at, range);
        if replace_matches(&mut result.text, matcher, "<private-address>") {
            result.redactions += 1;
        }
    }
    for (prefix, replacement) in [("/home/", "/home/<user>/"), ("/Users/", "/Users/<user>/")] {
        let matcher = |text: &[u8], at: usize| user_directory(text, at, prefix.as_bytes());
        if replace_matches(&mut result.text, matcher, replacement) {
            result.redactions += 1;
        }
    }
    result
}

/// `CopyNumericTelemetry`: the numbers and booleans of an object and of its
/// objects up to eight levels down, breadth first; nothing else.
fn copy_numeric_telemetry(source: Option<&Value>) -> Value {
    let mut result = Value::object();
    let Some(source) = source.filter(|value| value.is_object()) else {
        return result;
    };
    let mut pending: Vec<(&Value, Vec<String>)> = vec![(source, Vec::new())];
    let mut index = 0;
    while index < pending.len() {
        let (current, path) = pending[index].clone();
        index += 1;
        for (key, value) in current.members() {
            let mut destination = &mut result;
            for component in &path {
                destination = destination.member_mut(component);
            }
            if value.is_number() {
                destination.set(key, value.as_double(0.0));
            } else if value.is_bool() {
                destination.set(key, value.as_bool(false));
            } else if value.is_object() && path.len() < 8 {
                let mut nested = path.clone();
                nested.push(key.clone());
                pending.push((value, nested));
            }
        }
    }
    result
}

fn safe_identity(value: &str, base_url: &str, bearer_token: &str) -> String {
    let mut value = value.as_bytes();
    if value.len() > 512 {
        value = &value[..512];
    }
    // C++ truncates bytes; a split character becomes U+FFFD here.
    sanitize_text(&String::from_utf8_lossy(value), base_url, bearer_token).text
}

fn parse_single_model(response: &HttpResult) -> Result<ModelInfo, String> {
    if !response.transport_ok {
        return Err(format!(
            "model discovery transport failed ({})",
            response.transport_code
        ));
    }
    if !(200..300).contains(&response.status_code) {
        return Err(format!(
            "model discovery returned HTTP {}",
            response.status_code
        ));
    }
    let malformed = |reason: &str| format!("malformed model discovery response: {reason}");
    let root = json::parse(&response.body).map_err(|e| malformed(e.message()))?;
    let data = root
        .find("data")
        .filter(|data| data.is_array())
        .ok_or_else(|| malformed("response has no data array"))?;
    if data.len() != 1 {
        return Err(format!(
            "expected exactly one model from GET /v1/models, found {}; start a single-model server",
            data.len()
        ));
    }
    let model = &data.items()[0];
    let id = model
        .find("id")
        .filter(|id| model.is_object() && id.is_string() && !id.str().is_empty())
        .ok_or_else(|| malformed("sole model has no string id"))?;
    let mut info = ModelInfo {
        id: id.str().to_owned(),
        owned_by: model.member_str("owned_by", ""),
        revision: model.member_str("revision", ""),
        ..ModelInfo::default()
    };
    if info.revision.is_empty() {
        info.revision = model.member_str("model_revision", "");
    }
    if let Some(gufo) = model.find("gufo").filter(|gufo| gufo.is_object()) {
        info.artifact_id = gufo.member_str("artifact_id", "");
        if info.revision.is_empty() {
            info.revision = gufo.member_str("repository_revision", "");
        }
    }
    Ok(info)
}

fn messages_json(system: &str, user: &str) -> Value {
    Value::from(vec![
        Value::object()
            .with("role", "system")
            .with("content", system),
        Value::object().with("role", "user").with("content", user),
    ])
}

fn request_json(model: &ModelInfo, messages: Value, greedy: bool) -> Value {
    let mut request = Value::object()
        .with("model", model.id.as_str())
        .with("messages", messages)
        .with("max_completion_tokens", MAXIMUM_COMPLETION_TOKENS)
        .with("stream", false);
    if greedy {
        request.set("temperature", 0);
    }
    request
}

fn parse_completion(response: &HttpResult) -> Option<ParsedCompletion> {
    let root = json::parse(&response.body).ok()?;
    let choices = root
        .find("choices")
        .filter(|c| c.is_array() && c.len() == 1)?;
    let choice = &choices.items()[0];
    let message = choice.find("message").filter(|m| m.is_object())?;
    let finish_reason = choice.find("finish_reason").filter(|f| f.is_string())?;
    if !choice.is_object() {
        return None;
    }
    let content = message.find("content").filter(|c| c.is_string())?;
    let reasoning_content = match message.find("reasoning_content") {
        Some(reasoning) if reasoning.is_string() => reasoning.str().to_owned(),
        Some(_) => return None,
        None => String::new(),
    };
    Some(ParsedCompletion {
        content: content.str().to_owned(),
        reasoning_content,
        finish_reason: finish_reason.str().to_owned(),
        usage: copy_numeric_telemetry(root.find("usage")),
        timings: copy_numeric_telemetry(root.find("timings")),
        metrics: copy_numeric_telemetry(root.find("metrics")),
    })
}

fn api_error_code(response: &HttpResult) -> String {
    json::parse(&response.body)
        .ok()
        .and_then(|root| {
            root.find("error")
                .filter(|error| error.is_object())
                .map(|error| error.member_str("code", ""))
        })
        .unwrap_or_default()
}

fn is_context_rejection(response: &HttpResult, error_code: &str) -> bool {
    let body = String::from_utf8_lossy(&response.body);
    let combined = format!("{body} {error_code}").to_ascii_lowercase();
    combined.contains("context")
        && (combined.contains("length")
            || combined.contains("window")
            || combined.contains("token"))
}

/// Writes the report through a temporary file and a rename (`WriteReport`).
fn write_report(output: &Path, report: &Value) -> Result<(), String> {
    if output.as_os_str().is_empty() {
        return Err("--output is required".into());
    }
    let mut temporary = output.as_os_str().to_owned();
    temporary.push(format!(".tmp-{}", std::process::id()));
    let temporary = PathBuf::from(temporary);
    let mut file =
        std::fs::File::create(&temporary).map_err(|_| "cannot create output file".to_string())?;
    let written = report.dump().ok().is_some_and(|text| {
        file.write_all(text.as_bytes()).is_ok() && file.write_all(b"\n").is_ok()
    });
    drop(file);
    if !written {
        let _ = std::fs::remove_file(&temporary);
        return Err("failed while writing output file".into());
    }
    if std::fs::rename(&temporary, output).is_err() {
        let _ = std::fs::remove_file(&temporary);
        return Err("cannot finalize output file".into());
    }
    Ok(())
}

fn sanitized_command(options: &Options) -> String {
    let mut command = String::from("gufo eval");
    if options.base_url != DEFAULT_BASE_URL {
        command.push_str(" --base-url <endpoint>");
    }
    if options.questions != MAXIMUM_QUESTIONS {
        command.push_str(&format!(" --questions {}", options.questions));
    }
    if options.greedy {
        command.push_str(" --greedy");
    }
    command.push_str(" --output <output>");
    command
}

fn optional_identity(value: &str, base_url: &str, bearer_token: &str) -> Value {
    if value.is_empty() {
        Value::Null
    } else {
        Value::from(safe_identity(value, base_url, bearer_token))
    }
}

/// "clean", "dirty", or "unknown" for an unknown revision.
fn evaluator_state() -> &'static str {
    if REVISION == "unknown" {
        "unknown"
    } else if REVISION.contains("dirty") {
        "dirty"
    } else {
        "clean"
    }
}

/// Counters of the run, which every checkpoint writes.
#[derive(Default)]
struct Tally {
    passed: usize,
    failed: usize,
    execution_errors: usize,
    completion_tokens: usize,
    length_finishes: usize,
}

fn checkpoint(options: &Options, report: &mut Value, tally: &Tally) -> Result<(), String> {
    let completed = report.find("cases").map_or(0, Value::len);
    let summary = Value::object()
        .with("total_cases", options.questions)
        .with("completed_cases", completed)
        .with("passed", tally.passed)
        .with("failed", tally.failed)
        .with("not_graded", tally.execution_errors)
        .with("execution_errors", tally.execution_errors)
        .with("length_finishes", tally.length_finishes)
        .with("completion_tokens", tally.completion_tokens);
    report.set("summary", summary);
    write_report(&options.output, report)
}

fn run_header(options: &Options, model: &ModelInfo, bearer_token: &str) -> Value {
    let base_url = options.base_url.as_str();
    let server = Value::object()
        .with(
            "label",
            optional_identity(
                &environment("GUFO_EVAL_SERVER_LABEL"),
                base_url,
                bearer_token,
            ),
        )
        .with(
            "commit",
            optional_identity(
                &environment("GUFO_EVAL_SERVER_COMMIT"),
                base_url,
                bearer_token,
            ),
        )
        .with(
            "configuration",
            optional_identity(
                &environment("GUFO_EVAL_SERVER_CONFIG"),
                base_url,
                bearer_token,
            ),
        );
    let model_json = Value::object()
        .with("id", safe_identity(&model.id, base_url, bearer_token))
        .with(
            "owned_by",
            optional_identity(&model.owned_by, base_url, bearer_token),
        )
        .with(
            "artifact_id",
            optional_identity(&model.artifact_id, base_url, bearer_token),
        )
        .with(
            "revision",
            optional_identity(&model.revision, base_url, bearer_token),
        );
    Value::object()
        .with("evaluator_commit", REVISION)
        .with("evaluator_worktree", evaluator_state())
        .with("command", sanitized_command(options))
        .with("questions", options.questions)
        .with("request_order", "sequential")
        .with("independent_requests", true)
        .with("max_completion_tokens", MAXIMUM_COMPLETION_TOKENS)
        .with(
            "request_timeout_ms",
            HttpClient::REQUEST_TIMEOUT.as_millis() as f64,
        )
        .with(
            "temperature",
            if options.greedy {
                Value::from(0)
            } else {
                Value::Null
            },
        )
        .with(
            "temperature_policy",
            if options.greedy {
                "greedy"
            } else {
                "server_default"
            },
        )
        .with("thinking_policy", "server_default")
        .with("server", server)
        .with("model", model_json)
}

/// One case's request, execution, response and grade; the progress line.
fn run_case(
    options: &Options,
    suite: &EvalSuite,
    client: &HttpClient,
    model: &ModelInfo,
    bearer_token: &str,
    index: usize,
    tally: &mut Tally,
) -> (Value, String) {
    let base_url = options.base_url.as_str();
    let eval_case = &suite.cases[index];
    let user_prompt = dataset::build_user_prompt(eval_case);
    let request = request_json(
        model,
        messages_json(&suite.system_prompt, &user_prompt),
        options.greedy,
    );
    let body = request.dump().unwrap_or_default();
    let response = client.post_json("/chat/completions", &body);

    let mut item = Value::object()
        .with("index", index)
        .with("ds4_index", eval_case.ds4_index)
        .with("source", eval_case.source.as_str())
        .with("case_id", eval_case.id.as_str())
        .with("kind", eval_case.kind.name())
        .with(
            "messages",
            messages_json(&suite.system_prompt, &user_prompt),
        )
        .with("expected_answer", eval_case.answer.as_str());

    let mut completion = None;
    let (status, endpoint_error) = if !response.transport_ok {
        tally.execution_errors += 1;
        (
            "transport_error".to_string(),
            Value::from(response.transport_code),
        )
    } else if !(200..300).contains(&response.status_code) {
        let error_code = api_error_code(&response);
        tally.execution_errors += 1;
        let status = if is_context_rejection(&response, &error_code) {
            "context_rejected"
        } else {
            "http_error"
        };
        let code = if error_code.is_empty() {
            Value::from(format!("http_{}", response.status_code))
        } else {
            Value::from(safe_identity(&error_code, base_url, bearer_token))
        };
        (status.to_string(), code)
    } else {
        match parse_completion(&response) {
            None => {
                tally.execution_errors += 1;
                (
                    "malformed_response".to_string(),
                    Value::from("invalid_chat_completion"),
                )
            }
            Some(parsed) => {
                let status = parsed.finish_reason.clone();
                if status == "length" {
                    tally.length_finishes += 1;
                }
                completion = Some(parsed);
                (status, Value::Null)
            }
        }
    };
    let execution = Value::object()
        .with(
            "http_status",
            if response.transport_ok {
                Value::from(i64::from(response.status_code))
            } else {
                Value::Null
            },
        )
        .with("transport_ms", response.elapsed_ms)
        .with("status", status.as_str())
        .with("error_code", endpoint_error);
    item.set("execution", execution);

    let (response_json, grade, verdict) = match completion {
        Some(completion) => {
            let content = sanitize_text(&completion.content, base_url, bearer_token);
            let reasoning = sanitize_text(&completion.reasoning_content, base_url, bearer_token);
            let response_json = Value::object()
                .with("content", content.text.as_str())
                .with(
                    "reasoning_content",
                    if completion.reasoning_content.is_empty() {
                        Value::Null
                    } else {
                        Value::from(reasoning.text.as_str())
                    },
                )
                .with("finish_reason", completion.finish_reason.as_str())
                .with("redactions", content.redactions + reasoning.redactions);
            tally.completion_tokens += completion.usage.member_size("completion_tokens", 0);
            item.set("usage", completion.usage);
            item.set("timings", completion.timings);
            item.set("metrics", completion.metrics);
            let result = extract::grade_answer(eval_case, &completion.content);
            let verdict = if result.passed { "passed" } else { "failed" };
            if result.passed {
                tally.passed += 1;
            } else {
                tally.failed += 1;
            }
            let grade = Value::object()
                .with("extraction_status", result.extraction.status.name())
                .with(
                    "extracted_answer",
                    if result.extraction.status == ExtractionStatus::Found {
                        Value::from(result.extraction.answer)
                    } else {
                        Value::Null
                    },
                )
                .with("verdict", verdict);
            (response_json, grade, Some(verdict))
        }
        None => {
            item.set("usage", Value::object());
            item.set("timings", Value::object());
            item.set("metrics", Value::object());
            let response_json = Value::object()
                .with("content", Value::Null)
                .with("reasoning_content", Value::Null)
                .with("finish_reason", Value::Null)
                .with("redactions", 0);
            let grade = Value::object()
                .with("extraction_status", "not_run")
                .with("extracted_answer", Value::Null)
                .with("verdict", "not_graded");
            (response_json, grade, None)
        }
    };
    item.set("response", response_json);
    item.set("grade", grade);
    let progress = format!(
        "[{}/{}] {}/{}: {}",
        index + 1,
        options.questions,
        eval_case.source,
        eval_case.id,
        match verdict {
            Some(verdict) => format!("{verdict} ({status})"),
            None => status,
        }
    );
    (item, progress)
}

/// `Execute`: exit status 0 when every case ran, 1 on execution errors or
/// failures to start, 2 on invalid options.
fn execute(options: &Options) -> u8 {
    let fail = |status: u8, message: &str| {
        eprintln!("eval: {message}");
        status
    };
    if let Err(error) = http::validate_base_url(&options.base_url) {
        return fail(2, &error);
    }
    let suite = match dataset::load_eval_suite(dataset::PINNED_SUITE) {
        Ok(suite) => suite,
        Err(error) => return fail(1, &error),
    };
    if options.questions == 0 || options.questions > suite.cases.len() {
        return fail(
            2,
            &format!("--questions must be between 1 and {}", suite.cases.len()),
        );
    }
    if options.output.as_os_str().is_empty() {
        return fail(2, "--output is required");
    }
    let mut bearer_token = environment("OPENAI_API_KEY");
    if bearer_token.is_empty() {
        bearer_token = environment("GUFO_EVAL_API_KEY");
    }
    let client = HttpClient::new(
        &options.base_url,
        &bearer_token,
        HttpClient::REQUEST_TIMEOUT,
    );
    let mut model = match parse_single_model(&client.get("/models")) {
        Ok(model) => model,
        Err(error) => return fail(1, &error),
    };
    let model_revision = environment("GUFO_EVAL_MODEL_REVISION");
    let artifact_id = environment("GUFO_EVAL_ARTIFACT_ID");
    if !model_revision.is_empty() {
        model.revision = model_revision;
    }
    if !artifact_id.is_empty() {
        model.artifact_id = artifact_id;
    }

    let source = Value::object()
        .with("repository", suite.source_repository.as_str())
        .with("revision", suite.source_revision.as_str())
        .with("path", suite.source_path.as_str())
        .with("blob", suite.source_blob.as_str());
    let mut report = Value::object()
        .with("schema", "gufo.eval-result.v1")
        .with("benchmark", suite.benchmark.as_str())
        .with("benchmark_source", source)
        .with("run", run_header(options, &model, &bearer_token))
        .with("cases", Value::array())
        .with("summary", Value::object());
    let mut tally = Tally::default();
    if let Err(error) = checkpoint(options, &mut report, &tally) {
        return fail(1, &error);
    }
    for index in 0..options.questions {
        let (item, progress) = run_case(
            options,
            &suite,
            &client,
            &model,
            &bearer_token,
            index,
            &mut tally,
        );
        report.member_mut("cases").push(item);
        if let Err(error) = checkpoint(options, &mut report, &tally) {
            return fail(1, &error);
        }
        println!("{progress}");
    }
    println!(
        "eval: wrote {} cases to {} (passed={}, failed={}, execution_errors={})",
        options.questions,
        quoted_path(&options.output),
        tally.passed,
        tally.failed,
        tally.execution_errors
    );
    u8::from(tally.execution_errors != 0)
}

/// A path as `operator<<` of `std::filesystem::path` prints it: quoted, with
/// quotes and backslashes escaped.
fn quoted_path(path: &Path) -> String {
    let text = path.to_string_lossy();
    let mut quoted = String::with_capacity(text.len() + 2);
    quoted.push('"');
    for character in text.chars() {
        if character == '"' || character == '\\' {
            quoted.push('\\');
        }
        quoted.push(character);
    }
    quoted.push('"');
    quoted
}

fn parser(options: &mut Options) -> ArgParser<'_> {
    let Options {
        base_url,
        questions,
        greedy,
        output,
    } = options;
    let mut parser = ArgParser::new(
        "gufo eval",
        "Run the pinned Antirez DS4 capability questions against an OpenAI-compatible server.",
    );
    parser
        .add_option(
            "",
            "--base-url",
            "URL",
            "OpenAI-compatible API root (default: http://127.0.0.1:8080/v1)",
            "Endpoint",
            base_url,
        )
        .add_option(
            "",
            "--questions",
            "N",
            "Run the first N pinned cases (default: 75)",
            "Evaluation",
            questions,
        )
        .add_flag(
            "",
            "--greedy",
            "Send temperature: 0 (otherwise omit temperature)",
            "Evaluation",
            greedy,
        )
        .add_option(
            "",
            "--output",
            "PATH",
            "Write the sanitized machine-readable JSON result",
            "Output",
            output,
        );
    parser
}

/// `PrintEvalHelp`.
pub fn print_help() {
    parser(&mut Options::default()).print_help();
}

/// `RunEval`.
pub fn run(args: &[String]) -> ExitCode {
    let mut options = Options::default();
    let (parsed, help) = {
        let mut parser = parser(&mut options);
        (parser.parse(args), parser.is_help_requested())
    };
    if let Err(error) = parsed {
        eprintln!("eval: {error}");
        return ExitCode::from(2);
    }
    if help {
        print_help();
        return ExitCode::SUCCESS;
    }
    ExitCode::from(execute(&options))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn redact(text: &str) -> (String, usize) {
        let sanitized = sanitize_text(text, "", "");
        (sanitized.text, sanitized.redactions)
    }

    #[test]
    fn private_addresses_follow_the_cpp_patterns() {
        assert_eq!(
            redact("host 127.0.0.1 and 10.1.2.3:80").0,
            "host <private-address> and <private-address>:80"
        );
        assert_eq!(
            redact("192.168.1.50 172.16.0.1 172.31.255.255").0,
            "<private-address> <private-address> <private-address>"
        );
        // Outside the ranges, a fourth digit, a word character after the
        // address, or a word character before it: unchanged.
        for kept in [
            "172.32.0.1",
            "172.15.0.1",
            "127.0.0.1234",
            "10.0.0.1a",
            "a10.0.0.1",
            "192.168.1",
            "1127.0.0.1",
        ] {
            assert_eq!(redact(kept), (kept.to_string(), 0), "{kept}");
        }
        // A trailing dot is a boundary; every pattern counts once.
        assert_eq!(
            redact("10.0.0.1. 10.2.3.4"),
            ("<private-address>. <private-address>".into(), 1)
        );
    }

    #[test]
    fn user_directories_are_redacted() {
        assert_eq!(
            redact("/home/alice/model.gguf and /Users/bob/x"),
            ("/home/<user>/model.gguf and /Users/<user>/x".into(), 2)
        );
        for kept in ["/home/alice", "/home//x", "/home/a b/"] {
            assert_eq!(redact(kept).1, 0, "{kept}");
        }
    }

    #[test]
    fn telemetry_keeps_numbers_and_booleans_breadth_first() {
        let source = json::parse(
            br#"{"a":1,"s":"x","n":{"b":true,"t":"y","m":{"c":2}},"z":3,"o":{"q":"w"}}"#,
        )
        .unwrap();
        assert_eq!(
            copy_numeric_telemetry(Some(&source)).dump().unwrap(),
            r#"{"a":1,"z":3,"n":{"b":true,"m":{"c":2}},"o":null}"#
        );
    }

    #[test]
    fn context_rejections_need_context_and_a_limit_word() {
        let response = |body: &str| HttpResult {
            transport_ok: true,
            status_code: 400,
            body: body.into(),
            ..HttpResult::default()
        };
        assert!(is_context_rejection(
            &response("maximum context length exceeded"),
            ""
        ));
        assert!(is_context_rejection(&response("{}"), "context_window"));
        assert!(!is_context_rejection(&response("bad context"), ""));
    }
}
