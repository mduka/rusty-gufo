//! The pinned evaluation cases and their prompts (`dataset.cpp`).

use gufo_core::json::{self, Value};

/// The pinned Antirez DS4 capability questions, compiled into the binary
/// (C++ reads the same file from the source tree or `share/gufo/eval`).
pub const PINNED_SUITE: &[u8] =
    include_bytes!("../../../tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json");

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AnswerKind {
    #[default]
    MultipleChoice,
    Integer,
    LineSpec,
}

impl AnswerKind {
    /// `AnswerKindName`.
    pub fn name(self) -> &'static str {
        match self {
            Self::MultipleChoice => "mcq",
            Self::Integer => "integer",
            Self::LineSpec => "linespec",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvalCase {
    pub ds4_index: usize,
    pub source: String,
    pub id: String,
    pub domain: String,
    pub title: String,
    pub kind: AnswerKind,
    pub question: String,
    pub choices: Vec<String>,
    pub answer: String,
    pub dataset: String,
    pub dataset_url: String,
    pub license: String,
    pub audit_note: String,
    pub source_record_sha256: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvalSuite {
    pub benchmark: String,
    pub source_repository: String,
    pub source_revision: String,
    pub source_path: String,
    pub source_blob: String,
    pub system_prompt: String,
    pub cases: Vec<EvalCase>,
}

/// The member `key` of the JSON type `kind` (`RequireMember`).
fn require<'a>(
    object: &'a Value,
    key: &str,
    kind: fn(&Value) -> bool,
) -> Result<&'a Value, String> {
    object
        .find(key)
        .filter(|value| kind(value))
        .ok_or_else(|| format!("missing or invalid '{key}'"))
}

fn require_string(object: &Value, key: &str) -> Result<String, String> {
    require(object, key, Value::is_string).map(|value| value.str().to_owned())
}

fn parse_kind(value: &str) -> Result<AnswerKind, String> {
    match value {
        "mcq" => Ok(AnswerKind::MultipleChoice),
        "integer" => Ok(AnswerKind::Integer),
        "linespec" => Ok(AnswerKind::LineSpec),
        _ => Err(format!("unknown answer kind '{value}'")),
    }
}

fn parse_case(value: &Value) -> Result<EvalCase, String> {
    if !value.is_object() {
        return Err("case is not an object".into());
    }
    let mut eval_case = EvalCase {
        ds4_index: require(value, "ds4_index", Value::is_number)?.as_size(0),
        source: require_string(value, "source")?,
        id: require_string(value, "id")?,
        domain: require_string(value, "domain")?,
        title: require_string(value, "title")?,
        kind: parse_kind(&require_string(value, "kind")?)?,
        question: require_string(value, "question")?,
        answer: require_string(value, "answer")?,
        dataset: require_string(value, "dataset")?,
        dataset_url: require_string(value, "dataset_url")?,
        license: require_string(value, "license")?,
        audit_note: require_string(value, "audit_note")?,
        source_record_sha256: require_string(value, "source_record_sha256")?,
        ..EvalCase::default()
    };
    for choice in require(value, "choices", Value::is_array)?.items() {
        if !choice.is_string() {
            return Err("case choice is not a string".into());
        }
        eval_case.choices.push(choice.str().to_owned());
    }
    if eval_case.kind == AnswerKind::MultipleChoice && eval_case.choices.is_empty() {
        return Err("multiple-choice case has no choices".into());
    }
    if eval_case.kind != AnswerKind::MultipleChoice && !eval_case.choices.is_empty() {
        return Err("non-MCQ case unexpectedly has choices".into());
    }
    Ok(eval_case)
}

fn parse_suite(bytes: &[u8]) -> Result<EvalSuite, String> {
    let root = json::parse(bytes).map_err(|error| error.message().to_owned())?;
    if !root.is_object() || require_string(&root, "schema")? != "gufo.eval-cases.v2" {
        return Err("unsupported evaluation schema".into());
    }
    let mut suite = EvalSuite {
        benchmark: require_string(&root, "benchmark")?,
        source_repository: require_string(&root, "source_repository")?,
        source_revision: require_string(&root, "source_revision")?,
        source_path: require_string(&root, "source_path")?,
        source_blob: require_string(&root, "source_blob")?,
        system_prompt: require_string(&root, "system_prompt")?,
        cases: Vec::new(),
    };
    for value in require(&root, "cases", Value::is_array)?.items() {
        suite.cases.push(parse_case(value)?);
    }
    if suite.cases.is_empty() {
        return Err("evaluation suite is empty".into());
    }
    Ok(suite)
}

/// `LoadEvalSuite` of the evaluation data `bytes`.
pub fn load_eval_suite(bytes: &[u8]) -> Result<EvalSuite, String> {
    parse_suite(bytes).map_err(|error| format!("invalid evaluation data: {error}"))
}

/// `BuildUserPrompt`: the question, the lettered choices of a
/// multiple-choice case, and the format of the final answer line.
pub fn build_user_prompt(eval_case: &EvalCase) -> String {
    let mut prompt = format!("{}\n", eval_case.question);
    match eval_case.kind {
        AnswerKind::MultipleChoice => {
            prompt.push_str("\nChoices:\n");
            for (index, choice) in eval_case.choices.iter().enumerate() {
                let letter = char::from(b'A'.wrapping_add(index as u8));
                prompt.push_str(&format!("{letter}. {choice}\n"));
            }
            prompt.push_str(
                "\nSolve the question. At the end, write exactly one final line in this format \
                 and do not write anything after it:\nAnswer: <letter>",
            );
        }
        AnswerKind::LineSpec => prompt.push_str(
            "\nAt the end, write exactly one final line in this format and do not write \
             anything after it:\nAnswer: <line number or comma-separated line numbers>",
        ),
        AnswerKind::Integer => prompt.push_str(
            "\nSolve the problem. At the end, write exactly one final line in this format and \
             do not write anything after it:\nAnswer: <integer>",
        ),
    }
    prompt
}
