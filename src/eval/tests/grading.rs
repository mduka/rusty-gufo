//! The pinned suite, the answer extraction and the offline regrade: the
//! cases of `tests/models/deepseek_v4_flash/eval_test.cpp` and
//! `tests/quality/eval_regrade_test.cpp`, plus edge cases of the C++
//! extractor.

#![allow(clippy::unwrap_used)]

use gufo_eval::dataset::{self, AnswerKind, EvalCase};
use gufo_eval::extract::{ExtractionStatus, extract_answer, grade_answer};
use gufo_eval::regrade::{EvalCaseTrace, EvalTraceReport, regrade_trace};

fn multiple_choice(answer: &str, choices: usize) -> EvalCase {
    EvalCase {
        kind: AnswerKind::MultipleChoice,
        answer: answer.into(),
        choices: (0..choices)
            .map(|index| char::from(b'A' + index as u8).to_string())
            .collect(),
        ..EvalCase::default()
    }
}

fn case(kind: AnswerKind, answer: &str) -> EvalCase {
    EvalCase {
        kind,
        answer: answer.into(),
        ..EvalCase::default()
    }
}

#[test]
fn pinned_dataset() {
    let suite = dataset::load_eval_suite(dataset::PINNED_SUITE).unwrap();
    assert_eq!(
        suite.source_revision,
        "84cc882352757baf628a1776badf7cc54d584e28"
    );
    assert_eq!(
        suite.source_blob,
        "7aed5d5c5b5cdc74d1b3aa0310161e2b437c1d08"
    );
    assert_eq!(suite.cases.len(), 75);
    for (index, eval_case) in suite.cases.iter().enumerate() {
        assert_eq!(eval_case.ds4_index, index);
        assert!(!eval_case.source_record_sha256.is_empty());
        assert!(!eval_case.license.is_empty());
    }
    let expected = [
        ("recNu3MXkvWUzHZr9", "B"),
        ("001b51d76b4d422988f2c11f104a2c6c", "C"),
        ("aime2025-01", "70"),
        ("recoiTJPGUmzAkief", "C"),
    ];
    for (eval_case, (id, answer)) in suite.cases.iter().zip(expected) {
        assert_eq!(
            (eval_case.id.as_str(), eval_case.answer.as_str()),
            (id, answer)
        );
    }
    let prompt = dataset::build_user_prompt(&suite.cases[2]);
    assert!(prompt.starts_with(
        "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$\n"
    ));
    assert!(prompt.ends_with("Answer: <integer>"));
}

#[test]
fn invalid_data_names_the_problem() {
    let error = |text: &str| dataset::load_eval_suite(text.as_bytes()).unwrap_err();
    assert_eq!(
        error("[]"),
        "invalid evaluation data: unsupported evaluation schema"
    );
    assert_eq!(
        error("{}"),
        "invalid evaluation data: missing or invalid 'schema'"
    );
    assert_eq!(
        error(r#"{"schema":"gufo.eval-cases.v1"}"#),
        "invalid evaluation data: unsupported evaluation schema"
    );
}

#[test]
fn multiple_choice_extraction() {
    let mut eval_case = multiple_choice("F", 10);
    for text in [
        "</think>Answer: F\nThis answer is final; option H is tempting.",
        "</think>Answer: I think it is F",
        "</think>Answer: I'll go with F.",
    ] {
        assert!(grade_answer(&eval_case, text).passed, "{text}");
    }
    eval_case.answer = "D".into();
    for text in [
        "</think>Answer: It is not B, the answer is D",
        "</think>Answer: rules out C, leaving D",
        "</think>Answer: D, not B",
    ] {
        assert!(grade_answer(&eval_case, text).passed, "{text}");
    }
    assert_eq!(
        extract_answer(&eval_case, "</think>Answer: B or D").status,
        ExtractionStatus::Ambiguous
    );
    assert_eq!(
        extract_answer(&eval_case, "No final choice was produced.").status,
        ExtractionStatus::NoAnswer
    );
    // Without an answer line the last candidate wins, and C++ never finds
    // that ambiguous.
    let result = extract_answer(&eval_case, "Maybe B or maybe D");
    assert_eq!(
        (result.status, result.answer.as_str()),
        (ExtractionStatus::Found, "D")
    );
}

#[test]
fn integer_extraction() {
    let mut eval_case = case(AnswerKind::Integer, "293");
    assert!(grade_answer(&eval_case, "</think>Answer: m+n = 256+37 = 293").passed);
    eval_case.answer = "82".into();
    assert!(
        grade_answer(
            &eval_case,
            "</think>Answer: 082\nThe value 2025 is just the year."
        )
        .passed
    );
    assert_eq!(
        extract_answer(&eval_case, "</think>Answer: 81 or 82").status,
        ExtractionStatus::Ambiguous
    );
    assert_eq!(
        extract_answer(&eval_case, "No number given.").status,
        ExtractionStatus::NoAnswer
    );
    // Separators and signs normalize; negative zero is zero.
    eval_case.answer = "1000".into();
    assert!(grade_answer(&eval_case, "Answer: $1,000").passed);
    eval_case.answer = "0".into();
    assert!(grade_answer(&eval_case, "Answer: -0").passed);
}

#[test]
fn line_spec_extraction() {
    let eval_case = case(AnswerKind::LineSpec, "9-10");
    assert!(grade_answer(&eval_case, "Answer: line 10").passed);
    assert!(grade_answer(&eval_case, "Answer: lines 9-10").passed);
    assert!(!grade_answer(&eval_case, "Answer: lines 9-11").passed);
    assert_eq!(
        extract_answer(&eval_case, "No localization.").status,
        ExtractionStatus::NoAnswer
    );
    // Ranges in either order; an overflowing run yields its longest
    // representable suffix, as C++ retries `std::from_chars` byte by byte,
    // and lines from 100000 on are dropped.
    assert_eq!(
        extract_answer(&eval_case, "Answer: 12 - 10, 3").answer,
        "3,10,11,12"
    );
    assert_eq!(
        extract_answer(&eval_case, "Answer: 30000000001").answer,
        "1"
    );
    assert_eq!(
        extract_answer(&eval_case, "Answer: 123456").status,
        ExtractionStatus::NoAnswer
    );
}

#[test]
fn regrade_counts_passed_and_matching_cases() {
    let trace = |expected: &str, extracted: &str, passed: bool| EvalCaseTrace {
        expected_answer: expected.into(),
        extracted_answer: extracted.into(),
        passed,
        ..EvalCaseTrace::default()
    };
    let all = EvalTraceReport {
        model_id: "qwen35-4b".into(),
        cases: vec![trace("Paris", "Paris", true), trace("Tokyo", "Tokyo", true)],
        ..EvalTraceReport::default()
    };
    let summary = regrade_trace(&all);
    assert_eq!(
        (
            summary.total_cases,
            summary.passed_cases,
            summary.failed_cases
        ),
        (2, 2, 0)
    );
    assert_eq!(summary.pass_rate, 1.0);
    let partial = EvalTraceReport {
        cases: vec![
            trace("Paris", "London", false),
            trace("Tokyo", "Tokyo", true),
        ],
        ..EvalTraceReport::default()
    };
    let summary = regrade_trace(&partial);
    assert_eq!((summary.passed_cases, summary.failed_cases), (1, 1));
    assert_eq!(summary.pass_rate, 0.5);
}
