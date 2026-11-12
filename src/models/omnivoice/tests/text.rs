//! The C++ extension's `text_test.cpp`: language and voice-design
//! resolution, weights, durations and the text helpers against the
//! reference's prompt fixture; the prompt tokens need the tokenizer
//! (`--features models`).

#![allow(clippy::unwrap_used)]

mod config_samples;

use gufo_core::json::Value;
use gufo_omnivoice::text::{
    add_punctuation, chunk_text_punctuation, combine_text, estimate_target_frames,
    resolve_instruct, resolve_language, text_weight,
};

fn optional_string(entry: &Value, key: &str) -> Option<String> {
    entry
        .find(key)
        .filter(|value| !value.is_null())
        .map(|value| value.str().to_owned())
}

fn golden() -> Value {
    config_samples::golden("prompt_golden.json")
}

fn report(failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn languages() {
    let golden = golden();
    let mut failures = Vec::new();
    for entry in golden.find("languages").unwrap().items() {
        let input = entry.find("input").unwrap().str();
        let expected = optional_string(entry, "id");
        let got = resolve_language(input);
        if got != expected {
            failures.push(format!(
                "language {input}: got {got:?}, expected {expected:?}"
            ));
        }
    }
    report(failures);
}

#[test]
fn instructs() {
    let golden = golden();
    let mut failures = Vec::new();
    for entry in golden.find("instructs").unwrap().items() {
        let input = entry.find("input").unwrap().str();
        let use_chinese = entry.find("use_zh").unwrap().as_bool(false);
        let got = resolve_instruct(input, use_chinese);
        if entry.find("error").is_some() {
            if got.is_ok() {
                failures.push(format!("instruct {input}: got {got:?}, expected an error"));
            }
            continue;
        }
        let expected = optional_string(entry, "output");
        if got.as_ref() != Ok(&expected) {
            failures.push(format!(
                "instruct {input}: got {got:?}, expected {expected:?}"
            ));
        }
    }
    report(failures);
}

#[test]
fn weights_and_durations() {
    let golden = golden();
    let mut failures = Vec::new();
    for entry in golden.find("weights").unwrap().items() {
        let text = entry.find("text").unwrap().str();
        let expected = entry.find("weight").unwrap().as_double(0.0);
        let got = text_weight(text);
        if (got - expected).abs() > 1e-9 {
            failures.push(format!("weight {text}: got {got}, expected {expected}"));
        }
    }
    for entry in golden.find("durations").unwrap().items() {
        let text = entry.find("text").unwrap().str();
        let reference = optional_string(entry, "ref_text");
        let reference_frames = entry.find("ref_frames").unwrap().as_size(0);
        let speed = entry.find("speed").unwrap().as_double(0.0);
        let expected = entry.find("frames").unwrap().as_size(0);
        let got = estimate_target_frames(text, reference.as_deref(), reference_frames, speed);
        if got != expected {
            failures.push(format!(
                "duration {text} / {reference:?} speed {speed}: got {got}, expected {expected}"
            ));
        }
    }
    report(failures);
}

#[test]
fn text_helpers() {
    let golden = golden();
    let mut failures = Vec::new();
    for entry in golden.find("combined").unwrap().items() {
        let text = entry.find("text").unwrap().str();
        let reference = optional_string(entry, "ref_text");
        let expected = entry.find("combined").unwrap().str();
        let got = combine_text(text, reference.as_deref());
        if got != expected {
            failures.push(format!(
                "combine {text:?}: got {got:?}, expected {expected:?}"
            ));
        }
    }
    for entry in golden.find("punctuation").unwrap().items() {
        let input = entry.find("input").unwrap().str();
        let expected = entry.find("output").unwrap().str();
        let got = add_punctuation(input);
        if got != expected {
            failures.push(format!(
                "punctuation {input:?}: got {got:?}, expected {expected:?}"
            ));
        }
    }
    for entry in golden.find("chunks").unwrap().items() {
        let text = entry.find("text").unwrap().str();
        let expected: Vec<&str> = entry
            .find("chunks")
            .unwrap()
            .items()
            .iter()
            .map(Value::str)
            .collect();
        let got = chunk_text_punctuation(text, entry.find("chunk_len").unwrap().as_size(0), 3);
        if got != expected {
            failures.push(format!(
                "chunks {text:?}: got {got:?}, expected {expected:?}"
            ));
        }
    }
    report(failures);
}

#[cfg(feature = "models")]
#[test]
fn prompts() {
    use gufo_omnivoice::prompt::{PromptText, encode_prompt_text};
    use gufo_omnivoice::tokenizer::Tokenizer;

    let tokenizer = Tokenizer::load(&config_samples::model_root().join("tokenizer.json")).unwrap();
    let golden = golden();
    let mut failures = Vec::new();
    for entry in golden.find("prompts").unwrap().items() {
        let prompt = PromptText {
            text: entry.find("text").unwrap().str().to_owned(),
            reference_text: optional_string(entry, "ref_text"),
            language: optional_string(entry, "lang"),
            instruct: optional_string(entry, "instruct"),
            denoise: entry.find("denoise").unwrap().as_bool(false),
            has_reference: entry.find("ref_frames").unwrap().as_size(0) > 0,
        };
        let expected: Vec<u32> = entry
            .find("text_ids")
            .unwrap()
            .items()
            .iter()
            .map(|id| id.as_size(0) as u32)
            .collect();
        match encode_prompt_text(&tokenizer, &prompt) {
            Ok(ids) if ids == expected => {}
            Ok(ids) => failures.push(format!(
                "prompt {:?}: got {ids:?}, expected {expected:?}",
                prompt.text
            )),
            Err(error) => failures.push(format!("prompt {:?}: {error}", prompt.text)),
        }
    }
    report(failures);
}
