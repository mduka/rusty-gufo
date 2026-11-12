//! The C++ extension's `tokenizer_test.cpp`: the released tokenizer against
//! the transformers golden cases (`--features models`).

#![cfg(feature = "models")]
#![allow(clippy::unwrap_used)]

mod config_samples;

use gufo_omnivoice::tokenizer::Tokenizer;

#[test]
fn golden_cases() {
    let tokenizer = Tokenizer::load(&config_samples::model_root().join("tokenizer.json")).unwrap();
    assert_eq!(tokenizer.vocabulary_size(), 151676, "vocabulary size");
    assert_eq!(tokenizer.added_token_id("<|text_start|>"), Some(151674));
    assert_eq!(tokenizer.added_token_id("<|denoise|>"), Some(151669));
    let golden = config_samples::golden("tokenizer_golden.json");
    let mut failures = Vec::new();
    let cases = golden.find("cases").unwrap().items();
    for entry in cases {
        let text = entry.find("text").unwrap().str();
        let expected: Vec<u32> = entry
            .find("ids")
            .unwrap()
            .items()
            .iter()
            .map(|id| id.as_size(0) as u32)
            .collect();
        match tokenizer.encode(text) {
            Ok(ids) if ids == expected => {}
            Ok(ids) => failures.push(format!(
                "{text:?}\n  got      {ids:?}\n  expected {expected:?}"
            )),
            Err(error) => failures.push(format!("{text:?}: {error}")),
        }
    }
    assert!(
        tokenizer.encode(b"bad \xC3\x28 byte").is_err(),
        "invalid UTF-8 accepted"
    );
    assert!(
        failures.is_empty(),
        "{} of {} golden cases fail:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}
