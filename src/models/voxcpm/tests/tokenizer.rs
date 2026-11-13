//! The C++ extension's `tokenizer_test.cpp`: the released tokenizer against
//! the golden cases (`--features models`).

#![cfg(feature = "models")]
#![allow(clippy::unwrap_used)]

mod config_samples;

use gufo_voxcpm::tokenizer::Tokenizer;

#[test]
fn golden_cases() {
    let tokenizer = Tokenizer::load(&config_samples::model_root().join("tokenizer.json")).unwrap();
    assert_eq!(tokenizer.vocabulary_size(), 73448, "vocabulary size");
    assert_eq!(tokenizer.token_id("<|im_end|>"), Some(73440));
    assert_eq!(tokenizer.token_id("<s>"), Some(1));
    let golden = config_samples::golden("tokenizer_golden.json");
    let cases = golden.find("cases").unwrap().items();
    let mut failures = Vec::new();
    for entry in cases {
        let text = entry.find("text").unwrap().str();
        let expected: Vec<u32> = entry
            .find("ids")
            .unwrap()
            .items()
            .iter()
            .map(|id| id.as_size(0) as u32)
            .collect();
        match tokenizer.encode_to_vec(text) {
            Ok(ids) if ids == expected => {}
            Ok(ids) => failures.push(format!(
                "{text:?}\n  got      {ids:?}\n  expected {expected:?}"
            )),
            Err(error) => failures.push(format!("{text:?}: {error}")),
        }
    }
    let mut ids = Vec::new();
    assert!(
        tokenizer
            .encode_bytes(b"bad \xC3\x28 byte", &mut ids)
            .is_err(),
        "invalid UTF-8 accepted"
    );
    assert_eq!(cases.len(), 21, "golden cases");
    assert!(
        failures.is_empty(),
        "{} of {} golden cases fail:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

/// The tokenizers audio.cpp's GGUFs embed (bfloat16 and Q8_0, when given)
/// equal the directory's on the golden cases.
#[test]
fn gguf_tokenizers() {
    let released = Tokenizer::load(&config_samples::model_root().join("tokenizer.json")).unwrap();
    let golden = config_samples::golden("tokenizer_golden.json");
    for gguf in config_samples::gguf_paths() {
        let files = gufo_voxcpm::model_files::ModelFiles::open(&gguf).unwrap();
        let embedded = Tokenizer::parse(files.tokenizer()).unwrap();
        assert_eq!(embedded.vocabulary_size(), released.vocabulary_size());
        for entry in golden.find("cases").unwrap().items() {
            let text = entry.find("text").unwrap().str();
            assert_eq!(
                embedded.encode_to_vec(text).unwrap(),
                released.encode_to_vec(text).unwrap(),
                "{}: {text:?}",
                gguf.display()
            );
        }
    }
}
