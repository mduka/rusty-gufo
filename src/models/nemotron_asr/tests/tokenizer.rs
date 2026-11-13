//! The C++ extension's `tokenizer_test.cpp`: token IDs to transcripts
//! against the reference's decoding of every run in
//! `tests/fixtures/golden.json` (`--features models`: the released
//! `tokenizer.json`).

#![cfg(feature = "models")]
#![allow(clippy::unwrap_used)]

mod support;

use std::sync::Arc;

use gufo_nemotron_asr::tokenizer::{TextDecoder, Tokenizer};

#[test]
fn golden_runs() {
    let tokenizer =
        Arc::new(Tokenizer::load(&support::model_root().join("tokenizer.json")).unwrap());
    assert_eq!(tokenizer.size(), 13089, "13089 tokens");
    assert!(tokenizer.is_special(13087), "<pad> (blank) is special");
    assert_eq!(tokenizer.piece(518), "<de-DE>", "language tag piece");
    let golden = support::golden_json();
    let mut runs = 0;
    for (clip, entry) in golden.find("clips").unwrap().members() {
        for run in entry.find("runs").unwrap().items() {
            let mut text = TextDecoder::new(tokenizer.clone());
            for id in support::ids(run.find("ids").unwrap()) {
                text.push(id);
            }
            let expected = run.find("text").unwrap().str();
            assert_eq!(text.text(), expected, "{clip}");
            let raw = run.find("raw").unwrap().str();
            if let Some(tag) = raw.rfind('<') {
                let end = raw[tag..].find('>').map_or(raw.len(), |end| tag + end);
                assert_eq!(
                    text.language(),
                    Some(&raw[tag + 1..end]),
                    "{clip}: language tag"
                );
            }
            runs += 1;
        }
    }
    assert_eq!(runs, 16, "16 reference runs");
}
