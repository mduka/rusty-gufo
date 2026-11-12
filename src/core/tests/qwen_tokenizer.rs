//! The Qwen3.8 tokenizer against the C++ `QwenTokenizer` on a multilingual
//! corpus: cargo test -p gufo-core --features models --test qwen_tokenizer
//!
//! Needs the Qwen3.8 27B GGUF (`GUFO_QWEN27B_MODEL`, else the pinned file in
//! the Hugging Face cache) and fails without it.

#![cfg(feature = "models")]
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use gufo_core::gguf::{Gguf, read_header};
use gufo_core::qwen_tokenizer::QwenTokenizer;

fn model() -> PathBuf {
    std::env::var_os("GUFO_QWEN27B_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(
                ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/\
                 4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf",
            )
        })
}

#[test]
fn corpus_encodes_like_the_cpp_tokenizer() {
    let texts: Vec<String> =
        serde_json::from_str(include_str!("fixtures/tokenizer_corpus.json")).unwrap();
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen3.8-tokenizer.json")).unwrap();
    let (header, len) = read_header(&model()).unwrap();
    let gguf = Gguf::parse_prefix(&header, len).unwrap();
    let tokenizer = QwenTokenizer::from_gguf(&gguf).unwrap();
    assert_eq!(tokenizer.vocab_size(), 248_320);
    assert_eq!(tokenizer.eos, Some(248_046));
    // `<|endoftext|>` and `<|im_end|>`; 151643 is an ordinary token here.
    let stops = tokenizer.stop_tokens();
    assert!(stops.contains(248_044) && stops.contains(248_046) && !stops.contains(151_643));
    for (index, text) in texts.iter().enumerate() {
        let want: Vec<u32> = expected["ids"][index]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let ids = tokenizer.encode(text);
        assert_eq!(ids, want, "text {index}: {text:?}");
        // Special tokens decode verbatim, so every text round-trips after NFC.
        let decoded = String::from_utf8(tokenizer.decode(&ids)).unwrap();
        assert_eq!(
            unicode_nfc(&decoded),
            unicode_nfc(text),
            "text {index} round trip"
        );
    }
}

fn unicode_nfc(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.nfc().collect()
}
