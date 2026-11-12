//! The C++ extension's `tokenizer_test.cpp`: the pre-tokenizer's pieces and
//! the token IDs against Hugging Face's tokenizer.

#![allow(clippy::unwrap_used)]

mod common;

use gufo_nemotron::tokenizer::pre_tokenize;

#[test]
fn pre_tokenizer_pieces_match_hugging_face() {
    let fixture = common::golden("tokenizer_golden.json");
    let cases = fixture.find("cases").unwrap().items();
    assert!(!cases.is_empty(), "fixture has cases");
    for c in cases {
        let text = c.member_str("text", "");
        let pieces = pre_tokenize(text.as_bytes());
        let expected: Vec<(usize, usize)> = c
            .find("pieces")
            .unwrap()
            .items()
            .iter()
            .map(|p| (p.items()[0].as_size(0), p.items()[1].as_size(0)))
            .collect();
        let got: Vec<(usize, usize)> = pieces.iter().map(|r| (r.start, r.end)).collect();
        assert_eq!(got, expected, "pieces of {text:?}");
    }
}

#[cfg(feature = "models")]
#[test]
fn token_ids_match_hugging_face() {
    use gufo_nemotron::tokenizer::Tokenizer;
    use gufo_nemotron::weights::GgufShard;

    let ids = |array: &gufo_core::json::Value| -> Vec<u32> {
        array.items().iter().map(|v| v.as_size(0) as u32).collect()
    };
    let path = common::trunk_path();
    let file = gufo_hip::MappedFile::open(&path).unwrap();
    let shard = GgufShard::parse(file.bytes()).unwrap();
    let tokenizer = Tokenizer::from_gguf(&shard.gguf).unwrap();
    assert_eq!(tokenizer.vocab_size(), 131072, "vocabulary size");
    assert!(
        tokenizer.eos() == 11 && tokenizer.is_stop(11) && tokenizer.is_stop(2),
        "<|im_end|> and </s> stop generation"
    );
    assert!(
        tokenizer.is_control(10) && !tokenizer.is_control(12),
        "<|im_start|> is a control token, <think> a plain added token"
    );
    assert_eq!(
        tokenizer.added_token("<tool_call>"),
        Some(14),
        "<tool_call> id"
    );

    let fixture = common::golden("tokenizer_golden.json");
    for c in fixture.find("cases").unwrap().items() {
        let text = c.member_str("text", "");
        let with_control = tokenizer.encode(&text, true);
        let plain = tokenizer.encode(&text, false);
        assert_eq!(with_control, ids(c.find("ids").unwrap()), "IDs of {text:?}");
        assert_eq!(
            plain,
            ids(c.find("ids_plain").unwrap()),
            "plain IDs of {text:?}"
        );
        assert!(
            tokenizer.decode(&with_control) == text.as_bytes()
                && tokenizer.decode(&plain) == text.as_bytes(),
            "decoding restores {text:?}"
        );
    }
}
