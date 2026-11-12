//! The tokenizer against the C++ engine (`tests/fixtures/tokenizer.json`,
//! recorded by `tests/reference/ds4_ref_dump.cpp tokens`): token IDs of plain text,
//! of text with special tokens and of the chat prompt for 181 texts (code,
//! numbers, CJK, emoji, invalid UTF-8, an embedded NUL, the 75 evaluation
//! questions and choices, a long-context prompt), and the decoded text of
//! every vocabulary entry. Needs the model file, not the device.

#![cfg(feature = "models")]
#![allow(clippy::unwrap_used)]

mod common;

use gufo_core::json;
use gufo_deepseek_v4_flash::gguf::Model;
use gufo_deepseek_v4_flash::tokenizer::Vocab;

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn token_ids_and_texts_match_the_cpp_engine() {
    let fixture = json::parse(include_bytes!("fixtures/tokenizer.json")).unwrap();
    let model = Model::open(&common::model_path()).unwrap();
    let vocab = Vocab::load(&model).unwrap();
    let digest = |tokens: &[i32]| (tokens.len(), common::fnv(&common::token_bytes(tokens)));
    let expected = |case: &json::Value, key: &str| {
        let pair = case.find(key).unwrap().items();
        (pair[0].as_size(0), pair[1].str().to_owned())
    };
    for (index, case) in fixture.find("cases").unwrap().items().iter().enumerate() {
        let text = match case.find("text") {
            Some(text) => text.str().as_bytes().to_vec(),
            None => hex(case.find("hex").unwrap().str()),
        };
        let mut tokens = Vec::new();
        vocab.tokenize(&text, &mut tokens);
        assert_eq!(digest(&tokens), expected(case, "tokens"), "text {index}");
        tokens.clear();
        vocab.tokenize_rendered_chat(&text, &mut tokens);
        assert_eq!(
            digest(&tokens),
            expected(case, "rendered"),
            "rendered {index}"
        );
        tokens.clear();
        vocab.encode_chat(b"", &text, &mut tokens);
        assert_eq!(digest(&tokens), expected(case, "chat"), "chat {index}");
    }
    let mut decoded = Vec::new();
    for token in 0..vocab.len() as i32 {
        let bytes = vocab.token_text(token);
        decoded.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        decoded.extend_from_slice(&bytes);
    }
    assert_eq!(common::fnv(&decoded), fixture.member_str("token_text", ""));
}

/// The metadata, the tensor layout and the hash tables of the target, and
/// the DSpark support model, pass the loader's checks.
#[test]
fn the_artifacts_pass_the_loader_checks() {
    use gufo_deepseek_v4_flash::config;
    use gufo_deepseek_v4_flash::dspark_model::DsparkModel;
    use gufo_deepseek_v4_flash::weights::Weights;
    let model = Model::open(&common::model_path()).unwrap();
    config::validate(&model).unwrap();
    let weights = Weights::bind(&model).unwrap();
    assert_eq!(weights.layers.len(), 43);
    assert!(weights.layers[2].indexer_proj.is_some() && weights.layers[3].indexer_proj.is_none());
    assert!(
        weights.layers[3].attn_compressor_kv.is_some()
            && weights.layers[1].attn_compressor_kv.is_none()
    );
    let support = DsparkModel::bind(Model::open(&common::dspark_path()).unwrap()).unwrap();
    assert!(support.n_stages() >= 1 && support.block_size >= 2);
    assert!(support.target_layer_ids.windows(2).all(|w| w[0] < w[1]));
}
