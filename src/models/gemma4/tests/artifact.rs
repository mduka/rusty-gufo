//! The extension's `artifact_test.cpp` on the real Gemma 4 artifacts:
//! metadata, tensor binding, the template pin and tokenizer parity with the
//! recorded llama.cpp references. `GUFO_GEMMA4_MODEL` names the target
//! (`GUFO_GEMMA4_MTP_MODEL` optionally the drafter); without it the test
//! fails, since a missing model is not a pass.

#![cfg(feature = "models")]
#![allow(clippy::unwrap_used)]

mod common;

use gufo_core::gguf::Gguf;
use gufo_gemma4::chat_template::{render, sha256_hex, validate_gguf_template};
use gufo_gemma4::tokenizer::{TokenId, Tokenizer};
use gufo_gemma4::weights::{DraftWeights, GgufShard, ModelWeights};
use gufo_hip::MappedFile;

fn map(path: &std::path::Path) -> std::sync::Arc<MappedFile> {
    MappedFile::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn from_hex(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

fn ids_sha256(ids: &[TokenId]) -> String {
    let bytes: Vec<u8> = ids
        .iter()
        .flat_map(|&id| (id as u32).to_le_bytes())
        .collect();
    sha256_hex(&bytes)
}

fn check_target(gguf: &Gguf<'_>, w: &ModelWeights) {
    let c = &w.config;
    let experts = c.has_experts();
    if experts {
        assert!(
            c.num_layers == 30 && c.hidden_size == 2816 && c.ffn_size == 2112,
            "26B-A4B dimensions"
        );
        assert!(
            c.num_experts == 128 && c.expert_ffn_size == 704,
            "26B-A4B experts"
        );
    } else {
        assert!(
            c.num_layers == 60 && c.hidden_size == 5376 && c.ffn_size == 21504,
            "31B dimensions"
        );
    }
    assert!(
        c.num_heads == if experts { 16 } else { 32 }
            && c.global_layer_count() == if experts { 5 } else { 10 },
        "attention layout"
    );
    for l in 0..c.num_layers {
        let global = (l + 1) % 6 == 0;
        assert_eq!(c.is_sliding(l), !global, "sliding pattern at layer {l}");
        let kv = match (experts, global) {
            (true, true) => 2,
            (true, false) => 8,
            (false, true) => 4,
            (false, false) => 16,
        };
        assert_eq!(c.kv_heads[l as usize], kv, "KV heads");
        assert_eq!(
            w.layers[l as usize].attn_v.is_empty(),
            global,
            "K=V on global layers"
        );
    }
    assert!(
        c.head_dim_global == 512
            && c.head_dim_sliding == 256
            && c.sliding_window == 1024
            && c.final_logit_softcap == 30.0,
        "attention constants"
    );
    assert!(
        w.vocab_size == 262_144 && w.tied_output(),
        "tied 262144 vocabulary"
    );
    let rotating = w.rope_factors.values.iter().filter(|&&f| f == 1.0).count();
    assert_eq!(rotating, 64, "64 rotating global rope pairs");
    assert!(
        w.layers[0].output_scale > 0.0 && w.layers[0].output_scale < 1.0,
        "layer output scale read"
    );
    validate_gguf_template(gguf).unwrap();
}

fn check_tokenizer(tokenizer: &Tokenizer) {
    assert_eq!(tokenizer.vocab_size(), 262_144, "vocabulary size");
    // Unsloth stores <turn|> as EOS; the QAT files store <eos>, with <turn|>
    // as EOT. Both end generation, which the set below checks.
    assert!(
        tokenizer.bos_token() == 2 && matches!(tokenizer.eos_token(), 106 | 1),
        "BOS/EOS ids"
    );
    assert_eq!(
        tokenizer.end_of_generation_tokens(),
        [1, 50, 106],
        "end-of-generation set {{<eos>, <|tool_response>, <turn|>}}"
    );
    let goldens = common::read_json("tokenizer_goldens.json");
    let mut checked = 0;
    for entry in goldens.find("entries").unwrap().items() {
        let name = entry.member_str("name", "");
        let text = from_hex(&entry.member_str("text_hex", ""));
        let ids = tokenizer.encode(&text, false, true);
        if let Some(expected) = entry.find("llama_cpp_ids") {
            let want: Vec<TokenId> = expected
                .items()
                .iter()
                .map(|id| id.as_double(-1.0) as TokenId)
                .collect();
            assert_eq!(ids, want, "token ids differ from llama.cpp: {name}");
        } else {
            assert!(
                ids.len() == entry.member_size("llama_cpp_count", usize::MAX)
                    && ids_sha256(&ids) == entry.member_str("llama_cpp_ids_sha256", ""),
                "token ids differ from llama.cpp: {name}"
            );
        }
        // Valid text round-trips through decoding with special tokens shown.
        if entry.member_str("hf", "") == "agrees" {
            assert_eq!(
                tokenizer.decode(&ids, true),
                text,
                "decode does not round-trip: {name}"
            );
        }
        checked += 1;
    }
    assert!(checked >= 20, "tokenizer corpus incomplete");
}

/// Rendered golden prompts tokenize to stable ids; the prompt starts with a
/// single BOS emitted by the template.
fn check_rendered_prompts(tokenizer: &Tokenizer) {
    for c in common::load_template_cases() {
        let mut images = Vec::new();
        let rendered = render(&c.messages, &c.tools, &c.options, Some(&mut images))
            .unwrap_or_else(|e| panic!("{}: {e}", c.name));
        let ids = tokenizer.encode(rendered.text.as_bytes(), false, true);
        assert!(
            !ids.is_empty()
                && ids[0] == tokenizer.bos_token()
                && (ids.len() < 2 || ids[1] != tokenizer.bos_token()),
            "{}: exactly one leading BOS",
            c.name
        );
    }
}

fn check_draft(path: &std::path::Path, target: &ModelWeights) {
    let file = map(path);
    let shards = [GgufShard::parse(file.bytes()).unwrap()];
    let d = DraftWeights::bind(&shards, target).expect("drafter");
    assert!(
        d.config.num_layers == 4
            && d.config.hidden_size == 1024
            && d.config.target_hidden_size == target.config.hidden_size,
        "drafter dimensions"
    );
    let last = target.config.num_layers - 1;
    assert!(
        d.config.shared_kv_source(0, &target.config) == last - 1
            && d.config.shared_kv_source(3, &target.config) == last,
        "drafter reads the last sliding and global target layers"
    );
}

#[test]
fn artifacts() {
    let file = map(&common::env_path("GUFO_GEMMA4_MODEL"));
    let shards = [GgufShard::parse(file.bytes()).unwrap()];
    let weights = ModelWeights::bind(&shards).expect("target");
    check_target(&shards[0].gguf, &weights);
    let tokenizer = Tokenizer::from_gguf(&shards[0].gguf).expect("tokenizer");
    check_tokenizer(&tokenizer);
    check_rendered_prompts(&tokenizer);
    match common::optional_env_path("GUFO_GEMMA4_MTP_MODEL") {
        Some(draft) => check_draft(&draft, &weights),
        None => eprintln!("note: GUFO_GEMMA4_MTP_MODEL not set; drafter unchecked"),
    }
}
