//! Long-prompt recall on the Qwen3.5 4B: cargo test --profile gpu-test -p
//! gufo-qwen35 --features gpu --test recall
//!
//! A fact at the start of a chat prompt of about 23,000 tokens, its question
//! at the end: the greedy answer must name it. Past 16,128 positions Gufo's
//! C++ build has no output to compare with, so this checks that the prefill
//! attention (WMMA for the 16/4 heads, five prompt chunks of up to 4096 rows)
//! still attends across the whole context. Needs
//! `unsloth/Qwen3.5-4B-GGUF` `Qwen3.5-4B-Q4_K_M.gguf` at its pinned revision
//! in the Hugging Face cache and fails without it.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use gufo_core::gguf::{Gguf, read_header};
use gufo_core::qwen_tokenizer::QwenTokenizer;
use gufo_qwen35::Engine;

fn model() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots")
        .join("e87f176479d0855a907a41277aca2f8ee7a09523")
        .join("Qwen3.5-4B-Q4_K_M.gguf")
}

/// Distinct, plausible filler sentences: no sentence repeats.
fn filler(sentences: usize) -> String {
    const PLACES: [&str; 7] = [
        "harbor", "orchard", "library", "foundry", "market", "valley", "station",
    ];
    const THINGS: [&str; 6] = [
        "crates of apples",
        "spools of copper wire",
        "bolts of linen",
        "barrels of oil",
        "sacks of grain",
        "boxes of candles",
    ];
    const PEOPLE: [&str; 5] = [
        "the clerk",
        "a courier",
        "the foreman",
        "an inspector",
        "the night guard",
    ];
    (0..sentences)
        .map(|i| {
            format!(
                "On day {} {} at the {} counted {} {} and wrote the total into ledger {}. ",
                i + 1,
                PEOPLE[i % PEOPLE.len()],
                PLACES[(i / 5) % PLACES.len()],
                (i * 37) % 900 + 17,
                THINGS[(i / 3) % THINGS.len()],
                (i * 13) % 97 + 1,
            )
        })
        .collect()
}

#[test]
fn a_fact_at_the_start_of_23k_tokens_is_recalled() {
    let path = model();
    let (header, len) = read_header(&path).unwrap();
    let tokenizer = QwenTokenizer::from_gguf(&Gguf::parse_prefix(&header, len).unwrap()).unwrap();
    let text = format!(
        "Remember this fact: the access code of the blue vault is 7319-KESTREL.\n\n{}\n\nWhat is \
         the access code of the blue vault? Answer with the code only.",
        filler(773)
    );
    let prompt = tokenizer.encode(&format!(
        "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ));
    assert!(
        (22_000..24_500).contains(&prompt.len()),
        "prompt of {} tokens",
        prompt.len()
    );
    let mut engine = Engine::load(&path, prompt.len() as u32 + 64).unwrap();
    let answer = engine.generate(&prompt, 16, |_| true).unwrap();
    let answer = String::from_utf8_lossy(&tokenizer.decode(&answer)).into_owned();
    eprintln!("prompt of {} tokens, answer {answer:?}", prompt.len());
    assert!(answer.contains("7319-KESTREL"), "answer {answer:?}");
}
