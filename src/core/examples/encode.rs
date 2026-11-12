//! Encodes NUL-separated texts with the Qwen tokenizer of a GGUF file and
//! prints one line of token IDs per text, like
//! `src/core/tests/reference/icu_tokenizer_check.cpp encode`:
//! cargo run --release -p gufo-core --example encode -- MODEL.gguf TEXTS

use std::path::PathBuf;

use gufo_core::gguf::Gguf;
use gufo_core::qwen_tokenizer::QwenTokenizer;

fn main() -> Result<(), String> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let [model, texts] = args.as_slice() else {
        return Err("usage: encode MODEL.gguf TEXTS".into());
    };
    let (header, len) = gufo_core::gguf::read_header(model)?;
    let tokenizer = QwenTokenizer::from_gguf(&Gguf::parse_prefix(&header, len)?)?;
    let all = std::fs::read_to_string(texts).map_err(|e| e.to_string())?;
    for text in all.split('\0') {
        let ids: Vec<String> = tokenizer.encode(text).iter().map(u32::to_string).collect();
        println!("{} ", ids.join(" "));
    }
    Ok(())
}
