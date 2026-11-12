//! The extension's `tests/reference_probe.cpp`: runs the scalar Gemma 4
//! reference on a prompt and writes its tokens and logits for comparison
//! with `tools/llama_logits` of the extension and the GPU runtime.
//!
//!   reference_probe --model GGUF (--text TEXT | --chat MESSAGE | --tokens-in T.i32)
//!       --tokens-out T.i32 --logits-out L.g4lg [--half-kv | --q8-activations]
//!       [--chunk N] [--trace-out F] [--image-embd G4VE]
//!
//! `--image-embd` places a G4VE file's encoder rows on the prompt's run of
//! image soft tokens (with `--tokens-in`); it needs a single chunk. The
//! logit file is the extension's: magic `G4LG`, version 1, row count and
//! vocabulary as `u32`, the positions, then `[rows][vocab]` float32.

use std::process::ExitCode;
use std::time::Instant;

use gufo_core::text::{ChatMessage, ChatRole};
use gufo_gemma4::chat_template::{ChatOptions, render};
use gufo_gemma4::reference::{Reference, ReferenceImage, Storage};
use gufo_gemma4::tokenizer::{TokenId, Tokenizer};
use gufo_gemma4::vision_prompt::find_image_tokens;
use gufo_gemma4::weights::{GgufShard, ModelWeights};
use gufo_hip::MappedFile;

const USAGE: &str = "usage: reference_probe --model GGUF (--text T | --chat M | --tokens-in F) \
                     --tokens-out F --logits-out F [--half-kv] [--q8-activations] [--chunk N] \
                     [--trace-out F] [--image-embd F]";

fn words(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
}

fn run(args: &[String]) -> Result<(), String> {
    let (mut model, mut text, mut chat, mut tokens_in) =
        (String::new(), String::new(), String::new(), String::new());
    let (mut tokens_out, mut logits_out, mut trace_out, mut image_embd) =
        (String::new(), String::new(), String::new(), String::new());
    let (mut half_kv, mut q8, mut chunk) = (false, false, 0usize);
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let mut value = || -> Result<String, String> {
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| format!("missing value for {arg}"))
        };
        match arg {
            "--model" => model = value()?,
            "--text" => text = value()?,
            "--chat" => chat = value()?,
            "--tokens-in" => tokens_in = value()?,
            "--tokens-out" => tokens_out = value()?,
            "--logits-out" => logits_out = value()?,
            "--half-kv" => half_kv = true,
            "--q8-activations" => q8 = true,
            "--trace-out" => trace_out = value()?,
            "--chunk" => chunk = value()?.parse().map_err(|_| "--chunk takes a count")?,
            "--image-embd" => image_embd = value()?,
            _ => return Err(format!("unknown argument {arg}")),
        }
        i += 1;
    }
    let sources = [&text, &chat, &tokens_in]
        .iter()
        .filter(|s| !s.is_empty())
        .count();
    if model.is_empty() || tokens_out.is_empty() || logits_out.is_empty() || sources != 1 {
        return Err(USAGE.into());
    }
    let file = MappedFile::open(&model).map_err(|e| format!("{model}: {e}"))?;
    let shards = [GgufShard::parse(file.bytes())?];
    let weights = ModelWeights::bind(&shards)?;
    let tokenizer = Tokenizer::from_gguf(&shards[0].gguf)?;

    let tokens: Vec<TokenId> = if !tokens_in.is_empty() {
        let bytes = std::fs::read(&tokens_in).map_err(|e| format!("{tokens_in}: {e}"))?;
        words(&bytes).map(|w| w as TokenId).collect()
    } else if !chat.is_empty() {
        let message = ChatMessage::new(ChatRole::User, chat);
        let rendered = render(&[message], &[], &ChatOptions::default(), None)?;
        tokenizer.encode(rendered.text.as_bytes(), false, true)
    } else {
        tokenizer.encode(text.as_bytes(), true, true)
    };
    let token_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    std::fs::write(&tokens_out, token_bytes).map_err(|e| format!("{tokens_out}: {e}"))?;

    let storage = if q8 {
        Storage::Q8Activations
    } else if half_kv {
        Storage::HalfKv
    } else {
        Storage::Float32
    };
    let mut reference = Reference::new(&weights, &shards, storage);
    let step = if chunk == 0 { tokens.len() } else { chunk };
    if !trace_out.is_empty() && step != tokens.len() {
        return Err("--trace-out needs a single chunk".into());
    }
    let hidden = weights.config.hidden_size as usize;
    let mut image_rows = Vec::new();
    let mut image_span = None;
    if !image_embd.is_empty() {
        if step != tokens.len() {
            return Err("--image-embd needs a single chunk".into());
        }
        let bytes = std::fs::read(&image_embd).map_err(|e| format!("{image_embd}: {e}"))?;
        let header: Vec<u32> = words(bytes.get(..16).ok_or("truncated --image-embd")?).collect();
        if header[0] != 0x4556_3447 || header[3] as usize != hidden {
            return Err("--image-embd is not a G4VE file of the model's width".into());
        }
        let count = header[2] as usize;
        image_rows = bytes
            .get(16..16 + count * hidden * 4)
            .ok_or("truncated --image-embd")?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let soft = find_image_tokens(&tokenizer)
            .map_err(|e| e.to_string())?
            .soft;
        let first = tokens.iter().position(|&t| t == soft);
        let span = first.map(|first| {
            let end = tokens[first..]
                .iter()
                .position(|&t| t != soft)
                .map_or(tokens.len(), |n| first + n);
            (first, end)
        });
        match span {
            Some((first, end)) if end - first == count && !tokens[end..].contains(&soft) => {
                image_span = Some((first as u32, count as u32));
            }
            _ => return Err("--image-embd rows do not match the prompt's one image".into()),
        }
    }
    let images: Vec<ReferenceImage<'_>> = image_span
        .map(|(row, count)| ReferenceImage {
            row,
            count,
            embedding: &image_rows,
        })
        .into_iter()
        .collect();
    reference.set_trace(!trace_out.is_empty());
    let start = Instant::now();
    let mut logits = Vec::new();
    for begin in (0..tokens.len()).step_by(step.max(1)) {
        let end = (begin + step).min(tokens.len());
        let (rows, _) = reference.forward(&tokens[begin..end], &images, true)?;
        logits.extend_from_slice(&rows);
    }
    let seconds = start.elapsed().as_secs_f64();
    let mut out = Vec::with_capacity(16 + tokens.len() * 4 + logits.len() * 4);
    for word in [0x474C_3447u32, 1, tokens.len() as u32, weights.vocab_size] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    for p in 0..tokens.len() as u32 {
        out.extend_from_slice(&p.to_le_bytes());
    }
    for value in &logits {
        out.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&logits_out, out).map_err(|e| format!("{logits_out}: {e}"))?;
    if !trace_out.is_empty() {
        let mut out = Vec::new();
        for word in [
            weights.config.num_layers,
            tokens.len() as u32,
            weights.config.hidden_size,
        ] {
            out.extend_from_slice(&word.to_le_bytes());
        }
        for value in reference.layer_trace() {
            out.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&trace_out, out).map_err(|e| format!("{trace_out}: {e}"))?;
    }
    println!("{} tokens in {seconds} s", tokens.len());
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => {
            println!("PASS");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
