//! The C++ extension's `reference_probe.cpp`: runs the scalar CPU reference
//! over a token sequence (and greedy continuations), prints each position's
//! top tokens and optionally dumps the float32 logits, one vocabulary-wide
//! row per position, in the layout `tools/llama_parity/compare_logits.py`
//! reads.
//!
//! cargo run --release -p gufo-nemotron --example reference_probe -- \
//!   --model MODEL.gguf --tokens 1,2,3 [--generate N] [--dump logits.bin] [--top N]

use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use gufo_hip::MappedFile;
use gufo_nemotron::reference::ReferenceModel;
use gufo_nemotron::weights::{GgufShard, ModelWeights};

/// `std::atoi` of each comma-separated field.
fn parse_tokens(text: &str) -> Vec<i32> {
    if text.is_empty() {
        return Vec::new();
    }
    text.split(',')
        .map(|field| {
            let field = field.trim_start();
            let end = field
                .char_indices()
                .find(|&(i, c)| !(c.is_ascii_digit() || (i == 0 && (c == '-' || c == '+'))))
                .map_or(field.len(), |(i, _)| i);
            field[..end].parse().unwrap_or(0)
        })
        .collect()
}

/// The first maximum, as `std::max_element`.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

fn print_top(logits: &[f32], top: usize) {
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    let max_logit = logits[order[0] as usize];
    let denom: f64 = logits.iter().map(|&v| f64::from(v - max_logit).exp()).sum();
    for &id in &order[..top] {
        let logit = logits[id as usize];
        println!(
            "   {id:6}  logit={logit:9.4}  p={:.5}",
            f64::from(logit - max_logit).exp() / denom
        );
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (mut model, mut tokens_arg, mut dump) = (String::new(), String::new(), String::new());
    let (mut generate, mut top) = (0usize, 5usize);
    let mut i = 1;
    while i < args.len() {
        let arg = args[i].as_str();
        let Some(value) = args.get(i + 1).cloned() else {
            eprintln!("missing value for {arg}");
            return ExitCode::from(2);
        };
        match arg {
            "--model" => model = value,
            "--tokens" => tokens_arg = value,
            "--dump" => dump = value,
            "--generate" => generate = value.parse().unwrap_or(0),
            "--top" => top = value.parse().unwrap_or(5),
            _ => {
                eprintln!("unknown argument {arg}");
                return ExitCode::from(2);
            }
        }
        i += 2;
    }
    let tokens = parse_tokens(&tokens_arg);
    if model.is_empty() || tokens.is_empty() {
        eprintln!(
            "usage: reference_probe --model MODEL.gguf --tokens 1,2,3 [--generate N] \
             [--dump logits.bin] [--top N]"
        );
        return ExitCode::from(2);
    }
    let file = match MappedFile::open(&model) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("open {model}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let shards = match GgufShard::parse(file.bytes()) {
        Ok(shard) => [shard],
        Err(error) => {
            eprintln!("open {model}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let weights = match ModelWeights::bind(&shards) {
        Ok(weights) => weights,
        Err(error) => {
            eprintln!("bind: {error}");
            return ExitCode::FAILURE;
        }
    };
    let vocab = weights.config.vocab_size as usize;
    let mut reference = ReferenceModel::new(&weights, &shards, (tokens.len() + generate) as u32);
    let top = top.min(vocab);
    let mut out = if dump.is_empty() {
        None
    } else {
        match std::fs::File::create(&dump) {
            Ok(file) => Some(std::io::BufWriter::new(file)),
            Err(error) => {
                eprintln!("cannot write {dump}: {error}");
                return ExitCode::FAILURE;
            }
        }
    };
    let mut logits = vec![0.0f32; vocab];
    let mut next = tokens[0];
    for pos in 0..tokens.len() + generate {
        let token = if pos < tokens.len() {
            tokens[pos]
        } else {
            next
        };
        let start = Instant::now();
        if let Err(error) = reference.step(token, &mut logits) {
            eprintln!("step {pos}: {error}");
            return ExitCode::FAILURE;
        }
        let seconds = start.elapsed().as_secs_f64();
        next = argmax(&logits) as i32;
        println!("pos {pos} token {token} ({seconds:.2} s) -> argmax {next}");
        print_top(&logits, top);
        if let Some(out) = out.as_mut() {
            let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
            if let Err(error) = out.write_all(&bytes) {
                eprintln!("cannot write {dump}: {error}");
                return ExitCode::FAILURE;
            }
        }
    }
    if let Some(mut out) = out
        && let Err(error) = out.flush()
    {
        eprintln!("cannot write {dump}: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
