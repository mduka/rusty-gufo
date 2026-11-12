//! The comparison probe of the C++ reference build
//! (`tests/reference/runtime_probe.cpp`, built in the extension as
//! `gemma4_runtime_probe`): evaluates a prompt, dumps float32 logits of
//! every prompt row (raw `[n][vocab]`), optionally continues greedily and
//! reports rates. Its output lines match the C++ probe's for a comparison
//! of the two builds.
//!
//!   gemma4_runtime_probe --model GGUF [--mtp GGUF] (--tokens 1,2 | --text T |
//!       --text-file P | --chat T) [--dump L.bin] [--generate N]
//!       [--context N] [--step] [--draft-tokens N]
//!       [--draft-policy calibrated|confidence|fixed] [--temperature T]
//!       [--seed S] [--top-k K] [--top-p P]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_core::text::{ChatMessage, ChatRole};
use gufo_gemma4::chat_template::{ChatOptions, render};
use gufo_gemma4::draft_policy::{DraftCalibrationScope, DraftPolicy};
use gufo_gemma4::engine::{DEFAULT_DRAFT_TOKENS, Model, ModelOptions};
use gufo_gemma4::tokenizer::TokenId;

fn usage() -> ExitCode {
    eprintln!(
        "usage: gemma4_runtime_probe --model GGUF [--mtp GGUF] (--tokens 1,2 | --text T | \
         --text-file P | --chat T) [--dump L.bin] [--generate N] [--context N] [--step] \
         [--draft-tokens N] [--draft-policy P] [--temperature T] [--seed S]"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_path = String::new();
    let mut mtp = String::new();
    let mut tokens_arg = String::new();
    let mut text = String::new();
    let mut chat = String::new();
    let mut dump = String::new();
    let mut policy = "calibrated".to_string();
    let mut generate = 0usize;
    let mut context = 8192u32;
    let mut draft_tokens = DEFAULT_DRAFT_TOKENS;
    let mut step = false;
    let mut temperature = 0.0f32;
    let mut seed = 1i64;
    let mut top_k = 0i32;
    let mut top_p = 1.0f32;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let mut value = || -> Option<String> {
            i += 1;
            args.get(i).cloned()
        };
        let ok = match arg {
            "--model" => value().map(|v| model_path = v).is_some(),
            "--mtp" => value().map(|v| mtp = v).is_some(),
            "--tokens" => value().map(|v| tokens_arg = v).is_some(),
            "--text" => value().map(|v| text = v).is_some(),
            "--text-file" => value()
                .and_then(|path| std::fs::read(path).ok())
                .map(|bytes| text = String::from_utf8_lossy(&bytes).into_owned())
                .is_some(),
            "--chat" => value().map(|v| chat = v).is_some(),
            "--dump" => value().map(|v| dump = v).is_some(),
            "--generate" => value()
                .and_then(|v| v.parse().ok())
                .map(|v| generate = v)
                .is_some(),
            "--context" => value()
                .and_then(|v| v.parse().ok())
                .map(|v| context = v)
                .is_some(),
            "--step" => {
                step = true;
                true
            }
            "--draft-tokens" => value()
                .and_then(|v| v.parse().ok())
                .map(|v| draft_tokens = v)
                .is_some(),
            "--draft-policy" => value().map(|v| policy = v).is_some(),
            "--temperature" => value()
                .and_then(|v| v.parse().ok())
                .map(|v| temperature = v)
                .is_some(),
            "--seed" => value()
                .and_then(|v| v.parse().ok())
                .map(|v| seed = v)
                .is_some(),
            "--top-k" => value()
                .and_then(|v| v.parse().ok())
                .map(|v| top_k = v)
                .is_some(),
            "--top-p" => value()
                .and_then(|v| v.parse().ok())
                .map(|v| top_p = v)
                .is_some(),
            _ => {
                eprintln!("unknown argument {arg}");
                return ExitCode::from(2);
            }
        };
        if !ok {
            eprintln!("missing value for {arg}");
            return ExitCode::from(2);
        }
        i += 1;
    }
    if model_path.is_empty() {
        return usage();
    }
    let fail = |what: &str, error: String| {
        eprintln!("{what}: {error}");
        ExitCode::FAILURE
    };
    let Ok(draft_policy) = DraftPolicy::parse(&policy) else {
        return usage();
    };
    let options = ModelOptions {
        max_context: context,
        mtp_model_path: (!mtp.is_empty()).then(|| PathBuf::from(&mtp)),
        draft_tokens,
        draft_policy,
        draft_calibration: DraftCalibrationScope::Request,
        ..ModelOptions::default()
    };
    let load_start = Instant::now();
    let model = match Model::load(std::path::Path::new(&model_path), options) {
        Ok(model) => model,
        Err(error) => return fail("load", error),
    };
    println!(
        "loaded {:.2} GB in {:.1} s",
        model.resident_bytes() as f64 / 1e9,
        load_start.elapsed().as_secs_f64()
    );
    let tokens: Vec<TokenId> = if !chat.is_empty() {
        let message = ChatMessage::new(ChatRole::User, chat.clone());
        match render(&[message], &[], &ChatOptions::default(), None) {
            Ok(rendered) => model.tokenize(&rendered.text),
            Err(error) => return fail("render", error),
        }
    } else if !text.is_empty() {
        model.tokenize(&text)
    } else {
        tokens_arg
            .split(',')
            .filter(|t| !t.trim().is_empty())
            .map(|t| t.trim().parse::<TokenId>().unwrap_or(0))
            .collect()
    };
    if tokens.len() <= 1 {
        return fail("prompt", "no prompt".into());
    }
    let vocab = model.vocab_size() as usize;
    let mut session = match model.create_session(0) {
        Ok(session) => session,
        Err(error) => return fail("session", error),
    };
    let mut logits: Vec<f32> = Vec::new();
    let prefill_start = Instant::now();
    if step {
        if let Err(error) = session.sync(&tokens[..1]) {
            return fail("prefill", error);
        }
        match session.logits() {
            Ok(row) => logits.extend_from_slice(row),
            Err(error) => return fail("prefill", error),
        }
        for &token in &tokens[1..] {
            if let Err(error) = session.evaluate(token) {
                return fail("prefill", error);
            }
            match session.logits() {
                Ok(row) => logits.extend_from_slice(row),
                Err(error) => return fail("prefill", error),
            }
        }
    } else if !dump.is_empty() {
        if let Err(error) = session.evaluate_all(&tokens, &mut logits) {
            return fail("prefill", error);
        }
    } else if let Err(error) = session.sync(&tokens) {
        return fail("prefill", error);
    }
    let prefill = prefill_start.elapsed().as_secs_f64();
    println!(
        "prefill {} tokens in {prefill:.3} s ({:.1} tokens/s)",
        tokens.len(),
        tokens.len() as f64 / prefill
    );
    if !dump.is_empty() {
        if logits.len() != tokens.len() * vocab {
            return fail("dump", "logit rows".into());
        }
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        if let Err(error) = std::fs::write(&dump, bytes) {
            return fail("dump", error.to_string());
        }
    }
    if generate > 0 {
        let config = SamplingConfig {
            temperature,
            seed,
            top_k,
            top_p,
            ..SamplingConfig::default()
        };
        let mut sampler = match SamplerState::new(config, &[]) {
            Ok(sampler) => sampler,
            Err(error) => return fail("sampler", error.to_string()),
        };
        let mut generated: Vec<TokenId> = Vec::new();
        let start = Instant::now();
        while generated.len() < generate {
            let result = match session.decode_step(generate - generated.len(), &mut sampler, true) {
                Ok(result) => result,
                Err(error) => return fail("decode", error),
            };
            generated.extend_from_slice(&result.tokens);
            if result.stop || result.tokens.is_empty() {
                break;
            }
        }
        let decode = start.elapsed().as_secs_f64();
        let s = session.statistics();
        if s.cycles != 0 {
            println!(
                "cycles {}, drafted {}, accepted {}, copied {}/{}, siblings {}/{}",
                s.cycles,
                s.drafted,
                s.accepted,
                s.copied_accepted,
                s.copied,
                s.siblings_accepted,
                s.siblings
            );
        }
        println!(
            "generated {} tokens in {decode:.3} s ({:.2} tokens/s)",
            generated.len(),
            generated.len() as f64 / decode
        );
        let ids: Vec<String> = generated.iter().map(TokenId::to_string).collect();
        println!("ids: {}", ids.join(" "));
        println!(
            "text: {}",
            String::from_utf8_lossy(&model.decode(&generated))
        );
    }
    ExitCode::SUCCESS
}
