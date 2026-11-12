//! The C++ extension's `tests/runtime_probe.cpp`: runs the runtime over a
//! prompt, dumps float32 logits for every prompt position (the layout
//! `tools/llama_parity/compare_logits.py` reads), optionally continues
//! greedily (or with LiLiCorr rounds), and reports prefill and decode rates.
//! Its output matches the C++ probe's for a comparison of the two builds.
//!
//!   runtime_probe --model MODEL.gguf
//!       (--tokens 1,2,3 | --text TEXT | --text-file PATH) [--dump logits.bin]
//!       [--generate N] [--context N] [--step] [--depth N]
//!       [--drafter DRAFT.gguf [--temperature T]] [--verify-sweep R]
//!
//! --verify-sweep R (with --generate) replays the generated tokens in fresh
//! sessions and times, for blocks of n = 1 to 8 tokens, R rounds of verify
//! followed by commit of one token and of all n, R one-token appends and,
//! with a drafter, R drafts: the cost of a round by its length.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_nemotron::engine::{Model, ModelOptions, Session};
use gufo_nemotron::speculative::{DraftHistory, Frontier, speculative_step};

fn usage() -> ExitCode {
    eprintln!(
        "usage: runtime_probe --model MODEL.gguf (--tokens 1,2,3 | --text TEXT | --text-file PATH) \
         [--dump logits.bin] [--generate N] [--context N] [--step] [--depth N]"
    );
    ExitCode::from(2)
}

/// Milliseconds per call of `step` over `rounds` calls.
fn time_rounds(
    rounds: usize,
    mut step: impl FnMut(usize) -> Result<(), String>,
) -> Result<f64, String> {
    let start = Instant::now();
    for i in 0..rounds {
        step(i)?;
    }
    Ok(1000.0 * start.elapsed().as_secs_f64() / rounds as f64)
}

/// The --verify-sweep table: each row starts from a fresh session over the
/// prompt and walks the generated tokens, so verified blocks are the model's
/// own continuation.
fn verify_sweep(
    model: &Arc<Model>,
    context: u32,
    prompt: &[i32],
    generated: &[i32],
    rounds: usize,
) -> Result<(), String> {
    let vocab = model.config().vocab_size as usize;
    let mut rows = vec![0.0f32; 8 * vocab];
    let mut last = vec![0.0f32; vocab];
    let fresh = |last: &mut [f32]| -> Result<Session, String> {
        let mut session = model.create_session(context)?;
        session.append(prompt, last)?;
        Ok(session)
    };
    println!("verify sweep, {rounds} rounds per row (ms per round)");
    if rounds > generated.len() {
        return Err("too few generated tokens".into());
    }
    {
        let mut session = fresh(&mut last)?;
        let ms = time_rounds(rounds, |i| session.append(&generated[i..=i], &mut last))?;
        println!("append 1: {ms:.2}");
        if model.drafter().is_some() {
            let pos = session.position();
            let ms = time_rounds(rounds, |_| {
                session.draft_block(generated[rounds], pos).map(|_| ())
            })?;
            println!("draft: {ms:.2}");
        }
    }
    for n in 1..=8 {
        if rounds + n > generated.len() || rounds * n > generated.len() {
            return Err("too few generated tokens".into());
        }
        let mut a = fresh(&mut last)?;
        let mut b = fresh(&mut last)?;
        let out = &mut rows[..n * vocab];
        let one = time_rounds(rounds, |i| {
            a.verify(&generated[i..i + n], out)?;
            a.commit(1)
        })?;
        let all = time_rounds(rounds, |i| {
            b.verify(&generated[i * n..(i + 1) * n], out)?;
            b.commit(n as u32)
        })?;
        println!("verify {n}: commit 1 {one:.2}, commit {n} {all:.2}");
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut model_path, mut tokens_arg, mut text, mut dump, mut drafter_path) = (
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
    );
    let mut temperature = 0.0f32;
    let mut generate = 0u32;
    let mut context = 8192u32;
    let mut step = false;
    let mut depth = 0usize;
    let mut sweep = 0usize;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let mut value = || -> Option<String> {
            i += 1;
            args.get(i).cloned()
        };
        let ok = match arg {
            "--model" => value().map(|v| model_path = v).is_some(),
            "--tokens" => value().map(|v| tokens_arg = v).is_some(),
            "--text" => value().map(|v| text = v).is_some(),
            "--text-file" => value()
                .and_then(|path| std::fs::read(path).ok())
                .map(|bytes| text = String::from_utf8_lossy(&bytes).into_owned())
                .is_some(),
            "--dump" => value().map(|v| dump = v).is_some(),
            "--drafter" => value().map(|v| drafter_path = v).is_some(),
            "--temperature" => value()
                .and_then(|v| v.parse().ok())
                .map(|v| temperature = v)
                .is_some(),
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
            "--depth" => value()
                .and_then(|v| v.parse().ok())
                .map(|v| depth = v)
                .is_some(),
            "--verify-sweep" => value()
                .and_then(|v| v.parse().ok())
                .map(|v| sweep = v)
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
    if model_path.is_empty() || (tokens_arg.is_empty() && text.is_empty()) {
        return usage();
    }
    let fail = |what: &str, error: String| {
        eprintln!("{what}: {error}");
        ExitCode::FAILURE
    };
    let load_start = Instant::now();
    let mut model = match Model::load(
        std::path::Path::new(&model_path),
        ModelOptions {
            max_context: context,
        },
    ) {
        Ok(model) => model,
        Err(error) => return fail("load", error),
    };
    println!(
        "loaded {:.2} GB in {:.1} s",
        model.resident_bytes() as f64 / 1e9,
        load_start.elapsed().as_secs_f64()
    );
    if !drafter_path.is_empty()
        && let Err(error) = model.load_drafter(std::path::Path::new(&drafter_path))
    {
        return fail("drafter", error);
    }
    let model = Arc::new(model);
    let mut session = match model.create_session(context) {
        Ok(session) => session,
        Err(error) => return fail("session", error),
    };
    let mut tokens: Vec<i32> = if tokens_arg.is_empty() {
        model
            .tokenizer()
            .encode(&text, true)
            .into_iter()
            .map(|id| id as i32)
            .collect()
    } else {
        tokens_arg
            .split(',')
            .map(|t| t.trim().parse::<i32>().unwrap_or(0))
            .collect()
    };
    let vocab = model.config().vocab_size as usize;
    if depth > 0 {
        if depth >= tokens.len() {
            eprintln!("--depth must leave prompt tokens to prefill");
            return ExitCode::from(2);
        }
        let mut last = vec![0.0f32; vocab];
        let start = Instant::now();
        if let Err(error) = session.append(&tokens[..depth], &mut last) {
            return fail("depth prefill", error);
        }
        let seconds = start.elapsed().as_secs_f64();
        println!(
            "depth prefill {depth} tokens in {seconds:.3} s ({:.1} tokens/s)",
            depth as f64 / seconds
        );
        tokens.drain(..depth);
    }
    let mut logits = vec![
        0.0f32;
        if dump.is_empty() {
            vocab
        } else {
            tokens.len() * vocab
        }
    ];
    let prefill_start = Instant::now();
    if step {
        for (i, token) in tokens.iter().enumerate() {
            let row = if dump.is_empty() { 0 } else { i };
            if let Err(error) = session.append(
                std::slice::from_ref(token),
                &mut logits[row * vocab..(row + 1) * vocab],
            ) {
                return fail("prefill", error);
            }
        }
    } else if let Err(error) = session.append(&tokens, &mut logits) {
        return fail("prefill", error);
    }
    let prefill = prefill_start.elapsed().as_secs_f64();
    println!(
        "prefill {} tokens in {prefill:.3} s ({:.1} tokens/s)",
        tokens.len(),
        tokens.len() as f64 / prefill
    );
    if !dump.is_empty() {
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        if let Err(error) = std::fs::write(&dump, bytes) {
            return fail("dump", error.to_string());
        }
    }
    if generate > 0 {
        let mut frontier = Frontier {
            logits: logits[logits.len() - vocab..].to_vec(),
            ..Frontier::default()
        };
        let mut generated: Vec<i32> = Vec::new();
        let decode_start = Instant::now();
        let sampling = SamplingConfig {
            temperature,
            top_p: if temperature > 0.0 { 0.95 } else { 1.0 },
            seed: 1,
            ..SamplingConfig::default()
        };
        let mut sampler = match SamplerState::new(sampling, &[]) {
            Ok(sampler) => sampler,
            Err(error) => return fail("sampler", error.to_string()),
        };
        let (mut rounds, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
        let mut history = DraftHistory::default();
        let tokenizer = model.tokenizer();
        let is_stop = |t: i32| tokenizer.is_stop(t as u32);
        while model.drafter().is_some() && (generated.len() as u32) < generate {
            let round = match speculative_step(
                &mut session,
                &mut sampler,
                &mut frontier,
                &mut history,
                generate - generated.len() as u32,
                &is_stop,
                false,
            ) {
                Ok(round) => round,
                Err(error) => return fail("decode", error),
            };
            generated.extend_from_slice(&round.tokens);
            rounds += 1;
            drafted += round.drafted as usize;
            accepted += round.accepted as usize;
            if round.stop {
                break;
            }
        }
        if rounds != 0 {
            println!(
                "rounds {rounds}, drafted {drafted}, accepted {accepted} ({:.2} per round)",
                accepted as f64 / rounds as f64
            );
        }
        let mut i = 0;
        while model.drafter().is_none() && i < generate {
            // std::max_element: the first maximum.
            let row = &frontier.logits;
            let next = row
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (j, &v)| {
                    if v > bv { (j, v) } else { (bi, bv) }
                })
                .0 as i32;
            generated.push(next);
            if is_stop(next) {
                break;
            }
            if let Err(error) = session.append(&[next], &mut frontier.logits) {
                return fail("decode", error);
            }
            i += 1;
        }
        let decode = decode_start.elapsed().as_secs_f64();
        println!(
            "generated {} tokens in {decode:.3} s ({:.2} tokens/s)",
            generated.len(),
            generated.len() as f64 / decode
        );
        let ids: Vec<String> = generated.iter().map(i32::to_string).collect();
        println!("ids: {}", ids.join(" "));
        let ids: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
        println!("text: {}", String::from_utf8_lossy(&tokenizer.decode(&ids)));
        if sweep > 0
            && let Err(error) = verify_sweep(&model, context, &tokens, &generated, sweep)
        {
            return fail("sweep", error);
        }
    }
    ExitCode::SUCCESS
}
