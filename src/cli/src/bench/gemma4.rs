//! `gufo bench` for Gemma 4 (the extension's `RunGemma4Bench`): llama-bench
//! style prefill (pp) of each prompt length after each cached depth, and
//! generation (tg) of each length, on one session; with the
//! `gemma4-assistant` drafter tg runs MTP cycles. Concurrent and HTTP
//! measurements go through the serving benchmark.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use gufo_core::sampling::{SamplerState, TokenId as SamplerToken};
use gufo_gemma4::draft_policy::{DraftCalibrationScope, DraftPolicy};
use gufo_gemma4::engine::{DEFAULT_DRAFT_TOKENS, Model, ModelOptions};
use gufo_gemma4::provider_options::PrefillWeights;
use gufo_gemma4::tokenizer::TokenId;

use super::{CliOptions, left, right, stats, test_name};

/// Whether the model is a Gemma 4 target GGUF.
pub(super) fn supports(o: &CliOptions) -> bool {
    gufo_gemma4::runner::supports(Path::new(&*o.model_path.borrow()))
}

pub(super) fn run(o: &CliOptions) -> ExitCode {
    match bench(o, Instant::now()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("Error: {message}");
            ExitCode::from(1)
        }
    }
}

fn load_time(start: Instant, success: bool) {
    let text = format!(
        "[Model Load]: {} s",
        super::cpp_double(start.elapsed().as_secs_f64())
    );
    if success {
        println!("{text}");
    } else {
        eprintln!("{text} (failed)");
    }
}

/// The context every benchmark session needs: a warm-up's 4,096 tokens,
/// the deepest prefill, and generation after its depth (at least 16 tokens).
fn required_context(o: &CliOptions) -> usize {
    fn largest(values: &[usize]) -> usize {
        values.iter().copied().max().unwrap_or(0)
    }
    let depth = largest(&o.n_depths);
    4096.max(depth + largest(&o.n_prompts) + 1)
        .max(depth.max(16) + largest(&o.n_gens) + 1)
}

/// The engine options of `o` with Gemma's defaults for whatever was not
/// given.
fn model_options(o: &CliOptions, context: u32) -> Result<ModelOptions, String> {
    let mtp = o.speculative_backend == "mtp";
    if !o.speculative_backend.is_empty() && !mtp {
        return Err("Gemma 4 supports only --speculative mtp or off".into());
    }
    if mtp && o.mtp_model_path.is_empty() {
        return Err("--speculative mtp requires --mtp-model".into());
    }
    Ok(ModelOptions {
        max_context: context,
        mtp_model_path: mtp.then(|| o.mtp_model_path.clone().into()),
        draft_tokens: if o.draft_tokens_supplied {
            o.draft_tokens
        } else {
            DEFAULT_DRAFT_TOKENS
        },
        min_draft_tokens: o.min_draft_tokens,
        draft_policy: DraftPolicy::parse(&o.draft_policy)?,
        draft_calibration: DraftCalibrationScope::parse(&o.draft_calibration)?,
        draft_trace_path: (!o.draft_trace.is_empty()).then(|| o.draft_trace.clone().into()),
        prefill_weights: PrefillWeights::parse(&o.prefill_weights)?,
        ..ModelOptions::default()
    })
}

fn bench(o: &CliOptions, load_start: Instant) -> Result<(), String> {
    if o.concurrency != [1] || o.validate_prefill_tokens != 0 {
        return Err(
            "Gemma 4 bench supports C1; use the serving benchmark for concurrent requests".into(),
        );
    }
    let required = required_context(o);
    let context =
        u32::try_from(required).map_err(|_| "Gemma 4 benchmark context is out of range")?;
    let options = match model_options(o, context) {
        Ok(options) => options,
        Err(error) => {
            load_time(load_start, false);
            return Err(error);
        }
    };
    let model_path = o.model_path.borrow().clone();
    let model = match Model::load(Path::new(&model_path), options) {
        Ok(model) => model,
        Err(error) => {
            load_time(load_start, false);
            return Err(format!("creating the Gemma 4 model failed: {error}"));
        }
    };
    load_time(load_start, true);
    // A deterministic sequence: BOS, then the pattern repeated.
    let pattern = model.tokenize(
        "The quick brown fox jumps over the lazy dog. \
         Strix Halo executes this deterministic benchmark sequence. ",
    );
    if pattern.is_empty() {
        return Err("the benchmark token pattern is empty".into());
    }
    let mut tokens: Vec<TokenId> = vec![0; required];
    tokens[0] = model.tokenizer().bos_token();
    for (i, token) in tokens.iter_mut().enumerate().skip(1) {
        *token = pattern[(i - 1) % pattern.len()];
    }
    let size_gib =
        std::fs::metadata(&model_path).map_or(0, |m| m.len()) as f64 / (1024.0 * 1024.0 * 1024.0);
    let size = format!("{size_gib:.2} GiB");
    let name = model.model_name();
    println!(
        "| {} | {} | {} | {} | {} |\n| {} | {} | {} | {} | {} |",
        left("model", 32),
        right("size", 10),
        left("backend", 10),
        right("test", 18),
        right("t/s", 21),
        "-".repeat(32),
        "-".repeat(10),
        "-".repeat(10),
        "-".repeat(18),
        "-".repeat(21)
    );
    let print = |test: &str, runs: &[f64]| {
        let (mean, stddev) = stats(runs);
        println!(
            "| {} | {} | {} | {} | {} |",
            left(&name, 32),
            right(&size, 10),
            left("ROCm (HIP)", 10),
            right(test, 18),
            right(&format!("{mean:.2} ± {stddev:.2}"), 21)
        );
    };
    for &depth in &o.n_depths {
        for &length in &o.n_prompts {
            let mut runs = Vec::new();
            // The first repetition is an untimed warm-up of the chunk shapes.
            for repetition in 0..=o.repetitions {
                let mut session = model
                    .create_session(context)
                    .map_err(|e| format!("preparing depth failed: {e}"))?;
                if depth > 0 {
                    session
                        .sync(&tokens[..depth])
                        .map_err(|e| format!("preparing depth failed: {e}"))?;
                }
                let start = Instant::now();
                session
                    .sync(&tokens[..depth + length])
                    .map_err(|e| format!("running prefill failed: {e}"))?;
                let seconds = start.elapsed().as_secs_f64();
                if repetition > 0 {
                    runs.push(length as f64 / seconds);
                }
            }
            print(&test_name("pp", length, depth), &runs);
        }
        for &length in &o.n_gens {
            let prefix = if depth > 0 { depth } else { 16 };
            let mut runs = Vec::new();
            for _ in 0..o.repetitions {
                let mut session = model
                    .create_session(context)
                    .map_err(|e| format!("preparing generation failed: {e}"))?;
                session
                    .sync(&tokens[..prefix])
                    .map_err(|e| format!("preparing generation failed: {e}"))?;
                let history: Vec<SamplerToken> = tokens[..prefix]
                    .iter()
                    .map(|&t| t as SamplerToken)
                    .collect();
                let mut sampler = SamplerState::new(o.sampling.clone(), &history)
                    .map_err(|e| e.message().to_string())?;
                let mut generated: Vec<TokenId> = Vec::new();
                let start = Instant::now();
                while generated.len() < length {
                    let step = session
                        .decode_step(length - generated.len(), &mut sampler, false)
                        .map_err(|e| format!("running Gemma 4 decode failed: {e}"))?;
                    if step.tokens.is_empty() {
                        return Err("running Gemma 4 decode failed: no tokens".into());
                    }
                    generated.extend_from_slice(&step.tokens);
                }
                let seconds = start.elapsed().as_secs_f64();
                runs.push(length as f64 / seconds);
                if o.verbose {
                    let stats = session.statistics();
                    let shown = &generated[..generated.len().min(48)];
                    eprintln!(
                        "Gemma 4 tg depth={depth} cycles={} drafted={} accepted={} \
                         lookup_proposed={} lookup_accepted={} text={}",
                        stats.cycles,
                        stats.drafted,
                        stats.accepted,
                        stats.copied,
                        stats.copied_accepted,
                        String::from_utf8_lossy(&model.decode(shown))
                    );
                }
            }
            print(&test_name("tg", length, depth), &runs);
        }
    }
    println!();
    Ok(())
}
