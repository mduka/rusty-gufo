//! `gufo bench` for Nemotron 3.5 Bolt (the extension's `RunNemotronBench`):
//! single-session prefill (pp) and decode (tg) rates at each depth; with
//! NVIDIA's LiLiCorr drafter, tg runs draft-and-verify rounds. Concurrent
//! and HTTP measurements go through `gufo serve`.

use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use gufo_core::sampling::SamplerState;
use gufo_nemotron::engine::{MAX_BATCH, Model, ModelOptions, PREFILL_CHUNK, Session};
use gufo_nemotron::speculative::{DraftHistory, Frontier, speculative_step};

use super::{CliOptions, cpp_double};

/// Whether the model is a Nemotron-H MoE artifact this runtime supports.
pub(super) fn supports(o: &CliOptions) -> bool {
    gufo_nemotron::runner::supports(Path::new(&*o.model_path.borrow()))
}

pub(super) fn run(o: &CliOptions) -> ExitCode {
    match bench(o) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("Nemotron: {message}");
            ExitCode::from(1)
        }
    }
}

/// The trunk and, for `--speculative dflash2`, NVIDIA's LiLiCorr drafter.
fn load(o: &CliOptions, context: u32) -> Result<Arc<Model>, String> {
    let backend = o.speculative_backend.as_str();
    if !backend.is_empty() && backend != "dflash2" {
        return Err("Nemotron drafts with NVIDIA's LiLiCorr drafter only: \
                    --speculative dflash2 --dflash-model DRAFTER.gguf"
            .into());
    }
    if backend == "dflash2" && o.dflash_model_path.is_empty() {
        return Err("--speculative dflash2 needs --dflash-model".into());
    }
    let mut model = Model::load(
        Path::new(&*o.model_path.borrow()),
        ModelOptions {
            max_context: context,
        },
    )
    .map_err(|e| format!("load failed: {e}"))?;
    if backend == "dflash2" {
        model
            .load_drafter(Path::new(&o.dflash_model_path))
            .map_err(|e| format!("load failed: {e}"))?;
    }
    Ok(Arc::new(model))
}

fn bench(o: &CliOptions) -> Result<(), String> {
    if o.concurrency != [1] || o.validate_prefill_tokens != 0 {
        return Err("Nemotron bench measures one session; use gufo serve for concurrency".into());
    }
    fn largest(v: &[usize]) -> usize {
        v.iter().copied().max().unwrap_or(0)
    }
    // Room for the warm-up as well: a full prefill pass and one round.
    let context = (largest(&o.n_depths) + largest(&o.n_prompts).max(largest(&o.n_gens)) + 1)
        .max(PREFILL_CHUNK + MAX_BATCH + 2);
    let context = u32::try_from(context).map_err(|_| "the benchmark context is too large")?;
    let model = load(o, context)?;
    let mut session = model
        .create_session(context)
        .map_err(|e| format!("session creation failed: {e}"))?;
    let tokenizer = model.tokenizer();
    let seed = tokenizer.encode("The quick brown fox jumps over the lazy dog. ", false);
    if seed.is_empty() {
        return Err("the benchmark token pattern is empty".into());
    }
    let fill = |n: usize| -> Vec<i32> { (0..n).map(|i| seed[i % seed.len()] as i32).collect() };
    let draft_tokens = model
        .drafter()
        .map_or(0, |d| o.draft_tokens.min(d.config().slots()));
    let name = if model.drafter().is_none() {
        "Nemotron-3.5-Bolt"
    } else {
        "Nemotron-3.5-Bolt+LiLiCorr"
    };
    let mut frontier = Frontier {
        logits: vec![0.0; model.config().vocab_size as usize],
        ..Frontier::default()
    };
    let never = |_: i32| false;
    let sampler = || SamplerState::new(o.sampling.clone(), &[]).map_err(|e| e.to_string());
    // Warm the prefill, decode and verification routes.
    session.append(&fill(PREFILL_CHUNK + 1), &mut frontier.logits)?;
    frontier.head.candidates.clear();
    speculative_step(
        &mut session,
        &mut sampler()?,
        &mut frontier,
        &mut DraftHistory::default(),
        draft_tokens + 1,
        &never,
        false,
    )?;
    println!(
        "| model | test | depth | repetition | tokens/s | seconds |\n| :-- | :-- | --: | --: | --: | --: |"
    );
    for &depth in &o.n_depths {
        let prepare = |session: &mut Session| -> Result<(), String> {
            session.reset()?;
            if depth != 0 {
                session.append(&fill(depth), &mut [])?;
            }
            Ok(())
        };
        for r in 0..o.repetitions {
            for &n in &o.n_prompts {
                if n == 0 {
                    continue;
                }
                prepare(&mut session)?;
                let prompt = fill(n);
                let start = Instant::now();
                session.append(&prompt, &mut [])?;
                let dt = start.elapsed().as_secs_f64();
                println!(
                    "| {name} | pp{n} | {depth} | {r} | {} | {} |",
                    cpp_double(n as f64 / dt),
                    cpp_double(dt)
                );
            }
            for &n in &o.n_gens {
                if n == 0 {
                    continue;
                }
                prepare(&mut session)?;
                session.append(&fill(1), &mut frontier.logits)?;
                frontier.head.candidates.clear();
                let mut sampler = sampler()?;
                let mut history = DraftHistory::default();
                let mut generated = 0usize;
                let start = Instant::now();
                while generated < n {
                    let budget = (n - generated).min(draft_tokens as usize + 1) as u32;
                    let round = speculative_step(
                        &mut session,
                        &mut sampler,
                        &mut frontier,
                        &mut history,
                        budget,
                        &never,
                        false,
                    )?;
                    generated += round.tokens.len();
                }
                let dt = start.elapsed().as_secs_f64();
                println!(
                    "| {name} | tg{n} | {depth} | {r} | {} | {} |",
                    cpp_double(n as f64 / dt),
                    cpp_double(dt)
                );
            }
        }
    }
    Ok(())
}
