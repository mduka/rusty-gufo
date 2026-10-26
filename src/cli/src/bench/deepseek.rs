//! `gufo bench` for DeepSeek V4 Flash (`RunDeepSeekBenchmark`): prefill and
//! generation at each depth, alone from a prepared snapshot or as
//! concurrent sessions, autoregressive or with DSpark. Sampled runs lend
//! the request sampler to every DSpark cycle, as the server does.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_deepseek_v4_flash::dspark_sampler::DsparkSampler;
use gufo_deepseek_v4_flash::engine::Engine;
use gufo_deepseek_v4_flash::session::{DsparkItem, Session};
use gufo_deepseek_v4_flash::shape::MODEL_SHAPE_NAME;
use sha2::{Digest, Sha256};

use super::{CliOptions, compare, cpp_double, left, load_time, right, stats, test_name};

/// Whether the model is a DeepSeek V4 Flash (`deepseek4`) artifact.
pub(super) fn supports(o: &CliOptions) -> bool {
    gufo_deepseek_v4_flash::runner::supports(Path::new(&*o.model_path.borrow()))
}

pub(super) fn run(o: &CliOptions) -> ExitCode {
    match bench(o) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            // A failed validation has printed its report already.
            if !message.is_empty() {
                eprintln!("{message}");
            }
            ExitCode::from(1)
        }
    }
}

/// `MakeDeepSeekBenchmarkTokens`: a fixed sentence pair, repeated.
fn benchmark_tokens(engine: &Engine, count: usize) -> Result<Vec<i32>, String> {
    let mut pattern = Vec::new();
    engine.vocab.tokenize(
        b"The quick brown fox jumps over the lazy dog. Strix Halo executes this deterministic benchmark sequence. ",
        &mut pattern,
    );
    if pattern.is_empty() {
        return Err("DeepSeek benchmark token pattern is empty".into());
    }
    Ok((0..count).map(|i| pattern[i % pattern.len()]).collect())
}

fn history(tokens: &[i32]) -> Vec<u32> {
    tokens.iter().map(|&t| t as u32).collect()
}

fn sampler(config: &SamplingConfig, prefix: &[i32]) -> Result<SamplerState, String> {
    SamplerState::new(config.clone(), &history(prefix)).map_err(|e| e.to_string())
}

/// `output_sha256` of a request's token IDs (`int`, little endian).
fn sha256(tokens: &[i32]) -> String {
    let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The next autoregressive token: greedy (`SelectNext(0)`), or the request
/// sampler after a pending DSpark draw.
fn next_token(session: &mut Session, sampler: Option<&mut SamplerState>) -> Result<i32, String> {
    let Some(sampler) = sampler else {
        return Ok(session.argmax());
    };
    let mut token = session.take_pending_dspark_token();
    if token < 0 {
        let logits = session
            .logits()
            .ok_or("failed to copy DeepSeek V4 Flash logits")?;
        token = sampler.sample(logits).map_err(|e| e.to_string())? as i32;
    }
    sampler.accept(token as u32).map_err(|e| e.to_string())?;
    Ok(token)
}

fn verbose_tg(concurrency: usize, depth: usize, request: usize, session: &Session, tokens: &[i32]) {
    let s = session.dspark_stats();
    eprintln!(
        "DeepSeek tg C={concurrency} depth={depth} request={request} accepted={} drafted={} steps={} skipped={} output_sha256={}",
        s.support_accepted,
        s.support_drafted,
        s.steps,
        s.skipped,
        sha256(tokens)
    );
}

fn bench(o: &CliOptions) -> Result<(), String> {
    let load_start = Instant::now();
    if gufo_hip::device_count().unwrap_or(0) == 0 {
        load_time(load_start, false);
        return Err("Error: no HIP GPU is available for DeepSeek V4 Flash".into());
    }
    let max = |values: &[usize]| values.iter().copied().max().unwrap_or(0);
    let (max_depth, max_prompt, max_generation) =
        (max(&o.n_depths), max(&o.n_prompts), max(&o.n_gens));
    let limit = i32::MAX as usize - 1;
    if max_depth > limit
        || max_prompt > limit - max_depth
        || max_generation > limit - max_depth.max(16)
        || o.validate_prefill_tokens > limit
    {
        return Err("Error: requested DeepSeek benchmark context is too large".into());
    }
    let required = 4096
        .max(max_depth + max_prompt + 1)
        .max(max_depth.max(16) + max_generation + 1)
        .max(o.validate_prefill_tokens + 1);
    let context = required as u32;
    let dspark = o.speculative_backend == "dspark";
    if !o.speculative_backend.is_empty() && !dspark {
        return Err("Error: DeepSeek supports only --speculative dspark or off".into());
    }
    if dspark && o.dspark_model_path.is_empty() {
        return Err("Error: --speculative dspark requires --dspark-model".into());
    }
    if dspark && o.min_draft_tokens != 1 {
        return Err(
            "Error: DSpark uses model-owned adaptive drafting; custom draft floors are unsupported"
                .into(),
        );
    }
    let speculative = !o.speculative_backend.is_empty();
    // Both paths use the server's target distribution. Stochastic DSpark
    // preserves it, with its own reproducible RNG trace.
    let sampled = !o.sampling.can_use_unmodified_argmax();
    let config = &o.sampling;

    let model_path = o.model_path.borrow().clone();
    let support = dspark.then(|| Path::new(&o.dspark_model_path));
    let mut engine = match Engine::open(Path::new(&model_path), support) {
        Ok(engine) => engine,
        Err(error) => {
            eprintln!("Error creating DeepSeek V4 Flash model: {error}");
            load_time(load_start, false);
            return Err(String::new());
        }
    };
    load_time(load_start, true);
    let tokens = benchmark_tokens(&engine, required)?;
    let session = |engine: &mut Engine| {
        engine
            .create_session(context, speculative)
            .map_err(|e| e.to_string())
    };

    if o.validate_prefill_tokens != 0 {
        let prefix = &tokens[..o.validate_prefill_tokens];
        let failed = |e: String| format!("DeepSeek prefill validation failed: {e}");
        let mut sequential = session(&mut engine).map_err(failed)?;
        let mut batched = session(&mut engine).map_err(failed)?;
        sequential
            .sync(&mut engine, &prefix[..1])
            .map_err(|e| failed(e.to_string()))?;
        batched
            .sync(&mut engine, prefix)
            .map_err(|e| failed(e.to_string()))?;
        for &token in &prefix[1..] {
            sequential
                .eval(&mut engine, token)
                .map_err(|e| format!("DeepSeek scalar reference failed: {e}"))?;
        }
        let (Some(reference), Some(candidate)) = (sequential.logits(), batched.logits()) else {
            return Err("DeepSeek validation logit readback failed: ".into());
        };
        let comparison = compare(reference, candidate);
        let winner = candidate[comparison.reference_argmax];
        let rank = 1 + candidate.iter().filter(|&&v| v > winner).count();
        println!(
            "[Prefill Validation] tokens={} finite={} scalar_winner_rank={rank} rmse={} cosine={} max_error={}",
            prefix.len(),
            u8::from(comparison.finite),
            cpp_double(f64::from(comparison.root_mean_square_error)),
            cpp_double(f64::from(comparison.cosine_similarity)),
            cpp_double(f64::from(comparison.max_abs_diff))
        );
        if !comparison.finite
            || rank > 3
            || comparison.root_mean_square_error > 1.12
            || comparison.cosine_similarity < 0.979
            || comparison.max_abs_diff > 5.0
        {
            return Err(String::new());
        }
    }

    let size = std::fs::metadata(&model_path)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0 * 1024.0))
        .unwrap_or(0.0);
    let size = format!("{size:.2} GiB");
    println!(
        "| {} | {} | {} | {} | {} |",
        left("model", 32),
        right("size", 10),
        left("backend", 10),
        right("test", 18),
        right("t/s", 21)
    );
    println!(
        "| {} | {} | {} | {} | {} |",
        "-".repeat(32),
        "-".repeat(10),
        "-".repeat(10),
        "-".repeat(18),
        "-".repeat(21)
    );
    let result = |test: &str, runs: &[f64]| {
        let (mean, stddev) = stats(runs);
        println!(
            "| {} | {} | {} | {} | {} |",
            left(MODEL_SHAPE_NAME, 32),
            right(&size, 10),
            left("ROCm (HIP)", 10),
            right(test, 18),
            right(&format!("{mean:.2} ± {stddev:.2}"), 21)
        );
    };

    for &concurrency in &o.concurrency {
        if concurrency > 1 {
            for &depth in &o.n_depths {
                let prefix = if depth > 0 { depth } else { 16 };
                let mut base = session(&mut engine)
                    .map_err(|e| format!("Error preparing concurrent depth: {e}"))?;
                base.sync(&mut engine, &tokens[..prefix])
                    .map_err(|e| format!("Error preparing concurrent depth: {e}"))?;
                let snapshot = base
                    .save_snapshot(&engine)
                    .map_err(|e| format!("Error preparing concurrent snapshot: {e}"))?;
                drop(base);
                for generate in [false, true] {
                    let counts = if generate { &o.n_gens } else { &o.n_prompts };
                    for &count in counts {
                        let mut runs = Vec::new();
                        for _ in 0..o.repetitions {
                            let mut sessions = Vec::with_capacity(concurrency);
                            let mut generated: Vec<Vec<i32>> = vec![Vec::new(); concurrency];
                            let mut samplers = (0..concurrency)
                                .map(|_| sampler(config, &tokens[..prefix]))
                                .collect::<Result<Vec<_>, _>>()?;
                            for _ in 0..concurrency {
                                let failed =
                                    |e: String| format!("Error restoring concurrent depth: {e}");
                                let mut s = session(&mut engine).map_err(failed)?;
                                if generate || depth > 0 {
                                    s.restore_snapshot(&engine, &snapshot)
                                        .map_err(|e| failed(e.to_string()))?;
                                }
                                sessions.push(s);
                            }
                            let start = Instant::now();
                            if !generate {
                                for s in &mut sessions {
                                    s.sync(&mut engine, &tokens[..depth + count])
                                        .map_err(|e| format!("Error in concurrent prefill: {e}"))?;
                                }
                            } else if dspark {
                                loop {
                                    let active: Vec<usize> = (0..concurrency)
                                        .filter(|&i| generated[i].len() != count)
                                        .collect();
                                    if active.is_empty() {
                                        break;
                                    }
                                    let mut bridges: Vec<Option<DsparkSampler>> = active
                                        .iter()
                                        .map(|&i| sampled.then(|| DsparkSampler::new(&samplers[i])))
                                        .collect();
                                    let mut items: Vec<DsparkItem<'_>> = sessions
                                        .iter_mut()
                                        .enumerate()
                                        .filter(|(i, _)| generated[*i].len() != count)
                                        .zip(bridges.iter_mut())
                                        .map(|((i, s), bridge)| DsparkItem {
                                            session: s,
                                            max_tokens: count - generated[i].len(),
                                            max_draft_tokens: o.draft_tokens,
                                            sampler: bridge.as_mut(),
                                            stop_at_eos: false,
                                            emitted: Vec::new(),
                                        })
                                        .collect();
                                    Session::dspark_step_batch(&mut engine, &mut items)
                                        .map_err(|e| format!("Error in concurrent DSpark: {e}"))?;
                                    let emitted: Vec<Vec<i32>> =
                                        items.into_iter().map(|item| item.emitted).collect();
                                    for ((&i, emitted), bridge) in
                                        active.iter().zip(emitted).zip(&bridges)
                                    {
                                        if emitted.is_empty()
                                            || emitted.len() > count - generated[i].len()
                                        {
                                            return Err(
                                                "Error: DSpark returned an invalid benchmark token count"
                                                    .into(),
                                            );
                                        }
                                        if let Some(bridge) = bridge {
                                            samplers[i].set_rng_state(bridge.rng_state());
                                            for &token in &emitted {
                                                samplers[i]
                                                    .accept(token as u32)
                                                    .map_err(|e| e.to_string())?;
                                            }
                                        }
                                        generated[i].extend_from_slice(&emitted);
                                    }
                                }
                            } else {
                                for _ in 0..count {
                                    let mut items = Vec::with_capacity(concurrency);
                                    for ((s, sampler), generated) in sessions
                                        .iter_mut()
                                        .zip(samplers.iter_mut())
                                        .zip(generated.iter_mut())
                                    {
                                        let token = next_token(s, sampled.then_some(sampler))
                                            .map_err(|e| {
                                                format!("Error in concurrent sampling: {e}")
                                            })?;
                                        if token < 0 {
                                            return Err("Error in concurrent sampling: ".into());
                                        }
                                        generated.push(token);
                                        items.push((s, token));
                                    }
                                    Session::eval_batch(&mut engine, &mut items).map_err(|e| {
                                        format!("Error in concurrent target decode: {e}")
                                    })?;
                                }
                            }
                            let seconds = start.elapsed().as_secs_f64();
                            let produced = if generate { count } else { count * concurrency };
                            runs.push(produced as f64 / seconds);
                            if o.verbose && generate {
                                for (i, s) in sessions.iter().enumerate() {
                                    verbose_tg(concurrency, depth, i, s, &generated[i]);
                                }
                            }
                        }
                        let kind = if generate { "tg" } else { "pp" };
                        result(
                            &format!("{} C{concurrency}", test_name(kind, count, depth)),
                            &runs,
                        );
                    }
                }
            }
            continue;
        }

        let mut prepared: Option<Session> = None;
        let mut prepared_depth = 0;
        for &depth in &o.n_depths {
            let mut snapshot = None;
            if depth > 0 {
                let failed = |e: String| format!("Error preparing DeepSeek depth {depth}: {e}");
                // A shallower depth than the prepared one starts over.
                if prepared.is_none() || depth < prepared_depth {
                    prepared = Some(session(&mut engine).map_err(failed)?);
                }
                let base = prepared.as_mut().ok_or_else(|| failed(String::new()))?;
                base.sync(&mut engine, &tokens[..depth])
                    .map_err(|e| failed(e.to_string()))?;
                prepared_depth = depth;
                let payload = base
                    .save_snapshot(&engine)
                    .map_err(|e| format!("Error snapshotting DeepSeek depth {depth}: {e}"))?;
                if o.verbose {
                    eprintln!(
                        "Prepared DeepSeek depth {depth} snapshot_bytes={}",
                        payload.len()
                    );
                }
                snapshot = Some(payload);
            }

            for &prompt in &o.n_prompts {
                if depth + prompt >= required {
                    return Err("Error: DeepSeek prompt benchmark exceeds context".into());
                }
                let mut runs = Vec::with_capacity(o.repetitions);
                for _ in 0..o.repetitions {
                    let mut local = None;
                    let s = match (&snapshot, prepared.as_mut()) {
                        (Some(payload), Some(base)) => {
                            base.restore_snapshot(&engine, payload)
                                .map_err(|e| format!("Error restoring DeepSeek depth: {e}"))?;
                            base
                        }
                        _ => local.insert(
                            session(&mut engine)
                                .map_err(|e| format!("Error creating DeepSeek session: {e}"))?,
                        ),
                    };
                    let start = Instant::now();
                    s.sync(&mut engine, &tokens[..depth + prompt])
                        .map_err(|e| format!("Error running DeepSeek prefill: {e}"))?;
                    runs.push(prompt as f64 / start.elapsed().as_secs_f64());
                    if o.verbose {
                        eprintln!(
                            "DeepSeek pp depth={depth} payload_bytes={}",
                            s.payload_bytes()
                        );
                    }
                }
                result(&test_name("pp", prompt, depth), &runs);
            }

            for &generation in &o.n_gens {
                let prefix = if depth > 0 { depth } else { 16 };
                if prefix + generation >= required {
                    return Err("Error: DeepSeek generation benchmark exceeds context".into());
                }
                let mut runs = Vec::with_capacity(o.repetitions);
                for _ in 0..o.repetitions {
                    let failed = |e: String| format!("Error preparing DeepSeek generation: {e}");
                    let mut local = None;
                    let s = match (&snapshot, prepared.as_mut()) {
                        (Some(payload), Some(base)) => {
                            base.restore_snapshot(&engine, payload)
                                .map_err(|e| failed(e.to_string()))?;
                            base
                        }
                        _ => {
                            let s = local.insert(session(&mut engine).map_err(failed)?);
                            s.sync(&mut engine, &tokens[..prefix])
                                .map_err(|e| failed(e.to_string()))?;
                            s
                        }
                    };
                    let mut generated = Vec::with_capacity(generation);
                    let mut request = sampler(config, &tokens[..prefix])?;
                    let start = Instant::now();
                    while generated.len() < generation {
                        if dspark {
                            let mut bridge = sampled.then(|| DsparkSampler::new(&request));
                            let emitted = s
                                .dspark_step(
                                    &mut engine,
                                    generation - generated.len(),
                                    o.draft_tokens,
                                    bridge.as_mut(),
                                    false,
                                )
                                .map_err(|e| format!("Error running DSpark decode: {e}"))?;
                            if emitted.is_empty() {
                                return Err("Error running DSpark decode: ".into());
                            }
                            if let Some(bridge) = &bridge {
                                request.set_rng_state(bridge.rng_state());
                                for &token in &emitted {
                                    request.accept(token as u32).map_err(|e| e.to_string())?;
                                }
                            }
                            generated.extend_from_slice(&emitted);
                        } else {
                            let failed = |e: String| format!("Error running DeepSeek decode: {e}");
                            let token =
                                next_token(s, sampled.then_some(&mut request)).map_err(failed)?;
                            if token < 0 {
                                return Err(failed(String::new()));
                            }
                            s.eval(&mut engine, token)
                                .map_err(|e| failed(e.to_string()))?;
                            generated.push(token);
                        }
                    }
                    runs.push(generation as f64 / start.elapsed().as_secs_f64());
                    if o.verbose {
                        verbose_tg(1, depth, 0, s, &generated);
                    }
                }
                result(&test_name("tg", generation, depth), &runs);
            }

            if let (Some(payload), Some(base)) = (&snapshot, prepared.as_mut()) {
                base.restore_snapshot(&engine, payload)
                    .map_err(|e| format!("Error restoring final DeepSeek depth: {e}"))?;
            }
        }
    }
    println!();
    Ok(())
}
