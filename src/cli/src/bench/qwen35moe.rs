//! Qwen3.6 35B-A3B (the extension's `RunQwenBenchmark` in
//! `q35/model/drivers.cpp`): the dense Qwen benchmark over the `qwen35moe`
//! engine, one greedy request, autoregressive, with DFlash2 or with the MTP
//! head of the target GGUF, with its workload, context depths, repetitions,
//! table and `-v` traces and prefill validation.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use gufo_core::gguf::{Gguf, read_header};
use gufo_core::qwen_tokenizer::StopTokens;
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_qwen35moe::dflash::DFlash;
use gufo_qwen35moe::dflash::policy::DraftPolicy;
use gufo_qwen35moe::mtp::{DEFAULT_PRIME_CHUNK, MtpBackend, MtpDraftConfig, MtpExec, MtpModel};
use gufo_qwen35moe::speculative::SpeculativeState;
use gufo_qwen35moe::verifier::{StepRequest, Verifier};
use gufo_qwen35moe::{Engine, Snapshot, Speculative};
use sha2::{Digest, Sha256};

use super::{
    CliOptions, Table, compare, cpp_double, dense_backend, dense_refusal, quantization_label,
    test_name, tokens,
};

/// Whether `o` names a `qwen35moe` artifact, which this benchmark claims as
/// the extension's provider does (`gguf:qwen35moe`).
pub(super) fn supports(o: &CliOptions) -> bool {
    gufo_qwen35moe::runner::is_qwen35moe(Path::new(&*o.model_path.borrow()))
}

/// The benchmark the 35B runs.
struct Options {
    model: PathBuf,
    prompts: Vec<usize>,
    gens: Vec<usize>,
    depths: Vec<usize>,
    repetitions: usize,
    validate_prefill: usize,
    verbose: bool,
    /// `--speculative dflash2`: the drafter, its policy and block bound.
    dflash: Option<(PathBuf, DraftPolicy, u32)>,
    /// `--speculative mtp`: the GGUF of the head, draft tokens and floor.
    mtp: Option<(String, u32, u32)>,
}

pub(super) fn run(cli: &CliOptions) -> ExitCode {
    // The C++ driver refuses these first, with the dense driver's messages.
    if let Some(refusal) = dense_refusal(cli) {
        eprintln!("{refusal}");
        return ExitCode::from(1);
    }
    if !dense_backend(cli) {
        eprintln!("Error: DSpark requires a DeepSeek V4 Flash model");
        return ExitCode::from(1);
    }
    let options = Options {
        model: PathBuf::from(&*cli.model_path.borrow()),
        prompts: cli.n_prompts.clone(),
        gens: cli.n_gens.clone(),
        depths: cli.n_depths.clone(),
        repetitions: cli.repetitions,
        validate_prefill: cli.validate_prefill_tokens,
        verbose: cli.verbose,
        dflash: (cli.speculative_backend == "dflash2").then(|| {
            (
                PathBuf::from(&cli.dflash_model_path),
                if cli.draft_policy == "fixed" {
                    DraftPolicy::Fixed
                } else {
                    DraftPolicy::Adaptive
                },
                cli.draft_tokens,
            )
        }),
        mtp: (cli.speculative_backend == "mtp").then(|| {
            (
                cli.mtp_model_path.clone(),
                cli.draft_tokens,
                cli.min_draft_tokens,
            )
        }),
    };
    match bench(&options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // A failed validation has printed its report already.
            if !error.is_empty() {
                eprintln!("Error: {error}");
            }
            ExitCode::FAILURE
        }
    }
}

fn bench(o: &Options) -> Result<(), String> {
    let load_start = Instant::now();
    let failed = |error: String| {
        eprintln!(
            "[Model Load]: {} s (failed)",
            cpp_double(load_start.elapsed().as_secs_f64())
        );
        error
    };
    let load_failed = |error: String| {
        eprintln!("Error loading GGUF model '{}': {error}", o.model.display());
        failed(String::new())
    };
    let (header, file_len) = read_header(&o.model).map_err(load_failed)?;
    let gguf = Gguf::parse_prefix(&header, file_len).map_err(load_failed)?;
    let devices = gufo_hip::device_count().unwrap_or(0);
    if devices == 0 {
        eprintln!("Error: No HIP GPU devices available for benchmarking.");
        return Err(failed(String::new()));
    }
    let device = gufo_hip::device_properties(0).map_err(|e| failed(e.to_string()))?;
    let vram_mib = (device.total_global_memory as f64 / (1024.0 * 1024.0)) as usize;
    println!(
        "HIP: found {devices} ROCm devices (Total VRAM: {vram_mib} MiB):\n  \
         Device 0: {}, {}, Wave Size: {}, VRAM: {vram_mib} MiB",
        device.name, device.arch_name, device.warp_size
    );
    let max_depth = o.depths.iter().copied().max().unwrap_or(0);
    let max_test = o
        .prompts
        .iter()
        .chain(&o.gens)
        .copied()
        .max()
        .unwrap_or(0)
        .max(o.validate_prefill);
    let required = u32::try_from(4096.max(max_depth + max_test))
        .map_err(|_| failed("requested benchmark context is too large.".into()))?;
    let mut engine = Engine::load(&o.model, required).map_err(failed)?;
    println!(
        "[Model Load]: {} s",
        cpp_double(load_start.elapsed().as_secs_f64())
    );
    // The MTP head is packed once; each tg repetition gets a fresh draft.
    let mtp = match &o.mtp {
        Some((path, _, _)) => Some(load_mtp(&engine, path, o.verbose)?),
        None => None,
    };
    // One DFlash2 verifier across warmup, depth restores and tg, like C++.
    let mut speculative = match &o.dflash {
        Some((path, policy, draft_tokens)) => Some(Speculative::new(
            DFlash::load(path, engine.weights(), required, *draft_tokens, *policy)
                .map_err(|e| format!("DFlash2 initialization failed: {e}"))?,
        )),
        None => None,
    };

    let parameters: u64 = gguf.tensors().iter().map(|t| t.elements()).sum();
    let table = Table {
        model: format!(
            "{} {}",
            engine.config().model_name,
            quantization_label(&gguf)
        ),
        size: format!("{:.2} GiB", file_len as f64 / (1u64 << 30) as f64),
        params: format!("{:.2} B", parameters as f64 / 1e9),
    };
    if o.validate_prefill > 0 {
        validate_prefill(&mut engine, o.validate_prefill)?;
    }
    Table::header();

    // Warmup.
    engine.reset()?;
    engine.forward_prompt(&tokens(32, 0), 0, true)?;

    // A prepared depth: the target snapshot and, with DFlash2, the
    // verifier's (its own rollback slot belongs to verification).
    let mut prepared: Option<(usize, Snapshot, Option<SpeculativeState>)> = None;
    let mut prepared_next = None;
    for &depth in &o.depths {
        if depth > 0 {
            if o.verbose {
                eprintln!("Preparing context depth {depth}...");
            }
            let mut start = 0;
            let restore_started = Instant::now();
            match &prepared {
                Some((prepared_depth, snapshot, state)) if depth >= *prepared_depth => {
                    engine.restore_snapshot(snapshot)?;
                    if let (Some(spec), Some(state)) = (speculative.as_mut(), state) {
                        spec.restore(&mut engine, state)?;
                    }
                    start = *prepared_depth;
                }
                _ => engine.reset()?,
            }
            let restore_ms = restore_started.elapsed().as_secs_f64() * 1e3;
            let depth_tokens = tokens(depth - start, start);
            let compute_next = !o.gens.is_empty();
            let prepare_started = Instant::now();
            if !depth_tokens.is_empty() {
                let next = match speculative.as_mut() {
                    Some(spec) if start == 0 => {
                        Some(spec.prime(&mut engine, &depth_tokens, compute_next)?)
                    }
                    Some(spec) => Some(spec.extend_prompt(
                        &mut engine,
                        &depth_tokens,
                        start as u32,
                        compute_next,
                    )?),
                    None => engine.forward_prompt(&depth_tokens, start as u32, compute_next)?,
                };
                if compute_next {
                    prepared_next = next;
                }
            }
            let prepare_seconds = prepare_started.elapsed().as_secs_f64();
            let cache_started = Instant::now();
            let snapshot = engine.save_snapshot(depth as u32)?;
            let state = speculative
                .as_ref()
                .map(Speculative::snapshot)
                .transpose()?;
            if o.verbose {
                let mut line = format!(
                    "Prepared {} new tokens in {prepare_seconds:.3} s; cached depth {depth} in {:.2} ms",
                    depth - start,
                    cache_started.elapsed().as_secs_f64() * 1e3
                );
                if start > 0 {
                    let _ = write!(line, " after restoring depth {start} in {restore_ms:.2} ms");
                }
                eprintln!("{line}");
            }
            prepared = Some((depth, snapshot, state));
        }
        // As `restore_depth`, the reset or restore completes before a timed
        // run starts.
        let restore =
            |engine: &mut Engine, speculative: &mut Option<Speculative>| -> Result<(), String> {
                match (&prepared, depth) {
                    (_, 0) => engine.reset()?,
                    (Some((_, snapshot, state)), _) => {
                        engine.restore_snapshot(snapshot)?;
                        if let (Some(spec), Some(state)) = (speculative.as_mut(), state) {
                            spec.restore(engine, state)?;
                        }
                    }
                    (None, _) => return Err("benchmark depth was not prepared".into()),
                }
                engine
                    .model()
                    .stream()
                    .synchronize()
                    .map_err(|e| format!("benchmark restore: {e}"))
            };

        for &length in &o.prompts {
            let prompt = tokens(length, depth);
            let prefill = |engine: &mut Engine,
                           speculative: &mut Option<Speculative>|
             -> Result<(), String> {
                match speculative.as_mut() {
                    Some(spec) if depth == 0 => spec.prime(engine, &prompt, true).map(|_| ()),
                    Some(spec) => spec
                        .extend_prompt(engine, &prompt, depth as u32, true)
                        .map(|_| ()),
                    None => engine
                        .forward_prompt(&prompt, depth as u32, true)
                        .map(|_| ()),
                }
            };
            restore(&mut engine, &mut speculative)?;
            prefill(&mut engine, &mut speculative)?;
            let mut runs = Vec::with_capacity(o.repetitions);
            for _ in 0..o.repetitions {
                restore(&mut engine, &mut speculative)?;
                let started = Instant::now();
                prefill(&mut engine, &mut speculative)?;
                let seconds = started.elapsed().as_secs_f64();
                if seconds > 0.0 {
                    runs.push(length as f64 / seconds);
                }
            }
            table.row(&test_name("pp", length, depth), &runs);
        }

        for &count in &o.gens {
            restore(&mut engine, &mut speculative)?;
            engine.forward_token((depth % 1000 + 100) as u32, depth as u32, true)?;
            let mut runs = Vec::with_capacity(o.repetitions);
            let mut acceptance = Vec::new();
            for _ in 0..o.repetitions {
                restore(&mut engine, &mut speculative)?;
                let sequence = tokens(if depth > 0 { depth } else { 16 }, 0);
                let mut generated = Vec::with_capacity(if o.verbose { count } else { 0 });
                let started;
                if let (Some(model), Some((_, draft_tokens, min_draft_tokens))) = (&mtp, &o.mtp) {
                    // A fresh MTP verifier per repetition, primed over the
                    // whole sequence even at a prepared depth, as in C++.
                    let (model_ref, session) = engine.parts_mut();
                    let mut exec = MtpExec {
                        model: model_ref,
                        mtp: model,
                    };
                    let backend = MtpBackend::new(
                        exec.model.stream(),
                        model,
                        MtpDraftConfig {
                            max_context: required,
                            max_draft_tokens: *draft_tokens,
                            prime_chunk: DEFAULT_PRIME_CHUNK,
                        },
                    )
                    .map_err(|e| format!("MTP initialization failed: {e}"))?;
                    let mut verifier =
                        Verifier::with_draft(backend, *min_draft_tokens, *draft_tokens, true);
                    verifier.set_retain_frontier_logits(false);
                    let mut current = verifier.prime(&mut exec, session, &sequence, true)?;
                    let mut position = sequence.len() as u32;
                    let mut history = Vec::with_capacity(sequence.len() + 1 + count);
                    history.extend_from_slice(&sequence);
                    history.push(current);
                    let mut sampler = SamplerState::new(SamplingConfig::default(), &sequence)
                        .map_err(|e| e.to_string())?;
                    started = Instant::now();
                    let mut emitted = 0;
                    while emitted < count {
                        // EOS is ignored (999999 in C++), as in the other rows.
                        let step = Verifier::verify_step(
                            &mut exec,
                            &mut StepRequest {
                                verifier: &mut verifier,
                                session: &mut *session,
                                sequence: &history,
                                position,
                                current_token: current,
                                stops: StopTokens::default(),
                                max_emitted: (count - emitted) as u32,
                                sampler: &mut sampler,
                            },
                        )?;
                        for &token in &step.emitted {
                            history.push(token);
                            if o.verbose {
                                generated.push(token);
                            }
                            position += 1;
                            emitted += 1;
                            if emitted >= count {
                                break;
                            }
                        }
                        current = step.next_token;
                    }
                    acceptance.push(f64::from(verifier.stats().acceptance_rate()));
                } else {
                    let mut verifier = speculative.as_mut();
                    let (mut current, mut position) = match (depth, verifier.as_deref_mut()) {
                        (0, Some(spec)) => (spec.prime(&mut engine, &sequence, true)?, 16),
                        (0, None) => (
                            engine
                                .forward_prompt(&sequence, 0, true)?
                                .ok_or("prefill returned no token")?,
                            16,
                        ),
                        _ => (
                            prepared_next.ok_or("prepared generation depth has no target token")?,
                            depth as u32,
                        ),
                    };
                    // The committed history the draft may copy from, as C++
                    // keeps it: the sequence, the first token and every
                    // emitted one.
                    let mut history = Vec::new();
                    if verifier.is_some() {
                        history.reserve(sequence.len() + 1 + count);
                        history.extend_from_slice(&sequence);
                        history.push(current);
                    }
                    started = Instant::now();
                    if let Some(spec) = verifier {
                        let mut emitted = 0;
                        while emitted < count {
                            let step = spec.verify_step(
                                &mut engine,
                                &history,
                                current,
                                position,
                                999_999,
                                (count - emitted) as u32,
                            )?;
                            for &token in &step.emitted {
                                history.push(token);
                                if o.verbose {
                                    generated.push(token);
                                }
                                position += 1;
                                emitted += 1;
                                if emitted >= count {
                                    break;
                                }
                            }
                            current = step.next_token;
                        }
                        acceptance.push(f64::from(spec.stats().acceptance_rate()));
                    } else {
                        for _ in 0..count {
                            current = engine
                                .forward_token(current, position, true)?
                                .ok_or("decode returned no token")?;
                            position += 1;
                            if o.verbose {
                                generated.push(current);
                            }
                        }
                    }
                }
                let seconds = started.elapsed().as_secs_f64();
                if o.verbose {
                    let bytes: Vec<u8> = generated.iter().flat_map(|t| t.to_le_bytes()).collect();
                    let digest = Sha256::digest(&bytes);
                    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
                    eprintln!(
                        "[QwenBenchTrace]: depth={depth} count={} sha256={hex}",
                        generated.len()
                    );
                }
                if seconds > 0.0 {
                    runs.push(count as f64 / seconds);
                }
            }
            let mut name = test_name("tg", count, depth);
            if speculative.is_some() {
                name.push_str("-dflash2");
            } else if mtp.is_some() {
                name.push_str("-mtp");
            }
            table.row(&name, &runs);
            if o.verbose && !acceptance.is_empty() {
                let (mean, deviation) = super::stats(&acceptance);
                eprintln!("{name} acceptance={mean:.3} +/- {deviation:.3}");
            }
        }
    }
    println!();
    Ok(())
}

/// `QwenMtpGpuModel::Create` from `--mtp-model`. Failures print the C++
/// message themselves and return an empty error.
fn load_mtp(engine: &Engine, path: &str, verbose: bool) -> Result<Arc<MtpModel>, String> {
    let failed = |message: String| {
        eprintln!("{message}");
        String::new()
    };
    if path.is_empty() {
        return Err(failed("MTP requires --mtp-model".into()));
    }
    let path = Path::new(path);
    gufo_qwen35moe::mtp::open(path)
        .map_err(|e| failed(format!("Error loading MTP GGUF model: {e}")))?;
    let model = MtpModel::load(engine.model().stream(), path, engine.weights())
        .map_err(|e| failed(format!("Error creating GPU MTP model: {e}")))?;
    if verbose {
        eprintln!(
            "Packed GPU MTP view: {} bytes in {} s",
            model.packed_bytes(),
            cpp_double(model.pack_seconds())
        );
    }
    Ok(Arc::new(model))
}

/// `ValidatePrefill`: the logits after the benchmark prompt of `length`
/// tokens, run token by token and as one batch, must agree on the greedy
/// token.
fn validate_prefill(engine: &mut Engine, length: usize) -> Result<(), String> {
    let limit = engine.model().max_prompt_batch() as usize;
    if length > limit {
        return Err(format!(
            "--validate-prefill must be between 1 and {limit} tokens."
        ));
    }
    let prompt = tokens(length, 0);
    engine.reset()?;
    let mut sequential = None;
    for (position, &token) in prompt.iter().enumerate() {
        sequential = engine.forward_token(token, position as u32, position + 1 == length)?;
    }
    let reference = engine.logits()?;
    engine.reset()?;
    let batched = engine.forward_prompt(&prompt, 0, true)?;
    let candidate = engine.logits()?;
    let c = compare(&reference, &candidate);
    let top1 = c.reference_argmax == c.candidate_argmax;
    let (sequential, batched) = (sequential.unwrap_or(0), batched.unwrap_or(0));
    let yes = |value: bool| if value { "yes" } else { "no" };
    println!(
        "[Prefill Validation] tokens={length} sequential_top1={sequential} batched_top1={batched} top1_match={} finite={}\n  \
         max_abs_diff={:.8} mean_abs_diff={:.8} rmse={:.8} cosine_similarity={:.8}",
        yes(top1),
        yes(c.finite),
        c.max_abs_diff,
        c.mean_abs_diff,
        c.root_mean_square_error,
        c.cosine_similarity
    );
    if c.finite && top1 && sequential == batched {
        Ok(())
    } else {
        Err(String::new())
    }
}
