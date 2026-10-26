//! `gufo bench` (`src/cli/bench/bench.cpp`): the C++ option table, and the
//! prompt processing (pp) and token generation (tg) benchmark of dense
//! Qwen3.8 (autoregressive, with DFlash2 or with an MTP head) with its
//! workload, context depths, repetitions, table and `-v` traces and prefill
//! validation, of Flash-Next (`flash_next`) and of
//! DeepSeek V4 Flash (`deepseek`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use gufo_core::gguf::{Gguf, read_header};
use gufo_core::sampling::SamplingConfig;
use gufo_hip::GgmlType;
use gufo_qwen35::dflash::DFlash;
use gufo_qwen35::dflash::policy::DraftPolicy;
use gufo_qwen35::mtp::{MtpDraft, MtpModel};
use gufo_qwen35::speculative::SpeculativeState;
use gufo_qwen35::{Engine, Snapshot, Speculative};
use sha2::{Digest, Sha256};

use crate::arg_parser::ArgParser;
use crate::sampling_options::register_sampling_options;

/// `BenchOptions` of the C++ command.
struct CliOptions {
    model_path: RefCell<String>,
    n_prompts: Vec<usize>,
    n_gens: Vec<usize>,
    n_depths: Vec<usize>,
    concurrency: Vec<usize>,
    repetitions: usize,
    validate_prefill_tokens: usize,
    speculative_backend: String,
    mtp_model_path: String,
    dflash_model_path: String,
    draft_policy: String,
    dspark_model_path: String,
    draft_tokens: u32,
    min_draft_tokens: u32,
    sampling: SamplingConfig,
    verbose: bool,
    explicit_p: bool,
    explicit_n: bool,
    speculative_explicit: bool,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            model_path: RefCell::new(String::new()),
            n_prompts: vec![2048],
            n_gens: vec![128],
            n_depths: vec![0],
            concurrency: vec![1],
            repetitions: 1,
            validate_prefill_tokens: 0,
            speculative_backend: String::new(),
            mtp_model_path: String::new(),
            dflash_model_path: String::new(),
            draft_policy: String::new(),
            dspark_model_path: String::new(),
            draft_tokens: 7,
            min_draft_tokens: 1,
            sampling: SamplingConfig {
                seed: 0,
                ..SamplingConfig::default()
            },
            verbose: false,
            explicit_p: false,
            explicit_n: false,
            speculative_explicit: false,
        }
    }
}

/// `ParseCommaSeparatedSizes`: whole decimal counts; zeros are dropped
/// unless `keep_zero`, and an empty list or a trailing comma fails.
fn comma_separated_sizes(text: &str, keep_zero: bool) -> Option<Vec<usize>> {
    if text.is_empty() || text.ends_with(',') {
        return None;
    }
    let mut sizes = Vec::new();
    for part in text.split(',') {
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let value: usize = part.parse().ok()?;
        if value != 0 || keep_zero {
            sizes.push(value);
        }
    }
    Some(sizes)
}

/// A whole positive decimal count (`std::from_chars` over the full text).
fn positive<T: std::str::FromStr + Default + PartialEq>(text: &str) -> Option<T> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse().ok().filter(|value| *value != T::default())
}

/// `RegisterBenchOptions`.
fn parser(options: &mut CliOptions) -> ArgParser<'_> {
    let CliOptions {
        model_path,
        n_prompts,
        n_gens,
        n_depths,
        concurrency,
        repetitions,
        validate_prefill_tokens,
        speculative_backend,
        mtp_model_path,
        dflash_model_path,
        draft_policy,
        dspark_model_path,
        draft_tokens,
        min_draft_tokens,
        sampling,
        verbose,
        explicit_p,
        explicit_n,
        speculative_explicit,
    } = options;
    let model_path = &*model_path;
    let mut parser = ArgParser::new(
        "gufo bench",
        "Benchmark prompt processing (pp) and token generation (tg) throughput.",
    );
    parser
        .add_custom_option(
            "-m",
            "--model",
            "PATH",
            "Path to GGUF model file (required)",
            "Model",
            move |_, value| {
                *model_path.borrow_mut() = value.to_owned();
                Ok(())
            },
        )
        .add_custom_option(
            "-p",
            "--n-prompt",
            "n,n,...",
            "Prompt token lengths to benchmark (default: 2048)",
            "Workload",
            move |flag, value| {
                *n_prompts = comma_separated_sizes(value, false)
                    .ok_or_else(|| format!("Invalid token counts for {flag}"))?;
                *explicit_p = true;
                Ok(())
            },
        )
        .add_custom_option(
            "-n",
            "--n-gen",
            "n,n,...",
            "Number of text generation tokens to benchmark (default: 128)",
            "Workload",
            move |flag, value| {
                *n_gens = comma_separated_sizes(value, false)
                    .ok_or_else(|| format!("Invalid token counts for {flag}"))?;
                *explicit_n = true;
                Ok(())
            },
        )
        .add_custom_option(
            "-d",
            "--n-depth",
            "n,n,...",
            "Context depths prepared before timed region (default: 0)",
            "Workload",
            move |flag, value| {
                *n_depths = comma_separated_sizes(value, true)
                    .ok_or_else(|| format!("Invalid argument for {flag}"))?;
                Ok(())
            },
        )
        .add_custom_option(
            "-c",
            "--concurrency",
            "n,n,...",
            "DS4 simultaneous requests, 1..8 (default: 1); pp is aggregate, tg per user",
            "Workload",
            move |_, value| {
                *concurrency = comma_separated_sizes(value, true)
                    .filter(|sizes| {
                        !sizes.is_empty() && sizes.iter().all(|&n| (1..=8).contains(&n))
                    })
                    .ok_or("--concurrency requires DS4 request counts between 1 and 8")?;
                Ok(())
            },
        )
        .add_option(
            "-r",
            "--repetitions",
            "N",
            "Repetitions per test point for variance reduction (default: 1)",
            "Workload",
            repetitions,
        )
        .add_custom_option(
            "",
            "--validate-prefill",
            "N",
            "Compare batched prefill logits against sequential reference",
            "Validation",
            move |_, value| {
                *validate_prefill_tokens =
                    positive(value).ok_or("Invalid argument for --validate-prefill")?;
                Ok(())
            },
        )
        .add_custom_option(
            "",
            "--speculative",
            "MODE",
            "Draft backend: dflash2, mtp, dspark, or off",
            "Speculative",
            move |_, value| {
                *speculative_explicit = true;
                match value {
                    "off" => speculative_backend.clear(),
                    "dspark" | "dflash2" | "mtp" => *speculative_backend = value.to_owned(),
                    _ => return Err(format!("Unknown speculative backend: {value}")),
                }
                Ok(())
            },
        )
        .add_option(
            "",
            "--dflash-model",
            "PATH",
            "Path to Qwen DFlash2 GGUF file",
            "Speculative",
            dflash_model_path,
        )
        .add_option(
            "",
            "--draft-policy",
            "POLICY",
            "DFlash2 block length: fixed or adaptive (default: adaptive)",
            "Speculative",
            draft_policy,
        )
        .add_option(
            "",
            "--dspark-model",
            "PATH",
            "DeepSeek V4 Flash DSpark support GGUF",
            "Speculative",
            dspark_model_path,
        )
        .add_option(
            "",
            "--mtp-model",
            "PATH",
            "Path to quantized Qwen MTP draft head GGUF file",
            "Speculative",
            mtp_model_path,
        )
        .add_custom_option(
            "",
            "--draft-tokens",
            "N",
            "Maximum speculative draft tokens per verification step (default: 7)",
            "Speculative",
            move |_, value| {
                *draft_tokens = positive(value).ok_or("Invalid argument for --draft-tokens")?;
                Ok(())
            },
        )
        .add_custom_option(
            "",
            "--min-draft-tokens",
            "N",
            "Adaptive draft floor (default: 1)",
            "Speculative",
            move |_, value| {
                *min_draft_tokens =
                    positive(value).ok_or("Invalid argument for --min-draft-tokens")?;
                Ok(())
            },
        );
    register_sampling_options(&mut parser, sampling, "Sampling", false, false);
    parser
        .add_flag(
            "-v",
            "--verbose",
            "Print detailed timing, latency breakdown, and tok/s metrics",
            "General",
            verbose,
        )
        .set_positional_handler(move |value| {
            *model_path.borrow_mut() = value.to_owned();
            Ok(())
        });
    parser
}

/// `PrintBenchHelp`.
pub fn print_help() {
    parser(&mut CliOptions::default()).print_help();
}

/// `ParseBenchOptions`: the options, `None` after the help, or the exit
/// code of a usage error.
fn parse(args: &[String]) -> Result<Option<CliOptions>, ExitCode> {
    let mut o = CliOptions::default();
    let (parsed, help) = {
        let mut parser = parser(&mut o);
        (parser.parse(args), parser.is_help_requested())
    };
    let usage = |message: &str| {
        eprintln!("Error: {message}");
        print_help();
        ExitCode::from(2)
    };
    match parsed {
        // A parser that failed without a message prints the help, as C++.
        Err(error) if error.is_empty() => {
            print_help();
            return Ok(None);
        }
        Err(error) => return Err(usage(&error)),
        Ok(()) if help => {
            print_help();
            return Ok(None);
        }
        Ok(()) => {}
    }
    if o.explicit_p && !o.explicit_n {
        o.n_gens.clear();
    } else if !o.explicit_p && o.explicit_n {
        o.n_prompts.clear();
    }
    if o.repetitions == 0
        || (o.n_prompts.is_empty() && o.n_gens.is_empty() && o.validate_prefill_tokens == 0)
    {
        return Err(usage(
            "Benchmark needs positive repetitions and at least one workload",
        ));
    }
    if !o.speculative_explicit && !o.dspark_model_path.is_empty() {
        o.speculative_backend = "dspark".into();
    }
    if o.draft_tokens == 0 || o.min_draft_tokens == 0 || o.min_draft_tokens > o.draft_tokens {
        return Err(usage("min-draft-tokens cannot exceed draft-tokens"));
    }
    if let Err(error) = o.sampling.validate() {
        return Err(usage(error.message()));
    }
    if o.sampling.seed < 0 {
        return Err(usage("benchmark seed must be non-negative"));
    }
    if !o.draft_policy.is_empty()
        && ((o.draft_policy != "fixed" && o.draft_policy != "adaptive")
            || o.speculative_backend != "dflash2")
    {
        return Err(usage(
            "--draft-policy requires DFlash2 and fixed or adaptive",
        ));
    }
    if o.min_draft_tokens != 1 && o.speculative_backend == "dflash2" {
        return Err(usage(
            "DFlash2 requires --min-draft-tokens 1; bound blocks with --draft-tokens",
        ));
    }
    Ok(Some(o))
}

/// `--speculative dflash2` settings.
struct DFlashOptions {
    model: PathBuf,
    policy: DraftPolicy,
    draft_tokens: u32,
}

/// `--speculative mtp` settings.
struct MtpOptions {
    model: String,
    draft_tokens: u32,
    min_draft_tokens: u32,
}

/// The benchmark the Rust engine runs.
struct Options {
    model: PathBuf,
    prompts: Vec<usize>,
    gens: Vec<usize>,
    depths: Vec<usize>,
    repetitions: usize,
    validate_prefill: usize,
    verbose: bool,
    dflash: Option<DFlashOptions>,
    mtp: Option<MtpOptions>,
}

/// Whether dense Qwen runs the draft: none, DFlash2 or an MTP head. The
/// dense engine takes every artifact Flash-Next and DeepSeek do not; loading
/// reports the ones it cannot run.
fn dense_backend(o: &CliOptions) -> bool {
    matches!(o.speculative_backend.as_str(), "" | "dflash2" | "mtp")
}

/// The workloads the C++ command refuses for dense Qwen, with its message:
/// it benchmarks one greedy request; the serving benchmark covers the rest.
fn dense_refusal(o: &CliOptions) -> Option<&'static str> {
    if o.concurrency != [1] {
        Some(
            "Error: concurrent model benchmarks currently support DS4 only; use the serving benchmark for Qwen",
        )
    } else if !o.sampling.can_use_unmodified_argmax() {
        Some(
            "Error: sampled model benchmarks support DS4 and Flash-Next; use the serving benchmark for Qwen",
        )
    } else {
        None
    }
}

/// The C++ benchmark tokens: `(position % 1000) + 100` from `start`.
fn tokens(count: usize, start: usize) -> Vec<u32> {
    (start..start + count)
        .map(|position| (position % 1000 + 100) as u32)
        .collect()
}

/// Mean and sample standard deviation (`ComputeStats`).
fn stats(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() == 1 {
        return (mean, 0.0);
    }
    let variance =
        values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (values.len() - 1) as f64;
    (mean, variance.sqrt())
}

/// `test` like `MakeTestName`: `pp512`, `tg128 @ d4096`.
fn test_name(prefix: &str, count: usize, depth: usize) -> String {
    if depth == 0 {
        format!("{prefix}{count}")
    } else {
        format!("{prefix}{count} @ d{depth}")
    }
}

/// Right-aligns like `std::setw`, which counts bytes (the `±` takes two).
fn right(text: &str, width: usize) -> String {
    format!("{}{text}", " ".repeat(width.saturating_sub(text.len())))
}

fn left(text: &str, width: usize) -> String {
    format!("{text}{}", " ".repeat(width.saturating_sub(text.len())))
}

/// The dominant format by element count, "mixed" below four fifths
/// (`GgufReader::GetQuantizationLabel`).
fn quantization_label(gguf: &Gguf<'_>) -> String {
    let mut elements: HashMap<GgmlType, u64> = HashMap::new();
    let mut total = 0u64;
    for tensor in gguf.tensors() {
        *elements.entry(tensor.ty).or_default() += tensor.elements();
        total += tensor.elements();
    }
    let Some((ty, count)) = elements.into_iter().max_by_key(|&(_, count)| count) else {
        return "unknown".into();
    };
    if total == 0 {
        return "unknown".into();
    }
    let mut label = ty.name().to_string();
    if count * 5 < total * 4 {
        label.push_str(" mixed");
    }
    label
}

/// `std::ostream << double`: six significant digits, no trailing zeros
/// (`%g`); the exponent is the one after rounding to six digits.
pub(crate) fn cpp_double(value: f64) -> String {
    cpp_general(value, 6)
}

/// A double as `std::ostream` writes it with `std::setprecision(precision)`
/// in the default float format (`%g`).
pub(crate) fn cpp_general(value: f64, precision: usize) -> String {
    if value == 0.0 || !value.is_finite() {
        return format!("{value}");
    }
    let precision = precision.max(1);
    let text = format!("{value:.*e}", precision - 1);
    let (mantissa, power) = text.split_once('e').unwrap_or((&text, "0"));
    let exponent: i32 = power.parse().unwrap_or(0);
    if exponent < -4 || exponent >= precision as i32 {
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        return format!(
            "{mantissa}e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        );
    }
    let decimals = (precision as i32 - 1 - exponent).max(0) as usize;
    let text = format!("{value:.decimals$}");
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

struct Table {
    model: String,
    size: String,
    params: String,
}

impl Table {
    fn header() {
        println!(
            "| {} | {} | {} | {} | {} | {} |",
            left("model", 30),
            right("size", 10),
            right("params", 10),
            left("backend", 10),
            right("test", 15),
            right("t/s", 21)
        );
        println!(
            "| {} | {} | {} | {} | {} | {} |",
            "-".repeat(30),
            "-".repeat(10),
            "-".repeat(10),
            "-".repeat(10),
            "-".repeat(15),
            "-".repeat(21)
        );
    }

    fn row(&self, test: &str, runs: &[f64]) {
        let (mean, stddev) = stats(runs);
        println!(
            "| {} | {} | {} | {} | {} | {} |",
            left(&self.model, 30),
            right(&self.size, 10),
            right(&self.params, 10),
            left("ROCm (HIP)", 10),
            right(test, 15),
            right(&format!("{mean:.2} ± {stddev:.2}"), 21)
        );
    }
}

/// `gufo bench`.
pub fn run(args: &[String]) -> ExitCode {
    let cli = match parse(args) {
        Ok(Some(cli)) => cli,
        Ok(None) => return ExitCode::SUCCESS,
        Err(code) => return code,
    };
    if cli.model_path.borrow().is_empty() {
        eprintln!("Error: --model <PATH> is required");
        return ExitCode::from(2);
    }
    if gufo_qwen38_flash_next::runner::supports(Path::new(&*cli.model_path.borrow())) {
        return flash_next::run(&cli);
    }
    if deepseek::supports(&cli) {
        return deepseek::run(&cli);
    }
    if let Some(refusal) = dense_refusal(&cli) {
        eprintln!("{refusal}");
        return ExitCode::from(1);
    }
    if !dense_backend(&cli) {
        eprintln!("Error: DSpark requires a DeepSeek V4 Flash model");
        return ExitCode::from(1);
    }
    let dflash = (cli.speculative_backend == "dflash2").then(|| DFlashOptions {
        model: PathBuf::from(&cli.dflash_model_path),
        policy: if cli.draft_policy == "fixed" {
            DraftPolicy::Fixed
        } else {
            DraftPolicy::Adaptive
        },
        draft_tokens: cli.draft_tokens,
    });
    let options = Options {
        model: PathBuf::from(cli.model_path.into_inner()),
        prompts: cli.n_prompts,
        gens: cli.n_gens,
        depths: cli.n_depths,
        repetitions: cli.repetitions,
        validate_prefill: cli.validate_prefill_tokens,
        verbose: cli.verbose,
        dflash,
        mtp: (cli.speculative_backend == "mtp").then(|| MtpOptions {
            model: cli.mtp_model_path.clone(),
            draft_tokens: cli.draft_tokens,
            min_draft_tokens: cli.min_draft_tokens,
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
        Some(m) => Some(load_mtp(&engine, m, o.verbose)?),
        None => None,
    };
    // One DFlash2 verifier across warmup, depth restores and tg, like C++.
    let mut speculative = match &o.dflash {
        Some(d) => Some(Speculative::new(
            DFlash::load(
                &d.model,
                engine.weights(),
                required,
                d.draft_tokens,
                d.policy,
            )
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
                // A fresh MTP verifier per repetition, primed over the
                // whole sequence even at a prepared depth, as in C++.
                let mut fresh = match (&mtp, &o.mtp) {
                    (Some(model), Some(m)) => {
                        let stream = engine.model().stream();
                        let draft = MtpDraft::new(stream, model, required, m.draft_tokens)
                            .map_err(|e| format!("MTP initialization failed: {e}"))?;
                        Some(Speculative::mtp(
                            Arc::clone(model),
                            draft,
                            m.min_draft_tokens,
                            m.draft_tokens,
                        ))
                    }
                    _ => None,
                };
                let mtp_step = fresh.is_some();
                let mut verifier = match fresh.as_mut() {
                    Some(spec) => Some(spec),
                    None => speculative.as_mut(),
                };
                let (mut current, mut position) = match (depth, verifier.as_deref_mut()) {
                    (_, Some(spec)) if mtp_step => (
                        spec.prime(&mut engine, &sequence, true)?,
                        sequence.len() as u32,
                    ),
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
                let mut generated = Vec::with_capacity(if o.verbose { count } else { 0 });
                // The committed history the draft may copy from, as C++ keeps
                // it: the sequence, the first token and every emitted one.
                let mut history = Vec::new();
                if verifier.is_some() {
                    history.reserve(sequence.len() + 1 + count);
                    history.extend_from_slice(&sequence);
                    history.push(current);
                }
                let started = Instant::now();
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
                let (mean, deviation) = stats(&acceptance);
                eprintln!("{name} acceptance={mean:.3} +/- {deviation:.3}");
            }
        }
    }
    println!();
    if o.verbose && engine.graph_mismatches() > 0 {
        eprintln!("graph mismatches: {}", engine.graph_mismatches());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles_print_like_iostream() {
        assert_eq!(cpp_double(3.1234567), "3.12346");
        assert_eq!(cpp_double(12.5), "12.5");
        assert_eq!(cpp_double(0.000123456), "0.000123456");
        assert_eq!(cpp_double(1234567.0), "1.23457e+06");
        assert_eq!(cpp_double(2.0), "2");
        // The exponent after rounding decides, as for %g.
        assert_eq!(cpp_double(999_999.5), "1e+06");
        assert_eq!(cpp_double(0.000_012_345_6), "1.23456e-05");
        assert_eq!(cpp_double(0.000_099_999_99), "0.0001");
    }

    #[test]
    fn workloads_default_like_the_cpp_bench() {
        let parse = |list: &[&str]| {
            let args: Vec<String> = list.iter().map(|s| s.to_string()).collect();
            parse(&args).ok().flatten()
        };
        let o = parse(&["-m", "x.gguf"]).unwrap();
        assert_eq!(
            (o.n_prompts, o.n_gens, o.n_depths),
            (vec![2048], vec![128], vec![0])
        );
        let o = parse(&["x.gguf", "-p", "512,0,1024"]).unwrap();
        assert!(o.n_gens.is_empty());
        assert_eq!(o.n_prompts, [512, 1024], "zero counts are dropped");
        assert_eq!(*o.model_path.borrow(), "x.gguf");
        let o = parse(&["-n", "64", "-d", "0,4096", "-v", "--dspark-model", "d"]).unwrap();
        assert!(o.n_prompts.is_empty() && o.verbose);
        assert_eq!(o.n_depths, [0, 4096]);
        assert_eq!(o.speculative_backend, "dspark");
        for invalid in [
            &["-p", "0"][..],
            &["-r", "0"],
            &["-p", "1,"],
            &["-p", " 1"],
            &["-c", "9"],
            &["--draft-tokens", "0"],
            &["--speculative", "dflash2", "--min-draft-tokens", "2"],
            &["--draft-policy", "fixed"],
            &["--seed", "-1"],
        ] {
            assert!(parse(invalid).is_none(), "{invalid:?}");
        }
        assert_eq!(tokens(3, 999), [1099, 100, 101]);
        assert_eq!(test_name("tg", 128, 4096), "tg128 @ d4096");
        assert_eq!(right("1.00 ± 0.00", 13), " 1.00 ± 0.00");
    }

    /// The DSpark and DS4 workload options of the C++ `ds4.cli` contract;
    /// `prompt::tests` checks the same DSpark arguments.
    #[test]
    fn ds4_options_match_the_cpp_contract() {
        let parse = |list: &[&str]| {
            let args: Vec<String> = list.iter().map(|s| s.to_string()).collect();
            parse(&args).ok().flatten()
        };
        for (args, backend) in [
            (&["--dspark-model", "support.gguf"][..], "dspark"),
            (
                &["--speculative", "dspark", "--dspark-model", "support.gguf"],
                "dspark",
            ),
            (
                &["--speculative", "off", "--dspark-model", "support.gguf"],
                "",
            ),
            (
                &["--dspark-model", "support.gguf", "--speculative", "off"],
                "",
            ),
        ] {
            let o = parse(args).unwrap();
            assert_eq!(o.speculative_backend, backend, "{args:?}");
            assert_eq!(o.dspark_model_path, "support.gguf", "{args:?}");
        }
        let o = parse(&["--dspark-model", "support.gguf", "--draft-tokens", "3"]).unwrap();
        assert_eq!(o.draft_tokens, 3);
        let o = parse(&[
            "--concurrency",
            "1,2,4,6,8",
            "-p",
            "2048",
            "-n",
            "128",
            "-d",
            "0,4096,8192,12288,16384",
        ])
        .unwrap();
        assert_eq!(o.concurrency, [1, 2, 4, 6, 8]);
        assert_eq!(o.n_prompts, [2048]);
        assert_eq!(o.n_gens, [128]);
        assert_eq!(o.n_depths, [0, 4096, 8192, 12288, 16384]);
        for invalid in ["0", "9", "2,0", "-1", ""] {
            assert!(parse(&["--concurrency", invalid]).is_none(), "{invalid:?}");
        }
    }

    /// Dense Qwen refuses what the C++ command refuses, concurrency first.
    #[test]
    fn dense_refusals_match_the_cpp_command() {
        let refusal = |list: &[&str]| {
            let args: Vec<String> = list.iter().map(|s| s.to_string()).collect();
            dense_refusal(&parse(&args).ok().flatten().unwrap())
        };
        assert_eq!(refusal(&["--validate-prefill", "64"]), None);
        assert!(
            refusal(&["-c", "2", "--temperature", "0.7"])
                .unwrap()
                .contains("concurrent")
        );
        assert!(
            refusal(&["--temperature", "0.7"])
                .unwrap()
                .contains("sampled")
        );
        assert!(
            refusal(&["--repeat-penalty", "1.1"])
                .unwrap()
                .contains("sampled")
        );
    }

    /// `testing::CompareLogits` as the prefill validation reports it.
    #[test]
    fn logit_comparison_matches_compare_logits() {
        let c = compare(&[1.0, 3.0, 2.0], &[1.5, 2.0, 4.0]);
        assert!(c.finite);
        assert_eq!((c.reference_argmax, c.candidate_argmax), (1, 2));
        assert_eq!(c.max_abs_diff, 2.0);
        assert_eq!(c.mean_abs_diff, 3.5 / 3.0);
        let c = compare(&[1.0, f32::NAN], &[1.0, 2.0]);
        assert!(!c.finite);
        assert_eq!(
            (c.max_abs_diff, c.mean_abs_diff, c.cosine_similarity),
            (f32::INFINITY, f32::INFINITY, 0.0)
        );
    }
}

/// `QwenMtpGpuModel::Create` from `--mtp-model`. Failures print the C++
/// message themselves and return an empty error.
fn load_mtp(engine: &Engine, options: &MtpOptions, verbose: bool) -> Result<Arc<MtpModel>, String> {
    let failed = |message: String| {
        eprintln!("{message}");
        String::new()
    };
    if options.model.is_empty() {
        return Err(failed("MTP requires --mtp-model".into()));
    }
    let path = Path::new(&options.model);
    gufo_qwen35::mtp::open(path)
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

fn load_time(start: Instant, success: bool) {
    let text = format!(
        "[Model Load]: {} s",
        cpp_double(start.elapsed().as_secs_f64())
    );
    if success {
        println!("{text}");
    } else {
        eprintln!("{text} (failed)");
    }
}

/// The fields of `testing::CompareLogits` the validation reports.
struct Comparison {
    finite: bool,
    root_mean_square_error: f32,
    cosine_similarity: f32,
    max_abs_diff: f32,
    mean_abs_diff: f32,
    reference_argmax: usize,
    candidate_argmax: usize,
}

fn compare(reference: &[f32], candidate: &[f32]) -> Comparison {
    let mut c = Comparison {
        finite: true,
        root_mean_square_error: 0.0,
        cosine_similarity: 1.0,
        max_abs_diff: 0.0,
        mean_abs_diff: 0.0,
        reference_argmax: 0,
        candidate_argmax: 0,
    };
    let (mut absolute, mut squared, mut dot) = (0.0, 0.0, 0.0);
    let (mut reference_norm, mut candidate_norm) = (0.0, 0.0);
    let (mut best, mut best_candidate) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, (&r, &x)) in reference.iter().zip(candidate).enumerate() {
        if !r.is_finite() || !x.is_finite() {
            c.finite = false;
            c.max_abs_diff = f32::INFINITY;
            c.mean_abs_diff = f32::INFINITY;
            c.root_mean_square_error = f32::INFINITY;
            c.cosine_similarity = 0.0;
            continue;
        }
        if r > best {
            best = r;
            c.reference_argmax = i;
        }
        if x > best_candidate {
            best_candidate = x;
            c.candidate_argmax = i;
        }
        let diff = (r - x).abs();
        c.max_abs_diff = c.max_abs_diff.max(diff);
        absolute += f64::from(diff);
        squared += f64::from(diff) * f64::from(diff);
        dot += f64::from(r) * f64::from(x);
        reference_norm += f64::from(r) * f64::from(r);
        candidate_norm += f64::from(x) * f64::from(x);
    }
    if c.finite && !reference.is_empty() {
        c.mean_abs_diff = (absolute / reference.len() as f64) as f32;
        c.root_mean_square_error = (squared / reference.len() as f64).sqrt() as f32;
        let norms = (reference_norm * candidate_norm).sqrt();
        c.cosine_similarity = if norms > 0.0 {
            (dot / norms) as f32
        } else if reference_norm == candidate_norm {
            1.0
        } else {
            0.0
        };
    }
    c
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

mod deepseek;

/// Qwen3.8-Flash-Next (`RunQwen38FlashNextBenchmark`): pp and tg at each
/// depth on a repeated sentence, C1, autoregressive or with MTP, optionally
/// after checking batched prefill against token-by-token evaluation.
mod flash_next {
    use std::path::Path;
    use std::process::ExitCode;
    use std::time::Instant;

    use gufo_core::sampling::SamplerState;
    use gufo_qwen38_flash_next::engine::{Model, ModelOptions};
    use gufo_qwen38_flash_next::executor::SessionMode;
    use gufo_qwen38_flash_next::weights::GgufFiles;
    use sha2::{Digest, Sha256};

    use super::{CliOptions, compare, left, load_time, right, stats, test_name};

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

    fn bench(o: &CliOptions) -> Result<(), String> {
        let load_start = Instant::now();
        if gufo_hip::device_count().unwrap_or(0) == 0 {
            load_time(load_start, false);
            return Err("Error: no HIP GPU is available for Qwen3.8-Flash-Next".into());
        }
        let max = |values: &[usize]| values.iter().copied().max().unwrap_or(0);
        let (max_depth, max_prompt, max_generation) =
            (max(&o.n_depths), max(&o.n_prompts), max(&o.n_gens));
        if max_depth
            .max(max_prompt)
            .max(max_generation)
            .max(o.validate_prefill_tokens)
            > u32::MAX as usize
        {
            return Err("Error: benchmark workload exceeds the context range".into());
        }
        let required = 4096
            .max(max_depth + max_prompt + 1)
            .max(max_depth.max(16) + max_generation + 1)
            .max(o.validate_prefill_tokens + 1);
        let mtp = o.speculative_backend == "mtp";
        if !o.speculative_backend.is_empty() && !mtp {
            return Err("Error: Qwen3.8-Flash-Next supports only --speculative mtp or off".into());
        }
        if mtp && o.mtp_model_path.is_empty() {
            return Err("Error: --speculative mtp requires --mtp-model".into());
        }
        if o.concurrency != [1] {
            return Err(
                "Error: Flash-Next bench supports C1; use the serving benchmark for concurrent requests"
                    .into(),
            );
        }
        if mtp && o.min_draft_tokens != 1 {
            return Err("Error: Flash-Next MTP requires --min-draft-tokens 1".into());
        }
        let Ok(context) = u32::try_from(required) else {
            return Err("Error: Flash-Next context is out of range".into());
        };
        let path = o.model_path.borrow().clone();
        let loaded = Model::load(
            Path::new(&path),
            ModelOptions {
                max_context: context,
                mtp_model_path: mtp.then(|| o.mtp_model_path.clone().into()),
                max_draft_tokens: o.draft_tokens.max(1),
                ..ModelOptions::default()
            },
        );
        let mut model = match loaded {
            Ok(model) => model,
            Err(error) => {
                load_time(load_start, false);
                return Err(format!("Error creating Qwen3.8-Flash-Next model: {error}"));
            }
        };
        load_time(load_start, true);
        let mode = if o.speculative_backend.is_empty() {
            SessionMode::Autoregressive
        } else {
            SessionMode::Speculative
        };
        // A repeated sentence times reproducibly; it touches far fewer PLE
        // rows than varied requests.
        let pattern = model.tokenize(
            "The quick brown fox jumps over the lazy dog. \
             Strix Halo executes this deterministic benchmark sequence. ",
        );
        if pattern.is_empty() {
            return Err("Error: benchmark token pattern is empty".into());
        }
        let tokens: Vec<i32> = (0..required).map(|i| pattern[i % pattern.len()]).collect();

        if o.validate_prefill_tokens != 0 {
            // Batched prefill against token-by-token evaluation of the same
            // prefix; both run on the GPU, so this checks the batched kernels.
            let prefix = &tokens[..o.validate_prefill_tokens];
            let failed = |e: String| format!("Qwen3.8-Flash-Next prefill validation failed: {e}");
            let mut sequential = model.create_session(mode, context).map_err(failed)?;
            let mut batched = model.create_session(mode, context).map_err(failed)?;
            model.sync(&mut sequential, &prefix[..1]).map_err(failed)?;
            model.sync(&mut batched, prefix).map_err(failed)?;
            for &token in &prefix[1..] {
                model
                    .evaluate(&mut sequential, token)
                    .map_err(|e| format!("Qwen3.8-Flash-Next sequential reference failed: {e}"))?;
            }
            let (reference, candidate) = (sequential.logits(), batched.logits());
            let c = compare(reference, candidate);
            let winner = candidate[c.reference_argmax];
            let rank = 1 + candidate.iter().filter(|&&v| v > winner).count();
            // The C++ command's stdout is in fixed notation with two
            // decimals by now (the model load leaves it so).
            println!(
                "[Prefill Validation] tokens={} finite={} scalar_winner_rank={rank} rmse={:.2} cosine={:.2} max_error={:.2}",
                prefix.len(),
                u8::from(c.finite),
                c.root_mean_square_error,
                c.cosine_similarity,
                c.max_abs_diff
            );
            if !c.finite
                || rank > 3
                || c.root_mean_square_error > 1.12
                || c.cosine_similarity < 0.979
                || c.max_abs_diff > 5.0
            {
                return Err(String::new());
            }
        }

        let bytes: u64 = GgufFiles::open(Path::new(&path))?
            .paths
            .iter()
            .map(|p| std::fs::metadata(p).map_or(0, |m| m.len()))
            .sum();
        let size = format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64);
        let name = model.name().to_string();
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
        let print_result = |test: &str, runs: &[f64]| {
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
            for &prompt in &o.n_prompts {
                if depth + prompt >= required {
                    return Err("Error: prompt benchmark exceeds context".into());
                }
                // A full aligned prefix already warms the chunk shapes and
                // GEMM plans; other shapes need an untimed pass first.
                let chunk = model.prefill_capacity() as usize;
                let warm_prefix = chunk > 0
                    && depth >= chunk
                    && depth.is_multiple_of(chunk)
                    && prompt > 0
                    && prompt.is_multiple_of(chunk);
                let mut runs = Vec::new();
                for repetition in usize::from(warm_prefix)..=o.repetitions {
                    let mut session = model
                        .create_session(mode, context)
                        .map_err(|e| format!("Error preparing depth: {e}"))?;
                    if depth > 0 {
                        model
                            .sync(&mut session, &tokens[..depth])
                            .map_err(|e| format!("Error preparing depth: {e}"))?;
                    }
                    let start = Instant::now();
                    model
                        .sync(&mut session, &tokens[..depth + prompt])
                        .map_err(|e| format!("Error running prefill: {e}"))?;
                    let seconds = start.elapsed().as_secs_f64();
                    if repetition > 0 {
                        runs.push(prompt as f64 / seconds);
                    }
                }
                print_result(&test_name("pp", prompt, depth), &runs);
            }
            for &generation in &o.n_gens {
                let prefix = if depth > 0 { depth } else { 16 };
                if prefix + generation >= required {
                    return Err("Error: generation benchmark exceeds context".into());
                }
                let mut runs = Vec::new();
                for _ in 0..o.repetitions {
                    let mut session = model
                        .create_session(mode, context)
                        .map_err(|e| format!("Error preparing generation: {e}"))?;
                    model
                        .sync(&mut session, &tokens[..prefix])
                        .map_err(|e| format!("Error preparing generation: {e}"))?;
                    let history: Vec<u32> = tokens[..prefix].iter().map(|&t| t as u32).collect();
                    let mut sampler = SamplerState::new(o.sampling.clone(), &history)
                        .map_err(|e| e.to_string())?;
                    let mut generated: Vec<i32> = Vec::new();
                    let start = Instant::now();
                    while generated.len() < generation {
                        let decoded = model
                            .decode_step(
                                &mut session,
                                generation - generated.len(),
                                &mut sampler,
                                false,
                            )
                            .map_err(|e| format!("Error running Flash-Next decode: {e}"))?;
                        if decoded.tokens.is_empty() {
                            return Err("Error running Flash-Next decode: ".into());
                        }
                        generated.extend_from_slice(&decoded.tokens);
                    }
                    runs.push(generation as f64 / start.elapsed().as_secs_f64());
                    if o.verbose {
                        let stats = session.statistics();
                        let bytes: Vec<u8> =
                            generated.iter().flat_map(|t| t.to_le_bytes()).collect();
                        let digest: String = Sha256::digest(&bytes)
                            .iter()
                            .map(|b| format!("{b:02x}"))
                            .collect();
                        let text = model.decode(&generated[..generated.len().min(48)]);
                        eprintln!(
                            "Qwen3.8-Flash-Next tg depth={depth} cycles={} drafted={} accepted={} lookup_proposed={} lookup_accepted={} output_sha256={digest} text={}",
                            stats.cycles,
                            stats.drafted,
                            stats.accepted,
                            stats.lookup,
                            stats.lookup_accepted,
                            String::from_utf8_lossy(&text)
                        );
                    }
                }
                print_result(&test_name("tg", generation, depth), &runs);
            }
        }
        println!();
        Ok(())
    }
}
