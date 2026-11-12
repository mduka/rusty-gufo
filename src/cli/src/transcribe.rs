//! `gufo transcribe`: options, output and JSON report of the C++ command
//! (`src/cli/transcribe/transcribe.cpp`), plus the generated token IDs.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use gufo_qwen3_asr::{Audio, Request, Timings, Transcriber};
use serde_json::{Value, json};

use crate::arg_parser::ArgParser;

const MAXIMUM_AUDIO_BYTES: u64 = 256 << 20;

struct Options {
    model: PathBuf,
    audio: PathBuf,
    language: String,
    context: String,
    max_tokens: usize,
    capacity: usize,
    repeat: usize,
    warmup: usize,
    format: String,
    positionals: Vec<String>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            audio: PathBuf::new(),
            language: String::new(),
            context: String::new(),
            max_tokens: 256,
            capacity: 1024,
            repeat: 1,
            warmup: 0,
            format: "text".into(),
            positionals: Vec::new(),
        }
    }
}

/// The option table of the C++ command.
fn parser(options: &mut Options) -> ArgParser<'_> {
    let Options {
        model,
        audio,
        language,
        context,
        max_tokens,
        capacity,
        repeat,
        warmup,
        format,
        positionals,
    } = options;
    let mut parser = ArgParser::new(
        "gufo transcribe [OPTIONS] [WAV]",
        "Transcribe a PCM or float WAV with native HIP Qwen3-ASR-1.7B.",
    );
    parser
        .add_option(
            "-m",
            "--model",
            "DIR",
            "Qwen3-ASR-1.7B model directory",
            "Model",
            model,
        )
        .add_option("-f", "--audio", "WAV", "Input WAV path", "Input", audio)
        .add_option(
            "-l",
            "--language",
            "NAME",
            "Optional language name or ISO code; auto-detect by default",
            "Input",
            language,
        )
        .add_option(
            "",
            "--prompt",
            "TEXT",
            "Optional transcription context",
            "Input",
            context,
        )
        .add_option(
            "-n",
            "--max-tokens",
            "N",
            "Maximum generated tokens per audio chunk (default: 256)",
            "Generation",
            max_tokens,
        )
        .add_option(
            "-c",
            "--context",
            "N",
            "Prompt plus generation capacity (default: 1024)",
            "Generation",
            capacity,
        )
        .add_option(
            "",
            "--repeat",
            "N",
            "Measured transcriptions with one loaded runtime (default: 1)",
            "Benchmark",
            repeat,
        )
        .add_option(
            "",
            "--warmup",
            "N",
            "Discarded warmup transcriptions (default: 0)",
            "Benchmark",
            warmup,
        )
        .add_option(
            "",
            "--format",
            "NAME",
            "Output format: text or json (default: text)",
            "Output",
            format,
        )
        .collect_positionals(positionals);
    parser
}

/// `PrintTranscribeHelp`.
pub fn print_help() {
    parser(&mut Options::default()).print_help();
}

/// The options, `None` after the help, or the exit code of a usage error.
fn options(args: &[String]) -> Result<Option<Options>, ExitCode> {
    let mut options = Options::default();
    let (parsed, help) = {
        let mut parser = parser(&mut options);
        (parser.parse(args), parser.is_help_requested())
    };
    if let Err(error) = parsed {
        eprintln!("Error: {error}");
        print_help();
        return Err(ExitCode::from(2));
    }
    if help {
        print_help();
        return Ok(None);
    }
    let usage = |message: &str| {
        eprintln!("Error: {message}");
        Err(ExitCode::from(2))
    };
    if options.model.as_os_str().is_empty() {
        return usage("--model <DIR> is required");
    }
    if options.audio.as_os_str().is_empty() && options.positionals.len() == 1 {
        options.audio = PathBuf::from(&options.positionals[0]);
    } else if !options.positionals.is_empty() {
        return usage("supply exactly one WAV path");
    }
    if options.audio.as_os_str().is_empty()
        || options.max_tokens == 0
        || options.capacity < 32
        || options.max_tokens >= options.capacity
        || options.repeat == 0
        || options.repeat > 100
        || options.warmup > 100
        || (options.format != "text" && options.format != "json")
    {
        return usage("invalid audio, token capacity, or output format");
    }
    Ok(Some(options))
}

fn read_audio(path: &PathBuf) -> Result<Vec<u8>, String> {
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if size == 0 || size > MAXIMUM_AUDIO_BYTES {
        return Err("audio file must be between 1 byte and 256 MiB".into());
    }
    std::fs::read(path).map_err(|e| format!("cannot read audio file {}: {e}", path.display()))
}

fn timings_json(timings: &Timings, load_ms: f64) -> Value {
    json!({
        "load_ms": load_ms,
        "decode_audio_ms": timings.decode_audio_ms,
        "feature_extraction_ms": timings.feature_extraction_ms,
        "audio_encoder_ms": timings.audio_encoder_ms,
        "text_decoder_ms": timings.text_decoder_ms,
        "total_ms": timings.total_ms,
    })
}

pub fn run(args: &[String]) -> ExitCode {
    let options = match options(args) {
        Ok(Some(options)) => options,
        Ok(None) => return ExitCode::SUCCESS,
        Err(code) => return code,
    };
    let wav = match read_audio(&options.audio) {
        Ok(wav) => wav,
        Err(error) => {
            eprintln!("Error: {error}");
            return ExitCode::FAILURE;
        }
    };
    let begin = Instant::now();
    let transcriber = match Transcriber::load(&options.model, options.capacity) {
        Ok(transcriber) => transcriber,
        Err(error) => {
            eprintln!("Error loading Qwen3-ASR: {error}");
            return ExitCode::FAILURE;
        }
    };
    let load_ms = begin.elapsed().as_secs_f64() * 1e3;

    let request = Request {
        audio: Audio::Wav(&wav),
        context: &options.context,
        language: (!options.language.is_empty()).then_some(options.language.as_str()),
        max_new_tokens: options.max_tokens,
    };
    // Ctrl+C cancels the runs between phases and layers and exits with 130,
    // as the C++ command's SIGINT handler does. The handler stays: signal-hook
    // cannot restore the default action, and only the report follows.
    let interrupted = Arc::new(AtomicBool::new(false));
    if let Err(error) =
        signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(&interrupted))
    {
        eprintln!("Error transcribing audio: cannot handle SIGINT: {error}");
        return ExitCode::FAILURE;
    }
    let cancelled = || interrupted.load(Ordering::Relaxed);
    let mut first_text = None;
    let mut last = None;
    let mut runs = Vec::with_capacity(options.repeat);
    for run in 0..options.warmup + options.repeat {
        let transcribed = match transcriber.transcribe(&request, &cancelled, None) {
            Ok(transcribed) => transcribed,
            Err(error) => {
                eprintln!("Error transcribing audio: {error}");
                return if cancelled() {
                    ExitCode::from(130)
                } else {
                    ExitCode::FAILURE
                };
            }
        };
        if run < options.warmup {
            continue;
        }
        let expected = first_text.get_or_insert_with(|| transcribed.text.clone());
        if *expected != transcribed.text {
            eprintln!(
                "Error transcribing audio: repeated Qwen3-ASR transcription changed its transcript"
            );
            return ExitCode::FAILURE;
        }
        runs.push(transcribed.timings);
        last = Some(transcribed);
    }
    // Reported like the C++ command: the last measured run.
    let Some(result) = last else {
        return ExitCode::FAILURE;
    };
    if options.format != "json" {
        println!("{}", result.text);
        return ExitCode::SUCCESS;
    }
    let total_ms: f64 = runs.iter().map(|t| t.total_ms).sum();
    let report = json!({
        "schema": "gufo.qwen3-asr.transcription.v1",
        "backend": "native-hip",
        "language": result.language,
        "text": result.text,
        "audio_samples": result.audio_samples,
        "sample_rate": result.sample_rate,
        "mel_frames": result.mel_frames,
        "audio_tokens": result.audio_tokens,
        "prompt_tokens": result.prompt_tokens,
        "generated_tokens": result.generated_ids.len(),
        "generated_ids": result.generated_ids,
        "chunks": result.chunks,
        "timings": timings_json(&result.timings, load_ms),
        "runs": runs.iter().map(|t| timings_json(t, 0.0)).collect::<Vec<_>>(),
        "benchmark": {
            "warmup": options.warmup,
            "repeat": options.repeat,
            "mean_total_ms": total_ms / runs.len() as f64,
        },
    });
    println!("{report}");
    ExitCode::SUCCESS
}
