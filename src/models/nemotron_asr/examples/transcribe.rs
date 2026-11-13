//! The development driver of the C++ extension (`tools/transcribe`) in
//! Rust, for comparisons with the C++ build: prints the transcript, the
//! detected language and the time per run. The audio is a 16 kHz WAV
//! (Gufo 0.34.0's strict WAV reader, mixed to mono).
//!
//! cargo run --profile gpu-test -p gufo-nemotron-asr --example transcribe -- \
//!   MODEL AUDIO [--language L] [--lookahead N] [--repeat N] [--push MS]

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use gufo_nemotron_asr::language::resolve_language;
use gufo_nemotron_asr::model::{MAX_PASS_FRAMES, Model, Utterance};

fn read_wav(path: &str) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let limits = gufo_wav_decode::Limits {
        max_channels: 8,
        min_sample_rate: 1,
        max_sample_rate: 192_000,
        max_seconds: 4 * 3600,
        max_pcm_bits: 32,
    };
    let wav = gufo_wav_decode::parse(&bytes, &limits).map_err(|e| format!("{path}: {e:?}"))?;
    if wav.sample_rate != 16_000 {
        return Err(format!("{path}: the driver reads 16 kHz audio"));
    }
    let mut samples = vec![0.0f32; wav.samples()];
    wav.decode(0, &mut samples)
        .map_err(|e| format!("{path}: {e:?}"))?;
    let channels = wav.channels as usize;
    Ok(samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().fold(0.0f32, |sum, &s| sum + s) / channels as f32)
        .collect())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err(
            "usage: transcribe MODEL AUDIO [--language L] [--lookahead N] [--repeat N] [--push MS]"
                .into(),
        );
    }
    let (mut language, mut lookahead, mut repeat, mut push_ms) = (None, None, 1, 0);
    for pair in args[2..].chunks(2) {
        let value = pair
            .get(1)
            .ok_or_else(|| format!("{} needs a value", pair[0]))?;
        let number = || {
            value
                .parse::<usize>()
                .map_err(|e| format!("{}: {e}", pair[0]))
        };
        match pair[0].as_str() {
            "--language" => language = Some(value.clone()),
            "--lookahead" => lookahead = Some(number()?),
            "--repeat" => repeat = number()?,
            "--push" => push_ms = number()?,
            flag => return Err(format!("unknown option {flag}")),
        }
    }
    let load = Instant::now();
    let model = Model::load(Path::new(&args[0]))?;
    println!("loaded in {:.6} s", load.elapsed().as_secs_f64());
    let samples = read_wav(&args[1])?;
    let prompt = resolve_language(model.config(), language.as_deref())?;
    let look = lookahead.unwrap_or(model.max_lookahead());
    let seconds = samples.len() as f64 / 16_000.0;
    for run in 0..repeat {
        let begin = Instant::now();
        let mut utterance =
            Utterance::new(&model, look, prompt, MAX_PASS_FRAMES).map_err(|e| e.to_string())?;
        if push_ms == 0 {
            utterance.push(&samples, false).map_err(|e| e.to_string())?;
        } else {
            for piece in samples.chunks(push_ms * 16) {
                utterance.push(piece, true).map_err(|e| e.to_string())?;
            }
        }
        utterance.finish().map_err(|e| e.to_string())?;
        let elapsed = begin.elapsed().as_secs_f64();
        println!(
            "run {run}: {seconds} s audio in {elapsed:.6} s (RTF {:.6}), encoder {:.4} ms, decoder {:.4} ms, features {:.4} ms, {} chunks, {} tokens",
            elapsed / seconds,
            utterance.encoder_ms(),
            utterance.decoder_ms(),
            utterance.feature_ms(),
            utterance.chunks(),
            utterance.text().ids().len()
        );
        if run + 1 == repeat {
            println!(
                "language: {}",
                utterance.text().language().unwrap_or("(none)")
            );
            println!("{}", utterance.text().text());
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
