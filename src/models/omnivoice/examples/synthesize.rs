//! The development driver of the C++ extension (`tools/synthesize`) in
//! Rust, for comparisons with the C++ build: synthesizes TEXT into a 16-bit
//! WAV. With `GUFO_REF_CODES=FILE` it writes the last run's codes (frame
//! major, space separated) as the reference copy of the C++ driver does.
//!
//! cargo run --profile gpu-test -p gufo-omnivoice --example synthesize -- \
//!   MODEL_DIR OUTPUT.wav TEXT [--language L] [--instruct ITEMS] \
//!   [--reference WAV --reference-text TEXT] [--seed N] [--greedy] [--repeat N]

use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use gufo_omnivoice::synthesis::{SynthesisOptions, Synthesizer};

/// A PCM16 or float WAV as interleaved floats, its channels and rate.
fn read_wav(path: &str) -> Result<(Vec<f32>, u32, u32), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let le16 = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
    let le32 =
        |at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{path}: not a WAV file"));
    }
    let (mut format, mut channels, mut rate, mut bits) = (0u16, 0u32, 0u32, 0u16);
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let size = le32(at + 4) as usize;
        let body = at + 8;
        if body + size > bytes.len() {
            break;
        }
        match &bytes[at..at + 4] {
            b"fmt " if size >= 16 => {
                format = le16(body);
                channels = u32::from(le16(body + 2));
                rate = le32(body + 4);
                bits = le16(body + 14);
            }
            b"data" => {
                let data = &bytes[body..body + size];
                let samples = match (format, bits) {
                    (1, 16) => data
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|s| f32::from(i16::from_le_bytes(*s)) / 32768.0)
                        .collect(),
                    (3, 32) => data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|s| f32::from_le_bytes(*s))
                        .collect(),
                    _ => return Err(format!("{path}: unsupported WAV encoding")),
                };
                return Ok((samples, channels, rate));
            }
            _ => {}
        }
        at = body + size + (size & 1);
    }
    Err(format!("{path}: no data chunk"))
}

fn write_wav(path: &str, samples: &[f32], rate: u32) -> std::io::Result<()> {
    let data_bytes = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &sample in samples {
        // static_cast<std::int16_t>(clamp(sample) * 32767.0F) truncates.
        let value = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, out)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: synthesize MODEL_DIR OUTPUT.wav TEXT [--language L] [--instruct ITEMS] \
             [--reference WAV --reference-text TEXT] [--seed N] [--greedy] [--repeat N]"
        );
        return ExitCode::from(2);
    }
    let mut options = SynthesisOptions {
        text: args[3].clone(),
        ..SynthesisOptions::default()
    };
    let (mut reference_path, mut reference_text) = (String::new(), String::new());
    let mut repeats = 1usize;
    let mut index = 4;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = args.get(index + 1).cloned();
        match (flag, value) {
            ("--greedy", _) => {
                options.decoding.position_temperature = 0.0;
                index += 1;
                continue;
            }
            ("--language", Some(v)) => options.language = v,
            ("--instruct", Some(v)) => options.instruct = v,
            ("--reference", Some(v)) => reference_path = v,
            ("--reference-text", Some(v)) => reference_text = v,
            ("--repeat", Some(v)) => repeats = v.parse::<usize>().unwrap_or(1).max(1),
            ("--seed", Some(v)) => options.seed = v.parse().unwrap_or(0),
            _ => {
                eprintln!("unknown option {flag}");
                return ExitCode::from(2);
            }
        }
        index += 2;
    }
    let start = Instant::now();
    let mut synthesizer = match Synthesizer::load(std::path::Path::new(&args[1])) {
        Ok(s) => s,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    println!("load {} s", start.elapsed().as_secs_f64());
    let voice;
    if !reference_path.is_empty() {
        let (recording, channels, rate) = match read_wav(&reference_path) {
            Ok(wav) => wav,
            Err(error) => {
                eprintln!("cannot read {reference_path}: {error}");
                return ExitCode::FAILURE;
            }
        };
        let start = Instant::now();
        voice = match synthesizer.clone_voice(&recording, channels, rate, &reference_text) {
            Ok(voice) => voice,
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::FAILURE;
            }
        };
        println!(
            "reference: {} frames, encoded in {} s",
            voice.frames,
            start.elapsed().as_secs_f64()
        );
        options.reference = Some(&voice);
    }
    let mut synthesis = gufo_omnivoice::synthesis::Synthesis::default();
    for run in 0..repeats {
        let start = Instant::now();
        synthesis = match synthesizer.synthesize(&options, &|| false) {
            Ok(result) => result,
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::FAILURE;
            }
        };
        let seconds = start.elapsed().as_secs_f64();
        let duration = synthesis.audio.len() as f64 / f64::from(synthesizer.sample_rate());
        println!(
            "run {}: synthesis {seconds} s for {duration} s of audio (RTF {})",
            run + 1,
            seconds / duration
        );
    }
    if let Some(path) = std::env::var_os("GUFO_REF_CODES") {
        let line: String = synthesis.codes.iter().map(|c| format!("{c} ")).collect();
        if let Err(error) = std::fs::File::create(path).and_then(|mut f| writeln!(f, "{line}")) {
            eprintln!("cannot write codes: {error}");
            return ExitCode::FAILURE;
        }
    }
    if let Err(error) = write_wav(&args[2], &synthesis.audio, synthesizer.sample_rate()) {
        eprintln!("cannot write {}: {error}", args[2]);
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
