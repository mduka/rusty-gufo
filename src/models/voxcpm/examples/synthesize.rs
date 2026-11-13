//! The development driver of the C++ extension (`tools/synthesize`) in
//! Rust, for comparisons with the C++ build: synthesizes TEXT into a 16-bit
//! mono WAV. `--raw FILE` also writes the last run's float32 samples
//! (little-endian), for bit-for-bit comparisons.
//!
//! cargo run --profile gpu-test -p gufo-voxcpm --example synthesize -- \
//!   MODEL OUTPUT.wav TEXT [--instruct TEXT] [--reference WAV \
//!   [--reference-text TEXT]] [--seed N] [--cfg G] [--timesteps N] \
//!   [--repeat N] [--stream] [--raw FILE]

use std::process::ExitCode;
use std::time::Instant;

use gufo_voxcpm::synthesis::{ReferenceVoice, SynthesisOptions, Synthesizer};

/// A PCM16 or float WAV as interleaved floats, its channels and rate (PCM16
/// scaled by 1/32768 like Gufo's WAV decoder).
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

/// A 16-bit mono WAV; samples are clamped and rounded like the C++ driver's
/// `std::lround(clamp(sample) * 32767)`.
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
        // static_cast<std::int16_t>(std::lround(...)): rounds half away
        // from zero.
        let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i64 as i16;
        out.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, out)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: synthesize MODEL OUTPUT.wav TEXT [--instruct TEXT] [--reference WAV \
             [--reference-text TEXT]] [--seed N] [--cfg G] [--timesteps N] [--repeat N] \
             [--stream] [--raw FILE]"
        );
        return ExitCode::from(2);
    }
    let mut options = SynthesisOptions {
        text: args[3].clone(),
        ..SynthesisOptions::default()
    };
    let (mut reference_path, mut reference_text, mut raw) =
        (String::new(), String::new(), String::new());
    let mut repeats = 1usize;
    let mut stream = false;
    let mut index = 4;
    while index < args.len() {
        let flag = args[index].as_str();
        if flag == "--stream" {
            stream = true;
            index += 1;
            continue;
        }
        let Some(value) = args.get(index + 1).cloned() else {
            eprintln!("{flag} needs a value");
            return ExitCode::from(2);
        };
        let parsed = match flag {
            "--instruct" => {
                options.instruct = value;
                Ok(())
            }
            "--reference" => {
                reference_path = value;
                Ok(())
            }
            "--reference-text" => {
                reference_text = value;
                Ok(())
            }
            "--raw" => {
                raw = value;
                Ok(())
            }
            "--seed" => value.parse().map(|v| options.seed = v).map_err(|_| ()),
            "--cfg" => value.parse().map(|v| options.cfg = v).map_err(|_| ()),
            "--timesteps" => value.parse().map(|v| options.timesteps = v).map_err(|_| ()),
            "--repeat" => value.parse().map(|v| repeats = v).map_err(|_| ()),
            _ => {
                eprintln!("unknown option {flag}");
                return ExitCode::from(2);
            }
        };
        if parsed.is_err() {
            eprintln!("{flag}: invalid value");
            return ExitCode::from(2);
        }
        index += 2;
    }

    let start = Instant::now();
    let mut synthesizer = match Synthesizer::load(std::path::Path::new(&args[1])) {
        Ok(synthesizer) => synthesizer,
        Err(error) => {
            eprintln!("load failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("loaded in {} s", start.elapsed().as_secs_f64());
    let voice: Option<ReferenceVoice> = if reference_path.is_empty() {
        None
    } else {
        let (recording, channels, rate) = match read_wav(&reference_path) {
            Ok(wav) => wav,
            Err(error) => {
                eprintln!("reference failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        let start = Instant::now();
        match synthesizer.clone_voice(&recording, channels, rate, &reference_text) {
            Ok(voice) => {
                eprintln!("reference encoded in {} s", start.elapsed().as_secs_f64());
                Some(voice)
            }
            Err(error) => {
                eprintln!("reference failed: {error}");
                return ExitCode::FAILURE;
            }
        }
    };
    options.voice = voice.as_ref();

    let rate = synthesizer.sample_rate();
    let mut audio = Vec::new();
    for run in 0..repeats {
        audio.clear();
        let start = Instant::now();
        let mut first = 0.0f64;
        let mut streamed = Vec::new();
        let result = if stream {
            let sink: &mut dyn FnMut(&[f32]) -> bool = &mut |samples: &[f32]| {
                if streamed.is_empty() {
                    first = start.elapsed().as_secs_f64();
                }
                streamed.extend_from_slice(samples);
                true
            };
            synthesizer.synthesize(&options, None, Some(sink), Some(&mut audio))
        } else {
            synthesizer.synthesize(&options, None, None, Some(&mut audio))
        };
        if let Err(error) = result {
            eprintln!("synthesis failed: {error}");
            return ExitCode::FAILURE;
        }
        if stream {
            audio = streamed;
        }
        let seconds = start.elapsed().as_secs_f64();
        let length = audio.len() as f64 / f64::from(rate);
        eprint!(
            "run {run}: {length} s audio in {seconds} s (RTF {})",
            seconds / length.max(1e-9)
        );
        if stream {
            eprint!(", first audio after {first} s");
        }
        eprintln!();
    }
    if let Err(error) = write_wav(&args[2], &audio, rate) {
        eprintln!("cannot write {}: {error}", args[2]);
        return ExitCode::FAILURE;
    }
    if !raw.is_empty() {
        let bytes: Vec<u8> = audio.iter().flat_map(|v| v.to_le_bytes()).collect();
        if let Err(error) = std::fs::write(&raw, bytes) {
            eprintln!("cannot write {raw}: {error}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
