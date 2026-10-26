//! The Qwen3-ASR model contract the serving layer needs
//! (`src/models/qwen3_asr/transcription.hpp`, `ResampleMono16k` of
//! `audio.cpp`). Transcription is injected: the Rust engine
//! (`gufo-qwen3-asr`) is wired in by the binary, since this crate has no GPU
//! dependency.

use std::collections::HashMap;

/// Sample rate of the model frontend.
pub const AUDIO_SAMPLE_RATE: u32 = 16_000;

/// `TranscriptionRequest`. Exactly one audio source is set: WAV bytes, or
/// normalized mono audio at 16 kHz for live sessions. The C++ `on_text`
/// callback is a separate argument of the runner.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionRequest {
    pub wav: Vec<u8>,
    pub context: String,
    pub language: Option<String>,
    pub max_new_tokens: usize,
    pub pcm16k: Vec<f32>,
}

impl Default for TranscriptionRequest {
    fn default() -> Self {
        Self {
            wav: Vec::new(),
            context: String::new(),
            language: None,
            max_new_tokens: 256,
            pcm16k: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TranscriptionTimings {
    pub decode_audio_ms: f64,
    pub feature_extraction_ms: f64,
    pub audio_encoder_ms: f64,
    pub text_decoder_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionResult {
    pub language: String,
    pub text: String,
    pub decoded: String,
    pub generated_ids: Vec<u32>,
    pub sample_rate: u32,
    pub audio_samples: usize,
    pub mel_frames: usize,
    pub audio_tokens: usize,
    pub prompt_tokens: usize,
    pub chunks: usize,
    pub timings: TranscriptionTimings,
}

impl Default for TranscriptionResult {
    fn default() -> Self {
        Self {
            language: String::new(),
            text: String::new(),
            decoded: String::new(),
            generated_ids: Vec::new(),
            sample_rate: AUDIO_SAMPLE_RATE,
            audio_samples: 0,
            mel_frames: 0,
            audio_tokens: 0,
            prompt_tokens: 0,
            chunks: 0,
            timings: TranscriptionTimings::default(),
        }
    }
}

/// Receives the cumulative transcript during generation; returning false
/// cancels.
pub type TextSink<'a> = &'a mut dyn FnMut(&str) -> bool;

/// The transcription entry point (`AsrServiceOptions::Runner`): the request,
/// an optional progress sink and a cancellation check polled during
/// generation. The error is a human-readable message.
pub type AsrRunner = Box<
    dyn Fn(
            &TranscriptionRequest,
            Option<TextSink<'_>>,
            &dyn Fn() -> bool,
        ) -> Result<TranscriptionResult, String>
        + Send
        + Sync,
>;

/// Interleaved float audio.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub channels: u32,
    pub samples: Vec<f32>,
}

const RESAMPLER_RADIUS: i64 = 24;

struct ResamplerPhase {
    weights: Vec<f64>,
    offsets: Vec<i64>,
    normalization: f64,
}

fn reflect_index(mut index: i64, length: i64) -> i64 {
    if length <= 1 {
        return 0;
    }
    while index < 0 || index >= length {
        index = if index < 0 {
            -index
        } else {
            2 * length - index - 2
        };
    }
    index
}

/// `ResampleMono16k`: mixes to mono and resamples to 16 kHz with a
/// Hann-windowed sinc of radius 24, one tap set per distinct fractional
/// phase; 16 kHz input is copied exactly.
pub fn resample_mono_16k(audio: &AudioBuffer) -> Result<Vec<f32>, String> {
    let channels = audio.channels as usize;
    if audio.sample_rate == 0
        || channels == 0
        || audio.samples.is_empty()
        || !audio.samples.len().is_multiple_of(channels)
    {
        return Err("Qwen3-ASR audio buffer is invalid".into());
    }
    let input_frames = audio.samples.len() / channels;
    let mono: Vec<f32> = audio
        .samples
        .chunks_exact(channels)
        .map(|frame| {
            let sum = frame.iter().fold(0.0, |sum, &s| sum + f64::from(s));
            (sum / f64::from(audio.channels)) as f32
        })
        .collect();
    if audio.sample_rate == AUDIO_SAMPLE_RATE {
        return Ok(mono);
    }
    let output_frames = (input_frames as u64 * u64::from(AUDIO_SAMPLE_RATE))
        .div_ceil(u64::from(audio.sample_rate)) as usize;
    let scale = f64::from(audio.sample_rate) / f64::from(AUDIO_SAMPLE_RATE);
    let cutoff = (f64::from(AUDIO_SAMPLE_RATE) / f64::from(audio.sample_rate)).min(1.0);
    let mut phases: HashMap<u64, ResamplerPhase> = HashMap::new();
    let mut result = Vec::with_capacity(output_frames);
    for frame in 0..output_frames {
        let position = frame as f64 * scale;
        let center = position.floor() as i64;
        let fraction = position - center as f64;
        let phase_taps = phases.entry(fraction.to_bits()).or_insert_with(|| {
            let mut phase = ResamplerPhase {
                weights: Vec::new(),
                offsets: Vec::new(),
                normalization: 0.0,
            };
            for tap in (-RESAMPLER_RADIUS + 1)..=RESAMPLER_RADIUS {
                let distance = position - (center + tap) as f64;
                let window_position = distance.abs() / RESAMPLER_RADIUS as f64;
                if window_position >= 1.0 {
                    continue;
                }
                let angle = std::f64::consts::PI * cutoff * distance;
                let sinc = if angle == 0.0 {
                    1.0
                } else {
                    angle.sin() / angle
                };
                let window = 0.5 + 0.5 * (std::f64::consts::PI * window_position).cos();
                let weight = cutoff * sinc * window;
                phase.offsets.push(tap);
                phase.weights.push(weight);
                phase.normalization += weight;
            }
            phase
        });
        let mut weighted = 0.0;
        for (offset, weight) in phase_taps.offsets.iter().zip(&phase_taps.weights) {
            let reflected = reflect_index(center + offset, input_frames as i64);
            weighted += f64::from(mono[reflected as usize]) * weight;
        }
        result.push(if phase_taps.normalization == 0.0 {
            0.0
        } else {
            (weighted / phase_taps.normalization) as f32
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_keeps_16k_and_converts_24k() {
        let same = AudioBuffer {
            sample_rate: 16_000,
            channels: 2,
            samples: vec![1.0, 0.0, 0.5, 0.5],
        };
        assert_eq!(resample_mono_16k(&same).expect("resample"), [0.5, 0.5]);
        let constant = AudioBuffer {
            sample_rate: 24_000,
            channels: 1,
            samples: vec![0.25; 2400],
        };
        let output = resample_mono_16k(&constant).expect("resample");
        assert_eq!(output.len(), 1600);
        assert!(output.iter().all(|&s| (s - 0.25).abs() < 1e-6));
        assert!(resample_mono_16k(&AudioBuffer::default()).is_err());
    }
}
