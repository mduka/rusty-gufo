//! Reference audio (`audio.hpp`): interleaved samples and the windowed-sinc
//! mono resampler of `ResampleMono`, computed in the same double-precision
//! order as the C++ code.

use std::collections::HashMap;

/// Interleaved float samples (`AudioBuffer`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub channels: u32,
    pub samples: Vec<f32>,
}

const RESAMPLER_RADIUS: i64 = 24;
const RESAMPLER_TAPS: usize = 2 * RESAMPLER_RADIUS as usize;

struct ResamplerPhase {
    weights: [f64; RESAMPLER_TAPS],
    offsets: [i64; RESAMPLER_TAPS],
    taps: usize,
    normalization: f64,
}

/// `ReflectIndex`: reflection without repeating the edge sample.
pub fn reflect_index(mut index: i64, length: i64) -> i64 {
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

/// `ResampleMono`: the channel mean, resampled with a Hann-windowed sinc of
/// radius 24 and reflected edges. Taps are computed once per fractional
/// phase (the first output frame with that phase) and reused.
pub fn resample_mono(audio: &AudioBuffer, output_sample_rate: u32) -> Result<Vec<f32>, String> {
    if audio.sample_rate == 0
        || audio.channels == 0
        || audio.samples.is_empty()
        || !audio.samples.len().is_multiple_of(audio.channels as usize)
        || output_sample_rate == 0
    {
        return Err("Qwen3-TTS audio buffer is invalid".into());
    }
    let channels = audio.channels as usize;
    let input_frames = audio.samples.len() / channels;
    let mono: Vec<f32> = audio
        .samples
        .chunks_exact(channels)
        .map(|frame| {
            let sum = frame.iter().fold(0.0f64, |sum, &s| sum + f64::from(s));
            (sum / f64::from(audio.channels)) as f32
        })
        .collect();
    if audio.sample_rate == output_sample_rate {
        return Ok(mono);
    }
    let output_frames = (input_frames as u64 * u64::from(output_sample_rate))
        .div_ceil(u64::from(audio.sample_rate)) as usize;
    let mut result = vec![0.0f32; output_frames];
    let scale = f64::from(audio.sample_rate) / f64::from(output_sample_rate);
    let cutoff = 1.0f64.min(f64::from(output_sample_rate) / f64::from(audio.sample_rate));
    let mut phases: HashMap<u64, ResamplerPhase> = HashMap::new();
    for (frame, sample) in result.iter_mut().enumerate() {
        let position = frame as f64 * scale;
        let center = position.floor() as i64;
        let fraction = position - center as f64;
        let phase_taps = phases.entry(fraction.to_bits()).or_insert_with(|| {
            let mut phase = ResamplerPhase {
                weights: [0.0; RESAMPLER_TAPS],
                offsets: [0; RESAMPLER_TAPS],
                taps: 0,
                normalization: 0.0,
            };
            for tap in -RESAMPLER_RADIUS + 1..=RESAMPLER_RADIUS {
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
                phase.offsets[phase.taps] = tap;
                phase.weights[phase.taps] = weight;
                phase.taps += 1;
                phase.normalization += weight;
            }
            phase
        });
        let mut weighted = 0.0f64;
        for tap in 0..phase_taps.taps {
            let reflected = reflect_index(center + phase_taps.offsets[tap], input_frames as i64);
            weighted += f64::from(mono[reflected as usize]) * phase_taps.weights[tap];
        }
        *sample = if phase_taps.normalization == 0.0 {
            0.0
        } else {
            (weighted / phase_taps.normalization) as f32
        };
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_mixdown_and_reflection() {
        let audio = AudioBuffer {
            sample_rate: 24000,
            channels: 2,
            samples: vec![1.0, 0.0, 0.5, 0.5],
        };
        assert_eq!(resample_mono(&audio, 24000).unwrap(), [0.5, 0.5]);
        assert_eq!(reflect_index(-2, 5), 2);
        assert_eq!(reflect_index(6, 5), 2);
        assert_eq!(reflect_index(7, 1), 0);
    }

    #[test]
    fn resampling_length_rounds_up() {
        let audio = AudioBuffer {
            sample_rate: 16000,
            channels: 1,
            samples: vec![0.25; 1001],
        };
        let resampled = resample_mono(&audio, 24000).unwrap();
        assert_eq!(resampled.len(), 1502);
        // A constant stays (nearly) constant away from numerical noise.
        assert!(resampled.iter().all(|&s| (s - 0.25).abs() < 1e-6));
    }
}
