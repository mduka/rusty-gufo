//! CPU audio frontend: WAV decoding, mono 16 kHz resampling, log-mel features
//! and chunking. Every floating-point operation keeps the order and precision
//! of the C++ frontend (`audio.cpp`), so features are bit-identical.

use std::collections::HashMap;
use std::f64::consts::PI;

pub const SAMPLE_RATE: u32 = 16_000;
pub const MEL_BINS: usize = 128;
pub const FFT_SIZE: usize = 400;
pub const HOP_LENGTH: usize = 160;

const MAXIMUM_CHANNELS: u32 = 8;
const MAXIMUM_SAMPLE_RATE: u32 = 192_000;
const MAXIMUM_SECONDS: u64 = 30 * 60;
const FREQUENCY_BINS: usize = FFT_SIZE / 2 + 1;
const CENTER_PADDING: usize = FFT_SIZE / 2;
const CHUNK_FRAMES: usize = 100;
const CHUNK_TOKENS: usize = 13;
const RESAMPLER_RADIUS: i64 = 24;

/// Interleaved float samples.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub channels: u32,
    pub samples: Vec<f32>,
}

/// Mel-major `[MEL_BINS, frames]`, matching `WhisperFeatureExtractor`.
#[derive(Debug, Clone, PartialEq)]
pub struct LogMelFeatures {
    pub values: Vec<f32>,
    pub frames: usize,
}

/// A contiguous range of 16 kHz samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioChunk {
    pub offset: usize,
    pub samples: usize,
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Decodes a RIFF WAV with PCM16/24/32 or float32 interleaved samples.
pub fn decode_wav(bytes: &[u8]) -> Result<AudioBuffer, String> {
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Qwen3-ASR input must be a RIFF WAV".into());
    }
    let (mut format, mut channels, mut sample_rate) = (0u16, 0u16, 0u32);
    let (mut block_align, mut bits_per_sample) = (0u16, 0u16);
    let mut data: &[u8] = &[];
    let mut offset = 12usize;
    while offset + 8 <= bytes.len() {
        let chunk_size = u32_at(bytes, offset + 4) as usize;
        let data_offset = offset + 8;
        if chunk_size > bytes.len() - data_offset {
            return Err("Qwen3-ASR WAV is truncated".into());
        }
        match &bytes[offset..offset + 4] {
            b"fmt " => {
                if chunk_size < 16 {
                    return Err("Qwen3-ASR WAV fmt chunk is invalid".into());
                }
                format = u16_at(bytes, data_offset);
                channels = u16_at(bytes, data_offset + 2);
                sample_rate = u32_at(bytes, data_offset + 4);
                block_align = u16_at(bytes, data_offset + 12);
                bits_per_sample = u16_at(bytes, data_offset + 14);
            }
            b"data" => data = &bytes[data_offset..data_offset + chunk_size],
            _ => {}
        }
        let padded = chunk_size + (chunk_size & 1);
        if padded > bytes.len() - data_offset {
            break;
        }
        offset = data_offset + padded;
    }

    if (format != 1 && format != 3)
        || channels == 0
        || u32::from(channels) > MAXIMUM_CHANNELS
        || sample_rate == 0
        || sample_rate > MAXIMUM_SAMPLE_RATE
        || data.is_empty()
    {
        return Err("Qwen3-ASR WAV format is unsupported".into());
    }
    let bytes_per_sample = usize::from(bits_per_sample / 8);
    if (format == 1 && !matches!(bits_per_sample, 16 | 24 | 32))
        || (format == 3 && bits_per_sample != 32)
        || usize::from(block_align) != usize::from(channels) * bytes_per_sample
        || !data.len().is_multiple_of(usize::from(block_align))
    {
        return Err("Qwen3-ASR supports PCM16/24/32 or float32 interleaved WAV".into());
    }
    let frames = (data.len() / usize::from(block_align)) as u64;
    if frames == 0 || frames > MAXIMUM_SECONDS * u64::from(sample_rate) {
        return Err("Qwen3-ASR WAV must be at most 30 minutes".into());
    }

    let count = frames as usize * usize::from(channels);
    let mut samples = Vec::with_capacity(count);
    for index in 0..count {
        let at = index * bytes_per_sample;
        let value = match (format, bits_per_sample) {
            (1, 16) => f32::from(u16_at(data, at) as i16) / 32768.0f32,
            (1, 24) => {
                let mut pcm = i32::from(data[at])
                    | (i32::from(data[at + 1]) << 8)
                    | (i32::from(data[at + 2]) << 16);
                if pcm & 0x0080_0000 != 0 {
                    pcm |= 0xFF00_0000u32 as i32;
                }
                pcm as f32 / 8_388_608.0f32
            }
            (1, _) => u32_at(data, at) as i32 as f32 / 2_147_483_648.0f32,
            _ => {
                let value = f32::from_bits(u32_at(data, at));
                if !value.is_finite() {
                    return Err("Qwen3-ASR WAV contains non-finite samples".into());
                }
                value.clamp(-1.0, 1.0)
            }
        };
        samples.push(value);
    }
    Ok(AudioBuffer {
        sample_rate,
        channels: u32::from(channels),
        samples,
    })
}

/// `index` reflected into `0..length` (numpy "reflect" padding).
fn reflect(mut index: i64, length: i64) -> i64 {
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

/// One windowed-sinc tap set for a fractional output phase.
struct ResamplerPhase {
    offsets: Vec<i64>,
    weights: Vec<f64>,
    normalization: f64,
}

/// Mixes to mono and resamples to 16 kHz. Input already at 16 kHz is copied
/// exactly, preserving the official-reference waveform.
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
            let sum = frame.iter().fold(0.0f64, |sum, &s| sum + f64::from(s));
            (sum / f64::from(audio.channels)) as f32
        })
        .collect();
    if audio.sample_rate == SAMPLE_RATE {
        return Ok(mono);
    }

    let rate = u64::from(audio.sample_rate);
    let output_frames = (input_frames as u64 * u64::from(SAMPLE_RATE)).div_ceil(rate) as usize;
    let scale = f64::from(audio.sample_rate) / f64::from(SAMPLE_RATE);
    let cutoff = min_f64(1.0, f64::from(SAMPLE_RATE) / f64::from(audio.sample_rate));
    // Weights depend only on the fractional phase; each distinct phase is
    // computed from the first output frame that has it, as in the C++ code.
    let mut phases: HashMap<u64, ResamplerPhase> = HashMap::new();
    let mut result = Vec::with_capacity(output_frames);
    for frame in 0..output_frames {
        let position = frame as f64 * scale;
        let center = position.floor() as i64;
        let fraction = position - center as f64;
        let phase = phases.entry(fraction.to_bits()).or_insert_with(|| {
            let mut taps = ResamplerPhase {
                offsets: Vec::with_capacity(2 * RESAMPLER_RADIUS as usize),
                weights: Vec::with_capacity(2 * RESAMPLER_RADIUS as usize),
                normalization: 0.0,
            };
            for tap in -RESAMPLER_RADIUS + 1..=RESAMPLER_RADIUS {
                let distance = position - (center + tap) as f64;
                let window_position = distance.abs() / RESAMPLER_RADIUS as f64;
                if window_position >= 1.0 {
                    continue;
                }
                let angle = PI * cutoff * distance;
                let sinc = if angle == 0.0 {
                    1.0
                } else {
                    angle.sin() / angle
                };
                let window = 0.5 + 0.5 * (PI * window_position).cos();
                let weight = cutoff * sinc * window;
                taps.offsets.push(tap);
                taps.weights.push(weight);
                taps.normalization += weight;
            }
            taps
        });
        let mut weighted = 0.0f64;
        for (&tap, &weight) in phase.offsets.iter().zip(&phase.weights) {
            let source = reflect(center + tap, input_frames as i64) as usize;
            weighted += f64::from(mono[source]) * weight;
        }
        result.push(if phase.normalization == 0.0 {
            0.0
        } else {
            (weighted / phase.normalization) as f32
        });
    }
    Ok(result)
}

/// `std::min` for doubles: `b < a ? b : a`.
fn min_f64(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// `std::max` for doubles: `a < b ? b : a`.
fn max_f64(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

/// `std::max` for floats.
fn max_f32(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}

fn hertz_to_slaney_mel(hertz: f64) -> f64 {
    const MIN_LOG_HERTZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let log_step = 27.0 / 6.4f64.ln();
    if hertz >= MIN_LOG_HERTZ {
        MIN_LOG_MEL + (hertz / MIN_LOG_HERTZ).ln() * log_step
    } else {
        3.0 * hertz / 200.0
    }
}

fn slaney_mel_to_hertz(mel: f64) -> f64 {
    const MIN_LOG_HERTZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let log_step = 6.4f64.ln() / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HERTZ * (log_step * (mel - MIN_LOG_MEL)).exp()
    } else {
        200.0 * mel / 3.0
    }
}

/// `[FREQUENCY_BINS][MEL_BINS]` Slaney-normalised triangular filters.
fn mel_filterbank() -> Vec<f32> {
    let minimum_mel = hertz_to_slaney_mel(0.0);
    let maximum_mel = hertz_to_slaney_mel(f64::from(SAMPLE_RATE) / 2.0);
    let edges: Vec<f64> = (0..MEL_BINS + 2)
        .map(|index| {
            let mel =
                minimum_mel + (maximum_mel - minimum_mel) * index as f64 / (MEL_BINS + 1) as f64;
            slaney_mel_to_hertz(mel)
        })
        .collect();
    let mut filters = vec![0.0f32; FREQUENCY_BINS * MEL_BINS];
    for bin in 0..FREQUENCY_BINS {
        let frequency = (f64::from(SAMPLE_RATE) / 2.0) * bin as f64 / (FREQUENCY_BINS - 1) as f64;
        for mel in 0..MEL_BINS {
            let (left, center, right) = (edges[mel], edges[mel + 1], edges[mel + 2]);
            let lower = (frequency - left) / (center - left);
            let upper = (right - frequency) / (right - center);
            let triangle = max_f64(0.0, min_f64(lower, upper));
            let normalization = 2.0 / (right - left);
            filters[bin * MEL_BINS + mel] = (triangle * normalization) as f32;
        }
    }
    filters
}

/// Periodic-Hann windowed DFT coefficients, `[FREQUENCY_BINS][FFT_SIZE]`.
fn dft_table() -> (Vec<f32>, Vec<f32>) {
    let mut real = vec![0.0f32; FREQUENCY_BINS * FFT_SIZE];
    let mut imaginary = vec![0.0f32; FREQUENCY_BINS * FFT_SIZE];
    for bin in 0..FREQUENCY_BINS {
        for sample in 0..FFT_SIZE {
            let window = 0.5 - 0.5 * (2.0 * PI * sample as f64 / FFT_SIZE as f64).cos();
            let phase = -2.0 * PI * (bin * sample) as f64 / FFT_SIZE as f64;
            let index = bin * FFT_SIZE + sample;
            real[index] = (window * phase.cos()) as f32;
            imaginary[index] = (window * phase.sin()) as f32;
        }
    }
    (real, imaginary)
}

struct Tables {
    real: Vec<f32>,
    imaginary: Vec<f32>,
    filters: Vec<f32>,
}

fn tables() -> &'static Tables {
    static TABLES: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
    TABLES.get_or_init(|| {
        let (real, imaginary) = dft_table();
        Tables {
            real,
            imaginary,
            filters: mel_filterbank(),
        }
    })
}

/// Splits `0..count` into contiguous ranges over the available cores and runs
/// `body` on each. Every item keeps its own accumulation order, so results
/// equal the serial loop's.
fn for_each_range<T: Send>(
    output: &mut [T],
    items: usize,
    item_len: usize,
    body: impl Fn(usize, &mut [T]) + Sync,
) {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let workers = cores.min(items.max(1));
    let span = items.div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        let mut chunks = output.chunks_mut(span * item_len).enumerate();
        let first = chunks.next();
        for (index, chunk) in chunks {
            let body = &body;
            scope.spawn(move || {
                for (i, item) in chunk.chunks_mut(item_len).enumerate() {
                    body(index * span + i, item);
                }
            });
        }
        if let Some((_, chunk)) = first {
            for (i, item) in chunk.chunks_mut(item_len).enumerate() {
                body(i, item);
            }
        }
    });
}

/// Centered periodic-Hann 400-point STFT with a 160-sample hop, squared
/// magnitude, 128-bin Slaney mel bank and Whisper log scaling.
pub fn log_mel_features(waveform: &[f32]) -> Result<LogMelFeatures, String> {
    if waveform.len() <= CENTER_PADDING {
        return Err("Qwen3-ASR input is too short for reflected STFT padding".into());
    }
    if waveform.len() as u64 > MAXIMUM_SECONDS * u64::from(SAMPLE_RATE) {
        return Err("Qwen3-ASR input must be at most 30 minutes".into());
    }
    if !waveform.iter().all(|v| v.is_finite()) {
        return Err("Qwen3-ASR input contains non-finite samples".into());
    }
    let padded: Vec<f32> = (0..waveform.len() + 2 * CENTER_PADDING)
        .map(|index| {
            waveform[reflect(index as i64 - CENTER_PADDING as i64, waveform.len() as i64) as usize]
        })
        .collect();
    // WhisperFeatureExtractor drops the final centered STFT frame.
    let frames = (padded.len() - FFT_SIZE) / HOP_LENGTH;
    let tables = tables();

    // Frame-major power spectrum, transposed below; the per-frame DFT is the
    // dominant CPU cost and frames are independent.
    let mut power_by_frame = vec![0.0f32; frames * FREQUENCY_BINS];
    for_each_range(&mut power_by_frame, frames, FREQUENCY_BINS, |frame, out| {
        let samples = &padded[frame * HOP_LENGTH..frame * HOP_LENGTH + FFT_SIZE];
        for (bin, power) in out.iter_mut().enumerate() {
            let real_row = &tables.real[bin * FFT_SIZE..(bin + 1) * FFT_SIZE];
            let imaginary_row = &tables.imaginary[bin * FFT_SIZE..(bin + 1) * FFT_SIZE];
            let (mut real, mut imaginary) = (0.0f64, 0.0f64);
            for ((&s, &r), &i) in samples.iter().zip(real_row).zip(imaginary_row) {
                real += f64::from(s) * f64::from(r);
                imaginary += f64::from(s) * f64::from(i);
            }
            *power = (real * real + imaginary * imaginary) as f32;
        }
    });

    let mut values = vec![0.0f32; MEL_BINS * frames];
    let mut maxima = vec![f32::NEG_INFINITY; MEL_BINS];
    {
        let mut rows: Vec<(&mut [f32], &mut f32)> =
            values.chunks_mut(frames).zip(maxima.iter_mut()).collect();
        let workers = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(MEL_BINS);
        let span = MEL_BINS.div_ceil(workers);
        let filters = &tables.filters;
        let power = &power_by_frame;
        std::thread::scope(|scope| {
            for (group, chunk) in rows.chunks_mut(span).enumerate() {
                scope.spawn(move || {
                    for (i, (row, maximum)) in chunk.iter_mut().enumerate() {
                        let mel = group * span + i;
                        let mut band_maximum = f32::NEG_INFINITY;
                        for (frame, value) in row.iter_mut().enumerate() {
                            let spectrum =
                                &power[frame * FREQUENCY_BINS..(frame + 1) * FREQUENCY_BINS];
                            let mut sum = 0.0f32;
                            for (bin, &p) in spectrum.iter().enumerate() {
                                sum += filters[bin * MEL_BINS + mel] * p;
                            }
                            *value = max_f32(sum, 1.0e-10).log10();
                            band_maximum = max_f32(band_maximum, *value);
                        }
                        **maximum = band_maximum;
                    }
                });
            }
        });
    }
    let maximum = maxima.iter().fold(f32::NEG_INFINITY, |m, &v| max_f32(m, v));
    let floor = maximum - 8.0;
    for value in &mut values {
        *value = (max_f32(*value, floor) + 4.0) / 4.0;
    }
    Ok(LogMelFeatures { values, frames })
}

/// Audio embeddings the fixed convolutional frontend produces for
/// `feature_frames` log-mel frames.
pub fn audio_embedding_token_count(feature_frames: usize) -> Result<usize, String> {
    if feature_frames == 0 {
        return Err("Qwen3-ASR audio token count requires log-mel frames".into());
    }
    let chunks = 1 + (feature_frames - 1) / CHUNK_FRAMES;
    let tail_frames = feature_frames - (chunks - 1) * CHUNK_FRAMES;
    (chunks - 1)
        .checked_mul(CHUNK_TOKENS)
        .and_then(|t| t.checked_add(tail_frames.div_ceil(8)))
        .ok_or_else(|| "Qwen3-ASR audio token count overflow".into())
}

/// Maximum 16 kHz samples fitting `tokens` audio embeddings, capped at the
/// upstream 20-minute inference window.
pub fn audio_samples_for_token_budget(tokens: usize) -> usize {
    if tokens == 0 {
        return 0;
    }
    let limit = 1200 * SAMPLE_RATE as usize;
    let (mut low, mut high) = (0usize, limit / HOP_LENGTH);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if audio_embedding_token_count(middle).is_ok_and(|count| count <= tokens) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    ((low + 1) * HOP_LENGTH - 1).min(limit)
}

/// Low-energy cuts following the upstream 100 ms absolute-amplitude window;
/// chunks partition the samples exactly, without gaps or overlap.
pub fn split_audio(waveform: &[f32], maximum_samples: usize) -> Result<Vec<AudioChunk>, String> {
    if !waveform.is_empty() && waveform.len() <= maximum_samples {
        return Ok(vec![AudioChunk {
            offset: 0,
            samples: waveform.len(),
        }]);
    }
    if waveform.is_empty() || maximum_samples < SAMPLE_RATE as usize / 2 {
        return Err("Qwen3-ASR audio chunk capacity is too small".into());
    }
    let mut chunks = Vec::new();
    let mut offset = 0usize;
    let search = (5 * SAMPLE_RATE as usize).min(maximum_samples / 4);
    let window = SAMPLE_RATE as usize / 10;
    while waveform.len() - offset > maximum_samples {
        let right = offset + maximum_samples;
        let left = right - 2 * search;
        let mut sum = 0.0f64;
        for &sample in &waveform[left..left + window] {
            sum += f64::from(sample.abs());
        }
        let mut minimum = sum;
        let mut best = left;
        for i in left + window..right {
            sum += f64::from(waveform[i].abs()) - f64::from(waveform[i - window].abs());
            if sum < minimum {
                minimum = sum;
                best = i - window + 1;
            }
        }
        let mut boundary = best;
        for i in best + 1..best + window {
            if waveform[i].abs() < waveform[boundary].abs() {
                boundary = i;
            }
        }
        chunks.push(AudioChunk {
            offset,
            samples: boundary - offset,
        });
        offset = boundary;
    }
    chunks.push(AudioChunk {
        offset,
        samples: waveform.len() - offset,
    });
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(format: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let block = channels * bits / 8;
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&format.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
        bytes.extend_from_slice(&block.to_le_bytes());
        bytes.extend_from_slice(&bits.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn pcm_formats_scale_like_the_cpp_decoder() {
        let pcm16 = decode_wav(&wav(1, 1, 16_000, 16, &[0x00, 0x80, 0xFF, 0x7F])).unwrap();
        assert_eq!(pcm16.samples, [-1.0, 32767.0 / 32768.0]);
        let pcm24 = decode_wav(&wav(1, 2, 48_000, 24, &[0, 0, 0x80, 0xFF, 0xFF, 0x7F])).unwrap();
        assert_eq!(pcm24.channels, 2);
        assert_eq!(pcm24.samples, [-1.0, 8_388_607.0f32 / 8_388_608.0]);
        let float = decode_wav(&wav(3, 1, 8_000, 32, &2.5f32.to_le_bytes())).unwrap();
        assert_eq!(float.samples, [1.0]);
        let nan = wav(3, 1, 8_000, 32, &f32::NAN.to_le_bytes());
        assert!(decode_wav(&nan).unwrap_err().contains("non-finite"));
        assert!(decode_wav(&wav(1, 1, 16_000, 8, &[1, 2])).is_err());
        assert!(decode_wav(b"RIFF").is_err());
    }

    #[test]
    fn sixteen_kilohertz_mono_is_copied_exactly() {
        let audio = AudioBuffer {
            sample_rate: 16_000,
            channels: 2,
            samples: vec![0.25, 0.75, -1.0, 0.5],
        };
        assert_eq!(resample_mono_16k(&audio).unwrap(), [0.5, -0.25]);
        let tripled = AudioBuffer {
            sample_rate: 48_000,
            channels: 1,
            samples: vec![0.1; 4800],
        };
        let output = resample_mono_16k(&tripled).unwrap();
        assert_eq!(output.len(), 1600);
        assert!(output.iter().all(|v| (v - 0.1).abs() < 1e-6));
    }

    #[test]
    fn token_counts_follow_the_convolutional_frontend() {
        assert_eq!(audio_embedding_token_count(1).unwrap(), 1);
        assert_eq!(audio_embedding_token_count(100).unwrap(), 13);
        assert_eq!(audio_embedding_token_count(101).unwrap(), 14);
        assert_eq!(audio_embedding_token_count(1505).unwrap(), 196);
        assert!(audio_embedding_token_count(0).is_err());
        let samples = audio_samples_for_token_budget(196);
        let frames = samples / HOP_LENGTH;
        assert!(audio_embedding_token_count(frames).unwrap() <= 196);
        assert!(audio_embedding_token_count(frames + 1).unwrap() > 196);
    }

    #[test]
    fn splits_partition_the_waveform_at_quiet_points() {
        let mut waveform: Vec<f32> = (0..40_000).map(|i| ((i % 7) as f32 - 3.0) / 3.0).collect();
        waveform[30_000..31_600].fill(0.0);
        let chunks = split_audio(&waveform, 32_000).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].offset, 0);
        assert!((30_000..31_600).contains(&chunks[1].offset));
        assert_eq!(chunks[1].offset + chunks[1].samples, waveform.len());
        assert_eq!(split_audio(&waveform[..100], 32_000).unwrap().len(), 1);
        assert!(split_audio(&waveform, 1000).is_err());
    }
}
