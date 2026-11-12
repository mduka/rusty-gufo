//! Waveform helpers of OmniVoice's reference pipeline (`audio.cpp`, ported
//! from k2-fsa/OmniVoice's `omnivoice/utils/audio.py` and
//! `create_voice_clone_prompt`, Apache-2.0): the reference voice's
//! preparation and the generated audio's post-processing. Silence handling
//! reproduces pydub 0.25.1 (MIT), including its PCM16 round trip, and the
//! resampler `torchaudio.functional.resample` (BSD-2-Clause).

use std::f64::consts::PI;

// ---- pydub on mono PCM16 ------------------------------------------------

/// A mono PCM16 AudioSegment with prefix sums of squared samples.
struct Segment {
    samples: Vec<i16>,
    rate: u32,
    squares: Vec<i64>,
}

impl Segment {
    fn new(samples: Vec<i16>, rate: u32) -> Self {
        let mut squares = Vec::with_capacity(samples.len() + 1);
        squares.push(0i64);
        let mut total = 0i64;
        for &sample in &samples {
            let value = i64::from(sample);
            total += value * value;
            squares.push(total);
        }
        Self {
            samples,
            rate,
            squares,
        }
    }

    /// `len(segment)`: milliseconds, rounded half to even.
    fn length_ms(&self) -> i64 {
        (1000.0 * (self.samples.len() as f64 / f64::from(self.rate))).round_ties_even() as i64
    }

    /// Sample range of `segment[start_ms:end_ms]`. The range may run up to
    /// half a millisecond past the data, which pydub fills with zeros.
    fn range(&self, start_ms: i64, end_ms: i64) -> (i64, i64) {
        let length = self.length_ms();
        let per_ms = f64::from(self.rate) / 1000.0;
        let begin = (start_ms.min(length) as f64 * per_ms) as i64;
        let end = (end_ms.min(length) as f64 * per_ms) as i64;
        (begin, end)
    }

    /// `segment[start_ms:end_ms].rms`: `audioop.rms`, truncated to an
    /// integer.
    fn rms(&self, start_ms: i64, end_ms: i64) -> u64 {
        let (begin, end) = self.range(start_ms, end_ms);
        if end <= begin {
            return 0;
        }
        let size = self.samples.len() as i64;
        let sum = self.squares[end.min(size) as usize] - self.squares[begin.min(size) as usize];
        (sum as f64 / (end - begin) as f64).sqrt() as u64
    }

    /// `segment[start_ms:end_ms]`.
    fn slice(&self, start_ms: i64, end_ms: i64) -> Vec<i16> {
        let (begin, end) = self.range(start_ms, end_ms);
        if end <= begin {
            return Vec::new();
        }
        let size = self.samples.len() as i64;
        let mut slice = Vec::with_capacity((end - begin) as usize);
        if begin < size {
            slice.extend_from_slice(&self.samples[begin as usize..end.min(size) as usize]);
        }
        slice.resize((end - begin) as usize, 0);
        slice
    }
}

fn db_to_amplitude(db: f64) -> f64 {
    10.0f64.powf(db / 20.0)
}

/// `AudioSegment.dBFS` of `segment[start_ms:end_ms]`.
fn dbfs(segment: &Segment, start_ms: i64, end_ms: i64) -> f64 {
    let rms = segment.rms(start_ms, end_ms);
    if rms == 0 {
        return f64::NEG_INFINITY;
    }
    // math.log(ratio, 10) divides natural logarithms.
    20.0 * ((rms as f64 / 32768.0).ln() / 10.0f64.ln())
}

type Ranges = Vec<(i64, i64)>;

/// `pydub.silence.detect_silence`.
fn detect_silence(
    segment: &Segment,
    min_silence_ms: i64,
    threshold_db: f64,
    seek_ms: i64,
) -> Ranges {
    let length = segment.length_ms();
    if length < min_silence_ms {
        return Vec::new();
    }
    let threshold = db_to_amplitude(threshold_db) * 32768.0;
    let last_start = length - min_silence_ms;
    let mut starts = Vec::new();
    let mut check = |start: i64| {
        if segment.rms(start, start + min_silence_ms) as f64 <= threshold {
            starts.push(start);
        }
    };
    let mut start = 0;
    while start <= last_start {
        check(start);
        start += seek_ms;
    }
    if last_start % seek_ms != 0 {
        check(last_start);
    }
    let Some(&first) = starts.first() else {
        return Vec::new();
    };
    let mut ranges = Vec::new();
    let mut previous = first;
    let mut range_start = previous;
    for &start in &starts[1..] {
        let continuous = start == previous + seek_ms;
        let has_gap = start > previous + min_silence_ms;
        if !continuous && has_gap {
            ranges.push((range_start, previous + min_silence_ms));
            range_start = start;
        }
        previous = start;
    }
    ranges.push((range_start, previous + min_silence_ms));
    ranges
}

/// `pydub.silence.detect_nonsilent`.
fn detect_nonsilent(
    segment: &Segment,
    min_silence_ms: i64,
    threshold_db: f64,
    seek_ms: i64,
) -> Ranges {
    let silent = detect_silence(segment, min_silence_ms, threshold_db, seek_ms);
    let length = segment.length_ms();
    let (Some(&first), Some(&last)) = (silent.first(), silent.last()) else {
        return vec![(0, length)];
    };
    if first == (0, length) {
        return Vec::new();
    }
    let mut nonsilent = Vec::with_capacity(silent.len() + 1);
    let mut previous_end = 0;
    for &(start, end) in &silent {
        nonsilent.push((previous_end, start));
        previous_end = end;
    }
    if last.1 != length {
        nonsilent.push((previous_end, length));
    }
    if nonsilent.first() == Some(&(0, 0)) {
        nonsilent.remove(0);
    }
    nonsilent
}

/// The concatenated pieces of `pydub.silence.split_on_silence`.
fn join_nonsilent(
    segment: &Segment,
    min_silence_ms: i64,
    threshold_db: f64,
    keep_ms: i64,
    seek_ms: i64,
) -> Vec<i16> {
    let mut ranges = detect_nonsilent(segment, min_silence_ms, threshold_db, seek_ms);
    for range in &mut ranges {
        range.0 -= keep_ms;
        range.1 += keep_ms;
    }
    for index in 0..ranges.len().saturating_sub(1) {
        let last_end = ranges[index].1;
        let next_start = ranges[index + 1].0;
        if next_start < last_end {
            // Python's floor division.
            ranges[index].1 = (last_end + next_start).div_euclid(2);
            ranges[index + 1].0 = ranges[index].1;
        }
    }
    let length = segment.length_ms();
    let mut joined = Vec::new();
    for &(start, end) in &ranges {
        joined.extend(segment.slice(start.max(0), end.min(length)));
    }
    joined
}

/// `pydub.silence.detect_leading_silence` with 10 ms chunks.
fn leading_silence_ms(segment: &Segment, threshold_db: f64) -> i64 {
    let length = segment.length_ms();
    let mut trim = 0;
    while dbfs(segment, trim, trim + 10) < threshold_db && trim < length {
        trim += 10;
    }
    trim.min(length)
}

/// `omnivoice.utils.audio.remove_silence_edges`.
fn trim_edges(mut samples: Vec<i16>, rate: u32, leading_ms: i64, trailing_ms: i64) -> Vec<i16> {
    const THRESHOLD: f64 = -50.0;
    for keep in [leading_ms, trailing_ms] {
        let segment = Segment::new(samples, rate);
        let start = (leading_silence_ms(&segment, THRESHOLD) - keep).max(0);
        samples = segment.slice(start, segment.length_ms());
        samples.reverse();
    }
    samples
}

fn to_pcm16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        // `as` truncates toward zero, like the C++ cast.
        .map(|&sample| (sample * 32768.0f32).clamp(-32768.0, 32767.0) as i16)
        .collect()
}

fn from_pcm16(pcm: &[i16]) -> Vec<f32> {
    pcm.iter()
        .map(|&value| f32::from(value) / 32768.0)
        .collect()
}

/// `np.linspace(start, stop, count, dtype=np.float32)[index]`.
fn linspace(start: f64, stop: f64, count: usize, index: usize) -> f32 {
    if count > 1 && index == count - 1 {
        return stop as f32;
    }
    let step = if count > 1 {
        (stop - start) / (count - 1) as f64
    } else {
        0.0
    };
    (index as f64 * step + start) as f32
}

/// Mean of the channels of interleaved samples (`Downmix`).
pub fn downmix(interleaved: &[f32], channels: u32) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let channels = channels as usize;
    interleaved
        .chunks_exact(channels)
        .map(|frame| {
            let mut sum = 0.0f32;
            for &sample in frame {
                sum += sample;
            }
            sum / channels as f32
        })
        .collect()
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `torchaudio.functional.resample` with its defaults: Hann-windowed sinc,
/// six zero crossings, rolloff 0.99 (`Resample`).
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    const ZERO_CROSSINGS: i32 = 6;
    const ROLLOFF: f64 = 0.99;
    let divisor = gcd(from_rate, to_rate);
    let from = i64::from(from_rate / divisor);
    let to = i64::from(to_rate / divisor);
    let base = from.min(to) as f64 * ROLLOFF;
    let width = (f64::from(ZERO_CROSSINGS) * from as f64 / base).ceil() as i64;
    let taps = 2 * width + from;
    // The kernel in float32, operation by operation as torchaudio builds it
    // for a float32 waveform.
    let mut kernel = vec![0.0f32; (to * taps) as usize];
    let base_f = base as f32;
    let pi_f = PI as f32;
    let scale_f = (base / from as f64) as f32;
    let crossings = ZERO_CROSSINGS as f32;
    for phase in 0..to {
        for tap in 0..taps {
            let index = (tap - width) as f32 / from as f32;
            let mut t = (-phase) as f32 / to as f32 + index;
            t *= base_f;
            t = t.clamp(-crossings, crossings);
            let cosine = (t * pi_f / crossings / 2.0).cos();
            let window = cosine * cosine;
            t *= pi_f;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernel[(phase * taps + tap) as usize] = sinc * (window * scale_f);
        }
    }
    let length = samples.len() as i64;
    let output_length = (to as f64 * length as f64 / from as f64).ceil() as i64;
    // conv1d over the input padded by `width` before and `width + from`
    // after, stride `from`, one output channel per phase.
    (0..output_length)
        .map(|out| {
            let block = out / to;
            let phase = out % to;
            let weights = &kernel[(phase * taps) as usize..((phase + 1) * taps) as usize];
            let origin = block * from - width;
            let first = (-origin).max(0);
            let last = taps.min(length - origin);
            let mut sum = 0.0f64;
            for tap in first..last {
                sum +=
                    f64::from(weights[tap as usize]) * f64::from(samples[(origin + tap) as usize]);
            }
            sum as f32
        })
        .collect()
}

/// Root mean square (`Rms`).
pub fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for &sample in samples {
        sum += f64::from(sample) * f64::from(sample);
    }
    (sum / samples.len() as f64).sqrt()
}

/// `omnivoice.utils.audio.remove_silence`: quantizes to PCM16, drops
/// silences (below -50 dBFS) longer than `middle_ms` down to `middle_ms`,
/// keeps at most `leading_ms` / `trailing_ms` at the edges
/// (`RemoveSilence`).
pub fn remove_silence(
    samples: &[f32],
    rate: u32,
    middle_ms: i32,
    leading_ms: i32,
    trailing_ms: i32,
) -> Vec<f32> {
    const THRESHOLD: f64 = -50.0;
    const SEEK_MS: i64 = 10;
    let mut pcm = to_pcm16(samples);
    if middle_ms > 0 {
        let segment = Segment::new(pcm, rate);
        pcm = join_nonsilent(
            &segment,
            i64::from(middle_ms),
            THRESHOLD,
            i64::from(middle_ms),
            SEEK_MS,
        );
    }
    from_pcm16(&trim_edges(
        pcm,
        rate,
        i64::from(leading_ms),
        i64::from(trailing_ms),
    ))
}

/// `omnivoice.utils.audio.fade_and_pad_audio` (`FadeAndPad`; the
/// reference's defaults are 0.1 s of padding and of fade).
pub fn fade_and_pad(samples: &[f32], rate: u32, pad_seconds: f64, fade_seconds: f64) -> Vec<f32> {
    let mut processed = samples.to_vec();
    if processed.is_empty() {
        return processed;
    }
    let fade = (fade_seconds * f64::from(rate)) as usize;
    let pad = (pad_seconds * f64::from(rate)) as usize;
    let count = fade.min(processed.len() / 2);
    let size = processed.len();
    for index in 0..count {
        processed[index] *= linspace(0.0, 1.0, count, index);
        processed[size - count + index] *= linspace(1.0, 0.0, count, index);
    }
    if pad > 0 {
        let mut padded = vec![0.0f32; pad];
        padded.extend_from_slice(&processed);
        padded.resize(padded.len() + pad, 0.0);
        processed = padded;
    }
    processed
}

/// `omnivoice.utils.audio.cross_fade_chunks`: fades each boundary out and
/// in over a third of `gap_seconds` and puts a third of silence between
/// (`CrossFadeChunks`; the reference's default gap is 0.3 s).
pub fn cross_fade_chunks(chunks: &[Vec<f32>], rate: u32, gap_seconds: f64) -> Vec<f32> {
    let Some(first) = chunks.first() else {
        return Vec::new();
    };
    let mut merged = first.clone();
    if chunks.len() == 1 {
        return merged;
    }
    let fade = (gap_seconds * f64::from(rate)) as usize / 3;
    for next in &chunks[1..] {
        let fade_out = fade.min(merged.len());
        let size = merged.len();
        for index in 0..fade_out {
            merged[size - fade_out + index] *= linspace(1.0, 0.0, fade_out, index);
        }
        merged.resize(merged.len() + fade, 0.0);
        let fade_in = fade.min(next.len());
        let offset = merged.len();
        merged.extend_from_slice(next);
        for index in 0..fade_in {
            merged[offset + index] *= linspace(0.0, 1.0, fade_in, index);
        }
    }
    merged
}

/// A reference voice ready for the audio tokenizer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReferenceAudio {
    /// Mono samples at the codec rate, a whole number of codec frames.
    pub samples: Vec<f32>,
    /// RMS before level correction; the generated audio is scaled to it.
    pub rms: f64,
}

/// `create_voice_clone_prompt` with prompt preprocessing: downmix, resample
/// to `rate`, raise quiet voices to RMS 0.1, remove silences and cut the
/// tail to whole frames of `frame_samples` (`PrepareReference`).
pub fn prepare_reference(
    interleaved: &[f32],
    channels: u32,
    source_rate: u32,
    rate: u32,
    frame_samples: usize,
) -> Result<ReferenceAudio, String> {
    if channels == 0 || source_rate == 0 || interleaved.len() < channels as usize {
        return Err("reference audio is empty".into());
    }
    let mut mono = resample(&downmix(interleaved, channels), source_rate, rate);
    let level = rms(&mono);
    if level > 0.0 && level < 0.1 {
        // ref_wav * 0.1 / ref_rms, both in float32.
        let level = level as f32;
        for sample in &mut mono {
            *sample = *sample * 0.1f32 / level;
        }
    }
    let mut mono = remove_silence(&mono, rate, 200, 100, 200);
    if frame_samples > 0 {
        mono.truncate(mono.len() - mono.len() % frame_samples);
    }
    if mono.is_empty() {
        return Err("reference audio is shorter than one codec frame after silence removal".into());
    }
    Ok(ReferenceAudio {
        samples: mono,
        rms: level,
    })
}

/// `_post_process_audio`: removes long silences, matches a quiet
/// reference's level or peak-normalizes to 0.5 without one, then fades and
/// pads both ends (`PostProcess`).
pub fn post_process(samples: &[f32], rate: u32, reference_rms: Option<f64>) -> Vec<f32> {
    let mut audio = remove_silence(samples, rate, 500, 100, 100);
    match reference_rms {
        Some(reference) if reference < 0.1 => {
            let level = reference as f32;
            for sample in &mut audio {
                *sample = *sample * level / 0.1f32;
            }
        }
        None if !audio.is_empty() => {
            let mut peak = 0.0f32;
            for &sample in &audio {
                peak = peak.max(sample.abs());
            }
            if peak > 1e-6 {
                for sample in &mut audio {
                    *sample = *sample / peak * 0.5;
                }
            }
        }
        _ => {}
    }
    fade_and_pad(&audio, rate, 0.1, 0.1)
}
