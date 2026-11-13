//! The log-mel front end (`features.hpp`, `features.cpp`) of transformers'
//! `NemotronAsrStreamingFeatureExtractor` (Apache-2.0, see
//! licenses/transformers.txt), whose mel filters are librosa's (ISC, see
//! licenses/librosa.txt), computed incrementally: pre-emphasis, a centered
//! STFT with a symmetric Hann window (zero padding at the start), the power
//! spectrum, Slaney mel filters as librosa builds them and log(x + 2^-24).
//! Frames come out as soon as their window is complete, equal to a centered
//! pass over the whole signal.
//!
//! Every value is computed in the C++ extension's types and order (float32
//! samples and filters, a float64 FFT, the same libm calls), so the frames
//! are bit-identical to its `FeatureStream`.

use std::f64::consts::PI;

use crate::config::FeatureConfig;

const MIN_LOG_HZ: f64 = 1000.0;
const HZ_PER_MEL: f64 = 200.0 / 3.0;
/// 2^-24.
const LOG_GUARD: f32 = 1.0 / 16_777_216.0;

/// librosa's Slaney mel scale (`HzToMel`).
fn hz_to_mel(hz: f64) -> f64 {
    let min_log_mel = MIN_LOG_HZ / HZ_PER_MEL;
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= MIN_LOG_HZ {
        return min_log_mel + (hz / MIN_LOG_HZ).ln() / logstep;
    }
    hz / HZ_PER_MEL
}

/// `MelToHz`.
fn mel_to_hz(mel: f64) -> f64 {
    let min_log_mel = MIN_LOG_HZ / HZ_PER_MEL;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        return MIN_LOG_HZ * (logstep * (mel - min_log_mel)).exp();
    }
    HZ_PER_MEL * mel
}

/// `librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2,
/// norm="slaney")` as float32, row-major `[bins][n_fft / 2 + 1]`
/// (`MelFilters`).
fn mel_filters(rate: usize, n_fft: usize, bins: usize) -> Vec<f32> {
    let spectrum = n_fft / 2 + 1;
    let frequencies: Vec<f64> = (0..spectrum)
        .map(|k| k as f64 * rate as f64 / n_fft as f64)
        .collect();
    let low = hz_to_mel(0.0);
    let high = hz_to_mel(rate as f64 / 2.0);
    let points = bins + 2;
    let mel: Vec<f64> = (0..points)
        .map(|i| {
            // numpy.linspace: start + i * step, the last point exact.
            let step = (high - low) / (bins + 1) as f64;
            let point = if i + 1 == points {
                high
            } else {
                low + i as f64 * step
            };
            mel_to_hz(point)
        })
        .collect();
    let mut filters = vec![0.0f32; bins * spectrum];
    for i in 0..bins {
        let lower_width = mel[i + 1] - mel[i];
        let upper_width = mel[i + 2] - mel[i + 1];
        let norm = 2.0 / (mel[i + 2] - mel[i]);
        for (k, frequency) in frequencies.iter().enumerate() {
            let lower = -(mel[i] - frequency) / lower_width;
            let upper = (mel[i + 2] - frequency) / upper_width;
            // std::max(0.0, std::min(lower, upper)): `upper` unless `lower`
            // is smaller, then 0 unless that is larger.
            let smaller = if upper < lower { upper } else { lower };
            let weight = (if 0.0 < smaller { smaller } else { 0.0 }) as f32;
            filters[i * spectrum + k] = (f64::from(weight) * norm) as f32;
        }
    }
    filters
}

/// A complex float64 with the arithmetic of libstdc++'s `std::complex`
/// for finite values (no fused operations, as GCC builds the extension).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }

    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }

    /// (a + bi)(c + di) = (ac - bd) + (ad + bc)i.
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    fn scale(self, factor: f64) -> Self {
        Self::new(self.re * factor, self.im * factor)
    }

    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// `std::norm`: re² + im².
    fn norm(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// The incremental log-mel front end (`FeatureStream`).
#[derive(Debug, Clone)]
pub struct FeatureStream {
    config: FeatureConfig,
    bins: usize,
    spectrum: usize,
    /// n_fft, the Hann window centered.
    window: Vec<f32>,
    /// `[bins][spectrum]`.
    filters: Vec<f32>,
    /// Each filter's nonzero columns: `[filter_begin[m], filter_end[m])`.
    filter_begin: Vec<usize>,
    filter_end: Vec<usize>,
    /// e^(-2 pi i k / n_fft).
    twiddles: Vec<Complex>,
    /// Bit reversal of n_fft / 2 points.
    reversal: Vec<usize>,
    fft: Vec<Complex>,
    power: Vec<f32>,
    /// Pre-emphasized signal from absolute sample `base` on.
    signal: Vec<f32>,
    base: usize,
    /// Emphasized samples so far, silence included.
    length: usize,
    pushed: usize,
    frames: usize,
    previous: f32,
    started: bool,
}

impl FeatureStream {
    /// A stream for `mel_bins` mel frames of this STFT shape; an error for
    /// shapes the transform cannot take.
    pub fn new(config: &FeatureConfig, mel_bins: usize) -> Result<Self, String> {
        let n = config.n_fft;
        if n == 0 || n & (n - 1) != 0 || config.window > n || config.window < 2 || config.hop == 0 {
            return Err("unsupported STFT shape".into());
        }
        let spectrum = n / 2 + 1;
        let mut window = vec![0.0f32; n];
        let offset = (n - config.window) / 2;
        for i in 0..config.window {
            window[offset + i] =
                (0.5 - 0.5 * (2.0 * PI * i as f64 / (config.window - 1) as f64).cos()) as f32;
        }
        let filters = mel_filters(config.sample_rate, n, mel_bins);
        let mut filter_begin = vec![0; mel_bins];
        let mut filter_end = vec![0; mel_bins];
        for m in 0..mel_bins {
            let filter = &filters[m * spectrum..(m + 1) * spectrum];
            let mut begin = 0;
            while begin < spectrum && filter[begin] == 0.0 {
                begin += 1;
            }
            let mut end = spectrum;
            while end > begin && filter[end - 1] == 0.0 {
                end -= 1;
            }
            filter_begin[m] = begin;
            filter_end[m] = end;
        }
        // std::polar(1.0, theta): (1.0 * cos, 1.0 * sin), the products exact.
        let twiddles = (0..n / 2)
            .map(|k| {
                let theta = -2.0 * PI * k as f64 / n as f64;
                Complex::new(theta.cos(), theta.sin())
            })
            .collect();
        // The real transform of n samples is a complex one of n / 2.
        let half = n / 2;
        let mut bits = 0;
        while (1usize << bits) < half {
            bits += 1;
        }
        let reversal = (0..half)
            .map(|i| {
                let mut r = 0;
                for b in 0..bits {
                    r |= ((i >> b) & 1) << (bits - 1 - b);
                }
                r
            })
            .collect();
        Ok(Self {
            config: config.clone(),
            bins: mel_bins,
            spectrum,
            window,
            filters,
            filter_begin,
            filter_end,
            twiddles,
            reversal,
            fft: vec![Complex::default(); half],
            power: vec![0.0; spectrum],
            signal: Vec::new(),
            base: 0,
            length: 0,
            pushed: 0,
            frames: 0,
            previous: 0.0,
            started: false,
        })
    }

    /// Forgets the signal: the stream starts a new one (`Reset`).
    pub fn reset(&mut self) {
        self.signal.clear();
        self.base = 0;
        self.length = 0;
        self.pushed = 0;
        self.frames = 0;
        self.previous = 0.0;
        self.started = false;
    }

    fn append(&mut self, emphasized: f32) {
        self.signal.push(emphasized);
        self.length += 1;
    }

    /// Appends 16 kHz samples; completed frames are appended to `frames`
    /// (`[frames][mel_bins]`) (`Push`).
    pub fn push(&mut self, samples: &[f32], frames: &mut Vec<f32>) {
        let coefficient = self.config.preemphasis as f32;
        for &sample in samples {
            let emphasized = if self.started {
                sample - coefficient * self.previous
            } else {
                sample
            };
            self.append(emphasized);
            self.previous = sample;
            self.started = true;
        }
        self.pushed += samples.len();
        self.emit(frames);
    }

    /// Ends the signal (`Flush`): zeros after the pre-emphasized samples (as
    /// a centered STFT pads) until `total` frames exist; the new frames are
    /// appended to `frames`.
    pub fn flush(&mut self, total: usize, frames: &mut Vec<f32>) {
        let half = self.config.n_fft / 2;
        if self.frames >= total {
            return;
        }
        // Zeros until frame `total - 1` is complete.
        let needed = (total - 1) * self.config.hop + half;
        while self.length < needed {
            self.append(0.0);
        }
        self.emit(frames);
    }

    fn emit(&mut self, frames: &mut Vec<f32>) {
        let half = self.config.n_fft / 2;
        while self.length >= self.frames * self.config.hop + half {
            let at = frames.len();
            frames.resize(at + self.bins, 0.0);
            self.frame(self.frames, &mut frames[at..]);
            self.frames += 1;
        }
        // Keep the samples later frames still read.
        let first = self.frames * self.config.hop;
        let keep_from = first.saturating_sub(half);
        if keep_from > self.base + 4096 {
            self.signal.drain(..keep_from - self.base);
            self.base = keep_from;
        }
    }

    /// The windowed sample `i` of frame `start` (`windowed`), widened after
    /// the float32 product.
    fn windowed(&self, start: isize, i: usize) -> f64 {
        let at = start + i as isize;
        let mut sample = 0.0f32;
        if at >= 0 && (at as usize) < self.length {
            sample = self.signal[at as usize - self.base];
        }
        f64::from(sample * self.window[i])
    }

    fn frame(&mut self, index: usize, out: &mut [f32]) {
        let n = self.config.n_fft;
        let half = n / 2;
        // Window start in the centered signal: index * hop - n_fft / 2.
        let start = (index * self.config.hop) as isize - half as isize;
        // Even samples as real parts, odd ones as imaginary parts. Frames
        // inside the signal read it as one slice; the edges sample by sample.
        let inside = start >= 0 && start as usize + n <= self.length;
        if inside {
            let from = start as usize - self.base;
            let signal = &self.signal[from..from + n];
            for ((pair, window), &slot) in signal
                .as_chunks::<2>()
                .0
                .iter()
                .zip(self.window.as_chunks::<2>().0)
                .zip(&self.reversal[..half])
            {
                self.fft[slot] = Complex::new(
                    f64::from(pair[0] * window[0]),
                    f64::from(pair[1] * window[1]),
                );
            }
        } else {
            for i in 0..half {
                let value =
                    Complex::new(self.windowed(start, 2 * i), self.windowed(start, 2 * i + 1));
                self.fft[self.reversal[i]] = value;
            }
        }
        // Radix-2 butterflies over slices: the same products and sums, in
        // the same order, as the indexed loops of C++.
        let twiddles = &self.twiddles;
        let fft = &mut self.fft[..half];
        let mut size = 2;
        while size <= half {
            let stride = n / size;
            for block in fft.chunks_exact_mut(size) {
                let (low, high) = block.split_at_mut(size / 2);
                for ((a, b), twiddle) in low
                    .iter_mut()
                    .zip(high.iter_mut())
                    .zip(twiddles.iter().step_by(stride))
                {
                    let t = twiddle.mul(*b);
                    *b = a.sub(t);
                    *a = a.add(t);
                }
            }
            size *= 2;
        }
        // Split into the transforms of the even and odd samples and combine:
        // X[k] = E[k] + e^(-2 pi i k / n) O[k] for k = 0..n / 2.
        for k in 0..=half {
            let z = self.fft[k % half];
            let mirror = self.fft[(half - k) % half].conj();
            let even = z.add(mirror).scale(0.5);
            let odd = Complex::new(0.0, -0.5).mul(z.sub(mirror));
            let turn = if k < half {
                self.twiddles[k]
            } else {
                Complex::new(-1.0, 0.0)
            };
            self.power[k] = even.add(turn.mul(odd)).norm() as f32;
        }
        for (m, value) in out.iter_mut().enumerate().take(self.bins) {
            // Zero filter weights add exact zeros: skip them.
            let filter = &self.filters[m * self.spectrum..(m + 1) * self.spectrum];
            let columns = self.filter_begin[m]..self.filter_end[m];
            let mut sum = 0.0f32;
            for (weight, power) in filter[columns.clone()].iter().zip(&self.power[columns]) {
                sum += weight * power;
            }
            *value = (sum + LOG_GUARD).ln();
        }
    }

    /// Samples pushed (without the silence of `flush`).
    pub fn samples(&self) -> usize {
        self.pushed
    }

    /// Frames produced so far.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// The valid frames of a centered pass over `samples` samples
    /// (`FramesFor`).
    pub fn frames_for(&self, samples: usize) -> usize {
        samples / self.config.hop
    }

    /// The mel filters, `[mel_bins][n_fft / 2 + 1]`.
    pub fn filters(&self) -> &[f32] {
        &self.filters
    }
}
