//! The Gemma4 audio feature extractor (`audio_processor.cpp`): semicausal
//! padding, 321-sample analysis windows, periodic Hann, magnitude spectra,
//! HTK mel filters and natural log over mono 16-kHz samples.

use std::f64::consts::PI;
use std::path::Path;
use std::sync::OnceLock;

use gufo_core::json::{self, Value};

pub const AUDIO_SAMPLE_RATE: usize = 16000;
pub const MAX_AUDIO_SAMPLES: usize = 480000;
pub const AUDIO_MEL_BINS: usize = 128;

const FFT: usize = 512;
const FRAME: usize = 320;
const HOP: usize = 160;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioFeatures {
    /// Time-major [frames, 128]. Invalid frames are zero, including partial
    /// frames.
    pub values: Vec<f32>,
    pub mask: Vec<u8>,
}

impl AudioFeatures {
    /// The valid tokens after the two stride-2 convolutions, each of which
    /// slices the mask `[::2]`.
    pub fn soft_tokens(&self) -> usize {
        self.mask
            .iter()
            .step_by(4)
            .filter(|&&valid| valid != 0)
            .count()
    }
}

/// An audio clip's features and the offset of its placeholder tokens.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SequenceAudio {
    pub offset: usize,
    pub features: AudioFeatures,
}

/// `std::complex<double>` with libstdc++'s arithmetic.
#[derive(Debug, Clone, Copy, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    /// GCC's `_Complex double` product for finite operands.
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// `std::abs`, which calls `cabs`.
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

/// `std::min(a, b)`.
fn cpp_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// `std::max(a, b)`.
fn cpp_max(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

struct Frontend {
    window: [f32; FRAME],
    filters: Vec<[f64; AUDIO_MEL_BINS]>,
    roots: [Complex; FFT / 2],
}

impl Frontend {
    fn new() -> Self {
        let mut window = [0.0f32; FRAME];
        for (i, value) in window.iter_mut().enumerate() {
            *value = (0.5 - 0.5 * (2.0 * PI * i as f64 / FRAME as f64).cos()) as f32;
        }
        let mut roots = [Complex::default(); FFT / 2];
        for (i, root) in roots.iter_mut().enumerate() {
            // `std::polar(1.0, theta)`.
            let theta = -2.0 * PI * i as f64 / FFT as f64;
            *root = Complex {
                re: 1.0 * theta.cos(),
                im: 1.0 * theta.sin(),
            };
        }
        let mut edges = [0.0f64; AUDIO_MEL_BINS + 2];
        let max_mel = 2595.0 * (1.0f64 + 8000.0 / 700.0).log10();
        for (i, edge) in edges.iter_mut().enumerate() {
            *edge = 700.0
                * (10.0f64.powf(max_mel * i as f64 / (AUDIO_MEL_BINS + 1) as f64 / 2595.0) - 1.0);
        }
        let bins = FFT / 2 + 1;
        let mut filters = vec![[0.0f64; AUDIO_MEL_BINS]; bins];
        for (f, row) in filters.iter_mut().enumerate() {
            let hz = 8000.0 * f as f64 / (bins - 1) as f64;
            for (m, value) in row.iter_mut().enumerate() {
                *value = cpp_max(
                    0.0,
                    cpp_min(
                        (hz - edges[m]) / (edges[m + 1] - edges[m]),
                        (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]),
                    ),
                );
            }
        }
        Self {
            window,
            filters,
            roots,
        }
    }

    fn fft(&self, x: &mut [Complex; FFT]) {
        let mut j = 0usize;
        for i in 1..FFT {
            let mut bit = FFT >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                x.swap(i, j);
            }
        }
        let mut length = 2;
        while length <= FFT {
            for begin in (0..FFT).step_by(length) {
                for j in 0..length / 2 {
                    let even = x[begin + j];
                    let odd = x[begin + j + length / 2].mul(self.roots[j * FFT / length]);
                    x[begin + j] = even.add(odd);
                    x[begin + j + length / 2] = even.sub(odd);
                }
            }
            length *= 2;
        }
    }
}

/// `ValidateAudioProcessor`: refuses extractor settings for which the fixed
/// frontend would be incorrect.
pub fn validate_audio_processor(model_dir: &Path) -> Result<(), String> {
    let contents = std::fs::read(model_dir.join("processor_config.json")).unwrap_or_default();
    let supported = json::parse(&contents).is_ok_and(|root| {
        let config = root.find("feature_extractor");
        let member = |name: &str| config.and_then(|c| c.find(name));
        let flag = |name: &str, expected: bool| {
            member(name).is_some_and(|v| v.is_bool() && v.as_bool(false) == expected)
        };
        let null = |name: &str| member(name).is_some_and(Value::is_null);
        config.is_some_and(|config| {
            config.is_object()
                && config.member_str("feature_extractor_type", "") == "Gemma4AudioFeatureExtractor"
                && config.member_size("sampling_rate", 0) == AUDIO_SAMPLE_RATE
                && config.member_size("feature_size", 0) == AUDIO_MEL_BINS
                && config.member_size("frame_length", 0) == FRAME
                && config.member_size("hop_length", 0) == HOP
                && config.member_size("fft_length", 0) == FFT
                && config.member_double("min_frequency", -1.0) == 0.0
                && config.member_double("max_frequency", 0.0) == 8000.0
                && config.member_double("mel_floor", 0.0) == 0.001
                && config.member_double("dither", -1.0) == 0.0
                && config.member_double("preemphasis", -1.0) == 0.0
                && config.member_double("input_scale_factor", 0.0) == 1.0
                && config.member_double("padding_value", -1.0) == 0.0
                && config.member_str("padding_side", "") == "right"
                && flag("fft_overdrive", false)
                && flag("preemphasis_htk_flavor", true)
                && flag("return_attention_mask", true)
                && null("per_bin_mean")
                && null("per_bin_stddev")
        })
    });
    if supported {
        Ok(())
    } else {
        Err("embedding_gemma2: unsupported or missing audio processor configuration".into())
    }
}

/// Why `process_audio` refused its input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioProcessError {
    Invalid(String),
    Cancelled,
}

impl std::fmt::Display for AudioProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => f.write_str(message),
            Self::Cancelled => f.write_str("embedding audio preprocessing cancelled"),
        }
    }
}

/// `ProcessAudio`: the features of mono 16-kHz `waveform`. The official
/// extractor pads each batch's waveforms to a multiple of 128 samples; a
/// longer `padding_samples` reproduces multi-clip batches without changing
/// which features or subsampled tokens are valid.
pub fn process_audio(
    waveform: &[f32],
    padding_samples: usize,
    is_cancelled: Option<&dyn Fn() -> bool>,
) -> Result<AudioFeatures, AudioProcessError> {
    if waveform.is_empty()
        || waveform.len() > MAX_AUDIO_SAMPLES
        || padding_samples > MAX_AUDIO_SAMPLES
        || !waveform.iter().all(|x| x.is_finite())
    {
        return Err(AudioProcessError::Invalid(
            "embedding audio must contain 1 to 480000 finite 16-kHz samples".into(),
        ));
    }
    let padded = waveform.len().max(padding_samples).div_ceil(128) * 128;
    // The official unfold consumes frame_length+1 samples, even with zero
    // preemphasis. Its last sample decides the frame mask but is not FFT
    // input.
    let frames = if padded + FRAME / 2 < FRAME + 1 {
        0
    } else {
        (padded + FRAME / 2 - FRAME - 1) / HOP + 1
    };
    let mut result = AudioFeatures {
        values: vec![0.0; frames * AUDIO_MEL_BINS],
        mask: vec![0; frames],
    };
    static FRONTEND: OnceLock<Frontend> = OnceLock::new();
    let frontend = FRONTEND.get_or_init(Frontend::new);
    for frame in 0..frames {
        if is_cancelled.is_some_and(|cancelled| cancelled()) {
            return Err(AudioProcessError::Cancelled);
        }
        if frame * HOP + FRAME / 2 >= waveform.len() {
            continue;
        }
        result.mask[frame] = 1;
        let mut fft = [Complex::default(); FFT];
        for (j, value) in fft.iter_mut().take(FRAME).enumerate() {
            let sample = (frame * HOP + j) as isize - (FRAME / 2) as isize;
            if sample >= 0 && (sample as usize) < waveform.len() {
                // NumPy multiplies two float32 arrays before its float64 FFT.
                value.re = f64::from(waveform[sample as usize] * frontend.window[j]);
            }
        }
        frontend.fft(&mut fft);
        let mut mel = [0.0f64; AUDIO_MEL_BINS];
        for (f, filters) in frontend.filters.iter().enumerate() {
            let magnitude = fft[f].abs();
            for (sum, filter) in mel.iter_mut().zip(filters) {
                *sum += magnitude * filter;
            }
        }
        for (m, sum) in mel.iter().enumerate() {
            result.values[frame * AUDIO_MEL_BINS + m] = (sum + 0.001).ln() as f32;
        }
    }
    Ok(result)
}
