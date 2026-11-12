//! Log-mel features of the Base speaker encoder (`ComputeFeatures` of
//! `hip/speaker_encoder_runtime.cpp`): reflected padding, a Hann-windowed
//! 1024-point FFT with hop 256, a Slaney mel filterbank of 128 bins up to
//! 12 kHz, natural log. Every step keeps the C++ float operations; the
//! filterbank sum runs in x87 extended precision there, which [`X87Sum`]
//! reproduces exactly.

use std::sync::OnceLock;

use crate::audio::{AudioBuffer, reflect_index, resample_mono};

const SAMPLE_RATE: usize = 24000;
pub const MEL_BINS: usize = 128;
const FFT: usize = 1024;
const FREQUENCY_BINS: usize = FFT / 2 + 1;
const HOP: usize = 256;
const PREPAD: usize = (FFT - HOP) / 2;

/// `[frames, 128]` frame-major log-mel features.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeakerFeatures {
    pub values: Vec<f32>,
    pub frames: usize,
}

#[derive(Clone, Copy)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    /// The GCC expansion of `std::complex<float>` multiplication for finite
    /// operands: `(ac - bd) + (ad + bc)i`.
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

/// `Fft`: in-place iterative radix-2, twiddles by repeated multiplication.
fn fft(values: &mut [Complex]) {
    let size = values.len();
    let mut reversed = 0usize;
    for index in 1..size {
        let mut bit = size >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            values.swap(index, reversed);
        }
    }
    let mut length = 2;
    while length <= size {
        let angle = -2.0f32 * std::f32::consts::PI / length as f32;
        let root = Complex {
            re: angle.cos(),
            im: angle.sin(),
        };
        for offset in (0..size).step_by(length) {
            let mut factor = Complex { re: 1.0, im: 0.0 };
            for index in 0..length / 2 {
                let even = values[offset + index];
                let odd = factor.mul(values[offset + index + length / 2]);
                values[offset + index] = Complex {
                    re: even.re + odd.re,
                    im: even.im + odd.im,
                };
                values[offset + index + length / 2] = Complex {
                    re: even.re - odd.re,
                    im: even.im - odd.im,
                };
                factor = factor.mul(root);
            }
        }
        length <<= 1;
    }
}

const MEL_SCALE: f32 = 200.0 / 3.0;
const BOUNDARY_HERTZ: f32 = 1000.0;
const BOUNDARY_MEL: f32 = BOUNDARY_HERTZ / MEL_SCALE;
// The float of the C++ literal 0.06875177742094912F.
const LOG_STEP: f32 = 0.068_751_775;

fn hertz_to_mel(hertz: f32) -> f32 {
    if hertz >= BOUNDARY_HERTZ {
        BOUNDARY_MEL + (hertz / BOUNDARY_HERTZ).ln() / LOG_STEP
    } else {
        hertz / MEL_SCALE
    }
}

fn mel_to_hertz(mel: f32) -> f32 {
    if mel >= BOUNDARY_MEL {
        BOUNDARY_HERTZ * (LOG_STEP * (mel - BOUNDARY_MEL)).exp()
    } else {
        mel * MEL_SCALE
    }
}

/// `MakeMelFilterbank`: `[128, 513]`.
fn mel_filterbank() -> Vec<f32> {
    let mel_min = hertz_to_mel(0.0);
    let mel_max = hertz_to_mel(12000.0);
    let hertz: Vec<f32> = (0..MEL_BINS + 2)
        .map(|index| {
            mel_to_hertz(mel_min + (mel_max - mel_min) * index as f32 / (MEL_BINS + 1) as f32)
        })
        .collect();
    let mut filter = vec![0.0f32; MEL_BINS * FREQUENCY_BINS];
    for mel in 0..MEL_BINS {
        let (left, center, right) = (hertz[mel], hertz[mel + 1], hertz[mel + 2]);
        let normalization = 2.0f32 / (right - left).max(1.0e-12);
        for bin in 0..FREQUENCY_BINS {
            let frequency = SAMPLE_RATE as f32 * 0.5 * bin as f32 / (FREQUENCY_BINS - 1) as f32;
            let lower = (frequency - left) / (center - left).max(1.0e-12);
            let upper = (right - frequency) / (right - center).max(1.0e-12);
            filter[mel * FREQUENCY_BINS + bin] = 0.0f32.max(lower.min(upper)) * normalization;
        }
    }
    filter
}

/// A non-negative running sum in x87 extended precision (64-bit mantissa,
/// round to nearest even), as `long double` accumulates on x86-64 Linux.
/// The value is `mantissa * 2^exponent`, normalized with bit 63 set.
#[derive(Clone, Copy, Default)]
struct X87Sum {
    mantissa: u64,
    exponent: i32,
}

impl X87Sum {
    /// Adds an exact non-negative double (the product of two floats is
    /// exact in double, and so in extended precision).
    fn add(&mut self, value: f64) {
        if value == 0.0 {
            return;
        }
        let bits = value.to_bits();
        let biased = ((bits >> 52) & 0x7FF) as i32;
        let fraction = bits & ((1u64 << 52) - 1);
        let (mut mantissa, mut exponent) = if biased == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1u64 << 52), biased - 1075)
        };
        let shift = mantissa.leading_zeros();
        mantissa <<= shift;
        exponent -= shift as i32;
        if self.mantissa == 0 {
            self.mantissa = mantissa;
            self.exponent = exponent;
            return;
        }
        // Align at the larger exponent with 62 guard bits below the
        // mantissas; bits shifted out entirely only set the sticky flag.
        let (high, low) = if self.exponent >= exponent {
            ((self.mantissa, self.exponent), (mantissa, exponent))
        } else {
            ((mantissa, exponent), (self.mantissa, self.exponent))
        };
        let difference = (high.1 - low.1) as u32;
        let wide_high = u128::from(high.0) << 62;
        let wide_low = u128::from(low.0) << 62;
        let (aligned_low, lost) = if difference >= 128 {
            (0, wide_low != 0)
        } else {
            let shifted = wide_low >> difference;
            (shifted, shifted << difference != wide_low)
        };
        let sum = wide_high + aligned_low;
        // `sum` < 2^127: the leading bit is bit 125 or 126.
        let leading = 127 - sum.leading_zeros();
        let drop = leading - 63;
        let kept = (sum >> drop) as u64;
        let remainder = sum & ((1u128 << drop) - 1);
        let half = 1u128 << (drop - 1);
        let round_up = remainder > half || (remainder == half && (lost || kept & 1 == 1));
        let (mut mantissa, mut exponent) = (kept, high.1 - 62 + drop as i32);
        if round_up {
            let (next, carry) = mantissa.overflowing_add(1);
            mantissa = if carry {
                exponent += 1;
                1u64 << 63
            } else {
                next
            };
        }
        self.mantissa = mantissa;
        self.exponent = exponent;
    }

    /// The sum rounded to float (to nearest even). Values below the normal
    /// float range return 0, which the caller's `max(…, 1e-5)` maps alike.
    fn to_f32(self) -> f32 {
        if self.mantissa == 0 {
            return 0.0;
        }
        let mut kept = self.mantissa >> 40;
        let remainder = self.mantissa & ((1u64 << 40) - 1);
        let half = 1u64 << 39;
        let mut exponent = self.exponent + 40;
        if remainder > half || (remainder == half && kept & 1 == 1) {
            kept += 1;
            if kept == 1 << 24 {
                kept >>= 1;
                exponent += 1;
            }
        }
        let biased = exponent + 23 + 127;
        if biased <= 0 {
            return 0.0;
        }
        if biased >= 255 {
            return f32::INFINITY;
        }
        f32::from_bits(((biased as u32) << 23) | (kept as u32 & 0x7F_FFFF))
    }
}

/// `ComputeFeatures`.
pub fn compute_features(audio: &AudioBuffer) -> Result<SpeakerFeatures, String> {
    let waveform = resample_mono(audio, SAMPLE_RATE as u32)?;
    if waveform.is_empty() {
        return Err("Qwen3-TTS speaker encoder requires reference audio".into());
    }
    let padded: Vec<f32> = (0..waveform.len() + 2 * PREPAD)
        .map(|index| {
            let source = reflect_index(index as i64 - PREPAD as i64, waveform.len() as i64);
            waveform[source as usize]
        })
        .collect();
    if padded.len() < FFT {
        return Err("Qwen3-TTS reference audio is too short for speaker encoding".into());
    }
    let frames = 1 + (padded.len() - FFT) / HOP;
    let mut magnitude = vec![0.0f32; FREQUENCY_BINS * frames];
    let mut spectrum = vec![Complex { re: 0.0, im: 0.0 }; FFT];
    for frame in 0..frames {
        for (index, value) in spectrum.iter_mut().enumerate() {
            let window =
                0.5f32 - 0.5 * (2.0f32 * std::f32::consts::PI * index as f32 / FFT as f32).cos();
            *value = Complex {
                re: padded[frame * HOP + index] * window,
                im: 0.0,
            };
        }
        fft(&mut spectrum);
        for bin in 0..FREQUENCY_BINS {
            let z = spectrum[bin];
            // `std::norm` of libstdc++ without fast math: x*x + y*y.
            magnitude[bin * frames + frame] = (z.re * z.re + z.im * z.im + 1.0e-9).sqrt();
        }
    }

    static FILTER: OnceLock<Vec<f32>> = OnceLock::new();
    let filter = FILTER.get_or_init(mel_filterbank);
    let mut values = vec![0.0f32; frames * MEL_BINS];
    for frame in 0..frames {
        for mel in 0..MEL_BINS {
            let mut sum = X87Sum::default();
            for bin in 0..FREQUENCY_BINS {
                sum.add(
                    f64::from(filter[mel * FREQUENCY_BINS + bin])
                        * f64::from(magnitude[bin * frames + frame]),
                );
            }
            values[frame * MEL_BINS + mel] = sum.to_f32().max(1.0e-5).ln();
        }
    }
    Ok(SpeakerFeatures { values, frames })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact sum of the terms, for comparison where every partial sum
    /// fits a 64-bit mantissa.
    #[test]
    fn x87_sum_is_exact_while_it_fits() {
        let mut sum = X87Sum::default();
        for value in [1.5f64, 0.25, 1024.0, 3.0e-3] {
            sum.add(value);
        }
        assert_eq!(sum.to_f32(), (1.5f64 + 0.25 + 1024.0 + 3.0e-3) as f32);
    }

    #[test]
    fn x87_sum_rounds_to_64_bits() {
        // 2^64 + 1 is not representable with 64 mantissa bits: it rounds to
        // even (2^64), while 2^64 + 3 rounds up to 2^64 + 4.
        let mut even = X87Sum::default();
        even.add(2.0f64.powi(64));
        even.add(1.0);
        assert_eq!((even.mantissa, even.exponent), (1 << 63, 1));
        let mut up = X87Sum::default();
        up.add(2.0f64.powi(64));
        up.add(3.0);
        assert_eq!((up.mantissa, up.exponent), ((1 << 63) + 2, 1));
    }

    #[test]
    fn fft_of_an_impulse_is_flat() {
        let mut values = vec![Complex { re: 0.0, im: 0.0 }; 8];
        values[0].re = 1.0;
        fft(&mut values);
        assert!(values.iter().all(|v| v.re == 1.0 && v.im == 0.0));
    }
}
