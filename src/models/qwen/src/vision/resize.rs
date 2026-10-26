//! `ResizeImage` (`src/models/qwen/vision/prompt.cpp`): the Qwen2-VL smart
//! resize to multiples of 32 pixels between 65,536 and 16,777,216 pixels,
//! with PyTorch's antialiased uint8 bicubic filter (a = -0.5): per axis the
//! coefficients are quantized to `i16` with one shared precision, rows are
//! filtered first into `u8`, then columns.

use std::ops::{Add, Mul, Shr};

use gufo_core::image::{Image, MAX_IMAGE_PIXELS};
use gufo_core::text::{Error, Result};

use super::{MAX_ENCODER_PIXELS, RESIZE_FACTOR};

const MIN_PIXELS: u64 = 65536;

/// Round half to even, as Python's `round`.
fn round_even(x: f64) -> u32 {
    let floor = x.floor();
    let part = x - floor;
    let odd = (floor as u32) & 1 == 1;
    floor as u32 + u32::from(part > 0.5 || (part == 0.5 && odd))
}

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        (((-0.5 * x + 2.5) * x - 4.0) * x) + 2.0
    } else {
        0.0
    }
}

/// One axis: the first input index and quantized taps of every output.
struct FilterBank {
    begins: Vec<usize>,
    /// `starts[i]..starts[i + 1]` indexes the taps of output `i`.
    starts: Vec<usize>,
    weights: Vec<i16>,
    precision: u32,
    /// Whether every accumulation fits `i32`: the rounding term plus 255
    /// times the largest sum of absolute taps.
    fits_i32: bool,
}

impl FilterBank {
    fn new(input: u32, output: u32) -> Self {
        let scale = f64::from(input) / f64::from(output);
        let antialias = scale.max(1.0);
        let support = antialias * 2.0;
        let mut begins = Vec::with_capacity(output as usize);
        let mut starts = Vec::with_capacity(output as usize + 1);
        let mut coefficients = Vec::new();
        let mut maximum = 0f64;
        for i in 0..output {
            let center = (f64::from(i) + 0.5) * scale;
            let begin = ((center - support + 0.5) as i32).max(0);
            let end = (input as i32).min((center + support + 0.5) as i32);
            begins.push(begin as usize);
            starts.push(coefficients.len());
            let first = coefficients.len();
            let mut sum = 0f64;
            for j in begin..end {
                let weight = cubic((f64::from(j) + 0.5 - center) / antialias);
                coefficients.push(weight);
                sum += weight;
            }
            for weight in &mut coefficients[first..] {
                *weight /= sum;
                maximum = maximum.max(*weight);
            }
        }
        starts.push(coefficients.len());
        let mut precision = 0u32;
        while precision < 22 {
            if (0.5 + maximum * f64::from(1u32 << (precision + 1))) as i32 >= 1 << 15 {
                break;
            }
            precision += 1;
        }
        let weights: Vec<i16> = coefficients
            .iter()
            .map(|&weight| {
                let scaled = weight * f64::from(1u32 << precision);
                (if scaled < 0.0 {
                    scaled - 0.5
                } else {
                    scaled + 0.5
                }) as i16
            })
            .collect();
        let widest = starts
            .windows(2)
            .map(|range| {
                weights[range[0]..range[1]]
                    .iter()
                    .map(|&w| i64::from(w).abs())
                    .sum::<i64>()
            })
            .max()
            .unwrap_or(0);
        let fits_i32 = (1i64 << precision.saturating_sub(1)) + 255 * widest <= i64::from(i32::MAX);
        Self {
            begins,
            starts,
            weights,
            precision,
            fits_i32,
        }
    }

    fn taps(&self, i: usize) -> (usize, &[i16]) {
        (
            self.begins[i],
            &self.weights[self.starts[i]..self.starts[i + 1]],
        )
    }
}

/// The accumulator of a pass: `i32` when the bank's bound allows it,
/// otherwise `i64` as in C++.
trait Accumulator:
    Copy + Add<Output = Self> + Mul<Output = Self> + Shr<u32, Output = Self> + From<i16> + From<u8>
{
    fn pixel(self) -> u8;
}

impl Accumulator for i32 {
    fn pixel(self) -> u8 {
        self.clamp(0, 255) as u8
    }
}

impl Accumulator for i64 {
    fn pixel(self) -> u8 {
        self.clamp(0, 255) as u8
    }
}

/// `1 << (precision - 1)`, the rounding term of a pass.
fn rounding<A: Accumulator>(precision: u32) -> A {
    let mut value = A::from(1i16);
    for _ in 1..precision {
        value = value + value;
    }
    value
}

/// Filters every row of `source` (`width` pixels) to the bank's outputs.
fn horizontal<A: Accumulator>(source: &[u8], width: usize, bank: &FilterBank, out: &mut [u8]) {
    let outputs = bank.begins.len();
    let round = rounding::<A>(bank.precision);
    for (row, target) in source
        .chunks_exact(width * 3)
        .zip(out.chunks_exact_mut(outputs * 3))
    {
        for (x, pixel) in target.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let (begin, taps) = bank.taps(x);
            let input = &row[begin * 3..(begin + taps.len()) * 3];
            let mut acc = [round; 3];
            for (&w, sample) in taps.iter().zip(input.as_chunks::<3>().0) {
                let w = A::from(w);
                for c in 0..3 {
                    acc[c] = acc[c] + A::from(sample[c]) * w;
                }
            }
            for c in 0..3 {
                pixel[c] = (acc[c] >> bank.precision).pixel();
            }
        }
    }
}

/// Filters the columns of `source` (rows of `width * 3` bytes) to the bank's
/// output rows; each output row accumulates whole input rows.
fn vertical<A: Accumulator>(
    source: &[u8],
    width: usize,
    bank: &FilterBank,
    out: &mut [u8],
    acc: &mut Vec<A>,
) {
    let stride = width * 3;
    let round = rounding::<A>(bank.precision);
    for (y, target) in out.chunks_exact_mut(stride).enumerate() {
        let (begin, taps) = bank.taps(y);
        acc.clear();
        acc.resize(stride, round);
        for (j, &w) in taps.iter().enumerate() {
            let w = A::from(w);
            let row = &source[(begin + j) * stride..(begin + j + 1) * stride];
            for (a, &sample) in acc.iter_mut().zip(row) {
                *a = *a + A::from(sample) * w;
            }
        }
        for (pixel, &a) in target.iter_mut().zip(acc.iter()) {
            *pixel = (a >> bank.precision).pixel();
        }
    }
}

/// The target size of `(width, height)`.
fn target_size(width: u32, height: u32) -> (u32, u32) {
    let factor = RESIZE_FACTOR;
    let mut h = round_even(f64::from(height) / f64::from(factor)) * factor;
    let mut w = round_even(f64::from(width) / f64::from(factor)) * factor;
    let rounded = u64::from(h) * u64::from(w);
    if rounded > MAX_ENCODER_PIXELS {
        let beta = (f64::from(width) * f64::from(height) / MAX_ENCODER_PIXELS as f64).sqrt();
        h = factor.max(((f64::from(height) / beta / f64::from(factor)).floor() as u32) * factor);
        w = factor.max(((f64::from(width) / beta / f64::from(factor)).floor() as u32) * factor);
    } else if rounded < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (f64::from(width) * f64::from(height))).sqrt();
        h = ((f64::from(height) * beta / f64::from(factor)).ceil() as u32) * factor;
        w = ((f64::from(width) * beta / f64::from(factor)).ceil() as u32) * factor;
    }
    (w, h)
}

/// Resizes a decoded image to the encoder's grid.
pub fn resize_image(image: &Image) -> Result<Image> {
    let (width, height) = (image.width, image.height);
    if width == 0
        || height == 0
        || u64::from(width) * u64::from(height) > MAX_IMAGE_PIXELS as u64
        || image.pixels.len() as u64 != u64::from(width) * u64::from(height) * 3
        || f64::from(width.max(height)) / f64::from(width.min(height)) > 200.0
    {
        return Err(Error::invalid("invalid image dimensions or aspect ratio"));
    }
    let (w, h) = target_size(width, height);
    if w == width && h == height {
        return Ok(image.clone());
    }
    let columns = FilterBank::new(width, w);
    let rows = FilterBank::new(height, h);
    let mut intermediate = vec![0u8; w as usize * height as usize * 3];
    if columns.fits_i32 {
        horizontal::<i32>(&image.pixels, width as usize, &columns, &mut intermediate);
    } else {
        horizontal::<i64>(&image.pixels, width as usize, &columns, &mut intermediate);
    }
    let mut pixels = vec![0u8; w as usize * h as usize * 3];
    if rows.fits_i32 {
        vertical::<i32>(
            &intermediate,
            w as usize,
            &rows,
            &mut pixels,
            &mut Vec::new(),
        );
    } else {
        vertical::<i64>(
            &intermediate,
            w as usize,
            &rows,
            &mut pixels,
            &mut Vec::new(),
        );
    }
    Image::new(w, h, pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_follow_the_smart_resize() {
        // Rounds to multiples of 32 with ties to even.
        assert_eq!(target_size(640, 480), (640, 480));
        assert_eq!(target_size(650, 470), (640, 480));
        assert_eq!(target_size(336, 336), (320, 320));
        assert_eq!(target_size(368, 368), (384, 384));
        // Small images grow to at least 65,536 pixels.
        assert_eq!(target_size(100, 100), (256, 256));
        // Large images shrink below 16,777,216 pixels.
        let (w, h) = target_size(8000, 6000);
        assert!(u64::from(w) * u64::from(h) <= MAX_ENCODER_PIXELS);
        assert_eq!((w % 32, h % 32), (0, 0));
    }

    #[test]
    fn identity_sizes_keep_the_pixels_and_extreme_aspects_fail() {
        let image = Image::new(64, 1024, (0..64 * 1024 * 3).map(|i| i as u8).collect()).unwrap();
        assert_eq!(resize_image(&image).unwrap(), image);
        let thin = Image::new(201, 1, vec![0; 603]).unwrap();
        assert!(resize_image(&thin).is_err());
    }

    #[test]
    fn a_constant_image_stays_constant() {
        let image = Image::new(333, 251, vec![77; 333 * 251 * 3]).unwrap();
        let resized = resize_image(&image).unwrap();
        assert_eq!((resized.width, resized.height), (320, 256));
        assert!(resized.pixels.iter().all(|&p| p == 77));
    }
}
