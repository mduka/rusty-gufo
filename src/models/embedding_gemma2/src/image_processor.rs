//! `Gemma4ImageProcessor` for image embeddings (`image_processor.cpp`): an
//! aspect-preserving soft-token budget (280 by default), torchvision's uint8
//! antialiased bicubic resize, rescale to [0, 1] and 16x16 patches.

use std::path::Path;

use gufo_core::image::{Image, MAX_IMAGE_PIXELS};
use gufo_core::json;

pub const MAX_IMAGE_SIDE: u32 = 32768;
/// `Gemma4ImageProcessor`'s supported `max_soft_tokens` values; the
/// checkpoint defaults to 280.
pub const IMAGE_SOFT_TOKEN_BUDGETS: [u32; 5] = [70, 140, 280, 560, 1120];
pub const DEFAULT_IMAGE_SOFT_TOKENS: u32 = 280;
pub const MAX_IMAGE_SOFT_TOKENS: u32 = 1120;

/// Unpadded, row-major 16x16 RGB patches. The reference pads to nine patches
/// per soft token of the budget and masks the padding; omitting those keys
/// gives the same result.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImagePatches {
    pub rows: u32,
    pub columns: u32,
    /// [patch][pixel y][pixel x][channel], rescaled to [0, 1].
    pub pixels: Vec<f32>,
}

impl ImagePatches {
    pub fn soft_tokens(&self) -> usize {
        self.rows as usize * self.columns as usize / 9
    }
}

// Torchvision's uint8 antialiased bicubic path uses the Pillow filter
// (a = -0.5), separable passes, signed fixed-point coefficients, and uint8
// saturation after each pass. A floating-point resize followed by one cast
// changes pixels and, consequently, the embedding.
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        return ((1.5 * x - 2.5) * x) * x + 1.0;
    }
    if x < 2.0 {
        return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    }
    0.0
}

struct Filter {
    begin: usize,
    precision: u32,
    weights: Vec<i32>,
}

fn filters(input: i32, output: i32) -> Vec<Filter> {
    let scale = f64::from(input) / f64::from(output);
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let mut filters = Vec::with_capacity(output as usize);
    let mut normalized = Vec::with_capacity(output as usize);
    let mut maximum = 0.0f64;
    for i in 0..output {
        let center = (f64::from(i) + 0.5) * scale;
        let begin = 0.max((center - support + 0.5) as i32);
        let end = input.min((center + support + 0.5) as i32);
        let mut weights = Vec::new();
        let mut sum = 0.0;
        for j in begin..end {
            let value = cubic((f64::from(j) - center + 0.5) / filter_scale);
            weights.push(value);
            sum += value;
        }
        for value in &mut weights {
            *value /= sum;
            maximum = maximum.max(*value);
        }
        filters.push(Filter {
            begin: begin as usize,
            precision: 0,
            weights: Vec::new(),
        });
        normalized.push(weights);
    }
    let mut precision = 0u32;
    while precision < 22
        && ((0.5 + maximum * f64::from(1i32 << (precision + 1))) as i32) < (1 << 15)
    {
        precision += 1;
    }
    for (filter, weights) in filters.iter_mut().zip(normalized) {
        filter.precision = precision;
        for value in weights {
            let value = value * f64::from(1i32 << precision);
            filter.weights.push(
                (if value < 0.0 {
                    value - 0.5
                } else {
                    value + 0.5
                }) as i32,
            );
        }
    }
    filters
}

fn resize(image: &Image, width: i32, height: i32) -> Vec<u8> {
    let source_width = image.width as i32;
    let source_height = image.height as i32;
    if source_width == width && source_height == height {
        return image.pixels.clone();
    }
    let horizontal = filters(source_width, width);
    let vertical = filters(source_height, height);
    let (width, height) = (width as usize, height as usize);
    let source_width = source_width as usize;
    let mut intermediate = vec![0u8; image.height as usize * width * 3];
    let mut output = vec![0u8; height * width * 3];
    for y in 0..image.height as usize {
        for (x, filter) in horizontal.iter().enumerate() {
            for c in 0..3 {
                let mut sum: i64 = 1 << (filter.precision - 1);
                for (k, &weight) in filter.weights.iter().enumerate() {
                    sum += i64::from(weight)
                        * i64::from(image.pixels[(y * source_width + filter.begin + k) * 3 + c]);
                }
                intermediate[(y * width + x) * 3 + c] =
                    (sum >> filter.precision).clamp(0, 255) as u8;
            }
        }
    }
    for (y, filter) in vertical.iter().enumerate() {
        for x in 0..width {
            for c in 0..3 {
                let mut sum: i64 = 1 << (filter.precision - 1);
                for (k, &weight) in filter.weights.iter().enumerate() {
                    sum += i64::from(weight)
                        * i64::from(intermediate[((filter.begin + k) * width + x) * 3 + c]);
                }
                output[(y * width + x) * 3 + c] = (sum >> filter.precision).clamp(0, 255) as u8;
            }
        }
    }
    output
}

/// `ValidateImageProcessor`: refuses settings for which the fixed processor
/// would be incorrect.
pub fn validate_image_processor(model_dir: &Path) -> Result<(), String> {
    let contents = std::fs::read(model_dir.join("processor_config.json")).unwrap_or_default();
    let supported = json::parse(&contents).is_ok_and(|root| {
        let config = root.find("image_processor");
        let enabled = |key: &str, expected: bool| {
            config
                .and_then(|c| c.find(key))
                .is_some_and(|v| v.is_bool() && v.as_bool(false) == expected)
        };
        config.is_some_and(|config| {
            config.is_object()
                && config.member_str("image_processor_type", "") == "Gemma4ImageProcessor"
                && enabled("do_convert_rgb", true)
                && enabled("do_resize", true)
                && enabled("do_rescale", true)
                && enabled("do_normalize", false)
                && config.member_size("patch_size", 0) == 16
                && config.member_size("pooling_kernel_size", 0) == 3
                && config.member_size("max_soft_tokens", 0) == DEFAULT_IMAGE_SOFT_TOKENS as usize
                && config.member_size("resample", 0) == 3
                && (config.member_double("rescale_factor", 0.0) - 1.0 / 255.0).abs() < 1e-15
        })
    });
    if supported {
        Ok(())
    } else {
        Err("embedding_gemma2: unsupported or missing image processor configuration".into())
    }
}

/// `SupportedImageSoftTokens`: a `max_soft_tokens` value of
/// [`IMAGE_SOFT_TOKEN_BUDGETS`].
pub fn supported_image_soft_tokens(max_soft_tokens: u32) -> bool {
    IMAGE_SOFT_TOKEN_BUDGETS.contains(&max_soft_tokens)
}

/// `ProcessImage`: aspect-preserving resize to at most `max_soft_tokens` soft
/// tokens (a value of [`IMAGE_SOFT_TOKEN_BUDGETS`]), uint8 antialiased bicubic
/// resize, rescale, then patchify. Does not crop.
pub fn process_image(image: &Image, max_soft_tokens: u32) -> Result<ImagePatches, String> {
    if !supported_image_soft_tokens(max_soft_tokens) {
        return Err("image max_soft_tokens must be one of 70, 140, 280, 560, or 1120".into());
    }
    if image.width == 0
        || image.height == 0
        || image.width > MAX_IMAGE_SIDE
        || image.height > MAX_IMAGE_SIDE
        || image.width as usize * image.height as usize > MAX_IMAGE_PIXELS
        || image.pixels.len() != image.width as usize * image.height as usize * 3
    {
        return Err("invalid embedding image dimensions or RGB data".into());
    }
    let budget = max_soft_tokens as i32;
    const STRIDE: i32 = 48;
    let factor = ((f64::from(budget) * 9.0 * 256.0)
        / (f64::from(image.width) * f64::from(image.height)))
    .sqrt();
    let mut height = (factor * f64::from(image.height) / f64::from(STRIDE)).floor() as i32 * STRIDE;
    let mut width = (factor * f64::from(image.width) / f64::from(STRIDE)).floor() as i32 * STRIDE;
    if height == 0 {
        height = STRIDE;
        width = (image.width / image.height).min(max_soft_tokens) as i32 * STRIDE;
    } else if width == 0 {
        width = STRIDE;
        height = (image.height / image.width).min(max_soft_tokens) as i32 * STRIDE;
    }
    let resized = resize(image, width, height);
    let (width, height) = (width as usize, height as usize);
    let mut pixels = Vec::with_capacity(resized.len());
    for py in (0..height).step_by(16) {
        for px in (0..width).step_by(16) {
            for y in 0..16 {
                for x in 0..16 {
                    for c in 0..3 {
                        pixels.push(
                            f32::from(resized[((py + y) * width + px + x) * 3 + c])
                                * (1.0f32 / 255.0f32),
                        );
                    }
                }
            }
        }
    }
    Ok(ImagePatches {
        rows: (height / 16) as u32,
        columns: (width / 16) as u32,
        pixels,
    })
}
