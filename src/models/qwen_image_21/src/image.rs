//! Images in and out (`image.cpp`): RGBA decoding, PNG encoding, Pillow's
//! RGBA Lanczos resize, request validation and the flow schedule.
//!
//! PNG decoding reproduces libpng's simplified read API with
//! `PNG_FORMAT_RGBA`, which the C++ engine uses: palettes, low bit depths
//! and tRNS expand, gray repeats into three channels, a missing alpha is
//! opaque, and the colour channels are gamma corrected from the file's
//! encoding to sRGB when the two differ significantly. The file gamma is
//! sRGB's for an sRGB chunk, else a gAMA chunk's value, else 1/2.2 for 8-bit
//! and linear for 16-bit samples; libpng's 8-bit and 16-to-8-bit gamma tables
//! are rebuilt with the same arithmetic. Like the simplified API, chunks after
//! the image data are not read. JPEG goes through `gufo_core::image`
//! (`core::DecodeImage` in C++), whose pixels may differ from libjpeg-turbo's
//! by a few levels.

use std::io::Cursor;

use gufo_core::image::{MAX_ENCODED_IMAGE_BYTES, MAX_IMAGE_PIXELS};

use crate::host::fill_rows;
use gufo_core::text::{Error, Result};

/// Unpremultiplied RGBA8 (`Image`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Image {
    pub width: i32,
    pub height: i32,
    pub rgba: Vec<u8>,
}

/// `Request`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub prompt: String,
    pub images: Vec<Image>,
    pub width: i32,
    pub height: i32,
    pub steps: i32,
    pub seed: u64,
}

impl Default for Request {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            images: Vec::new(),
            width: 1024,
            height: 1024,
            steps: 40,
            seed: 0,
        }
    }
}

/// The seed of a request's initial noise (`InitialNoiseSeed`). An edit's
/// noise is bound to its ordered, resized RGBA references, so editing a
/// generated image at its own size and seed never replays the noise that
/// made it, and editing an edit draws anew; generation keeps its seed.
pub fn initial_noise_seed(seed: u64, images: &[Image]) -> Result<u64> {
    use sha2::{Digest, Sha256};
    if images.is_empty() {
        return Ok(seed);
    }
    let mut hash = Sha256::new();
    // The C++ domain is a NUL-terminated byte array, hashed whole.
    hash.update(b"Qwen-Image-2.1 edit noise v1\0");
    hash.update(seed.to_le_bytes());
    hash.update((images.len() as u64).to_le_bytes());
    for image in images {
        check_image(image)?;
        hash.update(u64::from(image.width.unsigned_abs()).to_le_bytes());
        hash.update(u64::from(image.height.unsigned_abs()).to_le_bytes());
        hash.update(&image.rgba);
    }
    let digest = hash.finalize();
    let mut low = [0u8; 8];
    low.copy_from_slice(&digest[..8]);
    Ok(u64::from_le_bytes(low))
}

/// `CheckImage`.
fn check_image(image: &Image) -> Result<()> {
    check_rgba(image.width, image.height, &image.rgba)
}

fn check_rgba(width: i32, height: i32, rgba: &[u8]) -> Result<()> {
    let pixels = i64::from(width) * i64::from(height);
    if width <= 0
        || height <= 0
        || pixels as u64 > MAX_IMAGE_PIXELS as u64
        || rgba.len() as u64 != pixels as u64 * 4
    {
        return Err(Error::invalid("invalid RGBA image dimensions"));
    }
    Ok(())
}

/// `ValidateRequest`.
pub fn validate_request(request: &Request) -> Result<()> {
    if request.prompt.len() > 64 << 10 {
        return Err(Error::invalid("image prompt exceeds 64 KiB"));
    }
    if request.width < 32
        || request.height < 32
        || request.width % 32 != 0
        || request.height % 32 != 0
        || request.width > 4096
        || request.height > 4096
        || i64::from(request.width) * i64::from(request.height) > 4_194_304
    {
        return Err(Error::invalid(
            "image size must be multiples of 32, at most 4096 per side and 4 megapixels",
        ));
    }
    if !(2..=100).contains(&request.steps) {
        return Err(Error::invalid("image steps must be between 2 and 100"));
    }
    if request.images.len() > 10 {
        return Err(Error::invalid(
            "Qwen-Image supports at most 10 reference images",
        ));
    }
    let mut pixels = 0usize;
    for image in &request.images {
        check_image(image)?;
        pixels += image.width as usize * image.height as usize;
    }
    if pixels > MAX_IMAGE_PIXELS {
        return Err(Error::invalid(
            "reference images exceed 32 megapixels combined",
        ));
    }
    Ok(())
}

/// `FlowSigmas`: the dynamically shifted, terminal-stretched flow-matching
/// schedule of `steps` steps followed by 0.
pub fn flow_sigmas(steps: i32, image_tokens: i32) -> Result<Vec<f32>> {
    if !(2..=100).contains(&steps) || image_tokens < 1 {
        return Err(Error::invalid("invalid image flow schedule"));
    }
    let mu = 0.5 + f64::from(image_tokens - 256) * (0.9 - 0.5) / f64::from(8192 - 256);
    let shift = mu.exp() as f32;
    let steps = steps as usize;
    let mut sigmas = vec![0.0f32; steps + 1];
    for (i, sigma) in sigmas.iter_mut().take(steps).enumerate() {
        let t = (1.0 - i as f64 / steps as f64) as f32;
        *sigma = shift / (shift + (1.0 / t - 1.0));
    }
    let scale = (1.0 - sigmas[steps - 1]) / (1.0 - 0.02f32);
    for sigma in sigmas.iter_mut().take(steps) {
        *sigma = 1.0 - (1.0 - *sigma) / scale;
    }
    sigmas[steps] = 0.0;
    Ok(sigmas)
}

// ---- Decoding ----

/// `DecodeImage`: PNG through the simplified libpng semantics, anything else
/// through the core decoder (JPEG), made opaque.
pub fn decode_image(bytes: &[u8]) -> Result<Image> {
    if bytes.len() > MAX_ENCODED_IMAGE_BYTES || bytes.len() < 8 {
        return Err(Error::invalid("invalid or oversized input image"));
    }
    if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        let rgb = gufo_core::image::decode_image(bytes)?;
        let mut rgba = Vec::with_capacity(rgb.pixels.len() / 3 * 4);
        for pixel in rgb.pixels.as_chunks::<3>().0 {
            rgba.extend_from_slice(pixel);
            rgba.push(255);
        }
        return Ok(Image {
            width: rgb.width as i32,
            height: rgb.height as i32,
            rgba,
        });
    }
    decode_png(bytes)
}

/// libpng's fixed-point gamma (`png_fixed_point`, 1.0 = 100000).
const FP_1: i64 = 100_000;
const GAMMA_SRGB: i64 = 220_000;
const GAMMA_SRGB_INVERSE: i64 = 45_455;
const GAMMA_THRESHOLD: i64 = 5_000;
/// `PNG_MAX_GAMMA_8`: input bits kept when 16-bit samples become 8-bit.
const MAX_GAMMA_8: u32 = 11;

/// A rounded double as `png_fixed_point`, 0 on overflow.
fn fixed(r: f64) -> i64 {
    let r = (r + 0.5).floor();
    if (-2_147_483_648.0..=2_147_483_647.0).contains(&r) {
        r as i64
    } else {
        0
    }
}

/// `png_reciprocal`.
fn reciprocal(a: i64) -> i64 {
    fixed(1e10 / a as f64)
}

/// `png_reciprocal2`: 1 / (a * b).
fn reciprocal2(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        return 0;
    }
    fixed(1e15 / a as f64 / b as f64)
}

/// `png_gamma_significant`.
fn significant(gamma: i64) -> bool {
    !(FP_1 - GAMMA_THRESHOLD..=FP_1 + GAMMA_THRESHOLD).contains(&gamma)
}

/// `png_gamma_threshold`: whether `file` to `screen` needs correction.
fn needs_correction(file: i64, screen: i64) -> bool {
    let product = (file as f64 * screen as f64 / FP_1 as f64 + 0.5).floor();
    !(-2_147_483_648.0..=2_147_483_647.0).contains(&product) || significant(product as i64)
}

/// `png_gamma_8bit_correct`.
fn correct_8bit(value: u32, gamma: i64) -> u8 {
    if value > 0 && value < 255 {
        (255.0 * (f64::from(value) / 255.0).powf(gamma as f64 * 0.00001) + 0.5).floor() as u8
    } else {
        value as u8
    }
}

/// `png_gamma_16bit_correct`.
fn correct_16bit(value: u32, gamma: i64) -> u32 {
    if value > 0 && value < 65535 {
        (65535.0 * (f64::from(value) / 65535.0).powf(gamma as f64 * 0.00001) + 0.5).floor() as u16
            as u32
    } else {
        value
    }
}

/// `png_build_8bit_table`.
fn table_8bit(gamma: i64) -> [u8; 256] {
    let mut table = [0u8; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        *entry = if significant(gamma) {
            correct_8bit(i as u32, gamma)
        } else {
            i as u8
        };
    }
    table
}

/// `png_build_16to8_table` with `gamma`, the inverse of the correction: the
/// 16-bit output (a multiple of 257) of input `v` is
/// `table[(v & 0xff) >> shift][v >> 8]`.
fn table_16to8(shift: u32, gamma: i64) -> Vec<[u16; 256]> {
    let num = 1u32 << (8 - shift);
    let max = (1u32 << (16 - shift)) - 1;
    let mut table = vec![[0u16; 256]; num as usize];
    let mut set = |last: u32, out: u16| {
        table[(last & (0xff >> shift)) as usize][(last >> (8 - shift)) as usize] = out;
    };
    let mut last = 0u32;
    for i in 0..255u32 {
        let out = (i * 257) as u16;
        let bound = correct_16bit(u32::from(out) + 128, gamma);
        let bound = (bound * max + 32768) / 65535 + 1;
        while last < bound {
            set(last, out);
            last += 1;
        }
    }
    while last < num << 8 {
        set(last, 65535);
        last += 1;
    }
    table
}

/// `png_do_scale_16_to_8` of one big-endian sample.
fn scale_16_to_8(high: u8, low: u8) -> u8 {
    let tmp = i32::from(high);
    (tmp + (((i32::from(low) - tmp + 128) * 65535) >> 24)) as u8
}

fn decode_png(bytes: &[u8]) -> Result<Image> {
    use png::{BitDepth, ColorType, Transformations};
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.ignore_checksums(false);
    decoder.set_limits(png::Limits {
        bytes: MAX_IMAGE_PIXELS * 8 + (16 << 20),
    });
    let header = decoder
        .read_header_info()
        .map_err(|_| Error::invalid("cannot decode PNG image"))?;
    let (width, height) = (header.width, header.height);
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_IMAGE_PIXELS as u64 {
        return Err(Error::invalid("PNG image dimensions exceed limits"));
    }
    decoder.set_transformations(Transformations::EXPAND);
    let mut reader = decoder
        .read_info()
        .map_err(|_| Error::invalid("cannot decode PNG image"))?;
    let info = reader.info();
    let sixteen = info.bit_depth == BitDepth::Sixteen;
    let color = info.color_type;
    // The chunk gamma: sRGB overrides gAMA (libpng 1.6.58, PNGv3 rules).
    let chunk_gamma = if info.srgb.is_some() {
        GAMMA_SRGB_INVERSE
    } else {
        info.gama_chunk.map_or(0, |g| i64::from(g.into_scaled()))
    };
    let sig_bit = match &info.sbit {
        Some(bits) if matches!(color, ColorType::Rgb | ColorType::Rgba | ColorType::Indexed) => {
            bits.iter().take(3).copied().max().unwrap_or(0)
        }
        Some(bits) => bits.first().copied().unwrap_or(0),
        None => 0,
    };
    let file_gamma = if chunk_gamma != 0 {
        chunk_gamma
    } else if sixteen {
        FP_1
    } else {
        GAMMA_SRGB_INVERSE
    };
    let correction =
        needs_correction(file_gamma, GAMMA_SRGB).then(|| reciprocal2(GAMMA_SRGB, file_gamma));
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| Error::invalid("invalid PNG image data"))?;
    let mut buffer = vec![0u8; size];
    let frame = reader
        .next_frame(&mut buffer)
        .map_err(|_| Error::invalid("invalid PNG image data"))?;
    let samples = &buffer[..frame.buffer_size()];
    let (channels, alpha) = match frame.color_type {
        ColorType::Grayscale => (1, false),
        ColorType::GrayscaleAlpha => (2, true),
        ColorType::Rgb => (3, false),
        ColorType::Rgba => (4, true),
        ColorType::Indexed => return Err(Error::invalid("invalid PNG image data")),
    };
    let colors = if alpha { channels - 1 } else { channels };
    // Every sample as 8 bits, colour channels gamma corrected.
    let count = width as usize * height as usize * channels;
    let mut values = Vec::with_capacity(count);
    if frame.bit_depth == BitDepth::Sixteen {
        let table = correction.map(|correction| {
            let shift = if sig_bit > 0 && sig_bit < 16 {
                16 - u32::from(sig_bit)
            } else {
                0
            };
            let shift = shift.clamp(16 - MAX_GAMMA_8, 8);
            (shift, table_16to8(shift, reciprocal(correction)))
        });
        for (i, sample) in samples.as_chunks::<2>().0.iter().take(count).enumerate() {
            let [mut high, mut low] = *sample;
            if let Some((shift, table)) = &table
                && i % channels < colors
            {
                let v = u16::from_be_bytes([high, low]);
                [high, low] =
                    table[usize::from((v & 0xff) >> shift)][usize::from(v >> 8)].to_be_bytes();
            }
            values.push(scale_16_to_8(high, low));
        }
    } else {
        let table = correction.map(table_8bit);
        for (i, &value) in samples.iter().take(count).enumerate() {
            values.push(match &table {
                Some(table) if i % channels < colors => table[usize::from(value)],
                _ => value,
            });
        }
    }
    if values.len() != count {
        return Err(Error::invalid("invalid PNG image data"));
    }
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    for pixel in values.chunks_exact(channels) {
        match channels {
            1 => rgba.extend_from_slice(&[pixel[0], pixel[0], pixel[0], 255]),
            2 => rgba.extend_from_slice(&[pixel[0], pixel[0], pixel[0], pixel[1]]),
            3 => rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]),
            _ => rgba.extend_from_slice(pixel),
        }
    }
    Ok(Image {
        width: width as i32,
        height: height as i32,
        rgba,
    })
}

// ---- Encoding ----

/// `EncodePng`: RGBA8 with an sRGB chunk, adaptive filtering and fast
/// compression (zlib level 1 with libpng in C++); the pixels round-trip
/// exactly, the compressed bytes differ.
pub fn encode_png(image: &Image) -> Result<Vec<u8>> {
    encode_rgba_png(image.width, image.height, &image.rgba)
}

/// [`encode_png`] over borrowed RGBA8 rows.
pub fn encode_rgba_png(width: i32, height: i32, rgba: &[u8]) -> Result<Vec<u8>> {
    check_rgba(width, height, rgba)?;
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    encoder.set_compression(png::Compression::Fast);
    encoder.set_filter(png::Filter::Adaptive);
    let failed = |_| Error::runtime("cannot encode output PNG");
    let mut writer = encoder.write_header().map_err(failed)?;
    writer.write_image_data(rgba).map_err(failed)?;
    writer.finish().map_err(failed)?;
    Ok(bytes)
}

// ---- Resizing ----

/// One output pixel's 22-bit fixed-point Lanczos taps (`Filter`).
struct Filter {
    begin: usize,
    weights: Vec<i32>,
}

/// `LanczosFilters`: Pillow's Lanczos-3 coefficients, normalised and
/// rounded to 22 fractional bits.
fn lanczos_filters(input: usize, output: usize) -> Vec<Filter> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = filter_scale * 3.0;
    let sinc = |z: f64| {
        if z == 0.0 {
            1.0
        } else {
            (z * std::f64::consts::PI).sin() / (z * std::f64::consts::PI)
        }
    };
    (0..output)
        .map(|dst| {
            let center = (dst as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as i64).min(input as i64) as usize;
            let mut values = Vec::with_capacity(end.saturating_sub(start));
            let mut sum = 0.0;
            for i in start..end {
                let x = (i as f64 - center + 0.5) / filter_scale;
                let w = if (-3.0..3.0).contains(&x) {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                };
                values.push(w);
                sum += w;
            }
            let weights = values
                .iter()
                .map(|&value| {
                    (value / sum * f64::from(1 << 22) + if value < 0.0 { -0.5 } else { 0.5 }) as i32
                })
                .collect();
            Filter {
                begin: start,
                weights,
            }
        })
        .collect()
}

/// `ResizeImage`: Pillow's RGBA Lanczos resize in premultiplied space, with
/// each pass rounded to 8 bits as Pillow does, alpha included.
pub fn resize_image(image: &Image, width: i32, height: i32) -> Result<Image> {
    check_image(image)?;
    if width <= 0 || height <= 0 || i64::from(width) * i64::from(height) > MAX_IMAGE_PIXELS as i64 {
        return Err(Error::invalid("invalid resized image dimensions"));
    }
    if width == image.width && height == image.height {
        return Ok(image.clone());
    }
    let (in_width, in_height) = (image.width as usize, image.height as usize);
    let (out_width, out_height) = (width as usize, height as usize);
    let mut source = image.rgba.clone();
    for pixel in source.as_chunks_mut::<4>().0 {
        let alpha = i32::from(pixel[3]);
        for channel in &mut pixel[..3] {
            let v = i32::from(*channel) * alpha + 128;
            *channel = ((v + (v >> 8)) >> 8) as u8;
        }
    }
    let horizontal = lanczos_filters(in_width, out_width);
    let vertical = lanczos_filters(in_height, out_height);
    let clamp = |sum: i64| (sum >> 22).clamp(0, 255) as u8;
    // Both passes are integer sums, so rows split across threads and taps
    // accumulate in any order with the serial result.
    let mut temp = vec![0u8; out_width * in_height * 4];
    fill_rows(&mut temp, out_width * 4, |first, rows| {
        for (y, row) in rows.chunks_exact_mut(out_width * 4).enumerate() {
            let source = &source[(first + y) * in_width * 4..(first + y + 1) * in_width * 4];
            for (f, pixel) in horizontal.iter().zip(row.as_chunks_mut::<4>().0) {
                let taps = &source[f.begin * 4..(f.begin + f.weights.len()) * 4];
                let mut sum = [1i64 << 21; 4];
                for (&weight, tap) in f.weights.iter().zip(taps.as_chunks::<4>().0) {
                    for c in 0..4 {
                        sum[c] += i64::from(weight) * i64::from(tap[c]);
                    }
                }
                *pixel = sum.map(clamp);
            }
        }
    });
    let mut output = vec![0u8; out_width * out_height * 4];
    let stride = out_width * 4;
    fill_rows(&mut output, stride, |first, rows| {
        let mut sums = vec![0i64; stride];
        for (y, row) in rows.chunks_exact_mut(stride).enumerate() {
            let f = &vertical[first + y];
            sums.fill(1 << 21);
            for (k, &weight) in f.weights.iter().enumerate() {
                let line = &temp[(f.begin + k) * stride..(f.begin + k + 1) * stride];
                for (sum, &value) in sums.iter_mut().zip(line) {
                    *sum += i64::from(weight) * i64::from(value);
                }
            }
            for (out, &sum) in row.iter_mut().zip(&sums) {
                *out = clamp(sum);
            }
        }
    });
    for pixel in output.as_chunks_mut::<4>().0 {
        let alpha = u32::from(pixel[3]);
        if alpha != 0 && alpha != 255 {
            for channel in &mut pixel[..3] {
                *channel = (u32::from(*channel) * 255 / alpha).min(255) as u8;
            }
        }
    }
    Ok(Image {
        width,
        height,
        rgba: output,
    })
}
