//! PNG, JPEG and WebP decoding to packed RGB8 in display orientation
//! (`src/core/image.cpp`).
//!
//! PNG goes through the `png` crate with the transforms of the C++ libpng
//! path, which follow Pillow's `convert("RGB")`: palettes and low bit depths
//! expand, 16-bit samples keep their high byte (16-bit grayscale saturates
//! instead), gray repeats into three channels, alpha is dropped without
//! compositing, and gamma and ICC metadata are ignored. The decoded pixels
//! equal libpng's. JPEG goes through `zune-jpeg` in strict mode; its pixels
//! may differ from libjpeg-turbo's by a few levels (a different IDCT and
//! upsampling), so comparisons with the C++ engine use PNG inputs. EXIF
//! orientation (JPEG APP1, PNG eXIf) applies to both. WebP, lossy or
//! lossless, goes through `image-webp`; like PNG its alpha is dropped rather
//! than blended, an animation is refused as libwebp's simple decoder does,
//! and orientation metadata is ignored, as in C++.

use std::io::Cursor;
use std::path::Path;

use crate::text::{Error, Result};

/// Encoded bytes accepted per image.
pub const MAX_ENCODED_IMAGE_BYTES: usize = 20 * 1024 * 1024;
/// Decoded pixels accepted per image.
pub const MAX_IMAGE_PIXELS: usize = 32 * 1024 * 1024;

/// Packed, unpremultiplied RGB8 in display orientation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Image {
    /// An image whose pixel buffer holds exactly `width * height` RGB
    /// triples.
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self> {
        if pixels.len() as u64 != u64::from(width) * u64::from(height) * 3 {
            return Err(Error::invalid("image pixels do not match its dimensions"));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }
}

fn validate_dimensions(width: u64, height: u64) -> Result<()> {
    if width == 0 || height == 0 || width * height > MAX_IMAGE_PIXELS as u64 {
        return Err(Error::invalid(
            "image dimensions exceed the decoded pixel limit",
        ));
    }
    Ok(())
}

/// Decodes a PNG, JPEG or WebP image. Dimensions are checked before pixels
/// are allocated.
pub fn decode_image(bytes: &[u8]) -> Result<Image> {
    if bytes.is_empty() || bytes.len() > MAX_ENCODED_IMAGE_BYTES {
        return Err(Error::invalid(
            "image exceeds encoded byte limit or is empty",
        ));
    }
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Ok(orient(decode_png(bytes)?, png_orientation(bytes)));
    }
    if bytes.starts_with(&[0xff, 0xd8]) {
        return Ok(orient(decode_jpeg(bytes)?, jpeg_orientation(bytes)));
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return decode_webp(bytes);
    }
    Err(Error::invalid("image must be PNG, JPEG or WebP"))
}

/// A still WebP image as RGB8 (`DecodeWebp`).
fn decode_webp(bytes: &[u8]) -> Result<Image> {
    let invalid = || Error::invalid("invalid WebP image");
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(bytes)).map_err(|_| invalid())?;
    if decoder.is_animated() {
        return Err(invalid());
    }
    let (width, height) = decoder.dimensions();
    validate_dimensions(width.into(), height.into())?;
    let size = decoder.output_buffer_size().ok_or_else(invalid)?;
    let mut pixels = vec![0u8; size];
    decoder.read_image(&mut pixels).map_err(|_| invalid())?;
    if decoder.has_alpha() {
        pixels = pixels
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
            .collect();
    }
    Image::new(width, height, pixels)
}

/// Reads an image file of at most [`MAX_ENCODED_IMAGE_BYTES`].
pub fn read_image_file(path: &Path) -> Result<Vec<u8>> {
    let length = std::fs::metadata(path)
        .map_err(|_| Error::invalid("cannot open image file"))?
        .len();
    if length == 0 || length > MAX_ENCODED_IMAGE_BYTES as u64 {
        return Err(Error::invalid(
            "image file exceeds encoded byte limit or is empty",
        ));
    }
    let bytes = std::fs::read(path).map_err(|_| Error::invalid("cannot read image file"))?;
    if bytes.is_empty() || bytes.len() > MAX_ENCODED_IMAGE_BYTES {
        return Err(Error::invalid(
            "image file exceeds encoded byte limit or is empty",
        ));
    }
    Ok(bytes)
}

fn decode_png(bytes: &[u8]) -> Result<Image> {
    use png::{BitDepth, ColorType, Transformations};
    let invalid = |_| Error::invalid("invalid PNG image");
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    // libpng verifies the zlib Adler-32 as well as the chunk CRCs.
    decoder.ignore_checksums(false);
    // The largest output buffer: RGBA8 after expansion and 16-bit stripping.
    decoder.set_limits(png::Limits {
        bytes: MAX_IMAGE_PIXELS * 4 + (16 << 20),
    });
    let info = decoder.read_header_info().map_err(invalid)?;
    let (width, height) = (info.width, info.height);
    validate_dimensions(u64::from(width), u64::from(height))?;
    let gray16 = info.color_type == ColorType::Grayscale && info.bit_depth == BitDepth::Sixteen;
    decoder.set_transformations(if gray16 {
        Transformations::IDENTITY
    } else {
        Transformations::EXPAND | Transformations::STRIP_16
    });
    let mut reader = decoder.read_info().map_err(invalid)?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| Error::invalid("invalid PNG image"))?;
    let mut buffer = vec![0u8; size];
    let frame = reader.next_frame(&mut buffer).map_err(invalid)?;
    reader.finish().map_err(invalid)?;
    let (color, depth) = (frame.color_type, frame.bit_depth);
    let samples = &buffer[..frame.buffer_size()];
    let count = width as usize * height as usize;
    let mut pixels = Vec::with_capacity(count * 3);
    match (color, depth) {
        // Pillow's 16-bit grayscale mode saturates the integer sample,
        // unlike 16-bit RGB's high-byte conversion.
        (ColorType::Grayscale, BitDepth::Sixteen) => {
            for sample in samples.as_chunks::<2>().0 {
                let value = u16::from_be_bytes(*sample).min(255) as u8;
                pixels.extend_from_slice(&[value; 3]);
            }
        }
        (ColorType::Grayscale, BitDepth::Eight) => {
            for &value in samples {
                pixels.extend_from_slice(&[value; 3]);
            }
        }
        (ColorType::GrayscaleAlpha, BitDepth::Eight) => {
            for sample in samples.as_chunks::<2>().0 {
                pixels.extend_from_slice(&[sample[0]; 3]);
            }
        }
        (ColorType::Rgb, BitDepth::Eight) => pixels.extend_from_slice(samples),
        (ColorType::Rgba, BitDepth::Eight) => {
            for sample in samples.as_chunks::<4>().0 {
                pixels.extend_from_slice(&sample[..3]);
            }
        }
        _ => return Err(Error::invalid("unsupported PNG pixel layout")),
    }
    Image::new(width, height, pixels).map_err(|_| Error::invalid("unsupported PNG pixel layout"))
}

fn decode_jpeg(bytes: &[u8]) -> Result<Image> {
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;
    let options = DecoderOptions::default()
        .jpeg_set_out_colorspace(ColorSpace::RGB)
        .set_strict_mode(true)
        .set_max_width(usize::from(u16::MAX))
        .set_max_height(usize::from(u16::MAX));
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(Cursor::new(bytes), options);
    decoder
        .decode_headers()
        .map_err(|_| Error::invalid("invalid JPEG header"))?;
    let (width, height) = decoder
        .dimensions()
        .ok_or_else(|| Error::invalid("invalid JPEG header"))?;
    validate_dimensions(width as u64, height as u64)?;
    if decoder.output_colorspace() != Some(ColorSpace::RGB) {
        return Err(Error::invalid("unsupported JPEG color space"));
    }
    let pixels = decoder
        .decode()
        .map_err(|_| Error::invalid("corrupt or truncated JPEG image"))?;
    Image::new(width as u32, height as u32, pixels)
        .map_err(|_| Error::invalid("corrupt or truncated JPEG image"))
}

/// The TIFF orientation tag (1 to 8, otherwise 1), bounds-checked; shared by
/// JPEG APP1 and PNG eXIf.
fn tiff_orientation(data: &[u8]) -> u32 {
    if data.len() < 8 {
        return 1;
    }
    let little = data.starts_with(b"II");
    if !little && !data.starts_with(b"MM") {
        return 1;
    }
    let read = |pos: usize, size: usize| -> u32 {
        match data.get(pos..pos.saturating_add(size)) {
            Some(bytes) if pos.checked_add(size).is_some() => {
                bytes.iter().enumerate().fold(0, |value, (i, &byte)| {
                    let shift = 8 * if little { i } else { size - 1 - i };
                    value | u32::from(byte) << shift
                })
            }
            _ => 0,
        }
    };
    if read(2, 2) != 42 {
        return 1;
    }
    let ifd = read(4, 4) as usize;
    if ifd > data.len() || data.len() - ifd < 2 {
        return 1;
    }
    let count = read(ifd, 2) as usize;
    if count > (data.len() - ifd - 2) / 12 {
        return 1;
    }
    for i in 0..count {
        let p = ifd + 2 + i * 12;
        if read(p, 2) == 0x112 && read(p + 2, 2) == 3 && read(p + 4, 4) == 1 {
            let value = read(p + 8, 2);
            return if (1..=8).contains(&value) { value } else { 1 };
        }
    }
    1
}

fn jpeg_orientation(bytes: &[u8]) -> u32 {
    let mut offset = 2;
    while offset < bytes.len() {
        if bytes[offset] != 0xff {
            break;
        }
        // JPEG permits repeated FF fill bytes before a marker.
        while offset < bytes.len() && bytes[offset] == 0xff {
            offset += 1;
        }
        if offset == bytes.len() {
            break;
        }
        let marker = bytes[offset];
        offset += 1;
        if marker == 0xda || marker == 0xd9 {
            break;
        }
        if marker == 0x01 || marker == 0xd8 || (0xd0..=0xd7).contains(&marker) {
            continue;
        }
        if bytes.len() - offset < 2 {
            break;
        }
        let length = usize::from(bytes[offset]) << 8 | usize::from(bytes[offset + 1]);
        if length < 2 || length > bytes.len() - offset {
            break;
        }
        let data = &bytes[offset + 2..offset + length];
        if marker == 0xe1 && data.len() >= 14 && data.starts_with(b"Exif\0\0") {
            return tiff_orientation(&data[6..]);
        }
        offset += length;
    }
    1
}

fn png_orientation(bytes: &[u8]) -> u32 {
    let mut offset = 8;
    while offset + 12 <= bytes.len() {
        let length = u32::from_be_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        if length > bytes.len() - offset - 12 {
            break;
        }
        if &bytes[offset + 4..offset + 8] == b"eXIf" {
            return tiff_orientation(&bytes[offset + 8..offset + 8 + length]);
        }
        offset += length + 12;
    }
    1
}

/// Applies an EXIF orientation (2 to 8 mirror or rotate; 1 and others keep
/// the image).
fn orient(image: Image, orientation: u32) -> Image {
    if !(2..=8).contains(&orientation) {
        return image;
    }
    let (w, h) = (image.width as usize, image.height as usize);
    let transpose = orientation >= 5;
    let (rw, rh) = if transpose { (h, w) } else { (w, h) };
    let mut pixels = vec![0u8; image.pixels.len()];
    for y in 0..rh {
        for x in 0..rw {
            let (sx, sy) = match orientation {
                2 => (w - 1 - x, y),
                3 => (w - 1 - x, h - 1 - y),
                4 => (x, h - 1 - y),
                5 => (y, x),
                6 => (y, h - 1 - x),
                7 => (w - 1 - y, h - 1 - x),
                _ => (w - 1 - y, x),
            };
            let source = (sy * w + sx) * 3;
            let target = (y * rw + x) * 3;
            pixels[target..target + 3].copy_from_slice(&image.pixels[source..source + 3]);
        }
    }
    Image {
        width: rw as u32,
        height: rh as u32,
        pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An uncompressed PNG with the given IHDR fields and raw scanlines
    /// (filter byte included), stored in zlib blocks without compression.
    fn png(width: u32, height: u32, depth: u8, color: u8, rows: &[u8], extra: &[u8]) -> Vec<u8> {
        fn chunk(out: &mut Vec<u8>, name: &[u8; 4], data: &[u8]) {
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            let start = out.len();
            out.extend_from_slice(name);
            out.extend_from_slice(data);
            let crc = crc32(&out[start..]);
            out.extend_from_slice(&crc.to_be_bytes());
        }
        fn crc32(data: &[u8]) -> u32 {
            let mut crc = !0u32;
            for &byte in data {
                crc ^= u32::from(byte);
                for _ in 0..8 {
                    crc = if crc & 1 != 0 {
                        0xedb8_8320 ^ (crc >> 1)
                    } else {
                        crc >> 1
                    };
                }
            }
            !crc
        }
        let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut header = Vec::new();
        header.extend_from_slice(&width.to_be_bytes());
        header.extend_from_slice(&height.to_be_bytes());
        header.extend_from_slice(&[depth, color, 0, 0, 0]);
        chunk(&mut out, b"IHDR", &header);
        out.extend_from_slice(extra);
        let mut zlib = vec![0x78, 0x01, 1];
        zlib.extend_from_slice(&(rows.len() as u16).to_le_bytes());
        zlib.extend_from_slice(&(!(rows.len() as u16)).to_le_bytes());
        zlib.extend_from_slice(rows);
        let (mut a, mut b) = (1u32, 0u32);
        for &byte in rows {
            a = (a + u32::from(byte)) % 65521;
            b = (b + a) % 65521;
        }
        zlib.extend_from_slice(&(b << 16 | a).to_be_bytes());
        chunk(&mut out, b"IDAT", &zlib);
        chunk(&mut out, b"IEND", &[]);
        out
    }

    #[test]
    fn png_layouts_convert_like_pillow() {
        // RGB8 passes through.
        let rgb = decode_image(&png(2, 1, 8, 2, &[0, 1, 2, 3, 4, 5, 6], &[])).unwrap();
        assert_eq!(rgb.pixels, [1, 2, 3, 4, 5, 6]);
        // RGBA drops alpha without compositing.
        let rgba = decode_image(&png(1, 1, 8, 6, &[0, 9, 8, 7, 0], &[])).unwrap();
        assert_eq!(rgba.pixels, [9, 8, 7]);
        // 16-bit RGB keeps the high byte.
        let rgb16 = decode_image(&png(1, 1, 16, 2, &[0, 1, 255, 2, 0, 3, 128], &[])).unwrap();
        assert_eq!(rgb16.pixels, [1, 2, 3]);
        // 16-bit gray saturates the whole sample.
        let gray16 = decode_image(&png(2, 1, 16, 0, &[0, 0, 200, 1, 0], &[])).unwrap();
        assert_eq!(gray16.pixels, [200, 200, 200, 255, 255, 255]);
        // 2-bit gray scales to 8 bits and repeats into RGB.
        let gray2 = decode_image(&png(4, 1, 2, 0, &[0, 0b0001_1011], &[])).unwrap();
        assert_eq!(
            gray2.pixels,
            [0, 0, 0, 85, 85, 85, 170, 170, 170, 255, 255, 255]
        );
        // Gray with alpha keeps the gray sample.
        let gray_alpha = decode_image(&png(1, 1, 8, 4, &[0, 77, 1], &[])).unwrap();
        assert_eq!(gray_alpha.pixels, [77, 77, 77]);
    }

    #[test]
    fn exif_orientation_rotates_png_pixels() {
        // Orientation 6 (rotate 90° clockwise) of a 2x1 image.
        let mut exif = b"MM\0\x2a\0\0\0\x08\0\x01".to_vec();
        exif.extend_from_slice(&[0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, 6, 0, 0]);
        let mut chunk = (exif.len() as u32).to_be_bytes().to_vec();
        chunk.extend_from_slice(b"eXIf");
        chunk.extend_from_slice(&exif);
        chunk.extend_from_slice(&[0; 4]); // ancillary CRC, not checked here
        let bytes = png(2, 1, 8, 2, &[0, 1, 2, 3, 4, 5, 6], &chunk);
        assert_eq!(png_orientation(&bytes), 6);
        let image = orient(
            Image::new(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap(),
            png_orientation(&bytes),
        );
        assert_eq!((image.width, image.height), (1, 2));
        assert_eq!(image.pixels, [1, 2, 3, 4, 5, 6]);
        let flipped = orient(Image::new(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap(), 2);
        assert_eq!(flipped.pixels, [4, 5, 6, 1, 2, 3]);
    }

    #[test]
    fn inputs_outside_the_limits_are_rejected() {
        assert!(decode_image(&[]).is_err());
        assert!(decode_image(b"GIF89a").is_err());
        // 8192 x 8192 exceeds the pixel limit before any allocation.
        let huge = png(8192, 8192, 8, 2, &[0], &[]);
        assert_eq!(
            decode_image(&huge).unwrap_err().message(),
            "image dimensions exceed the decoded pixel limit"
        );
        assert!(decode_image(&[0xff, 0xd8, 0xff, 0xd9]).is_err());
    }
}
