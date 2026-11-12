//! The C++ extension's `vision_test.cpp`: the patch grids and preprocessed
//! pixels against NVIDIA's processor, and the prompt expansion.

#![allow(clippy::unwrap_used)]
// NVIDIA's pixel statistics, digit for digit as the processor states them.
#![allow(clippy::excessive_precision)]

mod common;

use std::sync::Arc;

use gufo_core::image::Image;
use gufo_nemotron::config::VisionConfig;
use gufo_nemotron::vision_image::{ImageGrid, plan_grid, preprocess_image};
use gufo_nemotron::vision_prompt::{ImageFraming, VisionInput, expand_images};

/// The released projector's values (`VisionConfig::from_gguf` reads them
/// from the file; these tests need no weights).
fn bolt_vision() -> VisionConfig {
    VisionConfig {
        width: 1280,
        num_layers: 32,
        num_heads: 16,
        ff_size: 5120,
        patch_size: 16,
        merge: 2,
        prefix_tokens: 10,
        position_side: 128,
        projector_hidden: 20480,
        projection: 4096,
        min_patches: 1024,
        max_patches: 13312,
        mean: [0.48145466, 0.4578275, 0.40821073],
        deviation: [0.26862954, 0.26130258, 0.27577711],
        ..VisionConfig::default()
    }
}

/// `tools/vision_golden.py`'s synthetic image: gradients plus hash noise
/// (`vision_synthetic.hpp`).
fn synthetic(width: u32, height: u32) -> Image {
    let (w, h) = (u64::from(width), u64::from(height));
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3u64 {
                let hash = ((x * 73_856_093) ^ (y * 19_349_663) ^ (c * 83_492_791)) & 0xFFFF_FFFF;
                let v = (x * 255) / w + (y * 128) / h + c * 40 + (hash >> 13) % 48;
                pixels[((y * w + x) * 3 + c) as usize] = (v % 256) as u8;
            }
        }
    }
    Image {
        width,
        height,
        pixels,
    }
}

/// An uncompressed PNG of `image` (stored deflate blocks), enough for the
/// decoder.
fn encode_png(image: &Image) -> Vec<u8> {
    let mut table = [0u32; 256];
    for (n, entry) in table.iter_mut().enumerate() {
        let mut c = n as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *entry = c;
    }
    let mut png = vec![0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];
    let chunk = |png: &mut Vec<u8>, ty: &[u8; 4], data: &[u8]| {
        png.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = ty.to_vec();
        body.extend_from_slice(data);
        let mut crc = 0xFFFF_FFFFu32;
        for &b in &body {
            crc = table[((crc ^ u32::from(b)) & 0xFF) as usize] ^ (crc >> 8);
        }
        png.extend_from_slice(&body);
        png.extend_from_slice(&(crc ^ 0xFFFF_FFFF).to_be_bytes());
    };
    let mut header = image.width.to_be_bytes().to_vec();
    header.extend_from_slice(&image.height.to_be_bytes());
    header.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &header);
    let row = image.width as usize * 3;
    let mut raw = Vec::new();
    for y in 0..image.height as usize {
        raw.push(0);
        raw.extend_from_slice(&image.pixels[y * row..(y + 1) * row]);
    }
    let mut zlib = vec![0x78, 0x01];
    let mut at = 0;
    while at < raw.len() {
        let n = (raw.len() - at).min(65535);
        zlib.push(u8::from(at + n == raw.len()));
        let n16 = n as u16;
        zlib.extend_from_slice(&n16.to_le_bytes());
        zlib.extend_from_slice(&(!n16).to_le_bytes());
        zlib.extend_from_slice(&raw[at..at + n]);
        at += n;
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &v in &raw {
        a = (a + u32::from(v)) % 65521;
        b = (b + a) % 65521;
    }
    zlib.extend_from_slice(&((b << 16) | a).to_be_bytes());
    chunk(&mut png, b"IDAT", &zlib);
    chunk(&mut png, b"IEND", &[]);
    png
}

#[test]
fn grids_match_the_processor() {
    let config = bolt_vision();
    let golden = common::golden("vision_image_golden.json");
    for entry in golden.find("grids").unwrap().items() {
        let v = entry.items();
        let (width, height) = (v[0].as_size(0) as u32, v[1].as_size(0) as u32);
        let want = ImageGrid {
            rows: v[2].as_size(0) as u32,
            cols: v[3].as_size(0) as u32,
        };
        let got = plan_grid(&config, width, height, 0);
        if want.rows == 0 {
            assert!(got.is_none(), "grid of {width}x{height} should not exist");
        } else {
            assert_eq!(got, Some(want), "grid of {width}x{height}");
        }
    }
}

#[test]
fn preprocessed_pixels_match_the_processor() {
    let config = bolt_vision();
    let golden = common::golden("vision_image_golden.json");
    let samples = golden.member_size("samples", 0);
    for entry in golden.find("images").unwrap().items() {
        let width = entry.member_size("width", 0) as u32;
        let height = entry.member_size("height", 0) as u32;
        let grid = ImageGrid {
            rows: entry.member_size("rows", 0) as u32,
            cols: entry.member_size("cols", 0) as u32,
        };
        let values = preprocess_image(&config, &synthetic(width, height), &grid).unwrap();
        assert_eq!(
            values.len(),
            grid.patches() as usize * config.patch_values() as usize,
            "patch count of {width}x{height}"
        );
        let bits = entry.find("bits").unwrap().items();
        assert_eq!(bits.len(), samples, "sample count of {width}x{height}");
        let mut mismatched = 0;
        for (k, want) in bits.iter().enumerate() {
            let index = ((k as u64 * 2_654_435_761) % values.len() as u64) as usize;
            if values[index].to_bits() != want.as_size(0) as u32 {
                mismatched += 1;
            }
        }
        let sum: f64 = values.iter().map(|&v| f64::from(v)).sum();
        let abs_sum: f64 = values.iter().map(|&v| f64::from(v).abs()).sum();
        let want_sum = entry.member_double("sum", 0.0);
        let want_abs = entry.member_double("abs_sum", 0.0);
        println!(
            "{width}x{height} -> {}x{}: {mismatched}/{samples} samples differ, sum {sum} vs {want_sum}",
            grid.rows, grid.cols
        );
        assert_eq!(
            mismatched, 0,
            "preprocessed pixels of {width}x{height} differ"
        );
        assert!(
            (sum - want_sum).abs() <= 1e-6 * want_abs
                && (abs_sum - want_abs).abs() <= 1e-9 * want_abs,
            "pixel sums of {width}x{height} differ"
        );
    }
}

#[test]
fn prompt_expansion_positions_and_identities() {
    let config = bolt_vision();
    let framing = ImageFraming {
        placeholder: 18,
        start: 19,
        end: 20,
    };
    let png = |w, h| Some(Arc::new(encode_png(&synthetic(w, h))));
    let rendered = [1, 2, 18, 3, 18, 4];
    let images = vec![
        VisionInput {
            bytes: png(640, 480),
            low_detail: false,
        },
        VisionInput {
            bytes: png(64, 64),
            low_detail: false,
        },
    ];
    let prompt = expand_images(&config, &framing, &rendered, &images, "encoder-a", true).unwrap();
    // 640x480 is 30x40 patches (300 tokens); 64x64 grows to 32x32 (256).
    assert!(
        prompt.images.len() == 2 && prompt.images[0].rows == 300 && prompt.images[1].rows == 256,
        "image token counts"
    );
    assert!(
        prompt.images[0].offset == 3 && prompt.images[1].offset == 306,
        "image offsets"
    );
    assert_eq!(prompt.tokens.len(), 6 + 301 + 257, "expanded length");
    let t = &prompt.tokens;
    assert!(
        t[2] == 19 && t[303] == 20 && t[304] == 3 && t[305] == 19 && t[562] == 20 && t[563] == 4,
        "framing tokens"
    );
    for image in &prompt.images {
        for i in 0..image.rows {
            assert_eq!(
                t[(image.offset + i) as usize],
                18,
                "image rows hold <image>"
            );
        }
        assert_eq!(
            image.patches.as_ref().map(|p| p.len()),
            Some(image.grid.patches() as usize * config.patch_values() as usize),
            "preprocessed patches"
        );
    }
    assert!(
        prompt.expanded(0) == 0
            && prompt.expanded(2) == 2
            && prompt.expanded(3) == 304
            && prompt.expanded(5) == 563
            && prompt.expanded(6) == 564,
        "rendered to expanded positions"
    );
    // Identities: nothing before the first image, the first image before the
    // second, every image for the whole prompt.
    assert!(
        prompt.prefix_identities.len() == 2
            && prompt.prefix_identities[0].is_empty()
            && prompt.prefix_identities[1].len() == 32
            && prompt.cache_identity.len() == 32
            && prompt.cache_identity != prompt.prefix_identities[1],
        "prefix identities"
    );
    // A different second image keeps what precedes it.
    let mut changed = images.clone();
    let mut pixels = synthetic(64, 64);
    pixels.pixels[100] ^= 1;
    changed[1].bytes = Some(Arc::new(encode_png(&pixels)));
    let other = expand_images(&config, &framing, &rendered, &changed, "encoder-a", false).unwrap();
    assert!(
        other.tokens == prompt.tokens && other.images[1].patches.is_none(),
        "sizing keeps the tokens without pixels"
    );
    assert!(
        other.prefix_identities[1] == prompt.prefix_identities[1]
            && other.cache_identity != prompt.cache_identity,
        "a changed image changes only the identities that include it"
    );
    // Another encoder changes every image identity.
    let encoder = expand_images(&config, &framing, &rendered, &images, "encoder-b", false).unwrap();
    assert!(
        encoder.prefix_identities[1] != prompt.prefix_identities[1]
            && encoder.cache_identity != prompt.cache_identity,
        "the encoder is part of the identity"
    );
    // Low detail caps a large image at min_patches.
    let large = [VisionInput {
        bytes: png(2000, 1400),
        low_detail: true,
    }];
    let one = [18];
    let low = expand_images(&config, &framing, &one, &large, "e", false).unwrap();
    let patches = low.images[0].grid.patches();
    assert!(
        patches <= config.min_patches && patches > config.min_patches / 2,
        "low detail grid"
    );
    // Placeholders and images must agree, and images must decode.
    assert!(
        expand_images(&config, &framing, &rendered, &images[..1], "e", false).is_err(),
        "placeholder count mismatch"
    );
    let broken = [VisionInput {
        bytes: Some(Arc::new(vec![1, 2, 3])),
        low_detail: false,
    }];
    assert!(
        expand_images(&config, &framing, &one, &broken, "e", false).is_err(),
        "undecodable image"
    );
}
