//! PNG decoding against the C++ `DecodeImage` (libpng's simplified API) on a
//! corpus of colour types, bit depths, transparency, gamma and sBIT chunks
//! and damaged files, generated here byte for byte. The fixture
//! `png-decode.json` holds the C++ result of each file: the RGBA SHA-256 and
//! size, or the error message.
//!
//! Recording: `write_png_corpus` (ignored) writes the files to
//! `QWEN_IMAGE_PNG_CORPUS`, `tests/reference/cpp_reference decode OUT FILES...`
//! decodes them, and `record_png_fixture` (ignored) reads `OUT`.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use gufo_core::json::{self, Value};
use gufo_qwen_image_21::image::decode_image;
use png::{BitDepth, ColorType, ScaledFloat, SrgbRenderingIntent};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Deterministic sample bytes.
fn samples(count: usize, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) as u8
        })
        .collect()
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Inserts a chunk right after IHDR (and after any chunk inserted before).
fn insert_chunk(png: &mut Vec<u8>, at: &mut usize, kind: &[u8; 4], data: &[u8]) {
    let mut chunk = (data.len() as u32).to_be_bytes().to_vec();
    chunk.extend_from_slice(kind);
    chunk.extend_from_slice(data);
    let crc = crc32(&chunk[4..]);
    chunk.extend_from_slice(&crc.to_be_bytes());
    let length = chunk.len();
    png.splice(*at..*at, chunk);
    *at += length;
}

struct Spec {
    name: &'static str,
    width: u32,
    height: u32,
    color: ColorType,
    depth: BitDepth,
    palette: Option<Vec<u8>>,
    trns: Option<Vec<u8>>,
    gamma: Option<u32>,
    srgb: bool,
    /// Raw chunks inserted after IHDR, in order.
    chunks: Vec<(&'static [u8; 4], Vec<u8>)>,
}

impl Spec {
    fn new(name: &'static str, color: ColorType, depth: BitDepth) -> Self {
        Self {
            name,
            width: 13,
            height: 7,
            color,
            depth,
            palette: None,
            trns: None,
            gamma: None,
            srgb: false,
            chunks: Vec::new(),
        }
    }

    fn encode(&self) -> Vec<u8> {
        let channels = match self.color {
            ColorType::Grayscale | ColorType::Indexed => 1,
            ColorType::GrayscaleAlpha => 2,
            ColorType::Rgb => 3,
            ColorType::Rgba => 4,
        };
        let bits = self.depth as usize * channels;
        let row = (self.width as usize * bits).div_ceil(8);
        let mut data = samples(
            row * self.height as usize,
            self.name.len() as u32 * 7 + self.width,
        );
        if self.color == ColorType::Indexed && self.depth == BitDepth::Eight {
            // Indices within the palette; low bit depths use full palettes.
            let entries = (self.palette.as_ref().unwrap().len() / 3) as u8;
            for byte in &mut data {
                *byte %= entries;
            }
        }
        let mut bytes = Vec::new();
        let mut encoder = png::Encoder::new(&mut bytes, self.width, self.height);
        encoder.set_color(self.color);
        encoder.set_depth(self.depth);
        if let Some(palette) = &self.palette {
            encoder.set_palette(palette.clone());
        }
        if let Some(trns) = &self.trns {
            encoder.set_trns(trns.clone());
        }
        if let Some(gamma) = self.gamma {
            encoder.set_source_gamma(ScaledFloat::from_scaled(gamma));
        }
        if self.srgb {
            encoder.set_source_srgb(SrgbRenderingIntent::Perceptual);
        }
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&data).unwrap();
        writer.finish().unwrap();
        let mut at = 33; // signature and IHDR
        for (kind, chunk) in &self.chunks {
            insert_chunk(&mut bytes, &mut at, kind, chunk);
        }
        bytes
    }
}

/// The corpus: every file's name and bytes.
fn corpus() -> Vec<(String, Vec<u8>)> {
    use BitDepth::{Eight, Four, One, Sixteen, Two};
    use ColorType::{Grayscale, GrayscaleAlpha, Indexed, Rgb, Rgba};
    let palette: Vec<u8> = samples(20 * 3, 99);
    let mut specs = vec![
        Spec::new("rgba8", Rgba, Eight),
        Spec::new("rgb8", Rgb, Eight),
        Spec::new("gray8", Grayscale, Eight),
        Spec::new("grayalpha8", GrayscaleAlpha, Eight),
        Spec::new("gray1", Grayscale, One),
        Spec::new("gray2", Grayscale, Two),
        Spec::new("gray4", Grayscale, Four),
        Spec::new("rgba16", Rgba, Sixteen),
        Spec::new("rgb16", Rgb, Sixteen),
        Spec::new("gray16", Grayscale, Sixteen),
        Spec::new("grayalpha16", GrayscaleAlpha, Sixteen),
    ];
    let indexed = |name, depth: BitDepth, entries: usize| Spec {
        palette: Some(palette[..entries * 3].to_vec()),
        trns: Some(vec![0, 40, 255, 128, 7, 200, 13][..entries.min(7)].to_vec()),
        ..Spec::new(name, Indexed, depth)
    };
    specs.push(indexed("palette8", Eight, 20));
    specs.push(indexed("palette4", Four, 16));
    specs.push(indexed("palette2", Two, 4));
    specs.push(indexed("palette1", One, 2));
    specs.push(Spec {
        trns: Some(vec![0, 5]),
        ..Spec::new("gray4-trns", Grayscale, Four)
    });
    specs.push(Spec {
        trns: Some(vec![0, 77]),
        ..Spec::new("gray8-trns", Grayscale, Eight)
    });
    let rgb8 = samples(3, 5);
    specs.push(Spec {
        trns: Some(vec![0, rgb8[0], 0, rgb8[1], 0, rgb8[2]]),
        ..Spec::new("rgb8-trns", Rgb, Eight)
    });
    specs.push(Spec {
        trns: Some(vec![0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc]),
        ..Spec::new("rgb16-trns", Rgb, Sixteen)
    });
    specs.push(Spec {
        trns: Some(vec![0x40, 0x00]),
        ..Spec::new("gray16-trns", Grayscale, Sixteen)
    });
    for (name, color, depth, gamma) in [
        ("rgb8-linear", Rgb, Eight, 100_000),
        ("rgb8-srgb-gamma", Rgb, Eight, 45_455),
        ("rgba8-gamma50000", Rgba, Eight, 50_000),
        ("gray8-linear", Grayscale, Eight, 100_000),
        ("grayalpha8-gamma30000", GrayscaleAlpha, Eight, 30_000),
        ("rgb16-srgb-gamma", Rgb, Sixteen, 45_455),
        ("gray16-linear", Grayscale, Sixteen, 100_000),
        ("rgba16-gamma80000", Rgba, Sixteen, 80_000),
        ("gray2-linear", Grayscale, Two, 100_000),
    ] {
        specs.push(Spec {
            gamma: Some(gamma),
            ..Spec::new(name, color, depth)
        });
    }
    specs.push(Spec {
        gamma: Some(100_000),
        ..indexed("palette8-linear", Eight, 20)
    });
    specs.push(Spec {
        srgb: true,
        ..Spec::new("rgb16-srgb", Rgb, Sixteen)
    });
    specs.push(Spec {
        srgb: true,
        ..Spec::new("rgba8-srgb", Rgba, Eight)
    });
    for (name, color, bits) in [
        ("rgb16-sbit10", Rgb, vec![10, 10, 10]),
        ("rgb16-sbit8", Rgb, vec![8, 12, 8]),
        ("gray16-sbit7", Grayscale, vec![7]),
        ("grayalpha16-sbit12", GrayscaleAlpha, vec![12, 16]),
        ("rgba16-sbit16", Rgba, vec![16, 16, 16, 16]),
    ] {
        specs.push(Spec {
            chunks: vec![(b"sBIT", bits)],
            ..Spec::new(name, color, Sixteen)
        });
    }
    // sRGB wins over gAMA in either order.
    specs.push(Spec {
        chunks: vec![
            (b"sRGB", vec![0]),
            (b"gAMA", 100_000u32.to_be_bytes().to_vec()),
        ],
        ..Spec::new("rgb8-srgb-then-linear", Rgb, Eight)
    });
    specs.push(Spec {
        chunks: vec![
            (b"gAMA", 100_000u32.to_be_bytes().to_vec()),
            (b"sRGB", vec![0]),
        ],
        ..Spec::new("rgb8-linear-then-srgb", Rgb, Eight)
    });
    specs.push(Spec {
        chunks: vec![(b"gAMA", 0u32.to_be_bytes().to_vec())],
        ..Spec::new("rgb8-gamma0", Rgb, Eight)
    });
    specs.push(Spec {
        width: 1,
        height: 1,
        ..Spec::new("rgba8-1x1", Rgba, Eight)
    });
    specs.push(Spec {
        width: 301,
        height: 3,
        ..Spec::new("rgba8-wide", Rgba, Eight)
    });
    let mut files: Vec<(String, Vec<u8>)> = specs
        .iter()
        .map(|spec| (format!("{}.png", spec.name), spec.encode()))
        .collect();
    // Damaged files.
    let good = Spec::new("rgb8-damaged", Rgb, Eight).encode();
    files.push(("truncated.png".into(), good[..good.len() - 30].to_vec()));
    let mut crc = good.clone();
    let idat = crc.windows(4).position(|w| w == b"IDAT").unwrap();
    crc[idat + 6] ^= 0x55;
    files.push(("bad-idat-crc.png".into(), crc));
    let mut header = good;
    header[16..20].copy_from_slice(&0u32.to_be_bytes());
    files.push(("zero-width.png".into(), header));
    files
}

fn corpus_directory() -> PathBuf {
    std::env::var_os("QWEN_IMAGE_PNG_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../target/image-ref/png")
        })
}

#[test]
#[ignore = "writes the PNG corpus for tests/reference/cpp_reference decode"]
fn write_png_corpus() {
    let directory = corpus_directory();
    std::fs::create_dir_all(&directory).unwrap();
    for (name, bytes) in corpus() {
        std::fs::write(directory.join(name), bytes).unwrap();
    }
}

#[test]
#[ignore = "writes the fixture from `cpp_reference decode` output (QWEN_IMAGE_PNG_DECODED)"]
fn record_png_fixture() {
    let decoded = PathBuf::from(std::env::var_os("QWEN_IMAGE_PNG_DECODED").unwrap());
    let listing = std::fs::read_to_string(decoded.join("decoded.txt")).unwrap();
    let mut entries = Vec::new();
    for line in listing.lines() {
        let (name, rest) = line.split_once(' ').unwrap();
        let entry = if let Some(message) = rest.strip_prefix("error ") {
            Value::object().with("name", name).with("error", message)
        } else {
            let (width, height) = rest.split_once(' ').unwrap();
            let rgba = std::fs::read(decoded.join(format!("{name}.rgba"))).unwrap();
            Value::object()
                .with("name", name)
                .with("width", width.parse::<f64>().unwrap())
                .with("height", height.parse::<f64>().unwrap())
                .with("rgba", hex(&rgba))
        };
        entries.push(entry);
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/png-decode.json");
    std::fs::write(path, Value::from(entries).dump().unwrap() + "\n").unwrap();
}

#[test]
fn png_decoding_matches_libpng_simplified_reads() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/png-decode.json");
    let fixture = json::parse(&std::fs::read(path).unwrap()).unwrap();
    let files = corpus();
    assert_eq!(files.len(), fixture.len());
    let mut failures = Vec::new();
    for (name, bytes) in &files {
        let want = fixture
            .items()
            .iter()
            .find(|entry| entry.member_str("name", "") == *name)
            .unwrap();
        let outcome = match decode_image(bytes) {
            Ok(image) => format!("{} {} {}", image.width, image.height, hex(&image.rgba)),
            Err(error) => format!("error {}", error.message()),
        };
        let expected = if let Some(error) = want.find("error") {
            format!("error {}", error.str())
        } else {
            format!(
                "{} {} {}",
                want.member_double("width", 0.0),
                want.member_double("height", 0.0),
                want.member_str("rgba", "")
            )
        };
        if outcome != expected {
            failures.push(format!("{name}: {outcome} != {expected}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
