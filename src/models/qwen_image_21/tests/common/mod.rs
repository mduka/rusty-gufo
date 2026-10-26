//! The checkpoint and the parity cases shared by the model tests. The tests
//! fail (rather than skip) when the checkpoint is missing: a missing model is
//! not a pass.

#![allow(dead_code, clippy::unwrap_used)]

use std::path::PathBuf;

use gufo_core::json::{self, Value};
use gufo_qwen_image_21::{Image, Request};

/// `QWEN_IMAGE_21_ROOT`, else the snapshot pinned in UPSTREAM.md in the
/// Hugging Face cache.
pub fn model_root() -> PathBuf {
    std::env::var_os("QWEN_IMAGE_21_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(
                ".cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots/\
                 b3179ad355be050328e483a9dfdd9e60cd62adfa",
            )
        })
}

pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

pub fn cases() -> Value {
    json::parse(&std::fs::read(fixture("cases.json")).unwrap()).unwrap()
}

/// The synthetic reference image of cases.json.
pub fn pattern(width: i32, height: i32, pattern: u32) -> Image {
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height as u32 {
        for x in 0..width as u32 {
            rgba.push((x.wrapping_mul(3 + pattern).wrapping_add(5 * y) & 255) as u8);
            rgba.push((x.wrapping_mul(y).wrapping_add(17 * pattern) & 255) as u8);
            rgba.push(((x ^ y).wrapping_mul(7).wrapping_add(31 * pattern) & 255) as u8);
            rgba.push(if pattern % 2 == 1 {
                (((x + 2 * y) * 3) & 255) as u8
            } else {
                255
            });
        }
    }
    Image {
        width,
        height,
        rgba,
    }
}

/// One generation case: its name, request and observation filter.
pub struct Case {
    pub name: String,
    pub request: Request,
    pub all: bool,
}

pub fn generations() -> Vec<Case> {
    let cases = cases();
    cases
        .find("generations")
        .unwrap()
        .items()
        .iter()
        .map(|item| Case {
            name: item.member_str("name", ""),
            request: Request {
                prompt: item.member_str("prompt", ""),
                images: item
                    .find("images")
                    .unwrap()
                    .items()
                    .iter()
                    .map(|image| {
                        pattern(
                            image.member_double("width", 0.0) as i32,
                            image.member_double("height", 0.0) as i32,
                            image.member_double("pattern", 0.0) as u32,
                        )
                    })
                    .collect(),
                width: item.member_double("width", 0.0) as i32,
                height: item.member_double("height", 0.0) as i32,
                steps: item.member_double("steps", 0.0) as i32,
                seed: item.member_double("seed", 0.0) as u64,
            },
            all: item.member_str("observe", "") == "all",
        })
        .collect()
}

/// The observer filter of `tests/reference/cpp_reference.cpp`.
pub fn coarse(name: &str) -> bool {
    matches!(
        name,
        "noise"
            | "sigmas"
            | "text.embedding"
            | "text.output"
            | "vision.embedding"
            | "vision.output"
            | "vae.encoded"
            | "vae.decoder.mid"
            | "vae.decoded"
    ) || name.starts_with("latent.")
        || name.ends_with(".output")
        || name.ends_with(".modulation")
}
