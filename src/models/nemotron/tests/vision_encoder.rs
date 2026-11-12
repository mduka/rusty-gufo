//! The C++ extension's `vision_encoder_test.cpp`: the vision encoder on the
//! projector beside the trunk against NVIDIA's float32 and bfloat16 models
//! (`tools/vision_golden.py`), on its synthetic images.
//! GUFO_NEMOTRON_BOLT_GGUF=... cargo test --profile gpu-test -p gufo-nemotron \
//!   --features gpu,models --test vision_encoder -- --nocapture
//! `GUFO_NEMOTRON_VISION_OPERAND=bf16` selects BF16 operands;
//! `GUFO_NEMOTRON_VISION_REFERENCE=DIR` compares every stage with the
//! reference's dumps there (`tools/vision_reference.py`).

#![cfg(all(feature = "gpu", feature = "models"))]
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use gufo_core::image::Image;
use gufo_core::json::Value;
use gufo_hip::MappedFile;
use gufo_nemotron::config::Config;
use gufo_nemotron::vision_encoder::{VisionEncoder, VisionEncoderOptions};
use gufo_nemotron::vision_gemm::VisionOperand;
use gufo_nemotron::vision_image::{plan_grid, preprocess_image};
use gufo_nemotron::weights::GgufShard;

/// `tools/vision_golden.py`'s synthetic image: gradients plus hash noise
/// (`tests/vision_synthetic.hpp`).
fn synthetic(width: u32, height: u32) -> Image {
    let mut pixels = vec![0u8; width as usize * height as usize * 3];
    for y in 0..u64::from(height) {
        for x in 0..u64::from(width) {
            for c in 0..3u64 {
                let h = (x.wrapping_mul(73_856_093)
                    ^ y.wrapping_mul(19_349_663)
                    ^ c.wrapping_mul(83_492_791))
                    & 0xFFFF_FFFF;
                let v = (x * 255) / u64::from(width)
                    + (y * 128) / u64::from(height)
                    + c * 40
                    + (h >> 13) % 48;
                pixels[((y * u64::from(width) + x) * 3 + c) as usize] = (v % 256) as u8;
            }
        }
    }
    Image {
        width,
        height,
        pixels,
    }
}

/// Root-mean-square of `a - b` relative to that of `b`.
fn relative_rms(a: &[f64], b: &[f64]) -> f64 {
    let (mut diff, mut norm) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        diff += (x - y) * (x - y);
        norm += y * y;
    }
    (diff / norm).sqrt()
}

fn numbers(list: &Value) -> Vec<f64> {
    list.items().iter().map(|v| v.as_double(0.0)).collect()
}

fn read_dump(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .map(|bytes| {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        })
        .unwrap_or_default()
}

/// Relative RMS error of `got` against a dump, and the worst cosine
/// similarity of a row of `width`.
fn compare_stage(label: &str, got: &[f32], want: &[f32], width: usize, failures: &mut Vec<String>) {
    if want.len() != got.len() {
        failures.push(format!(
            "{label}: {} values, the reference has {}",
            got.len(),
            want.len()
        ));
        return;
    }
    let (mut diff, mut norm, mut worst_cos) = (0.0f64, 0.0f64, 1.0f64);
    for row in 0..got.len() / width {
        let (mut dot, mut ga, mut wa) = (0.0f64, 0.0f64, 0.0f64);
        for c in 0..width {
            let g = f64::from(got[row * width + c]);
            let w = f64::from(want[row * width + c]);
            diff += (g - w) * (g - w);
            norm += w * w;
            dot += g * w;
            ga += g * g;
            wa += w * w;
        }
        worst_cos = worst_cos.min(dot / (ga * wa).sqrt());
    }
    println!(
        "  {label:<34} rel rms {:.4}, min cos {worst_cos:.6}",
        (diff / norm).sqrt()
    );
}

#[test]
fn encoder_matches_nvidias_model() {
    let trunk = common::trunk_path();
    let name = trunk.file_name().unwrap().to_string_lossy().into_owned();
    let projector = trunk.parent().unwrap().join(format!("mmproj-{name}"));
    assert!(
        projector.is_file(),
        "no projector beside the trunk: {}",
        projector.display()
    );
    let config = {
        let file = MappedFile::open(&trunk).unwrap();
        let shard = GgufShard::parse(file.bytes()).unwrap();
        Config::from_gguf(&shard.gguf).unwrap()
    };
    let f16 = std::env::var("GUFO_NEMOTRON_VISION_OPERAND").map_or(true, |o| o != "bf16");
    let options = VisionEncoderOptions {
        operand: if f16 {
            VisionOperand::F16
        } else {
            VisionOperand::Bf16
        },
    };
    let start = Instant::now();
    let encoder = VisionEncoder::load(&projector, &config, options).unwrap();
    let loaded = start.elapsed().as_secs_f64();
    encoder.prepare().unwrap();
    let prepared = start.elapsed().as_secs_f64() - loaded;
    println!(
        "operand {}, loaded in {loaded:.2} s ({} MiB), prepared in {prepared:.2} s",
        if f16 { "f16" } else { "bf16" },
        encoder.resident_bytes() >> 20
    );
    let dumps = std::env::var_os("GUFO_NEMOTRON_VISION_REFERENCE");
    let golden = common::golden("vision_encoder_golden.json");
    let samples = golden.member_size("samples", 0);
    let vc = encoder.config().clone();
    let mut failures = Vec::new();
    for entry in golden.find("images").unwrap().items() {
        let width = entry.member_size("width", 0) as u32;
        let height = entry.member_size("height", 0) as u32;
        let size = format!("{width}x{height}");
        let grid = plan_grid(&vc, width, height, 0);
        if grid.is_none_or(|g| {
            g.rows as usize != entry.member_size("rows", 0)
                || g.cols as usize != entry.member_size("cols", 0)
        }) {
            failures.push(format!("{size}: grid differs from the reference"));
        }
        let Some(grid) = grid else {
            continue;
        };
        let patches = preprocess_image(&vc, &synthetic(width, height), &grid).unwrap();
        let stages: Mutex<BTreeMap<String, Vec<f32>>> = Mutex::new(BTreeMap::new());
        let observer = |stage: &str, values: &[f32]| {
            if dumps.is_some() || stage == "output" {
                stages
                    .lock()
                    .unwrap()
                    .insert(stage.to_string(), values.to_vec());
            }
        };
        // The first encode of a size allocates; time the second.
        let _ = encoder.encode(&patches, &grid, None);
        let begin = Instant::now();
        let embedding = encoder.encode(&patches, &grid, None);
        let seconds = begin.elapsed().as_secs_f64();
        let embedding = match embedding {
            Ok(embedding) => embedding,
            Err(error) => {
                failures.push(format!("{size}: {error}"));
                continue;
            }
        };
        encoder.encode(&patches, &grid, Some(&observer)).unwrap();
        let stages = stages.into_inner().unwrap();
        let output = &stages["output"];
        let tokens = entry.member_size("tokens", 0);
        let projection = vc.projection as usize;
        if embedding.rows() as usize != tokens || output.len() != tokens * projection {
            failures.push(format!("{size}: token count"));
        }
        println!("{size}: {tokens} tokens in {:.2} ms", seconds * 1e3);
        // Sampled outputs and per-token norms against both references.
        let got: Vec<f64> = (0..samples)
            .map(|k| f64::from(output[((k as u64 * 2_654_435_761) % output.len() as u64) as usize]))
            .collect();
        let norms: Vec<f64> = (0..tokens)
            .map(|t| {
                output[t * projection..(t + 1) * projection]
                    .iter()
                    .map(|&v| f64::from(v) * f64::from(v))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        let f32_samples = numbers(entry.find("float32").unwrap().find("samples").unwrap());
        let bf16_samples = numbers(entry.find("bfloat16").unwrap().find("samples").unwrap());
        let f32_norms = numbers(entry.find("float32").unwrap().find("norms").unwrap());
        let bf16_norms = numbers(entry.find("bfloat16").unwrap().find("norms").unwrap());
        let ours_f32 = relative_rms(&got, &f32_samples);
        let ours_bf16 = relative_rms(&got, &bf16_samples);
        let baseline = relative_rms(&bf16_samples, &f32_samples);
        let norm_f32 = relative_rms(&norms, &f32_norms);
        let norm_baseline = relative_rms(&bf16_norms, &f32_norms);
        println!(
            "  samples rel rms: vs float32 {ours_f32:.4}, vs bfloat16 {ours_bf16:.4} \
             (bfloat16 vs float32: {baseline:.4})\n  token norms rel rms: vs float32 \
             {norm_f32:.4} (bfloat16 vs float32: {norm_baseline:.4})"
        );
        // The encoder keeps float32 where NVIDIA's bfloat16 run rounds; with
        // FP16 operands it stays far closer to the float32 model, with BF16
        // ones at least as close.
        let limit = if f16 { 0.005 } else { 1.2 * baseline };
        let norm_limit = if f16 {
            0.005
        } else {
            1.2 * norm_baseline + 1e-3
        };
        if ours_f32 > limit {
            failures.push(format!("{size}: outputs drift from float32"));
        }
        if norm_f32 > norm_limit {
            failures.push(format!("{size}: token norms drift from float32"));
        }
        if let Some(dumps) = &dumps {
            let dumps = Path::new(dumps);
            for dtype in ["float32", "bfloat16"] {
                for (stage, values) in &stages {
                    let want = read_dump(&dumps.join(format!("{size}-{dtype}-{stage}.f32")));
                    if want.is_empty() {
                        continue;
                    }
                    let row = if stage == "output" {
                        projection
                    } else {
                        vc.width as usize
                    };
                    compare_stage(
                        &format!("{dtype} {stage}"),
                        values,
                        &want,
                        row,
                        &mut failures,
                    );
                }
            }
            for stage in ["embed", "block.0", "block.15", "block.31", "output"] {
                let a = read_dump(&dumps.join(format!("{size}-bfloat16-{stage}.f32")));
                let b = read_dump(&dumps.join(format!("{size}-float32-{stage}.f32")));
                if !a.is_empty() {
                    let row = if stage == "output" {
                        projection
                    } else {
                        vc.width as usize
                    };
                    compare_stage(
                        &format!("reference bfloat16 vs float32 {stage}"),
                        &a,
                        &b,
                        row,
                        &mut failures,
                    );
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
