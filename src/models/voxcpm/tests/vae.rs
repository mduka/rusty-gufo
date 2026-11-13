//! The C++ extension's `vae_test.cpp`: AudioVAE V2 on the GPU against the
//! float32 reference, decoding the zero-shot latents in patches (streamed)
//! and at once, and encoding the synthetic 16 kHz signal, from the Hugging
//! Face weights and from audio.cpp's bfloat16 and Q8_0 GGUFs.
//! cargo test --profile gpu-test -p gufo-voxcpm --features gpu,models --test vae

#![cfg(all(feature = "gpu", feature = "models"))]
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::Mutex;

use gufo_voxcpm::config::parse_config;
use gufo_voxcpm::model_files::ModelFiles;
use gufo_voxcpm::safetensors::{DType, SafetensorsFile};
use gufo_voxcpm::vae::AudioVae;

/// One model on the device at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn model(variable: &str) -> PathBuf {
    std::env::var_os(variable)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{variable} must name the VoxCPM2 checkpoint"))
}

fn get(golden: &SafetensorsFile, name: &str) -> Vec<f32> {
    golden
        .find(name)
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .f32_values()
        .unwrap()
}

fn transpose(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; values.len()];
    for r in 0..rows {
        for c in 0..columns {
            result[c * rows + r] = values[r * columns + c];
        }
    }
    result
}

/// The relative L2 error of `actual` against `expected`, or why it fails.
fn compare(name: &str, actual: &[f32], expected: &[f32], limit: f64, failures: &mut Vec<String>) {
    if actual.len() != expected.len() {
        failures.push(format!(
            "{name}: {} values, {} expected",
            actual.len(),
            expected.len()
        ));
        return;
    }
    let (mut error, mut norm, mut worst) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &e) in actual.iter().zip(expected) {
        let difference = f64::from(a) - f64::from(e);
        error += difference * difference;
        norm += f64::from(e) * f64::from(e);
        worst = worst.max(difference.abs());
    }
    let relative = (error / norm.max(1e-30)).sqrt();
    let ok = relative.is_finite() && relative <= limit;
    println!(
        "{} {name:<20} rel {relative:.2e} max {worst:.2e} (limit {limit:.2e})",
        if ok { "ok  " } else { "FAIL" }
    );
    if !ok {
        failures.push(format!("{name}: rel {relative:.2e} above {limit:.2e}"));
    }
}

fn check(variable: &str) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let files = ModelFiles::open(&model(variable)).unwrap();
    let config = parse_config(files.config()).unwrap();
    // The reference runs float32 weights; GGUFs may store some in float16
    // or bfloat16 (audio.cpp's bf16 file holds the whole AudioVAE in
    // bfloat16).
    let weight = files.vae("decoder.model.1.weight_v");
    let embed = files.vae("decoder.sr_cond_model.2.scale_embed.weight");
    let scale = if weight.is_some_and(|tensor| tensor.dtype != DType::F32) {
        200.0
    } else if embed.is_some_and(|tensor| tensor.dtype != DType::F32) {
        10.0
    } else {
        1.0
    };
    let golden = SafetensorsFile::open(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/generator_golden.safetensors"),
    )
    .unwrap();
    let mut vae = AudioVae::create(&config, &files).unwrap();
    let latent = config.vae.latent_dim;
    let stored = get(&golden, "zs.latents");
    let latents = transpose(&stored, latent, stored.len() / latent);
    let frames = latents.len() / latent;
    let expected = get(&golden, "vae.decoded");
    let mut failures = Vec::new();

    let mut streamed = Vec::new();
    vae.reset().unwrap();
    for frame in (0..frames).step_by(config.patch_size) {
        let end = (frame + config.patch_size) * latent;
        vae.decode(&latents[frame * latent..end], &mut streamed)
            .unwrap();
    }
    compare(
        "decode streamed",
        &streamed,
        &expected,
        1e-4 * scale,
        &mut failures,
    );

    let mut whole = Vec::new();
    vae.reset().unwrap();
    vae.decode(&latents, &mut whole).unwrap();
    compare(
        "decode at once",
        &whole,
        &expected,
        1e-4 * scale,
        &mut failures,
    );

    let signal = get(&golden, "vae.signal");
    let encoded = vae.encode(&signal).unwrap();
    let reference = get(&golden, "vae.encoded");
    compare(
        "encode",
        &encoded,
        &transpose(&reference, latent, reference.len() / latent),
        1e-4 * scale,
        &mut failures,
    );
    assert!(failures.is_empty(), "{variable}: {failures:#?}");
}

/// `openbmb/VoxCPM2`, the Hugging Face directory.
#[test]
fn hugging_face_weights() {
    check("GUFO_VOXCPM_MODEL");
}

/// audio.cpp's `voxcpm2-bf16.gguf`.
#[test]
fn bf16_gguf() {
    check("GUFO_VOXCPM_GGUF");
}

/// audio.cpp's `voxcpm2-q8_0.gguf`.
#[test]
fn q8_0_gguf() {
    check("GUFO_VOXCPM_Q8_GGUF");
}
