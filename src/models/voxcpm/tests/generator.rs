//! The C++ extension's `generator_test.cpp`: the GPU generator against the
//! float32 reference run of `tools/reference/generator_golden.py` (prefill,
//! three zero-shot patches with the reference's noise, a reference-mode
//! prompt and one DiT call), from the Hugging Face weights and from
//! audio.cpp's bfloat16 and Q8_0 GGUFs.
//! cargo test --profile gpu-test -p gufo-voxcpm --features gpu,models --test generator

#![cfg(all(feature = "gpu", feature = "models"))]
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::Mutex;

use gufo_voxcpm::config::parse_config;
use gufo_voxcpm::generator::{Generator, Prompt};
use gufo_voxcpm::model_files::ModelFiles;
use gufo_voxcpm::safetensors::{DType, SafetensorsFile};

/// One model on the device at a time.
static SERIAL: Mutex<()> = Mutex::new(());

/// The checkpoint `variable` names: the model tests fail (rather than skip)
/// without it.
fn model(variable: &str) -> PathBuf {
    std::env::var_os(variable)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{variable} must name the VoxCPM2 checkpoint"))
}

fn golden() -> SafetensorsFile {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/generator_golden.safetensors");
    SafetensorsFile::open(&path).unwrap()
}

fn get(golden: &SafetensorsFile, name: &str) -> Vec<f32> {
    golden
        .find(name)
        .unwrap_or_else(|| panic!("golden lacks {name}"))
        .f32_values()
        .unwrap()
}

fn ints(golden: &SafetensorsFile, name: &str) -> Vec<i32> {
    // std::lround: halves away from zero.
    get(golden, name)
        .into_iter()
        .map(|value| value.round() as i32)
        .collect()
}

/// `[rows][columns]` to `[columns][rows]` for each of `batch` matrices.
fn transpose(values: &[f32], rows: usize, columns: usize, batch: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; values.len()];
    for b in 0..batch {
        for r in 0..rows {
            for c in 0..columns {
                result[b * rows * columns + c * rows + r] =
                    values[b * rows * columns + r * columns + c];
            }
        }
    }
    result
}

/// Relative L2 errors against the reference, failing above their limits.
struct Checker {
    /// Q8_0 weights move every tensor by about 1 % against the bfloat16
    /// reference; their limits are five times wider.
    tolerance: f64,
    failures: Vec<String>,
}

impl Checker {
    fn compare(&mut self, name: &str, actual: &[f32], expected: &[f32], limit: f64) {
        if actual.len() != expected.len() {
            self.failures.push(format!(
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
        let limit = limit * self.tolerance;
        let ok = relative.is_finite() && relative <= limit;
        println!(
            "{} {name:<26} rel {relative:.2e} max {worst:.2e} (limit {limit:.2e})",
            if ok { "ok  " } else { "FAIL" }
        );
        if !ok {
            self.failures
                .push(format!("{name}: rel {relative:.2e} above {limit:.2e}"));
        }
    }
}

fn prompt_of(golden: &SafetensorsFile, prefix: &str) -> Prompt {
    Prompt {
        tokens: ints(golden, &format!("{prefix}.text_token")),
        text_mask: ints(golden, &format!("{prefix}.text_mask")),
        audio_mask: ints(golden, &format!("{prefix}.feat_mask")),
        feats: get(golden, &format!("{prefix}.feat_in")),
    }
}

fn run_case(
    generator: &mut Generator,
    golden: &SafetensorsFile,
    prefix: &str,
    patches: usize,
    checker: &mut Checker,
) {
    let patch = generator.config().patch_size;
    let feat = generator.config().feat_dim;
    generator.set_trace(true);
    generator.prefill(&prompt_of(golden, prefix)).unwrap();
    let trace = generator.trace().unwrap().clone();
    checker.compare(
        &format!("{prefix} lm prefill"),
        &trace.lm_prefill,
        &get(golden, &format!("{prefix}.lm.0")),
        1e-2,
    );
    checker.compare(
        &format!("{prefix} residual prefill"),
        &trace.residual_prefill,
        &get(golden, &format!("{prefix}.residual.0")),
        1e-2,
    );
    for step in 0..patches {
        // The first patch starts from the reference's prompt; later ones
        // carry the earlier patches' deviations (bf16 GEMM operands against
        // the float32 reference) through the autoregressive state.
        let limit = if step == 0 { 1e-2 } else { 1e-1 };
        let noise = transpose(
            &get(golden, &format!("{prefix}.noise.{step}")),
            feat,
            patch,
            1,
        );
        let mut latent = vec![0.0f32; patch * feat];
        let mut stop = [0.0f32; 2];
        generator
            .step(&noise, 10, 2.0, Some(&mut latent), &mut stop)
            .unwrap();
        let trace = generator.trace().unwrap();
        checker.compare(
            &format!("{prefix} mu {step}"),
            trace.mu.last().unwrap(),
            &get(golden, &format!("{prefix}.mu.{step}")),
            limit,
        );
        checker.compare(
            &format!("{prefix} stop {step}"),
            &stop,
            &get(golden, &format!("{prefix}.stop.{step}")),
            limit,
        );
        checker.compare(
            &format!("{prefix} latent {step}"),
            &latent,
            &transpose(
                &get(golden, &format!("{prefix}.feat.{step}")),
                feat,
                patch,
                1,
            ),
            2.0 * limit,
        );
        // The LM steps take this patch's latent, so they already carry its
        // deviation.
        let trace = generator.trace().unwrap();
        checker.compare(
            &format!("{prefix} lm step {step}"),
            trace.lm_step.last().unwrap(),
            &get(golden, &format!("{prefix}.lm_step.{step}")),
            1e-1,
        );
        checker.compare(
            &format!("{prefix} residual step {step}"),
            trace.residual_step.last().unwrap(),
            &get(golden, &format!("{prefix}.residual_step.{step}")),
            1e-1,
        );
    }
    generator.set_trace(false);
}

/// The generator of the checkpoint `variable` names against the golden
/// tensors.
fn check(variable: &str) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let files = ModelFiles::open(&model(variable)).unwrap();
    let config = parse_config(files.config()).unwrap();
    let q8 = files
        .model("base_lm.layers.0.mlp.down_proj.weight")
        .is_some_and(|tensor| tensor.dtype == DType::Q8_0);
    let mut checker = Checker {
        tolerance: if q8 { 5.0 } else { 1.0 },
        failures: Vec::new(),
    };
    let golden = golden();
    let mut generator = Generator::create(&config, &files).unwrap();
    let patch = config.patch_size;
    let feat = config.feat_dim;
    // One DiT evaluation with its CFG pair.
    let mu = get(&golden, "dit.mu");
    let x = transpose(&get(&golden, "dit.x"), feat, patch, 1);
    let cond = transpose(&get(&golden, "dit.cond"), feat, patch, 1);
    let mut velocity = vec![0.0f32; 2 * patch * feat];
    generator
        .velocity(&mu, &cond, &x, get(&golden, "dit.t")[0], &mut velocity)
        .unwrap();
    checker.compare(
        "dit velocity",
        &velocity,
        &transpose(&get(&golden, "dit.out"), feat, patch, 2),
        1e-2,
    );
    run_case(&mut generator, &golden, "zs", 3, &mut checker);
    run_case(&mut generator, &golden, "ref", 1, &mut checker);
    assert!(
        checker.failures.is_empty(),
        "{variable}: {:#?}",
        checker.failures
    );
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
