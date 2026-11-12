//! The C++ extension's `generator_test.cpp`: the backbone's logits and the
//! scoring step against the reference model, and greedy unmasking.
//! cargo test --profile gpu-test -p gufo-omnivoice --features gpu,models --test generator

#![cfg(all(feature = "gpu", feature = "models"))]
#![allow(clippy::unwrap_used)]

mod config_samples;

use std::time::Instant;

use gufo_core::json::Value;
use gufo_omnivoice::config::load_config;
use gufo_omnivoice::decoding::{ConditionalInput, DecodingConfig};
use gufo_omnivoice::generator::Generator;

const VOCAB: usize = 1025;

fn flatten(rows: &Value) -> Vec<f64> {
    rows.items()
        .iter()
        .flat_map(|row| row.items().iter().map(|value| value.as_double(0.0)))
        .collect()
}

fn input_from(entry: &Value) -> ConditionalInput {
    let ids: Vec<i32> = flatten(entry.find("input_ids").unwrap())
        .into_iter()
        .map(|id| id as i32)
        .collect();
    ConditionalInput {
        codebooks: 8,
        length: ids.len() / 8,
        audio_start: entry.find("audio_start").unwrap().as_size(0),
        target_frames: entry.find("target").unwrap().as_size(0),
        ids,
    }
}

/// Per `[codebook][frame]` argmax, max and log-sum-exp of logits rows
/// `[first, first + frames)`.
fn stats(logits: &[f32], first: usize, frames: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut argmax, mut maximum, mut lse) = (Vec::new(), Vec::new(), Vec::new());
    for c in 0..8 {
        for f in 0..frames {
            let start = (first + f) * 8 * VOCAB + c * VOCAB;
            let row = &logits[start..start + VOCAB];
            // The first maximum, as std::max_element.
            let (best_index, best) =
                row.iter()
                    .enumerate()
                    .fold(
                        (0, f32::NEG_INFINITY),
                        |(i, m), (j, &v)| {
                            if v > m { (j, v) } else { (i, m) }
                        },
                    );
            let sum: f64 = row
                .iter()
                .map(|&v| (f64::from(v) - f64::from(best)).exp())
                .sum();
            argmax.push(best_index as f64);
            maximum.push(f64::from(best));
            lse.push(f64::from(best) + sum.ln());
        }
    }
    (argmax, maximum, lse)
}

fn compare_logits(name: &str, logits: &[f32], first: usize, frames: usize, expected: &Value) {
    let (argmax, maximum, lse) = stats(logits, first, frames);
    let want_argmax = flatten(expected.find("argmax").unwrap());
    let want_max = flatten(expected.find("max").unwrap());
    let want_lse = flatten(expected.find("logsumexp").unwrap());
    let agree = argmax
        .iter()
        .zip(&want_argmax)
        .filter(|(a, b)| a == b)
        .count();
    let max_error = maximum
        .iter()
        .zip(&want_max)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let lse_error = lse
        .iter()
        .zip(&want_lse)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!(
        "  {name}: argmax {agree}/{}, max |d max| {max_error}, max |d logsumexp| {lse_error}",
        argmax.len()
    );
    assert!(
        agree as f64 / argmax.len() as f64 >= 0.95,
        "{name} argmax agreement"
    );
    assert!(lse_error < 0.1, "{name} log-sum-exp");
}

fn load() -> (Generator, Value) {
    let root = config_samples::model_root();
    let config = load_config(&root).unwrap();
    let start = Instant::now();
    let generator = Generator::load(&config, &root).unwrap();
    println!("  loaded in {:.2} s", start.elapsed().as_secs_f64());
    (generator, config_samples::golden("generator_golden.json"))
}

#[test]
fn forward_and_scoring_match_the_reference() {
    let (mut generator, golden) = load();
    let entry = golden.find("forward").unwrap();
    let input = input_from(entry);
    let logits = generator.forward(&input).unwrap();
    compare_logits(
        "conditional",
        &logits,
        0,
        input.target_frames,
        entry.find("cond").unwrap(),
    );
    compare_logits(
        "unconditional",
        &logits,
        input.target_frames,
        input.target_frames,
        entry.find("uncond").unwrap(),
    );
    let (prediction, confidence) = generator.score(&input, 2.0).unwrap();
    let want_prediction = flatten(entry.find("pred").unwrap());
    let want_confidence = flatten(entry.find("confidence").unwrap());
    let agree = prediction
        .iter()
        .zip(&want_prediction)
        .filter(|(a, b)| **a == **b as i32)
        .count();
    let confidence_error = confidence
        .iter()
        .zip(&want_confidence)
        .map(|(a, b)| (f64::from(*a) - b).abs())
        .fold(0.0, f64::max);
    println!(
        "  scoring: prediction {agree}/{}, max |d confidence| {confidence_error}",
        prediction.len()
    );
    assert!(agree * 100 >= prediction.len() * 95, "scoring agreement");
}

#[test]
fn greedy_generation_unmasks_every_code() {
    let (mut generator, golden) = load();
    let decoding = DecodingConfig {
        position_temperature: 0.0,
        ..DecodingConfig::default()
    };
    for entry in golden.find("generate").unwrap().items() {
        let input = input_from(entry);
        let start = Instant::now();
        let result = generator.generate(&input, &decoding, 0, &|| false).unwrap();
        let seconds = start.elapsed().as_secs_f64();
        let want_codes = flatten(entry.find("tokens").unwrap());
        let want_steps = flatten(entry.find("unmasked_at").unwrap());
        let mut agree = 0;
        let mut first_divergence = i32::MAX;
        for i in 0..want_codes.len() {
            let (code, step) = (want_codes[i] as i32, want_steps[i] as i32);
            if result.codes[i] == code {
                agree += 1;
            }
            if result.codes[i] != code || result.unmasked_at[i] != step {
                first_divergence = first_divergence.min(result.unmasked_at[i].min(step));
            }
        }
        let divergence = if first_divergence == i32::MAX {
            "none".to_string()
        } else {
            first_divergence.to_string()
        };
        println!(
            "  generate {:?}: codes {agree}/{}, first divergent step {divergence}, {seconds:.2} s",
            entry.find("text").unwrap().str(),
            want_codes.len()
        );
        assert!(
            result.codes.iter().all(|&code| (0..1024).contains(&code)),
            "every code unmasked"
        );
    }
}
