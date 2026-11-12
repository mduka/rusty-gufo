//! The C++ extension's `audio_test.cpp`: silence removal, fades, cross
//! fades, resampling, reference preparation and post-processing against the
//! reference pipeline's fixture.

#![allow(clippy::unwrap_used)]

mod config_samples;

use gufo_core::json::Value;
use gufo_omnivoice::audio::{
    cross_fade_chunks, fade_and_pad, post_process, prepare_reference, remove_silence, resample,
};

/// `tools/reference/audio_golden.py`'s integer signal generator.
fn generate(spec: &Value, rate: i64, seed: i64) -> Vec<f32> {
    let mut state = seed;
    let mut samples = Vec::new();
    for segment in spec.items() {
        let parts = segment.items();
        let ms = parts[0].as_size(0) as i64;
        let kind = parts[1].str();
        let amplitude = parts[2].as_size(0) as i64;
        let noise_amplitude = amplitude.min(64);
        for n in 0..ms * rate / 1000 {
            state = (state * 1_103_515_245 + 12345) & 0x7FFF_FFFF;
            let value = if kind == "tone" {
                const PERIOD: i64 = 55;
                let phase = n % PERIOD;
                let noise = (state >> 8) % (2 * noise_amplitude + 1) - noise_amplitude;
                (4 * amplitude * (2 * phase - PERIOD).abs()) / (2 * PERIOD) - amplitude + noise
            } else {
                (state >> 8) % (2 * amplitude + 1) - amplitude
            };
            samples.push(value.clamp(-32768, 32767) as f32 / 32768.0);
        }
    }
    samples
}

fn generate_signal(signal: &Value) -> Vec<f32> {
    let mut samples = generate(
        signal.find("spec").unwrap(),
        signal.find("rate").unwrap().as_size(0) as i64,
        signal.find("seed").unwrap().as_size(0) as i64,
    );
    if let Some(extra) = signal.find("extra") {
        samples.resize(samples.len() + extra.as_size(0), 0.25);
    }
    samples
}

/// Exact PCM16 output: length and sums.
fn compare_pcm(what: &str, got: &[f32], expected: &Value, failures: &mut Vec<String>) {
    let length = expected.find("length").unwrap().as_size(0);
    if got.len() != length {
        failures.push(format!("{what}: length {}, expected {length}", got.len()));
        return;
    }
    let (mut sum, mut squares) = (0i64, 0i64);
    for &sample in got {
        let value = (f64::from(sample) * 32768.0).round_ties_even() as i64;
        sum += value;
        squares += value * value;
    }
    let expected_sum = expected.find("sum").unwrap().as_double(0.0) as i64;
    let expected_squares = expected.find("sum_squares").unwrap().as_double(0.0) as i64;
    if sum != expected_sum || squares != expected_squares {
        failures.push(format!(
            "{what}: checksum {sum}/{squares}, expected {expected_sum}/{expected_squares}"
        ));
    }
}

/// Float output: length and strided samples within `tolerance`.
fn compare_samples(
    what: &str,
    got: &[f32],
    expected: &Value,
    stride: usize,
    tolerance: f64,
    failures: &mut Vec<String>,
) {
    let length = expected.find("length").unwrap().as_size(0);
    if got.len() != length {
        failures.push(format!("{what}: length {}, expected {length}", got.len()));
        return;
    }
    let worst = expected
        .find("samples")
        .unwrap()
        .items()
        .iter()
        .enumerate()
        .map(|(index, sample)| (f64::from(got[index * stride]) - sample.as_double(0.0)).abs())
        .fold(0.0f64, f64::max);
    if worst > tolerance {
        failures.push(format!("{what}: max difference {worst}"));
    }
}

#[test]
fn reference_pipeline_cases() {
    let golden = config_samples::golden("audio_golden.json");
    let stride = golden.find("stride").unwrap().as_size(0);
    let mut failures = Vec::new();
    let mut checks = 0;
    for (index, entry) in golden.find("silence").unwrap().items().iter().enumerate() {
        let input = generate_signal(entry.find("signal").unwrap());
        let params = entry.find("params").unwrap().items();
        let got = remove_silence(
            &input,
            24000,
            params[0].as_size(0) as i32,
            params[1].as_size(0) as i32,
            params[2].as_size(0) as i32,
        );
        compare_pcm(&format!("silence {index}"), &got, entry, &mut failures);
        checks += 1;
    }
    for (index, entry) in golden.find("fade").unwrap().items().iter().enumerate() {
        let got = fade_and_pad(
            &generate_signal(entry.find("signal").unwrap()),
            24000,
            0.1,
            0.1,
        );
        compare_samples(
            &format!("fade {index}"),
            &got,
            entry,
            stride,
            1e-7,
            &mut failures,
        );
        checks += 1;
    }
    for (index, entry) in golden
        .find("postprocess")
        .unwrap()
        .items()
        .iter()
        .enumerate()
    {
        let rms = entry.find("rms").unwrap();
        let level = (!rms.is_null()).then(|| rms.as_double(0.0));
        let got = post_process(
            &generate_signal(entry.find("signal").unwrap()),
            24000,
            level,
        );
        compare_samples(
            &format!("postprocess {index}"),
            &got,
            entry,
            stride,
            1e-6,
            &mut failures,
        );
        checks += 1;
    }
    for entry in golden.find("crossfade").unwrap().items() {
        let chunks: Vec<Vec<f32>> = entry
            .find("signals")
            .unwrap()
            .items()
            .iter()
            .map(generate_signal)
            .collect();
        let got = cross_fade_chunks(&chunks, 24000, 0.3);
        compare_samples("crossfade", &got, entry, stride, 1e-7, &mut failures);
        checks += 1;
    }
    for entry in golden.find("resample").unwrap().items() {
        let signal = entry.find("signal").unwrap();
        let source = signal.find("rate").unwrap().as_size(0) as u32;
        let target = entry.find("target").unwrap().as_size(0) as u32;
        let got = resample(&generate_signal(signal), source, target);
        compare_samples(
            &format!("resample {source} -> {target}"),
            &got,
            entry,
            stride,
            2e-6,
            &mut failures,
        );
        checks += 1;
    }
    for (index, entry) in golden.find("reference").unwrap().items().iter().enumerate() {
        checks += 1;
        let what = format!("reference {index}");
        let channels = entry.find("channels").unwrap().as_size(0) as u32;
        let rate = entry.find("rate").unwrap().as_size(0) as i64;
        let seeds = entry.find("seeds").unwrap().items();
        let spec = entry.find("spec").unwrap();
        let left = generate(spec, rate, seeds[0].as_size(0) as i64);
        let right = generate(spec, rate, seeds[1].as_size(0) as i64);
        let mut interleaved = Vec::new();
        for (frame, &sample) in left.iter().enumerate() {
            interleaved.push(sample);
            if channels == 2 {
                interleaved.push(right[frame]);
            }
        }
        let reference = match prepare_reference(&interleaved, channels, rate as u32, 24000, 960) {
            Ok(reference) => reference,
            Err(error) => {
                failures.push(format!("{what}: {error}"));
                continue;
            }
        };
        if (reference.rms - entry.find("rms").unwrap().as_double(0.0)).abs() > 1e-6 {
            failures.push(format!("{what}: rms {}", reference.rms));
        }
        // Resampling rounds differently in float32; PCM16 values may differ
        // by one step where they straddle a rounding boundary.
        let length = entry.find("length").unwrap().as_size(0);
        if reference.samples.len() != length {
            failures.push(format!("{what}: length {}", reference.samples.len()));
            continue;
        }
        let worst = entry
            .find("samples")
            .unwrap()
            .items()
            .iter()
            .enumerate()
            .map(|(sample, expected)| {
                (f64::from(reference.samples[sample * stride]) * 32768.0 - expected.as_double(0.0))
                    .abs()
            })
            .fold(0.0f64, f64::max);
        if worst > 1.0 {
            failures.push(format!("{what}: max PCM16 difference {worst}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {checks} audio cases fail:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
