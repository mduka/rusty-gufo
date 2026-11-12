//! The C++ extension's `codec_test.cpp`: the audio tokenizer's decoder and
//! encoder against the reference model.
//! cargo test --profile gpu-test -p gufo-omnivoice --features gpu,models --test codec

#![cfg(all(feature = "gpu", feature = "models"))]
#![allow(clippy::unwrap_used)]

mod config_samples;

use std::time::Instant;

use gufo_omnivoice::codec::Codec;
use gufo_omnivoice::config::load_config;

/// `tools/reference/generator_golden.py`'s encoder test signal.
fn signal(seconds: f64, seed: i64) -> Vec<f32> {
    let pi = std::f64::consts::PI;
    let mut state = seed;
    (0..(seconds * 24000.0) as usize)
        .map(|n| {
            state = (state * 1_103_515_245 + 12345) & 0x7FFF_FFFF;
            let x = n as f64;
            let envelope = 0.5 + 0.5 * (2.0 * pi * 3.0 * x / 24000.0).sin();
            let tone = (2.0 * pi * 180.0 * x / 24000.0).sin()
                + 0.4 * (2.0 * pi * 720.0 * x / 24000.0).sin();
            let noise = ((state >> 8) % 2001 - 1000) as f64 / 1000.0;
            (0.2 * envelope * tone + 0.02 * noise) as f32
        })
        .collect()
}

#[test]
fn decoder_and_encoder_match_the_reference() {
    let root = config_samples::model_root();
    let config = load_config(&root).unwrap();
    let mut codec = Codec::load(&config, &root).unwrap();
    let golden = config_samples::golden("generator_golden.json");

    // Decode.
    let entry = golden.find("decode").unwrap();
    let generation =
        &golden.find("generate").unwrap().items()[entry.find("tokens_from").unwrap().as_size(0)];
    let codes: Vec<i32> = generation
        .find("tokens")
        .unwrap()
        .items()
        .iter()
        .flat_map(|row| row.items().iter().map(|code| code.as_double(0.0) as i32))
        .collect();
    let frames = codes.len() / 8;
    let start = Instant::now();
    let audio = codec.decode(&codes, frames).unwrap();
    let seconds = start.elapsed().as_secs_f64();
    assert_eq!(
        audio.len(),
        entry.find("length").unwrap().as_size(0),
        "decoded length"
    );
    let squares: f64 = audio.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    let rms = (squares / audio.len() as f64).sqrt();
    let stride = golden.find("stride").unwrap().as_size(0);
    let worst = entry
        .find("samples")
        .unwrap()
        .items()
        .iter()
        .enumerate()
        .take_while(|(i, _)| i * stride < audio.len())
        .map(|(i, value)| (f64::from(audio[i * stride]) - value.as_double(0.0)).abs())
        .fold(0.0, f64::max);
    let expected_rms = entry.find("rms").unwrap().as_double(0.0);
    println!(
        "  decode {frames} frames: rms {rms} (reference {expected_rms}), max |d sample| {worst}, {seconds:.3} s"
    );
    assert!((rms - expected_rms).abs() < 1e-4, "decoded RMS");
    assert!(worst < 1e-3, "decoded samples");

    // Encode.
    let entry = golden.find("encode").unwrap();
    let source = entry.find("signal").unwrap();
    let samples = signal(
        source.find("seconds").unwrap().as_double(0.0),
        source.find("seed").unwrap().as_size(0) as i64,
    );
    let start = Instant::now();
    let (codes, frames) = codec.encode(&samples).unwrap();
    let seconds = start.elapsed().as_secs_f64();
    let expected = entry.find("codes").unwrap().items();
    assert_eq!(frames, expected[0].items().len(), "encoded frame count");
    let per_codebook: Vec<usize> = expected
        .iter()
        .enumerate()
        .map(|(c, row)| {
            row.items()
                .iter()
                .take(frames)
                .enumerate()
                .filter(|(f, code)| codes[c * frames + f] == code.as_double(0.0) as i32)
                .count()
        })
        .collect();
    println!(
        "  encode {frames} frames in {seconds:.3} s: {per_codebook:?} of {frames} per codebook"
    );
    let total: usize = per_codebook.iter().sum();
    assert!(total * 100 >= frames * expected.len() * 90, "encoded codes");
}
