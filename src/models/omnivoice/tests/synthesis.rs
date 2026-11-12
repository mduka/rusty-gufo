//! The C++ extension's `synthesis_test.cpp`: synthesis end to end, seeds,
//! voice design, cloning, chunking and cancellation.
//! cargo test --profile gpu-test -p gufo-omnivoice --features gpu,models --test synthesis

#![cfg(all(feature = "gpu", feature = "models"))]
#![allow(clippy::unwrap_used)]

mod config_samples;

use std::time::Instant;

use gufo_omnivoice::synthesis::{SynthesisOptions, Synthesizer};

fn seconds(audio: &[f32], synthesizer: &Synthesizer) -> f64 {
    audio.len() as f64 / f64::from(synthesizer.sample_rate())
}

fn rms(audio: &[f32]) -> f64 {
    if audio.is_empty() {
        return 0.0;
    }
    let sum: f64 = audio.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / audio.len() as f64).sqrt()
}

fn run(
    synthesizer: &mut Synthesizer,
    options: &SynthesisOptions<'_>,
    label: &str,
) -> (Vec<f32>, Vec<i32>) {
    let start = Instant::now();
    let result = synthesizer
        .synthesize(options, &|| false)
        .unwrap_or_else(|error| panic!("{label}: {error}"));
    println!(
        "  {label}: {:.2} s of audio in {:.2} s, rms {}",
        seconds(&result.audio, synthesizer),
        start.elapsed().as_secs_f64(),
        rms(&result.audio)
    );
    (result.audio, result.codes)
}

#[test]
fn synthesis_end_to_end() {
    let root = config_samples::model_root();
    let mut synthesizer = Synthesizer::load(&root).unwrap();
    let mut options = SynthesisOptions {
        text: "The little brown dog ran across the garden.".into(),
        language: "English".into(),
        seed: 42,
        ..SynthesisOptions::default()
    };
    let (first, codes) = run(&mut synthesizer, &options, "seed 42");
    let (second, _) = run(&mut synthesizer, &options, "seed 42 again");
    assert_eq!(first, second, "the same seed reproduces the audio");
    let duration = seconds(&first, &synthesizer);
    assert!(duration > 1.0 && duration < 6.0, "a plausible duration");
    assert!(rms(&first) > 0.01, "audible output");
    assert!(
        codes.len() % 8 == 0 && !codes.is_empty(),
        "frame-major codes"
    );
    options.seed = 43;
    let (third, _) = run(&mut synthesizer, &options, "seed 43");
    assert_ne!(third, first, "another seed changes the audio");

    // Voice design and its validation.
    options.instruct = "female, low pitch, british accent".into();
    run(&mut synthesizer, &options, "voice design");
    options.instruct = "very loud".into();
    let error = synthesizer.synthesize(&options, &|| false).unwrap_err();
    assert!(
        error.contains("unsupported voice design items"),
        "unknown voice-design items are rejected: {error}"
    );
    options.instruct.clear();

    // Cloning the first result.
    let voice = synthesizer
        .clone_voice(
            &first,
            1,
            synthesizer.sample_rate(),
            "The little brown dog ran across the garden",
        )
        .unwrap();
    assert!(
        voice.frames > 20 && voice.text.ends_with('.'),
        "an encoded reference with a punctuated transcript"
    );
    let cloned = SynthesisOptions {
        text: "Nice to meet you.".into(),
        reference: Some(&voice),
        ..options.clone()
    };
    run(&mut synthesizer, &cloned, "cloned voice");

    // A text estimated above 30 s is generated in chunks.
    options.text = "The quick brown fox jumps over the lazy dog near the quiet river bank, \
                    and then it runs back home before the evening rain. "
        .repeat(9);
    let (long_audio, _) = run(&mut synthesizer, &options, "chunked story");
    assert!(
        seconds(&long_audio, &synthesizer) > 30.0,
        "the long text is spoken in full"
    );

    // Cancellation stops the generation.
    options.text = "This request is cancelled before it starts.".into();
    let error = synthesizer.synthesize(&options, &|| true).unwrap_err();
    assert_eq!(error, "cancelled", "cancellation");
}
