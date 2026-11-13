//! The C++ extension's `features_test.cpp`: the log-mel front end against
//! `tools/reference/golden.py`, each fixture clip pushed in pieces, then
//! ended as a centered pass over the whole clip ends (zeros after the
//! pre-emphasized signal, samples / 160 frames).

#![allow(clippy::unwrap_used)]

mod support;

use gufo_nemotron_asr::config::FeatureConfig;
use gufo_nemotron_asr::features::FeatureStream;
use gufo_nemotron_asr::safetensors::SafetensorsFile;

const BINS: usize = 128;

/// The frames of `audio` pushed in pieces of 1 to `split` samples (all at
/// once for 0, 1 to 7 for 1), then flushed.
fn features(audio: &[f32], split: usize) -> (Vec<f32>, FeatureStream) {
    let mut stream = FeatureStream::new(&FeatureConfig::default(), BINS).unwrap();
    let mut frames = Vec::new();
    let mut splits = support::Splits::new(split as u32 + 3);
    let mut at = 0;
    while at < audio.len() {
        let n = if split == 0 {
            audio.len() - at
        } else {
            (audio.len() - at).min(splits.next(if split == 1 { 7 } else { split }))
        };
        stream.push(&audio[at..at + n], &mut frames);
        at += n;
    }
    let total = stream.frames_for(audio.len());
    stream.flush(total, &mut frames);
    assert_eq!(stream.frames(), total, "frame count");
    (frames, stream)
}

#[test]
fn clips_against_the_reference() {
    let golden = SafetensorsFile::open(&support::fixture("golden.safetensors")).unwrap();
    for clip in ["english", "german"] {
        let (audio, rate) = support::read_wav(&format!("{clip}.wav"));
        assert_eq!(rate, 16000, "{clip} is 16 kHz");
        let expected = support::floats(&golden, &format!("{clip}.features"));
        let mut whole = None;
        for split in [0, 1, 997] {
            let (frames, _) = features(&audio, split);
            support::assert_close(
                &format!("{clip} features, split {split}"),
                &frames,
                &expected,
                1e-4,
            );
            // Any split gives the same bits.
            match &whole {
                None => whole = Some(frames),
                Some(whole) => assert!(
                    whole
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(frames.iter().map(|v| v.to_bits())),
                    "{clip}: split {split} changes the frames"
                ),
            }
        }
    }
}

/// With `NEMOTRON_ASR_FEATURES_DUMP` set to a directory, writes the frames
/// and the filters as raw float32 for a byte comparison with the C++
/// extension's (`<clip>.features.f32`, `filters.f32`).
#[test]
fn dump_for_the_cxx_comparison() {
    let Some(directory) = std::env::var_os("NEMOTRON_ASR_FEATURES_DUMP") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    std::fs::create_dir_all(&directory).unwrap();
    let raw = |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    for clip in ["english", "german"] {
        let (audio, _) = support::read_wav(&format!("{clip}.wav"));
        std::fs::write(directory.join(format!("{clip}.samples.f32")), raw(&audio)).unwrap();
        let (frames, stream) = features(&audio, 997);
        std::fs::write(directory.join(format!("{clip}.features.f32")), raw(&frames)).unwrap();
        std::fs::write(directory.join("filters.f32"), raw(stream.filters())).unwrap();
    }
}
