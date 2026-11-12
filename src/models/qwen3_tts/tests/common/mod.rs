//! Checkpoint and fixture locations shared by the model tests. The tests fail
//! (rather than skip) when these are missing: a missing model is not a pass.

#![allow(dead_code, clippy::unwrap_used)]

use std::path::PathBuf;

use gufo_qwen3_tts::audio::AudioBuffer;
use gufo_qwen3_tts::config::ModelVariant;

fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap())
}

/// `QWEN3_TTS_<VARIANT>_ROOT`, else the pinned snapshot in the Hugging Face
/// cache.
pub fn model_root(variant: ModelVariant) -> PathBuf {
    let (name, snapshot, variable) = match variant {
        ModelVariant::CustomVoice => (
            "CustomVoice",
            "0c0e3051f131929182e2c023b9537f8b1c68adfe",
            "QWEN3_TTS_CUSTOMVOICE_ROOT",
        ),
        ModelVariant::VoiceDesign => (
            "VoiceDesign",
            "5ecdb67327fd37bb2e042aab12ff7391903235d3",
            "QWEN3_TTS_VOICEDESIGN_ROOT",
        ),
        _ => (
            "Base",
            "fd4b254389122332181a7c3db7f27e918eec64e3",
            "QWEN3_TTS_BASE_ROOT",
        ),
    };
    std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            home().join(format!(
            ".cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-{name}/snapshots/{snapshot}"
        ))
        })
}

/// The Qwen3-ASR English example (48 kHz, 24-bit mono) and its transcript,
/// as a Base voice-clone reference.
pub fn reference() -> (AudioBuffer, String) {
    let path = std::env::var_os("QWEN3_TTS_REFERENCE_WAV")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".cache/gufo/fixtures/qwen3-asr-en.wav"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {}: {e}", path.display()));
    let data = bytes
        .windows(4)
        .position(|w| w == b"data")
        .expect("data chunk");
    let size = u32::from_le_bytes(bytes[data + 4..data + 8].try_into().unwrap()) as usize;
    let samples = bytes[data + 8..data + 8 + size]
        .as_chunks::<3>()
        .0
        .iter()
        .map(|s| {
            let value = i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8;
            value as f32 / 8_388_608.0
        })
        .collect();
    let text = "Uh huh. Oh yeah, yeah. He wasn't even that big when I started listening to \
                him, but and his solo music didn't do overly well, but he did very well when \
                he started writing for other people.";
    (
        AudioBuffer {
            sample_rate: 48000,
            channels: 1,
            samples,
        },
        text.to_owned(),
    )
}
