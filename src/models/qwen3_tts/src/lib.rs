//! Qwen3-TTS 12Hz 1.7B (CustomVoice, VoiceDesign and Base) on gfx1151: the
//! host code of `src/models/qwen3_tts` in Rust, calling the same kernels in
//! the same order with the same rounding boundaries (docs/RUST.md).

#![forbid(unsafe_code)]

pub mod audio;
pub mod bf16;
pub mod checkpoint;
pub mod config;
pub mod decoder;
pub mod encoders;
pub mod features;
pub mod prompt;
pub mod sampling;
pub mod synthesis;
pub mod talker;
pub mod tokenizer;
pub mod weights;
