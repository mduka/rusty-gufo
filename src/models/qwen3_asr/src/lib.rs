//! Qwen3-ASR-1.7B: WAV in, transcript out, on one gfx1151 device.
//!
//! The CPU frontend (WAV decoding, resampling, log-mel features), tokenizer
//! and prompt reproduce the C++ engine bit for bit; the audio encoder and text
//! decoder drive the HIP kernels of `gufo-qwen3-asr-kernels` in the same order
//! and with the same hipBLASLt policy, so the generated token IDs match the
//! C++ path exactly.

#![forbid(unsafe_code)]

pub mod audio;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod gemm;
pub mod prompt;
pub mod safetensors;
pub mod tokenizer;
pub mod transcriber;
pub mod weights;

pub use transcriber::{Audio, Request, Timings, Transcriber, Transcription};
