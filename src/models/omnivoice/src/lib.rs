//! OmniVoice (k2-fsa) text-to-speech on gfx1151: the host code of the C++
//! OmniVoice extension for Gufo
//! ([gufo-extensions/omnivoice](https://github.com/gufo-extensions/omnivoice),
//! `4bf376c`) in Rust, calling the same kernels in the same order with the
//! same rounding boundaries; comments that cite a C++ file mean that tree.

#![forbid(unsafe_code)]

pub mod audio;
pub mod codec;
mod codec_encoder;
pub mod config;
pub mod decoding;
pub mod gemm;
pub mod generator;
pub mod languages;
pub mod probe;
pub mod prompt;
pub mod safetensors;
pub mod synthesis;
pub mod text;
pub mod tokenizer;
pub mod unicode;
