//! Nemotron 3.5 ASR Streaming 0.6B (NVIDIA): cache-aware FastConformer
//! encoder, language prompt and RNN-T transducer, from the Hugging Face
//! directory. A port of the Nemotron ASR Gufo extension
//! (gufo-extensions/nemotron_asr c256b69, version 0.1.2); its HIP kernels
//! are used verbatim.

#![forbid(unsafe_code)]

pub mod config;
pub mod features;
pub mod language;
pub mod model;
pub mod probe;
pub mod safetensors;
pub mod tokenizer;
