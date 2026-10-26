//! DeepSeek V4 Flash text generation on gfx1151 (docs/RUST.md): the port of the C++ DS4 engine's host code over
//! the kernels of `gufo-deepseek-v4-flash-kernels`.

#![forbid(unsafe_code)]

pub mod chat_template;
pub mod config;
pub mod dspark_model;
pub mod dspark_policy;
pub mod dspark_sampler;
pub mod engine;
pub mod gguf;
pub mod graph;
pub mod runner;
pub mod session;
pub mod shape;
pub mod tokenizer;
pub mod weights;
