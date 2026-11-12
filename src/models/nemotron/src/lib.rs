//! NVIDIA Nemotron-H (Nemotron 3.5 Bolt 74B-A9B, `gguf:nemotron_h_moe`) on
//! gfx1151: the host code of the C++ Nemotron extension for Gufo
//! ([gufo-extensions/nemotron](https://github.com/gufo-extensions/nemotron),
//! `62f85dc`) in Rust, calling the same kernels in the same order with the
//! same rounding boundaries; comments that cite a C++ file mean that tree.

#![forbid(unsafe_code)]

pub mod chat_template;
pub mod config;
pub mod cpu_ops;
pub mod device_model;
pub mod drafter;
pub mod engine;
pub mod iq2xxs;
pub mod logit_head;
pub mod nvfp4;
pub mod probe;
pub mod reference;
pub mod runner;
pub mod speculative;
pub mod tokenizer;
pub mod vision_encoder;
pub mod vision_gemm;
pub mod vision_image;
pub mod vision_prompt;
pub mod weights;
