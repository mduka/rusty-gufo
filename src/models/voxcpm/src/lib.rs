//! VoxCPM2 (OpenBMB) text-to-speech on gfx1151: the host code of the C++
//! VoxCPM extension for Gufo
//! ([gufo-extensions/voxcpm](https://github.com/gufo-extensions/voxcpm),
//! `0941cd1`) in Rust, calling the same kernels in the same order with the
//! same rounding boundaries; comments that cite a C++ file mean that tree.

#![forbid(unsafe_code)]

pub mod config;
pub mod gemm;
pub mod generator;
pub mod matrix;
pub mod model_files;
pub mod probe;
pub mod safetensors;
pub mod synthesis;
pub mod text;
pub mod tokenizer;
pub mod torch_checkpoint;
pub mod transformer;
pub mod vae;
pub mod weights;
