//! Qwen3-TTS kernels. Everything here is generated from `launch.toml`; the
//! generated FFI calls are the crate's only `unsafe`.

#![deny(unsafe_code)]

#[allow(unsafe_code)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/launch.rs"));
}

pub use generated::*;
