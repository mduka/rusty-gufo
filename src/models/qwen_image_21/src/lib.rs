//! Qwen-Image-2.1 on gfx1151: a port of the C++ engine in this directory
//! (`model.hpp`) with the same numerical contract, so a request replays the
//! C++ image bit for bit (tests/parity.rs).
//!
//! - [`weights`], [`tokenizer`]: the Diffusers directory and its tokenizer;
//! - [`image`]: decoding, PNG encoding, resizing, validation, the schedule;
//! - [`random`]: the request's noise (`std::mt19937_64` and libstdc++'s
//!   normal distribution);
//! - [`runtime`], [`pipeline`]: the device operators and the generation
//!   ([`Model`]).

#![forbid(unsafe_code)]

mod host;
pub mod image;
pub mod pipeline;
pub mod random;
pub mod runtime;
pub mod tokenizer;
pub mod weights;

pub use image::{Image, Request};
pub use pipeline::{Generation, Metrics, Model};
pub use runtime::Observer;
