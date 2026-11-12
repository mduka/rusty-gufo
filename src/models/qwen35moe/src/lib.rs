//! The `qwen35moe` architecture (Qwen3.6 35B-A3B) on one gfx1151 device:
//! hybrid Gated DeltaNet and gated attention layers with routed experts and a
//! gated shared expert, native MTP and the DFlash2 draft.
//!
//! A port of the qwen35moe runner of the Gufo extension
//! slimsami/strycast-qwen35moe-gufo-extension: weights, device state and the
//! forward pass drive its HIP kernels (`gufo-qwen35moe-kernels`) in the order
//! and with the arguments of its C++ runner, so logits and token IDs match it
//! bit for bit. The chat template and the vision encoder come from `gufo-qwen`.

#![forbid(unsafe_code)]

mod arena;
mod batch;
pub mod config;
mod decode;
pub mod dflash;
pub mod engine;
mod moe;
pub mod mtp;
mod pack;
pub mod persistent;
mod prefill;
pub mod routes;
pub mod runner;
pub mod speculative;
pub mod verifier;
mod verify;
pub mod weights;

pub use arena::{MAX_PROMPT_BATCH, MAX_TARGET_TAPS, MAX_VERIFY_ROWS};
pub use config::ModelConfig;
pub use engine::{Engine, Model, Session, Snapshot, Verification, VerifyRequest};
pub use speculative::{Speculative, Step};
pub use weights::Weights;
