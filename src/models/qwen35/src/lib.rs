//! The `qwen35` architecture (Qwen3.8 27B and the dense Qwen3.5 checkpoints)
//! on one gfx1151 device: hybrid Gated DeltaNet and gated attention layers
//! with a dense feed-forward, native MTP and the DFlash2 draft.
//!
//! Weights, device state and the forward pass drive the HIP kernels of
//! `gufo-qwen-kernels` in the order and with the arguments of the C++ engine
//! (`src/models/qwen/hip`), so logits and token IDs match it bit for bit. The
//! chat template and the vision encoder come from `gufo-qwen`.

#![forbid(unsafe_code)]

mod arena;
mod batch;
pub mod config;
mod decode;
pub mod dflash;
pub mod engine;
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
