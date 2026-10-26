//! Qwen3.8-Flash-Next (`qwen4exp`) on gfx1151, a port of the C++ package in
//! this directory: four hyper-connection residual streams, Gated DeltaNet
//! layers with a sparse-attention layer every fourth, 512 routed experts
//! plus a shared one, the PLE n-gram embedding read from disk, and the MTP
//! draft block of a sidecar GGUF.

#![forbid(unsafe_code)]

pub mod batch;
mod borrowed;
pub mod checkpoint;
pub mod config;
pub mod device;
pub mod draft_vocab;
pub mod engine;
pub mod executor;
mod forward;
pub mod lookup;
pub mod mtp;
pub mod ngram;
pub mod policy;
pub mod rope_scaling;
mod routes;
pub mod runner;
pub mod snapshot;
pub mod state;
pub mod weights;
