//! DeepSeek V4 Flash kernels and their routing.
//!
//! The kernels and their launchers are one HIP translation unit
//! (`kernels/rocm/backend.hip.cpp`); the launchers of `launch.toml` are
//! called through the generated wrappers, which check every extent and
//! index contract before a launch. [`Backend`] owns the device state the
//! C++ runtime kept in globals (the stream, the BLAS handles, the resident
//! weights, the scratch and the F16 weight cache), and the op modules route
//! each operation to its launchers as the C++ launchers chose (see
//! docs/RUST.md).

#![deny(unsafe_code)]

mod backend;
mod ops;
mod tensor;

#[allow(unsafe_code)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/launch.rs"));
}

/// The extent probes of the generated launchers (`tests/extents.rs`).
pub use generated::probes;

pub use backend::{Backend, BatchMode, Error, Model, Result, SnapshotReader};
pub use ops::{
    CANDIDATES_BYTES, CompressorWeights, ExpertSelection, HashTable, N_EXPERT, N_EXPERT_USED, Rope,
    RoutedExperts, RoutedMoe, Weight,
};
pub use tensor::{DeviceTensor, Input, Output, Tensor, TensorView, TensorViewMut, VIEW_ALIGNMENT};
