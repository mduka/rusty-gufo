//! The operations of DeepSeek V4 Flash, routed to the generated launchers
//! of `launch.toml` as the C++ launchers chose (docs/RUST.md): inputs as [`Input`](crate::Input), outputs as exclusive
//! [`Output`](crate::Output)s, device indices as `DeviceIndices` of their
//! contract, weights as [`Weight`]s the backend resolves inside its resident
//! ranges. The generated wrappers check every extent and index contract
//! before a launch; a refusal is an [`Error`](crate::Error) and enqueues
//! nothing.

// The wrappers keep the launchers' argument lists.
#![allow(clippy::too_many_arguments)]

mod attention;
mod compressor;
mod dspark;
mod hc;
mod indexer;
mod matmul;
mod moe;
mod norm;
mod routed;

pub use compressor::CompressorWeights;
pub use dspark::CANDIDATES_BYTES;
pub(crate) use matmul::Projections;
pub use moe::{ExpertSelection, HashTable, N_EXPERT, N_EXPERT_USED, RoutedExperts, RoutedMoe};

use gufo_hip::{DeviceCopy, DeviceSlice, DeviceSliceMut, LaunchError};

use crate::backend::{Error, Model, Result};
use crate::tensor::Tensor;

/// A weight: the offset of its first byte in a registered model mapping.
/// The runtime resolves it only inside a resident range of that mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Weight {
    pub model: Model,
    pub offset: u64,
}

impl Weight {
    pub const fn target(offset: u64) -> Self {
        Self {
            model: Model::Target,
            offset,
        }
    }

    pub const fn support(offset: u64) -> Self {
        Self {
            model: Model::Support,
            offset,
        }
    }
}

/// The product of `factors`, refused on overflow.
pub(crate) fn product(op: &'static str, factors: &[u64]) -> Result<u64> {
    factors
        .iter()
        .try_fold(1u64, |acc, &f| acc.checked_mul(f))
        .ok_or(Error { op })
}

/// `tensor` holds at least the product of `factors` bytes.
pub(crate) fn need(op: &'static str, tensor: &Tensor, factors: &[u64]) -> Result<()> {
    if tensor.bytes() >= product(op, factors)? {
        Ok(())
    } else {
        Err(Error { op })
    }
}

/// A generated launch's result as the backend reports failures.
pub(crate) fn launched<T>(
    op: &'static str,
    result: std::result::Result<T, LaunchError>,
) -> Result<T> {
    result.map_err(|error| {
        eprintln!("ds4: {op}: {error}");
        Error { op }
    })
}

/// Input bytes as `U` values; refused unless they hold whole, aligned
/// values.
pub(crate) fn typed<'a, U: DeviceCopy>(
    op: &'static str,
    bytes: DeviceSlice<'a, u8>,
) -> Result<DeviceSlice<'a, U>> {
    bytes.cast::<U>().ok_or(Error { op })
}

/// Output bytes as `U` values; see [`typed`].
pub(crate) fn typed_mut<'a, U: DeviceCopy>(
    op: &'static str,
    bytes: DeviceSliceMut<'a, u8>,
) -> Result<DeviceSliceMut<'a, U>> {
    bytes.cast::<U>().ok_or(Error { op })
}

pub(crate) fn ensure(op: &'static str, condition: bool) -> Result<()> {
    if condition { Ok(()) } else { Err(Error { op }) }
}

pub(crate) const F32: u64 = 4;
pub(crate) const F16: u64 = 2;
pub(crate) const I32: u64 = 4;

/// The RoPE parameters the rotating launchers take, in their order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rope {
    pub n_rot: u32,
    pub pos0: u32,
    pub n_ctx_orig: u32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}
