//! One MiniCPM4 stack of VoxCPM2 (`transformer.hpp`, `transformer.cpp`):
//! the KV cache, the LongRoPE table, the fused q/k/v and gate/up layers and
//! the projections around them. The residual stream, norms and attention
//! run in float32; GEMM operands are bfloat16.
//!
//! Ported from OpenBMB/VoxCPM's MiniCPM4 stack (src/voxcpm/modules/minicpm4;
//! Apache-2.0, Copyright OpenBMB), with fused q/k/v and gate/up
//! projections.

use gufo_hip::{
    Bf16, DeviceBuffer, DeviceCopy, DeviceSlice, DeviceSliceMut, HipError, IndexSlice, LaunchError,
    Stream,
};
use gufo_voxcpm_kernels as k;

use crate::config::{RopeConfig, TransformerConfig};
use crate::gemm::Bf16Gemm;
use crate::weights::{DeviceWeights, Projection, StackWeights};

/// Keys and values of one stack: `[layer][kv_head][capacity][head_dim]`
/// each (`KvCache`).
pub struct KvCache {
    keys: DeviceBuffer<f32>,
    values: DeviceBuffer<f32>,
    capacity: usize,
    layer_stride: usize,
}

impl KvCache {
    pub fn new(
        stream: &Stream,
        config: &TransformerConfig,
        capacity: usize,
    ) -> Result<Self, HipError> {
        let layer_stride = config.kv_heads * capacity * config.head_dim;
        Ok(Self {
            keys: stream.alloc_persistent((config.layers * layer_stride).max(1))?,
            values: stream.alloc_persistent((config.layers * layer_stride).max(1))?,
            capacity,
            layer_stride,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn range(&self, layer: usize) -> std::ops::Range<usize> {
        layer * self.layer_stride..(layer + 1) * self.layer_stride
    }
}

/// Which keys each row attends to; see the `attention` launcher.
/// `positions` (rotary table rows) and `cache_rows` hold one entry per row
/// (`AttentionLayout`).
pub struct AttentionLayout<'a> {
    pub positions: IndexSlice<'a>,
    pub cache_rows: IndexSlice<'a>,
    pub rows_per_sequence: usize,
    pub keys_per_sequence: usize,
    pub past: usize,
    pub causal: bool,
}

/// Scratch shared by the stacks of one generator (`StackScratch`).
pub struct StackScratch {
    normed: DeviceBuffer<Bf16>,
    qkv: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    attention: DeviceBuffer<Bf16>,
    gate_up: DeviceBuffer<f32>,
    activation: DeviceBuffer<Bf16>,
}

impl StackScratch {
    pub fn new(stream: &Stream) -> Result<Self, HipError> {
        Ok(Self {
            normed: stream.alloc_persistent(1)?,
            qkv: stream.alloc_persistent(1)?,
            q: stream.alloc_persistent(1)?,
            attention: stream.alloc_persistent(1)?,
            gate_up: stream.alloc_persistent(1)?,
            activation: stream.alloc_persistent(1)?,
        })
    }
}

/// Reallocates `buffer` when `count` exceeds its capacity; contents are
/// lost (`DeviceBuffer::Reserve`). Caches and scratch are exact `hipMalloc`
/// allocations like the C++ `DeviceBuffer`s: from the stream-ordered pool,
/// the prefill's skinny products, whose every block re-reads the
/// activations, ran up to a quarter slower (6 ms per prompt).
pub(crate) fn reserve<T: DeviceCopy>(
    stream: &Stream,
    buffer: &mut DeviceBuffer<T>,
    count: usize,
) -> Result<(), HipError> {
    if count > buffer.len() {
        *buffer = stream.alloc_persistent(count.max(1))?;
    }
    Ok(())
}

/// The rotary table `[position][cos (head_dim / 2) | sin (head_dim / 2)]`
/// of MiniCPM's LongRoPE, computed in float32 as the reference does
/// (`RopeTable`): `inv_freq = 1 / base^(arange(0, dim, 2) / dim)`,
/// `freqs = outer(t, 1 / factors) * inv_freq`, scaled cos and sin.
pub fn rope_table(rope: &RopeConfig, positions: usize) -> Vec<f32> {
    let half = rope.dim / 2;
    let factors = rope.factors();
    let scaling = rope.scaling() as f32;
    let mut inverse = vec![0.0f32; half];
    let mut factor_inverse = vec![0.0f32; half];
    for i in 0..half {
        let exponent = (2 * i) as f32 / rope.dim as f32;
        inverse[i] = 1.0 / (rope.theta as f32).powf(exponent);
        factor_inverse[i] = 1.0 / factors[i] as f32;
    }
    let mut table = vec![0.0f32; positions * rope.dim];
    for position in 0..positions {
        for i in 0..half {
            let angle = position as f32 * factor_inverse[i] * inverse[i];
            table[position * rope.dim + i] = angle.cos() * scaling;
            table[position * rope.dim + half + i] = angle.sin() * scaling;
        }
    }
    table
}

/// `y[rows][out] = x[rows][in] W^T + b` with bfloat16 `x` (`Project`).
pub fn project(
    weights: &DeviceWeights,
    projection: &Projection,
    x: DeviceSlice<'_, Bf16>,
    mut y: DeviceSliceMut<'_, f32>,
    rows: usize,
    gemm: &mut Bf16Gemm,
    stream: &Stream,
) -> Result<(), LaunchError> {
    gemm.linear(
        stream,
        x,
        weights.matrix(projection.weight),
        y.reborrow(),
        rows,
        projection.input,
        projection.output,
        false,
    )?;
    if let Some(bias) = projection.bias {
        k::add_bias(stream, y, weights.vector(bias), rows, projection.output)?;
    }
    Ok(())
}

/// The rotary table a stack reads: the device table and its row count.
#[derive(Clone, Copy)]
pub struct Rope<'a> {
    pub table: DeviceSlice<'a, f32>,
    pub rows: usize,
}

/// Runs one MiniCPM4 stack over `rows` rows of the float32 residual
/// stream `hidden` (updated in place) and writes the final norm's output to
/// `output` (float32). `rope` is the device rotary table, `None` for stacks
/// without rotary embedding (`ForwardStack`).
#[allow(clippy::too_many_arguments)]
pub fn forward_stack(
    weights: &DeviceWeights,
    stack: &StackWeights,
    rope: Option<Rope<'_>>,
    layout: &AttentionLayout<'_>,
    cache: &mut KvCache,
    mut hidden: DeviceSliceMut<'_, f32>,
    output: DeviceSliceMut<'_, f32>,
    rows: usize,
    scratch: &mut StackScratch,
    gemm: &mut Bf16Gemm,
    stream: &Stream,
) -> Result<(), String> {
    let launch = |error: LaunchError| error.to_string();
    let config = &stack.config;
    let width = config.hidden;
    let q_width = config.heads * config.head_dim;
    let qkv_width = (config.heads + 2 * config.kv_heads) * config.head_dim;
    let s = stream;
    (|| -> Result<(), HipError> {
        reserve(s, &mut scratch.normed, rows * width)?;
        reserve(s, &mut scratch.qkv, rows * qkv_width)?;
        reserve(s, &mut scratch.q, rows * q_width)?;
        reserve(s, &mut scratch.attention, rows * q_width)?;
        reserve(s, &mut scratch.gate_up, rows * 2 * config.intermediate)?;
        reserve(s, &mut scratch.activation, rows * config.intermediate)
    })()
    .map_err(|e| e.to_string())?;
    let capacity = cache.capacity;
    let rope = rope.filter(|_| config.rope);
    let table = rope.map(|rope| rope.table);
    // Without a table the kernel reads no row; the positions still stay
    // below the bound their buffer declares.
    let table_rows = rope.map_or(layout.positions.bound() as usize, |rope| rope.rows);
    for (index, layer) in stack.layers.iter().enumerate() {
        let range = cache.range(index);
        k::rms_norm(
            s,
            hidden.as_const(),
            weights.vector(layer.input_norm),
            scratch.normed.slice_mut(0..rows * width),
            None,
            rows,
            width,
            config.rms_eps,
        )
        .map_err(launch)?;
        gemm.linear(
            s,
            scratch.normed.slice(0..rows * width),
            weights.matrix(layer.qkv.weight),
            scratch.qkv.slice_mut(0..rows * qkv_width),
            rows,
            width,
            qkv_width,
            false,
        )
        .map_err(launch)?;
        k::qkv_rope(
            s,
            scratch.qkv.slice(0..rows * qkv_width),
            table,
            layout.positions,
            layout.cache_rows,
            scratch.q.slice_mut(0..rows * q_width),
            cache.keys.slice_mut(range.clone()),
            cache.values.slice_mut(range.clone()),
            rows,
            config.heads,
            config.kv_heads,
            config.head_dim,
            capacity,
            table_rows,
        )
        .map_err(launch)?;
        k::attention(
            s,
            scratch.q.slice(0..rows * q_width),
            cache.keys.slice(range.clone()),
            cache.values.slice(range),
            scratch.attention.slice_mut(0..rows * q_width),
            rows,
            layout.rows_per_sequence,
            layout.keys_per_sequence,
            layout.past,
            layout.causal,
            config.heads,
            config.kv_heads,
            config.head_dim,
            capacity,
        )
        .map_err(launch)?;
        gemm.linear(
            s,
            scratch.attention.slice(0..rows * q_width),
            weights.matrix(layer.o.weight),
            hidden.reborrow(),
            rows,
            q_width,
            width,
            true,
        )
        .map_err(launch)?;
        k::rms_norm(
            s,
            hidden.as_const(),
            weights.vector(layer.post_norm),
            scratch.normed.slice_mut(0..rows * width),
            None,
            rows,
            width,
            config.rms_eps,
        )
        .map_err(launch)?;
        gemm.linear(
            s,
            scratch.normed.slice(0..rows * width),
            weights.matrix(layer.gate_up.weight),
            scratch.gate_up.slice_mut(0..rows * 2 * config.intermediate),
            rows,
            width,
            2 * config.intermediate,
            false,
        )
        .map_err(launch)?;
        k::swi_glu(
            s,
            scratch.gate_up.slice(0..rows * 2 * config.intermediate),
            scratch.activation.slice_mut(0..rows * config.intermediate),
            rows,
            config.intermediate,
        )
        .map_err(launch)?;
        gemm.linear(
            s,
            scratch.activation.slice(0..rows * config.intermediate),
            weights.matrix(layer.down.weight),
            hidden.reborrow(),
            rows,
            config.intermediate,
            width,
            true,
        )
        .map_err(launch)?;
    }
    k::rms_norm(
        s,
        hidden.as_const(),
        weights.vector(stack.norm),
        scratch.normed.slice_mut(0..rows * width),
        Some(output),
        rows,
        width,
        config.rms_eps,
    )
    .map_err(launch)
}
