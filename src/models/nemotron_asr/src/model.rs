//! The encoder, prompt projector and RNN-T transducer (`model.cpp`,
//! `model.hpp`, `device.hpp`) of transformers' `Nemotron3_5AsrForRNNT` and
//! `NemotronAsrStreamingEncoder` (Apache-2.0), which port NVIDIA NeMo's
//! cache-aware FastConformer and RNN-T greedy decoding (Apache-2.0): one
//! chunked path for offline and live transcription, positional terms
//! tabulated per layer at load.
//!
//! The matrices of the subsampling, the conformer layers, the projections
//! and the joint's head are float16 tiles for the matrix cores, the
//! prediction network's matrices float16 rows, all summed in float32; the
//! subsampling's activations and cached rows and the prompt projector's
//! activations are float16 too (saturated at 65504). Everything else
//! (convolution kernels, norms, biases, positional tables) stays float32.
//! Only the positional tables come from `hipblasSgemm`, at load, as in C++.

use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use gufo_core::cancellable_gate::CancellableGate;
use gufo_hip::blas::{Blas, Op};
use gufo_hip::{
    DeviceBuffer, DeviceCopy, DeviceSlice, DeviceSliceMut, DeviceTensor, F16, HipError, Launch,
    LaunchError, Stream,
};
use gufo_nemotron_asr_kernels as k;

use crate::config::{self, Config};
use crate::features::FeatureStream;
use crate::safetensors::{DType, SafetensorsFile, Tensor};
use crate::tokenizer::{TextDecoder, Tokenizer};

/// Encoder frames (80 ms each) of a pass at most (`kMaxPassFrames`).
pub const MAX_PASS_FRAMES: usize = 224;
/// Idle stream devices a model keeps for reuse (`kPooledDevices`).
const POOLED_DEVICES: usize = 4;
/// Frames the joint takes per weight pass at most (`kMaxJointFrames`).
const MAX_JOINT_FRAMES: usize = 16;

/// Why a transcription stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The cancellation callback asked to stop (`Cancelled`).
    Cancelled,
    Failed(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => f.write_str("transcription cancelled"),
            Self::Failed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {}

impl From<HipError> for Error {
    fn from(error: HipError) -> Self {
        Self::Failed(error.to_string())
    }
}

impl From<LaunchError> for Error {
    fn from(error: LaunchError) -> Self {
        Self::Failed(error.to_string())
    }
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::Failed(message)
    }
}

type Result<T> = std::result::Result<T, Error>;

fn failed(message: impl Into<String>) -> Error {
    Error::Failed(message.into())
}

/// `i32` for the kernels' `int` extents.
fn int(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| failed("an extent exceeds the kernels' int range"))
}

struct LayerWeights {
    ff1_norm_w: DeviceTensor<f32>,
    ff1_norm_b: DeviceTensor<f32>,
    ff1_up: DeviceTensor<F16>,
    ff1_down: DeviceTensor<F16>,
    att_norm_w: DeviceTensor<f32>,
    att_norm_b: DeviceTensor<f32>,
    /// `[3 * hidden][hidden]`.
    qkv: DeviceTensor<F16>,
    out: DeviceTensor<F16>,
    bias_u: DeviceTensor<f32>,
    bias_v: DeviceTensor<f32>,
    /// `[pos_rows][hidden]`.
    pos: DeviceTensor<f32>,
    conv_norm_w: DeviceTensor<f32>,
    conv_norm_b: DeviceTensor<f32>,
    /// `[2 * hidden][hidden]`.
    pointwise1: DeviceTensor<F16>,
    /// `[hidden][kernel]`.
    depthwise: DeviceTensor<f32>,
    conv_inner_norm_w: DeviceTensor<f32>,
    conv_inner_norm_b: DeviceTensor<f32>,
    pointwise2: DeviceTensor<F16>,
    ff2_norm_w: DeviceTensor<f32>,
    ff2_norm_b: DeviceTensor<f32>,
    ff2_up: DeviceTensor<F16>,
    ff2_down: DeviceTensor<F16>,
    out_norm_w: DeviceTensor<f32>,
    out_norm_b: DeviceTensor<f32>,
}

/// The weights on the device (`Weights`).
struct Weights {
    stem_w: DeviceTensor<f32>,
    stem_b: DeviceTensor<f32>,
    stage_dw_w: [DeviceTensor<f32>; 2],
    stage_dw_b: [DeviceTensor<f32>; 2],
    stage_pw_w: [DeviceTensor<F16>; 2],
    stage_pw_b: [DeviceTensor<f32>; 2],
    /// Columns as `[frequency][channel]`.
    sub_linear_w: DeviceTensor<F16>,
    sub_linear_b: DeviceTensor<f32>,
    layers: Vec<LayerWeights>,
    /// `[prompt_hidden][hidden + prompts]`.
    prompt_w1: DeviceTensor<F16>,
    /// `[prompts][prompt_hidden]`, the one-hot column folded into the bias.
    prompt_bias: DeviceTensor<f32>,
    prompt_w2: DeviceTensor<F16>,
    prompt_b2: DeviceTensor<f32>,
    proj_w: DeviceTensor<F16>,
    proj_b: DeviceTensor<f32>,
    embedding: DeviceTensor<f32>,
    /// `[4 * width][2 * width]`: input | hidden.
    lstm_w: Vec<DeviceTensor<F16>>,
    /// `b_ih + b_hh`.
    lstm_b: Vec<DeviceTensor<f32>>,
    dec_proj_w: DeviceTensor<F16>,
    dec_proj_b: DeviceTensor<f32>,
    head_w: DeviceTensor<F16>,
    head_b: DeviceTensor<f32>,
    /// Mel bins, then the frequencies after each stage.
    frequencies: [usize; 4],
    pos_rows: usize,
    pos_zero: usize,
}

/// `y[rows][out] = x[rows][in] (row stride ldx) · Wᵀ`, W `[out][ldw]`
/// row-major, through `hipblasSgemm` (`Linear` on a hipBLAS handle).
#[allow(clippy::too_many_arguments)]
fn sgemm_linear(
    blas: &Blas,
    stream: &Stream,
    x: DeviceSlice<'_, f32>,
    rows: usize,
    input: usize,
    ldx: usize,
    w: DeviceSlice<'_, f32>,
    ldw: usize,
    out: usize,
    y: DeviceSliceMut<'_, f32>,
    ldy: usize,
) -> Result<()> {
    if rows == 0 {
        return Ok(());
    }
    blas.sgemm(
        stream,
        Op::T,
        Op::N,
        out,
        rows,
        input,
        1.0,
        w,
        ldw,
        x,
        ldx,
        y,
        ldy,
    )?;
    Ok(())
}

/// `y[rows][out] = scale · x[rows][in] Wᵀ`, plus `y` when `accumulate`, on
/// the matrix cores (`Linear` on tiles).
#[allow(clippy::too_many_arguments)]
fn wmma_linear(
    stream: &Stream,
    x: DeviceSlice<'_, F16>,
    tiles: DeviceSlice<'_, F16>,
    y: DeviceSliceMut<'_, f32>,
    rows: usize,
    input: usize,
    out: usize,
    accumulate: bool,
    scale: f32,
) -> Result<()> {
    supported(k::wmma_linear(
        stream, x, tiles, y, rows, input, out, accumulate, scale, None, false,
    )?)
}

/// `y[rows][out] = x[rows][in] Wᵀ + bias` on the matrix cores (`Linear`
/// with a bias).
#[allow(clippy::too_many_arguments)]
fn wmma_linear_bias(
    stream: &Stream,
    x: DeviceSlice<'_, F16>,
    tiles: DeviceSlice<'_, F16>,
    y: DeviceSliceMut<'_, f32>,
    rows: usize,
    input: usize,
    out: usize,
    bias: DeviceSlice<'_, f32>,
) -> Result<()> {
    supported(k::wmma_linear(
        stream,
        x,
        tiles,
        y,
        rows,
        input,
        out,
        false,
        1.0,
        Some(bias),
        false,
    )?)
}

/// `y = x Wᵀ + bias`, ReLU when `relu`, as float16 (`LinearHalf`).
#[allow(clippy::too_many_arguments)]
fn wmma_linear_half(
    stream: &Stream,
    x: DeviceSlice<'_, F16>,
    tiles: DeviceSlice<'_, F16>,
    y: DeviceSliceMut<'_, F16>,
    rows: usize,
    input: usize,
    out: usize,
    bias: DeviceSlice<'_, f32>,
    relu: bool,
) -> Result<()> {
    supported(k::wmma_linear_half(
        stream,
        x,
        tiles,
        y,
        rows,
        input,
        out,
        Some(bias),
        relu,
    )?)
}

fn supported(launch: Launch) -> Result<()> {
    match launch {
        Launch::Launched => Ok(()),
        Launch::Unsupported => Err(failed("unsupported matrix-core product shape")),
    }
}

/// A plain `hipMalloc` buffer of at least one element, as C++
/// `DeviceBuffer` reserves (pool memory is slower for hot buffers).
fn alloc<T: DeviceCopy>(stream: &Stream, len: usize) -> Result<DeviceBuffer<T>> {
    Ok(stream.alloc_persistent(len.max(1))?)
}

/// Uploads tensors to the device (`Loader`).
struct Loader<'a> {
    file: &'a SafetensorsFile,
    stream: &'a Stream,
    staging: Option<DeviceBuffer<f32>>,
    half_staging: Option<DeviceBuffer<F16>>,
}

impl<'a> Loader<'a> {
    fn get(&self, name: &str, shape: &[usize]) -> Result<&'a Tensor> {
        self.file.expect(name, DType::F32, shape).map_err(|error| {
            if error.is_empty() {
                failed(format!("model lacks {name}"))
            } else {
                failed(error)
            }
        })
    }

    fn values(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        Ok(self.get(name, shape)?.f32_values()?)
    }

    fn upload_values(&self, values: &[f32]) -> Result<DeviceTensor<f32>> {
        let mut buffer = alloc::<f32>(self.stream, values.len())?;
        self.stream
            .upload(values, buffer.slice_mut(0..values.len()))?;
        Ok(buffer.freeze(self.stream)?)
    }

    fn upload(&self, name: &str, shape: &[usize]) -> Result<DeviceTensor<f32>> {
        let tensor = self.get(name, shape)?;
        let count = tensor.elements();
        let mut buffer = alloc::<f32>(self.stream, count)?;
        self.stream
            .upload_bytes(tensor.data(), buffer.slice_mut(0..count))?;
        Ok(buffer.freeze(self.stream)?)
    }

    /// The float32 staging buffer holding `values` on the device.
    fn stage<'b>(
        stream: &Stream,
        staging: &'b mut Option<DeviceBuffer<f32>>,
        values: &[u8],
        count: usize,
    ) -> Result<&'b DeviceBuffer<f32>> {
        if staging.as_ref().is_none_or(|b| b.len() < count) {
            *staging = Some(alloc(stream, count)?);
        }
        let staging = staging.as_mut().ok_or_else(|| failed("staging buffer"))?;
        stream.upload_bytes(values, staging.slice_mut(0..count))?;
        Ok(staging)
    }

    /// Uploads float32 values as float16 (`UploadHalf`).
    fn upload_half_bytes(&mut self, bytes: &[u8], count: usize) -> Result<DeviceTensor<F16>> {
        let stream = self.stream;
        let staging = Self::stage(stream, &mut self.staging, bytes, count)?;
        let mut buffer = alloc::<F16>(stream, count)?;
        k::to_half(
            stream,
            staging.slice(0..count),
            buffer.slice_mut(0..count),
            count,
        )?;
        stream.synchronize()?;
        Ok(buffer.freeze(stream)?)
    }

    fn upload_half(&mut self, name: &str, shape: &[usize]) -> Result<DeviceTensor<F16>> {
        let tensor = self.get(name, shape)?;
        self.upload_half_bytes(tensor.data(), tensor.elements())
    }

    /// Uploads a float32 matrix `[rows][in]` as float16 tiles of the
    /// matrix-core products (`UploadTiles`).
    fn upload_tiles_bytes(
        &mut self,
        bytes: &[u8],
        rows: usize,
        input: usize,
    ) -> Result<DeviceTensor<F16>> {
        let stream = self.stream;
        let count = rows * input;
        if self.half_staging.as_ref().is_none_or(|b| b.len() < count) {
            self.half_staging = Some(alloc(stream, count)?);
        }
        let staging = Self::stage(stream, &mut self.staging, bytes, count)?.slice(0..count);
        let half = self
            .half_staging
            .as_mut()
            .ok_or_else(|| failed("staging buffer"))?;
        k::to_half(stream, staging, half.slice_mut(0..count), count)?;
        let mut buffer = alloc::<F16>(stream, count)?;
        k::half_to_tiles(
            stream,
            half.slice(0..count),
            rows,
            input,
            buffer.slice_mut(0..count),
        )?;
        stream.synchronize()?;
        Ok(buffer.freeze(stream)?)
    }

    fn upload_tiles(&mut self, name: &str, rows: usize, input: usize) -> Result<DeviceTensor<F16>> {
        let tensor = self.get(name, &[rows, input])?;
        self.upload_tiles_bytes(tensor.data(), rows, input)
    }

    /// `upload_tiles` for a convolution weight `[rows][in][1]`.
    fn upload_tiles_1x1(
        &mut self,
        name: &str,
        rows: usize,
        input: usize,
    ) -> Result<DeviceTensor<F16>> {
        let tensor = self.get(name, &[rows, input, 1])?;
        self.upload_tiles_bytes(tensor.data(), rows, input)
    }
}

fn bytes_of(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// The sinusoids of transformers' relative positional encoding for
/// distances `first..first + rows - 1`, computed in float32 as PyTorch does
/// (`Sinusoids`).
fn sinusoids(hidden: usize, first: isize, rows: usize) -> Vec<f32> {
    let mut table = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        let distance = (first + r as isize) as f32;
        for k in 0..hidden / 2 {
            let exponent = (2 * k) as f32 / hidden as f32;
            let inverse = 1.0f32 / 10000.0f32.powf(exponent);
            let angle = distance * inverse;
            table[r * hidden + 2 * k] = angle.sin();
            table[r * hidden + 2 * k + 1] = angle.cos();
        }
    }
    table
}

/// Loads `model.safetensors` (`LoadWeights`).
fn load_weights(
    config: &Config,
    root: &Path,
    max_lookahead: usize,
    stream: &Stream,
) -> Result<Weights> {
    let file = SafetensorsFile::open(&root.join("model.safetensors"))?;
    file.prefault()?;
    let mut load = Loader {
        file: &file,
        stream,
        staging: None,
        half_staging: None,
    };
    let e = &config.encoder;
    let h = e.hidden;
    let c = e.subsampling_channels;

    let mut frequencies = [e.mel_bins, 0, 0, 0];
    for i in 1..4 {
        frequencies[i] = frequencies[i - 1] / 2 + 1;
    }

    let sub = "encoder.subsampling.";
    let stem_w = load.upload(&format!("{sub}conv_in.weight"), &[c, 1, 3, 3])?;
    let stem_b = load.upload(&format!("{sub}conv_in.bias"), &[c])?;
    let stage = |load: &Loader<'_>, s: usize, name: &str, shape: &[usize]| {
        load.upload(&format!("{sub}layers.{s}.{name}"), shape)
    };
    let stage_dw_w = [
        stage(&load, 0, "depthwise_conv.weight", &[c, 1, 3, 3])?,
        stage(&load, 1, "depthwise_conv.weight", &[c, 1, 3, 3])?,
    ];
    let stage_dw_b = [
        stage(&load, 0, "depthwise_conv.bias", &[c])?,
        stage(&load, 1, "depthwise_conv.bias", &[c])?,
    ];
    let mut pointwise = |s: usize| -> Result<DeviceTensor<F16>> {
        let tensor = load.get(
            &format!("{sub}layers.{s}.pointwise_conv.weight"),
            &[c, c, 1, 1],
        )?;
        load.upload_tiles_bytes(tensor.data(), c, c)
    };
    let stage_pw_w = [pointwise(0)?, pointwise(1)?];
    let stage_pw_b = [
        stage(&load, 0, "pointwise_conv.bias", &[c])?,
        stage(&load, 1, "pointwise_conv.bias", &[c])?,
    ];
    let (sub_linear_w, sub_linear_b) = {
        // The reference flattens [channel][frequency]; ours is
        // [frequency][channel].
        let f = frequencies[3];
        let src = load.values(&format!("{sub}linear.weight"), &[h, c * f])?;
        let mut permuted = vec![0.0f32; h * c * f];
        for o in 0..h {
            for ch in 0..c {
                for q in 0..f {
                    permuted[o * c * f + q * c + ch] = src[o * c * f + ch * f + q];
                }
            }
        }
        (
            load.upload_tiles_bytes(&bytes_of(&permuted), h, c * f)?,
            load.upload(&format!("{sub}linear.bias"), &[h])?,
        )
    };

    // Distances a query sees: from (chunk - 1) ahead to left + chunk - 1 back.
    let left = e.left_context();
    let pos_zero = max_lookahead;
    let pos_rows = left + 2 * max_lookahead + 1;
    let sinusoid_table = load.upload_values(&sinusoids(h, -(max_lookahead as isize), pos_rows))?;
    let mut relative = alloc::<f32>(stream, h * h)?;
    let blas = Blas::with_default_atomics(stream)?;

    let mut layers = Vec::with_capacity(e.layers);
    for i in 0..e.layers {
        let p = format!("encoder.layers.{i}.");
        let ff1_norm_w = load.upload(&format!("{p}norm_feed_forward1.weight"), &[h])?;
        let ff1_norm_b = load.upload(&format!("{p}norm_feed_forward1.bias"), &[h])?;
        let ff1_up = load.upload_tiles(
            &format!("{p}feed_forward1.linear1.weight"),
            e.intermediate,
            h,
        )?;
        let ff1_down = load.upload_tiles(
            &format!("{p}feed_forward1.linear2.weight"),
            h,
            e.intermediate,
        )?;
        let att_norm_w = load.upload(&format!("{p}norm_self_att.weight"), &[h])?;
        let att_norm_b = load.upload(&format!("{p}norm_self_att.bias"), &[h])?;
        let qkv = {
            let mut joined = Vec::with_capacity(3 * h * h * 4);
            for name in ["q_proj", "k_proj", "v_proj"] {
                let tensor = load.get(&format!("{p}self_attn.{name}.weight"), &[h, h])?;
                joined.extend_from_slice(tensor.data());
            }
            load.upload_tiles_bytes(&joined, 3 * h, h)?
        };
        let out = load.upload_tiles(&format!("{p}self_attn.o_proj.weight"), h, h)?;
        let bias_u = load.upload(&format!("{p}self_attn.bias_u"), &[e.heads, e.head_dim()])?;
        let bias_v = load.upload(&format!("{p}self_attn.bias_v"), &[e.heads, e.head_dim()])?;
        let pos = {
            let r = load.get(&format!("{p}self_attn.relative_k_proj.weight"), &[h, h])?;
            stream.upload_bytes(r.data(), relative.slice_mut(0..h * h))?;
            let mut table = alloc::<f32>(stream, pos_rows * h)?;
            sgemm_linear(
                &blas,
                stream,
                sinusoid_table.view(),
                pos_rows,
                h,
                h,
                relative.slice(0..h * h),
                h,
                h,
                table.slice_mut(0..pos_rows * h),
                h,
            )?;
            stream.synchronize()?;
            table.freeze(stream)?
        };
        let conv_norm_w = load.upload(&format!("{p}norm_conv.weight"), &[h])?;
        let conv_norm_b = load.upload(&format!("{p}norm_conv.bias"), &[h])?;
        let pointwise1 =
            load.upload_tiles_1x1(&format!("{p}conv.pointwise_conv1.weight"), 2 * h, h)?;
        let depthwise = load.upload(
            &format!("{p}conv.depthwise_conv.weight"),
            &[h, 1, e.conv_kernel],
        )?;
        let conv_inner_norm_w = load.upload(&format!("{p}conv.norm.weight"), &[h])?;
        let conv_inner_norm_b = load.upload(&format!("{p}conv.norm.bias"), &[h])?;
        let pointwise2 = load.upload_tiles_1x1(&format!("{p}conv.pointwise_conv2.weight"), h, h)?;
        let ff2_norm_w = load.upload(&format!("{p}norm_feed_forward2.weight"), &[h])?;
        let ff2_norm_b = load.upload(&format!("{p}norm_feed_forward2.bias"), &[h])?;
        let ff2_up = load.upload_tiles(
            &format!("{p}feed_forward2.linear1.weight"),
            e.intermediate,
            h,
        )?;
        let ff2_down = load.upload_tiles(
            &format!("{p}feed_forward2.linear2.weight"),
            h,
            e.intermediate,
        )?;
        let out_norm_w = load.upload(&format!("{p}norm_out.weight"), &[h])?;
        let out_norm_b = load.upload(&format!("{p}norm_out.bias"), &[h])?;
        layers.push(LayerWeights {
            ff1_norm_w,
            ff1_norm_b,
            ff1_up,
            ff1_down,
            att_norm_w,
            att_norm_b,
            qkv,
            out,
            bias_u,
            bias_v,
            pos,
            conv_norm_w,
            conv_norm_b,
            pointwise1,
            depthwise,
            conv_inner_norm_w,
            conv_inner_norm_b,
            pointwise2,
            ff2_norm_w,
            ff2_norm_b,
            ff2_up,
            ff2_down,
            out_norm_w,
            out_norm_b,
        });
    }
    drop(blas);

    let ph = config.prompt_hidden;
    let prompts = config.prompts;
    let d = config.decoder_hidden;
    let (prompt_w1, prompt_bias) = {
        let w1 = load.values("prompt_projector.linear_1.weight", &[ph, h + prompts])?;
        let b1 = load.values("prompt_projector.linear_1.bias", &[ph])?;
        // The one-hot prompt selects one column of linear_1: fold it into the bias.
        let mut hidden = Vec::with_capacity(ph * h);
        for r in 0..ph {
            hidden.extend_from_slice(&w1[r * (h + prompts)..r * (h + prompts) + h]);
        }
        let w1_tiles = load.upload_tiles_bytes(&bytes_of(&hidden), ph, h)?;
        let mut folded = vec![0.0f32; prompts * ph];
        for q in 0..prompts {
            for r in 0..ph {
                folded[q * ph + r] = b1[r] + w1[r * (h + prompts) + h + q];
            }
        }
        (w1_tiles, load.upload_values(&folded)?)
    };
    let prompt_w2 = load.upload_tiles("prompt_projector.linear_2.weight", h, ph)?;
    let prompt_b2 = load.upload("prompt_projector.linear_2.bias", &[h])?;
    let proj_w = load.upload_tiles("encoder_projector.weight", d, h)?;
    let proj_b = load.upload("encoder_projector.bias", &[d])?;

    let embedding = load.upload("decoder.embedding.weight", &[config.vocab, d])?;
    let mut lstm_w = Vec::with_capacity(config.decoder_layers);
    let mut lstm_b = Vec::with_capacity(config.decoder_layers);
    for l in 0..config.decoder_layers {
        let wi = load.values(&format!("decoder.lstm.weight_ih_l{l}"), &[4 * d, d])?;
        let wh = load.values(&format!("decoder.lstm.weight_hh_l{l}"), &[4 * d, d])?;
        let bi = load.values(&format!("decoder.lstm.bias_ih_l{l}"), &[4 * d])?;
        let bh = load.values(&format!("decoder.lstm.bias_hh_l{l}"), &[4 * d])?;
        let mut joined = vec![0.0f32; 4 * d * 2 * d];
        let mut bias = vec![0.0f32; 4 * d];
        for r in 0..4 * d {
            joined[r * 2 * d..r * 2 * d + d].copy_from_slice(&wi[r * d..(r + 1) * d]);
            joined[r * 2 * d + d..(r + 1) * 2 * d].copy_from_slice(&wh[r * d..(r + 1) * d]);
            bias[r] = bi[r] + bh[r];
        }
        lstm_w.push(load.upload_half_bytes(&bytes_of(&joined), joined.len())?);
        lstm_b.push(load.upload_values(&bias)?);
    }
    let dec_proj_w = load.upload_half("decoder.decoder_projector.weight", &[d, d])?;
    let dec_proj_b = load.upload("decoder.decoder_projector.bias", &[d])?;
    let head_w = load.upload_tiles("joint.head.weight", config.vocab, d)?;
    let head_b = load.upload("joint.head.bias", &[config.vocab])?;
    drop(load);
    file.release_pages();
    Ok(Weights {
        stem_w,
        stem_b,
        stage_dw_w,
        stage_dw_b,
        stage_pw_w,
        stage_pw_b,
        sub_linear_w,
        sub_linear_b,
        layers,
        prompt_w1,
        prompt_bias,
        prompt_w2,
        prompt_b2,
        proj_w,
        proj_b,
        embedding,
        lstm_w,
        lstm_b,
        dec_proj_w,
        dec_proj_b,
        head_w,
        head_b,
        frequencies,
        pos_rows,
        pos_zero,
    })
}

/// The loaded checkpoint: configuration, tokenizer and weights on the GPU.
/// Utterances share it; each has its own HIP stream and state (`Model`).
pub struct Model {
    config: Config,
    tokenizer: Arc<Tokenizer>,
    weights: Weights,
    max_lookahead: usize,
    pool: Mutex<Vec<StreamDevice>>,
    /// One pass computes at a time, in the order utterances ask (FIFO).
    gate: CancellableGate,
}

impl Model {
    /// Loads a Hugging Face model directory (`Model::Load`); errors carry
    /// the `Nemotron ASR: ` prefix.
    pub fn load(root: &Path) -> std::result::Result<Self, String> {
        Self::load_inner(root).map_err(|error| format!("Nemotron ASR: {error}"))
    }

    fn load_inner(root: &Path) -> Result<Self> {
        let config = config::load_config(root)?;
        config::is_supported(&config)?;
        let tokenizer = Arc::new(Tokenizer::load(&root.join("tokenizer.json"))?);
        if tokenizer.size() + 1 < config.vocab {
            return Err(failed("tokenizer.json has fewer tokens than the joint"));
        }
        let max_lookahead = config
            .encoder
            .lookaheads
            .iter()
            .copied()
            .max()
            .ok_or_else(|| failed("the checkpoint lists no lookahead"))?;
        let stream = Stream::new()?;
        let weights = load_weights(&config, root, max_lookahead, &stream)?;
        Ok(Self {
            config,
            tokenizer,
            weights,
            max_lookahead,
            pool: Mutex::new(Vec::new()),
            gate: CancellableGate::default(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    /// Largest supported lookahead: the positional tables cover its chunks.
    pub fn max_lookahead(&self) -> usize {
        self.max_lookahead
    }

    fn lock_pool(&self) -> MutexGuard<'_, Vec<StreamDevice>> {
        self.pool.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// An utterance's device state for passes of up to `frames` encoder
    /// frames: the smallest idle one that holds them, else a new one
    /// (`AcquireDevice`).
    fn acquire_device(&self, frames: usize) -> Result<StreamDevice> {
        {
            let mut pool = self.lock_pool();
            let chosen = pool
                .iter()
                .enumerate()
                .filter(|(_, device)| device.frames >= frames)
                .min_by_key(|(_, device)| device.frames)
                .map(|(index, _)| index);
            if let Some(index) = chosen {
                return Ok(pool.remove(index));
            }
        }
        StreamDevice::new(&self.config, &self.weights, frames)
    }

    /// Work a cancelled utterance left queued finishes before another reuses
    /// the buffers; a device whose stream failed is dropped
    /// (`ReleaseDevice`).
    fn release_device(&self, device: StreamDevice) {
        if device.stream.synchronize().is_err() {
            return;
        }
        let mut pool = self.lock_pool();
        if pool.len() < POOLED_DEVICES {
            pool.push(device);
        }
    }
}

/// An utterance's HIP stream, scratch buffers and caches, for passes of up
/// to `frames` encoder frames (and 8 times as many mel frames). Utterances
/// of any lookahead reuse them (`StreamDevice`). Fields drop in order: the
/// buffers before their stream.
struct StreamDevice {
    frames: usize,
    mel: DeviceBuffer<f32>,
    // The subsampling's float16 activations: each stage's input with its
    // two history rows, the cached input row, the depthwise output.
    s1: DeviceBuffer<F16>,
    s1_cache: DeviceBuffer<F16>,
    s2_in: DeviceBuffer<F16>,
    s2: DeviceBuffer<F16>,
    s2_cache: DeviceBuffer<F16>,
    s3_in: DeviceBuffer<F16>,
    s3: DeviceBuffer<F16>,
    x: DeviceBuffer<f32>,
    wide: DeviceBuffer<f32>,
    mid: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    norm_half: DeviceBuffer<F16>,
    wide_half: DeviceBuffer<F16>,
    attention_half: DeviceBuffer<F16>,
    /// `[layers][left][hidden]`.
    key_cache: DeviceBuffer<f32>,
    value_cache: DeviceBuffer<f32>,
    /// `[layers][kernel - 1][hidden]`.
    conv_cache: DeviceBuffer<f32>,
    h: DeviceBuffer<f32>,
    c: DeviceBuffer<f32>,
    gates: DeviceBuffer<f32>,
    decoder_out: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    joint_input: DeviceBuffer<F16>,
    best: DeviceBuffer<u64>,
    best_host: Vec<u64>,
    stream: Stream,
}

impl StreamDevice {
    fn new(config: &Config, w: &Weights, capacity: usize) -> Result<Self> {
        let e = &config.encoder;
        let stream = Stream::new()?;
        let h = e.hidden;
        let ch = e.subsampling_channels;
        let max_mel = capacity * e.subsampling_factor;
        let t1 = max_mel / 2 + 2;
        let t2 = t1 / 2 + 2;
        let t3 = t2 / 2 + 2;
        let rows = capacity.max(t3);
        let widest = e.intermediate.max(3 * h);
        let width = config.decoder_hidden;
        let s = &stream;
        Ok(Self {
            frames: capacity,
            mel: alloc(s, (max_mel + 2) * e.mel_bins)?,
            s1: alloc(s, (t1 + 2) * w.frequencies[1] * ch)?,
            s1_cache: alloc(s, w.frequencies[1] * ch)?,
            s2_in: alloc(s, t2 * w.frequencies[2] * ch)?,
            s2: alloc(s, (t2 + 2) * w.frequencies[2] * ch)?,
            s2_cache: alloc(s, w.frequencies[2] * ch)?,
            s3_in: alloc(s, t3 * w.frequencies[3] * ch)?,
            s3: alloc(s, t3 * w.frequencies[3] * ch)?,
            x: alloc(s, rows * h)?,
            wide: alloc(s, rows * widest)?,
            mid: alloc(s, rows * h)?,
            projected: alloc(s, rows * width)?,
            norm_half: alloc(s, rows * h)?,
            wide_half: alloc(s, rows * e.intermediate.max(config.prompt_hidden))?,
            attention_half: alloc(s, rows * h)?,
            key_cache: alloc(s, e.layers * e.left_context() * h)?,
            value_cache: alloc(s, e.layers * e.left_context() * h)?,
            conv_cache: alloc(s, e.layers * (e.conv_kernel - 1) * h)?,
            h: alloc(s, config.decoder_layers * width)?,
            c: alloc(s, config.decoder_layers * width)?,
            gates: alloc(s, 4 * width)?,
            decoder_out: alloc(s, width)?,
            logits: alloc(s, MAX_JOINT_FRAMES * config.vocab)?,
            joint_input: alloc(s, MAX_JOINT_FRAMES * width)?,
            best: alloc(s, MAX_JOINT_FRAMES)?,
            best_host: vec![0; MAX_JOINT_FRAMES],
            stream,
        })
    }
}

/// Encoder activations of one pass, copied to the host (for tests).
#[derive(Debug, Default, Clone)]
pub struct EncoderTrace {
    /// `[frames][hidden]`.
    pub subsampled: Vec<f32>,
    /// `[frames][hidden]`, the encoder output.
    pub hidden: Vec<f32>,
    /// `[frames][decoder_hidden]`.
    pub projected: Vec<f32>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pending {
    Eager,
    Lazy,
    Final,
}

/// Rows a causal 3-tap stride-2 convolution makes of `rows` input rows.
fn strided(rows: usize) -> usize {
    if rows >= 3 { (rows - 3) / 2 + 1 } else { 0 }
}

/// The token of a packed arg max (`ArgmaxToken`).
fn argmax_token(packed: u64) -> u32 {
    0xFFFF_FFFF - (packed & 0xFFFF_FFFF) as u32
}

/// A polled cancellation check.
pub type Cancellation<'a> = Box<dyn Fn() -> bool + 'a>;
/// Called after each pass that emitted tokens; false stops.
pub type TokenCallback<'a> = Box<dyn FnMut(&TextDecoder) -> bool + 'a>;

/// One utterance in cache-aware streaming: 16 kHz samples in, tokens out
/// (C++ `Stream`). The encoder runs on whole chunks (1 + 8 · lookahead mel
/// frames, then 8 · (lookahead + 1)), greedy RNN-T decoding follows each
/// pass; [`Utterance::finish`] encodes the rest as offline transcription
/// ends an utterance (its last chunk partial). Feeding the same samples in
/// any split gives the same tokens.
pub struct Utterance<'m> {
    model: &'m Model,
    lookahead: usize,
    /// Encoder frames per chunk.
    chunk: usize,
    /// Mel frames of the first chunk.
    first_mel: usize,
    /// Mel frames of later chunks.
    chunk_mel: usize,
    /// Chunks per encoder pass at most.
    pass_chunks: usize,
    prompt: usize,
    features: FeatureStream,
    /// Mel frames not yet encoded.
    pending: Vec<f32>,
    /// The stem's cached input row.
    last_mel: Vec<f32>,
    device: Option<StreamDevice>,
    text: TextDecoder,
    frames_done: usize,
    chunks: usize,
    last_pass_frames: usize,
    finished: bool,
    encoder_ms: f64,
    decoder_ms: f64,
    feature_ms: f64,
    cancelled: Option<Cancellation<'m>>,
    on_tokens: Option<TokenCallback<'m>>,
}

fn since(begin: Instant) -> f64 {
    begin.elapsed().as_secs_f64() * 1000.0
}

impl<'m> Utterance<'m> {
    /// Passes hold up to `pass_frames` encoder frames (whole chunks, at
    /// least one): smaller passes report text sooner, take less device
    /// memory and let other utterances compute in between; larger ones
    /// finish sooner.
    pub fn new(
        model: &'m Model,
        lookahead: usize,
        prompt: usize,
        pass_frames: usize,
    ) -> Result<Self> {
        let config = &model.config;
        let e = &config.encoder;
        if !e.lookaheads.contains(&lookahead) {
            return Err(failed(format!("unsupported lookahead {lookahead}")));
        }
        if prompt >= config.prompts {
            return Err(failed("language prompt out of range"));
        }
        let chunk = lookahead + 1;
        let pass_chunks = (pass_frames.min(MAX_PASS_FRAMES) / chunk).max(1);
        let mut device = model.acquire_device(pass_chunks * chunk)?;
        let mut utterance = Self {
            model,
            lookahead,
            chunk,
            first_mel: config::first_chunk_frames(config, lookahead),
            chunk_mel: config::chunk_frames(config, lookahead),
            pass_chunks,
            prompt,
            features: FeatureStream::new(&config.features, e.mel_bins)?,
            pending: Vec::new(),
            last_mel: Vec::new(),
            device: None,
            text: TextDecoder::new(Arc::clone(&model.tokenizer)),
            frames_done: 0,
            chunks: 0,
            last_pass_frames: 0,
            finished: false,
            encoder_ms: 0.0,
            decoder_ms: 0.0,
            feature_ms: 0.0,
            cancelled: None,
            on_tokens: None,
        };
        let h = e.hidden;
        let width = config.decoder_hidden;
        let d = &mut device;
        let conv = e.layers * (e.conv_kernel - 1) * h;
        d.stream.fill_bytes(d.conv_cache.slice_mut(0..conv), 0)?;
        let states = config.decoder_layers * width;
        d.stream.fill_bytes(d.h.slice_mut(0..states), 0)?;
        d.stream.fill_bytes(d.c.slice_mut(0..states), 0)?;
        utterance.device = Some(device);
        // The prediction network starts from the blank token (a zero
        // embedding).
        utterance.decoder_step(config.blank as u32)?;
        utterance.device()?.stream.synchronize()?;
        Ok(utterance)
    }

    fn device(&self) -> Result<&StreamDevice> {
        self.device
            .as_ref()
            .ok_or_else(|| failed("the utterance has no device"))
    }

    /// Polled between passes, while waiting for the GPU and between
    /// decoding windows; true stops with [`Error::Cancelled`].
    pub fn set_cancellation(&mut self, cancelled: Cancellation<'m>) {
        self.cancelled = Some(cancelled);
    }

    /// Called after each pass that emitted tokens, outside the model's gate.
    pub fn set_text_callback(&mut self, on_tokens: TokenCallback<'m>) {
        self.on_tokens = Some(on_tokens);
    }

    pub fn text(&self) -> &TextDecoder {
        &self.text
    }

    pub fn lookahead(&self) -> usize {
        self.lookahead
    }

    pub fn encoder_frames(&self) -> usize {
        self.frames_done
    }

    pub fn chunks(&self) -> usize {
        self.chunks
    }

    pub fn mel_frames(&self) -> usize {
        self.features.frames()
    }

    pub fn samples(&self) -> usize {
        self.features.samples()
    }

    pub fn encoder_ms(&self) -> f64 {
        self.encoder_ms
    }

    pub fn decoder_ms(&self) -> f64 {
        self.decoder_ms
    }

    pub fn feature_ms(&self) -> f64 {
        self.feature_ms
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.as_ref().is_some_and(|cancelled| cancelled())
    }

    fn check_cancelled(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Appends samples and transcribes every completed chunk. With `eager`
    /// false (audio that is complete already) only full passes are encoded
    /// while more than one is pending; [`Utterance::finish`] splits the
    /// rest evenly.
    pub fn push(&mut self, samples: &[f32], eager: bool) -> Result<()> {
        if self.finished {
            return Err(failed("Push after Finish"));
        }
        let begin = Instant::now();
        self.features.push(samples, &mut self.pending);
        self.feature_ms += since(begin);
        self.encode_ready(if eager { Pending::Eager } else { Pending::Lazy })
    }

    /// Ends the utterance: the valid frames of a centered pass
    /// (samples / hop), the signal padded with zeros, and a last chunk with
    /// only the frames that exist, as offline transcription sees it.
    pub fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        let total = self.features.frames_for(self.features.samples());
        if total == 0 {
            return Ok(());
        }
        let begin = Instant::now();
        self.features.flush(total, &mut self.pending);
        self.feature_ms += since(begin);
        self.encode_ready(Pending::Final)?;
        if !self.pending.is_empty() {
            return Err(failed("frames left after the last pass"));
        }
        Ok(())
    }

    /// Encodes the complete chunks pending, at most `pass_chunks` per pass:
    /// eagerly all of them, lazily only while more than a pass would
    /// remain, and at the end everything, the last chunk partial, in passes
    /// of even size (a pass costs nearly as much for one chunk as for
    /// many).
    fn encode_ready(&mut self, mode: Pending) -> Result<()> {
        let bins = self.model.config.encoder.mel_bins;
        let mut at = 0; // frames of `pending` encoded
        let result = loop {
            if let Err(error) = self.check_cancelled() {
                break Err(error);
            }
            let available = self.pending.len() / bins - at;
            if available == 0 {
                break Ok(());
            }
            let first = self.frames_done == 0;
            let (first_mel, chunk_mel, pass_chunks) =
                (self.first_mel, self.chunk_mel, self.pass_chunks);
            let whole_frames = |chunks: usize| -> usize {
                if chunks == 0 {
                    0
                } else if first {
                    first_mel + (chunks - 1) * chunk_mel
                } else {
                    chunks * chunk_mel
                }
            };
            let whole = if first {
                if available < first_mel {
                    0
                } else {
                    1 + (available - first_mel) / chunk_mel
                }
            } else {
                available / chunk_mel
            };
            let mut last = false;
            let frames = match mode {
                Pending::Eager => {
                    if whole == 0 {
                        break Ok(());
                    }
                    whole_frames(whole.min(pass_chunks))
                }
                Pending::Lazy => {
                    if whole <= pass_chunks {
                        break Ok(());
                    }
                    whole_frames(pass_chunks)
                }
                Pending::Final => {
                    let partial = available - whole_frames(whole);
                    let chunks = whole + usize::from(partial > 0);
                    let passes = chunks.div_ceil(pass_chunks);
                    let take = chunks.div_ceil(passes);
                    last = take >= chunks;
                    if last { available } else { whole_frames(take) }
                }
            };
            let mel = std::mem::take(&mut self.pending);
            let pass = self.encode_pass(&mel[at * bins..(at + frames) * bins], None, true, last);
            self.pending = mel;
            if let Err(error) = pass {
                break Err(error);
            }
            at += frames;
        };
        self.pending.drain(..at * bins);
        result
    }

    /// Encodes (and unless `decode` is false, decodes) chunk-complete mel
    /// frames directly (`[frames][bins]`): the first call takes
    /// 1 + 8 · lookahead + k · 8 · (lookahead + 1) frames, later ones
    /// multiples of 8 · (lookahead + 1).
    pub fn encode_features(
        &mut self,
        mel: &[f32],
        trace: Option<&mut EncoderTrace>,
        decode: bool,
    ) -> Result<()> {
        self.encode_pass(mel, trace, decode, false)
    }

    fn encode_pass(
        &mut self,
        mel: &[f32],
        trace: Option<&mut EncoderTrace>,
        decode: bool,
        last: bool,
    ) -> Result<()> {
        let bins = self.model.config.encoder.mel_bins;
        if !mel.len().is_multiple_of(bins) {
            return Err(failed("mel features are not whole frames"));
        }
        let frames = mel.len() / bins;
        let before = self.text.ids().len();
        {
            // The GPU work of the pass; text callbacks follow outside the gate.
            let model = self.model;
            let lease = {
                let cancelled = || self.is_cancelled();
                model.gate.acquire(&cancelled)
            };
            let _lease = match lease {
                Ok(lease) => lease,
                Err(error) => {
                    return Err(if self.is_cancelled() {
                        Error::Cancelled
                    } else {
                        failed(error)
                    });
                }
            };
            self.encode(mel, frames, trace, last)?;
            if decode {
                self.decode(self.last_pass_frames)?;
            }
        }
        if decode
            && self.text.ids().len() != before
            && let Some(on_tokens) = self.on_tokens.as_mut()
            && !on_tokens(&self.text)
        {
            return Err(Error::Cancelled);
        }
        Ok(())
    }

    /// Encodes whole chunks, or when `last` any frames as the end of the
    /// utterance.
    #[allow(clippy::too_many_lines)]
    fn encode(
        &mut self,
        mel: &[f32],
        frames: usize,
        trace: Option<&mut EncoderTrace>,
        last: bool,
    ) -> Result<()> {
        let model = self.model;
        let config = &model.config;
        let w = &model.weights;
        let e = &config.encoder;
        let bins = e.mel_bins;
        let h = e.hidden;
        let ch = e.subsampling_channels;
        let first = self.frames_done == 0;
        let mut chunks = 0;
        if frames == 0 {
            return Err(failed("a pass needs frames"));
        }
        if !last && first {
            if frames < self.first_mel || !(frames - self.first_mel).is_multiple_of(self.chunk_mel)
            {
                return Err(failed("the first pass needs whole chunks"));
            }
            chunks = 1 + (frames - self.first_mel) / self.chunk_mel;
        } else if !last {
            if !frames.is_multiple_of(self.chunk_mel) {
                return Err(failed("a pass needs whole chunks"));
            }
            chunks = frames / self.chunk_mel;
        }
        if frames > self.pass_chunks * self.chunk_mel {
            return Err(failed("too many mel frames for one pass"));
        }
        let begin = Instant::now();
        // Rows of history before each stage's input: the causal padding (two
        // zero rows) at the start, else the input row the previous pass left.
        let pre = if first { 2 } else { 1 };
        // At the end, each stage's input continues with a zero row, the causal
        // convolutions' right padding when they see the whole utterance.
        let tail = usize::from(last);

        let d = self
            .device
            .as_mut()
            .ok_or_else(|| failed("the utterance has no device"))?;
        let s = &d.stream;
        // Stem input: the history rows, then the frames (and the tail).
        let rows_in = pre + frames + tail;
        {
            let mut input = vec![0.0f32; rows_in * bins];
            if !first {
                input[..self.last_mel.len()].copy_from_slice(&self.last_mel);
            }
            input[pre * bins..(pre + frames) * bins].copy_from_slice(&mel[..frames * bins]);
            self.last_mel.clear();
            self.last_mel
                .extend_from_slice(&mel[(frames - 1) * bins..frames * bins]);
            s.upload(&input, d.mel.slice_mut(0..input.len()))?;
        }
        let [f1, f2, f3] = [w.frequencies[1], w.frequencies[2], w.frequencies[3]];
        let t1 = strided(rows_in);
        {
            let begin1 = 2 * f1 * ch;
            k::subsample_input(
                s,
                d.mel.slice(0..rows_in * bins),
                int(bins)?,
                w.stem_w.view(),
                w.stem_b.view(),
                int(ch)?,
                d.s1.slice_mut(begin1..begin1 + t1 * f1 * ch),
                int(t1)?,
                int(f1)?,
            )?;
        }

        // Two depthwise-separable stages, each with its cached input row.
        let stage = StageWeights {
            channels: ch,
            pre,
            tail,
            first,
        };
        let t2 = stage.run(
            s,
            [&w.stage_dw_w[0], &w.stage_dw_b[0]],
            (&w.stage_pw_w[0], &w.stage_pw_b[0]),
            StageBuffers {
                input: &mut d.s1,
                cache: &mut d.s1_cache,
                rows: t1,
                freq: f1,
                out_dw: &mut d.s2_in,
                out: &mut d.s2,
                out_offset_rows: 2,
                out_freq: f2,
            },
        )?;
        let t = stage.run(
            s,
            [&w.stage_dw_w[1], &w.stage_dw_b[1]],
            (&w.stage_pw_w[1], &w.stage_pw_b[1]),
            StageBuffers {
                input: &mut d.s2,
                cache: &mut d.s2_cache,
                rows: t2,
                freq: f2,
                out_dw: &mut d.s3_in,
                out: &mut d.s3,
                out_offset_rows: 0,
                out_freq: f3,
            },
        )?;
        if !last && t != chunks * self.chunk {
            return Err(failed("subsampling produced an unexpected frame count"));
        }
        chunks = t.div_ceil(self.chunk);
        if t == 0 {
            s.synchronize()?;
            self.last_pass_frames = 0;
            self.encoder_ms += since(begin);
            return Ok(());
        }
        wmma_linear_bias(
            s,
            d.s3.slice(0..t * f3 * ch),
            w.sub_linear_w.view(),
            d.x.slice_mut(0..t * h),
            t,
            f3 * ch,
            h,
            w.sub_linear_b.view(),
        )?;
        let mut trace = trace;
        if let Some(trace) = trace.as_deref_mut() {
            trace.subsampled.resize(t * h, 0.0);
            s.download(d.x.slice(0..t * h), &mut trace.subsampled)?;
        }

        let left = e.left_context();
        let cached = left.min(self.frames_done);
        let keep = left.min(cached + t);
        let scale = 1.0f32 / (e.head_dim() as f32).sqrt();
        let inter = e.intermediate;

        // Half-step feed-forward: x += 0.5 · W2 silu(W1 LayerNorm(x)).
        let feed_forward = |d: &mut StreamDevice,
                            norm_w: &DeviceTensor<f32>,
                            norm_b: &DeviceTensor<f32>,
                            up: &DeviceTensor<F16>,
                            down: &DeviceTensor<F16>|
         -> Result<()> {
            let s = &d.stream;
            k::layer_norm_half(
                s,
                d.x.slice(0..t * h),
                norm_w.view(),
                norm_b.view(),
                d.norm_half.slice_mut(0..t * h),
                int(t)?,
                int(h)?,
                false,
            )?;
            wmma_linear(
                s,
                d.norm_half.slice(0..t * h),
                up.view(),
                d.wide.slice_mut(0..t * inter),
                t,
                h,
                inter,
                false,
                1.0,
            )?;
            k::silu_half(
                s,
                d.wide.slice(0..t * inter),
                d.wide_half.slice_mut(0..t * inter),
                t * inter,
            )?;
            wmma_linear(
                s,
                d.wide_half.slice(0..t * inter),
                down.view(),
                d.x.slice_mut(0..t * h),
                t,
                inter,
                h,
                true,
                0.5,
            )
        };

        let kernel = e.conv_kernel;
        for (l, lw) in w.layers.iter().enumerate() {
            let cache_rows = l * left * h..(l + 1) * left * h;
            let conv_rows = l * (kernel - 1) * h..(l + 1) * (kernel - 1) * h;
            feed_forward(d, &lw.ff1_norm_w, &lw.ff1_norm_b, &lw.ff1_up, &lw.ff1_down)?;
            let s = &d.stream;

            // Self-attention over the cached and the new frames.
            k::layer_norm_half(
                s,
                d.x.slice(0..t * h),
                lw.att_norm_w.view(),
                lw.att_norm_b.view(),
                d.norm_half.slice_mut(0..t * h),
                int(t)?,
                int(h)?,
                false,
            )?;
            wmma_linear(
                s,
                d.norm_half.slice(0..t * h),
                lw.qkv.view(),
                d.wide.slice_mut(0..t * 3 * h),
                t,
                h,
                3 * h,
                false,
                1.0,
            )?;
            let attention = k::rel_pos_attention_half(
                s,
                d.wide.slice(0..t * 3 * h),
                d.key_cache.slice(cache_rows.clone()),
                d.value_cache.slice(cache_rows.clone()),
                lw.pos.view(),
                lw.bias_u.view(),
                lw.bias_v.view(),
                d.attention_half.slice_mut(0..t * h),
                int(t)?,
                int(e.heads)?,
                int(3 * h)?,
                self.frames_done,
                self.frames_done - cached,
                int(self.chunk)?,
                int(left)?,
                int(w.pos_zero)?,
                int(w.pos_rows)?,
                scale,
            )?;
            if attention == Launch::Unsupported {
                return Err(failed("attention rows must be 16-byte aligned"));
            }
            k::update_key_value_cache(
                s,
                d.key_cache.slice_mut(cache_rows.clone()),
                d.value_cache.slice_mut(cache_rows),
                d.wide.slice(0..t * 3 * h),
                int(3 * h)?,
                int(cached)?,
                int(t)?,
                int(keep)?,
                int(h)?,
                int(left)?,
            )?;
            wmma_linear(
                s,
                d.attention_half.slice(0..t * h),
                lw.out.view(),
                d.x.slice_mut(0..t * h),
                t,
                h,
                h,
                true,
                1.0,
            )?;

            // Convolution module: GLU, causal depthwise convolution,
            // LayerNorm, SiLU.
            k::layer_norm_half(
                s,
                d.x.slice(0..t * h),
                lw.conv_norm_w.view(),
                lw.conv_norm_b.view(),
                d.norm_half.slice_mut(0..t * h),
                int(t)?,
                int(h)?,
                false,
            )?;
            wmma_linear(
                s,
                d.norm_half.slice(0..t * h),
                lw.pointwise1.view(),
                d.wide.slice_mut(0..t * 2 * h),
                t,
                h,
                2 * h,
                false,
                1.0,
            )?;
            k::glu_depthwise(
                s,
                d.wide.slice(0..t * 2 * h),
                d.conv_cache.slice(conv_rows.clone()),
                lw.depthwise.view(),
                d.mid.slice_mut(0..t * h),
                int(t)?,
                int(h)?,
                int(kernel)?,
            )?;
            k::update_conv_cache(
                s,
                d.wide.slice(0..t * 2 * h),
                d.conv_cache.slice_mut(conv_rows),
                int(t)?,
                int(h)?,
                int(kernel)?,
            )?;
            k::layer_norm_half(
                s,
                d.mid.slice(0..t * h),
                lw.conv_inner_norm_w.view(),
                lw.conv_inner_norm_b.view(),
                d.norm_half.slice_mut(0..t * h),
                int(t)?,
                int(h)?,
                true,
            )?;
            wmma_linear(
                s,
                d.norm_half.slice(0..t * h),
                lw.pointwise2.view(),
                d.x.slice_mut(0..t * h),
                t,
                h,
                h,
                true,
                1.0,
            )?;

            feed_forward(d, &lw.ff2_norm_w, &lw.ff2_norm_b, &lw.ff2_up, &lw.ff2_down)?;
            let s = &d.stream;
            k::layer_norm_in_place(
                s,
                d.x.slice_mut(0..t * h),
                lw.out_norm_w.view(),
                lw.out_norm_b.view(),
                int(t)?,
                int(h)?,
            )?;
        }
        let s = &d.stream;
        if let Some(trace) = trace.as_deref_mut() {
            trace.hidden.resize(t * h, 0.0);
            s.download(d.x.slice(0..t * h), &mut trace.hidden)?;
        }

        // Language prompt and projection to the joint's width.
        let ph = config.prompt_hidden;
        let width = config.decoder_hidden;
        k::to_half(
            s,
            d.x.slice(0..t * h),
            d.norm_half.slice_mut(0..t * h),
            t * h,
        )?;
        wmma_linear_half(
            s,
            d.norm_half.slice(0..t * h),
            w.prompt_w1.view(),
            d.wide_half.slice_mut(0..t * ph),
            t,
            h,
            ph,
            w.prompt_bias
                .slice(self.prompt * ph..(self.prompt + 1) * ph),
            true,
        )?;
        wmma_linear_half(
            s,
            d.wide_half.slice(0..t * ph),
            w.prompt_w2.view(),
            d.norm_half.slice_mut(0..t * h),
            t,
            ph,
            h,
            w.prompt_b2.view(),
            false,
        )?;
        wmma_linear_bias(
            s,
            d.norm_half.slice(0..t * h),
            w.proj_w.view(),
            d.projected.slice_mut(0..t * width),
            t,
            h,
            width,
            w.proj_b.view(),
        )?;
        if let Some(trace) = trace {
            trace.projected.resize(t * width, 0.0);
            s.download(d.projected.slice(0..t * width), &mut trace.projected)?;
        }
        s.synchronize()?;
        self.frames_done += t;
        self.chunks += chunks;
        self.last_pass_frames = t;
        self.encoder_ms += since(begin);
        Ok(())
    }

    /// One step of the prediction network (`DecoderStep`).
    fn decoder_step(&mut self, token: u32) -> Result<()> {
        let config = &self.model.config;
        let w = &self.model.weights;
        let width = config.decoder_hidden;
        let d = self
            .device
            .as_mut()
            .ok_or_else(|| failed("the utterance has no device"))?;
        let token = token as usize;
        if token >= config.vocab {
            return Err(failed("token outside the vocabulary"));
        }
        for l in 0..config.decoder_layers {
            let s = &d.stream;
            {
                let embedding;
                let x = if l == 0 {
                    embedding = w.embedding.slice(token * width..(token + 1) * width);
                    embedding
                } else {
                    d.h.slice((l - 1) * width..l * width)
                };
                k::gemv(
                    s,
                    w.lstm_w[l].view(),
                    x,
                    int(width)?,
                    d.h.slice(l * width..(l + 1) * width),
                    int(width)?,
                    Some(w.lstm_b[l].view()),
                    d.gates.slice_mut(0..4 * width),
                    int(4 * width)?,
                )?;
            }
            k::lstm_cell(
                s,
                d.gates.slice(0..4 * width),
                d.h.slice_mut(l * width..(l + 1) * width),
                d.c.slice_mut(l * width..(l + 1) * width),
                int(width)?,
            )?;
        }
        let last = config.decoder_layers - 1;
        let x = d.h.slice(last * width..(last + 1) * width);
        k::gemv(
            &d.stream,
            w.dec_proj_w.view(),
            x,
            int(width)?,
            d.h.slice(0..0),
            0,
            Some(w.dec_proj_b.view()),
            d.decoder_out.slice_mut(0..width),
            int(width)?,
        )?;
        Ok(())
    }

    /// Packed arg maxes of the joint for `frames` frames of the last pass
    /// from `frame` on (`Joint`).
    fn joint(&mut self, frame: usize, frames: usize) -> Result<&[u64]> {
        let config = &self.model.config;
        let w = &self.model.weights;
        let width = config.decoder_hidden;
        let vocab = config.vocab;
        let d = self
            .device
            .as_mut()
            .ok_or_else(|| failed("the utterance has no device"))?;
        let s = &d.stream;
        k::joint_input(
            s,
            d.projected.slice(frame * width..(frame + frames) * width),
            d.decoder_out.slice(0..width),
            d.joint_input.slice_mut(0..frames * width),
            int(frames)?,
            int(width)?,
        )?;
        wmma_linear(
            s,
            d.joint_input.slice(0..frames * width),
            w.head_w.view(),
            d.logits.slice_mut(0..frames * vocab),
            frames,
            width,
            vocab,
            false,
            1.0,
        )?;
        k::argmax_rows(
            s,
            d.logits.slice(0..frames * vocab),
            w.head_b.view(),
            int(frames)?,
            int(vocab)?,
            d.best.slice_mut(0..frames),
        )?;
        s.download_blocking(d.best.slice(0..frames), &mut d.best_host[..frames])?;
        Ok(&d.best_host[..frames])
    }

    /// Greedy RNN-T (`Decode`): at each frame emit the arg max until it is
    /// blank or `max_symbols` tokens came out, then move on. The joint
    /// scores a window of frames ahead against the current decoder state;
    /// the frames before the first non-blank one are blank as they would
    /// be one at a time, and a token changes the state, so the next window
    /// starts at its frame.
    fn decode(&mut self, frames: usize) -> Result<()> {
        let begin = Instant::now();
        let blank = self.model.config.blank as u32;
        let max_symbols = self.model.config.max_symbols;
        let mut frame = 0;
        let mut symbols = 0; // emitted at `frame`
        while frame < frames {
            self.check_cancelled()?;
            let count = MAX_JOINT_FRAMES.min(frames - frame);
            let best = self.joint(frame, count)?;
            let first = best
                .iter()
                .take_while(|&&packed| argmax_token(packed) == blank)
                .count();
            if first == count {
                frame += count;
                symbols = 0;
                continue;
            }
            let token = argmax_token(best[first]);
            if first > 0 {
                frame += first;
                symbols = 0;
            }
            self.text.push(token);
            self.decoder_step(token)?;
            symbols += 1;
            if symbols == max_symbols {
                frame += 1;
                symbols = 0;
            }
        }
        self.decoder_ms += since(begin);
        Ok(())
    }

    /// The prediction network's output for the current state (for tests).
    pub fn decoder_output(&self) -> Result<Vec<f32>> {
        let d = self.device()?;
        let mut out = vec![0.0f32; self.model.config.decoder_hidden];
        d.stream
            .download(d.decoder_out.slice(0..out.len()), &mut out)?;
        Ok(out)
    }

    /// The joint's logits for encoder frame `frame` of the last pass (for
    /// tests).
    pub fn joint_logits(&mut self, frame: usize) -> Result<Vec<f32>> {
        let config = &self.model.config;
        let w = &self.model.weights;
        let width = config.decoder_hidden;
        let vocab = config.vocab;
        if frame >= self.last_pass_frames {
            return Err(failed("frame outside the last pass"));
        }
        let d = self
            .device
            .as_mut()
            .ok_or_else(|| failed("the utterance has no device"))?;
        let s = &d.stream;
        k::joint_input(
            s,
            d.projected.slice(frame * width..(frame + 1) * width),
            d.decoder_out.slice(0..width),
            d.joint_input.slice_mut(0..width),
            1,
            int(width)?,
        )?;
        wmma_linear_bias(
            s,
            d.joint_input.slice(0..width),
            w.head_w.view(),
            d.logits.slice_mut(0..vocab),
            1,
            width,
            vocab,
            w.head_b.view(),
        )?;
        let mut out = vec![0.0f32; vocab];
        s.download(d.logits.slice(0..vocab), &mut out)?;
        Ok(out)
    }
}

impl Drop for Utterance<'_> {
    fn drop(&mut self) {
        if let Some(device) = self.device.take() {
            self.model.release_device(device);
        }
    }
}

/// The parts of a subsampling stage that both stages share.
struct StageWeights {
    channels: usize,
    pre: usize,
    tail: usize,
    first: bool,
}

/// One stage's buffers: rows from offset 2 of `input` are this pass's
/// input; `out` receives the pointwise output at `out_offset_rows`.
struct StageBuffers<'a> {
    input: &'a mut DeviceBuffer<F16>,
    cache: &'a mut DeviceBuffer<F16>,
    rows: usize,
    freq: usize,
    out_dw: &'a mut DeviceBuffer<F16>,
    out: &'a mut DeviceBuffer<F16>,
    out_offset_rows: usize,
    out_freq: usize,
}

impl StageWeights {
    /// A depthwise-separable stage with its cached input row; returns its
    /// output rows.
    fn run(
        &self,
        s: &Stream,
        [dw_w, dw_b]: [&DeviceTensor<f32>; 2],
        (pw_w, pw_b): (&DeviceTensor<F16>, &DeviceTensor<f32>),
        b: StageBuffers<'_>,
    ) -> Result<usize> {
        let ch = self.channels;
        let row = b.freq * ch;
        if self.first {
            s.fill_bytes(b.input.slice_mut(0..2 * row), 0)?;
        } else {
            s.copy(b.cache.slice(0..row), b.input.slice_mut(row..2 * row))?;
        }
        if self.tail != 0 {
            s.fill_bytes(b.input.slice_mut((2 + b.rows) * row..(3 + b.rows) * row), 0)?;
        }
        let start = (2 - self.pre) * row;
        let rows_out = strided(self.pre + b.rows + self.tail);
        k::subsample_depthwise(
            s,
            b.input
                .slice(start..start + (self.pre + b.rows + self.tail) * row),
            int(b.freq)?,
            dw_w.view(),
            dw_b.view(),
            int(ch)?,
            b.out_dw.slice_mut(0..rows_out * b.out_freq * ch),
            int(rows_out)?,
            int(b.out_freq)?,
        )?;
        s.copy(
            b.input.slice((2 + b.rows - 1) * row..(2 + b.rows) * row),
            b.cache.slice_mut(0..row),
        )?;
        let offset = b.out_offset_rows * b.out_freq * ch;
        let count = rows_out * b.out_freq * ch;
        if rows_out != 0 {
            wmma_linear_half(
                s,
                b.out_dw.slice(0..count),
                pw_w.view(),
                b.out.slice_mut(offset..offset + count),
                rows_out * b.out_freq,
                ch,
                ch,
                pw_b.view(),
                true,
            )?;
        }
        Ok(rows_out)
    }
}
