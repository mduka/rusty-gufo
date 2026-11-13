//! AudioVAE V2 (`audiovae.pth`) on the GPU in float32 (`vae.hpp`,
//! `vae.cpp`): the causal decoder from latents to `out_sample_rate` audio,
//! run chunk by chunk with the state of every causal convolution carried
//! between chunks (the reference's StreamingVAEDecoder, which equals
//! decoding at once), and the causal encoder from `sample_rate` audio to
//! latents for cloning. Weight normalization is folded into the weights at
//! load time.
//!
//! Ported from OpenBMB/VoxCPM's AudioVAE V2
//! (src/voxcpm/modules/audiovae/audio_vae_v2.py; Apache-2.0, Copyright
//! OpenBMB), whose convolution blocks and Snake activation follow Descript's
//! DAC (MIT, Copyright Descript): time-major float32, streaming convolution
//! state, weight norm folded at load.

use gufo_hip::blas::{Blas, Op};
use gufo_hip::{
    DeviceBuffer, DeviceCopy, DeviceSlice, DeviceSliceMut, Event, F16, HipError, LaunchError,
    PinnedBuffer, Stream,
};
use gufo_voxcpm_kernels as k;

use crate::config::{AudioVaeConfig, Config};
use crate::model_files::{ModelFiles, same_shape};
use crate::safetensors::DType;

/// Residual units and the outer convolutions.
const KERNEL: usize = 7;
const DILATIONS: [usize; 3] = [1, 3, 9];

/// An offset into the packed float32 weights.
type Offset = usize;

#[derive(Clone, Copy, Default)]
struct ResidualUnit {
    dilation: usize,
    alpha_in: Offset,
    /// `[channels][7]`
    depthwise: Offset,
    depthwise_bias: Offset,
    alpha_out: Offset,
    /// `[channels][channels]`
    pointwise: Offset,
    pointwise_bias: Offset,
}

struct DecoderBlock {
    input: usize,
    output: usize,
    stride: usize,
    /// Rate conditioning of the block's input.
    scale: Offset,
    bias: Offset,
    alpha: Offset,
    // ConvTranspose as two products: out[t][(r, c)] =
    // A[(r, c)] x[t] + B[(r, c)] x[t - 1] + bias[c].
    /// `[stride * out][in]`
    up_current: Offset,
    /// `[stride * out][in]`
    up_previous: Offset,
    /// `[stride * out]`
    up_bias: Offset,
    units: [ResidualUnit; 3],
}

struct EncoderBlock {
    input: usize,
    output: usize,
    stride: usize,
    units: [ResidualUnit; 3],
    alpha: Offset,
    /// `[2 * stride][out][in]`, one matrix per tap.
    down: Offset,
    down_bias: Offset,
}

/// `AudioVae::Weights`.
struct Weights {
    /// `[latent][7]`
    in_depthwise: Offset,
    in_depthwise_bias: Offset,
    /// `[decoder_dim][latent]`
    in_pointwise: Offset,
    in_pointwise_bias: Offset,
    blocks: Vec<DecoderBlock>,
    out_alpha: Offset,
    /// `[channels][7]`
    out_conv: Offset,
    out_bias: f32,
    out_channels: usize,

    /// `[encoder_dim][7]`
    encoder_in: Offset,
    encoder_in_bias: Offset,
    encoder_blocks: Vec<EncoderBlock>,
    /// `[3][latent][channels]`
    mu: Offset,
    mu_bias: Offset,
    mu_channels: usize,
}

/// Reads the checkpoint, folds weight normalization and packs the float32
/// weights in the layouts the products use (`Packer`).
struct Packer<'a> {
    files: &'a ModelFiles,
    data: Vec<f32>,
}

impl Packer<'_> {
    /// The tensor `name` as float32 (stored as F32, F16 or BF16).
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let tensor = self
            .files
            .vae(name)
            .ok_or_else(|| format!("the AudioVAE weights lack {name}"))?;
        if !same_shape(&tensor.shape, shape) {
            return Err(format!("unexpected shape of the AudioVAE tensor {name}"));
        }
        let count = tensor.elements();
        let data = tensor.data();
        match tensor.dtype {
            DType::F32 => Ok(data[..count * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|raw| f32::from_le_bytes(*raw))
                .collect()),
            DType::F16 => Ok(data[..count * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|raw| F16(u16::from_le_bytes(*raw)).to_f32())
                .collect()),
            DType::Bf16 => Ok(data[..count * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|raw| f32::from_bits(u32::from(u16::from_le_bytes(*raw)) << 16))
                .collect()),
            _ => Err(format!("unsupported type of the AudioVAE tensor {name}")),
        }
    }

    fn raw(&mut self, name: &str, shape: &[usize]) -> Result<Offset, String> {
        let values = self.get(name, shape)?;
        Ok(self.append(&values))
    }

    /// `weight = g * v / |v|`, the norm over all but dimension 0 (torch
    /// weight_norm with dim=0), in the checkpoint's layout.
    fn normalized(&self, prefix: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let v = self.get(&format!("{prefix}.weight_v"), shape)?;
        let g = self.get(&format!("{prefix}.weight_g"), &[shape[0], 1, 1])?;
        let rows = shape[0];
        let width = v.len() / rows;
        let mut weight = vec![0.0f32; v.len()];
        for row in 0..rows {
            let mut norm = 0.0f64;
            for &value in &v[row * width..(row + 1) * width] {
                let value = f64::from(value);
                norm += value * value;
            }
            let factor = f64::from(g[row]) / norm.sqrt();
            for i in 0..width {
                weight[row * width + i] = (f64::from(v[row * width + i]) * factor) as f32;
            }
        }
        Ok(weight)
    }

    fn append(&mut self, values: &[f32]) -> Offset {
        let offset = self.data.len();
        self.data.extend_from_slice(values);
        offset
    }

    /// A Conv1d `[out][in][k]` as one `[out][in]` matrix per tap:
    /// `[k][out][in]`.
    fn per_tap(
        &mut self,
        prefix: &str,
        output: usize,
        input: usize,
        kernel: usize,
    ) -> Result<Offset, String> {
        let w = self.normalized(prefix, &[output, input, kernel])?;
        let mut packed = vec![0.0f32; w.len()];
        for o in 0..output {
            for i in 0..input {
                for k in 0..kernel {
                    packed[(k * output + o) * input + i] = w[(o * input + i) * kernel + k];
                }
            }
        }
        Ok(self.append(&packed))
    }

    fn unit(
        &mut self,
        prefix: &str,
        channels: usize,
        dilation: usize,
    ) -> Result<ResidualUnit, String> {
        let alpha_in = self.raw(&format!("{prefix}.block.0.alpha"), &[1, channels, 1])?;
        let depthwise = self.normalized(&format!("{prefix}.block.1"), &[channels, 1, KERNEL])?;
        let depthwise = self.append(&depthwise);
        let depthwise_bias = self.raw(&format!("{prefix}.block.1.bias"), &[channels])?;
        let alpha_out = self.raw(&format!("{prefix}.block.2.alpha"), &[1, channels, 1])?;
        let pointwise = self.normalized(&format!("{prefix}.block.3"), &[channels, channels, 1])?;
        let pointwise = self.append(&pointwise);
        let pointwise_bias = self.raw(&format!("{prefix}.block.3.bias"), &[channels])?;
        Ok(ResidualUnit {
            dilation,
            alpha_in,
            depthwise,
            depthwise_bias,
            alpha_out,
            pointwise,
            pointwise_bias,
        })
    }
}

/// Packs every AudioVAE weight (`AudioVae::Create`, before the upload).
fn pack(config: &AudioVaeConfig, files: &ModelFiles) -> Result<(Weights, Vec<f32>), String> {
    let c = config;
    let mut pack = Packer {
        files,
        data: Vec::new(),
    };
    let latent = c.latent_dim;
    let in_depthwise = pack.normalized("decoder.model.0", &[latent, 1, KERNEL])?;
    let in_depthwise = pack.append(&in_depthwise);
    let in_depthwise_bias = pack.raw("decoder.model.0.bias", &[latent])?;
    let in_pointwise = pack.normalized("decoder.model.1", &[c.decoder_dim, latent, 1])?;
    let in_pointwise = pack.append(&in_pointwise);
    let in_pointwise_bias = pack.raw("decoder.model.1.bias", &[c.decoder_dim])?;
    let buckets = c.sr_bin_boundaries.len() + 1;
    let bucket = c.rate_bucket();
    let mut channels = c.decoder_dim;
    let mut blocks = Vec::with_capacity(c.decoder_rates.len());
    for (index, &stride) in c.decoder_rates.iter().enumerate() {
        let input = channels;
        let output = channels / 2;
        let layer = format!("decoder.model.{}", index + 2);
        let cond = format!("decoder.sr_cond_model.{}", index + 2);
        let scale = pack.get(&format!("{cond}.scale_embed.weight"), &[buckets, input])?;
        let bias = pack.get(&format!("{cond}.bias_embed.weight"), &[buckets, input])?;
        let scale = pack.append(&scale[bucket * input..(bucket + 1) * input]);
        let bias = pack.append(&bias[bucket * input..(bucket + 1) * input]);
        let alpha = pack.raw(&format!("{layer}.block.0.alpha"), &[1, input, 1])?;
        let s = stride;
        // ConvTranspose1d weight [in][out][2s]; weight_norm's dim 0 is `in`.
        let up = pack.normalized(&format!("{layer}.block.1"), &[input, output, 2 * s])?;
        let mut current = vec![0.0f32; s * output * input];
        let mut previous = vec![0.0f32; current.len()];
        for i in 0..input {
            for o in 0..output {
                for r in 0..s {
                    let taps = &up[(i * output + o) * 2 * s..];
                    current[(r * output + o) * input + i] = taps[r];
                    previous[(r * output + o) * input + i] = taps[r + s];
                }
            }
        }
        let up_current = pack.append(&current);
        let up_previous = pack.append(&previous);
        let up_bias = pack.get(&format!("{layer}.block.1.bias"), &[output])?;
        let mut repeated = Vec::with_capacity(s * output);
        for _ in 0..s {
            repeated.extend_from_slice(&up_bias);
        }
        let up_bias = pack.append(&repeated);
        let mut units = [ResidualUnit::default(); 3];
        for (u, unit) in units.iter_mut().enumerate() {
            *unit = pack.unit(&format!("{layer}.block.{}", u + 2), output, DILATIONS[u])?;
        }
        blocks.push(DecoderBlock {
            input,
            output,
            stride,
            scale,
            bias,
            alpha,
            up_current,
            up_previous,
            up_bias,
            units,
        });
        channels = output;
    }
    let last = c.decoder_rates.len() + 2;
    let out_channels = channels;
    let out_alpha = pack.raw(&format!("decoder.model.{last}.alpha"), &[1, channels, 1])?;
    let out_conv_name = format!("decoder.model.{}", last + 1);
    let out_conv = pack.normalized(&out_conv_name, &[1, channels, KERNEL])?;
    let out_conv = pack.append(&out_conv);
    let out_bias = pack.get(&format!("{out_conv_name}.bias"), &[1])?[0];

    let encoder_in = pack.normalized("encoder.block.0", &[c.encoder_dim, 1, KERNEL])?;
    let encoder_in = pack.append(&encoder_in);
    let encoder_in_bias = pack.raw("encoder.block.0.bias", &[c.encoder_dim])?;
    channels = c.encoder_dim;
    let mut encoder_blocks = Vec::with_capacity(c.encoder_rates.len());
    for (index, &stride) in c.encoder_rates.iter().enumerate() {
        let input = channels;
        let output = 2 * channels;
        let layer = format!("encoder.block.{}", index + 1);
        let mut units = [ResidualUnit::default(); 3];
        for (u, unit) in units.iter_mut().enumerate() {
            *unit = pack.unit(&format!("{layer}.block.{u}"), input, DILATIONS[u])?;
        }
        let alpha = pack.raw(&format!("{layer}.block.3.alpha"), &[1, input, 1])?;
        let down = pack.per_tap(&format!("{layer}.block.4"), output, input, 2 * stride)?;
        let down_bias = pack.raw(&format!("{layer}.block.4.bias"), &[output])?;
        encoder_blocks.push(EncoderBlock {
            input,
            output,
            stride,
            units,
            alpha,
            down,
            down_bias,
        });
        channels = output;
    }
    let mu_channels = channels;
    let mu = pack.per_tap("encoder.fc_mu", latent, channels, 3)?;
    let mu_bias = pack.raw("encoder.fc_mu.bias", &[latent])?;
    let weights = Weights {
        in_depthwise,
        in_depthwise_bias,
        in_pointwise,
        in_pointwise_bias,
        blocks,
        out_alpha,
        out_conv,
        out_bias,
        out_channels,
        encoder_in,
        encoder_in_bias,
        encoder_blocks,
        mu,
        mu_bias,
        mu_channels,
    };
    Ok((weights, pack.data))
}

/// Reallocates `buffer` when `count` exceeds its capacity; contents are
/// lost (`DeviceBuffer::Reserve`).
fn reserve<T: DeviceCopy>(
    stream: &Stream,
    buffer: &mut DeviceBuffer<T>,
    count: usize,
) -> Result<(), HipError> {
    if count > buffer.len() {
        *buffer = stream.alloc(count.max(1))?;
    }
    Ok(())
}

fn launch(error: LaunchError) -> String {
    error.to_string()
}

fn hip(error: HipError) -> String {
    error.to_string()
}

/// AudioVAE V2 on the GPU in float32 on its own stream (`AudioVae`).
/// `Send`, not `Sync`.
pub struct AudioVae {
    config: AudioVaeConfig,
    weights: Weights,
    storage: DeviceBuffer<f32>,
    /// Streaming state of each causal convolution, in decoding order.
    states: Vec<DeviceBuffer<f32>>,
    latents: DeviceBuffer<f32>,
    work: DeviceBuffer<f32>,
    hidden: DeviceBuffer<f32>,
    scratch: DeviceBuffer<f32>,
    samples: DeviceBuffer<f32>,
    pinned: PinnedBuffer<f32>,
    pending: usize,
    blas: Blas,
    stream: Stream,
}

impl AudioVae {
    /// Packs and uploads the AudioVAE weights of `files` (`Create`).
    pub fn create(config: &Config, files: &ModelFiles) -> Result<Self, String> {
        let (weights, data) = pack(&config.vae, files)?;
        let stream = Stream::new().map_err(hip)?;
        let mut storage = stream.alloc(data.len().max(1)).map_err(hip)?;
        stream
            .upload(&data, storage.slice_mut(0..data.len()))
            .map_err(hip)?;
        // The C++ handle keeps hipBLAS's default atomics mode.
        let blas = Blas::with_default_atomics(&stream).map_err(hip)?;
        // Streaming state, in decoding order: the input convolution, then
        // per block the transposed convolution's last frame and the residual
        // units' history, then the output convolution.
        let latent = config.vae.latent_dim;
        let mut sizes = vec![(KERNEL - 1) * latent];
        for block in &weights.blocks {
            sizes.push(block.input);
            for unit in &block.units {
                sizes.push((KERNEL - 1) * unit.dilation * block.output);
            }
        }
        sizes.push((KERNEL - 1) * weights.out_channels);
        let states = sizes
            .into_iter()
            .map(|size| stream.alloc(size.max(1)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(hip)?;
        let mut vae = Self {
            config: config.vae.clone(),
            weights,
            storage,
            states,
            latents: stream.alloc(1).map_err(hip)?,
            work: stream.alloc(1).map_err(hip)?,
            hidden: stream.alloc(1).map_err(hip)?,
            scratch: stream.alloc(1).map_err(hip)?,
            samples: stream.alloc(1).map_err(hip)?,
            pinned: PinnedBuffer::zeroed(1).map_err(hip)?,
            pending: 0,
            blas,
            stream,
        };
        vae.reset()?;
        Ok(vae)
    }

    /// The output sample rate.
    pub fn output_rate(&self) -> usize {
        self.config.out_sample_rate
    }

    /// The AudioVAE's stream.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Starts a new decoded stream (zero history).
    pub fn reset(&mut self) -> Result<(), String> {
        for state in &mut self.states {
            self.stream
                .fill_bytes(state.as_mut_slice(), 0)
                .map_err(hip)?;
        }
        self.stream.synchronize().map_err(hip)
    }

    /// Decodes `latents.len() / latent_dim` latent frames
    /// (`[frames][latent_dim]`, time-major, on the host) and appends
    /// `frames * decode_hop` samples to `audio`.
    pub fn decode(&mut self, latents: &[f32], audio: &mut Vec<f32>) -> Result<(), String> {
        let latent = self.config.latent_dim;
        let frames = latents.len() / latent;
        if frames == 0 {
            return Ok(());
        }
        reserve(&self.stream, &mut self.latents, frames * latent).map_err(hip)?;
        self.stream
            .upload(
                &latents[..frames * latent],
                self.latents.slice_mut(0..frames * latent),
            )
            .map_err(hip)?;
        self.run(frames)?;
        self.collect(audio)
    }

    /// Queues the decoding of `frames` device latents (of another stream)
    /// once `ready` has completed, on the AudioVAE's own stream; the latents
    /// are copied before anything else. [`AudioVae::collect`] appends the
    /// samples; one decoding may be pending at a time (`Enqueue`). The copy
    /// across streams waits for itself, where the C++ code queues it.
    pub fn enqueue(
        &mut self,
        latents: DeviceSlice<'_, f32>,
        frames: usize,
        ready: Option<&Event>,
    ) -> Result<(), String> {
        if frames == 0 {
            return Ok(());
        }
        if let Some(ready) = ready {
            self.stream.wait_event(ready).map_err(hip)?;
        }
        let count = frames * self.config.latent_dim;
        reserve(&self.stream, &mut self.latents, count).map_err(hip)?;
        self.stream
            .copy_across(latents.range(0..count), self.latents.slice_mut(0..count))
            .map_err(hip)?;
        self.run(frames)
    }

    /// Waits for the pending decoding and appends its samples (`Collect`).
    pub fn collect(&mut self, audio: &mut Vec<f32>) -> Result<(), String> {
        self.stream.synchronize().map_err(hip)?;
        let samples = self.pinned.host().map_err(hip)?;
        audio.extend_from_slice(&samples[..self.pending]);
        self.pending = 0;
        Ok(())
    }

    /// `y[rows][out] (+)= x[rows][in] (row stride x_stride) W^T` through
    /// `hipblasSgemm` (`Gemm`).
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        blas: &Blas,
        stream: &Stream,
        x: DeviceSlice<'_, f32>,
        x_stride: usize,
        w: DeviceSlice<'_, f32>,
        y: DeviceSliceMut<'_, f32>,
        rows: usize,
        input: usize,
        output: usize,
        accumulate: bool,
    ) -> Result<(), String> {
        blas.sgemm_beta(
            stream,
            Op::T,
            Op::N,
            output,
            rows,
            input,
            1.0,
            w,
            input,
            x,
            x_stride,
            if accumulate { 1.0 } else { 0.0 },
            y,
            output,
        )
        .map_err(launch)
    }

    /// Decodes `latents` (`[frames][latent_dim]`) into the pinned buffer
    /// (`Run`).
    fn run(&mut self, frames: usize) -> Result<(), String> {
        let w = &self.weights;
        let latent = self.config.latent_dim;
        let decoder_dim = self.config.decoder_dim;
        // The widest stage: frames * (product of strides so far) * channels.
        let mut widest = frames * latent.max(decoder_dim);
        let mut length = frames;
        for block in &w.blocks {
            length *= block.stride;
            widest = widest.max(length * block.output);
        }
        let history = (KERNEL - 1) * DILATIONS[2];
        let s = &self.stream;
        reserve(s, &mut self.hidden, widest).map_err(hip)?;
        reserve(s, &mut self.scratch, widest).map_err(hip)?;
        reserve(s, &mut self.work, widest + history * decoder_dim).map_err(hip)?;
        reserve(s, &mut self.samples, length).map_err(hip)?;
        if self.pinned.len() < length {
            self.pinned = PinnedBuffer::zeroed(length).map_err(hip)?;
        }
        let storage = &self.storage;
        // A weight slice from `offset` to the end of the storage (the
        // launchers check their extent within it).
        let at = |offset: Offset| storage.slice(offset..storage.len());
        let work = &mut self.work;
        let hidden = &mut self.hidden;
        let scratch = &mut self.scratch;
        let states = &mut self.states;
        let blas = &self.blas;
        let mut state = 0;
        // Copies the state in front of `work`, and afterwards keeps the last
        // `rows` rows of state ++ input as the new state.
        let load = |work: &mut DeviceBuffer<f32>,
                    states: &[DeviceBuffer<f32>],
                    state: usize,
                    rows: usize,
                    width: usize|
         -> Result<(), String> {
            s.copy(
                states[state].slice(0..rows * width),
                work.slice_mut(0..rows * width),
            )
            .map_err(hip)
        };
        let keep = |work: &DeviceBuffer<f32>,
                    states: &mut [DeviceBuffer<f32>],
                    state: &mut usize,
                    rows: usize,
                    width: usize,
                    frames_in: usize|
         -> Result<(), String> {
            let begin = frames_in * width;
            s.copy(
                work.slice(begin..begin + rows * width),
                states[*state].slice_mut(0..rows * width),
            )
            .map_err(hip)?;
            *state += 1;
            Ok(())
        };

        load(work, states, state, KERNEL - 1, latent)?;
        s.copy(
            self.latents.slice(0..frames * latent),
            work.slice_mut((KERNEL - 1) * latent..(KERNEL - 1 + frames) * latent),
        )
        .map_err(hip)?;
        k::depthwise_conv(
            s,
            work.slice(0..(KERNEL - 1 + frames) * latent),
            at(w.in_depthwise),
            at(w.in_depthwise_bias),
            scratch.slice_mut(0..frames * latent),
            frames,
            latent,
            KERNEL,
            1,
        )
        .map_err(launch)?;
        keep(work, states, &mut state, KERNEL - 1, latent, frames)?;
        Self::gemm(
            blas,
            s,
            scratch.slice(0..frames * latent),
            latent,
            at(w.in_pointwise),
            hidden.slice_mut(0..frames * decoder_dim),
            frames,
            latent,
            decoder_dim,
            false,
        )?;
        k::add_bias_f32(
            s,
            hidden.slice_mut(0..frames * decoder_dim),
            at(w.in_pointwise_bias),
            frames,
            decoder_dim,
        )
        .map_err(launch)?;

        let mut t = frames;
        for block in &w.blocks {
            // Snake of the rate-conditioned input, after one frame of
            // history.
            let input = block.input;
            load(work, states, state, 1, input)?;
            k::snake_affine(
                s,
                hidden.slice(0..t * input),
                work.slice_mut(input..(1 + t) * input),
                t,
                input,
                at(block.alpha),
                at(block.scale),
                at(block.bias),
            )
            .map_err(launch)?;
            keep(work, states, &mut state, 1, input, t)?;
            let width = block.stride * block.output;
            Self::gemm(
                blas,
                s,
                work.slice(input..(1 + t) * input),
                input,
                at(block.up_current),
                hidden.slice_mut(0..t * width),
                t,
                input,
                width,
                false,
            )?;
            Self::gemm(
                blas,
                s,
                work.slice(0..t * input),
                input,
                at(block.up_previous),
                hidden.slice_mut(0..t * width),
                t,
                input,
                width,
                true,
            )?;
            k::add_bias_f32(
                s,
                hidden.slice_mut(0..t * width),
                at(block.up_bias),
                t,
                width,
            )
            .map_err(launch)?;
            t *= block.stride;
            let c = block.output;
            for unit in &block.units {
                let rows = (KERNEL - 1) * unit.dilation;
                load(work, states, state, rows, c)?;
                k::snake(
                    s,
                    hidden.slice(0..t * c),
                    work.slice_mut(rows * c..(rows + t) * c),
                    t,
                    c,
                    at(unit.alpha_in),
                )
                .map_err(launch)?;
                keep(work, states, &mut state, rows, c, t)?;
                k::depthwise_conv(
                    s,
                    work.slice(0..(rows + t) * c),
                    at(unit.depthwise),
                    at(unit.depthwise_bias),
                    scratch.slice_mut(0..t * c),
                    t,
                    c,
                    KERNEL,
                    unit.dilation,
                )
                .map_err(launch)?;
                k::snake_in_place(s, scratch.slice_mut(0..t * c), t, c, at(unit.alpha_out))
                    .map_err(launch)?;
                Self::gemm(
                    blas,
                    s,
                    scratch.slice(0..t * c),
                    c,
                    at(unit.pointwise),
                    hidden.slice_mut(0..t * c),
                    t,
                    c,
                    c,
                    true,
                )?;
                k::add_bias_f32(s, hidden.slice_mut(0..t * c), at(unit.pointwise_bias), t, c)
                    .map_err(launch)?;
            }
        }
        let c = w.out_channels;
        load(work, states, state, KERNEL - 1, c)?;
        k::snake(
            s,
            hidden.slice(0..t * c),
            work.slice_mut((KERNEL - 1) * c..(KERNEL - 1 + t) * c),
            t,
            c,
            at(w.out_alpha),
        )
        .map_err(launch)?;
        keep(work, states, &mut state, KERNEL - 1, c, t)?;
        k::conv_to_mono(
            s,
            work.slice(0..(KERNEL - 1 + t) * c),
            at(w.out_conv),
            w.out_bias,
            self.samples.slice_mut(0..t),
            t,
            c,
            KERNEL,
        )
        .map_err(launch)?;
        s.download_async_front(self.samples.slice(0..t), &mut self.pinned)
            .map_err(hip)?;
        self.pending = t;
        Ok(())
    }

    /// Encodes mono audio at `sample_rate` whose length is a multiple of the
    /// hop; returns `[frames][latent_dim]` (the posterior mean; `Encode`).
    pub fn encode(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
        let hop = self.config.hop();
        if samples.is_empty() || !samples.len().is_multiple_of(hop) {
            return Err("AudioVAE input is not a multiple of its hop".into());
        }
        let w = &self.weights;
        let mut t = samples.len();
        let mut widest = t * self.config.encoder_dim;
        {
            let mut length = t;
            for block in &w.encoder_blocks {
                widest = widest.max(length * block.input);
                length /= block.stride;
                widest = widest.max(length * block.output);
            }
        }
        let history = (KERNEL - 1) * DILATIONS[2];
        let s = &self.stream;
        reserve(s, &mut self.hidden, widest).map_err(hip)?;
        reserve(s, &mut self.scratch, widest).map_err(hip)?;
        reserve(s, &mut self.work, widest + history * w.mu_channels).map_err(hip)?;
        reserve(s, &mut self.samples, t + KERNEL - 1).map_err(hip)?;
        let storage = &self.storage;
        let at = |offset: Offset| storage.slice(offset..storage.len());
        let work = &mut self.work;
        let hidden = &mut self.hidden;
        let scratch = &mut self.scratch;
        let blas = &self.blas;

        s.fill_bytes(self.samples.slice_mut(0..KERNEL - 1), 0)
            .map_err(hip)?;
        s.upload(samples, self.samples.slice_mut(KERNEL - 1..KERNEL - 1 + t))
            .map_err(hip)?;
        let encoder_dim = self.config.encoder_dim;
        k::conv_from_mono(
            s,
            self.samples.slice(0..t + KERNEL - 1),
            at(w.encoder_in),
            at(w.encoder_in_bias),
            hidden.slice_mut(0..t * encoder_dim),
            t,
            encoder_dim,
            KERNEL,
        )
        .map_err(launch)?;
        for block in &w.encoder_blocks {
            let c = block.input;
            for unit in &block.units {
                let rows = (KERNEL - 1) * unit.dilation;
                s.fill_bytes(work.slice_mut(0..rows * c), 0).map_err(hip)?;
                k::snake(
                    s,
                    hidden.slice(0..t * c),
                    work.slice_mut(rows * c..(rows + t) * c),
                    t,
                    c,
                    at(unit.alpha_in),
                )
                .map_err(launch)?;
                k::depthwise_conv(
                    s,
                    work.slice(0..(rows + t) * c),
                    at(unit.depthwise),
                    at(unit.depthwise_bias),
                    scratch.slice_mut(0..t * c),
                    t,
                    c,
                    KERNEL,
                    unit.dilation,
                )
                .map_err(launch)?;
                k::snake_in_place(s, scratch.slice_mut(0..t * c), t, c, at(unit.alpha_out))
                    .map_err(launch)?;
                Self::gemm(
                    blas,
                    s,
                    scratch.slice(0..t * c),
                    c,
                    at(unit.pointwise),
                    hidden.slice_mut(0..t * c),
                    t,
                    c,
                    c,
                    true,
                )?;
                k::add_bias_f32(s, hidden.slice_mut(0..t * c), at(unit.pointwise_bias), t, c)
                    .map_err(launch)?;
            }
            // Strided causal convolution (kernel 2s, left padding s): one
            // product per tap over every s-th frame.
            let stride = block.stride;
            s.fill_bytes(work.slice_mut(0..stride * c), 0)
                .map_err(hip)?;
            k::snake(
                s,
                hidden.slice(0..t * c),
                work.slice_mut(stride * c..(stride + t) * c),
                t,
                c,
                at(block.alpha),
            )
            .map_err(launch)?;
            let frames = t / stride;
            for tap in 0..2 * stride {
                Self::gemm(
                    blas,
                    s,
                    work.slice(tap * c..(stride + t) * c),
                    stride * c,
                    at(block.down + tap * block.output * c),
                    hidden.slice_mut(0..frames * block.output),
                    frames,
                    c,
                    block.output,
                    tap > 0,
                )?;
            }
            k::add_bias_f32(
                s,
                hidden.slice_mut(0..frames * block.output),
                at(block.down_bias),
                frames,
                block.output,
            )
            .map_err(launch)?;
            t = frames;
        }
        // fc_mu: kernel 3, left padding 2.
        let c = w.mu_channels;
        let latent = self.config.latent_dim;
        s.fill_bytes(work.slice_mut(0..2 * c), 0).map_err(hip)?;
        s.copy(hidden.slice(0..t * c), work.slice_mut(2 * c..(2 + t) * c))
            .map_err(hip)?;
        for tap in 0..3 {
            Self::gemm(
                blas,
                s,
                work.slice(tap * c..(2 + t) * c),
                c,
                at(w.mu + tap * latent * c),
                scratch.slice_mut(0..t * latent),
                t,
                c,
                latent,
                tap > 0,
            )?;
        }
        k::add_bias_f32(
            s,
            scratch.slice_mut(0..t * latent),
            at(w.mu_bias),
            t,
            latent,
        )
        .map_err(launch)?;
        let mut result = vec![0.0f32; t * latent];
        s.download(scratch.slice(0..t * latent), &mut result)
            .map_err(hip)?;
        Ok(result)
    }
}
