//! The audio tokenizer's encoder (`codec_encoder.cpp`): HuBERT, the
//! semantic and DAC encoders and the residual vector quantizer, in float32,
//! ported from Hugging Face transformers' HiggsAudioV2Tokenizer, DAC and
//! HuBERT modules.

use gufo_hip::blas::{Op, Strides};
use gufo_hip::{DeviceBuffer, DeviceSlice, Stream};
use gufo_omnivoice_kernels as k;

use crate::audio::resample;
use crate::codec::{
    Codec, DILATIONS, Ops, RESIDUAL_KERNEL, ResidualUnit, hip, launch, load_conv, require, reserve,
    upload, upload_tensor,
};
use crate::config::Config;
use crate::safetensors::SafetensorsFile;

/// nn.Linear: weight `[out][in]`.
pub(crate) struct Dense {
    weight: DeviceBuffer<f32>,
    bias: DeviceBuffer<f32>,
    input: usize,
    output: usize,
}

/// Conv1d as im2col plus one GEMM; weight `[out][in * kernel]` as PyTorch
/// stores it, no bias when the layer has none.
pub(crate) struct StridedConv {
    weight: DeviceBuffer<f32>,
    bias: Option<DeviceBuffer<f32>>,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    pad: usize,
}

impl StridedConv {
    fn output_length(&self, length: usize) -> usize {
        (length + 2 * self.pad - self.dilation * (self.kernel - 1) - 1) / self.stride + 1
    }
}

/// A post-norm HuBERT encoder layer; q, k and v share one projection.
struct HubertLayer {
    qkv: Dense,
    output: Dense,
    norm_weight: DeviceBuffer<f32>,
    norm_bias: DeviceBuffer<f32>,
    intermediate: Dense,
    final_dense: Dense,
    final_norm_weight: DeviceBuffer<f32>,
    final_norm_bias: DeviceBuffer<f32>,
}

struct SemanticUnit {
    conv1: StridedConv,
    conv2: StridedConv,
}

struct SemanticBlock {
    units: Vec<SemanticUnit>,
    conv: StridedConv,
}

struct AcousticBlock {
    units: Vec<ResidualUnit>,
    alpha: DeviceBuffer<f32>,
    downsample: StridedConv,
}

pub(crate) struct Encoder {
    // HuBERT: feature extractor, projection, positional convolution
    // ([tap][group][out][in], weight norm folded) and layers.
    features: Vec<StridedConv>,
    group_norm_weight: DeviceBuffer<f32>,
    group_norm_bias: DeviceBuffer<f32>,
    projection_norm_weight: DeviceBuffer<f32>,
    projection_norm_bias: DeviceBuffer<f32>,
    projection: Dense,
    position_weight: DeviceBuffer<f32>,
    position_bias: DeviceBuffer<f32>,
    encoder_norm_weight: DeviceBuffer<f32>,
    encoder_norm_bias: DeviceBuffer<f32>,
    layers: Vec<HubertLayer>,
    // Semantic encoder.
    semantic_input: StridedConv,
    semantic_blocks: Vec<SemanticBlock>,
    // DAC encoder.
    acoustic_input: StridedConv,
    acoustic_blocks: Vec<AcousticBlock>,
    acoustic_alpha: DeviceBuffer<f32>,
    acoustic_output: StridedConv,
    fc: Dense,
    // RVQ: project_in, the codebooks and their squared norms.
    project_in: Vec<Dense>,
    codebooks: Vec<DeviceBuffer<f32>>,
    codebook_norms: Vec<DeviceBuffer<f32>>,
    /// im2col scratch.
    columns: DeviceBuffer<f32>,
}

fn load_dense(
    stream: &Stream,
    file: &SafetensorsFile,
    prefix: &str,
    input: usize,
    output: usize,
) -> Result<Dense, String> {
    Ok(Dense {
        weight: upload_tensor(stream, file, &format!("{prefix}.weight"), &[output, input])?,
        bias: upload_tensor(stream, file, &format!("{prefix}.bias"), &[output])?,
        input,
        output,
    })
}

/// q, k and v projections stacked into one.
fn load_qkv(
    stream: &Stream,
    file: &SafetensorsFile,
    prefix: &str,
    width: usize,
) -> Result<Dense, String> {
    let mut weight = Vec::with_capacity(3 * width * width);
    let mut bias = Vec::with_capacity(3 * width);
    for name in ["q_proj", "k_proj", "v_proj"] {
        weight.extend(require(
            file,
            &format!("{prefix}{name}.weight"),
            &[width, width],
        )?);
        bias.extend(require(file, &format!("{prefix}{name}.bias"), &[width])?);
    }
    Ok(Dense {
        weight: upload(stream, &weight)?,
        bias: upload(stream, &bias)?,
        input: width,
        output: 3 * width,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_strided_conv(
    stream: &Stream,
    file: &SafetensorsFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    pad: usize,
    bias: bool,
) -> Result<StridedConv, String> {
    Ok(StridedConv {
        weight: upload_tensor(
            stream,
            file,
            &format!("{prefix}.weight"),
            &[output, input, kernel],
        )?,
        bias: if bias {
            Some(upload_tensor(
                stream,
                file,
                &format!("{prefix}.bias"),
                &[output],
            )?)
        } else {
            None
        },
        input,
        output,
        kernel,
        stride,
        dilation,
        pad,
    })
}

impl Encoder {
    /// Loads the encoder's weights (`LoadEncoder`).
    pub(crate) fn load(
        config: &Config,
        file: &SafetensorsFile,
        stream: &Stream,
    ) -> Result<Self, String> {
        let codec = &config.codec;
        let semantic = &codec.semantic;
        let acoustic = &codec.acoustic;
        let width = semantic.hidden_size;
        // HuBERT feature extractor: bias-free convolutions, the first one
        // with a per-channel group norm.
        let mut input = 1;
        let mut features = Vec::new();
        for i in 0..semantic.conv_channels.len() {
            let prefix = format!("semantic_model.feature_extractor.conv_layers.{i}");
            features.push(load_strided_conv(
                stream,
                file,
                &format!("{prefix}.conv"),
                input,
                semantic.conv_channels[i],
                semantic.conv_kernels[i],
                semantic.conv_strides[i],
                1,
                0,
                false,
            )?);
            input = semantic.conv_channels[i];
        }
        let feature_channels = input;
        let first_channels = semantic.conv_channels.first().copied().unwrap_or(0);
        let group_norm_weight = upload_tensor(
            stream,
            file,
            "semantic_model.feature_extractor.conv_layers.0.layer_norm.weight",
            &[first_channels],
        )?;
        let group_norm_bias = upload_tensor(
            stream,
            file,
            "semantic_model.feature_extractor.conv_layers.0.layer_norm.bias",
            &[first_channels],
        )?;
        let projection_norm_weight = upload_tensor(
            stream,
            file,
            "semantic_model.feature_projection.layer_norm.weight",
            &[feature_channels],
        )?;
        let projection_norm_bias = upload_tensor(
            stream,
            file,
            "semantic_model.feature_projection.layer_norm.bias",
            &[feature_channels],
        )?;
        let projection = load_dense(
            stream,
            file,
            "semantic_model.feature_projection.projection",
            feature_channels,
            width,
        )?;
        // Positional convolution: fold the weight norm (dim=2, one gain per
        // tap) and rearrange to [tap][group][out][in].
        let (position_weight, position_bias) = {
            let kernel = semantic.position_kernel;
            let groups = semantic.position_groups;
            let group_width = width / groups;
            let prefix = "semantic_model.encoder.pos_conv_embed.conv.";
            let g = require(
                file,
                &format!("{prefix}parametrizations.weight.original0"),
                &[1, 1, kernel],
            )?;
            let v = require(
                file,
                &format!("{prefix}parametrizations.weight.original1"),
                &[width, group_width, kernel],
            )?;
            let mut norms = vec![0.0f64; kernel];
            for o in 0..width {
                for i in 0..group_width {
                    for (t, norm) in norms.iter_mut().enumerate() {
                        let value = f64::from(v[(o * group_width + i) * kernel + t]);
                        *norm += value * value;
                    }
                }
            }
            let mut weight = vec![0.0f32; kernel * width * group_width];
            for o in 0..width {
                let group = o / group_width;
                let local = o % group_width;
                for i in 0..group_width {
                    for t in 0..kernel {
                        // float * float, promoted to double for the division,
                        // as the C++ computes it.
                        let product = g[t] * v[(o * group_width + i) * kernel + t];
                        let folded = (f64::from(product) / norms[t].sqrt()) as f32;
                        weight[((t * groups + group) * group_width + local) * group_width + i] =
                            folded;
                    }
                }
            }
            (
                upload(stream, &weight)?,
                upload_tensor(stream, file, &format!("{prefix}bias"), &[width])?,
            )
        };
        let encoder_norm_weight = upload_tensor(
            stream,
            file,
            "semantic_model.encoder.layer_norm.weight",
            &[width],
        )?;
        let encoder_norm_bias = upload_tensor(
            stream,
            file,
            "semantic_model.encoder.layer_norm.bias",
            &[width],
        )?;
        let mut layers = Vec::with_capacity(semantic.layers);
        for l in 0..semantic.layers {
            let prefix = format!("semantic_model.encoder.layers.{l}.");
            layers.push(HubertLayer {
                qkv: load_qkv(stream, file, &format!("{prefix}attention."), width)?,
                output: load_dense(
                    stream,
                    file,
                    &format!("{prefix}attention.out_proj"),
                    width,
                    width,
                )?,
                norm_weight: upload_tensor(
                    stream,
                    file,
                    &format!("{prefix}layer_norm.weight"),
                    &[width],
                )?,
                norm_bias: upload_tensor(
                    stream,
                    file,
                    &format!("{prefix}layer_norm.bias"),
                    &[width],
                )?,
                intermediate: load_dense(
                    stream,
                    file,
                    &format!("{prefix}feed_forward.intermediate_dense"),
                    width,
                    semantic.intermediate_size,
                )?,
                final_dense: load_dense(
                    stream,
                    file,
                    &format!("{prefix}feed_forward.output_dense"),
                    semantic.intermediate_size,
                    width,
                )?,
                final_norm_weight: upload_tensor(
                    stream,
                    file,
                    &format!("{prefix}final_layer_norm.weight"),
                    &[width],
                )?,
                final_norm_bias: upload_tensor(
                    stream,
                    file,
                    &format!("{prefix}final_layer_norm.bias"),
                    &[width],
                )?,
            });
        }
        // Semantic encoder: one bias-free k3 convolution, then blocks of ELU
        // residual units and a k3 convolution (the checkpoint's strides are 1).
        let semantic_input = load_strided_conv(
            stream,
            file,
            "encoder_semantic.conv",
            width,
            width,
            3,
            1,
            1,
            1,
            false,
        )?;
        let mut semantic_blocks = Vec::with_capacity(2);
        for b in 0..2 {
            let prefix = format!("encoder_semantic.conv_blocks.{b}");
            let mut units = Vec::with_capacity(2);
            for u in 0..2 {
                let unit = format!("{prefix}.res_units.{u}");
                units.push(SemanticUnit {
                    conv1: load_strided_conv(
                        stream,
                        file,
                        &format!("{unit}.conv1"),
                        width,
                        width,
                        3,
                        1,
                        1,
                        1,
                        false,
                    )?,
                    conv2: load_strided_conv(
                        stream,
                        file,
                        &format!("{unit}.conv2"),
                        width,
                        width,
                        1,
                        1,
                        1,
                        0,
                        false,
                    )?,
                });
            }
            semantic_blocks.push(SemanticBlock {
                units,
                conv: load_strided_conv(
                    stream,
                    file,
                    &format!("{prefix}.conv"),
                    width,
                    width,
                    3,
                    1,
                    1,
                    1,
                    true,
                )?,
            });
        }
        // DAC encoder.
        let acoustic_input = load_strided_conv(
            stream,
            file,
            "acoustic_encoder.conv1",
            1,
            acoustic.encoder_channels,
            RESIDUAL_KERNEL,
            1,
            1,
            3,
            true,
        )?;
        let mut channels = acoustic.encoder_channels;
        let mut acoustic_blocks = Vec::new();
        for (index, &stride) in acoustic.downsampling_ratios.iter().enumerate() {
            let prefix = format!("acoustic_encoder.block.{index}");
            let mut units = Vec::with_capacity(3);
            for (unit, &dilation) in DILATIONS.iter().enumerate() {
                let name = format!("{prefix}.res_unit{}", unit + 1);
                units.push(ResidualUnit {
                    alpha1: upload_tensor(
                        stream,
                        file,
                        &format!("{name}.snake1.alpha"),
                        &[1, channels, 1],
                    )?,
                    alpha2: upload_tensor(
                        stream,
                        file,
                        &format!("{name}.snake2.alpha"),
                        &[1, channels, 1],
                    )?,
                    conv1: load_conv(
                        stream,
                        file,
                        &format!("{name}.conv1"),
                        channels,
                        channels,
                        RESIDUAL_KERNEL,
                        dilation,
                    )?,
                    conv2: load_conv(
                        stream,
                        file,
                        &format!("{name}.conv2"),
                        channels,
                        channels,
                        1,
                        1,
                    )?,
                });
            }
            acoustic_blocks.push(AcousticBlock {
                units,
                alpha: upload_tensor(
                    stream,
                    file,
                    &format!("{prefix}.snake1.alpha"),
                    &[1, channels, 1],
                )?,
                downsample: load_strided_conv(
                    stream,
                    file,
                    &format!("{prefix}.conv1"),
                    channels,
                    2 * channels,
                    2 * stride,
                    stride,
                    1,
                    stride.div_ceil(2),
                    true,
                )?,
            });
            channels *= 2;
        }
        let acoustic_alpha = upload_tensor(
            stream,
            file,
            "acoustic_encoder.snake1.alpha",
            &[1, channels, 1],
        )?;
        let acoustic_output = load_strided_conv(
            stream,
            file,
            "acoustic_encoder.conv2",
            channels,
            acoustic.latent_channels,
            3,
            1,
            1,
            1,
            true,
        )?;
        let hidden = width + acoustic.latent_channels;
        let fc = load_dense(stream, file, "fc", hidden, hidden)?;
        let mut project_in = Vec::with_capacity(codec.codebooks);
        let mut codebooks = Vec::with_capacity(codec.codebooks);
        let mut codebook_norms = Vec::with_capacity(codec.codebooks);
        for c in 0..codec.codebooks {
            let prefix = format!("quantizer.quantizers.{c}");
            project_in.push(load_dense(
                stream,
                file,
                &format!("{prefix}.project_in"),
                hidden,
                codec.codebook_dim,
            )?);
            let entries = upload_tensor(
                stream,
                file,
                &format!("{prefix}.codebook.embed"),
                &[codec.codebook_size, codec.codebook_dim],
            )?;
            let mut norms = stream
                .alloc::<f32>(codec.codebook_size.max(1))
                .map_err(hip)?;
            k::squared_norms(
                stream,
                entries.slice(0..codec.codebook_size * codec.codebook_dim),
                norms.slice_mut(0..codec.codebook_size),
                codec.codebook_size,
                codec.codebook_dim,
            )
            .map_err(launch)?;
            codebooks.push(entries);
            codebook_norms.push(norms);
        }
        stream.synchronize().map_err(hip)?;
        Ok(Self {
            features,
            group_norm_weight,
            group_norm_bias,
            projection_norm_weight,
            projection_norm_bias,
            projection,
            position_weight,
            position_bias,
            encoder_norm_weight,
            encoder_norm_bias,
            layers,
            semantic_input,
            semantic_blocks,
            acoustic_input,
            acoustic_blocks,
            acoustic_alpha,
            acoustic_output,
            fc,
            project_in,
            codebooks,
            codebook_norms,
            columns: stream.alloc(1).map_err(hip)?,
        })
    }
}

/// `output[out][t]` = Conv1d over a channel-major input; returns the output
/// length.
fn convolve(
    ops: &Ops,
    columns: &mut DeviceBuffer<f32>,
    conv: &StridedConv,
    input: DeviceSlice<'_, f32>,
    length: usize,
    output: &mut DeviceBuffer<f32>,
) -> Result<usize, String> {
    let s = &ops.stream;
    let output_length = conv.output_length(length);
    let taps = conv.input * conv.kernel;
    let direct = conv.kernel == 1 && conv.stride == 1 && conv.pad == 0;
    if !direct {
        reserve(s, columns, taps * output_length).map_err(hip)?;
        k::im2col(
            s,
            input.range(0..conv.input * length),
            conv.input,
            length,
            conv.kernel,
            conv.stride,
            conv.dilation,
            conv.pad,
            output_length,
            columns.slice_mut(0..taps * output_length),
        )
        .map_err(launch)?;
    }
    let source = if direct {
        input.range(0..taps * output_length)
    } else {
        columns.slice(0..taps * output_length)
    };
    ops.blas
        .sgemm(
            s,
            Op::N,
            Op::N,
            output_length,
            conv.output,
            taps,
            1.0,
            source,
            output_length,
            conv.weight.as_slice(),
            taps,
            output.slice_mut(0..conv.output * output_length),
            output_length,
        )
        .map_err(launch)?;
    if let Some(bias) = &conv.bias {
        k::add_channel_bias(
            s,
            output.slice_mut(0..conv.output * output_length),
            output_length,
            bias.as_slice(),
            conv.output,
            output_length,
        )
        .map_err(launch)?;
    }
    Ok(output_length)
}

/// `y[rows, out] (+)= x[rows, in] * W^T + bias`.
fn linear(
    ops: &Ops,
    dense: &Dense,
    x: DeviceSlice<'_, f32>,
    rows: usize,
    y: &mut DeviceBuffer<f32>,
    accumulate: bool,
) -> Result<(), String> {
    let s = &ops.stream;
    ops.blas
        .sgemm_beta(
            s,
            Op::T,
            Op::N,
            dense.output,
            rows,
            dense.input,
            1.0,
            dense.weight.as_slice(),
            dense.input,
            x.range(0..rows * dense.input),
            dense.input,
            if accumulate { 1.0 } else { 0.0 },
            y.slice_mut(0..rows * dense.output),
            dense.output,
        )
        .map_err(launch)?;
    k::add_column_bias(
        s,
        y.slice_mut(0..rows * dense.output),
        dense.bias.as_slice(),
        rows,
        dense.output,
    )
    .map_err(launch)
}

impl Codec {
    /// `samples` is 24 kHz mono, a whole number of frames; returns
    /// `[codebook][frame]` codes and the frame count.
    pub fn encode(&mut self, samples: &[f32]) -> Result<(Vec<i32>, usize), String> {
        self.encode_inner(samples)
            .map_err(|e| format!("OmniVoice encode: {e}"))
    }

    fn encode_inner(&mut self, samples: &[f32]) -> Result<(Vec<i32>, usize), String> {
        let config = self.config.clone();
        let codec = &config.codec;
        let semantic = &codec.semantic;
        let acoustic = &codec.acoustic;
        let width = semantic.hidden_size;
        let hidden = width + acoustic.latent_channels;
        if samples.is_empty() || !samples.len().is_multiple_of(codec.frame_samples) {
            return Err("reference audio must be a whole number of frames".to_string());
        }
        let length = samples.len();
        let count = length / codec.frame_samples;
        let e = &mut self.encoder;
        let ops = &self.ops;
        let scratch = &mut self.scratch;
        let s = &ops.stream;
        let alloc = |len: usize| s.alloc::<f32>(len.max(1)).map_err(hip);

        // ---- Semantic features: HuBERT at 16 kHz with 160 samples of padding.
        let mut resampled = vec![0.0f32; 160];
        resampled.extend(resample(samples, codec.sample_rate, semantic.sample_rate));
        resampled.extend(std::iter::repeat_n(0.0f32, 160));
        let mut feature_length = resampled.len();
        let mut largest = feature_length;
        for conv in &e.features {
            feature_length = conv.output_length(feature_length);
            largest = largest.max(conv.output * feature_length);
        }
        let steps = feature_length; // HuBERT frames
        let mut a = alloc(largest.max(length * acoustic.encoder_channels))?;
        let mut b = alloc(a.len())?;
        s.upload(&resampled, a.slice_mut(0..resampled.len()))
            .map_err(hip)?;
        let mut current = resampled.len();
        for i in 0..e.features.len() {
            let conv = &e.features[i];
            current = convolve(ops, &mut e.columns, conv, a.as_slice(), current, &mut b)?;
            if i == 0 {
                k::group_norm_gelu(
                    s,
                    b.slice_mut(0..conv.output * current),
                    e.group_norm_weight.as_slice(),
                    e.group_norm_bias.as_slice(),
                    conv.output,
                    current,
                    1e-5,
                )
                .map_err(launch)?;
            } else {
                k::gelu(
                    s,
                    b.slice_mut(0..conv.output * current),
                    conv.output * current,
                )
                .map_err(launch)?;
            }
            std::mem::swap(&mut a, &mut b);
        }
        let feature_channels = e.features.last().map_or(1, |conv| conv.output);
        // [channels][steps] -> [steps][channels], LayerNorm, projection.
        let mut rows = alloc(steps * feature_channels.max(semantic.intermediate_size))?;
        k::transpose(
            s,
            a.slice(0..feature_channels * steps),
            feature_channels,
            steps,
            rows.slice_mut(0..steps * feature_channels),
            feature_channels,
        )
        .map_err(launch)?;
        k::layer_norm_in_place(
            s,
            e.projection_norm_weight.as_slice(),
            e.projection_norm_bias.as_slice(),
            rows.slice_mut(0..steps * feature_channels),
            steps,
            feature_channels,
            semantic.layer_norm_eps,
        )
        .map_err(launch)?;
        let mut state_rows = alloc(steps * width)?;
        linear(
            ops,
            &e.projection,
            rows.as_slice(),
            steps,
            &mut state_rows,
            false,
        )?;
        // Positional convolution over the channel-major, zero-padded states.
        let kernel = semantic.position_kernel;
        let pad = kernel / 2;
        let padded_steps = steps + 2 * pad;
        let groups = semantic.position_groups;
        let group_width = width / groups;
        let mut padded = alloc(width * padded_steps)?;
        s.fill_bytes(padded.slice_mut(0..width * padded_steps), 0)
            .map_err(hip)?;
        k::transpose(
            s,
            state_rows.slice(0..steps * width),
            steps,
            width,
            padded.slice_mut(pad..width * padded_steps),
            padded_steps,
        )
        .map_err(launch)?;
        let mut position = alloc(width * steps)?;
        for t in 0..kernel {
            let weight = t * groups * group_width * group_width;
            ops.blas
                .sgemm_strided_batched_beta(
                    s,
                    Op::N,
                    Op::N,
                    steps,
                    group_width,
                    group_width,
                    1.0,
                    padded.slice(t..width * padded_steps),
                    padded_steps,
                    e.position_weight
                        .slice(weight..weight + groups * group_width * group_width),
                    group_width,
                    if t == 0 { 0.0 } else { 1.0 },
                    position.slice_mut(0..width * steps),
                    steps,
                    Strides {
                        count: groups,
                        a: group_width * padded_steps,
                        b: group_width * group_width,
                        c: group_width * steps,
                    },
                )
                .map_err(launch)?;
        }
        k::add_channel_bias(
            s,
            position.slice_mut(0..width * steps),
            steps,
            e.position_bias.as_slice(),
            width,
            steps,
        )
        .map_err(launch)?;
        k::gelu(s, position.slice_mut(0..width * steps), width * steps).map_err(launch)?;
        k::transpose(
            s,
            position.slice(0..width * steps),
            width,
            steps,
            rows.slice_mut(0..steps * width),
            width,
        )
        .map_err(launch)?;
        k::accumulate(
            s,
            rows.slice(0..steps * width),
            state_rows.slice_mut(0..steps * width),
            steps * width,
        )
        .map_err(launch)?;
        k::layer_norm_in_place(
            s,
            e.encoder_norm_weight.as_slice(),
            e.encoder_norm_bias.as_slice(),
            state_rows.slice_mut(0..steps * width),
            steps,
            width,
            semantic.layer_norm_eps,
        )
        .map_err(launch)?;
        // Layers; the features average the layer input and every layer output.
        let mut mean = alloc(steps * width)?;
        s.copy(
            state_rows.slice(0..steps * width),
            mean.slice_mut(0..steps * width),
        )
        .map_err(hip)?;
        let heads = semantic.heads;
        let head_dim = width / heads;
        let mut qkv = alloc(steps * 3 * width)?;
        let mut scores = alloc(heads * steps * steps)?;
        let mut attention = alloc(steps * width)?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        for layer in &e.layers {
            linear(
                ops,
                &layer.qkv,
                state_rows.as_slice(),
                steps,
                &mut qkv,
                false,
            )?;
            let qkv_len = steps * 3 * width;
            ops.blas
                .sgemm_strided_batched(
                    s,
                    Op::T,
                    Op::N,
                    steps,
                    steps,
                    head_dim,
                    1.0,
                    qkv.slice(width..qkv_len),
                    3 * width,
                    qkv.slice(0..qkv_len),
                    3 * width,
                    scores.slice_mut(0..heads * steps * steps),
                    steps,
                    Strides {
                        count: heads,
                        a: head_dim,
                        b: head_dim,
                        c: steps * steps,
                    },
                )
                .map_err(launch)?;
            k::softmax_in_place(
                s,
                scores.slice_mut(0..heads * steps * steps),
                heads * steps,
                steps,
                scale,
            )
            .map_err(launch)?;
            ops.blas
                .sgemm_strided_batched(
                    s,
                    Op::N,
                    Op::N,
                    head_dim,
                    steps,
                    steps,
                    1.0,
                    qkv.slice(2 * width..qkv_len),
                    3 * width,
                    scores.slice(0..heads * steps * steps),
                    steps,
                    attention.slice_mut(0..steps * width),
                    width,
                    Strides {
                        count: heads,
                        a: head_dim,
                        b: steps * steps,
                        c: head_dim,
                    },
                )
                .map_err(launch)?;
            linear(
                ops,
                &layer.output,
                attention.as_slice(),
                steps,
                &mut state_rows,
                true,
            )?;
            k::layer_norm_in_place(
                s,
                layer.norm_weight.as_slice(),
                layer.norm_bias.as_slice(),
                state_rows.slice_mut(0..steps * width),
                steps,
                width,
                semantic.layer_norm_eps,
            )
            .map_err(launch)?;
            linear(
                ops,
                &layer.intermediate,
                state_rows.as_slice(),
                steps,
                &mut rows,
                false,
            )?;
            k::gelu(
                s,
                rows.slice_mut(0..steps * semantic.intermediate_size),
                steps * semantic.intermediate_size,
            )
            .map_err(launch)?;
            linear(
                ops,
                &layer.final_dense,
                rows.as_slice(),
                steps,
                &mut state_rows,
                true,
            )?;
            k::layer_norm_in_place(
                s,
                layer.final_norm_weight.as_slice(),
                layer.final_norm_bias.as_slice(),
                state_rows.slice_mut(0..steps * width),
                steps,
                width,
                semantic.layer_norm_eps,
            )
            .map_err(launch)?;
            k::accumulate(
                s,
                state_rows.slice(0..steps * width),
                mean.slice_mut(0..steps * width),
                steps * width,
            )
            .map_err(launch)?;
        }
        k::scale(
            s,
            mean.slice_mut(0..steps * width),
            steps * width,
            1.0 / (e.layers.len() + 1) as f32,
        )
        .map_err(launch)?;
        // Every semantic_stride-th frame, channel-major.
        if steps < count * codec.semantic_stride {
            return Err("reference audio is too short for the semantic encoder".to_string());
        }
        let picks: Vec<u32> = (0..count)
            .map(|f| (f * codec.semantic_stride) as u32)
            .collect();
        let bound = u32::try_from(steps).map_err(|_| "too many HuBERT frames".to_string())?;
        let mut pick_rows = s.alloc_indices(count.max(1), bound).map_err(hip)?;
        s.upload_indices(&picks, &mut pick_rows).map_err(hip)?;
        k::gather_rows(
            s,
            mean.slice(0..steps * width),
            pick_rows.slice(0..count),
            rows.slice_mut(0..count * width),
            count,
            width,
            steps,
        )
        .map_err(launch)?;
        let mut semantic_a = alloc(width * count)?;
        let mut semantic_b = alloc(width * count)?;
        let mut semantic_c = alloc(width * count)?;
        k::transpose(
            s,
            rows.slice(0..count * width),
            count,
            width,
            semantic_a.slice_mut(0..width * count),
            count,
        )
        .map_err(launch)?;
        // ---- Semantic encoder.
        convolve(
            ops,
            &mut e.columns,
            &e.semantic_input,
            semantic_a.as_slice(),
            count,
            &mut semantic_b,
        )?;
        for block in &e.semantic_blocks {
            for unit in &block.units {
                k::elu(
                    s,
                    semantic_b.slice(0..width * count),
                    semantic_a.slice_mut(0..width * count),
                    width * count,
                )
                .map_err(launch)?;
                convolve(
                    ops,
                    &mut e.columns,
                    &unit.conv1,
                    semantic_a.as_slice(),
                    count,
                    &mut semantic_c,
                )?;
                k::elu_in_place(s, semantic_c.slice_mut(0..width * count), width * count)
                    .map_err(launch)?;
                convolve(
                    ops,
                    &mut e.columns,
                    &unit.conv2,
                    semantic_c.as_slice(),
                    count,
                    &mut semantic_a,
                )?;
                k::accumulate(
                    s,
                    semantic_a.slice(0..width * count),
                    semantic_b.slice_mut(0..width * count),
                    width * count,
                )
                .map_err(launch)?;
            }
            convolve(
                ops,
                &mut e.columns,
                &block.conv,
                semantic_b.as_slice(),
                count,
                &mut semantic_a,
            )?;
            std::mem::swap(&mut semantic_a, &mut semantic_b);
        }
        // ---- DAC encoder into the first latent rows of the joint embedding.
        let mut joint = alloc(hidden * count)?;
        let latent = acoustic.latent_channels * count;
        s.copy(
            semantic_b.slice(0..width * count),
            joint.slice_mut(latent..latent + width * count),
        )
        .map_err(hip)?;
        s.upload(samples, b.slice_mut(0..length)).map_err(hip)?;
        current = convolve(
            ops,
            &mut e.columns,
            &e.acoustic_input,
            b.as_slice(),
            length,
            &mut a,
        )?;
        let mut channels = acoustic.encoder_channels;
        reserve(s, &mut scratch.y, channels * current).map_err(hip)?;
        reserve(s, &mut scratch.padded, channels * (current + 2 * 27)).map_err(hip)?;
        for block in &e.acoustic_blocks {
            for unit in &block.units {
                ops.residual(scratch, unit, &mut a, channels, current)
                    .map_err(launch)?;
            }
            k::snake(
                s,
                a.slice(0..channels * current),
                current,
                block.alpha.as_slice(),
                scratch.padded.slice_mut(0..channels * current),
                channels,
                current,
                0,
            )
            .map_err(launch)?;
            current = convolve(
                ops,
                &mut e.columns,
                &block.downsample,
                scratch.padded.as_slice(),
                current,
                &mut b,
            )?;
            channels *= 2;
            std::mem::swap(&mut a, &mut b);
        }
        k::snake(
            s,
            a.slice(0..channels * current),
            current,
            e.acoustic_alpha.as_slice(),
            scratch.padded.slice_mut(0..channels * current),
            channels,
            current,
            0,
        )
        .map_err(launch)?;
        if current != count {
            return Err("acoustic and semantic frame counts differ".to_string());
        }
        convolve(
            ops,
            &mut e.columns,
            &e.acoustic_output,
            scratch.padded.as_slice(),
            current,
            &mut joint,
        )?;
        // ---- fc and the residual vector quantizer.
        let mut residual = alloc(count * hidden)?;
        k::transpose(
            s,
            joint.slice(0..hidden * count),
            hidden,
            count,
            rows.slice_mut(0..count * hidden),
            hidden,
        )
        .map_err(launch)?;
        linear(ops, &e.fc, rows.as_slice(), count, &mut residual, false)?;
        let mut projected = alloc(count * codec.codebook_dim)?;
        let mut dots = alloc(count * codec.codebook_size)?;
        let entries =
            u32::try_from(codec.codebook_size).map_err(|_| "codebook too large".to_string())?;
        let mut indices = s
            .alloc_indices((codec.codebooks * count).max(1), entries)
            .map_err(hip)?;
        for c in 0..codec.codebooks {
            linear(
                ops,
                &e.project_in[c],
                residual.as_slice(),
                count,
                &mut projected,
                false,
            )?;
            ops.blas
                .sgemm(
                    s,
                    Op::T,
                    Op::N,
                    codec.codebook_size,
                    count,
                    codec.codebook_dim,
                    1.0,
                    e.codebooks[c].as_slice(),
                    codec.codebook_dim,
                    projected.slice(0..count * codec.codebook_dim),
                    codec.codebook_dim,
                    dots.slice_mut(0..count * codec.codebook_size),
                    codec.codebook_size,
                )
                .map_err(launch)?;
            k::nearest_entry(
                s,
                dots.slice(0..count * codec.codebook_size),
                projected.slice(0..count * codec.codebook_dim),
                e.codebook_norms[c].slice(0..codec.codebook_size),
                indices.slice_mut(c * count..(c + 1) * count),
                count,
                codec.codebook_size,
                codec.codebook_dim,
            )
            .map_err(launch)?;
            let table = c * codec.codebook_size * hidden;
            k::subtract_entries(
                s,
                residual.slice_mut(0..count * hidden),
                self.tables
                    .slice(table..table + codec.codebook_size * hidden),
                indices.slice(c * count..(c + 1) * count),
                count,
                hidden,
                codec.codebook_size,
            )
            .map_err(launch)?;
        }
        let mut codes = vec![0u32; codec.codebooks * count];
        s.download_blocking(indices.view().values().range(0..codes.len()), &mut codes)
            .map_err(hip)?;
        Ok((codes.into_iter().map(|code| code as i32).collect(), count))
    }
}
