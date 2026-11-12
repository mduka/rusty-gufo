//! The Higgs Audio v2 tokenizer on the GPU in float32 (`codec.cpp`,
//! `codec_state.hpp`): RVQ codes to 24 kHz audio through the DAC decoder.
//! The encoder (audio to codes) is in `codec_encoder.rs`. The DAC modules
//! port Hugging Face transformers' HiggsAudioV2Tokenizer.

use std::path::Path;

use gufo_hip::blas::{Blas, Op};
use gufo_hip::{
    DeviceBuffer, DeviceIndices, DeviceSlice, DeviceSliceMut, HipError, LaunchError, Stream,
};
use gufo_omnivoice_kernels as k;

use crate::codec_encoder::Encoder;
use crate::config::Config;
use crate::safetensors::{DType, SafetensorsFile};

pub(crate) const RESIDUAL_KERNEL: usize = 7;
pub(crate) const DILATIONS: [usize; 3] = [1, 3, 9];

/// Conv1d with weights rearranged to `[tap][out][in]`.
pub(crate) struct Conv {
    pub weight: DeviceBuffer<f32>,
    pub bias: DeviceBuffer<f32>,
    pub input: usize,
    pub output: usize,
    pub kernel: usize,
    pub dilation: usize,
}

impl Conv {
    pub fn pad(&self) -> usize {
        (self.kernel - 1) * self.dilation / 2
    }
}

/// ConvTranspose1d with weights rearranged to `[tap * out][in]`.
pub(crate) struct ConvTranspose {
    pub weight: DeviceBuffer<f32>,
    pub bias: DeviceBuffer<f32>,
    pub input: usize,
    pub output: usize,
    pub stride: usize,
}

pub(crate) struct ResidualUnit {
    pub alpha1: DeviceBuffer<f32>,
    pub alpha2: DeviceBuffer<f32>,
    pub conv1: Conv,
    pub conv2: Conv,
}

pub(crate) struct DecoderBlock {
    pub alpha: DeviceBuffer<f32>,
    pub upsample: ConvTranspose,
    pub units: Vec<ResidualUnit>,
}

pub(crate) fn hip(error: HipError) -> String {
    error.to_string()
}

pub(crate) fn launch(error: LaunchError) -> String {
    error.to_string()
}

/// A float32 tensor of this exact shape (`Require`) as host values.
pub(crate) fn require(
    file: &SafetensorsFile,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>, String> {
    let tensor = file.expect(name, DType::F32, shape)?;
    file.f32_values(tensor)
}

/// `Upload`.
pub(crate) fn upload(stream: &Stream, data: &[f32]) -> Result<DeviceBuffer<f32>, String> {
    let mut buffer = stream.alloc(data.len().max(1)).map_err(hip)?;
    stream
        .upload(data, buffer.slice_mut(0..data.len()))
        .map_err(hip)?;
    Ok(buffer)
}

/// `UploadTensor`: a float32 tensor of this exact shape on the device.
pub(crate) fn upload_tensor(
    stream: &Stream,
    file: &SafetensorsFile,
    name: &str,
    shape: &[usize],
) -> Result<DeviceBuffer<f32>, String> {
    let tensor = file.expect(name, DType::F32, shape)?;
    let mut buffer = stream.alloc(tensor.elements().max(1)).map_err(hip)?;
    stream
        .upload_bytes(file.bytes(tensor), buffer.slice_mut(0..tensor.elements()))
        .map_err(hip)?;
    Ok(buffer)
}

pub(crate) fn load_conv(
    stream: &Stream,
    file: &SafetensorsFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    dilation: usize,
) -> Result<Conv, String> {
    let source = require(file, &format!("{prefix}.weight"), &[output, input, kernel])?;
    let mut rearranged = vec![0.0f32; output * input * kernel];
    for o in 0..output {
        for i in 0..input {
            for t in 0..kernel {
                rearranged[(t * output + o) * input + i] = source[(o * input + i) * kernel + t];
            }
        }
    }
    Ok(Conv {
        weight: upload(stream, &rearranged)?,
        bias: upload_tensor(stream, file, &format!("{prefix}.bias"), &[output])?,
        input,
        output,
        kernel,
        dilation,
    })
}

fn load_conv_transpose(
    stream: &Stream,
    file: &SafetensorsFile,
    prefix: &str,
    input: usize,
    output: usize,
    stride: usize,
) -> Result<ConvTranspose, String> {
    let kernel = 2 * stride;
    let source = require(file, &format!("{prefix}.weight"), &[input, output, kernel])?;
    let mut rearranged = vec![0.0f32; kernel * output * input];
    for i in 0..input {
        for o in 0..output {
            for t in 0..kernel {
                rearranged[(t * output + o) * input + i] = source[(i * output + o) * kernel + t];
            }
        }
    }
    Ok(ConvTranspose {
        weight: upload(stream, &rearranged)?,
        bias: upload_tensor(stream, file, &format!("{prefix}.bias"), &[output])?,
        input,
        output,
        stride,
    })
}

/// Reallocates `buffer` when `count` exceeds its capacity; contents are
/// lost.
pub(crate) fn reserve(
    stream: &Stream,
    buffer: &mut DeviceBuffer<f32>,
    count: usize,
) -> Result<(), HipError> {
    if count > buffer.len() {
        *buffer = stream.alloc(count.max(1))?;
    }
    Ok(())
}

/// The stream and hipBLAS handle of the codec.
pub(crate) struct Ops {
    pub stream: Stream,
    pub blas: Blas,
}

/// Scratch of the convolutions the decoder and the encoder's DAC branch
/// share: y and padded signals, transposed-convolution columns.
pub(crate) struct Scratch {
    pub y: DeviceBuffer<f32>,
    pub padded: DeviceBuffer<f32>,
    pub columns: DeviceBuffer<f32>,
}

impl Ops {
    /// `output[out][t] = conv(input)` for an input padded by `conv.pad()` on
    /// both sides (row stride `length + 2 * pad`).
    pub fn apply(
        &self,
        conv: &Conv,
        input: DeviceSlice<'_, f32>,
        length: usize,
        mut output: DeviceSliceMut<'_, f32>,
    ) -> Result<(), LaunchError> {
        let (blas, stream) = (&self.blas, &self.stream);
        let stride = length + 2 * conv.pad();
        for t in 0..conv.kernel {
            let weight = t * conv.output * conv.input;
            blas.sgemm_beta(
                stream,
                Op::N,
                Op::N,
                length,
                conv.output,
                conv.input,
                1.0,
                input.range(t * conv.dilation..input.len()),
                stride,
                conv.weight.slice(weight..weight + conv.output * conv.input),
                conv.input,
                if t == 0 { 0.0 } else { 1.0 },
                output.reborrow(),
                length,
            )?;
        }
        k::add_channel_bias(
            stream,
            output,
            length,
            conv.bias.as_slice(),
            conv.output,
            length,
        )
    }

    /// `x[c][t] += unit(x)` over `channels` x `length` (`signal` holds
    /// `channels * length` values).
    pub fn residual(
        &self,
        scratch: &mut Scratch,
        unit: &ResidualUnit,
        signal: &mut DeviceBuffer<f32>,
        channels: usize,
        length: usize,
    ) -> Result<(), LaunchError> {
        let pad = unit.conv1.pad();
        let s = &self.stream;
        let count = channels * length;
        k::snake(
            s,
            signal.slice(0..count),
            length,
            unit.alpha1.as_slice(),
            scratch.padded.slice_mut(0..channels * (length + 2 * pad)),
            channels,
            length,
            pad,
        )?;
        self.apply(
            &unit.conv1,
            scratch.padded.slice(0..channels * (length + 2 * pad)),
            length,
            scratch.y.slice_mut(0..count),
        )?;
        k::snake(
            s,
            scratch.y.slice(0..count),
            length,
            unit.alpha2.as_slice(),
            scratch.padded.slice_mut(0..count),
            channels,
            length,
            0,
        )?;
        self.blas.sgemm_beta(
            s,
            Op::N,
            Op::N,
            length,
            channels,
            channels,
            1.0,
            scratch.padded.slice(0..count),
            length,
            unit.conv2.weight.as_slice(),
            channels,
            1.0,
            signal.slice_mut(0..count),
            length,
        )?;
        k::add_channel_bias(
            s,
            signal.slice_mut(0..count),
            length,
            unit.conv2.bias.as_slice(),
            channels,
            length,
        )
    }

    /// One decoder block: snake, transposed convolution into `output` and the
    /// residual units; returns the new length.
    fn block(
        &self,
        scratch: &mut Scratch,
        block: &DecoderBlock,
        input: &DeviceBuffer<f32>,
        length: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<usize, LaunchError> {
        let up = &block.upsample;
        let kernel = 2 * up.stride;
        let s = &self.stream;
        k::snake(
            s,
            input.slice(0..up.input * length),
            length,
            block.alpha.as_slice(),
            scratch.padded.slice_mut(0..up.input * length),
            up.input,
            length,
            0,
        )?;
        self.blas.sgemm_beta(
            s,
            Op::N,
            Op::N,
            length,
            kernel * up.output,
            up.input,
            1.0,
            scratch.padded.slice(0..up.input * length),
            length,
            up.weight.as_slice(),
            up.input,
            0.0,
            scratch.columns.slice_mut(0..kernel * up.output * length),
            length,
        )?;
        // padding = ceil(stride / 2); output_padding = stride % 2 keeps the
        // length at length * stride.
        let upsampled = length * up.stride;
        k::col2im(
            s,
            scratch.columns.slice(0..kernel * up.output * length),
            up.bias.as_slice(),
            output.slice_mut(0..up.output * upsampled),
            up.output,
            length,
            up.stride,
            kernel,
            up.stride.div_ceil(2),
        )?;
        for unit in &block.units {
            self.residual(scratch, unit, output, up.output, upsampled)?;
        }
        Ok(upsampled)
    }
}

/// The Higgs Audio v2 tokenizer on the GPU in float32: RVQ codes to 24 kHz
/// audio through the DAC decoder, and audio to codes through HuBERT, the
/// semantic and DAC encoders and the RVQ. Not thread-safe.
pub struct Codec {
    pub(crate) config: Config,
    pub(crate) ops: Ops,
    pub(crate) scratch: Scratch,
    /// `[codebooks][entries][hidden]`
    pub(crate) tables: DeviceBuffer<f32>,
    fc2_weight: DeviceBuffer<f32>,
    fc2_bias: DeviceBuffer<f32>,
    input_conv: Conv,
    blocks: Vec<DecoderBlock>,
    final_alpha: DeviceBuffer<f32>,
    output_conv: Conv,
    codes: DeviceIndices,
    quantized: DeviceBuffer<f32>,
    /// The signal between blocks.
    signals: [DeviceBuffer<f32>; 2],
    pub(crate) encoder: Encoder,
}

impl Codec {
    /// Loads `model_root/audio_tokenizer/model.safetensors`.
    pub fn load(config: &Config, model_root: &Path) -> Result<Self, String> {
        Self::load_inner(config, model_root).map_err(|e| format!("OmniVoice audio tokenizer: {e}"))
    }

    fn load_inner(config: &Config, model_root: &Path) -> Result<Self, String> {
        let file =
            SafetensorsFile::open(&model_root.join("audio_tokenizer").join("model.safetensors"))?;
        let codec = &config.codec;
        let acoustic = &codec.acoustic;
        let hidden = codec.semantic.hidden_size + acoustic.latent_channels;
        let stream = Stream::new().map_err(hip)?;
        // Reproducible sums: Blas::new disallows hipBLAS atomics, so the same
        // seed gives the same audio.
        let blas = Blas::new(&stream).map_err(hip)?;
        // RVQ decode tables: codebook entries through each project_out.
        let mut tables = stream
            .alloc::<f32>((codec.codebooks * codec.codebook_size * hidden).max(1))
            .map_err(hip)?;
        for c in 0..codec.codebooks {
            let prefix = format!("quantizer.quantizers.{c}");
            let embed = upload_tensor(
                &stream,
                &file,
                &format!("{prefix}.codebook.embed"),
                &[codec.codebook_size, codec.codebook_dim],
            )?;
            let project = upload_tensor(
                &stream,
                &file,
                &format!("{prefix}.project_out.weight"),
                &[hidden, codec.codebook_dim],
            )?;
            let bias = upload_tensor(
                &stream,
                &file,
                &format!("{prefix}.project_out.bias"),
                &[hidden],
            )?;
            let table = c * codec.codebook_size * hidden;
            let table_len = codec.codebook_size * hidden;
            blas.sgemm(
                &stream,
                Op::T,
                Op::N,
                hidden,
                codec.codebook_size,
                codec.codebook_dim,
                1.0,
                project.as_slice(),
                codec.codebook_dim,
                embed.as_slice(),
                codec.codebook_dim,
                tables.slice_mut(table..table + table_len),
                hidden,
            )
            .map_err(launch)?;
            k::add_column_bias(
                &stream,
                tables.slice_mut(table..table + table_len),
                bias.as_slice(),
                codec.codebook_size,
                hidden,
            )
            .map_err(launch)?;
            stream.synchronize().map_err(hip)?;
        }
        let fc2_weight = upload_tensor(
            &stream,
            &file,
            "fc2.weight",
            &[acoustic.latent_channels, hidden],
        )?;
        let fc2_bias = upload_tensor(&stream, &file, "fc2.bias", &[acoustic.latent_channels])?;
        let input_conv = load_conv(
            &stream,
            &file,
            "acoustic_decoder.conv1",
            acoustic.latent_channels,
            acoustic.decoder_channels,
            RESIDUAL_KERNEL,
            1,
        )?;
        let mut channels = acoustic.decoder_channels;
        let mut blocks = Vec::new();
        for (index, &ratio) in acoustic.upsampling_ratios.iter().enumerate() {
            let prefix = format!("acoustic_decoder.block.{index}");
            let out = channels / 2;
            let alpha = upload_tensor(
                &stream,
                &file,
                &format!("{prefix}.snake1.alpha"),
                &[1, channels, 1],
            )?;
            let upsample = load_conv_transpose(
                &stream,
                &file,
                &format!("{prefix}.conv_t1"),
                channels,
                out,
                ratio,
            )?;
            let mut units = Vec::with_capacity(3);
            for (unit, &dilation) in DILATIONS.iter().enumerate() {
                let name = format!("{prefix}.res_unit{}", unit + 1);
                units.push(ResidualUnit {
                    alpha1: upload_tensor(
                        &stream,
                        &file,
                        &format!("{name}.snake1.alpha"),
                        &[1, out, 1],
                    )?,
                    alpha2: upload_tensor(
                        &stream,
                        &file,
                        &format!("{name}.snake2.alpha"),
                        &[1, out, 1],
                    )?,
                    conv1: load_conv(
                        &stream,
                        &file,
                        &format!("{name}.conv1"),
                        out,
                        out,
                        RESIDUAL_KERNEL,
                        dilation,
                    )?,
                    conv2: load_conv(&stream, &file, &format!("{name}.conv2"), out, out, 1, 1)?,
                });
            }
            blocks.push(DecoderBlock {
                alpha,
                upsample,
                units,
            });
            channels = out;
        }
        let final_alpha = upload_tensor(
            &stream,
            &file,
            "acoustic_decoder.snake1.alpha",
            &[1, channels, 1],
        )?;
        let output_conv = load_conv(
            &stream,
            &file,
            "acoustic_decoder.conv2",
            channels,
            1,
            RESIDUAL_KERNEL,
            1,
        )?;
        let encoder = Encoder::load(config, &file, &stream)?;
        let bound =
            u32::try_from(codec.codebook_size).map_err(|_| "codebook too large".to_string())?;
        let one = |stream: &Stream| stream.alloc::<f32>(1).map_err(hip);
        Ok(Self {
            config: config.clone(),
            tables,
            fc2_weight,
            fc2_bias,
            input_conv,
            blocks,
            final_alpha,
            output_conv,
            codes: stream.alloc_indices(1, bound).map_err(hip)?,
            quantized: one(&stream)?,
            signals: [one(&stream)?, one(&stream)?],
            encoder,
            scratch: Scratch {
                y: one(&stream)?,
                padded: one(&stream)?,
                columns: one(&stream)?,
            },
            ops: Ops { blas, stream },
        })
    }

    fn reserve(&mut self, frames: usize) -> Result<(), HipError> {
        let acoustic = &self.config.codec.acoustic;
        let mut length = frames;
        let mut channels = acoustic.decoder_channels;
        let mut largest = channels * length;
        let mut largest_columns = 0;
        let mut largest_padded = acoustic.latent_channels.max(channels) * (length + 6);
        for &stride in &acoustic.upsampling_ratios {
            let out = channels / 2;
            largest_columns = largest_columns.max(2 * stride * out * length);
            largest_padded = largest_padded.max(channels * length);
            length *= stride;
            channels = out;
            largest = largest.max(channels * length);
            largest_padded = largest_padded.max(channels * (length + 2 * 27));
        }
        let s = &self.ops.stream;
        let codes = self.config.codec.codebooks * frames;
        if self.codes.len() < codes {
            self.codes = s.alloc_indices(codes, self.codes.bound())?;
        }
        reserve(
            s,
            &mut self.quantized,
            frames * (self.config.codec.semantic.hidden_size + acoustic.latent_channels),
        )?;
        for signal in &mut self.signals {
            reserve(s, signal, largest)?;
        }
        reserve(s, &mut self.scratch.y, largest)?;
        reserve(s, &mut self.scratch.padded, largest_padded)?;
        reserve(s, &mut self.scratch.columns, largest_columns)?;
        Ok(())
    }

    /// `codes` is `[codebook][frame]`; yields `frames * frame_samples`
    /// samples.
    pub fn decode(&mut self, codes: &[i32], frames: usize) -> Result<Vec<f32>, String> {
        self.decode_inner(codes, frames)
            .map_err(|e| format!("OmniVoice decode: {e}"))
    }

    fn decode_inner(&mut self, codes: &[i32], frames: usize) -> Result<Vec<f32>, String> {
        let codec = self.config.codec.clone();
        let acoustic = &codec.acoustic;
        let hidden = codec.semantic.hidden_size + acoustic.latent_channels;
        if codes.len() != codec.codebooks * frames {
            return Err("code count doesn't match the frames".to_string());
        }
        let host: Vec<u32> = codes
            .iter()
            .map(|&code| {
                usize::try_from(code)
                    .ok()
                    .filter(|&code| code < codec.codebook_size)
                    .map(|code| code as u32)
                    .ok_or_else(|| "code outside the codebook".to_string())
            })
            .collect::<Result<_, _>>()?;
        self.reserve(frames).map_err(hip)?;
        let pad = self.input_conv.pad();
        let stride = frames + 2 * pad;
        let latent = acoustic.latent_channels;
        let ops = &self.ops;
        let scratch = &mut self.scratch;
        let s = &ops.stream;
        s.upload_indices(&host, &mut self.codes).map_err(hip)?;
        k::codebook_sum(
            s,
            self.tables.as_slice(),
            self.codes.slice(0..host.len()),
            self.quantized.slice_mut(0..frames * hidden),
            codec.codebooks,
            codec.codebook_size,
            frames,
            hidden,
        )
        .map_err(launch)?;
        // fc2 into the input convolution's padded, channel-major buffer.
        s.fill_bytes(scratch.padded.slice_mut(0..latent * stride), 0)
            .map_err(hip)?;
        ops.blas
            .sgemm(
                s,
                Op::T,
                Op::N,
                frames,
                latent,
                hidden,
                1.0,
                self.quantized.slice(0..frames * hidden),
                hidden,
                self.fc2_weight.as_slice(),
                hidden,
                scratch.padded.slice_mut(pad..latent * stride),
                stride,
            )
            .map_err(launch)?;
        k::add_channel_bias(
            s,
            scratch.padded.slice_mut(pad..latent * stride),
            stride,
            self.fc2_bias.as_slice(),
            latent,
            frames,
        )
        .map_err(launch)?;
        let [first, second] = &mut self.signals;
        ops.apply(
            &self.input_conv,
            scratch.padded.slice(0..latent * stride),
            frames,
            first.slice_mut(0..acoustic.decoder_channels * frames),
        )
        .map_err(launch)?;
        let mut length = frames;
        let (mut current, mut other) = (first, second);
        for block in &self.blocks {
            length = ops
                .block(scratch, block, current, length, other)
                .map_err(launch)?;
            std::mem::swap(&mut current, &mut other);
        }
        let output_pad = self.output_conv.pad();
        let channels = self.output_conv.input;
        k::snake(
            s,
            current.slice(0..channels * length),
            length,
            self.final_alpha.as_slice(),
            scratch
                .padded
                .slice_mut(0..channels * (length + 2 * output_pad)),
            channels,
            length,
            output_pad,
        )
        .map_err(launch)?;
        ops.apply(
            &self.output_conv,
            scratch
                .padded
                .slice(0..channels * (length + 2 * output_pad)),
            length,
            scratch.y.slice_mut(0..length),
        )
        .map_err(launch)?;
        let mut audio = vec![0.0f32; length];
        s.download_blocking(scratch.y.slice(0..length), &mut audio)
            .map_err(hip)?;
        Ok(audio)
    }
}
