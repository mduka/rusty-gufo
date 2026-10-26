//! The Base checkpoint's reference encoders: the speech encoder that turns
//! reference audio into codec codes (`hip/speech_encoder_runtime.cpp`) and the
//! ECAPA-TDNN speaker encoder (`hip/speaker_encoder_runtime.cpp`), with the
//! shared weight conversion and rocBLAS GEMM of `hip/encoder_support.cpp`.
//! Every tensor is rounded to BF16 where the C++ code rounds it.

use gufo_hip::blas::{Op, RocBlas};
use gufo_hip::{DeviceBuffer, DeviceSlice, DeviceSliceMut, LaunchError, Stream};
use gufo_qwen3_tts_kernels as k;

use crate::audio::{AudioBuffer, resample_mono};
use crate::bf16::{bf16_to_f32, round_bf16};
use crate::checkpoint::{Checkpoint, DType, Tensor};
use crate::config::ModelVariant;
use crate::features::{MEL_BINS, compute_features};

const SAMPLE_RATE: u32 = 24000;
const HIDDEN: usize = 512;
const QUANTIZER_DIMENSION: usize = 256;
const CODEBOOK_SIZE: usize = 2048;
const CODE_GROUPS: usize = 16;

const PADDING_ZERO: u32 = 0;
const PADDING_REPLICATE: u32 = 1;
const PADDING_REFLECT: u32 = 2;

fn launch(error: LaunchError) -> String {
    format!("Qwen3-TTS encoder: {error}")
}

fn hip(operation: &str) -> impl Fn(gufo_hip::HipError) -> String + '_ {
    move |error| format!("{operation}: {error}")
}

fn alloc(stream: &Stream, len: usize) -> Result<DeviceBuffer<f32>, String> {
    stream
        .alloc_persistent::<f32>(len)
        .map_err(hip("hipMalloc Qwen3-TTS encoder"))
}

/// `ReadEncoderTensor`: f32 or widened BF16 values.
fn read_tensor(checkpoint: &Checkpoint, tensor: &Tensor, elements: usize) -> Vec<f32> {
    let bytes = checkpoint.bytes(tensor);
    if tensor.dtype == DType::F32 {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .take(elements)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    } else {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .take(elements)
            .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect()
    }
}

/// `EncoderWeights`: each tensor rounded to BF16 on the host and uploaded to
/// its own allocation.
struct EncoderWeights<'a> {
    stream: &'a Stream,
    checkpoint: &'a Checkpoint,
}

impl EncoderWeights<'_> {
    fn upload(&self, host: &[f32]) -> Result<DeviceBuffer<f32>, String> {
        let mut buffer = alloc(self.stream, host.len())?;
        self.stream
            .upload(host, buffer.as_mut_slice())
            .map_err(hip("copy Qwen3-TTS encoder weight"))?;
        Ok(buffer)
    }

    /// `EncoderWeights::Load`.
    fn load(&self, name: &str, shape: &[u64]) -> Result<DeviceBuffer<f32>, String> {
        let tensor = self
            .checkpoint
            .find(name)
            .filter(|t| t.shape == shape && matches!(t.dtype, DType::F32 | DType::Bf16))
            .ok_or_else(|| format!("unexpected Qwen3-TTS encoder tensor: {name}"))?;
        let elements = tensor
            .elements()
            .ok_or_else(|| format!("invalid Qwen3-TTS encoder tensor size: {name}"))?;
        let host: Vec<f32> = read_tensor(self.checkpoint, tensor, elements)
            .into_iter()
            .map(round_bf16)
            .collect();
        self.upload(&host)
    }

    /// `LoadBfloat16CodebookEmbedding`: `[dimension][codebook_size]` centroids
    /// `round(round(sum) / max(round(usage), 1e-5))`.
    fn codebook(
        &self,
        embedding_sum_name: &str,
        cluster_usage_name: &str,
    ) -> Result<DeviceBuffer<f32>, String> {
        let valid = |tensor: &&Tensor, shape: &[u64]| {
            tensor.shape == shape && matches!(tensor.dtype, DType::F32 | DType::Bf16)
        };
        let (Some(sums), Some(usage)) = (
            self.checkpoint
                .find(embedding_sum_name)
                .filter(|t| valid(t, &[CODEBOOK_SIZE as u64, QUANTIZER_DIMENSION as u64])),
            self.checkpoint
                .find(cluster_usage_name)
                .filter(|t| valid(t, &[CODEBOOK_SIZE as u64])),
        ) else {
            return Err("unexpected Qwen3-TTS encoder codebook tensors".into());
        };
        let sums = read_tensor(self.checkpoint, sums, CODEBOOK_SIZE * QUANTIZER_DIMENSION);
        let usage = read_tensor(self.checkpoint, usage, CODEBOOK_SIZE);
        let mut embedding = vec![0.0f32; sums.len()];
        for code in 0..CODEBOOK_SIZE {
            let denominator = round_bf16(usage[code]).max(1.0e-5);
            for column in 0..QUANTIZER_DIMENSION {
                embedding[column * CODEBOOK_SIZE + code] =
                    round_bf16(round_bf16(sums[code * QUANTIZER_DIMENSION + column]) / denominator);
            }
        }
        self.upload(&embedding)
    }
}

/// `EncoderF32Gemm::Run`.
#[allow(clippy::too_many_arguments)]
fn gemm_f32(
    stream: &Stream,
    gemm: &RocBlas,
    weight: DeviceSlice<'_, f32>,
    input: DeviceSlice<'_, f32>,
    output: DeviceSliceMut<'_, f32>,
    rows: usize,
    output_columns: usize,
    reduction: usize,
) -> Result<(), String> {
    let maximum = i32::MAX as usize;
    if rows == 0
        || output_columns == 0
        || reduction == 0
        || rows > maximum
        || output_columns > maximum
        || reduction > maximum
    {
        return Err("invalid Qwen3-TTS encoder GEMM shape".into());
    }
    gemm.gemm_f32(
        stream,
        Op::T,
        Op::N,
        output_columns,
        rows,
        reduction,
        weight,
        reduction,
        input,
        reduction,
        output,
        output_columns,
    )
    .map_err(launch)
}

fn round(stream: &Stream, values: DeviceSliceMut<'_, f32>, elements: usize) -> Result<(), String> {
    k::round_bf16(stream, values.range(0..elements), elements).map_err(launch)
}

struct Conv {
    weight: DeviceBuffer<f32>,
    bias: Option<DeviceBuffer<f32>>,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    padding: u32,
}

struct Linear {
    weight: DeviceBuffer<f32>,
    input_columns: usize,
    output_columns: usize,
}

struct TransformerWeights {
    norm1_weight: DeviceBuffer<f32>,
    norm1_bias: DeviceBuffer<f32>,
    norm2_weight: DeviceBuffer<f32>,
    norm2_bias: DeviceBuffer<f32>,
    attention_scale: DeviceBuffer<f32>,
    mlp_scale: DeviceBuffer<f32>,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    fc1: Linear,
    fc2: Linear,
}

struct ResidualWeights {
    first: Conv,
    second: Conv,
}

/// Frame-major codec codes `[frames, code_groups]` of the reference audio.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeechEncoderOutput {
    pub codes: Vec<u32>,
    pub frames: usize,
    pub code_groups: usize,
}

/// One captured SEANet layer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeechEncoderLayerTrace {
    pub values: Vec<f32>,
    pub channels: usize,
    pub frames: usize,
}

/// Frame-major stage captures (`SpeechEncoderTrace`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeechEncoderTrace {
    pub layers: Vec<SpeechEncoderLayerTrace>,
    pub convolutional: Vec<f32>,
    pub transformer: Vec<f32>,
    pub downsample: Vec<f32>,
    pub semantic_projection: Vec<f32>,
    pub acoustic_projection: Vec<f32>,
    pub convolutional_frames: usize,
    pub output_frames: usize,
}

/// `SpeechEncoderHipRuntime`.
pub struct SpeechEncoder {
    stream: Stream,
    gemm: RocBlas,
    encoder_norm_eps: f32,
    encoder_heads: usize,
    encoder_head_dim: usize,
    encoder_rope_theta: f32,
    encoder_sliding_window: usize,
    convolutions: Vec<Conv>,
    residuals: Vec<ResidualWeights>,
    transformers: Vec<TransformerWeights>,
    downsample: Conv,
    semantic_projection: Conv,
    acoustic_projection: Conv,
    semantic_codebook: DeviceBuffer<f32>,
    acoustic_codebooks: Vec<DeviceBuffer<f32>>,
}

fn ceil_divide(value: usize, divisor: usize) -> usize {
    value.div_ceil(divisor)
}

/// The buffers of one encoding (`Workspace`).
struct SpeechWorkspace {
    first: DeviceBuffer<f32>,
    second: DeviceBuffer<f32>,
    third: DeviceBuffer<f32>,
    fourth: DeviceBuffer<f32>,
    columns: DeviceBuffer<f32>,
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    attention: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    ffn: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    acoustic: DeviceBuffer<f32>,
    codes: DeviceBuffer<u32>,
}

impl SpeechEncoder {
    /// `SpeechEncoderHipRuntime::Create`.
    pub fn new(checkpoint: &Checkpoint) -> Result<Self, String> {
        if checkpoint.config.variant != ModelVariant::Base {
            return Err("Qwen3-TTS speech encoding requires a Base model".into());
        }
        let stream = Stream::new().map_err(hip("hipStreamCreate Qwen3-TTS speech encoder"))?;
        let weights = EncoderWeights {
            stream: &stream,
            checkpoint,
        };
        let conv = |prefix: &str,
                    input_channels: usize,
                    output_channels: usize,
                    kernel: usize,
                    stride: usize,
                    bias: bool,
                    padding: u32|
         -> Result<Conv, String> {
            Ok(Conv {
                weight: weights.load(
                    &format!("{prefix}.weight"),
                    &[output_channels as u64, input_channels as u64, kernel as u64],
                )?,
                bias: if bias {
                    Some(weights.load(&format!("{prefix}.bias"), &[output_channels as u64])?)
                } else {
                    None
                },
                input_channels,
                output_channels,
                kernel,
                stride,
                dilation: 1,
                padding,
            })
        };
        let linear = |prefix: &str, input_columns: usize, output_columns: usize| {
            Ok::<_, String>(Linear {
                weight: weights.load(
                    &format!("{prefix}.weight"),
                    &[output_columns as u64, input_columns as u64],
                )?,
                input_columns,
                output_columns,
            })
        };
        let vector = |name: &str, elements: usize| weights.load(name, &[elements as u64]);

        let names = [
            "encoder.encoder.layers.0.conv",
            "encoder.encoder.layers.3.conv",
            "encoder.encoder.layers.6.conv",
            "encoder.encoder.layers.9.conv",
            "encoder.encoder.layers.12.conv",
            "encoder.encoder.layers.14.conv",
        ];
        let shapes = [
            [1, 64, 7, 1],
            [64, 128, 8, 4],
            [128, 256, 10, 5],
            [256, 512, 12, 6],
            [512, 1024, 16, 8],
            [1024, 512, 3, 1],
        ];
        let mut convolutions = Vec::new();
        for (name, shape) in names.iter().zip(shapes) {
            convolutions.push(conv(
                name,
                shape[0],
                shape[1],
                shape[2],
                shape[3],
                true,
                PADDING_ZERO,
            )?);
        }
        let mut residuals = Vec::new();
        for (index, channels) in [(1, 64), (4, 128), (7, 256), (10, 512)] {
            let prefix = format!("encoder.encoder.layers.{index}.block.");
            residuals.push(ResidualWeights {
                first: conv(
                    &format!("{prefix}1.conv"),
                    channels,
                    channels / 2,
                    3,
                    1,
                    true,
                    PADDING_ZERO,
                )?,
                second: conv(
                    &format!("{prefix}3.conv"),
                    channels / 2,
                    channels,
                    1,
                    1,
                    true,
                    PADDING_ZERO,
                )?,
            });
        }
        let mut transformers = Vec::new();
        for layer in 0..8 {
            let prefix = format!("encoder.encoder_transformer.layers.{layer}");
            transformers.push(TransformerWeights {
                norm1_weight: vector(&format!("{prefix}.input_layernorm.weight"), 512)?,
                norm1_bias: vector(&format!("{prefix}.input_layernorm.bias"), 512)?,
                norm2_weight: vector(&format!("{prefix}.post_attention_layernorm.weight"), 512)?,
                norm2_bias: vector(&format!("{prefix}.post_attention_layernorm.bias"), 512)?,
                attention_scale: vector(&format!("{prefix}.self_attn_layer_scale.scale"), 512)?,
                mlp_scale: vector(&format!("{prefix}.mlp_layer_scale.scale"), 512)?,
                query: linear(&format!("{prefix}.self_attn.q_proj"), 512, 512)?,
                key: linear(&format!("{prefix}.self_attn.k_proj"), 512, 512)?,
                value: linear(&format!("{prefix}.self_attn.v_proj"), 512, 512)?,
                output: linear(&format!("{prefix}.self_attn.o_proj"), 512, 512)?,
                fc1: linear(&format!("{prefix}.mlp.fc1"), 512, 2048)?,
                fc2: linear(&format!("{prefix}.mlp.fc2"), 2048, 512)?,
            });
        }
        let downsample = conv(
            "encoder.downsample.conv",
            512,
            512,
            4,
            2,
            false,
            PADDING_REPLICATE,
        )?;
        let semantic_projection = conv(
            "encoder.quantizer.semantic_residual_vector_quantizer.input_proj",
            512,
            256,
            1,
            1,
            false,
            PADDING_ZERO,
        )?;
        let acoustic_projection = conv(
            "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj",
            512,
            256,
            1,
            1,
            false,
            PADDING_ZERO,
        )?;
        let semantic_codebook = weights.codebook(
            "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum",
            "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.cluster_usage",
        )?;
        let mut acoustic_codebooks = Vec::new();
        for layer in 0..CODE_GROUPS - 1 {
            let prefix = format!(
                "encoder.quantizer.acoustic_residual_vector_quantizer.layers.{layer}.codebook."
            );
            acoustic_codebooks.push(weights.codebook(
                &format!("{prefix}embed_sum"),
                &format!("{prefix}cluster_usage"),
            )?);
        }
        let tokenizer = &checkpoint.config.speech_tokenizer;
        let gemm = RocBlas::new(&stream).map_err(hip("rocblas_create_handle"))?;
        Ok(Self {
            gemm,
            encoder_norm_eps: tokenizer.encoder_norm_eps,
            encoder_heads: tokenizer.encoder_num_attention_heads as usize,
            encoder_head_dim: tokenizer.encoder_head_dim as usize,
            encoder_rope_theta: tokenizer.encoder_rope_theta,
            encoder_sliding_window: tokenizer.encoder_sliding_window as usize,
            convolutions,
            residuals,
            transformers,
            downsample,
            semantic_projection,
            acoustic_projection,
            semantic_codebook,
            acoustic_codebooks,
            stream,
        })
    }

    /// `RunConv`: im2col with the convolution's padding, GEMM, bias, BF16;
    /// returns the output length.
    fn run_conv(
        &self,
        conv: &Conv,
        input: DeviceSlice<'_, f32>,
        input_length: usize,
        mut output: DeviceSliceMut<'_, f32>,
        columns: &mut DeviceBuffer<f32>,
    ) -> Result<usize, String> {
        let stream = &self.stream;
        let output_length = ceil_divide(input_length, conv.stride);
        let effective_kernel = (conv.kernel - 1) * conv.dilation + 1;
        let left_padding = effective_kernel - conv.stride;
        let reduction = conv.input_channels * conv.kernel;
        k::encoder_conv1d_im2col(
            stream,
            input.range(0..input_length * conv.input_channels),
            columns.slice_mut(0..output_length * reduction),
            input_length,
            conv.input_channels,
            output_length,
            conv.kernel,
            conv.stride,
            conv.dilation,
            left_padding,
            conv.padding,
        )
        .map_err(launch)?;
        gemm_f32(
            stream,
            &self.gemm,
            conv.weight.as_slice(),
            columns.as_slice(),
            output.reborrow(),
            output_length,
            conv.output_channels,
            reduction,
        )?;
        let elements = output_length * conv.output_channels;
        if let Some(bias) = &conv.bias {
            k::add_bias(
                stream,
                output.reborrow().range(0..elements),
                bias.as_slice(),
                output_length,
                conv.output_channels,
            )
            .map_err(launch)?;
        }
        round(stream, output, elements)?;
        Ok(output_length)
    }

    /// `RunLinear`: GEMM, then BF16.
    fn run_linear(
        &self,
        linear: &Linear,
        input: DeviceSlice<'_, f32>,
        mut output: DeviceSliceMut<'_, f32>,
        rows: usize,
    ) -> Result<(), String> {
        gemm_f32(
            &self.stream,
            &self.gemm,
            linear.weight.as_slice(),
            input,
            output.reborrow(),
            rows,
            linear.output_columns,
            linear.input_columns,
        )?;
        round(&self.stream, output, rows * linear.output_columns)
    }

    /// `MakeWorkspace`: the largest buffers the encoding of `samples`
    /// samples needs.
    fn make_workspace(&self, samples: usize) -> Result<SpeechWorkspace, String> {
        let mut maximum_main = samples;
        let mut maximum_columns = 0usize;
        let mut length = samples;
        let include = |conv: &Conv, length: &mut usize, main: &mut usize, cols: &mut usize| {
            let output_length = ceil_divide(*length, conv.stride);
            *cols = (*cols).max(output_length * conv.input_channels * conv.kernel);
            *main = (*main).max(output_length * conv.output_channels);
            *length = output_length;
        };
        include(
            &self.convolutions[0],
            &mut length,
            &mut maximum_main,
            &mut maximum_columns,
        );
        for (index, residual) in self.residuals.iter().enumerate() {
            maximum_columns =
                maximum_columns.max(length * residual.first.input_channels * residual.first.kernel);
            maximum_main = maximum_main.max(length * residual.first.output_channels);
            maximum_columns = maximum_columns
                .max(length * residual.second.input_channels * residual.second.kernel);
            include(
                &self.convolutions[index + 1],
                &mut length,
                &mut maximum_main,
                &mut maximum_columns,
            );
        }
        include(
            &self.convolutions[self.convolutions.len() - 1],
            &mut length,
            &mut maximum_main,
            &mut maximum_columns,
        );
        let transformer_frames = length;
        include(
            &self.downsample,
            &mut length,
            &mut maximum_main,
            &mut maximum_columns,
        );
        let output_frames = length;
        maximum_columns = maximum_columns.max(output_frames * 512);
        let stream = &self.stream;
        Ok(SpeechWorkspace {
            first: alloc(stream, maximum_main)?,
            second: alloc(stream, maximum_main)?,
            third: alloc(stream, maximum_main)?,
            fourth: alloc(stream, maximum_main)?,
            columns: alloc(stream, maximum_columns)?,
            query: alloc(stream, transformer_frames * HIDDEN)?,
            key: alloc(stream, transformer_frames * HIDDEN)?,
            value: alloc(stream, transformer_frames * HIDDEN)?,
            attention: alloc(stream, transformer_frames * HIDDEN)?,
            normalized: alloc(stream, transformer_frames * HIDDEN)?,
            ffn: alloc(stream, transformer_frames * 2048)?,
            projected: alloc(stream, output_frames * QUANTIZER_DIMENSION)?,
            acoustic: alloc(stream, output_frames * QUANTIZER_DIMENSION)?,
            codes: stream
                .alloc_persistent::<u32>(output_frames * CODE_GROUPS)
                .map_err(hip("hipMalloc Qwen3-TTS encoder"))?,
        })
    }

    fn capture_layer(
        &self,
        values: &DeviceBuffer<f32>,
        channels: usize,
        frames: usize,
        trace: Option<&mut SpeechEncoderTrace>,
    ) -> Result<(), String> {
        let Some(trace) = trace else {
            return Ok(());
        };
        let mut host = vec![0.0f32; channels * frames];
        self.stream
            .download(values.slice(0..host.len()), &mut host)
            .map_err(hip("copy Qwen3-TTS speech encoder layer trace"))?;
        trace.layers.push(SpeechEncoderLayerTrace {
            values: host,
            channels,
            frames,
        });
        Ok(())
    }

    fn download(&self, values: &DeviceBuffer<f32>, elements: usize) -> Result<Vec<f32>, String> {
        let mut host = vec![0.0f32; elements];
        self.stream
            .download(values.slice(0..elements), &mut host)
            .map_err(hip("copy Qwen3-TTS speech encoder trace"))?;
        Ok(host)
    }

    /// `SpeechEncoderHipRuntime::Encode`.
    pub fn encode(
        &mut self,
        audio: &AudioBuffer,
        mut trace: Option<&mut SpeechEncoderTrace>,
    ) -> Result<SpeechEncoderOutput, String> {
        if let Some(trace) = trace.as_deref_mut() {
            *trace = SpeechEncoderTrace::default();
        }
        let waveform = resample_mono(audio, SAMPLE_RATE)?;
        if waveform.is_empty() {
            return Err("Qwen3-TTS speech encoder requires reference audio".into());
        }
        let mut w = self.make_workspace(waveform.len())?;
        let stream = &self.stream;
        stream
            .upload(&waveform, w.first.slice_mut(0..waveform.len()))
            .map_err(hip("copy Qwen3-TTS reference waveform"))?;
        round(stream, w.first.as_mut_slice(), waveform.len())?;

        let mut length = self.run_conv(
            &self.convolutions[0],
            w.first.as_slice(),
            waveform.len(),
            w.second.as_mut_slice(),
            &mut w.columns,
        )?;
        self.capture_layer(
            &w.second,
            self.convolutions[0].output_channels,
            length,
            trace.as_deref_mut(),
        )?;
        for (index, residual) in self.residuals.iter().enumerate() {
            let input_elements = length * residual.first.input_channels;
            stream
                .copy(
                    w.second.slice(0..input_elements),
                    w.third.slice_mut(0..input_elements),
                )
                .map_err(hip("copy Qwen3-TTS speech residual"))?;
            k::elu(stream, w.third.slice_mut(0..input_elements), input_elements).map_err(launch)?;
            round(stream, w.third.as_mut_slice(), input_elements)?;
            self.run_conv(
                &residual.first,
                w.third.as_slice(),
                length,
                w.fourth.as_mut_slice(),
                &mut w.columns,
            )?;
            let middle = length * residual.first.output_channels;
            k::elu(stream, w.fourth.slice_mut(0..middle), middle).map_err(launch)?;
            round(stream, w.fourth.as_mut_slice(), middle)?;
            self.run_conv(
                &residual.second,
                w.fourth.as_slice(),
                length,
                w.third.as_mut_slice(),
                &mut w.columns,
            )?;
            let output_elements = length * residual.second.output_channels;
            k::add(
                stream,
                w.second.slice(0..output_elements),
                w.third.slice(0..output_elements),
                w.first.slice_mut(0..output_elements),
                output_elements,
            )
            .map_err(launch)?;
            round(stream, w.first.as_mut_slice(), output_elements)?;
            self.capture_layer(
                &w.first,
                residual.second.output_channels,
                length,
                trace.as_deref_mut(),
            )?;
            k::elu(
                stream,
                w.first.slice_mut(0..output_elements),
                output_elements,
            )
            .map_err(launch)?;
            round(stream, w.first.as_mut_slice(), output_elements)?;
            self.capture_layer(
                &w.first,
                residual.second.output_channels,
                length,
                trace.as_deref_mut(),
            )?;
            let next = &self.convolutions[index + 1];
            length = self.run_conv(
                next,
                w.first.as_slice(),
                length,
                w.second.as_mut_slice(),
                &mut w.columns,
            )?;
            self.capture_layer(
                &w.second,
                next.output_channels,
                length,
                trace.as_deref_mut(),
            )?;
        }
        k::elu(stream, w.second.slice_mut(0..length * 1024), length * 1024).map_err(launch)?;
        round(stream, w.second.as_mut_slice(), length * 1024)?;
        self.capture_layer(&w.second, 1024, length, trace.as_deref_mut())?;
        let last = &self.convolutions[self.convolutions.len() - 1];
        length = self.run_conv(
            last,
            w.second.as_slice(),
            length,
            w.first.as_mut_slice(),
            &mut w.columns,
        )?;
        self.capture_layer(&w.first, last.output_channels, length, trace.as_deref_mut())?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.convolutional_frames = length;
            trace.convolutional = self.download(&w.first, length * HIDDEN)?;
        }

        // `current` is `first` through the transformer.
        let heads = self.encoder_heads;
        let head_dim = self.encoder_head_dim;
        let elements = length * HIDDEN;
        for layer in &self.transformers {
            k::layer_norm(
                stream,
                w.first.slice(0..elements),
                layer.norm1_weight.as_slice(),
                layer.norm1_bias.as_slice(),
                w.normalized.slice_mut(0..elements),
                length,
                HIDDEN,
                self.encoder_norm_eps,
            )
            .map_err(launch)?;
            round(stream, w.normalized.as_mut_slice(), elements)?;
            self.run_linear(
                &layer.query,
                w.normalized.as_slice(),
                w.query.as_mut_slice(),
                length,
            )?;
            self.run_linear(
                &layer.key,
                w.normalized.as_slice(),
                w.key.as_mut_slice(),
                length,
            )?;
            self.run_linear(
                &layer.value,
                w.normalized.as_slice(),
                w.value.as_mut_slice(),
                length,
            )?;
            k::rope(
                stream,
                w.query.slice_mut(0..length * heads * head_dim),
                w.key.slice_mut(0..length * heads * head_dim),
                length,
                heads,
                head_dim,
                self.encoder_rope_theta,
                0,
            )
            .map_err(launch)?;
            round(stream, w.query.as_mut_slice(), elements)?;
            round(stream, w.key.as_mut_slice(), elements)?;
            // The encoder transformer is windowed, not fully causal.
            k::sliding_causal_attention(
                stream,
                w.query.slice(0..length * heads * head_dim),
                w.key.slice(0..length * heads * head_dim),
                w.value.slice(0..length * heads * head_dim),
                w.attention.slice_mut(0..length * heads * head_dim),
                length,
                heads,
                head_dim,
                self.encoder_sliding_window,
                0,
                length,
            )
            .map_err(launch)?;
            round(stream, w.attention.as_mut_slice(), elements)?;
            self.run_linear(
                &layer.output,
                w.attention.as_slice(),
                w.second.as_mut_slice(),
                length,
            )?;
            k::add_scaled_channels(
                stream,
                w.first.slice_mut(0..elements),
                w.second.slice(0..elements),
                layer.attention_scale.as_slice(),
                length,
                HIDDEN,
            )
            .map_err(launch)?;
            round(stream, w.first.as_mut_slice(), elements)?;
            k::layer_norm(
                stream,
                w.first.slice(0..elements),
                layer.norm2_weight.as_slice(),
                layer.norm2_bias.as_slice(),
                w.normalized.slice_mut(0..elements),
                length,
                HIDDEN,
                self.encoder_norm_eps,
            )
            .map_err(launch)?;
            round(stream, w.normalized.as_mut_slice(), elements)?;
            self.run_linear(
                &layer.fc1,
                w.normalized.as_slice(),
                w.ffn.as_mut_slice(),
                length,
            )?;
            k::gelu(stream, w.ffn.slice_mut(0..length * 2048), length * 2048).map_err(launch)?;
            round(stream, w.ffn.as_mut_slice(), length * 2048)?;
            self.run_linear(
                &layer.fc2,
                w.ffn.as_slice(),
                w.second.as_mut_slice(),
                length,
            )?;
            k::add_scaled_channels(
                stream,
                w.first.slice_mut(0..elements),
                w.second.slice(0..elements),
                layer.mlp_scale.as_slice(),
                length,
                HIDDEN,
            )
            .map_err(launch)?;
            round(stream, w.first.as_mut_slice(), elements)?;
        }
        if let Some(trace) = trace.as_deref_mut() {
            trace.transformer = self.download(&w.first, elements)?;
        }

        length = self.run_conv(
            &self.downsample,
            w.first.as_slice(),
            length,
            w.second.as_mut_slice(),
            &mut w.columns,
        )?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.output_frames = length;
            trace.downsample = self.download(&w.second, length * HIDDEN)?;
        }
        self.run_conv(
            &self.semantic_projection,
            w.second.as_slice(),
            length,
            w.projected.as_mut_slice(),
            &mut w.columns,
        )?;
        self.run_conv(
            &self.acoustic_projection,
            w.second.as_slice(),
            length,
            w.acoustic.as_mut_slice(),
            &mut w.columns,
        )?;
        let projected = length * QUANTIZER_DIMENSION;
        round(stream, w.projected.as_mut_slice(), projected)?;
        round(stream, w.acoustic.as_mut_slice(), projected)?;
        if let Some(trace) = trace {
            trace.semantic_projection = self.download(&w.projected, projected)?;
            trace.acoustic_projection = self.download(&w.acoustic, projected)?;
        }
        let quantize = |residual: &mut DeviceBuffer<f32>,
                        codebook: &DeviceBuffer<f32>,
                        codes: &mut DeviceBuffer<u32>,
                        group: usize,
                        update: bool| {
            k::quantize_codebook(
                stream,
                residual.slice_mut(0..projected),
                codebook.as_slice(),
                codes.slice_mut(0..length * CODE_GROUPS),
                length,
                QUANTIZER_DIMENSION,
                CODEBOOK_SIZE,
                CODE_GROUPS,
                group,
                update,
            )
            .map_err(launch)
        };
        quantize(
            &mut w.projected,
            &self.semantic_codebook,
            &mut w.codes,
            0,
            false,
        )?;
        stream
            .copy(
                w.acoustic.slice(0..projected),
                w.projected.slice_mut(0..projected),
            )
            .map_err(hip("copy Qwen3-TTS acoustic residual"))?;
        for group in 1..CODE_GROUPS {
            quantize(
                &mut w.projected,
                &self.acoustic_codebooks[group - 1],
                &mut w.codes,
                group,
                true,
            )?;
        }
        let mut codes = vec![0u32; length * CODE_GROUPS];
        stream
            .download(w.codes.slice(0..codes.len()), &mut codes)
            .map_err(hip("copy Qwen3-TTS speech codes"))?;
        Ok(SpeechEncoderOutput {
            codes,
            frames: length,
            code_groups: CODE_GROUPS,
        })
    }
}

// --- Speaker encoder -----------------------------------------------------

const CHANNELS: usize = 512;
const SCALE: usize = 8;
const WIDTH: usize = CHANNELS / SCALE;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeakerEncoderOutput {
    pub embedding: Vec<f32>,
}

/// Frame-major stage captures (`SpeakerEncoderTrace`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeakerEncoderTrace {
    pub frames: usize,
    pub features: Vec<f32>,
    pub initial: Vec<f32>,
    pub block1: Vec<f32>,
    pub block2: Vec<f32>,
    pub block3: Vec<f32>,
    pub aggregate: Vec<f32>,
    pub pooled: Vec<f32>,
    pub embedding: Vec<f32>,
}

struct Res2Net {
    tdnn1: Conv,
    blocks: Vec<Conv>,
    tdnn2: Conv,
    se1: Conv,
    se2: Conv,
}

/// The scratch of one speaker encoding (`Workspace`).
struct SpeakerWorkspace {
    first: DeviceBuffer<f32>,
    second: DeviceBuffer<f32>,
    third: DeviceBuffer<f32>,
    columns: DeviceBuffer<f32>,
    chunk: DeviceBuffer<f32>,
    previous: DeviceBuffer<f32>,
    combined: DeviceBuffer<f32>,
    channel_vector: DeviceBuffer<f32>,
}

/// `SpeakerEncoderHipRuntime`.
pub struct SpeakerEncoder {
    stream: Stream,
    gemm: RocBlas,
    block0: Conv,
    blocks: Vec<Res2Net>,
    mfa: Conv,
    asp_tdnn: Conv,
    asp_conv: Conv,
    fc: Conv,
}

impl SpeakerEncoder {
    /// `SpeakerEncoderHipRuntime::Create`.
    pub fn new(checkpoint: &Checkpoint) -> Result<Self, String> {
        if checkpoint.config.variant != ModelVariant::Base
            || checkpoint.config.speaker_encoder.is_none()
        {
            return Err("Qwen3-TTS speaker encoding requires a Base model".into());
        }
        let stream = Stream::new().map_err(hip("hipStreamCreate Qwen3-TTS speaker encoder"))?;
        let weights = EncoderWeights {
            stream: &stream,
            checkpoint,
        };
        let conv = |suffix: &str,
                    input_channels: usize,
                    output_channels: usize,
                    kernel: usize,
                    dilation: usize|
         -> Result<Conv, String> {
            let prefix = format!("speaker_encoder.{suffix}");
            Ok(Conv {
                weight: weights.load(
                    &format!("{prefix}.weight"),
                    &[output_channels as u64, input_channels as u64, kernel as u64],
                )?,
                bias: Some(weights.load(&format!("{prefix}.bias"), &[output_channels as u64])?),
                input_channels,
                output_channels,
                kernel,
                stride: 1,
                dilation,
                padding: PADDING_REFLECT,
            })
        };
        let block0 = conv("blocks.0.conv", 128, 512, 5, 1)?;
        let mut blocks = Vec::new();
        for block in 1..=3 {
            let prefix = format!("blocks.{block}");
            let tdnn1 = conv(&format!("{prefix}.tdnn1.conv"), 512, 512, 1, 1)?;
            let tdnn2 = conv(&format!("{prefix}.tdnn2.conv"), 512, 512, 1, 1)?;
            let se1 = conv(&format!("{prefix}.se_block.conv1"), 512, 128, 1, 1)?;
            let se2 = conv(&format!("{prefix}.se_block.conv2"), 128, 512, 1, 1)?;
            let mut res2net_blocks = Vec::new();
            for index in 0..7 {
                res2net_blocks.push(conv(
                    &format!("{prefix}.res2net_block.blocks.{index}.conv"),
                    64,
                    64,
                    3,
                    block + 1,
                )?);
            }
            blocks.push(Res2Net {
                tdnn1,
                blocks: res2net_blocks,
                tdnn2,
                se1,
                se2,
            });
        }
        let mfa = conv("mfa.conv", 1536, 1536, 1, 1)?;
        let asp_tdnn = conv("asp.tdnn.conv", 4608, 128, 1, 1)?;
        let asp_conv = conv("asp.conv", 128, 1536, 1, 1)?;
        let fc = conv("fc", 3072, 2048, 1, 1)?;
        let gemm = RocBlas::new(&stream).map_err(hip("rocblas_create_handle"))?;
        Ok(Self {
            gemm,
            block0,
            blocks,
            mfa,
            asp_tdnn,
            asp_conv,
            fc,
            stream,
        })
    }

    /// `RunConv`: reflected same-length convolution, bias, BF16.
    fn run_conv(
        &self,
        conv: &Conv,
        input: DeviceSlice<'_, f32>,
        rows: usize,
        mut output: DeviceSliceMut<'_, f32>,
        columns: &mut DeviceBuffer<f32>,
    ) -> Result<(), String> {
        let stream = &self.stream;
        let padding = conv.dilation * (conv.kernel - 1) / 2;
        let reduction = conv.input_channels * conv.kernel;
        k::encoder_conv1d_im2col(
            stream,
            input.range(0..rows * conv.input_channels),
            columns.slice_mut(0..rows * reduction),
            rows,
            conv.input_channels,
            rows,
            conv.kernel,
            1,
            conv.dilation,
            padding,
            PADDING_REFLECT,
        )
        .map_err(launch)?;
        gemm_f32(
            stream,
            &self.gemm,
            conv.weight.as_slice(),
            columns.as_slice(),
            output.reborrow(),
            rows,
            conv.output_channels,
            reduction,
        )?;
        let elements = rows * conv.output_channels;
        if let Some(bias) = &conv.bias {
            k::add_bias(
                stream,
                output.reborrow().range(0..elements),
                bias.as_slice(),
                rows,
                conv.output_channels,
            )
            .map_err(launch)?;
        }
        round(stream, output, elements)
    }

    /// `Tdnn`: convolution, then ReLU.
    fn tdnn(
        &self,
        conv: &Conv,
        input: DeviceSlice<'_, f32>,
        rows: usize,
        mut output: DeviceSliceMut<'_, f32>,
        columns: &mut DeviceBuffer<f32>,
    ) -> Result<(), String> {
        self.run_conv(conv, input, rows, output.reborrow(), columns)?;
        let elements = rows * conv.output_channels;
        k::relu(&self.stream, output.range(0..elements), elements).map_err(launch)
    }

    /// `RunBlock`: one SE-Res2Net block. `input` stays separate from the
    /// block scratch, which the residual add reads it after.
    fn run_block(
        &self,
        block: &Res2Net,
        input: &DeviceBuffer<f32>,
        rows: usize,
        output: &mut DeviceBuffer<f32>,
        w: &mut SpeakerWorkspace,
    ) -> Result<(), String> {
        let stream = &self.stream;
        self.tdnn(
            &block.tdnn1,
            input.as_slice(),
            rows,
            w.first.as_mut_slice(),
            &mut w.columns,
        )?;
        for split in 0..SCALE {
            k::extract_channels(
                stream,
                w.first.slice(0..rows * CHANNELS),
                w.chunk.slice_mut(0..rows * WIDTH),
                rows,
                CHANNELS,
                split * WIDTH,
                WIDTH,
            )
            .map_err(launch)?;
            // Which buffer holds the branch for the insertion.
            let branch_is_second = if split > 0 {
                let branch = if split > 1 {
                    k::add(
                        stream,
                        w.chunk.slice(0..rows * WIDTH),
                        w.previous.slice(0..rows * WIDTH),
                        w.combined.slice_mut(0..rows * WIDTH),
                        rows * WIDTH,
                    )
                    .map_err(launch)?;
                    &w.combined
                } else {
                    &w.chunk
                };
                self.tdnn(
                    &block.blocks[split - 1],
                    branch.as_slice(),
                    rows,
                    w.second.as_mut_slice(),
                    &mut w.columns,
                )?;
                stream
                    .copy(
                        w.second.slice(0..rows * WIDTH),
                        w.previous.slice_mut(0..rows * WIDTH),
                    )
                    .map_err(hip("copy Qwen3-TTS Res2Net state"))?;
                true
            } else {
                stream
                    .copy(
                        w.chunk.slice(0..rows * WIDTH),
                        w.previous.slice_mut(0..rows * WIDTH),
                    )
                    .map_err(hip("copy Qwen3-TTS Res2Net first state"))?;
                false
            };
            let branch = if branch_is_second {
                &w.second
            } else {
                &w.chunk
            };
            k::insert_channels(
                stream,
                branch.slice(0..rows * WIDTH),
                w.third.slice_mut(0..rows * CHANNELS),
                rows,
                CHANNELS,
                split * WIDTH,
                WIDTH,
            )
            .map_err(launch)?;
        }
        self.tdnn(
            &block.tdnn2,
            w.third.as_slice(),
            rows,
            w.first.as_mut_slice(),
            &mut w.columns,
        )?;
        k::channel_mean(
            stream,
            w.first.slice(0..rows * CHANNELS),
            w.channel_vector.slice_mut(0..CHANNELS),
            rows,
            CHANNELS,
        )
        .map_err(launch)?;
        self.tdnn(
            &block.se1,
            w.channel_vector.as_slice(),
            1,
            w.second.as_mut_slice(),
            &mut w.columns,
        )?;
        self.run_conv(
            &block.se2,
            w.second.as_slice(),
            1,
            w.channel_vector.as_mut_slice(),
            &mut w.columns,
        )?;
        k::sigmoid_multiply_channels(
            stream,
            w.first.slice_mut(0..rows * CHANNELS),
            w.channel_vector.slice(0..CHANNELS),
            rows,
            CHANNELS,
        )
        .map_err(launch)?;
        k::add(
            stream,
            w.first.slice(0..rows * CHANNELS),
            input.slice(0..rows * CHANNELS),
            output.slice_mut(0..rows * CHANNELS),
            rows * CHANNELS,
        )
        .map_err(launch)?;
        round(stream, output.as_mut_slice(), rows * CHANNELS)
    }

    fn download(&self, values: &DeviceBuffer<f32>, elements: usize) -> Result<Vec<f32>, String> {
        let mut host = vec![0.0f32; elements];
        self.stream
            .download(values.slice(0..elements), &mut host)
            .map_err(hip("copy Qwen3-TTS speaker encoder trace"))?;
        Ok(host)
    }

    /// `SpeakerEncoderHipRuntime::Encode`.
    pub fn encode(
        &mut self,
        audio: &AudioBuffer,
        mut trace: Option<&mut SpeakerEncoderTrace>,
    ) -> Result<SpeakerEncoderOutput, String> {
        if let Some(trace) = trace.as_deref_mut() {
            *trace = SpeakerEncoderTrace::default();
        }
        let features = compute_features(audio)?;
        let frames = features.frames;
        let stream = &self.stream;
        let mut w = SpeakerWorkspace {
            first: alloc(stream, frames * 4608)?,
            second: alloc(stream, frames * 4608)?,
            third: alloc(stream, frames * 4608)?,
            columns: alloc(stream, frames * 4608 * 5)?,
            chunk: alloc(stream, frames * 64)?,
            previous: alloc(stream, frames * 64)?,
            combined: alloc(stream, frames * 64)?,
            channel_vector: alloc(stream, 3072)?,
        };
        // `fourth` of the C++ workspace: the initial block's output, which the
        // first Res2Net block reads after its scratch is written.
        let mut initial = alloc(stream, frames * 4608)?;
        let mut hidden1 = alloc(stream, frames * 512)?;
        let mut hidden2 = alloc(stream, frames * 512)?;
        let mut hidden3 = alloc(stream, frames * 512)?;
        stream
            .upload(
                &features.values,
                w.first.slice_mut(0..features.values.len()),
            )
            .map_err(hip("copy Qwen3-TTS speaker features"))?;
        round(stream, w.first.as_mut_slice(), features.values.len())?;
        self.tdnn(
            &self.block0,
            w.first.as_slice(),
            frames,
            initial.as_mut_slice(),
            &mut w.columns,
        )?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.frames = frames;
            trace.features = self.download(&w.first, frames * MEL_BINS)?;
            trace.initial = self.download(&initial, frames * CHANNELS)?;
        }
        self.run_block(&self.blocks[0], &initial, frames, &mut hidden1, &mut w)?;
        self.run_block(&self.blocks[1], &hidden1, frames, &mut hidden2, &mut w)?;
        self.run_block(&self.blocks[2], &hidden2, frames, &mut hidden3, &mut w)?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.block1 = self.download(&hidden1, frames * CHANNELS)?;
            trace.block2 = self.download(&hidden2, frames * CHANNELS)?;
            trace.block3 = self.download(&hidden3, frames * CHANNELS)?;
        }
        k::concatenate_three(
            stream,
            hidden1.slice(0..frames * CHANNELS),
            hidden2.slice(0..frames * CHANNELS),
            hidden3.slice(0..frames * CHANNELS),
            w.first.slice_mut(0..frames * CHANNELS * 3),
            frames,
            CHANNELS,
        )
        .map_err(launch)?;
        self.tdnn(
            &self.mfa,
            w.first.as_slice(),
            frames,
            w.second.as_mut_slice(),
            &mut w.columns,
        )?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.aggregate = self.download(&w.second, frames * 1536)?;
        }
        k::attentive_stats_input(
            stream,
            w.second.slice(0..frames * 1536),
            w.first.slice_mut(0..frames * 4608),
            frames,
            1536,
            1.0e-12,
        )
        .map_err(launch)?;
        round(stream, w.first.as_mut_slice(), frames * 4608)?;
        self.tdnn(
            &self.asp_tdnn,
            w.first.as_slice(),
            frames,
            w.third.as_mut_slice(),
            &mut w.columns,
        )?;
        k::tanh(stream, w.third.slice_mut(0..frames * 128), frames * 128).map_err(launch)?;
        self.run_conv(
            &self.asp_conv,
            w.third.as_slice(),
            frames,
            w.first.as_mut_slice(),
            &mut w.columns,
        )?;
        k::softmax_over_rows(stream, w.first.slice_mut(0..frames * 1536), frames, 1536)
            .map_err(launch)?;
        round(stream, w.first.as_mut_slice(), frames * 1536)?;
        k::weighted_stats(
            stream,
            w.second.slice(0..frames * 1536),
            w.first.slice(0..frames * 1536),
            w.channel_vector.slice_mut(0..3072),
            frames,
            1536,
            1.0e-12,
        )
        .map_err(launch)?;
        round(stream, w.channel_vector.as_mut_slice(), 3072)?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.pooled = self.download(&w.channel_vector, 3072)?;
        }
        self.run_conv(
            &self.fc,
            w.channel_vector.as_slice(),
            1,
            w.first.as_mut_slice(),
            &mut w.columns,
        )?;
        let mut embedding = vec![0.0f32; 2048];
        stream
            .download(w.first.slice(0..2048), &mut embedding)
            .map_err(hip("copy Qwen3-TTS speaker embedding"))?;
        if let Some(trace) = trace {
            trace.embedding.clone_from(&embedding);
        }
        Ok(SpeakerEncoderOutput { embedding })
    }
}
