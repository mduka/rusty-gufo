//! The f32 speech-tokenizer decoder (`hip/speech_decoder_runtime.cpp`):
//! codebooks, pre-convolution, the sliding-window transformer, two ConvNeXt
//! upsampling stages and the SnakeBeta vocoder, 1920 samples per code frame
//! at 24 kHz. Offline decoding restarts every 300 frames with 25 frames of
//! left context; incremental decoding keeps causal convolution histories and
//! attention K/V to produce the same audio in pieces. Launch order, scratch
//! assignment and the rocBLAS calls are the C++ ones.

use std::collections::HashMap;
use std::time::Instant;

use gufo_hip::blas::{Op, RocBlas};
use gufo_hip::{DeviceBuffer, DeviceIndices, DeviceSlice, DeviceSliceMut, LaunchError, Stream};
use gufo_qwen3_tts_kernels as k;

use crate::checkpoint::Checkpoint;
use crate::weights::{DeviceRegion, View, persistent_indices, resolve_f32};

const CODE_GROUPS: usize = 16;
const CODEBOOK_DIMENSION: usize = 256;
const SAMPLES_PER_CODE: usize = 1920;
const CHUNK_FRAMES: usize = 300;
const LEFT_CONTEXT_FRAMES: usize = 25;
const SCRATCH_BUFFERS: usize = 6;

fn launch(error: LaunchError) -> String {
    format!("Qwen3-TTS speech decoder: {error}")
}

fn hip(operation: &str) -> impl Fn(gufo_hip::HipError) -> String + '_ {
    move |error| format!("{operation}: {error}")
}

/// Convolutional tensors are row-major `[time, channels]`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeechDecoderTrace {
    pub quantizer_output: Vec<f32>,
    pub pre_conv_output: Vec<f32>,
    pub transformer_output: Vec<f32>,
    pub upsample_outputs: Vec<Vec<f32>>,
    pub decoder_outputs: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeechDecoderOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub code_frames: usize,
    pub decode_milliseconds: f64,
}

struct Linear {
    weight: View<f32>,
    bias: Option<View<f32>>,
    input_columns: usize,
    output_columns: usize,
}

/// A weight the region holds, or a reordered copy of a transposed
/// convolution's weight.
enum ConvWeight {
    Region(View<f32>),
    Owned(DeviceBuffer<f32>),
}

struct Conv {
    /// Keys the convolution's incremental history.
    id: usize,
    weight: ConvWeight,
    bias: View<f32>,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    depthwise: bool,
    transpose: bool,
}

struct SnakeBeta {
    alpha: View<f32>,
    beta: View<f32>,
}

struct Codebook {
    embedding_sum: View<f32>,
    cluster_usage: View<f32>,
}

struct TransformerLayer {
    input_norm: View<f32>,
    post_norm: View<f32>,
    attention_scale: View<f32>,
    mlp_scale: View<f32>,
    query: Linear,
    key: Linear,
    value: Linear,
    attention_output: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}

struct ConvNeXt {
    depthwise: Conv,
    norm_weight: View<f32>,
    norm_bias: View<f32>,
    pointwise1: Linear,
    pointwise2: Linear,
    gamma: View<f32>,
}

struct UpsampleStage {
    upsample: Conv,
    convnext: ConvNeXt,
}

struct ResidualUnit {
    activation1: SnakeBeta,
    conv1: Conv,
    activation2: SnakeBeta,
    conv2: Conv,
}

struct DecoderBlock {
    activation: SnakeBeta,
    upsample: Conv,
    residuals: [ResidualUnit; 3],
}

struct Weights {
    semantic_codebook: Codebook,
    acoustic_codebooks: Vec<Codebook>,
    semantic_projection: Linear,
    acoustic_projection: Linear,
    pre_conv: Conv,
    transformer_input: Linear,
    transformer_layers: Vec<TransformerLayer>,
    transformer_norm: View<f32>,
    transformer_output: Linear,
    upsample_stages: Vec<UpsampleStage>,
    decoder_input: Conv,
    decoder_blocks: Vec<DecoderBlock>,
    output_activation: SnakeBeta,
    output_conv: Conv,
}

/// The scratch buffers and the codes, grown for the largest chunk so far.
struct Workspace {
    capacity_frames: usize,
    codes: Option<DeviceIndices>,
    scratch: Vec<DeviceBuffer<f32>>,
}

/// The convolution buffers: im2col columns and the joined history rows.
#[derive(Default)]
struct ConvSpace {
    columns: Option<DeviceBuffer<f32>>,
    history_input: Option<DeviceBuffer<f32>>,
    history_output: Option<DeviceBuffer<f32>>,
}

#[derive(Default)]
struct ConvHistory {
    input: Option<DeviceBuffer<f32>>,
    rows: usize,
    channels: usize,
}

struct AttentionHistory {
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
}

/// The state that makes incremental decoding equal to offline decoding.
#[derive(Default)]
struct DecoderHistory {
    convolutions: HashMap<usize, ConvHistory>,
    attention: Vec<Option<AttentionHistory>>,
    codes: Vec<u32>,
    position: usize,
}

impl DecoderHistory {
    fn reset_block(&mut self) {
        self.position = 0;
        for history in self.convolutions.values_mut() {
            history.rows = 0;
        }
    }

    /// `DecoderHistory::CopyFrom`: retained convolution rows, attention K/V
    /// of the first `source.position` frames (in buffers of
    /// `attention_capacity` frames), codes and position.
    fn copy_from(
        &mut self,
        source: &DecoderHistory,
        attention_width: usize,
        attention_capacity: usize,
        stream: &Stream,
    ) -> Result<(), String> {
        fn copy(
            stream: &Stream,
            destination: &mut Option<DeviceBuffer<f32>>,
            input: Option<&DeviceBuffer<f32>>,
            count: usize,
            capacity: usize,
        ) -> Result<(), String> {
            if destination.as_ref().map_or(0, DeviceBuffer::len) < capacity {
                *destination = None;
                *destination = Some(
                    stream
                        .alloc_persistent::<f32>(capacity)
                        .map_err(hip("hipMalloc Qwen3-TTS speech decoder"))?,
                );
            }
            if count != 0 {
                let (Some(destination), Some(input)) = (destination.as_mut(), input) else {
                    return Err("copy decoder reference state: missing buffer".into());
                };
                stream
                    .copy(input.slice(0..count), destination.slice_mut(0..count))
                    .map_err(hip("copy decoder reference state"))?;
            }
            Ok(())
        }
        self.reset_block();
        for (id, previous) in &source.convolutions {
            let next = self.convolutions.entry(*id).or_default();
            let count = previous.rows * previous.channels;
            copy(
                stream,
                &mut next.input,
                previous.input.as_ref(),
                count,
                count,
            )?;
            next.rows = previous.rows;
            next.channels = previous.channels;
        }
        self.attention.resize_with(source.attention.len(), || None);
        for (layer, previous) in source.attention.iter().enumerate() {
            let count = source.position * attention_width;
            let capacity = attention_capacity * attention_width;
            let (mut key, mut value) = match self.attention[layer].take() {
                Some(cache) => (Some(cache.key), Some(cache.value)),
                None => (None, None),
            };
            copy(
                stream,
                &mut key,
                previous.as_ref().map(|c| &c.key),
                count,
                capacity,
            )?;
            copy(
                stream,
                &mut value,
                previous.as_ref().map(|c| &c.value),
                count,
                capacity,
            )?;
            if let (Some(key), Some(value)) = (key, value) {
                self.attention[layer] = Some(AttentionHistory { key, value });
            }
        }
        self.codes.clone_from(&source.codes);
        self.position = source.position;
        Ok(())
    }
}

/// `SpeechDecoderHipRuntime`.
pub struct SpeechDecoder {
    stream: Stream,
    gemm: RocBlas,
    region: DeviceRegion,
    weights: Weights,
    codebook_size: usize,
    output_sample_rate: u32,
    rms_norm_eps: f32,
    heads: usize,
    head_dim: usize,
    rope_theta: f32,
    sliding_window: usize,
    workspace: Workspace,
    conv: ConvSpace,
    history: DecoderHistory,
    reference_history: Option<DecoderHistory>,
}

/// The first scratch buffer not in `excluded` (`FindScratch`).
fn find_scratch(excluded: &[usize]) -> Result<usize, String> {
    (0..SCRATCH_BUFFERS)
        .find(|index| !excluded.contains(index))
        .ok_or_else(|| "Qwen3-TTS speech decoder scratch exhaustion".into())
}

/// Disjoint scratch buffers by index.
fn disjoint<const N: usize>(
    scratch: &mut [DeviceBuffer<f32>],
    indices: [usize; N],
) -> Result<[&mut DeviceBuffer<f32>; N], String> {
    scratch
        .get_disjoint_mut(indices)
        .map_err(|_| "Qwen3-TTS speech decoder scratch aliasing".to_string())
}

fn capture(
    stream: &Stream,
    buffer: &DeviceBuffer<f32>,
    elements: usize,
) -> Result<Vec<f32>, String> {
    let mut host = vec![0.0f32; elements];
    stream
        .download(buffer.slice(0..elements), &mut host)
        .map_err(hip("copy Qwen3-TTS speech decoder trace"))?;
    Ok(host)
}

struct Loader<'a> {
    stream: &'a Stream,
    checkpoint: &'a Checkpoint,
    region: &'a DeviceRegion,
    next_id: usize,
}

impl Loader<'_> {
    fn f32(&self, name: &str, shape: &[u64]) -> Result<View<f32>, String> {
        resolve_f32(self.checkpoint, self.region, name, shape)
    }

    fn vector(&self, name: &str, size: usize) -> Result<View<f32>, String> {
        self.f32(name, &[size as u64])
    }

    fn linear(
        &self,
        prefix: &str,
        input_columns: usize,
        output_columns: usize,
        bias: bool,
    ) -> Result<Linear, String> {
        Ok(Linear {
            weight: self.f32(
                &format!("{prefix}.weight"),
                &[output_columns as u64, input_columns as u64],
            )?,
            bias: if bias {
                Some(self.vector(&format!("{prefix}.bias"), output_columns)?)
            } else {
                None
            },
            input_columns,
            output_columns,
        })
    }

    fn conv(
        &mut self,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        dilation: usize,
        depthwise: bool,
    ) -> Result<Conv, String> {
        let weight = self.f32(
            &format!("{prefix}.weight"),
            &[
                output_channels as u64,
                if depthwise { 1 } else { input_channels as u64 },
                kernel as u64,
            ],
        )?;
        let bias = self.vector(&format!("{prefix}.bias"), output_channels)?;
        self.next_id += 1;
        Ok(Conv {
            id: self.next_id,
            weight: ConvWeight::Region(weight),
            bias,
            input_channels,
            output_channels,
            kernel,
            dilation,
            stride: 1,
            depthwise,
            transpose: false,
        })
    }

    fn transpose_conv(
        &mut self,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Conv, String> {
        let bias = self.vector(&format!("{prefix}.bias"), output_channels)?;
        let source = self.f32(
            &format!("{prefix}.weight"),
            &[input_channels as u64, output_channels as u64, kernel as u64],
        )?;
        let elements = input_channels * output_channels * kernel;
        let mut transformed = self
            .stream
            .alloc_persistent::<f32>(elements)
            .map_err(hip("hipMalloc Qwen3-TTS speech decoder"))?;
        k::transpose_conv_weight(
            self.stream,
            self.region.slice(source)?,
            transformed.as_mut_slice(),
            input_channels,
            output_channels,
            kernel,
        )
        .map_err(launch)?;
        self.next_id += 1;
        Ok(Conv {
            id: self.next_id,
            weight: ConvWeight::Owned(transformed),
            bias,
            input_channels,
            output_channels,
            kernel,
            dilation: 1,
            stride,
            depthwise: false,
            transpose: true,
        })
    }

    fn snake_beta(&self, alpha: &str, beta: &str, channels: usize) -> Result<SnakeBeta, String> {
        Ok(SnakeBeta {
            alpha: self.vector(alpha, channels)?,
            beta: self.vector(beta, channels)?,
        })
    }

    fn codebook(&self, prefix: &str, codebook_size: usize) -> Result<Codebook, String> {
        Ok(Codebook {
            embedding_sum: self.f32(
                &format!("{prefix}embedding_sum"),
                &[codebook_size as u64, CODEBOOK_DIMENSION as u64],
            )?,
            cluster_usage: self.f32(&format!("{prefix}cluster_usage"), &[codebook_size as u64])?,
        })
    }

    fn conv_projection(&self, prefix: &str) -> Result<Linear, String> {
        Ok(Linear {
            weight: self.f32(&format!("{prefix}.weight"), &[512, 256, 1])?,
            bias: None,
            input_columns: 256,
            output_columns: 512,
        })
    }
}

impl SpeechDecoder {
    /// `SpeechDecoderHipRuntime::Create`.
    pub fn new(checkpoint: &Checkpoint) -> Result<Self, String> {
        let properties = gufo_hip::device_properties(0).map_err(hip("hipGetDeviceProperties"))?;
        if !properties.arch_name.starts_with("gfx1151") || !properties.integrated {
            return Err("Qwen3-TTS speech decoder supports integrated gfx1151 Strix Halo".into());
        }
        let config = &checkpoint.config.speech_tokenizer;
        let regions = checkpoint.regions("decoder.")?;
        let [region] = regions.as_slice() else {
            return Err("Qwen3-TTS speech-tokenizer safetensors mapping is missing".into());
        };
        let stream = Stream::new().map_err(hip("hipStreamCreateWithFlags"))?;
        let region = DeviceRegion::upload(&stream, checkpoint, *region)
            .map_err(|_| "cannot initialize Qwen3-TTS speech weight mapping".to_string())?;
        if config.num_quantizers as usize != CODE_GROUPS
            || config.codebook_dim as usize != 2 * CODEBOOK_DIMENSION
            || config.upsample_rates != [8, 5, 4, 3]
            || config.upsampling_ratios != [2, 2]
        {
            return Err("unsupported Qwen3-TTS speech decoder configuration".into());
        }
        let codebook_size = config.codebook_size as usize;
        let weights = Self::load_weights(&stream, checkpoint, &region, codebook_size)?;
        stream
            .synchronize()
            .map_err(hip("synchronize Qwen3-TTS speech weights"))?;
        let gemm = RocBlas::new(&stream).map_err(hip("rocblas_create_handle"))?;
        Ok(Self {
            gemm,
            region,
            weights,
            codebook_size,
            output_sample_rate: config.output_sample_rate,
            rms_norm_eps: config.rms_norm_eps,
            heads: config.num_attention_heads as usize,
            head_dim: config.head_dim as usize,
            rope_theta: config.rope_theta,
            sliding_window: config.sliding_window as usize,
            workspace: Workspace {
                capacity_frames: 0,
                codes: None,
                scratch: Vec::new(),
            },
            conv: ConvSpace::default(),
            history: DecoderHistory::default(),
            reference_history: None,
            stream,
        })
    }

    fn load_weights(
        stream: &Stream,
        checkpoint: &Checkpoint,
        region: &DeviceRegion,
        codebook_size: usize,
    ) -> Result<Weights, String> {
        let config = &checkpoint.config.speech_tokenizer;
        let mut loader = Loader {
            stream,
            checkpoint,
            region,
            next_id: 0,
        };
        let semantic_codebook = loader.codebook(
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.",
            codebook_size,
        )?;
        let acoustic_codebooks = (0..CODE_GROUPS - 1)
            .map(|group| {
                loader.codebook(
                    &format!("decoder.quantizer.rvq_rest.vq.layers.{group}._codebook."),
                    codebook_size,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let semantic_projection =
            loader.conv_projection("decoder.quantizer.rvq_first.output_proj")?;
        let acoustic_projection =
            loader.conv_projection("decoder.quantizer.rvq_rest.output_proj")?;
        let pre_conv = loader.conv("decoder.pre_conv.conv", 512, 1024, 3, 1, false)?;
        let transformer_input =
            loader.linear("decoder.pre_transformer.input_proj", 1024, 512, true)?;
        let mut transformer_layers = Vec::new();
        for layer in 0..config.num_hidden_layers as usize {
            let prefix = format!("decoder.pre_transformer.layers.{layer}");
            transformer_layers.push(TransformerLayer {
                input_norm: loader.vector(&format!("{prefix}.input_layernorm.weight"), 512)?,
                post_norm: loader
                    .vector(&format!("{prefix}.post_attention_layernorm.weight"), 512)?,
                attention_scale: loader
                    .vector(&format!("{prefix}.self_attn_layer_scale.scale"), 512)?,
                mlp_scale: loader.vector(&format!("{prefix}.mlp_layer_scale.scale"), 512)?,
                query: loader.linear(&format!("{prefix}.self_attn.q_proj"), 512, 1024, false)?,
                key: loader.linear(&format!("{prefix}.self_attn.k_proj"), 512, 1024, false)?,
                value: loader.linear(&format!("{prefix}.self_attn.v_proj"), 512, 1024, false)?,
                attention_output: loader.linear(
                    &format!("{prefix}.self_attn.o_proj"),
                    1024,
                    512,
                    false,
                )?,
                gate: loader.linear(&format!("{prefix}.mlp.gate_proj"), 512, 1024, false)?,
                up: loader.linear(&format!("{prefix}.mlp.up_proj"), 512, 1024, false)?,
                down: loader.linear(&format!("{prefix}.mlp.down_proj"), 1024, 512, false)?,
            });
        }
        let transformer_norm = loader.vector("decoder.pre_transformer.norm.weight", 512)?;
        let transformer_output =
            loader.linear("decoder.pre_transformer.output_proj", 512, 1024, true)?;
        let mut upsample_stages = Vec::new();
        for stage in 0..2 {
            let prefix = format!("decoder.upsample.{stage}");
            let upsample = loader.transpose_conv(&format!("{prefix}.0.conv"), 1024, 1024, 2, 2)?;
            let depthwise =
                loader.conv(&format!("{prefix}.1.dwconv.conv"), 1024, 1024, 7, 1, true)?;
            upsample_stages.push(UpsampleStage {
                upsample,
                convnext: ConvNeXt {
                    depthwise,
                    norm_weight: loader.vector(&format!("{prefix}.1.norm.weight"), 1024)?,
                    norm_bias: loader.vector(&format!("{prefix}.1.norm.bias"), 1024)?,
                    pointwise1: loader.linear(&format!("{prefix}.1.pwconv1"), 1024, 4096, true)?,
                    pointwise2: loader.linear(&format!("{prefix}.1.pwconv2"), 4096, 1024, true)?,
                    gamma: loader.vector(&format!("{prefix}.1.gamma"), 1024)?,
                },
            });
        }
        let decoder_input = loader.conv("decoder.decoder.0.conv", 1024, 1536, 7, 1, false)?;
        let mut decoder_blocks = Vec::new();
        let mut input_channels = 1536;
        for (block, rate) in [8usize, 5, 4, 3].into_iter().enumerate() {
            let output_channels = input_channels / 2;
            let prefix = format!("decoder.decoder.{}.block", block + 1);
            let activation = loader.snake_beta(
                &format!("{prefix}.0.alpha"),
                &format!("{prefix}.0.beta"),
                input_channels,
            )?;
            let upsample = loader.transpose_conv(
                &format!("{prefix}.1.conv"),
                input_channels,
                output_channels,
                rate * 2,
                rate,
            )?;
            let mut residual = |unit: usize, dilation: usize| -> Result<ResidualUnit, String> {
                let unit_prefix = format!("{prefix}.{}", unit + 2);
                let activation1 = loader.snake_beta(
                    &format!("{unit_prefix}.act1.alpha"),
                    &format!("{unit_prefix}.act1.beta"),
                    output_channels,
                )?;
                let conv1 = loader.conv(
                    &format!("{unit_prefix}.conv1.conv"),
                    output_channels,
                    output_channels,
                    7,
                    dilation,
                    false,
                )?;
                let activation2 = loader.snake_beta(
                    &format!("{unit_prefix}.act2.alpha"),
                    &format!("{unit_prefix}.act2.beta"),
                    output_channels,
                )?;
                let conv2 = loader.conv(
                    &format!("{unit_prefix}.conv2.conv"),
                    output_channels,
                    output_channels,
                    1,
                    1,
                    false,
                )?;
                Ok(ResidualUnit {
                    activation1,
                    conv1,
                    activation2,
                    conv2,
                })
            };
            let residuals = [residual(0, 1)?, residual(1, 3)?, residual(2, 9)?];
            decoder_blocks.push(DecoderBlock {
                activation,
                upsample,
                residuals,
            });
            input_channels = output_channels;
        }
        let output_activation =
            loader.snake_beta("decoder.decoder.5.alpha", "decoder.decoder.5.beta", 96)?;
        let output_conv = loader.conv("decoder.decoder.6.conv", 96, 1, 7, 1, false)?;
        Ok(Weights {
            semantic_codebook,
            acoustic_codebooks,
            semantic_projection,
            acoustic_projection,
            pre_conv,
            transformer_input,
            transformer_layers,
            transformer_norm,
            transformer_output,
            upsample_stages,
            decoder_input,
            decoder_blocks,
            output_activation,
            output_conv,
        })
    }

    /// `Workspace::Ensure`: scratch, columns and codes for `frames` frames.
    fn ensure_workspace(&mut self, frames: usize) -> Result<(), String> {
        if frames <= self.workspace.capacity_frames {
            return Ok(());
        }
        let scratch_elements = frames
            .checked_mul(SAMPLES_PER_CODE * 96)
            .ok_or("Qwen3-TTS speech decoder frame overflow")?;
        let column_elements = scratch_elements
            .checked_mul(7)
            .ok_or("Qwen3-TTS speech decoder frame overflow")?;
        let stream = &self.stream;
        let alloc = |len: usize| {
            stream
                .alloc_persistent::<f32>(len)
                .map_err(hip("hipMalloc Qwen3-TTS speech decoder"))
        };
        // Released before the replacements are allocated, as `Reset` does.
        self.workspace.codes = None;
        self.workspace.scratch.clear();
        self.conv.columns = None;
        self.workspace.codes = Some(
            persistent_indices(stream, frames * CODE_GROUPS, self.codebook_size as u32)
                .map_err(hip("hipMalloc Qwen3-TTS speech decoder"))?,
        );
        for _ in 0..SCRATCH_BUFFERS {
            self.workspace.scratch.push(alloc(scratch_elements)?);
        }
        self.conv.columns = Some(alloc(column_elements)?);
        self.workspace.capacity_frames = frames;
        Ok(())
    }

    /// `DecodeChunk`: one chunk of codes to samples, with the incremental
    /// history when `incremental`.
    fn decode_chunk(
        &mut self,
        codes: &[u32],
        frames: usize,
        mut trace: Option<&mut SpeechDecoderTrace>,
        incremental: bool,
    ) -> Result<Vec<f32>, String> {
        self.ensure_workspace(frames + if incremental { LEFT_CONTEXT_FRAMES } else { 0 })?;
        let Self {
            stream,
            gemm,
            region,
            weights,
            workspace,
            conv,
            history,
            codebook_size,
            rms_norm_eps,
            heads,
            head_dim,
            rope_theta,
            sliding_window,
            ..
        } = self;
        let (codebook_size, epsilon) = (*codebook_size, *rms_norm_eps);
        let (heads, head_dim, rope_theta, window) =
            (*heads, *head_dim, *rope_theta, *sliding_window);
        let stream: &Stream = stream;
        let mut history = incremental.then_some(history);
        let indices = workspace
            .codes
            .as_mut()
            .ok_or("Qwen3-TTS speech decoder workspace is missing")?;
        stream
            .upload_indices(codes, indices)
            .map_err(hip("copy Qwen3-TTS codec frames"))?;
        let codes_view = indices.slice(0..frames * CODE_GROUPS);
        let scratch = &mut workspace.scratch;

        let decode = |stream: &Stream,
                      codebook: &Codebook,
                      group: usize,
                      output: &mut DeviceBuffer<f32>,
                      initialize: bool|
         -> Result<(), String> {
            k::decode_codebook(
                stream,
                codes_view,
                CODE_GROUPS as u32,
                group as u32,
                region.slice(codebook.embedding_sum)?,
                region.slice(codebook.cluster_usage)?,
                output.slice_mut(0..frames * CODEBOOK_DIMENSION),
                frames,
                CODEBOOK_DIMENSION,
                initialize,
                codebook_size,
            )
            .map_err(launch)
        };
        decode(stream, &weights.semantic_codebook, 0, &mut scratch[0], true)?;
        for group in 1..CODE_GROUPS {
            decode(
                stream,
                &weights.acoustic_codebooks[group - 1],
                group,
                &mut scratch[1],
                group == 1,
            )?;
        }
        {
            let [semantic, projected] = disjoint(scratch, [0, 2])?;
            run_linear(
                stream,
                gemm,
                region,
                &weights.semantic_projection,
                semantic.as_slice(),
                projected.as_mut_slice(),
                frames,
            )?;
            let [acoustic, projected] = disjoint(scratch, [1, 3])?;
            run_linear(
                stream,
                gemm,
                region,
                &weights.acoustic_projection,
                acoustic.as_slice(),
                projected.as_mut_slice(),
                frames,
            )?;
            let [hidden, semantic, acoustic] = disjoint(scratch, [0, 2, 3])?;
            k::add(
                stream,
                semantic.slice(0..frames * 512),
                acoustic.slice(0..frames * 512),
                hidden.slice_mut(0..frames * 512),
                frames * 512,
            )
            .map_err(launch)?;
        }
        if let Some(trace) = trace.as_deref_mut() {
            trace.quantizer_output = capture(stream, &scratch[0], frames * 512)?;
        }
        {
            let [hidden, output] = disjoint(scratch, [0, 1])?;
            run_conv(
                stream,
                gemm,
                region,
                conv,
                history.as_deref_mut(),
                &weights.pre_conv,
                hidden.as_slice(),
                output.as_mut_slice(),
                frames,
                frames,
            )?;
        }
        if let Some(trace) = trace.as_deref_mut() {
            trace.pre_conv_output = capture(stream, &scratch[1], frames * 1024)?;
        }
        let mut hidden = 0;
        {
            let [input, output] = disjoint(scratch, [1, 0])?;
            run_linear(
                stream,
                gemm,
                region,
                &weights.transformer_input,
                input.as_slice(),
                output.as_mut_slice(),
                frames,
            )?;
        }
        let position = history.as_ref().map_or(0, |h| h.position);
        let width = heads * head_dim;
        for (layer_index, layer) in weights.transformer_layers.iter().enumerate() {
            let normalized = find_scratch(&[hidden])?;
            {
                let [h, n] = disjoint(scratch, [hidden, normalized])?;
                k::rms_norm(
                    stream,
                    h.slice(0..frames * 512),
                    region.slice(layer.input_norm)?,
                    n.slice_mut(0..frames * 512),
                    frames,
                    512,
                    epsilon,
                )
                .map_err(launch)?;
            }
            let query = find_scratch(&[hidden, normalized])?;
            let key = find_scratch(&[hidden, normalized, query])?;
            let value = find_scratch(&[hidden, normalized, query, key])?;
            for (linear, output) in [
                (&layer.query, query),
                (&layer.key, key),
                (&layer.value, value),
            ] {
                let [n, o] = disjoint(scratch, [normalized, output])?;
                run_linear(
                    stream,
                    gemm,
                    region,
                    linear,
                    n.as_slice(),
                    o.as_mut_slice(),
                    frames,
                )?;
            }
            {
                let [q, k_] = disjoint(scratch, [query, key])?;
                k::rope(
                    stream,
                    q.slice_mut(0..frames * width),
                    k_.slice_mut(0..frames * width),
                    frames,
                    heads,
                    head_dim,
                    rope_theta,
                    position,
                )
                .map_err(launch)?;
            }
            let attention = find_scratch(&[hidden, normalized, query, key, value])?;
            {
                let [q, k_, v, out] = disjoint(scratch, [query, key, value, attention])?;
                if let Some(active) = history.as_deref_mut() {
                    active
                        .attention
                        .resize_with(weights.transformer_layers.len(), || None);
                    let cache_elements = (CHUNK_FRAMES + LEFT_CONTEXT_FRAMES) * width;
                    if active.attention[layer_index].is_none() {
                        active.attention[layer_index] = Some(AttentionHistory {
                            key: stream
                                .alloc_persistent::<f32>(cache_elements)
                                .map_err(hip("hipMalloc Qwen3-TTS speech decoder"))?,
                            value: stream
                                .alloc_persistent::<f32>(cache_elements)
                                .map_err(hip("hipMalloc Qwen3-TTS speech decoder"))?,
                        });
                    }
                    if position + frames > CHUNK_FRAMES + LEFT_CONTEXT_FRAMES {
                        return Err("incremental decoder attention exceeds its window".into());
                    }
                    let cache = active.attention[layer_index]
                        .as_mut()
                        .ok_or("Qwen3-TTS speech decoder attention cache is missing")?;
                    stream
                        .copy(
                            k_.slice(0..frames * width),
                            cache
                                .key
                                .slice_mut(position * width..(position + frames) * width),
                        )
                        .map_err(hip("append decoder attention keys"))?;
                    stream
                        .copy(
                            v.slice(0..frames * width),
                            cache
                                .value
                                .slice_mut(position * width..(position + frames) * width),
                        )
                        .map_err(hip("append decoder attention values"))?;
                    k::sliding_causal_attention(
                        stream,
                        q.slice(0..frames * width),
                        cache.key.slice(0..(position + frames) * width),
                        cache.value.slice(0..(position + frames) * width),
                        out.slice_mut(0..frames * width),
                        frames,
                        heads,
                        head_dim,
                        window,
                        position,
                        position + frames,
                    )
                    .map_err(launch)?;
                } else {
                    k::sliding_causal_attention(
                        stream,
                        q.slice(0..frames * width),
                        k_.slice(0..frames * width),
                        v.slice(0..frames * width),
                        out.slice_mut(0..frames * width),
                        frames,
                        heads,
                        head_dim,
                        window,
                        0,
                        frames,
                    )
                    .map_err(launch)?;
                }
            }
            {
                let [a, n] = disjoint(scratch, [attention, normalized])?;
                run_linear(
                    stream,
                    gemm,
                    region,
                    &layer.attention_output,
                    a.as_slice(),
                    n.as_mut_slice(),
                    frames,
                )?;
                let [h, n] = disjoint(scratch, [hidden, normalized])?;
                k::add_scaled_channels(
                    stream,
                    h.slice_mut(0..frames * 512),
                    n.slice(0..frames * 512),
                    region.slice(layer.attention_scale)?,
                    frames,
                    512,
                )
                .map_err(launch)?;
                k::rms_norm(
                    stream,
                    h.slice(0..frames * 512),
                    region.slice(layer.post_norm)?,
                    n.slice_mut(0..frames * 512),
                    frames,
                    512,
                    epsilon,
                )
                .map_err(launch)?;
            }
            let (gate, up, activation) = (query, key, value);
            {
                let [n, g] = disjoint(scratch, [normalized, gate])?;
                run_linear(
                    stream,
                    gemm,
                    region,
                    &layer.gate,
                    n.as_slice(),
                    g.as_mut_slice(),
                    frames,
                )?;
                let [n, u] = disjoint(scratch, [normalized, up])?;
                run_linear(
                    stream,
                    gemm,
                    region,
                    &layer.up,
                    n.as_slice(),
                    u.as_mut_slice(),
                    frames,
                )?;
                let [g, u, a] = disjoint(scratch, [gate, up, activation])?;
                k::swiglu(
                    stream,
                    g.slice(0..frames * 1024),
                    u.slice(0..frames * 1024),
                    a.slice_mut(0..frames * 1024),
                    frames * 1024,
                )
                .map_err(launch)?;
                let [a, n] = disjoint(scratch, [activation, normalized])?;
                run_linear(
                    stream,
                    gemm,
                    region,
                    &layer.down,
                    a.as_slice(),
                    n.as_mut_slice(),
                    frames,
                )?;
                let [h, n] = disjoint(scratch, [hidden, normalized])?;
                k::add_scaled_channels(
                    stream,
                    h.slice_mut(0..frames * 512),
                    n.slice(0..frames * 512),
                    region.slice(layer.mlp_scale)?,
                    frames,
                    512,
                )
                .map_err(launch)?;
            }
        }
        let normalized = find_scratch(&[hidden])?;
        {
            let [h, n] = disjoint(scratch, [hidden, normalized])?;
            k::rms_norm(
                stream,
                h.slice(0..frames * 512),
                region.slice(weights.transformer_norm)?,
                n.slice_mut(0..frames * 512),
                frames,
                512,
                epsilon,
            )
            .map_err(launch)?;
        }
        let transformer = find_scratch(&[hidden, normalized])?;
        {
            let [n, t] = disjoint(scratch, [normalized, transformer])?;
            run_linear(
                stream,
                gemm,
                region,
                &weights.transformer_output,
                n.as_slice(),
                t.as_mut_slice(),
                frames,
            )?;
        }
        if let Some(trace) = trace.as_deref_mut() {
            trace.transformer_output = capture(stream, &scratch[transformer], frames * 1024)?;
            trace.upsample_outputs.clear();
            trace.decoder_outputs.clear();
        }

        let mut length = frames;
        hidden = transformer;
        for stage in &weights.upsample_stages {
            let output_length = length * stage.upsample.stride;
            let upsampled = find_scratch(&[hidden])?;
            {
                let [h, u] = disjoint(scratch, [hidden, upsampled])?;
                run_conv(
                    stream,
                    gemm,
                    region,
                    conv,
                    history.as_deref_mut(),
                    &stage.upsample,
                    h.as_slice(),
                    u.as_mut_slice(),
                    length,
                    output_length,
                )?;
            }
            if let Some(trace) = trace.as_deref_mut() {
                trace.upsample_outputs.push(capture(
                    stream,
                    &scratch[upsampled],
                    output_length * 1024,
                )?);
            }
            let depthwise = find_scratch(&[hidden, upsampled])?;
            {
                let [u, d] = disjoint(scratch, [upsampled, depthwise])?;
                run_conv(
                    stream,
                    gemm,
                    region,
                    conv,
                    history.as_deref_mut(),
                    &stage.convnext.depthwise,
                    u.as_slice(),
                    d.as_mut_slice(),
                    output_length,
                    output_length,
                )?;
            }
            let normed = find_scratch(&[hidden, upsampled, depthwise])?;
            {
                let [d, n] = disjoint(scratch, [depthwise, normed])?;
                k::layer_norm(
                    stream,
                    d.slice(0..output_length * 1024),
                    region.slice(stage.convnext.norm_weight)?,
                    region.slice(stage.convnext.norm_bias)?,
                    n.slice_mut(0..output_length * 1024),
                    output_length,
                    1024,
                    1.0e-6,
                )
                .map_err(launch)?;
            }
            let wide = find_scratch(&[hidden, upsampled, depthwise, normed])?;
            {
                let [n, w] = disjoint(scratch, [normed, wide])?;
                run_linear(
                    stream,
                    gemm,
                    region,
                    &stage.convnext.pointwise1,
                    n.as_slice(),
                    w.as_mut_slice(),
                    output_length,
                )?;
                k::gelu(
                    stream,
                    w.slice_mut(0..output_length * 4096),
                    output_length * 4096,
                )
                .map_err(launch)?;
                let [w, d] = disjoint(scratch, [wide, depthwise])?;
                run_linear(
                    stream,
                    gemm,
                    region,
                    &stage.convnext.pointwise2,
                    w.as_slice(),
                    d.as_mut_slice(),
                    output_length,
                )?;
                let [u, d] = disjoint(scratch, [upsampled, depthwise])?;
                k::add_scaled_channels(
                    stream,
                    u.slice_mut(0..output_length * 1024),
                    d.slice(0..output_length * 1024),
                    region.slice(stage.convnext.gamma)?,
                    output_length,
                    1024,
                )
                .map_err(launch)?;
            }
            if let Some(trace) = trace.as_deref_mut() {
                trace.upsample_outputs.push(capture(
                    stream,
                    &scratch[upsampled],
                    output_length * 1024,
                )?);
            }
            hidden = upsampled;
            length = output_length;
        }

        let decoder_hidden = find_scratch(&[hidden])?;
        {
            let [h, d] = disjoint(scratch, [hidden, decoder_hidden])?;
            run_conv(
                stream,
                gemm,
                region,
                conv,
                history.as_deref_mut(),
                &weights.decoder_input,
                h.as_slice(),
                d.as_mut_slice(),
                length,
                length,
            )?;
        }
        if let Some(trace) = trace.as_deref_mut() {
            trace
                .decoder_outputs
                .push(capture(stream, &scratch[decoder_hidden], length * 1536)?);
        }
        hidden = decoder_hidden;
        let mut channels = 1536;
        for block in &weights.decoder_blocks {
            let activated = find_scratch(&[hidden])?;
            {
                let [h, a] = disjoint(scratch, [hidden, activated])?;
                snake_beta(
                    stream,
                    region,
                    &block.activation,
                    h.as_slice(),
                    a.as_mut_slice(),
                    length,
                    channels,
                )?;
            }
            let output_length = length * block.upsample.stride;
            let upsampled = find_scratch(&[hidden, activated])?;
            {
                let [a, u] = disjoint(scratch, [activated, upsampled])?;
                run_conv(
                    stream,
                    gemm,
                    region,
                    conv,
                    history.as_deref_mut(),
                    &block.upsample,
                    a.as_slice(),
                    u.as_mut_slice(),
                    length,
                    output_length,
                )?;
            }
            channels /= 2;
            for residual in &block.residuals {
                let branch = find_scratch(&[hidden, activated, upsampled])?;
                let work = find_scratch(&[hidden, activated, upsampled, branch])?;
                let [u, b] = disjoint(scratch, [upsampled, branch])?;
                snake_beta(
                    stream,
                    region,
                    &residual.activation1,
                    u.as_slice(),
                    b.as_mut_slice(),
                    output_length,
                    channels,
                )?;
                let [b, w] = disjoint(scratch, [branch, work])?;
                run_conv(
                    stream,
                    gemm,
                    region,
                    conv,
                    history.as_deref_mut(),
                    &residual.conv1,
                    b.as_slice(),
                    w.as_mut_slice(),
                    output_length,
                    output_length,
                )?;
                let [w, b] = disjoint(scratch, [work, branch])?;
                snake_beta(
                    stream,
                    region,
                    &residual.activation2,
                    w.as_slice(),
                    b.as_mut_slice(),
                    output_length,
                    channels,
                )?;
                let [b, w] = disjoint(scratch, [branch, work])?;
                run_conv(
                    stream,
                    gemm,
                    region,
                    conv,
                    history.as_deref_mut(),
                    &residual.conv2,
                    b.as_slice(),
                    w.as_mut_slice(),
                    output_length,
                    output_length,
                )?;
                let [u, w] = disjoint(scratch, [upsampled, work])?;
                k::add_in_place(
                    stream,
                    w.slice(0..output_length * channels),
                    u.slice_mut(0..output_length * channels),
                    output_length * channels,
                )
                .map_err(launch)?;
            }
            hidden = upsampled;
            length = output_length;
            if let Some(trace) = trace.as_deref_mut() {
                trace
                    .decoder_outputs
                    .push(capture(stream, &scratch[hidden], length * channels)?);
            }
        }

        let activated = find_scratch(&[hidden])?;
        {
            let [h, a] = disjoint(scratch, [hidden, activated])?;
            snake_beta(
                stream,
                region,
                &weights.output_activation,
                h.as_slice(),
                a.as_mut_slice(),
                length,
                96,
            )?;
        }
        if let Some(trace) = trace.as_deref_mut() {
            trace
                .decoder_outputs
                .push(capture(stream, &scratch[activated], length * 96)?);
        }
        let waveform = find_scratch(&[hidden, activated])?;
        {
            let [a, w] = disjoint(scratch, [activated, waveform])?;
            run_conv(
                stream,
                gemm,
                region,
                conv,
                history.as_deref_mut(),
                &weights.output_conv,
                a.as_slice(),
                w.as_mut_slice(),
                length,
                length,
            )?;
        }
        if let Some(trace) = trace {
            trace
                .decoder_outputs
                .push(capture(stream, &scratch[waveform], length)?);
        }
        k::clamp(
            stream,
            scratch[waveform].slice_mut(0..length),
            length,
            -1.0,
            1.0,
        )
        .map_err(launch)?;
        let mut samples = vec![0.0f32; length];
        stream
            .download(scratch[waveform].slice(0..length), &mut samples)
            .map_err(hip("copy Qwen3-TTS waveform"))?;
        if let Some(active) = history {
            active.position += frames;
        }
        Ok(samples)
    }

    fn check_codes(&self, codes: &[u32], message: &str) -> Result<(), String> {
        if codes
            .iter()
            .any(|&code| code as usize >= self.codebook_size)
        {
            return Err(message.into());
        }
        Ok(())
    }

    /// `Decode`: offline decoding in 300-frame chunks with 25 frames of left
    /// context. A trace covers at most one chunk.
    pub fn decode(
        &mut self,
        codes: &[u32],
        frames: usize,
        mut trace: Option<&mut SpeechDecoderTrace>,
    ) -> Result<SpeechDecoderOutput, String> {
        if frames == 0 || frames > usize::MAX / CODE_GROUPS || codes.len() != frames * CODE_GROUPS {
            return Err("Qwen3-TTS speech decoder expects [frames,16] codes".into());
        }
        self.check_codes(codes, "Qwen3-TTS speech decoder code is out of range")?;
        if trace.is_some() && frames > CHUNK_FRAMES {
            return Err("Qwen3-TTS speech decoder trace supports at most 300 frames".into());
        }
        let begin = Instant::now();
        let mut samples = Vec::with_capacity(frames * SAMPLES_PER_CODE);
        let mut start = 0;
        while start < frames {
            let end = (start + CHUNK_FRAMES).min(frames);
            let context = start.min(LEFT_CONTEXT_FRAMES);
            let chunk_start = start - context;
            let chunk_frames = end - chunk_start;
            let decoded = self.decode_chunk(
                &codes[chunk_start * CODE_GROUPS..end * CODE_GROUPS],
                chunk_frames,
                trace.as_deref_mut(),
                false,
            )?;
            let drop = context * SAMPLES_PER_CODE;
            let keep = (end - start) * SAMPLES_PER_CODE;
            if drop > decoded.len() || keep > decoded.len() - drop {
                return Err("Qwen3-TTS speech decoder chunk output is truncated".into());
            }
            samples.extend_from_slice(&decoded[drop..drop + keep]);
            start += CHUNK_FRAMES;
        }
        Ok(SpeechDecoderOutput {
            samples,
            sample_rate: self.output_sample_rate,
            code_frames: frames,
            decode_milliseconds: begin.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// `DecodeIncremental`: decodes only the new suffix of a growing code
    /// sequence (`begin_frame` 0 starts a request), keeping the offline
    /// decoder's 300/25-frame boundaries. Returns the suffix audio.
    pub fn decode_incremental(
        &mut self,
        codes: &[u32],
        frames: usize,
        begin_frame: usize,
    ) -> Result<SpeechDecoderOutput, String> {
        if frames == 0
            || begin_frame > frames
            || frames > usize::MAX / SAMPLES_PER_CODE
            || codes.len() != frames * CODE_GROUPS
        {
            return Err("invalid incremental speech decoder shape".into());
        }
        if begin_frame == 0 {
            self.history.codes.clear();
            self.history.reset_block();
        } else if self.history.codes.len() != begin_frame * CODE_GROUPS
            || self.history.codes[..] != codes[..begin_frame * CODE_GROUPS]
        {
            return Err("incremental speech decoder prefix does not match".into());
        }
        self.check_codes(codes, "incremental speech decoder code is out of range")?;
        let started = Instant::now();
        let result = self.decode_incremental_chunks(codes, frames, begin_frame);
        match result {
            Ok(samples) => Ok(SpeechDecoderOutput {
                samples,
                sample_rate: self.output_sample_rate,
                code_frames: frames - begin_frame,
                decode_milliseconds: started.elapsed().as_secs_f64() * 1000.0,
            }),
            Err(error) => {
                let _ = self.stream.synchronize();
                self.history.codes.clear();
                self.history.reset_block();
                Err(error)
            }
        }
    }

    fn decode_incremental_chunks(
        &mut self,
        codes: &[u32],
        frames: usize,
        begin_frame: usize,
    ) -> Result<Vec<f32>, String> {
        let mut samples = Vec::new();
        let mut cursor = begin_frame;
        while cursor < frames {
            if cursor.is_multiple_of(CHUNK_FRAMES) {
                self.history.reset_block();
                // The offline decoder restarts every 300 frames with 25 prior
                // frames: rebuild exactly that context without its audio.
                let mut warm = cursor - cursor.min(LEFT_CONTEXT_FRAMES);
                while warm < cursor {
                    let count = 16.min(cursor - warm);
                    let _ = self.decode_chunk(
                        &codes[warm * CODE_GROUPS..(warm + count) * CODE_GROUPS],
                        count,
                        None,
                        true,
                    )?;
                    warm += count;
                }
            }
            let count = 16
                .min(frames - cursor)
                .min(CHUNK_FRAMES - cursor % CHUNK_FRAMES);
            let chunk = &codes[cursor * CODE_GROUPS..(cursor + count) * CODE_GROUPS];
            samples.extend(self.decode_chunk(chunk, count, None, true)?);
            self.history.codes.extend_from_slice(chunk);
            cursor += count;
        }
        Ok(samples)
    }

    /// `DecodeAfterReference`: starts a voice-clone request and returns only
    /// the audio after `reference_frames`, keeping one exact state of the
    /// reference prefix for later requests with the same reference.
    pub fn decode_after_reference(
        &mut self,
        codes: &[u32],
        frames: usize,
        reference_frames: usize,
    ) -> Result<SpeechDecoderOutput, String> {
        if reference_frames >= frames
            || frames > usize::MAX / SAMPLES_PER_CODE
            || codes.len() != frames * CODE_GROUPS
        {
            return Err("invalid speech decoder reference shape".into());
        }
        let started = Instant::now();
        // End at an existing incremental work boundary; a partial reference
        // tail continues with generated codes in the same chunks as a cold
        // request.
        let prefix_frames = reference_frames - (reference_frames % CHUNK_FRAMES) % 16;
        let result = (|| -> Result<SpeechDecoderOutput, String> {
            if prefix_frames != 0 {
                let prefix = &codes[..prefix_frames * CODE_GROUPS];
                let width = self.heads * self.head_dim;
                let reusable = self
                    .reference_history
                    .as_ref()
                    .is_some_and(|saved| saved.codes[..] == *prefix);
                if !reusable {
                    self.decode_incremental(prefix, prefix_frames, 0)?;
                    let mut replacement = DecoderHistory::default();
                    replacement.copy_from(
                        &self.history,
                        width,
                        self.history.position,
                        &self.stream,
                    )?;
                    self.stream
                        .synchronize()
                        .map_err(hip("capture decoder reference state"))?;
                    self.reference_history = Some(replacement);
                } else if let Some(saved) = self.reference_history.as_ref() {
                    self.history.copy_from(
                        saved,
                        width,
                        CHUNK_FRAMES + LEFT_CONTEXT_FRAMES,
                        &self.stream,
                    )?;
                }
            }
            let mut output = self.decode_incremental(codes, frames, prefix_frames)?;
            let drop = (reference_frames - prefix_frames) * SAMPLES_PER_CODE;
            output.samples.drain(..drop.min(output.samples.len()));
            output.code_frames = frames - reference_frames;
            output.decode_milliseconds = started.elapsed().as_secs_f64() * 1000.0;
            Ok(output)
        })();
        if result.is_err() {
            let _ = self.stream.synchronize();
            self.history.codes.clear();
            self.history.reset_block();
        }
        result
    }
}

/// `RunLinear`: `output = input · weightᵀ (+ bias)` over `rows` rows.
fn run_linear(
    stream: &Stream,
    gemm: &RocBlas,
    region: &DeviceRegion,
    weights: &Linear,
    input: DeviceSlice<'_, f32>,
    mut output: DeviceSliceMut<'_, f32>,
    rows: usize,
) -> Result<(), String> {
    gemm_f32(
        stream,
        gemm,
        region.slice(weights.weight)?,
        input,
        output.reborrow(),
        rows,
        weights.output_columns,
        weights.input_columns,
    )?;
    if let Some(bias) = weights.bias {
        k::add_bias(
            stream,
            output.range(0..rows * weights.output_columns),
            region.slice(bias)?,
            rows,
            weights.output_columns,
        )
        .map_err(launch)?;
    }
    Ok(())
}

fn snake_beta(
    stream: &Stream,
    region: &DeviceRegion,
    weights: &SnakeBeta,
    input: DeviceSlice<'_, f32>,
    output: DeviceSliceMut<'_, f32>,
    rows: usize,
    columns: usize,
) -> Result<(), String> {
    k::snake_beta(
        stream,
        input.range(0..rows * columns),
        region.slice(weights.alpha)?,
        region.slice(weights.beta)?,
        output.range(0..rows * columns),
        rows,
        columns,
    )
    .map_err(launch)
}

fn conv_weight<'a>(
    region: &'a DeviceRegion,
    weights: &'a Conv,
) -> Result<DeviceSlice<'a, f32>, String> {
    match &weights.weight {
        ConvWeight::Region(view) => region.slice(*view),
        ConvWeight::Owned(buffer) => Ok(buffer.as_slice()),
    }
}

/// `RunConvRaw`: depthwise, transposed (GEMM into columns, then assembly) or
/// im2col and GEMM.
#[allow(clippy::too_many_arguments)]
fn run_conv_raw(
    stream: &Stream,
    gemm: &RocBlas,
    region: &DeviceRegion,
    columns: &mut DeviceBuffer<f32>,
    weights: &Conv,
    input: DeviceSlice<'_, f32>,
    mut output: DeviceSliceMut<'_, f32>,
    input_length: usize,
    output_length: usize,
) -> Result<(), String> {
    let weight = conv_weight(region, weights)?;
    let bias = region.slice(weights.bias)?;
    if weights.depthwise {
        if input_length != output_length || weights.input_channels != weights.output_channels {
            return Err("invalid Qwen3-TTS depthwise convolution shape".into());
        }
        return k::causal_depthwise_conv1d(
            stream,
            input.range(0..input_length * weights.input_channels),
            weight,
            Some(bias),
            output.range(0..output_length * weights.output_channels),
            input_length,
            weights.input_channels,
            weights.kernel,
            weights.dilation,
        )
        .map_err(launch);
    }
    if weights.transpose {
        if output_length != input_length * weights.stride {
            return Err("invalid Qwen3-TTS transposed convolution length".into());
        }
        let expanded_columns = weights.output_channels * weights.kernel;
        gemm_f32(
            stream,
            gemm,
            weight,
            input,
            columns.as_mut_slice(),
            input_length,
            expanded_columns,
            weights.input_channels,
        )?;
        return k::assemble_causal_conv_transpose1d(
            stream,
            columns.slice(0..input_length * expanded_columns),
            Some(bias),
            output.range(0..output_length * weights.output_channels),
            input_length,
            output_length,
            weights.output_channels,
            weights.kernel,
            weights.stride,
        )
        .map_err(launch);
    }
    if input_length != output_length {
        return Err("invalid Qwen3-TTS causal convolution length".into());
    }
    let reduction = weights.input_channels * weights.kernel;
    k::causal_conv1d_im2col(
        stream,
        input.range(0..input_length * weights.input_channels),
        columns.slice_mut(0..output_length * reduction),
        input_length,
        weights.input_channels,
        output_length,
        weights.kernel,
        weights.dilation,
    )
    .map_err(launch)?;
    gemm_f32(
        stream,
        gemm,
        weight,
        columns.as_slice(),
        output.reborrow(),
        output_length,
        weights.output_channels,
        reduction,
    )?;
    k::add_bias(
        stream,
        output.range(0..output_length * weights.output_channels),
        bias,
        output_length,
        weights.output_channels,
    )
    .map_err(launch)
}

fn grow(
    stream: &Stream,
    buffer: &mut Option<DeviceBuffer<f32>>,
    elements: usize,
) -> Result<(), String> {
    if buffer.as_ref().is_none_or(|b| b.len() < elements) {
        *buffer = None;
        *buffer = Some(
            stream
                .alloc_persistent::<f32>(elements)
                .map_err(hip("hipMalloc Qwen3-TTS speech decoder"))?,
        );
    }
    Ok(())
}

/// `RunConv`: with an active history the convolution sees its retained input
/// rows before the new ones and emits only the new outputs.
#[allow(clippy::too_many_arguments)]
fn run_conv(
    stream: &Stream,
    gemm: &RocBlas,
    region: &DeviceRegion,
    conv: &mut ConvSpace,
    history: Option<&mut DecoderHistory>,
    weights: &Conv,
    input: DeviceSlice<'_, f32>,
    output: DeviceSliceMut<'_, f32>,
    input_length: usize,
    output_length: usize,
) -> Result<(), String> {
    let keep = if weights.transpose {
        (weights.kernel - 1) / weights.stride
    } else {
        (weights.kernel - 1) * weights.dilation
    };
    let Some(active) = history.filter(|_| keep != 0) else {
        let columns = conv
            .columns
            .as_mut()
            .ok_or("Qwen3-TTS speech decoder workspace is missing")?;
        return run_conv_raw(
            stream,
            gemm,
            region,
            columns,
            weights,
            input,
            output,
            input_length,
            output_length,
        );
    };
    let channels = weights.input_channels;
    let entry = active.convolutions.entry(weights.id).or_default();
    entry.channels = channels;
    grow(stream, &mut entry.input, keep * channels)?;
    let previous = entry.rows;
    let rows = previous + input_length;
    let stride = if weights.transpose { weights.stride } else { 1 };
    grow(stream, &mut conv.history_input, rows * channels)?;
    grow(
        stream,
        &mut conv.history_output,
        rows * stride * weights.output_channels,
    )?;
    let (Some(joined), Some(joined_output), Some(columns), Some(retained)) = (
        conv.history_input.as_mut(),
        conv.history_output.as_mut(),
        conv.columns.as_mut(),
        entry.input.as_mut(),
    ) else {
        return Err("Qwen3-TTS speech decoder workspace is missing".into());
    };
    if previous != 0 {
        stream
            .copy(
                retained.slice(0..previous * channels),
                joined.slice_mut(0..previous * channels),
            )
            .map_err(hip("copy convolution history"))?;
    }
    stream
        .copy(
            input.range(0..input_length * channels),
            joined.slice_mut(previous * channels..rows * channels),
        )
        .map_err(hip("append convolution input"))?;
    entry.rows = keep.min(rows);
    stream
        .copy(
            joined.slice((rows - entry.rows) * channels..rows * channels),
            retained.slice_mut(0..entry.rows * channels),
        )
        .map_err(hip("retain convolution history"))?;
    run_conv_raw(
        stream,
        gemm,
        region,
        columns,
        weights,
        joined.as_slice(),
        joined_output.as_mut_slice(),
        rows,
        rows * stride,
    )?;
    let begin = previous * stride * weights.output_channels;
    stream
        .copy(
            joined_output.slice(begin..begin + output_length * weights.output_channels),
            output.range(0..output_length * weights.output_channels),
        )
        .map_err(hip("copy incremental convolution output"))
}

/// `F32Gemm::Run`: `output[rows, output_columns] = input[rows, reduction] ·
/// weight[output_columns, reduction]ᵀ` through `rocblas_gemm_ex`.
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
        return Err("invalid Qwen3-TTS speech decoder GEMM shape".into());
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
