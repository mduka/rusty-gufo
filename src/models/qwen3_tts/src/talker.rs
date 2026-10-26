//! The talker and its code predictor (`hip/talker_runtime.cpp`): text
//! projection, prompt assembly, prefill, one cached talker step per codec
//! frame and the sixteen-group frame prediction. Launch order, kernel routes
//! (grouped GEMV, then single GEMV, then hipBLAS) and rounding boundaries
//! are the C++ ones.

use gufo_hip::blas::{Blas, Op};
use gufo_hip::{
    Bf16, DeviceBuffer, DeviceCopy, DeviceIndices, DeviceSlice, DeviceSliceMut, GgmlType, Launch,
    LaunchError, PinnedBuffer, QuantSlice, Stream,
};
use gufo_qwen_kernels as qk;
use gufo_qwen3_tts_kernels as k;

use crate::checkpoint::Checkpoint;
use crate::config::{CodePredictorConfig, ModelConfig, TalkerConfig};
use crate::prompt::{
    IclPromptInput, NonIclPromptInput, PromptOutput, build_icl_prompt, build_non_icl_prompt,
};
use crate::sampling::{Mt19937, SamplingOptions, TokenScore, sample_codec};
use crate::weights::{DeviceRegion, View, persistent_indices, resolve_bf16, upload_norm};

pub type TalkerPromptOutput = PromptOutput;

/// Codec codes below this bound index every embedding table; generated and
/// encoded codes stay below it (the talker's EOS ends a generation).
const CODEC_TABLE_ROWS: usize = 2048;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TalkerPrefillOutput {
    pub logits: Vec<f32>,
    pub last_hidden: Vec<f32>,
    /// Only filled by [`Talker::prefill_trace`].
    pub layer0_output: Vec<f32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TalkerGenerationOutput {
    pub codes: Vec<u32>,
    pub frames: usize,
    pub code_groups: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CodePredictorOutput {
    pub codes: Vec<u32>,
    pub logits: Vec<f32>,
    /// Trace only: projected `[talker hidden, first-code embedding]` rows.
    pub projected_input: Vec<f32>,
}

pub struct VoiceClonePromptInput<'a> {
    pub input_ids: &'a [u32],
    pub reference_ids: &'a [u32],
    /// Frame-major `[reference_frames, 16]`.
    pub reference_codes: &'a [u32],
    pub speaker_embedding: &'a [f32],
    pub reference_frames: usize,
    pub language: &'a str,
    pub icl_mode: bool,
}

fn launch(error: LaunchError) -> String {
    format!("Qwen3-TTS talker: {error}")
}

fn hip(operation: &str) -> impl Fn(gufo_hip::HipError) -> String + '_ {
    move |error| format!("{operation}: {error}")
}

fn alloc<T: DeviceCopy>(stream: &Stream, len: usize) -> Result<DeviceBuffer<T>, String> {
    stream.alloc_persistent(len).map_err(hip("hipMalloc"))
}

fn pinned<T: DeviceCopy>(len: usize) -> Result<PinnedBuffer<T>, String> {
    PinnedBuffer::zeroed(len).map_err(hip("hipHostMalloc"))
}

struct LayerWeights {
    q: View<Bf16>,
    k: View<Bf16>,
    v: View<Bf16>,
    o: View<Bf16>,
    gate: View<Bf16>,
    up: View<Bf16>,
    down: View<Bf16>,
    input_norm: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    post_norm: DeviceBuffer<f32>,
}

/// The dimensions one layer stack runs with.
#[derive(Clone, Copy)]
struct StackShape {
    hidden_size: usize,
    intermediate_size: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    epsilon: f32,
    rope_theta: f32,
}

impl StackShape {
    fn talker(config: &TalkerConfig) -> Self {
        Self {
            hidden_size: config.hidden_size as usize,
            intermediate_size: config.intermediate_size as usize,
            heads: config.num_attention_heads as usize,
            kv_heads: config.num_key_value_heads as usize,
            head_dim: config.head_dim as usize,
            epsilon: config.rms_norm_eps,
            rope_theta: config.rope_theta,
        }
    }

    fn predictor(config: &CodePredictorConfig) -> Self {
        Self {
            hidden_size: config.hidden_size as usize,
            intermediate_size: config.intermediate_size as usize,
            heads: config.num_attention_heads as usize,
            kv_heads: config.num_key_value_heads as usize,
            head_dim: config.head_dim as usize,
            epsilon: config.rms_norm_eps,
            rope_theta: config.rope_theta,
        }
    }
}

/// The stream, its hipBLAS handle and the talker weights region.
struct Device {
    stream: Stream,
    blas: Blas,
    region: DeviceRegion,
}

struct Weights {
    layers: Vec<LayerWeights>,
    predictor_layers: Vec<LayerWeights>,
    final_norm: DeviceBuffer<f32>,
    predictor_norm: DeviceBuffer<f32>,
    codec_head: View<Bf16>,
    codec_embedding: View<Bf16>,
    text_embedding: View<Bf16>,
    text_projection_fc1: View<Bf16>,
    text_projection_fc1_bias: View<Bf16>,
    text_projection_fc2: View<Bf16>,
    text_projection_fc2_bias: View<Bf16>,
    predictor_projection: View<Bf16>,
    predictor_projection_bias: View<Bf16>,
    predictor_embeddings: Vec<View<Bf16>>,
    predictor_heads: Vec<View<Bf16>>,
}

struct Buffers {
    hidden: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    attention_output: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    feed_forward_output: DeviceBuffer<f32>,
    scratch: DeviceBuffer<Bf16>,
    logits: DeviceBuffer<f32>,
    text_ids: DeviceIndices,
    codec_ids: DeviceIndices,
    code_ids: DeviceIndices,
    codec_tables: DeviceBuffer<u8>,
    predictor_token: DeviceBuffer<u32>,
    host_logits: PinnedBuffer<f32>,
    host_hidden: PinnedBuffer<f32>,
    host_upload: PinnedBuffer<f32>,
    host_codes: PinnedBuffer<u32>,
    host_token: PinnedBuffer<u32>,
}

/// An f32 KV cache `[layer][kv_head][capacity][head_dim]`.
struct KvCache {
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
}

/// How the code predictor picks the fifteen sub-codebook tokens: greedy on
/// the device without a random engine, otherwise sampled on the host.
struct PredictorSampling<'a> {
    random: Option<&'a mut Mt19937>,
    top_k: usize,
    temperature: f32,
    top_p: f32,
}

/// `TalkerHipRuntime`.
pub struct Talker {
    config: ModelConfig,
    maximum_tokens: usize,
    cache_tokens: usize,
    device: Device,
    weights: Weights,
    buffers: Buffers,
    cache: KvCache,
    predictor_cache: KvCache,
    candidate_scratch: Vec<TokenScore>,
    predictor_logits: Vec<f32>,
    generated_first_code_flags: Vec<u8>,
}

fn layer_name(layer: usize, suffix: &str) -> String {
    format!("talker.model.layers.{layer}.{suffix}")
}

fn predictor_layer_name(layer: usize, suffix: &str) -> String {
    format!("talker.code_predictor.model.layers.{layer}.{suffix}")
}

fn load_layer(
    stream: &Stream,
    checkpoint: &Checkpoint,
    region: &DeviceRegion,
    shape: StackShape,
    name: impl Fn(&str) -> String,
) -> Result<LayerWeights, String> {
    let hidden = shape.hidden_size as u64;
    let q_rows = (shape.heads * shape.head_dim) as u64;
    let kv_rows = (shape.kv_heads * shape.head_dim) as u64;
    let ffn = shape.intermediate_size as u64;
    let matrix = |suffix: &str, rows: u64, columns: u64| {
        resolve_bf16(checkpoint, region, &name(suffix), &[rows, columns])
    };
    Ok(LayerWeights {
        q: matrix("self_attn.q_proj.weight", q_rows, hidden)?,
        k: matrix("self_attn.k_proj.weight", kv_rows, hidden)?,
        v: matrix("self_attn.v_proj.weight", kv_rows, hidden)?,
        o: matrix("self_attn.o_proj.weight", hidden, q_rows)?,
        gate: matrix("mlp.gate_proj.weight", ffn, hidden)?,
        up: matrix("mlp.up_proj.weight", ffn, hidden)?,
        down: matrix("mlp.down_proj.weight", hidden, ffn)?,
        input_norm: upload_norm(
            stream,
            checkpoint,
            &name("input_layernorm.weight"),
            shape.hidden_size,
        )?,
        q_norm: upload_norm(
            stream,
            checkpoint,
            &name("self_attn.q_norm.weight"),
            shape.head_dim,
        )?,
        k_norm: upload_norm(
            stream,
            checkpoint,
            &name("self_attn.k_norm.weight"),
            shape.head_dim,
        )?,
        post_norm: upload_norm(
            stream,
            checkpoint,
            &name("post_attention_layernorm.weight"),
            shape.hidden_size,
        )?,
    })
}

/// `Gemm`: the model-private GEMV for one to four rows, else hipBLAS.
#[allow(clippy::too_many_arguments)]
fn gemm(
    device: &Device,
    weights: View<Bf16>,
    inputs: DeviceSlice<'_, Bf16>,
    mut output: DeviceSliceMut<'_, f32>,
    batch: usize,
    rows: usize,
    columns: usize,
) -> Result<(), String> {
    let weights = device.region.slice(weights)?;
    let inputs = inputs.range(0..batch * columns);
    let route = k::bf16_gemv(
        &device.stream,
        weights,
        inputs,
        output.reborrow(),
        batch,
        rows,
        columns,
    )
    .map_err(launch)?;
    if matches!(route, Launch::Launched) {
        return Ok(());
    }
    device
        .blas
        .gemm(
            &device.stream,
            Op::T,
            Op::N,
            rows,
            batch,
            columns,
            weights,
            columns,
            inputs,
            columns,
            output,
            rows,
        )
        .map_err(launch)
}

/// One projection of a grouped GEMV.
struct Group<'a> {
    weights: View<Bf16>,
    output: DeviceSliceMut<'a, f32>,
    rows: usize,
}

/// `GemmGrouped`: projections sharing one input in one launch when the
/// grouped GEMV accepts them, otherwise one [`gemm`] each.
fn gemm_grouped(
    device: &Device,
    groups: &mut [Group<'_>],
    inputs: DeviceSlice<'_, Bf16>,
    batch: usize,
    columns: usize,
) -> Result<(), String> {
    let inputs = inputs.range(0..batch * columns);
    let route = match groups {
        [first, second] => k::bf16_gemv_grouped(
            &device.stream,
            device.region.slice(first.weights)?,
            first.output.reborrow(),
            first.rows,
            device.region.slice(second.weights)?,
            second.output.reborrow(),
            second.rows,
            None,
            None,
            0,
            inputs,
            batch,
            columns,
        ),
        [first, second, third] => k::bf16_gemv_grouped(
            &device.stream,
            device.region.slice(first.weights)?,
            first.output.reborrow(),
            first.rows,
            device.region.slice(second.weights)?,
            second.output.reborrow(),
            second.rows,
            Some(device.region.slice(third.weights)?),
            Some(third.output.reborrow()),
            third.rows,
            inputs,
            batch,
            columns,
        ),
        _ => Ok(Launch::Unsupported),
    }
    .map_err(launch)?;
    if matches!(route, Launch::Launched) {
        return Ok(());
    }
    for group in groups {
        gemm(
            device,
            group.weights,
            inputs,
            group.output.reborrow(),
            batch,
            group.rows,
            columns,
        )?;
    }
    Ok(())
}

/// The caches and the window one stack runs against.
struct Caches<'a> {
    key: DeviceSliceMut<'a, f32>,
    value: DeviceSliceMut<'a, f32>,
    /// Positions per layer and KV head.
    capacity: usize,
}

/// One decoder stack (`Prefill`, `RunTalkerToken`, `RunPredictor`): every
/// layer, ending in the final RMSNorm into the BF16 scratch. Returns the
/// hidden state after layer 0 when `capture_layer0`.
#[allow(clippy::too_many_arguments)]
fn run_stack(
    device: &Device,
    buffers: &mut Buffers,
    layers: &[LayerWeights],
    final_norm: &DeviceBuffer<f32>,
    shape: StackShape,
    tokens: usize,
    start_position: usize,
    caches: Caches<'_>,
    capture_layer0: bool,
) -> Result<Option<Vec<f32>>, String> {
    let stream = &device.stream;
    let hidden_elements = tokens * shape.hidden_size;
    let ffn_elements = tokens * shape.intermediate_size;
    let q_rows = shape.heads * shape.head_dim;
    let kv_rows = shape.kv_heads * shape.head_dim;
    let Caches {
        mut key,
        mut value,
        capacity,
    } = caches;
    let mut layer0 = None;
    k::bf16_residual_add_rms_norm(
        stream,
        buffers.hidden.as_mut_slice(),
        None,
        Some(layers[0].input_norm.as_slice()),
        buffers.scratch.as_mut_slice(),
        tokens,
        shape.hidden_size,
        shape.epsilon,
    )
    .map_err(launch)?;
    for (index, weights) in layers.iter().enumerate() {
        gemm_grouped(
            device,
            &mut [
                Group {
                    weights: weights.q,
                    output: buffers.q.as_mut_slice(),
                    rows: q_rows,
                },
                Group {
                    weights: weights.k,
                    output: buffers.k.as_mut_slice(),
                    rows: kv_rows,
                },
                Group {
                    weights: weights.v,
                    output: buffers.v.as_mut_slice(),
                    rows: kv_rows,
                },
            ],
            buffers.scratch.as_slice(),
            tokens,
            shape.hidden_size,
        )?;
        let cached = normalize_rotate_qkv(
            device,
            buffers,
            weights,
            shape,
            tokens,
            start_position,
            key.reborrow(),
            value.reborrow(),
            index,
            capacity,
        )?;
        k::bf16_attention(
            stream,
            buffers.q.as_slice(),
            buffers.k.as_slice(),
            buffers.v.as_slice(),
            key.reborrow(),
            value.reborrow(),
            buffers.scratch.as_mut_slice(),
            index,
            start_position,
            tokens,
            capacity,
            shape.heads,
            shape.kv_heads,
            shape.head_dim,
            cached,
        )
        .map_err(launch)?;
        gemm(
            device,
            weights.o,
            buffers.scratch.as_slice(),
            buffers.attention_output.as_mut_slice(),
            tokens,
            shape.hidden_size,
            q_rows,
        )?;
        k::bf16_residual_add_rms_norm(
            stream,
            buffers.hidden.as_mut_slice(),
            Some(buffers.attention_output.as_slice()),
            Some(weights.post_norm.as_slice()),
            buffers.scratch.as_mut_slice(),
            tokens,
            shape.hidden_size,
            shape.epsilon,
        )
        .map_err(launch)?;
        gemm_grouped(
            device,
            &mut [
                Group {
                    weights: weights.gate,
                    output: buffers.gate.as_mut_slice(),
                    rows: shape.intermediate_size,
                },
                Group {
                    weights: weights.up,
                    output: buffers.up.as_mut_slice(),
                    rows: shape.intermediate_size,
                },
            ],
            buffers.scratch.as_slice(),
            tokens,
            shape.hidden_size,
        )?;
        k::bf16_swiglu(
            stream,
            buffers.gate.as_slice(),
            buffers.up.as_slice(),
            None,
            Some(buffers.scratch.as_mut_slice()),
            ffn_elements,
        )
        .map_err(launch)?;
        gemm(
            device,
            weights.down,
            buffers.scratch.as_slice(),
            buffers.feed_forward_output.as_mut_slice(),
            tokens,
            shape.hidden_size,
            shape.intermediate_size,
        )?;
        // The next layer opens with an RMSNorm, so the trailing residual
        // folds into it; the last layer folds into the final norm.
        let next_norm = layers
            .get(index + 1)
            .map_or(final_norm, |next| &next.input_norm);
        k::bf16_residual_add_rms_norm(
            stream,
            buffers.hidden.as_mut_slice(),
            Some(buffers.feed_forward_output.as_slice()),
            Some(next_norm.as_slice()),
            buffers.scratch.as_mut_slice(),
            tokens,
            shape.hidden_size,
            shape.epsilon,
        )
        .map_err(launch)?;
        if index == 0 && capture_layer0 {
            let mut host = vec![0.0f32; hidden_elements];
            stream
                .download(buffers.hidden.slice(0..hidden_elements), &mut host)
                .map_err(hip("hipMemcpyAsync layer0 output"))?;
            layer0 = Some(host);
        }
    }
    Ok(layer0)
}

/// `NormalizeRotateQkv`: the fused q/k norm and rotation (with the cache
/// write) where the head fits a workgroup, else the separate kernels.
/// Returns whether the cache was written.
#[allow(clippy::too_many_arguments)]
fn normalize_rotate_qkv(
    device: &Device,
    buffers: &mut Buffers,
    weights: &LayerWeights,
    shape: StackShape,
    tokens: usize,
    start_position: usize,
    key_cache: DeviceSliceMut<'_, f32>,
    value_cache: DeviceSliceMut<'_, f32>,
    layer: usize,
    capacity: usize,
) -> Result<bool, String> {
    let stream = &device.stream;
    let route = k::bf16_qk_norm_rope(
        stream,
        buffers.q.as_mut_slice(),
        buffers.k.as_mut_slice(),
        buffers.v.as_mut_slice(),
        Some(weights.q_norm.as_slice()),
        Some(weights.k_norm.as_slice()),
        tokens,
        shape.heads as u32,
        shape.kv_heads as u32,
        shape.head_dim as u32,
        start_position as u32,
        shape.rope_theta,
        shape.epsilon,
        Some(key_cache),
        Some(value_cache),
        layer as u32,
        capacity as u32,
    )
    .map_err(launch)?;
    if matches!(route, Launch::Launched) {
        return Ok(true);
    }
    k::bf16_per_head_rms_norm(
        stream,
        buffers.q.as_mut_slice(),
        Some(weights.q_norm.as_slice()),
        tokens,
        shape.heads as u32,
        shape.head_dim as u32,
        shape.epsilon,
    )
    .map_err(launch)?;
    k::bf16_per_head_rms_norm(
        stream,
        buffers.k.as_mut_slice(),
        Some(weights.k_norm.as_slice()),
        tokens,
        shape.kv_heads as u32,
        shape.head_dim as u32,
        shape.epsilon,
    )
    .map_err(launch)?;
    k::bf16_rope_and_round_v(
        stream,
        buffers.q.as_mut_slice(),
        buffers.k.as_mut_slice(),
        buffers.v.as_mut_slice(),
        tokens,
        shape.heads as u32,
        shape.kv_heads as u32,
        shape.head_dim as u32,
        start_position as u32,
        shape.rope_theta,
    )
    .map_err(launch)?;
    Ok(false)
}

/// The table of every code group (`codec_embedding_tables`): the talker's
/// codec embedding, then the predictor's fifteen.
fn codec_tables<'a>(
    device: &'a Device,
    weights: &Weights,
) -> Result<Vec<DeviceSlice<'a, Bf16>>, String> {
    std::iter::once(weights.codec_embedding)
        .chain(weights.predictor_embeddings.iter().copied())
        .map(|view| device.region.slice(view))
        .collect()
}

/// `std::tolower` of every byte in the "C" locale.
fn ascii_lower(value: &str) -> String {
    value.to_ascii_lowercase()
}

impl Talker {
    /// `TalkerHipRuntime::Create`.
    pub fn new(checkpoint: &Checkpoint, maximum_tokens: usize) -> Result<Self, String> {
        if maximum_tokens == 0 {
            return Err("Qwen3-TTS maximum token count must be positive".into());
        }
        let config = checkpoint.config.clone();
        let talker = &config.talker;
        let predictor = &config.code_predictor;
        let regions = checkpoint.regions("talker.")?;
        let [region] = regions.as_slice() else {
            return Err("cannot initialize Qwen3-TTS GPU weight mapping".into());
        };
        let stream = Stream::new().map_err(hip("hipStreamCreateWithFlags"))?;
        let region = DeviceRegion::upload(&stream, checkpoint, *region)
            .map_err(|_| "cannot initialize Qwen3-TTS GPU weight mapping".to_string())?;
        let blas = Blas::new(&stream).map_err(hip("hipblasCreate"))?;
        let device = Device {
            stream,
            blas,
            region,
        };
        let weights = Self::load_weights(&device, checkpoint, &config)?;

        let stream = &device.stream;
        let hidden = talker.hidden_size as usize;
        let ffn = talker.intermediate_size as usize;
        let kv = (talker.num_key_value_heads * talker.head_dim) as usize;
        let groups = talker.num_code_groups as usize;
        let buffers = Buffers {
            hidden: alloc(stream, maximum_tokens * hidden)?,
            normalized: alloc(stream, maximum_tokens * hidden)?,
            q: alloc(stream, maximum_tokens * hidden)?,
            k: alloc(stream, maximum_tokens * kv)?,
            v: alloc(stream, maximum_tokens * kv)?,
            attention_output: alloc(stream, maximum_tokens * hidden)?,
            gate: alloc(stream, maximum_tokens * ffn)?,
            up: alloc(stream, maximum_tokens * ffn)?,
            feed_forward_output: alloc(stream, maximum_tokens * hidden)?,
            scratch: alloc(stream, maximum_tokens * hidden.max(ffn))?,
            logits: alloc(stream, talker.vocab_size as usize)?,
            text_ids: persistent_indices(stream, maximum_tokens, talker.text_vocab_size)
                .map_err(hip("hipMalloc"))?,
            codec_ids: persistent_indices(stream, maximum_tokens, talker.vocab_size)
                .map_err(hip("hipMalloc"))?,
            code_ids: persistent_indices(stream, maximum_tokens * groups, CODEC_TABLE_ROWS as u32)
                .map_err(hip("hipMalloc"))?,
            codec_tables: alloc(stream, groups * 8)?,
            predictor_token: alloc(stream, 1)?,
            host_logits: pinned(talker.vocab_size as usize)?,
            host_hidden: pinned(hidden)?,
            host_upload: pinned(hidden)?,
            host_codes: pinned(groups)?,
            host_token: pinned(1)?,
        };
        let cache_elements = talker.num_hidden_layers as usize * kv * maximum_tokens;
        let cache = KvCache {
            key: alloc(stream, cache_elements)?,
            value: alloc(stream, cache_elements)?,
        };
        let predictor_elements = predictor.num_hidden_layers as usize
            * (predictor.num_key_value_heads * predictor.head_dim) as usize
            * groups;
        let predictor_cache = KvCache {
            key: alloc(stream, predictor_elements)?,
            value: alloc(stream, predictor_elements)?,
        };
        Ok(Self {
            maximum_tokens,
            cache_tokens: 0,
            device,
            weights,
            buffers,
            cache,
            predictor_cache,
            candidate_scratch: Vec::new(),
            predictor_logits: Vec::new(),
            generated_first_code_flags: Vec::new(),
            config,
        })
    }

    fn load_weights(
        device: &Device,
        checkpoint: &Checkpoint,
        config: &ModelConfig,
    ) -> Result<Weights, String> {
        let stream = &device.stream;
        let region = &device.region;
        let talker = &config.talker;
        let predictor = &config.code_predictor;
        let shape = StackShape::talker(talker);
        let layers = (0..talker.num_hidden_layers as usize)
            .map(|layer| load_layer(stream, checkpoint, region, shape, |s| layer_name(layer, s)))
            .collect::<Result<Vec<_>, _>>()?;
        let hidden = u64::from(talker.hidden_size);
        let text_hidden = u64::from(talker.text_hidden_size);
        let vocab = u64::from(talker.vocab_size);
        let matrix = |name: &str, rows: u64, columns: u64| {
            resolve_bf16(checkpoint, region, name, &[rows, columns])
        };
        let vector =
            |name: &str, elements: u64| resolve_bf16(checkpoint, region, name, &[elements]);
        let final_norm = upload_norm(
            stream,
            checkpoint,
            "talker.model.norm.weight",
            talker.hidden_size as usize,
        )?;
        let codec_head = matrix("talker.codec_head.weight", vocab, hidden)?;
        let codec_embedding = matrix("talker.model.codec_embedding.weight", vocab, hidden)?;
        let text_embedding = matrix(
            "talker.model.text_embedding.weight",
            u64::from(talker.text_vocab_size),
            text_hidden,
        )?;
        let text_projection_fc1 = matrix(
            "talker.text_projection.linear_fc1.weight",
            text_hidden,
            text_hidden,
        )?;
        let text_projection_fc1_bias =
            vector("talker.text_projection.linear_fc1.bias", text_hidden)?;
        let text_projection_fc2 = matrix(
            "talker.text_projection.linear_fc2.weight",
            hidden,
            text_hidden,
        )?;
        let text_projection_fc2_bias = vector("talker.text_projection.linear_fc2.bias", hidden)?;

        let predictor_shape = StackShape::predictor(predictor);
        let predictor_layers = (0..predictor.num_hidden_layers as usize)
            .map(|layer| {
                load_layer(stream, checkpoint, region, predictor_shape, |s| {
                    predictor_layer_name(layer, s)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let predictor_norm = upload_norm(
            stream,
            checkpoint,
            "talker.code_predictor.model.norm.weight",
            predictor.hidden_size as usize,
        )?;
        let predictor_hidden = u64::from(predictor.hidden_size);
        let predictor_vocab = u64::from(predictor.vocab_size);
        let predictor_projection = matrix(
            "talker.code_predictor.small_to_mtp_projection.weight",
            predictor_hidden,
            hidden,
        )?;
        let predictor_projection_bias = vector(
            "talker.code_predictor.small_to_mtp_projection.bias",
            predictor_hidden,
        )?;
        let mut predictor_embeddings = Vec::new();
        let mut predictor_heads = Vec::new();
        for group in 0..talker.num_code_groups.saturating_sub(1) {
            predictor_embeddings.push(matrix(
                &format!("talker.code_predictor.model.codec_embedding.{group}.weight"),
                predictor_vocab,
                hidden,
            )?);
            predictor_heads.push(matrix(
                &format!("talker.code_predictor.lm_head.{group}.weight"),
                predictor_vocab,
                predictor_hidden,
            )?);
        }
        Ok(Weights {
            layers,
            predictor_layers,
            final_norm,
            predictor_norm,
            codec_head,
            codec_embedding,
            text_embedding,
            text_projection_fc1,
            text_projection_fc1_bias,
            text_projection_fc2,
            text_projection_fc2_bias,
            predictor_projection,
            predictor_projection_bias,
            predictor_embeddings,
            predictor_heads,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn maximum_tokens(&self) -> usize {
        self.maximum_tokens
    }

    /// `ProjectText`: text embeddings through the two-layer projection,
    /// `[tokens, hidden]` on the host.
    fn project_text(&mut self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        let config = &self.config.talker;
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        if token_ids.len() > self.maximum_tokens {
            return Err("Qwen3-TTS text projection exceeds capacity".into());
        }
        if token_ids
            .iter()
            .any(|&token| token >= config.text_vocab_size)
        {
            return Err("Qwen3-TTS text token is out of range".into());
        }
        let count = token_ids.len();
        let text_hidden = config.text_hidden_size as usize;
        let hidden = config.hidden_size as usize;
        let device = &self.device;
        let stream = &device.stream;
        let buffers = &mut self.buffers;
        let weights = &self.weights;
        stream
            .upload_indices(token_ids, &mut buffers.text_ids)
            .map_err(hip("hipMemcpyAsync text token ids"))?;
        qk::batched_embedding_lookup(
            stream,
            QuantSlice::new(device.region.bytes(weights.text_embedding), GgmlType::Bf16),
            buffers.text_ids.slice(0..count),
            buffers.hidden.as_mut_slice(),
            count,
            text_hidden,
            config.text_vocab_size as usize,
        )
        .map_err(launch)?;
        qk::float_to_bf16(
            stream,
            buffers.hidden.slice(0..count * text_hidden),
            buffers.scratch.as_mut_slice(),
            count * text_hidden,
        )
        .map_err(launch)?;
        gemm(
            device,
            weights.text_projection_fc1,
            buffers.scratch.as_slice(),
            buffers.normalized.as_mut_slice(),
            count,
            text_hidden,
            text_hidden,
        )?;
        k::bf16_bias_silu(
            stream,
            buffers.normalized.slice(0..count * text_hidden),
            device.region.slice(weights.text_projection_fc1_bias)?,
            Some(buffers.hidden.as_mut_slice()),
            Some(buffers.scratch.as_mut_slice()),
            count,
            text_hidden,
        )
        .map_err(launch)?;
        gemm(
            device,
            weights.text_projection_fc2,
            buffers.scratch.as_slice(),
            buffers.normalized.as_mut_slice(),
            count,
            hidden,
            text_hidden,
        )?;
        k::bf16_bias(
            stream,
            buffers.normalized.slice(0..count * hidden),
            device.region.slice(weights.text_projection_fc2_bias)?,
            buffers.hidden.as_mut_slice(),
            count,
            hidden,
        )
        .map_err(launch)?;
        let mut result = vec![0.0f32; count * hidden];
        stream
            .download(buffers.hidden.slice(0..count * hidden), &mut result)
            .map_err(hip("hipMemcpyAsync projected text"))?;
        Ok(result)
    }

    /// `LookupCodec`: codec embeddings, `[tokens, hidden]` on the host.
    fn lookup_codec(&mut self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        let config = &self.config.talker;
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        if token_ids.len() > self.maximum_tokens {
            return Err("Qwen3-TTS codec lookup exceeds capacity".into());
        }
        if token_ids.iter().any(|&token| token >= config.vocab_size) {
            return Err("Qwen3-TTS codec token is out of range".into());
        }
        let count = token_ids.len();
        let hidden = config.hidden_size as usize;
        let stream = &self.device.stream;
        stream
            .upload_indices(token_ids, &mut self.buffers.codec_ids)
            .map_err(hip("hipMemcpyAsync codec token ids"))?;
        qk::batched_embedding_lookup(
            stream,
            QuantSlice::new(
                self.device.region.bytes(self.weights.codec_embedding),
                GgmlType::Bf16,
            ),
            self.buffers.codec_ids.slice(0..count),
            self.buffers.hidden.as_mut_slice(),
            count,
            hidden,
            config.vocab_size as usize,
        )
        .map_err(launch)?;
        let mut result = vec![0.0f32; count * hidden];
        stream
            .download(self.buffers.hidden.slice(0..count * hidden), &mut result)
            .map_err(hip("hipMemcpyAsync codec embeddings"))?;
        Ok(result)
    }

    /// `ProjectPredictorInput`: `rows` talker-width rows of
    /// `attention_output` projected to the predictor width into `hidden`.
    fn project_predictor_input(&mut self, rows: usize) -> Result<(), String> {
        let talker = &self.config.talker;
        let predictor = &self.config.code_predictor;
        if rows == 0 || rows > talker.num_code_groups as usize {
            return Err("Qwen3-TTS predictor input shape is invalid".into());
        }
        let hidden = talker.hidden_size as usize;
        let predictor_hidden = predictor.hidden_size as usize;
        let device = &self.device;
        let buffers = &mut self.buffers;
        qk::float_to_bf16(
            &device.stream,
            buffers.attention_output.slice(0..rows * hidden),
            buffers.scratch.as_mut_slice(),
            rows * hidden,
        )
        .map_err(launch)?;
        gemm(
            device,
            self.weights.predictor_projection,
            buffers.scratch.as_slice(),
            buffers.hidden.as_mut_slice(),
            rows,
            predictor_hidden,
            hidden,
        )?;
        k::bf16_bias_in_place(
            &device.stream,
            device
                .region
                .slice(self.weights.predictor_projection_bias)?,
            buffers.hidden.slice_mut(0..rows * predictor_hidden),
            rows,
            predictor_hidden,
        )
        .map_err(launch)
    }

    /// `RunPredictor`: the predictor stack over `tokens` rows at
    /// `start_position`, then head `head_index`; the sampled or greedy code.
    fn run_predictor(
        &mut self,
        tokens: usize,
        start_position: usize,
        head_index: usize,
        trace_logits: Option<&mut Vec<f32>>,
        sampling: &mut PredictorSampling<'_>,
    ) -> Result<u32, String> {
        let predictor = self.config.code_predictor.clone();
        let context = self.config.talker.num_code_groups as usize;
        if tokens == 0
            || start_position + tokens > context
            || head_index >= self.weights.predictor_heads.len()
        {
            return Err("Qwen3-TTS predictor step is out of range".into());
        }
        let hidden = predictor.hidden_size as usize;
        let vocab = predictor.vocab_size as usize;
        run_stack(
            &self.device,
            &mut self.buffers,
            &self.weights.predictor_layers,
            &self.weights.predictor_norm,
            StackShape::predictor(&predictor),
            tokens,
            start_position,
            Caches {
                key: self.predictor_cache.key.as_mut_slice(),
                value: self.predictor_cache.value.as_mut_slice(),
                capacity: context,
            },
            false,
        )?;
        let device = &self.device;
        let stream = &device.stream;
        let buffers = &mut self.buffers;
        gemm(
            device,
            self.weights.predictor_heads[head_index],
            buffers
                .scratch
                .slice((tokens - 1) * hidden..tokens * hidden),
            buffers.logits.as_mut_slice(),
            1,
            vocab,
            hidden,
        )?;
        k::round_bf16_in_place(stream, buffers.logits.slice_mut(0..vocab), vocab)
            .map_err(launch)?;
        let wants_logits = sampling.random.is_some() || trace_logits.is_some();
        if wants_logits {
            stream
                .download_async_front(buffers.logits.slice(0..vocab), &mut buffers.host_logits)
                .map_err(hip("hipMemcpyAsync predictor logits"))?;
        }
        if let Some(random) = sampling.random.as_deref_mut() {
            let host = buffers
                .host_logits
                .host()
                .map_err(hip("hipStreamSynchronize predictor sampling"))?;
            let logits = &host[..vocab];
            self.predictor_logits.clear();
            self.predictor_logits.extend_from_slice(logits);
            if let Some(trace) = trace_logits {
                trace.clone_from(&self.predictor_logits);
            }
            self.candidate_scratch.clear();
            self.candidate_scratch
                .extend(
                    self.predictor_logits
                        .iter()
                        .enumerate()
                        .map(|(token, &score)| TokenScore {
                            token: token as u32,
                            score,
                        }),
                );
            return sample_codec(
                &mut self.candidate_scratch,
                sampling.top_k,
                sampling.top_p,
                sampling.temperature,
                Some(random),
            );
        }
        qk::argmax(
            stream,
            buffers.logits.slice(0..vocab),
            buffers.predictor_token.as_mut_slice(),
            vocab,
        )
        .map_err(launch)?;
        stream
            .download_async(buffers.predictor_token.as_slice(), &mut buffers.host_token)
            .map_err(hip("hipMemcpyAsync predictor token"))?;
        let token = buffers
            .host_token
            .host()
            .map_err(hip("hipStreamSynchronize predictor"))?[0];
        if let Some(trace) = trace_logits {
            let host = buffers
                .host_logits
                .host()
                .map_err(hip("hipStreamSynchronize predictor"))?;
            trace.clear();
            trace.extend_from_slice(&host[..vocab]);
        }
        if token as usize >= vocab {
            return Err("logit distribution contains no finite values".into());
        }
        Ok(token)
    }

    /// Copies the BF16 last hidden row into `attention_output` as f32 and
    /// downloads it with the BF16 logits, then waits (the end of `Prefill`
    /// and of `RunTalkerToken`).
    fn finish_step(&mut self, last_row: usize) -> Result<TalkerPrefillOutput, String> {
        let config = &self.config.talker;
        let hidden = config.hidden_size as usize;
        let vocab = config.vocab_size as usize;
        let device = &self.device;
        let stream = &device.stream;
        let buffers = &mut self.buffers;
        let last = buffers
            .scratch
            .slice(last_row * hidden..(last_row + 1) * hidden);
        gemm(
            device,
            self.weights.codec_head,
            last,
            buffers.logits.as_mut_slice(),
            1,
            vocab,
            hidden,
        )?;
        k::round_bf16_in_place(stream, buffers.logits.slice_mut(0..vocab), vocab)
            .map_err(launch)?;
        k::bf16_to_f32(
            stream,
            last,
            buffers.attention_output.slice_mut(0..hidden),
            hidden,
        )
        .map_err(launch)?;
        stream
            .download_async(
                buffers.attention_output.slice(0..hidden),
                &mut buffers.host_hidden,
            )
            .map_err(hip("hipMemcpyAsync talker hidden"))?;
        stream
            .download_async_front(buffers.logits.slice(0..vocab), &mut buffers.host_logits)
            .map_err(hip("hipMemcpyAsync talker logits"))?;
        let last_hidden = buffers
            .host_hidden
            .host()
            .map_err(hip("hipStreamSynchronize talker"))?
            .to_vec();
        let logits = buffers
            .host_logits
            .host()
            .map_err(hip("hipStreamSynchronize talker"))?[..vocab]
            .to_vec();
        Ok(TalkerPrefillOutput {
            logits,
            last_hidden,
            layer0_output: Vec::new(),
        })
    }

    /// `TalkerHipRuntime::Prefill`.
    pub fn prefill(
        &mut self,
        input_embeddings: &[f32],
        tokens: usize,
    ) -> Result<TalkerPrefillOutput, String> {
        self.prefill_with(input_embeddings, tokens, false)
    }

    /// [`Talker::prefill`] that also returns the hidden state after layer 0
    /// (the first parity boundary).
    pub fn prefill_trace(
        &mut self,
        input_embeddings: &[f32],
        tokens: usize,
    ) -> Result<TalkerPrefillOutput, String> {
        self.prefill_with(input_embeddings, tokens, true)
    }

    fn prefill_with(
        &mut self,
        input_embeddings: &[f32],
        tokens: usize,
        capture_layer0: bool,
    ) -> Result<TalkerPrefillOutput, String> {
        let config = self.config.talker.clone();
        if tokens == 0
            || tokens > self.maximum_tokens
            || input_embeddings.len() != tokens * config.hidden_size as usize
        {
            return Err("Qwen3-TTS prefill embedding shape is invalid".into());
        }
        self.cache_tokens = 0;
        let elements = input_embeddings.len();
        self.device
            .stream
            .upload(input_embeddings, self.buffers.hidden.slice_mut(0..elements))
            .map_err(hip("hipMemcpyAsync prefill embeddings"))?;
        // Each layer overwrites the entire visible prefix before causal
        // attention; unused capacity is never read, even after cancellation.
        let layer0 = run_stack(
            &self.device,
            &mut self.buffers,
            &self.weights.layers,
            &self.weights.final_norm,
            StackShape::talker(&config),
            tokens,
            0,
            Caches {
                key: self.cache.key.as_mut_slice(),
                value: self.cache.value.as_mut_slice(),
                capacity: self.maximum_tokens,
            },
            capture_layer0,
        )?;
        let mut output = self.finish_step(tokens - 1)?;
        output.layer0_output = layer0.unwrap_or_default();
        self.cache_tokens = tokens;
        Ok(output)
    }

    /// `RunTalkerToken`: one cached step over the prepared frame embedding.
    fn run_talker_token(&mut self) -> Result<TalkerPrefillOutput, String> {
        if self.cache_tokens == 0 || self.cache_tokens >= self.maximum_tokens {
            return Err("Qwen3-TTS talker cache position is invalid".into());
        }
        let config = self.config.talker.clone();
        run_stack(
            &self.device,
            &mut self.buffers,
            &self.weights.layers,
            &self.weights.final_norm,
            StackShape::talker(&config),
            1,
            self.cache_tokens,
            Caches {
                key: self.cache.key.as_mut_slice(),
                value: self.cache.value.as_mut_slice(),
                capacity: self.maximum_tokens,
            },
            false,
        )?;
        let output = self.finish_step(0)?;
        self.cache_tokens += 1;
        Ok(output)
    }

    /// `PrepareCodeFrameEmbedding`: the BF16 sum of a frame's sixteen codec
    /// embeddings plus the text row, into `hidden`.
    fn prepare_code_frame_embedding(
        &mut self,
        codes: &[u32],
        text_embedding: &[f32],
    ) -> Result<(), String> {
        let talker = &self.config.talker;
        let groups = talker.num_code_groups as usize;
        let hidden = talker.hidden_size as usize;
        if codes.len() != groups {
            return Err("Qwen3-TTS codec frame group count is invalid".into());
        }
        let stream = &self.device.stream;
        let buffers = &mut self.buffers;
        buffers
            .host_upload
            .host_mut()
            .map_err(hip("hipMemcpyAsync trailing text embedding"))?
            .copy_from_slice(text_embedding);
        stream
            .upload_async(
                &mut buffers.host_upload,
                buffers.feed_forward_output.slice_mut(0..hidden),
            )
            .map_err(hip("hipMemcpyAsync trailing text embedding"))?;
        // One dispatch replaces a lookup per code group plus the summing
        // pass, accumulating the groups in the same order.
        buffers
            .host_codes
            .host_mut()
            .map_err(hip("hipMemcpyAsync codec frame codes"))?
            .copy_from_slice(codes);
        stream
            .upload_indices_async(&mut buffers.host_codes, groups, &mut buffers.code_ids)
            .map_err(hip("hipMemcpyAsync codec frame codes"))?;
        let tables = codec_tables(&self.device, &self.weights)?;
        k::codec_embedding_sum(
            stream,
            &tables,
            buffers.codec_tables.as_mut_slice(),
            buffers.code_ids.slice(0..groups),
            Some(buffers.feed_forward_output.slice(0..hidden)),
            buffers.hidden.as_mut_slice(),
            1,
            groups,
            hidden,
            CODEC_TABLE_ROWS,
        )
        .map_err(launch)
    }

    /// `BuildReferenceCodecEmbeddings`: the summed codec embedding of every
    /// reference frame, `[frames, hidden]` on the host.
    fn build_reference_codec_embeddings(
        &mut self,
        codes: &[u32],
        frames: usize,
    ) -> Result<Vec<f32>, String> {
        let talker = &self.config.talker;
        let groups = talker.num_code_groups as usize;
        let hidden = talker.hidden_size as usize;
        if frames == 0 || frames > self.maximum_tokens || codes.len() != frames * groups {
            return Err("Qwen3-TTS reference codec embedding shape is invalid".into());
        }
        let stream = &self.device.stream;
        stream
            .upload_indices(codes, &mut self.buffers.code_ids)
            .map_err(|_| "Qwen3-TTS reference codec token is out of range".to_string())?;
        let tables = codec_tables(&self.device, &self.weights)?;
        k::codec_embedding_sum(
            stream,
            &tables,
            self.buffers.codec_tables.as_mut_slice(),
            self.buffers.code_ids.slice(0..codes.len()),
            None,
            self.buffers.hidden.as_mut_slice(),
            frames,
            groups,
            hidden,
            CODEC_TABLE_ROWS,
        )
        .map_err(launch)?;
        let mut result = vec![0.0f32; frames * hidden];
        stream
            .download(self.buffers.hidden.slice(0..frames * hidden), &mut result)
            .map_err(hip("hipMemcpyAsync reference codec embeddings"))?;
        Ok(result)
    }

    /// The codec ids of the thinking/language prefix for `language`.
    fn language_codec_ids(&self, language: &str, normalized: &str) -> Result<Vec<u32>, String> {
        let talker = &self.config.talker;
        if normalized == "auto" {
            return Ok(vec![
                talker.codec_nothink_id,
                talker.codec_think_bos_id,
                talker.codec_think_eos_id,
            ]);
        }
        let language_id = talker
            .codec_language_id
            .get(normalized)
            .ok_or_else(|| format!("unsupported Qwen3-TTS language: {language}"))?;
        Ok(vec![
            talker.codec_think_id,
            talker.codec_think_bos_id,
            *language_id,
            talker.codec_think_eos_id,
        ])
    }

    /// The projected TTS BOS, EOS and pad rows.
    fn special_rows(&mut self) -> Result<Vec<f32>, String> {
        let ids = [
            self.config.tts_bos_token_id,
            self.config.tts_eos_token_id,
            self.config.tts_pad_token_id,
        ];
        self.project_text(&ids)
    }

    /// `BuildCustomVoicePrompt`.
    pub fn build_custom_voice_prompt(
        &mut self,
        input_ids: &[u32],
        instruction_ids: &[u32],
        speaker: &str,
        language: &str,
    ) -> Result<TalkerPromptOutput, String> {
        self.build_conditioned_prompt(input_ids, instruction_ids, speaker, language, true)
    }

    /// `BuildVoiceDesignPrompt`.
    pub fn build_voice_design_prompt(
        &mut self,
        input_ids: &[u32],
        instruction_ids: &[u32],
        language: &str,
    ) -> Result<TalkerPromptOutput, String> {
        self.build_conditioned_prompt(input_ids, instruction_ids, "", language, false)
    }

    fn build_conditioned_prompt(
        &mut self,
        input_ids: &[u32],
        instruction_ids: &[u32],
        speaker: &str,
        language: &str,
        include_speaker: bool,
    ) -> Result<TalkerPromptOutput, String> {
        if input_ids.len() < 8 {
            return Err("Qwen3-TTS assistant prompt is too short".into());
        }
        let talker = self.config.talker.clone();
        let mut normalized_language = ascii_lower(language);
        if normalized_language.is_empty() {
            normalized_language = "auto".into();
        }
        let mut speaker_token = None;
        if include_speaker {
            let normalized_speaker = ascii_lower(speaker);
            let id = talker
                .spk_id
                .get(&normalized_speaker)
                .ok_or_else(|| format!("unsupported Qwen3-TTS speaker: {speaker}"))?;
            speaker_token = Some(*id);
            if (normalized_language == "chinese" || normalized_language == "auto")
                && let Some(Some(dialect)) = talker.spk_dialect.get(&normalized_speaker)
            {
                normalized_language.clone_from(dialect);
            }
        }
        let mut codec_ids = self.language_codec_ids(language, &normalized_language)?;
        codec_ids.extend(speaker_token);
        codec_ids.push(talker.codec_pad_id);
        codec_ids.push(talker.codec_bos_id);

        let hidden = talker.hidden_size as usize;
        let special = self.special_rows()?;
        let instruction = if instruction_ids.is_empty() {
            Vec::new()
        } else {
            self.project_text(instruction_ids)?
        };
        let role = self.project_text(&input_ids[..3])?;
        let codec = self.lookup_codec(&codec_ids)?;
        let text = self.project_text(&input_ids[3..input_ids.len() - 5])?;
        let codec_pad = self.lookup_codec(&[talker.codec_pad_id])?;
        let output = build_non_icl_prompt(&NonIclPromptInput {
            hidden_size: hidden,
            instruction: &instruction,
            role: &role,
            codec: &codec,
            text: &text,
            tts_bos: &special[..hidden],
            tts_eos: &special[hidden..2 * hidden],
            tts_pad: &special[2 * hidden..3 * hidden],
            codec_pad: &codec_pad,
        })?;
        if output.tokens == 0 || output.tokens > self.maximum_tokens {
            return Err("Qwen3-TTS prompt exceeds runtime capacity".into());
        }
        Ok(output)
    }

    /// `BuildVoiceClonePrompt`: full ICL or speaker-embedding-only Base.
    pub fn build_voice_clone_prompt(
        &mut self,
        input: &VoiceClonePromptInput<'_>,
    ) -> Result<TalkerPromptOutput, String> {
        let talker = self.config.talker.clone();
        let predictor_vocab = self.config.code_predictor.vocab_size;
        let groups = talker.num_code_groups as usize;
        if input.input_ids.len() < 8 {
            return Err("Qwen3-TTS assistant prompt is too short".into());
        }
        if input.speaker_embedding.len() != talker.hidden_size as usize {
            return Err("Qwen3-TTS speaker embedding shape is invalid".into());
        }
        if input.icl_mode
            && (input.reference_ids.len() < 5
                || input.reference_frames == 0
                || input.reference_codes.len() != input.reference_frames * groups)
        {
            return Err("Qwen3-TTS reference prompt is invalid".into());
        }
        let mut normalized_language = ascii_lower(input.language);
        if normalized_language.is_empty() {
            normalized_language = "auto".into();
        }
        let codec_ids = self.language_codec_ids(input.language, &normalized_language)?;
        let hidden = talker.hidden_size as usize;
        let special = self.special_rows()?;
        let role = self.project_text(&input.input_ids[..3])?;
        let mut codec = self.lookup_codec(&codec_ids)?;
        codec.extend_from_slice(input.speaker_embedding);
        codec.extend(self.lookup_codec(&[talker.codec_pad_id, talker.codec_bos_id])?);
        let target_ids = &input.input_ids[3..input.input_ids.len() - 5];
        let tts_bos = &special[..hidden];
        let tts_eos = &special[hidden..2 * hidden];
        let tts_pad = &special[2 * hidden..3 * hidden];
        let output =
            if !input.icl_mode {
                let text = self.project_text(target_ids)?;
                let codec_pad = self.lookup_codec(&[talker.codec_pad_id])?;
                build_non_icl_prompt(&NonIclPromptInput {
                    hidden_size: hidden,
                    instruction: &[],
                    role: &role,
                    codec: &codec,
                    text: &text,
                    tts_bos,
                    tts_eos,
                    tts_pad,
                    codec_pad: &codec_pad,
                })?
            } else {
                let reference_text_ids = &input.reference_ids[3..input.reference_ids.len() - 2];
                let combined_ids: Vec<u32> = reference_text_ids
                    .iter()
                    .chain(target_ids)
                    .copied()
                    .collect();
                let combined_text = self.project_text(&combined_ids)?;
                let mut reference_codec = self.lookup_codec(&[talker.codec_bos_id])?;
                for frame in input.reference_codes.chunks_exact(groups) {
                    for (group, &code) in frame.iter().enumerate() {
                        let vocabulary = if group == 0 {
                            talker.vocab_size
                        } else {
                            predictor_vocab
                        };
                        if code >= vocabulary {
                            return Err("Qwen3-TTS reference codec token is out of range".into());
                        }
                    }
                }
                reference_codec.extend(self.build_reference_codec_embeddings(
                    input.reference_codes,
                    input.reference_frames,
                )?);
                build_icl_prompt(&IclPromptInput {
                    hidden_size: hidden,
                    role: &role,
                    codec: &codec,
                    combined_text: &combined_text,
                    reference_codec: &reference_codec,
                    tts_bos,
                    tts_eos,
                    tts_pad,
                })?
            };
        if output.tokens == 0 || output.tokens > self.maximum_tokens {
            return Err("Qwen3-TTS prompt exceeds runtime capacity".into());
        }
        Ok(output)
    }

    /// `PredictFrame`: expands the first code into all sixteen groups.
    /// `reference_codes` feeds the predictor a fixed history (parity checks);
    /// the returned codes stay the predictor's own choices.
    fn predict_frame(
        &mut self,
        talker_hidden: &[f32],
        first_code: u32,
        collect_logits: bool,
        reference_codes: &[u32],
        sampling: &mut PredictorSampling<'_>,
    ) -> Result<CodePredictorOutput, String> {
        let talker = self.config.talker.clone();
        let predictor = self.config.code_predictor.clone();
        let groups = talker.num_code_groups as usize;
        let hidden = talker.hidden_size as usize;
        if talker_hidden.len() != hidden {
            return Err("Qwen3-TTS talker hidden shape is invalid".into());
        }
        if first_code >= talker.vocab_size {
            return Err("Qwen3-TTS first codec token is out of range".into());
        }
        if !reference_codes.is_empty()
            && (reference_codes.len() != groups
                || reference_codes[0] != first_code
                || reference_codes[1..]
                    .iter()
                    .any(|&token| token >= predictor.vocab_size))
        {
            return Err("Qwen3-TTS predictor reference history is invalid".into());
        }
        let mut output = CodePredictorOutput::default();
        {
            let stream = &self.device.stream;
            let buffers = &mut self.buffers;
            // Every frame starts at position zero; its two prefix rows and
            // each later codec row overwrite K/V before attention sees them.
            buffers
                .host_upload
                .host_mut()
                .map_err(hip("hipMemcpyAsync predictor talker hidden"))?
                .copy_from_slice(talker_hidden);
            stream
                .upload_async(
                    &mut buffers.host_upload,
                    buffers.attention_output.slice_mut(0..hidden),
                )
                .map_err(hip("hipMemcpyAsync predictor talker hidden"))?;
            qk::embedding_lookup_token(
                stream,
                QuantSlice::new(
                    self.device.region.bytes(self.weights.codec_embedding),
                    GgmlType::Bf16,
                ),
                first_code,
                buffers.attention_output.slice_mut(hidden..2 * hidden),
                hidden,
                talker.vocab_size as usize,
            )
            .map_err(launch)?;
        }
        self.project_predictor_input(2)?;
        if collect_logits {
            let rows = 2 * predictor.hidden_size as usize;
            output.projected_input = vec![0.0f32; rows];
            self.device
                .stream
                .download(
                    self.buffers.hidden.slice(0..rows),
                    &mut output.projected_input,
                )
                .map_err(hip("hipMemcpyAsync predictor projected input"))?;
        }
        output.codes.reserve(groups);
        output.codes.push(first_code);
        let mut head_logits = Vec::new();
        let mut code = self.run_predictor(
            2,
            0,
            0,
            collect_logits.then_some(&mut head_logits),
            sampling,
        )?;
        if collect_logits {
            output.logits.extend_from_slice(&head_logits);
        }
        output.codes.push(code);
        for head in 1..self.weights.predictor_heads.len() {
            let token = if reference_codes.is_empty() {
                code
            } else {
                reference_codes[head]
            };
            qk::embedding_lookup_token(
                &self.device.stream,
                QuantSlice::new(
                    self.device
                        .region
                        .bytes(self.weights.predictor_embeddings[head - 1]),
                    GgmlType::Bf16,
                ),
                token,
                self.buffers.attention_output.slice_mut(0..hidden),
                hidden,
                predictor.vocab_size as usize,
            )
            .map_err(launch)?;
            self.project_predictor_input(1)?;
            code = self.run_predictor(
                1,
                head + 1,
                head,
                collect_logits.then_some(&mut head_logits),
                sampling,
            )?;
            if collect_logits {
                output.logits.extend_from_slice(&head_logits);
            }
            output.codes.push(code);
        }
        if output.codes.len() != groups
            || (collect_logits
                && output.logits.len() != (groups - 1) * predictor.vocab_size as usize)
        {
            return Err("Qwen3-TTS code predictor output shape mismatch".into());
        }
        Ok(output)
    }

    /// `PredictCodeFrame`: greedy expansion of one frame.
    pub fn predict_code_frame(
        &mut self,
        talker_hidden: &[f32],
        first_code: u32,
    ) -> Result<Vec<u32>, String> {
        let mut greedy = PredictorSampling {
            random: None,
            top_k: 50,
            temperature: 0.9,
            top_p: 1.0,
        };
        Ok(self
            .predict_frame(talker_hidden, first_code, false, &[], &mut greedy)?
            .codes)
    }

    /// `PredictCodeFrameTrace`: greedy expansion keeping every head's logits.
    pub fn predict_code_frame_trace(
        &mut self,
        talker_hidden: &[f32],
        first_code: u32,
        reference_codes: &[u32],
    ) -> Result<CodePredictorOutput, String> {
        let mut greedy = PredictorSampling {
            random: None,
            top_k: 50,
            temperature: 0.9,
            top_p: 1.0,
        };
        self.predict_frame(
            talker_hidden,
            first_code,
            true,
            reference_codes,
            &mut greedy,
        )
    }

    /// `DecodeCodeFrame`: feeds one complete frame back into the talker.
    pub fn decode_code_frame(
        &mut self,
        codes: &[u32],
        text_embedding: &[f32],
    ) -> Result<TalkerPrefillOutput, String> {
        let talker = &self.config.talker;
        if codes.len() != talker.num_code_groups as usize
            || text_embedding.len() != talker.hidden_size as usize
        {
            return Err("Qwen3-TTS codec frame shape is invalid".into());
        }
        for (group, &code) in codes.iter().enumerate() {
            let vocabulary = if group == 0 {
                talker.vocab_size
            } else {
                self.config.code_predictor.vocab_size
            };
            if code >= vocabulary {
                return Err("Qwen3-TTS codec token is out of range".into());
            }
        }
        self.prepare_code_frame_embedding(codes, text_embedding)?;
        self.run_talker_token()
    }

    /// `Generate`: prefill, then per frame the first code (sampled with the
    /// repetition penalty, EOS allowed from the third step), its fifteen
    /// predicted groups and one talker step. Cancellation is checked before
    /// prefill and every frame; `on_codes` sees the codes after each frame.
    pub fn generate(
        &mut self,
        prompt: &TalkerPromptOutput,
        maximum_new_tokens: usize,
        sampling: &SamplingOptions,
        is_cancelled: &dyn Fn() -> bool,
        on_codes: Option<&mut dyn FnMut(&TalkerGenerationOutput) -> bool>,
    ) -> Result<TalkerGenerationOutput, String> {
        let config = self.config.talker.clone();
        let hidden = config.hidden_size as usize;
        let groups = config.num_code_groups as usize;
        if prompt.tokens == 0
            || prompt.embeddings.len() != prompt.tokens * hidden
            || prompt.tts_pad.len() != hidden
            || prompt.trailing_text.is_empty()
            || !prompt.trailing_text.len().is_multiple_of(hidden)
            || maximum_new_tokens == 0
        {
            return Err("Qwen3-TTS generation prompt is invalid".into());
        }
        if !sampling.is_valid() {
            return Err("Qwen3-TTS sampling options are invalid".into());
        }
        // One engine serves both the talker and the predictor, as in C++.
        let mut random = Mt19937::new(sampling.seed);
        let cancelled = "Qwen3-TTS native generation cancelled";
        if is_cancelled() {
            return Err(cancelled.into());
        }
        let mut current = self.prefill(&prompt.embeddings, prompt.tokens)?;
        let mut output = TalkerGenerationOutput {
            codes: Vec::with_capacity(maximum_new_tokens * groups),
            frames: 0,
            code_groups: groups,
        };
        let mut flags = std::mem::take(&mut self.generated_first_code_flags);
        flags.clear();
        flags.resize(config.vocab_size as usize, 0);
        let trailing_rows = prompt.trailing_text.len() / hidden;
        let mut on_codes = on_codes;
        let result = (|| -> Result<(), String> {
            for step in 0..maximum_new_tokens {
                if is_cancelled() {
                    return Err(cancelled.into());
                }
                let first_code = select_main_code(
                    &current.logits,
                    &flags,
                    step,
                    config.codec_eos_token_id,
                    sampling,
                    sampling.sample.then_some(&mut random),
                    &mut self.candidate_scratch,
                )?;
                if first_code == config.codec_eos_token_id {
                    break;
                }
                flags[first_code as usize] = 1;
                let mut predictor = PredictorSampling {
                    random: sampling.predictor_sample.then_some(&mut random),
                    top_k: sampling.predictor_top_k,
                    temperature: sampling.predictor_temperature,
                    top_p: sampling.predictor_top_p,
                };
                let frame = self
                    .predict_frame(&current.last_hidden, first_code, false, &[], &mut predictor)?
                    .codes;
                output.codes.extend_from_slice(&frame);
                output.frames += 1;
                if let Some(callback) = on_codes.as_deref_mut()
                    && !callback(&output)
                {
                    return Err("Qwen3-TTS audio stream cancelled or failed".into());
                }
                if step + 1 == maximum_new_tokens {
                    break;
                }
                let text_embedding = if step < trailing_rows {
                    &prompt.trailing_text[step * hidden..(step + 1) * hidden]
                } else {
                    &prompt.tts_pad[..]
                };
                current = self.decode_code_frame(&frame, text_embedding)?;
            }
            Ok(())
        })();
        self.generated_first_code_flags = flags;
        result.map(|()| output)
    }
}

/// `SelectMainCode`: codec tokens below 2048 and EOS (from the third step),
/// with the repetition penalty on first codes generated before.
fn select_main_code(
    logits: &[f32],
    generated_first_codes: &[u8],
    generation_step: usize,
    eos_token: u32,
    sampling: &SamplingOptions,
    random: Option<&mut Mt19937>,
    candidates: &mut Vec<TokenScore>,
) -> Result<u32, String> {
    candidates.clear();
    candidates.reserve(2049);
    for (token, &logit) in logits.iter().enumerate() {
        let token = token as u32;
        if (token >= 2048 && token != eos_token) || (generation_step < 2 && token == eos_token) {
            continue;
        }
        let mut score = logit;
        if generated_first_codes
            .get(token as usize)
            .is_some_and(|&flag| flag != 0)
        {
            score = if score < 0.0 {
                score * sampling.repetition_penalty
            } else {
                score / sampling.repetition_penalty
            };
        }
        candidates.push(TokenScore { token, score });
    }
    sample_codec(
        candidates,
        sampling.top_k,
        sampling.top_p,
        sampling.temperature,
        random,
    )
}
