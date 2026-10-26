//! The GPU text encoder (`hip/encoder_runtime.cpp`). One instance owns the
//! weights and the scratch buffers and runs one device batch at a time.
//! [`EncoderRuntime::embed`] is thread-safe: concurrent calls take turns in
//! arrival order, one device batch per turn, so a long request does not hold
//! the device until it finishes.
//!
//! Activations stay f32 between operations. Each GEMM consumes BF16
//! activations and the checkpoint's BF16 weights and produces f32.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};

use gufo_embedding_gemma2_kernels as k;
use gufo_hip::{
    Bf16, DeviceBuffer, DeviceCopy, DeviceIndices, Launch, LaunchError, PinnedBuffer, Stream,
};

use crate::config::{EMBED_SCALE, LayerType, ModelConfig};
use crate::gemm::{self, Gemm};
use crate::loader::{LoadedModel, TEXT_TENSOR_PREFIX, TensorStore, load_model_directory};

/// The model's context window. Longer sequences are refused, not truncated.
pub const MAX_SEQUENCE_TOKENS: usize = 8192;

/// Rows per chunk when a GEMM and its consumer run back to back. Chunking
/// bounds the f32 scratch. Unchunked, the gate/up buffer for 8,192 tokens
/// would be 8,192 x 4,096 x 4 bytes = 134 MB and the QKV buffer about 100 MB.
const CHUNK_TOKENS: usize = 1024;
// The GEMM pads a chunk up to a row bucket, so the scratch behind every chunk
// must hold a full chunk.
const _: () = assert!(MAX_SEQUENCE_TOKENS.is_multiple_of(CHUNK_TOKENS));

pub const CANCELLED_MESSAGE: &str = "embedding_gemma2: request cancelled";
pub const DEVICE_LOST_MESSAGE: &str = "embedding_gemma2: GPU device lost";

/// Rotary cos/sin for every position up to the context limit, one pair of
/// `[position][head_dim / 2]` tables per head width and base.
struct RopeTable {
    head_dim: u32,
    theta: f32,
    cos: DeviceBuffer<f32>,
    sin: DeviceBuffer<f32>,
}

struct LayerWeights {
    input_norm: DeviceBuffer<Bf16>,
    post_attention_norm: DeviceBuffer<Bf16>,
    pre_feedforward_norm: DeviceBuffer<Bf16>,
    post_feedforward_norm: DeviceBuffer<Bf16>,
    layer_scalar: DeviceBuffer<Bf16>,
    /// q, k and v stacked row-wise.
    qkv: DeviceBuffer<Bf16>,
    output: DeviceBuffer<Bf16>,
    query_norm: DeviceBuffer<Bf16>,
    key_norm: DeviceBuffer<Bf16>,
    /// gate and up stacked row-wise.
    gate_up: DeviceBuffer<Bf16>,
    down: DeviceBuffer<Bf16>,
    ple_gate: DeviceBuffer<Bf16>,
    ple_projection: DeviceBuffer<Bf16>,
    ple_post_norm: DeviceBuffer<Bf16>,
}

/// Host staging and device copy of one per-batch index table.
struct Indices {
    host: PinnedBuffer<u32>,
    device: DeviceIndices,
}

/// Everything that runs on the device, used by one turn at a time.
struct Device {
    stream: Stream,
    gemm: Gemm,

    embed_table: DeviceBuffer<Bf16>,
    final_norm: DeviceBuffer<Bf16>,
    embedding_projection: DeviceBuffer<Bf16>,
    ple_projection: DeviceBuffer<Bf16>,
    ple_norm: DeviceBuffer<Bf16>,
    layers: Vec<LayerWeights>,
    rope: Vec<RopeTable>,

    // Per-batch indices; the uploads are asynchronous, so the pinned host
    // tables are locked until their copies complete.
    token_ids: Indices,
    positions: Indices,
    sequence_begin: Indices,
    sequence_end: Indices,
    offsets: Indices,

    // Scratch, sized for MAX_SEQUENCE_TOKENS.
    state: DeviceBuffer<f32>,
    embeddings_bf16: DeviceBuffer<Bf16>,
    normed_bf16: DeviceBuffer<Bf16>,
    qkv_f32: DeviceBuffer<f32>,
    query_bf16: DeviceBuffer<Bf16>,
    key_bf16: DeviceBuffer<Bf16>,
    value_bf16: DeviceBuffer<Bf16>,
    context_bf16: DeviceBuffer<Bf16>,
    temp_f32: DeviceBuffer<f32>,
    gate_up_f32: DeviceBuffer<f32>,
    act_bf16: DeviceBuffer<Bf16>,
    ple_f32: DeviceBuffer<f32>,
    gate_f32: DeviceBuffer<f32>,
    projected_f32: DeviceBuffer<f32>,
    pooled_f32: DeviceBuffer<f32>,
    /// Target of the device probe that runs after a failed batch.
    probe: DeviceBuffer<u32>,
}

/// Device turns in arrival order, one batch each. A plain mutex lets the
/// thread that just released it take it straight back, so a long request
/// would still starve the requests queued behind it.
#[derive(Default)]
struct Turns {
    state: Mutex<(u64, u64)>,
    changed: Condvar,
}

struct Turn<'a> {
    turns: &'a Turns,
}

impl Turns {
    fn lock(&self) -> MutexGuard<'_, (u64, u64)> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Waits for this caller's turn: tickets are served in the order taken.
    fn take(&self) -> Turn<'_> {
        let mut state = self.lock();
        let ticket = state.0;
        state.0 += 1;
        while state.1 != ticket {
            state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        Turn { turns: self }
    }
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        self.turns.lock().1 += 1;
        self.turns.changed.notify_all();
    }
}

pub struct EncoderRuntime {
    config: ModelConfig,
    device: Mutex<Device>,
    lost: AtomicBool,
    turns: Turns,
}

fn hip(error: impl std::fmt::Display) -> String {
    error.to_string()
}

/// One BF16 tensor, uploaded synchronously into its own exact allocation.
fn upload(stream: &Stream, store: &TensorStore, name: &str) -> Result<DeviceBuffer<Bf16>, String> {
    upload_concatenated(stream, store, &[name.to_owned()])
}

/// BF16 tensors that share a column count, concatenated row-wise into one
/// device allocation.
fn upload_concatenated(
    stream: &Stream,
    store: &TensorStore,
    names: &[String],
) -> Result<DeviceBuffer<Bf16>, String> {
    let tensors = names
        .iter()
        .map(|name| store.find(name).ok_or_else(|| format!("missing {name}")))
        .collect::<Result<Vec<_>, _>>()?;
    let total: usize = tensors.iter().map(|t| t.byte_count / 2).sum();
    let mut buffer = stream.alloc_persistent::<Bf16>(total).map_err(hip)?;
    let mut offset = 0;
    for tensor in tensors {
        let elements = tensor.byte_count / 2;
        stream
            .upload_bytes(
                store.bytes(tensor),
                buffer.slice_mut(offset..offset + elements),
            )
            .map_err(hip)?;
        offset += elements;
    }
    Ok(buffer)
}

/// The tables computed in double precision, so the f32 values are correctly
/// rounded.
fn build_rope_table(
    stream: &Stream,
    head_dim: u32,
    theta: f32,
    positions: usize,
) -> Result<RopeTable, String> {
    let half = (head_dim / 2) as usize;
    let mut cosines = vec![0f32; positions * half];
    let mut sines = vec![0f32; positions * half];
    for i in 0..half {
        let frequency = f64::from(theta).powf(-((2 * i) as f64) / f64::from(head_dim));
        for p in 0..positions {
            let angle = p as f64 * frequency;
            cosines[p * half + i] = angle.cos() as f32;
            sines[p * half + i] = angle.sin() as f32;
        }
    }
    let mut cos = stream.alloc_persistent::<f32>(cosines.len()).map_err(hip)?;
    let mut sin = stream.alloc_persistent::<f32>(sines.len()).map_err(hip)?;
    stream.upload(&cosines, cos.as_mut_slice()).map_err(hip)?;
    stream.upload(&sines, sin.as_mut_slice()).map_err(hip)?;
    Ok(RopeTable {
        head_dim,
        theta,
        cos,
        sin,
    })
}

fn indices(stream: &Stream, len: usize, bound: usize) -> Result<Indices, String> {
    let bound = u32::try_from(bound).map_err(|_| "index bound exceeds 32 bits".to_owned())?;
    Ok(Indices {
        host: PinnedBuffer::zeroed(len).map_err(hip)?,
        device: stream
            .zeroed_indices(stream.alloc_persistent::<u32>(len).map_err(hip)?, bound)
            .map_err(hip)?,
    })
}

fn scratch<T: DeviceCopy>(stream: &Stream, len: usize) -> Result<DeviceBuffer<T>, String> {
    stream.alloc_persistent::<T>(len).map_err(hip)
}

impl Device {
    fn new(loaded: &LoadedModel) -> Result<Self, String> {
        let stream = Stream::new().map_err(hip)?;
        let gemm = Gemm::new(&stream)?;
        let store = &loaded.store;
        let text = &loaded.config.text;
        let prefix = TEXT_TENSOR_PREFIX;
        let tensor = |name: &str| upload(&stream, store, &format!("{prefix}{name}"));
        let embed_table = tensor("embed_tokens.weight")?;
        let final_norm = tensor("norm.weight")?;
        let embedding_projection = tensor("embedding_projection.weight")?;
        let ple_projection = tensor("ple.per_layer_model_projection.weight")?;
        let ple_norm = tensor("ple.per_layer_projection_norm.weight")?;

        let mut rope: Vec<RopeTable> = Vec::new();
        for geometry in &text.layers {
            if !rope
                .iter()
                .any(|t| t.head_dim == geometry.head_dim && t.theta == geometry.rope_theta)
            {
                rope.push(build_rope_table(
                    &stream,
                    geometry.head_dim,
                    geometry.rope_theta,
                    MAX_SEQUENCE_TOKENS,
                )?);
            }
        }
        let mut layers = Vec::with_capacity(text.num_hidden_layers as usize);
        for index in 0..text.num_hidden_layers as usize {
            let p = format!("{prefix}layers.{index}.");
            let one = |name: &str| upload(&stream, store, &format!("{p}{name}"));
            let stacked = |names: &[&str]| {
                let names: Vec<String> = names.iter().map(|n| format!("{p}{n}")).collect();
                upload_concatenated(&stream, store, &names)
            };
            layers.push(LayerWeights {
                input_norm: one("input_layernorm.weight")?,
                post_attention_norm: one("post_attention_layernorm.weight")?,
                pre_feedforward_norm: one("pre_feedforward_layernorm.weight")?,
                post_feedforward_norm: one("post_feedforward_layernorm.weight")?,
                layer_scalar: one("layer_scalar")?,
                qkv: stacked(&[
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ])?,
                output: one("self_attn.o_proj.weight")?,
                query_norm: one("self_attn.q_norm.weight")?,
                key_norm: one("self_attn.k_norm.weight")?,
                gate_up: stacked(&["mlp.gate_proj.weight", "mlp.up_proj.weight"])?,
                down: one("mlp.down_proj.weight")?,
                ple_gate: one("ple_block.per_layer_input_gate.weight")?,
                ple_projection: one("ple_block.per_layer_projection.weight")?,
                ple_post_norm: one("ple_block.post_per_layer_input_norm.weight")?,
            });
        }

        let mut qkv_width = 0;
        let mut query_width = 0;
        let mut key_width = 0;
        for layer in &text.layers {
            query_width =
                query_width.max(text.num_attention_heads as usize * layer.head_dim as usize);
            key_width = key_width.max(layer.num_key_value_heads as usize * layer.head_dim as usize);
            qkv_width = qkv_width.max(query_width + 2 * key_width);
        }
        let n = MAX_SEQUENCE_TOKENS;
        let hidden = text.hidden_size as usize;
        let per_layer = text.hidden_size_per_layer_input as usize;
        let intermediate = text.intermediate_size as usize;
        let embedding_dim = text.embedding_dim as usize;
        // Only one chunk of rows is live at a time.
        let chunk = n.min(CHUNK_TOKENS);
        Ok(Self {
            token_ids: indices(&stream, n, text.vocab_size as usize)?,
            positions: indices(&stream, n, n)?,
            sequence_begin: indices(&stream, n, n)?,
            sequence_end: indices(&stream, n, n + 1)?,
            offsets: indices(&stream, n + 1, n + 1)?,
            state: scratch(&stream, n * hidden)?,
            embeddings_bf16: scratch(&stream, n * hidden)?,
            normed_bf16: scratch(&stream, n * hidden)?,
            qkv_f32: scratch(&stream, chunk * qkv_width)?,
            query_bf16: scratch(&stream, n * query_width)?,
            key_bf16: scratch(&stream, n * key_width)?,
            value_bf16: scratch(&stream, n * key_width)?,
            context_bf16: scratch(&stream, n * query_width)?,
            temp_f32: scratch(&stream, n * hidden)?,
            gate_up_f32: scratch(&stream, chunk * 2 * intermediate)?,
            act_bf16: scratch(&stream, n * intermediate)?,
            ple_f32: scratch(&stream, n * per_layer)?,
            gate_f32: scratch(&stream, n * per_layer)?,
            projected_f32: scratch(&stream, n * embedding_dim)?,
            pooled_f32: scratch(&stream, n * embedding_dim)?,
            probe: scratch(&stream, 1)?,
            embed_table,
            final_norm,
            embedding_projection,
            ple_projection,
            ple_norm,
            layers,
            rope,
            gemm,
            stream,
        })
    }

    /// Mirrors the text server's probe: after a reset even a 4-byte memset
    /// fails, while an ordinary kernel or library failure leaves it working.
    /// A sticky error has already poisoned the stream, which then refuses the
    /// memset as well.
    fn usable(&mut self) -> bool {
        self.stream
            .fill_bytes(self.probe.as_mut_slice(), 0)
            .and_then(|()| self.stream.synchronize())
            .is_ok()
    }

    /// Runs one sequence for every GEMM row bucket, so hipBLASLt has tuned
    /// all of them and every kernel is loaded before the first request.
    fn warmup(&mut self, config: &ModelConfig) -> Result<(), String> {
        let mut rows = 0;
        while rows < CHUNK_TOKENS {
            rows = gemm::bucket(rows + 1, CHUNK_TOKENS);
            self.run_batch(config, &[vec![1000; rows]])?;
        }
        Ok(())
    }

    /// Packs `sequences` (at most MAX_SEQUENCE_TOKENS tokens together) into
    /// one batch and returns their embeddings.
    fn run_batch(
        &mut self,
        config: &ModelConfig,
        sequences: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>, String> {
        let text = &config.text;
        let hidden = text.hidden_size as usize;
        let per_layer = text.hidden_size_per_layer_input as usize;
        let intermediate = text.intermediate_size as usize;
        let embedding_dim = text.embedding_dim as usize;
        let epsilon = text.rms_norm_eps;
        let heads = text.num_attention_heads;
        let tokens: usize = sequences.iter().map(Vec::len).sum();
        let count = sequences.len();
        let n = MAX_SEQUENCE_TOKENS;
        let stream = &self.stream;

        // Pack the sequences and describe each token's sequence bounds.
        {
            let ids = self.token_ids.host.host_mut().map_err(hip)?;
            let positions = self.positions.host.host_mut().map_err(hip)?;
            let begins = self.sequence_begin.host.host_mut().map_err(hip)?;
            let ends = self.sequence_end.host.host_mut().map_err(hip)?;
            let offsets = self.offsets.host.host_mut().map_err(hip)?;
            let mut at = 0usize;
            offsets[0] = 0;
            for (s, sequence) in sequences.iter().enumerate() {
                let begin = at as u32;
                let end = (at + sequence.len()) as u32;
                for (t, &id) in sequence.iter().enumerate() {
                    ids[at] = id;
                    positions[at] = t as u32;
                    begins[at] = begin;
                    ends[at] = end;
                    at += 1;
                }
                offsets[s + 1] = end;
            }
        }
        for (table, len) in [
            (&mut self.token_ids, tokens),
            (&mut self.positions, tokens),
            (&mut self.sequence_begin, tokens),
            (&mut self.sequence_end, tokens),
            (&mut self.offsets, count + 1),
        ] {
            stream
                .upload_indices_async(&mut table.host, len, &mut table.device)
                .map_err(hip)?;
        }

        let launch = |error: LaunchError| error.to_string();
        k::embedding_lookup(
            stream,
            self.embed_table.as_slice(),
            self.token_ids.device.view(),
            self.state.as_mut_slice(),
            self.embeddings_bf16.as_mut_slice(),
            tokens,
            hidden,
            EMBED_SCALE,
            text.vocab_size as usize,
        )
        .map_err(launch)?;

        let ple_scale = 1.0f32 / (hidden as f32).sqrt();
        let chunks = || (0..tokens).step_by(CHUNK_TOKENS);
        for (index, (layer, geometry)) in self.layers.iter().zip(&text.layers).enumerate() {
            let kv_heads = geometry.num_key_value_heads;
            let head_dim = geometry.head_dim;
            let query_width = heads as usize * head_dim as usize;
            let key_width = kv_heads as usize * head_dim as usize;
            let full = geometry.layer_type == LayerType::FullAttention;
            let rope = self
                .rope
                .iter()
                .find(|t| t.head_dim == head_dim && t.theta == geometry.rope_theta)
                .ok_or("EmbeddingGemma 2 rotary table")?;

            k::rms_norm_to_bf16(
                stream,
                self.state.as_slice(),
                layer.input_norm.as_slice(),
                self.normed_bf16.as_mut_slice(),
                tokens,
                hidden,
                epsilon,
            )
            .map_err(launch)?;
            for row in chunks() {
                let rows = CHUNK_TOKENS.min(tokens - row);
                self.gemm
                    .run(
                        stream,
                        layer.qkv.as_slice(),
                        self.normed_bf16.slice(row * hidden..n * hidden),
                        self.qkv_f32.as_mut_slice(),
                        rows,
                        query_width + 2 * key_width,
                        hidden,
                        CHUNK_TOKENS,
                    )
                    .map_err(launch)?;
                k::qkv_norm_rope(
                    stream,
                    self.qkv_f32.as_slice(),
                    layer.query_norm.as_slice(),
                    layer.key_norm.as_slice(),
                    self.query_bf16
                        .slice_mut(row * query_width..n * query_width),
                    self.key_bf16.slice_mut(row * key_width..n * key_width),
                    self.value_bf16.slice_mut(row * key_width..n * key_width),
                    self.positions.device.slice(row..n),
                    rows,
                    heads,
                    kv_heads,
                    head_dim,
                    rope.cos.as_slice(),
                    rope.sin.as_slice(),
                    epsilon,
                    n,
                )
                .map_err(launch)?;
            }
            let key_rows = self.key_bf16.len() / key_width;
            let attended = k::attention(
                stream,
                self.query_bf16.as_slice(),
                self.key_bf16.as_slice(),
                self.value_bf16.as_slice(),
                self.context_bf16.as_mut_slice(),
                self.sequence_begin.device.view(),
                self.sequence_end.device.view(),
                tokens,
                heads,
                kv_heads,
                head_dim,
                if full { 0 } else { text.sliding_window },
                key_rows,
            )
            .map_err(launch)?;
            if attended != Launch::Launched {
                return Err("EmbeddingGemma 2 attention geometry".into());
            }
            for row in chunks() {
                let rows = CHUNK_TOKENS.min(tokens - row);
                self.gemm
                    .run(
                        stream,
                        layer.output.as_slice(),
                        self.context_bf16.slice(row * query_width..n * query_width),
                        self.temp_f32.as_mut_slice(),
                        rows,
                        hidden,
                        query_width,
                        CHUNK_TOKENS,
                    )
                    .map_err(launch)?;
                k::rms_norm_residual(
                    stream,
                    self.temp_f32.as_slice(),
                    layer.post_attention_norm.as_slice(),
                    self.state.slice_mut(row * hidden..(row + rows) * hidden),
                    rows,
                    hidden,
                    epsilon,
                )
                .map_err(launch)?;
            }

            k::rms_norm_to_bf16(
                stream,
                self.state.as_slice(),
                layer.pre_feedforward_norm.as_slice(),
                self.normed_bf16.as_mut_slice(),
                tokens,
                hidden,
                epsilon,
            )
            .map_err(launch)?;
            for row in chunks() {
                let rows = CHUNK_TOKENS.min(tokens - row);
                self.gemm
                    .run(
                        stream,
                        layer.gate_up.as_slice(),
                        self.normed_bf16.slice(row * hidden..n * hidden),
                        self.gate_up_f32.as_mut_slice(),
                        rows,
                        2 * intermediate,
                        hidden,
                        CHUNK_TOKENS,
                    )
                    .map_err(launch)?;
                k::geglu(
                    stream,
                    self.gate_up_f32.as_slice(),
                    self.act_bf16
                        .slice_mut(row * intermediate..n * intermediate),
                    rows,
                    intermediate,
                )
                .map_err(launch)?;
            }
            for row in chunks() {
                let rows = CHUNK_TOKENS.min(tokens - row);
                self.gemm
                    .run(
                        stream,
                        layer.down.as_slice(),
                        self.act_bf16.slice(row * intermediate..n * intermediate),
                        self.temp_f32.as_mut_slice(),
                        rows,
                        hidden,
                        intermediate,
                        CHUNK_TOKENS,
                    )
                    .map_err(launch)?;
                k::rms_norm_residual(
                    stream,
                    self.temp_f32.as_slice(),
                    layer.post_feedforward_norm.as_slice(),
                    self.state.slice_mut(row * hidden..(row + rows) * hidden),
                    rows,
                    hidden,
                    epsilon,
                )
                .map_err(launch)?;
            }

            // The per-layer input block takes this layer's slice of the
            // per-layer inputs, then runs a gated bottleneck that mixes it
            // into the residual stream.
            let projection = self
                .ple_projection
                .slice(index * per_layer * hidden..(index + 1) * per_layer * hidden);
            for row in chunks() {
                let rows = CHUNK_TOKENS.min(tokens - row);
                let state_rows = row * hidden..(row + rows) * hidden;
                self.gemm
                    .run(
                        stream,
                        projection,
                        self.embeddings_bf16.slice(row * hidden..n * hidden),
                        self.ple_f32.as_mut_slice(),
                        rows,
                        per_layer,
                        hidden,
                        CHUNK_TOKENS,
                    )
                    .map_err(launch)?;
                k::scaled_rms_norm(
                    stream,
                    self.ple_f32.slice_mut(0..rows * per_layer),
                    self.ple_norm.as_slice(),
                    rows,
                    per_layer,
                    ple_scale,
                    epsilon,
                )
                .map_err(launch)?;
                k::float_to_bf16(
                    stream,
                    self.state.slice(state_rows.clone()),
                    self.normed_bf16.slice_mut(state_rows.clone()),
                    rows * hidden,
                )
                .map_err(launch)?;
                self.gemm
                    .run(
                        stream,
                        layer.ple_gate.as_slice(),
                        self.normed_bf16.slice(row * hidden..n * hidden),
                        self.gate_f32.as_mut_slice(),
                        rows,
                        per_layer,
                        hidden,
                        CHUNK_TOKENS,
                    )
                    .map_err(launch)?;
                k::ple_gate(
                    stream,
                    self.gate_f32.as_slice(),
                    self.ple_f32.as_slice(),
                    self.act_bf16.as_mut_slice(),
                    rows * per_layer,
                )
                .map_err(launch)?;
                self.gemm
                    .run(
                        stream,
                        layer.ple_projection.as_slice(),
                        self.act_bf16.as_slice(),
                        self.temp_f32.as_mut_slice(),
                        rows,
                        hidden,
                        per_layer,
                        CHUNK_TOKENS,
                    )
                    .map_err(launch)?;
                k::ple_residual(
                    stream,
                    self.temp_f32.as_slice(),
                    layer.ple_post_norm.as_slice(),
                    layer.layer_scalar.as_slice(),
                    self.state.slice_mut(state_rows),
                    rows,
                    hidden,
                    epsilon,
                )
                .map_err(launch)?;
            }
        }

        k::rms_norm_to_bf16(
            stream,
            self.state.as_slice(),
            self.final_norm.as_slice(),
            self.normed_bf16.as_mut_slice(),
            tokens,
            hidden,
            epsilon,
        )
        .map_err(launch)?;
        for row in chunks() {
            let rows = CHUNK_TOKENS.min(tokens - row);
            self.gemm
                .run(
                    stream,
                    self.embedding_projection.as_slice(),
                    self.normed_bf16.slice(row * hidden..n * hidden),
                    self.projected_f32
                        .slice_mut(row * embedding_dim..n * embedding_dim),
                    rows,
                    embedding_dim,
                    hidden,
                    CHUNK_TOKENS,
                )
                .map_err(launch)?;
        }
        let pooled = k::mean_pool_normalize(
            stream,
            self.projected_f32.as_slice(),
            self.offsets.device.slice(0..count + 1),
            self.pooled_f32.as_mut_slice(),
            count,
            embedding_dim,
            n,
        )
        .map_err(launch)?;
        if pooled != Launch::Launched {
            return Err("EmbeddingGemma 2 pooling width".into());
        }
        let mut host = vec![0f32; count * embedding_dim];
        stream
            .download(self.pooled_f32.slice(0..host.len()), &mut host)
            .map_err(hip)?;
        Ok(host
            .chunks_exact(embedding_dim)
            .map(<[f32]>::to_vec)
            .collect())
    }
}

impl EncoderRuntime {
    /// Loads the checkpoint, uploads the text backbone and tunes every GEMM
    /// row bucket. The checkpoint is unmapped before the warm-up: the device
    /// then holds every weight, and unmapping releases the page-cache
    /// residency, about 650 MB of host memory.
    pub fn create(model_dir: &Path) -> Result<Self, String> {
        let loaded = load_model_directory(model_dir)?;
        let fail = |error: String| format!("embedding_gemma2: {error}");
        let mut device = Device::new(&loaded).map_err(fail)?;
        let LoadedModel { config, store } = loaded;
        drop(store);
        if let Err(error) = device.warmup(&config) {
            let _ = device.stream.synchronize();
            return Err(fail(error));
        }
        Ok(Self {
            config,
            device: Mutex::new(device),
            lost: AtomicBool::new(false),
            turns: Turns::default(),
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// True after a device failure that a probe confirmed.
    pub fn device_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    /// One mean-pooled, L2-normalised embedding of `config().text.
    /// embedding_dim` floats for each sequence. Sequences are packed into as
    /// few device batches of MAX_SEQUENCE_TOKENS as fit. The call polls
    /// `is_cancelled` at the start of each device turn; a cancelled call
    /// fails with [`CANCELLED_MESSAGE`]. After a device failure that a probe
    /// confirms, [`Self::device_lost`] turns true and every later call fails
    /// with [`DEVICE_LOST_MESSAGE`].
    pub fn embed(
        &self,
        sequences: &[Vec<u32>],
        is_cancelled: Option<&dyn Fn() -> bool>,
    ) -> Result<Vec<Vec<f32>>, String> {
        let text = &self.config.text;
        for sequence in sequences {
            if sequence.is_empty() {
                return Err("embedding_gemma2: empty token sequence".into());
            }
            if sequence.len() > MAX_SEQUENCE_TOKENS {
                return Err(format!(
                    "embedding_gemma2: sequence exceeds {MAX_SEQUENCE_TOKENS} tokens"
                ));
            }
            if sequence.iter().any(|&id| id >= text.vocab_size) {
                return Err("embedding_gemma2: token id out of range".into());
            }
        }
        let mut embeddings = Vec::with_capacity(sequences.len());
        let mut first = 0;
        while first < sequences.len() {
            let mut last = first;
            let mut tokens = 0;
            while last < sequences.len() && tokens + sequences[last].len() <= MAX_SEQUENCE_TOKENS {
                tokens += sequences[last].len();
                last += 1;
            }
            // Each batch takes a new turn, so requests that arrived meanwhile
            // run before the next batch of this one.
            let _turn = self.turns.take();
            if self.device_lost() {
                return Err(DEVICE_LOST_MESSAGE.into());
            }
            if is_cancelled.is_some_and(|cancelled| cancelled()) {
                return Err(CANCELLED_MESSAGE.into());
            }
            let mut device = self.device.lock().unwrap_or_else(|e| e.into_inner());
            match device.run_batch(&self.config, &sequences[first..last]) {
                Ok(batch) => embeddings.extend(batch),
                Err(failure) => {
                    // Kernels from the failed batch may still be queued. They
                    // read the index buffers the next batch overwrites, so
                    // drain the stream before the turn passes on.
                    let _ = device.stream.synchronize();
                    if !device.usable() {
                        self.lost.store(true, Ordering::Release);
                        return Err(format!("{DEVICE_LOST_MESSAGE}: {failure}"));
                    }
                    return Err(format!("embedding_gemma2: {failure}"));
                }
            }
            first = last;
        }
        Ok(embeddings)
    }
}
