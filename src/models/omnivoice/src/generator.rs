//! OmniVoice's backbone over the CFG batch (`generator.cpp`): the
//! bidirectional Qwen3 layers compute what Hugging Face transformers
//! computes, with float16 GEMM operands and a float32 residual stream, and
//! the iterative unmasking with classifier-free guidance. Rows are token
//! positions of the packed batch: the conditional sequence's rows first,
//! then its target frames again as the unconditional sequence.

use std::path::Path;

use gufo_hip::{DeviceBuffer, DeviceCopy, DeviceIndices, F16, HipError, LaunchError, Stream};
use gufo_omnivoice_kernels as k;

use crate::config::{BackboneConfig, Config};
use crate::decoding::{
    ConditionalInput, DecodingConfig, GenerationResult, PositionSelector, unmask_schedule,
};
use crate::gemm::{HalfGemm, MatmulShape, tuning_bucket};
use crate::safetensors::{DType, SafetensorsFile, Tensor};

const TEXT_EMBEDDINGS: &str = "llm.embed_tokens.weight";

struct Layer {
    /// `[(heads + 2 kv_heads) * head_dim, hidden]`
    qkv: DeviceBuffer<F16>,
    /// `[hidden, heads * head_dim]`
    output: DeviceBuffer<F16>,
    /// `[2 * intermediate, hidden]`
    gate_up: DeviceBuffer<F16>,
    /// `[hidden, intermediate]`
    down: DeviceBuffer<F16>,
    input_norm: DeviceBuffer<f32>,
    post_norm: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
}

/// Activations of the packed CFG batch, grown on demand.
struct Workspace {
    /// `[text rows, hidden]`
    text: DeviceBuffer<f32>,
    /// `[codebooks, rows]`, below the audio vocabulary; text rows hold 0
    /// (the embedding kernel reads their text rows instead).
    ids: DeviceIndices,
    positions: DeviceBuffer<i32>,
    /// Rows of the target frames in both sequences, below `rows`.
    target_rows: DeviceIndices,
    /// `[rows, hidden]`
    hidden: DeviceBuffer<f32>,
    /// `[rows, max(hidden, intermediate)]`
    normed: DeviceBuffer<F16>,
    /// `[rows, qkv width]`
    qkv: DeviceBuffer<f32>,
    /// `[heads, rows, head_dim]`
    q: DeviceBuffer<F16>,
    k: DeviceBuffer<F16>,
    v: DeviceBuffer<F16>,
    /// `[heads, longest, longest]`
    scores: DeviceBuffer<f32>,
    probabilities: DeviceBuffer<F16>,
    /// `[heads, rows, head_dim]`
    attention: DeviceBuffer<f32>,
    /// `[rows, heads * head_dim]`
    merged: DeviceBuffer<F16>,
    /// `[rows, 2 * intermediate]`
    gate_up: DeviceBuffer<f32>,
    /// `[rows, intermediate]`
    activated: DeviceBuffer<F16>,
    /// `[2 * frames, hidden]`
    final_rows: DeviceBuffer<f32>,
    final_normed: DeviceBuffer<F16>,
    /// `[2 * frames, codebooks * vocab]`
    logits: DeviceBuffer<f32>,
    prediction: DeviceBuffer<i32>,
    confidence: DeviceBuffer<f32>,
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

impl Workspace {
    fn new(stream: &Stream, vocab: u32) -> Result<Self, HipError> {
        Ok(Self {
            text: stream.alloc(1)?,
            ids: stream.alloc_indices(1, vocab)?,
            positions: stream.alloc(1)?,
            target_rows: stream.alloc_indices(1, 1)?,
            hidden: stream.alloc(1)?,
            normed: stream.alloc(1)?,
            qkv: stream.alloc(1)?,
            q: stream.alloc(1)?,
            k: stream.alloc(1)?,
            v: stream.alloc(1)?,
            scores: stream.alloc(1)?,
            probabilities: stream.alloc(1)?,
            attention: stream.alloc(1)?,
            merged: stream.alloc(1)?,
            gate_up: stream.alloc(1)?,
            activated: stream.alloc(1)?,
            final_rows: stream.alloc(1)?,
            final_normed: stream.alloc(1)?,
            logits: stream.alloc(1)?,
            prediction: stream.alloc(1)?,
            confidence: stream.alloc(1)?,
        })
    }
}

/// The bidirectional Qwen3 backbone with its audio embeddings and heads on
/// the GPU, and OmniVoice's iterative unmasking with classifier-free
/// guidance. Not thread-safe.
pub struct Generator {
    config: Config,
    weights: SafetensorsFile,
    /// `[codebooks * vocab, hidden]`
    audio_embeddings: DeviceBuffer<f32>,
    /// `[codebooks * vocab, hidden]`
    audio_heads: DeviceBuffer<F16>,
    final_norm: DeviceBuffer<f32>,
    layers: Vec<Layer>,
    work: Workspace,
    host_ids: Vec<u32>,
    gemm: HalfGemm,
    stream: Stream,
}

/// Converts float32 tensor bytes to float16 into `output`, in parallel.
fn convert_to_half(bytes: &[u8], output: &mut [F16]) {
    let count = output.len();
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .clamp(1, 16);
    let chunk = count.div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        for (index, part) in output.chunks_mut(chunk).enumerate() {
            let source = &bytes[index * chunk * 4..];
            scope.spawn(move || {
                for (value, raw) in part.iter_mut().zip(source.as_chunks::<4>().0) {
                    *value = F16::from_f32(f32::from_le_bytes(*raw));
                }
            });
        }
    });
}

/// Uploads a float32 tensor of this exact shape.
fn upload_float(
    stream: &Stream,
    file: &SafetensorsFile,
    name: &str,
    shape: &[usize],
) -> Result<DeviceBuffer<f32>, String> {
    let tensor = file.expect(name, DType::F32, shape)?;
    let mut buffer = stream.alloc(tensor.elements()).map_err(|e| e.to_string())?;
    stream
        .upload_bytes(file.bytes(tensor), buffer.as_mut_slice())
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

/// Converts float32 tensors with `columns` columns to float16 and uploads
/// them stacked by rows.
fn upload_half(
    stream: &Stream,
    file: &SafetensorsFile,
    parts: &[(String, usize)],
    columns: usize,
) -> Result<DeviceBuffer<F16>, String> {
    let mut tensors: Vec<&Tensor> = Vec::with_capacity(parts.len());
    let mut rows = 0;
    for (name, part_rows) in parts {
        tensors.push(file.expect(name, DType::F32, &[*part_rows, columns])?);
        rows += part_rows;
    }
    let mut host = vec![F16(0); rows * columns];
    let mut offset = 0;
    for tensor in tensors {
        let elements = tensor.elements();
        convert_to_half(file.bytes(tensor), &mut host[offset..offset + elements]);
        offset += elements;
    }
    let mut buffer = stream.alloc(host.len()).map_err(|e| e.to_string())?;
    stream
        .upload(&host, buffer.as_mut_slice())
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

fn launch_error(error: LaunchError) -> String {
    error.to_string()
}

fn hip_error(error: HipError) -> String {
    error.to_string()
}

impl Generator {
    /// Loads the backbone, the audio embeddings and heads of
    /// `model_root/model.safetensors`.
    pub fn load(config: &Config, model_root: &Path) -> Result<Self, String> {
        Self::load_inner(config, model_root).map_err(|e| format!("OmniVoice generator: {e}"))
    }

    fn load_inner(config: &Config, model_root: &Path) -> Result<Self, String> {
        let weights = SafetensorsFile::open(&model_root.join("model.safetensors"))?;
        let file = &weights;
        let b = &config.backbone;
        let audio_rows = config.audio_codebooks * config.audio_vocab_size;
        file.expect(TEXT_EMBEDDINGS, DType::F32, &[b.vocab_size, b.hidden_size])?;
        let stream = Stream::new().map_err(hip_error)?;
        // Reproducible sums: the same seed gives the same audio (Blas::new
        // disallows hipBLAS atomics).
        let gemm = HalfGemm::new(&stream).map_err(launch_error)?;
        let audio_embeddings = upload_float(
            &stream,
            file,
            "audio_embeddings.weight",
            &[audio_rows, b.hidden_size],
        )?;
        let audio_heads = upload_half(
            &stream,
            file,
            &[("audio_heads.weight".to_string(), audio_rows)],
            b.hidden_size,
        )?;
        let final_norm = upload_float(&stream, file, "llm.norm.weight", &[b.hidden_size])?;
        let q_rows = b.heads * b.head_dim;
        let kv_rows = b.kv_heads * b.head_dim;
        let mut layers = Vec::with_capacity(b.layers);
        for index in 0..b.layers {
            let prefix = format!("llm.layers.{index}.");
            let name = |suffix: &str| format!("{prefix}{suffix}");
            layers.push(Layer {
                qkv: upload_half(
                    &stream,
                    file,
                    &[
                        (name("self_attn.q_proj.weight"), q_rows),
                        (name("self_attn.k_proj.weight"), kv_rows),
                        (name("self_attn.v_proj.weight"), kv_rows),
                    ],
                    b.hidden_size,
                )?,
                output: upload_half(
                    &stream,
                    file,
                    &[(name("self_attn.o_proj.weight"), b.hidden_size)],
                    q_rows,
                )?,
                gate_up: upload_half(
                    &stream,
                    file,
                    &[
                        (name("mlp.gate_proj.weight"), b.intermediate_size),
                        (name("mlp.up_proj.weight"), b.intermediate_size),
                    ],
                    b.hidden_size,
                )?,
                down: upload_half(
                    &stream,
                    file,
                    &[(name("mlp.down_proj.weight"), b.hidden_size)],
                    b.intermediate_size,
                )?,
                input_norm: upload_float(
                    &stream,
                    file,
                    &name("input_layernorm.weight"),
                    &[b.hidden_size],
                )?,
                post_norm: upload_float(
                    &stream,
                    file,
                    &name("post_attention_layernorm.weight"),
                    &[b.hidden_size],
                )?,
                q_norm: upload_float(
                    &stream,
                    file,
                    &name("self_attn.q_norm.weight"),
                    &[b.head_dim],
                )?,
                k_norm: upload_float(
                    &stream,
                    file,
                    &name("self_attn.k_norm.weight"),
                    &[b.head_dim],
                )?,
            });
        }
        // Only the text embeddings stay on the host; their rows fault back in
        // per request.
        weights.release_pages();
        let vocab = u32::try_from(config.audio_vocab_size)
            .map_err(|_| "audio vocabulary too large".to_string())?;
        let work = Workspace::new(&stream, vocab).map_err(hip_error)?;
        Ok(Self {
            config: config.clone(),
            weights,
            audio_embeddings,
            audio_heads,
            final_norm,
            layers,
            work,
            host_ids: Vec::new(),
            gemm,
            stream,
        })
    }

    fn backbone(&self) -> &BackboneConfig {
        &self.config.backbone
    }

    fn qkv_width(&self) -> usize {
        let b = self.backbone();
        (b.heads + 2 * b.kv_heads) * b.head_dim
    }

    /// Uploads the packed CFG batch for `input` and returns its row count.
    fn prepare(&mut self, input: &ConditionalInput) -> Result<usize, String> {
        let b = self.config.backbone.clone();
        let length = input.length;
        let frames = input.target_frames;
        let rows = length + frames;
        let codebooks = input.codebooks;
        let heads = b.heads;
        let qkv_width = self.qkv_width();
        let s = &self.stream;
        let w = &mut self.work;
        (|| -> Result<(), HipError> {
            reserve(s, &mut w.text, input.audio_start.max(1) * b.hidden_size)?;
            reserve(s, &mut w.positions, rows)?;
            reserve(s, &mut w.hidden, rows * b.hidden_size)?;
            reserve(
                s,
                &mut w.normed,
                rows * b.hidden_size.max(b.intermediate_size),
            )?;
            reserve(s, &mut w.qkv, rows * qkv_width)?;
            reserve(s, &mut w.q, heads * rows * b.head_dim)?;
            reserve(s, &mut w.k, heads * rows * b.head_dim)?;
            reserve(s, &mut w.v, heads * rows * b.head_dim)?;
            reserve(s, &mut w.scores, heads * length * length)?;
            reserve(s, &mut w.probabilities, heads * length * length)?;
            reserve(s, &mut w.attention, heads * rows * b.head_dim)?;
            reserve(s, &mut w.merged, rows * heads * b.head_dim)?;
            reserve(s, &mut w.gate_up, rows * 2 * b.intermediate_size)?;
            reserve(s, &mut w.activated, rows * b.intermediate_size)?;
            reserve(s, &mut w.final_rows, 2 * frames * b.hidden_size)?;
            reserve(s, &mut w.final_normed, 2 * frames * b.hidden_size)?;
            reserve(
                s,
                &mut w.logits,
                2 * frames * codebooks * self.config.audio_vocab_size,
            )?;
            reserve(s, &mut w.prediction, codebooks * frames)?;
            reserve(s, &mut w.confidence, codebooks * frames)?;
            if w.ids.len() < codebooks * rows {
                w.ids = s.alloc_indices(codebooks * rows, w.ids.bound())?;
            }
            Ok(())
        })()
        .map_err(hip_error)?;
        // Text embeddings: Hugging Face's float32 rows of the first codebook's
        // IDs.
        let table =
            self.weights
                .expect(TEXT_EMBEDDINGS, DType::F32, &[b.vocab_size, b.hidden_size])?;
        let table = self.weights.bytes(table);
        let row_bytes = b.hidden_size * 4;
        let mut text = Vec::with_capacity(input.audio_start * b.hidden_size);
        for &id in &input.ids[..input.audio_start] {
            let id = usize::try_from(id)
                .ok()
                .filter(|&id| id < b.vocab_size)
                .ok_or_else(|| "text token outside the vocabulary".to_string())?;
            let row = &table[id * row_bytes..(id + 1) * row_bytes];
            text.extend(
                row.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|raw| f32::from_le_bytes(*raw)),
            );
        }
        let mut positions = vec![0i32; rows];
        let mut target_rows = vec![0u32; 2 * frames];
        for (r, position) in positions.iter_mut().enumerate().take(length) {
            *position = r as i32;
        }
        for f in 0..frames {
            positions[length + f] = f as i32;
            target_rows[f] = (length - frames + f) as u32;
            target_rows[frames + f] = (length + f) as u32;
        }
        let w = &mut self.work;
        (|| -> Result<(), HipError> {
            s.upload(&text, w.text.slice_mut(0..text.len()))?;
            s.upload(&positions, w.positions.slice_mut(0..rows))?;
            let bound = u32::try_from(rows).unwrap_or(u32::MAX);
            if w.target_rows.len() < target_rows.len() || w.target_rows.bound() != bound {
                w.target_rows = s.alloc_indices(target_rows.len().max(1), bound)?;
            }
            s.upload_indices(&target_rows, &mut w.target_rows)
        })()
        .map_err(hip_error)?;
        self.upload_ids(input)?;
        Ok(rows)
    }

    /// Codes of the packed batch: the conditional sequence, then its target
    /// frames again as the unconditional sequence. Text positions upload 0.
    fn upload_ids(&mut self, input: &ConditionalInput) -> Result<(), String> {
        let length = input.length;
        let frames = input.target_frames;
        let rows = length + frames;
        self.host_ids.clear();
        self.host_ids.resize(input.codebooks * rows, 0);
        for c in 0..input.codebooks {
            let source = &input.ids[c * length..(c + 1) * length];
            let row = &mut self.host_ids[c * rows..(c + 1) * rows];
            for (target, &id) in row[input.audio_start..length]
                .iter_mut()
                .zip(&source[input.audio_start..])
            {
                *target = id as u32;
            }
            for (target, &id) in row[length..].iter_mut().zip(&source[length - frames..]) {
                *target = id as u32;
            }
        }
        self.stream
            .upload_indices(&self.host_ids, &mut self.work.ids)
            .map_err(hip_error)
    }

    /// Bidirectional attention within one sequence of `length` rows starting
    /// at `first`, for every head at once.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        gemm: &mut HalfGemm,
        s: &Stream,
        w: &mut Workspace,
        b: &BackboneConfig,
        first: usize,
        length: usize,
        rows: usize,
    ) -> Result<(), LaunchError> {
        let d = b.head_dim;
        let heads = b.heads;
        // Tuning is shared by sequences and batches of similar size.
        let bucket = tuning_bucket(length);
        let bucket_rows = tuning_bucket(rows);
        // S[i][j] = q_i . k_j, row-major [length][length] per head.
        let scores = MatmulShape {
            transpose_a: true,
            m: length,
            n: length,
            k: d,
            lda: d,
            ldb: d,
            ldc: length,
            stride_a: rows * d,
            stride_b: rows * d,
            stride_c: length * length,
            batch: heads,
            ..MatmulShape::default()
        };
        let scores_bucket = MatmulShape {
            m: bucket,
            n: bucket,
            ldc: bucket,
            stride_a: bucket_rows * d,
            stride_b: bucket_rows * d,
            stride_c: bucket * bucket,
            ..scores
        };
        let end = heads * rows * d;
        gemm.matmul(
            s,
            &scores,
            &scores_bucket,
            w.k.slice(first * d..end),
            w.q.slice(first * d..end),
            w.scores.slice_mut(0..heads * length * length),
        )?;
        k::softmax(
            s,
            w.scores.slice(0..heads * length * length),
            w.probabilities.slice_mut(0..heads * length * length),
            heads * length,
            length,
            1.0 / (d as f32).sqrt(),
        )?;
        // O[i] = sum_j P[i][j] v_j into the head-major attention rows.
        let values = MatmulShape {
            m: d,
            n: length,
            k: length,
            lda: d,
            ldb: length,
            ldc: d,
            stride_a: rows * d,
            stride_b: length * length,
            stride_c: rows * d,
            batch: heads,
            ..MatmulShape::default()
        };
        let values_bucket = MatmulShape {
            n: bucket,
            k: bucket,
            ldb: bucket,
            stride_a: bucket_rows * d,
            stride_c: bucket_rows * d,
            stride_b: bucket * bucket,
            ..values
        };
        gemm.matmul(
            s,
            &values,
            &values_bucket,
            w.v.slice(first * d..end),
            w.probabilities.slice(0..heads * length * length),
            w.attention.slice_mut(first * d..end),
        )
    }

    /// The backbone over the packed batch and the audio heads on the target
    /// rows; leaves the logits in `work.logits`.
    fn run_forward(&mut self, input: &ConditionalInput, rows: usize) -> Result<(), LaunchError> {
        let b = self.config.backbone.clone();
        let length = input.length;
        let frames = input.target_frames;
        let hidden = b.hidden_size;
        let qkv_width = self.qkv_width();
        let vocab = self.config.audio_vocab_size;
        let heads_width = b.heads * b.head_dim;
        let s = &self.stream;
        {
            let w = &mut self.work;
            k::embed(
                s,
                w.text.slice(0..input.audio_start * hidden),
                input.audio_start,
                self.audio_embeddings.as_slice(),
                w.ids.slice(0..input.codebooks * rows),
                w.hidden.slice_mut(0..rows * hidden),
                rows,
                hidden,
                input.codebooks,
                vocab,
            )?;
        }
        for index in 0..self.layers.len() {
            let layer = &self.layers[index];
            let w = &mut self.work;
            k::rms_norm(
                s,
                w.hidden.slice(0..rows * hidden),
                layer.input_norm.as_slice(),
                w.normed.slice_mut(0..rows * hidden),
                rows,
                hidden,
                b.rms_norm_eps,
            )?;
            self.gemm.linear(
                s,
                w.normed.slice(0..rows * hidden),
                layer.qkv.as_slice(),
                w.qkv.slice_mut(0..rows * qkv_width),
                rows,
                hidden,
                qkv_width,
                false,
            )?;
            let qkv_len = b.heads * rows * b.head_dim;
            k::qk_norm_rope(
                s,
                w.qkv.slice(0..rows * qkv_width),
                layer.q_norm.as_slice(),
                layer.k_norm.as_slice(),
                w.positions.slice(0..rows),
                w.q.slice_mut(0..qkv_len),
                w.k.slice_mut(0..qkv_len),
                w.v.slice_mut(0..qkv_len),
                rows,
                b.heads,
                b.kv_heads,
                b.head_dim,
                b.rms_norm_eps,
                b.rope_theta,
            )?;
            Self::attention(&mut self.gemm, s, w, &b, 0, length, rows)?;
            Self::attention(&mut self.gemm, s, w, &b, length, frames, rows)?;
            k::merge_heads(
                s,
                w.attention.slice(0..qkv_len),
                w.merged.slice_mut(0..rows * heads_width),
                rows,
                b.heads,
                b.head_dim,
            )?;
            self.gemm.linear(
                s,
                w.merged.slice(0..rows * heads_width),
                layer.output.as_slice(),
                w.hidden.slice_mut(0..rows * hidden),
                rows,
                heads_width,
                hidden,
                true,
            )?;
            k::rms_norm(
                s,
                w.hidden.slice(0..rows * hidden),
                layer.post_norm.as_slice(),
                w.normed.slice_mut(0..rows * hidden),
                rows,
                hidden,
                b.rms_norm_eps,
            )?;
            self.gemm.linear(
                s,
                w.normed.slice(0..rows * hidden),
                layer.gate_up.as_slice(),
                w.gate_up.slice_mut(0..rows * 2 * b.intermediate_size),
                rows,
                hidden,
                2 * b.intermediate_size,
                false,
            )?;
            k::swiglu(
                s,
                w.gate_up.slice(0..rows * 2 * b.intermediate_size),
                w.activated.slice_mut(0..rows * b.intermediate_size),
                rows,
                b.intermediate_size,
            )?;
            self.gemm.linear(
                s,
                w.activated.slice(0..rows * b.intermediate_size),
                layer.down.as_slice(),
                w.hidden.slice_mut(0..rows * hidden),
                rows,
                b.intermediate_size,
                hidden,
                true,
            )?;
        }
        let w = &mut self.work;
        k::gather_rows(
            s,
            w.hidden.slice(0..rows * hidden),
            w.target_rows.slice(0..2 * frames),
            w.final_rows.slice_mut(0..2 * frames * hidden),
            2 * frames,
            hidden,
            rows,
        )?;
        k::rms_norm(
            s,
            w.final_rows.slice(0..2 * frames * hidden),
            self.final_norm.as_slice(),
            w.final_normed.slice_mut(0..2 * frames * hidden),
            2 * frames,
            hidden,
            b.rms_norm_eps,
        )?;
        let logits = 2 * frames * input.codebooks * vocab;
        self.gemm.linear(
            s,
            w.final_normed.slice(0..2 * frames * hidden),
            self.audio_heads.as_slice(),
            w.logits.slice_mut(0..logits),
            2 * frames,
            hidden,
            input.codebooks * vocab,
            false,
        )
    }

    /// The scoring step on the logits of the last forward pass.
    fn score_logits(&mut self, input: &ConditionalInput, guidance: f32) -> Result<(), LaunchError> {
        let frames = input.target_frames;
        let codebooks = input.codebooks;
        let vocab = self.config.audio_vocab_size;
        let w = &mut self.work;
        k::score(
            &self.stream,
            w.logits.slice(0..2 * frames * codebooks * vocab),
            w.prediction.slice_mut(0..codebooks * frames),
            w.confidence.slice_mut(0..codebooks * frames),
            frames,
            codebooks,
            vocab,
            self.config.audio_mask_id,
            guidance,
        )
    }

    fn download_scores(
        &mut self,
        count: usize,
        prediction: &mut [i32],
        confidence: &mut [f32],
    ) -> Result<(), HipError> {
        let w = &self.work;
        self.stream
            .download_blocking(w.prediction.slice(0..count), prediction)?;
        self.stream
            .download_blocking(w.confidence.slice(0..count), confidence)
    }

    /// One forward pass of the CFG batch: the target frames' logits,
    /// conditional then unconditional, `[2 * frames][codebooks * vocab]`.
    pub fn forward(&mut self, input: &ConditionalInput) -> Result<Vec<f32>, String> {
        let rows = self
            .prepare(input)
            .map_err(|e| format!("OmniVoice forward: {e}"))?;
        self.run_forward(input, rows)
            .map_err(|e| format!("OmniVoice forward: {e}"))?;
        let mut logits =
            vec![0.0f32; 2 * input.target_frames * input.codebooks * self.config.audio_vocab_size];
        self.stream
            .download_blocking(self.work.logits.slice(0..logits.len()), &mut logits)
            .map_err(|e| format!("OmniVoice forward: {e}"))?;
        Ok(logits)
    }

    /// Forward plus the scoring step: per `[codebook][frame]` the guided
    /// argmax and its log-probability.
    pub fn score(
        &mut self,
        input: &ConditionalInput,
        guidance: f32,
    ) -> Result<(Vec<i32>, Vec<f32>), String> {
        let fail = |e: String| format!("OmniVoice score: {e}");
        let rows = self.prepare(input).map_err(fail)?;
        self.run_forward(input, rows)
            .and_then(|()| self.score_logits(input, guidance))
            .map_err(|e| fail(e.to_string()))?;
        let count = input.codebooks * input.target_frames;
        let mut prediction = vec![0i32; count];
        let mut confidence = vec![0f32; count];
        self.download_scores(count, &mut prediction, &mut confidence)
            .map_err(|e| fail(e.to_string()))?;
        Ok((prediction, confidence))
    }

    /// Runs the unmasking loop from `input`'s masked target. `cancelled` is
    /// polled before every step.
    pub fn generate(
        &mut self,
        input: &ConditionalInput,
        decoding: &DecodingConfig,
        seed: u64,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<GenerationResult, String> {
        let fail = |e: String| format!("OmniVoice generation: {e}");
        let codebooks = input.codebooks;
        let frames = input.target_frames;
        let count = codebooks * frames;
        let mask = self.config.audio_mask_id;
        let mut current = input.clone();
        let rows = self.prepare(&current).map_err(fail)?;
        let mut result = GenerationResult {
            codes: vec![mask as i32; count],
            unmasked_at: vec![-1; count],
        };
        let mut prediction = vec![0i32; count];
        let mut confidence = vec![0f32; count];
        let mut selector = PositionSelector::new(decoding, seed);
        let schedule = unmask_schedule(count, decoding.steps, decoding.t_shift);
        for (step, &fill) in schedule.iter().enumerate() {
            if fill == 0 {
                continue;
            }
            if cancelled() {
                return Err("cancelled".to_string());
            }
            self.run_forward(&current, rows)
                .and_then(|()| self.score_logits(&current, decoding.guidance))
                .map_err(|e| fail(e.to_string()))?;
            self.download_scores(count, &mut prediction, &mut confidence)
                .map_err(|e| fail(e.to_string()))?;
            for index in selector.fill(
                fill,
                codebooks,
                frames,
                &prediction,
                &confidence,
                mask,
                &mut result.codes,
            ) {
                result.unmasked_at[index] = step as i32;
            }
            // Write the filled target back into both sequences.
            let length = current.length;
            for c in 0..codebooks {
                current.ids[c * length + length - frames..(c + 1) * length]
                    .copy_from_slice(&result.codes[c * frames..(c + 1) * frames]);
            }
            self.upload_ids(&current).map_err(fail)?;
        }
        self.stream.synchronize().map_err(|e| fail(e.to_string()))?;
        Ok(result)
    }
}
