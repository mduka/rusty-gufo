//! Text decoder: 28 Qwen3 layers over an f32 KV cache, greedy decoding.
//! Launch order, kernel routes and rounding boundaries are those of
//! `hip/text_decoder_runtime.cpp`.

use gufo_hip::{
    Bf16, DeviceBuffer, DeviceCopy, DeviceIndices, DeviceSlice, DeviceSliceMut, Launch,
    LaunchError, Stream,
};
use gufo_qwen3_asr_kernels as k;

use crate::config::TextConfig;
use crate::gemm::Projector;
use crate::prompt::{AUDIO_PAD, ENDOFTEXT, IM_END};
use crate::weights::DecoderWeights;

/// Largest `LaunchTextDecodeGemv` batch: q/k/v in one dispatch.
const ARGMAX_SCRATCH: usize = 2048;

fn launch(error: LaunchError) -> String {
    format!("Qwen3-ASR text decoder: {error}")
}

/// Buffers live as long as the model: exactly sized, outside the pool.
fn alloc<T: DeviceCopy>(stream: &Stream, len: usize) -> Result<DeviceBuffer<T>, String> {
    stream
        .alloc_persistent(len)
        .map_err(|e| format!("Qwen3-ASR text buffer: {e}"))
}

pub fn is_stop_token(token: u32) -> bool {
    token == ENDOFTEXT || token == IM_END
}

/// One projection of a fused batch: weights `[rows, columns]` into `output`.
struct Projection<'a> {
    weights: DeviceSlice<'a, Bf16>,
    output: DeviceSliceMut<'a, f32>,
    rows: usize,
}

pub struct TextDecoder {
    config: TextConfig,
    capacity: usize,
    cache_tokens: usize,
    projector: Projector,
    hidden: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    attention: DeviceBuffer<f32>,
    projection: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    feed_forward: DeviceBuffer<f32>,
    bf16_scratch: DeviceBuffer<Bf16>,
    gemv_input: DeviceBuffer<f32>,
    key_cache: DeviceBuffer<f32>,
    value_cache: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    argmax_scratch: DeviceBuffer<f32>,
    token_ids: DeviceIndices,
    selected: DeviceBuffer<u32>,
}

/// Host copies of the decoder's prefill parity boundaries.
#[derive(Default)]
pub struct DecoderTrace {
    /// After layer 0, `[prompt, 2048]`.
    pub layer0: Vec<f32>,
    /// BF16-rounded logits of the last prompt position.
    pub logits: Vec<f32>,
}

impl TextDecoder {
    pub fn new(stream: &Stream, config: &TextConfig, capacity: usize) -> Result<Self, String> {
        if capacity == 0 {
            return Err("Qwen3-ASR maximum text token count must be positive".into());
        }
        let hidden = config.hidden_size as usize;
        let ffn = config.intermediate_size as usize;
        let q_width = (config.num_attention_heads * config.head_dim) as usize;
        let kv_width = (config.num_key_value_heads * config.head_dim) as usize;
        let cache = config.num_hidden_layers as usize * kv_width * capacity;
        Ok(Self {
            config: config.clone(),
            capacity,
            cache_tokens: 0,
            projector: Projector::new(stream)?,
            hidden: alloc(stream, capacity * hidden)?,
            q: alloc(stream, capacity * q_width)?,
            k: alloc(stream, capacity * kv_width)?,
            v: alloc(stream, capacity * kv_width)?,
            attention: alloc(stream, capacity * hidden)?,
            projection: alloc(stream, capacity * hidden)?,
            gate: alloc(stream, capacity * ffn)?,
            up: alloc(stream, capacity * ffn)?,
            feed_forward: alloc(stream, capacity * hidden)?,
            bf16_scratch: alloc(stream, capacity * hidden.max(ffn))?,
            gemv_input: alloc(stream, hidden.max(ffn))?,
            key_cache: alloc(stream, cache)?,
            value_cache: alloc(stream, cache)?,
            logits: alloc(stream, config.vocab_size as usize)?,
            argmax_scratch: alloc(stream, ARGMAX_SCRATCH)?,
            token_ids: stream
                .zeroed_indices(alloc(stream, capacity)?, config.vocab_size)
                .map_err(|e| format!("Qwen3-ASR text buffer: {e}"))?,
            selected: alloc(stream, 1)?,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Projections sharing one BF16 activation matrix. Batch one runs them
    /// as a single fused GEMV dispatch, falling back to one GEMV per tensor;
    /// prefill issues one GEMM per tensor.
    fn project_fused(
        projector: &mut Projector,
        gemv_input: &mut DeviceBuffer<f32>,
        stream: &Stream,
        mut projections: Vec<Projection<'_>>,
        input: DeviceSlice<'_, Bf16>,
        batch: usize,
        columns: usize,
    ) -> Result<(), String> {
        if batch == 1 && projections.len() <= 3 {
            let rows: Option<Vec<u32>> = projections
                .iter()
                .map(|p| u32::try_from(p.rows).ok())
                .collect();
            if let Some(rows) = rows {
                let mut it = projections.iter_mut();
                let first = it.next().ok_or("empty projection batch")?;
                let (second, third) = (it.next(), it.next());
                let launched = k::decode_gemv(
                    stream,
                    first.weights,
                    first.output.reborrow(),
                    rows[0],
                    second.as_ref().map(|p| p.weights),
                    second.map(|p| p.output.reborrow()),
                    rows.get(1).copied().unwrap_or(0),
                    third.as_ref().map(|p| p.weights),
                    third.map(|p| p.output.reborrow()),
                    rows.get(2).copied().unwrap_or(0),
                    input.range(0..columns),
                    columns,
                )
                .map_err(launch)?;
                if launched == Launch::Launched {
                    return Ok(());
                }
            }
        }
        for p in projections {
            if batch == 1 {
                // Fallback for a geometry the fused kernel declines.
                k::bf16_to_f32(stream, input, gemv_input.as_mut_slice(), columns)
                    .map_err(launch)?;
                k::row_gemv(
                    stream,
                    p.weights,
                    gemv_input.as_slice(),
                    p.output,
                    p.rows,
                    columns,
                )
                .map_err(launch)?;
                continue;
            }
            projector
                .project(
                    stream,
                    p.weights,
                    input.range(0..batch * columns),
                    p.output,
                    batch,
                    p.rows,
                    columns,
                )
                .map_err(launch)?;
        }
        Ok(())
    }

    /// Embeds the prompt, replaces the audio-pad rows with `audio` (f32
    /// `[audio_tokens, hidden]`, e.g. the encoder output) and fills the KV
    /// cache.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(
        &mut self,
        stream: &Stream,
        w: &DecoderWeights,
        prompt: &[u32],
        audio: DeviceSlice<'_, f32>,
        audio_tokens: usize,
        cancelled: &dyn Fn() -> Result<(), String>,
        trace: Option<&mut DecoderTrace>,
    ) -> Result<(), String> {
        let hidden = self.config.hidden_size as usize;
        if prompt.is_empty()
            || prompt.len() > self.capacity
            || audio_tokens == 0
            || audio.len() != audio_tokens * hidden
        {
            return Err("Qwen3-ASR text prompt shape is invalid".into());
        }
        let mut audio_offset = prompt.len();
        let mut found = 0usize;
        for (index, &token) in prompt.iter().enumerate() {
            if token >= self.config.vocab_size {
                return Err("Qwen3-ASR prompt token is out of range".into());
            }
            if token == AUDIO_PAD {
                if found == 0 {
                    audio_offset = index;
                } else if index != audio_offset + found {
                    return Err("Qwen3-ASR audio prompt tokens must be contiguous".into());
                }
                found += 1;
            }
        }
        if found != audio_tokens {
            return Err("Qwen3-ASR prompt/audio embedding count differs".into());
        }
        stream
            .upload_indices(prompt, &mut self.token_ids)
            .map_err(|e| format!("Qwen3-ASR prompt ids: {e}"))?;
        k::embedding_lookup(
            stream,
            w.embedding.view(),
            self.token_ids.view(),
            self.hidden.as_mut_slice(),
            prompt.len(),
            hidden,
            self.config.vocab_size as usize,
        )
        .map_err(launch)?;
        let rows = audio_offset * hidden..(audio_offset + audio_tokens) * hidden;
        stream
            .copy(audio, self.hidden.slice_mut(rows.clone()))
            .map_err(|e| format!("Qwen3-ASR audio embeddings: {e}"))?;
        k::round_bf16(stream, self.hidden.slice_mut(rows), audio.len()).map_err(launch)?;
        self.run_tokens(stream, w, prompt.len(), 0, cancelled, trace)?;
        self.cache_tokens = prompt.len();
        Ok(())
    }

    /// Feeds one generated token.
    pub fn decode(
        &mut self,
        stream: &Stream,
        w: &DecoderWeights,
        token: u32,
        cancelled: &dyn Fn() -> Result<(), String>,
    ) -> Result<(), String> {
        if self.cache_tokens >= self.capacity {
            return Err("Qwen3-ASR generation exceeds KV capacity".into());
        }
        if token >= self.config.vocab_size {
            return Err("Qwen3-ASR generated token is out of range".into());
        }
        stream
            .upload_indices(&[token], &mut self.token_ids)
            .map_err(|e| format!("Qwen3-ASR generated token: {e}"))?;
        k::embedding_lookup(
            stream,
            w.embedding.view(),
            self.token_ids.view(),
            self.hidden.as_mut_slice(),
            1,
            self.config.hidden_size as usize,
            self.config.vocab_size as usize,
        )
        .map_err(launch)?;
        let position = self.cache_tokens;
        self.run_tokens(stream, w, 1, position, cancelled, None)?;
        self.cache_tokens += 1;
        Ok(())
    }

    /// Greedy token of the last logits; `None` when a logit is not finite.
    pub fn select_token(&mut self, stream: &Stream) -> Result<Option<u32>, String> {
        let vocab = self.config.vocab_size as usize;
        k::argmax(
            stream,
            self.logits.as_slice(),
            self.selected.as_mut_slice(),
            vocab,
            Some(self.argmax_scratch.as_mut_slice()),
        )
        .map_err(launch)?;
        let mut token = [0u32];
        stream
            .download(self.selected.as_slice(), &mut token)
            .map_err(|e| format!("Qwen3-ASR selected token: {e}"))?;
        Ok((token[0] < self.config.vocab_size).then_some(token[0]))
    }

    fn run_tokens(
        &mut self,
        stream: &Stream,
        w: &DecoderWeights,
        tokens: usize,
        start: usize,
        cancelled: &dyn Fn() -> Result<(), String>,
        mut trace: Option<&mut DecoderTrace>,
    ) -> Result<(), String> {
        let c = &self.config;
        if tokens == 0 || start + tokens > self.capacity {
            return Err("Qwen3-ASR text step exceeds KV capacity".into());
        }
        let hidden = c.hidden_size as usize;
        let ffn = c.intermediate_size as usize;
        let (heads, kv_heads, head_dim) =
            (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let q_rows = (heads * head_dim) as usize;
        let kv_rows = (kv_heads * head_dim) as usize;
        let capacity = u32::try_from(self.capacity).map_err(|_| "capacity exceeds u32")?;
        let start_position = u32::try_from(start).map_err(|_| "position exceeds u32")?;
        let (eps, theta) = (c.rms_norm_eps, c.rope_theta);
        let norm = |range: &std::ops::Range<usize>| w.norms.slice(range.clone());

        k::rms_norm(
            stream,
            self.hidden.as_slice(),
            norm(&w.layers[0].input_norm),
            self.bf16_scratch.as_mut_slice(),
            tokens,
            hidden,
            eps,
        )
        .map_err(launch)?;
        for (index, layer) in w.layers.iter().enumerate() {
            cancelled()?;
            let layer_index = index as u32;
            Self::project_fused(
                &mut self.projector,
                &mut self.gemv_input,
                stream,
                vec![
                    Projection {
                        weights: layer.q.view(),
                        output: self.q.as_mut_slice(),
                        rows: q_rows,
                    },
                    Projection {
                        weights: layer.k.view(),
                        output: self.k.as_mut_slice(),
                        rows: kv_rows,
                    },
                    Projection {
                        weights: layer.v.view(),
                        output: self.v.as_mut_slice(),
                        rows: kv_rows,
                    },
                ],
                self.bf16_scratch.as_slice(),
                tokens,
                hidden,
            )?;
            let rope = k::qk_norm_rope(
                stream,
                self.q.as_mut_slice(),
                self.k.as_mut_slice(),
                self.v.as_mut_slice(),
                norm(&layer.q_norm),
                norm(&layer.k_norm),
                tokens,
                heads,
                kv_heads,
                head_dim,
                start_position,
                theta,
                eps,
                Some(self.key_cache.as_mut_slice()),
                Some(self.value_cache.as_mut_slice()),
                layer_index,
                capacity,
            )
            .map_err(launch)?;
            if rope != Launch::Launched {
                return Err("Qwen3-ASR text q/k normalization shape is unsupported".into());
            }
            let decoded = tokens == 1
                && k::decode_attention(
                    stream,
                    self.q.as_slice(),
                    self.key_cache.as_slice(),
                    self.value_cache.as_slice(),
                    self.attention.as_mut_slice(),
                    Some(self.bf16_scratch.as_mut_slice()),
                    layer_index,
                    start_position,
                    capacity,
                    heads,
                    kv_heads,
                    head_dim,
                )
                .map_err(launch)?
                    == Launch::Launched;
            if !decoded {
                k::batched_attention(
                    stream,
                    self.q.as_slice(),
                    self.key_cache.as_slice(),
                    self.value_cache.as_slice(),
                    self.attention.as_mut_slice(),
                    Some(self.bf16_scratch.as_mut_slice()),
                    layer_index,
                    start_position,
                    tokens,
                    capacity,
                    heads,
                    kv_heads,
                    head_dim,
                )
                .map_err(launch)?;
            }
            Self::project_fused(
                &mut self.projector,
                &mut self.gemv_input,
                stream,
                vec![Projection {
                    weights: layer.o.view(),
                    output: self.projection.as_mut_slice(),
                    rows: hidden,
                }],
                self.bf16_scratch.as_slice(),
                tokens,
                q_rows,
            )?;
            k::residual_add_rms_norm(
                stream,
                self.hidden.as_mut_slice(),
                self.projection.as_slice(),
                norm(&layer.post_norm),
                self.bf16_scratch.as_mut_slice(),
                tokens,
                hidden,
                eps,
            )
            .map_err(launch)?;
            Self::project_fused(
                &mut self.projector,
                &mut self.gemv_input,
                stream,
                vec![
                    Projection {
                        weights: layer.gate.view(),
                        output: self.gate.as_mut_slice(),
                        rows: ffn,
                    },
                    Projection {
                        weights: layer.up.view(),
                        output: self.up.as_mut_slice(),
                        rows: ffn,
                    },
                ],
                self.bf16_scratch.as_slice(),
                tokens,
                hidden,
            )?;
            k::swiglu(
                stream,
                self.gate.as_slice(),
                self.up.as_slice(),
                self.bf16_scratch.as_mut_slice(),
                tokens * ffn,
            )
            .map_err(launch)?;
            Self::project_fused(
                &mut self.projector,
                &mut self.gemv_input,
                stream,
                vec![Projection {
                    weights: layer.down.view(),
                    output: self.feed_forward.as_mut_slice(),
                    rows: hidden,
                }],
                self.bf16_scratch.as_slice(),
                tokens,
                ffn,
            )?;
            let next_norm = w
                .layers
                .get(index + 1)
                .map_or(&w.final_norm, |next| &next.input_norm);
            k::residual_add_rms_norm(
                stream,
                self.hidden.as_mut_slice(),
                self.feed_forward.as_slice(),
                norm(next_norm),
                self.bf16_scratch.as_mut_slice(),
                tokens,
                hidden,
                eps,
            )
            .map_err(launch)?;
            if index == 0
                && let Some(trace) = trace.as_deref_mut()
            {
                trace.layer0 = download(stream, self.hidden.slice(0..tokens * hidden))?;
            }
        }

        let last = (tokens - 1) * hidden..tokens * hidden;
        let vocab = c.vocab_size as usize;
        Self::project_fused(
            &mut self.projector,
            &mut self.gemv_input,
            stream,
            vec![Projection {
                weights: w.lm_head.view(),
                output: self.logits.as_mut_slice(),
                rows: vocab,
            }],
            self.bf16_scratch.slice(last),
            1,
            hidden,
        )?;
        k::round_bf16(stream, self.logits.as_mut_slice(), vocab).map_err(launch)?;
        if let Some(trace) = trace {
            trace.logits = download(stream, self.logits.as_slice())?;
        }
        Ok(())
    }
}

fn download(stream: &Stream, source: DeviceSlice<'_, f32>) -> Result<Vec<f32>, String> {
    let mut values = vec![0.0; source.len()];
    stream
        .download(source, &mut values)
        .map_err(|e| format!("Qwen3-ASR trace download: {e}"))?;
    Ok(values)
}
