//! Audio encoder: chunked log-mel features through three stride-two
//! convolutions, `conv_out`, sinusoidal positions, 24 transformer layers and
//! the two output projections. Launch order and rounding boundaries are those
//! of `hip/audio_encoder_runtime.cpp`.

use gufo_hip::{Bf16, DeviceBuffer, DeviceCopy, DeviceSlice, LaunchError, Stream};
use gufo_qwen3_asr_kernels as k;

use crate::audio::MEL_BINS;
use crate::gemm::Projector;
use crate::weights::{
    AUDIO_FFN, AUDIO_HIDDEN, AUDIO_OUTPUT, CONV_CHANNELS, CONV_FLATTENED, EncoderWeights,
};

const CHUNK_FRAMES: usize = 100;
const CONV_TIME: usize = 13;
const HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const LAYER_NORM_EPSILON: f32 = 1.0e-5;

/// Checks for cancellation between layers; `Err` aborts the request.
pub type Checkpoint<'a> = &'a dyn Fn() -> Result<(), String>;

fn grow<T: DeviceCopy>(
    stream: &Stream,
    buffer: &mut Option<DeviceBuffer<T>>,
    len: usize,
) -> Result<(), String> {
    if buffer.as_ref().is_none_or(|b| b.len() < len) {
        *buffer = None;
        *buffer = Some(
            stream
                .alloc_persistent(len)
                .map_err(|e| format!("Qwen3-ASR encoder buffer: {e}"))?,
        );
    }
    Ok(())
}

fn launch(error: LaunchError) -> String {
    format!("Qwen3-ASR audio encoder: {error}")
}

/// Scratch buffers, grown on demand like the C++ runtime's.
#[derive(Default)]
struct Scratch {
    feature_input: Option<DeviceBuffer<f32>>,
    input: Option<DeviceBuffer<Bf16>>,
    conv1: Option<DeviceBuffer<Bf16>>,
    conv2: Option<DeviceBuffer<Bf16>>,
    conv3: Option<DeviceBuffer<Bf16>>,
    layout: Option<DeviceBuffer<Bf16>>,
    projected: Option<DeviceBuffer<f32>>,
    compact: Option<DeviceBuffer<f32>>,
    hidden: Option<DeviceBuffer<f32>>,
    normalized: Option<DeviceBuffer<Bf16>>,
    q: Option<DeviceBuffer<f32>>,
    k: Option<DeviceBuffer<f32>>,
    v: Option<DeviceBuffer<f32>>,
    attention: Option<DeviceBuffer<f32>>,
    attention_bf16: Option<DeviceBuffer<Bf16>>,
    projection: Option<DeviceBuffer<f32>>,
    ffn: Option<DeviceBuffer<f32>>,
    ffn_bf16: Option<DeviceBuffer<Bf16>>,
}

pub struct AudioEncoder {
    projector: Projector,
    scratch: Scratch,
    finite: DeviceBuffer<u32>,
}

/// Borrow a grown scratch buffer.
macro_rules! buf {
    ($field:expr) => {
        $field
            .as_mut()
            .ok_or_else(|| "Qwen3-ASR encoder scratch is missing".to_string())?
    };
}

impl AudioEncoder {
    pub fn new(stream: &Stream) -> Result<Self, String> {
        Ok(Self {
            projector: Projector::new(stream)?,
            scratch: Scratch::default(),
            finite: stream
                .alloc_persistent(1)
                .map_err(|e| format!("Qwen3-ASR encoder buffer: {e}"))?,
        })
    }

    fn ensure_chunks(&mut self, stream: &Stream, chunks: usize) -> Result<(), String> {
        let s = &mut self.scratch;
        grow(
            stream,
            &mut s.feature_input,
            chunks * MEL_BINS * CHUNK_FRAMES,
        )?;
        grow(stream, &mut s.input, chunks * MEL_BINS * CHUNK_FRAMES)?;
        grow(stream, &mut s.conv1, chunks * CONV_CHANNELS * 64 * 50)?;
        grow(stream, &mut s.conv2, chunks * CONV_CHANNELS * 32 * 25)?;
        grow(
            stream,
            &mut s.conv3,
            chunks * CONV_CHANNELS * 16 * CONV_TIME,
        )?;
        grow(stream, &mut s.layout, chunks * CONV_TIME * CONV_FLATTENED)?;
        grow(stream, &mut s.projected, chunks * CONV_TIME * AUDIO_HIDDEN)?;
        grow(stream, &mut s.compact, chunks * CONV_TIME * AUDIO_HIDDEN)
    }

    fn ensure_tokens(&mut self, stream: &Stream, tokens: usize) -> Result<(), String> {
        let s = &mut self.scratch;
        grow(stream, &mut s.hidden, tokens * AUDIO_HIDDEN)?;
        grow(stream, &mut s.normalized, tokens * AUDIO_HIDDEN)?;
        grow(stream, &mut s.q, tokens * AUDIO_HIDDEN)?;
        grow(stream, &mut s.k, tokens * AUDIO_HIDDEN)?;
        grow(stream, &mut s.v, tokens * AUDIO_HIDDEN)?;
        grow(stream, &mut s.attention, tokens * AUDIO_HIDDEN)?;
        grow(stream, &mut s.attention_bf16, tokens * AUDIO_HIDDEN)?;
        grow(stream, &mut s.projection, tokens * AUDIO_OUTPUT)?;
        grow(stream, &mut s.ffn, tokens * AUDIO_FFN)?;
        grow(stream, &mut s.ffn_bf16, tokens * AUDIO_FFN)
    }

    /// Convolutional frontend into `compact` (f32 `[tokens, 1024]`); returns
    /// the token count.
    fn frontend(
        &mut self,
        stream: &Stream,
        w: &EncoderWeights,
        log_mel: &[f32],
        frames: usize,
    ) -> Result<usize, String> {
        if frames == 0 || log_mel.len() != MEL_BINS * frames {
            return Err("Qwen3-ASR log-mel tensor must have shape [128, frames]".into());
        }
        let chunks = frames.div_ceil(CHUNK_FRAMES);
        self.ensure_chunks(stream, chunks)?;
        let s = &mut self.scratch;
        let feature_input = buf!(s.feature_input);
        stream
            .upload(log_mel, feature_input.slice_mut(0..log_mel.len()))
            .map_err(|e| format!("Qwen3-ASR log-mel upload: {e}"))?;
        let input = buf!(s.input);
        k::chunk_log_mel(
            stream,
            feature_input.as_slice(),
            input.as_mut_slice(),
            frames,
            chunks,
        )
        .map_err(launch)?;
        let conv1 = buf!(s.conv1);
        k::convolution3x3(
            stream,
            input.as_slice(),
            w.conv1_weight.view(),
            conv1.as_mut_slice(),
            chunks,
            1,
            128,
            100,
        )
        .map_err(launch)?;
        let spatial1 = 64 * 50;
        k::bias_gelu(
            stream,
            conv1.as_mut_slice(),
            w.conv1_bias.view(),
            chunks * CONV_CHANNELS * spatial1,
            CONV_CHANNELS,
            spatial1,
        )
        .map_err(launch)?;
        let conv2 = buf!(s.conv2);
        k::convolution3x3(
            stream,
            conv1.as_slice(),
            w.conv2_weight.view(),
            conv2.as_mut_slice(),
            chunks,
            CONV_CHANNELS,
            64,
            50,
        )
        .map_err(launch)?;
        let spatial2 = 32 * 25;
        k::bias_gelu(
            stream,
            conv2.as_mut_slice(),
            w.conv2_bias.view(),
            chunks * CONV_CHANNELS * spatial2,
            CONV_CHANNELS,
            spatial2,
        )
        .map_err(launch)?;
        let conv3 = buf!(s.conv3);
        k::convolution3x3(
            stream,
            conv2.as_slice(),
            w.conv3_weight.view(),
            conv3.as_mut_slice(),
            chunks,
            CONV_CHANNELS,
            32,
            25,
        )
        .map_err(launch)?;
        let spatial3 = 16 * CONV_TIME;
        k::bias_gelu(
            stream,
            conv3.as_mut_slice(),
            w.conv3_bias.view(),
            chunks * CONV_CHANNELS * spatial3,
            CONV_CHANNELS,
            spatial3,
        )
        .map_err(launch)?;
        let layout = buf!(s.layout);
        k::conv_output_layout(stream, conv3.as_slice(), layout.as_mut_slice(), chunks)
            .map_err(launch)?;
        let projected = buf!(s.projected);
        let rows = chunks * CONV_TIME;
        self.projector
            .project(
                stream,
                w.conv_out.view(),
                layout.slice(0..rows * CONV_FLATTENED),
                projected.slice_mut(0..rows * AUDIO_HIDDEN),
                rows,
                AUDIO_HIDDEN,
                CONV_FLATTENED,
            )
            .map_err(launch)?;
        let compact = buf!(s.compact);
        k::add_position_and_compact(
            stream,
            projected.as_slice(),
            compact.as_mut_slice(),
            frames,
            chunks,
        )
        .map_err(launch)?;
        let tail = frames - (chunks - 1) * CHUNK_FRAMES;
        Ok((chunks - 1) * CONV_TIME + tail.div_ceil(8))
    }

    /// Runs the whole encoder. The result, f32 `[tokens, 2048]`, stays valid
    /// until the next call; `layer0` receives the hidden state after the
    /// first layer when given (a parity boundary of the C++ tests).
    pub fn encode(
        &mut self,
        stream: &Stream,
        w: &EncoderWeights,
        log_mel: &[f32],
        frames: usize,
        checkpoint: Checkpoint<'_>,
        mut trace: Option<&mut EncoderTrace>,
    ) -> Result<(DeviceSlice<'_, f32>, usize), String> {
        checkpoint()?;
        let tokens = self.frontend(stream, w, log_mel, frames)?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.frontend = download(
                stream,
                buf!(self.scratch.compact).as_slice(),
                tokens * AUDIO_HIDDEN,
            )?;
        }
        self.ensure_tokens(stream, tokens)?;
        let hidden_len = tokens * AUDIO_HIDDEN;
        let s = &mut self.scratch;
        let (hidden, compact) = (buf!(s.hidden), buf!(s.compact));
        stream
            .copy(
                compact.slice(0..hidden_len),
                hidden.slice_mut(0..hidden_len),
            )
            .map_err(|e| format!("Qwen3-ASR frontend activations: {e}"))?;
        let normalized = buf!(s.normalized);
        let (q, kk, v) = (buf!(s.q), buf!(s.k), buf!(s.v));
        let (attention, attention_bf16) = (buf!(s.attention), buf!(s.attention_bf16));
        let projection = buf!(s.projection);
        let (ffn, ffn_bf16) = (buf!(s.ffn), buf!(s.ffn_bf16));
        let p = &mut self.projector;
        let (h, f) = (AUDIO_HIDDEN, AUDIO_FFN);

        for (index, layer) in w.layers.iter().enumerate() {
            checkpoint()?;
            k::layer_norm_bf16(
                stream,
                hidden.as_slice(),
                layer.attention_norm_weight.view(),
                layer.attention_norm_bias.view(),
                normalized.as_mut_slice(),
                tokens,
                h,
                LAYER_NORM_EPSILON,
            )
            .map_err(launch)?;
            for (weight, out) in [
                (&layer.q_weight, &mut *q),
                (&layer.k_weight, &mut *kk),
                (&layer.v_weight, &mut *v),
            ] {
                p.project(
                    stream,
                    weight.view(),
                    normalized.slice(0..tokens * h),
                    out.slice_mut(0..tokens * h),
                    tokens,
                    h,
                    h,
                )
                .map_err(launch)?;
            }
            for (out, bias) in [
                (&mut *q, &layer.q_bias),
                (&mut *kk, &layer.k_bias),
                (&mut *v, &layer.v_bias),
            ] {
                k::bias_round_bf16(stream, out.as_mut_slice(), bias.view(), tokens, h)
                    .map_err(launch)?;
            }
            k::segmented_self_attention(
                stream,
                q.as_slice(),
                kk.as_slice(),
                v.as_slice(),
                attention.as_mut_slice(),
                attention_bf16.as_mut_slice(),
                tokens,
                tokens,
                HEADS,
                HEAD_DIM,
            )
            .map_err(launch)?;
            p.project(
                stream,
                layer.out_weight.view(),
                attention_bf16.slice(0..tokens * h),
                projection.slice_mut(0..tokens * h),
                tokens,
                h,
                h,
            )
            .map_err(launch)?;
            k::bias_round_bf16(
                stream,
                projection.as_mut_slice(),
                layer.out_bias.view(),
                tokens,
                h,
            )
            .map_err(launch)?;
            k::residual_add_bf16(
                stream,
                hidden.as_mut_slice(),
                projection.as_slice(),
                hidden_len,
            )
            .map_err(launch)?;

            k::layer_norm_bf16(
                stream,
                hidden.as_slice(),
                layer.final_norm_weight.view(),
                layer.final_norm_bias.view(),
                normalized.as_mut_slice(),
                tokens,
                h,
                LAYER_NORM_EPSILON,
            )
            .map_err(launch)?;
            p.project(
                stream,
                layer.fc1_weight.view(),
                normalized.slice(0..tokens * h),
                ffn.slice_mut(0..tokens * f),
                tokens,
                f,
                h,
            )
            .map_err(launch)?;
            k::bias_gelu_bf16(
                stream,
                ffn.as_slice(),
                layer.fc1_bias.view(),
                ffn_bf16.as_mut_slice(),
                tokens,
                f,
            )
            .map_err(launch)?;
            p.project(
                stream,
                layer.fc2_weight.view(),
                ffn_bf16.slice(0..tokens * f),
                projection.slice_mut(0..tokens * h),
                tokens,
                h,
                f,
            )
            .map_err(launch)?;
            k::bias_round_bf16(
                stream,
                projection.as_mut_slice(),
                layer.fc2_bias.view(),
                tokens,
                h,
            )
            .map_err(launch)?;
            k::residual_add_bf16(
                stream,
                hidden.as_mut_slice(),
                projection.as_slice(),
                hidden_len,
            )
            .map_err(launch)?;
            if index == 0
                && let Some(trace) = trace.as_deref_mut()
            {
                trace.layer0 = download(stream, hidden.as_slice(), hidden_len)?;
            }
        }

        k::layer_norm_bf16(
            stream,
            hidden.as_slice(),
            w.ln_post_weight.view(),
            w.ln_post_bias.view(),
            normalized.as_mut_slice(),
            tokens,
            h,
            LAYER_NORM_EPSILON,
        )
        .map_err(launch)?;
        p.project(
            stream,
            w.proj1_weight.view(),
            normalized.slice(0..tokens * h),
            projection.slice_mut(0..tokens * h),
            tokens,
            h,
            h,
        )
        .map_err(launch)?;
        k::bias_gelu_bf16(
            stream,
            projection.as_slice(),
            w.proj1_bias.view(),
            normalized.as_mut_slice(),
            tokens,
            h,
        )
        .map_err(launch)?;
        let output_len = tokens * AUDIO_OUTPUT;
        p.project(
            stream,
            w.proj2_weight.view(),
            normalized.slice(0..tokens * h),
            projection.slice_mut(0..output_len),
            tokens,
            AUDIO_OUTPUT,
            h,
        )
        .map_err(launch)?;
        k::bias_round_bf16(
            stream,
            projection.as_mut_slice(),
            w.proj2_bias.view(),
            tokens,
            AUDIO_OUTPUT,
        )
        .map_err(launch)?;

        // Any nonzero value means finite; the kernel clears it otherwise.
        stream
            .fill_bytes(self.finite.as_mut_slice(), 0xFF)
            .map_err(|e| format!("Qwen3-ASR finite flag: {e}"))?;
        k::all_finite(
            stream,
            projection.as_slice(),
            output_len,
            self.finite.as_mut_slice(),
        )
        .map_err(launch)?;
        let mut finite = [0u32];
        stream
            .download(self.finite.as_slice(), &mut finite)
            .map_err(|e| format!("Qwen3-ASR audio encoder: {e}"))?;
        if finite[0] == 0 {
            return Err("Qwen3-ASR audio encoder produced non-finite embeddings".into());
        }
        if let Some(trace) = trace {
            trace.output = download(stream, projection.as_slice(), output_len)?;
        }
        Ok((projection.slice(0..output_len), tokens))
    }
}

/// Host copies of the encoder's parity boundaries.
#[derive(Default)]
pub struct EncoderTrace {
    /// After the convolutional frontend and positions, `[tokens, 1024]`.
    pub frontend: Vec<f32>,
    /// After layer 0, `[tokens, 1024]`.
    pub layer0: Vec<f32>,
    /// The output, `[tokens, 2048]`.
    pub output: Vec<f32>,
}

fn download(stream: &Stream, source: DeviceSlice<'_, f32>, len: usize) -> Result<Vec<f32>, String> {
    let mut values = vec![0.0; len];
    stream
        .download(source.range(0..len), &mut values)
        .map_err(|e| format!("Qwen3-ASR trace download: {e}"))?;
    Ok(values)
}
