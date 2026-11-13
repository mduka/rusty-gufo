//! VoxCPM2's autoregressive generator on the GPU (`generator.hpp`,
//! `generator.cpp`): the text-semantic LM, the residual acoustic LM with
//! their KV caches, the local encoder, the local DiT with its CFM sampler
//! and the stop predictor. One sequence at a time.
//!
//! Ported from OpenBMB/VoxCPM (Apache-2.0, Copyright OpenBMB):
//! VoxCPM2Model's generation step (src/voxcpm/model/voxcpm2.py), the local
//! encoder and local DiT (src/voxcpm/modules/locenc, locdit), the scalar
//! quantization layer and UnifiedCFM's Euler sampler. VoxCPM credits
//! CosyVoice (Apache-2.0, Copyright Alibaba Inc) for its flow-matching local
//! DiT.

use std::collections::HashMap;

use gufo_hip::{
    Bf16, DeviceBuffer, DeviceIndices, DeviceSlice, Event, HipError, LaunchError, PinnedBuffer,
    Stream,
};
use gufo_voxcpm_kernels as k;

use crate::config::Config;
use crate::gemm::Bf16Gemm;
use crate::model_files::ModelFiles;
use crate::transformer::{
    AttentionLayout, KvCache, Rope, StackScratch, forward_stack, project, reserve, rope_table,
};
use crate::weights::DeviceWeights;

/// The prompt rows of `VoxCPM2Model._inference`: per row a text token or an
/// audio latent patch (`Prompt`).
#[derive(Debug, Clone, Default)]
pub struct Prompt {
    pub tokens: Vec<i32>,
    pub text_mask: Vec<i32>,
    pub audio_mask: Vec<i32>,
    /// `[rows][patch_size][feat_dim]`; zero on text rows.
    pub feats: Vec<f32>,
}

/// One update of `UnifiedCFM.solve_euler` (`EulerStep`): its time and step,
/// and whether it evaluates the DiT.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EulerStep {
    pub t: f32,
    pub dt: f32,
    pub evaluate: bool,
}

/// `torch.linspace` in float32: the first half counts up from `start`, the
/// second half down from `end` (`Linspace`).
fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    let halfway = steps / 2;
    (0..steps)
        .map(|index| {
            if index < halfway {
                start + step * index as f32
            } else {
                end - step * (steps - index - 1) as f32
            }
        })
        .collect()
}

/// `SinusoidalPosEmb(width)(t)` with scale 1000, in float32 (`Sinusoid`).
fn sinusoid(t: f32, width: usize) -> Vec<f32> {
    let half = width / 2;
    let exponent = (10000.0f64.ln() / (half - 1) as f64) as f32;
    let mut embedding = vec![0.0f32; width];
    let scaled = 1000.0f32 * t;
    for i in 0..half {
        let frequency = (i as f32 * -exponent).exp();
        let angle = scaled * frequency;
        embedding[i] = angle.sin();
        embedding[half + i] = angle.cos();
    }
    embedding
}

/// The Euler steps of `UnifiedCFM.solve_euler` with the sway-sampled
/// schedule, in the reference's float32 arithmetic: CFG-Zero* leaves the
/// first `max(1, int(0.04 * (n + 1)))` steps at zero velocity
/// (`EulerSchedule`).
pub fn euler_schedule(timesteps: usize, sway: f32) -> Vec<EulerStep> {
    let mut span = linspace(1.0, 0.0, timesteps + 1);
    let quarter_turn = (std::f64::consts::PI / 2.0) as f32;
    for t in &mut span {
        *t = *t + sway * ((quarter_turn * *t).cos() - 1.0 + *t);
    }
    let zero_steps = ((span.len() as f64 * 0.04) as usize).max(1);
    let mut steps = Vec::with_capacity(span.len() - 1);
    let mut t = span[0];
    let mut dt = span[0] - span[1];
    for step in 1..span.len() {
        steps.push(EulerStep {
            t,
            dt,
            evaluate: step > zero_steps,
        });
        t -= dt;
        if step < span.len() - 1 {
            dt = t - span[step + 1];
        }
    }
    steps
}

/// Intermediate tensors for tests, copied to the host when tracing
/// (`Trace`).
#[derive(Debug, Clone, Default)]
pub struct Trace {
    /// `[rows][hidden]`, before FSQ.
    pub lm_prefill: Vec<f32>,
    /// `[rows][hidden]`
    pub residual_prefill: Vec<f32>,
    /// Per step `[2 * dit hidden]`.
    pub mu: Vec<Vec<f32>>,
    /// Per step, before FSQ.
    pub lm_step: Vec<Vec<f32>>,
    pub residual_step: Vec<Vec<f32>>,
}

/// Where [`Generator::encode`] reads its latents and writes its embeddings.
#[derive(Clone, Copy)]
enum EncodeTarget {
    /// `prompt_latents` into `prompt_embeds`.
    Prompt,
    /// The patch `x` into `embed`.
    Patch,
}

/// Where [`Generator::fsq`] reads and writes.
#[derive(Clone, Copy)]
enum FsqTarget {
    /// The prefill's LM output (`prefill_c`) into `prefill_a`.
    Prefill,
    /// The step's LM output (`step_b`) into `lm_hidden`.
    Step,
}

fn hip(error: HipError) -> String {
    error.to_string()
}

fn launch(error: LaunchError) -> String {
    error.to_string()
}

fn iota(count: usize) -> Vec<u32> {
    (0..count as u32).collect()
}

/// Uploads `values` into `indices` (below `bound`), reallocating when the
/// buffer is too short or its bound differs (`Upload`).
fn upload_indices(
    stream: &Stream,
    indices: &mut DeviceIndices,
    values: &[u32],
    bound: u32,
) -> Result<(), HipError> {
    if indices.len() < values.len() || indices.bound() != bound {
        *indices = stream.alloc_indices(values.len().max(1), bound.max(1))?;
    }
    stream.upload_indices(values, indices)
}

/// VoxCPM2's generator (`Generator`). `Send`, not `Sync`.
pub struct Generator {
    config: Config,
    weights: DeviceWeights,
    rope: DeviceBuffer<f32>,
    rope_rows: usize,
    base_cache: KvCache,
    residual_cache: KvCache,
    encoder_cache: Option<KvCache>,
    dit_cache: KvCache,
    scratch: StackScratch,
    position: usize,

    // Sequence state.
    /// `[hidden]`, after FSQ.
    lm_hidden: DeviceBuffer<f32>,
    /// `[hidden]`
    residual_hidden: DeviceBuffer<f32>,
    /// `[patch][feat]`
    prefix_cond: DeviceBuffer<f32>,
    /// `[patch][feat]`
    x: DeviceBuffer<f32>,
    /// `[2][patch][feat]`
    velocity: DeviceBuffer<f32>,
    /// `[2 * dit hidden]`
    mu: DeviceBuffer<f32>,
    /// `[patch][dit hidden]`
    cond_rows: DeviceBuffer<f32>,
    /// `[2]`
    stop: DeviceBuffer<f32>,
    /// `delta_time_mlp(sinusoid(0))`
    delta_time: DeviceBuffer<f32>,
    /// Keyed by the bits of `t`.
    time_vectors: HashMap<u32, DeviceBuffer<f32>>,

    // Index arrays.
    dit_positions: DeviceIndices,
    dit_rows: DeviceIndices,
    dit_outputs: DeviceIndices,
    ids: DeviceIndices,
    text_mask: DeviceBuffer<i32>,
    audio_mask: DeviceBuffer<i32>,
    audio_rows: DeviceIndices,
    positions: DeviceIndices,
    cache_rows: DeviceIndices,
    local_positions: DeviceIndices,
    local_rows: DeviceIndices,
    local_firsts: DeviceIndices,
    encoder_rows_count: usize,

    // Working buffers.
    encoder_bf16: DeviceBuffer<Bf16>,
    encoder_a: DeviceBuffer<f32>,
    encoder_b: DeviceBuffer<f32>,
    encoder_c: DeviceBuffer<f32>,
    fsq_bf16: DeviceBuffer<Bf16>,
    fsq_latent: DeviceBuffer<f32>,
    dit_bf16: DeviceBuffer<Bf16>,
    x_rows: DeviceBuffer<f32>,
    dit_hidden: DeviceBuffer<f32>,
    dit_output: DeviceBuffer<f32>,
    prefill_a: DeviceBuffer<f32>,
    prefill_b: DeviceBuffer<f32>,
    prefill_c: DeviceBuffer<f32>,
    prefill_d: DeviceBuffer<f32>,
    prefill_bf16: DeviceBuffer<Bf16>,
    prompt_latents: DeviceBuffer<f32>,
    prompt_embeds: DeviceBuffer<f32>,
    step_bf16: DeviceBuffer<Bf16>,
    step_a: DeviceBuffer<f32>,
    step_b: DeviceBuffer<f32>,
    embed: DeviceBuffer<f32>,
    /// The last two patches' latents, and the pinned staging of each step's
    /// noise and cache position (written only after the previous step's
    /// final synchronization).
    latent_ring: DeviceBuffer<f32>,
    latent_slot: usize,
    latent_event: Event,
    pinned_noise: PinnedBuffer<f32>,
    pinned_position: PinnedBuffer<u32>,
    trace: Option<Trace>,
    gemm: Bf16Gemm,
    stream: Stream,
}

impl Generator {
    /// Uploads the generator's weights and prepares its caches (`Create`).
    pub fn create(config: &Config, files: &ModelFiles) -> Result<Self, String> {
        let stream = Stream::new().map_err(hip)?;
        let gemm = Bf16Gemm::new(&stream).map_err(launch)?;
        let weights = DeviceWeights::upload(&stream, files, config)?;
        let s = &stream;
        let rope_rows = config.max_length.max(64);
        let table = rope_table(&config.rope, rope_rows);
        let mut rope = s.alloc(table.len().max(1)).map_err(hip)?;
        s.upload(&table, rope.slice_mut(0..table.len()))
            .map_err(hip)?;
        let base_cache = KvCache::new(s, &config.lm, config.max_length).map_err(hip)?;
        let residual_cache =
            KvCache::new(s, &config.residual_lm, config.max_length).map_err(hip)?;
        let dit_rows = 2 * (3 + 2 * config.patch_size);
        let dit_cache = KvCache::new(s, &config.dit, dit_rows).map_err(hip)?;

        // The DiT's rows: two sequences of 3 + 2P positions; the outputs are
        // the last P rows of each.
        let length = dit_rows / 2;
        let positions: Vec<u32> = (0..dit_rows).map(|row| (row % length) as u32).collect();
        let mut outputs = Vec::with_capacity(2 * config.patch_size);
        for sequence in 0..2 {
            for i in 0..config.patch_size {
                outputs.push((sequence * length + 3 + config.patch_size + i) as u32);
            }
        }
        let index = |values: &[u32], bound: usize| -> Result<DeviceIndices, HipError> {
            let mut indices = s.alloc_indices(values.len().max(1), bound.max(1) as u32)?;
            s.upload_indices(values, &mut indices)?;
            Ok(indices)
        };
        let dit_positions = index(&positions, length).map_err(hip)?;
        let dit_rows_indices = index(&iota(dit_rows), dit_rows).map_err(hip)?;
        let dit_outputs = index(&outputs, dit_rows).map_err(hip)?;

        let hidden = config.lm.hidden;
        let patch_values = config.patch_size * config.feat_dim;
        let alloc = |count: usize| s.alloc::<f32>(count.max(1)).map_err(hip);
        let indices = || s.alloc_indices(1, 1).map_err(hip);
        let mut generator = Self {
            config: config.clone(),
            rope,
            rope_rows,
            base_cache,
            residual_cache,
            encoder_cache: None,
            dit_cache,
            scratch: StackScratch::new(s).map_err(hip)?,
            position: 0,
            lm_hidden: alloc(hidden)?,
            residual_hidden: alloc(hidden)?,
            prefix_cond: alloc(patch_values)?,
            x: alloc(patch_values)?,
            velocity: alloc(2 * patch_values)?,
            mu: alloc(2 * config.dit.hidden)?,
            cond_rows: alloc(config.patch_size * config.dit.hidden)?,
            stop: alloc(2)?,
            delta_time: alloc(config.dit.hidden)?,
            time_vectors: HashMap::new(),
            dit_positions,
            dit_rows: dit_rows_indices,
            dit_outputs,
            ids: indices()?,
            text_mask: s.alloc(1).map_err(hip)?,
            audio_mask: s.alloc(1).map_err(hip)?,
            audio_rows: indices()?,
            positions: indices()?,
            cache_rows: indices()?,
            local_positions: indices()?,
            local_rows: indices()?,
            local_firsts: indices()?,
            encoder_rows_count: 0,
            encoder_bf16: s.alloc(1).map_err(hip)?,
            encoder_a: alloc(1)?,
            encoder_b: alloc(1)?,
            encoder_c: alloc(1)?,
            fsq_bf16: s.alloc(1).map_err(hip)?,
            fsq_latent: alloc(1)?,
            dit_bf16: s.alloc(1).map_err(hip)?,
            x_rows: alloc(1)?,
            dit_hidden: alloc(1)?,
            dit_output: alloc(1)?,
            prefill_a: alloc(1)?,
            prefill_b: alloc(1)?,
            prefill_c: alloc(1)?,
            prefill_d: alloc(1)?,
            prefill_bf16: s.alloc(1).map_err(hip)?,
            prompt_latents: alloc(1)?,
            prompt_embeds: alloc(1)?,
            step_bf16: s.alloc(1).map_err(hip)?,
            step_a: alloc(1)?,
            step_b: alloc(1)?,
            embed: alloc(1)?,
            latent_ring: alloc(2 * patch_values)?,
            latent_slot: 0,
            latent_event: Event::new().map_err(hip)?,
            pinned_noise: PinnedBuffer::zeroed(patch_values).map_err(hip)?,
            pinned_position: PinnedBuffer::zeroed(1).map_err(hip)?,
            trace: None,
            weights,
            gemm,
            stream,
        };
        generator.compute_delta_time()?;
        Ok(generator)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Prompt rows plus generated patches the caches can hold.
    pub fn capacity(&self) -> usize {
        self.config.max_length
    }

    /// Rows in the caches (the next patch's position).
    pub fn position(&self) -> usize {
        self.position
    }

    /// The last patch's latent `[patch_size][feat_dim]`, valid until the
    /// step after next, ready once [`Generator::latent_event`] completes.
    pub fn latent(&self) -> DeviceSlice<'_, f32> {
        let patch_values = self.config.patch_size * self.config.feat_dim;
        let begin = self.latent_slot * patch_values;
        self.latent_ring.slice(begin..begin + patch_values)
    }

    pub fn latent_event(&self) -> &Event {
        &self.latent_event
    }

    /// Copies intermediate tensors into a [`Trace`] while on (tests).
    pub fn set_trace(&mut self, on: bool) {
        self.trace = on.then(Trace::default);
    }

    /// The trace recorded since [`Generator::set_trace`].
    pub fn trace(&self) -> Option<&Trace> {
        self.trace.as_ref()
    }

    fn download(&self, data: DeviceSlice<'_, f32>) -> Result<Vec<f32>, String> {
        let mut values = vec![0.0f32; data.len()];
        self.stream.download(data, &mut values).map_err(hip)?;
        Ok(values)
    }

    /// `delta_time_mlp(sinusoid(0))`: dt is zero outside mean mode (the
    /// end of `Create`).
    fn compute_delta_time(&mut self) -> Result<(), String> {
        let zero = sinusoid(0.0, self.config.dit.hidden);
        let width = zero.len();
        let s = &self.stream;
        let w = self.weights.weights();
        let mut input = s.alloc::<f32>(width).map_err(hip)?;
        s.upload(&zero, input.as_mut_slice()).map_err(hip)?;
        let mut input_bf16 = s.alloc::<Bf16>(width).map_err(hip)?;
        let mut middle = s.alloc::<f32>(width).map_err(hip)?;
        k::to_bf16(s, input.as_slice(), input_bf16.as_mut_slice(), width, false).map_err(launch)?;
        project(
            &self.weights,
            &w.delta_time_1,
            input_bf16.as_slice(),
            middle.as_mut_slice(),
            1,
            &mut self.gemm,
            s,
        )
        .map_err(launch)?;
        k::to_bf16(s, middle.as_slice(), input_bf16.as_mut_slice(), width, true).map_err(launch)?;
        project(
            &self.weights,
            &w.delta_time_2,
            input_bf16.as_slice(),
            self.delta_time.slice_mut(0..width),
            1,
            &mut self.gemm,
            s,
        )
        .map_err(launch)?;
        s.synchronize().map_err(hip)
    }

    /// `time_mlp(sinusoid(t)) + delta_time_mlp(sinusoid(0))`, computed once
    /// per `t` (`TimeVector`).
    fn ensure_time_vector(&mut self, t: f32) -> Result<(), String> {
        if self.time_vectors.contains_key(&t.to_bits()) {
            return Ok(());
        }
        let width = self.config.dit.hidden;
        let sinusoid = sinusoid(t, width);
        let s = &self.stream;
        let w = self.weights.weights();
        let mut input = s.alloc::<f32>(width).map_err(hip)?;
        let mut input_bf16 = s.alloc::<Bf16>(width).map_err(hip)?;
        let mut middle = s.alloc::<f32>(width).map_err(hip)?;
        let mut vector = s.alloc::<f32>(width).map_err(hip)?;
        s.upload(&sinusoid, input.as_mut_slice()).map_err(hip)?;
        k::to_bf16(s, input.as_slice(), input_bf16.as_mut_slice(), width, false).map_err(launch)?;
        project(
            &self.weights,
            &w.time_1,
            input_bf16.as_slice(),
            middle.as_mut_slice(),
            1,
            &mut self.gemm,
            s,
        )
        .map_err(launch)?;
        k::to_bf16(s, middle.as_slice(), input_bf16.as_mut_slice(), width, true).map_err(launch)?;
        s.copy(self.delta_time.slice(0..width), vector.as_mut_slice())
            .map_err(hip)?;
        self.gemm
            .linear(
                s,
                input_bf16.as_slice(),
                self.weights.matrix(w.time_2.weight),
                vector.as_mut_slice(),
                1,
                width,
                width,
                true,
            )
            .map_err(launch)?;
        if let Some(bias) = w.time_2.bias {
            k::add_bias(
                s,
                vector.as_mut_slice(),
                self.weights.vector(bias),
                1,
                width,
            )
            .map_err(launch)?;
        }
        s.synchronize().map_err(hip)?;
        self.time_vectors.insert(t.to_bits(), vector);
        Ok(())
    }

    /// Scalar quantization of `rows` LM rows (`Fsq`).
    fn fsq(&mut self, target: FsqTarget, rows: usize) -> Result<(), String> {
        let hidden = self.config.lm.hidden;
        let latent = self.config.fsq_latent_dim;
        let s = &self.stream;
        reserve(s, &mut self.fsq_bf16, rows * hidden.max(latent)).map_err(hip)?;
        reserve(s, &mut self.fsq_latent, rows * latent).map_err(hip)?;
        let w = self.weights.weights();
        let (input, output) = match target {
            FsqTarget::Prefill => (
                self.prefill_c.slice(0..rows * hidden),
                self.prefill_a.slice_mut(0..rows * hidden),
            ),
            FsqTarget::Step => (
                self.step_b.slice(0..rows * hidden),
                self.lm_hidden.slice_mut(0..rows * hidden),
            ),
        };
        k::to_bf16(
            s,
            input,
            self.fsq_bf16.slice_mut(0..rows * hidden),
            rows * hidden,
            false,
        )
        .map_err(launch)?;
        project(
            &self.weights,
            &w.fsq_in,
            self.fsq_bf16.slice(0..rows * hidden),
            self.fsq_latent.slice_mut(0..rows * latent),
            rows,
            &mut self.gemm,
            s,
        )
        .map_err(launch)?;
        k::quantize(
            s,
            self.fsq_latent.slice(0..rows * latent),
            self.fsq_bf16.slice_mut(0..rows * latent),
            rows * latent,
            self.config.fsq_scale,
        )
        .map_err(launch)?;
        project(
            &self.weights,
            &w.fsq_out,
            self.fsq_bf16.slice(0..rows * latent),
            output,
            rows,
            &mut self.gemm,
            s,
        )
        .map_err(launch)
    }

    /// The local encoder over `patches` latent patches, projected to the
    /// LM's width (`Encode`).
    fn encode(&mut self, target: EncodeTarget, patches: usize) -> Result<(), String> {
        let patch = self.config.patch_size;
        let width = self.config.encoder.hidden;
        let feat = self.config.feat_dim;
        let hidden = self.config.lm.hidden;
        let rows = patches * (patch + 1);
        let s = &self.stream;
        if self
            .encoder_cache
            .as_ref()
            .is_none_or(|cache| cache.capacity() < rows)
        {
            self.encoder_cache =
                Some(KvCache::new(s, &self.config.encoder, rows.max(64)).map_err(hip)?);
        }
        if self.encoder_rows_count != patches {
            let positions: Vec<u32> = (0..rows).map(|row| (row % (patch + 1)) as u32).collect();
            let firsts: Vec<u32> = (0..patches)
                .map(|index| (index * (patch + 1)) as u32)
                .collect();
            upload_indices(s, &mut self.local_positions, &positions, (patch + 1) as u32)
                .map_err(hip)?;
            upload_indices(s, &mut self.local_rows, &iota(rows), rows as u32).map_err(hip)?;
            upload_indices(s, &mut self.local_firsts, &firsts, rows as u32).map_err(hip)?;
            self.encoder_rows_count = patches;
        }
        (|| -> Result<(), HipError> {
            reserve(s, &mut self.encoder_bf16, rows * width.max(feat))?;
            reserve(s, &mut self.encoder_a, rows * width)?;
            reserve(s, &mut self.encoder_b, rows * width)?;
            reserve(s, &mut self.encoder_c, rows * width)
        })()
        .map_err(hip)?;
        let w = self.weights.weights();
        let latents = match target {
            EncodeTarget::Prompt => self.prompt_latents.slice(0..patches * patch * feat),
            EncodeTarget::Patch => self.x.slice(0..patches * patch * feat),
        };
        k::to_bf16(
            s,
            latents,
            self.encoder_bf16.slice_mut(0..patches * patch * feat),
            patches * patch * feat,
            false,
        )
        .map_err(launch)?;
        project(
            &self.weights,
            &w.encoder_in,
            self.encoder_bf16.slice(0..patches * patch * feat),
            self.encoder_a.slice_mut(0..patches * patch * width),
            patches * patch,
            &mut self.gemm,
            s,
        )
        .map_err(launch)?;
        k::prepend_special(
            s,
            self.encoder_a.slice(0..patches * patch * width),
            self.weights.vector(w.special),
            self.encoder_b.slice_mut(0..rows * width),
            patches,
            patch,
            width,
        )
        .map_err(launch)?;
        let layout = AttentionLayout {
            positions: self.local_positions.slice(0..rows),
            cache_rows: self.local_rows.slice(0..rows),
            rows_per_sequence: patch + 1,
            keys_per_sequence: patch + 1,
            past: 0,
            causal: false,
        };
        let rope = Rope {
            table: self.rope.slice(0..self.rope_rows * self.config.rope.dim),
            rows: self.rope_rows,
        };
        let cache = self
            .encoder_cache
            .as_mut()
            .ok_or("the local encoder has no cache")?;
        forward_stack(
            &self.weights,
            &w.encoder,
            Some(rope),
            &layout,
            cache,
            self.encoder_b.slice_mut(0..rows * width),
            self.encoder_c.slice_mut(0..rows * width),
            rows,
            &mut self.scratch,
            &mut self.gemm,
            s,
        )?;
        k::gather_rows(
            s,
            self.encoder_c.slice(0..rows * width),
            self.local_firsts.slice(0..patches),
            self.encoder_a.slice_mut(0..patches * width),
            patches,
            width,
            rows,
        )
        .map_err(launch)?;
        k::to_bf16(
            s,
            self.encoder_a.slice(0..patches * width),
            self.encoder_bf16.slice_mut(0..patches * width),
            patches * width,
            false,
        )
        .map_err(launch)?;
        let embeds = match target {
            EncodeTarget::Prompt => self.prompt_embeds.slice_mut(0..patches * hidden),
            EncodeTarget::Patch => self.embed.slice_mut(0..patches * hidden),
        };
        project(
            &self.weights,
            &w.enc_to_lm,
            self.encoder_bf16.slice(0..patches * width),
            embeds,
            patches,
            &mut self.gemm,
            s,
        )
        .map_err(launch)
    }

    /// One DiT evaluation of the CFG pair at time `t` on `x`, into
    /// `velocity` (`EvaluateDit`).
    fn evaluate_dit(&mut self, t: f32) -> Result<(), String> {
        let patch = self.config.patch_size;
        let feat = self.config.feat_dim;
        let width = self.config.dit.hidden;
        let rows = 2 * (3 + 2 * patch);
        self.ensure_time_vector(t)?;
        let s = &self.stream;
        (|| -> Result<(), HipError> {
            reserve(s, &mut self.dit_bf16, rows * width)?;
            reserve(s, &mut self.x_rows, patch * width)?;
            reserve(s, &mut self.dit_hidden, rows * width)?;
            reserve(s, &mut self.dit_output, rows * width)
        })()
        .map_err(hip)?;
        let w = self.weights.weights();
        let time = self
            .time_vectors
            .get(&t.to_bits())
            .ok_or("missing DiT time vector")?;
        k::to_bf16(
            s,
            self.x.slice(0..patch * feat),
            self.dit_bf16.slice_mut(0..patch * feat),
            patch * feat,
            false,
        )
        .map_err(launch)?;
        project(
            &self.weights,
            &w.dit_in,
            self.dit_bf16.slice(0..patch * feat),
            self.x_rows.slice_mut(0..patch * width),
            patch,
            &mut self.gemm,
            s,
        )
        .map_err(launch)?;
        k::assemble_dit(
            s,
            self.mu.slice(0..2 * width),
            time.slice(0..width),
            self.cond_rows.slice(0..patch * width),
            self.x_rows.slice(0..patch * width),
            self.dit_hidden.slice_mut(0..rows * width),
            patch,
            width,
        )
        .map_err(launch)?;
        let layout = AttentionLayout {
            positions: self.dit_positions.slice(0..rows),
            cache_rows: self.dit_rows.slice(0..rows),
            rows_per_sequence: rows / 2,
            keys_per_sequence: rows / 2,
            past: 0,
            causal: false,
        };
        let rope = Rope {
            table: self.rope.slice(0..self.rope_rows * self.config.rope.dim),
            rows: self.rope_rows,
        };
        forward_stack(
            &self.weights,
            &w.dit,
            Some(rope),
            &layout,
            &mut self.dit_cache,
            self.dit_hidden.slice_mut(0..rows * width),
            self.dit_output.slice_mut(0..rows * width),
            rows,
            &mut self.scratch,
            &mut self.gemm,
            s,
        )?;
        k::gather_rows(
            s,
            self.dit_output.slice(0..rows * width),
            self.dit_outputs.slice(0..2 * patch),
            self.dit_hidden.slice_mut(0..2 * patch * width),
            2 * patch,
            width,
            rows,
        )
        .map_err(launch)?;
        k::to_bf16(
            s,
            self.dit_hidden.slice(0..2 * patch * width),
            self.dit_bf16.slice_mut(0..2 * patch * width),
            2 * patch * width,
            false,
        )
        .map_err(launch)?;
        project(
            &self.weights,
            &w.dit_out,
            self.dit_bf16.slice(0..2 * patch * width),
            self.velocity.slice_mut(0..2 * patch * feat),
            2 * patch,
            &mut self.gemm,
            s,
        )
        .map_err(launch)
    }

    /// Starts a sequence from `prompt` (at most `capacity() - 1` rows;
    /// `Prefill`).
    pub fn prefill(&mut self, prompt: &Prompt) -> Result<(), String> {
        let rows = prompt.tokens.len();
        let hidden = self.config.lm.hidden;
        let patch_values = self.config.patch_size * self.config.feat_dim;
        if rows == 0
            || rows >= self.capacity()
            || prompt.text_mask.len() != rows
            || prompt.audio_mask.len() != rows
            || prompt.feats.len() != rows * patch_values
        {
            return Err("malformed VoxCPM2 prompt".into());
        }
        let ids = prompt
            .tokens
            .iter()
            .map(|&token| u32::try_from(token).map_err(|_| "negative VoxCPM2 token".to_string()))
            .collect::<Result<Vec<u32>, String>>()?;
        let vocab = self.config.lm.vocab;
        let s = &self.stream;
        upload_indices(s, &mut self.ids, &ids, vocab as u32).map_err(hip)?;
        (|| -> Result<(), HipError> {
            reserve(s, &mut self.text_mask, rows)?;
            s.upload(&prompt.text_mask, self.text_mask.slice_mut(0..rows))?;
            reserve(s, &mut self.audio_mask, rows)?;
            s.upload(&prompt.audio_mask, self.audio_mask.slice_mut(0..rows))?;
            reserve(s, &mut self.prefill_a, rows * hidden)?;
            reserve(s, &mut self.prefill_b, rows * hidden)?;
            reserve(s, &mut self.prefill_c, rows * hidden)?;
            reserve(s, &mut self.prefill_d, rows * hidden)?;
            reserve(s, &mut self.prefill_bf16, rows * 2 * hidden)
        })()
        .map_err(hip)?;
        let w = self.weights.weights();
        // `combined` is prefill_a, `audio_embeds` prefill_b.
        k::embed(
            s,
            self.weights.vector(w.embed),
            self.ids.slice(0..rows),
            self.text_mask.slice(0..rows),
            self.prefill_a.slice_mut(0..rows * hidden),
            rows,
            hidden,
            1.0,
            vocab,
        )
        .map_err(launch)?;
        s.fill_bytes(self.prefill_b.slice_mut(0..rows * hidden), 0)
            .map_err(hip)?;

        let mut audio_rows = Vec::new();
        let mut latents = Vec::new();
        for row in 0..rows {
            if prompt.audio_mask[row] == 0 {
                continue;
            }
            audio_rows.push(row as u32);
            latents.extend_from_slice(&prompt.feats[row * patch_values..(row + 1) * patch_values]);
        }
        if !audio_rows.is_empty() {
            let count = audio_rows.len();
            upload_indices(s, &mut self.audio_rows, &audio_rows, rows as u32).map_err(hip)?;
            (|| -> Result<(), HipError> {
                reserve(s, &mut self.prompt_latents, latents.len())?;
                reserve(s, &mut self.prompt_embeds, count * hidden)?;
                s.upload(&latents, self.prompt_latents.slice_mut(0..latents.len()))
            })()
            .map_err(hip)?;
            self.encode(EncodeTarget::Prompt, count)?;
            let s = &self.stream;
            k::scatter_rows(
                s,
                self.prompt_embeds.slice(0..count * hidden),
                self.audio_rows.slice(0..count),
                self.prefill_a.slice_mut(0..rows * hidden),
                count,
                hidden,
                rows,
            )
            .map_err(launch)?;
            k::scatter_rows(
                s,
                self.prompt_embeds.slice(0..count * hidden),
                self.audio_rows.slice(0..count),
                self.prefill_b.slice_mut(0..rows * hidden),
                count,
                hidden,
                rows,
            )
            .map_err(launch)?;
            s.synchronize().map_err(hip)?;
        }

        let s = &self.stream;
        let capacity = self.capacity() as u32;
        let positions = iota(rows);
        upload_indices(s, &mut self.positions, &positions, capacity).map_err(hip)?;
        upload_indices(s, &mut self.cache_rows, &positions, capacity).map_err(hip)?;
        let w = self.weights.weights();
        let rope = Rope {
            table: self.rope.slice(0..self.rope_rows * self.config.rope.dim),
            rows: self.rope_rows,
        };
        let layout = AttentionLayout {
            positions: self.positions.slice(0..rows),
            cache_rows: self.cache_rows.slice(0..rows),
            rows_per_sequence: rows,
            keys_per_sequence: 0,
            past: 0,
            causal: true,
        };
        // The LM's output is prefill_c.
        forward_stack(
            &self.weights,
            &w.base_lm,
            Some(rope),
            &layout,
            &mut self.base_cache,
            self.prefill_a.slice_mut(0..rows * hidden),
            self.prefill_c.slice_mut(0..rows * hidden),
            rows,
            &mut self.scratch,
            &mut self.gemm,
            s,
        )?;
        if self.trace.is_some() {
            let values = self.download(self.prefill_c.slice(0..rows * hidden))?;
            if let Some(trace) = &mut self.trace {
                trace.lm_prefill = values;
            }
        }
        // The combined rows are spent: prefill_a takes the quantized rows.
        self.fsq(FsqTarget::Prefill, rows)?;
        let s = &self.stream;
        let w = self.weights.weights();
        // `selected` is prefill_d.
        k::select_rows(
            s,
            self.prefill_a.slice(0..rows * hidden),
            self.prefill_c.slice(0..rows * hidden),
            self.audio_mask.slice(0..rows),
            self.prefill_d.slice_mut(0..rows * hidden),
            rows,
            hidden,
        )
        .map_err(launch)?;
        s.copy(
            self.prefill_d.slice((rows - 1) * hidden..rows * hidden),
            self.lm_hidden.slice_mut(0..hidden),
        )
        .map_err(hip)?;

        k::concat_rows(
            s,
            self.prefill_d.slice(0..rows * hidden),
            None,
            hidden,
            self.prefill_b.slice(0..rows * hidden),
            None,
            hidden,
            self.prefill_bf16.slice_mut(0..rows * 2 * hidden),
            rows,
        )
        .map_err(launch)?;
        // The residual LM's input is prefill_a, its output prefill_c.
        project(
            &self.weights,
            &w.fusion,
            self.prefill_bf16.slice(0..rows * 2 * hidden),
            self.prefill_a.slice_mut(0..rows * hidden),
            rows,
            &mut self.gemm,
            s,
        )
        .map_err(launch)?;
        let layout = AttentionLayout {
            positions: self.positions.slice(0..rows),
            cache_rows: self.cache_rows.slice(0..rows),
            rows_per_sequence: rows,
            keys_per_sequence: 0,
            past: 0,
            causal: true,
        };
        forward_stack(
            &self.weights,
            &w.residual_lm,
            None,
            &layout,
            &mut self.residual_cache,
            self.prefill_a.slice_mut(0..rows * hidden),
            self.prefill_c.slice_mut(0..rows * hidden),
            rows,
            &mut self.scratch,
            &mut self.gemm,
            s,
        )?;
        if self.trace.is_some() {
            let values = self.download(self.prefill_c.slice(0..rows * hidden))?;
            if let Some(trace) = &mut self.trace {
                trace.residual_prefill = values;
            }
        }
        let s = &self.stream;
        s.copy(
            self.prefill_c.slice((rows - 1) * hidden..rows * hidden),
            self.residual_hidden.slice_mut(0..hidden),
        )
        .map_err(hip)?;
        s.upload(
            &prompt.feats[(rows - 1) * patch_values..rows * patch_values],
            self.prefix_cond.slice_mut(0..patch_values),
        )
        .map_err(hip)?;
        s.synchronize().map_err(hip)?;
        self.position = rows;
        Ok(())
    }

    /// Generates the next patch from `noise` (`[patch_size][feat_dim]`)
    /// with `timesteps` Euler steps and guidance `cfg`: writes the stop
    /// logits of the state before it (and its latent when `latent` is
    /// given), then advances both LMs past it. The latent also stays on the
    /// device: [`Generator::latent`] (`Step`).
    pub fn step(
        &mut self,
        noise: &[f32],
        timesteps: usize,
        cfg: f32,
        latent: Option<&mut [f32]>,
        stop_logits: &mut [f32; 2],
    ) -> Result<(), String> {
        if self.position == 0 || self.position >= self.capacity() {
            return Err("VoxCPM2 generator has no room for a patch".into());
        }
        let hidden = self.config.lm.hidden;
        let dit = self.config.dit.hidden;
        let patch = self.config.patch_size;
        let patch_values = patch * self.config.feat_dim;
        if noise.len() != patch_values {
            return Err("VoxCPM2 noise is not one patch".into());
        }
        let s = &self.stream;
        (|| -> Result<(), HipError> {
            reserve(s, &mut self.step_bf16, 2 * hidden)?;
            reserve(s, &mut self.step_a, hidden)?;
            reserve(s, &mut self.step_b, hidden)?;
            reserve(s, &mut self.embed, hidden)
        })()
        .map_err(hip)?;
        let w = self.weights.weights();

        // DiT condition and the stop predictor read the state before the
        // patch.
        k::to_bf16(
            s,
            self.lm_hidden.slice(0..hidden),
            self.step_bf16.slice_mut(0..hidden),
            hidden,
            false,
        )
        .map_err(launch)?;
        project(
            &self.weights,
            &w.lm_to_dit,
            self.step_bf16.slice(0..hidden),
            self.mu.slice_mut(0..dit),
            1,
            &mut self.gemm,
            s,
        )
        .map_err(launch)?;
        project(
            &self.weights,
            &w.stop_proj,
            self.step_bf16.slice(0..hidden),
            self.step_a.slice_mut(0..hidden),
            1,
            &mut self.gemm,
            s,
        )
        .map_err(launch)?;
        k::to_bf16(
            s,
            self.step_a.slice(0..hidden),
            self.step_bf16.slice_mut(0..hidden),
            hidden,
            true,
        )
        .map_err(launch)?;
        project(
            &self.weights,
            &w.stop_head,
            self.step_bf16.slice(0..hidden),
            self.stop.slice_mut(0..2),
            1,
            &mut self.gemm,
            s,
        )
        .map_err(launch)?;
        k::to_bf16(
            s,
            self.residual_hidden.slice(0..hidden),
            self.step_bf16.slice_mut(0..hidden),
            hidden,
            false,
        )
        .map_err(launch)?;
        project(
            &self.weights,
            &w.res_to_dit,
            self.step_bf16.slice(0..hidden),
            self.mu.slice_mut(dit..2 * dit),
            1,
            &mut self.gemm,
            s,
        )
        .map_err(launch)?;
        if self.trace.is_some() {
            let values = self.download(self.mu.slice(0..2 * dit))?;
            if let Some(trace) = &mut self.trace {
                trace.mu.push(values);
            }
        }
        let s = &self.stream;
        k::to_bf16(
            s,
            self.prefix_cond.slice(0..patch_values),
            self.step_bf16.slice_mut(0..patch_values),
            patch_values,
            false,
        )
        .map_err(launch)?;
        project(
            &self.weights,
            &w.dit_cond,
            self.step_bf16.slice(0..patch_values),
            self.cond_rows.slice_mut(0..patch * dit),
            patch,
            &mut self.gemm,
            s,
        )
        .map_err(launch)?;

        self.pinned_noise
            .host_mut()
            .map_err(hip)?
            .copy_from_slice(noise);
        s.upload_async(&mut self.pinned_noise, self.x.slice_mut(0..patch_values))
            .map_err(hip)?;
        for step in euler_schedule(timesteps, 1.0) {
            if !step.evaluate {
                continue; // zero velocity
            }
            self.evaluate_dit(step.t)?;
            k::guided_euler(
                &self.stream,
                self.velocity.slice(0..2 * patch_values),
                self.x.slice_mut(0..patch_values),
                patch_values,
                cfg,
                step.dt,
            )
            .map_err(launch)?;
        }
        let s = &self.stream;
        let slot = self.position % 2;
        s.copy(
            self.x.slice(0..patch_values),
            self.latent_ring
                .slice_mut(slot * patch_values..(slot + 1) * patch_values),
        )
        .map_err(hip)?;
        s.record(&self.latent_event).map_err(hip)?;
        self.latent_slot = slot;
        s.copy(
            self.x.slice(0..patch_values),
            self.prefix_cond.slice_mut(0..patch_values),
        )
        .map_err(hip)?;

        // The patch's embedding advances both LMs.
        self.encode(EncodeTarget::Patch, 1)?;
        let s = &self.stream;
        // One row at the next position: its rotary row and cache row
        // coincide.
        self.pinned_position.host_mut().map_err(hip)?[0] = self.position as u32;
        s.upload_indices_async(&mut self.pinned_position, 1, &mut self.positions)
            .map_err(hip)?;
        s.copy(
            self.embed.slice(0..hidden),
            self.step_a.slice_mut(0..hidden),
        )
        .map_err(hip)?;
        let w = self.weights.weights();
        let rope = Rope {
            table: self.rope.slice(0..self.rope_rows * self.config.rope.dim),
            rows: self.rope_rows,
        };
        let layout = AttentionLayout {
            positions: self.positions.slice(0..1),
            cache_rows: self.positions.slice(0..1),
            rows_per_sequence: 1,
            keys_per_sequence: 0,
            past: self.position,
            causal: true,
        };
        forward_stack(
            &self.weights,
            &w.base_lm,
            Some(rope),
            &layout,
            &mut self.base_cache,
            self.step_a.slice_mut(0..hidden),
            self.step_b.slice_mut(0..hidden),
            1,
            &mut self.scratch,
            &mut self.gemm,
            s,
        )?;
        if self.trace.is_some() {
            let values = self.download(self.step_b.slice(0..hidden))?;
            if let Some(trace) = &mut self.trace {
                trace.lm_step.push(values);
            }
        }
        self.fsq(FsqTarget::Step, 1)?;
        let s = &self.stream;
        let w = self.weights.weights();
        k::concat_rows(
            s,
            self.lm_hidden.slice(0..hidden),
            None,
            hidden,
            self.embed.slice(0..hidden),
            None,
            hidden,
            self.step_bf16.slice_mut(0..2 * hidden),
            1,
        )
        .map_err(launch)?;
        project(
            &self.weights,
            &w.fusion,
            self.step_bf16.slice(0..2 * hidden),
            self.step_a.slice_mut(0..hidden),
            1,
            &mut self.gemm,
            s,
        )
        .map_err(launch)?;
        let layout = AttentionLayout {
            positions: self.positions.slice(0..1),
            cache_rows: self.positions.slice(0..1),
            rows_per_sequence: 1,
            keys_per_sequence: 0,
            past: self.position,
            causal: true,
        };
        forward_stack(
            &self.weights,
            &w.residual_lm,
            None,
            &layout,
            &mut self.residual_cache,
            self.step_a.slice_mut(0..hidden),
            self.residual_hidden.slice_mut(0..hidden),
            1,
            &mut self.scratch,
            &mut self.gemm,
            s,
        )?;
        if self.trace.is_some() {
            let values = self.download(self.residual_hidden.slice(0..hidden))?;
            if let Some(trace) = &mut self.trace {
                trace.residual_step.push(values);
            }
        }
        // The stop logits and the latent were final before the encoder ran;
        // their downloads end the step like the C++ synchronization.
        let s = &self.stream;
        s.download(self.stop.slice(0..2), stop_logits)
            .map_err(hip)?;
        if let Some(latent) = latent {
            s.download(self.x.slice(0..patch_values), latent)
                .map_err(hip)?;
        }
        self.position += 1;
        Ok(())
    }

    /// The DiT's CFG pair of velocities `[2][patch_size][feat_dim]` for one
    /// evaluation at time `t` (tests; `Velocity`).
    pub fn velocity(
        &mut self,
        mu: &[f32],
        cond: &[f32],
        x: &[f32],
        t: f32,
        velocity: &mut [f32],
    ) -> Result<(), String> {
        let patch = self.config.patch_size;
        let patch_values = patch * self.config.feat_dim;
        let dit = self.config.dit.hidden;
        let s = &self.stream;
        (|| -> Result<(), HipError> {
            s.upload(&mu[..2 * dit], self.mu.slice_mut(0..2 * dit))?;
            s.upload(
                &cond[..patch_values],
                self.prefix_cond.slice_mut(0..patch_values),
            )?;
            s.upload(&x[..patch_values], self.x.slice_mut(0..patch_values))?;
            reserve(s, &mut self.step_bf16, patch_values)
        })()
        .map_err(hip)?;
        let w = self.weights.weights();
        k::to_bf16(
            s,
            self.prefix_cond.slice(0..patch_values),
            self.step_bf16.slice_mut(0..patch_values),
            patch_values,
            false,
        )
        .map_err(launch)?;
        project(
            &self.weights,
            &w.dit_cond,
            self.step_bf16.slice(0..patch_values),
            self.cond_rows.slice_mut(0..patch * dit),
            patch,
            &mut self.gemm,
            s,
        )
        .map_err(launch)?;
        self.evaluate_dit(t)?;
        self.stream
            .download(
                self.velocity.slice(0..2 * patch_values),
                &mut velocity[..2 * patch_values],
            )
            .map_err(hip)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_meets_both_ends() {
        let values = linspace(1.0, 0.0, 11);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[10], 0.0);
        assert_eq!(values.len(), 11);
        assert_eq!(linspace(0.5, 0.0, 1), vec![0.5]);
    }

    #[test]
    fn euler_schedule_skips_the_first_step() {
        let steps = euler_schedule(10, 1.0);
        assert_eq!(steps.len(), 10);
        // max(1, int(11 * 0.04)) = 1 zero-velocity step.
        assert!(!steps[0].evaluate);
        assert!(steps[1..].iter().all(|step| step.evaluate));
        assert_eq!(steps[0].t, 1.0);
        let end = steps.last().map(|step| step.t - step.dt).unwrap_or(1.0);
        assert!(end.abs() < 1e-6, "ends at {end}");
    }
}
