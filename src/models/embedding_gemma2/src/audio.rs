//! The audio tower (`hip/audio_runtime.hip`, `AudioRuntime`): float32
//! throughout, with the checkpoint's BF16 weights widened once at load. The
//! text encoder owns it and runs it on its stream inside its device batches.

use std::collections::HashMap;
use std::path::Path;

use gufo_embedding_gemma2_kernels as k;
use gufo_hip::blas::{Blas, Op};
use gufo_hip::{DeviceBuffer, DeviceSlice, DeviceSliceMut, LaunchError, Stream};

use crate::audio_processor::AudioFeatures;
use crate::loader::{Towers, expected_audio_tensors, load_model_directory};

const MAX_FRAMES: usize = 2999;
const MAX_TOKENS: usize = MAX_FRAMES.div_ceil(4);

fn hip(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn launch(error: LaunchError) -> String {
    error.to_string()
}

/// The widened weights by tensor name, and the clipped linears' bounds.
struct Weights {
    tensors: HashMap<String, DeviceBuffer<f32>>,
    scalars: HashMap<String, f32>,
}

impl Weights {
    fn get(&self, name: &str) -> Result<DeviceSlice<'_, f32>, String> {
        self.tensors
            .get(name)
            .map(DeviceBuffer::as_slice)
            .ok_or_else(|| format!("embedding_gemma2: missing audio weight {name}"))
    }

    fn scalar(&self, name: &str) -> Result<f32, String> {
        self.scalars
            .get(name)
            .copied()
            .ok_or_else(|| format!("embedding_gemma2: missing audio scalar {name}"))
    }
}

/// The stage buffers the layers pass between them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    State,
    Normed,
    Temp,
    Update,
    Q,
    K,
    V,
    Context,
    Projected,
}

/// The scratch of one clip, sized for the longest one.
struct Buffers {
    input: DeviceBuffer<f32>,
    conv0: DeviceBuffer<f32>,
    conv1: DeviceBuffer<f32>,
    state: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    temp: DeviceBuffer<f32>,
    update: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    context: DeviceBuffer<f32>,
    relative: DeviceBuffer<f32>,
    positions: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    features: DeviceBuffer<f32>,
}

impl Buffers {
    fn slots(&mut self) -> [(Slot, &mut DeviceBuffer<f32>); 9] {
        [
            (Slot::State, &mut self.state),
            (Slot::Normed, &mut self.normed),
            (Slot::Temp, &mut self.temp),
            (Slot::Update, &mut self.update),
            (Slot::Q, &mut self.q),
            (Slot::K, &mut self.k),
            (Slot::V, &mut self.v),
            (Slot::Context, &mut self.context),
            (Slot::Projected, &mut self.projected),
        ]
    }

    fn get_mut(&mut self, slot: Slot) -> DeviceSliceMut<'_, f32> {
        self.slots()
            .into_iter()
            .find(|(s, _)| *s == slot)
            .map(|(_, buffer)| buffer.as_mut_slice())
            .expect("audio stage slot")
    }

    /// Distinct buffers `input` (read) and `output` (written).
    fn pair(
        &mut self,
        input: Slot,
        output: Slot,
    ) -> (DeviceSlice<'_, f32>, DeviceSliceMut<'_, f32>) {
        assert_ne!(input, output, "audio stage buffers alias");
        let mut read = None;
        let mut write = None;
        for (slot, buffer) in self.slots() {
            if slot == input {
                read = Some(buffer.as_slice());
            } else if slot == output {
                write = Some(buffer.as_mut_slice());
            }
        }
        (read.expect("input slot"), write.expect("output slot"))
    }
}

pub struct AudioRuntime {
    blas: Blas,
    weights: Weights,
    b: Buffers,
}

/// `y[n, m] = x[n, width] · W[m, width]^T` (`Gemm`).
#[allow(clippy::too_many_arguments)]
fn gemm(
    blas: &Blas,
    stream: &Stream,
    weight: DeviceSlice<'_, f32>,
    x: DeviceSlice<'_, f32>,
    y: DeviceSliceMut<'_, f32>,
    n: usize,
    m: usize,
    width: usize,
) -> Result<(), String> {
    blas.sgemm(
        stream,
        Op::T,
        Op::N,
        m,
        n,
        width,
        1.0,
        weight,
        width,
        x,
        width,
        y,
        m,
    )
    .map_err(|e| format!("embedding_gemma2: audio hipBLAS failed: {e}"))
}

impl AudioRuntime {
    /// Loads and widens the audio tower's weights onto `stream`.
    pub fn new(model_dir: &Path, stream: &Stream) -> Result<Self, String> {
        let loaded = load_model_directory(
            model_dir,
            Towers {
                vision: false,
                audio: true,
            },
        )?;
        let blas = Blas::with_default_atomics(stream)
            .map_err(|e| format!("embedding_gemma2: audio hipBLAS failed: {e}"))?;
        let mut weights = Weights {
            tensors: HashMap::new(),
            scalars: HashMap::new(),
        };
        for expected in expected_audio_tensors() {
            let tensor = loaded
                .store
                .find(&expected.name)
                .ok_or_else(|| format!("embedding_gemma2: missing tensor {}", expected.name))?;
            let values: Vec<f32> = loaded
                .store
                .bytes(tensor)
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&bits| f32::from_bits(u32::from(u16::from_le_bytes(bits)) << 16))
                .collect();
            if expected.shape.is_empty() {
                weights.scalars.insert(expected.name, values[0]);
            } else {
                let mut buffer = stream.alloc_persistent::<f32>(values.len()).map_err(hip)?;
                stream.upload(&values, buffer.as_mut_slice()).map_err(hip)?;
                weights.tensors.insert(expected.name, buffer);
            }
        }
        let alloc = |count: usize| stream.alloc_persistent::<f32>(count).map_err(hip);
        let mut b = Buffers {
            input: alloc(MAX_FRAMES * 128)?,
            conv0: alloc(1500 * 64 * 128)?,
            conv1: alloc(MAX_TOKENS * 32 * 32)?,
            state: alloc(MAX_TOKENS * 1024)?,
            normed: alloc(MAX_TOKENS * 4096)?,
            temp: alloc(MAX_TOKENS * 4096)?,
            update: alloc(MAX_TOKENS * 2048)?,
            q: alloc(MAX_TOKENS * 1024)?,
            k: alloc(MAX_TOKENS * 1024)?,
            v: alloc(MAX_TOKENS * 1024)?,
            context: alloc(MAX_TOKENS * 1024)?,
            relative: alloc(13 * 1024)?,
            positions: alloc(13 * 1024)?,
            projected: alloc(MAX_TOKENS * 1536)?,
            features: alloc(MAX_TOKENS * 512)?,
        };
        let mut positions = vec![0.0f32; 13 * 1024];
        let step = (10000.0f64.ln() / 511.0) as f32;
        for t in 0..13usize {
            for d in 0..512usize {
                // Official inv_timescales and multiplication are float32.
                let inv = (-(d as f32) * step).exp();
                let angle = (12 - t as i32) as f32 * inv;
                positions[t * 1024 + d] = angle.sin();
                positions[t * 1024 + d + 512] = angle.cos();
            }
        }
        stream
            .upload(&positions, b.positions.as_mut_slice())
            .map_err(hip)?;
        Ok(Self { blas, weights, b })
    }

    /// A clipped linear (`Linear`): the input clipped into `normed`, the
    /// product into `y`, and `y` clipped in place.
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &mut self,
        stream: &Stream,
        name: &str,
        x: Slot,
        y: Slot,
        n: usize,
        m: usize,
        width: usize,
    ) -> Result<(), String> {
        let w = &self.weights;
        let (input, normed) = self.b.pair(x, Slot::Normed);
        k::audio_clip(
            stream,
            input,
            normed,
            (n * width) as i32,
            w.scalar(&format!("{name}.input_min"))?,
            w.scalar(&format!("{name}.input_max"))?,
        )
        .map_err(launch)?;
        let (normed, mut output) = self.b.pair(Slot::Normed, y);
        gemm(
            &self.blas,
            stream,
            w.get(&format!("{name}.linear.weight"))?,
            normed,
            output.reborrow(),
            n,
            m,
            width,
        )?;
        k::audio_clip_in_place(
            stream,
            output,
            (n * m) as i32,
            w.scalar(&format!("{name}.output_min"))?,
            w.scalar(&format!("{name}.output_max"))?,
        )
        .map_err(launch)
    }

    /// An RMS norm (`Normalize`, unweighted for an empty `weight`), in place
    /// when `x` and `y` are one buffer.
    #[allow(clippy::too_many_arguments)]
    fn normalize(
        &mut self,
        stream: &Stream,
        x: Slot,
        weight: &str,
        y: Slot,
        n: usize,
        width: usize,
        clipped: bool,
    ) -> Result<(), String> {
        let w = &self.weights;
        let weight = if weight.is_empty() {
            None
        } else {
            Some(w.get(weight)?)
        };
        if x == y {
            k::audio_norm_in_place(
                stream,
                self.b.get_mut(x),
                weight,
                n as i32,
                width as i32,
                false,
                clipped,
            )
        } else {
            let (input, output) = self.b.pair(x, y);
            k::audio_norm(
                stream,
                input,
                weight,
                output,
                n as i32,
                width as i32,
                false,
                clipped,
            )
        }
        .map_err(launch)
    }

    /// `state += x * scale` (`Add`).
    fn add(&mut self, stream: &Stream, x: Slot, n: usize, scale: f32) -> Result<(), String> {
        let (update, state) = self.b.pair(x, Slot::State);
        k::audio_residual(stream, state, update, (n * 1024) as i32, scale).map_err(launch)
    }

    fn activate(&mut self, stream: &Stream, x: Slot, count: usize) -> Result<(), String> {
        k::audio_activate(stream, self.b.get_mut(x), count as i32).map_err(launch)
    }

    /// `FeedForward`: a half-step residual feed-forward block.
    fn feed_forward(&mut self, stream: &Stream, p: &str, n: usize) -> Result<(), String> {
        self.normalize(
            stream,
            Slot::State,
            &format!("{p}pre_layer_norm.weight"),
            Slot::Update,
            n,
            1024,
            true,
        )?;
        self.linear(
            stream,
            &format!("{p}ffw_layer_1"),
            Slot::Update,
            Slot::Temp,
            n,
            4096,
            1024,
        )?;
        self.activate(stream, Slot::Temp, n * 4096)?;
        self.linear(
            stream,
            &format!("{p}ffw_layer_2"),
            Slot::Temp,
            Slot::Update,
            n,
            1024,
            4096,
        )?;
        self.normalize(
            stream,
            Slot::Update,
            &format!("{p}post_layer_norm.weight"),
            Slot::Update,
            n,
            1024,
            true,
        )?;
        self.add(stream, Slot::Update, n, 0.5)
    }

    /// `Run`: the `[valid tokens, 512]` features of one clip, valid until the
    /// next run. `is_cancelled` is polled after each layer, once the device
    /// has finished it.
    pub fn run(
        &mut self,
        stream: &Stream,
        features: &AudioFeatures,
        is_cancelled: Option<&(dyn Fn() -> bool + Send + Sync)>,
    ) -> Result<DeviceSlice<'_, f32>, String> {
        let frames = features.mask.len();
        let valid_frames = features.mask.iter().filter(|&&valid| valid == 1).count();
        let n0 = frames.div_ceil(2);
        let n = n0.div_ceil(2);
        let valid = features.soft_tokens();
        if frames == 0 || frames > MAX_FRAMES || valid == 0 || features.values.len() != frames * 128
        {
            return Err("embedding_gemma2: invalid audio feature shape".into());
        }
        stream
            .upload(
                &features.values,
                self.b.input.slice_mut(0..features.values.len()),
            )
            .map_err(hip)?;
        let s = "audio_tower.subsample_conv_projection.";
        let w = &self.weights;
        let b = &mut self.b;
        k::audio_conv2d(
            stream,
            b.input.as_slice(),
            w.get(&format!("{s}layer0.conv.weight"))?,
            b.conv0.as_mut_slice(),
            frames as i32,
            128,
            1,
            128,
            valid_frames as i32,
        )
        .map_err(launch)?;
        k::audio_norm_in_place(
            stream,
            b.conv0.as_mut_slice(),
            Some(w.get(&format!("{s}layer0.norm.weight"))?),
            (n0 * 64) as i32,
            128,
            true,
            false,
        )
        .map_err(launch)?;
        k::audio_conv2d(
            stream,
            b.conv0.as_slice(),
            w.get(&format!("{s}layer1.conv.weight"))?,
            b.conv1.as_mut_slice(),
            n0 as i32,
            64,
            128,
            32,
            valid_frames.div_ceil(2) as i32,
        )
        .map_err(launch)?;
        k::audio_norm_in_place(
            stream,
            b.conv1.as_mut_slice(),
            Some(w.get(&format!("{s}layer1.norm.weight"))?),
            (n * 32) as i32,
            32,
            true,
            false,
        )
        .map_err(launch)?;
        gemm(
            &self.blas,
            stream,
            w.get(&format!("{s}input_proj_linear.weight"))?,
            b.conv1.as_slice(),
            b.state.as_mut_slice(),
            n,
            1024,
            1024,
        )?;
        for layer in 0..12 {
            let p = format!("audio_tower.layers.{layer}.");
            self.feed_forward(stream, &format!("{p}feed_forward1."), n)?;
            self.normalize(
                stream,
                Slot::State,
                &format!("{p}norm_pre_attn.weight"),
                Slot::Update,
                n,
                1024,
                true,
            )?;
            self.linear(
                stream,
                &format!("{p}self_attn.q_proj"),
                Slot::Update,
                Slot::Q,
                n,
                1024,
                1024,
            )?;
            self.linear(
                stream,
                &format!("{p}self_attn.k_proj"),
                Slot::Update,
                Slot::K,
                n,
                1024,
                1024,
            )?;
            self.linear(
                stream,
                &format!("{p}self_attn.v_proj"),
                Slot::Update,
                Slot::V,
                n,
                1024,
                1024,
            )?;
            let w = &self.weights;
            let b = &mut self.b;
            gemm(
                &self.blas,
                stream,
                w.get(&format!("{p}self_attn.relative_k_proj.weight"))?,
                b.positions.as_slice(),
                b.relative.as_mut_slice(),
                13,
                1024,
                1024,
            )?;
            k::audio_attention(
                stream,
                b.q.as_slice(),
                b.k.as_slice(),
                b.v.as_slice(),
                b.relative.as_slice(),
                w.get(&format!("{p}self_attn.per_dim_scale"))?,
                b.context.as_mut_slice(),
                n as i32,
                valid as i32,
            )
            .map_err(launch)?;
            self.linear(
                stream,
                &format!("{p}self_attn.post"),
                Slot::Context,
                Slot::Update,
                n,
                1024,
                1024,
            )?;
            self.normalize(
                stream,
                Slot::Update,
                &format!("{p}norm_post_attn.weight"),
                Slot::Update,
                n,
                1024,
                true,
            )?;
            self.add(stream, Slot::Update, n, 1.0)?;
            self.normalize(
                stream,
                Slot::State,
                &format!("{p}lconv1d.pre_layer_norm.weight"),
                Slot::Update,
                n,
                1024,
                false,
            )?;
            self.linear(
                stream,
                &format!("{p}lconv1d.linear_start"),
                Slot::Update,
                Slot::Temp,
                n,
                2048,
                1024,
            )?;
            let w = &self.weights;
            let b = &mut self.b;
            k::audio_glu(
                stream,
                b.temp.as_slice(),
                b.context.as_mut_slice(),
                n as i32,
            )
            .map_err(launch)?;
            k::audio_depthwise(
                stream,
                b.context.as_slice(),
                w.get(&format!("{p}lconv1d.depthwise_conv1d.weight"))?,
                b.update.as_mut_slice(),
                n as i32,
            )
            .map_err(launch)?;
            self.normalize(
                stream,
                Slot::Update,
                &format!("{p}lconv1d.conv_norm.weight"),
                Slot::Update,
                n,
                1024,
                true,
            )?;
            self.activate(stream, Slot::Update, n * 1024)?;
            self.linear(
                stream,
                &format!("{p}lconv1d.linear_end"),
                Slot::Update,
                Slot::Temp,
                n,
                1024,
                1024,
            )?;
            self.add(stream, Slot::Temp, n, 1.0)?;
            self.feed_forward(stream, &format!("{p}feed_forward2."), n)?;
            self.normalize(
                stream,
                Slot::State,
                &format!("{p}norm_out.weight"),
                Slot::State,
                n,
                1024,
                true,
            )?;
            if let Some(cancelled) = is_cancelled {
                stream.synchronize().map_err(hip)?;
                if cancelled() {
                    return Err("audio encoding cancelled".into());
                }
            }
        }
        let w = &self.weights;
        let b = &mut self.b;
        gemm(
            &self.blas,
            stream,
            w.get("audio_tower.output_proj.weight")?,
            b.state.as_slice(),
            b.projected.as_mut_slice(),
            n,
            1536,
            1024,
        )?;
        k::audio_bias(
            stream,
            b.projected.as_mut_slice(),
            w.get("audio_tower.output_proj.bias")?,
            n as i32,
        )
        .map_err(launch)?;
        self.normalize(
            stream,
            Slot::Projected,
            "",
            Slot::Normed,
            valid,
            1536,
            false,
        )?;
        let w = &self.weights;
        let b = &mut self.b;
        gemm(
            &self.blas,
            stream,
            w.get("embed_audio.embedding_projection.weight")?,
            b.normed.as_slice(),
            b.features.as_mut_slice(),
            valid,
            512,
            1536,
        )?;
        Ok(self.b.features.slice(0..valid * 512))
    }

    /// The output projection's rows of the last run (`encoded_states`).
    pub fn encoded_states(&self) -> DeviceSlice<'_, f32> {
        self.b.projected.as_slice()
    }
}
