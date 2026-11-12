//! The vision tower (`EncoderRuntime::Impl::LoadVision` and `RunVision`):
//! 16 BF16 layers over the unpadded patches of one image, pooled into soft
//! tokens that replace the image's placeholder rows of the text batch.

use std::path::Path;

use gufo_embedding_gemma2_kernels as k;
use gufo_hip::{Bf16, DeviceBuffer, DeviceSliceMut, LaunchError, Stream};

use crate::encoder::{CHUNK_TOKENS, upload, upload_concatenated};
use crate::gemm::Gemm;
use crate::image_processor::{
    DEFAULT_IMAGE_SOFT_TOKENS, IMAGE_SOFT_TOKEN_BUDGETS, ImagePatches, MAX_IMAGE_SOFT_TOKENS,
    validate_image_processor,
};
use crate::loader::{Towers, load_model_directory};

const HIDDEN: usize = 768;

fn hip(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn launch(error: LaunchError) -> String {
    error.to_string()
}

struct Layer {
    input_norm: DeviceBuffer<Bf16>,
    post_attention_norm: DeviceBuffer<Bf16>,
    pre_feedforward_norm: DeviceBuffer<Bf16>,
    post_feedforward_norm: DeviceBuffer<Bf16>,
    /// q, k and v stacked row-wise.
    qkv: DeviceBuffer<Bf16>,
    output: DeviceBuffer<Bf16>,
    query_norm: DeviceBuffer<Bf16>,
    key_norm: DeviceBuffer<Bf16>,
    /// gate and up stacked row-wise.
    gate_up: DeviceBuffer<Bf16>,
    down: DeviceBuffer<Bf16>,
}

pub struct Vision {
    input: DeviceBuffer<Bf16>,
    position_table: DeviceBuffer<Bf16>,
    projection: DeviceBuffer<Bf16>,
    layers: Vec<Layer>,
    /// Allocated by the first image (`ReserveVision`).
    scratch: Option<Scratch>,
}

/// The tower's activations for one image of up to `capacity` soft tokens of
/// 3x3 patches.
struct Scratch {
    capacity: usize,
    pixels: DeviceBuffer<f32>,
    pixels_bf16: DeviceBuffer<Bf16>,
    state: DeviceBuffer<f32>,
    normed: DeviceBuffer<Bf16>,
    qkv: DeviceBuffer<f32>,
    query: DeviceBuffer<Bf16>,
    key: DeviceBuffer<Bf16>,
    value: DeviceBuffer<Bf16>,
    context: DeviceBuffer<Bf16>,
    temp: DeviceBuffer<f32>,
    gate_up: DeviceBuffer<f32>,
    act: DeviceBuffer<Bf16>,
    pooled: DeviceBuffer<Bf16>,
    features: DeviceBuffer<f32>,
}

impl Vision {
    /// `LoadVision`: validates the image processor and uploads the tower.
    pub fn load(model_dir: &Path, stream: &Stream) -> Result<Self, String> {
        validate_image_processor(model_dir)?;
        let weights = load_model_directory(
            model_dir,
            Towers {
                vision: true,
                audio: false,
            },
        )?;
        let store = &weights.store;
        let one = |name: &str| upload(stream, store, name);
        let mut layers = Vec::with_capacity(16);
        for i in 0..16 {
            let p = format!("vision_tower.encoder.layers.{i}.");
            let named = |name: &str| one(&format!("{p}{name}"));
            let stacked = |names: &[&str]| {
                let names: Vec<String> = names.iter().map(|n| format!("{p}{n}")).collect();
                upload_concatenated(stream, store, &names)
            };
            layers.push(Layer {
                input_norm: named("input_layernorm.weight")?,
                post_attention_norm: named("post_attention_layernorm.weight")?,
                pre_feedforward_norm: named("pre_feedforward_layernorm.weight")?,
                post_feedforward_norm: named("post_feedforward_layernorm.weight")?,
                qkv: stacked(&[
                    "self_attn.q_proj.linear.weight",
                    "self_attn.k_proj.linear.weight",
                    "self_attn.v_proj.linear.weight",
                ])?,
                output: named("self_attn.o_proj.linear.weight")?,
                query_norm: named("self_attn.q_norm.weight")?,
                key_norm: named("self_attn.k_norm.weight")?,
                gate_up: stacked(&["mlp.gate_proj.linear.weight", "mlp.up_proj.linear.weight"])?,
                down: named("mlp.down_proj.linear.weight")?,
            });
        }
        Ok(Self {
            input: one("vision_tower.patch_embedder.input_proj.weight")?,
            position_table: one("vision_tower.patch_embedder.position_embedding_table")?,
            projection: one("embed_vision.embedding_projection.weight")?,
            layers,
            scratch: None,
        })
    }
}

impl Scratch {
    /// `ReserveVision`: the scratch starts at the default budget and grows
    /// to the smallest budget that holds a larger image, so requests at the
    /// default keep their footprint.
    fn reserve<'s>(
        scratch: &'s mut Option<Scratch>,
        stream: &Stream,
        soft_tokens: usize,
    ) -> Result<&'s mut Scratch, String> {
        if scratch
            .as_ref()
            .is_some_and(|reserved| soft_tokens <= reserved.capacity)
        {
            return Ok(scratch.as_mut().expect("reserved scratch"));
        }
        // Validation caps an image at the largest budget.
        let floor = soft_tokens.max(DEFAULT_IMAGE_SOFT_TOKENS as usize);
        let tokens = IMAGE_SOFT_TOKEN_BUDGETS
            .iter()
            .map(|&budget| budget as usize)
            .find(|&budget| budget >= floor)
            .unwrap_or(MAX_IMAGE_SOFT_TOKENS as usize);
        // Earlier images may still be using the old buffers.
        stream.synchronize().map_err(hip)?;
        *scratch = None;
        let n = tokens * 9;
        let h = HIDDEN;
        let f32s = |len: usize| stream.alloc_persistent::<f32>(len).map_err(hip);
        let bf16s = |len: usize| stream.alloc_persistent::<Bf16>(len).map_err(hip);
        Ok(scratch.insert(Scratch {
            capacity: tokens,
            pixels: f32s(n * h)?,
            pixels_bf16: bf16s(n * h)?,
            state: f32s(n * h)?,
            normed: bf16s(n * h)?,
            qkv: f32s(n * h * 3)?,
            query: bf16s(n * h)?,
            key: bf16s(n * h)?,
            value: bf16s(n * h)?,
            context: bf16s(n * h)?,
            temp: f32s(n * h)?,
            gate_up: f32s(n.min(CHUNK_TOKENS) * 6144)?,
            act: bf16s(n * 3072)?,
            pooled: bf16s(tokens * h)?,
            features: f32s(tokens * 512)?,
        }))
    }
}

impl Vision {
    /// `RunVision`: encodes `image` and writes its soft tokens into the
    /// placeholder rows `state` and `embeddings` (residual stream and
    /// per-layer input copy) start at.
    pub fn run(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        image: &ImagePatches,
        state: DeviceSliceMut<'_, f32>,
        embeddings: DeviceSliceMut<'_, Bf16>,
    ) -> Result<(), String> {
        let tokens = image.soft_tokens();
        let Self {
            input,
            position_table,
            projection,
            layers,
            scratch,
        } = self;
        let s = Scratch::reserve(scratch, stream, tokens)?;
        let n = image.rows as usize * image.columns as usize;
        let epsilon = 1e-6f32;
        stream
            .upload(&image.pixels, s.pixels.slice_mut(0..image.pixels.len()))
            .map_err(hip)?;
        k::vision_pixels(
            stream,
            s.pixels.as_slice(),
            s.pixels_bf16.as_mut_slice(),
            n * HIDDEN,
        )
        .map_err(launch)?;
        gemm.blas_gemm(
            stream,
            input.as_slice(),
            s.pixels_bf16.slice(0..n * HIDDEN),
            s.temp.slice_mut(0..n * HIDDEN),
            n,
            HIDDEN,
            HIDDEN,
        )
        .map_err(launch)?;
        k::vision_positions(
            stream,
            s.temp.as_slice(),
            position_table.as_slice(),
            s.state.as_mut_slice(),
            n,
            image.columns,
        )
        .map_err(launch)?;
        for layer in layers.iter() {
            k::rms_norm_to_bf16(
                stream,
                s.state.as_slice(),
                layer.input_norm.as_slice(),
                s.normed.as_mut_slice(),
                n,
                HIDDEN,
                epsilon,
            )
            .map_err(launch)?;
            gemm.blas_gemm(
                stream,
                layer.qkv.as_slice(),
                s.normed.slice(0..n * HIDDEN),
                s.qkv.slice_mut(0..n * 2304),
                n,
                2304,
                HIDDEN,
            )
            .map_err(launch)?;
            k::vision_qkv(
                stream,
                s.qkv.as_slice(),
                layer.query_norm.as_slice(),
                layer.key_norm.as_slice(),
                s.query.as_mut_slice(),
                s.key.as_mut_slice(),
                s.value.as_mut_slice(),
                n,
                image.columns,
            )
            .map_err(launch)?;
            k::vision_attention(
                stream,
                s.query.as_slice(),
                s.key.as_slice(),
                s.value.as_slice(),
                s.context.as_mut_slice(),
                n,
            )
            .map_err(launch)?;
            gemm.blas_gemm(
                stream,
                layer.output.as_slice(),
                s.context.slice(0..n * HIDDEN),
                s.temp.slice_mut(0..n * HIDDEN),
                n,
                HIDDEN,
                HIDDEN,
            )
            .map_err(launch)?;
            k::rms_norm_residual(
                stream,
                s.temp.as_slice(),
                layer.post_attention_norm.as_slice(),
                s.state.slice_mut(0..n * HIDDEN),
                n,
                HIDDEN,
                epsilon,
            )
            .map_err(launch)?;
            k::rms_norm_to_bf16(
                stream,
                s.state.as_slice(),
                layer.pre_feedforward_norm.as_slice(),
                s.normed.as_mut_slice(),
                n,
                HIDDEN,
                epsilon,
            )
            .map_err(launch)?;
            for row in (0..n).step_by(CHUNK_TOKENS) {
                let count = CHUNK_TOKENS.min(n - row);
                gemm.blas_gemm(
                    stream,
                    layer.gate_up.as_slice(),
                    s.normed.slice(row * HIDDEN..(row + count) * HIDDEN),
                    s.gate_up.slice_mut(0..count * 6144),
                    count,
                    6144,
                    HIDDEN,
                )
                .map_err(launch)?;
                k::geglu(
                    stream,
                    s.gate_up.as_slice(),
                    s.act.slice_mut(row * 3072..(row + count) * 3072),
                    count,
                    3072,
                )
                .map_err(launch)?;
            }
            gemm.blas_gemm(
                stream,
                layer.down.as_slice(),
                s.act.slice(0..n * 3072),
                s.temp.slice_mut(0..n * HIDDEN),
                n,
                HIDDEN,
                3072,
            )
            .map_err(launch)?;
            k::rms_norm_residual(
                stream,
                s.temp.as_slice(),
                layer.post_feedforward_norm.as_slice(),
                s.state.slice_mut(0..n * HIDDEN),
                n,
                HIDDEN,
                epsilon,
            )
            .map_err(launch)?;
        }
        k::vision_pool(
            stream,
            s.state.as_slice(),
            s.pooled.as_mut_slice(),
            image.rows,
            image.columns,
        )
        .map_err(launch)?;
        gemm.blas_gemm(
            stream,
            projection.as_slice(),
            s.pooled.slice(0..tokens * HIDDEN),
            s.features.slice_mut(0..tokens * 512),
            tokens,
            512,
            HIDDEN,
        )
        .map_err(launch)?;
        k::vision_insert(stream, s.features.as_slice(), state, embeddings, tokens).map_err(launch)
    }
}
