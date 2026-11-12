//! Validated view of a Gemma 4 `gemma4v` BF16 vision sidecar
//! (`vision/weights.cpp`). Matrices are BF16 `[out][in]`; norms, the patch
//! embedding (`[1152][3][16][16]`), the position tables
//! (`[2][10240][1152]`) and the standardization vectors are F32.

use gufo_core::gguf::{Gguf, Value};
use gufo_formats::GgmlType;

pub const HIDDEN: u32 = 1152;
pub const HEADS: u32 = 16;
pub const HEAD_DIM: u32 = 72;
pub const FFN: u32 = 4304;
pub const LAYERS: u32 = 27;
/// Rows of each learned position table (x and y).
pub const POSITIONS: u32 = 10240;
pub const EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 100.0;
/// Pixel values per patch, ordered channel, row, column.
pub const PATCH_VALUES: u32 = 3 * 16 * 16;

/// One tensor of the sidecar: its GGUF dimensions, format and payload
/// location in the mapped file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Absolute byte offset in the file.
    pub offset: usize,
    pub bytes: usize,
}

impl VisionTensor {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// The payload in the mapped file the tensor was resolved from.
    ///
    /// # Panics
    /// When `file` is not that file.
    pub fn data<'a>(&self, file: &'a [u8]) -> &'a [u8] {
        &file[self.offset..self.offset + self.bytes]
    }

    /// The payload widened to F32 (F32 or BF16 tensors).
    pub fn widen(&self, file: &[u8]) -> Vec<f32> {
        let data = self.data(file);
        if self.ty == GgmlType::F32 {
            data.as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        } else {
            data.as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
                .collect()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionLayer {
    /// Pre-attention RMS weight.
    pub ln1: VisionTensor,
    pub q: VisionTensor,
    pub k: VisionTensor,
    pub v: VisionTensor,
    pub q_norm: VisionTensor,
    pub k_norm: VisionTensor,
    pub out: VisionTensor,
    pub attn_post_norm: VisionTensor,
    /// Pre-FFN RMS weight.
    pub ln2: VisionTensor,
    pub gate: VisionTensor,
    pub up: VisionTensor,
    pub down: VisionTensor,
    pub ffn_post_norm: VisionTensor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionWeights {
    pub output_width: u32,
    pub patch: VisionTensor,
    pub position: VisionTensor,
    pub std_bias: VisionTensor,
    pub std_scale: VisionTensor,
    /// `[output_width][1152]`
    pub projection: VisionTensor,
    pub layers: Vec<VisionLayer>,
}

impl VisionWeights {
    /// Fails when the sidecar is not a Gemma 4 vision projector for a text
    /// model of `output_width` (`Weights::Resolve`).
    pub fn resolve(gguf: &Gguf<'_>, output_width: u32) -> Result<Self, String> {
        let fail = |what: &str| format!("Gemma 4 vision sidecar: {what}");
        if gguf.string("general.architecture") != Some(b"clip".as_slice())
            || gguf.string("general.type") != Some(b"mmproj".as_slice())
            || gguf.bool("clip.has_vision_encoder") != Some(true)
            || gguf.string("clip.vision.projector_type") != Some(b"gemma4v".as_slice())
        {
            return Err(fail("not a gemma4v projector"));
        }
        let expect = |key: &str, value: u64| {
            if gguf.u64(key) == Some(value) {
                Ok(())
            } else {
                Err(fail(&format!("unexpected {key}")))
            }
        };
        expect("clip.vision.projection_dim", u64::from(output_width))?;
        expect("clip.vision.patch_size", 16)?;
        expect("clip.vision.embedding_length", u64::from(HIDDEN))?;
        expect("clip.vision.feed_forward_length", u64::from(FFN))?;
        expect("clip.vision.block_count", u64::from(LAYERS))?;
        expect("clip.vision.attention.head_count", u64::from(HEADS))?;
        if gguf.f32("clip.vision.attention.layer_norm_epsilon") != Some(EPS) {
            return Err(fail("unexpected layer_norm_epsilon"));
        }
        for (key, expected) in [
            ("clip.vision.image_mean", 0.0),
            ("clip.vision.image_std", 1.0),
        ] {
            match gguf.metadata(key) {
                Some(Value::FloatArray(values)) if values.as_slice() == [expected; 3] => {}
                _ => return Err(fail("unsupported image normalization")),
            }
        }
        // The pooling kernel is fixed at 3 unless the sidecar says otherwise.
        if let Some(merge) = gguf.u64("clip.vision.projector.scale_factor")
            && merge != 3
        {
            return Err(fail("unsupported pooling kernel"));
        }

        let require = |name: &str, shape: &[u64], ty: GgmlType| -> Result<VisionTensor, String> {
            let width = if ty == GgmlType::F32 { 4 } else { 2 };
            match gguf.tensor(name) {
                Some(t)
                    if t.dims == shape && t.ty == ty && t.bytes as u64 == t.elements() * width =>
                {
                    Ok(VisionTensor {
                        name: t.name.to_string(),
                        dims: t.dims.clone(),
                        ty: t.ty,
                        offset: t.offset,
                        bytes: t.bytes,
                    })
                }
                _ => Err(fail(&format!("invalid tensor {name}"))),
            }
        };
        let (hidden, ffn, head_dim) = (u64::from(HIDDEN), u64::from(FFN), u64::from(HEAD_DIM));
        let (f32_, bf16) = (GgmlType::F32, GgmlType::Bf16);
        let patch = require("v.patch_embd.weight", &[16, 16, 3, hidden], f32_)?;
        let position = require(
            "v.position_embd.weight",
            &[hidden, u64::from(POSITIONS), 2],
            f32_,
        )?;
        let std_bias = require("v.std_bias", &[hidden], f32_)?;
        let std_scale = require("v.std_scale", &[hidden], f32_)?;
        let projection = require(
            "mm.input_projection.weight",
            &[hidden, u64::from(output_width)],
            bf16,
        )?;
        let mut layers = Vec::with_capacity(LAYERS as usize);
        for l in 0..LAYERS {
            let p = format!("v.blk.{l}.");
            let t = |name: &str, shape: &[u64], ty| require(&format!("{p}{name}"), shape, ty);
            layers.push(VisionLayer {
                ln1: t("ln1.weight", &[hidden], f32_)?,
                q: t("attn_q.weight", &[hidden, hidden], bf16)?,
                k: t("attn_k.weight", &[hidden, hidden], bf16)?,
                v: t("attn_v.weight", &[hidden, hidden], bf16)?,
                q_norm: t("attn_q_norm.weight", &[head_dim], f32_)?,
                k_norm: t("attn_k_norm.weight", &[head_dim], f32_)?,
                out: t("attn_out.weight", &[hidden, hidden], bf16)?,
                attn_post_norm: t("attn_post_norm.weight", &[hidden], f32_)?,
                ln2: t("ln2.weight", &[hidden], f32_)?,
                gate: t("ffn_gate.weight", &[hidden, ffn], bf16)?,
                up: t("ffn_up.weight", &[hidden, ffn], bf16)?,
                down: t("ffn_down.weight", &[ffn, hidden], bf16)?,
                ffn_post_norm: t("ffn_post_norm.weight", &[hidden], f32_)?,
            });
        }
        // Clamp scalars (Gemma4ClippableLinear) or biases would change the
        // graph.
        if gguf.tensors().len() != (5 + 13 * LAYERS) as usize {
            return Err(fail(
                "unexpected tensors (clipped linears or biases are unsupported)",
            ));
        }
        Ok(Self {
            output_width,
            patch,
            position,
            std_bias,
            std_scale,
            projection,
            layers,
        })
    }
}
