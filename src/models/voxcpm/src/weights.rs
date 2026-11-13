//! The generator's weights on the device (`weights.hpp`, `weights.cpp`),
//! with q/k/v and gate/up of every layer placed side by side so that each
//! pair or triple is one product. GGML Q8_0 matrices stay quantized, in
//! tiles; the DiT's bfloat16 matrices are tiled too (its products have 22
//! rows); everything else is bfloat16 rows. Every tensor lives in one device
//! buffer; the weights name their byte ranges in it, where the C++ structs
//! hold pointers.

use gufo_hip::{Bf16, DeviceBuffer, DeviceSlice, F16, Stream};
use gufo_voxcpm_kernels as k;

use crate::config::{Config, TransformerConfig};
use crate::matrix::{Matrix, MatrixView, TILE_COLUMNS, TILE_ROWS, WeightFormat};
use crate::model_files::{ModelFiles, same_shape};
use crate::safetensors::{DType, Tensor};

/// Alignment of every tensor in the device buffer (`kAlignment`).
const ALIGNMENT: usize = 16;

/// A bfloat16 vector (or the embedding table) in the device buffer: its
/// byte offset and element count (`const Bf16*`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Vector {
    pub offset: usize,
    pub len: usize,
}

/// A linear layer `y = x W^T (+ b)` (`Projection`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Projection {
    pub weight: Matrix,
    pub bias: Option<Vector>,
    pub input: usize,
    pub output: usize,
}

/// One layer of a MiniCPM4 stack (`LayerWeights`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayerWeights {
    pub input_norm: Vector,
    /// `[q heads | k heads | v heads]`.
    pub qkv: Projection,
    pub o: Projection,
    pub post_norm: Vector,
    /// `[gate | up]`.
    pub gate_up: Projection,
    pub down: Projection,
}

/// A MiniCPM4 stack (`StackWeights`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StackWeights {
    pub config: TransformerConfig,
    pub layers: Vec<LayerWeights>,
    pub norm: Vector,
}

/// Every weight of the generator (`GeneratorWeights`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GeneratorWeights {
    pub base_lm: StackWeights,
    pub residual_lm: StackWeights,
    pub encoder: StackWeights,
    pub dit: StackWeights,
    /// `[vocab][hidden]`.
    pub embed: Vector,
    /// The local encoder's CLS row.
    pub special: Vector,
    /// `feat_encoder.in_proj`.
    pub encoder_in: Projection,
    pub enc_to_lm: Projection,
    pub lm_to_dit: Projection,
    pub res_to_dit: Projection,
    pub fusion: Projection,
    pub fsq_in: Projection,
    pub fsq_out: Projection,
    pub stop_proj: Projection,
    pub stop_head: Projection,
    pub dit_in: Projection,
    pub dit_cond: Projection,
    pub dit_out: Projection,
    pub time_1: Projection,
    pub time_2: Projection,
    pub delta_time_1: Projection,
    pub delta_time_2: Projection,
}

/// The device buffer an [`Uploader`] fills, with the stream that copies.
struct Target<'a> {
    stream: &'a Stream,
    storage: &'a mut DeviceBuffer<u8>,
    /// Staged file bytes of a conversion (`staging_`).
    staging: Option<DeviceBuffer<u8>>,
    /// bfloat16 rows before tiling (`rows_`).
    rows: Option<DeviceBuffer<Bf16>>,
}

/// Grows `buffer` to at least `count` elements (`DeviceBuffer::Reserve`).
fn reserve<T: gufo_hip::DeviceCopy>(
    stream: &Stream,
    buffer: &mut Option<DeviceBuffer<T>>,
    count: usize,
) -> Result<(), String> {
    if buffer.as_ref().map_or(0, DeviceBuffer::len) < count {
        *buffer = None;
        *buffer = Some(
            stream
                .alloc_persistent(count.max(1))
                .map_err(|e| e.to_string())?,
        );
    }
    Ok(())
}

fn gpu(error: impl std::fmt::Display) -> String {
    error.to_string()
}

impl Target<'_> {
    /// Copies `tensor`'s bytes into the staging buffer (`Stage`).
    fn stage(&mut self, tensor: &Tensor) -> Result<(), String> {
        reserve(self.stream, &mut self.staging, tensor.bytes)?;
        let staging = self.staging.as_mut().ok_or("staging")?;
        self.stream
            .upload_bytes(tensor.data(), staging.slice_mut(0..tensor.bytes))
            .map_err(gpu)
    }

    /// Writes `tensor` as `elements` bfloat16 values at byte `offset` of the
    /// storage (`offset` null: the tiling rows instead) (`ToBf16`).
    fn write_bf16(
        &mut self,
        tensor: &Tensor,
        name: &str,
        offset: Option<usize>,
    ) -> Result<(), String> {
        let elements = tensor.elements();
        let stream = self.stream;
        if tensor.dtype == DType::Bf16 {
            let bytes = elements * 2;
            if tensor.bytes != bytes {
                return Err(format!("{name} has an unexpected size"));
            }
            return match offset {
                Some(offset) => stream
                    .upload_bytes(
                        tensor.data(),
                        self.storage.slice_mut(offset..offset + bytes),
                    )
                    .map_err(gpu),
                None => {
                    let rows = self.rows.as_mut().ok_or("rows")?;
                    stream
                        .upload_bytes(tensor.data(), rows.slice_mut(0..elements))
                        .map_err(gpu)
                }
            };
        }
        match tensor.dtype {
            DType::F16 | DType::F32 | DType::Q8_0 => {}
            other => return Err(format!("{name} has an unsupported type {}", other.name())),
        }
        if tensor.dtype == DType::Q8_0 {
            check_q8(tensor, name)?;
        }
        self.stage(tensor)?;
        let staging = self.staging.as_ref().ok_or("staging")?;
        let raw = staging.slice(0..tensor.bytes);
        let target = match offset {
            Some(offset) => self
                .storage
                .slice_mut(offset..offset + elements * 2)
                .cast::<Bf16>()
                .ok_or("unaligned weight")?,
            None => self.rows.as_mut().ok_or("rows")?.slice_mut(0..elements),
        };
        match tensor.dtype {
            DType::F16 => {
                let half = raw.cast::<F16>().ok_or("unaligned staging")?;
                k::half_to_bf16(stream, half, target, elements)
            }
            DType::F32 => {
                let float = raw.cast::<f32>().ok_or("unaligned staging")?;
                k::float_to_bf16(stream, float, target, elements)
            }
            _ => k::q8_0_to_bf16(stream, raw, target, elements),
        }
        .map_err(gpu)?;
        stream.synchronize().map_err(gpu)
    }

    /// Writes `tensor` (`[out][in]`) in `format` at byte `offset` (`Write`).
    fn write(
        &mut self,
        tensor: &Tensor,
        name: &str,
        format: WeightFormat,
        out: usize,
        input: usize,
        offset: usize,
    ) -> Result<(), String> {
        let bytes = format.bytes(out, input);
        match format {
            WeightFormat::Bf16 => self.write_bf16(tensor, name, Some(offset)),
            WeightFormat::Bf16Tiles => {
                reserve(self.stream, &mut self.rows, out * input)?;
                self.write_bf16(tensor, name, None)?;
                let stream = self.stream;
                let mut target = self.storage.slice_mut(offset..offset + bytes);
                stream.fill_bytes(target.reborrow(), 0).map_err(gpu)?;
                let rows = self.rows.as_ref().ok_or("rows")?;
                k::bf16_to_tiles(
                    stream,
                    rows.slice(0..out * input),
                    out,
                    input,
                    target.cast::<Bf16>().ok_or("unaligned weight")?,
                )
                .map_err(gpu)?;
                stream.synchronize().map_err(gpu)
            }
            WeightFormat::Q8Tiles => {
                check_q8(tensor, name)?;
                self.stage(tensor)?;
                let stream = self.stream;
                let mut target = self.storage.slice_mut(offset..offset + bytes);
                stream.fill_bytes(target.reborrow(), 0).map_err(gpu)?;
                let staging = self.staging.as_ref().ok_or("staging")?;
                k::q8_0_to_tiles(stream, staging.slice(0..tensor.bytes), out, input, target)
                    .map_err(gpu)?;
                stream.synchronize().map_err(gpu)
            }
        }
    }
}

/// `CheckQ8`: whole Q8_0 blocks.
fn check_q8(tensor: &Tensor, name: &str) -> Result<(), String> {
    let elements = tensor.elements();
    if !elements.is_multiple_of(32) || tensor.bytes != elements / 32 * 34 {
        return Err(format!("{name} is not whole Q8_0 blocks"));
    }
    Ok(())
}

/// The format of a matrix (`Uploader::Format`).
fn format(q8: bool, tiles: bool, input: usize) -> WeightFormat {
    if !input.is_multiple_of(TILE_COLUMNS) {
        WeightFormat::Bf16
    } else if q8 {
        WeightFormat::Q8Tiles
    } else if tiles {
        WeightFormat::Bf16Tiles
    } else {
        WeightFormat::Bf16
    }
}

/// Places tensors one after another in the device buffer: Q8_0 matrices as
/// Q8_0 tiles, other matrices as bfloat16 rows or tiles, vectors as
/// bfloat16. Without a target it only counts the bytes (`Uploader`).
struct Uploader<'a> {
    files: &'a ModelFiles,
    target: Option<Target<'a>>,
    capacity: usize,
    used: usize,
    taken: usize,
}

impl<'a> Uploader<'a> {
    fn new(files: &'a ModelFiles, target: Option<Target<'a>>) -> Self {
        let capacity = target.as_ref().map_or(0, |t| t.storage.len());
        Self {
            files,
            target,
            capacity,
            used: 0,
            taken: 0,
        }
    }

    /// A vector (or the embedding table) as bfloat16 (`Take`).
    fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vector, String> {
        let tensor = self.find(name, shape)?;
        let elements = tensor.elements();
        let offset = self.reserve(elements * 2)?;
        if let Some(target) = &mut self.target {
            target.write_bf16(tensor, name, Some(offset))?;
        }
        Ok(Vector {
            offset,
            len: elements,
        })
    }

    /// An `[out][in]` matrix: Q8_0 tiles for a Q8_0 tensor when `q8`,
    /// bfloat16 tiles when `tiles`, bfloat16 rows otherwise or when the
    /// columns don't fill whole tiles (`TakeMatrix`).
    fn take_matrix(
        &mut self,
        name: &str,
        out: usize,
        input: usize,
        tiles: bool,
        q8: bool,
    ) -> Result<Matrix, String> {
        let tensor = self.find(name, &[out, input])?;
        let format = format(tensor.dtype == DType::Q8_0 && q8, tiles, input);
        let bytes = format.bytes(out, input);
        let offset = self.reserve(bytes)?;
        if let Some(target) = &mut self.target {
            target.write(tensor, name, format, out, input, offset)?;
        }
        Ok(Matrix {
            format,
            offset,
            bytes,
        })
    }

    /// `names` (`[outs[i]][in]` each) side by side as one matrix: Q8_0
    /// tiles when all are Q8_0, bfloat16 tiles when `tiles`, bfloat16 rows
    /// otherwise or when a part doesn't fill whole tiles of rows (`Fused`).
    fn fused(
        &mut self,
        names: &[String],
        outs: &[usize],
        input: usize,
        tiles: bool,
    ) -> Result<Matrix, String> {
        let mut tensors = Vec::with_capacity(names.len());
        let mut q8 = true;
        let mut whole_tiles = true;
        for (name, &out) in names.iter().zip(outs) {
            let tensor = self.find(name, &[out, input])?;
            q8 = q8 && tensor.dtype == DType::Q8_0;
            whole_tiles = whole_tiles && out.is_multiple_of(TILE_ROWS);
            tensors.push(tensor);
        }
        let format = format(q8 && whole_tiles, tiles && whole_tiles, input);
        let total: usize = outs.iter().map(|&out| format.bytes(out, input)).sum();
        let offset = self.reserve(total)?;
        if let Some(target) = &mut self.target {
            let mut at = offset;
            for ((tensor, name), &out) in tensors.iter().zip(names).zip(outs) {
                target.write(tensor, name, format, out, input, at)?;
                at += format.bytes(out, input);
            }
        }
        Ok(Matrix {
            format,
            offset,
            bytes: total,
        })
    }

    /// `Linear`.
    fn linear(
        &mut self,
        prefix: &str,
        input: usize,
        output: usize,
        bias: bool,
        tiles: bool,
    ) -> Result<Projection, String> {
        let weight = self.take_matrix(&format!("{prefix}.weight"), output, input, tiles, true)?;
        let bias = if bias {
            Some(self.take(&format!("{prefix}.bias"), &[output])?)
        } else {
            None
        };
        Ok(Projection {
            weight,
            bias,
            input,
            output,
        })
    }

    /// One MiniCPM4 stack; `tiles` stores its bfloat16 matrices as tiles
    /// (`Stack`).
    fn stack(
        &mut self,
        prefix: &str,
        config: &TransformerConfig,
        tiles: bool,
    ) -> Result<StackWeights, String> {
        let hidden = config.hidden;
        let q = config.heads * config.head_dim;
        let kv = config.kv_heads * config.head_dim;
        let mut layers = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            let layer = format!("{prefix}.layers.{index}");
            let input_norm = self.take(&format!("{layer}.input_layernorm.weight"), &[hidden])?;
            let attention = format!("{layer}.self_attn.");
            let qkv = Projection {
                weight: self.fused(
                    &[
                        format!("{attention}q_proj.weight"),
                        format!("{attention}k_proj.weight"),
                        format!("{attention}v_proj.weight"),
                    ],
                    &[q, kv, kv],
                    hidden,
                    tiles,
                )?,
                bias: None,
                input: hidden,
                output: q + 2 * kv,
            };
            let o = self.linear(&format!("{attention}o_proj"), q, hidden, false, tiles)?;
            let post_norm = self.take(
                &format!("{layer}.post_attention_layernorm.weight"),
                &[hidden],
            )?;
            let gate_up = Projection {
                weight: self.fused(
                    &[
                        format!("{layer}.mlp.gate_proj.weight"),
                        format!("{layer}.mlp.up_proj.weight"),
                    ],
                    &[config.intermediate, config.intermediate],
                    hidden,
                    tiles,
                )?,
                bias: None,
                input: hidden,
                output: 2 * config.intermediate,
            };
            let down = self.linear(
                &format!("{layer}.mlp.down_proj"),
                config.intermediate,
                hidden,
                false,
                tiles,
            )?;
            layers.push(LayerWeights {
                input_norm,
                qkv,
                o,
                post_norm,
                gate_up,
                down,
            });
        }
        let norm = self.take(&format!("{prefix}.norm.weight"), &[hidden])?;
        Ok(StackWeights {
            config: config.clone(),
            layers,
            norm,
        })
    }

    /// The checkpoint's tensor `name` of `shape` (`Find`).
    fn find(&mut self, name: &str, shape: &[usize]) -> Result<&'a Tensor, String> {
        let files: &'a ModelFiles = self.files;
        let tensor = files
            .model(name)
            .ok_or_else(|| format!("the checkpoint lacks {name}"))?;
        if !same_shape(&tensor.shape, shape) {
            return Err(format!("{name} has an unexpected shape"));
        }
        self.taken += 1;
        Ok(tensor)
    }

    /// The byte offset of the next `bytes` of the buffer (`Reserve`).
    fn reserve(&mut self, bytes: usize) -> Result<usize, String> {
        self.used = self.used.div_ceil(ALIGNMENT) * ALIGNMENT;
        if self.target.is_some() && self.used + bytes > self.capacity {
            return Err("weights exceed the planned device buffer".into());
        }
        let offset = self.used;
        self.used += bytes;
        Ok(offset)
    }
}

/// Takes every tensor of the generator in its order (`TakeAll`).
fn take_all(take: &mut Uploader<'_>, config: &Config) -> Result<GeneratorWeights, String> {
    let hidden = config.lm.hidden;
    let local = config.encoder.hidden;
    let dit = config.dit.hidden;
    let feat = config.feat_dim;
    let embed = take.take("base_lm.embed_tokens.weight", &[config.lm.vocab, hidden])?;
    let base_lm = take.stack("base_lm", &config.lm, false)?;
    let residual_lm = take.stack("residual_lm", &config.residual_lm, false)?;
    let encoder = take.stack("feat_encoder.encoder", &config.encoder, false)?;
    let special = take.take("feat_encoder.special_token", &[1, 1, 1, local])?;
    let encoder_in = take.linear("feat_encoder.in_proj", feat, local, true, false)?;
    // The DiT's products have 22 rows (the CFG pair of a patch): tiles.
    let dit_stack = take.stack("feat_decoder.estimator.decoder", &config.dit, true)?;
    let estimator = "feat_decoder.estimator.";
    let dit_in = take.linear(&format!("{estimator}in_proj"), feat, dit, true, false)?;
    let dit_cond = take.linear(&format!("{estimator}cond_proj"), feat, dit, true, false)?;
    let dit_out = take.linear(&format!("{estimator}out_proj"), dit, feat, true, false)?;
    let time_1 = take.linear(
        &format!("{estimator}time_mlp.linear_1"),
        dit,
        dit,
        true,
        false,
    )?;
    let time_2 = take.linear(
        &format!("{estimator}time_mlp.linear_2"),
        dit,
        dit,
        true,
        false,
    )?;
    let delta_time_1 = take.linear(
        &format!("{estimator}delta_time_mlp.linear_1"),
        dit,
        dit,
        true,
        false,
    )?;
    let delta_time_2 = take.linear(
        &format!("{estimator}delta_time_mlp.linear_2"),
        dit,
        dit,
        true,
        false,
    )?;
    let enc_to_lm = take.linear("enc_to_lm_proj", local, hidden, true, false)?;
    let lm_to_dit = take.linear("lm_to_dit_proj", hidden, dit, true, false)?;
    let res_to_dit = take.linear("res_to_dit_proj", hidden, dit, true, false)?;
    let fusion = take.linear("fusion_concat_proj", 2 * hidden, hidden, true, false)?;
    let fsq_in = take.linear(
        "fsq_layer.in_proj",
        hidden,
        config.fsq_latent_dim,
        true,
        false,
    )?;
    let fsq_out = take.linear(
        "fsq_layer.out_proj",
        config.fsq_latent_dim,
        hidden,
        true,
        false,
    )?;
    let stop_proj = take.linear("stop_proj", hidden, hidden, true, false)?;
    let stop_head = take.linear("stop_head", hidden, 2, false, false)?;
    Ok(GeneratorWeights {
        base_lm,
        residual_lm,
        encoder,
        dit: dit_stack,
        embed,
        special,
        encoder_in,
        enc_to_lm,
        lm_to_dit,
        res_to_dit,
        fusion,
        fsq_in,
        fsq_out,
        stop_proj,
        stop_head,
        dit_in,
        dit_cond,
        dit_out,
        time_1,
        time_2,
        delta_time_1,
        delta_time_2,
    })
}

/// Owns the device copy of the generator's weights (`DeviceWeights`).
pub struct DeviceWeights {
    storage: DeviceBuffer<u8>,
    weights: GeneratorWeights,
}

impl DeviceWeights {
    /// Checks every tensor's shape against `config` and uploads on `stream`:
    /// Q8_0 matrices as Q8_0 tiles; other GGUF types (F16, F32) and Q8_0
    /// vectors and embedding tables converted to bfloat16 (`Upload`). The
    /// buffer belongs to `stream`; the products read it there.
    pub fn upload(stream: &Stream, files: &ModelFiles, config: &Config) -> Result<Self, String> {
        // Count the bytes of the chosen formats first, then upload.
        let mut count = Uploader::new(files, None);
        take_all(&mut count, config)?;
        if count.taken != files.model_tensors().len() {
            return Err("the checkpoint holds tensors VoxCPM2 does not use".into());
        }
        let mut storage = stream
            .alloc_persistent::<u8>(count.used.max(1))
            .map_err(gpu)?;
        let weights = {
            let mut take = Uploader::new(
                files,
                Some(Target {
                    stream,
                    storage: &mut storage,
                    staging: None,
                    rows: None,
                }),
            );
            take.capacity = count.used;
            take_all(&mut take, config)?
        };
        Ok(Self { storage, weights })
    }

    pub fn weights(&self) -> &GeneratorWeights {
        &self.weights
    }

    /// Bytes of the device buffer (`bytes`).
    pub fn bytes(&self) -> usize {
        self.storage.len()
    }

    /// A vector's bfloat16 values.
    pub fn vector(&self, vector: Vector) -> DeviceSlice<'_, Bf16> {
        self.storage
            .slice(vector.offset..vector.offset + vector.len * 2)
            .cast::<Bf16>()
            .expect("weights are aligned to 16 bytes")
    }

    /// A matrix's bytes in its format.
    pub fn matrix(&self, matrix: Matrix) -> MatrixView<'_> {
        MatrixView {
            format: matrix.format,
            data: self
                .storage
                .slice(matrix.offset..matrix.offset + matrix.bytes),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_follow_the_extension() {
        assert_eq!(format(true, false, 2048), WeightFormat::Q8Tiles);
        assert_eq!(format(true, true, 2048), WeightFormat::Q8Tiles);
        assert_eq!(format(false, true, 1024), WeightFormat::Bf16Tiles);
        assert_eq!(format(false, false, 1024), WeightFormat::Bf16);
        // Columns that don't fill whole tiles stay bfloat16 rows.
        assert_eq!(format(true, true, 1000), WeightFormat::Bf16);
    }
}
