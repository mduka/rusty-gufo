//! Device copies of the trunk and the LiLiCorr drafter (`kernels/rocm/
//! device_model.cpp`). Small F32 tensors keep F32; matrices keep their GGUF
//! block format for the quantized matrix kernels. The weights are copied
//! from the mapped file with gufo-hip's loader; when they don't fit the
//! device heap above its dispatch reserve, they go to coarse-grain managed
//! memory (the C++ `WeightUpload` and `WeightAllocator`).

use std::sync::Arc;

use gufo_hip::{
    DeviceSlice, DeviceTensor, GgmlType, HipError, MappedFile, Pending, Placement, QuantSlice,
    Stream, WeightLoader,
};

use crate::config::BlockType;
use crate::iq2xxs::{IQ2XXS_GRID_WORDS, iq2xxs_grid_bf16};
use crate::weights::{DrafterWeights, ModelWeights, TensorRef};

/// One tensor resident on the device, in its file format.
#[derive(Clone)]
pub struct DeviceWeight {
    pub data: DeviceTensor<u8>,
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    /// NVFP4 per-tensor output scale.
    pub scale: f32,
}

impl DeviceWeight {
    pub fn bytes(&self) -> DeviceSlice<'_, u8> {
        self.data.view()
    }

    /// The tensor as F32 values (small tensors kept in F32); `None` when
    /// its bytes aren't whole, aligned floats.
    pub fn f32(&self) -> Option<DeviceSlice<'_, f32>> {
        self.data.view().cast::<f32>()
    }

    /// The tensor in its block format, for the quantized products.
    pub fn quant(&self) -> QuantSlice<'_> {
        QuantSlice::new(self.data.view(), self.ty)
    }
}

/// The weights of one trunk block; tensors of the other block kinds are
/// absent.
#[derive(Default)]
pub struct DeviceLayer {
    pub ty: Option<BlockType>,
    pub norm: Option<DeviceWeight>,
    // Mamba-2.
    pub ssm_in: Option<DeviceWeight>,
    pub ssm_conv1d: Option<DeviceWeight>,
    pub ssm_conv1d_b: Option<DeviceWeight>,
    pub ssm_dt_b: Option<DeviceWeight>,
    pub ssm_a: Option<DeviceWeight>,
    pub ssm_d: Option<DeviceWeight>,
    pub ssm_norm: Option<DeviceWeight>,
    pub ssm_out: Option<DeviceWeight>,
    // Attention.
    pub attn_q: Option<DeviceWeight>,
    pub attn_k: Option<DeviceWeight>,
    pub attn_v: Option<DeviceWeight>,
    pub attn_out: Option<DeviceWeight>,
    // LatentMoE.
    pub num_experts: usize,
    pub experts_used: usize,
    pub expert_ff: usize,
    pub router: Option<DeviceWeight>,
    pub router_bias: Option<DeviceWeight>,
    pub latent_down: Option<DeviceWeight>,
    pub latent_up: Option<DeviceWeight>,
    pub up_exps: Option<DeviceWeight>,
    pub down_exps: Option<DeviceWeight>,
    pub up_shexp: Option<DeviceWeight>,
    pub down_shexp: Option<DeviceWeight>,
}

/// Device copies of every trunk tensor.
pub struct DeviceModel {
    /// The IQ2_XXS grid as BF16 pairs for the vector products; immutable,
    /// so the shared expert reads it on the side stream too.
    pub iq2xxs_grid: DeviceTensor<u32>,
    pub token_embd: DeviceWeight,
    pub output_norm: DeviceWeight,
    pub output: DeviceWeight,
    pub layers: Vec<DeviceLayer>,
    pub resident_bytes: usize,
}

fn hip(error: HipError) -> String {
    error.to_string()
}

/// Queues tensors on one loader and hands out their device copies.
struct Upload<'s> {
    loader: WeightLoader<'s>,
    files: &'s [Arc<MappedFile>],
    queued: Vec<(Pending<u8>, GgmlType, usize, usize, f32)>,
    bytes: usize,
}

impl<'s> Upload<'s> {
    /// One loader over `total` bytes: device memory, or managed memory when
    /// they don't fit the heap above its dispatch reserve.
    fn new(stream: &'s Stream, files: &'s [Arc<MappedFile>], total: usize) -> Result<Self, String> {
        let (free, _) = gufo_hip::device_memory().map_err(hip)?;
        let managed = !gufo_hip::heap_fits_above_reserve(free as u64, total as u64);
        Ok(Self {
            loader: WeightLoader::new(stream, Placement::DirectCopy).managed_arenas(managed),
            files,
            queued: Vec::new(),
            bytes: 0,
        })
    }

    /// Queues `t` and returns its index.
    fn add(&mut self, t: &TensorRef) -> Result<usize, String> {
        let file = self
            .files
            .get(t.shard as usize)
            .ok_or_else(|| format!("device allocation failed for {}", t.name))?;
        let bytes = t.size_bytes();
        let offset = usize::try_from(t.file_offset)
            .map_err(|_| format!("device allocation failed for {}", t.name))?;
        let pending = self
            .loader
            .add::<u8>(file, offset, bytes)
            .map_err(|e| format!("device allocation failed for {}: {e}", t.name))?;
        self.queued.push((
            pending,
            t.ty,
            (t.rows * t.experts) as usize,
            t.cols as usize,
            t.scale,
        ));
        self.bytes += bytes;
        Ok(self.queued.len() - 1)
    }

    /// Runs the copies; the weights in queue order.
    fn finish(self) -> Result<(Vec<DeviceWeight>, usize), String> {
        let Self {
            loader,
            queued,
            bytes,
            ..
        } = self;
        let loaded = loader
            .finish()
            .map_err(|e| format!("weight upload failed: {e}"))?;
        let weights = queued
            .into_iter()
            .map(|(pending, ty, rows, cols, scale)| DeviceWeight {
                data: loaded.get(pending),
                ty,
                rows,
                cols,
                scale,
            })
            .collect();
        Ok((weights, bytes))
    }
}

impl DeviceModel {
    /// Uploads the trunk (`DeviceModel::Upload`): checks the formats the
    /// kernels support, then copies every tensor.
    pub fn upload(
        weights: &ModelWeights,
        files: &[Arc<MappedFile>],
        stream: &Stream,
    ) -> Result<Self, String> {
        // Only Mamba projections carry a per-tensor scale; the forward pass
        // applies it there and nowhere else.
        for l in &weights.layers {
            for t in [
                &l.attn_q,
                &l.attn_k,
                &l.attn_v,
                &l.attn_out,
                &l.latent_down,
                &l.latent_up,
                &l.up_exps,
                &l.down_exps,
                &l.up_shexp,
                &l.down_shexp,
            ] {
                if !t.is_empty() && t.scale != 1.0 {
                    return Err(format!("unexpected per-tensor scale on {}", t.name));
                }
            }
        }
        if weights.token_embd.ty != GgmlType::Q8_0 || weights.output.scale != 1.0 {
            return Err("the embedding must be Q8_0 and the head unscaled".into());
        }
        for l in &weights.layers {
            for t in [&l.up_exps, &l.down_exps, &l.up_shexp, &l.down_shexp] {
                if !t.is_empty() && t.ty != GgmlType::Iq2Xxs {
                    return Err(format!("experts must be IQ2_XXS: {}", t.name));
                }
            }
            for t in [&l.ssm_in, &l.ssm_out] {
                if !t.is_empty() && t.ty != GgmlType::Nvfp4 {
                    return Err(format!("Mamba projections must be NVFP4: {}", t.name));
                }
            }
        }
        let grid = iq2xxs_grid_bf16();
        let mut iq2xxs_grid = stream
            .alloc::<u32>(IQ2XXS_GRID_WORDS)
            .map_err(|_| "could not upload the IQ2_XXS grid".to_string())?;
        stream
            .upload(&grid, iq2xxs_grid.as_mut_slice())
            .map_err(|_| "could not upload the IQ2_XXS grid".to_string())?;
        let iq2xxs_grid = iq2xxs_grid
            .freeze(stream)
            .map_err(|_| "could not upload the IQ2_XXS grid".to_string())?;

        // Every tensor of the trunk in the C++ walk's order.
        let mut tensors: Vec<&TensorRef> =
            vec![&weights.token_embd, &weights.output_norm, &weights.output];
        for l in &weights.layers {
            tensors.push(&l.norm);
            tensors.extend(match l.ty {
                BlockType::Mamba => vec![
                    &l.ssm_in,
                    &l.ssm_conv1d,
                    &l.ssm_conv1d_b,
                    &l.ssm_dt_b,
                    &l.ssm_a,
                    &l.ssm_d,
                    &l.ssm_norm,
                    &l.ssm_out,
                ],
                BlockType::Attention => vec![&l.attn_q, &l.attn_k, &l.attn_v, &l.attn_out],
                BlockType::Moe => vec![
                    &l.router,
                    &l.router_bias,
                    &l.latent_down,
                    &l.latent_up,
                    &l.up_exps,
                    &l.down_exps,
                    &l.up_shexp,
                    &l.down_shexp,
                ],
            });
        }
        let total = tensors.iter().map(|t| t.size_bytes()).sum();
        let mut upload = Upload::new(stream, files, total)?;
        for t in &tensors {
            upload.add(t)?;
        }
        let (weights_out, bytes) = upload.finish()?;
        let mut it = weights_out.into_iter();
        let mut next = || it.next().ok_or_else(|| "weight upload failed".to_string());
        let token_embd = next()?;
        let output_norm = next()?;
        let output = next()?;
        let mut layers = Vec::with_capacity(weights.layers.len());
        for (il, l) in weights.layers.iter().enumerate() {
            let mut d = DeviceLayer {
                ty: Some(l.ty),
                norm: Some(next()?),
                ..DeviceLayer::default()
            };
            match l.ty {
                BlockType::Mamba => {
                    d.ssm_in = Some(next()?);
                    d.ssm_conv1d = Some(next()?);
                    d.ssm_conv1d_b = Some(next()?);
                    d.ssm_dt_b = Some(next()?);
                    d.ssm_a = Some(next()?);
                    d.ssm_d = Some(next()?);
                    d.ssm_norm = Some(next()?);
                    d.ssm_out = Some(next()?);
                }
                BlockType::Attention => {
                    d.attn_q = Some(next()?);
                    d.attn_k = Some(next()?);
                    d.attn_v = Some(next()?);
                    d.attn_out = Some(next()?);
                }
                BlockType::Moe => {
                    d.num_experts = l.num_experts as usize;
                    d.experts_used = weights.config.experts_used[il] as usize;
                    d.expert_ff = weights.config.expert_ff[il] as usize;
                    d.router = Some(next()?);
                    d.router_bias = Some(next()?);
                    d.latent_down = Some(next()?);
                    d.latent_up = Some(next()?);
                    d.up_exps = Some(next()?);
                    d.down_exps = Some(next()?);
                    d.up_shexp = Some(next()?);
                    d.down_shexp = Some(next()?);
                }
            }
            layers.push(d);
        }
        Ok(Self {
            iq2xxs_grid,
            token_embd,
            output_norm,
            output,
            layers,
            resident_bytes: bytes,
        })
    }
}

/// Device copies of one drafter block.
pub struct DeviceDrafterLayer {
    pub attn_norm: DeviceWeight,
    pub attn_conv_base: DeviceWeight,
    pub attn_conv_proj: DeviceWeight,
    pub attn_q: DeviceWeight,
    pub attn_k: DeviceWeight,
    pub attn_v: DeviceWeight,
    pub attn_q_norm: DeviceWeight,
    pub attn_k_norm: DeviceWeight,
    pub attn_out: DeviceWeight,
    pub attn_sinks: DeviceWeight,
    pub ffn_norm: DeviceWeight,
    pub ffn_conv_base: DeviceWeight,
    pub ffn_conv_proj: DeviceWeight,
    pub ffn_gate: DeviceWeight,
    pub ffn_up: DeviceWeight,
    pub ffn_down: DeviceWeight,
}

/// Device copies of one LiLiCorr layer.
pub struct DeviceLiliCorrLayer {
    pub attn_norm: DeviceWeight,
    pub mlp_norm: DeviceWeight,
    pub in_proj: DeviceWeight,
    pub in_proj_b: DeviceWeight,
    pub out_proj: DeviceWeight,
    pub out_proj_b: DeviceWeight,
    pub mlp_up: DeviceWeight,
    pub mlp_up_b: DeviceWeight,
    pub mlp_down: DeviceWeight,
    pub mlp_down_b: DeviceWeight,
}

/// Device copies of the LiLiCorr drafter, in their file formats.
pub struct DeviceDrafter {
    pub fc: DeviceWeight,
    pub enc_norm: DeviceWeight,
    pub output_norm: DeviceWeight,
    pub layers: Vec<DeviceDrafterLayer>,
    pub token_proj: DeviceWeight,
    pub token_proj_b: DeviceWeight,
    pub pass_proj: DeviceWeight,
    pub pass_proj_b: DeviceWeight,
    pub context_proj: DeviceWeight,
    pub context_proj_b: DeviceWeight,
    pub feature_norm: DeviceWeight,
    pub feature_norm_b: DeviceWeight,
    pub feature_up: DeviceWeight,
    pub feature_up_b: DeviceWeight,
    pub feature_down: DeviceWeight,
    pub feature_down_b: DeviceWeight,
    pub slot_embd: DeviceWeight,
    pub rank_embd: DeviceWeight,
    pub attention_bias: DeviceWeight,
    pub lc_layers: Vec<DeviceLiliCorrLayer>,
    pub lc_output_norm: DeviceWeight,
    pub anchor_norm: DeviceWeight,
    pub factor_in: DeviceWeight,
    pub factor_in_b: DeviceWeight,
    pub out_head: DeviceWeight,
    pub out_head_b: DeviceWeight,
    pub in_head: DeviceWeight,
    pub in_head_b: DeviceWeight,
    pub anchor_head: DeviceWeight,
    pub anchor_head_b: DeviceWeight,
    pub resident_bytes: usize,
}

impl DeviceDrafter {
    /// Uploads the drafter (`DeviceDrafter::Upload`).
    pub fn upload(
        w: &DrafterWeights,
        files: &[Arc<MappedFile>],
        stream: &Stream,
    ) -> Result<Self, String> {
        let mut tensors: Vec<&TensorRef> = vec![
            &w.fc,
            &w.enc_norm,
            &w.output_norm,
            &w.token_proj,
            &w.token_proj_b,
            &w.pass_proj,
            &w.pass_proj_b,
            &w.context_proj,
            &w.context_proj_b,
            &w.feature_norm,
            &w.feature_norm_b,
            &w.feature_up,
            &w.feature_up_b,
            &w.feature_down,
            &w.feature_down_b,
            &w.slot_embd,
            &w.rank_embd,
            &w.attention_bias,
            &w.lc_output_norm,
            &w.anchor_norm,
            &w.factor_in,
            &w.factor_in_b,
            &w.out_head,
            &w.out_head_b,
            &w.in_head,
            &w.in_head_b,
            &w.anchor_head,
            &w.anchor_head_b,
        ];
        for l in &w.layers {
            tensors.extend([
                &l.attn_norm,
                &l.attn_conv_base,
                &l.attn_conv_proj,
                &l.attn_q,
                &l.attn_k,
                &l.attn_v,
                &l.attn_q_norm,
                &l.attn_k_norm,
                &l.attn_out,
                &l.attn_sinks,
                &l.ffn_norm,
                &l.ffn_conv_base,
                &l.ffn_conv_proj,
                &l.ffn_gate,
                &l.ffn_up,
                &l.ffn_down,
            ]);
        }
        for l in &w.lc_layers {
            tensors.extend([
                &l.attn_norm,
                &l.mlp_norm,
                &l.in_proj,
                &l.in_proj_b,
                &l.out_proj,
                &l.out_proj_b,
                &l.mlp_up,
                &l.mlp_up_b,
                &l.mlp_down,
                &l.mlp_down_b,
            ]);
        }
        let total = tensors.iter().map(|t| t.size_bytes()).sum();
        let mut upload = Upload::new(stream, files, total)?;
        for t in &tensors {
            upload.add(t)?;
        }
        let (weights, bytes) = upload.finish().map_err(|e| {
            if e.is_empty() {
                "drafter upload failed".into()
            } else {
                e
            }
        })?;
        let mut it = weights.into_iter();
        let mut next = || it.next().ok_or_else(|| "drafter upload failed".to_string());
        let mut d = Self {
            fc: next()?,
            enc_norm: next()?,
            output_norm: next()?,
            token_proj: next()?,
            token_proj_b: next()?,
            pass_proj: next()?,
            pass_proj_b: next()?,
            context_proj: next()?,
            context_proj_b: next()?,
            feature_norm: next()?,
            feature_norm_b: next()?,
            feature_up: next()?,
            feature_up_b: next()?,
            feature_down: next()?,
            feature_down_b: next()?,
            slot_embd: next()?,
            rank_embd: next()?,
            attention_bias: next()?,
            lc_output_norm: next()?,
            anchor_norm: next()?,
            factor_in: next()?,
            factor_in_b: next()?,
            out_head: next()?,
            out_head_b: next()?,
            in_head: next()?,
            in_head_b: next()?,
            anchor_head: next()?,
            anchor_head_b: next()?,
            layers: Vec::with_capacity(w.layers.len()),
            lc_layers: Vec::with_capacity(w.lc_layers.len()),
            resident_bytes: bytes,
        };
        for _ in &w.layers {
            d.layers.push(DeviceDrafterLayer {
                attn_norm: next()?,
                attn_conv_base: next()?,
                attn_conv_proj: next()?,
                attn_q: next()?,
                attn_k: next()?,
                attn_v: next()?,
                attn_q_norm: next()?,
                attn_k_norm: next()?,
                attn_out: next()?,
                attn_sinks: next()?,
                ffn_norm: next()?,
                ffn_conv_base: next()?,
                ffn_conv_proj: next()?,
                ffn_gate: next()?,
                ffn_up: next()?,
                ffn_down: next()?,
            });
        }
        for _ in &w.lc_layers {
            d.lc_layers.push(DeviceLiliCorrLayer {
                attn_norm: next()?,
                mlp_norm: next()?,
                in_proj: next()?,
                in_proj_b: next()?,
                out_proj: next()?,
                out_proj_b: next()?,
                mlp_up: next()?,
                mlp_up_b: next()?,
                mlp_down: next()?,
                mlp_down_b: next()?,
            });
        }
        Ok(d)
    }
}
