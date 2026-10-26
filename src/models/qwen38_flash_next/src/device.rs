//! The trunk (and the MTP draft block) in device memory, a port of
//! `DeviceModel::Upload` (`kernels/rocm/device_model.cpp`). Weights keep
//! their GGUF encoding; projections of one input are stacked into one Q8_0
//! matrix, the F32 router and alpha/beta stacks narrowed to F16, and the MTP
//! input projection split into its two halves. The n-gram table stays on
//! disk.

use std::sync::Arc;

use gufo_formats::GgmlType;
use gufo_hip::{
    DeviceTensor, F16, LoadedWeights, MappedFile, Pending, Placement, QuantSlice, QuantTensor,
    Stream, WeightLoader,
};
use gufo_qwen38_flash_next_kernels as k;

use crate::config::Config;
use crate::weights::{LayerRefs, MixerKind, MixerRefs, ModelTensors, MtpTensors, TensorRef};

/// Zero bytes after every weight: the quantized tier reads whole 256-value
/// k-iterations, past the last row of a matrix whose rows are only a
/// multiple of 32 values (`kTailMargin`).
pub const TAIL: usize = 4096;

/// A weight matrix in its GGUF encoding, followed by [`TAIL`] zero bytes.
#[derive(Clone)]
pub struct Matrix {
    weights: QuantTensor,
    pub rows: u32,
    pub cols: u32,
    pub experts: u32,
}

impl Matrix {
    pub fn format(&self) -> GgmlType {
        self.weights.format()
    }

    /// The encoded matrix with its tail.
    pub fn view(&self) -> QuantSlice<'_> {
        self.weights.view()
    }

    /// An F32 matrix as floats (the fused mixer inject).
    pub fn floats(&self) -> Option<gufo_hip::DeviceSlice<'_, f32>> {
        (self.format() == GgmlType::F32).then_some(())?;
        let len = self.rows as usize * self.cols as usize * self.experts as usize;
        self.weights.bytes().slice(0..len * 4).cast::<f32>()
    }

    /// The last `rows` rows of a single matrix, sharing its bytes and tail.
    pub fn last_rows(&self, rows: u32) -> Option<Matrix> {
        let (values, bytes) = self.format().block();
        if self.experts != 1 || rows > self.rows || !(self.cols as usize).is_multiple_of(values) {
            return None;
        }
        let row_bytes = self.cols as usize / values * bytes;
        Some(Self {
            weights: self.weights.tail((self.rows - rows) as usize * row_bytes)?,
            rows,
            cols: self.cols,
            experts: 1,
        })
    }

    /// The first `rows` rows of a single matrix, sharing its bytes; the rows
    /// after them stand in for its tail.
    pub fn first_rows(&self, rows: u32) -> Option<Matrix> {
        let (values, bytes) = self.format().block();
        if self.experts != 1 || rows >= self.rows || !(self.cols as usize).is_multiple_of(values) {
            return None;
        }
        let row_bytes = self.cols as usize / values * bytes;
        Some(Self {
            weights: self.weights.head(rows as usize * row_bytes + TAIL)?,
            rows,
            cols: self.cols,
            experts: 1,
        })
    }

    /// Bytes of one row of a single matrix.
    pub fn row_bytes(&self) -> Option<usize> {
        let (values, bytes) = self.format().block();
        (self.experts == 1 && (self.cols as usize).is_multiple_of(values))
            .then(|| self.cols as usize / values * bytes)
    }

    fn quant(weights: QuantTensor, rows: u64, cols: u64, experts: u64) -> Self {
        Self {
            weights,
            rows: rows as u32,
            cols: cols as u32,
            experts: experts as u32,
        }
    }
}

/// One hyper-connection mixer.
#[derive(Clone)]
pub struct Mixer {
    pub norm: DeviceTensor<f32>,
    pub down: Matrix,
    pub up: Matrix,
    pub inject: Option<Matrix>,
}

pub struct Linear {
    /// q|k|v and z rows stacked (both Q8_0): [hidden -> channels + value_dim].
    pub ssm_in: Option<Matrix>,
    /// Separate projections otherwise.
    pub qkv: Option<Matrix>,
    pub gate: Option<Matrix>,
    pub conv1d: DeviceTensor<f32>,
    /// alpha and beta rows stacked, narrowed to F16: [hidden -> 2 * v_heads].
    pub alpha_beta: Matrix,
    pub dt: DeviceTensor<f32>,
    pub a: DeviceTensor<f32>,
    pub norm: DeviceTensor<f32>,
    pub out: Matrix,
}

pub struct Attention {
    /// [q|gate ; k ; v] rows stacked when all three are Q8_0.
    pub qkv: Option<Matrix>,
    /// Separate projections otherwise.
    pub q: Option<Matrix>,
    pub k: Option<Matrix>,
    pub v: Option<Matrix>,
    pub out: Matrix,
    pub q_norm: DeviceTensor<f32>,
    pub k_norm: DeviceTensor<f32>,
    pub indexer_q: Matrix,
    pub indexer_k: Matrix,
    pub indexer_q_norm: DeviceTensor<f32>,
    pub indexer_k_norm: DeviceTensor<f32>,
}

pub enum Block {
    Linear(Linear),
    Attention(Attention),
}

impl Block {
    /// The first projection of the mixer output the block reads.
    pub fn first_projection(&self) -> Option<&Matrix> {
        match self {
            Block::Linear(l) => l.ssm_in.as_ref().or(l.qkv.as_ref()),
            Block::Attention(a) => a.qkv.as_ref().or(a.q.as_ref()),
        }
    }
}

pub struct Ple {
    pub key: Matrix,
    pub value: Matrix,
    pub norm_key: DeviceTensor<f32>,
    pub norm_query: DeviceTensor<f32>,
    pub norm_conv: DeviceTensor<f32>,
    pub conv1d: DeviceTensor<f32>,
}

pub struct Moe {
    /// Router rows and the shared-expert gate row, narrowed to F16:
    /// [hidden -> num_experts + 1].
    pub router: Matrix,
    pub gate_exps: Matrix,
    pub up_exps: Matrix,
    pub down_exps: Matrix,
    pub shexp_gate: Matrix,
    pub shexp_up: Matrix,
    pub shexp_down: Matrix,
}

pub struct Nextn {
    pub enorm: DeviceTensor<f32>,
    pub hnorm: DeviceTensor<f32>,
    /// The two halves of `nextn.eh_proj`.
    pub fc_embedding: Matrix,
    pub fc_hidden: Matrix,
    pub head: Mixer,
}

pub struct Layer {
    pub hc_attn: Mixer,
    pub hc_ffn: Mixer,
    pub block: Block,
    pub ple: Option<Ple>,
    pub moe: Moe,
    pub nextn: Option<Nextn>,
}

impl Layer {
    pub fn linear(&self) -> bool {
        matches!(self.block, Block::Linear(_))
    }
}

pub struct DeviceModel {
    pub config: Config,
    pub token_embd: Matrix,
    pub output: Matrix,
    pub hc_head: Mixer,
    pub layers: Vec<Layer>,
    pub mtp: Option<Layer>,
    /// Device bytes as the C++ upload counts them (every tensor plus its tail).
    pub resident_bytes: usize,
    /// Widest K among the BF16/F16 matrices (hipBLASLt activation staging).
    pub max_half_cols: usize,
    /// Widest K among the dense Q8_0 matrices (decode quantization).
    pub max_q8_cols: usize,
    /// The small MTP draft head, when the draft scores a vocabulary subset.
    pub draft: Option<DraftHead>,
}

/// The leading output rows the MTP draft head scores in place, and the
/// special tokens above them that start every session's draft table
/// (see [`crate::draft_vocab`]).
pub struct DraftHead {
    pub prefix: Matrix,
    pub specials: Vec<i32>,
    /// Bytes of one output row.
    pub row_bytes: usize,
}

impl DeviceModel {
    /// Lets drafts score output rows `[0, prefix_rows)` in place plus a
    /// per-session table that starts with `specials` (`SetDraftVocab`).
    /// Draft proposals only steer speculation: the target still verifies
    /// every token, so greedy text is unchanged; sampled text can differ,
    /// since a smaller head changes acceptance and so the later draws.
    pub fn set_draft_vocab(&mut self, prefix_rows: u32, specials: Vec<i32>) -> Result<(), String> {
        let row_bytes = self
            .output
            .row_bytes()
            .ok_or("draft vocabulary needs a row-addressable output head")?;
        let prefix = self
            .output
            .first_rows(prefix_rows)
            .filter(|_| prefix_rows > 0)
            .ok_or("draft vocabulary prefix must leave rows out")?;
        if specials
            .iter()
            .any(|&id| id < prefix_rows as i32 || id as u32 >= self.output.rows)
        {
            return Err("draft special tokens must lie above the prefix".into());
        }
        self.draft = Some(DraftHead {
            prefix,
            specials,
            row_bytes,
        });
        Ok(())
    }

    /// Uploads the trunk from `files` (its shards, in order) and, with
    /// `mtp`, the draft block from the sidecar's files.
    pub fn upload(
        stream: &Stream,
        model: &ModelTensors,
        files: &[Arc<MappedFile>],
        mtp: Option<(&MtpTensors, &[Arc<MappedFile>])>,
    ) -> Result<Self, String> {
        // The production embedding, dense and routed kernels do not read
        // Q6_K (only the CPU reference does).
        let supported = |t: &TensorRef| {
            if t.ty == GgmlType::Q6K {
                Err(format!("unsupported HIP tensor format Q6_K: {}", t.name))
            } else {
                Ok(())
            }
        };
        let layer_supported = |l: &LayerRefs| {
            supported(&l.moe.gate_exps)?;
            supported(&l.moe.up_exps)?;
            supported(&l.moe.down_exps)
        };
        supported(&model.token_embd)?;
        supported(&model.output)?;
        for l in &model.layers {
            layer_supported(l)?;
        }
        if let Some((m, _)) = mtp {
            layer_supported(&m.block)?;
        }
        let mut up = Uploader {
            stream,
            files,
            resident_bytes: 0,
            max_half_cols: 1,
            max_q8_cols: 32,
        };
        // One loader for every tensor: its arenas are allocated while the
        // earlier ones fill, like the C++ upload's single stager.
        let all = up.group(|g| {
            let token_embd = g.matrix(&model.token_embd)?;
            let tied = model.output == model.token_embd;
            let output = if tied {
                None
            } else {
                Some(g.matrix(&model.output)?)
            };
            let hc_head = g.mixer(&model.hc_head)?;
            let layers = model
                .layers
                .iter()
                .map(|l| g.layer(l))
                .collect::<Result<Vec<_>, _>>()?;
            let mtp = match mtp {
                Some((m, mtp_files)) => {
                    // The sidecar's shards index its own files.
                    let trunk = std::mem::replace(&mut g.up.files, mtp_files);
                    let layer = g.layer(&m.block);
                    g.up.files = trunk;
                    Some(layer?)
                }
                None => None,
            };
            Ok(((token_embd, output, hc_head), layers, mtp))
        })?;
        let ((token_embd, output, hc_head), layers, mtp) = up.redeem(all);
        let output = output.unwrap_or_else(|| token_embd.clone());
        Ok(Self {
            config: model.config.clone(),
            token_embd,
            output,
            hc_head,
            layers,
            mtp,
            resident_bytes: up.resident_bytes,
            max_half_cols: up.max_half_cols,
            max_q8_cols: up.max_q8_cols,
            draft: None,
        })
    }
}

/// A tensor queued with a group's loader, redeemed once it finished.
enum Queued {
    Matrix(Pending<u8>, GgmlType, u64, u64, u64),
    Vector(Pending<f32>),
    /// F32 stacks narrowed to F16 after loading.
    Narrow(Pending<f32>, u64, u64),
    /// `nextn.eh_proj`, split into its two halves after loading.
    Split(Pending<u8>, TensorRef),
}

/// Queued tensors of one group: loaded tensors after `finish`.
struct Group<'u, 's> {
    up: &'u mut Uploader<'s>,
    loader: WeightLoader<'s>,
    /// The F32 and combined sources of conversions: their arenas go once
    /// the converted tensors exist.
    sources: WeightLoader<'s>,
    queued: Vec<Queued>,
}

/// A handle into a group's queue.
#[derive(Clone, Copy)]
struct Slot(usize);

/// What a group's closure returns: slots, redeemed into tensors.
trait Redeem {
    type Output;
    fn redeem(self, loaded: &Loaded) -> Self::Output;
}

struct Loaded {
    matrices: Vec<Option<Matrix>>,
    vectors: Vec<Option<DeviceTensor<f32>>>,
    splits: Vec<Option<(Matrix, Matrix)>>,
}

impl Redeem for Slot {
    type Output = Matrix;
    fn redeem(self, loaded: &Loaded) -> Matrix {
        loaded.matrices[self.0].clone().expect("a matrix slot")
    }
}

struct VectorSlot(usize);

impl Redeem for VectorSlot {
    type Output = DeviceTensor<f32>;
    fn redeem(self, loaded: &Loaded) -> DeviceTensor<f32> {
        loaded.vectors[self.0].clone().expect("a vector slot")
    }
}

struct SplitSlot(usize);

impl Redeem for SplitSlot {
    type Output = (Matrix, Matrix);
    fn redeem(self, loaded: &Loaded) -> (Matrix, Matrix) {
        loaded.splits[self.0].clone().expect("a split slot")
    }
}

impl<T: Redeem> Redeem for Option<T> {
    type Output = Option<T::Output>;
    fn redeem(self, loaded: &Loaded) -> Self::Output {
        self.map(|s| s.redeem(loaded))
    }
}

struct MixerSlots {
    norm: VectorSlot,
    down: Slot,
    up: Slot,
    inject: Option<Slot>,
}

impl Redeem for MixerSlots {
    type Output = Mixer;
    fn redeem(self, loaded: &Loaded) -> Mixer {
        Mixer {
            norm: self.norm.redeem(loaded),
            down: self.down.redeem(loaded),
            up: self.up.redeem(loaded),
            inject: self.inject.redeem(loaded),
        }
    }
}

impl<T: Redeem> Redeem for Vec<T> {
    type Output = Vec<T::Output>;
    fn redeem(self, loaded: &Loaded) -> Self::Output {
        self.into_iter().map(|t| t.redeem(loaded)).collect()
    }
}

impl<A: Redeem, B: Redeem, C: Redeem> Redeem for (A, B, C) {
    type Output = (A::Output, B::Output, C::Output);
    fn redeem(self, loaded: &Loaded) -> Self::Output {
        (
            self.0.redeem(loaded),
            self.1.redeem(loaded),
            self.2.redeem(loaded),
        )
    }
}

struct Uploader<'s> {
    stream: &'s Stream,
    files: &'s [Arc<MappedFile>],
    resident_bytes: usize,
    max_half_cols: usize,
    max_q8_cols: usize,
}

/// Loaded group results kept until redeemed.
struct Finished<T> {
    slots: T,
    loaded: Loaded,
}

impl<'s> Uploader<'s> {
    /// Queues a group of tensors with `queue`, loads them into one arena,
    /// then narrows and splits what needs it.
    fn group<T>(
        &mut self,
        queue: impl FnOnce(&mut Group<'_, 's>) -> Result<T, String>,
    ) -> Result<Finished<T>, String> {
        let mut group = Group {
            loader: WeightLoader::new(self.stream, Placement::DirectCopy),
            sources: WeightLoader::new(self.stream, Placement::DirectCopy),
            queued: Vec::new(),
            up: self,
        };
        let slots = queue(&mut group)?;
        let Group {
            loader,
            sources,
            queued,
            ..
        } = group;
        let loaded = loader
            .finish()
            .map_err(|e| format!("weight upload failed: {e}"))?;
        let sources = sources
            .finish()
            .map_err(|e| format!("weight upload failed: {e}"))?;
        let loaded = self.convert(queued, loaded, sources)?;
        Ok(Finished { slots, loaded })
    }

    fn redeem<T: Redeem>(&self, finished: Finished<T>) -> T::Output {
        finished.slots.redeem(&finished.loaded)
    }

    /// Redeems every queued tensor: narrowed F16 stacks and split halves get
    /// their own buffers, after which the F32 and combined sources, still in
    /// the loader's arena, are released with it.
    fn convert(
        &mut self,
        queued: Vec<Queued>,
        loaded: LoadedWeights,
        sources: LoadedWeights,
    ) -> Result<Loaded, String> {
        let error = |e: gufo_hip::HipError| format!("weight conversion failed: {e}");
        let mut out = Loaded {
            matrices: Vec::new(),
            vectors: Vec::new(),
            splits: Vec::new(),
        };
        for q in queued {
            let (mut matrix, mut vector, mut split) = (None, None, None);
            match q {
                Queued::Matrix(pending, format, rows, cols, experts) => {
                    let tensor = loaded.get(pending);
                    matrix = Some(Matrix::quant(
                        QuantTensor::new(tensor, format),
                        rows,
                        cols,
                        experts,
                    ));
                }
                Queued::Vector(pending) => vector = Some(loaded.get(pending)),
                Queued::Narrow(pending, rows, cols) => {
                    let source = sources.get(pending);
                    let count = (rows * cols) as usize;
                    let mut half = self
                        .stream
                        .alloc_persistent::<u8>(count * 2 + TAIL)
                        .map_err(error)?;
                    let target = half
                        .slice_mut(0..count * 2)
                        .cast::<F16>()
                        .ok_or("misaligned F16 stack")?;
                    k::narrow_f16(self.stream, source.slice(0..count), target, count)
                        .map_err(|e| format!("weight conversion failed: {e}"))?;
                    self.stream
                        .fill_bytes(half.slice_mut(count * 2..count * 2 + TAIL), 0)
                        .map_err(error)?;
                    let tensor = half.freeze(self.stream).map_err(error)?;
                    self.resident_bytes += count * 2 + TAIL;
                    self.max_half_cols = self.max_half_cols.max(cols as usize);
                    matrix = Some(Matrix::quant(
                        QuantTensor::new(tensor, GgmlType::F16),
                        rows,
                        cols,
                        1,
                    ));
                }
                Queued::Split(pending, t) => {
                    // GGUF packs [fc_embedding | fc_hidden] across each row:
                    // split on a block boundary without touching a weight.
                    let combined = sources.get(pending);
                    let row_bytes = t.row_bytes() / 2;
                    let rows = t.rows as usize;
                    let part_bytes = row_bytes * rows;
                    let mut halves = Vec::with_capacity(2);
                    for part in 0..2 {
                        let mut buffer = self
                            .stream
                            .alloc_persistent::<u8>(part_bytes + TAIL)
                            .map_err(error)?;
                        self.stream
                            .copy_2d(
                                combined.slice(part * row_bytes..t.bytes),
                                2 * row_bytes,
                                buffer.slice_mut(0..part_bytes),
                                row_bytes,
                                row_bytes,
                                rows,
                            )
                            .map_err(error)?;
                        self.stream
                            .fill_bytes(buffer.slice_mut(part_bytes..part_bytes + TAIL), 0)
                            .map_err(error)?;
                        let tensor = buffer.freeze(self.stream).map_err(error)?;
                        self.resident_bytes += part_bytes + TAIL;
                        halves.push(Matrix::quant(
                            QuantTensor::new(tensor, t.ty),
                            t.rows,
                            t.cols / 2,
                            1,
                        ));
                    }
                    let fc_hidden = halves.pop().expect("two halves");
                    let fc_embedding = halves.pop().expect("two halves");
                    if t.ty == GgmlType::Q8_0 {
                        self.max_q8_cols = self.max_q8_cols.max(t.cols as usize / 2);
                    }
                    split = Some((fc_embedding, fc_hidden));
                }
            }
            out.matrices.push(matrix);
            out.vectors.push(vector);
            out.splits.push(split);
        }
        // The sources above were read by stream-ordered work; the freezes
        // waited for it before their arenas can go.
        drop((loaded, sources));
        Ok(out)
    }
}

struct LayerSlots {
    hc_attn: MixerSlots,
    hc_ffn: MixerSlots,
    block: BlockSlots,
    ple: Option<PleSlots>,
    moe: MoeSlots,
    nextn: Option<NextnSlots>,
}

enum BlockSlots {
    Linear {
        ssm_in: Option<Slot>,
        qkv: Option<Slot>,
        gate: Option<Slot>,
        conv1d: VectorSlot,
        alpha_beta: Slot,
        dt: VectorSlot,
        a: VectorSlot,
        norm: VectorSlot,
        out: Slot,
    },
    Attention {
        qkv: Option<Slot>,
        q: Option<Slot>,
        k: Option<Slot>,
        v: Option<Slot>,
        out: Slot,
        q_norm: VectorSlot,
        k_norm: VectorSlot,
        indexer_q: Slot,
        indexer_k: Slot,
        indexer_q_norm: VectorSlot,
        indexer_k_norm: VectorSlot,
    },
}

struct PleSlots {
    key: Slot,
    value: Slot,
    norm_key: VectorSlot,
    norm_query: VectorSlot,
    norm_conv: VectorSlot,
    conv1d: VectorSlot,
}

struct MoeSlots {
    router: Slot,
    gate_exps: Slot,
    up_exps: Slot,
    down_exps: Slot,
    shexp_gate: Slot,
    shexp_up: Slot,
    shexp_down: Slot,
}

struct NextnSlots {
    enorm: VectorSlot,
    hnorm: VectorSlot,
    eh_proj: SplitSlot,
    head: MixerSlots,
}

impl Redeem for LayerSlots {
    type Output = Layer;
    fn redeem(self, d: &Loaded) -> Layer {
        let block = match self.block {
            BlockSlots::Linear {
                ssm_in,
                qkv,
                gate,
                conv1d,
                alpha_beta,
                dt,
                a,
                norm,
                out,
            } => Block::Linear(Linear {
                ssm_in: ssm_in.redeem(d),
                qkv: qkv.redeem(d),
                gate: gate.redeem(d),
                conv1d: conv1d.redeem(d),
                alpha_beta: alpha_beta.redeem(d),
                dt: dt.redeem(d),
                a: a.redeem(d),
                norm: norm.redeem(d),
                out: out.redeem(d),
            }),
            BlockSlots::Attention {
                qkv,
                q,
                k,
                v,
                out,
                q_norm,
                k_norm,
                indexer_q,
                indexer_k,
                indexer_q_norm,
                indexer_k_norm,
            } => Block::Attention(Attention {
                qkv: qkv.redeem(d),
                q: q.redeem(d),
                k: k.redeem(d),
                v: v.redeem(d),
                out: out.redeem(d),
                q_norm: q_norm.redeem(d),
                k_norm: k_norm.redeem(d),
                indexer_q: indexer_q.redeem(d),
                indexer_k: indexer_k.redeem(d),
                indexer_q_norm: indexer_q_norm.redeem(d),
                indexer_k_norm: indexer_k_norm.redeem(d),
            }),
        };
        Layer {
            hc_attn: self.hc_attn.redeem(d),
            hc_ffn: self.hc_ffn.redeem(d),
            block,
            ple: self.ple.map(|p| Ple {
                key: p.key.redeem(d),
                value: p.value.redeem(d),
                norm_key: p.norm_key.redeem(d),
                norm_query: p.norm_query.redeem(d),
                norm_conv: p.norm_conv.redeem(d),
                conv1d: p.conv1d.redeem(d),
            }),
            moe: Moe {
                router: self.moe.router.redeem(d),
                gate_exps: self.moe.gate_exps.redeem(d),
                up_exps: self.moe.up_exps.redeem(d),
                down_exps: self.moe.down_exps.redeem(d),
                shexp_gate: self.moe.shexp_gate.redeem(d),
                shexp_up: self.moe.shexp_up.redeem(d),
                shexp_down: self.moe.shexp_down.redeem(d),
            },
            nextn: self.nextn.map(|n| {
                let (fc_embedding, fc_hidden) = n.eh_proj.redeem(d);
                Nextn {
                    enorm: n.enorm.redeem(d),
                    hnorm: n.hnorm.redeem(d),
                    fc_embedding,
                    fc_hidden,
                    head: n.head.redeem(d),
                }
            }),
        }
    }
}

impl Group<'_, '_> {
    fn file(&self, t: &TensorRef) -> Result<&Arc<MappedFile>, String> {
        self.up
            .files
            .get(t.shard)
            .ok_or_else(|| format!("no file for shard {} of {}", t.shard, t.name))
    }

    fn push(&mut self, q: Queued) -> usize {
        self.queued.push(q);
        self.queued.len() - 1
    }

    fn fail(t: &TensorRef, e: gufo_hip::HipError) -> String {
        format!("upload failed for {}: {e}", t.name)
    }

    /// One weight with its tail.
    fn matrix(&mut self, t: &TensorRef) -> Result<Slot, String> {
        let file = Arc::clone(self.file(t)?);
        let pending = self
            .loader
            .add_parts::<u8>(&[(&file, t.offset, t.bytes)], TAIL)
            .map_err(|e| Self::fail(t, e))?;
        self.up.resident_bytes += t.bytes + TAIL;
        if matches!(t.ty, GgmlType::Bf16 | GgmlType::F16) {
            self.up.max_half_cols = self.up.max_half_cols.max(t.cols as usize);
        }
        if t.ty == GgmlType::Q8_0 && t.experts == 1 {
            self.up.max_q8_cols = self.up.max_q8_cols.max(t.cols as usize);
        }
        Ok(Slot(self.push(Queued::Matrix(
            pending, t.ty, t.rows, t.cols, t.experts,
        ))))
    }

    /// An F32 vector (norms, biases, convolution taps).
    fn vector(&mut self, t: &TensorRef) -> Result<VectorSlot, String> {
        let file = Arc::clone(self.file(t)?);
        let pending = self
            .loader
            .add::<f32>(&file, t.offset, t.bytes / 4)
            .map_err(|e| Self::fail(t, e))?;
        // Counted like the C++ upload, which pads every tensor.
        self.up.resident_bytes += t.bytes + TAIL;
        Ok(VectorSlot(self.push(Queued::Vector(pending))))
    }

    /// Matrices of one type and width stacked along rows: Q8_0 into one
    /// decode GEMV, F32 (router, alpha/beta) narrowed to F16 afterwards.
    fn stack(&mut self, parts: &[&TensorRef]) -> Result<Slot, String> {
        let first = parts[0];
        let ty = first.ty;
        if parts.iter().any(|t| t.ty != ty || t.cols != first.cols)
            || !matches!(ty, GgmlType::F32 | GgmlType::Q8_0)
        {
            return Err("stacked upload needs F32 or Q8_0 tensors of one shape".into());
        }
        let files: Vec<Arc<MappedFile>> = parts
            .iter()
            .map(|t| self.file(t).map(Arc::clone))
            .collect::<Result<_, _>>()?;
        let ranges: Vec<_> = parts
            .iter()
            .zip(&files)
            .map(|(t, f)| (f, t.offset, t.bytes))
            .collect();
        let rows: u64 = parts.iter().map(|t| t.rows).sum();
        let bytes: usize = parts.iter().map(|t| t.bytes).sum();
        let fail = |e| format!("upload failed for {}: {e}", first.name);
        if ty == GgmlType::Q8_0 {
            let pending = self.loader.add_parts::<u8>(&ranges, TAIL).map_err(fail)?;
            self.up.resident_bytes += bytes + TAIL;
            self.up.max_q8_cols = self.up.max_q8_cols.max(first.cols as usize);
            return Ok(Slot(
                self.push(Queued::Matrix(pending, ty, rows, first.cols, 1)),
            ));
        }
        // The F32 source lives only until it is narrowed.
        let pending = self.sources.add_parts::<f32>(&ranges, 0).map_err(fail)?;
        Ok(Slot(self.push(Queued::Narrow(pending, rows, first.cols))))
    }

    fn split(&mut self, t: &TensorRef) -> Result<SplitSlot, String> {
        let file = Arc::clone(self.file(t)?);
        let pending = self
            .sources
            .add_parts::<u8>(&[(&file, t.offset, t.bytes)], 0)
            .map_err(|e| Self::fail(t, e))?;
        Ok(SplitSlot(self.push(Queued::Split(pending, t.clone()))))
    }

    fn mixer(&mut self, m: &MixerRefs) -> Result<MixerSlots, String> {
        Ok(MixerSlots {
            norm: self.vector(&m.norm)?,
            down: self.matrix(&m.down)?,
            up: self.matrix(&m.up)?,
            inject: m.inject.as_ref().map(|t| self.matrix(t)).transpose()?,
        })
    }

    fn layer(&mut self, l: &LayerRefs) -> Result<LayerSlots, String> {
        let hc_attn = self.mixer(&l.hc_attn)?;
        let hc_ffn = self.mixer(&l.hc_ffn)?;
        // Projections of one input share one Q8_0 GEMV where the formats
        // allow; otherwise they stay separate.
        let stackable = |parts: &[&TensorRef]| {
            parts
                .iter()
                .all(|t| t.ty == GgmlType::Q8_0 && t.cols == parts[0].cols)
        };
        let block = match &l.mixer {
            MixerKind::Linear(s) => {
                let (ssm_in, qkv, gate) = if stackable(&[&s.qkv, &s.gate]) {
                    (Some(self.stack(&[&s.qkv, &s.gate])?), None, None)
                } else {
                    (
                        None,
                        Some(self.matrix(&s.qkv)?),
                        Some(self.matrix(&s.gate)?),
                    )
                };
                BlockSlots::Linear {
                    ssm_in,
                    qkv,
                    gate,
                    conv1d: self.vector(&s.conv1d)?,
                    alpha_beta: self.stack(&[&s.alpha, &s.beta])?,
                    dt: self.vector(&s.dt)?,
                    a: self.vector(&s.a)?,
                    norm: self.vector(&s.norm)?,
                    out: self.matrix(&s.out)?,
                }
            }
            MixerKind::Attention(a) => {
                let (qkv, q, k, v) = if stackable(&[&a.q, &a.k, &a.v]) {
                    (Some(self.stack(&[&a.q, &a.k, &a.v])?), None, None, None)
                } else {
                    (
                        None,
                        Some(self.matrix(&a.q)?),
                        Some(self.matrix(&a.k)?),
                        Some(self.matrix(&a.v)?),
                    )
                };
                BlockSlots::Attention {
                    qkv,
                    q,
                    k,
                    v,
                    out: self.matrix(&a.out)?,
                    q_norm: self.vector(&a.q_norm)?,
                    k_norm: self.vector(&a.k_norm)?,
                    indexer_q: self.matrix(&a.indexer_q)?,
                    indexer_k: self.matrix(&a.indexer_k)?,
                    indexer_q_norm: self.vector(&a.indexer_q_norm)?,
                    indexer_k_norm: self.vector(&a.indexer_k_norm)?,
                }
            }
        };
        let ple = match &l.ple {
            Some(p) => Some(PleSlots {
                key: self.matrix(&p.key)?,
                value: self.matrix(&p.value)?,
                norm_key: self.vector(&p.norm_key)?,
                norm_query: self.vector(&p.norm_query)?,
                norm_conv: self.vector(&p.norm_conv)?,
                conv1d: self.vector(&p.conv1d)?,
            }),
            None => None,
        };
        let m = &l.moe;
        let moe = MoeSlots {
            router: self.stack(&[&m.router, &m.shexp_gate_inp])?,
            gate_exps: self.matrix(&m.gate_exps)?,
            up_exps: self.matrix(&m.up_exps)?,
            down_exps: self.matrix(&m.down_exps)?,
            shexp_gate: self.matrix(&m.shexp_gate)?,
            shexp_up: self.matrix(&m.shexp_up)?,
            shexp_down: self.matrix(&m.shexp_down)?,
        };
        let nextn = match &l.nextn {
            Some(n) => Some(NextnSlots {
                enorm: self.vector(&n.enorm)?,
                hnorm: self.vector(&n.hnorm)?,
                eh_proj: self.split(&n.eh_proj)?,
                head: self.mixer(&n.head)?,
            }),
            None => None,
        };
        Ok(LayerSlots {
            hc_attn,
            hc_ffn,
            block,
            ple,
            moe,
            nextn,
        })
    }
}
