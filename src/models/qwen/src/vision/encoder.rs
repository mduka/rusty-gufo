//! The Qwen3.8 ViT (`src/models/qwen/vision/encoder.hip`): 27 pre-norm
//! blocks of width 1152 over 16x16 patches with 2D rotary attention, then a
//! 2x2 merger into the language model's width. The element-wise kernels are
//! shared with the C++ encoder; projections run through hipBLASLt with the
//! C++ plan choice (`HipblasLtGemm::RunBf16`), else hipBLAS, and attention
//! through strided batched hipBLAS GEMMs over query tiles, so embeddings
//! equal the C++ ones bit for bit.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use gufo_core::GgmlType;
use gufo_core::gguf::{Gguf, Value};
use gufo_core::image::Image;
use gufo_core::text::{Error, Result};
use gufo_hip::blas::{AlgoPolicy, Blas, BlasLt, Heuristic, Layout, MatmulShape, Op, Strides};
use gufo_hip::{
    Bf16, DeviceSlice, DeviceSliceMut, DeviceTensor, Launch, LaunchError, MappedFile, Placement,
    Stream, WeightLoader,
};
use gufo_qwen_kernels as k;
use sha2::{Digest, Sha256};

use super::MAX_ENCODER_PIXELS;

const HIDDEN: usize = 1152;
const HEADS: usize = 16;
const HEAD_DIM: usize = 72;
const FFN: usize = 4304;
const MERGE_HIDDEN: usize = HIDDEN * 4;
const PATCH_DIM: usize = 3 * 2 * 16 * 16;
const LAYERS: usize = 27;
const POSITIONS: usize = 48 * 48;
const TENSORS: usize = 334;

/// The canonical sidecar name next to a model.
pub const SIDECAR: &str = "mmproj-BF16.gguf";

/// A request-scoped cancellation check, polled between layers and tiles.
pub type CancellationCheck<'a> = &'a dyn Fn() -> bool;

/// Receives a stage's hidden state (`patch`, `layer0` to `layer26`,
/// `embedding`) for comparisons; each observation synchronises the device.
pub type Observer<'a> = &'a mut dyn FnMut(&str, &[f32]);

/// Image embeddings: `rows` merged tokens of the projector width, readable
/// on every stream.
pub struct Embedding {
    pub data: DeviceTensor<f32>,
    pub rows: u32,
    pub width: u32,
}

/// hipBLASLt's heuristic rank for a projection `Y[batch, m] = X · A[m, k]^T`
/// (`SelectHipblasLtHeuristicRank`): wide shapes of large batches take rank
/// 4, tall or FFN-down shapes rank 5, the rest the last of the results; an
/// unusable rank falls back to the first usable result.
struct VisionRank {
    batch: usize,
    m: usize,
    k: usize,
}

impl VisionRank {
    fn rank(&self, results: usize) -> usize {
        const LARGE_BATCH: usize = 256;
        const WIDE_FFN: usize = 3;
        const LARGE_WIDE: usize = 4;
        const TALL: usize = 5;
        if self.batch >= LARGE_BATCH {
            if self.m <= 1024 {
                return LARGE_WIDE;
            }
            return if self.k >= WIDE_FFN * self.m {
                TALL
            } else {
                LARGE_WIDE
            };
        }
        let tall_m = if self.batch >= 128 {
            self.k
        } else {
            2 * self.k
        };
        if self.m > tall_m || self.k >= WIDE_FFN * self.m {
            TALL
        } else {
            results - 1
        }
    }
}

impl AlgoPolicy for VisionRank {
    fn choose(&self, ranked: &[Heuristic], capacity: usize) -> Option<usize> {
        let rank = self.rank(ranked.len());
        if ranked.get(rank).is_some_and(|h| h.is_usable(capacity)) {
            return Some(rank);
        }
        ranked.iter().position(|h| h.is_usable(capacity))
    }
}

struct Block {
    ln1: [DeviceTensor<f32>; 2],
    ln2: [DeviceTensor<f32>; 2],
    qkv: (DeviceTensor<Bf16>, DeviceTensor<f32>),
    out: (DeviceTensor<Bf16>, DeviceTensor<f32>),
    up: (DeviceTensor<Bf16>, DeviceTensor<f32>),
    down: (DeviceTensor<Bf16>, DeviceTensor<f32>),
}

struct Weights {
    /// Both temporal patch kernels as one BF16 matrix `[1152][c][t][16][16]`.
    patch: DeviceTensor<Bf16>,
    patch_bias: DeviceTensor<f32>,
    positions: DeviceTensor<f32>,
    blocks: Vec<Block>,
    post_ln: [DeviceTensor<f32>; 2],
    merger: (DeviceTensor<Bf16>, DeviceTensor<f32>),
    projector: (DeviceTensor<Bf16>, DeviceTensor<f32>),
}

/// The device side, created with the first image.
struct Device {
    stream: Stream,
    lt: BlasLt,
    blas: Blas,
    weights: Weights,
}

pub struct Encoder {
    path: PathBuf,
    file: Arc<MappedFile>,
    output_width: u32,
    identity: OnceLock<String>,
    resident: AtomicUsize,
    device: Mutex<Option<Device>>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::invalid(message)
}

fn device_error(error: impl std::fmt::Display) -> Error {
    Error::runtime(format!("vision encoder: {error}"))
}

impl Encoder {
    /// Opens and validates a Qwen3.8 vision sidecar for a language model of
    /// `output_width` (5120 for the 27B, 2560 for Flash-Next).
    pub fn new(path: &Path, output_width: u32) -> Result<Self> {
        let file = MappedFile::open(path)
            .map_err(|e| invalid(format!("cannot open {}: {e}", path.display())))?;
        validate(&Gguf::parse(file.bytes()).map_err(invalid)?, output_width)?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            output_width,
            identity: OnceLock::new(),
            resident: AtomicUsize::new(0),
            device: Mutex::new(None),
        })
    }

    /// An explicit `sidecar` must exist; otherwise the canonical sidecar
    /// next to `target` or one directory up, if any.
    pub fn open(target: &Path, sidecar: &Path, output_width: u32) -> Result<Option<Arc<Self>>> {
        if !sidecar.as_os_str().is_empty() {
            return Self::new(sidecar, output_width).map(|e| Some(Arc::new(e)));
        }
        let mut directory = target.parent();
        for _ in 0..2 {
            let Some(dir) = directory else { break };
            let path = dir.join(SIDECAR);
            if path.is_file() {
                return Self::new(&path, output_width).map(|e| Some(Arc::new(e)));
            }
            directory = dir.parent();
        }
        Ok(None)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn output_width(&self) -> u32 {
        self.output_width
    }

    /// Lowercase hex SHA-256 of the whole sidecar, computed once.
    pub fn identity(&self) -> &str {
        self.identity.get_or_init(|| {
            Sha256::digest(self.file.bytes())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect()
        })
    }

    /// Device bytes of the uploaded weights (zero before the upload).
    pub fn resident_bytes(&self) -> usize {
        self.resident.load(Ordering::Acquire)
    }

    /// Uploads the weights now instead of with the first image (upstream
    /// `Encoder::Upload`); an existing upload is kept.
    pub fn prepare(&self) -> Result<()> {
        let mut guard = self
            .device
            .lock()
            .map_err(|_| Error::logic("vision encoder poisoned"))?;
        if guard.is_none() {
            *guard = Some(self.upload()?);
        }
        Ok(())
    }

    /// Encodes a resized image (edges multiples of 32 pixels). Weights upload
    /// with the first image unless [`Self::prepare`] uploaded them; scratch
    /// is bounded by the image.
    pub fn encode(
        &self,
        image: &Image,
        observer: Option<Observer<'_>>,
        cancelled: Option<CancellationCheck<'_>>,
    ) -> Result<Embedding> {
        let check = || match cancelled {
            Some(cancelled) if cancelled() => Err(Error::runtime("vision encoding cancelled")),
            _ => Ok(()),
        };
        check()?;
        let (w, h) = (image.width as usize, image.height as usize);
        if w == 0
            || h == 0
            || !w.is_multiple_of(32)
            || !h.is_multiple_of(32)
            || (w * h) as u64 > MAX_ENCODER_PIXELS
            || image.pixels.len() != w * h * 3
        {
            return Err(invalid("vision encoder requires a resized RGB32 grid"));
        }
        let mut guard = self
            .device
            .lock()
            .map_err(|_| Error::logic("vision encoder poisoned"))?;
        if guard.is_none() {
            let device = self.upload()?;
            *guard = Some(device);
        }
        let Some(device) = guard.as_mut() else {
            return Err(Error::logic("vision encoder has no device state"));
        };
        let result = device.encode(image, self.output_width as usize, observer, &check);
        if let Err(error) = &result
            && device.stream.poisoned().is_some()
        {
            // A failed device cannot be reused; the next image starts over.
            *guard = None;
            self.resident.store(0, Ordering::Release);
            return Err(error.clone());
        }
        result
    }

    fn upload(&self) -> Result<Device> {
        let gguf = Gguf::parse(self.file.bytes()).map_err(invalid)?;
        let stream = Stream::new().map_err(device_error)?;
        let blas = Blas::new(&stream).map_err(device_error)?;
        let lt = BlasLt::new(&stream, 0).map_err(device_error)?;
        let mut loader = WeightLoader::new(&stream, Placement::DirectCopy);
        let mut bytes = 0usize;
        let tensor = |name: &str| {
            gguf.tensor(name)
                .ok_or_else(|| invalid(format!("invalid BF16 Qwen vision tensor: {name}")))
        };
        let mut f32s = |loader: &mut WeightLoader<'_>, name: &str| -> Result<_> {
            let t = tensor(name)?;
            bytes += t.bytes;
            loader
                .add::<f32>(&self.file, t.offset, t.elements() as usize)
                .map_err(device_error)
        };
        let mut pending_f32 = Vec::new();
        let mut pending_bf16 = Vec::new();
        let mut names_f32 = Vec::new();
        let mut names_bf16 = Vec::new();
        for layer in 0..LAYERS {
            let p = format!("v.blk.{layer}.");
            for norm in ["ln1", "ln2"] {
                for part in ["weight", "bias"] {
                    names_f32.push(format!("{p}{norm}.{part}"));
                }
            }
            for name in ["attn_qkv", "attn_out", "ffn_up", "ffn_down"] {
                names_bf16.push(format!("{p}{name}.weight"));
                names_f32.push(format!("{p}{name}.bias"));
            }
        }
        for name in [
            "v.patch_embd.bias",
            "v.position_embd.weight",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "mm.0.bias",
            "mm.2.bias",
        ] {
            names_f32.push(name.into());
        }
        names_bf16.push("mm.0.weight".into());
        names_bf16.push("mm.2.weight".into());
        for name in &names_f32 {
            pending_f32.push(f32s(&mut loader, name)?);
        }
        for name in &names_bf16 {
            let t = tensor(name)?;
            bytes += t.bytes;
            pending_bf16.push(
                loader
                    .add::<Bf16>(&self.file, t.offset, t.elements() as usize)
                    .map_err(device_error)?,
            );
        }
        let loaded = loader.finish().map_err(device_error)?;
        let mut f32_tensors: std::collections::HashMap<String, DeviceTensor<f32>> = names_f32
            .into_iter()
            .zip(pending_f32)
            .map(|(name, p)| (name, loaded.get(p)))
            .collect();
        let mut bf16_tensors: std::collections::HashMap<String, DeviceTensor<Bf16>> = names_bf16
            .into_iter()
            .zip(pending_bf16)
            .map(|(name, p)| (name, loaded.get(p)))
            .collect();
        let mut take_f32 = |name: &str| {
            f32_tensors
                .remove(name)
                .ok_or_else(|| Error::logic(format!("vision tensor {name} missing")))
        };
        let mut take_bf16 = |name: &str| {
            bf16_tensors
                .remove(name)
                .ok_or_else(|| Error::logic(format!("vision tensor {name} missing")))
        };
        let mut blocks = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let p = format!("v.blk.{layer}.");
            let mut pair = |name: &str| -> Result<_> {
                Ok((
                    take_bf16(&format!("{p}{name}.weight"))?,
                    take_f32(&format!("{p}{name}.bias"))?,
                ))
            };
            let (qkv, out, up, down) = (
                pair("attn_qkv")?,
                pair("attn_out")?,
                pair("ffn_up")?,
                pair("ffn_down")?,
            );
            blocks.push(Block {
                ln1: [
                    take_f32(&format!("{p}ln1.weight"))?,
                    take_f32(&format!("{p}ln1.bias"))?,
                ],
                ln2: [
                    take_f32(&format!("{p}ln2.weight"))?,
                    take_f32(&format!("{p}ln2.bias"))?,
                ],
                qkv,
                out,
                up,
                down,
            });
        }
        // Both temporal patch kernels, F32 `[1152][3][16][16]` each, become
        // one BF16 matrix `[1152][3][2][256]` (round to nearest even).
        let kernel = |name: &str| -> Result<&[u8]> {
            let t = tensor(name)?;
            Ok(&self.file.bytes()[t.offset..t.offset + t.bytes])
        };
        let (first, second) = (
            kernel("v.patch_embd.weight")?,
            kernel("v.patch_embd.weight.1")?,
        );
        let value = |bytes: &[u8], i: usize| {
            f32::from_le_bytes([
                bytes[4 * i],
                bytes[4 * i + 1],
                bytes[4 * i + 2],
                bytes[4 * i + 3],
            ])
        };
        let mut patch = vec![Bf16::default(); HIDDEN * PATCH_DIM];
        for hidden in 0..HIDDEN {
            for c in 0..3 {
                for t in 0..2 {
                    for p in 0..256 {
                        let source = if t == 0 { first } else { second };
                        patch[hidden * PATCH_DIM + (c * 2 + t) * 256 + p] =
                            Bf16::from_f32(value(source, (hidden * 3 + c) * 256 + p));
                    }
                }
            }
        }
        let mut buffer = stream.alloc::<Bf16>(patch.len()).map_err(device_error)?;
        stream
            .upload(&patch, buffer.as_mut_slice())
            .map_err(device_error)?;
        let patch = buffer.freeze(&stream).map_err(device_error)?;
        let weights = Weights {
            patch,
            patch_bias: take_f32("v.patch_embd.bias")?,
            positions: take_f32("v.position_embd.weight")?,
            blocks,
            post_ln: [take_f32("v.post_ln.weight")?, take_f32("v.post_ln.bias")?],
            merger: (take_bf16("mm.0.weight")?, take_f32("mm.0.bias")?),
            projector: (take_bf16("mm.2.weight")?, take_f32("mm.2.bias")?),
        };
        self.resident.store(
            bytes + HIDDEN * PATCH_DIM * size_of::<Bf16>(),
            Ordering::Release,
        );
        Ok(Device {
            stream,
            lt,
            blas,
            weights,
        })
    }
}

/// The metadata and tensor checks of the C++ encoder's constructor.
fn validate(gguf: &Gguf<'_>, width: u32) -> Result<()> {
    let expect = |key: &str, value: u64| {
        if gguf.u64(key) != Some(value) {
            return Err(invalid(format!("incompatible Qwen vision metadata: {key}")));
        }
        Ok(())
    };
    expect("clip.vision.projection_dim", u64::from(width))?;
    expect("clip.vision.patch_size", 16)?;
    expect("clip.vision.embedding_length", HIDDEN as u64)?;
    expect("clip.vision.feed_forward_length", FFN as u64)?;
    expect("clip.vision.block_count", LAYERS as u64)?;
    expect("clip.vision.attention.head_count", HEADS as u64)?;
    expect("clip.vision.spatial_merge_size", 2)?;
    expect("clip.vision.image_size", 768)?;
    // The tensor layout decides; quantizers name their sidecars freely.
    if (width != 5120 && width != 2560)
        || gguf.string("general.architecture") != Some(b"clip")
        || gguf.string("general.type") != Some(b"mmproj")
        || gguf.bool("clip.has_vision_encoder") != Some(true)
        || gguf.bool("clip.use_gelu") != Some(true)
        || gguf.f32("clip.vision.attention.layer_norm_epsilon") != Some(1e-6)
    {
        return Err(invalid("vision sidecar does not match the Qwen3.8 model"));
    }
    for key in ["clip.vision.image_mean", "clip.vision.image_std"] {
        if gguf.metadata(key) != Some(&Value::FloatArray(vec![0.5, 0.5, 0.5])) {
            return Err(invalid("unsupported Qwen image normalization"));
        }
    }
    match gguf.metadata("clip.vision.is_deepstack_layers") {
        Some(Value::UIntArray(layers))
            if layers.len() == LAYERS && layers.iter().all(|&l| l == 0) => {}
        _ => return Err(invalid("Qwen3.8 vision requires a final-layer-only merger")),
    }
    if gguf.string("clip.projector_type") != Some(b"qwen3vl_merger") {
        return Err(invalid("unsupported Qwen vision projector"));
    }
    let require = |name: &str, shape: &[u64], ty: GgmlType| match gguf.tensor(name) {
        Some(t) if t.dims == shape && t.ty == ty => Ok(()),
        _ => Err(invalid(format!("invalid BF16 Qwen vision tensor: {name}"))),
    };
    let (hidden, ffn, merge) = (HIDDEN as u64, FFN as u64, MERGE_HIDDEN as u64);
    for layer in 0..LAYERS {
        let p = format!("v.blk.{layer}.");
        for norm in ["ln1", "ln2"] {
            require(&format!("{p}{norm}.weight"), &[hidden], GgmlType::F32)?;
            require(&format!("{p}{norm}.bias"), &[hidden], GgmlType::F32)?;
        }
        for (name, input, output) in [
            ("attn_qkv", hidden, 3 * hidden),
            ("attn_out", hidden, hidden),
            ("ffn_up", hidden, ffn),
            ("ffn_down", ffn, hidden),
        ] {
            require(
                &format!("{p}{name}.weight"),
                &[input, output],
                GgmlType::Bf16,
            )?;
            require(&format!("{p}{name}.bias"), &[output], GgmlType::F32)?;
        }
    }
    require("v.patch_embd.weight", &[16, 16, 3, hidden], GgmlType::F32)?;
    require("v.patch_embd.weight.1", &[16, 16, 3, hidden], GgmlType::F32)?;
    require("v.patch_embd.bias", &[hidden], GgmlType::F32)?;
    require(
        "v.position_embd.weight",
        &[hidden, POSITIONS as u64],
        GgmlType::F32,
    )?;
    require("v.post_ln.weight", &[hidden], GgmlType::F32)?;
    require("v.post_ln.bias", &[hidden], GgmlType::F32)?;
    require("mm.0.weight", &[merge, merge], GgmlType::Bf16)?;
    require("mm.0.bias", &[merge], GgmlType::F32)?;
    require("mm.2.weight", &[merge, u64::from(width)], GgmlType::Bf16)?;
    require("mm.2.bias", &[u64::from(width)], GgmlType::F32)?;
    if gguf.tensors().len() != TENSORS {
        return Err(invalid("unexpected tensors in Qwen vision projector"));
    }
    Ok(())
}

fn launch(error: LaunchError) -> Error {
    device_error(error)
}

/// `output[rows, out] = input[rows, in] · weight[out, in]^T`.
#[allow(clippy::too_many_arguments)]
fn project(
    lt: &mut BlasLt,
    blas: &Blas,
    stream: &Stream,
    weight: &DeviceTensor<Bf16>,
    input: DeviceSlice<'_, Bf16>,
    mut output: DeviceSliceMut<'_, f32>,
    rows: usize,
    out: usize,
    inner: usize,
) -> Result<()> {
    let shape = MatmulShape {
        trans_a: Some(Op::T),
        trans_b: None,
        a: Layout::new(inner, out, inner),
        b: Layout::new(inner, rows, inner),
        c: Layout::new(out, rows, out),
        batch: None,
    };
    let policy = VisionRank {
        batch: rows,
        m: out,
        k: inner,
    };
    let launched = lt
        .matmul(
            stream,
            &shape,
            weight.view(),
            input,
            output.reborrow(),
            &policy,
        )
        .map_err(launch)?;
    if launched == Launch::Launched {
        return Ok(());
    }
    blas.gemm(
        stream,
        Op::T,
        Op::N,
        out,
        rows,
        inner,
        weight.view(),
        inner,
        input,
        inner,
        output,
        out,
    )
    .map_err(launch)
}

impl Device {
    fn encode(
        &mut self,
        image: &Image,
        output_width: usize,
        mut observer: Option<Observer<'_>>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Embedding> {
        let Device {
            stream,
            lt,
            blas,
            weights,
        } = self;
        let stream = &*stream;
        let (grid_w, grid_h) = (image.width as usize / 16, image.height as usize / 16);
        let count = grid_w * grid_h;
        let merged = count / 4;
        let alloc = |len| stream.alloc(len).map_err(device_error);
        let alloc16 = |len| stream.alloc::<Bf16>(len).map_err(device_error);
        let mut pixels = stream
            .alloc::<u8>(image.pixels.len())
            .map_err(device_error)?;
        let mut hidden = alloc(count * HIDDEN)?;
        let mut work = alloc(count * FFN.max(3 * HIDDEN))?;
        let mut normalized = alloc16(count * FFN.max(PATCH_DIM))?;
        let mut qkv = alloc16(3 * count * HIDDEN)?;
        let mut attention = alloc16(count * HIDDEN)?;
        // Smaller attention tiles reduce scratch traffic for the 4096-patch
        // shape while keeping its BLAS reductions; ragged tails need their
        // own GEMM shapes to keep the rounding.
        let tile = count.min(if count == 4096 { 64 } else { 128 });
        let mut scores = alloc(HEADS * tile * count)?;
        let mut probabilities = alloc16(HEADS * tile * count)?;
        let mut output_tile = alloc(tile * HIDDEN)?;
        stream
            .upload(&image.pixels, pixels.as_mut_slice())
            .map_err(device_error)?;
        let (count32, grid_w32, grid_h32) = (count as u32, grid_w as u32, grid_h as u32);
        k::vision_patchify(
            stream,
            pixels.as_slice(),
            normalized.as_mut_slice(),
            image.width,
            grid_w32,
            count32,
        )
        .map_err(launch)?;
        project(
            lt,
            blas,
            stream,
            &weights.patch,
            normalized.as_slice(),
            hidden.as_mut_slice(),
            count,
            HIDDEN,
            PATCH_DIM,
        )?;
        k::vision_patch_position(
            stream,
            hidden.as_mut_slice(),
            weights.patch_bias.view(),
            weights.positions.view(),
            grid_h32,
            grid_w32,
            count32,
        )
        .map_err(launch)?;
        let mut observe = |name: &str, data: DeviceSlice<'_, f32>| -> Result<()> {
            if let Some(observer) = observer.as_mut() {
                let mut host = vec![0f32; data.len()];
                stream.download(data, &mut host).map_err(device_error)?;
                observer(name, &host);
            }
            Ok(())
        };
        observe("patch", hidden.as_slice())?;
        let size = (count * HIDDEN) as u32;
        for layer in 0..LAYERS {
            check()?;
            let Block {
                ln1,
                ln2,
                qkv: qkv_w,
                out: out_w,
                up: up_w,
                down: down_w,
            } = &weights.blocks[layer];
            k::vision_layer_norm(
                stream,
                hidden.as_slice(),
                normalized.as_mut_slice(),
                ln1[0].view(),
                ln1[1].view(),
                count32,
                HIDDEN as u32,
            )
            .map_err(launch)?;
            project(
                lt,
                blas,
                stream,
                &qkv_w.0,
                normalized.as_slice(),
                work.as_mut_slice(),
                count,
                3 * HIDDEN,
                HIDDEN,
            )?;
            {
                let (q, rest) = qkv.as_mut_slice().split_at(count * HIDDEN);
                let (key, value) = rest.split_at(count * HIDDEN);
                k::vision_qkv_rope(
                    stream,
                    work.as_slice(),
                    qkv_w.1.view(),
                    q,
                    key,
                    value,
                    count32,
                    grid_w32,
                )
                .map_err(launch)?;
            }
            let plane = count * HIDDEN;
            let mut begin = 0;
            while begin < count {
                check()?;
                let rows = tile.min(count - begin);
                let q = qkv.slice(0..plane);
                let key = qkv.slice(plane..2 * plane);
                let value = qkv.slice(2 * plane..3 * plane);
                blas.gemm_strided_batched(
                    stream,
                    Op::T,
                    Op::N,
                    count,
                    rows,
                    HEAD_DIM,
                    key,
                    HEAD_DIM,
                    q.range(begin * HEAD_DIM..plane),
                    HEAD_DIM,
                    scores.as_mut_slice(),
                    count,
                    Strides {
                        count: HEADS,
                        a: count * HEAD_DIM,
                        b: count * HEAD_DIM,
                        c: rows * count,
                    },
                )
                .map_err(launch)?;
                k::vision_softmax(
                    stream,
                    scores.as_slice(),
                    probabilities.as_mut_slice(),
                    (HEADS * rows) as u32,
                    count32,
                )
                .map_err(launch)?;
                blas.gemm_strided_batched(
                    stream,
                    Op::N,
                    Op::N,
                    HEAD_DIM,
                    rows,
                    count,
                    value,
                    HEAD_DIM,
                    probabilities.as_slice(),
                    count,
                    output_tile.as_mut_slice(),
                    HEAD_DIM,
                    Strides {
                        count: HEADS,
                        a: count * HEAD_DIM,
                        b: rows * count,
                        c: rows * HEAD_DIM,
                    },
                )
                .map_err(launch)?;
                k::vision_attention_rows(
                    stream,
                    output_tile.as_slice(),
                    attention.as_mut_slice(),
                    begin as u32,
                    rows as u32,
                    count32,
                )
                .map_err(launch)?;
                begin += rows;
            }
            project(
                lt,
                blas,
                stream,
                &out_w.0,
                attention.as_slice(),
                work.as_mut_slice(),
                count,
                HIDDEN,
                HIDDEN,
            )?;
            k::vision_bias_residual(
                stream,
                hidden.as_mut_slice(),
                work.as_slice(),
                out_w.1.view(),
                size,
                HIDDEN as u32,
            )
            .map_err(launch)?;
            k::vision_layer_norm(
                stream,
                hidden.as_slice(),
                normalized.as_mut_slice(),
                ln2[0].view(),
                ln2[1].view(),
                count32,
                HIDDEN as u32,
            )
            .map_err(launch)?;
            project(
                lt,
                blas,
                stream,
                &up_w.0,
                normalized.as_slice(),
                work.as_mut_slice(),
                count,
                FFN,
                HIDDEN,
            )?;
            k::vision_activate(
                stream,
                work.as_slice(),
                up_w.1.view(),
                normalized.as_mut_slice(),
                (count * FFN) as u32,
                FFN as u32,
                true,
            )
            .map_err(launch)?;
            project(
                lt,
                blas,
                stream,
                &down_w.0,
                normalized.as_slice(),
                work.as_mut_slice(),
                count,
                HIDDEN,
                FFN,
            )?;
            k::vision_bias_residual(
                stream,
                hidden.as_mut_slice(),
                work.as_slice(),
                down_w.1.view(),
                size,
                HIDDEN as u32,
            )
            .map_err(launch)?;
            observe(&format!("layer{layer}"), hidden.as_slice())?;
        }
        let post_ln = &weights.post_ln;
        k::vision_layer_norm(
            stream,
            hidden.as_slice(),
            normalized.as_mut_slice(),
            post_ln[0].view(),
            post_ln[1].view(),
            count32,
            HIDDEN as u32,
        )
        .map_err(launch)?;
        let merger = &weights.merger;
        project(
            lt,
            blas,
            stream,
            &merger.0,
            normalized.as_slice(),
            work.as_mut_slice(),
            merged,
            MERGE_HIDDEN,
            MERGE_HIDDEN,
        )?;
        k::vision_activate(
            stream,
            work.as_slice(),
            merger.1.view(),
            normalized.as_mut_slice(),
            (merged * MERGE_HIDDEN) as u32,
            MERGE_HIDDEN as u32,
            false,
        )
        .map_err(launch)?;
        let projector = &weights.projector;
        let mut embedding = alloc(merged * output_width)?;
        project(
            lt,
            blas,
            stream,
            &projector.0,
            normalized.as_slice(),
            embedding.as_mut_slice(),
            merged,
            output_width,
            MERGE_HIDDEN,
        )?;
        k::vision_finish(
            stream,
            embedding.as_mut_slice(),
            projector.1.view(),
            (merged * output_width) as u32,
            output_width as u32,
        )
        .map_err(launch)?;
        let data = embedding.freeze(stream).map_err(device_error)?;
        check()?;
        observe("embedding", data.view())?;
        Ok(Embedding {
            data,
            rows: merged as u32,
            width: output_width as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_follow_the_cpp_plan_choice() {
        let rank = |batch, m, k| VisionRank { batch, m, k }.rank(16);
        // Large batches: wide shapes rank 4, FFN-down (k >= 3m) rank 5.
        assert_eq!(rank(4096, 1152, 1152), 4);
        assert_eq!(rank(4096, 1152, 4304), 5);
        assert_eq!(rank(4096, 3456, 1152), 4);
        assert_eq!(rank(1024, 5120, 4608), 4);
        // Small batches: tall shapes rank 5, the rest the last result.
        assert_eq!(rank(64, 3456, 1152), 5);
        assert_eq!(rank(64, 1152, 1152), 15);
        assert_eq!(rank(200, 1152, 1152), 15);
        assert_eq!(rank(200, 1153, 1152), 5);
    }
}
