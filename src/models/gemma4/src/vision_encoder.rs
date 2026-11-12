//! The Gemma 4 vision tower and embedder on the GPU (`vision/encoder.hpp`,
//! `vision/encoder.hip`): BF16 weights converted to binary16 GEMM operands,
//! FP32 accumulation and residual stream, following
//! [`crate::vision_reference`], whose stage names the observer receives.
//!
//! The kernels are the extension's (`gufo-gemma4-kernels`, `vision_*`);
//! the products are its `UnquantizedF16Gemm` with hipBLAS `GemmEx` behind
//! it, and the patch embedding its FP32 `hipblasSgemm`, on a handle in
//! hipBLAS's default atomics mode as the C++ creates it.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use gufo_core::gguf::Gguf;
use gufo_core::image::Image;
use gufo_gemma4_kernels as k;
use gufo_hip::blas::{Blas, Op};
use gufo_hip::{
    DeviceBuffer, DeviceCopy, DeviceSlice, DeviceTensor, F16, GgmlType, HipError, Launch,
    LaunchError, MappedFile, PinnedBuffer, QuantSlice, Stream,
};
use sha2::{Digest, Sha256};

use crate::vision_prompt::{PATCH_SIZE, POOL_SIZE, SIDE_MULTIPLE};
use crate::vision_weights::{
    FFN, HEAD_DIM, HEADS, HIDDEN, LAYERS, PATCH_VALUES, POSITIONS, ROPE_THETA, VisionTensor,
    VisionWeights,
};

/// The sidecar's file name beside the target (`Encoder::Open`).
pub const SIDECAR_FILE_NAME: &str = "mmproj-BF16.gguf";

/// Fused Q, K and V rows (`kQkv`).
const QKV: u32 = 3 * HIDDEN;
/// Fused gate and up rows (`kGateUp`).
const GATE_UP: u32 = 2 * FFN;
/// The down projection's K padded to whole 32-value blocks for the binary16
/// GEMM, zero past [`FFN`] (`kFfnPadded`).
const FFN_PADDED: u32 = FFN.div_ceil(32) * 32;
/// Rotary pairs per axis: each half of a head rotates as NEOX pairs
/// (`kRopePairs`).
const ROPE_PAIRS: u32 = HEAD_DIM / 4;
/// Attention operands pad each head to whole 16-wide WMMA steps
/// (`kPaddedDim`) and each key sequence to whole tiles (`kKeyTile`).
const PADDED_DIM: u32 = 80;
const KEY_TILE: u32 = 64;
/// `UnquantizedF16Gemm` takes 128 x 128 tiles from 128 rows on: a 630-row
/// layer (a 70-token image) takes 0.75 instead of 1.14 ms.
const WIDE_FROM: usize = 128;
/// Largest image the encoder takes, in pixels.
const MAX_PIXELS: u64 = 16_777_216;

/// The sidecar a target uses (`Encoder::Open`): an explicit path as given
/// (it must exist when loaded), otherwise [`SIDECAR_FILE_NAME`] beside the
/// target or one directory up; `None` when there is none.
pub fn sidecar_path(target: &Path, explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return Some(path.to_path_buf());
    }
    let mut directory = target.parent();
    for _ in 0..2 {
        let dir = directory?;
        let path = dir.join(SIDECAR_FILE_NAME);
        if path.is_file() {
            return Some(path);
        }
        directory = dir.parent();
    }
    None
}

/// Sees each named stage's rows as the encoder computes them
/// (`Encoder::Observer`): "patch", "layer<i>", "pooled" and "embedding".
pub type Observer<'o> = &'o mut dyn FnMut(&str, &[f32]);
/// True once the request that wants the encoding is cancelled
/// (`Encoder::CancellationCheck`).
pub type CancellationCheck<'c> = &'c dyn Fn() -> bool;

/// One block's resident weights (`Encoder::Impl::Layer`); GEMM weights as
/// binary16 `[out][in]` rows, norms as F32.
struct Layer {
    qkv: DeviceBuffer<u8>,
    out: DeviceBuffer<u8>,
    gate_up: DeviceBuffer<u8>,
    /// `[1152][4320]`, zero past 4304.
    down: DeviceBuffer<u8>,
    ln1: DeviceBuffer<u8>,
    ln2: DeviceBuffer<u8>,
    q_norm: DeviceBuffer<u8>,
    k_norm: DeviceBuffer<u8>,
    attn_post_norm: DeviceBuffer<u8>,
    ffn_post_norm: DeviceBuffer<u8>,
}

struct Uploaded {
    patch: DeviceBuffer<u8>,
    position: DeviceBuffer<u8>,
    std_bias: DeviceBuffer<u8>,
    std_scale: DeviceBuffer<u8>,
    projection: DeviceBuffer<u8>,
    layers: Vec<Layer>,
}

/// The encoder's stream, its hipBLAS handle and the uploaded weights; one
/// encode or upload at a time.
struct Device {
    stream: Stream,
    blas: Blas,
    uploaded: Option<Uploaded>,
}

/// The vision encoder of one sidecar (`vision::Encoder`). `Send + Sync`;
/// encodes are serialized inside.
pub struct VisionEncoder {
    file: Arc<MappedFile>,
    weights: VisionWeights,
    identity: OnceLock<String>,
    resident_bytes: AtomicUsize,
    device: Mutex<Option<Device>>,
}

fn hip(error: HipError) -> String {
    format!("Gemma 4 vision: {error}")
}

fn launched(result: Result<Launch, LaunchError>, what: &str) -> Result<(), String> {
    match result {
        Ok(Launch::Launched) => Ok(()),
        Ok(Launch::Unsupported) => Err(format!("Gemma 4 vision {what}: unsupported geometry")),
        Err(error) => Err(format!("Gemma 4 vision {what}: {error}")),
    }
}

fn blas(result: Result<(), LaunchError>, what: &str) -> Result<(), String> {
    result.map_err(|error| format!("Gemma 4 vision hipBLAS {what}: {error}"))
}

/// Device memory as `hip::AllocateDevice` places it: `hipMalloc` while the
/// device heap keeps its dispatch reserve, coarse-grain managed memory
/// otherwise or when that fails.
fn allocate_device<T: DeviceCopy>(stream: &Stream, len: usize) -> Result<DeviceBuffer<T>, String> {
    let bytes = (len * size_of::<T>()) as u64;
    let fits = gufo_hip::device_memory()
        .is_ok_and(|(free, _)| gufo_hip::heap_fits_above_reserve(free as u64, bytes));
    if fits && let Ok(buffer) = stream.alloc_persistent(len) {
        return Ok(buffer);
    }
    stream.alloc_managed_coarse(len).map_err(hip)
}

/// F32 values of a raw buffer.
fn floats(buffer: &DeviceBuffer<u8>) -> Result<DeviceSlice<'_, f32>, String> {
    buffer
        .as_slice()
        .cast::<f32>()
        .ok_or_else(|| "Gemma 4 vision: misaligned F32 weight".to_string())
}

/// Bytes of each of the two page-locked buffers an upload streams through.
const STAGING_BYTES: usize = 16 << 20;

/// The host side of an upload: the mapped sidecar and the page-locked
/// buffers its pieces pass through, so no copy waits for the previous one.
struct Source<'f> {
    file: &'f [u8],
    staging: [PinnedBuffer<u8>; 2],
}

/// Uploads tensors as they are stored, joined in order (`Upload`).
fn upload_raw(
    stream: &Stream,
    source: &mut Source<'_>,
    parts: &[&VisionTensor],
    total: &mut usize,
) -> Result<DeviceBuffer<u8>, String> {
    let bytes: usize = parts.iter().map(|part| part.bytes).sum();
    let mut buffer = allocate_device::<u8>(stream, bytes)?;
    let mut offset = 0;
    for part in parts {
        stream
            .upload_bytes_staged(
                part.data(source.file),
                buffer.slice_mut(offset..offset + part.bytes),
                &mut source.staging,
            )
            .map_err(hip)?;
        offset += part.bytes;
    }
    *total += bytes;
    Ok(buffer)
}

/// A BF16 GEMM weight, uploaded and converted to binary16 (`UploadHalf`).
fn upload_half(
    stream: &Stream,
    source: &mut Source<'_>,
    parts: &[&VisionTensor],
    total: &mut usize,
) -> Result<DeviceBuffer<u8>, String> {
    let mut buffer = upload_raw(stream, source, parts, total)?;
    let bytes = buffer.len();
    launched(
        k::vision_bf16_to_half(stream, buffer.as_mut_slice(), bytes / 2),
        "weight conversion",
    )?;
    Ok(buffer)
}

/// A `[rows][cols]` BF16 GEMM weight as binary16 rows of `padded` values,
/// zero past `cols` (`UploadHalfPadded`).
fn upload_half_padded(
    stream: &Stream,
    source: &mut Source<'_>,
    part: &VisionTensor,
    rows: usize,
    cols: usize,
    padded: usize,
    total: &mut usize,
) -> Result<DeviceBuffer<u8>, String> {
    let count = rows * padded;
    let mut buffer = allocate_device::<u8>(stream, count * 2)?;
    stream.fill_bytes(buffer.as_mut_slice(), 0).map_err(hip)?;
    stream
        .upload_2d(
            part.data(source.file),
            cols * 2,
            buffer.as_mut_slice(),
            padded * 2,
            cols * 2,
            rows,
        )
        .map_err(hip)?;
    launched(
        k::vision_bf16_to_half(stream, buffer.as_mut_slice(), count),
        "weight conversion",
    )?;
    *total += count * 2;
    Ok(buffer)
}

/// Every weight, in the C++ order (`UploadWeights`).
fn upload_weights(
    stream: &Stream,
    file: &[u8],
    w: &VisionWeights,
) -> Result<(Uploaded, usize), String> {
    let mut total = 0;
    let t = &mut total;
    let pinned = || PinnedBuffer::<u8>::zeroed(STAGING_BYTES).map_err(hip);
    let mut source = Source {
        file,
        staging: [pinned()?, pinned()?],
    };
    let src = &mut source;
    let patch = upload_raw(stream, src, &[&w.patch], t)?;
    let position = upload_raw(stream, src, &[&w.position], t)?;
    let std_bias = upload_raw(stream, src, &[&w.std_bias], t)?;
    let std_scale = upload_raw(stream, src, &[&w.std_scale], t)?;
    let projection = upload_half(stream, src, &[&w.projection], t)?;
    let mut layers = Vec::with_capacity(LAYERS as usize);
    for l in &w.layers {
        layers.push(Layer {
            qkv: upload_half(stream, src, &[&l.q, &l.k, &l.v], t)?,
            out: upload_half(stream, src, &[&l.out], t)?,
            gate_up: upload_half(stream, src, &[&l.gate, &l.up], t)?,
            down: upload_half_padded(
                stream,
                src,
                &l.down,
                HIDDEN as usize,
                FFN as usize,
                FFN_PADDED as usize,
                t,
            )?,
            ln1: upload_raw(stream, src, &[&l.ln1], t)?,
            ln2: upload_raw(stream, src, &[&l.ln2], t)?,
            q_norm: upload_raw(stream, src, &[&l.q_norm], t)?,
            k_norm: upload_raw(stream, src, &[&l.k_norm], t)?,
            attn_post_norm: upload_raw(stream, src, &[&l.attn_post_norm], t)?,
            ffn_post_norm: upload_raw(stream, src, &[&l.ffn_post_norm], t)?,
        });
    }
    stream.synchronize().map_err(hip)?;
    Ok((
        Uploaded {
            patch,
            position,
            std_bias,
            std_scale,
            projection,
            layers,
        },
        total,
    ))
}

/// `y[rows][out] = x[rows][in] W[out][in]^T` with binary16 inputs and FP32
/// results (`Gemm`): the extension's GEMM, or hipBLAS `GemmEx` where it
/// declines.
#[allow(clippy::too_many_arguments)]
fn gemm(
    stream: &Stream,
    handle: &Blas,
    w: &DeviceBuffer<u8>,
    x: &DeviceBuffer<F16>,
    y: &mut DeviceBuffer<f32>,
    rows: usize,
    out: usize,
    input: usize,
) -> Result<(), String> {
    let launch = k::unquantized_f16_gemm(
        stream,
        QuantSlice::new(w.as_slice(), GgmlType::F16),
        x.slice(0..rows * input),
        y.slice_mut(0..rows * out),
        rows,
        out,
        input,
        WIDE_FROM,
    );
    match launch {
        Ok(Launch::Launched) => Ok(()),
        Ok(Launch::Unsupported) => {
            let weights = w
                .as_slice()
                .cast::<F16>()
                .ok_or_else(|| "Gemma 4 vision: misaligned binary16 weight".to_string())?;
            blas(
                handle.gemm::<F16, F16, f32>(
                    stream,
                    Op::T,
                    Op::N,
                    out,
                    rows,
                    input,
                    weights,
                    input,
                    x.slice(0..rows * input),
                    input,
                    y.slice_mut(0..rows * out),
                    out,
                ),
                "GemmEx",
            )
        }
        Err(error) => Err(format!("Gemma 4 vision product: {error}")),
    }
}

/// Rotary tables for every patch column and row (`[cos][sin]`, each
/// `[position][18]`), in double like the C++.
fn rope_table(positions: u32) -> Vec<f32> {
    let pairs = ROPE_PAIRS as usize;
    let count = positions as usize;
    let mut table = vec![0.0f32; 2 * count * pairs];
    for p in 0..count {
        for i in 0..pairs {
            let angle =
                p as f64 * f64::from(ROPE_THETA).powf(-2.0 * i as f64 / f64::from(HEAD_DIM / 2));
            table[p * pairs + i] = angle.cos() as f32;
            table[(count + p) * pairs + i] = angle.sin() as f32;
        }
    }
    table
}

impl VisionEncoder {
    /// The encoder of the sidecar a target uses, or `None` without one
    /// (`Encoder::Open`). `output_width` is the language model's hidden size.
    pub fn open(
        target: &Path,
        explicit: Option<&Path>,
        output_width: u32,
    ) -> Result<Option<Self>, String> {
        sidecar_path(target, explicit)
            .map(|path| Self::load(&path, output_width))
            .transpose()
    }

    /// Opens and validates the sidecar at `path`; weights upload in
    /// [`Self::upload`] or the first [`Self::encode`].
    pub fn load(path: &Path, output_width: u32) -> Result<Self, String> {
        let file = MappedFile::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let weights = {
            let gguf = Gguf::parse(file.bytes()).map_err(|e| format!("{}: {e}", path.display()))?;
            VisionWeights::resolve(&gguf, output_width)?
        };
        Ok(Self {
            file,
            weights,
            identity: OnceLock::new(),
            resident_bytes: AtomicUsize::new(0),
            device: Mutex::new(None),
        })
    }

    /// SHA-256 of the whole sidecar as lowercase hex, computed once.
    pub fn identity(&self) -> &str {
        self.identity.get_or_init(|| {
            let digest = Sha256::digest(self.file.bytes());
            digest.iter().map(|byte| format!("{byte:02x}")).collect()
        })
    }

    /// Device bytes of the uploaded weights; zero before the upload.
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes.load(Ordering::Acquire)
    }

    /// The language model's hidden size the embeddings have.
    pub fn output_width(&self) -> u32 {
        self.weights.output_width
    }

    /// Uploads the weights now instead of in the first encode; repeated
    /// calls keep the existing upload (`Upload`).
    pub fn upload(&self) -> Result<(), String> {
        let mut device = self.lock()?;
        self.ensure_uploaded(&mut device).map(|_| ())
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Option<Device>>, String> {
        self.device
            .lock()
            .map_err(|_| "Gemma 4 vision encoder poisoned".to_string())
    }

    fn ensure_uploaded<'d>(
        &self,
        device: &'d mut Option<Device>,
    ) -> Result<&'d mut Device, String> {
        if device.is_none() {
            let stream = Stream::new().map_err(hip)?;
            let blas = Blas::with_default_atomics(&stream).map_err(hip)?;
            *device = Some(Device {
                stream,
                blas,
                uploaded: None,
            });
        }
        let Some(device) = device.as_mut() else {
            return Err("Gemma 4 vision: no device".into());
        };
        if device.uploaded.is_none() {
            let (uploaded, bytes) =
                upload_weights(&device.stream, self.file.bytes(), &self.weights)?;
            device.uploaded = Some(uploaded);
            self.resident_bytes.store(bytes, Ordering::Release);
        }
        Ok(device)
    }

    /// Language-model input embeddings (`[rows][output_width]` f32) of an
    /// image already resized by `vision_prompt` (sides multiples of 48)
    /// (`Encode`).
    pub fn encode(&self, image: &Image) -> Result<Arc<DeviceTensor<f32>>, String> {
        self.encode_observed(image, None, None)
    }

    /// [`Self::encode`] with an observer of every stage and a cancellation
    /// check before each block (`Encode` with its observer and
    /// `CancellationCheck`).
    pub fn encode_observed(
        &self,
        image: &Image,
        mut observer: Option<Observer<'_>>,
        is_cancelled: Option<CancellationCheck<'_>>,
    ) -> Result<Arc<DeviceTensor<f32>>, String> {
        let check_cancelled = || match is_cancelled {
            Some(cancelled) if cancelled() => Err("vision encoding cancelled".to_string()),
            _ => Ok(()),
        };
        check_cancelled()?;
        let (width, height) = (image.width, image.height);
        if width == 0
            || height == 0
            || !width.is_multiple_of(SIDE_MULTIPLE)
            || !height.is_multiple_of(SIDE_MULTIPLE)
            || width / PATCH_SIZE > POSITIONS
            || height / PATCH_SIZE > POSITIONS
            || u64::from(width) * u64::from(height) > MAX_PIXELS
            || image.pixels.len() != width as usize * height as usize * 3
        {
            return Err("vision encoder requires a resized RGB image".into());
        }
        let mut guard = self.lock()?;
        let device = self.ensure_uploaded(&mut guard)?;
        let Device {
            stream,
            blas: handle,
            uploaded,
        } = device;
        let Some(u) = uploaded.as_ref() else {
            return Err("Gemma 4 vision: weights not uploaded".into());
        };
        let stream = &*stream;
        let handle = &*handle;

        let columns = width / PATCH_SIZE;
        let patch_rows = height / PATCH_SIZE;
        let count = columns * patch_rows;
        let pooled_columns = columns / POOL_SIZE;
        let rows = pooled_columns * (patch_rows / POOL_SIZE);
        let out_width = self.weights.output_width;
        let n = count as usize;
        let hidden_n = n * HIDDEN as usize;

        let positions = columns.max(patch_rows);
        let table = rope_table(positions);
        let half = table.len() / 2;

        let padded = count.div_ceil(KEY_TILE) * KEY_TILE;
        let head_values = (HEADS * padded * PADDED_DIM) as usize;
        let mut pixels = stream.alloc::<u8>(image.pixels.len()).map_err(hip)?;
        let mut rope = stream.alloc::<f32>(table.len()).map_err(hip)?;
        let mut hidden = stream.alloc::<f32>(hidden_n).map_err(hip)?;
        let mut work = stream
            .alloc::<f32>(n * GATE_UP.max(QKV).max(PATCH_VALUES) as usize)
            .map_err(hip)?;
        let mut normed = stream.alloc::<F16>(n * FFN_PADDED as usize).map_err(hip)?;
        // Q, K and transposed V; the padding stays zero.
        let mut q = stream.alloc::<F16>(head_values).map_err(hip)?;
        let mut kk = stream.alloc::<F16>(head_values).map_err(hip)?;
        let mut vt = stream.alloc::<F16>(head_values).map_err(hip)?;
        stream.fill_bytes(q.as_mut_slice(), 0).map_err(hip)?;
        stream.fill_bytes(kk.as_mut_slice(), 0).map_err(hip)?;
        stream.fill_bytes(vt.as_mut_slice(), 0).map_err(hip)?;
        let mut attention = stream.alloc::<F16>(hidden_n).map_err(hip)?;
        let mut pooled = stream
            .alloc::<f32>(rows as usize * HIDDEN as usize)
            .map_err(hip)?;
        stream
            .upload(&image.pixels, pixels.as_mut_slice())
            .map_err(hip)?;
        stream.upload(&table, rope.as_mut_slice()).map_err(hip)?;

        let mut observe = |stage: &str, data: DeviceSlice<'_, f32>| -> Result<(), String> {
            if let Some(observer) = observer.as_deref_mut() {
                let mut copy = vec![0.0f32; data.len()];
                stream.download(data, &mut copy).map_err(hip)?;
                observer(stage, &copy);
            }
            Ok(())
        };

        // Patch embedding in FP32, as the F32 kernel is stored.
        let patch_n = n * PATCH_VALUES as usize;
        launched(
            k::vision_patchify(
                stream,
                pixels.as_slice(),
                work.slice_mut(0..patch_n),
                width,
                columns,
                count,
                height,
            ),
            "patches",
        )?;
        blas(
            handle.sgemm(
                stream,
                Op::T,
                Op::N,
                HIDDEN as usize,
                n,
                PATCH_VALUES as usize,
                1.0,
                floats(&u.patch)?,
                PATCH_VALUES as usize,
                work.slice(0..patch_n),
                PATCH_VALUES as usize,
                hidden.as_mut_slice(),
                HIDDEN as usize,
            ),
            "Sgemm",
        )?;
        launched(
            k::vision_add_positions(
                stream,
                hidden.as_mut_slice(),
                floats(&u.position)?,
                columns,
                count,
            ),
            "positions",
        )?;
        observe("patch", hidden.as_slice())?;

        let (cos, sin) = (rope.slice(0..half), rope.slice(half..2 * half));
        let qkv_n = n * QKV as usize;
        let gate_up_n = n * GATE_UP as usize;
        let ffn_n = n * FFN_PADDED as usize;
        for (l, layer) in u.layers.iter().enumerate() {
            check_cancelled()?;
            launched(
                k::vision_rms_norm_half(
                    stream,
                    hidden.as_slice(),
                    Some(floats(&layer.ln1)?),
                    normed.slice_mut(0..hidden_n),
                    count,
                    HIDDEN,
                ),
                "norm",
            )?;
            gemm(
                stream,
                handle,
                &layer.qkv,
                &normed,
                &mut work,
                n,
                QKV as usize,
                HIDDEN as usize,
            )?;
            launched(
                k::vision_qkv_post(
                    stream,
                    work.slice(0..qkv_n),
                    floats(&layer.q_norm)?,
                    floats(&layer.k_norm)?,
                    cos,
                    sin,
                    q.as_mut_slice(),
                    kk.as_mut_slice(),
                    vt.as_mut_slice(),
                    columns,
                    padded,
                    count,
                    positions,
                ),
                "qkv",
            )?;
            launched(
                k::vision_attention(
                    stream,
                    q.as_slice(),
                    kk.as_slice(),
                    vt.as_slice(),
                    attention.as_mut_slice(),
                    count,
                    padded,
                ),
                "attention",
            )?;
            gemm(
                stream,
                handle,
                &layer.out,
                &attention,
                &mut work,
                n,
                HIDDEN as usize,
                HIDDEN as usize,
            )?;
            launched(
                k::vision_post_norm_add(
                    stream,
                    hidden.as_mut_slice(),
                    work.slice(0..hidden_n),
                    floats(&layer.attn_post_norm)?,
                    count,
                ),
                "attention residual",
            )?;
            launched(
                k::vision_rms_norm_half(
                    stream,
                    hidden.as_slice(),
                    Some(floats(&layer.ln2)?),
                    normed.slice_mut(0..hidden_n),
                    count,
                    HIDDEN,
                ),
                "norm",
            )?;
            gemm(
                stream,
                handle,
                &layer.gate_up,
                &normed,
                &mut work,
                n,
                GATE_UP as usize,
                HIDDEN as usize,
            )?;
            launched(
                k::vision_geglu(
                    stream,
                    work.slice(0..gate_up_n),
                    normed.slice_mut(0..ffn_n),
                    count,
                ),
                "geglu",
            )?;
            gemm(
                stream,
                handle,
                &layer.down,
                &normed,
                &mut work,
                n,
                HIDDEN as usize,
                FFN_PADDED as usize,
            )?;
            launched(
                k::vision_post_norm_add(
                    stream,
                    hidden.as_mut_slice(),
                    work.slice(0..hidden_n),
                    floats(&layer.ffn_post_norm)?,
                    count,
                ),
                "feed-forward residual",
            )?;
            observe(&format!("layer{l}"), hidden.as_slice())?;
        }

        launched(
            k::vision_pool(
                stream,
                hidden.as_slice(),
                pooled.as_mut_slice(),
                columns,
                pooled_columns,
                rows,
                count,
            ),
            "pooling",
        )?;
        observe("pooled", pooled.as_slice())?;
        let pooled_n = rows as usize * HIDDEN as usize;
        launched(
            k::vision_standardize_norm(
                stream,
                pooled.as_slice(),
                floats(&u.std_bias)?,
                floats(&u.std_scale)?,
                normed.slice_mut(0..pooled_n),
                rows,
            ),
            "standardization",
        )?;
        let mut embedding = allocate_device::<f32>(stream, rows as usize * out_width as usize)?;
        gemm(
            stream,
            handle,
            &u.projection,
            &normed,
            &mut embedding,
            rows as usize,
            out_width as usize,
            HIDDEN as usize,
        )?;
        // Freezing waits for the stream, as the C++ synchronizes here.
        let embedding = embedding.freeze(stream).map_err(hip)?;
        check_cancelled()?;
        observe("embedding", embedding.view())?;
        Ok(Arc::new(embedding))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sidecar_is_found_beside_the_target_or_one_level_up() {
        let root = std::env::temp_dir().join(format!("gufo-gemma4-sidecar-{}", std::process::id()));
        let models = root.join("models");
        let nested = models.join("gemma");
        std::fs::create_dir_all(&nested).expect("directories");
        let target = nested.join("gemma-4.gguf");
        assert_eq!(sidecar_path(&target, None), None);
        let above = models.join(SIDECAR_FILE_NAME);
        std::fs::write(&above, b"").expect("sidecar");
        assert_eq!(sidecar_path(&target, None), Some(above.clone()));
        let beside = nested.join(SIDECAR_FILE_NAME);
        std::fs::write(&beside, b"").expect("sidecar");
        assert_eq!(sidecar_path(&target, None), Some(beside));
        // Two levels up is too far.
        std::fs::remove_file(nested.join(SIDECAR_FILE_NAME)).expect("remove");
        std::fs::remove_file(&above).expect("remove");
        std::fs::write(root.join(SIDECAR_FILE_NAME), b"").expect("sidecar");
        assert_eq!(sidecar_path(&target, None), None);
        let explicit = Path::new("/elsewhere/mmproj.gguf");
        assert_eq!(
            sidecar_path(&target, Some(explicit)),
            Some(explicit.to_path_buf())
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_encoder_is_shared_across_threads() {
        fn shared<T: Send + Sync>() {}
        shared::<VisionEncoder>();
    }

    #[test]
    fn the_rope_table_holds_cosines_then_sines() {
        let table = rope_table(3);
        let pairs = ROPE_PAIRS as usize;
        assert_eq!(table.len(), 2 * 3 * pairs);
        // Position 0 does not rotate.
        assert!(table[..pairs].iter().all(|&c| c == 1.0));
        assert!(table[3 * pairs..4 * pairs].iter().all(|&s| s == 0.0));
        // Pair 0 rotates by the position itself.
        assert_eq!(table[2 * pairs], 2.0f64.cos() as f32);
        assert_eq!(table[(3 + 2) * pairs], 2.0f64.sin() as f32);
    }
}
