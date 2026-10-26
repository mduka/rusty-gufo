//! The device runtime (`hip/runtime.hip`): BF16 matrices from a caching
//! pool, weights uploaded on first use and kept resident, and the operators
//! the pipeline composes. Every operator writes a fresh matrix in stream
//! order, as in C++: the allocation policy (best fit up to twice the size,
//! at most 8 GiB cached) and the order of allocations are the C++ ones, so
//! the device memory peak matches.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, PoisonError};

use gufo_core::text::{Error, Result};
use gufo_hip::blas::{
    AlgoPolicy, Blas, BlasLt, Heuristic, Layout, MatmulBatch, MatmulShape, Op, Strides,
};
use gufo_hip::{
    Bf16, DeviceBuffer, DeviceIndices, DeviceSlice, DeviceSliceMut, HipError, IndexSpec, Launch,
    LaunchError, PinnedBuffer, Stream,
};
use gufo_qwen_image_21_kernels as k;

use crate::weights::Weights;

/// Freed blocks the pool keeps for reuse (`Pool::kMaxCachedBytes`).
const MAX_CACHED_BYTES: usize = 8 << 30;
/// The largest single tensor (`Runtime::Raw`).
const MAX_TENSOR_BYTES: usize = 32 << 30;
/// `hipErrorOutOfMemory`.
const OUT_OF_MEMORY: i32 = 2;
/// Each of the two page-locked buffers weight uploads go through.
const STAGING_BYTES: usize = 64 << 20;

pub(crate) fn hip_error(error: HipError) -> Error {
    Error::runtime(format!("Qwen-Image HIP: {error}"))
}

/// The weight upload buffers, allocated on first use. A copy straight from
/// the file mapping lets HIP pin the mapped page-cache pages, and every
/// reclaim of one then evicts the process's GPU queues.
fn staging_buffers(slot: &mut Option<[PinnedBuffer<u8>; 2]>) -> Result<&mut [PinnedBuffer<u8>; 2]> {
    match slot {
        Some(buffers) => Ok(buffers),
        None => {
            let piece = || PinnedBuffer::zeroed(STAGING_BYTES).map_err(hip_error);
            Ok(slot.insert([piece()?, piece()?]))
        }
    }
}

pub(crate) fn launch_error(error: LaunchError) -> Error {
    Error::runtime(format!("Qwen-Image launch: {error}"))
}

/// The C++ shape errors (`std::logic_error`).
fn logic(message: &str) -> Error {
    Error::logic(message)
}

/// Dimensions the kernels take as `int`.
fn dim(value: usize) -> Result<u32> {
    u32::try_from(value)
        .ok()
        .filter(|&v| v <= i32::MAX as u32)
        .ok_or_else(|| logic("Qwen-Image dimension exceeds int"))
}

// ---- Pool ----

/// `Runtime::Pool`: freed blocks by size, oldest first per size.
#[derive(Default)]
struct Pool {
    free: BTreeMap<usize, VecDeque<DeviceBuffer<u8>>>,
    cached: usize,
}

impl Pool {
    fn release(&mut self, buffer: DeviceBuffer<u8>) {
        let bytes = buffer.len();
        if bytes <= MAX_CACHED_BYTES - self.cached {
            self.free.entry(bytes).or_default().push_back(buffer);
            self.cached += bytes;
        }
    }

    /// The smallest cached block of at least `bytes` and at most twice that.
    fn take(&mut self, bytes: usize) -> Option<DeviceBuffer<u8>> {
        let (&size, _) = self.free.range(bytes..).next()?;
        if size > bytes.saturating_mul(2) {
            return None;
        }
        let queue = self.free.get_mut(&size)?;
        let buffer = queue.pop_front()?;
        if queue.is_empty() {
            self.free.remove(&size);
        }
        self.cached -= size;
        Some(buffer)
    }

    fn clear(&mut self) {
        self.free.clear();
        self.cached = 0;
    }
}

/// A pool block; dropping the last matrix over it returns it to the pool.
struct Block {
    buffer: Option<DeviceBuffer<u8>>,
    pool: Arc<Mutex<Pool>>,
}

impl Drop for Block {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .release(buffer);
        }
    }
}

// ---- Matrix ----

/// `Matrix`: a row-major BF16 view of a pool block. Clones and slices share
/// the block; `packed` marks the dense kernels' `[cols / 16][rows][16]`
/// layout. A matrix is written only while it is the block's sole owner.
#[derive(Clone)]
pub struct Matrix {
    block: Arc<Block>,
    /// Byte range of the view in the block.
    offset: usize,
    bytes: usize,
    pub rows: usize,
    pub cols: usize,
    pub packed: bool,
}

impl Matrix {
    pub fn size(&self) -> usize {
        self.rows * self.cols
    }

    fn raw(&self) -> Result<DeviceSlice<'_, u8>> {
        let buffer = self
            .block
            .buffer
            .as_ref()
            .ok_or_else(|| logic("released Qwen-Image tensor"))?;
        Ok(buffer
            .as_slice()
            .range(self.offset..self.offset + self.bytes))
    }

    fn raw_mut(&mut self) -> Result<DeviceSliceMut<'_, u8>> {
        let range = self.offset..self.offset + self.bytes;
        let block = Arc::get_mut(&mut self.block)
            .ok_or_else(|| logic("Qwen-Image tensor written while shared"))?;
        let buffer = block
            .buffer
            .as_mut()
            .ok_or_else(|| logic("released Qwen-Image tensor"))?;
        Ok(buffer.as_mut_slice().range(range))
    }

    /// The BF16 elements.
    pub fn bf16(&self) -> Result<DeviceSlice<'_, Bf16>> {
        self.raw()?
            .cast()
            .ok_or_else(|| logic("misaligned Qwen-Image tensor"))
    }

    pub fn bf16_mut(&mut self) -> Result<DeviceSliceMut<'_, Bf16>> {
        self.raw_mut()?
            .cast()
            .ok_or_else(|| logic("misaligned Qwen-Image tensor"))
    }

    /// The bytes as FP32: rotary tables (two BF16 slots per float) and FP32
    /// scratch.
    pub fn f32(&self) -> Result<DeviceSlice<'_, f32>> {
        self.raw()?
            .cast()
            .ok_or_else(|| logic("misaligned Qwen-Image tensor"))
    }

    pub fn f32_mut(&mut self) -> Result<DeviceSliceMut<'_, f32>> {
        self.raw_mut()?
            .cast()
            .ok_or_else(|| logic("misaligned Qwen-Image tensor"))
    }

    /// `Matrix::Slice`: rows `start..start + count`.
    pub fn slice(&self, start: usize, count: usize) -> Result<Matrix> {
        if self.packed || start > self.rows || count > self.rows - start {
            return Err(logic("invalid Qwen-Image tensor slice"));
        }
        Ok(Matrix {
            block: Arc::clone(&self.block),
            offset: self.offset + start * self.cols * 2,
            bytes: count * self.cols * 2,
            rows: count,
            cols: self.cols,
            packed: false,
        })
    }

    /// `Matrix::Reshape`.
    pub fn reshape(&self, rows: usize, cols: usize) -> Result<Matrix> {
        if self.packed || rows == 0 || cols == 0 || rows * cols != self.size() {
            return Err(logic("invalid Qwen-Image tensor reshape"));
        }
        Ok(Matrix {
            rows,
            cols,
            ..self.clone()
        })
    }

    /// Whether `self` and `other` view the same elements with the same
    /// shape (`PackedInput`'s identity).
    fn same_view(&self, other: &Matrix) -> bool {
        Arc::ptr_eq(&self.block, &other.block)
            && self.offset == other.offset
            && self.rows == other.rows
            && self.cols == other.cols
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norm {
    Layer = 0,
    Rms = 1,
    ZeroRms = 2,
    Vae = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Silu = 0,
    GeluTanh = 1,
    Gelu = 2,
}

/// The development observer (`Observer`): FP32 views of BF16 boundaries,
/// filtered before they are copied to the host.
pub trait Observer {
    fn wants(&self, name: &str) -> bool;
    fn write(&mut self, name: &str, values: &[f32], rows: usize, cols: usize);
}

/// `DenseShape`: the projection shapes of the dense WMMA kernels.
fn dense_shape(channels: usize, inner: usize) -> bool {
    (channels == 4096 && (inner == 4096 || inner == 12288)) || (channels == 12288 && inner == 4096)
}

/// `Convolve3Shape`: the channel counts of the implicit 3x3 convolution.
fn convolve3_shape(channels: usize, output_channels: usize) -> bool {
    ((96..=384).contains(&channels)
        && channels.is_multiple_of(16)
        && (96..=384).contains(&output_channels)
        && output_channels.is_multiple_of(16))
        || ((channels == 576 || channels == 1152) && matches!(output_channels, 288 | 576 | 1152))
}

/// The hipBLASLt plan policy of `blas.cpp`: a fixed heuristic rank per
/// shape class, measured on the DiT and VAE shapes (the default choices of
/// the FP32-output shapes are 4–6x slower on gfx1151); the result must need
/// no workspace.
struct RankPolicy(usize);

impl RankPolicy {
    fn new(rows: usize, channels: usize, float_output: bool) -> Self {
        Self(if float_output {
            if channels < 256 { 5 } else { 4 }
        } else if rows <= 512 {
            12
        } else if rows >= 4096 {
            0
        } else {
            1
        })
    }
}

impl AlgoPolicy for RankPolicy {
    fn choose(&self, ranked: &[Heuristic], _workspace_capacity: usize) -> Option<usize> {
        (!ranked.is_empty()).then(|| self.0.min(ranked.len() - 1))
    }
}

/// The attention limits of the last masked attention, uploaded once for
/// every layer that reuses them.
struct Limits {
    values: Vec<i32>,
    keys: usize,
    indices: DeviceIndices,
}

/// `Runtime`.
pub struct Runtime {
    pool: Arc<Mutex<Pool>>,
    device_weights: HashMap<String, Matrix>,
    prefetched: Vec<&'static str>,
    packed_source: Option<Matrix>,
    packed_input: Option<Matrix>,
    weight_bytes: usize,
    limits: Option<Limits>,
    /// Index uploads of this request, freed once the stream is idle.
    indices: Vec<DeviceIndices>,
    /// Page-locked buffers the weights are uploaded through, allocated with
    /// the first weight (see [`Stream::upload_bytes_staged`]).
    staging: Option<[PinnedBuffer<u8>; 2]>,
    lt: BlasLt,
    blas: Blas,
    stream: Stream,
}

impl Runtime {
    pub fn new() -> Result<Self> {
        let stream = Stream::new().map_err(hip_error)?;
        let blas = Blas::with_default_atomics(&stream)
            .map_err(|e| Error::runtime(format!("Qwen-Image hipBLAS failed: {e}")))?;
        let lt = BlasLt::new(&stream, 0)
            .map_err(|e| Error::runtime(format!("Qwen-Image hipBLASLt failed: {e}")))?;
        Ok(Self {
            pool: Arc::default(),
            device_weights: HashMap::new(),
            prefetched: Vec::new(),
            packed_source: None,
            packed_input: None,
            weight_bytes: 0,
            limits: None,
            indices: Vec::new(),
            staging: None,
            lt,
            blas,
            stream,
        })
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(hip_error)
    }

    /// Waits for the stream, then frees this request's index uploads.
    pub fn finish_request(&mut self) -> Result<()> {
        let result = self.synchronize();
        self.indices.clear();
        self.limits = None;
        result
    }

    /// `Runtime::Raw`: an uninitialised pool block of `bytes`.
    fn raw(&mut self, bytes: usize) -> Result<Matrix> {
        if bytes == 0 || bytes > MAX_TENSOR_BYTES {
            return Err(Error::length("Qwen-Image tensor exceeds memory limit"));
        }
        let cached = self
            .pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take(bytes);
        let buffer = match cached {
            Some(buffer) => buffer,
            None => match self.stream.alloc_persistent::<u8>(bytes) {
                Ok(buffer) => buffer,
                Err(error) if error.code == OUT_OF_MEMORY => {
                    let mut pool = self.pool.lock().unwrap_or_else(PoisonError::into_inner);
                    if pool.free.is_empty() {
                        return Err(hip_error(error));
                    }
                    pool.clear();
                    drop(pool);
                    self.stream
                        .alloc_persistent::<u8>(bytes)
                        .map_err(hip_error)?
                }
                Err(error) => return Err(hip_error(error)),
            },
        };
        Ok(Matrix {
            block: Arc::new(Block {
                buffer: Some(buffer),
                pool: Arc::clone(&self.pool),
            }),
            offset: 0,
            bytes,
            rows: 1,
            cols: 1,
            packed: false,
        })
    }

    /// `Runtime::New`.
    pub fn new_matrix(&mut self, rows: usize, cols: usize) -> Result<Matrix> {
        if rows == 0 || cols == 0 {
            return Err(logic("empty Qwen-Image tensor"));
        }
        let bytes = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| Error::length("Qwen-Image tensor exceeds memory limit"))?;
        let mut matrix = self.raw(bytes)?;
        matrix.bytes = bytes;
        matrix.rows = rows;
        matrix.cols = cols;
        Ok(matrix)
    }

    /// `Runtime::Upload`: FP32 host values rounded to BF16.
    pub fn upload(&mut self, values: &[f32], rows: usize, cols: usize) -> Result<Matrix> {
        if values.len() != rows * cols {
            return Err(logic("Qwen-Image upload size mismatch"));
        }
        let mut result = self.new_matrix(rows, cols)?;
        let packed: Vec<Bf16> = values.iter().map(|&v| Bf16::from_f32(v)).collect();
        self.stream
            .upload(&packed, result.bf16_mut()?)
            .map_err(hip_error)?;
        Ok(result)
    }

    /// `Runtime::UploadFloat`: FP32 tables in two BF16 slots per float.
    pub fn upload_float(&mut self, values: &[f32], rows: usize, cols: usize) -> Result<Matrix> {
        if values.len() != rows * cols {
            return Err(logic("FP32 upload size mismatch"));
        }
        let mut out = self.new_matrix(rows, cols * 2)?;
        self.stream
            .upload(values, out.f32_mut()?)
            .map_err(hip_error)?;
        Ok(out)
    }

    /// `Runtime::Weight`: uploads a weight on first use, converting FP32 to
    /// BF16 and packing dense projections and 3x3 convolutions for the WMMA
    /// kernels. The first weight of a component faults the whole component
    /// in first.
    pub fn weight(&mut self, weights: &Weights, name: &str) -> Result<Matrix> {
        if let Some(matrix) = self.device_weights.get(name) {
            return Ok(matrix.clone());
        }
        for prefix in [
            "transformer.",
            "text_encoder.model.language_model.",
            "text_encoder.model.visual.",
            "vae.encoder.",
            "vae.decoder.",
        ] {
            if name.starts_with(prefix) && !self.prefetched.contains(&prefix) {
                weights.prefetch(prefix)?;
                self.prefetched.push(prefix);
                break;
            }
        }
        let w = weights.get(name)?;
        let rows = w.shape.first().copied().unwrap_or(1);
        let count = w.bytes() / if w.bf16 { 2 } else { 4 };
        if count / rows > i32::MAX as usize {
            return Err(Error::runtime(
                "Qwen-Image weight exceeds matrix dimensions",
            ));
        }
        let cols = count / rows;
        let mut m = self.new_matrix(rows, cols)?;
        let host = weights.bytes(w);
        if w.bf16 {
            let staging = staging_buffers(&mut self.staging)?;
            self.stream
                .upload_bytes_staged(host, m.bf16_mut()?, staging)
                .map_err(hip_error)?;
        } else {
            let mut raw = self.raw(w.bytes())?;
            let staging = staging_buffers(&mut self.staging)?;
            self.stream
                .upload_bytes_staged(host, raw.f32_mut()?, staging)
                .map_err(hip_error)?;
            k::float_to_bf16(&self.stream, raw.f32()?, m.bf16_mut()?, count)
                .map_err(launch_error)?;
        }
        let packed_convolution = w.shape.len() == 4
            && w.shape[2] == 3
            && w.shape[3] == 3
            && convolve3_shape(w.shape[1], rows);
        let packed_dense = w.shape.len() == 2
            && dense_shape(rows, cols)
            && name.ends_with(".weight")
            && !weights.contains(&format!("{}.bias", &name[..name.len() - 7]));
        if packed_convolution || packed_dense {
            let mut packed = self.new_matrix(rows, cols)?;
            k::pack_dense(
                &self.stream,
                m.bf16()?,
                packed.bf16_mut()?,
                dim(rows)?,
                dim(cols)?,
            )
            .map_err(launch_error)?;
            packed.packed = true;
            m = packed;
        }
        self.synchronize()?;
        self.weight_bytes += count * 2;
        self.device_weights.insert(name.to_owned(), m.clone());
        Ok(m)
    }

    /// `Runtime::PackedInput`: the packed copy of `x`, cached for the next
    /// projection of the same input (QKV, gate/up). The cache keeps `x`
    /// alive, so a new tensor cannot reuse its block and pass for it.
    fn packed_input(&mut self, x: &Matrix) -> Result<Matrix> {
        if x.packed {
            return Ok(x.clone());
        }
        if !self
            .packed_source
            .as_ref()
            .is_some_and(|source| source.same_view(x))
        {
            let mut packed = self.new_matrix(x.rows, x.cols)?;
            k::pack_dense(
                &self.stream,
                x.bf16()?,
                packed.bf16_mut()?,
                dim(x.rows)?,
                dim(x.cols)?,
            )
            .map_err(launch_error)?;
            packed.packed = true;
            self.packed_source = Some(x.clone());
            self.packed_input = Some(packed);
        }
        self.packed_input
            .clone()
            .ok_or_else(|| logic("missing Qwen-Image packed input"))
    }

    /// `Runtime::Matmul`: `output = x · wᵀ (+ bias)` into `output`, which
    /// holds `x.rows × w.rows` elements.
    fn matmul(
        &mut self,
        x: &Matrix,
        w: &Matrix,
        mut output: DeviceSliceMut<'_, Bf16>,
        bias: Option<&Matrix>,
        convolution: bool,
    ) -> Result<()> {
        let (rows, channels, inner) = (x.rows, w.rows, x.cols);
        if x.cols != w.cols || output.len() != rows * channels {
            return Err(logic("Qwen-Image GEMM shape mismatch"));
        }
        if bias.is_some_and(|b| b.size() != channels) {
            return Err(logic("Qwen-Image bias shape mismatch"));
        }
        if w.packed {
            if bias.is_some() {
                return Err(logic("Qwen-Image packed projection contract mismatch"));
            }
            let packed = self.packed_input(x)?;
            let launch = k::dense(
                &self.stream,
                w.bf16()?,
                packed.bf16()?,
                output.reborrow(),
                dim(rows)?,
                dim(channels)?,
                dim(inner)?,
            )
            .map_err(launch_error)?;
            if launch == Launch::Unsupported {
                return Err(logic("unsupported Qwen-Image packed projection"));
            }
            return Ok(());
        }
        if x.packed {
            return Err(logic(
                "Qwen-Image packed input requires a packed projection",
            ));
        }
        let shape = MatmulShape {
            trans_a: Some(Op::T),
            trans_b: None,
            a: Layout::new(inner, channels, inner),
            b: Layout::new(inner, rows, inner),
            c: Layout::new(channels, rows, channels),
            batch: None,
        };
        let policy = RankPolicy::new(rows, channels, bias.is_some());
        if let Some(bias) = bias {
            let mut accumulation = self.raw(output.len() * 4)?;
            let mut product = accumulation.f32_mut()?;
            let launched = rows >= 64
                && self
                    .lt
                    .matmul(
                        &self.stream,
                        &shape,
                        w.bf16()?,
                        x.bf16()?,
                        product.reborrow(),
                        &policy,
                    )
                    .map_err(launch_error)?
                    == Launch::Launched;
            if !launched {
                self.blas
                    .gemm(
                        &self.stream,
                        Op::T,
                        Op::N,
                        channels,
                        rows,
                        inner,
                        w.bf16()?,
                        w.cols,
                        x.bf16()?,
                        x.cols,
                        product,
                        channels,
                    )
                    .map_err(launch_error)?;
            }
            k::add_bias(
                &self.stream,
                accumulation.f32()?,
                bias.bf16()?,
                output,
                dim(channels)?,
                rows * channels,
                convolution,
            )
            .map_err(launch_error)?;
        } else {
            let launched = rows >= 64
                && self
                    .lt
                    .matmul(
                        &self.stream,
                        &shape,
                        w.bf16()?,
                        x.bf16()?,
                        output.reborrow(),
                        &policy,
                    )
                    .map_err(launch_error)?
                    == Launch::Launched;
            if !launched {
                self.blas
                    .gemm(
                        &self.stream,
                        Op::T,
                        Op::N,
                        channels,
                        rows,
                        inner,
                        w.bf16()?,
                        w.cols,
                        x.bf16()?,
                        x.cols,
                        output,
                        channels,
                    )
                    .map_err(launch_error)?;
            }
        }
        Ok(())
    }

    /// `Runtime::Linear`.
    pub fn linear(&mut self, weights: &Weights, x: &Matrix, name: &str) -> Result<Matrix> {
        let weight_name = format!("{name}.weight");
        let w = self.weight(weights, &weight_name)?;
        let mut out = self.new_matrix(x.rows, w.rows)?;
        let bias_name = format!("{name}.bias");
        let bias = if weights.contains(&bias_name) {
            Some(self.weight(weights, &bias_name)?)
        } else {
            None
        };
        let convolution = weights.get(&weight_name)?.shape.len() > 2;
        self.matmul(x, &w, out.bf16_mut()?, bias.as_ref(), convolution)?;
        Ok(out)
    }

    /// `Runtime::Normalize`; `weight` names the parameter without suffix.
    pub fn normalize(
        &mut self,
        weights: &Weights,
        x: &Matrix,
        kind: Norm,
        weight: Option<&str>,
        epsilon: f32,
        silu: bool,
    ) -> Result<Matrix> {
        let (mut w, mut bias) = (None, None);
        if let Some(weight) = weight {
            let suffix = if kind == Norm::Vae {
                ".gamma"
            } else {
                ".weight"
            };
            let matrix = self.weight(weights, &format!("{weight}{suffix}"))?;
            let bias_name = format!("{weight}.bias");
            if weights.contains(&bias_name) {
                bias = Some(self.weight(weights, &bias_name)?);
            }
            if matrix.size() != x.cols {
                return Err(logic("Qwen-Image norm shape mismatch"));
            }
            if bias.as_ref().is_some_and(|b| b.size() != matrix.size()) {
                return Err(logic("Qwen-Image norm bias shape mismatch"));
            }
            w = Some(matrix);
        }
        let mut out = self.new_matrix(x.rows, x.cols)?;
        let launch = k::normalize_rows(
            &self.stream,
            x.bf16()?,
            out.bf16_mut()?,
            w.as_ref().map(Matrix::bf16).transpose()?,
            bias.as_ref().map(Matrix::bf16).transpose()?,
            dim(x.rows)?,
            dim(x.cols)?,
            kind as u32,
            epsilon,
            silu,
        )
        .map_err(launch_error)?;
        if launch == Launch::Unsupported {
            return Err(logic("Qwen-Image norm without weight"));
        }
        Ok(out)
    }

    /// `Runtime::Activate`.
    pub fn activate(&mut self, x: &Matrix, activation: Activation) -> Result<Matrix> {
        let mut out = self.new_matrix(x.rows, x.cols)?;
        k::activate(
            &self.stream,
            x.bf16()?,
            out.bf16_mut()?,
            x.size(),
            activation as u32,
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// `Runtime::SwiGlu`.
    pub fn swiglu(&mut self, gate: &Matrix, up: &Matrix) -> Result<Matrix> {
        if gate.rows != up.rows || gate.cols != up.cols {
            return Err(logic("SwiGLU shape"));
        }
        let mut out = self.new_matrix(gate.rows, gate.cols)?;
        k::swiglu(
            &self.stream,
            gate.bf16()?,
            up.bf16()?,
            out.bf16_mut()?,
            gate.size(),
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// `Runtime::GatedLinear`: the fused SwiGLU projection; the output keeps
    /// the packed layout.
    pub fn gated_linear(
        &mut self,
        weights: &Weights,
        x: &Matrix,
        gate: &str,
        up: &str,
    ) -> Result<Matrix> {
        let g = self.weight(weights, &format!("{gate}.weight"))?;
        let u = self.weight(weights, &format!("{up}.weight"))?;
        if !g.packed || !u.packed || g.rows != u.rows || g.cols != u.cols || x.cols != g.cols {
            return Err(logic("Qwen-Image gated projection contract mismatch"));
        }
        let mut out = self.new_matrix(x.rows, g.rows)?;
        let packed = self.packed_input(x)?;
        let launch = k::dense_swiglu(
            &self.stream,
            g.bf16()?,
            u.bf16()?,
            packed.bf16()?,
            out.bf16_mut()?,
            dim(x.rows)?,
            dim(g.rows)?,
            dim(x.cols)?,
        )
        .map_err(launch_error)?;
        if launch == Launch::Unsupported {
            return Err(logic("unsupported Qwen-Image gated projection"));
        }
        out.packed = true;
        Ok(out)
    }

    /// `Runtime::Add`.
    pub fn add(&mut self, a: &Matrix, b: &Matrix) -> Result<Matrix> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(logic("residual shape"));
        }
        let mut out = self.new_matrix(a.rows, a.cols)?;
        k::add(
            &self.stream,
            a.bf16()?,
            b.bf16()?,
            out.bf16_mut()?,
            a.size(),
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// `Runtime::ModulationFactors`.
    pub fn modulation_factors(&mut self, modulation: &Matrix, gates: bool) -> Result<Matrix> {
        if modulation.rows != 2 || modulation.cols != if gates { 16384 } else { 4096 } {
            return Err(logic("Qwen-Image modulation factors shape mismatch"));
        }
        let mut out = self.new_matrix(modulation.rows, modulation.cols)?;
        k::prepare_modulation(
            &self.stream,
            modulation.bf16()?,
            out.bf16_mut()?,
            dim(modulation.size())?,
            gates,
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// `Runtime::NormalizeModulate`.
    pub fn normalize_modulate(
        &mut self,
        x: &Matrix,
        modulation: &Matrix,
        part: usize,
        prefix: usize,
    ) -> Result<Matrix> {
        if x.cols != 4096
            || modulation.rows != 2
            || (modulation.cols != 4096 && modulation.cols != 16384)
            || part >= modulation.cols / x.cols
            || prefix > x.rows
        {
            return Err(logic("Qwen-Image normalized modulation shape mismatch"));
        }
        let mut out = self.new_matrix(x.rows, x.cols)?;
        k::normalize_modulate(
            &self.stream,
            x.bf16()?,
            modulation.bf16()?,
            out.bf16_mut()?,
            dim(x.rows)?,
            dim(modulation.cols)?,
            dim(part)?,
            dim(prefix)?,
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// `Runtime::Modulate`.
    pub fn modulate(
        &mut self,
        x: &Matrix,
        modulation: &Matrix,
        part: usize,
        prefix: usize,
        residual: Option<&Matrix>,
    ) -> Result<Matrix> {
        if modulation.rows != 2
            || (modulation.cols != x.cols && modulation.cols != x.cols * 4)
            || part >= modulation.cols / x.cols
            || prefix > x.rows
            || residual.is_some_and(|r| r.rows != x.rows || r.cols != x.cols)
        {
            return Err(logic("Qwen-Image modulation shape mismatch"));
        }
        let mut out = self.new_matrix(x.rows, x.cols)?;
        k::modulate(
            &self.stream,
            x.bf16()?,
            modulation.bf16()?,
            residual.map(Matrix::bf16).transpose()?,
            out.bf16_mut()?,
            dim(x.rows)?,
            dim(x.cols)?,
            dim(modulation.cols)?,
            dim(part)?,
            dim(prefix)?,
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// `Runtime::Rope`.
    pub fn rope(
        &mut self,
        x: &Matrix,
        heads: usize,
        head_dim: usize,
        frequencies: &Matrix,
        mode: u32,
    ) -> Result<Matrix> {
        if x.cols != heads * head_dim
            || frequencies.rows != x.rows
            || frequencies.cols != head_dim * 2
        {
            return Err(logic("rotary shape mismatch"));
        }
        let mut out = self.new_matrix(x.rows, x.cols)?;
        k::rotary(
            &self.stream,
            x.bf16()?,
            out.bf16_mut()?,
            frequencies.f32()?,
            dim(x.rows)?,
            dim(heads)?,
            dim(head_dim)?,
            mode,
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// `Runtime::NormalizeRope`.
    pub fn normalize_rope(
        &mut self,
        weights: &Weights,
        x: &Matrix,
        weight: &str,
        heads: usize,
        frequencies: &Matrix,
    ) -> Result<Matrix> {
        let w = self.weight(weights, &format!("{weight}.weight"))?;
        if x.packed
            || x.cols != heads * 128
            || w.size() != 128
            || frequencies.rows != x.rows
            || frequencies.cols != 256
        {
            return Err(logic("Qwen-Image normalized rotary shape mismatch"));
        }
        let mut out = self.new_matrix(x.rows, x.cols)?;
        k::normalize_rotary128(
            &self.stream,
            x.bf16()?,
            out.bf16_mut()?,
            w.bf16()?,
            frequencies.f32()?,
            dim(x.rows)?,
            dim(heads)?,
            1e-6,
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// The device limits for `values`, reusing the last upload.
    fn limits(&mut self, values: &[i32], keys: usize) -> Result<()> {
        if self
            .limits
            .as_ref()
            .is_some_and(|l| l.keys == keys && l.values == values)
        {
            return Ok(());
        }
        let bound = u32::try_from(keys + 1).map_err(|_| logic("attention shape mismatch"))?;
        let buffer = self
            .stream
            .alloc_persistent::<u32>(values.len())
            .map_err(hip_error)?;
        let mut indices = self
            .stream
            .zeroed_indices_with(buffer, IndexSpec::below(bound).signed())
            .map_err(hip_error)?;
        let host: Vec<u32> = values.iter().map(|&v| v as u32).collect();
        self.stream
            .upload_indices(&host, &mut indices)
            .map_err(hip_error)?;
        if let Some(previous) = self.limits.take() {
            self.indices.push(previous.indices);
        }
        self.limits = Some(Limits {
            values: values.to_vec(),
            keys,
            indices,
        });
        Ok(())
    }

    /// `Runtime::Attention`: grouped-query attention; `key_limits[q]` keys
    /// are visible to query row q when given.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &mut self,
        q: &Matrix,
        k_matrix: &Matrix,
        v: &Matrix,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        key_limits: &[i32],
    ) -> Result<Matrix> {
        let keys = k_matrix.rows;
        if q.cols != heads * head_dim
            || k_matrix.cols != kv_heads * head_dim
            || v.cols != k_matrix.cols
            || v.rows != keys
            || kv_heads == 0
            || !heads.is_multiple_of(kv_heads)
            || (!key_limits.is_empty() && key_limits.len() != q.rows)
        {
            return Err(logic("attention shape mismatch"));
        }
        let mut qh = self.new_matrix(q.rows, heads * head_dim)?;
        let mut kh = self.new_matrix(keys, heads * head_dim)?;
        let mut vh = self.new_matrix(v.rows, heads * head_dim)?;
        let mut out = self.new_matrix(q.rows, q.cols)?;
        let (rows, h, d) = (dim(q.rows)?, dim(heads)?, dim(head_dim)?);
        k::reorder_heads(&self.stream, q.bf16()?, qh.bf16_mut()?, rows, h, h, d)
            .map_err(launch_error)?;
        k::reorder_heads(
            &self.stream,
            k_matrix.bf16()?,
            kh.bf16_mut()?,
            dim(keys)?,
            h,
            dim(kv_heads)?,
            d,
        )
        .map_err(launch_error)?;
        k::reorder_heads(
            &self.stream,
            v.bf16()?,
            vh.bf16_mut()?,
            dim(v.rows)?,
            h,
            dim(kv_heads)?,
            d,
        )
        .map_err(launch_error)?;
        let masked = !key_limits.is_empty();
        if masked {
            if key_limits.iter().any(|&n| n < 1 || n as usize > keys) {
                return Err(logic("invalid attention key limit"));
            }
            self.limits(key_limits, keys)?;
        }
        if head_dim == 128 && keys >= 512 && q.rows >= 128 {
            let mut masked_rows = q.rows;
            while masked && masked_rows > 0 && key_limits[masked_rows - 1] as usize == keys {
                masked_rows -= 1;
            }
            let count = dim(keys)?;
            // First-step target image rows see every key: the unmasked
            // kernel serves that suffix.
            let split = masked && q.rows - masked_rows >= 128;
            let masked_count = if split { masked_rows } else { q.rows };
            if masked && masked_count > 0 {
                let limits = self
                    .limits
                    .as_ref()
                    .ok_or_else(|| logic("missing attention limits"))?;
                k::fused_attention128_masked(
                    &self.stream,
                    qh.bf16()?,
                    kh.bf16()?,
                    vh.bf16()?,
                    out.bf16_mut()?,
                    h,
                    rows,
                    count,
                    0,
                    dim(masked_count)?,
                    limits.indices.view(),
                )
                .map_err(launch_error)?;
            }
            if !masked || split {
                let start = if split { masked_rows } else { 0 };
                k::fused_attention128(
                    &self.stream,
                    qh.bf16()?,
                    kh.bf16()?,
                    vh.bf16()?,
                    out.bf16_mut()?,
                    h,
                    rows,
                    count,
                    dim(start)?,
                    dim(q.rows - start)?,
                )
                .map_err(launch_error)?;
            }
            return Ok(out);
        }
        let mut oh = self.new_matrix(q.rows, heads * head_dim)?;
        let mut vf = self.raw(vh.size() * 4)?;
        let mut of = self.raw(oh.size() * 4)?;
        k::bf16_to_float(&self.stream, vh.bf16()?, vf.f32_mut()?, vh.size())
            .map_err(launch_error)?;
        let tile = q.rows.min(128);
        let mut scores = self.raw(heads * tile * keys * 4)?;
        let mut probabilities = self.raw(heads * tile * keys * 4)?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut start = 0;
        while start < q.rows {
            let count = tile.min(q.rows - start);
            let queries = qh
                .bf16()?
                .range(start * head_dim..q.rows * heads * head_dim);
            let shape = MatmulShape {
                trans_a: Some(Op::T),
                trans_b: None,
                a: Layout::new(head_dim, keys, head_dim),
                b: Layout::new(head_dim, count, head_dim),
                c: Layout::new(keys, count, keys),
                batch: Some(MatmulBatch {
                    count: heads,
                    a: keys * head_dim,
                    b: q.rows * head_dim,
                    c: keys * count,
                }),
            };
            let policy = RankPolicy::new(count, keys, true);
            let launched = count >= 64
                && self
                    .lt
                    .matmul(
                        &self.stream,
                        &shape,
                        kh.bf16()?,
                        queries,
                        scores.f32_mut()?,
                        &policy,
                    )
                    .map_err(launch_error)?
                    == Launch::Launched;
            if !launched {
                self.blas
                    .gemm_strided_batched(
                        &self.stream,
                        Op::T,
                        Op::N,
                        keys,
                        count,
                        head_dim,
                        kh.bf16()?,
                        head_dim,
                        queries,
                        head_dim,
                        scores.f32_mut()?,
                        keys,
                        Strides {
                            count: heads,
                            a: keys * head_dim,
                            b: q.rows * head_dim,
                            c: keys * count,
                        },
                    )
                    .map_err(launch_error)?;
            }
            if masked {
                let limits = self
                    .limits
                    .as_ref()
                    .ok_or_else(|| logic("missing attention limits"))?;
                k::softmax_masked(
                    &self.stream,
                    scores.f32()?,
                    probabilities.f32_mut()?,
                    h,
                    dim(count)?,
                    dim(keys)?,
                    dim(start)?,
                    limits.indices.view(),
                    scale,
                )
                .map_err(launch_error)?;
            } else {
                k::softmax(
                    &self.stream,
                    scores.f32()?,
                    probabilities.f32_mut()?,
                    h,
                    dim(count)?,
                    dim(keys)?,
                    dim(start)?,
                    scale,
                )
                .map_err(launch_error)?;
            }
            let output = of.f32_mut()?;
            let output_len = output.len();
            self.blas
                .gemm_strided_batched(
                    &self.stream,
                    Op::N,
                    Op::N,
                    head_dim,
                    count,
                    keys,
                    vf.f32()?,
                    head_dim,
                    probabilities.f32()?,
                    keys,
                    output.range(start * head_dim..output_len),
                    head_dim,
                    Strides {
                        count: heads,
                        a: keys * head_dim,
                        b: keys * count,
                        c: q.rows * head_dim,
                    },
                )
                .map_err(launch_error)?;
            start += tile;
        }
        let size = oh.size();
        k::float_to_bf16(&self.stream, of.f32()?, oh.bf16_mut()?, size).map_err(launch_error)?;
        k::restore_heads(&self.stream, oh.bf16()?, out.bf16_mut()?, rows, h, d)
            .map_err(launch_error)?;
        Ok(out)
    }

    /// `Runtime::Concat`: rows of every matrix in order.
    pub fn concat(&mut self, matrices: &[&Matrix]) -> Result<Matrix> {
        let first = matrices
            .first()
            .ok_or_else(|| logic("empty tensor concatenation"))?;
        if matrices.iter().any(|m| m.cols != first.cols) {
            return Err(logic("concat shape"));
        }
        let rows = matrices.iter().map(|m| m.rows).sum();
        let mut out = self.new_matrix(rows, first.cols)?;
        let mut offset = 0;
        for m in matrices {
            let size = m.size();
            let destination = out.bf16_mut()?.range(offset..offset + size);
            self.stream
                .copy(m.bf16()?, destination)
                .map_err(hip_error)?;
            offset += size;
        }
        Ok(out)
    }

    /// `Runtime::Columns`.
    pub fn columns(&mut self, x: &Matrix, start: usize, count: usize) -> Result<Matrix> {
        if count == 0 || start > x.cols || count > x.cols - start {
            return Err(logic("column slice"));
        }
        let mut out = self.new_matrix(x.rows, count)?;
        k::copy_columns(
            &self.stream,
            x.bf16()?,
            out.bf16_mut()?,
            dim(x.rows)?,
            dim(x.cols)?,
            dim(start)?,
            dim(count)?,
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// Uploads `values` as indices below `bound`, kept until the request
    /// ends.
    fn indices(&mut self, values: &[u32], spec: IndexSpec) -> Result<usize> {
        let buffer = self
            .stream
            .alloc_persistent::<u32>(values.len())
            .map_err(hip_error)?;
        let mut indices = self
            .stream
            .zeroed_indices_with(buffer, spec)
            .map_err(hip_error)?;
        self.stream
            .upload_indices(values, &mut indices)
            .map_err(hip_error)?;
        self.indices.push(indices);
        Ok(self.indices.len() - 1)
    }

    /// `Runtime::Embed`: rows `ids` of `table`.
    pub fn embed(&mut self, table: &Matrix, ids: &[u32]) -> Result<Matrix> {
        if ids.iter().any(|&id| id as usize >= table.rows) {
            return Err(logic("token ID range"));
        }
        let index = self.indices(ids, IndexSpec::below(dim(table.rows)?))?;
        let mut out = self.new_matrix(ids.len(), table.cols)?;
        k::embedding(
            &self.stream,
            table.bf16()?,
            out.bf16_mut()?,
            self.indices[index].view(),
            dim(ids.len())?,
            dim(table.cols)?,
            dim(table.rows)?,
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// `Runtime::ReplaceRows`: `x` with rows `positions` replaced by (or
    /// added to) the rows of `values`.
    pub fn replace_rows(
        &mut self,
        x: &Matrix,
        values: &Matrix,
        positions: &[usize],
        add: bool,
    ) -> Result<Matrix> {
        if values.rows != positions.len() || values.cols != x.cols {
            return Err(logic("image scatter shape"));
        }
        if positions.iter().any(|&i| i >= x.rows) {
            return Err(logic("image scatter index"));
        }
        let host: Vec<u32> = positions.iter().map(|&i| i as u32).collect();
        let index = self.indices(&host, IndexSpec::below(dim(x.rows)?).signed())?;
        let mut out = self.new_matrix(x.rows, x.cols)?;
        self.stream
            .copy(x.bf16()?, out.bf16_mut()?)
            .map_err(hip_error)?;
        k::scatter_rows(
            &self.stream,
            values.bf16()?,
            out.bf16_mut()?,
            self.indices[index].view(),
            dim(values.rows)?,
            dim(values.cols)?,
            add,
            dim(x.rows)?,
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// `Runtime::Conv`: a square convolution of `[height * width][channels]`
    /// rows; `padding` -1 pads only bottom and right, `upscale` 2 doubles the
    /// input by nearest neighbour first.
    #[allow(clippy::too_many_arguments)]
    pub fn conv(
        &mut self,
        weights: &Weights,
        x: &Matrix,
        height: usize,
        width: usize,
        name: &str,
        stride: usize,
        padding: i32,
        upscale: usize,
    ) -> Result<Matrix> {
        let weight_name = format!("{name}.weight");
        let spec = weights.get(&weight_name)?.shape.clone();
        if spec.len() != 4 || spec[1] != x.cols || spec[2] != spec[3] || x.rows != height * width {
            return Err(logic(&format!("convolution shape: {name}")));
        }
        let kernel = spec[2];
        let total_padding = if padding == -1 {
            1
        } else {
            2 * padding as usize
        };
        let extent = |size: usize| {
            (size * upscale + total_padding)
                .checked_sub(kernel)
                .map(|n| n / stride + 1)
        };
        let (Some(out_height), Some(out_width)) = (extent(height), extent(width)) else {
            return Err(logic(&format!("convolution shape: {name}")));
        };
        let mut out = self.new_matrix(out_height * out_width, spec[0])?;
        let w = self.weight(weights, &weight_name)?;
        let bias_name = format!("{name}.bias");
        let bias = if weights.contains(&bias_name) {
            Some(self.weight(weights, &bias_name)?)
        } else {
            None
        };
        if w.packed {
            let output_channels = out.cols;
            let launch = if kernel == 3 {
                k::convolve3(
                    &self.stream,
                    x.bf16()?,
                    w.bf16()?,
                    bias.as_ref().map(Matrix::bf16).transpose()?,
                    out.bf16_mut()?,
                    dim(height)?,
                    dim(width)?,
                    dim(x.cols)?,
                    dim(output_channels)?,
                    dim(stride)?,
                    dim(total_padding)?,
                    dim(upscale)?,
                )
                .map_err(launch_error)?
            } else {
                Launch::Unsupported
            };
            if launch == Launch::Unsupported {
                return Err(logic("unsupported Qwen-Image packed convolution"));
            }
            return Ok(out);
        }
        const TILE: usize = 1024;
        let mut col = self.new_matrix(TILE.min(out.rows), w.cols)?;
        let pad = padding.max(0) as usize;
        let mut start = 0;
        while start < out.rows {
            let count = TILE.min(out.rows - start);
            let input_len = count * w.cols;
            let expanded = kernel == 3
                && k::expand_convolution3(
                    &self.stream,
                    x.bf16()?,
                    col.bf16_mut()?.range(0..input_len),
                    dim(height)?,
                    dim(width)?,
                    dim(x.cols)?,
                    dim(stride)?,
                    dim(pad)?,
                    dim(upscale)?,
                    dim(out_width)?,
                    dim(start)?,
                    dim(count)?,
                )
                .map_err(launch_error)?
                    == Launch::Launched;
            if !expanded {
                k::im2col(
                    &self.stream,
                    x.bf16()?,
                    col.bf16_mut()?.range(0..input_len),
                    dim(height)?,
                    dim(width)?,
                    dim(x.cols)?,
                    dim(kernel)?,
                    dim(stride)?,
                    dim(pad)?,
                    dim(upscale)?,
                    dim(out_width)?,
                    dim(start)?,
                    dim(count)?,
                )
                .map_err(launch_error)?;
            }
            let input = col.slice(0, count)?;
            let channels = out.cols;
            let output = out
                .bf16_mut()?
                .range(start * channels..(start + count) * channels);
            self.matmul(&input, &w, output, bias.as_ref(), true)?;
            drop(input);
            start += TILE;
        }
        Ok(out)
    }

    /// `Runtime::DownShortcut`.
    pub fn down_shortcut(
        &mut self,
        x: &Matrix,
        height: usize,
        width: usize,
        out_channels: usize,
        temporal: usize,
        spatial: usize,
    ) -> Result<Matrix> {
        let mut out = self.new_matrix((height / spatial) * (width / spatial), out_channels)?;
        k::down_shortcut(
            &self.stream,
            x.bf16()?,
            out.bf16_mut()?,
            dim(height)?,
            dim(width)?,
            dim(x.cols)?,
            dim(out_channels)?,
            dim(temporal)?,
            dim(spatial)?,
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// `Runtime::UpShortcut`.
    pub fn up_shortcut(
        &mut self,
        x: &Matrix,
        height: usize,
        width: usize,
        out_channels: usize,
        temporal: usize,
    ) -> Result<Matrix> {
        let mut out = self.new_matrix(height * width * 4, out_channels)?;
        k::up_shortcut(
            &self.stream,
            x.bf16()?,
            out.bf16_mut()?,
            dim(height)?,
            dim(width)?,
            dim(x.cols)?,
            dim(out_channels)?,
            dim(temporal)?,
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// `Runtime::Euler`.
    pub fn euler(&mut self, x: &Matrix, velocity: &Matrix, dt: f32) -> Result<Matrix> {
        let mut out = self.new_matrix(x.rows, x.cols)?;
        let size = out.size();
        k::euler(
            &self.stream,
            x.bf16()?,
            velocity.bf16()?,
            out.bf16_mut()?,
            size,
            dt,
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// `Runtime::LatentScale`.
    pub fn latent_scale(
        &mut self,
        x: &Matrix,
        mean: &[f32],
        deviation: &[f32],
        encode: bool,
    ) -> Result<Matrix> {
        let m = self.upload(mean, 1, 64)?;
        let s = self.upload(deviation, 1, 64)?;
        let mut out = self.new_matrix(x.rows, x.cols)?;
        let size = out.size();
        k::scale_latents(
            &self.stream,
            x.bf16()?,
            out.bf16_mut()?,
            m.bf16()?,
            s.bf16()?,
            size,
            encode,
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// `Runtime::VisionPosition`.
    pub fn vision_position(
        &mut self,
        weights: &Weights,
        x: &Matrix,
        height: usize,
        width: usize,
    ) -> Result<Matrix> {
        let w = self.weight(weights, "text_encoder.model.visual.pos_embed.weight")?;
        if x.cols != 1152 || x.rows != height * width || w.rows != 2304 || w.cols != 1152 {
            return Err(logic("Qwen-Image vision position shape mismatch"));
        }
        let mut out = self.new_matrix(x.rows, x.cols)?;
        k::vision_position(
            &self.stream,
            x.bf16()?,
            w.bf16()?,
            out.bf16_mut()?,
            dim(height)?,
            dim(width)?,
        )
        .map_err(launch_error)?;
        Ok(out)
    }

    /// `Runtime::Download`: the matrix as FP32.
    pub fn download(&mut self, matrix: &Matrix) -> Result<Vec<f32>> {
        let mut tmp = self.raw(matrix.size() * 4)?;
        k::bf16_to_float(&self.stream, matrix.bf16()?, tmp.f32_mut()?, matrix.size())
            .map_err(launch_error)?;
        let mut result = vec![0.0f32; matrix.size()];
        self.stream
            .download(tmp.f32()?, &mut result)
            .map_err(hip_error)?;
        Ok(result)
    }

    /// `Runtime::Observe`.
    pub fn observe(
        &mut self,
        observer: Option<&mut dyn Observer>,
        name: &str,
        matrix: &Matrix,
    ) -> Result<()> {
        if let Some(observer) = observer
            && observer.wants(name)
        {
            let values = self.download(matrix)?;
            observer.write(name, &values, matrix.rows, matrix.cols);
        }
        Ok(())
    }
}
