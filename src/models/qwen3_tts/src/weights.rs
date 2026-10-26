//! Device copies of the checkpoint (`DeviceRegion` and `UploadNorm` of the C++
//! runtimes). A component's tensors are copied as one page-aligned file range
//! into one allocation, shifted so that the safetensors payload starts on a
//! 16-byte boundary: every tensor keeps its payload alignment, which decides
//! the load width (and so the reduction order) of the BF16 GEMV kernels.

use std::marker::PhantomData;

use gufo_hip::{Bf16, DeviceBuffer, DeviceCopy, DeviceSlice, DeviceTensor, Stream};

use crate::bf16::bf16_to_f32;
use crate::checkpoint::{Checkpoint, DType, Region, Tensor};

/// One tensor of a [`DeviceRegion`], checked for type, shape and alignment.
pub struct View<T> {
    offset: usize,
    len: usize,
    _element: PhantomData<T>,
}

impl<T> Clone for View<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for View<T> {}

impl<T> View<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub struct DeviceRegion {
    buffer: DeviceTensor<u8>,
    file: usize,
    begin: usize,
    size: usize,
    shift: usize,
}

impl DeviceRegion {
    /// Copies `region` (`TryCopy`): prefault the mapping, allocate
    /// `size + shift` bytes, copy the range `shift` bytes in.
    pub fn upload(
        stream: &Stream,
        checkpoint: &Checkpoint,
        region: Region,
    ) -> Result<Self, String> {
        const LOAD_ALIGNMENT: usize = 16;
        let shift = (LOAD_ALIGNMENT - region.payload_alignment % LOAD_ALIGNMENT) % LOAD_ALIGNMENT;
        let file = checkpoint.file(region.file);
        let range = region.begin..region.begin + region.size;
        // Faulting the pages in first is only a speed-up, as in C++.
        let _ = file.prefault(range.clone());
        let mut buffer = stream
            .alloc_persistent::<u8>(region.size + shift)
            .map_err(|e| format!("cannot make Qwen3-TTS safetensors GPU-visible: {e}"))?;
        stream
            .upload(
                &file.bytes()[range],
                buffer.slice_mut(shift..shift + region.size),
            )
            .map_err(|e| format!("cannot make Qwen3-TTS safetensors GPU-visible: {e}"))?;
        let buffer = buffer
            .freeze(stream)
            .map_err(|e| format!("cannot make Qwen3-TTS safetensors GPU-visible: {e}"))?;
        Ok(Self {
            buffer,
            file: region.file,
            begin: region.begin,
            size: region.size,
            shift,
        })
    }

    /// `Resolve`: the device view of `tensor`, which must lie inside the
    /// region and be aligned for `T`.
    pub fn view<T: DeviceCopy>(&self, tensor: &Tensor) -> Option<View<T>> {
        if tensor.file != self.file
            || tensor.offset < self.begin
            || tensor.offset - self.begin >= self.size
            || tensor.bytes > self.size - (tensor.offset - self.begin)
            || !tensor.bytes.is_multiple_of(size_of::<T>())
        {
            return None;
        }
        let offset = self.shift + tensor.offset - self.begin;
        self.buffer
            .slice(offset..offset + tensor.bytes)
            .cast::<T>()
            .map(|_| View {
                offset,
                len: tensor.bytes / size_of::<T>(),
                _element: PhantomData,
            })
    }

    pub fn bytes<T: DeviceCopy>(&self, view: View<T>) -> DeviceSlice<'_, u8> {
        self.buffer
            .slice(view.offset..view.offset + view.len * size_of::<T>())
    }

    /// The elements of `view` (an error only for a view of another region).
    pub fn slice<T: DeviceCopy>(&self, view: View<T>) -> Result<DeviceSlice<'_, T>, String> {
        if view.offset + view.len * size_of::<T>() > self.buffer.len() {
            return Err("Qwen3-TTS weight view outside its region".into());
        }
        self.bytes(view)
            .cast::<T>()
            .ok_or_else(|| "Qwen3-TTS weight view is misaligned".into())
    }
}

/// `RequireTensor`: a BF16 tensor of exactly `shape`.
pub fn require_bf16<'a>(
    checkpoint: &'a Checkpoint,
    name: &str,
    shape: &[u64],
) -> Result<&'a Tensor, String> {
    require(checkpoint, name, shape, DType::Bf16, "Qwen3-TTS tensor")
}

/// `RequireF32Tensor` of the speech decoder.
pub fn require_f32<'a>(
    checkpoint: &'a Checkpoint,
    name: &str,
    shape: &[u64],
) -> Result<&'a Tensor, String> {
    require(
        checkpoint,
        name,
        shape,
        DType::F32,
        "Qwen3-TTS speech tensor",
    )
}

fn require<'a>(
    checkpoint: &'a Checkpoint,
    name: &str,
    shape: &[u64],
    dtype: DType,
    what: &str,
) -> Result<&'a Tensor, String> {
    let tensor = checkpoint
        .find(name)
        .ok_or_else(|| format!("missing {what}: {name}"))?;
    if tensor.dtype != dtype || tensor.shape != shape {
        return Err(format!("unexpected {what} layout: {name}"));
    }
    Ok(tensor)
}

/// The BF16 payload of `tensor` widened to f32 (`DecodeBfloat16`).
pub fn decode_bf16(checkpoint: &Checkpoint, tensor: &Tensor) -> Vec<f32> {
    checkpoint
        .bytes(tensor)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
        .collect()
}

/// `UploadNorm`: a BF16 vector of `size` values as its own f32 allocation.
pub fn upload_norm(
    stream: &Stream,
    checkpoint: &Checkpoint,
    name: &str,
    size: usize,
) -> Result<DeviceBuffer<f32>, String> {
    let tensor = require_bf16(checkpoint, name, &[size as u64])?;
    let host = decode_bf16(checkpoint, tensor);
    let mut buffer = stream
        .alloc_persistent::<f32>(host.len())
        .map_err(|e| format!("hipMalloc: {e}"))?;
    stream
        .upload(&host, buffer.as_mut_slice())
        .map_err(|e| format!("hipMemcpy norm: {e}"))?;
    Ok(buffer)
}

/// A device view of a BF16 tensor the region holds (`ResolveMatrix`,
/// `ResolveVector`).
pub fn resolve_bf16(
    checkpoint: &Checkpoint,
    region: &DeviceRegion,
    name: &str,
    shape: &[u64],
) -> Result<View<Bf16>, String> {
    let tensor = require_bf16(checkpoint, name, shape)?;
    region
        .view::<Bf16>(tensor)
        .ok_or_else(|| format!("Qwen3-TTS tensor is outside the mapped weights: {name}"))
}

/// A device view of an f32 tensor the region holds (`ResolveF32`).
pub fn resolve_f32(
    checkpoint: &Checkpoint,
    region: &DeviceRegion,
    name: &str,
    shape: &[u64],
) -> Result<View<f32>, String> {
    let tensor = require_f32(checkpoint, name, shape)?;
    region
        .view::<f32>(tensor)
        .ok_or_else(|| format!("Qwen3-TTS speech tensor is outside mapped file: {name}"))
}

/// Zeroed indices below `bound` in an exactly sized allocation (`hipMalloc`,
/// as the C++ buffers), not in the stream-ordered pool, which reserves device
/// memory in 128 MiB blocks.
pub fn persistent_indices(
    stream: &Stream,
    len: usize,
    bound: u32,
) -> Result<gufo_hip::DeviceIndices, gufo_hip::HipError> {
    stream.zeroed_indices(stream.alloc_persistent::<u32>(len)?, bound)
}
