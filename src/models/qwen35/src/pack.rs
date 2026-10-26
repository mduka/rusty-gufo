//! Host packing of GGUF tensors into device tensors, shared by the DFlash2
//! draft and the MTP head: `CopyVectorF32` and `PackMatrixBf16` of their C++
//! models (`hip/dflash/model.cpp`, `hip/mtp/model.cpp`). Each tensor gets its
//! own exactly sized allocation, as in C++.

use gufo_core::gguf::Tensor;
use gufo_formats::dequant::dequantize;
use gufo_hip::{Bf16, DeviceCopy, DeviceTensor, GgmlType, MappedFile, QuantTensor, Stream};

use crate::arena::alloc;

/// Host rows per BF16 packing chunk (`kPackChunkBytes`).
const PACK_CHUNK_BYTES: usize = 8 << 20;

pub(crate) struct Packer<'a> {
    pub(crate) stream: &'a Stream,
    pub(crate) file: &'a MappedFile,
    /// The model errors name, as "DFlash-2" or "Qwen MTP".
    pub(crate) model: &'static str,
    /// Bytes packed so far.
    pub(crate) bytes: usize,
}

impl Packer<'_> {
    pub(crate) fn payload(&self, tensor: &Tensor<'_>) -> &[u8] {
        &self.file.bytes()[tensor.offset..tensor.offset + tensor.bytes]
    }

    pub(crate) fn upload<T: DeviceCopy>(
        &mut self,
        values: &[T],
    ) -> Result<DeviceTensor<T>, String> {
        let model = self.model;
        let error = |e| format!("{model} weights: {e}");
        let mut buffer = alloc::<T>(self.stream, values.len())?;
        self.stream
            .upload(values, buffer.as_mut_slice())
            .map_err(error)?;
        self.bytes += size_of_val(values);
        buffer.freeze(self.stream).map_err(error)
    }

    /// An F32 copy of an F32 or BF16 vector (`CopyVectorF32`).
    pub(crate) fn vector(&mut self, tensor: &Tensor<'_>) -> Result<DeviceTensor<f32>, String> {
        let payload = self.payload(tensor);
        let values: Vec<f32> = match tensor.ty {
            GgmlType::F32 => payload
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect(),
            _ => payload
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| Bf16(u16::from_le_bytes(b)).to_f32())
                .collect(),
        };
        self.upload(&values)
    }

    /// BF16 rows (`PackMatrixBf16`): copied if BF16, else dequantised in
    /// chunks of rows and rounded to nearest even.
    pub(crate) fn bf16(
        &mut self,
        tensor: &Tensor<'_>,
        rows: usize,
        cols: usize,
    ) -> Result<DeviceTensor<Bf16>, String> {
        let (stream, model) = (self.stream, self.model);
        let error = |e| format!("{model} weights: {e}");
        let mut buffer = alloc::<Bf16>(stream, rows * cols)?;
        self.bf16_rows(tensor, rows, cols, |start, halves| {
            stream
                .upload(
                    halves,
                    buffer.slice_mut(start * cols..start * cols + halves.len()),
                )
                .map_err(error)
        })?;
        buffer.freeze(stream).map_err(error)
    }

    /// [`Packer::bf16`] as a BF16 block tensor for the GEMV kernels.
    pub(crate) fn bf16_matrix(
        &mut self,
        tensor: &Tensor<'_>,
        rows: usize,
        cols: usize,
    ) -> Result<QuantTensor, String> {
        let (stream, model) = (self.stream, self.model);
        let error = |e| format!("{model} weights: {e}");
        let mut buffer = alloc::<u8>(stream, rows * cols * 2)?;
        self.bf16_rows(tensor, rows, cols, |start, halves| {
            let bytes: Vec<u8> = halves.iter().flat_map(|h| h.0.to_le_bytes()).collect();
            let at = start * cols * 2;
            stream
                .upload(&bytes, buffer.slice_mut(at..at + bytes.len()))
                .map_err(error)
        })?;
        Ok(QuantTensor::new(
            buffer.freeze(stream).map_err(error)?,
            GgmlType::Bf16,
        ))
    }

    /// Hands the BF16 rows of `tensor` to `emit` in chunks, with the first
    /// row of each, and counts them as packed.
    fn bf16_rows(
        &mut self,
        tensor: &Tensor<'_>,
        rows: usize,
        cols: usize,
        mut emit: impl FnMut(usize, &[Bf16]) -> Result<(), String>,
    ) -> Result<(), String> {
        let model = self.model;
        let payload = self.payload(tensor);
        if tensor.ty == GgmlType::Bf16 {
            let values: Vec<Bf16> = payload
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| Bf16(u16::from_le_bytes(b)))
                .collect();
            emit(0, &values)?;
        } else {
            let row_bytes = tensor.bytes / rows;
            let chunk_rows = rows.min((PACK_CHUNK_BYTES / (cols * 2)).max(1));
            let mut floats = vec![0f32; chunk_rows * cols];
            let mut halves = vec![Bf16::default(); chunk_rows * cols];
            for start in (0..rows).step_by(chunk_rows) {
                let count = chunk_rows.min(rows - start);
                for row in 0..count {
                    let source = &payload[(start + row) * row_bytes..(start + row + 1) * row_bytes];
                    let out = &mut floats[row * cols..(row + 1) * cols];
                    if tensor.ty == GgmlType::F32 {
                        for (value, bytes) in out.iter_mut().zip(source.as_chunks::<4>().0) {
                            *value = f32::from_le_bytes(*bytes);
                        }
                    } else {
                        dequantize(tensor.ty, source, out).ok_or_else(|| {
                            format!("unsupported {model} tensor format for {}", tensor.name)
                        })?;
                    }
                }
                for (half, &value) in halves.iter_mut().zip(&floats[..count * cols]) {
                    *half = Bf16::from_f32(value);
                }
                emit(start, &halves[..count * cols])?;
            }
        }
        self.bytes += rows * cols * 2;
        Ok(())
    }
}
