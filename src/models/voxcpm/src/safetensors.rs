//! A read-only mapping of a safetensors file (`safetensors.hpp`,
//! `safetensors.cpp`), the OmniVoice port's reader with the GGML Q8_0 type
//! that tensors of audio.cpp's GGUFs may carry. A [`Tensor`] keeps its file's
//! mapping alive and reads its payload from it.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use gufo_core::json::{self, Value};
use gufo_hip::MappedFile;

/// Element types. `Q8_0` (GGML: blocks of 32 int8 values with one float16
/// scale, 34 bytes) only comes from GGUF files and has no element size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    Bf16,
    F16,
    I64,
    I32,
    U8,
    Q8_0,
}

impl DType {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "F32" => Self::F32,
            "BF16" => Self::Bf16,
            "F16" => Self::F16,
            "I64" => Self::I64,
            "I32" => Self::I32,
            "U8" => Self::U8,
            _ => return None,
        })
    }

    /// Bytes per element, 0 for `Q8_0` (`DTypeSize`).
    pub fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::Bf16 | Self::F16 => 2,
            Self::I64 => 8,
            Self::U8 => 1,
            Self::Q8_0 => 0,
        }
    }

    /// `DTypeName`.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Bf16 => "BF16",
            Self::F16 => "F16",
            Self::I64 => "I64",
            Self::I32 => "I32",
            Self::U8 => "U8",
            Self::Q8_0 => "Q8_0",
        }
    }
}

/// One tensor of a mapped file (safetensors, a torch checkpoint's storage or
/// a GGUF), in PyTorch's dimension order. `offset` is its byte position in
/// the file; [`Tensor::data`] stays valid while the tensor lives.
#[derive(Clone)]
pub struct Tensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bytes: usize,
    pub offset: u64,
    mapping: Arc<MappedFile>,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("name", &self.name)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("bytes", &self.bytes)
            .field("offset", &self.offset)
            .finish()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.dtype == other.dtype
            && self.shape == other.shape
            && self.bytes == other.bytes
            && self.offset == other.offset
            && Arc::ptr_eq(&self.mapping, &other.mapping)
    }
}

impl Tensor {
    pub(crate) fn new(
        name: String,
        dtype: DType,
        shape: Vec<usize>,
        bytes: usize,
        offset: u64,
        mapping: Arc<MappedFile>,
    ) -> Self {
        Self {
            name,
            dtype,
            shape,
            bytes,
            offset,
            mapping,
        }
    }

    /// The element count (`Elements`).
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// The payload bytes.
    pub fn data(&self) -> &[u8] {
        let begin = self.offset as usize;
        &self.mapping.bytes()[begin..begin + self.bytes]
    }

    /// The mapped file that holds the payload.
    pub fn mapping(&self) -> &Arc<MappedFile> {
        &self.mapping
    }

    /// A copy of an F32 tensor's values (little-endian payload).
    pub fn f32_values(&self) -> Result<Vec<f32>, String> {
        if self.dtype != DType::F32 {
            return Err(format!(
                "{} is {}, expected F32",
                self.name,
                self.dtype.name()
            ));
        }
        Ok(self
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect())
    }
}

pub(crate) fn shape_text(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(ToString::to_string).collect();
    format!("[{}]", parts.join(", "))
}

/// A header entry before its mapping is attached.
struct Entry {
    name: String,
    dtype: DType,
    shape: Vec<usize>,
    bytes: usize,
    offset: u64,
}

fn parse_entry(
    name: &str,
    entry: &Value,
    payload_offset: u64,
    payload_bytes: u64,
) -> Result<Entry, String> {
    if !entry.is_object() {
        return Err(format!("{name}: header entry is not an object"));
    }
    let dtype_name = entry.member_str("dtype", "");
    let dtype = DType::parse(&dtype_name)
        .ok_or_else(|| format!("{name}: unsupported dtype \"{dtype_name}\""))?;
    let shape_value = entry
        .find("shape")
        .filter(|shape| shape.is_array())
        .ok_or_else(|| format!("{name}: missing shape"))?;
    let mut elements: usize = 1;
    let mut shape = Vec::new();
    for dimension in shape_value.items() {
        let size = dimension.as_size(usize::MAX);
        if size == usize::MAX {
            return Err(format!("{name}: malformed shape"));
        }
        if size != 0 && elements > usize::MAX / size {
            return Err(format!("{name}: shape overflows"));
        }
        elements *= size;
        shape.push(size);
    }
    let offsets = entry
        .find("data_offsets")
        .filter(|offsets| offsets.is_array() && offsets.len() == 2)
        .ok_or_else(|| format!("{name}: missing data_offsets"))?;
    let begin = offsets.items()[0].as_size(usize::MAX) as u64;
    let end = offsets.items()[1].as_size(usize::MAX) as u64;
    if begin == u64::MAX || end == u64::MAX || begin > end || end > payload_bytes {
        return Err(format!("{name}: data_offsets lie outside the payload"));
    }
    let element_bytes = dtype.size() as u64;
    if elements as u64 > u64::MAX / element_bytes || elements as u64 * element_bytes != end - begin
    {
        return Err(format!(
            "{name}: {} bytes don't hold shape {}",
            end - begin,
            shape_text(&shape)
        ));
    }
    Ok(Entry {
        name: name.to_owned(),
        dtype,
        shape,
        bytes: (end - begin) as usize,
        offset: payload_offset + begin,
    })
}

fn parse_entries(bytes: &[u8]) -> Result<Vec<Entry>, String> {
    if bytes.len() < 8 {
        return Err("file is shorter than its header length".into());
    }
    let mut length = [0u8; 8];
    length.copy_from_slice(&bytes[..8]);
    let header_bytes = u64::from_le_bytes(length);
    // The format caps headers at 100 MB.
    if header_bytes == 0 || header_bytes > 100_000_000 || header_bytes > (bytes.len() - 8) as u64 {
        return Err("implausible header length".into());
    }
    let header_end = 8 + header_bytes as usize;
    let header = json::parse(&bytes[8..header_end]).map_err(|error| format!("header: {error}"))?;
    if !header.is_object() {
        return Err("header is not a JSON object".into());
    }
    let payload_offset = header_end as u64;
    let payload_bytes = bytes.len() as u64 - payload_offset;
    let mut entries = Vec::new();
    for (name, entry) in header.members() {
        if name == "__metadata__" {
            continue;
        }
        entries.push(parse_entry(name, entry, payload_offset, payload_bytes)?);
    }
    if entries.is_empty() {
        return Err("no tensors".into());
    }
    let mut by_offset: Vec<&Entry> = entries.iter().collect();
    by_offset.sort_by_key(|entry| entry.offset);
    for pair in by_offset.windows(2) {
        if pair[0].offset + pair[0].bytes as u64 > pair[1].offset {
            return Err(format!("{} overlaps {}", pair[0].name, pair[1].name));
        }
    }
    entries.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(entries)
}

/// A read-only mapping of a safetensors file. Every header entry must be
/// well formed and lie inside the payload; tensors may not overlap.
pub struct SafetensorsFile {
    path: PathBuf,
    mapping: Arc<MappedFile>,
    /// Sorted by name.
    tensors: Vec<Tensor>,
}

impl SafetensorsFile {
    /// Maps and validates `path` (`Open`).
    pub fn open(path: &Path) -> Result<Self, String> {
        let mapping = MappedFile::open(path)
            .map_err(|error| format!("cannot map {}: {error}", path.display()))?;
        let entries = parse_entries(mapping.bytes())
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let tensors = entries
            .into_iter()
            .map(|entry| {
                Tensor::new(
                    entry.name,
                    entry.dtype,
                    entry.shape,
                    entry.bytes,
                    entry.offset,
                    mapping.clone(),
                )
            })
            .collect();
        Ok(Self {
            path: path.to_owned(),
            mapping,
            tensors,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    pub fn mapping(&self) -> &Arc<MappedFile> {
        &self.mapping
    }

    /// `Find`.
    pub fn find(&self, name: &str) -> Option<&Tensor> {
        self.tensors
            .binary_search_by(|tensor| tensor.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.tensors[index])
    }

    /// The tensor `name` with exactly this type and shape, or the mismatch
    /// (`Expect`).
    pub fn expect(&self, name: &str, dtype: DType, shape: &[usize]) -> Result<&Tensor, String> {
        let Some(tensor) = self.find(name) else {
            let file = self
                .path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            return Err(format!("{file} lacks {name}"));
        };
        if tensor.dtype != dtype || tensor.shape != shape {
            return Err(format!(
                "{name} is {} {}, expected {} {}",
                tensor.dtype.name(),
                shape_text(&tensor.shape),
                dtype.name(),
                shape_text(shape)
            ));
        }
        Ok(tensor)
    }

    /// Drops resident pages once the weights live on the device; tensors
    /// stay readable and fault their pages back in (`ReleasePages`).
    pub fn release_pages(&self) {
        self.mapping.release_pages();
    }
}
