//! Safetensors shards: an 8-byte little-endian header length, a JSON header
//! naming every tensor's dtype, shape and byte range, then the payload.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use gufo_hip::MappedFile;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    Bf16,
    F16,
    I64,
    I32,
    U8,
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

    pub fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::Bf16 | Self::F16 => 2,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }
}

/// One tensor: its type, shape and absolute byte range in shard `file`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub file: usize,
    pub offset: usize,
    pub bytes: usize,
}

impl TensorInfo {
    pub fn elements(&self) -> usize {
        self.bytes / self.dtype.size()
    }
}

/// Parses a shard header. Entries with an unknown dtype or an inconsistent
/// size are skipped, as the C++ loader does; a malformed header or a byte
/// range outside the file is an error.
pub fn parse_header(bytes: &[u8], file: usize) -> Result<Vec<(String, TensorInfo)>, String> {
    let length = bytes
        .get(..8)
        .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .ok_or("safetensors file is shorter than its header length")?;
    let payload = usize::try_from(length)
        .ok()
        .and_then(|l| l.checked_add(8))
        .filter(|&end| length > 0 && end <= bytes.len())
        .ok_or("safetensors header length is invalid")?;
    let header: Value = serde_json::from_slice(&bytes[8..payload])
        .map_err(|e| format!("safetensors header is not JSON: {e}"))?;
    let header = header
        .as_object()
        .ok_or("safetensors header is not an object")?;
    let available = bytes.len() - payload;
    let mut tensors = Vec::with_capacity(header.len());
    for (name, spec) in header {
        if name == "__metadata__" {
            continue;
        }
        let Some(spec) = spec.as_object() else {
            continue;
        };
        let Some(dtype) = spec
            .get("dtype")
            .and_then(Value::as_str)
            .and_then(DType::parse)
        else {
            continue;
        };
        let Some(shape) = shape_of(spec) else {
            continue;
        };
        let offsets = spec
            .get("data_offsets")
            .and_then(Value::as_array)
            .and_then(|o| match o.as_slice() {
                [begin, end] => Some((begin.as_u64()?, end.as_u64()?)),
                _ => None,
            });
        let Some((Ok(begin), Ok(end))) =
            offsets.map(|(b, e)| (usize::try_from(b), usize::try_from(e)))
        else {
            continue;
        };
        let elements = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(usize::try_from(d).ok()?));
        let expected = elements.and_then(|e| e.checked_mul(dtype.size()));
        if begin >= end || end > available || expected != Some(end - begin) {
            continue;
        }
        tensors.push((
            name.clone(),
            TensorInfo {
                dtype,
                shape,
                file,
                offset: payload + begin,
                bytes: end - begin,
            },
        ));
    }
    Ok(tensors)
}

fn shape_of(spec: &serde_json::Map<String, Value>) -> Option<Vec<u64>> {
    match spec.get("shape") {
        None => Some(Vec::new()),
        Some(dims) => dims.as_array()?.iter().map(Value::as_u64).collect(),
    }
}

/// The mapped shards of a checkpoint and every tensor in them.
pub struct Checkpoint {
    files: Vec<Arc<MappedFile>>,
    tensors: HashMap<String, TensorInfo>,
}

impl Checkpoint {
    pub fn open(model_dir: &Path, shards: &[&str]) -> Result<Self, String> {
        let mut files = Vec::with_capacity(shards.len());
        let mut tensors = HashMap::new();
        for (index, shard) in shards.iter().enumerate() {
            let path = model_dir.join(shard);
            let file = MappedFile::open(&path)
                .map_err(|e| format!("cannot map {}: {e}", path.display()))?;
            let entries = parse_header(file.bytes(), index)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            if entries.is_empty() {
                return Err(format!("{} holds no tensors", path.display()));
            }
            for (name, info) in entries {
                if tensors.insert(name.clone(), info).is_some() {
                    return Err(format!("tensor {name} appears twice"));
                }
            }
            files.push(file);
        }
        Ok(Self { files, tensors })
    }

    pub fn file(&self, index: usize) -> &Arc<MappedFile> {
        &self.files[index]
    }

    pub fn files(&self) -> &[Arc<MappedFile>] {
        &self.files
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &TensorInfo)> {
        self.tensors
            .iter()
            .map(|(name, info)| (name.as_str(), info))
    }

    /// A BF16 tensor of exactly `shape`.
    pub fn bf16(&self, name: &str, shape: &[u64]) -> Result<&TensorInfo, String> {
        self.get(name)
            .filter(|t| t.dtype == DType::Bf16 && t.shape == shape)
            .ok_or_else(|| format!("unexpected Qwen3-ASR tensor: {name}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shard(header: &str, payload: usize) -> Vec<u8> {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + payload, 0);
        bytes
    }

    #[test]
    fn tensors_get_absolute_ranges() {
        let header = r#"{"__metadata__":{"format":"pt"},
            "a":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},
            "b":{"dtype":"F32","shape":[],"data_offsets":[12,16]}}"#;
        let bytes = shard(header, 16);
        let mut tensors = parse_header(&bytes, 1).unwrap();
        tensors.sort_by(|x, y| x.0.cmp(&y.0));
        let payload = 8 + header.len();
        assert_eq!(tensors[0].0, "a");
        assert_eq!(
            tensors[0].1,
            TensorInfo {
                dtype: DType::Bf16,
                shape: vec![2, 3],
                file: 1,
                offset: payload,
                bytes: 12
            }
        );
        assert_eq!(tensors[0].1.elements(), 6);
        assert_eq!((tensors[1].1.offset, tensors[1].1.bytes), (payload + 12, 4));
    }

    #[test]
    fn inconsistent_entries_are_skipped_and_broken_headers_rejected() {
        let header = r#"{"size":{"dtype":"BF16","shape":[4],"data_offsets":[0,6]},
            "type":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[0,4]},
            "outside":{"dtype":"U8","shape":[64],"data_offsets":[0,64]},
            "ok":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#;
        let tensors = parse_header(&shard(header, 8), 0).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].0, "ok");
        assert!(parse_header(&[1, 0], 0).is_err());
        assert!(parse_header(&shard("{", 0), 0).is_err());
        let mut long = shard("{}", 0);
        long[0] = 200;
        assert!(parse_header(&long, 0).is_err());
    }
}
