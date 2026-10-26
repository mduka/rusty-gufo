//! The checkpoint directory (`loader.cpp`, `tensor.cpp`): `config.json`,
//! `model.safetensors` and `speech_tokenizer/model.safetensors`, mapped
//! read-only. Tensors the C++ loader would skip (unknown type, empty or
//! inconsistent extent) are skipped alike.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use gufo_core::json;
use gufo_hip::MappedFile;

use crate::config::ModelConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    Bf16,
    F16,
    I64,
    I32,
    U8,
    Unknown,
}

impl DType {
    fn parse(name: &str) -> Self {
        match name {
            "F32" | "f32" => Self::F32,
            "BF16" | "bf16" => Self::Bf16,
            "F16" | "f16" => Self::F16,
            "I64" | "i64" => Self::I64,
            "I32" | "i32" => Self::I32,
            "U8" | "u8" => Self::U8,
            _ => Self::Unknown,
        }
    }

    /// `DTypeSize`.
    pub fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::Bf16 | Self::F16 => 2,
            Self::I64 => 8,
            Self::U8 => 1,
            Self::Unknown => 0,
        }
    }
}

/// One tensor of a mapped file.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    /// The mapped file holding the payload.
    pub file: usize,
    /// Byte offset of the payload in that file.
    pub offset: usize,
    pub bytes: usize,
}

impl Tensor {
    /// `Tensor::NumElements`; `None` on overflow.
    pub fn elements(&self) -> Option<usize> {
        elements(&self.shape)
    }
}

fn elements(shape: &[u64]) -> Option<usize> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        usize::try_from(dimension)
            .ok()
            .and_then(|d| count.checked_mul(d))
    })
}

/// The page-aligned envelope of one component's tensors in one file
/// (`MappedRegion` of `RegionsFor`). Copying it to `16 - payload_alignment`
/// bytes past a 16-byte boundary keeps every tensor at its payload alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub file: usize,
    pub begin: usize,
    pub size: usize,
    /// The file's payload offset modulo 16.
    pub payload_alignment: usize,
}

struct MappedSafetensors {
    file: Arc<MappedFile>,
    payload_offset: usize,
}

/// `LoadResult`: the configuration and every tensor of both files.
pub struct Checkpoint {
    pub config: ModelConfig,
    files: Vec<MappedSafetensors>,
    tensors: Vec<Tensor>,
    by_name: HashMap<String, usize>,
}

impl Checkpoint {
    /// `LoadModelDirectory`, with its error messages.
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let config = ModelConfig::load(model_dir).ok_or_else(|| {
            format!(
                "qwen3_tts: cannot parse {}/config.json",
                model_dir.display()
            )
        })?;
        if !config.is_supported() {
            return Err("qwen3_tts: expected a supported 12Hz 1.7B CustomVoice, \
                        VoiceDesign, or Base model configuration"
                .into());
        }
        let mut checkpoint = Self {
            config,
            files: Vec::new(),
            tensors: Vec::new(),
            by_name: HashMap::new(),
        };
        let main = model_dir.join("model.safetensors");
        if !checkpoint.add_file(&main) {
            return Err(format!("qwen3_tts: cannot load {}", main.display()));
        }
        let speech = model_dir.join("speech_tokenizer").join("model.safetensors");
        if !checkpoint.add_file(&speech) {
            return Err(format!("qwen3_tts: cannot load {}", speech.display()));
        }
        Ok(checkpoint)
    }

    /// `LoadSafetensorsFile`: false when the file is unreadable or holds no
    /// usable tensor.
    fn add_file(&mut self, path: &Path) -> bool {
        let Ok(file) = MappedFile::open(path) else {
            return false;
        };
        let bytes = file.bytes();
        if bytes.len() < 8 {
            return false;
        }
        let header_bytes = u64::from_le_bytes(bytes[..8].try_into().unwrap_or_default());
        if header_bytes == 0 || header_bytes > (bytes.len() - 8) as u64 {
            return false;
        }
        let header_end = 8 + header_bytes as usize;
        let Ok(text) = std::str::from_utf8(&bytes[8..header_end]) else {
            return false;
        };
        let Ok(header) = json::parse_str(text) else {
            return false;
        };
        if !header.is_object() {
            return false;
        }
        let payload_offset = header_end;
        let index = self.files.len();
        let before = self.tensors.len();
        for (name, value) in header.members() {
            if name == "__metadata__" || !value.is_object() {
                continue;
            }
            let dtype = DType::parse(&value.member_str("dtype", ""));
            let shape: Vec<u64> = value
                .find("shape")
                .filter(|shape| shape.is_array())
                .map(|shape| shape.items().iter().map(|d| d.as_size(0) as u64).collect())
                .unwrap_or_default();
            let Some(offsets) = value.find("data_offsets") else {
                continue;
            };
            if !offsets.is_array() || offsets.len() != 2 {
                continue;
            }
            let begin = offsets.items()[0].as_size(0);
            let end = offsets.items()[1].as_size(0);
            let Some(count) = elements(&shape) else {
                continue;
            };
            let element_bytes = dtype.size();
            if element_bytes == 0 || begin >= end || end > bytes.len() - payload_offset {
                continue;
            }
            if count.checked_mul(element_bytes) != Some(end - begin) {
                continue;
            }
            self.by_name
                .entry(name.clone())
                .or_insert(self.tensors.len());
            self.tensors.push(Tensor {
                name: name.clone(),
                dtype,
                shape,
                file: index,
                offset: payload_offset + begin,
                bytes: end - begin,
            });
        }
        if self.tensors.len() == before {
            return false;
        }
        self.files.push(MappedSafetensors {
            file,
            payload_offset,
        });
        true
    }

    /// `TensorStore::Find`: the first tensor of that name.
    pub fn find(&self, name: &str) -> Option<&Tensor> {
        self.by_name.get(name).map(|&index| &self.tensors[index])
    }

    pub fn file(&self, index: usize) -> &Arc<MappedFile> {
        &self.files[index].file
    }

    /// The payload bytes of `tensor`.
    pub fn bytes(&self, tensor: &Tensor) -> &[u8] {
        &self.files[tensor.file].file.bytes()[tensor.offset..tensor.offset + tensor.bytes]
    }

    /// `RegionsFor`: one page-aligned envelope per file holding tensors whose
    /// names start with `prefix`.
    pub fn regions(&self, prefix: &str) -> Result<Vec<Region>, String> {
        let page = rustix::param::page_size();
        if page == 0 || !page.is_multiple_of(16) {
            return Err("TTS cannot determine mapped weight alignment".into());
        }
        let mut result = Vec::new();
        for (index, mapped) in self.files.iter().enumerate() {
            let (mut begin, mut end) = (mapped.file.len(), 0usize);
            for tensor in self
                .tensors
                .iter()
                .filter(|t| t.file == index && t.name.starts_with(prefix))
            {
                if tensor.bytes > mapped.file.len() - tensor.offset {
                    return Err("TTS tensor exceeds its mapped shard".into());
                }
                begin = begin.min(tensor.offset);
                end = end.max(tensor.offset + tensor.bytes);
            }
            if end != 0 {
                begin -= begin % page;
                result.push(Region {
                    file: index,
                    begin,
                    size: end - begin,
                    payload_alignment: mapped.payload_offset % 16,
                });
            }
        }
        Ok(result)
    }
}

/// `LooksLikeQwen3Tts`: `config.json` names a supported checkpoint.
pub fn looks_like_qwen3_tts(model_dir: &Path) -> bool {
    ModelConfig::load(model_dir).is_some_and(|config| config.is_supported())
}
