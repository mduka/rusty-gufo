//! The files of a VoxCPM2 checkpoint in either layout (`model_files.hpp`,
//! `model_files.cpp`):
//!  - the openbmb Hugging Face directory: `config.json`, `tokenizer.json`,
//!    `model.safetensors` (bfloat16) and `audiovae.pth`;
//!  - one audio.cpp GGUF (`general.architecture` "audiocpp", family
//!    "voxcpm2"), which embeds those files' contents: the generator's
//!    tensors under "weights/", AudioVAE's under "audiovae_weights/", names
//!    too long for GGUF listed in `audiocpp.tensor_names`, linear weights in
//!    BF16, F16, F32 or Q8_0.
//!
//! Tensors keep the reference's names and PyTorch's dimension order.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use gufo_core::gguf::{self, Gguf};
use gufo_formats::GgmlType;
use gufo_hip::MappedFile;

use crate::safetensors::{DType, SafetensorsFile, Tensor};
use crate::torch_checkpoint::TorchCheckpoint;

const MODEL_PREFIX: &str = "weights/";
const VAE_PREFIX: &str = "audiovae_weights/";
const ABBREVIATED: &str = "_audiocpp.";

/// Whether two shapes agree up to dimensions of size 1 (GGUF drops trailing
/// ones of GGML's order, so `[1, C, 1]` arrives as `[C, 1]`); the data
/// layout is the same (`SameShape`).
pub fn same_shape(a: &[usize], b: &[usize]) -> bool {
    a.iter()
        .filter(|&&size| size != 1)
        .eq(b.iter().filter(|&&size| size != 1))
}

/// The files of a VoxCPM2 checkpoint (`ModelFiles`).
pub struct ModelFiles {
    config: String,
    tokenizer: String,
    safetensors: Option<SafetensorsFile>,
    /// Kept so the AudioVAE storage stays mapped with its tensors.
    _checkpoint: Option<TorchCheckpoint>,
    gguf: Option<Arc<MappedFile>>,
    model: Vec<Tensor>,
    vae: Vec<Tensor>,
    model_index: HashMap<String, usize>,
    vae_index: HashMap<String, usize>,
}

fn read_text(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|_| format!("cannot read {}", path.display()))?;
    String::from_utf8(bytes).map_err(|_| format!("{} is not UTF-8", path.display()))
}

fn from_ggml(ty: GgmlType) -> Option<DType> {
    Some(match ty {
        GgmlType::F32 => DType::F32,
        GgmlType::F16 => DType::F16,
        GgmlType::Bf16 => DType::Bf16,
        GgmlType::Q8_0 => DType::Q8_0,
        GgmlType::I32 => DType::I32,
        _ => return None,
    })
}

/// The integers of a metadata array, whatever their signedness.
fn integers(reader: &Gguf<'_>, key: &str) -> Result<Vec<u64>, String> {
    match reader.metadata(key) {
        None => Err(format!("GGUF lacks {key}")),
        Some(gguf::Value::UIntArray(values)) => Ok(values.clone()),
        Some(gguf::Value::IntArray(values)) => Ok(values.iter().map(|&v| v as u64).collect()),
        Some(_) => Err(format!("GGUF {key} is not an integer array")),
    }
}

/// A string array of the metadata, empty when absent
/// (`GetMetadataStringArray`).
fn strings<'a>(reader: &Gguf<'a>, key: &str) -> Vec<&'a [u8]> {
    reader.strings(key).map(<[_]>::to_vec).unwrap_or_default()
}

/// An embedded file of an audio.cpp GGUF.
fn embedded(reader: &Gguf<'_>, name: &str) -> Result<String, String> {
    let names = strings(reader, "audiocpp.embedded_files.names");
    let offsets = integers(reader, "audiocpp.embedded_files.offsets")?;
    let data = integers(reader, "audiocpp.embedded_files.data")?;
    for (index, &entry) in names.iter().enumerate() {
        if entry != name.as_bytes() {
            continue;
        }
        if index + 1 >= offsets.len()
            || offsets[index] > offsets[index + 1]
            || offsets[index + 1] > data.len() as u64
        {
            return Err("GGUF embedded file table is malformed".into());
        }
        let bytes: Vec<u8> = data[offsets[index] as usize..offsets[index + 1] as usize]
            .iter()
            .map(|&value| value as u8)
            .collect();
        return String::from_utf8(bytes).map_err(|_| format!("GGUF embedded {name} is not UTF-8"));
    }
    Err(format!("GGUF lacks the embedded {name}"))
}

fn is_voxcpm2(reader: &Gguf<'_>) -> bool {
    reader.string("general.architecture") == Some(b"audiocpp".as_slice())
        && reader.string("audiocpp.model_spec.family") == Some(b"voxcpm2".as_slice())
}

/// `std::stoul` of the abbreviation's index: leading digits.
fn abbreviated_index(text: &str) -> Option<usize> {
    let digits: &str = &text[..text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len())];
    digits.parse().ok()
}

fn index_of(tensors: &[Tensor]) -> HashMap<String, usize> {
    let mut index = HashMap::with_capacity(tensors.len());
    for (position, tensor) in tensors.iter().enumerate() {
        index.entry(tensor.name.clone()).or_insert(position);
    }
    index
}

impl ModelFiles {
    /// Whether `path` is a GGUF file of audio.cpp's VoxCPM2 layout (reads the
    /// header only; `IsVoxCpm2Gguf`).
    pub fn is_voxcpm2_gguf(path: &Path) -> bool {
        if !path.is_file() || path.extension().is_none_or(|extension| extension != "gguf") {
            return false;
        }
        let Ok((bytes, len)) = gguf::read_header(path) else {
            return false;
        };
        Gguf::parse_prefix(&bytes, len).is_ok_and(|reader| is_voxcpm2(&reader))
    }

    /// Opens a Hugging Face directory or an audio.cpp GGUF (`Open`).
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut files = if path.is_dir() {
            Self::open_directory(path)?
        } else {
            Self::open_gguf(path)?
        };
        files.model_index = index_of(&files.model);
        files.vae_index = index_of(&files.vae);
        Ok(files)
    }

    fn open_directory(path: &Path) -> Result<Self, String> {
        let config = read_text(&path.join("config.json"))?;
        let tokenizer = read_text(&path.join("tokenizer.json"))?;
        let safetensors = SafetensorsFile::open(&path.join("model.safetensors"))?;
        let checkpoint = TorchCheckpoint::open(&path.join("audiovae.pth"))?;
        let model = safetensors.tensors().to_vec();
        let vae = checkpoint.tensors().to_vec();
        Ok(Self {
            config,
            tokenizer,
            safetensors: Some(safetensors),
            _checkpoint: Some(checkpoint),
            gguf: None,
            model,
            vae,
            model_index: HashMap::new(),
            vae_index: HashMap::new(),
        })
    }

    fn open_gguf(path: &Path) -> Result<Self, String> {
        let mapping =
            MappedFile::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let reader =
            Gguf::parse(mapping.bytes()).map_err(|error| format!("{}: {error}", path.display()))?;
        if !is_voxcpm2(&reader) {
            return Err(format!(
                "{} is not an audio.cpp VoxCPM2 GGUF",
                path.display()
            ));
        }
        let config = embedded(&reader, "config.json")?;
        let tokenizer = embedded(&reader, "tokenizer.json")?;
        let names = strings(&reader, "audiocpp.tensor_names");
        let mut model = Vec::new();
        let mut vae = Vec::new();
        for info in reader.tensors() {
            let mut name = info.name.to_owned();
            if let Some(rest) = name.strip_prefix(ABBREVIATED) {
                let listed = abbreviated_index(rest)
                    .and_then(|index| names.get(index))
                    .ok_or_else(|| format!("GGUF tensor {name} has no listed name"))?;
                name = String::from_utf8(listed.to_vec())
                    .map_err(|_| format!("GGUF tensor {name} has no listed name"))?;
            }
            let dtype = from_ggml(info.ty)
                .ok_or_else(|| format!("GGUF tensor {name} has an unsupported type"))?;
            // GGML lists dimensions innermost first.
            let shape: Vec<usize> = info.dims.iter().rev().map(|&size| size as usize).collect();
            let make = |stripped: &str| {
                Tensor::new(
                    stripped.to_owned(),
                    dtype,
                    shape.clone(),
                    info.bytes,
                    info.offset as u64,
                    mapping.clone(),
                )
            };
            if let Some(stripped) = name.strip_prefix(MODEL_PREFIX) {
                model.push(make(stripped));
            } else if let Some(stripped) = name.strip_prefix(VAE_PREFIX) {
                vae.push(make(stripped));
            } else {
                return Err(format!("GGUF tensor {name} belongs to no component"));
            }
        }
        drop(reader);
        Ok(Self {
            config,
            tokenizer,
            safetensors: None,
            _checkpoint: None,
            gguf: Some(mapping),
            model,
            vae,
            model_index: HashMap::new(),
            vae_index: HashMap::new(),
        })
    }

    /// The text of `config.json`.
    pub fn config(&self) -> &str {
        &self.config
    }

    /// The text of `tokenizer.json`.
    pub fn tokenizer(&self) -> &str {
        &self.tokenizer
    }

    /// Whether the checkpoint is an audio.cpp GGUF.
    pub fn gguf(&self) -> bool {
        self.gguf.is_some()
    }

    /// A generator tensor (model.safetensors names), if present.
    pub fn model(&self, name: &str) -> Option<&Tensor> {
        self.model_index.get(name).map(|&index| &self.model[index])
    }

    /// An AudioVAE tensor (audiovae.pth names), if present.
    pub fn vae(&self, name: &str) -> Option<&Tensor> {
        self.vae_index.get(name).map(|&index| &self.vae[index])
    }

    /// Every generator tensor.
    pub fn model_tensors(&self) -> &[Tensor] {
        &self.model
    }

    /// Every AudioVAE tensor.
    pub fn vae_tensors(&self) -> &[Tensor] {
        &self.vae
    }

    /// Drops resident pages of the weight files once they live on the device
    /// (`ReleasePages`: the safetensors file; a GGUF's pages stay).
    pub fn release_pages(&self) {
        if let Some(safetensors) = &self.safetensors {
            safetensors.release_pages();
        }
    }
}
