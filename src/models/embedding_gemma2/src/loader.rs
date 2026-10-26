//! `model.safetensors` (`loader.cpp`, `tensor.cpp`): maps the checkpoint,
//! indexes the text-backbone tensors and checks that each one exists with the
//! shape and dtype the configuration requires. The vision and audio towers
//! share the file and are never read.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use gufo_core::json;
use gufo_hip::MappedFile;

use crate::config::{ModelConfig, TextConfig};

/// Tensor name prefix of the text backbone in model.safetensors.
pub const TEXT_TENSOR_PREFIX: &str = "language_model.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    Bf16,
    F16,
    Unknown,
}

impl DType {
    fn parse(name: &str) -> Self {
        match name {
            "F32" => Self::F32,
            "BF16" => Self::Bf16,
            "F16" => Self::F16,
            _ => Self::Unknown,
        }
    }

    pub fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 | Self::F16 => 2,
            Self::Unknown => 0,
        }
    }
}

/// One tensor of the mapped checkpoint: its type, shape and byte range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    /// Absolute byte offset in the file.
    pub offset: usize,
    pub byte_count: usize,
}

impl Tensor {
    /// The product of the dimensions, `None` on overflow.
    pub fn num_elements(&self) -> Option<usize> {
        self.shape.iter().try_fold(1usize, |elements, &dimension| {
            elements.checked_mul(usize::try_from(dimension).ok()?)
        })
    }
}

/// The indexed tensors of one mapped checkpoint.
pub struct TensorStore {
    file: Arc<MappedFile>,
    tensors: HashMap<String, Tensor>,
}

impl TensorStore {
    pub fn find(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn file(&self) -> &Arc<MappedFile> {
        &self.file
    }

    /// The bytes of `tensor`, which belongs to this store.
    pub fn bytes(&self, tensor: &Tensor) -> &[u8] {
        &self.file.bytes()[tensor.offset..tensor.offset + tensor.byte_count]
    }
}

/// A validated checkpoint. The mapping lives as long as the store.
pub struct LoadedModel {
    pub config: ModelConfig,
    pub store: TensorStore,
}

/// A tensor the text backbone needs, with its expected shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedTensor {
    pub name: String,
    pub shape: Vec<u64>,
}

/// Indexes the text-backbone tensors of one safetensors file. Tensors that
/// belong to the other towers are skipped without validation.
fn load_safetensors(path: &Path) -> Result<TensorStore, String> {
    let cannot_map = || format!("cannot map {}", path.display());
    let file = MappedFile::open(path).map_err(|_| cannot_map())?;
    let bytes = file.bytes();
    if bytes.len() < 8 {
        return Err(cannot_map());
    }
    let mut length = [0u8; 8];
    length.copy_from_slice(&bytes[..8]);
    let header_size = u64::from_le_bytes(length);
    let invalid = || format!("invalid safetensors header in {}", path.display());
    if header_size == 0 || header_size > bytes.len() as u64 - 8 {
        return Err(invalid());
    }
    let payload_offset = 8 + header_size as usize;
    let header = json::parse(&bytes[8..payload_offset])
        .map_err(|_| format!("unparsable safetensors header in {}", path.display()))?;
    if !header.is_object() {
        return Err(invalid());
    }
    let available = (bytes.len() - payload_offset) as u64;
    let mut tensors = HashMap::new();
    for (name, specification) in header.members() {
        if name == "__metadata__" || !name.starts_with(TEXT_TENSOR_PREFIX) {
            continue;
        }
        if !specification.is_object() {
            return Err(format!("invalid tensor entry {name}"));
        }
        let mut tensor = Tensor {
            name: name.clone(),
            dtype: DType::parse(&specification.member_str("dtype", "")),
            shape: Vec::new(),
            offset: 0,
            byte_count: 0,
        };
        if let Some(shape) = specification.find("shape").filter(|s| s.is_array()) {
            tensor.shape = shape
                .items()
                .iter()
                .map(|dimension| dimension.as_size(0) as u64)
                .collect();
        }
        let offsets = specification
            .find("data_offsets")
            .filter(|o| o.is_array() && o.len() == 2)
            .ok_or_else(|| format!("missing data_offsets for {name}"))?;
        let begin = offsets.items()[0].as_size(0) as u64;
        let end = offsets.items()[1].as_size(0) as u64;
        let element_size = tensor.dtype.size() as u64;
        if element_size == 0 {
            return Err(format!("unsupported dtype for {name}"));
        }
        let elements = tensor.num_elements().map(|e| e as u64);
        let consistent = elements.is_some_and(|elements| {
            begin < end
                && end <= available
                && elements <= u64::MAX / element_size
                && end - begin == elements * element_size
        });
        if !consistent {
            return Err(format!("inconsistent tensor extent for {name}"));
        }
        tensor.offset = payload_offset + begin as usize;
        tensor.byte_count = (end - begin) as usize;
        if tensors.insert(name.clone(), tensor).is_some() {
            return Err(format!("duplicate tensor {name}"));
        }
    }
    Ok(TensorStore { file, tensors })
}

/// Every tensor of one decoder layer.
fn append_layer_tensors(config: &TextConfig, index: usize, tensors: &mut Vec<ExpectedTensor>) {
    let layer = &config.layers[index];
    let hidden = u64::from(config.hidden_size);
    let per_layer = u64::from(config.hidden_size_per_layer_input);
    let query = u64::from(config.num_attention_heads * layer.head_dim);
    let key = u64::from(layer.num_key_value_heads * layer.head_dim);
    let intermediate = u64::from(config.intermediate_size);
    let prefix = format!("{TEXT_TENSOR_PREFIX}layers.{index}.");
    let mut add = |suffix: &str, shape: &[u64]| {
        tensors.push(ExpectedTensor {
            name: format!("{prefix}{suffix}"),
            shape: shape.to_vec(),
        });
    };
    add("input_layernorm.weight", &[hidden]);
    add("post_attention_layernorm.weight", &[hidden]);
    add("pre_feedforward_layernorm.weight", &[hidden]);
    add("post_feedforward_layernorm.weight", &[hidden]);
    add("layer_scalar", &[1]);
    add("self_attn.q_proj.weight", &[query, hidden]);
    add("self_attn.k_proj.weight", &[key, hidden]);
    add("self_attn.v_proj.weight", &[key, hidden]);
    add("self_attn.o_proj.weight", &[hidden, query]);
    add("self_attn.q_norm.weight", &[u64::from(layer.head_dim)]);
    add("self_attn.k_norm.weight", &[u64::from(layer.head_dim)]);
    add("mlp.gate_proj.weight", &[intermediate, hidden]);
    add("mlp.up_proj.weight", &[intermediate, hidden]);
    add("mlp.down_proj.weight", &[hidden, intermediate]);
    add(
        "ple_block.per_layer_input_gate.weight",
        &[per_layer, hidden],
    );
    add(
        "ple_block.per_layer_projection.weight",
        &[hidden, per_layer],
    );
    add("ple_block.post_per_layer_input_norm.weight", &[hidden]);
}

/// Names every tensor the text backbone needs, with its expected shape.
pub fn expected_text_tensors(config: &TextConfig) -> Vec<ExpectedTensor> {
    let hidden = u64::from(config.hidden_size);
    let per_layer = u64::from(config.hidden_size_per_layer_input);
    let mut tensors = vec![
        ExpectedTensor {
            name: format!("{TEXT_TENSOR_PREFIX}embed_tokens.weight"),
            shape: vec![u64::from(config.vocab_size), hidden],
        },
        ExpectedTensor {
            name: format!("{TEXT_TENSOR_PREFIX}norm.weight"),
            shape: vec![hidden],
        },
        ExpectedTensor {
            name: format!("{TEXT_TENSOR_PREFIX}embedding_projection.weight"),
            shape: vec![u64::from(config.embedding_dim), hidden],
        },
        ExpectedTensor {
            name: format!("{TEXT_TENSOR_PREFIX}ple.per_layer_model_projection.weight"),
            shape: vec![u64::from(config.num_hidden_layers) * per_layer, hidden],
        },
        ExpectedTensor {
            name: format!("{TEXT_TENSOR_PREFIX}ple.per_layer_projection_norm.weight"),
            shape: vec![per_layer],
        },
    ];
    for index in 0..config.layers.len() {
        append_layer_tensors(config, index, &mut tensors);
    }
    tensors
}

/// `LoadModelDirectory`: maps `<model_dir>/model.safetensors`, indexes the
/// text-backbone tensors and checks each against the configuration.
pub fn load_model_directory(model_dir: &Path) -> Result<LoadedModel, String> {
    let config = ModelConfig::load(model_dir).ok_or_else(|| {
        format!(
            "embedding_gemma2: cannot parse {}/config.json",
            model_dir.display()
        )
    })?;
    if !config.is_supported() {
        return Err(
            "embedding_gemma2: expected the google/embeddinggemma-2 BF16 configuration".into(),
        );
    }
    let store = load_safetensors(&model_dir.join("model.safetensors"))
        .map_err(|error| format!("embedding_gemma2: {error}"))?;
    for expected in expected_text_tensors(&config.text) {
        let tensor = store
            .find(&expected.name)
            .ok_or_else(|| format!("embedding_gemma2: missing tensor {}", expected.name))?;
        if tensor.dtype != DType::Bf16 || tensor.shape != expected.shape {
            return Err(format!(
                "embedding_gemma2: unexpected dtype or shape for {}",
                expected.name
            ));
        }
    }
    Ok(LoadedModel { config, store })
}

/// Whether `model_dir` holds the supported configuration.
pub fn looks_like_embedding_gemma2(model_dir: &Path) -> bool {
    ModelConfig::load(model_dir).is_some_and(|config| config.is_supported())
}
