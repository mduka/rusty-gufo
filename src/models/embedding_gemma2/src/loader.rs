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

/// The towers a load includes besides the text backbone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Towers {
    pub vision: bool,
    pub audio: bool,
}

/// Indexes the requested backbones of one safetensors file. Unused towers
/// are skipped without validation.
fn load_safetensors(path: &Path, towers: Towers) -> Result<TensorStore, String> {
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
        let included = name.starts_with(TEXT_TENSOR_PREFIX)
            || (towers.vision
                && (name.starts_with("vision_tower.") || name.starts_with("embed_vision.")))
            || (towers.audio
                && (name.starts_with("audio_tower.") || name.starts_with("embed_audio.")));
        if name == "__metadata__" || !included {
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

fn expected(name: String, shape: &[u64]) -> ExpectedTensor {
    ExpectedTensor {
        name,
        shape: shape.to_vec(),
    }
}

/// The vision tower and its projection (`ExpectedVisionTensors`).
pub fn expected_vision_tensors() -> Vec<ExpectedTensor> {
    let mut tensors = vec![
        expected(
            "vision_tower.patch_embedder.input_proj.weight".into(),
            &[768, 768],
        ),
        expected(
            "vision_tower.patch_embedder.position_embedding_table".into(),
            &[2, 10240, 768],
        ),
        expected(
            "embed_vision.embedding_projection.weight".into(),
            &[512, 768],
        ),
    ];
    for i in 0..16 {
        let p = format!("vision_tower.encoder.layers.{i}.");
        for name in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ] {
            tensors.push(expected(format!("{p}{name}"), &[768]));
        }
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            tensors.push(expected(
                format!("{p}self_attn.{name}.linear.weight"),
                &[768, 768],
            ));
        }
        for name in ["q_norm.weight", "k_norm.weight"] {
            tensors.push(expected(format!("{p}self_attn.{name}"), &[64]));
        }
        for name in ["gate_proj", "up_proj"] {
            tensors.push(expected(
                format!("{p}mlp.{name}.linear.weight"),
                &[3072, 768],
            ));
        }
        tensors.push(expected(
            format!("{p}mlp.down_proj.linear.weight"),
            &[768, 3072],
        ));
    }
    tensors
}

/// The audio tower and its projection (`ExpectedAudioTensors`).
pub fn expected_audio_tensors() -> Vec<ExpectedTensor> {
    let s = "audio_tower.subsample_conv_projection.";
    let mut tensors = vec![
        expected(format!("{s}layer0.conv.weight"), &[128, 1, 3, 3]),
        expected(format!("{s}layer0.norm.weight"), &[128]),
        expected(format!("{s}layer1.conv.weight"), &[32, 128, 3, 3]),
        expected(format!("{s}layer1.norm.weight"), &[32]),
        expected(format!("{s}input_proj_linear.weight"), &[1024, 1024]),
        expected("audio_tower.output_proj.weight".into(), &[1536, 1024]),
        expected("audio_tower.output_proj.bias".into(), &[1536]),
        expected(
            "embed_audio.embedding_projection.weight".into(),
            &[512, 1536],
        ),
    ];
    for i in 0..12 {
        let p = format!("audio_tower.layers.{i}.");
        let linear = |tensors: &mut Vec<ExpectedTensor>, name: &str, out: u64, input: u64| {
            tensors.push(expected(format!("{p}{name}.linear.weight"), &[out, input]));
            for bound in ["input_min", "input_max", "output_min", "output_max"] {
                tensors.push(expected(format!("{p}{name}.{bound}"), &[]));
            }
        };
        for ff in ["feed_forward1.", "feed_forward2."] {
            linear(&mut tensors, &format!("{ff}ffw_layer_1"), 4096, 1024);
            linear(&mut tensors, &format!("{ff}ffw_layer_2"), 1024, 4096);
            for norm in ["pre_layer_norm.weight", "post_layer_norm.weight"] {
                tensors.push(expected(format!("{p}{ff}{norm}"), &[1024]));
            }
        }
        for proj in ["q_proj", "k_proj", "v_proj", "post"] {
            linear(&mut tensors, &format!("self_attn.{proj}"), 1024, 1024);
        }
        tensors.push(expected(
            format!("{p}self_attn.relative_k_proj.weight"),
            &[1024, 1024],
        ));
        tensors.push(expected(format!("{p}self_attn.per_dim_scale"), &[128]));
        linear(&mut tensors, "lconv1d.linear_start", 2048, 1024);
        linear(&mut tensors, "lconv1d.linear_end", 1024, 1024);
        tensors.push(expected(
            format!("{p}lconv1d.depthwise_conv1d.weight"),
            &[1024, 1, 5],
        ));
        for norm in [
            "lconv1d.pre_layer_norm.weight",
            "lconv1d.conv_norm.weight",
            "norm_pre_attn.weight",
            "norm_post_attn.weight",
            "norm_out.weight",
        ] {
            tensors.push(expected(format!("{p}{norm}"), &[1024]));
        }
    }
    tensors
}

/// `LoadModelDirectory`: maps `<model_dir>/model.safetensors`, indexes the
/// text backbone and the requested towers, and checks each tensor against
/// the configuration. Audio tensors are skipped unless requested.
pub fn load_model_directory(model_dir: &Path, towers: Towers) -> Result<LoadedModel, String> {
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
    let store = load_safetensors(&model_dir.join("model.safetensors"), towers)
        .map_err(|error| format!("embedding_gemma2: {error}"))?;
    let mut expected_tensors = expected_text_tensors(&config.text);
    if towers.vision {
        if !config.supported_vision {
            return Err("embedding_gemma2: unsupported vision configuration".into());
        }
        expected_tensors.extend(expected_vision_tensors());
    }
    if towers.audio {
        if !config.supported_audio {
            return Err("embedding_gemma2: unsupported audio configuration".into());
        }
        expected_tensors.extend(expected_audio_tensors());
    }
    for expected in expected_tensors {
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
