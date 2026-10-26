//! The Diffusers directory (`weights.cpp`): the supported configuration of
//! every component, and the safetensors shards of the text encoder,
//! transformer and VAE, mapped read-only under `text_encoder.`,
//! `transformer.` and `vae.` names. Errors keep the C++ messages.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Bound, Range};
use std::path::Path;
use std::sync::Arc;

use gufo_core::json::{self, Value};
use gufo_core::text::{Error, Result};
use gufo_hip::MappedFile;

/// One tensor of a mapped shard (`Weight`).
#[derive(Debug, Clone)]
pub struct Weight {
    pub shape: Vec<usize>,
    /// The mapped shard and the payload's byte range in it.
    pub file: usize,
    pub range: Range<usize>,
    pub bf16: bool,
}

impl Weight {
    pub fn bytes(&self) -> usize {
        self.range.len()
    }
}

/// `ReadJson`: a metadata file of at most 32 MiB.
pub fn read_json(path: &Path) -> Result<Value> {
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if size == 0 || size > 32 << 20 {
        return Err(Error::runtime(format!(
            "cannot read Qwen-Image metadata: {}",
            path.display()
        )));
    }
    let data = std::fs::read(path)
        .ok()
        .filter(|data| data.len() as u64 == size)
        .ok_or_else(|| {
            Error::runtime(format!("truncated Qwen-Image metadata: {}", path.display()))
        })?;
    json::parse(&data)
}

/// `Integer`: an integral number in `[0, 2^40]`.
fn integer(value: &Value) -> Result<usize> {
    let number = value.as_double(-1.0);
    if !value.is_number()
        || !number.is_finite()
        || !(0.0..=(1u64 << 40) as f64).contains(&number)
        || number.fract() != 0.0
    {
        return Err(Error::runtime("invalid Qwen-Image integer metadata"));
    }
    Ok(number as usize)
}

fn unsupported(key: &str) -> Error {
    Error::runtime(format!("unsupported Qwen-Image configuration: {key}"))
}

fn boolean(object: &Value, key: &str, expected: bool) -> Result<()> {
    match object.find(key) {
        Some(value) if value.is_bool() && value.as_bool(!expected) == expected => Ok(()),
        _ => Err(unsupported(key)),
    }
}

fn equal(object: &Value, key: &str, expected: f64) -> Result<()> {
    match object.find(key) {
        Some(value) if value.is_number() && value.as_double(f64::NAN) == expected => Ok(()),
        _ => Err(unsupported(key)),
    }
}

fn equal_all(object: &Value, expected: &[(&str, f64)]) -> Result<()> {
    expected
        .iter()
        .try_for_each(|&(key, value)| equal(object, key, value))
}

fn array_equal(object: &Value, key: &str, expected: &[usize]) -> Result<()> {
    let value = object
        .find(key)
        .filter(|value| value.is_array() && value.len() == expected.len())
        .ok_or_else(|| Error::runtime(format!("invalid Qwen-Image array: {key}")))?;
    for (item, &want) in value.items().iter().zip(expected) {
        if integer(item)? != want {
            return Err(Error::runtime(format!(
                "unsupported Qwen-Image array: {key}"
            )));
        }
    }
    Ok(())
}

fn is_true(object: &Value, key: &str) -> bool {
    object
        .find(key)
        .is_some_and(|value| value.is_bool() && value.as_bool(false))
}

/// `Weights`: every tensor by its prefixed name, over the mapped shards.
pub struct Weights {
    files: Vec<Arc<MappedFile>>,
    weights: BTreeMap<String, Weight>,
    vae_config: Value,
}

impl Weights {
    /// Validates the configuration of every component, then maps the
    /// shards.
    pub fn load(root: &Path) -> Result<Self> {
        let index = read_json(&root.join("model_index.json"))?;
        if index.member_str("_class_name", "") != "QwenImage21Pipeline" {
            return Err(Error::runtime(
                "expected a Qwen-Image-2.1 Diffusers directory",
            ));
        }
        let dit = read_json(&root.join("transformer/config.json"))?;
        equal_all(
            &dit,
            &[
                ("num_layers", 32.0),
                ("num_attention_heads", 32.0),
                ("attention_head_dim", 128.0),
                ("context_in_dim", 4096.0),
                ("in_channels", 64.0),
                ("out_channels", 64.0),
                ("patch_size", 1.0),
                ("mlp_ratio", 3.0),
            ],
        )?;
        equal(&dit, "eps", 1e-6)?;
        array_equal(&dit, "axes_dims_rope", &[16, 56, 56])?;
        if !is_true(&dit, "causal_condition") {
            return Err(Error::runtime("Qwen-Image requires causal conditioning"));
        }
        let text = read_json(&root.join("text_encoder/config.json"))?;
        if text.member_str("model_type", "") != "qwen3_vl" {
            return Err(Error::runtime("expected Qwen3-VL image text encoder"));
        }
        let (Some(tc), Some(vc)) = (text.find("text_config"), text.find("vision_config")) else {
            return Err(Error::runtime("missing Qwen3-VL configuration"));
        };
        equal_all(
            tc,
            &[
                ("hidden_size", 4096.0),
                ("intermediate_size", 12288.0),
                ("num_hidden_layers", 36.0),
                ("num_attention_heads", 32.0),
                ("num_key_value_heads", 8.0),
                ("head_dim", 128.0),
                ("vocab_size", 151936.0),
            ],
        )?;
        equal(tc, "rms_norm_eps", 1e-6)?;
        equal(tc, "rope_theta", 5_000_000.0)?;
        let rope = tc.find("rope_scaling");
        if !rope.is_some_and(|rope| {
            rope.member_str("rope_type", "") == "default"
                && rope
                    .find("mrope_interleaved")
                    .is_some_and(|v| v.as_bool(false))
        }) {
            return Err(Error::runtime("unsupported Qwen3-VL rotary configuration"));
        }
        if let Some(rope) = rope {
            array_equal(rope, "mrope_section", &[24, 20, 20])?;
        }
        equal_all(
            vc,
            &[
                ("hidden_size", 1152.0),
                ("intermediate_size", 4304.0),
                ("depth", 27.0),
                ("num_heads", 16.0),
                ("patch_size", 16.0),
                ("spatial_merge_size", 2.0),
                ("temporal_patch_size", 2.0),
                ("num_position_embeddings", 2304.0),
                ("out_hidden_size", 4096.0),
            ],
        )?;
        array_equal(vc, "deepstack_visual_indexes", &[8, 16, 24])?;
        if tc.member_str("hidden_act", "") != "silu"
            || vc.member_str("hidden_act", "") != "gelu_pytorch_tanh"
        {
            return Err(Error::runtime("unsupported Qwen3-VL activation"));
        }
        boolean(tc, "attention_bias", false)?;
        let vae_config = read_json(&root.join("vae/config.json"))?;
        equal_all(
            &vae_config,
            &[
                ("z_dim", 64.0),
                ("base_dim", 96.0),
                ("decoder_base_dim", 144.0),
                ("in_channels", 4.0),
                ("out_channels", 4.0),
                ("num_res_blocks", 2.0),
                ("scale_factor_spatial", 16.0),
            ],
        )?;
        array_equal(&vae_config, "dim_mult", &[1, 2, 4, 8, 8])?;
        boolean(&vae_config, "is_residual", true)?;
        let temporal = vae_config.find("temperal_downsample");
        let attention = vae_config.find("attn_scales");
        let (Some(temporal), Some(attention)) = (temporal, attention) else {
            return Err(Error::runtime("unsupported VAE block layout"));
        };
        if !temporal.is_array()
            || temporal.len() != 4
            || !attention.is_array()
            || !attention.is_empty()
        {
            return Err(Error::runtime("unsupported VAE block layout"));
        }
        for (i, item) in temporal.items().iter().enumerate() {
            if !item.is_bool() || item.as_bool(false) != (i != 0) {
                return Err(Error::runtime("unsupported VAE temporal layout"));
            }
        }
        for key in ["latents_mean", "latents_std"] {
            let values = vae_config
                .find(key)
                .filter(|values| values.is_array() && values.len() == 64)
                .ok_or_else(|| Error::runtime("invalid VAE latent normalization"))?;
            for value in values.items() {
                let number = value.as_double(f64::NAN);
                if !value.is_number()
                    || !number.is_finite()
                    || (key == "latents_std" && number <= 0.0)
                {
                    return Err(Error::runtime("invalid VAE latent normalization"));
                }
            }
        }
        let scheduler = read_json(&root.join("scheduler/scheduler_config.json"))?;
        equal_all(
            &scheduler,
            &[
                ("base_image_seq_len", 256.0),
                ("max_image_seq_len", 8192.0),
                ("base_shift", 0.5),
                ("max_shift", 0.9),
                ("shift_terminal", 0.02),
                ("num_train_timesteps", 1000.0),
            ],
        )?;
        if scheduler.member_str("time_shift_type", "") != "exponential"
            || !scheduler
                .find("use_dynamic_shifting")
                .is_some_and(|v| v.as_bool(false))
        {
            return Err(Error::runtime("unsupported Qwen-Image flow schedule"));
        }
        for key in [
            "invert_sigmas",
            "stochastic_sampling",
            "use_beta_sigmas",
            "use_exponential_sigmas",
            "use_karras_sigmas",
        ] {
            boolean(&scheduler, key, false)?;
        }
        let processor = read_json(&root.join("processor/preprocessor_config.json"))?;
        equal_all(
            &processor,
            &[
                ("patch_size", 16.0),
                ("temporal_patch_size", 2.0),
                ("merge_size", 2.0),
            ],
        )?;
        equal(&processor, "rescale_factor", 1.0 / 255.0)?;
        for key in ["do_convert_rgb", "do_normalize", "do_rescale", "do_resize"] {
            boolean(&processor, key, true)?;
        }
        for key in ["image_mean", "image_std"] {
            let values = processor
                .find(key)
                .filter(|values| values.is_array() && values.len() == 3)
                .ok_or_else(|| Error::runtime("invalid vision normalization"))?;
            if values
                .items()
                .iter()
                .any(|value| !value.is_number() || value.as_double(f64::NAN) != 0.5)
            {
                return Err(Error::runtime("unsupported vision normalization"));
            }
        }
        let mut weights = Self {
            files: Vec::new(),
            weights: BTreeMap::new(),
            vae_config,
        };
        for (component, filename) in [
            ("text_encoder", "model.safetensors.index.json"),
            (
                "transformer",
                "diffusion_pytorch_model.safetensors.index.json",
            ),
        ] {
            let manifest = read_json(&root.join(component).join(filename))?;
            let weight_map = manifest
                .find("weight_map")
                .filter(|map| map.is_object())
                .ok_or_else(|| Error::runtime("missing safetensors weight map"))?;
            let mut shards = BTreeSet::new();
            for (_, file) in weight_map.members() {
                let name = file.str();
                if !file.is_string()
                    || Path::new(name).file_name().and_then(|n| n.to_str()) != Some(name)
                {
                    return Err(Error::runtime("invalid safetensors shard path"));
                }
                shards.insert(name.to_owned());
            }
            for shard in &shards {
                weights.add_file(&root.join(component).join(shard), &format!("{component}."))?;
            }
            for (name, _) in weight_map.members() {
                if !weights.contains(&format!("{component}.{name}")) {
                    return Err(Error::runtime(format!("missing safetensors tensor {name}")));
                }
            }
        }
        weights.add_file(
            &root.join("vae/diffusion_pytorch_model.safetensors"),
            "vae.",
        )?;
        Ok(weights)
    }

    /// `Weights::Load`: one shard, whose tensors must tile its payload.
    fn add_file(&mut self, path: &Path, prefix: &str) -> Result<()> {
        let size = std::fs::File::open(path)
            .and_then(|file| file.metadata())
            .map_err(|_| Error::runtime(format!("cannot open {}", path.display())))?
            .len();
        if size < 8 {
            return Err(Error::runtime(format!(
                "invalid safetensors file {}",
                path.display()
            )));
        }
        let file = MappedFile::open(path)
            .map_err(|_| Error::runtime(format!("cannot map {}", path.display())))?;
        let size = file.len();
        let data = file.bytes();
        let header_size = u64::from_le_bytes(data[..8].try_into().unwrap_or_default());
        if header_size == 0 || header_size > 32 << 20 || header_size > (size - 8) as u64 {
            return Err(Error::runtime("invalid safetensors header"));
        }
        let header_end = 8 + header_size as usize;
        let header = json::parse(&data[8..header_end])?;
        if !header.is_object() {
            return Err(Error::runtime("invalid safetensors header"));
        }
        let payload_size = size - header_end;
        let index = self.files.len();
        let mut ranges = Vec::new();
        for (name, entry) in header.members() {
            if name == "__metadata__" {
                continue;
            }
            let dtype = entry.member_str("dtype", "");
            if dtype != "BF16" && dtype != "F32" {
                return Err(Error::runtime(format!(
                    "unsupported Qwen-Image weight dtype {dtype}"
                )));
            }
            let shape = entry.find("shape");
            let offsets = entry.find("data_offsets");
            let (Some(shape), Some(offsets)) = (shape, offsets) else {
                return Err(Error::runtime(format!("invalid tensor layout {name}")));
            };
            if !shape.is_array() || shape.len() > 5 || !offsets.is_array() || offsets.len() != 2 {
                return Err(Error::runtime(format!("invalid tensor layout {name}")));
            }
            let bf16 = dtype == "BF16";
            let mut bytes = if bf16 { 2usize } else { 4 };
            let mut dims = Vec::with_capacity(shape.len());
            for dim in shape.items() {
                let value = integer(dim)?;
                if value == 0 || value > i32::MAX as usize || bytes > payload_size / value {
                    return Err(Error::runtime(format!("invalid tensor shape {name}")));
                }
                dims.push(value);
                bytes *= value;
            }
            let begin = integer(&offsets.items()[0])?;
            let end = integer(&offsets.items()[1])?;
            if begin > end || end > payload_size || end - begin != bytes {
                return Err(Error::runtime(format!("invalid tensor extent {name}")));
            }
            ranges.push((begin, end));
            let weight = Weight {
                shape: dims,
                file: index,
                range: header_end + begin..header_end + end,
                bf16,
            };
            if self
                .weights
                .insert(format!("{prefix}{name}"), weight)
                .is_some()
            {
                return Err(Error::runtime(format!("duplicate tensor {name}")));
            }
        }
        ranges.sort_unstable();
        let mut cursor = 0;
        for (begin, end) in ranges {
            if begin != cursor {
                return Err(Error::runtime("noncontiguous safetensors payload"));
            }
            cursor = end;
        }
        if cursor != payload_size {
            return Err(Error::runtime("unexpected safetensors data"));
        }
        self.files.push(file);
        Ok(())
    }

    /// `Weights::Prefetch`: faults the envelope of every tensor under
    /// `prefix` in each shard with parallel readers.
    pub fn prefetch(&self, prefix: &str) -> Result<()> {
        for (index, file) in self.files.iter().enumerate() {
            let (mut begin, mut end) = (file.len(), 0);
            for (_, weight) in self
                .weights
                .range::<str, _>((Bound::Included(prefix), Bound::Unbounded))
                .take_while(|(name, _)| name.starts_with(prefix))
                .filter(|(_, weight)| weight.file == index)
            {
                begin = begin.min(weight.range.start);
                end = end.max(weight.range.end);
            }
            if end > begin {
                file.prefault(begin..end)
                    .map_err(|e| Error::runtime(format!("cannot read mapped weights: {e}")))?;
            }
        }
        Ok(())
    }

    /// `Weights::Get`.
    pub fn get(&self, name: &str) -> Result<&Weight> {
        self.weights
            .get(name)
            .ok_or_else(|| Error::runtime(format!("missing Qwen-Image tensor {name}")))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    /// The payload of `weight`.
    pub fn bytes(&self, weight: &Weight) -> &[u8] {
        &self.files[weight.file].bytes()[weight.range.clone()]
    }

    pub fn vae_config(&self) -> &Value {
        &self.vae_config
    }
}
