//! Synthetic safetensors contracts of the Qwen-Image weights and their
//! upload (`tests/models/qwen_image_21/weights_test.cpp`):
//! cargo test -p gufo-qwen-image-21 --test weights, and with
//! `--profile gpu-test --features gpu` the streamed upload.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use gufo_qwen_image_21::weights::Weights;

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// A checkpoint directory with every configuration and small tensors,
/// removed on drop.
struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("gufo-image-weights-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let fixture = Self { root };
        let root = &fixture.root;
        write(
            &root.join("model_index.json"),
            r#"{"_class_name":"QwenImage21Pipeline"}"#,
        );
        write(
            &root.join("transformer/config.json"),
            r#"{
      "num_layers":32,"num_attention_heads":32,"attention_head_dim":128,
      "context_in_dim":4096,"in_channels":64,"out_channels":64,
      "patch_size":1,"mlp_ratio":3,"eps":0.000001,
      "axes_dims_rope":[16,56,56],"causal_condition":true}"#,
        );
        write(
            &root.join("text_encoder/config.json"),
            r#"{"model_type":"qwen3_vl",
      "text_config":{"hidden_size":4096,"intermediate_size":12288,
        "num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,
        "head_dim":128,"vocab_size":151936,"rms_norm_eps":0.000001,
        "rope_theta":5000000,"hidden_act":"silu","attention_bias":false,
        "rope_scaling":{"rope_type":"default","mrope_interleaved":true,
          "mrope_section":[24,20,20]}},
      "vision_config":{"hidden_size":1152,"intermediate_size":4304,"depth":27,
        "num_heads":16,"patch_size":16,"spatial_merge_size":2,
        "temporal_patch_size":2,"num_position_embeddings":2304,
        "out_hidden_size":4096,"deepstack_visual_indexes":[8,16,24],
        "hidden_act":"gelu_pytorch_tanh"}}"#,
        );
        let mean = vec!["0"; 64].join(",");
        let std = vec!["1"; 64].join(",");
        write(
            &root.join("vae/config.json"),
            &format!(
                r#"{{
      "z_dim":64,"base_dim":96,"decoder_base_dim":144,"in_channels":4,
      "out_channels":4,"num_res_blocks":2,"scale_factor_spatial":16,
      "dim_mult":[1,2,4,8,8],"is_residual":true,
      "temperal_downsample":[false,true,true,true],"attn_scales":[],
      "latents_mean":[{mean}],"latents_std":[{std}]}}"#
            ),
        );
        write(
            &root.join("scheduler/scheduler_config.json"),
            r#"{
      "base_image_seq_len":256,"max_image_seq_len":8192,"base_shift":0.5,
      "max_shift":0.9,"shift_terminal":0.02,"num_train_timesteps":1000,
      "time_shift_type":"exponential","use_dynamic_shifting":true,
      "invert_sigmas":false,"stochastic_sampling":false,"use_beta_sigmas":false,
      "use_exponential_sigmas":false,"use_karras_sigmas":false}"#,
        );
        write(
            &root.join("processor/preprocessor_config.json"),
            r#"{
      "patch_size":16,"temporal_patch_size":2,"merge_size":2,
      "rescale_factor":0.00392156862745098,"do_convert_rgb":true,
      "do_normalize":true,"do_rescale":true,"do_resize":true,
      "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}"#,
        );
        write(
            &root.join("text_encoder/model.safetensors.index.json"),
            r#"{"weight_map":{
      "model.language_model.a":"a.safetensors",
      "model.language_model.b":"b.safetensors"}}"#,
        );
        fixture.tensor(
            "text_encoder/a.safetensors",
            "model.language_model.a",
            true,
            false,
            1,
        );
        fixture.tensor(
            "text_encoder/b.safetensors",
            "model.language_model.b",
            false,
            false,
            1,
        );
        write(
            &root.join("transformer/diffusion_pytorch_model.safetensors.index.json"),
            r#"{"weight_map":{"a.weight":"a.safetensors",
             "b.weight":"b.safetensors","c.weight":"c.safetensors"}}"#,
        );
        for name in ["a", "b", "c"] {
            fixture.tensor(
                &format!("transformer/{name}.safetensors"),
                &format!("{name}.weight"),
                false,
                false,
                1,
            );
        }
        fixture.tensor(
            "vae/diffusion_pytorch_model.safetensors",
            "quant_conv.a",
            true,
            false,
            1,
        );
        fixture
    }

    /// One tensor per file at a deliberately unaligned payload offset
    /// (37 past a 4 KiB boundary), ending at the end of the file: four
    /// values, or 4096 x 4096 with row `r` scaled by `2^(r % 4)` times
    /// `scale` (rows are 4096 elements, so the factor follows `i / 4096`).
    fn tensor(&self, relative: &str, name: &str, bf16: bool, large: bool, scale: i32) {
        let values = [1.0f32, -2.5, 0.3333, 13.0625].map(|v| v * scale as f32);
        let count = if large { 4096 * 4096 } else { values.len() };
        let bytes = count * if bf16 { 2 } else { 4 };
        let mut header = format!(
            r#"{{"{name}":{{"dtype":"{}","shape":{},"data_offsets":[0,{bytes}]}}}}"#,
            if bf16 { "BF16" } else { "F32" },
            if large { "[4096,4096]" } else { "[4]" },
        );
        while (8 + header.len()) % 4096 != 37 {
            header.push(' ');
        }
        let mut out = Vec::with_capacity(8 + header.len() + bytes);
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        for i in 0..count {
            let value = values[i % 4]
                * if large {
                    (1 << ((i / 4096) % 4)) as f32
                } else {
                    1.0
                };
            if bf16 {
                let bits = value.to_bits();
                let rounded = ((bits + 32767 + ((bits >> 16) & 1)) >> 16) as u16;
                out.extend_from_slice(&rounded.to_le_bytes());
            } else {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        std::fs::write(self.root.join(relative), out).unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// `FilesAndGroups`: the shards, groups in name order and the payload
/// offsets of every tensor.
#[test]
fn files_and_groups() {
    let fixture = Fixture::new("groups");
    let weights = Weights::load(&fixture.root).unwrap();
    assert_eq!(weights.files().len(), 6);
    assert!(weights.group("absent.").is_empty());
    let group = weights.group("text_encoder.model.language_model.");
    assert_eq!(group.len(), 2);
    assert_eq!(group[0].0, "text_encoder.model.language_model.a");
    assert_eq!(group[1].0, "text_encoder.model.language_model.b");
    assert!(group[0].1.bf16 && !group[1].1.bf16);
    assert_ne!(group[0].1.file, group[1].1.file);
    let all = weights.group("");
    assert_eq!(all.len(), 6);
    for (name, weight) in all {
        assert!(std::ptr::eq(weight, weights.get(name).unwrap()));
        assert_eq!(weight.range.start % 4096, 37);
        let file = &weights.files()[weight.file];
        assert_eq!(weight.range.end, file.len());
        assert_eq!(weights.bytes(weight), &file.bytes()[weight.range.clone()]);
    }
}

/// `Upload`: the streamed components read the descriptors the shards were
/// mapped from (the pathnames now name replacements), FP32 sources convert
/// to BF16, three 96 MiB staged tensors cross the batch budget, and dense
/// projections are packed as `[K/16][row][16]`. The VAE tensor lies outside
/// the streamed prefixes and takes the mapped per-tensor path.
#[cfg(feature = "gpu")]
#[test]
fn streamed_uploads_read_the_retained_files() {
    use gufo_qwen_image_21::runtime::Runtime;

    let fixture = Fixture::new("upload");
    for (index, name) in ["a", "b", "c"].into_iter().enumerate() {
        fixture.tensor(
            &format!("transformer/{name}.safetensors"),
            &format!("{name}.weight"),
            false,
            true,
            1 << index,
        );
    }
    let weights = Weights::load(&fixture.root).unwrap();
    let replaced: Vec<PathBuf> = walk(&fixture.root)
        .into_iter()
        .filter(|path| path.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    for path in &replaced {
        let mut old = path.clone().into_os_string();
        old.push(".old");
        std::fs::rename(path, old).unwrap();
        write(path, "replacement must not supply GPU bytes");
    }
    let mut runtime = Runtime::new().unwrap();
    let expected = vec![1.0f32, -2.5, 171.0 / 512.0, 13.0625];
    let a = runtime
        .weight(&weights, "text_encoder.model.language_model.a")
        .unwrap();
    let b = runtime
        .weight(&weights, "text_encoder.model.language_model.b")
        .unwrap();
    assert_eq!(runtime.download(&a).unwrap(), expected);
    assert_eq!(runtime.download(&b).unwrap(), expected);
    assert_eq!(runtime.weight_bytes(), 16);
    let vae = runtime.weight(&weights, "vae.quant_conv.a").unwrap();
    assert_eq!(runtime.download(&vae).unwrap(), expected);
    for (index, name) in ["a", "b", "c"].into_iter().enumerate() {
        let key = format!("transformer.{name}.weight");
        let matrix = runtime.weight(&weights, &key).unwrap();
        assert_eq!(runtime.weight_bytes(), 24 + 3 * 4096 * 4096 * 2);
        assert!(matrix.packed && matrix.rows == 4096 && matrix.cols == 4096);
        assert!(runtime.weight(&weights, &key).unwrap().same_view(&matrix));
        let packed = runtime.download(&matrix).unwrap();
        for row in [0usize, 19, 4095] {
            for col in [0usize, 1, 2, 3, 16, 2047, 4095] {
                let offset = col / 16 * 4096 * 16 + row * 16 + col % 16;
                let want = expected[col % 4] * (1 << index) as f32 * (1 << (row % 4)) as f32;
                assert_eq!(packed[offset], want, "{key} row {row} col {col}");
            }
        }
    }
}

#[cfg(feature = "gpu")]
fn walk(directory: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            paths.extend(walk(&path));
        } else {
            paths.push(path);
        }
    }
    paths
}
