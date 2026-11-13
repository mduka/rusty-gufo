//! The C++ extension's `weights_test.cpp`: the torch checkpoint reader on
//! small `torch.save` files, and both weight files of the released model
//! (`--features models`), plus the tensor index of audio.cpp's GGUFs.

#![allow(clippy::unwrap_used)]

mod config_samples;

use config_samples::fixture;
use gufo_voxcpm::safetensors::{DType, Tensor};
use gufo_voxcpm::torch_checkpoint::TorchCheckpoint;

fn f32s(tensor: &Tensor) -> Vec<f32> {
    tensor
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn i32s(tensor: &Tensor) -> Vec<i32> {
    tensor
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| i32::from_le_bytes(*word))
        .collect()
}

fn bf16s(tensor: &Tensor) -> Vec<f32> {
    tensor
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|half| f32::from_bits(u32::from(u16::from_le_bytes(*half)) << 16))
        .collect()
}

#[test]
fn samples() {
    let sample = TorchCheckpoint::open(&fixture("sample.pth")).unwrap();
    assert_eq!(sample.tensors().len(), 6, "six tensors");
    let weight = sample.find("layer.weight").unwrap();
    assert!(
        weight.dtype == DType::F32
            && weight.shape == [2, 3]
            && f32s(weight) == [1.5, -2.0, 3.25, 0.0, 4.0, -0.5],
        "float32 matrix"
    );
    let bias = sample.find("layer.bias").unwrap();
    assert!(
        bias.dtype == DType::I32 && i32s(bias) == [7, -8, 9, 2147483647],
        "int32 vector"
    );
    let half = sample.find("half").unwrap();
    assert!(
        half.dtype == DType::Bf16 && bf16s(half)[..2] == [1.0, -3.0],
        "bfloat16 vector"
    );
    let tail = sample.find("tail").unwrap();
    let head = sample.find("head").unwrap();
    assert!(
        f32s(tail) == [6.0, 7.0, 8.0, 9.0] && f32s(head) == [0.0, 1.0],
        "views at offsets into a shared storage"
    );
    let scalar = sample.find("scalar").unwrap();
    assert!(
        scalar.shape.is_empty() && f32s(scalar) == [0.25],
        "zero-dimensional tensor"
    );
    let error = TorchCheckpoint::open(&fixture("view.pth")).err().unwrap();
    assert!(
        error.contains("not contiguous"),
        "a transposed view is rejected: {error}"
    );
    assert!(
        TorchCheckpoint::open(&fixture("tokenizer_golden.json")).is_err(),
        "a non-zip file is rejected"
    );
}

#[cfg(feature = "models")]
#[test]
fn released_model() {
    use gufo_voxcpm::config::load_config;
    use gufo_voxcpm::safetensors::SafetensorsFile;

    let root = config_samples::model_root();
    let config = load_config(&root).unwrap();
    let model = SafetensorsFile::open(&root.join("model.safetensors")).unwrap();
    assert_eq!(model.tensors().len(), 577, "577 model tensors");
    let hidden = config.lm.hidden;
    model
        .expect(
            "base_lm.embed_tokens.weight",
            DType::Bf16,
            &[config.lm.vocab, hidden],
        )
        .unwrap();
    model
        .expect(
            "feat_encoder.special_token",
            DType::Bf16,
            &[1, 1, 1, config.encoder.hidden],
        )
        .unwrap();
    model
        .expect(
            "fusion_concat_proj.weight",
            DType::Bf16,
            &[hidden, 2 * hidden],
        )
        .unwrap();

    let vae = TorchCheckpoint::open(&root.join("audiovae.pth")).unwrap();
    assert_eq!(vae.tensors().len(), 312, "312 AudioVAE tensors");
    let boundaries = vae.find("decoder.sr_bin_boundaries").unwrap();
    assert!(
        boundaries.dtype == DType::I32 && i32s(boundaries) == [20000, 30000, 40000],
        "rate bucket boundaries"
    );
    let gain = vae.find("encoder.fc_mu.weight_g").unwrap();
    assert!(
        gain.dtype == DType::F32 && gain.shape == [64, 1, 1],
        "weight-norm gains keep their shape"
    );
    let finite = vae
        .tensors()
        .iter()
        .filter(|tensor| tensor.dtype == DType::F32)
        .all(|tensor| f32s(tensor).iter().all(|value| value.is_finite()));
    assert!(finite, "AudioVAE weights are finite");
}

/// audio.cpp's GGUFs (bfloat16 and Q8_0, when given) hold the generator
/// tensors of the Hugging Face checkpoint under the same names and shapes
/// (up to dimensions of size 1), and the same configuration; the Q8_0 file
/// keeps its matrices 8-bit.
#[cfg(feature = "models")]
#[test]
fn gguf_tensor_index() {
    use std::collections::BTreeSet;

    use gufo_voxcpm::config::parse_config;
    use gufo_voxcpm::model_files::{ModelFiles, same_shape};

    let root = config_samples::model_root();
    let released = ModelFiles::open(&root).unwrap();
    assert!(!released.gguf());
    let released_config = parse_config(released.config()).unwrap();
    let names = |tensors: &[Tensor]| -> BTreeSet<String> {
        tensors.iter().map(|tensor| tensor.name.clone()).collect()
    };
    for path in config_samples::gguf_paths() {
        let files = ModelFiles::open(&path).unwrap();
        assert!(files.gguf());
        assert_eq!(
            parse_config(files.config()).unwrap(),
            released_config,
            "{}: embedded config.json",
            path.display()
        );
        assert_eq!(
            names(files.model_tensors()),
            names(released.model_tensors()),
            "{}: generator tensor names",
            path.display()
        );
        for tensor in released.model_tensors() {
            let other = files.model(&tensor.name).unwrap();
            assert!(
                same_shape(&tensor.shape, &other.shape),
                "{}: {} is {:?}, expected {:?}",
                path.display(),
                tensor.name,
                other.shape,
                tensor.shape
            );
            if other.dtype == DType::Q8_0 {
                // 34-byte blocks of 32 values along the rows.
                assert_eq!(other.bytes, other.elements() / 32 * 34, "{}", tensor.name);
            } else {
                assert_eq!(
                    other.bytes,
                    other.elements() * other.dtype.size(),
                    "{}",
                    tensor.name
                );
            }
        }
        assert!(files.vae("decoder.model.0.weight_v").is_some());
        let quantized = files
            .model_tensors()
            .iter()
            .filter(|tensor| tensor.dtype == DType::Q8_0)
            .count();
        let q8 = path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().contains("q8_0"));
        assert_eq!(quantized > 0, q8, "{}: Q8_0 matrices", path.display());
    }
}
