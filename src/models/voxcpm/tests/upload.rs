//! The generator's weights on the device against the files: every released
//! layout uploads, picks the formats `weights.cpp` picks (Q8_0 tiles for an
//! 8-bit file, bfloat16 tiles for the DiT, bfloat16 rows otherwise), and the
//! device copy of a sample of matrices, fused matrices and vectors equals the
//! file's values converted on the host. Each file named by
//! `GUFO_VOXCPM_MODEL` (the Hugging Face directory), `GUFO_VOXCPM_GGUF`
//! (`voxcpm2-bf16.gguf`) and `GUFO_VOXCPM_Q8_GGUF` (`voxcpm2-q8_0.gguf`) is
//! checked; at least one must be set.
//!
//! cargo test --profile gpu-test -p gufo-voxcpm --features gpu,models --test upload

#![cfg(all(feature = "gpu", feature = "models"))]
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use gufo_hip::{Bf16, Stream};
use gufo_voxcpm::config::parse_config;
use gufo_voxcpm::matrix::{MatrixView, WeightFormat};
use gufo_voxcpm::model_files::ModelFiles;
use gufo_voxcpm::safetensors::{DType, Tensor};
use gufo_voxcpm::weights::{DeviceWeights, Projection};
use gufo_voxcpm_kernels as k;

/// bfloat16 bits of `value`, rounded to nearest even (`Bf16Bits` without
/// NaNs, which the checkpoints don't hold).
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

fn half_to_f32(bits: u16) -> f32 {
    gufo_hip::F16(bits).to_f32()
}

/// The tensor's values as the upload's bfloat16: copied, converted from
/// float16 or float32, or dequantized from Q8_0 (`bf16(scale * q)`).
fn host_bf16(tensor: &Tensor) -> Vec<u16> {
    let data = tensor.data();
    match tensor.dtype {
        DType::Bf16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| u16::from_le_bytes(b))
            .collect(),
        DType::F16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| bf16_bits(half_to_f32(u16::from_le_bytes(b))))
            .collect(),
        DType::F32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| bf16_bits(f32::from_le_bytes(b)))
            .collect(),
        DType::Q8_0 => data
            .as_chunks::<34>()
            .0
            .iter()
            .flat_map(|block| {
                let scale = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
                block[2..]
                    .iter()
                    .map(move |&q| bf16_bits(scale * f32::from(q as i8)))
            })
            .collect(),
        other => panic!("{} has the unexpected type {}", tensor.name, other.name()),
    }
}

/// The device matrix `[out][in]` as bfloat16 rows, unpacked on the device.
fn device_rows(stream: &Stream, view: MatrixView<'_>, out: usize, input: usize) -> Vec<u16> {
    let mut rows = stream.alloc::<Bf16>(out * input).unwrap();
    match view.format {
        WeightFormat::Bf16 => {
            stream
                .copy(view.data.cast::<Bf16>().unwrap(), rows.as_mut_slice())
                .unwrap();
        }
        WeightFormat::Bf16Tiles => {
            k::bf16_tiles_to_rows(
                stream,
                view.data.cast::<Bf16>().unwrap(),
                out,
                input,
                rows.as_mut_slice(),
            )
            .unwrap();
        }
        WeightFormat::Q8Tiles => {
            k::q8_tiles_to_bf16(stream, view.data, out, input, rows.as_mut_slice()).unwrap();
        }
    }
    let mut host = vec![Bf16::default(); out * input];
    stream.download(rows.as_slice(), &mut host).unwrap();
    host.into_iter().map(|v| v.0).collect()
}

/// A projection's weight against the concatenated `names`.
fn check_matrix(
    stream: &Stream,
    files: &ModelFiles,
    weights: &DeviceWeights,
    projection: &Projection,
    names: &[&str],
    expected: WeightFormat,
) {
    assert_eq!(projection.weight.format, expected, "{}", names[0]);
    let mut host = Vec::new();
    for name in names {
        host.extend(host_bf16(files.model(name).unwrap()));
    }
    let device = device_rows(
        stream,
        weights.matrix(projection.weight),
        projection.output,
        projection.input,
    );
    assert_eq!(host.len(), device.len(), "{}", names[0]);
    let first = host.iter().zip(&device).position(|(a, b)| a != b);
    assert!(
        first.is_none(),
        "{}: first difference at {first:?}",
        names[0]
    );
}

fn check_file(path: PathBuf) {
    let files = ModelFiles::open(&path).unwrap();
    let config = parse_config(files.config()).unwrap();
    let stream = Stream::new().unwrap();
    let device = DeviceWeights::upload(&stream, &files, &config).unwrap();
    let w = device.weights();
    let q8 = files
        .model("base_lm.layers.0.self_attn.o_proj.weight")
        .unwrap()
        .dtype
        == DType::Q8_0;
    let rows_format = if q8 {
        WeightFormat::Q8Tiles
    } else {
        WeightFormat::Bf16
    };
    let dit_format = if q8 {
        WeightFormat::Q8Tiles
    } else {
        WeightFormat::Bf16Tiles
    };
    println!(
        "{}: {} bytes on the device, {}",
        path.display(),
        device.bytes(),
        if q8 { "Q8_0 tiles" } else { "bfloat16" }
    );
    let base = &w.base_lm.layers[0];
    check_matrix(
        &stream,
        &files,
        &device,
        &base.o,
        &["base_lm.layers.0.self_attn.o_proj.weight"],
        rows_format,
    );
    check_matrix(
        &stream,
        &files,
        &device,
        &base.gate_up,
        &[
            "base_lm.layers.0.mlp.gate_proj.weight",
            "base_lm.layers.0.mlp.up_proj.weight",
        ],
        rows_format,
    );
    let dit = &w.dit.layers[11];
    check_matrix(
        &stream,
        &files,
        &device,
        &dit.qkv,
        &[
            "feat_decoder.estimator.decoder.layers.11.self_attn.q_proj.weight",
            "feat_decoder.estimator.decoder.layers.11.self_attn.k_proj.weight",
            "feat_decoder.estimator.decoder.layers.11.self_attn.v_proj.weight",
        ],
        dit_format,
    );
    check_matrix(
        &stream,
        &files,
        &device,
        &dit.down,
        &["feat_decoder.estimator.decoder.layers.11.mlp.down_proj.weight"],
        dit_format,
    );
    // The stop head's two rows don't fill a tile row; its columns do.
    let stop_head = files.model("stop_head.weight").unwrap();
    let stop_format = if stop_head.dtype == DType::Q8_0 {
        WeightFormat::Q8Tiles
    } else {
        WeightFormat::Bf16
    };
    check_matrix(
        &stream,
        &files,
        &device,
        &w.stop_head,
        &["stop_head.weight"],
        stop_format,
    );
    // Vectors and the embedding table are bfloat16.
    for (vector, name) in [
        (w.embed, "base_lm.embed_tokens.weight"),
        (w.special, "feat_encoder.special_token"),
        (w.base_lm.norm, "base_lm.norm.weight"),
        (w.stop_proj.bias.unwrap(), "stop_proj.bias"),
    ] {
        let mut host = vec![Bf16::default(); vector.len];
        stream.download(device.vector(vector), &mut host).unwrap();
        let host: Vec<u16> = host.into_iter().map(|v| v.0).collect();
        assert!(host == host_bf16(files.model(name).unwrap()), "{name}");
    }
}

#[test]
fn every_layout_uploads_its_tensors_exactly() {
    let paths: Vec<PathBuf> = [
        "GUFO_VOXCPM_MODEL",
        "GUFO_VOXCPM_GGUF",
        "GUFO_VOXCPM_Q8_GGUF",
    ]
    .into_iter()
    .filter_map(std::env::var_os)
    .map(PathBuf::from)
    .collect();
    assert!(
        !paths.is_empty(),
        "set GUFO_VOXCPM_MODEL, GUFO_VOXCPM_GGUF or GUFO_VOXCPM_Q8_GGUF"
    );
    // One model on the device at a time.
    for path in paths {
        check_file(path);
    }
}
