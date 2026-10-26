//! Uploads the local artifact with its MTP sidecar and compares samples
//! with the files: weights byte for byte, zero tails, the F16 stacks and the
//! split MTP projection:
//! cargo test -p gufo-qwen38-flash-next --features gpu --test upload

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::Arc;

use common::{mtp_path, trunk_path};
use gufo_formats::GgmlType;
use gufo_hip::{F16, MappedFile, QuantSlice, Stream};
use gufo_qwen38_flash_next::device::{Block, DeviceModel, TAIL};
use gufo_qwen38_flash_next::weights::{GgufFiles, MixerKind, ModelTensors, MtpTensors, TensorRef};

fn file_bytes(paths: &[PathBuf], t: &TensorRef, range: std::ops::Range<usize>) -> Vec<u8> {
    let file = std::fs::File::open(&paths[t.shard]).unwrap();
    let mut bytes = vec![0u8; range.len()];
    file.read_exact_at(&mut bytes, (t.offset + range.start) as u64)
        .unwrap();
    bytes
}

fn device_bytes(
    stream: &Stream,
    weights: QuantSlice<'_>,
    range: std::ops::Range<usize>,
) -> Vec<u8> {
    let mut bytes = vec![0u8; range.len()];
    stream
        .download(weights.bytes().range(range), &mut bytes)
        .unwrap();
    bytes
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let mant = f32::from(h & 0x3ff);
    match exp {
        0 => sign * mant * 2f32.powi(-24),
        31 => f32::NAN,
        _ => sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15),
    }
}

#[test]
fn the_artifact_uploads_with_its_mtp_block() {
    if !trunk_path().exists() || !mtp_path().exists() {
        eprintln!("skipped: the Flash-Next artifacts are not present");
        return;
    }
    let files = GgufFiles::open(&trunk_path()).unwrap();
    let model = ModelTensors::bind(&files.parse().unwrap()).unwrap();
    let mtp_files = GgufFiles::open(&mtp_path()).unwrap();
    let mtp = MtpTensors::bind(&mtp_files.parse().unwrap(), &model.config).unwrap();
    let open = |paths: &[PathBuf]| -> Vec<Arc<MappedFile>> {
        paths.iter().map(|p| MappedFile::open(p).unwrap()).collect()
    };
    let (mapped, mtp_mapped) = (open(&files.paths), open(&mtp_files.paths));
    let stream = Stream::new().unwrap();
    let started = std::time::Instant::now();
    let device = DeviceModel::upload(&stream, &model, &mapped, Some((&mtp, &mtp_mapped))).unwrap();
    eprintln!(
        "uploaded {:.1} GiB in {:.1} s",
        device.resident_bytes as f64 / f64::from(1 << 30),
        started.elapsed().as_secs_f64()
    );
    assert_eq!(device.layers.len(), 48);
    assert_eq!(device.max_half_cols, 2560);
    assert_eq!(device.max_q8_cols, 10_240);

    // A dense matrix, the embedding table and a routed expert stack: first
    // and last bytes as in the file, then a zero tail.
    let layer = &model.layers[3];
    let MixerKind::Attention(attention) = &layer.mixer else {
        panic!("layer 3 is an attention layer");
    };
    let Block::Attention(device_attention) = &device.layers[3].block else {
        panic!("layer 3 is an attention layer");
    };
    for (t, weights) in [
        (&attention.out, device_attention.out.view()),
        (&model.token_embd, device.token_embd.view()),
        (&layer.moe.down_exps, device.layers[3].moe.down_exps.view()),
    ] {
        assert_eq!(weights.format(), t.ty);
        assert_eq!(weights.len(), t.bytes + TAIL);
        assert_eq!(
            device_bytes(&stream, weights, 0..4096),
            file_bytes(&files.paths, t, 0..4096)
        );
        let end = t.bytes - 4096..t.bytes;
        assert_eq!(
            device_bytes(&stream, weights, end.clone()),
            file_bytes(&files.paths, t, end)
        );
        assert!(
            device_bytes(&stream, weights, t.bytes..t.bytes + TAIL)
                .iter()
                .all(|&b| b == 0)
        );
    }
    // The stacked [q|gate ; k ; v]: v's first row right after q and k.
    let qkv = device_attention.qkv.as_ref().unwrap();
    assert_eq!(qkv.rows, 12_288 + 512 + 512);
    let v_start = attention.q.bytes + attention.k.bytes;
    assert_eq!(
        device_bytes(&stream, qkv.view(), v_start..v_start + 1024),
        file_bytes(&files.paths, &attention.v, 0..1024)
    );
    // The router, narrowed to F16, and the shared-expert gate row after it.
    let router = &device.layers[3].moe.router;
    assert_eq!(
        (router.format(), router.rows, router.cols),
        (GgmlType::F16, 513, 2560)
    );
    let halves = device_bytes(&stream, router.view(), 0..router.view().len());
    let as_f16 = |i: usize| f16_to_f32(u16::from_le_bytes([halves[2 * i], halves[2 * i + 1]]));
    let floats = |t: &TensorRef, n: usize| -> Vec<f32> {
        file_bytes(&files.paths, t, 0..4 * n)
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    for (i, x) in floats(&layer.moe.router, 64).into_iter().enumerate() {
        assert!(
            (as_f16(i) - x).abs() <= x.abs() / 1024.0 + 1e-7,
            "router[{i}]"
        );
    }
    let gate = floats(&layer.moe.shexp_gate_inp, 8);
    for (i, x) in gate.into_iter().enumerate() {
        assert!(
            (as_f16(512 * 2560 + i) - x).abs() <= x.abs() / 1024.0 + 1e-7,
            "shexp gate[{i}]"
        );
    }
    let _ = F16::default();

    // The MTP input projection: each row's halves in their own matrix.
    let nextn = device.mtp.as_ref().unwrap().nextn.as_ref().unwrap();
    let eh = &mtp.block.nextn.as_ref().unwrap().eh_proj;
    let half = eh.row_bytes() / 2;
    assert_eq!(
        (nextn.fc_embedding.cols, nextn.fc_hidden.cols),
        (2560, 2560)
    );
    for row in [0usize, 1, 2559] {
        let source = file_bytes(&mtp_files.paths, eh, row * 2 * half..(row + 1) * 2 * half);
        let embedding = device_bytes(
            &stream,
            nextn.fc_embedding.view(),
            row * half..(row + 1) * half,
        );
        let hidden = device_bytes(
            &stream,
            nextn.fc_hidden.view(),
            row * half..(row + 1) * half,
        );
        assert_eq!(embedding, source[..half]);
        assert_eq!(hidden, source[half..]);
    }
}
