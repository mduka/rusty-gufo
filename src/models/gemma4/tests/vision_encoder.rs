//! The extension's `vision_encoder_test.cpp`: the HIP vision encoder against
//! the scalar reference on a small deterministic image, after an explicit
//! upload that the encodes reuse.
//!
//! `synthetic_sidecar` runs everywhere with a gfx1151 (`--features gpu`): a
//! sidecar of the real shape and tensor names with random BF16 weights,
//! written once to Cargo's target temp directory (about 1.2 GB). It checks
//! the encoder's host code, not the trained weights' numerics.
//! `released_sidecar` reads the sidecar named by `GUFO_GEMMA4_MMPROJ` (31B
//! or 26B-A4B) and is ignored by default, since that file is optional:
//! `cargo test --profile gpu-test -p gufo-gemma4 --features gpu,models
//! --test vision_encoder -- --ignored`.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use gufo_core::gguf::Gguf;
use gufo_core::image::Image;
use gufo_gemma4::vision_encoder::VisionEncoder;
use gufo_gemma4::vision_reference::VisionReference;
use gufo_gemma4::vision_weights::{FFN, HEAD_DIM, HIDDEN, LAYERS, POSITIONS, VisionWeights};
use gufo_hip::{MappedFile, Stream};

/// 144 x 96: 54 patches, 6 soft tokens; the C++ test's texture.
fn test_image() -> Image {
    let (width, height) = (144u32, 96u32);
    let pixels = (0..width as usize * height as usize * 3)
        .map(|i| ((i as u64).wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect();
    Image {
        width,
        height,
        pixels,
    }
}

fn relative_rms(data: &[f32], reference: &[f32]) -> f64 {
    let (mut err, mut norm) = (0.0f64, 0.0f64);
    for (&x, &r) in data.iter().zip(reference) {
        err += (f64::from(x) - f64::from(r)).powi(2);
        norm += f64::from(r).powi(2);
    }
    (err / norm).sqrt()
}

/// Encodes `test_image` with the reference and the GPU encoder and compares
/// every stage within the C++ test's limits.
fn check_encoder(path: &Path, width: u32) {
    let file = MappedFile::open(path).expect("sidecar");
    let weights = {
        let gguf = Gguf::parse(file.bytes()).expect("GGUF");
        VisionWeights::resolve(&gguf, width).expect("vision weights")
    };
    let image = test_image();
    let mut expected = BTreeMap::new();
    let mut keep = |stage: &str, data: &[f32]| {
        expected.insert(stage.to_string(), data.to_vec());
    };
    let reference = VisionReference::new(&weights, file.bytes())
        .encode(&image, Some(&mut keep))
        .expect("reference");

    let encoder = VisionEncoder::load(path, width).expect("encoder");
    assert_eq!(encoder.resident_bytes(), 0, "opening the sidecar uploaded");
    assert_eq!(encoder.identity().len(), 64);
    encoder.upload().expect("upload");
    let resident = encoder.resident_bytes();
    assert!(resident > 0, "upload did not allocate the weights");
    encoder.upload().expect("second upload");
    assert_eq!(
        encoder.resident_bytes(),
        resident,
        "a second upload allocated the weights again"
    );

    let mut seen = 0;
    let mut compare = |stage: &str, data: &[f32]| {
        let reference = expected
            .get(stage)
            .unwrap_or_else(|| panic!("{stage}: no reference stage"));
        assert_eq!(reference.len(), data.len(), "{stage}: size");
        let relative = relative_rms(data, reference);
        // FP32 patch embedding; binary16 GEMM inputs afterwards.
        let limit = if stage == "patch" { 1e-5 } else { 1e-2 };
        println!("{stage} rel rms {relative:.3e}");
        assert!(relative < limit, "{stage} diverges: {relative:e}");
        seen += 1;
    };
    let embedding = encoder
        .encode_observed(&image, Some(&mut compare), None)
        .expect("encode");
    assert_eq!(seen, expected.len(), "encoder skipped stages");
    assert_eq!(encoder.resident_bytes(), resident, "encode uploaded again");
    assert_eq!(embedding.len(), 6 * width as usize, "embedding shape");
    assert_eq!(reference.len(), embedding.len());

    // Encoding is deterministic, and a cancelled request stops.
    let again = encoder.encode(&image).expect("second encode");
    let stream = Stream::new().expect("stream");
    let mut a = vec![0.0f32; embedding.len()];
    let mut b = vec![0.0f32; again.len()];
    stream.download(embedding.view(), &mut a).expect("download");
    stream.download(again.view(), &mut b).expect("download");
    assert!(
        a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
        "encodes of one image differ"
    );
    let cancelled = || true;
    assert!(
        encoder
            .encode_observed(&image, None, Some(&cancelled))
            .is_err()
    );
    let unresized = Image {
        width: 100,
        height: 96,
        pixels: vec![0; 100 * 96 * 3],
    };
    assert!(encoder.encode(&unresized).is_err());
}

// ---------------------------------------------------------------------------
// A synthetic sidecar of the real shape.
// ---------------------------------------------------------------------------

/// The 26B-A4B's text width.
const SYNTHETIC_WIDTH: u32 = 2816;
const SYNTHETIC_NAME: &str = "gemma4-vision-synthetic-v1-2816.gguf";

#[derive(Clone, Copy)]
enum Fill {
    /// Uniform in `[low, high)`.
    Uniform(f32, f32),
}

struct Spec {
    name: String,
    dims: Vec<u64>,
    bf16: bool,
    fill: Fill,
}

impl Spec {
    fn elements(&self) -> u64 {
        self.dims.iter().product()
    }

    fn bytes(&self) -> u64 {
        self.elements() * if self.bf16 { 2 } else { 4 }
    }
}

fn specs() -> Vec<Spec> {
    let (h, f, d) = (u64::from(HIDDEN), u64::from(FFN), u64::from(HEAD_DIM));
    let uniform = |a: f32| Fill::Uniform(-a, a);
    let scale = |k: u64| (3.0 / k as f32).sqrt();
    let norm = Fill::Uniform(0.5, 1.5);
    let mut specs = vec![
        Spec {
            name: "v.patch_embd.weight".into(),
            dims: vec![16, 16, 3, h],
            bf16: false,
            fill: uniform(scale(768)),
        },
        Spec {
            name: "v.position_embd.weight".into(),
            dims: vec![h, u64::from(POSITIONS), 2],
            bf16: false,
            fill: uniform(0.5),
        },
        Spec {
            name: "v.std_bias".into(),
            dims: vec![h],
            bf16: false,
            fill: uniform(1.0),
        },
        Spec {
            name: "v.std_scale".into(),
            dims: vec![h],
            bf16: false,
            fill: norm,
        },
        Spec {
            name: "mm.input_projection.weight".into(),
            dims: vec![h, u64::from(SYNTHETIC_WIDTH)],
            bf16: true,
            fill: uniform(scale(h)),
        },
    ];
    for l in 0..LAYERS {
        let p = format!("v.blk.{l}.");
        let mut push = |name: &str, dims: Vec<u64>, bf16: bool, fill: Fill| {
            specs.push(Spec {
                name: format!("{p}{name}"),
                dims,
                bf16,
                fill,
            });
        };
        push("ln1.weight", vec![h], false, norm);
        for name in ["attn_q.weight", "attn_k.weight", "attn_v.weight"] {
            push(name, vec![h, h], true, uniform(scale(h)));
        }
        // Small Q and K norms keep the scale-1 attention from saturating.
        push(
            "attn_q_norm.weight",
            vec![d],
            false,
            Fill::Uniform(0.3, 0.5),
        );
        push(
            "attn_k_norm.weight",
            vec![d],
            false,
            Fill::Uniform(0.3, 0.5),
        );
        push("attn_out.weight", vec![h, h], true, uniform(scale(h)));
        push("attn_post_norm.weight", vec![h], false, norm);
        push("ln2.weight", vec![h], false, norm);
        push("ffn_gate.weight", vec![h, f], true, uniform(scale(h)));
        push("ffn_up.weight", vec![h, f], true, uniform(scale(h)));
        push("ffn_down.weight", vec![f, h], true, uniform(scale(f)));
        push("ffn_post_norm.weight", vec![h], false, norm);
    }
    specs
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn text(out: &mut impl Write, s: &str) {
    out.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
    out.write_all(s.as_bytes()).unwrap();
}

fn kv_u32(out: &mut impl Write, key: &str, value: u32) {
    text(out, key);
    out.write_all(&4u32.to_le_bytes()).unwrap();
    out.write_all(&value.to_le_bytes()).unwrap();
}

/// A file writer that counts its bytes for the GGUF alignment.
struct Counted {
    out: BufWriter<std::fs::File>,
    written: u64,
}

impl Counted {
    fn put(&mut self, bytes: &[u8]) {
        self.out.write_all(bytes).unwrap();
        self.written += bytes.len() as u64;
    }

    /// Pads to the next multiple of 32 bytes.
    fn align(&mut self) {
        let pad = (self.written.div_ceil(32) * 32 - self.written) as usize;
        self.put(&[0u8; 32][..pad]);
    }
}

/// Writes the sidecar as a GGUF v3 file (alignment 32), streaming each
/// tensor's random values.
fn write_sidecar(path: &Path) {
    let specs = specs();
    let file = std::fs::File::create(path).expect("create sidecar");
    let mut out = Counted {
        out: BufWriter::with_capacity(16 << 20, file),
        written: 0,
    };
    let mut header = Vec::new();
    header.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(specs.len() as u64).to_le_bytes());
    header.extend_from_slice(&14u64.to_le_bytes());
    {
        let h = &mut header;
        let str_kv = |h: &mut Vec<u8>, key: &str, value: &str| {
            text(h, key);
            h.extend_from_slice(&8u32.to_le_bytes());
            text(h, value);
        };
        str_kv(h, "general.architecture", "clip");
        str_kv(h, "general.type", "mmproj");
        text(h, "clip.has_vision_encoder");
        h.extend_from_slice(&7u32.to_le_bytes());
        h.push(1);
        str_kv(h, "clip.vision.projector_type", "gemma4v");
        kv_u32(h, "clip.vision.projection_dim", SYNTHETIC_WIDTH);
        kv_u32(h, "clip.vision.patch_size", 16);
        kv_u32(h, "clip.vision.embedding_length", HIDDEN);
        kv_u32(h, "clip.vision.feed_forward_length", FFN);
        kv_u32(h, "clip.vision.block_count", LAYERS);
        kv_u32(h, "clip.vision.attention.head_count", 16);
        text(h, "clip.vision.attention.layer_norm_epsilon");
        h.extend_from_slice(&6u32.to_le_bytes());
        h.extend_from_slice(&1e-6f32.to_le_bytes());
        for (key, value) in [
            ("clip.vision.image_mean", 0.0f32),
            ("clip.vision.image_std", 1.0),
        ] {
            text(h, key);
            h.extend_from_slice(&9u32.to_le_bytes());
            h.extend_from_slice(&6u32.to_le_bytes());
            h.extend_from_slice(&3u64.to_le_bytes());
            for _ in 0..3 {
                h.extend_from_slice(&value.to_le_bytes());
            }
        }
        kv_u32(h, "clip.vision.projector.scale_factor", 3);
        let mut offset = 0u64;
        for spec in &specs {
            text(h, &spec.name);
            h.extend_from_slice(&(spec.dims.len() as u32).to_le_bytes());
            for d in &spec.dims {
                h.extend_from_slice(&d.to_le_bytes());
            }
            // GGML F32 is 0, BF16 is 30.
            h.extend_from_slice(&(if spec.bf16 { 30u32 } else { 0 }).to_le_bytes());
            h.extend_from_slice(&offset.to_le_bytes());
            offset += spec.bytes().div_ceil(32) * 32;
        }
    }
    out.put(&header);
    out.align();
    let mut chunk = Vec::with_capacity(1 << 20);
    for (index, spec) in specs.iter().enumerate() {
        let mut state = 0x6765_6d6d_6134_7600 ^ index as u64;
        let Fill::Uniform(low, high) = spec.fill;
        let mut left = spec.elements();
        while left > 0 {
            chunk.clear();
            let take = left.min(1 << 18);
            for _ in 0..take {
                let unit = (splitmix(&mut state) >> 40) as f32 / (1u64 << 24) as f32;
                let value = low + (high - low) * unit;
                if spec.bf16 {
                    chunk.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
                } else {
                    chunk.extend_from_slice(&value.to_le_bytes());
                }
            }
            out.put(&chunk);
            left -= take;
        }
        out.align();
    }
    out.out.flush().unwrap();
}

/// The synthetic sidecar, written on first use and reused after.
fn synthetic_sidecar_path() -> PathBuf {
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let path = directory.join(SYNTHETIC_NAME);
    let usable = path.is_file()
        && MappedFile::open(&path).is_ok_and(|file| {
            Gguf::parse(file.bytes())
                .ok()
                .is_some_and(|gguf| VisionWeights::resolve(&gguf, SYNTHETIC_WIDTH).is_ok())
        });
    if !usable {
        let partial = directory.join(format!("{SYNTHETIC_NAME}.{}", std::process::id()));
        write_sidecar(&partial);
        std::fs::rename(&partial, &path).expect("publish sidecar");
    }
    path
}

#[test]
fn synthetic_sidecar() {
    check_encoder(&synthetic_sidecar_path(), SYNTHETIC_WIDTH);
}

#[test]
#[ignore = "needs GUFO_GEMMA4_MMPROJ, the BF16 vision sidecar"]
fn released_sidecar() {
    let path = std::env::var_os("GUFO_GEMMA4_MMPROJ")
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .expect("GUFO_GEMMA4_MMPROJ must name the Gemma 4 vision sidecar");
    let file = MappedFile::open(&path).expect("sidecar");
    // The projection width of the target the sidecar belongs to: 5376 for
    // the 31B, 2816 for the 26B-A4B.
    let width = Gguf::parse(file.bytes())
        .expect("GGUF")
        .u32("clip.vision.projection_dim")
        .expect("projection_dim");
    drop(file);
    check_encoder(&path, width);
}
