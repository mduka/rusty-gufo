//! Image input of Qwen3.6 35B-A3B (`qwen35moe`, projector width 2048)
//! against its C++ reference, the qwen35moe Gufo extension with its vision
//! gate admitting width 2048 (`fixtures/qwen3.6-35b-a3b-vision.json`, whose
//! `source` names the recording). The extension's encoder and prompt
//! preparation are upstream's `src/models/qwen/vision` unchanged apart from
//! that gate, so the 35B uses this crate's `Encoder` and `vision::prepare`
//! as they are. The fixture comes from the reference run on ROCm 10.1's
//! BLAS libraries; on other hipBLAS/hipBLASLt releases the larger images'
//! encodings may differ.
//!
//! The synthetic 200x150 image (the fixture set's `images/make_synthetic.py`)
//! is generated here. Its decoding and resizing need nothing else, its
//! prompt needs the target's tokenizer (`models`: `GUFO_QWEN35MOE_MODEL`, the
//! UD-Q4_K_M GGUF, of which only the header is read) and its encoding needs
//! the sidecar (`gpu`: `GUFO_QWEN35MOE_MMPROJ`, the 35B's
//! `mmproj-BF16.gguf`). The two photos and the synthetic image's embedding
//! row files stay in the reference fixture set (`GUFO_QWEN35MOE_FIXTURES`,
//! default `target/qwen35moe`).
//!
//!   cargo test -p gufo-qwen --test vision_qwen35moe
//!   cargo test -p gufo-qwen --features models --test vision_qwen35moe
//!   cargo test --profile gpu-test -p gufo-qwen --features gpu --test vision_qwen35moe

#![allow(clippy::unwrap_used)]

#[cfg(any(feature = "models", feature = "gpu"))]
use std::path::PathBuf;

use gufo_core::image::{Image, decode_image};
use gufo_core::text::ImageDetail;
use gufo_qwen::vision::{RESIZE_FACTOR, resize_image};
use serde_json::Value;
use sha2::{Digest, Sha256};

const FIXTURE: &str = include_str!("fixtures/qwen3.6-35b-a3b-vision.json");
const SYNTHETIC: &str = "synthetic-200x150";

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).unwrap()
}

fn image_entry<'a>(fixture: &'a Value, name: &str) -> &'a Value {
    fixture["images"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["image"] == name)
        .unwrap_or_else(|| panic!("no image {name} in the fixture"))
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn u32_of(value: &Value) -> u32 {
    u32::try_from(value.as_u64().unwrap()).unwrap()
}

fn str_of(value: &Value) -> &str {
    value.as_str().unwrap()
}

/// The reference fixture set: the photos, the full vision fixtures and the
/// embedding row files.
#[cfg(any(feature = "models", feature = "gpu"))]
fn reference_set() -> PathBuf {
    std::env::var_os("GUFO_QWEN35MOE_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../target/qwen35moe"),
        PathBuf::from,
    )
}

/// One pixel of `make_synthetic.py`: sky gradient, sun, a red house with a
/// door and a roof, striped ground.
fn synthetic_pixel(x: u32, y: u32) -> [u8; 3] {
    let (dx, dy) = (i64::from(x) - 150, i64::from(y) - 35);
    if dx * dx + dy * dy <= 18 * 18 {
        return [250, 210, 40];
    }
    if (60..120).contains(&x) && (70..115).contains(&y) {
        if (82..98).contains(&x) && (90..115).contains(&y) {
            return [90, 50, 20];
        }
        return [200, 40, 40];
    }
    if y >= 115 {
        return [40, (140 + 30 * ((x / 8 + y / 8) % 2)) as u8, 50];
    }
    // Python compares the integer with the float product exactly.
    if (50..70).contains(&y) && (f64::from(x) - 90.0).abs() <= f64::from(y - 50) * 1.6 {
        return [120, 60, 30];
    }
    [
        (100 + y / 2) as u8,
        (150 + y / 3) as u8,
        (255 - y / 4) as u8,
    ]
}

/// The synthetic image as an 8-bit RGB PNG. Its bytes differ from the
/// reference file (another deflate), its pixels do not.
fn synthetic_png() -> Vec<u8> {
    let (width, height) = (200, 150);
    let pixels: Vec<u8> = (0..height)
        .flat_map(|y| (0..width).flat_map(move |x| synthetic_pixel(x, y)))
        .collect();
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&pixels).unwrap();
    writer.finish().unwrap();
    bytes
}

/// A fixture image's PNG: the synthetic one generated, a photo read from the
/// reference set and checked against its recorded hash.
#[cfg(any(feature = "models", feature = "gpu"))]
fn png_bytes(fixture: &Value, name: &str) -> Vec<u8> {
    if name == SYNTHETIC {
        return synthetic_png();
    }
    let path = reference_set().join("images").join(format!("{name}.png"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}; set GUFO_QWEN35MOE_FIXTURES to the qwen35moe reference fixtures",
            path.display()
        )
    });
    assert_eq!(
        sha256(&bytes),
        str_of(&image_entry(fixture, name)["file_sha256"]),
        "{name}: not the reference's file"
    );
    bytes
}

/// Decodes and resizes a fixture image (`core::DecodeImage`, `ResizeImage`)
/// and checks both against the reference's pixels and grid.
fn check_pixels(fixture: &Value, name: &str, bytes: &[u8]) -> Image {
    let entry = image_entry(fixture, name);
    let decoded = decode_image(bytes).unwrap();
    let expected = &entry["decoded"];
    assert_eq!(
        (decoded.width, decoded.height),
        (u32_of(&expected["width"]), u32_of(&expected["height"])),
        "{name}: decoded size"
    );
    assert_eq!(
        sha256(&decoded.pixels),
        str_of(&expected["rgb8_sha256"]),
        "{name}: decoded pixels"
    );
    let resized = resize_image(&decoded, ImageDetail::Auto).unwrap();
    let expected = &entry["resized"];
    assert_eq!(
        (resized.width, resized.height),
        (u32_of(&expected["width"]), u32_of(&expected["height"])),
        "{name}: resized size"
    );
    assert_eq!(
        sha256(&resized.pixels),
        str_of(&expected["rgb8_sha256"]),
        "{name}: resized pixels"
    );
    assert_eq!(
        (
            resized.height / RESIZE_FACTOR,
            resized.width / RESIZE_FACTOR
        ),
        (u32_of(&entry["grid_height"]), u32_of(&entry["grid_width"])),
        "{name}: grid"
    );
    resized
}

#[test]
fn the_synthetic_image_decodes_and_resizes_as_the_reference() {
    let fixture = fixture();
    check_pixels(&fixture, SYNTHETIC, &synthetic_png());
}

#[cfg(feature = "models")]
mod prompts {
    use std::sync::Arc;

    use gufo_core::gguf::{Gguf, read_header};
    use gufo_core::qwen_tokenizer::QwenTokenizer;
    use gufo_core::text::{ChatMessage, ChatRole, ImagePart, ReasoningOptions};
    use gufo_qwen::chat_template::{render, resolve_options};
    use gufo_qwen::vision::{self, IMAGE_TOKEN};

    use super::*;

    fn tokenizer() -> QwenTokenizer {
        // Unset, the path is the variable itself, so the failure says what to set.
        let path = std::env::var_os("GUFO_QWEN35MOE_MODEL")
            .map_or_else(|| "$GUFO_QWEN35MOE_MODEL".into(), PathBuf::from);
        let (header, len) =
            read_header(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let gguf = Gguf::parse_prefix(&header, len).unwrap();
        assert_eq!(gguf.string("general.architecture"), Some(&b"qwen35moe"[..]));
        assert_eq!(gguf.u64("qwen35moe.embedding_length"), Some(2048));
        QwenTokenizer::from_gguf(&gguf).unwrap()
    }

    /// `vision::Prepare` of a fixture case as `gufo prompt --image ...
    /// --think off` calls it (`drivers.cpp` `PrepareVision`): one user
    /// message with every image at offset 0, thinking off, the CLI context.
    /// The encoder identity enters only the cache identity.
    fn check_case(tokenizer: &QwenTokenizer, fixture: &Value, case: &Value) {
        let name = str_of(&case["name"]);
        let mut message = ChatMessage::new(ChatRole::User, str_of(&case["user_text"]));
        let mut resized = Vec::new();
        for image in case["images"].as_array().unwrap() {
            let image = str_of(image);
            let bytes = png_bytes(fixture, image);
            resized.push(check_pixels(fixture, image, &bytes));
            message.images.push(ImagePart {
                offset: 0,
                bytes: Arc::new(bytes),
                detail: ImageDetail::Auto,
            });
        }
        let messages = [message];
        let reasoning = ReasoningOptions {
            enabled: Some(false),
            ..ReasoningOptions::default()
        };
        let options = resolve_options(&reasoning, false);
        let rendered = render(&messages, &[], &options).unwrap();
        assert_eq!(rendered.text, str_of(&case["rendered"]), "{name}: rendered");
        let max_context = u32_of(&fixture["max_context"]);
        let prompt = vision::prepare(
            tokenizer,
            &messages,
            &[],
            &options,
            "mmproj",
            max_context,
            None,
        )
        .unwrap();

        let tokens = &prompt.tokens;
        assert_eq!(
            tokens.len(),
            case["prompt_len"].as_u64().unwrap() as usize,
            "{name}"
        );
        let grids = case["image_grids"].as_array().unwrap();
        assert_eq!(prompt.rope.images.len(), grids.len(), "{name}");
        let mut text = Vec::new();
        for (index, &token) in tokens.iter().enumerate() {
            let grid = prompt.rope.images.iter().find(|g| {
                let index = index as u32;
                g.offset <= index && index < g.offset + g.height * g.width
            });
            match grid {
                Some(_) => assert_eq!(token, IMAGE_TOKEN, "{name}: token {index}"),
                None => text.push(token),
            }
        }
        let expected_text: Vec<u32> = case["text_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(u32_of)
            .collect();
        assert_eq!(text, expected_text, "{name}: ids outside the image grids");
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        assert_eq!(
            sha256(&bytes),
            str_of(&case["prompt_ids_le_u32_sha256"]),
            "{name}: prompt ids"
        );
        for (index, (grid, expected)) in prompt.rope.images.iter().zip(grids).enumerate() {
            assert_eq!(
                (grid.offset, grid.height, grid.width),
                (
                    u32_of(&expected["offset"]),
                    u32_of(&expected["height"]),
                    u32_of(&expected["width"])
                ),
                "{name}: grid {index}"
            );
            assert!(
                *prompt.images[index].pixels == resized[index],
                "{name}: prepared pixels of image {index}"
            );
        }
        assert_eq!(
            i64::from(prompt.rope.delta()),
            case["rope_delta"].as_i64().unwrap(),
            "{name}: rope_delta"
        );
        assert_eq!(
            prompt.stable_prefix_tokens,
            case["stable_prefix_tokens"].as_u64().unwrap() as usize,
            "{name}: stable prefix"
        );
    }

    fn case<'a>(fixture: &'a Value, name: &str) -> &'a Value {
        fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == name)
            .unwrap()
    }

    #[test]
    fn the_synthetic_prompt_matches_the_reference() {
        let fixture = fixture();
        check_case(&tokenizer(), &fixture, case(&fixture, SYNTHETIC));
    }

    /// The photos and the pair (synthetic image first, then the bison).
    #[test]
    fn the_photo_prompts_match_the_reference() {
        let fixture = fixture();
        let tokenizer = tokenizer();
        let mut checked = 0;
        for case in fixture["cases"].as_array().unwrap() {
            if case["name"] != SYNTHETIC {
                check_case(&tokenizer, &fixture, case);
                checked += 1;
            }
        }
        assert_eq!(checked, 3);
    }
}

#[cfg(feature = "gpu")]
mod encoder {
    use std::sync::OnceLock;

    use gufo_hip::Stream;
    use gufo_qwen::vision::{Encoder, Observer};

    use super::*;

    /// One encoder for every test, as the reference encodes every image
    /// with one.
    fn encoder() -> &'static Encoder {
        static ENCODER: OnceLock<Encoder> = OnceLock::new();
        ENCODER.get_or_init(|| {
            // Unset, the path is the variable itself, so the failure says what to set.
            let path = std::env::var_os("GUFO_QWEN35MOE_MMPROJ")
                .map_or_else(|| "$GUFO_QWEN35MOE_MMPROJ".into(), PathBuf::from);
            let encoder = Encoder::new(&path, 2048)
                .unwrap_or_else(|e| panic!("{}: {}", path.display(), e.message()));
            assert_eq!(encoder.output_width(), 2048);
            encoder
        })
    }

    fn fnv1a64(values: &[f32]) -> String {
        let hash = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
            });
        format!("{hash:016x}")
    }

    fn le_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// The embedding of an encode without observer.
    fn encode_plain(image: &Image) -> Vec<f32> {
        let embedding = encoder().encode(image, None, None).unwrap();
        let stream = Stream::new().unwrap();
        let mut host = vec![0f32; embedding.data.len()];
        stream.download(embedding.data.view(), &mut host).unwrap();
        host
    }

    /// Encodes `resized` with an observer, as the reference's
    /// `EncodeObserved`, and checks every stage digest (`patch`, `layer0`
    /// to `layer26`, `embedding`), the per-row digests `rows` and the row
    /// heads, then that an encode without observer gives the same bits.
    fn check_encoding(entry: &Value, resized: &Image, rows: &[Value]) -> Vec<f32> {
        let name = str_of(&entry["image"]);
        let mut stages = Vec::new();
        let mut embedding = Vec::new();
        let mut observe = |stage: &str, values: &[f32]| {
            stages.push((stage.to_string(), fnv1a64(values)));
            if stage == "embedding" {
                embedding = values.to_vec();
            }
        };
        let observer: Observer<'_> = &mut observe;
        let encoded = encoder().encode(resized, Some(observer), None).unwrap();
        assert_eq!(
            (encoded.rows, encoded.width),
            (
                u32_of(&entry["embedding_rows"]),
                u32_of(&entry["embedding_width"])
            ),
            "{name}: embedding shape"
        );
        let expected: Vec<(String, String)> = entry["stages_fnv1a64"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| (str_of(&pair[0]).to_string(), str_of(&pair[1]).to_string()))
            .collect();
        assert_eq!(expected.len(), 29);
        // The first differing stage tells where the encoders part.
        for (actual, expected) in stages.iter().zip(&expected) {
            assert_eq!(actual, expected, "{name}: stage");
        }
        assert_eq!(stages.len(), expected.len(), "{name}: stages");
        assert_eq!(
            fnv1a64(&embedding),
            str_of(&entry["embedding_fnv1a64"]),
            "{name}: embedding"
        );

        let width = encoded.width as usize;
        assert_eq!(rows.len(), encoded.rows as usize, "{name}: row digests");
        let differing: Vec<usize> = embedding
            .chunks(width)
            .zip(rows)
            .enumerate()
            .filter(|(_, (row, expected))| fnv1a64(row) != str_of(expected))
            .map(|(index, _)| index)
            .collect();
        assert!(differing.is_empty(), "{name}: rows {differing:?} differ");
        let head = |key: &str| -> Vec<u32> {
            entry[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| (v.as_f64().unwrap() as f32).to_bits())
                .collect()
        };
        let bits = |values: &[f32]| -> Vec<u32> { values.iter().map(|v| v.to_bits()).collect() };
        let last = (encoded.rows as usize - 1) * width;
        assert_eq!(bits(&embedding[..8]), head("embedding_row0_head"), "{name}");
        assert_eq!(
            bits(&embedding[last..last + 8]),
            head("embedding_last_row_head"),
            "{name}"
        );

        assert!(entry["encode_repeat_identical"].as_bool().unwrap());
        assert!(
            le_bytes(&encode_plain(resized)) == le_bytes(&embedding),
            "{name}: an encode without observer differs"
        );
        embedding
    }

    #[test]
    fn the_synthetic_image_encodes_as_the_reference() {
        let fixture = fixture();
        let entry = image_entry(&fixture, SYNTHETIC);
        let resized = check_pixels(&fixture, SYNTHETIC, &synthetic_png());
        let embedding = check_encoding(
            entry,
            &resized,
            entry["embedding_rows_fnv1a64"].as_array().unwrap(),
        );
        // The bytes of the reference's embedding row file.
        assert_eq!(
            sha256(&le_bytes(&embedding)),
            str_of(&entry["embedding_file_sha256"])
        );
    }

    /// The photos, and the synthetic image against the reference set's
    /// embedding row files.
    #[test]
    fn the_photos_encode_as_the_reference() {
        let fixture = fixture();
        for name in ["bison-1024x668", "aldrin-960x966"] {
            let entry = image_entry(&fixture, name);
            let resized = check_pixels(&fixture, name, &png_bytes(&fixture, name));
            check_encoding(
                entry,
                &resized,
                entry["embedding_rows_fnv1a64"].as_array().unwrap(),
            );
        }

        let entry = image_entry(&fixture, SYNTHETIC);
        let resized = check_pixels(&fixture, SYNTHETIC, &synthetic_png());
        let embedding = le_bytes(&encode_plain(&resized));
        for quant in ["ud-q4_k_m", "q8_0"] {
            let path = reference_set()
                .join("rows")
                .join(quant)
                .join(str_of(&entry["embedding_file"]));
            let expected = std::fs::read(&path).unwrap_or_else(|e| {
                panic!(
                    "{}: {e}; set GUFO_QWEN35MOE_FIXTURES to the qwen35moe reference fixtures",
                    path.display()
                )
            });
            assert!(expected == embedding, "{}", path.display());
        }
    }
}
