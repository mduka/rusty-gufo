//! The vision path against dumps of the C++ one (`src/models/qwen/tests/reference/cpp_vision_dump.cpp`):
//! `GUFO_VISION_DUMP` names its output directory, holding `images.txt` with
//! the input paths in order. PNG inputs decode to the same pixels; every
//! decoded image (JPEGs from the C++ decoder's pixels, since their decoders
//! may differ) resizes to the same grid; failing inputs fail on both.
//!   cargo test -p gufo-qwen35 --test vision -- --ignored
//! With the `gpu` feature and a dump made with a sidecar, every encoder stage
//! (patch, the 27 blocks, the embedding) and the sidecar identity match.
//!
//! Without a dump, the `gpu` feature also runs an image request on a session
//! whose decode graph a text request captured; it needs the Qwen3.8 27B GGUF
//! (`GUFO_QWEN27B_MODEL`, else the pinned file in the Hugging Face cache) and
//! its `mmproj-BF16.gguf` (`GUFO_QWEN27B_MMPROJ`), and fails without them.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use gufo_core::image::{Image, decode_image};
use gufo_qwen::vision::resize_image;
#[cfg(feature = "gpu")]
use gufo_qwen::vision::{Encoder, Observer};

fn dump() -> PathBuf {
    PathBuf::from(std::env::var_os("GUFO_VISION_DUMP").expect("GUFO_VISION_DUMP"))
}

/// A dumped "W H\n" + RGB8 image.
fn read_image(path: &Path) -> Image {
    let data = std::fs::read(path).unwrap();
    let newline = data.iter().position(|&b| b == b'\n').unwrap();
    let header = std::str::from_utf8(&data[..newline]).unwrap();
    let (w, h) = header.split_once(' ').unwrap();
    Image::new(
        w.parse().unwrap(),
        h.parse().unwrap(),
        data[newline + 1..].to_vec(),
    )
    .unwrap()
}

#[test]
#[ignore = "needs a C++ dump (GUFO_VISION_DUMP)"]
fn decoded_and_resized_images_match_the_cpp_path() {
    let dir = dump();
    let images = std::fs::read_to_string(dir.join("images.txt")).unwrap();
    let mut compared = 0;
    for (index, path) in images.lines().enumerate() {
        let bytes = std::fs::read(path).unwrap();
        let png = bytes.starts_with(b"\x89PNG");
        let error = dir.join(format!("{index}.error"));
        if error.exists() {
            let decoded = decode_image(&bytes);
            let rejected = match &decoded {
                Err(_) => true,
                Ok(image) => resize_image(image).is_err(),
            };
            // A PNG's CRC and zlib checks agree with libpng; JPEG decoders
            // may differ in what they tolerate.
            assert!(!png || rejected, "{path}: C++ rejects it");
            continue;
        }
        let decoded = read_image(&dir.join(format!("{index}.rgb")));
        if png {
            assert!(
                decode_image(&bytes).unwrap() == decoded,
                "{path}: decoded pixels"
            );
        }
        let resized = resize_image(&decoded).unwrap();
        let expected = read_image(&dir.join(format!("{index}.resized")));
        assert_eq!(
            (resized.width, resized.height),
            (expected.width, expected.height),
            "{path}"
        );
        assert!(resized.pixels == expected.pixels, "{path}: resized pixels");
        compared += 1;
    }
    assert!(compared > 0);
    println!("{compared} images match");
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a C++ dump with a sidecar (GUFO_VISION_DUMP)"]
fn encoder_stages_match_the_cpp_encoder() {
    let dir = dump();
    let sidecar = std::fs::read_to_string(dir.join("sidecar.txt")).unwrap();
    let mut lines = sidecar.lines();
    let path = PathBuf::from(lines.next().unwrap());
    let width: u32 = lines.next().unwrap().parse().unwrap();
    let encoder = Encoder::new(&path, width).unwrap();
    let identity = std::fs::read_to_string(dir.join("identity.txt")).unwrap();
    assert_eq!(encoder.identity(), identity.trim());
    let images = std::fs::read_to_string(dir.join("images.txt")).unwrap();
    for index in 0..images.lines().count() {
        let resized = read_image(&dir.join(format!("{index}.resized")));
        let mut stages = Vec::new();
        let mut observe = |stage: &str, values: &[f32]| {
            let expected = std::fs::read(dir.join(format!("{index}.{stage}.f32"))).unwrap();
            let actual: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let differing = expected
                .chunks(4)
                .zip(actual.chunks(4))
                .filter(|(a, b)| a != b)
                .count();
            stages.push((stage.to_string(), expected.len() == actual.len(), differing));
        };
        let observer: Observer<'_> = &mut observe;
        let embedding = encoder.encode(&resized, Some(observer), None).unwrap();
        assert_eq!(
            embedding.rows,
            resized.width / 32 * (resized.height / 32),
            "image {index}"
        );
        assert_eq!(stages.len(), 29, "image {index}");
        for (stage, same_len, differing) in stages {
            assert!(
                same_len && differing == 0,
                "image {index} {stage}: {differing} values differ"
            );
        }
    }
    assert!(encoder.resident_bytes() > 0);
}

#[cfg(feature = "gpu")]
fn model() -> PathBuf {
    std::env::var_os("GUFO_QWEN27B_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(
                ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/\
                 4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf",
            )
        })
}

#[cfg(feature = "gpu")]
fn sidecar() -> PathBuf {
    // Unset, the path is the variable itself, so the failure says what to set.
    std::env::var_os("GUFO_QWEN27B_MMPROJ")
        .map_or_else(|| "$GUFO_QWEN27B_MMPROJ".into(), PathBuf::from)
}

/// A 64×64 red PNG.
#[cfg(feature = "gpu")]
fn red_png() -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, 64, 64);
    encoder.set_color(png::ColorType::Rgb);
    let mut writer = encoder.write_header().unwrap();
    writer
        .write_image_data(&[255, 0, 0].repeat(64 * 64))
        .unwrap();
    writer.finish().unwrap();
    bytes
}

/// A text request captures the session's decode graph before its first
/// image creates the mRoPE table; the image request then captures a graph of
/// its own instead of replaying the text one (which differs in the `rope`
/// argument), and decodes as on a fresh session.
#[cfg(feature = "gpu")]
#[test]
fn an_image_after_text_decodes_as_on_a_fresh_session() {
    use std::sync::Arc;

    use gufo_core::sampling::SamplingConfig;
    use gufo_core::text::{ChatMessage, ChatRequest, ChatRole, ImagePart};
    use gufo_qwen::vision::{ImageCache, prepare_request};
    use gufo_qwen35::Engine;

    let mut engine = Engine::load(&model(), 4096).unwrap();
    let encoder = Encoder::open(&model(), &sidecar(), 5120).unwrap().unwrap();
    let tokenizer = engine.weights().tokenizer.clone();
    let request = |text: &str, image: Option<Vec<u8>>| {
        let mut message = ChatMessage::new(ChatRole::User, text);
        if let Some(bytes) = image {
            message.images.push(ImagePart {
                offset: 0,
                bytes: Arc::new(bytes),
            });
        }
        ChatRequest {
            messages: vec![message],
            ..ChatRequest::default()
        }
    };
    let cache = ImageCache::default();
    let (text, _) = prepare_request(
        &tokenizer,
        &request("Say hi.", None),
        Some(&encoder),
        4096,
        &cache,
    )
    .unwrap();
    let (image, context) = prepare_request(
        &tokenizer,
        &request("What color is this image?", Some(red_png())),
        Some(&encoder),
        4096,
        &cache,
    )
    .unwrap();
    let prompt = Arc::clone(&context.unwrap().prompt);
    let greedy = SamplingConfig::default();

    let (model, session) = engine.parts_mut();
    model
        .configure_vision(session, None, Some(Arc::clone(&encoder)))
        .unwrap();
    model
        .generate_from_prefix(session, &text.tokens, 0, 8, &greedy, |_| true)
        .unwrap();
    model.reset(session).unwrap();
    model
        .configure_vision(
            session,
            Some(Arc::clone(&prompt)),
            Some(Arc::clone(&encoder)),
        )
        .unwrap();
    let after_text = model
        .generate_from_prefix(session, &image.tokens, 0, 8, &greedy, |_| true)
        .unwrap();
    assert_eq!(session.captured_graphs(), 2);
    assert_eq!(session.graph_mismatches(), 0);

    let mut fresh = model.new_session().unwrap();
    model
        .configure_vision(&mut fresh, Some(prompt), Some(encoder))
        .unwrap();
    let first = model
        .generate_from_prefix(&mut fresh, &image.tokens, 0, 8, &greedy, |_| true)
        .unwrap();
    assert_eq!(after_text, first);
    assert_eq!(after_text.len(), 8);
}
