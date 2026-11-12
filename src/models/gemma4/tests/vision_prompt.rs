//! The extension's `vision_prompt_test.cpp`: Gemma4ImageProcessor's resize
//! sizes and the expansion of image placeholders into soft tokens with their
//! cache identities; plus `text_contract_test.cpp`'s image prefill stops.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use gufo_core::image::Image;
use gufo_core::text::{ChatMessage, ChatRole, ErrorKind, ImageDetail, ImagePart};
use gufo_gemma4::chat_template::ChatOptions;
use gufo_gemma4::text_contract::prefill_stop;
use gufo_gemma4::tokenizer::{TokenType, Tokenizer, Vocabulary};
use gufo_gemma4::vision_prompt::{
    ImageSize, ImageSpan, PreparedImage, Prompt, find_image_tokens, prepare, resize_image,
    resized_size,
};

/// Expected sizes come from transformers' get_aspect_ratio_preserving_size.
#[test]
fn sizing() {
    for (width, height, budget, resized_width, resized_height) in [
        (64, 64, 280, 768, 768),
        (1920, 1080, 280, 1056, 576),
        (1080, 1920, 280, 576, 1056),
        (10000, 10, 280, 13440, 48),
        (10, 5000, 280, 48, 13440),
        (48, 48, 70, 384, 384),
        (4032, 3024, 1120, 1824, 1344),
        (100, 1, 280, 8016, 48),
        (1, 1, 280, 768, 768),
        (333, 777, 140, 336, 864),
        (768, 768, 280, 768, 768),
        (3000, 2, 560, 26880, 48),
    ] {
        let size = resized_size(width, height, budget).unwrap();
        assert_eq!(
            size,
            ImageSize {
                width: resized_width,
                height: resized_height
            },
            "resize of {width}x{height} at {budget}"
        );
        assert!(
            (size.width / 48) * (size.height / 48) <= budget,
            "resize exceeds its token budget"
        );
    }
    let flat = Image::new(100, 60, vec![77; 100 * 60 * 3]).unwrap();
    let resized = resize_image(&flat, 280).unwrap();
    assert!(
        resized.width.is_multiple_of(48) && resized.height.is_multiple_of(48),
        "resized sides are not multiples of 48"
    );
    assert!(
        resized.pixels.iter().all(|&v| v == 77),
        "bicubic resize changed a flat image"
    );
}

fn build_tokenizer() -> Tokenizer {
    let mut v = Vocabulary::default();
    let mut add = |text: &str, ty: TokenType| {
        v.tokens.push(text.as_bytes().to_vec());
        v.types.push(ty);
    };
    add("<pad>", TokenType::Control);
    add("<eos>", TokenType::Control);
    add("<bos>", TokenType::Control);
    add("<unk>", TokenType::Unknown);
    for special in [
        "<|turn>",
        "<turn|>",
        "<|image>",
        "<|image|>",
        "<image|>",
        "<|channel>",
        "<channel|>",
    ] {
        add(special, TokenType::Control);
    }
    for b in 0..256 {
        add(&format!("<0x{b:02X}>"), TokenType::Byte);
    }
    v.bos = 2;
    v.eos = 1;
    v.pad = 0;
    v.unknown = 3;
    Tokenizer::new(v).unwrap()
}

fn base64(text: &str) -> Vec<u8> {
    let value = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        _ => 63,
    };
    let digits: Vec<u8> = text.bytes().filter(|&c| c != b'=').map(value).collect();
    digits
        .chunks(4)
        .flat_map(|chunk| {
            let bits = chunk
                .iter()
                .enumerate()
                .fold(0u32, |acc, (i, &d)| acc | (u32::from(d) << (18 - 6 * i)));
            let bytes = bits.to_be_bytes();
            bytes[1..chunk.len()].to_vec()
        })
        .collect()
}

/// A 64x64 PNG; the processor enlarges it to 768x768, 256 soft tokens.
fn png() -> Arc<Vec<u8>> {
    Arc::new(base64(
        "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAYklEQVR4nO3PMQ0AIADA\
         MEAD/jUiAREcDcmqYJtn7/GzpQNeNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWg\
         NaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaBdCLsBmEpLi1UAAAAASUVORK5C\
         YII=",
    ))
}

fn part(offset: usize, bytes: &Arc<Vec<u8>>) -> ImagePart {
    ImagePart {
        offset,
        bytes: bytes.clone(),
        detail: ImageDetail::Auto,
    }
}

#[test]
fn prompt() {
    let tokenizer = build_tokenizer();
    let tokens = find_image_tokens(&tokenizer).unwrap();
    let png = png();
    let mut message = ChatMessage::new(ChatRole::User, "ab");
    message.images = vec![part(1, &png), part(2, &png)];
    let messages = [message];
    let options = ChatOptions::default();
    let prepared = |identity: &str, context: u32, budget: u32| {
        prepare(
            &tokenizer,
            &messages,
            &[],
            &options,
            identity,
            context,
            budget,
        )
    };
    let prompt = prepared("encoder", 4096, 280).unwrap();
    assert_eq!(prompt.images.len(), 2, "two images expected");
    for image in &prompt.images {
        assert!(
            image.pixels.width == 768 && image.pixels.height == 768,
            "64x64 image not resized to 768x768"
        );
        assert_eq!(image.span.rows, 256, "768x768 image is 256 soft tokens");
        let (offset, rows) = (image.span.offset as usize, image.span.rows as usize);
        assert!(
            prompt.tokens[offset - 1] == tokens.start && prompt.tokens[offset + rows] == tokens.end,
            "soft tokens are not framed by <|image> and <image|>"
        );
        assert!(
            prompt.tokens[offset..offset + rows]
                .iter()
                .all(|&t| t == tokens.soft),
            "image span holds other tokens"
        );
    }
    // "a" (byte), image, "b", image, then the turn close.
    let (first, second) = (prompt.images[0].span, prompt.images[1].span);
    assert_eq!(
        second.offset,
        first.offset + first.rows + 3,
        "one text token separates the images"
    );
    assert!(
        prompt.stable_prefix_tokens > (second.offset + second.rows) as usize
            && prompt.stable_prefix_tokens < prompt.tokens.len(),
        "stable prefix does not cover the images"
    );
    assert!(
        !prompt.cache_identity.is_empty(),
        "image prompt has no identity"
    );
    assert!(
        prompt.identity_for_prefix(first.offset as usize).is_empty()
            && prompt.identity_for_prefix(first.offset as usize + 1)
                == prompt.images[0].prefix_identity
            && prompt.identity_for_prefix(prompt.tokens.len()) == prompt.images[1].prefix_identity,
        "prefix identities do not follow the images"
    );
    assert_ne!(
        prompt.images[0].prefix_identity, prompt.images[1].prefix_identity,
        "second image does not extend the identity"
    );

    // Identity changes with the encoder; tokens do not.
    let other = prepared("other", 4096, 280).unwrap();
    assert!(
        other.tokens == prompt.tokens && other.cache_identity != prompt.cache_identity,
        "encoder identity not part of the cache identity"
    );
    assert_eq!(
        prepared("", 4096, 280).unwrap_err().kind(),
        ErrorKind::InvalidArgument,
        "image accepted without an encoder"
    );
    assert_eq!(
        prepared("encoder", 300, 280).unwrap_err().kind(),
        ErrorKind::Length,
        "image beyond the context accepted"
    );

    // The largest budget keeps more detail; others are rejected.
    let large = prepared("encoder", 4096, 1120).unwrap();
    assert!(
        large.images[0].span.rows == 1089
            && large.images[0].pixels.width == 1584
            && large.cache_identity != prompt.cache_identity,
        "a 1120-token budget must give 33x33 soft tokens"
    );
    assert_eq!(
        prepared("encoder", 4096, 100).unwrap_err().kind(),
        ErrorKind::InvalidArgument,
        "unsupported image budget accepted"
    );

    let text = prepare(
        &tokenizer,
        &[ChatMessage::new(ChatRole::User, "ab")],
        &[],
        &options,
        "",
        4096,
        280,
    )
    .unwrap();
    assert!(
        text.images.is_empty() && text.cache_identity.is_empty(),
        "text prompt carries image state"
    );
}

/// `text_contract_test.cpp` `CheckPrefillStops`.
#[test]
fn prefill_stops() {
    let mut prompt = Prompt::default();
    // Image soft tokens [10, 15) closed at 15, and [30, 33) closed at 33.
    for (offset, rows) in [(10u32, 5u32), (30, 3)] {
        prompt.images.push(PreparedImage {
            pixels: Image::new(0, 0, Vec::new()).unwrap(),
            span: ImageSpan { offset, rows },
            prefix_identity: [0; 32],
        });
    }
    assert!(
        prefill_stop(None, 12, false) == 12 && prefill_stop(None, 12, true) == 12,
        "no images declare no spans"
    );
    for (position, down, up) in [
        (0, 0, 0),
        (9, 9, 9),
        (10, 10, 10),
        (11, 10, 16),
        (14, 10, 16),
        (15, 10, 16),
        (16, 16, 16),
        (29, 29, 29),
        (30, 30, 30),
        (31, 30, 34),
        (33, 30, 34),
        (34, 34, 34),
        (100, 100, 100),
    ] {
        let got_down = prefill_stop(Some(&prompt), position, false);
        let got_up = prefill_stop(Some(&prompt), position, true);
        assert!(
            got_down == down && got_up == up,
            "prefill stop at {position}"
        );
        for stop in [got_down, got_up] {
            assert!(
                prefill_stop(Some(&prompt), stop, false) == stop
                    && prefill_stop(Some(&prompt), stop, true) == stop,
                "stops are fixed points at {position}"
            );
        }
    }
}
