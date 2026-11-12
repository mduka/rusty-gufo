//! WebP decoding against libwebp's own RGB decode of the same files
//! (`tests/reference/libwebp_reference.c`, `WebPDecodeRGBInto` as the C++ server
//! decodes), lossy and lossless, with and without alpha, and the rejection
//! of truncated, oversized and animated input:
//! cargo test -p gufo-core --test image

#![allow(clippy::unwrap_used)]

use gufo_core::image::decode_image;

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/webp/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

#[test]
fn webp_decodes_like_libwebp() {
    for name in ["lossless", "lossless_alpha", "lossy", "lossy_alpha"] {
        let image = decode_image(&fixture(&format!("{name}.webp"))).unwrap();
        let expected = fixture(&format!("{name}.rgb"));
        assert_eq!((image.width, image.height), (40, 28), "{name}");
        let differing = image
            .pixels
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a != b)
            .count();
        let largest = image
            .pixels
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap_or(0);
        assert!(
            image.pixels == expected,
            "{name}: {differing} of {} samples differ from libwebp, by up to {largest}",
            expected.len()
        );
    }
}

#[test]
fn webp_alpha_is_dropped_not_blended() {
    // Lossless keeps the RGB of semi-transparent pixels exactly.
    let opaque = decode_image(&fixture("lossless.webp")).unwrap();
    let alpha = decode_image(&fixture("lossless_alpha.webp")).unwrap();
    assert_eq!(opaque.pixels, alpha.pixels);
}

#[test]
fn invalid_webp_is_rejected() {
    let rejects = |bytes: &[u8], message: &str| {
        let error = decode_image(bytes).unwrap_err().to_string();
        assert!(error.contains(message), "{error}");
    };
    let bytes = fixture("lossless.webp");
    rejects(&bytes[..24], "invalid WebP");
    // VP8L's packed dimensions: 8192 x 8192 exceeds the decoded pixel
    // budget before any pixel is allocated.
    let mut oversized = bytes.clone();
    assert_eq!(oversized[20], 0x2f, "a VP8L signature");
    let dimensions = 8191u32 | (8191 << 14);
    oversized[21..25].copy_from_slice(&dimensions.to_le_bytes());
    rejects(&oversized, "decoded pixel limit");
    // An animation header (VP8X with the animation flag) is refused, as
    // libwebp's simple decoder refuses it.
    let mut animated = b"RIFF\0\0\0\0WEBPVP8X".to_vec();
    animated.extend_from_slice(&10u32.to_le_bytes());
    animated.extend_from_slice(&[0x02, 0, 0, 0, 39, 0, 0, 27, 0, 0]);
    let size = (animated.len() - 8) as u32;
    animated[4..8].copy_from_slice(&size.to_le_bytes());
    rejects(&animated, "invalid WebP");
    rejects(b"RIFF\0\0\0\0WEBQ", "PNG, JPEG or WebP");
}
