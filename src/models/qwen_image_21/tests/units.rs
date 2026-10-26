//! The image, schedule and validation cases of the C++ test
//! (`tests/models/qwen_image_21/model_test.cpp`); no device or checkpoint.

#![allow(clippy::unwrap_used)]

use gufo_qwen_image_21::image::{
    decode_image, encode_png, flow_sigmas, initial_noise_seed, resize_image, validate_request,
};
use gufo_qwen_image_21::{Image, Request};

fn small() -> Image {
    Image {
        width: 2,
        height: 2,
        rgba: vec![
            255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 127, 63, 21, 255,
        ],
    }
}

#[test]
fn png_round_trips_every_byte() {
    let image = small();
    assert_eq!(decode_image(&encode_png(&image).unwrap()).unwrap(), image);
    let mut textured = Image {
        width: 65,
        height: 17,
        rgba: vec![0; 65 * 17 * 4],
    };
    for (i, value) in textured.rgba.iter_mut().enumerate() {
        *value = ((i * 73) ^ (i >> 3)) as u8;
    }
    assert_eq!(
        decode_image(&encode_png(&textured).unwrap()).unwrap(),
        textured
    );
}

#[test]
fn resize_follows_pillow_rgba_lanczos() {
    let image = small();
    assert_eq!(resize_image(&image, 2, 2).unwrap().rgba, image.rgba);
    // Premultiplication, two uint8 passes, unpremultiplication.
    let resized: [u8; 64] = [
        255, 0, 0, 255, 225, 33, 0, 254, 87, 180, 0, 151, 0, 255, 0, 80, 255, 0, 0, 196, 215, 37,
        1, 186, 103, 131, 6, 167, 63, 238, 10, 141, 255, 0, 0, 59, 179, 46, 10, 98, 120, 81, 16,
        187, 132, 106, 22, 220, 0, 0, 0, 0, 8, 102, 51, 30, 130, 48, 23, 203, 175, 61, 29, 255,
    ];
    assert_eq!(resize_image(&image, 4, 4).unwrap().rgba, resized);
}

#[test]
fn flow_schedule_ends_at_the_terminal_sigma_and_zero() {
    let schedule = flow_sigmas(40, 4096).unwrap();
    assert_eq!(schedule.len(), 41);
    assert_eq!((schedule[0], schedule[40]), (1.0, 0.0));
    assert!((schedule[39] - 0.02).abs() < 1e-6);
    assert!(schedule.windows(2).all(|w| w[1].is_finite() && w[1] < w[0]));
    assert!(flow_sigmas(1, 4096).is_err());
}

#[test]
fn requests_are_validated_like_cpp() {
    validate_request(&Request {
        prompt: "a red cube".into(),
        ..Request::default()
    })
    .unwrap();
    for invalid in [
        Request {
            width: 31,
            ..Request::default()
        },
        Request {
            steps: 1,
            ..Request::default()
        },
        Request {
            height: 8192,
            ..Request::default()
        },
    ] {
        let error = validate_request(&invalid).unwrap_err();
        assert_eq!(error.kind(), gufo_core::text::ErrorKind::InvalidArgument);
    }
}

#[test]
fn codec_errors_keep_their_cpp_kind_and_message() {
    use gufo_core::text::ErrorKind;
    use gufo_qwen_image_21::image::encode_rgba_png;
    let cases: [(&[u8], &str); 3] = [
        (b"short", "invalid or oversized input image"),
        (b"\x89PNG\r\n\x1a\nbroken", "cannot decode PNG image"),
        (b"neither png nor jpeg", "image must be PNG, JPEG or WebP"),
    ];
    for (bytes, message) in cases {
        let error = decode_image(bytes).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidArgument, "{message}");
        assert_eq!(error.message(), message);
    }
    let error = encode_rgba_png(2, 2, &[0; 3]).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(error.message(), "invalid RGBA image dimensions");
}

#[test]
fn a_missing_checkpoint_reports_the_cpp_error() {
    let error =
        gufo_qwen_image_21::weights::Weights::load(std::path::Path::new("/nonexistent/qwen-image"))
            .err()
            .unwrap();
    assert_eq!(error.kind(), gufo_core::text::ErrorKind::Runtime);
    assert_eq!(
        error.message(),
        "cannot read Qwen-Image metadata: /nonexistent/qwen-image/model_index.json"
    );
}

/// Initial noise never repeats the noise that generated a reference at the
/// same seed and size, and re-editing the result draws anew.
#[test]
fn edit_noise_binds_its_references() {
    let image = small();
    assert_eq!(initial_noise_seed(42, &[]).unwrap(), 42);
    assert_eq!(initial_noise_seed(0, &[]).unwrap(), 0);
    let edit = initial_noise_seed(42, std::slice::from_ref(&image)).unwrap();
    // Upstream's independent hashlib SHA-256 golden: little-endian fields
    // and digest.
    assert_eq!(edit, 9_781_428_742_393_689_377);
    assert_eq!(
        initial_noise_seed(42, std::slice::from_ref(&image)).unwrap(),
        edit
    );
    assert_ne!(
        initial_noise_seed(43, std::slice::from_ref(&image)).unwrap(),
        edit
    );
    let seed = |images: &[Image]| initial_noise_seed(42, images).unwrap();
    let mut changed = image.clone();
    changed.rgba[0] ^= 1;
    assert_ne!(seed(std::slice::from_ref(&changed)), edit);
    let mut alpha = image.clone();
    alpha.rgba[3] ^= 1;
    assert_ne!(seed(std::slice::from_ref(&alpha)), edit);
    let pair = seed(&[image.clone(), alpha.clone()]);
    assert_ne!(pair, edit);
    assert_ne!(pair, seed(&[alpha, image.clone()]));
    let reshaped = Image {
        width: 1,
        height: 4,
        ..image.clone()
    };
    assert_ne!(seed(&[reshaped]), edit);
    let roundtrip = decode_image(&encode_png(&image).unwrap()).unwrap();
    assert_eq!(seed(&[roundtrip]), edit);
    assert!(initial_noise_seed(42, &[Image { width: 0, ..image }]).is_err());
}
