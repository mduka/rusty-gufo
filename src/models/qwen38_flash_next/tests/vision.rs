//! An image prompt on a session whose decode graphs a text prompt captured:
//! the image creates the session's mRoPE table, so its steps capture graphs
//! of their own instead of replaying the text ones (which differ in the
//! `rope` argument), and decode as on a fresh session:
//! cargo test -p gufo-qwen38-flash-next --features gpu --test vision
//!
//! Needs the Flash-Next trunk and its vision sidecar (`GUFO_FLASH_NEXT_MMPROJ`,
//! else the pinned one), and fails without them.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use gufo_core::image::Image;
use gufo_qwen::vision::{IMAGE_TOKEN, ImageGrid, PreparedImage, Prompt, RopeLayout};
use gufo_qwen38_flash_next::engine::{Model, ModelOptions};
use gufo_qwen38_flash_next::executor::SessionMode;

const CONTEXT: u32 = 256;

#[test]
fn an_image_after_text_decodes_as_on_a_fresh_session() {
    let vision = common::vision_sidecar();
    assert!(vision.is_file(), "{}: no vision sidecar", vision.display());
    let mut model = Model::load(
        &common::trunk_path(),
        ModelOptions {
            max_context: CONTEXT,
            vision_model_path: Some(vision),
            ..ModelOptions::default()
        },
    )
    .unwrap();
    let grid = ImageGrid {
        offset: 0,
        height: 2,
        width: 2,
    };
    let mut pixels = vec![0u8; 64 * 64 * 3];
    for value in pixels.iter_mut().step_by(3) {
        *value = 255;
    }
    let identity = [1; 32];
    let red = Arc::new(Prompt {
        tokens: vec![IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 42],
        stable_prefix_tokens: 5,
        rope: RopeLayout { images: vec![grid] },
        images: vec![Arc::new(PreparedImage {
            pixels: Image::new(64, 64, pixels).unwrap().into(),
            grid,
            prefix_identity: identity,
        })],
        cache_identity: identity.to_vec(),
    });
    let tokens: Vec<i32> = red.tokens.iter().map(|&t| t as i32).collect();

    let mut reused = model
        .create_session(SessionMode::Autoregressive, CONTEXT)
        .unwrap();
    model.sync(&mut reused, &[42, 43]).unwrap();
    for token in 44..48 {
        model.evaluate(&mut reused, token).unwrap();
    }
    model
        .configure_vision(&mut reused, Some(Arc::clone(&red)))
        .unwrap();
    model.sync(&mut reused, &tokens).unwrap();

    let mut fresh = model
        .create_session(SessionMode::Autoregressive, CONTEXT)
        .unwrap();
    model.configure_vision(&mut fresh, Some(red)).unwrap();
    model.sync(&mut fresh, &tokens).unwrap();
    for token in 44..48 {
        model.evaluate(&mut reused, token).unwrap();
        model.evaluate(&mut fresh, token).unwrap();
        assert!(
            reused.logits() == fresh.logits(),
            "logits differ after token {token}"
        );
    }
}
