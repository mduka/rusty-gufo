//! Host video preprocessing against the pinned official processor
//! (`video_processor_test.cpp`): the sampled frame indices, the patch grid
//! and the SHA-256 of every frame's decoded, resized patch bytes, plus the
//! decoder's bounds. Needs the FFmpeg libraries the build found:
//! cargo test -p gufo-embedding-gemma2 --test video

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use gufo_core::json;
use gufo_embedding_gemma2::video_processor::{
    VideoError, VideoOptions, frame_video, process_video, sample_video_frames,
};
use sha2::{Digest, Sha256};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn refused(result: Result<impl std::fmt::Debug, VideoError>, name: &str) {
    match result {
        Err(VideoError::Invalid(_)) => {}
        other => panic!("{name}: expected a refusal, got {other:?}"),
    }
}

#[test]
fn frames_match_the_reference_processor() {
    let root = json::parse(&std::fs::read(fixtures().join("video_golden.json")).unwrap()).unwrap();
    for probe in root.find("cases").unwrap().items() {
        let name = probe.member_str("name", "");
        let settings = probe.find("options").unwrap();
        let options = VideoOptions {
            fps: settings.member_double("fps", 1.0),
            timestamps: settings
                .find("add_timestamps")
                .is_some_and(|v| v.as_bool(false)),
            max_frames: settings.member_size("max_frames", 32),
            truncate_overflow: settings.member_str("overflow_strategy", "") == "truncate",
        };
        for video in probe.find("videos").unwrap().items() {
            let bytes = std::fs::read(fixtures().join(video.member_str("path", ""))).unwrap();
            assert_eq!(sha256(&bytes), video.member_str("sha256", ""), "media hash");
            let frames = process_video(&bytes, &options, None, 32).unwrap();
            let indices = video.find("indices").unwrap().items();
            let expected = video.find("patches").unwrap().items();
            assert_eq!(frames.len(), indices.len(), "{name} sampled frame count");
            for (i, frame) in frames.iter().enumerate() {
                assert_eq!(
                    frame.index,
                    indices[i].as_size(0),
                    "{name} selected frame index"
                );
                assert_eq!(
                    (frame.patches.rows as usize, frame.patches.columns as usize),
                    (
                        expected[i].member_size("rows", 0),
                        expected[i].member_size("columns", 0)
                    ),
                    "{name} patch geometry"
                );
                // `std::lround(value * 255)`.
                let pixels: Vec<u8> = frame
                    .patches
                    .pixels
                    .iter()
                    .map(|&value| (value * 255.0).round() as u8)
                    .collect();
                assert_eq!(
                    sha256(&pixels),
                    expected[i].member_str("sha256", ""),
                    "{name} decoded RGB and resized patch bytes"
                );
            }
            refused(
                process_video(&bytes, &options, None, 0),
                "request frame budget",
            );
            let framed = frame_video(&frames, &options);
            assert!(
                !framed.is_empty() && framed.contains("<|video|>"),
                "{name} video framing"
            );
            let cancelled = || true;
            refused(
                process_video(&bytes, &VideoOptions::default(), Some(&cancelled), 32),
                "cancelled decoder",
            );
        }
        println!("PASS {name}");
    }
}

#[test]
fn resource_bounds_refuse_videos() {
    for name in ["oversized.mp4", "overduration.mp4", "overframes.mp4"] {
        let bytes = std::fs::read(fixtures().join("video").join(name))
            .unwrap_or_else(|e| panic!("missing resource fixture {name}: {e}"));
        refused(
            process_video(&bytes, &VideoOptions::default(), None, 32),
            name,
        );
    }
    refused(
        process_video(&[], &VideoOptions::default(), None, 32),
        "empty video",
    );
    refused(
        process_video(&[1, 2, 3, 4], &VideoOptions::default(), None, 32),
        "malformed video",
    );
    refused(
        process_video(
            &vec![0; 20 * 1024 * 1024 + 1],
            &VideoOptions::default(),
            None,
            32,
        ),
        "encoded video bound",
    );
}

#[test]
fn sampling_matches_the_reference_arithmetic() {
    let defaults = VideoOptions::default();
    assert_eq!(
        sample_video_frames(60, 30000.0 / 1001.0, &defaults).unwrap(),
        [0, 29],
        "fractional floor sampling"
    );
    let tiny_fps = VideoOptions {
        fps: 1e-320,
        ..defaults
    };
    assert_eq!(
        sample_video_frames(60, 30.0, &tiny_fps).unwrap(),
        [0],
        "tiny positive FPS avoids nonfinite arithmetic"
    );
    let rounding = VideoOptions {
        max_frames: 23,
        ..defaults
    };
    let rounded = sample_video_frames(31, 1.0, &rounding).unwrap();
    assert!(
        rounded.len() == 23 && rounded[11] == 14 && rounded[22] == 30,
        "uniform overflow matches NumPy floating linspace floor"
    );
    let capped = sample_video_frames(66, 2.0, &defaults).unwrap();
    assert!(
        capped.len() == 32 && capped[0] == 0 && capped[31] == 64 && capped[30] == 60,
        "uniform overflow boundary"
    );
    assert!(
        sample_video_frames(1, 0.0, &defaults).is_err(),
        "missing fps"
    );
    assert!(
        sample_video_frames(4097, 25.0, &defaults).is_err(),
        "frame work limit"
    );
    assert!(
        sample_video_frames(301, 1.0, &defaults).is_err(),
        "duration limit"
    );
}

#[test]
fn timestamps_frame_each_video_frame() {
    let bytes = std::fs::read(fixtures().join("video/portrait.webm")).unwrap();
    let options = VideoOptions {
        timestamps: true,
        ..VideoOptions::default()
    };
    let frames = process_video(&bytes, &options, None, 32).unwrap();
    let framed = frame_video(&frames, &options);
    let soft_tokens = frames[0].patches.soft_tokens();
    let frame = |time: &str| format!("{time} <|image>{}<image|>", "<|video|>".repeat(soft_tokens));
    // Frames 0, 2 and 4 of a 2-FPS clip.
    assert_eq!(
        framed,
        [frame("00:00"), frame("00:01"), frame("00:02")].join(" ")
    );
}
