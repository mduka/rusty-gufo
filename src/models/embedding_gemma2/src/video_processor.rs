//! The EmbeddingGemma 2 video processor (`video_processor.cpp`): frames
//! sampled at a requested rate from an in-memory clip, each resized like an
//! image at a 140-soft-token budget and framed by the image markers around
//! video placeholders.

use std::path::Path;

use gufo_core::image::Image;
use gufo_core::json;
pub use gufo_ffmpeg::Error as VideoError;
use gufo_ffmpeg::{Budget, Decoder};

use crate::config::{BEGIN_IMAGE_TOKEN, END_IMAGE_TOKEN, VIDEO_TOKEN};
use crate::image_processor::{ImagePatches, process_image};

/// The most bytes of one encoded video.
const MAX_VIDEO_BYTES: usize = 20 * 1024 * 1024;
/// The soft-token budget of every video frame.
const FRAME_SOFT_TOKENS: u32 = 140;

/// The processor controls of one request (`VideoOptions`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoOptions {
    pub fps: f64,
    pub max_frames: usize,
    pub timestamps: bool,
    pub truncate_overflow: bool,
}

impl Default for VideoOptions {
    fn default() -> Self {
        Self {
            fps: 1.0,
            max_frames: 32,
            timestamps: false,
            truncate_overflow: false,
        }
    }
}

/// One sampled frame: its index in the clip, its time and its patches.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoFrame {
    pub index: usize,
    pub timestamp: f64,
    pub patches: ImagePatches,
}

/// `SampleVideoFrames`: exact FPS/floor selection followed by uniform
/// linspace/floor overflow.
pub fn sample_video_frames(
    frames: usize,
    fps: f64,
    options: &VideoOptions,
) -> Result<Vec<usize>, String> {
    if frames == 0
        || frames > 4096
        || !fps.is_finite()
        || fps <= 0.0
        || !options.fps.is_finite()
        || options.fps <= 0.0
        || options.fps > 120.0
        || options.max_frames == 0
        || options.max_frames > 32
        || frames as f64 / fps > 300.0
    {
        return Err("invalid embedding video rate, duration or frame count".into());
    }
    let sampled = ((frames as f64 / fps * options.fps) as usize).max(1);
    let kept = sampled.min(options.max_frames);
    // NumPy linspace multiplies by a double step, then floors for dtype=int.
    // Rational integer division differs at floating rounding boundaries.
    let step = if kept > 1 {
        (sampled - 1) as f64 / (kept - 1) as f64
    } else {
        0.0
    };
    let indices = (0..kept)
        .map(|j| {
            let i = if sampled <= kept || options.truncate_overflow {
                j
            } else if j + 1 == kept && kept > 1 {
                sampled - 1
            } else {
                (j as f64 * step) as usize
            };
            if i == 0 {
                0
            } else {
                (frames - 1).min((i as f64 * (fps / options.fps)) as usize)
            }
        })
        .collect();
    Ok(indices)
}

/// `ValidateVideoProcessor`: refuses settings for which the fixed processor
/// would be incorrect.
pub fn validate_video_processor(model_dir: &Path) -> Result<(), String> {
    let contents = std::fs::read(model_dir.join("processor_config.json")).unwrap_or_default();
    let supported = json::parse(&contents).is_ok_and(|root| {
        let Some(c) = root.find("video_processor") else {
            return false;
        };
        let flag = |name: &str, expected: bool| {
            c.find(name)
                .is_some_and(|v| v.is_bool() && v.as_bool(false) == expected)
        };
        c.member_str("video_processor_type", "") == "EmbeddingGemma2VideoProcessor"
            && flag("do_convert_rgb", true)
            && flag("do_resize", true)
            && flag("do_rescale", true)
            && flag("do_normalize", false)
            && flag("do_sample_frames", true)
            && flag("add_timestamps", false)
            && c.member_double("fps", 0.0) == 1.0
            && c.member_size("max_frames", 0) == 32
            && c.member_str("overflow_strategy", "") == "uniform"
            && c.member_size("max_soft_tokens", 0) == 140
            && c.member_size("patch_size", 0) == 16
            && c.member_size("pooling_kernel_size", 0) == 3
            && c.member_size("resample", 0) == 3
            && (c.member_double("rescale_factor", 0.0) - 1.0 / 255.0).abs() < 1e-15
    });
    if supported {
        Ok(())
    } else {
        Err("embedding_gemma2: unsupported or missing video processor configuration".into())
    }
}

/// `ProcessVideo`: decodes `bytes` in memory, never opening nested URLs or
/// files, from the first video stream (audio is ignored, as in the visual
/// reference path), and returns the sampled frames. Bounds: 20 MiB, 300
/// seconds, 4096 decoded frames, 4M pixels per frame, 512M decoded pixels
/// over both passes, 15 seconds including conversion, and at most
/// `frame_budget` sampled frames.
pub fn process_video(
    bytes: &[u8],
    options: &VideoOptions,
    is_cancelled: Option<&dyn Fn() -> bool>,
    frame_budget: usize,
) -> Result<Vec<VideoFrame>, VideoError> {
    let invalid = |message: &str| VideoError::Invalid(message.into());
    if bytes.is_empty() || bytes.len() > MAX_VIDEO_BYTES {
        return Err(invalid("embedding video is empty or exceeds 20 MiB"));
    }
    let budget = Budget::new(is_cancelled);
    budget.check()?;
    // Finish loading before a decoder can need cleanup.
    gufo_ffmpeg::load()?;
    let (count, fps) = {
        let mut decoder = Decoder::open(bytes, &budget)?;
        let fps = decoder.average_frame_rate();
        let count = decoder.decode(|_, _| Ok(()))?;
        let declared = decoder.declared_frames();
        if declared > 0 && declared as usize != count {
            return Err(invalid(
                "embedding video frame count differs from container; truncated input",
            ));
        }
        (count, fps)
    };
    let selected = sample_video_frames(count, fps, options).map_err(VideoError::Invalid)?;
    if selected.len() > frame_budget {
        return Err(invalid("embedding request video frame budget exhausted"));
    }
    let mut decoder = Decoder::open(bytes, &budget)?;
    let mut frames: Vec<VideoFrame> = Vec::with_capacity(selected.len());
    decoder.decode(|index, frame| {
        if frames.len() >= selected.len() || index != selected[frames.len()] {
            return Ok(());
        }
        let pixels = frame.to_rgb24()?;
        // The decoder bounded the dimensions to positive values.
        let image = Image::new(frame.width() as u32, frame.height() as u32, pixels)
            .map_err(|e| VideoError::Invalid(e.message().to_owned()))?;
        let patches = process_image(&image, FRAME_SOFT_TOKENS).map_err(VideoError::Invalid)?;
        loop {
            frames.push(VideoFrame {
                index,
                timestamp: index as f64 / fps,
                patches: patches.clone(),
            });
            if !(frames.len() < selected.len() && selected[frames.len()] == index) {
                break;
            }
        }
        budget.check()
    })?;
    if frames.len() != selected.len() {
        return Err(invalid("embedding video lost selected frames"));
    }
    Ok(frames)
}

/// `FrameVideo`: the reference per-frame image boundaries around video
/// placeholders, each frame optionally after its `mm:ss` time.
pub fn frame_video(frames: &[VideoFrame], options: &VideoOptions) -> String {
    let mut text = String::new();
    for frame in frames {
        if options.timestamps {
            if !text.is_empty() {
                text.push(' ');
            }
            let seconds = frame.timestamp as i32;
            text.push_str(&format!("{:02}:{:02} ", seconds / 60, seconds % 60));
        }
        text.push_str(BEGIN_IMAGE_TOKEN);
        for _ in 0..frame.patches.soft_tokens() {
            text.push_str(VIDEO_TOKEN);
        }
        text.push_str(END_IMAGE_TOKEN);
    }
    text
}
