//! The extension's `gemma4_vision_probe` GPU path: encodes a raw RGB8 image
//! (sides multiples of 48) with the HIP encoder, or with `--reference` the
//! scalar reference, and writes the embeddings as a G4VE file (`u32` magic
//! `G4VE`, version 1, rows, width, then `[rows][width]` f32), the layout the
//! C++ probe's `--output` writes, for a comparison of the two builds.
//!
//!   gemma4_vision_encode --mmproj F --image RGB W H --output G4VE [--reference]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use gufo_core::gguf::Gguf;
use gufo_core::image::Image;
use gufo_gemma4::vision_encoder::VisionEncoder;
use gufo_gemma4::vision_reference::VisionReference;
use gufo_gemma4::vision_weights::VisionWeights;
use gufo_hip::{MappedFile, Stream};

fn run(args: &[String]) -> Result<(), String> {
    let (mut mmproj, mut image_path, mut output) = (None, None, None);
    let (mut width, mut height) = (0u32, 0u32);
    let mut reference = false;
    let mut i = 0;
    let value = |i: usize| args.get(i).cloned().ok_or("missing value");
    while i < args.len() {
        match args[i].as_str() {
            "--mmproj" => {
                mmproj = Some(PathBuf::from(value(i + 1)?));
                i += 1;
            }
            "--image" => {
                image_path = Some(PathBuf::from(value(i + 1)?));
                width = value(i + 2)?.parse().map_err(|_| "width")?;
                height = value(i + 3)?.parse().map_err(|_| "height")?;
                i += 3;
            }
            "--output" => {
                output = Some(PathBuf::from(value(i + 1)?));
                i += 1;
            }
            "--reference" => reference = true,
            other => return Err(format!("unknown argument {other}")),
        }
        i += 1;
    }
    let (Some(mmproj), Some(image_path), Some(output)) = (mmproj, image_path, output) else {
        return Err("usage: --mmproj F --image RGB W H --output G4VE [--reference]".into());
    };
    let pixels = std::fs::read(&image_path).map_err(|e| e.to_string())?;
    let image = Image {
        width,
        height,
        pixels,
    };
    let file = MappedFile::open(&mmproj).map_err(|e| e.to_string())?;
    let projection_width = Gguf::parse(file.bytes())?
        .u32("clip.vision.projection_dim")
        .ok_or("no projection_dim")?;
    let (rows, embedding) = if reference {
        let gguf = Gguf::parse(file.bytes())?;
        let weights = VisionWeights::resolve(&gguf, projection_width)?;
        let embedding = VisionReference::new(&weights, file.bytes()).encode(&image, None)?;
        (embedding.len() / projection_width as usize, embedding)
    } else {
        let start = Instant::now();
        let encoder = VisionEncoder::load(&mmproj, projection_width)?;
        let loaded = start.elapsed();
        // Uploads the weights and warms up, like the C++ probe.
        drop(encoder.encode(&image)?);
        println!(
            "open {:.1} ms, upload and first encode {:.1} ms",
            loaded.as_secs_f64() * 1e3,
            (start.elapsed() - loaded).as_secs_f64() * 1e3
        );
        let start = Instant::now();
        let device = encoder.encode(&image)?;
        let rows = device.len() / projection_width as usize;
        println!(
            "gpu encode {width}x{height} ({rows} rows): {:.1} ms",
            start.elapsed().as_secs_f64() * 1e3
        );
        let mut host = vec![0.0f32; device.len()];
        Stream::new()
            .and_then(|stream| stream.download(device.view(), &mut host))
            .map_err(|e| e.to_string())?;
        (rows, host)
    };
    let mut bytes = Vec::with_capacity(16 + embedding.len() * 4);
    for word in [0x4556_3447u32, 1, rows as u32, projection_width] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    for value in &embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&output, bytes).map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
