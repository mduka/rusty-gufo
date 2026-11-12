//! The C++ extension's `nemotron_vision_probe`: times image decoding,
//! preprocessing and encoding per image, with the encoder's throughput.
//!
//! cargo run --profile gpu-test -p gufo-nemotron --example vision_probe -- \
//!   TRUNK.gguf MMPROJ.gguf [--repeat N] WxH|IMAGE...

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use gufo_core::image::{Image, decode_image, read_image_file};
use gufo_hip::MappedFile;
use gufo_nemotron::config::{Config, VisionConfig};
use gufo_nemotron::vision_encoder::{VisionEncoder, VisionEncoderOptions};
use gufo_nemotron::vision_image::{ImageGrid, image_tokens, plan_grid, preprocess_image};
use gufo_nemotron::weights::GgufShard;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// `tools/vision_golden.py`'s synthetic image (`tests/vision_synthetic.hpp`).
fn synthetic(width: u32, height: u32) -> Image {
    let mut pixels = vec![0u8; width as usize * height as usize * 3];
    for y in 0..u64::from(height) {
        for x in 0..u64::from(width) {
            for c in 0..3u64 {
                let h = (x.wrapping_mul(73_856_093)
                    ^ y.wrapping_mul(19_349_663)
                    ^ c.wrapping_mul(83_492_791))
                    & 0xFFFF_FFFF;
                let v = (x * 255) / u64::from(width)
                    + (y * 128) / u64::from(height)
                    + c * 40
                    + (h >> 13) % 48;
                pixels[((y * u64::from(width) + x) * 3 + c) as usize] = (v % 256) as u8;
            }
        }
    }
    Image {
        width,
        height,
        pixels,
    }
}

/// Multiply-adds times two of one encode: ViT products and attention, then
/// the projector.
fn flops(c: &VisionConfig, grid: &ImageGrid) -> f64 {
    let n = f64::from(grid.patches());
    let t = n + f64::from(c.prefix_tokens);
    let w = f64::from(c.width);
    let vit = 2.0 * t * (4.0 * w * w + 2.0 * w * f64::from(c.ff_size)) + 4.0 * t * t * w;
    let merged = n / f64::from(c.merge * c.merge);
    let projector = 2.0
        * merged
        * (f64::from(c.merged_width()) * f64::from(c.projector_hidden)
            + f64::from(c.projector_hidden) * f64::from(c.projection));
    2.0 * n * f64::from(c.patch_values()) * w + f64::from(c.num_layers) * vit + projector
}

fn run(args: &[String]) -> Result<(), String> {
    let trunk = {
        let file = MappedFile::open(&args[1]).map_err(|e| e.to_string())?;
        let shard = GgufShard::parse(file.bytes())?;
        Config::from_gguf(&shard.gguf)?
    };
    let encoder =
        VisionEncoder::load(Path::new(&args[2]), &trunk, VisionEncoderOptions::default())?;
    encoder.prepare()?;
    let c = encoder.config().clone();
    let mut repeat = 5usize;
    let mut i = 3;
    while i < args.len() {
        let arg = &args[i];
        i += 1;
        if arg == "--repeat" && i < args.len() {
            repeat = args[i].parse::<usize>().unwrap_or(1).max(1);
            i += 1;
            continue;
        }
        let mut decode = Vec::new();
        let image = match arg.split_once('x') {
            Some((w, h))
                if arg.chars().all(|ch| ch.is_ascii_digit() || ch == 'x')
                    && !w.is_empty()
                    && !h.is_empty() =>
            {
                synthetic(
                    w.parse().map_err(|_| format!("{arg}: bad size"))?,
                    h.parse().map_err(|_| format!("{arg}: bad size"))?,
                )
            }
            _ => {
                let bytes = read_image_file(Path::new(arg)).map_err(|e| e.to_string())?;
                let mut image = None;
                for _ in 0..repeat {
                    let start = Instant::now();
                    image = Some(decode_image(&bytes).map_err(|e| e.to_string())?);
                    decode.push(start.elapsed().as_secs_f64());
                }
                image.ok_or("no image")?
            }
        };
        let grid = plan_grid(&c, image.width, image.height, 0).ok_or(format!("{arg}: no grid"))?;
        let mut preprocess = Vec::new();
        let mut patches = Vec::new();
        for _ in 0..repeat {
            let start = Instant::now();
            patches = preprocess_image(&c, &image, &grid)?;
            preprocess.push(start.elapsed().as_secs_f64());
        }
        encoder.encode(&patches, &grid, None)?;
        let mut encode = Vec::new();
        for _ in 0..repeat {
            let start = Instant::now();
            encoder.encode(&patches, &grid, None)?;
            encode.push(start.elapsed().as_secs_f64());
        }
        let seconds = median(encode);
        let mut line = format!(
            "{arg} ({}x{}): grid {}x{}, {} tokens",
            image.width,
            image.height,
            grid.rows,
            grid.cols,
            image_tokens(&c, &grid)
        );
        if !decode.is_empty() {
            line += &format!(", decode {:.1} ms", median(decode) * 1e3);
        }
        line += &format!(
            ", preprocess {:.1} ms, encode {:.1} ms ({:.1} TFLOPS)",
            median(preprocess) * 1e3,
            seconds * 1e3,
            flops(&c, &grid) / seconds * 1e-12
        );
        println!("{line}");
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: vision_probe TRUNK.gguf MMPROJ.gguf [--repeat N] WxH|IMAGE...");
        return ExitCode::from(2);
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("vision_probe: {error}");
            ExitCode::FAILURE
        }
    }
}
