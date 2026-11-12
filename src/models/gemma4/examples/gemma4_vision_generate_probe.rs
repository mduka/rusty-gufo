//! The extension's `gemma4_vision_generate_probe`: greedy generation from an
//! image question, then the teacher-forced logits of every position from
//! the last image's closing token on. `DIR` receives what the C++ probe
//! writes (`tokens.i32`, `image<i>.rgb` with the resized pixels, and
//! `gufo.g4lg`), for a comparison of the two builds.
//!
//!   gemma4_vision_generate_probe --model GGUF --mmproj GGUF --image FILE
//!       [--question TEXT] [--tokens N] [--out-dir DIR] [--image-tokens N]

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use gufo_core::text::{ChatMessage, ChatRole, ImageDetail, ImagePart};
use gufo_gemma4::chat_template::ChatOptions;
use gufo_gemma4::engine::{ImageSpan, Model, ModelOptions};
use gufo_gemma4::tokenizer::TokenId;
use gufo_gemma4::vision_encoder::VisionEncoder;
use gufo_gemma4::vision_prompt::{DEFAULT_SOFT_TOKENS, prepare};

const MAX_CONTEXT: u32 = 8192;

/// `std::max_element`: the first maximum.
fn argmax(logits: &[f32]) -> TokenId {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (j, &v)| {
            if v > bv { (j, v) } else { (bi, bv) }
        })
        .0 as TokenId
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

fn run(args: &[String]) -> Result<(), String> {
    let (mut model_path, mut mmproj, mut image_path, mut out_dir) =
        (String::new(), String::new(), String::new(), String::new());
    let mut question = "Describe this image in detail.".to_string();
    let mut count = 64usize;
    let mut image_tokens = DEFAULT_SOFT_TOKENS;
    for pair in args.chunks(2) {
        let [arg, value] = pair else {
            return Err(format!("missing value for {}", pair[0]));
        };
        match arg.as_str() {
            "--model" => model_path = value.clone(),
            "--mmproj" => mmproj = value.clone(),
            "--image" => image_path = value.clone(),
            "--question" => question = value.clone(),
            "--tokens" => count = value.parse().map_err(|_| "--tokens")?,
            "--out-dir" => out_dir = value.clone(),
            "--image-tokens" => image_tokens = value.parse().map_err(|_| "--image-tokens")?,
            other => return Err(format!("unknown option {other}")),
        }
    }
    if model_path.is_empty() || mmproj.is_empty() || image_path.is_empty() {
        return Err("--model, --mmproj and --image are required".into());
    }
    let model = Model::load(
        Path::new(&model_path),
        ModelOptions {
            max_context: MAX_CONTEXT,
            ..ModelOptions::default()
        },
    )?;
    let encoder = VisionEncoder::load(Path::new(&mmproj), model.config().hidden_size)?;

    let bytes = std::fs::read(&image_path).map_err(|e| format!("{image_path}: {e}"))?;
    let mut message = ChatMessage::new(ChatRole::User, question);
    message.images.push(ImagePart {
        offset: 0,
        bytes: Arc::new(bytes),
        detail: ImageDetail::Auto,
    });
    let prompt = prepare(
        model.tokenizer(),
        &[message],
        &[],
        &ChatOptions::default(),
        encoder.identity(),
        MAX_CONTEXT,
        image_tokens,
    )
    .map_err(|e| e.to_string())?;
    let mut embeddings = Vec::with_capacity(prompt.images.len());
    let mut spans = Vec::with_capacity(prompt.images.len());
    for image in &prompt.images {
        let start = Instant::now();
        embeddings.push(encoder.encode(&image.pixels)?);
        eprintln!(
            "image {}x{}: {} rows, encoded in {:.1} ms",
            image.pixels.width,
            image.pixels.height,
            image.span.rows,
            start.elapsed().as_secs_f64() * 1e3
        );
        spans.push(ImageSpan {
            offset: image.span.offset,
            rows: image.span.rows,
            identity: image.prefix_identity,
        });
    }
    let embed = |i: usize| {
        embeddings
            .get(i)
            .cloned()
            .ok_or_else(|| format!("no image {i}"))
    };

    let mut session = model.create_session(MAX_CONTEXT)?;
    let start = Instant::now();
    session.sync_with(&prompt.tokens, &spans, &embed, &[], &[])?;
    eprintln!(
        "prefill {} tokens: {:.1} ms",
        prompt.tokens.len(),
        start.elapsed().as_secs_f64() * 1e3
    );
    let mut tokens: Vec<TokenId> = prompt.tokens.clone();
    let mut generated = Vec::new();
    for _ in 0..count {
        let token = argmax(session.logits()?);
        if model.is_stop_token(token) {
            break;
        }
        generated.push(token);
        tokens.push(token);
        session.evaluate(token)?;
    }
    println!("{}", String::from_utf8_lossy(&model.decode(&generated)));
    if out_dir.is_empty() {
        return Ok(());
    }

    // Teacher-forced rows from the last image's closing token on.
    let last = prompt.images.last().ok_or("the prompt has no image")?.span;
    let first = (last.offset + last.rows + 1) as usize;
    let mut fresh = model.create_session(MAX_CONTEXT)?;
    fresh.sync_with(&tokens[..first], &spans, &embed, &[], &[])?;
    let mut positions = vec![(first - 1) as u32];
    let mut logits = fresh.logits()?.to_vec();
    let mut rows = Vec::new();
    fresh.evaluate_all(&tokens[first..], &mut rows)?;
    positions.extend((first..tokens.len()).map(|p| p as u32));
    logits.extend_from_slice(&rows);

    let dir = PathBuf::from(&out_dir);
    let mut file = Vec::with_capacity(16 + positions.len() * 4 + logits.len() * 4);
    for word in [
        0x474C_3447u32,
        1,
        positions.len() as u32,
        model.vocab_size(),
    ] {
        file.extend_from_slice(&word.to_le_bytes());
    }
    for p in &positions {
        file.extend_from_slice(&p.to_le_bytes());
    }
    for v in &logits {
        file.extend_from_slice(&v.to_le_bytes());
    }
    write(&dir.join("gufo.g4lg"), &file)?;
    let token_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    write(&dir.join("tokens.i32"), &token_bytes)?;
    for (i, image) in prompt.images.iter().enumerate() {
        write(&dir.join(format!("image{i}.rgb")), &image.pixels.pixels)?;
        eprintln!(
            "image{i}.rgb {} {}",
            image.pixels.width, image.pixels.height
        );
    }
    Ok(())
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
