//! Nemotron 3.5 Bolt prompt and chat (the extension's `RunNemotronPrompt`
//! and `RunNemotronChat` in `drivers.cpp`): one 32,768-token session,
//! autoregressive or with NVIDIA's LiLiCorr drafter, and images through its
//! vision projector at the start of the first user turn.

use std::io::{BufRead, Write};
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use gufo_core::sampling::{SamplerState, TokenId};
use gufo_core::text::ReasoningOptions;
use gufo_nemotron::chat_template::{
    ChatMessage, ChatRole, render_chat, resolve_chat_options, tokenize_chat,
};
use gufo_nemotron::engine::{InputRows, Model, ModelOptions, Session};
use gufo_nemotron::speculative::{DraftHistory, Frontier, speculative_step};
use gufo_nemotron::vision_encoder::{VisionEmbedding, VisionEncoder, VisionEncoderOptions};
use gufo_nemotron::vision_prompt::{ImageFraming, VisionInput, expand_images, find_image_framing};

use super::{PromptOptions, cpp_double, parse_effort};

/// Terminal sessions hold the prompt plus the requested output.
const CLI_CONTEXT: u32 = 32768;

/// Whether the model is a Nemotron-H MoE artifact this runtime supports.
pub(super) fn supports(options: &PromptOptions) -> bool {
    gufo_nemotron::runner::supports(Path::new(&options.model_path))
}

/// The extension's `Reasoning`: an effort selects the template's mode
/// without turning thinking on by itself.
fn reasoning(options: &PromptOptions) -> ReasoningOptions {
    let mut reasoning = ReasoningOptions::default();
    match options.reasoning_mode.as_str() {
        "on" => reasoning.enabled = Some(true),
        "off" => reasoning.enabled = Some(false),
        _ => {}
    }
    if options.reasoning_effort != "auto" {
        reasoning.effort = parse_effort(&options.reasoning_effort);
    }
    match options.preserve_thinking.as_str() {
        "on" => reasoning.preserve_thinking = Some(true),
        "off" => reasoning.preserve_thinking = Some(false),
        _ => {}
    }
    reasoning
}

/// The trunk and, for `--speculative dflash2`, NVIDIA's LiLiCorr drafter.
fn load(options: &PromptOptions) -> Result<Arc<Model>, String> {
    let backend = options.speculative_backend.as_str();
    if !backend.is_empty() && backend != "dflash2" {
        return Err("Nemotron drafts with NVIDIA's LiLiCorr drafter only: \
                    --speculative dflash2 --dflash-model DRAFTER.gguf"
            .into());
    }
    if backend == "dflash2" && options.dflash_model_path.is_empty() {
        return Err("--speculative dflash2 needs --dflash-model".into());
    }
    let mut model = Model::load(
        Path::new(&options.model_path),
        ModelOptions {
            max_context: CLI_CONTEXT,
        },
    )
    .map_err(|e| format!("load failed: {e}"))?;
    if backend == "dflash2" {
        model
            .load_drafter(Path::new(&options.dflash_model_path))
            .map_err(|e| format!("load failed: {e}"))?;
    }
    Ok(Arc::new(model))
}

fn prompt_tokens(
    model: &Model,
    options: &PromptOptions,
    history: &[ChatMessage],
) -> Result<Vec<i32>, String> {
    let ids: Vec<TokenId> = if options.use_chat_template {
        let (text, spans) = render_chat(history, &[], &resolve_chat_options(&reasoning(options)))?;
        tokenize_chat(model.tokenizer(), &text, &spans)
    } else {
        model.tokenizer().encode(&options.prompt_text, true)
    };
    ids.into_iter()
        .map(|id| i32::try_from(id).map_err(|_| "token outside the vocabulary".to_string()))
        .collect()
}

/// The images of `--image` with NVIDIA's projector (`--mmproj`). They go at
/// the start of the first user turn, as llama.cpp's mtmd-cli puts them, and
/// are encoded once.
struct Images {
    encoder: Option<VisionEncoder>,
    framing: Option<ImageFraming>,
    inputs: Vec<VisionInput>,
    embeddings: Vec<Arc<VisionEmbedding>>,
}

/// The position of an image's span with its embedding.
type ImageRows = Vec<(u32, Arc<VisionEmbedding>)>;

/// The chat prompt with each image expanded into its span, and the
/// positions of those spans with their embeddings.
fn expand_prompt(
    rendered: Vec<i32>,
    images: &mut Images,
    verbose: bool,
) -> Result<(Vec<i32>, ImageRows), String> {
    let (Some(encoder), Some(framing)) = (&images.encoder, &images.framing) else {
        return Ok((rendered, Vec::new()));
    };
    if images.inputs.is_empty() {
        return Ok((rendered, Vec::new()));
    }
    let ids: Vec<TokenId> = rendered.iter().map(|&t| t as TokenId).collect();
    let encode = images.embeddings.is_empty();
    let prompt = expand_images(
        encoder.config(),
        framing,
        &ids,
        &images.inputs,
        encoder.identity(),
        encode,
    )?;
    if encode {
        let start = Instant::now();
        let mut tokens = 0usize;
        for image in &prompt.images {
            let patches = image
                .patches
                .as_ref()
                .ok_or("an image has no preprocessed pixels")?;
            images
                .embeddings
                .push(encoder.encode(patches, &image.grid, None)?);
            tokens += image.rows as usize;
        }
        if verbose {
            eprintln!(
                "[Vision]: {} images, {tokens} tokens encoded in {} s",
                prompt.images.len(),
                cpp_double(start.elapsed().as_secs_f64())
            );
        }
    }
    let rows = prompt
        .images
        .iter()
        .zip(&images.embeddings)
        .map(|(image, embedding)| (image.offset, Arc::clone(embedding)))
        .collect();
    let tokens = prompt
        .tokens
        .iter()
        .map(|&t| i32::try_from(t).map_err(|_| "token outside the vocabulary".to_string()))
        .collect::<Result<_, _>>()?;
    Ok((tokens, rows))
}

/// Extends the session with `prompt` when it continues the executed
/// tokens, otherwise restarts it, then samples until a stop token or
/// `max_tokens` (`Generate`). With a drafter each round emits up to
/// `draft_tokens + 1` tokens. `rows` replace the embeddings of image
/// tokens.
fn generate(
    options: &PromptOptions,
    model: &Model,
    session: &mut Session,
    executed: &mut Vec<i32>,
    prompt: &[i32],
    mut reply: Option<&mut Vec<u8>>,
    rows: &[(u32, Arc<VisionEmbedding>)],
) -> Result<(), String> {
    let tokenizer = model.tokenizer();
    if prompt.is_empty() {
        return Err("the prompt produced no tokens".into());
    }
    if prompt.len() + options.max_tokens > session.max_context() as usize {
        return Err(format!(
            "prompt and output exceed the {}-token terminal context",
            session.max_context()
        ));
    }
    let continues = executed.len() < prompt.len() && prompt.starts_with(executed);
    if !continues {
        session
            .reset()
            .map_err(|e| format!("session reset failed: {e}"))?;
        executed.clear();
    }
    let mut frontier = Frontier {
        logits: vec![0.0; model.config().vocab_size as usize],
        ..Frontier::default()
    };
    let prefill_start = Instant::now();
    let suffix = &prompt[executed.len()..];
    let input_rows: Vec<InputRows<'_>> = rows
        .iter()
        .map(|(position, embedding)| InputRows {
            position: *position,
            rows: embedding.data(),
        })
        .collect();
    session
        .append_with_rows(suffix, &mut frontier.logits, &input_rows)
        .map_err(|e| format!("prefill failed: {e}"))?;
    let prefilled = suffix.len();
    executed.clear();
    executed.extend_from_slice(prompt);
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();
    if options.verbose {
        println!(
            "[Engine]: Nemotron 3.5 Bolt ROCm (gfx1151)\nPrompt tokens: {} ({prefilled} prefilled in {} s)\n--- Generation Output ---",
            prompt.len(),
            cpp_double(prefill_seconds)
        );
    }
    let history: Vec<TokenId> = prompt.iter().map(|&t| t as TokenId).collect();
    let mut sampler = SamplerState::new(options.sampling.clone(), &history)
        .map_err(|e| e.message().to_string())?;
    let generation_start = Instant::now();
    let draft_tokens = model
        .drafter()
        .map_or(0, |d| options.draft_tokens.min(d.config().slots()));
    let is_stop = |token: i32| u32::try_from(token).is_ok_and(|t| tokenizer.is_stop(t));
    let (mut generated, mut rounds, mut drafted, mut accepted) = (0usize, 0usize, 0usize, 0usize);
    let mut stdout = std::io::stdout();
    let mut draft_history = DraftHistory::default();
    while generated < options.max_tokens {
        let budget = (options.max_tokens - generated).min(draft_tokens as usize + 1) as u32;
        let round = speculative_step(
            session,
            &mut sampler,
            &mut frontier,
            &mut draft_history,
            budget,
            &is_stop,
            false,
        )
        .map_err(|e| format!("decode failed: {e}"))?;
        rounds += usize::from(round.drafted != 0);
        drafted += round.drafted as usize;
        accepted += round.accepted as usize;
        for (i, &token) in round.tokens.iter().enumerate() {
            if round.stop && i + 1 == round.tokens.len() {
                break;
            }
            let piece = tokenizer.token_bytes(token as TokenId);
            if let Some(reply) = reply.as_deref_mut() {
                reply.extend_from_slice(piece);
            }
            let _ = stdout.write_all(piece);
            let _ = stdout.flush();
            executed.push(token);
            generated += 1;
        }
        if round.stop {
            break;
        }
    }
    println!();
    if options.verbose && rounds > 0 {
        println!(
            "Draft rounds: {rounds}, drafted {drafted}, accepted {accepted} ({} per round)",
            cpp_double(accepted as f64 / rounds as f64)
        );
    }
    if options.verbose && generated > 0 {
        let seconds = generation_start.elapsed().as_secs_f64();
        println!(
            "Generated {generated} tokens in {} s ({} tok/s)",
            cpp_double(seconds),
            cpp_double(generated as f64 / seconds)
        );
    }
    Ok(())
}

/// `RunPrompt` of the extension's drivers: one prompt, or an interactive
/// conversation re-rendered every turn.
fn run(options: &PromptOptions, interactive: bool, load_start: Instant) -> Result<(), String> {
    if !options.image_paths.is_empty() && options.vision_model_path.is_empty() {
        return Err("--image needs NVIDIA's vision projector: --mmproj MMPROJ.gguf".into());
    }
    if !options.image_paths.is_empty() && !options.use_chat_template {
        return Err("--image needs chat framing, not --raw".into());
    }
    if interactive && !options.use_chat_template {
        return Err("interactive chat needs chat framing; use prompt --raw for raw text".into());
    }
    let model = load(options)?;
    eprintln!(
        "[ModelLoad]: {} s",
        cpp_double(load_start.elapsed().as_secs_f64())
    );
    let mut images = Images {
        encoder: None,
        framing: None,
        inputs: Vec::new(),
        embeddings: Vec::new(),
    };
    if !options.vision_model_path.is_empty() {
        images.encoder = Some(VisionEncoder::load(
            Path::new(&options.vision_model_path),
            model.config(),
            VisionEncoderOptions::default(),
        )?);
        images.framing = Some(find_image_framing(model.tokenizer())?);
        for path in &options.image_paths {
            let bytes = gufo_core::image::read_image_file(Path::new(path))
                .map_err(|e| e.message().to_string())?;
            images.inputs.push(VisionInput {
                bytes: Some(Arc::new(bytes)),
                low_detail: false,
            });
        }
    }
    // The images open the first user turn.
    let image_offsets = vec![0usize; images.inputs.len()];
    let mut session = model
        .create_session(CLI_CONTEXT)
        .map_err(|e| format!("session creation failed: {e}"))?;
    let mut history = Vec::new();
    if !options.system_prompt.is_empty() {
        history.push(ChatMessage {
            role: ChatRole::System,
            content: options.system_prompt.clone(),
            ..ChatMessage::default()
        });
    }
    let mut executed = Vec::new();
    if !interactive {
        history.push(ChatMessage {
            role: ChatRole::User,
            content: options.prompt_text.clone(),
            image_offsets: image_offsets.clone(),
            ..ChatMessage::default()
        });
        let (prompt, rows) = expand_prompt(
            prompt_tokens(&model, options, &history)?,
            &mut images,
            options.verbose,
        )?;
        return generate(
            options,
            &model,
            &mut session,
            &mut executed,
            &prompt,
            None,
            &rows,
        );
    }
    let thinking = resolve_chat_options(&reasoning(options)).enable_thinking;
    println!("=== Gufo Interactive Chat (Nemotron 3.5 Bolt) ===\nType 'exit' or Ctrl+D to quit.\n");
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    loop {
        print!(">>> User: ");
        let _ = std::io::stdout().flush();
        let mut line = Vec::new();
        match input.read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        if line.last() == Some(&b'\n') {
            line.pop();
        }
        let user = String::from_utf8_lossy(&line).into_owned();
        if user == "exit" || user == "quit" {
            break;
        }
        if user.is_empty() {
            continue;
        }
        let first_turn = !history.iter().any(|m| m.role == ChatRole::User);
        history.push(ChatMessage {
            role: ChatRole::User,
            content: user,
            image_offsets: if first_turn {
                image_offsets.clone()
            } else {
                Vec::new()
            },
            ..ChatMessage::default()
        });
        print!("<<< Assistant: ");
        let _ = std::io::stdout().flush();
        let mut reply = Vec::new();
        let (prompt, rows) = expand_prompt(
            prompt_tokens(&model, options, &history)?,
            &mut images,
            options.verbose,
        )?;
        generate(
            options,
            &model,
            &mut session,
            &mut executed,
            &prompt,
            Some(&mut reply),
            &rows,
        )?;
        let reply = String::from_utf8_lossy(&reply).into_owned();
        let mut response = ChatMessage {
            role: ChatRole::Assistant,
            ..ChatMessage::default()
        };
        const END: &str = "</think>";
        match reply.find(END) {
            Some(boundary) if thinking => {
                response.reasoning = reply[..boundary].to_string();
                response.content = reply[boundary + END.len()..].to_string();
            }
            _ => response.content = reply,
        }
        history.push(response);
        println!();
    }
    Ok(())
}

fn exit(result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Nemotron: {error}");
            ExitCode::from(1)
        }
    }
}

/// `RunNemotronPrompt`.
pub(super) fn run_prompt(options: &PromptOptions, load_start: Instant) -> ExitCode {
    exit(run(options, false, load_start))
}

/// `RunNemotronChat`.
pub(super) fn run_chat(options: &PromptOptions, load_start: Instant) -> ExitCode {
    exit(run(options, true, load_start))
}
