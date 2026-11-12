//! Gemma 4 prompt and chat (the extension's `RunGemma4Prompt` and
//! `RunGemma4Chat` in `drivers.cpp`): one 4,096-token session outside the
//! serving pool, autoregressive or with the `gemma4-assistant` MTP drafter,
//! and images through the BF16 vision sidecar.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use gufo_core::sampling::{SamplerState, TokenId as SamplerToken};
use gufo_core::text::{ChatMessage, ChatRole};
use gufo_gemma4::chat_template::{ChatOptions, resolve_chat_options};
use gufo_gemma4::draft_policy::{DraftCalibrationScope, DraftPolicy};
use gufo_gemma4::engine::{DEFAULT_DRAFT_TOKENS, ImageSpan, Model, ModelOptions, Session};
use gufo_gemma4::provider_options::{PrefillWeights, image_tokens};
use gufo_gemma4::tokenizer::TokenId;
use gufo_gemma4::vision_encoder::VisionEncoder;
use gufo_gemma4::vision_prompt::{self, Prompt as VisionPrompt};
use gufo_hip::DeviceTensor;

use super::{PromptOptions, attach_images, cpp_double, load_time, reasoning};

/// `kDefaultContext`: the context of the model and its one session.
const CLI_CONTEXT: u32 = 4096;

/// Whether the model is a Gemma 4 target GGUF.
pub(super) fn supports(options: &PromptOptions) -> bool {
    gufo_gemma4::runner::supports(Path::new(&options.model_path))
}

/// The engine options of `options` (`LoadModel`), with Gemma's defaults for
/// whatever was not given.
fn model_options(options: &PromptOptions, context: u32) -> Result<ModelOptions, String> {
    let backend = options.speculative_backend.as_str();
    let mtp = backend == "mtp";
    if !backend.is_empty() && !mtp {
        return Err("Gemma 4 supports only --speculative mtp".into());
    }
    if mtp && options.mtp_model_path.is_empty() {
        return Err("--speculative mtp requires --mtp-model".into());
    }
    image_tokens(options.image_tokens)?;
    Ok(ModelOptions {
        max_context: context,
        mtp_model_path: mtp.then(|| options.mtp_model_path.clone().into()),
        draft_tokens: if options.draft_tokens_supplied {
            options.draft_tokens
        } else {
            DEFAULT_DRAFT_TOKENS
        },
        min_draft_tokens: options.min_draft_tokens,
        draft_policy: DraftPolicy::parse(&options.draft_policy)?,
        draft_calibration: DraftCalibrationScope::parse(&options.draft_calibration)?,
        draft_trace_path: (!options.draft_trace.is_empty())
            .then(|| options.draft_trace.clone().into()),
        prefill_weights: PrefillWeights::parse(&options.prefill_weights)?,
        ..ModelOptions::default()
    })
}

fn load_model(options: &PromptOptions, load_start: Instant) -> Result<Arc<Model>, String> {
    let model_options = model_options(options, CLI_CONTEXT)?;
    match Model::load(Path::new(&options.model_path), model_options) {
        Ok(model) => {
            load_time(load_start, true);
            Ok(model)
        }
        Err(error) => {
            load_time(load_start, false);
            Err(format!("loading the model failed: {error}"))
        }
    }
}

/// The vision sidecar, when images are attached or `--mmproj` is given
/// (`LoadVision`).
fn load_vision(options: &PromptOptions, model: &Model) -> Result<Option<VisionEncoder>, String> {
    if options.image_paths.is_empty() && options.vision_model_path.is_empty() {
        return Ok(None);
    }
    let explicit = (!options.vision_model_path.is_empty())
        .then(|| Path::new(options.vision_model_path.as_str()));
    match VisionEncoder::open(
        Path::new(&options.model_path),
        explicit,
        model.config().hidden_size,
    )? {
        Some(encoder) => Ok(Some(encoder)),
        None => Err("image input requires a matching --mmproj BF16 sidecar".into()),
    }
}

/// A rendered conversation with its images, encoded on first use.
struct Prompt {
    tokens: Vec<TokenId>,
    spans: Vec<ImageSpan>,
    vision: Option<VisionPrompt>,
}

/// `RenderPrompt`.
fn render_prompt(
    model: &Model,
    encoder: Option<&VisionEncoder>,
    history: &[ChatMessage],
    options: &ChatOptions,
    image_tokens: u32,
) -> Result<Prompt, String> {
    let prepared = vision_prompt::prepare(
        model.tokenizer(),
        history,
        &[],
        options,
        encoder.map_or("", |e| e.identity()),
        model.max_context(),
        image_tokens,
    )
    .map_err(|e| e.message().to_string())?;
    let spans = prepared
        .images
        .iter()
        .map(|image| ImageSpan {
            offset: image.span.offset,
            rows: image.span.rows,
            identity: image.prefix_identity,
        })
        .collect();
    Ok(Prompt {
        tokens: prepared.tokens.clone(),
        spans,
        vision: Some(prepared),
    })
}

/// Generates one reply after `input` (`Generate`); `reply` receives the
/// decoded text. The session keeps what an earlier turn left, and the sync
/// reuses its common prefix.
fn generate(
    options: &PromptOptions,
    model: &Model,
    session: &mut Session,
    encoder: Option<&VisionEncoder>,
    input: &Prompt,
    mut reply: Option<&mut Vec<u8>>,
) -> Result<(), String> {
    let prompt = &input.tokens;
    if prompt.is_empty() {
        return Err("the prompt produced no tokens".into());
    }
    if prompt.len() + options.max_tokens > session.context_size() as usize {
        return Err(format!(
            "prompt and output exceed the {}-token CLI context",
            session.context_size()
        ));
    }
    let prefill_start = Instant::now();
    let embeddings = std::cell::RefCell::new(BTreeMap::<usize, Arc<DeviceTensor<f32>>>::new());
    let embed = |index: usize| -> Result<Arc<DeviceTensor<f32>>, String> {
        if let Some(embedding) = embeddings.borrow().get(&index) {
            return Ok(Arc::clone(embedding));
        }
        let encoder = encoder.ok_or("Gemma 4 image input has no encoder")?;
        let image = input
            .vision
            .as_ref()
            .and_then(|prompt| prompt.images.get(index))
            .ok_or("Gemma 4 image index outside the prompt")?;
        let embedding = encoder.encode(&image.pixels)?;
        embeddings
            .borrow_mut()
            .insert(index, Arc::clone(&embedding));
        Ok(embedding)
    };
    session
        .sync_with(prompt, &input.spans, &embed, &[], &[])
        .map_err(|e| format!("prefill failed: {e}"))?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();
    if options.verbose {
        println!(
            "[Engine]: Gemma 4 ROCm (gfx1151)\nModel: {}\nPrompt tokens: {}\n--- Generation Output ---",
            model.model_name(),
            prompt.len()
        );
    }
    let history: Vec<SamplerToken> = prompt.iter().map(|&t| t as SamplerToken).collect();
    let mut sampler = SamplerState::new(options.sampling.clone(), &history)
        .map_err(|e| e.message().to_string())?;
    let decode_start = Instant::now();
    let mut generated = 0usize;
    let mut stop = false;
    let mut stdout = std::io::stdout();
    while generated < options.max_tokens && !stop {
        let step = session
            .decode_step(options.max_tokens - generated, &mut sampler, true)
            .map_err(|e| format!("decode failed: {e}"))?;
        for &token in &step.tokens {
            if model.is_stop_token(token) {
                stop = true;
                break;
            }
            let piece = model.token_text(token);
            if let Some(reply) = reply.as_deref_mut() {
                reply.extend_from_slice(&piece);
            }
            let _ = stdout.write_all(&piece);
            let _ = stdout.flush();
            generated += 1;
        }
        stop = stop || step.stop || step.tokens.is_empty();
    }
    println!();
    if options.verbose && generated > 0 {
        let seconds = decode_start.elapsed().as_secs_f64();
        println!(
            "Prefill {} tokens in {} s ({} tok/s)\nGenerated {generated} tokens in {} s ({} tok/s)",
            prompt.len(),
            cpp_double(prefill_seconds),
            cpp_double(prompt.len() as f64 / prefill_seconds),
            cpp_double(seconds),
            cpp_double(generated as f64 / seconds)
        );
        if model.has_mtp() {
            let stats = session.statistics();
            let acceptance = if stats.drafted != 0 {
                stats.accepted as f64 / stats.drafted as f64
            } else {
                0.0
            };
            println!(
                "[Speculative]: cycles={} drafted={} accepted={} acceptance={} \
                 lookup_proposed={} lookup_accepted={} siblings={} siblings_accepted={}",
                stats.cycles,
                stats.drafted,
                stats.accepted,
                cpp_double(acceptance),
                stats.copied,
                stats.copied_accepted,
                stats.siblings,
                stats.siblings_accepted
            );
        }
    }
    Ok(())
}

/// `RunGemma4Prompt`.
fn run_once(options: &PromptOptions, load_start: Instant) -> Result<(), String> {
    let model = load_model(options, load_start)?;
    let mut session = model
        .create_session(CLI_CONTEXT)
        .map_err(|e| format!("session creation failed: {e}"))?;
    let encoder = load_vision(options, &model)?;
    let prompt = if options.use_chat_template {
        let mut messages = Vec::new();
        if !options.system_prompt.is_empty() {
            messages.push(ChatMessage::new(
                ChatRole::System,
                options.system_prompt.clone(),
            ));
        }
        let mut user = ChatMessage::new(ChatRole::User, options.prompt_text.clone());
        attach_images(options, &mut user)?;
        messages.push(user);
        render_prompt(
            &model,
            encoder.as_ref(),
            &messages,
            &resolve_chat_options(&reasoning(options)),
            image_tokens(options.image_tokens)?,
        )?
    } else if !options.image_paths.is_empty() {
        return Err("--image requires chat framing".into());
    } else {
        Prompt {
            tokens: model
                .tokenizer()
                .encode(options.prompt_text.as_bytes(), true, true),
            spans: Vec::new(),
            vision: None,
        }
    };
    generate(
        options,
        &model,
        &mut session,
        encoder.as_ref(),
        &prompt,
        None,
    )
}

/// `RunGemma4Chat`: the conversation is re-rendered every turn.
fn run_interactive(options: &PromptOptions, load_start: Instant) -> Result<(), String> {
    if !options.use_chat_template {
        return Err("interactive chat requires chat framing; use prompt --raw for raw text".into());
    }
    let model = load_model(options, load_start)?;
    let mut session = model
        .create_session(CLI_CONTEXT)
        .map_err(|e| format!("session creation failed: {e}"))?;
    let encoder = load_vision(options, &model)?;
    let chat_options = resolve_chat_options(&reasoning(options));
    let budget = image_tokens(options.image_tokens)?;
    let mut history = Vec::new();
    if !options.system_prompt.is_empty() {
        history.push(ChatMessage::new(
            ChatRole::System,
            options.system_prompt.clone(),
        ));
    }
    println!("=== Gufo Interactive Chat (Gemma 4) ===\nType 'exit' or Ctrl+D to quit.\n");
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
        let text = String::from_utf8_lossy(&line).into_owned();
        if text == "exit" || text == "quit" {
            break;
        }
        if text.is_empty() {
            continue;
        }
        let mut user = ChatMessage::new(ChatRole::User, text);
        // The images open the first user turn.
        if !history
            .iter()
            .any(|m: &ChatMessage| m.role == ChatRole::User)
        {
            attach_images(options, &mut user)?;
        }
        history.push(user);
        let prompt = render_prompt(&model, encoder.as_ref(), &history, &chat_options, budget)?;
        print!("<<< Assistant: ");
        let _ = std::io::stdout().flush();
        let mut reply = Vec::new();
        generate(
            options,
            &model,
            &mut session,
            encoder.as_ref(),
            &prompt,
            Some(&mut reply),
        )?;
        // The template drops thinking channels from model history itself.
        history.push(ChatMessage::new(
            ChatRole::Assistant,
            String::from_utf8_lossy(&reply).into_owned(),
        ));
        println!();
    }
    Ok(())
}

fn exit(result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Gemma 4: {error}");
            ExitCode::from(1)
        }
    }
}

/// `RunGemma4Prompt`.
pub(super) fn run_prompt(options: &PromptOptions, load_start: Instant) -> ExitCode {
    exit(run_once(options, load_start))
}

/// `RunGemma4Chat`.
pub(super) fn run_chat(options: &PromptOptions, load_start: Instant) -> ExitCode {
    exit(run_interactive(options, load_start))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(args: &[&str]) -> PromptOptions {
        PromptOptions {
            speculative_backend: String::new(),
            ..PromptOptions::default()
        }
        .with(args)
    }

    impl PromptOptions {
        fn with(mut self, args: &[&str]) -> Self {
            for pair in args.chunks(2) {
                match pair {
                    ["speculative", value] => self.speculative_backend = (*value).into(),
                    ["mtp", value] => self.mtp_model_path = (*value).into(),
                    ["policy", value] => self.draft_policy = (*value).into(),
                    ["calibration", value] => self.draft_calibration = (*value).into(),
                    ["trace", value] => self.draft_trace = (*value).into(),
                    ["prefill", value] => self.prefill_weights = (*value).into(),
                    ["tokens", value] => {
                        self.draft_tokens = value.parse().expect("count");
                        self.draft_tokens_supplied = true;
                    }
                    ["images", value] => self.image_tokens = Some(value.parse().expect("count")),
                    _ => panic!("unknown test option {pair:?}"),
                }
            }
            self
        }
    }

    #[test]
    fn engine_options_follow_gemma_defaults() {
        let o = model_options(&options(&[]), 4096).expect("options");
        assert_eq!(o.max_context, 4096);
        assert_eq!(o.mtp_model_path, None);
        assert_eq!(o.draft_tokens, DEFAULT_DRAFT_TOKENS);
        assert_eq!(o.draft_policy, DraftPolicy::Calibrated);
        assert_eq!(o.draft_calibration, DraftCalibrationScope::Shared);
        assert_eq!(o.prefill_weights, PrefillWeights::Off);
        assert_eq!(o.draft_trace_path, None);
    }

    #[test]
    fn engine_options_take_the_given_values() {
        let o = model_options(
            &options(&[
                "speculative",
                "mtp",
                "mtp",
                "mtp.gguf",
                "policy",
                "fixed",
                "calibration",
                "request",
                "trace",
                "trace.jsonl",
                "prefill",
                "decoded",
                "tokens",
                "4",
            ]),
            4096,
        )
        .expect("options");
        assert_eq!(o.mtp_model_path, Some("mtp.gguf".into()));
        assert_eq!(o.draft_tokens, 4);
        assert_eq!(o.draft_policy, DraftPolicy::Fixed);
        assert_eq!(o.draft_calibration, DraftCalibrationScope::Request);
        assert_eq!(o.draft_trace_path, Some("trace.jsonl".into()));
        assert_eq!(o.prefill_weights, PrefillWeights::Decoded);
    }

    #[test]
    fn engine_options_refuse_invalid_values() {
        for args in [
            &["speculative", "dflash2"][..],
            &["speculative", "mtp"],
            &["policy", "adaptive"],
            &["calibration", "global"],
            &["prefill", "dense"],
            &["images", "100"],
        ] {
            assert!(model_options(&options(args), 4096).is_err(), "{args:?}");
        }
        assert!(model_options(&options(&["images", "560"]), 4096).is_ok());
    }
}
