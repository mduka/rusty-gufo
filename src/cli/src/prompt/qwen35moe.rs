//! Qwen3.6 35B-A3B prompt and chat (the extension's `RunQwenPrompt` and
//! `RunQwenChat` in `q35/model/drivers.cpp`): the dense Qwen drivers over
//! the `qwen35moe` engine at the CLI's 4,096-token context, autoregressive,
//! with DFlash2 or with the MTP head of the target GGUF, and images through
//! the BF16 vision sidecar. Prompts follow the model's own Qwen3.6 template
//! when the GGUF carries it ([`Dialect::Qwen36`]), not the Qwen3.8 one the
//! extension renders; the extension's Ornith dialect belongs to
//! Ornith-1.5-122B and is not ported.

use std::io::{BufRead, Write};
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use gufo_core::qwen_tokenizer::QwenTokenizer;
use gufo_core::text::{ChatMessage, ChatRole};
use gufo_hip::Stream;
use gufo_qwen::chat_template::{
    ChatTemplateOptions, Dialect, encode_rendered, render, resolve_dialect_options,
};
use gufo_qwen::vision::{Encoder, Prompt as VisionPrompt};
use gufo_qwen35moe::Engine as Target;
use gufo_qwen35moe::dflash::draft::{DraftConfig, DraftContext};
use gufo_qwen35moe::dflash::executor::DraftScratch;
use gufo_qwen35moe::dflash::model::DFlashModel;
use gufo_qwen35moe::dflash::policy::DraftPolicy;
use gufo_qwen35moe::mtp::{DEFAULT_PRIME_CHUNK, MtpBackend, MtpDraftConfig, MtpExec, MtpModel};
use gufo_qwen35moe::verifier::{SpecExec, Verifier};

use super::{
    CONTEXT, PromptOptions, attach_images, cpp_double, load_qwen_vision, load_time, reasoning,
    system_message, token_trace,
};

/// Whether `options` name a `qwen35moe` artifact, which this driver claims
/// as the extension's provider does (`gguf:qwen35moe`).
pub(super) fn supports(options: &PromptOptions) -> bool {
    gufo_qwen35moe::runner::is_qwen35moe(Path::new(&options.model_path))
}

/// The template dialect of the GGUF at `path`: Qwen3.6's own for its pinned
/// template.
fn dialect(path: &str) -> Dialect {
    gufo_core::gguf::read_header(Path::new(path))
        .ok()
        .and_then(|(header, len)| {
            gufo_core::gguf::Gguf::parse_prefix(&header, len)
                .ok()
                .map(|gguf| Dialect::of_gguf(&gguf))
        })
        .unwrap_or_default()
}

/// The chat options of `o` in `dialect`. An effort has no prompt effect in
/// Qwen3.6's template; as on the other models it turns thinking on.
fn chat_options(o: &PromptOptions, dialect: Dialect) -> ChatTemplateOptions {
    resolve_dialect_options(dialect, &reasoning(o), o.add_vision_id)
}

/// `PrepareVision`: the image prompt of `messages` in `dialect`.
fn prepare_vision(
    o: &PromptOptions,
    dialect: Dialect,
    tokenizer: &QwenTokenizer,
    messages: &[ChatMessage],
    encoder: Option<&Encoder>,
) -> Result<Arc<VisionPrompt>, String> {
    let encoder = encoder
        .ok_or_else(|| "image input requires a matching --mmproj BF16 sidecar".to_string())?;
    gufo_qwen::vision::prepare(
        tokenizer,
        messages,
        &[],
        &chat_options(o, dialect),
        encoder.identity(),
        CONTEXT,
        None,
    )
    .map(Arc::new)
    .map_err(|e| e.message().to_string())
}

/// The target with its session and an optional draft.
struct Engine {
    target: Target,
    draft: Option<Draft>,
}

enum Draft {
    DFlash(Box<DFlashDraft>),
    Mtp(Box<MtpVerifier>),
}

/// The MTP head of the target GGUF and its verifier, on the target's
/// stream.
struct MtpVerifier {
    model: MtpModel,
    verifier: Verifier<MtpBackend>,
}

struct DFlashDraft {
    model: DFlashModel,
    scratch: DraftScratch,
    /// The scratch's and the draft session's stream, apart from the target's.
    stream: Stream,
    verifier: Verifier,
}

impl Engine {
    fn load(path: &str) -> Result<Self, String> {
        Ok(Self {
            target: Target::load(Path::new(path), CONTEXT)?,
            draft: None,
        })
    }

    /// `CreateQwenVerifier`: the requested draft, if any.
    fn attach_draft(&mut self, options: &PromptOptions) -> Result<(), String> {
        match options.speculative_backend.as_str() {
            "dflash2" => self.attach_dflash(options),
            "mtp" => self.attach_mtp(options),
            _ => Ok(()),
        }
    }

    /// `CreateQwenVerifier` for the MTP head: an adaptive draft length,
    /// batched verification, no frontier logits.
    fn attach_mtp(&mut self, options: &PromptOptions) -> Result<(), String> {
        let failed = |e: String| format!("Failed to initialize MTP backend: {e}");
        let stream = self.target.model().stream();
        let model = MtpModel::load(
            stream,
            Path::new(&options.mtp_model_path),
            self.target.weights(),
        )
        .map_err(failed)?;
        let backend = MtpBackend::new(
            stream,
            &model,
            MtpDraftConfig {
                max_context: CONTEXT,
                max_draft_tokens: options.draft_tokens,
                prime_chunk: DEFAULT_PRIME_CHUNK,
            },
        )
        .map_err(failed)?;
        let mut verifier = Verifier::with_draft(
            backend,
            options.min_draft_tokens,
            options.draft_tokens,
            true,
        );
        verifier.set_retain_frontier_logits(false);
        self.draft = Some(Draft::Mtp(Box::new(MtpVerifier { model, verifier })));
        Ok(())
    }

    /// `CreateQwenVerifier` for DFlash2: a fixed block length, batched
    /// verification, no frontier logits.
    fn attach_dflash(&mut self, options: &PromptOptions) -> Result<(), String> {
        let failed = |e: String| format!("Failed to initialize DFlash backend: {e}");
        let policy = DraftPolicy::parse(&options.draft_policy)?;
        let model = DFlashModel::load(
            self.target.model().stream(),
            Path::new(&options.dflash_model_path),
            self.target.weights(),
        )
        .map_err(failed)?;
        let config = DraftConfig::for_model(&model, CONTEXT, options.draft_tokens, policy)
            .map_err(failed)?;
        let stream = Stream::new().map_err(|e| failed(format!("HIP stream: {e}")))?;
        let mut scratch = DraftScratch::new(&stream, &model, CONTEXT).map_err(failed)?;
        let mut verifier = Verifier::new(
            &DraftContext {
                stream: &stream,
                model: &model,
                scratch: &mut scratch,
            },
            config,
            options.min_draft_tokens,
            options.draft_tokens,
        )
        .map_err(failed)?;
        verifier.set_retain_frontier_logits(false);
        self.draft = Some(Draft::DFlash(Box::new(DFlashDraft {
            model,
            scratch,
            stream,
            verifier,
        })));
        Ok(())
    }

    fn generate(
        &mut self,
        prompt: &[u32],
        options: &PromptOptions,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, String> {
        match &mut self.draft {
            Some(Draft::DFlash(draft)) => {
                let (model, session) = self.target.parts_mut();
                let mut exec = SpecExec {
                    model,
                    draft_model: &draft.model,
                    draft_scratch: &mut draft.scratch,
                    draft_stream: &draft.stream,
                };
                draft.verifier.generate(
                    &mut exec,
                    session,
                    prompt,
                    options.max_tokens,
                    &options.sampling,
                    on_token,
                )
            }
            Some(Draft::Mtp(draft)) => {
                let (model, session) = self.target.parts_mut();
                let mut exec = MtpExec {
                    model,
                    mtp: &draft.model,
                };
                draft.verifier.generate(
                    &mut exec,
                    session,
                    prompt,
                    options.max_tokens,
                    &options.sampling,
                    on_token,
                )
            }
            None => {
                let (model, session) = self.target.parts_mut();
                model.generate(
                    session,
                    prompt,
                    options.max_tokens,
                    &options.sampling,
                    on_token,
                )
            }
        }
    }

    /// Acceptance, drafted, accepted and verification steps of the draft.
    fn speculative_stats(&self) -> Option<(f32, u64, u64, u64)> {
        match &self.draft {
            Some(Draft::DFlash(draft)) => {
                let s = draft.verifier.stats();
                Some((
                    s.acceptance_rate(),
                    s.draft_tokens,
                    s.accepted_tokens,
                    s.verification_steps,
                ))
            }
            Some(Draft::Mtp(draft)) => {
                let s = draft.verifier.stats();
                Some((
                    s.acceptance_rate(),
                    s.draft_tokens,
                    s.accepted_tokens,
                    s.verification_steps,
                ))
            }
            None => None,
        }
    }

    /// The trunk's encoder for `--image` or `--mmproj` (`LoadQwenVision`).
    fn load_vision(&self, options: &PromptOptions) -> Result<Option<Arc<Encoder>>, String> {
        let config = self.target.config();
        load_qwen_vision(options, &config.architecture, config.hidden_size)
    }
}

/// `GenerateQwenGpuResponse`: streams the pieces, then the verbose report;
/// appends the reply's bytes to `reply`.
fn respond(
    engine: &mut Engine,
    tokenizer: &QwenTokenizer,
    options: &PromptOptions,
    prompt: &[u32],
    mut reply: Option<&mut Vec<u8>>,
) -> Result<(), String> {
    let mut generated = Vec::new();
    let start = Instant::now();
    let result = engine.generate(prompt, options, |token| {
        let piece = tokenizer.decode(&[token]);
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(&piece);
        let _ = out.flush();
        if let Some(reply) = reply.as_deref_mut() {
            reply.extend_from_slice(&piece);
        }
        generated.push(token);
        true
    });
    let seconds = start.elapsed().as_secs_f64();
    result?;
    println!();
    if options.verbose {
        if let Some((acceptance, drafted, accepted, steps)) = engine.speculative_stats() {
            eprintln!(
                "[Speculative]: acceptance={} drafted={drafted} accepted={accepted} verification_steps={steps}",
                cpp_double(f64::from(acceptance))
            );
        }
        let rate = if seconds > 0.0 {
            generated.len() as f64 / seconds
        } else {
            0.0
        };
        println!(
            "Generated {} tokens on GPU in {}s ({} tok/s)",
            generated.len(),
            cpp_double(seconds),
            cpp_double(rate)
        );
        token_trace(&generated);
    }
    Ok(())
}

/// The verbose engine report and the prompt tokens with their pieces.
fn print_prompt(
    engine: &Engine,
    tokenizer: &QwenTokenizer,
    options: &PromptOptions,
    prompt: &[u32],
) {
    let c = engine.target.config();
    let mut text = format!(
        "[Engine]: AMD Strix Halo gfx1151 GPU Executor\nModel: {} ({} layers, hidden={}, heads={})\nPrompt tokens ({}): ",
        c.architecture,
        c.num_layers,
        c.hidden_size,
        c.num_attention_heads,
        prompt.len()
    )
    .into_bytes();
    for &token in prompt {
        text.extend_from_slice(format!("[{token}: '").as_bytes());
        text.extend_from_slice(&tokenizer.decode(&[token]));
        text.extend_from_slice(b"'] ");
    }
    text.extend_from_slice(
        format!(
            "\nMax tokens: {}\n--- Generation Output ---\n",
            options.max_tokens
        )
        .as_bytes(),
    );
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(&text);
    let _ = out.flush();
}

/// `RunQwenPrompt` after parsing: renders the prompt, loads the model, then
/// prepares the images and generates.
pub(super) fn run_prompt(o: &PromptOptions, load_start: Instant) -> ExitCode {
    let dialect = dialect(&o.model_path);
    let mut rendered = o.prompt_text.clone();
    // A raw prompt is framing throughout; a templated one keeps its message
    // text out of control-token matching.
    let mut content_spans = Vec::new();
    let mut messages: Vec<ChatMessage> = Vec::new();
    if o.use_chat_template {
        messages.extend(system_message(o));
        messages.push(ChatMessage::new(ChatRole::User, o.prompt_text.clone()));
        match render(&messages, &[], &chat_options(o, dialect)) {
            Ok(text) => {
                rendered = text.text;
                content_spans = text.content_spans;
            }
            Err(_) => {
                eprintln!("Error formatting chat template.");
                return ExitCode::from(1);
            }
        }
    }
    let mut engine = match Engine::load(&o.model_path) {
        Ok(engine) => engine,
        Err(error) => {
            eprintln!("Error creating Qwen GPU executor: {error}");
            load_time(load_start, false);
            return ExitCode::from(1);
        }
    };
    load_time(load_start, true);
    let tokenizer = engine.target.weights().tokenizer.clone();
    let mut prompt = encode_rendered(&tokenizer, &rendered, 0, rendered.len(), &content_spans);
    let vision = (|| {
        let encoder = engine.load_vision(o)?;
        if o.image_paths.is_empty() {
            return Ok(None);
        }
        if let Some(last) = messages.last_mut() {
            attach_images(o, last)?;
        }
        let vision = prepare_vision(o, dialect, &tokenizer, &messages, encoder.as_deref())?;
        let (model, session) = engine.target.parts_mut();
        model.configure_vision(session, Some(vision.clone()), encoder)?;
        Ok::<_, String>(Some(vision))
    })();
    match vision {
        Ok(Some(vision)) => prompt = vision.tokens.clone(),
        Ok(None) => {}
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(1);
        }
    }
    if o.verbose {
        print_prompt(&engine, &tokenizer, o, &prompt);
    }
    let result = (|| {
        engine.attach_draft(o)?;
        respond(&mut engine, &tokenizer, o, &prompt, None)
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}

/// `RunQwenChat` after parsing: the terminal conversation, every turn
/// rendered from the whole history; the images join the first user turn.
pub(super) fn run_chat(o: &PromptOptions, load_start: Instant) -> ExitCode {
    let dialect = dialect(&o.model_path);
    let mut engine = match Engine::load(&o.model_path) {
        Ok(engine) => engine,
        Err(error) => {
            eprintln!("Error creating Qwen GPU executor: {error}");
            load_time(load_start, false);
            return ExitCode::from(1);
        }
    };
    let encoder = match engine
        .load_vision(o)
        .and_then(|encoder| engine.attach_draft(o).map(|()| encoder))
    {
        Ok(encoder) => encoder,
        Err(error) => {
            eprintln!("{error}");
            load_time(load_start, false);
            return ExitCode::from(1);
        }
    };
    if o.verbose {
        println!("[Engine]: AMD Strix Halo gfx1151 GPU Executor");
    }
    load_time(load_start, true);
    let tokenizer = engine.target.weights().tokenizer.clone();
    println!(
        "=== Gufo Interactive Chat ({}) ===\nType 'exit' or Ctrl+D to quit.\n",
        engine.target.config().architecture
    );
    let mut history: Vec<ChatMessage> = system_message(o).into_iter().collect();
    let mut first_turn = true;
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
        history.push(ChatMessage::new(ChatRole::User, user));
        let options = chat_options(o, dialect);
        let rendered = match render(&history, &[], &options) {
            Ok(rendered) => rendered,
            Err(_) => {
                eprintln!("Error formatting chat template.");
                return ExitCode::from(1);
            }
        };
        let mut prompt = encode_rendered(
            &tokenizer,
            &rendered.text,
            0,
            rendered.text.len(),
            &rendered.content_spans,
        );
        print!("<<< Assistant: ");
        let _ = std::io::stdout().flush();
        let vision = (|| {
            if std::mem::take(&mut first_turn)
                && let Some(last) = history.last_mut()
            {
                attach_images(o, last)?;
            }
            if o.image_paths.is_empty() {
                return Ok(None);
            }
            let vision = prepare_vision(o, dialect, &tokenizer, &history, encoder.as_deref())?;
            let (model, session) = engine.target.parts_mut();
            model.configure_vision(session, Some(vision.clone()), encoder.clone())?;
            Ok::<_, String>(Some(vision.tokens.clone()))
        })();
        match vision {
            Ok(Some(tokens)) => prompt = tokens,
            Ok(None) => {}
            Err(error) => {
                eprintln!("Generation failed: {error}");
                return ExitCode::from(1);
            }
        }
        let mut reply_bytes = Vec::new();
        if let Err(error) = respond(&mut engine, &tokenizer, o, &prompt, Some(&mut reply_bytes)) {
            eprintln!("Generation failed: {error}");
            return ExitCode::from(1);
        }
        println!();
        let reply_text = String::from_utf8_lossy(&reply_bytes).into_owned();
        let mut reply = ChatMessage::new(ChatRole::Assistant, String::new());
        if options.enable_thinking {
            const END: &str = "</think>";
            match reply_text.find(END) {
                Some(boundary) => {
                    reply.thought = reply_text[..boundary].to_owned();
                    reply.content = reply_text[boundary + END.len()..].to_owned();
                }
                None => reply.thought = reply_text,
            }
        } else {
            reply.content = reply_text;
        }
        history.push(reply);
    }
    ExitCode::SUCCESS
}
