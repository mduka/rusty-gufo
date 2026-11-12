//! DeepSeek V4 Flash prompt and chat (`RunDeepSeekPrompt`,
//! `RunDeepSeekChat`): one 4096-token session, autoregressive or with
//! DSpark, whose cycles borrow the request sampler as the server's do.

use std::io::{BufRead, Write};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use gufo_core::sampling::SamplerState;
use gufo_deepseek_v4_flash::chat_template::{
    ChatMessage, ChatTemplateOptions, resolve_chat_options, validate_gguf_template,
};
use gufo_deepseek_v4_flash::dspark_sampler::DsparkSampler;
use gufo_deepseek_v4_flash::engine::Engine;
use gufo_deepseek_v4_flash::gguf::Model;
use gufo_deepseek_v4_flash::session::Session;
use gufo_deepseek_v4_flash::shape::MODEL_SHAPE_NAME;

use super::{PromptOptions, cpp_double, load_time, reasoning, token_trace};

/// `kDefaultContext`.
const CONTEXT: u32 = 4096;

/// Whether the model is a DeepSeek V4 Flash (`deepseek4`) artifact.
pub(super) fn supports(options: &PromptOptions) -> bool {
    gufo_deepseek_v4_flash::runner::supports(Path::new(&options.model_path))
}

/// The C++ refusal of image input, before anything loads.
fn refuse_images(options: &PromptOptions) -> Option<ExitCode> {
    (!options.image_paths.is_empty() || !options.vision_model_path.is_empty()).then(|| {
        eprintln!("DeepSeek does not support image input");
        ExitCode::from(2)
    })
}

/// `LoadDeepSeekModel` and the session `RunDeepSeek*` creates.
fn load(options: &PromptOptions, load_start: Instant) -> Option<(Engine, Session)> {
    let failed = |message: String| {
        eprintln!("{message}");
        load_time(load_start, false);
        None
    };
    match Model::open(Path::new(&options.model_path)) {
        Ok(model) => {
            if let Err(error) = validate_gguf_template(&model) {
                return failed(format!("Unsupported DeepSeek chat template: {error}"));
            }
        }
        Err(error) => {
            return failed(format!(
                "Error loading GGUF model '{}': {error}",
                options.model_path
            ));
        }
    }
    let spec = options.speculative_backend.as_str();
    let dspark = spec == "dspark";
    if !spec.is_empty() && !dspark {
        return failed("DeepSeek V4 Flash supports only --speculative dspark".into());
    }
    if dspark && options.dspark_model_path.is_empty() {
        return failed("--speculative dspark requires --dspark-model".into());
    }
    if dspark && options.min_draft_tokens != 1 {
        eprintln!("DSpark uses model-owned adaptive drafting; --min-draft-tokens is unsupported");
        return None;
    }
    let support = dspark.then(|| Path::new(&options.dspark_model_path));
    let mut engine = match Engine::open(Path::new(&options.model_path), support) {
        Ok(engine) => engine,
        Err(error) => return failed(format!("Error creating DeepSeek V4 Flash model: {error}")),
    };
    load_time(load_start, true);
    match engine.create_session(CONTEXT, !spec.is_empty()) {
        Ok(session) => Some((engine, session)),
        Err(error) => {
            eprintln!("DeepSeek session creation failed: {error}");
            None
        }
    }
}

/// The template options of the command line's reasoning flags.
fn chat_options(options: &PromptOptions) -> ChatTemplateOptions {
    resolve_chat_options(&reasoning(options))
}

/// `GenerateDeepSeekResponse`: syncs the prompt, streams the pieces until a
/// stop token or the budget, then the DSpark and verbose reports; appends
/// the reply's bytes to `reply`.
fn respond(
    options: &PromptOptions,
    engine: &mut Engine,
    session: &mut Session,
    prompt: &[i32],
    mut reply: Option<&mut Vec<u8>>,
) -> Result<(), String> {
    let vocab = std::sync::Arc::clone(&engine.vocab);
    let max_tokens = options.max_tokens;
    if prompt.is_empty() {
        return Err("DeepSeek V4 Flash prompt produced no tokens".into());
    }
    let context = CONTEXT as usize;
    if prompt.len() >= context || max_tokens >= context - prompt.len() {
        return Err("DeepSeek V4 Flash prompt and output exceed the 4096-token CLI context".into());
    }
    session
        .sync(engine, prompt)
        .map_err(|e| format!("DeepSeek V4 Flash prefill failed: {e}"))?;
    // Chat can extend an existing prefix. A new turn keeps its support KV
    // but discards the previous turn's deferred draw and controller history.
    session.begin_request();
    if options.verbose {
        print!(
            "[Engine]: DeepSeek V4 Flash ROCm (gfx1151)\nModel: {MODEL_SHAPE_NAME}\nPrompt tokens: {}\nMax tokens: {max_tokens}\n--- Generation Output ---\n",
            prompt.len()
        );
    }
    let history: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();
    let mut sampler =
        SamplerState::new(options.sampling.clone(), &history).map_err(|e| e.to_string())?;
    let reusable = reply.is_some();
    let mut emit = |token: i32| {
        let piece = vocab.token_text(token);
        if let Some(reply) = reply.as_deref_mut() {
            reply.extend_from_slice(&piece);
        }
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(&piece);
        let _ = out.flush();
    };
    let use_dspark = session.dspark_enabled();
    let start = Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    if use_dspark {
        let mut stop = false;
        while generated.len() < max_tokens && !stop {
            let mut bridge = (!options.sampling.can_use_unmodified_argmax())
                .then(|| DsparkSampler::new(&sampler));
            let emitted = session
                .dspark_step(
                    engine,
                    max_tokens - generated.len(),
                    options.draft_tokens,
                    bridge.as_mut(),
                    true,
                )
                .map_err(|e| format!("\nDeepSeek V4 Flash speculative decode failed: {e}"))?;
            if emitted.is_empty() {
                break;
            }
            if let Some(bridge) = &bridge {
                sampler.set_rng_state(bridge.rng_state());
            }
            for token in emitted {
                if vocab.is_stop(token) || generated.len() >= max_tokens {
                    stop = true;
                    break;
                }
                sampler.accept(token as u32).map_err(|e| e.to_string())?;
                emit(token);
                generated.push(token as u32);
            }
        }
    } else {
        while generated.len() < max_tokens {
            let logits = session.logits().ok_or_else(|| {
                "\nDeepSeek V4 Flash token selection failed: failed to copy DeepSeek V4 Flash logits"
                    .to_string()
            })?;
            let token = sampler.sample(logits).map_err(|e| e.to_string())? as i32;
            if vocab.is_stop(token) {
                break;
            }
            sampler.accept(token as u32).map_err(|e| e.to_string())?;
            emit(token);
            generated.push(token as u32);
            // A reusable chat session owns the complete emitted prefix, like
            // the server and DSpark: leaving the last token to the next bulk
            // prefill changes that call's boundary and numerics.
            if reusable || generated.len() < max_tokens {
                session
                    .eval(engine, token)
                    .map_err(|e| format!("\nDeepSeek V4 Flash decode failed: {e}"))?;
            }
        }
    }
    println!();
    if use_dspark {
        let s = session.dspark_stats();
        let ratio = |num: u64, den: u64| {
            cpp_double(if den != 0 {
                num as f64 / den as f64
            } else {
                0.0
            })
        };
        eprintln!(
            "[Speculative]: acceptance={} drafted={} accepted={} verification_steps={} skipped={} positional_acceptance={} positional_accepted={} full_block_rate={} full_blocks={} anchors={} verifier_rows={}",
            ratio(s.support_accepted, s.support_drafted),
            s.support_drafted,
            s.support_accepted,
            s.steps,
            s.skipped,
            ratio(s.positional_accepted, s.support_drafted),
            s.positional_accepted,
            ratio(s.full_blocks, s.steps),
            s.full_blocks,
            s.anchors,
            s.verifier_rows
        );
    }
    if options.verbose && !generated.is_empty() {
        let seconds = start.elapsed().as_secs_f64();
        println!(
            "Generated {} tokens on ROCm in {}s ({} tok/s)",
            generated.len(),
            cpp_double(seconds),
            cpp_double(generated.len() as f64 / seconds)
        );
        token_trace(&generated);
    }
    Ok(())
}

fn message(role: &str, content: String) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content,
        ..ChatMessage::default()
    }
}

/// `RunDeepSeekPrompt`.
pub(super) fn run_prompt(options: &PromptOptions, load_start: Instant) -> ExitCode {
    if let Some(code) = refuse_images(options) {
        return code;
    }
    let Some((mut engine, mut session)) = load(options, load_start) else {
        return ExitCode::from(1);
    };
    let prompt = if options.use_chat_template {
        let mut messages = Vec::new();
        if !options.system_prompt.is_empty() {
            messages.push(message("system", options.system_prompt.clone()));
        }
        messages.push(message("user", options.prompt_text.clone()));
        engine.encode_chat_messages(&messages, &[], &chat_options(options))
    } else {
        let mut tokens = Vec::new();
        engine
            .vocab
            .tokenize(options.prompt_text.as_bytes(), &mut tokens);
        tokens
    };
    match respond(options, &mut engine, &mut session, &prompt, None) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}

/// `RunDeepSeekChat`: the conversation is re-rendered every turn; the
/// session keeps the common prefix.
pub(super) fn run_chat(options: &PromptOptions, load_start: Instant) -> ExitCode {
    if let Some(code) = refuse_images(options) {
        return code;
    }
    if !options.use_chat_template {
        eprintln!("DeepSeek interactive chat requires chat framing; use prompt --raw for raw text");
        return ExitCode::from(1);
    }
    let Some((mut engine, mut session)) = load(options, load_start) else {
        return ExitCode::from(1);
    };
    let mut history = Vec::new();
    if !options.system_prompt.is_empty() {
        history.push(message("system", options.system_prompt.clone()));
    }
    let template = chat_options(options);
    println!("=== Gufo Interactive Chat (DeepSeek V4 Flash) ===\nType 'exit' or Ctrl+D to quit.\n");
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
        history.push(message("user", user));
        let prompt = engine.encode_chat_messages(&history, &[], &template);
        print!("<<< Assistant: ");
        let _ = std::io::stdout().flush();
        let mut reply = Vec::new();
        if let Err(error) = respond(
            options,
            &mut engine,
            &mut session,
            &prompt,
            Some(&mut reply),
        ) {
            eprintln!("{error}");
            return ExitCode::from(1);
        }
        let mut response = message("assistant", String::new());
        if template.enable_thinking {
            const END: &[u8] = b"</think>";
            match reply.windows(END.len()).position(|w| w == END) {
                Some(boundary) => {
                    response.reasoning_content =
                        String::from_utf8_lossy(&reply[..boundary]).into_owned();
                    response.content =
                        String::from_utf8_lossy(&reply[boundary + END.len()..]).into_owned();
                }
                None => response.reasoning_content = String::from_utf8_lossy(&reply).into_owned(),
            }
        } else {
            response.content = String::from_utf8_lossy(&reply).into_owned();
        }
        history.push(response);
        println!();
    }
    ExitCode::SUCCESS
}
