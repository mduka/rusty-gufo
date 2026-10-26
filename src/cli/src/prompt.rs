//! `gufo prompt` and `gufo chat` on the Rust engines, a port of
//! `src/cli/prompt/prompt.cpp` for the GPU path: dense Qwen autoregressive,
//! with DFlash2 or with an MTP head, Flash-Next autoregressive or with MTP,
//! images on the 27B and Flash-Next, DeepSeek V4 Flash autoregressive or with DSpark
//! (`deepseek`), the chat template, the verbose report and the terminal
//! chat. The C++ `--cpu` fallback is not offered.

use std::cell::RefCell;
use std::io::{BufRead, Write};
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use gufo_core::qwen_tokenizer::QwenTokenizer;
use gufo_core::sampling::{
    SamplingConfig, SamplingOverrides, resolve_text_sampling, text_preset_of,
};
use gufo_core::text::{ChatMessage, ChatRole, ImagePart, ReasoningEffort, ReasoningOptions};
use gufo_hip::Stream;
use gufo_qwen::chat_template::{encode_rendered, render, resolve_options};
use gufo_qwen::vision::{Encoder, Prompt as VisionPrompt};
use gufo_qwen35::dflash::draft::{DraftConfig, DraftContext};
use gufo_qwen35::dflash::executor::DraftScratch;
use gufo_qwen35::dflash::model::DFlashModel;
use gufo_qwen35::dflash::policy::DraftPolicy;
use gufo_qwen35::mtp::{MtpDraft, MtpModel};
use gufo_qwen35::verifier::{SpecExec, Verifier};
use gufo_qwen35::{Engine as Target, Speculative};
use sha2::{Digest, Sha256};

mod deepseek;

use crate::arg_parser::ArgParser;
use crate::bench::cpp_double;
use crate::sampling_options::{register_sampling_options, sampling_options_supplied};

/// `kDefaultContext`, the context `QwenGpuExecutor::CreateFromGguf` takes.
const CONTEXT: u32 = 4096;

/// `PromptOptions`.
struct PromptOptions {
    model_path: String,
    vision_model_path: String,
    image_paths: Vec<String>,
    add_vision_id: bool,
    prompt_text: String,
    prompt_file: String,
    system_prompt: String,
    max_tokens: usize,
    sampling: SamplingConfig,
    /// The sampling options given on the command line; the others follow
    /// the model's recommended defaults.
    sampling_supplied: SamplingOverrides,
    reasoning_mode: String,
    reasoning_effort: String,
    preserve_thinking: String,
    display_prompt: bool,
    use_chat_template: bool,
    verbose: bool,
    speculative_backend: String,
    mtp_model_path: String,
    dflash_model_path: String,
    draft_policy: String,
    dspark_model_path: String,
    draft_tokens: u32,
    min_draft_tokens: u32,
}

impl Default for PromptOptions {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            vision_model_path: String::new(),
            image_paths: Vec::new(),
            add_vision_id: false,
            prompt_text: String::new(),
            prompt_file: String::new(),
            system_prompt: String::new(),
            max_tokens: 128,
            sampling: SamplingConfig::default(),
            sampling_supplied: SamplingOverrides::default(),
            reasoning_mode: "auto".into(),
            reasoning_effort: "auto".into(),
            preserve_thinking: "auto".into(),
            display_prompt: true,
            use_chat_template: true,
            verbose: false,
            speculative_backend: String::new(),
            mtp_model_path: String::new(),
            dflash_model_path: String::new(),
            draft_policy: String::new(),
            dspark_model_path: String::new(),
            draft_tokens: 7,
            min_draft_tokens: 1,
        }
    }
}

/// `std::from_chars` for a positive count.
fn positive_count(value: &str) -> Option<u32> {
    if value.starts_with('+') {
        return None;
    }
    value.parse().ok().filter(|&count| count != 0)
}

/// `RegisterTextOptions`.
/// `-p` and the positionals of `prompt` share `text`, as in C++.
fn register<'a>(
    parser: &mut ArgParser<'a>,
    options: &'a mut PromptOptions,
    text: &'a RefCell<String>,
    speculative_explicit: &'a mut bool,
    command: &str,
) {
    let PromptOptions {
        model_path,
        vision_model_path,
        image_paths,
        add_vision_id,
        prompt_text: _,
        prompt_file,
        system_prompt,
        max_tokens,
        sampling,
        sampling_supplied: _,
        reasoning_mode,
        reasoning_effort,
        preserve_thinking,
        display_prompt,
        use_chat_template,
        verbose,
        speculative_backend,
        mtp_model_path,
        dflash_model_path,
        draft_policy,
        dspark_model_path,
        draft_tokens,
        min_draft_tokens,
    } = options;
    parser.add_option(
        "-m",
        "--model",
        "PATH",
        "Path to GGUF model file",
        "Model",
        model_path,
    );
    // RegisterImageOptions
    parser
        .add_flag(
            "",
            "--add-vision-id",
            "Prefix image inputs with Picture N: in the chat template",
            "Prompt",
            add_vision_id,
        )
        .add_option(
            "",
            "--mmproj",
            "PATH",
            "Qwen BF16 vision sidecar (auto-discovered beside model)",
            "Model",
            vision_model_path,
        )
        .add_custom_option(
            "",
            "--image",
            "PATH",
            "PNG/JPEG attached before text in the first user turn; repeat for multiple images",
            "Prompt",
            move |_, value| {
                if value.is_empty() {
                    return Err("--image requires a file path".into());
                }
                image_paths.push(value.to_owned());
                Ok(())
            },
        );
    let prompt = command == "prompt";
    if prompt {
        parser.add_custom_option(
            "-p",
            "--prompt",
            "TEXT",
            "Direct input prompt text",
            "Prompt",
            move |_, value| {
                *text.borrow_mut() = value.to_owned();
                Ok(())
            },
        );
        parser.add_option(
            "-f",
            "--file",
            "PATH",
            "File path to read input prompt text from",
            "Prompt",
            prompt_file,
        );
    }
    parser.add_option(
        "",
        "--system",
        "PROMPT",
        "Optional system role instructions prepended to the prompt",
        "Prompt",
        system_prompt,
    );
    if prompt {
        parser
            .add_inverse_flag(
                "",
                "--raw",
                "Disable chat template framing and pass raw tokens",
                "Prompt",
                use_chat_template,
            )
            .add_inverse_flag(
                "",
                "--no-display-prompt",
                "Suppress echoing the prompt before generated response",
                "Prompt",
                display_prompt,
            );
    }
    parser.add_option(
        "-n",
        "--max-tokens",
        "N",
        "Maximum number of new tokens to generate (default: 128)",
        "Sampling",
        max_tokens,
    );
    register_sampling_options(parser, sampling, "Sampling", true, true);
    parser
        .add_option(
            "",
            "--think",
            "MODE",
            "Reasoning mode: on, off, or auto (default: model)",
            "Reasoning",
            reasoning_mode,
        )
        .add_option(
            "",
            "--reasoning-effort",
            "LEVEL",
            "Effort: auto, minimal, low, medium, high, xhigh, or max",
            "Reasoning",
            reasoning_effort,
        )
        .add_option(
            "",
            "--preserve-thinking",
            "MODE",
            "Replay prior reasoning: on, off, or auto",
            "Reasoning",
            preserve_thinking,
        );
    parser
        .add_custom_option(
            "",
            "--speculative",
            "MODE",
            "Draft backend: dspark (DeepSeek V4 Flash), dflash2, mtp, or off",
            "Speculative",
            move |_, value| {
                *speculative_explicit = true;
                match value {
                    "off" => speculative_backend.clear(),
                    "dspark" | "dflash2" | "mtp" => *speculative_backend = value.to_owned(),
                    _ => return Err(format!("Unknown speculative backend: {value}")),
                }
                Ok(())
            },
        )
        .add_option(
            "",
            "--dflash-model",
            "PATH",
            "Path to Qwen DFlash2 GGUF file",
            "Speculative",
            dflash_model_path,
        )
        .add_option(
            "",
            "--draft-policy",
            "POLICY",
            "DFlash2 block length: fixed or adaptive (default: adaptive)",
            "Speculative",
            draft_policy,
        )
        .add_option(
            "",
            "--dspark-model",
            "PATH",
            "Path to the DeepSeek V4 Flash DSpark support GGUF file",
            "Speculative",
            dspark_model_path,
        )
        .add_option(
            "",
            "--mtp-model",
            "PATH",
            "Path to quantized Qwen MTP draft head GGUF file",
            "Speculative",
            mtp_model_path,
        )
        .add_custom_option(
            "-d",
            "--draft-tokens",
            "N",
            "Maximum speculative draft tokens evaluated per step (default: 7)",
            "Speculative",
            move |_, value| {
                *draft_tokens = positive_count(value)
                    .ok_or_else(|| format!("Invalid integer for draft-tokens: {value}"))?;
                Ok(())
            },
        )
        .add_custom_option(
            "",
            "--min-draft-tokens",
            "N",
            "Adaptive draft floor (default: 1)",
            "Speculative",
            move |_, value| {
                *min_draft_tokens = positive_count(value)
                    .ok_or_else(|| format!("Invalid integer for min-draft-tokens: {value}"))?;
                Ok(())
            },
        )
        .add_flag(
            "-v",
            "--verbose",
            "Print detailed timing, latency breakdown, and tok/s metrics",
            "General",
            verbose,
        );
    if prompt {
        // `JoinPositionals`.
        parser.set_positional_handler(move |value| {
            let mut text = text.borrow_mut();
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(value);
            Ok(())
        });
    }
}

/// `PrintTextHelp`.
fn help_text(command: &str) -> String {
    let mut options = PromptOptions::default();
    let text = RefCell::new(String::new());
    let mut explicit = false;
    let description = if command == "chat" {
        "Interactive conversation."
    } else {
        "Execute one prompt request and exit."
    };
    let mut parser = ArgParser::new(format!("gufo {command}"), description);
    register(&mut parser, &mut options, &text, &mut explicit, command);
    parser.format_help()
}

pub fn print_help(command: &str) {
    print!("{}", help_text(command));
    let _ = std::io::stdout().flush();
}

fn parse_effort(value: &str) -> Option<ReasoningEffort> {
    match value {
        "minimal" => Some(ReasoningEffort::Minimal),
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::XHigh),
        "max" => Some(ReasoningEffort::Max),
        _ => None,
    }
}

/// `PromptReasoningOptions`.
/// `ResolvePromptSampling`: the model's recommended sampling for the
/// thinking mode under the options given on the command line.
fn resolve_sampling(options: &mut PromptOptions) {
    options.sampling = resolve_text_sampling(
        text_preset_of(Path::new(&options.model_path)),
        reasoning(options).enabled,
        &options.sampling,
        &options.sampling_supplied,
    );
}

fn reasoning(options: &PromptOptions) -> ReasoningOptions {
    let mut reasoning = ReasoningOptions::default();
    match options.reasoning_mode.as_str() {
        "on" => reasoning.enabled = Some(true),
        "off" => reasoning.enabled = Some(false),
        _ => {}
    }
    if options.reasoning_effort != "auto" {
        reasoning.effort = parse_effort(&options.reasoning_effort);
        reasoning.enabled = Some(true);
    }
    match options.preserve_thinking.as_str() {
        "on" => reasoning.preserve_thinking = Some(true),
        "off" => reasoning.preserve_thinking = Some(false),
        _ => {}
    }
    reasoning
}

/// `ParseTextOptions`: `Ok(None)` for help, `Err` with the C++ message
/// (possibly empty).
fn parse(args: &[String], command: &str) -> Result<Option<PromptOptions>, String> {
    let mut o = PromptOptions::default();
    let text = RefCell::new(String::new());
    let mut explicit = false;
    let supplied;
    {
        let mut parser = ArgParser::new("gufo prompt", "Execute one prompt request and exit.");
        register(&mut parser, &mut o, &text, &mut explicit, command);
        parser.parse(args)?;
        if parser.is_help_requested() {
            return Ok(None);
        }
        supplied = sampling_options_supplied(&parser);
    }
    o.sampling_supplied = supplied;
    o.prompt_text = text.into_inner();
    o.sampling.validate().map_err(|e| e.message().to_owned())?;
    if (!o.image_paths.is_empty() || !o.vision_model_path.is_empty()) && !o.use_chat_template {
        return Err("image input requires ROCm and chat framing".into());
    }
    if !explicit && !o.dspark_model_path.is_empty() {
        o.speculative_backend = "dspark".into();
    }
    if o.speculative_backend == "dflash2" && o.dflash_model_path.is_empty() {
        return Err("DFlash2 requires --dflash-model".into());
    }
    if o.speculative_backend == "mtp" && o.mtp_model_path.is_empty() {
        return Err("MTP requires --mtp-model".into());
    }
    if o.draft_tokens == 0 || o.min_draft_tokens == 0 || o.min_draft_tokens > o.draft_tokens {
        return Err("min-draft-tokens cannot exceed draft-tokens".into());
    }
    if !o.draft_policy.is_empty()
        && ((o.draft_policy != "fixed" && o.draft_policy != "adaptive")
            || o.speculative_backend != "dflash2")
    {
        return Err("--draft-policy requires DFlash2 and fixed or adaptive".into());
    }
    if o.min_draft_tokens != 1 && o.speculative_backend == "dflash2" {
        return Err(
            "DFlash2 requires --min-draft-tokens 1; bound blocks with --draft-tokens".into(),
        );
    }
    if !matches!(o.reasoning_mode.as_str(), "on" | "off" | "auto") {
        return Err("--think must be on, off, or auto".into());
    }
    if o.reasoning_effort != "auto" && parse_effort(&o.reasoning_effort).is_none() {
        return Err(
            "--reasoning-effort must be auto, minimal, low, medium, high, xhigh, or max".into(),
        );
    }
    if o.reasoning_mode == "off" && o.reasoning_effort != "auto" {
        return Err("--reasoning-effort cannot be set while --think is off".into());
    }
    if !matches!(o.preserve_thinking.as_str(), "on" | "off" | "auto") {
        return Err("--preserve-thinking must be on, off, or auto".into());
    }
    Ok(Some(o))
}

/// Whether dense Qwen runs `options`' draft: none, DFlash2 or an MTP head.
/// The dense engine takes every artifact Flash-Next and DeepSeek do not;
/// loading reports the ones it cannot run, as does a host without a device.
fn dense_backend(options: &PromptOptions) -> bool {
    matches!(options.speculative_backend.as_str(), "" | "dflash2" | "mtp")
}

/// `CreateQwenVerifier`'s message for another draft.
fn unknown_backend(options: &PromptOptions) -> ExitCode {
    eprintln!(
        "Unknown speculative backend: {}",
        options.speculative_backend
    );
    ExitCode::from(1)
}

/// Whether the Rust Flash-Next engine runs `options`: a `qwen4exp`
/// artifact.
fn flash_next_engine(options: &PromptOptions) -> bool {
    gufo_qwen38_flash_next::runner::supports(Path::new(&options.model_path))
}

/// `AttachImages`: the images, read from their files, before the text of
/// `message`.
fn attach_images(options: &PromptOptions, message: &mut ChatMessage) -> Result<(), String> {
    for path in &options.image_paths {
        let bytes = gufo_core::image::read_image_file(Path::new(path))
            .map_err(|e| e.message().to_string())?;
        message.images.push(ImagePart {
            offset: 0,
            bytes: std::sync::Arc::new(bytes),
        });
    }
    Ok(())
}

/// `PrepareVision`: the image prompt of `messages` for `encoder`.
fn prepare_vision(
    options: &PromptOptions,
    tokenizer: &QwenTokenizer,
    messages: &[ChatMessage],
    encoder: Option<&Encoder>,
    max_context: u32,
) -> Result<std::sync::Arc<VisionPrompt>, String> {
    let encoder = encoder
        .ok_or_else(|| "image input requires a matching --mmproj BF16 sidecar".to_string())?;
    gufo_qwen::vision::prepare(
        tokenizer,
        messages,
        &[],
        &resolve_options(&reasoning(options), options.add_vision_id),
        encoder.identity(),
        max_context,
        None,
    )
    .map(std::sync::Arc::new)
    .map_err(|e| e.message().to_string())
}

/// `LoadQwenVision`: the 27B's encoder when images or a sidecar are given.
fn load_qwen_vision(
    options: &PromptOptions,
    config: &gufo_qwen35::ModelConfig,
) -> Result<Option<std::sync::Arc<Encoder>>, String> {
    if options.image_paths.is_empty() && options.vision_model_path.is_empty() {
        return Ok(None);
    }
    if config.hidden_size != 5120 {
        return Err("image input supports Qwen3.8-27B and Flash-Next".into());
    }
    Encoder::open(
        Path::new(&options.model_path),
        Path::new(&options.vision_model_path),
        5120,
    )
    .map_err(|e| e.message().to_string())?
    .map(Some)
    .ok_or_else(|| "image input requires a matching --mmproj BF16 sidecar".into())
}

/// The CLI context of Flash-Next (`kDefaultContext`).
const FLASH_NEXT_CONTEXT: u32 = 4096;

/// `LoadFlashNextModel`: the model and a session of the requested mode.
fn load_flash_next(
    options: &PromptOptions,
    load_start: Instant,
) -> Option<(
    gufo_qwen38_flash_next::engine::Model,
    gufo_qwen38_flash_next::engine::Session,
)> {
    use gufo_qwen38_flash_next::engine::{Model, ModelOptions};
    use gufo_qwen38_flash_next::executor::SessionMode;
    let spec = options.speculative_backend.as_str();
    if (!spec.is_empty() && spec != "mtp")
        || (spec == "mtp" && options.min_draft_tokens != 1)
        || gufo_hip::device_memory().is_err()
    {
        eprintln!("Flash-Next requires ROCm and supports MTP with --min-draft-tokens 1");
        return None;
    }
    if options.use_chat_template
        && let Err(error) = flash_next_template(&options.model_path)
    {
        eprintln!("Unsupported Flash-Next chat template: {error}");
        return None;
    }
    let loaded = Model::load(
        Path::new(&options.model_path),
        ModelOptions {
            max_context: FLASH_NEXT_CONTEXT,
            mtp_model_path: (spec == "mtp").then(|| options.mtp_model_path.clone().into()),
            max_draft_tokens: options.draft_tokens,
            vision_model_path: (!options.vision_model_path.is_empty())
                .then(|| options.vision_model_path.clone().into()),
            ..ModelOptions::default()
        },
    );
    load_time(load_start, loaded.is_ok());
    let model = match loaded {
        Ok(model) => model,
        Err(error) => {
            eprintln!("Flash-Next load failed: {error}");
            return None;
        }
    };
    let mode = if spec.is_empty() {
        SessionMode::Autoregressive
    } else {
        SessionMode::Speculative
    };
    match model.create_session(mode, FLASH_NEXT_CONTEXT) {
        Ok(session) => Some((model, session)),
        Err(error) => {
            eprintln!("Flash-Next session failed: {error}");
            None
        }
    }
}

/// The artifact's chat template is the pinned Qwen3.8 one.
fn flash_next_template(path: &str) -> Result<(), String> {
    let (header, len) = gufo_core::gguf::read_header(Path::new(path))?;
    let gguf = gufo_core::gguf::Gguf::parse_prefix(&header, len).map_err(|e| e.to_string())?;
    let qwen38 = gufo_qwen::chat_template::is_qwen38_named(&gguf);
    gufo_qwen::chat_template::ChatTemplate::validate_gguf(&gguf, qwen38)
}

/// `GenerateFlashNextResponse`: syncs the prompt, streams the decoded
/// pieces, then the verbose report; appends the reply's bytes to `reply`.
fn respond_flash_next(
    model: &mut gufo_qwen38_flash_next::engine::Model,
    session: &mut gufo_qwen38_flash_next::engine::Session,
    options: &PromptOptions,
    prompt: &[u32],
    mut reply: Option<&mut Vec<u8>>,
) -> Result<(), String> {
    let context = session.context_size() as usize;
    if prompt.is_empty() || prompt.len() >= context || options.max_tokens > context - prompt.len() {
        return Err("Flash-Next prompt and output exceed the 4096-token CLI context".into());
    }
    let input: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    model
        .sync(session, &input)
        .map_err(|e| format!("Flash-Next prefill failed: {e}"))?;
    let mut sampler = gufo_core::sampling::SamplerState::new(options.sampling.clone(), prompt)
        .map_err(|e| e.to_string())?;
    let mut generated: Vec<u32> = Vec::new();
    let stats_before = session.statistics();
    let start = Instant::now();
    while generated.len() < options.max_tokens {
        let decoded = model
            .decode_step(
                session,
                options.max_tokens - generated.len(),
                &mut sampler,
                true,
            )
            .map_err(|e| format!("Flash-Next decode failed: {e}"))?;
        for &token in &decoded.tokens {
            let piece = model.decode(&[token]);
            let mut out = std::io::stdout().lock();
            let _ = out.write_all(&piece);
            let _ = out.flush();
            if let Some(reply) = reply.as_deref_mut() {
                reply.extend_from_slice(&piece);
            }
            generated.push(token as u32);
        }
        if decoded.stop {
            break;
        }
    }
    println!();
    if options.verbose {
        let seconds = start.elapsed().as_secs_f64();
        eprintln!(
            "Generated {} tokens ({} tok/s)",
            generated.len(),
            cpp_double(generated.len() as f64 / seconds)
        );
        let stats = session.statistics();
        if stats.cycles != stats_before.cycles {
            eprintln!(
                "[Speculative]: cycles={} drafted={} accepted={} lookup_proposed={} lookup_accepted={}",
                stats.cycles - stats_before.cycles,
                stats.drafted - stats_before.drafted,
                stats.accepted - stats_before.accepted,
                stats.lookup - stats_before.lookup,
                stats.lookup_accepted - stats_before.lookup_accepted
            );
        }
        token_trace(&generated);
    }
    Ok(())
}

fn load_time(start: Instant, success: bool) {
    let text = format!(
        "[Model Load]: {} s",
        cpp_double(start.elapsed().as_secs_f64())
    );
    if success {
        println!("{text}");
    } else {
        eprintln!("{text} (failed)");
    }
}

/// `PrintTokenTrace`.
fn token_trace(tokens: &[u32]) {
    let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    eprintln!("[TokenTrace]: count={} sha256={digest}", tokens.len());
}

/// The target with its session and an optional verifier.
struct Engine {
    target: Target,
    draft: Option<Draft>,
}

enum Draft {
    DFlash(Box<DFlashDraft>),
    Mtp(Speculative),
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
        let draft = MtpDraft::new(stream, &model, CONTEXT, options.draft_tokens).map_err(failed)?;
        self.draft = Some(Draft::Mtp(Speculative::mtp(
            Arc::new(model),
            draft,
            options.min_draft_tokens,
            options.draft_tokens,
        )));
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
            Some(Draft::Mtp(speculative)) => speculative.generate(
                &mut self.target,
                prompt,
                options.max_tokens,
                &options.sampling,
                on_token,
            ),
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
        let stats = match &engine.draft {
            Some(Draft::DFlash(draft)) => {
                let s = draft.verifier.stats();
                Some((
                    s.acceptance_rate(),
                    s.draft_tokens,
                    s.accepted_tokens,
                    s.verification_steps,
                ))
            }
            Some(Draft::Mtp(speculative)) => {
                let s = speculative.stats();
                Some((
                    s.acceptance_rate(),
                    s.draft_tokens as u64,
                    s.accepted_tokens as u64,
                    s.verification_steps as u64,
                ))
            }
            None => None,
        };
        if let Some((acceptance, drafted, accepted, steps)) = stats {
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

fn system_message(options: &PromptOptions) -> Option<ChatMessage> {
    (!options.system_prompt.is_empty())
        .then(|| ChatMessage::new(ChatRole::System, options.system_prompt.clone()))
}

/// `RunPrompt`.
pub fn run_prompt(args: &[String]) -> ExitCode {
    let mut o = match parse(args, "prompt") {
        Ok(Some(o)) => o,
        Ok(None) => {
            print_help("prompt");
            return ExitCode::SUCCESS;
        }
        Err(error) if error.is_empty() => {
            print_help("prompt");
            return ExitCode::SUCCESS;
        }
        Err(error) => {
            eprintln!("Error: {error}");
            print_help("prompt");
            return ExitCode::from(2);
        }
    };
    if !o.prompt_file.is_empty() {
        let Ok(content) = std::fs::read(&o.prompt_file) else {
            eprintln!("Error: could not open prompt file '{}'", o.prompt_file);
            return ExitCode::from(1);
        };
        let content = String::from_utf8_lossy(&content).into_owned();
        o.prompt_text = if o.prompt_text.is_empty() {
            content
        } else {
            format!("{content}\n{}", o.prompt_text)
        };
    }
    if o.prompt_text.is_empty() && o.model_path.is_empty() {
        print_help("prompt");
        return ExitCode::SUCCESS;
    }
    if o.model_path.is_empty() {
        println!(
            "gufo prompt: prompt received: \"{}\"\n(Specify --model <PATH.gguf> to execute local model generation)",
            o.prompt_text
        );
        return ExitCode::SUCCESS;
    }
    if deepseek::supports(&o) {
        resolve_sampling(&mut o);
        return deepseek::run_prompt(&o, Instant::now());
    }
    let flash_next = flash_next_engine(&o);
    if !flash_next && !dense_backend(&o) {
        return unknown_backend(&o);
    }
    resolve_sampling(&mut o);
    let load_start = Instant::now();
    let mut rendered = o.prompt_text.clone();
    // A raw prompt is framing throughout; a templated one keeps its message
    // text out of control-token matching.
    let mut content_spans = Vec::new();
    let mut messages: Vec<ChatMessage> = Vec::new();
    if o.use_chat_template {
        messages.extend(system_message(&o));
        messages.push(ChatMessage::new(ChatRole::User, o.prompt_text.clone()));
        match render(
            &messages,
            &[],
            &resolve_options(&reasoning(&o), o.add_vision_id),
        ) {
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
    if flash_next {
        let Some((mut model, mut session)) = load_flash_next(&o, load_start) else {
            return ExitCode::from(1);
        };
        let prompt = (|| {
            if o.image_paths.is_empty() {
                return Ok(encode_rendered(
                    model.tokenizer(),
                    &rendered,
                    0,
                    rendered.len(),
                    &content_spans,
                ));
            }
            if let Some(last) = messages.last_mut() {
                attach_images(&o, last)?;
            }
            let vision = prepare_vision(
                &o,
                model.tokenizer(),
                &messages,
                model.vision_encoder().map(|e| &**e),
                FLASH_NEXT_CONTEXT,
            )?;
            let tokens: Vec<u32> = vision.tokens.clone();
            model.configure_vision(&mut session, Some(vision))?;
            Ok::<_, String>(tokens)
        })();
        let prompt = match prompt {
            Ok(prompt) => prompt,
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::from(1);
            }
        };
        return match respond_flash_next(&mut model, &mut session, &o, &prompt, None) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("{error}");
                ExitCode::from(1)
            }
        };
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
        let encoder = load_qwen_vision(&o, engine.target.config())?;
        if o.image_paths.is_empty() {
            return Ok(None);
        }
        if let Some(last) = messages.last_mut() {
            attach_images(&o, last)?;
        }
        let vision = prepare_vision(&o, &tokenizer, &messages, encoder.as_deref(), CONTEXT)?;
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
        let c = engine.target.config();
        let mut out = std::io::stdout().lock();
        let mut text = format!(
            "[Engine]: AMD Strix Halo gfx1151 GPU Executor\nModel: {} ({} layers, hidden={}, heads={})\nPrompt tokens ({}): ",
            c.architecture,
            c.num_layers,
            c.hidden_size,
            c.num_attention_heads,
            prompt.len()
        )
        .into_bytes();
        for &token in &prompt {
            text.extend_from_slice(format!("[{token}: '").as_bytes());
            text.extend_from_slice(&tokenizer.decode(&[token]));
            text.extend_from_slice(b"'] ");
        }
        text.extend_from_slice(
            format!(
                "\nMax tokens: {}\n--- Generation Output ---\n",
                o.max_tokens
            )
            .as_bytes(),
        );
        let _ = out.write_all(&text);
        let _ = out.flush();
    }
    let result = (|| {
        engine.attach_draft(&o)?;
        respond(&mut engine, &tokenizer, &o, &prompt, None)
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}

/// The engine of a chat: dense Qwen3.8 or Flash-Next with its session.
enum ChatEngine {
    Dense(Box<Engine>),
    FlashNext(
        Box<gufo_qwen38_flash_next::engine::Model>,
        Box<gufo_qwen38_flash_next::engine::Session>,
    ),
}

/// `RunChat`.
pub fn run_chat(args: &[String]) -> ExitCode {
    let mut o = match parse(args, "chat") {
        Ok(Some(o)) => o,
        Ok(None) => {
            print_help("chat");
            return ExitCode::SUCCESS;
        }
        Err(error) if error.is_empty() => {
            print_help("chat");
            return ExitCode::SUCCESS;
        }
        Err(error) => {
            eprintln!("Error: {error}");
            print_help("chat");
            return ExitCode::from(2);
        }
    };
    if o.model_path.is_empty() {
        println!(
            "gufo chat: interactive conversation mode\n(Specify --model <PATH.gguf> to load model weights)"
        );
        return ExitCode::SUCCESS;
    }
    if deepseek::supports(&o) {
        resolve_sampling(&mut o);
        return deepseek::run_chat(&o, Instant::now());
    }
    let flash_next = flash_next_engine(&o);
    if !flash_next && !dense_backend(&o) {
        return unknown_backend(&o);
    }
    resolve_sampling(&mut o);
    let load_start = Instant::now();
    let mut encoder = None;
    let mut engine = if flash_next {
        let Some((model, session)) = load_flash_next(&o, load_start) else {
            return ExitCode::from(1);
        };
        encoder = model.vision_encoder().cloned();
        ChatEngine::FlashNext(Box::new(model), Box::new(session))
    } else {
        let mut engine = match Engine::load(&o.model_path) {
            Ok(engine) => engine,
            Err(error) => {
                eprintln!("Error creating Qwen GPU executor: {error}");
                load_time(load_start, false);
                return ExitCode::from(1);
            }
        };
        let attached = load_qwen_vision(&o, engine.target.config()).and_then(|vision| {
            encoder = vision;
            engine.attach_draft(&o)
        });
        if let Err(error) = attached {
            eprintln!("{error}");
            load_time(load_start, false);
            return ExitCode::from(1);
        }
        if o.verbose {
            println!("[Engine]: AMD Strix Halo gfx1151 GPU Executor");
        }
        ChatEngine::Dense(Box::new(engine))
    };
    // The C++ chat prints the load time once more after a Flash-Next load.
    load_time(load_start, true);
    let (architecture, tokenizer) = match &engine {
        ChatEngine::Dense(engine) => (
            engine.target.config().architecture.clone(),
            engine.target.weights().tokenizer.clone(),
        ),
        ChatEngine::FlashNext(model, _) => ("qwen4exp".to_string(), model.tokenizer().clone()),
    };
    println!("=== Gufo Interactive Chat ({architecture}) ===\nType 'exit' or Ctrl+D to quit.\n");
    let mut history: Vec<ChatMessage> = system_message(&o).into_iter().collect();
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
        let options = resolve_options(&reasoning(&o), o.add_vision_id);
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
        // The images join the first user turn; every turn prepares them.
        let vision = (|| {
            if std::mem::take(&mut first_turn)
                && let Some(last) = history.last_mut()
            {
                attach_images(&o, last)?;
            }
            if o.image_paths.is_empty() {
                return Ok(None);
            }
            let vision = prepare_vision(&o, &tokenizer, &history, encoder.as_deref(), CONTEXT)?;
            match &mut engine {
                ChatEngine::Dense(engine) => {
                    let (model, session) = engine.target.parts_mut();
                    model.configure_vision(session, Some(vision.clone()), encoder.clone())?
                }
                ChatEngine::FlashNext(model, session) => {
                    model.configure_vision(session, Some(vision.clone()))?
                }
            }
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
        let responded = match &mut engine {
            ChatEngine::Dense(engine) => {
                respond(engine, &tokenizer, &o, &prompt, Some(&mut reply_bytes))
            }
            ChatEngine::FlashNext(model, session) => {
                respond_flash_next(model, session, &o, &prompt, Some(&mut reply_bytes)).map_err(
                    |error| {
                        eprintln!("{error}");
                        error
                    },
                )
            }
        };
        if let Err(error) = responded {
            if !flash_next {
                eprintln!("Generation failed: {error}");
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_prompt(list: &[&str]) -> PromptOptions {
        let args: Vec<String> = list.iter().map(|s| s.to_string()).collect();
        parse(&args, "prompt").unwrap().unwrap()
    }

    /// `--image` has no count limit, but needs a path.
    #[test]
    fn image_flags_take_any_count() {
        let mut args = Vec::new();
        for _ in 0..17 {
            args.extend(["--image", "a.png"]);
        }
        args.push("Compare");
        assert_eq!(parse_prompt(&args).image_paths.len(), 17);
        let empty: Vec<String> = ["--image", ""].iter().map(|s| s.to_string()).collect();
        assert!(parse(&empty, "prompt").is_err());
    }

    /// The DSpark options of the C++ `ds4.cli` contract; `bench::tests`
    /// checks the same arguments.
    #[test]
    fn dspark_options_match_the_cpp_contract() {
        for (args, backend) in [
            (&["--dspark-model", "support.gguf"][..], "dspark"),
            (
                &["--speculative", "dspark", "--dspark-model", "support.gguf"],
                "dspark",
            ),
            (
                &["--speculative", "off", "--dspark-model", "support.gguf"],
                "",
            ),
            (
                &["--dspark-model", "support.gguf", "--speculative", "off"],
                "",
            ),
        ] {
            let o = parse_prompt(args);
            assert_eq!(o.speculative_backend, backend, "{args:?}");
            assert_eq!(o.dspark_model_path, "support.gguf", "{args:?}");
        }
        let o = parse_prompt(&["--dspark-model", "support.gguf", "--draft-tokens", "3"]);
        assert_eq!(o.draft_tokens, 3);
    }
}
