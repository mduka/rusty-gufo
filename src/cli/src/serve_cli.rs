//! The `gufo serve` command line (`src/cli/serve/serve.cpp`) up to model
//! loading: help texts, server and modality options, every usage check with
//! its C++ message and exit code, and the configuration the serving wiring
//! loads. Voice presets are built separately ([`build_voice_presets`])
//! because C++ builds them inside the model load log.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use gufo_core::json::Value;
use gufo_core::sampling::{SamplingConfig, SamplingOverrides};
use gufo_core::text::{ReasoningEffort, ReasoningOptions};
use gufo_serve::disk_store::DEFAULT_CAPACITY_BYTES;
use gufo_serve::logging::LogLevel;
use gufo_serve::openai_chat::{ApiCompat, ResponsesReasoningTarget};
use gufo_serve::sampling_request::parse_sampling_config;
use gufo_serve::scheduler::{
    DEFAULT_DECODE_ACTIVE_PREFILL_TOKENS, DEFAULT_MAX_BUFFERED_OUTPUT_BYTES,
    DEFAULT_MAX_BUFFERED_OUTPUT_BYTES_TOTAL, DEFAULT_MAX_OUTPUT_BYTES, TextPrefillPolicy,
    TextSchedulerPolicy,
};
use gufo_serve::tts_model::decode_wav;
use gufo_serve::tts_service::TtsVoicePreset;

use crate::arg_parser::ArgParser;
use crate::sampling_options::{register_sampling_options, sampling_options_supplied};

const MODALITIES: [&str; 5] = ["llm", "tts", "asr", "image", "embedding"];

/// Options every modality shares; they are accepted on either side of the
/// modality word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerOptions {
    pub host: String,
    pub port: i32,
    pub session_count: usize,
    pub max_connections: usize,
    pub max_request_body_bytes: usize,
    pub api_key: String,
    /// llama-server's `--sse-ping-interval`: seconds a stream may stay
    /// silent before a ping comment; zero or -1 disables pings.
    pub sse_ping_interval: i64,
    /// `-v`, shorthand for `--log-level=debug`.
    pub verbose: bool,
    pub log_level: Option<LogLevel>,
    /// `--arbiter`: the `gufo arbiter` to report to; empty runs alone.
    pub arbiter: String,
}

impl ServerOptions {
    /// The log threshold `--log-level` or `-v` selected; `Info` by default.
    pub fn resolved_log_level(&self) -> LogLevel {
        if self.verbose {
            LogLevel::Debug
        } else {
            self.log_level.unwrap_or(LogLevel::Info)
        }
    }
}

const LOG_LEVEL_HELP: &str = "Log verbosity: error, warn, info or debug (default: info)";
const VERBOSE_HELP: &str = "Shorthand for --log-level=debug";

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8080,
            session_count: 1,
            max_connections: 16,
            max_request_body_bytes: 8 * 1024 * 1024,
            api_key: String::new(),
            sse_ping_interval: 15,
            verbose: false,
            log_level: None,
            arbiter: String::new(),
        }
    }
}

/// `TextSpeculativeBackend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeBackend {
    Disabled,
    DFlash,
    DSpark,
    Mtp,
}

impl SpeculativeBackend {
    /// The `speculative=` value of the load log.
    pub fn name(self) -> &'static str {
        match self {
            Self::DFlash => "dflash2",
            Self::DSpark => "dspark",
            Self::Mtp => "mtp",
            Self::Disabled => "off",
        }
    }
}

/// `speculative::DFlashDraftPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFlashDraftPolicy {
    Fixed,
    Adaptive,
}

/// Everything `serve llm` loads with (the arguments of
/// `InferenceBackend::load` and the backend defaults).
#[derive(Debug, Clone, PartialEq)]
pub struct LlmOptions {
    pub model: String,
    /// `--mmproj`; empty when not given.
    pub vision_model: String,
    /// Empty keeps the model's own ID.
    pub served_model_name: String,
    /// Zero: the model's native context, resolved at load.
    pub max_context: u32,
    /// Zero: until EOS or the remaining context is exhausted.
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
    /// The sampling options given on the command line; the others follow
    /// the model's recommended defaults.
    pub sampling_supplied: SamplingOverrides,
    pub reasoning: ReasoningOptions,
    /// `--responses-reasoning-target`: where `/v1/responses` puts the raw
    /// reasoning.
    pub responses_reasoning_target: ResponsesReasoningTarget,
    /// `--api-compat`: where the compatibility routes follow llama.cpp.
    pub api_compat: ApiCompat,
    /// `--enable-yarn`: Flash-Next's static YaRN, required for a context
    /// above the native one.
    pub enable_yarn: bool,
    pub speculative: SpeculativeBackend,
    /// The DSpark, MTP or (for every other backend) DFlash2 path.
    pub draft_model_path: String,
    pub max_draft_tokens: u32,
    pub min_draft_tokens: u32,
    pub dflash_policy: DFlashDraftPolicy,
    pub prefill: TextPrefillPolicy,
    pub scheduler: TextSchedulerPolicy,
    /// Zero: automatic, see `TextRunnerRamCacheOptions`.
    pub cache_ram_bytes: usize,
    /// Empty disables the disk cache.
    pub cache_disk_directory: PathBuf,
    pub cache_disk_bytes: usize,
    pub cache_disk_staging_bytes: usize,
}

/// `serve tts` and `serve asr` (`SpeechOptions`); the voice fields are
/// TTS-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechOptions {
    pub model: PathBuf,
    pub context: usize,
    pub served_model_name: String,
    /// `--voice NAME=PATH` in command-line order.
    pub voice_specs: Vec<(String, String)>,
    pub voice_text_specs: BTreeMap<String, String>,
    pub voice_lang_specs: BTreeMap<String, String>,
}

impl SpeechOptions {
    fn new(tts: bool) -> Self {
        Self {
            model: PathBuf::new(),
            context: if tts { 4096 } else { 1024 },
            served_model_name: String::new(),
            voice_specs: Vec::new(),
            voice_text_specs: BTreeMap::new(),
            voice_lang_specs: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Modality {
    Llm(Box<LlmOptions>),
    Tts(SpeechOptions),
    Asr(SpeechOptions),
    Image {
        model: PathBuf,
        served_model_name: String,
    },
    Embedding {
        model: PathBuf,
        served_model_name: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServeRequest {
    pub server: ServerOptions,
    pub modality: Modality,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ServeParse {
    /// Help or a usage error was printed; exit with this code.
    Exit(u8),
    Run(Box<ServeRequest>),
}

fn add_server_options<'a>(
    parser: &mut ArgParser<'a>,
    server: &'a mut ServerOptions,
    include_sessions: bool,
) {
    let ServerOptions {
        host,
        port,
        session_count,
        max_connections,
        max_request_body_bytes,
        api_key,
        sse_ping_interval,
        verbose,
        log_level,
        arbiter,
    } = server;
    parser
        .add_option("-i", "--host", "IP", "Bind address", "Server", host)
        .add_option("-p", "--port", "N", "Port to listen on", "Server", port);
    if include_sessions {
        parser.add_option(
            "-j",
            "--sessions",
            "N",
            "Preallocated GPU request sessions",
            "Server",
            session_count,
        );
    }
    parser
        .add_option(
            "",
            "--max-connections",
            "N",
            "Maximum simultaneous HTTP connections",
            "Server",
            max_connections,
        )
        .add_option(
            "",
            "--max-request-bytes",
            "N",
            "Maximum HTTP request body bytes",
            "Server",
            max_request_body_bytes,
        )
        .add_option("", "--api-key", "KEY", "API key", "Server", api_key)
        .add_option(
            "",
            "--arbiter",
            "HOST:PORT",
            "Report to this gufo arbiter and follow its GPU grants",
            "Server",
            arbiter,
        )
        .add_option(
            "",
            "--sse-ping-interval",
            "N",
            "Seconds a stream may stay silent before a ping (-1 disables)",
            "Server",
            sse_ping_interval,
        )
        .add_custom_option(
            "",
            "--log-level",
            "LEVEL",
            LOG_LEVEL_HELP,
            "Logging",
            move |flag, value| {
                *log_level = Some(
                    LogLevel::from_name(value)
                        .ok_or_else(|| format!("{flag} must be error, warn, info or debug"))?,
                );
                Ok(())
            },
        )
        .add_flag("-v", "--verbose", VERBOSE_HELP, "Logging", verbose);
}

/// `SplitNameValue`: `NAME=VALUE` with both sides non-empty.
fn split_name_value(spec: &str, flag: &str) -> Result<(String, String), String> {
    match spec.find('=') {
        Some(separator) if separator > 0 && separator + 1 < spec.len() => Ok((
            spec[..separator].to_owned(),
            spec[separator + 1..].to_owned(),
        )),
        _ => Err(format!("{flag} expects NAME=VALUE")),
    }
}

fn add_speech_options<'a>(parser: &mut ArgParser<'a>, tts: bool, options: &'a mut SpeechOptions) {
    let SpeechOptions {
        model,
        context,
        served_model_name,
        voice_specs,
        voice_text_specs,
        voice_lang_specs,
    } = options;
    parser
        .add_option(
            "-m",
            "--model",
            "DIR",
            if tts {
                "Qwen3-TTS 12Hz 1.7B model directory"
            } else {
                "Qwen3-ASR 1.7B model directory"
            },
            "Model",
            model,
        )
        .add_option(
            "-c",
            "--context",
            "N",
            if tts {
                "Context capacity (default: 4096)"
            } else {
                "Context capacity per audio chunk (default: 1024)"
            },
            "Model",
            context,
        )
        .add_option(
            "",
            "--served-model-name",
            "NAME",
            "Public API model ID",
            "Model",
            served_model_name,
        );
    if !tts {
        return;
    }
    parser
        .add_custom_option(
            "",
            "--voice",
            "NAME=PATH",
            "Register a named Qwen3-TTS Base voice from a reference WAV (repeatable)",
            "Model",
            move |_, value| {
                voice_specs.push(split_name_value(value, "--voice")?);
                Ok(())
            },
        )
        .add_custom_option(
            "",
            "--voice-lang",
            "NAME=LANGUAGE",
            "Language a --voice speaks; used when a request omits 'language' (repeatable)",
            "Model",
            move |_, value| {
                let (name, language) = split_name_value(value, "--voice-lang")?;
                if voice_lang_specs.contains_key(&name) {
                    return Err("duplicate --voice-lang name".into());
                }
                voice_lang_specs.insert(name, language);
                Ok(())
            },
        )
        .add_custom_option(
            "",
            "--voice-text",
            "NAME=TEXT|PATH",
            "Reference transcript for a --voice, given inline or as a file path; defaults \
             to a .txt sidecar beside the WAV (repeatable)",
            "Model",
            move |_, value| {
                let (name, text) = split_name_value(value, "--voice-text")?;
                if voice_text_specs.contains_key(&name) {
                    return Err("duplicate --voice-text name".into());
                }
                voice_text_specs.insert(name, text);
                Ok(())
            },
        );
}

fn image_parser<'a>(
    model: &'a mut PathBuf,
    name: &'a mut String,
    server: &'a mut ServerOptions,
) -> ArgParser<'a> {
    let mut parser = ArgParser::new(
        "gufo serve image",
        "Serve Qwen-Image-2.1 generation and editing.",
    );
    parser
        .add_option(
            "-m",
            "--model",
            "DIR",
            "Qwen-Image-2.1 safetensors directory",
            "Model",
            model,
        )
        .add_option(
            "",
            "--served-model-name",
            "NAME",
            "Public API model ID",
            "Model",
            name,
        );
    add_server_options(&mut parser, server, false);
    parser
}

fn embedding_parser<'a>(
    model: &'a mut PathBuf,
    name: &'a mut String,
    server: &'a mut ServerOptions,
) -> ArgParser<'a> {
    let mut parser = ArgParser::new(
        "gufo serve embedding",
        "Serve EmbeddingGemma 2 text embeddings.",
    );
    parser
        .add_option(
            "-m",
            "--model",
            "DIR",
            "EmbeddingGemma 2 model directory",
            "Model",
            model,
        )
        .add_option(
            "",
            "--served-model-name",
            "NAME",
            "Public API model ID (default: embeddinggemma-2)",
            "Model",
            name,
        );
    add_server_options(&mut parser, server, false);
    parser
}

/// The default `--served-model-name` of `serve embedding`.
const EMBEDDING_MODEL_ID: &str = "embeddinggemma-2";

fn speech_parser<'a>(
    tts: bool,
    options: &'a mut SpeechOptions,
    server: &'a mut ServerOptions,
) -> ArgParser<'a> {
    let mut parser = ArgParser::new(
        if tts {
            "gufo serve tts"
        } else {
            "gufo serve asr"
        },
        if tts {
            "Serve Qwen3-TTS speech synthesis."
        } else {
            "Serve Qwen3-ASR transcription."
        },
    );
    add_speech_options(&mut parser, tts, options);
    add_server_options(&mut parser, server, false);
    parser
}

/// The raw `serve llm` options before validation.
struct LlmCli {
    model: String,
    served_model_name: String,
    /// Zero: the model's native context.
    max_context: u32,
    /// -1: until EOS or the context is full.
    max_tokens: i64,
    sampling: SamplingConfig,
    sampling_supplied: SamplingOverrides,
    reasoning_mode: String,
    reasoning_effort: String,
    preserve_thinking: String,
    /// -1: unlimited.
    reasoning_budget: i64,
    responses_reasoning_target: String,
    api_compat: String,
    enable_yarn: bool,
    speculative_backend: String,
    dflash_model_path: String,
    draft_policy: String,
    dspark_model_path: String,
    mtp_model_path: String,
    vision_model_path: String,
    draft_tokens: usize,
    min_draft_tokens: usize,
    prefill_chunk_tokens: usize,
    max_pending_requests: usize,
    max_pending_requests_per_client: usize,
    request_timeout_ms: u64,
    stall_timeout_ms: u64,
    max_output_bytes: usize,
    max_buffered_output_bytes: usize,
    max_buffered_output_bytes_total: usize,
    cache_ram_bytes: usize,
    cache_disk_directory: PathBuf,
    cache_disk_bytes: usize,
    cache_disk_staging_bytes: usize,
    log_progress: bool,
}

impl LlmCli {
    fn new() -> Self {
        Self {
            model: String::new(),
            served_model_name: String::new(),
            max_context: 0,
            max_tokens: -1,
            sampling: SamplingConfig::default(),
            sampling_supplied: SamplingOverrides::default(),
            reasoning_mode: "auto".into(),
            reasoning_effort: "auto".into(),
            preserve_thinking: "auto".into(),
            reasoning_budget: -1,
            responses_reasoning_target: "summary".into(),
            api_compat: "gufo".into(),
            enable_yarn: false,
            speculative_backend: String::new(),
            dflash_model_path: String::new(),
            draft_policy: String::new(),
            dspark_model_path: String::new(),
            mtp_model_path: String::new(),
            vision_model_path: String::new(),
            draft_tokens: 7,
            min_draft_tokens: 1,
            prefill_chunk_tokens: DEFAULT_DECODE_ACTIVE_PREFILL_TOKENS,
            max_pending_requests: 16,
            // `--max-pending` unless supplied, set after parsing.
            max_pending_requests_per_client: 0,
            request_timeout_ms: 0,
            stall_timeout_ms: 0,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            max_buffered_output_bytes: DEFAULT_MAX_BUFFERED_OUTPUT_BYTES,
            max_buffered_output_bytes_total: DEFAULT_MAX_BUFFERED_OUTPUT_BYTES_TOTAL,
            cache_ram_bytes: 0,
            cache_disk_directory: PathBuf::new(),
            cache_disk_bytes: DEFAULT_CAPACITY_BYTES,
            // Zero: automatic, see `ContinuationDiskStoreOptions`.
            cache_disk_staging_bytes: 0,
            log_progress: false,
        }
    }
}

fn llm_parser<'a>(cli: &'a mut LlmCli, server: &'a mut ServerOptions) -> ArgParser<'a> {
    let LlmCli {
        model,
        served_model_name,
        max_context,
        max_tokens,
        sampling,
        sampling_supplied: _,
        reasoning_mode,
        reasoning_effort,
        preserve_thinking,
        reasoning_budget,
        responses_reasoning_target,
        api_compat,
        enable_yarn,
        speculative_backend,
        dflash_model_path,
        draft_policy,
        dspark_model_path,
        mtp_model_path,
        vision_model_path,
        draft_tokens,
        min_draft_tokens,
        prefill_chunk_tokens,
        max_pending_requests,
        max_pending_requests_per_client,
        request_timeout_ms,
        stall_timeout_ms,
        max_output_bytes,
        max_buffered_output_bytes,
        max_buffered_output_bytes_total,
        cache_ram_bytes,
        cache_disk_directory,
        cache_disk_bytes,
        cache_disk_staging_bytes,
        log_progress,
    } = cli;
    let mut parser = ArgParser::new(
        "gufo serve llm",
        "Start the OpenAI/Anthropic-compatible text LLM HTTP server.",
    );
    parser
        .add_option(
            "-m",
            "--model",
            "PATH",
            "Path to GGUF model file (required)",
            "Model",
            model,
        )
        .add_option(
            "",
            "--mmproj",
            "PATH",
            "Qwen BF16 vision sidecar (auto-discovered beside model)",
            "Model",
            vision_model_path,
        )
        .add_option(
            "",
            "--served-model-name",
            "ID",
            "Model identifier exposed by the OpenAI API",
            "Model",
            served_model_name,
        )
        .add_option(
            "-c",
            "--context",
            "N",
            "Context tokens per session (default: 0 = model native context)",
            "Model",
            max_context,
        )
        .add_flag(
            "",
            "--enable-yarn",
            "Flash-Next: static YaRN, required for --context above the native 262144; it \
             applies to every request",
            "Model",
            enable_yarn,
        )
        .add_option(
            "-n",
            "--max-tokens",
            "N",
            "Default new-token limit (default: -1 = until EOS or context full)",
            "Sampling Defaults",
            max_tokens,
        );
    register_sampling_options(&mut parser, sampling, "Sampling Defaults", true, true);
    parser
        .add_option(
            "",
            "--think",
            "MODE",
            "Default reasoning mode: on, off, or auto (default: model)",
            "Reasoning Defaults",
            reasoning_mode,
        )
        .add_option(
            "",
            "--reasoning-effort",
            "LEVEL",
            "Default effort: auto, minimal, low, medium, high, xhigh, or max",
            "Reasoning Defaults",
            reasoning_effort,
        )
        .add_option(
            "",
            "--preserve-thinking",
            "MODE",
            "Replay prior reasoning: on, off, or auto",
            "Reasoning Defaults",
            preserve_thinking,
        )
        .add_option(
            "",
            "--reasoning-budget",
            "N",
            "Reasoning tokens before </think> is forced: -1 unlimited, 0 none (default: -1)",
            "Reasoning Defaults",
            reasoning_budget,
        )
        .add_option(
            "",
            "--responses-reasoning-target",
            "TARGET",
            "Responses reasoning field: summary, reasoning-text, or reasoning (default: summary)",
            "Reasoning Defaults",
            responses_reasoning_target,
        )
        .add_option(
            "",
            "--speculative",
            "MODE",
            "HTTP draft backend: dspark, dflash2, mtp, or off",
            "Speculative",
            speculative_backend,
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
            "Path to DeepSeek V4 Flash DSpark support GGUF file",
            "Speculative",
            dspark_model_path,
        )
        .add_option(
            "",
            "--mtp-model",
            "PATH",
            "Path to the Qwen MTP draft GGUF (Qwen3.8-Flash-Next: the \
             mtp-...-shared-*.gguf sidecar)",
            "Speculative",
            mtp_model_path,
        )
        .add_option(
            "-d",
            "--draft-tokens",
            "N",
            "Maximum speculative draft tokens evaluated per step (default: 7)",
            "Speculative",
            draft_tokens,
        )
        .add_option(
            "",
            "--min-draft-tokens",
            "N",
            "Adaptive draft floor (default: 1)",
            "Speculative",
            min_draft_tokens,
        )
        .add_option(
            "",
            "--prefill-chunk",
            "N",
            "Maximum prompt tokens between active decode rounds (default: 512)",
            "Scheduling",
            prefill_chunk_tokens,
        )
        .add_option(
            "",
            "--max-pending",
            "N",
            "Maximum queued generation requests (default: 16)",
            "Scheduling",
            max_pending_requests,
        )
        .add_option(
            "",
            "--max-pending-per-client",
            "N",
            "Maximum queued requests per client IP (default: --max-pending)",
            "Scheduling",
            max_pending_requests_per_client,
        )
        .add_option(
            "",
            "--request-timeout-ms",
            "MS",
            "Queue plus generation timeout; 0 disables it (default: 0)",
            "Scheduling",
            request_timeout_ms,
        )
        .add_option(
            "",
            "--stall-timeout-ms",
            "MS",
            "Exit as device_lost when generation makes no progress for this long; at least \
             60000, 0 disables it (default: 0)",
            "Scheduling",
            stall_timeout_ms,
        )
        .add_option(
            "",
            "--max-output-bytes",
            "N",
            "Maximum generated bytes per request (default: 1048576)",
            "Scheduling",
            max_output_bytes,
        )
        .add_option(
            "",
            "--max-buffered-output-bytes",
            "N",
            "Maximum queued stream bytes per request (default: 65536)",
            "Scheduling",
            max_buffered_output_bytes,
        )
        .add_option(
            "",
            "--max-buffered-output-total",
            "N",
            "Maximum queued stream bytes across requests (default: 262144)",
            "Scheduling",
            max_buffered_output_bytes_total,
        )
        .add_option(
            "",
            "--cache-ram-bytes",
            "N",
            "Retained RAM-cache byte budget (default: 0 = auto: half of free RAM, at most 32 GiB; \
             explicit values may use free RAM minus 4 GiB)",
            "Cache",
            cache_ram_bytes,
        )
        .add_option(
            "",
            "--cache-disk",
            "DIR",
            "Opt-in restart-safe continuation cache directory",
            "Cache",
            cache_disk_directory,
        )
        .add_option(
            "",
            "--cache-disk-bytes",
            "N",
            "Retained disk-cache byte budget (default: 8589934592)",
            "Cache",
            cache_disk_bytes,
        )
        .add_option(
            "",
            "--cache-disk-staging-bytes",
            "N",
            "RAM limit for queued snapshots and each disk read (default: 0 = auto, at most the \
             disk budget and 1/8 available RAM)",
            "Cache",
            cache_disk_staging_bytes,
        )
        .add_flag(
            "",
            "--log-progress",
            "Log live prefill and decode progress (needs --log-level=info or debug)",
            "Logging",
            log_progress,
        );
    add_server_options(&mut parser, server, true);
    parser.add_option(
        "",
        "--api-compat",
        "MODE",
        "Request semantics where Gufo and llama.cpp differ: gufo or llama.cpp (default: gufo)",
        "Server",
        api_compat,
    );
    parser
}

/// `PrintServeHelp`: the overview or one modality's options.
pub fn serve_help(program_name: &str, subcommand: &str) -> String {
    let mut server = ServerOptions::default();
    match subcommand {
        "image" => {
            let (mut model, mut name) = (PathBuf::new(), "Qwen-Image-2.1".to_owned());
            rename(
                image_parser(&mut model, &mut name, &mut server),
                program_name,
                "image",
            )
        }
        "embedding" => {
            let (mut model, mut name) = (PathBuf::new(), EMBEDDING_MODEL_ID.to_owned());
            rename(
                embedding_parser(&mut model, &mut name, &mut server),
                program_name,
                "embedding",
            )
        }
        "tts" | "asr" => {
            let tts = subcommand == "tts";
            let mut options = SpeechOptions::new(tts);
            rename(
                speech_parser(tts, &mut options, &mut server),
                program_name,
                subcommand,
            )
        }
        "llm" => {
            let mut cli = LlmCli::new();
            rename(llm_parser(&mut cli, &mut server), program_name, "llm")
        }
        _ => format!(
            "Usage: {program_name} serve [SERVER_OPTIONS] [COMMAND] [OPTIONS]\n\n\
             Start the OpenAI-compatible HTTP server for a specific model modality.\n\n\
             Commands:\n  \
             llm       Serve text LLM endpoints (/v1/chat/completions, /v1/completions) [default]\n  \
             image     Serve Qwen-Image-2.1 (/v1/images/generations, /v1/images/edits)\n  \
             tts       Serve Qwen3-TTS speech synthesis (/v1/audio/speech)\n  \
             asr       Serve Qwen3-ASR transcription (/v1/audio/transcriptions)\n  \
             embedding Serve EmbeddingGemma 2 text embeddings (/v1/embeddings)\n\n\
             Server:\n  \
             -i, --host <IP>        Bind address (default: 127.0.0.1)\n  \
             -p, --port <N>         Port to listen on (default: 8080)\n  \
             -j, --sessions <N>     Preallocated GPU request sessions (default: 1)\n      \
             --max-connections <N>\n                         \
             Maximum simultaneous HTTP connections (default: 16)\n      \
             --max-request-bytes <N>\n                         \
             Maximum HTTP request body bytes (default: 8388608)\n      \
             --api-key <KEY>    Require Bearer authorization for requests\n      \
             --sse-ping-interval <N>\n                         \
             Seconds a stream may stay silent before a ping (-1 disables) (default: 15)\n\n\
             Logging:\n      \
             --log-level <LEVEL>\n                         \
             {LOG_LEVEL_HELP}\n  \
             -v, --verbose          {VERBOSE_HELP}\n\n\
             General:\n  \
             -h, --help             Print help\n"
        ),
    }
}

/// The help of a modality parser under `<program> serve <modality>`.
fn rename(parser: ArgParser<'_>, program_name: &str, subcommand: &str) -> String {
    let help = parser.format_help();
    let usage = format!("Usage: gufo serve {subcommand} ");
    match help.strip_prefix(&usage) {
        Some(rest) => format!("Usage: {program_name} serve {subcommand} {rest}"),
        None => help,
    }
}

/// `ParseReasoningEffort`.
fn parse_reasoning_effort(value: &str) -> Option<ReasoningEffort> {
    Some(match value {
        "minimal" => ReasoningEffort::Minimal,
        "low" => ReasoningEffort::Low,
        "medium" => ReasoningEffort::Medium,
        "high" => ReasoningEffort::High,
        "xhigh" => ReasoningEffort::XHigh,
        "max" => ReasoningEffort::Max,
        _ => return None,
    })
}

/// `ResolveReasoningDefaults`.
pub fn resolve_reasoning_defaults(
    mode: &str,
    effort: &str,
    preserve: &str,
) -> Result<ReasoningOptions, String> {
    let mut options = ReasoningOptions::default();
    match mode {
        "on" => options.enabled = Some(true),
        "off" => options.enabled = Some(false),
        "auto" => {}
        _ => return Err("--think must be on, off, or auto".into()),
    }
    if effort != "auto" {
        let parsed = parse_reasoning_effort(effort)
            .ok_or("--reasoning-effort must be auto, minimal, low, medium, high, xhigh, or max")?;
        if options.enabled == Some(false) {
            return Err("--reasoning-effort cannot be set while --think is off".into());
        }
        options.enabled = Some(true);
        options.effort = Some(parsed);
    }
    match preserve {
        "on" => options.preserve_thinking = Some(true),
        "off" => options.preserve_thinking = Some(false),
        "auto" => {}
        _ => return Err("--preserve-thinking must be on, off, or auto".into()),
    }
    Ok(options)
}

/// `std::from_chars` into an `int`, as the port environment variables use.
fn parse_port(value: &str) -> Option<i32> {
    if value.starts_with('+') {
        return None;
    }
    value.parse().ok()
}

/// The shortest `--stall-timeout-ms`: deep snapshot captures and
/// post-failure device probes legitimately hold progress for tens of
/// seconds.
const MIN_STALL_TIMEOUT_MS: u64 = 60_000;

/// `HOST:PORT` with a non-empty host and a port from 1 to 65535.
fn is_host_port(value: &str) -> bool {
    value.rsplit_once(':').is_some_and(|(host, port)| {
        !host.is_empty() && port.parse::<u16>().is_ok_and(|port| port != 0)
    })
}

fn valid_server_options(server: &ServerOptions, err: &mut dyn Write) -> bool {
    let message = if !gufo_serve::http::server::is_listen_address(&server.host) {
        "--host must be an IPv4 or IPv6 address"
    } else if !(0..=65535).contains(&server.port) {
        "--port must be between 0 and 65535"
    } else if server.session_count == 0
        || server.max_connections == 0
        || server.max_request_body_bytes == 0
    {
        "server limits must be positive"
    } else if server.sse_ping_interval < -1 {
        "--sse-ping-interval must be -1 or more"
    } else if !server.arbiter.is_empty() && !is_host_port(&server.arbiter) {
        "--arbiter must be HOST:PORT"
    } else if server.verbose && server.log_level.is_some() {
        // Two spellings of one setting conflict rather than take precedence.
        "--verbose is shorthand for --log-level=debug, so it cannot combine with --log-level"
    } else {
        return true;
    };
    let _ = writeln!(err, "Error: {message}");
    false
}

/// Parses `gufo serve` arguments (after `serve`) with the process
/// environment, printing help and usage errors as C++ does.
pub fn parse_serve(args: &[String]) -> ServeParse {
    let env = |name: &str| std::env::var_os(name).map(|value| value.to_string_lossy().into_owned());
    parse_serve_with(
        args,
        &env,
        &mut std::io::stdout().lock(),
        &mut std::io::stderr().lock(),
    )
}

/// [`parse_serve`] over an injected environment and output streams.
pub fn parse_serve_with(
    args: &[String],
    env: &dyn Fn(&str) -> Option<String>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ServeParse {
    let mut server = ServerOptions::default();
    let non_empty = |name: &str| env(name).filter(|value| !value.is_empty());
    if let Some(host) = non_empty("HOST").or_else(|| non_empty("GUFO_HOST")) {
        server.host = host;
    }
    if let Some(port) = non_empty("PORT") {
        if let Some(port) = parse_port(&port) {
            server.port = port;
        }
    } else if let Some(port) = non_empty("GUFO_PORT").and_then(|port| parse_port(&port)) {
        server.port = port;
    }

    // Identify the modality only after leading server options. The selected
    // parser consumes every option together, so `--served-model-name -v`
    // and `--model audio` cannot be mistaken for server flags or modalities.
    let mut subcommand = "llm".to_owned();
    let mut sub_args: Vec<String> = args.to_vec();
    {
        let mut scan_targets = ServerOptions::default();
        let mut scanner = ArgParser::new("gufo serve", "");
        add_server_options(&mut scanner, &mut scan_targets, true);
        let mut i = 0;
        while i < args.len() {
            let arg = args[i].as_str();
            let equals = arg.find('=');
            let name = equals.map_or(arg, |equals| &arg[..equals]);
            if let Some(option) = scanner.find_option(name) {
                if !option.is_flag && equals.is_none() {
                    i += 1;
                }
                i += 1;
                continue;
            }
            if MODALITIES.contains(&arg) {
                subcommand = arg.to_owned();
                sub_args.remove(i);
            } else if matches!(arg, "--help" | "-h" | "help") {
                let topic = if arg == "help" && i + 1 < args.len() {
                    args[i + 1].as_str()
                } else {
                    ""
                };
                if !topic.is_empty() && !MODALITIES.contains(&topic) {
                    let _ = writeln!(err, "Error: unknown serve command '{topic}'");
                    return ServeParse::Exit(2);
                }
                let _ = write!(out, "{}", serve_help("gufo", topic));
                return ServeParse::Exit(0);
            }
            break;
        }
    }

    let help = |out: &mut dyn Write, topic: &str| {
        let _ = write!(out, "{}", serve_help("gufo", topic));
    };
    let run = |server: ServerOptions, modality: Modality| {
        ServeParse::Run(Box::new(ServeRequest { server, modality }))
    };
    match subcommand.as_str() {
        "image" => {
            let (mut model, mut name) = (PathBuf::new(), "Qwen-Image-2.1".to_owned());
            let (parsed, help_requested) = {
                let mut parser = image_parser(&mut model, &mut name, &mut server);
                (parser.parse(&sub_args), parser.is_help_requested())
            };
            if let Err(message) = parsed {
                let _ = writeln!(err, "Error: {message}");
                return ServeParse::Exit(2);
            }
            if help_requested {
                help(out, "image");
                return ServeParse::Exit(0);
            }
            if !valid_server_options(&server, err) {
                return ServeParse::Exit(2);
            }
            if model.as_os_str().is_empty() {
                let _ = writeln!(err, "Error: --model <DIR> is required for image server");
                return ServeParse::Exit(2);
            }
            run(
                server,
                Modality::Image {
                    model,
                    served_model_name: name,
                },
            )
        }
        "embedding" => {
            let (mut model, mut name) = (PathBuf::new(), EMBEDDING_MODEL_ID.to_owned());
            let (parsed, help_requested) = {
                let mut parser = embedding_parser(&mut model, &mut name, &mut server);
                (parser.parse(&sub_args), parser.is_help_requested())
            };
            if let Err(message) = parsed {
                let _ = writeln!(err, "Error: {message}");
                return ServeParse::Exit(2);
            }
            if help_requested {
                help(out, "embedding");
                return ServeParse::Exit(0);
            }
            if !valid_server_options(&server, err) {
                return ServeParse::Exit(2);
            }
            if model.as_os_str().is_empty() {
                let _ = writeln!(err, "Error: --model <DIR> is required for embedding server");
                return ServeParse::Exit(2);
            }
            run(
                server,
                Modality::Embedding {
                    model,
                    served_model_name: name,
                },
            )
        }
        "tts" | "asr" => {
            let tts = subcommand == "tts";
            let mut options = SpeechOptions::new(tts);
            let (parsed, help_requested) = {
                let mut parser = speech_parser(tts, &mut options, &mut server);
                (parser.parse(&sub_args), parser.is_help_requested())
            };
            if let Err(message) = parsed {
                let _ = writeln!(err, "Error: {message}");
                return ServeParse::Exit(2);
            }
            if help_requested {
                help(out, &subcommand);
                return ServeParse::Exit(0);
            }
            if !valid_server_options(&server, err) {
                return ServeParse::Exit(2);
            }
            if options.model.as_os_str().is_empty() {
                let _ = writeln!(
                    err,
                    "Error: --model <DIR> is required for {subcommand} server"
                );
                return ServeParse::Exit(2);
            }
            let minimum = if tts { 1 } else { 32 };
            if options.context < minimum {
                let _ = writeln!(err, "Error: --context must be at least {minimum}");
                return ServeParse::Exit(2);
            }
            run(
                server,
                if tts {
                    Modality::Tts(options)
                } else {
                    Modality::Asr(options)
                },
            )
        }
        _ => {
            let mut cli = LlmCli::new();
            let (parsed, help_requested, supplied, per_client_supplied) = {
                let mut parser = llm_parser(&mut cli, &mut server);
                let parsed = parser.parse(&sub_args);
                (
                    parsed,
                    parser.is_help_requested(),
                    sampling_options_supplied(&parser),
                    parser.was_supplied("--max-pending-per-client"),
                )
            };
            cli.sampling_supplied = supplied;
            // The per-client cap defaults to the whole queue; an explicit value stays bounded by it.
            if !per_client_supplied {
                cli.max_pending_requests_per_client = cli.max_pending_requests;
            }
            if let Err(message) = parsed {
                let _ = writeln!(err, "Error: {message}");
                help(out, "llm");
                return ServeParse::Exit(2);
            }
            if help_requested {
                help(out, "llm");
                return ServeParse::Exit(0);
            }
            if !valid_server_options(&server, err) {
                return ServeParse::Exit(2);
            }
            // A quieter threshold would discard every progress line.
            if cli.log_progress && server.resolved_log_level() > LogLevel::Info {
                let _ = writeln!(
                    err,
                    "Error: --log-progress needs --log-level=info or --log-level=debug"
                );
                return ServeParse::Exit(2);
            }
            match resolve_llm(cli) {
                Ok(options) => run(server, Modality::Llm(Box::new(options))),
                Err(message) => {
                    let _ = writeln!(err, "Error: {message}");
                    ServeParse::Exit(2)
                }
            }
        }
    }
}

/// The checks of `serve llm` after parsing, in C++ order.
fn resolve_llm(mut cli: LlmCli) -> Result<LlmOptions, String> {
    let sampling_valid = parse_sampling_config(&Value::object(), &cli.sampling).is_ok();
    // `std::chrono::milliseconds` holds at most i64::MAX.
    if cli.max_tokens < -1
        || cli.max_tokens == 0
        || cli.max_tokens > i64::from(u32::MAX)
        || cli.prefill_chunk_tokens == 0
        || cli.max_pending_requests == 0
        || cli.max_pending_requests_per_client == 0
        || cli.max_pending_requests_per_client > cli.max_pending_requests
        || cli.max_output_bytes == 0
        || cli.max_buffered_output_bytes == 0
        || cli.max_buffered_output_bytes_total == 0
        || (!cli.cache_disk_directory.as_os_str().is_empty() && cli.cache_disk_bytes == 0)
        || cli.request_timeout_ms > i64::MAX as u64
        || cli.stall_timeout_ms > i64::MAX as u64
        || (cli.stall_timeout_ms != 0 && cli.stall_timeout_ms < MIN_STALL_TIMEOUT_MS)
        || !sampling_valid
    {
        return Err("sampling and scheduling limits are invalid".into());
    }
    if cli.draft_tokens == 0
        || cli.min_draft_tokens == 0
        || cli.min_draft_tokens > cli.draft_tokens
        || cli.draft_tokens > u32::MAX as usize
    {
        return Err("speculative draft limits are invalid".into());
    }
    let mut reasoning = resolve_reasoning_defaults(
        &cli.reasoning_mode,
        &cli.reasoning_effort,
        &cli.preserve_thinking,
    )?;
    if cli.reasoning_budget < -1 {
        return Err("--reasoning-budget must be -1 or nonnegative".into());
    }
    reasoning.budget_tokens = usize::try_from(cli.reasoning_budget).ok();
    let responses_reasoning_target =
        ResponsesReasoningTarget::parse(&cli.responses_reasoning_target)
            .ok_or("--responses-reasoning-target must be summary, reasoning-text, or reasoning")?;
    let api_compat =
        ApiCompat::parse(&cli.api_compat).ok_or("--api-compat must be gufo or llama.cpp")?;
    if cli.speculative_backend.is_empty() && !cli.dspark_model_path.is_empty() {
        cli.speculative_backend = "dspark".into();
    }
    let speculative = match cli.speculative_backend.as_str() {
        "" | "off" => SpeculativeBackend::Disabled,
        "dflash2" => SpeculativeBackend::DFlash,
        "dspark" => SpeculativeBackend::DSpark,
        "mtp" => SpeculativeBackend::Mtp,
        other => {
            return Err(format!(
                "speculative backend '{other}' is not supported by the HTTP server"
            ));
        }
    };
    if !cli.draft_policy.is_empty() && speculative != SpeculativeBackend::DFlash {
        return Err("--draft-policy requires DFlash2".into());
    }
    let dflash_policy = match cli.draft_policy.as_str() {
        "fixed" => DFlashDraftPolicy::Fixed,
        "" | "adaptive" => DFlashDraftPolicy::Adaptive,
        _ => return Err("DFlash2 draft policy must be fixed or adaptive".into()),
    };
    if speculative == SpeculativeBackend::DFlash
        && (cli.dflash_model_path.is_empty() || cli.min_draft_tokens != 1)
    {
        return Err(
            "DFlash2 requires --dflash-model and --min-draft-tokens 1; bound blocks \
                    with --draft-tokens"
                .into(),
        );
    }
    let draft_model_path = match speculative {
        SpeculativeBackend::DSpark => cli.dspark_model_path,
        SpeculativeBackend::Mtp => cli.mtp_model_path,
        _ => cli.dflash_model_path,
    };
    if cli.model.is_empty() {
        return Err("--model <PATH> is required".into());
    }
    Ok(LlmOptions {
        model: cli.model,
        vision_model: cli.vision_model_path,
        served_model_name: cli.served_model_name,
        max_context: cli.max_context,
        max_tokens: usize::try_from(cli.max_tokens).unwrap_or(0),
        sampling: cli.sampling,
        sampling_supplied: cli.sampling_supplied,
        reasoning,
        responses_reasoning_target,
        api_compat,
        enable_yarn: cli.enable_yarn,
        speculative,
        draft_model_path,
        max_draft_tokens: cli.draft_tokens as u32,
        min_draft_tokens: cli.min_draft_tokens as u32,
        dflash_policy,
        prefill: TextPrefillPolicy {
            decode_active_tokens: cli.prefill_chunk_tokens,
        },
        scheduler: TextSchedulerPolicy {
            max_pending_requests: cli.max_pending_requests,
            max_pending_requests_per_client: cli.max_pending_requests_per_client,
            max_output_bytes_per_request: cli.max_output_bytes,
            max_buffered_output_bytes_per_request: cli.max_buffered_output_bytes,
            max_buffered_output_bytes_total: cli.max_buffered_output_bytes_total,
            request_timeout: Duration::from_millis(cli.request_timeout_ms),
            stall_timeout: Duration::from_millis(cli.stall_timeout_ms),
            log_progress: cli.log_progress,
            ..TextSchedulerPolicy::default()
        },
        cache_ram_bytes: cli.cache_ram_bytes,
        cache_disk_directory: cli.cache_disk_directory,
        cache_disk_bytes: cli.cache_disk_bytes,
        cache_disk_staging_bytes: cli.cache_disk_staging_bytes,
    })
}

/// `ReadTextFile`: the file's text without trailing line breaks, or `None`
/// when it cannot be opened. Unreadable content counts as empty (a
/// directory opens but reads nothing), and invalid UTF-8 is replaced.
fn read_text_file(path: &Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut bytes = Vec::new();
    let _ = file.read_to_end(&mut bytes);
    let mut text = String::from_utf8_lossy(&bytes).into_owned();
    while text.ends_with('\n') || text.ends_with('\r') {
        text.pop();
    }
    Some(text)
}

/// `ResolveReferenceText`: an existing file's contents, otherwise the value
/// itself as the transcript.
fn resolve_reference_text(value: &str) -> String {
    let path = Path::new(value);
    if path.is_file()
        && let Some(text) = read_text_file(path)
    {
        return text;
    }
    value.to_owned()
}

/// `BuildVoicePresets`: the registered voices from `--voice NAME=WAV` with
/// their `--voice-text` and `--voice-lang` overrides, in any flag order.
/// C++ runs it after the load log started; its error goes to
/// `Error: <message>` with exit code 2.
pub fn build_voice_presets(
    options: &SpeechOptions,
) -> Result<BTreeMap<String, TtsVoicePreset>, String> {
    let mut presets = BTreeMap::new();
    for (name, wav) in &options.voice_specs {
        if presets.contains_key(name) {
            return Err(format!("duplicate --voice name '{name}'"));
        }
        let wav_path = Path::new(wav);
        let Ok(mut file) = std::fs::File::open(wav_path) else {
            return Err(format!(
                "cannot open voice reference {}",
                wav_path.display()
            ));
        };
        let mut bytes = Vec::new();
        let _ = file.read_to_end(&mut bytes);
        let reference_audio =
            decode_wav(&bytes).map_err(|error| format!("{}: {error}", wav_path.display()))?;
        // An explicit --voice-text wins; otherwise a `.txt` sidecar beside the
        // WAV. With neither, the voice clones from the speaker embedding
        // alone, which needs no transcript.
        let reference_text = match options.voice_text_specs.get(name) {
            Some(text) => resolve_reference_text(text),
            None => read_text_file(&wav_path.with_extension("txt")).unwrap_or_default(),
        };
        presets.insert(
            name.clone(),
            TtsVoicePreset {
                reference_audio,
                speaker_embedding_only: reference_text.is_empty(),
                reference_text,
                language: options
                    .voice_lang_specs
                    .get(name)
                    .cloned()
                    .unwrap_or_default(),
            },
        );
    }
    if let Some(name) = options
        .voice_text_specs
        .keys()
        .find(|name| !presets.contains_key(*name))
    {
        return Err(format!("--voice-text names unknown voice '{name}'"));
    }
    if let Some(name) = options
        .voice_lang_specs
        .keys()
        .find(|name| !presets.contains_key(*name))
    {
        return Err(format!("--voice-lang names unknown voice '{name}'"));
    }
    Ok(presets)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| (*s).to_owned()).collect()
    }

    /// Runs the parser with `env` and returns (result, stdout, stderr).
    fn parse(args: &[&str], env: &[(&str, &str)]) -> (ServeParse, String, String) {
        let env: Vec<(String, String)> = env
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        let lookup = move |name: &str| {
            env.iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
        };
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let result = parse_serve_with(&strings(args), &lookup, &mut out, &mut err);
        (
            result,
            String::from_utf8_lossy(&out).into_owned(),
            String::from_utf8_lossy(&err).into_owned(),
        )
    }

    fn fixture(name: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/serve_help")
            .join(name);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    #[test]
    fn help_texts_match_cpp() {
        for (topic, file) in [
            ("", "serve.txt"),
            ("llm", "llm.txt"),
            ("image", "image.txt"),
            ("tts", "tts.txt"),
            ("asr", "asr.txt"),
            ("embedding", "embedding.txt"),
        ] {
            assert_eq!(serve_help("gufo", topic), fixture(file), "{topic}");
        }
        assert!(serve_help("x", "llm").starts_with("Usage: x serve llm [OPTIONS]\n"));
    }

    /// Every case of `tests/fixtures/serve_cli_cases.json`: arguments,
    /// environment and the C++ binary's exit code, stderr and stdout (a help
    /// fixture or nothing), recorded from `gufo serve` of the C++ build.
    #[test]
    fn usage_errors_and_help_match_cpp() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/serve_cli_cases.json");
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let cases: serde_json::Value =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}"));
        let cases = cases.as_array().unwrap_or_else(|| panic!("case list"));
        assert!(cases.len() > 40, "{}", cases.len());
        for case in cases {
            let text = |key: &str| case[key].as_str().unwrap_or_default().to_owned();
            let args: Vec<String> = case["args"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_owned))
                        .collect()
                })
                .unwrap_or_default();
            let env: Vec<(String, String)> = case["env"]
                .as_object()
                .map(|o| {
                    o.iter()
                        .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_owned()))
                        .collect()
                })
                .unwrap_or_default();
            let args: Vec<&str> = args.iter().map(String::as_str).collect();
            let env: Vec<(&str, &str)> =
                env.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
            let (result, out, err) = parse(&args, &env);
            let exit = case["exit"].as_u64().unwrap_or(99) as u8;
            let expected_out = case["stdout_help"]
                .as_str()
                .map(fixture)
                .unwrap_or_default();
            assert_eq!(result, ServeParse::Exit(exit), "{args:?} {env:?}");
            assert_eq!(out, expected_out, "{args:?} {env:?}");
            assert_eq!(err, text("stderr"), "{args:?} {env:?}");
        }
    }

    /// `--max-pending-per-client` defaults to `--max-pending`, so lowering
    /// the queue alone is valid; an explicit value stays bounded by it
    /// (`serve_test.py`).
    #[test]
    fn per_client_queue_cap_defaults_to_the_queue() {
        let caps = |extra: &[&str]| {
            let args: Vec<&str> = ["llm", "--model", "missing.gguf"]
                .iter()
                .chain(extra)
                .copied()
                .collect();
            let Modality::Llm(options) = run_request(&args, &[]).modality else {
                panic!("llm");
            };
            (
                options.scheduler.max_pending_requests,
                options.scheduler.max_pending_requests_per_client,
            )
        };
        assert_eq!(caps(&[]), (16, 16));
        assert_eq!(caps(&["--max-pending", "2"]), (2, 2));
        assert_eq!(caps(&["--max-pending", "1"]), (1, 1));
        assert_eq!(
            caps(&["--max-pending", "2", "--max-pending-per-client", "2"]),
            (2, 2)
        );
        assert_eq!(caps(&["--max-pending-per-client", "16"]), (16, 16));
        assert_eq!(caps(&["--max-pending-per-client", "3"]), (16, 3));
        for extra in [
            &["--max-pending", "2", "--max-pending-per-client", "3"][..],
            &["--max-pending-per-client", "17"],
            &["--max-pending-per-client", "0"],
            &["--max-pending", "0"],
        ] {
            let args: Vec<&str> = ["llm", "--model", "missing.gguf"]
                .iter()
                .chain(extra)
                .copied()
                .collect();
            let (result, _, err) = parse(&args, &[]);
            assert!(
                matches!(result, ServeParse::Exit(2))
                    && err.contains("sampling and scheduling limits are invalid"),
                "{extra:?}: {result:?} {err}"
            );
        }
    }

    fn run_request(args: &[&str], env: &[(&str, &str)]) -> ServeRequest {
        match parse(args, env) {
            (ServeParse::Run(request), _, _) => *request,
            (other, out, err) => panic!("{args:?}: {other:?}\n{out}{err}"),
        }
    }

    /// `--log-level` and its `-v` shorthand select one threshold; both
    /// together conflict, and `--log-progress` needs the info tier.
    #[test]
    fn log_level_options() {
        let load = ["llm", "--model", "missing.gguf"];
        let level = |extra: &[&str]| {
            let args: Vec<&str> = load.iter().chain(extra).copied().collect();
            run_request(&args, &[]).server.resolved_log_level()
        };
        assert_eq!(level(&[]), LogLevel::Info);
        for spelling in [&["-v"][..], &["--verbose"], &["--log-level", "debug"]] {
            assert_eq!(level(spelling), LogLevel::Debug, "{spelling:?}");
        }
        assert_eq!(level(&["--log-level=warn"]), LogLevel::Warn);
        // Server options are accepted on either side of the modality.
        assert_eq!(
            run_request(&["--log-level", "error", "llm", "--model", "m.gguf"], &[])
                .server
                .resolved_log_level(),
            LogLevel::Error
        );
        let fails = |extra: &[&str], message: &str| {
            let args: Vec<&str> = load.iter().chain(extra).copied().collect();
            let (result, _, err) = parse(&args, &[]);
            assert!(
                matches!(result, ServeParse::Exit(2)) && err.contains(message),
                "{extra:?}: {result:?} {err}"
            );
        };
        for both in [
            &["--log-level", "info", "-v"][..],
            &["-v", "--log-level", "warn"],
            &["--verbose", "--log-level=debug"],
        ] {
            fails(both, "cannot combine with --log-level");
        }
        for bad in ["trace", "Debug", "", "info,debug"] {
            fails(
                &["--log-level", bad],
                "--log-level must be error, warn, info or debug",
            );
        }
        for quiet in ["warn", "error"] {
            fails(
                &["--log-progress", "--log-level", quiet],
                "--log-progress needs --log-level=info",
            );
        }
        for chatty in [&["--log-level=info"][..], &["--log-level=debug"], &["-v"]] {
            let args: Vec<&str> = load
                .iter()
                .chain(&["--log-progress"])
                .chain(chatty)
                .copied()
                .collect();
            let _ = run_request(&args, &[]);
        }
        for topic in ["", "llm", "tts", "asr", "image", "embedding"] {
            let help = serve_help("gufo", topic);
            assert!(
                help.contains("Logging:")
                    && !help.contains("Server Options:")
                    && help.contains("--log-level")
                    && help.contains("-v, --verbose"),
                "{topic}"
            );
        }
    }

    /// llama-server's `--sse-ping-interval`, in seconds; -1 disables.
    #[test]
    fn sse_ping_interval_option() {
        let interval = |extra: &[&str]| {
            let args: Vec<&str> = ["llm", "--model", "m.gguf"]
                .iter()
                .chain(extra)
                .copied()
                .collect();
            run_request(&args, &[]).server.sse_ping_interval
        };
        assert_eq!(interval(&[]), 15);
        assert_eq!(interval(&["--sse-ping-interval", "30"]), 30);
        assert_eq!(interval(&["--sse-ping-interval=-1"]), -1);
        let (result, _, err) = parse(&["llm", "--model", "m.gguf", "--sse-ping-interval=-2"], &[]);
        assert!(
            matches!(result, ServeParse::Exit(2))
                && err.contains("--sse-ping-interval must be -1 or more"),
            "{err}"
        );
    }

    /// `--api-compat` selects Gufo's or llama.cpp's request semantics; Gufo
    /// is the default and no other value is accepted.
    #[test]
    fn api_compat_option() {
        let compat = |extra: &[&str]| {
            let args: Vec<&str> = ["llm", "--model", "m.gguf"]
                .iter()
                .chain(extra)
                .copied()
                .collect();
            let Modality::Llm(options) = run_request(&args, &[]).modality else {
                panic!("llm");
            };
            options.api_compat
        };
        assert_eq!(compat(&[]), ApiCompat::Gufo);
        assert_eq!(compat(&["--api-compat", "gufo"]), ApiCompat::Gufo);
        assert_eq!(compat(&["--api-compat", "llama.cpp"]), ApiCompat::LlamaCpp);
        for value in ["llama-cpp", "upstream", ""] {
            let (result, _, err) = parse(&["llm", "--model", "m.gguf", "--api-compat", value], &[]);
            assert!(
                matches!(result, ServeParse::Exit(2))
                    && err.contains("--api-compat must be gufo or llama.cpp"),
                "{value}: {err}"
            );
        }
    }

    #[test]
    fn responses_reasoning_target_option() {
        let target = |extra: &[&str]| {
            let args: Vec<&str> = ["llm", "--model", "m.gguf"]
                .iter()
                .chain(extra)
                .copied()
                .collect();
            let Modality::Llm(options) = run_request(&args, &[]).modality else {
                panic!("llm");
            };
            options.responses_reasoning_target
        };
        assert_eq!(target(&[]), ResponsesReasoningTarget::Summary);
        for (value, expected) in [
            ("summary", ResponsesReasoningTarget::Summary),
            ("reasoning-text", ResponsesReasoningTarget::ReasoningText),
            ("reasoning", ResponsesReasoningTarget::Reasoning),
        ] {
            assert_eq!(target(&["--responses-reasoning-target", value]), expected);
        }
        // One target only: no lists, no aliases.
        for value in ["summary,reasoning-text", "reasoning_text", "none"] {
            let (result, _, err) = parse(
                &[
                    "llm",
                    "--model",
                    "m.gguf",
                    "--responses-reasoning-target",
                    value,
                ],
                &[],
            );
            assert!(
                matches!(result, ServeParse::Exit(2))
                    && err.contains(
                        "--responses-reasoning-target must be summary, reasoning-text, or reasoning"
                    ),
                "{value}: {err}"
            );
        }
    }

    #[test]
    fn llm_options_resolve() {
        let request = run_request(
            &[
                "-p",
                "9000",
                "llm",
                "-m",
                "model.gguf",
                "--mmproj",
                "vision.gguf",
                "-c",
                "8192",
                "-n",
                "64",
                "-t",
                "0.7",
                "--seed",
                "3",
                "--think",
                "on",
                "--reasoning-effort",
                "high",
                "--preserve-thinking",
                "off",
                "--speculative",
                "dflash2",
                "--dflash-model",
                "draft.gguf",
                "--draft-policy",
                "fixed",
                "-d",
                "5",
                "--prefill-chunk",
                "256",
                "--max-pending",
                "8",
                "--max-pending-per-client",
                "2",
                "--request-timeout-ms",
                "1500",
                "--cache-disk",
                "cache",
                "-j",
                "4",
                "--api-key=k",
                "-v",
            ],
            &[("HOST", "0.0.0.0")],
        );
        assert_eq!(
            request.server,
            ServerOptions {
                host: "0.0.0.0".into(),
                port: 9000,
                session_count: 4,
                api_key: "k".into(),
                verbose: true,
                ..ServerOptions::default()
            }
        );
        let Modality::Llm(options) = request.modality else {
            panic!("llm");
        };
        assert_eq!(options.model, "model.gguf");
        assert_eq!(options.vision_model, "vision.gguf");
        assert_eq!((options.max_context, options.max_tokens), (8192, 64));
        assert_eq!(options.sampling.temperature, 0.7);
        assert_eq!(options.sampling.seed, 3);
        assert_eq!(
            options.reasoning,
            ReasoningOptions {
                enabled: Some(true),
                effort: Some(ReasoningEffort::High),
                preserve_thinking: Some(false),
                budget_tokens: None,
            }
        );
        assert_eq!(options.speculative, SpeculativeBackend::DFlash);
        assert_eq!(options.speculative.name(), "dflash2");
        assert_eq!(options.draft_model_path, "draft.gguf");
        assert_eq!((options.max_draft_tokens, options.min_draft_tokens), (5, 1));
        assert_eq!(options.dflash_policy, DFlashDraftPolicy::Fixed);
        assert_eq!(options.prefill.decode_active_tokens, 256);
        assert_eq!(options.scheduler.max_pending_requests, 8);
        assert_eq!(options.scheduler.max_pending_requests_per_client, 2);
        assert_eq!(
            options.scheduler.request_timeout,
            Duration::from_millis(1500)
        );
        assert_eq!(options.cache_disk_directory, PathBuf::from("cache"));
        assert_eq!(options.cache_disk_bytes, 8 << 30);
        assert_eq!(options.cache_disk_staging_bytes, 0, "automatic staging");

        // Defaults, and DSpark selected by its model alone.
        let request = run_request(
            &["--dspark-model", "s.gguf", "-m", "m.gguf"],
            &[("PORT", "x"), ("GUFO_PORT", "7")],
        );
        assert_eq!(request.server.port, 8080);
        let Modality::Llm(options) = request.modality else {
            panic!("llm");
        };
        assert_eq!(options.model, "m.gguf");
        assert_eq!(options.speculative, SpeculativeBackend::DSpark);
        assert_eq!(options.draft_model_path, "s.gguf");
        assert_eq!(options.dflash_policy, DFlashDraftPolicy::Adaptive);
        // The per-client cap defaults to the whole queue.
        assert_eq!(
            options.scheduler,
            TextSchedulerPolicy {
                max_pending_requests_per_client: 16,
                ..TextSchedulerPolicy::default()
            }
        );
        assert_eq!(options.prefill, TextPrefillPolicy::default());
        assert_eq!(options.reasoning, ReasoningOptions::default());
        assert_eq!(options.sampling, SamplingConfig::default());

        let request = run_request(
            &["--dflash-model", "d.gguf", "-m", "m.gguf"],
            &[("GUFO_PORT", "7"), ("GUFO_HOST", "1.2.3.4")],
        );
        assert_eq!(
            (request.server.host.as_str(), request.server.port),
            ("1.2.3.4", 7)
        );
        let Modality::Llm(options) = request.modality else {
            panic!("llm");
        };
        // Without a backend the DFlash2 path is still the draft path.
        assert_eq!(options.speculative, SpeculativeBackend::Disabled);
        assert_eq!(options.draft_model_path, "d.gguf");
    }

    #[test]
    fn llm_defaults_follow_the_model() {
        let limits = |args: &[&str]| {
            let Modality::Llm(options) = run_request(args, &[]).modality else {
                panic!("llm");
            };
            (options.max_context, options.max_tokens)
        };
        assert_eq!(
            limits(&["llm", "-m", "m"]),
            (0, 0),
            "native context, until EOS"
        );
        assert_eq!(limits(&["llm", "-m", "m", "-n", "-1"]), (0, 0));
        assert_eq!(
            limits(&["llm", "-m", "m", "-c", "4096", "-n", "7"]),
            (4096, 7)
        );
    }

    #[test]
    fn the_embedding_modality_resolves() {
        let request = run_request(&["-p", "8090", "embedding", "--model", "eg2"], &[]);
        assert_eq!(request.server.port, 8090);
        assert_eq!(
            request.modality,
            Modality::Embedding {
                model: "eg2".into(),
                served_model_name: "embeddinggemma-2".into(),
            }
        );
        let request = run_request(
            &["embedding", "-m", "eg2", "--served-model-name", "vectors"],
            &[],
        );
        assert!(matches!(
            request.modality,
            Modality::Embedding { served_model_name, .. } if served_model_name == "vectors"
        ));
        for (args, message) in [
            (
                &["embedding"][..],
                "Error: --model <DIR> is required for embedding server\n",
            ),
            (
                &["embedding", "-m", "x", "-j", "2"],
                "Error: Unknown option: -j\n",
            ),
            (
                &["embedding", "-m", "x", "--context", "8"],
                "Error: Unknown option: --context\n",
            ),
        ] {
            let (parsed, _, err) = parse(args, &[]);
            assert_eq!(parsed, ServeParse::Exit(2), "{args:?}");
            assert_eq!(err, message, "{args:?}");
        }
        let (parsed, out, _) = parse(&["help", "embedding"], &[]);
        assert_eq!(parsed, ServeParse::Exit(0));
        assert_eq!(out, fixture("embedding.txt"));
    }

    #[test]
    fn media_modalities_resolve() {
        let request = run_request(&["image", "-m", "img", "--served-model-name", "pic"], &[]);
        assert_eq!(
            request.modality,
            Modality::Image {
                model: "img".into(),
                served_model_name: "pic".into(),
            }
        );
        let request = run_request(
            &[
                "tts",
                "-m",
                "tts",
                "--voice",
                "a=a.wav",
                "--voice-text=a=hello",
                "--voice-lang",
                "a=german",
                "-c",
                "1",
            ],
            &[],
        );
        let Modality::Tts(options) = request.modality else {
            panic!("tts");
        };
        assert_eq!(options.context, 1);
        assert_eq!(options.voice_specs, [("a".to_owned(), "a.wav".to_owned())]);
        assert_eq!(options.voice_text_specs["a"], "hello");
        assert_eq!(options.voice_lang_specs["a"], "german");
        let request = run_request(&["asr", "--model", "asr"], &[]);
        let Modality::Asr(options) = request.modality else {
            panic!("asr");
        };
        assert_eq!(options.context, 1024);
    }

    fn wav(path: &Path) {
        // 16 kHz mono PCM16, four samples.
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&44u32.to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&16000u32.to_le_bytes());
        bytes.extend_from_slice(&32000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&[0, 0, 0, 64, 0, 192, 0, 0]);
        std::fs::write(path, bytes).unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn voice_presets_follow_the_cpp_rules() {
        let directory = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let dir = directory.path();
        let (a, b, c) = (dir.join("a.wav"), dir.join("b.wav"), dir.join("c.wav"));
        wav(&a);
        wav(&b);
        wav(&c);
        std::fs::write(dir.join("a.txt"), "sidecar text\r\n\n").unwrap_or_else(|e| panic!("{e}"));
        let transcript = dir.join("transcript.txt");
        std::fs::write(&transcript, "from file\n").unwrap_or_else(|e| panic!("{e}"));
        let mut options = SpeechOptions::new(true);
        let spec = |name: &str, path: &Path| (name.to_owned(), path.display().to_string());
        options.voice_specs = vec![spec("a", &a), spec("b", &b), spec("c", &c)];
        options
            .voice_text_specs
            .insert("b".into(), transcript.display().to_string());
        options.voice_lang_specs.insert("c".into(), "german".into());
        let presets = build_voice_presets(&options).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(presets["a"].reference_text, "sidecar text");
        assert!(!presets["a"].speaker_embedding_only);
        assert_eq!(presets["b"].reference_text, "from file");
        assert_eq!(presets["c"].reference_text, "");
        assert!(presets["c"].speaker_embedding_only);
        assert_eq!(presets["c"].language, "german");
        assert_eq!(presets["a"].reference_audio.sample_rate, 16000);

        options
            .voice_text_specs
            .insert("b".into(), "inline words".into());
        let presets = build_voice_presets(&options).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(presets["b"].reference_text, "inline words");

        let mut duplicate = options.clone();
        duplicate.voice_specs.push(spec("a", &a));
        assert_eq!(
            build_voice_presets(&duplicate),
            Err("duplicate --voice name 'a'".into())
        );
        let mut missing = options.clone();
        let absent = dir.join("absent.wav");
        missing.voice_specs = vec![spec("z", &absent)];
        missing.voice_text_specs.clear();
        missing.voice_lang_specs.clear();
        assert_eq!(
            build_voice_presets(&missing),
            Err(format!("cannot open voice reference {}", absent.display()))
        );
        let bad = dir.join("bad.wav");
        std::fs::write(&bad, b"not a wav").unwrap_or_else(|e| panic!("{e}"));
        missing.voice_specs = vec![spec("z", &bad)];
        assert_eq!(
            build_voice_presets(&missing),
            Err(format!(
                "{}: Qwen3-TTS reference audio must be a RIFF WAV",
                bad.display()
            ))
        );
        let mut unknown = options.clone();
        unknown.voice_text_specs.insert("q".into(), "x".into());
        unknown.voice_lang_specs.insert("p".into(), "x".into());
        assert_eq!(
            build_voice_presets(&unknown),
            Err("--voice-text names unknown voice 'q'".into())
        );
        unknown.voice_text_specs.remove("q");
        assert_eq!(
            build_voice_presets(&unknown),
            Err("--voice-lang names unknown voice 'p'".into())
        );
    }

    #[test]
    fn reasoning_defaults_resolve() {
        assert_eq!(
            resolve_reasoning_defaults("auto", "low", "on"),
            Ok(ReasoningOptions {
                enabled: Some(true),
                effort: Some(ReasoningEffort::Low),
                preserve_thinking: Some(true),
                budget_tokens: None,
            })
        );
        assert_eq!(
            resolve_reasoning_defaults("off", "auto", "auto"),
            Ok(ReasoningOptions {
                enabled: Some(false),
                ..ReasoningOptions::default()
            })
        );
    }
}
