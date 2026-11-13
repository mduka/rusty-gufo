//! `gufo serve`, after parsing (`RunServe` of the C++ server): loads the
//! modality's model, builds its service and runs the HTTP server. Dense
//! Qwen3.8, Qwen3.6 35B-A3B, Qwen3.8-Flash-Next, DeepSeek V4 Flash, Nemotron
//! 3.5 Bolt and Gemma 4 (text) run under the pool and scheduler; ASR, TTS
//! (Qwen3-TTS, OmniVoice, VoxCPM2), image and embeddings have their own
//! services.

use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use gufo_core::sampling::text_preset_of;
use gufo_core::text::{ChatRequest, Error, GenerationErrorCode, Result, TextModelRunner};
use gufo_qwen_image_21 as qwen_image;
use gufo_qwen3_asr::{Audio, Request as AsrRequest, Transcriber};
use gufo_qwen3_tts as tts;
use gufo_qwen35::dflash::policy::DraftPolicy;
use gufo_qwen35::runner::{
    DFlashOptions as QwenDFlashOptions, QwenRunner, RunnerOptions as QwenRunnerOptions,
};
use gufo_serve::arbiter::{ArbiterClient, Identity, PlaybackBuffer, Presence, StatusSource};
use gufo_serve::asr_model::{AsrRunner, TranscriptionResult, TranscriptionTimings};
use gufo_serve::asr_service::{AsrService, AsrServiceOptions};
use gufo_serve::backend::TextGenerationBackend;
use gufo_serve::gpu_share::GpuShare;
use gufo_serve::http::HttpServerOptions;
use gufo_serve::http::server::{HttpServer, HttpServices, LoadingResponder};
use gufo_serve::image_api::{
    Image as ServeImage, ImageCodec, ImageMetrics, ImageResult, ImageRunner, ImageService,
};
use gufo_serve::logging;
use gufo_serve::pool::{TextRunnerDiskCacheOptions, TextRunnerPool};
use gufo_serve::scheduled_backend::ScheduledBackend;
use gufo_serve::tts_model::{
    ConditioningCapabilities, ModelVariant as TtsVariant, SynthesisResult as TtsResult, TtsRunner,
};
use gufo_serve::tts_service::{TtsRuntimeFactory, TtsService, TtsServiceOptions};

use crate::gpu_queues::{
    QueueProfile, describe_queue_plan, describe_queue_pressure, plan_queues, query_queue_census,
};
use crate::serve_cli::{self, LlmOptions, Modality, ServeParse, SpeechOptions};

mod embedding;
mod omnivoice;
mod voxcpm;

/// `ModelLoadLog`: start, completion or failure of a model load.
struct ModelLoadLog {
    kind: String,
    start: Instant,
    completed: bool,
}

impl ModelLoadLog {
    fn new(kind: &str, artifact: &Path) -> Self {
        logging::info(
            "loader",
            &format!(
                "event=load_started kind={kind} artifact={} {}",
                artifact.display(),
                logging::memory_status()
            ),
        );
        Self {
            kind: kind.into(),
            start: Instant::now(),
            completed: false,
        }
    }

    fn complete(&mut self, details: &str, gpu_loaded: bool) {
        self.completed = true;
        let mut message = format!(
            "event=load_completed kind={} elapsed_ms={} {details} {}",
            self.kind,
            self.start.elapsed().as_millis(),
            logging::memory_status()
        );
        if gpu_loaded && let Ok((free, total)) = gufo_hip::device_memory() {
            message.push_str(&format!(
                " gpu_device_used_mib={} gpu_device_total_mib={}",
                (total - free) / (1024 * 1024),
                total / (1024 * 1024)
            ));
        }
        logging::info("loader", &message);
    }
}

impl Drop for ModelLoadLog {
    fn drop(&mut self) {
        if !self.completed {
            logging::error(
                "loader",
                &format!(
                    "event=load_failed kind={} {}",
                    self.kind,
                    logging::memory_status()
                ),
            );
        }
    }
}

/// Asks the HIP runtime for no more hardware compute queues than the
/// modality needs and the device has left, before anything starts it, and
/// logs the plan; an overfull device is warned about, not refused.
fn budget_hardware_queues(modality: &Modality) {
    let (kind, profile) = match modality {
        Modality::Llm(_) => ("llm", QueueProfile::Text),
        Modality::Tts(_) => ("tts", QueueProfile::Audio),
        Modality::Asr(_) => ("asr", QueueProfile::Audio),
        Modality::Image { .. } => ("image", QueueProfile::Unmeasured),
        Modality::Embedding { .. } => ("embedding", QueueProfile::Unmeasured),
    };
    let census = query_queue_census(Path::new("/sys"));
    let operator = std::env::var("GPU_MAX_HW_QUEUES").ok();
    let plan = plan_queues(profile, &census, operator.as_deref());
    if plan.cap != 0
        && let Err(error) = gufo_hip::cap_hardware_queues(plan.cap)
    {
        logging::warn(
            "server",
            &format!("event=queue_budget_unapplied error=\"{error}\""),
        );
    }
    logging::info("server", &describe_queue_plan(kind, &plan));
    if plan.may_exceed_budget {
        logging::warn("server", &describe_queue_pressure(&plan));
    }
}

/// Runs `gufo serve` with the arguments after `serve`.
pub fn run(args: &[String]) -> u8 {
    let request = match serve_cli::parse_serve(args) {
        ServeParse::Exit(code) => return code,
        ServeParse::Run(request) => request,
    };
    let server_options = request.server;
    // Armed before the first startup line, so every modality, loader and
    // queue diagnostic follows it.
    let log_level = server_options.resolved_log_level();
    logging::set_level(log_level);
    budget_hardware_queues(&request.modality);
    // The only record of what was asked for when a load fails or hangs
    // before the listening banner.
    logging::debug(
        "server",
        &format!(
            "event=options host={} port={} max_connections={} max_request_bytes={} api_key={} \
             log_level={}",
            server_options.host,
            server_options.port,
            server_options.max_connections,
            server_options.max_request_body_bytes,
            if server_options.api_key.is_empty() {
                "unset"
            } else {
                "set"
            },
            log_level.name()
        ),
    );
    let mut backend: Option<Arc<dyn TextGenerationBackend>> = None;
    let mut services = HttpServices::default();
    // `--arbiter`: text generation follows the arbiter's GPU grants and a
    // streamed TTS reports its playback buffer; other servers only report.
    let arbiter = (!server_options.arbiter.is_empty()).then(|| server_options.arbiter.clone());
    let gpu_share = arbiter.as_ref().map(|_| Arc::new(GpuShare::default()));
    let playback = arbiter
        .as_ref()
        .map(|_| Arc::new(PlaybackBuffer::default()));
    let modality_name = match &request.modality {
        Modality::Llm(_) => "llm",
        Modality::Tts(_) => "tts",
        Modality::Asr(_) => "asr",
        Modality::Image { .. } => "image",
        Modality::Embedding { .. } => "embedding",
    };
    let (responses_reasoning_target, api_compat) = match &request.modality {
        Modality::Llm(options) => (options.responses_reasoning_target, options.api_compat),
        _ => Default::default(),
    };
    match request.modality {
        Modality::Image {
            model,
            served_model_name,
        } => match load_image(&model, &served_model_name) {
            Ok(service) => services.images = Some(Arc::new(service)),
            Err(code) => return code,
        },
        Modality::Tts(options) => match load_tts(&options, playback.clone()) {
            Ok(service) => services.tts = Some(Arc::new(service)),
            Err(code) => return code,
        },
        Modality::Asr(options) => match load_asr(&options) {
            Ok(service) => services.asr = Some(Arc::new(service)),
            Err(code) => return code,
        },
        Modality::Embedding {
            model,
            served_model_name,
            default_prompt_name,
        } => match embedding::load(&model, &served_model_name, &default_prompt_name) {
            Ok(service) => services.embeddings = Some(Arc::new(service)),
            Err(code) => return code,
        },
        Modality::Llm(mut options) => {
            options.scheduler.gpu_share = gpu_share.clone();
            // Loading, the disk snapshot tier included, can take minutes:
            // the port answers 503 `Loading model` meanwhile, as
            // llama-server's does, so supervisors see a loading server
            // rather than a refused connection.
            let mut loading = LoadingResponder::start(&server_options.host, server_options.port);
            let loaded = load_text(&options, server_options.session_count);
            loading.stop();
            match loaded {
                Ok(loaded) => backend = Some(loaded),
                Err(code) => return code,
            }
        }
    }

    let text_backend = backend.clone();
    let model_id = backend
        .as_ref()
        .map(|b| b.model_id())
        .or_else(|| services.tts.as_ref().map(|t| t.model_id()))
        .or_else(|| services.asr.as_ref().map(|a| a.model_id()))
        .or_else(|| services.embeddings.as_ref().map(|e| e.model_id()))
        .or_else(|| services.images.as_ref().map(|i| i.model_id().to_owned()))
        .unwrap_or_default();
    let host = server_options.host.clone();
    let server = match HttpServer::new(
        &server_options.host,
        server_options.port,
        backend,
        services,
        HttpServerOptions {
            max_request_body_bytes: server_options.max_request_body_bytes,
            max_connections: server_options.max_connections,
            api_key: server_options.api_key,
            sse_heartbeat_interval: std::time::Duration::from_secs(
                server_options.sse_ping_interval.max(0).unsigned_abs(),
            ),
            responses_reasoning_target,
            api_compat,
        },
    ) {
        Ok(server) => server,
        Err(error) => {
            eprintln!("Error starting HTTP server: {error}");
            return 1;
        }
    };
    if let Err(error) = server.start() {
        eprintln!("Error starting HTTP server: {error}");
        return 1;
    }
    // Reports from the bound address until `run` returns.
    let _arbiter = match arbiter {
        Some(address) => {
            let source: Arc<dyn StatusSource> = match (&gpu_share, &playback, modality_name) {
                (Some(share), _, "llm") => share.clone(),
                (_, Some(playback), "tts") => playback.clone(),
                _ => Arc::new(Presence),
            };
            let identity = Identity {
                modality: modality_name.to_owned(),
                model: model_id,
                address: format!("{host}:{}", server.port()),
                pid: std::process::id(),
            };
            let share = gpu_share.filter(|_| modality_name == "llm");
            match ArbiterClient::start(address, identity, source, share) {
                Ok(client) => Some(client),
                Err(error) => {
                    eprintln!("Error starting the arbiter connection: {error}");
                    return 1;
                }
            }
        }
        None => None,
    };
    // `run` joins every request thread, so no hook call outlives it.
    let fatal = Arc::new(OnceLock::new());
    {
        let fatal = fatal.clone();
        server.on_fatal_failure(move |code| {
            let _ = fatal.set(code);
            shutdown_for_restart(code);
        });
    }
    server.run(true);
    // Releasing model state on a lost device can block or crash, and a
    // failed scheduler left it mid-step; the kernel reclaims it. The backend
    // is checked too: an external SIGTERM can stop the server after the
    // failure but before a request ran the hook.
    if let Some(code) = fatal
        .get()
        .copied()
        .or_else(|| text_backend.and_then(|b| b.fatal_failure()))
    {
        gufo_hip::exit_immediately(restart_exit_status(code));
    }
    0
}

/// The exit status after the GPU context is lost (`EX_TEMPFAIL`): only a
/// new process recovers, so a supervisor must restart it.
const DEVICE_LOST_EXIT_STATUS: i32 = 75;

/// The exit status after the scheduler thread panicked (`EX_SOFTWARE`).
const SCHEDULER_FAILED_EXIT_STATUS: i32 = 70;

fn restart_exit_status(code: GenerationErrorCode) -> i32 {
    match code {
        GenerationErrorCode::SchedulerFailed => SCHEDULER_FAILED_EXIT_STATUS,
        _ => DEVICE_LOST_EXIT_STATUS,
    }
}

/// Leaves through the `SIGTERM` shutdown, so a supervisor restarts the
/// server, and bounds that teardown: joining requests or freeing device
/// memory may block on the dead device. The watchdog is armed before
/// logging or shutdown, so neither can prevent the exit.
fn shutdown_for_restart(code: GenerationErrorCode) {
    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
    let status = restart_exit_status(code);
    let watchdog = std::thread::Builder::new()
        .name("gufo-restart".into())
        .spawn(move || {
            std::thread::sleep(TIMEOUT);
            gufo_hip::exit_immediately(status)
        });
    if watchdog.is_err() {
        gufo_hip::exit_immediately(status);
    }
    logging::error(
        "server",
        &format!(
            "event={}_shutdown exit_status={status} timeout_s={}",
            code.stable_code(),
            TIMEOUT.as_secs()
        ),
    );
    let _ = rustix::process::kill_process(rustix::process::getpid(), rustix::process::Signal::TERM);
}

fn load_image(model: &Path, name: &str) -> std::result::Result<ImageService, u8> {
    let mut load_log = ModelLoadLog::new("qwen_image_21", model);
    // C++ checks the ID before loading the model; the placeholder runner is
    // never called because construction fails.
    let service = if name.is_empty() {
        ImageService::new(
            name,
            Box::new(|_, _| Err(Error::logic("Qwen-Image-2.1 is not loaded"))),
            Arc::new(QwenImageCodec),
        )
    } else {
        image_runner(model)
            .and_then(|runner| ImageService::new(name, runner, Arc::new(QwenImageCodec)))
    };
    match service {
        Ok(service) => {
            load_log.complete(
                &format!("model={name} weights=mapped upload=on_demand"),
                true,
            );
            Ok(service)
        }
        Err(error) => {
            eprintln!("Error loading Qwen-Image-2.1: {}", error.message());
            Err(1)
        }
    }
}

/// Qwen-Image-2.1 on the Rust engine, in the shape `ImageService` takes.
fn image_runner(model: &Path) -> Result<ImageRunner> {
    let model = qwen_image::Model::load(model)?;
    Ok(Box::new(move |request, cancelled| {
        let native = qwen_image::Request {
            prompt: request.prompt.clone(),
            images: request
                .images
                .iter()
                .map(|image| qwen_image::Image {
                    width: image.width,
                    height: image.height,
                    rgba: image.rgba.clone(),
                })
                .collect(),
            width: request.width,
            height: request.height,
            steps: request.steps,
            seed: request.seed,
        };
        let cancelled = cancelled.map(|check| &**check as &dyn Fn() -> bool);
        let generation = model.generate(&native, cancelled, None)?;
        let metrics = generation.metrics;
        Ok(ImageResult {
            image: ServeImage {
                width: generation.image.width,
                height: generation.image.height,
                rgba: generation.image.rgba,
            },
            metrics: ImageMetrics {
                prompt_ms: metrics.prompt_ms,
                denoise_ms: metrics.denoise_ms,
                vae_ms: metrics.vae_ms,
                total_ms: metrics.total_ms,
                resident_weight_bytes: metrics.resident_weight_bytes,
            },
        })
    }))
}

/// PNG/JPEG decoding and PNG encoding of the Qwen-Image engine.
struct QwenImageCodec;

impl ImageCodec for QwenImageCodec {
    fn decode(&self, bytes: &[u8]) -> Result<ServeImage> {
        let image = qwen_image::image::decode_image(bytes)?;
        Ok(ServeImage {
            width: image.width,
            height: image.height,
            rgba: image.rgba,
        })
    }

    fn encode_png(&self, image: &ServeImage) -> Result<Vec<u8>> {
        qwen_image::image::encode_rgba_png(image.width, image.height, &image.rgba)
    }
}

/// A speech model of the extensions (OmniVoice: a directory; VoxCPM2: a
/// directory or an audio.cpp GGUF file): an automatic voice and voice
/// design besides cloning, on its own runtime instead of Qwen3-TTS's.
struct NativeTts {
    name: &'static str,
    model_id: &'static str,
    sample_rate: std::result::Result<u32, String>,
    conditioning: ConditioningCapabilities,
    voices: [&'static str; 2],
    factory: TtsRuntimeFactory,
}

fn load_tts(
    options: &SpeechOptions,
    playback: Option<Arc<PlaybackBuffer>>,
) -> std::result::Result<TtsService, u8> {
    let mut load_log = ModelLoadLog::new("tts", &options.model);
    let voice_presets = match serve_cli::build_voice_presets(options) {
        Ok(presets) => presets,
        Err(error) => {
            eprintln!("Error: {error}");
            return Err(2);
        }
    };
    let native = if omnivoice::probe(&options.model) {
        Some(NativeTts {
            name: "OmniVoice",
            model_id: omnivoice::MODEL_ID,
            sample_rate: omnivoice::sample_rate(&options.model),
            conditioning: omnivoice::conditioning(),
            voices: [omnivoice::AUTO_VOICE, omnivoice::DESIGN_VOICE],
            factory: Box::new(|root: &Path, _, _| omnivoice::runtime(root)),
        })
    } else if voxcpm::probe(&options.model) {
        Some(NativeTts {
            name: "VoxCPM2",
            model_id: voxcpm::MODEL_ID,
            sample_rate: voxcpm::sample_rate(&options.model),
            conditioning: voxcpm::conditioning(),
            voices: [voxcpm::AUTO_VOICE, voxcpm::DESIGN_VOICE],
            factory: Box::new(|root: &Path, _, _| voxcpm::runtime(root)),
        })
    } else {
        None
    };
    let service = if let Some(native) = native {
        let sample_rate = match native.sample_rate {
            Ok(rate) => rate,
            Err(error) => {
                eprintln!("Error enabling {} TTS service: {error}", native.name);
                return Err(1);
            }
        };
        let model_id = if options.served_model_name.is_empty() {
            native.model_id.to_string()
        } else {
            options.served_model_name.clone()
        };
        TtsService::new(TtsServiceOptions {
            model_root: options.model.clone(),
            native_context_tokens: options.context,
            validate_model: false,
            sample_rate,
            variant: TtsVariant::Base,
            conditioning: Some(native.conditioning),
            model_id,
            voices: native.voices.map(str::to_string).to_vec(),
            voice_presets,
            runtime_factory: Some(native.factory),
            playback,
            ..TtsServiceOptions::default()
        })
    } else {
        TtsService::new(TtsServiceOptions {
            model_root: options.model.clone(),
            native_context_tokens: options.context,
            validate_model: true,
            model_id: options.served_model_name.clone(),
            voice_presets,
            runtime_factory: Some(Box::new(tts_runtime)),
            playback,
            ..TtsServiceOptions::default()
        })
    };
    if !service.ready() {
        eprintln!(
            "Error enabling TTS service: {}",
            service.initialization_error()
        );
        return Err(1);
    }
    load_log.complete(
        &format!(
            "model={} sessions=1 context_tokens={}",
            service.model_id(),
            options.context
        ),
        true,
    );
    Ok(service)
}

/// The native Qwen3-TTS runtime on the Rust engine
/// (`SynthesisHipRuntime`), created once the service has validated the
/// checkpoint; requests run one at a time.
fn tts_runtime(
    model_root: &Path,
    context: usize,
    variant: TtsVariant,
) -> std::result::Result<TtsRunner, String> {
    let variant = match variant {
        TtsVariant::Base => tts::config::ModelVariant::Base,
        TtsVariant::VoiceDesign => tts::config::ModelVariant::VoiceDesign,
        TtsVariant::CustomVoice => tts::config::ModelVariant::CustomVoice,
        TtsVariant::Unsupported => tts::config::ModelVariant::Unsupported,
    };
    let synthesizer = std::sync::Mutex::new(tts::synthesis::Synthesizer::new(
        model_root, context, variant,
    )?);
    Ok(Box::new(move |request, on_audio, is_cancelled| {
        let s = &request.sampling;
        let native = tts::synthesis::SynthesisRequest {
            text: request.text.clone(),
            speaker: request.speaker.clone(),
            language: request.language.clone(),
            instruct: request.instruct.clone(),
            reference_audio: tts::audio::AudioBuffer {
                sample_rate: request.reference_audio.sample_rate,
                channels: request.reference_audio.channels,
                samples: request.reference_audio.samples.clone(),
            },
            reference_text: request.reference_text.clone(),
            speaker_embedding_only: request.speaker_embedding_only,
            max_new_tokens: request.max_new_tokens,
            sampling: tts::sampling::SamplingOptions {
                sample: s.sample,
                seed: s.seed,
                top_k: s.top_k,
                temperature: s.temperature,
                top_p: s.top_p,
                predictor_sample: s.predictor_sample,
                predictor_top_k: s.predictor_top_k,
                predictor_temperature: s.predictor_temperature,
                predictor_top_p: s.predictor_top_p,
                repetition_penalty: s.repetition_penalty,
            },
        };
        let mut synthesizer = synthesizer.lock().unwrap_or_else(|e| e.into_inner());
        let result = match on_audio {
            Some(sink) => synthesizer.generate(&native, is_cancelled, Some(sink))?,
            None => synthesizer.generate(&native, is_cancelled, None)?,
        };
        Ok(TtsResult {
            sample_rate: result.sample_rate,
            code_groups: result.code_groups,
            samples: result.samples,
            codes: result.codes,
            sample_count: result.sample_count,
        })
    }))
}

fn load_asr(options: &SpeechOptions) -> std::result::Result<AsrService, u8> {
    let mut load_log = ModelLoadLog::new("asr", &options.model);
    let service = AsrService::new(AsrServiceOptions {
        model_root: options.model.clone(),
        native_context_tokens: options.context,
        validate_model: true,
        model_validator: Some(Box::new(|root| {
            gufo_qwen3_asr::config::ModelConfig::load(root).is_ok_and(|c| c.is_supported())
        })),
        model_id: options.served_model_name.clone(),
        runner: None,
        runtime_factory: Some(Box::new(asr_runtime)),
    });
    if !service.ready() {
        eprintln!(
            "Error enabling Qwen3-ASR service: {}",
            service.initialization_error()
        );
        return Err(1);
    }
    load_log.complete(
        &format!(
            "model={} sessions=1 context_tokens={}",
            service.model_id(),
            options.context
        ),
        true,
    );
    Ok(service)
}

/// The native Qwen3-ASR runtime on the Rust engine, which admits requests
/// per audio chunk (`TranscriptionHipRuntime`).
fn asr_runtime(model_root: &Path, context: usize) -> std::result::Result<AsrRunner, String> {
    let transcriber = Transcriber::load(model_root, context)?;
    Ok(Box::new(move |request, on_text, cancelled| {
        // The APIs never send both or neither; C++ checks this after
        // admission, with this message.
        let audio = match (request.wav.is_empty(), request.pcm16k.is_empty()) {
            (false, true) => Audio::Wav(&request.wav),
            (true, false) => Audio::Pcm16k(&request.pcm16k),
            _ => return Err("Qwen3-ASR requires exactly one audio source".into()),
        };
        let transcription = transcriber.transcribe(
            &AsrRequest {
                audio,
                context: &request.context,
                language: request.language.as_deref(),
                max_new_tokens: request.max_new_tokens,
            },
            cancelled,
            on_text,
        )?;
        let timings = transcription.timings;
        Ok(TranscriptionResult {
            language: transcription.language,
            text: transcription.text,
            decoded: String::from_utf8_lossy(&transcription.decoded).into_owned(),
            generated_ids: transcription.generated_ids,
            sample_rate: transcription.sample_rate,
            audio_samples: transcription.audio_samples,
            mel_frames: transcription.mel_frames,
            audio_tokens: transcription.audio_tokens,
            prompt_tokens: transcription.prompt_tokens,
            chunks: transcription.chunks,
            timings: TranscriptionTimings {
                decode_audio_ms: timings.decode_audio_ms,
                feature_extraction_ms: timings.feature_extraction_ms,
                audio_encoder_ms: timings.audio_encoder_ms,
                text_decoder_ms: timings.text_decoder_ms,
                total_ms: timings.total_ms,
            },
        })
    }))
}

/// The backend over `runner` with the request policy of `options`.
fn start_backend<R: TextModelRunner>(
    runner: R,
    disk_cache: Option<TextRunnerDiskCacheOptions>,
    options: &LlmOptions,
    session_count: usize,
) -> Result<ScheduledBackend<R>> {
    let pool = TextRunnerPool::with_ram_cache(
        Arc::new(runner),
        session_count,
        disk_cache,
        options.cache_ram,
    )?;
    let backend =
        ScheduledBackend::new(Arc::new(pool), options.prefill, options.scheduler.clone())?;
    backend.set_model_ids(options.served_model_names.clone());
    // As `gufo --version` prints it.
    backend.set_served_artifact(
        &options.model,
        &format!("{} ({})", crate::RELEASE_VERSION, crate::REVISION),
    );
    backend.set_sampling_preset(text_preset_of(Path::new(&options.model)));
    backend.set_sampling_defaults(
        options.max_tokens,
        &options.sampling,
        options.sampling_supplied,
    )?;
    backend.set_reasoning_defaults(options.reasoning);
    Ok(backend)
}

/// Starts the runner of the text model at `options.model`: Gemma 4,
/// Flash-Next, DeepSeek V4 Flash, Nemotron 3.5 Bolt, Qwen3.6 35B-A3B
/// (`qwen35moe`), or else dense Qwen (`qwen35`, with the 27B's vision
/// sidecar), autoregressive or with DFlash2, whose loader reports any other
/// artifact.
fn start_text(
    options: &LlmOptions,
    session_count: usize,
) -> Result<Arc<dyn TextGenerationBackend>> {
    let model = Path::new(&options.model);
    if gufo_gemma4::runner::supports(model) {
        return start_gemma4(options, session_count);
    }
    if let Some(flag) = options.gemma4.first_given() {
        return Err(Error::invalid(format!("{flag} applies only to Gemma 4")));
    }
    if options.speculative == serve_cli::SpeculativeBackend::Mtp && !options.draft_policy.is_empty()
    {
        return Err(Error::invalid(
            "--draft-policy applies to DFlash2 and Gemma 4 MTP",
        ));
    }
    if gufo_qwen38_flash_next::runner::supports(model) {
        return start_flash_next(options, session_count);
    }
    if options.enable_yarn {
        return Err(Error::invalid(
            "--enable-yarn applies only to Qwen3.8-Flash-Next",
        ));
    }
    if gufo_deepseek_v4_flash::runner::supports(model) {
        return start_deepseek(options, session_count);
    }
    if gufo_nemotron::runner::supports(model) {
        return start_nemotron(options, session_count);
    }
    if gufo_qwen35moe::runner::is_qwen35moe(model) {
        return start_qwen35moe(options, session_count);
    }
    // Dense Qwen with the refusals of the C++ `InferenceBackend::load` (its
    // `QwenGpuModel` overload); its loader reports any other artifact.
    let dflash = match options.speculative {
        serve_cli::SpeculativeBackend::Disabled => None,
        serve_cli::SpeculativeBackend::DFlash => Some(QwenDFlashOptions {
            model_path: options.draft_model_path.clone().into(),
            max_draft_tokens: options.max_draft_tokens,
            min_draft_tokens: options.min_draft_tokens,
            policy: match options.dflash_policy {
                serve_cli::DFlashDraftPolicy::Fixed => DraftPolicy::Fixed,
                serve_cli::DFlashDraftPolicy::Adaptive => DraftPolicy::Adaptive,
            },
        }),
        serve_cli::SpeculativeBackend::DSpark => {
            return Err(Error::invalid(
                "DSpark HTTP decoding requires a DeepSeek model",
            ));
        }
        serve_cli::SpeculativeBackend::Mtp => {
            return Err(Error::invalid(
                "MTP HTTP decoding requires a Qwen Flash-Next or Gemma 4 model",
            ));
        }
    };
    let disk_cache = !options.cache_disk_directory.as_os_str().is_empty();
    let runner = QwenRunner::load(&QwenRunnerOptions {
        model_path: model.into(),
        max_context: options.max_context,
        persistence: disk_cache,
        dflash,
        vision_model_path: (!options.vision_model.is_empty())
            .then(|| options.vision_model.clone().into()),
    })?;
    prepare_vision(runner.vision_encoder())?;
    let disk_cache = disk_cache.then(|| TextRunnerDiskCacheOptions {
        directory: options.cache_disk_directory.clone(),
        capacity_bytes: options.cache_disk_bytes,
        staging_capacity_bytes: options.cache_disk_staging_bytes,
        ..TextRunnerDiskCacheOptions::default()
    });
    Ok(Arc::new(start_backend(
        runner,
        disk_cache,
        options,
        session_count,
    )?))
}

/// Qwen3.6 35B-A3B (`qwen35moe`) text serving, autoregressive, with DFlash2
/// or with its MTP head, with the checks of the extension's
/// `CreateQwenTextRunner`; the extension's `--kv-cache q8_0` is not offered.
fn start_qwen35moe(
    options: &LlmOptions,
    session_count: usize,
) -> Result<Arc<dyn TextGenerationBackend>> {
    use gufo_qwen35moe::dflash::draft::Draft;
    use gufo_qwen35moe::dflash::policy::DraftPolicy;
    use gufo_qwen35moe::mtp::MtpBackend;
    use gufo_qwen35moe::runner::{
        DFlashOptions, MtpOptions, QwenRunner, RunnerOptions, ServeDraft,
    };
    let (dflash, mtp) = match options.speculative {
        serve_cli::SpeculativeBackend::Disabled => (None, None),
        serve_cli::SpeculativeBackend::DFlash => (
            Some(DFlashOptions {
                model_path: options.draft_model_path.clone().into(),
                max_draft_tokens: options.max_draft_tokens,
                min_draft_tokens: options.min_draft_tokens,
                policy: match options.dflash_policy {
                    serve_cli::DFlashDraftPolicy::Fixed => DraftPolicy::Fixed,
                    serve_cli::DFlashDraftPolicy::Adaptive => DraftPolicy::Adaptive,
                },
            }),
            None,
        ),
        serve_cli::SpeculativeBackend::Mtp => (
            None,
            Some(MtpOptions {
                model_path: options.draft_model_path.clone().into(),
                max_draft_tokens: options.max_draft_tokens,
                min_draft_tokens: options.min_draft_tokens,
            }),
        ),
        serve_cli::SpeculativeBackend::DSpark => {
            return Err(Error::invalid(
                "DSpark HTTP decoding requires a DeepSeek model",
            ));
        }
    };
    let disk_cache = !options.cache_disk_directory.as_os_str().is_empty();
    let runner_options = RunnerOptions {
        model_path: options.model.clone().into(),
        max_context: options.max_context,
        persistence: disk_cache,
        dflash,
        sessions: session_count,
        vision_model_path: (!options.vision_model.is_empty())
            .then(|| options.vision_model.clone().into()),
        mtp,
    };
    // One runner type per draft kind.
    fn start<D: ServeDraft>(
        runner_options: &RunnerOptions,
        options: &LlmOptions,
        session_count: usize,
    ) -> Result<Arc<dyn TextGenerationBackend>> {
        let runner = QwenRunner::<D>::load(runner_options)?;
        prepare_vision(runner.vision_encoder())?;
        let disk_cache = runner_options
            .persistence
            .then(|| TextRunnerDiskCacheOptions {
                directory: options.cache_disk_directory.clone(),
                capacity_bytes: options.cache_disk_bytes,
                staging_capacity_bytes: options.cache_disk_staging_bytes,
                ..TextRunnerDiskCacheOptions::default()
            });
        Ok(Arc::new(start_backend(
            runner,
            disk_cache,
            options,
            session_count,
        )?))
    }
    if runner_options.mtp.is_some() {
        start::<MtpBackend>(&runner_options, options, session_count)
    } else {
        start::<Draft>(&runner_options, options, session_count)
    }
}

/// Hashes the vision sidecar and uploads its weights while the model loads,
/// before the pool claims state capacity and the RAM cache is sized
/// (upstream `PrepareVision`): otherwise the first image request
/// hashes the whole sidecar in its queue time and uploads the weights in its
/// prefill, after admission and the cache were sized without them.
fn prepare_vision(encoder: Option<&gufo_qwen::vision::Encoder>) -> Result<()> {
    let Some(encoder) = encoder else {
        return Ok(());
    };
    let started = std::time::Instant::now();
    let _ = encoder.identity();
    let hashed = std::time::Instant::now();
    encoder.prepare().map_err(|error| {
        Error::runtime(format!(
            "could not upload the Qwen vision sidecar: {}",
            error.message()
        ))
    })?;
    let uploaded = std::time::Instant::now();
    let heap_free = gufo_hip::device_memory().map_or(0, |(free, _)| free);
    logging::info(
        "loader",
        &format!(
            "event=vision_prepared resident_bytes={} identity_ms={:.6} upload_ms={:.6} \
             device_heap_free_bytes={heap_free} {}",
            encoder.resident_bytes(),
            (hashed - started).as_secs_f64() * 1e3,
            (uploaded - hashed).as_secs_f64() * 1e3,
            logging::memory_status()
        ),
    );
    Ok(())
}

/// Qwen3.8-Flash-Next text serving, with the checks the C++
/// `InferenceBackend::load` makes for a `qwen4exp` model.
fn start_flash_next(
    options: &LlmOptions,
    session_count: usize,
) -> Result<Arc<dyn TextGenerationBackend>> {
    use gufo_qwen38_flash_next::runner::{FlashNextRunner, MtpOptions, RunnerOptions};
    let mtp = match options.speculative {
        serve_cli::SpeculativeBackend::Disabled => None,
        serve_cli::SpeculativeBackend::Mtp => {
            if options.draft_model_path.is_empty() {
                return Err(Error::invalid(
                    "Qwen3.8-Flash-Next MTP HTTP decoding requires --mtp-model",
                ));
            }
            if options.max_draft_tokens == 0 || options.min_draft_tokens != 1 {
                return Err(Error::invalid(
                    "Flash-Next MTP requires a positive draft limit and --min-draft-tokens 1",
                ));
            }
            Some(MtpOptions {
                model_path: options.draft_model_path.clone().into(),
                max_draft_tokens: options.max_draft_tokens,
            })
        }
        _ => {
            return Err(Error::invalid(
                "Qwen3.8-Flash-Next HTTP models support only MTP speculative decoding \
                 (--speculative mtp --mtp-model)",
            ));
        }
    };
    let disk_cache = !options.cache_disk_directory.as_os_str().is_empty();
    let runner = FlashNextRunner::load(&RunnerOptions {
        model_path: options.model.clone().into(),
        max_context: options.max_context,
        persistence: disk_cache,
        mtp,
        sessions: session_count,
        vision_model_path: (!options.vision_model.is_empty())
            .then(|| options.vision_model.clone().into()),
        enable_yarn: options.enable_yarn,
    })?;
    let expanded = runner.expanded_q8_0_tensors();
    if expanded != 0 {
        logging::warn(
            "loader",
            &format!(
                "event=q8_0_expanded tensors={expanded} \
                 reason=hc_inject_and_ssm_alpha_beta_are_q8_0 (the official GGUF keeps them \
                 F32; expanded to F32 at load, re-quantize them as F32)"
            ),
        );
    }
    if options.enable_yarn {
        let native = gufo_qwen38_flash_next::runner::native_context(Path::new(&options.model));
        if let Some(scaling) = gufo_qwen38_flash_next::rope_scaling::rope_scaling_for_context(
            options.max_context,
            native,
        ) {
            logging::info(
                "loader",
                &format!(
                    "event=yarn factor={} original_context={native} context={}",
                    scaling.factor, options.max_context
                ),
            );
        }
    }
    prepare_vision(runner.vision_encoder())?;
    let disk_cache = disk_cache.then(|| TextRunnerDiskCacheOptions {
        directory: options.cache_disk_directory.clone(),
        capacity_bytes: options.cache_disk_bytes,
        staging_capacity_bytes: options.cache_disk_staging_bytes,
        ..TextRunnerDiskCacheOptions::default()
    });
    Ok(Arc::new(start_backend(
        runner,
        disk_cache,
        options,
        session_count,
    )?))
}

/// DeepSeek V4 Flash text serving, autoregressive or with DSpark, with the
/// checks of the C++ `InferenceBackend::load`; the draft limits are checked
/// before the model loads.
fn start_deepseek(
    options: &LlmOptions,
    session_count: usize,
) -> Result<Arc<dyn TextGenerationBackend>> {
    use gufo_deepseek_v4_flash::runner::{DeepSeekRunner, RunnerOptions};
    if !options.vision_model.is_empty() {
        return Err(Error::invalid("DeepSeek does not support --mmproj"));
    }
    let use_dspark = match options.speculative {
        serve_cli::SpeculativeBackend::Disabled => false,
        serve_cli::SpeculativeBackend::DSpark => true,
        _ => {
            return Err(Error::invalid(
                "DeepSeek HTTP models support only DSpark speculative decoding",
            ));
        }
    };
    if use_dspark && options.draft_model_path.is_empty() {
        return Err(Error::invalid(
            "DeepSeek DSpark HTTP decoding requires --dspark-model",
        ));
    }
    if use_dspark
        && (options.max_draft_tokens == 0
            || options.min_draft_tokens == 0
            || options.min_draft_tokens > options.max_draft_tokens)
    {
        return Err(Error::invalid("DeepSeek DSpark draft limits are invalid"));
    }
    if use_dspark && options.min_draft_tokens != 1 {
        return Err(Error::invalid(
            "DSpark uses model-owned adaptive drafting; custom draft floors are unsupported",
        ));
    }
    let disk_cache = !options.cache_disk_directory.as_os_str().is_empty();
    if disk_cache && options.cache_disk_bytes == 0 {
        return Err(Error::invalid(
            "DeepSeek persistent disk cache configuration is invalid",
        ));
    }
    let runner = DeepSeekRunner::load(&RunnerOptions {
        model_path: options.model.clone().into(),
        max_context: options.max_context,
        persistence: disk_cache,
        dspark_model_path: use_dspark.then(|| options.draft_model_path.clone().into()),
        max_draft_tokens: options.max_draft_tokens,
    })?;
    let disk_cache = disk_cache.then(|| TextRunnerDiskCacheOptions {
        directory: options.cache_disk_directory.clone(),
        capacity_bytes: options.cache_disk_bytes,
        staging_capacity_bytes: options.cache_disk_staging_bytes,
        ..TextRunnerDiskCacheOptions::default()
    });
    Ok(Arc::new(start_backend(
        runner,
        disk_cache,
        options,
        session_count,
    )?))
}

/// Nemotron 3.5 Bolt text serving, autoregressive or with NVIDIA's LiLiCorr
/// drafter, and with its vision projector for images (the C++ extension's
/// `CreateNemotronTextRunner` checks). Continuations stay in RAM: the runner
/// has no persistent snapshot format.
fn start_nemotron(
    options: &LlmOptions,
    session_count: usize,
) -> Result<Arc<dyn TextGenerationBackend>> {
    use gufo_nemotron::runner::{NemotronRunner, RunnerOptions};
    let drafting = match options.speculative {
        serve_cli::SpeculativeBackend::Disabled => false,
        serve_cli::SpeculativeBackend::DFlash => true,
        _ => {
            return Err(Error::invalid(
                "Nemotron drafts with NVIDIA's LiLiCorr drafter only: \
                 --speculative dflash2 --dflash-model DRAFTER.gguf",
            ));
        }
    };
    if !options.cache_disk_directory.as_os_str().is_empty() {
        return Err(Error::invalid(
            "Nemotron does not support the persistent disk cache",
        ));
    }
    let runner = NemotronRunner::load(&RunnerOptions {
        model_path: options.model.clone().into(),
        max_context: options.max_context,
        draft_model_path: drafting.then(|| options.draft_model_path.clone().into()),
        max_draft_tokens: options.max_draft_tokens,
        vision_model_path: (!options.vision_model.is_empty())
            .then(|| options.vision_model.clone().into()),
    })?;
    // The runner prepared the projector's matrix products while loading.
    if let Some(encoder) = runner.vision_encoder() {
        let heap_free = gufo_hip::device_memory().map_or(0, |(free, _)| free);
        logging::info(
            "loader",
            &format!(
                "event=vision_prepared resident_bytes={} device_heap_free_bytes={heap_free} {}",
                encoder.resident_bytes(),
                logging::memory_status()
            ),
        );
    }
    Ok(Arc::new(start_backend(
        runner,
        None,
        options,
        session_count,
    )?))
}

/// The draft limit the loaded backend uses: Gemma 4 defaults to its own.
fn draft_limit(options: &LlmOptions) -> u32 {
    if !options.draft_tokens_supplied && gufo_gemma4::runner::supports(Path::new(&options.model)) {
        gufo_gemma4::engine::DEFAULT_DRAFT_TOKENS
    } else {
        options.max_draft_tokens
    }
}

/// Gemma 4 text serving, autoregressive or with its `gemma4-assistant` MTP
/// drafter, with the BF16 vision sidecar for images and the RAM and disk
/// caches (the extension's `CreateGemma4TextRunner` checks).
fn start_gemma4(
    options: &LlmOptions,
    session_count: usize,
) -> Result<Arc<dyn TextGenerationBackend>> {
    use gufo_gemma4::draft_policy::{DraftCalibrationScope, DraftPolicy};
    use gufo_gemma4::engine::DEFAULT_DRAFT_TOKENS;
    use gufo_gemma4::provider_options::{PrefillWeights, image_tokens};
    use gufo_gemma4::runner::{Gemma4Runner, RunnerOptions};
    let mtp = match options.speculative {
        serve_cli::SpeculativeBackend::Disabled => false,
        serve_cli::SpeculativeBackend::Mtp => true,
        _ => {
            return Err(Error::invalid(
                "Gemma 4 HTTP models support only MTP speculative decoding \
                 (--speculative mtp --mtp-model)",
            ));
        }
    };
    let draft_tokens = if options.draft_tokens_supplied {
        options.max_draft_tokens
    } else {
        DEFAULT_DRAFT_TOKENS
    };
    if mtp && (options.draft_model_path.is_empty() || draft_tokens == 0) {
        return Err(Error::invalid(
            "Gemma 4 MTP requires --mtp-model and a positive draft limit",
        ));
    }
    let gemma4 = &options.gemma4;
    let persistence = !options.cache_disk_directory.as_os_str().is_empty();
    if persistence && options.cache_disk_bytes == 0 {
        return Err(Error::invalid(
            "Gemma 4 persistent disk cache configuration is invalid",
        ));
    }
    let runner = Gemma4Runner::load(&RunnerOptions {
        model_path: options.model.clone().into(),
        max_context: options.max_context,
        mtp_model_path: mtp.then(|| options.draft_model_path.clone().into()),
        draft_tokens,
        min_draft_tokens: options.min_draft_tokens,
        draft_policy: DraftPolicy::parse(&options.draft_policy).map_err(Error::invalid)?,
        draft_calibration: DraftCalibrationScope::parse(&gemma4.draft_calibration)
            .map_err(Error::invalid)?,
        draft_trace_path: (!gemma4.draft_trace.is_empty())
            .then(|| gemma4.draft_trace.clone().into()),
        prefill_weights: PrefillWeights::parse(&gemma4.prefill_weights).map_err(Error::invalid)?,
        vision_model_path: (!options.vision_model.is_empty())
            .then(|| options.vision_model.clone().into()),
        image_tokens: image_tokens(gemma4.image_tokens).map_err(Error::invalid)?,
        persistence,
    })?;
    // The runner hashed and uploaded the sidecar while loading.
    if let Some(encoder) = runner.vision_encoder() {
        let heap_free = gufo_hip::device_memory().map_or(0, |(free, _)| free);
        logging::info(
            "loader",
            &format!(
                "event=vision_prepared resident_bytes={} device_heap_free_bytes={heap_free} {}",
                encoder.resident_bytes(),
                logging::memory_status()
            ),
        );
    }
    let disk_cache = persistence.then(|| TextRunnerDiskCacheOptions {
        directory: options.cache_disk_directory.clone(),
        capacity_bytes: options.cache_disk_bytes,
        staging_capacity_bytes: options.cache_disk_staging_bytes,
        ..TextRunnerDiskCacheOptions::default()
    });
    Ok(Arc::new(start_backend(
        runner,
        disk_cache,
        options,
        session_count,
    )?))
}

/// `--context 0`: the native context of the GGUF metadata
/// (`<architecture>.context_length`), per session.
fn resolve_context(options: &LlmOptions) -> Result<u32> {
    if options.max_context != 0 {
        return Ok(options.max_context);
    }
    // An unreadable file fails in the loader, with its message.
    let Ok((header, len)) = gufo_core::gguf::read_header(Path::new(&options.model)) else {
        return Ok(0);
    };
    let Ok(gguf) = gufo_core::gguf::Gguf::parse_prefix(&header, len) else {
        return Ok(0);
    };
    let architecture =
        String::from_utf8_lossy(gguf.string("general.architecture").unwrap_or_default());
    gguf.u64(&format!("{architecture}.context_length"))
        .filter(|&native| native >= 2)
        .and_then(|native| u32::try_from(native).ok())
        .ok_or_else(|| Error::invalid("GGUF has no valid native context length; specify --context"))
}

fn load_text(
    options: &LlmOptions,
    session_count: usize,
) -> std::result::Result<Arc<dyn TextGenerationBackend>, u8> {
    let mut load_log = ModelLoadLog::new("text", Path::new(&options.model));
    let options = &match resolve_context(options) {
        Ok(max_context) => LlmOptions {
            max_context,
            ..options.clone()
        },
        Err(error) => {
            eprintln!(
                "Error loading model '{}': {}",
                options.model,
                error.message()
            );
            return Err(1);
        }
    };
    let backend = match start_text(options, session_count) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!(
                "Error loading model '{}': {}",
                options.model,
                error.message()
            );
            return Err(1);
        }
    };
    let ids = backend.model_ids();
    let served_names = if ids.len() > 1 {
        format!(" served_names={}", ids.join(","))
    } else {
        String::new()
    };
    load_log.complete(
        &format!(
            "model={} sessions={session_count} context_tokens={} speculative={} \
             draft_limit={} disk_cache={}{served_names}",
            backend.model_id(),
            options.max_context,
            options.speculative.name(),
            draft_limit(options),
            if options.cache_disk_directory.as_os_str().is_empty() {
                "off"
            } else {
                "enabled"
            }
        ),
        true,
    );
    log_sampling_defaults(&*backend, options);
    Ok(backend)
}

/// The effective server defaults for the default thinking mode; requests
/// may override each setting.
fn log_sampling_defaults(backend: &dyn TextGenerationBackend, options: &LlmOptions) {
    use gufo_serve::backend::InitialOutputState;
    use gufo_serve::metrics::format_general;
    let sampling = backend
        .sampling_defaults()
        .resolve(options.reasoning.enabled);
    let request = ChatRequest {
        reasoning: options.reasoning,
        ..ChatRequest::default()
    };
    let thinking = match backend.initial_output_state(&request) {
        InitialOutputState::Reasoning => "on",
        InitialOutputState::Content => "off",
        InitialOutputState::Auto => "auto",
    };
    let float = |value: f32| format_general(f64::from(value));
    logging::info(
        "sampling",
        &format!(
            "event=defaults thinking={thinking} temperature={} top_k={} top_p={} min_p={} \
             min_keep={} seed={} repeat_penalty={} repeat_last_n={} frequency_penalty={} \
             presence_penalty={}",
            float(sampling.temperature),
            sampling.top_k,
            float(sampling.top_p),
            float(sampling.min_p),
            sampling.min_keep,
            sampling.seed,
            float(sampling.repeat_penalty),
            sampling.repeat_last_n,
            float(sampling.frequency_penalty),
            float(sampling.presence_penalty),
        ),
    );
}
