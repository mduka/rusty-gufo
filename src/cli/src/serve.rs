//! `gufo serve`, after parsing (`RunServe` of the C++ server): loads the
//! modality's model, builds its service and runs the HTTP server. Dense
//! Qwen3.8, Qwen3.8-Flash-Next and DeepSeek V4 Flash (text) run under the
//! pool and scheduler; ASR, TTS, image and embeddings have their own
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
use gufo_serve::pool::{TextRunnerDiskCacheOptions, TextRunnerPool, TextRunnerRamCacheOptions};
use gufo_serve::scheduled_backend::ScheduledBackend;
use gufo_serve::tts_model::{ModelVariant as TtsVariant, SynthesisResult as TtsResult, TtsRunner};
use gufo_serve::tts_service::{TtsService, TtsServiceOptions};

use crate::gpu_queues::{
    QueueProfile, describe_queue_plan, describe_queue_pressure, plan_queues, query_queue_census,
};
use crate::serve_cli::{self, LlmOptions, Modality, ServeParse, SpeechOptions};

mod embedding;

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
        } => match embedding::load(&model, &served_model_name) {
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
    let service = TtsService::new(TtsServiceOptions {
        model_root: options.model.clone(),
        native_context_tokens: options.context,
        validate_model: true,
        model_id: options.served_model_name.clone(),
        voice_presets,
        runtime_factory: Some(Box::new(tts_runtime)),
        playback,
        ..TtsServiceOptions::default()
    });
    if !service.ready() {
        eprintln!(
            "Error enabling Qwen3-TTS service: {}",
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
        TextRunnerRamCacheOptions {
            capacity_bytes: options.cache_ram_bytes,
        },
    )?;
    let backend =
        ScheduledBackend::new(Arc::new(pool), options.prefill, options.scheduler.clone())?;
    backend.set_model_id(&options.served_model_name);
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

/// Dense Qwen (`qwen35`, with the 27B's vision sidecar), autoregressive or
/// with DFlash2, Flash-Next and DeepSeek V4 Flash run on the Rust runners;
/// every other text model on the C++ runner.
fn start_text(
    options: &LlmOptions,
    session_count: usize,
) -> Result<Arc<dyn TextGenerationBackend>> {
    let model = Path::new(&options.model);
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
    // Dense Qwen with the C++ runner's refusals (`CreateRunner`); its loader
    // reports any other artifact.
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
                "MTP HTTP decoding requires a Qwen Flash-Next model",
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

/// Qwen3.8-Flash-Next text serving on the Rust runner, with the C++
/// runner's checks (`CreateTextModelRunner`).
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

/// DeepSeek V4 Flash text serving on the Rust runner, autoregressive or with
/// DSpark, with the C++ runner's checks (`InferenceBackend::load`); the
/// draft limits are checked before the model loads.
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
    load_log.complete(
        &format!(
            "model={} sessions={session_count} context_tokens={} speculative={} \
             draft_limit={} disk_cache={}",
            backend.model_id(),
            options.max_context,
            options.speculative.name(),
            options.max_draft_tokens,
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
