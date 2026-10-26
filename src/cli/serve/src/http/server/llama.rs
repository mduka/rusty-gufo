//! The llama.cpp routes (`/completion`, `/props`, `/slots`, `/metrics`) and
//! the model list.

use super::*;

pub(super) fn llama_completion(
    request: &HttpRequest,
    backend: &Arc<dyn TextGenerationBackend>,
) -> Result<HttpResponse> {
    compatibility_errors((|| {
        let body = try_response!(parse_body(request));
        let (max_tokens, sampling) = try_response!(read_compatibility_options(
            &body,
            backend,
            "n_predict",
            &["stop"],
            backend.reasoning_defaults().enabled,
        ));
        let prompt = match body.find("prompt") {
            Some(Value::String(prompt)) if !prompt.is_empty() => prompt,
            _ => {
                return Ok(invalid_compatibility_request(
                    "'prompt' must be a nonempty string",
                ));
            }
        };
        let stop_sequences =
            match parse_stop_sequences(body.find("stop"), StopSequenceFormat::OpenAi) {
                Ok(sequences) => sequences,
                Err(message) => return Ok(invalid_compatibility_request(&message)),
            };
        let cache_prompt = try_response!(read_cache_prompt(&body));
        let result = backend.complete(
            prompt,
            max_tokens,
            &sampling,
            request.is_cancelled.as_ref(),
            None,
            &request.client_id,
            &stop_sequences,
            cache_prompt,
        )?;
        let limited = result.finish_reason == FinishReason::Length;
        let matched = result.finish_reason == FinishReason::StopSequence;
        let response = Value::object()
            .with("content", text_of(&result))
            .with("stop", true)
            .with("stopped_eos", !limited && !result.cancelled && !matched)
            .with("stopped_length", limited)
            .with("stopped_word", matched)
            .with("stopped_limit", limited)
            .with("stopping_word", result.stop_sequence.as_str())
            .with("tokens_predicted", result.completion_tokens)
            .with("tokens_evaluated", result.prompt_tokens)
            .with("tokens_cached", result.cached_prompt_tokens)
            .with("timings", generation_timings(&result))
            .with("usage", usage_json(&result));
        Ok(with_timing(ok(&response), &result))
    })())
}

/// llama-server props for a single-model server: what it runs, so a client
/// can record which model, build and settings answered it. `?model=` is accepted and not required, as in llama-server
/// outside router mode. Fields the backend does not report are omitted rather
/// than sent as zero.
pub(super) fn llama_props(
    request: &HttpRequest,
    backend: &Arc<dyn TextGenerationBackend>,
) -> Result<HttpResponse> {
    let requested = request.query_param("model");
    let alias = backend.model_id();
    let mut response = Value::object()
        .with(
            "model",
            if requested.is_empty() {
                alias.clone()
            } else {
                requested
            },
        )
        .with("model_alias", alias)
        .with("template", "")
        .with("model_info", Value::object());
    let path = backend.model_path();
    if !path.is_empty() {
        response = response.with("model_path", path);
    }
    let build = backend.build_info();
    if !build.is_empty() {
        response = response.with("build_info", build);
    }
    let sessions = backend.session_states().len();
    if sessions > 0 {
        response = response.with("total_slots", sessions);
    }
    // What a request without sampling fields gets.
    let defaults = backend.sampling_defaults();
    let sampling = defaults.resolve(backend.reasoning_defaults().enabled);
    // -1 means no default limit, as in llama-server, which reports both keys.
    let n_predict = if defaults.max_tokens > 0 {
        defaults.max_tokens as i64
    } else {
        -1
    };
    let params = Value::object()
        .with("temperature", f64::from(sampling.temperature))
        .with("top_k", sampling.top_k)
        .with("top_p", f64::from(sampling.top_p))
        .with("min_p", f64::from(sampling.min_p))
        .with("seed", sampling.seed)
        .with("repeat_penalty", f64::from(sampling.repeat_penalty))
        .with("repeat_last_n", sampling.repeat_last_n)
        .with("frequency_penalty", f64::from(sampling.frequency_penalty))
        .with("presence_penalty", f64::from(sampling.presence_penalty))
        .with("max_tokens", n_predict)
        .with("n_predict", n_predict);
    let mut generation = Value::object();
    let context = backend.max_context();
    if context > 0 {
        generation = generation.with("n_ctx", context);
    }
    Ok(ok(&response
        .with(
            "default_generation_settings",
            generation.with("params", params),
        )
        .with(
            "modalities",
            Value::object()
                .with("vision", backend.supports_images())
                .with("video", false)
                .with("audio", false),
        )))
}

/// llama-server slots, one per execution session. `task_id`, `state` and an
/// always-empty `prompt` remain for older clients; prompt text is not exposed.
pub(super) fn llama_slots(
    _: &HttpRequest,
    backend: &Arc<dyn TextGenerationBackend>,
) -> Result<HttpResponse> {
    let mut sessions = backend.session_states();
    if sessions.is_empty() {
        sessions.push(SessionState::default());
    }
    let context = backend.max_context();
    let model = backend.model_id();
    let slots = sessions
        .iter()
        .enumerate()
        .map(|(index, session)| {
            let task = if session.processing {
                session.request_id as i64
            } else {
                -1
            };
            let next_token = Value::object()
                .with("has_next_token", session.processing)
                .with("has_new_line", false)
                .with(
                    "n_remain",
                    if session.processing {
                        session.remaining_tokens as i64
                    } else {
                        -1
                    },
                )
                .with("n_decoded", session.generated_tokens);
            Value::object()
                .with("id", index)
                .with("n_ctx", context)
                .with("speculative", session.speculative)
                .with("is_processing", session.processing)
                .with("id_task", task)
                .with("task_id", task)
                .with("state", i64::from(session.processing))
                .with("n_prompt_tokens", session.prompt_tokens)
                .with("n_prompt_tokens_cache", session.cached_prompt_tokens)
                .with("n_prompt_tokens_processed", session.processed_prompt_tokens)
                .with("prompt", "")
                .with("next_token", Value::from(vec![next_token]))
                .with("model", model.as_str())
        })
        .collect::<Vec<_>>();
    Ok(ok(&Value::from(slots)))
}

/// Context held by in-flight requests across all sessions.
pub(super) fn kv_cache_usage_ratio(backend: &dyn TextGenerationBackend) -> f64 {
    let sessions = backend.session_states();
    let capacity = sessions.len() * backend.max_context() as usize;
    if capacity == 0 {
        return 0.0;
    }
    let used: usize = sessions
        .iter()
        .map(|session| session.prompt_tokens + session.generated_tokens)
        .sum();
    used as f64 / capacity as f64
}

pub(super) fn llama_metrics(
    _: &HttpRequest,
    backend: &Arc<dyn TextGenerationBackend>,
) -> Result<HttpResponse> {
    Ok(HttpResponse::new(
        200,
        "OK",
        prometheus_text(kv_cache_usage_ratio(backend.as_ref())),
    )
    .with_header("Content-Type", "text/plain; version=0.0.4"))
}

pub(super) fn model_entry(id: &str, owned_by: &str) -> Value {
    Value::object()
        .with("id", id)
        .with("object", "model")
        .with("created", unix_time())
        .with("owned_by", owned_by)
}

pub(super) fn list_models(
    backend: Option<&Arc<dyn TextGenerationBackend>>,
    services: &HttpServices,
) -> HttpResponse {
    let mut data = Value::array();
    if let Some(backend) = backend {
        let mut entry = model_entry(&backend.model_id(), "gufo");
        if backend.max_context() > 0 {
            entry.set("context_length", backend.max_context());
        }
        // Image input follows the loaded vision encoder, not the model name.
        let mut modalities = vec![Value::from("text")];
        if backend.supports_images() {
            modalities.push(Value::from("image"));
        }
        entry.set(
            "architecture",
            Value::object().with("input_modalities", Value::from(modalities)),
        );
        data.push(entry);
    }
    if let Some(video) = services.video_jobs.as_ref().filter(|v| v.ready()) {
        let _ = video;
        data.push(
            model_entry("minimax-h3", "operator-supplied-minimax").with("capability", "video"),
        );
        for preset in [
            "minimax-h3-exact",
            "minimax-h3-fast",
            "minimax-h3-aggressive",
            "minimax-h3-dev",
            "minimax-h3-fullres",
        ] {
            data.push(
                model_entry(preset, "operator-supplied-minimax")
                    .with("root", "minimax-h3")
                    .with("capability", "video"),
            );
        }
    }
    if let Some(tts) = services.tts.as_ref().filter(|t| t.ready()) {
        data.push(
            model_entry(&tts.model_id(), "operator-supplied-qwen").with("capability", "audio_tts"),
        );
    }
    if let Some(asr) = services.asr.as_ref().filter(|a| a.ready()) {
        data.push(
            model_entry(&asr.model_id(), "operator-supplied-qwen").with("capability", "audio_asr"),
        );
    }
    if let Some(images) = &services.images {
        data.push(
            model_entry(images.model_id(), "operator-supplied-qwen").with("capability", "image"),
        );
    }
    if let Some(embeddings) = services.embeddings.as_ref().filter(|e| e.ready()) {
        data.push(
            model_entry(&embeddings.model_id(), "operator-supplied-gemma")
                .with("capability", "embedding"),
        );
    }
    ok(&Value::object().with("object", "list").with("data", data))
}
