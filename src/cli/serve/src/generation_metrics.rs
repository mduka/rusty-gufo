//! Generation metrics for responses and logs
//! (`src/cli/serve/generation_metrics.hpp`).

use std::fmt::Write as _;

use gufo_core::json::Value;

use crate::backend::{FinishReason, GenerationResult, PromptProgress};

pub fn prefill_tokens_per_second(result: &GenerationResult) -> f64 {
    if result.prefill_ms > 0.0 {
        result.prefill_tokens as f64 * 1000.0 / result.prefill_ms
    } else {
        0.0
    }
}

fn decode_tokens_per_second(result: &GenerationResult) -> f64 {
    if result.decode_ms > 0.0 {
        1000.0 * result.completion_tokens as f64 / result.decode_ms
    } else {
        0.0
    }
}

/// The request-log summary, with one decimal for every float.
pub fn generation_log_details(result: &GenerationResult) -> String {
    let finish = if result.cancelled {
        "cancelled"
    } else if result.finish_reason == FinishReason::Length {
        "length"
    } else if result.finish_reason == FinishReason::StopSequence {
        "stop_sequence"
    } else {
        "stop"
    };
    let cache = if result.cache_disk_hit {
        "disk"
    } else if result.cache_hit {
        "memory"
    } else {
        "miss"
    };
    let mut out = format!(
        "prompt_tokens={} prefill_tokens={} generated_tokens={} finish={finish} cache={cache} \
         cached_tokens={} cache_restore_ms={:.1} queue_depth={} resident_at_admission={} \
         queue_ms={:.1} shared_prefix_wait_ms={:.1} ttft_ms={:.1} prefill_tps={:.1} \
         decode_tps={:.1} batch_width={} plan={} \
         draft_accepted={} draft_proposed={} draft_rounds={}",
        result.prompt_tokens,
        result.prefill_tokens,
        result.completion_tokens,
        result.cached_prompt_tokens,
        result.cache_restore_ms,
        result.queue_depth_at_submit,
        result.resident_requests_at_admission,
        result.queue_ms,
        result.shared_prefix_wait_ms,
        result.ttft_ms,
        prefill_tokens_per_second(result),
        decode_tokens_per_second(result),
        result.physical_execution_width,
        result.execution_plan,
        result.draft_accepted_tokens,
        result.draft_tokens,
        result.draft_rounds,
    );
    if result.draft_tokens > 0 {
        let _ = write!(
            out,
            " acceptance_pct={:.1}",
            100.0 * result.draft_accepted_tokens as f64 / result.draft_tokens as f64
        );
    }
    if result.lookup_tokens > 0 {
        let _ = write!(
            out,
            " lookup_accepted={} lookup_proposed={}",
            result.lookup_accepted_tokens, result.lookup_tokens
        );
    }
    if result.cache_snapshot_bytes > 0 {
        let _ = write!(out, " cache_snapshot_bytes={}", result.cache_snapshot_bytes);
    }
    if result.cache_disk_queued_bytes > 0 {
        let _ = write!(
            out,
            " cache_disk_queued_bytes={}",
            result.cache_disk_queued_bytes
        );
    }
    if !result.cache_hit && !result.cache_miss_reason.is_empty() {
        let _ = write!(
            out,
            " cache_miss_reason={} common_prefix_tokens={} nearest_checkpoint_tokens={}",
            result.cache_miss_reason,
            result.cache_common_prefix_tokens,
            result.cache_checkpoint_tokens
        );
    }
    out
}

/// llama.cpp-style timings; timed prefill counts the work actually executed.
pub fn generation_timings(result: &GenerationResult) -> Value {
    let per_token = |ms: f64, n: usize| if n > 0 { ms / n as f64 } else { 0.0 };
    let mut timings = Value::object()
        .with("prompt_n", result.prefill_tokens)
        .with("prompt_ms", result.prefill_ms)
        .with(
            "prompt_per_token_ms",
            per_token(result.prefill_ms, result.prefill_tokens),
        )
        .with("prompt_per_second", prefill_tokens_per_second(result))
        .with("predicted_n", result.completion_tokens)
        .with("predicted_ms", result.decode_ms)
        .with(
            "predicted_per_token_ms",
            per_token(result.decode_ms, result.completion_tokens),
        )
        .with(
            "predicted_per_second",
            if result.decode_ms > 0.0 {
                result.completion_tokens as f64 * 1000.0 / result.decode_ms
            } else {
                0.0
            },
        )
        .with("cache_n", result.cached_prompt_tokens)
        .with("cache_restore_ms", result.cache_restore_ms)
        .with("cache_snapshot_ms", result.cache_snapshot_ms)
        .with("cache_disk_enqueue_ms", result.cache_disk_enqueue_ms)
        .with("draft_rounds", result.draft_rounds)
        .with("draft_n", result.draft_tokens)
        .with("draft_n_accepted", result.draft_accepted_tokens);
    if result.lookup_tokens > 0 {
        timings = timings
            .with("lookup_n", result.lookup_tokens)
            .with("lookup_n_accepted", result.lookup_accepted_tokens);
    }
    timings
}

/// The llama-server `prompt_progress` object of a streamed
/// `return_progress` request.
pub fn prompt_progress_json(progress: &PromptProgress) -> Value {
    Value::object()
        .with("total", progress.total)
        .with("cache", progress.cache)
        .with("processed", progress.processed)
        .with("time_ms", progress.time_ms)
}

/// The `Server-Timing` header value, three decimals.
pub fn server_timing(result: &GenerationResult) -> String {
    format!(
        "ttft;dur={:.3}, inter_token;dur={:.3}, max_inter_token;dur={:.3}",
        result.ttft_ms, result.mean_inter_token_ms, result.max_inter_token_ms
    )
}
