//! Process-wide generation counters behind `/metrics`
//! (`src/cli/serve/generation_metrics.hpp`), and the `std::ostream` number
//! formats the server's text uses.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use crate::backend::GenerationResult;
use crate::generation_metrics::prefill_tokens_per_second;

// Token counters advance per prefill chunk and per generated token, so
// scrapes see work while requests are still running.
static TOTAL_PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
static TOTAL_GENERATED_TOKENS: AtomicU64 = AtomicU64::new(0);
// Request totals, added once per request when it finishes; seconds and
// speeds are `f64` bits.
static TOTAL_CACHED_PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
static TOTAL_PROMPT_SECONDS: AtomicU64 = AtomicU64::new(0);
static TOTAL_GENERATION_SECONDS: AtomicU64 = AtomicU64::new(0);
static MAX_SEQUENCE_TOKENS: AtomicU64 = AtomicU64::new(0);
static TOTAL_DRAFT_ROUNDS: AtomicU64 = AtomicU64::new(0);
static TOTAL_DRAFT_TOKENS: AtomicU64 = AtomicU64::new(0);
static TOTAL_DRAFT_ACCEPTED_TOKENS: AtomicU64 = AtomicU64::new(0);
static TOTAL_LOOKUP_TOKENS: AtomicU64 = AtomicU64::new(0);
static TOTAL_LOOKUP_ACCEPTED_TOKENS: AtomicU64 = AtomicU64::new(0);
static LAST_PROMPT_SPEED: AtomicU64 = AtomicU64::new(0);
static LAST_GENERATION_SPEED: AtomicU64 = AtomicU64::new(0);
// Schedulers add their own deltas, so a draining scheduler and its
// replacement can both contribute during a model reload.
static REQUESTS_PROCESSING: AtomicI64 = AtomicI64::new(0);
static REQUESTS_DEFERRED: AtomicI64 = AtomicI64::new(0);
static DEVICE_LOST_TOTAL: AtomicU64 = AtomicU64::new(0);

/// A snapshot of the counters, for tests and diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MetricsSnapshot {
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub cached_prompt_tokens: u64,
    pub prompt_seconds: f64,
    pub generation_seconds: f64,
    pub max_sequence_tokens: u64,
    pub draft_rounds: u64,
    pub draft_tokens: u64,
    pub draft_accepted_tokens: u64,
    pub requests_processing: i64,
    pub requests_deferred: i64,
    pub device_lost_total: u64,
}

pub fn snapshot() -> MetricsSnapshot {
    let load = |value: &AtomicU64| value.load(Ordering::Relaxed);
    MetricsSnapshot {
        prompt_tokens: load(&TOTAL_PROMPT_TOKENS),
        generated_tokens: load(&TOTAL_GENERATED_TOKENS),
        cached_prompt_tokens: load(&TOTAL_CACHED_PROMPT_TOKENS),
        prompt_seconds: f64::from_bits(load(&TOTAL_PROMPT_SECONDS)),
        generation_seconds: f64::from_bits(load(&TOTAL_GENERATION_SECONDS)),
        max_sequence_tokens: load(&MAX_SEQUENCE_TOKENS),
        draft_rounds: load(&TOTAL_DRAFT_ROUNDS),
        draft_tokens: load(&TOTAL_DRAFT_TOKENS),
        draft_accepted_tokens: load(&TOTAL_DRAFT_ACCEPTED_TOKENS),
        requests_processing: REQUESTS_PROCESSING.load(Ordering::Relaxed),
        requests_deferred: REQUESTS_DEFERRED.load(Ordering::Relaxed),
        device_lost_total: load(&DEVICE_LOST_TOTAL),
    }
}

/// Uncached prompt tokens as a prefill chunk executes them.
pub fn add_prompt_tokens(count: usize) {
    TOTAL_PROMPT_TOKENS.fetch_add(count as u64, Ordering::Relaxed);
}

/// Tokens as they are generated.
pub fn add_generated_tokens(count: usize) {
    TOTAL_GENERATED_TOKENS.fetch_add(count as u64, Ordering::Relaxed);
}

/// Admitted requests, including cache preparation and cleanup.
pub fn add_requests_processing(delta: i64) {
    REQUESTS_PROCESSING.fetch_add(delta, Ordering::Relaxed);
}

/// Requests waiting for a session.
pub fn add_requests_deferred(delta: i64) {
    REQUESTS_DEFERRED.fetch_add(delta, Ordering::Relaxed);
}

/// A confirmed GPU context loss, once per scheduler.
pub fn add_device_lost() {
    DEVICE_LOST_TOTAL.fetch_add(1, Ordering::Relaxed);
}

fn add_seconds(total: &AtomicU64, ms: f64) {
    let _ = total.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
        Some((f64::from_bits(bits) + ms / 1000.0).to_bits())
    });
}

/// Adds one finished request to the request totals and speed gauges. The
/// scheduler calls this when a request ends, whether or not its client reads
/// the result; failed requests are not added.
pub fn record_request_metrics(result: &GenerationResult) {
    TOTAL_CACHED_PROMPT_TOKENS.fetch_add(
        result.cached_prompt_tokens.min(result.prompt_tokens) as u64,
        Ordering::Relaxed,
    );
    add_seconds(&TOTAL_PROMPT_SECONDS, result.prefill_ms);
    add_seconds(&TOTAL_GENERATION_SECONDS, result.decode_ms);
    MAX_SEQUENCE_TOKENS.fetch_max(
        (result.prompt_tokens + result.completion_tokens) as u64,
        Ordering::Relaxed,
    );
    TOTAL_DRAFT_ROUNDS.fetch_add(result.draft_rounds as u64, Ordering::Relaxed);
    TOTAL_DRAFT_TOKENS.fetch_add(result.draft_tokens as u64, Ordering::Relaxed);
    TOTAL_DRAFT_ACCEPTED_TOKENS.fetch_add(result.draft_accepted_tokens as u64, Ordering::Relaxed);
    TOTAL_LOOKUP_TOKENS.fetch_add(result.lookup_tokens as u64, Ordering::Relaxed);
    TOTAL_LOOKUP_ACCEPTED_TOKENS.fetch_add(result.lookup_accepted_tokens as u64, Ordering::Relaxed);
    let prompt_per_second = prefill_tokens_per_second(result);
    let generated_per_second = if result.decode_ms > 0.0 && result.completion_tokens > 0 {
        result.completion_tokens as f64 / (result.decode_ms / 1000.0)
    } else {
        0.0
    };
    if prompt_per_second > 0.0 {
        LAST_PROMPT_SPEED.store(prompt_per_second.to_bits(), Ordering::Relaxed);
    }
    if generated_per_second > 0.0 {
        LAST_GENERATION_SPEED.store(generated_per_second.to_bits(), Ordering::Relaxed);
    }
}

/// Counts a completed request of a backend without a scheduler; scheduled
/// requests were recorded by the scheduler already.
pub fn record_server_metrics(result: &GenerationResult) {
    if result.token_metrics_recorded {
        return;
    }
    // Such backends may only report prompt and cache totals. Infer their
    // completed uncached prompt, but never work for a partially cancelled
    // request.
    let prompt_tokens = if result.prefill_tokens > 0 || result.cancelled {
        result.prefill_tokens
    } else {
        result.prompt_tokens - result.cached_prompt_tokens.min(result.prompt_tokens)
    };
    add_prompt_tokens(prompt_tokens);
    add_generated_tokens(result.completion_tokens);
    record_request_metrics(result);
}

/// The Prometheus text of `GET /metrics`; `kv_cache_usage_ratio` is the
/// in-flight context over sessions times context.
pub fn prometheus_text(kv_cache_usage_ratio: f64) -> String {
    let count = |value: &AtomicU64| value.load(Ordering::Relaxed);
    let real = |value: f64| format_significant(value, 17);
    let bits = |value: &AtomicU64| real(f64::from_bits(count(value)));
    let mut out = format!(
        "# HELP gufo_device_lost_total Confirmed GPU context losses\n\
         # TYPE gufo_device_lost_total counter\n\
         gufo_device_lost_total {}\n",
        count(&DEVICE_LOST_TOTAL)
    );
    let metric = |out: &mut String, name: &str, kind: &str, help: &str, value: String| {
        out.push_str(&format!(
            "# HELP llamacpp:{name} {help}\n# TYPE llamacpp:{name} {kind}\nllamacpp:{name} {value}\n"
        ));
    };
    metric(
        &mut out,
        "prompt_tokens_total",
        "counter",
        "Total prompt tokens processed, excluding cache hits",
        count(&TOTAL_PROMPT_TOKENS).to_string(),
    );
    metric(
        &mut out,
        "prompt_tokens_cached_total",
        "counter",
        "Total prompt tokens reused from cache",
        count(&TOTAL_CACHED_PROMPT_TOKENS).to_string(),
    );
    metric(
        &mut out,
        "prompt_seconds_total",
        "counter",
        "Prompt process time",
        bits(&TOTAL_PROMPT_SECONDS),
    );
    metric(
        &mut out,
        "tokens_predicted_total",
        "counter",
        "Total tokens generated",
        count(&TOTAL_GENERATED_TOKENS).to_string(),
    );
    metric(
        &mut out,
        "tokens_predicted_seconds_total",
        "counter",
        "Predict process time",
        bits(&TOTAL_GENERATION_SECONDS),
    );
    metric(
        &mut out,
        "n_tokens_max",
        "counter",
        "Largest observed n_tokens.",
        count(&MAX_SEQUENCE_TOKENS).to_string(),
    );
    metric(
        &mut out,
        "spec_decode_num_drafts_total",
        "counter",
        "Total speculative verification rounds",
        count(&TOTAL_DRAFT_ROUNDS).to_string(),
    );
    metric(
        &mut out,
        "spec_decode_num_draft_tokens_total",
        "counter",
        "Total draft tokens proposed",
        count(&TOTAL_DRAFT_TOKENS).to_string(),
    );
    metric(
        &mut out,
        "spec_decode_num_accepted_tokens_total",
        "counter",
        "Total draft tokens accepted",
        count(&TOTAL_DRAFT_ACCEPTED_TOKENS).to_string(),
    );
    // Gufo's own counters, without the llamacpp prefix.
    for (name, help, total) in [
        (
            "gufo_spec_lookup_tokens_total",
            "Total prompt-lookup copies proposed",
            &TOTAL_LOOKUP_TOKENS,
        ),
        (
            "gufo_spec_lookup_accepted_tokens_total",
            "Total prompt-lookup copies accepted",
            &TOTAL_LOOKUP_ACCEPTED_TOKENS,
        ),
    ] {
        out.push_str(&format!(
            "# HELP {name} {help}\n# TYPE {name} counter\n{name} {}\n",
            count(total)
        ));
    }
    metric(
        &mut out,
        "prompt_tokens_seconds",
        "gauge",
        "Prompt processing speed in tokens per second",
        bits(&LAST_PROMPT_SPEED),
    );
    metric(
        &mut out,
        "predicted_tokens_seconds",
        "gauge",
        "Generation speed in tokens per second",
        bits(&LAST_GENERATION_SPEED),
    );
    metric(
        &mut out,
        "requests_processing",
        "gauge",
        "Number of admitted requests, including cache preparation",
        REQUESTS_PROCESSING.load(Ordering::Relaxed).to_string(),
    );
    metric(
        &mut out,
        "requests_deferred",
        "gauge",
        "Number of requests waiting for a session",
        REQUESTS_DEFERRED.load(Ordering::Relaxed).to_string(),
    );
    metric(
        &mut out,
        "kv_cache_usage_ratio",
        "gauge",
        "In-flight prompt and generated tokens over sessions times context; excludes retained \
         cache",
        real(kv_cache_usage_ratio),
    );
    out
}

/// `std::ostream << double` with the default precision 6 (`%g`): six
/// significant digits, fixed or scientific by exponent, trailing zeros
/// removed.
pub fn format_general(value: f64) -> String {
    format_significant(value, 6)
}

/// `std::ostream << std::setprecision(precision) << double` (`%.{precision}g`).
pub fn format_significant(value: f64, precision: usize) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.into();
    }
    if !value.is_finite() {
        return if value.is_nan() {
            "nan".into()
        } else if value > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    let precision = precision.max(1);
    let scientific = format!("{value:.*e}", precision - 1);
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let trim = |s: &str| {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            s.to_owned()
        }
    };
    if (-4..precision as i32).contains(&exponent) {
        let decimals = (precision as i32 - 1 - exponent).max(0) as usize;
        trim(&format!("{value:.decimals$}"))
    } else {
        format!(
            "{}e{}{:02}",
            trim(mantissa),
            if exponent < 0 { '-' } else { '+' },
            exponent.unsigned_abs()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{format_general, format_significant};

    #[test]
    fn general_format_matches_ostream() {
        // Pairs from libstdc++ `std::ostringstream << double`.
        let cases = [
            (0.0, "0"),
            (500.0, "500"),
            (123.456_789, "123.457"),
            (1e-5, "1e-05"),
            (1_234_567.0, "1.23457e+06"),
            (0.0001, "0.0001"),
            (123_456.0, "123456"),
            (1e21, "1e+21"),
            (1.234_567_89, "1.23457"),
            (2.5e-7, "2.5e-07"),
            (999_999.5, "1e+06"),
            (99_999.95, "99999.9"),
            (0.1, "0.1"),
            (1e6, "1e+06"),
            (100.0, "100"),
        ];
        for (value, expected) in cases {
            assert_eq!(format_general(value), expected, "{value}");
        }
    }

    #[test]
    fn max_digits_format_matches_ostream() {
        // `std::setprecision(std::numeric_limits<double>::max_digits10)`.
        let cases = [
            (0.0, "0"),
            (0.1, "0.10000000000000001"),
            (13.0 / 131_072.0, "9.918212890625e-05"),
            (123.456, "123.456"),
            (1.0 / 3.0, "0.33333333333333331"),
            (2.5e20, "2.5e+20"),
        ];
        for (value, expected) in cases {
            assert_eq!(format_significant(value, 17), expected, "{value}");
        }
    }
}
