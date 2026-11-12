//! Sampling controls in request bodies (`src/cli/serve/sampling_request.hpp`).

use gufo_core::json::Value;
use gufo_core::sampling::SamplingConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplingRequestError {
    pub message: String,
    pub code: String,
}

const MAX_EXACT_JSON_INTEGER: i64 = 9_007_199_254_740_991;
const MAX_SAMPLING_WINDOW: usize = u32::MAX as usize;

fn error_code(field: &str) -> String {
    format!("invalid_{}", field.replace('-', "_"))
}

/// Reads a finite float field; absent (or null when `nullable`) leaves
/// `output` unchanged.
fn read_float(
    body: &Value,
    field: &str,
    output: &mut f32,
    nullable: bool,
) -> Result<(), SamplingRequestError> {
    let Some(value) = body.find(field).filter(|v| !(nullable && v.is_null())) else {
        return Ok(());
    };
    let number = value.as_double(0.0);
    if !value.is_number()
        || !number.is_finite()
        || number < -f64::from(f32::MAX)
        || number > f64::from(f32::MAX)
    {
        return Err(SamplingRequestError {
            message: format!("'{field}' must be a finite number"),
            code: error_code(field),
        });
    }
    *output = number as f32;
    Ok(())
}

/// Reads an integer field in `[minimum, maximum]` (both exact in `f64` for
/// the fields used here), leaving `output` unchanged when absent.
pub fn read_integer<T>(
    body: &Value,
    field: &str,
    minimum: i64,
    maximum: u64,
    output: &mut T,
) -> Result<(), SamplingRequestError>
where
    T: TryFrom<i64>,
{
    read_integer_or_null(body, field, minimum, maximum, output, false)
}

/// [`read_integer`] that also leaves `output` unchanged for null when
/// `nullable`.
pub(crate) fn read_integer_or_null<T>(
    body: &Value,
    field: &str,
    minimum: i64,
    maximum: u64,
    output: &mut T,
    nullable: bool,
) -> Result<(), SamplingRequestError>
where
    T: TryFrom<i64>,
{
    let Some(value) = body.find(field).filter(|v| !(nullable && v.is_null())) else {
        return Ok(());
    };
    let number = if value.is_number() {
        value.as_double(0.0)
    } else {
        0.0
    };
    let invalid = || SamplingRequestError {
        message: format!("'{field}' must be an integer between {minimum} and {maximum}"),
        code: error_code(field),
    };
    if !value.is_number()
        || !number.is_finite()
        || number.floor() != number
        || number < minimum as f64
        || number > maximum as f64
    {
        return Err(invalid());
    }
    *output = T::try_from(number as i64).map_err(|_| invalid())?;
    Ok(())
}

/// The llama.cpp sampler fields gufo does not implement
/// (`kLlamaSamplerFields`).
const LLAMA_SAMPLER_FIELDS: [&str; 15] = [
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "typical_p",
    "tfs_z",
    "dynatemp_range",
    "dynatemp_exponent",
    "top_n_sigma",
    "xtc_probability",
    "xtc_threshold",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
    "dry_sequence_breakers",
];

fn invalid_sampling(field: &str, what: &str) -> SamplingRequestError {
    SamplingRequestError {
        message: format!("'{field}' must be {what}"),
        code: "invalid_sampling".into(),
    }
}

/// A present, non-null number; null counts as absent. The first malformed
/// field is kept in `error`.
fn sampler_number(
    body: &Value,
    field: &str,
    error: &mut Option<SamplingRequestError>,
) -> Option<f64> {
    let value = body.find(field).filter(|value| !value.is_null())?;
    if !value.is_number() || !value.as_double(0.0).is_finite() {
        error.get_or_insert_with(|| invalid_sampling(field, "a finite number"));
        return None;
    }
    Some(value.as_double(0.0))
}

/// [`sampler_number`] that must also be a window length in
/// `[0, 2147483647]`.
fn sampler_window(
    body: &Value,
    field: &str,
    error: &mut Option<SamplingRequestError>,
) -> Option<f64> {
    let value = sampler_number(body, field, error);
    if let Some(value) = value
        && (value.floor() != value || value < 0.0 || value > f64::from(i32::MAX))
    {
        error.get_or_insert_with(|| invalid_sampling(field, "an integer between 0 and 2147483647"));
    }
    value
}

/// llama.cpp sampler fields that gufo does not implement
/// (`CheckNeutralLlamaSamplers`). A field that is absent, null, or set to a
/// value that leaves llama.cpp's sampler chain unchanged is ignored, so
/// clients that always send their sampler settings (Open WebUI 0.12 sends
/// mirostat 0) still work. A value that would activate one of these
/// samplers is rejected rather than sampling another distribution. Neutral
/// values follow llama.cpp: the server's clamps and each sampler's early
/// return.
fn check_neutral_llama_samplers(body: &Value) -> Result<(), SamplingRequestError> {
    let unsupported = |field: &str| SamplingRequestError {
        message: format!("request field '{field}' is not supported with a value that enables it"),
        code: "unsupported_sampling".into(),
    };
    let mut error = None;
    let mirostat = sampler_number(body, "mirostat", &mut error);
    sampler_number(body, "mirostat_tau", &mut error);
    sampler_number(body, "mirostat_eta", &mut error);
    let typical_p = sampler_number(body, "typical_p", &mut error);
    sampler_number(body, "tfs_z", &mut error);
    let dynatemp_range = sampler_number(body, "dynatemp_range", &mut error);
    sampler_number(body, "dynatemp_exponent", &mut error);
    let top_n_sigma = sampler_number(body, "top_n_sigma", &mut error);
    let xtc_probability = sampler_number(body, "xtc_probability", &mut error);
    let xtc_threshold = sampler_number(body, "xtc_threshold", &mut error);
    let dry_multiplier = sampler_number(body, "dry_multiplier", &mut error);
    sampler_number(body, "dry_base", &mut error);
    sampler_window(body, "dry_allowed_length", &mut error);
    let dry_penalty_last_n = sampler_window(body, "dry_penalty_last_n", &mut error);
    if error.is_none()
        && let Some(breakers) = body
            .find("dry_sequence_breakers")
            .filter(|breakers| !breakers.is_null())
        && !(breakers.is_array()
            && !breakers.is_empty()
            && breakers.items().iter().all(Value::is_string))
    {
        error = Some(invalid_sampling(
            "dry_sequence_breakers",
            "a nonempty array of strings",
        ));
    }
    if let Some(error) = error {
        return Err(error);
    }
    // mirostat_tau/eta only matter with mirostat 1 or 2, dynatemp_exponent
    // only with a positive dynatemp_range, and llama.cpp no longer has TFS.
    if mirostat.is_some_and(|value| value > 0.0) {
        return Err(unsupported("mirostat"));
    }
    if typical_p.is_some_and(|value| value < 1.0) {
        return Err(unsupported("typical_p"));
    }
    if dynatemp_range.is_some_and(|value| value > 0.0) {
        return Err(unsupported("dynatemp_range"));
    }
    if top_n_sigma.is_some_and(|value| value > 0.0) {
        return Err(unsupported("top_n_sigma"));
    }
    // XTC acts only when the clamped probability is positive and the clamped
    // threshold is at most 0.5 (llama.cpp defaults: 0.0 and 0.1).
    if xtc_probability.unwrap_or(0.0).clamp(0.0, 1.0) > 0.0
        && xtc_threshold.unwrap_or(0.1).clamp(0.0, 1.0) <= 0.5
    {
        return Err(unsupported(if xtc_probability.is_some() {
            "xtc_probability"
        } else {
            "xtc_threshold"
        }));
    }
    // DRY acts only with a nonzero multiplier and a nonzero window
    // (common.h defaults 0.0 and 64; the server limits the window to
    // [0, INT32_MAX]). The sampler skips dry_base < 1, but the server
    // replaces such a base with its default first, so dry_base never
    // disables DRY.
    if dry_multiplier.unwrap_or(0.0) != 0.0 && dry_penalty_last_n.unwrap_or(64.0) != 0.0 {
        return Err(unsupported("dry_multiplier"));
    }
    Ok(())
}

/// All shared sampling controls, `defaults` for omitted fields.
pub fn parse_sampling_config(
    body: &Value,
    defaults: &SamplingConfig,
) -> Result<SamplingConfig, SamplingRequestError> {
    let mut output = defaults.clone();
    // These controls alter proposal or target probabilities: reject
    // unsupported spellings instead of sampling a different distribution.
    for (field, value) in body.members() {
        // OpenAI clients send an empty bias by default.
        if field == "logit_bias" && (value.is_null() || (value.is_object() && value.is_empty())) {
            continue;
        }
        let draft_control = field.starts_with("draft_")
            || field.ends_with("_draft")
            || field == "draft"
            || field == "speculative";
        if draft_control
            || matches!(field.as_str(), "samplers" | "logit_bias")
            || ((field.starts_with("xtc_") || field.starts_with("dry_"))
                && !LLAMA_SAMPLER_FIELDS.contains(&field.as_str()))
        {
            return Err(SamplingRequestError {
                message: format!("request field '{field}' is not supported"),
                code: "unsupported_sampling".into(),
            });
        }
    }
    check_neutral_llama_samplers(body)?;
    // The public API's ranges, checked on the JSON number before it narrows
    // to `f32` (1.00000001 must not round into top-p's range); native
    // sampler controls may be broader. llama-server clamps instead.
    for (field, minimum, maximum) in [
        ("temperature", 0.0, 2.0),
        ("top_p", 0.0, 1.0),
        ("min_p", 0.0, 1.0),
        ("frequency_penalty", -2.0, 2.0),
        ("presence_penalty", -2.0, 2.0),
    ] {
        if let Some(value) = body.find(field).filter(|v| v.is_number())
            && !(minimum..=maximum).contains(&value.as_double(0.0))
        {
            return Err(SamplingRequestError {
                message: format!("'{field}' must be between {minimum} and {maximum}"),
                code: error_code(field),
            });
        }
    }
    // OpenAI's nullable fields keep the server defaults when null.
    read_float(body, "temperature", &mut output.temperature, true)?;
    read_integer(body, "top_k", 0, i32::MAX as u64, &mut output.top_k)?;
    read_float(body, "top_p", &mut output.top_p, true)?;
    read_float(body, "min_p", &mut output.min_p, false)?;
    read_integer(
        body,
        "min_keep",
        0,
        MAX_SAMPLING_WINDOW as u64,
        &mut output.min_keep,
    )?;
    read_integer_or_null(
        body,
        "seed",
        -1,
        MAX_EXACT_JSON_INTEGER as u64,
        &mut output.seed,
        true,
    )?;
    read_float(body, "repeat_penalty", &mut output.repeat_penalty, false)?;
    read_integer(
        body,
        "repeat_last_n",
        0,
        MAX_SAMPLING_WINDOW as u64,
        &mut output.repeat_last_n,
    )?;
    read_float(
        body,
        "frequency_penalty",
        &mut output.frequency_penalty,
        true,
    )?;
    read_float(body, "presence_penalty", &mut output.presence_penalty, true)?;
    output.validate().map_err(|error| SamplingRequestError {
        message: error.message().to_owned(),
        code: "invalid_sampling".into(),
    })?;
    // The public ranges hold for the resolved values too, so server
    // defaults outside them are refused like request fields.
    if output.temperature > 2.0 {
        return Err(SamplingRequestError {
            message: "'temperature' must be between 0 and 2".into(),
            code: error_code("temperature"),
        });
    }
    for (field, value) in [
        ("frequency_penalty", output.frequency_penalty),
        ("presence_penalty", output.presence_penalty),
    ] {
        if !(-2.0..=2.0).contains(&value) {
            return Err(SamplingRequestError {
                message: format!("'{field}' must be between -2 and 2"),
                code: error_code(field),
            });
        }
    }
    Ok(output)
}
