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
fn read_integer_or_null<T>(
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
            || matches!(
                field.as_str(),
                "samplers"
                    | "typical_p"
                    | "tfs_z"
                    | "mirostat"
                    | "mirostat_eta"
                    | "mirostat_tau"
                    | "dynatemp_range"
                    | "dynatemp_exponent"
                    | "top_n_sigma"
                    | "logit_bias"
            )
            || field.starts_with("xtc_")
            || field.starts_with("dry_")
        {
            return Err(SamplingRequestError {
                message: format!("request field '{field}' is not supported"),
                code: "unsupported_sampling".into(),
            });
        }
    }
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
