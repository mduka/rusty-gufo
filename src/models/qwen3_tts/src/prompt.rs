//! Talker prompt embeddings from projected host rows (`prompt.cpp`): the
//! non-ICL prompt of CustomVoice, VoiceDesign and speaker-only Base, and the
//! in-context voice-clone prompt of Base. Added rows round to BF16 as the
//! upstream BF16 additions do.

use crate::bf16::round_bf16;

/// `PromptOutput`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PromptOutput {
    pub embeddings: Vec<f32>,
    pub trailing_text: Vec<f32>,
    pub tts_pad: Vec<f32>,
    pub tokens: usize,
}

/// `NonIclPromptInput`.
pub struct NonIclPromptInput<'a> {
    pub hidden_size: usize,
    pub instruction: &'a [f32],
    pub role: &'a [f32],
    /// Codec prefix followed by codec_pad and codec_bos rows.
    pub codec: &'a [f32],
    pub text: &'a [f32],
    pub tts_bos: &'a [f32],
    pub tts_eos: &'a [f32],
    pub tts_pad: &'a [f32],
    pub codec_pad: &'a [f32],
}

/// `IclPromptInput`.
pub struct IclPromptInput<'a> {
    pub hidden_size: usize,
    pub role: &'a [f32],
    /// Codec prefix, speaker embedding, codec_pad and codec_bos rows.
    pub codec: &'a [f32],
    /// Projected reference text followed by projected target text.
    pub combined_text: &'a [f32],
    /// codec_bos followed by one summed 16-codebook embedding per frame.
    pub reference_codec: &'a [f32],
    pub tts_bos: &'a [f32],
    pub tts_eos: &'a [f32],
    pub tts_pad: &'a [f32],
}

fn require_rows(
    values: &[f32],
    hidden_size: usize,
    name: &str,
    allow_empty: bool,
) -> Result<(), String> {
    if hidden_size == 0
        || !values.len().is_multiple_of(hidden_size)
        || (!allow_empty && values.is_empty())
    {
        return Err(format!("Qwen3-TTS {name} embedding shape is invalid"));
    }
    Ok(())
}

fn append_added_row(destination: &mut Vec<f32>, left: &[f32], right: &[f32]) -> Result<(), String> {
    if left.len() != right.len() {
        return Err("Qwen3-TTS prompt row shape mismatch".into());
    }
    destination.extend(left.iter().zip(right).map(|(l, r)| round_bf16(l + r)));
    Ok(())
}

fn row(values: &[f32], row: usize, hidden_size: usize) -> &[f32] {
    &values[row * hidden_size..(row + 1) * hidden_size]
}

/// `BuildPrefix`: leading rows, the role, then the codec prefix under TTS
/// pad rows and the last-but-one codec row under TTS BOS.
fn build_prefix(
    leading_rows: &[f32],
    role: &[f32],
    codec: &[f32],
    tts_bos: &[f32],
    tts_pad: &[f32],
    hidden_size: usize,
    output: &mut PromptOutput,
) -> Result<(), String> {
    output.embeddings.extend_from_slice(leading_rows);
    output.embeddings.extend_from_slice(role);
    let codec_rows = codec.len() / hidden_size;
    if codec_rows < 2 {
        return Err("Qwen3-TTS codec prompt is too short".into());
    }
    for index in 0..codec_rows - 2 {
        append_added_row(
            &mut output.embeddings,
            tts_pad,
            row(codec, index, hidden_size),
        )?;
    }
    append_added_row(
        &mut output.embeddings,
        tts_bos,
        row(codec, codec_rows - 2, hidden_size),
    )
}

fn finish(output: &mut PromptOutput, hidden_size: usize) -> Result<(), String> {
    if output.embeddings.is_empty() || !output.embeddings.len().is_multiple_of(hidden_size) {
        return Err("Qwen3-TTS prompt embedding shape mismatch".into());
    }
    output.tokens = output.embeddings.len() / hidden_size;
    Ok(())
}

/// `BuildNonIclPrompt`.
pub fn build_non_icl_prompt(input: &NonIclPromptInput<'_>) -> Result<PromptOutput, String> {
    let hidden = input.hidden_size;
    require_rows(input.instruction, hidden, "instruction", true)?;
    require_rows(input.role, hidden, "role", false)?;
    require_rows(input.codec, hidden, "codec", false)?;
    require_rows(input.text, hidden, "text", true)?;
    require_rows(input.tts_bos, hidden, "TTS BOS", false)?;
    require_rows(input.tts_eos, hidden, "TTS EOS", false)?;
    require_rows(input.tts_pad, hidden, "TTS pad", false)?;
    require_rows(input.codec_pad, hidden, "codec pad", false)?;
    if input.tts_bos.len() != hidden
        || input.tts_eos.len() != hidden
        || input.tts_pad.len() != hidden
        || input.codec_pad.len() != hidden
    {
        return Err("Qwen3-TTS special embedding must contain one row".into());
    }
    let mut output = PromptOutput {
        tts_pad: input.tts_pad.to_vec(),
        trailing_text: input.tts_pad.to_vec(),
        ..PromptOutput::default()
    };
    build_prefix(
        input.instruction,
        input.role,
        input.codec,
        input.tts_bos,
        input.tts_pad,
        hidden,
        &mut output,
    )?;
    for index in 0..input.text.len() / hidden {
        append_added_row(
            &mut output.embeddings,
            row(input.text, index, hidden),
            input.codec_pad,
        )?;
    }
    append_added_row(&mut output.embeddings, input.tts_eos, input.codec_pad)?;
    let codec_rows = input.codec.len() / hidden;
    append_added_row(
        &mut output.embeddings,
        input.tts_pad,
        row(input.codec, codec_rows - 1, hidden),
    )?;
    finish(&mut output, hidden)?;
    Ok(output)
}

/// `BuildIclPrompt`.
pub fn build_icl_prompt(input: &IclPromptInput<'_>) -> Result<PromptOutput, String> {
    let hidden = input.hidden_size;
    require_rows(input.role, hidden, "role", false)?;
    require_rows(input.codec, hidden, "codec", false)?;
    require_rows(input.combined_text, hidden, "combined text", true)?;
    require_rows(input.reference_codec, hidden, "reference codec", false)?;
    require_rows(input.tts_bos, hidden, "TTS BOS", false)?;
    require_rows(input.tts_eos, hidden, "TTS EOS", false)?;
    require_rows(input.tts_pad, hidden, "TTS pad", false)?;
    if input.tts_bos.len() != hidden
        || input.tts_eos.len() != hidden
        || input.tts_pad.len() != hidden
    {
        return Err("Qwen3-TTS special embedding must contain one row".into());
    }
    let mut output = PromptOutput {
        tts_pad: input.tts_pad.to_vec(),
        ..PromptOutput::default()
    };
    build_prefix(
        &[],
        input.role,
        input.codec,
        input.tts_bos,
        input.tts_pad,
        hidden,
        &mut output,
    )?;
    let mut text = input.combined_text.to_vec();
    text.extend_from_slice(input.tts_eos);
    let text_rows = text.len() / hidden;
    let codec_rows = input.reference_codec.len() / hidden;
    for index in 0..text_rows.min(codec_rows) {
        append_added_row(
            &mut output.embeddings,
            row(&text, index, hidden),
            row(input.reference_codec, index, hidden),
        )?;
    }
    if text_rows > codec_rows {
        output.trailing_text = text[codec_rows * hidden..].to_vec();
    } else {
        for index in text_rows..codec_rows {
            append_added_row(
                &mut output.embeddings,
                input.tts_pad,
                row(input.reference_codec, index, hidden),
            )?;
        }
        output.trailing_text = output.tts_pad.clone();
    }
    finish(&mut output, hidden)?;
    Ok(output)
}
