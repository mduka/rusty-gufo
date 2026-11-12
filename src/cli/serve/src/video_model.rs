//! The MiniMax H3 surface the video job service and API need
//! (`src/models/minimax_h3/generation.hpp`): generation parameters and
//! requests, the pure helpers (presets, duration alignment, parameter
//! validation with the released geometry and VisualVAE constraints, the
//! parameters and telemetry JSON) and the injectable interfaces for what
//! needs the model: generation itself and the checkpoint inventory check.
//! No model implements them yet; a GGUF-based MiniMax H3 port follows.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use sha2::{Digest, Sha256};

pub const H3_GENERATION_SCHEMA: &str = "gufo.minimax-h3-text-generation.v1";
pub const H3_FRAMES_PER_SECOND: i32 = 24;
pub const H3_MAXIMUM_FRAMES: i32 = 345;
pub const H3_MAXIMUM_EVALUATIONS: i32 = 1000;
const H3_CANVAS_MULTIPLE: i32 = 32;
const H3_MAXIMUM_PIXELS: i64 = 768 * 1344;
const H3_VAE_SPATIAL_RATIO: i32 = 16;
const H3_VIDEO_VAE_CHUNK_LATENTS: i32 = 7;
const H3_VIDEO_VAE_CHUNK_STRIDE: i32 = 5;
const H3_VIDEO_VAE_FRAME_STRIDE: i32 = 17;
const MODEL_REVISION: &str = "42ed227ee7df40d41602854ae760620d6eb651fe";
const REFERENCE_REVISION: &str = "8974cc055ea9c02fcd14cc27dfda3e1027c05153";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationParameters {
    pub preset: String,
    pub internal_width: i32,
    pub internal_height: i32,
    pub output_width: i32,
    pub output_height: i32,
    pub frames: i32,
    pub evaluations: i32,
    pub active_blocks: i32,
    pub reuse_interval: i32,
    pub row_parallel_attention: bool,
    pub decode_audio: bool,
    pub mux: bool,
    pub selected_frames: Vec<i32>,
}

impl Default for GenerationParameters {
    fn default() -> Self {
        Self {
            preset: String::new(),
            internal_width: 0,
            internal_height: 0,
            output_width: 0,
            output_height: 0,
            frames: 0,
            evaluations: 0,
            active_blocks: 0,
            reuse_interval: 0,
            row_parallel_attention: true,
            decode_audio: true,
            mux: true,
            selected_frames: Vec::new(),
        }
    }
}

/// `minimax_h3::GenerationRequest`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GenerationRequest {
    pub model_root: PathBuf,
    pub source_manifest: PathBuf,
    pub output_path: PathBuf,
    pub frames_directory: PathBuf,
    pub latents_directory: PathBuf,
    pub prompt: String,
    pub seed: u64,
    pub parameters: GenerationParameters,
}

/// `minimax_h3::CancellationToken`.
#[derive(Debug, Default)]
pub struct CancellationToken(AtomicBool);

impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct InspectionTelemetry {
    pub json_bytes_read: u64,
    pub safetensors_header_bytes_read: u64,
    pub payload_bytes_read: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PromptEncoderLayerTelemetry {
    pub layer: u64,
    pub weight_bytes: u64,
    pub load_ms: f64,
    pub prefetch_wait_ms: f64,
    pub submit_ms: f64,
    pub gpu_ms: f64,
    pub dispatches: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PromptEncoderTelemetry {
    pub layers: Vec<PromptEncoderLayerTelemetry>,
    pub peak_device_bytes: u64,
    pub peak_registered_host_bytes: u64,
    pub live_registered_host_bytes_after: u64,
    pub total_dispatches: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenoiserTelemetry {
    pub setup_ms: f64,
    pub text_refiner_ms: f64,
    pub adaln_precompute_ms: f64,
    pub core_load_ms: f64,
    pub forward_ms: Vec<f64>,
    pub sampler_ms: f64,
    pub core_weight_bytes: u64,
    pub persistent_bytes: u64,
    pub scratch_bytes: u64,
    pub peak_live_bytes: u64,
    pub cumulative_allocation_bytes: u64,
    pub core_gemm_plan_validations: u64,
    pub minor_page_faults: u64,
    pub major_page_faults: u64,
    pub swap_bytes: u64,
    pub dispatches: u64,
    pub row_parallel_attention: bool,
}

impl Default for DenoiserTelemetry {
    fn default() -> Self {
        Self {
            setup_ms: 0.0,
            text_refiner_ms: 0.0,
            adaln_precompute_ms: 0.0,
            core_load_ms: 0.0,
            forward_ms: Vec::new(),
            sampler_ms: 0.0,
            core_weight_bytes: 0,
            persistent_bytes: 0,
            scratch_bytes: 0,
            peak_live_bytes: 0,
            cumulative_allocation_bytes: 0,
            core_gemm_plan_validations: 0,
            minor_page_faults: 0,
            major_page_faults: 0,
            swap_bytes: 0,
            dispatches: 0,
            row_parallel_attention: true,
        }
    }
}

/// Video and audio VAE telemetry share their shape (`VideoVaeTelemetry`,
/// `AudioVaeTelemetry`); `stage_ms` holds the video tiles or audio stages.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VaeTelemetry {
    pub load_ms: f64,
    pub stage_ms: Vec<f64>,
    pub decode_ms: f64,
    pub peak_live_bytes: u64,
    pub minor_page_faults: u64,
    pub major_page_faults: u64,
    pub swap_bytes: u64,
    pub dispatches: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MediaMuxTelemetry {
    pub video_bytes: u64,
    pub audio_bytes: u64,
    pub maximum_audio_staging_bytes: u64,
    pub child_status: i32,
    pub cancelled: bool,
}

impl Default for MediaMuxTelemetry {
    fn default() -> Self {
        Self {
            video_bytes: 0,
            audio_bytes: 0,
            maximum_audio_staging_bytes: 0,
            child_status: -1,
            cancelled: false,
        }
    }
}

/// `minimax_h3::GenerationTelemetry`, the fields its JSON reports.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerationTelemetry {
    pub inventory_ms: f64,
    pub tokenizer_ms: f64,
    pub prompt_ms: f64,
    pub denoiser_ms: f64,
    pub video_vae_ms: f64,
    pub first_preview_ms: f64,
    pub audio_vae_ms: f64,
    pub media_ms: f64,
    pub total_ms: f64,
    pub prompt_tokens: u64,
    pub inventory: InspectionTelemetry,
    pub prompt_encoder: PromptEncoderTelemetry,
    pub denoiser: DenoiserTelemetry,
    pub video_vae: VaeTelemetry,
    pub audio_vae: VaeTelemetry,
    pub media: MediaMuxTelemetry,
}

/// Reports `(phase, completed, total)` while a generation runs.
pub type GenerationProgress<'a> = dyn FnMut(&str, i32, i32) + 'a;

/// `minimax_h3::GenerateTextVideo`: writes the MP4 to `output_path`, or the
/// selected PPM frames to `frames_directory`; `Err` carries the detail.
pub type VideoRunner = Box<
    dyn Fn(
            &GenerationRequest,
            &CancellationToken,
            &mut GenerationProgress<'_>,
            &mut GenerationTelemetry,
        ) -> Result<(), String>
        + Send
        + Sync,
>;

/// `minimax_h3::ModelInventory::Inspect(model_root, source_manifest)`.
pub type InventoryCheck = Box<dyn Fn(&Path, &Path) -> Result<(), String> + Send + Sync>;

/// `AlignFrameCount`: at least 5 and congruent to 5 modulo 17.
pub fn align_frame_count(requested_frames: i32) -> i32 {
    let mut value = requested_frames.max(5);
    let remainder = (value - 5).rem_euclid(17);
    if remainder != 0 {
        value += 17 - remainder;
    }
    value
}

fn video_latent_frame_count(frame_count: i32) -> i32 {
    if frame_count <= 5 {
        2
    } else {
        ((frame_count - 5) / 17) * 5 + 2
    }
}

/// `ResolveDurationFrames`: `1` selects the 22-frame diagnostic extension;
/// 5..15 seconds at 24 fps snap upward to the VAE frame grid and must fit in
/// 345 frames.
pub fn resolve_duration_frames(seconds: &str) -> Result<i32, String> {
    // `std::from_chars` accepts neither a sign prefix nor whitespace.
    let parsed = if seconds.starts_with('+') || seconds.trim() != seconds {
        None
    } else {
        seconds.parse::<f64>().ok()
    };
    if let Some(duration) = parsed.filter(|d| d.is_finite() && !is_special(seconds)) {
        if duration == 1.0 {
            return Ok(22);
        }
        if (5.0..=15.0).contains(&duration) {
            let frames =
                align_frame_count((duration * f64::from(H3_FRAMES_PER_SECOND)).ceil() as i32);
            if frames <= H3_MAXIMUM_FRAMES {
                return Ok(frames);
            }
        }
    }
    Err(
        "MiniMax H3 seconds must be 1 (diagnostic) or 5..15 with 24-fps VAE alignment \
         fitting within 345 frames (14.375 seconds)"
            .into(),
    )
}

/// Spellings Rust parses but `std::from_chars` of a finite value does not.
fn is_special(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("inf") || lower.contains("nan")
}

/// `ResolveGenerationPreset`.
pub fn resolve_generation_preset(name: &str) -> Result<GenerationParameters, String> {
    let square =
        |preset: &str, internal: i32, output: i32, evaluations, active_blocks, reuse_interval| {
            GenerationParameters {
                preset: preset.into(),
                internal_width: internal,
                internal_height: internal,
                output_width: output,
                output_height: output,
                frames: 22,
                evaluations,
                active_blocks,
                reuse_interval,
                ..GenerationParameters::default()
            }
        };
    Ok(match name {
        "exact" | "exact-512" => square("exact-512", 512, 512, 49, 50, 1),
        "exact-1344x768" | "fullres" => GenerationParameters {
            preset: "exact-1344x768".into(),
            internal_width: 1344,
            internal_height: 768,
            output_width: 1344,
            output_height: 768,
            frames: 124,
            evaluations: 49,
            active_blocks: 50,
            reuse_interval: 1,
            ..GenerationParameters::default()
        },
        "fast" | "fast-384" => square("fast-384", 384, 512, 19, 45, 2),
        "aggressive" | "aggressive-320" => square("aggressive-320", 320, 512, 19, 40, 3),
        "dev" | "development-256" => GenerationParameters {
            decode_audio: false,
            mux: false,
            selected_frames: vec![0, 11, 21],
            ..square("development-256", 256, 256, 4, 50, 1)
        },
        _ => {
            return Err(
                "unknown MiniMax H3 preset; expected exact, exact-1344x768, fast, \
                        aggressive, or dev"
                    .into(),
            );
        }
    })
}

struct Geometry {
    width: i32,
    height: i32,
    latent_width: i32,
    latent_height: i32,
    frames: i32,
    video_latent_frames: i32,
}

/// `ResolveGenerationGeometry`, without the audio latent count it also
/// computes.
fn resolve_generation_geometry(
    width: i32,
    height: i32,
    requested_frames: i32,
) -> Result<Geometry, String> {
    if width < H3_CANVAS_MULTIPLE
        || height < H3_CANVAS_MULTIPLE
        || width % H3_CANVAS_MULTIPLE != 0
        || height % H3_CANVAS_MULTIPLE != 0
    {
        return Err("MiniMax H3 width and height must be multiples of 32 and at least 32".into());
    }
    if i64::from(width) * i64::from(height) > H3_MAXIMUM_PIXELS {
        return Err("MiniMax H3 canvas exceeds the released 768*1344 pixel limit".into());
    }
    if !(5..=H3_MAXIMUM_FRAMES).contains(&requested_frames) {
        return Err("MiniMax H3 requested frame count must be in the released 5..345 range".into());
    }
    let frames = align_frame_count(requested_frames);
    if frames > H3_MAXIMUM_FRAMES {
        return Err(
            "MiniMax H3 aligned frame count exceeds the released 15-second limit (345 aligned frames)"
                .into(),
        );
    }
    Ok(Geometry {
        width,
        height,
        latent_width: width / H3_VAE_SPATIAL_RATIO,
        latent_height: height / H3_VAE_SPATIAL_RATIO,
        frames,
        video_latent_frames: video_latent_frame_count(frames),
    })
}

/// The checks of `ResolveVideoVaePlan` that can reject a geometry (the tile
/// plan itself never fails for 32-aligned extents).
fn check_video_vae_plan(geometry: &Geometry) -> Result<(), String> {
    let latent_time = geometry.video_latent_frames;
    if latent_time < H3_VIDEO_VAE_CHUNK_LATENTS {
        return Err("MiniMax H3 VisualVAE decoding requires at least 22 aligned frames".into());
    }
    if geometry.width < 1
        || geometry.height < 1
        || geometry.latent_width < 1
        || geometry.latent_height < 1
        || geometry.width != geometry.latent_width * H3_VAE_SPATIAL_RATIO
        || geometry.height != geometry.latent_height * H3_VAE_SPATIAL_RATIO
        || (latent_time - 2) % H3_VIDEO_VAE_CHUNK_STRIDE != 0
    {
        return Err("invalid MiniMax H3 VisualVAE geometry".into());
    }
    let chunks = (latent_time - 2) / H3_VIDEO_VAE_CHUNK_STRIDE;
    if chunks * H3_VIDEO_VAE_FRAME_STRIDE + 5 != geometry.frames {
        return Err("MiniMax H3 VisualVAE temporal geometry differs from output".into());
    }
    for extent in [geometry.height, geometry.width] {
        if extent < 1 || extent % H3_VAE_SPATIAL_RATIO != 0 {
            return Err("MiniMax H3 VisualVAE extent must be a multiple of 16".into());
        }
    }
    Ok(())
}

/// `ValidateGenerationParameters`.
pub fn validate_generation_parameters(parameters: &GenerationParameters) -> Result<(), String> {
    if parameters.preset.is_empty()
        || parameters.output_width < 2
        || parameters.output_height < 2
        || parameters.output_width & 1 != 0
        || parameters.output_height & 1 != 0
        || parameters.evaluations < 2
        || parameters.evaluations > H3_MAXIMUM_EVALUATIONS
        || !(3..=50).contains(&parameters.active_blocks)
        || !(1..=32).contains(&parameters.reuse_interval)
    {
        return Err("invalid MiniMax H3 generation parameters".into());
    }
    let geometry = match resolve_generation_geometry(
        parameters.internal_width,
        parameters.internal_height,
        parameters.frames,
    ) {
        Ok(geometry) if geometry.frames == parameters.frames => geometry,
        Ok(_) => return Err("MiniMax H3 generation requires a legal frame count".into()),
        Err(error) => return Err(error),
    };
    check_video_vae_plan(&geometry)?;
    let mut previous = -1;
    for &frame in &parameters.selected_frames {
        if frame <= previous || frame < 0 || frame >= parameters.frames {
            return Err("MiniMax H3 selected frames must be increasing and in range".into());
        }
        previous = frame;
    }
    if parameters.mux && (!parameters.decode_audio || !parameters.selected_frames.is_empty()) {
        return Err("MiniMax H3 MP4 output requires audio and every video frame".into());
    }
    Ok(())
}

/// `EscapeJson` of the H3 writers: quotes, backslashes, the named control
/// escapes and `\u00XX` for other control bytes.
fn escape_json(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{8}' => output.push_str("\\b"),
            '\u{c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(output, "\\u{:04x}", c as u32);
            }
            c => output.push(c),
        }
    }
    output
}

/// `GenerationParametersJson`.
pub fn generation_parameters_json(
    parameters: &GenerationParameters,
    seed: u64,
    prompt_sha256: &str,
) -> String {
    let selected: Vec<String> = parameters
        .selected_frames
        .iter()
        .map(|f| f.to_string())
        .collect();
    format!(
        "{{\n  \"schema\": \"{H3_GENERATION_SCHEMA}\",\n  \"backend\": \"rocm-hip-gfx1151\",\n  \
         \"precision\": \"bf16-f32\",\n  \"model_repository\": \"MiniMaxAI/MiniMax-H3\",\n  \
         \"model_revision\": \"{MODEL_REVISION}\",\n  \"reference_repository\": \"antirez/h3.c\",\n  \
         \"reference_revision\": \"{REFERENCE_REVISION}\",\n  \"preset\": \"{}\",\n  \
         \"prompt_sha256\": \"{}\",\n  \"seed\": {seed},\n  \"internal_width\": {},\n  \
         \"internal_height\": {},\n  \"output_width\": {},\n  \"output_height\": {},\n  \
         \"fps\": {H3_FRAMES_PER_SECOND},\n  \"frames\": {},\n  \"evaluations\": {},\n  \
         \"active_blocks\": {},\n  \"reuse_interval\": {},\n  \"attention_kernel\": \"{}\",\n  \
         \"token_reduction\": false,\n  \"decode_audio\": {},\n  \"mux\": {},\n  \
         \"selected_frames\": [{}]\n}}\n",
        escape_json(&parameters.preset),
        escape_json(prompt_sha256),
        parameters.internal_width,
        parameters.internal_height,
        parameters.output_width,
        parameters.output_height,
        parameters.frames,
        parameters.evaluations,
        parameters.active_blocks,
        parameters.reuse_interval,
        if parameters.row_parallel_attention {
            "row_parallel"
        } else {
            "scalar"
        },
        parameters.decode_audio,
        parameters.mux,
        selected.join(", "),
    )
}

fn fixed3(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("{v:.3}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `GenerationTelemetryJson`: floating-point values with three decimals.
pub fn generation_telemetry_json(t: &GenerationTelemetry) -> String {
    let layers = &t.prompt_encoder.layers;
    let prompt_weight_bytes: u64 = layers.iter().map(|l| l.weight_bytes).sum();
    let sum = |f: fn(&PromptEncoderLayerTelemetry) -> f64| layers.iter().map(f).sum::<f64>();
    let layer_json: Vec<String> = layers
        .iter()
        .map(|l| {
            format!(
                "{{\"layer\": {}, \"weight_bytes\": {}, \"load_ms\": {:.3}, \"prefetch_wait_ms\": {:.3}, \
                 \"submit_ms\": {:.3}, \"gpu_ms\": {:.3}, \"dispatches\": {}}}",
                l.layer, l.weight_bytes, l.load_ms, l.prefetch_wait_ms, l.submit_ms, l.gpu_ms, l.dispatches
            )
        })
        .collect();
    let mut out = String::new();
    let _ = write!(
        out,
        "{{\n  \"schema\": \"gufo.minimax-h3-generation-telemetry.v1\",\n  \"prompt_tokens\": {},\n  \
         \"inventory_ms\": {:.3},\n  \"inventory_json_bytes\": {},\n  \"inventory_header_bytes\": {},\n  \
         \"inventory_payload_bytes\": {},\n  \"tokenizer_ms\": {:.3},\n  \"prompt_ms\": {:.3},\n  \
         \"prompt_weight_bytes\": {},\n  \"prompt_load_ms\": {:.3},\n  \"prompt_prefetch_wait_ms\": {:.3},\n  \
         \"prompt_submit_ms\": {:.3},\n  \"prompt_gpu_ms\": {:.3},\n  \"prompt_peak_device_bytes\": {},\n  \
         \"prompt_peak_registered_host_bytes\": {},\n  \"prompt_live_registered_host_bytes_after\": {},\n  \
         \"prompt_layers\": [{}],\n",
        t.prompt_tokens,
        t.inventory_ms,
        t.inventory.json_bytes_read,
        t.inventory.safetensors_header_bytes_read,
        t.inventory.payload_bytes_read,
        t.tokenizer_ms,
        t.prompt_ms,
        prompt_weight_bytes,
        sum(|l| l.load_ms),
        sum(|l| l.prefetch_wait_ms),
        sum(|l| l.submit_ms),
        sum(|l| l.gpu_ms),
        t.prompt_encoder.peak_device_bytes,
        t.prompt_encoder.peak_registered_host_bytes,
        t.prompt_encoder.live_registered_host_bytes_after,
        layer_json.join(", "),
    );
    let d = &t.denoiser;
    let _ = write!(
        out,
        "  \"denoiser_ms\": {:.3},\n  \"denoiser_setup_ms\": {:.3},\n  \"denoiser_text_refiner_ms\": {:.3},\n  \
         \"denoiser_adaln_precompute_ms\": {:.3},\n  \"denoiser_core_load_ms\": {:.3},\n  \
         \"denoiser_sampler_ms\": {:.3},\n  \"denoiser_evaluations\": {},\n  \"denoiser_forward_ms\": [{}],\n  \
         \"denoiser_core_weight_bytes\": {},\n  \"denoiser_persistent_bytes\": {},\n  \
         \"denoiser_scratch_bytes\": {},\n  \"denoiser_cumulative_allocation_bytes\": {},\n  \
         \"denoiser_core_gemm_plan_validations\": {},\n",
        t.denoiser_ms,
        d.setup_ms,
        d.text_refiner_ms,
        d.adaln_precompute_ms,
        d.core_load_ms,
        d.sampler_ms,
        d.forward_ms.len(),
        fixed3(&d.forward_ms),
        d.core_weight_bytes,
        d.persistent_bytes,
        d.scratch_bytes,
        d.cumulative_allocation_bytes,
        d.core_gemm_plan_validations,
    );
    let (v, a, m) = (&t.video_vae, &t.audio_vae, &t.media);
    let _ = write!(
        out,
        "  \"video_vae_ms\": {:.3},\n  \"video_vae_load_ms\": {:.3},\n  \"video_vae_decode_ms\": {:.3},\n  \
         \"video_vae_tile_ms\": [{}],\n  \"first_preview_ms\": {:.3},\n  \"audio_vae_ms\": {:.3},\n  \
         \"audio_vae_load_ms\": {:.3},\n  \"audio_vae_decode_ms\": {:.3},\n  \"audio_vae_stage_ms\": [{}],\n  \
         \"media_ms\": {:.3},\n  \"media_maximum_audio_staging_bytes\": {},\n  \"media_child_status\": {},\n  \
         \"media_cancelled\": {},\n  \"total_ms\": {:.3},\n  \"peak_denoiser_bytes\": {},\n  \
         \"peak_video_vae_bytes\": {},\n  \"peak_audio_vae_bytes\": {},\n  \"prompt_dispatches\": {},\n  \
         \"denoiser_dispatches\": {},\n  \"denoiser_attention_kernel\": \"{}\",\n  \"video_vae_dispatches\": {},\n  \
         \"audio_vae_dispatches\": {},\n  \"media_video_bytes\": {},\n  \"media_audio_bytes\": {},\n  \
         \"minor_page_faults\": {},\n  \"major_page_faults\": {},\n  \"swap_bytes\": {}\n}}\n",
        t.video_vae_ms,
        v.load_ms,
        v.decode_ms,
        fixed3(&v.stage_ms),
        t.first_preview_ms,
        t.audio_vae_ms,
        a.load_ms,
        a.decode_ms,
        fixed3(&a.stage_ms),
        t.media_ms,
        m.maximum_audio_staging_bytes,
        m.child_status,
        m.cancelled,
        t.total_ms,
        d.peak_live_bytes,
        v.peak_live_bytes,
        a.peak_live_bytes,
        t.prompt_encoder.total_dispatches,
        d.dispatches,
        if d.row_parallel_attention {
            "row_parallel"
        } else {
            "scalar"
        },
        v.dispatches,
        a.dispatches,
        m.video_bytes,
        m.audio_bytes,
        d.minor_page_faults + v.minor_page_faults + a.minor_page_faults,
        d.major_page_faults + v.major_page_faults + a.major_page_faults,
        d.swap_bytes.max(v.swap_bytes).max(a.swap_bytes),
    );
    out
}

/// Lowercase hex SHA-256 (`minimax_h3::Sha256`).
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_align_to_the_vae_grid() {
        for (seconds, frames) in [
            ("1", 22),
            ("5", 124),
            ("8", 192),
            ("10", 243),
            ("14.375", 345),
        ] {
            assert_eq!(resolve_duration_frames(seconds), Ok(frames), "{seconds}");
        }
        for seconds in [
            "0", "4", "14.376", "15", "16", "+5", " 5", "inf", "nan", "5x", "",
        ] {
            assert!(resolve_duration_frames(seconds).is_err(), "{seconds}");
        }
    }

    #[test]
    fn presets_validate() {
        for name in ["exact", "exact-1344x768", "fast", "aggressive", "dev"] {
            let parameters = resolve_generation_preset(name).expect("preset");
            assert_eq!(
                validate_generation_parameters(&parameters),
                Ok(()),
                "{name}"
            );
        }
        let mut short = resolve_generation_preset("dev").expect("preset");
        short.frames = 5;
        short.selected_frames = vec![4];
        assert_eq!(
            validate_generation_parameters(&short),
            Err("MiniMax H3 VisualVAE decoding requires at least 22 aligned frames".into())
        );
        let mut beyond = resolve_generation_preset("fast").expect("preset");
        beyond.frames = 362;
        assert!(validate_generation_parameters(&beyond).is_err());
    }

    #[test]
    fn parameters_json_matches_the_cpp_layout() {
        let parameters = resolve_generation_preset("dev").expect("preset");
        let json = generation_parameters_json(&parameters, 7, "abc");
        assert!(json.starts_with("{\n  \"schema\": \"gufo.minimax-h3-text-generation.v1\",\n"));
        assert!(json.ends_with("  \"mux\": false,\n  \"selected_frames\": [0, 11, 21]\n}\n"));
        assert!(gufo_core::json::parse_str(&json).is_ok());
        let telemetry = generation_telemetry_json(&GenerationTelemetry::default());
        assert!(gufo_core::json::parse_str(&telemetry).is_ok());
        assert!(telemetry.contains("\"media_child_status\": -1,"));
    }
}
