//! Image prompts against the C++ reference, bit for bit (`vision`
//! fixtures, the probe's mode `vision`):
//! cargo test --profile gpu-test -p gufo-qwen35moe --features gpu --test vision
//!
//! Needs `GUFO_QWEN35MOE_MODEL` (UD-Q4_K_M), `GUFO_QWEN35MOE_Q8_MODEL`
//! (Q8_0) and `GUFO_QWEN35MOE_MMPROJ` (the BF16 vision sidecar), and fails
//! without them. The synthetic image (the fixture set's `make_synthetic.py`)
//! is generated here; the two photos come from the reference fixture set
//! (`GUFO_QWEN35MOE_FIXTURES`, default `target/qwen35moe`).
//!
//! Per case, as `gufo prompt --image ... --think off` builds it (`drivers.cpp`
//! `LoadQwenVision`, `PrepareVision`): one user message with the images, the
//! prompt from `vision::prepare` (context 4096), then `Reset`,
//! `ConfigureVision` and `ForwardPromptBatch` of the whole prompt, which
//! encodes the images on first use and injects them over the image-pad rows
//! at their mRoPE positions; then 32 greedy `ForwardToken` steps. Tokens and
//! the FNV-1a digest of every logits row match, and the prefill's top five;
//! then greedy `Generate` of 32 tokens with the images still configured.
//! The MTP and DFlash2 verifiers of `gufo prompt --speculative` run on the
//! same prompts; DFlash2 also needs `GUFO_QWEN35MOE_DFLASH_MODEL`.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use common::{Q8_0, Quant, SERIAL, UD_Q4_K_M, fnv1a64, trace, u32s};
use gufo_core::qwen_tokenizer::StopTokens;
use gufo_core::sampling::{SamplerState, SamplingConfig};
use gufo_core::text::{ChatMessage, ChatRole, ImageDetail, ImagePart, ReasoningOptions};
use gufo_qwen::chat_template::resolve_options;
use gufo_qwen::vision::{self, Encoder};
use gufo_qwen35moe::dflash::DFlash;
use gufo_qwen35moe::dflash::policy::DraftPolicy;
use gufo_qwen35moe::mtp::{DEFAULT_PRIME_CHUNK, MtpBackend, MtpDraftConfig, MtpExec, MtpModel};
use gufo_qwen35moe::verifier::{StepRequest, Verifier};
use gufo_qwen35moe::{Engine, Speculative};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// `drivers.cpp` `PrepareVision`: the CLI's context for image prompts.
const PREPARE_CONTEXT: u32 = 4096;
const SYNTHETIC: &str = "synthetic-200x150";

fn reference_set() -> PathBuf {
    std::env::var_os("GUFO_QWEN35MOE_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../target/qwen35moe"),
        PathBuf::from,
    )
}

fn sidecar() -> PathBuf {
    // Unset, the path is the variable itself, so the failure says what to set.
    std::env::var_os("GUFO_QWEN35MOE_MMPROJ")
        .map_or_else(|| "$GUFO_QWEN35MOE_MMPROJ".into(), PathBuf::from)
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// One pixel of `make_synthetic.py`: sky gradient, sun, a red house with a
/// door and a roof, striped ground.
fn synthetic_pixel(x: u32, y: u32) -> [u8; 3] {
    let (dx, dy) = (i64::from(x) - 150, i64::from(y) - 35);
    if dx * dx + dy * dy <= 18 * 18 {
        return [250, 210, 40];
    }
    if (60..120).contains(&x) && (70..115).contains(&y) {
        if (82..98).contains(&x) && (90..115).contains(&y) {
            return [90, 50, 20];
        }
        return [200, 40, 40];
    }
    if y >= 115 {
        return [40, (140 + 30 * ((x / 8 + y / 8) % 2)) as u8, 50];
    }
    // Python compares the integer with the float product exactly.
    if (50..70).contains(&y) && (f64::from(x) - 90.0).abs() <= f64::from(y - 50) * 1.6 {
        return [120, 60, 30];
    }
    [
        (100 + y / 2) as u8,
        (150 + y / 3) as u8,
        (255 - y / 4) as u8,
    ]
}

/// The synthetic image as an 8-bit RGB PNG: other bytes than the reference
/// file (another deflate), the same pixels.
fn synthetic_png() -> Vec<u8> {
    let (width, height) = (200, 150);
    let pixels: Vec<u8> = (0..height)
        .flat_map(|y| (0..width).flat_map(move |x| synthetic_pixel(x, y)))
        .collect();
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&pixels).unwrap();
    writer.finish().unwrap();
    bytes
}

/// A fixture image's PNG: the synthetic one generated, a photo read from the
/// reference set and checked against its recorded hash.
fn png_bytes(fixture: &Value, name: &str) -> Vec<u8> {
    if name == SYNTHETIC {
        return synthetic_png();
    }
    let path = reference_set().join("images").join(format!("{name}.png"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}; set GUFO_QWEN35MOE_FIXTURES to the qwen35moe reference fixtures",
            path.display()
        )
    });
    let entry = fixture["images"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["image"] == name)
        .unwrap();
    assert_eq!(
        sha256(&bytes),
        entry["file_sha256"].as_str().unwrap(),
        "{name}"
    );
    bytes
}

/// The five largest logits, descending, the lowest id first on ties (the
/// probe's `top5`), as `(id, bits)`.
fn top5(logits: &[f32]) -> Vec<(u32, u32)> {
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_by(|&a, &b| {
        logits[b as usize]
            .total_cmp(&logits[a as usize])
            .then(a.cmp(&b))
    });
    order
        .into_iter()
        .take(5)
        .map(|id| (id, logits[id as usize].to_bits()))
        .collect()
}

/// A fixture `top5` entry (`[[id, value], ...]`, values printed with
/// `%.9g`, which parse back to the exact f32).
fn expected_top5(entry: &Value) -> Vec<(u32, u32)> {
    entry
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| {
            let id = u32::try_from(pair[0].as_u64().unwrap()).unwrap();
            (id, (pair[1].as_f64().unwrap() as f32).to_bits())
        })
        .collect()
}

/// The image prompt of a fixture case (`vision::Prepare`).
fn prepare(engine: &Engine, encoder: &Encoder, fixture: &Value, case: &Value) -> vision::Prompt {
    let mut message = ChatMessage::new(ChatRole::User, case["user_text"].as_str().unwrap());
    for image in case["images"].as_array().unwrap() {
        message.images.push(ImagePart {
            offset: 0,
            bytes: Arc::new(png_bytes(fixture, image.as_str().unwrap())),
            detail: ImageDetail::Auto,
        });
    }
    let reasoning = ReasoningOptions {
        enabled: Some(false),
        ..ReasoningOptions::default()
    };
    vision::prepare(
        &engine.weights().tokenizer,
        &[message],
        &[],
        &resolve_options(&reasoning, false),
        encoder.identity(),
        PREPARE_CONTEXT,
        None,
    )
    .unwrap()
}

fn images_match(quant: Quant) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = quant.fixture("vision");
    let max_context = u32::try_from(fixture["max_context"].as_u64().unwrap()).unwrap();
    let steps = fixture["steps"].as_u64().unwrap() as usize;
    let model = quant.model();
    let mut engine = Engine::load(&model, max_context).unwrap();
    // `LoadQwenVision`: the trunk's width selects the projector.
    let width =
        vision::projection_width(&engine.config().architecture, engine.config().hidden_size);
    assert_eq!(width, Some(2048));
    let encoder = Encoder::open(&model, &sidecar(), 2048)
        .unwrap_or_else(|e| panic!("{}: {}", sidecar().display(), e.message()))
        .unwrap();

    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let name = format!("{}: {}", quant.set, case["name"].as_str().unwrap());
        let prompt = Arc::new(prepare(&engine, &encoder, &fixture, case));
        assert_eq!(prompt.tokens, u32s(&case["prompt"]), "{name}: prompt");
        assert_eq!(
            i64::from(prompt.rope.delta()),
            case["rope_delta"].as_i64().unwrap(),
            "{name}: rope_delta"
        );

        engine.reset().unwrap();
        let (model, session) = engine.parts_mut();
        model
            .configure_vision(
                session,
                Some(Arc::clone(&prompt)),
                Some(Arc::clone(&encoder)),
            )
            .unwrap();
        let next = engine.forward_prompt(&prompt.tokens, 0, true).unwrap();
        let logits = engine.logits().unwrap();
        let prefill_top5 = top5(&logits);
        if prefill_top5 != expected_top5(&case["top5"][0]) {
            failures.push(format!(
                "{name}: prefill logits {} (C++ {}), top5 {prefill_top5:?}",
                fnv1a64(&logits),
                case["logits_fnv1a64"][0]
            ));
            continue;
        }
        assert_eq!(case["ids"].as_array().unwrap().len(), steps + 1);
        if let Some(divergence) = trace(&mut engine, case, next, prompt.tokens.len(), false) {
            failures.push(format!("{name}: {divergence}"));
            continue;
        }

        let expected = u32s(&case["generate"]["ids"]);
        let max_new = case["generate"]["max_new_tokens"].as_u64().unwrap() as usize;
        let generated = engine.generate(&prompt.tokens, max_new, |_| true).unwrap();
        if generated != expected {
            let first = generated
                .iter()
                .zip(&expected)
                .position(|(a, b)| a != b)
                .unwrap_or(generated.len().min(expected.len()));
            failures.push(format!("{name}: generate: first differing token {first}"));
        }
    }
    // The images went through the encoder.
    assert!(encoder.resident_bytes() > 0);
    let (model, session) = engine.parts_mut();
    model.configure_vision(session, None, None).unwrap();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn qwen36_35b_a3b_ud_q4_k_m() {
    images_match(UD_Q4_K_M);
}

#[test]
fn qwen36_35b_a3b_q8_0() {
    images_match(Q8_0);
}

/// A speculative verifier of the CLI (`drivers.cpp` `CreateQwenVerifier`
/// with the host defaults, as the probe's `MtpBackend`/`DFlashBackend` and
/// `CliOptions` build them).
#[derive(Clone, Copy)]
enum Drafter {
    /// The target GGUF's own MTP head; draft length 7 to 1, adaptive. The
    /// head reads the target's image input at its own positions.
    Mtp,
    /// The DFlash2 drafter (`GUFO_QWEN35MOE_DFLASH_MODEL`), 7 tokens, the
    /// default policy, no copy rows; it has no image input and drafts from
    /// the target's features.
    DFlash2,
}

fn dflash_model() -> PathBuf {
    std::env::var_os("GUFO_QWEN35MOE_DFLASH_MODEL")
        .map_or_else(|| "$GUFO_QWEN35MOE_DFLASH_MODEL".into(), PathBuf::from)
}

/// The probe's `RunVerifier` on each image prompt: `Prime`, then
/// `VerifyStep` over the committed sequence (EOS ignored) until 32 tokens;
/// every step's drafted, accepted and emitted tokens match.
fn speculative_images_match(quant: Quant, drafter: Drafter) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = quant.fixture("vision");
    let max_context = u32::try_from(fixture["max_context"].as_u64().unwrap()).unwrap();
    let total = fixture["speculative_tokens"].as_u64().unwrap() as usize;
    let model = quant.model();
    let mut engine = Engine::load(&model, max_context).unwrap();
    let encoder = Encoder::open(&model, &sidecar(), 2048)
        .unwrap_or_else(|e| panic!("{}: {}", sidecar().display(), e.message()))
        .unwrap();
    // The MTP head runs on the serving verifier, a fresh one per case as
    // the probe creates its backend; DFlash2 on the command's verifier.
    let (key, mut speculative, mtp) = match drafter {
        Drafter::Mtp => {
            let mtp = MtpModel::load(engine.model().stream(), &model, engine.weights()).unwrap();
            ("mtp", None, Some(mtp))
        }
        Drafter::DFlash2 => {
            let draft = DFlash::load(
                &dflash_model(),
                engine.weights(),
                max_context,
                7,
                DraftPolicy::Adaptive,
            )
            .unwrap();
            ("dflash", Some(Speculative::new(draft)), None)
        }
    };

    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let name = format!("{} {key}: {}", quant.set, case["name"].as_str().unwrap());
        let expected = &case[key];
        let prompt = Arc::new(prepare(&engine, &encoder, &fixture, case));
        assert_eq!(prompt.tokens, u32s(&case["prompt"]), "{name}: prompt");
        let (target, session) = engine.parts_mut();
        target
            .configure_vision(
                session,
                Some(Arc::clone(&prompt)),
                Some(Arc::clone(&encoder)),
            )
            .unwrap();
        let mut verifier = mtp.as_ref().map(|mtp| {
            let config = MtpDraftConfig {
                max_context,
                max_draft_tokens: 7,
                prime_chunk: DEFAULT_PRIME_CHUNK,
            };
            let backend = MtpBackend::new(engine.model().stream(), mtp, config).unwrap();
            let mut verifier = Verifier::with_draft(backend, 1, 7, true);
            verifier.set_retain_frontier_logits(false);
            verifier
        });
        let mut sampler = SamplerState::new(SamplingConfig::default(), &prompt.tokens).unwrap();
        let mut current = match (&mut verifier, &mut speculative, &mtp) {
            (Some(verifier), _, Some(mtp)) => {
                let (model, session) = engine.parts_mut();
                verifier
                    .prime(&mut MtpExec { model, mtp }, session, &prompt.tokens, true)
                    .unwrap()
            }
            (_, Some(speculative), _) => speculative
                .prime(&mut engine, &prompt.tokens, true)
                .unwrap(),
            _ => unreachable!(),
        };
        if u64::from(current) != expected["first"].as_u64().unwrap() {
            failures.push(format!(
                "{name}: first token {current}, C++ {}",
                expected["first"]
            ));
            continue;
        }
        let mut sequence = prompt.tokens.clone();
        let mut position = sequence.len() as u32;
        sequence.push(current);
        let mut ids = Vec::new();
        let mut diverged = false;
        for (index, want) in expected["steps"].as_array().unwrap().iter().enumerate() {
            let (got, emitted, next) = match (&mut verifier, &mut speculative, &mtp) {
                (Some(verifier), _, Some(mtp)) => {
                    let (model, session) = engine.parts_mut();
                    // EOS ignored (999999 in the probe).
                    let step = Verifier::verify_step(
                        &mut MtpExec { model, mtp },
                        &mut StepRequest {
                            verifier,
                            session,
                            sequence: &sequence,
                            position,
                            current_token: current,
                            stops: StopTokens::default(),
                            max_emitted: (total - ids.len()) as u32,
                            sampler: &mut sampler,
                        },
                    )
                    .unwrap();
                    let mut got = vec![step.draft_count as u32, step.accepted_count as u32];
                    got.extend(&step.emitted);
                    (got, step.emitted, step.next_token)
                }
                (_, Some(speculative), _) => {
                    let step = speculative
                        .verify_step(
                            &mut engine,
                            &sequence,
                            current,
                            position,
                            999_999,
                            (total - ids.len()) as u32,
                        )
                        .unwrap();
                    let mut got = vec![step.drafted as u32, step.accepted as u32];
                    got.extend(&step.emitted);
                    (got, step.emitted, step.next_token)
                }
                _ => unreachable!(),
            };
            if got != u32s(want) {
                failures.push(format!("{name}: step {index}: {got:?}, C++ {want}"));
                diverged = true;
                break;
            }
            for &token in &emitted {
                if ids.len() < total {
                    ids.push(token);
                    sequence.push(token);
                    position += 1;
                }
            }
            current = next;
        }
        if !diverged && ids != u32s(&expected["ids"]) {
            failures.push(format!("{name}: ids differ"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn mtp_with_images_ud_q4_k_m() {
    speculative_images_match(UD_Q4_K_M, Drafter::Mtp);
}

#[test]
fn mtp_with_images_q8_0() {
    speculative_images_match(Q8_0, Drafter::Mtp);
}

#[test]
fn dflash2_with_images_ud_q4_k_m() {
    speculative_images_match(UD_Q4_K_M, Drafter::DFlash2);
}

#[test]
fn dflash2_with_images_q8_0() {
    speculative_images_match(Q8_0, Drafter::DFlash2);
}
