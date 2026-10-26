//! One resident model and any number of sessions, a port of `Model` and
//! `Session` (`engine.cpp`): prompt sync with prefix reuse, MTP proposals
//! (greedy drafts verified on the device, sampled proposals with rejection
//! and residual correction, the length policy), and host snapshots of
//! payload version 14.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use gufo_core::qwen_tokenizer::QwenTokenizer;
use gufo_core::sampling::{self, SamplerState, TokenId};
use gufo_hip::{MappedFile, Stream};
use gufo_qwen::vision::{Encoder, Prompt};

use crate::batch::{BATCH_SESSIONS, BatchItem, MtpBatchItem, MtpHeadItem};
use crate::borrowed::Handle;
use crate::checkpoint::{CheckpointStorage, PrefillCheckpoint};
use crate::config::Config;
use crate::device::DeviceModel;
use crate::draft_vocab::{DRAFT_EXTRA_ROWS, DRAFT_VOCAB, DraftVocab, draft_subset_allowed};
use crate::executor::{Executor, ForwardMode, Options, SessionMode};
use crate::lookup::{self, LookupState, PromptLookup};
use crate::mtp::{self, ArgmaxCandidate, MTP_CANDIDATES, MtpCandidates, MtpOutput};
use crate::ngram::NgramTable;
use crate::policy::{
    self, BatchController, BatchRow, DraftDecision, LengthController, LengthState,
    MAX_DRAFT_TOKENS, Proposal,
};
use crate::rope_scaling::{RopeScaling, validate_rope_scaling};
use crate::snapshot::SnapshotInfo;
use crate::state;
use crate::weights::{GgufFiles, ModelTensors, MtpTensors};

mod batch;
mod decode;
mod session;
mod snapshots;

/// gfx1151, served repository prompts: 4096-token chunks prefill 24K to 102K
/// tokens 6 to 11% faster than 2048. Prefill logits do
/// not depend on the chunk size: queries before the indexer budget always
/// take the dense attention tiles.
const PREFILL_CHUNK_TOKENS: u32 = 4096;
/// A prompt ending this close past a chunk finishes in that chunk: a
/// separate tail pass costs about as much as this many more chunk rows
///.
const PREFILL_TAIL_TOKENS: u32 = 128;

/// Compatibility version of session snapshots; bump on payload or inference
/// arithmetic changes (`kSnapshotPayloadVersion`).
pub const SNAPSHOT_PAYLOAD_VERSION: u32 = 18;
const SESSION_MAGIC: [u8; 8] = *b"QFNSESS1";
/// `sizeof(SessionSnapshotHeader)`.
const SESSION_HEADER_BYTES: usize = 128;

#[derive(Debug, Clone)]
pub struct ModelOptions {
    pub max_context: u32,
    /// The MTP sidecar (`mtp-...-shared-*.gguf`); without it decoding is
    /// autoregressive.
    pub mtp_model_path: Option<PathBuf>,
    /// Most proposals per cycle; acceptance history picks the length.
    pub max_draft_tokens: u32,
    /// Serving capacity the calibrated cost model assumes, fixed so seeded
    /// requests replay independently of scheduling.
    pub decode_concurrency: u32,
    /// The vision sidecar, which must exist; without one the canonical
    /// `mmproj-BF16.gguf` next to the model (or one directory up) serves
    /// images when present.
    pub vision_model_path: Option<PathBuf>,
    /// Static YaRN, which the caller derives from the requested context
    /// (`rope_scaling_for_context`); `None` keeps the artifact's own, which
    /// is always off: no GGUF key sets it.
    pub rope_scaling: Option<RopeScaling>,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            max_context: 4096,
            mtp_model_path: None,
            max_draft_tokens: MAX_DRAFT_TOKENS,
            decode_concurrency: 1,
            vision_model_path: None,
            rope_scaling: None,
        }
    }
}

pub struct Model {
    options: ModelOptions,
    snapshots: Snapshotter,
    name: String,
    tokenizer: QwenTokenizer,
    executor: Executor,
    batch_policy: BatchController,
    vision: Option<Arc<Encoder>>,
    /// The special tokens above the draft prefix, ascending, when MTP drafts
    /// may use the small head ([`crate::draft_vocab`]).
    draft_specials: Option<Vec<i32>>,
}

/// Draft cycles and their acceptance (`SpeculativeStats`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    pub cycles: u64,
    pub drafted: u64,
    pub accepted: u64,
    /// Cycles whose draft scored the small head.
    pub small_head_cycles: u64,
    /// Of drafted and accepted: proposals copied by prompt lookup.
    pub lookup: u64,
    pub lookup_accepted: u64,
}

/// Tokens of one decode step and whether it stopped.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DecodeResult {
    pub tokens: Vec<i32>,
    pub stop: bool,
}

/// One conversation's context: its tokens, the last token's logits and the
/// executor session with the device state (`Session`).
pub struct Session {
    native: state::Session,
    tokens: Vec<i32>,
    logits: Vec<f32>,
    draft_token: i32,
    verify_logits: Vec<f32>,
    /// First position whose trunk hidden row is kept.
    hidden_base: u32,
    draft_length: LengthController,
    stats: SpeculativeStats,
    /// The attached request's image prompt; it may append images after the
    /// computed context.
    image_prompt: Option<Arc<Prompt>>,
    /// The device argmax of `logits` from greedy verification, with the
    /// mutation epoch and token count it holds for.
    anchor_hint: Option<(u32, u64, usize)>,
    valid: bool,
    /// This request's draft vocabulary; the device table follows it.
    draft_vocab: DraftVocab,
    /// The token history was replaced: the table restarts from it.
    draft_vocab_stale: bool,
    /// The committed tokens indexed for prompt lookup.
    lookup: PromptLookup,
}

/// Captures idle sessions without the model lock, on a transfer stream of
/// its own; other sessions may run meanwhile.
pub struct Snapshotter {
    saver: crate::snapshot::Saver,
    vocab: u32,
    concurrency: u32,
}

/// A session's context (`SessionSnapshot`): the session header, tokens,
/// logits and length policy on the host, and the executor's state either
/// borrowed (frozen on the device, append-only rows left in the session) or
/// as host bytes. Its payload is what [`Model::restore_snapshot`] and the
/// disk take.
pub struct SessionSnapshot {
    head: Vec<u8>,
    executor: Part,
}

enum Part {
    Host(Vec<u8>),
    Borrowed(Handle),
}

/// One session's decode step in a batch (`DecodeRequest`).
pub struct DecodeRequest<'a> {
    pub session: &'a mut Session,
    pub max_tokens: usize,
    pub sampler: &'a mut SamplerState,
    pub stop_at_eos: bool,
}

/// One session's appended token in a batch (`AdvanceRequest`).
pub struct AdvanceRequest<'a> {
    pub session: &'a mut Session,
    pub token: i32,
}

/// A batch request's progress (`BatchOutcome`).
#[derive(Default)]
struct Outcome {
    completed: bool,
    error: String,
    result: DecodeResult,
}

/// The work of one draft-verify cycle between preparation and verification
/// (`PendingDecode`).
#[derive(Default)]
struct PendingDecode {
    chain: Vec<i32>,
    proposals: Vec<Proposal>,
    base: u32,
    speculative: bool,
    sampled: bool,
    gpu_greedy: bool,
    gpu_verification: bool,
    width: usize,
    draft_sampler: Option<SamplerState>,
    draft_rng: u64,
    candidates: MtpCandidates,
    draft: i32,
    /// This cycle drafts through the small head.
    draft_subset: bool,
    /// The length policy prices this sampled small-head cycle with its own
    /// cost curve.
    sampled_subset_cost: bool,
    /// The occupancy the length policy prices this cycle at; 0 keeps the
    /// configured concurrency.
    policy_concurrency: u32,
    /// Trailing chain tokens copied by prompt lookup (after the MTP drafts),
    /// the committed index they were copied from, and the copy budget,
    /// identical in isolated and batched execution and zero on a request's
    /// first speculative step.
    lookup: u32,
    lookup_start: usize,
    lookup_cap: usize,
}

/// The longest visible context of the fused WMMA attention kernel for this
/// geometry, 0 for another one (`WmmaCausalAttentionContextLimit`).
fn wmma_context_limit(c: &Config) -> u32 {
    const MAX_MASK_WORDS: u32 = 2048;
    const RATIO: u32 = 4;
    if (c.num_heads, c.num_kv_heads, c.head_dim, c.compress_ratio) == (24, 2, 256, RATIO) {
        MAX_MASK_WORDS * RATIO * 32
    } else {
        0
    }
}

fn open_all(paths: &[PathBuf]) -> Result<Vec<Arc<MappedFile>>, String> {
    paths
        .iter()
        .map(|p| MappedFile::open(p).map_err(|e| format!("{}: {e}", p.display())))
        .collect()
}

impl Model {
    pub fn load(path: &Path, options: ModelOptions) -> Result<Self, String> {
        if options.decode_concurrency == 0 || options.decode_concurrency > 8 {
            return Err("decode concurrency must be between one and eight".into());
        }
        if options.max_draft_tokens == 0 {
            return Err("draft token limit must be positive".into());
        }
        if options.mtp_model_path.is_some() && options.max_draft_tokens > MAX_DRAFT_TOKENS {
            return Err("Flash-Next MTP supports at most seven draft tokens".into());
        }
        let files = GgufFiles::open(path)?;
        let headers = files.parse()?;
        let mut tensors = ModelTensors::bind(&headers)?;
        if let Some(scaling) = options.rope_scaling {
            validate_rope_scaling(&scaling, tensors.config.context_length)?;
            tensors.config.rope_scaling = scaling;
        }
        let c = &tensors.config;
        if options.max_context == 0 || options.max_context > c.max_context_length() {
            return Err(format!(
                "context exceeds the model's {} tokens",
                c.max_context_length()
            ));
        }
        let sidecar = options.vision_model_path.clone().unwrap_or_default();
        let vision =
            Encoder::open(path, &sidecar, c.hidden_size).map_err(|e| e.message().to_string())?;
        let tokenizer = QwenTokenizer::from_gguf(&headers[0])?;
        let name = headers[0].string("general.name").map_or_else(
            || "Qwen3.8-Flash-Next".into(),
            |s| String::from_utf8_lossy(s).into_owned(),
        );
        let ngram = match (&tensors.ple_table, c.ple_layer) {
            (Some(t), Some(_)) => {
                let shard = std::fs::File::open(&files.paths[t.shard])
                    .map_err(|e| format!("{}: {e}", files.paths[t.shard].display()))?;
                Some(Arc::new(NgramTable::open(
                    &shard,
                    t.offset as u64,
                    t.rows,
                    c.ple_head_dim,
                    t.ty,
                )?))
            }
            _ => None,
        };
        let mtp = match &options.mtp_model_path {
            Some(p) => {
                let mtp_files = GgufFiles::open(p)?;
                let mtp = MtpTensors::bind(&mtp_files.parse()?, c)?;
                Some((mtp_files, mtp))
            }
            None => None,
        };
        let mapped = open_all(&files.paths)?;
        let stream = Stream::new().map_err(|e| format!("hipStreamCreate: {e}"))?;
        let device = match &mtp {
            Some((mtp_files, mtp)) => {
                let mtp_mapped = open_all(&mtp_files.paths)?;
                DeviceModel::upload(&stream, &tensors, &mapped, Some((mtp, &mtp_mapped)))?
            }
            None => DeviceModel::upload(&stream, &tensors, &mapped, None)?,
        };
        drop(stream);
        let mut device = device;
        let mut draft_specials = None;
        if mtp.is_some() && device.output.rows > DRAFT_VOCAB {
            let mut specials: Vec<i32> = tokenizer
                .special_ids()
                .filter(|&id| id >= DRAFT_VOCAB && id < device.output.rows)
                .map(|id| id as i32)
                .collect();
            specials.sort_unstable();
            specials.dedup();
            // A vocabulary with more special tokens than table rows drafts
            // over the full head.
            if specials.len() < DRAFT_EXTRA_ROWS as usize {
                device.set_draft_vocab(DRAFT_VOCAB, specials.clone())?;
                draft_specials = Some(specials);
            }
        }
        let max_batch = (PREFILL_CHUNK_TOKENS + PREFILL_TAIL_TOKENS).min(options.max_context);
        let max_logit_rows = if mtp.is_some() {
            max_batch.min(options.max_draft_tokens + 1)
        } else {
            1
        };
        let executor = Executor::new(
            Arc::new(device),
            ngram,
            Options {
                max_batch,
                max_logit_rows,
                max_speculative: max_logit_rows,
                max_context: options.max_context,
            },
        )?;
        let snapshots = Snapshotter {
            saver: executor.saver()?,
            vocab: executor.config().vocab_size,
            concurrency: options.decode_concurrency,
        };
        Ok(Self {
            options,
            snapshots,
            name,
            tokenizer,
            executor,
            batch_policy: BatchController::default(),
            vision,
            draft_specials,
        })
    }

    /// The vision encoder, when a sidecar was found.
    pub fn vision_encoder(&self) -> Option<&Arc<Encoder>> {
        self.vision.as_ref()
    }

    /// Installs a request's image prompt (`None` for text) on the session
    /// (`Session::ConfigureVision`): other images than the session consumed
    /// reset its context; images after the computed context may be new.
    pub fn configure_vision(
        &mut self,
        s: &mut Session,
        prompt: Option<Arc<Prompt>>,
    ) -> Result<(), String> {
        let count = s.tokens.len();
        let identity = prompt
            .as_ref()
            .map_or(&[][..], |p| p.identity_for_prefix(count));
        if count != 0 && identity != s.image_identity(count) {
            self.reset(s);
        }
        let was_valid = s.valid;
        s.valid = false;
        self.executor
            .configure_vision(&mut s.native, prompt.clone(), self.vision.clone())?;
        s.image_prompt = prompt;
        s.valid = was_valid;
        Ok(())
    }

    pub fn tokenize(&self, text: &str) -> Vec<i32> {
        self.tokenizer
            .encode(text)
            .into_iter()
            .map(|t| t as i32)
            .collect()
    }

    pub fn decode(&self, tokens: &[i32]) -> Vec<u8> {
        let ids: Vec<TokenId> = tokens.iter().map(|&t| t as TokenId).collect();
        self.tokenizer.decode(&ids)
    }

    pub fn tokenizer(&self) -> &QwenTokenizer {
        &self.tokenizer
    }

    /// `Model::IsStopToken`: the tokenizer's EOS or `<|endoftext|>`.
    pub fn is_stop_token(&self, token: i32) -> bool {
        let stops = self.tokenizer.stop_tokens();
        u32::try_from(token).is_ok_and(|token| stops.contains(token))
    }

    pub fn vocab_size(&self) -> u32 {
        self.executor.config().vocab_size
    }

    pub fn prefill_capacity(&self) -> u32 {
        PREFILL_CHUNK_TOKENS.min(self.options.max_context)
    }

    /// The widest prefill pass: a remainder up to this long finishes in one
    /// pass (`PrefillThroughCapacity`).
    pub fn prefill_through_capacity(&self) -> u32 {
        (PREFILL_CHUNK_TOKENS + PREFILL_TAIL_TOKENS).min(self.options.max_context)
    }

    pub fn max_context(&self) -> u32 {
        self.options.max_context
    }

    pub fn has_mtp(&self) -> bool {
        self.executor.has_mtp()
    }

    pub fn decode_concurrency(&self) -> u32 {
        self.options.decode_concurrency
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn config(&self) -> &Config {
        self.executor.config()
    }

    pub fn executor(&mut self) -> &mut Executor {
        &mut self.executor
    }

    pub fn resident_bytes(&self) -> usize {
        self.executor.resident_bytes()
    }

    /// Worst-case private device state, the configured rollback cap
    /// included.
    pub fn session_bytes(&self, mode: SessionMode, context: u32) -> usize {
        let depth = if mode == SessionMode::Speculative {
            self.executor.options().max_speculative - 1
        } else {
            0
        };
        self.executor.session_bytes(mode, context, depth)
    }
}
