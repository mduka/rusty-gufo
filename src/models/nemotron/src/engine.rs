//! One resident model and its sessions (`engine.cpp`): prefill in chunks,
//! decoding, draft verification and commit, batched decoding and host
//! snapshots.
//!
//! Unlike the C++, whose sessions each own a stream, the model owns one
//! stream and hipBLAS handle that every session's work runs on, under a
//! lock: device work is serialized by the serving runtime anyway, and one
//! stream lets a batched decode write the state of several sessions. This
//! port launches the forward pass directly: the C++ captures the one-token
//! decode and the batched decode in HIP graphs and runs the shared expert of
//! short chunks on a side stream. Both are optimizations with the same
//! arithmetic; the shared expert runs first on the main stream here, which
//! accumulates into the residual stream in the same order.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use gufo_hip::blas::{Blas, Op};
use gufo_hip::{
    Bf16, DeviceBuffer, DeviceCopy, DeviceIndices, DeviceSlice, DeviceSliceMut, Event, F16,
    HipError, Launch, LaunchError, MappedFile, Stream,
};
use gufo_nemotron_kernels as k;
use gufo_nemotron_kernels::sizes;

use crate::config::{BlockType, Config};
use crate::device_model::{DeviceLayer, DeviceModel, DeviceWeight};
use crate::drafter::{DraftContext, Drafter};
use crate::logit_head::{LogitHead, LogitHeadFilter};
use crate::tokenizer::Tokenizer;
use crate::weights::{GgufShard, ModelWeights};
use gufo_core::sampling::LogitCandidate;

/// Tokens of one prefill pass: `append` runs longer inputs in passes of this
/// many, and continuing from a multiple of it repeats an uninterrupted
/// prefill's arithmetic.
pub const PREFILL_CHUNK: usize = 2048;
/// Sessions one batched decode takes, and tokens one `verify` takes.
pub const MAX_BATCH: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelOptions {
    /// Longest session the model accepts, zero for the native context; each
    /// session allocates its own KV cache up to its requested context.
    pub max_context: u32,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self { max_context: 32768 }
    }
}

/// Input rows that replace token embeddings: device rows of the trunk's
/// width, float32, for the positions from `position` on (image embeddings
/// at their placeholder tokens).
#[derive(Clone, Copy)]
pub struct InputRows<'a> {
    pub position: u32,
    pub rows: DeviceSlice<'a, f32>,
}

/// Host copy of a session at one position: K and V rows up to the position
/// for each attention block, and the conv and SSM states of each Mamba
/// block.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionSnapshot {
    pub position: u32,
    /// Per attention block: K then V, FP16 bits.
    pub kv: Vec<Vec<u16>>,
    /// Per Mamba block: conv then SSM state.
    pub recurrent: Vec<Vec<f32>>,
    /// Drafter context, when it has one.
    pub draft: Vec<u8>,
}

impl SessionSnapshot {
    pub fn bytes(&self) -> usize {
        self.draft.len()
            + self.kv.iter().map(|rows| rows.len() * 2).sum::<usize>()
            + self
                .recurrent
                .iter()
                .map(|state| state.len() * 4)
                .sum::<usize>()
    }
}

/// One resident model on the GPU; any number of sessions with independent
/// KV caches and Mamba states.
pub struct Model {
    options: ModelOptions,
    name: String,
    files: Vec<Arc<MappedFile>>,
    weights: ModelWeights,
    tokenizer: Tokenizer,
    device: DeviceModel,
    drafter: Option<Drafter>,
    exec: Mutex<Exec>,
}

/// The stream and hipBLAS handle every session of a model runs on, the
/// side stream of the shared experts, and the scratch of batched decoding.
struct Exec {
    stream: Stream,
    side: Side,
    blas: Blas,
    batch: Option<Batch>,
}

/// A second stream and the events that fork onto it and join it again
/// (`Session::State`'s side, fork and join).
struct Side {
    stream: Stream,
    start: Event,
    done: Event,
}

impl Drop for Exec {
    fn drop(&mut self) {
        let _ = self.side.stream.synchronize();
        let _ = self.stream.synchronize();
        let _ = k::mmq_release_stream(&self.stream);
    }
}

/// The scratch of batched decoding: chunk buffers of `MAX_BATCH` rows, the
/// device table the row kernels read the sessions' buffers through, and the
/// rows' positions by their bound.
struct Batch {
    st: State,
    tables: DeviceBuffer<u8>,
    positions: HashMap<u32, DeviceIndices>,
}

/// The sessions of a batched decode as the forward pass reads them.
struct BatchRows<'a, 'b> {
    sessions: &'a mut [&'b mut Session],
    tables: DeviceSliceMut<'a, u8>,
    positions: gufo_hip::IndexSlice<'a>,
    /// The smallest context of the rows.
    capacity: usize,
}

fn hip(error: HipError) -> String {
    error.to_string()
}

fn launch(error: LaunchError) -> String {
    error.to_string()
}

impl Model {
    /// Maps and binds the trunk GGUF and uploads it.
    pub fn load(model_path: &Path, options: ModelOptions) -> Result<Model, String> {
        let name = model_path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_default();
        let file = MappedFile::open(model_path)
            .map_err(|e| format!("open {}: {e}", model_path.display()))?;
        let files = vec![file];
        let (weights, tokenizer) = {
            let shard = GgufShard::parse(files[0].bytes())
                .map_err(|e| format!("open {}: {e}", model_path.display()))?;
            let shards = [shard];
            let weights = ModelWeights::bind(&shards)?;
            let tokenizer = Tokenizer::from_gguf(&shards[0].gguf)?;
            (weights, tokenizer)
        };
        let mut options = options;
        if options.max_context == 0 {
            options.max_context = weights.config.context_length;
        } else if options.max_context > weights.config.context_length {
            return Err("max_context exceeds the native context".into());
        }
        let stream = Stream::new().map_err(hip)?;
        // The C++ session handle keeps hipBLAS's default atomics mode, which
        // decides the kernels of the router and drafter products.
        let blas = Blas::with_default_atomics(&stream).map_err(hip)?;
        supported(
            k::mmq_bind_stream(&stream),
            "quantized matrix kernels initialization",
        )?;
        let device = DeviceModel::upload(&weights, &files, &stream)?;
        stream.synchronize().map_err(hip)?;
        Ok(Model {
            options,
            name,
            files,
            weights,
            tokenizer,
            device,
            drafter: None,
            exec: Mutex::new(Exec {
                stream,
                side: Side {
                    stream: Stream::new().map_err(hip)?,
                    start: Event::new().map_err(hip)?,
                    done: Event::new().map_err(hip)?,
                },
                blas,
                batch: None,
            }),
        })
    }

    /// Loads NVIDIA's LiLiCorr DFlash drafter for this trunk. Sessions
    /// created afterwards keep its context and can draft.
    pub fn load_drafter(&mut self, path: &Path) -> Result<(), String> {
        self.drafter = Some(Drafter::load(path, &self.weights.config)?);
        // A batch scratch built without the drafter captures no features.
        self.exec().batch = None;
        Ok(())
    }

    fn exec(&self) -> MutexGuard<'_, Exec> {
        self.exec.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn config(&self) -> &Config {
        &self.weights.config
    }

    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }

    /// The mapped trunk files, for identities and the CPU reference.
    pub fn files(&self) -> &[Arc<MappedFile>] {
        &self.files
    }

    pub fn max_context(&self) -> u32 {
        self.options.max_context
    }

    pub fn resident_bytes(&self) -> usize {
        self.device.resident_bytes
    }

    pub fn device(&self) -> &DeviceModel {
        &self.device
    }

    pub fn model_name(&self) -> &str {
        &self.name
    }

    pub fn drafter(&self) -> Option<&Drafter> {
        self.drafter.as_ref()
    }

    /// Device bytes one session of `context` tokens allocates for its
    /// caches and states.
    pub fn session_bytes(&self, context: u32) -> usize {
        let c = &self.weights.config;
        let mut bytes = 0usize;
        for block in &c.blocks {
            match block {
                BlockType::Attention => {
                    bytes += 2 * context as usize * c.attention_kv_dim() as usize * 2;
                }
                BlockType::Mamba => {
                    bytes += 4
                        * ((c.ssm_conv_kernel as usize - 1) * c.ssm_conv_channels() as usize
                            + c.ssm_heads as usize
                                * c.ssm_head_dim() as usize
                                * c.ssm_state_size as usize);
                }
                BlockType::Moe => {}
            }
        }
        bytes
    }

    /// A new session of up to `max_context` tokens, reset to position zero.
    pub fn create_session(self: &Arc<Self>, max_context: u32) -> Result<Session, String> {
        if max_context == 0 || max_context > self.options.max_context {
            return Err("session context exceeds the model's configured maximum".into());
        }
        let state = {
            let exec = self.exec();
            State::new(
                self,
                &exec.stream,
                &exec.blas,
                Some(max_context),
                PREFILL_CHUNK,
            )?
        };
        let mut session = Session {
            model: Arc::clone(self),
            st: state,
            position: 0,
            max_context,
            verified: 0,
            needs_reset: false,
        };
        session.reset()?;
        Ok(session)
    }
}

/// Allocates `n` elements (at least one) on `stream`.
fn alloc<T: DeviceCopy>(stream: &Stream, n: usize) -> Result<DeviceBuffer<T>, String> {
    stream
        .alloc::<T>(n.max(1))
        .map_err(|_| "could not allocate session memory".to_string())
}

/// The routing of a short chunk's expert slots by expert
/// (`iq2xxs_route_short`): slot order, tiles and tile count.
struct ShortRoute {
    order: DeviceIndices,
    tiles: DeviceIndices,
    num_tiles: DeviceIndices,
}

/// Device buffers of a session (`Session::State`).
struct State {
    // Per block; `None` for blocks of the other kinds.
    k_cache: Vec<Option<DeviceBuffer<F16>>>,
    v_cache: Vec<Option<DeviceBuffer<F16>>>,
    conv_state: Vec<Option<DeviceBuffer<f32>>>,
    ssm_state: Vec<Option<DeviceBuffer<f32>>>,
    // Chunk scratch.
    tokens: DeviceIndices,
    x: DeviceBuffer<f32>,
    h: DeviceBuffer<f32>,
    proj: DeviceBuffer<f32>,
    xbc: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    ctx: DeviceBuffer<f32>,
    /// Prefill inputs of the NVFP4 Mamba projections, as FP16.
    h16: DeviceBuffer<F16>,
    y16: DeviceBuffer<F16>,
    router: DeviceBuffer<f32>,
    route_weights: DeviceBuffer<f32>,
    latent: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    acc: DeviceBuffer<f32>,
    shared_up: DeviceBuffer<f32>,
    shared: DeviceBuffer<f32>,
    /// BF16 activations of the expert products of chunks of 2 to
    /// GEMV_MAX_TOKENS tokens: h and the latent rows, and the up products'
    /// outputs, which feed the down products.
    h_bf16: DeviceBuffer<Bf16>,
    latent_bf16: DeviceBuffer<Bf16>,
    shared_up_bf16: DeviceBuffer<Bf16>,
    up_bf16: DeviceBuffer<Bf16>,
    /// Routed expert IDs, by expert count (their index bound).
    ids: HashMap<usize, DeviceIndices>,
    /// Short routings, by slot and expert count (their index bounds).
    short_routes: HashMap<(usize, usize), ShortRoute>,
    route_cursor: DeviceBuffer<i32>,
    /// Q8_1 activations of short chunks for the vector products.
    xq: DeviceBuffer<u8>,
    /// The chunk's first position, by token count (index bounds depend on
    /// it); short chunks keep theirs.
    positions: HashMap<usize, DeviceIndices>,
    attention_partial: DeviceBuffer<f32>,
    /// One row: the frontier after the last token.
    last_logits: DeviceBuffer<f32>,
    /// One row per token when requested.
    logits: DeviceBuffer<f32>,
    head_scratch: DeviceBuffer<u8>,
    head_records: DeviceBuffer<u8>,
    head_candidates: DeviceBuffer<u8>,
    host_head_records: Vec<u8>,
    host_head_candidates: Vec<u8>,
    // Drafter: trunk features of the chunk (feature_width floats per token)
    // at the blocks in feature_slot.
    draft: Option<DraftContext>,
    features: DeviceBuffer<f32>,
    feature_slot: Vec<Option<usize>>,
    feature_width: usize,
    /// Leading chunk rows the drafter will never read.
    inject_skip: usize,
    /// Per Mamba block, the chunk's ssm_in rows and convolved x|B|C of a
    /// verification, so `commit` can replay the accepted prefix.
    verify_proj: Vec<Option<DeviceBuffer<f32>>>,
    verify_xbc: Vec<Option<DeviceBuffer<f32>>>,
    /// Rows of the current chunk that replace its token embeddings: chunk
    /// row and the rows' source range in the append's InputRows.
    input_rows: Vec<(usize, usize, usize, usize)>,
}

impl State {
    /// A session's state with caches for `max_context` tokens, or the
    /// scratch of batched decoding without caches (`None`), for chunks of
    /// `chunk` tokens.
    fn new(
        model: &Model,
        stream: &Stream,
        blas: &Blas,
        max_context: Option<u32>,
        chunk: usize,
    ) -> Result<Self, String> {
        let c = &model.weights.config;
        let device = &model.device;
        let n = c.num_layers as usize;
        let mut k_cache = Vec::with_capacity(n);
        let mut v_cache = Vec::with_capacity(n);
        let mut conv_state = Vec::with_capacity(n);
        let mut ssm_state = Vec::with_capacity(n);
        let kv = max_context.unwrap_or(0) as usize * c.attention_kv_dim() as usize;
        for &block in &c.blocks {
            let attention = block == BlockType::Attention && max_context.is_some();
            let mamba = block == BlockType::Mamba && max_context.is_some();
            k_cache.push(if attention {
                Some(alloc(stream, kv)?)
            } else {
                None
            });
            v_cache.push(if attention {
                Some(alloc(stream, kv)?)
            } else {
                None
            });
            conv_state.push(if mamba {
                Some(alloc(
                    stream,
                    (c.ssm_conv_kernel as usize - 1) * c.ssm_conv_channels() as usize,
                )?)
            } else {
                None
            });
            ssm_state.push(if mamba {
                Some(alloc(
                    stream,
                    c.ssm_heads as usize * c.ssm_head_dim() as usize * c.ssm_state_size as usize,
                )?)
            } else {
                None
            });
        }
        let t = chunk;
        let g = t.min(sizes::GEMV_MAX_TOKENS);
        let h = c.hidden_size as usize;
        let mut max_ff = 0usize;
        let mut max_experts = 0usize;
        let mut max_used = 0usize;
        for (il, layer) in device.layers.iter().enumerate() {
            if c.blocks[il] != BlockType::Moe {
                continue;
            }
            max_ff = max_ff.max(c.expert_ff[il] as usize);
            max_experts = max_experts.max(layer.num_experts);
        }
        for &used in &c.experts_used {
            max_used = max_used.max(used as usize);
        }
        let vocab = c.vocab_size as usize;
        let vocab_bound = u32::try_from(vocab).map_err(|_| "vocabulary too large".to_string())?;
        let head_rows = MAX_BATCH;
        let _ = blas;
        let mut st = State {
            tokens: stream.alloc_indices(t, vocab_bound).map_err(hip)?,
            x: alloc(stream, t * h)?,
            h: alloc(stream, t * h)?,
            proj: alloc(stream, t * c.ssm_in_projection() as usize)?,
            xbc: alloc(stream, t * c.ssm_conv_channels() as usize)?,
            y: alloc(stream, t * c.ssm_inner_size as usize)?,
            out: alloc(stream, t * h)?,
            q: alloc(stream, t * c.attention_q_dim() as usize)?,
            k: alloc(stream, t * c.attention_kv_dim() as usize)?,
            v: alloc(stream, t * c.attention_kv_dim() as usize)?,
            ctx: alloc(stream, t * c.attention_q_dim() as usize)?,
            h16: alloc(stream, t * h)?,
            y16: alloc(stream, t * c.ssm_inner_size as usize)?,
            router: alloc(stream, t * max_experts)?,
            route_weights: alloc(stream, t * max_used)?,
            latent: alloc(stream, t * c.latent_size as usize)?,
            up: alloc(stream, t * max_used * max_ff)?,
            down: alloc(stream, t * max_used * c.latent_size as usize)?,
            acc: alloc(stream, t * c.latent_size as usize)?,
            shared_up: alloc(stream, t * c.shared_expert_ff as usize)?,
            shared: alloc(stream, t * h)?,
            h_bf16: alloc(stream, g * h)?,
            latent_bf16: alloc(stream, g * c.latent_size as usize)?,
            shared_up_bf16: alloc(stream, g * c.shared_expert_ff as usize)?,
            up_bf16: alloc(stream, g * max_used * max_ff)?,
            ids: HashMap::new(),
            short_routes: HashMap::new(),
            route_cursor: alloc(stream, max_experts)?,
            xq: alloc(
                stream,
                sizes::GEMV_MAX_TOKENS * h.max(c.ssm_inner_size as usize) / 32
                    * sizes::Q8_1_BLOCK_BYTES,
            )?,
            positions: HashMap::new(),
            attention_partial: alloc(
                stream,
                sizes::attention_short_partial(sizes::ATTENTION_SHORT_TOKENS, c.num_heads as usize),
            )?,
            last_logits: alloc(stream, vocab)?,
            logits: alloc(stream, 1)?,
            head_scratch: alloc(stream, sizes::logit_head_scratch_bytes(head_rows, vocab))?,
            head_records: alloc(stream, head_rows * sizes::LOGIT_HEAD_RECORD_BYTES)?,
            head_candidates: alloc(
                stream,
                head_rows * sizes::LOGIT_HEAD_CANDIDATES * sizes::LOGIT_HEAD_CANDIDATE_BYTES,
            )?,
            host_head_records: vec![0; head_rows * sizes::LOGIT_HEAD_RECORD_BYTES],
            host_head_candidates: vec![
                0;
                head_rows
                    * sizes::LOGIT_HEAD_CANDIDATES
                    * sizes::LOGIT_HEAD_CANDIDATE_BYTES
            ],
            draft: None,
            features: alloc(stream, 1)?,
            feature_slot: Vec::new(),
            feature_width: 0,
            inject_skip: 0,
            verify_proj: Vec::with_capacity(n),
            verify_xbc: Vec::with_capacity(n),
            input_rows: Vec::new(),
            k_cache,
            v_cache,
            conv_state,
            ssm_state,
        };
        if let Some(drafter) = model.drafter.as_ref() {
            let dc = drafter.config();
            if max_context.is_some() {
                st.draft = Some(DraftContext::create(stream, blas, drafter, device, t)?);
            }
            st.feature_slot = vec![None; n];
            for (i, &layer) in dc.target_layers.iter().enumerate() {
                if let Some(slot) = st.feature_slot.get_mut(layer as usize) {
                    *slot = Some(i);
                }
            }
            st.feature_width = dc.feature_size() as usize;
            st.features = stream
                .alloc(t * st.feature_width)
                .map_err(|_| "could not allocate drafter feature memory".to_string())?;
        }
        for &block in &c.blocks {
            let mamba = block == BlockType::Mamba && max_context.is_some();
            let proj = if mamba {
                Some(
                    stream
                        .alloc(MAX_BATCH * c.ssm_in_projection() as usize)
                        .map_err(|_| "could not allocate verification memory".to_string())?,
                )
            } else {
                None
            };
            let xbc = if mamba {
                Some(
                    stream
                        .alloc(MAX_BATCH * c.ssm_conv_channels() as usize)
                        .map_err(|_| "could not allocate verification memory".to_string())?,
                )
            } else {
                None
            };
            st.verify_proj.push(proj);
            st.verify_xbc.push(xbc);
        }
        Ok(st)
    }

    /// The device position buffer for chunks of `n` tokens of a session of
    /// `capacity`, holding `position`.
    fn upload_position(
        &mut self,
        stream: &Stream,
        n: usize,
        capacity: usize,
        position: u32,
    ) -> Result<(), String> {
        let bound = u32::try_from(capacity - n + 1).map_err(|_| "context too large".to_string())?;
        if let Entry::Vacant(slot) = self.positions.entry(n) {
            slot.insert(stream.alloc_indices(1, bound).map_err(hip)?);
        }
        let indices = self
            .positions
            .get_mut(&n)
            .ok_or_else(|| "could not upload tokens".to_string())?;
        stream
            .upload_indices(&[position], indices)
            .map_err(|_| "could not upload tokens".to_string())
    }
}

/// Fails a launch that its adapter refused.
fn supported(result: Result<Launch, LaunchError>, what: &str) -> Result<(), String> {
    match result.map_err(launch)? {
        Launch::Launched => Ok(()),
        Launch::Unsupported => Err(format!("{what} failed")),
    }
}

fn mmq(result: Result<Launch, LaunchError>, what: &str) -> Result<(), String> {
    match result.map_err(launch)? {
        Launch::Launched => Ok(()),
        Launch::Unsupported => Err(format!("quantized matrix product failed: {what}")),
    }
}

fn f32s(w: &Option<DeviceWeight>) -> Result<DeviceSlice<'_, f32>, String> {
    w.as_ref()
        .and_then(DeviceWeight::f32)
        .ok_or_else(|| "a block lacks an F32 tensor".to_string())
}

fn weight(w: &Option<DeviceWeight>) -> Result<&DeviceWeight, String> {
    w.as_ref()
        .ok_or_else(|| "a block lacks a tensor".to_string())
}

/// `out[n][rows] = W x` over F32 rows (`nb_mmq_dense`).
fn dense(
    s: &Stream,
    w: &DeviceWeight,
    x: DeviceSlice<'_, f32>,
    out: DeviceSliceMut<'_, f32>,
    n: usize,
    what: &str,
) -> Result<(), String> {
    use gufo_hip::GgmlType as T;
    let r = match w.ty {
        T::Q8_0 => k::mmq_dense_q8_0(s, w.bytes(), x, out, w.rows, n, w.cols),
        T::Iq2Xxs => k::mmq_dense_iq2_xxs(s, w.bytes(), x, out, w.rows, n, w.cols),
        T::Nvfp4 => k::mmq_dense_nvfp4(s, w.bytes(), x, out, w.rows, n, w.cols),
        _ => return Err(format!("quantized matrix product failed: {what}")),
    };
    mmq(r, what)
}

/// The short-chunk product on the Q8_1 rows (`nb_mmq_dense_q8_1`, scale 1).
fn q8_dense(
    s: &Stream,
    w: &DeviceWeight,
    xq: DeviceSlice<'_, u8>,
    out: DeviceSliceMut<'_, f32>,
    n: usize,
    what: &str,
) -> Result<(), String> {
    use gufo_hip::GgmlType as T;
    let r = match w.ty {
        T::Q8_0 => k::mmq_dense_q8_1_q8_0(s, w.bytes(), xq, out, w.rows, n, w.cols),
        T::Iq2Xxs => k::mmq_dense_q8_1_iq2_xxs(s, w.bytes(), xq, out, w.rows, n, w.cols),
        T::Nvfp4 => k::mmq_dense_q8_1_nvfp4(s, w.bytes(), xq, out, w.rows, n, w.cols),
        _ => return Err(format!("quantized matrix product failed: {what}")),
    };
    mmq(r, what)
}

/// One token: `out = scale W xq + out`, fused into the product.
fn q8_residual_one(
    s: &Stream,
    w: &DeviceWeight,
    xq: DeviceSlice<'_, u8>,
    scale: f32,
    out: DeviceSliceMut<'_, f32>,
    what: &str,
) -> Result<(), String> {
    use gufo_hip::GgmlType as T;
    let r = match w.ty {
        T::Q8_0 => k::mmq_residual_q8_0(s, w.bytes(), xq, scale, out, w.rows, w.cols),
        T::Iq2Xxs => k::mmq_residual_iq2_xxs(s, w.bytes(), xq, scale, out, w.rows, w.cols),
        T::Nvfp4 => k::mmq_residual_nvfp4(s, w.bytes(), xq, scale, out, w.rows, w.cols),
        _ => return Err(format!("quantized matrix product failed: {what}")),
    };
    mmq(r, what)
}

/// Runs one chunk of at most `PREFILL_CHUNK` tokens through the trunk
/// (`Forward`).
struct Forward<'a> {
    c: &'a Config,
    m: &'a DeviceModel,
    s: &'a Stream,
    /// The shared experts of short chunks run here beside the router and
    /// the routed experts.
    side: &'a Side,
    blas: &'a Blas,
    /// Keeps the Mamba states and records their inputs instead.
    verify: bool,
    capacity: usize,
}

impl Forward<'_> {
    /// Enqueues the chunk; `head_rows` final rows (0, 1 or n) go through the
    /// output head into `logits` (or the per-token row buffer). With a
    /// drafter, the chunk's features from `st.inject_skip` on go into its
    /// context, at `first` + `inject_skip`.
    #[allow(clippy::too_many_arguments)]
    fn enqueue(
        &self,
        st: &mut State,
        n: usize,
        head_rows: usize,
        per_token_logits: bool,
        rows: &[InputRows<'_>],
        first: u32,
        mut batch: Option<&mut BatchRows<'_, '_>>,
    ) -> Result<(), String> {
        let c = self.c;
        let hd = c.hidden_size as usize;
        let features = !st.feature_slot.is_empty() && st.inject_skip < n;
        supported(
            k::embed_q8_0(
                self.s,
                self.m.token_embd.bytes(),
                st.tokens.slice(0..n),
                st.x.slice_mut(0..n * hd),
                n,
                hd,
                c.vocab_size as usize,
            ),
            "embedding",
        )?;
        for &(row, count, source, from) in &st.input_rows {
            let rows = rows[source].rows;
            self.s
                .copy(
                    rows.range(from * hd..(from + count) * hd),
                    st.x.slice_mut(row * hd..(row + count) * hd),
                )
                .map_err(hip)?;
        }
        for (il, layer) in self.m.layers.iter().enumerate() {
            if features && let Some(slot) = st.feature_slot[il] {
                // The chunk's residual stream, the input of a target block,
                // into its feature column.
                let width = st.feature_width;
                self.s
                    .copy_2d(
                        st.x.slice(0..n * hd),
                        hd,
                        st.features.slice_mut(slot * hd..n * width),
                        width,
                        hd,
                        n,
                    )
                    .map_err(hip)?;
            }
            let norm = f32s(&layer.norm)?;
            let ty = layer.ty.ok_or("a block has no kind")?;
            if n <= sizes::GEMV_MAX_TOKENS {
                // Short chunks: the vector products read the normalized rows
                // as Q8_1; the router and the shared expert read them as F32.
                let h = (ty == BlockType::Moe).then(|| st.h.slice_mut(0..n * hd));
                supported(
                    k::rms_norm_q8(
                        self.s,
                        st.x.slice(0..n * hd),
                        norm,
                        h,
                        Some(st.xq.as_mut_slice()),
                        n,
                        hd,
                        c.rms_eps,
                    ),
                    "rms_norm",
                )?;
            } else if ty == BlockType::Mamba {
                // Only the FP16 projection reads a Mamba block's rows.
                supported(
                    k::rms_norm_f16(
                        self.s,
                        st.x.slice(0..n * hd),
                        norm,
                        st.h16.slice_mut(0..n * hd),
                        n,
                        hd,
                        c.rms_eps,
                    ),
                    "rms_norm",
                )?;
            } else {
                supported(
                    k::rms_norm(
                        self.s,
                        st.x.slice(0..n * hd),
                        norm,
                        st.h.slice_mut(0..n * hd),
                        n,
                        hd,
                        c.rms_eps,
                    ),
                    "rms_norm",
                )?;
            }
            match ty {
                BlockType::Mamba => self.mamba(st, layer, il, n, batch.as_deref_mut())?,
                BlockType::Attention => self.attention(st, layer, il, n, batch.as_deref_mut())?,
                BlockType::Moe => self.moe(st, layer, n)?,
            }
        }
        if features && let Some(draft) = st.draft.as_mut() {
            let skip = st.inject_skip;
            let width = st.feature_width;
            draft.inject(
                self.s,
                self.blas,
                st.features.slice(skip * width..n * width),
                first + skip as u32,
                n - skip,
            )?;
        }
        if head_rows > 0 {
            self.head(st, n, head_rows, per_token_logits)?;
        }
        Ok(())
    }

    /// `x += scale W xq` for `n` short rows: fused for one token, a separate
    /// add for more. The scale stays out of the activations, whose Q8_1
    /// block scales are FP16.
    fn q8_residual(
        &self,
        st: &mut State,
        w: &DeviceWeight,
        scale: f32,
        n: usize,
        what: &str,
    ) -> Result<(), String> {
        let hd = self.c.hidden_size as usize;
        if n == 1 {
            return q8_residual_one(
                self.s,
                w,
                st.xq.as_slice(),
                scale,
                st.x.slice_mut(0..hd),
                what,
            );
        }
        q8_dense(
            self.s,
            w,
            st.xq.as_slice(),
            st.out.slice_mut(0..n * w.rows),
            n,
            what,
        )?;
        supported(
            k::add_scaled(
                self.s,
                st.x.slice_mut(0..n * hd),
                st.out.slice(0..n * hd),
                scale,
                None,
                n * hd,
            ),
            "add_scaled",
        )
    }

    fn mamba(
        &self,
        st: &mut State,
        l: &DeviceLayer,
        il: usize,
        n: usize,
        batch: Option<&mut BatchRows<'_, '_>>,
    ) -> Result<(), String> {
        let c = self.c;
        let stride = c.ssm_in_projection() as usize;
        let inner = c.ssm_inner_size as usize;
        let channels = c.ssm_conv_channels() as usize;
        let kernel = c.ssm_conv_kernel as usize;
        let ssm_in = weight(&l.ssm_in)?;
        let ssm_out = weight(&l.ssm_out)?;
        let scale = ssm_in.scale;
        let short = n <= sizes::GEMV_MAX_TOKENS;
        let hd = c.hidden_size as usize;
        let s = self.s;
        let (proj, xbc) = if self.verify {
            (
                st.verify_proj[il]
                    .as_mut()
                    .ok_or("no verification memory")?,
                st.verify_xbc[il].as_mut().ok_or("no verification memory")?,
            )
        } else {
            (&mut st.proj, &mut st.xbc)
        };
        if short {
            q8_dense(
                s,
                ssm_in,
                st.xq.as_slice(),
                proj.slice_mut(0..n * stride),
                n,
                "ssm_in",
            )?;
        } else {
            supported(
                k::nvfp4_gemm(
                    s,
                    ssm_in.bytes(),
                    st.h16.slice(0..n * hd),
                    proj.slice_mut(0..n * stride),
                    ssm_in.rows,
                    ssm_in.cols,
                    n,
                ),
                "nvfp4_gemm",
            )?;
        }
        if let Some(rows) = batch {
            // Each row's conv and SSM state, read and advanced in place.
            let mut states = Vec::with_capacity(n);
            for session in rows.sessions.iter_mut() {
                let state = session.st.conv_state[il].as_mut().ok_or("no conv state")?;
                states.push(state.as_mut_slice());
            }
            supported(
                k::mamba_conv_rows(
                    s,
                    proj.slice(0..n * stride),
                    stride,
                    inner,
                    scale,
                    f32s(&l.ssm_conv1d)?,
                    f32s(&l.ssm_conv1d_b)?,
                    &mut states,
                    rows.tables.reborrow(),
                    xbc.slice_mut(0..n * channels),
                    n,
                    channels,
                    kernel,
                ),
                "mamba_conv",
            )?;
            drop(states);
            let mut states = Vec::with_capacity(n);
            for session in rows.sessions.iter_mut() {
                let state = session.st.ssm_state[il].as_mut().ok_or("no SSM state")?;
                states.push(state.as_mut_slice());
            }
            supported(
                k::mamba_scan_rows(
                    s,
                    xbc.slice(0..n * channels),
                    channels,
                    proj.slice(0..n * stride),
                    stride,
                    inner + channels,
                    scale,
                    f32s(&l.ssm_dt_b)?,
                    f32s(&l.ssm_a)?,
                    f32s(&l.ssm_d)?,
                    &mut states,
                    rows.tables.reborrow(),
                    st.y.slice_mut(0..n * inner),
                    n,
                    c.ssm_heads as usize,
                    c.ssm_head_dim() as usize,
                    c.ssm_groups as usize,
                ),
                "mamba_scan",
            )?;
        } else {
            let conv_state = st.conv_state[il].as_mut().ok_or("no conv state")?;
            supported(
                k::mamba_conv(
                    s,
                    proj.slice(0..n * stride),
                    stride,
                    inner,
                    scale,
                    f32s(&l.ssm_conv1d)?,
                    f32s(&l.ssm_conv1d_b)?,
                    conv_state.as_slice(),
                    xbc.slice_mut(0..n * channels),
                    n,
                    channels,
                    kernel,
                ),
                "mamba_conv",
            )?;
            if !self.verify {
                supported(
                    k::mamba_conv_state(
                        s,
                        proj.slice(0..n * stride),
                        stride,
                        inner,
                        scale,
                        conv_state.as_mut_slice(),
                        n,
                        channels,
                        kernel,
                    ),
                    "mamba_conv_state",
                )?;
            }
            let ssm_state = st.ssm_state[il].as_mut().ok_or("no SSM state")?;
            supported(
                k::mamba_scan(
                    s,
                    xbc.slice(0..n * channels),
                    channels,
                    proj.slice(0..n * stride),
                    stride,
                    inner + channels,
                    scale,
                    f32s(&l.ssm_dt_b)?,
                    f32s(&l.ssm_a)?,
                    f32s(&l.ssm_d)?,
                    ssm_state.as_mut_slice(),
                    st.y.slice_mut(0..n * inner),
                    n,
                    c.ssm_heads as usize,
                    c.ssm_head_dim() as usize,
                    c.ssm_groups as usize,
                    !self.verify,
                ),
                "mamba_scan",
            )?;
        }
        if short {
            supported(
                k::mamba_gate_norm_q8(
                    s,
                    st.y.slice(0..n * inner),
                    proj.slice(0..n * stride),
                    stride,
                    scale,
                    f32s(&l.ssm_norm)?,
                    st.xq.as_mut_slice(),
                    n,
                    inner,
                    c.ssm_groups as usize,
                    c.rms_eps,
                ),
                "mamba_gate_norm",
            )?;
            return self.q8_residual(st, ssm_out, ssm_out.scale, n, "ssm_out");
        }
        supported(
            k::mamba_gate_norm_f16(
                s,
                st.y.slice_mut(0..n * inner),
                proj.slice(0..n * stride),
                stride,
                scale,
                f32s(&l.ssm_norm)?,
                st.y16.slice_mut(0..n * inner),
                n,
                inner,
                c.ssm_groups as usize,
                c.rms_eps,
            ),
            "mamba_gate_norm",
        )?;
        supported(
            k::nvfp4_gemm(
                s,
                ssm_out.bytes(),
                st.y16.slice(0..n * inner),
                st.out.slice_mut(0..n * hd),
                ssm_out.rows,
                ssm_out.cols,
                n,
            ),
            "nvfp4_gemm",
        )?;
        supported(
            k::add_scaled(
                s,
                st.x.slice_mut(0..n * hd),
                st.out.slice(0..n * hd),
                ssm_out.scale,
                None,
                n * hd,
            ),
            "add_scaled",
        )
    }

    fn attention(
        &self,
        st: &mut State,
        l: &DeviceLayer,
        il: usize,
        n: usize,
        batch: Option<&mut BatchRows<'_, '_>>,
    ) -> Result<(), String> {
        let c = self.c;
        let hd = c.hidden_size as usize;
        let qd = c.attention_q_dim() as usize;
        let kvd = c.attention_kv_dim() as usize;
        let short = n <= sizes::GEMV_MAX_TOKENS;
        let s = self.s;
        let (wq, wk, wv) = (weight(&l.attn_q)?, weight(&l.attn_k)?, weight(&l.attn_v)?);
        if short {
            q8_dense(
                s,
                wq,
                st.xq.as_slice(),
                st.q.slice_mut(0..n * qd),
                n,
                "attn_q",
            )?;
            q8_dense(
                s,
                wk,
                st.xq.as_slice(),
                st.k.slice_mut(0..n * kvd),
                n,
                "attn_k",
            )?;
            q8_dense(
                s,
                wv,
                st.xq.as_slice(),
                st.v.slice_mut(0..n * kvd),
                n,
                "attn_v",
            )?;
        } else {
            dense(
                s,
                wq,
                st.h.slice(0..n * hd),
                st.q.slice_mut(0..n * qd),
                n,
                "attn_q",
            )?;
            dense(
                s,
                wk,
                st.h.slice(0..n * hd),
                st.k.slice_mut(0..n * kvd),
                n,
                "attn_k",
            )?;
            dense(
                s,
                wv,
                st.h.slice(0..n * hd),
                st.v.slice_mut(0..n * kvd),
                n,
                "attn_v",
            )?;
        }
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let heads = c.num_heads as usize;
        let kv_heads = c.num_kv_heads as usize;
        if let Some(rows) = batch {
            let capacity = rows.capacity;
            let mut k_caches = Vec::with_capacity(n);
            let mut v_caches = Vec::with_capacity(n);
            for session in rows.sessions.iter_mut() {
                let row = &mut session.st;
                k_caches.push(row.k_cache[il].as_mut().ok_or("no K cache")?.as_mut_slice());
                v_caches.push(row.v_cache[il].as_mut().ok_or("no V cache")?.as_mut_slice());
            }
            supported(
                k::kv_append_rows(
                    s,
                    st.k.slice(0..n * kvd),
                    st.v.slice(0..n * kvd),
                    &mut k_caches,
                    &mut v_caches,
                    rows.tables.reborrow(),
                    rows.positions,
                    n,
                    kvd,
                    capacity,
                ),
                "kv_append",
            )?;
            let k_caches: Vec<_> = k_caches.iter().map(DeviceSliceMut::as_const).collect();
            let v_caches: Vec<_> = v_caches.iter().map(DeviceSliceMut::as_const).collect();
            supported(
                k::attention_short_rows(
                    s,
                    st.q.slice(0..n * qd),
                    &k_caches,
                    &v_caches,
                    rows.tables.reborrow(),
                    st.ctx.slice_mut(0..n * qd),
                    rows.positions,
                    n,
                    heads,
                    kv_heads,
                    scale,
                    st.attention_partial
                        .slice_mut(0..sizes::attention_short_partial(n, heads)),
                    capacity,
                ),
                "attention",
            )?;
        } else {
            let pos = st.positions.get(&n).ok_or("no position for the chunk")?;
            let k_cache = st.k_cache[il].as_mut().ok_or("no K cache")?;
            let v_cache = st.v_cache[il].as_mut().ok_or("no V cache")?;
            supported(
                k::kv_append(
                    s,
                    st.k.slice(0..n * kvd),
                    st.v.slice(0..n * kvd),
                    k_cache.slice_mut(0..self.capacity * kvd),
                    v_cache.slice_mut(0..self.capacity * kvd),
                    pos.view(),
                    n,
                    kvd,
                    self.capacity,
                ),
                "kv_append",
            )?;
            if n <= sizes::ATTENTION_SHORT_TOKENS {
                supported(
                    k::attention_short(
                        s,
                        st.q.slice(0..n * qd),
                        k_cache.slice(0..self.capacity * kvd),
                        v_cache.slice(0..self.capacity * kvd),
                        st.ctx.slice_mut(0..n * qd),
                        pos.view(),
                        n,
                        heads,
                        kv_heads,
                        scale,
                        st.attention_partial
                            .slice_mut(0..sizes::attention_short_partial(n, heads)),
                        self.capacity,
                    ),
                    "attention",
                )?;
            } else {
                supported(
                    k::attention(
                        s,
                        st.q.slice(0..n * qd),
                        k_cache.slice(0..self.capacity * kvd),
                        v_cache.slice(0..self.capacity * kvd),
                        st.ctx.slice_mut(0..n * qd),
                        pos.view(),
                        n,
                        heads,
                        kv_heads,
                        scale,
                        self.capacity,
                    ),
                    "attention",
                )?;
            }
        }
        let out = weight(&l.attn_out)?;
        if short {
            supported(
                k::quantize_q8_1(self.s, st.ctx.slice(0..n * qd), st.xq.as_mut_slice(), n, qd),
                "quantize_q8_1",
            )?;
            return self.q8_residual(st, out, 1.0, n, "attn_output");
        }
        dense(
            self.s,
            out,
            st.ctx.slice(0..n * qd),
            st.out.slice_mut(0..n * hd),
            n,
            "attn_output",
        )?;
        supported(
            k::add_scaled(
                self.s,
                st.x.slice_mut(0..n * hd),
                st.out.slice(0..n * hd),
                1.0,
                None,
                n * hd,
            ),
            "add_scaled",
        )
    }

    fn moe(&self, st: &mut State, l: &DeviceLayer, n: usize) -> Result<(), String> {
        let c = self.c;
        let hd = c.hidden_size as usize;
        let latent = c.latent_size as usize;
        let used = l.experts_used;
        let ff = l.expert_ff;
        let shared_ff = c.shared_expert_ff as usize;
        let experts = l.num_experts;
        let short = n <= sizes::GEMV_MAX_TOKENS;
        // Several tokens run the expert products on BF16 activations: the
        // shared expert for exactly n rows, the routed experts once per
        // expert for all the slots that chose it. One token keeps the staged
        // kernels.
        let bf16 = short && n > 1;
        let tiled = bf16 && n * used <= sizes::ROUTE_SHORT_SLOTS;
        let grid = self.m.iq2xxs_grid.view();
        let s = self.s;
        let (up_shexp, down_shexp) = (weight(&l.up_shexp)?, weight(&l.down_shexp)?);
        // The shared expert depends only on h: it overlaps the router, the
        // routed experts and their small kernels on the side stream, which
        // reads h and owns the residual stream until the join; nothing else
        // touches it meanwhile.
        let fork = if short {
            let side = self.side;
            let fork = s
                .fork(&side.stream, &side.start, &side.done)
                .map_err(|e| format!("side stream fork: {e}"))?;
            fork.lend(st.h.allocation())
                .and_then(|()| {
                    [
                        st.x.allocation(),
                        st.shared_up.allocation(),
                        st.h_bf16.allocation(),
                        st.shared_up_bf16.allocation(),
                    ]
                    .into_iter()
                    .try_for_each(|memory| fork.hand_over(memory))
                })
                .map_err(|e| format!("side stream fork: {e}"))?;
            Some(fork)
        } else {
            None
        };
        if let Some(fork) = &fork {
            let side = fork.side();
            if bf16 {
                supported(
                    k::activations_to_bf16(
                        side,
                        st.h.slice(0..n * hd),
                        st.h_bf16.slice_mut(0..n * hd),
                        n * hd,
                    ),
                    "activations_to_bf16",
                )?;
                supported(
                    k::iq2xxs_gemv_bf16_store_bf16(
                        side,
                        up_shexp.bytes(),
                        grid,
                        st.h_bf16.slice(0..n * hd),
                        st.shared_up_bf16.slice_mut(0..n * shared_ff),
                        shared_ff,
                        hd,
                        n,
                        true,
                    ),
                    "iq2xxs_gemv",
                )?;
                supported(
                    k::iq2xxs_gemv_bf16(
                        side,
                        down_shexp.bytes(),
                        grid,
                        st.shared_up_bf16.slice(0..n * shared_ff),
                        st.x.slice_mut(0..n * hd),
                        hd,
                        shared_ff,
                        n,
                        false,
                        true,
                    ),
                    "iq2xxs_gemv",
                )?;
            } else {
                supported(
                    k::iq2xxs_gemv(
                        side,
                        up_shexp.bytes(),
                        grid,
                        st.h.slice(0..n * hd),
                        st.shared_up.slice_mut(0..n * shared_ff),
                        shared_ff,
                        hd,
                        n,
                        true,
                        false,
                    ),
                    "iq2xxs_gemv",
                )?;
                supported(
                    k::iq2xxs_gemv(
                        side,
                        down_shexp.bytes(),
                        grid,
                        st.shared_up.slice(0..n * shared_ff),
                        st.x.slice_mut(0..n * hd),
                        hd,
                        shared_ff,
                        n,
                        false,
                        true,
                    ),
                    "iq2xxs_gemv",
                )?;
            }
        }
        // Router logits in F32: [n][experts] = h[n][H] * router[experts][H]^T.
        // Decode-sized batches stream the weights once; hipBLAS takes prefill.
        let router = f32s(&l.router)?;
        if n <= sizes::ROUTER_MAX_TOKENS {
            supported(
                k::router_logits(
                    s,
                    router,
                    st.h.slice(0..n * hd),
                    st.router.slice_mut(0..n * experts),
                    experts,
                    hd,
                    n,
                ),
                "router_logits",
            )?;
        } else {
            self.blas
                .sgemm(
                    s,
                    Op::T,
                    Op::N,
                    experts,
                    n,
                    hd,
                    1.0,
                    router,
                    hd,
                    st.h.slice(0..n * hd),
                    hd,
                    st.router.slice_mut(0..n * experts),
                    experts,
                )
                .map_err(|_| "router GEMM failed".to_string())?;
        }
        if let Entry::Vacant(slot) = st.ids.entry(experts) {
            let bound = u32::try_from(experts).map_err(|_| "too many experts".to_string())?;
            slot.insert(
                self.s
                    .alloc_indices(PREFILL_CHUNK * used.max(1), bound)
                    .map_err(hip)?,
            );
        }
        let ids = st.ids.get_mut(&experts).ok_or("no expert IDs")?;
        let slots = n * used;
        supported(
            k::router_top_k(
                s,
                st.router.slice(0..n * experts),
                f32s(&l.router_bias)?,
                experts,
                n,
                used,
                c.expert_weights_norm,
                c.expert_weights_scale,
                ids.slice_mut(0..slots),
                st.route_weights.slice_mut(0..slots),
            ),
            "router_top_k",
        )?;
        let latent_down = weight(&l.latent_down)?;
        if short {
            q8_dense(
                s,
                latent_down,
                st.xq.as_slice(),
                st.latent.slice_mut(0..n * latent),
                n,
                "ffn_latent_down",
            )?;
        } else {
            dense(
                s,
                latent_down,
                st.h.slice(0..n * hd),
                st.latent.slice_mut(0..n * latent),
                n,
                "ffn_latent_down",
            )?;
        }
        let (up_exps, down_exps) = (weight(&l.up_exps)?, weight(&l.down_exps)?);
        let latent_up = weight(&l.latent_up)?;
        if tiled {
            // Each expert runs once for all of its slots, on BF16 rows; the
            // up products store BF16 rows for the down products.
            supported(
                k::activations_to_bf16(
                    s,
                    st.latent.slice(0..n * latent),
                    st.latent_bf16.slice_mut(0..n * latent),
                    n * latent,
                ),
                "activations_to_bf16",
            )?;
            let route = match st.short_routes.entry((slots, experts)) {
                Entry::Occupied(route) => route.into_mut(),
                Entry::Vacant(slot) => {
                    let bound =
                        |v: usize| u32::try_from(v).map_err(|_| "too many slots".to_string());
                    slot.insert(ShortRoute {
                        order: s.alloc_indices(slots, bound(slots)?).map_err(hip)?,
                        tiles: s
                            .alloc_indices(slots * 4, bound(slots + experts + 1)?)
                            .map_err(hip)?,
                        num_tiles: s.alloc_indices(1, bound(slots + 1)?).map_err(hip)?,
                    })
                }
            };
            supported(
                k::iq2xxs_route_short(
                    s,
                    ids.slice(0..slots),
                    route.order.view_mut(),
                    route.tiles.view_mut(),
                    route.num_tiles.view_mut(),
                    slots,
                    experts,
                ),
                "iq2xxs_route_short",
            )?;
            supported(
                k::iq2xxs_gemv_tiles_store_bf16(
                    s,
                    ids.slice(0..slots),
                    up_exps.bytes(),
                    grid,
                    st.latent_bf16.slice(0..n * latent),
                    st.up_bf16.slice_mut(0..slots * ff),
                    route.order.view(),
                    route.tiles.view(),
                    route.num_tiles.view(),
                    ff,
                    latent,
                    slots,
                    used,
                    true,
                    experts,
                ),
                "iq2xxs_gemv",
            )?;
            supported(
                k::iq2xxs_gemv_tiles_per_slot(
                    s,
                    ids.slice(0..slots),
                    down_exps.bytes(),
                    grid,
                    st.up_bf16.slice(0..slots * ff),
                    st.down.slice_mut(0..slots * latent),
                    route.order.view(),
                    route.tiles.view(),
                    route.num_tiles.view(),
                    latent,
                    ff,
                    slots,
                    used,
                    false,
                    false,
                    experts,
                ),
                "iq2xxs_gemv",
            )?;
        } else if short {
            // The vector products apply ReLU² as they store.
            supported(
                k::iq2xxs_gemv_routed(
                    s,
                    up_exps.bytes(),
                    grid,
                    st.latent.slice(0..n * latent),
                    ids.slice(0..slots),
                    st.up.slice_mut(0..slots * ff),
                    ff,
                    latent,
                    n,
                    used,
                    true,
                    false,
                    experts,
                ),
                "iq2xxs_gemv",
            )?;
            supported(
                k::iq2xxs_gemv_routed_per_slot(
                    s,
                    down_exps.bytes(),
                    grid,
                    st.up.slice(0..slots * ff),
                    ids.slice(0..slots),
                    st.down.slice_mut(0..slots * latent),
                    latent,
                    ff,
                    n,
                    used,
                    false,
                    false,
                    experts,
                ),
                "iq2xxs_gemv",
            )?;
        }
        if short {
            supported(
                k::weighted_slot_sum_q8(
                    s,
                    st.down.slice(0..slots * latent),
                    st.route_weights.slice(0..slots),
                    st.xq.as_mut_slice(),
                    n,
                    used,
                    latent,
                ),
                "weighted_slot_sum",
            )?;
            if let Some(fork) = fork {
                fork.join().map_err(|e| format!("side stream join: {e}"))?;
            }
            return self.q8_residual(st, latent_up, 1.0, n, "ffn_latent_up");
        }
        // Routed experts over the slots grouped by expert (the routed product
        // applies ReLU² as it stores), then the dense shared expert.
        let slot_bound = u32::try_from(slots).map_err(|_| "too many slots".to_string())?;
        let mut order = self.s.alloc_indices(slots, slot_bound).map_err(hip)?;
        let max_tiles = sizes::iq2xxs_route_max_tiles(slots, experts);
        let tile_bound =
            u32::try_from(slots + experts + 33).map_err(|_| "too many tiles".to_string())?;
        let mut tiles = self
            .s
            .alloc_indices(max_tiles * 4, tile_bound)
            .map_err(hip)?;
        let count_bound =
            u32::try_from(slots / 32 + experts + 1).map_err(|_| "too many tiles".to_string())?;
        let mut num_tiles = self.s.alloc_indices(1, count_bound).map_err(hip)?;
        supported(
            k::iq2xxs_route_slots(
                s,
                ids.slice(0..slots),
                st.route_cursor.slice_mut(0..experts),
                order.view_mut(),
                tiles.view_mut(),
                num_tiles.view_mut(),
                slots,
                experts,
            ),
            "iq2xxs_route_slots",
        )?;
        supported(
            k::iq2xxs_routed_gemm(
                s,
                ids.slice(0..slots),
                up_exps.bytes(),
                grid,
                st.latent.slice(0..n * latent),
                st.up.slice_mut(0..slots * ff),
                order.view(),
                tiles.view(),
                num_tiles.view(),
                ff,
                latent,
                true,
                slots,
                experts,
                used,
            ),
            "iq2xxs_routed_gemm",
        )?;
        supported(
            k::iq2xxs_routed_gemm_per_slot(
                s,
                ids.slice(0..slots),
                down_exps.bytes(),
                grid,
                st.up.slice(0..slots * ff),
                st.down.slice_mut(0..slots * latent),
                order.view(),
                tiles.view(),
                num_tiles.view(),
                latent,
                ff,
                false,
                slots,
                experts,
                used,
            ),
            "iq2xxs_routed_gemm",
        )?;
        supported(
            k::weighted_slot_sum(
                s,
                st.down.slice(0..slots * latent),
                st.route_weights.slice(0..slots),
                st.acc.slice_mut(0..n * latent),
                n,
                used,
                latent,
            ),
            "weighted_slot_sum",
        )?;
        dense(
            s,
            latent_up,
            st.acc.slice(0..n * latent),
            st.out.slice_mut(0..n * hd),
            n,
            "ffn_latent_up",
        )?;
        dense(
            s,
            up_shexp,
            st.h.slice(0..n * hd),
            st.shared_up.slice_mut(0..n * shared_ff),
            n,
            "ffn_up_shexp",
        )?;
        supported(
            k::relu2(s, st.shared_up.slice_mut(0..n * shared_ff), n * shared_ff),
            "relu2",
        )?;
        dense(
            s,
            down_shexp,
            st.shared_up.slice(0..n * shared_ff),
            st.shared.slice_mut(0..n * hd),
            n,
            "ffn_down_shexp",
        )?;
        supported(
            k::add_scaled(
                s,
                st.x.slice_mut(0..n * hd),
                st.out.slice(0..n * hd),
                1.0,
                Some(st.shared.slice(0..n * hd)),
                n * hd,
            ),
            "add_scaled",
        )
    }

    /// The output head over the last `rows` rows of the chunk into the
    /// per-token logits (`per_token`) or `last_logits`.
    fn head(&self, st: &mut State, n: usize, rows: usize, per_token: bool) -> Result<(), String> {
        let c = self.c;
        let hd = c.hidden_size as usize;
        let vocab = c.vocab_size as usize;
        let last = (n - rows) * hd..n * hd;
        let s = self.s;
        let target = if per_token {
            st.logits.slice_mut(0..rows * vocab)
        } else {
            st.last_logits.slice_mut(0..vocab)
        };
        let norm = self
            .m
            .output_norm
            .f32()
            .ok_or("the output norm is not F32")?;
        if rows <= sizes::GEMV_MAX_TOKENS {
            supported(
                k::rms_norm_q8(
                    s,
                    st.x.slice(last),
                    norm,
                    None,
                    Some(st.xq.as_mut_slice()),
                    rows,
                    hd,
                    c.rms_eps,
                ),
                "rms_norm",
            )?;
            return q8_dense(s, &self.m.output, st.xq.as_slice(), target, rows, "output");
        }
        supported(
            k::rms_norm(
                s,
                st.x.slice(last),
                norm,
                st.h.slice_mut(0..rows * hd),
                rows,
                hd,
                c.rms_eps,
            ),
            "rms_norm",
        )?;
        dense(
            s,
            &self.m.output,
            st.h.slice(0..rows * hd),
            target,
            rows,
            "output",
        )
    }
}

/// One sequence: its position, KV cache, Mamba conv/SSM states and scratch.
pub struct Session {
    model: Arc<Model>,
    st: State,
    position: u32,
    max_context: u32,
    /// Tokens of a `verify` awaiting `commit`.
    verified: u32,
    needs_reset: bool,
}

/// Summaries the device computed for rows `0..n` of a logits buffer.
fn enqueue_heads(
    s: &Stream,
    st: &mut State,
    per_token: bool,
    n: usize,
    vocab: usize,
    filter: &LogitHeadFilter,
) -> Result<bool, String> {
    if filter.keep > sizes::LOGIT_HEAD_CANDIDATES || filter.minimum > sizes::LOGIT_HEAD_CANDIDATES {
        return Ok(false);
    }
    let rows = if per_token {
        st.logits.slice(0..n * vocab)
    } else {
        st.last_logits.slice(0..vocab)
    };
    supported(
        k::logit_heads(
            s,
            rows,
            n,
            vocab,
            filter.temperature,
            filter.keep,
            filter.top_p,
            filter.minimum,
            st.head_scratch.as_mut_slice(),
            st.head_records
                .slice_mut(0..n * sizes::LOGIT_HEAD_RECORD_BYTES),
            st.head_candidates
                .slice_mut(0..n * sizes::LOGIT_HEAD_CANDIDATES * sizes::LOGIT_HEAD_CANDIDATE_BYTES),
        ),
        "could not summarize the logits",
    )
    .map_err(|_| "could not summarize the logits".to_string())?;
    let records = n * sizes::LOGIT_HEAD_RECORD_BYTES;
    let candidates = n * sizes::LOGIT_HEAD_CANDIDATES * sizes::LOGIT_HEAD_CANDIDATE_BYTES;
    s.download_blocking(
        st.head_records.slice(0..records),
        &mut st.host_head_records[..records],
    )
    .map_err(|_| "could not summarize the logits".to_string())?;
    s.download_blocking(
        st.head_candidates.slice(0..candidates),
        &mut st.host_head_candidates[..candidates],
    )
    .map_err(|_| "could not summarize the logits".to_string())?;
    Ok(true)
}

/// Row `r`'s head that `enqueue_heads` or a batch downloaded, or an empty
/// head when it summarized nothing.
fn take_heads_row(st: &State, r: usize, summarized: bool, temperature: f32, head: &mut LogitHead) {
    head.candidates.clear();
    if !summarized {
        return;
    }
    let at = r * sizes::LOGIT_HEAD_RECORD_BYTES;
    let mut record = [0u8; sizes::LOGIT_HEAD_RECORD_BYTES];
    record.copy_from_slice(&st.host_head_records[at..at + sizes::LOGIT_HEAD_RECORD_BYTES]);
    let (softmax_sum, maximum, count) = sizes::logit_head_record(&record);
    if count <= 0 {
        return;
    }
    head.maximum = maximum;
    head.softmax_sum = softmax_sum;
    head.temperature = temperature;
    let base = r * sizes::LOGIT_HEAD_CANDIDATES * sizes::LOGIT_HEAD_CANDIDATE_BYTES;
    let count = (count as usize).min(sizes::LOGIT_HEAD_CANDIDATES);
    head.candidates.extend((0..count).map(|i| {
        let at = base + i * sizes::LOGIT_HEAD_CANDIDATE_BYTES;
        let b = &st.host_head_candidates[at..at + 8];
        LogitCandidate {
            token: u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            value: f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        }
    }));
}

/// The heads `enqueue_heads` downloaded, or empty heads when it summarized
/// nothing.
fn take_heads(st: &State, summarized: bool, temperature: f32, heads: &mut [LogitHead]) {
    for (r, head) in heads.iter_mut().enumerate() {
        take_heads_row(st, r, summarized, temperature, head);
    }
}

impl Session {
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    pub fn position(&self) -> u32 {
        self.position
    }

    pub fn max_context(&self) -> u32 {
        self.max_context
    }

    /// False after a failed operation, until `reset`.
    pub fn is_coherent(&self) -> bool {
        !self.needs_reset
    }

    /// Drafts the block that starts with `anchor` at position `pos0`;
    /// `None` when the model had no drafter when the session was created.
    pub fn draft_block(
        &mut self,
        anchor: i32,
        pos0: u32,
    ) -> Result<Option<crate::drafter::DraftLattice>, String> {
        let model = Arc::clone(&self.model);
        let exec = model.exec();
        match self.st.draft.as_mut() {
            Some(draft) => draft
                .draft(&exec.stream, &exec.blas, anchor, pos0, None)
                .map(Some),
            None => Ok(None),
        }
    }

    /// Whether the session keeps a drafter context.
    pub fn has_draft(&self) -> bool {
        self.st.draft.is_some()
    }

    /// Appends `tokens` at the current position, in chunks. `logits`
    /// receives the next-token distribution after the last token when it
    /// holds one vocabulary row, or after every token when it holds one row
    /// per token; with an empty slice only the device keeps the last row
    /// (`read_logits`).
    pub fn append(&mut self, tokens: &[i32], logits: &mut [f32]) -> Result<(), String> {
        self.append_rows(tokens, logits, None, &mut [], &[])
    }

    /// `append` whose inputs at the positions of `rows` are those rows
    /// instead of the tokens' embeddings.
    pub fn append_with_rows(
        &mut self,
        tokens: &[i32],
        logits: &mut [f32],
        rows: &[InputRows<'_>],
    ) -> Result<(), String> {
        self.append_rows(tokens, logits, None, &mut [], rows)
    }

    /// `append` returning one row, which the device also summarizes as
    /// `filter` requires: the head's candidates stay empty when the row
    /// needs more than a short head.
    pub fn append_with_head(
        &mut self,
        tokens: &[i32],
        logits: &mut [f32],
        filter: &LogitHeadFilter,
        head: &mut LogitHead,
    ) -> Result<(), String> {
        if logits.len() != self.model.config().vocab_size as usize {
            return Err("a summarized append returns one logits row".into());
        }
        self.append_rows(
            tokens,
            logits,
            Some(filter),
            std::slice::from_mut(head),
            &[],
        )
    }

    fn fail<T>(&mut self, message: impl Into<String>) -> Result<T, String> {
        Err(message.into())
    }

    fn append_rows(
        &mut self,
        tokens: &[i32],
        logits: &mut [f32],
        filter: Option<&LogitHeadFilter>,
        heads: &mut [LogitHead],
        rows: &[InputRows<'_>],
    ) -> Result<(), String> {
        if self.needs_reset {
            self.reset()?;
        }
        // An uncommitted verify left the states untouched.
        self.verified = 0;
        let model = Arc::clone(&self.model);
        let exec = model.exec();
        let s = &exec.stream;
        let c = model.config();
        let vocab = c.vocab_size as usize;
        let hd = c.hidden_size as usize;
        // One token's single row is the last-row case.
        let all_positions = tokens.len() > 1 && logits.len() == tokens.len() * vocab;
        if !logits.is_empty() && logits.len() != vocab && !all_positions {
            return self.fail("logits must hold one row or one row per token");
        }
        if tokens.len() > (self.max_context - self.position) as usize {
            return self.fail("the session context is full");
        }
        let mut host_tokens = Vec::with_capacity(tokens.len());
        for &t in tokens {
            match u32::try_from(t) {
                Ok(t) if t < c.vocab_size => host_tokens.push(t),
                _ => return self.fail("token outside the vocabulary"),
            }
        }
        for r in rows {
            if r.rows.len() % hd != 0 {
                return self.fail("input rows must be whole trunk rows");
            }
        }
        // A draft anchored at the end of this append or later reads at most
        // a window (plus its own block) back; older rows skip the drafter.
        let mut keep_from = 0u32;
        if let Some(d) = model.drafter() {
            let end = self.position + tokens.len() as u32;
            let reach = d.config().sliding_window + d.config().block_size;
            keep_from = end.saturating_sub(reach);
        }
        let forward = Forward {
            c,
            m: model.device(),
            s,
            side: &exec.side,
            blas: &exec.blas,
            verify: false,
            capacity: self.max_context as usize,
        };
        let mut done = 0usize;
        let mut summarized = false;
        let result = (|| -> Result<(), String> {
            while done < tokens.len() {
                let n = PREFILL_CHUNK.min(tokens.len() - done);
                let last = done + n == tokens.len();
                let first = self.position;
                let st = &mut self.st;
                st.input_rows.clear();
                for (source, r) in rows.iter().enumerate() {
                    let count = r.rows.len() / hd;
                    let from = (r.position as usize).max(first as usize);
                    let to = (r.position as usize + count).min(first as usize + n);
                    if from < to {
                        st.input_rows.push((
                            from - first as usize,
                            to - from,
                            source,
                            from - r.position as usize,
                        ));
                    }
                }
                st.inject_skip = n.min(keep_from.saturating_sub(first) as usize);
                s.upload_indices(&host_tokens[done..done + n], &mut st.tokens)
                    .map_err(|_| "could not upload tokens".to_string())?;
                st.upload_position(s, n, forward.capacity, first)?;
                // The last row's head always runs, so last_logits holds the
                // frontier for read_logits.
                let head_rows = if all_positions { n } else { usize::from(last) };
                if all_positions && st.logits.len() < n * vocab {
                    st.logits = s
                        .alloc(n * vocab)
                        .map_err(|_| "could not allocate the logits buffer".to_string())?;
                }
                forward.enqueue(st, n, head_rows, all_positions, rows, first, None)?;
                if head_rows > 0 && !logits.is_empty() {
                    let dst = if all_positions {
                        &mut logits[done * vocab..(done + n) * vocab]
                    } else {
                        &mut logits[..vocab]
                    };
                    let source = if all_positions {
                        st.logits.slice(0..head_rows * vocab)
                    } else {
                        st.last_logits.slice(0..vocab)
                    };
                    s.download_blocking(source, dst)
                        .map_err(|_| "could not copy logits".to_string())?;
                }
                if last && all_positions {
                    s.copy(
                        st.logits.slice((n - 1) * vocab..n * vocab),
                        st.last_logits.as_mut_slice(),
                    )
                    .map_err(|_| "could not copy logits".to_string())?;
                }
                if let (Some(filter), true) = (filter, last) {
                    summarized = enqueue_heads(s, st, false, 1, vocab, filter)?;
                }
                s.synchronize()
                    .map_err(|e| format!("forward failed: {e}"))?;
                self.position += n as u32;
                done += n;
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.needs_reset = true;
            return Err(error);
        }
        if let Some(filter) = filter {
            take_heads(&self.st, summarized, filter.temperature, heads);
        }
        Ok(())
    }

    /// Runs 1 to 8 `tokens` at `position()` and returns one logits row per
    /// token without advancing: the Mamba states stay put and their inputs
    /// are kept until `commit` keeps a prefix. Another append or verify
    /// discards them.
    pub fn verify(&mut self, tokens: &[i32], logits: &mut [f32]) -> Result<(), String> {
        self.verify_rows(tokens, logits, None, &mut [])
    }

    /// `verify` with the head of each row: `heads[i]` for row i.
    pub fn verify_with_heads(
        &mut self,
        tokens: &[i32],
        logits: &mut [f32],
        filter: &LogitHeadFilter,
        heads: &mut [LogitHead],
    ) -> Result<(), String> {
        if heads.len() != tokens.len() {
            return Err("verify summarizes one head per token".into());
        }
        self.verify_rows(tokens, logits, Some(filter), heads)
    }

    fn verify_rows(
        &mut self,
        tokens: &[i32],
        logits: &mut [f32],
        filter: Option<&LogitHeadFilter>,
        heads: &mut [LogitHead],
    ) -> Result<(), String> {
        if self.needs_reset {
            self.reset()?;
        }
        self.verified = 0;
        let model = Arc::clone(&self.model);
        let exec = model.exec();
        let s = &exec.stream;
        let c = model.config();
        let vocab = c.vocab_size as usize;
        let n = tokens.len();
        if n == 0 || n > MAX_BATCH {
            return Err("verify takes 1 to 8 tokens".into());
        }
        if logits.len() != n * vocab {
            return Err("verify returns one logits row per token".into());
        }
        if n > (self.max_context - self.position) as usize {
            return Err("the session context is full".into());
        }
        let mut host_tokens = Vec::with_capacity(n);
        for &t in tokens {
            match u32::try_from(t) {
                Ok(t) if t < c.vocab_size => host_tokens.push(t),
                _ => return Err("token outside the vocabulary".into()),
            }
        }
        let first = self.position;
        let forward = Forward {
            c,
            m: model.device(),
            s,
            side: &exec.side,
            blas: &exec.blas,
            verify: true,
            capacity: self.max_context as usize,
        };
        let st = &mut self.st;
        let result = (|| -> Result<bool, String> {
            if st.logits.len() < n * vocab {
                st.logits = s
                    .alloc(n * vocab)
                    .map_err(|_| "could not allocate the logits buffer".to_string())?;
            }
            st.inject_skip = 0;
            st.input_rows.clear();
            s.upload_indices(&host_tokens, &mut st.tokens)
                .map_err(|_| "could not upload tokens".to_string())?;
            st.upload_position(s, n, forward.capacity, first)?;
            forward.enqueue(st, n, n, true, &[], first, None)?;
            let summarized = match filter {
                Some(filter) => enqueue_heads(s, st, true, n, vocab, filter)?,
                None => false,
            };
            s.download_blocking(st.logits.slice(0..n * vocab), logits)
                .map_err(|e| format!("verify failed: {e}"))?;
            Ok(summarized)
        })();
        let summarized = match result {
            Ok(summarized) => summarized,
            Err(error) => {
                self.needs_reset = true;
                return Err(error);
            }
        };
        if let Some(filter) = filter {
            take_heads(&self.st, summarized, filter.temperature, heads);
        }
        self.verified = n as u32;
        Ok(())
    }

    /// Keeps the first `n` tokens of the last verify and advances past them;
    /// zero discards the verification.
    pub fn commit(&mut self, n: u32) -> Result<(), String> {
        if n > self.verified {
            self.needs_reset = true;
            return Err("commit exceeds the verified tokens".into());
        }
        self.verified = 0;
        if n == 0 {
            return Ok(());
        }
        let model = Arc::clone(&self.model);
        let exec = model.exec();
        let s = &exec.stream;
        let c = model.config();
        let device = model.device();
        let stride = c.ssm_in_projection() as usize;
        let inner = c.ssm_inner_size as usize;
        let channels = c.ssm_conv_channels() as usize;
        let kernel = c.ssm_conv_kernel as usize;
        let vocab = c.vocab_size as usize;
        let n = n as usize;
        let st = &mut self.st;
        // Replays the kept prefix through the same kernels the forward pass
        // uses, so the states equal those of appending the prefix.
        let result = (|| -> Result<(), String> {
            for (il, &block) in c.blocks.iter().enumerate() {
                if block != BlockType::Mamba {
                    continue;
                }
                let l = &device.layers[il];
                let scale = weight(&l.ssm_in)?.scale;
                let proj = st.verify_proj[il]
                    .as_ref()
                    .ok_or("no verification memory")?;
                let xbc = st.verify_xbc[il].as_ref().ok_or("no verification memory")?;
                let conv_state = st.conv_state[il].as_mut().ok_or("no conv state")?;
                supported(
                    k::mamba_conv_state(
                        s,
                        proj.slice(0..n * stride),
                        stride,
                        inner,
                        scale,
                        conv_state.as_mut_slice(),
                        n,
                        channels,
                        kernel,
                    ),
                    "mamba_conv_state",
                )?;
                let ssm_state = st.ssm_state[il].as_mut().ok_or("no SSM state")?;
                supported(
                    k::mamba_scan(
                        s,
                        xbc.slice(0..n * channels),
                        channels,
                        proj.slice(0..n * stride),
                        stride,
                        inner + channels,
                        scale,
                        f32s(&l.ssm_dt_b)?,
                        f32s(&l.ssm_a)?,
                        f32s(&l.ssm_d)?,
                        ssm_state.as_mut_slice(),
                        st.y.slice_mut(0..n * inner),
                        n,
                        c.ssm_heads as usize,
                        c.ssm_head_dim() as usize,
                        c.ssm_groups as usize,
                        true,
                    ),
                    "mamba_scan",
                )?;
            }
            // The verified row of the last kept token is the new frontier.
            s.copy(
                st.logits.slice((n - 1) * vocab..n * vocab),
                st.last_logits.as_mut_slice(),
            )
            .map_err(hip)?;
            s.synchronize().map_err(hip)
        })();
        if result.is_err() {
            self.needs_reset = true;
            return Err("could not commit the verified tokens".into());
        }
        self.position += n as u32;
        Ok(())
    }

    /// Appends one token to each of 2 to 8 sessions of one model in a single
    /// pass that reads the weights once: `tokens[i]` at `sessions[i]`'s
    /// position, its next-token row into `logits[i]` and, when `filters[i]`
    /// is set, its summary into `heads[i]`. Each session's recurrent and
    /// attention state advances with the arithmetic of a one-token append;
    /// the matrix products take the rows together.
    pub fn decode_batch(
        sessions: &mut [&mut Session],
        tokens: &[i32],
        logits: &mut [&mut [f32]],
        filters: &[Option<LogitHeadFilter>],
        heads: &mut [LogitHead],
    ) -> Result<(), String> {
        let n = sessions.len();
        if !(2..=MAX_BATCH).contains(&n) {
            return Err("a batch decodes 2 to 8 sessions".into());
        }
        if tokens.len() != n || logits.len() != n || filters.len() != n || heads.len() != n {
            return Err("a batch takes one token, row, filter and head per session".into());
        }
        let model = Arc::clone(&sessions[0].model);
        let c = model.config();
        let vocab = c.vocab_size as usize;
        let mut host_tokens = Vec::with_capacity(n);
        let mut positions = Vec::with_capacity(n);
        for r in 0..n {
            if !Arc::ptr_eq(&sessions[r].model, &model) {
                return Err("batched sessions must share their model".into());
            }
            match u32::try_from(tokens[r]) {
                Ok(t) if t < c.vocab_size => host_tokens.push(t),
                _ => return Err("token outside the vocabulary".into()),
            }
            if logits[r].len() != vocab {
                return Err("each batched session returns one logits row".into());
            }
            let session = &mut *sessions[r];
            if session.needs_reset {
                session.reset()?;
            }
            // An uncommitted verify left the states untouched.
            session.verified = 0;
            if session.position >= session.max_context {
                return Err("the session context is full".into());
            }
            positions.push(session.position);
        }
        let capacity = sessions.iter().map(|s| s.max_context).min().unwrap_or(0) as usize;
        let mut exec = model.exec();
        let exec = &mut *exec;
        let (s, blas) = (&exec.stream, &exec.blas);
        if exec.batch.is_none() {
            let mut st = State::new(&model, s, blas, None, MAX_BATCH)?;
            st.logits = alloc(s, MAX_BATCH * vocab)?;
            exec.batch = Some(Batch {
                st,
                tables: alloc(s, 2 * MAX_BATCH * 8)?,
                positions: HashMap::new(),
            });
        }
        let batch = exec
            .batch
            .as_mut()
            .ok_or("could not allocate batch memory")?;
        let result = (|| -> Result<Vec<bool>, String> {
            let bound = u32::try_from(capacity).map_err(|_| "context too large".to_string())?;
            if let Entry::Vacant(slot) = batch.positions.entry(bound) {
                slot.insert(s.alloc_indices(MAX_BATCH, bound).map_err(hip)?);
            }
            let row_positions = batch
                .positions
                .get_mut(&bound)
                .ok_or("could not upload the batch")?;
            s.upload_indices(&positions, row_positions)
                .map_err(|_| "could not upload the batch".to_string())?;
            let st = &mut batch.st;
            s.upload_indices(&host_tokens, &mut st.tokens)
                .map_err(|_| "could not upload the batch".to_string())?;
            st.inject_skip = 0;
            st.input_rows.clear();
            let forward = Forward {
                c,
                m: model.device(),
                s,
                side: &exec.side,
                blas,
                verify: false,
                capacity,
            };
            {
                let mut rows = BatchRows {
                    sessions: &mut *sessions,
                    tables: batch.tables.as_mut_slice(),
                    positions: row_positions.slice(0..n),
                    capacity,
                };
                forward.enqueue(st, n, n, true, &[], 0, Some(&mut rows))?;
            }
            // Each session injects its row's features into its own drafter
            // context.
            if !st.feature_slot.is_empty() {
                let width = st.feature_width;
                for (r, session) in sessions.iter_mut().enumerate() {
                    if let Some(draft) = session.st.draft.as_mut() {
                        draft
                            .inject(
                                s,
                                blas,
                                st.features.slice(r * width..(r + 1) * width),
                                positions[r],
                                1,
                            )
                            .map_err(|e| format!("drafter injection failed: {e}"))?;
                    }
                }
            }
            // Heads with each row's own filter, then the rows to the host.
            let mut summarized = vec![false; n];
            for r in 0..n {
                let Some(filter) = filters[r] else {
                    continue;
                };
                if filter.keep > sizes::LOGIT_HEAD_CANDIDATES
                    || filter.minimum > sizes::LOGIT_HEAD_CANDIDATES
                {
                    continue;
                }
                let records = r * sizes::LOGIT_HEAD_RECORD_BYTES;
                let candidates =
                    r * sizes::LOGIT_HEAD_CANDIDATES * sizes::LOGIT_HEAD_CANDIDATE_BYTES;
                supported(
                    k::logit_heads(
                        s,
                        st.logits.slice(r * vocab..(r + 1) * vocab),
                        1,
                        vocab,
                        filter.temperature,
                        filter.keep,
                        filter.top_p,
                        filter.minimum,
                        st.head_scratch.as_mut_slice(),
                        st.head_records
                            .slice_mut(records..records + sizes::LOGIT_HEAD_RECORD_BYTES),
                        st.head_candidates.slice_mut(
                            candidates
                                ..candidates
                                    + sizes::LOGIT_HEAD_CANDIDATES
                                        * sizes::LOGIT_HEAD_CANDIDATE_BYTES,
                        ),
                    ),
                    "could not summarize the logits",
                )?;
                summarized[r] = true;
            }
            if summarized.iter().any(|&x| x) {
                let records = n * sizes::LOGIT_HEAD_RECORD_BYTES;
                let candidates =
                    n * sizes::LOGIT_HEAD_CANDIDATES * sizes::LOGIT_HEAD_CANDIDATE_BYTES;
                s.download_blocking(
                    st.head_records.slice(0..records),
                    &mut st.host_head_records[..records],
                )
                .map_err(hip)?;
                s.download_blocking(
                    st.head_candidates.slice(0..candidates),
                    &mut st.host_head_candidates[..candidates],
                )
                .map_err(hip)?;
            }
            for (r, session) in sessions.iter_mut().enumerate() {
                s.download_blocking(st.logits.slice(r * vocab..(r + 1) * vocab), logits[r])
                    .map_err(hip)?;
                s.copy(
                    st.logits.slice(r * vocab..(r + 1) * vocab),
                    session.st.last_logits.as_mut_slice(),
                )
                .map_err(hip)?;
            }
            s.synchronize().map_err(hip)?;
            Ok(summarized)
        })();
        let summarized = match result {
            Ok(summarized) => summarized,
            Err(error) => {
                // The rows' states may be partly advanced; none of them can
                // continue.
                for session in sessions.iter_mut() {
                    session.needs_reset = true;
                }
                return Err(format!("batch decode failed: {error}"));
            }
        };
        let st = &batch.st;
        for (r, session) in sessions.iter_mut().enumerate() {
            let temperature = filters[r].map_or(0.0, |f| f.temperature);
            take_heads_row(st, r, summarized[r], temperature, &mut heads[r]);
            session.position += 1;
        }
        Ok(())
    }

    /// Returns to position zero with empty caches and zero states.
    pub fn reset(&mut self) -> Result<(), String> {
        let model = Arc::clone(&self.model);
        let exec = model.exec();
        let s = &exec.stream;
        let st = &mut self.st;
        let result = (|| -> Result<(), HipError> {
            for buffer in st
                .conv_state
                .iter_mut()
                .chain(st.ssm_state.iter_mut())
                .flatten()
            {
                s.fill_bytes(buffer.as_mut_slice(), 0)?;
            }
            s.synchronize()
        })();
        self.position = 0;
        self.verified = 0;
        self.needs_reset = result.is_err();
        result.map_err(|_| "could not clear the session state".to_string())
    }

    /// Marks the session empty without device work; the next append resets
    /// it.
    pub fn invalidate(&mut self) {
        self.position = 0;
        self.verified = 0;
        self.needs_reset = true;
    }

    /// Copies the logits row after the session's last token to the host.
    pub fn read_logits(&self, logits: &mut [f32]) -> Result<(), String> {
        if logits.len() != self.model.config().vocab_size as usize {
            return Err("logits must hold one vocabulary row".into());
        }
        let exec = self.model.exec();
        exec.stream
            .download_blocking(self.st.last_logits.as_slice(), logits)
            .map_err(|_| "could not read the logits".to_string())
    }

    /// Bytes `save` writes at the current position.
    pub fn snapshot_bytes(&self) -> usize {
        let c = self.model.config();
        let mut bytes = 0usize;
        for (il, &block) in c.blocks.iter().enumerate() {
            match block {
                BlockType::Attention => {
                    bytes += 2 * self.position as usize * c.attention_kv_dim() as usize * 2;
                }
                BlockType::Mamba => {
                    bytes += 4
                        * (self.st.conv_state[il].as_ref().map_or(0, DeviceBuffer::len)
                            + self.st.ssm_state[il].as_ref().map_or(0, DeviceBuffer::len));
                }
                BlockType::Moe => {}
            }
        }
        bytes + self.st.draft.as_ref().map_or(0, DraftContext::state_bytes)
    }

    pub fn save(&self) -> Result<SessionSnapshot, String> {
        if self.needs_reset {
            return Err("the session was invalidated".into());
        }
        let c = self.model.config();
        let st = &self.st;
        let exec = self.model.exec();
        let s = &exec.stream;
        let kv_count = self.position as usize * c.attention_kv_dim() as usize;
        let mut out = SessionSnapshot {
            position: self.position,
            ..SessionSnapshot::default()
        };
        let copy = || "could not copy the session state to the host".to_string();
        for (il, &block) in c.blocks.iter().enumerate() {
            match block {
                BlockType::Attention => {
                    let mut rows = vec![F16(0); 2 * kv_count];
                    let (k_rows, v_rows) = rows.split_at_mut(kv_count);
                    let k_cache = st.k_cache[il].as_ref().ok_or_else(copy)?;
                    let v_cache = st.v_cache[il].as_ref().ok_or_else(copy)?;
                    s.download_blocking(k_cache.slice(0..kv_count), k_rows)
                        .map_err(|_| copy())?;
                    s.download_blocking(v_cache.slice(0..kv_count), v_rows)
                        .map_err(|_| copy())?;
                    out.kv.push(rows.into_iter().map(|h| h.0).collect());
                }
                BlockType::Mamba => {
                    let conv = st.conv_state[il].as_ref().ok_or_else(copy)?;
                    let ssm = st.ssm_state[il].as_ref().ok_or_else(copy)?;
                    let mut state = vec![0.0f32; conv.len() + ssm.len()];
                    let (conv_rows, ssm_rows) = state.split_at_mut(conv.len());
                    s.download_blocking(conv.as_slice(), conv_rows)
                        .map_err(|_| copy())?;
                    s.download_blocking(ssm.as_slice(), ssm_rows)
                        .map_err(|_| copy())?;
                    out.recurrent.push(state);
                }
                BlockType::Moe => {}
            }
        }
        if let Some(draft) = st.draft.as_ref() {
            out.draft = vec![0; draft.state_bytes()];
            draft.save_state(s, &mut out.draft).map_err(|_| copy())?;
        }
        Ok(out)
    }

    /// Replaces this session's state; the snapshot may come from any session
    /// of the same model with a context no larger than this one.
    pub fn restore(&mut self, snapshot: &SessionSnapshot) -> Result<(), String> {
        let model = Arc::clone(&self.model);
        let c = model.config();
        if snapshot.position > self.max_context {
            self.needs_reset = true;
            return Err("the snapshot exceeds this session's context".into());
        }
        let kv_count = snapshot.position as usize * c.attention_kv_dim() as usize;
        let exec = model.exec();
        let s = &exec.stream;
        let st = &mut self.st;
        let result = (|| -> Result<(), String> {
            let mismatch = || "the snapshot does not match this model".to_string();
            let copy = || "could not copy the snapshot to the device".to_string();
            let (mut attention, mut mamba) = (0usize, 0usize);
            for (il, &block) in c.blocks.iter().enumerate() {
                match block {
                    BlockType::Attention => {
                        let rows = snapshot.kv.get(attention).ok_or_else(mismatch)?;
                        if rows.len() != 2 * kv_count {
                            return Err(mismatch());
                        }
                        attention += 1;
                        let halves: Vec<F16> = rows.iter().map(|&bits| F16(bits)).collect();
                        let k_cache = st.k_cache[il].as_mut().ok_or_else(mismatch)?;
                        s.upload(&halves[..kv_count], k_cache.slice_mut(0..kv_count))
                            .map_err(|_| copy())?;
                        let v_cache = st.v_cache[il].as_mut().ok_or_else(mismatch)?;
                        s.upload(&halves[kv_count..], v_cache.slice_mut(0..kv_count))
                            .map_err(|_| copy())?;
                    }
                    BlockType::Mamba => {
                        let state = snapshot.recurrent.get(mamba).ok_or_else(mismatch)?;
                        mamba += 1;
                        let conv = st.conv_state[il].as_mut().ok_or_else(mismatch)?;
                        let conv_len = conv.len();
                        let ssm_len = st.ssm_state[il].as_ref().map_or(0, DeviceBuffer::len);
                        if state.len() != conv_len + ssm_len {
                            return Err(mismatch());
                        }
                        s.upload(&state[..conv_len], conv.as_mut_slice())
                            .map_err(|_| copy())?;
                        let ssm = st.ssm_state[il].as_mut().ok_or_else(mismatch)?;
                        s.upload(&state[conv_len..], ssm.as_mut_slice())
                            .map_err(|_| copy())?;
                    }
                    BlockType::Moe => {}
                }
            }
            // A snapshot without drafter state leaves the drafter's context
            // stale: drafts get worse until it refills, outputs stay the same.
            if let Some(draft) = st.draft.as_mut()
                && !snapshot.draft.is_empty()
            {
                draft
                    .restore_state(s, &snapshot.draft)
                    .map_err(|_| "could not restore the drafter state".to_string())?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.needs_reset = true;
            return Err(error);
        }
        self.position = snapshot.position;
        self.verified = 0;
        self.needs_reset = false;
        Ok(())
    }
}
