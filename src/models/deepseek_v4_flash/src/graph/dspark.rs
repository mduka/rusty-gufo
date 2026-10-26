//! DSpark drafting (the "DSpark Drafting" part of `rocm_graph.cpp`). The
//! target's hidden states at the support model's target layers are captured
//! during prefill, decode and verification, projected into each support
//! stage's KV ring, and one support pass over [anchor, noise, ...] proposes a
//! whole block, which the Markov path selector turns into one trajectory.

use gufo_deepseek_v4_flash_kernels::{
    Backend, BatchMode, DeviceTensor, Error, Input, Model, Result, Rope, TensorView, Weight,
};

use super::alloc::{BatchWorkspace, lent, lent_pair, rows_mut, verify_streams};
use super::decode::{Target, present};
use super::prefill::{FfnLayer, Slots};
use super::*;
use crate::dspark_model::{DsparkModel, MAX_BLOCK, MAX_STAGES};
use crate::dspark_policy::ConfidencePolicy;
use crate::dspark_sampler::{CANDIDATES, Candidates, DsparkSampler};
use crate::gguf::TensorInfo;
use crate::weights::Weights;

/// The drafting state of a session with a support model (the `dspark_*`
/// fields of the C++ graph).
pub(crate) struct DsparkState {
    pub n_stages: u32,
    pub block_size: u32,
    pub target_layer_ids: Vec<u32>,
    /// Per stage, the injected and drafted KV rows, a ring of `cache_cap`.
    pub kv_cache: Vec<DeviceTensor>,
    pub cache_cap: u32,
    /// Captured target features, `capture_rows_cap` rows of the target
    /// layers side by side, and their fused projection.
    pub features: DeviceTensor,
    pub fused: DeviceTensor,
    pub capture_rows_cap: u32,
    pub markov_key: DeviceTensor,
    pub markov_index: DeviceTensor,
    /// The sampled cohorts' shortlists and their scratch, made on first use.
    pub candidates: Option<DeviceTensor>,
    pub candidate_scratch: Option<DeviceTensor>,
    pub capture_enabled: bool,
    /// Support KV covers the positions below this, contiguous from zero.
    pub context_len: u32,
    pub capture_mask: u32,
    pub capture_batch_mask: u32,
    pub capture_batch_start: u32,
    pub capture_batch_tokens: u32,
}

impl DsparkState {
    fn feature_width(&self) -> u32 {
        self.target_layer_ids.len() as u32 * N_EMBD
    }

    /// The feature slot of target layer `il`, if the support model samples it.
    fn slot(&self, il: u32) -> Option<u32> {
        self.target_layer_ids
            .iter()
            .position(|&l| l == il)
            .map(|s| s as u32)
    }

    fn complete_mask(&self) -> u32 {
        let n = self.target_layer_ids.len() as u32;
        if n >= 32 { 0 } else { (1 << n) - 1 }
    }

    /// `rocm_graph_dspark_capture_rows`: the features of `n_rows` stream rows
    /// of layer `il`.
    fn capture_rows(&mut self, b: &Backend, il: u32, hc: &impl Input, n_rows: u32) -> Result<()> {
        let Some(slot) = self.slot(il) else {
            return Ok(());
        };
        if n_rows == 0 || n_rows > self.capture_rows_cap {
            return Err(Error {
                op: "dspark capture",
            });
        }
        let feature_width = self.feature_width();
        b.dspark_capture_features(
            &mut self.features,
            hc,
            feature_width,
            slot * N_EMBD,
            N_EMBD,
            N_HC,
            n_rows,
        )?;
        self.capture_mask |= 1 << slot;
        Ok(())
    }
}

/// The RoPE of the support blocks, which have no compressor.
fn plain_rope(pos0: u32) -> Rope {
    Rope {
        n_rot: N_ROT,
        pos0,
        n_ctx_orig: 0,
        freq_base: ROPE_FREQ_BASE,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: ROPE_YARN_BETA_FAST,
        beta_slow: ROPE_YARN_BETA_SLOW,
    }
}

/// A support model weight.
fn sw(t: &TensorInfo) -> Weight {
    Weight::support(t.offset)
}

fn rows(t: &impl Input, row0: u32, n: u32, row_floats: u64) -> Result<TensorView<'_>> {
    t.view(
        u64::from(row0) * row_floats * F32,
        u64::from(n) * row_floats * F32,
    )
    .ok_or(Error { op: "dspark view" })
}

/// `ds4_dspark_visible_span`: the support ring rows a block at
/// `prefix_tokens` attends to (the last window of committed rows and the
/// block), as (start, count); count 0 when they do not fit.
fn visible_span(prefix_tokens: u32, draft_rows: u32, cache_capacity: u32) -> (u32, u32) {
    if draft_rows == 0 || draft_rows > cache_capacity || prefix_tokens > u32::MAX - draft_rows {
        return (0, 0);
    }
    let past = prefix_tokens.min(N_SWA);
    if past > cache_capacity - draft_rows {
        return (0, 0);
    }
    ((prefix_tokens - past) % cache_capacity, past + draft_rows)
}

/// Where the support head writes its hidden rows: the FFN norm rows of the
/// workspace for one request, its second DSpark verification stream for a
/// batch.
#[derive(Clone, Copy)]
enum Hidden {
    FfnNorm,
    DsparkVerify,
}

/// The parts of a support stage's attention the caller wants run (the
/// defaulted parameters of the C++ encoder).
#[derive(Clone, Copy)]
pub(crate) struct StageAttn {
    pub front_ready: bool,
    pub front_only: bool,
    pub project_qb: bool,
    pub project_q_a_kv: bool,
}

impl Default for StageAttn {
    fn default() -> Self {
        Self {
            front_ready: false,
            front_only: false,
            project_qb: true,
            project_q_a_kv: true,
        }
    }
}

/// One request of a batched draft (`ds4_rocm_dspark_draft_item`).
pub struct DraftItem<'a> {
    pub graph: &'a mut Graph,
    pub target_next_token: i32,
    pub position: u32,
    pub max_draft_tokens: u32,
    /// The request's sampler; stochastic proposals draw from it.
    pub sampler: Option<&'a mut DsparkSampler>,
    /// Requests stepping together, for the confidence policy's cost.
    pub concurrency: usize,
    /// The proposed tokens after the anchor.
    pub tokens: Vec<i32>,
}

/// A request of a sampled selection.
struct SampledRequest<'a> {
    target_next_token: i32,
    position: u32,
    max_draft_tokens: u32,
    concurrency: usize,
    sampler: Option<&'a mut DsparkSampler>,
    tokens: Vec<i32>,
}

fn proposes(sampler: &Option<&mut DsparkSampler>) -> bool {
    sampler.as_ref().is_some_and(|s| s.stochastic())
}

/// `rocm_graph_dspark_select_sampled`: each request's block position by
/// position from the Markov-coupled shortlist and its confidence, which the
/// confidence policy may stop on; requests without stochastic proposals take
/// the shortlist's best token. `coordinator` holds the logits and its
/// workspace `ws` the hidden rows, `rows_per_item` apart.
fn select_sampled(
    t: &Target<'_>,
    coordinator: &mut Graph,
    ws: &mut BatchWorkspace,
    requests: &mut [SampledRequest<'_>],
    rows_per_item: u32,
) -> Result<()> {
    let b = t.b;
    let refuse = Error {
        op: "dspark select sampled",
    };
    let d = t.dspark.ok_or(refuse)?;
    let ds = coordinator.dspark.as_mut().ok_or(refuse)?;
    let (candidates, scratch) = match (&mut ds.candidates, &mut ds.candidate_scratch) {
        (Some(c), Some(s)) => (c, s),
        _ => return Err(refuse),
    };
    let spec = coordinator.spec.as_ref().ok_or(refuse)?;
    let last = d.stages.last().ok_or(refuse)?;
    let (w1, w2, confidence) = (
        sw(present(&last.markov_w1)?),
        sw(present(&last.markov_w2)?),
        sw(present(&last.confidence)?),
    );
    let count = requests.len();
    let mut previous = Vec::with_capacity(count);
    let mut policies = Vec::with_capacity(count);
    let mut max_rows = 0;
    for request in requests.iter_mut() {
        if request.target_next_token < 0 || request.target_next_token as u32 >= N_VOCAB {
            return Err(refuse);
        }
        previous.push(request.target_next_token);
        let rows = request.max_draft_tokens.min(rows_per_item);
        policies.push(ConfidencePolicy::new(
            request.position,
            request.concurrency,
            rows,
        ));
        request.tokens.clear();
        max_rows = max_rows.max(rows);
    }
    let mut stopped = vec![false; count];
    let vocab_row = u64::from(N_VOCAB) * F32;
    let hidden_row = u64::from(N_EMBD) * F32;
    for row in 0..max_rows {
        let span_rows = (count as u64 - 1) * u64::from(rows_per_item) + 1;
        let logits = spec
            .logits
            .view(u64::from(row) * vocab_row, span_rows * vocab_row)
            .ok_or(refuse)?;
        // DS4 confidence reads hc_head(x) before the norm; the head's
        // weighted sum survives the tied projection in these rows.
        let hidden = ws
            .ffn_cur
            .view(u64::from(row) * hidden_row, span_rows * hidden_row)
            .ok_or(refuse)?;
        b.dspark_candidates(
            candidates,
            scratch,
            &logits,
            u64::from(rows_per_item) * u64::from(N_VOCAB),
            &hidden,
            u64::from(rows_per_item) * u64::from(N_EMBD),
            &previous,
            &mut ws.prefill_tokens,
            w1,
            w2,
            confidence,
            N_VOCAB,
            d.markov_rank,
            N_EMBD,
        )?;
        let mut bytes = vec![0u8; count * 68];
        b.read(candidates, 0, &mut bytes)?;
        for (index, request) in requests.iter_mut().enumerate() {
            if stopped[index] || row >= request.max_draft_tokens {
                continue;
            }
            let shortlist = Candidates::from_bytes(&bytes[index * 68..]).ok_or(refuse)?;
            let token = match request.sampler.as_deref_mut() {
                Some(sampler) if sampler.stochastic() => {
                    if !policies[index].include(shortlist.confidence) {
                        stopped[index] = true;
                        continue;
                    }
                    sampler.propose(row, &shortlist).map_err(|_| refuse)?
                }
                _ => shortlist.ids[0],
            };
            if token < 0 || token as u32 >= N_VOCAB {
                return Err(refuse);
            }
            request.tokens.push(token);
            previous[index] = token;
        }
        let more = requests
            .iter()
            .enumerate()
            .any(|(index, request)| !stopped[index] && row + 1 < request.max_draft_tokens);
        if !more {
            break;
        }
    }
    Ok(())
}

impl Graph {
    /// The sampled selection's buffers, allocated once.
    fn ensure_dspark_candidates(&mut self, b: &Backend) -> Result<()> {
        let ds = self.dspark.as_mut().ok_or(Error {
            op: "dspark candidates",
        })?;
        if ds.candidates.is_none() {
            ds.candidates = Some(b.alloc(8 * 68)?);
        }
        if ds.candidate_scratch.is_none() {
            let blocks = u64::from(N_VOCAB).div_ceil(256);
            ds.candidate_scratch = Some(b.alloc(8 * blocks * CANDIDATES as u64 * 8)?);
        }
        Ok(())
    }

    /// `rocm_graph_dspark_attach`: drafting state for `dspark`, whose
    /// verification rows double as the draft block's logits.
    pub(crate) fn dspark_attach(
        &mut self,
        b: &Backend,
        weights: &Weights,
        dspark: &DsparkModel,
    ) -> Result<()> {
        if self.dspark.is_some() {
            return Ok(());
        }
        let n_stages = dspark.stages.len() as u32;
        let block = dspark.block_size;
        if n_stages == 0 || n_stages > MAX_STAGES || block == 0 || block > MAX_BLOCK {
            return Err(Error {
                op: "dspark attach",
            });
        }
        self.spec_prepare(b, weights, block + 1)?;
        let cache_cap = self.raw_cap;
        let capture_rows_cap = block + 1;
        let feature_width = u64::from(dspark.target_layer_ids.len() as u32 * N_EMBD);
        let alloc = || -> Result<DsparkState> {
            let mut kv_cache = Vec::with_capacity(n_stages as usize);
            for _ in 0..n_stages {
                kv_cache.push(b.alloc(u64::from(cache_cap) * u64::from(N_HEAD_DIM) * F32)?);
            }
            Ok(DsparkState {
                n_stages,
                block_size: block,
                target_layer_ids: dspark.target_layer_ids.clone(),
                kv_cache,
                cache_cap,
                features: b.alloc(u64::from(capture_rows_cap) * feature_width * F32)?,
                fused: b.alloc(u64::from(capture_rows_cap) * u64::from(N_EMBD) * F32)?,
                capture_rows_cap,
                markov_key: b.alloc(8 * 8)?,
                markov_index: b.alloc(8 * 4)?,
                candidates: None,
                candidate_scratch: None,
                capture_enabled: true,
                context_len: 0,
                capture_mask: 0,
                capture_batch_mask: 0,
                capture_batch_start: 0,
                capture_batch_tokens: 0,
            })
        };
        let state = alloc()
            .inspect_err(|_| eprintln!("ds4: ROCm failed to allocate DSpark drafting state"))?;
        eprintln!(
            "ds4: DSpark drafting attached stages={n_stages} block={block} ring={cache_cap} rows features={:.1} MiB",
            (u64::from(capture_rows_cap) * feature_width * F32) as f64 / 1048576.0
        );
        self.dspark = Some(state);
        Ok(())
    }

    /// `rocm_graph_dspark_resize_capture`: capture rows for a prompt chunk,
    /// or back to one verification block.
    pub(crate) fn dspark_resize_capture(&mut self, b: &Backend, rows_cap: u32) -> Result<()> {
        let prefill_cap = self.prefill_cap;
        let Some(ds) = &mut self.dspark else {
            return Ok(());
        };
        if !ds.capture_enabled {
            return Ok(());
        }
        if rows_cap == 0 || rows_cap > prefill_cap {
            return Err(Error {
                op: "dspark resize capture",
            });
        }
        if rows_cap == ds.capture_rows_cap {
            return Ok(());
        }
        let features = b.alloc(u64::from(rows_cap) * u64::from(ds.feature_width()) * F32)?;
        let fused = b.alloc(u64::from(rows_cap) * u64::from(N_EMBD) * F32)?;
        ds.features = features;
        ds.fused = fused;
        ds.capture_rows_cap = rows_cap;
        ds.capture_mask = 0;
        ds.capture_batch_mask = 0;
        ds.capture_batch_start = 0;
        ds.capture_batch_tokens = 0;
        Ok(())
    }

    /// `rocm_graph_dspark_capture_rows` for rows of a coordinator's batch.
    pub(crate) fn dspark_capture_rows(
        &mut self,
        b: &Backend,
        il: u32,
        hc: &impl Input,
        n_rows: u32,
    ) -> Result<()> {
        match &mut self.dspark {
            Some(ds) => ds.capture_rows(b, il, hc, n_rows),
            None => Ok(()),
        }
    }

    /// `rocm_graph_dspark_capture_decode_layer`: layer `il`'s output of a
    /// decode step.
    pub(crate) fn dspark_capture_decode_layer(&mut self, t: &Target<'_>, il: u32) -> Result<()> {
        match &mut self.dspark {
            Some(ds) if ds.capture_enabled => ds.capture_rows(t.b, il, &self.d.cur_hc, 1),
            _ => Ok(()),
        }
    }

    /// `rocm_graph_dspark_capture_batch_layer`: layer `il`'s output of a
    /// prompt chunk or verification block; another (start, count) starts a
    /// new capture window.
    pub(crate) fn dspark_capture_batch_layer(
        &mut self,
        t: &Target<'_>,
        ws: &BatchWorkspace,
        il: u32,
        start: u32,
        n_tokens: u32,
    ) -> Result<()> {
        let cur = self.batch_cur_index;
        let Some(ds) = &mut self.dspark else {
            return Ok(());
        };
        if !ds.capture_enabled {
            return Ok(());
        }
        let Some(slot) = ds.slot(il) else {
            return Ok(());
        };
        if ds.capture_batch_mask == 0
            || ds.capture_batch_start != start
            || ds.capture_batch_tokens != n_tokens
        {
            ds.capture_batch_mask = 0;
            ds.capture_mask = 0;
            ds.capture_batch_start = start;
            ds.capture_batch_tokens = n_tokens;
        }
        ds.capture_rows(t.b, il, &ws.hc[cur], n_tokens)?;
        ds.capture_batch_mask |= 1 << slot;
        Ok(())
    }

    /// `rocm_graph_dspark_capture_complete`: every target layer captured.
    pub(crate) fn dspark_capture_ready(&self) -> bool {
        self.dspark.as_ref().is_some_and(|ds| {
            ds.capture_enabled && ds.complete_mask() != 0 && ds.capture_mask == ds.complete_mask()
        })
    }

    /// A support model is attached and captures target features.
    pub(crate) fn dspark_capturing(&self) -> bool {
        self.dspark.as_ref().is_some_and(|ds| ds.capture_enabled)
    }

    /// `ds4_rocm_graph_dspark_capture_reset`.
    pub(crate) fn dspark_capture_reset(&mut self) {
        if let Some(ds) = &mut self.dspark {
            ds.capture_mask = 0;
            ds.capture_batch_mask = 0;
            ds.capture_batch_start = 0;
            ds.capture_batch_tokens = 0;
        }
    }

    /// `ds4_rocm_graph_dspark_set_capture_enabled`; disabling forgets the
    /// support context.
    pub(crate) fn dspark_set_capture_enabled(&mut self, enabled: bool) {
        if let Some(ds) = &mut self.dspark {
            ds.capture_enabled = enabled;
            if !enabled {
                ds.context_len = 0;
                ds.capture_mask = 0;
                ds.capture_batch_mask = 0;
                ds.capture_batch_start = 0;
                ds.capture_batch_tokens = 0;
            }
        }
    }

    /// `ds4_rocm_graph_dspark_batch_capture_rows`: the (start, rows) of a
    /// complete batch capture.
    pub(crate) fn dspark_batch_capture(&self) -> Option<(u32, u32)> {
        let ds = self.dspark.as_ref()?;
        let want = ds.complete_mask();
        (want != 0 && ds.capture_batch_mask == want)
            .then_some((ds.capture_batch_start, ds.capture_batch_tokens))
    }

    /// `ds4_rocm_graph_dspark_context_len`.
    pub fn dspark_context_len(&self) -> u32 {
        self.dspark.as_ref().map_or(0, |ds| ds.context_len)
    }

    /// `ds4_rocm_graph_dspark_truncate_context`.
    pub(crate) fn dspark_truncate_context(&mut self, length: u32) {
        if let Some(ds) = &mut self.dspark {
            ds.context_len = ds.context_len.min(length);
        }
    }

    /// `ds4_rocm_graph_dspark_block_size`: 0 without a support model.
    pub fn dspark_block_size(&self) -> u32 {
        self.dspark.as_ref().map_or(0, |ds| ds.block_size)
    }

    /// `rocm_graph_dspark_inject`: the captured features into every
    /// stage's ring at positions `pos0 .. pos0 + n_rows`, through the
    /// graph's workspace `ws`.
    pub(crate) fn dspark_inject_rows(
        &mut self,
        t: &Target<'_>,
        ws: &mut BatchWorkspace,
        pos0: u32,
        n_rows: u32,
    ) -> Result<()> {
        let b = t.b;
        let d = t.dspark.ok_or(Error {
            op: "dspark inject",
        })?;
        let ds = self.dspark.as_mut().ok_or(Error {
            op: "dspark inject",
        })?;
        if n_rows == 0 || n_rows > ds.capture_rows_cap {
            return Err(Error {
                op: "dspark inject",
            });
        }
        let first = &d.stages[0];
        let n = u64::from(n_rows);
        let mut kv_raw = rows_mut(&mut ws.kv_raw, 0, n_rows, u64::from(N_HEAD_DIM))?;
        let mut kv = rows_mut(&mut ws.kv, 0, n_rows, u64::from(N_HEAD_DIM))?;
        // The target layers fuse once; every stage reads the fused rows.
        let feature_width = u64::from(ds.feature_width());
        let mut fused = rows_mut(&mut ds.fused, 0, n_rows, u64::from(N_EMBD))?;
        b.matmul_q8_0(
            &mut fused,
            sw(present(&first.main_proj)?),
            feature_width,
            u64::from(N_EMBD),
            &ds.features,
            n,
        )?;
        b.rms_norm_weight_rows_in_place(
            &mut ds
                .fused
                .view_mut(0, n * u64::from(N_EMBD) * F32)
                .ok_or(Error { op: "dspark view" })?,
            sw(present(&first.main_norm)?),
            N_EMBD,
            n_rows,
            RMS_EPS,
        )?;
        let fused = rows(&ds.fused, 0, n_rows, u64::from(N_EMBD))?;
        let rope = plain_rope(pos0);
        for (stage, cache) in d.stages.iter().zip(&mut ds.kv_cache) {
            let layer = &stage.block;
            b.matmul_q8_0(
                &mut kv_raw,
                sw(&layer.attn_kv),
                u64::from(N_EMBD),
                u64::from(N_HEAD_DIM),
                &fused,
                n,
            )?;
            b.rms_norm_weight_rows(
                &mut kv,
                &kv_raw,
                sw(&layer.attn_kv_a_norm),
                N_HEAD_DIM,
                n_rows,
                RMS_EPS,
            )?;
            // in place
            b.rope_tail(&mut kv, n_rows, N_HEAD_KV, N_HEAD_DIM, false, rope)?;
            // in place
            b.fp8_kv_quantize(&mut kv, n_rows, N_HEAD_DIM, N_ROT)?;
            b.store_raw_kv_batch(cache, &kv, ds.cache_cap, pos0, n_rows, N_HEAD_DIM)?;
        }
        // Coverage stays contiguous from position zero.
        if pos0 <= ds.context_len && pos0 + n_rows > ds.context_len {
            ds.context_len = pos0 + n_rows;
        }
        Ok(())
    }

    /// `ds4_rocm_graph_dspark_inject`: as one command batch.
    pub(crate) fn dspark_inject(
        &mut self,
        t: &Target<'_>,
        ws: &mut BatchWorkspace,
        pos0: u32,
        n_rows: u32,
    ) -> Result<()> {
        let b = t.b;
        self.check_workspace(ws)?;
        let result = b
            .begin_commands()
            .and_then(|()| self.dspark_inject_rows(t, ws, pos0, n_rows));
        match result {
            Ok(()) => b.end_commands(),
            Err(error) => {
                let _ = b.synchronize();
                Err(error)
            }
        }
    }

    /// `rocm_graph_encode_dspark_stage_attention`: one support stage's
    /// attention over the draft block at `pos0`: non-causal over the ring's
    /// committed window and the block, on the batch tensors `s`, which a
    /// batch rebinds to a coordinator's rows.
    pub(crate) fn encode_dspark_stage_attention(
        &mut self,
        t: &Target<'_>,
        stage: u32,
        pos0: u32,
        n_rows: u32,
        o: StageAttn,
        mut s: Slots<'_>,
    ) -> Result<()> {
        let b = t.b;
        let d = t.dspark.ok_or(Error { op: "dspark stage" })?;
        let ds = self.dspark.as_mut().ok_or(Error { op: "dspark stage" })?;
        if stage >= ds.n_stages || n_rows == 0 {
            return Err(Error { op: "dspark stage" });
        }
        let (span_start, span_count) = visible_span(pos0, n_rows, ds.cache_cap);
        if span_count == 0 {
            return Err(Error { op: "dspark span" });
        }
        let layer = &d.stages[stage as usize].block;
        let q_rank = layer.attn_q_a.dim[1];
        let n = u64::from(n_rows);
        let embd = u64::from(N_EMBD);
        let rope = plain_rope(pos0);
        if !o.front_ready {
            let mut hc_mix = rows_mut(&mut s.hc_mix, 0, n_rows, MIX_HC)?;
            let mut hc_split = rows_mut(&mut s.hc_split, 0, n_rows, MIX_HC)?;
            let mut attn_cur = rows_mut(&mut s.attn_cur, 0, n_rows, embd)?;
            let mut flat_hc = s.scratch.flat_hc()?;
            b.rms_norm_plain_rows(&mut flat_hc, &s.cur_hc, HC_DIM as u32, n_rows, RMS_EPS)?;
            b.matmul_f16(
                &mut hc_mix,
                sw(&layer.hc_attn_fn),
                HC_DIM,
                MIX_HC,
                &flat_hc,
                n,
            )?;
            b.hc_split_weighted_sum(
                &mut attn_cur,
                &mut hc_split,
                &hc_mix,
                &s.cur_hc,
                sw(&layer.hc_attn_scale),
                sw(&layer.hc_attn_base),
                N_EMBD,
                N_HC,
                N_HC_SINKHORN_ITER,
                HC_EPS,
            )?;
            b.rms_norm_weight_rows(
                &mut s.attn_norm,
                &s.attn_cur,
                sw(&layer.attn_norm),
                N_EMBD,
                n_rows,
                RMS_EPS,
            )?;
        }
        if o.project_q_a_kv {
            b.matmul_q8_0(
                &mut s.qr,
                sw(&layer.attn_q_a),
                embd,
                q_rank,
                &s.attn_norm,
                n,
            )?;
            b.rms_norm_weight_rows(
                &mut s.qr_norm,
                &s.qr,
                sw(&layer.attn_q_a_norm),
                q_rank as u32,
                n_rows,
                RMS_EPS,
            )?;
        }
        if o.project_qb {
            let mut q = s.scratch.q()?;
            b.matmul_q8_0(&mut q, sw(&layer.attn_q_b), q_rank, Q_DIM, &s.qr_norm, n)?;
        }
        if o.front_only {
            return Ok(());
        }
        let mut q = s.scratch.q()?;
        // in place
        b.head_rms_norm(&mut q, n_rows, N_HEAD, N_HEAD_DIM, RMS_EPS)?;
        // in place
        b.rope_tail(&mut q, n_rows, N_HEAD, N_HEAD_DIM, false, rope)?;
        if o.project_q_a_kv {
            b.matmul_q8_0(
                &mut s.kv_raw,
                sw(&layer.attn_kv),
                embd,
                u64::from(N_HEAD_DIM),
                &s.attn_norm,
                n,
            )?;
            b.rms_norm_weight_rows(
                &mut s.kv,
                &s.kv_raw,
                sw(&layer.attn_kv_a_norm),
                N_HEAD_DIM,
                n_rows,
                RMS_EPS,
            )?;
        }
        // in place
        b.rope_tail(&mut s.kv, n_rows, N_HEAD_KV, N_HEAD_DIM, false, rope)?;
        // in place
        b.fp8_kv_quantize(&mut s.kv, n_rows, N_HEAD_DIM, N_ROT)?;
        let cache = &mut ds.kv_cache[stage as usize];
        b.store_raw_kv_batch(cache, &s.kv, ds.cache_cap, pos0, n_rows, N_HEAD_DIM)?;
        b.attention_noncausal_raw_batch_heads(
            &mut s.heads,
            sw(&layer.attn_sinks),
            &q,
            cache,
            n_rows,
            span_count,
            ds.cache_cap,
            span_start,
            N_HEAD,
            N_HEAD_DIM,
        )?;
        // in place
        b.rope_tail(&mut s.heads, n_rows, N_HEAD, N_HEAD_DIM, true, rope)?;
        b.attention_output_q8_batch(
            &mut s.attn_out,
            &mut s.attn_low,
            sw(&layer.attn_output_a),
            sw(&layer.attn_output_b),
            u64::from(GROUP_DIM),
            u64::from(N_LORA_O),
            N_OUT_GROUP,
            embd,
            &s.heads,
            n_rows,
        )?;
        let mut after_attn_hc = rows_mut(&mut s.after_attn_hc, 0, n_rows, HC_DIM)?;
        let hc_split = rows(&s.hc_split, 0, n_rows, MIX_HC)?;
        b.hc_expand_split(
            &mut after_attn_hc,
            &s.attn_out,
            &s.cur_hc,
            &hc_split,
            N_EMBD,
            N_HC,
        )
    }

    /// The FFN half of support stage `stage` over the first `n_rows` rows of
    /// `ws`: the target's batched FFN on the support weights.
    fn encode_dspark_stage_ffn(
        &mut self,
        t: &Target<'_>,
        ws: &mut BatchWorkspace,
        stage: u32,
        pos0: u32,
        n_rows: u32,
    ) -> Result<()> {
        let d = t.dspark.ok_or(Error { op: "dspark stage" })?;
        let layer = &d.stages[stage as usize].block;
        // Support stages bind like the target's unhashed layers; a hash table
        // would route by unchecked expert IDs.
        if layer.ffn_gate_tid2eid.is_some() {
            return Err(Error { op: "dspark stage" });
        }
        let ffn = FfnLayer {
            layer,
            model: Model::Support,
            hash: None,
        };
        self.encode_layer_ffn_batch(t, ws, ffn, pos0, n_rows, None)
    }

    /// `rocm_graph_encode_dspark_stage`: attention, FFN, and the stream rows
    /// copied back (the support pass does not swap streams).
    fn encode_dspark_stage(
        &mut self,
        t: &Target<'_>,
        ws: &mut BatchWorkspace,
        stage: u32,
        pos0: u32,
        n_rows: u32,
    ) -> Result<()> {
        let cur = self.batch_cur_index;
        let own = Slots::own(ws, cur);
        self.encode_dspark_stage_attention(t, stage, pos0, n_rows, StageAttn::default(), own)?;
        self.encode_dspark_stage_ffn(t, ws, stage, pos0, n_rows)?;
        let (cur_hc, next_hc) = ws.hc_pair(cur);
        t.b.copy(cur_hc, 0, next_hc, 0, u64::from(n_rows) * HC_DIM * F32)
    }

    /// `rocm_graph_encode_dspark_final_hidden`: the last stage's head over
    /// the first `n_rows` stream rows of `ws` into its `hidden` rows from
    /// `hidden_row_offset`, ready for the tied target head.
    fn encode_dspark_final_hidden(
        &mut self,
        t: &Target<'_>,
        ws: &mut BatchWorkspace,
        n_rows: u32,
        hidden: Hidden,
        hidden_row_offset: u32,
    ) -> Result<()> {
        let b = t.b;
        let d = t.dspark.ok_or(Error { op: "dspark head" })?;
        let rows_cap = self.spec.as_ref().map_or(0, |s| s.rows_cap);
        if n_rows == 0 || n_rows > rows_cap {
            return Err(Error { op: "dspark head" });
        }
        let last = d.stages.last().ok_or(Error { op: "dspark head" })?;
        let rows_hc = rows(&ws.hc[self.batch_cur_index], 0, n_rows, HC_DIM)?;
        let mut head_pre = rows_mut(&mut ws.hc_mix, 0, n_rows, u64::from(N_HC))?;
        let mut head_weights = rows_mut(&mut ws.hc_split, 0, n_rows, u64::from(N_HC))?;
        let mut head_embd = rows_mut(&mut ws.ffn_cur, 0, n_rows, u64::from(N_EMBD))?;
        let hidden_out = match hidden {
            Hidden::FfnNorm => &mut ws.ffn_norm,
            Hidden::DsparkVerify => verify_streams(&mut ws.dspark_verify)?.1,
        };
        let mut head_norm = rows_mut(hidden_out, hidden_row_offset, n_rows, u64::from(N_EMBD))?;
        let mut flat_hc = ws.stage.flat_hc();
        b.rms_norm_plain_rows(&mut flat_hc, &rows_hc, HC_DIM as u32, n_rows, RMS_EPS)?;
        b.matmul_f16(
            &mut head_pre,
            sw(present(&last.hc_head_fn)?),
            HC_DIM,
            u64::from(N_HC),
            &flat_hc,
            u64::from(n_rows),
        )?;
        b.output_hc_weights(
            &mut head_weights,
            &head_pre,
            sw(present(&last.hc_head_scale)?),
            sw(present(&last.hc_head_base)?),
            N_HC,
            HC_EPS,
        )?;
        b.hc_weighted_sum(&mut head_embd, &rows_hc, &head_weights, N_EMBD, N_HC)?;
        b.rms_norm_weight_rows(
            &mut head_norm,
            &head_embd,
            sw(present(&last.norm)?),
            N_EMBD,
            n_rows,
            RMS_EPS,
        )
    }

    /// `rocm_graph_encode_dspark_logits`: the tied target head over
    /// `n_rows` support hidden rows into the verification logits.
    fn encode_dspark_logits(
        &mut self,
        t: &Target<'_>,
        hidden_rows: &impl Input,
        n_rows: u32,
    ) -> Result<()> {
        let spec = self.spec.as_mut().ok_or(Error {
            op: "dspark logits",
        })?;
        if n_rows == 0 || n_rows > spec.rows_cap {
            return Err(Error {
                op: "dspark logits",
            });
        }
        t.b.matmul_q8_0(
            &mut spec.logits,
            w(&t.weights.output),
            u64::from(N_EMBD),
            t.weights.output.dim[1],
            hidden_rows,
            u64::from(n_rows),
        )
    }

    /// `rocm_graph_dspark_select_block`: the greedy Markov trajectory of
    /// `n_rows` verification logits rows after `target_next_token`.
    fn dspark_select_block(
        &mut self,
        t: &Target<'_>,
        target_next_token: i32,
        n_rows: u32,
    ) -> Result<Vec<i32>> {
        let b = t.b;
        let d = t.dspark.ok_or(Error {
            op: "dspark select",
        })?;
        let logits_rows = &self
            .spec
            .as_ref()
            .ok_or(Error { op: "dspark draft" })?
            .logits;
        let ds = self.dspark.as_mut().ok_or(Error {
            op: "dspark select",
        })?;
        let last = d.stages.last().ok_or(Error {
            op: "dspark select",
        })?;
        let (w1, w2) = (sw(present(&last.markov_w1)?), sw(present(&last.markov_w2)?));
        let mut tokens = Vec::with_capacity(n_rows as usize);
        let mut previous = target_next_token;
        for row in 0..n_rows {
            if previous < 0 || previous as u32 >= N_VOCAB {
                return Err(Error {
                    op: "dspark select",
                });
            }
            let logits_row = rows(logits_rows, row, 1, u64::from(N_VOCAB))?;
            b.dspark_markov_argmax(
                &mut ds.markov_index,
                &mut ds.markov_key,
                &logits_row,
                w1,
                w2,
                N_VOCAB,
                d.markov_rank,
                previous as u32,
            )?;
            let mut selected = [0u8; 4];
            b.read(&ds.markov_index, 0, &mut selected)?;
            let selected = i32::from_le_bytes(selected);
            if selected < 0 || selected as u32 >= N_VOCAB {
                return Err(Error {
                    op: "dspark select",
                });
            }
            tokens.push(selected);
            previous = selected;
        }
        Ok(tokens)
    }

    /// `rocm_graph_dspark_prepare_hidden`: one support pass over
    /// [`target_next_token`, noise, ...] at `pos0`, its final hidden rows
    /// into the `hidden` rows of `ws`.
    fn dspark_prepare_hidden(
        &mut self,
        t: &Target<'_>,
        ws: &mut BatchWorkspace,
        target_next_token: i32,
        pos0: u32,
        hidden: Hidden,
    ) -> Result<()> {
        let b = t.b;
        let d = t.dspark.ok_or(Error { op: "dspark draft" })?;
        let ds = self.dspark.as_ref().ok_or(Error { op: "dspark draft" })?;
        let block = d.block_size;
        let rows_cap = self.spec.as_ref().map_or(0, |s| s.rows_cap);
        if block == 0 || block > rows_cap || block > self.prefill_cap {
            return Err(Error { op: "dspark draft" });
        }
        if pos0 == 0 || ds.context_len < pos0 || block > ds.cache_cap {
            return Err(Error { op: "dspark draft" });
        }
        let mut tokens = vec![d.noise_token_id as i32; block as usize];
        tokens[0] = target_next_token;
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        b.write(&mut ws.prefill_tokens, 0, &bytes)?;
        let result = (|| {
            b.begin_commands()?;
            let embd = &t.weights.token_embd;
            b.embed_tokens_hc(
                &mut ws.hc[self.batch_cur_index],
                &ws.prefill_tokens,
                w(embd),
                N_VOCAB,
                block,
                N_EMBD,
                N_HC,
            )?;
            for stage in 0..d.stages.len() as u32 {
                self.encode_dspark_stage(t, ws, stage, pos0, block)?;
            }
            self.encode_dspark_final_hidden(t, ws, block, hidden, 0)?;
            b.end_commands()
        })();
        if result.is_err() {
            let _ = b.synchronize();
        }
        result
    }

    /// `rocm_graph_dspark_draft`, greedy: the support model's block after
    /// `target_next_token` at `pos0`, through the graph's workspace `ws`;
    /// the caller keeps the prefix it verifies.
    pub(crate) fn dspark_draft(
        &mut self,
        t: &Target<'_>,
        ws: &mut BatchWorkspace,
        target_next_token: i32,
        pos0: u32,
        sampler: Option<&mut DsparkSampler>,
        max_tokens: u32,
        concurrency: usize,
    ) -> Result<Vec<i32>> {
        let b = t.b;
        let d = t.dspark.ok_or(Error { op: "dspark draft" })?;
        let block = d.block_size;
        self.check_workspace(ws)?;
        // Narrow-batch routing for the support model without the verifier's
        // arithmetic candidates.
        b.set_batch_mode(BatchMode::Support);
        let result = (|| {
            self.dspark_prepare_hidden(t, ws, target_next_token, pos0, Hidden::FfnNorm)?;
            let logits = b
                .begin_commands()
                .and_then(|()| self.encode_dspark_logits(t, &ws.ffn_norm, block));
            match logits {
                Ok(()) => b.end_commands()?,
                Err(error) => {
                    let _ = b.synchronize();
                    return Err(error);
                }
            }
            if proposes(&sampler) {
                self.ensure_dspark_candidates(b)?;
                let mut requests = [SampledRequest {
                    target_next_token,
                    position: pos0,
                    max_draft_tokens: max_tokens,
                    concurrency,
                    sampler,
                    tokens: Vec::new(),
                }];
                select_sampled(t, self, ws, &mut requests, block)?;
                return Ok(std::mem::take(&mut requests[0].tokens));
            }
            self.dspark_select_block(t, target_next_token, block)
        })();
        b.set_batch_mode(BatchMode::Off);
        result
    }
}

/// The batch tensors of a request's projected support attention in a
/// batched draft: rows `row0 .. row0 + n` of the coordinator's streams,
/// queries and keys, and the request's own other tensors, of `own` or of
/// the coordinator's workspace when the request shares it.
fn projected_slots<'a>(
    coordinator: &'a mut BatchWorkspace,
    own: Option<&'a mut BatchWorkspace>,
    cur: usize,
    row0: u32,
    n: u32,
) -> Result<Slots<'a>> {
    let embd = u64::from(N_EMBD);
    let head_dim = u64::from(N_HEAD_DIM);
    match own {
        Some(own) => {
            let rows = Slots::own(coordinator, cur);
            Ok(Slots {
                cur_hc: rows_mut(rows.cur_hc, row0, n, HC_DIM)?,
                scratch: rows.scratch.rows(row0, n),
                hc_split: rows_mut(rows.hc_split, row0, n, MIX_HC)?,
                attn_norm: rows_mut(rows.attn_norm, row0, n, embd)?,
                kv: rows_mut(rows.kv, row0, n, head_dim)?,
                ..Slots::own(own, cur)
            })
        }
        None => {
            let own = Slots::own(coordinator, cur);
            Ok(Slots {
                cur_hc: rows_mut(own.cur_hc, row0, n, HC_DIM)?,
                scratch: own.scratch.rows(row0, n),
                hc_split: rows_mut(own.hc_split, row0, n, MIX_HC)?,
                attn_norm: rows_mut(own.attn_norm, row0, n, embd)?,
                kv: rows_mut(own.kv, row0, n, head_dim)?,
                ..own
            })
        }
    }
}

/// `rocm_graph_dspark_prepare_hidden_batch`: one support pass for every
/// request's block, its final hidden rows into the second DSpark
/// verification stream. Each request's stateful attention runs on its own
/// session's ring with its rows staged in the coordinator's workspace of
/// `workspaces`; the stateless FFN runs once over all rows, one router group
/// per request.
fn prepare_hidden_batch(
    t: &Target<'_>,
    workspaces: &mut [BatchWorkspace],
    items: &mut [DraftItem<'_>],
    block: u32,
) -> Result<()> {
    let b = t.b;
    let d = t.dspark.ok_or(Error { op: "dspark batch" })?;
    let count = items.len();
    let coordinator_ws = lent(workspaces, &items[0].graph.workspace)?;
    let own_ws = items
        .iter()
        .map(|item| lent(workspaces, &item.graph.workspace))
        .collect::<Result<Vec<_>>>()?;
    let coordinator = &items[0].graph;
    let verify_rows_cap = match &workspaces[coordinator_ws].dspark_verify {
        Some((_, _, rows)) => *rows,
        None => return Err(Error { op: "dspark batch" }),
    };
    let rows_per = block;
    let total = count as u32 * block;
    if block == 0
        || block > d.block_size
        || total > coordinator.prefill_cap
        || total > verify_rows_cap
        || total > super::batch::SPEC_MAX_ROWS
    {
        return Err(Error { op: "dspark batch" });
    }
    let mut flat = vec![d.noise_token_id as i32; total as usize];
    for (index, item) in items.iter().enumerate() {
        let g = &item.graph;
        let context = g.dspark_context_len();
        let cache_cap = g.dspark.as_ref().map_or(0, |ds| ds.cache_cap);
        if g.dspark.is_none()
            || item.position == 0
            || context < item.position
            || rows_per > g.prefill_cap
            || rows_per > cache_cap
        {
            return Err(Error { op: "dspark batch" });
        }
        flat[index * block as usize] = item.target_next_token;
    }
    let bytes: Vec<u8> = flat.iter().flat_map(|t| t.to_le_bytes()).collect();
    b.write(&mut workspaces[coordinator_ws].prefill_tokens, 0, &bytes)?;
    let hc_row = HC_DIM * F32;
    let combine_attn_q8 = count >= 4 && count.is_multiple_of(2) && rows_per <= 6;
    let offsets: Vec<u32> = (0..=count as u32).map(|i| i * rows_per).collect();
    let result = (|| {
        b.begin_commands()?;
        let ws = &mut workspaces[coordinator_ws];
        let embd = &t.weights.token_embd;
        let (verify_cur_hc, _) = verify_streams(&mut ws.dspark_verify)?;
        b.embed_tokens_hc(
            verify_cur_hc,
            &ws.prefill_tokens,
            w(embd),
            N_VOCAB,
            total,
            N_EMBD,
            N_HC,
        )?;
        for stage in 0..d.stages.len() as u32 {
            let layer = &d.stages[stage as usize].block;
            if combine_attn_q8 {
                let q_rank = layer.attn_q_a.dim[1];
                let cur = items[0].graph.batch_cur_index;
                for (index, item) in items.iter_mut().enumerate() {
                    let row0 = offsets[index];
                    let block_offset = index as u64 * u64::from(block) * hc_row;
                    let ws = &mut workspaces[coordinator_ws];
                    let (verify_cur_hc, _) = verify_streams(&mut ws.dspark_verify)?;
                    b.copy(
                        &mut ws.hc[cur],
                        u64::from(row0) * hc_row,
                        verify_cur_hc,
                        block_offset,
                        u64::from(block) * hc_row,
                    )?;
                    // The front reads and writes only the coordinator's rows.
                    let slots =
                        Slots::own(ws, cur).rows(row0, rows_per, q_rank, u64::from(N_HEAD_DIM))?;
                    let front = StageAttn {
                        front_ready: false,
                        front_only: true,
                        project_qb: false,
                        project_q_a_kv: false,
                    };
                    item.graph.encode_dspark_stage_attention(
                        t,
                        stage,
                        item.position,
                        rows_per,
                        front,
                        slots,
                    )?;
                }
                let ws = &mut workspaces[coordinator_ws];
                b.matmul_q8_0_group_pairs(
                    &mut ws.qr,
                    sw(&layer.attn_q_a),
                    u64::from(N_EMBD),
                    q_rank,
                    &ws.attn_norm,
                    u64::from(total),
                    &offsets,
                )?;
                b.matmul_q8_0_group_pairs(
                    &mut ws.kv_raw,
                    sw(&layer.attn_kv),
                    u64::from(N_EMBD),
                    u64::from(N_HEAD_DIM),
                    &ws.attn_norm,
                    u64::from(total),
                    &offsets,
                )?;
                b.qkv_rms_norm_rows(
                    &mut ws.qr_norm,
                    &ws.qr,
                    sw(&layer.attn_q_a_norm),
                    q_rank as u32,
                    &mut ws.kv,
                    &ws.kv_raw,
                    sw(&layer.attn_kv_a_norm),
                    N_HEAD_DIM,
                    total,
                    RMS_EPS,
                )?;
                let (mut q, _) = ws.stage.attention();
                b.matmul_q8_0_group_pairs(
                    &mut q,
                    sw(&layer.attn_q_b),
                    q_rank,
                    Q_DIM,
                    &ws.qr_norm,
                    u64::from(total),
                    &offsets,
                )?;
                for (index, item) in items.iter_mut().enumerate() {
                    let row0 = offsets[index];
                    let block_offset = index as u64 * u64::from(block) * hc_row;
                    let (ws, mut own) = lent_pair(workspaces, coordinator_ws, own_ws[index]);
                    let slots = projected_slots(ws, own.as_deref_mut(), cur, row0, rows_per)?;
                    let projected = StageAttn {
                        front_ready: true,
                        front_only: false,
                        project_qb: false,
                        project_q_a_kv: false,
                    };
                    item.graph.encode_dspark_stage_attention(
                        t,
                        stage,
                        item.position,
                        rows_per,
                        projected,
                        slots,
                    )?;
                    let (_, verify_after_attn_hc) = verify_streams(&mut ws.dspark_verify)?;
                    let source = match own.as_deref() {
                        Some(own) => &own.after_attn_hc,
                        None => &ws.after_attn_hc,
                    };
                    b.copy(
                        verify_after_attn_hc,
                        block_offset,
                        source,
                        0,
                        u64::from(block) * hc_row,
                    )?;
                }
            } else {
                for (index, item) in items.iter_mut().enumerate() {
                    let block_offset = index as u64 * u64::from(block) * hc_row;
                    let item_cur = item.graph.batch_cur_index;
                    let (ws, mut own) = lent_pair(workspaces, coordinator_ws, own_ws[index]);
                    let (verify_cur_hc, _) = verify_streams(&mut ws.dspark_verify)?;
                    let item_hc = match own.as_deref_mut() {
                        Some(own) => &mut own.hc[item_cur],
                        None => &mut ws.hc[item_cur],
                    };
                    b.copy(
                        item_hc,
                        0,
                        verify_cur_hc,
                        block_offset,
                        u64::from(block) * hc_row,
                    )?;
                    let slots = match own.as_deref_mut() {
                        Some(own) => Slots::own(own, item_cur),
                        None => Slots::own(ws, item_cur),
                    };
                    item.graph.encode_dspark_stage_attention(
                        t,
                        stage,
                        item.position,
                        rows_per,
                        StageAttn::default(),
                        slots,
                    )?;
                    let (_, verify_after_attn_hc) = verify_streams(&mut ws.dspark_verify)?;
                    let source = match own.as_deref() {
                        Some(own) => &own.after_attn_hc,
                        None => &ws.after_attn_hc,
                    };
                    b.copy(
                        verify_after_attn_hc,
                        block_offset,
                        source,
                        0,
                        u64::from(block) * hc_row,
                    )?;
                }
            }
            let ws = &mut workspaces[coordinator_ws];
            let (_, verify_after_attn_hc) = verify_streams(&mut ws.dspark_verify)?;
            b.copy(
                &mut ws.after_attn_hc,
                0,
                verify_after_attn_hc,
                0,
                u64::from(total) * hc_row,
            )?;
            let position = items[0].position;
            let coordinator = &mut *items[0].graph;
            coordinator.encode_dspark_stage_ffn(t, ws, stage, position, total)?;
            let (verify_cur_hc, _) = verify_streams(&mut ws.dspark_verify)?;
            b.copy(
                verify_cur_hc,
                0,
                &ws.hc[1 - coordinator.batch_cur_index],
                0,
                u64::from(total) * hc_row,
            )?;
        }
        let ws = &mut workspaces[coordinator_ws];
        let coordinator = &mut *items[0].graph;
        let (verify_cur_hc, _) = verify_streams(&mut ws.dspark_verify)?;
        b.copy(
            &mut ws.hc[coordinator.batch_cur_index],
            0,
            verify_cur_hc,
            0,
            u64::from(total) * hc_row,
        )?;
        coordinator.encode_dspark_final_hidden(t, ws, total, Hidden::DsparkVerify, 0)?;
        b.end_commands()
    })();
    if result.is_err() {
        let _ = b.synchronize();
    }
    result
}

/// `rocm_graph_dspark_select_block_batch`: the greedy Markov trajectories
/// of 2, 4, 6 or 8 requests, `rows_per_item` logits rows each; `ws` is the
/// coordinator's workspace.
fn select_block_batch(
    t: &Target<'_>,
    ws: &mut BatchWorkspace,
    items: &mut [DraftItem<'_>],
    logits_rows: &impl Input,
    rows_per_item: u32,
) -> Result<()> {
    let b = t.b;
    let d = t.dspark.ok_or(Error {
        op: "dspark select",
    })?;
    let count = items.len();
    if rows_per_item == 0 || !matches!(count, 2 | 4 | 6 | 8) {
        return Err(Error {
            op: "dspark select",
        });
    }
    let last = d.stages.last().ok_or(Error {
        op: "dspark select",
    })?;
    let (w1, w2) = (sw(present(&last.markov_w1)?), sw(present(&last.markov_w2)?));
    let mut previous = Vec::with_capacity(count);
    let mut limits = Vec::with_capacity(count);
    for item in items.iter() {
        if item.max_draft_tokens == 0
            || item.target_next_token < 0
            || item.target_next_token as u32 >= N_VOCAB
        {
            return Err(Error {
                op: "dspark select",
            });
        }
        previous.push(item.target_next_token);
        limits.push(item.max_draft_tokens.min(rows_per_item));
    }
    let mut active = vec![true; count];
    let mut selected_rows = vec![0u32; count];
    let mut tokens = vec![Vec::new(); count];
    let vocab_row = u64::from(N_VOCAB) * F32;
    let stride = u64::from(rows_per_item) * u64::from(N_VOCAB);
    let ds = items[0].graph.dspark.as_mut().ok_or(Error {
        op: "dspark select",
    })?;
    for row in 0..rows_per_item {
        let span = ((count as u64 - 1) * u64::from(rows_per_item) + 1) * vocab_row;
        let view = logits_rows
            .view(u64::from(row) * vocab_row, span)
            .ok_or(Error {
                op: "dspark select",
            })?;
        b.dspark_markov_argmax_batch(
            &mut ds.markov_index,
            &mut ds.markov_key,
            &view,
            &previous,
            &mut ws.prefill_tokens,
            w1,
            w2,
            N_VOCAB,
            d.markov_rank,
            stride,
        )?;
        let mut bytes = vec![0u8; count * 4];
        b.read(&ds.markov_index, 0, &mut bytes)?;
        for index in 0..count {
            let selected = i32::from_le_bytes([
                bytes[4 * index],
                bytes[4 * index + 1],
                bytes[4 * index + 2],
                bytes[4 * index + 3],
            ]);
            if selected < 0 || selected as u32 >= N_VOCAB {
                return Err(Error {
                    op: "dspark select",
                });
            }
            if !active[index] || row >= limits[index] {
                active[index] = false;
                continue;
            }
            tokens[index].push(selected);
            previous[index] = selected;
            selected_rows[index] = row + 1;
        }
    }
    for (item, (tokens, rows)) in items.iter_mut().zip(tokens.into_iter().zip(selected_rows)) {
        debug_assert_eq!(tokens.len(), rows as usize);
        item.tokens = tokens;
    }
    Ok(())
}

/// `rocm_graph_dspark_draft_head_batch`, greedy: each request drafts with
/// its own support attention and ring; the tied target head runs once over
/// all hidden rows, which the first request's workspace of `workspaces`
/// holds.
pub fn dspark_draft_head_batch(
    t: &Target<'_>,
    workspaces: &mut [BatchWorkspace],
    items: &mut [DraftItem<'_>],
) -> Result<()> {
    let b = t.b;
    let count = items.len();
    if !(2..=8).contains(&count) {
        return Err(Error { op: "dspark batch" });
    }
    let coordinator_ws = lent(workspaces, &items[0].graph.workspace)?;
    let block = items[0].graph.dspark_block_size();
    let verify_rows_cap = workspaces[coordinator_ws]
        .dspark_verify
        .as_ref()
        .map_or(0, |v| v.2);
    if block == 0 {
        return Err(Error { op: "dspark batch" });
    }
    let requested = items
        .iter()
        .map(|i| block.min(i.max_draft_tokens))
        .max()
        .unwrap_or(0);
    let total = count as u32 * requested;
    if requested == 0
        || total > items[0].graph.prefill_cap
        || total > verify_rows_cap
        || total > super::batch::SPEC_MAX_ROWS
    {
        return Err(Error { op: "dspark batch" });
    }
    items[0].graph.spec_prepare(b, t.weights, total)?;
    for item in items.iter_mut() {
        if item.graph.dspark_block_size() != block {
            return Err(Error { op: "dspark batch" });
        }
        item.tokens.clear();
    }
    b.set_batch_mode(BatchMode::Support);
    let result = (|| {
        prepare_hidden_batch(t, workspaces, items, requested)?;
        let ws = &mut workspaces[coordinator_ws];
        let hidden = &ws
            .dspark_verify
            .as_ref()
            .ok_or(Error { op: "dspark batch" })?
            .1;
        let coordinator = &mut *items[0].graph;
        let logits = b
            .begin_commands()
            .and_then(|()| coordinator.encode_dspark_logits(t, hidden, total));
        match logits {
            Ok(()) => b.end_commands()?,
            Err(error) => {
                let _ = b.synchronize();
                return Err(error);
            }
        }
        if items.iter().any(|item| proposes(&item.sampler)) {
            items[0].graph.ensure_dspark_candidates(b)?;
            let (head, tail) = items
                .split_first_mut()
                .ok_or(Error { op: "dspark batch" })?;
            let coordinator: &mut Graph = &mut *head.graph;
            let mut requests = vec![SampledRequest {
                target_next_token: head.target_next_token,
                position: head.position,
                max_draft_tokens: head.max_draft_tokens,
                concurrency: head.concurrency,
                sampler: head.sampler.as_deref_mut(),
                tokens: Vec::new(),
            }];
            requests.extend(tail.iter_mut().map(|item| SampledRequest {
                target_next_token: item.target_next_token,
                position: item.position,
                max_draft_tokens: item.max_draft_tokens,
                concurrency: item.concurrency,
                sampler: item.sampler.as_deref_mut(),
                tokens: Vec::new(),
            }));
            select_sampled(t, coordinator, ws, &mut requests, requested)?;
            let tokens: Vec<Vec<i32>> = requests.into_iter().map(|r| r.tokens).collect();
            for (item, tokens) in items.iter_mut().zip(tokens) {
                item.tokens = tokens;
            }
            return Ok(());
        }
        let spec = items[0]
            .graph
            .spec
            .take()
            .ok_or(Error { op: "dspark batch" })?;
        let selected = select_block_batch(t, ws, items, &spec.logits, requested);
        items[0].graph.spec = Some(spec);
        selected
    })();
    b.set_batch_mode(BatchMode::Off);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rows the attention kernel reads (`check_dspark_window` in
    /// `tests/models/deepseek_v4_flash/attention_test.hip`, whose span helper
    /// mirrors this one): the last 128 committed target rows followed by
    /// every draft row, in ring order, built independently of the helper.
    #[test]
    fn the_visible_span_is_the_official_window_in_ring_order() {
        let cap = 4352;
        for prefix in [1u32, 127, 128, 129, 4096, 4351, 4352, 8707] {
            for block in [1u32, 5] {
                let first = prefix.saturating_sub(128);
                let official: Vec<u32> = (first..prefix + block).map(|pos| pos % cap).collect();
                let (start, count) = visible_span(prefix, block, cap);
                let rows: Vec<u32> = (0..count).map(|i| (start + i) % cap).collect();
                assert_eq!(rows, official, "prefix {prefix}, block {block}");
            }
        }
        assert_eq!(visible_span(10, 0, cap), (0, 0), "an empty block");
        assert_eq!(
            visible_span(10, cap + 1, cap),
            (0, 0),
            "a block beyond the ring"
        );
        assert_eq!(
            visible_span(u32::MAX, 1, cap),
            (0, 0),
            "a position overflow"
        );
        assert_eq!(
            visible_span(200, 4300, cap),
            (0, 0),
            "a window beyond the ring"
        );
    }
}
