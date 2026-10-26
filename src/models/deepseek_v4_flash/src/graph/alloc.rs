//! Graph state and its allocation (`rocm_graph_alloc_raw_cap`,
//! `ds4_rocm_graph_create`, the batch workspace and the capacity policy).

use std::sync::Arc;

use gufo_deepseek_v4_flash_kernels::{
    Backend, DeviceTensor, Error, ExpertSelection, Result, TensorViewMut,
};

use super::*;
use crate::weights::Weights;

// The normalized streams fit in the query half of the stage scratch.
const _: () = assert!(HC_DIM <= Q_DIM);

/// The prompt chunk of a context (`ds4_default_prefill_cap_for_prompt`):
/// whole prompts up to 4,096 tokens, longer ones in 4,096-token chunks.
pub fn prefill_capacity(context: u32) -> u32 {
    context.clamp(1, 4096)
}

/// Suffixes at least this long extend a cached prefix with batched prefill
/// (`rocm_graph_resume_prefill_min_tokens`).
pub fn resume_prefill_min_tokens() -> u32 {
    4
}

/// Raw SWA ring rows (`rocm_graph_raw_cap_for_context`): the window plus
/// one prompt chunk, at most the context, padded to 256 rows and at most
/// 8,192.
pub fn raw_cap_for_context(context: u32, prefill_cap: u32) -> u32 {
    let raw_window = N_SWA.min(context).max(1);
    let mut wanted = u64::from(raw_window) + u64::from(prefill_cap);
    wanted = wanted.min(u64::from(context)).max(1);
    wanted = crate::gguf::align_up(wanted, 256).min(8192);
    (wanted as u32).max(raw_window)
}

/// A session's claim on the engine's batch workspace it uses: the workspace
/// keeps one token and every graph that borrows it a clone, so the count
/// tells which workspaces sessions still use, and the engine lends a
/// session the workspace its lease names for each call.
#[derive(Debug)]
pub struct WorkspaceLease(Arc<()>);

/// The stage scratch of a batch workspace, whose ranges the stages reuse in
/// turn; each stage borrows its views exclusively.
pub(crate) struct StageScratch {
    tensor: DeviceTensor,
    q_bytes: u64,
    hc_bytes: u64,
    shared_mid_bytes: u64,
    shared_out_bytes: u64,
    routed_mid_bytes: u64,
    routed_down_bytes: u64,
    down_offset: u64,
}

/// The routed-expert stage's views of the stage scratch.
pub(crate) struct RoutedScratch<'a> {
    pub gate: TensorViewMut<'a>,
    pub up: TensorViewMut<'a>,
    pub mid: TensorViewMut<'a>,
    pub down: TensorViewMut<'a>,
}

/// The shared-expert stage's views of the stage scratch, and the routed
/// down rows, which survive into it.
pub(crate) struct SharedScratch<'a> {
    pub gate: TensorViewMut<'a>,
    pub up: TensorViewMut<'a>,
    pub mid: TensorViewMut<'a>,
    pub out: TensorViewMut<'a>,
    pub routed_down: TensorViewMut<'a>,
}

/// The attention stage's normalized streams and then its queries, which
/// share the query half of the stage scratch, so each phase borrows its
/// view in turn; all rows, or a coordinator's rows of one request.
pub(crate) struct ScratchRows<'a> {
    q: TensorViewMut<'a>,
    hc_bytes: u64,
    rows: Option<(u32, u32)>,
}

fn carve<T>(view: Option<T>) -> T {
    view.unwrap_or_else(|| unreachable!("stage scratch ranges lie inside it"))
}

/// Rows `row0 .. row0 + n` of `row_floats` floats of `t`, exclusively.
pub(crate) fn rows_mut<'a>(
    t: impl Into<TensorViewMut<'a>>,
    row0: u32,
    n: u32,
    row_floats: u64,
) -> Result<TensorViewMut<'a>> {
    t.into()
        .view_mut(
            u64::from(row0) * row_floats * F32,
            u64::from(n) * row_floats * F32,
        )
        .ok_or(Error { op: "batch view" })
}

impl StageScratch {
    /// `batch_flat_hc`: the normalized streams.
    pub(crate) fn flat_hc(&mut self) -> TensorViewMut<'_> {
        carve(self.tensor.view_mut(0, self.hc_bytes))
    }

    /// `batch_q` and `batch_heads`; indexer scoring borrows the heads for
    /// packed keys and queries before attention writes them.
    pub(crate) fn attention(&mut self) -> (TensorViewMut<'_>, TensorViewMut<'_>) {
        let (q, rest) = carve(self.tensor.split_at_mut(self.q_bytes));
        (q, carve(rest.view_mut(0, self.q_bytes)))
    }

    /// The attention stage's streams and queries, and its heads, all rows.
    pub(crate) fn split(&mut self) -> (ScratchRows<'_>, TensorViewMut<'_>) {
        let hc_bytes = self.hc_bytes;
        let (q, heads) = self.attention();
        let rows = ScratchRows {
            q,
            hc_bytes,
            rows: None,
        };
        (rows, heads)
    }

    /// The routed MoE's gate, up and mid rows from the start, its down rows
    /// beyond both sets of FFN intermediates.
    pub(crate) fn routed(&mut self) -> RoutedScratch<'_> {
        let mid = self.routed_mid_bytes;
        let routed_down = self.routed_down_bytes;
        let (mids, rest) = carve(self.tensor.split_at_mut(self.down_offset));
        let (gate, mids) = carve(mids.split_at_mut(mid));
        let (up, mids) = carve(mids.split_at_mut(mid));
        RoutedScratch {
            gate,
            up,
            mid: carve(mids.view_mut(0, mid)),
            down: carve(rest.view_mut(0, routed_down)),
        }
    }

    /// The shared expert's gate, up, mid and output rows from the start, and
    /// the routed down rows.
    pub(crate) fn shared(&mut self) -> SharedScratch<'_> {
        let mid = self.shared_mid_bytes;
        let out = self.shared_out_bytes;
        let routed_down = self.routed_down_bytes;
        let (mids, rest) = carve(self.tensor.split_at_mut(self.down_offset));
        let (gate, mids) = carve(mids.split_at_mut(mid));
        let (up, mids) = carve(mids.split_at_mut(mid));
        let (mid, mids) = carve(mids.split_at_mut(mid));
        SharedScratch {
            gate,
            up,
            mid,
            out: carve(mids.view_mut(0, out)),
            routed_down: carve(rest.view_mut(0, routed_down)),
        }
    }
}

impl<'a> ScratchRows<'a> {
    /// Rows `row0 .. row0 + n` of all rows.
    pub(crate) fn rows(self, row0: u32, n: u32) -> Self {
        debug_assert!(self.rows.is_none());
        Self {
            rows: Some((row0, n)),
            ..self
        }
    }

    /// `batch_flat_hc`.
    pub(crate) fn flat_hc(&mut self) -> Result<TensorViewMut<'_>> {
        let all = carve(self.q.reborrow().view_mut(0, self.hc_bytes));
        match self.rows {
            Some((row0, n)) => rows_mut(all, row0, n, HC_DIM),
            None => Ok(all),
        }
    }

    /// `batch_q`.
    pub(crate) fn q(&mut self) -> Result<TensorViewMut<'_>> {
        match self.rows {
            Some((row0, n)) => rows_mut(&mut self.q, row0, n, Q_DIM),
            None => Ok(self.q.reborrow()),
        }
    }
}

/// The prompt-chunk tensors sessions share (the `batch_*` set of C++): one
/// stage scratch whose ranges the stages reuse in turn, the hyper-connection
/// streams, projections and router state of `prefill_cap` rows.
pub struct BatchWorkspace {
    pub prefill_cap: u32,
    /// The token the leases of sessions clone.
    lease: WorkspaceLease,
    pub(crate) stage: StageScratch,
    pub prefill_tokens: DeviceTensor,
    /// The two stream buffers; each graph swaps which is current.
    pub(crate) hc: [DeviceTensor; 2],
    pub hc_mix: DeviceTensor,
    pub hc_split: DeviceTensor,
    pub attn_cur: DeviceTensor,
    pub attn_norm: DeviceTensor,
    pub qr: DeviceTensor,
    pub qr_norm: DeviceTensor,
    pub kv_raw: DeviceTensor,
    pub kv: DeviceTensor,
    pub comp_kv: DeviceTensor,
    pub comp_sc: DeviceTensor,
    pub indexer_q: DeviceTensor,
    pub indexer_weights: DeviceTensor,
    pub attn_low: DeviceTensor,
    pub attn_out: DeviceTensor,
    pub after_attn_hc: DeviceTensor,
    pub ffn_cur: DeviceTensor,
    pub ffn_norm: DeviceTensor,
    pub router_logits: DeviceTensor,
    pub router_probs: DeviceTensor,
    pub router: ExpertSelection,
    pub routed_out: DeviceTensor,
    /// The DSpark verification streams (eight blocks), when the engine has
    /// a support model.
    pub dspark_verify: Option<(DeviceTensor, DeviceTensor, u32)>,
}

impl BatchWorkspace {
    fn alloc(b: &Backend, weights: &Weights, prefill_cap: u32) -> Result<Self> {
        let layer = &weights.layers[0];
        let pc = u64::from(prefill_cap);
        let q_rank = layer.attn_q_a.dim[1];
        let shared_dim = layer.ffn_gate_shexp.dim[1];
        let routed_mid_dim = layer.ffn_gate_exps.dim[1];
        let comp_width_max = 2 * u64::from(N_HEAD_DIM.max(N_INDEXER_HEAD_DIM));
        let embd = u64::from(N_EMBD);
        // Stages run in order, including across grouped requests: HC norm,
        // attention Q and heads, routed MoE, shared expert. Only routed down
        // survives into the shared-expert stage, beyond both sets of FFN
        // intermediates.
        let q_bytes = pc * Q_DIM * F32;
        let hc_bytes = pc * HC_DIM * F32;
        let routed_mid_bytes = pc * u64::from(N_EXPERT_USED) * routed_mid_dim * F32;
        let routed_down_bytes = pc * u64::from(N_EXPERT_USED) * embd * F32;
        let shared_mid_bytes = pc * shared_dim * F32;
        let shared_out_bytes = pc * embd * F32;
        let down_offset = (3 * routed_mid_bytes).max(3 * shared_mid_bytes + shared_out_bytes);
        let stage_scratch = b.alloc(
            (2 * q_bytes)
                .max(hc_bytes)
                .max(down_offset + routed_down_bytes),
        )?;
        Ok(Self {
            prefill_cap,
            lease: WorkspaceLease(Arc::new(())),
            stage: StageScratch {
                tensor: stage_scratch,
                q_bytes,
                hc_bytes,
                shared_mid_bytes,
                shared_out_bytes,
                routed_mid_bytes,
                routed_down_bytes,
                down_offset,
            },
            prefill_tokens: b.alloc(pc * 4)?,
            hc: [b.alloc(pc * HC_DIM * F32)?, b.alloc(pc * HC_DIM * F32)?],
            hc_mix: b.alloc(pc * MIX_HC * F32)?,
            hc_split: b.alloc(pc * MIX_HC * F32)?,
            attn_cur: b.alloc(pc * embd * F32)?,
            attn_norm: b.alloc(pc * embd * F32)?,
            qr: b.alloc(pc * q_rank * F32)?,
            qr_norm: b.alloc(pc * q_rank * F32)?,
            kv_raw: b.alloc(pc * u64::from(N_HEAD_DIM) * F32)?,
            kv: b.alloc(pc * u64::from(N_HEAD_DIM) * F32)?,
            comp_kv: b.alloc(pc * comp_width_max * F32)?,
            comp_sc: b.alloc(pc * comp_width_max * F32)?,
            indexer_q: b.alloc(pc * INDEXER_Q_DIM * F32)?,
            indexer_weights: b.alloc(pc * u64::from(N_INDEXER_HEAD) * F32)?,
            attn_low: b.alloc(pc * LOW_DIM * F32)?,
            attn_out: b.alloc(pc * embd * F32)?,
            after_attn_hc: b.alloc(pc * HC_DIM * F32)?,
            ffn_cur: b.alloc(pc * embd * F32)?,
            ffn_norm: b.alloc(pc * embd * F32)?,
            router_logits: b.alloc(pc * u64::from(N_EXPERT) * F32)?,
            router_probs: b.alloc(pc * u64::from(N_EXPERT) * F32)?,
            router: b.alloc_expert_selection(prefill_cap)?,
            routed_out: b.alloc(pc * embd * F32)?,
            dspark_verify: None,
        })
    }

    /// Stream buffer `cur` and the other one.
    pub(crate) fn hc_pair(&mut self, cur: usize) -> (&mut DeviceTensor, &mut DeviceTensor) {
        let [first, second] = &mut self.hc;
        if cur == 0 {
            (first, second)
        } else {
            (second, first)
        }
    }

    /// `lease` names this workspace.
    pub(crate) fn lent_by(&self, lease: &WorkspaceLease) -> bool {
        Arc::ptr_eq(&self.lease.0, &lease.0)
    }
}

/// The DSpark verification streams of `dspark_verify`: the current one and
/// the one after attention.
pub(crate) fn verify_streams(
    dspark_verify: &mut Option<(DeviceTensor, DeviceTensor, u32)>,
) -> Result<(&mut DeviceTensor, &mut DeviceTensor)> {
    match dspark_verify {
        Some((cur, after, _)) => Ok((cur, after)),
        None => Err(Error {
            op: "dspark verify streams",
        }),
    }
}

/// The workspace of `workspaces` that `lease` names.
pub(crate) fn lent(workspaces: &[BatchWorkspace], lease: &WorkspaceLease) -> Result<usize> {
    workspaces
        .iter()
        .position(|ws| ws.lent_by(lease))
        .ok_or(Error {
            op: "batch workspace",
        })
}

/// Workspace `coordinator` of `workspaces` and workspace `own` of a request
/// of its batch when it is another one, exclusively.
pub(crate) fn lent_pair(
    workspaces: &mut [BatchWorkspace],
    coordinator: usize,
    own: usize,
) -> (&mut BatchWorkspace, Option<&mut BatchWorkspace>) {
    if coordinator == own {
        return (&mut workspaces[own], None);
    }
    let (low, high) = workspaces.split_at_mut(coordinator.max(own));
    let (first, second) = (&mut low[coordinator.min(own)], &mut high[0]);
    if coordinator < own {
        (first, Some(second))
    } else {
        (second, Some(first))
    }
}

/// The decode tensors of one session (the per-token set of C++).
pub struct DecodeTensors {
    pub cur_hc: DeviceTensor,
    pub flat_hc: DeviceTensor,
    pub hc_mix: DeviceTensor,
    pub hc_split: DeviceTensor,
    pub attn_cur: DeviceTensor,
    pub attn_norm: DeviceTensor,
    pub qr: DeviceTensor,
    pub qr_norm: DeviceTensor,
    pub q: DeviceTensor,
    pub kv_raw: DeviceTensor,
    pub kv: DeviceTensor,
    pub comp_kv_cur: DeviceTensor,
    pub comp_sc_cur: DeviceTensor,
    pub indexer_q: DeviceTensor,
    pub indexer_weights: DeviceTensor,
    pub indexer_scores: DeviceTensor,
    pub comp_selected: DeviceTensor,
    pub heads: DeviceTensor,
    pub attn_low: DeviceTensor,
    pub attn_out: DeviceTensor,
    pub after_attn_hc: DeviceTensor,
    pub ffn_cur: DeviceTensor,
    pub ffn_norm: DeviceTensor,
    pub shared_gate: DeviceTensor,
    pub shared_up: DeviceTensor,
    pub shared_mid: DeviceTensor,
    pub shared_out: DeviceTensor,
    pub router_logits: DeviceTensor,
    pub router_probs: DeviceTensor,
    pub router: ExpertSelection,
    pub routed_gate: DeviceTensor,
    pub routed_up: DeviceTensor,
    pub routed_mid: DeviceTensor,
    pub routed_down: DeviceTensor,
    pub routed_out: DeviceTensor,
    pub after_ffn_hc: DeviceTensor,
    pub output_pre: DeviceTensor,
    pub output_weights: DeviceTensor,
    pub output_embd: DeviceTensor,
    pub output_norm: DeviceTensor,
    pub logits: DeviceTensor,
}

/// The persistent KV state of one layer: the raw SWA ring and, for
/// compressed layers, the compressed caches, the F16 mirror (ratio 4), the
/// indexer cache and the compressor frontiers.
pub struct LayerCache {
    pub raw: DeviceTensor,
    pub attn_comp: Option<DeviceTensor>,
    pub attn_comp_f16: Option<DeviceTensor>,
    pub attn_state_kv: Option<DeviceTensor>,
    pub attn_state_score: Option<DeviceTensor>,
    pub index_comp: Option<DeviceTensor>,
    pub index_state_kv: Option<DeviceTensor>,
    pub index_state_score: Option<DeviceTensor>,
    pub n_comp: u32,
    pub n_index_comp: u32,
    /// Compressed rows the context can reach.
    pub comp_cap: u32,
}

/// The executor state of one session (`ds4_rocm_graph`).
pub struct Graph {
    pub d: DecodeTensors,
    pub layers: Vec<LayerCache>,
    pub raw_cap: u32,
    pub raw_window: u32,
    /// The largest compressed row count of any layer.
    pub comp_cap: u32,
    pub prefill_cap: u32,
    pub managed_kv_cache: bool,
    /// The engine's batch workspace this session uses, lent per call.
    pub workspace: WorkspaceLease,
    /// Which workspace stream buffer is `batch_cur_hc` (C++ swaps the
    /// graph's copies of the pointers).
    pub batch_cur_index: usize,
    /// Verification captures prefix states for partial rollback.
    pub spec_capture_prefixes: bool,
    /// Verification and multi-session state, made on first use.
    pub(crate) spec: Option<super::batch::SpecState>,
    /// The drafting state, when a support model is attached.
    pub(crate) dspark: Option<super::dspark::DsparkState>,
}

/// `rocm_graph_kv_cache_bytes_for_context`.
fn kv_cache_bytes(context: u32, raw_cap: u32) -> u64 {
    let mut bytes = u64::from(N_LAYER) * u64::from(raw_cap) * u64::from(N_HEAD_DIM) * F32;
    for il in 0..N_LAYER {
        let r = ratio(il);
        if r == 0 {
            continue;
        }
        let comp_cap = u64::from(context / r + 2);
        bytes += comp_cap * u64::from(N_HEAD_DIM) * F32;
        if r == 4 {
            bytes += comp_cap * u64::from(N_HEAD_DIM) * 2;
            bytes += comp_cap * u64::from(N_INDEXER_HEAD_DIM) * F32;
        }
    }
    bytes
}

/// The smallest compression ratio of the model (4).
fn min_ratio(context: u32) -> u32 {
    (0..N_LAYER)
        .map(ratio)
        .filter(|&r| r != 0)
        .min()
        .unwrap_or(if context != 0 { context } else { 1 })
}

impl Graph {
    /// `ds4_rocm_graph_create`: a session of `context` positions with
    /// prompt chunks of `prefill_cap`, leasing the first workspace of
    /// `workspaces` large enough, or allocating one at its head. Idle older
    /// workspaces, and an idle head too small, are dropped first.
    pub fn create(
        b: &Backend,
        weights: &Weights,
        workspaces: &mut Vec<BatchWorkspace>,
        context: u32,
        prefill_cap: u32,
        dspark_block: Option<u32>,
    ) -> Result<Self> {
        let mut index = 0;
        workspaces.retain(|ws| {
            let leased = Arc::strong_count(&ws.lease.0) > 1;
            let keep = leased || (index == 0 && ws.prefill_cap >= prefill_cap);
            index += 1;
            keep
        });
        let raw_cap = raw_cap_for_context(context, prefill_cap);
        let existing = workspaces
            .iter()
            .position(|ws| ws.prefill_cap >= prefill_cap);
        let index = match existing {
            Some(index) => index,
            None => {
                let mut ws = BatchWorkspace::alloc(b, weights, prefill_cap)?;
                if let Some(block) = dspark_block {
                    let rows = 8 * (block + 1);
                    let elems = u64::from(rows) * HC_DIM;
                    ws.dspark_verify = Some((b.alloc(elems * F32)?, b.alloc(elems * F32)?, rows));
                }
                eprintln!("ds4: sessions share one {prefill_cap}-row batch workspace");
                workspaces.insert(0, ws);
                0
            }
        };
        Self::alloc(
            b,
            weights,
            raw_cap,
            context,
            prefill_cap,
            &workspaces[index],
        )
    }

    /// `rocm_graph_alloc_raw_cap`.
    fn alloc(
        b: &Backend,
        weights: &Weights,
        raw_cap: u32,
        context: u32,
        prefill_cap: u32,
        ws: &BatchWorkspace,
    ) -> Result<Self> {
        let raw_cap = raw_cap.max(1);
        let context = if context == 0 { raw_cap } else { context };
        let prefill_cap = prefill_cap.max(1);
        let raw_window = N_SWA.min(context).max(1);
        let raw_cap = raw_cap.max(raw_window).min(context).max(1);
        let min_ratio = min_ratio(context);
        let comp_cap = (context / min_ratio + 2).max(2);
        let layer = &weights.layers[0];
        let q_rank = layer.attn_q_a.dim[1];
        let shared_dim = layer.ffn_gate_shexp.dim[1];
        let routed_mid_dim = layer.ffn_gate_exps.dim[1];
        let vocab_dim = weights.output.dim[1];
        let comp_width_max = 2 * u64::from(N_HEAD_DIM.max(N_INDEXER_HEAD_DIM));
        let pc = u64::from(prefill_cap);
        let embd = u64::from(N_EMBD);

        // rocm_graph_context_bytes_for_kv_policy
        let kv_bytes = kv_cache_bytes(context, raw_cap);
        let policy_comp_cap = u64::from((context / min_ratio + 2).max(2));
        let context_bytes = kv_bytes + policy_comp_cap * pc * F32;
        let managed = b.should_use_managed_kv_cache(kv_bytes, context_bytes);
        if managed {
            eprintln!(
                "ds4: ROCm using managed KV cache for ctx={context} (maximum KV {:.2} GiB, context estimate {:.2} GiB); this may degrade performance but is needed for very large contexts",
                kv_bytes as f64 / 1073741824.0,
                context_bytes as f64 / 1073741824.0
            );
        }
        let kv_alloc = |bytes: u64| {
            if managed {
                b.alloc_managed(bytes)
            } else {
                b.alloc(bytes)
            }
        };

        let d_cur_hc = b.alloc(HC_DIM * F32)?;
        let d_flat_hc = b.alloc(HC_DIM * F32)?;
        let d_hc_mix = b.alloc(MIX_HC * F32)?;
        let d_hc_split = b.alloc(MIX_HC * F32)?;
        let d_attn_cur = b.alloc(embd * F32)?;
        let d_attn_norm = b.alloc(embd * F32)?;
        let d_qr = b.alloc(q_rank * F32)?;
        let d_qr_norm = b.alloc(q_rank * F32)?;
        let d_q = b.alloc(Q_DIM * F32)?;
        let d_kv_raw = b.alloc(u64::from(N_HEAD_DIM) * F32)?;
        let d_kv = b.alloc(u64::from(N_HEAD_DIM) * F32)?;
        let mut layers = Vec::with_capacity(N_LAYER as usize);
        for il in 0..N_LAYER {
            let raw = kv_alloc(u64::from(raw_cap) * u64::from(N_HEAD_DIM) * F32)?;
            let r = ratio(il);
            let layer_comp_cap = context.checked_div(r).map_or(0, |rows| (rows + 2).max(2));
            let mut cache = LayerCache {
                raw,
                attn_comp: None,
                attn_comp_f16: None,
                attn_state_kv: None,
                attn_state_score: None,
                index_comp: None,
                index_state_kv: None,
                index_state_score: None,
                n_comp: 0,
                n_index_comp: 0,
                comp_cap: layer_comp_cap,
            };
            if r != 0 {
                // The first prompt chunk plus one raw window of decode
                // headroom; the caches grow on demand.
                let initial =
                    u64::from(layer_comp_cap).min((pc + u64::from(N_SWA)) / u64::from(r) + 2);
                let coff: u64 = if r == 4 { 2 } else { 1 };
                let attn_width = coff * u64::from(N_HEAD_DIM);
                let attn_rows = coff * u64::from(r);
                cache.attn_comp = Some(kv_alloc(initial * u64::from(N_HEAD_DIM) * F32)?);
                if r == 4 {
                    cache.attn_comp_f16 = Some(kv_alloc(initial * u64::from(N_HEAD_DIM) * 2)?);
                }
                let mut state_kv = b.alloc(attn_width * attn_rows * F32)?;
                b.fill_f32(&mut state_kv, 0.0, attn_width * attn_rows)?;
                let mut state_score = b.alloc(attn_width * attn_rows * F32)?;
                b.fill_f32(&mut state_score, NEG_INF, attn_width * attn_rows)?;
                cache.attn_state_kv = Some(state_kv);
                cache.attn_state_score = Some(state_score);
                if r == 4 {
                    let index_width = coff * u64::from(N_INDEXER_HEAD_DIM);
                    let index_rows = coff * u64::from(r);
                    cache.index_comp =
                        Some(kv_alloc(initial * u64::from(N_INDEXER_HEAD_DIM) * F32)?);
                    let mut state_kv = b.alloc(index_width * index_rows * F32)?;
                    b.fill_f32(&mut state_kv, 0.0, index_width * index_rows)?;
                    let mut state_score = b.alloc(index_width * index_rows * F32)?;
                    b.fill_f32(&mut state_score, NEG_INF, index_width * index_rows)?;
                    cache.index_state_kv = Some(state_kv);
                    cache.index_state_score = Some(state_score);
                }
            }
            layers.push(cache);
        }
        // The first prefill chunk and single-token decoding at any depth;
        // later chunks grow this disposable scratch.
        let initial_scores =
            u64::from(comp_cap).max(u64::from(comp_cap).min(pc / u64::from(min_ratio) + 2) * pc);
        let d = DecodeTensors {
            cur_hc: d_cur_hc,
            flat_hc: d_flat_hc,
            hc_mix: d_hc_mix,
            hc_split: d_hc_split,
            attn_cur: d_attn_cur,
            attn_norm: d_attn_norm,
            qr: d_qr,
            qr_norm: d_qr_norm,
            q: d_q,
            kv_raw: d_kv_raw,
            kv: d_kv,
            comp_kv_cur: b.alloc(comp_width_max * F32)?,
            comp_sc_cur: b.alloc(comp_width_max * F32)?,
            indexer_q: b.alloc(INDEXER_Q_DIM * F32)?,
            indexer_weights: b.alloc(u64::from(N_INDEXER_HEAD) * F32)?,
            indexer_scores: b.alloc(initial_scores * F32)?,
            comp_selected: b.alloc(u64::from(N_INDEXER_TOP_K) * pc * 4)?,
            heads: b.alloc(Q_DIM * F32)?,
            attn_low: b.alloc(LOW_DIM * F32)?,
            attn_out: b.alloc(embd * F32)?,
            after_attn_hc: b.alloc(HC_DIM * F32)?,
            ffn_cur: b.alloc(embd * F32)?,
            ffn_norm: b.alloc(embd * F32)?,
            shared_gate: b.alloc(shared_dim * F32)?,
            shared_up: b.alloc(shared_dim * F32)?,
            shared_mid: b.alloc(shared_dim * F32)?,
            shared_out: b.alloc(embd * F32)?,
            router_logits: b.alloc(u64::from(N_EXPERT) * F32)?,
            router_probs: b.alloc(u64::from(N_EXPERT) * F32)?,
            router: b.alloc_expert_selection(1)?,
            routed_gate: b.alloc(u64::from(N_EXPERT_USED) * routed_mid_dim * F32)?,
            routed_up: b.alloc(u64::from(N_EXPERT_USED) * routed_mid_dim * F32)?,
            routed_mid: b.alloc(u64::from(N_EXPERT_USED) * routed_mid_dim * F32)?,
            routed_down: b.alloc(u64::from(N_EXPERT_USED) * embd * F32)?,
            routed_out: b.alloc(embd * F32)?,
            after_ffn_hc: b.alloc(HC_DIM * F32)?,
            output_pre: b.alloc(u64::from(N_HC) * F32)?,
            output_weights: b.alloc(u64::from(N_HC) * F32)?,
            output_embd: b.alloc(embd * F32)?,
            output_norm: b.alloc(embd * F32)?,
            logits: b.alloc(vocab_dim * F32)?,
        };
        if ws.prefill_cap < prefill_cap {
            return Err(gufo_deepseek_v4_flash_kernels::Error {
                op: "batch workspace too small",
            });
        }
        Ok(Self {
            d,
            layers,
            raw_cap,
            raw_window,
            comp_cap,
            prefill_cap,
            managed_kv_cache: managed,
            workspace: WorkspaceLease(Arc::clone(&ws.lease.0)),
            batch_cur_index: 0,
            spec_capture_prefixes: false,
            spec: None,
            dspark: None,
        })
    }

    /// Refuses a workspace other than the one this graph leases.
    pub(crate) fn check_workspace(&self, ws: &BatchWorkspace) -> Result<()> {
        if ws.lent_by(&self.workspace) {
            Ok(())
        } else {
            Err(Error {
                op: "batch workspace",
            })
        }
    }

    /// `rocm_graph_reset_prefill_state`: no compressed rows, no support
    /// context or pending capture, fresh compressor frontiers.
    pub fn reset(&mut self, b: &Backend) -> Result<()> {
        if let Some(ds) = &mut self.dspark {
            ds.context_len = 0;
            ds.capture_mask = 0;
            ds.capture_batch_mask = 0;
            ds.capture_batch_start = 0;
            ds.capture_batch_tokens = 0;
        }
        for (il, cache) in self.layers.iter_mut().enumerate() {
            cache.n_comp = 0;
            cache.n_index_comp = 0;
            let r = ratio(il as u32);
            if r == 0 {
                continue;
            }
            let coff: u64 = if r == 4 { 2 } else { 1 };
            let attn = coff * u64::from(N_HEAD_DIM) * coff * u64::from(r);
            if let (Some(kv), Some(score)) = (&mut cache.attn_state_kv, &mut cache.attn_state_score)
            {
                b.fill_f32(kv, 0.0, attn)?;
                b.fill_f32(score, NEG_INF, attn)?;
            }
            if r == 4 {
                let index = coff * u64::from(N_INDEXER_HEAD_DIM) * coff * u64::from(r);
                if let (Some(kv), Some(score)) =
                    (&mut cache.index_state_kv, &mut cache.index_state_score)
                {
                    b.fill_f32(kv, 0.0, index)?;
                    b.fill_f32(score, NEG_INF, index)?;
                }
            }
        }
        Ok(())
    }

    /// `rocm_graph_raw_span_for_batch`: raw rows a batch from `pos0` sees.
    pub fn raw_span_for_batch(&self, pos0: u32, n_tokens: u32) -> u32 {
        if self.raw_cap == 0 || n_tokens == 0 {
            return 0;
        }
        let window = if self.raw_window != 0 {
            self.raw_window
        } else {
            N_SWA
        };
        let last_pos = pos0 + n_tokens - 1;
        let mut needed = u64::from(n_tokens);
        if window != 0 {
            needed += if n_tokens == 1 {
                u64::from(window) - 1
            } else {
                u64::from(window)
            };
        }
        needed = needed
            .min(u64::from(last_pos) + 1)
            .min(u64::from(self.raw_cap));
        needed as u32
    }

    /// `rocm_graph_raw_start_for_span`.
    pub fn raw_start_for_span(&self, last_pos: u32, n_raw: u32) -> u32 {
        if self.raw_cap == 0 || n_raw == 0 {
            return 0;
        }
        (last_pos + 1 - n_raw) % self.raw_cap
    }

    /// `rocm_graph_reserve_compressed_rows`: grows the compressed caches of
    /// `il` geometrically to hold `required` rows, copying the live rows.
    pub fn reserve_compressed_rows(&mut self, b: &Backend, il: u32, required: u32) -> Result<()> {
        let managed = self.managed_kv_cache;
        let cache = &mut self.layers[il as usize];
        let refuse = gufo_deepseek_v4_flash_kernels::Error {
            op: "reserve compressed rows",
        };
        if required > cache.comp_cap {
            return Err(refuse);
        }
        let row_bytes = u64::from(N_HEAD_DIM) * F32;
        let Some(attn) = &cache.attn_comp else {
            return Err(refuse);
        };
        let capacity = attn.bytes() / row_bytes;
        if u64::from(required) <= capacity {
            return Ok(());
        }
        let next = u64::from(cache.comp_cap).min(u64::from(required).max(capacity * 2));
        let alloc = |bytes: u64| {
            if managed {
                b.alloc_managed(bytes)
            } else {
                b.alloc(bytes)
            }
        };
        let mut attention = alloc(next * row_bytes)?;
        let indexed = ratio(il) == 4;
        let (mut mirror, mut indexer) = if indexed {
            (
                Some(alloc(next * u64::from(N_HEAD_DIM) * 2)?),
                Some(alloc(next * u64::from(N_INDEXER_HEAD_DIM) * F32)?),
            )
        } else {
            (None, None)
        };
        let copy = |to: &mut DeviceTensor, from: &Option<DeviceTensor>, bytes: u64| -> Result<()> {
            match from {
                Some(from) if bytes != 0 => b.copy(to, 0, from, 0, bytes),
                _ => Ok(()),
            }
        };
        copy(
            &mut attention,
            &cache.attn_comp,
            u64::from(cache.n_comp) * row_bytes,
        )?;
        if let (Some(mirror), Some(indexer)) = (&mut mirror, &mut indexer) {
            copy(
                mirror,
                &cache.attn_comp_f16,
                u64::from(cache.n_comp) * u64::from(N_HEAD_DIM) * 2,
            )?;
            copy(
                indexer,
                &cache.index_comp,
                u64::from(cache.n_index_comp) * u64::from(N_INDEXER_HEAD_DIM) * F32,
            )?;
        }
        // Every replacement and copy succeeded before the old caches go.
        cache.attn_comp = Some(attention);
        if indexed {
            cache.attn_comp_f16 = mirror;
            cache.index_comp = indexer;
        }
        Ok(())
    }

    /// `rocm_graph_reserve_compressed_position`: rows for positions below
    /// `end`.
    pub fn reserve_compressed_position(&mut self, b: &Backend, il: u32, end: u64) -> Result<()> {
        let r = ratio(il);
        if r == 0 {
            return Ok(());
        }
        let cap = u64::from(self.layers[il as usize].comp_cap);
        self.reserve_compressed_rows(b, il, cap.min(end / u64::from(r) + 2) as u32)
    }

    /// `rocm_graph_reserve_indexer_scores`: grows the score scratch
    /// geometrically to `n_comp * n_tokens` floats.
    pub fn reserve_indexer_scores(
        &mut self,
        b: &Backend,
        n_comp: u32,
        n_tokens: u32,
    ) -> Result<()> {
        if n_comp == 0 || n_comp > self.comp_cap || n_tokens == 0 || n_tokens > self.prefill_cap {
            return Err(gufo_deepseek_v4_flash_kernels::Error {
                op: "reserve indexer scores",
            });
        }
        let required = u64::from(n_comp) * u64::from(n_tokens) * F32;
        let capacity = self.d.indexer_scores.bytes();
        if required <= capacity {
            return Ok(());
        }
        let maximum = u64::from(self.comp_cap) * u64::from(self.prefill_cap) * F32;
        self.d.indexer_scores = b.alloc(maximum.min(required.max(capacity * 2)))?;
        Ok(())
    }
}
