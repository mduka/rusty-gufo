//! Expert routing (`ds4_rocm_router.hip.hpp`) and the selection the routed
//! experts read ([`super::routed`]).
//!
//! The routed-expert kernels bound expert IDs only from below (negative
//! IDs read expert 0), so every ID they read must be below 256: the
//! selection's IDs are indices of that bound (padding `-1` for rows of NaN
//! scores), which only the router writes, from a top-k over the 256 experts
//! or the entries of a [`HashTable`] (the kernel writes `-1` for any entry
//! outside the experts; [`Backend::hash_table`] checks them all at load).

use gufo_hip::{DeviceIndices, DeviceSlice, IndexSpec};

use crate::backend::{Backend, Error, Model, Result};
use crate::generated;
use crate::tensor::{DeviceTensor, Input, Output};

use super::{F32, I32, Weight, ensure, launched, product, typed, typed_mut};

/// Experts of the model and per token (`DS4_ROCM_N_EXPERT`,
/// `DS4_ROCM_N_EXPERT_USED`).
pub const N_EXPERT: u32 = 256;
pub const N_EXPERT_USED: u32 = 6;

/// The token-to-experts table of a hash-routed layer (`ffn_gate_tid2eid`,
/// `rows` tokens of six I32 expert IDs), every entry checked below
/// [`N_EXPERT`] when it was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HashTable {
    weight: Weight,
    rows: u32,
}

impl HashTable {
    pub fn rows(&self) -> u32 {
        self.rows
    }
}

/// Expert IDs and routing weights (F32), [`N_EXPERT_USED`] per row for up
/// to `rows` rows (see the module documentation), and the route plan the
/// routed experts derive from them.
pub struct ExpertSelection {
    pub(super) selected: DeviceIndices,
    pub(super) weights: DeviceTensor,
    pub(super) plan: RoutePlan,
    pub(super) rows: u32,
    /// Rows the router wrote since the last read by the experts; for the
    /// debug check that the experts read written rows only.
    pub(super) written: u32,
}

/// The route plan of a selection (`routed_moe_launch`'s sorted pairs and
/// expert tiles), sized for all of its rows: written only by the plan
/// launchers, from one generation of the selection, and read by kernels
/// that require exactly that derivation.
pub(super) struct RoutePlan {
    pub(super) counts: DeviceIndices,
    pub(super) offsets: DeviceIndices,
    pub(super) cursors: DeviceTensor,
    pub(super) sorted: DeviceIndices,
    pub(super) tile_offsets: DeviceTensor,
    /// Tiles of four or eight pairs, and the second tiling of sixteen.
    pub(super) tiles: Tiles,
    pub(super) tiles16: Tiles,
    /// The hot and the cold expert lists the host builds from the counts.
    pub(super) hot: DeviceIndices,
    pub(super) hot_low: DeviceIndices,
    /// Every expert in order, for the kernels that walk them all.
    pub(super) all: DeviceIndices,
    /// The (hot expert, row group) list of the wide down kernel.
    pub(super) tile_map: DeviceTensor,
}

/// One tiling of a route plan.
pub(super) struct Tiles {
    pub(super) total: DeviceIndices,
    pub(super) experts: DeviceIndices,
    pub(super) starts: DeviceIndices,
}

/// The routed experts of one layer: the gate, up and down tensors of all
/// 256 experts, their quantization types and shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutedExperts {
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
    pub gate_type: u32,
    pub down_type: u32,
    pub in_dim: u32,
    pub mid_dim: u32,
    pub out_dim: u32,
}

/// Bytes per 256 values of the routed expert types (IQ2_XXS, Q2_K, Q4_K).
fn block_bytes(ty: u32) -> Option<u64> {
    match ty {
        16 => Some(66),
        10 => Some(84),
        12 => Some(144),
        _ => None,
    }
}

impl RoutedExperts {
    /// Row and expert bytes of the gate (and up) and down tensors, as the
    /// C++ graph computes them (`ds4_routed_expert_row_bytes`).
    pub fn strides(&self) -> Option<(u64, u64, u64, u64)> {
        let gate_row = u64::from(self.in_dim / 256) * block_bytes(self.gate_type)?;
        let down_row = u64::from(self.mid_dim / 256) * block_bytes(self.down_type)?;
        Some((
            gate_row,
            gate_row.checked_mul(u64::from(self.mid_dim))?,
            down_row,
            down_row.checked_mul(u64::from(self.out_dim))?,
        ))
    }
}

/// The scratch tensors of a routed call, sized for the selection's rows;
/// the routes use all of them as scratch, and `out` (or, deferred, `down`)
/// holds the result.
pub struct RoutedMoe<'a> {
    pub out: &'a mut dyn Output,
    pub gate: &'a mut dyn Output,
    pub up: &'a mut dyn Output,
    pub mid: &'a mut dyn Output,
    pub down: &'a mut dyn Output,
}

impl ExpertSelection {
    pub fn rows(&self) -> u32 {
        self.rows
    }
}

impl Backend {
    /// A zeroed selection of `rows` rows and its route plan.
    pub fn alloc_expert_selection(&self, rows: u32) -> Result<ExpertSelection> {
        const OP: &str = "alloc expert selection";
        ensure(OP, rows != 0 && rows <= 2_796_202)?;
        let pairs = rows * N_EXPERT_USED;
        let bytes = product(OP, &[u64::from(pairs), I32])?;
        let mut weights = self.alloc(bytes)?;
        self.write(&mut weights, 0, &vec![0u8; bytes as usize])?;
        let indices = |len: u32, spec: IndexSpec| -> Result<DeviceIndices> {
            let stream = self.stream();
            stream
                .alloc_persistent::<u32>(len.max(1) as usize)
                .and_then(|buffer| stream.zeroed_indices_with(buffer, spec))
                .map_err(|_| Error { op: OP })
        };
        let tiles = || -> Result<Tiles> {
            Ok(Tiles {
                total: indices(1, IndexSpec::below(pairs + 1))?,
                experts: indices(pairs, IndexSpec::below(N_EXPERT))?,
                starts: indices(pairs, IndexSpec::below(pairs))?,
            })
        };
        let experts = IndexSpec::below(N_EXPERT);
        let mut all = indices(N_EXPERT, experts)?;
        let order: Vec<u32> = (0..N_EXPERT).collect();
        self.stream()
            .upload_indices(&order, &mut all)
            .map_err(|_| Error { op: OP })?;
        // `routed_moe_launch`'s bound: a 32-row group per pair and the
        // expert table, plus one.
        let map_len = pairs / 32 + N_EXPERT + 1;
        let plan = RoutePlan {
            counts: indices(N_EXPERT, IndexSpec::below(pairs + 1))?,
            offsets: indices(N_EXPERT + 1, IndexSpec::below(pairs + 1))?,
            cursors: self.alloc(u64::from(N_EXPERT) * I32)?,
            sorted: indices(pairs, IndexSpec::below(pairs))?,
            tile_offsets: self.alloc(u64::from(N_EXPERT + 1) * I32)?,
            tiles: tiles()?,
            tiles16: tiles()?,
            hot: indices(N_EXPERT, experts)?,
            hot_low: indices(N_EXPERT, experts)?,
            all,
            tile_map: self.alloc(u64::from(map_len) * I32)?,
        };
        Ok(ExpertSelection {
            selected: indices(pairs, IndexSpec::below(N_EXPERT).signed().padded())?,
            weights,
            plan,
            rows,
            written: 0,
        })
    }

    /// Checks the hash-routing table of `rows` tokens at `offset` of the
    /// target mapping: every expert ID in `[0, N_EXPERT)`. The device copy
    /// holds the same bytes ([`Backend::load_target_spans`]).
    pub fn hash_table(&self, offset: u64, rows: u32) -> Result<HashTable> {
        const OP: &str = "hash table";
        let bytes = product(OP, &[u64::from(rows), u64::from(N_EXPERT_USED), I32])?;
        let map = self.mapped(Model::Target)?;
        let start = usize::try_from(offset).map_err(|_| Error { op: OP })?;
        let end = usize::try_from(offset.checked_add(bytes).ok_or(Error { op: OP })?)
            .map_err(|_| Error { op: OP })?;
        ensure(OP, rows != 0 && offset.is_multiple_of(4))?;
        let table = map.bytes().get(start..end).ok_or(Error { op: OP })?;
        let valid = table.as_chunks::<4>().0.iter().all(|id| {
            let id = i32::from_le_bytes([id[0], id[1], id[2], id[3]]);
            (0..N_EXPERT as i32).contains(&id)
        });
        ensure(OP, valid)?;
        Ok(HashTable {
            weight: Weight::target(offset),
            rows,
        })
    }

    /// `ds4_gpu_router_select_tensor`: one token's six experts and weights
    /// from 256 router logits (`probs` receives the probabilities), by
    /// top-k with an optional bias, or by `hash` row `token`.
    pub fn router_select(
        &self,
        selection: &mut ExpertSelection,
        probs: &mut impl Output,
        logits: &impl Input,
        bias: Option<Weight>,
        hash: Option<&HashTable>,
        token: u32,
    ) -> Result<()> {
        self.router(
            "router select",
            selection,
            0,
            probs,
            logits,
            None,
            bias,
            hash,
            token,
            1,
        )?;
        selection.written = 1;
        Ok(())
    }

    /// `ds4_gpu_router_select_batch_tensor`: rows `row0 .. row0 + n_tokens`
    /// of the selection from as many logit rows, with the device `tokens`
    /// from row `token_row0` for hash routing (the kernel clamps them into
    /// the table).
    pub fn router_select_batch(
        &self,
        selection: &mut ExpertSelection,
        row0: u32,
        probs: &mut impl Output,
        logits: &impl Input,
        tokens: &impl Input,
        token_row0: u32,
        bias: Option<Weight>,
        hash: Option<&HashTable>,
        n_tokens: u32,
    ) -> Result<()> {
        const OP: &str = "router select batch";
        let tokens = typed::<i32>(OP, tokens.input())?;
        let first = token_row0 as usize;
        let end = first
            .checked_add(n_tokens as usize)
            .filter(|&end| end <= tokens.len())
            .ok_or(Error { op: OP })?;
        self.router(
            OP,
            selection,
            row0,
            probs,
            logits,
            Some(tokens.range(first..end)),
            bias,
            hash,
            0,
            n_tokens,
        )?;
        selection.written = selection.written.max(row0 + n_tokens);
        Ok(())
    }

    /// The router kernel over selection rows `row0 .. row0 + n_tokens`.
    fn router(
        &self,
        op: &'static str,
        selection: &mut ExpertSelection,
        row0: u32,
        probs: &mut impl Output,
        logits: &impl Input,
        tokens: Option<DeviceSlice<'_, i32>>,
        bias: Option<Weight>,
        hash: Option<&HashTable>,
        token: u32,
        n_tokens: u32,
    ) -> Result<()> {
        ensure(
            op,
            n_tokens != 0 && u64::from(row0) + u64::from(n_tokens) <= u64::from(selection.rows),
        )?;
        if let (Some(b), Some(h)) = (bias, hash) {
            ensure(op, b.model == h.weight.model)?;
        }
        // Rows are six elements; the kernel stores them one at a time.
        let used = N_EXPERT_USED as usize;
        let rows = row0 as usize * used..(row0 + n_tokens) as usize * used;
        let selected = selection.selected.slice_mut(rows.clone());
        let weights = typed_mut::<f32>(op, selection.weights.output())?.range(rows);
        // Hash-routed layers take no bias.
        let bias = match (bias, hash) {
            (Some(b), None) => Some(typed::<f32>(
                op,
                self.weight_bytes(b, u64::from(N_EXPERT) * F32)?,
            )?),
            _ => None,
        };
        let table = hash
            .map(|h| {
                let bytes = product(op, &[u64::from(h.rows), u64::from(N_EXPERT_USED), I32])?;
                typed::<i32>(op, self.weight_bytes(h.weight, bytes)?)
            })
            .transpose()?;
        launched(
            op,
            generated::router_select(
                self.stream(),
                selected,
                weights,
                typed_mut::<f32>(op, probs.output())?,
                bias,
                table,
                typed::<f32>(op, logits.input())?,
                tokens,
                token,
                hash.map_or(1, |h| h.rows),
                n_tokens,
            ),
        )
    }
}
