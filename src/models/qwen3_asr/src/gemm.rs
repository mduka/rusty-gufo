//! Multi-token BF16 projections `Y[batch, m] = X[batch, k] · A[m, k]^T` with an
//! f32 result: hipBLASLt with this model's algorithm policy, hipBLAS when
//! hipBLASLt offers no usable algorithm (as `hip/blas.cpp` does).

use gufo_hip::blas::{AlgoPolicy, Blas, BlasLt, Heuristic, Layout, MatmulShape, Op};
use gufo_hip::{Bf16, DeviceSlice, DeviceSliceMut, Launch, LaunchError, Stream};

/// hipBLASLt's rank-6 heuristic without workspace, else the first usable one.
///
/// Rank 6 was measured end to end on this model's shapes (ROCm 7.2): not the
/// fastest rank (8 is 1% faster), but the only one at least as accurate as the
/// reference route on every prefill boundary. The index points into a
/// library-generated list, so it holds only for the hipBLASLt version it was
/// measured with; re-sweep after a toolchain change.
pub struct AsrRank;

const MEASURED_RANK: usize = 6;

impl AlgoPolicy for AsrRank {
    fn choose(&self, ranked: &[Heuristic], capacity: usize) -> Option<usize> {
        let rank = MEASURED_RANK.min(ranked.len().checked_sub(1)?);
        if ranked[rank].is_usable(capacity) {
            return Some(rank);
        }
        ranked.iter().position(|h| h.is_usable(capacity))
    }
}

/// The BLAS handles of one component (encoder or decoder prefill). Each
/// component keeps its own, as the C++ engine does: plans are cached per
/// handle, and hipBLASLt's heuristics depend on what was planned before.
pub struct Projector {
    lt: BlasLt,
    blas: Blas,
}

impl Projector {
    pub fn new(stream: &Stream) -> Result<Self, String> {
        let blas = Blas::new(stream).map_err(|e| format!("Qwen3-ASR hipBLAS: {e}"))?;
        let lt = BlasLt::new(stream, 0).map_err(|e| format!("Qwen3-ASR hipBLASLt: {e}"))?;
        Ok(Self { lt, blas })
    }

    /// `output[batch, m] = input[batch, k] · weight[m, k]^T`.
    #[allow(clippy::too_many_arguments)]
    pub fn project(
        &mut self,
        stream: &Stream,
        weight: DeviceSlice<'_, Bf16>,
        input: DeviceSlice<'_, Bf16>,
        mut output: DeviceSliceMut<'_, f32>,
        batch: usize,
        m: usize,
        k: usize,
    ) -> Result<(), LaunchError> {
        let shape = MatmulShape {
            trans_a: Some(Op::T),
            trans_b: None,
            a: Layout::new(k, m, k),
            b: Layout::new(k, batch, k),
            c: Layout::new(m, batch, m),
            batch: None,
        };
        if self
            .lt
            .matmul(stream, &shape, weight, input, output.reborrow(), &AsrRank)?
            == Launch::Launched
        {
            return Ok(());
        }
        self.blas.gemm(
            stream,
            Op::T,
            Op::N,
            m,
            batch,
            k,
            weight,
            k,
            input,
            k,
            output,
            m,
        )
    }
}
