//! BF16 projections `Y[tokens, m] = X[tokens, k] · W[m, k]^T` with an f32
//! result (`hip/blas.cpp`).
//!
//! rocBLAS has no matrix-core kernel for a transposed BF16 operand with an
//! f32 result and runs a scalar kernel at about 5 TFLOPS on gfx1151.
//! hipBLASLt reaches 17 to 27 TFLOPS at 1,024 rows, but its ranked heuristic
//! list is a poor guide here. The top entry is no faster than rocBLAS, and
//! the best rank moves between 4 and 8 with the shape and the token count
//! (see `tests/gemm_bench.rs`). The first call for a shape therefore times
//! every usable heuristic result on the caller's own operands and keeps the
//! fastest. Later calls reuse it.
//!
//! The token count rounds up to a bucket so that nearby batch sizes share one
//! tuned plan. The encoder never passes more than 1,024 rows, which bounds
//! the buckets to twenty and lets it tune them all while the model loads. The
//! extra rows are computed and ignored, so the input and output views must
//! hold `capacity_tokens` rows.

use std::collections::HashSet;

use gufo_hip::blas::{AlgoPolicy, Blas, BlasLt, Heuristic, Layout, MatmulShape, Op};
use gufo_hip::{Bf16, DeviceSlice, DeviceSliceMut, Launch, LaunchError, Stream};

/// Heuristic results requested per shape.
const MAX_ALGORITHMS: usize = 32;
/// Timed launches per candidate after its untimed first launch.
const TIMED_RUNS: usize = 3;

/// The measured choice: every workspace-free result, timed on the shape's
/// first operands; no solution names are resolved, as the C++ plans resolve
/// none.
pub struct Fastest;

impl AlgoPolicy for Fastest {
    fn requested(&self) -> usize {
        MAX_ALGORITHMS
    }

    fn choose(&self, _: &[Heuristic], _: usize) -> Option<usize> {
        None
    }

    fn resolve_names(&self) -> bool {
        false
    }

    fn timed_runs(&self) -> Option<usize> {
        Some(TIMED_RUNS)
    }
}

/// Rows the GEMM computes for `tokens` input rows. Up to 256 rows a GEMM is
/// launch-bound and padding is free. Above that, a 64-row step pads at most
/// 24%, at 257 rows, and at most 12% from 513 rows on.
pub fn bucket(tokens: usize, capacity_tokens: usize) -> usize {
    let step = if tokens <= 256 { 32 } else { 64 };
    tokens
        .div_ceil(step)
        .saturating_mul(step)
        .min(capacity_tokens)
}

/// The BLAS handles of the encoder: the tuned hipBLASLt plans, and hipBLAS
/// for shapes hipBLASLt cannot serve. Not thread-safe; the encoder runs it
/// in its device turns and during load.
pub struct Gemm {
    lt: Option<BlasLt>,
    blas: Blas,
    /// Shapes reported as served by hipBLAS.
    reported: HashSet<(usize, usize, usize)>,
}

fn projection_shape(rows: usize, m: usize, k: usize) -> MatmulShape {
    MatmulShape {
        trans_a: Some(Op::T),
        trans_b: None,
        a: Layout::new(k, m, k),
        b: Layout::new(k, rows, k),
        c: Layout::new(m, rows, m),
        batch: None,
    }
}

impl Gemm {
    pub fn new(stream: &Stream) -> Result<Self, String> {
        // hipBLAS keeps its default atomics mode, as the C++ handle does.
        let blas = Blas::with_default_atomics(stream)
            .map_err(|e| format!("EmbeddingGemma 2 hipBLAS: {e}"))?;
        let lt = match BlasLt::new(stream, 0) {
            Ok(lt) => Some(lt),
            Err(_) => {
                eprintln!("EmbeddingGemma 2 hipBLASLt is unavailable; using hipBLAS");
                None
            }
        };
        Ok(Self {
            lt,
            blas,
            reported: HashSet::new(),
        })
    }

    /// `Y[tokens, m] = X[tokens, k] · W[m, k]^T` through the tuned hipBLASLt
    /// plan of the token bucket, or hipBLAS when hipBLASLt has none for the
    /// shape. `input` and `output` start at the first row and hold at least
    /// `bucket(tokens, capacity_tokens)` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        stream: &Stream,
        weight: DeviceSlice<'_, Bf16>,
        input: DeviceSlice<'_, Bf16>,
        mut output: DeviceSliceMut<'_, f32>,
        tokens: usize,
        m: usize,
        k: usize,
        capacity_tokens: usize,
    ) -> Result<(), LaunchError> {
        if let Some(lt) = self.lt.as_mut()
            && tokens != 0
            && m != 0
            && k != 0
            && tokens <= capacity_tokens
        {
            let rows = bucket(tokens, capacity_tokens);
            let launched = lt.matmul(
                stream,
                &projection_shape(rows, m, k),
                weight,
                input.range(0..rows * k),
                output.reborrow().range(0..rows * m),
                &Fastest,
            )?;
            if launched == Launch::Launched {
                return Ok(());
            }
            if self.reported.insert((rows, m, k)) {
                eprintln!(
                    "EmbeddingGemma 2 hipBLASLt has no usable algorithm for m={m} k={k} \
                     rows={rows}; using hipBLAS"
                );
            }
        }
        self.blas_gemm(stream, weight, input, output, tokens, m, k)
    }

    /// The same product through hipBLAS, for exactly `tokens` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn blas_gemm(
        &self,
        stream: &Stream,
        weight: DeviceSlice<'_, Bf16>,
        input: DeviceSlice<'_, Bf16>,
        output: DeviceSliceMut<'_, f32>,
        tokens: usize,
        m: usize,
        k: usize,
    ) -> Result<(), LaunchError> {
        // Row-major Y[tokens, m] is column-major Y^T[m, tokens] = W^T X^T.
        self.blas.gemm(
            stream,
            Op::T,
            Op::N,
            m,
            tokens,
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

#[cfg(test)]
mod tests {
    use super::bucket;

    #[test]
    fn buckets_round_to_32_then_64_rows_within_the_capacity() {
        assert_eq!(bucket(1, 1024), 32);
        assert_eq!(bucket(32, 1024), 32);
        assert_eq!(bucket(33, 1024), 64);
        assert_eq!(bucket(256, 1024), 256);
        assert_eq!(bucket(257, 1024), 320);
        assert_eq!(bucket(1000, 1024), 1024);
        assert_eq!(bucket(1024, 1024), 1024);
        assert_eq!(bucket(30, 16), 16);
        let mut rows = 0;
        let mut buckets = 0;
        while rows < 1024 {
            rows = bucket(rows + 1, 1024);
            buckets += 1;
        }
        assert_eq!(buckets, 20, "twenty buckets up to 1,024 rows");
    }
}
