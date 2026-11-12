//! float16 products with float32 accumulation and output on hipBLASLt
//! (`gemm.cpp`). On gfx1151 hipBLASLt runs them on the matrix cores, where
//! rocBLAS's mixed-precision GEMM does not, and its ranking of algorithms is
//! a poor predictor, so candidates are timed once per representative shape
//! ([`BlasLt::matmul_tuned`]). Shapes hipBLASLt can't serve fall back to
//! hipBLAS.

use gufo_hip::blas::{self, Blas, BlasLt, Layout, MatmulBatch, Op, Strides, Tuning};
use gufo_hip::{DeviceSlice, DeviceSliceMut, F16, Launch, LaunchError, Stream};

/// Workspace for algorithms that split the reduction.
const WORKSPACE_BYTES: usize = 32 << 20;
/// hipBLASLt results timed per representative shape, and timed launches
/// per result.
const TUNING: Tuning = Tuning {
    candidates: 32,
    runs: 3,
};

/// A column-major (BLAS) product `C[m][n] (+)= op(A)[m][k] * op(B)[k][n]`,
/// optionally strided-batched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatmulShape {
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
    pub stride_a: usize,
    pub stride_b: usize,
    pub stride_c: usize,
    pub batch: usize,
    pub accumulate: bool,
}

impl Default for MatmulShape {
    fn default() -> Self {
        Self {
            transpose_a: false,
            transpose_b: false,
            m: 0,
            n: 0,
            k: 0,
            lda: 0,
            ldb: 0,
            ldc: 0,
            stride_a: 0,
            stride_b: 0,
            stride_c: 0,
            batch: 1,
            accumulate: false,
        }
    }
}

fn op(transpose: bool) -> Op {
    if transpose { Op::T } else { Op::N }
}

impl MatmulShape {
    /// hipBLASLt's view: the A operation is always set, the B operation only
    /// when it transposes, and batch attributes only for more than one
    /// product (as the C++ descriptors).
    fn blaslt(&self) -> blas::MatmulShape {
        let (a_rows, a_columns) = if self.transpose_a {
            (self.k, self.m)
        } else {
            (self.m, self.k)
        };
        let (b_rows, b_columns) = if self.transpose_b {
            (self.n, self.k)
        } else {
            (self.k, self.n)
        };
        blas::MatmulShape {
            trans_a: Some(op(self.transpose_a)),
            trans_b: self.transpose_b.then_some(Op::T),
            a: Layout::new(a_rows, a_columns, self.lda),
            b: Layout::new(b_rows, b_columns, self.ldb),
            c: Layout::new(self.m, self.n, self.ldc),
            batch: (self.batch > 1).then_some(MatmulBatch {
                count: self.batch,
                a: self.stride_a,
                b: self.stride_b,
                c: self.stride_c,
            }),
        }
    }
}

/// Rounds up to a multiple of 128, the bucket size of tuned shapes.
pub const fn tuning_bucket(size: usize) -> usize {
    size.div_ceil(128) * 128
}

/// float16 GEMMs on one stream (`HalfGemm`). `Send`, not `Sync`.
pub struct HalfGemm {
    lt: Option<BlasLt>,
    fallback: Blas,
}

impl HalfGemm {
    /// A hipBLASLt handle with its workspace; without one every product
    /// runs on hipBLAS.
    pub fn new(stream: &Stream) -> Result<Self, LaunchError> {
        Ok(Self {
            lt: BlasLt::new(stream, WORKSPACE_BYTES).ok(),
            fallback: Blas::new(stream)?,
        })
    }

    /// `y[rows][out] (+)= x[rows][in] * w[out][in]^T`.
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &mut self,
        stream: &Stream,
        x: DeviceSlice<'_, F16>,
        w: DeviceSlice<'_, F16>,
        y: DeviceSliceMut<'_, f32>,
        rows: usize,
        input: usize,
        output: usize,
        accumulate: bool,
    ) -> Result<(), LaunchError> {
        let shape = MatmulShape {
            transpose_a: true,
            m: output,
            n: rows,
            k: input,
            lda: input,
            ldb: input,
            ldc: output,
            accumulate,
            ..MatmulShape::default()
        };
        let representative = MatmulShape {
            n: tuning_bucket(rows),
            ..shape
        };
        self.matmul(stream, &shape, &representative, w, x, y)
    }

    /// `shape` with the algorithm tuned for `representative`, a shape of the
    /// same structure with sizes rounded up so that tuning is shared.
    pub fn matmul(
        &mut self,
        stream: &Stream,
        shape: &MatmulShape,
        representative: &MatmulShape,
        a: DeviceSlice<'_, F16>,
        b: DeviceSlice<'_, F16>,
        mut c: DeviceSliceMut<'_, f32>,
    ) -> Result<(), LaunchError> {
        if let Some(lt) = &mut self.lt {
            let launch = lt.matmul_tuned(
                stream,
                &shape.blaslt(),
                &representative.blaslt(),
                a,
                b,
                c.reborrow(),
                shape.accumulate,
                TUNING,
            )?;
            if launch == Launch::Launched {
                return Ok(());
            }
        }
        self.fallback.gemm_strided_batched_beta(
            stream,
            op(shape.transpose_a),
            op(shape.transpose_b),
            shape.m,
            shape.n,
            shape.k,
            a,
            shape.lda,
            b,
            shape.ldb,
            if shape.accumulate { 1.0 } else { 0.0 },
            c,
            shape.ldc,
            Strides {
                count: shape.batch,
                a: shape.stride_a,
                b: shape.stride_b,
                c: shape.stride_c,
            },
        )
    }
}
