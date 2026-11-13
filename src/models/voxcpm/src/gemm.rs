//! Products of bfloat16 activations with the generator's weights, with
//! float32 accumulation and output (`gemm.hpp`, `gemm.cpp`). Up to 32 rows
//! they read every weight once: bfloat16 rows with the skinny kernels
//! (`skinny_gemm.hpp`), tiles (bfloat16 or Q8_0) on the matrix cores
//! (`wmma_products.hpp`), which also take up to 128 rows in chunks of 32.
//! More rows run on hipBLASLt, tiles expanded to bfloat16 rows first (Q8_0
//! rounded to bfloat16). On gfx1151 hipBLASLt runs them on the matrix cores,
//! and its ranking of algorithms is a poor predictor, so candidates are timed
//! once per representative shape ([`BlasLt::matmul_tuned`]). Shapes hipBLASLt
//! can't serve fall back to hipBLAS.

use gufo_hip::blas::{self, Blas, BlasLt, Layout, MatmulBatch, Op, Strides, Tuning};
use gufo_hip::{
    Bf16, DeviceBuffer, DeviceCopy, DeviceSlice, DeviceSliceMut, HipError, Launch, LaunchError,
    Stream,
};
use gufo_voxcpm_kernels as k;

use crate::matrix::{MatrixView, WeightFormat};

/// Rows up to which tiles stay on the matrix cores, in chunks of 32
/// (`kChunkedRows`).
pub const CHUNKED_ROWS: usize = 128;
/// Rows of one matrix-core product.
const WMMA_ROWS: usize = 32;
/// Workspace for algorithms that split the reduction (`kWorkspaceBytes`).
const WORKSPACE_BYTES: usize = 32 << 20;
/// hipBLASLt results timed per representative shape, and timed launches
/// per result (`kCandidates`, `kRepeats`).
const TUNING: Tuning = Tuning {
    candidates: 32,
    runs: 3,
};
/// Fewest 32-column blocks per wave when splitting (`kMinBlocks` of
/// `wmma_products.hip`).
const MIN_BLOCKS: usize = 4;

/// How a matrix-core product is split over waves (`WmmaPlan`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WmmaPlan {
    pub splits: usize,
    /// 32-column blocks of the reduction per split.
    pub blocks_per_split: usize,
    /// Scratch the product needs.
    pub partial_floats: usize,
    pub counters: usize,
}

impl Default for WmmaPlan {
    fn default() -> Self {
        Self {
            splits: 1,
            blocks_per_split: 0,
            partial_floats: 0,
            counters: 0,
        }
    }
}

/// The split of a product of `rows` (at most 32) rows over at least about
/// `target_waves` waves, 0 for the measured default of the row count
/// (`PlanWmma`, host code of `wmma_products.hip`): fewer, longer waves suit
/// the DiT's 22 rows (two activation tiles per wave), more suit 1-16 rows.
pub fn plan_wmma(rows: usize, input: usize, output: usize, target_waves: usize) -> WmmaPlan {
    let target_waves = match target_waves {
        0 if rows > 16 => 160,
        0 => 640,
        waves => waves,
    };
    let mut plan = WmmaPlan::default();
    let tiles = output.div_ceil(16);
    let k_blocks = input / 32;
    if tiles == 0 || k_blocks == 0 {
        return plan;
    }
    let splits = target_waves.div_ceil(tiles);
    let splits = splits.min(k_blocks / MIN_BLOCKS.min(k_blocks)).max(1);
    plan.blocks_per_split = k_blocks.div_ceil(splits);
    plan.splits = k_blocks.div_ceil(plan.blocks_per_split);
    if plan.splits > 1 {
        plan.partial_floats = plan.splits * rows * output;
        plan.counters = tiles;
    }
    plan
}

/// A column-major (BLAS) product `C[m][n] (+)= op(A)[m][k] * op(B)[k][n]`,
/// optionally strided-batched (`MatmulShape`).
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
    /// product (as the C++ descriptors, `State::Describe`).
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

/// Rounds up to a multiple of 128, the bucket size of tuned shapes
/// (`TuningBucket`).
pub const fn tuning_bucket(size: usize) -> usize {
    size.div_ceil(128) * 128
}

/// The products of bfloat16 rows: hipBLASLt with its timed algorithms and
/// the hipBLAS fallback (`State`'s handles).
struct RowProducts {
    lt: Option<BlasLt>,
    fallback: Blas,
}

impl RowProducts {
    /// `Bf16Gemm::LinearRows`: `y[rows][out] (+)= x[rows][in] * w[out][in]^T`
    /// for bfloat16 rows.
    #[allow(clippy::too_many_arguments)]
    fn linear_rows(
        &mut self,
        stream: &Stream,
        x: DeviceSlice<'_, Bf16>,
        w: DeviceSlice<'_, Bf16>,
        mut y: DeviceSliceMut<'_, f32>,
        rows: usize,
        input: usize,
        output: usize,
        accumulate: bool,
    ) -> Result<(), LaunchError> {
        // A few rows: hipBLASLt's tiles waste most of their work, while the
        // product is bound by reading the weights once.
        let launch = k::skinny_gemm(stream, x, w, y.reborrow(), rows, input, output, accumulate)?;
        if launch == Launch::Launched {
            return Ok(());
        }
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

    /// `Bf16Gemm::Matmul`.
    fn matmul(
        &mut self,
        stream: &Stream,
        shape: &MatmulShape,
        representative: &MatmulShape,
        a: DeviceSlice<'_, Bf16>,
        b: DeviceSlice<'_, Bf16>,
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

/// Grows `buffer` to at least `count` elements; contents are lost when it
/// grows (`DeviceBuffer::Reserve`).
fn reserve<T: DeviceCopy>(
    stream: &Stream,
    buffer: &mut Option<DeviceBuffer<T>>,
    count: usize,
) -> Result<(), HipError> {
    if buffer.as_ref().map_or(0, DeviceBuffer::len) < count {
        *buffer = None;
        *buffer = Some(stream.alloc_persistent(count)?);
    }
    Ok(())
}

/// `rows * width`, or the error of an extent that does not fit.
fn extent(rows: usize, width: usize, param: &'static str) -> Result<usize, LaunchError> {
    rows.checked_mul(width)
        .ok_or(LaunchError::ExtentOverflow { param })
}

/// Products of bfloat16 activations with the generator's weights on one
/// stream (`Bf16Gemm`). `Send`, not `Sync`.
pub struct Bf16Gemm {
    products: RowProducts,
    /// Split reductions of the matrix-core products; the counters are zero
    /// between products.
    partials: Option<DeviceBuffer<f32>>,
    counters: Option<DeviceBuffer<u32>>,
    /// Tiles unpacked to bfloat16 rows for products of more than 128 rows.
    unpacked: Option<DeviceBuffer<Bf16>>,
}

impl Bf16Gemm {
    /// A hipBLASLt handle with its workspace (without one every product of
    /// more rows runs on hipBLAS) and the fallback hipBLAS handle, which
    /// keeps hipBLAS's default atomics mode like the C++ `hipblasCreate`.
    pub fn new(stream: &Stream) -> Result<Self, LaunchError> {
        Ok(Self {
            products: RowProducts {
                lt: BlasLt::new(stream, WORKSPACE_BYTES).ok(),
                fallback: Blas::with_default_atomics(stream)?,
            },
            partials: None,
            counters: None,
            unpacked: None,
        })
    }

    /// `y[rows][out] (+)= x[rows][in] * w[out][in]^T` (`Linear`).
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &mut self,
        stream: &Stream,
        x: DeviceSlice<'_, Bf16>,
        w: MatrixView<'_>,
        mut y: DeviceSliceMut<'_, f32>,
        rows: usize,
        input: usize,
        output: usize,
        accumulate: bool,
    ) -> Result<(), LaunchError> {
        let q8 = match w.format {
            WeightFormat::Bf16 => {
                let w = w.data.cast::<Bf16>().ok_or(LaunchError::InvalidArgument {
                    param: "w",
                    reason: "bfloat16 rows of odd size or alignment",
                })?;
                return self
                    .products
                    .linear_rows(stream, x, w, y, rows, input, output, accumulate);
            }
            WeightFormat::Q8Tiles => true,
            WeightFormat::Bf16Tiles => false,
        };
        // Up to CHUNKED_ROWS rows, the matrix cores read the tiles once per
        // 32 rows: as fast as expanding them for hipBLASLt (one 8-bit read
        // against an 8-bit read, a 16-bit write and a 16-bit read), and the
        // Q8_0 values stay exact.
        if rows <= CHUNKED_ROWS {
            let x_len = extent(rows, input, "x")?;
            let y_len = extent(rows, output, "y")?;
            if x.len() < x_len {
                return Err(LaunchError::ExtentTooShort {
                    param: "x",
                    required: x_len,
                    actual: x.len(),
                });
            }
            if y.len() < y_len {
                return Err(LaunchError::ExtentTooShort {
                    param: "y",
                    required: y_len,
                    actual: y.len(),
                });
            }
            let mut launched = true;
            let mut first = 0;
            while first < rows && launched {
                let count = (rows - first).min(WMMA_ROWS);
                let plan = plan_wmma(count, input, output, 0);
                if plan.splits > 1 {
                    reserve(stream, &mut self.partials, plan.partial_floats)?;
                    if self.counters.as_ref().map_or(0, DeviceBuffer::len) < plan.counters {
                        reserve(stream, &mut self.counters, plan.counters)?;
                        if let Some(counters) = &mut self.counters {
                            stream.fill_bytes(counters.as_mut_slice(), 0)?;
                        }
                    }
                }
                let partial_floats = self.partials.as_ref().map_or(0, DeviceBuffer::len);
                let counter_count = self.counters.as_ref().map_or(0, DeviceBuffer::len);
                let partials = self.partials.as_mut().map(DeviceBuffer::as_mut_slice);
                let counters = self.counters.as_mut().map(DeviceBuffer::as_mut_slice);
                let chunk_x = x.range(first * input..(first + count) * input);
                let chunk_y = y.reborrow().range(first * output..(first + count) * output);
                let launch = if q8 {
                    k::wmma_q8(
                        stream,
                        chunk_x,
                        w.data,
                        chunk_y,
                        count,
                        input,
                        output,
                        accumulate,
                        plan.splits,
                        plan.blocks_per_split,
                        partials,
                        partial_floats,
                        counters,
                        counter_count,
                    )?
                } else {
                    let tiles = w.data.cast::<Bf16>().ok_or(LaunchError::InvalidArgument {
                        param: "w",
                        reason: "bfloat16 tiles of odd size or alignment",
                    })?;
                    k::wmma_bf16(
                        stream,
                        chunk_x,
                        tiles,
                        chunk_y,
                        count,
                        input,
                        output,
                        accumulate,
                        plan.splits,
                        plan.blocks_per_split,
                        partials,
                        partial_floats,
                        counters,
                        counter_count,
                    )?
                };
                launched = launch == Launch::Launched;
                first += WMMA_ROWS;
            }
            if launched {
                return Ok(());
            }
            return Err(LaunchError::InvalidArgument {
                param: "w",
                reason: "the matrix-core product does not suit its shape",
            });
        }
        reserve(stream, &mut self.unpacked, extent(output, input, "w")?)?;
        let Some(unpacked) = &mut self.unpacked else {
            unreachable!("reserved above");
        };
        if q8 {
            k::q8_tiles_to_bf16(stream, w.data, output, input, unpacked.as_mut_slice())?;
        } else {
            let tiles = w.data.cast::<Bf16>().ok_or(LaunchError::InvalidArgument {
                param: "w",
                reason: "bfloat16 tiles of odd size or alignment",
            })?;
            k::bf16_tiles_to_rows(stream, tiles, output, input, unpacked.as_mut_slice())?;
        }
        let rows_w = unpacked.slice(0..output * input);
        self.products
            .linear_rows(stream, x, rows_w, y, rows, input, output, accumulate)
    }

    /// `shape` with the algorithm tuned for `representative`, a shape of the
    /// same structure with sizes rounded up so that tuning is shared
    /// (`Matmul`).
    pub fn matmul(
        &mut self,
        stream: &Stream,
        shape: &MatmulShape,
        representative: &MatmulShape,
        a: DeviceSlice<'_, Bf16>,
        b: DeviceSlice<'_, Bf16>,
        c: DeviceSliceMut<'_, f32>,
    ) -> Result<(), LaunchError> {
        self.products.matmul(stream, shape, representative, a, b, c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_split_like_the_extension() {
        // One 16-row tile of the stop head: 64 blocks over at most 16 splits.
        let plan = plan_wmma(1, 2048, 2, 0);
        assert_eq!((plan.splits, plan.blocks_per_split), (16, 4));
        assert_eq!((plan.partial_floats, plan.counters), (32, 1));
        // The DiT's 22 rows: 160 waves over 64 tiles of 1024 outputs.
        let plan = plan_wmma(22, 1024, 1024, 0);
        assert_eq!((plan.splits, plan.blocks_per_split), (3, 11));
        assert_eq!((plan.partial_floats, plan.counters), (3 * 22 * 1024, 64));
        // Enough tiles: one split, no scratch.
        let plan = plan_wmma(4, 2048, 12288, 0);
        assert_eq!(
            plan,
            WmmaPlan {
                splits: 1,
                blocks_per_split: 64,
                partial_floats: 0,
                counters: 0,
            }
        );
        // No whole block of the reduction or no output: an empty plan.
        assert_eq!(plan_wmma(1, 16, 8, 0), WmmaPlan::default());
        assert_eq!(plan_wmma(1, 64, 0, 0), WmmaPlan::default());
    }

    #[test]
    fn tuning_buckets_round_up_to_128() {
        assert_eq!(tuning_bucket(1), 128);
        assert_eq!(tuning_bucket(128), 128);
        assert_eq!(tuning_bucket(130), 256);
    }
}
