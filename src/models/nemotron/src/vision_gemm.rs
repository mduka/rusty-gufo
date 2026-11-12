//! Products of 16-bit operands (BF16 or FP16) with float32 accumulation and
//! output on hipBLASLt (`kernels/rocm/vision_gemm.cpp`):
//! `y[rows][out] = x[rows][in] * w[out][in]^T`. On gfx1151 hipBLASLt runs
//! them on the matrix cores, where rocBLAS's mixed-precision GEMM does not,
//! and its ranking of algorithms predicts their speed poorly, so its
//! candidates are timed once per shape class (rows rounded up to a power of
//! two, [`BlasLt::matmul_tuned`]). Shapes hipBLASLt can't serve fall back to
//! hipBLAS. Adapted from the OmniVoice extension's wrapper.

use gufo_hip::blas::{self, Blas, BlasElement, BlasLt, Layout, Op, Tuning};
use gufo_hip::{Bf16, DeviceSlice, DeviceSliceMut, F16, Launch, LaunchError, Stream};

/// Format of the matrix-product operands (`VisionOperand`). The 16-bit
/// buffers hold the bits of either; [`VisionOperand::code`] is the kernels'
/// `op` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VisionOperand {
    Bf16,
    #[default]
    F16,
}

impl VisionOperand {
    /// The kernels' operand code: 0 for BF16, 1 for FP16.
    pub fn code(self) -> u32 {
        match self {
            Self::Bf16 => 0,
            Self::F16 => 1,
        }
    }
}

/// Workspace for algorithms that split the reduction.
const WORKSPACE_BYTES: usize = 32 << 20;
/// hipBLASLt results timed per shape class, and timed launches per result.
const TUNING: Tuning = Tuning {
    candidates: 16,
    runs: 2,
};
/// Rows of the shape a class is timed at: classes of more rows share it.
const TUNING_ROWS: usize = 2048;

/// The column-major product of `y[rows][out] = x[rows][in] * w[out][in]^T`:
/// A = w transposed, B = x; the A operation is always set and the B one
/// never, as the C++ descriptors.
fn shape(rows: usize, input: usize, output: usize) -> blas::MatmulShape {
    blas::MatmulShape {
        trans_a: Some(Op::T),
        trans_b: None,
        a: Layout::new(input, output, input),
        b: Layout::new(input, rows, input),
        c: Layout::new(output, rows, output),
        batch: None,
    }
}

/// The shape a product is timed at: rows rounded up to a power of two
/// (`Representative`).
fn representative(rows: usize, input: usize, output: usize) -> blas::MatmulShape {
    let mut rounded = 128;
    while rounded < rows && rounded < TUNING_ROWS {
        rounded *= 2;
    }
    shape(rounded, input, output)
}

/// The BF16 values whose bits `slice` holds.
fn bf16_bits(slice: DeviceSlice<'_, F16>) -> Result<DeviceSlice<'_, Bf16>, LaunchError> {
    slice.cast::<Bf16>().ok_or(LaunchError::InvalidArgument {
        param: "x",
        reason: "operand bits are not whole 16-bit values",
    })
}

/// 16-bit products on one stream (`VisionGemm`). `Send`, not `Sync`.
pub struct VisionGemm {
    operand: VisionOperand,
    lt: Option<BlasLt>,
    fallback: Blas,
}

impl VisionGemm {
    pub fn new(operand: VisionOperand, stream: &Stream) -> Result<Self, LaunchError> {
        Ok(Self {
            operand,
            lt: BlasLt::new(stream, WORKSPACE_BYTES).ok(),
            // The C++ fallback handle keeps hipBLAS's default atomics mode.
            fallback: Blas::with_default_atomics(stream)?,
        })
    }

    pub fn operand(&self) -> VisionOperand {
        self.operand
    }

    /// Times the algorithms of [`linear`](Self::linear)'s shape now, before
    /// it runs.
    pub fn prepare(
        &mut self,
        stream: &Stream,
        rows: usize,
        input: usize,
        output: usize,
    ) -> Result<(), LaunchError> {
        if rows == 0 {
            return Ok(());
        }
        let Some(lt) = &mut self.lt else {
            return Ok(());
        };
        let (exact, class) = (
            shape(rows, input, output),
            representative(rows, input, output),
        );
        match self.operand {
            VisionOperand::Bf16 => {
                lt.prepare_tuned::<Bf16, Bf16, f32>(stream, &exact, &class, false, TUNING)
            }
            VisionOperand::F16 => {
                lt.prepare_tuned::<F16, F16, f32>(stream, &exact, &class, false, TUNING)
            }
        }
    }

    /// `y[rows][out] = x[rows][in] * w[out][in]^T`; `x` and `w` hold operand
    /// bits.
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
    ) -> Result<(), LaunchError> {
        if rows == 0 {
            return Ok(());
        }
        match self.operand {
            VisionOperand::F16 => self.run::<F16>(stream, x, w, y, rows, input, output),
            VisionOperand::Bf16 => {
                self.run::<Bf16>(stream, bf16_bits(x)?, bf16_bits(w)?, y, rows, input, output)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run<T: BlasElement>(
        &mut self,
        stream: &Stream,
        x: DeviceSlice<'_, T>,
        w: DeviceSlice<'_, T>,
        mut y: DeviceSliceMut<'_, f32>,
        rows: usize,
        input: usize,
        output: usize,
    ) -> Result<(), LaunchError> {
        if let Some(lt) = &mut self.lt {
            let launch = lt.matmul_tuned(
                stream,
                &shape(rows, input, output),
                &representative(rows, input, output),
                w,
                x,
                y.reborrow(),
                false,
                TUNING,
            )?;
            if launch == Launch::Launched {
                return Ok(());
            }
        }
        self.fallback.gemm(
            stream,
            Op::T,
            Op::N,
            output,
            rows,
            input,
            w,
            input,
            x,
            input,
            y,
            output,
        )
    }
}
