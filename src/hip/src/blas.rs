//! hipBLAS and hipBLASLt behind checked, graph-aware calls.
//!
//! All library calls go through `shim/blas.cpp`, compiled against the
//! installed ROCm headers. Matrices are column-major as in BLAS: a
//! [`Layout`] of `rows × cols` with leading dimension `ld` needs
//! `(cols - 1) * ld + rows` elements. Operands are checked like launcher
//! arguments (stream, extent) and are retained by graph capture.

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::hash::{BuildHasherDefault, Hasher};
use std::marker::PhantomData;

use crate::error::{HipError, LaunchError};
use crate::launch::{self as l, Arg, Launch};
use crate::memory::{Bf16, DeviceBuffer, DeviceCopy, DeviceSlice, DeviceSliceMut, F16};
use crate::stream::{Event, Stream};
use crate::sys;

const ALGO_BYTES: usize = 32;
const COMPUTE_32F: i32 = 2;
const GEMM_DEFAULT: i32 = 160;

mod ffi {
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Heuristic {
        pub algo: [u8; super::ALGO_BYTES],
        pub workspace_bytes: u64,
        pub state: i32,
        pub index: i32,
        pub waves: f32,
    }

    // SAFETY: declarations match shim/blas.cpp, which the compiler checks
    // against the ROCm headers.
    unsafe extern "C" {
        pub fn gufo_blas_create(handle: *mut *mut c_void, stream: *mut c_void) -> i32;
        pub fn gufo_blas_create_default(handle: *mut *mut c_void, stream: *mut c_void) -> i32;
        pub fn gufo_blas_destroy(handle: *mut c_void) -> i32;
        pub fn gufo_rocblas_create(handle: *mut *mut c_void, stream: *mut c_void) -> i32;
        pub fn gufo_rocblas_destroy(handle: *mut c_void) -> i32;
        pub fn gufo_rocblas_gemm_f32(
            handle: *mut c_void,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            c: *mut f32,
            ldc: i32,
        ) -> i32;
        pub fn gufo_blas_gemm_ex(
            handle: *mut c_void,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const f32,
            a: *const c_void,
            a_type: i32,
            lda: i32,
            b: *const c_void,
            b_type: i32,
            ldb: i32,
            beta: *const f32,
            c: *mut c_void,
            c_type: i32,
            ldc: i32,
            compute: i32,
            algo: i32,
        ) -> i32;
        pub fn gufo_blas_gemm_strided_batched_ex(
            handle: *mut c_void,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const f32,
            a: *const c_void,
            a_type: i32,
            lda: i32,
            stride_a: i64,
            b: *const c_void,
            b_type: i32,
            ldb: i32,
            stride_b: i64,
            beta: *const f32,
            c: *mut c_void,
            c_type: i32,
            ldc: i32,
            stride_c: i64,
            batch_count: i32,
            compute: i32,
            algo: i32,
        ) -> i32;
        pub fn gufo_blas_sgemm(
            handle: *mut c_void,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: *const f32,
            c: *mut f32,
            ldc: i32,
        ) -> i32;
        pub fn gufo_blas_sgemm_strided_batched(
            handle: *mut c_void,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const f32,
            a: *const f32,
            lda: i32,
            stride_a: i64,
            b: *const f32,
            ldb: i32,
            stride_b: i64,
            beta: *const f32,
            c: *mut f32,
            ldc: i32,
            stride_c: i64,
            batch_count: i32,
        ) -> i32;
        pub fn gufo_blaslt_create(handle: *mut *mut c_void) -> i32;
        pub fn gufo_blaslt_destroy(handle: *mut c_void) -> i32;
        pub fn gufo_blaslt_desc_create(desc: *mut *mut c_void, compute: i32, scale: i32) -> i32;
        pub fn gufo_blaslt_desc_set_transpose(desc: *mut c_void, operand: i32, op: i32) -> i32;
        pub fn gufo_blaslt_desc_destroy(desc: *mut c_void) -> i32;
        pub fn gufo_blaslt_layout_create(
            layout: *mut *mut c_void,
            data_type: i32,
            rows: u64,
            cols: u64,
            ld: i64,
        ) -> i32;
        pub fn gufo_blaslt_layout_set_batch(layout: *mut c_void, count: i32, stride: i64) -> i32;
        pub fn gufo_blaslt_layout_destroy(layout: *mut c_void) -> i32;
        pub fn gufo_blaslt_heuristics(
            handle: *mut c_void,
            desc: *mut c_void,
            a: *mut c_void,
            b: *mut c_void,
            c: *mut c_void,
            max_workspace: u64,
            requested: i32,
            results: *mut Heuristic,
            found: *mut i32,
        ) -> i32;
        pub fn gufo_blaslt_resolve_names(handle: *mut c_void, algo: *const u8) -> i32;
        pub fn gufo_blaslt_kernel_name(
            handle: *mut c_void,
            algo: *const u8,
            name: *mut u8,
            capacity: u64,
            length: *mut u64,
        ) -> i32;
        pub fn gufo_blaslt_is_supported(
            handle: *mut c_void,
            desc: *mut c_void,
            a: *mut c_void,
            b: *mut c_void,
            c: *mut c_void,
            algo: *const u8,
            workspace: *mut u64,
        ) -> i32;
        pub fn gufo_blaslt_matmul(
            handle: *mut c_void,
            desc: *mut c_void,
            alpha: *const f32,
            a: *const c_void,
            a_layout: *mut c_void,
            b: *const c_void,
            b_layout: *mut c_void,
            beta: *const f32,
            c: *mut c_void,
            c_layout: *mut c_void,
            algo: *const u8,
            workspace: *mut c_void,
            workspace_bytes: u64,
            stream: *mut c_void,
        ) -> i32;
    }
}

fn blas_check(status: i32, operation: &'static str) -> Result<(), HipError> {
    if status == 0 {
        Ok(())
    } else {
        Err(HipError::blas(status, operation))
    }
}

/// Element types of BLAS operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    F16,
    Bf16,
}

impl DataType {
    /// The `hipDataType` code.
    fn code(self) -> i32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 2,
            Self::Bf16 => 14,
        }
    }
}

/// Device element types with a BLAS data type.
pub trait BlasElement: DeviceCopy {
    const TYPE: DataType;
}

impl BlasElement for f32 {
    const TYPE: DataType = DataType::F32;
}

impl BlasElement for Bf16 {
    const TYPE: DataType = DataType::Bf16;
}

impl BlasElement for F16 {
    const TYPE: DataType = DataType::F16;
}

/// Transpose operation of an operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    N,
    T,
}

impl Op {
    fn code(self) -> i32 {
        match self {
            Self::N => 111,
            Self::T => 112,
        }
    }
}

/// A column-major matrix: `rows × cols`, columns `ld` elements apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Layout {
    pub rows: usize,
    pub cols: usize,
    pub ld: usize,
}

impl Layout {
    pub fn new(rows: usize, cols: usize, ld: usize) -> Self {
        Self { rows, cols, ld }
    }

    /// Elements the matrix spans.
    fn extent(&self, param: &'static str) -> Result<usize, LaunchError> {
        if self.ld < self.rows {
            return Err(LaunchError::InvalidArgument {
                param,
                reason: "leading dimension below the row count",
            });
        }
        if self.cols == 0 || self.rows == 0 {
            return Ok(0);
        }
        l::add(l::mul(self.cols - 1, self.ld, param)?, self.rows, param)
    }

    fn args(&self, name: &'static str) -> [Arg<'static>; 3] {
        [
            Arg::scalar(name, self.rows as u64),
            Arg::scalar(name, self.cols as u64),
            Arg::scalar(name, self.ld as u64),
        ]
    }
}

fn to_i32(value: usize, param: &'static str) -> Result<i32, LaunchError> {
    i32::try_from(value).map_err(|_| LaunchError::ExtentOverflow { param })
}

/// A rocBLAS handle bound to one stream, with atomics disabled, for the f32
/// GEMMs the C++ engines issue through rocBLAS directly (the Qwen3-TTS
/// speech decoder and encoders). Not recordable. `Send`, not `Sync`.
pub struct RocBlas {
    handle: *mut c_void,
    stream_id: u64,
    _not_sync: PhantomData<Cell<()>>,
}

// SAFETY: the handle may move between threads; `!Sync` keeps calls serial.
unsafe impl Send for RocBlas {}

impl RocBlas {
    pub fn new(stream: &Stream) -> Result<Self, HipError> {
        stream.ensure_healthy()?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: valid out-pointer and live stream.
        let status = unsafe { ffi::gufo_rocblas_create(&mut handle, stream.as_raw()) };
        blas_check(status, "rocblas_create_handle")?;
        Ok(Self {
            handle,
            stream_id: stream.id(),
            _not_sync: PhantomData,
        })
    }

    /// `C = op(A) · op(B)` in f32 (alpha 1, beta 0; `rocblas_gemm_ex` with
    /// the standard algorithm): `C` is `m × n`, `op(A)` is `m × k`, `op(B)`
    /// is `k × n`, all column-major.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f32(
        &self,
        stream: &Stream,
        trans_a: Op,
        trans_b: Op,
        m: usize,
        n: usize,
        k: usize,
        a: DeviceSlice<'_, f32>,
        lda: usize,
        b: DeviceSlice<'_, f32>,
        ldb: usize,
        mut c: DeviceSliceMut<'_, f32>,
        ldc: usize,
    ) -> Result<(), LaunchError> {
        l::begin(stream)?;
        if stream.id() != self.stream_id {
            return Err(HipError::refused("rocBLAS handle used on another stream").into());
        }
        if stream.is_recording() {
            return Err(HipError::refused("rocBLAS calls are not recorded").into());
        }
        let stored = |op: Op, rows, cols, ld| match op {
            Op::N => Layout::new(rows, cols, ld),
            Op::T => Layout::new(cols, rows, ld),
        };
        let a_layout = stored(trans_a, m, k, lda);
        let b_layout = stored(trans_b, k, n, ldb);
        let c_layout = Layout::new(m, n, ldc);
        l::require_in(stream, "a", &a, a_layout.extent("a")?)?;
        l::require_in(stream, "b", &b, b_layout.extent("b")?)?;
        l::require_out(stream, "c", &c, c_layout.extent("c")?)?;
        let dims = [
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(k, "k")?,
            to_i32(lda, "lda")?,
            to_i32(ldb, "ldb")?,
            to_i32(ldc, "ldc")?,
        ];
        // SAFETY: every operand belongs to this handle's stream and covers the
        // extent its layout needs; dimensions fit in i32.
        let status = unsafe {
            ffi::gufo_rocblas_gemm_f32(
                self.handle,
                trans_a.code(),
                trans_b.code(),
                dims[0],
                dims[1],
                dims[2],
                a.as_ptr(),
                dims[3],
                b.as_ptr(),
                dims[4],
                c.as_mut_ptr(),
                dims[5],
            )
        };
        Ok(blas_check(status, "rocblas_gemm_ex")?)
    }
}

impl Drop for RocBlas {
    fn drop(&mut self) {
        // SAFETY: the handle came from `gufo_rocblas_create` and is dropped once.
        let _ = unsafe { ffi::gufo_rocblas_destroy(self.handle) };
    }
}

/// The problems of a strided batched GEMM: `count` of them, whose operands
/// start `a`, `b` and `c` elements after the previous problem's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strides {
    pub count: usize,
    pub a: usize,
    pub b: usize,
    pub c: usize,
}

impl Strides {
    /// Whether the outputs of `m`-row problems with leading dimension `ldc`
    /// (`matrix` elements each) are disjoint: each starts past the previous
    /// one's end, or they interleave, problem `i` owning rows
    /// `i * c .. i * c + m` of every column of `ldc` rows.
    fn disjoint_outputs(&self, m: usize, ldc: usize, matrix: usize) -> bool {
        self.count <= 1
            || self.c >= matrix
            || (self.c >= m
                && (self.count - 1)
                    .checked_mul(self.c)
                    .and_then(|start| start.checked_add(m))
                    .is_some_and(|rows| rows <= ldc))
    }

    /// Elements a batched operand spans: the last problem's start plus one
    /// matrix.
    fn extent(
        &self,
        stride: usize,
        matrix: usize,
        param: &'static str,
    ) -> Result<usize, LaunchError> {
        if self.count == 0 || matrix == 0 {
            return Ok(0);
        }
        l::add(l::mul(self.count - 1, stride, param)?, matrix, param)
    }
}

/// A hipBLAS handle bound to one stream, with atomics disabled so results
/// are deterministic (as in the C++ engines). `Send`, not `Sync`.
pub struct Blas {
    handle: *mut c_void,
    stream_id: u64,
    _not_sync: PhantomData<Cell<()>>,
}

// SAFETY: the handle may move between threads; `!Sync` keeps calls serial.
unsafe impl Send for Blas {}

impl Blas {
    pub fn new(stream: &Stream) -> Result<Self, HipError> {
        Self::create(stream, true)
    }

    /// A handle that keeps hipBLAS's default atomics mode, for engines whose
    /// C++ code never sets it (Qwen-Image-2.1): the mode can change which
    /// kernels run.
    pub fn with_default_atomics(stream: &Stream) -> Result<Self, HipError> {
        Self::create(stream, false)
    }

    fn create(stream: &Stream, deterministic: bool) -> Result<Self, HipError> {
        stream.ensure_healthy()?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: valid out-pointer and live stream.
        let status = unsafe {
            if deterministic {
                ffi::gufo_blas_create(&mut handle, stream.as_raw())
            } else {
                ffi::gufo_blas_create_default(&mut handle, stream.as_raw())
            }
        };
        blas_check(status, "hipblasCreate")?;
        Ok(Self {
            handle,
            stream_id: stream.id(),
            _not_sync: PhantomData,
        })
    }

    /// `C = op(A) · op(B)` with f32 compute (alpha 1, beta 0): `C` is
    /// `m × n`, `op(A)` is `m × k`, `op(B)` is `k × n`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm<A: BlasElement, B: BlasElement, C: BlasElement>(
        &self,
        stream: &Stream,
        trans_a: Op,
        trans_b: Op,
        m: usize,
        n: usize,
        k: usize,
        a: DeviceSlice<'_, A>,
        lda: usize,
        b: DeviceSlice<'_, B>,
        ldb: usize,
        mut c: DeviceSliceMut<'_, C>,
        ldc: usize,
    ) -> Result<(), LaunchError> {
        l::begin(stream)?;
        if stream.id() != self.stream_id {
            return Err(HipError::refused("hipBLAS handle used on another stream").into());
        }
        let stored = |op: Op, rows, cols, ld| match op {
            Op::N => Layout::new(rows, cols, ld),
            Op::T => Layout::new(cols, rows, ld),
        };
        let a_layout = stored(trans_a, m, k, lda);
        let b_layout = stored(trans_b, k, n, ldb);
        let c_layout = Layout::new(m, n, ldc);
        l::require_in(stream, "a", &a, a_layout.extent("a")?)?;
        l::require_in(stream, "b", &b, b_layout.extent("b")?)?;
        l::require_out(stream, "c", &c, c_layout.extent("c")?)?;
        let dims = [
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(k, "k")?,
            to_i32(lda, "lda")?,
            to_i32(ldb, "ldb")?,
            to_i32(ldc, "ldc")?,
        ];
        if stream.is_recording()
            && l::record(
                stream,
                "hipblasGemmEx",
                &[
                    Arg::buffer("a", Some(a.allocation()), a.address(), a.len()),
                    Arg::buffer("b", Some(b.allocation()), b.address(), b.len()),
                    Arg::buffer("c", Some(c.allocation()), c.address(), c.len()),
                    Arg::scalar("trans_a", trans_a.code() as u64),
                    Arg::scalar("trans_b", trans_b.code() as u64),
                    Arg::scalar("m", m as u64),
                    Arg::scalar("n", n as u64),
                    Arg::scalar("k", k as u64),
                    Arg::scalar("lda", lda as u64),
                    Arg::scalar("ldb", ldb as u64),
                    Arg::scalar("ldc", ldc as u64),
                ],
            )?
            .is_some()
        {
            return Ok(());
        }
        let (alpha, beta) = (1.0f32, 0.0f32);
        // SAFETY: every operand belongs to this handle's stream and covers the
        // extent its layout needs; dimensions fit in i32.
        let status = unsafe {
            ffi::gufo_blas_gemm_ex(
                self.handle,
                trans_a.code(),
                trans_b.code(),
                dims[0],
                dims[1],
                dims[2],
                &alpha,
                a.as_ptr().cast(),
                A::TYPE.code(),
                dims[3],
                b.as_ptr().cast(),
                B::TYPE.code(),
                dims[4],
                &beta,
                c.as_mut_ptr().cast(),
                C::TYPE.code(),
                dims[5],
                COMPUTE_32F,
                GEMM_DEFAULT,
            )
        };
        Ok(blas_check(status, "hipblasGemmEx")?)
    }

    /// `batch.count` problems `C_i = op(A_i) · op(B_i)` as [`Blas::gemm`],
    /// operand `i` starting `i` strides into its slice. Output problems must
    /// not overlap: `batch.c` covers at least one `C` matrix.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_strided_batched<A: BlasElement, B: BlasElement, C: BlasElement>(
        &self,
        stream: &Stream,
        trans_a: Op,
        trans_b: Op,
        m: usize,
        n: usize,
        k: usize,
        a: DeviceSlice<'_, A>,
        lda: usize,
        b: DeviceSlice<'_, B>,
        ldb: usize,
        mut c: DeviceSliceMut<'_, C>,
        ldc: usize,
        batch: Strides,
    ) -> Result<(), LaunchError> {
        l::begin(stream)?;
        if stream.id() != self.stream_id {
            return Err(HipError::refused("hipBLAS handle used on another stream").into());
        }
        let stored = |op: Op, rows, cols, ld| match op {
            Op::N => Layout::new(rows, cols, ld),
            Op::T => Layout::new(cols, rows, ld),
        };
        let a_matrix = stored(trans_a, m, k, lda).extent("a")?;
        let b_matrix = stored(trans_b, k, n, ldb).extent("b")?;
        let c_matrix = Layout::new(m, n, ldc).extent("c")?;
        if !batch.disjoint_outputs(m, ldc, c_matrix) {
            return Err(LaunchError::InvalidArgument {
                param: "c",
                reason: "output problems overlap",
            });
        }
        l::require_in(stream, "a", &a, batch.extent(batch.a, a_matrix, "a")?)?;
        l::require_in(stream, "b", &b, batch.extent(batch.b, b_matrix, "b")?)?;
        l::require_out(stream, "c", &c, batch.extent(batch.c, c_matrix, "c")?)?;
        let dims = [
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(k, "k")?,
            to_i32(lda, "lda")?,
            to_i32(ldb, "ldb")?,
            to_i32(ldc, "ldc")?,
            to_i32(batch.count, "batch")?,
        ];
        let stride =
            |s: usize, param| i64::try_from(s).map_err(|_| LaunchError::ExtentOverflow { param });
        let strides = [
            stride(batch.a, "a")?,
            stride(batch.b, "b")?,
            stride(batch.c, "c")?,
        ];
        if stream.is_recording()
            && l::record(
                stream,
                "hipblasGemmStridedBatchedEx",
                &[
                    Arg::buffer("a", Some(a.allocation()), a.address(), a.len()),
                    Arg::buffer("b", Some(b.allocation()), b.address(), b.len()),
                    Arg::buffer("c", Some(c.allocation()), c.address(), c.len()),
                    Arg::scalar("trans_a", trans_a.code() as u64),
                    Arg::scalar("trans_b", trans_b.code() as u64),
                    Arg::scalar("m", m as u64),
                    Arg::scalar("n", n as u64),
                    Arg::scalar("k", k as u64),
                    Arg::scalar("lda", lda as u64),
                    Arg::scalar("ldb", ldb as u64),
                    Arg::scalar("ldc", ldc as u64),
                    Arg::scalar("stride_a", batch.a as u64),
                    Arg::scalar("stride_b", batch.b as u64),
                    Arg::scalar("stride_c", batch.c as u64),
                    Arg::scalar("batch", batch.count as u64),
                ],
            )?
            .is_some()
        {
            return Ok(());
        }
        let (alpha, beta) = (1.0f32, 0.0f32);
        // SAFETY: every operand belongs to this handle's stream and covers the
        // extent of all its problems; output problems are disjoint; dimensions
        // fit in i32 and strides in i64.
        let status = unsafe {
            ffi::gufo_blas_gemm_strided_batched_ex(
                self.handle,
                trans_a.code(),
                trans_b.code(),
                dims[0],
                dims[1],
                dims[2],
                &alpha,
                a.as_ptr().cast(),
                A::TYPE.code(),
                dims[3],
                strides[0],
                b.as_ptr().cast(),
                B::TYPE.code(),
                dims[4],
                strides[1],
                &beta,
                c.as_mut_ptr().cast(),
                C::TYPE.code(),
                dims[5],
                strides[2],
                dims[6],
                COMPUTE_32F,
                GEMM_DEFAULT,
            )
        };
        Ok(blas_check(status, "hipblasGemmStridedBatchedEx")?)
    }

    /// The raw handle, for C++ launch code that borrows it on this handle's
    /// stream while its engine moves to Rust (DeepSeek V4 Flash).
    pub fn as_raw(&self) -> *mut c_void {
        self.handle
    }

    /// `C = alpha · op(A) · op(B)` through `hipblasSgemm` (beta 0), for
    /// engines whose C++ code called it: its kernel choice can differ from
    /// [`Blas::gemm`]'s for the same F32 problem.
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm(
        &self,
        stream: &Stream,
        trans_a: Op,
        trans_b: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: DeviceSlice<'_, f32>,
        lda: usize,
        b: DeviceSlice<'_, f32>,
        ldb: usize,
        mut c: DeviceSliceMut<'_, f32>,
        ldc: usize,
    ) -> Result<(), LaunchError> {
        l::begin(stream)?;
        if stream.id() != self.stream_id {
            return Err(HipError::refused("hipBLAS handle used on another stream").into());
        }
        let stored = |op: Op, rows, cols, ld| match op {
            Op::N => Layout::new(rows, cols, ld),
            Op::T => Layout::new(cols, rows, ld),
        };
        l::require_in(stream, "a", &a, stored(trans_a, m, k, lda).extent("a")?)?;
        l::require_in(stream, "b", &b, stored(trans_b, k, n, ldb).extent("b")?)?;
        l::require_out(stream, "c", &c, Layout::new(m, n, ldc).extent("c")?)?;
        let dims = [
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(k, "k")?,
            to_i32(lda, "lda")?,
            to_i32(ldb, "ldb")?,
            to_i32(ldc, "ldc")?,
        ];
        if stream.is_recording()
            && l::record(
                stream,
                "hipblasSgemm",
                &[
                    Arg::buffer("a", Some(a.allocation()), a.address(), a.len()),
                    Arg::buffer("b", Some(b.allocation()), b.address(), b.len()),
                    Arg::buffer("c", Some(c.allocation()), c.address(), c.len()),
                    Arg::scalar("trans_a", trans_a.code() as u64),
                    Arg::scalar("trans_b", trans_b.code() as u64),
                    Arg::scalar("m", m as u64),
                    Arg::scalar("n", n as u64),
                    Arg::scalar("k", k as u64),
                    Arg::scalar("alpha", u64::from(alpha.to_bits())),
                    Arg::scalar("lda", lda as u64),
                    Arg::scalar("ldb", ldb as u64),
                    Arg::scalar("ldc", ldc as u64),
                ],
            )?
            .is_some()
        {
            return Ok(());
        }
        let beta = 0.0f32;
        // SAFETY: every operand belongs to this handle's stream and covers the
        // extent its layout needs; dimensions fit in i32.
        let status = unsafe {
            ffi::gufo_blas_sgemm(
                self.handle,
                trans_a.code(),
                trans_b.code(),
                dims[0],
                dims[1],
                dims[2],
                &alpha,
                a.as_ptr(),
                dims[3],
                b.as_ptr(),
                dims[4],
                &beta,
                c.as_mut_ptr(),
                dims[5],
            )
        };
        Ok(blas_check(status, "hipblasSgemm")?)
    }

    /// `batch.count` problems `C_i = alpha · op(A_i) · op(B_i)` through
    /// `hipblasSgemmStridedBatched`, operand `i` starting `i` strides into its
    /// slice (a stride may be 0 to share an operand). Output problems must
    /// not overlap.
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_strided_batched(
        &self,
        stream: &Stream,
        trans_a: Op,
        trans_b: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: DeviceSlice<'_, f32>,
        lda: usize,
        b: DeviceSlice<'_, f32>,
        ldb: usize,
        mut c: DeviceSliceMut<'_, f32>,
        ldc: usize,
        batch: Strides,
    ) -> Result<(), LaunchError> {
        l::begin(stream)?;
        if stream.id() != self.stream_id {
            return Err(HipError::refused("hipBLAS handle used on another stream").into());
        }
        let stored = |op: Op, rows, cols, ld| match op {
            Op::N => Layout::new(rows, cols, ld),
            Op::T => Layout::new(cols, rows, ld),
        };
        let a_matrix = stored(trans_a, m, k, lda).extent("a")?;
        let b_matrix = stored(trans_b, k, n, ldb).extent("b")?;
        let c_matrix = Layout::new(m, n, ldc).extent("c")?;
        if !batch.disjoint_outputs(m, ldc, c_matrix) {
            return Err(LaunchError::InvalidArgument {
                param: "c",
                reason: "output problems overlap",
            });
        }
        l::require_in(stream, "a", &a, batch.extent(batch.a, a_matrix, "a")?)?;
        l::require_in(stream, "b", &b, batch.extent(batch.b, b_matrix, "b")?)?;
        l::require_out(stream, "c", &c, batch.extent(batch.c, c_matrix, "c")?)?;
        let dims = [
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(k, "k")?,
            to_i32(lda, "lda")?,
            to_i32(ldb, "ldb")?,
            to_i32(ldc, "ldc")?,
            to_i32(batch.count, "batch")?,
        ];
        let stride =
            |s: usize, param| i64::try_from(s).map_err(|_| LaunchError::ExtentOverflow { param });
        let strides = [
            stride(batch.a, "a")?,
            stride(batch.b, "b")?,
            stride(batch.c, "c")?,
        ];
        if stream.is_recording()
            && l::record(
                stream,
                "hipblasSgemmStridedBatched",
                &[
                    Arg::buffer("a", Some(a.allocation()), a.address(), a.len()),
                    Arg::buffer("b", Some(b.allocation()), b.address(), b.len()),
                    Arg::buffer("c", Some(c.allocation()), c.address(), c.len()),
                    Arg::scalar("trans_a", trans_a.code() as u64),
                    Arg::scalar("trans_b", trans_b.code() as u64),
                    Arg::scalar("m", m as u64),
                    Arg::scalar("n", n as u64),
                    Arg::scalar("k", k as u64),
                    Arg::scalar("alpha", u64::from(alpha.to_bits())),
                    Arg::scalar("lda", lda as u64),
                    Arg::scalar("ldb", ldb as u64),
                    Arg::scalar("ldc", ldc as u64),
                    Arg::scalar("stride_a", batch.a as u64),
                    Arg::scalar("stride_b", batch.b as u64),
                    Arg::scalar("stride_c", batch.c as u64),
                    Arg::scalar("batch", batch.count as u64),
                ],
            )?
            .is_some()
        {
            return Ok(());
        }
        let beta = 0.0f32;
        // SAFETY: every operand belongs to this handle's stream and covers the
        // extent of all its problems; output problems are disjoint; dimensions
        // fit in i32 and strides in i64.
        let status = unsafe {
            ffi::gufo_blas_sgemm_strided_batched(
                self.handle,
                trans_a.code(),
                trans_b.code(),
                dims[0],
                dims[1],
                dims[2],
                &alpha,
                a.as_ptr(),
                dims[3],
                strides[0],
                b.as_ptr(),
                dims[4],
                strides[1],
                &beta,
                c.as_mut_ptr(),
                dims[5],
                strides[2],
                dims[6],
            )
        };
        Ok(blas_check(status, "hipblasSgemmStridedBatched")?)
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        // SAFETY: created by gufo_blas_create, destroyed once.
        unsafe {
            let _ = ffi::gufo_blas_destroy(self.handle);
        }
    }
}

/// One ranked hipBLASLt heuristic result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Heuristic {
    /// hipBLASLt's solution index (`hipblaslt_ext::getIndexFromAlgo`).
    pub index: i32,
    pub workspace_bytes: u64,
    /// `hipblasStatus_t` of the result; only 0 is valid.
    pub state: i32,
    pub waves: f32,
}

impl Heuristic {
    pub fn is_usable(&self, workspace_capacity: usize) -> bool {
        self.state == 0 && self.workspace_bytes <= workspace_capacity as u64
    }
}

/// How a model picks hipBLASLt algorithms. The choice decides which kernel
/// runs and therefore the accumulation order, so it belongs to the model and
/// its numerical contract.
pub trait AlgoPolicy {
    /// Heuristic results to request.
    fn requested(&self) -> usize {
        16
    }

    /// Workspace offered to hipBLASLt. Zero makes workspace-using algorithms
    /// unusable, which also changes the ranked list.
    fn workspace_limit(&self) -> usize {
        0
    }

    /// Index into `ranked` to install, or `None` to fall back.
    fn choose(&self, ranked: &[Heuristic], workspace_capacity: usize) -> Option<usize>;

    /// Install only a result that `matmulIsAlgoSupported` accepts without
    /// workspace, asked in ranked order for the results that are valid and
    /// workspace-free, stopping at the first (the Flash-Next C++ plans; the
    /// question itself touches hipBLASLt's solution library). `choose` is
    /// not consulted then.
    fn verify_supported(&self) -> bool {
        false
    }

    /// Resolve solution and kernel names after installing an algorithm. This
    /// initialises hipBLASLt's solution library, which changes the heuristic
    /// lists of later shapes; the C++ engines do it.
    fn resolve_names(&self) -> bool {
        true
    }

    /// Candidates to launch in this order with a shape's first operands,
    /// installing the first that launches (the DeepSeek V4 Flash C++ plans);
    /// usable ones only are tried. `None`, the default, installs `choose`'s
    /// pick without launching it. Such a policy plans on its first `matmul`
    /// only, not in `prepare`.
    fn trial_order(&self, ranked: &[Heuristic], workspace_capacity: usize) -> Option<Vec<usize>> {
        let _ = (ranked, workspace_capacity);
        None
    }

    /// Times every usable result on a shape's first operands and installs
    /// the fastest (the EmbeddingGemma 2 C++ plans): each runs once untimed,
    /// which loads its kernel, then this many times between two timed
    /// events; only a strictly faster result replaces an earlier one, and one
    /// whose launch or timing fails is skipped. The call then launches the
    /// installed plan for its result. `None`, the default, installs
    /// `choose`'s pick without launching it. Such a policy plans on its first
    /// `matmul` only, not in `prepare`, and is not combined with
    /// `trial_order`.
    fn timed_runs(&self) -> Option<usize> {
        None
    }
}

/// Installs the first usable result.
pub struct FirstUsable;

impl AlgoPolicy for FirstUsable {
    fn choose(&self, ranked: &[Heuristic], workspace_capacity: usize) -> Option<usize> {
        ranked.iter().position(|h| h.is_usable(workspace_capacity))
    }
}

/// A matmul `C = op(A) · op(B)` by the stored layouts of its operands.
/// Transposes left `None` are not set at all: setting hipBLASLt's default
/// explicitly changes its heuristic results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatmulShape {
    pub trans_a: Option<Op>,
    pub trans_b: Option<Op>,
    pub a: Layout,
    pub b: Layout,
    pub c: Layout,
    /// Strided batches, set on every layout; `None` for one matmul.
    pub batch: Option<MatmulBatch>,
}

/// `count` matmuls whose operands start `a`, `b` and `c` elements after the
/// previous one's (hipBLASLt's batch count and strided batch offsets).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatmulBatch {
    pub count: usize,
    pub a: usize,
    pub b: usize,
    pub c: usize,
}

impl MatmulShape {
    /// Elements an operand spans over every batch.
    fn extent(
        &self,
        layout: &Layout,
        stride: impl Fn(&MatmulBatch) -> usize,
        param: &'static str,
    ) -> Result<usize, LaunchError> {
        let matrix = layout.extent(param)?;
        match &self.batch {
            Some(batch) if batch.count > 1 && matrix > 0 => l::add(
                l::mul(batch.count - 1, stride(batch), param)?,
                matrix,
                param,
            ),
            _ => Ok(matrix),
        }
    }
}

struct Plan {
    desc: *mut c_void,
    layouts: [*mut c_void; 3],
    algo: [u8; ALGO_BYTES],
    index: i32,
    workspace_bytes: u64,
}

impl Drop for Plan {
    fn drop(&mut self) {
        // SAFETY: each handle was created by the shim and is destroyed once;
        // null handles (failed creation) are skipped.
        unsafe {
            for layout in self.layouts.into_iter().filter(|l| !l.is_null()) {
                let _ = ffi::gufo_blaslt_layout_destroy(layout);
            }
            if !self.desc.is_null() {
                let _ = ffi::gufo_blaslt_desc_destroy(self.desc);
            }
        }
    }
}

/// A shape's descriptors and ranked results before the policy's choice.
struct Draft {
    plan: Plan,
    raw: Vec<ffi::Heuristic>,
    ranked: Vec<Heuristic>,
    capacity: usize,
}

type PlanKey = (MatmulShape, DataType, DataType, DataType);

/// FxHash's word mixing, for the plan cache on the launch path: its keys are
/// a few integers of trusted shapes, and SipHash would cost more than the
/// lookup itself (several hashes per GEMM of a tiled convolution).
#[derive(Default, Clone, Copy)]
struct WordHasher(u64);

impl WordHasher {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for WordHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }

    fn write_u8(&mut self, value: u8) {
        self.add(u64::from(value));
    }

    fn write_u32(&mut self, value: u32) {
        self.add(u64::from(value));
    }

    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    fn write_isize(&mut self, value: isize) {
        self.add(value as u64);
    }
}

type PlanMap = HashMap<PlanKey, Option<Plan>, BuildHasherDefault<WordHasher>>;

/// A hipBLASLt handle with a per-shape plan cache and a workspace on one
/// stream. `Send`, not `Sync`.
pub struct BlasLt {
    handle: *mut c_void,
    workspace: Option<DeviceBuffer<u8>>,
    stream_id: u64,
    plans: PlanMap,
    _not_sync: PhantomData<Cell<()>>,
}

// SAFETY: handle and plans may move between threads; `!Sync` keeps calls serial.
unsafe impl Send for BlasLt {}

impl BlasLt {
    pub fn new(stream: &Stream, workspace_bytes: usize) -> Result<Self, HipError> {
        stream.ensure_healthy()?;
        let workspace = if workspace_bytes > 0 {
            Some(stream.alloc::<u8>(workspace_bytes)?)
        } else {
            None
        };
        let mut handle = std::ptr::null_mut();
        // SAFETY: valid out-pointer.
        let status = unsafe { ffi::gufo_blaslt_create(&mut handle) };
        blas_check(status, "hipblasLtCreate")?;
        Ok(Self {
            handle,
            workspace,
            stream_id: stream.id(),
            plans: PlanMap::default(),
            _not_sync: PhantomData,
        })
    }

    /// The raw handle, for C++ launch code that borrows it while its engine
    /// moves to Rust (DeepSeek V4 Flash); the code keeps its own plans.
    pub fn as_raw(&self) -> *mut c_void {
        self.handle
    }

    /// Cached shapes, including those hipBLASLt cannot serve.
    pub fn plans(&self) -> usize {
        self.plans.len()
    }

    /// The installed solution index for a shape, once planned.
    pub fn installed_index<A: BlasElement, B: BlasElement, C: BlasElement>(
        &self,
        shape: &MatmulShape,
    ) -> Option<i32> {
        let key = (*shape, A::TYPE, B::TYPE, C::TYPE);
        self.plans.get(&key)?.as_ref().map(|plan| plan.index)
    }

    /// Plans a shape with `policy` without running it, as the first `matmul`
    /// of the shape would; the installed solution index, or `None` when
    /// hipBLASLt cannot serve the shape. Not while a graph is recorded.
    pub fn prepare<A: BlasElement, B: BlasElement, C: BlasElement>(
        &mut self,
        stream: &Stream,
        shape: &MatmulShape,
        policy: &dyn AlgoPolicy,
    ) -> Result<Option<i32>, LaunchError> {
        let key = (*shape, A::TYPE, B::TYPE, C::TYPE);
        if !self.plans.contains_key(&key) {
            if stream.is_recording() {
                return Err(HipError::capture_violation("hipBLASLt planning").into());
            }
            let plan = match self.describe(&key, policy) {
                Some(draft)
                    if policy.trial_order(&draft.ranked, draft.capacity).is_some()
                        || policy.timed_runs().is_some() =>
                {
                    return Err(HipError::refused(
                        "trial-launch policies plan on their first matmul",
                    )
                    .into());
                }
                Some(draft) => self.choose_plan(draft, policy),
                None => None,
            };
            self.plans.insert(key, plan);
        }
        Ok(self.installed_index::<A, B, C>(shape))
    }

    /// The kernel name of the algorithm installed for a planned shape
    /// (`hipblaslt_ext::getKernelNameFromAlgo`); `None` for an unplanned or
    /// unserved shape. The lookup initialises hipBLASLt's solution library,
    /// which changes the heuristic lists of later shapes.
    pub fn installed_kernel_name<A: BlasElement, B: BlasElement, C: BlasElement>(
        &self,
        shape: &MatmulShape,
    ) -> Result<Option<String>, HipError> {
        let key = (*shape, A::TYPE, B::TYPE, C::TYPE);
        let Some(plan) = self.plans.get(&key).and_then(Option::as_ref) else {
            return Ok(None);
        };
        let mut name = vec![0u8; 256];
        loop {
            let mut length = 0u64;
            // SAFETY: `name` holds `name.len()` writable bytes; the handle is
            // live and the algorithm was installed by the shim.
            let status = unsafe {
                ffi::gufo_blaslt_kernel_name(
                    self.handle,
                    plan.algo.as_ptr(),
                    name.as_mut_ptr(),
                    name.len() as u64,
                    &mut length,
                )
            };
            blas_check(status, "hipblaslt_ext::getKernelNameFromAlgo")?;
            let length = usize::try_from(length)
                .map_err(|_| HipError::refused("hipBLASLt kernel name length"))?;
            if length <= name.len() {
                name.truncate(length);
                return Ok(Some(String::from_utf8_lossy(&name).into_owned()));
            }
            name.resize(length, 0);
        }
    }

    /// `C = op(A) · op(B)` with f32 compute (alpha 1, beta 0). The first call
    /// of a shape plans it with `policy`; that must not happen while a graph
    /// is recorded (warm-up plans, capture replays the plan).
    pub fn matmul<A: BlasElement, B: BlasElement, C: BlasElement>(
        &mut self,
        stream: &Stream,
        shape: &MatmulShape,
        a: DeviceSlice<'_, A>,
        b: DeviceSlice<'_, B>,
        mut c: DeviceSliceMut<'_, C>,
        policy: &dyn AlgoPolicy,
    ) -> Result<Launch, LaunchError> {
        l::begin(stream)?;
        if stream.id() != self.stream_id {
            return Err(HipError::refused("hipBLASLt workspace used on another stream").into());
        }
        l::require_in(stream, "a", &a, shape.extent(&shape.a, |b| b.a, "a")?)?;
        l::require_in(stream, "b", &b, shape.extent(&shape.b, |b| b.b, "b")?)?;
        l::require_out(stream, "c", &c, shape.extent(&shape.c, |b| b.c, "c")?)?;
        let key = (*shape, A::TYPE, B::TYPE, C::TYPE);
        if !self.plans.contains_key(&key) {
            if stream.is_recording() {
                return Err(HipError::capture_violation("hipBLASLt planning").into());
            }
            let plan = match self.describe(&key, policy) {
                Some(draft) => match policy.trial_order(&draft.ranked, draft.capacity) {
                    Some(order) => {
                        let installed = self.trial(draft, &order, policy, stream, a, b, c);
                        let launch = if installed.is_some() {
                            Launch::Launched
                        } else {
                            Launch::Unsupported
                        };
                        self.plans.insert(key, installed);
                        return Ok(launch);
                    }
                    None => match policy.timed_runs() {
                        Some(runs) => self.fastest(draft, runs, policy, stream, a, b, c.reborrow()),
                        None => self.choose_plan(draft, policy),
                    },
                },
                None => None,
            };
            self.plans.insert(key, plan);
        }
        let Some(plan) = self.plans.get(&key).and_then(Option::as_ref) else {
            return Ok(Launch::Unsupported);
        };
        let workspace_ptr = self
            .workspace
            .as_mut()
            .map_or(std::ptr::null_mut(), |w| w.as_mut_slice().as_mut_ptr())
            .cast::<c_void>();
        if stream.is_recording() && shape.batch.is_some() {
            return Err(HipError::capture_violation("batched hipBLASLt matmul").into());
        }
        if stream.is_recording() {
            let workspace_arg = match &self.workspace {
                Some(w) => Arg::buffer(
                    "workspace",
                    Some(w.allocation()),
                    w.as_slice().address(),
                    w.len(),
                ),
                None => Arg::buffer("workspace", None, 0, 0),
            };
            let [ar, ac, ald] = shape.a.args("a_layout");
            let [br, bc, bld] = shape.b.args("b_layout");
            let [cr, cc, cld] = shape.c.args("c_layout");
            let recorded = l::record(
                stream,
                "hipblasLtMatmul",
                &[
                    Arg::buffer("a", Some(a.allocation()), a.address(), a.len()),
                    Arg::buffer("b", Some(b.allocation()), b.address(), b.len()),
                    Arg::buffer("c", Some(c.allocation()), c.address(), c.len()),
                    workspace_arg,
                    Arg::scalar("algo", plan.index as u64),
                    Arg::scalar("workspace_bytes", plan.workspace_bytes),
                    ar,
                    ac,
                    ald,
                    br,
                    bc,
                    bld,
                    cr,
                    cc,
                    cld,
                ],
            )?;
            if let Some(result) = recorded {
                return Ok(result);
            }
        }
        let (alpha, beta) = (1.0f32, 0.0f32);
        // SAFETY: operands belong to this workspace's stream and cover their
        // layouts; the plan's handles and algorithm come from the shim.
        let status = unsafe {
            ffi::gufo_blaslt_matmul(
                self.handle,
                plan.desc,
                &alpha,
                a.as_ptr().cast(),
                plan.layouts[0],
                b.as_ptr().cast(),
                plan.layouts[1],
                &beta,
                c.as_mut_ptr().cast(),
                plan.layouts[2],
                plan.algo.as_ptr(),
                workspace_ptr,
                plan.workspace_bytes,
                stream.as_raw(),
            )
        };
        blas_check(status, "hipblasLtMatmul")?;
        Ok(Launch::Launched)
    }

    /// Descriptors and hipBLASLt's ranked results for one shape; `None` for a
    /// shape hipBLASLt cannot serve, which the cache keeps, like the C++
    /// engines do.
    fn describe(&self, key: &PlanKey, policy: &dyn AlgoPolicy) -> Option<Draft> {
        let (shape, a_type, b_type, c_type) = *key;
        let mut plan = Plan {
            desc: std::ptr::null_mut(),
            layouts: [std::ptr::null_mut(); 3],
            algo: [0; ALGO_BYTES],
            index: -1,
            workspace_bytes: 0,
        };
        // SAFETY: out-pointers are valid; created handles are owned by `plan`
        // and destroyed by its Drop on every early return.
        unsafe {
            if ffi::gufo_blaslt_desc_create(&mut plan.desc, COMPUTE_32F, DataType::F32.code()) != 0
            {
                return None;
            }
            for (operand, op) in [(0, shape.trans_a), (1, shape.trans_b)] {
                if let Some(op) = op
                    && ffi::gufo_blaslt_desc_set_transpose(plan.desc, operand, op.code()) != 0
                {
                    return None;
                }
            }
            let layouts = [(a_type, shape.a), (b_type, shape.b), (c_type, shape.c)];
            for (slot, (data_type, layout)) in plan.layouts.iter_mut().zip(layouts) {
                let ld = i64::try_from(layout.ld).ok()?;
                if ffi::gufo_blaslt_layout_create(
                    slot,
                    data_type.code(),
                    layout.rows as u64,
                    layout.cols as u64,
                    ld,
                ) != 0
                {
                    return None;
                }
            }
            if let Some(batch) = shape.batch {
                let count = i32::try_from(batch.count).ok()?;
                for (layout, stride) in plan.layouts.iter().zip([batch.a, batch.b, batch.c]) {
                    let stride = i64::try_from(stride).ok()?;
                    if ffi::gufo_blaslt_layout_set_batch(*layout, count, stride) != 0 {
                        return None;
                    }
                }
            }
        }
        let capacity = policy
            .workspace_limit()
            .min(self.workspace.as_ref().map_or(0, DeviceBuffer::len));
        let requested = policy.requested().clamp(1, 64);
        let mut raw = vec![
            ffi::Heuristic {
                algo: [0; ALGO_BYTES],
                workspace_bytes: 0,
                state: 0,
                index: 0,
                waves: 0.0,
            };
            requested
        ];
        let mut found = 0i32;
        // SAFETY: `raw` holds `requested` results; handles are live.
        let status = unsafe {
            ffi::gufo_blaslt_heuristics(
                self.handle,
                plan.desc,
                plan.layouts[0],
                plan.layouts[1],
                plan.layouts[2],
                capacity as u64,
                requested as i32,
                raw.as_mut_ptr(),
                &mut found,
            )
        };
        if status != 0 || found <= 0 {
            return None;
        }
        raw.truncate(found as usize);
        let ranked: Vec<Heuristic> = raw
            .iter()
            .map(|h| Heuristic {
                index: h.index,
                workspace_bytes: h.workspace_bytes,
                state: h.state,
                waves: h.waves,
            })
            .collect();
        Some(Draft {
            plan,
            raw,
            ranked,
            capacity,
        })
    }

    /// The policy's choice from a shape's ranked results, installed unlaunched.
    fn choose_plan(&self, draft: Draft, policy: &dyn AlgoPolicy) -> Option<Plan> {
        let Draft {
            plan,
            raw,
            ranked,
            capacity,
        } = draft;
        let chosen = if policy.verify_supported() {
            ranked.iter().zip(&raw).position(|(h, result)| {
                if h.state != 0 || h.workspace_bytes != 0 {
                    return false;
                }
                let mut workspace = 0u64;
                // SAFETY: live handles and an algorithm hipBLASLt returned.
                let status = unsafe {
                    ffi::gufo_blaslt_is_supported(
                        self.handle,
                        plan.desc,
                        plan.layouts[0],
                        plan.layouts[1],
                        plan.layouts[2],
                        result.algo.as_ptr(),
                        &mut workspace,
                    )
                };
                status == 0 && workspace == 0
            })?
        } else {
            policy.choose(&ranked, capacity)?
        };
        let result = raw.get(chosen)?;
        if !ranked[chosen].is_usable(capacity) {
            return None;
        }
        Some(self.install(plan, result, policy))
    }

    /// `plan` with the algorithm of `result`.
    fn install(&self, mut plan: Plan, result: &ffi::Heuristic, policy: &dyn AlgoPolicy) -> Plan {
        plan.algo = result.algo;
        plan.index = result.index;
        plan.workspace_bytes = result.workspace_bytes;
        if policy.resolve_names() {
            // SAFETY: live handle and an algorithm returned by hipBLASLt.
            let _ = unsafe { ffi::gufo_blaslt_resolve_names(self.handle, plan.algo.as_ptr()) };
        }
        plan
    }

    /// Launches the usable candidates of `order` with this call's operands
    /// (checked by `matmul`) until one succeeds, and returns its plan.
    #[allow(clippy::too_many_arguments)]
    fn trial<A: BlasElement, B: BlasElement, C: BlasElement>(
        &mut self,
        draft: Draft,
        order: &[usize],
        policy: &dyn AlgoPolicy,
        stream: &Stream,
        a: DeviceSlice<'_, A>,
        b: DeviceSlice<'_, B>,
        mut c: DeviceSliceMut<'_, C>,
    ) -> Option<Plan> {
        let Draft {
            plan,
            raw,
            ranked,
            capacity,
        } = draft;
        let workspace_ptr = self
            .workspace
            .as_mut()
            .map_or(std::ptr::null_mut(), |w| w.as_mut_slice().as_mut_ptr())
            .cast::<c_void>();
        let (alpha, beta) = (1.0f32, 0.0f32);
        let chosen = order.iter().copied().find(|&index| {
            let (Some(result), Some(heuristic)) = (raw.get(index), ranked.get(index)) else {
                return false;
            };
            if !heuristic.is_usable(capacity) {
                return false;
            }
            // SAFETY: as in `matmul`, with a usable algorithm hipBLASLt ranked
            // for these descriptors; its workspace fits the capacity.
            let status = unsafe {
                ffi::gufo_blaslt_matmul(
                    self.handle,
                    plan.desc,
                    &alpha,
                    a.as_ptr().cast(),
                    plan.layouts[0],
                    b.as_ptr().cast(),
                    plan.layouts[1],
                    &beta,
                    c.as_mut_ptr().cast(),
                    plan.layouts[2],
                    result.algo.as_ptr(),
                    workspace_ptr,
                    result.workspace_bytes,
                    stream.as_raw(),
                )
            };
            status == 0
        })?;
        Some(self.install(plan, &raw[chosen], policy))
    }

    /// Times each usable result with this call's operands (checked by
    /// `matmul`) and returns the plan of the fastest; see
    /// [`AlgoPolicy::timed_runs`]. The output is overwritten by every run.
    #[allow(clippy::too_many_arguments)]
    fn fastest<A: BlasElement, B: BlasElement, C: BlasElement>(
        &mut self,
        draft: Draft,
        runs: usize,
        policy: &dyn AlgoPolicy,
        stream: &Stream,
        a: DeviceSlice<'_, A>,
        b: DeviceSlice<'_, B>,
        mut c: DeviceSliceMut<'_, C>,
    ) -> Option<Plan> {
        let Draft {
            plan,
            raw,
            ranked,
            capacity,
        } = draft;
        let workspace_ptr = self
            .workspace
            .as_mut()
            .map_or(std::ptr::null_mut(), |w| w.as_mut_slice().as_mut_ptr())
            .cast::<c_void>();
        let (a_ptr, b_ptr, c_ptr) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
        let (alpha, beta) = (1.0f32, 0.0f32);
        let launch = |result: &ffi::Heuristic| {
            // SAFETY: as in `matmul`, with a usable algorithm hipBLASLt ranked
            // for these descriptors; its workspace fits the capacity.
            let status = unsafe {
                ffi::gufo_blaslt_matmul(
                    self.handle,
                    plan.desc,
                    &alpha,
                    a_ptr.cast(),
                    plan.layouts[0],
                    b_ptr.cast(),
                    plan.layouts[1],
                    &beta,
                    c_ptr.cast(),
                    plan.layouts[2],
                    result.algo.as_ptr(),
                    workspace_ptr,
                    result.workspace_bytes,
                    stream.as_raw(),
                )
            };
            status == 0
        };
        let time = |result: &ffi::Heuristic| -> Option<f32> {
            let (Ok(begin), Ok(end)) = (Event::timed(), Event::timed()) else {
                return None;
            };
            // The first launch loads the kernel and is not timed.
            if !launch(result) || stream.record(&begin).is_err() {
                return None;
            }
            if !(0..runs).all(|_| launch(result)) || stream.record(&end).is_err() {
                return None;
            }
            end.synchronize().ok()?;
            Event::elapsed_ms(&begin, &end).ok()
        };
        let mut best: Option<(usize, f32)> = None;
        for (index, (heuristic, result)) in ranked.iter().zip(&raw).enumerate() {
            if !heuristic.is_usable(capacity) {
                continue;
            }
            match time(result) {
                Some(ms) => {
                    if best.is_none_or(|(_, fastest)| ms < fastest) {
                        best = Some((index, ms));
                    }
                }
                None => {
                    // A failed launch or event leaves an error that a later
                    // check would report as its own.
                    // SAFETY: reads and clears this thread's last HIP error.
                    let _ = unsafe { sys::hipGetLastError() };
                }
            }
        }
        let (chosen, _) = best?;
        Some(self.install(plan, &raw[chosen], policy))
    }
}

impl Drop for BlasLt {
    fn drop(&mut self) {
        self.plans.clear();
        // SAFETY: created by gufo_blaslt_create, destroyed once after its plans.
        unsafe {
            let _ = ffi::gufo_blaslt_destroy(self.handle);
        }
    }
}
