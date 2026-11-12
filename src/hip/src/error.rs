//! HIP and launch errors.

use std::ffi::CStr;
use std::fmt;

use crate::sys;

/// What kind of failure a [`HipError`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// A HIP runtime call returned `code`.
    Runtime,
    /// Refused because an earlier sticky error poisoned the stream; `code` and
    /// `operation` name that error.
    Poisoned,
    /// Not allowed while the stream is capturing or validating a graph.
    CaptureViolation,
    /// Refused by gufo before calling HIP (wrong stream, length mismatch,
    /// runtime mismatch, ...).
    Refused,
    /// A hipBLAS or hipBLASLt call returned the `hipblasStatus_t` in `code`
    /// (-1: the shim caught a C++ exception).
    Blas,
}

/// Failure of a HIP runtime call, or an operation gufo refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipError {
    pub code: i32,
    pub operation: &'static str,
    pub kind: ErrorKind,
}

impl HipError {
    pub(crate) fn new(code: i32, operation: &'static str) -> Self {
        Self {
            code,
            operation,
            kind: ErrorKind::Runtime,
        }
    }

    pub(crate) fn refused(operation: &'static str) -> Self {
        Self {
            code: sys::HIP_SUCCESS,
            operation,
            kind: ErrorKind::Refused,
        }
    }

    pub(crate) fn blas(code: i32, operation: &'static str) -> Self {
        Self {
            code,
            operation,
            kind: ErrorKind::Blas,
        }
    }

    pub(crate) fn capture_violation(operation: &'static str) -> Self {
        Self {
            code: sys::HIP_SUCCESS,
            operation,
            kind: ErrorKind::CaptureViolation,
        }
    }

    /// Device faults that leave the HIP context unusable. Everything else
    /// (invalid arguments, launch configuration, out of memory) is local to
    /// the failed call.
    pub fn is_sticky(&self) -> bool {
        self.kind == ErrorKind::Runtime
            && matches!(
                self.code,
                214 /* ECC not correctable */
            | 700 /* illegal address */
            | 702 /* launch timeout */
            | 710 /* device assert */
            | 719 /* launch failure */
            | 999 /* unknown */
            )
    }
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ErrorKind::Runtime => {}
            ErrorKind::Poisoned => f.write_str("stream poisoned by an earlier failure: ")?,
            ErrorKind::CaptureViolation => {
                return write!(
                    f,
                    "{} is not allowed while a graph is recorded",
                    self.operation
                );
            }
            ErrorKind::Refused => return f.write_str(self.operation),
            ErrorKind::Blas => {
                return write!(f, "{}: hipBLAS status {}", self.operation, self.code);
            }
        }
        write!(
            f,
            "{}: {} ({})",
            self.operation,
            error_string(self.code),
            self.code
        )
    }
}

impl std::error::Error for HipError {}

fn error_string(code: i32) -> String {
    // SAFETY: hipGetErrorString returns a static NUL-terminated string for
    // every code, including unknown ones.
    unsafe { CStr::from_ptr(sys::hipGetErrorString(code)) }
        .to_string_lossy()
        .into_owned()
}

pub(crate) fn check(code: sys::HipError, operation: &'static str) -> Result<(), HipError> {
    #[cfg(test)]
    if let Some(code) = crate::fault::take(operation) {
        return Err(HipError::new(code, operation));
    }
    if code == sys::HIP_SUCCESS {
        Ok(())
    } else {
        Err(HipError::new(code, operation))
    }
}

/// Why a generated launch wrapper refused or failed a launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchError {
    /// A buffer is shorter than the extent the kernel contract declares.
    ExtentTooShort {
        param: &'static str,
        required: usize,
        actual: usize,
    },
    /// The declared extent does not fit in `usize`.
    ExtentOverflow { param: &'static str },
    /// A buffer belongs to another stream than the launch.
    ForeignStream { param: &'static str },
    /// An argument violates the operation's contract (for example a leading
    /// dimension below the row count).
    InvalidArgument {
        param: &'static str,
        reason: &'static str,
    },
    /// The scalars violate a `requires` relation of the kernel contract.
    Precondition {
        launcher: &'static str,
        requirement: &'static str,
    },
    /// An index buffer may hold values beyond what the kernel can index.
    IndexBound {
        param: &'static str,
        bound: u32,
        limit: usize,
    },
    /// An index buffer's mask or padding differs from what the kernel
    /// handles (see `IndexSpec`).
    IndexSpec {
        param: &'static str,
        reason: &'static str,
    },
    /// Kernel-written indices were not derived from the source buffer the
    /// launch passes, in its current generation and geometry.
    Provenance {
        param: &'static str,
        source: &'static str,
    },
    /// The launch failed, or the stream was already poisoned.
    Hip(HipError),
    /// The C++ launcher threw; the shim converted it into a status.
    LauncherException { launcher: &'static str },
    /// Graph validation: launch number `launch` of the replayed body differs
    /// from the capture in `argument` (a scalar frozen into the graph, a
    /// replaced buffer, or a different launch sequence for the same key).
    GraphMismatch {
        launch: usize,
        launcher: &'static str,
        argument: &'static str,
    },
}

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExtentTooShort {
                param,
                required,
                actual,
            } => {
                write!(
                    f,
                    "`{param}` holds {actual} elements, the kernel needs {required}"
                )
            }
            Self::ExtentOverflow { param } => write!(f, "extent of `{param}` overflows"),
            Self::ForeignStream { param } => write!(f, "`{param}` belongs to another stream"),
            Self::InvalidArgument { param, reason } => write!(f, "`{param}`: {reason}"),
            Self::Precondition {
                launcher,
                requirement,
            } => write!(f, "{launcher} requires `{requirement}`"),
            Self::IndexBound {
                param,
                bound,
                limit,
            } => write!(
                f,
                "`{param}` holds indices below {bound}, the kernel indexes {limit} rows"
            ),
            Self::IndexSpec { param, reason } => write!(f, "`{param}`: {reason}"),
            Self::Provenance { param, source } => write!(
                f,
                "`{param}` was not computed from the current `{source}` of this geometry"
            ),
            Self::Hip(error) => error.fmt(f),
            Self::LauncherException { launcher } => write!(f, "{launcher} threw"),
            Self::GraphMismatch {
                launch,
                launcher,
                argument,
            } => write!(
                f,
                "graph replay differs from its capture at launch {launch} ({launcher}): `{argument}`"
            ),
        }
    }
}

impl std::error::Error for LaunchError {}

impl From<HipError> for LaunchError {
    fn from(error: HipError) -> Self {
        Self::Hip(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_faults_are_sticky_and_argument_errors_are_not() {
        assert!(HipError::new(700, "kernel").is_sticky());
        assert!(HipError::new(719, "kernel").is_sticky());
        assert!(!HipError::new(1, "hipMemcpyAsync").is_sticky());
        assert!(!HipError::new(2, "hipMallocAsync").is_sticky());
        assert!(!HipError::new(9, "launch").is_sticky());
    }
}
