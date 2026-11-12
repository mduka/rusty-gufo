//! Whether this process's HIP context still runs work. A GPU reset (for
//! example `amdgpu` recovering from a MES hang) invalidates the context for
//! good: every later work unit fails, and only a new process recovers.

use std::ffi::c_void;
use std::sync::Mutex;

use crate::error::{HipError, check};
use crate::stream::require_build_runtime;
use crate::sys;

const PROBE_BYTES: usize = size_of::<u32>();

/// One [`DeviceProbe::poll`] result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceProbeStatus {
    Usable,
    /// The probe may be queued behind long kernels on the shared hardware
    /// queues; poll again.
    Pending,
    /// A hard HIP error: the context is lost.
    Lost,
}

/// A four-byte device buffer and a private stream, allocated with the model
/// so probing a lost context needs no allocation. Context loss is sticky:
/// after a reset even this memset fails with a hard error.
pub struct DeviceProbe {
    probe: Mutex<Probe>,
}

struct Probe {
    stream: sys::HipStream,
    buffer: *mut c_void,
    /// A memset was submitted and has not completed yet.
    pending: bool,
}

// SAFETY: the stream and buffer are process-global HIP handles, used only
// under the mutex.
unsafe impl Send for Probe {}

impl DeviceProbe {
    pub fn new() -> Result<Self, HipError> {
        require_build_runtime()?;
        let mut stream = std::ptr::null_mut();
        // SAFETY: `stream` is a valid out-pointer.
        let status =
            unsafe { sys::hipStreamCreateWithFlags(&mut stream, sys::HIP_STREAM_NON_BLOCKING) };
        check(status, "hipStreamCreateWithFlags")?;
        let mut buffer = std::ptr::null_mut();
        // SAFETY: valid out-pointer and live stream.
        let status = unsafe { sys::hipMallocAsync(&mut buffer, PROBE_BYTES, stream) };
        if let Err(error) = check(status, "hipMallocAsync") {
            // SAFETY: the stream was created above and holds no work.
            unsafe { sys::hipStreamDestroy(stream) };
            return Err(error);
        }
        Ok(Self {
            probe: Mutex::new(Probe {
                stream,
                buffer,
                pending: false,
            }),
        })
    }

    /// Submits a memset unless one is outstanding, then reports its state
    /// without waiting.
    pub fn poll(&self) -> DeviceProbeStatus {
        let mut probe = self.probe.lock().unwrap_or_else(|e| e.into_inner());
        if !probe.pending {
            // SAFETY: plain runtime calls on live handles; the buffer holds
            // `PROBE_BYTES` bytes of this stream. A failed work unit leaves
            // its error on the thread, cleared first so it is not read as the
            // probe's.
            let submitted = unsafe {
                sys::hipGetLastError();
                sys::hipMemsetAsync(probe.buffer, 0, PROBE_BYTES, probe.stream) == sys::HIP_SUCCESS
                    && sys::hipGetLastError() == sys::HIP_SUCCESS
            };
            if !submitted {
                return DeviceProbeStatus::Lost;
            }
            probe.pending = true;
        }
        // SAFETY: live stream.
        match unsafe { sys::hipStreamQuery(probe.stream) } {
            sys::HIP_ERROR_NOT_READY => DeviceProbeStatus::Pending,
            status => {
                probe.pending = false;
                if status == sys::HIP_SUCCESS {
                    DeviceProbeStatus::Usable
                } else {
                    DeviceProbeStatus::Lost
                }
            }
        }
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        // SAFETY: the buffer belongs to this stream and is freed in its
        // order; neither handle is used afterwards. Teardown errors are moot.
        unsafe {
            sys::hipFreeAsync(self.buffer, self.stream);
            sys::hipStreamDestroy(self.stream);
        }
    }
}

/// Ends the process with `status` at once, without exit handlers or static
/// destructors: HIP teardown can block on a lost device, and the kernel
/// reclaims the process's resources.
pub fn exit_immediately(status: i32) -> ! {
    // SAFETY: `_exit` takes no pointers and does not return.
    unsafe { libc::_exit(status) }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    fn probe_reports_a_usable_device() {
        let probe = DeviceProbe::new().expect("probe");
        let status = loop {
            match probe.poll() {
                DeviceProbeStatus::Pending => std::thread::yield_now(),
                status => break status,
            }
        };
        assert_eq!(status, DeviceProbeStatus::Usable);
        // A completed probe submits a fresh memset on the next poll.
        assert_ne!(probe.poll(), DeviceProbeStatus::Lost);
    }
}
