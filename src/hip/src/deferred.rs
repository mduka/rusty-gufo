//! Frees that must not happen while a graph is being captured.
//!
//! Freeing shared device memory needs a device-wide synchronisation, and
//! `hipHostFree` performs one implicitly. Synchronising the device also
//! synchronises every capturing stream, which invalidates captures running on
//! other threads (`hipErrorStreamCaptureInvalidated`). A stream-ordered free on
//! a stream that is capturing would be recorded into the graph and repeated on
//! every replay.
//!
//! Such frees are queued instead. Every capture holds a read guard of `GATE`
//! for its whole duration; `collect` frees the queue only when it obtains the
//! write guard without waiting, i.e. while no stream in the process captures.
//! Captures call `collect` when they end.

use std::cell::Cell;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};

use crate::error::HipError;
use crate::host::HostRelease;
use crate::mapped::MappedFile;
use crate::stream::StreamInner;
use crate::sys;

static GATE: RwLock<()> = RwLock::new(());
static PENDING: Mutex<Vec<Deferred>> = Mutex::new(Vec::new());

pub(crate) enum Deferred {
    /// Device memory from `hipMallocAsync` on `home`; `shared` memory may have
    /// been read on any stream.
    Device {
        ptr: NonNull<c_void>,
        home: Arc<StreamInner>,
        shared: bool,
    },
    /// Exactly sized device memory from `hipMalloc`; `hipFree` synchronises
    /// the device implicitly.
    Exact { ptr: NonNull<c_void> },
    /// Page-locked host memory from `hipHostMalloc`.
    Host { ptr: NonNull<c_void> },
    /// A file mapping registered with `hipHostRegister`; unmapped when the
    /// last reference goes, after unregistering.
    Registered { mapping: Arc<MappedFile> },
    /// Host-visible memory: page-locked or a registered memory file.
    HostMemory(HostRelease),
}

// SAFETY: the entries are addresses owned by the queue; freeing them from any
// thread is fine.
unsafe impl Send for Deferred {}

thread_local! {
    /// Capture guards the current thread holds.
    static HELD: Cell<usize> = const { Cell::new(0) };
}

/// A capture's read guard of `GATE`.
pub(crate) struct CaptureGuard {
    _guard: RwLockReadGuard<'static, ()>,
}

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        HELD.with(|held| held.set(held.get() - 1));
    }
}

/// Held by a capture from before `hipStreamBeginCapture` until after
/// `hipStreamEndCapture`.
pub(crate) fn capture_guard() -> CaptureGuard {
    let guard = GATE.read().unwrap_or_else(|poisoned| poisoned.into_inner());
    HELD.with(|held| held.set(held.get() + 1));
    CaptureGuard { _guard: guard }
}

/// Blocks until no capture runs, frees the whole queue and keeps new
/// captures from starting while the guard lives, for rare operations that may
/// synchronise the device (registering weight files, exact allocations).
/// Unlike `collect`, the queue is empty afterwards, so a file's previous
/// registration is undone before it is registered again. Refused on a thread
/// that is itself capturing (a capture body), which would wait for itself.
pub(crate) fn exclusive(
    operation: &'static str,
) -> Result<RwLockWriteGuard<'static, ()>, HipError> {
    if HELD.with(Cell::get) > 0 {
        return Err(HipError::capture_violation(operation));
    }
    let guard = GATE
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    free_all(&mut lock_pending());
    Ok(guard)
}

pub(crate) fn defer(entry: Deferred) {
    lock_pending().push(entry);
    collect();
}

fn lock_pending() -> std::sync::MutexGuard<'static, Vec<Deferred>> {
    PENDING
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Frees what is safe to free now. Never blocks on a running capture.
pub fn collect() {
    let mut pending = lock_pending();
    if pending.is_empty() {
        return;
    }
    // Stream-ordered frees only need their own stream to stop capturing.
    pending.retain(|entry| match entry {
        Deferred::Device {
            ptr,
            home,
            shared: false,
        } if !home.is_capturing() => {
            // SAFETY: allocated by hipMallocAsync on `home`, used only on
            // `home`, which no longer captures; the free follows every use.
            unsafe {
                let _ = sys::hipFreeAsync(ptr.as_ptr(), home.raw);
            }
            false
        }
        _ => true,
    });
    if pending.is_empty() {
        return;
    }
    let _exclusive = match GATE.try_write() {
        Ok(guard) => guard,
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => return,
    };
    free_all(&mut pending);
}

/// Frees every entry; the caller holds the write guard of `GATE`, so no
/// stream captures and none can start: synchronising the device is safe.
fn free_all(pending: &mut Vec<Deferred>) {
    if pending.is_empty() {
        return;
    }
    // SAFETY: plain device synchronisation.
    unsafe {
        let _ = sys::hipDeviceSynchronize();
    }
    for entry in pending.drain(..) {
        match entry {
            Deferred::Device { ptr, home, .. } => {
                // SAFETY: every stream finished its work above, so no kernel
                // still uses the memory; the entry is freed once, here.
                unsafe {
                    let _ = sys::hipFreeAsync(ptr.as_ptr(), home.raw);
                }
            }
            Deferred::Exact { ptr } => {
                // SAFETY: as above; allocated by hipMalloc and freed once.
                unsafe {
                    let _ = sys::hipFree(ptr.as_ptr());
                }
            }
            Deferred::Host { ptr } => {
                // SAFETY: as above; no copy still uses the pinned memory.
                unsafe {
                    let _ = sys::hipHostFree(ptr.as_ptr());
                }
            }
            Deferred::Registered { mapping } => {
                // SAFETY: as above; the mapping was registered once and no
                // kernel still reads it. Dropping the entry may unmap it.
                unsafe {
                    let _ = sys::hipHostUnregister(mapping.host_address());
                }
            }
            // As above: no device work uses the memory any more.
            Deferred::HostMemory(release) => release.release(),
        }
    }
}
