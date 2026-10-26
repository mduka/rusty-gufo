//! Streams and events.

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Once, OnceLock};

use crate::error::{ErrorKind, HipError, LaunchError, check};
use crate::memory::{
    DeviceBuffer, DeviceCopy, DeviceIndexPair, DeviceIndices, DeviceSlice, DeviceSliceMut,
    IndexSpec, PinnedBuffer,
};
use crate::record::{Arg, Recorder};
use crate::sys;

static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(1);

/// `HIP_VERSION` of the HIP runtime loaded into this process.
pub fn runtime_version() -> Result<i32, HipError> {
    let mut version = 0;
    // SAFETY: valid out-pointer.
    let status = unsafe { sys::hipRuntimeGetVersion(&mut version) };
    check(status, "hipRuntimeGetVersion")?;
    Ok(version)
}

/// Free and total bytes of the current device (`hipMemGetInfo`).
pub fn device_memory() -> Result<(usize, usize), HipError> {
    let (mut free, mut total) = (0, 0);
    // SAFETY: valid out-pointers.
    let status = unsafe { sys::hipMemGetInfo(&mut free, &mut total) };
    check(status, "hipMemGetInfo")?;
    Ok((free.min(REPORTED_FREE_CAP.load(Ordering::Relaxed)), total))
}

static REPORTED_FREE_CAP: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Caps the free memory [`device_memory`] reports, process-wide; `usize::MAX`
/// lifts the cap. For tests of what callers do without device headroom.
#[doc(hidden)]
pub fn cap_reported_device_free(bytes: usize) {
    REPORTED_FREE_CAP.store(bytes, Ordering::Relaxed);
}

/// Refuses to run on another HIP runtime than the one the kernels were built
/// against; distribution and AMD packages share the same library soname.
pub(crate) fn require_build_runtime() -> Result<(), HipError> {
    static CHECKED: OnceLock<Result<(), HipError>> = OnceLock::new();
    *CHECKED.get_or_init(|| {
        let loaded = runtime_version()?;
        if loaded == crate::BUILD_HIP_VERSION {
            return Ok(());
        }
        let message = format!(
            "HIP runtime {loaded} is loaded, but gufo was built against {} from {}",
            crate::BUILD_HIP_VERSION,
            crate::BUILD_ROCM_ROOT
        );
        Err(HipError::refused(Box::leak(message.into_boxed_str())))
    })
}

/// Free memory the stream-ordered pool keeps mapped across synchronisations;
/// more goes back to the system. With HIP's threshold of 0 the pool returns
/// every free block at each synchronisation, and mapping a 128 MiB block
/// again costs 5 to 8 ms: Flash-Next prompt snapshots paid that on every
/// request. 2 GiB is the reserve C++ grows its cache backing by before a
/// prefill.
const POOL_RETAINED_BYTES: u64 = 2 << 30;

/// Sets the default pool's release threshold to [`POOL_RETAINED_BYTES`],
/// once per process, after the pool's first allocation: HIP 7.15 ignores a
/// threshold set before it, and the pool keeps returning every free block.
/// The threshold bounds the free memory the pool keeps, beside any amount in
/// use. It only steers how much stays mapped, so a failure is not an error.
pub(crate) fn retain_pool_memory() {
    static DONE: Once = Once::new();
    DONE.call_once(|| {
        let mut pool = std::ptr::null_mut();
        // SAFETY: valid out-pointer; Gufo uses device 0 only.
        if unsafe { sys::hipDeviceGetDefaultMemPool(&mut pool, 0) } != sys::HIP_SUCCESS {
            return;
        }
        let mut threshold = POOL_RETAINED_BYTES;
        // SAFETY: a live pool; the release threshold takes a u64.
        unsafe {
            let _ = sys::hipMemPoolSetAttribute(
                pool,
                sys::HIP_MEM_POOL_ATTR_RELEASE_THRESHOLD,
                (&raw mut threshold).cast(),
            );
        }
    });
}

pub(crate) struct StreamInner {
    pub(crate) raw: sys::HipStream,
    pub(crate) id: u64,
    capturing: AtomicBool,
}

impl StreamInner {
    /// Whether a graph capture is recording this stream.
    pub(crate) fn is_capturing(&self) -> bool {
        self.capturing.load(Ordering::Acquire)
    }
}

// SAFETY: HIP stream handles may be used from any thread. Launch order is
// protected by `Stream` being !Sync, not by this handle.
unsafe impl Send for StreamInner {}
// SAFETY: shared access only enqueues frees (thread-safe in HIP) and reads `id`.
unsafe impl Sync for StreamInner {}

impl Drop for StreamInner {
    fn drop(&mut self) {
        // Runs after the last buffer of this stream enqueued its free.
        // SAFETY: `raw` was created by hipStreamCreateWithFlags and is
        // destroyed exactly once, here.
        unsafe {
            let _ = sys::hipStreamSynchronize(self.raw);
            let _ = sys::hipStreamDestroy(self.raw);
        }
    }
}

/// An in-order HIP stream.
///
/// `Send` but not `Sync`: one thread enqueues at a time, so launch order is
/// that thread's program order. After a sticky device fault the stream is
/// poisoned and refuses further work; the engine owning it must be rebuilt.
pub struct Stream {
    inner: Arc<StreamInner>,
    poison: Cell<Option<HipError>>,
    recorder: RefCell<Option<Recorder>>,
    _not_sync: PhantomData<Cell<()>>,
}

impl Stream {
    pub fn new() -> Result<Self, HipError> {
        require_build_runtime()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer.
        let status =
            unsafe { sys::hipStreamCreateWithFlags(&mut raw, sys::HIP_STREAM_NON_BLOCKING) };
        check(status, "hipStreamCreateWithFlags")?;
        let id = NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            inner: Arc::new(StreamInner {
                raw,
                id,
                capturing: AtomicBool::new(false),
            }),
            poison: Cell::new(None),
            recorder: RefCell::new(None),
            _not_sync: PhantomData,
        })
    }

    pub fn id(&self) -> u64 {
        self.inner.id
    }

    pub(crate) fn inner(&self) -> &Arc<StreamInner> {
        &self.inner
    }

    /// Raw handle for generated launch wrappers.
    pub fn as_raw(&self) -> sys::HipStream {
        self.inner.raw
    }

    /// The sticky error that poisoned this stream, if any.
    pub fn poisoned(&self) -> Option<HipError> {
        self.poison.get()
    }

    /// Fails fast when an earlier sticky error poisoned the stream.
    pub fn ensure_healthy(&self) -> Result<(), HipError> {
        match self.poison.get() {
            None => Ok(()),
            Some(cause) => Err(HipError {
                kind: ErrorKind::Poisoned,
                ..cause
            }),
        }
    }

    /// Records the outcome of an operation on this stream, poisoning it on a
    /// sticky error.
    pub fn observe<T>(&self, result: Result<T, HipError>) -> Result<T, HipError> {
        if let Err(error) = &result
            && error.is_sticky()
            && self.poison.get().is_none()
        {
            self.poison.set(Some(*error));
        }
        result
    }

    /// Whether a graph capture or validation is recording launches. Generated
    /// wrappers report their arguments to the recorder while this is true.
    pub fn is_recording(&self) -> bool {
        self.recorder.borrow().is_some()
    }

    pub(crate) fn set_capturing(&self, capturing: bool) {
        self.inner.capturing.store(capturing, Ordering::Release);
    }

    pub(crate) fn begin_recording(&self, recorder: Recorder) {
        *self.recorder.borrow_mut() = Some(recorder);
    }

    pub(crate) fn take_recorder(&self) -> Option<Recorder> {
        self.recorder.borrow_mut().take()
    }

    pub(crate) fn with_recorder<R>(&self, f: impl FnOnce(&mut Recorder) -> R) -> Option<R> {
        self.recorder.borrow_mut().as_mut().map(f)
    }

    /// Host-side operations that cannot be part of a graph.
    fn deny_while_recording(&self, operation: &'static str) -> Result<(), HipError> {
        if self.is_recording() {
            return Err(HipError::capture_violation(operation));
        }
        Ok(())
    }

    pub fn synchronize(&self) -> Result<(), HipError> {
        self.ensure_healthy()?;
        self.deny_while_recording("hipStreamSynchronize")?;
        // SAFETY: the handle is live for the lifetime of `self`.
        let status = unsafe { sys::hipStreamSynchronize(self.inner.raw) };
        self.observe(check(status, "hipStreamSynchronize"))
    }

    /// Stream-ordered allocation of `len` elements. Contents are unspecified
    /// but valid, because every `DeviceCopy` type accepts any bit pattern.
    pub fn alloc<T: DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<T>, HipError> {
        self.ensure_healthy()?;
        self.deny_while_recording("hipMallocAsync")?;
        self.observe(DeviceBuffer::allocate(&self.inner, len, false))
    }

    /// Exactly sized allocation (`hipMalloc`) for memory that lives as long
    /// as a model: weights, caches, scratch. The stream-ordered pool behind
    /// [`Stream::alloc`] reserves device memory in 128 MiB blocks, so its
    /// footprint rounds up by as much; this one does not, but dropping it
    /// waits for the whole device (through the free queue, like an immutable
    /// tensor), and allocating waits until no graph is being captured.
    /// Contents are unspecified but valid.
    pub fn alloc_persistent<T: DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<T>, HipError> {
        self.ensure_healthy()?;
        self.deny_while_recording("hipMalloc")?;
        self.observe(DeviceBuffer::allocate(&self.inner, len, true))
    }

    /// Managed memory of this stream (`hipMallocManaged`, attached
    /// globally), which the host addresses too ([`DeviceBuffer::host`]).
    /// Dropping it waits for the whole device, like an exact allocation.
    pub fn alloc_managed<T: DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<T>, HipError> {
        self.ensure_healthy()?;
        self.deny_while_recording("hipMallocManaged")?;
        self.observe(DeviceBuffer::allocate_managed(&self.inner, len))
    }

    /// Coarse-grain managed memory: ordinary system memory the device caches
    /// as it caches `hipMalloc` memory, outside the device heap (upstream
    /// `AllocateManaged`). For contents filled by copies or kernels, which
    /// need no coherence with the host during a kernel.
    pub fn alloc_managed_coarse<T: DeviceCopy>(
        &self,
        len: usize,
    ) -> Result<DeviceBuffer<T>, HipError> {
        let buffer = self.alloc_managed(len)?;
        // The advice is a caching hint; fine-grain memory stays correct.
        let _ = buffer.advise_coarse_grain();
        Ok(buffer)
    }

    /// Copies from the host address of managed or host memory (`source`, a
    /// buffer of this stream) into `destination`, in stream order and
    /// without waiting (`hipMemcpyAsync`, host to device). Host views of the
    /// source wait for the copy, because they wait for this stream.
    pub fn copy_from_host_view<T: DeviceCopy>(
        &self,
        source: DeviceSlice<'_, T>,
        mut destination: DeviceSliceMut<'_, T>,
    ) -> Result<(), HipError> {
        self.check_copy(source.len(), destination.len(), source.home_stream())?;
        if source.home_stream().is_none() || destination.home_stream() != self.id() {
            return Err(HipError::refused(
                "host-view copy with a buffer of another stream",
            ));
        }
        let host = source
            .allocation()
            .host_address(source.as_ptr().cast())
            .ok_or(HipError::refused("host-view copy from device-only memory"))?;
        // SAFETY: both ranges hold `len` elements of live allocations of this
        // stream: `host` is the source's host address, which the host
        // memory keeps until the free queue has synchronised the device.
        let status = unsafe {
            sys::hipMemcpyAsync(
                destination.as_mut_ptr().cast(),
                host.cast_const().cast(),
                source.len() * size_of::<T>(),
                sys::HIP_MEMCPY_HOST_TO_DEVICE,
                self.inner.raw,
            )
        };
        self.observe(check(status, "hipMemcpyAsync H2D"))
    }

    /// Copies `source` into the host address of managed or host memory
    /// (`destination`, a buffer of this stream), in stream order and without
    /// waiting (`hipMemcpyAsync`, device to host).
    pub fn copy_to_host_view<T: DeviceCopy>(
        &self,
        source: DeviceSlice<'_, T>,
        mut destination: DeviceSliceMut<'_, T>,
    ) -> Result<(), HipError> {
        self.check_copy(source.len(), destination.len(), source.home_stream())?;
        if destination.home_stream() != self.id() {
            return Err(HipError::refused(
                "host-view copy into a buffer of another stream",
            ));
        }
        let host = destination
            .allocation()
            .host_address(destination.as_mut_ptr().cast_const().cast())
            .ok_or(HipError::refused("host-view copy into device-only memory"))?;
        // SAFETY: as in `copy_from_host_view`; the destination is mutably
        // borrowed, and its host views wait for this stream.
        let status = unsafe {
            sys::hipMemcpyAsync(
                host.cast(),
                source.as_ptr().cast(),
                source.len() * size_of::<T>(),
                sys::HIP_MEMCPY_DEVICE_TO_HOST,
                self.inner.raw,
            )
        };
        self.observe(check(status, "hipMemcpyAsync D2H"))
    }

    /// Records `event` after all work enqueued so far.
    pub fn record(&self, event: &Event) -> Result<(), HipError> {
        self.ensure_healthy()?;
        self.deny_while_recording("hipEventRecord")?;
        // SAFETY: both handles are live.
        let status = unsafe { sys::hipEventRecord(event.raw, self.inner.raw) };
        self.observe(check(status, "hipEventRecord"))
    }

    /// Copies pageable host memory to the device and waits, so `host` may be
    /// reused as soon as this returns.
    pub fn upload<T: DeviceCopy>(
        &self,
        host: &[T],
        mut device: DeviceSliceMut<'_, T>,
    ) -> Result<(), HipError> {
        self.check_copy(host.len(), device.len(), Some(device.home_stream()))?;
        // SAFETY: both ranges hold `host.len()` elements, and the stream is
        // synchronised before `host` can be released.
        let status = unsafe {
            sys::hipMemcpyAsync(
                device.as_mut_ptr().cast(),
                host.as_ptr().cast(),
                size_of_val(host),
                sys::HIP_MEMCPY_HOST_TO_DEVICE,
                self.inner.raw,
            )
        };
        self.observe(check(status, "hipMemcpyAsync H2D"))?;
        self.synchronize()
    }

    /// [`Stream::upload`] on HIP's blocking copy path
    /// (`hipMemcpyWithStream`): complete on return without a separate
    /// stream synchronisation, which is faster for the few bytes a step
    /// writes before its launches.
    pub fn upload_blocking<T: DeviceCopy>(
        &self,
        host: &[T],
        mut device: DeviceSliceMut<'_, T>,
    ) -> Result<(), HipError> {
        self.check_copy(host.len(), device.len(), Some(device.home_stream()))?;
        // SAFETY: both ranges hold `host.len()` elements; the copy has
        // completed when the call returns, before `host` can be released.
        let status = unsafe {
            sys::hipMemcpyWithStream(
                device.as_mut_ptr().cast(),
                host.as_ptr().cast(),
                size_of_val(host),
                sys::HIP_MEMCPY_HOST_TO_DEVICE,
                self.inner.raw,
            )
        };
        self.observe(check(status, "hipMemcpyWithStream H2D"))
    }

    /// [`Stream::download`] on HIP's blocking copy path
    /// (`hipMemcpyWithStream`), for the few bytes a host decision waits
    /// on: an asynchronous copy plus a stream synchronisation is slower
    /// there.
    pub fn download_blocking<T: DeviceCopy>(
        &self,
        device: DeviceSlice<'_, T>,
        host: &mut [T],
    ) -> Result<(), HipError> {
        self.check_copy(host.len(), device.len(), device.home_stream())?;
        // SAFETY: as in `upload_blocking`; `host` stays mutably borrowed
        // until the copy has completed.
        let status = unsafe {
            sys::hipMemcpyWithStream(
                host.as_mut_ptr().cast(),
                device.as_ptr().cast(),
                size_of_val(host),
                sys::HIP_MEMCPY_DEVICE_TO_HOST,
                self.inner.raw,
            )
        };
        self.observe(check(status, "hipMemcpyWithStream D2H"))
    }

    /// Copies device memory into pageable host memory and waits.
    pub fn download<T: DeviceCopy>(
        &self,
        device: DeviceSlice<'_, T>,
        host: &mut [T],
    ) -> Result<(), HipError> {
        self.check_copy(host.len(), device.len(), device.home_stream())?;
        // SAFETY: as in `upload`; `host` stays mutably borrowed until the
        // synchronisation below returns.
        let status = unsafe {
            sys::hipMemcpyAsync(
                host.as_mut_ptr().cast(),
                device.as_ptr().cast(),
                size_of_val(host),
                sys::HIP_MEMCPY_DEVICE_TO_HOST,
                self.inner.raw,
            )
        };
        self.observe(check(status, "hipMemcpyAsync D2H"))?;
        self.synchronize()
    }

    /// Sets every byte of `device` to `byte`, in stream order. For test
    /// patterns and zero-initialisation.
    pub fn fill_bytes<T: DeviceCopy>(
        &self,
        mut device: DeviceSliceMut<'_, T>,
        byte: u8,
    ) -> Result<(), HipError> {
        self.ensure_healthy()?;
        self.deny_while_recording("hipMemsetAsync")?;
        if device.home_stream() != self.id() {
            return Err(HipError::refused(
                "fill of a buffer that belongs to another stream",
            ));
        }
        // SAFETY: the view covers `len` elements of a live allocation of this
        // stream; any byte pattern is a valid `DeviceCopy` value.
        let status = unsafe {
            sys::hipMemsetAsync(
                device.as_mut_ptr().cast(),
                i32::from(byte),
                device.len() * size_of::<T>(),
                self.inner.raw,
            )
        };
        self.observe(check(status, "hipMemsetAsync"))
    }

    /// Copies the raw bytes of `device` into `host` and waits; `host` must
    /// hold exactly the view's size in bytes.
    pub fn download_bytes<T: DeviceCopy>(
        &self,
        device: DeviceSlice<'_, T>,
        host: &mut [u8],
    ) -> Result<(), HipError> {
        self.check_copy(
            host.len(),
            device.len() * size_of::<T>(),
            device.home_stream(),
        )?;
        // SAFETY: both ranges hold `host.len()` bytes, and `host` stays
        // mutably borrowed until the synchronisation below returns.
        let status = unsafe {
            sys::hipMemcpyAsync(
                host.as_mut_ptr().cast(),
                device.as_ptr().cast(),
                host.len(),
                sys::HIP_MEMCPY_DEVICE_TO_HOST,
                self.inner.raw,
            )
        };
        self.observe(check(status, "hipMemcpyAsync D2H"))?;
        self.synchronize()
    }

    /// Copies raw bytes (a serialised payload, of any alignment) into
    /// `device` and waits; `host` must hold exactly the view's size in bytes.
    /// Every [`DeviceCopy`] type is valid for any bit pattern.
    pub fn upload_bytes<T: DeviceCopy>(
        &self,
        host: &[u8],
        mut device: DeviceSliceMut<'_, T>,
    ) -> Result<(), HipError> {
        self.check_copy(
            host.len(),
            device.len() * size_of::<T>(),
            Some(device.home_stream()),
        )?;
        if host.is_empty() {
            return Ok(());
        }
        // SAFETY: both ranges hold `host.len()` bytes, and the stream is
        // synchronised before `host` can be released.
        let status = unsafe {
            sys::hipMemcpyAsync(
                device.as_mut_ptr().cast(),
                host.as_ptr().cast(),
                host.len(),
                sys::HIP_MEMCPY_HOST_TO_DEVICE,
                self.inner.raw,
            )
        };
        self.observe(check(status, "hipMemcpyAsync H2D"))?;
        self.synchronize()
    }

    /// A device-to-device copy that may also be part of a graph body: while
    /// a graph is captured it is recorded like a launch (its buffers are
    /// retained) and captured as a memcpy node; while one is validated it is
    /// only compared. Otherwise it is [`Stream::copy`].
    pub fn graph_copy<T: DeviceCopy>(
        &self,
        source: DeviceSlice<'_, T>,
        mut destination: DeviceSliceMut<'_, T>,
    ) -> Result<(), LaunchError> {
        if !self.is_recording() {
            return Ok(self.copy(source, destination)?);
        }
        self.ensure_healthy()?;
        if source.len() != destination.len() {
            return Err(HipError::refused("device copy length mismatch").into());
        }
        if source.home_stream().is_some_and(|home| home != self.id())
            || destination.home_stream() != self.id()
        {
            return Err(LaunchError::ForeignStream {
                param: "graph_copy",
            });
        }
        let bytes = source.len() * size_of::<T>();
        let recorded = crate::launch::record(
            self,
            "hipMemcpyAsync D2D",
            &[
                Arg::buffer("source", Some(source.allocation()), source.address(), bytes),
                Arg::buffer(
                    "destination",
                    Some(destination.allocation()),
                    destination.address(),
                    bytes,
                ),
            ],
        )?;
        if recorded.is_some() {
            // Validation: compared, not executed.
            return Ok(());
        }
        // SAFETY: both views hold `bytes` of live allocations of this stream's
        // order; the capture retains them for every replay.
        let status = unsafe {
            sys::hipMemcpyAsync(
                destination.as_mut_ptr().cast(),
                source.as_ptr().cast(),
                bytes,
                sys::HIP_MEMCPY_DEVICE_TO_DEVICE,
                self.inner.raw,
            )
        };
        Ok(self.observe(check(status, "hipMemcpyAsync D2D"))?)
    }

    /// Copies `rows` rows of `width` elements, `source_pitch` and
    /// `target_pitch` elements apart, in stream order (`hipMemcpy2DAsync`).
    pub fn copy_2d<T: DeviceCopy>(
        &self,
        source: DeviceSlice<'_, T>,
        source_pitch: usize,
        mut target: DeviceSliceMut<'_, T>,
        target_pitch: usize,
        width: usize,
        rows: usize,
    ) -> Result<(), HipError> {
        self.ensure_healthy()?;
        self.deny_while_recording("hipMemcpy2DAsync")?;
        check_2d(source.len(), source_pitch, width, rows)?;
        check_2d(target.len(), target_pitch, width, rows)?;
        if source.home_stream().is_some_and(|home| home != self.id())
            || target.home_stream() != self.id()
        {
            return Err(HipError::refused(
                "2D copy of a buffer that belongs to another stream",
            ));
        }
        if rows == 0 || width == 0 {
            return Ok(());
        }
        let bytes = size_of::<T>();
        // SAFETY: `check_2d` bounds every row of both views inside their
        // live allocations; the copy runs in this stream's order.
        let status = unsafe {
            sys::hipMemcpy2DAsync(
                target.as_mut_ptr().cast(),
                target_pitch * bytes,
                source.as_ptr().cast(),
                source_pitch * bytes,
                width * bytes,
                rows,
                sys::HIP_MEMCPY_DEVICE_TO_DEVICE,
                self.inner.raw,
            )
        };
        self.observe(check(status, "hipMemcpy2DAsync D2D"))
    }

    /// Copies `rows` rows of `width` elements from the device into pageable
    /// host memory, `source_pitch` and `host_pitch` elements apart, and waits.
    pub fn download_2d<T: DeviceCopy>(
        &self,
        source: DeviceSlice<'_, T>,
        source_pitch: usize,
        host: &mut [T],
        host_pitch: usize,
        width: usize,
        rows: usize,
    ) -> Result<(), HipError> {
        self.ensure_healthy()?;
        self.deny_while_recording("host/device copy")?;
        check_2d(source.len(), source_pitch, width, rows)?;
        check_2d(host.len(), host_pitch, width, rows)?;
        if source.home_stream().is_some_and(|home| home != self.id()) {
            return Err(HipError::refused(
                "copy of a buffer that belongs to another stream",
            ));
        }
        if rows == 0 || width == 0 {
            return Ok(());
        }
        let bytes = size_of::<T>();
        // SAFETY: as in `copy_2d`; `host` stays mutably borrowed until the
        // synchronisation below returns.
        let status = unsafe {
            sys::hipMemcpy2DAsync(
                host.as_mut_ptr().cast(),
                host_pitch * bytes,
                source.as_ptr().cast(),
                source_pitch * bytes,
                width * bytes,
                rows,
                sys::HIP_MEMCPY_DEVICE_TO_HOST,
                self.inner.raw,
            )
        };
        self.observe(check(status, "hipMemcpy2DAsync D2H"))?;
        self.synchronize()
    }

    /// Copies `rows` rows of `width` elements from pageable host memory to
    /// the device, `host_pitch` and `target_pitch` elements apart, and
    /// waits, so `host` may be reused as soon as this returns.
    pub fn upload_2d<T: DeviceCopy>(
        &self,
        host: &[T],
        host_pitch: usize,
        mut target: DeviceSliceMut<'_, T>,
        target_pitch: usize,
        width: usize,
        rows: usize,
    ) -> Result<(), HipError> {
        self.ensure_healthy()?;
        self.deny_while_recording("host/device copy")?;
        check_2d(host.len(), host_pitch, width, rows)?;
        check_2d(target.len(), target_pitch, width, rows)?;
        if target.home_stream() != self.id() {
            return Err(HipError::refused(
                "copy into a buffer that belongs to another stream",
            ));
        }
        if rows == 0 || width == 0 {
            return Ok(());
        }
        let bytes = size_of::<T>();
        // SAFETY: `check_2d` bounds every row of both ranges; the stream is
        // synchronised before `host` can be released.
        let status = unsafe {
            sys::hipMemcpy2DAsync(
                target.as_mut_ptr().cast(),
                target_pitch * bytes,
                host.as_ptr().cast(),
                host_pitch * bytes,
                width * bytes,
                rows,
                sys::HIP_MEMCPY_HOST_TO_DEVICE,
                self.inner.raw,
            )
        };
        self.observe(check(status, "hipMemcpy2DAsync H2D"))?;
        self.synchronize()
    }

    /// Enqueues a pinned-to-device copy and returns immediately. The pinned
    /// buffer refuses host access until the copy has completed.
    pub fn upload_async<T: DeviceCopy>(
        &self,
        host: &mut PinnedBuffer<T>,
        mut device: DeviceSliceMut<'_, T>,
    ) -> Result<(), HipError> {
        self.check_copy(host.len(), device.len(), Some(device.home_stream()))?;
        let bytes = host.len() * size_of::<T>();
        // Created first: if this fails, no copy is in flight.
        let (source, event) = host.begin_transfer()?;
        // SAFETY: both ranges hold `bytes`; the pinned buffer blocks host
        // access until the copy is confirmed complete (see `fence`).
        let status = unsafe {
            sys::hipMemcpyAsync(
                device.as_mut_ptr().cast(),
                source.cast_const().cast(),
                bytes,
                sys::HIP_MEMCPY_HOST_TO_DEVICE,
                self.inner.raw,
            )
        };
        self.fence(host, event, check(status, "hipMemcpyAsync H2D"))
    }

    /// [`Stream::upload_async`] of the first `len` elements of `host` into
    /// `device` (of that length).
    pub fn upload_async_front<T: DeviceCopy>(
        &self,
        host: &mut PinnedBuffer<T>,
        len: usize,
        mut device: DeviceSliceMut<'_, T>,
    ) -> Result<(), HipError> {
        if len > host.len() {
            return Err(HipError::refused("upload longer than its pinned buffer"));
        }
        self.check_copy(len, device.len(), Some(device.home_stream()))?;
        let bytes = len * size_of::<T>();
        let (source, event) = host.begin_transfer()?;
        // SAFETY: `len` elements of the pinned buffer and of `device`; the
        // pinned buffer blocks host access until the copy is confirmed
        // complete (see `fence`).
        let status = unsafe {
            sys::hipMemcpyAsync(
                device.as_mut_ptr().cast(),
                source.cast_const().cast(),
                bytes,
                sys::HIP_MEMCPY_HOST_TO_DEVICE,
                self.inner.raw,
            )
        };
        self.fence(host, event, check(status, "hipMemcpyAsync H2D"))
    }

    /// Copies raw bytes (of any alignment) into `device` through the two
    /// page-locked `staging` buffers, a buffer's length at a time, and
    /// returns once the last piece is enqueued. `host` is read only on the
    /// host, so it may be released right away; each buffer waits for its
    /// previous copy before it is refilled. A copy straight from a file
    /// mapping lets HIP pin the mapped page-cache pages for the transfer;
    /// when the host reclaims them, each invalidation evicts the process's
    /// GPU queues.
    pub fn upload_bytes_staged<T: DeviceCopy>(
        &self,
        host: &[u8],
        mut device: DeviceSliceMut<'_, T>,
        staging: &mut [PinnedBuffer<u8>; 2],
    ) -> Result<(), HipError> {
        self.check_copy(
            host.len(),
            device.len() * size_of::<T>(),
            Some(device.home_stream()),
        )?;
        let piece = staging[0].len().min(staging[1].len());
        if piece == 0 && !host.is_empty() {
            return Err(HipError::refused("empty staging buffer"));
        }
        let target = device.as_mut_ptr().cast::<u8>();
        for (index, chunk) in host.chunks(piece.max(1)).enumerate() {
            let buffer = &mut staging[index % 2];
            buffer.host_mut()?[..chunk.len()].copy_from_slice(chunk);
            let (source, event) = buffer.begin_transfer()?;
            // SAFETY: the piece lies within `device` (`check_copy` bounds
            // the whole copy) and within the pinned buffer, which blocks
            // host access until the copy is confirmed complete (see
            // `fence`). The device side belongs to this stream, so its
            // release is ordered after the copy.
            let status = unsafe {
                sys::hipMemcpyAsync(
                    target.wrapping_add(index * piece).cast(),
                    source.cast_const().cast(),
                    chunk.len(),
                    sys::HIP_MEMCPY_HOST_TO_DEVICE,
                    self.inner.raw,
                )
            };
            self.fence(buffer, event, check(status, "hipMemcpyAsync H2D"))?;
        }
        Ok(())
    }

    /// Enqueues a device-to-pinned copy and returns immediately. The pinned
    /// buffer refuses host access until the copy has completed.
    pub fn download_async<T: DeviceCopy>(
        &self,
        device: DeviceSlice<'_, T>,
        host: &mut PinnedBuffer<T>,
    ) -> Result<(), HipError> {
        self.check_copy(host.len(), device.len(), device.home_stream())?;
        let bytes = host.len() * size_of::<T>();
        let (destination, event) = host.begin_transfer()?;
        // SAFETY: as in `upload_async`.
        let status = unsafe {
            sys::hipMemcpyAsync(
                destination.cast(),
                device.as_ptr().cast(),
                bytes,
                sys::HIP_MEMCPY_DEVICE_TO_HOST,
                self.inner.raw,
            )
        };
        self.fence(host, event, check(status, "hipMemcpyAsync D2H"))
    }

    /// Enqueues a device-to-pinned copy of `device` into the front of `host`
    /// and returns immediately; like [`Stream::download_async`], the pinned
    /// buffer refuses host access until the copy has completed.
    pub fn download_async_front<T: DeviceCopy>(
        &self,
        device: DeviceSlice<'_, T>,
        host: &mut PinnedBuffer<T>,
    ) -> Result<(), HipError> {
        if device.len() > host.len() {
            return Err(HipError::refused("download longer than its pinned buffer"));
        }
        self.check_copy(device.len(), device.len(), device.home_stream())?;
        let bytes = device.len() * size_of::<T>();
        let (destination, event) = host.begin_transfer()?;
        // SAFETY: the first `device.len()` elements of the pinned buffer; it
        // blocks host access until the copy is confirmed complete.
        let status = unsafe {
            sys::hipMemcpyAsync(
                destination.cast(),
                device.as_ptr().cast(),
                bytes,
                sys::HIP_MEMCPY_DEVICE_TO_HOST,
                self.inner.raw,
            )
        };
        self.fence(host, event, check(status, "hipMemcpyAsync D2H"))
    }

    /// Starts a [`PinnedDownload`] into `host`: waits for the buffer's last
    /// copy, then lets any number of copies stay in flight together.
    pub fn begin_download<'a, T: DeviceCopy>(
        &'a self,
        host: &'a mut PinnedBuffer<T>,
    ) -> Result<PinnedDownload<'a, T>, HipError> {
        self.ensure_healthy()?;
        self.deny_while_recording("hipMemcpy2DAsync")?;
        let (base, event) = host.begin_transfer()?;
        Ok(PinnedDownload {
            stream: self,
            host,
            base,
            event: Some(event),
            enqueued: false,
        })
    }

    /// Copies device memory to device memory in stream order, without
    /// waiting. `source` may be an immutable tensor or a buffer of this stream.
    pub fn copy<T: DeviceCopy>(
        &self,
        source: DeviceSlice<'_, T>,
        mut destination: DeviceSliceMut<'_, T>,
    ) -> Result<(), HipError> {
        self.check_copy(source.len(), destination.len(), source.home_stream())?;
        if destination.home_stream() != self.id() {
            return Err(HipError::refused(
                "copy into a buffer that belongs to another stream",
            ));
        }
        // SAFETY: both views cover `len` elements of live allocations used in
        // this stream's order; the borrows keep them alive until the call
        // returns, and stream-ordered frees wait for the copy.
        let status = unsafe {
            sys::hipMemcpyAsync(
                destination.as_mut_ptr().cast(),
                source.as_ptr().cast(),
                source.len() * size_of::<T>(),
                sys::HIP_MEMCPY_DEVICE_TO_DEVICE,
                self.inner.raw,
            )
        };
        self.observe(check(status, "hipMemcpyAsync D2D"))
    }

    /// Copies `source`, which may belong to any stream, into `destination`
    /// (of this stream) and waits. For state that leaves its stream: a
    /// snapshot copied on a transfer stream while the owning stream runs
    /// other work. Work the owner still has queued on `source` is not ordered
    /// before the copy (ordering between streams is the caller's contract,
    /// as everywhere); the wait keeps both allocations alive for the whole
    /// copy, because neither can be freed while borrowed here.
    pub fn copy_across<T: DeviceCopy>(
        &self,
        source: DeviceSlice<'_, T>,
        destination: DeviceSliceMut<'_, T>,
    ) -> Result<(), HipError> {
        let width = source.len();
        self.copy_2d_across(source, width, destination, width, width, 1)
    }

    /// [`Stream::copy_across`] of every `(source, destination)` pair,
    /// waiting once after all of them: a snapshot's many sections in one
    /// transfer.
    pub fn copy_across_all<T: DeviceCopy>(
        &self,
        copies: &mut [(DeviceSlice<'_, T>, DeviceSliceMut<'_, T>)],
    ) -> Result<(), HipError> {
        self.ensure_healthy()?;
        self.deny_while_recording("hipMemcpyAsync")?;
        for (source, target) in copies.iter() {
            if source.len() != target.len() {
                return Err(HipError::refused("device copy length mismatch"));
            }
            if target.home_stream() != self.id() {
                return Err(HipError::refused(
                    "copy into a buffer that belongs to another stream",
                ));
            }
        }
        let mut enqueued = Ok(());
        for (source, target) in copies.iter_mut() {
            if source.is_empty() {
                continue;
            }
            // SAFETY: both views hold `len` elements of live allocations,
            // which the borrows keep alive until this returns; the wait below
            // follows every enqueued copy, so no stream-ordered free of either
            // allocation can overtake one.
            let status = unsafe {
                sys::hipMemcpyAsync(
                    target.as_mut_ptr().cast(),
                    source.as_ptr().cast(),
                    source.len() * size_of::<T>(),
                    sys::HIP_MEMCPY_DEVICE_TO_DEVICE,
                    self.inner.raw,
                )
            };
            enqueued = self.observe(check(status, "hipMemcpyAsync D2D"));
            if enqueued.is_err() {
                break;
            }
        }
        let waited = self.synchronize();
        enqueued.and(waited)
    }

    /// [`Stream::copy_2d`] from a buffer of any stream, waiting like
    /// [`Stream::copy_across`].
    pub fn copy_2d_across<T: DeviceCopy>(
        &self,
        source: DeviceSlice<'_, T>,
        source_pitch: usize,
        mut target: DeviceSliceMut<'_, T>,
        target_pitch: usize,
        width: usize,
        rows: usize,
    ) -> Result<(), HipError> {
        self.ensure_healthy()?;
        self.deny_while_recording("hipMemcpy2DAsync")?;
        if rows == 1 && source.len() != target.len() {
            return Err(HipError::refused("device copy length mismatch"));
        }
        check_2d(source.len(), source_pitch, width, rows)?;
        check_2d(target.len(), target_pitch, width, rows)?;
        if target.home_stream() != self.id() {
            return Err(HipError::refused(
                "copy into a buffer that belongs to another stream",
            ));
        }
        if rows == 0 || width == 0 {
            return Ok(());
        }
        let bytes = size_of::<T>();
        // SAFETY: `check_2d` bounds every row of both views inside their
        // allocations. The borrows keep both allocations from being freed
        // until this returns, and the synchronisation below returns only
        // after the copy has finished, so no stream-ordered free of either
        // allocation (enqueued later, on any stream) can overtake it.
        let status = unsafe {
            sys::hipMemcpy2DAsync(
                target.as_mut_ptr().cast(),
                target_pitch * bytes,
                source.as_ptr().cast(),
                source_pitch * bytes,
                width * bytes,
                rows,
                sys::HIP_MEMCPY_DEVICE_TO_DEVICE,
                self.inner.raw,
            )
        };
        self.observe(check(status, "hipMemcpy2DAsync D2D"))?;
        self.synchronize()
    }

    /// Copies the raw bytes of `source`, which may belong to any stream, into
    /// `host` and waits, like [`Stream::copy_across`]: for host snapshots of
    /// an idle session while its stream runs other sessions' work.
    pub fn download_bytes_across<T: DeviceCopy>(
        &self,
        source: DeviceSlice<'_, T>,
        host: &mut [u8],
    ) -> Result<(), HipError> {
        self.check_copy(host.len(), source.len() * size_of::<T>(), None)?;
        if host.is_empty() {
            return Ok(());
        }
        // SAFETY: both ranges hold `host.len()` bytes; the borrows keep the
        // allocation and `host` alive, and the synchronisation below returns
        // only after the copy has finished.
        let status = unsafe {
            sys::hipMemcpyAsync(
                host.as_mut_ptr().cast(),
                source.as_ptr().cast(),
                host.len(),
                sys::HIP_MEMCPY_DEVICE_TO_HOST,
                self.inner.raw,
            )
        };
        self.observe(check(status, "hipMemcpyAsync D2H"))?;
        self.synchronize()
    }

    /// Copies the raw bytes of `sources`, which may belong to any stream, one
    /// after another into the front of `host` and waits once, like
    /// [`Stream::download_bytes_across`]: page-locked memory lets the copies
    /// queue back to back instead of waiting for each, which a pageable
    /// destination does. Returns the bytes copied.
    pub fn download_bytes_across_pinned<T: DeviceCopy>(
        &self,
        sources: &[DeviceSlice<'_, T>],
        host: &mut PinnedBuffer<u8>,
    ) -> Result<usize, HipError> {
        let total = sources
            .iter()
            .try_fold(0usize, |sum, source| {
                sum.checked_add(source.len().checked_mul(size_of::<T>())?)
            })
            .ok_or(HipError::refused("pinned download size overflow"))?;
        if total > host.len() {
            return Err(HipError::refused("download longer than its pinned buffer"));
        }
        self.check_copy(total, total, None)?;
        if total == 0 {
            return Ok(0);
        }
        let (destination, event) = host.begin_transfer()?;
        let mut copied = 0usize;
        let mut enqueued = Ok(());
        for source in sources {
            let bytes = source.len() * size_of::<T>();
            if bytes == 0 {
                continue;
            }
            // SAFETY: `copied + bytes <= total <= host.len()` bytes of the
            // pinned buffer and `bytes` of `source`; the borrows keep the
            // allocations alive, and this call returns only after every
            // enqueued copy has finished (the sources may belong to another
            // stream, whose frees do not wait for this one).
            let status = unsafe {
                sys::hipMemcpyAsync(
                    destination.wrapping_add(copied).cast(),
                    source.as_ptr().cast(),
                    bytes,
                    sys::HIP_MEMCPY_DEVICE_TO_HOST,
                    self.inner.raw,
                )
            };
            enqueued = check(status, "hipMemcpyAsync D2H");
            if enqueued.is_err() {
                break;
            }
            copied += bytes;
        }
        if let Err(error) = self.observe(enqueued) {
            // The copies before the failed one may be in flight.
            if copied != 0 || error.is_sticky() {
                match self.synchronize() {
                    Err(unconfirmed) => host.set_unknown(unconfirmed),
                    Ok(()) if error.is_sticky() => host.set_unknown(error),
                    Ok(()) => {}
                }
            }
            return Err(error);
        }
        self.fence(host, event, Ok(()))?;
        host.wait()?;
        Ok(copied)
    }

    /// `len` zeroed indices, all below `bound` (at least 1).
    pub fn alloc_indices(&self, len: usize, bound: u32) -> Result<DeviceIndices, HipError> {
        self.alloc_indices_with(len, IndexSpec::below(bound))
    }

    /// `len` zeroed indices with `spec` (a positive bound).
    pub fn alloc_indices_with(
        &self,
        len: usize,
        spec: IndexSpec,
    ) -> Result<DeviceIndices, HipError> {
        self.zeroed_indices_with(self.alloc::<u32>(len)?, spec)
    }

    /// Turns `buffer` (of this stream) into zeroed indices below `bound`, for
    /// index buffers from [`Stream::alloc_persistent`].
    pub fn zeroed_indices(
        &self,
        buffer: DeviceBuffer<u32>,
        bound: u32,
    ) -> Result<DeviceIndices, HipError> {
        self.zeroed_indices_with(buffer, IndexSpec::below(bound))
    }

    /// [`Stream::zeroed_indices`] with a full spec.
    pub fn zeroed_indices_with(
        &self,
        mut buffer: DeviceBuffer<u32>,
        spec: IndexSpec,
    ) -> Result<DeviceIndices, HipError> {
        if spec.bound == 0 {
            return Err(HipError::refused("index bound must be positive"));
        }
        self.fill_bytes(buffer.as_mut_slice(), 0)?;
        Ok(DeviceIndices::new(buffer, spec))
    }

    /// Checks that every value of `host` satisfies the spec of `indices`,
    /// then uploads it to the front of `indices` and waits.
    pub fn upload_indices(
        &self,
        host: &[u32],
        indices: &mut DeviceIndices,
    ) -> Result<(), HipError> {
        let spec = indices.spec();
        if host.iter().any(|&value| !spec.admits(value)) {
            return Err(HipError::refused("index out of range for its buffer"));
        }
        if host.len() > indices.len() {
            return Err(HipError::refused("more indices than the buffer holds"));
        }
        let buffer = indices.upload_target();
        self.upload(host, buffer.slice_mut(0..host.len()))
    }

    /// Checks the first `len` values of `host` against the spec of
    /// `indices`, then enqueues their upload to the front of `indices`
    /// without waiting. The pinned buffer refuses host access until the copy
    /// has completed, so the checked values are the copied ones.
    pub fn upload_indices_async(
        &self,
        host: &mut PinnedBuffer<u32>,
        len: usize,
        indices: &mut DeviceIndices,
    ) -> Result<(), HipError> {
        let spec = indices.spec();
        let values = host.host()?;
        if len > values.len() || len > indices.len() {
            return Err(HipError::refused("more indices than the buffers hold"));
        }
        if values[..len].iter().any(|&value| !spec.admits(value)) {
            return Err(HipError::refused("index out of range for its buffer"));
        }
        let buffer = indices.upload_target();
        self.upload_async_front(host, len, buffer.slice_mut(0..len))
    }

    /// Turns `buffer` (of this stream) into a zeroed [`DeviceIndexPair`]
    /// whose second range starts at `split`.
    pub fn zeroed_index_pair(
        &self,
        mut buffer: DeviceBuffer<u32>,
        split: usize,
        specs: [IndexSpec; 2],
    ) -> Result<DeviceIndexPair, HipError> {
        if specs.iter().any(|spec| spec.bound == 0) {
            return Err(HipError::refused("index bound must be positive"));
        }
        if split > buffer.len() {
            return Err(HipError::refused("index pair split beyond its buffer"));
        }
        self.fill_bytes(buffer.as_mut_slice(), 0)?;
        Ok(DeviceIndexPair::new(buffer, split, specs))
    }

    /// Checks the first `pair.len()` values of `host`, each against its
    /// range's spec, then enqueues their upload into `pair` as one copy
    /// without waiting. The pinned buffer refuses host access until the copy
    /// has completed, so the checked values are the copied ones.
    pub fn upload_index_pair_async(
        &self,
        host: &mut PinnedBuffer<u32>,
        pair: &mut DeviceIndexPair,
    ) -> Result<(), HipError> {
        let (len, split, [first, second]) = (pair.len(), pair.split(), pair.specs());
        let values = host.host()?;
        if len > values.len() {
            return Err(HipError::refused(
                "index pair longer than its pinned buffer",
            ));
        }
        let admitted = values[..split].iter().all(|&value| first.admits(value))
            && values[split..len].iter().all(|&value| second.admits(value));
        if !admitted {
            return Err(HipError::refused("index out of range for its buffer"));
        }
        let buffer = pair.upload_target();
        self.upload_async_front(host, len, buffer.slice_mut(0..len))
    }

    /// Tracks an enqueued copy with `event`. Whenever completion cannot be
    /// confirmed, the pinned buffer is locked for good.
    fn fence<T: DeviceCopy>(
        &self,
        host: &mut PinnedBuffer<T>,
        event: Event,
        enqueued: Result<(), HipError>,
    ) -> Result<(), HipError> {
        if let Err(error) = self.observe(enqueued) {
            // HIP rejects invalid copies before enqueueing them; after a
            // device fault the transfer state is unknown.
            if error.is_sticky() {
                host.set_unknown(error);
            }
            return Err(error);
        }
        match self.record(&event) {
            Ok(()) => {
                host.set_in_flight(event);
                Ok(())
            }
            Err(error) => {
                // The copy is enqueued but untracked: wait for the stream.
                if let Err(unconfirmed) = self.synchronize() {
                    host.set_unknown(unconfirmed);
                }
                Err(error)
            }
        }
    }

    /// `home` is the device side's home stream; `None` (an immutable tensor)
    /// may be read on any stream.
    fn check_copy(&self, host: usize, device: usize, home: Option<u64>) -> Result<(), HipError> {
        self.ensure_healthy()?;
        self.deny_while_recording("host/device copy")?;
        if host != device {
            return Err(HipError::refused("host/device copy length mismatch"));
        }
        if home.is_some_and(|home| home != self.id()) {
            return Err(HipError::refused(
                "copy of a buffer that belongs to another stream",
            ));
        }
        Ok(())
    }
}

/// Whether `rows` rows of `width` elements, `pitch` apart, lie within `len`.
/// Device-to-pinned copies that stay in flight together
/// ([`Stream::begin_download`]): each [`PinnedDownload::copy_2d`] enqueues
/// without waiting, and [`PinnedDownload::finish`] (or dropping the download)
/// records one event that every host access of the buffer waits for. The
/// download borrows the buffer mutably, so the host cannot touch it while
/// copies are being enqueued: a prefill writes the rows its layers leave into
/// one buffer and waits once, when they are read.
pub struct PinnedDownload<'a, T: DeviceCopy> {
    stream: &'a Stream,
    host: &'a mut PinnedBuffer<T>,
    base: *mut T,
    /// `None` once fenced, or when a device fault left the transfer state
    /// unknown (the buffer then stays locked).
    event: Option<Event>,
    enqueued: bool,
}

impl<T: DeviceCopy> PinnedDownload<'_, T> {
    /// Enqueues `rows` rows of `width` elements of `source`, `source_pitch`
    /// apart, into the pinned buffer from element `offset`, `host_pitch`
    /// apart.
    pub fn copy_2d(
        &mut self,
        source: DeviceSlice<'_, T>,
        source_pitch: usize,
        offset: usize,
        host_pitch: usize,
        width: usize,
        rows: usize,
    ) -> Result<(), HipError> {
        let s = self.stream;
        s.ensure_healthy()?;
        s.deny_while_recording("hipMemcpy2DAsync")?;
        check_2d(source.len(), source_pitch, width, rows)?;
        let room = self
            .host
            .len()
            .checked_sub(offset)
            .ok_or(HipError::refused(
                "download past the end of its pinned buffer",
            ))?;
        check_2d(room, host_pitch, width, rows)?;
        if source.home_stream().is_some_and(|home| home != s.id()) {
            return Err(HipError::refused(
                "copy of a buffer that belongs to another stream",
            ));
        }
        if rows == 0 || width == 0 || self.event.is_none() {
            return Ok(());
        }
        let bytes = size_of::<T>();
        // SAFETY: the rows lie inside both allocations (checked above). The
        // pinned buffer is borrowed by this download, whose event every host
        // access waits for; the device allocation is freed in stream order,
        // after the copy.
        let status = unsafe {
            sys::hipMemcpy2DAsync(
                self.base.add(offset).cast(),
                host_pitch * bytes,
                source.as_ptr().cast(),
                source_pitch * bytes,
                width * bytes,
                rows,
                sys::HIP_MEMCPY_DEVICE_TO_HOST,
                s.inner.raw,
            )
        };
        if let Err(error) = s.observe(check(status, "hipMemcpy2DAsync D2H")) {
            // HIP rejects invalid copies before enqueueing them; after a
            // device fault the earlier copies' state is unknown.
            if error.is_sticky() {
                self.event = None;
                self.host.set_unknown(error);
            }
            return Err(error);
        }
        self.enqueued = true;
        Ok(())
    }

    /// Records the event of every enqueued copy.
    pub fn finish(mut self) -> Result<(), HipError> {
        self.fence()
    }

    fn fence(&mut self) -> Result<(), HipError> {
        let Some(event) = self.event.take() else {
            return Ok(());
        };
        if !self.enqueued {
            return Ok(());
        }
        match self.stream.record(&event) {
            Ok(()) => {
                self.host.set_in_flight(event);
                Ok(())
            }
            Err(error) => {
                // The copies are enqueued but untracked: wait for the stream.
                if let Err(unconfirmed) = self.stream.synchronize() {
                    self.host.set_unknown(unconfirmed);
                }
                Err(error)
            }
        }
    }
}

impl<T: DeviceCopy> Drop for PinnedDownload<'_, T> {
    fn drop(&mut self) {
        let _ = self.fence();
    }
}

fn check_2d(len: usize, pitch: usize, width: usize, rows: usize) -> Result<(), HipError> {
    if rows == 0 || width == 0 {
        return Ok(());
    }
    let end = (rows - 1)
        .checked_mul(pitch)
        .and_then(|start| start.checked_add(width));
    if width > pitch || end.is_none_or(|end| end > len) {
        return Err(HipError::refused("2D copy rows lie outside the buffer"));
    }
    Ok(())
}

/// A HIP event, for completion tracking and, with [`Event::timed`], timing.
pub struct Event {
    raw: sys::HipEvent,
}

// SAFETY: HIP event handles may be used from any thread.
unsafe impl Send for Event {}

impl Event {
    /// Completion-only event (no timestamp; cheaper to record).
    pub fn new() -> Result<Self, HipError> {
        Self::with_flags(sys::HIP_EVENT_DISABLE_TIMING)
    }

    /// Event that records a timestamp, for [`Event::elapsed_ms`].
    pub fn timed() -> Result<Self, HipError> {
        Self::with_flags(sys::HIP_EVENT_DEFAULT)
    }

    fn with_flags(flags: u32) -> Result<Self, HipError> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: valid out-pointer.
        let status = unsafe { sys::hipEventCreateWithFlags(&mut raw, flags) };
        check(status, "hipEventCreateWithFlags")?;
        Ok(Self { raw })
    }

    pub fn synchronize(&self) -> Result<(), HipError> {
        // SAFETY: live handle.
        check(
            unsafe { sys::hipEventSynchronize(self.raw) },
            "hipEventSynchronize",
        )
    }

    /// Whether all work before the recording has completed, without blocking.
    pub fn query(&self) -> Result<bool, HipError> {
        // SAFETY: live handle.
        match unsafe { sys::hipEventQuery(self.raw) } {
            sys::HIP_SUCCESS => Ok(true),
            sys::HIP_ERROR_NOT_READY => Ok(false),
            code => Err(HipError::new(code, "hipEventQuery")),
        }
    }

    /// Milliseconds between two completed timed events.
    pub fn elapsed_ms(start: &Event, end: &Event) -> Result<f32, HipError> {
        let mut ms = 0.0f32;
        // SAFETY: valid out-pointer and live handles.
        let status = unsafe { sys::hipEventElapsedTime(&mut ms, start.raw, end.raw) };
        check(status, "hipEventElapsedTime")?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: created by hipEventCreateWithFlags, destroyed once. HIP
        // defers the release until recorded work has completed.
        let _ = unsafe { sys::hipEventDestroy(self.raw) };
    }
}

#[cfg(all(test, feature = "gpu"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::fault;

    const LEN: usize = 1024;

    fn setup() -> (Stream, DeviceBuffer<u32>, PinnedBuffer<u32>) {
        let stream = Stream::new().unwrap();
        let mut device = stream.alloc::<u32>(LEN).unwrap();
        stream.upload(&[7; LEN], device.as_mut_slice()).unwrap();
        (stream, device, PinnedBuffer::zeroed(LEN).unwrap())
    }

    fn pool_attribute(attribute: i32) -> u64 {
        let mut pool = std::ptr::null_mut();
        let mut value = 0u64;
        // SAFETY: valid out-pointers; both attributes are u64 values.
        unsafe {
            check(sys::hipDeviceGetDefaultMemPool(&mut pool, 0), "pool").unwrap();
            check(
                sys::hipMemPoolGetAttribute(pool, attribute, (&raw mut value).cast()),
                "attribute",
            )
            .unwrap();
        }
        value
    }

    #[test]
    fn the_pool_keeps_up_to_its_reserve_free() {
        let stream = Stream::new().unwrap();
        // In use beside the reserve: the threshold bounds free memory only.
        let live = stream.alloc::<u8>(3 << 30).unwrap();
        assert_eq!(
            pool_attribute(sys::HIP_MEM_POOL_ATTR_RELEASE_THRESHOLD),
            POOL_RETAINED_BYTES
        );
        let bytes = 256 << 20;
        drop(stream.alloc::<u8>(bytes).unwrap());
        stream.synchronize().unwrap();
        // HIP's default threshold of 0, or one set before the pool's first
        // allocation, would have returned the block here.
        let reserved = pool_attribute(sys::HIP_MEM_POOL_ATTR_RESERVED_MEM_CURRENT);
        assert!(reserved >= (3 << 30) + bytes as u64);
        // Beyond the reserve, free memory goes back to the system.
        drop(live);
        stream.synchronize().unwrap();
        assert!(
            pool_attribute(sys::HIP_MEM_POOL_ATTR_RESERVED_MEM_CURRENT) <= reserved - (1 << 30)
        );
    }

    #[test]
    fn a_failed_event_creation_enqueues_nothing() {
        // The review's sequence: event creation fails, then synchronisation
        // would fail too. The event now comes first, so no copy is in flight.
        let (stream, device, mut pinned) = setup();
        fault::inject("hipEventCreateWithFlags", 2);
        fault::inject("hipStreamSynchronize", 700);
        assert!(
            stream
                .download_async(device.as_slice(), &mut pinned)
                .is_err()
        );
        assert!(pinned.is_ready().unwrap());
        assert_eq!(pinned.host().unwrap(), &[0; LEN][..], "nothing was copied");
        assert!(
            fault::take("hipStreamSynchronize").is_some(),
            "no synchronisation was needed"
        );
    }

    #[test]
    fn a_failed_record_is_covered_by_a_stream_synchronisation() {
        let (stream, device, mut pinned) = setup();
        fault::inject("hipEventRecord", 2);
        assert!(
            stream
                .download_async(device.as_slice(), &mut pinned)
                .is_err()
        );
        assert_eq!(
            pinned.host().unwrap(),
            &[7; LEN][..],
            "the enqueued copy completed"
        );
    }

    #[test]
    fn an_unconfirmed_copy_locks_the_buffer_for_good() {
        let (stream, device, mut pinned) = setup();
        fault::inject("hipEventRecord", 2);
        fault::inject("hipStreamSynchronize", 700);
        assert!(
            stream
                .download_async(device.as_slice(), &mut pinned)
                .is_err()
        );
        assert!(pinned.is_ready().is_err());
        assert!(pinned.host_mut().is_err());
        assert!(
            pinned.host().is_err(),
            "still locked without further faults"
        );
        assert!(
            stream.poisoned().is_some(),
            "the device fault poisons the stream"
        );
        drop(pinned); // leaked, not freed
    }

    #[test]
    fn a_failed_wait_locks_the_buffer_for_good() {
        let (stream, device, mut pinned) = setup();
        stream
            .download_async(device.as_slice(), &mut pinned)
            .unwrap();
        fault::inject("hipEventSynchronize", 700);
        assert!(pinned.host().is_err());
        assert!(pinned.host().is_err(), "a later wait does not unlock it");
    }

    #[test]
    fn a_rejected_upload_leaves_the_buffer_usable() {
        let (stream, mut device, mut pinned) = setup();
        fault::inject("hipMemcpyAsync H2D", 1);
        assert!(
            stream
                .upload_async(&mut pinned, device.as_mut_slice())
                .is_err()
        );
        assert!(
            pinned.host_mut().is_ok(),
            "an invalid copy was never enqueued"
        );
    }
}
