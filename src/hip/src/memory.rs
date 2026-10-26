//! Device allocations, borrowed device views and pinned host memory.

use std::cell::RefCell;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use gufo_formats::GgmlType;

use crate::deferred::{self, Deferred};
use crate::device::{PointerAttributes, pointer_attributes};
use crate::error::{HipError, check};
use crate::host::{HostRelease, Residency, residency};
use crate::mapped::MappedFile;
use crate::stream::{Event, Stream, StreamInner};
use crate::sys;

/// Plain-old-data element types that may live in device memory.
///
/// # Safety
/// Implementors must be `Copy` and valid for every bit pattern, because
/// kernels and DMA write them without Rust's involvement.
pub unsafe trait DeviceCopy: Copy + 'static {}

// SAFETY: primitive numbers are valid for every bit pattern.
unsafe impl DeviceCopy for f32 {}
// SAFETY: as above.
unsafe impl DeviceCopy for u8 {}
// SAFETY: as above.
unsafe impl DeviceCopy for i8 {}
// SAFETY: as above.
unsafe impl DeviceCopy for u16 {}
// SAFETY: as above.
unsafe impl DeviceCopy for u32 {}
// SAFETY: as above.
unsafe impl DeviceCopy for i32 {}
// SAFETY: as above.
unsafe impl DeviceCopy for u64 {}
// SAFETY: transparent wrapper around u16.
unsafe impl DeviceCopy for Bf16 {}
// SAFETY: transparent wrapper around u16.
unsafe impl DeviceCopy for F16 {}

/// IEEE bfloat16 storage, bit-compatible with `hip_bfloat16`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Bf16(pub u16);

impl Bf16 {
    /// Round to nearest even, matching `hip_bfloat16(float)` for finite values.
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
        Self((rounded >> 16) as u16)
    }

    pub fn to_f32(self) -> f32 {
        f32::from_bits(u32::from(self.0) << 16)
    }
}

/// IEEE binary16 storage, bit-compatible with `__half`. Host code only moves
/// it; kernels interpret it.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct F16(pub u16);

static NEXT_ALLOCATION_ID: AtomicU64 = AtomicU64::new(1);

/// One device allocation: the unit that buffers, tensors and captured graphs
/// keep alive. The memory is freed when the last holder drops it.
///
/// `owned` is true while a user-facing handle ([`DeviceBuffer`] or any clone
/// of a [`DeviceTensor`]) exists. A captured graph that still holds an
/// allocation whose owner is gone is stale: it would write into memory the
/// program no longer reads.
pub struct Allocation {
    ptr: NonNull<c_void>,
    bytes: usize,
    id: u64,
    home: Arc<StreamInner>,
    shared: bool,
    owned: AtomicBool,
    backing: Backing,
}

/// Where the memory behind an allocation comes from.
enum Backing {
    /// `hipMallocAsync` on the home stream.
    Device,
    /// `hipMalloc`: exactly sized, freed after a device synchronisation.
    Exact,
    /// A read-only file mapping registered with `hipHostRegister`; `ptr` is
    /// its device address. The mapping stays alive while the allocation does.
    Mapped(Arc<MappedFile>),
    /// `hipMallocManaged`: one address for the host and the device.
    Managed,
    /// Host-visible memory (`HostMemory`); `ptr` is its device alias. Taken
    /// when the allocation is released.
    Host(Option<HostRelease>),
}

// SAFETY: the allocation is only an address and bookkeeping; device access is
// ordered by streams, and `owned` is atomic.
unsafe impl Send for Allocation {}
// SAFETY: as above.
unsafe impl Sync for Allocation {}

impl Allocation {
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Whether a buffer or tensor handle still owns this memory.
    pub fn is_owned(&self) -> bool {
        self.owned.load(Ordering::Acquire)
    }

    /// Stream that may write this memory and orders its free; `None` for
    /// immutable memory that every stream may read.
    pub fn home_stream(&self) -> Option<u64> {
        (!self.shared).then_some(self.home.id)
    }

    /// Whether the host addresses this memory too (managed or host memory).
    pub fn is_host_visible(&self) -> bool {
        self.host_base().is_some()
    }

    /// Whether the host addresses this memory at its device address, so
    /// host code may write through a device pointer into it (managed
    /// memory, and host memory whose device alias is its host address).
    pub fn is_host_aliased(&self) -> bool {
        self.host_base() == Some(self.ptr.cast())
    }

    /// The host address of the allocation's first byte, for managed and
    /// host memory.
    pub(crate) fn host_base(&self) -> Option<NonNull<u8>> {
        match &self.backing {
            Backing::Managed => Some(self.ptr.cast()),
            Backing::Host(Some(release)) => Some(release.host()),
            _ => None,
        }
    }

    /// The host address of device address `device` within this allocation.
    pub(crate) fn host_address(&self, device: *const c_void) -> Option<*mut u8> {
        let offset = (device as usize).checked_sub(self.ptr.as_ptr() as usize)?;
        if offset > self.bytes {
            return None;
        }
        self.host_base()
            .map(|base| base.as_ptr().wrapping_add(offset))
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        match &mut self.backing {
            Backing::Mapped(mapping) => {
                // Unregistering may synchronise the device like a shared free.
                deferred::defer(Deferred::Registered {
                    mapping: Arc::clone(mapping),
                });
                return;
            }
            Backing::Exact | Backing::Managed => {
                // hipFree synchronises the device implicitly.
                deferred::defer(Deferred::Exact { ptr: self.ptr });
                return;
            }
            Backing::Host(release) => {
                // Freeing host memory may synchronise the device.
                if let Some(release) = release.take() {
                    deferred::defer(Deferred::HostMemory(release));
                }
                return;
            }
            Backing::Device => {}
        }
        // Shared memory needs a device-wide synchronisation, and a free on a
        // capturing stream would become part of the graph: both wait until no
        // capture runs (see `deferred`).
        if self.shared || self.home.is_capturing() {
            deferred::defer(Deferred::Device {
                ptr: self.ptr,
                home: Arc::clone(&self.home),
                shared: self.shared,
            });
            return;
        }
        // SAFETY: the pointer came from hipMallocAsync on `home`, which the Arc
        // keeps alive. Stream-owned memory is only used on its home stream, so
        // the stream-ordered free follows every use.
        unsafe {
            let _ = sys::hipFreeAsync(self.ptr.as_ptr(), self.home.raw);
        }
    }
}

/// The user-facing owner of an allocation; retires it when dropped.
pub(crate) struct Owner {
    allocation: Arc<Allocation>,
}

impl Owner {
    /// Immutable memory readable on every stream: a registered file mapping
    /// whose device address is `device`. Bookkeeping uses `home`.
    pub(crate) fn mapped(
        home: &Arc<StreamInner>,
        device: NonNull<c_void>,
        mapping: Arc<MappedFile>,
    ) -> Self {
        Self {
            allocation: Arc::new(Allocation {
                ptr: device,
                bytes: mapping.len(),
                id: NEXT_ALLOCATION_ID.fetch_add(1, Ordering::Relaxed),
                home: Arc::clone(home),
                shared: true,
                owned: AtomicBool::new(true),
                backing: Backing::Mapped(mapping),
            }),
        }
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        self.allocation.owned.store(false, Ordering::Release);
    }
}

/// Owned, mutable device memory of one stream, freed in that stream's order.
pub struct DeviceBuffer<T: DeviceCopy> {
    owner: Owner,
    len: usize,
    _element: PhantomData<T>,
}

impl<T: DeviceCopy> DeviceBuffer<T> {
    /// `exact`: `hipMalloc` instead of the stream-ordered pool.
    pub(crate) fn allocate(
        home: &Arc<StreamInner>,
        len: usize,
        exact: bool,
    ) -> Result<Self, HipError> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(HipError::refused("device allocation size overflow"))?;
        let mut raw = std::ptr::null_mut::<c_void>();
        let (status, operation, backing) = if exact {
            // Allocation may synchronise the device: wait until no capture
            // runs (the free queue is drained on the way).
            let _exclusive = deferred::exclusive("hipMalloc")?;
            // SAFETY: valid out-pointer.
            (
                unsafe { sys::hipMalloc(&mut raw, bytes.max(1)) },
                "hipMalloc",
                Backing::Exact,
            )
        } else {
            // SAFETY: valid out-pointer and a live stream.
            (
                unsafe { sys::hipMallocAsync(&mut raw, bytes.max(1), home.raw) },
                "hipMallocAsync",
                Backing::Device,
            )
        };
        check(status, operation)?;
        let ptr = NonNull::new(raw).ok_or(HipError::refused("device allocation returned null"))?;
        if !exact {
            crate::stream::retain_pool_memory();
        }
        let allocation = Allocation {
            ptr,
            bytes,
            id: NEXT_ALLOCATION_ID.fetch_add(1, Ordering::Relaxed),
            home: Arc::clone(home),
            shared: false,
            owned: AtomicBool::new(true),
            backing,
        };
        Ok(Self {
            owner: Owner {
                allocation: Arc::new(allocation),
            },
            len,
            _element: PhantomData,
        })
    }

    /// Managed memory of `home` (`hipMallocManaged`, attached globally).
    pub(crate) fn allocate_managed(home: &Arc<StreamInner>, len: usize) -> Result<Self, HipError> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(HipError::refused("managed allocation size overflow"))?;
        let mut raw = std::ptr::null_mut::<c_void>();
        // Allocation may synchronise the device: wait until no capture runs.
        let _exclusive = deferred::exclusive("hipMallocManaged")?;
        // SAFETY: valid out-pointer.
        let status =
            unsafe { sys::hipMallocManaged(&mut raw, bytes.max(1), sys::HIP_MEM_ATTACH_GLOBAL) };
        check(status, "hipMallocManaged")?;
        let ptr = NonNull::new(raw).ok_or(HipError::refused("hipMallocManaged returned null"))?;
        Ok(Self::adopt(home, ptr, bytes, len, Backing::Managed))
    }

    /// A buffer of `home` over host memory whose device alias is `device`.
    pub(crate) fn host_visible(
        home: &Arc<StreamInner>,
        device: NonNull<c_void>,
        release: HostRelease,
    ) -> DeviceBuffer<u8> {
        let bytes = release.bytes();
        DeviceBuffer::adopt(home, device, bytes, bytes, Backing::Host(Some(release)))
    }

    fn adopt(
        home: &Arc<StreamInner>,
        ptr: NonNull<c_void>,
        bytes: usize,
        len: usize,
        backing: Backing,
    ) -> Self {
        Self {
            owner: Owner {
                allocation: Arc::new(Allocation {
                    ptr,
                    bytes,
                    id: NEXT_ALLOCATION_ID.fetch_add(1, Ordering::Relaxed),
                    home: Arc::clone(home),
                    shared: false,
                    owned: AtomicBool::new(true),
                    backing,
                }),
            },
            len,
            _element: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn allocation(&self) -> &Arc<Allocation> {
        &self.owner.allocation
    }

    /// The host address of managed or host memory once no device work uses
    /// it: waits for the home stream, the only one that may use the buffer.
    /// Refused for device-only memory and while a captured graph holds the
    /// allocation, because a replay could write it.
    fn settled_host_base(&self) -> Result<NonNull<u8>, HipError> {
        let allocation = &self.owner.allocation;
        let base = allocation
            .host_base()
            .ok_or(HipError::refused("host view of device-only memory"))?;
        if Arc::strong_count(allocation) != 1 {
            return Err(HipError::refused(
                "host view of memory a captured graph holds",
            ));
        }
        // SAFETY: a live stream handle.
        let status = unsafe { sys::hipStreamSynchronize(allocation.home.raw) };
        check(status, "hipStreamSynchronize")?;
        Ok(base)
    }

    /// The host view of managed or host memory (see `settled_host_base`).
    pub fn host(&self) -> Result<&[T], HipError> {
        let base = self.settled_host_base()?;
        // SAFETY: `len` elements of live memory the host addresses. No device
        // work is pending, and none can write it while `self` is borrowed:
        // device writes need `&mut self`, and no graph holds the memory.
        Ok(unsafe { std::slice::from_raw_parts(base.as_ptr().cast::<T>(), self.len) })
    }

    /// The mutable host view of managed or host memory.
    pub fn host_mut(&mut self) -> Result<&mut [T], HipError> {
        let base = self.settled_host_base()?;
        // SAFETY: as in `host`, and `&mut self` excludes every device view.
        Ok(unsafe { std::slice::from_raw_parts_mut(base.as_ptr().cast::<T>(), self.len) })
    }

    /// Resident host pages of managed or host memory (`mincore`); `None`
    /// for device-only memory.
    pub fn host_residency(&self) -> Option<Result<Residency, String>> {
        let base = self.owner.allocation.host_base()?;
        Some(residency(base, self.len * size_of::<T>()))
    }

    /// What HIP reports about the buffer's device address
    /// (`hipPointerGetAttributes`).
    pub fn pointer_attributes(&self) -> Result<PointerAttributes, HipError> {
        pointer_attributes(self.owner.allocation.ptr.as_ptr())
    }

    /// Advises HIP to keep managed memory on `device`
    /// (`hipMemAdvise`, `hipMemAdviseSetPreferredLocation`).
    pub fn advise_preferred_location(&self, device: i32) -> Result<(), HipError> {
        // SAFETY: the range is this buffer's live allocation; advice changes
        // placement, not contents.
        let status = unsafe {
            sys::hipMemAdvise(
                self.owner.allocation.ptr.as_ptr(),
                self.len * size_of::<T>(),
                sys::HIP_MEM_ADVISE_SET_PREFERRED_LOCATION,
                device,
            )
        };
        check(status, "hipMemAdvise")
    }

    /// Lets the device cache managed memory as it caches device memory
    /// (`hipMemAdvise`, `hipMemAdviseSetCoarseGrain`). A refused advice
    /// leaves the memory fine-grain and is cleared from the sticky error.
    pub fn advise_coarse_grain(&self) -> Result<(), HipError> {
        // SAFETY: the range is this buffer's live allocation; advice changes
        // caching, not contents.
        let status = unsafe {
            sys::hipMemAdvise(
                self.owner.allocation.ptr.as_ptr(),
                self.len * size_of::<T>(),
                sys::HIP_MEM_ADVISE_SET_COARSE_GRAIN,
                0,
            )
        };
        let result = check(status, "hipMemAdvise");
        if result.is_err() {
            // SAFETY: reads and clears the thread's last HIP error.
            let _ = unsafe { sys::hipGetLastError() };
        }
        result
    }

    /// Migrates managed memory to `device`, or to the host for `None`, in
    /// the order of `stream`, the home stream (`hipMemPrefetchAsync`).
    pub fn prefetch(&self, stream: &Stream, device: Option<i32>) -> Result<(), HipError> {
        if self.owner.allocation.home.id != stream.id() {
            return Err(HipError::refused(
                "prefetch on a stream that does not own the buffer",
            ));
        }
        // SAFETY: as in `advise_preferred_location`; migration keeps the
        // contents, and stream-ordered frees wait for it.
        let status = unsafe {
            sys::hipMemPrefetchAsync(
                self.owner.allocation.ptr.as_ptr(),
                self.len * size_of::<T>(),
                device.unwrap_or(sys::HIP_CPU_DEVICE_ID),
                stream.as_raw(),
            )
        };
        check(status, "hipMemPrefetchAsync")
    }

    pub fn as_slice(&self) -> DeviceSlice<'_, T> {
        DeviceSlice::whole(&self.owner.allocation, self.len)
    }

    pub fn as_mut_slice(&mut self) -> DeviceSliceMut<'_, T> {
        DeviceSliceMut {
            ptr: self.owner.allocation.ptr.as_ptr().cast(),
            len: self.len,
            allocation: &self.owner.allocation,
            _borrow: PhantomData,
        }
    }

    /// Sub-view; panics on an out-of-range request, like slice indexing.
    pub fn slice(&self, range: Range<usize>) -> DeviceSlice<'_, T> {
        self.as_slice().range(range)
    }

    pub fn slice_mut(&mut self, range: Range<usize>) -> DeviceSliceMut<'_, T> {
        self.as_mut_slice().range(range)
    }

    /// Views of several ranges for one launch: the `shared` ranges may
    /// overlap one another, an `exclusive` range overlaps no other range.
    /// Panics otherwise, or on an out-of-range request, like slice indexing.
    pub fn views<const S: usize, const E: usize>(
        &mut self,
        shared: [Range<usize>; S],
        exclusive: [Range<usize>; E],
    ) -> ([DeviceSlice<'_, T>; S], [DeviceSliceMut<'_, T>; E]) {
        for range in shared.iter().chain(&exclusive) {
            assert!(
                range.start <= range.end && range.end <= self.len,
                "device slice out of range"
            );
        }
        // Empty views name no memory, so they overlap nothing.
        let overlap = |a: &Range<usize>, b: &Range<usize>| {
            !a.is_empty() && !b.is_empty() && a.start < b.end && b.start < a.end
        };
        for (i, range) in exclusive.iter().enumerate() {
            assert!(
                !shared.iter().any(|other| overlap(range, other))
                    && !exclusive[i + 1..].iter().any(|other| overlap(range, other)),
                "an exclusive device view overlaps another view"
            );
        }
        let allocation = &self.owner.allocation;
        let base = allocation.ptr.as_ptr().cast::<T>();
        // `wrapping_add` keeps this safe code: the addresses are only
        // dereferenced on the device, inside the bounds checked above.
        (
            shared.map(|range| DeviceSlice {
                ptr: base.cast_const().wrapping_add(range.start),
                len: range.end - range.start,
                allocation,
                _borrow: PhantomData,
            }),
            exclusive.map(|range| DeviceSliceMut {
                ptr: base.wrapping_add(range.start),
                len: range.end - range.start,
                allocation,
                _borrow: PhantomData,
            }),
        )
    }

    /// The buffer's bytes for readers on other streams ([`SharedBytes`]).
    pub fn share(&self) -> SharedBytes {
        SharedBytes {
            allocation: Arc::clone(&self.owner.allocation),
            len: self.len * size_of::<T>(),
        }
    }

    /// Turns the buffer into an immutable tensor that every stream may read.
    ///
    /// Waits for `stream` (the home stream) so that no write is still in
    /// flight. Fails while a captured graph holds the memory, because the graph
    /// could write it again.
    pub fn freeze(self, stream: &Stream) -> Result<DeviceTensor<T>, HipError> {
        if self.owner.allocation.home.id != stream.id() {
            return Err(HipError::refused(
                "freeze on a stream that does not own the buffer",
            ));
        }
        stream.synchronize()?;
        let len = self.len;
        let mut owner = self.owner;
        let allocation = Arc::get_mut(&mut owner.allocation).ok_or(HipError::refused(
            "freeze of a buffer that a captured graph still holds",
        ))?;
        allocation.shared = true;
        Ok(DeviceTensor {
            owner: Arc::new(owner),
            offset: 0,
            len,
            _element: PhantomData,
        })
    }
}

/// Immutable device memory shared by handles and readable on every stream,
/// for weights and other constant data. Clones share the allocation.
pub struct DeviceTensor<T: DeviceCopy> {
    owner: Arc<Owner>,
    /// Byte offset into the allocation; several tensors may share one.
    offset: usize,
    len: usize,
    _element: PhantomData<T>,
}

impl<T: DeviceCopy> Clone for DeviceTensor<T> {
    fn clone(&self) -> Self {
        Self {
            owner: Arc::clone(&self.owner),
            offset: self.offset,
            len: self.len,
            _element: PhantomData,
        }
    }
}

impl<T: DeviceCopy> DeviceTensor<T> {
    /// `len` elements at byte `offset` of a shared allocation. The caller
    /// checked bounds and alignment.
    pub(crate) fn from_shared(owner: Arc<Owner>, offset: usize, len: usize) -> Self {
        debug_assert!(owner.allocation.shared);
        debug_assert!(offset + len * size_of::<T>() <= owner.allocation.bytes);
        Self {
            owner,
            offset,
            len,
            _element: PhantomData,
        }
    }

    pub(crate) fn owner(&self) -> &Arc<Owner> {
        &self.owner
    }

    /// Byte offset of the tensor within its allocation.
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn allocation(&self) -> &Arc<Allocation> {
        &self.owner.allocation
    }

    /// Read-only view for a launch on any stream.
    pub fn view(&self) -> DeviceSlice<'_, T> {
        DeviceSlice::at(&self.owner.allocation, self.offset, self.len)
    }

    pub fn slice(&self, range: Range<usize>) -> DeviceSlice<'_, T> {
        self.view().range(range)
    }

    /// The elements from `from` to the end, sharing the allocation, like the
    /// trailing rows of a stacked matrix; `None` past the end.
    pub fn tail(&self, from: usize) -> Option<DeviceTensor<T>> {
        (from <= self.len).then(|| DeviceTensor {
            owner: Arc::clone(&self.owner),
            offset: self.offset + from * size_of::<T>(),
            len: self.len - from,
            _element: PhantomData,
        })
    }

    /// The first `len` elements, sharing the allocation: the leading rows
    /// of a matrix; `None` past the end.
    pub fn head(&self, len: usize) -> Option<DeviceTensor<T>> {
        (len <= self.len).then(|| DeviceTensor {
            owner: Arc::clone(&self.owner),
            offset: self.offset,
            len,
            _element: PhantomData,
        })
    }

    /// One tensor over `self` and `next` when `next` starts right where
    /// `self` ends in the same allocation, like two adjacent GGUF tensors
    /// that a kernel reads as one matrix with more rows; `None` otherwise.
    pub fn join(&self, next: &DeviceTensor<T>) -> Option<DeviceTensor<T>> {
        let end = self
            .offset
            .checked_add(self.len.checked_mul(size_of::<T>())?)?;
        (Arc::ptr_eq(&self.owner, &next.owner) && end == next.offset).then(|| DeviceTensor {
            owner: Arc::clone(&self.owner),
            offset: self.offset,
            len: self.len + next.len,
            _element: PhantomData,
        })
    }
}

/// A buffer's bytes for copies on other streams while its owner keeps
/// using it: a borrowed snapshot names rows that stay in the owner's buffer
/// until the owner would overwrite them. The handle keeps the allocation
/// alive, the owner dropped or not; which bytes still hold the borrowed
/// values is the borrowing protocol's contract, as ordering between streams
/// is everywhere.
#[derive(Clone)]
pub struct SharedBytes {
    allocation: Arc<Allocation>,
    len: usize,
}

impl SharedBytes {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes `range` for [`Stream::copy_across`] and the other copies that
    /// read a buffer of another stream; panics out of range.
    pub fn slice(&self, range: Range<usize>) -> DeviceSlice<'_, u8> {
        DeviceSlice::whole(&self.allocation, self.len).range(range)
    }

    /// Whether this is the memory of `buffer`.
    pub fn is_buffer<T: DeviceCopy>(&self, buffer: &DeviceBuffer<T>) -> bool {
        Arc::ptr_eq(&self.allocation, buffer.allocation())
    }

    /// Whether both handles name the same memory.
    pub fn same_memory(&self, other: &SharedBytes) -> bool {
        Arc::ptr_eq(&self.allocation, &other.allocation)
    }
}

/// Shared view of device memory for one launch argument.
pub struct DeviceSlice<'a, T> {
    ptr: *const T,
    len: usize,
    allocation: &'a Arc<Allocation>,
    _borrow: PhantomData<&'a T>,
}

impl<T> Clone for DeviceSlice<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for DeviceSlice<'_, T> {}

/// Exclusive view of device memory for one launch argument.
pub struct DeviceSliceMut<'a, T> {
    ptr: *mut T,
    len: usize,
    allocation: &'a Arc<Allocation>,
    _borrow: PhantomData<&'a mut T>,
}

impl<'a, T> DeviceSlice<'a, T> {
    fn whole(allocation: &'a Arc<Allocation>, len: usize) -> Self {
        Self::at(allocation, 0, len)
    }

    fn at(allocation: &'a Arc<Allocation>, offset: usize, len: usize) -> Self {
        Self {
            // `wrapping_byte_add` keeps this safe code; the address is only
            // dereferenced on the device, within the allocation.
            ptr: allocation
                .ptr
                .as_ptr()
                .cast::<T>()
                .wrapping_byte_add(offset),
            len,
            allocation,
            _borrow: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The home stream, or `None` for an immutable tensor.
    pub fn home_stream(&self) -> Option<u64> {
        self.allocation.home_stream()
    }

    pub fn allocation(&self) -> &'a Arc<Allocation> {
        self.allocation
    }

    /// Device address; only meaningful to the device.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Device address as an integer, for recording.
    pub fn address(&self) -> usize {
        self.ptr as usize
    }

    pub fn range(self, range: Range<usize>) -> Self {
        assert!(
            range.start <= range.end && range.end <= self.len,
            "device slice out of range"
        );
        // `wrapping_add` keeps this safe code: the address is only dereferenced
        // on the device, inside the bounds checked above.
        Self {
            ptr: self.ptr.wrapping_add(range.start),
            len: range.end - range.start,
            ..self
        }
    }
}

/// Whether `len` values of `T` at `address` are whole, aligned `U` values.
fn castable<T, U>(address: usize, len: usize) -> Option<usize> {
    let bytes = len.checked_mul(size_of::<T>())?;
    (size_of::<U>() != 0
        && bytes.is_multiple_of(size_of::<U>())
        && address.is_multiple_of(align_of::<U>()))
    .then(|| bytes / size_of::<U>())
}

impl<'a, T: DeviceCopy> DeviceSlice<'a, T> {
    /// The same bytes as `U` values, for kernels that reuse a buffer under
    /// another element type (the C++ engine's `void*` scratch). Every
    /// `DeviceCopy` type is valid for any bit pattern, so only whole values and
    /// alignment are required; `None` otherwise.
    pub fn cast<U: DeviceCopy>(self) -> Option<DeviceSlice<'a, U>> {
        let len = castable::<T, U>(self.address(), self.len)?;
        Some(DeviceSlice {
            ptr: self.ptr.cast::<U>(),
            len,
            allocation: self.allocation,
            _borrow: PhantomData,
        })
    }
}

impl<'a, T: DeviceCopy> DeviceSliceMut<'a, T> {
    /// Like [`DeviceSlice::cast`], for a written buffer.
    pub fn cast<U: DeviceCopy>(self) -> Option<DeviceSliceMut<'a, U>> {
        let len = castable::<T, U>(self.address(), self.len)?;
        Some(DeviceSliceMut {
            ptr: self.ptr.cast::<U>(),
            len,
            allocation: self.allocation,
            _borrow: PhantomData,
        })
    }
}

impl<'a, T> DeviceSliceMut<'a, T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The home stream; mutable views always have one.
    pub fn home_stream(&self) -> u64 {
        self.allocation.home.id
    }

    pub fn allocation(&self) -> &'a Arc<Allocation> {
        self.allocation
    }

    /// Device address; only meaningful to the device.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Device address as an integer, for recording.
    pub fn address(&self) -> usize {
        self.ptr as usize
    }

    /// Shorter-lived exclusive view, to pass to a launch and keep the original.
    pub fn reborrow(&mut self) -> DeviceSliceMut<'_, T> {
        DeviceSliceMut {
            ptr: self.ptr,
            len: self.len,
            allocation: self.allocation,
            _borrow: PhantomData,
        }
    }

    pub fn as_const(&self) -> DeviceSlice<'_, T> {
        DeviceSlice {
            ptr: self.ptr,
            len: self.len,
            allocation: self.allocation,
            _borrow: PhantomData,
        }
    }

    pub fn range(self, range: Range<usize>) -> Self {
        assert!(
            range.start <= range.end && range.end <= self.len,
            "device slice out of range"
        );
        Self {
            ptr: self.ptr.wrapping_add(range.start),
            len: range.end - range.start,
            ..self
        }
    }

    /// Two disjoint exclusive views, like `<[T]>::split_at_mut`.
    pub fn split_at(self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.len, "device split out of range");
        let tail = Self {
            ptr: self.ptr.wrapping_add(mid),
            len: self.len - mid,
            allocation: self.allocation,
            _borrow: PhantomData,
        };
        (Self { len: mid, ..self }, tail)
    }
}

/// What an index buffer promises about each of its values `v`:
/// `v & mask < bound`, or `v == u32::MAX` (`-1` as `i32`, padding the
/// consuming kernels skip) when `padded`. When `signed`, every value but the
/// padding is also at most `i32::MAX`, so kernels that take `int32` indices
/// read it as non-negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexSpec {
    pub bound: u32,
    /// The bits the bound applies to: all of them, or for example the low
    /// 16 bits of a tile map that keeps an expert there and a tile index
    /// above.
    pub mask: u32,
    pub padded: bool,
    pub signed: bool,
}

impl IndexSpec {
    /// Every value below `bound`.
    pub fn below(bound: u32) -> Self {
        Self {
            bound,
            mask: u32::MAX,
            padded: false,
            signed: false,
        }
    }

    /// Also admits the padding value `u32::MAX`.
    pub fn padded(self) -> Self {
        Self {
            padded: true,
            ..self
        }
    }

    /// The bound applies to `value & mask` only.
    pub fn masked(self, mask: u32) -> Self {
        Self { mask, ..self }
    }

    /// Every value but the padding also fits a non-negative `i32`.
    pub fn signed(self) -> Self {
        Self {
            signed: true,
            ..self
        }
    }

    pub fn admits(&self, value: u32) -> bool {
        let valid = value & self.mask < self.bound && (!self.signed || value <= i32::MAX as u32);
        valid || (self.padded && value == u32::MAX)
    }

    /// Whether every value but the padding is a non-negative `i32`.
    pub fn fits_i32(&self) -> bool {
        self.signed || (self.mask == u32::MAX && self.bound <= 1 << 31)
    }
}

/// The source of kernel-written indices that are valid only together with
/// the indices they were computed from (expert counts of exactly these
/// expert IDs, for example): the source buffer's allocation, the device
/// address and length of the source view, the buffer's write generation at
/// the time, and a tag of the producing launch's geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Derivation {
    pub source: u64,
    pub address: usize,
    pub len: usize,
    pub generation: u64,
    pub tag: u64,
}

/// Kernel-written values of a [`DeviceIndices`] range and their source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Product {
    from: Derivation,
    address: usize,
    len: usize,
}

/// Device `u32` values that satisfy an [`IndexSpec`], for kernels that use
/// them to index a table (token IDs into an embedding table, expert IDs into
/// stacked expert weights, for example).
///
/// Only [`Stream::upload_indices`], which checks every value on the host
/// first, and kernels whose launch contract guarantees the spec
/// (`indices_out` in launch.toml, through [`DeviceIndices::view_mut`]) write
/// the buffer; allocation zero-initialises it. A kernel contract that
/// declares an `indices` parameter therefore knows the largest index it can
/// receive without reading device memory.
///
/// Every write increments the buffer's generation. A kernel-written range
/// may also record the indices it was derived from, so that a consumer of
/// exactly that range can require both to belong together.
pub struct DeviceIndices {
    buffer: DeviceBuffer<u32>,
    spec: IndexSpec,
    generation: u64,
    derived: Option<Product>,
}

impl DeviceIndices {
    pub(crate) fn new(buffer: DeviceBuffer<u32>, spec: IndexSpec) -> Self {
        Self {
            buffer,
            spec,
            generation: 0,
            derived: None,
        }
    }

    /// The buffer for a host upload, which invalidates any derivation.
    pub(crate) fn upload_target(&mut self) -> &mut DeviceBuffer<u32> {
        self.generation += 1;
        self.derived = None;
        &mut self.buffer
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Every value is below this (after the mask; see [`IndexSpec`]).
    pub fn bound(&self) -> u32 {
        self.spec.bound
    }

    pub fn spec(&self) -> IndexSpec {
        self.spec
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The source of the range the last kernel write derived from it.
    pub fn derived(&self) -> Option<Derivation> {
        self.derived.map(|product| product.from)
    }

    pub fn view(&self) -> IndexSlice<'_> {
        self.slice(0..self.len())
    }

    /// Sub-view; panics on an out-of-range request, like slice indexing.
    pub fn slice(&self, range: Range<usize>) -> IndexSlice<'_> {
        let values = self.buffer.slice(range);
        // Only the range the kernel wrote carries its derivation.
        let derived = self
            .derived
            .filter(|p| p.address == values.address() && p.len == values.len())
            .map(|p| p.from);
        IndexSlice {
            values,
            spec: self.spec,
            generation: self.generation,
            derived,
        }
    }

    /// A view that a kernel writes through, for launchers whose contract
    /// guarantees this buffer's spec (`indices_out`).
    pub fn view_mut(&mut self) -> IndexSliceMut<'_> {
        let len = self.len();
        self.slice_mut(0..len)
    }

    pub fn slice_mut(&mut self, range: Range<usize>) -> IndexSliceMut<'_> {
        IndexSliceMut {
            values: self.buffer.slice_mut(range),
            spec: self.spec,
            generation: &mut self.generation,
            derived: &mut self.derived,
        }
    }

    /// Records that the values in `range` were computed from `derivation`.
    ///
    /// # Safety
    /// Those values must be valid together with the source the derivation
    /// names (only generated extent probes use this, on probe values that
    /// launch.toml chooses to stay within every extent).
    #[doc(hidden)]
    pub unsafe fn assume_derived(&mut self, range: Range<usize>, derivation: Derivation) {
        let product = self.buffer.slice(range);
        self.derived = Some(Product {
            from: derivation,
            address: product.address(),
            len: product.len(),
        });
    }
}

/// Two ranges of device `u32` indices with their own [`IndexSpec`]s in one
/// allocation, so that one host upload writes both: a decode step's token ID
/// (below the vocabulary) and position (below the context), as the C++
/// executor copies them in one call. Only [`Stream::upload_index_pair_async`]
/// writes it, after checking each value against its range's spec;
/// allocation zero-initialises it.
pub struct DeviceIndexPair {
    buffer: DeviceBuffer<u32>,
    split: usize,
    specs: [IndexSpec; 2],
    generation: u64,
}

impl DeviceIndexPair {
    pub(crate) fn new(buffer: DeviceBuffer<u32>, split: usize, specs: [IndexSpec; 2]) -> Self {
        Self {
            buffer,
            split,
            specs,
            generation: 0,
        }
    }

    /// The buffer for a host upload.
    pub(crate) fn upload_target(&mut self) -> &mut DeviceBuffer<u32> {
        self.generation += 1;
        &mut self.buffer
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Where the second range starts.
    pub fn split(&self) -> usize {
        self.split
    }

    pub fn specs(&self) -> [IndexSpec; 2] {
        self.specs
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The first range, `0..split`.
    pub fn first(&self) -> IndexSlice<'_> {
        self.part(0..self.split, self.specs[0])
    }

    /// The second range, `split..len`.
    pub fn second(&self) -> IndexSlice<'_> {
        self.part(self.split..self.len(), self.specs[1])
    }

    fn part(&self, range: Range<usize>, spec: IndexSpec) -> IndexSlice<'_> {
        IndexSlice {
            values: self.buffer.slice(range),
            spec,
            generation: self.generation,
            derived: None,
        }
    }
}

/// Read-only view of [`DeviceIndices`] for one launch argument.
#[derive(Clone, Copy)]
pub struct IndexSlice<'a> {
    values: DeviceSlice<'a, u32>,
    spec: IndexSpec,
    generation: u64,
    derived: Option<Derivation>,
}

impl<'a> IndexSlice<'a> {
    pub fn values(&self) -> DeviceSlice<'a, u32> {
        self.values
    }

    pub fn bound(&self) -> u32 {
        self.spec.bound
    }

    pub fn spec(&self) -> IndexSpec {
        self.spec
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The derivation a kernel-written consumer records for these values.
    pub fn derivation(&self, tag: u64) -> Derivation {
        Derivation {
            source: self.values.allocation().id(),
            address: self.values.address(),
            len: self.values.len(),
            generation: self.generation,
            tag,
        }
    }

    pub fn derived(&self) -> Option<Derivation> {
        self.derived
    }
}

/// Written view of [`DeviceIndices`] for the output of a kernel whose launch
/// contract guarantees the buffer's spec. It hands out no device view, so
/// only generated launchers can write through it.
pub struct IndexSliceMut<'a> {
    values: DeviceSliceMut<'a, u32>,
    spec: IndexSpec,
    generation: &'a mut u64,
    derived: &'a mut Option<Product>,
}

impl IndexSliceMut<'_> {
    pub fn spec(&self) -> IndexSpec {
        self.spec
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn allocation(&self) -> &Arc<Allocation> {
        self.values.allocation()
    }

    pub fn address(&self) -> usize {
        self.values.address()
    }

    pub fn home_stream(&self) -> u64 {
        self.values.home_stream()
    }

    /// The device pointer, for the generated FFI call.
    pub fn as_mut_ptr(&mut self) -> *mut u32 {
        self.values.as_mut_ptr()
    }

    /// Marks the buffer as rewritten by the launch about to run, this range
    /// derived from `derived` or from nothing.
    pub(crate) fn commit(&mut self, derived: Option<Derivation>) {
        *self.generation += 1;
        *self.derived = derived.map(|from| Product {
            from,
            address: self.values.address(),
            len: self.values.len(),
        });
    }
}

/// Quantized weights: device bytes and their GGML storage format. Kernel
/// contracts size them from the format's block geometry (`gufo_formats`).
#[derive(Clone)]
pub struct QuantTensor {
    bytes: DeviceTensor<u8>,
    format: GgmlType,
}

impl QuantTensor {
    pub fn new(bytes: DeviceTensor<u8>, format: GgmlType) -> Self {
        Self { bytes, format }
    }

    pub fn format(&self) -> GgmlType {
        self.format
    }

    pub fn bytes(&self) -> &DeviceTensor<u8> {
        &self.bytes
    }

    /// `self` and `next` as one tensor of the same format when they are
    /// adjacent in one allocation ([`DeviceTensor::join`]).
    pub fn join(&self, next: &QuantTensor) -> Option<QuantTensor> {
        (self.format == next.format).then_some(())?;
        Some(Self {
            bytes: self.bytes.join(&next.bytes)?,
            format: self.format,
        })
    }

    pub fn view(&self) -> QuantSlice<'_> {
        QuantSlice {
            bytes: self.bytes.view(),
            format: self.format,
        }
    }

    /// The bytes from `from` on in the same format, for the trailing rows of
    /// a stacked matrix ([`DeviceTensor::tail`]).
    pub fn tail(&self, from: usize) -> Option<QuantTensor> {
        Some(Self {
            bytes: self.bytes.tail(from)?,
            format: self.format,
        })
    }

    /// The first `len` bytes in the same format, for the leading rows of a
    /// matrix ([`DeviceTensor::head`]).
    pub fn head(&self, len: usize) -> Option<QuantTensor> {
        Some(Self {
            bytes: self.bytes.head(len)?,
            format: self.format,
        })
    }
}

/// Read-only view of quantized weights for one launch argument.
#[derive(Clone, Copy)]
pub struct QuantSlice<'a> {
    bytes: DeviceSlice<'a, u8>,
    format: GgmlType,
}

impl<'a> QuantSlice<'a> {
    pub fn new(bytes: DeviceSlice<'a, u8>, format: GgmlType) -> Self {
        Self { bytes, format }
    }

    pub fn bytes(&self) -> DeviceSlice<'a, u8> {
        self.bytes
    }

    pub fn format(&self) -> GgmlType {
        self.format
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// Page-locked host memory for asynchronous copies.
///
/// The buffer remembers the event of its last asynchronous copy, and every
/// host access waits for it first. Host code therefore never observes a
/// partially transferred buffer and never overwrites one the DMA engine is
/// still reading — even if the caller forgets about the copy. When completion
/// cannot be confirmed (a failed synchronisation), the buffer stays locked for
/// good and is leaked instead of freed.
pub struct PinnedBuffer<T: DeviceCopy> {
    ptr: NonNull<T>,
    len: usize,
    transfer: RefCell<Transfer>,
}

enum Transfer {
    /// No copy in flight; the event of the last one, reused by the next.
    Idle(Option<Event>),
    InFlight(Event),
    /// Completion could not be confirmed; the DMA engine may still use the memory.
    Unknown(HipError),
}

// SAFETY: owned host allocation; the in-flight event is moved with it.
unsafe impl<T: DeviceCopy> Send for PinnedBuffer<T> {}

impl<T: DeviceCopy> PinnedBuffer<T> {
    /// Allocates `len` zeroed elements.
    pub fn zeroed(len: usize) -> Result<Self, HipError> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(HipError::refused("pinned allocation size overflow"))?;
        let mut raw = std::ptr::null_mut::<c_void>();
        // SAFETY: valid out-pointer.
        let status =
            unsafe { sys::hipHostMalloc(&mut raw, bytes.max(1), sys::HIP_HOST_MALLOC_DEFAULT) };
        check(status, "hipHostMalloc")?;
        let ptr = NonNull::new(raw.cast::<T>())
            .ok_or(HipError::refused("hipHostMalloc returned null"))?;
        // SAFETY: `bytes` were just allocated; all-zero is a valid `T`
        // (DeviceCopy types accept every bit pattern).
        unsafe { std::ptr::write_bytes(raw.cast::<u8>(), 0, bytes) };
        Ok(Self {
            ptr,
            len,
            transfer: RefCell::new(Transfer::Idle(None)),
        })
    }

    pub fn from_slice(values: &[T]) -> Result<Self, HipError> {
        let mut buffer = Self::zeroed(values.len())?;
        buffer.host_mut()?.copy_from_slice(values);
        Ok(buffer)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Host view, after the last asynchronous copy completed.
    pub fn host(&self) -> Result<&[T], HipError> {
        self.wait()?;
        // SAFETY: `len` initialised elements; no copy is in flight, and new
        // copies need `&mut self`, which cannot coexist with this borrow.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) })
    }

    /// Mutable host view, after the last asynchronous copy completed.
    pub fn host_mut(&mut self) -> Result<&mut [T], HipError> {
        self.wait()?;
        // SAFETY: as in `host`, and `&mut self` makes the borrow exclusive.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) })
    }

    /// Whether the last asynchronous copy has completed, without blocking.
    pub fn is_ready(&self) -> Result<bool, HipError> {
        match &*self.transfer.borrow() {
            Transfer::Idle(_) => Ok(true),
            Transfer::InFlight(event) => event.query(),
            Transfer::Unknown(cause) => Err(*cause),
        }
    }

    /// Blocks until the last asynchronous copy has completed. Fails for good
    /// once completion could not be confirmed.
    pub fn wait(&self) -> Result<(), HipError> {
        let mut transfer = self.transfer.borrow_mut();
        let result = match &*transfer {
            Transfer::Idle(_) => return Ok(()),
            Transfer::Unknown(cause) => return Err(*cause),
            Transfer::InFlight(event) => event.synchronize(),
        };
        let previous = std::mem::replace(&mut *transfer, Transfer::Idle(None));
        *transfer = match (result, previous) {
            (Ok(()), Transfer::InFlight(event)) => Transfer::Idle(Some(event)),
            (Ok(()), _) => Transfer::Idle(None),
            (Err(cause), _) => Transfer::Unknown(cause),
        };
        result
    }

    /// Waits for the last copy and returns the buffer's address with an
    /// event to track the next one (the last copy's, or a new one).
    pub(crate) fn begin_transfer(&mut self) -> Result<(*mut T, Event), HipError> {
        self.wait()?;
        let event = match self.transfer.get_mut() {
            Transfer::Idle(event) => event.take(),
            _ => None,
        };
        let event = match event {
            Some(event) => event,
            None => Event::new()?,
        };
        Ok((self.ptr.as_ptr(), event))
    }

    pub(crate) fn set_in_flight(&mut self, event: Event) {
        *self.transfer.get_mut() = Transfer::InFlight(event);
    }

    /// A copy may have been enqueued without a way to confirm its completion.
    pub(crate) fn set_unknown(&mut self, cause: HipError) {
        *self.transfer.get_mut() = Transfer::Unknown(cause);
    }
}

impl<T: DeviceCopy> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if self.wait().is_err() {
            // The DMA engine may still use this memory; leaking is the only
            // safe option.
            return;
        }
        // hipHostFree synchronises the device implicitly, so it waits until no
        // capture runs.
        deferred::defer(Deferred::Host {
            ptr: self.ptr.cast(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::Bf16;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(Bf16::from_f32(1.0).0, 0x3F80);
        // 1 + 2^-8 is halfway between two BF16 values: ties to even (down).
        assert_eq!(Bf16::from_f32(1.0 + 2f32.powi(-8)).0, 0x3F80);
        // 1 + 3 * 2^-8 is halfway as well: ties to even (up).
        assert_eq!(Bf16::from_f32(1.0 + 3.0 * 2f32.powi(-8)).0, 0x3F82);
        assert_eq!(Bf16(0xC0A0).to_f32(), -5.0);
    }
}
