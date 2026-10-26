//! Memory the host and the device both address, beside managed memory
//! ([`Stream::alloc_managed`]): page-locked host memory and shared memory
//! files registered with HIP, the host-visible paths `gufo diagnose
//! --benchmark allocation` measures. [`HostMemory::into_device`] turns such
//! memory into a [`DeviceBuffer`] through its device alias; the buffer's host
//! views wait for the device ([`DeviceBuffer::host`]).
//!
//! Releasing host memory may synchronise the device (`hipHostFree`,
//! `hipHostUnregister`), so it goes through the free queue like exact
//! device memory.

use std::ffi::{CString, c_void};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;

use crate::deferred::{self, Deferred};
use crate::error::{HipError, check};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::sys;

/// Resident pages of a host range (`mincore`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Residency {
    pub resident: usize,
    pub pages: usize,
}

/// The host resources behind host-visible memory, released together.
pub(crate) struct HostRelease {
    host: NonNull<u8>,
    bytes: usize,
    kind: HostKind,
}

enum HostKind {
    /// `hipHostMalloc`.
    Pinned,
    /// A shared mapping of a memory file, `registered` with `hipHostRegister`.
    Memfd { fd: OwnedFd, registered: bool },
}

// SAFETY: an address, a length and a file descriptor, released once by
// whichever thread holds them.
unsafe impl Send for HostRelease {}

impl HostRelease {
    pub(crate) fn host(&self) -> NonNull<u8> {
        self.host
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    /// Frees the memory. The caller made sure that no device work uses it
    /// (the free queue synchronises the device first).
    pub(crate) fn release(self) {
        match self.kind {
            HostKind::Pinned => {
                // SAFETY: allocated by hipHostMalloc and freed once, here.
                unsafe {
                    let _ = sys::hipHostFree(self.host.as_ptr().cast());
                }
            }
            HostKind::Memfd { fd, registered } => {
                if registered {
                    // SAFETY: registered once by `HostMemory::register`.
                    unsafe {
                        let _ = sys::hipHostUnregister(self.host.as_ptr().cast());
                    }
                }
                // SAFETY: the mapping of `bytes` that `HostMemory::memfd`
                // created, unmapped once; no view into it outlives its owner.
                unsafe {
                    libc::munmap(self.host.as_ptr().cast(), self.bytes);
                }
                drop(fd);
            }
        }
    }
}

/// Resident pages of `bytes` at `host` (`mincore`), with the messages of
/// the C++ diagnostic on failure.
pub(crate) fn residency(host: NonNull<u8>, bytes: usize) -> Result<Residency, String> {
    if bytes == 0 {
        return Err("mincore unavailable: no host mapping".into());
    }
    // SAFETY: sysconf has no preconditions.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err("mincore unavailable: page size query failed".into());
    }
    let page = page_size as usize;
    let address = host.as_ptr() as usize;
    let aligned = address & !(page - 1);
    let prefix = address - aligned;
    let Some(span) = bytes.checked_add(prefix) else {
        return Err("mincore unavailable: range overflow".into());
    };
    let pages = span.div_ceil(page);
    const CHUNK_PAGES: usize = 1 << 20;
    let mut states = vec![0u8; pages.min(CHUNK_PAGES)];
    let (mut resident, mut completed) = (0, 0);
    while completed < pages {
        let count = states.len().min(pages - completed);
        let chunk = (aligned + completed * page) as *mut c_void;
        // SAFETY: `states` holds `count` entries for `count` pages of a
        // mapping the caller keeps alive; mincore only reports on them.
        if unsafe { libc::mincore(chunk, count * page, states.as_mut_ptr()) } != 0 {
            return Err("mincore unavailable for this allocation".into());
        }
        resident += states[..count].iter().filter(|&&s| s & 1 != 0).count();
        completed += count;
    }
    Ok(Residency { resident, pages })
}

/// Host memory the device can address once it has a device alias:
/// page-locked memory or a shared memory file.
pub struct HostMemory {
    /// Always present until the memory is released or becomes a buffer.
    release: Option<HostRelease>,
}

impl HostMemory {
    /// Page-locked host memory (`hipHostMalloc`), mapped into the device
    /// when `mapped` (`hipHostMallocMapped`).
    pub fn pinned(bytes: usize, mapped: bool) -> Result<Self, HipError> {
        let flags = if mapped {
            sys::HIP_HOST_MALLOC_MAPPED
        } else {
            sys::HIP_HOST_MALLOC_DEFAULT
        };
        // Allocation may synchronise the device: wait until no capture runs.
        let _exclusive = deferred::exclusive("hipHostMalloc")?;
        let mut raw = std::ptr::null_mut::<c_void>();
        // SAFETY: valid out-pointer.
        let status = unsafe { sys::hipHostMalloc(&mut raw, bytes.max(1), flags) };
        check(status, "hipHostMalloc")?;
        let host = NonNull::new(raw.cast::<u8>())
            .ok_or(HipError::refused("hipHostMalloc returned null"))?;
        Ok(Self {
            release: Some(HostRelease {
                host,
                bytes,
                kind: HostKind::Pinned,
            }),
        })
    }

    /// An unnamed memory file of `bytes` (`memfd_create`, `ftruncate`),
    /// mapped shared and read-write; the host touches no page of it.
    pub fn memfd(name: &str, bytes: usize) -> io::Result<Self> {
        let name = CString::new(name).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let length = libc::off_t::try_from(bytes)
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        // SAFETY: a NUL-terminated name and valid flags.
        let raw = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
        if raw < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: a fresh descriptor that nothing else owns.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        // SAFETY: a valid descriptor.
        if unsafe { libc::ftruncate(fd.as_raw_fd(), length) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: a new shared mapping of the whole file at an address the
        // kernel chooses; nothing else maps it.
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if mapping == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let host = NonNull::new(mapping.cast::<u8>())
            .ok_or_else(|| io::Error::other("mmap returned null"))?;
        Ok(Self {
            release: Some(HostRelease {
                host,
                bytes,
                kind: HostKind::Memfd {
                    fd,
                    registered: false,
                },
            }),
        })
    }

    fn parts(&self) -> &HostRelease {
        self.release
            .as_ref()
            .expect("host memory is present until it is consumed")
    }

    pub fn len(&self) -> usize {
        self.parts().bytes
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Registers a memory file with HIP (`hipHostRegister`, mapped);
    /// page-locked memory is registered already.
    pub fn register(&mut self) -> Result<(), HipError> {
        let parts = self
            .release
            .as_mut()
            .expect("host memory is present until it is consumed");
        let HostKind::Memfd { registered, .. } = &mut parts.kind else {
            return Err(HipError::refused(
                "page-locked memory is registered already",
            ));
        };
        if *registered {
            return Err(HipError::refused("memory file is registered already"));
        }
        let _exclusive = deferred::exclusive("hipHostRegister")?;
        // SAFETY: a live mapping of `bytes`.
        let status = unsafe {
            sys::hipHostRegister(
                parts.host.as_ptr().cast(),
                parts.bytes,
                sys::HIP_HOST_REGISTER_MAPPED,
            )
        };
        check(status, "hipHostRegister")?;
        *registered = true;
        Ok(())
    }

    /// The memory as host bytes; no device work can use it yet.
    pub fn as_slice(&self) -> &[u8] {
        let parts = self.parts();
        // SAFETY: `bytes` of live host memory without a device alias, so no
        // device work reads or writes it.
        unsafe { std::slice::from_raw_parts(parts.host.as_ptr(), parts.bytes) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let parts = self.parts();
        // SAFETY: as in `as_slice`, and `&mut self` makes the view unique.
        unsafe { std::slice::from_raw_parts_mut(parts.host.as_ptr(), parts.bytes) }
    }

    /// Resident pages of the mapping.
    pub fn residency(&self) -> Result<Residency, String> {
        let parts = self.parts();
        residency(parts.host, parts.bytes)
    }

    /// The device alias (`hipHostGetDevicePointer`) as a buffer of
    /// `stream`; on failure the memory comes back with the error.
    pub fn into_device(mut self, stream: &Stream) -> Result<DeviceBuffer<u8>, (Self, HipError)> {
        let parts = self.parts();
        let mut device = std::ptr::null_mut::<c_void>();
        // SAFETY: valid out-pointer and host memory allocated or registered
        // for the device.
        let status =
            unsafe { sys::hipHostGetDevicePointer(&mut device, parts.host.as_ptr().cast(), 0) };
        if let Err(error) = check(status, "hipHostGetDevicePointer") {
            return Err((self, error));
        }
        let Some(device) = NonNull::new(device) else {
            return Err((
                self,
                HipError::refused("hipHostGetDevicePointer returned null"),
            ));
        };
        let release = self
            .release
            .take()
            .expect("host memory is present until it is consumed");
        Ok(DeviceBuffer::<u8>::host_visible(
            stream.inner(),
            device,
            release,
        ))
    }
}

impl std::fmt::Debug for HostMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let parts = self.parts();
        let kind = match parts.kind {
            HostKind::Pinned => "pinned",
            HostKind::Memfd {
                registered: true, ..
            } => "registered memory file",
            HostKind::Memfd { .. } => "memory file",
        };
        f.debug_struct("HostMemory")
            .field("kind", &kind)
            .field("bytes", &parts.bytes)
            .finish()
    }
}

impl Drop for HostMemory {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            deferred::defer(Deferred::HostMemory(release));
        }
    }
}
