//! Read-only file mappings for weights.

use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crate::deferred;
use crate::error::{HipError, check};
use crate::memory::Owner;
use crate::stream::Stream;
use crate::sys;

const PREFAULT_CHUNK: usize = 16 << 20;
const PREFAULT_READERS: usize = 16;

/// Advises the kernel that `len` bytes at `offset` of `file` are read at
/// random (`posix_fadvise`, `POSIX_FADV_RANDOM`): a page cache miss then
/// reads its page, not a readahead window.
pub fn advise_random_access(file: &File, offset: u64, len: u64) -> io::Result<()> {
    let invalid = || io::Error::from(io::ErrorKind::InvalidInput);
    let offset = libc::off_t::try_from(offset).map_err(|_| invalid())?;
    let len = libc::off_t::try_from(len).map_err(|_| invalid())?;
    // SAFETY: a live descriptor; the advice only changes readahead.
    match unsafe { libc::posix_fadvise(file.as_raw_fd(), offset, len, libc::POSIX_FADV_RANDOM) } {
        0 => Ok(()),
        error => Err(io::Error::from_raw_os_error(error)),
    }
}

/// A zeroed host buffer that transparent huge pages may back: a device copy
/// of hundreds of MiB into it faults once per 2 MiB page instead of once per
/// 4 KiB page, which under concurrent device work costs several times the
/// copy itself. The advice is a hint the kernel may refuse; it pins nothing
/// and leaves the contents zero.
pub fn huge_page_buffer(len: usize) -> Vec<u8> {
    // Large zeroed allocations come from fresh anonymous mappings, so the
    // advice precedes the first touch.
    let mut buffer = vec![0u8; len];
    // SAFETY: sysconf has no preconditions.
    let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap_or(0);
    if page > 0 {
        let address = buffer.as_mut_ptr() as usize;
        let skip = (page - address % page) % page;
        let whole = len.saturating_sub(skip) / page * page;
        if whole > 0 {
            // SAFETY: `[address + skip, + whole)` lies inside `buffer`'s
            // allocation; the advice changes only how pages are backed.
            unsafe {
                let _ = libc::madvise(
                    buffer.as_mut_ptr().add(skip).cast(),
                    whole,
                    libc::MADV_HUGEPAGE,
                );
            }
        }
    }
    buffer
}

/// A weight file mapped read-only, or an anonymous copy of one
/// ([`MappedFile::host_copy`]).
///
/// Weight files are treated as immutable while mapped: another process that
/// truncates or rewrites the file would change or remove the mapped pages
/// (SIGBUS on truncation), as with the C++ loaders.
pub struct MappedFile {
    ptr: NonNull<u8>,
    len: usize,
    file: File,
    registration: Mutex<Weak<Owner>>,
}

// SAFETY: the mapping is read-only and owned by this value; sharing its
// address across threads is fine.
unsafe impl Send for MappedFile {}
// SAFETY: as above; `registration` is behind a mutex.
unsafe impl Sync for MappedFile {}

impl MappedFile {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Arc<Self>> {
        let file = File::open(path.as_ref())?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "file too large"))?;
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "empty weight file",
            ));
        }
        // SAFETY: a fresh private read-only mapping of an open descriptor.
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let ptr = NonNull::new(raw.cast::<u8>()).ok_or_else(io::Error::last_os_error)?;
        Ok(Arc::new(Self {
            ptr,
            len,
            file,
            registration: Mutex::new(Weak::new()),
        }))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The mapped bytes, for parsing headers.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: `len` readable bytes mapped for the lifetime of `self`;
        // weight files are immutable while mapped (see the type docs).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub(crate) fn file(&self) -> &File {
        &self.file
    }

    /// Faults `range` into memory with parallel readers, which exposes the
    /// disk's queue depth (`MADV_POPULATE_READ`, falling back to a hint).
    pub fn prefault(&self, range: Range<usize>) -> io::Result<()> {
        if range.start > range.end || range.end > self.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "prefault range outside the file",
            ));
        }
        let page = 4096;
        let begin = range.start / page * page;
        let bytes = range.end - begin;
        let chunks = bytes.div_ceil(PREFAULT_CHUNK);
        let next = AtomicUsize::new(0);
        let failure = Mutex::new(None);
        let populate = || {
            loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= chunks {
                    break;
                }
                let offset = begin + index * PREFAULT_CHUNK;
                let len = PREFAULT_CHUNK.min(range.end - offset);
                let address = self.ptr.as_ptr().wrapping_add(offset).cast::<c_void>();
                // SAFETY: a page-aligned range inside the mapping.
                let result = unsafe { libc::madvise(address, len, libc::MADV_POPULATE_READ) };
                if result != 0 {
                    let error = io::Error::last_os_error();
                    // Older kernels lack MADV_POPULATE_READ: a hint still helps.
                    if error.raw_os_error() == Some(libc::EINVAL) {
                        // SAFETY: as above.
                        unsafe {
                            let _ = libc::madvise(address, len, libc::MADV_WILLNEED);
                        }
                    } else if let Ok(mut slot) = failure.lock() {
                        slot.get_or_insert(error);
                    }
                }
            }
        };
        std::thread::scope(|scope| {
            for _ in 1..PREFAULT_READERS.min(chunks) {
                scope.spawn(populate);
            }
            populate();
        });
        match failure.into_inner().ok().flatten() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// An anonymous private copy of the whole file in transparent huge pages,
    /// as the C++ Qwen text loader makes before registering its weights: the
    /// device then reads them through 2 MiB pages instead of the page cache's
    /// 4 KiB ones. Parallel workers populate, copy and release the file pages
    /// in 16 MiB chunks, so the source mapping keeps no resident page table.
    pub fn host_copy(&self) -> io::Result<Arc<Self>> {
        // SAFETY: a fresh anonymous mapping.
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                self.len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let ptr = NonNull::new(raw.cast::<u8>()).ok_or_else(io::Error::last_os_error)?;
        // Owns the mapping from here on, so every error path unmaps it.
        let copy = Self {
            ptr,
            len: self.len,
            file: self.file.try_clone()?,
            registration: Mutex::new(Weak::new()),
        };
        // SAFETY: the whole new mapping; a hint that may be refused.
        unsafe {
            let _ = libc::madvise(raw, self.len, libc::MADV_HUGEPAGE);
        }
        let chunks = self.len.div_ceil(PREFAULT_CHUNK);
        let next = AtomicUsize::new(0);
        let failure = Mutex::new(None);
        let (source, destination) = (self.ptr.as_ptr() as usize, ptr.as_ptr() as usize);
        let work = || {
            while failure.lock().map(|slot| slot.is_none()).unwrap_or(false) {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= chunks {
                    break;
                }
                let offset = index * PREFAULT_CHUNK;
                let len = PREFAULT_CHUNK.min(self.len - offset);
                let input = (source + offset) as *mut c_void;
                // SAFETY: a page-aligned range inside the source mapping.
                if unsafe { libc::madvise(input, len, libc::MADV_POPULATE_READ) } != 0 {
                    let error = io::Error::last_os_error();
                    // Without MADV_POPULATE_READ the copy faults the pages in.
                    if error.raw_os_error() != Some(libc::EINVAL) {
                        if let Ok(mut slot) = failure.lock() {
                            slot.get_or_insert(error);
                        }
                        break;
                    }
                }
                // SAFETY: `len` bytes inside both mappings, which do not
                // overlap; each chunk is written by one worker only.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        input.cast::<u8>().cast_const(),
                        (destination + offset) as *mut u8,
                        len,
                    );
                    // The copy owns the bytes now: drop the source's page
                    // table entries (the page cache keeps the file).
                    let _ = libc::madvise(input, len, libc::MADV_DONTNEED);
                }
            }
        };
        std::thread::scope(|scope| {
            for _ in 1..PREFAULT_READERS.min(chunks) {
                scope.spawn(work);
            }
            work();
        });
        match failure.into_inner().ok().flatten() {
            Some(error) => Err(error),
            None => Ok(Arc::new(copy)),
        }
    }

    /// The whole mapping registered with HIP as read-only, mapped host memory:
    /// one registration per file, shared while any tensor uses it.
    pub(crate) fn register(self: &Arc<Self>, stream: &Stream) -> Result<Arc<Owner>, HipError> {
        let mut registration = self
            .registration
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(owner) = registration.upgrade() {
            return Ok(owner);
        }
        // Registration may synchronise the device: wait until no capture runs.
        // This also undoes a previous registration of this file that still
        // waits in the queue; `collect` alone would skip it while another
        // thread holds the gate, and HIP refuses a second registration
        // (hipErrorHostMemoryAlreadyRegistered).
        let _exclusive = deferred::exclusive("hipHostRegister")?;
        let host = self.ptr.as_ptr().cast::<c_void>();
        // SAFETY: the mapping lives as long as the registration (the owner
        // holds an Arc to `self`); read-only mapped registration.
        let status = unsafe {
            sys::hipHostRegister(
                host,
                self.len,
                sys::HIP_HOST_REGISTER_MAPPED | sys::HIP_HOST_REGISTER_READ_ONLY,
            )
        };
        check(status, "hipHostRegister")?;
        let mut device = std::ptr::null_mut::<c_void>();
        // SAFETY: `host` was registered above; valid out-pointer.
        let status = unsafe { sys::hipHostGetDevicePointer(&mut device, host, 0) };
        let device = check(status, "hipHostGetDevicePointer").and_then(|()| {
            NonNull::new(device).ok_or(HipError::refused("hipHostGetDevicePointer returned null"))
        });
        let device = match device {
            Ok(device) => device,
            Err(error) => {
                // SAFETY: registered above and not used.
                unsafe {
                    let _ = sys::hipHostUnregister(host);
                }
                return Err(error);
            }
        };
        let owner = Arc::new(Owner::mapped(stream.inner(), device, Arc::clone(self)));
        *registration = Arc::downgrade(&owner);
        Ok(owner)
    }

    pub(crate) fn host_address(&self) -> *mut c_void {
        self.ptr.as_ptr().cast()
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        // SAFETY: mapped in `open` with this length; no registration holds it
        // any more (registrations keep an Arc to the file).
        unsafe {
            let _ = libc::munmap(self.ptr.as_ptr().cast(), self.len);
        }
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::unwrap_used)]
    fn a_queued_registration_is_undone_before_the_file_registers_again() {
        let path = std::env::temp_dir().join(format!("gufo-requeue-{}.bin", std::process::id()));
        std::fs::write(&path, vec![7u8; 1 << 16]).unwrap();
        let file = MappedFile::open(&path).unwrap();
        let stream = Stream::new().unwrap();
        let owner = file.register(&stream).unwrap();
        // A capture elsewhere holds the gate, so the release is only queued.
        let capture = deferred::capture_guard();
        drop(owner);
        let registering = std::thread::spawn({
            let file = Arc::clone(&file);
            move || file.register(&Stream::new()?).map(|_| ())
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        // The capture ends without collecting, as when its own collect loses
        // the race against the waiting registration.
        drop(capture);
        assert_eq!(registering.join().unwrap(), Ok(()));
        let _ = std::fs::remove_file(&path);
    }
}
