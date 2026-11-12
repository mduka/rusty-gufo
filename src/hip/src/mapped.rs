//! Read-only file mappings for weights.

use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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

    /// Drops this process's resident pages of the mapping once its contents
    /// live elsewhere, uploaded and synchronized (`ReleaseMappedPages`). The
    /// mapping stays valid: a later read refaults the page from the file.
    /// Best effort; a mapping registered with HIP, which reads it as weight
    /// memory, keeps its pages.
    pub fn release_pages(&self) {
        if self
            .registration
            .lock()
            .is_ok_and(|registration| registration.strong_count() != 0)
        {
            return;
        }
        // SAFETY: the whole private read-only mapping, page-aligned by mmap;
        // dropping its page-table entries leaves every byte readable.
        unsafe {
            let _ = libc::madvise(self.ptr.as_ptr().cast(), self.len, libc::MADV_DONTNEED);
        }
    }

    /// Visits the whole mapping in `chunk_bytes` pieces (a multiple of the
    /// page size) from up to `workers` threads (`VisitMappedChunks`). Each
    /// chunk is populated from the page cache or disk before `visit(worker,
    /// offset, chunk)` runs and its page-table entries are dropped
    /// afterwards; the page cache keeps the file data. A `false` from
    /// `visit` stops the remaining chunks.
    pub fn visit_chunks(
        &self,
        chunk_bytes: usize,
        workers: usize,
        visit: impl Fn(usize, usize, &[u8]) -> bool + Sync,
    ) -> io::Result<()> {
        if chunk_bytes == 0 || !chunk_bytes.is_multiple_of(4096) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "chunk size must be a positive multiple of the page size",
            ));
        }
        let chunks = self.len.div_ceil(chunk_bytes);
        let next = AtomicUsize::new(0);
        let stopped = AtomicBool::new(false);
        let failure = Mutex::new(None);
        let visit_chunks = |worker: usize| {
            while !stopped.load(Ordering::Relaxed) {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= chunks {
                    break;
                }
                let offset = index * chunk_bytes;
                let size = chunk_bytes.min(self.len - offset);
                let address = self.ptr.as_ptr().wrapping_add(offset).cast::<c_void>();
                // SAFETY: a page-aligned range inside the mapping.
                if unsafe { libc::madvise(address, size, libc::MADV_POPULATE_READ) } != 0 {
                    if let Ok(mut slot) = failure.lock() {
                        slot.get_or_insert(io::Error::last_os_error());
                    }
                    stopped.store(true, Ordering::Relaxed);
                    break;
                }
                let keep_going = visit(worker, offset, &self.bytes()[offset..offset + size]);
                // The visitor owns a copy of these bytes now. Keep the
                // mapping valid for the reader without retaining its
                // populated page-table entries.
                // SAFETY: as above; the range stays readable.
                unsafe {
                    let _ = libc::madvise(address, size, libc::MADV_DONTNEED);
                }
                if !keep_going {
                    stopped.store(true, Ordering::Relaxed);
                    break;
                }
            }
        };
        std::thread::scope(|scope| {
            for worker in 1..workers.clamp(1, chunks.max(1)) {
                let visit_chunks = &visit_chunks;
                scope.spawn(move || visit_chunks(worker));
            }
            visit_chunks(0);
        });
        match failure.into_inner().ok().flatten() {
            Some(error) => Err(error),
            None => Ok(()),
        }
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

/// Host-only checks of the mapping (`tests/core/platform_prefault_test.cpp`).
#[cfg(test)]
mod host_tests {
    use super::*;

    /// This process's resident kilobytes of the mapping holding `address`
    /// (`ResidentKiB`): `mincore` would report page-cache residency, which
    /// `MADV_DONTNEED` deliberately preserves.
    fn resident_kib(address: *const u8) -> usize {
        let smaps = std::fs::read_to_string("/proc/self/smaps").expect("smaps");
        let address = address as usize;
        let mut inside = false;
        for line in smaps.lines() {
            let range = line
                .split_whitespace()
                .next()
                .and_then(|r| r.split_once('-'));
            if let Some((begin, end)) = range
                && let (Ok(begin), Ok(end)) = (
                    usize::from_str_radix(begin, 16),
                    usize::from_str_radix(end, 16),
                )
            {
                inside = (begin..end).contains(&address);
            } else if inside && let Some(rss) = line.strip_prefix("Rss:") {
                return rss
                    .trim()
                    .trim_end_matches("kB")
                    .trim()
                    .parse()
                    .expect("Rss");
            }
        }
        panic!("mapping missing from smaps");
    }

    /// Every byte arrives once, at its offset, from a worker below the
    /// limit, and a `false` stops the remaining chunks
    /// (`tests/core/platform_pages_test.cpp`).
    #[test]
    fn chunk_visits_cover_the_file_once() {
        // An odd size leaves a short last chunk.
        let len = (5 << 20) + 12345;
        let bytes: Vec<u8> = (0..len).map(|i| ((i * 131) ^ (i >> 9)) as u8).collect();
        let path = std::env::temp_dir().join(format!("gufo-visit-{}.bin", std::process::id()));
        std::fs::write(&path, &bytes).expect("write");
        let file = MappedFile::open(&path).expect("map");
        let (chunk, workers) = (1 << 20, 3);
        let copy = Mutex::new(vec![0u8; len]);
        let seen = Mutex::new(vec![0; len / chunk + 1]);
        file.visit_chunks(chunk, workers, |worker, offset, data| {
            assert!(worker < workers);
            assert_eq!(offset % chunk, 0);
            assert_eq!(data.len(), chunk.min(len - offset));
            copy.lock().expect("copy")[offset..offset + data.len()].copy_from_slice(data);
            seen.lock().expect("seen")[offset / chunk] += 1;
            true
        })
        .expect("visit");
        assert!(copy.into_inner().expect("copy") == bytes);
        assert!(
            seen.into_inner()
                .expect("seen")
                .iter()
                .all(|&count| count == 1)
        );
        let visited = AtomicUsize::new(0);
        file.visit_chunks(chunk, 1, |_, _, _| {
            visited.fetch_add(1, Ordering::Relaxed);
            false
        })
        .expect("visit");
        assert_eq!(visited.into_inner(), 1, "a false return stops the visit");
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn released_pages_leave_the_process_and_refault() {
        let path = std::env::temp_dir().join(format!("gufo-release-{}.bin", std::process::id()));
        let bytes: Vec<u8> = (0..4 * 4096).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &bytes).expect("write");
        let file = MappedFile::open(&path).expect("map");
        file.prefault(0..bytes.len()).expect("prefault");
        let base = file.bytes().as_ptr();
        assert_eq!(resident_kib(base), bytes.len() / 1024);
        file.release_pages();
        assert_eq!(resident_kib(base), 0);
        assert_eq!(file.bytes(), bytes.as_slice(), "the mapping stays readable");
        assert_eq!(resident_kib(base), bytes.len() / 1024);
        drop(file);
        let _ = std::fs::remove_file(&path);
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
