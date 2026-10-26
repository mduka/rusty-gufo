//! Loading weights into immutable device tensors.
//!
//! ```text
//! let file = MappedFile::open(path)?;
//! let mut loader = WeightLoader::new(&stream, Placement::DirectCopy);
//! let q_proj = loader.add::<Bf16>(&file, offset, len)?;   // Pending<Bf16>
//! let weights = loader.finish()?;                         // waits for every copy
//! let q_proj: DeviceTensor<Bf16> = weights.get(q_proj);
//! ```
//!
//! `Placement::DirectCopy` uploads the tensors into device arenas that are
//! allocated while earlier ones fill: parallel workers read the file with
//! `pread` (`O_DIRECT` where the filesystem allows it, so the page cache is
//! not filled twice) into pinned staging buffers and copy each chunk on their
//! own stream, like the C++ `WeightUpload`.
//! `Placement::CachedCopy` registers the prefaulted mapping and lets the device
//! copy from it into arenas that are allocated while earlier copies run,
//! which is fastest when the page cache holds the file.
//! `Placement::Map` prefaults the file and registers the mapping once; tensors
//! are offsets into it (zero copy on the unified memory of gfx1151).
//! `Placement::HostCopy` registers an anonymous huge-page copy of each file
//! instead, like the C++ Qwen text loader.
//! Tensors exist only after `finish`, so no stream can read a weight that is
//! still being copied.
//!
//! With a copy placement a tensor may also join several file ranges
//! ([`WeightLoader::add_parts`]: projections of one input stacked into one
//! matrix) and end in zero bytes that kernels reading whole blocks past the
//! last row may touch.

use std::fs::{File, OpenOptions};
use std::io;
use std::marker::PhantomData;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::ScopedJoinHandle;

use crate::error::{HipError, check};
use crate::mapped::MappedFile;
use crate::memory::{DeviceBuffer, DeviceCopy, DeviceTensor, Owner, PinnedBuffer};
use crate::stream::Stream;
use crate::sys;

/// Device offset alignment of tensors in the copy arenas.
const ARENA_ALIGNMENT: usize = 256;
/// Largest arena of a cached copy (larger tensors get their own). hipMalloc
/// rounds every allocation of 1 MB and more up to 2 MiB, so fewer, larger
/// arenas waste less; the copy of the last one is not overlapped.
const CACHED_ARENA: usize = 1 << 30;
/// Size limit of the last arena of a cached copy: allocating it (~40 ms per
/// GB) must cover the copy of the arena before (~12 ms per GB).
const CACHED_TAIL: usize = 256 << 20;
/// Largest arena of a direct copy (larger tensors get their own).
const DIRECT_ARENA: usize = 1 << 30;
/// Arenas a direct copy allocates ahead of the one its workers fill:
/// allocating everything first contends with the copies (Flash-Next,
/// 79 GiB: 23.5 s against 21.7 s).
const ARENAS_AHEAD: usize = 2;
const CHUNK: usize = 16 << 20;
const DIRECT_ALIGNMENT: usize = 4096;
const UPLOADERS: usize = 16;

static NEXT_LOADER_ID: AtomicU64 = AtomicU64::new(1);

/// Where loaded weights live and how they get there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Device memory, read with `O_DIRECT`: bypasses the page cache, so a
    /// large file loaded once is not held twice in unified memory. Limited by
    /// the disk. Tensors are aligned to 256 bytes.
    DirectCopy,
    /// Device memory, copied by the device from the registered, prefaulted
    /// mapping: fast when the file is already in the page cache, which then
    /// keeps it. Tensors are aligned to 256 bytes.
    CachedCopy,
    /// The registered file mapping; tensors keep their file offsets, which
    /// must be aligned to their element type.
    Map,
    /// A registered anonymous copy of each file in transparent huge pages
    /// ([`MappedFile::host_copy`]); tensors keep their file offsets, which
    /// must be aligned to their element type. The C++ Qwen text engine reads
    /// its weights this way.
    HostCopy,
}

/// Reads every page of `files` into the page cache and maps it.
fn prefault(files: &[&Arc<MappedFile>]) -> Result<(), HipError> {
    for file in files {
        file.prefault(0..file.len())
            .map_err(|_| HipError::refused("cannot read the mapped weight file"))?;
    }
    Ok(())
}

/// One file range of a tensor, at its offset in the copy arena.
struct Entry {
    file: Arc<MappedFile>,
    file_offset: usize,
    bytes: usize,
    arena_offset: usize,
}

/// A queued tensor: `parts` consecutive entries from `first`, then `tail`
/// zero bytes.
struct Queued {
    first: usize,
    parts: usize,
    bytes: usize,
    tail: usize,
}

impl Queued {
    fn total(&self) -> usize {
        self.bytes + self.tail
    }
}

/// A tensor queued in a [`WeightLoader`]; redeemed with [`LoadedWeights::get`].
#[must_use]
pub struct Pending<T> {
    loader: u64,
    index: usize,
    len: usize,
    _element: PhantomData<fn() -> T>,
}

/// Collects tensors from mapped files and places them on the device.
pub struct WeightLoader<'s> {
    stream: &'s Stream,
    placement: Placement,
    /// The copy placements allocate their arenas in coarse-grain managed
    /// memory instead of the device heap.
    managed: bool,
    id: u64,
    entries: Vec<Entry>,
    queued: Vec<Queued>,
    arena_bytes: usize,
}

/// The tensors of a finished [`WeightLoader`].
pub struct LoadedWeights {
    loader: u64,
    tensors: Vec<(Arc<Owner>, usize)>,
}

impl LoadedWeights {
    /// The tensor for `pending`. Panics for a handle of another loader.
    pub fn get<T: DeviceCopy>(&self, pending: Pending<T>) -> DeviceTensor<T> {
        assert_eq!(
            pending.loader, self.loader,
            "tensor handle from another WeightLoader"
        );
        let (owner, offset) = &self.tensors[pending.index];
        DeviceTensor::from_shared(Arc::clone(owner), *offset, pending.len)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl<'s> WeightLoader<'s> {
    pub fn new(stream: &'s Stream, placement: Placement) -> Self {
        Self {
            stream,
            placement,
            managed: false,
            id: NEXT_LOADER_ID.fetch_add(1, Ordering::Relaxed),
            entries: Vec::new(),
            queued: Vec::new(),
            arena_bytes: 0,
        }
    }

    /// Places the arenas of [`Placement::DirectCopy`] and
    /// [`Placement::CachedCopy`] in coarse-grain managed memory
    /// ([`Stream::alloc_managed_coarse`]): on unified memory the device heap
    /// is a fraction of RAM, and weights larger than it fit there.
    pub fn managed_arenas(mut self, managed: bool) -> Self {
        self.managed = managed;
        self
    }

    fn alloc_arena(&self, bytes: usize) -> Result<DeviceBuffer<u8>, HipError> {
        if self.managed {
            self.stream.alloc_managed_coarse::<u8>(bytes)
        } else {
            self.stream.alloc_persistent::<u8>(bytes)
        }
    }

    /// Queues `len` elements of `T` at byte `offset` of `file`.
    pub fn add<T: DeviceCopy>(
        &mut self,
        file: &Arc<MappedFile>,
        offset: usize,
        len: usize,
    ) -> Result<Pending<T>, HipError> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(HipError::refused("weight size overflows"))?;
        if matches!(self.placement, Placement::Map | Placement::HostCopy)
            && !offset.is_multiple_of(align_of::<T>())
        {
            return Err(HipError::refused(
                "mapped weight is not aligned to its element type",
            ));
        }
        self.add_parts(&[(file, offset, bytes)], 0)
    }

    /// Queues one tensor of the byte ranges `(file, offset, bytes)` in order,
    /// followed by `tail` zero bytes: stacked projections of one input, and
    /// the margin of kernels that read whole blocks past the last row. The
    /// total must be whole elements of `T`. Only the copy placements join
    /// ranges or add a tail.
    pub fn add_parts<T: DeviceCopy>(
        &mut self,
        parts: &[(&Arc<MappedFile>, usize, usize)],
        tail: usize,
    ) -> Result<Pending<T>, HipError> {
        if parts.is_empty() {
            return Err(HipError::refused("a weight needs at least one file range"));
        }
        if matches!(self.placement, Placement::Map | Placement::HostCopy)
            && (parts.len() != 1 || tail != 0)
        {
            return Err(HipError::refused(
                "joined or padded weights need a copy placement",
            ));
        }
        let mut bytes = 0usize;
        for &(file, offset, size) in parts {
            let end = offset
                .checked_add(size)
                .ok_or(HipError::refused("weight range overflows"))?;
            if end > file.len() {
                return Err(HipError::refused("weight range lies outside its file"));
            }
            bytes = bytes
                .checked_add(size)
                .ok_or(HipError::refused("weight size overflows"))?;
        }
        let total = bytes
            .checked_add(tail)
            .ok_or(HipError::refused("weight size overflows"))?;
        if !total.is_multiple_of(size_of::<T>()) {
            return Err(HipError::refused(
                "weight bytes are not whole elements of its type",
            ));
        }
        let start = self.arena_bytes.next_multiple_of(ARENA_ALIGNMENT);
        let end = start
            .checked_add(total)
            .ok_or(HipError::refused("weight arena size overflows"))?;
        let first = self.entries.len();
        let mut arena_offset = start;
        for &(file, offset, size) in parts {
            self.entries.push(Entry {
                file: Arc::clone(file),
                file_offset: offset,
                bytes: size,
                arena_offset,
            });
            arena_offset += size;
        }
        self.arena_bytes = end;
        self.queued.push(Queued {
            first,
            parts: parts.len(),
            bytes,
            tail,
        });
        Ok(Pending {
            loader: self.id,
            index: self.queued.len() - 1,
            len: total / size_of::<T>(),
            _element: PhantomData,
        })
    }

    /// Places every queued tensor and waits until all of them are complete.
    pub fn finish(self) -> Result<LoadedWeights, HipError> {
        self.stream.ensure_healthy()?;
        if self.stream.is_recording() {
            return Err(HipError::capture_violation("weight loading"));
        }
        let tensors = match self.placement {
            Placement::DirectCopy => self.finish_direct()?,
            Placement::CachedCopy => self.finish_cached()?,
            Placement::Map => self.finish_map()?,
            Placement::HostCopy => self.finish_host_copy()?,
        };
        Ok(LoadedWeights {
            loader: self.id,
            tensors,
        })
    }

    /// Copies on the device from the registered file mappings into arenas of
    /// at most [`CACHED_ARENA`] bytes. Allocating device memory costs about
    /// as much as copying it on the host (the driver provides zeroed pages at
    /// ~25 GB/s), so each arena is allocated while the copies into the
    /// previous ones run; the device copies themselves reach ~80 GB/s. Queue
    /// small tensors last: they form a short final arena.
    fn finish_cached(&self) -> Result<Vec<(Arc<Owner>, usize)>, HipError> {
        self.finish_cached_with(CACHED_ARENA, CACHED_TAIL)
    }

    fn finish_cached_with(
        &self,
        arena_limit: usize,
        tail_limit: usize,
    ) -> Result<Vec<(Arc<Owner>, usize)>, HipError> {
        let files = self.files();
        std::thread::scope(|scope| {
            // The files prefault while this thread allocates the first arena,
            // which takes about as long.
            let prefaulted = scope.spawn(|| prefault(&files));
            self.copy_cached(arena_limit, tail_limit, Some(prefaulted))
        })
    }

    fn copy_cached(
        &self,
        arena_limit: usize,
        tail_limit: usize,
        mut prefaulted: Option<ScopedJoinHandle<'_, Result<(), HipError>>>,
    ) -> Result<Vec<(Arc<Owner>, usize)>, HipError> {
        let mut registrations: Vec<(&Arc<MappedFile>, DeviceTensor<u8>)> = Vec::new();
        // The trailing tensors that fit in CACHED_TAIL get their own last
        // arena: its copy is the only one no allocation overlaps.
        let mut tail = self.queued.len();
        let mut tail_bytes = 0usize;
        while tail > 0 && tail_bytes + self.queued[tail - 1].total() <= tail_limit {
            tail -= 1;
            tail_bytes += self.queued[tail].total().next_multiple_of(ARENA_ALIGNMENT);
        }
        let mut arenas = Vec::new();
        let mut start = 0;
        while start < self.queued.len() {
            // A run of tensors that fills one arena (at least one tensor).
            let end = if start < tail {
                tail
            } else {
                self.queued.len()
            };
            let mut offsets = Vec::new();
            let mut bytes = 0usize;
            for q in &self.queued[start..end] {
                let offset = bytes.next_multiple_of(ARENA_ALIGNMENT);
                if !offsets.is_empty() && offset + q.total() > arena_limit {
                    break;
                }
                offsets.push(offset);
                bytes = offset + q.total();
            }
            let mut arena = self.alloc_arena(bytes)?;
            if let Some(prefaulted) = prefaulted.take() {
                prefaulted
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))?;
            }
            for (q, &offset) in self.queued[start..].iter().zip(&offsets) {
                let mut at = offset;
                for e in &self.entries[q.first..q.first + q.parts] {
                    let index = match registrations
                        .iter()
                        .position(|(f, _)| Arc::ptr_eq(f, &e.file))
                    {
                        Some(index) => index,
                        None => {
                            let owner = e.file.register(self.stream)?;
                            registrations
                                .push((&e.file, DeviceTensor::from_shared(owner, 0, e.file.len())));
                            registrations.len() - 1
                        }
                    };
                    self.stream.copy(
                        registrations[index]
                            .1
                            .slice(e.file_offset..e.file_offset + e.bytes),
                        arena.slice_mut(at..at + e.bytes),
                    )?;
                    at += e.bytes;
                }
                if q.tail != 0 {
                    self.stream
                        .fill_bytes(arena.slice_mut(at..at + q.tail), 0)?;
                }
            }
            start += offsets.len();
            arenas.push((arena, offsets));
        }
        let mut tensors = Vec::with_capacity(self.queued.len());
        for (arena, offsets) in arenas {
            // The first freeze waits for every copy on the stream.
            let tensor = arena.freeze(self.stream)?;
            tensors.extend(offsets.into_iter().map(|o| (Arc::clone(tensor.owner()), o)));
        }
        // The registrations are released once no copy needs them any more.
        drop(registrations);
        Ok(tensors)
    }

    /// `O_DIRECT` reads into pinned staging by parallel workers, each copying
    /// its chunks on its own stream, into arenas of at most [`DIRECT_ARENA`]
    /// bytes that this thread allocates a little ahead of the workers
    /// (allocating costs ~40 ms per GB, reading ~270).
    fn finish_direct(&self) -> Result<Vec<(Arc<Owner>, usize)>, HipError> {
        self.finish_direct_with(DIRECT_ARENA)
    }

    fn finish_direct_with(&self, arena_limit: usize) -> Result<Vec<(Arc<Owner>, usize)>, HipError> {
        // Runs of whole tensors, each an arena of at least one tensor.
        let mut arenas: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut placed = Vec::with_capacity(self.queued.len());
        for q in &self.queued {
            let fits = arenas.last().is_some_and(|(bytes, offsets)| {
                !offsets.is_empty()
                    && bytes.next_multiple_of(ARENA_ALIGNMENT) + q.total() <= arena_limit
            });
            if !fits {
                arenas.push((0, Vec::new()));
            }
            let index = arenas.len() - 1;
            let (bytes, offsets) = arenas.last_mut().expect("pushed above");
            let offset = bytes.next_multiple_of(ARENA_ALIGNMENT);
            offsets.push(offset);
            *bytes = offset + q.total();
            placed.push((index, offset));
        }
        let mut tasks = Vec::new();
        for (q, &(arena, offset)) in self.queued.iter().zip(&placed) {
            let first = self.entries[q.first].arena_offset;
            for (entry, e) in self.entries[q.first..q.first + q.parts].iter().enumerate() {
                let mut done = 0;
                while done < e.bytes {
                    let size = CHUNK.min(e.bytes - done);
                    tasks.push(Task {
                        entry: q.first + entry,
                        offset: done,
                        size,
                        arena,
                        destination: offset + e.arena_offset - first + done,
                    });
                    done += size;
                }
            }
        }
        let published = Published {
            bases: Mutex::new(Vec::with_capacity(arenas.len())),
            ready: Condvar::new(),
            started: AtomicUsize::new(0),
        };
        let mut buffers = Vec::with_capacity(arenas.len());
        let next = AtomicUsize::new(0);
        let stop = AtomicBool::new(false);
        let failure = Mutex::new(None);
        let fail = |error: HipError| {
            stop.store(true, Ordering::Relaxed);
            if let Ok(mut slot) = failure.lock() {
                slot.get_or_insert(error);
            }
            published.ready.notify_all();
        };
        // The workers share only these, not the loader and its stream.
        let (entries, tasks, published, next, stop, fail) =
            (&self.entries, &tasks, &published, &next, &stop, &fail);
        std::thread::scope(|scope| {
            for _ in 0..UPLOADERS.min(tasks.len()) {
                scope.spawn(move || {
                    if let Err(error) = upload_worker(entries, tasks, published, next, stop) {
                        fail(error);
                    }
                });
            }
            for (index, &(bytes, _)) in arenas.iter().enumerate() {
                // Allocating contends with the copies in flight: stay just
                // ahead of the workers instead of allocating everything first.
                if published.wait_for_workers(index, stop).is_err() || stop.load(Ordering::Relaxed)
                {
                    break;
                }
                // Upload workers write from their own streams: each
                // allocation is complete before its address is published.
                let allocated = self.alloc_arena(bytes).and_then(|mut arena| {
                    self.stream.synchronize()?;
                    let base = arena.as_mut_slice().address();
                    Ok((arena, base))
                });
                match allocated {
                    Ok((arena, base)) => {
                        buffers.push(arena);
                        if let Ok(mut bases) = published.bases.lock() {
                            bases.push(base);
                        }
                        published.ready.notify_all();
                    }
                    Err(error) => fail(error),
                }
            }
        });
        if let Some(error) = failure.into_inner().ok().flatten() {
            return Err(error);
        }
        for (q, &(arena, offset)) in self.queued.iter().zip(&placed) {
            if q.tail != 0 {
                let at = offset + q.bytes;
                self.stream
                    .fill_bytes(buffers[arena].slice_mut(at..at + q.tail), 0)?;
            }
        }
        let mut owners = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            // The first freeze waits for the tail fills on the stream.
            owners.push(Arc::clone(buffer.freeze(self.stream)?.owner()));
        }
        Ok(placed
            .into_iter()
            .map(|(arena, offset)| (Arc::clone(&owners[arena]), offset))
            .collect())
    }

    /// Faults every distinct file into the page cache, in parallel.
    fn prefault_files(&self) -> Result<(), HipError> {
        prefault(&self.files())
    }

    /// Each queued file once, in queue order.
    fn files(&self) -> Vec<&Arc<MappedFile>> {
        let mut files: Vec<&Arc<MappedFile>> = Vec::new();
        for e in &self.entries {
            if !files.iter().any(|file| Arc::ptr_eq(file, &e.file)) {
                files.push(&e.file);
            }
        }
        files
    }

    fn finish_host_copy(&self) -> Result<Vec<(Arc<Owner>, usize)>, HipError> {
        let mut owners: Vec<(&Arc<MappedFile>, Arc<Owner>)> = Vec::new();
        let mut tensors = Vec::with_capacity(self.queued.len());
        // Single-range tensors without a tail only (`add_parts`).
        for e in self.queued.iter().map(|q| &self.entries[q.first]) {
            let owner = match owners.iter().find(|(file, _)| Arc::ptr_eq(file, &e.file)) {
                Some((_, owner)) => Arc::clone(owner),
                None => {
                    let copy = e
                        .file
                        .host_copy()
                        .map_err(|_| HipError::refused("cannot copy the mapped weight file"))?;
                    let owner = copy.register(self.stream)?;
                    owners.push((&e.file, Arc::clone(&owner)));
                    owner
                }
            };
            tensors.push((owner, e.file_offset));
        }
        Ok(tensors)
    }

    fn finish_map(&self) -> Result<Vec<(Arc<Owner>, usize)>, HipError> {
        self.prefault_files()?;
        let mut owners: Vec<(Arc<MappedFile>, Arc<Owner>)> = Vec::new();
        let mut tensors = Vec::with_capacity(self.queued.len());
        // Single-range tensors without a tail only (`add_parts`).
        for e in self.queued.iter().map(|q| &self.entries[q.first]) {
            let owner = match owners.iter().find(|(file, _)| Arc::ptr_eq(file, &e.file)) {
                Some((_, owner)) => Arc::clone(owner),
                None => {
                    let owner = e.file.register(self.stream)?;
                    owners.push((Arc::clone(&e.file), Arc::clone(&owner)));
                    owner
                }
            };
            tensors.push((owner, e.file_offset));
        }
        Ok(tensors)
    }
}

struct Task {
    entry: usize,
    offset: usize,
    size: usize,
    /// The arena and the byte offset in it.
    arena: usize,
    destination: usize,
}

/// Device addresses of the arenas allocated so far, in order.
struct Published {
    bases: Mutex<Vec<usize>>,
    ready: Condvar,
    /// The latest arena a worker started on.
    started: AtomicUsize,
}

impl Published {
    /// Records that a worker needs arena `index` (`usize::MAX`: none any
    /// more).
    fn start(&self, index: usize) {
        if self.started.fetch_max(index, Ordering::Relaxed) < index {
            // Taking the lock orders this against a waiting allocator.
            drop(self.bases.lock());
            self.ready.notify_all();
        }
    }

    /// Waits until the workers are at most [`ARENAS_AHEAD`] arenas behind
    /// arena `index`, or the upload stops.
    fn wait_for_workers(&self, index: usize, stop: &AtomicBool) -> Result<(), HipError> {
        let poisoned = || HipError::refused("weight upload lock poisoned");
        let mut bases = self.bases.lock().map_err(|_| poisoned())?;
        while index
            > self
                .started
                .load(Ordering::Relaxed)
                .saturating_add(ARENAS_AHEAD)
            && !stop.load(Ordering::Relaxed)
        {
            bases = self.ready.wait(bases).map_err(|_| poisoned())?;
        }
        drop(bases);
        Ok(())
    }

    /// Waits until arena `index` is allocated; `None` once the upload stops.
    fn base(&self, index: usize, stop: &AtomicBool) -> Result<Option<usize>, HipError> {
        let poisoned = || HipError::refused("weight upload lock poisoned");
        let mut bases = self.bases.lock().map_err(|_| poisoned())?;
        loop {
            if let Some(&base) = bases.get(index) {
                return Ok(Some(base));
            }
            if stop.load(Ordering::Relaxed) {
                return Ok(None);
            }
            bases = self.ready.wait(bases).map_err(|_| poisoned())?;
        }
    }
}

/// One upload worker: takes chunks in order, reads each into its pinned
/// staging and copies it into its arena once that is allocated.
fn upload_worker(
    entries: &[Entry],
    tasks: &[Task],
    published: &Published,
    next: &AtomicUsize,
    stop: &AtomicBool,
) -> Result<(), HipError> {
    let stream = Stream::new()?;
    let mut staging = PinnedBuffer::<u8>::zeroed(CHUNK + DIRECT_ALIGNMENT)?;
    let mut direct_files: Vec<(*const MappedFile, Option<File>)> = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        let Some(task) = tasks.get(next.fetch_add(1, Ordering::Relaxed)) else {
            // No task is left: the allocator need not wait for this worker.
            published.start(usize::MAX);
            break;
        };
        // Recorded before waiting for the arena, so the allocator always
        // runs ahead of a waiting worker.
        published.start(task.arena);
        let entry = &entries[task.entry];
        let offset = entry.file_offset + task.offset;
        let file = Arc::as_ptr(&entry.file);
        if !direct_files.iter().any(|(known, _)| *known == file) {
            direct_files.push((file, open_direct(entry.file.file())));
        }
        let direct_file = direct_files
            .iter()
            .find(|(known, _)| *known == file)
            .and_then(|(_, f)| f.as_ref());
        let payload = read_chunk(
            &entry.file,
            direct_file,
            offset,
            task.size,
            staging.host_mut()?,
        )
        .map_err(|_| HipError::refused("reading the weight file failed"))?;
        let Some(base) = published.base(task.arena, stop)? else {
            break;
        };
        let source = staging.host()?[payload].as_ptr();
        let destination = base + task.destination;
        // SAFETY: the destination lies inside its arena (the chunk is within
        // its tensor, the tensor within the arena), which is published only
        // once allocated and outlives the workers; the source is pinned
        // staging memory, alive and unchanged until the copy completes below.
        let status = unsafe {
            sys::hipMemcpyAsync(
                destination as *mut std::ffi::c_void,
                source.cast(),
                task.size,
                sys::HIP_MEMCPY_HOST_TO_DEVICE,
                stream.as_raw(),
            )
        };
        stream.observe(check(status, "hipMemcpyAsync weight upload"))?;
        stream.synchronize()?;
    }
    Ok(())
}

/// An independent `O_DIRECT` description of an open file, or `None` when the
/// filesystem refuses it. Reopening the descriptor avoids the pathname.
fn open_direct(file: &File) -> Option<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(format!("/proc/self/fd/{}", file.as_raw_fd()))
        .ok()
}

/// Reads `size` bytes at `offset` into `staging` and returns where they are.
fn read_chunk(
    file: &MappedFile,
    direct: Option<&File>,
    offset: usize,
    size: usize,
    staging: &mut [u8],
) -> io::Result<Range<usize>> {
    if let Some(direct) = direct {
        let begin = offset / DIRECT_ALIGNMENT * DIRECT_ALIGNMENT;
        let skip = offset - begin;
        let length = (skip + size).next_multiple_of(DIRECT_ALIGNMENT);
        match read_fully(direct, &mut staging[..length], begin, skip + size) {
            Ok(()) => return Ok(skip..skip + size),
            // Some filesystems accept O_DIRECT at open but reject the reads.
            Err(error) if matches!(error.raw_os_error(), Some(libc::EINVAL | libc::EOPNOTSUPP)) => {
            }
            Err(error) => return Err(error),
        }
    }
    read_fully(file.file(), &mut staging[..size], offset, size)?;
    // SAFETY: advisory call on an open descriptor.
    unsafe {
        let _ = libc::posix_fadvise(
            file.file().as_raw_fd(),
            offset as libc::off_t,
            size as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        );
    }
    Ok(0..size)
}

/// Reads until `required` bytes are in `buffer` (it may be longer, for
/// `O_DIRECT` alignment; the file may end before its end).
fn read_fully(file: &File, buffer: &mut [u8], offset: usize, required: usize) -> io::Result<()> {
    let mut got = 0;
    while got < required {
        match file.read_at(&mut buffer[got..], (offset + got) as u64) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => got += n,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small limits: several arenas plus a final one of the trailing
    /// entries; every tensor still reads back its file bytes.
    #[test]
    #[cfg(feature = "gpu")]
    fn cached_copies_split_into_arenas_with_a_short_last_one() {
        let path = std::env::temp_dir().join(format!("gufo-arenas-{}.bin", std::process::id()));
        let bytes: Vec<u8> = (0..1u32 << 16)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 9) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let file = MappedFile::open(&path).unwrap();
        let stream = Stream::new().unwrap();
        let mut loader = WeightLoader::new(&stream, Placement::CachedCopy);
        let sizes = [3000, 1000, 5000, 700, 2500, 900, 300, 200];
        let mut offset = 1;
        let mut queued = Vec::new();
        for size in sizes {
            queued.push((offset, size, loader.add::<u8>(&file, offset, size).unwrap()));
            offset += size + 7;
        }
        let placed = loader.finish_cached_with(4096, 600).unwrap();
        let weights = LoadedWeights {
            loader: loader.id,
            tensors: placed,
        };
        let tensors: Vec<_> = queued
            .into_iter()
            .map(|(offset, size, pending)| (offset, size, weights.get(pending)))
            .collect();
        let arenas: Vec<u64> = tensors.iter().map(|t| t.2.allocation().id()).collect();
        let mut distinct = arenas.clone();
        distinct.dedup();
        // [3000, 1000 at 3072] [5000 alone] [700, 2500 at 768] [900], and the
        // tail [300, 200] in an arena of its own.
        assert_eq!(distinct.len(), 5, "{arenas:?}");
        assert_eq!((arenas[0], arenas[3]), (arenas[1], arenas[4]));
        assert_eq!(arenas[6], arenas[7]);
        assert_ne!(arenas[5], arenas[6]);
        for (offset, size, tensor) in tensors {
            let mut back = vec![0u8; size];
            stream.download(tensor.view(), &mut back).unwrap();
            assert_eq!(back, bytes[offset..offset + size], "tensor at {offset}");
            assert_eq!(tensor.offset() % ARENA_ALIGNMENT, 0);
        }
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    /// Small limits: direct copies fill arenas that are allocated while
    /// the workers upload, joined tensors and tails included; every tensor
    /// reads back its file bytes.
    #[test]
    #[cfg(feature = "gpu")]
    fn direct_copies_split_into_arenas() {
        let path = std::env::temp_dir().join(format!("gufo-direct-{}.bin", std::process::id()));
        let bytes: Vec<u8> = (0..1u32 << 16)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 11) as u8 | 1)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let file = MappedFile::open(&path).unwrap();
        let stream = Stream::new().unwrap();
        let mut loader = WeightLoader::new(&stream, Placement::DirectCopy);
        let sizes = [3000, 1000, 5000, 700, 2500, 900, 300, 200];
        let mut offset = 1;
        let mut queued = Vec::new();
        for size in sizes {
            let mut expected = bytes[offset..offset + size].to_vec();
            let pending = if size == 2500 {
                // Two ranges and a tail in one tensor.
                expected.extend_from_slice(&bytes[40_000..40_100]);
                expected.extend_from_slice(&[0; 32]);
                loader
                    .add_parts::<u8>(&[(&file, offset, size), (&file, 40_000, 100)], 32)
                    .unwrap()
            } else {
                loader.add::<u8>(&file, offset, size).unwrap()
            };
            queued.push((expected, pending));
            offset += size + 7;
        }
        let placed = loader.finish_direct_with(4096).unwrap();
        let weights = LoadedWeights {
            loader: loader.id,
            tensors: placed,
        };
        let tensors: Vec<_> = queued
            .into_iter()
            .map(|(expected, pending)| (expected, weights.get(pending)))
            .collect();
        let arenas: Vec<u64> = tensors.iter().map(|t| t.1.allocation().id()).collect();
        let mut distinct = arenas.clone();
        distinct.dedup();
        // [3000, 1000 at 3072] [5000 alone] [700, 2632 at 768] [900, 300, 200].
        assert_eq!(distinct.len(), 4, "{arenas:?}");
        assert_eq!((arenas[0], arenas[3]), (arenas[1], arenas[4]));
        assert_eq!((arenas[5], arenas[5]), (arenas[6], arenas[7]));
        for (expected, tensor) in tensors {
            let mut back = vec![0u8; expected.len()];
            stream.download(tensor.view(), &mut back).unwrap();
            assert_eq!(back, expected);
            assert_eq!(tensor.offset() % ARENA_ALIGNMENT, 0);
        }
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    /// Arenas without bytes to copy (empty tensors with tails) at the end
    /// still get allocated: the allocator never waits for workers that have
    /// nothing left to do.
    #[test]
    #[cfg(feature = "gpu")]
    fn direct_copies_finish_with_empty_tensors() {
        let path = std::env::temp_dir().join(format!("gufo-empty-{}.bin", std::process::id()));
        std::fs::write(&path, vec![5u8; 8192]).unwrap();
        let file = MappedFile::open(&path).unwrap();
        let stream = Stream::new().unwrap();
        let mut loader = WeightLoader::new(&stream, Placement::DirectCopy);
        let data = loader.add_parts::<u8>(&[(&file, 0, 4000)], 0).unwrap();
        let empty: Vec<_> = (0..6)
            .map(|_| loader.add_parts::<u8>(&[(&file, 0, 0)], 4000).unwrap())
            .collect();
        let placed = loader.finish_direct_with(4096).unwrap();
        let weights = LoadedWeights {
            loader: loader.id,
            tensors: placed,
        };
        let mut back = vec![0u8; 4000];
        stream
            .download(weights.get(data).view(), &mut back)
            .unwrap();
        assert!(back.iter().all(|&b| b == 5));
        for pending in empty {
            let tensor = weights.get(pending);
            let mut back = vec![1u8; tensor.len()];
            stream.download(tensor.view(), &mut back).unwrap();
            assert!(back.iter().all(|&b| b == 0));
        }
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    /// Joined ranges land back to back and end in zero bytes, in both copy
    /// placements; the mapping placements refuse them.
    #[test]
    #[cfg(feature = "gpu")]
    fn joined_weights_with_a_zero_tail() {
        let path = std::env::temp_dir().join(format!("gufo-joined-{}.bin", std::process::id()));
        let bytes: Vec<u8> = (0..1u32 << 14)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 7) as u8 | 1)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let file = MappedFile::open(&path).unwrap();
        let stream = Stream::new().unwrap();
        let mut expected = bytes[100..400].to_vec();
        expected.extend_from_slice(&bytes[5000..5212]);
        expected.extend_from_slice(&[0; 64]);
        for (placement, managed) in [
            (Placement::DirectCopy, false),
            (Placement::CachedCopy, false),
            (Placement::DirectCopy, true),
            (Placement::CachedCopy, true),
        ] {
            let mut loader = WeightLoader::new(&stream, placement).managed_arenas(managed);
            let single = loader.add::<u8>(&file, 7, 33).unwrap();
            let joined = loader
                .add_parts::<u8>(&[(&file, 100, 300), (&file, 5000, 212)], 64)
                .unwrap();
            let weights = loader.finish().unwrap();
            let (single, joined) = (weights.get(single), weights.get(joined));
            let mut back = vec![0u8; joined.len()];
            stream.download(joined.view(), &mut back).unwrap();
            assert_eq!(back, expected, "{placement:?}");
            let mut back = vec![0u8; 33];
            stream.download(single.view(), &mut back).unwrap();
            assert_eq!(back, bytes[7..40], "{placement:?}");
            assert_eq!(joined.offset() % ARENA_ALIGNMENT, 0);
            let attributes = crate::device::pointer_attributes(
                single.view().address() as *const std::ffi::c_void
            )
            .unwrap();
            assert_eq!(attributes.managed, managed, "{placement:?}");
        }
        for placement in [Placement::Map, Placement::HostCopy] {
            let mut loader = WeightLoader::new(&stream, placement);
            assert!(loader.add_parts::<u8>(&[(&file, 0, 64)], 16).is_err());
            assert!(
                loader
                    .add_parts::<u8>(&[(&file, 0, 64), (&file, 64, 64)], 0)
                    .is_err()
            );
        }
        let mut loader = WeightLoader::new(&stream, Placement::DirectCopy);
        assert!(loader.add_parts::<f32>(&[(&file, 0, 6)], 0).is_err());
        assert!(loader.add_parts::<u8>(&[(&file, 1 << 14, 1)], 0).is_err());
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    /// Both read paths return the same bytes; the buffered one is the fallback
    /// for filesystems that refuse `O_DIRECT` reads.
    #[test]
    fn direct_and_buffered_reads_agree() {
        let path = std::env::temp_dir().join(format!("gufo-read-chunk-{}.bin", std::process::id()));
        let bytes: Vec<u8> = (0..(3 << 20) as u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 11) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let file = MappedFile::open(&path).unwrap();
        let mut staging = vec![0u8; CHUNK + DIRECT_ALIGNMENT];
        for (offset, size) in [(0, 4096), (12_345, 1 << 20), ((3 << 20) - 777, 777)] {
            let expected = &bytes[offset..offset + size];
            let buffered = read_chunk(&file, None, offset, size, &mut staging).unwrap();
            assert_eq!(&staging[buffered], expected, "buffered read at {offset}");
            if let Some(direct) = open_direct(file.file()) {
                // O_DIRECT needs an aligned buffer: use an aligned window.
                let shift = staging.as_ptr().align_offset(DIRECT_ALIGNMENT);
                let window = &mut staging[shift..shift + CHUNK];
                let range = read_chunk(&file, Some(&direct), offset, size, window).unwrap();
                assert_eq!(&window[range], expected, "direct read at {offset}");
            }
        }
        drop(file);
        std::fs::remove_file(&path).unwrap();
    }
}
