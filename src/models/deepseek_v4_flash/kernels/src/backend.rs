//! The DeepSeek V4 Flash device state: one [`Backend`] at a time, as the
//! C++ engine allowed one runtime per process (the vendored MMQ tier keeps
//! process-wide scratch).

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use gufo_hip::blas::{Blas, BlasLt};
use gufo_hip::{DeviceSlice, Launch, MappedFile, PinnedBuffer, Placement, Stream, WeightLoader};

use crate::ops::Projections;
use crate::tensor::{DeviceTensor, Input, Output};

static ACTIVE: AtomicBool = AtomicBool::new(false);

/// A failed DS4 operation; its reason went to stderr, as in C++.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error {
    pub op: &'static str,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ds4 {} failed", self.op)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Which registered mapping a weight offset refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// The target GGUF.
    Target,
    /// The DSpark support GGUF.
    Support,
}

/// Kernel selection of verification and support blocks; prefill and decode
/// use `Off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchMode {
    #[default]
    Off = 0,
    Verifier = 1,
    Support = 2,
}

/// A device copy of the target file bytes at `offset`.
struct TargetSpan {
    offset: u64,
    tensor: gufo_hip::DeviceTensor<u8>,
}

/// Support file bytes at `offset` held at `arena` in the support arena.
struct SupportRange {
    offset: u64,
    bytes: u64,
    arena: usize,
}

impl SupportRange {
    fn contains(&self, offset: u64, bytes: u64) -> bool {
        offset >= self.offset
            && offset
                .checked_add(bytes)
                .is_some_and(|end| end <= self.offset + self.bytes)
    }
}

/// The device state of this process. Not `Sync`: launches and transfers
/// run on one thread at a time, in order on the backend's stream.
pub struct Backend {
    target: Option<Arc<MappedFile>>,
    support: Option<Arc<MappedFile>>,
    /// Device copies of target file ranges, by offset, in file order.
    target_spans: Vec<TargetSpan>,
    /// The support model's resident ranges in one managed arena.
    support_arena: Option<gufo_hip::DeviceTensor<u8>>,
    support_ranges: Vec<SupportRange>,
    /// The last `set_batch_mode`, which some wrappers check against.
    batch_mode: Cell<BatchMode>,
    /// KV caches are managed memory whatever their size.
    force_managed_kv: Cell<bool>,
    /// Between `begin_commands` and the wait that ends them: copies then
    /// stay in stream order, the closing wait covering them.
    in_commands: Cell<bool>,
    /// The deferred routed sum (`ds4_gpu_set_routed_defer_sum`): requested,
    /// and taken by a routed call since the request.
    pub(crate) routed_defer: Cell<(bool, bool)>,
    /// The hipBLAS handle (default atomics, as the C++ runtime made it) and
    /// the hipBLASLt handle; dropped before the stream.
    pub(crate) blas: Blas,
    pub(crate) blaslt: RefCell<Option<BlasLt>>,
    /// The projections' scratch, F16 input mirror and F16 weight cache.
    pub(crate) projections: Projections,
    /// The page-locked buffer of [`read_f32`](Self::read_f32).
    read_staging: RefCell<Option<PinnedBuffer<f32>>>,
    /// The idle transfers of [`SnapshotReader`]s.
    snapshot_transfers: Arc<Mutex<Vec<SnapshotTransfer>>>,
    /// The stream all work is ordered on, the vendored MMQ tier's too.
    stream: Stream,
    _not_sync: PhantomData<Cell<()>>,
}

/// Device heap bytes kept free for driver command submission
/// (`kDispatchReserve`).
const DISPATCH_RESERVE: u64 = 4 << 30;

/// Whether `bytes` fit in `free` heap bytes and still leave the dispatch
/// reserve (`HeapFitsAboveReserve`).
fn heap_fits_above_reserve(free: u64, bytes: u64) -> bool {
    bytes <= free && free - bytes >= DISPATCH_RESERVE
}

impl Backend {
    /// The backend's stream, hipBLAS and hipBLASLt handles, and the MMQ
    /// tier on a gfx1151 (`ds4_gpu_init`). Fails while another backend
    /// exists.
    pub fn init() -> Result<Self> {
        if ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(Error {
                op: "init (one backend per process)",
            });
        }
        let stream = match Stream::new() {
            Ok(stream) => stream,
            Err(_) => {
                ACTIVE.store(false, Ordering::Release);
                return Err(Error { op: "stream" });
            }
        };
        // The C++ runtime also asked for TF32 math, which hipBLAS refuses on
        // gfx1151 (status 7), and ignored the refusal.
        let blas = match Blas::with_default_atomics(&stream) {
            Ok(blas) => blas,
            Err(error) => {
                eprintln!("ds4: hipBLAS create handle failed: {error}");
                ACTIVE.store(false, Ordering::Release);
                return Err(Error { op: "hipBLAS" });
            }
        };
        let blaslt = BlasLt::new(&stream, 0)
            .map_err(|error| eprintln!("ds4: hipBLASLt create handle failed: {error}"))
            .ok();
        let backend = Self {
            target: None,
            support: None,
            target_spans: Vec::new(),
            support_arena: None,
            support_ranges: Vec::new(),
            batch_mode: Cell::new(BatchMode::Off),
            force_managed_kv: Cell::new(false),
            in_commands: Cell::new(false),
            routed_defer: Cell::new((false, false)),
            blas,
            blaslt: RefCell::new(blaslt),
            projections: Projections::default(),
            read_staging: RefCell::new(None),
            snapshot_transfers: Arc::default(),
            stream,
            _not_sync: PhantomData,
        };
        let gfx1151 = match gufo_hip::device_properties(0) {
            Ok(p) => {
                eprintln!(
                    "ds4: ROCm backend initialized on {} (sm_{}{})",
                    p.name, p.major, p.minor
                );
                p.major == 11 && p.minor == 5
            }
            Err(_) => false,
        };
        // The vendored MMQ tier is the largest prefill win available here;
        // the routes scope it to prompt chunks.
        let mmq_ready = gfx1151
            && matches!(
                crate::generated::mmq_init(&backend.stream),
                Ok(Launch::Launched)
            );
        eprintln!(
            "ds4: ROCm native MMQ {}",
            if mmq_ready { "enabled" } else { "unavailable" }
        );
        backend.projections.mmq_ready.set(mmq_ready);
        backend.projections.gfx1151.set(gfx1151);
        Ok(backend)
    }

    /// Registers the target GGUF mapping; weight offsets of [`Model::Target`]
    /// refer to it.
    pub fn set_model(&mut self, map: Arc<MappedFile>) -> Result<()> {
        self.target = Some(map);
        Ok(())
    }

    /// Uploads the target file ranges `spans` (offset and bytes, one device
    /// allocation each, as the C++ engine allocates them, in file order
    /// without overlap) with gufo-hip's direct copy as the resident weights.
    /// Returns the bytes.
    pub fn load_target_spans(&mut self, spans: &[(u64, u64)]) -> Result<u64> {
        const OP: &str = "load target spans";
        let fail = |error: gufo_hip::HipError| {
            eprintln!("ds4: ROCm model upload failed: {error}");
            Error { op: OP }
        };
        let map = Arc::clone(self.target.as_ref().ok_or(Error { op: OP })?);
        let stream = Stream::new().map_err(fail)?;
        // On unified memory the device heap is a fraction of RAM (about 61
        // of 122 GiB with the default TTM limit): weights that do not fit
        // above the dispatch reserve go wholly to coarse-grain managed
        // memory, decided once so the heap keeps room for session state
        // (upstream `WeightAllocator`).
        let weight_bytes: u64 = spans.iter().map(|&(_, bytes)| bytes).sum();
        let (free, _) = gufo_hip::device_memory().map_err(fail)?;
        let managed = !heap_fits_above_reserve(free as u64, weight_bytes);
        if managed {
            eprintln!(
                "ds4: weights ({:.2} GiB) do not fit the device heap above its {:.2} GiB \
                 dispatch reserve; placing them in managed memory",
                weight_bytes as f64 / 1_073_741_824.0,
                DISPATCH_RESERVE as f64 / 1_073_741_824.0
            );
        }
        let mut loader = WeightLoader::new(&stream, Placement::DirectCopy).managed_arenas(managed);
        let mut pending = Vec::with_capacity(spans.len());
        for &(offset, bytes) in spans {
            let offset = usize::try_from(offset).map_err(|_| Error { op: OP })?;
            let bytes = usize::try_from(bytes).map_err(|_| Error { op: OP })?;
            pending.push(loader.add::<u8>(&map, offset, bytes).map_err(fail)?);
        }
        let loaded = loader.finish().map_err(fail)?;
        let mut total = 0;
        for (pending, &(offset, bytes)) in pending.into_iter().zip(spans) {
            // A weight lies in the last span starting at or before it.
            if self
                .target_spans
                .last()
                .is_some_and(|last| last.offset + last.tensor.len() as u64 > offset)
            {
                return Err(Error {
                    op: "load target spans (order)",
                });
            }
            self.target_spans.push(TargetSpan {
                offset,
                tensor: loaded.get(pending),
            });
            total += bytes;
        }
        Ok(total)
    }

    /// Uploads the support model's tensor ranges (offset, bytes, label) of
    /// `map` into one managed arena of `arena_bytes`, each 256-byte aligned
    /// so quantized block loads stay aligned, as
    /// `ds4_gpu_reserve_support_map` and `_cache_support_range` did. A range
    /// inside an uploaded one is not copied again.
    pub fn set_support(
        &mut self,
        map: Arc<MappedFile>,
        arena_bytes: u64,
        ranges: &[(u64, u64, &str)],
    ) -> Result<()> {
        const OP: &str = "support arena";
        if self.support.is_some() {
            return Err(Error {
                op: "support model already registered",
            });
        }
        let fail = |_| Error { op: OP };
        let host = map.bytes();
        let size = host.len() as u64;
        let capacity = usize::try_from(arena_bytes).map_err(|_| Error { op: OP })?;
        let mut arena = self.stream.alloc_managed::<u8>(capacity).map_err(fail)?;
        let mut placed: Vec<SupportRange> = Vec::with_capacity(ranges.len());
        let mut used = 0u64;
        for &(offset, len, label) in ranges {
            if len == 0 || placed.iter().any(|r| r.contains(offset, len)) {
                continue;
            }
            if offset > size || len > size - offset {
                return Err(Error {
                    op: "support range outside the file",
                });
            }
            let aligned = used.next_multiple_of(256);
            if aligned > arena_bytes || len > arena_bytes - aligned {
                eprintln!(
                    "ds4: ROCm support arena exhausted for {label} ({:.2} MiB requested, {:.2} MiB free)",
                    len as f64 / 1048576.0,
                    arena_bytes.saturating_sub(aligned) as f64 / 1048576.0
                );
                return Err(Error { op: OP });
            }
            let (start, file) = (aligned as usize, offset as usize);
            self.stream
                .upload(
                    &host[file..file + len as usize],
                    arena.as_mut_slice().range(start..start + len as usize),
                )
                .map_err(fail)?;
            placed.push(SupportRange {
                offset,
                bytes: len,
                arena: start,
            });
            used = aligned + len;
        }
        let arena = arena.freeze(&self.stream).map_err(fail)?;
        self.support = Some(map);
        self.support_arena = Some(arena);
        self.support_ranges = placed;
        Ok(())
    }

    /// The device bytes of `bytes` at `weight`, as the C++ runtime resolved
    /// them (`hip_model_range_ptr`): inside the target span that starts at
    /// or before it, or inside a support range. An empty range resolves to
    /// an empty slice.
    pub(crate) fn weight_bytes(
        &self,
        weight: crate::ops::Weight,
        bytes: u64,
    ) -> Result<DeviceSlice<'_, u8>> {
        let refuse = Error {
            op: "weight outside the resident ranges",
        };
        let end = weight.offset.checked_add(bytes).ok_or(refuse)?;
        match weight.model {
            Model::Target => {
                let index = self
                    .target_spans
                    .partition_point(|span| span.offset <= weight.offset);
                let span = index
                    .checked_sub(1)
                    .and_then(|i| self.target_spans.get(i))
                    .filter(|span| end <= span.offset + span.tensor.len() as u64)
                    .ok_or(refuse)?;
                let start = (weight.offset - span.offset) as usize;
                Ok(span.tensor.view().range(start..start + bytes as usize))
            }
            Model::Support => {
                let arena = self.support_arena.as_ref().ok_or(refuse)?;
                let range = self
                    .support_ranges
                    .iter()
                    .find(|r| r.contains(weight.offset, bytes))
                    .ok_or(refuse)?;
                let start = range.arena + (weight.offset - range.offset) as usize;
                Ok(arena.view().range(start..start + bytes as usize))
            }
        }
    }

    /// The stream the generated launchers order work on.
    pub(crate) fn stream(&self) -> &Stream {
        &self.stream
    }

    /// The registered mapping of `model`.
    pub(crate) fn mapped(&self, model: Model) -> Result<&Arc<MappedFile>> {
        match model {
            Model::Target => self.target.as_ref(),
            Model::Support => self.support.as_ref(),
        }
        .ok_or(Error {
            op: "weights of an unregistered model",
        })
    }

    /// A device allocation of `bytes` (at least one).
    pub fn alloc(&self, bytes: u64) -> Result<DeviceTensor> {
        const OP: &str = "tensor alloc";
        // As `ds4_gpu_tensor_alloc`: an exact allocation, one byte for none.
        let bytes = usize::try_from(bytes.max(1)).map_err(|_| Error { op: OP })?;
        self.stream
            .alloc_persistent::<u8>(bytes)
            .map(DeviceTensor::from_buffer)
            .map_err(|_| Error { op: OP })
    }

    /// A managed allocation of `bytes` (at least one), for KV caches the
    /// policy pages on demand ([`Self::should_use_managed_kv_cache`]).
    pub fn alloc_managed(&self, bytes: u64) -> Result<DeviceTensor> {
        const OP: &str = "managed tensor alloc";
        let bytes = usize::try_from(bytes.max(1)).map_err(|_| Error { op: OP })?;
        self.stream
            .alloc_managed::<u8>(bytes)
            .map(DeviceTensor::from_buffer)
            .map_err(|_| Error { op: OP })
    }

    /// `ds4_gpu_should_use_managed_kv_cache`: very large KV caches are
    /// where device-only allocations can make a unified-memory machine
    /// unresponsive, so they, and caches of large contexts that would leave
    /// less free memory than a quarter of the device (8 to 40 GiB), are
    /// demand-paged managed memory.
    pub fn should_use_managed_kv_cache(&self, kv_cache_bytes: u64, context_bytes: u64) -> bool {
        const GIB: u64 = 1 << 30;
        if kv_cache_bytes == 0 {
            return false;
        }
        if self.force_managed_kv.get() || kv_cache_bytes >= 8 * GIB {
            return true;
        }
        if context_bytes < 8 * GIB {
            return false;
        }
        let Ok((free, total)) = gufo_hip::device_memory() else {
            return false;
        };
        let (free, total) = (free as u64, total as u64);
        let reserve = (total / 4).clamp(8 * GIB, 40 * GIB);
        context_bytes > free || free - context_bytes < reserve
    }

    /// Allocates the KV caches of later sessions as managed memory whatever
    /// their size, as the policy does for very large contexts. For the tests
    /// of that route.
    pub fn force_managed_kv_cache(&self) {
        self.force_managed_kv.set(true);
    }

    /// Disables the Q8 to F16 weight cache for the rest of the registered
    /// model, as an allocation failure does: the Q8 routes run instead. For
    /// the tests that record and check those routes.
    pub fn disable_f16_weight_cache(&self) {
        self.projections.f16_weights.borrow_mut().disable();
    }

    /// Copies `data` into `tensor` at `offset`, after all queued work.
    pub fn write(&self, tensor: &mut impl Output, offset: u64, data: &[u8]) -> Result<()> {
        const OP: &str = "tensor write";
        let device = tensor.output();
        let range = byte_range(offset, data.len(), device.len()).ok_or(Error { op: OP })?;
        // The blocking path, as the C++ runtime copied: small writes and
        // reads per speculative cycle are slower as an asynchronous copy
        // plus a stream synchronisation.
        self.stream
            .upload_blocking(data, device.range(range))
            .map_err(|error| {
                eprintln!("ds4: ROCm tensor write failed: {error}");
                Error { op: OP }
            })
    }

    /// Reads `data.len()` bytes of `tensor` at `offset`, after all queued
    /// work.
    pub fn read(&self, tensor: &impl Input, offset: u64, data: &mut [u8]) -> Result<()> {
        const OP: &str = "tensor read";
        let device = tensor.input();
        let range = byte_range(offset, data.len(), device.len()).ok_or(Error { op: OP })?;
        self.stream
            .download_blocking(device.range(range), data)
            .map_err(|error| {
                eprintln!("ds4: ROCm tensor read failed: {error}");
                Error { op: OP }
            })
    }

    /// Reads `data.len()` floats of `tensor` at byte `offset`, after all
    /// queued work, through a page-locked staging buffer the backend keeps:
    /// a direct DMA copy, several times faster than one into pageable memory
    /// (the C++ runtime's), which the logits of every decode step pay.
    pub fn read_f32(&self, tensor: &impl Input, offset: u64, data: &mut [f32]) -> Result<()> {
        const OP: &str = "tensor read";
        let refuse = Error { op: OP };
        let bytes = std::mem::size_of_val(data) as u64;
        let source = tensor.input();
        let end = offset
            .checked_add(bytes)
            .filter(|&end| end <= source.len() as u64)
            .ok_or(refuse)?;
        let source = source
            .range(offset as usize..end as usize)
            .cast::<f32>()
            .ok_or(refuse)?;
        let mut staging = self.read_staging.borrow_mut();
        if staging
            .as_ref()
            .is_none_or(|pinned| pinned.len() < data.len())
        {
            *staging = None;
            *staging = Some(PinnedBuffer::zeroed(data.len()).map_err(|error| {
                eprintln!("ds4: ROCm pinned read buffer failed: {error}");
                refuse
            })?);
        }
        let pinned = staging.as_mut().ok_or(refuse)?;
        let copied = self
            .stream
            .download_async_front(source, pinned)
            .and_then(|()| {
                pinned
                    .host()
                    .map(|host| data.copy_from_slice(&host[..data.len()]))
            });
        copied.map_err(|error| {
            eprintln!("ds4: ROCm tensor read failed: {error}");
            refuse
        })
    }

    /// A reader of frozen sessions' tensors for snapshot captures on other
    /// threads.
    pub fn snapshot_reader(&self) -> SnapshotReader {
        SnapshotReader {
            transfers: Arc::clone(&self.snapshot_transfers),
        }
    }

    /// Copies `bytes` from `src` at `src_offset` to `dst` at `dst_offset`.
    /// Inside a command bracket the copy stays in stream order and the
    /// bracket's closing wait covers it; outside one it waits, as the C++
    /// runtime's copy did for every copy (snapshot reads on another stream
    /// rely on finished state between operations).
    pub fn copy(
        &self,
        dst: &mut impl Output,
        dst_offset: u64,
        src: &impl Input,
        src_offset: u64,
        bytes: u64,
    ) -> Result<()> {
        const OP: &str = "tensor copy";
        if bytes == 0 {
            return Ok(());
        }
        let range = |offset: u64, len: usize| {
            offset
                .checked_add(bytes)
                .filter(|&end| end <= len as u64)
                .map(|end| offset as usize..end as usize)
                .ok_or(Error { op: OP })
        };
        let source = src.input();
        let source = source.range(range(src_offset, source.len())?);
        let destination = dst.output();
        let len = destination.len();
        let destination = destination.range(range(dst_offset, len)?);
        let copied = self.stream.copy(source, destination).and_then(|()| {
            if self.in_commands.get() {
                Ok(())
            } else {
                self.stream.synchronize()
            }
        });
        copied.map_err(|error| {
            eprintln!("ds4: ROCm tensor copy failed: {error}");
            Error { op: OP }
        })
    }

    /// Sets the first `count` floats of `tensor` to `value`.
    pub fn fill_f32(&self, tensor: &mut impl Output, value: f32, count: u64) -> Result<()> {
        const OP: &str = "tensor fill f32";
        let count = usize::try_from(count).map_err(|_| Error { op: OP })?;
        crate::ops::launched(
            OP,
            crate::generated::fill_f32(
                &self.stream,
                crate::ops::typed_mut(OP, tensor.output())?,
                value,
                count,
            ),
        )
    }

    /// Waits for all queued work.
    pub fn synchronize(&self) -> Result<()> {
        self.in_commands.set(false);
        self.wait("synchronize")
    }

    /// Waits for the stream, which orders all work: the C++ runtime's
    /// device-wide wait covered nothing else.
    fn wait(&self, op: &'static str) -> Result<()> {
        self.stream.synchronize().map_err(|error| {
            eprintln!("ds4: ROCm {op} failed: {error}");
            Error { op }
        })
    }

    /// The batch routes of the next launches: off for prefill and decode,
    /// whose kernel choice stays as baselined, and the verifier's or the
    /// support model's while a DSpark block is encoded.
    pub fn set_batch_mode(&self, mode: BatchMode) {
        self.batch_mode.set(mode);
    }

    pub fn batch_mode(&self) -> BatchMode {
        self.batch_mode.get()
    }

    /// Opens a command bracket (nothing to enqueue on ROCm).
    pub fn begin_commands(&self) -> Result<()> {
        self.in_commands.set(true);
        Ok(())
    }

    /// Runs `work` as one command bracket: its copies stay in stream order
    /// and one wait at the end covers them; a failure still waits. Inside an
    /// open bracket `work` simply joins it.
    pub fn commands<T>(&self, work: impl FnOnce() -> Result<T>) -> Result<T> {
        if self.in_commands.get() {
            return work();
        }
        self.begin_commands()?;
        match work() {
            Ok(value) => self.end_commands().map(|()| value),
            Err(error) => {
                if self.synchronize().is_err() {
                    eprintln!("ds4: ROCm synchronize after a command failure also failed");
                }
                Err(error)
            }
        }
    }

    /// Closes a command bracket: waits for all queued work.
    pub fn end_commands(&self) -> Result<()> {
        self.in_commands.set(false);
        self.wait("end commands")
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        if self.stream.synchronize().is_err() {
            eprintln!("ds4: ROCm synchronize at cleanup failed");
        }
        // The MMQ tier's process-wide scratch (`ds4_gpu_cleanup`).
        if self.projections.mmq_ready.get() && crate::generated::mmq_release(&self.stream).is_err()
        {
            eprintln!("ds4: ROCm MMQ release failed");
        }
        self.target_spans.clear();
        self.support_ranges.clear();
        self.support_arena = None;
        ACTIVE.store(false, Ordering::Release);
    }
}

/// `offset .. offset + len` when it lies within `limit` bytes.
fn byte_range(offset: u64, len: usize, limit: usize) -> Option<std::ops::Range<usize>> {
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(len)?;
    (end <= limit).then_some(start..end)
}

/// The page-locked staging of one snapshot transfer.
const SNAPSHOT_STAGING: usize = 8 << 20;

/// A transfer stream and its page-locked staging buffer.
struct SnapshotTransfer {
    stream: Stream,
    staging: PinnedBuffer<u8>,
}

impl SnapshotTransfer {
    fn new() -> std::result::Result<Self, gufo_hip::HipError> {
        Ok(Self {
            stream: Stream::new()?,
            staging: PinnedBuffer::zeroed(SNAPSHOT_STAGING)?,
        })
    }

    /// Appends `spans` to `out`, one wait per staging buffer filled.
    fn read(
        &mut self,
        spans: &[DeviceSlice<'_, u8>],
        out: &mut Vec<u8>,
    ) -> std::result::Result<(), gufo_hip::HipError> {
        let capacity = self.staging.len();
        let mut group = Vec::new();
        let mut filled = 0;
        for &span in spans {
            let mut rest = span;
            while !rest.is_empty() {
                let take = rest.len().min(capacity - filled);
                group.push(rest.range(0..take));
                rest = rest.range(take..rest.len());
                filled += take;
                if filled == capacity {
                    self.drain(&mut group, out)?;
                    filled = 0;
                }
            }
        }
        self.drain(&mut group, out)
    }

    fn drain(
        &mut self,
        group: &mut Vec<DeviceSlice<'_, u8>>,
        out: &mut Vec<u8>,
    ) -> std::result::Result<(), gufo_hip::HipError> {
        let copied = self
            .stream
            .download_bytes_across_pinned(group, &mut self.staging)?;
        out.extend_from_slice(&self.staging.host()?[..copied]);
        group.clear();
        Ok(())
    }
}

/// Reads a completed session's tensors for a snapshot on any thread, while
/// other sessions run, on a transfer stream of its own (as
/// `ds4_gpu_tensor_snapshot_read` did). The scheduler freezes the session it
/// captures, so its tensors are complete and unchanged. The C++ read copied
/// each of a capture's 229 spans (22 MB in a 512-token context) to pageable
/// memory and waited for it, about 115 ms while the stream ran another
/// session's decode, 8 to 15 ms idle. The spans now queue into page-locked
/// staging, one wait per 8 MiB (7 to 17 ms either way), and transfers are
/// reused instead of a stream per capture thread.
#[derive(Clone)]
pub struct SnapshotReader {
    transfers: Arc<Mutex<Vec<SnapshotTransfer>>>,
}

impl SnapshotReader {
    /// Appends the bytes of `spans`, views of frozen session tensors, to
    /// `out` in order.
    pub fn read_spans(&self, spans: &[DeviceSlice<'_, u8>], out: &mut Vec<u8>) -> Result<()> {
        const OP: &str = "tensor snapshot read";
        let fail = |error: gufo_hip::HipError| {
            eprintln!("ds4: ROCm {OP} failed: {error}");
            Error { op: OP }
        };
        let idle = self
            .transfers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        let mut transfer = match idle {
            Some(transfer) => transfer,
            None => SnapshotTransfer::new().map_err(fail)?,
        };
        transfer.read(spans, out).map_err(fail)?;
        // A failed transfer is dropped rather than reused.
        self.transfers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(transfer);
        Ok(())
    }
}

#[cfg(test)]
mod placement_tests {
    use super::{DISPATCH_RESERVE, heap_fits_above_reserve};

    /// Weights go to the device heap only above the dispatch reserve
    /// (upstream `heap_placement_test`).
    #[test]
    fn weights_fit_the_heap_only_above_the_dispatch_reserve() {
        const GIB: u64 = 1 << 30;
        assert!(heap_fits_above_reserve(118 * GIB, 81 * GIB));
        assert!(heap_fits_above_reserve(85 * GIB, 81 * GIB));
        assert!(!heap_fits_above_reserve(85 * GIB - 1, 81 * GIB));
        assert!(!heap_fits_above_reserve(61 * GIB, 81 * GIB));
        assert!(!heap_fits_above_reserve(DISPATCH_RESERVE - 1, 0));
    }
}
