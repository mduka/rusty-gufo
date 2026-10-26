//! Snapshots that borrow the session's append-only rows: a capture freezes the mutable state (recurrent
//! state, convolution and PLE history) and the small ring, residual and
//! trunk-row regions in device storage, and leaves the K/V, pooled block and
//! draft K/V rows in the session, which only appends past them. Before a
//! write would overwrite borrowed rows (the pass after a rewind or reset, a
//! restore, the session's end) the rows are copied out once per set of
//! borrowing snapshots (`Preserve`); without device headroom the affected
//! snapshots move into host payloads instead. Byte readers materialize the
//! unchanged host payload. All of a session's snapshots live in its
//! registry, behind the one lock every reader and writer of their rows takes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use gufo_hip::{DeviceBuffer, DeviceSlice, SharedBytes, Stream, device_memory, huge_page_buffer};

use crate::snapshot::{HEADER_BYTES, Header};

/// Device memory every snapshot allocation leaves free beyond the
/// executor's lazy scratch, for the runtime and driver
/// (`kSnapshotDeviceSlackBytes`).
pub(crate) const DEVICE_SLACK_BYTES: usize = 512 << 20;
/// Released mutable-state blocks kept for the next captures: committing
/// device pages runs at about 25 GB/s and stalls other HIP calls
/// (`kSpareBlocks`).
pub(crate) const SPARE_BLOCKS: usize = 2;

/// The device storage of one executor's snapshots (`SnapshotAllocator`):
/// a transfer stream that owns every block, the spare mutable-state blocks
/// and the headroom every allocation leaves.
pub(crate) struct Pool {
    device: Mutex<Stream>,
    host: Mutex<Stream>,
    spares: Mutex<Vec<DeviceBuffer<u8>>>,
    reserve: usize,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

impl Pool {
    /// `scratch` is the device memory the executor may still allocate.
    pub fn new(scratch: usize) -> Result<Arc<Self>, String> {
        let stream = || Stream::new().map_err(|e| format!("hipStreamCreate: {e}"));
        Ok(Arc::new(Self {
            device: Mutex::new(stream()?),
            host: Mutex::new(stream()?),
            spares: Mutex::new(Vec::new()),
            reserve: scratch + DEVICE_SLACK_BYTES,
        }))
    }

    /// Whether allocating `bytes` keeps the reserve free. The snapshot
    /// budget counts host payloads, but an APU's device limit can be lower
    /// than its host RAM, and ROCm's pool corrupts its heap when a page
    /// commit fails: storage is never requested beyond the headroom.
    fn fits(&self, bytes: usize) -> bool {
        device_memory().is_ok_and(|(free, _)| free >= bytes && free - bytes >= self.reserve)
    }

    fn block(&self, stream: &Stream, bytes: usize) -> Option<DeviceBuffer<u8>> {
        if !self.fits(bytes) {
            return None;
        }
        stream.alloc::<u8>(bytes).ok()
    }

    /// A mutable-state block: a released one of the same size, else new.
    fn spare(&self, stream: &Stream, bytes: usize) -> Option<DeviceBuffer<u8>> {
        let mut spares = lock(&self.spares);
        if let Some(at) = spares.iter().position(|b| b.len() == bytes) {
            return Some(spares.swap_remove(at));
        }
        drop(spares);
        self.block(stream, bytes)
    }

    fn release(&self, buffer: DeviceBuffer<u8>) {
        let mut spares = lock(&self.spares);
        spares.retain(|b| b.len() == buffer.len());
        if spares.len() < SPARE_BLOCKS {
            spares.push(buffer);
        }
    }
}

/// One borrowed row region of a snapshot (`SnapshotState::Region`).
struct Region {
    /// The session buffer the rows live in.
    source: SharedBytes,
    /// The region's payload offset and length.
    offset: usize,
    bytes: usize,
    /// Bytes `[0, pending)` exist only in the session.
    pending: usize,
    /// Bytes `[0, live)` the session still holds unchanged.
    live: usize,
    row_bytes: usize,
    draft: bool,
    pooled: bool,
    /// Copies of the rows the session overwrote since.
    saved: Vec<Saved>,
}

/// Region bytes `[at, at + bytes)` at `block_at` of a preserved block.
struct Saved {
    at: usize,
    bytes: usize,
    block: Arc<DeviceBuffer<u8>>,
    block_at: usize,
}

/// Payload bytes `[offset, offset + bytes)` at `at` of a frozen block.
#[derive(Clone, Copy)]
pub(crate) struct Piece {
    pub offset: usize,
    pub bytes: usize,
    pub at: usize,
}

/// Trunk and draft rows a write keeps, and their pooled blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Keep {
    pub position: u32,
    pub mtp_position: u32,
    pub blocks: u32,
    pub mtp_blocks: u32,
}

impl Keep {
    fn bytes(&self, region: &Region) -> usize {
        let rows = match (region.draft, region.pooled) {
            (false, false) => self.position,
            (true, false) => self.mtp_position,
            (false, true) => self.blocks,
            (true, true) => self.mtp_blocks,
        };
        (rows as usize * region.row_bytes).min(region.bytes)
    }
}

/// A row region of a capture before it is registered.
pub(crate) struct RowSection {
    pub source: SharedBytes,
    pub offset: usize,
    pub bytes: usize,
    pub row_bytes: usize,
    pub draft: bool,
    pub pooled: bool,
}

/// A registered snapshot (`SnapshotState`).
pub(crate) struct Entry {
    pub header: Header,
    /// The image grids after the sections.
    pub grids: Vec<u8>,
    /// The payload's mutable prefix, frozen: its payload offset and block.
    mutable: Option<(usize, DeviceBuffer<u8>)>,
    /// The other small regions, frozen in one block.
    extras: Option<(DeviceBuffer<u8>, Vec<Piece>)>,
    regions: Vec<Region>,
    /// The host payload once materialized.
    payload: Option<Arc<Vec<u8>>>,
}

impl Entry {
    pub fn len(&self) -> usize {
        self.header.payload_bytes as usize
    }

    pub fn materialized(&self) -> Option<&Arc<Vec<u8>>> {
        self.payload.as_ref()
    }

    /// The frozen device copy of payload bytes `[offset, offset + len)` of
    /// the mutable prefix, when they lie in it.
    pub fn mutable(&self, offset: usize, len: usize) -> Option<DeviceSlice<'_, u8>> {
        let (at, block) = self.mutable.as_ref()?;
        let start = offset.checked_sub(*at)?;
        (start + len <= block.len()).then(|| block.slice(start..start + len))
    }

    /// The frozen pieces overlapping payload bytes `[offset, offset + len)`:
    /// (offset within the range, device bytes).
    pub fn extras(&self, offset: usize, len: usize) -> Vec<(usize, DeviceSlice<'_, u8>)> {
        let Some((block, pieces)) = &self.extras else {
            return Vec::new();
        };
        pieces
            .iter()
            .filter_map(|p| {
                let begin = offset.max(p.offset);
                let end = (offset + len).min(p.offset + p.bytes);
                (begin < end).then(|| {
                    let at = p.at + begin - p.offset;
                    (begin - offset, block.slice(at..at + end - begin))
                })
            })
            .collect()
    }

    fn region(&self, offset: usize) -> Option<&Region> {
        self.regions.iter().find(|r| r.offset == offset)
    }

    /// Rows of the region at payload `offset` the source still holds
    /// unchanged, as bytes.
    pub fn live_bytes(&self, offset: usize) -> Option<usize> {
        self.region(offset).map(|r| r.live)
    }

    /// The device copies of the region at payload `offset` from region byte
    /// `from` on: (region offset, bytes), the still-live rows from the
    /// source first, then preserved copies. `None` without such a region.
    pub fn rows(&self, offset: usize, from: usize) -> Option<Vec<(usize, DeviceSlice<'_, u8>)>> {
        let region = self.region(offset)?;
        let mut parts = Vec::new();
        if from < region.live {
            parts.push((from, region.source.slice(from..region.live)));
        }
        for saved in &region.saved {
            let begin = from.max(region.live).max(saved.at);
            let end = saved.at + saved.bytes;
            if begin < end {
                let at = saved.block_at + begin - saved.at;
                parts.push((begin, saved.block.slice(at..at + end - begin)));
            }
        }
        Some(parts)
    }

    /// The live rows a restore into the borrowed session keeps in place: the
    /// least over each kind of region, at most the snapshot's positions.
    pub fn kept(&self) -> Keep {
        let h = &self.header;
        let mut keep = Keep {
            position: h.position,
            mtp_position: h.mtp_position,
            blocks: h.blocks,
            mtp_blocks: h.mtp_blocks,
        };
        for region in &self.regions {
            let rows = (region.live / region.row_bytes) as u32;
            let slot = match (region.draft, region.pooled) {
                (false, false) => &mut keep.position,
                (true, false) => &mut keep.mtp_position,
                (false, true) => &mut keep.blocks,
                (true, true) => &mut keep.mtp_blocks,
            };
            *slot = (*slot).min(rows);
        }
        keep
    }

    fn release(&mut self, pool: &Pool) {
        if let Some((_, block)) = self.mutable.take() {
            pool.release(block);
        }
        self.extras = None;
        self.regions.clear();
    }

    /// Copies everything into the host payload and drops the device parts
    /// (`MaterializeLocked`).
    fn materialize(&mut self, pool: &Pool) -> Result<Arc<Vec<u8>>, String> {
        if let Some(payload) = &self.payload {
            return Ok(Arc::clone(payload));
        }
        let mut out = huge_page_buffer(self.len());
        self.header.write(&mut out);
        let grids = out.len() - self.grids.len();
        out[grids..].copy_from_slice(&self.grids);
        let stream = lock(&pool.host);
        let mut copy = |source: DeviceSlice<'_, u8>, at: usize| {
            stream
                .download_bytes_across(source, &mut out[at..at + source.len()])
                .map_err(|e| format!("snapshot materialize: {e}"))
        };
        if let Some((at, block)) = &self.mutable {
            copy(block.as_slice(), *at)?;
        }
        if let Some((block, pieces)) = &self.extras {
            for p in pieces {
                copy(block.slice(p.at..p.at + p.bytes), p.offset)?;
            }
        }
        for region in &self.regions {
            if region.pending != 0 {
                copy(region.source.slice(0..region.pending), region.offset)?;
            }
            for saved in &region.saved {
                let source = saved
                    .block
                    .slice(saved.block_at..saved.block_at + saved.bytes);
                copy(source, region.offset + saved.at)?;
            }
        }
        drop(stream);
        self.release(pool);
        let payload = Arc::new(out);
        self.payload = Some(Arc::clone(&payload));
        Ok(payload)
    }
}

/// The snapshots that borrow one session's rows (`SnapshotRegistry`).
pub(crate) struct Registry {
    /// Names the session's storage for the continuation cache.
    id: u64,
    pool: Arc<Pool>,
    state: Mutex<State>,
}

static NEXT_REGISTRY: AtomicU64 = AtomicU64::new(1);

pub(crate) struct State {
    entries: HashMap<u64, Entry>,
    next: u64,
    /// No snapshot borrows trunk or draft rows at or past these.
    position: u32,
    mtp_position: u32,
    /// Set when preserving failed: the borrowed rows are lost.
    failure: Option<String>,
}

impl State {
    pub fn entry(&self, id: u64) -> Option<&Entry> {
        self.entries.get(&id)
    }

    pub fn check(&self) -> Result<(), String> {
        self.failure.as_ref().map_or(Ok(()), |f| Err(f.clone()))
    }
}

impl Registry {
    pub fn new(pool: Arc<Pool>) -> Arc<Self> {
        Arc::new(Self {
            id: NEXT_REGISTRY.fetch_add(1, Ordering::Relaxed),
            pool,
            state: Mutex::new(State {
                entries: HashMap::new(),
                next: 0,
                position: 0,
                mtp_position: 0,
                failure: None,
            }),
        })
    }

    pub fn pool(&self) -> &Arc<Pool> {
        &self.pool
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn lock(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    /// Freezes a capture: `mutable` (payload offset, sources) into one block,
    /// `extras` (payload offset, source) into another, and registers the row
    /// sections as borrowed. `None` without device headroom; the caller then
    /// writes a host payload. The sources must be idle.
    pub fn capture(
        self: &Arc<Self>,
        header: Header,
        grids: Vec<u8>,
        mutable: Vec<(usize, DeviceSlice<'_, u8>)>,
        extras: Vec<(usize, DeviceSlice<'_, u8>)>,
        rows: Vec<RowSection>,
    ) -> Result<Option<Handle>, String> {
        let mut state = self.lock();
        state.check()?;
        let stream = lock(&self.pool.device);
        let mutable_at = mutable.first().map_or(HEADER_BYTES, |(at, _)| *at);
        let mutable_len: usize = mutable.iter().map(|(_, s)| s.len()).sum();
        let extras_len: usize = extras.iter().map(|(_, s)| s.len()).sum();
        let Some(mut mutable_block) = self.pool.spare(&stream, mutable_len.max(1)) else {
            return Ok(None);
        };
        let mut extras_block = if extras_len == 0 {
            None
        } else {
            match self.pool.block(&stream, extras_len) {
                Some(block) => Some(block),
                None => {
                    self.pool.release(mutable_block);
                    return Ok(None);
                }
            }
        };
        let mut copies = Vec::with_capacity(mutable.len() + extras.len());
        let mut rest = mutable_block.as_mut_slice();
        let mut expected = mutable_at;
        for (offset, source) in &mutable {
            if *offset != expected {
                return Err("snapshot mutable sections are not contiguous".into());
            }
            let (head, tail) = rest.split_at(source.len());
            copies.push((*source, head));
            rest = tail;
            expected += source.len();
        }
        let mut pieces = Vec::with_capacity(extras.len());
        if let Some(block) = extras_block.as_mut() {
            let mut rest = block.as_mut_slice();
            let mut at = 0;
            for (offset, source) in &extras {
                let (head, tail) = rest.split_at(source.len());
                copies.push((*source, head));
                rest = tail;
                pieces.push(Piece {
                    offset: *offset,
                    bytes: source.len(),
                    at,
                });
                at += source.len();
            }
        }
        stream
            .copy_across_all(&mut copies)
            .map_err(|e| format!("snapshot freeze: {e}"))?;
        drop(copies);
        drop(stream);
        let regions = rows
            .into_iter()
            .map(|r| Region {
                source: r.source,
                offset: r.offset,
                bytes: r.bytes,
                pending: r.bytes,
                live: r.bytes,
                row_bytes: r.row_bytes,
                draft: r.draft,
                pooled: r.pooled,
                saved: Vec::new(),
            })
            .collect();
        state.position = state.position.max(header.position);
        state.mtp_position = state.mtp_position.max(header.mtp_position);
        let id = state.next;
        state.next += 1;
        let len = header.payload_bytes as usize;
        state.entries.insert(
            id,
            Entry {
                header,
                grids,
                mutable: Some((mutable_at, mutable_block)),
                extras: extras_block.map(|block| (block, pieces)),
                regions,
                payload: None,
            },
        );
        Ok(Some(Handle {
            registry: Arc::clone(self),
            id,
            len,
        }))
    }

    /// Copies every borrowed row a write keeping only `keep` would overwrite
    /// (`SnapshotRegistry::Preserve`).
    pub fn preserve(&self, keep: Keep) -> Result<(), String> {
        let mut state = self.lock();
        self.preserve_locked(&mut state, keep)
    }

    /// [`preserve`](Self::preserve) under a held lock.
    pub fn preserve_locked(&self, state: &mut State, keep: Keep) -> Result<(), String> {
        state.check()?;
        if keep.position >= state.position && keep.mtp_position >= state.mtp_position {
            return Ok(());
        }
        let result = self.preserve_entries(state, keep);
        if let Err(e) = &result {
            // The rows may be half copied: no reader may trust them.
            state.failure = Some(e.clone());
        }
        result
    }

    fn preserve_entries(&self, state: &mut State, keep: Keep) -> Result<(), String> {
        /// Bytes `[begin, end)` of one source buffer and the regions
        /// (entry id, region index) that need them.
        struct Copy {
            source: SharedBytes,
            begin: usize,
            end: usize,
            owners: Vec<(u64, usize)>,
        }
        let mut ids: Vec<u64> = state.entries.keys().copied().collect();
        ids.sort_unstable();
        /// The overwritten ranges of one source buffer: the bytes kept and
        /// each borrowing region (entry id, region index, pending end).
        struct Group {
            source: SharedBytes,
            kept: usize,
            regions: Vec<(u64, usize, usize)>,
        }
        // Group the overwritten ranges by source buffer; split them at every
        // region's end, so each piece is copied once for all who need it.
        let mut groups: Vec<Group> = Vec::new();
        for &id in &ids {
            for (index, region) in state.entries[&id].regions.iter().enumerate() {
                let kept = keep.bytes(region);
                if kept >= region.pending {
                    continue;
                }
                match groups
                    .iter_mut()
                    .find(|g| g.source.same_memory(&region.source))
                {
                    Some(group) => group.regions.push((id, index, region.pending)),
                    None => groups.push(Group {
                        source: region.source.clone(),
                        kept,
                        regions: vec![(id, index, region.pending)],
                    }),
                }
            }
        }
        let mut copies = Vec::new();
        for Group {
            source,
            kept,
            regions,
        } in groups
        {
            let mut ends: Vec<usize> = regions.iter().map(|r| r.2).collect();
            ends.push(kept);
            ends.sort_unstable();
            ends.dedup();
            for pair in ends.windows(2) {
                let owners = regions
                    .iter()
                    .filter(|r| r.2 >= pair[1])
                    .map(|r| (r.0, r.1))
                    .collect();
                copies.push(Copy {
                    source: source.clone(),
                    begin: pair[0],
                    end: pair[1],
                    owners,
                });
            }
        }
        // One block per set of consumers: protecting a deep prefix takes a
        // few allocations, and dropping one snapshot frees its own rows.
        let consumers = |c: &Copy| {
            let mut ids: Vec<u64> = c.owners.iter().map(|o| o.0).collect();
            ids.dedup();
            ids
        };
        copies.sort_by_key(consumers);
        let mut sets: Vec<(Vec<u64>, usize)> = Vec::new();
        for copy in &copies {
            let set = consumers(copy);
            match sets.last_mut() {
                Some(last) if last.0 == set => last.1 += copy.end - copy.begin,
                _ => sets.push((set, copy.end - copy.begin)),
            }
        }
        let needed: usize = sets.iter().map(|s| s.1).sum();
        let stream = lock(&self.pool.device);
        let mut blocks = Vec::with_capacity(sets.len());
        if needed != 0 && self.pool.fits(needed) {
            for (_, bytes) in &sets {
                match stream.alloc::<u8>(*bytes) {
                    Ok(block) => blocks.push(block),
                    Err(_) => {
                        blocks.clear();
                        break;
                    }
                }
            }
        }
        if needed != 0 && blocks.len() != sets.len() {
            // No device room for the overwritten rows: move each affected
            // snapshot into its host payload, which the snapshot budget
            // already charges.
            drop(stream);
            let mut affected: Vec<u64> = copies.iter().flat_map(consumers).collect();
            affected.sort_unstable();
            affected.dedup();
            for id in affected {
                if let Some(entry) = state.entries.get_mut(&id) {
                    entry.materialize(&self.pool)?;
                }
            }
        } else if needed != 0 {
            // Enqueue every piece into its set's block, then wait once.
            let mut targets = Vec::with_capacity(copies.len());
            let mut set = 0;
            let mut used = 0;
            for copy in &copies {
                if consumers(copy) != sets[set].0 {
                    set += 1;
                    used = 0;
                }
                targets.push((set, used));
                used += copy.end - copy.begin;
            }
            {
                let mut views: Vec<_> = blocks.iter_mut().map(|b| Some(b.as_mut_slice())).collect();
                let mut pairs = Vec::with_capacity(copies.len());
                for (copy, &(set, _)) in copies.iter().zip(&targets) {
                    let view = views[set].take().ok_or("snapshot preserve block reused")?;
                    let (head, tail) = view.split_at(copy.end - copy.begin);
                    views[set] = Some(tail);
                    pairs.push((copy.source.slice(copy.begin..copy.end), head));
                }
                stream
                    .copy_across_all(&mut pairs)
                    .map_err(|e| format!("snapshot preserve: {e}"))?;
            }
            drop(stream);
            let blocks: Vec<Arc<DeviceBuffer<u8>>> = blocks.into_iter().map(Arc::new).collect();
            for (copy, &(set, at)) in copies.iter().zip(&targets) {
                for &(id, index) in &copy.owners {
                    if let Some(entry) = state.entries.get_mut(&id) {
                        entry.regions[index].saved.push(Saved {
                            at: copy.begin,
                            bytes: copy.end - copy.begin,
                            block: Arc::clone(&blocks[set]),
                            block_at: at,
                        });
                    }
                }
            }
        }
        for entry in state.entries.values_mut() {
            for region in &mut entry.regions {
                let kept = keep.bytes(region);
                region.pending = region.pending.min(kept);
                region.live = region.live.min(kept);
            }
        }
        state.position = state.position.min(keep.position);
        state.mtp_position = state.mtp_position.min(keep.mtp_position);
        Ok(())
    }
}

/// A registered snapshot; dropping it releases its storage.
pub(crate) struct Handle {
    registry: Arc<Registry>,
    id: u64,
    len: usize,
}

impl Handle {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn registry(&self) -> &Arc<Registry> {
        &self.registry
    }

    /// The complete host payload (`SessionSnapshot::bytes`): materialized
    /// once, under the lock writers take, so no row changes meanwhile.
    pub fn payload(&self) -> Result<Arc<Vec<u8>>, String> {
        let mut state = self.registry.lock();
        state.check()?;
        let entry = state
            .entries
            .get_mut(&self.id)
            .ok_or("snapshot is no longer registered")?;
        entry.materialize(&self.registry.pool)
    }

    /// Device bytes the snapshot holds, counting each shared block once
    /// (`DeviceBytes`).
    pub fn device_bytes(&self) -> usize {
        let state = self.registry.lock();
        let Some(entry) = state.entries.get(&self.id) else {
            return 0;
        };
        let mut blocks: Vec<*const DeviceBuffer<u8>> = Vec::new();
        let mut bytes = entry.mutable.as_ref().map_or(0, |(_, b)| b.len())
            + entry.extras.as_ref().map_or(0, |(b, _)| b.len());
        for region in &entry.regions {
            for saved in &region.saved {
                let block = Arc::as_ptr(&saved.block);
                if !blocks.contains(&block) {
                    blocks.push(block);
                    bytes += saved.block.len();
                }
            }
        }
        bytes
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        let mut state = self.registry.lock();
        if let Some(mut entry) = state.entries.remove(&self.id) {
            entry.release(&self.registry.pool);
        }
        // What the remaining snapshots still borrow bounds the next check.
        let (mut position, mut mtp_position) = (0, 0);
        for entry in state.entries.values() {
            if !entry.regions.is_empty() {
                position = position.max(entry.header.position);
                mtp_position = mtp_position.max(entry.header.mtp_position);
            }
        }
        state.position = position;
        state.mtp_position = mtp_position;
    }
}
