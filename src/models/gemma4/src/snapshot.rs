//! Session snapshots (`engine.cpp`: `SessionSnapshot`, `SettleHeld`,
//! `SnapshotBytes`, `SaveSnapshot`, `ReadHead`, `RestoreSnapshot`). A
//! snapshot holds a session's tokens, image records, the KV rows later
//! tokens can still attend (every global row, the last window-1 sliding
//! rows), the frontier hidden state and the last logits. Rows of whole
//! position blocks live in immutable device blocks that snapshots of one
//! session's history share, so a snapshot copies only the blocks its
//! session wrote since the last one and the rows around them; the session
//! holds its blocks weakly and a restore skips the ones its cache still
//! holds. The payload reads back in the layout [`Session::restore_payload`]
//! accepts (payload version 3, the C++'s).

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, Weak};

use gufo_hip::{DeviceTensor, Stream};

use crate::config::Config;
use crate::engine::{ImageSpan, Session};
use crate::executor::hip;

/// `kSnapshotMagic`.
const MAGIC: [u8; 8] = *b"G4SNAP01";
/// Bytes of the header: magic, then version, position, hidden, vocab,
/// layers and window as u32 (`SnapshotHeader`).
const HEADER_BYTES: usize = 32;
/// Image record: u32 offset, u32 rows, 32-byte identity.
const IMAGE_RECORD_BYTES: usize = 8 + 32;
/// Positions per snapshot block of the global and the sliding layers. Ring
/// sizes are multiples of both, so a block never wraps in a sliding layer's
/// ring.
const GLOBAL_BLOCK_ROWS: u32 = 256;
const SLIDING_BLOCK_ROWS: u32 = 64;
/// Device memory every snapshot allocation leaves free.
const DEVICE_RESERVE_BYTES: u64 = 1 << 30;
/// Host bytes one gather of `SessionSnapshot::stream` fills.
const STAGING_BYTES: usize = 64 << 20;

fn block_rows(sliding: bool) -> u32 {
    if sliding {
        SLIDING_BLOCK_ROWS
    } else {
        GLOBAL_BLOCK_ROWS
    }
}

/// First position whose KV a later token may attend in `layer`
/// (`FirstLiveRow`).
fn first_live_row(c: &Config, layer: u32, position: u32) -> u32 {
    if !c.is_sliding(layer) || position < c.sliding_window {
        return 0;
    }
    position - (c.sliding_window - 1)
}

/// The blocks of a snapshot at `n` (`BlockShape`): global layers' blocks
/// `[0, global)`, sliding layers' blocks `[sliding_first, sliding_end)`.
#[derive(Debug, Clone, Copy)]
struct Shape {
    n: u32,
    live: u32,
    global: u32,
    sliding_first: u32,
    sliding_end: u32,
}

fn shape_of(c: &Config, n: u32) -> Shape {
    let live = if n < c.sliding_window {
        0
    } else {
        n - (c.sliding_window - 1)
    };
    let sliding_first = live.div_ceil(SLIDING_BLOCK_ROWS);
    Shape {
        n,
        live,
        global: n / GLOBAL_BLOCK_ROWS,
        sliding_first,
        sliding_end: (n / SLIDING_BLOCK_ROWS).max(sliding_first),
    }
}

/// Byte offsets of each layer's K and V rows inside a block of its kind
/// (`BlockLayout`).
struct Layout {
    k_at: Vec<u64>,
    v_at: Vec<u64>,
    global_bytes: u64,
    sliding_bytes: u64,
}

fn layout_of(c: &Config, key_widths: &[u32]) -> Layout {
    let mut layout = Layout {
        k_at: Vec::new(),
        v_at: Vec::new(),
        global_bytes: 0,
        sliding_bytes: 0,
    };
    for l in 0..c.num_layers {
        let sliding = c.is_sliding(l);
        let rows = u64::from(block_rows(sliding));
        let bytes = if sliding {
            &mut layout.sliding_bytes
        } else {
            &mut layout.global_bytes
        };
        layout.k_at.push(*bytes);
        *bytes += rows * u64::from(key_widths[l as usize]) * 2;
        layout.v_at.push(*bytes);
        *bytes += rows * u64::from(c.kv_dim(l)) * 2;
    }
    layout
}

/// Positions `[first, first + rows)` of a layer's K or V rows, contiguous in
/// the cache and in the snapshot: in block `block` of the layer's kind, or
/// in the rest (`None`), at byte `offset` (`BodyRun`).
#[derive(Debug, Clone, Copy)]
struct BodyRun {
    layer: u32,
    values: bool,
    first: u32,
    rows: u32,
    row_bytes: u64,
    block: Option<u32>,
    offset: u64,
}

/// Visits a snapshot body in payload order: layer by layer, K rows then V
/// rows by position; returns the bytes of the rest rows (`VisitBody`).
fn visit_body(
    c: &Config,
    key_widths: &[u32],
    layout: &Layout,
    s: &Shape,
    ring: u32,
    mut visit: impl FnMut(&BodyRun) -> Result<(), String>,
) -> Result<u64, String> {
    let mut rest = 0u64;
    for l in 0..c.num_layers {
        let sliding = c.is_sliding(l);
        let rows = block_rows(sliding);
        let blocked_begin = if sliding { s.sliding_first * rows } else { 0 };
        let blocked_end = if sliding { s.sliding_end } else { s.global } * rows;
        for values in [false, true] {
            let row = u64::from(if values {
                c.kv_dim(l)
            } else {
                key_widths[l as usize]
            }) * 2;
            let mut p = if sliding { s.live } else { 0 };
            while p < s.n {
                let mut run = BodyRun {
                    layer: l,
                    values,
                    first: p,
                    rows: 0,
                    row_bytes: row,
                    block: None,
                    offset: rest,
                };
                let mut end = s.n;
                if p >= blocked_begin && p < blocked_end {
                    let block = p / rows;
                    run.block = Some(block);
                    end = (block + 1) * rows;
                    let base = if values {
                        layout.v_at[l as usize]
                    } else {
                        layout.k_at[l as usize]
                    };
                    run.offset = base + u64::from(p - block * rows) * row;
                } else if p < blocked_begin {
                    end = blocked_begin.min(s.n);
                }
                if sliding {
                    end = end.min((p / ring + 1) * ring);
                }
                run.rows = end - p;
                if run.block.is_none() {
                    rest += u64::from(run.rows) * row;
                }
                visit(&run)?;
                p = end;
            }
        }
    }
    Ok(rest)
}

/// One position block of a snapshot: `bytes` at `offset` of a shared
/// region.
#[derive(Clone)]
struct Block {
    region: Arc<DeviceTensor<u8>>,
    offset: usize,
}

/// A session's weak reference to a block its cache still holds.
#[derive(Clone, Default)]
pub(crate) struct HeldBlock {
    region: Weak<DeviceTensor<u8>>,
    offset: usize,
}

impl HeldBlock {
    fn of(block: &Block) -> Self {
        Self {
            region: Arc::downgrade(&block.region),
            offset: block.offset,
        }
    }

    fn get(&self) -> Option<Block> {
        self.region.upgrade().map(|region| Block {
            region,
            offset: self.offset,
        })
    }

    fn expired(&self) -> bool {
        self.region.strong_count() == 0
    }

    fn holds(&self, block: &Block) -> bool {
        self.get().is_some_and(|held| {
            Arc::ptr_eq(&held.region, &block.region) && held.offset == block.offset
        })
    }
}

/// The blocks a session's cache still holds, by position block.
#[derive(Default)]
pub(crate) struct Held {
    global: Vec<HeldBlock>,
    sliding: Vec<HeldBlock>,
}

/// Device bytes of a snapshot body, in payload order.
#[derive(Clone)]
struct Piece {
    region: Arc<DeviceTensor<u8>>,
    offset: usize,
    bytes: usize,
}

/// Snapshot memory of one model: the private stream that gathers payloads
/// to the host, created on first use (`SnapshotMemory`).
#[derive(Default)]
pub(crate) struct SnapshotMemory {
    stream: Mutex<Option<Stream>>,
}

/// A device-resident copy of a session's context (`SessionSnapshot`). Its
/// payload reads back in the layout [`Session::restore_payload`] accepts.
pub struct SessionSnapshot {
    /// Header, tokens, image records.
    head: Vec<u8>,
    global: Vec<Block>,
    sliding: Vec<Block>,
    sliding_first: u32,
    /// The rows outside the blocks, then the hidden state.
    rest: Arc<DeviceTensor<u8>>,
    body: Vec<Piece>,
    body_bytes: u64,
    /// Logits.
    tail: Vec<u8>,
    memory: Arc<SnapshotMemory>,
}

impl SessionSnapshot {
    pub fn size_bytes(&self) -> u64 {
        self.head.len() as u64 + self.body_bytes + self.tail.len() as u64
    }

    /// Writes the payload; fails on a size mismatch or a failed device copy.
    pub fn copy_to(&self, destination: &mut [u8]) -> Result<(), String> {
        if destination.len() as u64 != self.size_bytes() {
            return Err("snapshot payload size mismatch".into());
        }
        let mut at = 0;
        self.stream(&mut |piece| {
            destination[at..at + piece.len()].copy_from_slice(piece);
            at += piece.len();
        })
    }

    /// Hands the payload to `sink` in order, in pieces of at most 64 MiB;
    /// the body gathers on the model's private snapshot stream.
    pub fn stream(&self, sink: &mut dyn FnMut(&[u8])) -> Result<(), String> {
        let mut guard = self.memory.stream.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some(Stream::new().map_err(hip)?);
        }
        let stream = guard.as_ref().ok_or("snapshot stream")?;
        sink(&self.head);
        let mut staging = vec![0u8; STAGING_BYTES.min(self.body_bytes as usize).max(1)];
        let mut filled = 0usize;
        for piece in &self.body {
            let mut done = 0;
            while done < piece.bytes {
                let bytes = (piece.bytes - done).min(staging.len() - filled);
                let at = piece.offset + done;
                stream
                    .download(
                        piece.region.slice(at..at + bytes),
                        &mut staging[filled..filled + bytes],
                    )
                    .map_err(|e| format!("snapshot copy failed: {e}"))?;
                filled += bytes;
                done += bytes;
                if filled == staging.len() {
                    sink(&staging[..filled]);
                    filled = 0;
                }
            }
        }
        if filled != 0 {
            sink(&staging[..filled]);
        }
        sink(&self.tail);
        Ok(())
    }
}

/// Device memory for snapshot regions, refused when it would leave less
/// than the reserve free (`SnapshotMemory::Allocate`).
fn allocate(stream: &Stream, bytes: usize) -> Result<gufo_hip::DeviceBuffer<u8>, String> {
    let (free, _) = gufo_hip::device_memory().map_err(hip)?;
    let free = free as u64;
    if free < bytes as u64 || free - (bytes as u64) < DEVICE_RESERVE_BYTES {
        return Err("snapshot allocation failed".into());
    }
    stream
        .alloc_persistent::<u8>(bytes.max(1))
        .map_err(|_| "snapshot allocation failed".to_string())
}

impl Session {
    /// Drops held blocks whose rows the cache rewrote since the last call
    /// (`SettleHeld`).
    fn settle_held(&self, held: &mut Held) {
        let from = u64::from(self.cache.written_from.load(Ordering::Relaxed));
        let to = u64::from(self.cache.written_to.load(Ordering::Relaxed));
        let ring = u64::from(self.cache.ring);
        let settle = |list: &mut Vec<HeldBlock>, sliding: bool| {
            let rows = u64::from(block_rows(sliding));
            for (i, block) in list.iter_mut().enumerate() {
                let i = i as u64;
                // Rewritten, or (sliding) a write a ring further on took a
                // slot.
                if (i + 1) * rows > from || (sliding && i * rows + ring < to) {
                    *block = HeldBlock::default();
                }
            }
            while list.last().is_some_and(HeldBlock::expired) {
                list.pop();
            }
        };
        settle(&mut held.global, false);
        settle(&mut held.sliding, true);
        self.cache.written_from.store(u32::MAX, Ordering::Relaxed);
        self.cache.written_to.store(0, Ordering::Relaxed);
    }

    /// Payload bytes of a snapshot of the session's first `position` tokens
    /// (with the images inside them), whether or not the session is there
    /// yet: what a checkpoint at that position will take.
    pub fn snapshot_bytes_at(&self, position: u32) -> u64 {
        let c = self.model.config();
        let images = self
            .images
            .iter()
            .filter(|image| image.offset as u64 + u64::from(image.rows) <= u64::from(position))
            .count();
        let exec = self.model.exec();
        let key_widths = exec.widths();
        let mut bytes = HEADER_BYTES as u64
            + u64::from(position) * 4
            + 4
            + (images * IMAGE_RECORD_BYTES) as u64;
        for l in 0..c.num_layers {
            bytes += u64::from(position - first_live_row(c, l, position))
                * u64::from(key_widths[l as usize] + c.kv_dim(l))
                * 2;
        }
        bytes + u64::from(c.hidden_size) * 4 + u64::from(self.model.vocab_size()) * 4
    }

    /// Payload bytes of a snapshot of the session now (`SnapshotBytes`).
    pub fn snapshot_bytes(&self) -> u64 {
        self.snapshot_bytes_at(self.tokens.len() as u32)
    }

    /// Captures the session (`SaveSnapshot`); only an evaluated session can
    /// be saved.
    pub fn save_snapshot(&self) -> Result<SessionSnapshot, String> {
        if !self.valid || self.tokens.is_empty() {
            return Err("only an evaluated session can be saved".into());
        }
        let model = &self.model;
        let c = model.config();
        let n = self.tokens.len() as u32;
        let mut head = Vec::with_capacity(
            HEADER_BYTES + self.tokens.len() * 4 + 4 + self.images.len() * IMAGE_RECORD_BYTES,
        );
        head.extend_from_slice(&MAGIC);
        for value in [
            Session::SNAPSHOT_PAYLOAD_VERSION,
            n,
            c.hidden_size,
            model.vocab_size(),
            c.num_layers,
            c.sliding_window,
        ] {
            head.extend_from_slice(&value.to_le_bytes());
        }
        for token in &self.tokens {
            head.extend_from_slice(&token.to_le_bytes());
        }
        head.extend_from_slice(&(self.images.len() as u32).to_le_bytes());
        for image in &self.images {
            head.extend_from_slice(&image.offset.to_le_bytes());
            head.extend_from_slice(&image.rows.to_le_bytes());
            head.extend_from_slice(&image.identity);
        }
        let exec = model.exec();
        let logits: Vec<f32> = if self.logits_on_device {
            let mut out = Vec::new();
            exec.copy_cache_logits(&self.cache, &mut out)?;
            out
        } else {
            self.logits.clone()
        };
        let tail: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let key_widths = exec.widths().to_vec();
        let layout = layout_of(c, &key_widths);
        let shape = shape_of(c, n);
        let ring = self.cache.ring;
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        self.settle_held(&mut held);
        // Blocks the cache still holds are shared; the others are copied,
        // each kind's new blocks into one region.
        let gather = |held: &[HeldBlock], first: u32, count: u32| -> (Vec<Option<Block>>, usize) {
            let mut blocks = Vec::with_capacity(count as usize);
            let mut missing = 0;
            for i in 0..count {
                let block = held.get((first + i) as usize).and_then(HeldBlock::get);
                missing += usize::from(block.is_none());
                blocks.push(block);
            }
            (blocks, missing)
        };
        let (global, global_missing) = gather(&held.global, 0, shape.global);
        let (sliding, sliding_missing) = gather(
            &held.sliding,
            shape.sliding_first,
            shape.sliding_end - shape.sliding_first,
        );
        let hidden_bytes = u64::from(c.hidden_size) * 4;
        let rest_bytes = visit_body(c, &key_widths, &layout, &shape, ring, |_| Ok(()))?;
        let stream = exec.stream();
        let mut rest = allocate(stream, (rest_bytes + hidden_bytes) as usize)?;
        let mut global_region = if global_missing != 0 {
            Some(allocate(
                stream,
                global_missing * layout.global_bytes as usize,
            )?)
        } else {
            None
        };
        let mut sliding_region = if sliding_missing != 0 {
            Some(allocate(
                stream,
                sliding_missing * layout.sliding_bytes as usize,
            )?)
        } else {
            None
        };
        // Where each fresh block lands in its region.
        let index_of = |blocks: &[Option<Block>]| -> Vec<Option<usize>> {
            let mut next = 0;
            blocks
                .iter()
                .map(|b| {
                    if b.is_none() {
                        next += 1;
                        Some(next - 1)
                    } else {
                        None
                    }
                })
                .collect()
        };
        let fresh_global = index_of(&global);
        let fresh_sliding = index_of(&sliding);
        // Body pieces: (region kind, offset, bytes); kinds 0 rest, 1 a fresh
        // global block, 2 a fresh sliding block, 3 a shared block.
        enum At {
            Rest(usize),
            FreshGlobal(usize),
            FreshSliding(usize),
            Shared(Block),
        }
        let mut pieces: Vec<(At, usize)> = Vec::new();
        visit_body(c, &key_widths, &layout, &shape, ring, |r| {
            let sliding_layer = c.is_sliding(r.layer);
            let bytes = (u64::from(r.rows) * r.row_bytes) as usize;
            let cache_buffer = if r.values {
                &self.cache.v[r.layer as usize]
            } else {
                &self.cache.k[r.layer as usize]
            };
            let slot = if sliding_layer {
                r.first % ring
            } else {
                r.first
            };
            let from = slot as usize * r.row_bytes as usize;
            let source = cache_buffer
                .as_slice()
                .cast::<u8>()
                .ok_or("snapshot cache alignment")?
                .range(from..from + bytes);
            let at = match r.block {
                None => {
                    let to = r.offset as usize;
                    stream
                        .copy(source, rest.slice_mut(to..to + bytes))
                        .map_err(hip)?;
                    At::Rest(to)
                }
                Some(block) if sliding_layer => {
                    let i = (block - shape.sliding_first) as usize;
                    match fresh_sliding[i] {
                        Some(k) => {
                            let to = k * layout.sliding_bytes as usize + r.offset as usize;
                            let region = sliding_region.as_mut().ok_or("snapshot region")?;
                            stream
                                .copy(source, region.slice_mut(to..to + bytes))
                                .map_err(hip)?;
                            At::FreshSliding(to)
                        }
                        None => {
                            let b = sliding[i].clone().ok_or("snapshot block")?;
                            At::Shared(Block {
                                offset: b.offset + r.offset as usize,
                                region: b.region,
                            })
                        }
                    }
                }
                Some(block) => {
                    let i = block as usize;
                    match fresh_global[i] {
                        Some(k) => {
                            let to = k * layout.global_bytes as usize + r.offset as usize;
                            let region = global_region.as_mut().ok_or("snapshot region")?;
                            stream
                                .copy(source, region.slice_mut(to..to + bytes))
                                .map_err(hip)?;
                            At::FreshGlobal(to)
                        }
                        None => {
                            let b = global[i].clone().ok_or("snapshot block")?;
                            At::Shared(Block {
                                offset: b.offset + r.offset as usize,
                                region: b.region,
                            })
                        }
                    }
                }
            };
            pieces.push((at, bytes));
            Ok(())
        })?;
        let hidden_at = rest_bytes as usize;
        stream
            .copy(
                self.cache
                    .hidden
                    .slice(0..c.hidden_size as usize)
                    .cast::<u8>()
                    .ok_or("hidden alignment")?,
                rest.slice_mut(hidden_at..hidden_at + hidden_bytes as usize),
            )
            .map_err(hip)?;
        // Freezing waits for the copies.
        let rest = Arc::new(
            rest.freeze(stream)
                .map_err(|e| format!("snapshot copy failed: {e}"))?,
        );
        let global_region = match global_region {
            Some(r) => Some(Arc::new(
                r.freeze(stream)
                    .map_err(|e| format!("snapshot copy failed: {e}"))?,
            )),
            None => None,
        };
        let sliding_region = match sliding_region {
            Some(r) => Some(Arc::new(
                r.freeze(stream)
                    .map_err(|e| format!("snapshot copy failed: {e}"))?,
            )),
            None => None,
        };
        drop(exec);
        let complete = |blocks: Vec<Option<Block>>,
                        fresh: &[Option<usize>],
                        region: &Option<Arc<DeviceTensor<u8>>>,
                        bytes: u64|
         -> Result<Vec<Block>, String> {
            blocks
                .into_iter()
                .zip(fresh)
                .map(|(b, f)| match (b, f) {
                    (Some(b), _) => Ok(b),
                    (None, Some(k)) => Ok(Block {
                        region: Arc::clone(region.as_ref().ok_or("snapshot region")?),
                        offset: k * bytes as usize,
                    }),
                    (None, None) => Err("snapshot block".to_string()),
                })
                .collect()
        };
        let global = complete(global, &fresh_global, &global_region, layout.global_bytes)?;
        let sliding = complete(
            sliding,
            &fresh_sliding,
            &sliding_region,
            layout.sliding_bytes,
        )?;
        let mut body: Vec<Piece> = Vec::new();
        let mut body_bytes = 0u64;
        let mut add_piece = |region: &Arc<DeviceTensor<u8>>, offset: usize, bytes: usize| {
            if let Some(last) = body.last_mut()
                && Arc::ptr_eq(&last.region, region)
                && last.offset + last.bytes == offset
            {
                last.bytes += bytes;
            } else {
                body.push(Piece {
                    region: Arc::clone(region),
                    offset,
                    bytes,
                });
            }
            body_bytes += bytes as u64;
        };
        for (at, bytes) in pieces {
            match at {
                At::Rest(o) => add_piece(&rest, o, bytes),
                At::FreshGlobal(o) => {
                    add_piece(global_region.as_ref().ok_or("snapshot region")?, o, bytes)
                }
                At::FreshSliding(o) => {
                    add_piece(sliding_region.as_ref().ok_or("snapshot region")?, o, bytes)
                }
                At::Shared(b) => add_piece(&b.region, b.offset, bytes),
            }
        }
        add_piece(&rest, hidden_at, hidden_bytes as usize);
        let new_global = shape.global as usize;
        if held.global.len() < new_global {
            held.global.resize(new_global, HeldBlock::default());
        }
        if held.sliding.len() < shape.sliding_end as usize {
            held.sliding
                .resize(shape.sliding_end as usize, HeldBlock::default());
        }
        for (i, block) in global.iter().enumerate() {
            held.global[i] = HeldBlock::of(block);
        }
        for (i, block) in sliding.iter().enumerate() {
            held.sliding[shape.sliding_first as usize + i] = HeldBlock::of(block);
        }
        Ok(SessionSnapshot {
            head,
            global,
            sliding,
            sliding_first: shape.sliding_first,
            rest,
            body,
            body_bytes,
            tail,
            memory: Arc::clone(&model.snapshot_memory),
        })
    }

    /// Reads a payload head of a `total`-byte payload into the tokens and
    /// images; returns its size (`ReadHead`).
    fn read_head(&mut self, head: &[u8], total: u64) -> Result<usize, String> {
        self.reset();
        let model = Arc::clone(&self.model);
        let c = model.config();
        if head.len() < HEADER_BYTES {
            return Err("snapshot is truncated".into());
        }
        let word =
            |at: usize| u32::from_le_bytes([head[at], head[at + 1], head[at + 2], head[at + 3]]);
        if head[..8] != MAGIC
            || word(8) != Session::SNAPSHOT_PAYLOAD_VERSION
            || word(16) != c.hidden_size
            || word(20) != model.vocab_size()
            || word(24) != c.num_layers
            || word(28) != c.sliding_window
        {
            return Err("snapshot belongs to another model or version".into());
        }
        let n = word(12);
        if n == 0 || n > self.cache.max_context {
            return Err("snapshot position exceeds the session context".into());
        }
        let images_at = HEADER_BYTES + n as usize * 4;
        if head.len() < images_at + 4 {
            self.reset();
            return Err("snapshot size does not match its header".into());
        }
        let image_count = word(images_at) as usize;
        if image_count > (head.len() - images_at - 4) / IMAGE_RECORD_BYTES {
            self.reset();
            return Err("snapshot size does not match its header".into());
        }
        self.tokens = (0..n as usize)
            .map(|i| word(HEADER_BYTES + i * 4) as i32)
            .collect();
        self.images = vec![ImageSpan::default(); image_count];
        if total != self.snapshot_bytes() {
            self.reset();
            return Err("snapshot size does not match its header".into());
        }
        let mut at = images_at + 4;
        let mut previous_end = 0usize;
        for i in 0..image_count {
            let offset = word(at);
            let rows = word(at + 4);
            let mut identity = [0u8; 32];
            identity.copy_from_slice(&head[at + 8..at + 40]);
            at += IMAGE_RECORD_BYTES;
            if rows == 0
                || (offset as usize) < previous_end
                || offset as usize + rows as usize >= n as usize
            {
                self.reset();
                return Err("snapshot image placement is invalid".into());
            }
            previous_end = offset as usize + rows as usize;
            self.images[i] = ImageSpan {
                offset,
                rows,
                identity,
            };
        }
        Ok(images_at + 4 + image_count * IMAGE_RECORD_BYTES)
    }

    pub fn restore_snapshot(&mut self, snapshot: &SessionSnapshot) -> Result<(), String> {
        let head_bytes = self.read_head(&snapshot.head, snapshot.size_bytes())?;
        let model = Arc::clone(&self.model);
        let c = model.config();
        let n = self.tokens.len() as u32;
        let shape = shape_of(c, n);
        if head_bytes != snapshot.head.len()
            || snapshot.global.len() != shape.global as usize
            || snapshot.sliding_first != shape.sliding_first
            || snapshot.sliding.len() != (shape.sliding_end - shape.sliding_first) as usize
        {
            self.reset();
            return Err("snapshot size does not match its header".into());
        }
        let result = self.restore_blocks(snapshot, &shape);
        if let Err(e) = result {
            self.reset();
            return Err(e);
        }
        self.logits = snapshot
            .tail
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        self.logits_on_device = false;
        self.valid = true;
        Ok(())
    }

    fn restore_blocks(&mut self, snapshot: &SessionSnapshot, shape: &Shape) -> Result<(), String> {
        let model = Arc::clone(&self.model);
        let c = model.config();
        let exec = model.exec();
        let key_widths = exec.widths().to_vec();
        let layout = layout_of(c, &key_widths);
        let ring = self.cache.ring;
        let mut held = std::mem::take(&mut *self.held.lock().unwrap_or_else(|e| e.into_inner()));
        self.settle_held(&mut held);
        let stream = exec.stream();
        // Blocks the cache still holds keep their rows.
        let holds = |i: u32, sliding: bool| -> bool {
            if sliding {
                let local = (i - shape.sliding_first) as usize;
                held.sliding
                    .get(i as usize)
                    .is_some_and(|h| h.holds(&snapshot.sliding[local]))
            } else {
                held.global
                    .get(i as usize)
                    .is_some_and(|h| h.holds(&snapshot.global[i as usize]))
            }
        };
        let cache = &mut self.cache;
        let rest_bytes = visit_body(c, &key_widths, &layout, shape, ring, |r| {
            let sliding = c.is_sliding(r.layer);
            let bytes = (u64::from(r.rows) * r.row_bytes) as usize;
            let (region, from) = match r.block {
                None => (&snapshot.rest, r.offset as usize),
                Some(block) => {
                    if holds(block, sliding) {
                        return Ok(());
                    }
                    let b = if sliding {
                        &snapshot.sliding[(block - shape.sliding_first) as usize]
                    } else {
                        &snapshot.global[block as usize]
                    };
                    (&b.region, b.offset + r.offset as usize)
                }
            };
            let buffer = if r.values {
                &mut cache.v[r.layer as usize]
            } else {
                &mut cache.k[r.layer as usize]
            };
            let slot = if sliding { r.first % ring } else { r.first };
            let to = slot as usize * r.row_bytes as usize;
            let destination = buffer
                .as_mut_slice()
                .cast::<u8>()
                .ok_or("snapshot cache alignment")?
                .range(to..to + bytes);
            stream
                .copy(region.slice(from..from + bytes), destination)
                .map_err(hip)
        })?;
        let hidden = c.hidden_size as usize * 4;
        let at = rest_bytes as usize;
        stream
            .copy(
                snapshot.rest.slice(at..at + hidden),
                cache
                    .hidden
                    .as_mut_slice()
                    .cast::<u8>()
                    .ok_or("hidden alignment")?
                    .range(0..hidden),
            )
            .map_err(hip)?;
        stream
            .synchronize()
            .map_err(|e| format!("snapshot restore failed: {e}"))?;
        held.global = snapshot.global.iter().map(HeldBlock::of).collect();
        held.sliding = vec![HeldBlock::default(); shape.sliding_end as usize];
        for (i, block) in snapshot.sliding.iter().enumerate() {
            held.sliding[shape.sliding_first as usize + i] = HeldBlock::of(block);
        }
        *self.held.lock().unwrap_or_else(|e| e.into_inner()) = held;
        Ok(())
    }

    pub fn restore_payload(&mut self, payload: &[u8]) -> Result<(), String> {
        let head_bytes = self.read_head(payload, payload.len() as u64)?;
        let result = self.restore_rows(payload, head_bytes);
        if let Err(e) = result {
            self.reset();
            return Err(e);
        }
        let vocab = self.model.vocab_size() as usize;
        self.logits = payload[payload.len() - vocab * 4..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        self.logits_on_device = false;
        self.valid = true;
        Ok(())
    }

    fn restore_rows(&mut self, payload: &[u8], head_bytes: usize) -> Result<(), String> {
        let model = Arc::clone(&self.model);
        let c = model.config();
        let n = self.tokens.len() as u32;
        let exec = model.exec();
        let key_widths = exec.widths().to_vec();
        let stream = exec.stream();
        *self.held.lock().unwrap_or_else(|e| e.into_inner()) = Held::default();
        let mut body = head_bytes;
        let ring = self.cache.ring;
        let cache = &mut self.cache;
        for l in 0..c.num_layers {
            let first = first_live_row(c, l, n);
            let sliding = c.is_sliding(l);
            for values in [false, true] {
                let width = if values {
                    c.kv_dim(l)
                } else {
                    key_widths[l as usize]
                };
                let row = width as usize * 2;
                let buffer = if values {
                    &mut cache.v[l as usize]
                } else {
                    &mut cache.k[l as usize]
                };
                let mut p = first;
                while p < n {
                    let slot = if sliding { p % ring } else { p };
                    let run = if sliding {
                        (n - p).min(ring - slot)
                    } else {
                        n - p
                    };
                    let bytes = run as usize * row;
                    let to = slot as usize * row;
                    let destination = buffer
                        .as_mut_slice()
                        .cast::<u8>()
                        .ok_or("snapshot cache alignment")?
                        .range(to..to + bytes);
                    stream
                        .upload_blocking(&payload[body..body + bytes], destination)
                        .map_err(|e| format!("snapshot restore failed: {e}"))?;
                    body += bytes;
                    p += run;
                }
            }
        }
        let hidden = c.hidden_size as usize * 4;
        stream
            .upload_blocking(
                &payload[body..body + hidden],
                cache
                    .hidden
                    .as_mut_slice()
                    .cast::<u8>()
                    .ok_or("hidden alignment")?
                    .range(0..hidden),
            )
            .map_err(|e| format!("snapshot restore failed: {e}"))?;
        stream
            .synchronize()
            .map_err(|e| format!("snapshot restore failed: {e}"))
    }
}
