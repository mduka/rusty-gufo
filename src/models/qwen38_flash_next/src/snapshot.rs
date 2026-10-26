//! A session's complete context as one host byte payload, byte-identical to
//! the C++ executor's (`SaveSnapshot`, `RestoreSnapshot`, `WalkSnapshot` in
//! `kernels/rocm/executor.cpp`): a fixed header with every geometry value
//! the sections derive from, then recurrent and PLE state, KV and indexer
//! caches up to the position, and the draft block's caches, residual and
//! kept trunk rows.

use gufo_hip::{DeviceSlice, DeviceSliceMut, F16, HipError};

use std::sync::{Arc, Mutex};

use gufo_hip::Stream;

use gufo_qwen::vision::{ImageGrid, RopeLayout};

use crate::borrowed::{Handle, Keep, Pool, Registry, RowSection};
use crate::checkpoint::{CheckpointStorage, PrefillCheckpoint};
use crate::config::Config;
use crate::executor::Executor;
use crate::state::Session;

const MAGIC: [u8; 8] = *b"QFNSNAP6";
/// `sizeof(SnapshotHeader)`: 8 magic bytes, 17 `u32`s, the rope scaling
/// (factor, original context, beta fast and slow), two `i32`s of n-gram
/// history, 4 padding bytes, then the `u64` payload size.
pub const HEADER_BYTES: usize = 112;
/// One image grid after the sections (`vision::ImageGrid`: offset, height,
/// width as `u32`).
pub(crate) const GRID_BYTES: usize = 12;

/// Where a snapshot was taken (`SnapshotInfo`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SnapshotInfo {
    pub position: u32,
    pub hidden_rows: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Header {
    pub(crate) num_layers: u32,
    pub(crate) full_attention_interval: u32,
    pub(crate) conv_elems: u32,
    pub(crate) state_elems: u32,
    pub(crate) kv_row: u32,
    pub(crate) indexer_head_dim: u32,
    pub(crate) compress_ratio: u32,
    pub(crate) hc_dim: u32,
    pub(crate) ple_elems: u32,
    pub(crate) has_mtp: u32,
    pub(crate) position: u32,
    pub(crate) blocks: u32,
    pub(crate) mtp_position: u32,
    pub(crate) mtp_blocks: u32,
    /// Whether the draft residual is saved (a full draft forward ran last).
    pub(crate) mtp_residual_valid: u32,
    pub(crate) hidden_rows: u32,
    pub(crate) image_count: u32,
    /// The normalized rope scaling rotated K was written with: the factor,
    /// the original context and the two betas, the floats as their bits.
    pub(crate) rope: [u32; 4],
    pub(crate) ngram_prev: [i32; 2],
    pub(crate) payload_bytes: u64,
}

impl Header {
    /// This executor's geometry and the session's position
    /// (`MakeSnapshotHeader`).
    fn new(c: &Config, session: &Session) -> Self {
        Self {
            num_layers: c.num_layers,
            full_attention_interval: c.full_attention_interval,
            conv_elems: (c.ssm_conv_kernel - 1) * c.ssm_conv_channels(),
            state_elems: c.ssm_num_v_heads * c.ssm_head_dim * c.ssm_head_dim,
            kv_row: c.attention_kv_dim(),
            indexer_head_dim: c.indexer_head_dim,
            compress_ratio: c.compress_ratio,
            hc_dim: c.hc_dim(),
            ple_elems: if c.ple_layer.is_some() {
                c.ple_conv_history() * c.hc_dim()
            } else {
                0
            },
            has_mtp: u32::from(session.mtp_enabled),
            position: session.position,
            // Only images the computed context consumed; the attached
            // request may append more.
            image_count: session
                .vision
                .layout()
                .images
                .iter()
                .filter(|image| image.offset < session.position)
                .count() as u32,
            // Disabled scaling normalizes to the defaults, so two disabled
            // configurations share snapshots.
            rope: {
                let r = c.rope_scaling.normalized();
                [
                    r.factor.to_bits(),
                    r.original_context,
                    r.beta_fast.to_bits(),
                    r.beta_slow.to_bits(),
                ]
            },
            ..Self::default()
        }
    }

    /// The same geometry, positions aside (`SameGeometry`).
    fn same_geometry(&self, other: &Self) -> bool {
        (
            self.num_layers,
            self.full_attention_interval,
            self.conv_elems,
            self.state_elems,
            self.kv_row,
            self.indexer_head_dim,
            self.compress_ratio,
            self.hc_dim,
            self.ple_elems,
            self.has_mtp,
            self.rope,
        ) == (
            other.num_layers,
            other.full_attention_interval,
            other.conv_elems,
            other.state_elems,
            other.kv_row,
            other.indexer_head_dim,
            other.compress_ratio,
            other.hc_dim,
            other.ple_elems,
            other.has_mtp,
            other.rope,
        )
    }

    pub(crate) fn write(&self, out: &mut [u8]) {
        out[..8].copy_from_slice(&MAGIC);
        let words = [
            self.num_layers,
            self.full_attention_interval,
            self.conv_elems,
            self.state_elems,
            self.kv_row,
            self.indexer_head_dim,
            self.compress_ratio,
            self.hc_dim,
            self.ple_elems,
            self.has_mtp,
            self.position,
            self.blocks,
            self.mtp_position,
            self.mtp_blocks,
            self.mtp_residual_valid,
            self.hidden_rows,
            self.image_count,
        ];
        for (i, w) in words.iter().enumerate() {
            out[8 + 4 * i..12 + 4 * i].copy_from_slice(&w.to_le_bytes());
        }
        for (i, w) in self.rope.iter().enumerate() {
            out[76 + 4 * i..80 + 4 * i].copy_from_slice(&w.to_le_bytes());
        }
        out[92..96].copy_from_slice(&self.ngram_prev[0].to_le_bytes());
        out[96..100].copy_from_slice(&self.ngram_prev[1].to_le_bytes());
        out[100..104].fill(0);
        out[104..112].copy_from_slice(&self.payload_bytes.to_le_bytes());
    }

    /// `None` without the magic.
    fn read(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < HEADER_BYTES || bytes[..8] != MAGIC {
            return None;
        }
        let word = |i: usize| {
            u32::from_le_bytes(bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap_or_default())
        };
        let int = |at: usize| i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap_or_default());
        Some(Self {
            num_layers: word(0),
            full_attention_interval: word(1),
            conv_elems: word(2),
            state_elems: word(3),
            kv_row: word(4),
            indexer_head_dim: word(5),
            compress_ratio: word(6),
            hc_dim: word(7),
            ple_elems: word(8),
            has_mtp: word(9),
            position: word(10),
            blocks: word(11),
            mtp_position: word(12),
            mtp_blocks: word(13),
            mtp_residual_valid: word(14),
            hidden_rows: word(15),
            image_count: word(16),
            rope: [word(17), word(18), word(19), word(20)],
            ngram_prev: [int(92), int(96)],
            payload_bytes: u64::from_le_bytes(bytes[104..112].try_into().unwrap_or_default()),
        })
    }
}

/// A session buffer a snapshot section copies.
#[derive(Debug, Clone, Copy)]
enum Buffer {
    Conv(usize),
    State(usize),
    Ple,
    KCache(usize),
    VCache(usize),
    IndexK(usize),
    BlockK(usize),
    MtpIndexK,
    MtpBlockK,
    MtpKCache,
    MtpVCache,
    MtpResidual,
    MtpHidden,
}

/// One section: `bytes` at payload offset `offset`, from element `start`
/// of `buffer`.
#[derive(Debug, Clone, Copy)]
struct Section {
    buffer: Buffer,
    start: usize,
    offset: usize,
    bytes: usize,
}

/// The sections of `h` in payload order and the payload size
/// (`WalkSnapshot`). Indexer rows are stored chronologically: the ring of a
/// session (capacity `ring`) may wrap, and the other session may have
/// another capacity.
fn walk(h: &Header, ring: u32) -> (Vec<Section>, usize) {
    let mut sections = Vec::new();
    let mut offset = HEADER_BYTES;
    let mut region = |buffer: Buffer, start: usize, bytes: usize| {
        if bytes != 0 {
            sections.push(Section {
                buffer,
                start,
                offset,
                bytes,
            });
        }
        offset += bytes;
    };
    let is_attention = |il: u32| (il + 1).is_multiple_of(h.full_attention_interval);
    for il in 0..h.num_layers {
        if !is_attention(il) {
            region(Buffer::Conv(il as usize), 0, h.conv_elems as usize * 4);
            region(Buffer::State(il as usize), 0, h.state_elems as usize * 4);
        }
    }
    region(Buffer::Ple, 0, h.ple_elems as usize * 4);
    let idim = h.indexer_head_dim as usize;
    let kv_bytes = h.position as usize * h.kv_row as usize * 2;
    let ring_rows = |begin: u32, rows: u32| {
        let first = begin & (ring - 1);
        let tail = rows.min(ring - first);
        (first as usize, tail as usize, (rows - tail) as usize)
    };
    for il in 0..h.num_layers {
        if is_attention(il) {
            let l = il as usize;
            region(Buffer::KCache(l), 0, kv_bytes);
            region(Buffer::VCache(l), 0, kv_bytes);
            let begin = h.blocks * h.compress_ratio;
            let (first, tail, wrap) = ring_rows(begin, h.position - begin);
            region(Buffer::IndexK(l), first * idim, tail * idim * 4);
            region(Buffer::IndexK(l), 0, wrap * idim * 4);
            region(Buffer::BlockK(l), 0, h.blocks as usize * idim * 2);
        }
    }
    if h.has_mtp != 0 {
        let begin = h.mtp_blocks * h.compress_ratio;
        let (first, tail, wrap) = ring_rows(begin, h.mtp_position - begin);
        region(Buffer::MtpIndexK, first * idim, tail * idim * 4);
        region(Buffer::MtpIndexK, 0, wrap * idim * 4);
        region(Buffer::MtpBlockK, 0, h.mtp_blocks as usize * idim * 2);
        let mtp_kv = h.mtp_position as usize * h.kv_row as usize * 2;
        region(Buffer::MtpKCache, 0, mtp_kv);
        region(Buffer::MtpVCache, 0, mtp_kv);
        let residual = if h.mtp_residual_valid != 0 {
            h.hc_dim as usize * 4
        } else {
            0
        };
        region(Buffer::MtpResidual, 0, residual);
        region(
            Buffer::MtpHidden,
            0,
            h.hidden_rows as usize * h.hc_dim as usize * 4,
        );
    }
    // Image grids of the vision layout follow; text sessions have none.
    (sections, offset + h.image_count as usize * GRID_BYTES)
}

/// How a borrowed snapshot keeps a section: frozen with the mutable state,
/// borrowed rows of an append-only buffer, or frozen among the small rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Mutable,
    Rows {
        row_bytes: usize,
        draft: bool,
        pooled: bool,
    },
    Extra,
}

impl Buffer {
    fn kind(self, h: &Header) -> Kind {
        let kv = h.kv_row as usize * 2;
        let block = h.indexer_head_dim as usize * 2;
        let rows = |row_bytes, draft, pooled| Kind::Rows {
            row_bytes,
            draft,
            pooled,
        };
        match self {
            Buffer::Conv(_) | Buffer::State(_) | Buffer::Ple => Kind::Mutable,
            Buffer::KCache(_) | Buffer::VCache(_) => rows(kv, false, false),
            Buffer::BlockK(_) => rows(block, false, true),
            Buffer::MtpKCache | Buffer::MtpVCache => rows(kv, true, false),
            Buffer::MtpBlockK => rows(block, true, true),
            Buffer::IndexK(_) | Buffer::MtpIndexK | Buffer::MtpResidual | Buffer::MtpHidden => {
                Kind::Extra
            }
        }
    }
}

/// The bytes behind a session buffer of an append-only section, for borrowed
/// snapshots.
fn share(session: &Session, s: &Section) -> Result<gufo_hip::SharedBytes, String> {
    let missing = || format!("snapshot section {:?} is not in this session", s.buffer);
    let attention = |il: usize| {
        session
            .attention
            .get(il)
            .and_then(Option::as_ref)
            .ok_or_else(missing)
    };
    let mtp = || session.mtp.as_ref().ok_or_else(missing);
    if s.start != 0 {
        return Err("a borrowed row section must start its buffer".into());
    }
    Ok(match s.buffer {
        Buffer::KCache(il) => attention(il)?.k_cache.share(),
        Buffer::VCache(il) => attention(il)?.v_cache.share(),
        Buffer::BlockK(il) => attention(il)?.block_k.share(),
        Buffer::MtpKCache => mtp()?.attention.k_cache.share(),
        Buffer::MtpVCache => mtp()?.attention.v_cache.share(),
        Buffer::MtpBlockK => mtp()?.attention.block_k.share(),
        _ => return Err(missing()),
    })
}

impl<'a> View<'a> {
    fn into_bytes(self) -> Result<DeviceSlice<'a, u8>, String> {
        match self {
            View::F32(v) => v.cast::<u8>(),
            View::F16(v) => v.cast::<u8>(),
        }
        .ok_or_else(|| "snapshot section is not byte addressable".into())
    }
}

impl<'a> ViewMut<'a> {
    fn into_bytes(self) -> Result<DeviceSliceMut<'a, u8>, String> {
        match self {
            ViewMut::F32(v) => v.cast::<u8>(),
            ViewMut::F16(v) => v.cast::<u8>(),
        }
        .ok_or_else(|| "snapshot section is not byte addressable".into())
    }
}

/// A session buffer as an `f32` or F16 view.
enum View<'a> {
    F32(DeviceSlice<'a, f32>),
    F16(DeviceSlice<'a, F16>),
}

enum ViewMut<'a> {
    F32(DeviceSliceMut<'a, f32>),
    F16(DeviceSliceMut<'a, F16>),
}

/// Section `s` of `session`, or of a prefill checkpoint's mutable state in
/// `captured` (the append-only rows stay the session's).
fn view<'a>(
    session: &'a Session,
    captured: Option<&'a CheckpointStorage>,
    s: &Section,
) -> Result<View<'a>, String> {
    let missing = || format!("snapshot section {:?} is not in this session", s.buffer);
    let f32s = |b: &'a gufo_hip::DeviceBuffer<f32>| -> Result<View<'a>, String> {
        let end = s.start + s.bytes / 4;
        (end <= b.len())
            .then(|| View::F32(b.slice(s.start..end)))
            .ok_or_else(missing)
    };
    let halves = |b: &'a gufo_hip::DeviceBuffer<F16>| -> Result<View<'a>, String> {
        let end = s.start + s.bytes / 2;
        (end <= b.len())
            .then(|| View::F16(b.slice(s.start..end)))
            .ok_or_else(missing)
    };
    let linear = |il: usize| {
        session
            .linear
            .get(il)
            .and_then(Option::as_ref)
            .ok_or_else(missing)
    };
    let attention = |il: usize| {
        session
            .attention
            .get(il)
            .and_then(Option::as_ref)
            .ok_or_else(missing)
    };
    let mtp = || session.mtp.as_ref().ok_or_else(missing);
    if let Some(storage) = captured {
        let part = |slice: Option<DeviceSlice<'a, f32>>| -> Result<View<'a>, String> {
            let slice = slice.ok_or_else(missing)?;
            let end = s.start + s.bytes / 4;
            (end <= slice.len())
                .then(|| View::F32(slice.range(s.start..end)))
                .ok_or_else(missing)
        };
        match s.buffer {
            Buffer::Conv(il) => return part(storage.linear(il).map(|(conv, _)| conv)),
            Buffer::State(il) => return part(storage.linear(il).map(|(_, state)| state)),
            Buffer::Ple => return part(storage.ple()),
            Buffer::MtpHidden => return part(Some(storage.hidden())),
            Buffer::MtpResidual => {
                return Err("a prefill checkpoint keeps no draft residual".into());
            }
            _ => {}
        }
    }
    match s.buffer {
        Buffer::Conv(il) => f32s(&linear(il)?.conv_state),
        Buffer::State(il) => f32s(&linear(il)?.state),
        Buffer::Ple => f32s(session.ple_history.as_ref().ok_or_else(missing)?),
        Buffer::KCache(il) => halves(&attention(il)?.k_cache),
        Buffer::VCache(il) => halves(&attention(il)?.v_cache),
        Buffer::IndexK(il) => f32s(&attention(il)?.index_k),
        Buffer::BlockK(il) => halves(&attention(il)?.block_k),
        Buffer::MtpIndexK => f32s(&mtp()?.attention.index_k),
        Buffer::MtpBlockK => halves(&mtp()?.attention.block_k),
        Buffer::MtpKCache => halves(&mtp()?.attention.k_cache),
        Buffer::MtpVCache => halves(&mtp()?.attention.v_cache),
        Buffer::MtpResidual => f32s(&mtp()?.h),
        Buffer::MtpHidden => f32s(&mtp()?.target_hidden),
    }
}

fn view_mut<'a>(session: &'a mut Session, s: &Section) -> Result<ViewMut<'a>, String> {
    let missing = || format!("snapshot section {:?} is not in this session", s.buffer);
    fn f32s<'a>(b: &'a mut gufo_hip::DeviceBuffer<f32>, s: &Section) -> Option<ViewMut<'a>> {
        let end = s.start + s.bytes / 4;
        (end <= b.len()).then(|| ViewMut::F32(b.slice_mut(s.start..end)))
    }
    fn halves<'a>(b: &'a mut gufo_hip::DeviceBuffer<F16>, s: &Section) -> Option<ViewMut<'a>> {
        let end = s.start + s.bytes / 2;
        (end <= b.len()).then(|| ViewMut::F16(b.slice_mut(s.start..end)))
    }
    let view = match s.buffer {
        Buffer::Conv(il) => session
            .linear
            .get_mut(il)
            .and_then(Option::as_mut)
            .and_then(|l| f32s(&mut l.conv_state, s)),
        Buffer::State(il) => session
            .linear
            .get_mut(il)
            .and_then(Option::as_mut)
            .and_then(|l| f32s(&mut l.state, s)),
        Buffer::Ple => session.ple_history.as_mut().and_then(|b| f32s(b, s)),
        Buffer::KCache(il) => session
            .attention
            .get_mut(il)
            .and_then(Option::as_mut)
            .and_then(|a| halves(&mut a.k_cache, s)),
        Buffer::VCache(il) => session
            .attention
            .get_mut(il)
            .and_then(Option::as_mut)
            .and_then(|a| halves(&mut a.v_cache, s)),
        Buffer::IndexK(il) => session
            .attention
            .get_mut(il)
            .and_then(Option::as_mut)
            .and_then(|a| f32s(&mut a.index_k, s)),
        Buffer::BlockK(il) => session
            .attention
            .get_mut(il)
            .and_then(Option::as_mut)
            .and_then(|a| halves(&mut a.block_k, s)),
        Buffer::MtpIndexK => session
            .mtp
            .as_mut()
            .and_then(|m| f32s(&mut m.attention.index_k, s)),
        Buffer::MtpBlockK => session
            .mtp
            .as_mut()
            .and_then(|m| halves(&mut m.attention.block_k, s)),
        Buffer::MtpKCache => session
            .mtp
            .as_mut()
            .and_then(|m| halves(&mut m.attention.k_cache, s)),
        Buffer::MtpVCache => session
            .mtp
            .as_mut()
            .and_then(|m| halves(&mut m.attention.v_cache, s)),
        Buffer::MtpResidual => session.mtp.as_mut().and_then(|m| f32s(&mut m.h, s)),
        Buffer::MtpHidden => session
            .mtp
            .as_mut()
            .and_then(|m| f32s(&mut m.target_hidden, s)),
    };
    view.ok_or_else(missing)
}

/// The session's image grids, as the payload ends with them.
fn write_grids(session: &Session, out: &mut [u8]) {
    for (grid, out) in session
        .vision
        .layout()
        .images
        .iter()
        .zip(out.as_chunks_mut::<GRID_BYTES>().0)
    {
        out[0..4].copy_from_slice(&grid.offset.to_le_bytes());
        out[4..8].copy_from_slice(&grid.height.to_le_bytes());
        out[8..12].copy_from_slice(&grid.width.to_le_bytes());
    }
}

/// Captures idle sessions of one executor: its geometry and a transfer
/// stream of its own, so captures need neither the executor nor its stream
/// (`SaveSnapshot` on a capture worker).
pub struct Saver {
    owner: u64,
    config: Config,
    max_speculative: u32,
    transfer: Mutex<Stream>,
    /// The executor's device storage for borrowed snapshots.
    pool: Arc<Pool>,
}

impl Saver {
    pub(crate) fn new(
        owner: u64,
        config: Config,
        max_speculative: u32,
        pool: Arc<Pool>,
    ) -> Result<Self, String> {
        Ok(Self {
            owner,
            config,
            max_speculative,
            transfer: Mutex::new(Stream::new().map_err(|e| format!("hipStreamCreate: {e}"))?),
            pool,
        })
    }

    fn header(&self, session: &Session, hidden_rows: u32) -> Header {
        let mut h = Header::new(&self.config, session);
        h.blocks = session.blocks;
        if let Some(mtp) = &session.mtp {
            h.mtp_blocks = mtp.blocks;
            h.mtp_position = mtp.position;
            h.mtp_residual_valid = u32::from(mtp.residual_valid);
        }
        h.hidden_rows = hidden_rows;
        h
    }

    /// Payload bytes of the session's context with `hidden_rows` kept trunk
    /// rows (`SnapshotBytes`).
    pub fn bytes(&self, session: &Session, hidden_rows: u32) -> u64 {
        let h = self.header(session, hidden_rows);
        walk(&h, session.index_capacity).1 as u64
    }

    /// Copies the session's context into `payload`, which must have exactly
    /// [`Saver::bytes`] bytes. The session must be idle; other sessions may
    /// run meanwhile (`SaveSnapshot`).
    pub fn save(
        &self,
        session: &Session,
        hidden_rows: u32,
        payload: &mut [u8],
    ) -> Result<(), String> {
        if session.owner != self.owner {
            return Err("session belongs to another executor".into());
        }
        if session.spec_tokens != 0 {
            return Err("snapshot with a pending speculative batch; roll back first".into());
        }
        if hidden_rows > session.position || hidden_rows > self.max_speculative {
            return Err("kept trunk rows exceed the position or batch".into());
        }
        let mut h = self.header(session, hidden_rows);
        h.ngram_prev = session.ngram.prev;
        self.write(session, None, h, payload)
    }

    /// The session's context as a snapshot that borrows its append-only
    /// rows (`SaveSnapshot`, `kBorrowed`); `None` without device headroom,
    /// when the caller saves host bytes. The session must be idle.
    pub(crate) fn borrow(
        &self,
        session: &Session,
        hidden_rows: u32,
    ) -> Result<Option<Handle>, String> {
        if session.owner != self.owner {
            return Err("session belongs to another executor".into());
        }
        if session.spec_tokens != 0 {
            return Err("snapshot with a pending speculative batch; roll back first".into());
        }
        if hidden_rows > session.position || hidden_rows > self.max_speculative {
            return Err("kept trunk rows exceed the position or batch".into());
        }
        let mut h = self.header(session, hidden_rows);
        h.ngram_prev = session.ngram.prev;
        self.capture(session, None, h)
    }

    /// [`save_checkpoint`](Self::save_checkpoint) as a borrowed snapshot:
    /// the checkpoint's mutable state is frozen before the next forward
    /// reuses `storage`.
    pub(crate) fn borrow_checkpoint(
        &self,
        session: &Session,
        checkpoint: &PrefillCheckpoint,
        storage: &CheckpointStorage,
    ) -> Result<Option<Handle>, String> {
        if session.owner != self.owner {
            return Err("session belongs to another executor".into());
        }
        if session.position < checkpoint.frontier.position {
            return Err("the checkpointed pass has not run".into());
        }
        let mut h = self.checkpoint_header(session, &checkpoint.frontier);
        h.ngram_prev = checkpoint.ngram.prev;
        self.capture(session, Some(storage), h)
    }

    fn capture(
        &self,
        session: &Session,
        captured: Option<&CheckpointStorage>,
        mut h: Header,
    ) -> Result<Option<Handle>, String> {
        let (sections, bytes) = walk(&h, session.index_capacity);
        h.payload_bytes = bytes as u64;
        let registry = session
            .snapshots
            .get_or_init(|| Registry::new(Arc::clone(&self.pool)));
        if !Arc::ptr_eq(registry.pool(), &self.pool) {
            return Err("session belongs to another executor".into());
        }
        let mut mutable = Vec::new();
        let mut extras = Vec::new();
        let mut rows = Vec::new();
        for s in &sections {
            if session.peek_cancelled() {
                return Err("generation cancelled".into());
            }
            match s.buffer.kind(&h) {
                Kind::Mutable => {
                    mutable.push((s.offset, view(session, captured, s)?.into_bytes()?))
                }
                Kind::Extra => extras.push((s.offset, view(session, captured, s)?.into_bytes()?)),
                Kind::Rows {
                    row_bytes,
                    draft,
                    pooled,
                } => rows.push(RowSection {
                    source: share(session, s)?,
                    offset: s.offset,
                    bytes: s.bytes,
                    row_bytes,
                    draft,
                    pooled,
                }),
            }
        }
        let mut grids = vec![0u8; h.image_count as usize * GRID_BYTES];
        write_grids(session, &mut grids);
        registry.capture(h, grids, mutable, extras, rows)
    }

    /// Payload bytes of the checkpoint a pass from `session` would take at
    /// `position` (`PrefillCheckpointBytes`).
    pub fn checkpoint_bytes(&self, session: &Session, position: u32) -> u64 {
        let frontier =
            crate::checkpoint::Frontier::new(&self.config, session, position, self.max_speculative);
        self.frontier_bytes(session, &frontier)
    }

    /// Payload bytes of a checkpoint at `frontier` of `session`'s pass.
    pub(crate) fn frontier_bytes(
        &self,
        session: &Session,
        frontier: &crate::checkpoint::Frontier,
    ) -> u64 {
        let h = self.checkpoint_header(session, frontier);
        walk(&h, session.index_capacity).1 as u64
    }

    fn checkpoint_header(&self, session: &Session, f: &crate::checkpoint::Frontier) -> Header {
        let mut h = Header::new(&self.config, session);
        h.position = f.position;
        h.blocks = f.blocks;
        h.image_count = 0;
        if let Some((position, blocks)) = f.mtp {
            h.mtp_position = position;
            h.mtp_blocks = blocks;
            h.mtp_residual_valid = 0;
        }
        h.hidden_rows = f.hidden_rows;
        h
    }

    /// Copies the checkpoint `checkpoint` a pass from `session` captured
    /// into `storage` (`SaveSnapshot` of upstream's checkpoint view): the
    /// pass's mutable state at the boundary with the session's rows below
    /// it. The pass must have finished; `payload` must have exactly
    /// [`Saver::checkpoint_bytes`] bytes.
    pub(crate) fn save_checkpoint(
        &self,
        session: &Session,
        checkpoint: &PrefillCheckpoint,
        storage: &CheckpointStorage,
        payload: &mut [u8],
    ) -> Result<(), String> {
        if session.owner != self.owner {
            return Err("session belongs to another executor".into());
        }
        if session.position < checkpoint.frontier.position {
            return Err("the checkpointed pass has not run".into());
        }
        let mut h = self.checkpoint_header(session, &checkpoint.frontier);
        h.ngram_prev = checkpoint.ngram.prev;
        self.write(session, Some(storage), h, payload)
    }

    fn write(
        &self,
        session: &Session,
        captured: Option<&CheckpointStorage>,
        mut h: Header,
        payload: &mut [u8],
    ) -> Result<(), String> {
        let (sections, bytes) = walk(&h, session.index_capacity);
        h.payload_bytes = bytes as u64;
        if payload.len() != bytes {
            return Err("snapshot buffer size does not match the payload".into());
        }
        h.write(payload);
        let grids = payload.len() - h.image_count as usize * GRID_BYTES;
        write_grids(session, &mut payload[grids..]);
        let transfer = self.transfer.lock().unwrap_or_else(|e| e.into_inner());
        for s in &sections {
            if session.peek_cancelled() {
                return Err("generation cancelled".into());
            }
            let out = &mut payload[s.offset..s.offset + s.bytes];
            let result = match view(session, captured, s)? {
                View::F32(v) => transfer.download_bytes_across(v, out),
                View::F16(v) => transfer.download_bytes_across(v, out),
            };
            result.map_err(|e| format!("snapshot copy: {e}"))?;
        }
        Ok(())
    }
}

impl Executor {
    /// A capture context with a transfer stream of its own.
    pub fn saver(&self) -> Result<Saver, String> {
        Saver::new(
            self.id,
            self.config().clone(),
            self.core.options.max_speculative,
            Arc::clone(&self.pool),
        )
    }

    /// Payload bytes of the session's context (`SnapshotBytes`).
    pub fn snapshot_bytes(&self, session: &Session, hidden_rows: u32) -> u64 {
        self.saver.bytes(session, hidden_rows)
    }

    /// [`Saver::save`] on the executor's own capture context.
    pub fn save_snapshot(
        &self,
        session: &Session,
        hidden_rows: u32,
        payload: &mut [u8],
    ) -> Result<(), String> {
        self.saver.save(session, hidden_rows, payload)
    }

    /// Replaces the session's context with `payload`, keeping at most the
    /// rollback rows the restored operation needs (`next_drafts`); never
    /// allocates (`RestoreSnapshot`).
    pub fn restore_snapshot(
        &mut self,
        session: &mut Session,
        payload: &[u8],
        next_drafts: u32,
    ) -> Result<SnapshotInfo, String> {
        if session.owner != self.id {
            return Err("session belongs to another executor".into());
        }
        let h = Header::read(payload);
        let mine = Header::new(self.config(), session);
        let Some(h) = h.filter(|h| h.same_geometry(&mine)) else {
            return Err(if payload.len() < HEADER_BYTES {
                "snapshot payload is truncated".into()
            } else {
                "snapshot was taken with another model geometry".into()
            });
        };
        let grids = &payload[payload
            .len()
            .saturating_sub(h.image_count as usize * GRID_BYTES)..];
        let (sections, layout) =
            self.prepare_restore(session, &h, grids, payload.len(), next_drafts)?;
        // Every row is overwritten: snapshots borrowing them copy them out.
        session.preserve(0, 0)?;
        let result = (|| -> Result<(), String> {
            for s in &sections {
                if session.check_cancelled() {
                    return Err("generation cancelled".into());
                }
                let bytes = &payload[s.offset..s.offset + s.bytes];
                let copied: Result<(), HipError> = match view_mut(session, s)? {
                    ViewMut::F32(v) => self.stream.upload_bytes(bytes, v),
                    ViewMut::F16(v) => self.stream.upload_bytes(bytes, v),
                };
                copied.map_err(|e| format!("snapshot restore: {e}"))?;
            }
            Ok(())
        })();
        self.finish_restore(session, &h, &layout, result)
    }

    /// [`restore_snapshot`](Self::restore_snapshot) of a borrowed snapshot:
    /// device copies only, and into the session it borrows from, its rows
    /// still in place stay (`RestoreSnapshot` with a `SnapshotState`).
    pub(crate) fn restore_borrowed(
        &mut self,
        session: &mut Session,
        snapshot: &Handle,
        next_drafts: u32,
    ) -> Result<SnapshotInfo, String> {
        if session.owner != self.id {
            return Err("session belongs to another executor".into());
        }
        let source = Arc::clone(snapshot.registry());
        let target = session.snapshots.get().cloned();
        let same = target.as_ref().is_some_and(|t| Arc::ptr_eq(t, &source));
        // Both registries stay locked until the copies finished: neither
        // side may overwrite or free the rows meanwhile. Lock in address
        // order.
        let other = target.as_ref().filter(|_| !same);
        let source_first = other.is_none_or(|o| Arc::as_ptr(&source) < Arc::as_ptr(o));
        let (mut source_state, mut target_state) = if source_first {
            let s = source.lock();
            (s, other.map(|o| o.lock()))
        } else {
            let t = other.map(|o| o.lock());
            (source.lock(), t)
        };
        source_state.check()?;
        if let Some(t) = &target_state {
            t.check()?;
        }
        let entry = source_state
            .entry(snapshot.id())
            .ok_or("snapshot is no longer registered")?;
        if let Some(payload) = entry.materialized().cloned() {
            drop(target_state);
            drop(source_state);
            return self.restore_snapshot(session, &payload, next_drafts);
        }
        let h = entry.header;
        let grids = entry.grids.clone();
        let kept = if same { entry.kept() } else { Keep::default() };
        let mine = Header::new(self.config(), session);
        if !h.same_geometry(&mine) {
            return Err("snapshot was taken with another model geometry".into());
        }
        let (sections, layout) =
            self.prepare_restore(session, &h, &grids, h.payload_bytes as usize, next_drafts)?;
        // Other snapshots of the target copy out what the restore
        // overwrites; into its own session, this one keeps its live rows.
        if same {
            source.preserve_locked(&mut source_state, kept)?;
        } else if let (Some(t), Some(state)) = (other, target_state.as_mut()) {
            t.preserve_locked(state, Keep::default())?;
        }
        let entry = source_state
            .entry(snapshot.id())
            .ok_or("snapshot is no longer registered")?;
        let stream = &self.stream;
        let result = (|| -> Result<(), String> {
            let copy = |source: DeviceSlice<'_, u8>, target: DeviceSliceMut<'_, u8>| {
                stream
                    .copy_across(source, target)
                    .map_err(|e| format!("snapshot restore: {e}"))
            };
            for s in &sections {
                if session.check_cancelled() {
                    return Err("generation cancelled".into());
                }
                match s.buffer.kind(&h) {
                    Kind::Mutable => {
                        let frozen = entry
                            .mutable(s.offset, s.bytes)
                            .ok_or("snapshot mutable state is missing")?;
                        copy(frozen, view_mut(session, s)?.into_bytes()?)?;
                    }
                    Kind::Extra => {
                        for (at, piece) in entry.extras(s.offset, s.bytes) {
                            let target = view_mut(session, s)?.into_bytes()?;
                            copy(piece, target.range(at..at + piece.len()))?;
                        }
                    }
                    Kind::Rows { .. } => {
                        let from = if same {
                            entry.live_bytes(s.offset).unwrap_or(0)
                        } else {
                            0
                        };
                        let parts = entry
                            .rows(s.offset, from)
                            .ok_or("snapshot rows are missing")?;
                        for (at, part) in parts {
                            let target = view_mut(session, s)?.into_bytes()?;
                            copy(part, target.range(at..at + part.len()))?;
                        }
                    }
                }
            }
            Ok(())
        })();
        drop(target_state);
        drop(source_state);
        self.finish_restore(session, &h, &layout, result)
    }

    /// Validates `h` against the session and prepares it for a restore:
    /// the sections and image layout to apply, with the queued work drained
    /// and the rollback rows trimmed.
    fn prepare_restore(
        &mut self,
        session: &mut Session,
        h: &Header,
        grids: &[u8],
        payload_len: usize,
        next_drafts: u32,
    ) -> Result<(Vec<Section>, RopeLayout), String> {
        let h = *h;
        let max_blocks = h.position / h.compress_ratio;
        if h.image_count > h.position
            || h.position == 0
            || h.position > session.max_context
            || h.blocks > max_blocks
            || h.mtp_position > h.position
            || h.mtp_blocks > h.mtp_position / h.compress_ratio
            || h.mtp_position - h.mtp_blocks * h.compress_ratio > session.index_capacity
            || h.hidden_rows > h.position
            || h.hidden_rows > self.core.options.max_speculative
            || h.mtp_residual_valid > 1
            || (h.mtp_residual_valid != 0 && h.mtp_position == 0)
            || (h.has_mtp == 0
                && (h.mtp_position != 0 || h.hidden_rows != 0 || h.mtp_residual_valid != 0))
        {
            return Err("snapshot positions do not fit this session".into());
        }
        if h.position - h.blocks * h.compress_ratio > session.index_capacity {
            return Err("unpooled indexer rows exceed the ring capacity".into());
        }
        let (sections, bytes) = walk(&h, session.index_capacity);
        if h.payload_bytes != payload_len as u64 || bytes != payload_len {
            return Err("snapshot payload size does not match its header".into());
        }
        if grids.len() != h.image_count as usize * GRID_BYTES {
            return Err("snapshot image grids are truncated".into());
        }
        let word = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let layout = RopeLayout {
            images: grids
                .as_chunks::<GRID_BYTES>()
                .0
                .iter()
                .map(|g| ImageGrid {
                    offset: word(&g[0..4]),
                    height: word(&g[4..8]),
                    width: word(&g[8..12]),
                })
                .collect(),
        };
        layout
            .validate(session.max_context)
            .map_err(|e| e.message().to_string())?;
        if !layout.images.is_empty()
            && layout.prefix(h.position) != session.vision.layout().prefix(h.position)
        {
            return Err("image snapshot layout does not match its prompt attachment".into());
        }
        if layout.images.is_empty() {
            session
                .vision
                .configure(None, None, &self.stream)
                .map_err(|e| e.message().to_string())?;
        }
        // The session's queued work targets buffers the copies overwrite.
        self.stream
            .synchronize()
            .map_err(|e| format!("restore drain: {e}"))?;
        let remaining = session.max_context - h.position;
        session.trim_rollback(
            next_drafts
                .min(self.core.options.max_speculative - 1)
                .min(remaining.saturating_sub(1)),
        );
        Ok((sections, layout))
    }

    /// Applies the restored positions, or resets the session after a failed
    /// copy.
    fn finish_restore(
        &mut self,
        session: &mut Session,
        h: &Header,
        layout: &RopeLayout,
        result: Result<(), String>,
    ) -> Result<SnapshotInfo, String> {
        if let Err(e) = result {
            let _ = session.reset(&self.stream);
            return Err(e);
        }
        session
            .vision
            .restore_layout(layout, &self.stream)
            .map_err(|e| e.message().to_string())?;
        session.position = h.position;
        session.blocks = h.blocks;
        if let Some(mtp) = &mut session.mtp {
            mtp.position = h.mtp_position;
            mtp.blocks = h.mtp_blocks;
            mtp.residual_valid = h.mtp_residual_valid != 0;
        }
        session.spec_base = h.position;
        session.spec_tokens = 0;
        session.ngram.prev = h.ngram_prev;
        Ok(SnapshotInfo {
            position: h.position,
            hidden_rows: h.hidden_rows,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_header_has_the_cpp_layout() {
        let h = Header {
            num_layers: 48,
            position: 7,
            mtp_residual_valid: 1,
            image_count: 2,
            rope: [
                4.0f32.to_bits(),
                262_144,
                32.0f32.to_bits(),
                1.0f32.to_bits(),
            ],
            ngram_prev: [-1, 5],
            payload_bytes: 0x0102_0304_0506_0708,
            ..Header::default()
        };
        let mut bytes = [0xffu8; HEADER_BYTES];
        h.write(&mut bytes);
        assert_eq!(&bytes[..8], b"QFNSNAP6");
        assert_eq!(bytes[8], 48);
        assert_eq!(bytes[48], 7);
        assert_eq!(bytes[64], 1, "the residual flag follows mtp_blocks");
        assert_eq!(bytes[72], 2);
        assert_eq!(
            &bytes[76..80],
            &4.0f32.to_le_bytes(),
            "the rope factor follows image_count"
        );
        assert_eq!(&bytes[80..84], &262_144u32.to_le_bytes());
        assert_eq!(&bytes[92..96], &(-1i32).to_le_bytes());
        assert_eq!(&bytes[100..104], &[0; 4], "padding before the payload size");
        assert_eq!(&bytes[104..112], &0x0102_0304_0506_0708u64.to_le_bytes());
        assert_eq!(Header::read(&bytes), Some(h));
    }
}
