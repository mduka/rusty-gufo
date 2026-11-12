//! Session snapshot payloads (version 4), byte-identical to C++. The fixed
//! header records enough of the layout to reject a payload written for a
//! different runtime; the body holds the checkpoint tokens, the last logits,
//! the compressed row counts, the raw window rows in logical order, the live
//! compressed rows and the compressor and indexer frontiers, and the DSpark
//! request state (and support KV, with a support model). Restoring copies
//! the bytes into the session's existing tensors.

use gufo_deepseek_v4_flash_kernels::{Backend, Input, Output, Result, SnapshotReader};
use gufo_hip::DeviceSlice;

use super::alloc::LayerCache;
use super::decode::{cached, cached_mut};
use super::*;
use crate::dspark_model::MAX_TARGET_LAYERS;
use crate::dspark_policy::RequestState;

/// `DS4_SESSION_PAYLOAD_MAGIC`: "DSV4".
pub const PAYLOAD_MAGIC: u32 = 0x3456_5344;
/// `DS4_SESSION_PAYLOAD_VERSION`.
pub const PAYLOAD_VERSION: u32 = 4;
const HEADER_FIELDS: usize = 19 + MAX_TARGET_LAYERS as usize;
/// Device writes go in pieces of this size (`DS4_SESSION_IO_CHUNK`).
const IO_CHUNK: usize = 8 << 20;
const ROW_BYTES: u64 = N_HEAD_DIM as u64 * F32;

fn attn_state_bytes(r: u32) -> u64 {
    let coff: u64 = if r == 4 { 2 } else { 1 };
    coff * u64::from(N_HEAD_DIM) * coff * u64::from(r) * F32
}

fn index_state_bytes(r: u32) -> u64 {
    let coff: u64 = if r == 4 { 2 } else { 1 };
    coff * u64::from(N_INDEXER_HEAD_DIM) * coff * u64::from(r) * F32
}

/// The session state a payload restores besides the graph's caches.
pub struct Restored {
    pub checkpoint: Vec<i32>,
    pub logits: Vec<f32>,
    pub state: RequestState,
}

type PayloadResult<T> = std::result::Result<T, &'static str>;

/// `bytes` of `tensor` from `offset`, a span of the payload
/// (`payload_write_tensor_span`).
fn span(tensor: &impl Input, offset: u64, bytes: u64) -> PayloadResult<DeviceSlice<'_, u8>> {
    let device = tensor.input();
    let len = device.len() as u64;
    if offset > len || bytes > len - offset {
        return Err("session tensor is smaller than the payload");
    }
    Ok(device.range(offset as usize..(offset + bytes) as usize))
}

/// The reading side of a payload, counting what remains.
struct Reader<'a> {
    data: &'a [u8],
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: u64) -> PayloadResult<&'a [u8]> {
        let n = usize::try_from(n).map_err(|_| "truncated session payload")?;
        if self.data.len() < n {
            return Err("truncated session payload");
        }
        let (head, tail) = self.data.split_at(n);
        self.data = tail;
        Ok(head)
    }

    fn u32(&mut self) -> PayloadResult<u32> {
        let b = self.bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Writes the next `bytes` into `tensor` from `offset`
    /// (`payload_read_tensor_span`).
    fn span(
        &mut self,
        b: &Backend,
        tensor: &mut impl Output,
        offset: u64,
        bytes: u64,
    ) -> PayloadResult<()> {
        if offset > tensor.bytes() || bytes > tensor.bytes() - offset {
            return Err("session tensor is smaller than the payload");
        }
        let data = self.bytes(bytes)?;
        for (i, chunk) in data.chunks(IO_CHUNK).enumerate() {
            b.write(tensor, offset + (i * IO_CHUNK) as u64, chunk)
                .map_err(|_| "failed to restore accelerator session tensor")?;
        }
        Ok(())
    }
}

/// The spans of a ring window of `live` rows ending at row `context`: at
/// most two contiguous runs (`payload_ring_tensor`).
fn ring_spans(capacity: u32, context: u32, live: u32) -> [(u64, u64); 2] {
    let first = (context - live) % capacity;
    let before_wrap = live.min(capacity - first);
    [
        (
            u64::from(first) * ROW_BYTES,
            u64::from(before_wrap) * ROW_BYTES,
        ),
        (0, u64::from(live - before_wrap) * ROW_BYTES),
    ]
}

impl LayerCache {
    /// `rocm_graph_rebuild_attn_comp_mirror`: the F16 mirror of the first
    /// `rows` compressed rows.
    fn rebuild_comp_mirror(&mut self, b: &Backend, rows: u32) -> Result<()> {
        match &mut self.attn_comp_f16 {
            Some(mirror) if rows != 0 => b.convert_f32_to_f16(
                mirror,
                cached(&self.attn_comp)?,
                u64::from(rows) * u64::from(N_HEAD_DIM),
            ),
            _ => Ok(()),
        }
    }
}

impl Graph {
    /// `session_raw_live_rows`: the raw rows a payload keeps, the last
    /// window; the next suffix writes its own rows before attention reads.
    fn raw_live_rows(&self, checkpoint_len: u32) -> u32 {
        let window = if self.raw_window != 0 {
            self.raw_window
        } else {
            N_SWA
        };
        window.min(self.raw_cap).min(checkpoint_len)
    }

    /// The header fields of the DSpark support state: stages, block,
    /// support ring capacity, injected rows, capture enabled, target layer
    /// count and IDs; zero without a support model.
    fn dspark_header(&self) -> [u32; 6 + MAX_TARGET_LAYERS as usize] {
        let mut header = [0; 6 + MAX_TARGET_LAYERS as usize];
        if let Some(ds) = &self.dspark {
            header[0] = ds.n_stages;
            header[1] = ds.block_size;
            header[2] = ds.cache_cap;
            header[3] = ds.context_len;
            header[4] = u32::from(ds.capture_enabled);
            header[5] = ds.target_layer_ids.len() as u32;
            header[6..6 + ds.target_layer_ids.len()].copy_from_slice(&ds.target_layer_ids);
        }
        header
    }

    /// The support ring rows a payload keeps: the last window of the
    /// injected context (`session_dspark_live_rows`).
    fn dspark_live_rows(&self) -> u32 {
        self.dspark_context_len().min(N_SWA)
    }

    /// `ds4_rocm_graph_snapshot_bytes`: the payload size of a checkpoint of
    /// `checkpoint_len` tokens, or 0 for none.
    pub fn snapshot_bytes(&self, checkpoint_len: usize) -> u64 {
        let Ok(tokens) = u32::try_from(checkpoint_len) else {
            return 0;
        };
        if tokens == 0 {
            return 0;
        }
        // A partial capture is not a complete continuation.
        if self.dspark_capturing() && self.dspark_context_len() != tokens {
            return 0;
        }
        let mut bytes = HEADER_FIELDS as u64 * 4 + 12 * 8 + 8 * 4;
        if let Some(ds) = &self.dspark
            && ds.context_len != 0
        {
            bytes += u64::from(ds.n_stages) * u64::from(self.dspark_live_rows()) * ROW_BYTES;
        }
        bytes += u64::from(tokens) * 4;
        bytes += u64::from(N_VOCAB) * F32;
        bytes += 2 * u64::from(N_LAYER) * 4;
        let raw_live = u64::from(self.raw_live_rows(tokens));
        for (il, cache) in (0..N_LAYER).zip(&self.layers) {
            bytes += raw_live * ROW_BYTES;
            let r = ratio(il);
            if r == 0 {
                continue;
            }
            bytes += u64::from(cache.n_comp) * ROW_BYTES + 2 * attn_state_bytes(r);
            if r == 4 {
                bytes += u64::from(cache.n_index_comp) * u64::from(N_INDEXER_HEAD_DIM) * F32
                    + 2 * index_state_bytes(r);
            }
        }
        bytes
    }

    /// `rocm_graph_save_payload`: the payload of the checkpoint `checkpoint`
    /// with its `logits`. The session's device work has completed.
    pub fn save_payload(
        &self,
        reader: &SnapshotReader,
        checkpoint: &[i32],
        logits: &[f32],
        prefill_capacity: u32,
        context_size: u32,
        state: &RequestState,
    ) -> PayloadResult<Vec<u8>> {
        let total = self.snapshot_bytes(checkpoint.len());
        if total == 0 || logits.len() < N_VOCAB as usize {
            return Err("session has no valid checkpoint to snapshot");
        }
        let total = usize::try_from(total)
            .map_err(|_| "session snapshot is too large for this platform")?;
        let tokens = checkpoint.len() as u32;
        let raw_live = self.raw_live_rows(tokens);
        let mut out = Vec::with_capacity(total);
        let mut header = vec![
            PAYLOAD_MAGIC,
            PAYLOAD_VERSION,
            context_size,
            prefill_capacity,
            self.raw_cap,
            self.raw_window,
            self.comp_cap,
            tokens,
            N_LAYER,
            N_HEAD_DIM,
            N_INDEXER_HEAD_DIM,
            N_VOCAB,
            raw_live,
        ];
        header.extend(self.dspark_header());
        debug_assert_eq!(header.len(), HEADER_FIELDS);
        for value in header {
            out.extend(value.to_le_bytes());
        }
        for counter in state.counters() {
            out.extend((counter as u32).to_le_bytes());
            out.extend(((counter >> 32) as u32).to_le_bytes());
        }
        for flag in state.flags() {
            out.extend(flag.to_le_bytes());
        }
        for &token in checkpoint {
            out.extend((token as u32).to_le_bytes());
        }
        for &value in &logits[..N_VOCAB as usize] {
            out.extend(value.to_le_bytes());
        }
        for cache in &self.layers {
            out.extend(cache.n_comp.to_le_bytes());
        }
        for cache in &self.layers {
            out.extend(cache.n_index_comp.to_le_bytes());
        }
        // The device spans follow the host fields; they are read together.
        let mut spans = Vec::new();
        for (il, cache) in (0..N_LAYER).zip(&self.layers) {
            // The raw ring in logical position order.
            for (offset, bytes) in ring_spans(self.raw_cap, tokens, raw_live) {
                if bytes != 0 {
                    spans.push(span(&cache.raw, offset, bytes)?);
                }
            }
            let r = ratio(il);
            if r == 0 {
                continue;
            }
            let missing = |_| "session tensor is smaller than the payload";
            spans.push(span(
                cached(&cache.attn_comp).map_err(missing)?,
                0,
                u64::from(cache.n_comp) * ROW_BYTES,
            )?);
            spans.push(span(
                cached(&cache.attn_state_kv).map_err(missing)?,
                0,
                attn_state_bytes(r),
            )?);
            spans.push(span(
                cached(&cache.attn_state_score).map_err(missing)?,
                0,
                attn_state_bytes(r),
            )?);
            if r == 4 {
                spans.push(span(
                    cached(&cache.index_comp).map_err(missing)?,
                    0,
                    u64::from(cache.n_index_comp) * u64::from(N_INDEXER_HEAD_DIM) * F32,
                )?);
                spans.push(span(
                    cached(&cache.index_state_kv).map_err(missing)?,
                    0,
                    index_state_bytes(r),
                )?);
                spans.push(span(
                    cached(&cache.index_state_score).map_err(missing)?,
                    0,
                    index_state_bytes(r),
                )?);
            }
        }
        // Only the committed target window is visible to the next draft.
        if let Some(ds) = &self.dspark
            && ds.context_len != 0
        {
            let live = self.dspark_live_rows();
            for cache in &ds.kv_cache {
                for (offset, bytes) in ring_spans(ds.cache_cap, ds.context_len, live) {
                    if bytes != 0 {
                        spans.push(span(cache, offset, bytes)?);
                    }
                }
            }
        }
        reader
            .read_spans(&spans, &mut out)
            .map_err(|_| "failed to read accelerator session tensor")?;
        debug_assert_eq!(out.len(), total);
        Ok(out)
    }

    /// `rocm_graph_load_payload`: checks `payload` against this graph and
    /// copies its caches in. On success the row counts are the payload's;
    /// on failure the caches may hold part of it, and the caller's
    /// checkpoint is invalid.
    pub fn load_payload(
        &mut self,
        b: &Backend,
        payload: &[u8],
        context_size: u32,
    ) -> PayloadResult<Restored> {
        let mut r = Reader { data: payload };
        let mut h = [0u32; HEADER_FIELDS];
        for value in &mut h {
            *value = r.u32()?;
        }
        if h[0] != PAYLOAD_MAGIC || h[1] != PAYLOAD_VERSION {
            return Err("unsupported session payload version");
        }
        let (saved_ctx, saved_raw_cap, saved_raw_window, saved_comp_cap) = (h[2], h[4], h[5], h[6]);
        let (saved_tokens, saved_raw_live) = (h[7], h[12]);
        if saved_ctx > context_size || saved_tokens >= context_size {
            return Err("KV checkpoint does not fit current context");
        }
        if h[8] != N_LAYER || h[9] != N_HEAD_DIM || h[10] != N_INDEXER_HEAD_DIM || h[11] != N_VOCAB
        {
            return Err("KV checkpoint was written for a different DS4 layout");
        }
        // The prefill chunk is scratch capacity, not KV layout.
        if saved_raw_window != self.raw_window {
            return Err("KV checkpoint graph chunk layout does not match current runtime");
        }
        // Raw rows are logical, so any ring large enough takes them, but the
        // live count must be the window the token count implies.
        let expected_raw_live = saved_tokens.min(saved_raw_window);
        if saved_raw_cap == 0
            || saved_raw_live != expected_raw_live
            || saved_raw_live > saved_raw_cap
            || saved_raw_live > self.raw_cap
        {
            return Err("KV checkpoint raw ring layout does not match current context");
        }
        if saved_comp_cap > self.comp_cap {
            return Err("KV checkpoint compressed cache is larger than current context");
        }
        let dspark = self.dspark_header();
        let (stages, block, ring, targets) = (dspark[0], dspark[1], dspark[2], dspark[5]);
        if saved_tokens == 0
            || h[13] != stages
            || h[14] != block
            || h[15] != ring
            || h[16] > saved_tokens
            || h[17] > 1
            || h[18] != targets
            || (h[17] != 0 && h[16] != saved_tokens)
            || (h[17] == 0 && h[16] != 0)
        {
            return Err("incompatible DSpark snapshot layout or coverage");
        }
        if h[19..] != dspark[6..] {
            return Err("incompatible DSpark target feature layout");
        }
        let mut counters = [0u64; 12];
        for counter in &mut counters {
            let low = r.u32()?;
            let high = r.u32()?;
            *counter = u64::from(low) | (u64::from(high) << 32);
        }
        let mut flags = [0u32; 8];
        for flag in &mut flags {
            *flag = r.u32()?;
        }
        let state =
            RequestState::from_payload(counters, flags).ok_or("invalid DSpark controller state")?;
        if self.dspark.is_some() && (h[17] != 0) == state.force_plain_request {
            return Err("inconsistent DSpark capture and request state");
        }

        let mut checkpoint = Vec::with_capacity(saved_tokens as usize);
        for _ in 0..saved_tokens {
            let token = r.u32()?;
            if token >= N_VOCAB {
                return Err("invalid checkpoint token ID");
            }
            checkpoint.push(token as i32);
        }
        let logits: Vec<f32> = r
            .bytes(u64::from(N_VOCAB) * F32)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        if !logits.iter().all(|v| v.is_finite()) {
            return Err("checkpoint contains non-finite logits");
        }
        let mut n_comp = vec![0u32; N_LAYER as usize];
        for (count, cache) in n_comp.iter_mut().zip(&self.layers) {
            *count = r.u32()?;
            if *count > saved_comp_cap || *count > cache.comp_cap {
                return Err("KV checkpoint has invalid compressed row count");
            }
        }
        let mut n_index_comp = vec![0u32; N_LAYER as usize];
        for (count, cache) in n_index_comp.iter_mut().zip(&self.layers) {
            *count = r.u32()?;
            if *count > saved_comp_cap || *count > cache.comp_cap {
                return Err("KV checkpoint has invalid indexer row count");
            }
        }

        b.synchronize()
            .map_err(|_| "failed to synchronize accelerator before KV restore")?;
        for il in 0..N_LAYER {
            let ratio = ratio(il);
            if ratio == 0 {
                continue;
            }
            let rows = u64::from(n_comp[il as usize].max(n_index_comp[il as usize]))
                + u64::from(N_SWA.div_ceil(ratio))
                + 2;
            let required = rows.min(u64::from(self.layers[il as usize].comp_cap)) as u32;
            self.reserve_compressed_rows(b, il, required)
                .map_err(|_| "failed to reserve compressed cache for KV restore")?;
        }
        let missing = |_| "session tensor is smaller than the payload";
        for (il, cache) in (0..N_LAYER).zip(self.layers.iter_mut()) {
            for (offset, bytes) in ring_spans(self.raw_cap, saved_tokens, saved_raw_live) {
                if bytes != 0 {
                    r.span(b, &mut cache.raw, offset, bytes)?;
                }
            }
            let ratio = ratio(il);
            if ratio == 0 {
                continue;
            }
            let rows = n_comp[il as usize];
            r.span(
                b,
                cached_mut(&mut cache.attn_comp).map_err(missing)?,
                0,
                u64::from(rows) * ROW_BYTES,
            )?;
            if ratio == 4 {
                cache
                    .rebuild_comp_mirror(b, rows)
                    .map_err(|_| "failed to rebuild FP16 attention cache mirror")?;
            }
            r.span(
                b,
                cached_mut(&mut cache.attn_state_kv).map_err(missing)?,
                0,
                attn_state_bytes(ratio),
            )?;
            r.span(
                b,
                cached_mut(&mut cache.attn_state_score).map_err(missing)?,
                0,
                attn_state_bytes(ratio),
            )?;
            if ratio == 4 {
                r.span(
                    b,
                    cached_mut(&mut cache.index_comp).map_err(missing)?,
                    0,
                    u64::from(n_index_comp[il as usize]) * u64::from(N_INDEXER_HEAD_DIM) * F32,
                )?;
                r.span(
                    b,
                    cached_mut(&mut cache.index_state_kv).map_err(missing)?,
                    0,
                    index_state_bytes(ratio),
                )?;
                r.span(
                    b,
                    cached_mut(&mut cache.index_state_score).map_err(missing)?,
                    0,
                    index_state_bytes(ratio),
                )?;
            }
        }
        if let Some(ds) = &mut self.dspark
            && h[16] != 0
        {
            let live = h[16].min(N_SWA);
            let cap = ds.cache_cap;
            for cache in &mut ds.kv_cache {
                for (offset, bytes) in ring_spans(cap, h[16], live) {
                    if bytes != 0 {
                        r.span(b, cache, offset, bytes)?;
                    }
                }
            }
        }
        if !r.data.is_empty() {
            return Err("KV checkpoint has trailing payload bytes");
        }
        b.synchronize()
            .map_err(|_| "failed to synchronize accelerator after KV restore")?;
        for (cache, (&comp, &index)) in self.layers.iter_mut().zip(n_comp.iter().zip(&n_index_comp))
        {
            cache.n_comp = comp;
            cache.n_index_comp = index;
        }
        if let Some(ds) = &mut self.dspark {
            ds.capture_enabled = h[17] != 0;
            ds.context_len = h[16];
            ds.capture_mask = 0;
            ds.capture_batch_mask = 0;
            ds.capture_batch_start = 0;
            ds.capture_batch_tokens = 0;
        }
        Ok(Restored {
            checkpoint,
            logits,
            state,
        })
    }
}
