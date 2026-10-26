//! The persistent continuation payload of the Qwen serving runner, byte for
//! byte as the C++ runner writes it, so that both servers share a disk cache:
//! - `GQWRUN01` v3 (`SerializePersistentSnapshot` in
//!   `src/cli/serve/inference_backend.cpp`): position, frontier token,
//!   pending tokens, then the compact target state, the frontier logits and,
//!   with speculation, the verifier state;
//! - `GQKVSNP1` (`QwenGpuSnapshot::SerializeCompact`, `hip/arena.cpp`): the
//!   live K and V rows, the convolution and DeltaNet state and the image
//!   grids;
//! - `GSPVER01` v2 (`SpeculativeVerifierSnapshot`,
//!   `src/core/speculative/speculative_verifier.cpp`): acceptance statistics,
//!   the draft length and the rolling acceptance window around the draft
//!   state;
//! - `GDFDRF01` v4 (`hip/dflash/backend.cpp`): the DFlash2 draft controller,
//!   prompt-lookup state and pending target features around `GDFGPU01` v2
//!   (`hip/dflash/executor.cpp`), the draft's injected K and V history; v3
//!   payloads, without lookup state, still restore.
//!
//! Layouts compute every section's offset and write or check the headers;
//! the caller copies the device sections into place.

use std::ops::Range;

use gufo_core::prompt_lookup::LookupState;
pub use gufo_qwen::vision::ImageGrid;
use gufo_qwen::vision::validate_image_grids;

pub const RUNNER_MAGIC: [u8; 8] = *b"GQWRUN01";
pub const RUNNER_VERSION: u32 = 3;
pub const RUNNER_HEADER_BYTES: usize = 112;
/// Pending tokens a snapshot can carry for resumption.
pub const MAX_RESUME_TOKENS: usize = 9;
const RUNNER_SPECULATIVE_FLAG: u32 = 1 << 0;
const RUNNER_PENDING_TOKEN_FLAG: u32 = 1 << 1;

pub const COMPACT_MAGIC: [u8; 8] = *b"GQKVSNP1";
pub const COMPACT_HEADER_BYTES: usize = 80;

pub const VERIFIER_MAGIC: [u8; 8] = *b"GSPVER01";
pub const VERIFIER_VERSION: u32 = 2;
pub const VERIFIER_HEADER_BYTES: usize = 96;

pub const DRAFT_MAGIC: [u8; 8] = *b"GDFDRF01";
pub const DRAFT_VERSION: u32 = 4;
/// The previous draft version, whose header keeps bytes 72..80 zero.
const DRAFT_VERSION_WITHOUT_LOOKUP: u32 = 3;
pub const DRAFT_HEADER_BYTES: usize = 80;
const DRAFT_PRIMED_FLAG: u32 = 1 << 0;

pub const DRAFT_GPU_MAGIC: [u8; 8] = *b"GDFGPU01";
pub const DRAFT_GPU_VERSION: u32 = 2;
pub const DRAFT_GPU_HEADER_BYTES: usize = 64;

fn put_u32(destination: &mut [u8], offset: usize, value: u32) {
    destination[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(destination: &mut [u8], offset: usize, value: u64) {
    destination[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

/// Callers check the header length first.
fn get_u32(source: &[u8], offset: usize) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&source[offset..offset + 4]);
    u32::from_le_bytes(bytes)
}

fn get_u64(source: &[u8], offset: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&source[offset..offset + 8]);
    u64::from_le_bytes(bytes)
}

fn size(value: u64, what: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{what} size overflows"))
}

fn checked(values: &[usize], what: &str) -> Result<usize, String> {
    values
        .iter()
        .try_fold(0usize, |total, &value| total.checked_add(value))
        .ok_or_else(|| format!("{what} total overflows"))
}

fn product(values: &[usize], what: &str) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1usize, |total, &value| total.checked_mul(value))
        .ok_or_else(|| format!("{what} size overflows"))
}

/// Attention KV precision (`QwenKvCacheStorage`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvStorage {
    F32 = 0,
    F16 = 1,
}

impl KvStorage {
    pub fn element_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// DeltaNet state precision (`QwenRecurrentStateStorage`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecurrentStorage {
    F32 = 0,
    Bf16 = 1,
}

impl RecurrentStorage {
    pub fn element_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

/// The target state (`CompactSnapshotLayout`): after the header, the live K
/// rows `[attention layer][valid_context][kv_width]`, the V rows, the
/// convolution state (f32), the DeltaNet state and 12 bytes per image grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactLayout {
    pub attention_layers: u32,
    pub kv_width: u32,
    pub max_context: u32,
    pub valid_context: u32,
    pub kv_storage: KvStorage,
    pub recurrent_storage: RecurrentStorage,
    pub conv_elements: usize,
    pub deltanet_elements: usize,
    pub images: Vec<ImageGrid>,
    live_kv_bytes_per_plane: usize,
    conv_bytes: usize,
    deltanet_bytes: usize,
    total_bytes: usize,
}

impl CompactLayout {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        attention_layers: u32,
        kv_width: u32,
        max_context: u32,
        valid_context: u32,
        kv_storage: KvStorage,
        recurrent_storage: RecurrentStorage,
        conv_elements: usize,
        deltanet_elements: usize,
        images: Vec<ImageGrid>,
    ) -> Result<Self, String> {
        const WHAT: &str = "Qwen GPU allocation";
        let live_kv_bytes_per_plane = product(
            &[
                attention_layers as usize,
                valid_context as usize,
                kv_width as usize,
                kv_storage.element_bytes(),
            ],
            WHAT,
        )?;
        let conv_bytes = product(&[conv_elements, 4], WHAT)?;
        let deltanet_bytes = product(
            &[deltanet_elements, recurrent_storage.element_bytes()],
            WHAT,
        )?;
        let total_bytes = checked(
            &[
                COMPACT_HEADER_BYTES,
                live_kv_bytes_per_plane,
                live_kv_bytes_per_plane,
                conv_bytes,
                deltanet_bytes,
                product(&[images.len(), 12], WHAT)?,
            ],
            WHAT,
        )?;
        Ok(Self {
            attention_layers,
            kv_width,
            max_context,
            valid_context,
            kv_storage,
            recurrent_storage,
            conv_elements,
            deltanet_elements,
            images,
            live_kv_bytes_per_plane,
            conv_bytes,
            deltanet_bytes,
            total_bytes,
        })
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn live_kv_elements_per_plane(&self) -> usize {
        self.live_kv_bytes_per_plane / self.kv_storage.element_bytes()
    }

    pub fn k(&self) -> Range<usize> {
        COMPACT_HEADER_BYTES..COMPACT_HEADER_BYTES + self.live_kv_bytes_per_plane
    }

    pub fn v(&self) -> Range<usize> {
        let start = self.k().end;
        start..start + self.live_kv_bytes_per_plane
    }

    pub fn conv(&self) -> Range<usize> {
        let start = self.v().end;
        start..start + self.conv_bytes
    }

    pub fn deltanet(&self) -> Range<usize> {
        let start = self.conv().end;
        start..start + self.deltanet_bytes
    }

    fn grids(&self) -> usize {
        self.deltanet().end
    }

    /// Writes the header and image grids; the caller fills the state
    /// sections. Only the header has reserved bytes: clearing the payload
    /// first would only compete with inference for bandwidth.
    pub fn write(&self, destination: &mut [u8]) -> Result<(), String> {
        if destination.len() != self.total_bytes {
            return Err("Qwen compact snapshot destination size is invalid".into());
        }
        validate_image_grids(&self.images, self.max_context)?;
        destination[..COMPACT_HEADER_BYTES].fill(0);
        destination[..8].copy_from_slice(&COMPACT_MAGIC);
        put_u32(destination, 8, if self.images.is_empty() { 1 } else { 2 });
        put_u32(destination, 12, COMPACT_HEADER_BYTES as u32);
        put_u32(destination, 16, self.attention_layers);
        put_u32(destination, 20, self.kv_width);
        put_u32(destination, 24, self.max_context);
        put_u32(destination, 28, self.valid_context);
        put_u32(destination, 32, self.kv_storage as u32);
        put_u32(destination, 36, self.recurrent_storage as u32);
        put_u64(destination, 40, self.live_kv_elements_per_plane() as u64);
        put_u64(destination, 48, self.conv_elements as u64);
        put_u64(destination, 56, self.deltanet_elements as u64);
        put_u64(destination, 64, self.total_bytes as u64);
        put_u32(destination, 72, self.images.len() as u32);
        for (index, grid) in self.images.iter().enumerate() {
            let offset = self.grids() + index * 12;
            put_u32(destination, offset, grid.offset);
            put_u32(destination, offset + 4, grid.height);
            put_u32(destination, offset + 8, grid.width);
        }
        Ok(())
    }

    /// `ParseCompactSnapshotLayout`: the layout a payload describes, checked
    /// for internal consistency. Compatibility with an arena is the
    /// caller's check.
    pub fn parse(payload: &[u8]) -> Result<Self, String> {
        if payload.len() < COMPACT_HEADER_BYTES || payload[..8] != COMPACT_MAGIC {
            return Err("Qwen compact snapshot header is invalid".into());
        }
        let version = get_u32(payload, 8);
        let image_count = get_u32(payload, 72) as usize;
        if !(version == 1 || version == 2)
            || image_count > get_u32(payload, 28) as usize
            || (version == 1 && image_count != 0)
            || get_u32(payload, 12) != COMPACT_HEADER_BYTES as u32
            || get_u32(payload, 76) != 0
        {
            return Err("Qwen compact snapshot version is invalid".into());
        }
        let kv_storage = match get_u32(payload, 32) {
            0 => KvStorage::F32,
            1 => KvStorage::F16,
            _ => return Err("Qwen compact snapshot precision is invalid".into()),
        };
        let recurrent_storage = match get_u32(payload, 36) {
            0 => RecurrentStorage::F32,
            1 => RecurrentStorage::Bf16,
            _ => return Err("Qwen compact snapshot precision is invalid".into()),
        };
        const WHAT: &str = "Qwen compact snapshot";
        let mut layout = Self::new(
            get_u32(payload, 16),
            get_u32(payload, 20),
            get_u32(payload, 24),
            get_u32(payload, 28),
            kv_storage,
            recurrent_storage,
            size(get_u64(payload, 48), WHAT)?,
            size(get_u64(payload, 56), WHAT)?,
            vec![ImageGrid::default(); image_count],
        )?;
        if layout.live_kv_elements_per_plane() != size(get_u64(payload, 40), WHAT)?
            || layout.total_bytes != size(get_u64(payload, 64), WHAT)?
            || layout.total_bytes != payload.len()
        {
            return Err("Qwen compact snapshot size is invalid".into());
        }
        let grids = layout.grids();
        for (index, grid) in layout.images.iter_mut().enumerate() {
            let offset = grids + index * 12;
            *grid = ImageGrid {
                offset: get_u32(payload, offset),
                height: get_u32(payload, offset + 4),
                width: get_u32(payload, offset + 8),
            };
        }
        validate_image_grids(&layout.images, layout.max_context)?;
        Ok(layout)
    }
}

/// The runner-level fields of a persistent snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunnerHeader {
    /// Tokens the state covers.
    pub position: u64,
    /// The token whose logits follow the state.
    pub frontier: u32,
    pub speculative: bool,
    pub pending_tokens: Vec<u32>,
    pub logits_count: usize,
    pub target_bytes: usize,
    pub verifier_bytes: usize,
}

/// What a runner accepts on restore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunnerLimits {
    pub max_context: u32,
    pub vocab_size: u32,
    pub speculative: bool,
}

impl RunnerHeader {
    fn logits_bytes(&self) -> Result<usize, String> {
        product(&[self.logits_count, 4], "Qwen persistent snapshot")
    }

    pub fn total_bytes(&self) -> Result<usize, String> {
        checked(
            &[
                RUNNER_HEADER_BYTES,
                self.target_bytes,
                self.logits_bytes()?,
                self.verifier_bytes,
            ],
            "Qwen persistent snapshot",
        )
    }

    /// The compact target state.
    pub fn target(&self) -> Range<usize> {
        RUNNER_HEADER_BYTES..RUNNER_HEADER_BYTES + self.target_bytes
    }

    /// The frontier logits, little-endian f32.
    pub fn logits(&self) -> Range<usize> {
        let start = self.target().end;
        start..start + self.logits_count * 4
    }

    /// The speculative verifier state; empty without speculation.
    pub fn verifier(&self) -> Range<usize> {
        let start = self.logits().end;
        start..start + self.verifier_bytes
    }

    /// Zeroes `destination` and writes the header.
    pub fn write(&self, destination: &mut [u8]) -> Result<(), String> {
        let total = self.total_bytes()?;
        if destination.len() != total || self.pending_tokens.len() > MAX_RESUME_TOKENS {
            return Err("Qwen persistent snapshot destination size is invalid".into());
        }
        destination.fill(0);
        destination[..8].copy_from_slice(&RUNNER_MAGIC);
        put_u32(destination, 8, RUNNER_VERSION);
        put_u32(destination, 12, RUNNER_HEADER_BYTES as u32);
        put_u64(destination, 16, self.position);
        put_u32(destination, 24, self.frontier);
        let flags = if self.speculative {
            RUNNER_SPECULATIVE_FLAG
        } else {
            0
        } | if self.pending_tokens.is_empty() {
            0
        } else {
            RUNNER_PENDING_TOKEN_FLAG
        };
        put_u32(destination, 28, flags);
        put_u64(destination, 32, self.logits_count as u64);
        put_u64(destination, 40, self.target_bytes as u64);
        put_u64(destination, 48, total as u64);
        put_u64(destination, 56, self.verifier_bytes as u64);
        put_u32(destination, 64, self.pending_tokens.len() as u32);
        for (index, &token) in self.pending_tokens.iter().enumerate() {
            put_u32(destination, 72 + index * 4, token);
        }
        Ok(())
    }

    /// The header checks of `RestorePersistentSnapshot`.
    pub fn parse(payload: &[u8], limits: RunnerLimits) -> Result<Self, String> {
        if payload.len() < RUNNER_HEADER_BYTES
            || payload[..8] != RUNNER_MAGIC
            || get_u32(payload, 8) != RUNNER_VERSION
            || get_u32(payload, 12) != RUNNER_HEADER_BYTES as u32
        {
            return Err("Qwen persistent snapshot header is invalid".into());
        }
        const WHAT: &str = "Qwen persistent snapshot";
        let position = size(get_u64(payload, 16), WHAT)?;
        let frontier = get_u32(payload, 24);
        let flags = get_u32(payload, 28);
        let logits_count = size(get_u64(payload, 32), WHAT)?;
        let target_bytes = size(get_u64(payload, 40), WHAT)?;
        let total_bytes = size(get_u64(payload, 48), WHAT)?;
        let verifier_bytes = size(get_u64(payload, 56), WHAT)?;
        let speculative = flags & RUNNER_SPECULATIVE_FLAG != 0;
        let has_pending = flags & RUNNER_PENDING_TOKEN_FLAG != 0;
        let pending_count = get_u32(payload, 64) as usize;
        if position == 0
            || position > limits.max_context as usize
            || flags & !(RUNNER_SPECULATIVE_FLAG | RUNNER_PENDING_TOKEN_FLAG) != 0
            || has_pending != (pending_count != 0)
            || pending_count > MAX_RESUME_TOKENS
            || get_u32(payload, 68) != 0
            || get_u32(payload, 108) != 0
            || speculative != limits.speculative
            || frontier >= limits.vocab_size
            || logits_count != limits.vocab_size as usize
            || target_bytes == 0
            || speculative != (verifier_bytes != 0)
            || total_bytes != payload.len()
        {
            return Err("Qwen persistent snapshot metadata is invalid".into());
        }
        let mut pending_tokens = Vec::with_capacity(pending_count);
        for index in 0..MAX_RESUME_TOKENS {
            let token = get_u32(payload, 72 + index * 4);
            if index < pending_count {
                if token >= limits.vocab_size {
                    return Err("Qwen snapshot pending token is invalid".into());
                }
                pending_tokens.push(token);
            } else if token != 0 {
                return Err("Qwen snapshot has nonzero reserved tokens".into());
            }
        }
        let header = Self {
            position: position as u64,
            frontier,
            speculative,
            pending_tokens,
            logits_count,
            target_bytes,
            verifier_bytes,
        };
        if header.total_bytes()? != payload.len() {
            return Err("Qwen persistent snapshot payload size is invalid".into());
        }
        Ok(header)
    }
}

/// The speculative verifier's fields (`SpeculativeVerifierSnapshot`): after
/// the header, the rolling acceptance rates (f32) and the draft state.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifierHeader {
    pub draft_bytes: usize,
    pub rolling_acceptance: Vec<f32>,
    pub total_draft_tokens: u64,
    pub total_accepted_tokens: u64,
    pub total_verification_steps: u64,
    pub total_emitted_tokens: u64,
    pub current_draft_length: u32,
}

/// What a verifier accepts on restore (`SpeculativeOptions`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifierLimits {
    pub rolling_window: usize,
    pub min_draft_tokens: u32,
    pub max_draft_tokens: u32,
}

impl VerifierHeader {
    fn rolling_bytes(&self) -> Result<usize, String> {
        product(
            &[self.rolling_acceptance.len(), 4],
            "speculative verifier persistent",
        )
    }

    pub fn total_bytes(&self) -> Result<usize, String> {
        checked(
            &[
                VERIFIER_HEADER_BYTES,
                self.rolling_bytes()?,
                self.draft_bytes,
            ],
            "speculative verifier persistent",
        )
    }

    /// The draft backend's state.
    pub fn draft(&self) -> Range<usize> {
        let start = VERIFIER_HEADER_BYTES + self.rolling_acceptance.len() * 4;
        start..start + self.draft_bytes
    }

    /// Zeroes `destination` and writes the header and rolling window.
    pub fn write(&self, destination: &mut [u8]) -> Result<(), String> {
        if destination.len() != self.total_bytes()? {
            return Err("speculative verifier persistent destination size is invalid".into());
        }
        destination.fill(0);
        destination[..8].copy_from_slice(&VERIFIER_MAGIC);
        put_u32(destination, 8, VERIFIER_VERSION);
        put_u32(destination, 12, VERIFIER_HEADER_BYTES as u32);
        put_u64(destination, 16, self.draft_bytes as u64);
        put_u64(destination, 24, self.rolling_acceptance.len() as u64);
        put_u64(destination, 32, self.total_draft_tokens);
        put_u64(destination, 40, self.total_accepted_tokens);
        put_u64(destination, 48, self.total_verification_steps);
        put_u64(destination, 56, self.total_emitted_tokens);
        put_u32(destination, 64, self.current_draft_length);
        put_u64(destination, 72, destination.len() as u64);
        for (index, rate) in self.rolling_acceptance.iter().enumerate() {
            put_u32(
                destination,
                VERIFIER_HEADER_BYTES + index * 4,
                rate.to_bits(),
            );
        }
        Ok(())
    }

    /// The checks of `SpeculativeVerifier::RestorePersistentSnapshot` before
    /// the draft restores.
    pub fn parse(payload: &[u8], limits: VerifierLimits) -> Result<Self, String> {
        if payload.len() < VERIFIER_HEADER_BYTES
            || payload[..8] != VERIFIER_MAGIC
            || get_u32(payload, 8) != VERIFIER_VERSION
            || get_u32(payload, 12) != VERIFIER_HEADER_BYTES as u32
            || get_u32(payload, 68) != 0
            || get_u64(payload, 80) != 0
            || get_u64(payload, 88) != 0
        {
            return Err("speculative verifier persistent header is invalid".into());
        }
        const WHAT: &str = "speculative verifier persistent";
        let draft_bytes = size(get_u64(payload, 16), WHAT)?;
        let rolling_count = size(get_u64(payload, 24), WHAT)?;
        let total_draft_tokens = get_u64(payload, 32);
        let total_accepted_tokens = get_u64(payload, 40);
        let total_verification_steps = get_u64(payload, 48);
        let total_emitted_tokens = get_u64(payload, 56);
        let current_draft_length = get_u32(payload, 64);
        let total_bytes = size(get_u64(payload, 72), WHAT)?;
        if rolling_count > usize::MAX / 4
            || rolling_count > limits.rolling_window
            || total_accepted_tokens > total_draft_tokens
            || current_draft_length < limits.min_draft_tokens
            || current_draft_length > limits.max_draft_tokens
            || total_bytes != payload.len()
        {
            return Err("speculative verifier persistent metadata is invalid".into());
        }
        if checked(
            &[VERIFIER_HEADER_BYTES, rolling_count * 4, draft_bytes],
            WHAT,
        )? != payload.len()
        {
            return Err("speculative verifier persistent payload size is invalid".into());
        }
        let rolling_acceptance = (0..rolling_count)
            .map(|index| {
                let rate = f32::from_bits(get_u32(payload, VERIFIER_HEADER_BYTES + index * 4));
                if rate.is_finite() && (0.0..=1.0).contains(&rate) {
                    Ok(rate)
                } else {
                    Err("speculative verifier rolling acceptance is invalid".to_string())
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            draft_bytes,
            rolling_acceptance,
            total_draft_tokens,
            total_accepted_tokens,
            total_verification_steps,
            total_emitted_tokens,
            current_draft_length,
        })
    }
}

/// The draft's injected K and V history (`QwenDFlashGpuSnapshot`): after
/// the header, K then V, each `[layer][elements_per_layer]` f32.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftGpuLayout {
    pub num_layers: u32,
    pub kv_width: u32,
    pub max_context: u32,
    pub valid_context: u32,
    pub history_capacity: u64,
    elements_per_layer: usize,
}

impl DraftGpuLayout {
    pub fn new(
        num_layers: u32,
        kv_width: u32,
        max_context: u32,
        valid_context: u32,
        history_capacity: u64,
    ) -> Result<Self, String> {
        if history_capacity == 0
            || history_capacity > u64::from(max_context)
            || valid_context > max_context
        {
            return Err("DFlash GPU snapshot metadata is malformed".into());
        }
        let rows = u64::from(valid_context).min(history_capacity) as usize;
        let elements_per_layer = product(&[rows, kv_width as usize], "DFlash GPU persistent")?;
        product(
            &[num_layers as usize, elements_per_layer, 4, 2],
            "DFlash GPU persistent",
        )?;
        Ok(Self {
            num_layers,
            kv_width,
            max_context,
            valid_context,
            history_capacity,
            elements_per_layer,
        })
    }

    pub fn elements_per_layer(&self) -> usize {
        self.elements_per_layer
    }

    fn bytes_per_kind(&self) -> usize {
        self.num_layers as usize * self.elements_per_layer * 4
    }

    pub fn total_bytes(&self) -> usize {
        DRAFT_GPU_HEADER_BYTES + 2 * self.bytes_per_kind()
    }

    pub fn k(&self) -> Range<usize> {
        DRAFT_GPU_HEADER_BYTES..DRAFT_GPU_HEADER_BYTES + self.bytes_per_kind()
    }

    pub fn v(&self) -> Range<usize> {
        let start = self.k().end;
        start..start + self.bytes_per_kind()
    }

    pub fn write(&self, destination: &mut [u8]) -> Result<(), String> {
        if destination.len() != self.total_bytes() {
            return Err("DFlash GPU persistent destination size is invalid".into());
        }
        // The K/V copies overwrite the payload.
        destination[..DRAFT_GPU_HEADER_BYTES].fill(0);
        destination[..8].copy_from_slice(&DRAFT_GPU_MAGIC);
        put_u32(destination, 8, DRAFT_GPU_VERSION);
        put_u32(destination, 12, DRAFT_GPU_HEADER_BYTES as u32);
        put_u32(destination, 16, self.num_layers);
        put_u32(destination, 20, self.kv_width);
        put_u32(destination, 24, self.max_context);
        put_u32(destination, 28, self.valid_context);
        put_u64(destination, 32, self.elements_per_layer as u64);
        put_u64(destination, 40, 2 * self.bytes_per_kind() as u64);
        put_u64(destination, 48, self.total_bytes() as u64);
        put_u64(destination, 56, self.history_capacity);
        Ok(())
    }

    /// The header checks of `QwenDFlashGpuExecutor::RestorePersistentSnapshot`
    /// against the executor's `expected` shape.
    pub fn parse(payload: &[u8], expected: &Self) -> Result<Self, String> {
        if payload.len() < DRAFT_GPU_HEADER_BYTES
            || payload[..8] != DRAFT_GPU_MAGIC
            || get_u32(payload, 8) != DRAFT_GPU_VERSION
            || get_u32(payload, 12) != DRAFT_GPU_HEADER_BYTES as u32
            || get_u64(payload, 56) != expected.history_capacity
        {
            return Err("DFlash GPU persistent header is invalid".into());
        }
        const WHAT: &str = "DFlash GPU persistent";
        let num_layers = get_u32(payload, 16);
        let kv_width = get_u32(payload, 20);
        let max_context = get_u32(payload, 24);
        let valid_context = get_u32(payload, 28);
        let elements_per_layer = size(get_u64(payload, 32), WHAT)?;
        let payload_bytes = size(get_u64(payload, 40), WHAT)?;
        let total_bytes = size(get_u64(payload, 48), WHAT)?;
        let incompatible = || "DFlash GPU persistent metadata is incompatible".to_string();
        if num_layers != expected.num_layers
            || kv_width != expected.kv_width
            || max_context != expected.max_context
            || valid_context > expected.max_context
        {
            return Err(incompatible());
        }
        let layout = Self::new(
            num_layers,
            kv_width,
            max_context,
            valid_context,
            expected.history_capacity,
        )
        .map_err(|_| incompatible())?;
        if elements_per_layer != layout.elements_per_layer {
            return Err(incompatible());
        }
        if payload_bytes != 2 * layout.bytes_per_kind()
            || total_bytes != payload.len()
            || layout.total_bytes() != payload.len()
        {
            return Err("DFlash GPU persistent payload size is invalid".into());
        }
        Ok(layout)
    }
}

/// The DFlash2 draft backend's fields (`DFlashDraftSnapshot`).
#[derive(Debug, Clone, PartialEq)]
pub struct DraftHeader {
    pub primed: bool,
    pub target_feature_width: u32,
    /// Pending target features (f32 values).
    pub pending_features: usize,
    pub gpu_bytes: usize,
    /// The draft's valid context (`ValidContext` of its GPU snapshot).
    pub injected_context: u32,
    pub max_context: u32,
    pub controller_state: f32,
    pub policy: u32,
    pub max_draft_tokens: u32,
    pub last_full_width: u32,
    /// The prompt-lookup resume point and request boundary (default in a v3
    /// payload).
    pub lookup: LookupState,
}

/// What a draft backend accepts on restore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftLimits {
    pub target_feature_width: u32,
    pub max_context: u32,
    pub policy: u32,
    pub max_draft_tokens: u32,
}

impl DraftHeader {
    pub fn total_bytes(&self) -> Result<usize, String> {
        checked(
            &[
                DRAFT_HEADER_BYTES,
                self.gpu_bytes,
                product(&[self.pending_features, 4], "DFlash draft persistent")?,
            ],
            "DFlash draft persistent",
        )
    }

    /// The `GDFGPU01` history.
    pub fn gpu(&self) -> Range<usize> {
        DRAFT_HEADER_BYTES..DRAFT_HEADER_BYTES + self.gpu_bytes
    }

    /// Pending target features, little-endian f32.
    pub fn pending(&self) -> Range<usize> {
        let start = self.gpu().end;
        start..start + self.pending_features * 4
    }

    pub fn write(&self, destination: &mut [u8]) -> Result<(), String> {
        if destination.len() != self.total_bytes()?
            || self.target_feature_width == 0
            || !self
                .pending_features
                .is_multiple_of(self.target_feature_width as usize)
        {
            return Err("DFlash draft persistent destination is invalid".into());
        }
        // The nested serializer and the pending features overwrite the
        // payload.
        destination[..DRAFT_HEADER_BYTES].fill(0);
        destination[..8].copy_from_slice(&DRAFT_MAGIC);
        put_u32(destination, 8, DRAFT_VERSION);
        put_u32(destination, 12, DRAFT_HEADER_BYTES as u32);
        put_u32(
            destination,
            16,
            if self.primed { DRAFT_PRIMED_FLAG } else { 0 },
        );
        put_u32(destination, 20, self.target_feature_width);
        put_u64(destination, 24, self.pending_features as u64);
        put_u64(destination, 32, self.gpu_bytes as u64);
        put_u64(destination, 40, destination.len() as u64);
        put_u32(destination, 48, self.injected_context);
        put_u32(destination, 52, self.max_context);
        put_u32(destination, 56, self.controller_state.to_bits());
        put_u32(destination, 60, self.policy);
        put_u32(destination, 64, self.max_draft_tokens);
        put_u32(destination, 68, self.last_full_width);
        put_u32(destination, 72, self.lookup.resume);
        put_u32(destination, 76, self.lookup.request_start);
        Ok(())
    }

    /// The header checks of `QwenDFlashGpuDraftBackend::RestorePersistentSnapshot`;
    /// the controller and history checks follow in the backend.
    pub fn parse(payload: &[u8], limits: DraftLimits) -> Result<Self, String> {
        if payload.len() < DRAFT_HEADER_BYTES
            || payload[..8] != DRAFT_MAGIC
            || (get_u32(payload, 8) != DRAFT_VERSION
                && get_u32(payload, 8) != DRAFT_VERSION_WITHOUT_LOOKUP)
            || get_u32(payload, 12) != DRAFT_HEADER_BYTES as u32
            || (get_u32(payload, 8) == DRAFT_VERSION_WITHOUT_LOOKUP && get_u64(payload, 72) != 0)
        {
            return Err("DFlash draft persistent header is invalid".into());
        }
        const WHAT: &str = "DFlash draft persistent";
        let flags = get_u32(payload, 16);
        let header = Self {
            primed: flags & DRAFT_PRIMED_FLAG != 0,
            target_feature_width: get_u32(payload, 20),
            pending_features: size(get_u64(payload, 24), WHAT)?,
            gpu_bytes: size(get_u64(payload, 32), WHAT)?,
            injected_context: get_u32(payload, 48),
            max_context: get_u32(payload, 52),
            controller_state: f32::from_bits(get_u32(payload, 56)),
            policy: get_u32(payload, 60),
            max_draft_tokens: get_u32(payload, 64),
            last_full_width: get_u32(payload, 68),
            lookup: if get_u32(payload, 8) >= DRAFT_VERSION {
                LookupState::read(&payload[72..80])
            } else {
                LookupState::default()
            },
        };
        let total_bytes = size(get_u64(payload, 40), WHAT)?;
        let width = header.target_feature_width as usize;
        if flags & !DRAFT_PRIMED_FLAG != 0
            || width == 0
            || header.target_feature_width != limits.target_feature_width
            || header.pending_features > usize::MAX / 4
            || !header.pending_features.is_multiple_of(width)
            || header.max_context != limits.max_context
            || header.policy != limits.policy
            || header.max_draft_tokens != limits.max_draft_tokens
            || header.injected_context > limits.max_context
            || header.pending_features / width
                > (limits.max_context - header.injected_context) as usize
            || total_bytes != payload.len()
        {
            return Err("DFlash draft persistent metadata is incompatible".into());
        }
        if !header
            .lookup
            .valid(header.injected_context as usize + header.pending_features / width)
        {
            return Err("DFlash persistent lookup positions are invalid".into());
        }
        if !header.primed && (header.injected_context != 0 || header.pending_features != 0) {
            return Err("DFlash draft unprimed payload contains state".into());
        }
        if header.total_bytes()? != payload.len() {
            return Err("DFlash draft persistent payload size is invalid".into());
        }
        Ok(header)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compact(images: Vec<ImageGrid>) -> CompactLayout {
        CompactLayout::new(
            2,
            8,
            64,
            5,
            KvStorage::F16,
            RecurrentStorage::F32,
            12,
            20,
            images,
        )
        .unwrap()
    }

    /// No fixed image count: the grids are bounded by the populated context
    /// (`TestCompactImageLayout` and the 257-grid
    /// `RopeLayout`).
    #[test]
    fn image_grids_are_bounded_by_the_context() {
        let grids: Vec<ImageGrid> = (0..257)
            .map(|i| ImageGrid {
                offset: i * 64,
                height: 8,
                width: 8,
            })
            .collect();
        assert!(validate_image_grids(&grids, 257 * 64).is_ok());
        assert!(validate_image_grids(&grids, 257 * 64 - 1).is_err());

        let ones: Vec<ImageGrid> = (0..257)
            .map(|offset| ImageGrid {
                offset,
                height: 1,
                width: 1,
            })
            .collect();
        let layout = CompactLayout::new(
            2,
            8,
            512,
            257,
            KvStorage::F16,
            RecurrentStorage::F32,
            12,
            20,
            ones,
        )
        .unwrap();
        let mut bytes = vec![0; layout.total_bytes()];
        layout.write(&mut bytes).unwrap();
        assert_eq!(CompactLayout::parse(&bytes).unwrap(), layout);
        // More images than populated tokens.
        put_u32(&mut bytes, 28, 256);
        assert!(CompactLayout::parse(&bytes).is_err());
    }

    #[test]
    fn compact_sections_follow_the_header_in_order() {
        let layout = compact(vec![ImageGrid {
            offset: 1,
            height: 2,
            width: 2,
        }]);
        assert_eq!(layout.k(), 80..80 + 2 * 5 * 8 * 2);
        assert_eq!(layout.v(), 240..400);
        assert_eq!(layout.conv(), 400..448);
        assert_eq!(layout.deltanet(), 448..528);
        assert_eq!(layout.total_bytes(), 540);
        let mut bytes = vec![0xaa; layout.total_bytes()];
        layout.write(&mut bytes).unwrap();
        assert_eq!(&bytes[..8], b"GQKVSNP1");
        assert_eq!(get_u32(&bytes, 8), 2, "version 2 carries image grids");
        assert_eq!(get_u64(&bytes, 40), 80);
        assert_eq!(CompactLayout::parse(&bytes).unwrap(), layout);

        let plain = compact(Vec::new());
        let mut bytes = vec![0; plain.total_bytes()];
        plain.write(&mut bytes).unwrap();
        assert_eq!(get_u32(&bytes, 8), 1);
        assert_eq!(CompactLayout::parse(&bytes).unwrap(), plain);
        bytes[76] = 1;
        assert_eq!(
            CompactLayout::parse(&bytes).unwrap_err(),
            "Qwen compact snapshot version is invalid"
        );
        bytes[76] = 0;
        bytes[32] = 2;
        assert_eq!(
            CompactLayout::parse(&bytes).unwrap_err(),
            "Qwen compact snapshot precision is invalid"
        );
        bytes[32] = 1;
        assert_eq!(
            CompactLayout::parse(&bytes[..bytes.len() - 1]).unwrap_err(),
            "Qwen compact snapshot size is invalid"
        );
    }

    #[test]
    fn image_grids_must_be_ordered_and_inside_the_context() {
        let grid = |offset, height, width| ImageGrid {
            offset,
            height,
            width,
        };
        assert!(validate_image_grids(&[grid(0, 2, 2), grid(4, 1, 1)], 5).is_ok());
        for bad in [
            vec![grid(0, 0, 2)],
            vec![grid(0, 2, 2), grid(3, 1, 1)],
            vec![grid(4, 1, 2)],
        ] {
            assert_eq!(
                validate_image_grids(&bad, 5).unwrap_err(),
                "invalid image position layout"
            );
        }
    }

    fn runner(speculative: bool) -> RunnerHeader {
        RunnerHeader {
            position: 7,
            frontier: 3,
            speculative,
            pending_tokens: vec![4, 5],
            logits_count: 10,
            target_bytes: 100,
            verifier_bytes: if speculative { 30 } else { 0 },
        }
    }

    #[test]
    fn runner_header_round_trips_and_rejects_foreign_state() {
        let limits = RunnerLimits {
            max_context: 16,
            vocab_size: 10,
            speculative: true,
        };
        let header = runner(true);
        let mut bytes = vec![0xff; header.total_bytes().unwrap()];
        header.write(&mut bytes).unwrap();
        assert_eq!(bytes.len(), 112 + 100 + 40 + 30);
        assert_eq!(get_u32(&bytes, 28), 3, "speculative and pending flags");
        assert_eq!(RunnerHeader::parse(&bytes, limits).unwrap(), header);
        assert_eq!(header.logits(), 212..252);
        assert_eq!(header.verifier(), 252..282);

        for limits in [
            RunnerLimits {
                speculative: false,
                ..limits
            },
            RunnerLimits {
                max_context: 6,
                ..limits
            },
            RunnerLimits {
                vocab_size: 11,
                ..limits
            },
        ] {
            assert_eq!(
                RunnerHeader::parse(&bytes, limits).unwrap_err(),
                "Qwen persistent snapshot metadata is invalid"
            );
        }
        let mut reserved = bytes.clone();
        put_u32(&mut reserved, 72 + 4 * 4, 1);
        assert_eq!(
            RunnerHeader::parse(&reserved, limits).unwrap_err(),
            "Qwen snapshot has nonzero reserved tokens"
        );
        let mut token = bytes.clone();
        put_u32(&mut token, 72, 10);
        assert_eq!(
            RunnerHeader::parse(&token, limits).unwrap_err(),
            "Qwen snapshot pending token is invalid"
        );
        bytes[8] = 2;
        assert_eq!(
            RunnerHeader::parse(&bytes, limits).unwrap_err(),
            "Qwen persistent snapshot header is invalid"
        );
    }

    #[test]
    fn verifier_payload_round_trips() {
        let limits = VerifierLimits {
            rolling_window: 16,
            min_draft_tokens: 1,
            max_draft_tokens: 7,
        };
        let verifier = VerifierHeader {
            draft_bytes: 20,
            rolling_acceptance: vec![0.5, 1.0, 0.0],
            total_draft_tokens: 30,
            total_accepted_tokens: 12,
            total_verification_steps: 5,
            total_emitted_tokens: 17,
            current_draft_length: 4,
        };
        let mut bytes = vec![0xff; verifier.total_bytes().unwrap()];
        verifier.write(&mut bytes).unwrap();
        assert_eq!(verifier.draft(), 108..128);
        assert_eq!(VerifierHeader::parse(&bytes, limits).unwrap(), verifier);
        put_u32(&mut bytes, 96, 1.5f32.to_bits());
        assert_eq!(
            VerifierHeader::parse(&bytes, limits).unwrap_err(),
            "speculative verifier rolling acceptance is invalid"
        );
        assert_eq!(
            VerifierHeader::parse(
                &bytes,
                VerifierLimits {
                    max_draft_tokens: 3,
                    ..limits
                }
            )
            .unwrap_err(),
            "speculative verifier persistent metadata is invalid"
        );
    }

    #[test]
    fn draft_payloads_round_trip() {
        let gpu = DraftGpuLayout::new(5, 8, 64, 40, 32).unwrap();
        assert_eq!(
            gpu.elements_per_layer(),
            32 * 8,
            "the history ring caps the rows"
        );
        let mut gpu_bytes = vec![1; gpu.total_bytes()];
        gpu.write(&mut gpu_bytes).unwrap();
        assert_eq!(DraftGpuLayout::parse(&gpu_bytes, &gpu).unwrap(), gpu);
        let other = DraftGpuLayout::new(5, 8, 128, 40, 32).unwrap();
        assert_eq!(
            DraftGpuLayout::parse(&gpu_bytes, &other).unwrap_err(),
            "DFlash GPU persistent metadata is incompatible"
        );

        let limits = DraftLimits {
            target_feature_width: 3,
            max_context: 64,
            policy: 1,
            max_draft_tokens: 7,
        };
        let draft = DraftHeader {
            primed: true,
            target_feature_width: 3,
            pending_features: 6,
            gpu_bytes: gpu.total_bytes(),
            injected_context: 40,
            max_context: 64,
            controller_state: 2.5,
            policy: 1,
            max_draft_tokens: 7,
            last_full_width: 4,
            lookup: LookupState {
                resume: 41,
                request_start: 30,
            },
        };
        let mut bytes = vec![0; draft.total_bytes().unwrap()];
        draft.write(&mut bytes).unwrap();
        assert_eq!(DraftHeader::parse(&bytes, limits).unwrap(), draft);
        assert_eq!(draft.pending(), bytes.len() - 24..bytes.len());
        // A v3 payload has no lookup state and keeps its bytes zero.
        let mut legacy = bytes.clone();
        put_u32(&mut legacy, 8, 3);
        assert_eq!(
            DraftHeader::parse(&legacy, limits).unwrap_err(),
            "DFlash draft persistent header is invalid"
        );
        legacy[72..80].fill(0);
        assert_eq!(
            DraftHeader::parse(&legacy, limits).unwrap(),
            DraftHeader {
                lookup: LookupState::default(),
                ..draft.clone()
            }
        );
        // The injected and pending positions bound the lookup state.
        let mut beyond = bytes.clone();
        put_u32(&mut beyond, 72, 43);
        assert_eq!(
            DraftHeader::parse(&beyond, limits).unwrap_err(),
            "DFlash persistent lookup positions are invalid"
        );
        put_u32(&mut beyond, 72, 42);
        assert!(DraftHeader::parse(&beyond, limits).is_ok());
        assert_eq!(
            DraftHeader::parse(
                &bytes,
                DraftLimits {
                    max_draft_tokens: 8,
                    ..limits
                }
            )
            .unwrap_err(),
            "DFlash draft persistent metadata is incompatible"
        );
    }
}
