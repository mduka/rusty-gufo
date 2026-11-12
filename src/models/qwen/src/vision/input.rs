//! Per-session image inputs (`src/models/qwen/vision/device_input.hip`): the
//! mRoPE table the attention kernels read, and the image embeddings that
//! replace the token embeddings of image-pad rows.

use std::sync::Arc;

use gufo_core::text::{Error, Result};
use gufo_hip::{DeviceSliceMut, LaunchError, RowTable, Stream};
use gufo_qwen_kernels as k;

use super::{Embedding, Encoder, Prompt, RopeLayout};

/// A session's image state. The table keeps its device address once
/// created, so captured decode graphs follow a new request's layout; text
/// sessions that never saw an image pass no table at all.
#[derive(Default)]
pub struct VisionInput {
    layout: RopeLayout,
    table: Option<RowTable<i32, 3>>,
    prompt: Option<Arc<Prompt>>,
    encoder: Option<Arc<Encoder>>,
    embeddings: Vec<Option<Arc<Embedding>>>,
}

impl VisionInput {
    /// Installs a request's prompt (`None` for text), keeping the
    /// embeddings when the same encoder sees the same images again.
    pub fn configure(
        &mut self,
        prompt: Option<Arc<Prompt>>,
        encoder: Option<Arc<Encoder>>,
        stream: &Stream,
    ) -> Result<()> {
        if prompt.as_ref().is_some_and(|p| !p.images.is_empty()) && encoder.is_none() {
            return Err(Error::invalid("image input has no vision encoder"));
        }
        let same_encoder = match (&self.encoder, &encoder) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        let reuse = match (&self.prompt, &prompt) {
            (Some(old), Some(new)) => {
                !new.cache_identity.is_empty()
                    && same_encoder
                    && old.cache_identity == new.cache_identity
            }
            _ => false,
        };
        let layout = prompt.as_ref().map(|p| p.rope.clone()).unwrap_or_default();
        self.prompt = prompt;
        self.encoder = encoder;
        if !reuse {
            self.embeddings.clear();
        }
        let images = self.prompt.as_ref().map_or(0, |p| p.images.len());
        self.embeddings.resize(images, None);
        self.restore_layout(&layout, stream)
    }

    /// Installs a restored snapshot's layout; embeddings are not needed for
    /// tokens already in the caches.
    pub fn restore_layout(&mut self, layout: &RopeLayout, stream: &Stream) -> Result<()> {
        layout.validate(i32::MAX as u32)?;
        if *layout == self.layout && (self.table.is_some() || layout.images.is_empty()) {
            return Ok(());
        }
        let positions: Vec<[i32; 3]> = (0..layout.prefix_length())
            .map(|token| layout.position(token))
            .collect();
        let device = |e| Error::runtime(format!("vision layout upload: {e}"));
        if self.table.is_none() {
            self.table = Some(RowTable::new(stream).map_err(device)?);
        }
        if let Some(table) = &mut self.table {
            table
                .set(stream, &positions, layout.delta())
                .map_err(device)?;
        }
        self.layout = layout.clone();
        Ok(())
    }

    pub fn layout(&self) -> &RopeLayout {
        &self.layout
    }

    /// The table for the attention kernels; `None` for a session that never
    /// had an image layout (plain positions).
    pub fn rope(&self) -> Option<&RowTable<i32, 3>> {
        self.table.as_ref()
    }

    /// Encodes every image that overlaps tokens `position..position + count`
    /// and is not encoded yet. Runs outside graph capture, before a forward
    /// that injects those rows.
    pub fn prepare(
        &mut self,
        position: u32,
        count: u32,
        cancelled: Option<&dyn Fn() -> bool>,
    ) -> Result<()> {
        let Some(prompt) = &self.prompt else {
            return Ok(());
        };
        for (index, image) in prompt.images.iter().enumerate() {
            let grid = image.grid;
            let end = grid.offset + grid.height * grid.width;
            if position.max(grid.offset) >= (position + count).min(end)
                || self.embeddings[index].is_some()
            {
                continue;
            }
            let encoder = self
                .encoder
                .as_ref()
                .ok_or_else(|| Error::invalid("image input has no vision encoder"))?;
            let embedding = encoder.encode(&image.pixels, None, cancelled)?;
            self.embeddings[index] = Some(Arc::new(embedding));
        }
        Ok(())
    }

    /// Writes the embeddings of image rows among tokens
    /// `position..position + count` into `hidden` (`count` rows of `hc`
    /// streams of `width`), after the token embedding; [`VisionInput::prepare`]
    /// must have covered the range.
    pub fn inject(
        &self,
        stream: &Stream,
        mut hidden: DeviceSliceMut<'_, f32>,
        position: u32,
        count: u32,
        width: u32,
        hc: u32,
    ) -> std::result::Result<(), LaunchError> {
        let Some(prompt) = &self.prompt else {
            return Ok(());
        };
        let row = width as usize * hc as usize;
        for (index, image) in prompt.images.iter().enumerate() {
            let grid = image.grid;
            let end = grid.offset + grid.height * grid.width;
            let (begin, stop) = (position.max(grid.offset), (position + count).min(end));
            if begin >= stop {
                continue;
            }
            let embedding =
                self.embeddings[index]
                    .as_ref()
                    .ok_or(LaunchError::InvalidArgument {
                        param: "embedding",
                        reason: "image rows injected before encoding",
                    })?;
            if embedding.width != width || embedding.rows != grid.height * grid.width {
                return Err(LaunchError::InvalidArgument {
                    param: "embedding",
                    reason: "vision embedding does not match language input",
                });
            }
            let rows = stop - begin;
            let target = (begin - position) as usize * row;
            let source = (begin - grid.offset) as usize * width as usize;
            k::vision_inject_rows(
                stream,
                hidden
                    .reborrow()
                    .range(target..target + rows as usize * row),
                embedding
                    .data
                    .slice(source..source + rows as usize * width as usize),
                rows,
                width,
                hc,
            )?;
        }
        Ok(())
    }

    /// Device bytes held for the session: the table and the embeddings.
    pub fn bytes(&self) -> usize {
        let table = self.table.as_ref().map_or(0, |t| {
            16 + t.rows().map_or(0, |r| r.len() * size_of::<i32>())
        });
        let embeddings: usize = self
            .embeddings
            .iter()
            .flatten()
            .map(|e| e.rows as usize * e.width as usize * size_of::<f32>())
            .sum();
        table + embeddings
    }
}
