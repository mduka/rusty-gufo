//! The Qwen3.8 vision path shared by the 27B and Flash-Next language trunks
//! (`src/models/qwen/vision`): image prompt preparation and the interleaved
//! mRoPE layout. Each model supplies its own projector width.

mod encoder;
mod image_cache;
mod input;
mod prompt;
mod resize;

pub use encoder::{CancellationCheck, Embedding, Encoder, Observer, SIDECAR};
pub use image_cache::{ImageCache, ImageCacheStats};
pub use input::VisionInput;
pub(crate) use prompt::context_exceeded;
pub use prompt::{
    ImageContext, ImageGrid, PreparedImage, Prompt, RopeLayout, count_request_tokens, prepare,
    prepare_request, validate_image_grids,
};
pub use resize::resize_image;

/// The `<|image_pad|>` token each merged grid cell occupies.
pub const IMAGE_TOKEN: u32 = 248056;
/// ViT patch edge in pixels.
pub const PATCH_SIZE: u32 = 16;
/// Patches merged per language token along each axis.
pub const MERGE_SIZE: u32 = 2;
/// Image edges are multiples of this many pixels.
pub const RESIZE_FACTOR: u32 = PATCH_SIZE * MERGE_SIZE;
/// The largest resized image the encoder accepts.
pub const MAX_ENCODER_PIXELS: u64 = 16_777_216;
