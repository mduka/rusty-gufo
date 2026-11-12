//! Image prompts (`vision/prompt.cpp`): Gemma4ImageProcessor's
//! aspect-preserving resize to a soft-token budget, and the expansion of each
//! image placeholder into `<|image>`, its soft tokens and `<image|>`, with
//! the cache identity of every image prefix.

use gufo_core::image::{Image, decode_image};
use gufo_core::text::{ChatMessage, ChatTool, Error, Result};
use sha2::{Digest, Sha256};

use crate::chat_template::{ChatOptions, IMAGE_PLACEHOLDER, render};
use crate::tokenizer::{TokenId, Tokenizer};
use crate::vision_resize::resize_bicubic;

pub const PATCH_SIZE: u32 = 16;
/// Average-pooling kernel over the patch grid; one soft token per 3x3.
pub const POOL_SIZE: u32 = 3;
pub const SIDE_MULTIPLE: u32 = PATCH_SIZE * POOL_SIZE;
/// Gemma4ImageProcessor's default `max_soft_tokens`.
pub const DEFAULT_SOFT_TOKENS: u32 = 280;
/// The soft-token budgets the checkpoint supports (model card, "variable
/// image resolution"): more tokens keep more detail at a higher cost.
pub const SOFT_TOKEN_BUDGETS: [u32; 5] = [70, 140, 280, 560, 1120];

pub fn is_soft_token_budget(tokens: u32) -> bool {
    SOFT_TOKEN_BUDGETS.contains(&tokens)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

/// Gemma4ImageProcessor's aspect-preserving target: the largest
/// multiple-of-48 size with at most `soft_tokens` pooled cells, filling the
/// budget whether that enlarges or shrinks the image (`ResizedSize`).
pub fn resized_size(width: u32, height: u32, soft_tokens: u32) -> Result<ImageSize> {
    if width == 0 || height == 0 || soft_tokens == 0 {
        return Err(Error::invalid("invalid image size or token budget"));
    }
    // transformers get_aspect_ratio_preserving_size, in the same double math.
    let max_patches = f64::from(soft_tokens) * f64::from(POOL_SIZE) * f64::from(POOL_SIZE);
    let target_pixels = max_patches * f64::from(PATCH_SIZE) * f64::from(PATCH_SIZE);
    let factor = (target_pixels / (f64::from(height) * f64::from(width))).sqrt();
    let side = f64::from(SIDE_MULTIPLE);
    let multiple = u64::from(SIDE_MULTIPLE);
    let mut target_height = (factor * f64::from(height) / side).floor() as u64 * multiple;
    let mut target_width = (factor * f64::from(width) / side).floor() as u64 * multiple;
    if target_height == 0 && target_width == 0 {
        return Err(Error::invalid("image resizes to 0x0"));
    }
    let max_side = u64::from(soft_tokens) * multiple;
    if target_height == 0 {
        target_height = multiple;
        target_width = (u64::from(width / height) * multiple).min(max_side);
    } else if target_width == 0 {
        target_width = multiple;
        target_height = (u64::from(height / width) * multiple).min(max_side);
    }
    if (target_height * target_width) as f64 > target_pixels {
        return Err(Error::invalid("image resize exceeds its patch budget"));
    }
    Ok(ImageSize {
        width: target_width as u32,
        height: target_height as u32,
    })
}

/// [`resized_size`] applied with the bicubic resize (`ResizeImage`).
pub fn resize_image(image: &Image, soft_tokens: u32) -> Result<Image> {
    let size = resized_size(image.width, image.height, soft_tokens)?;
    resize_bicubic(image, size.width, size.height).map_err(Error::invalid)
}

/// `<|image>`, the soft token `<|image|>` and `<image|>`, looked up by text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageTokens {
    pub start: TokenId,
    pub soft: TokenId,
    pub end: TokenId,
}

pub fn find_image_tokens(tokenizer: &Tokenizer) -> Result<ImageTokens> {
    let find = |text: &str| tokenizer.find_token(text.as_bytes());
    match (find("<|image>"), find(IMAGE_PLACEHOLDER), find("<image|>")) {
        (Some(start), Some(soft), Some(end)) => Ok(ImageTokens { start, soft, end }),
        _ => Err(Error::invalid("gemma4 vocabulary lacks image tokens")),
    }
}

/// Soft-token rows of one image. They attend to each other bidirectionally
/// in sliding layers, so a forward never splits them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImageSpan {
    /// First soft token.
    pub offset: u32,
    pub rows: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedImage {
    /// Resized RGB8, both sides multiples of 48.
    pub pixels: Image,
    pub span: ImageSpan,
    /// Covers every image up to and including this one, with its placement,
    /// the preprocessing version and the encoder identity.
    pub prefix_identity: [u8; 32],
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Prompt {
    pub tokens: Vec<TokenId>,
    /// Tokens before the generation prompt; zero when it does not tokenize
    /// apart.
    pub stable_prefix_tokens: usize,
    pub images: Vec<PreparedImage>,
    /// Identity of all images; empty for a text-only prompt.
    pub cache_identity: Vec<u8>,
}

impl Prompt {
    /// Identity of the images starting before `token_count`; empty when none
    /// does (`IdentityForPrefix`).
    pub fn identity_for_prefix(&self, token_count: usize) -> &[u8] {
        self.images
            .iter()
            .rev()
            .find(|image| (image.span.offset as usize) < token_count)
            .map_or(&[], |image| &image.prefix_identity)
    }
}

fn hash_u32(hash: &mut Sha256, value: u32) {
    hash.update(value.to_le_bytes());
}

/// Renders the conversation and expands each image into `<|image>`, its soft
/// tokens and `<image|>`. Images are decoded and resized here, once
/// (`Prepare`).
pub fn prepare(
    tokenizer: &Tokenizer,
    messages: &[ChatMessage],
    tools: &[ChatTool],
    options: &ChatOptions,
    encoder_identity: &str,
    max_context: u32,
    soft_tokens: u32,
) -> Result<Prompt> {
    if !is_soft_token_budget(soft_tokens) {
        return Err(Error::invalid(
            "Gemma 4 image budget must be 70, 140, 280, 560 or 1120 tokens",
        ));
    }
    let has_images = messages.iter().any(|m| !m.images.is_empty());
    if has_images && encoder_identity.is_empty() {
        return Err(Error::invalid(
            "image input requires the model's matching --mmproj sidecar",
        ));
    }
    let mut offsets = Vec::new();
    let rendered = render(messages, tools, options, Some(&mut offsets))
        .map_err(|e| Error::invalid(format!("Gemma 4 chat template: {e}")))?;
    let text = rendered.text.as_str();
    let max_context = max_context as usize;
    let mut prompt = Prompt::default();
    let append = |prompt: &mut Prompt, piece: &str| -> Result<()> {
        let tokens = tokenizer.encode(piece.as_bytes(), false, true);
        if tokens.len() > max_context.saturating_sub(prompt.tokens.len()) {
            return Err(Error::length(format!(
                "prompt exceeds the {max_context}-token context; increase --context or shorten \
                 the conversation"
            )));
        }
        prompt.tokens.extend_from_slice(&tokens);
        Ok(())
    };

    let mut identity = Sha256::new();
    identity.update(b"gemma4-image-bicubic-rgb8-bf16-v1");
    identity.update(encoder_identity.as_bytes());
    hash_u32(&mut identity, soft_tokens);
    let image_tokens = if has_images {
        Some(find_image_tokens(tokenizer)?)
    } else {
        None
    };
    let mut cursor = 0;
    let mut index = 0;
    for message in messages {
        for image in &message.images {
            let (Some(&offset), Some(tokens)) = (offsets.get(index), image_tokens) else {
                return Err(Error::logic("image rendering lost a part"));
            };
            append(&mut prompt, &text[cursor..offset])?;
            let pixels = resize_image(&decode_image(&image.bytes)?, soft_tokens)?;
            let rows = (pixels.width / SIDE_MULTIPLE) * (pixels.height / SIDE_MULTIPLE);
            if rows as usize + 2 > max_context.saturating_sub(prompt.tokens.len()) {
                return Err(Error::length("image tokens exceed model context"));
            }
            prompt.tokens.push(tokens.start);
            let span = ImageSpan {
                offset: prompt.tokens.len() as u32,
                rows,
            };
            prompt
                .tokens
                .extend(std::iter::repeat_n(tokens.soft, rows as usize));
            prompt.tokens.push(tokens.end);
            hash_u32(&mut identity, span.offset);
            hash_u32(&mut identity, pixels.width);
            hash_u32(&mut identity, pixels.height);
            identity.update(&pixels.pixels);
            let prefix_identity: [u8; 32] = identity.clone().finalize().into();
            prompt.images.push(PreparedImage {
                pixels,
                span,
                prefix_identity,
            });
            cursor = offset + IMAGE_PLACEHOLDER.len();
            index += 1;
        }
    }
    if index != offsets.len() {
        return Err(Error::logic("image rendering added a part"));
    }
    append(&mut prompt, &text[cursor..])?;

    // The stable prefix ends before a special token, after every image.
    if rendered.stable_prefix_offset < text.len() {
        let suffix = tokenizer.encode(
            &text.as_bytes()[rendered.stable_prefix_offset..],
            false,
            true,
        );
        if suffix.len() < prompt.tokens.len() && prompt.tokens.ends_with(&suffix) {
            prompt.stable_prefix_tokens = prompt.tokens.len() - suffix.len();
        }
    }
    if !prompt.images.is_empty() {
        prompt.cache_identity = identity.finalize().to_vec();
    }
    Ok(prompt)
}
