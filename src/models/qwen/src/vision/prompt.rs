//! Image prompts (`src/models/qwen/vision/prompt.cpp`): the rendered chat
//! with each `<|image_pad|>` replaced by the image's merged token grid, the
//! interleaved mRoPE layout of those grids, and the cache identity.

use std::sync::Arc;

use gufo_core::bpe::TokenId;
use gufo_core::image::{Image, decode_image};
use gufo_core::qwen_tokenizer::QwenTokenizer;
use gufo_core::text::{
    ChatMessage, ChatRequest, ChatTool, ContinuationInputPrefix, Error, PromptContext, Result,
};
use sha2::{Digest, Sha256};

use super::{Encoder, IMAGE_TOKEN, ImageCache, RESIZE_FACTOR, resize_image};
use crate::chat_template::{
    ChatTemplateOptions, PreparedPrompt, encode_rendered, render, render_and_tokenize,
    request_options, request_tools, stable_prefix_tokens,
};

/// An image's language-token grid.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImageGrid {
    /// First image-pad token.
    pub offset: u32,
    pub height: u32,
    pub width: u32,
}

/// `RopeLayout::Validate`: non-empty grids, in order, inside the context.
pub fn validate_image_grids(images: &[ImageGrid], max_context: u32) -> Result<(), String> {
    // Every non-overlapping image occupies at least one context position
    //.
    if images.len() > max_context as usize {
        return Err("image grids exceed model context".into());
    }
    let mut previous_end = 0u64;
    for image in images {
        let count = u64::from(image.height) * u64::from(image.width);
        if image.height == 0
            || image.width == 0
            || u64::from(image.offset) < previous_end
            || u64::from(image.offset) + count > u64::from(max_context)
            || max_context > i32::MAX as u32
        {
            return Err("invalid image position layout".into());
        }
        previous_end = u64::from(image.offset) + count;
    }
    Ok(())
}

/// The preprocessing version in the cache identity.
pub(super) const PREPROCESSING: &[u8] = b"qwen38-image-bicubic-rgb8-bf16-v1";

/// A prompt longer than the session context.
pub(crate) fn context_exceeded(max_context: u32) -> Error {
    Error::length(format!(
        "prompt exceeds the {max_context}-token context; increase --context or shorten the \
         conversation"
    ))
}

/// Request-owned position state: enough to restore RoPE without pixels or
/// encoder scratch, so continuation snapshots carry only the grids.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RopeLayout {
    pub images: Vec<ImageGrid>,
}

impl RopeLayout {
    /// The `[T, H, W]` rotary positions of a physical token: image tokens
    /// take their grid row and column from the image's start, later tokens
    /// shift by the images' `max(height, width) - height * width`.
    pub fn position(&self, physical: u32) -> [i32; 3] {
        let mut delta = 0i64;
        for image in &self.images {
            if physical < image.offset {
                break;
            }
            let count = u64::from(image.height) * u64::from(image.width);
            if u64::from(physical - image.offset) < count {
                let start = (i64::from(image.offset) + delta) as i32;
                let index = physical - image.offset;
                return [
                    start,
                    start + (index / image.width) as i32,
                    start + (index % image.width) as i32,
                ];
            }
            delta += i64::from(image.height.max(image.width)) - count as i64;
        }
        let value = (i64::from(physical) + delta) as i32;
        [value; 3]
    }

    /// The shift of every token after the last image.
    pub fn delta(&self) -> i32 {
        let end = self.prefix_length();
        self.position(end)[0] - end as i32
    }

    /// The tokens up to the end of the last image.
    pub fn prefix_length(&self) -> u32 {
        self.images
            .last()
            .map_or(0, |last| last.offset + last.height * last.width)
    }

    /// The images contributing at least one embedding to the first
    /// `token_count` tokens.
    pub fn prefix(&self, token_count: u32) -> RopeLayout {
        RopeLayout {
            images: self
                .images
                .iter()
                .take_while(|image| image.offset < token_count)
                .copied()
                .collect(),
        }
    }

    pub fn validate(&self, max_context: u32) -> Result<()> {
        validate_image_grids(&self.images, max_context).map_err(Error::invalid)
    }
}

/// A resized RGB8 image and its grid; the encoder repeats it temporally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedImage {
    /// Resized RGB8, shared with the runner's [`ImageCache`].
    pub pixels: Arc<Image>,
    pub grid: ImageGrid,
    /// The cache identity of the prompt through this image.
    pub prefix_identity: [u8; 32],
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Prompt {
    pub tokens: Vec<TokenId>,
    /// The tokens an interrupted assistant followed by a user turn leaves
    /// unchanged.
    pub stable_prefix_tokens: usize,
    pub rope: RopeLayout,
    pub images: Vec<Arc<PreparedImage>>,
    /// SHA-256 over the decoded pixels, their grids, the preprocessing
    /// version and the encoder identity; empty without images. The tokens
    /// remain a separate key.
    pub cache_identity: Vec<u8>,
}

impl Prompt {
    /// The cache identity of the first `token_count` tokens: that of the
    /// last image they reach, empty before the first.
    pub fn identity_for_prefix(&self, token_count: usize) -> &[u8] {
        self.images
            .iter()
            .rev()
            .find(|image| (image.grid.offset as usize) < token_count)
            .map_or(&[], |image| &image.prefix_identity)
    }
}

/// Renders and tokenizes `messages`, decoding and resizing every image.
/// `encoder_identity` is empty without a vision encoder, which only a
/// text-only conversation accepts.
pub fn prepare(
    tokenizer: &QwenTokenizer,
    messages: &[ChatMessage],
    tools: &[ChatTool],
    options: &ChatTemplateOptions,
    encoder_identity: &str,
    max_context: u32,
    image_cache: Option<&ImageCache>,
) -> Result<Prompt> {
    let rendered = render(messages, tools, options).map_err(Error::invalid)?;
    let text = rendered.text.as_str();
    let mut prompt = Prompt::default();
    let append = |prompt: &mut Prompt, begin: usize, end: usize| -> Result<()> {
        let tokens = encode_rendered(tokenizer, text, begin, end, &rendered.content_spans);
        if tokens.len() > max_context as usize - prompt.tokens.len() {
            return Err(context_exceeded(max_context));
        }
        prompt.tokens.extend_from_slice(&tokens);
        Ok(())
    };
    let mut identity = Sha256::new();
    identity.update(PREPROCESSING);
    identity.update(encoder_identity.as_bytes());
    let mut cursor = 0;
    let mut offsets = rendered.image_offsets.iter();
    for image in messages.iter().flat_map(|message| &message.images) {
        if encoder_identity.is_empty() {
            return Err(Error::invalid(
                "image input requires the model's matching mmproj",
            ));
        }
        let &offset = offsets
            .next()
            .ok_or_else(|| Error::logic("image rendering lost a part"))?;
        append(&mut prompt, cursor, offset)?;
        let pixels = match image_cache {
            Some(cache) => cache.get(&image.bytes)?,
            None => Arc::new(resize_image(&decode_image(&image.bytes)?)?),
        };
        let grid = ImageGrid {
            offset: prompt.tokens.len() as u32,
            height: pixels.height / RESIZE_FACTOR,
            width: pixels.width / RESIZE_FACTOR,
        };
        let count = grid.height as usize * grid.width as usize;
        if count > max_context as usize - prompt.tokens.len() {
            return Err(Error::length("image tokens exceed model context"));
        }
        for value in [grid.offset, grid.height, grid.width] {
            identity.update(value.to_le_bytes());
        }
        identity.update(&pixels.pixels);
        prompt
            .tokens
            .resize(prompt.tokens.len() + count, IMAGE_TOKEN);
        prompt.rope.images.push(grid);
        prompt.images.push(Arc::new(PreparedImage {
            pixels,
            grid,
            prefix_identity: identity.clone().finalize().into(),
        }));
        cursor = offset + "<|image_pad|>".len();
    }
    append(&mut prompt, cursor, text.len())?;
    prompt.stable_prefix_tokens = stable_prefix_tokens(tokenizer, &rendered, &prompt.tokens)?;
    prompt.rope.validate(max_context)?;
    if !prompt.images.is_empty() {
        prompt.cache_identity = identity.finalize().to_vec();
    }
    Ok(prompt)
}

/// A request's image prompt as the serving prompt context
/// (`QwenImageContext`); its identity joins the tokens as cache key.
#[derive(Debug)]
pub struct ImageContext {
    pub prompt: Arc<Prompt>,
    /// The identity of the prompt before each image, so an appended image
    /// keeps the checkpoints of the earlier text and images.
    pub cache_prefixes: Vec<ContinuationInputPrefix>,
}

impl ImageContext {
    pub fn new(prompt: Arc<Prompt>) -> Self {
        let cache_prefixes = prompt
            .images
            .iter()
            .map(|image| {
                let offset = image.grid.offset as usize;
                ContinuationInputPrefix {
                    token_count: offset,
                    identity: prompt.identity_for_prefix(offset).to_vec(),
                }
            })
            .collect();
        Self {
            prompt,
            cache_prefixes,
        }
    }
}

impl PromptContext for ImageContext {
    fn cache_identity(&self) -> &[u8] {
        &self.prompt.cache_identity
    }

    fn cache_prefixes(&self) -> &[ContinuationInputPrefix] {
        &self.cache_prefixes
    }
}

/// `PrepareQwenPrompt` (`src/cli/serve/inference_backend.cpp`): the tokens,
/// the snapshot boundary before the generation suffix, and the image
/// context when the request has images. The sidecar identity enters only
/// for requests with images.
pub fn prepare_request(
    tokenizer: &QwenTokenizer,
    request: &ChatRequest,
    encoder: Option<&Encoder>,
    max_context: u32,
    image_cache: &ImageCache,
) -> Result<(PreparedPrompt, Option<Arc<ImageContext>>)> {
    let has_images = request.messages.iter().any(|m| !m.images.is_empty());
    let options = request_options(request, max_context);
    let identity = match encoder {
        Some(encoder) if has_images => encoder.identity(),
        _ => "",
    };
    let prompt = prepare(
        tokenizer,
        &request.messages,
        request_tools(request),
        &options,
        identity,
        max_context,
        // A request that opts out of the prompt cache leaves no images
        // behind either.
        request.cache_prompt.then_some(image_cache),
    )?;
    let cache_prefix_tokens = prompt.stable_prefix_tokens;
    let tokens = prompt.tokens.clone();
    let context =
        (!prompt.images.is_empty()).then(|| Arc::new(ImageContext::new(Arc::new(prompt))));
    Ok((
        PreparedPrompt {
            tokens,
            cache_prefix_tokens,
        },
        context,
    ))
}

/// `CountQwenPromptTokens`: the prompt
/// tokens of `request` as generation prepares them, without admission to
/// an execution context, the vision encoder or the image cache. Images take
/// generation's resizing, token expansion and projector check; only the
/// layout's signed position limit applies, so a client can size a prompt
/// longer than the context before trimming it. `None` when the template
/// does not render.
pub fn count_request_tokens(
    tokenizer: &QwenTokenizer,
    request: &ChatRequest,
    encoder: Option<&Encoder>,
    max_context: u32,
) -> Result<Option<usize>> {
    let options = request_options(request, max_context);
    let tools = request_tools(request);
    if request.messages.iter().all(|m| m.images.is_empty()) {
        return Ok(
            render_and_tokenize(tokenizer, &request.messages, tools, &options)
                .ok()
                .map(|tokens| tokens.len()),
        );
    }
    let identity = encoder.map_or("", Encoder::identity);
    let prompt = prepare(
        tokenizer,
        &request.messages,
        tools,
        &options,
        identity,
        i32::MAX as u32,
        None,
    )?;
    Ok(Some(prompt.tokens.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_continue_after_image_grids() {
        // Tokens 0..2 text, a 2x3 grid at 2, then text.
        let layout = RopeLayout {
            images: vec![ImageGrid {
                offset: 2,
                height: 2,
                width: 3,
            }],
        };
        assert_eq!(layout.position(1), [1, 1, 1]);
        assert_eq!(layout.position(2), [2, 2, 2]);
        assert_eq!(layout.position(4), [2, 2, 4]);
        assert_eq!(layout.position(5), [2, 3, 2]);
        assert_eq!(layout.prefix_length(), 8);
        // After the grid: 8 + (max(2, 3) - 6) = 5.
        assert_eq!(layout.position(8), [5, 5, 5]);
        assert_eq!(layout.delta(), -3);
        assert_eq!(RopeLayout::default().delta(), 0);
    }

    #[test]
    fn prefixes_keep_the_images_they_consumed() {
        let grid = |offset, height, width| ImageGrid {
            offset,
            height,
            width,
        };
        let layout = RopeLayout {
            images: vec![grid(5, 2, 3), grid(15, 3, 2)],
        };
        assert!(layout.prefix(5).images.is_empty());
        assert_eq!(layout.prefix(6).images.len(), 1);
        assert_eq!(layout.prefix(15).images.len(), 1);
        assert_eq!(layout.prefix(16), layout);
        for count in 0..32 {
            let prefix = layout.prefix(count);
            for token in 0..count {
                assert_eq!(prefix.position(token), layout.position(token));
            }
        }
        let image = |grid, fill| {
            Arc::new(PreparedImage {
                pixels: Arc::new(Image::new(32, 32, vec![0; 32 * 32 * 3]).unwrap()),
                grid,
                prefix_identity: [fill; 32],
            })
        };
        let prompt = Prompt {
            images: vec![image(layout.images[0], 1), image(layout.images[1], 2)],
            ..Prompt::default()
        };
        assert!(prompt.identity_for_prefix(5).is_empty());
        assert_eq!(prompt.identity_for_prefix(6), [1; 32]);
        assert_eq!(prompt.identity_for_prefix(15), [1; 32]);
        assert_eq!(prompt.identity_for_prefix(16), [2; 32]);
        // Each image's boundary carries the identity of what precedes it.
        let context = ImageContext::new(Arc::new(prompt));
        assert_eq!(
            context.cache_prefixes,
            [
                ContinuationInputPrefix {
                    token_count: 5,
                    identity: Vec::new(),
                },
                ContinuationInputPrefix {
                    token_count: 15,
                    identity: vec![1; 32],
                },
            ]
        );
    }
}
