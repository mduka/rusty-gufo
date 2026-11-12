//! Image placeholders of a rendered prompt expanded into their embedding
//! rows, with the cache identities of the images (`vision_prompt.cpp`).

use std::sync::Arc;

use gufo_core::image::decode_image;
use sha2::{Digest, Sha256};

use crate::config::VisionConfig;
use crate::tokenizer::{TokenId, Tokenizer};
use crate::vision_image::{ImageGrid, image_tokens, plan_grid, preprocess_image};

/// Names what an image's embedding rows are computed from; changing the
/// preprocessing must change it, so cached states of other prompts don't
/// match.
const PREPROCESSING: &str = "nemotron-omni-image-v1;torch-bicubic-aa-f32;clip-normalized";

fn add_u32(hasher: &mut Sha256, value: u32) {
    hasher.update(value.to_le_bytes());
}

fn add_text(hasher: &mut Sha256, text: &str) {
    add_u32(hasher, text.len() as u32);
    hasher.update(text.as_bytes());
}

/// The tokens around an image's embedding rows. The chat template renders
/// `placeholder` (`<image>`) per image; NVIDIA's processor replaces it with
/// `start` (`<img>`), one `placeholder` per embedding row and `end`
/// (`</img>`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImageFraming {
    pub placeholder: TokenId,
    pub start: TokenId,
    pub end: TokenId,
}

/// The framing tokens of `tokenizer` (`FindImageFraming`); an error when
/// one is not a single token.
pub fn find_image_framing(tokenizer: &Tokenizer) -> Result<ImageFraming, String> {
    let single = |text: &str| {
        let ids = tokenizer.encode(text, true);
        if ids.len() != 1 {
            return Err(format!(
                "Nemotron vocabulary has no single token for {text}"
            ));
        }
        Ok(ids[0])
    };
    Ok(ImageFraming {
        placeholder: single("<image>")?,
        start: single("<img>")?,
        end: single("</img>")?,
    })
}

/// One image of a request: its encoded bytes (PNG, JPEG or WebP) and whether
/// the client asked for low detail, which caps it at min_patches patches.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VisionInput {
    pub bytes: Option<Arc<Vec<u8>>>,
    pub low_detail: bool,
}

/// One image of an expanded prompt.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PromptImage {
    pub grid: ImageGrid,
    /// Position of its first embedding row, after its `<img>`.
    pub offset: u32,
    /// Embedding rows: one per merged token.
    pub rows: u32,
    /// The preprocessed patches; `None` when the prompt was only sized.
    pub patches: Option<Arc<Vec<f32>>>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct VisionPrompt {
    pub tokens: Vec<TokenId>,
    pub images: Vec<PromptImage>,
    /// Identity of every image (empty without images), and per image the
    /// identity of the images before it (empty for the first). Each covers
    /// the preprocessing, the encoder, the image's place, grid and pixels.
    pub cache_identity: Vec<u8>,
    pub prefix_identities: Vec<Vec<u8>>,
    /// Index of each placeholder among the rendered tokens.
    pub placeholders: Vec<usize>,
}

impl VisionPrompt {
    /// Position in `tokens` of the rendered token at `rendered`
    /// (`Expanded`).
    pub fn expanded(&self, rendered: usize) -> usize {
        let mut position = rendered;
        for (i, &placeholder) in self.placeholders.iter().enumerate() {
            if placeholder >= rendered {
                break;
            }
            position += self.images[i].rows as usize + 1;
        }
        position
    }
}

/// Expands each placeholder of the rendered prompt into its image's span,
/// decoding and, with `preprocess`, resizing the images in order; without it
/// the prompt is only sized (`ExpandImages`). `encoder_identity` names the
/// encoder whose embeddings the identities stand for. Fails when
/// placeholders and images disagree or an image can't be decoded or sized.
pub fn expand_images(
    config: &VisionConfig,
    framing: &ImageFraming,
    rendered: &[TokenId],
    images: &[VisionInput],
    encoder_identity: &str,
    preprocess: bool,
) -> Result<VisionPrompt, String> {
    let mut prompt = VisionPrompt::default();
    for (i, &token) in rendered.iter().enumerate() {
        if token == framing.placeholder {
            prompt.placeholders.push(i);
        }
    }
    if prompt.placeholders.len() != images.len() {
        return Err(format!(
            "the prompt has {} image placeholders for {} images",
            prompt.placeholders.len(),
            images.len()
        ));
    }
    let mut all = Sha256::new();
    let mut next = 0;
    for (k, input) in images.iter().enumerate() {
        let label = format!("image {}", k + 1);
        let bytes = input
            .bytes
            .as_ref()
            .ok_or_else(|| format!("{label} has no data"))?;
        let image = decode_image(bytes).map_err(|e| format!("{label} cannot be decoded: {e}"))?;
        let budget = if input.low_detail {
            config.min_patches
        } else {
            0
        };
        let grid = plan_grid(config, image.width, image.height, budget).ok_or_else(|| {
            format!(
                "{label} ({}x{}) is too elongated to encode",
                image.width, image.height
            )
        })?;
        let at = prompt.placeholders[k];
        prompt.tokens.extend_from_slice(&rendered[next..at]);
        next = at + 1;
        let mut out = PromptImage {
            grid,
            rows: image_tokens(config, &grid),
            ..PromptImage::default()
        };
        prompt.tokens.push(framing.start);
        out.offset = prompt.tokens.len() as u32;
        prompt
            .tokens
            .extend(std::iter::repeat_n(framing.placeholder, out.rows as usize));
        prompt.tokens.push(framing.end);
        if preprocess {
            out.patches = Some(Arc::new(preprocess_image(config, &image, &grid)?));
        }
        if k == 0 {
            prompt.prefix_identities.push(Vec::new());
        } else {
            prompt
                .prefix_identities
                .push(all.clone().finalize().to_vec());
        }
        let mut one = Sha256::new();
        add_text(&mut one, PREPROCESSING);
        add_text(&mut one, encoder_identity);
        add_u32(&mut one, out.offset);
        add_u32(&mut one, grid.rows);
        add_u32(&mut one, grid.cols);
        add_u32(&mut one, image.width);
        add_u32(&mut one, image.height);
        one.update(&image.pixels);
        all.update(one.finalize());
        prompt.images.push(out);
    }
    prompt.tokens.extend_from_slice(&rendered[next..]);
    if !images.is_empty() {
        prompt.cache_identity = all.finalize().to_vec();
    }
    Ok(prompt)
}
