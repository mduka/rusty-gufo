//! The generation (`hip/pipeline.cpp`): the Qwen3-VL text encoder with its
//! vision tower and DeepStack taps, the reference images' VAE latents, the
//! single-stream DiT with block-causal conditioning whose condition K/V is
//! cached after the first step, the flow-matching Euler steps and the VAE
//! decoder. Every operation and host computation follows the C++ order, so
//! the device plans, allocations and results are the C++ ones.

use std::path::Path;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use gufo_core::text::{Error, Result};

use crate::host::fill_rows;
use crate::image::{Image, Request, flow_sigmas, resize_image, validate_request};
use crate::random::{Mt19937_64, Normal};
use crate::runtime::{Activation, Matrix, Norm, Observer, Runtime};
use crate::tokenizer::Tokenizer;
use crate::weights::Weights;

/// `<|image_pad|>`.
const IMAGE_PAD: u32 = 151_655;
/// The pipeline's system prompt; its tokens are dropped from the encoder
/// output.
const SYSTEM: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";

/// `Metrics`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Metrics {
    pub prompt_ms: f64,
    pub denoise_ms: f64,
    pub vae_ms: f64,
    pub total_ms: f64,
    pub resident_weight_bytes: usize,
}

/// `Result`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Generation {
    pub image: Image,
    pub metrics: Metrics,
}

/// Host BF16 rounding (`Bf16` in `pipeline.cpp`), for finite values.
fn bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

/// `std::nearbyint` (ties to even) of `x` in `[0, 255]`, as a byte: adding
/// 2^23 rounds to an integer under the default rounding mode, and
/// subtracting it again is exact.
fn round_byte(x: f32) -> u8 {
    const SHIFT: f32 = 8_388_608.0;
    ((x + SHIFT) - SHIFT) as u8
}

fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// The state of one request.
struct Context<'a, 'o> {
    rt: &'a mut Runtime,
    weights: &'a Weights,
    cancelled: Option<&'a dyn Fn() -> bool>,
    observer: Option<&'a mut (dyn Observer + 'o)>,
}

impl Context<'_, '_> {
    fn cancel(&self) -> Result<()> {
        if self.cancelled.is_some_and(|cancelled| cancelled()) {
            return Err(Error::runtime("image request cancelled"));
        }
        Ok(())
    }

    fn observe(&mut self, name: &str, matrix: &Matrix) -> Result<()> {
        let observer: Option<&mut dyn Observer> = match &mut self.observer {
            Some(observer) => Some(&mut **observer),
            None => None,
        };
        self.rt.observe(observer, name, matrix)
    }

    fn linear(&mut self, x: &Matrix, name: &str) -> Result<Matrix> {
        self.rt.linear(self.weights, x, name)
    }

    fn normalize(&mut self, x: &Matrix, kind: Norm, weight: &str) -> Result<Matrix> {
        self.rt
            .normalize(self.weights, x, kind, Some(weight), 1e-6, false)
    }

    /// The VAE's channel RMSNorm followed by SiLU.
    fn normalize_silu(&mut self, x: &Matrix, weight: &str) -> Result<Matrix> {
        self.rt
            .normalize(self.weights, x, Norm::Vae, Some(weight), 1e-6, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        x: &Matrix,
        height: usize,
        width: usize,
        name: &str,
        stride: usize,
        padding: i32,
        upscale: usize,
    ) -> Result<Matrix> {
        self.rt.conv(
            self.weights,
            x,
            height,
            width,
            name,
            stride,
            padding,
            upscale,
        )
    }
}

// ---- Vision ----

struct Visual {
    embedded: Matrix,
    deep: [Option<Matrix>; 3],
}

/// `VisionMerge`: 2x2 patch merger.
fn vision_merge(
    cx: &mut Context<'_, '_>,
    x: &Matrix,
    name: &str,
    postshuffle: bool,
) -> Result<Matrix> {
    let input = if postshuffle {
        x.reshape(x.rows / 4, x.cols * 4)?
    } else {
        x.clone()
    };
    let norm = cx.normalize(&input, Norm::Layer, &format!("{name}.norm"))?;
    let flat = norm.reshape(x.rows / 4, x.cols * 4)?;
    let hidden = cx.linear(&flat, &format!("{name}.linear_fc1"))?;
    let hidden = cx.rt.activate(&hidden, Activation::Gelu)?;
    cx.linear(&hidden, &format!("{name}.linear_fc2"))
}

/// `EncodeVision`: the Qwen3-VL vision tower over a 32-pixel aligned
/// image, in the trained 2x2 merge order.
fn encode_vision(cx: &mut Context<'_, '_>, image: &Image) -> Result<Visual> {
    let height = (image.height / 16) as usize;
    let width = (image.width / 16) as usize;
    if height < 2 || width < 2 || !height.is_multiple_of(2) || !width.is_multiple_of(2) {
        return Err(Error::invalid("vision input must be aligned to 32 pixels"));
    }
    const PATCH: usize = 3 * 2 * 16 * 16;
    let tokens = height * width;
    let image_width = image.width as usize;
    // The trained 2x2 merge order of the patches.
    let position = |token: usize| {
        let py = (token / 4 / (width / 2)) * 2 + (token / 2) % 2;
        let px = ((token / 4) % (width / 2)) * 2 + token % 2;
        (py, px)
    };
    let mut patches = vec![0.0f32; tokens * PATCH];
    fill_rows(&mut patches, PATCH, |first, rows| {
        for (index, patch) in rows.as_chunks_mut::<PATCH>().0.iter_mut().enumerate() {
            let (py, px) = position(first + index);
            for c in 0..3 {
                for t in 0..2 {
                    for y in 0..16 {
                        for x in 0..16 {
                            let pixel = ((py * 16 + y) * image_width + px * 16 + x) * 4;
                            let alpha = i32::from(image.rgba[pixel + 3]);
                            // PIL's paste over white: DIV255 with integer rounding.
                            let product = i32::from(image.rgba[pixel + c]) * alpha
                                + 255 * (255 - alpha)
                                + 128;
                            let color = (product + (product >> 8)) >> 8;
                            patch[((c * 2 + t) * 16 + y) * 16 + x] =
                                (color as f32 / 255.0 - 0.5) / 0.5;
                        }
                    }
                }
            }
        }
    });
    // `std::pow` depends on the pair only: computed once per pair, the
    // same values as per cell.
    let divisors: [f32; 18] = std::array::from_fn(|pair| 10000.0f32.powf(pair as f32 / 18.0));
    let mut frequencies = vec![0.0f32; tokens * 72];
    fill_rows(&mut frequencies, 72, |first, rows| {
        for (index, row) in rows.as_chunks_mut::<72>().0.iter_mut().enumerate() {
            let (py, px) = position(first + index);
            for pair in 0..36 {
                let position = if pair < 18 { py } else { px };
                let angle = position as f32 / divisors[pair % 18];
                row[pair * 2] = angle.cos();
                row[pair * 2 + 1] = angle.sin();
            }
        }
    });
    // Temporaries of a C++ full expression die at its end: each block
    // below frees them before the next statement allocates, so the pool
    // holds the C++ tensors at every allocation.
    let prefix = "text_encoder.model.visual.";
    let mut x = {
        let uploaded = cx.rt.upload(&patches, tokens, PATCH)?;
        cx.linear(&uploaded, &format!("{prefix}patch_embed.proj"))?
    };
    x = cx.rt.vision_position(cx.weights, &x, height, width)?;
    cx.observe("vision.embedding", &x)?;
    let rope = cx.rt.upload_float(&frequencies, tokens, 72)?;
    let mut deep: [Option<Matrix>; 3] = [None, None, None];
    for layer in 0..27 {
        cx.cancel()?;
        let p = format!("{prefix}blocks.{layer}");
        let norm = cx.normalize(&x, Norm::Layer, &format!("{p}.norm1"))?;
        let qkv = cx.linear(&norm, &format!("{p}.attn.qkv"))?;
        let q = {
            let columns = cx.rt.columns(&qkv, 0, 1152)?;
            cx.rt.rope(&columns, 16, 72, &rope, 1)?
        };
        let k = {
            let columns = cx.rt.columns(&qkv, 1152, 1152)?;
            cx.rt.rope(&columns, 16, 72, &rope, 1)?
        };
        let v = cx.rt.columns(&qkv, 2304, 1152)?;
        {
            let attention = cx.rt.attention(&q, &k, &v, 16, 16, 72, &[])?;
            let projected = cx.linear(&attention, &format!("{p}.attn.proj"))?;
            x = cx.rt.add(&x, &projected)?;
        }
        {
            let norm = cx.normalize(&x, Norm::Layer, &format!("{p}.norm2"))?;
            let hidden = cx.linear(&norm, &format!("{p}.mlp.linear_fc1"))?;
            let activated = cx.rt.activate(&hidden, Activation::GeluTanh)?;
            let mlp = cx.linear(&activated, &format!("{p}.mlp.linear_fc2"))?;
            x = cx.rt.add(&x, &mlp)?;
        }
        cx.observe(&format!("vision.block.{layer}"), &x)?;
        if layer == 8 || layer == 16 || layer == 24 {
            let index = layer / 8 - 1;
            deep[index] = Some(vision_merge(
                cx,
                &x,
                &format!("{prefix}deepstack_merger_list.{index}"),
                true,
            )?);
        }
    }
    let embedded = vision_merge(cx, &x, &format!("{prefix}merger"), false)?;
    cx.observe("vision.output", &embedded)?;
    Ok(Visual { embedded, deep })
}

// ---- Prompt ----

struct EncodedPrompt {
    hidden: Matrix,
    ids: Vec<u32>,
    images: Vec<Image>,
    drop: usize,
}

/// `EncodePrompt`: the chat prompt with one image placeholder per
/// reference, through the text encoder; the hidden states before the final
/// norm, without the system prompt.
fn encode_prompt(
    cx: &mut Context<'_, '_>,
    tokenizer: &Tokenizer,
    request: &Request,
) -> Result<EncodedPrompt> {
    let mut images = Vec::with_capacity(request.images.len());
    let mut text = String::from(SYSTEM);
    text.push_str("<|im_start|>user\n");
    for (i, image) in request.images.iter().enumerate() {
        if i > 0 {
            text.push(' ');
        }
        text.push_str(&format!(
            "<image{}><|vision_start|><|image_pad|><|vision_end|>",
            i + 1
        ));
        let ratio = f64::from(image.width) / f64::from(image.height);
        let width = (1024.0 * 1024.0 * ratio).sqrt();
        let height = width / ratio;
        let rw = (width / 32.0).round_ties_even() as i32 * 32;
        let rh = (height / 32.0).round_ties_even() as i32 * 32;
        if !(32..=8192).contains(&rw) || !(32..=8192).contains(&rh) {
            return Err(Error::invalid(
                "reference image aspect ratio exceeds limits",
            ));
        }
        images.push(resize_image(image, rw, rh)?);
    }
    text.push_str(if request.prompt.is_empty() {
        " "
    } else {
        &request.prompt
    });
    text.push_str("<|im_end|>\n<|im_start|>assistant\n");
    let raw_ids = tokenizer.encode(&text);
    let mut ids = Vec::with_capacity(raw_ids.len());
    let mut image_positions = Vec::new();
    let mut positions: Vec<[i32; 3]> = Vec::new();
    let mut visuals = Vec::new();
    let mut image_index = 0;
    let mut position = 0i32;
    for id in raw_ids {
        if id != IMAGE_PAD {
            ids.push(id);
            positions.push([position; 3]);
            position += 1;
            continue;
        }
        let image = images
            .get(image_index)
            .ok_or_else(|| Error::invalid("prompt contains an unmatched image token"))?;
        image_index += 1;
        let (h, w) = (image.height / 32, image.width / 32);
        for y in 0..h {
            for x in 0..w {
                image_positions.push(ids.len());
                ids.push(IMAGE_PAD);
                positions.push([position, position + y, position + x]);
            }
        }
        position += h.max(w);
        visuals.push(encode_vision(cx, image)?);
    }
    if image_index != images.len() {
        return Err(Error::logic("image prompt layout mismatch"));
    }
    if ids.len() > 16384 {
        return Err(Error::invalid("image prompt exceeds 16384 encoder tokens"));
    }
    let drop = tokenizer.encode(SYSTEM).len();
    let prefix = "text_encoder.model.language_model.";
    let mut x = {
        let table = cx
            .rt
            .weight(cx.weights, &format!("{prefix}embed_tokens.weight"))?;
        cx.rt.embed(&table, &ids)?
    };
    let mut deep: Vec<Matrix> = Vec::new();
    if !visuals.is_empty() {
        let embeds: Vec<&Matrix> = visuals.iter().map(|v| &v.embedded).collect();
        {
            let values = cx.rt.concat(&embeds)?;
            x = cx.rt.replace_rows(&x, &values, &image_positions, false)?;
        }
        for i in 0..3 {
            let embeds = visuals
                .iter()
                .map(|v| {
                    v.deep[i]
                        .as_ref()
                        .ok_or_else(|| Error::logic("missing DeepStack tap"))
                })
                .collect::<Result<Vec<_>>>()?;
            deep.push(cx.rt.concat(&embeds)?);
        }
    }
    let rows = ids.len();
    let divisors: [f32; 64] = std::array::from_fn(|pair| 5_000_000.0f32.powf(pair as f32 / 64.0));
    let mut frequencies = vec![0.0f32; rows * 128];
    fill_rows(&mut frequencies, 128, |first, table| {
        for (row, position) in table
            .as_chunks_mut::<128>()
            .0
            .iter_mut()
            .zip(&positions[first..])
        {
            for pair in 0..64 {
                let axis = if pair < 60 { pair % 3 } else { 0 };
                let angle = position[axis] as f32 / divisors[pair];
                row[pair * 2] = angle.cos();
                row[pair * 2 + 1] = angle.sin();
            }
        }
    });
    let rope = cx.rt.upload_float(&frequencies, x.rows, 128)?;
    let limits: Vec<i32> = (1..=x.rows as i32).collect();
    cx.observe("text.embedding", &x)?;
    for layer in 0..36 {
        cx.cancel()?;
        let p = format!("{prefix}layers.{layer}");
        let norm = cx.normalize(&x, Norm::Rms, &format!("{p}.input_layernorm"))?;
        let rows = x.rows;
        let q = {
            let projected = cx.linear(&norm, &format!("{p}.self_attn.q_proj"))?;
            let normed = cx.normalize(
                &projected.reshape(rows * 32, 128)?,
                Norm::Rms,
                &format!("{p}.self_attn.q_norm"),
            )?;
            cx.rt
                .rope(&normed.reshape(rows, 4096)?, 32, 128, &rope, 2)?
        };
        let k = {
            let projected = cx.linear(&norm, &format!("{p}.self_attn.k_proj"))?;
            let normed = cx.normalize(
                &projected.reshape(rows * 8, 128)?,
                Norm::Rms,
                &format!("{p}.self_attn.k_norm"),
            )?;
            cx.rt.rope(&normed.reshape(rows, 1024)?, 8, 128, &rope, 2)?
        };
        let v = cx.linear(&norm, &format!("{p}.self_attn.v_proj"))?;
        {
            let attention = cx.rt.attention(&q, &k, &v, 32, 8, 128, &limits)?;
            let projected = cx.linear(&attention, &format!("{p}.self_attn.o_proj"))?;
            x = cx.rt.add(&x, &projected)?;
        }
        let norm2 = cx.normalize(&x, Norm::Rms, &format!("{p}.post_attention_layernorm"))?;
        {
            // GCC evaluates the C++ call's arguments right to left.
            let up = cx.linear(&norm2, &format!("{p}.mlp.up_proj"))?;
            let gate = cx.linear(&norm2, &format!("{p}.mlp.gate_proj"))?;
            let hidden = cx.rt.swiglu(&gate, &up)?;
            let mlp = cx.linear(&hidden, &format!("{p}.mlp.down_proj"))?;
            x = cx.rt.add(&x, &mlp)?;
        }
        // The DeepStack taps (three, with reference images) join the
        // first layers.
        if let Some(tap) = deep.get(layer) {
            x = cx.rt.replace_rows(&x, tap, &image_positions, true)?;
        }
        cx.observe(&format!("text.block.{layer}"), &x)?;
    }
    // The diffusion checkpoint expects pre-final-norm hidden states.
    let hidden = x.slice(drop, x.rows - drop)?;
    cx.observe("text.output", &hidden)?;
    Ok(EncodedPrompt {
        hidden,
        ids,
        images,
        drop,
    })
}

// ---- VAE ----

/// `VaeResidual`.
fn vae_residual(
    cx: &mut Context<'_, '_>,
    x: &Matrix,
    height: usize,
    width: usize,
    name: &str,
) -> Result<Matrix> {
    let shortcut = if cx.weights.contains(&format!("{name}.conv_shortcut.weight")) {
        cx.conv(x, height, width, &format!("{name}.conv_shortcut"), 1, 0, 1)?
    } else {
        x.clone()
    };
    let mut h = {
        let norm = cx.normalize_silu(x, &format!("{name}.norm1"))?;
        cx.conv(&norm, height, width, &format!("{name}.conv1"), 1, 1, 1)?
    };
    {
        let norm = cx.normalize_silu(&h, &format!("{name}.norm2"))?;
        h = cx.conv(&norm, height, width, &format!("{name}.conv2"), 1, 1, 1)?;
    }
    cx.rt.add(&h, &shortcut)
}

/// `VaeMid`: residual, single-head spatial attention, residual.
fn vae_mid(
    cx: &mut Context<'_, '_>,
    x: &Matrix,
    height: usize,
    width: usize,
    p: &str,
) -> Result<Matrix> {
    let mut h = vae_residual(cx, x, height, width, &format!("{p}.resnets.0"))?;
    let channels = h.cols;
    let qkv = {
        let norm = cx.rt.normalize(
            cx.weights,
            &h,
            Norm::Vae,
            Some(&format!("{p}.attentions.0.norm")),
            1e-6,
            false,
        )?;
        cx.conv(
            &norm,
            height,
            width,
            &format!("{p}.attentions.0.to_qkv"),
            1,
            0,
            1,
        )?
    };
    let attention = {
        // GCC evaluates the C++ call's arguments right to left.
        let v = cx.rt.columns(&qkv, channels * 2, channels)?;
        let k = cx.rt.columns(&qkv, channels, channels)?;
        let q = cx.rt.columns(&qkv, 0, channels)?;
        cx.rt.attention(&q, &k, &v, 1, 1, channels, &[])?
    };
    {
        let projected = cx.conv(
            &attention,
            height,
            width,
            &format!("{p}.attentions.0.proj"),
            1,
            0,
            1,
        )?;
        h = cx.rt.add(&h, &projected)?;
    }
    vae_residual(cx, &h, height, width, &format!("{p}.resnets.1"))
}

/// `VaeConstants`.
fn vae_constants(weights: &Weights, key: &str) -> Vec<f32> {
    weights
        .vae_config()
        .find(key)
        .map(|values| {
            values
                .items()
                .iter()
                .map(|v| v.as_double(0.0) as f32)
                .collect()
        })
        .unwrap_or_default()
}

/// `EncodeVae`: a reference image's normalised posterior mean.
fn encode_vae(cx: &mut Context<'_, '_>, image: &Image) -> Result<Matrix> {
    let input: Vec<f32> = image
        .rgba
        .iter()
        .map(|&v| f32::from(v) / 255.0 * 2.0 - 1.0)
        .collect();
    let (mut height, mut width) = (image.height as usize, image.width as usize);
    let mut x = {
        let uploaded = cx.rt.upload(&input, height * width, 4)?;
        cx.conv(&uploaded, height, width, "vae.encoder.conv_in", 1, 1, 1)?
    };
    cx.observe("vae.encoder.input", &x)?;
    const CHANNELS: [usize; 5] = [96, 192, 384, 768, 768];
    for (block, &channels) in CHANNELS.iter().enumerate() {
        cx.cancel()?;
        let p = format!("vae.encoder.down_blocks.{block}");
        let temporal = if block > 0 && block < 4 { 2 } else { 1 };
        let spatial = if block < 4 { 2 } else { 1 };
        let skip = cx
            .rt
            .down_shortcut(&x, height, width, channels, temporal, spatial)?;
        for layer in 0..2 {
            x = vae_residual(cx, &x, height, width, &format!("{p}.resnets.{layer}"))?;
        }
        if block < 4 {
            // ZeroPad2d((0, 1, 0, 1)) and a valid stride-2 convolution: the
            // output is exactly half the size.
            x = cx.conv(
                &x,
                height,
                width,
                &format!("{p}.downsampler.resample.1"),
                2,
                -1,
                1,
            )?;
            height /= 2;
            width /= 2;
        }
        x = cx.rt.add(&x, &skip)?;
        cx.observe(&format!("vae.encoder.block.{block}"), &x)?;
    }
    x = vae_mid(cx, &x, height, width, "vae.encoder.mid_block")?;
    {
        let norm = cx.normalize_silu(&x, "vae.encoder.norm_out")?;
        x = cx.conv(&norm, height, width, "vae.encoder.conv_out", 1, 1, 1)?;
    }
    x = cx.conv(&x, height, width, "vae.quant_conv", 1, 0, 1)?;
    x = cx.rt.columns(&x, 0, 64)?;
    let mean = vae_constants(cx.weights, "latents_mean");
    let deviation = vae_constants(cx.weights, "latents_std");
    x = cx.rt.latent_scale(&x, &mean, &deviation, true)?;
    cx.observe("vae.encoded", &x)?;
    Ok(x)
}

/// `DecodeVae`: latents to RGBA8, rounded as Diffusers does in BF16.
fn decode_vae(
    cx: &mut Context<'_, '_>,
    latent: &Matrix,
    mut height: usize,
    mut width: usize,
) -> Result<Image> {
    let mean = vae_constants(cx.weights, "latents_mean");
    let deviation = vae_constants(cx.weights, "latents_std");
    let mut x = cx.rt.latent_scale(latent, &mean, &deviation, false)?;
    x = cx.conv(&x, height, width, "vae.post_quant_conv", 1, 0, 1)?;
    x = cx.conv(&x, height, width, "vae.decoder.conv_in", 1, 1, 1)?;
    x = vae_mid(cx, &x, height, width, "vae.decoder.mid_block")?;
    cx.observe("vae.decoder.mid", &x)?;
    const CHANNELS: [usize; 5] = [1152, 1152, 576, 288, 144];
    for (block, &channels) in CHANNELS.iter().enumerate() {
        cx.cancel()?;
        let p = format!("vae.decoder.up_blocks.{block}");
        let skip = if block < 4 {
            Some(
                cx.rt
                    .up_shortcut(&x, height, width, channels, if block < 3 { 2 } else { 1 })?,
            )
        } else {
            None
        };
        for layer in 0..3 {
            x = vae_residual(cx, &x, height, width, &format!("{p}.resnets.{layer}"))?;
        }
        if let Some(skip) = skip {
            // The first and only image frame skips all temporal convolutions.
            x = cx.conv(
                &x,
                height,
                width,
                &format!("{p}.upsampler.resample.1"),
                1,
                1,
                2,
            )?;
            height *= 2;
            width *= 2;
            x = cx.rt.add(&x, &skip)?;
        }
        cx.observe(&format!("vae.decoder.block.{block}"), &x)?;
    }
    {
        let norm = cx.normalize_silu(&x, "vae.decoder.norm_out")?;
        x = cx.conv(&norm, height, width, "vae.decoder.conv_out", 1, 1, 1)?;
    }
    cx.observe("vae.decoded", &x)?;
    if x.cols != 4 {
        return Err(Error::runtime("expected RGBA VAE output"));
    }
    let pixels = cx.rt.download(&x)?;
    if !pixels.iter().all(|pixel| pixel.is_finite()) {
        return Err(Error::runtime("nonfinite image output"));
    }
    // Diffusers divides and adds in the VAE dtype before FP32. A branch-free
    // loop the compiler vectorises: `round_ties_even` would call libm per
    // pixel on baseline x86-64.
    let rgba = pixels
        .iter()
        .map(|&pixel| {
            let v = bf16(bf16(pixel.clamp(-1.0, 1.0) / 2.0) + 0.5);
            round_byte(v * 255.0)
        })
        .collect();
    Ok(Image {
        width: width as i32,
        height: height as i32,
        rgba,
    })
}

// ---- Diffusion ----

struct DiffusionContext {
    prefix: Matrix,
    rope: Matrix,
    limits: Vec<i32>,
    keys: Vec<Option<Matrix>>,
    values: Vec<Option<Matrix>>,
}

/// `PrepareDiffusion`: the condition rows (projected text and reference
/// latents), the 3-axis rotary table of every row and the block-causal key
/// limits.
fn prepare_diffusion(
    cx: &mut Context<'_, '_>,
    prompt: &EncodedPrompt,
    height: usize,
    width: usize,
) -> Result<DiffusionContext> {
    let text = {
        let norm = cx.rt.normalize(
            cx.weights,
            &prompt.hidden,
            Norm::ZeroRms,
            Some("transformer.txt_in.text_norm"),
            1e-6,
            false,
        )?;
        let projected = cx.linear(&norm, "transformer.txt_in.in_layer")?;
        let activated = cx.rt.activate(&projected, Activation::GeluTanh)?;
        cx.linear(&activated, "transformer.txt_in.out_layer")?
    };
    let mut parts = Vec::new();
    let mut positions: Vec<[i32; 3]> = Vec::new();
    let mut limits = Vec::new();
    let mut position = 0i32;
    let mut cursor = 0;
    let mut image_index = 0;
    let length = text.rows;
    while cursor < length {
        if prompt.ids[prompt.drop + cursor] == IMAGE_PAD {
            let image = prompt
                .images
                .get(image_index)
                .ok_or_else(|| Error::logic("image prompt layout mismatch"))?;
            image_index += 1;
            let slots = (image.height / 32) as usize * (image.width / 32) as usize;
            let (h, w) = (image.height / 16, image.width / 16);
            let projected = {
                let latent = encode_vae(cx, image)?;
                cx.linear(&latent, "transformer.img_in")?
            };
            parts.push(projected);
            let end = (positions.len() + (h * w) as usize) as i32;
            for y in 0..h {
                for x in 0..w {
                    positions.push([position, y - (h - h / 2), x - (w - w / 2)]);
                    limits.push(end);
                }
            }
            position += h.max(w);
            cursor += slots;
        } else {
            let start = cursor;
            while cursor < length && prompt.ids[prompt.drop + cursor] != IMAGE_PAD {
                positions.push([position; 3]);
                limits.push(positions.len() as i32);
                position += 1;
                cursor += 1;
            }
            parts.push(text.slice(start, cursor - start)?);
        }
    }
    let prefix = cx.rt.concat(&parts.iter().collect::<Vec<_>>())?;
    let (h, w) = (height as i32, width as i32);
    let end = (positions.len() + height * width) as i32;
    for y in 0..h {
        for x in 0..w {
            positions.push([position, y - (h - h / 2), x - (w - w / 2)]);
            limits.push(end);
        }
    }
    let axis = |pair: usize| {
        if pair < 8 {
            0
        } else if pair < 36 {
            1
        } else {
            2
        }
    };
    // `std::pow` depends on the pair only.
    let divisors: [f32; 64] = std::array::from_fn(|pair| {
        let first = [0, 8, 36][axis(pair)];
        let count = if axis(pair) == 0 { 8 } else { 28 };
        10000.0f32.powf((pair - first) as f32 / count as f32)
    });
    let mut rope = vec![0.0f32; positions.len() * 128];
    fill_rows(&mut rope, 128, |first, table| {
        for (row, position) in table
            .as_chunks_mut::<128>()
            .0
            .iter_mut()
            .zip(&positions[first..])
        {
            for pair in 0..64 {
                let angle = position[axis(pair)] as f32 / divisors[pair];
                row[pair * 2] = angle.cos();
                row[pair * 2 + 1] = angle.sin();
            }
        }
    });
    let rope = cx.rt.upload_float(&rope, positions.len(), 128)?;
    Ok(DiffusionContext {
        prefix,
        rope,
        limits,
        keys: vec![None; 32],
        values: vec![None; 32],
    })
}

/// `Denoise`: one DiT velocity. The first step runs the condition rows too
/// and keeps their K/V; later steps attend to the cached rows.
fn denoise(
    cx: &mut Context<'_, '_>,
    context: &mut DiffusionContext,
    latent: &Matrix,
    sigma: f32,
    step: usize,
) -> Result<Matrix> {
    let prefix = if step == 0 { context.prefix.rows } else { 0 };
    let mut x = cx.linear(latent, "transformer.img_in")?;
    if prefix > 0 {
        x = cx.rt.concat(&[&context.prefix, &x])?;
    }
    let rope = if prefix > 0 {
        context.rope.clone()
    } else {
        context.rope.slice(context.prefix.rows, latent.rows)?
    };
    let timestep = bf16(bf16(sigma * 1000.0) / 1000.0);
    // `std::log(10000.0F)` is folded at compile time in C++: the correctly
    // rounded value.
    let log_period = 10000.0f64.ln() as f32;
    let mut time = vec![0.0f32; 512];
    for row in 0..2 {
        for i in 0..128 {
            let angle = if row == 0 { timestep * 1000.0 } else { 0.0 }
                * (-log_period * i as f32 / 128.0).exp();
            time[row * 256 + i] = angle.cos();
            time[row * 256 + 128 + i] = angle.sin();
        }
    }
    let temb = {
        let uploaded = cx.rt.upload(&time, 2, 256)?;
        let projected = cx.linear(
            &uploaded,
            "transformer.time_text_embed.timestep_embedder.linear_1",
        )?;
        let activated = cx.rt.activate(&projected, Activation::Silu)?;
        cx.linear(
            &activated,
            "transformer.time_text_embed.timestep_embedder.linear_2",
        )?
    };
    let activated = cx.rt.activate(&temb, Activation::Silu)?;
    let modulation = cx.linear(&activated, "transformer.modulation.1")?;
    cx.observe(&format!("dit.{step}.input"), &x)?;
    cx.observe(&format!("dit.{step}.modulation"), &modulation)?;
    let factors = cx.rt.modulation_factors(&modulation, true)?;
    for layer in 0..32 {
        cx.cancel()?;
        let p = format!("transformer.transformer_blocks.{layer}");
        let norm = cx.rt.normalize_modulate(&x, &factors, 0, prefix)?;
        let q = {
            let projected = cx.linear(&norm, &format!("{p}.attn.to_q"))?;
            cx.rt.normalize_rope(
                cx.weights,
                &projected,
                &format!("{p}.attn.norm_q"),
                32,
                &rope,
            )?
        };
        let mut k = {
            let projected = cx.linear(&norm, &format!("{p}.attn.to_k"))?;
            cx.rt.normalize_rope(
                cx.weights,
                &projected,
                &format!("{p}.attn.norm_k"),
                32,
                &rope,
            )?
        };
        let mut v = cx.linear(&norm, &format!("{p}.attn.to_v"))?;
        if prefix > 0 {
            // Copy only the prefix rows: a view would keep the full K/V of
            // every layer alive.
            context.keys[layer] = Some(cx.rt.concat(&[&k.slice(0, prefix)?])?);
            context.values[layer] = Some(cx.rt.concat(&[&v.slice(0, prefix)?])?);
        } else {
            let missing = || Error::logic("missing condition K/V");
            k = cx
                .rt
                .concat(&[context.keys[layer].as_ref().ok_or_else(missing)?, &k])?;
            v = cx
                .rt
                .concat(&[context.values[layer].as_ref().ok_or_else(missing)?, &v])?;
        }
        let limits: &[i32] = if prefix > 0 { &context.limits } else { &[] };
        let attention = {
            let attended = cx.rt.attention(&q, &k, &v, 32, 32, 128, limits)?;
            cx.linear(&attended, &format!("{p}.attn.to_out.0"))?
        };
        x = cx.rt.modulate(&attention, &factors, 1, prefix, Some(&x))?;
        let norm2 = cx.rt.normalize_modulate(&x, &factors, 2, prefix)?;
        let mlp = {
            let gated = cx.rt.gated_linear(
                cx.weights,
                &norm2,
                &format!("{p}.img_mlp.gate_layer"),
                &format!("{p}.img_mlp.proj"),
            )?;
            cx.linear(&gated, &format!("{p}.img_mlp.out"))?
        };
        x = cx.rt.modulate(&mlp, &factors, 3, prefix, Some(&x))?;
        cx.observe(&format!("dit.{step}.block.{layer}"), &x)?;
    }
    let scale = cx.linear(&activated, "transformer.norm_out.linear")?;
    {
        // The C++ call evaluates the factors before the slice.
        let scale_factors = cx.rt.modulation_factors(&scale, false)?;
        let tail = x.slice(prefix, latent.rows)?;
        x = cx.rt.normalize_modulate(&tail, &scale_factors, 0, 0)?;
    }
    x = cx.linear(&x, "transformer.proj_out")?;
    cx.observe(&format!("dit.{step}.output"), &x)?;
    Ok(x)
}

/// `Model`: the checkpoint, its tokenizer and the device runtime; one
/// generation at a time.
pub struct Model {
    weights: Weights,
    tokenizer: Tokenizer,
    runtime: Mutex<Runtime>,
}

impl Model {
    pub fn load(root: &Path) -> Result<Self> {
        let weights = Weights::load(root)?;
        let tokenizer = Tokenizer::load(root)?;
        let runtime = Mutex::new(Runtime::new()?);
        Ok(Self {
            weights,
            tokenizer,
            runtime,
        })
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    /// `Model::Generate`. Requests have independent noise and caches.
    pub fn generate(
        &self,
        request: &Request,
        cancelled: Option<&dyn Fn() -> bool>,
        observer: Option<&mut (dyn Observer + '_)>,
    ) -> Result<Generation> {
        validate_request(request)?;
        let mut rt = self.runtime.lock().unwrap_or_else(PoisonError::into_inner);
        let mut cx = Context {
            rt: &mut rt,
            weights: &self.weights,
            cancelled,
            observer,
        };
        cx.cancel()?;
        let result = generate(&mut cx, &self.tokenizer, request);
        // Every launch of the request must finish before its buffers return
        // to the pool and the next request starts.
        let finished = cx.rt.finish_request();
        let generation = result?;
        finished?;
        Ok(generation)
    }
}

fn generate(
    cx: &mut Context<'_, '_>,
    tokenizer: &Tokenizer,
    request: &Request,
) -> Result<Generation> {
    let start = Instant::now();
    let mut metrics = Metrics::default();
    let prompt = encode_prompt(cx, tokenizer, request)?;
    let height = (request.height / 16) as usize;
    let width = (request.width / 16) as usize;
    let mut context = prepare_diffusion(cx, &prompt, height, width)?;
    cx.rt.synchronize()?;
    metrics.prompt_ms = milliseconds(start);
    // Independent full-precision normal draws: seed replay never depends on
    // admission order, wall time or another request. An edit's references
    // join the seed, so a generated source is never edited with its own
    // noise.
    let mut rng = Mt19937_64::new(crate::image::initial_noise_seed(
        request.seed,
        &prompt.images,
    )?);
    let mut normal = Normal::default();
    let noise: Vec<f32> = (0..height * width * 64)
        .map(|_| normal.sample(&mut rng))
        .collect();
    let mut latent = cx.rt.upload(&noise, height * width, 64)?;
    cx.observe("noise", &latent)?;
    let denoise_start = Instant::now();
    let sigmas = flow_sigmas(request.steps, (height * width) as i32)?;
    if let Some(observer) = cx.observer.as_deref_mut()
        && observer.wants("sigmas")
    {
        observer.write("sigmas", &sigmas, 1, sigmas.len());
    }
    for step in 0..request.steps as usize {
        let velocity = denoise(cx, &mut context, &latent, sigmas[step], step)?;
        latent = cx
            .rt
            .euler(&latent, &velocity, sigmas[step + 1] - sigmas[step])?;
        cx.rt.synchronize()?;
        cx.observe(&format!("latent.{step}"), &latent)?;
    }
    cx.rt.synchronize()?;
    metrics.denoise_ms = milliseconds(denoise_start);
    let vae_start = Instant::now();
    let image = decode_vae(cx, &latent, height, width)?;
    metrics.vae_ms = milliseconds(vae_start);
    metrics.total_ms = milliseconds(start);
    metrics.resident_weight_bytes = cx.rt.weight_bytes();
    Ok(Generation { image, metrics })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_rounding_matches_nearbyint() {
        // Every float in [0, 255] at quarter steps and around each half.
        let mut x = 0.0f32;
        while x <= 255.0 {
            for value in [
                x,
                f32::from_bits(x.to_bits().saturating_sub(1)),
                f32::from_bits(x.to_bits() + 1),
            ] {
                if (0.0..=255.0).contains(&value) {
                    assert_eq!(round_byte(value), value.round_ties_even() as u8, "{value}");
                }
            }
            x += 0.25;
        }
    }
}
