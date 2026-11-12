//! NVIDIA's image processor for the vision projector (`vision_image.cpp`):
//! the patch grid of an image, PyTorch's antialiased bicubic resize in
//! float32 and the normalized patches.
//!
//! PyTorch's CPU kernels run as their AVX2 or AVX-512 builds, in which the
//! compiler fuses multiply-adds and vectorizes loops; the explicit fused
//! multiply-adds and the split sums below reproduce those roundings bit for
//! bit.

use gufo_core::image::Image;

use crate::config::VisionConfig;

/// The grid of whole patches NVIDIA's image processor scales an image to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ImageGrid {
    pub rows: u32,
    pub cols: u32,
}

impl ImageGrid {
    pub fn patches(&self) -> u32 {
        self.rows * self.cols
    }
}

/// PyTorch's filter for antialiased bicubic resampling (Keys, a = -0.5), in
/// float like `HelperInterpCubic::aa_filter`.
fn aa_filter(x: f32) -> f32 {
    const A: f32 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        return ((A + 2.0).mul_add(x, -(A + 3.0)) * x).mul_add(x, 1.0);
    }
    if x < 2.0 {
        return A
            .mul_add(x, -(5.0 * A))
            .mul_add(x, 8.0 * A)
            .mul_add(x, -(4.0 * A));
    }
    0.0
}

/// Source span and normalized weights of every output position along one
/// axis, as PyTorch's `_compute_index_ranges_weights` computes them with
/// antialiasing. The expressions keep its mix of float and double
/// arithmetic, which decides the last bits of the weights.
struct Taps {
    first: Vec<i64>,
    count: Vec<i64>,
    /// `[output][stride]`
    weights: Vec<f32>,
    stride: i64,
}

fn compute_taps(input_size: i64, output_size: i64) -> Taps {
    const INTERP_SIZE: f64 = 4.0;
    const ALIGN_CORNERS_DELTA: f64 = 0.0;
    let scale = input_size as f32 / output_size as f32;
    let support: f32 = if f64::from(scale) >= 1.0 {
        ((INTERP_SIZE * 0.5) * f64::from(scale)) as f32
    } else {
        (INTERP_SIZE * 0.5) as f32
    };
    let invscale: f32 = if f64::from(scale) >= 1.0 {
        (1.0 / f64::from(scale)) as f32
    } else {
        1.0
    };
    let stride = (support.ceil() * 2.0 + 1.0) as i64;
    let outputs = output_size as usize;
    let mut taps = Taps {
        first: vec![0; outputs],
        count: vec![0; outputs],
        weights: vec![0.0; outputs * stride as usize],
        stride,
    };
    for i in 0..output_size {
        let center = (f64::from(scale) * (i as f64 + 0.5 - ALIGN_CORNERS_DELTA)) as f32;
        let xmin = ((f64::from(center - support) + 0.5 + ALIGN_CORNERS_DELTA) as i64).max(0);
        let xsize = ((f64::from(center + support) + 0.5 + ALIGN_CORNERS_DELTA) as i64)
            .min(input_size)
            - xmin;
        let xsize = xsize.clamp(0, stride);
        let w = &mut taps.weights[(i * stride) as usize..((i + 1) * stride) as usize];
        let mut total = 0.0f32;
        for j in 0..xsize {
            let offset = (j + xmin) as f32 - center;
            let value = aa_filter(
                ((f64::from(offset) + 0.5 - ALIGN_CORNERS_DELTA) * f64::from(invscale)) as f32,
            );
            w[j as usize] = value;
            total += value;
        }
        if total != 0.0 {
            for value in &mut w[..xsize as usize] {
                *value /= total;
            }
        }
        taps.first[i as usize] = xmin;
        taps.count[i as usize] = xsize;
    }
    taps
}

/// One output value as PyTorch's `interpolate_aa_single_dim` sums it: the
/// first product, then the remaining taps in order. The compiled loop
/// multiplies four taps at a time and adds the products one by one; the
/// taps left over are fused multiply-adds.
fn interpolate(source: &[f32], step: usize, weights: &[f32], count: i64) -> f32 {
    let mut output = source[0] * weights[0];
    let vectorized = 1 + (count - 1) / 4 * 4;
    let mut j = 1;
    while j < vectorized {
        let product = source[j as usize * step] * weights[j as usize];
        output += product;
        j += 1;
    }
    while j < count {
        output = source[j as usize * step].mul_add(weights[j as usize], output);
        j += 1;
    }
    output
}

/// The processor's `_compute_target_patches`: the nearest grid of whole
/// patches, scaled into [min_patches, budget] with sides divisible by the
/// merge (`PlanGrid`). The budget is max_patches unless given; a budget of
/// min_patches only shrinks images. `None` for an empty image or one so
/// elongated that a side would vanish.
pub fn plan_grid(
    config: &VisionConfig,
    width: u32,
    height: u32,
    max_patches: u32,
) -> Option<ImageGrid> {
    if width == 0 || height == 0 || config.patch_size == 0 || config.merge == 0 {
        return None;
    }
    let patch = f64::from(config.patch_size);
    let budget = f64::from(if max_patches == 0 {
        config.max_patches
    } else {
        max_patches.min(config.max_patches)
    });
    let minimum = f64::from(config.min_patches);
    let merge = f64::from(config.merge);
    // Python's round() of `x + 0.5` rounds halves to even.
    let mut h = (f64::from(height) / patch + 0.5).round_ties_even();
    let mut w = (f64::from(width) / patch + 0.5).round_ties_even();
    let factor = (budget / (h * w)).sqrt().min(1.0);
    h = (factor * h).floor();
    w = (factor * w).floor();
    if budget > minimum && h * w < minimum {
        if h * w == 0.0 {
            return None;
        }
        let up = (minimum / (h * w)).sqrt();
        h = (up * h).ceil();
        w = (up * w).ceil();
    }
    let rest = h % merge;
    if rest != 0.0 {
        if (h + merge - rest) * w <= budget {
            h += merge - rest;
        } else {
            h = merge.max(h - rest);
        }
    }
    let rest = w % merge;
    if rest != 0.0 {
        if h * (w + merge - rest) <= budget {
            w += merge - rest;
        } else {
            w = merge.max(w - rest);
        }
    }
    if h < 1.0 || w < 1.0 || h * w > budget {
        return None;
    }
    Some(ImageGrid {
        rows: h as u32,
        cols: w as u32,
    })
}

/// Trunk tokens of one image: one per merge x merge patches (`ImageTokens`).
pub fn image_tokens(config: &VisionConfig, grid: &ImageGrid) -> u32 {
    grid.patches() / (config.merge * config.merge)
}

/// PyTorch's antialiased bicubic interpolation in float32
/// (`interpolate(mode="bicubic", align_corners=False, antialias=True)`) of
/// RGB8 `image` to `width` x `height`: three planes, row-major
/// (`ResizePlanes`).
pub fn resize_planes(image: &Image, width: u32, height: u32) -> Result<Vec<f32>, String> {
    let in_w = image.width as usize;
    let in_h = image.height as usize;
    if in_w == 0 || in_h == 0 || width == 0 || height == 0 || image.pixels.len() != in_w * in_h * 3
    {
        return Err("invalid image for resizing".into());
    }
    let (width, height) = (width as usize, height as usize);
    // The uint8 pixels as float planes, the tensor the processor resizes.
    let mut planes = vec![0.0f32; 3 * in_w * in_h];
    for c in 0..3 {
        let plane = &mut planes[c * in_w * in_h..(c + 1) * in_w * in_h];
        for (i, value) in plane.iter_mut().enumerate() {
            *value = f32::from(image.pixels[i * 3 + c]);
        }
    }
    // Width first, then height, each pass skipped when its size is kept.
    let mut cur_w = in_w;
    if width != in_w {
        let taps = compute_taps(in_w as i64, width as i64);
        let stride = taps.stride as usize;
        let mut out = vec![0.0f32; 3 * in_h * width];
        for r in 0..3 * in_h {
            let src = &planes[r * in_w..(r + 1) * in_w];
            let dst = &mut out[r * width..(r + 1) * width];
            for (x, value) in dst.iter_mut().enumerate() {
                *value = interpolate(
                    &src[taps.first[x] as usize..],
                    1,
                    &taps.weights[x * stride..],
                    taps.count[x],
                );
            }
        }
        planes = out;
        cur_w = width;
    }
    if height != in_h {
        let taps = compute_taps(in_h as i64, height as i64);
        let stride = taps.stride as usize;
        let mut out = vec![0.0f32; 3 * height * cur_w];
        for c in 0..3 {
            let src = &planes[c * in_h * cur_w..(c + 1) * in_h * cur_w];
            let dst = &mut out[c * height * cur_w..(c + 1) * height * cur_w];
            for y in 0..height {
                let weights = &taps.weights[y * stride..];
                let row = &src[taps.first[y] as usize * cur_w..];
                for x in 0..cur_w {
                    dst[y * cur_w + x] = interpolate(&row[x..], cur_w, weights, taps.count[y]);
                }
            }
        }
        planes = out;
    }
    Ok(planes)
}

/// The processor's pixels for `grid`: resized as above, normalized as
/// (x / 255 - mean) / deviation and cut into patches, rows x cols of
/// PatchValues floats, row-major over the grid, each patch channel-major and
/// then by pixel row (`PreprocessImage`).
pub fn preprocess_image(
    config: &VisionConfig,
    image: &Image,
    grid: &ImageGrid,
) -> Result<Vec<f32>, String> {
    let patch = config.patch_size as usize;
    let width = grid.cols as usize * patch;
    let height = grid.rows as usize * patch;
    let planes = resize_planes(image, width as u32, height as u32)?;
    let values = config.patch_values() as usize;
    let mut patches = vec![0.0f32; grid.patches() as usize * values];
    for c in 0..3 {
        let mean = config.mean[c];
        let deviation = config.deviation[c];
        let plane = &planes[c * width * height..(c + 1) * width * height];
        for y in 0..height {
            let row = y / patch;
            let py = y % patch;
            for x in 0..width {
                let p = row * grid.cols as usize + x / patch;
                patches[p * values + (c * patch + py) * patch + x % patch] =
                    (plane[y * width + x] / 255.0 - mean) / deviation;
            }
        }
    }
    Ok(patches)
}
