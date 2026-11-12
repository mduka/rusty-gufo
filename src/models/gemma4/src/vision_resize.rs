//! Antialiased bicubic RGB8 resize (`vision/resize.cpp`, copied there from
//! `src/core/image.cpp` of franzmoca/gufo's `feat/gemma` branch), bit-exact
//! with PyTorch/torchvision uint8 `resize(..., BICUBIC, antialias=True)`:
//! separable, horizontal first, each axis' coefficients quantized to `i16`
//! with one shared precision.

use gufo_core::image::Image;

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        (((-0.5 * x + 2.5) * x - 4.0) * x) + 2.0
    } else {
        0.0
    }
}

struct Filter {
    begin: usize,
    weights: Vec<i16>,
}

struct FilterBank {
    rows: Vec<Filter>,
    precision: u32,
}

fn filters(input: u32, output: u32) -> FilterBank {
    let scale = f64::from(input) / f64::from(output);
    let antialias = scale.max(1.0);
    let support = antialias * 2.0;
    let mut begins = Vec::with_capacity(output as usize);
    let mut coefficients: Vec<Vec<f64>> = Vec::with_capacity(output as usize);
    let mut maximum = 0f64;
    for i in 0..output {
        let center = (f64::from(i) + 0.5) * scale;
        let begin = ((center - support + 0.5) as i32).max(0);
        let end = (input as i32).min((center + support + 0.5) as i32);
        let mut weights: Vec<f64> = (begin..end)
            .map(|j| cubic((f64::from(j) + 0.5 - center) / antialias))
            .collect();
        let sum: f64 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= sum;
            maximum = maximum.max(*weight);
        }
        begins.push(begin as usize);
        coefficients.push(weights);
    }
    // PyTorch's uint8 antialiased resize quantizes each axis' coefficients
    // to int16, choosing one shared precision for that axis.
    let mut precision = 0u32;
    while precision < 22 {
        if (0.5 + maximum * f64::from(1u32 << (precision + 1))) as i32 >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    let rows = begins
        .into_iter()
        .zip(coefficients)
        .map(|(begin, weights)| Filter {
            begin,
            weights: weights
                .into_iter()
                .map(|weight| {
                    let scaled = weight * f64::from(1u32 << precision);
                    (if scaled < 0.0 {
                        scaled - 0.5
                    } else {
                        scaled + 0.5
                    }) as i16
                })
                .collect(),
        })
        .collect();
    FilterBank { rows, precision }
}

fn pixel(value: i64, precision: u32) -> u8 {
    (value >> precision).clamp(0, 255) as u8
}

/// Antialiased bicubic RGB8 resize to `width` x `height` (`ResizeBicubic`).
pub fn resize_bicubic(image: &Image, width: u32, height: u32) -> Result<Image, String> {
    if image.width == 0
        || image.height == 0
        || width == 0
        || height == 0
        || image.pixels.len() != image.width as usize * image.height as usize * 3
    {
        return Err("invalid image resize".into());
    }
    if width == image.width && height == image.height {
        return Ok(image.clone());
    }
    let horizontal = filters(image.width, width);
    let vertical = filters(image.height, height);
    let (in_width, width_us, height_us) = (image.width as usize, width as usize, height as usize);
    let mut intermediate = vec![0u8; width_us * image.height as usize * 3];
    let round_h = (1i64 << horizontal.precision) >> 1;
    for y in 0..image.height as usize {
        for x in 0..width_us {
            let filter = &horizontal.rows[x];
            for c in 0..3 {
                let mut value = round_h;
                for (j, &weight) in filter.weights.iter().enumerate() {
                    value += i64::from(image.pixels[(y * in_width + filter.begin + j) * 3 + c])
                        * i64::from(weight);
                }
                intermediate[(y * width_us + x) * 3 + c] = pixel(value, horizontal.precision);
            }
        }
    }
    let mut pixels = vec![0u8; width_us * height_us * 3];
    let round_v = (1i64 << vertical.precision) >> 1;
    for y in 0..height_us {
        let filter = &vertical.rows[y];
        for x in 0..width_us {
            for c in 0..3 {
                let mut value = round_v;
                for (j, &weight) in filter.weights.iter().enumerate() {
                    value += i64::from(intermediate[((filter.begin + j) * width_us + x) * 3 + c])
                        * i64::from(weight);
                }
                pixels[(y * width_us + x) * 3 + c] = pixel(value, vertical.precision);
            }
        }
    }
    Ok(Image {
        width,
        height,
        pixels,
    })
}
