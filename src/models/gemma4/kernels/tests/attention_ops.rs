//! The extension's `attention_ops_test.cpp`: Gemma 4's attention kernels
//! against an FP64 formula over the same binary16 caches. Window and ring
//! addressing, the draft key limit, split-K and single-pass modes, batch
//! invariance of the split mode, image rows that see their whole image, and
//! the derived keys of the global layers. (The C++ test's timings are not
//! repeated here.)
//!
//! cargo test --profile gpu-test -p gufo-gemma4-kernels --features gpu --test attention_ops

#![cfg(feature = "gpu")]
#![allow(
    clippy::unwrap_used,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

mod common;

use common::*;
use gufo_gemma4_kernels as k;
use gufo_gemma4_kernels::sizes;
use gufo_hip::F16;

const NO_LIMIT: u32 = u32::MAX;
const PAIRS: usize = 64;

#[derive(Clone, Copy)]
struct Case {
    head_dim: u32,
    heads: u32,
    kv_heads: u32,
    rows: u32,
    first_position: u32,
    window: u32,
    ring: u32,
    shared_position: bool,
    key_limit: u32,
    name: &'static str,
    /// Rows checked against FP64.
    row_stride: u32,
    /// Keys rebuilt from V (the K cache holds the rotated dims only).
    derived: bool,
    /// Rows [image_row, image_row + image_count) form one image.
    image_row: u32,
    image_count: u32,
}

#[allow(clippy::too_many_arguments)]
const fn case(
    head_dim: u32,
    heads: u32,
    kv_heads: u32,
    rows: u32,
    first_position: u32,
    window: u32,
    ring: u32,
    shared_position: bool,
    key_limit: u32,
    name: &'static str,
) -> Case {
    Case {
        head_dim,
        heads,
        kv_heads,
        rows,
        first_position,
        window,
        ring,
        shared_position,
        key_limit,
        name,
        row_stride: 1,
        derived: false,
        image_row: 0,
        image_count: 0,
    }
}

impl Case {
    const fn stride(self, row_stride: u32) -> Self {
        Self { row_stride, ..self }
    }
    const fn derived(self) -> Self {
        Self {
            derived: true,
            ..self
        }
    }
    const fn image(self, image_row: u32, image_count: u32) -> Self {
        Self {
            image_row,
            image_count,
            ..self
        }
    }
    fn position(&self, row: u32) -> u32 {
        if self.shared_position {
            self.first_position
        } else {
            self.first_position + row
        }
    }
    /// A row's exclusive key end: its image's end for image rows.
    fn key_end(&self, row: u32) -> u32 {
        let position = self.position(row);
        if row >= self.image_row && row < self.image_row + self.image_count {
            (position + 1).max(self.first_position + self.image_row + self.image_count)
        } else {
            position + 1
        }
    }
    fn split(&self) -> bool {
        self.rows as usize <= sizes::SPLIT_ROWS
    }
}

struct Caches {
    q: Vec<f32>,
    k: Vec<F16>,
    v: Vec<F16>,
    /// Slots of a linear cache (the ring's for a ring cache).
    slots: u32,
}

fn run(c: &Case, caches: &Caches) -> Vec<f32> {
    let s = stream();
    let d = c.head_dim as usize;
    let n_out = (c.rows * c.heads) as usize * d;
    let dq = up(&s, &caches.q);
    let dk = up(&s, &caches.k);
    let dv = up(&s, &caches.v);
    let mut dout = filled::<f32>(&s, n_out, 0xFF);
    let ends: Vec<u32> = (0..c.rows)
        .map(|r| {
            if r >= c.image_row && r < c.image_row + c.image_count {
                c.first_position + c.image_row + c.image_count
            } else {
                0
            }
        })
        .collect();
    let dends = up(&s, &ends);
    let key_ends = (c.image_count != 0).then(|| dends.as_slice());
    let result = if c.ring != 0 {
        if c.split() {
            let partial_keys = c.ring.max(c.window + c.rows);
            let mut partials = zeros::<f32>(
                &s,
                sizes::attention_ring_partial_floats(
                    c.rows as usize,
                    c.heads as usize,
                    partial_keys as usize,
                ),
            );
            k::attention_ring(
                &s,
                dq.as_slice(),
                dk.as_slice(),
                dv.as_slice(),
                dout.as_mut_slice(),
                partials.as_mut_slice(),
                c.rows,
                c.heads,
                c.kv_heads,
                c.head_dim,
                c.first_position,
                c.shared_position,
                c.key_limit,
                c.window,
                c.ring,
                key_ends,
                0,
                0,
                None,
                0,
                partial_keys,
            )
        } else {
            k::attention_ring_prefill(
                &s,
                dq.as_slice(),
                dk.as_slice(),
                dv.as_slice(),
                Some(dout.as_mut_slice()),
                c.rows,
                c.heads,
                c.kv_heads,
                c.head_dim,
                c.first_position,
                c.shared_position,
                c.key_limit,
                c.window,
                c.ring,
                key_ends,
                None,
            )
        }
    } else if c.split() {
        let mut partials = zeros::<f32>(
            &s,
            sizes::attention_linear_partial_floats(
                c.rows as usize,
                c.heads as usize,
                d,
                caches.slots as usize,
            ),
        );
        if c.derived {
            k::attention_derived(
                &s,
                dq.as_slice(),
                dk.as_slice(),
                dv.as_slice(),
                dout.as_mut_slice(),
                partials.as_mut_slice(),
                c.rows,
                c.heads,
                c.kv_heads,
                c.head_dim,
                c.first_position,
                c.shared_position,
                c.key_limit,
                c.window,
                PAIRS as u32,
                0,
                0,
                None,
                0,
                caches.slots,
            )
        } else {
            k::attention_linear(
                &s,
                dq.as_slice(),
                dk.as_slice(),
                dv.as_slice(),
                dout.as_mut_slice(),
                partials.as_mut_slice(),
                c.rows,
                c.heads,
                c.kv_heads,
                c.head_dim,
                c.first_position,
                c.shared_position,
                c.key_limit,
                c.window,
                0,
                0,
                None,
                0,
                caches.slots,
            )
        }
    } else if c.derived {
        k::attention_derived_prefill(
            &s,
            dq.as_slice(),
            dk.as_slice(),
            dv.as_slice(),
            Some(dout.as_mut_slice()),
            c.rows,
            c.heads,
            c.kv_heads,
            c.head_dim,
            c.first_position,
            c.shared_position,
            c.key_limit,
            c.window,
            PAIRS as u32,
            None,
            caches.slots,
        )
    } else {
        k::attention_linear_prefill(
            &s,
            dq.as_slice(),
            dk.as_slice(),
            dv.as_slice(),
            Some(dout.as_mut_slice()),
            c.rows,
            c.heads,
            c.kv_heads,
            c.head_dim,
            c.first_position,
            c.shared_position,
            c.key_limit,
            c.window,
            None,
            caches.slots,
        )
    };
    ok(result, c.name);
    down(&s, &dout, n_out)
}

fn check(c: &Case, rng: &mut Rng) {
    let d = c.head_dim as usize;
    let last = c.position(c.rows - 1).max(c.first_position + c.rows - 1);
    let max_keys = (c.position(c.rows - 1) + 1).min(c.key_limit);
    // Linear caches cover the rows' own positions (the launch contract's
    // `first_position + rows <= slots`), as the trunk's max_context caches do.
    let slots = if c.ring != 0 {
        c.ring
    } else {
        max_keys.max(last + 1)
    };
    let kv = c.kv_heads as usize;
    let mut q: Vec<f32> = rng.normals((c.rows * c.heads) as usize * d, 0.35);
    let mut kc: Vec<F16> = (0..slots as usize * kv * d)
        .map(|_| half(rng.normal()))
        .collect();
    let vc: Vec<F16> = (0..kc.len()).map(|_| half(rng.normal())).collect();
    // Reference keys (FP32 copies of the binary16 cache).
    let mut kref: Vec<f32> = kc.iter().map(|h| h.to_f32()).collect();
    // Derived keys: V is the normalized K projection, K = rope(w * V) turns
    // the first 64 pairs (Gemma 4 global rope, theta 1e6) and the K cache
    // keeps only those rotated dims; every other key dim is w * V, which the
    // producer folds into the query (q * w) so the kernel reads V.
    if c.derived {
        let weight: Vec<f32> = (0..d).map(|_| rng.uniform(0.75, 1.25)).collect();
        for row in 0..q.len() / d {
            for i in 0..d {
                if i % (d / 2) >= PAIRS {
                    q[row * d + i] *= weight[i];
                }
            }
        }
        kc = vec![F16(0); slots as usize * kv * 2 * PAIRS];
        for key in 0..max_keys as usize {
            let slot = key;
            for h in 0..kv {
                let row = (slot * kv + h) * d;
                let rot = (slot * kv + h) * 2 * PAIRS;
                for i in 0..d / 2 {
                    if i >= PAIRS {
                        kref[row + i] = vc[row + i].to_f32();
                        kref[row + d / 2 + i] = vc[row + d / 2 + i].to_f32();
                        continue;
                    }
                    let x0 = f64::from(vc[row + i].to_f32()) * f64::from(weight[i]);
                    let x1 = f64::from(vc[row + d / 2 + i].to_f32()) * f64::from(weight[d / 2 + i]);
                    let theta = key as f64 * 1e6f64.powf(-2.0 * i as f64 / d as f64);
                    let (cs, sn) = (theta.cos(), theta.sin());
                    kc[rot + i] = half((x0 * cs - x1 * sn) as f32);
                    kc[rot + PAIRS + i] = half((x0 * sn + x1 * cs) as f32);
                    kref[row + i] = kc[rot + i].to_f32();
                    kref[row + d / 2 + i] = kc[rot + PAIRS + i].to_f32();
                }
            }
        }
    }
    let caches = Caches {
        q,
        k: kc,
        v: vc,
        slots,
    };
    let out = run(c, &caches);
    // Repeated launches produce the same bits.
    assert!(
        same_bits(&run(c, &caches), &out),
        "{}: output differs between launches",
        c.name
    );
    let mut worst = 0f64;
    for r in (0..c.rows).step_by(c.row_stride as usize) {
        let pos = c.position(r);
        let hi = c.key_end(r).min(c.key_limit);
        let lo = if c.window != 0 && pos + 1 > c.window {
            pos + 1 - c.window
        } else {
            0
        };
        for h in 0..c.heads as usize {
            let kvh = h / (c.heads / c.kv_heads) as usize;
            let qh = &caches.q[(r as usize * c.heads as usize + h) * d..][..d];
            let slot_of = |key: u32| (if c.ring != 0 { key % c.ring } else { key }) as usize;
            let mut scores = Vec::new();
            let mut m = f64::NEG_INFINITY;
            for key in lo..hi {
                let at = (slot_of(key) * kv + kvh) * d;
                let s: f64 = (0..d)
                    .map(|i| f64::from(qh[i]) * f64::from(kref[at + i]))
                    .sum();
                scores.push(s);
                m = m.max(s);
            }
            let mut total = 0f64;
            for s in &mut scores {
                *s = (*s - m).exp();
                total += *s;
            }
            for i in 0..d {
                let mut acc = 0f64;
                for key in lo..hi {
                    let at = (slot_of(key) * kv + kvh) * d + i;
                    acc += scores[(key - lo) as usize] * f64::from(caches.v[at].to_f32());
                }
                let want = acc / total;
                let got = f64::from(out[(r as usize * c.heads as usize + h) * d + i]);
                assert!(got.is_finite(), "{}: non-finite output", c.name);
                worst = worst.max((got - want).abs());
            }
        }
    }
    // Split mode accumulates in FP32; prefill runs binary16 WMMA operands
    // (queries and probabilities rounded to binary16).
    let limit = if c.split() { 2e-5 } else { 1e-2 };
    println!("{}: max error {worst}", c.name);
    assert!(worst < limit, "{}: max error {worst}", c.name);

    // Split mode: every row equals its single-row evaluation bit for bit.
    if c.rows > 1 && c.split() && !c.shared_position && c.image_count == 0 {
        let hd = c.heads as usize * d;
        for r in 0..c.rows {
            let single = Case {
                rows: 1,
                first_position: c.first_position + r,
                ..*c
            };
            let one_caches = Caches {
                q: caches.q[r as usize * hd..(r as usize + 1) * hd].to_vec(),
                k: caches.k.clone(),
                v: caches.v.clone(),
                slots,
            };
            let one = run(&single, &one_caches);
            assert!(
                same_bits(&one, &out[r as usize * hd..(r as usize + 1) * hd]),
                "{}: row {r} depends on its batch",
                c.name
            );
        }
    }
}

#[test]
fn attention() {
    let mut rng = Rng(20_260_926);
    let cases = [
        case(
            256,
            32,
            16,
            1,
            0,
            1024,
            3072,
            false,
            NO_LIMIT,
            "sliding first token",
        ),
        case(
            256,
            32,
            16,
            1,
            1023,
            1024,
            3072,
            false,
            NO_LIMIT,
            "sliding full window",
        ),
        case(
            256,
            32,
            16,
            1,
            5000,
            1024,
            3072,
            false,
            NO_LIMIT,
            "sliding ring wrap",
        ),
        case(
            256,
            8,
            4,
            5,
            1020,
            1024,
            1280,
            false,
            NO_LIMIT,
            "sliding verify window edge",
        ),
        case(
            256,
            8,
            4,
            8,
            3070,
            1024,
            1280,
            false,
            NO_LIMIT,
            "sliding verify wrap",
        ),
        case(
            256,
            8,
            4,
            6,
            2000,
            1024,
            1280,
            false,
            NO_LIMIT,
            "sliding verify six rows",
        ),
        case(
            256,
            8,
            4,
            40,
            1000,
            1024,
            1280,
            false,
            NO_LIMIT,
            "sliding prefill",
        ),
        case(
            256,
            8,
            4,
            70,
            3000,
            1024,
            1280,
            false,
            NO_LIMIT,
            "sliding prefill ring wrap",
        ),
        case(
            256,
            32,
            16,
            37,
            2,
            1024,
            3072,
            false,
            NO_LIMIT,
            "sliding prefill start",
        ),
        // Image rows attend forward to the end of their image.
        case(
            256,
            8,
            4,
            300,
            900,
            1024,
            1536,
            false,
            NO_LIMIT,
            "sliding prefill image across the window",
        )
        .image(20, 280),
        case(
            256,
            8,
            4,
            70,
            3000,
            1024,
            1280,
            false,
            NO_LIMIT,
            "sliding prefill image ring wrap",
        )
        .image(10, 50),
        case(
            256,
            32,
            16,
            40,
            0,
            1024,
            3072,
            false,
            NO_LIMIT,
            "sliding prefill image at start",
        )
        .image(0, 40),
        case(
            256,
            8,
            4,
            12,
            2000,
            1024,
            1280,
            false,
            NO_LIMIT,
            "sliding split image",
        )
        .image(3, 7),
        case(
            512,
            32,
            4,
            1,
            0,
            0,
            0,
            false,
            NO_LIMIT,
            "global first token",
        ),
        case(
            512,
            32,
            4,
            1,
            2999,
            0,
            0,
            false,
            NO_LIMIT,
            "global multi-split",
        ),
        case(512, 8, 1, 7, 1500, 0, 0, false, NO_LIMIT, "global verify"),
        case(512, 8, 1, 33, 600, 0, 0, false, NO_LIMIT, "global prefill"),
        case(
            512,
            32,
            4,
            47,
            0,
            0,
            0,
            false,
            NO_LIMIT,
            "global prefill from zero",
        ),
        case(
            512,
            8,
            1,
            3,
            1500,
            0,
            0,
            true,
            1500,
            "draft global frontier",
        ),
        case(
            256,
            8,
            4,
            3,
            2000,
            1024,
            3072,
            true,
            2000,
            "draft sliding frontier",
        ),
        // Production global layers derive K from V.
        case(
            512,
            32,
            4,
            1,
            2999,
            0,
            0,
            false,
            NO_LIMIT,
            "derived global multi-split",
        )
        .derived(),
        case(
            512,
            8,
            1,
            7,
            1500,
            0,
            0,
            false,
            NO_LIMIT,
            "derived global verify",
        )
        .derived(),
        case(
            512,
            8,
            1,
            8,
            2100,
            0,
            0,
            false,
            NO_LIMIT,
            "derived global verify eight rows",
        )
        .derived(),
        case(
            512,
            8,
            1,
            33,
            600,
            0,
            0,
            false,
            NO_LIMIT,
            "derived global prefill",
        )
        .derived(),
        case(
            512,
            32,
            4,
            47,
            0,
            0,
            0,
            false,
            NO_LIMIT,
            "derived global prefill from zero",
        )
        .derived(),
        case(
            512,
            8,
            1,
            3,
            1500,
            0,
            0,
            true,
            1500,
            "derived draft global frontier",
        )
        .derived(),
        case(
            512,
            32,
            4,
            1,
            32767,
            0,
            0,
            false,
            NO_LIMIT,
            "global decode 32K",
        )
        .derived(),
        case(
            512,
            32,
            4,
            5,
            32763,
            0,
            0,
            false,
            NO_LIMIT,
            "global verify 32K",
        )
        .derived(),
        case(
            256,
            32,
            16,
            1,
            32767,
            1024,
            3072,
            false,
            NO_LIMIT,
            "sliding decode 32K",
        ),
        case(
            256,
            32,
            16,
            5,
            32763,
            1024,
            3072,
            false,
            NO_LIMIT,
            "sliding verify 32K",
        ),
        case(
            512,
            32,
            4,
            512,
            32768,
            0,
            0,
            false,
            NO_LIMIT,
            "global prefill 32K",
        )
        .stride(97)
        .derived(),
        case(
            512,
            32,
            4,
            2048,
            32768,
            0,
            0,
            false,
            NO_LIMIT,
            "global prefill 2048 rows at 32K",
        )
        .stride(389)
        .derived(),
        case(
            256,
            32,
            16,
            512,
            32768,
            1024,
            1536,
            false,
            NO_LIMIT,
            "sliding prefill 32K",
        )
        .stride(97),
        // The 26B-A4B's sliding layers: 16 query heads over 8 KV heads.
        case(
            256,
            16,
            8,
            2048,
            32768,
            1024,
            3072,
            false,
            NO_LIMIT,
            "26B sliding prefill 2048 rows at 32K",
        )
        .stride(389),
        // The 26B-A4B's global layers: 16 query heads over 2 KV heads.
        case(
            512,
            16,
            2,
            2048,
            32768,
            0,
            0,
            false,
            NO_LIMIT,
            "26B global prefill 2048 rows at 32K",
        )
        .stride(389)
        .derived(),
    ];
    for c in &cases {
        check(c, &mut rng);
    }
}
