//! The C++ extension's `kernel_ops_test.cpp`: the trunk's HIP kernels (other
//! than the quantized matrix products) against direct float/double CPU
//! evaluations of the same formulas, including state carried across chunks.
//!
//! cargo test --profile gpu-test -p gufo-nemotron --features gpu --test kernel_ops

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::too_many_lines)]

mod kernel_common;

use gufo_core::sampling::{LogitCandidate, SamplerState, SamplingConfig};
use gufo_hip::{F16, Stream};
use gufo_nemotron::logit_head::{LogitHead, LogitHeadFilter, head_filter_for, sample_with_head};
use gufo_nemotron_kernels as k;
use gufo_nemotron_kernels::sizes;
use kernel_common::*;

#[test]
fn embed_and_norm() {
    let s = stream();
    let mut rng = Rng(0x6B65_726E_656C_0001);
    const HIDDEN: usize = 256;
    const VOCAB: usize = 7;
    let mut table = vec![0u8; VOCAB * (HIDDEN / 32) * 34];
    let mut expected_rows = vec![0f32; VOCAB * HIDDEN];
    for r in 0..VOCAB {
        for b in 0..HIDDEN / 32 {
            let block = &mut table[(r * (HIDDEN / 32) + b) * 34..][..34];
            let d = half(0.01 * (b + 1) as f32);
            block[..2].copy_from_slice(&d.0.to_le_bytes());
            for i in 0..32 {
                let v = rng.below(255) as i32 - 127;
                block[2 + i] = v as i8 as u8;
                expected_rows[r * HIDDEN + b * 32 + i] = d.to_f32() * v as f32;
            }
        }
    }
    let tokens = [3, 0, 6, 3];
    let d_table = up(&s, &table);
    let d_tokens = indices(&s, &tokens, VOCAB as u32);
    let mut d_x = zeros::<f32>(&s, tokens.len() * HIDDEN);
    ok(
        k::embed_q8_0(
            &s,
            d_table.as_slice(),
            d_tokens.view(),
            d_x.as_mut_slice(),
            4,
            HIDDEN,
            VOCAB,
        ),
        "embed",
    );
    let x = down(&s, &d_x, tokens.len() * HIDDEN);
    let want: Vec<f32> = tokens
        .iter()
        .flat_map(|&t| {
            expected_rows[t as usize * HIDDEN..(t as usize + 1) * HIDDEN]
                .iter()
                .copied()
        })
        .collect();
    assert_eq!(rel_err(&x, &want), 0.0, "Q8_0 embedding rows");

    let w = rng.random(HIDDEN, -1.0, 1.0);
    let d_w = up(&s, &w);
    let mut d_out = zeros::<f32>(&s, x.len());
    ok(
        k::rms_norm(
            &s,
            d_x.as_slice(),
            d_w.as_slice(),
            d_out.as_mut_slice(),
            4,
            HIDDEN,
            1e-5,
        ),
        "rmsnorm",
    );
    let out = down(&s, &d_out, x.len());
    let mut reference = vec![0f32; x.len()];
    for t in 0..4 {
        let ss: f64 = x[t * HIDDEN..(t + 1) * HIDDEN]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();
        let scale = 1.0 / ((ss / HIDDEN as f64) as f32 + 1e-5).sqrt();
        for i in 0..HIDDEN {
            reference[t * HIDDEN + i] = x[t * HIDDEN + i] * scale * w[i];
        }
    }
    let e = rel_err(&out, &reference);
    println!("rmsnorm rel err {e:.2e}");
    assert!(e < 1e-5, "RMSNorm");
    // The FP16 variant stores the same values rounded to nearest even.
    let mut d_out16 = zeros::<F16>(&s, x.len());
    ok(
        k::rms_norm_f16(
            &s,
            d_x.as_slice(),
            d_w.as_slice(),
            d_out16.as_mut_slice(),
            4,
            HIDDEN,
            1e-5,
        ),
        "rmsnorm16",
    );
    let out16 = down(&s, &d_out16, x.len());
    assert!(
        out16.iter().zip(&out).all(|(h, f)| *h == half(*f)),
        "FP16 RMSNorm rounds the F32 result"
    );
}

/// Conv, scan and gated norm over chunks against a token-by-token CPU
/// evaluation that carries the same state.
#[test]
fn mamba_chunks() {
    let s = stream();
    let mut rng = Rng(0x6B65_726E_656C_0002);
    const HEADS: usize = 16;
    const HEAD_DIM: usize = 64;
    const GROUPS: usize = 4;
    const STATE: usize = 128;
    const KERNEL: usize = 4;
    const INNER: usize = HEADS * HEAD_DIM;
    const CHANNELS: usize = INNER + 2 * GROUPS * STATE;
    const STRIDE: usize = INNER + CHANNELS + HEADS;
    const SCALE: f32 = 0.75;
    const EPS: f32 = 1e-5;
    let chunks = [5usize, 1, 19]; // 19 spans several staged tiles.
    let total: usize = chunks.iter().sum();
    let proj = rng.random(total * STRIDE, -1.0, 1.0);
    let conv_w = rng.random(CHANNELS * KERNEL, -0.5, 0.5);
    let conv_b = rng.random(CHANNELS, -0.1, 0.1);
    let dt_b = rng.random(HEADS, -2.0, 0.0);
    let a = rng.random(HEADS, -2.0, -0.1);
    let d = rng.random(HEADS, -1.0, 1.0);
    let norm_w = rng.random(INNER, 0.5, 1.5);

    // CPU reference, one token at a time.
    let mut conv_state = vec![0f32; (KERNEL - 1) * CHANNELS];
    let mut ssm = vec![0f32; INNER * STATE];
    let mut want = vec![0f32; total * INNER];
    for t in 0..total {
        let row = &proj[t * STRIDE..(t + 1) * STRIDE];
        let mut xbc = vec![0f32; CHANNELS];
        for c in 0..CHANNELS {
            let mut acc = 0f32;
            for j in 0..KERNEL - 1 {
                acc += conv_state[j * CHANNELS + c] * conv_w[c * KERNEL + j];
            }
            acc += row[INNER + c] * SCALE * conv_w[c * KERNEL + KERNEL - 1];
            xbc[c] = silu(acc + conv_b[c]);
        }
        for j in 0..KERNEL.saturating_sub(2) {
            let (dst, src) = conv_state.split_at_mut((j + 1) * CHANNELS);
            dst[j * CHANNELS..].copy_from_slice(&src[..CHANNELS]);
        }
        for c in 0..CHANNELS {
            conv_state[(KERNEL - 2) * CHANNELS + c] = row[INNER + c] * SCALE;
        }
        let y = &mut want[t * INNER..(t + 1) * INNER];
        for h in 0..HEADS {
            let dt_sp = softplus(row[INNER + CHANNELS + h] * SCALE + dt_b[h]);
            let da = (dt_sp * a[h]).exp();
            let g = h / (HEADS / GROUPS);
            for i in 0..HEAD_DIM {
                let ii = h * HEAD_DIM + i;
                let mut sum = 0f64;
                for n in 0..STATE {
                    let st = &mut ssm[ii * STATE + n];
                    *st = *st * da + xbc[INNER + g * STATE + n] * (xbc[ii] * dt_sp);
                    sum += f64::from(*st) * f64::from(xbc[INNER + GROUPS * STATE + g * STATE + n]);
                }
                y[ii] = sum as f32 + xbc[ii] * d[h];
            }
        }
        let gs = INNER / GROUPS;
        for g in 0..GROUPS {
            let mut ss = 0f64;
            for j in 0..gs {
                let v = &mut y[g * gs + j];
                *v *= silu(row[g * gs + j] * SCALE);
                ss += f64::from(*v) * f64::from(*v);
            }
            let scale = 1.0 / ((ss / gs as f64) as f32 + EPS).sqrt();
            for j in 0..gs {
                y[g * gs + j] *= scale * norm_w[g * gs + j];
            }
        }
    }

    let d_proj = up(&s, &proj);
    let d_conv_w = up(&s, &conv_w);
    let d_conv_b = up(&s, &conv_b);
    let d_dt_b = up(&s, &dt_b);
    let d_a = up(&s, &a);
    let d_d = up(&s, &d);
    let d_norm = up(&s, &norm_w);
    let mut d_conv_state = zeros::<f32>(&s, (KERNEL - 1) * CHANNELS);
    let mut d_ssm = zeros::<f32>(&s, INNER * STATE);
    let mut d_xbc = zeros::<f32>(&s, total * CHANNELS);
    let mut d_y = zeros::<f32>(&s, total * INNER);
    let mut offset = 0;
    for &n in &chunks {
        let p = d_proj.slice(offset * STRIDE..(offset + n) * STRIDE);
        ok(
            k::mamba_conv(
                &s,
                p,
                STRIDE,
                INNER,
                SCALE,
                d_conv_w.as_slice(),
                d_conv_b.as_slice(),
                d_conv_state.as_slice(),
                d_xbc.slice_mut(offset * CHANNELS..(offset + n) * CHANNELS),
                n,
                CHANNELS,
                KERNEL,
            ),
            "conv",
        );
        ok(
            k::mamba_conv_state(
                &s,
                p,
                STRIDE,
                INNER,
                SCALE,
                d_conv_state.as_mut_slice(),
                n,
                CHANNELS,
                KERNEL,
            ),
            "conv state",
        );
        ok(
            k::mamba_scan(
                &s,
                d_xbc.slice(offset * CHANNELS..(offset + n) * CHANNELS),
                CHANNELS,
                p,
                STRIDE,
                INNER + CHANNELS,
                SCALE,
                d_dt_b.as_slice(),
                d_a.as_slice(),
                d_d.as_slice(),
                d_ssm.as_mut_slice(),
                d_y.slice_mut(offset * INNER..(offset + n) * INNER),
                n,
                HEADS,
                HEAD_DIM,
                GROUPS,
                true,
            ),
            "scan",
        );
        // The FP16 variant on a copy stores the F32 result rounded.
        let len = n * INNER;
        let mut y_copy = zeros::<f32>(&s, len);
        s.copy(
            d_y.slice(offset * INNER..offset * INNER + len),
            y_copy.as_mut_slice(),
        )
        .unwrap();
        let mut y16 = zeros::<F16>(&s, len);
        ok(
            k::mamba_gate_norm_f16(
                &s,
                y_copy.as_mut_slice(),
                p,
                STRIDE,
                SCALE,
                d_norm.as_slice(),
                y16.as_mut_slice(),
                n,
                INNER,
                GROUPS,
                EPS,
            ),
            "gate norm16",
        );
        ok(
            k::mamba_gate_norm(
                &s,
                d_y.slice_mut(offset * INNER..offset * INNER + len),
                p,
                STRIDE,
                SCALE,
                d_norm.as_slice(),
                n,
                INNER,
                GROUPS,
                EPS,
            ),
            "gate norm",
        );
        let f32s = down(&s, &d_y, (offset + n) * INNER)[offset * INNER..].to_vec();
        let f16s = down(&s, &y16, len);
        assert!(
            f16s.iter().zip(&f32s).all(|(h, f)| *h == half(*f)),
            "FP16 gated norm rounds the F32 result"
        );
        offset += n;
    }
    let got = down(&s, &d_y, want.len());
    let e = rel_err(&got, &want);
    println!("mamba conv/scan/gate-norm rel err {e:.2e}");
    assert!(e < 1e-4, "Mamba-2 chunked conv, scan and gated group norm");
    // The device keeps the state state-major per head: [head][n][row].
    let device_state = down(&s, &d_ssm, ssm.len());
    let mut got_state = vec![0f32; ssm.len()];
    for h in 0..HEADS {
        for n in 0..STATE {
            for i in 0..HEAD_DIM {
                got_state[(h * HEAD_DIM + i) * STATE + n] =
                    device_state[(h * STATE + n) * HEAD_DIM + i];
            }
        }
    }
    let es = rel_err(&got_state, &ssm);
    println!("mamba carried state rel err {es:.2e}");
    assert!(es < 1e-4, "Mamba-2 state after the last chunk");
}

const HEADS: usize = 32;
const KV_HEADS: usize = 2;
const HD: usize = 128;
const KV: usize = KV_HEADS * HD;

fn position(s: &Stream, value: usize, capacity: usize, n: usize) -> gufo_hip::DeviceIndices {
    indices(s, &[value as i32], (capacity - n + 1) as u32)
}

/// Chunks through the KV cache; the last chunk spans several key tiles.
#[test]
fn attention_chunks() {
    let s = stream();
    let mut rng = Rng(0x6B65_726E_656C_0003);
    let chunks = [150usize, 3, 140];
    let total: usize = chunks.iter().sum();
    let q = rng.random(total * HEADS * HD, -1.0, 1.0);
    let kk = rng.random(total * KV, -1.0, 1.0);
    let vv = rng.random(total * KV, -1.0, 1.0);
    let scale = 1.0 / (HD as f32).sqrt();
    // CPU reference on half-rounded K/V, as stored in the cache.
    let kh: Vec<f32> = kk.iter().map(|&v| half_round(v)).collect();
    let vh: Vec<f32> = vv.iter().map(|&v| half_round(v)).collect();
    let mut want = vec![0f32; q.len()];
    for p in 0..total {
        for h in 0..HEADS {
            let kvh = h / (HEADS / KV_HEADS);
            let mut sc: Vec<f64> = (0..=p)
                .map(|j| {
                    (0..HD)
                        .map(|dd| {
                            f64::from(q[(p * HEADS + h) * HD + dd] * scale)
                                * f64::from(kh[j * KV + kvh * HD + dd])
                        })
                        .sum()
                })
                .collect();
            let m = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0;
            for x in &mut sc {
                *x = (*x - m).exp();
                denom += *x;
            }
            for dd in 0..HD {
                let acc: f64 = (0..=p)
                    .map(|j| sc[j] * f64::from(vh[j * KV + kvh * HD + dd]))
                    .sum();
                want[(p * HEADS + h) * HD + dd] = (acc / denom) as f32;
            }
        }
    }
    // The chunked kernel's arithmetic: exp2-domain scores of the FP16
    // scaled query, 64-key tiles with a running maximum, FP16 probabilities
    // for P V while the normalizer sums them unrounded.
    let log2e = std::f64::consts::LOG2_E;
    let mut want_kernel = vec![0f32; q.len()];
    for p in 0..total {
        for h in 0..HEADS {
            let kvh = h / (HEADS / KV_HEADS);
            let qr: Vec<f32> = (0..HD)
                .map(|dd| {
                    half_round(q[(p * HEADS + h) * HD + dd] * (f64::from(scale) * log2e) as f32)
                })
                .collect();
            let (mut m, mut l) = (f64::NEG_INFINITY, 0f64);
            let mut o = vec![0f64; HD];
            let mut k0 = 0;
            while k0 <= p {
                let k1 = (p + 1).min(k0 + 64);
                let sc: Vec<f64> = (k0..k1)
                    .map(|j| {
                        (0..HD)
                            .map(|dd| f64::from(qr[dd]) * f64::from(kh[j * KV + kvh * HD + dd]))
                            .sum()
                    })
                    .collect();
                let tile_max = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let new_m = m.max(tile_max);
                let corr = (m - new_m).exp2();
                l *= corr;
                for x in &mut o {
                    *x *= corr;
                }
                for j in k0..k1 {
                    let pj = (sc[j - k0] - new_m).exp2();
                    l += pj;
                    let ph = f64::from(half_round(pj as f32));
                    for dd in 0..HD {
                        o[dd] += ph * f64::from(vh[j * KV + kvh * HD + dd]);
                    }
                }
                m = new_m;
                k0 += 64;
            }
            for dd in 0..HD {
                want_kernel[(p * HEADS + h) * HD + dd] = (o[dd] / l) as f32;
            }
        }
    }

    let d_q = up(&s, &q);
    let d_k = up(&s, &kk);
    let d_v = up(&s, &vv);
    let mut kc = zeros::<F16>(&s, total * KV);
    let mut vc = zeros::<F16>(&s, total * KV);
    let mut d_out = zeros::<f32>(&s, q.len());
    let mut pos = 0;
    for &n in &chunks {
        let d_pos = position(&s, pos, total, n);
        ok(
            k::kv_append(
                &s,
                d_k.slice(pos * KV..(pos + n) * KV),
                d_v.slice(pos * KV..(pos + n) * KV),
                kc.as_mut_slice(),
                vc.as_mut_slice(),
                d_pos.view(),
                n,
                KV,
                total,
            ),
            "kv append",
        );
        ok(
            k::attention(
                &s,
                d_q.slice(pos * HEADS * HD..(pos + n) * HEADS * HD),
                kc.as_slice(),
                vc.as_slice(),
                d_out.slice_mut(pos * HEADS * HD..(pos + n) * HEADS * HD),
                d_pos.view(),
                n,
                HEADS,
                KV_HEADS,
                scale,
                total,
            ),
            "attention",
        );
        pos += n;
    }
    let got = down(&s, &d_out, q.len());
    let e = rel_err(&got, &want_kernel);
    let e_exact = rel_err(&got, &want);
    println!("attention rel err {e:.2e} (exact softmax {e_exact:.2e})");
    assert!(
        e < 3e-4,
        "chunked causal GQA attention over the half KV cache"
    );
    assert!(
        e_exact < 1e-3,
        "FP16 query and probabilities stay near exact"
    );

    // Split short chunks ending at several depths, including fewer keys than
    // splits.
    let short_partial = sizes::attention_short_partial(sizes::ATTENTION_SHORT_TOKENS, HEADS);
    let mut d_partial = zeros::<f32>(&s, short_partial);
    let mut d_short = zeros::<f32>(&s, sizes::ATTENTION_SHORT_TOKENS * HEADS * HD);
    let mut short = |first: usize, n: usize, d_short: &mut gufo_hip::DeviceBuffer<f32>| {
        let d_pos = position(&s, first, total, n);
        ok(
            k::attention_short(
                &s,
                d_q.slice(first * HEADS * HD..(first + n) * HEADS * HD),
                kc.as_slice(),
                vc.as_slice(),
                d_short.slice_mut(0..n * HEADS * HD),
                d_pos.view(),
                n,
                HEADS,
                KV_HEADS,
                scale,
                d_partial.slice_mut(0..sizes::attention_short_partial(n, HEADS)),
                total,
            ),
            "short attention",
        );
        down(&s, d_short, n * HEADS * HD)
    };
    let mut worst = 0f64;
    for (last, n) in [
        (0, 1),
        (5, 1),
        (31, 3),
        (64, 8),
        (200, 5),
        (total - 1, 1),
        (total - 1, 8),
    ] {
        let first = last + 1 - n;
        let got_short = short(first, n, &mut d_short);
        worst = worst.max(rel_err(
            &got_short,
            &want[first * HEADS * HD..(last + 1) * HEADS * HD],
        ));
    }
    println!("split short attention rel err {worst:.2e}");
    assert!(worst < 1e-3, "split short-chunk attention");

    // A verified block's rows equal one-token calls at their positions, bit
    // for bit.
    let mut differ = 0;
    for (first, n) in [
        (36, 2),
        (36, 8),
        (60, 8),
        (100, 8),
        (200, 4),
        (total - 8, 8),
    ] {
        let block = short(first, n, &mut d_short);
        for i in 0..n {
            let one = short(first + i, 1, &mut d_short);
            differ += one
                .iter()
                .zip(&block[i * HEADS * HD..(i + 1) * HEADS * HD])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
        }
    }
    println!("verified blocks vs one-token attention: {differ} values differ");
    assert_eq!(
        differ, 0,
        "a verified block's rows equal one-token attention"
    );
}

/// Verified blocks that cross 4,096 and 8,192 keys: each row equals a
/// one-token pass at its position, bit for bit.
#[test]
fn attention_split_boundary() {
    let s = stream();
    let mut rng = Rng(0x6B65_726E_656C_0004);
    const KEYS: usize = 8200;
    const BLOCK: usize = 8;
    let scale = 1.0 / (HD as f32).sqrt();
    let kh: Vec<F16> = rng
        .random(KEYS * KV, -1.0, 1.0)
        .into_iter()
        .map(half)
        .collect();
    let vh: Vec<F16> = rng
        .random(KEYS * KV, -1.0, 1.0)
        .into_iter()
        .map(half)
        .collect();
    let kc = up(&s, &kh);
    let vc = up(&s, &vh);
    let d_q = up(&s, &rng.random(BLOCK * HEADS * HD, -1.0, 1.0));
    let mut d_partial = zeros::<f32>(&s, sizes::attention_short_partial(BLOCK, HEADS));
    let mut d_out = zeros::<f32>(&s, BLOCK * HEADS * HD);
    let mut run = |q0: usize, first: usize, n: usize| {
        let d_pos = position(&s, first, KEYS, n);
        ok(
            k::attention_short(
                &s,
                d_q.slice(q0 * HEADS * HD..(q0 + n) * HEADS * HD),
                kc.as_slice(),
                vc.as_slice(),
                d_out.slice_mut(0..n * HEADS * HD),
                d_pos.view(),
                n,
                HEADS,
                KV_HEADS,
                scale,
                d_partial.slice_mut(0..sizes::attention_short_partial(n, HEADS)),
                KEYS,
            ),
            "attention",
        );
        down(&s, &d_out, n * HEADS * HD)
    };
    let mut differ = 0;
    for first in [4092, 8188] {
        let block = run(0, first, BLOCK);
        for i in 0..BLOCK {
            let one = run(i, first + i, 1);
            differ += one
                .iter()
                .zip(&block[i * HEADS * HD..(i + 1) * HEADS * HD])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
        }
    }
    println!("verified blocks across split boundaries: {differ} values differ");
    assert_eq!(
        differ, 0,
        "verified blocks across split boundaries equal one-token attention"
    );
}

#[test]
fn router_logits() {
    let s = stream();
    let mut rng = Rng(0x6B65_726E_656C_0005);
    const COLS: usize = 4096;
    for rows in [240usize, 496] {
        for tokens in [1usize, 3, 8] {
            let w = rng.random(rows * COLS, -1.0, 1.0);
            let x = rng.random(tokens * COLS, -1.0, 1.0);
            let mut want = vec![0f32; tokens * rows];
            for t in 0..tokens {
                for r in 0..rows {
                    let acc: f64 = (0..COLS)
                        .map(|c| f64::from(w[r * COLS + c]) * f64::from(x[t * COLS + c]))
                        .sum();
                    want[t * rows + r] = acc as f32;
                }
            }
            let d_w = up(&s, &w);
            let d_x = up(&s, &x);
            let mut d_out = zeros::<f32>(&s, want.len());
            ok(
                k::router_logits(
                    &s,
                    d_w.as_slice(),
                    d_x.as_slice(),
                    d_out.as_mut_slice(),
                    rows,
                    COLS,
                    tokens,
                ),
                "router logits",
            );
            let e = rel_err(&down(&s, &d_out, want.len()), &want);
            assert!(e < 1e-5, "router logits {rows}x{tokens}: {e:.2e}");
        }
    }
}

#[test]
fn routing() {
    let s = stream();
    let mut rng = Rng(0x6B65_726E_656C_0006);
    const TOKENS: usize = 5;
    const TOP: usize = 9;
    const WIDTH: usize = 96;
    for experts in [240usize, 496] {
        let mut logits = rng.random(TOKENS * experts, -4.0, 4.0);
        let bias = rng.random(experts, -0.05, 0.05);
        // A tie in the selection score must choose the lower expert.
        logits[7] = logits[3];
        let mut want_ids = Vec::new();
        let mut want_w = Vec::new();
        for t in 0..TOKENS {
            let sel = |e: usize| sigmoid(logits[t * experts + e]) + bias[e];
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&x, &y| sel(y).partial_cmp(&sel(x)).unwrap());
            let sum: f32 = order[..TOP]
                .iter()
                .map(|&e| sigmoid(logits[t * experts + e]))
                .sum();
            for &e in &order[..TOP] {
                want_ids.push(e as i32);
                want_w.push(sigmoid(logits[t * experts + e]) / sum.max(2f32.powi(-14)) * 5.0);
            }
        }
        let d_logits = up(&s, &logits);
        let d_bias = up(&s, &bias);
        let mut d_ids = index_out(&s, TOKENS * TOP, experts as u32);
        let mut d_w = zeros::<f32>(&s, TOKENS * TOP);
        ok(
            k::router_top_k(
                &s,
                d_logits.as_slice(),
                d_bias.as_slice(),
                experts,
                TOKENS,
                TOP,
                true,
                5.0,
                d_ids.view_mut(),
                d_w.as_mut_slice(),
            ),
            "router",
        );
        assert_eq!(
            down_indices(&s, &d_ids, want_ids.len()),
            want_ids,
            "router top-9 ids for {experts}"
        );
        let w = down(&s, &d_w, want_w.len());
        assert!(rel_err(&w, &want_w) < 1e-6, "router weights");

        let slots = rng.random(TOKENS * TOP * WIDTH, -1.0, 1.0);
        let d_slots = up(&s, &slots);
        let mut d_sum = zeros::<f32>(&s, TOKENS * WIDTH);
        ok(
            k::weighted_slot_sum(
                &s,
                d_slots.as_slice(),
                d_w.as_slice(),
                d_sum.as_mut_slice(),
                TOKENS,
                TOP,
                WIDTH,
            ),
            "slot sum",
        );
        let mut want_sum = vec![0f32; TOKENS * WIDTH];
        for t in 0..TOKENS {
            for sl in 0..TOP {
                for c in 0..WIDTH {
                    want_sum[t * WIDTH + c] += w[t * TOP + sl] * slots[(t * TOP + sl) * WIDTH + c];
                }
            }
        }
        assert!(
            rel_err(&down(&s, &d_sum, want_sum.len()), &want_sum) < 1e-6,
            "weighted expert sum"
        );
    }
}

fn sampling(
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    min_keep: usize,
) -> SamplingConfig {
    SamplingConfig {
        temperature,
        top_k,
        top_p,
        min_p,
        min_keep,
        ..SamplingConfig::default()
    }
}

/// Device logit heads against the host: the maximum, an F64 softmax sum,
/// every token at least the head's smallest value, the coverage the filters
/// need, and, for top-k heads, the sampler drawing the same token from head
/// and row.
#[test]
fn logit_heads() {
    let s = stream();
    let mut rng = Rng(0x6B65_726E_656C_0007);
    const ROWS: usize = 8;
    for vocab in [131_072usize, 5000] {
        // Rows: sharp, moderate, top ties, flat, no finite value, and
        // moderate rows with NaN and infinities.
        let mut logits = vec![0f32; ROWS * vocab];
        for r in 0..ROWS {
            let row = &mut logits[r * vocab..(r + 1) * vocab];
            let spread = match r {
                0 => 3.0,
                3 => 0.05,
                _ => 2.0,
            };
            for v in row.iter_mut() {
                *v = spread * rng.normal();
            }
            if r == 0 {
                row[vocab / 3] = 25.0;
            }
            if r == 2 {
                for i in 0..6 {
                    row[(i * 7919) % vocab] = 9.5;
                }
            }
            if r == 4 {
                row.fill(f32::NEG_INFINITY);
            }
            if r >= 5 {
                row[(r * 104_729) % vocab] = f32::NAN;
                row[(r * 1_299_709) % vocab] = f32::INFINITY;
                row[(r * 15_485_863) % vocab] = f32::NEG_INFINITY;
                row[(r * 17) % vocab] = 7.0;
            }
        }
        let d_logits = up(&s, &logits);
        let mut scratch = zeros::<u8>(&s, sizes::logit_head_scratch_bytes(ROWS, vocab));
        let mut d_records = zeros::<u8>(&s, ROWS * sizes::LOGIT_HEAD_RECORD_BYTES);
        let mut d_candidates = zeros::<u8>(
            &s,
            ROWS * sizes::LOGIT_HEAD_CANDIDATES * sizes::LOGIT_HEAD_CANDIDATE_BYTES,
        );
        let configs = [
            sampling(1.0, 0, 0.95, 0.0, 0),
            sampling(0.7, 0, 0.9, 0.0, 3),
            sampling(1.0, 40, 0.95, 0.0, 0),
            sampling(1.3, 50, 1.0, 0.05, 0),
        ];
        for config in &configs {
            let top_k = head_filter_for(config);
            let filter = top_k.unwrap_or(LogitHeadFilter {
                temperature: config.temperature,
                keep: 0,
                top_p: config.top_p,
                minimum: config.min_keep.max(1),
            });
            ok(
                k::logit_heads(
                    &s,
                    d_logits.as_slice(),
                    ROWS,
                    vocab,
                    filter.temperature,
                    filter.keep,
                    filter.top_p,
                    filter.minimum,
                    scratch.as_mut_slice(),
                    d_records.as_mut_slice(),
                    d_candidates.as_mut_slice(),
                ),
                "logit heads",
            );
            let records = down(&s, &d_records, ROWS * sizes::LOGIT_HEAD_RECORD_BYTES);
            let candidates = down(
                &s,
                &d_candidates,
                ROWS * sizes::LOGIT_HEAD_CANDIDATES * sizes::LOGIT_HEAD_CANDIDATE_BYTES,
            );
            for r in 0..ROWS {
                let at = format!(
                    "logit head (vocab {vocab}, T {}, keep {}) row {r}",
                    filter.temperature, filter.keep
                );
                let row = &logits[r * vocab..(r + 1) * vocab];
                let record: &[u8; 16] = records[r * 16..(r + 1) * 16].try_into().unwrap();
                let (softmax_sum, maximum_got, count) = sizes::logit_head_record(record);
                let mut finite: Vec<f32> = row.iter().copied().filter(|v| v.is_finite()).collect();
                let maximum = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let temperature = f64::from(filter.temperature);
                let sum: f64 = row
                    .iter()
                    .filter(|v| v.is_finite())
                    .map(|&v| ((f64::from(v) - f64::from(maximum)) / temperature).exp())
                    .sum();
                finite.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let mut needed = filter.keep;
                if filter.keep == 0 {
                    let mut cumulative = 0f64;
                    needed = 0;
                    while needed < finite.len()
                        && (needed < filter.minimum || cumulative < f64::from(filter.top_p) * sum)
                    {
                        cumulative +=
                            ((f64::from(finite[needed]) - f64::from(maximum)) / temperature).exp();
                        needed += 1;
                    }
                }
                if finite.is_empty() || needed > sizes::LOGIT_HEAD_CANDIDATES {
                    assert_eq!(count, 0, "{at}: no head where none fits");
                } else if needed <= 256 && r != 3 {
                    // The flat row crowds more than a head into the bin of its cut.
                    assert!(count > 0, "{at}: a head where one fits");
                }
                if count == 0 {
                    continue;
                }
                assert_eq!(maximum_got, maximum, "{at}: maximum");
                assert!(
                    (softmax_sum - sum).abs() <= 1e-12 * sum,
                    "{at}: softmax sum"
                );
                let mut head = LogitHead {
                    maximum: maximum_got,
                    temperature: filter.temperature,
                    softmax_sum,
                    candidates: Vec::new(),
                };
                let mut seen = vec![false; vocab];
                let mut sorted = Vec::new();
                let mut smallest = f32::INFINITY;
                let mut valid = true;
                for c in 0..count as usize {
                    let at_byte = (r * sizes::LOGIT_HEAD_CANDIDATES + c) * 8;
                    let token =
                        u32::from_le_bytes(candidates[at_byte..at_byte + 4].try_into().unwrap());
                    let value = f32::from_le_bytes(
                        candidates[at_byte + 4..at_byte + 8].try_into().unwrap(),
                    );
                    let in_range = (token as usize) < vocab;
                    valid =
                        valid && in_range && !seen[token as usize] && value == row[token as usize];
                    if in_range {
                        seen[token as usize] = true;
                    }
                    smallest = smallest.min(value);
                    sorted.push(value);
                    head.candidates.push(LogitCandidate { token, value });
                }
                assert!(valid, "{at}: unique candidates with their logits");
                let above = row
                    .iter()
                    .filter(|v| v.is_finite() && **v >= smallest)
                    .count();
                assert_eq!(
                    above, count as usize,
                    "{at}: every token above the head's floor"
                );
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let mass: f64 = sorted
                    .iter()
                    .map(|&v| ((f64::from(v) - f64::from(maximum)) / temperature).exp())
                    .sum();
                let covers = if filter.keep > 0 {
                    sorted.len() >= filter.keep
                } else {
                    sorted.len() >= filter.minimum && mass >= f64::from(filter.top_p) * sum
                };
                assert!(covers, "{at}: covers the filters");
                if top_k.is_none() {
                    continue;
                }
                for seed in 0..32 {
                    let seeded = SamplingConfig {
                        seed,
                        ..config.clone()
                    };
                    let mut a = SamplerState::new(seeded.clone(), &[]).unwrap();
                    let mut b = SamplerState::new(seeded, &[]).unwrap();
                    assert_eq!(
                        a.sample(row).unwrap(),
                        sample_with_head(&mut b, row, &head).unwrap(),
                        "{at}: the head draws the row's token"
                    );
                    assert_eq!(a.rng_state(), b.rng_state(), "{at}: same RNG state");
                }
            }
        }
    }
}

/// One token of each of several sequences with their own states: the row
/// kernels match one-token calls of the sequence kernels bit for bit.
#[test]
fn batch_rows() {
    let s = stream();
    let mut rng = Rng(0x6B65_726E_656C_0008);
    const ROWS: usize = 3;
    {
        const HEADS: usize = 16;
        const HEAD_DIM: usize = 64;
        const GROUPS: usize = 4;
        const STATE: usize = 128;
        const KERNEL: usize = 4;
        const INNER: usize = HEADS * HEAD_DIM;
        const CHANNELS: usize = INNER + 2 * GROUPS * STATE;
        const STRIDE: usize = INNER + CHANNELS + HEADS;
        const SCALE: f32 = 0.75;
        const CONV: usize = (KERNEL - 1) * CHANNELS;
        const SSM: usize = INNER * STATE;
        let proj = rng.random(ROWS * STRIDE, -1.0, 1.0);
        let conv_w = rng.random(CHANNELS * KERNEL, -0.5, 0.5);
        let conv_b = rng.random(CHANNELS, -0.1, 0.1);
        let dt_b = rng.random(HEADS, -2.0, 0.0);
        let a = rng.random(HEADS, -2.0, -0.1);
        let d = rng.random(HEADS, -1.0, 1.0);
        let d_proj = up(&s, &proj);
        let (d_conv_w, d_conv_b) = (up(&s, &conv_w), up(&s, &conv_b));
        let (d_dt_b, d_a, d_d) = (up(&s, &dt_b), up(&s, &a), up(&s, &d));
        let (mut seq_conv, mut seq_ssm, mut row_conv, mut row_ssm) =
            (vec![], vec![], vec![], vec![]);
        for _ in 0..ROWS {
            let conv = rng.random(CONV, -1.0, 1.0);
            let ssm = rng.random(SSM, -0.5, 0.5);
            seq_conv.push(up(&s, &conv));
            seq_ssm.push(up(&s, &ssm));
            row_conv.push(up(&s, &conv));
            row_ssm.push(up(&s, &ssm));
        }
        let mut seq_xbc = zeros::<f32>(&s, ROWS * CHANNELS);
        let mut seq_y = zeros::<f32>(&s, ROWS * INNER);
        for r in 0..ROWS {
            let p = d_proj.slice(r * STRIDE..(r + 1) * STRIDE);
            ok(
                k::mamba_conv(
                    &s,
                    p,
                    STRIDE,
                    INNER,
                    SCALE,
                    d_conv_w.as_slice(),
                    d_conv_b.as_slice(),
                    seq_conv[r].as_slice(),
                    seq_xbc.slice_mut(r * CHANNELS..(r + 1) * CHANNELS),
                    1,
                    CHANNELS,
                    KERNEL,
                ),
                "conv",
            );
            ok(
                k::mamba_conv_state(
                    &s,
                    p,
                    STRIDE,
                    INNER,
                    SCALE,
                    seq_conv[r].as_mut_slice(),
                    1,
                    CHANNELS,
                    KERNEL,
                ),
                "conv state",
            );
            ok(
                k::mamba_scan(
                    &s,
                    seq_xbc.slice(r * CHANNELS..(r + 1) * CHANNELS),
                    CHANNELS,
                    p,
                    STRIDE,
                    INNER + CHANNELS,
                    SCALE,
                    d_dt_b.as_slice(),
                    d_a.as_slice(),
                    d_d.as_slice(),
                    seq_ssm[r].as_mut_slice(),
                    seq_y.slice_mut(r * INNER..(r + 1) * INNER),
                    1,
                    HEADS,
                    HEAD_DIM,
                    GROUPS,
                    true,
                ),
                "scan",
            );
        }
        let mut tables = zeros::<u8>(&s, ROWS * 8);
        let mut row_xbc = zeros::<f32>(&s, ROWS * CHANNELS);
        let mut row_y = zeros::<f32>(&s, ROWS * INNER);
        let mut states: Vec<_> = row_conv.iter_mut().map(|b| b.as_mut_slice()).collect();
        ok(
            k::mamba_conv_rows(
                &s,
                d_proj.as_slice(),
                STRIDE,
                INNER,
                SCALE,
                d_conv_w.as_slice(),
                d_conv_b.as_slice(),
                &mut states,
                tables.as_mut_slice(),
                row_xbc.as_mut_slice(),
                ROWS,
                CHANNELS,
                KERNEL,
            ),
            "conv rows",
        );
        let mut states: Vec<_> = row_ssm.iter_mut().map(|b| b.as_mut_slice()).collect();
        ok(
            k::mamba_scan_rows(
                &s,
                row_xbc.as_slice(),
                CHANNELS,
                d_proj.as_slice(),
                STRIDE,
                INNER + CHANNELS,
                SCALE,
                d_dt_b.as_slice(),
                d_a.as_slice(),
                d_d.as_slice(),
                &mut states,
                tables.as_mut_slice(),
                row_y.as_mut_slice(),
                ROWS,
                HEADS,
                HEAD_DIM,
                GROUPS,
            ),
            "scan rows",
        );
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(
            bits(down(&s, &row_xbc, ROWS * CHANNELS)),
            bits(down(&s, &seq_xbc, ROWS * CHANNELS)),
            "row convolution matches one-token convolutions"
        );
        assert_eq!(
            bits(down(&s, &row_y, ROWS * INNER)),
            bits(down(&s, &seq_y, ROWS * INNER)),
            "row scan matches one-token scans"
        );
        for r in 0..ROWS {
            assert_eq!(
                bits(down(&s, &row_conv[r], CONV)),
                bits(down(&s, &seq_conv[r], CONV)),
                "row conv state {r}"
            );
            assert_eq!(
                bits(down(&s, &row_ssm[r], SSM)),
                bits(down(&s, &seq_ssm[r], SSM)),
                "row scan state {r}"
            );
        }
    }
    {
        const CONTEXT: usize = 512;
        let positions = [0usize, 70, 300];
        let scale = 1.0 / (HD as f32).sqrt();
        let q = rng.random(ROWS * HEADS * HD, -1.0, 1.0);
        let k_new = rng.random(ROWS * KV, -1.0, 1.0);
        let v_new = rng.random(ROWS * KV, -1.0, 1.0);
        let (d_q, d_k, d_v) = (up(&s, &q), up(&s, &k_new), up(&s, &v_new));
        let (mut seq_k, mut seq_v, mut row_k, mut row_v) = (vec![], vec![], vec![], vec![]);
        for _ in 0..ROWS {
            let kc: Vec<F16> = rng
                .random(CONTEXT * KV, -1.0, 1.0)
                .into_iter()
                .map(half)
                .collect();
            let vc: Vec<F16> = rng
                .random(CONTEXT * KV, -1.0, 1.0)
                .into_iter()
                .map(half)
                .collect();
            seq_k.push(up(&s, &kc));
            seq_v.push(up(&s, &vc));
            row_k.push(up(&s, &kc));
            row_v.push(up(&s, &vc));
        }
        let mut partial = zeros::<f32>(
            &s,
            sizes::attention_short_partial(sizes::ATTENTION_SHORT_TOKENS, HEADS),
        );
        let mut seq_out = zeros::<f32>(&s, ROWS * HEADS * HD);
        for r in 0..ROWS {
            let pos = position(&s, positions[r], CONTEXT, 1);
            ok(
                k::kv_append(
                    &s,
                    d_k.slice(r * KV..(r + 1) * KV),
                    d_v.slice(r * KV..(r + 1) * KV),
                    seq_k[r].as_mut_slice(),
                    seq_v[r].as_mut_slice(),
                    pos.view(),
                    1,
                    KV,
                    CONTEXT,
                ),
                "kv append",
            );
            ok(
                k::attention_short(
                    &s,
                    d_q.slice(r * HEADS * HD..(r + 1) * HEADS * HD),
                    seq_k[r].as_slice(),
                    seq_v[r].as_slice(),
                    seq_out.slice_mut(r * HEADS * HD..(r + 1) * HEADS * HD),
                    pos.view(),
                    1,
                    HEADS,
                    KV_HEADS,
                    scale,
                    partial.slice_mut(0..sizes::attention_short_partial(1, HEADS)),
                    CONTEXT,
                ),
                "attention",
            );
            s.synchronize().unwrap();
        }
        let row_positions: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let d_pos = indices(&s, &row_positions, CONTEXT as u32);
        let mut tables = zeros::<u8>(&s, 2 * ROWS * 8);
        let mut row_out = zeros::<f32>(&s, ROWS * HEADS * HD);
        {
            let mut ks: Vec<_> = row_k.iter_mut().map(|b| b.as_mut_slice()).collect();
            let mut vs: Vec<_> = row_v.iter_mut().map(|b| b.as_mut_slice()).collect();
            ok(
                k::kv_append_rows(
                    &s,
                    d_k.as_slice(),
                    d_v.as_slice(),
                    &mut ks,
                    &mut vs,
                    tables.as_mut_slice(),
                    d_pos.view(),
                    ROWS,
                    KV,
                    CONTEXT,
                ),
                "kv append rows",
            );
        }
        let ks: Vec<_> = row_k.iter().map(|b| b.as_slice()).collect();
        let vs: Vec<_> = row_v.iter().map(|b| b.as_slice()).collect();
        ok(
            k::attention_short_rows(
                &s,
                d_q.as_slice(),
                &ks,
                &vs,
                tables.as_mut_slice(),
                row_out.as_mut_slice(),
                d_pos.view(),
                ROWS,
                HEADS,
                KV_HEADS,
                scale,
                partial.slice_mut(0..sizes::attention_short_partial(ROWS, HEADS)),
                CONTEXT,
            ),
            "attention rows",
        );
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(
            bits(down(&s, &row_out, ROWS * HEADS * HD)),
            bits(down(&s, &seq_out, ROWS * HEADS * HD)),
            "row attention matches one-token attention"
        );
        for r in 0..ROWS {
            assert_eq!(
                down(&s, &row_k[r], CONTEXT * KV),
                down(&s, &seq_k[r], CONTEXT * KV),
                "row K append {r}"
            );
            assert_eq!(
                down(&s, &row_v[r], CONTEXT * KV),
                down(&s, &seq_v[r], CONTEXT * KV),
                "row V append {r}"
            );
        }
    }
}
