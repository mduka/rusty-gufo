//! The C++ extension's `drafter_ops_test.cpp`: the LiLiCorr drafter's HIP
//! kernels against direct double CPU evaluations of llama.cpp's formulas:
//! the BF16 weight-streaming product, the block-local dynamic convolution,
//! NeoX RoPE, ring attention with sinks and a sliding window across the
//! ring's wrap, the candidate top-8 with log-sum-exp, the candidate
//! features, node attention and the lattice scores.
//!
//! cargo test --profile gpu-test -p gufo-nemotron --features gpu --test drafter_ops

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

mod kernel_common;

use gufo_hip::{Bf16, F16};
use gufo_nemotron_kernels as k;
use kernel_common::*;

#[test]
fn gemv_bf16() {
    let s = stream();
    let mut rng = Rng(0x6472_6166_7465_7201);
    const COLS: usize = 4096;
    for rows in [256usize, 1024, 4096] {
        for tokens in [1usize, 3, 8] {
            let w: Vec<Bf16> = rng
                .random(rows * COLS, -1.0, 1.0)
                .into_iter()
                .map(bf16)
                .collect();
            let x = rng.random(tokens * COLS, -1.0, 1.0);
            let mut want = vec![0f32; tokens * rows];
            for t in 0..tokens {
                for r in 0..rows {
                    let acc: f64 = (0..COLS)
                        .map(|c| f64::from(w[r * COLS + c].to_f32()) * f64::from(x[t * COLS + c]))
                        .sum();
                    want[t * rows + r] = acc as f32;
                }
            }
            let dw = up(&s, &w);
            let dx = up(&s, &x);
            let mut out = zeros::<f32>(&s, want.len());
            ok(
                k::gemv_bf16(
                    &s,
                    dw.as_slice(),
                    dx.as_slice(),
                    out.as_mut_slice(),
                    rows,
                    COLS,
                    tokens,
                ),
                "bf16 gemv",
            );
            let e = rel_err(&down(&s, &out, want.len()), &want);
            assert!(e < 1e-5, "bf16 gemv {rows}x{tokens}: {e:.2e}");
        }
    }
}

#[test]
fn dflash_conv() {
    let s = stream();
    let mut rng = Rng(0x6472_6166_7465_7202);
    const TOKENS: usize = 16; // two blocks
    const BLOCK: usize = 8;
    const HIDDEN: usize = 64;
    const GROUP: usize = 16;
    const KERNEL: usize = 2;
    const GROUPS: usize = HIDDEN / GROUP;
    let x = rng.random(TOKENS * HIDDEN, -1.0, 1.0);
    let dynamic = rng.random(TOKENS * 2 * KERNEL * GROUPS, -1.0, 1.0);
    let base = rng.random(2 * KERNEL * HIDDEN, -1.0, 1.0);
    let (dx, dd, db) = (up(&s, &x), up(&s, &dynamic), up(&s, &base));
    let mut out = zeros::<f32>(&s, x.len());
    for side in 0..2 {
        let mut want = vec![0f32; x.len()];
        for t in 0..TOKENS {
            for c in 0..HIDDEN {
                let mut acc = 0f64;
                for tap in 0..KERNEL.min(t % BLOCK + 1) {
                    let kk = side * KERNEL + tap;
                    let w = f64::from(dynamic[t * 2 * KERNEL * GROUPS + kk * GROUPS + c / GROUP])
                        + f64::from(base[kk * HIDDEN + c]);
                    acc += w * f64::from(x[(t - tap) * HIDDEN + c]);
                }
                want[t * HIDDEN + c] = acc as f32;
            }
        }
        ok(
            k::dflash_conv(
                &s,
                dx.as_slice(),
                dd.as_slice(),
                db.as_slice(),
                out.as_mut_slice(),
                TOKENS,
                BLOCK,
                HIDDEN,
                GROUP,
                KERNEL,
                side,
            ),
            "dflash conv",
        );
        assert!(
            rel_err(&down(&s, &out, want.len()), &want) < 1e-6,
            "dflash conv side {side}"
        );
    }
}

#[test]
fn rope_neox() {
    let s = stream();
    let mut rng = Rng(0x6472_6166_7465_7203);
    const TOKENS: usize = 3;
    const HEADS: usize = 4;
    const DIM: usize = 128;
    const FIRST: i32 = 1000;
    let x = rng.random(TOKENS * HEADS * DIM, -1.0, 1.0);
    let mut want = x.clone();
    for t in 0..TOKENS {
        for h in 0..HEADS {
            let row = &mut want[(t * HEADS + h) * DIM..(t * HEADS + h + 1) * DIM];
            let half = DIM / 2;
            for i in 0..half {
                let angle =
                    f64::from(FIRST + t as i32) * 10000f64.powf(-2.0 * i as f64 / DIM as f64);
                let (a, b) = (f64::from(row[i]), f64::from(row[i + half]));
                row[i] = (a * angle.cos() - b * angle.sin()) as f32;
                row[i + half] = (a * angle.sin() + b * angle.cos()) as f32;
            }
        }
    }
    let mut dx = up(&s, &x);
    let first = up(&s, &[FIRST]);
    ok(
        k::rope_neox(
            &s,
            dx.as_mut_slice(),
            first.as_slice(),
            TOKENS,
            HEADS,
            DIM,
            10000.0,
        ),
        "rope",
    );
    // The device evaluates the angle in float, as ggml does.
    assert!(rel_err(&down(&s, &dx, x.len()), &want) < 5e-4, "neox rope");
}

#[test]
fn ring_attention() {
    let s = stream();
    let mut rng = Rng(0x6472_6166_7465_7204);
    const HEADS: usize = 32;
    const KV_HEADS: usize = 2;
    const DIM: usize = 128;
    const KV: usize = KV_HEADS * DIM;
    const RING: usize = 64;
    const WINDOW: usize = 40;
    const TOKENS: usize = 8;
    const FIRST: usize = 100; // positions 44..107 occupy the ring, wrapped
    const CONTEXT: usize = RING - TOKENS;
    const POSITIONS: usize = 4096;
    let scale = 1.0 / (DIM as f32).sqrt();
    let ck = rng.random(CONTEXT * KV, -1.0, 1.0);
    let cv = rng.random(CONTEXT * KV, -1.0, 1.0);
    let bk = rng.random(TOKENS * KV, -1.0, 1.0);
    let bv = rng.random(TOKENS * KV, -1.0, 1.0);
    let q = rng.random(TOKENS * HEADS * DIM, -1.0, 1.0);
    let sinks = rng.random(HEADS, -2.0, 2.0);
    let mut kr = zeros::<F16>(&s, RING * KV);
    let mut vr = zeros::<F16>(&s, RING * KV);
    let (dck, dcv, dbk, dbv) = (up(&s, &ck), up(&s, &cv), up(&s, &bk), up(&s, &bv));
    let (dq, dsinks) = (up(&s, &q), up(&s, &sinks));
    let mut out = zeros::<f32>(&s, q.len());
    let ctx_first = indices(
        &s,
        &[(FIRST - CONTEXT) as i32],
        (POSITIONS - CONTEXT + 1) as u32,
    );
    let first = indices(&s, &[FIRST as i32], (POSITIONS - TOKENS + 1) as u32);
    ok(
        k::ring_append(
            &s,
            dck.as_slice(),
            dcv.as_slice(),
            kr.as_mut_slice(),
            vr.as_mut_slice(),
            ctx_first.view(),
            CONTEXT,
            KV,
            RING,
            POSITIONS,
        ),
        "ring context",
    );
    ok(
        k::ring_append(
            &s,
            dbk.as_slice(),
            dbv.as_slice(),
            kr.as_mut_slice(),
            vr.as_mut_slice(),
            first.view(),
            TOKENS,
            KV,
            RING,
            POSITIONS,
        ),
        "ring block",
    );
    ok(
        k::drafter_attention(
            &s,
            dq.as_slice(),
            kr.as_slice(),
            vr.as_slice(),
            dsinks.as_slice(),
            out.as_mut_slice(),
            first.view(),
            TOKENS,
            HEADS,
            KV_HEADS,
            RING,
            WINDOW,
            scale,
            POSITIONS,
        ),
        "drafter attention",
    );
    let key = |pos: usize, kvh: usize, d: usize, value: bool| {
        let j = pos - (FIRST - CONTEXT);
        let (src, row) = if j < CONTEXT {
            (if value { &cv } else { &ck }, j)
        } else {
            (if value { &bv } else { &bk }, j - CONTEXT)
        };
        half_round(src[row * KV + kvh * DIM + d])
    };
    let mut want = vec![0f32; q.len()];
    for t in 0..TOKENS {
        let pos = FIRST + t;
        let lo = (pos + 1).saturating_sub(WINDOW);
        let hi = FIRST + TOKENS - 1;
        for h in 0..HEADS {
            let kvh = h / (HEADS / KV_HEADS);
            let scores: Vec<f64> = (lo..=hi)
                .map(|p| {
                    let dot: f64 = (0..DIM)
                        .map(|d| {
                            f64::from(q[(t * HEADS + h) * DIM + d])
                                * f64::from(key(p, kvh, d, false))
                        })
                        .sum();
                    dot * f64::from(scale)
                })
                .collect();
            let m = scores.iter().copied().fold(f64::from(sinks[h]), f64::max);
            let z =
                (f64::from(sinks[h]) - m).exp() + scores.iter().map(|v| (v - m).exp()).sum::<f64>();
            for d in 0..DIM {
                let acc: f64 = (lo..=hi)
                    .map(|p| (scores[p - lo] - m).exp() * f64::from(key(p, kvh, d, true)))
                    .sum();
                want[(t * HEADS + h) * DIM + d] = (acc / z) as f32;
            }
        }
    }
    assert!(
        rel_err(&down(&s, &out, want.len()), &want) < 1e-5,
        "drafter attention with sinks over a wrapped ring"
    );
}

#[test]
fn top_k8() {
    let s = stream();
    let mut rng = Rng(0x6472_6166_7465_7205);
    const ROWS: usize = 3;
    const VOCAB: usize = 5000;
    let mut logits = rng.random(ROWS * VOCAB, -8.0, 8.0);
    // Ties in the top 8 resolve to the lower token id.
    logits[VOCAB + 4321] = 20.0;
    logits[VOCAB + 1234] = 20.0;
    let dl = up(&s, &logits);
    let mut ids = zeros::<i32>(&s, ROWS * 8);
    let mut values = zeros::<f32>(&s, ROWS * 8);
    let mut lse = zeros::<f32>(&s, ROWS);
    ok(
        k::top_k8(
            &s,
            dl.as_slice(),
            ROWS,
            VOCAB,
            ids.as_mut_slice(),
            values.as_mut_slice(),
            lse.as_mut_slice(),
        ),
        "top-8",
    );
    let (ids, values, lse) = (
        down(&s, &ids, ROWS * 8),
        down(&s, &values, ROWS * 8),
        down(&s, &lse, ROWS),
    );
    let mut want_lse = vec![0f32; ROWS];
    for r in 0..ROWS {
        let row = &logits[r * VOCAB..(r + 1) * VOCAB];
        let mut order: Vec<usize> = (0..VOCAB).collect();
        order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
        for j in 0..8 {
            assert_eq!(
                ids[r * 8 + j],
                order[j] as i32,
                "top-8 candidates in descending order"
            );
            assert_eq!(values[r * 8 + j], row[order[j]], "top-8 values");
        }
        let m = f64::from(row.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let z: f64 = row.iter().map(|&v| (f64::from(v) - m).exp()).sum();
        want_lse[r] = (m + z.ln()) as f32;
    }
    assert!(rel_err(&lse, &want_lse) < 1e-6, "top-8 log-sum-exp");
}

#[test]
fn lilicorr() {
    let s = stream();
    let mut rng = Rng(0x6472_6166_7465_7206);
    const SLOTS: usize = 7;
    const TOP: usize = 8;
    const NODES: usize = SLOTS * TOP;
    const WIDTH: usize = 1024;
    const HEADS: usize = 8;
    const DIM: usize = 128;
    // Features.
    let mut values = vec![0f32; NODES];
    let mut lse = vec![0f32; SLOTS];
    for sl in 0..SLOTS {
        let mut v = rng.random(TOP, -6.0, 6.0);
        v.sort_by(|a, b| b.partial_cmp(a).unwrap());
        values[sl * TOP..(sl + 1) * TOP].copy_from_slice(&v);
        lse[sl] = v[0] + 0.7;
    }
    let fw = rng.random(5, -1.0, 1.0);
    let fb = rng.random(5, -1.0, 1.0);
    let mut want = vec![0f32; NODES * 5];
    for n in 0..NODES {
        let (sl, r) = (n / TOP, n % TOP);
        let lp = f64::from(values[n]) - f64::from(lse[sl]);
        let lp0 = f64::from(values[sl * TOP]) - f64::from(lse[sl]);
        let f = [
            lp,
            lp.exp(),
            lp - lp0,
            r as f64 / 7.0,
            if r == 0 { 1.0 } else { 0.0 },
        ];
        let mean: f64 = f.iter().map(|v| v / 5.0).sum();
        let var: f64 = f.iter().map(|v| (v - mean) * (v - mean) / 5.0).sum();
        for i in 0..5 {
            want[n * 5 + i] =
                ((f[i] - mean) / (var + 1e-5).sqrt() * f64::from(fw[i]) + f64::from(fb[i])) as f32;
        }
    }
    let (dv, dlse, dfw, dfb) = (up(&s, &values), up(&s, &lse), up(&s, &fw), up(&s, &fb));
    let mut feat = zeros::<f32>(&s, want.len());
    ok(
        k::lilicorr_features(
            &s,
            dv.as_slice(),
            dlse.as_slice(),
            dfw.as_slice(),
            dfb.as_slice(),
            feat.as_mut_slice(),
            SLOTS,
        ),
        "features",
    );
    assert!(
        rel_err(&down(&s, &feat, want.len()), &want) < 1e-5,
        "lilicorr candidate features"
    );

    // Node attention with a per-head additive bias.
    let qkv = rng.random(NODES * 3 * WIDTH, -1.0, 1.0);
    let bias = rng.random(HEADS * NODES * NODES, -3.0, 3.0);
    let scale = 1.0 / (DIM as f32).sqrt();
    let mut att = vec![0f32; NODES * WIDTH];
    for h in 0..HEADS {
        for qi in 0..NODES {
            let mut p: Vec<f64> = (0..NODES)
                .map(|j| {
                    let dot: f64 = (0..DIM)
                        .map(|d| {
                            f64::from(qkv[qi * 3 * WIDTH + h * DIM + d])
                                * f64::from(qkv[j * 3 * WIDTH + WIDTH + h * DIM + d])
                        })
                        .sum();
                    dot * f64::from(scale) + f64::from(bias[(h * NODES + qi) * NODES + j])
                })
                .collect();
            let m = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut z = 0f64;
            for v in &mut p {
                *v = (*v - m).exp();
                z += *v;
            }
            for d in 0..DIM {
                let acc: f64 = (0..NODES)
                    .map(|j| p[j] * f64::from(qkv[j * 3 * WIDTH + 2 * WIDTH + h * DIM + d]))
                    .sum();
                att[qi * WIDTH + h * DIM + d] = (acc / z) as f32;
            }
        }
    }
    let (dqkv, dbias) = (up(&s, &qkv), up(&s, &bias));
    let mut datt = zeros::<f32>(&s, att.len());
    ok(
        k::node_attention(
            &s,
            dqkv.as_slice(),
            dbias.as_slice(),
            datt.as_mut_slice(),
            NODES,
            HEADS,
            scale,
        ),
        "node attention",
    );
    assert!(
        rel_err(&down(&s, &datt, att.len()), &att) < 1e-5,
        "lilicorr node attention"
    );

    // Normalized factors and lattice scores.
    let mut input = rng.random(NODES * WIDTH, -1.0, 1.0);
    let mut output = rng.random(NODES * WIDTH, -1.0, 1.0);
    let mut anchor = rng.random(WIDTH, -1.0, 1.0);
    let (in_b, out_b, anchor_b) = (
        rng.random(WIDTH, -1.0, 1.0),
        rng.random(WIDTH, -1.0, 1.0),
        rng.random(WIDTH, -1.0, 1.0),
    );
    let (mut din, mut dout, mut danchor) = (up(&s, &input), up(&s, &output), up(&s, &anchor));
    let (din_b, dout_b, danchor_b) = (up(&s, &in_b), up(&s, &out_b), up(&s, &anchor_b));
    let normalize = |x: &mut [f32], b: &[f32], rows: usize| {
        for r in 0..rows {
            let row = &mut x[r * WIDTH..(r + 1) * WIDTH];
            let mut ss = 0f64;
            for c in 0..WIDTH {
                row[c] += b[c];
                ss += f64::from(row[c]) * f64::from(row[c]);
            }
            let inv = 1.0 / ss.sqrt().max(1e-4);
            for v in row.iter_mut() {
                *v = (f64::from(*v) * inv) as f32;
            }
        }
    };
    normalize(&mut input, &in_b, NODES);
    normalize(&mut output, &out_b, NODES);
    normalize(&mut anchor, &anchor_b, 1);
    ok(
        k::bias_l2_norm(&s, din.as_mut_slice(), din_b.as_slice(), NODES, WIDTH, 1e-4),
        "l2",
    );
    ok(
        k::bias_l2_norm(
            &s,
            dout.as_mut_slice(),
            dout_b.as_slice(),
            NODES,
            WIDTH,
            1e-4,
        ),
        "l2",
    );
    ok(
        k::bias_l2_norm(
            &s,
            danchor.as_mut_slice(),
            danchor_b.as_slice(),
            1,
            WIDTH,
            1e-4,
        ),
        "l2",
    );
    assert!(
        rel_err(&down(&s, &din, input.len()), &input) < 1e-5,
        "lilicorr l2 norm"
    );
    let dot = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum::<f64>()
    };
    let mut scores = vec![0f32; TOP + (SLOTS - 1) * TOP * TOP];
    for j in 0..TOP {
        scores[j] = (8.0 * dot(&input[j * WIDTH..(j + 1) * WIDTH], &anchor)) as f32;
    }
    for sl in 1..SLOTS {
        for prev in 0..TOP {
            for next in 0..TOP {
                let a = &input[(sl * TOP + next) * WIDTH..(sl * TOP + next + 1) * WIDTH];
                let b =
                    &output[((sl - 1) * TOP + prev) * WIDTH..((sl - 1) * TOP + prev + 1) * WIDTH];
                scores[TOP + (sl - 1) * TOP * TOP + prev * TOP + next] = (8.0 * dot(a, b)) as f32;
            }
        }
    }
    let mut dscores = zeros::<f32>(&s, scores.len());
    ok(
        k::lattice_scores(
            &s,
            din.as_slice(),
            dout.slice(0..(SLOTS - 1) * TOP * WIDTH),
            danchor.as_slice(),
            dscores.as_mut_slice(),
            SLOTS,
            WIDTH,
            8.0,
        ),
        "lattice",
    );
    assert!(
        rel_err(&down(&s, &dscores, scores.len()), &scores) < 1e-5,
        "lilicorr lattice scores"
    );
}
