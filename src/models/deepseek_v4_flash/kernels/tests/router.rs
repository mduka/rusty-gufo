//! Router probabilities (`router_test` of the C++ tests): the square root of
//! the softplus of 256 logits per row, for dense small scores, threshold
//! crossings, ties and zero rows, within the independent FP64 bound that
//! both the ROCm 7.2 F32 path and the 10.1 path (rounding only the final
//! score) meet. The selected experts and their weights stay on the device
//! for the routed experts, so their part of the C++ test has no
//! counterpart here:
//! cargo test -p gufo-deepseek-v4-flash-kernels --features gpu --test router

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_deepseek_v4_flash_kernels::{Backend, N_EXPERT};

fn run(b: &Backend, rows: u32) {
    let experts = N_EXPERT as usize;
    let mut logits = vec![0.0f32; rows as usize * experts];
    for row in 0..rows as usize {
        for expert in 0..experts {
            // Dense small scores, threshold crossings, tied scores and zero
            // rows.
            let mut value = ((expert * 73 + row * 17) % 257) as i32 as f32 - 128.0;
            value /= 5.5;
            if row % 4 == 0 {
                value = 0.0;
            } else if row % 4 == 1 {
                value /= 17.0;
            }
            logits[row * experts + expert] = value;
        }
    }
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut x = b.alloc(bytes.len() as u64).unwrap();
    b.write(&mut x, 0, &bytes).unwrap();
    let mut tokens = b.alloc(u64::from(rows) * 4).unwrap();
    b.write(&mut tokens, 0, &vec![0u8; rows as usize * 4])
        .unwrap();
    let mut probs = b.alloc(bytes.len() as u64).unwrap();
    let mut selection = b.alloc_expert_selection(rows).unwrap();
    b.router_select_batch(
        &mut selection,
        0,
        &mut probs,
        &x,
        &tokens,
        0,
        None,
        None,
        rows,
    )
    .unwrap();
    let mut out = vec![0u8; bytes.len()];
    b.read(&probs, 0, &mut out).unwrap();
    for (i, chunk) in out.as_chunks::<4>().0.iter().enumerate() {
        let actual = f32::from_le_bytes(*chunk);
        let value = f64::from(logits[i]);
        let softplus = if value > 20.0 {
            value
        } else if value < -10.0 {
            value.exp()
        } else {
            value.exp().ln_1p()
        };
        let expected = softplus.sqrt() as f32;
        assert!(
            actual.is_finite() && (actual - expected).abs() <= 2e-7 * (1.0 + expected),
            "router probability rows={rows} row={} expert={}: {actual:.9} != {expected:.9}",
            i / experts,
            i % experts
        );
    }
}

#[test]
fn router_probabilities_meet_the_fp64_bound() {
    let b = Backend::init().unwrap();
    for rows in [1u32, 4, 31, 32, 129] {
        run(&b, rows);
    }
}
