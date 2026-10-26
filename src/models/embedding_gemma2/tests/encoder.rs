//! The GPU encoder against the f32 CPU reference on fixed token sequences;
//! packing sequences into one batch must not change any embedding beyond
//! rounding, and concurrent calls take device turns one batch at a time:
//! cargo test --profile gpu-test -p gufo-embedding-gemma2 --features gpu,models --test encoder

#![cfg(all(feature = "gpu", feature = "models"))]
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Mutex;
use std::time::Duration;

use common::cosine;
use common::reference::CpuEncoder;
use gufo_embedding_gemma2::EncoderRuntime;
use gufo_embedding_gemma2::loader::load_model_directory;

/// `<bos>`, ids from 1,000 to 100,999 and `<eos>`, deterministic per seed.
fn random_tokens(length: usize, seed: u64) -> Vec<u32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut tokens = vec![2];
    while tokens.len() + 1 < length {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        tokens.push(1000 + (state % 100_000) as u32);
    }
    tokens.push(1);
    tokens
}

/// A long call holds the device for one batch at a time. A short call that
/// arrives during the first batch runs before the second, and neither result
/// changes. The cancellation hook runs at the start of each turn, so it
/// records the order of turns.
fn check_turns(runtime: &EncoderRuntime) {
    let long_call = vec![
        random_tokens(8000, 11),
        random_tokens(8000, 12),
        random_tokens(8000, 13),
    ];
    let short_call = vec![random_tokens(16, 14)];
    let long_alone = runtime.embed(&long_call, None).unwrap();
    let short_alone = runtime.embed(&short_call, None).unwrap();

    let order = Mutex::new(String::new());
    let short_result = Mutex::new(None);
    std::thread::scope(|scope| {
        let on_long_turn = || {
            {
                let mut order = order.lock().unwrap();
                order.push('L');
                if order.len() > 1 {
                    return false;
                }
            }
            scope.spawn(|| {
                let on_short_turn = || {
                    order.lock().unwrap().push('S');
                    false
                };
                *short_result.lock().unwrap() =
                    Some(runtime.embed(&short_call, Some(&on_short_turn)));
            });
            // The short call queues for a turn while this batch holds one.
            std::thread::sleep(Duration::from_millis(200));
            false
        };
        let long_result = runtime.embed(&long_call, Some(&on_long_turn)).unwrap();
        for (result, alone) in long_result.iter().zip(&long_alone) {
            assert!(
                cosine(result, alone) > 0.9999,
                "interleaving must not change a long-call embedding"
            );
        }
    });
    let order = order.into_inner().unwrap();
    println!("turn order {order}");
    assert_eq!(
        order, "LSLL",
        "a short call runs between the batches of a long one"
    );
    let short_result = short_result
        .into_inner()
        .unwrap()
        .expect("the short call ran")
        .expect("the concurrent short call succeeds");
    assert!(
        cosine(&short_result[0], &short_alone[0]) > 0.9999,
        "interleaving must not change a short-call embedding"
    );
}

#[test]
fn gpu_embeddings_match_the_cpu_reference_and_take_turns() {
    let root = common::checkpoint("model.safetensors");
    let runtime = EncoderRuntime::create(&root).unwrap_or_else(|e| panic!("{e}"));
    let sequences = vec![
        random_tokens(9, 1),
        random_tokens(24, 2),
        random_tokens(3, 3),
        random_tokens(48, 4),
    ];
    let batched = runtime.embed(&sequences, None).unwrap();
    assert_eq!(batched.len(), sequences.len(), "one embedding per sequence");

    let loaded = load_model_directory(&root).unwrap();
    let cpu = CpuEncoder::new(&loaded.config.text, &loaded.store);
    for (i, sequence) in sequences.iter().enumerate() {
        assert_eq!(batched[i].len(), 768, "embedding width");
        let norm: f64 = batched[i]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();
        assert!((norm - 1.0).abs() < 1e-3, "embeddings are unit length");
        let expected = cpu.embed(sequence);
        let against_cpu = cosine(&batched[i], &expected);
        println!("sequence {i} cosine vs CPU reference {against_cpu:.8}");
        assert!(
            against_cpu > 0.9995,
            "GPU embedding disagrees with the CPU reference"
        );
        let single = runtime.embed(std::slice::from_ref(sequence), None).unwrap();
        let alone = cosine(&single[0], &batched[i]);
        println!("sequence {i} cosine alone vs batched {alone:.8}");
        // The tuned GEMM may pick a different algorithm for a different batch
        // size, which changes f32 accumulation order but not the result by
        // more than rounding. A leak between sequences would be far larger.
        assert!(alone > 0.9999, "batching must not change an embedding");
    }

    check_turns(&runtime);
    assert!(!runtime.device_lost(), "no device failure was reported");

    assert!(
        runtime.embed(&[Vec::new()], None).is_err(),
        "empty input is rejected"
    );
    assert!(
        runtime.embed(&[vec![2, 262_144]], None).is_err(),
        "out-of-range token is rejected"
    );
    let cancelled = || true;
    assert_eq!(
        runtime.embed(&sequences, Some(&cancelled)).err().unwrap(),
        gufo_embedding_gemma2::encoder::CANCELLED_MESSAGE,
        "a cancelled call fails before its first batch"
    );
}
