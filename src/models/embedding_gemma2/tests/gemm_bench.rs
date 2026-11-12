//! Times every GEMM shape the encoder issues through hipBLAS and through
//! every workspace-free hipBLASLt heuristic result, and prints the best. It
//! does not go through the encoder's tuned plans and asserts nothing.
//! Timings are meaningful only on gfx1151, on an otherwise idle device:
//! cargo test --profile gpu-test -p gufo-embedding-gemma2 --features gpu --test gemm_bench -- --ignored --nocapture

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::cell::Cell;

use gufo_embedding_gemma2::gemm::Gemm;
use gufo_hip::blas::{AlgoPolicy, BlasLt, Heuristic, Layout, MatmulShape, Op};
use gufo_hip::{Bf16, Event, Launch, Stream};

const RUNS: usize = 20;
const MAX_ALGORITHMS: usize = 32;

/// Installs ranked result `rank` when it needs no workspace, and records how
/// many results hipBLASLt returned.
struct Rank {
    rank: usize,
    found: Cell<usize>,
}

impl AlgoPolicy for Rank {
    fn requested(&self) -> usize {
        MAX_ALGORITHMS
    }

    fn choose(&self, ranked: &[Heuristic], capacity: usize) -> Option<usize> {
        self.found.set(ranked.len());
        ranked
            .get(self.rank)
            .filter(|h| h.is_usable(capacity))
            .map(|_| self.rank)
    }

    fn resolve_names(&self) -> bool {
        false
    }
}

fn time_ms(stream: &Stream, mut launch: impl FnMut()) -> f64 {
    for _ in 0..3 {
        launch();
    }
    stream.synchronize().unwrap();
    let (begin, end) = (Event::timed().unwrap(), Event::timed().unwrap());
    stream.record(&begin).unwrap();
    for _ in 0..RUNS {
        launch();
    }
    stream.record(&end).unwrap();
    end.synchronize().unwrap();
    f64::from(Event::elapsed_ms(&begin, &end).unwrap()) / RUNS as f64
}

#[test]
#[ignore = "benchmark: prints timings, asserts nothing"]
fn gemm_shapes_through_hipblas_and_every_hipblaslt_result() {
    let stream = Stream::new().unwrap();
    let gemm = Gemm::new(&stream).unwrap();
    let shapes = [
        ("qkv_sliding", 2048, 512),
        ("qkv_full", 3072, 512),
        ("o_sliding", 512, 1024),
        ("o_full", 512, 2048),
        ("gate_up", 4096, 512),
        ("down", 512, 2048),
        ("ple_512", 512, 512),
        ("final", 768, 512),
    ];
    let token_counts = [32, 128, 256, 512, 1024];
    let mut weights = stream.alloc::<Bf16>(4096 * 2048).unwrap();
    let mut inputs = stream.alloc::<Bf16>(1024 * 2048).unwrap();
    let mut outputs = stream.alloc::<f32>(1024 * 4096).unwrap();
    stream.fill_bytes(weights.as_mut_slice(), 0x3c).unwrap();
    stream.fill_bytes(inputs.as_mut_slice(), 0x3c).unwrap();

    println!("shape tokens | hipblas ms (TFLOPS) | lt rank: ms ...");
    for (name, m, k) in shapes {
        for tokens in token_counts {
            let flops = 2.0 * tokens as f64 * m as f64 * k as f64;
            let blas = time_ms(&stream, || {
                gemm.blas_gemm(
                    &stream,
                    weights.slice(0..m * k),
                    inputs.slice(0..tokens * k),
                    outputs.slice_mut(0..tokens * m),
                    tokens,
                    m,
                    k,
                )
                .unwrap();
            });
            let mut line = format!("{name} {tokens} | {blas:.4} ({:.2}) |", flops / blas / 1e9);
            let shape = MatmulShape {
                trans_a: Some(Op::T),
                trans_b: None,
                a: Layout::new(k, m, k),
                b: Layout::new(k, tokens, k),
                c: Layout::new(m, tokens, m),
                batch: None,
            };
            let mut best = (f64::INFINITY, None);
            let mut found = MAX_ALGORITHMS;
            let mut rank = 0;
            while rank < found {
                let policy = Rank {
                    rank,
                    found: Cell::new(0),
                };
                let mut lt = BlasLt::new(&stream, 0).unwrap();
                let mut run = || {
                    lt.matmul(
                        &stream,
                        &shape,
                        weights.slice(0..m * k),
                        inputs.slice(0..tokens * k),
                        outputs.slice_mut(0..tokens * m),
                        &policy,
                    )
                    .unwrap()
                };
                let usable = run() == Launch::Launched;
                found = policy.found.get();
                if usable {
                    let ms = time_ms(&stream, || {
                        let _ = run();
                    });
                    if ms < best.0 {
                        best = (ms, Some(rank));
                    }
                    if rank < 3 {
                        line.push_str(&format!(" r{rank}:{ms:.4}"));
                    }
                }
                rank += 1;
            }
            println!(
                "{line} | best r{} {:.4} ({:.2} TFLOPS) of {found}",
                best.1
                    .map_or_else(|| "-".to_owned(), |rank| rank.to_string()),
                best.0,
                flops / best.0 / 1e9
            );
        }
    }
}
