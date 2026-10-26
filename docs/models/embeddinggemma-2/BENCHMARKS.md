# EmbeddingGemma 2 benchmarks

The Rust engine has not been measured yet: every row here is a **TODO** for it.
The numbers below were measured on 2026-10-23 with the C++ engine, whose HIP
kernels, launch shapes, row-chunked GEMMs and hipBLASLt timing the Rust engine
keeps unchanged.  Rerun the HTTP table with `target/release/gufo` from
`cargo build --release -p gufo` before quoting it for this build.

Those measurements ran on Linux with a Radeon 8060S GPU of architecture gfx1151, using BF16 safetensors and a resident model.  Each build ran its rows in two passes on one node, the only GPU pod on it.  Each row is 60 requests per pass, the table shows the mean of the two passes' medians, and token counts include `<bos>` and `<eos>`.  The 4 × 8 × 128 and 16 × 8 × 128 rows come from a second pod on the same node after a reboot, with three passes per build.

Model load took **2.2 s**.  About 0.53 s of that is the 540 MB upload, and the other 1.7 s is the warm-up that times each hipBLASLt algorithm for every GEMM shape and row bucket.

## HTTP

These rows time `gufo serve embedding` on loopback with [`http_bench.py`](../../../tools/models/embedding_gemma2/http_bench.py).  Each input is a run of distinct `word<N>` words.

```sh
cargo build --release -p gufo
./target/release/gufo serve embedding --model models/embeddinggemma-2 &
python3 tools/models/embedding_gemma2/http_bench.py \
  --base-url http://127.0.0.1:8080/v1 --model embeddinggemma-2
```

| Concurrency × inputs × words | Tokens/request | Median | p95 | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 1 × 1 × 16 | 40 | 6.69 ms | 7.54 ms | 5,805 |
| 1 × 1 × 128 | 404 | 20.45 ms | 21.49 ms | 19,507 |
| 1 × 1 × 512 | 1,940 | 147.96 ms | 151.34 ms | 13,101 |
| 1 × 8 × 128 | 3,495 | 149.16 ms | 150.31 ms | 23,371 |
| 1 × 32 × 128 | 15,670 | 708.58 ms | 711.57 ms | 22,088 |
| 4 × 1 × 128 | 404 | 78.93 ms | 80.81 ms | 20,337 |
| 16 × 1 × 128 | 404 | 313.23 ms | 317.89 ms | 20,462 |
| 4 × 8 × 128 | 3,495 | 584.30 ms | 588.65 ms | 23,896 |
| 16 × 8 × 128 | 3,495 | 2,342.86 ms | 2,346.70 ms | 23,840 |

Against hipBLAS-only GEMMs, the tuned hipBLASLt plans are 1.5 to 2.3 times faster at the median on every row of this table.  Concurrent requests share one device and take turns one batch at a time, so concurrency adds queueing time but not throughput.  A long request yields the device after each batch, so a short request queued behind it waits for one batch instead of the whole request.

## GEMM benchmark

The GEMM benchmark times each encoder GEMM shape through hipBLAS and through every hipBLASLt algorithm that needs no workspace.  These are kernel timings, not request latency.

```sh
cargo test --profile gpu-test -p gufo-embedding-gemma2 --features gpu \
  --test gemm_bench -- --ignored --nocapture
```

At 1,024 rows, four of the shapes compare as follows.

| Shape | hipBLAS | Best hipBLASLt |
| --- | ---: | ---: |
| `gate_up` | 0.88 ms, 4.86 TFLOPS | 0.16 ms, 27.01 TFLOPS |
| `down` | 0.46 ms, 4.69 TFLOPS | 0.08 ms, 25.60 TFLOPS |
| `ple_512` | 0.12 ms, 4.49 TFLOPS | 0.03 ms, 20.98 TFLOPS |
| `final` | 0.17 ms, 4.72 TFLOPS | 0.05 ms, 17.41 TFLOPS |

## Reference comparison

The llama.cpp comparison is TODO.  It requires a llama.cpp build with EmbeddingGemma 2 support and the matching GGUF.  Measure both over HTTP with the same texts and report the ratio per row.
