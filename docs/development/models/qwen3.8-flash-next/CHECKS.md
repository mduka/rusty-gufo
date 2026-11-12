# Qwen3.8 Flash-Next: checks

The engine runs in Rust. Its session, batch, sampling, prefill, snapshot and
serving checks live in
[`src/models/qwen38_flash_next/tests`](../../../../src/models/qwen38_flash_next/tests)
(`quality.rs`, `serving.rs`, `snapshot.rs`; `GUFO_FLASH_NEXT_QUALITY=batch`,
`prefill`, `cache` or `replay` selects a part of `quality.rs`), its unit checks
(metadata, n-gram hashing and reads, the MTP policy and proposal sampling)
next to the code:

```sh
cargo test -p gufo-qwen38-flash-next --lib
cargo test --profile gpu-test -p gufo-qwen38-flash-next --features gpu \
  --test quality --test serving --test snapshot -- --ignored --nocapture
```

The independent MTP audit against the CPU formulas
(`qwen38_flash_next_gpu_probe --mtp-audit`), the deep AR/MTP execution check
(`--execution-only 133120`), the scalar reference probe and the C++ operator
checks of the kernels went with the C++ engine; Gufo's
[quality report](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen3.8-flash-next/QUALITY.md#reproduce) builds them with its
CMake presets, and scores its F32 reference fixtures with them.

The [vLLM](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/models/qwen4_exp/amd/mtp.py)
and [SGLang](https://github.com/sgl-project/sglang/blob/993d1fccbaafe3e79d91567d2fc1d665cc94fa50/python/sglang/srt/models/qwen4_exp_mtp.py)
formulas supply independent predictor checks; pinned Transformers ignores MTP
weights. [Vision checks](../qwen3.8-27b/CHECKS.md).

## Tools and artifacts

Model tests are in `src/models/qwen38_flash_next/tests`, focused microbenchmarks in
`tools/qwen-flash`. [Quality](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen3.8-flash-next/QUALITY.md) summarizes qualification and focused checks.
Build a microbenchmark with
`tools/bench/build.sh tools/qwen-flash/projection_plans.hip`;
`dense_blaslt_sweep.hip` times hipBLASLt on the dense prefill shapes.
No historical logit dump is required. New retained result summaries belong in
`artifacts/`; generated traces stay in the ignored top-level `artifacts/` tree.
