# Qwen3.5 (dense): checks

The dense Qwen3.5 models run on `gufo-qwen35`, the engine of Qwen3.8 27B;
the engine's own checks are in [Qwen3.8 27B checks](../qwen3.8-27b/CHECKS.md).
What is specific to the small models is the prefill and generation test
against Gufo v0.28.1's C++ build:

```sh
cargo test --profile gpu-test -p gufo-qwen35 --features gpu,models --test prefill
```

`tests/prefill.rs` compares the Rust prefill logits and greedy generation
with fixtures recorded by `tests/reference/cpp_decode_dump.cpp`
(`tests/reference/record_prefill.py` writes one), bit for bit, on synthetic
chat prompts of 19 to 5,089 tokens: the Q8_1 WMMA prefill, two chunks
(4096 + 993 rows) with the second attending to the first, and per checkpoint
the head shapes (24/4, 16/4 and 8/2 heads of 256) and projection formats it
carries (Q8_0, the K-quants and IQ formats, BF16, tied LM heads):

| Checkpoint | Repository at revision |
| --- | --- |
| Qwen3.5 0.8B Q8_0, BF16, IQ4_XS | `unsloth/Qwen3.5-0.8B-GGUF` `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` |
| Qwen3.5 2B Q4_K_M | `unsloth/Qwen3.5-2B-GGUF` `f6d5376be1edb4d416d56da11e5397a961aca8ae` |
| Qwen3.5 4B Q4_K_M | `unsloth/Qwen3.5-4B-GGUF` `e87f176479d0855a907a41277aca2f8ee7a09523` |
| Qwen3.5 9B Q4_K_M | `unsloth/Qwen3.5-9B-GGUF` `3885219b6810b007914f3a7950a8d1b469d598a5` |

The test reads each file from the Hugging Face cache at that revision and
fails without it.
