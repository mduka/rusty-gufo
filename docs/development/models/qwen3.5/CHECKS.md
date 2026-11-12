# Qwen3.5 (dense): checks

The dense Qwen3.5 models run on `gufo-qwen35`, the engine of Qwen3.8 27B;
the engine's own checks are in [Qwen3.8 27B checks](../qwen3.8-27b/CHECKS.md).
What is specific to the small models is their prefill attention, the
prefill and generation test against Gufo v0.28.1's C++ build and a
long-prompt recall check.

The prefill attention of the 16/4 and 8/2 heads runs on WMMA, the kernel of
the 27B's 24/4 heads instantiated for their head counts; C++ runs these
shapes through its baseline attention. The kernel test checks every
instantiated shape against an FP64 CPU reference (relative L2 error below
2e-3, the largest error below 3% of the output RMS; measured 2e-4 to 6.3e-4
and 1.4% from 512 to 32K visible positions), with every output row written,
and checks that the rows a call keeps from `first_query` equal the full
call's bits:

```sh
cargo test --profile gpu-test -p gufo-qwen-kernels --features gpu --test attention
```

The 24/4 kernels are the 27B's device code unchanged: compared function by
function with the code objects before the head counts became template
parameters, the instructions and resource metadata are identical.

```sh
cargo test --profile gpu-test -p gufo-qwen35 --features gpu,models --test prefill
```

`tests/prefill.rs` compares the Rust prefill and greedy generation with
fixtures recorded by `tests/reference/cpp_decode_dump.cpp`
(`tests/reference/record_prefill.py` writes one) on synthetic chat prompts of
19 to 5,089 tokens: the Q8_1 WMMA prefill, two chunks (4096 + 993 rows) with
the second attending to the first, and per checkpoint the head shapes (24/4,
16/4 and 8/2 heads of 256) and projection formats it carries (Q8_0, the
K-quants and IQ formats, BF16, tied LM heads). The Qwen3.8 27B matches bit
for bit. The Qwen3.5 checkpoints match the C++ tokens: every generated token,
and the greedy token of every step after each prompt but the near ties listed
in the test (`KNOWN_DIVERGENCES`), where C++'s token must be the runner-up
within 0.1 and the test goes on with it. One is listed: the 2B Q4_K_M after
the 353-token prompt picks "I" (40), 0.085 above C++'s "It" (2064); the eight
later steps match. The fixtures hold digests of the logits, not their values,
so the logits themselves are not compared. With `--nocapture` the test prints
the smallest margin between the two best logits of each case: 0.011 to 0.84
over the Qwen3.5 cases.

| Checkpoint | Repository at revision |
| --- | --- |
| Qwen3.5 0.8B Q8_0, BF16, IQ4_XS | `unsloth/Qwen3.5-0.8B-GGUF` `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` |
| Qwen3.5 2B Q4_K_M | `unsloth/Qwen3.5-2B-GGUF` `f6d5376be1edb4d416d56da11e5397a961aca8ae` |
| Qwen3.5 4B Q4_K_M | `unsloth/Qwen3.5-4B-GGUF` `e87f176479d0855a907a41277aca2f8ee7a09523` |
| Qwen3.5 9B Q4_K_M | `unsloth/Qwen3.5-9B-GGUF` `3885219b6810b007914f3a7950a8d1b469d598a5` |

The test reads each file from the Hugging Face cache at that revision and
fails without it.

Past 16,128 positions Gufo's C++ build gives no output. `tests/recall.rs`
puts a fact at the start of a chat prompt of about 23,000 tokens (five prefill
chunks) on the 4B Q4_K_M and asks for it at the end; the greedy answer must
name it:

```sh
cargo test --profile gpu-test -p gufo-qwen35 --features gpu --test recall
```
