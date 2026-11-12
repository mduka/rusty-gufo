# Qwen3.5 (dense, 0.8B to 9B)

Qwen's small dense Qwen3.5 models (`gguf:qwen35`) for text, on the engine
that also runs [Qwen3.8 27B](../qwen3.8-27b/README.md). Unsloth's GGUFs of
the 0.8B, 2B, 4B and 9B load in Q8_0, BF16, the K-quants and the IQ formats;
the checkpoints below are checked.

Speed: [performance](../../PERFORMANCE.md#qwen35-dense) · Quality: the tokens
are Gufo's ([verification](#verification)).

## Models

| Model | Checked files (Unsloth) | Weights |
| --- | --- | --- |
| Qwen3.5 0.8B | [`Qwen3.5-0.8B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/tree/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0): Q8_0, BF16, IQ4_XS | 0.76 GiB (Q8_0) |
| Qwen3.5 2B | [`Qwen3.5-2B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/tree/f6d5376be1edb4d416d56da11e5397a961aca8ae): Q4_K_M | 1.19 GiB |
| Qwen3.5 4B | [`Qwen3.5-4B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/e87f176479d0855a907a41277aca2f8ee7a09523): Q4_K_M | 2.55 GiB |
| Qwen3.5 9B | [`Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5): Q4_K_M | 5.29 GiB |

The weights aren't part of this repository; Qwen publishes the models under
the Apache License 2.0.

## Use

```sh
hf download unsloth/Qwen3.5-4B-GGUF \
  --revision e87f176479d0855a907a41277aca2f8ee7a09523 \
  Qwen3.5-4B-Q4_K_M.gguf --local-dir models/qwen3.5
MODEL=models/qwen3.5/Qwen3.5-4B-Q4_K_M.gguf
./target/release/gufo chat --model "$MODEL"
./target/release/gufo serve llm --model "$MODEL"
```

Generation is autoregressive; the speculative modes of the 27B need its own
drafter. Thinking, reasoning controls, tool calls and sampling defaults
follow the model's chat template as on the 27B
([reasoning controls](../../SERVER.md#reasoning-controls)).

## Known limits

- Text only. The vision sidecars of the 0.8B and 2B carry another vision
  tower, which the Qwen vision encoder does not take.
- BF16 checkpoints prefill about eight times slower than Q8_0 (an exact
  GEMM keeps the decode's accumulation order); prefer a quantized file.
- Decoding is 0.2 to 0.4% slower than with Gufo's C++ build below the 9B
  ([experiments](../../development/models/qwen3.8-27b/EXPERIMENTS.md)).

## Verification

Prefill and greedy generation give the tokens of Gufo v0.28.1's C++ build on
the six checked files, for chat prompts of 19 to 5,089 tokens: every
generated token, and the greedy token of 269 of the 270 steps checked after
each prompt. The other is a near tie: on the 2B, after a 353-token prompt,
"I" leads "It" by 0.085 in the logits where C++ picks "It"; given "It", the
later steps match again. The logits differ from C++ in the last bits: the prefill attention runs on the matrix cores (WMMA, with FP16
queries and probabilities), as on the Qwen3.8 27B and the Qwen3.6 35B-A3B,
where C++ keeps these head shapes on its baseline attention. Past 16,128
positions Gufo's C++ build gives no output to compare; a recall check with a
fact at the start of a 23,000-token prompt on the 4B answers correctly. The
checks are listed in
[Qwen3.5 checks](../../development/models/qwen3.5/CHECKS.md).
