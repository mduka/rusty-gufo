# Qwen3.6 35B-A3B: checks

Tests (`GUFO_QWEN35MOE_MODEL` names the UD-Q4_K_M target,
`GUFO_QWEN35MOE_Q8_MODEL` the Q8_0 one, `GUFO_QWEN35MOE_MMPROJ` the BF16
vision sidecar, `GUFO_QWEN35MOE_DFLASH_MODEL` the DFlash2 draft and
`GUFO_QWEN35MOE_FIXTURES` the local reference set; [testing](../../TESTING.md)):

```sh
cargo test --profile gpu-test -p gufo-qwen35moe-kernels --features gpu   # extents of every launcher
cargo test --profile gpu-test -p gufo-qwen35moe --features gpu           # the model against the C++ fixtures
cargo test --profile gpu-test -p gufo-qwen --features gpu,models --test vision_qwen35moe
```

## The reference

The reference is the C++ extension
[slimsami/strycast-qwen35moe-gufo-extension](https://github.com/slimsami/strycast-qwen35moe-gufo-extension)
at commit `4882b7a` (generated from Strycast `8c946372`), built into Gufo
main `0fb50ce4` with ROCm 10.1 and run on ROCm 10.1's runtime libraries. A
probe built from the extension records the fixtures: digests (FNV-1a over
the f32 bits) of full logit rows, tokens, and the drafted, accepted and
emitted tokens of every speculative step. Every mode repeats exactly across
two processes. The JSON fixtures are in `src/models/qwen35moe/tests/fixtures`;
the photos and a few full logit rows stay in the local reference set. The
probe, the patches that hook it and the vision gate below into the extension,
and the script that records the fixtures are in
[`tests/reference`](../../../../src/models/qwen35moe/tests/reference/README.md),
whose README gives the build and the recording.

The extension's C++ build refuses images for this model: its vision gate
admits the projection widths of the 27B and of Ornith only. For the image
fixtures a local three-line patch admits the width 2048 in the gate; the
encoder's arithmetic and weight formats are unchanged. The encoder sources
equal Gufo v0.28.1's, so rusty-gufo's shared Qwen encoder serves the 35B as
well, with 2048 added to its admitted widths.

## What matches the reference bit for bit

On UD-Q4_K_M and Q8_0, logits and tokens:

| Case | Coverage |
| --- | --- |
| Prefill | Prompts of 1 to 5,089 tokens across every chunk route (up to 8 rows, 128-row BF16 router, 256-row tile, 512 and 1,024 routed-expert thresholds, the 4,096-row chunk), incremental prefill at an offset, and 8 decode steps after each |
| Decode | A prompt fed token by token, then 140 steps across the split-K boundary |
| Sessions and snapshots | Continuation from a prefix; save, restore and compact serialization; a continuation equal to a cold prefill |
| Sampling | Three seeded GPU-sampled generations |
| Batched decode | Widths 2, 3 and 4 against the same sessions decoded alone |
| MTP | The full-head prompt hidden rows and 8 steps; 64 tokens through the verifier on four prompts; two sessions verified together under the concurrent draft cap |
| DFlash2 | The CLI configuration and the serving configuration (wide copies of up to 64 rows) on four prompts; two sessions under the draft cap |
| Images | Three images (synthetic, 1024×668, 960×966) and a two-image prompt: pixels, the 29 encoder stages, every embedding row, the prompt ids and positions, prefill logits, 32 greedy steps, and MTP and DFlash2 with images |

`tests/verify.rs` also checks verification chains of 16 to 64 rows and the
growing SSM replay ring against single-token decoding, including a decode
step captured and replayed after the ring grew.

The CLI and the server were compared with the extension's: `gufo prompt`
gives the same prompt ids and tokens (greedy, seeded sampled, MTP, DFlash2,
images), and `gufo serve llm` the same responses and draft counters for
buffered and streamed requests, tool calls, multi-turn conversations and
concurrent sessions. Prompts with thinking on differ on purpose: rusty-gufo
renders the model's own chat template, the extension Qwen3.8's.

A token-level plausibility check with llama.cpp (`llama-mtmd-cli`, greedy,
32 tokens) gives the same image descriptions as the extension for the
1024×668 photo on both files and for the 960×966 photo on UD-Q4_K_M.

## Not covered

The extension's 8-bit KV cache (`--kv-cache q8_0`) and Ornith 1.5 122B are
not ported. No C++ fixture covers sampled speculative proposals.
