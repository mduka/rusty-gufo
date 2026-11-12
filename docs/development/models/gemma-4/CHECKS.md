# Gemma 4: checks

Tests (`GUFO_GEMMA4_MODEL` names the target, `GUFO_GEMMA4_MTP_MODEL` the
drafter, `GUFO_GEMMA4_MMPROJ` the vision sidecar; [testing](../../TESTING.md)):

```sh
cargo test -p gufo-gemma4                                          # CPU: config, tokenizer, template, draft policy, vision prompt
cargo test --profile gpu-test -p gufo-gemma4-kernels --features gpu # kernels against CPU evaluations
GUFO_GEMMA4_MODEL=$MODEL GUFO_GEMMA4_MTP_MODEL=$MTP \
  cargo test --profile gpu-test -p gufo-gemma4 --features gpu,models
GUFO_GEMMA4_MMPROJ=$MMPROJ cargo test --profile gpu-test -p gufo-gemma4 \
  --features gpu,models --test vision_encoder -- --ignored          # the released sidecar
```

The CPU tests compare the tokenizer with llama.cpp's and the chat template
with the Jinja template (27 golden renders, byte for byte). The kernel tests
check every kernel against a CPU evaluation within the extension's
tolerances, and the batch-invariant kernels bit for bit. The model tests
check the engine against the CPU reference (a port of the extension's,
bit-identical to it), multi-row forwards, prefix reuse and rewinds,
snapshots after the sliding ring wraps, lookahead, prefill weight copies,
MTP verification rows against single-token decoding, greedy MTP against
autoregressive decoding, siblings, batched against single sessions and MTP
cycles after a restore.

Against the C++ extension, `examples/gemma4_runtime_probe.rs` writes what a
probe built from the extension writes: the logits of every prompt row and a
continuation through `DecodeStep`. That probe is
[`tests/reference/runtime_probe.cpp`](../../../../src/models/gemma4/tests/reference/runtime_probe.cpp),
built in the extension's tree (`a31ce42`, on Gufo `0e2de43e`) with
`runtime-probe-hook.patch` as its header describes.
`examples/gemma4_vision_generate_probe.rs` does the same for the extension's
image probe: the resized pixels, a greedy
answer to an image question and the logits from the image's end on. On
Gemma 4 26B-A4B UD-Q4_K_M with its BF16 vision sidecar the two builds give
byte-identical logits and the same tokens:

| Case | Result |
| --- | --- |
| 27-token chat prompt, 64 greedy tokens | 7.1 million logits identical; same tokens |
| 4,114-token prompt past the sliding window and ring, 32 greedy tokens | 1.08 billion logits identical; same tokens |
| Prompt appended one token at a time; forwards of 7, 9 and 16 rows | identical |
| Q8_0 file: chat prompt, 48 greedy tokens; 9 rows | identical |
| MTP greedy, policies `fixed`, `calibrated` and `confidence` | same tokens and drafting statistics |
| Sampled (temperature 1, seed 7), without MTP and with MTP at top-k 0 and 64 | same tokens |
| Repetitive prompt (prompt lookup) | same tokens; 81 of 82 copies accepted in both |
| Image question, 64 greedy tokens: a 512×512 RGBA PNG (768×768, 256 soft tokens) and a 2048×1152 PNG (1056×576, 264 soft tokens) | same resized pixels and tokens; 21 million logits identical per image |

The vision encoder gives the extension encoder's embeddings bit for bit,
on the released sidecar and on a synthetic one with random weights. Every kernel the two builds share uses the same scratch memory
(`tools/ci/check-kernel-resources.py --compare`).

Speed in single runs, not a clean A/B measurement: autoregressive decoding
57.0 against 57.6 tokens/s, MTP decoding 137.6 against 137.4 tokens/s, a
4,114-token prefill 1.22 s against 1.21 s; the first prompt of a process is
about 20 ms slower and the first image encode about 40 ms slower
([experiments](EXPERIMENTS.md)).
