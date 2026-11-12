# Nemotron 3.5 Bolt 74B-A9B: checks

Tests (`GUFO_NEMOTRON_BOLT_GGUF` names the trunk; the drafter and projector
are found beside it by NVIDIA's file names; [testing](../../TESTING.md)):

```sh
cargo test -p gufo-nemotron                                     # CPU: config, tokenizer, template, vision preprocessing
cargo test --profile gpu-test -p gufo-nemotron --features gpu   # kernels against CPU evaluations
GUFO_NEMOTRON_BOLT_GGUF=$MODEL \
  cargo test --profile gpu-test -p gufo-nemotron --features gpu,models
```

The CPU tests compare the tokenizer and the chat template with Hugging Face's
(golden fixtures from `tools/`), and the vision preprocessing with NVIDIA's
processor. The kernel tests check every kernel against a CPU evaluation of its
formula, the batched-row kernels bit for bit against the one-row kernels and
the expert products on BF16 activations bit for bit against the staged ones.
The model tests check one-token decoding against prefill, snapshots, device
logit heads, batched decoding of two to eight sessions (bit-identical to
decoding them one by one), draft-and-verify rounds against greedy decoding,
the runner (also two requests drafting in turn), and the vision encoder
against NVIDIA's float32 model (relative RMS 0.0007 to 0.0008 in FP16
operands).

Against the C++ extension, `examples/runtime_probe.rs` writes what the
extension's `nemotron_runtime_probe` writes: the logits of every prompt
position and a continuation, greedy or sampled, with or without LiLiCorr
(`--verify-sweep` times rounds by their length). On the extension's parity
prompts the two builds give bit-identical logits, the same tokens and, with
the drafter, the same rounds:

| Case | Logits compared | Differ | Generated tokens |
| --- | ---: | ---: | --- |
| 28-token prompt, 32 greedy tokens | 3.7 million | 0 | identical |
| 141-token prompt, 64 greedy tokens | 18.5 million | 0 | identical |
| 28 tokens appended one by one | 3.7 million | 0 | — |
| 40-token code prompt, 96 tokens with LiLiCorr | 5.2 million | 0 | identical; 15 rounds, 81 of 95 drafts accepted in both |
| 28-token prose prompt with LiLiCorr, to its end | 3.7 million | 0 | identical (37 tokens); 7 rounds, 30 of 44 drafts |
| The same two prompts sampled with LiLiCorr (temperature 1, top-p 0.95, seed 1) | 8.9 million | 0 | identical; 29 rounds, 99 of 160 drafts (code), 7 rounds, 32 of 42 (prose) |

Every kernel the two builds share uses the same scratch memory
(`tools/ci/check-kernel-resources.py --compare`).

Speed, both builds' `gpu-test` probes in the order C++, Rust, Rust, C++
on an otherwise idle machine on the code prompt (`--generate 256
--verify-sweep 24`, greedy): LiLiCorr decoding 103.3 and 103.1 against 103.9
and 103.2 tokens/s; per call 27.9 against 28.0 ms for one token, 51.1
against 51.1 ms to verify eight, 10.8 against 10.7 to 11.0 ms for a draft.
Single runs of autoregressive decoding: 36.0 and 36.5 against 35.4 and
36.1 tokens/s; prefill is about 10 % slower ([experiments](EXPERIMENTS.md)).

The extension's quality report, benchmarks and tools are in
[gufo-extensions/nemotron](https://github.com/gufo-extensions/nemotron/tree/62f85dc).
