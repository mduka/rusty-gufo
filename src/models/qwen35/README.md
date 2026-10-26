# Qwen3.5 architecture (`qwen35`)

The dense Qwen3.8 27B and Qwen3.5 runtime owns target inference, native MTP
and DFlash2. The chat template, vision encoder and the HIP kernels come from
the shared [Qwen crate](../qwen/README.md), which Flash-Next uses as well. The
`qwen35moe` checkpoints (Qwen3.5 and 3.6 mixture of experts) share these
layers apart from the feed-forward and belong here once supported.

[27B usage](../../../docs/models/qwen3.8-27b/README.md) ·
[27B quality](../../../docs/models/qwen3.8-27b/QUALITY.md)

## Source ownership

| Path | Responsibility |
|---|---|
| [`src/config.rs`](src/config.rs), [`src/weights.rs`](src/weights.rs) | GGUF configuration, tensor binding and the role-specific format and shape checks. |
| [`src/engine.rs`](src/engine.rs) | The model and its sessions: decode, prefill, batched decode, verification, snapshots, hidden-row capture and the generation loops. |
| [`src/decode.rs`](src/decode.rs), [`src/prefill.rs`](src/prefill.rs), [`src/batch.rs`](src/batch.rs), [`src/verify.rs`](src/verify.rs) | The launch chains of a decode step, a prefill chunk, a batched decode step and a verification pass. |
| [`src/arena.rs`](src/arena.rs), [`src/routes.rs`](src/routes.rs) | Scratch and session buffers; the pure per-layer route decisions. |
| [`src/persistent.rs`](src/persistent.rs) | Snapshot payloads for the disk continuation cache. |
| [`src/speculative.rs`](src/speculative.rs), [`src/verifier.rs`](src/verifier.rs) | Single-session speculative decoding (bench, prompt with MTP) and the serving verifier (DFlash2, sampled proposals, batches). |
| [`src/dflash/`](src/dflash/) | The DFlash2 draft: model, executor, block drafting and its adaptive policy. |
| [`src/mtp.rs`](src/mtp.rs), [`src/pack.rs`](src/pack.rs) | The MTP head (model, executor, draft) and the BF16/F32 weight packing it shares with DFlash2. |
| [`src/runner.rs`](src/runner.rs) | The serving runner. |
| [`tests/`](tests/), [`examples/`](examples/) | Parity tests against the C++ fixtures; [`tests/reference/`](tests/reference/) holds the C++ dump programs that recorded them and [`examples/`](examples/) the Rust side. |

The HIP sources stay at upstream's paths under [`../qwen/hip/`](../qwen/hip/);
the kernel crate [`../qwen/kernels/`](../qwen/kernels/) declares their
launchers.

## State and arithmetic

Execution mode is explicit per session. Each request owns positions, KV and
recurrent/convolution state, RNG, proposals and acceptance/controller feedback.
Batch projection work may be shared; state and reduction order must remain
independent. Snapshots bind complete model/projector identities and sampling mode.

Use the model quality runners before changing arithmetic, layouts or cache
boundaries. Production routes compile for gfx1151. GPU correctness builds
(`gpu-test`) are optimized with assertions; release binaries provide speed
results.
Pinned upstream source/license records remain beside imported kernels and under
[`../qwen/reference/`](../qwen/reference/). Do not add alternate runtime routes
without measured benefit.
