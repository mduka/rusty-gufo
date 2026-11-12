# Qwen3.6 35B-A3B (`qwen35moe`)

The `qwen35moe` architecture: Qwen3.6 35B-A3B, hybrid Gated DeltaNet and
gated attention layers with 256 routed experts (8 active) and a gated shared
expert, native MTP and the DFlash2 draft. The package ports the 35B runner of
the C++ extension `slimsami/strycast-qwen35moe-gufo-extension` (commit
`4882b7a`, generated from Strycast `8c946372`, on Gufo main `0fb50ce4`);
weights, device state and the forward pass drive its HIP kernels ([`kernels/`](kernels/)) in the order and with the
arguments of its C++ runner. The chat template and the vision encoder come
from the shared [Qwen crate](../qwen/README.md). It started as a copy of the
dense [`qwen35`](../qwen35/README.md) package, whose layer structure it shares
apart from the feed-forward.

## Source ownership

| Path | Responsibility |
|---|---|
| [`src/config.rs`](src/config.rs), [`src/weights.rs`](src/weights.rs) | GGUF configuration with the expert hyperparameters, tensor binding and the role-specific format and shape checks. |
| [`src/engine.rs`](src/engine.rs) | The model and its sessions: decode, prefill, batched decode, verification, snapshots, hidden-row capture and the generation loops. |
| [`src/moe.rs`](src/moe.rs) | The routed and shared experts: router, top-k, expert GEMMs and their routes by row count. |
| [`src/decode.rs`](src/decode.rs), [`src/prefill.rs`](src/prefill.rs), [`src/batch.rs`](src/batch.rs), [`src/verify.rs`](src/verify.rs) | The launch chains of a decode step, a prefill chunk, a batched decode step and a verification pass. |
| [`src/arena.rs`](src/arena.rs), [`src/routes.rs`](src/routes.rs) | Scratch and session buffers; the pure per-layer route decisions. |
| [`src/persistent.rs`](src/persistent.rs) | Snapshot payloads for the disk continuation cache. |
| [`src/verifier.rs`](src/verifier.rs), [`src/speculative.rs`](src/speculative.rs) | The verifier over any draft (`DraftBackend`): serving, and MTP in every command (DFlash2, MTP, sampled proposals, batches); the benchmark's single-session DFlash2 verifier. |
| [`src/dflash/`](src/dflash/) | The DFlash2 draft: model, executor, block drafting and its adaptive policy. |
| [`src/mtp.rs`](src/mtp.rs), [`src/mtp/`](src/mtp/), [`src/pack.rs`](src/pack.rs) | The MTP head of the target GGUF (model, head, draft, `MtpBackend`) and the BF16/F32 weight packing it shares with DFlash2. |
| [`src/runner.rs`](src/runner.rs) | The serving runner (`gufo serve llm`), `QwenRunner<Draft>` (autoregressive or DFlash2) or `QwenRunner<MtpBackend>`. |
| [`tests/`](tests/), [`examples/`](examples/) | Parity tests against fixtures of the extension's C++ build ([checks](../../../docs/development/models/qwen3.6-35b-a3b/CHECKS.md)). |

## Commands

`gufo serve llm`, `gufo prompt`, `gufo chat` and `gufo bench` claim every
`qwen35moe` artifact, as the extension's provider claims `gguf:qwen35moe`
(`src/cli/src/serve.rs`, `src/cli/src/prompt/qwen35moe.rs`,
`src/cli/src/bench/qwen35moe.rs`). `prompt`, `chat` and `bench` run at the
extension's CLI context of 4,096 tokens, autoregressive, with
`--speculative dflash2 --dflash-model` or with `--speculative mtp
--mtp-model` naming the target GGUF itself; MTP runs on the serving
verifier (`Verifier<MtpBackend>`, adaptive length, no frontier logits).
`bench` measures one greedy request and refuses `-c` and sampled workloads
with the dense driver's messages, as the C++ driver does.

The loader refuses, before it copies anything, a file whose routed experts
have a format the ported kernels lack, naming the tensor and its format
(`weights::check_expert_formats`): gate/up must be one of Q4_K, Q5_K or
Q8_0, down one of Q4_K, Q5_K, Q6_K or Q8_0, the router and shared-expert
gate F32. UD-Q6_K_XL (Q6_K gate/up) and UD-Q8_K_XL (BF16 experts) need the
C++ fallback routes that are not ported.

## Chat template

Prompts follow the model's own template when the GGUF carries the pinned
Qwen3.6 one (`QWEN36_TEMPLATE_SHA256`, `Dialect::Qwen36` in `gufo-qwen`): no
reasoning-effort instruction, earlier turns' reasoning only with
`preserve_thinking: true`, an assistant's reasoning read from `<think>` in
its content when it carries none, Python spellings `True`, `False` and `None`
of boolean and null tool arguments. This deviates on purpose from the C++
extension, which renders every Qwen model with the compiled Qwen3.8
template; prompts with thinking off render the same in both. Reasoning
effort is accepted and has no prompt effect, since the template has no
slot for it; as on the other models an effort turns thinking on and `none`
turns it off, and no effort sets a reasoning budget. Other templates keep
the Qwen3.8 formatter, as in C++. More than two leading system or developer
messages, and later ones, are hoisted into the system turn as on the other
Qwen models, where the template keeps two and drops the rest.

## Serving runner

The runner is the dense one, generic over its draft (`ServeDraft`: DFlash2
or the MTP head), with the extension's differences:

- **Draft cap.** A routed-expert target reads mostly different experts per
  stacked verification row, so concurrent speculative sessions draft less
  (`ConcurrentDraftCap`): DFlash2 two tokens at two verifying sessions and one
  from three, MTP one and none. A seeded sampled request takes the cap of the
  configured sessions (`--sessions`), so it drafts the same at any load. The
  runner sets the cap on the session's draft before each verification
  (`Verifier::set_step_draft_cap`, the extension's `SetStepDraftCap`).
- **Wide copies.** A prompt-lookup copy of a request decoding alone verifies
  up to 64 rows (`DraftConfig::with_max_copy_rows`, the verifier's draft
  limit 63), while neural blocks and batched copies stay at
  `--draft-tokens`. Each session claims the larger SSM replay ring such a
  copy needs, and the width halves at load until the configured sessions
  fit the free device memory, down to 16 rows (`LimitCopyRows`). A
  verification that emitted more rows than the draft may leave pending
  (eight) cannot be rewound on cancellation.
- **Sampling defaults.** None: the extension's `QwenSamplingDefaults` gives
  presets to the 27B shape only, and `gufo_core::sampling::text_preset`
  returns `Unspecified` for this one. Requests and CLI options without
  sampling settings decode greedily, in both implementations. The GGUF's
  `general.sampling.*` keys are not read by either.
- **Context.** `--context 0` (the default) serves the native 262,144 tokens
  (`qwen35moe.context_length`), as the extension's `max_context == 0`.
- **Prefill budget.** A prefill step takes at most `min(budget, 4096)` tokens
  (`GetMaxPromptBatch`); the scheduler's `--prefill-chunk` (512) while others
  decode is the same in both servers. Chunks of up to eight rows run the
  decode-equivalent verification forward, longer ones the prompt forward.
- **Continuations.** The extension keeps its state in Gufo main's component
  cache (`cache::Slot`, prefill rollback, partial prefill retention,
  `-component-v2` identities). This runner keeps rusty-gufo's snapshot pool,
  disk payload and identity of the dense runner.

- **MTP.** `QwenRunner<MtpBackend>` serves the head of the target GGUF with
  the adaptive draft length and retained frontier logits, its own state ABI
  and identity lines, and the draft's `GMTPGPU0`/`GMTDRF01` payloads in the
  disk cache. After a disk restore the draft drafts with the image positions
  it saw last until the next prime or append: the trait's proposal,
  feedback and restore calls do not see the session.

Not ported yet:

- `--kv-cache q8_0`, the extension's 8-bit attention KV cache.
- The Ornith-1.5-122B template dialect and vision width (3072).
