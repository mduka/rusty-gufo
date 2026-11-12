# Server internals

How `gufo serve` is built inside and the rules its code follows. How to run
and use the server is in the user guide, [server](../SERVER.md).

## Rules

Diagnostic telemetry is limited to status, token counts, timing, and
cancellation state. It must not contain prompt text, model paths, machine
identity, request IDs, or token IDs.

The adapter must not implement a second inference path. It converts messages
into the same prompt and sampling structures used by `/v1/responses`.

## Transport

The Rust runtime uses bounded HTTP/1.1 requests, one connection worker per
request, and a separate inference scheduler. Connections close after each
response. Chat Completions and Responses support Server-Sent Events; socket closure
propagates cancellation to the scheduler. JSON parsing rejects duplicate keys,
invalid numbers and nesting beyond 128 containers.

Python may be used for development tools and API conformance tests, but it must
not be required to serve requests.

## Checks of the sampling defaults

The [functional suite](../../tests/functional/README.md) with
`--suite sampling-defaults --sampling-preset qwen38` (or
`deepseek4`) compares omitted and explicit settings, including C2 replay.

## Cancellation and continuation check

For a focused cancellation check, run
`python3 tests/functional/continuation.py --output /tmp/cache-check.json`
against a private server named `cache-test` on port 5815.
It checks interruption during reasoning and visible output, with and without
reasoning replay, and greedy/seeded sampling. Use
`--tools --discard-assistant` to exercise interrupted agent tool turns; add
`--prefix-repetitions 5500` for a roughly 50K-token prefix.
For persistence, enable `--cache-disk` before the check, restart the same server,
and add `--restore /tmp/cache-check.json`.
Use `--image /path/to/image.png` for Qwen image conversations.
Add `--append-image` to introduce the image after a cached text turn, and
`--reasoning-effort high` to check a specific thinking effort.
Each case continues for a third turn; repeat `--case NAME` to select only the
cases needed for a change.
The check requires exact snapshot and matched-history replay in memory. After a
restart, sampled output may vary when a disk restore re-prefills a gap; greedy
and zero-prefill restores still require equality. The report separates successful
validation (`status`) from observed assistant-message equality (`exact`) and its
requirement (`exact_required`). The SDK conversation suite checks cache bypass.

## Process and state ownership

HTTP workers submit requests through `TextGenerationBackend`. The text
scheduler owns request state and serializes model execution. Model runners own
weights, tokenization, prefill/decode, speculative verification and complete
cache snapshots. HTTP handlers do not implement model kernels.

## Internal request model

Text endpoints use `ChatRequest` and `SamplingConfig` at the
[backend boundary](../../src/cli/serve/src/backend.rs).
Chat messages retain roles, reasoning, tool calls and image content until the
model's tokenizer and template turn them into model input. The scheduler owns
request limits, cancellation, cache accounting and completion state.

## Focused checks

- `cargo test -p gufo-core --lib json`: number precision, Unicode escapes,
  malformed input and depth limits.
- `cargo test -p gufo-serve --test http_server --test http_logging`: transport
  framing, authentication, compatibility validation, sampling forwarding,
  request logs at the default and debug tiers, and streaming failures without
  loading a model.
- `cargo test -p gufo --test serve` and
  `python3 tests/cli/serve_test.py target/gpu-test/gufo`: executable help,
  argument wiring and rejected configurations, including that `--log-level`
  and `-v` change the emitted log rather than only parsing.
- `cargo test -p gufo-serve --test openai_chat`: chat parsing, streaming,
  images, tools and sampling controls.
- Per-model serving tests: greedy/sampled decoding, batching, cache reuse and
  cancellation. Use the affected model's benchmark README for commands and limits.

For session-boundary changes, also run a six-turn conversation: remember a fact,
force one token-limit stop, recall the fact, update it and recall the update.
Compare greedy AR/speculative token traces and repeat each sampled mode with the
same seed. Mix EOS and token-limit stops rather than testing only fixed-length
generation. This is a session/replay check, not a capability or distribution test.

2026-09-19: six-turn greedy AR/speculative traces matched for DS4/DSpark,
Flash-Next/MTP and Qwen27B Q4/Q8 with the Q4 DFlash2 draft. Speculative runs
repeated exactly at temperature 0.6, seed 7, with a 32-token turn limit.
Two simultaneous Qwen27B Q4/DFlash2 HTTP conversations also retained independent
facts through six turns, with streaming and memory-cache reuse.

DSpark's short boundary check is available without the complete serving suite:

```sh
# Set GUFO_DEEPSEEK_V4_FLASH_MODEL and GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL first.
cargo test --profile gpu-test -p gufo-deepseek-v4-flash --features gpu,models \
  --lib eos -- --ignored
```
