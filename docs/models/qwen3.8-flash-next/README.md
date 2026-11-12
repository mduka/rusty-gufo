# Qwen3.8 Flash-Next

Hybrid recurrent/QSA mixture-of-experts text/image model on gfx1151.
Supported target: `unsloth/Qwen3.8-Flash-Next-GGUF`, **UD-Q4_K_XL** (four shards).
Optional shared-Q8 MTP predictor; optional BF16 vision projector.
Original unquantized-model and GGUF-conversion parity remain unqualified.

Speed: [performance](../../PERFORMANCE.md#qwen38-flash-next) · Quality:
[Gufo's quality report](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen3.8-flash-next/QUALITY.md),
which applies because the outputs are Gufo's ([verification](#verification)).

## Load and run

```sh
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
MODEL=models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
MTP=models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
./target/release/gufo chat --model "$MODEL"
./target/release/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

The loader discovers the remaining shards. Omit the speculative options for AR;
AR sessions allocate no predictor state even if a shared model has MTP loaded.
Adaptive MTP is the default, with `--draft-tokens` capping 1–7 proposals per
step. Besides the neural head, MTP decoding proposes copies of earlier text
of the session (prompt lookup), with one to eight sessions. The target
verifies every draft over its full vocabulary: greedy output with MTP is
byte-identical to autoregressive decoding, and sampled output keeps the
target distribution, although a seeded sampled stream differs from one
without drafts because the random draws fall at other points. Timings report
`lookup_n` / `lookup_n_accepted`; how the drafter chooses its vocabulary,
copies and widths is described in the
[MTP notes](../../development/models/qwen3.8-flash-next/MTP.md).

The official template defaults to thinking on, `xhigh` effort and preserving
prior reasoning. Use the [reasoning controls](../../SERVER.md#reasoning-controls)
for explicit effort/thinking overrides. Native context is 262144. A larger
context needs static YaRN and `--enable-yarn`: `gufo serve llm` refuses to
start with `--context` above 262144 without it, and `gufo bench`, which has no
`--enable-yarn`, stays within the native context. With the flag, the factor
is context / 262144 (`--context 409600` gives 1.5625).
The flag is an error at or below 262144, where it would have no effect.
Static YaRN applies to every request on that server, including short ones,
and changes the output distribution more as the factor grows; in the first
measurements, multiple-choice accuracy moved within noise at factors 1.5625
to 3.875. See the
[measurements](https://github.com/gufo-org/gufo/pull/350#issuecomment-6323981940),
the
[opt-in proposal](https://github.com/gufo-org/gufo/pull/350#issuecomment-6323982386)
and the
[accuracy check](https://github.com/gufo-org/gufo/pull/350#issuecomment-6234093998).
No GGUF key enables YaRN.
Prefill chunks whose visible context exceeds 262144 tokens use the per-token
attention kernel rather than the fused WMMA kernel. Memory grows with used
context and selected rollback depth (about 27.5 KiB per context token with
MTP, measured on gfx1151); admission reserves the configured capacity before
creating sessions.

## Images

Use this model's `mmproj-BF16.gguf`, discovered beside the target or selected
with `--mmproj`. PNG/JPEG CLI and HTTP requests use the
[same image interface](../qwen3.8-27b/README.md#images). Image state participates
in prefill, decoding, verification, multi-turn reuse and disk cache identity.
The predictor embeds shifted text IDs; visual information comes from target
hidden states and mRoPE. Image snapshots require matching prompt attachment.

## Verification

The session and forward tests match dumps of Gufo v0.28.1's C++ build, and
every workload of the
[A/B comparison](../../development/PARITY.md#performance-against-gufo)
(autoregressive, MTP, serving, sampled) gave the same outputs on both
builds. The checks are listed in
[Qwen3.8 Flash-Next checks](../../development/models/qwen3.8-flash-next/CHECKS.md).
