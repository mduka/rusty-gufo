# DeepSeek V4 Flash

Text generation on gfx1151 from the Flash 0731 mixed IQ2/Q2/Q8 GGUF.
Target weights occupy **80.76 GiB**; use a 128 GiB system and leave room for
request state. AR and optional DSpark support prompt, chat and HTTP serving.
Target-model parity remains open; see the quality limits before relying on
AR/DSpark agreement as a quality claim.

Speed: [performance](../../PERFORMANCE.md#deepseek-v4-flash) · Quality:
[Gufo's quality report](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/deepseek-v4-flash/QUALITY.md),
which applies because the outputs are Gufo's ([verification](#verification)).

## Load and run

```sh
hf download antirez/deepseek-v4-gguf \
  --revision 1cd7b564460821938add0475a60b942c409295e0 \
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --local-dir models/deepseek-v4-flash
hf download antirez/deepseek-v4-gguf \
  --revision e7f04037032990db0346398d249baf9fb9df1ccc \
  DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --local-dir models/deepseek-v4-flash
MODEL=models/deepseek-v4-flash/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
DSPARK=models/deepseek-v4-flash/DeepSeek-V4-Flash-DSpark-support-0731.gguf
./target/release/gufo chat --model "$MODEL"
./target/release/gufo serve llm --model "$MODEL" --context 32768 --sessions 1
```

Add `--dspark-model "$DSPARK"` to enable DSpark.
It verifies proposals against the target, commits only accepted prefixes and
keeps sampling/controller state private to each request. Speed depends on
acceptance and context; the support model is optional.

Thinking defaults to enabled with `high` effort, matching the
[DeepSeek API](https://api-docs.deepseek.com/guides/thinking_mode).
Use `--think off` for non-thinking chat or `--reasoning-effort low` for
lighter reasoning.

The server exposes OpenAI-compatible chat, streaming, tools, reasoning controls,
and RAM/disk continuation caches. See [server configuration](../../SERVER.md)
and [CLI options](../../CLI.md). Images are unsupported.

## Verification

On the pinned antirez/ds4 trajectory, rusty-gufo and Gufo v0.28.1's C++
build give identical logits, and seeded sampling over HTTP agrees in both.
Agreement with the official DeepSeek V4 Flash model remains open, as
Gufo's quality report describes. The checks are listed in
[DeepSeek V4 Flash checks](../../development/models/deepseek-v4-flash/CHECKS.md).
