# Qwen3.6 35B-A3B

Qwen's Qwen3.6 35B-A3B (`gguf:qwen35moe`), a mixture of experts with 35B
parameters and 3B active per token: 40 hybrid Gated DeltaNet and gated
attention layers whose feed-forward sends each token to 8 of 256 experts
beside a shared expert, plus one native MTP layer. It reads text and images
and decodes autoregressively or with MTP or DFlash2 drafts.

rusty-gufo runs it with the HIP kernels of the
[Qwen3.6 35B-A3B extension for Gufo](https://github.com/slimsami/strycast-qwen35moe-gufo-extension)
(C++, commit `4882b7a`, generated from Sami Slimani's Strycast) and its own
host code; its outputs equal the extension's ([verification](#verification)).

Speed: [performance](../../PERFORMANCE.md#qwen36-35b-a3b) · Quality and
memory: Strycast's
[35B guide](https://github.com/slimsami/strycast/tree/main/docs/models/qwen3.6-35b-a3b).

## Files

| File | Repository | Size | Checked |
| --- | --- | --- | --- |
| `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/tree/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d) | 21.1 GiB | Yes |
| `Qwen3.6-35B-A3B-Q8_0.gguf` | the same | 35.2 GiB | Yes |
| `mmproj-BF16.gguf` (vision) | the same | 0.84 GiB | Yes |
| `Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf` (DFlash2 draft) | [`aminya/Qwen3.6-35B-A3B-DFlash2-GGUF`](https://huggingface.co/aminya/Qwen3.6-35B-A3B-DFlash2-GGUF/tree/1e26e3499d865b9d2fc55e019d65b3f0e8ab6846) | 0.53 GiB | Yes |

Both targets carry the MTP layer. The routed experts run on Q4_K, Q5_K and
Q8_0 gate and up projections and Q4_K, Q5_K, Q6_K and Q8_0 down projections;
quantizations with other expert formats, such as the repository's UD-Q6_K_XL
(Q6_K gate and up) and UD-Q8_K_XL (BF16 experts in one layer), are not
supported yet. The weights aren't part of this repository; Qwen publishes
the model, and the DFlash2 draft its authors, under the Apache License 2.0.

## Use

```sh
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
  --revision 5bc3e238d916f48a861bac2f8a1990a0e9b7e98d \
  Qwen3.6-35B-A3B-UD-Q4_K_M.gguf mmproj-BF16.gguf \
  --local-dir models/qwen3.6-35b-a3b
hf download aminya/Qwen3.6-35B-A3B-DFlash2-GGUF \
  --revision 1e26e3499d865b9d2fc55e019d65b3f0e8ab6846 \
  Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf --local-dir models/qwen3.6-35b-a3b
MODEL=models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
DRAFT=models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf
./target/release/gufo chat --model "$MODEL"
./target/release/gufo prompt --model "$MODEL" --image photo.png -p 'Describe it.'
./target/release/gufo serve llm --model "$MODEL" --speculative mtp --mtp-model "$MODEL"
./target/release/gufo serve llm --model "$MODEL" --speculative dflash2 --dflash-model "$DRAFT"
```

- MTP drafts with the target's own MTP layer, so `--mtp-model` names the
  target file. DFlash2 drafts with the separate draft. Greedy output with
  either draft equals greedy output without it.
- Concurrent speculative sessions draft less, because each verification row
  reads mostly different experts: DFlash2 two tokens at two verifying
  sessions and one from three, MTP one and then none. A request decoding
  alone may verify a prompt-lookup copy of up to 64 rows with DFlash2.
- Images: PNG, JPEG and WebP with the BF16 vision sidecar, found beside the
  model as `mmproj-BF16.gguf` or given with `--mmproj`. The F16 sidecar is
  not accepted.
- Prompts use the model's own chat template. Thinking is on unless a
  request turns it off (`--think off`, `"reasoning_effort": "none"` or
  `chat_template_kwargs.enable_thinking=false`); other effort levels turn it
  on but change nothing else, since the template has no effort setting.
  Earlier turns' reasoning stays in the history only with
  `chat_template_kwargs.preserve_thinking=true`. The extension renders the
  Qwen3.8 template instead, which adds "Reasoning effort is set to xhigh" by
  default, so prompts with thinking on differ from it on purpose.
- The context defaults to the native 262,144 tokens.
- Without sampling settings in the request, decoding is greedy, as in the
  extension.

## Known limits

- With thinking on, a follow-up turn processes the previous reply again
  instead of reusing it from the cache, because the template drops earlier
  reasoning from the history; `preserve_thinking=true` keeps it.
- The model often thinks for a long time; tight `max_tokens` limits can end
  a reply inside its reasoning.
- The extension's 8-bit KV cache (`--kv-cache q8_0`) is not available.
- Ornith 1.5 122B, which shares the architecture, is not supported.
- The extension's C++ build takes images for this model only with its vision
  gate opened to the projection width 2048 (a local three-line patch); the
  image results equal that build.

## Verification

Prefill, decode, sessions, snapshots, seeded sampling, batched sessions,
MTP, DFlash2 and image input reproduce the logits and tokens of the
extension's C++ build bit for bit on UD-Q4_K_M and Q8_0, and the CLI and
server give the same answers. The checks are listed in
[Qwen3.6 35B-A3B checks](../../development/models/qwen3.6-35b-a3b/CHECKS.md).
