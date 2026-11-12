# Gemma 4

Google's Gemma 4 instruction-tuned models (`gguf:gemma4`) in Unsloth's GGUFs:
the dense 31B, its quantization-aware-trained 31B QAT (Q4_0) and the
26B-A4B mixture of experts. One engine serves all three, with the
`gemma4-assistant` MTP drafter, prompt lookup, the BF16 vision sidecar,
batched sessions, native tool calls and the shared serving cache.

rusty-gufo runs them with the HIP kernels of
[Francesco Moca's Gemma 4 extension for Gufo](https://github.com/franzmoca/gemma4-gufo-extension)
(C++, commit `a31ce42`) and its own host code; its outputs equal the
extension's ([verification](#verification)). Only the 26B-A4B has been
checked.

Speed: [performance](../../PERFORMANCE.md#gemma-4) · Quality: the extension's
model cards below.

## Models

| Model | Files | Checked |
| --- | --- | --- |
| Gemma 4 26B-A4B | `unsloth/gemma-4-26B-A4B-it-GGUF`: UD-Q4_K_XL, UD-Q6_K_XL, UD-Q8_K_XL (UD-Q4_K_M and Q8_0 use the same formats); `mtp-gemma-4-26B-A4B-it.gguf`; `mmproj-BF16.gguf` | UD-Q4_K_M and Q8_0, with the Q8_0 drafter; images with `mmproj-BF16.gguf` |
| Gemma 4 31B | `unsloth/gemma-4-31B-it-GGUF`: UD-Q4_K_XL, UD-Q8_K_XL; `MTP/mtp-gemma-4-31B-it-Q8_0.gguf`; `mmproj-BF16.gguf` | No |
| Gemma 4 31B QAT | `unsloth/gemma-4-31B-it-qat-GGUF`: UD-Q4_K_XL (all Q4_0); its own Q4_0 drafter and vision sidecar | No |

The weights aren't part of this repository. Google publishes Gemma 4 under
the [Apache License 2.0](https://ai.google.dev/gemma/docs/gemma_4_license).
The extension's model cards describe each model's
architecture, quantization formats, benchmarks and quality:
[26B-A4B](https://github.com/franzmoca/gemma4-gufo-extension/blob/a31ce42/docs/models/gemma-4-26b-a4b/README.md),
[31B](https://github.com/franzmoca/gemma4-gufo-extension/blob/a31ce42/docs/models/gemma-4-31b/README.md),
[31B QAT](https://github.com/franzmoca/gemma4-gufo-extension/blob/a31ce42/docs/models/gemma-4-31b-qat/README.md).

## Use

```sh
hf download unsloth/gemma-4-26B-A4B-it-GGUF \
  --revision c099eb48e663fd284577b04978a94ffccb261841 \
  gemma-4-26B-A4B-it-UD-Q4_K_M.gguf mtp-gemma-4-26B-A4B-it.gguf \
  mmproj-BF16.gguf \
  --local-dir models/gemma-4-26b-a4b
MODEL=models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
MTP=models/gemma-4-26b-a4b/mtp-gemma-4-26B-A4B-it.gguf
./target/release/gufo chat --model "$MODEL"
./target/release/gufo prompt --model "$MODEL" --image photo.jpg -p 'Describe it.'
./target/release/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --context 131072
```

- Thinking arrives as `reasoning_content`; the model writes it in its
  `<|channel>thought` channel. Reasoning budgets count from the thought header
  on. Tool calls use Gemma's native `<|tool_call>call:NAME{…}<tool_call|>`
  syntax, buffered and streamed, and constrained calls follow it too.
- Sampling defaults to temperature 1.0, top-k 64 and top-p 0.95, Google's
  generation config.
- `--speculative mtp --mtp-model` loads the drafter. `-d`/`--draft-tokens`
  caps the drafts per cycle (1 to 15, default 15) and `--draft-policy`
  decides how many a cycle verifies: `calibrated` (default) verifies a draft
  while its estimated acceptance is worth one more verification row,
  `confidence` stops below a fixed confidence floor, and `fixed` always
  drafts up to the cap. `--min-draft-tokens` sets a floor,
  `--draft-calibration request` restarts the calibration with every request
  (an exact seeded replay), and `--draft-trace PATH` appends one JSON line per
  verification cycle. Prompt lookup copies earlier prompt text as drafts, and
  siblings verify the drafter's second choice beside its chain. Every
  verification row equals decoding its token alone, so greedy output with
  the drafter equals greedy output without it.
- Concurrent requests decode together in one verification forward of up to
  16 rows.
- `--prefill-weights compact|decoded` keeps prefill copies of the
  projections: more memory for faster prefill, with identical results.
- Images: PNG, JPEG and WebP with the vision sidecar, found beside the model
  as `mmproj-BF16.gguf` or given with `--mmproj`. `--image-tokens` sets the
  soft tokens per image (70, 140, 280 (default), 560 or 1120). An image is
  evaluated whole: prefill steps stop before it or take all of it.
- Conversations continue from the RAM cache, and from the disk cache with
  `--cache-disk`. Snapshots share the position blocks of one session's
  history, so a snapshot copies only what changed since the last one.

## Known limits

- Unchecked: the 31B and 31B QAT models and their vision sidecars.
- Constrained tool calls accept object members in sorted order only. The
  extension, built on a newer Gufo, also accepts llama.cpp's required-first
  order, so such calls can differ from the C++ extension's.
- The extension's [known functional failures](https://github.com/franzmoca/gemma4-gufo-extension/blob/a31ce42/README.md#known-limits)
  apply.


## Verification

Gemma 4 26B-A4B gives byte-identical logits and the same tokens as the C++
extension it comes from (gemma4-gufo-extension `a31ce42` on Gufo
`0e2de43e`): autoregressive, with MTP under every draft policy, sampled, with
prompt lookup and for image questions. The 31B and 31B QAT models are not
checked. The checks are listed in
[Gemma 4 checks](../../development/models/gemma-4/CHECKS.md).
