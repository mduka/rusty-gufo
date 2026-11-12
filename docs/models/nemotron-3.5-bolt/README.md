# Nemotron 3.5 Bolt 74B-A9B

NVIDIA's Nemotron 3.5 Bolt is a hybrid Nemotron-H model (`gguf:nemotron_h_moe`,
88 blocks): Mamba-2 blocks, attention blocks without positional encoding and
LatentMoE blocks with 496 routed ReLU² experts plus a shared expert. NVIDIA's
24 GB GGUF keeps the experts in IQ2_XXS, the Mamba projections in NVFP4 and the
rest in Q8_0. The optional LiLiCorr drafter drafts blocks of up to seven
tokens, and the optional vision projector (C-RADIO v4-H) takes images.

rusty-gufo runs it with the HIP kernels of the
[Nemotron extension for Gufo](https://github.com/gufo-extensions/nemotron)
(C++, commit `62f85dc`, version 0.2.3), including its llama.cpp-derived
quantized matrix kernels, and its own host code; its outputs equal the
extension's ([verification](#verification)). Not carried over: 0.2.3's
prompt passes in layer groups beside decoding sessions
([known limits](#known-limits)).

Speed: [performance](../../PERFORMANCE.md#nemotron-35-bolt-74b-a9b) · Quality:
[the extension's quality report](https://github.com/gufo-extensions/nemotron/blob/62f85dc/docs/models/nemotron-3.5-bolt/QUALITY.md).

## Model

| Model ID | Files | Modes |
| --- | --- | --- |
| `nemotron-3.5-bolt` | NVIDIA's [`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/tree/e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246); optionally the [LiLiCorr drafter](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF/tree/182d2987939e12f458298b5d5f6ef98912f4ad6c) and the vision projector `mmproj-…gguf` beside it | Text, images, autoregressive or LiLiCorr drafting |

The weights aren't part of this repository. NVIDIA publishes them under its
own license, shown on the model pages linked above; read it before you
download them.

## Use

```sh
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF \
  --revision e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246 \
  NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
  mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
  --local-dir models/nemotron-3.5-bolt
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF \
  --revision 182d2987939e12f458298b5d5f6ef98912f4ad6c \
  NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf \
  --local-dir models/nemotron-3.5-bolt
MODEL=models/nemotron-3.5-bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
DRAFT=models/nemotron-3.5-bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf
PROJECTOR=models/nemotron-3.5-bolt/mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
./target/release/gufo prompt --model "$MODEL" --prompt 'What is the capital of France?' --think off
./target/release/gufo serve llm --model "$MODEL" --mmproj "$PROJECTOR" \
  --speculative dflash2 --dflash-model "$DRAFT"
```

- Thinking is on by default and its text arrives as `reasoning_content`;
  `reasoning_effort: "none"` turns it off, `"minimal"`, `"low"` and
  `"medium"` select the template's efficient mode.
- Tool calls use the template's Qwen3-Coder XML format.
- Sampling defaults to temperature 1.0 and top-p 0.95, as NVIDIA recommends.
  For top-k sampling the GPU summarizes each row, so the host sorts a few
  candidates instead of the vocabulary.
- NVIDIA's LiLiCorr drafter is a DFlash drafter, so it loads with
  `--speculative dflash2 --dflash-model`. Each round drafts up to
  `--draft-tokens` (at most 7, the default) tokens and the trunk verifies
  them in one pass, up to where the chance of reaching a draft falls below
  15 % (at least one draft): the drafter's lattice gives that chance, scaled
  by how the request's text has accepted its uncertain drafts so far. A
  verified row equals decoding its token alone, so greedy output with the
  drafter equals greedy output without it.
- Concurrent requests decode together, one token each, in batches of up to
  eight. After the first 16 tokens of their answers, two or three requests
  draft one round each in turn instead while their measured rounds emit
  tokens faster than a shared pass (code and JSON, rarely prose). A request
  decoding alone drafts.
- Conversations continue from the RAM cache through host snapshots of the
  attention rows, Mamba states and the drafter's rings. There is no disk
  cache for this model yet.
- With `--mmproj` the server takes PNG, JPEG and WebP images in user and tool
  messages; `prompt` and `chat` put `--image` files at the start of the first
  user turn. Images are prepared as NVIDIA's processor prepares them (1,024 to
  13,312 patches of 16x16 pixels, 2x2 patches per token, `detail: "low"` caps
  an image at 256 tokens) and encoded when prefill reaches them.

## Known limits

- Images only: the API has no video input, and the projector has no audio
  encoder.
- A prompt pass beside decoding sessions runs whole (up to 2,048 tokens,
  about 2 s), and so does the image encoder, so other sessions wait for it.
  The extension splits such a pass into layer groups since 0.2.3, which
  needs a scheduler feature rusty-gufo does not have yet.
- The extension's [quality report](https://github.com/gufo-extensions/nemotron/blob/62f85dc/docs/models/nemotron-3.5-bolt/QUALITY.md)
  lists which of Gufo's functional API suites fail for reasons of the model or
  its template (history reasoning, the reasoning-effort hint, forced tool
  calls at effort `high`).
- Prompts prefill about 10 % slower than with the C++ extension
  ([experiments](../../development/models/nemotron-3.5-bolt/EXPERIMENTS.md)).

## Verification

rusty-gufo gives bit-identical logits, the same tokens and the same drafting
statistics as the C++ extension it comes from (gufo-extensions/nemotron
`62f85dc` on Gufo `0e2de43e`), greedy and sampled, with and without
LiLiCorr. The checks are listed in
[Nemotron 3.5 Bolt 74B-A9B checks](../../development/models/nemotron-3.5-bolt/CHECKS.md).
