# Models: development

Per model, what is checked against its C++ reference and what has been tried
to make it faster. How to use a model is on its page under
[docs/models](../../models/README.md).

| Model | Package | Reference | Checks | Experiments |
| --- | --- | --- | --- | --- |
| Qwen3.8 27B | `src/models/qwen35` | Gufo v0.28.1 | [checks](qwen3.8-27b/CHECKS.md) | [experiments](qwen3.8-27b/EXPERIMENTS.md) |
| Qwen3.5 (dense) | `src/models/qwen35` | Gufo v0.28.1 | [checks](qwen3.5/CHECKS.md) | [experiments](qwen3.5/EXPERIMENTS.md) |
| Qwen3.6 35B-A3B | `src/models/qwen35moe` | strycast-qwen35moe-gufo-extension `4882b7a` | [checks](qwen3.6-35b-a3b/CHECKS.md) | [experiments](qwen3.6-35b-a3b/EXPERIMENTS.md) |
| Qwen3.8 Flash-Next | `src/models/qwen38_flash_next` | Gufo v0.28.1 | [checks](qwen3.8-flash-next/CHECKS.md), [MTP notes](qwen3.8-flash-next/MTP.md) | [experiments](qwen3.8-flash-next/EXPERIMENTS.md) |
| DeepSeek V4 Flash | `src/models/deepseek_v4_flash` | Gufo v0.28.1 | [checks](deepseek-v4-flash/CHECKS.md) | [experiments](deepseek-v4-flash/EXPERIMENTS.md) |
| Gemma 4 | `src/models/gemma4` | gemma4-gufo-extension `a31ce42` | [checks](gemma-4/CHECKS.md) | [experiments](gemma-4/EXPERIMENTS.md) |
| Nemotron 3.5 Bolt | `src/models/nemotron` | gufo-extensions/nemotron `62f85dc` | [checks](nemotron-3.5-bolt/CHECKS.md) | [experiments](nemotron-3.5-bolt/EXPERIMENTS.md) |
| Qwen3-ASR 1.7B | `src/models/qwen3_asr` | Gufo v0.28.1 | [checks](qwen3-asr/CHECKS.md) | [experiments](qwen3-asr/EXPERIMENTS.md) |
| Qwen3-TTS 1.7B | `src/models/qwen3_tts` | Gufo v0.28.1 | [checks](qwen3-tts/CHECKS.md) | [experiments](qwen3-tts/EXPERIMENTS.md) |
| OmniVoice | `src/models/omnivoice` | gufo-extensions/omnivoice `4bf376c` | [checks](omnivoice/CHECKS.md) | [experiments](omnivoice/EXPERIMENTS.md) |
| VoxCPM2 | `src/models/voxcpm` | gufo-extensions/voxcpm `0941cd1` | [checks](voxcpm2/CHECKS.md) | [experiments](voxcpm2/EXPERIMENTS.md) |
| Nemotron 3.5 ASR Streaming 0.6B | `src/models/nemotron_asr` | gufo-extensions/nemotron_asr `c256b69` (0.1.2) | [checks](nemotron-3.5-asr/CHECKS.md) | [experiments](nemotron-3.5-asr/EXPERIMENTS.md) |
| EmbeddingGemma 2 | `src/models/embedding_gemma2` | Gufo v0.28.1 | [checks](embeddinggemma-2/CHECKS.md) | [experiments](embeddinggemma-2/EXPERIMENTS.md) |
| Qwen-Image-2.1 | `src/models/qwen_image_21` | Gufo v0.28.1 | [checks](qwen-image-2.1/CHECKS.md) | [experiments](qwen-image-2.1/EXPERIMENTS.md) |

The extensions are built on the Gufo revisions [relation to Gufo](../PARITY.md)
names; it explains the references and the equivalence rule.

- `CHECKS.md`: the tests and comparisons that tie the model to its reference,
  how to run them, their results and the open gaps.
- `EXPERIMENTS.md`: what has been tried for the model, above all what did not
  help, so that later optimization rounds do not try it again; ideas not
  tried yet come last. The core and what spans models are in
  [optimizations](../OPTIMIZATIONS.md).

Gufo's own benchmark, quality and experiment records measured its C++ engine;
they stay in Gufo's repository at v0.28.1 (and in the extensions'
repositories) and are linked from the pages here. Measurements of rusty-gufo
go to [performance](../../PERFORMANCE.md) ([benchmarking](../BENCHMARKING.md)).
Gufo's records of its component cache (`src/cache`) describe the C++ engine;
rusty-gufo keeps its own continuation cache
([differences](../PARITY.md#implemented-differently)).

Keep executable fixtures with their tests. New logit/trace dumps and local
profiles belong in the ignored top-level `artifacts/`; commit only
independent reference evidence a maintained check needs.

Checks distinguish agreement with an independent model implementation from
consistency between execution modes. Greedy speculative decoding reproduces
autoregressive decoding; sampled speculation can use different random draws.
Seeded replay is checked within the same execution configuration. For
numerical comparisons, lower KL divergence, total variation, relative L2 and
word-error rate are better; higher cosine similarity and image PSNR are
better. These bounded checks are not broad capability or perceptual scores.
