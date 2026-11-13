# Models

Every model runs on Linux with an AMD Strix Halo (`gfx1151`) and ROCm 10.1.
The weights are not part of rusty-gufo: download them as each page describes
and mind their licenses.

| Model | Files | Inputs and outputs | Page |
| --- | --- | --- | --- |
| Qwen3.8 27B | Unsloth's Q4/Q8 GGUF, DFlash2 drafter, vision sidecar | Text, images → text | [Qwen3.8 27B](qwen3.8-27b/README.md) |
| Qwen3.5 0.8B to 9B (dense) | Unsloth's GGUFs | Text → text | [Qwen3.5](qwen3.5/README.md) |
| Qwen3.6 35B-A3B | Unsloth's Q4/Q8 GGUF with MTP layer, DFlash2 draft, vision sidecar | Text, images → text | [Qwen3.6 35B-A3B](qwen3.6-35b-a3b/README.md) |
| Qwen3.8 Flash-Next | Unsloth's sharded Q4 GGUF, MTP sidecar, vision sidecar | Text, images → text | [Qwen3.8 Flash-Next](qwen3.8-flash-next/README.md) |
| DeepSeek V4 Flash | antirez's Flash 0731 mixed IQ2/Q2/Q8 GGUF, DSpark support GGUF | Text → text | [DeepSeek V4 Flash](deepseek-v4-flash/README.md) |
| Gemma 4 (26B-A4B, 31B, 31B QAT) | Unsloth's GGUFs, MTP drafter, vision sidecar | Text, images → text | [Gemma 4](gemma-4/README.md) |
| Nemotron 3.5 Bolt 74B-A9B | NVIDIA's 24 GB GGUF (IQ2_XXS, NVFP4, Q8_0), LiLiCorr drafter, vision projector | Text, images → text | [Nemotron 3.5 Bolt](nemotron-3.5-bolt/README.md) |
| Qwen3-ASR 1.7B | BF16 safetensors | Audio → text | [Qwen3-ASR](qwen3-asr/README.md) |
| Qwen3-TTS 12Hz 1.7B | BF16 safetensors (CustomVoice, VoiceDesign, Base) | Text, reference audio → speech | [Qwen3-TTS](qwen3-tts/README.md) |
| OmniVoice | float32 safetensors | Text, reference audio → speech in 646 languages | [OmniVoice](omnivoice/README.md) |
| VoxCPM2 | BF16 safetensors, or audio.cpp's BF16/Q8_0 GGUF | Text, reference audio → speech in 30 languages | [VoxCPM2](voxcpm2/README.md) |
| EmbeddingGemma 2 | BF16 safetensors | Text, images, video, audio → embeddings | [EmbeddingGemma 2](embeddinggemma-2/README.md) |
| Qwen-Image-2.1 | BF16 safetensors | Text, images → images | [Qwen-Image-2.1](qwen-image-2.1/README.md) |

Each page gives the files with their pinned revisions, how to run the model,
its options and known limits, and how its outputs are verified. Current
speed is on the [performance](../PERFORMANCE.md) page. What developers need
per model (checks and experiments) is under
[development/models](../development/models/README.md).
