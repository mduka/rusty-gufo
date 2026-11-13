# Changelog

Notable user-facing changes are recorded here. Versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

* VoxCPM2 (OpenBMB) text-to-speech in 30 languages through `gufo serve
  tts`, from the VoxCPM extension for Gufo: an automatic voice, voice design
  and style from `instructions`, and cloning with or without a transcript;
  from the Hugging Face directory or one of audio.cpp's GGUFs (BF16, Q8_0
  kept 8-bit on the GPU), 48 kHz output streamed patch by patch.
* The dense Qwen3.5 models prefill their attention on the matrix cores (the
  Qwen3.8 27B's WMMA kernel): long prompts many times faster; logits no
  longer bit-identical to Gufo's, the same tokens on the checked prompts but
  for one near tie.

## 0.1.0 (2026-11-12)

The first release: an AI engine in Rust for AMD Strix Halo (`gfx1151`) on a
common base with [Gufo](https://github.com/gufo-org/gufo). Its host code is new;
its GPU kernels are those of Gufo v0.28.1 and of the C++ extensions for Gufo,
and every model's outputs equal those of the C++ build it comes from, bit for
bit on that build's deterministic paths
([relation to Gufo](docs/development/PARITY.md)).

### Features

* Engines, server, scheduler, CLI and the HIP runtime layer written in Rust,
  built as one Cargo workspace into one `gufo` binary.
* Text models: Qwen3.8 27B (Q4/Q8, images, autoregressive, DFlash2), the dense
  Qwen3.5 0.8B to 9B (text), Qwen3.8 Flash-Next (Q4, images, autoregressive,
  MTP) and DeepSeek V4 Flash (autoregressive, DSpark).
* Gemma 4 (26B-A4B, 31B, 31B QAT) with MTP drafting, prompt lookup, image
  input and native tool calls, from Francesco Moca's Gemma 4 extension for
  Gufo ([Gemma 4](docs/models/gemma-4/README.md)); only 26B-A4B is checked.
* Qwen3.6 35B-A3B with MTP and DFlash2 drafting, batched decoding and image
  input, from the Qwen3.6 35B-A3B extension for Gufo, generated from Strycast
  ([Qwen3.6 35B-A3B](docs/models/qwen3.6-35b-a3b/README.md)).
* Nemotron 3.5 Bolt 74B-A9B with LiLiCorr drafting (adaptive draft lengths,
  concurrent requests drafting in turn), batched decoding and image input,
  from the Nemotron extension for Gufo
  ([Nemotron 3.5 Bolt](docs/models/nemotron-3.5-bolt/README.md)).
* Speech, embedding and image models: Qwen3-ASR 1.7B, Qwen3-TTS 1.7B,
  OmniVoice with an automatic voice, voice design and voice cloning in 646
  languages (from the OmniVoice extension for Gufo), EmbeddingGemma 2 (text,
  image, video and audio embeddings) and Qwen-Image-2.1.
* OpenAI-compatible HTTP APIs (Chat Completions, Completions, Responses,
  Embeddings, audio speech and transcriptions, images), the Anthropic
  Messages API, sentence-wise speech streaming and realtime transcription over
  WebSocket, and the llama.cpp server's extensions ([server](docs/SERVER.md)).
* `gufo arbiter`: servers share one GPU so that streamed speech stays in real
  time while a language model generates ([sharing the GPU](docs/ARBITER.md)).
* Video inputs decoded by FFmpeg, whose libraries load on the first video
  request.

### Not included

* From Gufo v0.28.1: MiniMax H3 video generation, Windows support, the
  `serve llm --trace` content log, the CPU fallback of `prompt` and `chat`,
  and `bench --enable-yarn`
  ([differences from Gufo](docs/development/PARITY.md#differences-from-gufo)).
