# rusty-gufo

An open source AI engine for AMD Strix Halo, written in Rust. rusty-gufo
serves text, speech, embedding and image models through OpenAI-compatible
APIs from one binary, built and tuned for Ryzen AI MAX+ systems with the
Radeon 8060S (`gfx1151`) and up to 128 GiB of unified memory.

Its arbiter lets a whole voice assistant share that one GPU: a language
model, speech synthesis and speech recognition run as separate servers, and
streamed speech stays in real time while the language model generates
([sharing the GPU](#sharing-the-gpu)).

rusty-gufo is a Rust engine on a common base with
[Gufo](https://github.com/gufo-org/gufo), the C++ engine for the same
hardware. Its engines, server, scheduler, CLI and HIP runtime layer are a
new implementation in Rust; its GPU kernels are the HIP sources of Gufo and
of the C++ extensions for Gufo that brought Qwen3.6 35B-A3B, Gemma 4,
Nemotron 3.5 Bolt, Nemotron 3.5 ASR, OmniVoice and VoxCPM2. Every model
produces the same
output tokens as the C++ build it comes from, checked bit for bit, and its
speed, load time and memory are measured against that build
([relation to Gufo](docs/development/PARITY.md)). The binary is still called
`gufo`.

- **Models:** Qwen3.8 27B and the dense Qwen3.5 0.8B to 9B, Qwen3.6 35B-A3B,
  Qwen3.8 Flash-Next, DeepSeek V4 Flash, Gemma 4, Nemotron 3.5 Bolt,
  Qwen3-ASR, Nemotron 3.5 ASR, Qwen3-TTS, OmniVoice, VoxCPM2,
  EmbeddingGemma 2 and Qwen-Image-2.1.
- **Speculative decoding:** DFlash2, MTP, DSpark and LiLiCorr drafters, with
  greedy output identical to decoding without them.
- **APIs:** OpenAI Chat Completions, Responses, embeddings, audio and images,
  Anthropic Messages, and the llama.cpp server's extensions; streaming, tools,
  structured output, reasoning controls and images.
- **Serving:** concurrent requests in shared batches, conversation caches in
  RAM and on disk, Prometheus metrics.

## Sharing the GPU

On one GPU, a language model takes the device from a speech server beside
it, and streamed audio breaks up; CPU priorities don't reach GPU work.
`gufo arbiter` coordinates the servers: while streamed speech falls behind
real time, it grants text generation only a share of the GPU, otherwise text
generates at full speed.

```sh
./target/release/gufo arbiter
./target/release/gufo serve tts --model "$TTS" --arbiter 127.0.0.1:7071
./target/release/gufo serve llm --model "$LLM" --arbiter 127.0.0.1:7071
```

With Qwen3.8 Flash-Next decoding beside Qwen3-TTS streaming 50.5 s of speech,
speech fell to 0.79× real time without the arbiter, with 39 gaps and 14.5 s of
silence; with it, speech stays above real time and text generation keeps
about 88% of its speed. Rules, interfaces and the measurements:
[sharing the GPU](docs/ARBITER.md).

## Models

| Model | Inputs → outputs | Modes | Page |
| --- | --- | --- | --- |
| Qwen3.8 27B | Text, images → text | Autoregressive, DFlash2 | [Qwen3.8 27B](docs/models/qwen3.8-27b/README.md) |
| Qwen3.5 0.8B to 9B (dense) | Text → text | Autoregressive | [Qwen3.5](docs/models/qwen3.5/README.md) |
| Qwen3.6 35B-A3B | Text, images → text | Autoregressive, MTP, DFlash2 | [Qwen3.6 35B-A3B](docs/models/qwen3.6-35b-a3b/README.md) |
| Qwen3.8 Flash-Next | Text, images → text | Autoregressive, MTP | [Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next/README.md) |
| DeepSeek V4 Flash | Text → text | Autoregressive, DSpark | [DeepSeek V4 Flash](docs/models/deepseek-v4-flash/README.md) |
| Gemma 4 (26B-A4B; 31B and 31B QAT unchecked) | Text, images → text | Autoregressive, MTP | [Gemma 4](docs/models/gemma-4/README.md) |
| Nemotron 3.5 Bolt 74B-A9B | Text, images → text | Autoregressive, LiLiCorr | [Nemotron 3.5 Bolt](docs/models/nemotron-3.5-bolt/README.md) |
| Qwen3-ASR 1.7B | Audio → text | Batch and streaming transcription | [Qwen3-ASR](docs/models/qwen3-asr/README.md) |
| Nemotron 3.5 ASR Streaming 0.6B | Audio → text in 40 languages | Batch and streaming transcription, language detection | [Nemotron 3.5 ASR](docs/models/nemotron-3.5-asr/README.md) |
| Qwen3-TTS 12Hz 1.7B | Text, reference audio → speech | Custom voices, voice design, cloning | [Qwen3-TTS](docs/models/qwen3-tts/README.md) |
| OmniVoice | Text, reference audio → speech in 646 languages | Automatic voice, voice design, cloning | [OmniVoice](docs/models/omnivoice/README.md) |
| VoxCPM2 | Text, reference audio → speech in 30 languages | Automatic voice, voice design, cloning | [VoxCPM2](docs/models/voxcpm2/README.md) |
| EmbeddingGemma 2 | Text, images, video, audio → embeddings | | [EmbeddingGemma 2](docs/models/embeddinggemma-2/README.md) |
| Qwen-Image-2.1 | Text, images → images | Generation and editing | [Qwen-Image-2.1](docs/models/qwen-image-2.1/README.md) |

Each page gives the files with pinned download commands, the options and the
known limits. The weights are not part of rusty-gufo and keep their
publishers' licenses. Current speed: [performance](docs/PERFORMANCE.md).

## Requirements

- An AMD Strix Halo system (`gfx1151`) running Linux x86-64 with the `amdgpu`
  kernel driver. The GPU uses system memory; the largest models (DeepSeek V4
  Flash, Qwen3.8 Flash-Next) need a 128 GiB system.
- ROCm 10.1 for building and running; the [container](#container) brings its
  own.
- Read and write access to `/dev/kfd` and `/dev/dri/renderD*`, usually through
  the `render` and `video` groups (log out and back in after changing group
  membership).

`gufo diagnose` checks the platform, the GPU, the ROCm runtime and device
access.

## Build from source

The build needs the Rust toolchain, a C/C++ toolchain for linking, FFmpeg's
development files and ROCm 10.1.0 from AMD's package repository. On Ubuntu
26.04, with [AMD's repository](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
set up as the [Containerfile](Containerfile) does:

```sh
sudo apt install build-essential pkg-config \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
  amdrocm-base10.1 amdrocm-llvm10.1 amdrocm-runtime-dev10.1 \
  amdrocm-blas-dev10.1 amdrocm-hipblas-common-dev10.1 amdrocm-ccl-dev10.1 \
  amdrocm-blas10.1-gfx1151
curl -fsSL https://sh.rustup.rs | sh     # the toolchain comes from rust-toolchain.toml
ROCM_PATH=/opt/rocm/core-10.1 cargo build --release -p gufo
./target/release/gufo diagnose
```

The build takes compiler, headers and runtime from one ROCm prefix:
`$ROCM_PATH`, else `/opt/rocm/core`, else `/usr`. It refuses a ROCm release
other than 10.1.0 unless `GUFO_ALLOW_UNQUALIFIED_ROCM=1` is set, because the
outputs are checked only on that release. Python, Triton, Composable Kernel
and MIOpen are not needed to build or run.

To install, copy `target/release/gufo` onto your `PATH`; a link named
`gufo-server` starts the server directly. Model downloads use the `hf` tool
of the `huggingface_hub` Python package.

## Container

The [Containerfile](Containerfile) builds `gufo` against ROCm 10.1 from AMD's
package repository and ships it with that runtime and FFmpeg. The host needs
only the `amdgpu` kernel driver. Models are mounted, not built in:

```sh
podman build -t rusty-gufo .
podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups \
  -p 127.0.0.1:8080:8080 -v "$PWD/models:/models:ro" \
  -v rusty-gufo-cache:/root/.cache \
  rusty-gufo serve --host 0.0.0.0 --port 8080 llm \
  --model /models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
```

The cache volume keeps the code objects ROCm links on the first load, which a
new container would otherwise link again.

## Quickstart

Download Qwen3.8 27B with its DFlash2 drafter and start the server:

```sh
hf download unsloth/Qwen3.8-27B-GGUF \
  --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  Qwen3.8-27B-UD-Q4_K_XL.gguf --local-dir models/qwen3.8-27b
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF \
  --revision 2d9571f8ce46e151f61c6499c99dee6079e1d610 \
  Qwen3.8-27B-DFlash2-Q4_K_M.gguf --local-dir models/qwen3.8-27b
./target/release/gufo serve llm \
  --model models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --speculative dflash2 \
  --dflash-model models/qwen3.8-27b/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

Then ask it something through the OpenAI-compatible API:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-UD-Q4_K_XL",
    "messages": [{"role": "user", "content": "Say something"}]
  }'
```

The server listens on `127.0.0.1:8080` by default; give it `--api-key` before
binding it to another address. For Open WebUI, VS Code, the OpenAI SDK and
Responses-API clients (for example Codex), set the API base URL to
`http://127.0.0.1:8080/v1`. The text server uses the model's native context
and generates until EOS or the context is full; `--context N` sets the
context per session and `--max-tokens N` a default response limit that
clients can override.

Client notes:

- **Codex with Qwen:** Codex sends developer messages mid-conversation, after
  compaction or a settings change. Qwen's template accepts only one leading
  system turn, so the server moves them there. The request that introduces one
  is prefilled again; later requests reuse the cache.
- **Claude Code with Qwen:** not supported. Claude Code adds a system message
  to `messages` on every turn, so the server moves a new one to the leading
  system turn each time and every request prefills the whole history again
  ([gufo-org/gufo#522](https://github.com/gufo-org/gufo/issues/522)).

## Documentation

Using rusty-gufo:

- [Models](docs/models/README.md): one page per model.
- [Command-line interface](docs/CLI.md): every command and its options.
- [Server](docs/SERVER.md): running the server, its endpoints and API
  behavior, limits, metrics and logs.
- [KV cache](docs/KV-CACHE.md): how conversations reuse earlier work, in RAM
  and on disk.
- [Sharing the GPU](docs/ARBITER.md): `gufo arbiter` for several servers on
  one GPU.
- [Performance](docs/PERFORMANCE.md): current speed per model.
- [Changelog](CHANGELOG.md).

Developing rusty-gufo:

- [Contributor guide](docs/development/README.md): prerequisites, layout,
  rules and the developer documents (design, relation to Gufo, testing,
  performance work, per-model checks and experiments).
- [Contributing](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) for coding
  agents.

## Credits

rusty-gufo builds on the work of [Gufo](https://github.com/gufo-org/gufo)
and the Gufo contributors: their GPU kernels, model guides, quality reports,
tests and tools, and much of Gufo's design. Thank you to everyone who built
it. Gufo in turn draws on these open source projects:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for compact model serving, GGUF, and CPU/GPU correctness paths.
- [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp), [Nathanw1014/strix-halo-llamacpp](https://github.com/Nathanw1014/strix-halo-llamacpp), and [gaetan-puleo/llama-cpp-strix-halo](https://github.com/gaetan-puleo/llama-cpp-strix-halo) for Strix Halo optimization inspiration.
- [vLLM](https://github.com/vllm-project/vllm) for continuous batching and paged request scheduling.
- [hipEngine](https://github.com/shisa-ai/hipEngine) for torch-free HIP execution, and native speculative-cycle work.
- [ds4](https://github.com/antirez/ds4) for DeepSeek V4 Flash, MoE scheduling, and DSpark.
- [audio.cpp](https://github.com/0xShug0/audio.cpp) for audio models for tts and asr tasks.

The models beyond Gufo v0.28.1 come from C++ extensions for Gufo (MIT):

- [Gemma 4](https://github.com/franzmoca/gemma4-gufo-extension) by Francesco
  Moca, for Google's Gemma 4.
- [Nemotron](https://github.com/gufo-extensions/nemotron), for NVIDIA's
  Nemotron 3.5 Bolt.
- [OmniVoice](https://github.com/gufo-extensions/omnivoice), for
  [k2-fsa's OmniVoice](https://github.com/k2-fsa/OmniVoice).
- [VoxCPM](https://github.com/gufo-extensions/voxcpm), for
  [OpenBMB's VoxCPM2](https://github.com/OpenBMB/VoxCPM).
- [Nemotron ASR](https://github.com/gufo-extensions/nemotron_asr), for
  NVIDIA's Nemotron 3.5 ASR Streaming 0.6B.
- [Qwen3.6 35B-A3B](https://github.com/slimsami/strycast-qwen35moe-gufo-extension),
  generated from Sami Slimani's [Strycast](https://github.com/slimsami/strycast),
  for Qwen's Qwen3.6 35B-A3B.

rusty-gufo is an independent project; the Gufo project does not run or
endorse it. The two exchange work: rusty-gufo takes over Gufo's GPU kernels,
and parts of its Rust code are contributed back to Gufo.

## Contributing

Contributions are welcome; start with [CONTRIBUTING.md](CONTRIBUTING.md) and
the [contributor guide](docs/development/README.md).

## License

rusty-gufo is licensed under either the [MIT license](LICENSE-MIT) or the
[Apache License, Version 2.0](LICENSE-APACHE), at your option. Code taken
from Gufo, its extensions and other projects keeps its MIT license. Adapted
code and dependencies retain their own notices in [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`; distribute
them with the binary. Model weights are not bundled and retain their
publishers' terms.
