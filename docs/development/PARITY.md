# Relation to Gufo

rusty-gufo is a Rust engine on a common base with
[Gufo](https://github.com/gufo-org/gufo), the C++ inference engine for the
same hardware. Its host code (engines, server, scheduler, CLI and the HIP
runtime layer) is a new implementation in Rust; its GPU kernels are the HIP
sources of Gufo and of the C++ extensions for Gufo that brought Qwen3.6
35B-A3B, Gemma 4, Nemotron 3.5 Bolt and OmniVoice. It does not follow Gufo
release by release. The C++ builds below stay the references
for inference; everything around inference is free to change.

| Models | Reference |
| --- | --- |
| Qwen3.8 27B, dense Qwen3.5, Qwen3.8 Flash-Next, DeepSeek V4 Flash, Qwen3-ASR, Qwen3-TTS, EmbeddingGemma 2, Qwen-Image-2.1 | Gufo [v0.28.1](https://github.com/gufo-org/gufo/releases/tag/v0.28.1) |
| Gemma 4 | [gemma4-gufo-extension](https://github.com/franzmoca/gemma4-gufo-extension) `a31ce42`, built on Gufo `0e2de43e` |
| Nemotron 3.5 Bolt | [gufo-extensions/nemotron](https://github.com/gufo-extensions/nemotron) `62f85dc`, built on Gufo `0e2de43e` |
| OmniVoice | [gufo-extensions/omnivoice](https://github.com/gufo-extensions/omnivoice) `4bf376c`, built on Gufo `0e2de43e` |
| Qwen3.6 35B-A3B | [strycast-qwen35moe-gufo-extension](https://github.com/slimsami/strycast-qwen35moe-gufo-extension) `4882b7a`, built on Gufo `0fb50ce4`; for images with its vision gate opened to width 2048 |

1. **Inference stays equivalent.** From the same inputs, every model produces
   the same output tokens as the C++ build of its reference with the same
   ROCm, greedy and with a fixed seed. For models without text tokens, the
   data before the encoder counts: audio codes, image latents, embedding
   vectors. The kernels are the reference's HIP sources at its paths;
   fixtures recorded from the C++ build and A/B runs against it
   ([the C++ reference](#the-c-reference)) check them. Each model's checks
   are listed in its `CHECKS.md` ([models](models/README.md)).
2. **The rest may differ.** A response may order its JSON members
   differently, other encoders may write audio, images and video, and crates
   may replace hand-written infrastructure such as the HTTP server or JSON.
   JSON that reaches the prompt or a grammar keeps its member order and
   spelling: chat templates render tool schemas into the prompt, and the
   grammar compiler follows a schema's member order.
3. **Speed, load time and memory** stay at least as good as the reference's,
   by the gates of [Performance against Gufo](#performance-against-gufo).
4. **Gufo's later changes** are taken one by one where they help. A kernel
   change from a later release moves the reference: its fixtures are recorded
   again from a C++ build of that release. HIP and C++ sources keep Gufo's
   paths. Comments that cite a C++ file (`src/cli/serve/openai_chat.cpp`,
   `tests/cli/http_server_test.cpp`) mean the reference's tree: Gufo v0.28.1,
   where `git show v0.28.1:<path>` in a Gufo clone finds them, or for the
   Gemma 4, Nemotron and OmniVoice packages the extension at the commit
   above.

## Differences from Gufo

Against Gufo v0.28.1, for users and contributors alike. Everything not listed behaves as Gufo does.

### Not carried over

| Gufo feature | Status |
| --- | --- |
| MiniMax H3 video generation (`gufo video`) | Not ported: its 134 GiB of BF16 safetensors exceed what a 128 GiB Strix Halo holds beside other work, so a port could not be compared with the C++ engine. A Rust implementation on GGUFs is the planned replacement. |
| `serve llm --trace` (request and reply content to a JSONL file) | Not ported, by decision: it writes client content to disk. |
| `prompt`/`chat --cpu` (CPU fallback) | Not ported: without a device, loading fails. |
| `bench --enable-yarn` | Not ported: `gufo bench` measures the native context; `serve llm --enable-yarn` exists. |
| Dense Qwen with F32 or Q8_K projections, or with other head and state shapes (Gufo's hipBLAS/hipBLASLt projection routes, unfused QK-norm/RoPE, baseline prefill attention and plain recurrence) | Not ported: no published checkpoint needs them, and loading refuses such files. Both sides refuse F16 projections. |
| The DeepSeek V4 Flash profiling and trace hooks (`GUFO_DEEPSEEK_ROCM_*_PROFILE`, `GUFO_DEEPSEEK_DSPARK_TRACE`, `_WIDTH_TRACE`, `_TIMING`) | Not ported: diagnostics only; `tools/prof/prof.py` profiles the Rust binary. |
| `tune_hipblaslt` and its plan database (`GUFO_HIPBLASLT_PLAN_CACHE`), `gufo-kernel-bench` | Not ported: tools outside `gufo`. Plans are chosen per shape at run time, as Gufo does without a database. |
| Windows support (Gufo's `src/core/platform` layer, Windows builds and their workflows, Gufo #775, Gufo #805, Gufo #872) | Not ported: Rust uses the standard library and `gufo-hip`, not Gufo's platform layer. |
| The Nix flake (package, development shell and reference packages), its ROCr queue-memory patch and the Devlab CI workflows | Not ported: Cargo builds the binary against the installed ROCm, the Python tools install from `tools/requirements.txt`, the benchmark references build with `tools/bench/build-reference.sh`, and hosted CI runs the ROCm-free checks. A ROCm built with Gufo's [queue-memory patch](https://github.com/gufo-org/gufo/blob/v0.27.0/.devops/nix/patches/rocr-fixed-cwsr-rocm10.patch) keeps the same protection against queue evictions during host memory compaction. |

### Implemented differently

| Area | Rust |
| --- | --- |
| Conversation cache | Rust keeps its own continuation cache (RAM and disk tiers, same-slot in-place restore) with Gufo's policies: the deepest checkpoint across RAM and disk, metadata-only startup, oversized checkpoints on disk only and the wait for checkpoints being published. It does not use Gufo's `src/cache` component. Disk checkpoints are written in 1 MiB pieces but serialised and restored as whole payloads, so a disk transfer stages the whole checkpoint in host memory. Two consequences show in the functional suites on Flash-Next: an exactly repeated request restores the state before the assistant's opening and prefills its 7 tokens again, where `src/cache` keeps the whole prompt (`sampling-defaults`); and after earlier suites have filled the cache, `cache-edits` misses an intermediate checkpoint (alone, it passes). Restoring a Flash-Next snapshot into another session copies all of its rows; Gufo tracks snapshot lineage and keeps rows both sessions already share. A Flash-Next capture freezes the mutable state on the device and leaves the rows in the session; once a request has generated 10 tokens, the frozen state of the snapshots captured so far moves into pageable host memory on a thread of its own and the device blocks return for the next captures. Gufo copies captures into pinned host memory, 2 GiB of which it commits at load and grows before each prefill; restoring the state from pageable memory takes about 1.5 ms longer. |
| Flash-Next n-gram rows across cache checkpoints | Read through the page cache; the prefetch across checkpoint boundaries is not ported. |
| Image sessions and HIP graphs | Graph keys carry whether a session has an image RoPE table, so text graphs survive an image's arrival and image graphs are captured once per shape. |
| JPEG decoding | `zune-jpeg` instead of libjpeg-turbo: pixels may differ by a few levels (another IDCT and upsampling), so outputs on JPEG images may differ. `tests/functional/embedding_images.py` therefore accepts a cosine of 0.999 between a JPEG's embedding and that of Pillow's decoded copy, where Gufo asks for 0.99999 (measured: 0.99983 against C++'s 0.999999). PNG and WebP decode as Gufo's do. |
| Flash-Next quantized-GEMM scratch | Each routed call of the vendored tier allocates from scratch the executor passes (`qfn_mmq_set_scratch`), so a captured graph keeps memory that cannot be freed under it; Gufo's tier grows one arena per stream in blocks (Gufo #740). |
| DeepSeek V4 Flash quantized-GEMM scratch | The vendored tier reuses its persistent scratch on the engine's stream; Gufo's C++ engine runs on the legacy stream, the only one where the tier reuses it. |
| `--cache-ram-bytes off` with `--cache-disk` | Keeps no RAM checkpoint and pins no cache memory; the Rust pool still restores the disk checkpoints. Gufo's component cache instead keeps up to 512 MiB of backing for Qwen and Flash-Next captures (Gufo #783). |
| Dense Qwen weights on the device (Gufo #708, Gufo #794) | A whole-file device image, filled through eight pinned staging buffers that drop the mapping's pages as they go; when the image would leave the device heap less than its dispatch reserve, or its allocation fails, the file falls back to a registered host copy. Gufo falls back to its per-tensor weight upload. |
| Qwen-Image weight upload (Gufo #830) | `gufo_hip::FileUpload` owns each destination buffer from `copy` until `finish` hands it back, so no safe caller can reach memory a reader thread is still filling. The reader count, staging size, queue bound and the overlap of staging and copying are Gufo's `WeightUpload`. |
| Video embedding input (Gufo #869) | The same FFmpeg libraries, loaded the same way on the first video; `gufo-ffmpeg` reaches struct fields through a C shim built against the build's headers and refuses libraries of another major version, where a mismatch would be undefined behaviour in C++. |
| Audio embedding input (Gufo #819, Gufo #828) | WAV, FLAC, Ogg Vorbis and MP3 decode in Rust (`hound`, `claxon`, `lewton`, `nanomp3`) and other sample rates resample with `rubato`; Gufo uses libsndfile and libsoxr. PCM WAV and FLAC decode exactly, MP3 and Ogg within 3e-7; resampled inputs differ (a few percent of the signal energy near the new Nyquist frequency for 24 to 48 kHz stereo). From the 16 kHz PCM on, features, encoder and embeddings equal Gufo's bit for bit. |
| Generated-token logprobs (Gufo #814) | Streaming buffer limits count the logprob records at Rust struct sizes, a few bytes apart from C++. Requests that select the unmodified argmax verify in the batched path, which Gufo also uses unless batched verification is off. |
| DFlash2 prompt features | The 27B target captures prompt features into one pinned buffer per model, which grows to the longest prompt and serves every session; Gufo pins about 400 MiB of host memory per session at startup. |
| Message text that spells a control token | Tokenized as text by whole message span; Gufo decides per tokenized slice, so a prompt slice that cuts such a span can tokenize differently (and reuse less cache). |
| Responses reasoning | Trimmed like the chat completion's reasoning, as llama.cpp does; Gufo keeps the newline before `</think>`. |
| A stream cancelled before any output | Still commits its empty response (logged as `disconnected` once the client is gone); Gufo writes nothing and logs `completed`. |
| Range errors | Bounds print as integers, without `std::to_string`'s decimals. |
| Unicode property escapes in schema patterns | General categories and the binary properties listed under [structured output](../SERVER.md#structured-output) are accepted; scripts, blocks and other properties are rejected, where Gufo resolves them through ICU. |
| A panic on the scheduler thread | Fails resident and queued requests with 503 `scheduler_failed` and exits with status 70 ([lifecycle](../SERVER.md#lifecycle-and-limits)); Gufo terminates. |
| Structured-output vocabulary at startup | A build failure does not stop the server; the first constrained request builds it again and reports the error. Gufo fails startup. |
| DeepSeek V4 Flash startup lines | Written to stderr directly, so `--log-level` does not filter them. |
| Load and image-cache diagnostics | No `phase=artifact_identity` load line and no `event=image_cache` debug line. |
| Dense Qwen3.5 prefill attention (16/4 and 8/2 heads) | The WMMA attention of the Qwen3.8 27B, instantiated for these head counts, at any prompt length; Gufo runs `LaunchBatchedAttention`, which holds a row's scores in LDS up to 16,128 positions. The logits differ in the last bits (FP16 queries and probabilities); the checked prompts give Gufo's tokens but for one near tie of the two best logits ([Qwen3.5 checks](models/qwen3.5/CHECKS.md)). Long prompts prefill many times faster. |

### Only in rusty-gufo

| Option | Meaning |
| --- | --- |
| `--api-compat gufo\|llama.cpp` | Request semantics where Gufo and llama.cpp differ; `gufo` (the default) is Gufo's ([server](../SERVER.md)). |
| `--responses-reasoning-target summary\|reasoning-text\|reasoning` | Where `/v1/responses` puts the raw reasoning; `summary` (the default) is Gufo's. |
| `--sse-ping-interval N` | llama-server's SSE keep-alive interval; Gufo keeps streams alive at a fixed interval. |
| `gufo arbiter`, `--arbiter HOST:PORT` | GPU sharing between servers: text generation yields while a streamed TTS falls behind real time ([sharing the GPU](../ARBITER.md)). |


## Performance against Gufo

A/B runs of `target/release/gufo` against a C++ build of v0.28.1 with the
same ROCm 10.1 runtime (`tools/bench/ab.py`; `tools/bench/ab-upstream.sh`
runs every workload below, with the models named in its header). A
workload runs as whole runs of one binary, the C++ build twice and then
Rust twice, with a third run of a side whose two runs disagree; the C++
runs stay the baseline for later candidates while the reference and the
toolchain stay the same.
Rust must be no slower than C++ in prompt processing and generation within
0.5%, load no more than 5% slower and peak memory no more than 1% higher;
each verdict uses the upper bound of the bootstrapped 95% interval of the
ratio of a Rust run's rounds to the pooled C++ rounds. Peak memory is
everything a run takes from the machine: host memory the kernel cannot
reclaim, which on the APU includes the GTT, plus VRAM. Device memory (VRAM
plus GTT) is reported but not judged, as both sides keep some buffers in
different places: C++ copies serving snapshots into pinned host memory it
commits at load, Rust keeps them on the device until a decode is under way.

### Results

Measured on 2026-11-04 and 05 on a Radeon 8060S (gfx1151) with 128 GiB and
ROCm 10.1, 4 to 8 rounds per run, with the code of release 0.1.0 before
Gemma 4, Nemotron 3.5 Bolt and OmniVoice were added (they do not touch these
workloads). Each cell
gives, per Rust run, the median ratio Rust/C++ as a change in percent (bold
where that run fails the gate), and in brackets the highest upper bound of
the runs' 95% intervals.

| Workload | Request | Gated stages | Load | Memory | Device memory | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| q27-q4-ar | −0.08 / −0.04 (≤ +0.04) | pp **+0.68** / **+0.56** (≤ +1.19) | +0.58 / +1.30 (≤ +2.23) | +0.19 / +0.11 (≤ +0.81) | −0.37 | FAIL (0 of 2) |
| q27-q4-dflash2 | +0.01 / +0.05 (≤ +0.11) | pp **+1.09** / **+1.20** (≤ +1.38) | +0.92 / +1.71 (≤ +2.85) | −0.18 / −0.30 (≤ +0.39) | −0.34 | FAIL (0 of 2) |
| q27-q8-ar | ±0.00 / ±0.00 (≤ +0.14) | pp −0.97 / −0.51 (≤ −0.18) | +0.57 / +0.55 (≤ +1.40) | +0.08 / +0.17 (≤ +0.56) | −0.22 | PASS (2 of 2) |
| q27-q8-dflash2 | −0.02 / ±0.00 (≤ +0.04) | pp −0.26 / −0.14 (≤ +0.38) | +1.36 / +1.52 (≤ +2.58) | −0.13 / −0.07 (≤ +0.21) | −0.20 | PASS (2 of 2) |
| q27-q4-ar-d32k | −0.05 / ±0.00 (≤ +0.09) | pp −0.06 / ±0.00 (≤ +0.11) | +2.34 / +1.65 (≤ +3.46) | −0.38 / −0.32 (≤ −0.11) | −0.30 | PASS (2 of 2) |
| fn-ar | −0.39 / −0.42 / −0.39 (≤ −0.20) | pp +0.04 / −0.04 / −0.04 (≤ +0.15) | **+9.49** / −0.25 / **+3.10** (≤ +20.66) | −0.36 / −0.29 / −0.38 (≤ −0.27) | −0.52 | FAIL (1 of 3) |
| fn-mtp | −3.62 / −3.62 (≤ −1.06) | pp +0.03 / +0.02 (≤ +0.16) | −3.27 / −3.64 (≤ +4.92) | −0.49 / −0.51 (≤ −0.42) | −0.53 | PASS (2 of 2) |
| fn-ar-d32k | −0.19 / −0.24 (≤ −0.17) | pp −0.03 / +0.06 (≤ +0.15) | **+2.82** / **+0.79** (≤ +22.65) | −0.19 / −0.16 (≤ −0.12) | −0.38 | FAIL (0 of 2) |
| ds-ar | −0.09 / −0.06 (≤ −0.03) | pp −1.26 / −1.68 (≤ −0.26) | **−2.23** / **−1.16** (≤ +18.94) | −0.04 / −0.04 (≤ +0.03) | −0.09 | FAIL (0 of 2) |
| ds-dspark | −1.44 / −1.44 / −1.44 (≤ −1.39) | pp ±0.00 / −0.08 / +0.10 (≤ +0.26) | −3.87 / **−2.45** / −3.99 (≤ +5.18) | −0.04 / −0.05 / −0.04 (≤ −0.02) | −0.08 | PASS (2 of 3) |
| serve-q27-dflash2 | −0.16 / ±0.00 (≤ +0.19) | c1 −0.16 / ±0.00 (≤ +0.19); c4 +0.04 / +0.24 (≤ +0.37); TTFT +0.12 / +0.27 (≤ +0.44); decode −0.25 / −0.02 (≤ +0.12) | −26.18 / −25.57 (≤ −24.29) | −27.91 / −27.91 (≤ −27.56) | −9.37 | PASS (2 of 2) |
| serve-fn-mtp | −0.38 / −0.36 / −0.36 (≤ −0.23) | c1 −0.38 / −0.36 / −0.36 (≤ −0.23); c4 **−0.18** / **+0.45** / **−0.03** (≤ +2.53); TTFT −0.03 / +0.03 / −0.03 (≤ +0.20); decode −0.39 / −0.36 / −0.37 (≤ −0.24) | **−0.39** / **−0.21** / **+3.93** (≤ +13.75) | −1.04 / −1.04 / −1.01 (≤ −0.91) | +0.98 | FAIL (0 of 3) |
| serve-ds-dspark | **−0.91** / −0.75 / −0.80 (≤ +3.15) | c1 **−0.91** / −0.75 / −0.80 (≤ +3.15); c4 **+3.13** / **−0.03** / −1.18 (≤ +7.48); TTFT **−0.77** / −0.45 / −0.45 (≤ +3.34); decode **−0.83** / −0.70 / −0.78 (≤ +3.24) | **+10.79** / **+5.63** / **−0.27** (≤ +17.36) | +0.20 / +0.08 / +0.14 (≤ +0.35) | +0.03 | FAIL (0 of 3) |
| serve-q27-dflash2-t08 | +0.14 / +0.30 (≤ +0.46) | c1 +0.14 / +0.30 (≤ +0.46); c4 **+0.49** / **+0.84** (≤ +1.17); TTFT **+0.44** / **+0.58** (≤ +0.76); decode +0.07 / +0.26 (≤ +0.38) | −27.40 / −27.88 (≤ −26.98) | −27.43 / −27.44 (≤ −27.21) | −9.33 | FAIL (0 of 2) |
| serve-fn-mtp-t08 | −0.70 / −0.73 (≤ −0.53) | c1 −0.70 / −0.73 (≤ −0.53); c4 **+0.22** / **−0.16** (≤ +2.34); TTFT +0.11 / +0.20 (≤ +0.32); decode −0.81 / −0.85 (≤ −0.62) | **+2.37** / **+6.94** (≤ +18.97) | −0.98 / −0.98 (≤ −0.91) | +0.98 | FAIL (0 of 2) |
| serve-ds-dspark-t08 | −1.13 / −1.10 / −1.11 (≤ −0.89) | c1 −1.13 / −1.10 / −1.11 (≤ −0.89); c4 **+1.23** / −1.14 / −1.12 (≤ +3.49); TTFT −0.69 / −0.73 / −0.78 (≤ −0.13); decode −1.31 / −1.26 / −1.29 (≤ −1.16) | **−0.19** / −0.15 / **−0.24** (≤ +14.33) | +0.16 / +0.31 / +0.06 (≤ +0.35) | +0.12 | FAIL (1 of 3) |
| asr | −4.31 / −4.37 / −4.40 (≤ −4.28) | – | **+1.28** / −9.95 / −6.14 (≤ +6.45) | −4.59 / −4.75 / −4.34 (≤ −1.19) | +0.04 | PASS (2 of 3) |
| tts-customvoice | −4.78 / −3.53 / −3.34 (≤ −2.17) | first audio −2.82 / −2.48 / −2.50 (≤ −1.81); streamed −3.60 / −3.50 / −3.52 (≤ −3.40) | **+4.82** / +3.20 / **+3.14** (≤ +8.01) | **−0.09** / **−0.51** / **−0.50** (≤ +1.52) | ±0.00 | FAIL (0 of 3) |
| tts-voicedesign | −3.07 / −2.95 / −4.14 (≤ −2.64) | first audio −2.31 / −2.10 / −2.21 (≤ −1.60); streamed −3.48 / −3.54 / −3.45 (≤ −2.77) | **+15.39** / −1.57 / **+19.97** (≤ +29.14) | **+0.51** / **+0.16** / **+0.34** (≤ +2.54) | ±0.00 | FAIL (0 of 3) |
| image | −0.56 / −0.20 / −0.53 (≤ +0.06) | denoise −0.35 / +0.03 / −0.32 (≤ +0.27); prompt −4.94 / −5.17 / −5.30 (≤ −4.59); VAE −0.92 / −0.86 / −0.94 (≤ −0.34) | −16.32 / **−8.17** / **−0.33** (≤ +10.01) | −0.08 / −0.02 / +0.08 (≤ +0.53) | ±0.00 | FAIL (1 of 3) |
| embedding | −1.63 / −1.26 / −1.58 (≤ −0.31) | single **−1.39** / **−0.22** / **+0.93** (≤ +5.23) | −0.13 / −1.47 / −0.13 (≤ +0.15) | **+0.62** / **+0.20** / **+0.72** (≤ +2.20) | ±0.00 | FAIL (0 of 3) |

The text workloads (`gufo bench`) prefill 512 prompt tokens and generate
128 (64 after 32,768 tokens of context for `d32k`) on Qwen3.8 27B
(UD-Q4_K_XL, UD-Q8_K_XL; autoregressive or with the DFlash2 draft),
Flash-Next (UD-Q4_K_XL; autoregressive or with MTP) and DeepSeek V4 Flash
(IQ2XXS; autoregressive or with DSpark); `pp` is prefill time per token. The serving workloads run the HTTP server at concurrency 1 and 4
(`c1`, `c4`: time per output token); those named `t08` sample at temperature
0.8 with a fixed seed and without top-k. Speech, image and embedding requests
go through their servers: an English recording, the CustomVoice and
VoiceDesign checkpoints, eight steps at 1024×1024, and EmbeddingGemma 2 with
a batch of 32 inputs of about 512 tokens (the request) and one short input
(`single`).

Every invocation of every workload gave the same outputs on both sides.
Image and video embeddings, which these workloads do not cover, take one of
two values per server process at some budgets on both sides: the BLAS plan of
a shape is chosen by timing at its first call, so C++ differs from C++ as
often as from Rust (up to 3e-4 per element, a cosine of at least 0.999997).

Seven workloads pass. Rust needs less memory in every run of 14 of 21
workloads, 28% less when serving the 27B with DFlash2 (see
[DFlash2 prompt features](#implemented-differently)). Most failures have
causes that are not Rust's. Of the alternating rounds cited here, q27-q4-ar,
the first 27B prefill number and the first sampled 27B numbers ran with that
build; the others ran with earlier builds that differ only in the host
sampler and in the waits after host copies, which measure the same:

- **Gufo's speech loaders.** Since v0.27.0, the TTS and ASR loads drop
  the checkpoint's pages after the upload
  ([#845](https://github.com/gufo-org/gufo/pull/845)), so every load reads
  the files again and its peak memory varies with what the kernel has
  reclaimed. The C++ builds of v0.23.0 and v0.27.0 measured in alternating
  rounds load the CustomVoice checkpoint in 615 and 1,644 ms, with a peak of
  6.47–6.49 and 6.48–6.75 GiB. In alternating rounds against v0.28.1, the peak
  spreads over 6.35–6.96 GiB on C++ and 6.49–6.81 GiB on Rust, so the median
  ratios of +1.0 and +1.9% have intervals that reach below 1; per process,
  Rust holds about 19 MiB (+0.3%) more. The load takes +3.3% on CustomVoice,
  one step of the harness's 50 ms poll, and +1.3% (≤ +3.2) on VoiceDesign.
- **Machine state.** Decoding the large models alternates between fast and
  slow rounds on both sides, and slow rounds follow slow loads: with
  80 to 100 GiB of weights on a 128 GiB machine, the kernel reclaims and swaps
  while a round runs. The Flash-Next and DeepSeek workloads fail only through
  this spread: load intervals reach +23%, `c4` intervals +7.5%, and in one
  serve-ds-dspark run the request and decode intervals +3.2%. Their request and
  decode medians lie below C++'s in every run.
- **Run order.** The APU runs at its power limit, so a hotter GPU clocks
  lower and a run that follows other runs is slower; the Rust runs of a
  workload follow its C++ runs. Measured in alternating rounds (ABBA, 8
  rounds per side), q27-q4-ar's prefill is +0.04% (≤ +0.56).
- **Load time.** Loads of the large models take 20 to 25 s on both sides,
  from round to round. The image workload's load median lies below C++'s in
  every run, its interval reaches +10%.

Rust is slower than C++ here, and the causes are open:

- **27B DFlash2 prefill:** +0.47% (≤ +0.56) in alternating rounds, +0.38 and
  +0.63% in two earlier series (+0.62% at v0.27.0). Over 21 repetitions per
  run the hot gap is +0.19% (419.24 against 420.02 tokens per second), and from
  a GPU cooled to 45 °C before each run, in swapped order, Rust leads by 0.15%
  (468.42 against 467.71). Prefill moves far more with temperature than
  between the binaries: 468 tokens per second at 45 to 57 °C, 420 at 80 °C.
  Ruled out: bench arguments, the waits after host copies (a build without
  them measures the same), CPU time per prefill, clocks and power. AB and BA
  rounds show the same gap, so it is not the later slot running hotter.
  Generation is 0.16% ahead in the same rounds.
- **27B DFlash2 serving, the first token:** about 4 ms. +0.32% (≤ +0.59;
  1,331.6 against 1,327.2 ms) sampled in alternating rounds, +0.40% greedy, and
  +0.32% from a cooled GPU. Greedy `c1` and `c4` take +0.2% hot and +0.07 to
  +0.09% cooled. Sampled, decoding is equal (+0.05%) and `c4` takes +0.28%
  (≤ +0.70), +0.51% (≤ +0.89) in the series before.
- **Embedding, one request:** +0.60% over 200 requests per server (5.871
  against 5.836 ms, 35 µs), and +1.63% (≤ +2.78) for three requests right
  after a batch, over 24 rounds, where latency varies by about 15% from
  request to request. The batch request is 0.82% faster and memory is +0.29%
  (≤ +1.45) in the same rounds.
- **Flash-Next serving, a repeated prompt** (against v0.27.0, below): +1.4
  to +1.7 ms to the first token, +0.5 to +1.4% to the tenth. Here the cause
  is known. The restore copies the snapshot's frozen state (about 118 MB) from
  pageable host memory ([conversation cache](#implemented-differently)), which
  takes 3.5 to 4.6 ms; C++ copies it from the pinned host memory it commits at
  load, in 2.2 to 2.8 ms. Matching that needs pinned memory held for the
  cache, and Rust reserves no host memory up front. Pinning a buffer for each
  capture instead costs 35 to 55 ms to the first token, and batching the
  restore's copies gains nothing, as the transfer from pageable memory
  dominates.

Sampling without top-k is new in this comparison: per verified token the host
normalises a target distribution over the whole vocabulary, and per rejected
draft it draws from the residual. Flash-Next with MTP took 20.34 ms per token
against C++'s 19.56 (+4%), with identical outputs, while top-k 40 was equal
(20.79 ms on both). `src/core/src/sampling.rs` now divides in pairs, hashes a
token ID with one multiplication and filters non-candidates before the lookup.
Against the same C++ source in a microbenchmark, the distribution takes 1.37
against 1.47 ms and the residual draw 0.77 against 0.75 ms, and
`serve-fn-mtp-t08` decodes 0.8% and `serve-ds-dspark-t08` 1.3% faster than
C++.

Against v0.27.0 on 2026-11-01, Flash-Next serving with MTP was measured from a
GPU cooled to 45 °C, alternating and in swapped order, with the snapshot
handling of the [conversation cache](#implemented-differently): time to the
first and the tenth streamed token at concurrency 1, four servers per side. A
new 443-token prompt takes 440.7 to 440.9 and 571.6 to 572.4 ms against C++'s
441.0 to 442.8 and 572.8 to 575.0 ms (one server per side, whose first requests
ran in the slow machine state, left out). The second turn of a conversation
(568 tokens cached, 29 prefilled) takes 132.6 to 133.1 and 326.3 to 330.9 ms
against 134.7 to 137.0 and 329.8 to 332.2 ms, the third 136.6 to 139.0 and
269.3 to 271.0 ms against 138.5 to 142.1 and 272.1 to 275.9 ms. An exactly
repeated prompt restores its snapshot from host memory and takes 6.7 to 6.9 and
132.7 to 133.5 ms against 5.2 to 5.3 and 131.6 to 132.0 ms.

### The C++ reference

Build the reference from Gufo's release tag in a directory of its own,
against the same ROCm prefix as `target/release/gufo`. With a
distribution ROCm and a newer GCC installed beside it, the release preset
alone yields a mixed binary, so the HIP compilations get the prefix's
include directory (`/usr/include/hip` shadows it otherwise) and the GCC
whose `libstdc++` headers the C++ files use, and the binary's RUNPATH is
set to the prefix:

```sh
R=/opt/rocm/core-10.1 G=/usr/lib/gcc/x86_64-linux-gnu/15
git clone --branch v0.28.1 --depth 1 https://github.com/gufo-org/gufo.git \
  target/upstream-v0.28.1/src
cd target/upstream-v0.28.1/src
cmake --preset release -DCMAKE_PREFIX_PATH=$R \
  -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++ \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$R -isystem $R/include --gcc-install-dir=$G"
cmake --build --preset release
cp build/release/gufo ../gufo-cpp
patchelf --set-rpath $R/lib:$G ../gufo-cpp
ldd ../gufo-cpp | grep -E 'amdhip64|hipblas|rocblas'   # every one from $R
```

`tools/bench/kernel-isa.py ../gufo-cpp target/release/gufo` compares the
device code of the two builds function by function: with the same ROCm,
every function both have must have the same instructions. Each side also
holds a few template variants the other does not instantiate (the C++
launchers instantiate some no path launches).

`ab.py` refuses to compare binaries that resolve different `libamdhip64`
libraries and records both binaries' hashes, the runtime, the GPU DPM level
and the CPU governor. `gufo diagnose` prints the toolchain the build pinned,
not the runtime it loads; check that with `ldd`.

## Open work

- **Conversation cache:** the parts of Gufo's `src/cache` behind the two
  suite results under "Implemented differently"; a fallback to a retained RAM
  snapshot when a deeper disk restore fails while RAM holds only a live
  frontier (today the request prefills its whole prompt); and the functional
  runners that still read Gufo's component-cache logs
  ([functional tests](../../tests/functional/README.md)). The log check of
  `disk-spacing` reports two stored Flash-Next checkpoints 381 tokens apart
  (C++ passes it); Rust's events carry no checkpoint ids, and the cause is
  open.
- **27B DFlash2, short replies:** two requests that arrive together and end
  within a speculative cycle or two run one after the other
  (`serial-fallback`), where C++ verifies them in one batch, with the same
  texts and about the same total time. The width check of the `batch` suite
  therefore fails on the 27B with DFlash2; earlier builds of this code did the
  same.
- **Performance gaps:** the open causes under [results](#results); also the
  17 MiB of additional anonymous memory in the speech servers.
- **Small dense Qwen:** decode stays 0.2 to 0.4% behind C++ below the 9B
  ([27B experiments](models/qwen3.8-27b/EXPERIMENTS.md), Q8); Flash-Next's
  bit-exact follow-ups F10 and F11 are in its
  [experiments](models/qwen3.8-flash-next/EXPERIMENTS.md).
- **References:** the 27B BF16 comparison as a Rust test (Gufo's
  `qwen27b_target_test`), Flash-Next's independent MTP audit and deep
  execution check (`--mtp-audit`, `--execution-only`), Flash-Next's F32
  reference fixtures with their GPU probe and scorer (Gufo #758), Flash-Next's
  operator chains `hc_route_ops_test` and `routed_wmma_ops_test` and the
  m=17 split of `blaslt_ops_test` (the C++ session fixtures cover these routes
  end to end), the two disabled DeepSeek baseline checks and the `reference`
  suite of `tools/ds4/check.py`.
- **Freeing shared tensors:** the last drop of an immutable tensor waits for
  the whole device (not during a graph capture). Swapping models at run time
  would need it to wait for the tensor's actual users instead.
- **Dependencies:** `tools/ci/check-dependencies.py` checks the crate
  inventory against [third-party notices](../../THIRD_PARTY_NOTICES.md) and
  hosted CI the policy of [`deny.toml`](../../deny.toml); generated crate
  licences (`cargo-about`) are open.
- **Responses reasoning:** `--responses-reasoning-target` picks one target.
  Several at once, none (which needs `encrypted_content` for tool loops) and
  generated summaries for `reasoning.summary` are ideas.
- **Images on Qwen3.5 0.8B and 2B:** their `mmproj` carries another vision
  tower (12 layers of width 768), which Gufo does not take either.
