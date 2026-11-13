# Design

rusty-gufo's host code is Rust; its GPU kernels stay HIP. This page explains
why, how the workspace is laid out and built, and the safety model that
keeps the HIP boundary sound. How the code relates to Gufo, the C++ engine
whose kernels it uses, and how it is measured against Gufo's C++ build is in
[relation to Gufo](PARITY.md).

## Why Rust

This project is developed with coding agents, as Gufo is; the repository carries instructions
([AGENTS.md](../../AGENTS.md)) and skills for them. In C++, a lifetime, aliasing
or thread-safety mistake compiles and shows up later as memory corruption,
a wrong token or a hang under load. The Rust compiler rejects those mistakes
before anything runs, with the line at fault, which an agent can act on; a
corruption at runtime names nothing. rusty-gufo keeps the kernels of Gufo's
C++ engine and implements the host code anew:

| Stays HIP/C++ | Rust |
| --- | --- |
| `__global__` kernels | Engines, weight loading, sessions, KV and snapshot management |
| Launchers: grid and block choice, template dispatch, geometry checks | Server, scheduler, continuation caches, CLI, `eval` |
| Adapted llama.cpp MMQ sources | GGUF, sampling, tokenizers, quantization, diagnostics |
| | hipBLAS/hipBLASLt plan selection, HIP graphs, streams, allocation |

What it buys:

- **Hand-written `unsafe` lives in two crates.** It is confined to
  `gufo-hip` (the GPU) and `gufo-ffmpeg` (the FFmpeg calls of video
  decoding), apart from the kernel build script's jobserver handoff; kernel
  crates allow `unsafe` only in their generated launcher module, every other
  crate forbids it, and `tools/ci/check-unsafe.py` enforces that.
- **Kernel launches are checked before they are enqueued.** Each launcher's
  contract (`launch.toml`) declares the extent of every buffer, the stream it
  must belong to and the bounds of every index buffer; a buffer one element
  short, from another stream or aliased as input and output is refused (the
  last at compile time).
- **Graph replays cannot touch freed or replaced memory.** A captured graph
  holds every allocation it uses, and every replay is validated against the
  launches that captured it.
- **One build system.** Cargo builds the one `gufo` binary, without CMake
  or Nix. Production dependencies are small and mostly pure Rust (PNG, JPEG
  and WebP decoding, Unicode normalization, rustls).

Kernels, launch geometry and arithmetic order are Gufo's, so outputs are
bit-identical to the C++ engine wherever the C++ path is deterministic, and
speed, load time and memory are checked against it
([performance against Gufo](PARITY.md#performance-against-gufo)). Built with
the same ROCm, every device function the Rust binary shares with Gufo's
C++ build has the same instructions and resources
(`tools/bench/kernel-isa.py`; llama.cpp's abort stubs aside, which print
their source path). The HIP sources differ from Gufo's only in host
code: the C++ launchers that Rust replaces are gone, adapters give the
launcher contract its flat signatures, and the two vendored quantized-GEMM
tiers take the caller's scratch (Flash-Next) and the engine's own stream
(DeepSeek), see
[implemented differently](PARITY.md#implemented-differently).

## Workspace

| Crate | Path | Role |
| --- | --- | --- |
| `gufo` | `src/cli` | The binary: every command and the model bindings of `serve` |
| `gufo-serve` | `src/cli/serve` | HTTP server, APIs, scheduler, runner pool, continuation caches |
| `gufo-core` | `src/core` | GGUF, tokenizers, sampling, grammars, the text runner interface |
| `gufo-eval` | `src/eval` | `gufo eval` against an OpenAI-compatible server |
| `gufo-formats` | `src/formats` | GGML storage formats and block geometry |
| `gufo-hip` | `src/hip` | Streams, memory, graphs, weights, BLAS: the hand-written `unsafe` of the GPU |
| `gufo-ffmpeg` | `src/ffmpeg` | Video decoding through FFmpeg, loaded on the first video: the `unsafe` of the FFmpeg calls |
| `gufo-kernel-build` | `src/kernel_build` | `launch.toml` to C++ shim and Rust wrappers, `hipcc` |
| `gufo-cli-kernels` | `src/cli/kernels` | The binary's own kernels (the allocation benchmark of `gufo diagnose`) |
| `gufo-hip-smoke` | `src/hip/smoke` | Starts the HIP runtime as a `gufo` executable does; the template for executables' build scripts |
| `gufo-<model>`, `gufo-<model>-kernels` | `src/models/<model>` | A model's engine (`#![forbid(unsafe_code)]`) and its kernels with their contract |

The models are `qwen35` (the `qwen35` architecture: Qwen3.8 27B and the
dense Qwen3.5 family), `qwen38_flash_next`, `deepseek_v4_flash`,
`qwen35moe` (Qwen3.6 35B-A3B), `qwen3_asr`, `qwen3_tts`, `qwen_image_21`,
`embedding_gemma2`, `gemma4`, `nemotron` (Nemotron 3.5 Bolt), `omnivoice`
and `voxcpm` (VoxCPM2). `qwen`
holds what the Qwen language models share: the chat template, the vision
encoder and the kernel crate over Gufo's `src/models/qwen` HIP sources,
which stay at Gufo's paths.
Model code, kernels, tests and numerical contracts stay together, and no
model calls another model's kernels (the Qwen language models share
`gufo-qwen-kernels`), so a kernel change cannot alter another model's output
or speed.

## Safety model

### `unsafe` policy

- Workspace lints: `unsafe_op_in_unsafe_fn = "deny"`,
  `clippy::undocumented_unsafe_blocks = "deny"` and `clippy::unwrap_used =
  "deny"` (tests may opt out).
- `#![forbid(unsafe_code)]` in every crate except `gufo-hip`,
  `gufo-ffmpeg`, `gufo-kernel-build` and the kernel crates. Kernel crates
  deny it with one `#[allow(unsafe_code)]` on their generated module;
  `gufo-kernel-build` denies it except for the build-time jobserver handoff.
- `gufo-ffmpeg` holds the FFmpeg boundary of video embeddings: `dlopen` of
  libavformat, libavcodec, libavutil and libswscale on the first video, the
  calls, and FFmpeg's IO and interrupt callbacks into the decoder. Struct
  fields go through a C shim compiled against the build's headers
  (`src/ffmpeg/shim/fields.c`), so no Rust code assumes FFmpeg's layouts, and
  libraries of another major version than the headers are refused.
- Changes to `src/hip/` and `src/ffmpeg/` need human review: that is where a
  mistake becomes memory corruption.

A safe launch wrapper is only as sound as its contract, so the **trusted
base** is `gufo-hip`, `gufo-kernel-build`, every `launch.toml` (a wrong
extent compiles and lets a kernel overrun a buffer that passed the check),
the C++ launchers and kernels behind them, the BLAS, device-property and
pointer-attribute shims in `src/hip/shim/`, and `gufo-ffmpeg` with its field
shim. Changes there get the same
review as `unsafe` code. Every launcher declares `probes`: the generated
test runs each with buffers of exactly the declared extent and a 4,096-element
guard region, and fails when a kernel writes into the guard. Probes catch
extents too small for what a kernel writes, not for what it reads; those
remain a review item.

### `gufo-hip`

- `Stream`: `Send`, not `Sync`, so launch order is program order. After a
  sticky device fault the stream is poisoned and every further operation
  fails fast; the owning engine is recreated.
- `DeviceBuffer<T>`: the single owner of mutable memory of one stream, freed
  in stream order. `DeviceTensor<T>`: immutable memory readable on every
  stream (weights). `DeviceSlice`/`DeviceSliceMut`: borrowed views with
  length and allocation; read-only arguments accept tensors and buffers of
  the launch stream, written arguments only buffers of the launch stream.
- `Stream::alloc_persistent` sizes long-lived memory exactly; the stream
  pool behind `Stream::alloc` rounds up by up to 128 MiB.
- `DeviceIndices`: `u32` values within a bound fixed at allocation, written
  only by a checked host upload or by kernels whose contract guarantees the
  bound, so a kernel that indexes a table with them cannot be handed an
  out-of-range index.
- `PinnedBuffer<T>`: host memory for asynchronous copies whose host view
  waits for the last copy, so host code can neither read a half-transferred
  buffer nor overwrite one the DMA engine still reads.
- Frees that synchronise the device, and frees on a capturing stream, go
  through a queue that drains only while no stream captures; otherwise a
  free on one thread would invalidate a capture on another. Allocations
  that synchronise the device and the creation of hipBLASLt handles, whose
  synchronous `hipMemset` HIP refuses during any capture in the process,
  wait until no stream captures.
- `GraphCache`: the first call runs eagerly, the second captures, later calls
  replay. A graph retains every allocation it captured, so it never writes
  freed memory; a replaced buffer is detected before replay and recaptured;
  allocation or synchronisation during capture keeps that key eager. Every
  replay re-runs the body record-only and compares buffers, lengths, scalars
  and launch order with the capture (about 160 ns per launch, while the
  device executes the graph); a mismatch fails the step and demotes the key
  to eager execution. Per-step values therefore live in device buffers.
- Weights: `MappedFile` (read-only mmap) and `WeightLoader` with the
  placements `DirectCopy` (O_DIRECT reads through pinned staging),
  `CachedCopy` (device copies from the registered mapping into exactly sized
  arenas), `Map` (the registered mapping itself) and `HostCopy` (a
  huge-page copy, registered read-only). Tensors exist only after `finish()`,
  so no stream reads a weight still being copied.
- BLAS: `Blas` (hipBLAS) and `BlasLt` (hipBLASLt with per-shape plans and
  workspace) behind `src/hip/shim/blas.cpp`, whose `static_assert`s check
  every call and struct against the installed ROCm headers. Operands are
  checked like launch arguments; the model's `AlgoPolicy` chooses the
  algorithm, since that decides the accumulation order.

### `launch.toml`

One file per kernel crate declares its launchers; the build generates a C++
shim and a Rust wrapper for each. A launcher is a non-template C++ function
`LaunchX(pointers..., scalars..., hipStream_t) -> void | bool`, where `false`
means "geometry unsupported, nothing enqueued".

```toml
[[launcher]]
name = "text_swiglu"
cpp = "LaunchTextSwiGLU"
returns = "void"                       # or "supported" (bool)
params = [
  { name = "gate", kind = "in", elem = "f32", len = "count" },
  { name = "up", kind = "in", elem = "f32", len = "count" },
  { name = "output", kind = "out", elem = "bf16", len = "count" },
  { name = "count", kind = "scalar", type = "usize" },
]
probes = [{ count = 257 }]
```

- Parameter kinds: `in`, `inout`, `out` (device views), `indices` and
  `indices_out` (bounded `u32` buffers), `scalar`, `quant` (GGML-quantized
  weights checked against their format and block size), `format`, `const`,
  `alias` and `stream`. `len` expressions use integers, scalar parameters,
  `+ - * / %` and parentheses, all overflow-checked; `requires` adds
  relations no extent expresses; `contract_only` scalars size extents
  without being C++ arguments; `host_visible` marks a buffer the C++
  launcher writes from the host, which must be memory the host addresses at
  its device address.
- The shim `static_assert`s the launcher's exact function-pointer type (any
  drift in constness, width or order fails the build), converts exceptions
  to a status so nothing unwinds into Rust, and reads `hipGetLastError()`.
- The Rust wrapper checks streams, extents, index bounds and requirements,
  records the launch for graph capture and validation, then calls the shim.
  The checked path costs the same host time per launch as a direct C++
  `<<<>>>` launch (about 1.9 µs).
- Template dispatch stays inside C++; adapters (`adapters.hip` in a kernel
  crate) flatten what the contract cannot describe directly.

## Build

Linux x86-64 on Strix Halo (`gfx1151`) only; see the
[README](../../README.md#build-from-source) for the prerequisites.

- `cargo build --release -p gufo` builds `target/release/gufo`; each kernel
  crate's `build.rs` is `gufo_kernel_build::compile_kernels("launch.toml")`.
- One ROCm prefix supplies compiler, headers and runtime: `$ROCM_PATH`, else
  `/opt/rocm/core`, else `/usr`. The binary's RUNPATH points to its `lib`, and
  `gufo-hip` refuses to run when another HIP runtime was loaded (the
  distribution and AMD runtimes share the soname `libamdhip64.so.7`). Every
  crate producing an executable calls `gufo_kernel_build::link_hip_runtime()`.
- Kernels compile with `--offload-arch=gfx1151 -O2 -g -DNDEBUG -fPIC -Wall
  -Wextra` (Gufo's release flags) plus each source's `std`, `flags` and
  `defines`; `hipcc` gets the prefix's `--rocm-path` and include directory and
  `--gcc-install-dir` of the linker driver, so ROCm's clang uses the GCC whose
  `libstdc++` the link takes.
- Kernel objects are keyed by compiler identity and command line and rebuilt
  when a file of their depfile changes; `hipcc` runs in parallel within
  Cargo's jobserver.
- `GUFO_RELEASE_VERSION` and `GUFO_REVISION` set what `gufo --version`
  prints.
- `gufo-hip`'s `build.rs` refuses a ROCm release other than the qualified
  10.1.0 unless `GUFO_ALLOW_UNQUALIFIED_ROCM=1` is set: the release's compiler
  fixes the kernels' instructions and its libraries choose the GEMM
  solutions, so output tokens are checked against Gufo only on it.
- The Python tools install into a virtual environment from
  `tools/requirements.txt`; the benchmark references build with
  `tools/bench/build-reference.sh`.

## Tests and the gate

The gate (`tools/ci/check-rust.sh`), the test tiers and what they need are
described in [testing](TESTING.md).
