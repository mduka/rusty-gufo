# Development

How to build, test and change rusty-gufo. The user documentation starts at
the [README](../../README.md).

rusty-gufo is a Cargo workspace for Linux x86-64 on AMD Strix Halo
(`gfx1151`): Rust host code over HIP kernels, which each model's kernel crate
compiles with ROCm's `hipcc`. The kernels come from
[Gufo](https://github.com/gufo-org/gufo) and the C++ extensions for it, and
every model's output stays equal to that of its C++ reference
([relation to Gufo](PARITY.md)).

## Documents

| Document | Contents |
| --- | --- |
| [Design](DESIGN.md) | Why Rust, the workspace, the safety model, `launch.toml`, the build |
| [Relation to Gufo](PARITY.md) | The C++ references, differences from Gufo, the A/B comparison and its results, open work |
| [Testing](TESTING.md) | Test tiers, the gate, model files and local fixtures, choosing checks |
| [Models](models/README.md) | Per model: checks against the reference and experiments |
| [Performance engineering](PERFORMANCE.md) | The optimization loop, measurement contract, profiling and kernel tools |
| [Benchmarking](BENCHMARKING.md) | The benchmark driver, serving measurements, quality method, reporting |
| [Optimizations](OPTIMIZATIONS.md) | Cross-model optimizations: tried, in the code, not tried yet |
| [Server internals](SERVER-INTERNALS.md) | The request model, scheduler, HIP execution and the rules behind the API |
| [KV cache internals](KV-CACHE-INTERNALS.md) | Invariants, code map and how to verify cache changes |
| [Functional tests](../../tests/functional/README.md) | The text API suites against a running server |
| [Tools](../../tools/README.md) | Gate, profiling, benchmark and reference tools |
| [AGENTS.md](../../AGENTS.md) | Instructions for coding agents |

## Prerequisites

The build prerequisites (ROCm 10.1, the Rust toolchain, FFmpeg development
files) are in the [README](../../README.md#build-from-source). The Python
tools (PyTorch with ROCm, transformers, the OpenAI SDK, the pinned
clang-format) install into a virtual environment:

```sh
python3 -m venv .venv
.venv/bin/pip install -r tools/requirements.txt
```

Its PyTorch brings its own ROCm libraries; the `gufo` binary uses the
installed ROCm. `rocprofv3` comes with ROCm. The optional benchmark
baselines (llama.cpp, its Flash-Next MTP fork, ds4 and audio.cpp) build from
pinned commits with `tools/bench/build-reference.sh`.

Model tests need the models' weights ([model files](TESTING.md#model-files)).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/cli` | The `gufo` binary: commands, terminal interfaces and the server (`src/cli/serve`, crate `gufo-serve`) |
| `src/core` | Shared host code: GGUF loading, tokenizers, sampling, grammars, the text runner interface, diagnostics |
| `src/hip` | `gufo-hip`: the HIP runtime layer (memory, streams, graphs, weights, BLAS) |
| `src/ffmpeg` | `gufo-ffmpeg`: video decoding through FFmpeg, loaded on the first video |
| `src/kernel_build` | `gufo-kernel-build`: compiles a kernel crate's HIP sources from its `launch.toml` and generates checked launchers |
| `src/formats` | GGML storage formats and host dequantisation |
| `src/eval` | `gufo eval`: HTTP capability evaluation |
| `src/models/<model>` | One model package (config, weights, engine, sessions, runner) and its kernel crate under `kernels/` |
| `tests` | Fixtures shared across packages, the [functional suites](../../tests/functional/README.md), the DeepSeek operator checks, shared reference tools and the tool tests |
| `tools` | The gate, profiling, benchmarks, offline evaluation and reference runners ([tools](../../tools/README.md)) |
| `docs` | User documentation; `docs/development` this documentation |
| `artifacts`, `models` | Ignored local outputs and weights; never committed |

Within a package, `src/` is production code: Rust and the HIP or C++ the
binary needs. `tests/` holds its tests and fixtures; the programs that record
fixtures or compare against a reference sit in `tests/reference/`,
`reference/` or `tools/` ([reference programs](TESTING.md#reference-programs)),
and Cargo builds none of them. `examples/` holds the Rust side of those
comparisons, which Cargo builds and clippy checks.

## Build

```sh
cargo build --release -p gufo          # target/release/gufo
./target/release/gufo diagnose
```

`rust-toolchain.toml` pins the toolchain. The build takes compiler, headers
and runtime from one ROCm prefix: `$ROCM_PATH`, else `/opt/rocm/core`, else
`/usr` ([design](DESIGN.md#build)). It refuses a ROCm release other than the
qualified 10.1.0 unless `GUFO_ALLOW_UNQUALIFIED_ROCM=1` is set, because
outputs are checked only on that release. Kernel objects are keyed by
compiler and flags, so switching ROCm rebuilds them. `gufo --version` prints
the package version, or `GUFO_RELEASE_VERSION` and `GUFO_REVISION` when they
are set at build time.

## Rules

- **Inference stays equivalent** to each model's C++ reference: the same
  output tokens from the same inputs (for speech, image and embedding models
  the audio codes, latents and vectors). Everything around inference may
  change: JSON member order in responses, audio, image and video encoders,
  and infrastructure a crate can replace. JSON that reaches the prompt or a
  grammar keeps its member order and spelling. A change that alters output
  bits is not merged as an optimization; it is recorded as an option in the
  model's experiments. Details in [relation to Gufo](PARITY.md).
- **Speed, load time and memory** stay at least as good as the reference's,
  by the gates of [performance against Gufo](PARITY.md#performance-against-gufo).
  Successful optimizations become the default path, without extra switches.
- **`unsafe`** lives in `gufo-hip`, `gufo-ffmpeg`, the generated launcher
  modules of the kernel crates and the kernel build's one call into Cargo's
  jobserver; every other crate forbids it (`tools/ci/check-unsafe.py`).
  Changes to `src/hip/`, `src/ffmpeg/`, the shims and any `launch.toml` get
  the review of `unsafe` code ([safety model](DESIGN.md#safety-model)).
- **Qwen control-token spellings** live in `src/core/src/qwen_tokenizer.rs`
  and `src/models/qwen/src/chat_template.rs`, mirrored for Python in
  `tools/gufo/control_tokens.py`. Keep them identical, inspect their uses when
  changing behavior, and keep reference fixtures independent of them.
- **One canonical long option and backend name** per behavior; no aliases.
- **Model code stays with its model:** code, kernels, tests, tools and
  numerical contracts live in its package, and no model calls another
  model's kernels.
- **Comments that cite a C++ file or function** mean the reference's tree:
  Gufo v0.28.1, or for the Gemma 4, Nemotron and OmniVoice packages the
  extension commit their crate documentation names. Kernel sources taken
  from there keep their comments.

Style: Rust follows `rustfmt` and the workspace lints; HIP and C++ follow
[.clang-format](../../.clang-format) (Google style, C++20, two-space indent,
80 columns), checked by `tools/ci/check-format.py`.

## Development loop

Run the smallest check that covers the change ([choosing
checks](TESTING.md#choosing-checks)):

```sh
cargo test -p <package>                                    # host-only tests
cargo test --profile gpu-test -p <package> --features gpu  # GPU tests
cargo test --profile gpu-test -p <package> --features gpu,models --test <name>
python3 tools/ci/check-format.py                           # hand-written C++ and HIP
python3 tools/ci/check-docs.py --root .                    # documentation
tools/ci/check-rust.sh                                     # the scoped gate
```

## Adding code

**A test.** Unit tests sit beside the code; integration tests in the
package's `tests/`. GPU tests are gated by the package's `gpu` feature,
tests that need model files by `models` or `#[ignore]`; a test that needs
weights fails with an explicit message when they are missing. Model QUALITY
checks are `#[ignore]` tests run on demand.

**A kernel.** Add the source to the model's `kernels/launch.toml` with
every launcher's parameters and extents, and probes that keep each launch
inside its declared buffers (`tests/extents.rs` of the kernel crate runs
them). Review a `launch.toml` change like `unsafe` code: a wrong extent
compiles ([`launch.toml`](DESIGN.md#launchtoml)).

**A model.** Create `src/models/<model>` as a package with its kernel crate
under `kernels/`, add both to the workspace and to the GPU aliases in
`.cargo/config.toml`, and keep its tests and tools in the package. Document
it in `docs/models/<model>/README.md` (users: files, use, limits) with a row
in [the model index](../models/README.md) and the README's model table, and in
`docs/development/models/<model>/CHECKS.md` (checks against the reference)
and `EXPERIMENTS.md` (what was tried) with a row in
[the developer model index](models/README.md); `tools/ci/check-docs.py`
requires all three. Its current speed goes to
[performance](../PERFORMANCE.md).

**A dependency.** Record it in
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md);
`tools/ci/check-dependencies.py` validates that shipped and evaluation
components carry license, version and source records, and `cargo deny`
checks [`deny.toml`](../../deny.toml)'s policy.

**Documentation.** User documentation lives in `docs/`, developer
documentation in `docs/development/`. `tools/ci/check-docs.py` enforces the
presence of the maintained documents, resolves every local link and heading
anchor offline, and parses fenced JSON blocks.

## Performance work

Measure with `target/release/gufo` on an otherwise idle machine, keeping
compiler and dependency versions fixed across compared runs. Follow
[performance engineering](PERFORMANCE.md) and the
[kernel optimization skill](../../.agents/skills/optimize-kernel/SKILL.md):
`tools/bench/build.sh` for standalone HIP experiments,
`tools/bench/gfx1151_peak.hip` for measured hardware ceilings,
`tools/prof/prof.py` for pipeline and wall-time profiles,
`tools/prof/isa_mix.py` for instruction analysis and `tools/bench/ab.py` for
interleaved comparisons of two binaries. What a model's change gained or
lost goes into its `EXPERIMENTS.md`, cross-model results into
[optimizations](OPTIMIZATIONS.md), and current numbers into the user
[performance](../PERFORMANCE.md) page.

## Submitting changes

- Follow Conventional Commits with a single-line message and mark breaking
  changes with `!`. Pull request titles take the same form; squash merging
  retains the title on `main`.
- A pull request says which checks ran ([choosing
  checks](TESTING.md#choosing-checks)). Hosted CI runs only the checks that
  need no ROCm; maintainers run the full gate on a gfx1151 before a merge.
- Contributions are licensed under the [MIT license](../../LICENSE-MIT) and
  the [Apache License, Version 2.0](../../LICENSE-APACHE)
  ([CONTRIBUTING.md](../../CONTRIBUTING.md)).
