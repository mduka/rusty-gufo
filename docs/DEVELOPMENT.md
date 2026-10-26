# Development

Production targets Linux x86-64 on AMD Strix Halo (`gfx1151`). Gufo is a Cargo
workspace: Rust host code over HIP kernels, which each model's kernel crate
compiles with hipcc. System dependencies and the qualified ROCm version are
listed in [the README](../README.md#build-from-source). Repository rules for
agents and contributors are in [AGENTS.md](../AGENTS.md). The
page [Gufo in Rust](RUST.md) explains the Rust design, its safety model and
how it differs from and follows upstream.

## Repository layout

| Path | Ownership |
| --- | --- |
| `src/cli` | The `gufo` binary: commands, terminal interfaces and the server (`src/cli/serve`, crate `gufo-serve`) |
| `src/core` | Shared host code: GGUF loading, tokenizers, sampling, chat requests, diagnostics |
| `src/hip` | `gufo-hip`: the HIP runtime, memory, streams, graphs, BLAS; the only crate with `unsafe` |
| `src/kernel_build` | `gufo-kernel-build`: compiles a kernel crate's HIP sources from its `launch.toml` and generates checked launchers |
| `src/formats` | GGUF types and host dequantisation |
| `src/models/<model>` | One model package: config, weights, engine, sessions, runner; `kernels/` holds its kernel crate and HIP sources |
| `src/eval` | HTTP capability evaluation |
| `tests` | Fixtures shared across packages, the functional suites, the DeepSeek operator checks and the tool tests |
| `tools` | The gate, profiling, benchmarks, offline evaluation and reference runners |
| `docs` | Cross-cutting guides; `docs/models/<model>` holds per-model documents |
| `.devops/nix` | Reference packages for benchmarks |
| `artifacts`, `models` | Ignored local outputs and weights; never committed |

Model code, kernels, tests, tools and numerical contracts stay with their model.
A model's kernels are its own: no model calls another model's kernels (the
Qwen language models share `gufo-qwen-kernels`), so a kernel change cannot
alter another model's output or speed.

Within a package, `src/` is production code: Rust and the HIP or C++ the
binary needs. `tests/` holds its tests and fixtures, and `tests/reference/`
the programs that compare against upstream's C++ engine (or another
reference, such as ICU or libwebp) or record the fixtures; Cargo builds none
of them. `examples/` holds the Rust side of those dumps, which Cargo builds
and clippy checks.

## Build

```sh
cargo build --release -p gufo          # target/release/gufo
./target/release/gufo diagnose
```

`rust-toolchain.toml` pins the toolchain. The build takes compiler, headers
and runtime from one ROCm prefix: `$ROCM_PATH`, else `/opt/rocm/core`, else
`/usr` (details in [Gufo in Rust](RUST.md#build)).
Kernel objects are keyed by compiler and flags, so switching ROCm rebuilds
them. `GUFO_RELEASE_VERSION` and `GUFO_REVISION` set the version string.

`nix develop` provides profiling, reference and client tools (PyTorch with
ROCm, the OpenAI SDK, rocprofiler). Its ROCm is not the binary's, so build,
test and measure Gufo outside it.

## The gate

Run `tools/ci/check-rust.sh` before every commit. It checks formatting,
clippy with warnings as errors, the GPU and model tests of the packages the
change reaches and their dependents, the DeepSeek operator checks when their
kernels change, clang-format of the hand-written C++, the unsafe policy, the
tool tests, the documentation and the dependency inventory. `--full` runs
every test, for larger steps and whenever a change reaches further than its
files. The pieces:

```sh
cargo gpu-test                         # every test with GPU features (.cargo/config.toml)
cargo gpu-clippy
cargo fmt --all --check
python3 tools/ci/check-unsafe.py
python3 tools/ci/gate_scope.py         # the packages a scoped run tests
python3 tests/tools/ab_test.py         # A/B harness, no GPU needed
python3 tools/ci/check-docs.py --root .
```

## Development loop

Run the smallest check that covers the change. Formatting and the Python
repository checks need no GPU.

```sh
cargo test -p <package>                                    # host-only tests
cargo test --profile gpu-test -p <package> --features gpu  # GPU tests
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
```

`gpu-test` is optimised like release, with debug assertions and overflow
checks. [Testing](TESTING.md) maps change categories to the checks that
cover them. Do not run full model sweeps or duplicate suites for routine
edits; a failure caused by a missing model or device is not a quality pass.

## Conventions

- Rust follows `rustfmt` and the workspace lints; crates other than
  `gufo-hip` forbid `unsafe` (kernel crates allow it only on their
  generated launchers).
- HIP and C++ follow [.clang-format](../.clang-format): Google style,
  C++20, two-space indent, 80 columns.
- One canonical long option and backend name per behavior; no aliases.
- Successful optimizations become the default path, without extra switches, and
  production paths retain quality.

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
compiles.

**A model.** Create `src/models/<model>` as a package with its kernel crate
under `kernels/`, add both to the workspace, keep its tests and tools in
the package, add a row to [the model index](models/README.md) and the four
required documents `README.md`, `BENCHMARKS.md`, `QUALITY.md` and
`EXPERIMENTS.md` under `docs/models/<model>/`.

**A dependency.** Record it in `THIRD_PARTY_NOTICES.md`;
`tools/ci/check-dependencies.py` validates that shipped and evaluation
components carry license, version and source records.

**Documentation.** `tools/ci/check-docs.py` enforces the presence of the
maintained documents, resolves every local link and heading anchor offline, and
parses fenced JSON blocks.

## Performance work

Measure with `target/release/gufo`, keeping compiler and dependency versions
fixed across compared runs. Follow [performance engineering](PERFORMANCE.md)
and the [kernel optimization skill](../.agents/skills/optimize-kernel/SKILL.md).
Use `tools/bench/build.sh` for standalone HIP experiments,
`tools/bench/gfx1151_peak.hip` for measured hardware ceilings,
`tools/prof/prof.py` for pipeline and wall-time profiles,
`tools/prof/isa_mix.py` for instruction analysis and `tools/bench/ab.py` for
interleaved comparisons of two binaries. Retained numbers belong in
`docs/models/<model>/BENCHMARKS.md` under the rules in
[benchmarking](BENCHMARKS.md).

## Submitting changes

Hosted runners have no gfx1151, so hosted CI runs only the checks that need
no ROCm ([testing](TESTING.md#local-checks)). The local gate is the check.

- Prefer `jj` when available (`jj version`); otherwise use Git.
- Follow Conventional Commits with a single-line message. Pull request
  titles follow [the release policy](RELEASING.md); squash merging retains
  the title on `main`.
- Use `gh` for GitHub operations after checking `gh auth status`.
- Preserve unrelated work and stage only task-owned paths.

## Related documents

[CLI](CLI.md), [server](SERVER.md), [testing](TESTING.md),
[releasing](RELEASING.md), [performance](PERFORMANCE.md), [benchmarking](BENCHMARKS.md),
[models](models/README.md) and [Gufo in Rust](RUST.md).
