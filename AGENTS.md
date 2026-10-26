# AGENTS.md

## Platform and build

Production targets Linux x86-64 AMD Strix Halo (`gfx1151`) only. Gufo is a
Cargo workspace built against an installed ROCm (`$ROCM_PATH`, else
`/opt/rocm/core`, else `/usr`); `rust-toolchain.toml` pins the toolchain.
Linux source-build prerequisites are in [README.md](README.md#build-from-source).

```sh
cargo build --release -p gufo          # production binary
./target/release/gufo diagnose
nix develop                            # profiling, reference and client tools
```

Build and measure outside `nix develop`: its ROCm is not the one the binary
links. Measure performance with `target/release/gufo`. Preserve
compiler/dependency versions when comparing results.

## Focused tests

`tools/ci/check-rust.sh` is the gate before every commit: formatting,
clippy, the GPU and model tests of the packages a change reaches, the
unsafe policy, the tool tests and the documentation and dependency checks.
`--full` runs every test, at the end of each larger step. Formatting and Python
repository checks may run anywhere; GPU tests need the gfx1151. Before
committing hand-written C++ (kernels, `src/hip/shim`), run:

```sh
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
# Add --fix to apply formatting, then rerun the check.
```

Run the smallest check covering the change during iteration:

```sh
cargo test -p <package>                                    # host-only tests
cargo test --profile gpu-test -p <package> --features gpu  # GPU tests
cargo test --profile gpu-test -p <package> --features gpu,models --test <name>
```

`gpu-test` is optimised like release, with debug assertions and overflow
checks. Model QUALITY checks are `#[ignore]` tests; see
[docs/TESTING.md](docs/TESTING.md). Do not run full model sweeps or
duplicate suites for routine edits. A missing-model failure is not a
quality pass. Broaden checks when shared behavior changes or failures
expose risk.

Text API functional/regression tests live in [tests/functional/](tests/functional/README.md)
and run against `target/release/gufo`. Select suites explicitly. Model-specific changes need the affected model/modes;
use all four profiles only for shared behavior changes. Run correctness on the
candidate; use main for matched affected timing controls and suspected regressions,
not a duplicate full correctness run. Reuse baseline evidence only when revision,
toolchain, harness, model, settings and cache history match. Include `long-context`
and `cache` for continuation changes; verify loaded mode and actual drafts.
Correctness and expected prefill/cache work are strict; timings are
mandatory per request/phase at 5% and 3 ms, never averaged across requests.
Run once, investigate flags, then alternate main/PR only for affected histories;
use unchanged-main controls when needed and retain every result. Fix confirmed
regressions before publishing. Noisy evidence stays visibly inconclusive and
unqualified; do not widen margins or stop at reporting failures. Use the code
diff to select affected metrics; unrelated timing variance does not justify
another full matrix.
For reported coding-agent loops, also replay the affected workflow with
`tests/functional/pi_agent.py`; retain its sessions and inspect actual tool results.
Retain numerical quality tests and the standard speed benchmark. Keep these
tests in `tests/functional/`, outside hosted CI, which has no ROCm or gfx1151
([.github/workflows/ci.yml](.github/workflows/ci.yml)); avoid full sweeps.

## Profiling and kernels

Apply [.agents/skills/optimize-kernel/SKILL.md](.agents/skills/optimize-kernel/SKILL.md).
Use `tools/bench/build.sh` for standalone HIP experiments,
`tools/bench/gfx1151_peak.hip` for measured hardware ceilings,
`tools/prof/prof.py` for pipeline/wall-time profiles, and
`tools/prof/isa_mix.py` for instruction analysis. Production paths must retain
quality; successful optimizations become the default, without extra switches.
Kernel launchers are declared in each kernel crate's `launch.toml`, with
extent probes for every launcher.

## Development

- Keep model code, tests, tools and numerical contracts with their model.
  In a package, `src/` holds production code, `tests/` its tests and
  fixtures, `tests/reference/` the programs that check against upstream's
  C++ (or another reference) or record reference data, and `examples/` the
  Rust side of those dumps. The repository's `tools/` holds development
  tools, benchmarks and profilers.
- Qwen control-token spellings live in `src/core/src/qwen_tokenizer.rs` and
  `src/models/qwen/src/chat_template.rs`, mirrored for Python in
  `tools/gufo/control_tokens.py`. Keep them identical, inspect their uses
  when reviewing or changing behavior, and keep reference fixtures independent.
- Use one canonical long option and backend name per behavior; avoid aliases.
- `unsafe` lives in `gufo-hip`, the generated launcher modules and the
  kernel build's one call into Cargo's jobserver (`tools/ci/check-unsafe.py`).
- Port upstream changes as [Gufo in Rust](docs/RUST.md#following-upstream)
  describes: kernels keep upstream's paths and arithmetic, host code follows
  the C++ order. Comments that cite a C++ file mean upstream's tree at the
  reference release.
- Use `gh` for GitHub operations after checking `gh auth status`.
- Follow Conventional Commits with a single-line message. Title pull requests
  according to [the release policy](docs/RELEASING.md); squash merging retains
  that title on `main`. Mark breaking changes with `!` in the title, and leave
  version and changelog updates to the release workflow.
- Prefer `jj` when available (`jj version`); otherwise use Git.
- Follow the user's remote workflow and preserve unrelated work.
