# AGENTS.md

Instructions for coding agents working on rusty-gufo. The contributor guide,
[docs/development/README.md](docs/development/README.md), applies in full:
read its rules before changing code.

## Build and check

Production targets Linux x86-64 on AMD Strix Halo (`gfx1151`) with ROCm
10.1; GPU tests need the device.

```sh
cargo build --release -p gufo                              # the binary
cargo test -p <package>                                    # host-only tests
cargo test --profile gpu-test -p <package> --features gpu  # GPU tests
cargo test --profile gpu-test -p <package> --features gpu,models --test <name>
tools/ci/check-rust.sh                                     # the scoped gate
python3 tools/ci/check-format.py                           # before committing hand-written C++/HIP
python3 tools/ci/check-docs.py --root .                    # after documentation changes
```

- Run the smallest check that covers the change
  ([choosing checks](docs/development/TESTING.md#choosing-checks)). Do not
  run full model sweeps or duplicate suites for routine edits; broaden
  checks when shared behavior changes or a failure exposes risk.
- After a rebase or squash, scope the gate against the revision that passed
  (`tools/ci/check-rust.sh --base <rev>`) instead of rerunning everything.
- A failure caused by a missing model or device is not a quality pass.
- Measure performance with `target/release/gufo` on an otherwise idle
  machine, with the same compiler and dependency versions on both sides.
- Text API changes need the affected functional suites and their timing
  protocol ([functional tests](tests/functional/README.md)); for reported
  coding-agent loops, also replay the workflow with
  `tests/functional/pi_agent.py` and inspect the actual tool results.
- For kernel and profiling work, apply
  [.agents/skills/optimize-kernel/SKILL.md](.agents/skills/optimize-kernel/SKILL.md).

## Rules that are easy to miss

- Output equivalence: the same output tokens as the model's C++ reference
  ([relation to Gufo](docs/development/PARITY.md)). An optimization that
  changes output bits is not merged as one.
- `unsafe` stays in `gufo-hip`, `gufo-ffmpeg`, the generated launchers and
  the kernel build's jobserver call (`tools/ci/check-unsafe.py`).
- Qwen control-token spellings are mirrored in three files (see the
  contributor guide); keep them identical.
- One canonical long option and backend name per behavior; no aliases.

## Git

- Conventional Commits with a single-line message; mark breaking changes
  with `!`. Pull request titles take the same form.
- Stage only the paths the task owns and preserve unrelated work.
- Prefer `jj` when available (`jj version`); otherwise use Git. Use `gh` for
  GitHub operations after checking `gh auth status`.
