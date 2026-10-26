#!/usr/bin/env bash
# Local gate for the Rust workspace, run before every commit
# (docs/RUST.md). Needs the ROCm
# prefix the workspace builds against and a gfx1151.
#
#   tools/ci/check-rust.sh             scoped: tests of the packages the changes reach
#   tools/ci/check-rust.sh --full      every test (step and milestone ends)
#   tools/ci/check-rust.sh --base REV  scope against REV instead of HEAD
set -euo pipefail
cd "$(dirname "$0")/../.."

full=0
base=HEAD
while [ $# -gt 0 ]; do
  case $1 in
    --full) full=1 ;;
    --base) base=${2:?--base needs a revision}; shift ;;
    *) echo "usage: $0 [--full] [--base REV]" >&2; exit 2 ;;
  esac
  shift
done

step() { printf '\n== %s\n' "$1"; }

packages=
if [ "$full" = 0 ]; then
  step "test scope (changes against $base)"
  packages=$(python3 tools/ci/gate_scope.py --base "$base")
  [ "$packages" = full ] && full=1
fi

step "rustfmt"
cargo fmt --all --check

step "clippy with GPU features (warnings are errors)"
cargo gpu-clippy --quiet -- -D warnings

if [ "$full" = 1 ]; then
  step "tests with GPU features"
  cargo gpu-test --quiet
elif [ -n "$packages" ]; then
  # Builds every test as the full gate does, so artifacts stay shared, and
  # runs only the binaries of the selected packages.
  step "tests with GPU features of: $packages"
  GUFO_GATE_PACKAGES="$packages" \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="$PWD/tools/ci/gate_runner.sh" \
    cargo gpu-test --quiet
else
  step "tests with GPU features: no package reached, skipped"
fi

# Standalone HIP programs over the DeepSeek kernels (ds4.projections,
# ds4.attention); they run when those kernels are in scope.
if [ "$full" = 1 ] || [[ " $packages " == *" gufo-deepseek-v4-flash-kernels "* ]]; then
  step "DeepSeek V4 Flash operator checks"
  log=$(mktemp)
  if ! tests/models/deepseek_v4_flash/run_operator_checks.sh >"$log" 2>&1; then
    cat "$log"
    rm -f "$log"
    exit 1
  fi
  grep -E '^== |skipped$' "$log" | paste -sd ' '
  rm -f "$log"
fi

step "clang-format of the hand-written C++ (CI's pinned formatter)"
# The same check and formatter version as hosted CI (.github/workflows/ci.yml).
if ! command -v nix >/dev/null; then
  echo "nix not found: the C++ format check needs CI's pinned clang-format" >&2
  exit 1
fi
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py

step "unsafe policy"
python3 tools/ci/check-unsafe.py

step "tool tests (A/B harness, gate scope, kernel resources, docs and dependency checks, profiler, serving bench, corpus, DS4 dataset, CLI parser)"
log=$(mktemp)
trap 'rm -f "$log"' EXIT
for test in tests/tools/ab_test.py tests/tools/gate_scope_test.py \
  tests/tools/kernel_resources_test.py tests/tools/check_docs_test.py \
  tests/tools/check_dependencies_test.py; do
  if ! PYTHONDONTWRITEBYTECODE=1 python3 "$test" >"$log" 2>&1; then
    cat "$log"
    exit 1
  fi
  printf '%s: ' "$test"
  grep -E '^(Ran|OK)' "$log" | paste -sd ' '
done
for test in tests/tools/prof_test.py tests/tools/serving_bench_test.py \
  tests/tools/speculative_corpus_test.py; do
  if ! PYTHONDONTWRITEBYTECODE=1 python3 "$test" >"$log" 2>&1; then
    cat "$log"
    exit 1
  fi
  echo "$test: $(tail -1 "$log")"
done
if ! python3 tests/models/deepseek_v4_flash/dataset_test.py >"$log" 2>&1; then
  cat "$log"
  exit 1
fi
echo "tests/models/deepseek_v4_flash/dataset_test.py: OK"
# The CLI parser of the binary the tests just built.
if [ "$full" = 1 ] || [ -n "$packages" ]; then
  if ! python3 tests/cli/serve_test.py target/gpu-test/gufo >"$log" 2>&1; then
    cat "$log"
    exit 1
  fi
  echo "tests/cli/serve_test.py: $(tail -1 "$log")"
fi

# A compiler update can make a kernel spill with no source change. The check
# reads the code objects of the binary the tests just built (no GPU) and
# fails on scratch beyond tools/ci/kernel-resources.json.
if [ "$full" = 1 ] || [ -n "$packages" ]; then
  step "kernel scratch against tools/ci/kernel-resources.json"
  python3 tools/ci/check-kernel-resources.py target/gpu-test/gufo \
    --baseline tools/ci/kernel-resources.json
fi

step "documentation and dependency inventory"
python3 tools/ci/check-docs.py --root .
python3 tools/ci/check-dependencies.py

if [ "$full" = 1 ]; then
  printf '\nPASS: Rust workspace checks (full)\n'
else
  printf '\nPASS: Rust workspace checks (scoped to the changes against %s)\n' "$base"
fi
