#!/usr/bin/env bash
# Local gate for the Rust workspace (docs/development/TESTING.md, "Choosing
# checks"). Needs the ROCm prefix the workspace builds against and a gfx1151.
#
#   tools/ci/check-rust.sh             scoped: tests of the packages the changes
#                                      reach, without the long test binaries
#   tools/ci/check-rust.sh --long      scoped, with the long test binaries
#   tools/ci/check-rust.sh --full      every test (before a push; long)
#   tools/ci/check-rust.sh --base REV  scope against REV instead of HEAD, e.g.
#                                      the revision a fold or rebase started from
#
# Test binaries count as long by their local time: every run records
# each binary's seconds in $GUFO_GATE_DURATIONS (default
# target/gate-durations.tsv, outside Git), and the scoped gate skips those over
# $GUFO_GATE_LONG_SECONDS (default 60). It ends with the time of each step and
# this run's binaries over that limit.
set -euo pipefail
cd "$(dirname "$0")/../.."

full=0
long=0
base=HEAD
while [ $# -gt 0 ]; do
  case $1 in
    --full) full=1 ;;
    --long) long=1 ;;
    --base) base=${2:?--base needs a revision}; shift ;;
    *) echo "usage: $0 [--full | --long] [--base REV]" >&2; exit 2 ;;
  esac
  shift
done

durations=
step_name=
step_start=$SECONDS
step() {
  if [ -n "$step_name" ]; then
    durations+=$(printf '%5ds  %s\n' $((SECONDS - step_start)) "$step_name")$'\n'
  fi
  step_name=$1
  step_start=$SECONDS
  printf '\n== %s\n' "$1"
}
times=$PWD/target/gate-times.tsv
export GUFO_GATE_DURATIONS=${GUFO_GATE_DURATIONS:-$PWD/target/gate-durations.tsv}
export GUFO_GATE_LONG_SECONDS=${GUFO_GATE_LONG_SECONDS:-60}

# An explicit --full runs the long tests; a scoped run that reaches every
# package runs them all but the long ones, unless --long.
skip_long=$((1 - long))
[ "$full" = 1 ] && skip_long=0
packages=
if [ "$full" = 0 ]; then
  step "test scope (changes against $base)"
  packages=$(python3 tools/ci/gate_scope.py --base "$base")
  [ "$packages" = full ] && full=1
fi
without=
[ "$skip_long" = 1 ] && without=", without the long test binaries"

step "rustfmt"
cargo fmt --all --check

step "clippy with GPU features (warnings are errors)"
cargo gpu-clippy --quiet -- -D warnings

# Every test binary runs through tools/ci/gate_runner.sh, which times it and,
# under tools/ci/gate-at.sh, keeps the binaries that passed at this revision in
# an earlier run ($GUFO_GATE_PASSED).
mkdir -p target
: >"$times"
runner=$PWD/tools/ci/gate_runner.sh
export GUFO_GATE_KEPT=$PWD/target/gate-kept.tsv
: >"$GUFO_GATE_KEPT"
if [ -n "${GUFO_GATE_PASSED:-}" ] && [ -s "$GUFO_GATE_PASSED" ]; then
  printf '\nresuming: %s test binaries passed in earlier runs at this revision (%s)\n' \
    "$(wc -l <"$GUFO_GATE_PASSED")" "$GUFO_GATE_PASSED"
fi
if [ "$full" = 1 ]; then
  step "tests with GPU features${without#,}"
  env -u GUFO_GATE_PACKAGES GUFO_GATE_SKIP_LONG=$skip_long GUFO_GATE_TIMES="$times" \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="$runner" \
    cargo gpu-test --quiet
elif [ -n "$packages" ]; then
  # Builds every test as the full gate does, so artifacts stay shared, and
  # runs only the binaries of the selected packages.
  step "tests with GPU features of: $packages$without"
  GUFO_GATE_PACKAGES="$packages" GUFO_GATE_SKIP_LONG=$skip_long \
    GUFO_GATE_TIMES="$times" CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="$runner" \
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
python3 tools/ci/check-format.py

step "unsafe policy"
python3 tools/ci/check-unsafe.py

step "tool tests (A/B harness, gate scope, runner and durations, kernel resources, docs and dependency checks, profiler, serving bench, corpus, DS4 dataset, CLI parser)"
log=$(mktemp)
trap 'rm -f "$log"' EXIT
for test in tests/tools/ab_test.py tests/tools/gate_scope_test.py \
  tests/tools/gate_runner_test.py tests/tools/gate_durations_test.py \
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

step "summary"
printf '\nTime per step:\n%s' "$durations"
# This run's times update the local record the next scoped runs read.
python3 tools/ci/gate_durations.py "$times" "$GUFO_GATE_DURATIONS"
slow=$(awk -F '\t' -v limit="$GUFO_GATE_LONG_SECONDS" '$1 > limit' "$times" | sort -rn)
if [ -n "$slow" ]; then
  printf 'Test binaries over %ss (seconds, package, binary):\n%s\n' \
    "$GUFO_GATE_LONG_SECONDS" "$slow"
fi
if [ -s "$GUFO_GATE_KEPT" ]; then
  printf 'Test binaries kept from earlier runs at this revision: %s\n' \
    "$(wc -l <"$GUFO_GATE_KEPT")"
fi
if [ "$full" = 1 ]; then
  printf '\nPASS: Rust workspace checks (full%s)\n' "$without"
else
  printf '\nPASS: Rust workspace checks (scoped to the changes against %s%s)\n' "$base" "$without"
fi
