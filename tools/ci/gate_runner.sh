#!/bin/sh
# Test runner of the gate (tools/ci/check-rust.sh): Cargo builds every test
# as in the full gate and runs each test binary through this script. Cargo
# sets CARGO_PKG_NAME for each test it runs.
#
#   GUFO_GATE_PACKAGES     when set, only the binaries of these packages run
#                          (the scoped gate)
#   GUFO_GATE_SKIP_LONG    1: skip a binary whose recorded time is over
#                          GUFO_GATE_LONG_SECONDS
#   GUFO_GATE_DURATIONS    the local record of binary times
#                          (tools/ci/gate_durations.py)
#   GUFO_GATE_TIMES        a file that gets one line per binary run: seconds,
#                          package, binary
if [ -n "${GUFO_GATE_PACKAGES+set}" ]; then
  case " $GUFO_GATE_PACKAGES " in
  *" $CARGO_PKG_NAME "*) ;;
  *) exit 0 ;;
  esac
fi
# The test target's name: the binary's without Cargo's metadata hash.
binary=$(basename "$1" | sed 's/-[0-9a-f]\{16\}$//')
if [ "${GUFO_GATE_SKIP_LONG:-0}" = 1 ] && [ -f "${GUFO_GATE_DURATIONS:-}" ]; then
  seconds=$(awk -F '\t' -v package="$CARGO_PKG_NAME" -v binary="$binary" \
    -v limit="${GUFO_GATE_LONG_SECONDS:-60}" \
    '!/^#/ && $2 == package && $3 == binary && $1 > limit { print $1 }' \
    "$GUFO_GATE_DURATIONS")
  if [ -n "$seconds" ]; then
    echo "gate: skipped $CARGO_PKG_NAME $binary (${seconds}s locally; --long runs it)" >&2
    exit 0
  fi
fi
start=$(date +%s.%N)
"$@"
code=$?
if [ -n "${GUFO_GATE_TIMES:-}" ]; then
  awk -v start="$start" -v end="$(date +%s.%N)" -v package="$CARGO_PKG_NAME" \
    -v binary="$binary" \
    'BEGIN { printf "%.1f\t%s\t%s\n", end - start, package, binary }' >>"$GUFO_GATE_TIMES"
fi
exit $code
