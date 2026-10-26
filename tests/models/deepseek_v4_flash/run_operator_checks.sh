#!/usr/bin/env bash
# The DeepSeek V4 Flash operator checks (`ds4.projections`, `ds4.attention`):
# standalone HIP programs that run the kernels of kernels/rocm against host
# references. Built with hipcc as the kernel crate builds kernels (ROCm
# prefix and GCC installation), with the optimisation they were written
# for, into target/hip-checks. Needs a gfx1151; exit 77 means skipped.
set -euo pipefail
cd "$(dirname "$0")/../../.."

rocm=${ROCM_PATH:-/opt/rocm/core}
[ -x "$rocm/bin/hipcc" ] || rocm=/usr
hipcc=${HIPCC:-$rocm/bin/hipcc}
gcc=$(dirname "$("${CC:-cc}" -print-libgcc-file-name)")
out=target/hip-checks
mkdir -p "$out"

flags=(--rocm-path="$rocm" -isystem"$rocm/include" --gcc-install-dir="$gcc"
  -x hip -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -O3 -ffast-math -fno-finite-math-only
  -I. -Wl,-rpath,"$rocm/lib")
dir=tests/models/deepseek_v4_flash
status=0
for check in "projections:projection_test.hip moe_test.hip" "attention:attention_test.hip indexer_test.hip"; do
  name=${check%%:*}
  read -r -a sources <<<"${check#*:}"
  binary=$out/ds4_$name
  if ! "$hipcc" "${flags[@]}" "${sources[@]/#/$dir/}" -o "$binary" 2>"$binary.log"; then
    cat "$binary.log"
    exit 1
  fi
  echo "== ds4.$name"
  set +e
  "$binary"
  code=$?
  set -e
  case $code in
    0) ;;
    77) echo "ds4.$name skipped" ;;
    *) echo "ds4.$name failed ($code)"; status=1 ;;
  esac
done
exit $status
