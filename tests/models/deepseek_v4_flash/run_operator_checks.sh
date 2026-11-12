#!/usr/bin/env bash
# The DeepSeek V4 Flash operator checks (`ds4.projections`, `ds4.attention`,
# `ds4.quantization`): standalone HIP programs that run the kernels of
# kernels/rocm against host references. Built with hipcc as the kernel crate builds kernels (ROCm
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
mmq=src/models/deepseek_v4_flash/kernels/rocm/mmq
status=0
for check in "projections:projection_test.hip moe_test.hip" "attention:attention_test.hip indexer_test.hip" "quantization:quantize_test.hip"; do
  name=${check%%:*}
  read -r -a sources <<<"${check#*:}"
  sources=("${sources[@]/#/$dir/}")
  check_flags=("${flags[@]}")
  if [ "$name" = quantization ]; then
    # The MMQ activation prelude, in the C++17 dialect and defines of the
    # vendored tier (launch.toml).
    sources+=("$mmq/quantize.hip.cpp")
    check_flags+=(-std=c++17 -DGGML_USE_HIP -DGGML_HIP_NO_VMM -DDS4_ROCM_WMMA_MMQ_Y=64)
  fi
  binary=$out/ds4_$name
  if ! "$hipcc" "${check_flags[@]}" "${sources[@]}" -o "$binary" 2>"$binary.log"; then
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
