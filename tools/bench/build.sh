#!/usr/bin/env bash
# Compile the standalone gfx1151 microbenchmarks with hipcc.
#
# Built like the kernel crates build their kernels: compiler, headers and
# runtime from one ROCm prefix ($ROCM_PATH, else /opt/rocm/core, else /usr)
# and the host GCC's installation.
#
#   tools/bench/build.sh [name ...]
#
# With no arguments every *.hip under tools/bench is built into /tmp.
set -euo pipefail

cd "$(dirname "$0")/../.."

rocm=${ROCM_PATH:-/opt/rocm/core}
[ -x "$rocm/bin/hipcc" ] || rocm=/usr
hipcc=${HIPCC:-$rocm/bin/hipcc}
gcc=$(dirname "$("${CC:-cc}" -print-libgcc-file-name)")
inc=(--rocm-path="$rocm" -isystem"$rocm/include" --gcc-install-dir="$gcc")
lib=(-L"$rocm/lib" -Wl,-rpath,"$rocm/lib")

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  for f in tools/bench/*.hip; do
    targets+=("$(basename "$f" .hip)")
  done
fi

for t in "${targets[@]}"; do
  if [[ "$t" == *.hip ]]; then
    src="$t"
    t="$(basename "$t" .hip)"
  else
    src="tools/bench/${t}.hip"
  fi
  out="/tmp/${t}"
  extra=()
  archive=
  if grep -q "hipblas" "$src"; then
    extra+=(-lhipblas -lhipblaslt -lrocblas)
  fi
  echo "==> $src -> $out"
  compile=("$hipcc" -O3 --offload-arch=gfx1151 -std=c++20
    -I. "${inc[@]}" -Rpass-analysis=kernel-resource-usage)
  if [[ "$src" -ef tools/qwen27b/dflash_gemm_bench.hip ]]; then
    # Exact projection comparisons use the production compiler optimization level.
    compile[1]=-O2
  fi
  if [[ "$src" -ef tools/qwen27b/verify_sweep/sweep.hip ]]; then
    # The sweep times the production dispatch: the Qwen kernel crate's
    # archive of the release build, at the production optimization level.
    compile[1]=-O2
    compile+=(-DENGINE_ENABLE_HIP=1)
    archive=$(ls -t target/release/build/gufo-qwen-kernels-*/out/libgufo_kernels_qwen.a \
      2>/dev/null | head -1)
    if [ -z "$archive" ]; then
      echo "$src needs the release build: cargo build --release -p gufo" >&2
      exit 1
    fi
    extra+=("$archive")
  fi
  native_sources=()
  if [[ "$src" -ef tools/qwen27b/dflash_gemm_bench.hip ||
        "$src" -ef tools/qwen27b/prefill_gemm_bench.hip ]]; then
    native_sources+=(src/models/qwen/hip/kernels/small_batch_wave64.hip)
    native_sources+=(src/models/qwen/hip/kernels/small_batch_quant16_wave64.hip)
  fi
  if [[ "$src" -ef tools/qwen27b/prefill_gemm_bench.hip ]]; then
    native_sources+=(src/models/qwen/hip/kernels/prefill_quant_wave64.hip)
  fi
  if [[ "$src" -ef tools/qwen27b/attention_bench.hip ]]; then
    native_sources+=(
      src/models/qwen/hip/kernels/attention_decode_verify_wave64.hip)
  fi
  if ((${#native_sources[@]})) || [ -n "$archive" ]; then
    # HIP helpers and their callers must share the native wave size, and an
    # archive is a link input, which the compile step would read as HIP source.
    objects="$(mktemp -d "/tmp/${t}.XXXXXX")"
    trap 'rm -rf "$objects"' EXIT
    (
      "${compile[@]}" -c "$src" -o "$objects/main.o" || exit 1
      native_objects=()
      for unit in "${native_sources[@]}"; do
        object="$objects/$(basename "$unit").o"
        native_extra=()
        if [[ "$unit" == src/models/qwen/hip/kernels/small_batch_quant16_wave64.hip ]]; then
          native_extra+=(-Xarch_device -mllvm=-amdgpu-sched-strategy=iterative-ilp)
        fi
        # The kernel crate's flag for this unit (src/models/qwen/kernels/launch.toml).
        if [[ "$unit" == src/models/qwen/hip/kernels/prefill_quant_wave64.hip ]]; then
          native_extra+=(-fno-slp-vectorize)
        fi
        "${compile[@]}" -DENGINE_ENABLE_HIP=1 -mwavefrontsize64 \
          "${native_extra[@]}" \
          -c "$unit" -o "$object" || exit 1
        native_objects+=("$object")
      done
      "$hipcc" "${inc[@]}" --offload-arch=gfx1151 "${lib[@]}" \
        "$objects/main.o" "${native_objects[@]}" "${extra[@]}" -o "$out"
    ) >"/tmp/${t}.build.log" 2>&1 ||
      { tail -40 "/tmp/${t}.build.log"; exit 1; }
    rm -rf "$objects"
    trap - EXIT
  else
    "${compile[@]}" "${lib[@]}" \
      "$src" "${extra[@]}" -o "$out" >"/tmp/${t}.build.log" 2>&1 ||
      { tail -40 "/tmp/${t}.build.log"; exit 1; }
  fi
  grep -E "Function Name|VGPRs:|Occupancy|VGPRs Spill|LDS Size" \
    "/tmp/${t}.build.log" | sed 's/.*remark: //; s/ \[-Rpass.*//' \
    >"/tmp/${t}.res.txt" || true
done
echo "ok"
