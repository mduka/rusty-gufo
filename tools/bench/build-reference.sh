#!/usr/bin/env bash
# Builds a pinned benchmark reference runtime against the installed ROCm
# (`$ROCM_PATH`, else /opt/rocm/core) into target/references/<name>/bin.
#
#   tools/bench/build-reference.sh llama-cpp       # llama-server, llama-bench, ... (Qwen AR/DFlash2)
#   tools/bench/build-reference.sh llama-cpp-mtp   # llama-server-mtp, llama-bench-mtp (Flash-Next MTP)
#   tools/bench/build-reference.sh ds4             # ds4-server, ds4-bench (DeepSeek AR/DSpark)
#   tools/bench/build-reference.sh audio-cpp       # audiocpp_server, audiocpp_cli (ASR/TTS)
#
# Put target/references/<name>/bin on PATH for tools/bench/model-bench.py.
# Each source is checked out at its pinned commit; a checkout whose HEAD
# differs is refused. The references are never part of the gufo binary.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
rocm=${ROCM_PATH:-/opt/rocm/core}
jobs=${JOBS:-$(nproc)}

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
  exit 2
}

[ $# -eq 1 ] || usage
name=$1
case $name in
  llama-cpp)
    url=https://github.com/ggml-org/llama.cpp.git
    rev=68d9053afd4f4d0752ced6187585f862355a40be ;;
  llama-cpp-mtp)
    # The Flash-Next MTP implementation from ggml-org/llama.cpp#28243.
    url=https://github.com/danielhanchen/llama.cpp.git
    rev=6fcaa16f4b360649933a54d1f91ad40ed35c0e11 ;;
  ds4)
    url=https://github.com/antirez/ds4.git
    rev=0aaea5a238fb41a35106a551e73c8409dfb751ac ;;
  audio-cpp)
    url=https://github.com/0xShug0/audio.cpp.git
    rev=e3de8e3f3cbfac55ffa58df71426c41550a8598b ;;
  *) usage ;;
esac

[ -x "$rocm/bin/hipcc" ] || { echo "no hipcc under $rocm; set ROCM_PATH" >&2; exit 1; }
work=$root/target/references/$name
src=$work/src
bin=$work/bin

if [ ! -d "$src/.git" ]; then
  git init -q "$src"
  git -C "$src" remote add origin "$url"
fi
if [ "$(git -C "$src" rev-parse -q --verify HEAD 2>/dev/null)" != "$rev" ]; then
  if [ -n "$(git -C "$src" status --porcelain 2>/dev/null)" ]; then
    echo "$src has local changes; remove it to rebuild" >&2
    exit 1
  fi
  git -C "$src" fetch -q --depth 1 origin "$rev"
  git -C "$src" checkout -q --detach FETCH_HEAD
fi
[ "$(git -C "$src" rev-parse HEAD)" = "$rev" ] || { echo "$src is not at $rev" >&2; exit 1; }
mkdir -p "$bin"

case $name in
  llama-cpp | llama-cpp-mtp)
    cmake -S "$src" -B "$work/build" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_NATIVE=OFF \
      -DLLAMA_HIP_UMA=ON \
      -DCMAKE_HIP_COMPILER="$rocm/llvm/bin/clang" \
      -DCMAKE_HIP_FLAGS="--rocm-path=$rocm -mllvm --amdgpu-unroll-threshold-local=600" \
      -DLLAMA_BUILD_COMMIT:STRING="${rev:0:8}"
    cmake --build "$work/build" -j "$jobs"
    # Distinct names (llama-server-mtp) let both revisions share one PATH.
    suffix=${name#llama-cpp}
    for tool in "$work"/build/bin/llama-*; do
      ln -sf "$tool" "$bin/$(basename "$tool")$suffix"
    done
    ;;
  ds4)
    # ds4's Strix Halo target with a fixed CPU target and no build-time GPU
    # detection.
    includes=""
    for dir in "$rocm/include" "$rocm/include/hipblas" "$rocm/include/hipblaslt" \
      "$rocm/include/rocblas" "$rocm/include/rocwmma" "$rocm/include/hipcub" \
      "$rocm/include/rocprim"; do
      [ -d "$dir" ] && includes+=" -I$dir"
    done
    make -C "$src" strix-halo -j "$jobs" \
      HIPCC="$rocm/bin/hipcc" ROCM_ARCH=gfx1151 NATIVE_CPU_FLAG=-march=znver5 \
      ROCM_CFLAGS="-O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --rocm-path=$rocm --offload-arch=gfx1151$includes" \
      ROCM_LDLIBS="-lm -pthread -L$rocm/lib -Wl,-rpath,$rocm/lib -lhipblas -lhipblaslt -lrocblas"
    for tool in ds4 ds4-server ds4-bench ds4-eval ds4-agent; do
      ln -sf "$src/$tool" "$bin/$tool"
    done
    ;;
  audio-cpp)
    cmake -S "$src" -B "$work/build" -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DENGINE_ENABLE_HIP=ON -DGPU_TARGETS=gfx1151 \
      -DENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=ON \
      -DENGINE_ENABLE_NATIVE_CPU=ON -DENGINE_ENABLE_LLAMAFILE=ON \
      -DAUDIOCPP_MODEL_SET=full \
      -DCMAKE_C_COMPILER="$rocm/llvm/bin/clang" \
      -DCMAKE_CXX_COMPILER="$rocm/llvm/bin/clang++" \
      -DCMAKE_HIP_COMPILER="$rocm/llvm/bin/clang"
    cmake --build "$work/build" -j "$jobs" --target audiocpp_server audiocpp_cli
    for tool in audiocpp_server audiocpp_cli; do
      found=$(find "$work/build" -type f -name "$tool" -perm -u+x | head -n 1)
      [ -n "$found" ] || { echo "$tool was not built" >&2; exit 1; }
      ln -sf "$found" "$bin/$tool"
    done
    ;;
esac
printf '%s %s\n' "$name" "$rev" > "$work/revision"
echo "built $name at ${rev:0:8}: $bin"
