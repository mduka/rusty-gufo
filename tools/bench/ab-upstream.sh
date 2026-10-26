#!/usr/bin/env bash
# The A/B runs of docs/RUST.md ("Performance against upstream"): the Rust build
# against a C++ build of the upstream reference release, per workload of the
# results table, with ab.py's default gates (request and stage times within
# 0.5%, load within 5%, peak memory within 1%).
#
#   A=<C++ gufo> [B=target/release/gufo] [OUT=target/ab/upstream] \
#     tools/bench/ab-upstream.sh [NAME...]        (no names: every workload)
#
# A workload runs as whole runs of one binary (`ab.py record`: a warmup round,
# then the workload's rounds, one process each), the first time in AABB
# order: A twice, then B twice, so each side has a first run and a second that
# may profit from the page cache. A's runs are the baseline, kept in
# $OUT/baseline-<A's SHA-256>/<name>-<SHA-256 of rounds, repeat and workload
# arguments> and reused while A and the workload stay the same: later
# candidates record only BB. `ab.py judge` judges each B run on its own
# against the pooled A runs. When the two runs of a side differ by more than
# a gate's tolerance, or only one B run passes, a third run of that side
# follows, and two of three B runs must pass. Every invocation keeps its B
# runs and verdicts (judge-<runs>.json and .log) in a new
# $OUT/candidate-<B's SHA-256>-<time>/<name>.
#
# The APU runs at its power limit, so a hotter GPU clocks lower and a run
# that follows another one is slower. The first and the third run of each
# side therefore wait until the GPU has cooled to COOL °C (default 50, 0: no
# wait; at most COOL_TIMEOUT seconds, default 300), and the second follows
# the first directly on both sides: pAA(pA) pBB(pB). Each run records the
# temperature it started at.
#
# Models come from the environment; a workload whose model is unset is
# skipped:
#   GUFO_QWEN27B_MODEL (UD-Q4_K_XL), GUFO_QWEN27B_Q8_MODEL (UD-Q8_K_XL),
#   GUFO_QWEN27B_DFLASH (the DFlash2 draft)
#   GUFO_FLASH_NEXT_MODEL, GUFO_FLASH_NEXT_MTP_MODEL
#   GUFO_DEEPSEEK_V4_FLASH_MODEL, GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL
#   GUFO_QWEN3_ASR_MODEL, GUFO_QWEN3_ASR_AUDIO (an English recording, WAV)
#   GUFO_QWEN3_TTS_CUSTOMVOICE, GUFO_QWEN3_TTS_VOICEDESIGN (checkpoint dirs)
#   GUFO_QWEN_IMAGE_MODEL, GUFO_EMBEDDING_GEMMA2_MODEL (checkpoint dirs)
# Run outside `nix develop`, with nothing else on the GPU.
set -u
cd "$(dirname "$0")/../.."
A=${A:?set A to the C++ reference binary (docs/RUST.md, "The C++ reference")}
B=${B:-target/release/gufo}
OUT=${OUT:-target/ab/upstream}
digest() { # file: the first 12 hex digits of its SHA-256
  local sum
  sum=$(sha256sum "$1") || return
  echo "${sum:0:12}"
}
A_SHA=$(digest "$A") && B_SHA=$(digest "$B") || exit 2
CANDIDATE=$OUT/candidate-$B_SHA-$(date +%Y%m%dT%H%M%S)
mkdir -p "$OUT" && mkdir "$CANDIDATE" || exit 2
SELECTED=("$@")
failed=0

COOL=${COOL:-50} COOL_TIMEOUT=${COOL_TIMEOUT:-300}
record() { # run --baseline|--candidate binary rounds repeat workload args...
  local run=$1 cool=()
  case $(basename "$run") in
    A1 | A3 | B1 | B3) [ "$COOL" = 0 ] || cool=(--cool-to "$COOL" --cool-timeout "$COOL_TIMEOUT") ;;
  esac
  echo "   $(basename "$run") $(date +%T)"
  python3 tools/bench/ab.py record "$2" "$3" "${cool[@]}" --rounds "$4" --repeat "$5" \
    --output "$run.json" "${@:6}" >"$run.log" 2>&1
}

finish() { # name status log
  [ "$2" -eq 0 ] || failed=1
  echo "$1 exit=$2 $(date +%T)" | tee -a "$CANDIDATE/status.log"
  tail -n 40 "$3"
}

ab() { # name rounds repeat VARIABLE... -- workload args...
  local name=$1 rounds=$2 repeat=$3 missing=()
  shift 3
  while [ "$1" != "--" ]; do
    [ -n "${!1:-}" ] || missing+=("$1")
    shift
  done
  shift
  if [ ${#SELECTED[@]} -ne 0 ] && [[ " ${SELECTED[*]} " != *" $name "* ]]; then
    return
  fi
  if [ ${#missing[@]} -ne 0 ]; then
    echo "== $name skipped: ${missing[*]} unset"
    return
  fi
  echo "== $name $(date +%T)"
  local extra=()
  # hipBLASLt timing picks the embedding GEMMs per load on both sides, so
  # its outputs repeat within a side only (the workload checks that).
  case $name in embedding) extra=(--allow-output-mismatch) ;; esac
  local base dir status i last baseline=() runs=()
  base=$OUT/baseline-$A_SHA/$name-$(printf '%s\0' "$rounds" "$repeat" "$@" | sha256sum | cut -c1-12)
  dir=$CANDIDATE/$name
  mkdir -p "$base" "$dir"
  for i in 1 2; do
    if [ -f "$base/A$i.json" ]; then
      echo "   A$i stored in $base"
    elif ! record "$base/A$i" --baseline "$A" "$rounds" "$repeat" "$@"; then
      finish "$name" 2 "$base/A$i.log"
      return
    fi
    baseline+=(--baseline-run "$base/A$i.json")
  done
  if [ -f "$base/A3.json" ]; then
    echo "   A3 stored in $base"
  else
    # ab.py judge exits 3 when the two baseline runs differ.
    python3 tools/bench/ab.py judge "${extra[@]}" "${baseline[@]}" >"$base/check.log" 2>&1
    status=$?
    if [ $status -eq 3 ]; then
      echo "   A1 and A2 differ"
      if ! record "$base/A3" --baseline "$A" "$rounds" "$repeat" "$@"; then
        finish "$name" 2 "$base/A3.log"
        return
      fi
    elif [ $status -ne 0 ]; then
      finish "$name" $status "$base/check.log"
      return
    fi
  fi
  [ ! -f "$base/A3.json" ] || baseline+=(--baseline-run "$base/A3.json")
  for i in 1 2; do
    if ! record "$dir/B$i" --candidate "$B" "$rounds" "$repeat" "$@"; then
      finish "$name" 2 "$dir/B$i.log"
      return
    fi
    runs+=(--candidate-run "$dir/B$i.json")
  done
  last=$dir/judge-2
  python3 tools/bench/ab.py judge "${extra[@]}" "${baseline[@]}" "${runs[@]}" \
    --output "$last.json" >"$last.log" 2>&1
  status=$?
  # 3: B1 and B2 differ, or only one of them passes.
  if [ $status -eq 3 ]; then
    echo "   B1 and B2 disagree"
    if ! record "$dir/B3" --candidate "$B" "$rounds" "$repeat" "$@"; then
      finish "$name" 2 "$dir/B3.log"
      return
    fi
    runs+=(--candidate-run "$dir/B3.json")
    last=$dir/judge-3
    python3 tools/bench/ab.py judge "${extra[@]}" "${baseline[@]}" "${runs[@]}" \
      --output "$last.json" >"$last.log" 2>&1
    status=$?
  fi
  finish "$name" $status "$last.log"
}

Q4=${GUFO_QWEN27B_MODEL:-} Q8=${GUFO_QWEN27B_Q8_MODEL:-} DF=${GUFO_QWEN27B_DFLASH:-}
FN=${GUFO_FLASH_NEXT_MODEL:-} MTP=${GUFO_FLASH_NEXT_MTP_MODEL:-}
DS=${GUFO_DEEPSEEK_V4_FLASH_MODEL:-} DSP=${GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL:-}

# Text: prompt processing and generation, autoregressive and speculative.
ab q27-q4-ar 8 3 GUFO_QWEN27B_MODEL -- \
  text --model "$Q4" --prompt-tokens 512 --gen-tokens 128
ab q27-q4-dflash2 8 3 GUFO_QWEN27B_MODEL GUFO_QWEN27B_DFLASH -- \
  text --model "$Q4" --dflash-model "$DF" --prompt-tokens 512 --gen-tokens 128
ab q27-q8-ar 6 3 GUFO_QWEN27B_Q8_MODEL -- \
  text --model "$Q8" --prompt-tokens 512 --gen-tokens 128
ab q27-q8-dflash2 6 3 GUFO_QWEN27B_Q8_MODEL GUFO_QWEN27B_DFLASH -- \
  text --model "$Q8" --dflash-model "$DF" --prompt-tokens 512 --gen-tokens 128
ab q27-q4-ar-d32k 6 2 GUFO_QWEN27B_MODEL -- \
  text --model "$Q4" --prompt-tokens 512 --gen-tokens 64 --depth 32768
ab fn-ar 8 3 GUFO_FLASH_NEXT_MODEL -- \
  text --model "$FN" --prompt-tokens 512 --gen-tokens 128
ab fn-mtp 8 3 GUFO_FLASH_NEXT_MODEL GUFO_FLASH_NEXT_MTP_MODEL -- \
  text --model "$FN" --mtp-model "$MTP" --prompt-tokens 512 --gen-tokens 128
ab fn-ar-d32k 6 2 GUFO_FLASH_NEXT_MODEL -- \
  text --model "$FN" --prompt-tokens 512 --gen-tokens 64 --depth 32768
ab ds-ar 6 3 GUFO_DEEPSEEK_V4_FLASH_MODEL -- \
  text --model "$DS" --prompt-tokens 512 --gen-tokens 128
ab ds-dspark 6 3 GUFO_DEEPSEEK_V4_FLASH_MODEL GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL -- \
  text --model "$DS" --dspark-model "$DSP" --prompt-tokens 512 --gen-tokens 128
# Serving: the HTTP path with its scheduler, sampling and concurrency.
ab serve-q27-dflash2 6 1 GUFO_QWEN27B_MODEL GUFO_QWEN27B_DFLASH -- \
  serve --model "$Q4" --name dflash2 \
  --server-args "--speculative dflash2 --dflash-model $DF" --concurrency 1,4
ab serve-fn-mtp 6 1 GUFO_FLASH_NEXT_MODEL GUFO_FLASH_NEXT_MTP_MODEL -- \
  serve --model "$FN" --name mtp \
  --server-args "--speculative mtp --mtp-model $MTP" --concurrency 1,4
ab serve-ds-dspark 4 1 GUFO_DEEPSEEK_V4_FLASH_MODEL GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL -- \
  serve --model "$DS" --name dspark \
  --server-args "--speculative dspark --dspark-model $DSP" --concurrency 1,4
# Speech, images and embeddings.
ab asr 8 3 GUFO_QWEN3_ASR_MODEL GUFO_QWEN3_ASR_AUDIO -- \
  asr --model "${GUFO_QWEN3_ASR_MODEL:-}" --audio "${GUFO_QWEN3_ASR_AUDIO:-}"
ab tts-customvoice 6 3 GUFO_QWEN3_TTS_CUSTOMVOICE -- \
  tts --model "${GUFO_QWEN3_TTS_CUSTOMVOICE:-}" --name customvoice
ab tts-voicedesign 6 3 GUFO_QWEN3_TTS_VOICEDESIGN -- \
  tts --model "${GUFO_QWEN3_TTS_VOICEDESIGN:-}" --name voicedesign --voice "" \
  --request '{"instructions":"A calm, warm female voice."}'
ab image 4 1 GUFO_QWEN_IMAGE_MODEL -- \
  image --model "${GUFO_QWEN_IMAGE_MODEL:-}" --steps 8 --size 1024x1024
ab embedding 8 3 GUFO_EMBEDDING_GEMMA2_MODEL -- \
  embedding --model "${GUFO_EMBEDDING_GEMMA2_MODEL:-}"
exit $failed
