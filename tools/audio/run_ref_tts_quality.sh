#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
reference_root="${QWEN3_TTS_REFERENCE_ROOT:?set QWEN3_TTS_REFERENCE_ROOT to the official Qwen3-TTS checkout}"
model_root="${QWEN3_TTS_MODEL_ROOT:?set QWEN3_TTS_MODEL_ROOT to the Qwen3-TTS-12Hz-1.7B-CustomVoice snapshot}"
official_python="${QWEN3_TTS_PYTHON:-$reference_root/.venv/bin/python}"
output_root="$repo_root/artifacts/qwen3_tts/quality"
text_path="$repo_root/tests/models/qwen3_tts/quality/long-form.txt"
mode="${1:-sampled}"
extra=()
if [[ "$mode" == "greedy" ]]; then
  extra+=(--greedy)
elif [[ "$mode" != "sampled" ]]; then
  echo "usage: tools/audio/run_ref_tts_quality.sh [greedy|sampled]" >&2
  exit 2
fi
if [[ -n "${QWEN3_TTS_MAX_NEW_TOKENS:-}" ]]; then
  extra+=(--max-new-tokens "$QWEN3_TTS_MAX_NEW_TOKENS")
fi


exec "$official_python" \
  "$repo_root/tests/models/qwen3_tts/reference/generate_artifacts.py" \
  --reference-root "$reference_root" \
  --model "$model_root" \
  --out "$output_root" \
  --text-file "$text_path" \
  "${extra[@]}"
