# Qwen3-ASR 1.7B: checks

Official [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff)
and [checkpoint](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5).
Tests and the independent runner live in
[`src/models/qwen3_asr`](../../../../src/models/qwen3_asr).
Select the changed test: `frontend` or `transcription` of `gufo-qwen3-asr`, or
`checkpoint`, `graphs` or `text_ops` of its kernel crate:

```sh
cargo test --profile gpu-test -p gufo-qwen3-asr --features gpu --test <name>
cargo test --profile gpu-test -p gufo-qwen3-asr-kernels --features gpu --test <name>
```

Generate official tensors with `reference/run_official.py --text-attention eager
--audio-attention sdpa`; compare finite tensors, full logits, every generated ID
and the transcript. Keep raw captures outside Git.

Measure transcription speed with:

```sh
cargo build --release -p gufo
python3 tools/bench/qwen3-asr-bench.py \
  --model "$MODEL" --audio speech.wav --warmup 1 --repeat 3
```

The benchmark takes the CLI's decoded sample count and reads float WAV.
Profile separately with `tools/prof/prof.py --stages qwen-asr`; raw traces
stay outside Git.
