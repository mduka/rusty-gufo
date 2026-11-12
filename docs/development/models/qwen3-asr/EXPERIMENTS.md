# Qwen3-ASR experiments

What has been tried on this engine for Qwen3-ASR 1.7B, so that later
optimization rounds do not try it again. Above all it keeps what did not help
or made things worse, with the measurement and the reason; what the code
already does gets a line. Gufo's
[experiments](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen3-asr/EXPERIMENTS.md)
keep the same for its C++ engine, whose kernels this engine runs: read both
before an optimization round. Ideas not tried yet come last. The core and what
spans models are in [optimizations](../../OPTIMIZATIONS.md).

## Tried without gain

Nothing on this engine yet.

## Not tried yet

**A1. Lazy encoder upload** (Qwen3-ASR; bit-exact). C++ uploads the 24
encoder layers during the first request, Rust at load (serve load 415
against 346 ms); load plus first request take the same time.
