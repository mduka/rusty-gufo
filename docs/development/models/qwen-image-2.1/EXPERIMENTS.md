# Qwen-Image-2.1 experiments

What has been tried on this engine for Qwen-Image-2.1, so that later
optimization rounds do not try it again. Above all it keeps what did not help
or made things worse, with the measurement and the reason; what the code
already does gets a line. Gufo's
[experiments](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen-image-2.1/EXPERIMENTS.md)
keep the same for its C++ engine, whose kernels this engine runs: read both
before an optimization round. Ideas not tried yet come last. The core and what
spans models are in [optimizations](../../OPTIMIZATIONS.md).

## Tried without gain

Nothing on this engine yet.

## Not tried yet

**I1. Device memory below the C++ pool order** (Qwen-Image; bit-exact).
Temporaries die where the C++ full expressions end, so that peak memory
matches C++ (32.42 GiB for a 1024² generation, 40.55 GiB for an edit).
Freeing activations at their last use, or reusing them in place, could
lower both peaks.

**I2. Tokenizer hashing** (Qwen-Image; bit-exact). The ~300,000-entry
vocabulary map is the suspected cause of `/ready` at ~309 against ~261 ms;
C++ showed the same spread in some rounds.
