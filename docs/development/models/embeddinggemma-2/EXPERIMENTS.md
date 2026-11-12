# EmbeddingGemma 2 experiments

What has been tried on this engine for EmbeddingGemma 2, so that later
optimization rounds do not try it again. Above all it keeps what did not help
or made things worse, with the measurement and the reason; what the code
already does gets a line. Gufo's
[experiments](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/embeddinggemma-2/EXPERIMENTS.md)
keep the same for its C++ engine, whose kernels, launch shapes, GEMM plans and
timing-based tuning this engine keeps: read both before an optimization round.
Ideas not tried yet come last. The core and what spans models are in
[optimizations](../../OPTIMIZATIONS.md).

Nothing has been tried on this engine yet.
