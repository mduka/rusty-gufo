# Qwen3.5 (dense): experiments

The small dense models share their engine with Qwen3.8 27B, so their
experiments are recorded with it: the open ideas Q1 to Q12 of the
[Qwen3.8 27B experiments](../qwen3.8-27b/EXPERIMENTS.md) name the Qwen3.5
checkpoints they concern (BF16 prefill, prefill attention for 8/2 and 16/4
heads, the 27B fast paths for the other geometries, smaller LM heads, the
residual decode gap to C++ and load time).
