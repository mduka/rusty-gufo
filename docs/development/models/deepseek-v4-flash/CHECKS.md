# DeepSeek V4 Flash: checks

The checks run on the Rust engine (`tests/quality.rs`, `tests/serving.rs`
and the EOS boundary of `gufo-deepseek-v4-flash`, the sampled CLI chat of
`gufo`); the kernel operator oracles remain C++. Use the smallest affected
[maintained suite](../../../../tools/ds4/README.md):

```sh
tools/ds4/check.py fast
tools/ds4/check.py kernels
tools/ds4/check.py model --model "$MODEL" --dspark-model "$DSPARK"
```

Independent FP64 operators and exact sparse top-k checks supplement model/state
replay. Keep full-logit captures outside Git; do not relax quality gates.

On the pinned antirez/ds4 trajectory the optimized C++ and Rust builds give
identical logits, and HTTP seeded sampling agrees in both engines; Gufo's
[quality report](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/deepseek-v4-flash/QUALITY.md) has the measured values
and limits.

## Retained evidence

Gufo's [`artifacts/`](https://github.com/gufo-org/gufo/tree/v0.28.1/docs/models/deepseek-v4-flash/artifacts)
retain controller calibration, sampling/qualification summaries and three
unresolved target-arithmetic audits. Maintained executable fixtures remain with
their tests.
