# VoxCPM2 experiments

Nothing has been tried on this engine for VoxCPM2 yet. The extension's
kernel experiments (skinny products, the DiT on the matrix cores, AudioVAE
on its own stream, the attention kernel) and its comparison with audio.cpp
are in its
[experiments](https://github.com/gufo-extensions/voxcpm/blob/0941cd1/docs/models/voxcpm2/EXPERIMENTS.md)
and [benchmarks](https://github.com/gufo-extensions/voxcpm/blob/0941cd1/docs/models/voxcpm2/BENCHMARKS.md).

## Not tried yet

- `GPU_MAX_HW_QUEUES=2` for `gufo serve tts`, which lets AudioVAE decode
  beside generation (3–4 % faster in the extension's measurement on Gufo).
