# Qwen-Image-2.1: checks

The Rust engine (`src/models/qwen_image_21`) reproduces the C++ engine bit
for bit: every observed tensor and image of `tests/parity.rs` and the PNG
decoding of `tests/decode.rs` match fixtures recorded from C++.

```sh
cargo test --profile gpu-test -p gufo-qwen-image-21 --features gpu,models
```

Gufo's [quality report](https://github.com/gufo-org/gufo/blob/v0.28.1/docs/models/qwen-image-2.1/QUALITY.md#reproduce) compares its
engine with Diffusers through its C++ probe and
`tools/models/qwen_image_21/reference.py`, with teacher forcing or complete
trajectories. The probe is kept as
[`cpp_probe.cpp`](../../../../src/models/qwen_image_21/tests/reference/cpp_probe.cpp),
whose header gives the build and the comparison commands.
