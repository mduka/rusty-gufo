#!/usr/bin/env python3
"""Reference outputs of the vision encoder for Nemotron 3.5 Bolt.

Runs transformers' port of NVIDIA's model code (`RadioModel` and the omni
model's projector and pixel shuffle; NVIDIA's remote code in the BF16
checkpoint describes the same model) with the weights of the mmproj GGUF on
the synthetic images of tools/vision_golden.py, once in float32 and once in
bfloat16, the precision NVIDIA runs the encoder in.

It writes tests/fixtures/vision_encoder_golden.json (sampled outputs and
per-token norms for nemotron_vision_encoder_test) and, with --dump DIR, every
stage as raw little-endian float32 files that the test compares in full when
GUFO_NEMOTRON_VISION_REFERENCE names DIR:
<image>-<dtype>-<stage>.f32 for the stages embed, block.0 .. block.31 and
output.

    python3 tools/vision_reference.py --mmproj MMPROJ.gguf [--dump DIR]
"""

import argparse
import json
import struct
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from transformers.models.nemotron_h_omni.modeling_nemotron_h_omni import (
    NemotronH_Omni_Reasoning_V3,
    NemotronH_Omni_Reasoning_V3MultiModalProjector,
)
from transformers.models.radio.configuration_radio import RadioConfig
from transformers.models.radio.modeling_radio import RadioModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_golden import MEAN, STD, grid, normalized_image  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
IMAGES = [(512, 512), (1000, 700)]
SAMPLES = 1024


def read_gguf(path):
    """Tensors of a GGUF file as float32 arrays (F32 and BF16 only)."""
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
              10: "Q", 11: "q", 12: "d"}
    with open(path, "rb") as f:
        def read(fmt):
            return struct.unpack("<" + fmt,
                                 f.read(struct.calcsize("<" + fmt)))[0]

        def string():
            return f.read(read("Q")).decode()

        def value(kind):
            if kind in scalar:
                return read(scalar[kind])
            if kind == 8:
                return string()
            item = read("I")
            return [value(item) for _ in range(read("Q"))]

        assert f.read(4) == b"GGUF"
        read("I")
        count, metadata = read("Q"), read("Q")
        meta = {}
        for _ in range(metadata):
            key = string()
            meta[key] = value(read("I"))
        infos = []
        for _ in range(count):
            name = string()
            dims = [read("Q") for _ in range(read("I"))]
            infos.append((name, dims, read("I"), read("Q")))
        alignment = meta.get("general.alignment", 32)
        base = (f.tell() + alignment - 1) // alignment * alignment
    data = np.memmap(path, dtype=np.uint8, mode="r")
    tensors = {}
    for name, dims, kind, offset in infos:
        n = int(np.prod(dims))
        shape = tuple(reversed(dims))
        start = base + offset
        if kind == 0:
            a = data[start:start + 4 * n].view(np.float32).reshape(shape)
        elif kind == 30:
            raw = data[start:start + 2 * n].view(np.uint16).reshape(shape)
            a = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            raise ValueError(f"{name}: unsupported GGUF type {kind}")
        tensors[name] = torch.from_numpy(np.array(a, dtype=np.float32))
    return meta, tensors


def build(tensors):
    """The vision tower and projector with the mmproj weights, in float32."""
    config = RadioConfig(norm_mean=MEAN, norm_std=STD,
                         video_temporal_patch_size=2, summary_idxs=[0, 1])
    config._attn_implementation = "sdpa"
    radio = RadioModel(config)
    radio.make_preprocessor_external()
    state = {
        "embeddings.patch_projection.weight":
            tensors["v.patch_embd.weight"].reshape(1280, -1),
        "embeddings.video_patch_projection.weight":
            tensors["v.patch_embd.weight.1"].reshape(1280, -1),
        "embeddings.position_embedding":
            tensors["v.position_embd.weight"].unsqueeze(0),
        "embeddings.cls_register_token": tensors["v.class_embd"],
    }
    names = {"ln1": "norm1", "ln2": "norm2", "attn_q": "attention.q_proj",
             "attn_k": "attention.k_proj", "attn_v": "attention.v_proj",
             "attn_out": "attention.o_proj", "ffn_up": "mlp.fc1",
             "ffn_down": "mlp.fc2"}
    for i in range(config.num_hidden_layers):
        for gguf_name, hf_name in names.items():
            for part in ("weight", "bias"):
                state[f"encoder.layer.{i}.{hf_name}.{part}"] = \
                    tensors[f"v.blk.{i}.{gguf_name}.{part}"]
    missing, unexpected = radio.load_state_dict(state, strict=False)
    # Layer scales are not stored: their initialization is the identity.
    assert all("layer_scale" in k or k == "summary_idxs" for k in missing), \
        missing
    assert not unexpected, unexpected
    projector = NemotronH_Omni_Reasoning_V3MultiModalProjector(5120, 20480,
                                                               4096)
    projector.load_state_dict({
        "layer_norm.weight": tensors["mm.model.mlp.0.weight"],
        "linear_1.weight": tensors["mm.model.mlp.1.weight"],
        "linear_2.weight": tensors["mm.model.mlp.3.weight"],
    })
    for layer in radio.encoder.layer:
        assert torch.all(layer.layer_scale1.lambda1 == 1.0)
        assert torch.all(layer.layer_scale2.lambda1 == 1.0)
    return radio.eval(), projector.eval()


@torch.no_grad()
def encode(radio, projector, pixels, rows, cols, dtype):
    """get_image_features for one image: stages as float32 arrays."""
    radio = radio.to(dtype)
    projector = projector.to(dtype)
    out = radio(pixels.to(dtype), output_hidden_states=True)
    features = out.features.view(1, rows, cols, -1)
    merged = NemotronH_Omni_Reasoning_V3.pixel_shuffle(
        None, features, scale_factor=0.5).flatten(0, 2)
    stages = {"embed": out.hidden_states[0]}
    for i, h in enumerate(out.hidden_states[1:]):
        stages[f"block.{i}"] = h
    stages["output"] = projector(merged)
    return {k: v.float().reshape(-1).numpy() for k, v in stages.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--out", type=Path,
                        default=REPO / "tests/fixtures/vision_encoder_golden.json")
    args = parser.parse_args()
    _, tensors = read_gguf(args.mmproj)
    radio, projector = build(tensors)
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
    images = []
    for width, height in IMAGES:
        rows, cols = grid(width, height)
        pixels = normalized_image(width, height, rows, cols)
        entry = {"width": width, "height": height, "rows": rows, "cols": cols,
                 "tokens": rows * cols // 4}
        for dtype in (torch.float32, torch.bfloat16):
            name = str(dtype).removeprefix("torch.")
            stages = encode(radio, projector, pixels, rows, cols, dtype)
            output = stages["output"]
            picks = [(k * 2654435761) % output.size for k in range(SAMPLES)]
            norms = np.linalg.norm(output.reshape(-1, 4096).astype(np.float64),
                                   axis=1)
            entry[name] = {
                "samples": [float(output[i]) for i in picks],
                "norms": [float(v) for v in norms],
            }
            if args.dump:
                for stage, values in stages.items():
                    values.astype("<f4").tofile(
                        args.dump / f"{width}x{height}-{name}-{stage}.f32")
            print(f"{width}x{height} {name}: {output.size // 4096} tokens, "
                  f"rms {float(np.sqrt(np.mean(output.astype(np.float64) ** 2))):.4f}")
        images.append(entry)
    golden = {
        "reference": f"torch {version('torch')}, transformers "
                     f"{version('transformers')}",
        "samples": SAMPLES,
        "images": images,
    }
    args.out.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
