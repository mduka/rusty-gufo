#!/usr/bin/env python3
"""Regenerate tests/fixtures/vision_image_golden.json.

NVIDIA's image processor for Nemotron 3.5 Bolt
(image_processing_nemotron_h_omni.py in the BF16 checkpoint, revision
e09a4cedb1787db1760375ee5539f222cae0c672) plans a grid of 16-pixel patches,
resizes the RGB8 image to it with PyTorch's antialiased bicubic interpolation
in float32 and normalizes it with the CLIP statistics. This script computes
the grids with transformers' port of that processor (`get_patch_grid_size`,
the same algorithm) and resizes and normalizes the way the processor's
`_preprocess` does. The images are synthetic; the C++ test regenerates their
pixels from the same formula. No model weights are needed.

    python3 tools/vision_golden.py     # torch, numpy, transformers >= 5.22
"""

import argparse
import ast
import importlib.util
import json
import math
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch


def load_grid_function():
    """transformers' get_patch_grid_size, taken from its source: the module
    imports torchvision, which the plain function does not need."""
    spec = importlib.util.find_spec("transformers")
    source = (Path(spec.origin).parent / "models/nemotron_h_omni/"
              "image_processing_nemotron_h_omni.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "get_patch_grid_size")
    namespace = {"math": math}
    exec(compile(ast.Module([node], []), "get_patch_grid_size", "exec"),
         namespace)
    return namespace["get_patch_grid_size"]


get_patch_grid_size = load_grid_function()

REPO = Path(__file__).resolve().parents[1]
PATCH = 16
MERGE = 2
MIN_PATCHES = 1024
MAX_PATCHES = 13312
# preprocessor_config.json: norm_mean, norm_std.
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
SAMPLES = 1024

GRID_SIZES = [
    (1, 1), (16, 16), (31, 17), (512, 512), (640, 480), (1000, 700),
    (1024, 768), (1072, 1072), (1088, 1088), (1920, 1080), (1080, 1920),
    (2560, 1440), (3840, 2160), (4000, 3000), (8000, 6000), (3000, 200),
    (200, 3000), (17, 4097), (10000, 1), (1, 10000), (100000, 1),
]
IMAGE_SIZES = [
    (640, 480), (1000, 700), (1920, 1080), (4000, 3000), (3000, 200),
    (512, 512), (333, 2000),
]


def pixels(width, height):
    """The synthetic RGB8 image: gradients plus hash noise."""
    x = np.arange(width, dtype=np.uint64)[None, :, None]
    y = np.arange(height, dtype=np.uint64)[:, None, None]
    c = np.arange(3, dtype=np.uint64)[None, None, :]
    h = ((x * 73856093) ^ (y * 19349663) ^ (c * 83492791)) & 0xFFFFFFFF
    v = (x * 255) // width + (y * 128) // height + c * 40 + (h >> 13) % 48
    return (v % 256).astype(np.uint8)


def grid(width, height):
    try:
        return get_patch_grid_size(height, width, PATCH, MERGE, MIN_PATCHES,
                                   MAX_PATCHES)
    except ZeroDivisionError:  # a side vanished
        return None


def normalized_image(width, height, rows, cols):
    """The processor's pixel values: (1, 3, rows * 16, cols * 16) float32."""
    arr = pixels(width, height)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(
        dtype=torch.float32)
    target_h, target_w = rows * PATCH, cols * PATCH
    if t.shape[-2] != target_h or t.shape[-1] != target_w:
        t = torch.nn.functional.interpolate(
            t, size=(target_h, target_w), mode="bicubic",
            align_corners=False, antialias=True)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (t / 255.0 - mean) / std


def preprocess(width, height, rows, cols):
    """The processor's pixel values as the model's patches."""
    t = normalized_image(width, height, rows, cols)
    # RADIO's _image_to_patches: row-major patches, channel-major inside.
    patches = t.reshape(1, 3, rows, PATCH, cols, PATCH)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(rows * cols, -1)
    return patches.contiguous().numpy().reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        default=REPO / "tests/fixtures/vision_image_golden.json")
    args = parser.parse_args()
    torch.set_num_threads(1)

    rng = random.Random(20261109)
    sizes = GRID_SIZES + [(rng.randint(1, 9000), rng.randint(1, 9000))
                          for _ in range(300)]
    grids = []
    for width, height in sizes:
        g = grid(width, height)
        grids.append([width, height] + (list(g) if g else [0, 0]))

    images = []
    for width, height in IMAGE_SIZES:
        rows, cols = grid(width, height)
        values = preprocess(width, height, rows, cols)
        bits = values.view(np.uint32)
        total = values.size
        picks = [(k * 2654435761) % total for k in range(SAMPLES)]
        images.append({
            "width": width,
            "height": height,
            "rows": rows,
            "cols": cols,
            "sum": float(values.astype(np.float64).sum()),
            "abs_sum": float(np.abs(values.astype(np.float64)).sum()),
            "min": float(values.min()),
            "max": float(values.max()),
            "bits": [int(bits[i]) for i in picks],
        })

    golden = {
        "reference": f"torch {version('torch')}, transformers "
                     f"{version('transformers')}",
        "patch_size": PATCH,
        "samples": SAMPLES,
        "grids": grids,
        "images": images,
    }
    text = json.dumps(golden, indent=1)
    args.out.write_text(text + "\n")
    print(f"wrote {args.out} ({len(grids)} grids, {len(images)} images)")


if __name__ == "__main__":
    main()
