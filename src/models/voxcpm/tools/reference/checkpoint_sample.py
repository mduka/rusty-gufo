"""Small torch.save checkpoints for tests/torch_checkpoint_test.cpp.

usage: checkpoint_sample.py OUT_DIR

sample.pth wraps a state dict like VoxCPM's audiovae.pth ({"metadata": ...,
"state_dict": OrderedDict}) with float32, bfloat16 and int32 tensors, a
tensor at an offset into a shared storage and a zero-dimensional one.
view.pth holds a transposed (non-contiguous) tensor, which the reader
rejects.
"""

import collections
import sys

import torch


def main():
    out = sys.argv[1]
    shared = torch.arange(10, dtype=torch.float32)
    state = collections.OrderedDict()
    state["layer.weight"] = torch.tensor([[1.5, -2.0, 3.25], [0.0, 4.0, -0.5]])
    state["layer.bias"] = torch.tensor([7, -8, 9, 2**31 - 1], dtype=torch.int32)
    state["half"] = torch.tensor([1.0, -3.0], dtype=torch.bfloat16)
    state["tail"] = shared[6:]
    state["head"] = shared[:2]
    state["scalar"] = torch.tensor(0.25)
    torch.save({"metadata": {"kwargs": {"sample_rate": 16000}},
                "state_dict": state}, f"{out}/sample.pth")
    torch.save({"weight": torch.arange(6.0).reshape(2, 3).t()},
               f"{out}/view.pth")


if __name__ == "__main__":
    main()
