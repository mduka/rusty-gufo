#!/usr/bin/env python3
"""Packs the C++ extension's outputs into tests/fixtures/cxx_reference.safetensors.

The dump comes from `nemotron_asr_model_dump MODEL OUT_DIR`, a development
tool on top of the extension (gufo-extensions/nemotron_asr c256b69) that
writes the outputs `tests/model.rs` compares bit for bit: model_test's
encoder case (decoder start, three chunks' subsampled, encoder and projected
activations, the joint's first logits; golden features, lookahead 13) and,
for every run of golden.json, the token IDs and the prediction network's
final output. Files are raw little-endian float32, `*.ids` int32.

    pack_cxx_reference.py DUMP_DIR OUT.safetensors
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path


def main() -> None:
    dump, out = Path(sys.argv[1]), Path(sys.argv[2])
    header: dict[str, object] = {}
    payload = bytearray()
    for path in sorted(dump.iterdir()):
        data = path.read_bytes()
        dtype = "I32" if path.name.endswith(".ids") else "F32"
        header[path.name] = {
            "dtype": dtype,
            "shape": [len(data) // 4],
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload += data
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    out.write_bytes(struct.pack("<Q", len(text)) + text + payload)


if __name__ == "__main__":
    main()
