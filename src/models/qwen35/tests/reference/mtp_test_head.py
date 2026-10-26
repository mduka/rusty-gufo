#!/usr/bin/env python3
"""The dense Qwen MTP test head (docs/RUST.md).

No Qwen3.8 27B MTP head exists locally. unsloth's Qwen3.6-27B MTP GGUF
(Qwen3.6-27B-Q4_K_M.gguf) carries the `blk.64.nextn` layer with the 27B's
shapes, but the C++ head (`src/models/qwen/mtp_reference.cpp`) takes only F32,
BF16, Q3_K, Q4_K and Q6_K matrices and its fusion projection is Q8_0. This
writes a GGUF with all its metadata, token_embd, output, output_norm and the
blk.64 tensors, each Q8_0 matrix widened exactly to F32 (an f16 scale times an
int8 fits in an f32):

    python3 src/models/qwen35/tests/reference/mtp_test_head.py Qwen3.6-27B-Q4_K_M.gguf \
        target/qwen-mtp/qwen36-mtp-head.gguf
"""

import array
import struct
import sys

SIZES = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 11: (256, 110), 12: (256, 144),
         13: (256, 176), 14: (256, 210)}


def main(source: str, target: str) -> None:
    with open(source, "rb") as f:
        data = f.read(64 << 20)
        pos = 0

        def rd(fmt):
            nonlocal pos
            values = struct.unpack_from("<" + fmt, data, pos)
            pos += struct.calcsize("<" + fmt)
            return values

        def rstr():
            nonlocal pos
            (n,) = rd("Q")
            s = data[pos:pos + n]
            pos += n
            return s

        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

        def skip(t):
            nonlocal pos
            if t == 8:
                rstr()
            elif t == 9:
                at, n = rd("IQ")
                for _ in range(n):
                    skip(at)
            else:
                rd(scalar[t])

        assert data[:4] == b"GGUF"
        pos = 4
        (version,) = rd("I")
        n_tensors, n_kv = rd("QQ")
        kv_start = pos
        alignment = 32
        for _ in range(n_kv):
            key = rstr()
            (t,) = rd("I")
            if key == b"general.alignment":
                (alignment,) = rd("I")
            else:
                skip(t)
        kv_bytes = data[kv_start:pos]
        infos = []
        for _ in range(n_tensors):
            name = rstr()
            (nd,) = rd("I")
            dims = rd("Q" * nd)
            t, offset = rd("IQ")
            infos.append((name, dims, t, offset))
        data_start = (pos + alignment - 1) // alignment * alignment

        keep = [i for i in infos if i[0].startswith(b"blk.64.")
                or i[0] in (b"token_embd.weight", b"output.weight", b"output_norm.weight")]
        tensors = []
        for name, dims, t, offset in keep:
            count = 1
            for d in dims:
                count *= d
            block, size = SIZES[t]
            f.seek(data_start + offset)
            raw = f.read(count // block * size)
            if t == 8:
                values = array.array("f")
                for b in range(0, len(raw), 34):
                    (scale,) = struct.unpack_from("<e", raw, b)
                    values.extend(scale * q for q in array.array("b", raw[b + 2:b + 34]))
                raw, t = values.tobytes(), 0
                print(f"widened {name.decode()} to F32", file=sys.stderr)
            tensors.append((name, dims, t, raw))

    header = bytearray(b"GGUF" + struct.pack("<IQQ", version, len(tensors), n_kv) + kv_bytes)
    offset = 0
    placed = []
    for name, dims, t, raw in tensors:
        header += struct.pack("<Q", len(name)) + name + struct.pack("<I", len(dims))
        header += struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", t, offset)
        placed.append((offset, raw))
        offset = (offset + len(raw) + alignment - 1) // alignment * alignment
    header += b"\0" * ((-len(header)) % alignment)
    with open(target, "wb") as out:
        out.write(header)
        base = len(header)
        for start, raw in placed:
            out.seek(base + start)
            out.write(raw)
        out.truncate(base + offset)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
