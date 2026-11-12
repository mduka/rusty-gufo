"""Writes synthetic-200x150.png: a deterministic 8-bit RGB test image (sky
gradient, a sun disc, a red house with a door and a striped green ground) whose
size makes the Qwen preprocessing raise it to the minimum pixel count
(200x150 -> 320x224, a 10 x 7 merged grid). Usage: make_synthetic.py OUT.png"""
import struct, sys, zlib

W, H = 200, 150


def pixel(x, y):
    if (x - 150) ** 2 + (y - 35) ** 2 <= 18 ** 2:          # sun
        return (250, 210, 40)
    if 60 <= x < 120 and 70 <= y < 115:                      # house
        if 82 <= x < 98 and 90 <= y < 115:                   # door
            return (90, 50, 20)
        return (200, 40, 40)
    if y >= 115:                                             # ground stripes
        return (40, 140 + 30 * ((x // 8 + y // 8) % 2), 50)
    if 50 <= y < 70 and abs(x - 90) <= (y - 50) * 1.6:       # roof
        return (120, 60, 30)
    return (100 + y // 2, 150 + y // 3, 255 - y // 4)       # sky


raw = b"".join(b"\0" + bytes(c for x in range(W) for c in pixel(x, y)) for y in range(H))


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
open(sys.argv[1], "wb").write(png)
