#!/usr/bin/env python3
"""Compares the device code of two gufo binaries function by function.

  tools/bench/kernel-isa.py A B [--json REPORT] [--llvm DIR]

Every gfx1151 code object in the clang offload bundles of each binary is
disassembled; per function, the instructions are hashed after the operands
that depend on the code object's layout are normalized (branch labels
numbered per function, the PC-relative offsets of `s_getpc_b64` sequences).
The report lists the functions only one binary has, the functions whose
instructions differ, and the kernels whose resource metadata (registers,
spills, LDS, scratch) differ. llama.cpp's `no_device_code` abort stubs,
which print their source file, are listed apart and do not fail the check. Build both binaries with the same ROCm
(docs/RUST.md, "The C++ reference"): another compiler gives other code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
LABEL = re.compile(r"\bL\d+\b")
METADATA = re.compile(
    r"^(\.(?:sgpr_count|vgpr_count|sgpr_spill_count|vgpr_spill_count|group_segment_fixed_size|"
    r"private_segment_fixed_size|max_flat_workgroup_size|wavefront_size|uses_dynamic_stack)):\s+(\S+)$"
)


def code_objects(path: Path) -> list[bytes]:
    """The amdgcn entries of every offload bundle in `path`."""
    data = path.read_bytes()
    found = []
    pos = 0
    while (start := data.find(MAGIC, pos)) >= 0:
        (entries,) = struct.unpack_from("<Q", data, start + 24)
        p = start + 32
        for _ in range(entries):
            offset, size, triple_len = struct.unpack_from("<QQQ", data, p)
            p += 24
            triple = data[p:p + triple_len].decode()
            p += triple_len
            if "amdgcn" in triple:
                found.append(data[start + offset:start + offset + size])
        pos = start + len(MAGIC)
    return found


def functions(llvm: Path, blob: bytes, scratch: str) -> tuple[dict[str, list[str]], dict[str, dict[str, str]]]:
    """Normalized instructions per function and resource metadata per kernel."""
    if blob[:4] == b"CCOB":
        raise SystemExit("compressed code objects are not supported")
    path = os.path.join(scratch, "object.o")
    with open(path, "wb") as f:
        f.write(blob)
    listing = subprocess.run(
        [str(llvm / "llvm-objdump"), "-d", "--no-show-raw-insn", "--symbolize-operands", "--no-leading-addr", path],
        check=True, capture_output=True, text=True,
    ).stdout
    found: dict[str, list[str]] = {}
    name = None
    body: list[str] = []
    pc_relative = 0
    for line in listing.splitlines():
        m = re.match(r"^<(.+)>:$", line.strip())
        if m and re.fullmatch(r"L\d+", m.group(1)):
            body.append(f"<{m.group(1)}>:")
            continue
        if m:
            if name is not None:
                found[name] = body
            name, body, pc_relative = m.group(1), [], 0
            continue
        text = line.split("//")[0].strip()
        if name is None or not text or text == "...":
            continue
        if text.startswith("s_getpc_b64"):
            pc_relative = 2
        elif pc_relative and text.startswith(("s_add_u32", "s_addc_u32")):
            # The offset to a global depends on where the linker put it.
            text = text.split(",")[0] + ",REL"
            pc_relative -= 1
        body.append(text)
    if name is not None:
        found[name] = body
    for key, lines in found.items():
        labels: dict[str, int] = {}
        found[key] = [LABEL.sub(lambda m: f"L{labels.setdefault(m.group(0), len(labels))}", x) for x in lines]
    notes = subprocess.run([str(llvm / "llvm-readelf"), "--notes", path], check=True, capture_output=True,
                           text=True).stdout
    # Each kernel is one entry of the amdhsa.kernels list ("  - " at its
    # first field); its own fields sit four spaces in, `.name` among them.
    metadata: dict[str, dict[str, str]] = {}
    entries: list[dict[str, str]] = []
    inside = False
    for line in notes.splitlines():
        if line.strip() == "amdhsa.kernels:":
            inside = True
            continue
        if not inside:
            continue
        if line.startswith("  - "):
            entries.append({})
            line = "    " + line[4:]
        elif not line.startswith("    ") and not line.startswith("  -"):
            inside = False
            continue
        if entries and line.startswith("    .") and (m := re.match(r"^    (\.\w+):\s*(\S*)$", line)):
            entries[-1][m.group(1)] = m.group(2)
    for entry in entries:
        if ".name" in entry:
            metadata[entry[".name"]] = {k: v for k, v in entry.items() if METADATA.match(f"{k}: {v}")}
    return found, metadata


def inventory(llvm: Path, binary: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    hashes: dict[str, set[str]] = defaultdict(set)
    resources: dict[str, set[str]] = defaultdict(set)
    with tempfile.TemporaryDirectory() as scratch:
        for blob in code_objects(binary):
            found, metadata = functions(llvm, blob, scratch)
            for name, body in found.items():
                hashes[name].add(hashlib.sha256("\n".join(body).encode()).hexdigest())
            for name, values in metadata.items():
                resources[name].add(json.dumps(values, sort_keys=True))
    return hashes, resources


def demangle(names: list[str]) -> list[str]:
    try:
        out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True)
        return out.stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--json", type=Path, help="write the lists here")
    parser.add_argument("--llvm", type=Path,
                        default=Path(os.environ.get("ROCM_PATH", "/opt/rocm/core")) / "llvm/bin",
                        help="directory with llvm-objdump and llvm-readelf")
    args = parser.parse_args(argv)
    a, a_resources = inventory(args.llvm, args.a)
    b, b_resources = inventory(args.llvm, args.b)
    both = sorted(set(a) & set(b))
    different = [n for n in both if not a[n] & b[n]]
    report = {
        "only_a": sorted(set(a) - set(b)),
        "only_b": sorted(set(b) - set(a)),
        "different": [n for n in different if "no_device_code" not in n],
        "stubs_different": [n for n in different if "no_device_code" in n],
        "resources_different": sorted(
            n for n in set(a_resources) & set(b_resources) if not a_resources[n] & b_resources[n]),
    }
    identical = len(both) - len(different)
    print(f"functions: {len(a)} in A, {len(b)} in B, {len(both)} in both, {identical} identical")
    for key, title in (("different", "different instructions"),
                       ("stubs_different", "abort stubs with other messages"), ("only_a", "only in A"),
                       ("only_b", "only in B"), ("resources_different", "kernels with other resources")):
        print(f"{title}: {len(report[key])}")
        for name in demangle(report[key]):
            print(f"  {name}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0 if not report["different"] and not report["resources_different"] else 1


if __name__ == "__main__":
    sys.exit(main())
