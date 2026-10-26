#!/usr/bin/env python3
"""
check-unsafe.py — Rust `unsafe` policy for gufo (docs/RUST.md)

Validates that:
1. Hand-written `unsafe` appears only in src/hip (gufo-hip) and in the
   jobserver handoff of src/kernel_build/src/compile.rs.
2. Kernel crates (src/models/*/kernels, src/cli/kernels) allow `unsafe_code`
   only on their generated module and deny it everywhere else.
3. Every other crate root forbids `unsafe_code`.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List

# The gufo-hip crate itself; other crates below src/hip (e.g. smoke) are checked.
UNSAFE_ANYWHERE = ("src/hip/src/", "src/hip/build.rs")
UNSAFE_CRATE_ROOTS = {"src/hip/src/lib.rs"}
UNSAFE_FILES = {"src/kernel_build/src/compile.rs"}
DENY_ROOTS = {"src/kernel_build/src/lib.rs"}
KERNEL_ROOT = re.compile(r"^src/(models/[^/]+|cli)/kernels/src/lib\.rs$")
UNSAFE_TOKEN = re.compile(r"\bunsafe\s*(\{|fn\b|impl\b|extern\b|trait\b)|allow\s*\(\s*unsafe_code\s*\)")
GENERATED_MODULE = re.compile(r"#\[allow\(unsafe_code\)\]\s*mod generated\s*\{")
RAW_STRING = re.compile(r'r(#*)"(?:.|\n)*?"\1')
STRING = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)


def blank_literal(match: re.Match) -> str:
    return '""' + "\n" * match.group(0).count("\n")


def code_only(text: str) -> str:
    """Blank string literals (keeping line numbers), then drop // comments."""
    text = STRING.sub(blank_literal, RAW_STRING.sub(blank_literal, text))
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def crate_roots(root: Path) -> List[Path]:
    roots = []
    for manifest in root.glob("src/**/Cargo.toml"):
        for name in ("lib.rs", "main.rs"):
            candidate = manifest.parent / "src" / name
            if candidate.exists():
                roots.append(candidate)
    return roots


def check(root: Path) -> List[str]:
    errors = []
    for path in sorted(root.glob("src/**/*.rs")):
        rel = path.relative_to(root).as_posix()
        if (
            "/target/" in rel
            or rel.startswith(UNSAFE_ANYWHERE)
            or rel in UNSAFE_FILES
        ):
            continue
        code = code_only(path.read_text(encoding="utf-8"))
        if KERNEL_ROOT.match(rel):
            code = GENERATED_MODULE.sub("mod generated {", code)
        for number, line in enumerate(code.splitlines(), start=1):
            if UNSAFE_TOKEN.search(line):
                errors.append(
                    f"{rel}:{number}: unsafe outside gufo-hip (src/hip/src): {line.strip()}"
                )

    for path in crate_roots(root):
        rel = path.relative_to(root).as_posix()
        if rel in UNSAFE_CRATE_ROOTS:
            continue
        text = path.read_text(encoding="utf-8")
        if rel in DENY_ROOTS or KERNEL_ROOT.match(rel):
            if "#![deny(unsafe_code)]" not in text:
                errors.append(f"{rel}: crate root must declare #![deny(unsafe_code)]")
        elif "#![forbid(unsafe_code)]" not in text:
            errors.append(f"{rel}: crate root must declare #![forbid(unsafe_code)]")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    errors = check(args.root)
    for error in errors:
        print(f"[check-unsafe] {error}", file=sys.stderr)
    if errors:
        return 1
    print(
        "[check-unsafe] PASSED: unsafe code is confined to gufo-hip and generated launchers."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
