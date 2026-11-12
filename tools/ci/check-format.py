#!/usr/bin/env python3
"""Run the same C++ formatting check locally and in CI.

The formatter is the clang-format release pinned in tools/ci/requirements.txt,
from the repository's .venv, else from PATH.
"""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys


def formatter(root: Path) -> str:
    """The pinned clang-format, or exit with how to install it."""
    pins = (root / "tools/ci/requirements.txt").read_text(encoding="utf-8")
    wanted = re.search(r"^clang-format==(\S+)$", pins, re.MULTILINE).group(1)
    hint = ("install it with `pip install -r tools/ci/requirements.txt` "
            "into a virtual environment (README: Build from source)")
    found = shutil.which("clang-format", path=str(root / ".venv/bin")) or shutil.which(
        "clang-format")
    if found is None:
        sys.exit(f"clang-format {wanted} not found; {hint}")
    output = subprocess.run([found, "--version"], capture_output=True, text=True).stdout
    version = re.search(r"clang-format version (\S+)", output)
    if version is None or version.group(1) != wanted:
        sys.exit(f"{found} is not clang-format {wanted}; {hint}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="format files in place")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    excluded = (
        "src/models/deepseek_v4_flash/kernels/rocm/",
        "src/models/gemma4/kernels/rocm/",
        "src/models/nemotron/kernels/rocm/mmq/",
        "src/models/qwen38_flash_next/kernels/rocm/mmq/",
    )
    files = sorted(
        str(path.relative_to(root))
        for directory in ("src", "tests")
        for path in (root / directory).rglob("*")
        if path.is_file()
        and path.suffix in {".cpp", ".h", ".hpp"}
        and not {"fixtures", "vendor"}.intersection(path.relative_to(root).parts)
        and not path.relative_to(root).as_posix().startswith(excluded)
    )
    flags = ["-i"] if args.fix else ["--dry-run", "--Werror"]
    result = subprocess.run([formatter(root), *flags, *files], cwd=root)
    if result.returncode == 0:
        print(f"PASS: {'Formatted' if args.fix else 'Checked'} {len(files)} C++ files")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
