#!/usr/bin/env python3
"""Focused Qwen27B checks on gfx1151; model artifacts are required for model and serving."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BINARY = "target/release/gufo"
KERNELS = "gufo-hip/gpu,gufo-qwen-kernels/gpu"
BUILD = ["cargo", "build", "--release", "-p", "gufo"]
CLI = ["python3", "tests/models/qwen27b/cli_test.py", BINARY]
# NumPy comes from the development shell.
DEV = ["nix", "develop", "-c"]
SUITES = {
    "fast": [
        DEV + ["python3", "tests/models/qwen27b/tools_test.py"],
        ["cargo", "test", "-p", "gufo-qwen", "--lib"],
        ["cargo", "test", "-p", "gufo-qwen35", "--lib"],
        ["cargo", "test", "-p", "gufo", "--bin", "gufo", "--", "prompt::tests", "bench::tests"],
        BUILD,
        CLI + ["--scope", "options"],
    ],
    "kernels": [
        ["cargo", "test", "--profile", "gpu-test", "-p", "gufo-qwen-kernels",
         "--features", KERNELS],
        # This existing tool compares the optimized resident recurrence with
        # the independent per-token storage geometry, bit for bit.
        ["tools/bench/build.sh", "tools/qwen27b/deltanet_bench.hip"],
        ["/tmp/deltanet_bench", "1", "1"],
        ["/tmp/deltanet_bench", "8", "1"],
    ],
    "serving": [BUILD, CLI + ["--scope", "http"]],
    "model": [
        # The 27B engine: prefill, decode, sampling, DFlash2, MTP, sessions,
        # verification and vision (which needs the mmproj sidecar).
        ["cargo", "test", "--profile", "gpu-test", "-p", "gufo-qwen35", "--features", "gpu,models"],
        # The chat template's Hugging Face token goldens on the 27B GGUF.
        ["cargo", "test", "--profile", "gpu-test", "-p", "gufo-qwen", "--features", "models"],
        BUILD,
        CLI,
    ],
}
# The engine tests read GUFO_QWEN27B_<NAME>, the CLI test GUFO_QWEN27B_<NAME>_MODEL.
ARTIFACTS = {"model": ["MODEL"], "mtp_model": ["MTP", "MTP_MODEL"],
             "dflash_model": ["DFLASH", "DFLASH_MODEL"], "mmproj": ["MMPROJ"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=[*SUITES, "all"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--mtp-model", type=Path)
    parser.add_argument("--dflash-model", type=Path)
    parser.add_argument("--mmproj", type=Path, help="the 27B mmproj-BF16.gguf (model suite)")
    args = parser.parse_args()
    if os.environ.get("IN_NIX_SHELL"):
        parser.error("run outside nix develop: its ROCm is not the binary's")
    environment = os.environ.copy()
    required = (["model", "mtp_model", "dflash_model", "mmproj"] if args.suite in ("model", "all") else
                ["model", "dflash_model"] if args.suite == "serving" else [])
    for name in required:
        names = ["GUFO_QWEN27B_" + suffix for suffix in ARTIFACTS[name]]
        value = getattr(args, name) or next(
            (environment[n] for n in names if environment.get(n)), None)
        if not value or not Path(value).is_file():
            parser.error(f"missing artifact: --{name.replace('_', '-')}")
        for variable in names:
            environment[variable] = str(Path(value).resolve())
    suites = [SUITES[args.suite]] if args.suite in SUITES else list(SUITES.values())
    commands = []
    for command in (command for suite in suites for command in suite):
        if command not in commands:
            commands.append(command)
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
