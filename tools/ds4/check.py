#!/usr/bin/env python3
"""Run the owned DS4 checks; model artifacts are required for the model suites."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
# The kernel operator checks are standalone HIP programs and the fixture
# provenance check a Python script; the engine, its sessions, serving, CLI
# and the answer grading run in Rust. Model QUALITY checks are ignored by
# the ordinary test run.
FEATURES = ("gufo-hip/gpu,gufo-deepseek-v4-flash-kernels/gpu,gufo-deepseek-v4-flash/gpu,"
            "gufo-deepseek-v4-flash/models,gufo-core/models")
SUITES = {
    "fast": [
        ["python3", "tests/models/deepseek_v4_flash/dataset_test.py"],
        ["cargo", "test", "-p", "gufo-eval"],
        ["cargo", "test", "-p", "gufo-deepseek-v4-flash", "--lib",
         "--test", "chat_template", "--test", "load"],
        ["cargo", "test", "-p", "gufo", "--bin", "gufo", "--", "prompt::tests", "bench::tests"],
    ],
    "kernels": [
        ["tests/models/deepseek_v4_flash/run_operator_checks.sh"],
        ["cargo", "test", "--profile", "gpu-test", "-p", "gufo-deepseek-v4-flash-kernels",
         "--features", "gufo-hip/gpu,gufo-deepseek-v4-flash-kernels/gpu"],
    ],
    "model": [
        ["cargo", "test", "--profile", "gpu-test", "-p", "gufo-deepseek-v4-flash",
         "--features", FEATURES, "--test", "forward", "--test", "dspark", "--test", "routes",
         "--test", "q8_routes", "--test", "chat_template",
         "--test", "tokenizer"],
        ["cargo", "test", "--profile", "gpu-test", "-p", "gufo-deepseek-v4-flash",
         "--features", FEATURES, "--lib", "--test", "quality", "--test", "serving", "--",
         "--ignored", "--nocapture", "--test-threads", "1"],
        ["cargo", "test", "-p", "gufo", "--features", "gpu", "--test", "deepseek", "--",
         "--ignored"],
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=(*SUITES, "reference", "benchmark", "all"),
                        default="all", nargs="?")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dspark-model", type=Path)
    parser.add_argument("--output", type=Path,
                        help="new reference directory or benchmark report file")
    parser.add_argument("--upstream", type=Path,
                        help="clean antirez/ds4 checkout at the pinned reference revision")
    parser.add_argument("--prefill-only", action="store_true",
                        help="reference: compare post-prefill logits, skipping continuations")
    parser.add_argument("--ar-log", type=Path)
    parser.add_argument("--dspark-log", type=Path)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--concurrency", default="1,2,4,6,8",
                        help="benchmark: comma-separated C values the logs cover")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="benchmark: --temperature both logs were produced with")
    parser.add_argument("--seed", type=int, default=0,
                        help="benchmark: --seed both logs were produced with")
    parser.add_argument("--top-k", type=int, default=0,
                        help="benchmark: --top-k both logs were produced with")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="benchmark: --top-p both logs were produced with")
    parser.add_argument("--min-keep", type=int, default=0,
                        help="benchmark: --min-keep both logs were produced with")
    args = parser.parse_args()
    if args.suite == "reference" and not os.environ.get("IN_NIX_SHELL"):
        parser.error("the reference suite runs its Python tools in nix develop")
    if args.prefill_only and args.suite != "reference":
        parser.error("--prefill-only requires the reference suite")
    if args.suite == "benchmark":
        if not args.ar_log or not args.dspark_log or not args.output:
            parser.error("benchmark requires --ar-log, --dspark-log and --output")
        if args.model or args.dspark_model or args.upstream:
            parser.error("benchmark validates existing logs; model/upstream options do not apply")
        if (not math.isfinite(args.temperature) or args.temperature < 0
                or args.seed < 0 or args.top_k < 0 or args.min_keep < 0
                or not 0 < args.top_p <= 1):
            parser.error("benchmark needs valid sampling controls and a fixed seed")
        from benchmark import summarize
        concurrency = tuple(int(value) for value in args.concurrency.split(","))
        sampling = ({"temperature": args.temperature, "seed": args.seed,
                     "top_k": args.top_k, "top_p": args.top_p,
                     "min_keep": args.min_keep}
                    if args.temperature > 0.0 else None)
        summarize(args.ar_log, args.dspark_log, args.repetitions, args.output,
                  concurrency, sampling)
        return
    if (args.ar_log or args.dspark_log or args.repetitions != 2 or
            args.concurrency != "1,2,4,6,8" or args.temperature or args.seed or
            args.top_k or args.top_p != 1 or args.min_keep):
        parser.error("benchmark log/repetition options require the benchmark suite")
    environment = os.environ.copy()
    reference = args.suite == "reference"
    if reference != (args.output is not None):
        parser.error("--output is required only for reference")
    if reference != (args.upstream is not None):
        parser.error("--upstream is required only for reference")
    if reference and (args.output.exists() or not args.upstream.is_dir()):
        parser.error("reference needs a new output directory and existing upstream checkout")
    if args.suite in ("model", "all", "reference"):
        models = [(args.model, "GUFO_DEEPSEEK_V4_FLASH_MODEL")]
        if not reference:
            models.append((args.dspark_model, "GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL"))
        for value, name in models:
            value = value or environment.get(name)
            if not value or not Path(value).is_file():
                parser.error(f"model checks require an existing artifact: {name}")
            environment[name] = str(Path(value).absolute())
    suites = [SUITES[args.suite]] if args.suite in SUITES else (
        [] if reference else list(SUITES.values()))
    commands = [command for suite in suites for command in suite]
    if reference:
        commands.append(["cargo", "build", "--release", "-p", "gufo-deepseek-v4-flash",
                         "--example", "reference"])
    for command in commands:
        subprocess.run(command, cwd=ROOT,
                       env=environment, check=True)
    if reference:
        from reference import run
        run(Path(environment["GUFO_DEEPSEEK_V4_FLASH_MODEL"]),
            args.output.resolve(), args.upstream.resolve(), environment, args.prefill_only)


if __name__ == "__main__":
    main()
