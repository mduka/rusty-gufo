#!/usr/bin/env python3
"""Tests for tools/ci/gate_runner.sh, the gate's runner of each test binary:
the record of binaries that passed, which lets a stopped run resume."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest

RUNNER = Path(__file__).resolve().parents[2] / "tools" / "ci" / "gate_runner.sh"


class GateRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.runs = self.dir / "runs"
        self.passed = self.dir / "passed.tsv"
        self.kept = self.dir / "kept.tsv"
        self.binary = self.dir / "prefill-0123456789abcdef"
        self.write_binary(0)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_binary(self, code: int, variant: str = "") -> None:
        # A stand-in test binary: counts its runs and exits with CODE.
        self.binary.write_text(f"#!/bin/sh\n# {variant}\necho run >>'{self.runs}'\nexit {code}\n")
        self.binary.chmod(self.binary.stat().st_mode | stat.S_IXUSR)

    def run_binary(self, *args: str, passed: bool = True) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if not k.startswith("GUFO_GATE_")}
        env["CARGO_PKG_NAME"] = "gufo-qwen35"
        if passed:
            env["GUFO_GATE_PASSED"] = str(self.passed)
            env["GUFO_GATE_KEPT"] = str(self.kept)
        return subprocess.run(["sh", str(RUNNER), str(self.binary), *args],
                              env=env, capture_output=True, text=True, check=False)

    def count(self) -> int:
        return len(self.runs.read_text().splitlines()) if self.runs.exists() else 0

    def test_a_binary_that_passed_is_kept(self) -> None:
        self.assertEqual(self.run_binary().returncode, 0)
        second = self.run_binary()
        self.assertEqual(second.returncode, 0)
        self.assertEqual(self.count(), 1)
        self.assertIn("kept gufo-qwen35 prefill", second.stderr)
        self.assertEqual(self.kept.read_text(), "gufo-qwen35\tprefill\n")
        fields = self.passed.read_text().splitlines()[0].split("\t")
        self.assertEqual(len(fields[0]), 64)
        self.assertEqual(fields[1:3], ["gufo-qwen35", "prefill"])

    def test_a_failed_binary_runs_again(self) -> None:
        self.write_binary(1)
        self.assertEqual(self.run_binary().returncode, 1)
        self.assertEqual(self.run_binary().returncode, 1)
        self.assertEqual(self.count(), 2)
        self.assertFalse(self.passed.exists())

    def test_changed_contents_or_arguments_run_again(self) -> None:
        self.run_binary()
        self.write_binary(0, "rebuilt")
        self.run_binary()
        self.run_binary("--ignored")
        self.assertEqual(self.count(), 3)
        self.run_binary("--ignored")
        self.assertEqual(self.count(), 3)

    def test_without_a_record_every_run_runs(self) -> None:
        self.run_binary(passed=False)
        self.run_binary(passed=False)
        self.assertEqual(self.count(), 2)
        self.assertFalse(self.passed.exists())


if __name__ == "__main__":
    unittest.main()
