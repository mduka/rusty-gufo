#!/usr/bin/env python3
"""Tests for tools/ci/gate_durations.py, the gate's local record of test binary
times."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "ci"))
import gate_durations as d  # noqa: E402


class GateDurationsTest(unittest.TestCase):
    def test_a_run_replaces_what_it_measured_and_keeps_the_rest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record, run = Path(tmp, "record.tsv"), Path(tmp, "run.tsv")
            run.write_text("3.0\tgufo-qwen35\tdecode\n187.5\tgufo-qwen35\tprefill\n")
            d.merge(run, record, "2026-01-01")
            # A scoped run that skipped prefill measures decode only.
            run.write_text("2.5\tgufo-qwen35\tdecode\n")
            d.merge(run, record, "2026-01-02")
            lines = record.read_text().splitlines()
            self.assertTrue(lines[0].startswith("#"))
            self.assertEqual(lines[1:], [
                "187.5\tgufo-qwen35\tprefill\t2026-01-01",
                "2.5\tgufo-qwen35\tdecode\t2026-01-02",
            ])

    def test_a_missing_record_starts_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record, run = Path(tmp, "new", "record.tsv"), Path(tmp, "run.tsv")
            run.write_text("")
            d.merge(run, record, "2026-01-01")
            self.assertEqual(record.read_text(), d.HEADER)


if __name__ == "__main__":
    unittest.main()
