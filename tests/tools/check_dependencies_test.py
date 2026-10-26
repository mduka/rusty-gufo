#!/usr/bin/env python3
"""CPU-only regressions for the third-party dependency inventory check."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_dependencies", ROOT / "tools/ci/check-dependencies.py")
DEPENDENCIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPENDENCIES)
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
LOCK = ROOT / "Cargo.lock"
FLAKE = ROOT / "flake.nix"


class CheckDependenciesTest(unittest.TestCase):
    def verify(self, notices, lock=LOCK, flake=FLAKE):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            return DEPENDENCIES.verify_dependencies(notices, lock, flake), output.getvalue()

    def test_repository_inventory_passes(self):
        status, output = self.verify(NOTICES)
        self.assertEqual(status, 0, output)

    def test_each_required_row_is_needed(self):
        lines = NOTICES.read_text().splitlines(keepends=True)
        required = (DEPENDENCIES.REQUIRED_SHIPPED_COMPONENTS
                    | DEPENDENCIES.REQUIRED_EVALUATION_COMPONENTS)
        with tempfile.TemporaryDirectory() as directory:
            notices = Path(directory) / "THIRD_PARTY_NOTICES.md"
            for name in sorted(required):
                with self.subTest(component=name):
                    row = [line for line in lines if line.startswith("|")
                           and line.split("|")[1].replace("*", "").replace("`", "").strip()
                           == name]
                    self.assertEqual(len(row), 1)
                    notices.write_text("".join(line for line in lines if line is not row[0]))
                    status, output = self.verify(notices)
                    self.assertNotEqual(status, 0)
                    self.assertIn(f"'{name}' is missing", output)

    def test_each_locked_crate_is_needed(self):
        lines = LOCK.read_text().splitlines(keepends=True)
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "Cargo.lock"
            for crate in DEPENDENCIES.REQUIRED_LOCKED_CRATES:
                with self.subTest(crate=crate):
                    name = f'name = "{crate}"\n'
                    self.assertIn(name, lines)
                    lock.write_text("".join(line for line in lines if line != name))
                    status, output = self.verify(NOTICES, lock)
                    self.assertNotEqual(status, 0)
                    self.assertIn(f"crate '{crate}' missing", output)

    def test_missing_dependency_sources_fail(self):
        missing = ROOT / "does-not-exist"
        for lock, flake in ((missing, FLAKE), (LOCK, missing)):
            with self.subTest(lock=lock.name, flake=flake.name):
                status, output = self.verify(NOTICES, lock, flake)
                self.assertNotEqual(status, 0)
                self.assertIn("does-not-exist does not exist", output)


if __name__ == "__main__":
    unittest.main()
