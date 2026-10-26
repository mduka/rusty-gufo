#!/usr/bin/env python3
"""Tests for tools/ci/gate_scope.py, the test selection of the scoped gate."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "ci"))
import gate_scope as g  # noqa: E402

DIRS = {
    "gufo": "src/cli",
    "gufo-serve": "src/cli/serve",
    "gufo-core": "src/core",
    "gufo-qwen": "src/models/qwen",
    "gufo-qwen-kernels": "src/models/qwen/kernels",
}
DEPENDENTS = {
    "gufo-core": {"gufo-serve", "gufo-qwen"},
    "gufo-serve": {"gufo"},
    "gufo-qwen-kernels": {"gufo-qwen"},
    "gufo-qwen": {"gufo"},
}


def select(changed, records=None, readers=None):
    return g.select(changed, DIRS, DEPENDENTS, records or {}, readers or {})


class GateScopeTest(unittest.TestCase):
    def test_owner_takes_the_longest_directory(self) -> None:
        self.assertEqual(g.owner("src/cli/serve/src/lib.rs", DIRS), "gufo-serve")
        self.assertEqual(g.owner("src/cli/src/main.rs", DIRS), "gufo")
        self.assertEqual(g.owner("src/models/qwen/kernels/launch.toml", DIRS), "gufo-qwen-kernels")
        self.assertIsNone(g.owner("src/clix/a.rs", DIRS))
        self.assertIsNone(g.owner("docs/a.md", DIRS))

    def test_dependents_are_selected(self) -> None:
        reason, direct, selected = select(["src/models/qwen/kernels/src/lib.rs"])
        self.assertIsNone(reason)
        self.assertEqual(set(direct), {"gufo-qwen-kernels"})
        self.assertEqual(selected, {"gufo-qwen-kernels", "gufo-qwen", "gufo"})

    def test_build_records_reach_the_compiling_package(self) -> None:
        records = {"src/models/qwen/hip/gemv.hip": {"gufo-qwen-kernels"}}
        _, direct, selected = select(["src/models/qwen/hip/gemv.hip"], records)
        self.assertEqual(set(direct), {"gufo-qwen", "gufo-qwen-kernels"})
        self.assertIn("gufo", selected)

    def test_files_outside_packages(self) -> None:
        readers = {"schema.json": {"gufo"}}
        _, direct, selected = select(["tests/fixtures/schema.json"], readers=readers)
        self.assertEqual(selected, {"gufo"})
        reason, direct, selected = select(["docs/A.md", "tools/x.py", "tests/models/x_test.cpp", "flake.lock"])
        self.assertEqual((reason, direct, selected), (None, {}, set()))

    def test_operator_check_sources_reach_their_kernels(self) -> None:
        _, direct, selected = select(["tests/models/deepseek_v4_flash/attention_test.hip"])
        self.assertEqual(selected, {"gufo-deepseek-v4-flash-kernels"})
        reason, direct, selected = select(["tests/models/deepseek_v4_flash/dataset_test.py"])
        self.assertEqual((reason, selected), (None, set()))

    def test_full_gate(self) -> None:
        for path in ("Cargo.lock", "Cargo.toml", ".cargo/config.toml", "rust-toolchain.toml"):
            self.assertIn(path, select(["src/core/src/lib.rs", path])[0])
        self.assertIn("no package claims", select(["src/testing/compare/x.txt"])[0])

    def test_make_dependencies(self) -> None:
        line = "/t/deps/a-1.d: src/a.rs src/../x/b.json c\\ d.txt\n"
        self.assertEqual(g.make_dependencies(line), ["src/a.rs", "src/../x/b.json", "c d.txt"])
        root = Path("/r")
        self.assertEqual(g.relative(root, root / "src/a", "../../x/y.json"), "x/y.json")
        self.assertIsNone(g.relative(root, root, "/usr/include/z.h"))


if __name__ == "__main__":
    unittest.main()
