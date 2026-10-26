#!/usr/bin/env python3
"""
gate_scope.py — the packages whose tests the scoped Rust gate runs
(tools/ci/check-rust.sh; docs/RUST.md)

Maps every file changed against a base revision (default HEAD: staged,
unstaged and untracked files) to the workspace packages that read it:

1. the package whose directory holds it (longest manifest-directory prefix);
2. the packages whose last build in target/ listed it as an input: the
   rerun-if-changed lines of build scripts (kernel sources and the headers
   of their hipcc depfiles) and rustc dep-info (include_str!,
   include_bytes!);
3. outside package directories, the packages that name the file in a Rust
   string literal (fixtures read at run time).

Every workspace package that depends on a selected one (normal, build or
dev dependency) is selected as well. A file that reaches no package needs no
tests when it is documentation, a tool, a root test file (Python tool
tests, fixtures) or project metadata. Any other file, the workspace manifests,
and a target/ without build records select the full gate.

Prints `full` or the selected package names on one line (empty when no
package is reached); the reasons go to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Set, Tuple

FULL_FILES = {"Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "clippy.toml"}
FULL_PREFIXES = (".cargo/",)
STATIC_PREFIXES = ("docs/", "tools/", "tests/", "licenses/", "assets/", ".agents/", ".devops/", ".github/")
STATIC_FILES = {"LICENSE", "NOTICE", "flake.nix", "flake.lock", ".gitignore"}
# Sources the gate compiles outside Cargo, with the package whose selection
# runs them: the DeepSeek operator checks (run_operator_checks.sh).
CHECK_SOURCES = {"tests/models/deepseek_v4_flash/": "gufo-deepseek-v4-flash-kernels"}
CHECK_SUFFIXES = (".hip", ".sh")
RERUN = re.compile(r"^cargo::?rerun-if-changed=(.+)$")
LITERAL = re.compile(r'"(?:[^"\\\n]|\\.)*"')

Records = Dict[str, Set[str]]


def owner(path: str, dirs: Dict[str, str]) -> Optional[str]:
    """The package whose manifest directory is the longest prefix of `path`."""
    best = None
    for package, directory in dirs.items():
        if path.startswith(directory + "/") and (best is None or len(directory) > len(dirs[best])):
            best = package
    return best


def is_static(path: str) -> bool:
    return path in STATIC_FILES or path.endswith(".md") or path.startswith(STATIC_PREFIXES)


def relative(root: Path, base: Path, name: str) -> Optional[str]:
    """`name` (absolute, or relative to `base`) relative to the repository."""
    path = os.path.normpath(os.path.join(base, name))
    prefix = str(root) + os.sep
    return path[len(prefix):] if path.startswith(prefix) else None


def make_dependencies(line: str) -> List[str]:
    """The prerequisites of a Makefile rule line `target: a b\\ c`."""
    _, _, rest = line.partition(": ")
    return [p.replace("\\ ", " ") for p in re.split(r"(?<!\\)\s+", rest.strip()) if p]


def load_workspace(root: Path) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """Package directories (repository-relative) and reverse dependencies."""
    metadata = json.loads(subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=root, check=True, capture_output=True, text=True).stdout)
    dirs = {p["name"]: relative(root, root, str(Path(p["manifest_path"]).parent)) for p in metadata["packages"]}
    dependents: Dict[str, Set[str]] = defaultdict(set)
    for package in metadata["packages"]:
        for dependency in package["dependencies"]:
            if dependency["name"] in dirs:
                dependents[dependency["name"]].add(package["name"])
    return dirs, dependents


def changed_files(root: Path, base: str) -> List[str]:
    def git(*args: str) -> List[str]:
        out = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout
        return [line for line in out.splitlines() if line]
    return sorted(set(git("diff", "--name-only", "--no-renames", base, "--")
                      + git("ls-files", "--others", "--exclude-standard")))


def build_records(root: Path, dirs: Dict[str, str]) -> Tuple[Records, bool]:
    """Inputs per file from Cargo's build records in target/, and whether any
    build script record of a workspace package exists."""
    records: Records = defaultdict(set)
    target = root / "target"
    found = False

    def add(base: Path, name: str, package: str) -> None:
        path = relative(root, base, name)
        if path is not None:
            records[path].add(package)

    profiles = [p for p in target.iterdir() if (p / ".fingerprint").is_dir()] if target.is_dir() else []
    for profile in profiles:
        for output in profile.glob("build/*/output"):
            package = output.parent.name.rsplit("-", 1)[0]
            if package not in dirs:
                continue
            found = True
            for line in output.read_text(errors="replace").splitlines():
                match = RERUN.match(line)
                if match:
                    add(root / dirs[package], match.group(1), package)
        for depinfo in profile.glob("deps/*.d"):
            with depinfo.open(errors="replace") as f:
                files = make_dependencies(f.readline())
            source = relative(root, root, files[0]) if files else None
            package = owner(source, dirs) if source else None
            if package is not None:
                for name in files:
                    add(root, name, package)
    return records, found


def literal_readers(root: Path, dirs: Dict[str, str], names: Iterable[str]) -> Dict[str, Set[str]]:
    """The packages naming each file name in a Rust string literal."""
    names = set(names)
    if not names:
        return {}
    sources = subprocess.run(["git", "ls-files", "-z", "*.rs"], cwd=root, check=True,
                             capture_output=True, text=True).stdout.split("\0")
    literals: Dict[str, List[str]] = defaultdict(list)
    for source in filter(None, sources):
        package = owner(source, dirs)
        path = root / source
        if package is not None and path.is_file():
            literals[package].extend(LITERAL.findall(path.read_text(errors="replace")))
    joined = {package: "\n".join(found) for package, found in literals.items()}
    readers: Dict[str, Set[str]] = defaultdict(set)
    for name in names:
        for package, text in joined.items():
            if f'"{name}' in text or f"/{name}" in text:
                readers[name].add(package)
    return readers


def select(changed: List[str], dirs: Dict[str, str], dependents: Dict[str, Set[str]], records: Records,
           readers: Dict[str, Set[str]]) -> Tuple[Optional[str], Dict[str, List[str]], Set[str]]:
    """(reason for the full gate, files per directly reached package, selection)."""
    direct: Dict[str, List[str]] = defaultdict(list)
    for path in changed:
        if path in FULL_FILES or path.startswith(FULL_PREFIXES):
            return f"{path} configures the whole workspace", {}, set()
        own = owner(path, dirs)
        packages = set(records.get(path, ()))
        if own is not None:
            packages.add(own)
        else:
            packages |= readers.get(PurePosixPath(path).name, set())
        for prefix, package in CHECK_SOURCES.items():
            if path.startswith(prefix) and path.endswith(CHECK_SUFFIXES):
                packages.add(package)
        if not packages and not is_static(path):
            return f"no package claims {path}", {}, set()
        for package in packages:
            direct[package].append(path)
    selected = set(direct)
    queue = list(direct)
    while queue:
        for dependent in dependents.get(queue.pop(), ()):
            if dependent not in selected:
                selected.add(dependent)
                queue.append(dependent)
    return None, dict(direct), selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--base", default="HEAD", help="revision the changes are taken against")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    dirs, dependents = load_workspace(root)
    changed = changed_files(root, args.base)
    records, found = build_records(root, dirs)
    unowned = [p for p in changed if owner(p, dirs) is None and p not in records]
    readers = literal_readers(root, dirs, {PurePosixPath(p).name for p in unowned})
    reason, direct, selected = select(changed, dirs, dependents, records, readers)
    if reason is None and direct and not found:
        reason = "target/ holds no build records to map kernel and C++ sources"

    def say(text: str) -> None:
        print(text, file=sys.stderr)

    if reason is not None:
        say(f"full gate: {reason}")
        print("full")
        return 0
    reached = sum(len(files) for files in direct.values())
    say(f"{len(changed)} changed file(s) against {args.base}; packages they reach:")
    width = max((len(p) for p in direct), default=0)
    for package in sorted(direct):
        files = sorted(set(direct[package]))
        more = f" (+{len(files) - 1} more)" if len(files) > 1 else ""
        say(f"  {package:<{width}}  {files[0]}{more}")
    if not reached:
        say("  none: GPU and model tests are not needed")
    dependent = sorted(selected - set(direct))
    if dependent:
        say(f"  dependents: {' '.join(dependent)}")
    print(" ".join(sorted(selected)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
