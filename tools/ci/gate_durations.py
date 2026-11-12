#!/usr/bin/env python3
"""Keeps the local record of test binary times the gate reads.

tools/ci/check-rust.sh runs every test binary through gate_runner.sh, which
appends "seconds, package, binary" to a per-run file. This script merges such a
run into the record (by default target/gate-durations.tsv, outside Git):
seconds, package, binary and the date measured, slowest first. The times are
the local ones; the scoped gate skips the binaries over its limit.

    python3 tools/ci/gate_durations.py RUN_TIMES RECORD
"""

import sys
import time
from pathlib import Path

HEADER = "# seconds\tpackage\ttest binary\tmeasured (local times; tools/ci/check-rust.sh)\n"


def read_record(path: Path) -> dict:
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            cells = line.split("\t")
            if line.startswith("#") or len(cells) < 3:
                continue
            rows[(cells[1], cells[2])] = cells
    return rows


def merge(run_times: Path, record: Path, today: str) -> dict:
    rows = read_record(record)
    for line in run_times.read_text().splitlines():
        cells = line.split("\t")
        if len(cells) == 3:
            seconds, package, binary = cells
            rows[(package, binary)] = [seconds, package, binary, today]
    record.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda cells: -float(cells[0]))
    record.write_text(HEADER + "".join("\t".join(cells) + "\n" for cells in ordered))
    return rows


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    merge(Path(sys.argv[1]), Path(sys.argv[2]), time.strftime("%Y-%m-%d"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
