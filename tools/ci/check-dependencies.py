#!/usr/bin/env python3
"""
check-dependencies.py — Third-party dependency inventory consistency checker for gufo

Validates that:
1. The declared direct-dependency inventory includes required components.
2. License identifiers and version/source records are present.
This is a consistency check, not a legal compliance determination.
3. Emits a deterministic machine-readable dependency inventory report.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

# Expected runtime and build dependencies that must be tracked
REQUIRED_SHIPPED_COMPONENTS = {
    "ROCm / HIP",
    "hipBLAS",
    "hipBLASLt",
    "rocBLAS",
    "Composable Kernel",
    "hipCUB",
    "rocPRIM",
    "rocWMMA",
    "ureq / rustls / ring",
    "png / zune-jpeg / image-webp",
    "unicode-normalization / unicode-properties",
    "sha2 / sha1_smol",
    "Other Rust crates",
    "DS4",
    "llama.cpp / ggml",
}

# Crates of the shipped rows above that Cargo.lock must pin.
REQUIRED_LOCKED_CRATES = (
    "ureq",
    "rustls",
    "ring",
    "png",
    "zune-jpeg",
    "image-webp",
    "unicode-normalization",
    "unicode-properties",
    "sha2",
    "sha1_smol",
)

REQUIRED_EVALUATION_COMPONENTS = {
    "PyTorch ROCm",
    "Torchvision",
    "ROCprofiler SDK / ROCTx",
}


def parse_third_party_notices(notices_path: Path) -> Dict[str, Dict[str, str]]:
    """Parse THIRD_PARTY_NOTICES.md markdown table into structured component data."""
    if not notices_path.is_file():
        raise FileNotFoundError(f"Missing notices file: {notices_path}")

    content = notices_path.read_text(encoding="utf-8")
    components: Dict[str, Dict[str, str]] = {}

    for line in content.splitlines():
        line_clean = line.strip()
        if not line_clean.startswith("|") or "Component Name" in line_clean or "---" in line_clean:
            continue

        parts = [p.strip() for p in line_clean.split("|")]
        # Form: | [0] empty | [1] Name | [2] Rel | [3] SPDX | [4] Version | [5] Source | [6] empty |
        if len(parts) >= 6:
            raw_name = parts[1].replace("*", "").replace("`", "").strip()
            if not raw_name:
                continue
            components[raw_name] = {
                "name": raw_name,
                "relationship": parts[2].strip(),
                "spdx": parts[3].replace("`", "").strip(),
                "version": parts[4].replace("`", "").strip(),
                "source": parts[5].strip(),
            }

    return components


def check_container_rocm(build_script: Path, containerfile: Path) -> List[str]:
    """The Containerfile's ROCM_RELEASE must be the QUALIFIED_ROCM of gufo-hip."""
    for source in (build_script, containerfile):
        if not source.is_file():
            return [f"Dependency source {source} does not exist"]
    qualified = re.search(r'^const QUALIFIED_ROCM: &str = "([^"]+)";$',
                          build_script.read_text(encoding="utf-8"), re.MULTILINE)
    release = re.search(r"^ARG ROCM_RELEASE=(\S+)$",
                        containerfile.read_text(encoding="utf-8"), re.MULTILINE)
    if qualified is None or release is None:
        return [f"No QUALIFIED_ROCM in {build_script} or ROCM_RELEASE in {containerfile}"]
    if qualified.group(1) != release.group(1):
        return [f"{containerfile.name} installs ROCm {release.group(1)}, but "
                f"{build_script} qualifies {qualified.group(1)}"]
    return []


def verify_dependencies(
    notices_file: Path,
    cargo_lock_file: Path,
    requirements_file: Path,
    json_report_path: Path = None,
    qualified_rocm_file: Path = Path("src/hip/build.rs"),
    containerfile: Path = Path("Containerfile"),
) -> int:
    errors: List[str] = []
    components = parse_third_party_notices(notices_file)

    # 1. Check that all required components have their own inventory rows.
    # Exact names: a substring would also match rows such as "DS4 GB10/GX10 fork".
    for req in REQUIRED_SHIPPED_COMPONENTS:
        if req not in components:
            errors.append(f"Required shipped dependency '{req}' is missing from {notices_file.name}")
    for req in REQUIRED_EVALUATION_COMPONENTS:
        if req not in components:
            errors.append(
                f"Required evaluation dependency '{req}' is missing from "
                f"{notices_file.name}"
            )

    # 2. Check that Cargo.lock pins the crates of the shipped rows. ROCm is
    # the installed one, so no file pins it.
    for source in (cargo_lock_file, requirements_file):
        if not source.is_file():
            errors.append(f"Dependency source {source} does not exist")
    if cargo_lock_file.is_file():
        locked = set(re.findall(r'^name = "([^"]+)"$',
                                cargo_lock_file.read_text(encoding="utf-8"), re.MULTILINE))
        for crate in REQUIRED_LOCKED_CRATES:
            if crate not in locked:
                errors.append(f"Expected crate '{crate}' missing from {cargo_lock_file.name}")

    # 3. Check that the Python tools pin the ROCm builds of the evaluation
    # dependencies.
    if requirements_file.is_file():
        lines = [line.strip() for line in
                 requirements_file.read_text(encoding="utf-8").splitlines()]
        if not any(line.startswith("--extra-index-url https://download.pytorch.org/whl/rocm")
                   for line in lines):
            errors.append(
                f"Expected the PyTorch ROCm wheel index in {requirements_file.name}"
            )
        for dep in ("torch", "torchvision"):
            if not any(re.fullmatch(rf"{dep}==[0-9.]+\+rocm[0-9.]+", line)
                       for line in lines):
                errors.append(
                    f"Expected evaluation dependency '{dep}' pinned to a ROCm build "
                    f"in {requirements_file.name}"
                )

    # 4. Check that the container image installs the qualified ROCm release.
    errors.extend(check_container_rocm(qualified_rocm_file, containerfile))

    # 5. Check for invalid or empty SPDX licenses
    for name, info in components.items():
        spdx = info.get("spdx", "")
        if not spdx or spdx.lower() == "unknown" or spdx.lower() == "todo":
            errors.append(f"Component '{name}' has invalid SPDX license: '{spdx}'")
        if not info.get("version") or info.get("version").lower() == "todo":
            errors.append(f"Component '{name}' has missing pinned version")

    # Generate deterministic inventory report
    report = {
        "status": "FAIL" if errors else "PASS",
        "tracked_components_count": len(components),
        "inventory": components,
        "errors_count": len(errors),
        "errors": errors,
    }

    if json_report_path:
        json_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    if errors:
        print(f"\n[check-dependencies] FAILED: {len(errors)} dependency consistency errors found:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(
        f"[check-dependencies] PASSED: All {len(components)} third-party dependencies "
        f"have license/version records and required package bindings."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify third-party dependency notices and license inventory consistency."
    )
    parser.add_argument(
        "--notices",
        default="THIRD_PARTY_NOTICES.md",
        help="Path to THIRD_PARTY_NOTICES.md",
    )
    parser.add_argument(
        "--cargo-lock",
        default="Cargo.lock",
        help="Path to Cargo.lock",
    )
    parser.add_argument(
        "--requirements",
        default="tools/requirements.txt",
        help="Path to the Python tools' requirements",
    )
    parser.add_argument(
        "--json-report",
        help="Path to output machine-readable JSON inventory report",
    )

    args = parser.parse_args()

    notices_path = Path(args.notices)
    cargo_lock_path = Path(args.cargo_lock)
    requirements_path = Path(args.requirements)
    report_path = Path(args.json_report) if args.json_report else None

    return verify_dependencies(
        notices_path, cargo_lock_path, requirements_path, report_path
    )


if __name__ == "__main__":
    sys.exit(main())
