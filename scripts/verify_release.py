#!/usr/bin/env python3
"""Fail closed when the public framework artifact contains internal material."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "environment.yml",
    "artifact/ARCHITECTURE.md",
    "artifact/FUNCTION_SUPPORT.md",
    "artifact/PAPER_CODE_MAP.md",
    "artifact/CORRECTNESS.md",
    "artifact/datasets/README.md",
    "artifact/datasets/manifest.tsv",
    "artifact/datasets/zenodo.json",
    "scripts/build_eggpu.sh",
    "scripts/check_eggpu_compat.py",
    "scripts/correctness_smoke.py",
    "scripts/run_smoke.sh",
)
FORBIDDEN_TOP_LEVEL = ("EG_Evaluation", "writing", "results")
FORBIDDEN_PATH_PARTS = {"__pycache__", ".pytest_cache", "build", ".eggs"}
TEXT_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".md", ".py", ".sh", ".txt", ".yml", ".yaml", ".json", ".cff", ".tsv"}
SENSITIVE = (
    # Assemble the local workspace prefix so this audit rule does not match
    # its own source file when the verifier itself is tracked.
    re.compile(re.escape("/home/" + "dataset-assist-0/")),
    re.compile(r"connect\.bjb2\.seetacloud\.com", re.I),
    re.compile(r"(?:password|passwd)\s*[:=]", re.I),
    re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
)


def tracked_files() -> list[Path]:
    """Return regular files in Git's index, never generated/untracked files."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--cached", "-z"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot enumerate Git-tracked release files: {exc}") from exc

    paths = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode("utf-8", errors="surrogateescape"))
        path = ROOT / relative
        if path.is_file():
            paths.append(path)
    return paths


def main() -> int:
    errors = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")
    for name in FORBIDDEN_TOP_LEVEL:
        if (ROOT / name).exists():
            errors.append(f"forbidden top-level artifact: {name}")

    try:
        candidates = tracked_files()
    except RuntimeError as exc:
        errors.append(str(exc))
        candidates = []

    for path in candidates:
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PATH_PARTS for part in relative.parts):
            errors.append(f"generated path present: {relative}")
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SENSITIVE:
            if pattern.search(text):
                errors.append(f"sensitive/internal text in {relative}: {pattern.pattern}")

    if errors:
        for error in sorted(set(errors)):
            print(f"ERROR {error}", file=sys.stderr)
        return 2
    print("PASS release content and path checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
