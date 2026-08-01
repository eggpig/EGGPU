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
    "scripts/build_eggpu.sh",
    "scripts/check_eggpu_compat.py",
    "scripts/correctness_smoke.py",
    "scripts/run_smoke.sh",
)
ALLOWED_TOP_LEVEL_DIRS = {"Easy-Graph", "artifact", "scripts"}
ALLOWED_TOP_LEVEL_FILES = {
    ".gitignore",
    "CITATION.cff",
    "LICENSE",
    "NOTICE",
    "README.md",
    "environment.yml",
}
GENERATED_PATH_PARTS = {"__pycache__", ".pytest_cache", "build", ".eggs"}
INTERNAL_PATH_PARTS = {
    "benchmarking",
    "results",
    "writing",
    "EG" + "_Evaluation",
}
SENSITIVE = (
    re.compile(r"/(?:home|Users|users)/[^/\s]+/"),
    re.compile(
        r"(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)"
        r"\s*[:=]\s*[^\s,;]+",
        re.I,
    ),
    re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    re.compile(r"(?:root|admin|ubuntu)@[A-Za-z0-9.-]+", re.I),
)
INTERNAL_REFERENCES = (
    re.compile("EG" + r"_Evaluation"),
    re.compile(r"(?:^|[/\\])" + "bench" + r"marking(?:[/\\]|$)"),
    re.compile(r"(?:^|[/\\])" + "writ" + r"ing(?:[/\\]|$)"),
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
    try:
        candidates = tracked_files()
    except RuntimeError as exc:
        errors.append(str(exc))
        candidates = []

    for path in candidates:
        relative = path.relative_to(ROOT)
        top = relative.parts[0]
        if len(relative.parts) == 1:
            if top not in ALLOWED_TOP_LEVEL_FILES:
                errors.append(f"unexpected top-level file: {relative}")
        elif top not in ALLOWED_TOP_LEVEL_DIRS:
            errors.append(f"unexpected top-level directory: {top}")

        if any(part in GENERATED_PATH_PARTS for part in relative.parts):
            errors.append(f"generated path present: {relative}")
            continue
        if any(part in INTERNAL_PATH_PARTS for part in relative.parts):
            errors.append(f"internal path present: {relative}")
            continue
        if not path.is_file():
            continue
        try:
            raw = path.read_bytes()
            if b"\0" in raw:
                continue
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SENSITIVE:
            if pattern.search(text):
                errors.append(f"sensitive/internal text in {relative}: {pattern.pattern}")
        for pattern in INTERNAL_REFERENCES:
            if pattern.search(text):
                errors.append(f"internal repository reference in {relative}")

    if errors:
        for error in sorted(set(errors)):
            print(f"ERROR {error}", file=sys.stderr)
        return 2
    print("PASS release content and path checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
