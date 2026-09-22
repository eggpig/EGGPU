#!/usr/bin/env python3
"""Consolidate disjoint large-graph CPU shards into the canonical result view."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from run_cpu_large_matrix import FUNCTIONS, write_outputs


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_key(record: dict) -> tuple[str, str, str]:
    return (
        str(record.get("dataset", "")),
        str(record.get("baseline", "")),
        str(record.get("function", "")),
    )


def record_score(record: dict) -> tuple[int, int, int]:
    status = str(record.get("status", ""))
    status_score = {
        "ok": 4,
        "skipped": 3,
        "failed": 2,
    }.get(status, 1)
    timing = int(record.get("timing_samples") or record.get("successful_timing_processes") or 0)
    memory = int(record.get("memory_samples") or 0)
    return status_score, timing, memory


def qualification_key(record: dict) -> tuple[str, str]:
    return str(record.get("dataset", "")), str(record.get("baseline", ""))


def load_json_list(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected JSON list: {path}")
    return [dict(item) for item in data]


def load_result_records(source: Path) -> list[dict]:
    records = load_json_list(source / "cpu_large_matrix.json")
    by_key = {record_key(record): record for record in records}
    for path in sorted(source.glob("*.json")):
        if path.name in {
            "cpu_large_matrix.json",
            "build_qualifications.json",
            "CPU_LARGE_MATRIX_CONSOLIDATION.json",
        } or path.name.endswith("_build.json"):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or record.get("function") not in FUNCTIONS:
            continue
        key = record_key(record)
        previous = by_key.get(key)
        if previous is None or record_score(record) > record_score(previous):
            by_key[key] = dict(record)
    return list(by_key.values())


def load_qualifications(source: Path) -> list[dict]:
    qualifications = load_json_list(source / "build_qualifications.json")
    by_key = {qualification_key(record): record for record in qualifications}
    for path in sorted(source.glob("*_build.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        key = qualification_key(record)
        if not all(key):
            continue
        previous = by_key.get(key)
        if previous is None or record_score(record) > record_score(previous):
            by_key[key] = dict(record)
    return list(by_key.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--source", action="append", required=True, type=Path)
    args = parser.parse_args()

    canonical = args.canonical.resolve()
    sources = [canonical, *(path.resolve() for path in args.source)]
    canonical.mkdir(parents=True, exist_ok=True)

    records: dict[tuple[str, str, str], dict] = {}
    qualifications: dict[tuple[str, str], dict] = {}
    provenance = []
    conflicts = []

    for source in sources:
        matrix_path = source / "cpu_large_matrix.json"
        qualification_path = source / "build_qualifications.json"
        provenance.append(
            {
                "source": str(source),
                "matrix_sha256": digest(matrix_path) if matrix_path.is_file() else "",
                "qualification_sha256": (
                    digest(qualification_path) if qualification_path.is_file() else ""
                ),
            }
        )
        for record in load_result_records(source):
            key = record_key(record)
            previous = records.get(key)
            if previous is None or record_score(record) > record_score(previous):
                if previous is not None:
                    conflicts.append(
                        {
                            "key": key,
                            "selected_source": str(source),
                            "old_score": record_score(previous),
                            "new_score": record_score(record),
                        }
                    )
                selected = dict(record)
                selected["consolidated_source"] = str(source)
                records[key] = selected
        for qualification in load_qualifications(source):
            key = qualification_key(qualification)
            previous = qualifications.get(key)
            if previous is None or record_score(qualification) > record_score(previous):
                selected = dict(qualification)
                selected["consolidated_source"] = str(source)
                qualifications[key] = selected

    ordered_records = [records[key] for key in sorted(records)]
    ordered_qualifications = [qualifications[key] for key in sorted(qualifications)]
    write_outputs(canonical, ordered_records, ordered_qualifications)

    for record in ordered_records:
        dataset, baseline, function = record_key(record)
        path = canonical / f"{dataset}_{baseline}_{function}.json"
        path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    evidence_root = canonical / "parallel_shard_evidence"
    evidence_root.mkdir(exist_ok=True)
    for source in sources[1:]:
        target = evidence_root / source.name
        target.mkdir(exist_ok=True)
        for filename in ("cpu_large_matrix.json", "cpu_large_matrix.csv", "build_qualifications.json"):
            src = source / filename
            if src.is_file():
                shutil.copy2(src, target / filename)

    manifest = {
        "status": "complete",
        "canonical": str(canonical),
        "record_count": len(ordered_records),
        "qualification_count": len(ordered_qualifications),
        "sources": provenance,
        "selection_conflicts": conflicts,
    }
    (canonical / "CPU_LARGE_MATRIX_CONSOLIDATION.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
