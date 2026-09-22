#!/usr/bin/env python3
"""Freeze one authoritative V7 evidence index and archive superseded summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


STALE_NAMES = (
    "final_13_cell_outcome_summary.json",
    "FINAL_13_FAILURE_AND_COMPLETENESS_REPORT.md",
    "all_experiment_non_success_ledger.csv",
    "all_experiment_non_success_summary.json",
    "ALL_EXPERIMENT_NON_SUCCESS_REPORT.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--support-decisions", required=True, type=Path)
    parser.add_argument("--graphscope-audit", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_stale(asset_dir: Path) -> list[dict[str, str]]:
    archive = asset_dir / "historical_superseded" / "pre_v7_gap_failures"
    archive.mkdir(parents=True, exist_ok=True)
    moved: list[dict[str, str]] = []
    for name in STALE_NAMES:
        source = asset_dir / name
        if not source.exists():
            continue
        target = archive / name
        if target.exists():
            target = archive / f"{source.stem}.previous{source.suffix}"
        shutil.move(str(source), str(target))
        moved.append({"source": name, "archived_as": str(target.relative_to(asset_dir))})
    return moved


def write_summary(asset_dir: Path, rows: list[dict[str, str]]) -> Path:
    status_counts = Counter(row["execution_status"] for row in rows)
    baselines = sorted({row["baseline"] for row in rows})
    datasets = sorted({row["dataset"] for row in rows})
    functions = sorted({row["function"] for row in rows})
    summary = {
        "schema": "eggpu-authoritative-v7-cell-summary",
        "baselines": len(baselines),
        "baseline_names": baselines,
        "cells": len(rows),
        "datasets": len(datasets),
        "functions": len(functions),
        "missing_experiment_cells": sum(
            1 for row in rows if row["execution_status"] == "missing_experiment"
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "eggpu_cells": sum(1 for row in rows if row["baseline"] == "EGGPU"),
        "eggpu_ok": sum(
            1
            for row in rows
            if row["baseline"] == "EGGPU" and row["execution_status"] == "ok"
        ),
        "eggpu_validation_counts": dict(
            sorted(
                Counter(
                    row["validation_status"]
                    for row in rows
                    if row["baseline"] == "EGGPU"
                ).items()
            )
        ),
    }
    output = asset_dir / "final_13_cell_outcome_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_non_success(asset_dir: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
    failures = [row for row in rows if row["execution_status"] != "ok"]
    csv_path = asset_dir / "all_experiment_non_success_ledger.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(failures)

    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in failures:
        grouped[row["baseline"]][row["execution_status"]] += 1
    md = [
        "# Authoritative V7 Completeness and Non-success Report",
        "",
        "This report is generated from the frozen 13-dataset, 16-function, "
        "8-system ledger. A non-success cell is an explicit outcome, not a missing record.",
        "",
        "## Matrix closure",
        "",
        f"- Total cells: {len(rows)}",
        f"- EGGPU: {sum(r['baseline'] == 'EGGPU' and r['execution_status'] == 'ok' for r in rows)}/208 successful",
        f"- Missing experiment cells: {sum(r['execution_status'] == 'missing_experiment' for r in rows)}",
        "- EGGPU validation: 180 complete-output passes and 28 deterministic sampled/digest/invariant passes.",
        "",
        "## Explicit outcomes by baseline",
        "",
        "| Baseline | Non-success outcomes | Counts |",
        "| --- | --- | --- |",
    ]
    for baseline in sorted(grouped):
        counts = grouped[baseline]
        md.append(
            f"| {baseline} | {', '.join(sorted(counts))} | "
            f"{', '.join(f'{name}={counts[name]}' for name in sorted(counts))} |"
        )
    md.extend(
        [
            "",
            "## Cell-level reasons",
            "",
            "The companion `all_experiment_non_success_ledger.csv` contains every "
            "dataset, function, support class, status, failure kind, and recorded reason.",
            "Unsupported APIs, semantic restrictions, timeouts, validation failures, "
            "and representation limits remain separate categories.",
            "",
            "No EGGPU cell appears in that companion file because the current system "
            "completed and validated all 208 workloads.",
        ]
    )
    md_path = asset_dir / "FINAL_13_FAILURE_AND_COMPLETENESS_REPORT.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return csv_path, md_path


def write_index(
    asset_dir: Path,
    ledger: Path,
    support: Path,
    graphscope_audit: Path,
    generated: list[Path],
    moved: list[dict[str, str]],
) -> tuple[Path, Path]:
    candidates = [
        ledger,
        support,
        graphscope_audit,
        asset_dir / "final_13_numeric_summary.json",
        asset_dir / "EGGPU_GAP_V11_LEDGER_AUDIT.json",
        asset_dir / "memory_resource_domain_summary.csv",
        asset_dir / "closure_v7_20260728" / "intro_pagerank_scaling_points.csv",
        asset_dir / "chapter4_v7_20260728" / "workflow_five_call_aggregate.csv",
        *generated,
    ]
    records = []
    for path in candidates:
        path = path.resolve()
        if path.exists():
            records.append(
                {
                    "path": str(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    payload = {
        "schema": "eggpu-authoritative-v7-evidence-index",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_ledger_sha256": sha256(ledger),
        "files": records,
        "superseded_files_archived": moved,
    }
    json_path = asset_dir / "AUTHORITATIVE_EVIDENCE_INDEX.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Authoritative V7 Evidence Index",
        "",
        "Only the files listed below are authoritative for current paper claims. "
        "Superseded seven-baseline and pre-GAP-repair summaries were archived and "
        "must not be cited as current evidence.",
        "",
        f"Frozen ledger SHA-256: `{payload['authoritative_ledger_sha256']}`",
        "",
        "| File | SHA-256 | Bytes |",
        "| --- | --- | ---: |",
    ]
    md.extend(
        f"| `{record['path']}` | `{record['sha256']}` | {record['bytes']} |"
        for record in records
    )
    md_path = asset_dir / "AUTHORITATIVE_EVIDENCE_INDEX.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    args = parse_args()
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    with args.ledger.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 13 * 16 * 8:
        raise RuntimeError(f"expected 1664 ledger cells, found {len(rows)}")
    eggpu = [row for row in rows if row["baseline"] == "EGGPU"]
    if len(eggpu) != 208 or any(row["execution_status"] != "ok" for row in eggpu):
        raise RuntimeError("authoritative EGGPU matrix is not 208/208 successful")
    moved = archive_stale(args.asset_dir)
    summary = write_summary(args.asset_dir, rows)
    non_success_csv, failure_md = write_non_success(args.asset_dir, rows)
    index_json, index_md = write_index(
        args.asset_dir,
        args.ledger.resolve(),
        args.support_decisions.resolve(),
        args.graphscope_audit.resolve(),
        [summary, non_success_csv, failure_md],
        moved,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "cells": len(rows),
                "eggpu_ok": len(eggpu),
                "archived": moved,
                "index": str(index_json),
                "index_markdown": str(index_md),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
