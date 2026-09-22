#!/usr/bin/env python3
"""Revalidate one saved large-graph Gunrock PageRank record without retiming it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import run_full_baselines as shared
from run_gunrock_large_matrix import flatten, validate_vector_output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-large-result", type=Path, required=True)
    parser.add_argument("--output-large-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eggpu-result-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=100.0)
    parser.add_argument("--pagerank-alpha", type=float, default=0.75)
    parser.add_argument("--pagerank-tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()

    source = args.source_large_result.resolve()
    output = args.output_large_result.resolve()
    manifest_path = args.manifest.resolve()
    source_json = source / "gunrock_large_matrix.json"
    records = json.loads(source_json.read_text(encoding="utf-8"))
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata["_manifest_path"] = str(manifest_path)
    dataset = str(metadata["name"])
    matches = [
        index for index, record in enumerate(records)
        if record.get("dataset") == dataset and record.get("function") == "PageRank"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {dataset}/PageRank record, found {len(matches)}")
    index = matches[0]
    original = records[index]
    matrix_path = Path(original["matrix_market_input"]).resolve()
    executable = shared.find_gunrock_exe("pr")
    if executable is None:
        raise FileNotFoundError("pinned Gunrock PageRank executable is unavailable")

    output.mkdir(parents=True, exist_ok=True)
    validation_args = SimpleNamespace(
        out_dir=output,
        gpu=args.gpu,
        timeout=args.timeout,
        pagerank_alpha=args.pagerank_alpha,
        pagerank_tolerance=args.pagerank_tolerance,
        eggpu_result_dir=args.eggpu_result_dir.resolve(),
    )
    probe = validate_vector_output(
        validation_args,
        metadata,
        "PageRank",
        matrix_path,
        executable,
    )
    corrected = dict(original)
    corrected["external_validation_probe"] = probe
    corrected["revalidation"] = {
        "policy": "fixed-point residual; timing and memory samples are unchanged",
        "source_record": str(source / f"{dataset}_PageRank.json"),
        "source_record_sha256": sha256(source / f"{dataset}_PageRank.json"),
        "measurement_window": "separate_unmeasured_validation_process",
    }
    if probe["status"] == "pass":
        corrected["status"] = "ok"
        corrected["failure_kind"] = ""
        corrected["validation"] = "pass"
        corrected["validation_note"] = probe["note"]
    else:
        corrected["status"] = "semantic_mismatch"
        corrected["failure_kind"] = probe.get("failure_kind", "validation_failed")
        corrected["validation"] = "semantic_mismatch"
        corrected["validation_note"] = probe["note"]
    records[index] = corrected

    (output / f"{dataset}_PageRank.json").write_text(
        json.dumps(corrected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    aggregate_json = output / "gunrock_large_matrix.json"
    aggregate_json.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = [flatten(record) for record in records]
    fields = sorted({key for row in rows for key in row})
    aggregate_csv = output / "gunrock_large_matrix.csv"
    with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": probe["status"],
        "dataset": dataset,
        "function": "PageRank",
        "policy": "fixed-point residual; no timing or memory sample was rerun",
        "source_aggregate": str(source_json),
        "source_aggregate_sha256": sha256(source_json),
        "derived_aggregate": str(aggregate_json),
        "derived_aggregate_sha256": sha256(aggregate_json),
        "probe": probe,
    }
    (output / "PAGERANK_REVALIDATION.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "PAGERANK_REVALIDATION.md").write_text(
        "\n".join([
            "# Gunrock Large PageRank Revalidation",
            "",
            f"- Status: **{probe['status']}**.",
            "- Authority: PageRank fixed-point equation with the paper alpha.",
            "- Measurement: separate unmeasured result-export process; saved timing and memory samples are unchanged.",
            f"- Mean residual: `{probe.get('fixed_point_evidence', {}).get('mean_residual')}`.",
            f"- Residual threshold: `{probe.get('fixed_point_evidence', {}).get('mean_residual_tolerance')}`.",
            f"- Original cross-solver sample max absolute difference: `{probe.get('max_abs_error')}` (diagnostic only).",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if probe["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
