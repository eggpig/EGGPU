#!/usr/bin/env python3
"""Assemble the immutable, provenance-bound v5 large-graph evidence core.

The assembler never edits source result directories.  It selects one
authoritative main-protocol outcome per (dataset, function), keeps extended
and subset workloads as supplements, and records the source and SHA-256 of
every selected artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path


FUNCTIONS = [
    "PageRank",
    "MST",
    "LCC",
    "WCC",
    "SCC",
    "BFS",
    "Dijkstra",
    "BellmanFord",
    "SSSP",
    "KCore",
    "BC",
    "Closeness",
    "EffectiveSize",
    "Efficiency",
    "Constraint",
    "Hierarchy",
]
GAP_FAST = {
    "PageRank",
    "WCC",
    "BFS",
    "Dijkstra",
    "BellmanFord",
    "SSSP",
    "Closeness",
}
GAP_REPRESENTATION_LIMIT = {"MST", "LCC", "KCore"}
GAP_STRUCTURAL = {"EffectiveSize", "Efficiency", "Constraint", "Hierarchy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-core", required=True, type=Path)
    parser.add_argument("--orkut", required=True, type=Path)
    parser.add_argument("--gap-fast", required=True, type=Path)
    parser.add_argument("--gap-bc", required=True, type=Path)
    parser.add_argument("--gap-scc-main-probe", required=True, type=Path)
    parser.add_argument("--gap-scc-extended", required=True, type=Path)
    parser.add_argument("--gap-structural-probe", required=True, type=Path)
    parser.add_argument("--gap-structural-subset", required=True, type=Path)
    parser.add_argument("--gunrock-large", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def select_dataset(rows: list[dict[str, str]], dataset: str) -> list[dict[str, str]]:
    return [dict(row) for row in rows if row.get("dataset") == dataset]


def select_functions(
    rows: list[dict[str, str]], dataset: str, functions: set[str]
) -> list[dict[str, str]]:
    return [
        dict(row)
        for row in rows
        if row.get("dataset") == dataset and row.get("function") in functions
    ]


def timing_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("measurement") == "timing"]


def validate_success_contract(
    rows: list[dict[str, str]], dataset: str, functions: set[str]
) -> None:
    selected = timing_rows(select_functions(rows, dataset, functions))
    by_function = {row.get("function"): row for row in selected}
    if set(by_function) != functions:
        raise ValueError(
            f"{dataset}: timing functions differ: "
            f"expected={sorted(functions)}, observed={sorted(by_function)}"
        )
    for function, row in by_function.items():
        if row.get("status") != "ok":
            raise ValueError(f"{dataset}/{function}: expected ok, got {row.get('status')}")
        if row.get("result_validation") != "pass":
            raise ValueError(
                f"{dataset}/{function}: validation={row.get('result_validation')}"
            )
        if int(float(row.get("timing_process_samples") or 0)) != 5:
            raise ValueError(
                f"{dataset}/{function}: timing_process_samples="
                f"{row.get('timing_process_samples')}"
            )
        memory = [
            item
            for item in rows
            if item.get("dataset") == dataset
            and item.get("function") == function
            and item.get("measurement") == "memory"
            and item.get("status") == "ok"
        ]
        if len(memory) != 3:
            raise ValueError(f"{dataset}/{function}: memory rows={len(memory)}, expected=3")


def validate_probe(
    rows: list[dict[str, str]], functions: set[str], label: str
) -> None:
    selected = timing_rows(rows)
    by_function = {row.get("function"): row for row in selected}
    if set(by_function) != functions:
        raise ValueError(
            f"{label}: expected={sorted(functions)}, observed={sorted(by_function)}"
        )
    for function, row in by_function.items():
        if row.get("status") not in {"failed", "skipped"}:
            raise ValueError(f"{label}/{function}: unexpected status {row.get('status')}")
        if row.get("failure_kind") != "timeout":
            raise ValueError(
                f"{label}/{function}: expected timeout, got {row.get('failure_kind')}"
            )


def copy_raw_aggregates(
    source: Path,
    output_raw: Path,
    dataset: str,
    functions: set[str],
    manifest: list[dict[str, object]],
) -> None:
    output_raw.mkdir(parents=True, exist_ok=True)
    for function in sorted(functions):
        source_path = source / "raw" / f"{dataset}_{function}_timing.json"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        target_path = output_raw / source_path.name
        shutil.copy2(source_path, target_path)
        manifest.append(
            {
                "role": "EGGPU raw timing aggregate",
                "dataset": dataset,
                "function": function,
                "source": str(source_path.resolve()),
                "source_sha256": sha256(source_path),
                "selected_copy": str(target_path.resolve()),
                "selected_copy_sha256": sha256(target_path),
            }
        )


def copy_table(
    source: Path,
    target: Path,
    role: str,
    manifest: list[dict[str, object]],
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    manifest.append(
        {
            "role": role,
            "source": str(source.resolve()),
            "source_sha256": sha256(source),
            "selected_copy": str(target.resolve()),
            "selected_copy_sha256": sha256(target),
        }
    )


def main() -> None:
    args = parse_args()
    paths = {
        key: value.resolve()
        for key, value in vars(args).items()
        if isinstance(value, Path)
    }
    output = paths["output"]
    if output.exists():
        raise FileExistsError(
            f"refusing to modify existing authoritative output: {output}"
        )
    output.mkdir(parents=True)

    old_eggpu_path = paths["old_core"] / "eggpu_large_matrix" / "scaling_all.csv"
    orkut_path = paths["orkut"] / "scaling_all.csv"
    gap_fast_path = paths["gap_fast"] / "scaling_all.csv"
    gap_bc_path = paths["gap_bc"] / "scaling_all.csv"
    gap_scc_probe_path = paths["gap_scc_main_probe"] / "scaling_all.csv"
    gap_scc_extended_path = paths["gap_scc_extended"] / "scaling_all.csv"
    gap_structural_probe_path = paths["gap_structural_probe"] / "scaling_all.csv"
    gap_structural_subset_path = paths["gap_structural_subset"] / "scaling_all.csv"

    old_rows = read_csv(old_eggpu_path)
    orkut_rows = select_dataset(read_csv(orkut_path), "com-Orkut")
    gap_fast_rows = select_functions(
        read_csv(gap_fast_path), "GAP-twitter", GAP_FAST
    )
    gap_bc_rows = select_functions(read_csv(gap_bc_path), "GAP-twitter", {"BC"})
    gap_scc_probe_rows = select_functions(
        read_csv(gap_scc_probe_path), "GAP-twitter", {"SCC"}
    )
    gap_structural_probe_rows = select_functions(
        read_csv(gap_structural_probe_path), "GAP-twitter", GAP_STRUCTURAL
    )
    representation_rows = select_functions(
        old_rows, "GAP-twitter", GAP_REPRESENTATION_LIMIT
    )

    validate_success_contract(orkut_rows, "com-Orkut", set(FUNCTIONS))
    validate_success_contract(gap_fast_rows, "GAP-twitter", GAP_FAST)
    validate_success_contract(gap_bc_rows, "GAP-twitter", {"BC"})
    validate_probe(gap_scc_probe_rows, {"SCC"}, "GAP SCC main-protocol probe")
    validate_probe(
        gap_structural_probe_rows,
        GAP_STRUCTURAL,
        "GAP structural-hole all-node main-protocol probe",
    )

    representation_timing = timing_rows(representation_rows)
    if {row.get("function") for row in representation_timing} != GAP_REPRESENTATION_LIMIT:
        raise ValueError("GAP representation-limit rows are incomplete")
    for row in representation_timing:
        if row.get("status") != "skipped" or row.get("failure_kind") != "representation_limit":
            raise ValueError(
                f"GAP/{row.get('function')}: invalid representation-limit row"
            )

    main_rows = (
        orkut_rows
        + gap_fast_rows
        + gap_bc_rows
        + gap_scc_probe_rows
        + gap_structural_probe_rows
        + representation_rows
    )
    gap_timing = timing_rows(select_dataset(main_rows, "GAP-twitter"))
    gap_functions = [row.get("function") for row in gap_timing]
    if len(gap_functions) != len(FUNCTIONS) or set(gap_functions) != set(FUNCTIONS):
        raise ValueError(
            f"GAP main outcomes are not exactly 16 cells: {gap_functions}"
        )

    eggpu_output = output / "eggpu_large_matrix"
    selected_csv = eggpu_output / "scaling_all.csv"
    write_csv(selected_csv, main_rows)

    selected_artifacts: list[dict[str, object]] = []
    for role, path in (
        ("historical static representation-limit evidence", old_eggpu_path),
        ("v5 com-Orkut all-function result", orkut_path),
        ("v5 GAP main-protocol fast-function result", gap_fast_path),
        ("v5 GAP BC result", gap_bc_path),
        ("v5 GAP SCC 100-second probe", gap_scc_probe_path),
        ("v5 GAP structural-hole all-node 100-second probe", gap_structural_probe_path),
    ):
        selected_artifacts.append(
            {"role": role, "source": str(path), "source_sha256": sha256(path)}
        )
    selected_artifacts.append(
        {
            "role": "assembled v5 EGGPU large main table",
            "selected_copy": str(selected_csv),
            "selected_copy_sha256": sha256(selected_csv),
        }
    )

    output_raw = eggpu_output / "raw"
    copy_raw_aggregates(
        paths["orkut"],
        output_raw,
        "com-Orkut",
        set(FUNCTIONS),
        selected_artifacts,
    )
    copy_raw_aggregates(
        paths["gap_fast"],
        output_raw,
        "GAP-twitter",
        GAP_FAST,
        selected_artifacts,
    )
    copy_raw_aggregates(
        paths["gap_bc"],
        output_raw,
        "GAP-twitter",
        {"BC"},
        selected_artifacts,
    )

    copy_table(
        paths["old_core"] / "cpu_large_matrix" / "cpu_large_matrix.csv",
        output / "cpu_large_matrix" / "cpu_large_matrix.csv",
        "unchanged sparse-native CPU large table",
        selected_artifacts,
    )
    copy_table(
        paths["old_core"]
        / "nxcugraph_large_matrix"
        / "nxcugraph_large_matrix.csv",
        output / "nxcugraph_large_matrix" / "nxcugraph_large_matrix.csv",
        "unchanged strict nx-cugraph large table",
        selected_artifacts,
    )
    gunrock_source = paths["gunrock_large"] / "gunrock_large_matrix.csv"
    copy_table(
        gunrock_source,
        output / "gunrock_large_matrix" / "gunrock_large_matrix.csv",
        "latest correctness-qualified Gunrock large table",
        selected_artifacts,
    )

    supplement_records = []
    for name, source, csv_path, semantics in (
        (
            "gap_scc_extended_600s",
            paths["gap_scc_extended"],
            gap_scc_extended_path,
            "same all-node SCC semantics; 600-second extended timeout; supplement only",
        ),
        (
            "gap_structural_subset16",
            paths["gap_structural_subset"],
            gap_structural_subset_path,
            "16 deterministic query nodes; not interchangeable with all-node structural-hole semantics",
        ),
    ):
        supplement_records.append(
            {
                "name": name,
                "semantics": semantics,
                "source_directory": str(source),
                "summary_csv": str(csv_path),
                "summary_csv_sha256": sha256(csv_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "state": "authoritative-selection",
        "replacement_policy": str(
            (
                Path(__file__).resolve().parent
                / "final_evidence_selection_policy_20260726.json"
            ).resolve()
        ),
        "main_protocol": {
            "timeout_seconds": 100,
            "eggpu_timing_samples": 5,
            "eggpu_paper_estimator": "minimum",
            "memory_samples": 3,
            "datasets": ["com-Orkut", "GAP-twitter"],
            "functions_per_dataset": 16,
        },
        "selection_notes": [
            "Raw source directories remain immutable.",
            "GAP BC was collected with a 240-second watchdog, but all five measured steady calls completed below the 100-second main limit; its numeric samples therefore satisfy the main timing threshold.",
            "GAP SCC extended-timeout values and structural-hole subset values are supplements and are not inserted into the main 16-function matrix.",
            "The three GAP undirected-projection functions retain the signed-int32 CSR representation-limit outcome because v5 did not change the graph-index ABI.",
        ],
        "selected_artifacts": selected_artifacts,
        "supplements": supplement_records,
    }
    manifest_path = output / "ASSEMBLY_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    audit = {
        "status": "pass",
        "output": str(output),
        "eggpu_main_timing_cells": {
            "com-Orkut": len(
                timing_rows(select_dataset(main_rows, "com-Orkut"))
            ),
            "GAP-twitter": len(gap_timing),
        },
        "gap_main_status_counts": {},
    }
    for row in gap_timing:
        status = row.get("status", "unknown")
        audit["gap_main_status_counts"][status] = (
            audit["gap_main_status_counts"].get(status, 0) + 1
        )
    (output / "ASSEMBLY_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
