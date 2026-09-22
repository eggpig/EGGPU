#!/usr/bin/env python3
"""Fail-closed audit for the V15 Stage 4 intro and workflow outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


INTRO_DATASETS = {
    "R-MAT-S20-EF16": (20, 2**20),
    "R-MAT-S22-EF16": (22, 2**22),
    "R-MAT-S24-EF16": (24, 2**24),
    "R-MAT-S26-EF16": (26, 2**26),
}
WORKFLOW_DATASETS = (
    "ca-HepTh",
    "LastFM",
    "p2p-Gnutella04",
    "ca-HepPh",
    "email-Enron",
    "ca-CondMat",
)
WORKFLOW_BASELINES = ("EGGPU", "EGGPU-isolated")
WORKFLOW_CALLS = (
    (1, "WCC"),
    (2, "PageRank"),
    (3, "BFS"),
    (4, "SSSP"),
    (5, "Closeness"),
)
EXPECTED_SAMPLE_INDICES = {1, 2, 3, 4, 5}
INTRO_TIMER_BOUNDARY = (
    "EGGPU public invocation through complete result return; "
    "benchmark validation excluded"
)
WORKFLOW_TIMER_BOUNDARY = (
    "public operation invocation through its usable result return; "
    "contract-required lazy-result materialization is included; "
    "benchmark validation is excluded"
)


class AuditFailure(ValueError):
    """Raised when a Stage 4 artifact violates the frozen protocol."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"missing required file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path):
    sha256_file(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AuditFailure(f"invalid JSON in {path}: {error}") from error


def read_csv(path: Path) -> list[dict[str, str]]:
    sha256_file(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_json_field(value: str, label: str):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise AuditFailure(f"{label} is not valid JSON") from error


def parse_bool(value, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise AuditFailure(f"{label} is not a serialized boolean: {value!r}")


def positive_float(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise AuditFailure(f"{label} is not numeric: {value!r}") from error
    require(math.isfinite(number) and number > 0.0, f"{label} is not positive")
    return number


def require_close(observed, expected, label: str) -> None:
    observed_float = float(observed)
    expected_float = float(expected)
    require(
        math.isclose(
            observed_float,
            expected_float,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ),
        f"{label} mismatch: {observed_float} != {expected_float}",
    )


def runtime_identity(
    provenance: dict,
    *,
    expected_native: str,
    expected_python: str,
    label: str,
    require_python_snapshot: bool = True,
) -> None:
    require(isinstance(provenance, dict), f"{label}: runtime is not an object")
    require(
        provenance.get("native_sha256") == expected_native,
        f"{label}: native SHA mismatch",
    )
    snapshot = provenance.get("runtime_python_snapshot")
    if snapshot is None and not require_python_snapshot:
        return
    require(isinstance(snapshot, dict), f"{label}: missing Python snapshot")
    require(
        snapshot.get("digest") == expected_python,
        f"{label}: runtime Python SHA mismatch",
    )


def loaded_runtime_fingerprint(provenance: dict, label: str) -> str:
    """Fingerprint the loaded modules recorded in each fresh worker."""

    require(isinstance(provenance, dict), f"{label}: runtime is not an object")
    modules = provenance.get("modules")
    require(isinstance(modules, dict), f"{label}: missing loaded modules")
    normalized_modules = {}
    for name in ("easygraph", "cpp_easygraph"):
        module = modules.get(name)
        require(isinstance(module, dict), f"{label}: missing module {name}")
        module_sha = module.get("sha256")
        require(
            isinstance(module_sha, str) and len(module_sha) == 64,
            f"{label}: invalid module SHA for {name}",
        )
        normalized_modules[name] = {
            "sha256": module_sha,
            "relative_to_runtime": module.get("relative_to_runtime"),
        }
    normalized = {
        "native_sha256": provenance.get("native_sha256"),
        "resolved_root": provenance.get("resolved_root"),
        "modules": normalized_modules,
    }
    return canonical_sha256(normalized)


def artifact_hashes(root: Path, paths: list[Path]) -> dict[str, str]:
    output = {}
    for path in sorted(paths, key=lambda item: str(item.relative_to(root))):
        relative = str(path.relative_to(root))
        require(relative not in output, f"duplicate input path: {relative}")
        output[relative] = sha256_file(path)
    return output


def audit_intro(
    root: Path,
    *,
    expected_native: str,
    expected_python: str,
) -> dict[str, object]:
    root = root.resolve()
    require(root.is_dir(), f"intro result directory is missing: {root}")
    metadata_path = root / "run_metadata.json"
    csv_path = root / "scaling_all.csv"
    json_path = root / "scaling_all.json"
    raw_dir = root / "raw"
    require(raw_dir.is_dir(), f"intro raw directory is missing: {raw_dir}")

    metadata = read_json(metadata_path)
    require(isinstance(metadata, dict), "intro run metadata is not an object")
    require(
        metadata.get("native_binary_sha256") == expected_native,
        "intro run metadata native SHA mismatch",
    )
    require(
        metadata.get("runtime_python_digest") == expected_python,
        "intro run metadata runtime Python SHA mismatch",
    )
    repository_runtime = metadata.get("repository_runtime_provenance")
    runtime_identity(
        repository_runtime,
        expected_native=expected_native,
        expected_python=expected_python,
        label="intro repository runtime",
    )
    expected_loaded_runtime = loaded_runtime_fingerprint(
        repository_runtime, "intro repository runtime"
    )
    require(int(metadata.get("repeat", -1)) == 5, "intro repeat must be five")
    require(int(metadata.get("warmup", -1)) == 2, "intro warmup must be two")
    require(
        int(metadata.get("timing_processes", -1)) == 0,
        "intro must use five fresh timing processes",
    )
    require(
        int(metadata.get("memory_repeat", -1)) == 0,
        "intro Stage 4 output must not contain a memory pass",
    )
    requested = metadata.get("requested_functions")
    require(
        requested == "PageRank" or requested == ["PageRank"],
        "intro must request only PageRank",
    )

    rows = read_csv(csv_path)
    require(len(rows) == 4, f"intro aggregate rows={len(rows)}, expected four")
    aggregate_json = read_json(json_path)
    require(
        isinstance(aggregate_json, list) and len(aggregate_json) == 4,
        "intro scaling_all.json must contain four records",
    )
    by_dataset = {}
    for row in rows:
        dataset = row.get("dataset", "")
        require(dataset in INTRO_DATASETS, f"unexpected intro dataset: {dataset}")
        require(dataset not in by_dataset, f"duplicate intro dataset: {dataset}")
        scale, nodes = INTRO_DATASETS[dataset]
        require(row.get("status") == "ok", f"{dataset}: aggregate is not ok")
        require(row.get("measurement") == "timing", f"{dataset}: not timing")
        require(row.get("function") == "PageRank", f"{dataset}: not PageRank")
        require(int(row.get("num_nodes") or -1) == nodes, f"{dataset}: node count")
        require(parse_bool(row.get("directed"), f"{dataset}: directed"), f"{dataset}: undirected")
        require(
            int(row.get("rmat_scale") or -1) == scale,
            f"{dataset}: R-MAT scale mismatch",
        )
        require(
            int(row.get("rmat_edge_factor") or -1) == 16,
            f"{dataset}: edge factor mismatch",
        )
        require(
            int(row.get("timing_process_samples") or -1) == 5,
            f"{dataset}: aggregate lacks five processes",
        )
        require(
            int(row.get("first_use_calls") or -1) == 1,
            f"{dataset}: first-use call count mismatch",
        )
        require(
            int(row.get("additional_warmup_calls") or -1) == 2,
            f"{dataset}: warmup call count mismatch",
        )
        require(
            int(row.get("preceding_public_calls") or -1) == 3,
            f"{dataset}: preceding call count mismatch",
        )
        require(
            int(row.get("measured_call_position") or -1) == 4,
            f"{dataset}: measured call is not call four",
        )
        require(
            int(row.get("measured_calls_per_process") or -1) == 1,
            f"{dataset}: process must contribute exactly one measured call",
        )
        require(
            row.get("paper_estimator") == "minimum_of_five",
            f"{dataset}: wrong estimator",
        )
        require(
            row.get("result_validation") == "pass",
            f"{dataset}: aggregate validation did not pass",
        )
        require(
            parse_bool(
                row.get("validation_outside_timer"),
                f"{dataset}: validation_outside_timer",
            ),
            f"{dataset}: validation was inside timer",
        )
        require(
            row.get("timer_boundary") == INTRO_TIMER_BOUNDARY,
            f"{dataset}: timer boundary mismatch",
        )
        samples = parse_json_field(
            row.get("steady_e2e_samples", ""),
            f"{dataset}: steady_e2e_samples",
        )
        require(
            isinstance(samples, list) and len(samples) == 5,
            f"{dataset}: aggregate must retain five E2E samples",
        )
        numeric = [
            positive_float(value, f"{dataset}: aggregate sample") for value in samples
        ]
        require_close(
            row.get("steady_e2e_mean"),
            statistics.mean(numeric),
            f"{dataset}: E2E mean",
        )
        require_close(
            row.get("steady_e2e_stdev"),
            statistics.stdev(numeric),
            f"{dataset}: E2E sample standard deviation",
        )
        require_close(
            row.get("steady_e2e_minimum"),
            min(numeric),
            f"{dataset}: E2E minimum",
        )
        require_close(
            row.get("submission_e2e_seconds"),
            min(numeric),
            f"{dataset}: submission estimator",
        )
        by_dataset[dataset] = {"row": row, "samples": numeric}
    require(set(by_dataset) == set(INTRO_DATASETS), "intro dataset set mismatch")
    json_by_dataset = {}
    for record in aggregate_json:
        require(isinstance(record, dict), "intro JSON aggregate record is not an object")
        dataset = record.get("dataset")
        require(dataset in INTRO_DATASETS, "intro JSON has an unexpected dataset")
        require(dataset not in json_by_dataset, f"intro JSON duplicates {dataset}")
        require(record.get("status") == "ok", f"intro JSON {dataset}: status")
        require(
            record.get("measurement") == "timing"
            and record.get("function") == "PageRank",
            f"intro JSON {dataset}: measurement/function",
        )
        require(
            int(record.get("preceding_public_calls", -1)) == 3
            and int(record.get("measured_call_position", -1)) == 4,
            f"intro JSON {dataset}: call protocol",
        )
        require(
            record.get("validation_outside_timer") is True,
            f"intro JSON {dataset}: validation boundary",
        )
        samples = (record.get("steady_e2e") or {}).get("samples")
        require(
            isinstance(samples, list) and len(samples) == 5,
            f"intro JSON {dataset}: sample batch",
        )
        for index, (actual, target) in enumerate(
            zip(samples, by_dataset[dataset]["samples"], strict=True), start=1
        ):
            require_close(actual, target, f"intro JSON {dataset}: sample {index}")
        json_by_dataset[dataset] = record
    require(
        set(json_by_dataset) == set(INTRO_DATASETS),
        "intro JSON dataset set mismatch",
    )

    raw_paths = sorted(
        raw_dir.glob("R-MAT-S*-EF16_PageRank_timing_[1-5].json")
    )
    require(len(raw_paths) == 20, f"intro raw files={len(raw_paths)}, expected 20")
    expected_raw_names = {
        f"{dataset}_PageRank_timing_{index}.json"
        for dataset in INTRO_DATASETS
        for index in EXPECTED_SAMPLE_INDICES
    }
    require(
        {path.name for path in raw_paths} == expected_raw_names,
        "intro raw filename set mismatch",
    )

    raw_samples: dict[str, dict[int, float]] = defaultdict(dict)
    loaded_runtime_fingerprints = set()
    for path in raw_paths:
        payload = read_json(path)
        require(isinstance(payload, dict), f"{path.name}: raw payload is not an object")
        dataset = payload.get("dataset")
        require(dataset in INTRO_DATASETS, f"{path.name}: unexpected dataset")
        index = int(path.stem.rsplit("_", 1)[1])
        embedded_index = payload.get("timing_process_index")
        if embedded_index is not None:
            require(
                int(embedded_index) == index,
                f"{path.name}: process index mismatch",
            )
        require(payload.get("status") == "ok", f"{path.name}: status is not ok")
        require(
            payload.get("measurement") == "timing",
            f"{path.name}: measurement mismatch",
        )
        require(
            payload.get("function") == "PageRank",
            f"{path.name}: function mismatch",
        )
        require(
            int(payload.get("preceding_public_calls", -1)) == 3,
            f"{path.name}: preceding call count mismatch",
        )
        require(
            int(payload.get("measured_call_position", -1)) == 4,
            f"{path.name}: measured call mismatch",
        )
        require(
            int(payload.get("measured_calls_per_process", -1)) == 1,
            f"{path.name}: measured call count mismatch",
        )
        require(
            payload.get("timer_boundary") == INTRO_TIMER_BOUNDARY,
            f"{path.name}: timer boundary mismatch",
        )
        require(
            payload.get("validation_outside_timer") is True,
            f"{path.name}: validation was not outside timer",
        )
        validation = payload.get("result_validation")
        require(
            isinstance(validation, dict) and validation.get("status") == "pass",
            f"{path.name}: result validation did not pass",
        )
        runtime_identity(
            payload.get("runtime_provenance"),
            expected_native=expected_native,
            expected_python=expected_python,
            label=path.name,
            require_python_snapshot=False,
        )
        loaded_runtime_fingerprints.add(
            loaded_runtime_fingerprint(
                payload.get("runtime_provenance"), path.name
            )
        )
        samples = (payload.get("steady_e2e") or {}).get("samples")
        require(
            isinstance(samples, list) and len(samples) == 1,
            f"{path.name}: raw process must contain one E2E sample",
        )
        raw_samples[dataset][index] = positive_float(
            samples[0], f"{path.name}: E2E sample"
        )

    for dataset in INTRO_DATASETS:
        indexed = raw_samples[dataset]
        require(
            set(indexed) == EXPECTED_SAMPLE_INDICES,
            f"{dataset}: raw process set mismatch",
        )
        observed = [indexed[index] for index in sorted(indexed)]
        expected = by_dataset[dataset]["samples"]
        for index, (actual, target) in enumerate(
            zip(observed, expected, strict=True), start=1
        ):
            require_close(actual, target, f"{dataset}: raw sample {index}")
    require(
        loaded_runtime_fingerprints == {expected_loaded_runtime},
        "intro fresh processes did not load one repository runtime",
    )

    inputs = artifact_hashes(
        root, [metadata_path, csv_path, json_path, *raw_paths]
    )
    return {
        "aggregate_rows": len(rows),
        "datasets": list(INTRO_DATASETS),
        "raw_sample_files": len(raw_paths),
        "samples_per_dataset": 5,
        "measured_call_position": 4,
        "validation_outside_timer": True,
        "input_sha256": inputs,
        "input_manifest_sha256": canonical_sha256(inputs),
    }


def audit_workflow(
    root: Path,
    *,
    expected_native: str,
    expected_python: str,
) -> dict[str, object]:
    root = root.resolve()
    require(root.is_dir(), f"workflow result directory is missing: {root}")
    metadata_path = root / "metadata.json"
    samples_path = root / "cumulative_workflow_samples.csv"
    failures_path = root / "cumulative_workflow_failures.csv"
    unsupported_path = root / "cumulative_workflow_unsupported.csv"
    raw_dir = root / "raw"
    require(raw_dir.is_dir(), f"workflow raw directory is missing: {raw_dir}")

    metadata = read_json(metadata_path)
    require(isinstance(metadata, dict), "workflow metadata is not an object")
    require(
        metadata.get("protocol") == "fresh_process_cumulative_same_graph_workflow_v2",
        "workflow protocol mismatch",
    )
    require(
        tuple(metadata.get("datasets") or ()) == WORKFLOW_DATASETS,
        "workflow dataset order mismatch",
    )
    require(
        tuple(metadata.get("baselines") or ()) == WORKFLOW_BASELINES,
        "workflow baseline order mismatch",
    )
    require(
        tuple(metadata.get("workflow") or ())
        == tuple(function for _, function in WORKFLOW_CALLS),
        "workflow call order mismatch",
    )
    require(int(metadata.get("repeat", -1)) == 5, "workflow repeat must be five")
    require(int(metadata.get("sources", -1)) == 8, "workflow source count mismatch")
    require(int(metadata.get("failures", -1)) == 0, "workflow metadata has failures")
    require(not metadata.get("unsupported_calls"), "workflow has unsupported calls")
    require(
        metadata.get("validation_outside_timer") is True,
        "workflow metadata does not guarantee off-timer validation",
    )
    runtime_identity(
        metadata.get("runtime_provenance"),
        expected_native=expected_native,
        expected_python=expected_python,
        label="workflow metadata",
    )
    expected_loaded_runtime = loaded_runtime_fingerprint(
        metadata.get("runtime_provenance"), "workflow metadata"
    )
    observed_runtimes = metadata.get("observed_eggpu_runtimes")
    require(
        isinstance(observed_runtimes, list) and len(observed_runtimes) == 1,
        "workflow must observe exactly one EGGPU runtime",
    )
    require(
        observed_runtimes[0].get("native_sha256") == expected_native,
        "workflow observed runtime native SHA mismatch",
    )
    require(
        loaded_runtime_fingerprint(
            observed_runtimes[0], "workflow observed runtime"
        )
        == expected_loaded_runtime,
        "workflow observed runtime differs from requested runtime",
    )

    failures = read_csv(failures_path)
    unsupported = read_csv(unsupported_path)
    require(not failures, "workflow failure ledger is not empty")
    require(not unsupported, "workflow unsupported ledger is not empty")

    rows = read_csv(samples_path)
    require(len(rows) == 300, f"workflow rows={len(rows)}, expected 300")
    groups: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    unique_keys = set()
    loaded_runtime_fingerprints = set()
    for row in rows:
        dataset = row.get("dataset", "")
        baseline = row.get("baseline", "")
        sample_index = int(row.get("sample_index") or -1)
        position = int(row.get("call_position") or -1)
        function = row.get("function", "")
        key = (dataset, baseline, sample_index, position, function)
        require(key not in unique_keys, f"duplicate workflow row: {key}")
        unique_keys.add(key)
        require(dataset in WORKFLOW_DATASETS, f"unexpected workflow dataset: {dataset}")
        require(baseline in WORKFLOW_BASELINES, f"unexpected workflow baseline: {baseline}")
        require(sample_index in EXPECTED_SAMPLE_INDICES, f"invalid sample index: {key}")
        require((position, function) in WORKFLOW_CALLS, f"invalid call boundary: {key}")
        require(row.get("status") == "ok", f"{key}: status is not ok")
        require(
            row.get("result_validation") == "pass",
            f"{key}: result validation did not pass",
        )
        require(
            parse_bool(
                row.get("validation_outside_timer"),
                f"{key}: validation_outside_timer",
            ),
            f"{key}: validation was inside timer",
        )
        require(
            row.get("timer_boundary") == WORKFLOW_TIMER_BOUNDARY,
            f"{key}: timer boundary mismatch",
        )
        positive_float(row.get("call_seconds"), f"{key}: call_seconds")
        positive_float(row.get("cumulative_seconds"), f"{key}: cumulative_seconds")
        provenance = parse_json_field(
            row.get("runtime_provenance", ""),
            f"{key}: runtime_provenance",
        )
        runtime_identity(
            provenance,
            expected_native=expected_native,
            expected_python=expected_python,
            label=str(key),
            require_python_snapshot=False,
        )
        loaded_runtime_fingerprints.add(
            loaded_runtime_fingerprint(provenance, str(key))
        )
        groups[(dataset, baseline, sample_index)].append(row)

    require(len(groups) == 60, f"workflow fresh processes={len(groups)}, expected 60")
    expected_groups = {
        (dataset, baseline, sample)
        for dataset in WORKFLOW_DATASETS
        for baseline in WORKFLOW_BASELINES
        for sample in EXPECTED_SAMPLE_INDICES
    }
    require(set(groups) == expected_groups, "workflow process intersection mismatch")
    require(
        loaded_runtime_fingerprints == {expected_loaded_runtime},
        "workflow rows did not load one requested runtime",
    )
    for group_key, group in groups.items():
        ordered = sorted(group, key=lambda row: int(row["call_position"]))
        require(
            [
                (int(row["call_position"]), row["function"]) for row in ordered
            ]
            == list(WORKFLOW_CALLS),
            f"{group_key}: call order mismatch",
        )
        cumulative = 0.0
        for row in ordered:
            cumulative += float(row["call_seconds"])
            require_close(
                row["cumulative_seconds"],
                cumulative,
                f"{group_key}/{row['function']}: cumulative time",
            )

    raw_paths = sorted(raw_dir.glob("*.json"))
    require(len(raw_paths) == 60, f"workflow raw files={len(raw_paths)}, expected 60")
    expected_raw_names = {
        f"{dataset}_{baseline}_{sample}.json"
        for dataset, baseline, sample in expected_groups
    }
    require(
        {path.name for path in raw_paths} == expected_raw_names,
        "workflow raw filename set mismatch",
    )
    for path in raw_paths:
        payload = read_json(path)
        require(isinstance(payload, dict), f"{path.name}: raw payload is not an object")
        runtime_identity(
            payload.get("runtime_provenance"),
            expected_native=expected_native,
            expected_python=expected_python,
            label=path.name,
            require_python_snapshot=False,
        )
        require(
            loaded_runtime_fingerprint(
                payload.get("runtime_provenance"), path.name
            )
            == expected_loaded_runtime,
            f"{path.name}: raw runtime differs from requested runtime",
        )
        raw_rows = payload.get("rows")
        require(
            isinstance(raw_rows, list) and len(raw_rows) == 5,
            f"{path.name}: raw payload must contain five calls",
        )
        observed_calls = [
            (int(row.get("call_position", -1)), row.get("function"))
            for row in raw_rows
        ]
        require(observed_calls == list(WORKFLOW_CALLS), f"{path.name}: call order")
        for row in raw_rows:
            require(row.get("status") == "ok", f"{path.name}: raw status")
            require(
                row.get("result_validation") == "pass",
                f"{path.name}: raw validation",
            )
            require(
                row.get("validation_outside_timer") is True,
                f"{path.name}: raw validation boundary",
            )
            require(
                row.get("timer_boundary") == WORKFLOW_TIMER_BOUNDARY,
                f"{path.name}: raw timer boundary",
            )
            runtime_identity(
                row.get("runtime_provenance"),
                expected_native=expected_native,
                expected_python=expected_python,
                label=f"{path.name}/{row.get('function')}",
                require_python_snapshot=False,
            )
            require(
                loaded_runtime_fingerprint(
                    row.get("runtime_provenance"),
                    f"{path.name}/{row.get('function')}",
                )
                == expected_loaded_runtime,
                f"{path.name}: row runtime differs from requested runtime",
            )

    inputs = artifact_hashes(
        root,
        [
            metadata_path,
            samples_path,
            failures_path,
            unsupported_path,
            *raw_paths,
        ],
    )
    return {
        "rows": len(rows),
        "datasets": list(WORKFLOW_DATASETS),
        "baselines": list(WORKFLOW_BASELINES),
        "workflow": [function for _, function in WORKFLOW_CALLS],
        "fresh_processes": len(groups),
        "samples_per_dataset_baseline": 5,
        "raw_process_files": len(raw_paths),
        "validation_outside_timer": True,
        "input_sha256": inputs,
        "input_manifest_sha256": canonical_sha256(inputs),
    }


def audit_stage4(
    *,
    intro_result_dir: Path,
    workflow_result_dir: Path,
    expected_native_sha256: str,
    expected_runtime_python_sha256: str,
) -> dict[str, object]:
    for value, label in (
        (expected_native_sha256, "expected native SHA"),
        (expected_runtime_python_sha256, "expected runtime Python SHA"),
    ):
        require(
            len(value) == 64 and all(character in "0123456789abcdef" for character in value),
            f"{label} must be a lowercase SHA-256 digest",
        )
    intro = audit_intro(
        intro_result_dir,
        expected_native=expected_native_sha256,
        expected_python=expected_runtime_python_sha256,
    )
    workflow = audit_workflow(
        workflow_result_dir,
        expected_native=expected_native_sha256,
        expected_python=expected_runtime_python_sha256,
    )
    combined_inputs = {
        "intro": intro["input_manifest_sha256"],
        "workflow": workflow["input_manifest_sha256"],
    }
    return {
        "schema_version": "eggpu_v15_stage4_audit_v1",
        "status": "pass",
        "expected_runtime": {
            "native_sha256": expected_native_sha256,
            "runtime_python_sha256": expected_runtime_python_sha256,
        },
        "intro": intro,
        "workflow": workflow,
        "combined_input_manifest_sha256": canonical_sha256(combined_inputs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intro-result-dir", required=True, type=Path)
    parser.add_argument("--workflow-result-dir", required=True, type=Path)
    parser.add_argument("--expected-native-sha256", required=True)
    parser.add_argument("--expected-runtime-python-sha256", required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = audit_stage4(
            intro_result_dir=args.intro_result_dir,
            workflow_result_dir=args.workflow_result_dir,
            expected_native_sha256=args.expected_native_sha256,
            expected_runtime_python_sha256=(
                args.expected_runtime_python_sha256
            ),
        )
    except (AuditFailure, OSError, ValueError) as error:
        print(f"V15 Stage 4 audit failed: {error}", file=sys.stderr)
        return 1
    serialized = json.dumps(
        result, indent=2, ensure_ascii=False, sort_keys=True
    ) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
