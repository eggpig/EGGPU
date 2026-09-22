#!/usr/bin/env python3
"""Assemble and audit the V15 PageRank protocol correction.

Eleven regular-matrix PageRank measurements were produced with a conservative
generic keyword adapter inside the public-call E2E timer.  Orkut and
GAP-twitter already use the direct scaling runner and remain authoritative.
This tool:

* unconditionally replaces the complete build/E2E/kernel triplet for all
  eleven affected cells with a protocol-valid five-sample rerun;
* retains the two direct-call anchors;
* emits deterministic combined regular-matrix input directories; and
* proves, by normalized per-cell hashes, that all 195 non-PageRank cells are
  byte-semantically unchanged.

No old/new speed comparison participates in adoption.  Timing values are read
only after the action set has been fixed, and only to validate the measurement
contract (finite, positive, and 0 < kernel < E2E).
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


EXPECTED_NATIVE_SHA256 = (
    "d2a93a3da0ffd0c554c9dab52c9d05d8c1f5f5b4904bcfd19fd95b695bb06eaf"
)
EXPECTED_RUNTIME_SHA256 = (
    "1be28aeb48374c65642b26f7233a7764f66fb3fb8003e6434e04744e08b643fb"
)
PROTOCOL_NAME = "eggpu_pagerank_direct_public_call_fail_closed_v1"
ADOPTION_POLICY = (
    "protocol_validity_only_unconditional_no_relative_timing_selection"
)
FUNCTIONS = (
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
)
METRICS = ("build", "e2e", "kernel")
REGULAR_REPLACEMENTS = (
    "ca-HepTh",
    "LastFM",
    "web-NotreDame",
    "p2p-Gnutella04",
    "ER-100k",
    "ca-HepPh",
    "soc-Slashdot0811",
    "com-youtube",
    "ca-CondMat",
    "soc-Epinions1",
    "email-Enron",
)
DIRECT_ANCHORS = ("com-Orkut", "GAP-twitter")
MAIN_CORE_FILES = (
    "results_samples.csv",
    "results_long.csv",
    "results_build.csv",
    "results_e2e.csv",
    "results_kernel.csv",
    "correctness_validation.csv",
    "run_metadata.json",
)
OPTIONAL_COPY_FILES = (
    "measurement_schema.json",
    "baseline_versions.json",
    "dataset_stats.json",
    "results_memory.csv",
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
SUM_RE = re.compile(
    r"(?:^|[;,\s])sum=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
CORRECTNESS_FIELD_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,;]+)"
)


class GateError(ValueError):
    """Fail-closed protocol or evidence error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_value(payload: object) -> object:
    """Represent non-finite correctness sentinels without lossy JSON output."""

    if isinstance(payload, float) and not math.isfinite(payload):
        if math.isnan(payload):
            label = "nan"
        elif payload > 0:
            label = "positive_infinity"
        else:
            label = "negative_infinity"
        return {"__nonfinite_float__": label}
    if isinstance(payload, dict):
        return {
            str(key): canonical_json_value(value)
            for key, value in payload.items()
        }
    if isinstance(payload, (list, tuple)):
        return [canonical_json_value(value) for value in payload]
    return payload


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        canonical_json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha(payload: object) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON evidence: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{path}: expected a JSON object")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.is_file(), f"missing CSV evidence: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        require(bool(fields), f"{path}: missing CSV header")
        return fields, [dict(row) for row in reader]


def correctness_fields(value: object) -> dict[str, str]:
    return {
        key: raw.strip()
        for key, raw in CORRECTNESS_FIELD_RE.findall(str(value or ""))
    }


def pagerank_detail_paths(result_dir: Path) -> list[Path]:
    """Resolve the validation vectors referenced by correction timing rows."""

    _fields, rows = read_csv(result_dir / "results_samples.csv")
    paths: set[Path] = set()
    for row in rows:
        if (
            row.get("baseline") != "EGGPU"
            or row.get("function") != "PageRank"
            or row.get("metric") != "e2e"
            or row.get("status") != "ok"
        ):
            continue
        parsed = correctness_fields(row.get("correctness"))
        detail = parsed.get("detail", "")
        require(bool(detail), f"{result_dir}: PageRank detail path is absent")
        path = Path(detail).resolve(strict=True)
        require(
            path.is_relative_to(result_dir),
            f"{result_dir}: PageRank detail escapes the result directory: {path}",
        )
        paths.add(path)
    require(bool(paths), f"{result_dir}: no PageRank validation detail found")
    return sorted(paths, key=lambda path: str(path))


def write_csv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finite_positive(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"{label}: not numeric: {value!r}") from exc
    require(math.isfinite(number) and number > 0, f"{label}: must be > 0")
    return number


def integer(value: object, label: str) -> int:
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise GateError(f"{label}: not an integer: {value!r}") from exc
    return result


def nested(payload: Mapping[str, object], *keys: str) -> object | None:
    current: object = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def normalized_sha(value: object, label: str) -> str:
    text = str(value or "").strip().lower()
    require(bool(SHA_RE.fullmatch(text)), f"{label}: invalid SHA-256 {value!r}")
    return text


def identity_from_metadata(path: Path) -> tuple[str, str, bool]:
    payload = read_json(path)
    candidate_values = [
        nested(payload, "candidate_sha256"),
        nested(payload, "native_binary_sha256"),
        nested(payload, "repository_runtime_provenance", "native_sha256"),
        nested(
            payload,
            "repository_runtime_provenance",
            "modules",
            "cpp_easygraph",
            "sha256",
        ),
        nested(payload, "build_artifacts", "active_cpp_easygraph", "sha256"),
        nested(payload, "baseline_versions", "cpp_easygraph", "sha256"),
    ]
    candidates = {
        normalized_sha(value, f"{path}: candidate")
        for value in candidate_values
        if value
    }
    require(len(candidates) == 1, f"{path}: conflicting/missing candidate SHA")

    snapshots = [
        nested(payload, "runtime_python_snapshot"),
        nested(
            payload,
            "repository_runtime_provenance",
            "runtime_python_snapshot",
        ),
    ]
    runtime_values: set[str] = set()
    symlink_values: set[bool] = set()
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        if snapshot.get("digest"):
            runtime_values.add(
                normalized_sha(snapshot["digest"], f"{path}: runtime digest")
            )
        if "package_is_symlink" in snapshot:
            symlink_values.add(bool(snapshot["package_is_symlink"]))
    direct_runtime = nested(payload, "runtime_python_digest")
    if direct_runtime:
        runtime_values.add(
            normalized_sha(direct_runtime, f"{path}: runtime digest")
        )
    require(
        len(runtime_values) == 1,
        f"{path}: conflicting/missing runtime Python digest",
    )
    require(
        symlink_values == {False},
        f"{path}: runtime package must be explicitly non-symlink",
    )
    return next(iter(candidates)), next(iter(runtime_values)), False


def require_identity(
    result_dir: Path, expected_native: str, expected_runtime: str
) -> dict:
    metadata_path = result_dir / "run_metadata.json"
    candidate, runtime, package_is_symlink = identity_from_metadata(
        metadata_path
    )
    require(
        candidate == expected_native,
        f"{result_dir}: native SHA differs: {candidate}",
    )
    require(
        runtime == expected_runtime,
        f"{result_dir}: runtime Python SHA differs: {runtime}",
    )
    require(
        package_is_symlink is False,
        f"{result_dir}: runtime package is a symlink",
    )
    return read_json(metadata_path)


def directory_evidence_digest(result_dir: Path, kind: str) -> dict[str, object]:
    result_dir = result_dir.resolve(strict=True)
    if kind in {"original", "correction", "supplemental", "combined"}:
        names = list(MAIN_CORE_FILES)
        if kind == "correction":
            names.append("pagerank_protocol_attestation.json")
        entries = []
        for name in sorted(names):
            path = result_dir / name
            require(path.is_file(), f"{result_dir}: missing {name}")
            entries.append(
                {"path": name, "sha256": sha256(path), "bytes": path.stat().st_size}
            )
        if kind == "correction":
            for path in pagerank_detail_paths(result_dir):
                entries.append(
                    {
                        "path": str(path.relative_to(result_dir)),
                        "sha256": sha256(path),
                        "bytes": path.stat().st_size,
                    }
                )
    elif kind == "anchor":
        paths = [result_dir / "scaling_all.csv", result_dir / "run_metadata.json"]
        paths.extend(sorted((result_dir / "raw").glob("*_timing.json")))
        require(len(paths) > 2, f"{result_dir}: anchor timing JSON is absent")
        entries = []
        for path in paths:
            require(path.is_file(), f"{result_dir}: missing {path.name}")
            entries.append(
                {
                    "path": str(path.relative_to(result_dir)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    else:
        raise GateError(f"unknown evidence kind: {kind}")
    return {
        "path": str(result_dir),
        "kind": kind,
        "digest": canonical_sha(entries),
        "files": entries,
    }


def validate_attestation(result_dir: Path) -> dict:
    path = result_dir / "pagerank_protocol_attestation.json"
    payload = read_json(path)
    require(payload.get("status") == "pass", f"{path}: status is not pass")
    require(
        payload.get("protocol") == PROTOCOL_NAME,
        f"{path}: protocol differs",
    )
    require(
        bool(SHA_RE.fullmatch(str(payload.get("source_sha256", "")).lower())),
        f"{path}: source SHA-256 is absent or invalid",
    )
    timed = payload.get("timed_public_call")
    validation = payload.get("result_validation")
    kernel = payload.get("kernel_timing")
    signature = payload.get("signature_preflight")
    require(isinstance(timed, Mapping), f"{path}: timed call evidence missing")
    require(
        timed.get("status") == "pass"
        and timed.get("call") == "eg.pagerank"
        and timed.get("graph_argument") == "g"
        and timed.get("keyword_bindings")
        == {
            "alpha": "pr_alpha",
            "max_iter": "pr_max_iter",
            "tol": "pr_tol",
        }
        and timed.get("weight") is None
        and timed.get("return_only_callable") is True
        and timed.get("generic_adapter_absent") is True,
        f"{path}: direct-call evidence differs",
    )
    require(
        isinstance(signature, Mapping)
        and signature.get("status") == "pass"
        and signature.get("call") == "inspect.signature(eg.pagerank)"
        and signature.get("outside_timer") is True,
        f"{path}: signature preflight is not outside timer",
    )
    require(
        isinstance(validation, Mapping)
        and validation.get("status") == "pass"
        and validation.get("call") == "validate_pagerank_result"
        and validation.get("outside_timer") is True,
        f"{path}: validation is not outside timer",
    )
    require(
        set(validation.get("checks", []))
        == {"shape", "finite", "nonnegative", "unit_sum"},
        f"{path}: PageRank validation checks differ",
    )
    require(
        isinstance(kernel, Mapping)
        and kernel.get("status") == "pass"
        and kernel.get("call") == "require_kernel_time"
        and kernel.get("missing_value_policy") == "fail_closed"
        and kernel.get("algorithm_time_fallback_absent") is True,
        f"{path}: kernel timing is not fail-closed",
    )
    return payload


def sample_seconds(row: Mapping[str, str], label: str) -> float:
    seconds = finite_positive(row.get("seconds"), f"{label}.seconds")
    value = finite_positive(row.get("value"), f"{label}.value")
    require(
        math.isclose(seconds, value, rel_tol=1.0e-12, abs_tol=1.0e-15),
        f"{label}: seconds/value differ: {seconds} vs {value}",
    )
    require(row.get("unit") == "s", f"{label}: unit must be seconds")
    require(
        row.get("measurement_phase") == "timing",
        f"{label}: measurement_phase must be timing",
    )
    return seconds


def validate_page_rank_sum(rows: Sequence[Mapping[str, str]], label: str) -> None:
    for row in rows:
        correctness = str(row.get("correctness", ""))
        match = SUM_RE.search(correctness)
        require(bool(match), f"{label}: PageRank sum is absent")
        total = float(match.group(1))
        require(
            math.isfinite(total) and abs(total - 1.0) <= 1.0e-6,
            f"{label}: PageRank sum {total} differs from 1",
        )


def array_digest(values: object) -> str:
    import numpy as np

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def validate_page_rank_detail(
    rows: Sequence[Mapping[str, str]], result_dir: Path, label: str
) -> Path:
    import numpy as np

    records: set[tuple[str, str, str]] = set()
    for row in rows:
        parsed = correctness_fields(row.get("correctness"))
        detail = parsed.get("detail", "")
        detail_kind = parsed.get("detail_kind", "")
        detail_sha = parsed.get("detail_sha", "").lower()
        require(bool(detail), f"{label}: PageRank detail path is absent")
        require(detail_kind == "vector", f"{label}: detail_kind must be vector")
        require(
            bool(re.fullmatch(r"[0-9a-f]{16}", detail_sha)),
            f"{label}: detail_sha is absent or invalid",
        )
        detail_path = Path(detail).resolve(strict=True)
        require(
            detail_path.is_relative_to(result_dir),
            f"{label}: detail path escapes result directory: {detail_path}",
        )
        records.add((str(detail_path), detail_kind, detail_sha))
    require(
        len(records) == 1,
        f"{label}: the five PageRank samples do not share one validation vector",
    )
    detail_text, _kind, expected_digest = next(iter(records))
    detail_path = Path(detail_text)
    with np.load(detail_path, allow_pickle=False) as payload:
        require("values" in payload, f"{label}: detail vector has no values")
        kind = (
            str(payload["kind"].tolist())
            if "kind" in payload
            else ""
        )
        values = np.asarray(payload["values"])
    require(kind == "vector", f"{label}: NPZ kind is {kind!r}, not vector")
    require(
        values.ndim == 1 and values.size > 0,
        f"{label}: PageRank detail must be a non-empty 1D vector",
    )
    numeric = np.asarray(values, dtype=np.float64)
    require(np.isfinite(numeric).all(), f"{label}: PageRank detail is non-finite")
    require(
        float(numeric.min(initial=0.0)) >= -1.0e-12,
        f"{label}: PageRank detail contains a negative value",
    )
    total = float(numeric.sum(dtype=np.float64))
    require(
        math.isclose(total, 1.0, rel_tol=1.0e-6, abs_tol=1.0e-8),
        f"{label}: PageRank detail sum {total} differs from 1",
    )
    require(
        array_digest(values) == expected_digest,
        f"{label}: PageRank detail digest differs",
    )
    return detail_path


def validate_regular_correction(
    result_dir: Path, expected_native: str, expected_runtime: str
) -> set[str]:
    result_dir = result_dir.resolve(strict=True)
    metadata = require_identity(result_dir, expected_native, expected_runtime)
    args = metadata.get("benchmark_args")
    require(isinstance(args, Mapping), f"{result_dir}: benchmark_args missing")
    require(args.get("baselines") == ["EGGPU"], f"{result_dir}: baselines differ")
    require(args.get("functions") == ["PageRank"], f"{result_dir}: functions differ")
    for field, expected in (
        ("repeat", 5),
        ("warmup", 2),
        ("easygraph_warmup", 2),
        ("pr_max_iter", 200),
    ):
        require(
            integer(args.get(field), f"{result_dir}: {field}") == expected,
            f"{result_dir}: {field} differs",
        )
    for field, expected in (
        ("pr_alpha", 0.75),
        ("pr_eps", 1.0e-6),
    ):
        observed = float(args.get(field))
        require(
            math.isclose(observed, expected, rel_tol=0, abs_tol=1.0e-15),
            f"{result_dir}: {field} differs",
        )
    require(
        args.get("eggpu_execution_protocol") == "steady-state"
        and args.get("measurement_mode") == "timing",
        f"{result_dir}: timing protocol differs",
    )
    validate_attestation(result_dir)

    fields, rows = read_csv(result_dir / "results_samples.csv")
    required_fields = {
        "dataset",
        "function",
        "baseline",
        "metric",
        "seconds",
        "value",
        "unit",
        "status",
        "sample_index",
        "sample_count",
        "measurement_phase",
        "measurement_scope",
        "timer_kind",
        "measurement_window",
        "correctness",
    }
    require(
        required_fields.issubset(fields),
        f"{result_dir}: results_samples schema is incomplete",
    )
    selected = [row for row in rows if row.get("baseline") == "EGGPU"]
    require(selected, f"{result_dir}: no EGGPU samples")
    require(
        {row.get("function") for row in selected} == {"PageRank"},
        f"{result_dir}: correction contains a non-PageRank function",
    )
    datasets = {str(row.get("dataset")) for row in selected}
    require(
        datasets <= set(REGULAR_REPLACEMENTS),
        f"{result_dir}: unexpected dataset(s): {sorted(datasets)}",
    )
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[(str(row["dataset"]), str(row["metric"]))].append(row)

    for dataset in sorted(datasets):
        metric_rows: dict[str, dict[int, dict[str, str]]] = {}
        for metric in METRICS:
            group = grouped.get((dataset, metric), [])
            require(
                len(group) == 5,
                f"{result_dir}/{dataset}/{metric}: expected 5 samples",
            )
            require(
                all(row.get("status") == "ok" for row in group),
                f"{result_dir}/{dataset}/{metric}: a sample is not ok",
            )
            indices = [
                integer(row.get("sample_index"), "sample_index")
                for row in group
            ]
            require(
                sorted(indices) == [1, 2, 3, 4, 5],
                f"{result_dir}/{dataset}/{metric}: sample indices differ",
            )
            require(
                all(
                    integer(row.get("sample_count"), "sample_count") == 5
                    for row in group
                ),
                f"{result_dir}/{dataset}/{metric}: sample_count differs",
            )
            metric_rows[metric] = dict(zip(indices, group))

        require(
            sum(len(rows_by_index) for rows_by_index in metric_rows.values())
            == 15,
            f"{result_dir}/{dataset}: expected 15 timing rows",
        )
        expected_contract = {
            "build": (
                "baseline_graph_construction",
                "perf_counter_wall",
                "graph_construction",
            ),
            "e2e": (
                "user_visible_function_call",
                "perf_counter_wall",
                "python_function_call",
            ),
            "kernel": (
                "algorithm_compute",
                "cuda_event",
                "device_execution",
            ),
        }
        for metric, (scope, timer, window) in expected_contract.items():
            for index, row in metric_rows[metric].items():
                require(
                    (
                        row.get("measurement_scope"),
                        row.get("timer_kind"),
                        row.get("measurement_window"),
                    )
                    == (scope, timer, window),
                    f"{result_dir}/{dataset}/{metric}/{index}: timer contract differs",
                )
        for index in range(1, 6):
            build = sample_seconds(
                metric_rows["build"][index], f"{dataset}/build/{index}"
            )
            e2e = sample_seconds(
                metric_rows["e2e"][index], f"{dataset}/e2e/{index}"
            )
            kernel = sample_seconds(
                metric_rows["kernel"][index], f"{dataset}/kernel/{index}"
            )
            require(build > 0, f"{dataset}/build/{index}: must be positive")
            require(
                0 < kernel < e2e,
                f"{dataset}/sample {index}: require 0 < kernel < e2e",
            )
        validate_page_rank_sum(
            [metric_rows["e2e"][index] for index in range(1, 6)],
            f"{result_dir}/{dataset}",
        )
        validate_page_rank_detail(
            [metric_rows["e2e"][index] for index in range(1, 6)],
            result_dir,
            f"{result_dir}/{dataset}",
        )

    _long_fields, aggregates = read_csv(result_dir / "results_long.csv")
    aggregates = [
        row
        for row in aggregates
        if row.get("baseline") == "EGGPU"
        and row.get("function") == "PageRank"
    ]
    for dataset in sorted(datasets):
        cell = [row for row in aggregates if row.get("dataset") == dataset]
        require(
            len(cell) == 3 and {row.get("metric") for row in cell} == set(METRICS),
            f"{result_dir}/{dataset}: expected three aggregate timing rows",
        )
        require(
            all(row.get("status") == "ok" for row in cell),
            f"{result_dir}/{dataset}: aggregate status differs",
        )
        for row in cell:
            for count_field in ("sample_count", "n_total", "n_valid"):
                if row.get(count_field) not in {"", None}:
                    require(
                        integer(row[count_field], count_field) == 5,
                        f"{result_dir}/{dataset}: {count_field} differs",
                    )

    _validation_fields, validation = read_csv(
        result_dir / "correctness_validation.csv"
    )
    for dataset in sorted(datasets):
        cell = [
            row
            for row in validation
            if row.get("dataset") == dataset
            and row.get("function") == "PageRank"
            and row.get("baseline") == "EGGPU"
        ]
        require(
            len(cell) == 1,
            f"{result_dir}/{dataset}: expected one validation row",
        )
        status = str(cell[0].get("validation_status", "")).strip().lower()
        require(
            status == "pass",
            f"{result_dir}/{dataset}: validation status must be 'pass', got {status!r}",
        )
    return datasets


def main_sample_cells(result_dir: Path) -> set[tuple[str, str]]:
    _fields, rows = read_csv(result_dir / "results_samples.csv")
    return {
        (str(row.get("dataset")), str(row.get("function")))
        for row in rows
        if row.get("baseline") == "EGGPU"
        and row.get("metric") in METRICS
        and row.get("dataset")
        and row.get("function")
    }


def regular_cell_payloads(
    result_dirs: Sequence[Path],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    output: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for result_dir in result_dirs:
        _sample_fields, samples = read_csv(result_dir / "results_samples.csv")
        _long_fields, aggregates = read_csv(result_dir / "results_long.csv")
        _validation_fields, validation = read_csv(
            result_dir / "correctness_validation.csv"
        )
        cells = main_sample_cells(result_dir)
        for dataset, function in sorted(cells):
            if function == "PageRank":
                continue
            sample_rows = [
                row
                for row in samples
                if row.get("baseline") == "EGGPU"
                and row.get("dataset") == dataset
                and row.get("function") == function
                and row.get("metric") in METRICS
            ]
            aggregate_rows = [
                row
                for row in aggregates
                if row.get("baseline") == "EGGPU"
                and row.get("dataset") == dataset
                and row.get("function") == function
                and row.get("metric") in METRICS
            ]
            validation_rows = [
                row
                for row in validation
                if row.get("baseline") == "EGGPU"
                and row.get("dataset") == dataset
                and row.get("function") == function
            ]
            contribution = {
                "samples": sorted(
                    sample_rows,
                    key=lambda row: (
                        str(row.get("metric")),
                        str(row.get("sample_index")),
                        canonical_sha(row),
                    ),
                ),
                "aggregates": sorted(
                    aggregate_rows,
                    key=lambda row: (
                        str(row.get("metric")),
                        canonical_sha(row),
                    ),
                ),
                "validation": sorted(
                    validation_rows, key=canonical_sha
                ),
            }
            output[(dataset, function)].append(contribution)
    for contributions in output.values():
        contributions.sort(key=canonical_sha)
    return dict(output)


def anchor_timing_payloads(
    result_dirs: Sequence[Path],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    output: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for result_dir in result_dirs:
        _fields, rows = read_csv(result_dir / "scaling_all.csv")
        for row in rows:
            if row.get("measurement") != "timing":
                continue
            dataset = str(row.get("dataset", ""))
            function = str(row.get("function", ""))
            if not dataset or not function:
                continue
            raw_path = result_dir / "raw" / f"{dataset}_{function}_timing.json"
            raw = read_json(raw_path)
            output[(dataset, function)].append(
                {"summary": row, "raw_timing": raw}
            )
    for contributions in output.values():
        contributions.sort(key=canonical_sha)
    return dict(output)


def validate_anchor_pagerank(
    payloads: Mapping[tuple[str, str], list[dict[str, object]]]
) -> None:
    observed = {
        dataset
        for (dataset, function) in payloads
        if function == "PageRank"
    }
    require(
        observed == set(DIRECT_ANCHORS),
        f"anchor PageRank datasets differ: {sorted(observed)}",
    )
    for dataset in DIRECT_ANCHORS:
        contributions = payloads[(dataset, "PageRank")]
        require(
            len(contributions) == 1,
            f"{dataset}/PageRank: expected one anchor contribution",
        )
        raw = contributions[0]["raw_timing"]
        require(
            isinstance(raw, Mapping) and raw.get("status") == "ok",
            f"{dataset}/PageRank: raw status differs",
        )
        require(
            raw.get("result_validation") == {"status": "pass", "failures": []}
            or (
                isinstance(raw.get("result_validation"), Mapping)
                and raw["result_validation"].get("status") == "pass"
                and not raw["result_validation"].get("failures")
            ),
            f"{dataset}/PageRank: validation failed",
        )
        require(
            raw.get("validation_outside_timer") is True,
            f"{dataset}/PageRank: validation is not outside timer",
        )
        boundary = str(raw.get("timer_boundary", "")).lower()
        require(
            "public invocation" in boundary
            and ("validation excluded" in boundary or "validation" in boundary),
            f"{dataset}/PageRank: timer boundary differs",
        )
        require(
            integer(
                raw.get("timing_process_samples"),
                f"{dataset}: timing_process_samples",
            )
            == 5,
            f"{dataset}/PageRank: timing_process_samples differs",
        )
        load = nested(raw, "load", "samples")
        e2e = nested(raw, "steady_e2e", "samples")
        kernel = nested(raw, "steady_kernel", "samples")
        require(
            isinstance(load, list)
            and isinstance(e2e, list)
            and isinstance(kernel, list),
            f"{dataset}/PageRank: timing sample arrays are absent",
        )
        require(
            len(load) == len(e2e) == len(kernel) == 5,
            f"{dataset}/PageRank: expected 15 timing values",
        )
        for index, (build_value, e2e_value, kernel_value) in enumerate(
            zip(load, e2e, kernel), start=1
        ):
            finite_positive(build_value, f"{dataset}/build/{index}")
            e2e_seconds = finite_positive(e2e_value, f"{dataset}/e2e/{index}")
            kernel_seconds = finite_positive(
                kernel_value, f"{dataset}/kernel/{index}"
            )
            require(
                kernel_seconds < e2e_seconds,
                f"{dataset}/sample {index}: require 0 < kernel < e2e",
            )


def cell_hashes(
    regular: Mapping[tuple[str, str], list[dict[str, object]]],
    anchors: Mapping[tuple[str, str], list[dict[str, object]]],
) -> dict[tuple[str, str], str]:
    merged: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for source in (regular, anchors):
        for key, contributions in source.items():
            if key[1] == "PageRank":
                continue
            merged[key].extend(contributions)
    return {
        key: canonical_sha(sorted(contributions, key=canonical_sha))
        for key, contributions in sorted(merged.items())
    }


def row_identity(row: Mapping[str, str], kind: str) -> tuple[str, ...]:
    base = (
        str(row.get("dataset", "")),
        str(row.get("function", "")),
        str(row.get("baseline", "")),
    )
    if kind == "samples":
        return base + (
            str(row.get("metric", "")),
            str(row.get("sample_index", "")),
        )
    if kind in {"long", "metric"}:
        return base + (str(row.get("metric", "")),)
    if kind == "validation":
        return base
    raise GateError(f"unknown row identity kind: {kind}")


def correction_rows_for_datasets(
    correction_by_dataset: Mapping[str, Path],
    datasets: set[str],
    filename: str,
    kind: str,
) -> tuple[list[str], dict[tuple[str, ...], dict[str, str]]]:
    fields: list[str] | None = None
    output: dict[tuple[str, ...], dict[str, str]] = {}
    for dataset in sorted(datasets):
        path = correction_by_dataset[dataset] / filename
        current_fields, rows = read_csv(path)
        if fields is None:
            fields = current_fields
        require(
            fields == current_fields,
            f"{path}: correction CSV schema differs",
        )
        selected = [
            row
            for row in rows
            if row.get("dataset") == dataset
            and row.get("function") == "PageRank"
            and row.get("baseline") == "EGGPU"
        ]
        for row in selected:
            key = row_identity(row, kind)
            require(key not in output, f"{path}: duplicate correction row {key}")
            output[key] = row
    return list(fields or []), output


def overlay_csv_file(
    original_dir: Path,
    output_dir: Path,
    correction_by_dataset: Mapping[str, Path],
    datasets: set[str],
    filename: str,
    kind: str,
) -> None:
    original_fields, original_rows = read_csv(original_dir / filename)
    correction_fields, correction_rows = correction_rows_for_datasets(
        correction_by_dataset, datasets, filename, kind
    )
    require(
        original_fields == correction_fields,
        f"{filename}: original/correction schema differs",
    )
    used: set[tuple[str, ...]] = set()
    output_rows: list[dict[str, str]] = []
    for row in original_rows:
        is_target = (
            row.get("dataset") in datasets
            and row.get("function") == "PageRank"
            and row.get("baseline") == "EGGPU"
        )
        if not is_target:
            output_rows.append(row)
            continue
        key = row_identity(row, kind)
        require(
            key in correction_rows,
            f"{filename}: missing correction row {key}",
        )
        output_rows.append(correction_rows[key])
        used.add(key)
    require(
        used == set(correction_rows),
        f"{filename}: correction rows do not match original slots",
    )
    write_csv(output_dir / filename, original_fields, output_rows)


def render_validation_markdown(csv_path: Path, output_path: Path) -> None:
    fields, rows = read_csv(csv_path)
    preferred = [
        field
        for field in (
            "dataset",
            "function",
            "baseline",
            "reference",
            "validation_status",
            "details",
        )
        if field in fields
    ]

    def clean(value: object) -> str:
        return str(value or "").replace("|", "\\|").replace("\n", " ")

    lines = [
        "# Correctness validation",
        "",
        "| " + " | ".join(preferred) + " |",
        "|" + "|".join("---" for _ in preferred) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(clean(row.get(field)) for field in preferred) + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assemble_one_main_dir(
    original_dir: Path,
    physical_output_dir: Path,
    logical_output_dir: str,
    correction_by_dataset: Mapping[str, Path],
) -> set[str]:
    cells = main_sample_cells(original_dir)
    datasets = {
        dataset
        for dataset, function in cells
        if function == "PageRank"
    }
    require(datasets, f"{original_dir}: contains no PageRank cell")
    require(
        datasets <= set(REGULAR_REPLACEMENTS),
        f"{original_dir}: unexpected PageRank datasets {sorted(datasets)}",
    )
    physical_output_dir.mkdir(parents=True, exist_ok=False)
    for filename, kind in (
        ("results_samples.csv", "samples"),
        ("results_long.csv", "long"),
        ("results_build.csv", "metric"),
        ("results_e2e.csv", "metric"),
        ("results_kernel.csv", "metric"),
        ("correctness_validation.csv", "validation"),
    ):
        overlay_csv_file(
            original_dir,
            physical_output_dir,
            correction_by_dataset,
            datasets,
            filename,
            kind,
        )
    render_validation_markdown(
        physical_output_dir / "correctness_validation.csv",
        physical_output_dir / "correctness_validation.md",
    )
    for filename in OPTIONAL_COPY_FILES:
        source = original_dir / filename
        if source.is_file():
            shutil.copyfile(source, physical_output_dir / filename)

    original_metadata = read_json(original_dir / "run_metadata.json")
    metadata = copy.deepcopy(original_metadata)
    metadata["pagerank_protocol_correction"] = {
        "status": "pass",
        "protocol": PROTOCOL_NAME,
        "action": "replace_protocol_invalid",
        "datasets": sorted(datasets),
        "adoption_policy": ADOPTION_POLICY,
        "relative_performance_considered_for_adoption": False,
        "complete_triplet_replaced": ["build", "e2e", "kernel"],
        "samples_per_metric": 5,
        "timing_rows_per_cell": 15,
        "validation_outside_timer": True,
        "original_result": {
            "logical_name": original_dir.name,
            "run_metadata_sha256": sha256(original_dir / "run_metadata.json"),
            "results_samples_sha256": sha256(
                original_dir / "results_samples.csv"
            ),
        },
        "correction_results": [
            {
                "logical_name": correction_by_dataset[dataset].name,
                "dataset": dataset,
                "run_metadata_sha256": sha256(
                    correction_by_dataset[dataset] / "run_metadata.json"
                ),
                "results_samples_sha256": sha256(
                    correction_by_dataset[dataset] / "results_samples.csv"
                ),
                "attestation_sha256": sha256(
                    correction_by_dataset[dataset]
                    / "pagerank_protocol_attestation.json"
                ),
            }
            for dataset in sorted(datasets)
        ],
        "assembled_directory": logical_output_dir,
    }
    metadata["artifacts"] = {
        **(
            metadata.get("artifacts")
            if isinstance(metadata.get("artifacts"), Mapping)
            else {}
        ),
        "files": [
            "measurement_schema.json",
            "dataset_stats.json",
            "notes.txt",
            "results_samples.csv",
            "results_long.csv",
            "results_build.csv",
            "results_kernel.csv",
            "results_e2e.csv",
            "results_memory.csv",
            "correctness_validation.csv",
            "correctness_validation.md",
            "baseline_versions.json",
            "pagerank_protocol_attestations.json",
            "PAGERANK_PROTOCOL_CORRECTION.json",
        ],
    }
    write_json(physical_output_dir / "run_metadata.json", metadata)

    attestations = []
    for dataset in sorted(datasets):
        source = correction_by_dataset[dataset]
        attestations.append(
            {
                "dataset": dataset,
                "result_logical_name": source.name,
                "sha256": sha256(
                    source / "pagerank_protocol_attestation.json"
                ),
                "attestation": validate_attestation(source),
            }
        )
    write_json(
        physical_output_dir / "pagerank_protocol_attestations.json",
        {"status": "pass", "attestations": attestations},
    )
    correction_record = metadata["pagerank_protocol_correction"]
    write_json(
        physical_output_dir / "PAGERANK_PROTOCOL_CORRECTION.json",
        correction_record,
    )
    (physical_output_dir / "notes.txt").write_text(
        "V15 deterministic PageRank protocol correction.\n"
        "All listed PageRank triplets are adopted because the old timer "
        "boundary was invalid, never because a rerun was faster.\n",
        encoding="utf-8",
    )
    return datasets


def expected_grid() -> set[tuple[str, str]]:
    return {
        (dataset, function)
        for dataset in (*REGULAR_REPLACEMENTS, *DIRECT_ANCHORS)
        for function in FUNCTIONS
    }


def file_hash_manifest(root: Path) -> dict[str, str]:
    names = (
        "pagerank_protocol_action_ledger.csv",
        "non_pagerank_cell_hashes.csv",
        "main_timing_dirs.txt",
        "anchor_timing_dirs.txt",
    )
    hashes = {}
    for name in names:
        path = root / name
        require(path.is_file(), f"missing generated audit file: {path}")
        hashes[name] = sha256(path)
    for path in sorted((root / "combined_main_timing").glob("*")):
        if not path.is_dir():
            continue
        for filename in (
            *MAIN_CORE_FILES,
            "correctness_validation.md",
            "PAGERANK_PROTOCOL_CORRECTION.json",
            "pagerank_protocol_attestations.json",
        ):
            evidence = path / filename
            require(evidence.is_file(), f"missing combined evidence: {evidence}")
            relative = str(evidence.relative_to(root))
            hashes[relative] = sha256(evidence)
    return dict(sorted(hashes.items()))


def assemble(
    *,
    original_dirs: Sequence[Path],
    correction_dirs: Sequence[Path],
    supplemental_dirs: Sequence[Path],
    anchor_dirs: Sequence[Path],
    output_root: Path,
    expected_native: str = EXPECTED_NATIVE_SHA256,
    expected_runtime: str = EXPECTED_RUNTIME_SHA256,
) -> dict[str, object]:
    expected_native = normalized_sha(expected_native, "expected native SHA")
    expected_runtime = normalized_sha(expected_runtime, "expected runtime SHA")
    original_dirs = sorted(
        {path.resolve(strict=True) for path in original_dirs},
        key=lambda path: str(path),
    )
    correction_dirs = sorted(
        {path.resolve(strict=True) for path in correction_dirs},
        key=lambda path: str(path),
    )
    supplemental_dirs = sorted(
        {path.resolve(strict=True) for path in supplemental_dirs},
        key=lambda path: str(path),
    )
    anchor_dirs = sorted(
        {path.resolve(strict=True) for path in anchor_dirs},
        key=lambda path: str(path),
    )
    output_root = output_root.resolve()
    require(original_dirs, "at least one original regular result is required")
    require(correction_dirs, "at least one correction result is required")
    require(anchor_dirs, "anchor timing results are required")
    require(not output_root.exists(), f"refusing to overwrite {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    for result_dir in (*original_dirs, *supplemental_dirs, *anchor_dirs):
        require_identity(result_dir, expected_native, expected_runtime)

    original_cells: set[tuple[str, str]] = set()
    for result_dir in original_dirs:
        original_cells.update(main_sample_cells(result_dir))
    expected_regular_grid = {
        (dataset, function)
        for dataset in REGULAR_REPLACEMENTS
        for function in FUNCTIONS
    }
    require(
        original_cells == expected_regular_grid,
        "original regular matrix is not exactly 11 datasets x 16 functions",
    )
    for result_dir in supplemental_dirs:
        supplemental_cells = main_sample_cells(result_dir)
        require(
            all(function != "PageRank" for _dataset, function in supplemental_cells),
            f"{result_dir}: supplemental evidence may not contain PageRank",
        )

    correction_by_dataset: dict[str, Path] = {}
    attestation_payload: dict | None = None
    for result_dir in correction_dirs:
        datasets = validate_regular_correction(
            result_dir, expected_native, expected_runtime
        )
        current_attestation = validate_attestation(result_dir)
        if attestation_payload is None:
            attestation_payload = current_attestation
        else:
            require(
                current_attestation == attestation_payload,
                "correction runs do not share one source-bound attestation",
            )
        for dataset in datasets:
            require(
                dataset not in correction_by_dataset,
                f"duplicate correction dataset: {dataset}",
            )
            correction_by_dataset[dataset] = result_dir
    require(
        set(correction_by_dataset) == set(REGULAR_REPLACEMENTS),
        "correction set is not exactly the 11 affected regular datasets",
    )

    anchors_before = anchor_timing_payloads(anchor_dirs)
    validate_anchor_pagerank(anchors_before)
    anchor_cells = set(anchors_before)
    expected_anchor_grid = {
        (dataset, function)
        for dataset in DIRECT_ANCHORS
        for function in FUNCTIONS
    }
    require(
        anchor_cells == expected_anchor_grid,
        "anchor evidence is not exactly Orkut/GAP x 16 functions",
    )
    require(
        original_cells | anchor_cells == expected_grid(),
        "regular plus anchor evidence does not form the 208-cell matrix",
    )

    before_regular = regular_cell_payloads(
        [*original_dirs, *supplemental_dirs]
    )
    before_hashes = cell_hashes(before_regular, anchors_before)
    require(
        len(before_hashes) == 195,
        f"non-PageRank cell count before correction is {len(before_hashes)}",
    )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.assembling.",
            dir=str(output_root.parent),
        )
    )
    completed = False
    try:
        combined_root = staging / "combined_main_timing"
        combined_root.mkdir()
        combined_physical_dirs: list[Path] = []
        combined_final_dirs: list[Path] = []
        replaced: set[str] = set()
        for index, original_dir in enumerate(original_dirs, start=1):
            logical_name = f"{index:02d}_{original_dir.name}"
            physical = combined_root / logical_name
            final = output_root / "combined_main_timing" / logical_name
            datasets = assemble_one_main_dir(
                original_dir,
                physical,
                str(Path("combined_main_timing") / logical_name),
                correction_by_dataset,
            )
            require(
                not (replaced & datasets),
                f"regular PageRank dataset assembled twice: {sorted(replaced & datasets)}",
            )
            replaced.update(datasets)
            combined_physical_dirs.append(physical)
            combined_final_dirs.append(final)
        require(
            replaced == set(REGULAR_REPLACEMENTS),
            "assembled PageRank replacement set differs",
        )

        after_regular = regular_cell_payloads(
            [*combined_physical_dirs, *supplemental_dirs]
        )
        anchors_after = anchor_timing_payloads(anchor_dirs)
        after_hashes = cell_hashes(after_regular, anchors_after)
        require(
            before_hashes == after_hashes,
            "a non-PageRank cell changed during PageRank assembly",
        )
        global_before = canonical_sha(
            [
                {"dataset": key[0], "function": key[1], "sha256": value}
                for key, value in sorted(before_hashes.items())
            ]
        )
        global_after = canonical_sha(
            [
                {"dataset": key[0], "function": key[1], "sha256": value}
                for key, value in sorted(after_hashes.items())
            ]
        )
        require(
            global_before == global_after,
            "global non-PageRank normalized hash changed",
        )

        action_fields = (
            "dataset",
            "function",
            "action",
            "reason",
            "source_kind",
            "protocol",
            "samples_per_metric",
            "timing_rows_per_cell",
            "validation_outside_timer",
            "relative_performance_considered_for_adoption",
        )
        action_rows = []
        for dataset in REGULAR_REPLACEMENTS:
            action_rows.append(
                {
                    "dataset": dataset,
                    "function": "PageRank",
                    "action": "replace_protocol_invalid",
                    "reason": (
                        "original E2E included generic signature/keyword "
                        "adaptation inside the public-call timer"
                    ),
                    "source_kind": "regular_correction",
                    "protocol": PROTOCOL_NAME,
                    "samples_per_metric": 5,
                    "timing_rows_per_cell": 15,
                    "validation_outside_timer": "true",
                    "relative_performance_considered_for_adoption": "false",
                }
            )
        for dataset in DIRECT_ANCHORS:
            action_rows.append(
                {
                    "dataset": dataset,
                    "function": "PageRank",
                    "action": "retain_equivalent_direct_protocol",
                    "reason": (
                        "run_eggpu_scaling already times one direct public "
                        "invocation and validates after return"
                    ),
                    "source_kind": "direct_anchor",
                    "protocol": "eggpu_scaling_direct_public_call",
                    "samples_per_metric": 5,
                    "timing_rows_per_cell": 15,
                    "validation_outside_timer": "true",
                    "relative_performance_considered_for_adoption": "false",
                }
            )
        write_csv(
            staging / "pagerank_protocol_action_ledger.csv",
            action_fields,
            action_rows,
        )

        hash_fields = ("dataset", "function", "before_sha256", "after_sha256")
        hash_rows = [
            {
                "dataset": key[0],
                "function": key[1],
                "before_sha256": before_hashes[key],
                "after_sha256": after_hashes[key],
            }
            for key in sorted(before_hashes)
        ]
        write_csv(
            staging / "non_pagerank_cell_hashes.csv",
            hash_fields,
            hash_rows,
        )

        main_timing_dirs = [
            *combined_final_dirs,
            *supplemental_dirs,
        ]
        (staging / "main_timing_dirs.txt").write_text(
            "".join(f"{path}\n" for path in main_timing_dirs),
            encoding="utf-8",
        )
        (staging / "anchor_timing_dirs.txt").write_text(
            "".join(f"{path}\n" for path in anchor_dirs),
            encoding="utf-8",
        )

        input_evidence = []
        for kind, paths in (
            ("original", original_dirs),
            ("correction", correction_dirs),
            ("supplemental", supplemental_dirs),
            ("anchor", anchor_dirs),
        ):
            for path in paths:
                input_evidence.append(directory_evidence_digest(path, kind))

        # All files named here are written before their hashes are recorded.
        output_hashes = file_hash_manifest(staging)
        audit = {
            "status": "pass",
            "schema_version": 1,
            "protocol": PROTOCOL_NAME,
            "candidate_sha256": expected_native,
            "runtime_python_snapshot_sha256": expected_runtime,
            "adoption_policy": ADOPTION_POLICY,
            "relative_performance_considered_for_adoption": False,
            "relative_speed_comparison_performed": False,
            "actions": {
                "replace_protocol_invalid": len(REGULAR_REPLACEMENTS),
                "retain_equivalent_direct_protocol": len(DIRECT_ANCHORS),
                "total_pagerank_cells": 13,
            },
            "measurement_contract": {
                "samples_per_metric": 5,
                "metrics": list(METRICS),
                "timing_rows_per_cell": 15,
                "kernel_relation": "0 < kernel < e2e for every sample",
                "validation_outside_timer": True,
                "missing_kernel_time": "fail_closed",
            },
            "matrix": {
                "functions": len(FUNCTIONS),
                "datasets": 13,
                "eggpu_cells": 208,
                "pagerank_cells": 13,
                "non_pagerank_cells": len(before_hashes),
            },
            "non_pagerank_identity": {
                "status": "pass",
                "normalized_cell_count": len(before_hashes),
                "before_sha256": global_before,
                "after_sha256": global_after,
                "all_cell_hashes_equal": True,
            },
            "combined_main_timing_dirs": [
                str(path) for path in combined_final_dirs
            ],
            "supplemental_main_timing_dirs": [
                str(path) for path in supplemental_dirs
            ],
            "anchor_timing_dirs": [str(path) for path in anchor_dirs],
            "input_evidence": input_evidence,
            "output_file_sha256": output_hashes,
        }
        write_json(
            staging / "V15_PAGERANK_PROTOCOL_CORRECTION_AUDIT.json",
            audit,
        )
        os.replace(staging, output_root)
        completed = True
        return audit
    finally:
        if not completed and staging.exists():
            shutil.rmtree(staging)


def read_directory_list(path: Path, label: str) -> list[Path]:
    require(path.is_file(), f"missing {label}: {path}")
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    require(bool(raw_lines), f"{label} is empty")
    require(
        all(line and line == line.strip() for line in raw_lines),
        f"{label} contains a blank or whitespace-padded path",
    )
    paths = [Path(line).resolve(strict=True) for line in raw_lines]
    require(
        all(path.is_dir() for path in paths),
        f"{label} contains a non-directory path",
    )
    require(len(paths) == len(set(paths)), f"{label} contains a duplicate path")
    return paths


def audit_directory_list(
    audit: Mapping[str, object], field: str, label: str
) -> list[Path]:
    raw = audit.get(field)
    require(isinstance(raw, list) and raw, f"audit {label} is absent or empty")
    require(
        all(
            isinstance(value, str)
            and bool(value)
            and value == value.strip()
            for value in raw
        ),
        f"audit {label} contains an invalid path",
    )
    paths = [Path(value).resolve(strict=True) for value in raw]
    require(len(paths) == len(set(paths)), f"audit {label} contains duplicates")
    return paths


def verify_existing(
    output_root: Path,
    *,
    expected_native: str = EXPECTED_NATIVE_SHA256,
    expected_runtime: str = EXPECTED_RUNTIME_SHA256,
) -> dict[str, object]:
    output_root = output_root.resolve(strict=True)
    audit_path = output_root / "V15_PAGERANK_PROTOCOL_CORRECTION_AUDIT.json"
    audit = read_json(audit_path)
    require(audit.get("status") == "pass", "correction audit status differs")
    require(audit.get("protocol") == PROTOCOL_NAME, "correction protocol differs")
    require(
        audit.get("candidate_sha256")
        == normalized_sha(expected_native, "expected native SHA"),
        "correction candidate SHA differs",
    )
    require(
        audit.get("runtime_python_snapshot_sha256")
        == normalized_sha(expected_runtime, "expected runtime SHA"),
        "correction runtime SHA differs",
    )
    require(
        audit.get("adoption_policy") == ADOPTION_POLICY
        and audit.get("relative_performance_considered_for_adoption") is False
        and audit.get("relative_speed_comparison_performed") is False,
        "correction adoption policy is not protocol-only",
    )
    actions = audit.get("actions")
    require(
        isinstance(actions, Mapping)
        and actions.get("replace_protocol_invalid") == 11
        and actions.get("retain_equivalent_direct_protocol") == 2
        and actions.get("total_pagerank_cells") == 13,
        "correction action counts differ",
    )
    non_pr = audit.get("non_pagerank_identity")
    require(
        isinstance(non_pr, Mapping)
        and non_pr.get("status") == "pass"
        and non_pr.get("normalized_cell_count") == 195
        and non_pr.get("before_sha256") == non_pr.get("after_sha256")
        and non_pr.get("all_cell_hashes_equal") is True,
        "195-cell non-PageRank identity gate differs",
    )
    output_hashes = audit.get("output_file_sha256")
    require(isinstance(output_hashes, Mapping), "output hash manifest missing")
    for relative, expected in output_hashes.items():
        path = output_root / str(relative)
        require(path.is_file(), f"generated correction evidence missing: {path}")
        require(
            sha256(path) == expected,
            f"generated correction evidence changed: {path}",
        )
    _action_fields, action_rows = read_csv(
        output_root / "pagerank_protocol_action_ledger.csv"
    )
    require(
        len(action_rows) == len(REGULAR_REPLACEMENTS) + len(DIRECT_ANCHORS),
        "PageRank action ledger row count differs",
    )
    require(
        {
            row.get("dataset")
            for row in action_rows
            if row.get("action") == "replace_protocol_invalid"
        }
        == set(REGULAR_REPLACEMENTS),
        "PageRank replacement action set differs",
    )
    require(
        {
            row.get("dataset")
            for row in action_rows
            if row.get("action") == "retain_equivalent_direct_protocol"
        }
        == set(DIRECT_ANCHORS),
        "PageRank retained-anchor action set differs",
    )
    require(
        all(
            row.get("function") == "PageRank"
            and str(
                row.get("relative_performance_considered_for_adoption", "")
            ).lower()
            == "false"
            for row in action_rows
        ),
        "PageRank action ledger semantics differ",
    )
    _hash_fields, non_pr_rows = read_csv(
        output_root / "non_pagerank_cell_hashes.csv"
    )
    require(
        len(non_pr_rows) == 195
        and all(
            row.get("function") != "PageRank"
            and row.get("before_sha256") == row.get("after_sha256")
            for row in non_pr_rows
        ),
        "195-cell non-PageRank hash ledger differs",
    )
    evidence = audit.get("input_evidence")
    require(isinstance(evidence, list), "input evidence manifest missing")
    for record in evidence:
        require(isinstance(record, Mapping), "invalid input evidence record")
        observed = directory_evidence_digest(
            Path(str(record.get("path"))),
            str(record.get("kind")),
        )
        require(
            observed["digest"] == record.get("digest"),
            f"input evidence changed: {record.get('path')}",
        )
    timing_dirs = read_directory_list(
        output_root / "main_timing_dirs.txt", "main timing directory list"
    )
    expected_timing_dirs = [
        *audit_directory_list(
            audit, "combined_main_timing_dirs", "combined main timing dirs"
        ),
        *(
            audit_directory_list(
                audit,
                "supplemental_main_timing_dirs",
                "supplemental main timing dirs",
            )
            if audit.get("supplemental_main_timing_dirs")
            else []
        ),
    ]
    require(
        timing_dirs == expected_timing_dirs,
        "main timing directory list differs semantically from the audit",
    )
    anchor_dirs = read_directory_list(
        output_root / "anchor_timing_dirs.txt", "anchor timing directory list"
    )
    expected_anchor_dirs = audit_directory_list(
        audit, "anchor_timing_dirs", "anchor timing dirs"
    )
    require(
        anchor_dirs == expected_anchor_dirs,
        "anchor timing directory list differs semantically from the audit",
    )
    return audit


def validate_correction_batch(
    correction_dirs: Sequence[Path],
    *,
    expected_native: str = EXPECTED_NATIVE_SHA256,
    expected_runtime: str = EXPECTED_RUNTIME_SHA256,
) -> dict[str, object]:
    """Validate any non-empty partial correction batch without anchors."""

    expected_native = normalized_sha(expected_native, "expected native SHA")
    expected_runtime = normalized_sha(expected_runtime, "expected runtime SHA")
    correction_dirs = sorted(
        {path.resolve(strict=True) for path in correction_dirs},
        key=lambda path: str(path),
    )
    require(correction_dirs, "at least one correction result is required")
    datasets_by_dir: dict[str, list[str]] = {}
    discovered: dict[str, str] = {}
    common_attestation: dict | None = None
    for result_dir in correction_dirs:
        datasets = validate_regular_correction(
            result_dir, expected_native, expected_runtime
        )
        current_attestation = validate_attestation(result_dir)
        if common_attestation is None:
            common_attestation = current_attestation
        else:
            require(
                current_attestation == common_attestation,
                "partial correction runs do not share one source-bound attestation",
            )
        datasets_by_dir[str(result_dir)] = sorted(datasets)
        for dataset in datasets:
            require(
                dataset not in discovered,
                f"duplicate correction dataset: {dataset}",
            )
            discovered[dataset] = str(result_dir)
    return {
        "status": "pass",
        "mode": "partial_correction_batch_validation",
        "protocol": PROTOCOL_NAME,
        "candidate_sha256": expected_native,
        "runtime_python_snapshot_sha256": expected_runtime,
        "adoption_policy": ADOPTION_POLICY,
        "relative_performance_considered_for_adoption": False,
        "correction_result_dirs": datasets_by_dir,
        "dataset_count": len(discovered),
        "datasets": sorted(discovered),
        "complete_11_dataset_set": (
            set(discovered) == set(REGULAR_REPLACEMENTS)
        ),
        "remaining_datasets": sorted(
            set(REGULAR_REPLACEMENTS) - set(discovered)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-result-dir", type=Path, action="append", default=[])
    parser.add_argument("--correction-result-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--supplemental-main-result-dir",
        type=Path,
        action="append",
        default=[],
        help="Unchanged regular timing recovery, e.g. sampled Closeness.",
    )
    parser.add_argument("--anchor-result-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--verify-existing-root",
        type=Path,
        help="Recheck an assembled output and all hash-bound inputs.",
    )
    parser.add_argument(
        "--validate-correction-only",
        action="store_true",
        help=(
            "Validate any partial set of --correction-result-dir inputs "
            "without requiring anchors or assembling the 11-cell union."
        ),
    )
    parser.add_argument(
        "--expected-native-sha256", default=EXPECTED_NATIVE_SHA256
    )
    parser.add_argument(
        "--expected-runtime-sha256", default=EXPECTED_RUNTIME_SHA256
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_correction_only:
        require(
            args.verify_existing_root is None
            and args.output_root is None
            and not args.original_result_dir
            and not args.supplemental_main_result_dir
            and not args.anchor_result_dir,
            "--validate-correction-only accepts only correction result dirs",
        )
        audit = validate_correction_batch(
            args.correction_result_dir,
            expected_native=args.expected_native_sha256,
            expected_runtime=args.expected_runtime_sha256,
        )
    elif args.verify_existing_root:
        require(
            not any(
                (
                    args.original_result_dir,
                    args.correction_result_dir,
                    args.supplemental_main_result_dir,
                    args.anchor_result_dir,
                )
            )
            and args.output_root is None,
            "--verify-existing-root may not be combined with assembly inputs",
        )
        audit = verify_existing(
            args.verify_existing_root,
            expected_native=args.expected_native_sha256,
            expected_runtime=args.expected_runtime_sha256,
        )
    else:
        require(args.output_root is not None, "--output-root is required")
        audit = assemble(
            original_dirs=args.original_result_dir,
            correction_dirs=args.correction_result_dir,
            supplemental_dirs=args.supplemental_main_result_dir,
            anchor_dirs=args.anchor_result_dir,
            output_root=args.output_root,
            expected_native=args.expected_native_sha256,
            expected_runtime=args.expected_runtime_sha256,
        )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
