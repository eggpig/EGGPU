#!/usr/bin/env python3
"""Overlay strictly qualified EGGPU results onto a final-13 ledger.

The input ledger is never modified.  Main-matrix inputs are read from
``run_full_baselines.py`` result directories; scale-anchor inputs are read
from ``run_eggpu_scaling.py`` result directories.  Timing and memory inputs
are separate, explicit streams.  Unified-candidate mode requires all 208
timing cells (five samples) and all 208 memory cells (three samples) to carry
the same compiled-extension and frozen-Python-runtime identities before any
ledger row is written.

The legacy timing-only mode remains available for archived V9 reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from stable_timing_protocol import (
    DEFAULT_MAX_OVER_MEDIAN_LIMIT,
    DEFAULT_MEDIAN_OVER_MIN_LIMIT,
    FORMAL_BATCH_SELECTION_POLICY as BATCH_SELECTION_POLICY,
    GATE_CALIBRATION,
    PAPER_ESTIMATOR as PAPER_TIMING_ESTIMATOR,
    VARIANCE_POLICY as DEFAULT_VARIANCE_POLICY,
    StableTimingProtocolError,
    gate_calibration_payload,
    validate_gate_calibration,
)

EXPECTED_LEDGER_CELLS = 13 * 16 * 8
EXPECTED_EGGPU_CELLS = 13 * 16
ANCHOR_DATASETS = {"com-Orkut", "GAP-twitter"}
METRICS = ("build", "e2e", "kernel")
MEMORY_METRICS = {
    "gpu": ("memory_peak_gpu_proc_mb", "memory_peak_gpu_proc_delta_mb"),
    "host": ("memory_peak_rss_mb", "memory_peak_rss_delta_mb"),
}
MAIN_MEMORY_WINDOW = "isolated_memory_subprocess"
ANCHOR_MEMORY_WINDOW = "isolated_worker_process_full_lifetime"
ANCHOR_RUN_METADATA_PROTOCOL = "eggpu_scaling_frozen_runtime_v10"
SHA256_RE = re.compile(r"\b([0-9a-fA-F]{64})\b")


class GateError(ValueError):
    """A result exists but does not satisfy the publication gate."""


@dataclass(frozen=True)
class TimingStats:
    samples: tuple[float, ...]
    minimum: float
    mean: float
    stdev: float
    maximum: float
    median: float
    coefficient_of_variation: float
    max_over_median: float
    median_over_minimum: float
    variance_policy: str
    stability_status: str


@dataclass(frozen=True)
class Candidate:
    dataset: str
    function: str
    source_kind: str
    result_source: str
    candidate_sha256: str
    metrics: dict[str, TimingStats]
    sample_count: int = 5
    runtime_python_sha256: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return self.dataset, self.function


@dataclass(frozen=True)
class MemoryCandidate:
    dataset: str
    function: str
    source_kind: str
    result_source: str
    candidate_sha256: str
    runtime_python_sha256: str
    gpu_peak_mb: tuple[float, float]
    host_rss_peak_mb: tuple[float, float]
    measurement_window: str
    sample_count: int = 3

    @property
    def key(self) -> tuple[str, str]:
        return self.dataset, self.function


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(
    path: Path, rows: Iterable[dict[str, object]], fields: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_atomic(
    path: Path, rows: Iterable[dict[str, object]], fields: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(
                handle, fieldnames=fields, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def finite_number(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise GateError(f"{field} is not finite: {value!r}")
    return number


def integer(value: object, field: str) -> int:
    number = finite_number(value, field)
    result = int(number)
    if number != result:
        raise GateError(f"{field} is not an integer: {value!r}")
    return result


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "ok"}


def require_close(actual: object, expected: float, field: str) -> None:
    value = finite_number(actual, field)
    if not math.isclose(value, expected, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise GateError(f"{field}={value} does not match samples={expected}")


def normalize_sha(value: object, field: str = "candidate_sha256") -> str:
    text = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise GateError(f"{field} is not a SHA-256 digest: {value!r}")
    return text


def nested_values(payload: dict, paths: Iterable[tuple[str, ...]]) -> list[str]:
    values = []
    for path in paths:
        current: object = payload
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current:
            values.append(str(current))
    return values


def candidate_sha_from_result_dir(
    result_dir: Path, fallback_sha: str | None
) -> tuple[str, str]:
    values: list[tuple[str, str]] = []
    metadata_paths = (
        result_dir / "run_metadata.json",
        result_dir / "baseline_versions.json",
        result_dir / "anchor_run_metadata.json",
        result_dir / "protocol.json",
    )
    sha_paths = (
        ("candidate_sha256",),
        ("native_binary_sha256",),
        ("repository_runtime_provenance", "native_sha256"),
        (
            "repository_runtime_provenance",
            "modules",
            "cpp_easygraph",
            "sha256",
        ),
        ("baseline_versions", "cpp_easygraph", "sha256"),
        ("build_artifacts", "active_cpp_easygraph", "sha256"),
        ("cpp_easygraph", "sha256"),
    )
    for path in metadata_paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for value in nested_values(payload, sha_paths):
            values.append((normalize_sha(value), str(path.resolve())))

    # Historical scaling runs recorded the extension hash in a sibling launch
    # log rather than in the result directory.
    for suffix in (".launch.log", ".console.log", ".run.log"):
        path = Path(str(result_dir) + suffix)
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "cpp_easygraph" not in line and "candidate" not in line:
                continue
            for match in SHA256_RE.findall(line):
                values.append((normalize_sha(match), str(path.resolve())))

    if fallback_sha:
        values.append((normalize_sha(fallback_sha, "--candidate-sha256"), "CLI"))
    unique = sorted({value for value, _source in values})
    if not unique:
        raise GateError(
            "candidate SHA is absent; provide run_metadata.json, a launch log "
            "with the cpp_easygraph SHA, or --candidate-sha256"
        )
    if len(unique) != 1:
        raise GateError(f"conflicting candidate SHA values: {unique}")
    sources = sorted({source for value, source in values if value == unique[0]})
    return unique[0], ";".join(sources)


def runtime_python_snapshot_from_result_dir(
    result_dir: Path,
    *,
    required: bool,
) -> tuple[str, str, bool | None]:
    """Read the run-recorded frozen Python package identity.

    A compiled extension hash is insufficient because dispatch and result
    reconstruction live in ``easygraph/**/*.py``.  Unified mode therefore
    accepts only a recorded content digest and an explicit non-symlink
    package assertion.  It never derives provenance from a mutable live tree.
    """

    values: list[tuple[str, str]] = []
    symlink_values: list[tuple[bool, str]] = []
    for path in (
        result_dir / "run_metadata.json",
        result_dir / "anchor_run_metadata.json",
        result_dir / "protocol.json",
    ):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        snapshots = []
        if isinstance(payload.get("runtime_python_snapshot"), dict):
            snapshots.append(
                (
                    payload["runtime_python_snapshot"],
                    "runtime_python_snapshot",
                )
            )
        repository = payload.get("repository_runtime_provenance")
        if isinstance(repository, dict) and isinstance(
            repository.get("runtime_python_snapshot"), dict
        ):
            snapshots.append(
                (
                    repository["runtime_python_snapshot"],
                    "repository_runtime_provenance.runtime_python_snapshot",
                )
            )
        for snapshot, field_prefix in snapshots:
            digest = snapshot.get("digest")
            if digest:
                values.append(
                    (
                        normalize_sha(digest, f"{field_prefix}.digest"),
                        str(path.resolve()),
                    )
                )
            if "package_is_symlink" in snapshot:
                if not isinstance(snapshot["package_is_symlink"], bool):
                    raise GateError(
                        f"{field_prefix}.package_is_symlink must be a JSON boolean"
                    )
                symlink_values.append(
                    (snapshot["package_is_symlink"], str(path.resolve()))
                )
        for key in (
            "runtime_python_snapshot_sha256",
            "runtime_python_snapshot_digest",
            "runtime_python_digest",
        ):
            if payload.get(key):
                values.append(
                    (
                        normalize_sha(payload[key], key),
                        str(path.resolve()),
                    )
                )

    unique = sorted({value for value, _source in values})
    if not unique:
        if required:
            raise GateError(
                "runtime Python snapshot is absent; unified mode requires a "
                "recorded runtime_python_snapshot.digest"
            )
        return "", "", None
    if len(unique) != 1:
        raise GateError(f"conflicting runtime Python snapshot digests: {unique}")

    observed_symlink = sorted({value for value, _source in symlink_values})
    if len(observed_symlink) > 1:
        raise GateError(
            "conflicting runtime_python_snapshot.package_is_symlink values"
        )
    package_is_symlink = observed_symlink[0] if observed_symlink else None
    if required:
        if package_is_symlink is None:
            raise GateError(
                "unified mode requires an explicit "
                "runtime_python_snapshot.package_is_symlink field"
            )
        if package_is_symlink:
            raise GateError(
                "unified mode rejects a symlinked EasyGraph Python runtime"
            )
    sources = sorted({source for value, source in values if value == unique[0]})
    return unique[0], ";".join(sources), package_is_symlink


def normalized_recorded_path(value: object, field: str) -> Path:
    """Normalize an absolute path string without consulting the live filesystem."""

    text = str(value or "").strip()
    path = Path(text)
    if not text or not path.is_absolute():
        raise GateError(f"{field} is not an absolute path: {value!r}")
    return Path(os.path.normpath(text))


def require_recorded_path_within(
    value: object,
    root: Path,
    field: str,
) -> Path:
    path = normalized_recorded_path(value, field)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise GateError(
            f"{field} resolves outside the recorded runtime root: {path}"
        ) from exc
    return path


def require_protocol_integer(
    payload: dict[str, object],
    key: str,
    expected: int,
    field_prefix: str,
) -> int:
    value = payload.get(key)
    if isinstance(value, bool):
        raise GateError(f"{field_prefix}.{key} must be the integer {expected}")
    observed = integer(value, f"{field_prefix}.{key}")
    if observed != expected:
        raise GateError(
            f"{field_prefix}.{key}={observed}, expected {expected}"
        )
    return observed


def validate_anchor_run_metadata_v10(
    payload: dict[str, object],
    candidate_sha: str,
    runtime_sha: str,
    *,
    measurement_kind: str,
) -> dict[str, object]:
    """Validate and normalize the self-contained scaling-run provenance."""

    if payload.get("protocol") != ANCHOR_RUN_METADATA_PROTOCOL:
        raise GateError(
            "run_metadata.protocol is not "
            f"{ANCHOR_RUN_METADATA_PROTOCOL!r}"
        )
    for key, expected in (
        ("repeat", 5),
        ("timing_processes", 0),
        ("warmup", 1),
        ("memory_repeat", 3),
    ):
        require_protocol_integer(payload, key, expected, "run_metadata")

    recorded_candidate_fields = {
        "run_metadata.native_binary_sha256": payload.get(
            "native_binary_sha256"
        ),
    }
    repository = payload.get("repository_runtime_provenance")
    if not isinstance(repository, dict):
        raise GateError(
            "run_metadata.repository_runtime_provenance is missing"
        )
    recorded_candidate_fields[
        "run_metadata.repository_runtime_provenance.native_sha256"
    ] = repository.get("native_sha256")
    modules = repository.get("modules")
    if not isinstance(modules, dict):
        raise GateError(
            "run_metadata.repository_runtime_provenance.modules is missing"
        )
    cpp_module = modules.get("cpp_easygraph")
    easygraph_module = modules.get("easygraph")
    if not isinstance(cpp_module, dict) or not isinstance(easygraph_module, dict):
        raise GateError(
            "run_metadata runtime modules must include easygraph and cpp_easygraph"
        )
    recorded_candidate_fields[
        "run_metadata.repository_runtime_provenance."
        "modules.cpp_easygraph.sha256"
    ] = cpp_module.get("sha256")
    for field, value in recorded_candidate_fields.items():
        observed = normalize_sha(value, field)
        if observed != candidate_sha:
            raise GateError(
                f"{field}={observed} differs from candidate {candidate_sha}"
            )

    snapshot = repository.get("runtime_python_snapshot")
    if not isinstance(snapshot, dict):
        raise GateError(
            "run_metadata.repository_runtime_provenance."
            "runtime_python_snapshot is missing"
        )
    for field, value in (
        ("run_metadata.runtime_python_digest", payload.get("runtime_python_digest")),
        (
            "run_metadata.repository_runtime_provenance."
            "runtime_python_snapshot.digest",
            snapshot.get("digest"),
        ),
    ):
        observed = normalize_sha(value, field)
        if observed != runtime_sha:
            raise GateError(
                f"{field}={observed} differs from runtime {runtime_sha}"
            )
    if snapshot.get("package_is_symlink") is not False:
        raise GateError(
            "run_metadata runtime_python_snapshot.package_is_symlink "
            "must be false"
        )
    if snapshot.get("algorithm") != "sha256":
        raise GateError(
            "run_metadata runtime_python_snapshot.algorithm must be 'sha256'"
        )
    if isinstance(snapshot.get("file_count"), bool):
        raise GateError("runtime_python_snapshot.file_count must be positive")
    if integer(
        snapshot.get("file_count"), "runtime_python_snapshot.file_count"
    ) < 1:
        raise GateError("runtime_python_snapshot.file_count must be positive")

    resolved_root = normalized_recorded_path(
        repository.get("resolved_root"),
        "repository_runtime_provenance.resolved_root",
    )
    requested_root = normalized_recorded_path(
        repository.get("requested_root"),
        "repository_runtime_provenance.requested_root",
    )
    if normalized_recorded_path(
        payload.get("easygraph_repo"), "run_metadata.easygraph_repo"
    ) != resolved_root:
        raise GateError(
            "run_metadata.easygraph_repo differs from the recorded resolved root"
        )
    if normalized_recorded_path(
        snapshot.get("runtime_root"),
        "runtime_python_snapshot.runtime_root",
    ) != resolved_root:
        raise GateError(
            "runtime_python_snapshot.runtime_root differs from the "
            "recorded resolved root"
        )
    package_path = require_recorded_path_within(
        snapshot.get("package_path"),
        requested_root,
        "runtime_python_snapshot.package_path",
    )
    package_resolved_path = require_recorded_path_within(
        snapshot.get("package_resolved_path"),
        resolved_root,
        "runtime_python_snapshot.package_resolved_path",
    )
    if package_path != requested_root / "easygraph":
        raise GateError(
            "runtime_python_snapshot.package_path is not the requested "
            "easygraph package"
        )
    if package_resolved_path != resolved_root / "easygraph":
        raise GateError(
            "runtime_python_snapshot.package_resolved_path is not the "
            "resolved easygraph package"
        )

    python_origins = payload.get("python_origins")
    if not isinstance(python_origins, dict):
        raise GateError("run_metadata.python_origins is missing")
    for module_name, expected_native in (
        ("easygraph", False),
        ("cpp_easygraph", True),
    ):
        module = modules[module_name]
        module_origin = require_recorded_path_within(
            module.get("module_origin_resolved"),
            resolved_root,
            (
                "repository_runtime_provenance.modules."
                f"{module_name}.module_origin_resolved"
            ),
        )
        top_origin = require_recorded_path_within(
            python_origins.get(module_name),
            resolved_root,
            f"run_metadata.python_origins.{module_name}",
        )
        if top_origin != module_origin:
            raise GateError(
                f"run_metadata.python_origins.{module_name} does not match "
                "the repository runtime provenance"
            )
        relative = str(module.get("relative_to_runtime") or "").strip()
        if (
            not relative
            or Path(relative).is_absolute()
            or Path(os.path.normpath(str(resolved_root / relative)))
            != module_origin
        ):
            raise GateError(
                "repository_runtime_provenance.modules."
                f"{module_name}.relative_to_runtime is invalid"
            )
        if module.get("is_native_extension") is not expected_native:
            raise GateError(
                "repository_runtime_provenance.modules."
                f"{module_name}.is_native_extension is invalid"
            )
        normalize_sha(
            module.get("sha256"),
            (
                "repository_runtime_provenance.modules."
                f"{module_name}.sha256"
            ),
        )

    requested_functions = payload.get("requested_functions")
    requested_manifests = payload.get("requested_manifests")
    if (
        not isinstance(requested_functions, str)
        or not requested_functions.strip()
    ):
        raise GateError("run_metadata.requested_functions is empty")
    if (
        not isinstance(requested_manifests, list)
        or not requested_manifests
        or any(
            not isinstance(item, str) or not item.strip()
            for item in requested_manifests
        )
    ):
        raise GateError("run_metadata.requested_manifests is empty or invalid")

    normalized = {
        "protocol_name": ANCHOR_RUN_METADATA_PROTOCOL,
        "aggregate": "arithmetic_mean",
        "aggregation": "arithmetic_mean",
        "repeat": 5,
        "timing_processes": 0,
        "warmup": 1,
        "memory_repeat": 3,
        "first_use_calls_per_process": 1,
        "extra_untimed_warmups_per_process": 1,
        "measured_call_ordinal": 3,
        "measured_calls_per_process": 1,
        "candidate_sha256": candidate_sha,
        "runtime_python_snapshot_sha256": runtime_sha,
        "runtime_package_is_symlink": False,
    }
    if measurement_kind == "timing":
        normalized["independent_process_samples"] = 5
    elif measurement_kind == "memory":
        normalized.update(
            {
                "independent_process_samples": 3,
                "memory_measurement_window": ANCHOR_MEMORY_WINDOW,
            }
        )
    else:
        raise GateError(f"invalid anchor measurement kind: {measurement_kind!r}")
    return normalized


def anchor_provenance_evidence(
    result_dir: Path,
    candidate_sha: str,
    runtime_sha: str,
    *,
    strict_provenance: bool,
    measurement_kind: str,
) -> dict[str, object]:
    """Select self-contained V10 metadata first, with a legacy sidecar fallback."""

    run_metadata_path = result_dir / "run_metadata.json"
    protocol_path = result_dir / "protocol.json"
    protocol_payload: dict[str, object] = {}
    if protocol_path.is_file():
        protocol_payload = json.loads(protocol_path.read_text(encoding="utf-8"))
        if not isinstance(protocol_payload, dict):
            raise GateError(f"{protocol_path}: expected a JSON object")

    run_metadata_payload: dict[str, object] = {}
    if run_metadata_path.is_file():
        run_metadata_payload = json.loads(
            run_metadata_path.read_text(encoding="utf-8")
        )
        if not isinstance(run_metadata_payload, dict):
            raise GateError(f"{run_metadata_path}: expected a JSON object")
    is_v10_shape = bool(
        run_metadata_payload
        and (
            run_metadata_payload.get("protocol")
            == ANCHOR_RUN_METADATA_PROTOCOL
            or "repository_runtime_provenance" in run_metadata_payload
            or "native_binary_sha256" in run_metadata_payload
            or "runtime_python_digest" in run_metadata_payload
        )
    )

    common = {
        "run_metadata_path": (
            str(run_metadata_path.resolve()) if run_metadata_path.is_file() else ""
        ),
        "run_metadata_sha256": (
            sha256(run_metadata_path) if run_metadata_path.is_file() else ""
        ),
        "protocol_path": (
            str(protocol_path.resolve()) if protocol_path.is_file() else ""
        ),
        "protocol_json_sha256": (
            sha256(protocol_path) if protocol_path.is_file() else ""
        ),
    }
    if is_v10_shape:
        measurement_protocol = validate_anchor_run_metadata_v10(
            run_metadata_payload,
            candidate_sha,
            runtime_sha,
            measurement_kind=measurement_kind,
        )
        return {
            **common,
            "anchor_metadata_path": str(run_metadata_path.resolve()),
            "anchor_metadata_sha256": sha256(run_metadata_path),
            "anchor_metadata_schema": "run_metadata_v10",
            "runtime_resolved_root": str(
                normalized_recorded_path(
                    run_metadata_payload[
                        "repository_runtime_provenance"
                    ]["resolved_root"],
                    "repository_runtime_provenance.resolved_root",
                )
            ),
            "measurement_protocol": measurement_protocol,
            "protocol_evidence": (
                "run_metadata.json records the frozen runtime, in-repository "
                "module origins, and repeat=5/timing_processes=0/warmup=1/"
                "memory_repeat=3"
            ),
        }

    if not protocol_payload:
        if strict_provenance:
            raise GateError(
                f"{result_dir}: unified anchor provenance requires "
                "run_metadata.json V10 or protocol.json"
            )
        return {
            **common,
            "anchor_metadata_path": "",
            "anchor_metadata_sha256": "",
            "anchor_metadata_schema": "unrecorded_legacy",
            "measurement_protocol": {},
            "protocol_evidence": (
                "five worker JSON records with one measured sample each; "
                "run_eggpu_scaling output does not encode the warmup count"
            ),
        }

    warmup_value = protocol_payload.get("warmup")
    legacy_warmup = protocol_payload.get(
        "extra_untimed_warmups_per_process"
    )
    if warmup_value is None:
        warmup_value = legacy_warmup
    elif legacy_warmup is not None and integer(
        warmup_value, "protocol.warmup"
    ) != integer(
        legacy_warmup, "protocol.extra_untimed_warmups_per_process"
    ):
        raise GateError(
            "protocol warmup conflicts with "
            "extra_untimed_warmups_per_process"
        )

    if strict_provenance:
        protocol_candidate = normalize_sha(
            protocol_payload.get("candidate_sha256"),
            "protocol.candidate_sha256",
        )
        if protocol_candidate != candidate_sha:
            raise GateError("protocol candidate SHA differs from the run candidate")
        snapshot = protocol_payload.get("runtime_python_snapshot")
        if not isinstance(snapshot, dict):
            raise GateError(
                "legacy protocol requires runtime_python_snapshot provenance"
            )
        protocol_runtime = normalize_sha(
            snapshot.get("digest"),
            "protocol.runtime_python_snapshot.digest",
        )
        if protocol_runtime != runtime_sha:
            raise GateError(
                "protocol runtime Python digest differs from the run runtime"
            )
        if snapshot.get("package_is_symlink") is not False:
            raise GateError(
                "protocol runtime_python_snapshot.package_is_symlink "
                "must be false"
            )
        for key, expected in (
            ("repeat", 5),
            ("timing_processes", 0),
            ("memory_repeat", 3),
        ):
            require_protocol_integer(
                protocol_payload, key, expected, "protocol"
            )
        if isinstance(warmup_value, bool) or integer(
            warmup_value, "protocol.warmup"
        ) != 1:
            raise GateError("protocol.warmup must be 1")
        for key, expected in (
            ("first_use_calls_per_process", 1),
            ("measured_call_ordinal", 3),
            ("measured_calls_per_process", 1),
        ):
            require_protocol_integer(
                protocol_payload, key, expected, "protocol"
            )
        if protocol_payload.get("aggregate") != "arithmetic_mean":
            raise GateError("protocol.aggregate must be 'arithmetic_mean'")

    keys = (
        "protocol_name",
        "aggregate",
        "repeat",
        "timing_processes",
        "warmup",
        "memory_repeat",
        "first_use_calls_per_process",
        "extra_untimed_warmups_per_process",
        "measured_call_ordinal",
        "measured_calls_per_process",
        "candidate_sha256",
        "dataset",
    )
    measurement_protocol = {
        key: protocol_payload.get(key)
        for key in keys
        if key in protocol_payload
    }
    if warmup_value is not None:
        normalized_warmup = integer(
            warmup_value, "protocol.warmup"
        )
        measurement_protocol["warmup"] = normalized_warmup
        measurement_protocol[
            "extra_untimed_warmups_per_process"
        ] = normalized_warmup
    snapshot = protocol_payload.get("runtime_python_snapshot")
    if isinstance(snapshot, dict) and snapshot.get("digest"):
        measurement_protocol["runtime_python_snapshot_sha256"] = normalize_sha(
            snapshot.get("digest"),
            "protocol.runtime_python_snapshot.digest",
        )
        measurement_protocol["runtime_package_is_symlink"] = snapshot.get(
            "package_is_symlink"
        )
    measurement_protocol["independent_process_samples"] = (
        5 if measurement_kind == "timing" else 3
    )
    if measurement_kind == "memory":
        measurement_protocol["memory_measurement_window"] = (
            ANCHOR_MEMORY_WINDOW
        )
    elif measurement_kind != "timing":
        raise GateError(f"invalid anchor measurement kind: {measurement_kind!r}")

    return {
        **common,
        "anchor_metadata_path": str(protocol_path.resolve()),
        "anchor_metadata_sha256": sha256(protocol_path),
        "anchor_metadata_schema": "protocol_json_legacy",
        "measurement_protocol": measurement_protocol,
        "protocol_evidence": (
            "five worker JSON records with one measured sample each; the "
            "companion protocol.json records the warmup count and "
            "measured-call ordinal"
        ),
    }


def validate_anchor_raw_runtime_identity(
    record: dict[str, object],
    candidate_sha: str,
    runtime_sha: str,
    runtime_root: object,
    label: str,
) -> None:
    """Bind an accepted V10 raw record to its directory-level runtime."""

    expected_root = normalized_recorded_path(
        runtime_root, "anchor runtime resolved root"
    )
    identity = record.get("runtime_identity")
    if not isinstance(identity, dict):
        raise GateError(f"{label}: runtime_identity is missing")
    observed_candidate = normalize_sha(
        identity.get("native_sha256"),
        f"{label}.runtime_identity.native_sha256",
    )
    observed_runtime = normalize_sha(
        identity.get("runtime_python_digest"),
        f"{label}.runtime_identity.runtime_python_digest",
    )
    observed_root = normalized_recorded_path(
        identity.get("resolved_root"),
        f"{label}.runtime_identity.resolved_root",
    )
    if observed_candidate != candidate_sha:
        raise GateError(f"{label}: raw native SHA differs from run_metadata")
    if observed_runtime != runtime_sha:
        raise GateError(
            f"{label}: raw runtime Python digest differs from run_metadata"
        )
    if observed_root != expected_root:
        raise GateError(f"{label}: raw runtime root differs from run_metadata")

    requested = record.get("requested_runtime_provenance")
    if not isinstance(requested, dict):
        raise GateError(f"{label}: requested_runtime_provenance is missing")
    requested_snapshot = requested.get("runtime_python_snapshot")
    if not isinstance(requested_snapshot, dict):
        raise GateError(
            f"{label}: requested runtime Python snapshot is missing"
        )
    if normalize_sha(
        requested.get("native_sha256"),
        f"{label}.requested_runtime_provenance.native_sha256",
    ) != candidate_sha:
        raise GateError(
            f"{label}: requested runtime native SHA differs from run_metadata"
        )
    if normalize_sha(
        requested_snapshot.get("digest"),
        f"{label}.requested_runtime_provenance.runtime_python_snapshot.digest",
    ) != runtime_sha:
        raise GateError(
            f"{label}: requested runtime Python digest differs from run_metadata"
        )
    if requested_snapshot.get("package_is_symlink") is not False:
        raise GateError(
            f"{label}: requested runtime package_is_symlink must be false"
        )
    if normalized_recorded_path(
        requested.get("resolved_root"),
        f"{label}.requested_runtime_provenance.resolved_root",
    ) != expected_root:
        raise GateError(
            f"{label}: requested runtime root differs from run_metadata"
        )

    loaded = record.get("runtime_provenance")
    if not isinstance(loaded, dict):
        raise GateError(f"{label}: runtime_provenance is missing")
    if normalize_sha(
        loaded.get("native_sha256"),
        f"{label}.runtime_provenance.native_sha256",
    ) != candidate_sha:
        raise GateError(
            f"{label}: loaded runtime native SHA differs from run_metadata"
        )
    if normalized_recorded_path(
        loaded.get("resolved_root"),
        f"{label}.runtime_provenance.resolved_root",
    ) != expected_root:
        raise GateError(f"{label}: loaded runtime root differs from run_metadata")


def locate_main_csv(result_dir: Path, name: str, phase: str = "timing") -> Path:
    direct = result_dir / name
    if direct.is_file():
        return direct
    phased = result_dir / "measurement_passes" / phase / name
    if phased.is_file():
        return phased
    raise GateError(f"missing {name}")


def sample_stats(
    values: list[float], field: str, expected_samples: int = 5
) -> tuple[float, float]:
    if len(values) != expected_samples:
        raise GateError(
            f"{field} has {len(values)} samples, expected {expected_samples}"
        )
    return statistics.mean(values), statistics.stdev(values)


def summarize_timing_samples(
    values: list[float],
    field: str,
    *,
    expected_samples: int = 5,
    max_over_median_limit: float = DEFAULT_MAX_OVER_MEDIAN_LIMIT,
    median_over_min_limit: float = DEFAULT_MEDIAN_OVER_MIN_LIMIT,
) -> TimingStats:
    if len(values) != expected_samples:
        raise GateError(
            f"{field} has {len(values)} samples, expected {expected_samples}"
        )
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise GateError(f"{field} contains a negative or non-finite sample")
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    coefficient_of_variation = (
        stdev / mean if mean > 0 else (0.0 if stdev == 0 else math.inf)
    )
    median = statistics.median(values)
    maximum = max(values)
    minimum = min(values)
    max_over_median = (
        maximum / median
        if median > 0
        else (1.0 if maximum == 0 else math.inf)
    )
    median_over_minimum = (
        median / minimum
        if minimum > 0
        else (1.0 if maximum == 0 else math.inf)
    )
    stable = (
        max_over_median <= max_over_median_limit
        and median_over_minimum <= median_over_min_limit
    )
    return TimingStats(
        samples=tuple(values),
        minimum=minimum,
        mean=mean,
        stdev=stdev,
        maximum=maximum,
        median=median,
        coefficient_of_variation=coefficient_of_variation,
        max_over_median=max_over_median,
        median_over_minimum=median_over_minimum,
        variance_policy=DEFAULT_VARIANCE_POLICY,
        stability_status="pass" if stable else "fail",
    )


def main_candidates(
    result_dir: Path,
    fallback_sha: str | None,
    *,
    strict_provenance: bool = False,
    max_over_median_limit: float = DEFAULT_MAX_OVER_MEDIAN_LIMIT,
    median_over_min_limit: float = DEFAULT_MEDIAN_OVER_MIN_LIMIT,
) -> tuple[list[Candidate], list[dict[str, object]], dict[str, object]]:
    result_dir = result_dir.resolve()
    candidate_sha, sha_source = candidate_sha_from_result_dir(
        result_dir, fallback_sha
    )
    if strict_provenance and set(sha_source.split(";")) == {"CLI"}:
        raise GateError(
            f"{result_dir}: unified mode requires run-recorded candidate SHA evidence"
        )
    runtime_sha, runtime_source, package_is_symlink = (
        runtime_python_snapshot_from_result_dir(
            result_dir, required=strict_provenance
        )
    )
    samples_path = locate_main_csv(result_dir, "results_samples.csv")
    aggregates_path = locate_main_csv(result_dir, "results_long.csv")
    _sample_fields, samples = read_csv(samples_path)
    _long_fields, aggregates = read_csv(aggregates_path)

    sample_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    discovered: set[tuple[str, str]] = set()
    for row in samples:
        if row.get("baseline") != "EGGPU" or row.get("metric") not in METRICS:
            continue
        key = (row.get("dataset", ""), row.get("function", ""), row["metric"])
        discovered.add(key[:2])
        sample_groups.setdefault(key, []).append(row)

    aggregate_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in aggregates:
        if row.get("baseline") != "EGGPU" or row.get("metric") not in METRICS:
            continue
        key = (row.get("dataset", ""), row.get("function", ""), row["metric"])
        aggregate_groups.setdefault(key, []).append(row)

    validation_by_key: dict[tuple[str, str], list[str]] = {}
    validation_path = result_dir / "correctness_validation.csv"
    if validation_path.is_file():
        _fields, validation_rows = read_csv(validation_path)
        for row in validation_rows:
            if row.get("baseline") == "EGGPU":
                validation_by_key.setdefault(
                    (row.get("dataset", ""), row.get("function", "")), []
                ).append(str(row.get("validation_status", "")).strip().lower())

    accepted: list[Candidate] = []
    rejected: list[dict[str, object]] = []
    for dataset, function in sorted(discovered):
        try:
            metric_stats: dict[str, TimingStats] = {}
            for metric in METRICS:
                key = (dataset, function, metric)
                group = sample_groups.get(key, [])
                if len(group) != 5:
                    raise GateError(
                        f"{metric}: {len(group)} raw samples, expected 5"
                    )
                if any(row.get("status") != "ok" for row in group):
                    raise GateError(f"{metric}: a raw sample is not ok")
                if any(
                    row.get("measurement_phase") not in {"", "timing", None}
                    for row in group
                ):
                    raise GateError(f"{metric}: a sample is not from timing phase")
                if any(row.get("unit") not in {"", "s", None} for row in group):
                    raise GateError(f"{metric}: a sample does not use seconds")
                indices = sorted(integer(row.get("sample_index"), "sample_index") for row in group)
                if indices != [1, 2, 3, 4, 5]:
                    raise GateError(f"{metric}: sample indices are {indices}")
                values = [
                    finite_number(
                        row.get("value", row.get("seconds")),
                        f"{metric}.sample_value",
                    )
                    for row in group
                ]
                if any(value < 0 for value in values):
                    raise GateError(f"{metric}: a sample is negative")
                if any(not str(row.get("correctness", "")).strip() for row in group):
                    raise GateError(f"{metric}: correctness digest is missing")
                timing_stats = summarize_timing_samples(
                    values,
                    metric,
                    max_over_median_limit=max_over_median_limit,
                    median_over_min_limit=median_over_min_limit,
                )

                aggregate = aggregate_groups.get(key, [])
                if len(aggregate) != 1:
                    raise GateError(
                        f"{metric}: {len(aggregate)} aggregate rows, expected 1"
                    )
                summary = aggregate[0]
                if summary.get("status") != "ok":
                    raise GateError(f"{metric}: aggregate status is not ok")
                for count_field in ("sample_count", "n_total", "n_valid"):
                    if summary.get(count_field) not in {"", None}:
                        if integer(summary[count_field], count_field) != 5:
                            raise GateError(f"{metric}: {count_field} is not 5")
                if summary.get("publishable") not in {"", None} and not truthy(
                    summary["publishable"]
                ):
                    raise GateError(f"{metric}: aggregate is not publishable")
                aggregation = str(summary.get("aggregation", "")).strip()
                if aggregation and aggregation != "arithmetic_mean":
                    raise GateError(
                        f"{metric}: aggregation is {aggregation!r}, not arithmetic_mean"
                    )
                require_close(
                    summary.get("mean_seconds", summary.get("value")),
                    timing_stats.mean,
                    f"{metric}.mean",
                )
                require_close(
                    summary.get("std_seconds", summary.get("std_value")),
                    timing_stats.stdev,
                    f"{metric}.stdev",
                )
                summary_min = summary.get(
                    "min_seconds", summary.get("min_value")
                )
                summary_max = summary.get(
                    "max_seconds", summary.get("max_value")
                )
                summary_cv = summary.get("cv")
                if strict_provenance and summary_min in {"", None}:
                    raise GateError(f"{metric}: aggregate minimum is missing")
                if strict_provenance and summary_max in {"", None}:
                    raise GateError(f"{metric}: aggregate maximum is missing")
                if strict_provenance and summary_cv in {"", None}:
                    raise GateError(f"{metric}: aggregate CV is missing")
                if summary_min not in {"", None}:
                    require_close(
                        summary_min,
                        timing_stats.minimum,
                        f"{metric}.minimum",
                    )
                if summary_max not in {"", None}:
                    require_close(
                        summary_max,
                        timing_stats.maximum,
                        f"{metric}.maximum",
                    )
                if summary_cv not in {"", None}:
                    require_close(
                        summary_cv,
                        timing_stats.coefficient_of_variation,
                        f"{metric}.cv",
                    )
                metric_stats[metric] = timing_stats

            statuses = validation_by_key.get((dataset, function), [])
            if any(
                status in {"fail", "semantic_mismatch", "mismatch", "error"}
                for status in statuses
            ):
                raise GateError(
                    f"correctness validation contains a failing status: {statuses}"
                )
            accepted.append(
                Candidate(
                    dataset=dataset,
                    function=function,
                    source_kind="main",
                    result_source=str(result_dir),
                    candidate_sha256=candidate_sha,
                    metrics=metric_stats,
                    runtime_python_sha256=runtime_sha,
                )
            )
        except GateError as exc:
            rejected.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "source_kind": "main",
                    "result_source": str(result_dir),
                    "reason": str(exc),
                }
            )
    run_metadata_path = result_dir / "run_metadata.json"
    measurement_protocol: dict[str, object] = {}
    run_metadata_sha256 = ""
    if run_metadata_path.is_file():
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        benchmark_args = run_metadata.get("benchmark_args", {})
        measurement_protocol = {
            "repeat": benchmark_args.get("repeat"),
            "warmup": benchmark_args.get("warmup"),
            "easygraph_warmup": benchmark_args.get("easygraph_warmup"),
            "eggpu_execution_protocol": benchmark_args.get(
                "eggpu_execution_protocol"
            ),
            "measurement_mode": benchmark_args.get("measurement_mode"),
            "measured_call_ordinal": (
                int(benchmark_args["warmup"]) + 1
                if benchmark_args.get("warmup") is not None
                else None
            ),
            "aggregation": "arithmetic_mean",
            "paper_estimator": PAPER_TIMING_ESTIMATOR,
            "independent_process_samples": 5,
            "stability_metric": "e2e",
            "variance_policy": DEFAULT_VARIANCE_POLICY,
            "max_over_median_limit": max_over_median_limit,
            "median_over_min_limit": median_over_min_limit,
        }
        run_metadata_sha256 = sha256(run_metadata_path)

    provenance = {
        "result_dir": str(result_dir),
        "source_kind": "main",
        "measurement_kind": "timing",
        "candidate_sha256": candidate_sha,
        "candidate_sha_source": sha_source,
        "runtime_python_sha256": runtime_sha,
        "runtime_python_sha_source": runtime_source,
        "runtime_package_is_symlink": package_is_symlink,
        "run_metadata_path": (
            str(run_metadata_path.resolve()) if run_metadata_path.is_file() else ""
        ),
        "run_metadata_sha256": run_metadata_sha256,
        "measurement_protocol": measurement_protocol,
        "results_samples_path": str(samples_path.resolve()),
        "results_samples_sha256": sha256(samples_path),
        "results_long_path": str(aggregates_path.resolve()),
        "results_long_sha256": sha256(aggregates_path),
    }
    return accepted, rejected, provenance


def validation_pass(value: object) -> bool:
    if isinstance(value, dict):
        return value.get("status") == "pass" and not value.get("failures")
    return str(value).strip().lower() == "pass"


def json_stats(
    payload: dict,
    key: str,
    field_prefix: str,
    *,
    max_over_median_limit: float = DEFAULT_MAX_OVER_MEDIAN_LIMIT,
    median_over_min_limit: float = DEFAULT_MEDIAN_OVER_MIN_LIMIT,
) -> TimingStats:
    stats = payload.get(key)
    if not isinstance(stats, dict):
        raise GateError(f"{field_prefix}: missing {key} statistics")
    values = [
        finite_number(value, f"{field_prefix}.{key}.sample")
        for value in stats.get("samples", [])
    ]
    timing_stats = summarize_timing_samples(
        values,
        f"{field_prefix}.{key}",
        max_over_median_limit=max_over_median_limit,
        median_over_min_limit=median_over_min_limit,
    )
    require_close(
        stats.get("mean"), timing_stats.mean, f"{field_prefix}.{key}.mean"
    )
    require_close(
        stats.get("stdev"),
        timing_stats.stdev,
        f"{field_prefix}.{key}.stdev",
    )
    if stats.get("best") not in {"", None}:
        require_close(
            stats.get("best"),
            timing_stats.minimum,
            f"{field_prefix}.{key}.best",
        )
    if stats.get("cv") not in {"", None}:
        require_close(
            stats.get("cv"),
            timing_stats.coefficient_of_variation,
            f"{field_prefix}.{key}.cv",
        )
    return timing_stats


def anchor_candidates(
    result_dir: Path,
    fallback_sha: str | None,
    *,
    strict_provenance: bool = False,
    max_over_median_limit: float = DEFAULT_MAX_OVER_MEDIAN_LIMIT,
    median_over_min_limit: float = DEFAULT_MEDIAN_OVER_MIN_LIMIT,
) -> tuple[list[Candidate], list[dict[str, object]], dict[str, object]]:
    result_dir = result_dir.resolve()
    candidate_sha, sha_source = candidate_sha_from_result_dir(
        result_dir, fallback_sha
    )
    if strict_provenance and set(sha_source.split(";")) == {"CLI"}:
        raise GateError(
            f"{result_dir}: unified mode requires run-recorded candidate SHA evidence"
        )
    runtime_sha, runtime_source, package_is_symlink = (
        runtime_python_snapshot_from_result_dir(
            result_dir, required=strict_provenance
        )
    )
    anchor_evidence = anchor_provenance_evidence(
        result_dir,
        candidate_sha,
        runtime_sha,
        strict_provenance=strict_provenance,
        measurement_kind="timing",
    )
    anchor_evidence["measurement_protocol"].update(
        {
            "paper_estimator": PAPER_TIMING_ESTIMATOR,
            "stability_metric": "e2e",
            "variance_policy": DEFAULT_VARIANCE_POLICY,
            "max_over_median_limit": max_over_median_limit,
            "median_over_min_limit": median_over_min_limit,
        }
    )
    bind_v10_raw_runtime = (
        anchor_evidence.get("anchor_metadata_schema")
        == "run_metadata_v10"
    )
    summary_path = result_dir / "scaling_all.csv"
    if not summary_path.is_file():
        raise GateError("missing scaling_all.csv")
    _fields, rows = read_csv(summary_path)
    timing_rows = [
        row
        for row in rows
        if row.get("measurement") == "timing"
        and row.get("dataset")
        and row.get("function")
    ]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in timing_rows:
        grouped.setdefault((row["dataset"], row["function"]), []).append(row)

    accepted: list[Candidate] = []
    rejected: list[dict[str, object]] = []
    aggregate_hashes: dict[str, str] = {}
    for (dataset, function), group in sorted(grouped.items()):
        try:
            if len(group) != 1:
                raise GateError(f"{len(group)} timing summary rows, expected 1")
            row = group[0]
            if row.get("status") != "ok":
                raise GateError(f"summary status is {row.get('status')!r}")
            if row.get("result_validation") != "pass":
                raise GateError("summary result_validation is not pass")
            if integer(
                row.get("timing_process_samples"), "timing_process_samples"
            ) != 5:
                raise GateError("timing_process_samples is not 5")

            aggregate_path = (
                result_dir / "raw" / f"{dataset}_{function}_timing.json"
            )
            if not aggregate_path.is_file():
                raise GateError(f"missing aggregate {aggregate_path.name}")
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            if aggregate.get("status") != "ok":
                raise GateError("raw aggregate status is not ok")
            if not validation_pass(aggregate.get("result_validation")):
                raise GateError("raw aggregate result validation is not pass")
            if integer(
                aggregate.get("timing_process_samples"),
                "aggregate.timing_process_samples",
            ) != 5:
                raise GateError("aggregate timing_process_samples is not 5")
            if bind_v10_raw_runtime:
                validate_anchor_raw_runtime_identity(
                    aggregate,
                    candidate_sha,
                    runtime_sha,
                    anchor_evidence.get("runtime_resolved_root"),
                    f"{dataset}/{function}.aggregate",
                )

            build_stats = json_stats(
                aggregate,
                "load",
                f"{dataset}/{function}",
                max_over_median_limit=max_over_median_limit,
                median_over_min_limit=median_over_min_limit,
            )
            e2e_stats = json_stats(
                aggregate,
                "steady_e2e",
                f"{dataset}/{function}",
                max_over_median_limit=max_over_median_limit,
                median_over_min_limit=median_over_min_limit,
            )
            kernel_stats = json_stats(
                aggregate,
                "steady_kernel",
                f"{dataset}/{function}",
                max_over_median_limit=max_over_median_limit,
                median_over_min_limit=median_over_min_limit,
            )
            json_stats(aggregate, "first_use_e2e", f"{dataset}/{function}")

            require_close(
                row.get("load_seconds"),
                build_stats.mean,
                "load_seconds",
            )
            require_close(
                row.get("load_stdev_seconds"),
                build_stats.stdev,
                "load_stdev_seconds",
            )
            require_close(
                row.get("steady_e2e_mean"),
                e2e_stats.mean,
                "steady_e2e_mean",
            )
            require_close(
                row.get("steady_e2e_stdev"),
                e2e_stats.stdev,
                "steady_e2e_stdev",
            )
            require_close(
                row.get("steady_kernel_mean"),
                kernel_stats.mean,
                "steady_kernel_mean",
            )
            require_close(
                row.get("steady_kernel_stdev"),
                kernel_stats.stdev,
                "steady_kernel_stdev",
            )

            worker_results = []
            for index in range(1, 6):
                worker_path = (
                    result_dir
                    / "raw"
                    / f"{dataset}_{function}_timing_{index}.json"
                )
                if not worker_path.is_file():
                    raise GateError(f"missing worker {worker_path.name}")
                worker = json.loads(worker_path.read_text(encoding="utf-8"))
                if worker.get("status") != "ok":
                    raise GateError(f"worker {index} status is not ok")
                if not validation_pass(worker.get("result_validation")):
                    raise GateError(f"worker {index} validation is not pass")
                if bind_v10_raw_runtime:
                    validate_anchor_raw_runtime_identity(
                        worker,
                        candidate_sha,
                        runtime_sha,
                        anchor_evidence.get("runtime_resolved_root"),
                        f"{dataset}/{function}.worker{index}",
                    )
                for metric_key in ("steady_e2e", "steady_kernel"):
                    samples = worker.get(metric_key, {}).get("samples", [])
                    if len(samples) != 1:
                        raise GateError(
                            f"worker {index} {metric_key} has "
                            f"{len(samples)} measured samples, expected 1"
                        )
                worker_results.append(worker.get("result"))
            if any(result is None for result in worker_results):
                raise GateError("a worker result digest is missing")

            aggregate_hashes[str(aggregate_path)] = sha256(aggregate_path)
            accepted.append(
                Candidate(
                    dataset=dataset,
                    function=function,
                    source_kind="anchor",
                    result_source=str(result_dir),
                    candidate_sha256=candidate_sha,
                    metrics={
                        "build": build_stats,
                        "e2e": e2e_stats,
                        "kernel": kernel_stats,
                    },
                    runtime_python_sha256=runtime_sha,
                )
            )
        except GateError as exc:
            rejected.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "source_kind": "anchor",
                    "result_source": str(result_dir),
                    "reason": str(exc),
                }
            )
    provenance = {
        "result_dir": str(result_dir),
        "source_kind": "anchor",
        "measurement_kind": "timing",
        "candidate_sha256": candidate_sha,
        "candidate_sha_source": sha_source,
        "runtime_python_sha256": runtime_sha,
        "runtime_python_sha_source": runtime_source,
        "runtime_package_is_symlink": package_is_symlink,
        **anchor_evidence,
        "scaling_all_sha256": sha256(summary_path),
        "accepted_aggregate_sha256": aggregate_hashes,
    }
    return accepted, rejected, provenance


def choose_memory_metric(
    rows: list[dict[str, str]],
    dataset: str,
    function: str,
    candidates: tuple[str, ...],
    *,
    measurement_window: str,
) -> tuple[str, list[dict[str, str]]]:
    for metric in candidates:
        group = [
            row
            for row in rows
            if row.get("baseline") == "EGGPU"
            and row.get("dataset") == dataset
            and row.get("function") == function
            and row.get("metric") == metric
            and row.get("measurement_window") == measurement_window
            and row.get("measurement_phase") in {"", "memory", None}
        ]
        if group:
            return metric, group
    raise GateError(
        f"missing {candidates} in measurement window {measurement_window!r}"
    )


def validate_main_memory_metric(
    sample_rows: list[dict[str, str]],
    aggregate_rows: list[dict[str, str]],
    dataset: str,
    function: str,
    metric_candidates: tuple[str, ...],
) -> tuple[float, float, str]:
    metric, group = choose_memory_metric(
        sample_rows,
        dataset,
        function,
        metric_candidates,
        measurement_window=MAIN_MEMORY_WINDOW,
    )
    if len(group) != 3:
        raise GateError(f"{metric}: {len(group)} raw samples, expected 3")
    if any(row.get("status") != "ok" for row in group):
        raise GateError(f"{metric}: a raw memory sample is not ok")
    if any(row.get("unit") != "MiB" for row in group):
        raise GateError(f"{metric}: a raw memory sample does not use MiB")
    if any(row.get("measurement_phase") != "memory" for row in group):
        raise GateError(f"{metric}: a raw sample is not from memory phase")
    indices = sorted(integer(row.get("sample_index"), "sample_index") for row in group)
    if indices != [1, 2, 3]:
        raise GateError(f"{metric}: sample indices are {indices}")
    for row in group:
        if integer(row.get("sample_count"), "sample_count") != 3:
            raise GateError(f"{metric}: a raw sample does not declare sample_count=3")
        if not str(row.get("correctness", "")).strip():
            raise GateError(f"{metric}: correctness digest is missing")
    values = [
        finite_number(
            row.get("value", row.get("seconds")),
            f"{metric}.sample_value",
        )
        for row in group
    ]
    if any(value <= 0 for value in values):
        raise GateError(f"{metric}: a process-lifetime peak is non-positive")
    mean, stdev = sample_stats(values, metric, expected_samples=3)

    summaries = [
        row
        for row in aggregate_rows
        if row.get("baseline") == "EGGPU"
        and row.get("dataset") == dataset
        and row.get("function") == function
        and row.get("metric") == metric
        and row.get("measurement_window") == MAIN_MEMORY_WINDOW
        and row.get("measurement_phase") in {"", "memory", None}
    ]
    if len(summaries) != 1:
        raise GateError(
            f"{metric}: {len(summaries)} memory aggregate rows, expected 1"
        )
    summary = summaries[0]
    if summary.get("status") != "ok":
        raise GateError(f"{metric}: aggregate status is not ok")
    for count_field in ("sample_count", "n_total", "n_valid"):
        if summary.get(count_field) not in {"", None}:
            if integer(summary[count_field], count_field) != 3:
                raise GateError(f"{metric}: {count_field} is not 3")
    if summary.get("publishable") not in {"", None} and not truthy(
        summary["publishable"]
    ):
        raise GateError(f"{metric}: aggregate is not publishable")
    aggregation = str(summary.get("aggregation", "")).strip()
    if aggregation and aggregation != "arithmetic_mean":
        raise GateError(
            f"{metric}: aggregation is {aggregation!r}, not arithmetic_mean"
        )
    require_close(
        summary.get("mean_value", summary.get("value")),
        mean,
        f"{metric}.mean",
    )
    require_close(
        summary.get("std_value", summary.get("std_seconds")),
        stdev,
        f"{metric}.stdev",
    )
    return mean, stdev, metric


def main_memory_candidates(
    result_dir: Path,
    fallback_sha: str | None,
    *,
    strict_provenance: bool = False,
) -> tuple[list[MemoryCandidate], list[dict[str, object]], dict[str, object]]:
    """Parse three isolated-process memory samples per regular-matrix cell."""

    result_dir = result_dir.resolve()
    candidate_sha, sha_source = candidate_sha_from_result_dir(
        result_dir, fallback_sha
    )
    if strict_provenance and set(sha_source.split(";")) == {"CLI"}:
        raise GateError(
            f"{result_dir}: unified mode requires run-recorded candidate SHA evidence"
        )
    runtime_sha, runtime_source, package_is_symlink = (
        runtime_python_snapshot_from_result_dir(
            result_dir, required=strict_provenance
        )
    )
    samples_path = locate_main_csv(result_dir, "results_samples.csv", "memory")
    aggregates_path = locate_main_csv(result_dir, "results_long.csv", "memory")
    _sample_fields, samples = read_csv(samples_path)
    _aggregate_fields, aggregates = read_csv(aggregates_path)
    discovered = {
        (row.get("dataset", ""), row.get("function", ""))
        for row in samples
        if row.get("baseline") == "EGGPU"
        and row.get("measurement_window") == MAIN_MEMORY_WINDOW
        and row.get("metric") in {
            metric for choices in MEMORY_METRICS.values() for metric in choices
        }
    }

    validation_by_key: dict[tuple[str, str], list[str]] = {}
    validation_path = result_dir / "correctness_validation.csv"
    if validation_path.is_file():
        _fields, validation_rows = read_csv(validation_path)
        for row in validation_rows:
            if row.get("baseline") == "EGGPU":
                validation_by_key.setdefault(
                    (row.get("dataset", ""), row.get("function", "")), []
                ).append(str(row.get("validation_status", "")).strip().lower())

    accepted: list[MemoryCandidate] = []
    rejected: list[dict[str, object]] = []
    selected_metrics: dict[str, dict[str, str]] = {}
    for dataset, function in sorted(discovered):
        try:
            gpu_mean, gpu_std, gpu_metric = validate_main_memory_metric(
                samples,
                aggregates,
                dataset,
                function,
                MEMORY_METRICS["gpu"],
            )
            host_mean, host_std, host_metric = validate_main_memory_metric(
                samples,
                aggregates,
                dataset,
                function,
                MEMORY_METRICS["host"],
            )
            statuses = validation_by_key.get((dataset, function), [])
            if any(
                status in {"fail", "semantic_mismatch", "mismatch", "error"}
                for status in statuses
            ):
                raise GateError(
                    f"correctness validation contains a failing status: {statuses}"
                )
            accepted.append(
                MemoryCandidate(
                    dataset=dataset,
                    function=function,
                    source_kind="main",
                    result_source=str(result_dir),
                    candidate_sha256=candidate_sha,
                    runtime_python_sha256=runtime_sha,
                    gpu_peak_mb=(gpu_mean, gpu_std),
                    host_rss_peak_mb=(host_mean, host_std),
                    measurement_window=MAIN_MEMORY_WINDOW,
                )
            )
            selected_metrics[f"{dataset}/{function}"] = {
                "gpu": gpu_metric,
                "host": host_metric,
            }
        except GateError as exc:
            rejected.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "source_kind": "main",
                    "measurement_kind": "memory",
                    "result_source": str(result_dir),
                    "reason": str(exc),
                }
            )

    run_metadata_path = result_dir / "run_metadata.json"
    measurement_protocol: dict[str, object] = {}
    run_metadata_sha256 = ""
    if run_metadata_path.is_file():
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        benchmark_args = run_metadata.get("benchmark_args", {})
        measurement_protocol = {
            "repeat": benchmark_args.get("repeat"),
            "warmup": benchmark_args.get("warmup"),
            "easygraph_warmup": benchmark_args.get("easygraph_warmup"),
            "eggpu_execution_protocol": benchmark_args.get(
                "eggpu_execution_protocol"
            ),
            "measurement_mode": benchmark_args.get("measurement_mode"),
            "aggregation": "arithmetic_mean",
            "independent_process_samples": 3,
            "measurement_window": MAIN_MEMORY_WINDOW,
        }
        run_metadata_sha256 = sha256(run_metadata_path)
    provenance = {
        "result_dir": str(result_dir),
        "source_kind": "main",
        "measurement_kind": "memory",
        "candidate_sha256": candidate_sha,
        "candidate_sha_source": sha_source,
        "runtime_python_sha256": runtime_sha,
        "runtime_python_sha_source": runtime_source,
        "runtime_package_is_symlink": package_is_symlink,
        "run_metadata_path": (
            str(run_metadata_path.resolve()) if run_metadata_path.is_file() else ""
        ),
        "run_metadata_sha256": run_metadata_sha256,
        "measurement_protocol": measurement_protocol,
        "results_samples_sha256": sha256(samples_path),
        "results_long_sha256": sha256(aggregates_path),
        "selected_metrics": selected_metrics,
    }
    return accepted, rejected, provenance


def anchor_memory_candidates(
    result_dir: Path,
    fallback_sha: str | None,
    *,
    strict_provenance: bool = False,
) -> tuple[list[MemoryCandidate], list[dict[str, object]], dict[str, object]]:
    """Parse three validated scale-runner memory workers per anchor cell."""

    result_dir = result_dir.resolve()
    candidate_sha, sha_source = candidate_sha_from_result_dir(
        result_dir, fallback_sha
    )
    if strict_provenance and set(sha_source.split(";")) == {"CLI"}:
        raise GateError(
            f"{result_dir}: unified mode requires run-recorded candidate SHA evidence"
        )
    runtime_sha, runtime_source, package_is_symlink = (
        runtime_python_snapshot_from_result_dir(
            result_dir, required=strict_provenance
        )
    )
    anchor_evidence = anchor_provenance_evidence(
        result_dir,
        candidate_sha,
        runtime_sha,
        strict_provenance=strict_provenance,
        measurement_kind="memory",
    )
    bind_v10_raw_runtime = (
        anchor_evidence.get("anchor_metadata_schema")
        == "run_metadata_v10"
    )
    summary_path = result_dir / "scaling_all.csv"
    if not summary_path.is_file():
        raise GateError("missing scaling_all.csv")
    _fields, rows = read_csv(summary_path)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if (
            row.get("measurement") == "memory"
            and row.get("dataset")
            and row.get("function")
        ):
            grouped.setdefault((row["dataset"], row["function"]), []).append(row)

    accepted: list[MemoryCandidate] = []
    rejected: list[dict[str, object]] = []
    worker_hashes: dict[str, str] = {}
    for (dataset, function), group in sorted(grouped.items()):
        try:
            if len(group) != 3:
                raise GateError(f"{len(group)} memory rows, expected 3")
            if any(row.get("status") != "ok" for row in group):
                raise GateError("a memory summary row is not ok")
            if any(row.get("result_validation") != "pass" for row in group):
                raise GateError("a memory summary validation is not pass")

            gpu_field = next(
                (
                    field
                    for field in ("gpu_proc_peak_mb", "gpu_proc_peak_delta_mb")
                    if all(str(row.get(field, "")).strip() for row in group)
                ),
                None,
            )
            host_field = next(
                (
                    field
                    for field in ("rss_peak_mb", "rss_peak_delta_mb")
                    if all(str(row.get(field, "")).strip() for row in group)
                ),
                None,
            )
            if gpu_field is None or host_field is None:
                raise GateError("memory summary lacks GPU or host process peaks")
            gpu_values = [
                finite_number(row[gpu_field], f"{gpu_field}.sample")
                for row in group
            ]
            host_values = [
                finite_number(row[host_field], f"{host_field}.sample")
                for row in group
            ]
            if any(value <= 0 for value in (*gpu_values, *host_values)):
                raise GateError("a process-lifetime memory peak is non-positive")

            for index, row in enumerate(group, start=1):
                worker_path = (
                    result_dir
                    / "raw"
                    / f"{dataset}_{function}_memory_{index}.json"
                )
                if not worker_path.is_file():
                    raise GateError(f"missing worker {worker_path.name}")
                worker = json.loads(worker_path.read_text(encoding="utf-8"))
                if worker.get("status") != "ok":
                    raise GateError(f"worker {index} status is not ok")
                if not validation_pass(worker.get("result_validation")):
                    raise GateError(f"worker {index} validation is not pass")
                if bind_v10_raw_runtime:
                    validate_anchor_raw_runtime_identity(
                        worker,
                        candidate_sha,
                        runtime_sha,
                        anchor_evidence.get("runtime_resolved_root"),
                        f"{dataset}/{function}.memory_worker{index}",
                    )
                memory = worker.get("memory")
                if not isinstance(memory, dict):
                    raise GateError(f"worker {index} memory payload is missing")
                if (
                    memory.get("memory_monitor_origin")
                    != "coordinator_process_child_tree"
                ):
                    raise GateError(
                        f"worker {index} memory monitor origin is invalid"
                    )
                if memory.get("measurement_window") != ANCHOR_MEMORY_WINDOW:
                    raise GateError(
                        f"worker {index} measurement window is invalid"
                    )
                for field in ("monitor_rss_samples", "monitor_gpu_proc_samples"):
                    if integer(memory.get(field), f"worker.{field}") < 1:
                        raise GateError(f"worker {index} {field} is zero")
                require_close(
                    memory.get(gpu_field),
                    gpu_values[index - 1],
                    f"worker.{gpu_field}",
                )
                worker_host_field = (
                    "rss_mb" if host_field == "rss_peak_mb" else host_field
                )
                require_close(
                    memory.get(worker_host_field),
                    host_values[index - 1],
                    f"worker.{worker_host_field}",
                )
                worker_hashes[str(worker_path.resolve())] = sha256(worker_path)

            accepted.append(
                MemoryCandidate(
                    dataset=dataset,
                    function=function,
                    source_kind="anchor",
                    result_source=str(result_dir),
                    candidate_sha256=candidate_sha,
                    runtime_python_sha256=runtime_sha,
                    gpu_peak_mb=sample_stats(
                        gpu_values, f"{dataset}/{function}.gpu", 3
                    ),
                    host_rss_peak_mb=sample_stats(
                        host_values, f"{dataset}/{function}.host", 3
                    ),
                    measurement_window=ANCHOR_MEMORY_WINDOW,
                )
            )
        except GateError as exc:
            rejected.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "source_kind": "anchor",
                    "measurement_kind": "memory",
                    "result_source": str(result_dir),
                    "reason": str(exc),
                }
            )

    provenance = {
        "result_dir": str(result_dir),
        "source_kind": "anchor",
        "measurement_kind": "memory",
        "candidate_sha256": candidate_sha,
        "candidate_sha_source": sha_source,
        "runtime_python_sha256": runtime_sha,
        "runtime_python_sha_source": runtime_source,
        "runtime_package_is_symlink": package_is_symlink,
        **anchor_evidence,
        "scaling_all_sha256": sha256(summary_path),
        "accepted_worker_sha256": worker_hashes,
    }
    return accepted, rejected, provenance


def validate_ledger(
    fields: list[str], rows: list[dict[str, str]]
) -> dict[tuple[str, str], int]:
    required = {
        "dataset",
        "function",
        "baseline",
        "execution_status",
        "validation_status",
    }
    missing = required - set(fields)
    if missing:
        raise ValueError(f"base ledger is missing columns: {sorted(missing)}")
    if len(rows) != EXPECTED_LEDGER_CELLS:
        raise ValueError(
            f"base ledger has {len(rows)} cells, expected {EXPECTED_LEDGER_CELLS}"
        )
    all_keys = [
        (row["dataset"], row["function"], row["baseline"]) for row in rows
    ]
    if len(set(all_keys)) != len(all_keys):
        raise ValueError("base ledger contains duplicate dataset/function/baseline keys")
    eggpu = [
        (index, row)
        for index, row in enumerate(rows)
        if row["baseline"] == "EGGPU"
    ]
    if len(eggpu) != EXPECTED_EGGPU_CELLS:
        raise ValueError(
            f"base ledger has {len(eggpu)} EGGPU cells, "
            f"expected {EXPECTED_EGGPU_CELLS}"
        )
    if any(row["execution_status"] != "ok" for _index, row in eggpu):
        raise ValueError("base ledger does not have EGGPU 208/208 ok")
    return {
        (row["dataset"], row["function"]): index for index, row in eggpu
    }


def comparison_row(
    old: dict[str, str], new: dict[str, str], candidate: Candidate
) -> dict[str, object]:
    result: dict[str, object] = {
        "dataset": candidate.dataset,
        "function": candidate.function,
        "source_kind": candidate.source_kind,
        "candidate_sha256": candidate.candidate_sha256,
        "runtime_python_snapshot_sha256": candidate.runtime_python_sha256,
        "old_result_source": old.get("result_source", ""),
        "new_result_source": new.get("result_source", ""),
        "old_sample_count": old.get("sample_count", ""),
        "new_sample_count": new.get("sample_count", ""),
    }
    for metric in METRICS:
        for suffix in (
            "paper_seconds",
            "raw_min_seconds",
            "raw_median_seconds",
            "raw_mean_seconds",
            "std_seconds",
            "raw_max_seconds",
            "coefficient_of_variation",
            "max_over_median",
            "median_over_minimum",
            "variance_policy",
            "stability_status",
            "estimator",
        ):
            field = f"{metric}_{suffix}"
            result[f"old_{field}"] = old.get(field, "")
            result[f"new_{field}"] = new.get(field, "")
    return result


def overlay(
    base_fields: list[str],
    base_rows: list[dict[str, str]],
    candidates: list[Candidate],
    expected_replacements: int | None,
) -> tuple[list[str], list[dict[str, str]], list[dict[str, object]], list[dict[str, object]]]:
    eggpu_index = validate_ledger(base_fields, base_rows)
    fields = list(base_fields)
    if "candidate_sha256" not in fields:
        fields.append("candidate_sha256")
    if "runtime_python_snapshot_sha256" not in fields:
        fields.append("runtime_python_snapshot_sha256")
    if "timing_result_source" not in fields:
        fields.append("timing_result_source")
    for metric in METRICS:
        for suffix in (
            "raw_min_seconds",
            "raw_median_seconds",
            "raw_max_seconds",
            "coefficient_of_variation",
            "max_over_median",
            "median_over_minimum",
            "variance_policy",
            "stability_status",
        ):
            field = f"{metric}_{suffix}"
            if field not in fields:
                fields.append(field)

    candidate_by_key: dict[tuple[str, str], Candidate] = {}
    duplicate_keys = []
    for candidate in candidates:
        if candidate.key in candidate_by_key:
            duplicate_keys.append(candidate.key)
        candidate_by_key[candidate.key] = candidate
    if duplicate_keys:
        raise ValueError(
            "accepted candidates contain duplicate dataset/function keys: "
            f"{sorted(set(duplicate_keys))}"
        )
    candidate_shas = sorted(
        {candidate.candidate_sha256 for candidate in candidates}
    )
    if len(candidate_shas) > 1:
        raise ValueError(f"accepted candidates use multiple binaries: {candidate_shas}")

    output_rows = [dict(row) for row in base_rows]
    comparisons = []
    rejected = []
    for key, candidate in sorted(candidate_by_key.items()):
        index = eggpu_index.get(key)
        if index is None:
            rejected.append(
                {
                    "dataset": candidate.dataset,
                    "function": candidate.function,
                    "source_kind": candidate.source_kind,
                    "result_source": candidate.result_source,
                    "reason": "candidate key is absent from base EGGPU ledger",
                }
            )
            continue
        old = dict(output_rows[index])
        if old.get("validation_status") not in {"pass", "sampled_pass"}:
            rejected.append(
                {
                    "dataset": candidate.dataset,
                    "function": candidate.function,
                    "source_kind": candidate.source_kind,
                    "result_source": candidate.result_source,
                    "reason": (
                        "base cell lacks a frozen pass validation; timing-only "
                        "overlay cannot establish semantics"
                    ),
                }
            )
            continue
        new = output_rows[index]
        new.update(
            {
                "execution_status": "ok",
                "failure_kind": "",
                # Timing reruns do not supersede the frozen semantic audit.
                "validation_status": old["validation_status"],
                "result_source": candidate.result_source,
                "timing_result_source": candidate.result_source,
                "sample_count": "5",
                "candidate_sha256": candidate.candidate_sha256,
                "runtime_python_snapshot_sha256": (
                    candidate.runtime_python_sha256
                ),
            }
        )
        for metric, timing_stats in candidate.metrics.items():
            new[f"{metric}_paper_seconds"] = str(timing_stats.minimum)
            new[f"{metric}_raw_min_seconds"] = str(timing_stats.minimum)
            new[f"{metric}_raw_median_seconds"] = str(
                timing_stats.median
            )
            new[f"{metric}_raw_mean_seconds"] = str(timing_stats.mean)
            new[f"{metric}_std_seconds"] = str(timing_stats.stdev)
            new[f"{metric}_raw_max_seconds"] = str(timing_stats.maximum)
            new[f"{metric}_coefficient_of_variation"] = str(
                timing_stats.coefficient_of_variation
            )
            new[f"{metric}_max_over_median"] = str(
                timing_stats.max_over_median
            )
            new[f"{metric}_median_over_minimum"] = str(
                timing_stats.median_over_minimum
            )
            new[f"{metric}_variance_policy"] = (
                timing_stats.variance_policy
            )
            new[f"{metric}_stability_status"] = (
                timing_stats.stability_status
            )
            new[f"{metric}_estimator"] = PAPER_TIMING_ESTIMATOR
        comparisons.append(comparison_row(old, new, candidate))

    if expected_replacements is not None and len(comparisons) != expected_replacements:
        raise ValueError(
            f"replacement count is {len(comparisons)}, "
            f"expected {expected_replacements}"
        )
    # Re-audit invariants after replacement.  Memory fields are never touched.
    validate_ledger(fields, output_rows)
    return fields, output_rows, comparisons, rejected


def validate_unified_candidate_identity(
    timing_candidates: list[Candidate],
    memory_candidates: list[MemoryCandidate],
) -> dict[str, str]:
    """Require complete, key-aligned timing and memory from one frozen runtime."""

    timing_by_key = {candidate.key: candidate for candidate in timing_candidates}
    memory_by_key = {candidate.key: candidate for candidate in memory_candidates}
    if len(timing_by_key) != len(timing_candidates):
        raise ValueError("timing candidates contain duplicate workload keys")
    if len(memory_by_key) != len(memory_candidates):
        raise ValueError("memory candidates contain duplicate workload keys")
    if len(timing_by_key) != EXPECTED_EGGPU_CELLS:
        raise ValueError(
            f"unified mode has {len(timing_by_key)} timing cells, "
            f"expected {EXPECTED_EGGPU_CELLS}"
        )
    if len(memory_by_key) != EXPECTED_EGGPU_CELLS:
        raise ValueError(
            f"unified mode has {len(memory_by_key)} memory cells, "
            f"expected {EXPECTED_EGGPU_CELLS}"
        )
    if set(timing_by_key) != set(memory_by_key):
        missing_memory = sorted(set(timing_by_key) - set(memory_by_key))
        missing_timing = sorted(set(memory_by_key) - set(timing_by_key))
        raise ValueError(
            "timing/memory workload sets differ: "
            f"missing_memory={missing_memory}, missing_timing={missing_timing}"
        )
    for measurement, candidate_set in (
        ("timing", timing_candidates),
        ("memory", memory_candidates),
    ):
        misplaced = sorted(
            candidate.key
            for candidate in candidate_set
            if (
                candidate.source_kind == "anchor"
            )
            != (candidate.dataset in ANCHOR_DATASETS)
        )
        if misplaced:
            raise ValueError(
                f"{measurement} candidates misclassify regular/anchor cells: "
                f"{misplaced}"
            )

    shas = {
        candidate.candidate_sha256
        for candidate in (*timing_candidates, *memory_candidates)
    }
    runtimes = {
        candidate.runtime_python_sha256
        for candidate in (*timing_candidates, *memory_candidates)
    }
    if len(shas) != 1:
        raise ValueError(
            f"unified timing/memory inputs use multiple candidate SHAs: {sorted(shas)}"
        )
    if "" in runtimes or len(runtimes) != 1:
        raise ValueError(
            "unified timing/memory inputs lack one common frozen Python runtime: "
            f"{sorted(runtimes)}"
        )
    for key, timing in timing_by_key.items():
        memory = memory_by_key[key]
        if (
            timing.candidate_sha256 != memory.candidate_sha256
            or timing.runtime_python_sha256 != memory.runtime_python_sha256
        ):
            raise ValueError(f"{key}: timing/memory candidate identity differs")
    return {
        "candidate_sha256": next(iter(shas)),
        "runtime_python_snapshot_sha256": next(iter(runtimes)),
    }


def memory_comparison_row(
    old: dict[str, str],
    new: dict[str, str],
    candidate: MemoryCandidate,
) -> dict[str, object]:
    result: dict[str, object] = {
        "dataset": candidate.dataset,
        "function": candidate.function,
        "source_kind": candidate.source_kind,
        "candidate_sha256": candidate.candidate_sha256,
        "runtime_python_snapshot_sha256": candidate.runtime_python_sha256,
        "old_memory_result_source": old.get("memory_result_source", ""),
        "new_memory_result_source": new.get("memory_result_source", ""),
        "old_memory_sample_count": old.get("memory_sample_count", ""),
        "new_memory_sample_count": new.get("memory_sample_count", ""),
        "old_memory_measurement_window": old.get(
            "memory_measurement_window", ""
        ),
        "new_memory_measurement_window": new.get(
            "memory_measurement_window", ""
        ),
    }
    for field in (
        "gpu_peak_mb_mean",
        "gpu_peak_mb_std",
        "host_rss_peak_mb_mean",
        "host_rss_peak_mb_std",
    ):
        result[f"old_{field}"] = old.get(field, "")
        result[f"new_{field}"] = new.get(field, "")
    return result


def overlay_memory(
    base_fields: list[str],
    base_rows: list[dict[str, str]],
    candidates: list[MemoryCandidate],
    expected_replacements: int | None,
) -> tuple[
    list[str],
    list[dict[str, str]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Overlay memory only after every candidate has passed its sample gate."""

    eggpu_index = validate_ledger(base_fields, base_rows)
    fields = list(base_fields)
    for field in (
        "memory_result_source",
        "memory_candidate_sha256",
        "memory_runtime_python_snapshot_sha256",
    ):
        if field not in fields:
            fields.append(field)

    candidate_by_key: dict[tuple[str, str], MemoryCandidate] = {}
    duplicate_keys = []
    for candidate in candidates:
        if candidate.key in candidate_by_key:
            duplicate_keys.append(candidate.key)
        candidate_by_key[candidate.key] = candidate
    if duplicate_keys:
        raise ValueError(
            "accepted memory candidates contain duplicate dataset/function keys: "
            f"{sorted(set(duplicate_keys))}"
        )

    output_rows = [dict(row) for row in base_rows]
    comparisons: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for key, candidate in sorted(candidate_by_key.items()):
        index = eggpu_index.get(key)
        if index is None:
            rejected.append(
                {
                    "dataset": candidate.dataset,
                    "function": candidate.function,
                    "source_kind": candidate.source_kind,
                    "measurement_kind": "memory",
                    "result_source": candidate.result_source,
                    "reason": "candidate key is absent from base EGGPU ledger",
                }
            )
            continue
        old = dict(output_rows[index])
        if old.get("validation_status") not in {"pass", "sampled_pass"}:
            rejected.append(
                {
                    "dataset": candidate.dataset,
                    "function": candidate.function,
                    "source_kind": candidate.source_kind,
                    "measurement_kind": "memory",
                    "result_source": candidate.result_source,
                    "reason": (
                        "base cell lacks a frozen pass validation; memory-only "
                        "overlay cannot establish semantics"
                    ),
                }
            )
            continue
        new = output_rows[index]
        new.update(
            {
                "gpu_peak_mb_mean": str(candidate.gpu_peak_mb[0]),
                "gpu_peak_mb_std": str(candidate.gpu_peak_mb[1]),
                "host_rss_peak_mb_mean": str(candidate.host_rss_peak_mb[0]),
                "host_rss_peak_mb_std": str(candidate.host_rss_peak_mb[1]),
                "memory_sample_count": str(candidate.sample_count),
                "memory_measurement_window": candidate.measurement_window,
                "memory_result_source": candidate.result_source,
                "memory_candidate_sha256": candidate.candidate_sha256,
                "memory_runtime_python_snapshot_sha256": (
                    candidate.runtime_python_sha256
                ),
            }
        )
        comparisons.append(memory_comparison_row(old, new, candidate))

    if expected_replacements is not None and len(comparisons) != expected_replacements:
        raise ValueError(
            f"memory replacement count is {len(comparisons)}, "
            f"expected {expected_replacements}"
        )
    validate_ledger(fields, output_rows)
    return fields, output_rows, comparisons, rejected


def comparison_path(audit_path: Path) -> Path:
    if audit_path.suffix:
        return audit_path.with_suffix(".cells.csv")
    return Path(str(audit_path) + ".cells.csv")


def memory_comparison_path(audit_path: Path) -> Path:
    if audit_path.suffix:
        return audit_path.with_suffix(".memory_cells.csv")
    return Path(str(audit_path) + ".memory_cells.csv")


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_stability_evidence(
    timing_provenance: list[dict[str, object]],
) -> list[dict[str, object]]:
    evidence: dict[str, dict[str, object]] = {}
    for source in timing_provenance:
        if source.get("measurement_kind") != "timing":
            continue
        source_kind = str(source.get("source_kind", ""))
        result_source = str(Path(str(source["result_dir"])).resolve())
        if source_kind == "main":
            path = str(Path(str(source["results_samples_path"])).resolve())
            record = {
                "source_kind": source_kind,
                "result_source": result_source,
                "evidence_path": path,
                "evidence_sha256": source["results_samples_sha256"],
            }
            evidence[path] = record
        elif source_kind == "anchor":
            hashes = source.get("accepted_aggregate_sha256")
            if not isinstance(hashes, dict):
                raise GateError(
                    f"{result_source}: anchor aggregate hashes are missing"
                )
            for raw_path, digest in hashes.items():
                path = str(Path(str(raw_path)).resolve())
                evidence[path] = {
                    "source_kind": source_kind,
                    "result_source": result_source,
                    "evidence_path": path,
                    "evidence_sha256": digest,
                }
        else:
            raise GateError(
                f"unknown timing provenance source kind: {source_kind!r}"
            )
    return [evidence[path] for path in sorted(evidence)]


def validate_timing_stability_audit(
    audit_path: Path,
    candidates: list[Candidate],
    timing_provenance: list[dict[str, object]],
    *,
    max_over_median_limit: float,
    median_over_min_limit: float,
) -> tuple[dict[str, object], list[Candidate]]:
    """Bind the 208 accepted E2E batches to a pass-only stability audit."""

    audit_path = audit_path.resolve()
    if not audit_path.is_file():
        raise FileNotFoundError(f"missing stability audit: {audit_path}")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GateError("stability audit must be a JSON object")

    expected_scalars = {
        "status": "pass",
        "metric": "e2e",
        "paper_estimator": PAPER_TIMING_ESTIMATOR,
        "acceptance_estimator_independent": True,
        "variance_policy": DEFAULT_VARIANCE_POLICY,
        "sample_std_and_cv_role": (
            "reported_diagnostics_not_acceptance_gate"
        ),
        "failed_batch_policy": (
            "reject_entire_batch_do_not_select_across_batches"
        ),
        "batch_selection_policy": BATCH_SELECTION_POLICY,
        "expected_samples_per_cell": 5,
        "audited_cells": EXPECTED_EGGPU_CELLS,
        "unique_workload_keys": EXPECTED_EGGPU_CELLS,
        "stable_cells": EXPECTED_EGGPU_CELLS,
        "unstable_cells": 0,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise GateError(
                f"stability audit {key}={payload.get(key)!r}, "
                f"expected {expected!r}"
            )
    if payload.get("failures") != []:
        raise GateError("stability audit contains failed timing batches")
    for key, expected in (
        ("max_over_median_limit", max_over_median_limit),
        ("median_over_min_limit", median_over_min_limit),
    ):
        require_close(payload.get(key), expected, f"stability_audit.{key}")
    try:
        validate_gate_calibration(
            payload.get("gate_calibration"),
            repo_root=Path(__file__).resolve().parents[1],
        )
    except (StableTimingProtocolError, TypeError, ValueError) as exc:
        raise GateError(
            f"stability audit gate calibration evidence differs: {exc}"
        ) from exc

    candidate_groups: dict[tuple[str, str], list[Candidate]] = {}
    for candidate in candidates:
        candidate_groups.setdefault(candidate.key, []).append(candidate)
    if len(candidate_groups) != EXPECTED_EGGPU_CELLS:
        raise GateError(
            f"stability gate has {len(candidate_groups)} workload keys, "
            f"expected {EXPECTED_EGGPU_CELLS}"
        )
    nonunique_groups = {
        key: len(group)
        for key, group in candidate_groups.items()
        if len(group) != 1
    }
    if nonunique_groups:
        raise GateError(
            "formal stability input requires one unique original complete "
            f"batch per key; observed multiplicities: {nonunique_groups}"
        )
    candidate_shas = {
        candidate.candidate_sha256 for candidate in candidates
    }
    runtime_shas = {
        candidate.runtime_python_sha256 for candidate in candidates
    }
    if len(candidate_shas) != 1 or "" in runtime_shas or len(runtime_shas) != 1:
        raise GateError(
            "stability candidates do not use one candidate/runtime identity"
        )
    candidate_sha = next(iter(candidate_shas))
    runtime_sha = next(iter(runtime_shas))
    if payload.get("candidate_sha256") != candidate_sha:
        raise GateError("stability audit candidate SHA differs")
    if payload.get("runtime_python_snapshot_sha256") != runtime_sha:
        raise GateError("stability audit runtime Python digest differs")
    if payload.get("runtime_package_is_symlink") is not False:
        raise GateError("stability audit runtime is not proven non-symlinked")

    expected_main_dirs = sorted(
        str(Path(str(source["result_dir"])).resolve())
        for source in timing_provenance
        if source.get("source_kind") == "main"
        and source.get("measurement_kind") == "timing"
    )
    expected_anchor_dirs = sorted(
        str(Path(str(source["result_dir"])).resolve())
        for source in timing_provenance
        if source.get("source_kind") == "anchor"
        and source.get("measurement_kind") == "timing"
    )
    if sorted(payload.get("main_result_dirs", [])) != expected_main_dirs:
        raise GateError("stability audit main result directories differ")
    if sorted(payload.get("anchor_result_dirs", [])) != expected_anchor_dirs:
        raise GateError("stability audit anchor result directories differ")
    expected_identities = sorted(
        (
            {
                "result_dir": str(Path(str(source["result_dir"])).resolve()),
                "candidate_sha256": source["candidate_sha256"],
                "runtime_python_snapshot_sha256": (
                    source["runtime_python_sha256"]
                ),
                "runtime_package_is_symlink": False,
            }
            for source in timing_provenance
            if source.get("measurement_kind") == "timing"
        ),
        key=lambda identity: identity["result_dir"],
    )
    observed_identities = payload.get("result_dir_identities")
    if not isinstance(observed_identities, list):
        raise GateError("stability audit result-directory identities are missing")
    if sorted(
        observed_identities,
        key=lambda identity: str(identity.get("result_dir", "")),
    ) != expected_identities:
        raise GateError("stability audit result-directory identities differ")

    expected_evidence = expected_stability_evidence(timing_provenance)
    if payload.get("input_evidence") != expected_evidence:
        raise GateError("stability audit input evidence hashes differ")
    if payload.get("input_evidence_sha256") != canonical_json_sha256(
        expected_evidence
    ):
        raise GateError("stability audit evidence-set digest differs")

    rows_path = Path(str(payload.get("rows_csv_path", ""))).resolve()
    if not rows_path.is_file():
        raise FileNotFoundError(
            f"missing stability audit rows CSV: {rows_path}"
        )
    if sha256(rows_path) != payload.get("rows_csv_sha256"):
        raise GateError("stability audit rows CSV hash changed")
    _fields, rows = read_csv(rows_path)
    if len(rows) != EXPECTED_EGGPU_CELLS:
        raise GateError(
            f"stability audit CSV has {len(rows)} rows, "
            f"expected {EXPECTED_EGGPU_CELLS}"
        )
    row_keys = [
        (str(row.get("dataset", "")), str(row.get("function", "")))
        for row in rows
    ]
    if len(set(row_keys)) != EXPECTED_EGGPU_CELLS:
        raise GateError("stability audit CSV has duplicate workload keys")
    if set(row_keys) != set(candidate_groups):
        raise GateError("stability audit workload keys differ from candidates")
    audited_keys = sorted(
        (dataset, function, "e2e") for dataset, function in row_keys
    )
    if payload.get("audited_keys_sha256") != canonical_json_sha256(
        audited_keys
    ):
        raise GateError("stability audit workload-key digest differs")

    overrides = payload.get("batch_overrides_applied")
    if not isinstance(overrides, list):
        raise GateError("stability audit replacement records are missing")
    if payload.get(
        "batch_overrides_applied_sha256"
    ) != canonical_json_sha256(overrides):
        raise GateError("stability audit replacement-record digest differs")
    manifest_path_text = str(payload.get("override_manifest_path", ""))
    manifest_sha = str(payload.get("override_manifest_sha256", ""))
    if overrides or manifest_path_text or manifest_sha:
        raise GateError(
            "formal stability audit forbids batch replacements; each key must "
            "use its unique original complete raw5"
        )

    evidence_by_path = {
        str(record["evidence_path"]): record
        for record in expected_evidence
    }

    selected_candidates: list[Candidate] = []
    for row in rows:
        key = (row["dataset"], row["function"])
        group = candidate_groups[key]
        row_result_source = str(Path(row["result_source"]).resolve())
        matching_candidates = [
            candidate
            for candidate in group
            if candidate.source_kind == row.get("source_kind")
            and str(Path(candidate.result_source).resolve())
            == row_result_source
        ]
        if len(matching_candidates) != 1:
            raise GateError(
                f"{key}: stability-selected batch does not identify exactly "
                "one timing candidate"
            )
        candidate = matching_candidates[0]
        timing_stats = candidate.metrics["e2e"]
        if row.get("metric") != "e2e":
            raise GateError(f"{key}: stability audit metric is not e2e")
        if integer(row.get("sample_count"), "stability.sample_count") != 5:
            raise GateError(f"{key}: stability audit sample count is not 5")
        if row.get("stability_status") != "pass":
            raise GateError(f"{key}: stability audit batch is not stable")
        if row.get(
            "batch_acceptance_status"
        ) != "accepted_complete_batch":
            raise GateError(f"{key}: stability audit batch is not accepted")
        if row.get("variance_policy") != DEFAULT_VARIANCE_POLICY:
            raise GateError(f"{key}: stability variance policy differs")
        if row.get("paper_estimator") != PAPER_TIMING_ESTIMATOR:
            raise GateError(f"{key}: stability paper estimator differs")
        if row.get("sample_std_and_cv_role") != (
            "reported_diagnostics_not_acceptance_gate"
        ):
            raise GateError(f"{key}: SD/CV diagnostic role differs")
        if row.get("source_kind") != candidate.source_kind:
            raise GateError(f"{key}: stability source kind differs")
        row_evidence_path = str(Path(row["evidence_path"]).resolve())
        expected_row_evidence = evidence_by_path.get(row_evidence_path)
        if expected_row_evidence is None:
            raise GateError(f"{key}: stability row evidence is unbound")
        if (
            expected_row_evidence["result_source"] != row_result_source
            or expected_row_evidence["source_kind"] != candidate.source_kind
            or row.get("evidence_sha256")
            != expected_row_evidence["evidence_sha256"]
        ):
            raise GateError(f"{key}: stability row evidence identity differs")
        if candidate.source_kind == "anchor":
            expected_anchor_evidence = str(
                (
                    Path(candidate.result_source)
                    / "raw"
                    / f"{candidate.dataset}_{candidate.function}_timing.json"
                ).resolve()
            )
            if row_evidence_path != expected_anchor_evidence:
                raise GateError(
                    f"{key}: anchor stability row uses the wrong aggregate"
                )
        expected_unique_selection = {
            "batch_selection_status": "unique_input_batch",
            "replacement_selection_basis": "",
            "replacement_reason": "",
            "original_result_source": "",
            "original_evidence_path": "",
            "original_evidence_sha256": "",
            "original_stability_status": "",
            "replacement_manifest_path": "",
            "replacement_manifest_sha256": "",
        }
        for field, expected in expected_unique_selection.items():
            if str(row.get(field, "")) != expected:
                raise GateError(
                    f"{key}: unique-batch selection field {field} differs"
                )
        for field, expected in (
            ("minimum_seconds", timing_stats.minimum),
            ("arithmetic_mean_seconds", timing_stats.mean),
            ("median_seconds", timing_stats.median),
            ("maximum_seconds", timing_stats.maximum),
            ("sample_std_seconds", timing_stats.stdev),
            (
                "coefficient_of_variation",
                timing_stats.coefficient_of_variation,
            ),
            ("max_over_median", timing_stats.max_over_median),
            ("median_over_minimum", timing_stats.median_over_minimum),
            (
                "max_over_min",
                (
                    timing_stats.maximum / timing_stats.minimum
                    if timing_stats.minimum > 0
                    else (
                        1.0 if timing_stats.maximum == 0 else math.inf
                    )
                ),
            ),
            ("max_over_median_limit", max_over_median_limit),
            ("median_over_min_limit", median_over_min_limit),
            ("submission_seconds", timing_stats.minimum),
        ):
            require_close(row.get(field), expected, f"{key}.{field}")
        if row.get("failure_reasons") != "[]":
            raise GateError(f"{key}: accepted batch has failure reasons")
        try:
            samples = json.loads(str(row.get("samples_seconds", "")))
        except json.JSONDecodeError as exc:
            raise GateError(f"{key}: stability samples are invalid JSON") from exc
        if not isinstance(samples, list) or len(samples) != 5:
            raise GateError(f"{key}: stability samples are incomplete")
        for index, (observed, expected) in enumerate(
            zip(samples, timing_stats.samples),
            start=1,
        ):
            require_close(observed, expected, f"{key}.sample{index}")
        try:
            raw5 = json.loads(str(row.get("raw5_seconds", "")))
        except json.JSONDecodeError as exc:
            raise GateError(f"{key}: raw5_seconds is invalid JSON") from exc
        if raw5 != samples:
            raise GateError(f"{key}: raw5_seconds differs from samples_seconds")
        selected_candidates.append(candidate)

    if len(selected_candidates) != EXPECTED_EGGPU_CELLS or len(
        {candidate.key for candidate in selected_candidates}
    ) != EXPECTED_EGGPU_CELLS:
        raise GateError("stability audit did not select 208 unique candidates")

    record = {
        "status": "pass",
        "variance_policy": DEFAULT_VARIANCE_POLICY,
        "paper_estimator": PAPER_TIMING_ESTIMATOR,
        "batch_selection_policy": BATCH_SELECTION_POLICY,
        "gate_calibration": gate_calibration_payload(),
        "audited_cells": EXPECTED_EGGPU_CELLS,
        "candidate_sha256": candidate_sha,
        "runtime_python_snapshot_sha256": runtime_sha,
        "audit_path": str(audit_path),
        "audit_sha256": sha256(audit_path),
        "rows_csv_path": str(rows_path),
        "rows_csv_sha256": sha256(rows_path),
        "override_manifest_path": manifest_path_text,
        "override_manifest_sha256": manifest_sha,
        "batch_overrides_applied": overrides,
        "batch_overrides_applied_sha256": canonical_json_sha256(overrides),
        "max_over_median_limit": max_over_median_limit,
        "median_over_min_limit": median_over_min_limit,
    }
    return record, selected_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ledger", required=True, type=Path)
    parser.add_argument(
        "--candidate-mode",
        choices=("legacy-timing-only", "unified-timing-memory"),
        default="legacy-timing-only",
        help=(
            "Archived V9 reconstruction keeps frozen memory. Unified mode "
            "requires complete timing and memory inputs from one recorded "
            "binary and non-symlinked Python runtime."
        ),
    )
    parser.add_argument(
        "--main-result-dir",
        action="append",
        default=[],
        type=Path,
        help="Legacy alias for --main-timing-result-dir.",
    )
    parser.add_argument(
        "--anchor-result-dir",
        action="append",
        default=[],
        type=Path,
        help="Legacy alias for --anchor-timing-result-dir.",
    )
    parser.add_argument(
        "--main-timing-result-dir", action="append", default=[], type=Path
    )
    parser.add_argument(
        "--anchor-timing-result-dir", action="append", default=[], type=Path
    )
    parser.add_argument(
        "--main-memory-result-dir", action="append", default=[], type=Path
    )
    parser.add_argument(
        "--anchor-memory-result-dir", action="append", default=[], type=Path
    )
    parser.add_argument("--output-ledger", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument(
        "--expected-replacements",
        type=int,
        help="Legacy alias for --expected-timing-replacements.",
    )
    parser.add_argument("--expected-timing-replacements", type=int)
    parser.add_argument("--expected-memory-replacements", type=int)
    parser.add_argument(
        "--stability-audit",
        type=Path,
        help=(
            "Pass-only E2E raw-five audit from "
            "audit_eggpu_timing_stability.py; required in unified mode."
        ),
    )
    parser.add_argument(
        "--stability-max-over-median-limit",
        type=float,
        default=DEFAULT_MAX_OVER_MEDIAN_LIMIT,
    )
    parser.add_argument(
        "--stability-median-over-min-limit",
        type=float,
        default=DEFAULT_MEDIAN_OVER_MIN_LIMIT,
    )
    parser.add_argument(
        "--candidate-sha256",
        help=(
            "Explicit provenance fallback, primarily for scaling outputs whose "
            "runner does not write run_metadata.json."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every gate and print the audit without writing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unified = args.candidate_mode == "unified-timing-memory"
    if args.stability_max_over_median_limit < 1:
        raise ValueError("--stability-max-over-median-limit must be at least 1")
    if args.stability_median_over_min_limit < 1:
        raise ValueError("--stability-median-over-min-limit must be at least 1")
    if unified and args.stability_audit is None:
        raise ValueError(
            "unified mode requires an explicit --stability-audit"
        )
    if unified and (args.main_result_dir or args.anchor_result_dir):
        raise ValueError(
            "unified mode requires explicit --main-timing-result-dir and "
            "--anchor-timing-result-dir; legacy aliases are not accepted"
        )
    if not unified and (
        args.main_memory_result_dir or args.anchor_memory_result_dir
    ):
        raise ValueError(
            "memory result directories require "
            "--candidate-mode unified-timing-memory"
        )
    main_timing_dirs = [
        *args.main_result_dir,
        *args.main_timing_result_dir,
    ]
    anchor_timing_dirs = [
        *args.anchor_result_dir,
        *args.anchor_timing_result_dir,
    ]
    if unified and (
        not main_timing_dirs
        or not anchor_timing_dirs
        or not args.main_memory_result_dir
        or not args.anchor_memory_result_dir
    ):
        raise ValueError(
            "unified mode requires regular and anchor timing directories plus "
            "regular and anchor memory directories"
        )
    expected_timing = (
        args.expected_timing_replacements
        if args.expected_timing_replacements is not None
        else args.expected_replacements
    )
    if (
        args.expected_timing_replacements is not None
        and args.expected_replacements is not None
        and args.expected_timing_replacements != args.expected_replacements
    ):
        raise ValueError(
            "--expected-replacements and --expected-timing-replacements disagree"
        )
    expected_memory = args.expected_memory_replacements
    if unified:
        expected_timing = (
            EXPECTED_EGGPU_CELLS
            if expected_timing is None
            else expected_timing
        )
        expected_memory = (
            EXPECTED_EGGPU_CELLS
            if expected_memory is None
            else expected_memory
        )
        if (
            expected_timing != EXPECTED_EGGPU_CELLS
            or expected_memory != EXPECTED_EGGPU_CELLS
        ):
            raise ValueError(
                "unified mode requires exactly 208 timing and 208 memory "
                "replacements"
            )

    base = args.base_ledger.resolve()
    output = args.output_ledger.resolve()
    audit_path = args.audit_json.resolve()
    cells_path = comparison_path(audit_path)
    memory_cells_path = memory_comparison_path(audit_path)
    if not args.dry_run:
        existing = [
            path
            for path in (output, audit_path, cells_path, memory_cells_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing outputs: "
                + ", ".join(str(path) for path in existing)
            )

    fields, rows = read_csv(base)
    validate_ledger(fields, rows)
    candidates: list[Candidate] = []
    rejected: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    for result_dir in main_timing_dirs:
        found, failures, source = main_candidates(
            result_dir,
            args.candidate_sha256,
            strict_provenance=unified,
            max_over_median_limit=(
                args.stability_max_over_median_limit
            ),
            median_over_min_limit=args.stability_median_over_min_limit,
        )
        candidates.extend(found)
        rejected.extend(failures)
        provenance.append(source)
    for result_dir in anchor_timing_dirs:
        found, failures, source = anchor_candidates(
            result_dir,
            args.candidate_sha256,
            strict_provenance=unified,
            max_over_median_limit=(
                args.stability_max_over_median_limit
            ),
            median_over_min_limit=args.stability_median_over_min_limit,
        )
        candidates.extend(found)
        rejected.extend(failures)
        provenance.append(source)

    stability_audit = None
    if args.stability_audit is not None:
        stability_audit, candidates = validate_timing_stability_audit(
            args.stability_audit,
            candidates,
            provenance,
            max_over_median_limit=(
                args.stability_max_over_median_limit
            ),
            median_over_min_limit=args.stability_median_over_min_limit,
        )

    out_fields, out_rows, comparisons, overlay_rejected = overlay(
        fields, rows, candidates, expected_timing
    )
    rejected.extend(overlay_rejected)
    memory_candidates: list[MemoryCandidate] = []
    memory_comparisons: list[dict[str, object]] = []
    unified_identity: dict[str, str] | None = None
    if unified:
        for result_dir in args.main_memory_result_dir:
            found, failures, source = main_memory_candidates(
                result_dir,
                args.candidate_sha256,
                strict_provenance=True,
            )
            memory_candidates.extend(found)
            rejected.extend(failures)
            provenance.append(source)
        for result_dir in args.anchor_memory_result_dir:
            found, failures, source = anchor_memory_candidates(
                result_dir,
                args.candidate_sha256,
                strict_provenance=True,
            )
            memory_candidates.extend(found)
            rejected.extend(failures)
            provenance.append(source)
        unified_identity = validate_unified_candidate_identity(
            candidates, memory_candidates
        )
        (
            out_fields,
            out_rows,
            memory_comparisons,
            memory_overlay_rejected,
        ) = overlay_memory(
            out_fields,
            out_rows,
            memory_candidates,
            expected_memory,
        )
        rejected.extend(memory_overlay_rejected)

    accepted_keys = {candidate.key for candidate in candidates}
    accepted_memory_keys = {candidate.key for candidate in memory_candidates}

    def rejection_is_superseded(rejection: dict[str, object]) -> bool:
        key = (rejection.get("dataset"), rejection.get("function"))
        if rejection.get("measurement_kind", "timing") == "memory":
            return key in accepted_memory_keys
        return key in accepted_keys

    superseded_rejections = [
        rejection
        for rejection in rejected
        if rejection_is_superseded(rejection)
    ]
    active_rejections = [
        rejection
        for rejection in rejected
        if not rejection_is_superseded(rejection)
    ]
    audit = {
        "status": "pass",
        "dry_run": bool(args.dry_run),
        "base_ledger": str(base),
        "base_ledger_sha256": sha256(base),
        "output_ledger": str(output),
        "comparison_csv": str(cells_path),
        "memory_comparison_csv": (
            str(memory_cells_path) if unified else None
        ),
        "candidate_mode": args.candidate_mode,
        "paper_timing_estimator": PAPER_TIMING_ESTIMATOR,
        "timing_variance_policy": DEFAULT_VARIANCE_POLICY,
        "stability_max_over_median_limit": (
            args.stability_max_over_median_limit
        ),
        "stability_median_over_min_limit": (
            args.stability_median_over_min_limit
        ),
        "stability_acceptance_uses_minimum_value": False,
        "ledger_cells": len(out_rows),
        "eggpu_cells": sum(row["baseline"] == "EGGPU" for row in out_rows),
        "eggpu_ok_cells": sum(
            row["baseline"] == "EGGPU" and row["execution_status"] == "ok"
            for row in out_rows
        ),
        "accepted_candidate_cells": len(candidates),
        "replacement_count": len(comparisons),
        "expected_replacements": expected_timing,
        "timing_accepted_candidate_cells": len(candidates),
        "timing_replacement_count": len(comparisons),
        "expected_timing_replacements": expected_timing,
        "memory_accepted_candidate_cells": len(memory_candidates),
        "memory_replacement_count": len(memory_comparisons),
        "expected_memory_replacements": expected_memory,
        "candidate_sha256": sorted(
            {
                candidate.candidate_sha256
                for candidate in (*candidates, *memory_candidates)
            }
        ),
        "runtime_python_snapshot_sha256": sorted(
            {
                candidate.runtime_python_sha256
                for candidate in (*candidates, *memory_candidates)
                if candidate.runtime_python_sha256
            }
        ),
        "unified_identity": unified_identity,
        "timing_stability_audit": stability_audit,
        "sources": provenance,
        "rejected_candidates": active_rejections,
        "superseded_rejections": superseded_rejections,
        "superseded_rejection_count": len(superseded_rejections),
        "warmup_provenance_note": (
            "run_full_baselines stores warmup settings in run_metadata.json. "
            "V10 run_eggpu_scaling directories bind five fresh workers to a "
            "hashed run_metadata.json recording one first-use call, one extra "
            "untimed call, and measurement on the third call. Legacy timing-only "
            "reconstruction may use a hashed protocol.json sidecar."
        ),
        "memory_provenance_note": (
            "Unified mode replaces all 208 EGGPU memory cells from three "
            "independent isolated-process samples. Regular cells are read from "
            "results_samples/results_long; anchors are checked against three "
            "worker JSON records."
            if unified
            else (
                "Legacy timing-only mode does not modify archived memory "
                "columns."
            )
        ),
    }
    if not args.dry_run:
        write_csv_atomic(output, out_rows, out_fields)
        comparison_fields = list(comparisons[0]) if comparisons else [
            "dataset",
            "function",
            "source_kind",
            "candidate_sha256",
        ]
        write_csv_atomic(cells_path, comparisons, comparison_fields)
        if unified:
            memory_comparison_fields = (
                list(memory_comparisons[0])
                if memory_comparisons
                else [
                    "dataset",
                    "function",
                    "source_kind",
                    "candidate_sha256",
                    "runtime_python_snapshot_sha256",
                ]
            )
            write_csv_atomic(
                memory_cells_path,
                memory_comparisons,
                memory_comparison_fields,
            )
        audit["output_ledger_sha256"] = sha256(output)
        audit["comparison_csv_sha256"] = sha256(cells_path)
        if unified:
            audit["memory_comparison_csv_sha256"] = sha256(memory_cells_path)
        write_text_atomic(
            audit_path,
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
        )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
