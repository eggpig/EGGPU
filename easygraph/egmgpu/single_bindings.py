"""Evidence-gated single-device and request-parallel native bindings.

The 16 entries in this module are wiring descriptions, not availability
claims.  Registration requires retained correctness evidence from the exact
native symbol and hardware topology.  Both SINGLE and REQUEST_PARALLEL use
the same C++ dispatcher; the scheduler reserves one device and the dispatcher
submits the request to that device's process-global fixed worker.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple

from .native_bindings import NativeBindingError
from .native_bindings import NativeBindingEvidenceError
from .native_bindings import NativeBindingResultError
from .snapshot import FrozenNativeGraph
from .snapshot import native_graph_handle
from .snapshot import snapshot_public_nodes
from .types import ExecutionMode
from .types import NativeCallContext


SINGLE_NATIVE_PROBE_SCHEMA = "egmgpu.single-request-native-probe/v3"
SINGLE_NATIVE_BINDING = "cpp_egmgpu_single_execute"

_EXACT_COMPARATOR_FUNCTIONS = frozenset({"mst", "wcc", "scc", "kcore"})
_INTEGER_DISTANCE_FUNCTIONS = frozenset({"bfs"})
_FLOAT_COMPARATOR_FUNCTIONS = frozenset(
    {
        "pagerank",
        "lcc",
        "dijkstra",
        "bellman_ford",
        "sssp",
        "bc",
        "closeness",
        "effective_size",
        "efficiency",
        "constraint",
        "hierarchy",
    }
)
_FIXED_FLOAT_RTOL = 1.0e-6
_FIXED_FLOAT_ATOL = 1.0e-8


class SingleResultKind(str, Enum):
    NODE_MAPPING = "node_mapping"
    DENSE_SEQUENCE = "dense_sequence"
    COMPONENTS = "components"
    SINGLE_SOURCE_DISTANCE = "single_source_distance"
    MULTI_SOURCE_DISTANCE = "multi_source_distance"
    INTEGER_SEQUENCE = "integer_sequence"
    MST_GRAPH = "mst_graph"


class SingleBindingReadiness(str, Enum):
    WIREABLE = "wireable"
    AVAILABLE = "available"


@dataclass(frozen=True)
class SingleNativeBindingSpec:
    name: str
    result_kind: SingleResultKind
    binding: str = SINGLE_NATIVE_BINDING

    def __post_init__(self) -> None:
        if not self.name or not self.binding:
            raise ValueError("single native binding name and symbol must be non-empty")


@dataclass(frozen=True)
class SingleNativeEvidence:
    result_path: str
    sha256: str
    extension_path: str
    extension_sha256: str
    deployment_fingerprint: str
    deployment_measurement_path: str
    deployment_measurement_sha256: str
    device_group_fingerprint: str
    devices: Tuple[int, ...]
    passed_functions: Tuple[str, ...]
    two_device_smoke_passed: bool
    multi_device_stress_passed: bool
    completed_stress_calls: int
    public_api_smoke_passed: bool

    def __post_init__(self) -> None:
        devices = tuple(sorted(int(device) for device in self.devices))
        if len(devices) < 2 or len(set(devices)) != len(devices):
            raise NativeBindingEvidenceError(
                "single/request evidence requires at least two unique devices"
            )
        object.__setattr__(self, "devices", devices)
        functions = tuple(sorted(str(name) for name in self.passed_functions))
        if not functions:
            raise NativeBindingEvidenceError("no function passed the native probe")
        object.__setattr__(self, "passed_functions", functions)
        digest = str(self.sha256).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise NativeBindingEvidenceError("single/request evidence has invalid SHA-256")
        object.__setattr__(self, "sha256", digest)
        extension_digest = str(self.extension_sha256).lower()
        if len(extension_digest) != 64 or any(
            c not in "0123456789abcdef" for c in extension_digest
        ):
            raise NativeBindingEvidenceError(
                "single/request extension has invalid SHA-256"
            )
        object.__setattr__(self, "extension_sha256", extension_digest)
        if not self.extension_path:
            raise NativeBindingEvidenceError("compiled extension path is required")
        deployment_digest = str(self.deployment_fingerprint).lower()
        if len(deployment_digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in deployment_digest
        ):
            raise NativeBindingEvidenceError(
                "deployment fingerprint is not a SHA-256"
            )
        object.__setattr__(self, "deployment_fingerprint", deployment_digest)
        if not self.deployment_measurement_path:
            raise NativeBindingEvidenceError("deployment measurement path is required")
        for label, value in (
            ("deployment measurement", self.deployment_measurement_sha256),
            ("device group", self.device_group_fingerprint),
        ):
            digest = str(value).lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise NativeBindingEvidenceError(
                    "{} fingerprint is not a SHA-256".format(label)
                )
        if not self.two_device_smoke_passed or not self.multi_device_stress_passed:
            raise NativeBindingEvidenceError(
                "both the two-device smoke and selected-group mixed stress must pass"
            )
        # A mixed stress record must exercise every admitted function in more
        # than one scheduling round; a single hand-written success per name is
        # not production concurrency evidence.
        if int(self.completed_stress_calls) < 2 * len(functions):
            raise NativeBindingEvidenceError(
                "mixed-request evidence must cover every function in multiple rounds"
            )
        object.__setattr__(self, "completed_stress_calls", int(self.completed_stress_calls))
        if not self.public_api_smoke_passed:
            raise NativeBindingEvidenceError(
                "the public egmgpu.submit smoke test must pass"
            )

    @property
    def topology_fingerprint(self) -> str:
        """Compatibility alias for the full measured deployment identity."""

        return self.deployment_fingerprint


@dataclass(frozen=True)
class SingleBindingStatus:
    spec: SingleNativeBindingSpec
    readiness: SingleBindingReadiness
    evidence: Optional[SingleNativeEvidence] = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _passed(record: Any) -> bool:
    return bool(
        isinstance(record, dict)
        and record.get("outcome") == "ok"
        and record.get("correctness_passed") is True
    )


def _digest_field(record: Mapping, name: str) -> str:
    value = str(record.get(name, "")).lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise NativeBindingEvidenceError(
            "comparison field {} is not a SHA-256 digest".format(name)
        )
    return value


def _decode_evidence_number(value: Any) -> float:
    if value == "nan":
        return float("nan")
    if value == "+inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeBindingEvidenceError("numeric comparison contains a non-number")
    return float(value)


def _close_reported(found: Any, expected: float) -> bool:
    try:
        value = float(found)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(
        value, expected, rel_tol=1.0e-10, abs_tol=1.0e-12
    )


def _validate_comparison(function_name: str, comparison: Any) -> bool:
    if not isinstance(comparison, Mapping):
        raise NativeBindingEvidenceError("direct probe lacks a semantic comparator")
    kind = comparison.get("kind")
    expected_digest = _digest_field(comparison, "expected_digest")
    observed_digest = _digest_field(comparison, "observed_digest")
    digest_match = expected_digest == observed_digest
    if comparison.get("exact_digest_match") is not digest_match:
        raise NativeBindingEvidenceError("comparison digest-match flag is inconsistent")
    if kind == "exact_canonical":
        if function_name not in _EXACT_COMPARATOR_FUNCTIONS:
            raise NativeBindingEvidenceError(
                "{} requires a numeric semantic comparator".format(function_name)
            )
        calculated = digest_match
    elif kind == "numeric_allclose":
        if function_name in _INTEGER_DISTANCE_FUNCTIONS:
            allowed_rtol = 0.0
            allowed_atol = 0.0
        elif function_name in _FLOAT_COMPARATOR_FUNCTIONS:
            allowed_rtol = _FIXED_FLOAT_RTOL
            allowed_atol = _FIXED_FLOAT_ATOL
        else:
            raise NativeBindingEvidenceError(
                "{} does not allow a numeric comparator".format(function_name)
            )
        expected_structure = _digest_field(
            comparison, "expected_structure_digest"
        )
        observed_structure = _digest_field(
            comparison, "observed_structure_digest"
        )
        try:
            rtol = float(comparison["rtol"])
            atol = float(comparison["atol"])
            sample_count = int(comparison["sample_count"])
            expected_values = tuple(
                _decode_evidence_number(value)
                for value in comparison["expected_values"]
            )
            observed_values = tuple(
                _decode_evidence_number(value)
                for value in comparison["observed_values"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeBindingEvidenceError(
                "numeric comparison witness is incomplete"
            ) from exc
        if not (
            math.isfinite(rtol)
            and math.isfinite(atol)
            and rtol >= 0.0
            and atol >= 0.0
            and sample_count >= 1
            and len(expected_values) == sample_count
            and len(observed_values) == sample_count
        ):
            raise NativeBindingEvidenceError("numeric comparison dimensions are invalid")
        if rtol != allowed_rtol or atol != allowed_atol:
            raise NativeBindingEvidenceError(
                "{} comparison must use fixed rtol={} and atol={}".format(
                    function_name, allowed_rtol, allowed_atol
                )
            )
        nonfinite_match = True
        max_abs = 0.0
        max_rel = 0.0
        max_normalized = 0.0
        for left, right in zip(expected_values, observed_values):
            if math.isnan(left) or math.isnan(right):
                nonfinite_match = nonfinite_match and math.isnan(left) and math.isnan(right)
                continue
            if math.isinf(left) or math.isinf(right):
                nonfinite_match = nonfinite_match and left == right
                continue
            error = abs(left - right)
            max_abs = max(max_abs, error)
            max_rel = max(max_rel, error / max(abs(left), atol or 1.0e-300))
            denominator = atol + rtol * abs(left)
            normalized = 0.0 if error == 0.0 else (
                float("inf") if denominator == 0.0 else error / denominator
            )
            max_normalized = max(max_normalized, normalized)
        if function_name in _INTEGER_DISTANCE_FUNCTIONS:
            for value in expected_values + observed_values:
                if math.isfinite(value) and not value.is_integer():
                    raise NativeBindingEvidenceError(
                        "BFS distance witness must contain only integers or infinity"
                    )
        if comparison.get("nonfinite_domain_match") is not nonfinite_match:
            raise NativeBindingEvidenceError(
                "numeric comparison non-finite-domain flag is inconsistent"
            )
        if not (
            _close_reported(comparison.get("max_abs_error"), max_abs)
            and _close_reported(comparison.get("max_rel_error"), max_rel)
            and _close_reported(
                comparison.get("max_normalized_error"), max_normalized
            )
        ):
            raise NativeBindingEvidenceError(
                "numeric comparison error metrics do not match its raw witness"
            )
        calculated = bool(
            expected_structure == observed_structure
            and nonfinite_match
            and max_normalized <= 1.0
        )
    else:
        raise NativeBindingEvidenceError(
            "unsupported semantic comparator {!r}".format(kind)
        )
    if comparison.get("passed") is not calculated:
        raise NativeBindingEvidenceError("semantic comparator pass flag is inconsistent")
    return calculated


def _validate_direct_probe(function_name: str, probe: Any) -> bool:
    if not isinstance(probe, Mapping):
        raise NativeBindingEvidenceError("direct probe record is missing")
    calculated = _validate_comparison(function_name, probe.get("comparison"))
    claimed = _passed(probe)
    if claimed is not calculated:
        raise NativeBindingEvidenceError(
            "direct probe outcome does not match its semantic comparison"
        )
    return calculated


def _canonical_device_group_from_measurement(
    measurement: Mapping, devices: Tuple[int, ...]
) -> Mapping:
    topology = measurement.get("topology")
    if not isinstance(topology, Mapping):
        raise NativeBindingEvidenceError(
            "deployment measurement has no topology object"
        )
    raw_gpus = topology.get("gpus")
    peer = topology.get("p2p_access")
    if not isinstance(raw_gpus, list) or not isinstance(peer, Mapping):
        raise NativeBindingEvidenceError(
            "deployment measurement lacks GPU identity or P2P facts"
        )
    by_id = {}
    for item in raw_gpus:
        if not isinstance(item, Mapping):
            raise NativeBindingEvidenceError("deployment GPU identity is invalid")
        device = int(item.get("index", -1))
        memory_mib = item.get("memory_total_mib")
        if type(memory_mib) is not int or memory_mib <= 0:
            raise NativeBindingEvidenceError(
                "deployment GPU nominal memory must be a positive integer"
            )
        by_id[device] = {
            "device_id": device,
            "name": str(item.get("name", "")),
            "pci_bus_id": _canonical_pci_bus_id(item.get("pci_bus_id", "")),
            "compute_capability": str(item.get("compute_capability", "")),
            "total_global_memory": memory_mib * (1 << 20),
        }
    if any(device not in by_id for device in devices):
        raise NativeBindingEvidenceError(
            "evidence device group is absent from the deployment measurement"
        )
    matrix = []
    for source in devices:
        row = []
        for target in devices:
            key = "{}->{}".format(source, target)
            if source != target and key not in peer:
                raise NativeBindingEvidenceError(
                    "deployment measurement lacks a selected P2P fact"
                )
            row.append(
                True
                if source == target
                else bool(peer[key])
            )
        matrix.append(row)
    return {
        "devices": [by_id[device] for device in devices],
        "peer_access": matrix,
    }


def _canonical_pci_bus_id(value: Any) -> str:
    fields = str(value).strip().upper().split(":")
    return ":".join(fields[-2:]) if len(fields) >= 2 else str(value).strip().upper()


def _canonical_group_payload(value: Any, devices: Tuple[int, ...]) -> Mapping:
    if not isinstance(value, Mapping):
        raise NativeBindingEvidenceError("device-group identity is missing")
    raw_devices = value.get("devices")
    peer = value.get("peer_access")
    if not isinstance(raw_devices, list) or not isinstance(peer, list):
        raise NativeBindingEvidenceError("device-group identity is incomplete")
    normalized = []
    for item in raw_devices:
        if not isinstance(item, Mapping):
            raise NativeBindingEvidenceError("device-group GPU identity is invalid")
        memory = item.get("total_global_memory")
        if type(memory) is not int or memory <= 0:
            raise NativeBindingEvidenceError(
                "device-group allocatable memory must be a positive integer"
            )
        normalized.append(
            {
                "device_id": int(item.get("device_id", -1)),
                "name": str(item.get("name", "")),
                "pci_bus_id": _canonical_pci_bus_id(item.get("pci_bus_id", "")),
                "compute_capability": str(item.get("compute_capability", "")),
                "total_global_memory": memory,
            }
        )
    if tuple(item["device_id"] for item in normalized) != devices:
        raise NativeBindingEvidenceError(
            "device-group identity does not match its selected devices"
        )
    try:
        matrix = [[bool(peer[i][j]) for j in range(len(devices))] for i in range(len(devices))]
    except (IndexError, TypeError) as exc:
        raise NativeBindingEvidenceError("device-group P2P matrix is incomplete") from exc
    if len(peer) != len(devices) or any(len(row) != len(devices) for row in peer):
        raise NativeBindingEvidenceError("device-group P2P matrix has invalid dimensions")
    return {"devices": normalized, "peer_access": matrix}


def _validate_group_against_measurement(
    group: Mapping, measured: Mapping
) -> None:
    """Bind a native group to deployment facts without conflating memory APIs."""

    group_devices = group["devices"]
    measured_devices = measured["devices"]
    if len(group_devices) != len(measured_devices):
        raise NativeBindingEvidenceError(
            "device-group identity differs from the deployment measurement"
        )
    gib = 1 << 30
    for native, deployment in zip(group_devices, measured_devices):
        for field in (
            "device_id",
            "name",
            "pci_bus_id",
            "compute_capability",
        ):
            if native[field] != deployment[field]:
                raise NativeBindingEvidenceError(
                    "device-group identity differs from the deployment measurement"
                )
        # CUDA's totalGlobalMem is allocatable bytes after driver/ECC reserve;
        # nvidia-smi's memory.total is nominal MiB.  Match the exact capacity
        # class used by partitioned evidence while retaining the native byte
        # count in the signed device-group fingerprint.
        runtime_memory = native["total_global_memory"]
        deployment_memory = deployment["total_global_memory"]
        if (
            type(runtime_memory) is not int
            or runtime_memory <= 0
            or type(deployment_memory) is not int
            or deployment_memory <= 0
        ):
            raise NativeBindingEvidenceError(
                "device-group memory values must be positive integers"
            )
        runtime_capacity_class = (runtime_memory + gib - 1) // gib
        deployment_capacity_class = (deployment_memory + gib - 1) // gib
        if runtime_capacity_class != deployment_capacity_class:
            raise NativeBindingEvidenceError(
                "device-group memory class differs from the deployment measurement"
            )
    size = len(group_devices)
    for identity in (group, measured):
        matrix = identity.get("peer_access")
        if (
            not isinstance(matrix, list)
            or len(matrix) != size
            or any(not isinstance(row, list) or len(row) != size for row in matrix)
            or any(type(cell) is not bool for row in matrix for cell in row)
        ):
            raise NativeBindingEvidenceError(
                "device-group P2P identity is not a full matrix"
            )
    if group["peer_access"] != measured["peer_access"]:
        raise NativeBindingEvidenceError(
            "device-group P2P identity differs from the deployment measurement"
        )


def _mapping_sha256(value: Mapping) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_single_native_evidence(path: Any) -> SingleNativeEvidence:
    """Load one retained all-function correctness result without probing GPUs."""

    result_path = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeBindingEvidenceError(
            "cannot parse single/request probe result {}: {}".format(result_path, exc)
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema") != SINGLE_NATIVE_PROBE_SCHEMA:
        raise NativeBindingEvidenceError("unsupported single/request native probe schema")
    deployment = payload.get("deployment")
    device_group = payload.get("device_group")
    devices = payload.get("devices")
    functions = payload.get("functions")
    request_parallel = payload.get("request_parallel")
    extension = payload.get("extension")
    if (
        not isinstance(deployment, Mapping)
        or not isinstance(device_group, Mapping)
        or not isinstance(devices, list)
    ):
        raise NativeBindingEvidenceError(
            "probe deployment, device group, and devices are required"
        )
    if (
        not isinstance(functions, dict)
        or not isinstance(request_parallel, dict)
        or not isinstance(extension, dict)
    ):
        raise NativeBindingEvidenceError("probe function and request-parallel records are required")

    extension_path_value = extension.get("path")
    extension_digest = str(extension.get("sha256", "")).lower()
    if not isinstance(extension_path_value, str):
        raise NativeBindingEvidenceError("compiled extension path is required")
    # The producing extension path is provenance only.  Requiring that old
    # absolute path to remain present would make a checksummed evidence bundle
    # unusable after cleanup or repository relocation.  Installation hashes
    # the *currently loaded* extension and compares it with this digest.
    if len(extension_digest) != 64 or any(
        character not in "0123456789abcdef" for character in extension_digest
    ):
        raise NativeBindingEvidenceError(
            "compiled extension identity is not a SHA-256"
        )

    normalized_devices = tuple(sorted(int(device) for device in devices))
    measurement_path_value = deployment.get("path")
    measurement_sha = str(deployment.get("sha256", "")).lower()
    deployment_fingerprint = str(
        deployment.get("topology_fingerprint", "")
    ).lower()
    if not isinstance(measurement_path_value, str):
        raise NativeBindingEvidenceError("deployment measurement path is required")
    measurement_relative = Path(measurement_path_value)
    for label, digest in (
        ("deployment measurement", measurement_sha),
        ("deployment fingerprint", deployment_fingerprint),
    ):
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise NativeBindingEvidenceError(
                "{} is not a SHA-256".format(label)
            )
    if measurement_relative.is_absolute():
        raise NativeBindingEvidenceError(
            "deployment measurement path must be relative to the evidence bundle"
        )
    measurement_root = result_path.parent.resolve(strict=True)
    measurement_path = (measurement_root / measurement_relative).resolve(strict=True)
    try:
        measurement_path.relative_to(measurement_root)
    except ValueError as exc:
        raise NativeBindingEvidenceError(
            "deployment measurement path escapes the evidence bundle"
        ) from exc
    if _sha256(measurement_path) != measurement_sha:
        raise NativeBindingEvidenceError(
            "deployment measurement changed after request evidence was captured"
        )
    try:
        measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeBindingEvidenceError(
            "cannot parse retained deployment measurement"
        ) from exc
    if (
        not isinstance(measurement, Mapping)
        or str(measurement.get("topology_fingerprint", "")).lower()
        != deployment_fingerprint
    ):
        raise NativeBindingEvidenceError(
            "deployment fingerprint differs from its retained measurement"
        )
    group_identity = _canonical_group_payload(
        device_group.get("identity"), normalized_devices
    )
    group_fingerprint = str(device_group.get("fingerprint", "")).lower()
    if _mapping_sha256(group_identity) != group_fingerprint:
        raise NativeBindingEvidenceError(
            "device-group fingerprint differs from its retained identity"
        )
    measured_group = _canonical_device_group_from_measurement(
        measurement, normalized_devices
    )
    _validate_group_against_measurement(group_identity, measured_group)

    passed_functions = []
    for name, record in functions.items():
        if not isinstance(record, dict):
            continue
        if record.get("binding") != SINGLE_NATIVE_BINDING:
            continue
        try:
            passed = _validate_direct_probe(str(name), record.get("direct_probe"))
        except NativeBindingEvidenceError as exc:
            raise NativeBindingEvidenceError(
                "{} direct probe is invalid: {}".format(name, exc)
            ) from exc
        if passed:
            passed_functions.append(str(name))

    two = request_parallel.get("two_device_same_graph")
    multi = request_parallel.get("multi_device_mixed_stress")
    two_functions = tuple(two.get("functions", ())) if isinstance(two, dict) else ()
    two_devices = (
        tuple(int(device) for device in two.get("devices", ()))
        if isinstance(two, dict)
        else ()
    )
    two_assignments = two.get("assignments") if isinstance(two, dict) else None
    two_passed = (
        _passed(two)
        and two_devices == normalized_devices[:2]
        and tuple(str(name) for name in two_functions)
        == ("pagerank", "effective_size")
        and isinstance(two_assignments, list)
        and two_assignments
        == [
            {"function": "pagerank", "device": normalized_devices[0]},
            {"function": "effective_size", "device": normalized_devices[1]},
        ]
    )
    stress_devices = (
        tuple(int(d) for d in multi.get("devices", ()))
        if isinstance(multi, dict)
        else ()
    )
    stress_function_list = (
        tuple(str(name) for name in multi.get("functions", ()))
        if isinstance(multi, dict)
        else ()
    )
    stressed = set(stress_function_list)
    assignments = multi.get("assignments") if isinstance(multi, dict) else None
    observations = (
        multi.get("native_observations") if isinstance(multi, dict) else None
    )
    assignment_valid = _validate_mixed_stress_witness(
        assignments,
        observations,
        normalized_devices,
        tuple(sorted(passed_functions)),
    )
    multi_passed = (
        _passed(multi)
        and stress_devices == normalized_devices
        and len(stress_devices) >= 2
        and len(stress_function_list) == len(stressed)
        and stressed == set(passed_functions)
        and multi.get("fixed_worker_device_check") is True
        and multi.get("mismatches") == []
        and assignment_valid
    )
    completed = int(multi.get("completed_calls", 0)) if isinstance(multi, dict) else 0
    if isinstance(assignments, list) and completed != len(assignments):
        raise NativeBindingEvidenceError(
            "mixed stress completed_calls differs from its assignment witness"
        )
    public_smoke = payload.get("public_api_smoke")
    return SingleNativeEvidence(
        result_path=str(result_path),
        sha256=_sha256(result_path),
        extension_path=extension_path_value,
        extension_sha256=extension_digest,
        deployment_fingerprint=deployment_fingerprint,
        deployment_measurement_path=str(measurement_path),
        deployment_measurement_sha256=measurement_sha,
        device_group_fingerprint=group_fingerprint,
        devices=normalized_devices,
        passed_functions=tuple(passed_functions),
        two_device_smoke_passed=two_passed,
        multi_device_stress_passed=multi_passed,
        completed_stress_calls=completed,
        public_api_smoke_passed=_passed(public_smoke),
    )


def _validate_mixed_stress_witness(
    assignments: Any,
    observations: Any,
    devices: Tuple[int, ...],
    functions: Tuple[str, ...],
) -> bool:
    """Recompute multi-round placement and fixed-worker agreement."""

    if not isinstance(assignments, list) or not isinstance(observations, list):
        return False
    if not functions or len(assignments) < 2 * len(functions):
        return False
    expected = []
    repetitions = set()
    try:
        for item in assignments:
            if not isinstance(item, Mapping):
                return False
            repetition = int(item["repetition"])
            function = str(item["function"])
            requested = int(item["requested_device"])
            assigned = tuple(int(value) for value in item["assigned_devices"])
            if (
                repetition < 0
                or function not in functions
                or requested not in devices
                or assigned != (requested,)
            ):
                return False
            repetitions.add(repetition)
            expected.append((repetition, function, requested))
        observed = [
            (str(item["function"]), int(item["device"]))
            for item in observations
            if isinstance(item, Mapping)
        ]
    except (KeyError, TypeError, ValueError):
        return False
    if len(observed) != len(observations) or len(observed) != len(assignments):
        return False
    if repetitions != set(range(len(repetitions))) or len(repetitions) < 2:
        return False
    for repetition in repetitions:
        if {
            function
            for observed_repetition, function, _ in expected
            if observed_repetition == repetition
        } != set(functions):
            return False
    if {requested for _, _, requested in expected} != set(devices):
        return False
    expected_counts: Dict[Tuple[str, int], int] = {}
    observed_counts: Dict[Tuple[str, int], int] = {}
    for _, function, device in expected:
        expected_counts[(function, device)] = (
            expected_counts.get((function, device), 0) + 1
        )
    for function, device in observed:
        observed_counts[(function, device)] = (
            observed_counts.get((function, device), 0) + 1
        )
    return expected_counts == observed_counts


def _require_raw(raw: Any, spec: SingleNativeBindingSpec, device: int) -> Mapping:
    if not isinstance(raw, Mapping):
        raise NativeBindingResultError("{} returned a non-mapping result".format(spec.binding))
    if raw.get("operation") != spec.name:
        raise NativeBindingResultError("native result operation does not match request")
    if int(raw.get("device", -1)) != int(device):
        raise NativeBindingResultError("native result came from an unreserved device")
    kernel = raw.get("kernel_seconds")
    if kernel is None or not math.isfinite(float(kernel)) or float(kernel) < 0.0:
        raise NativeBindingResultError("native result has invalid kernel timing")
    return raw


def _dense_values(raw: Mapping, key: str = "values_dense"):
    if key not in raw:
        raise NativeBindingResultError("native result is missing {}".format(key))
    import numpy as np

    values = np.asarray(raw[key])
    if values.ndim != 1:
        raise NativeBindingResultError("{} must be one-dimensional".format(key))
    return values


def _snapshot_node_domain(snapshot, node_count):
    if snapshot is None or bool(getattr(snapshot, "source_is_immutable", False)):
        return range(node_count)
    nodes = snapshot_public_nodes(snapshot)
    if len(nodes) != node_count:
        raise NativeBindingResultError("frozen public node order has an invalid length")
    return nodes


def _selected_nodes(arguments: Mapping, snapshot, node_count):
    nodes = arguments.get("nodes")
    if nodes is None:
        return _snapshot_node_domain(snapshot, node_count), False
    payload = None if snapshot is None else snapshot.require_native_payload()
    if isinstance(payload, FrozenNativeGraph):
        try:
            if nodes in payload.public_index:
                return (nodes,), True
        except TypeError:
            pass
    elif isinstance(nodes, int) and 0 <= nodes < node_count:
        return (nodes,), True
    return tuple(nodes), False


def _map_public_arguments(arguments: Mapping, snapshot):
    """Freeze iterables and map public labels to one snapshot's CSR indices."""

    public_arguments = dict(arguments)
    native_arguments = dict(arguments)
    payload = snapshot.require_native_payload()
    sources = public_arguments.get("sources")
    if sources is not None:
        frozen_sources = tuple(sources)
        public_arguments["sources"] = frozen_sources
        native_arguments["sources"] = frozen_sources
    nodes = public_arguments.get("nodes")
    if nodes is not None and not isinstance(nodes, int):
        if not (
            isinstance(payload, FrozenNativeGraph)
            and _is_public_scalar_node(nodes, payload.public_index)
        ):
            frozen_nodes = tuple(nodes)
            public_arguments["nodes"] = frozen_nodes
            native_arguments["nodes"] = frozen_nodes
    if not isinstance(payload, FrozenNativeGraph):
        return public_arguments, native_arguments
    index = payload.public_index

    def mapped(label, role):
        try:
            return int(index[label])
        except (KeyError, TypeError) as exc:
            raise NativeBindingError(
                "{} {!r} is not present in the submitted graph snapshot".format(
                    role, label
                )
            ) from exc

    for name in ("source", "target"):
        value = public_arguments.get(name)
        if value is not None:
            native_arguments[name] = mapped(value, name)

    sources = public_arguments.get("sources")
    if sources is not None:
        native_arguments["sources"] = tuple(
            mapped(source, "source") for source in sources
        )

    nodes = public_arguments.get("nodes")
    if nodes is not None:
        scalar = _is_public_scalar_node(nodes, index)
        if scalar:
            native_arguments["nodes"] = mapped(nodes, "node")
        else:
            native_arguments["nodes"] = tuple(
                mapped(node, "node") for node in nodes
            )
    return public_arguments, native_arguments


def _is_public_scalar_node(value, index):
    try:
        return value in index
    except TypeError:
        return False


def _mst_graph(raw: Mapping, graph: Any, weight: Any, snapshot=None):
    import easygraph as eg
    import numpy as np

    from easygraph.utils import gpu_eggpu_backend as backend

    sources = np.asarray(raw.get("mst_sources"), dtype=np.int32).reshape(-1)
    targets = np.asarray(raw.get("mst_targets"), dtype=np.int32).reshape(-1)
    weights = np.asarray(raw.get("mst_weights"), dtype=np.float64).reshape(-1)
    if not (len(sources) == len(targets) == len(weights)):
        raise NativeBindingResultError("native MST arrays have inconsistent lengths")
    tree = eg.Graph()
    node_count = (
        len(graph) if snapshot is None else int(snapshot.vertex_count)
    )
    if snapshot is not None and not bool(getattr(snapshot, "source_is_immutable", False)):
        public_nodes = _snapshot_node_domain(snapshot, node_count)
        if len(public_nodes) != node_count:
            raise NativeBindingResultError(
                "frozen MST node order has an invalid length"
            )
        tree.add_nodes_from(public_nodes)
        for source, target, value in zip(sources, targets, weights):
            tree.add_edge(
                public_nodes[int(source)],
                public_nodes[int(target)],
                **{str(weight): float(value)},
            )
        return tree
    identity = backend._IdentityIndexDict(node_count)
    tree._node = backend._ContiguousNodeDict(node_count)
    tree._adj = {}
    tree._node_index = identity
    tree._id = node_count
    tree.cache = {
        "index2node": identity,
        "node_index": identity,
        "_compact_zero_based_node_count": node_count,
        "_lazy_edge_count": len(sources),
        "_lazy_edge_arrays": (sources, targets, weights, weight),
    }
    return tree


def _adapt_public_result(
    spec: SingleNativeBindingSpec,
    raw: Mapping,
    graph: Any,
    arguments: Mapping,
    snapshot=None,
):
    import numpy as np

    node_count = len(graph) if snapshot is None else int(snapshot.vertex_count)
    if spec.result_kind is SingleResultKind.NODE_MAPPING:
        from easygraph.utils import gpu_eggpu_backend as backend

        values = _dense_values(raw)
        if spec.name == "pagerank":
            nodes = _snapshot_node_domain(snapshot, node_count)
            scalar = False
        else:
            nodes, scalar = _selected_nodes(arguments, snapshot, node_count)
        if len(values) != len(nodes):
            raise NativeBindingResultError("native node-value result has invalid length")
        if scalar:
            return float(values[0])
        return backend._DenseValueDict(nodes, values, dtype=float)
    if spec.result_kind is SingleResultKind.DENSE_SEQUENCE:
        return _dense_values(raw).astype(np.float64, copy=False).tolist()
    if spec.result_kind is SingleResultKind.COMPONENTS:
        from easygraph.utils import gpu_eggpu_backend as backend

        labels = _dense_values(raw, "integer_values_dense").astype(np.int32, copy=False)
        if len(labels) != node_count:
            raise NativeBindingResultError("native component label count is invalid")
        public_nodes = _snapshot_node_domain(snapshot, node_count)
        return backend._DenseComponentCollection(public_nodes, labels)
    if spec.result_kind is SingleResultKind.INTEGER_SEQUENCE:
        values = _dense_values(raw, "integer_values_dense").astype(np.int32, copy=False)
        if len(values) != node_count:
            raise NativeBindingResultError("native integer result has invalid length")
        return values
    if spec.result_kind in {
        SingleResultKind.SINGLE_SOURCE_DISTANCE,
        SingleResultKind.MULTI_SOURCE_DISTANCE,
    }:
        from easygraph.utils import gpu_eggpu_backend as backend

        if spec.result_kind is SingleResultKind.SINGLE_SOURCE_DISTANCE:
            sources = [arguments["source"]]
        else:
            sources = list(arguments["sources"])
        values = _dense_values(raw)
        expected = len(sources) * node_count
        if len(values) != expected:
            raise NativeBindingResultError("native distance matrix has invalid length")
        matrix = values.reshape((len(sources), node_count))
        public_nodes = _snapshot_node_domain(snapshot, node_count)
        result = backend._DenseMultiSourceSSSPDict(
            sources,
            public_nodes,
            matrix,
            result_name="EGMGPU {} result".format(spec.name),
        )
        if spec.result_kind is SingleResultKind.SINGLE_SOURCE_DISTANCE:
            return result[sources[0]]
        return result
    if spec.result_kind is SingleResultKind.MST_GRAPH:
        return _mst_graph(
            raw, graph, arguments.get("weight", "weight"), snapshot=snapshot
        )
    raise NativeBindingResultError("unknown public-result adapter")


class SingleNativeExecutor:
    """Controller adapter for one evidence-backed operation."""

    def __init__(
        self,
        spec: SingleNativeBindingSpec,
        public_function: Callable[..., Any],
        bindings: Any,
        evidence: SingleNativeEvidence,
        raw_result_sink: Optional[Callable[[Mapping], None]] = None,
    ) -> None:
        self.spec = spec
        self.public_function = public_function
        self.evidence = evidence
        self._signature = inspect.signature(public_function)
        self._binding = getattr(bindings, spec.binding, None)
        if not callable(self._binding):
            raise NativeBindingError("compiled extension lacks {}".format(spec.binding))
        self._sink = raw_result_sink

    @property
    def eligible_device_groups(self) -> Tuple[Tuple[int, ...], ...]:
        """Evidence-backed one-device groups accepted by this executor."""

        return tuple((device,) for device in self.evidence.devices)

    def __call__(self, context: NativeCallContext, args, kwargs):
        if context.mode not in {ExecutionMode.SINGLE, ExecutionMode.REQUEST_PARALLEL}:
            raise NativeBindingError("single dispatcher received a cooperative mode")
        if len(context.devices) != 1:
            raise NativeBindingError("single/request execution requires exactly one device")
        device = int(context.devices[0])
        if device not in self.evidence.devices:
            raise NativeBindingEvidenceError(
                "reserved device is outside retained native probe evidence"
            )
        snapshot = context.graph_snapshot
        if snapshot is None or not snapshot.native_payload_ready:
            raise NativeBindingError(
                "single/request native execution requires an immutable native snapshot"
            )
        try:
            bound = self._signature.bind(*args, **kwargs)
        except TypeError as exc:
            raise NativeBindingError(
                "arguments do not match {}: {}".format(self.spec.name, exc)
            ) from exc
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        graph = arguments.pop(next(iter(self._signature.parameters)))
        if arguments.get("n_workers") is not None:
            raise NativeBindingError("native EGMGPU execution does not accept n_workers")
        public_arguments, native_arguments = _map_public_arguments(
            arguments, snapshot
        )

        raw = self._binding(
            native_graph_handle(snapshot.require_native_payload()),
            self.spec.name,
            native_arguments,
            device,
        )
        raw = _require_raw(raw, self.spec, device)
        if self._sink is not None:
            self._sink(raw)
        return _adapt_public_result(
            self.spec, raw, graph, public_arguments, snapshot=snapshot
        )


class SingleNativeBindingRegistry:
    def __init__(self, specs: Iterable[SingleNativeBindingSpec]) -> None:
        self._status: Dict[str, SingleBindingStatus] = {}
        for spec in specs:
            if spec.name in self._status:
                raise ValueError("duplicate single native binding {}".format(spec.name))
            self._status[spec.name] = SingleBindingStatus(
                spec, SingleBindingReadiness.WIREABLE
            )

    def status(self, name: str) -> SingleBindingStatus:
        try:
            return self._status[str(name)]
        except KeyError as exc:
            raise KeyError("unknown single native binding {!r}".format(name)) from exc

    def record_evidence(self, evidence: SingleNativeEvidence) -> Tuple[str, ...]:
        available = []
        for name in evidence.passed_functions:
            current = self.status(name)
            self._status[name] = SingleBindingStatus(
                current.spec, SingleBindingReadiness.AVAILABLE, evidence
            )
            available.append(name)
        return tuple(sorted(available))

    def install_available(
        self,
        registrar: Any,
        functions: Mapping[str, Callable[..., Any]],
        bindings: Any,
        topology_fingerprint: str,
        raw_result_sink: Optional[Callable[[Mapping], None]] = None,
    ) -> Tuple[str, ...]:
        register_many = getattr(
            registrar, "register_native_executors_atomic", None
        )
        if not callable(register_many):
            register_many = getattr(
                registrar, "_register_native_executors_atomic", None
            )
        register = registrar if callable(registrar) else getattr(
            registrar, "register_native_executor", None
        )
        if not callable(register_many) and not callable(register):
            raise TypeError("registrar must accept native executor registrations")
        binding_path_value = getattr(bindings, "__file__", None)
        if not binding_path_value:
            raise NativeBindingEvidenceError(
                "native bindings do not expose a compiled extension path"
            )
        binding_path = Path(binding_path_value).expanduser().resolve(strict=True)
        binding_digest = _sha256(binding_path)
        pending = []
        for name, function in functions.items():
            status = self.status(name)
            if status.readiness is not SingleBindingReadiness.AVAILABLE:
                raise NativeBindingEvidenceError(
                    "{} is wireable but lacks passing retained evidence".format(name)
                )
            if status.evidence.topology_fingerprint != topology_fingerprint:
                raise NativeBindingEvidenceError(
                    "{} evidence was captured on a different topology".format(name)
                )
            retained_result = Path(status.evidence.result_path).resolve(strict=True)
            if _sha256(retained_result) != status.evidence.sha256:
                raise NativeBindingEvidenceError(
                    "{} retained evidence changed after it was loaded".format(name)
                )
            deployment_measurement = Path(
                status.evidence.deployment_measurement_path
            ).resolve(strict=True)
            if (
                _sha256(deployment_measurement)
                != status.evidence.deployment_measurement_sha256
            ):
                raise NativeBindingEvidenceError(
                    "{} retained deployment measurement changed after evidence "
                    "was loaded".format(name)
                )
            if binding_digest != status.evidence.extension_sha256:
                raise NativeBindingEvidenceError(
                    "{} evidence was captured with a different compiled extension".format(
                        name
                    )
                )
            executor = SingleNativeExecutor(
                status.spec,
                function,
                bindings,
                status.evidence,
                raw_result_sink=raw_result_sink,
            )
            pending.append((name, function, executor))
        registrations = tuple(
            registration
            for _, function, executor in pending
            for registration in (
                (function, ExecutionMode.SINGLE, executor),
                (function, ExecutionMode.REQUEST_PARALLEL, executor),
            )
        )
        if callable(register_many):
            register_many(registrations)
        else:
            for registration in registrations:
                register(*registration)
        return tuple(name for name, _, _ in pending)

    def matrix(self) -> Dict[str, str]:
        return {
            name: status.readiness.value
            for name, status in sorted(self._status.items())
        }


def default_single_native_specs() -> Tuple[SingleNativeBindingSpec, ...]:
    mapping = (
        ("pagerank", SingleResultKind.NODE_MAPPING),
        ("mst", SingleResultKind.MST_GRAPH),
        ("lcc", SingleResultKind.NODE_MAPPING),
        ("wcc", SingleResultKind.COMPONENTS),
        ("scc", SingleResultKind.COMPONENTS),
        ("bfs", SingleResultKind.SINGLE_SOURCE_DISTANCE),
        ("dijkstra", SingleResultKind.SINGLE_SOURCE_DISTANCE),
        ("bellman_ford", SingleResultKind.MULTI_SOURCE_DISTANCE),
        ("sssp", SingleResultKind.MULTI_SOURCE_DISTANCE),
        ("kcore", SingleResultKind.INTEGER_SEQUENCE),
        ("bc", SingleResultKind.DENSE_SEQUENCE),
        ("closeness", SingleResultKind.DENSE_SEQUENCE),
        ("effective_size", SingleResultKind.NODE_MAPPING),
        ("efficiency", SingleResultKind.NODE_MAPPING),
        ("constraint", SingleResultKind.NODE_MAPPING),
        ("hierarchy", SingleResultKind.NODE_MAPPING),
    )
    return tuple(SingleNativeBindingSpec(name, kind) for name, kind in mapping)


def default_single_native_registry() -> SingleNativeBindingRegistry:
    return SingleNativeBindingRegistry(default_single_native_specs())


def default_single_public_functions() -> Dict[str, Callable[..., Any]]:
    """Resolve the 16 exact EasyGraph callables used as scheduler keys.

    Native executor lookup is identity based.  Keeping this mapping in one
    place prevents aliases such as ``BFS`` or ``Dijkstra`` from silently
    registering a different callable than the documented public API.
    """

    import easygraph as eg

    return {
        "pagerank": eg.pagerank,
        "mst": eg.minimum_spanning_tree,
        "lcc": eg.clustering,
        "wcc": eg.connected_components,
        "scc": eg.strongly_connected_components,
        "bfs": eg.bfs,
        "dijkstra": eg.single_source_dijkstra,
        "bellman_ford": eg.multi_source_bellman_ford,
        "sssp": eg.multi_source_dijkstra,
        "kcore": eg.k_core,
        "bc": eg.betweenness_centrality,
        "closeness": eg.closeness_centrality,
        "effective_size": eg.effective_size,
        "efficiency": eg.efficiency,
        "constraint": eg.constraint,
        "hierarchy": eg.hierarchy,
    }


def register_probed_single_bindings(
    functions: Mapping[str, Callable[..., Any]],
    evidence: Any,
    *,
    topology_fingerprint: str,
    bindings: Any = None,
    registrar: Any = None,
    raw_result_sink: Optional[Callable[[Mapping], None]] = None,
) -> Tuple[str, ...]:
    """Explicitly register only functions proven by retained GPU evidence."""

    if bindings is None:
        import cpp_easygraph as bindings
    if registrar is None:
        # The module-level helper validates the complete batch before making
        # any route visible to a concurrently submitting thread.
        from . import api as registrar
    loaded = (
        evidence
        if isinstance(evidence, SingleNativeEvidence)
        else load_single_native_evidence(evidence)
    )
    registry = default_single_native_registry()
    registry.record_evidence(loaded)
    if set(functions).difference(loaded.passed_functions):
        raise NativeBindingEvidenceError(
            "requested functions are not all present in passing evidence"
        )
    return registry.install_available(
        registrar,
        functions,
        bindings,
        topology_fingerprint,
        raw_result_sink=raw_result_sink,
    )


__all__ = [
    "SINGLE_NATIVE_BINDING",
    "SINGLE_NATIVE_PROBE_SCHEMA",
    "SingleBindingReadiness",
    "SingleNativeBindingRegistry",
    "SingleNativeBindingSpec",
    "SingleNativeEvidence",
    "SingleNativeExecutor",
    "SingleResultKind",
    "default_single_native_registry",
    "default_single_native_specs",
    "default_single_public_functions",
    "load_single_native_evidence",
    "register_probed_single_bindings",
]
