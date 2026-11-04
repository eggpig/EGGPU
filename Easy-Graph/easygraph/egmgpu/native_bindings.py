"""Evidence-gated adapters for implemented replicated native bindings.

This module does not register an executor at import time.  The six binding
descriptions below are merely *wireable*.  A binding becomes eligible for
registration only after :func:`load_native_probe_evidence` validates a raw
GPU correctness result and its checksum.  Registration is still explicit and
does not start ``RuntimeManager``; the public API remains lazy until submit.

Python arguments and final public-result conversion happen on a controller
thread.  Each pybind entry point is synchronous and is responsible for
releasing the GIL before it submits native-only tasks to the C++ workers.
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

from .types import ExecutionMode
from .types import NativeCallContext


NATIVE_PROBE_SCHEMA = "egmgpu.native-binding-probe/v3"
_REPLICATED_RTOL = 2.0e-7
_REPLICATED_ATOL = 2.0e-9


class NativeBindingError(RuntimeError):
    """Base error for native binding wiring and result adaptation."""


class NativeBindingEvidenceError(NativeBindingError):
    """Raw probe evidence is absent, corrupt, mismatched, or unsuccessful."""


class NativeBindingResultError(NativeBindingError):
    """A native binding returned an invalid raw result."""


class NativeBindingReadiness(str, Enum):
    WIREABLE = "wireable"
    AVAILABLE = "available"


class PublicResultKind(str, Enum):
    DENSE_SEQUENCE = "dense_sequence"
    NODE_MAPPING = "node_mapping"


@dataclass(frozen=True)
class NativeBindingSpec:
    """One audited pybind path; this does not imply runtime availability."""

    name: str
    binding: str
    result_kind: PublicResultKind
    mode: ExecutionMode = ExecutionMode.REPLICATED

    def __post_init__(self) -> None:
        if not self.name or not self.binding:
            raise ValueError("native binding name and symbol must be non-empty")
        if self.mode is not ExecutionMode.REPLICATED:
            raise ValueError("the current native binding set is replicated-only")


@dataclass(frozen=True)
class NativeBindingProbeEvidence:
    """Validated identity of one retained raw GPU correctness result."""

    function: str
    binding: str
    mode: ExecutionMode
    devices: Tuple[int, ...]
    deployment_fingerprint: str
    deployment_measurement_path: str
    deployment_measurement_sha256: str
    device_group_fingerprint: str
    result_path: str
    sha256: str
    extension_sha256: str
    passed: bool
    completed_calls: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ExecutionMode.parse(self.mode))
        devices = tuple(int(device) for device in self.devices)
        if len(devices) < 2 or len(set(devices)) != len(devices):
            raise NativeBindingEvidenceError(
                "replicated probe evidence requires at least two unique devices"
            )
        devices = tuple(sorted(devices))
        object.__setattr__(self, "devices", devices)
        if not self.deployment_fingerprint:
            raise NativeBindingEvidenceError(
                "probe evidence requires a deployment fingerprint"
            )
        digest = str(self.sha256).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise NativeBindingEvidenceError("probe evidence has an invalid SHA-256")
        object.__setattr__(self, "sha256", digest)
        extension_digest = str(self.extension_sha256).lower()
        if len(extension_digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in extension_digest
        ):
            raise NativeBindingEvidenceError(
                "probe evidence has an invalid extension SHA-256"
            )
        object.__setattr__(self, "extension_sha256", extension_digest)
        if not self.deployment_measurement_path:
            raise NativeBindingEvidenceError(
                "probe evidence requires a deployment measurement path"
            )
        for label, value in (
            ("deployment measurement", self.deployment_measurement_sha256),
            ("device group", self.device_group_fingerprint),
        ):
            candidate = str(value).lower()
            if len(candidate) != 64 or any(
                character not in "0123456789abcdef" for character in candidate
            ):
                raise NativeBindingEvidenceError(
                    "probe evidence has an invalid {} SHA-256".format(label)
                )
        if int(self.completed_calls) < 1:
            raise NativeBindingEvidenceError(
                "probe evidence must contain at least one completed call"
            )
        object.__setattr__(self, "completed_calls", int(self.completed_calls))

    @property
    def topology_fingerprint(self) -> str:
        return self.deployment_fingerprint


@dataclass(frozen=True)
class NativeRawResult:
    """Validated timing metadata plus the Python-compatible public result."""

    function: str
    binding: str
    execution_mode: ExecutionMode
    devices_used: Tuple[int, ...]
    kernel_seconds: float
    per_device_kernel_seconds: Tuple[float, ...]
    public_result: Any
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class NativeBindingStatus:
    spec: NativeBindingSpec
    readiness: NativeBindingReadiness
    evidence: Optional[NativeBindingProbeEvidence] = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _extension_sha256(bindings: Any) -> str:
    """Return the immutable identity of the currently loaded extension."""

    injected = getattr(bindings, "__egmgpu_extension_sha256__", None)
    if injected is not None:
        digest = str(injected).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise NativeBindingEvidenceError(
                "loaded extension has an invalid injected SHA-256"
            )
        return digest
    path = getattr(bindings, "__file__", None)
    if not path:
        raise NativeBindingEvidenceError(
            "compiled bindings do not expose an extension file identity"
        )
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise NativeBindingEvidenceError(
            "compiled extension path is not a file"
        )
    return _file_sha256(resolved)


def load_native_probe_evidence(path: Any) -> NativeBindingProbeEvidence:
    """Load strict retained evidence without starting or probing a GPU."""

    result_path = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeBindingEvidenceError(
            "cannot parse native probe result {}: {}".format(result_path, exc)
        ) from exc
    if not isinstance(payload, dict):
        raise NativeBindingEvidenceError("native probe result must be a JSON object")
    if payload.get("schema") != NATIVE_PROBE_SCHEMA:
        raise NativeBindingEvidenceError(
            "unsupported native probe schema {!r}".format(payload.get("schema"))
        )
    required = (
        "function",
        "binding",
        "mode",
        "devices",
        "deployment",
        "device_group",
        "outcome",
        "correctness_passed",
        "completed_calls",
        "extension_sha256",
        "comparator",
        "cases",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise NativeBindingEvidenceError(
            "native probe result is missing {}".format(", ".join(missing))
        )
    devices = tuple(sorted(int(device) for device in payload["devices"]))
    deployment_path, deployment_sha, deployment_fingerprint = (
        _validate_replicated_deployment(result_path, payload["deployment"], devices)
    )
    group_fingerprint = _validate_replicated_group(
        payload["device_group"], devices, deployment_path
    )
    computed_calls = _validate_replicated_numeric_witness(
        str(payload["function"]), devices, payload["comparator"], payload["cases"]
    )
    if int(payload["completed_calls"]) != computed_calls:
        raise NativeBindingEvidenceError(
            "replicated completed_calls differs from its raw witnesses"
        )
    passed = bool(
        str(payload["outcome"]).lower() == "ok"
        and payload["correctness_passed"] is True
        and computed_calls > 0
    )
    return NativeBindingProbeEvidence(
        function=str(payload["function"]),
        binding=str(payload["binding"]),
        mode=ExecutionMode.parse(payload["mode"]),
        devices=devices,
        deployment_fingerprint=deployment_fingerprint,
        deployment_measurement_path=str(deployment_path),
        deployment_measurement_sha256=deployment_sha,
        device_group_fingerprint=group_fingerprint,
        result_path=str(result_path),
        sha256=_file_sha256(result_path),
        extension_sha256=str(payload["extension_sha256"]),
        passed=passed,
        completed_calls=computed_calls,
    )


def _valid_digest(value: Any, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise NativeBindingEvidenceError("{} is not a SHA-256".format(label))
    return digest


def _canonical_pci(value: Any) -> str:
    fields = str(value).strip().upper().split(":")
    return ":".join(fields[-2:]) if len(fields) >= 2 else str(value).strip().upper()


def _canonical_group(value: Any, devices: Tuple[int, ...]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeBindingEvidenceError("replicated device-group identity is missing")
    raw_devices = value.get("devices")
    peer = value.get("peer_access")
    if not isinstance(raw_devices, list) or not isinstance(peer, list):
        raise NativeBindingEvidenceError("replicated device-group identity is incomplete")
    normalized = []
    for item in raw_devices:
        if not isinstance(item, Mapping):
            raise NativeBindingEvidenceError("replicated GPU identity is invalid")
        memory = item.get("total_global_memory")
        if type(memory) is not int or memory <= 0:
            raise NativeBindingEvidenceError(
                "replicated allocatable memory must be a positive integer"
            )
        normalized.append(
            {
                "device_id": int(item.get("device_id", -1)),
                "name": str(item.get("name", "")),
                "pci_bus_id": _canonical_pci(item.get("pci_bus_id", "")),
                "compute_capability": str(item.get("compute_capability", "")),
                "total_global_memory": memory,
            }
        )
    if tuple(item["device_id"] for item in normalized) != devices:
        raise NativeBindingEvidenceError("replicated device identity differs from devices")
    if len(peer) != len(devices) or any(
        not isinstance(row, list) or len(row) != len(devices) for row in peer
    ):
        raise NativeBindingEvidenceError("replicated P2P matrix has invalid dimensions")
    return {
        "devices": normalized,
        "peer_access": [[bool(value) for value in row] for row in peer],
    }


def _canonical_measured_group(measurement: Mapping, devices: Tuple[int, ...]):
    topology = measurement.get("topology")
    if not isinstance(topology, Mapping):
        raise NativeBindingEvidenceError("deployment record has no topology")
    raw = topology.get("gpus")
    p2p = topology.get("p2p_access")
    if not isinstance(raw, list) or not isinstance(p2p, Mapping):
        raise NativeBindingEvidenceError("deployment record lacks GPU/P2P identity")
    by_id = {int(item["index"]): item for item in raw if isinstance(item, Mapping)}
    if any(device not in by_id for device in devices):
        raise NativeBindingEvidenceError("replicated devices are absent from deployment")
    for device in devices:
        memory_mib = by_id[device].get("memory_total_mib")
        if type(memory_mib) is not int or memory_mib <= 0:
            raise NativeBindingEvidenceError(
                "deployment GPU nominal memory must be a positive integer"
            )
    for source in devices:
        for target in devices:
            if source != target and "{}->{}".format(source, target) not in p2p:
                raise NativeBindingEvidenceError(
                    "deployment record lacks a selected P2P fact"
                )
    return {
        "devices": [
            {
                "device_id": device,
                "name": str(by_id[device].get("name", "")),
                "pci_bus_id": _canonical_pci(by_id[device].get("pci_bus_id", "")),
                "compute_capability": str(by_id[device].get("compute_capability", "")),
                "total_global_memory": by_id[device]["memory_total_mib"]
                * (1 << 20),
            }
            for device in devices
        ],
        "peer_access": [
            [
                True
                if source == target
                else bool(p2p.get("{}->{}".format(source, target), False))
                for target in devices
            ]
            for source in devices
        ],
    }


def _validate_group_against_measurement(
    group: Mapping, measured: Mapping
) -> None:
    """Bind replicated native identity to exact deployment hardware facts."""

    group_devices = group["devices"]
    measured_devices = measured["devices"]
    if len(group_devices) != len(measured_devices):
        raise NativeBindingEvidenceError(
            "device-group identity differs from retained deployment"
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
                    "device-group identity differs from retained deployment"
                )
        # CUDA reports allocatable bytes after driver/ECC reserve; nvidia-smi
        # reports nominal MiB.  Use the partitioned binding's exact GiB class
        # rule and keep the original CUDA byte value in the signed evidence.
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
                "device-group memory class differs from retained deployment"
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
            "device-group P2P identity differs from retained deployment"
        )


def _mapping_digest(value: Mapping) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_replicated_deployment(result_path, value, devices):
    if not isinstance(value, Mapping):
        raise NativeBindingEvidenceError("replicated deployment identity is missing")
    relative = Path(str(value.get("path", "")))
    if relative.is_absolute() or not str(relative):
        raise NativeBindingEvidenceError(
            "replicated deployment path must be bundle-relative"
        )
    root = result_path.parent.resolve(strict=True)
    retained = (root / relative).resolve(strict=True)
    try:
        retained.relative_to(root)
    except ValueError as exc:
        raise NativeBindingEvidenceError(
            "replicated deployment path escapes the evidence bundle"
        ) from exc
    checksum = _valid_digest(value.get("sha256"), "deployment checksum")
    if _file_sha256(retained) != checksum:
        raise NativeBindingEvidenceError("replicated deployment checksum changed")
    try:
        measurement = json.loads(retained.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeBindingEvidenceError("cannot parse replicated deployment") from exc
    fingerprint = _valid_digest(
        value.get("topology_fingerprint"), "deployment fingerprint"
    )
    if str(measurement.get("topology_fingerprint", "")).lower() != fingerprint:
        raise NativeBindingEvidenceError(
            "deployment fingerprint differs from retained measurement"
        )
    return retained, checksum, fingerprint


def _validate_replicated_group(value, devices, deployment_path):
    if not isinstance(value, Mapping):
        raise NativeBindingEvidenceError("replicated device group is missing")
    identity = _canonical_group(value.get("identity"), devices)
    fingerprint = _valid_digest(value.get("fingerprint"), "device-group fingerprint")
    if _mapping_digest(identity) != fingerprint:
        raise NativeBindingEvidenceError("device-group fingerprint is inconsistent")
    measurement = json.loads(deployment_path.read_text(encoding="utf-8"))
    _validate_group_against_measurement(
        identity, _canonical_measured_group(measurement, devices)
    )
    return fingerprint


def _numeric_token(value: Any) -> float:
    if value == "nan":
        return float("nan")
    if value == "+inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeBindingEvidenceError("replicated numeric witness has a non-number")
    return float(value)


def _allclose_witness(expected, observed, rtol, atol):
    if len(expected) != len(observed):
        return False, float("inf")
    maximum = 0.0
    for left, right in zip(expected, observed):
        if math.isnan(left) or math.isnan(right):
            if not (math.isnan(left) and math.isnan(right)):
                return False, float("inf")
            continue
        if math.isinf(left) or math.isinf(right):
            if left != right:
                return False, float("inf")
            continue
        error = abs(left - right)
        maximum = max(maximum, error)
        if error > atol + rtol * abs(left):
            return False, maximum
    return True, maximum


def _validate_replicated_numeric_witness(function, devices, comparator, cases):
    if function not in {"bc", "closeness", "effective_size", "efficiency", "constraint", "hierarchy"}:
        raise NativeBindingEvidenceError("unknown replicated function witness")
    if not isinstance(comparator, Mapping) or comparator.get("kind") != "numeric_allclose":
        raise NativeBindingEvidenceError("replicated comparator must be numeric_allclose")
    try:
        rtol = float(comparator["rtol"])
        atol = float(comparator["atol"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeBindingEvidenceError("replicated comparator is incomplete") from exc
    if rtol != _REPLICATED_RTOL or atol != _REPLICATED_ATOL:
        raise NativeBindingEvidenceError("replicated comparator uses non-fixed tolerance")
    if not isinstance(cases, list) or not cases:
        raise NativeBindingEvidenceError("replicated evidence requires case witnesses")
    identifiers = set()
    calls = 0
    for case in cases:
        if not isinstance(case, Mapping):
            raise NativeBindingEvidenceError("replicated case witness is invalid")
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in identifiers:
            raise NativeBindingEvidenceError("replicated case IDs must be unique")
        identifiers.add(case_id)
        domain = case.get("domain")
        if not isinstance(domain, Mapping) or domain.get("kind") != "dense_zero_based":
            raise NativeBindingEvidenceError("replicated case domain is invalid")
        count = int(domain.get("vertex_count", -1))
        expected = tuple(_numeric_token(value) for value in case.get("reference_values", ()))
        single = tuple(_numeric_token(value) for value in case.get("single_device_values", ()))
        if count < 1 or len(expected) != count or len(single) != count:
            raise NativeBindingEvidenceError("replicated witness shape is invalid")
        single_ok, single_error = _allclose_witness(expected, single, rtol, atol)
        if not single_ok or not math.isclose(
            float(case.get("single_max_abs_error", float("nan"))),
            single_error,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise NativeBindingEvidenceError(
                "single-device reference comparison does not match its witness"
            )
        if function in {"bc", "closeness"}:
            if case.get("reference_source") != "networkx":
                raise NativeBindingEvidenceError(
                    "centrality evidence requires an independent NetworkX reference"
                )
        elif case.get("reference_source") != "single_device_native":
            raise NativeBindingEvidenceError(
                "structural evidence requires its declared single-device reference"
            )
        repeats = case.get("multi_device_repeats")
        if not isinstance(repeats, list) or not repeats:
            raise NativeBindingEvidenceError("replicated case has no multi-device repeats")
        seen_repeats = set()
        for repeat in repeats:
            if not isinstance(repeat, Mapping):
                raise NativeBindingEvidenceError(
                    "replicated multi-device repeat witness is invalid"
                )
            try:
                index = int(repeat.get("repeat", -1))
                repeat_devices = tuple(
                    int(value) for value in repeat.get("devices", ())
                )
            except (TypeError, ValueError) as exc:
                raise NativeBindingEvidenceError(
                    "replicated repeat identity or device group is invalid"
                ) from exc
            if index < 0 or index in seen_repeats or repeat_devices != devices:
                raise NativeBindingEvidenceError(
                    "replicated repeat identity or device group is invalid"
                )
            seen_repeats.add(index)
            observed = tuple(
                _numeric_token(value) for value in repeat.get("observed_values", ())
            )
            okay, error = _allclose_witness(expected, observed, rtol, atol)
            if not okay or len(observed) != count or not math.isclose(
                float(repeat.get("max_abs_error", float("nan"))),
                error,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise NativeBindingEvidenceError(
                    "multi-device comparison does not match its raw witness"
                )
            calls += 1
    return calls


def _validate_evidence_file(evidence: NativeBindingProbeEvidence) -> None:
    path = Path(evidence.result_path).expanduser().resolve(strict=True)
    actual = _file_sha256(path)
    if actual != evidence.sha256:
        raise NativeBindingEvidenceError(
            "native probe result checksum changed: {}".format(path)
        )
    reloaded = load_native_probe_evidence(path)
    if reloaded != evidence:
        raise NativeBindingEvidenceError(
            "injected evidence does not match its retained raw result"
        )


class NativeRawResultAdapter:
    """Validate native output and recover the unchanged public return type."""

    def adapt(
        self,
        spec: NativeBindingSpec,
        raw: Any,
        reserved_devices: Iterable[int],
    ) -> NativeRawResult:
        if not isinstance(raw, Mapping):
            raise NativeBindingResultError(
                "{} returned a non-mapping raw result".format(spec.binding)
            )
        required = ("kernel_seconds",)
        missing = [name for name in required if name not in raw]
        if missing:
            raise NativeBindingResultError(
                "{} raw result is missing {}".format(
                    spec.binding, ", ".join(missing)
                )
            )
        try:
            kernel_seconds = float(raw["kernel_seconds"])
            per_device = tuple(
                float(value) for value in raw.get("per_device_kernel_seconds", ())
            )
            reserved = tuple(int(device) for device in reserved_devices)
            devices_used = tuple(
                int(value) for value in raw.get("devices_used", reserved)
            )
        except (TypeError, ValueError) as exc:
            raise NativeBindingResultError(
                "{} returned invalid timing metadata".format(spec.binding)
            ) from exc
        if kernel_seconds < 0 or any(value < 0 for value in per_device):
            raise NativeBindingResultError("native kernel times must be non-negative")
        if per_device and len(per_device) != len(devices_used):
            raise NativeBindingResultError(
                "per-device times and devices_used have different lengths"
            )
        if any(device not in reserved for device in devices_used):
            raise NativeBindingResultError(
                "native result names a device outside its scheduler reservation"
            )
        if (
            tuple(sorted(devices_used)) != devices_used
            or len(set(devices_used)) != len(devices_used)
        ):
            raise NativeBindingResultError(
                "native result devices must be unique and sorted"
            )

        if spec.result_kind is PublicResultKind.DENSE_SEQUENCE:
            values = raw.get("values_dense")
            if values is None:
                raise NativeBindingResultError("native dense result has no values_dense")
            if isinstance(values, (str, bytes, Mapping)):
                raise NativeBindingResultError("values_dense must be an ordered sequence")
            if hasattr(values, "tolist"):
                public_result = values.tolist()
                if isinstance(public_result, (str, bytes, Mapping)):
                    raise NativeBindingResultError(
                        "values_dense must be an ordered sequence"
                    )
                try:
                    for value in public_result:
                        float(value)
                except (TypeError, ValueError) as exc:
                    raise NativeBindingResultError(
                        "values_dense contains a non-numeric value"
                    ) from exc
            else:
                try:
                    public_result = [float(value) for value in values]
                except TypeError as exc:
                    raise NativeBindingResultError(
                        "values_dense is not iterable"
                    ) from exc
                except ValueError as exc:
                    raise NativeBindingResultError(
                        "values_dense contains a non-numeric value"
                    ) from exc
        elif spec.result_kind is PublicResultKind.NODE_MAPPING:
            values = raw.get("values")
            if not isinstance(values, Mapping):
                raise NativeBindingResultError("native structural result has no mapping")
            # The native binding already creates the exact public mapping.
            # Validate scalar convertibility without materializing a second
            # dict or converting its optional dense diagnostics.
            try:
                for value in values.values():
                    float(value)
            except (TypeError, ValueError) as exc:
                raise NativeBindingResultError(
                    "native structural mapping contains a non-numeric value"
                ) from exc
            public_result = values
        else:  # pragma: no cover - enum exhaustiveness guard.
            raise AssertionError("unhandled native result kind")

        return NativeRawResult(
            function=spec.name,
            binding=spec.binding,
            execution_mode=spec.mode,
            devices_used=devices_used,
            kernel_seconds=kernel_seconds,
            per_device_kernel_seconds=per_device,
            public_result=public_result,
            raw=raw,
        )


class ReplicatedNativeExecutor:
    """Controller adapter for one synchronous replicated pybind entry point."""

    def __init__(
        self,
        spec: NativeBindingSpec,
        public_function: Callable[..., Any],
        bindings: Any,
        evidence: NativeBindingProbeEvidence,
        raw_result_sink: Optional[Callable[[NativeRawResult], None]] = None,
    ) -> None:
        if not callable(public_function):
            raise TypeError("public_function must be callable")
        binding = getattr(bindings, spec.binding, None)
        if not callable(binding):
            raise NativeBindingError(
                "compiled extension has no callable {}".format(spec.binding)
            )
        if raw_result_sink is not None and not callable(raw_result_sink):
            raise TypeError("raw_result_sink must be callable")
        self.spec = spec
        self.public_function = public_function
        self.evidence = evidence
        self._extension_sha256 = _extension_sha256(bindings)
        if self._extension_sha256 != evidence.extension_sha256:
            raise NativeBindingEvidenceError(
                "loaded extension SHA-256 has no retained evidence for {}".format(
                    spec.name
                )
            )
        self._signature = inspect.signature(public_function)
        self._binding = binding
        self._sink = raw_result_sink
        self._result_adapter = NativeRawResultAdapter()

    @property
    def eligible_device_groups(self) -> Tuple[Tuple[int, ...], ...]:
        """Exact retained-evidence groups the scheduler may reserve."""

        return (self.evidence.devices,)

    def __call__(
        self,
        context: NativeCallContext,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if context.mode is not ExecutionMode.REPLICATED:
            raise NativeBindingError(
                "{} requires replicated execution".format(self.spec.name)
            )
        if context.graph_snapshot is None:
            raise NativeBindingError("replicated native execution requires a snapshot")
        if not bool(
            getattr(context.graph_snapshot, "source_is_immutable", False)
        ):
            raise NativeBindingError(
                "replicated execution currently accepts only an immutable bulk "
                "CSR graph; mutable EasyGraph snapshots support single and "
                "request-parallel execution"
            )
        if tuple(context.devices) != self.evidence.devices:
            raise NativeBindingEvidenceError(
                "the reserved device group does not match retained probe evidence"
            )
        cpp_graph = context.graph_snapshot.require_native_payload()
        try:
            bound = self._signature.bind(*args, **kwargs)
        except TypeError as exc:
            raise NativeBindingError(
                "arguments do not match {}: {}".format(
                    getattr(self.public_function, "__name__", self.spec.name), exc
                )
            ) from exc
        bound.apply_defaults()
        arguments = bound.arguments
        if arguments.get("n_workers") is not None:
            raise NativeBindingError(
                "replicated native execution does not accept n_workers"
            )

        devices = list(context.devices)
        if self.spec.name == "bc":
            raw = self._binding(
                cpp_graph,
                arguments.get("sources"),
                arguments.get("weight"),
                bool(arguments.get("normalized", True)),
                bool(arguments.get("endpoints", False)),
                devices,
            )
        elif self.spec.name == "closeness":
            raw = self._binding(
                cpp_graph,
                arguments.get("sources"),
                arguments.get("weight"),
                devices,
            )
        else:
            raw = self._binding(
                cpp_graph,
                arguments.get("nodes"),
                arguments.get("weight"),
                devices,
            )
        adapted = self._result_adapter.adapt(self.spec, raw, context.devices)
        if self._sink is not None:
            self._sink(adapted)
        return adapted.public_result


class ReplicatedNativeBindingRegistry:
    """Availability state for six implemented replicated native bindings."""

    def __init__(self, specs: Iterable[NativeBindingSpec]) -> None:
        self._status: Dict[str, NativeBindingStatus] = {}
        for spec in specs:
            if spec.name in self._status:
                raise ValueError("duplicate native binding {}".format(spec.name))
            self._status[spec.name] = NativeBindingStatus(
                spec=spec,
                readiness=NativeBindingReadiness.WIREABLE,
            )

    def status(self, name: str) -> NativeBindingStatus:
        try:
            return self._status[str(name)]
        except KeyError as exc:
            raise KeyError("unknown replicated native binding {!r}".format(name)) from exc

    def record_probe(
        self,
        name: str,
        evidence: NativeBindingProbeEvidence,
    ) -> NativeBindingStatus:
        current = self.status(name)
        _validate_evidence_file(evidence)
        if evidence.function != current.spec.name:
            raise NativeBindingEvidenceError("probe function does not match binding spec")
        if evidence.binding != current.spec.binding:
            raise NativeBindingEvidenceError("probe symbol does not match binding spec")
        if evidence.mode is not current.spec.mode:
            raise NativeBindingEvidenceError("probe mode does not match binding spec")
        readiness = (
            NativeBindingReadiness.AVAILABLE
            if evidence.passed
            else NativeBindingReadiness.WIREABLE
        )
        updated = NativeBindingStatus(current.spec, readiness, evidence)
        self._status[str(name)] = updated
        return updated

    def install_available(
        self,
        registrar: Any,
        functions: Mapping[str, Callable[..., Any]],
        bindings: Any,
        topology_fingerprint: str,
        raw_result_sink: Optional[Callable[[NativeRawResult], None]] = None,
    ) -> Tuple[str, ...]:
        register = (
            registrar
            if callable(registrar)
            else getattr(registrar, "register_native_executor", None)
        )
        if not callable(register):
            raise TypeError("registrar must accept native executor registrations")
        if not topology_fingerprint:
            raise NativeBindingEvidenceError(
                "registration requires the current topology fingerprint"
            )
        extension_sha256 = _extension_sha256(bindings)
        pending = []
        for name, function in functions.items():
            status = self.status(name)
            if status.readiness is not NativeBindingReadiness.AVAILABLE:
                raise NativeBindingEvidenceError(
                    "{} is wireable but has no passing retained GPU probe".format(name)
                )
            _validate_evidence_file(status.evidence)
            if status.evidence.topology_fingerprint != topology_fingerprint:
                raise NativeBindingEvidenceError(
                    "{} probe evidence was captured on a different topology".format(
                        name
                    )
                )
            if status.evidence.extension_sha256 != extension_sha256:
                raise NativeBindingEvidenceError(
                    "{} loaded extension SHA-256 has no retained evidence".format(
                        name
                    )
                )
            pending.append(
                (
                    name,
                    function,
                    ReplicatedNativeExecutor(
                        status.spec,
                        function,
                        bindings,
                        status.evidence,
                        raw_result_sink=raw_result_sink,
                    ),
                )
            )
        # Everything that can fail without mutation is validated above.
        for _, function, executor in pending:
            register(function, ExecutionMode.REPLICATED, executor)
        return tuple(name for name, _, _ in pending)

    def matrix(self) -> Dict[str, str]:
        return {
            name: status.readiness.value
            for name, status in sorted(self._status.items())
        }


def default_replicated_native_specs() -> Tuple[NativeBindingSpec, ...]:
    return (
        NativeBindingSpec(
            "bc",
            "cpp_egmgpu_replicated_betweenness",
            PublicResultKind.DENSE_SEQUENCE,
        ),
        NativeBindingSpec(
            "closeness",
            "cpp_egmgpu_replicated_closeness",
            PublicResultKind.DENSE_SEQUENCE,
        ),
        *tuple(
            NativeBindingSpec(
                name,
                "cpp_egmgpu_replicated_{}".format(name),
                PublicResultKind.NODE_MAPPING,
            )
            for name in ("effective_size", "efficiency", "constraint", "hierarchy")
        ),
    )


def default_replicated_native_registry() -> ReplicatedNativeBindingRegistry:
    return ReplicatedNativeBindingRegistry(default_replicated_native_specs())


def register_probed_replicated_bindings(
    functions: Mapping[str, Callable[..., Any]],
    evidence: Mapping[str, Any],
    *,
    topology_fingerprint: str,
    bindings: Any = None,
    registrar: Any = None,
    raw_result_sink: Optional[Callable[[NativeRawResult], None]] = None,
) -> Tuple[str, ...]:
    """Validate evidence and lazily register selected production bindings.

    ``evidence`` values may be retained raw-result paths or already loaded
    :class:`NativeBindingProbeEvidence` objects.  No GPU probe is synthesized
    here.  If ``registrar`` is omitted, registrations enter the lazy public
    API and still do not start workers until a request is submitted.
    """

    if bindings is None:
        import cpp_easygraph as bindings
    if registrar is None:
        from .api import register_native_executor as registrar

    registry = default_replicated_native_registry()
    if set(functions) != set(evidence):
        raise NativeBindingEvidenceError(
            "functions and evidence must name the same native bindings"
        )
    for name, value in evidence.items():
        loaded = (
            value
            if isinstance(value, NativeBindingProbeEvidence)
            else load_native_probe_evidence(value)
        )
        registry.record_probe(name, loaded)
    return registry.install_available(
        registrar,
        functions,
        bindings,
        topology_fingerprint,
        raw_result_sink=raw_result_sink,
    )
