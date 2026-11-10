"""Evidence-gated production adapters for native partitioned bindings.

Importing this module performs no GPU probe, graph load, executor
registration, or runtime startup.  A binding is initially ``WIREABLE``.  It
becomes ``AVAILABLE`` only after a normalized evidence index and every raw
file referenced by that index pass checksum and capability validation.

The production adapters accept only :class:`ImmutableMultiPartitionGraph`
objects returned by :func:`load_partitioned_graph`.  Consequently a regular
Python graph can never be passed accidentally to a partitioned CUDA binding.
The pybind entry point remains responsible for releasing the GIL around its
native worker gang.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import os
import operator

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Tuple

from .native_bindings import NativeBindingError
from .native_bindings import NativeBindingEvidenceError
from .native_bindings import NativeBindingReadiness
from .native_bindings import NativeBindingResultError
from .types import ExecutionMode
from .types import NativeCallContext
from .sssp_policy import load_sssp_policy_routes, invoke_sssp_policy


PARTITIONED_EVIDENCE_SCHEMA = "egmgpu.partitioned-binding-evidence-index/v3"
BFS_FRONTIER_CALIBRATION_SCHEMA = (
    "egmgpu.partitioned-bfs-frontier-calibration/v1"
)
DEFAULT_PARTITIONED_EVIDENCE_INDEX = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "validation"
    / "partitioned_binding_evidence_v3.json"
)
_HEX = frozenset("0123456789abcdef")
_GRAPH_TOKEN = object()
_BFS_FRONTIER_POLICIES = ("sparse", "dense", "adaptive")
_BFS_THRESHOLD_FIELDS = (
    "dense_vertex_enter",
    "dense_vertex_exit",
    "dense_edge_enter",
    "dense_edge_exit",
    "dense_boundary_enter",
)


class PartitionedResultKind(str, Enum):
    NODE_SCORES = "node_scores"
    COMPONENT_SETS = "component_sets"
    HOP_DISTANCES = "hop_distances"
    WEIGHTED_DISTANCES = "weighted_distances"


@dataclass(frozen=True)
class BackendRequirement:
    name: str
    cases: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("backend name must be non-empty")
        cases = tuple(str(item) for item in self.cases)
        if any(not item for item in cases) or len(set(cases)) != len(cases):
            raise ValueError("backend evidence cases must be non-empty and unique")
        object.__setattr__(self, "cases", cases)


@dataclass(frozen=True)
class PartitionedBindingSpec:
    name: str
    binding: str
    result_kind: PartitionedResultKind
    backend_requirements: Tuple[BackendRequirement, ...]
    mode: ExecutionMode = ExecutionMode.PARTITIONED

    def __post_init__(self) -> None:
        if not self.name or not self.binding:
            raise ValueError("partitioned binding name and symbol must be non-empty")
        if self.mode is not ExecutionMode.PARTITIONED:
            raise ValueError("partitioned bindings require partitioned mode")
        names = tuple(item.name for item in self.backend_requirements)
        if len(set(names)) != len(names):
            raise ValueError("partitioned backend requirements must be unique")


@dataclass(frozen=True)
class EvidenceFile:
    path: str
    sha256: str
    role: str


@dataclass(frozen=True)
class BfsAdaptiveObservation:
    backend: str
    case: str
    trace_rounds: int
    sparse_rounds: int
    dense_rounds: int
    frontier_switches: int


@dataclass(frozen=True)
class BfsFrontierGate:
    """Exact-group frontier policies and thresholds admitted by evidence."""

    qualified_policies: Tuple[str, ...]
    frontier_thresholds: Tuple[Tuple[str, float], ...]
    threshold_source: str
    adaptive_observations: Tuple[BfsAdaptiveObservation, ...]
    auto_policy: str
    calibration_file: Optional[EvidenceFile]

    @property
    def thresholds(self) -> Dict[str, float]:
        return dict(self.frontier_thresholds)

    @property
    def adaptive_auto_qualified(self) -> bool:
        return self.auto_policy == "adaptive" and self.calibration_file is not None


@dataclass(frozen=True)
class BackendEvidence:
    backend: str
    requested_backend: str
    observed_backend: str
    cases: Tuple[str, ...]
    correctness_passed: bool
    correctness_criterion: str
    extension_sha256: str
    raw_files: Tuple[EvidenceFile, ...]


@dataclass(frozen=True)
class TopologyEvidence:
    fingerprint: str
    devices: Tuple[int, ...]
    identity: Mapping[str, Any]


@dataclass(frozen=True)
class PartitionedBindingEvidence:
    function: str
    binding: str
    mode: ExecutionMode
    devices: Tuple[int, ...]
    device_group_fingerprint: str
    # This is the full measured-deployment fingerprint consumed by the
    # planner, not the selected device-group fingerprint above.
    topology_fingerprint: str
    required_backends: Tuple[str, ...]
    runs: Tuple[BackendEvidence, ...]
    bfs_frontier_gate: Optional[BfsFrontierGate]
    index_path: str
    index_sha256: str
    # Captured only from a checksum- and deployment-validated topology entry.
    # None preserves the historical fully-peer-connected qualification contract.
    topology_identity: Optional[Mapping[str, Any]] = None

    @property
    def extension_sha256s(self) -> Tuple[str, ...]:
        return tuple(sorted({item.extension_sha256 for item in self.runs}))


@dataclass(frozen=True)
class PartitionedEvidenceIndex:
    path: str
    sha256: str
    extension_sha256: str
    deployment_fingerprint: str
    deployment_measurement: EvidenceFile
    topologies: Mapping[str, TopologyEvidence]
    binding_groups: Mapping[str, Tuple[PartitionedBindingEvidence, ...]]
    # Backward-compatible shortest qualified group per function.
    bindings: Mapping[str, PartitionedBindingEvidence]


@dataclass(frozen=True)
class PartitionedBindingStatus:
    spec: PartitionedBindingSpec
    readiness: NativeBindingReadiness
    evidence: Optional[PartitionedBindingEvidence] = None
    detail: str = "retained evidence has not been recorded"
    evidence_groups: Tuple[PartitionedBindingEvidence, ...] = ()


@dataclass(frozen=True)
class PartitionedRawResult:
    function: str
    binding: str
    execution_mode: ExecutionMode
    devices_used: Tuple[int, ...]
    communication_backend: str
    kernel_seconds: Optional[float]
    device_processing_seconds: float
    public_result: Any
    raw: Mapping[str, Any]


class ImmutableMultiPartitionGraph:
    """Read-only Python owner for one validated native partition manifest."""

    __slots__ = ("_native", "_sealed", "__weakref__")
    _eggpu_bulk_csr = True
    # Public EasyGraph decorators inspect ``cflag`` before EGMGPU dispatch.
    # This is not a legacy EasyGraph C++ graph, so that hybrid path stays off.
    cflag = 0

    def __init__(self, native: Any, token: object) -> None:
        if token is not _GRAPH_TOKEN:
            raise TypeError("use load_partitioned_graph() to create this object")
        object.__setattr__(self, "_native", native)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("a loaded multi-partition graph is immutable")
        object.__setattr__(self, name, value)

    @property
    def _cpp_graph(self) -> Any:
        return self._native

    @property
    def _mutation_generation(self) -> int:
        return int(self._native.generation)

    @property
    def _eggpu_bulk_signature(self) -> Tuple[Any, ...]:
        return (
            "multi-partition",
            str(self._native.graph_id),
            int(self._native.generation),
            str(self._native.view_signature),
            str(self._native.topology_signature),
            str(self._native.weight_signature),
        )

    @property
    def num_nodes(self) -> int:
        return int(self._native.vertex_count)

    @property
    def num_edges(self) -> int:
        return int(self._native.edge_count)

    @property
    def num_entries(self) -> int:
        return int(self._native.adjacency_count)

    def number_of_nodes(self) -> int:
        return self.num_nodes

    def number_of_edges(self) -> int:
        return self.num_edges

    def __len__(self) -> int:
        return self.num_nodes

    def __iter__(self):
        return iter(range(self.num_nodes))

    @property
    def nodes(self) -> range:
        """Dense global IDs exposed as a read-only, allocation-free view."""

        return range(self.num_nodes)

    def is_directed(self) -> bool:
        return bool(self._native.directed)

    def is_multigraph(self) -> bool:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _valid_sha256(value: Any, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in _HEX for character in digest):
        raise NativeBindingEvidenceError("{} is not a SHA-256 digest".format(field))
    return digest


def _object(
    value: Any,
    required: Iterable[str],
    optional: Iterable[str],
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise NativeBindingEvidenceError("{} must be a JSON object".format(context))
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set.difference(value))
    unknown = sorted(set(value).difference(allowed))
    if missing:
        raise NativeBindingEvidenceError(
            "{} is missing {}".format(context, ", ".join(missing))
        )
    if unknown:
        raise NativeBindingEvidenceError(
            "{} contains unknown fields {}".format(context, ", ".join(unknown))
        )
    return value


def _canonical_topology(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def topology_fingerprint(identity: Mapping[str, Any]) -> str:
    """Return the canonical fingerprint for one exact device group."""

    normalized = _parse_topology_identity(identity, "topology identity")
    return _canonical_topology(normalized)


def _deployment_identity(
    payload: Any,
) -> Tuple[str, Dict[int, Dict[str, Any]], Dict[Tuple[int, int], bool]]:
    """Recompute the full deployment fingerprint and retained identities.

    The calibrated transport file fingerprints the complete topology snapshot;
    this is the identity consumed by the deployment planner.  Partition
    validators separately fingerprint the exact device group that ran each
    case, so neither identity can be substituted for the other.
    """

    if not isinstance(payload, dict):
        raise NativeBindingEvidenceError("deployment measurement must be a JSON object")
    if int(payload.get("schema_version", -1)) != 1:
        raise NativeBindingEvidenceError("unsupported deployment measurement schema")
    topology = payload.get("topology")
    benchmark = payload.get("benchmark")
    devices = payload.get("devices")
    if (
        not isinstance(topology, dict)
        or not isinstance(benchmark, dict)
        or not isinstance(devices, list)
        or len(devices) < 2
    ):
        raise NativeBindingEvidenceError("deployment measurement is incomplete")
    selected = tuple(int(device) for device in devices)
    if selected != tuple(sorted(set(selected))):
        raise NativeBindingEvidenceError(
            "deployment measurement devices must be sorted and unique"
        )
    fingerprint_fields = (
        "driver_version",
        "cuda_driver_version",
        "gpus",
        "nvidia_smi_topology",
        "p2p_access",
        "transport_bandwidth_gbps",
        "transport_latency_us",
    )
    if any(field not in topology for field in fingerprint_fields):
        raise NativeBindingEvidenceError(
            "deployment measurement lacks fingerprinted topology fields"
        )
    fingerprint_payload = {
        field: topology[field] for field in fingerprint_fields
    }
    fingerprint_payload["nvidia_smi_topology"] = str(
        fingerprint_payload["nvidia_smi_topology"]
    ).strip()
    encoded = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    retained_fingerprint = _valid_sha256(
        payload.get("topology_fingerprint"),
        "deployment measurement topology fingerprint",
    )
    if fingerprint != retained_fingerprint:
        raise NativeBindingEvidenceError(
            "deployment measurement topology fingerprint does not match its facts"
        )

    raw_devices = topology["gpus"]
    if not isinstance(raw_devices, list):
        raise NativeBindingEvidenceError("deployment measurement has no GPU list")
    records: Dict[int, Dict[str, Any]] = {}
    for raw in raw_devices:
        if not isinstance(raw, dict):
            raise NativeBindingEvidenceError("deployment GPU record must be an object")
        device_id = int(raw.get("index", raw.get("device_id", -1)))
        if device_id < 0 or device_id in records:
            raise NativeBindingEvidenceError("deployment GPU IDs are invalid")
        if "total_global_memory" in raw:
            memory = int(raw["total_global_memory"])
        elif "memory_total_mib" in raw:
            memory = int(raw["memory_total_mib"]) * (1 << 20)
        else:
            raise NativeBindingEvidenceError("deployment GPU has no memory capacity")
        records[device_id] = {
            "device_id": device_id,
            "uuid": str(raw.get("uuid", "")),
            "name": str(raw.get("name", "")),
            "pci_bus_id": str(raw.get("pci_bus_id", "")),
            "compute_capability": str(raw.get("compute_capability", "")),
            "total_global_memory": memory,
        }
    if set(selected) != set(records):
        raise NativeBindingEvidenceError(
            "deployment measurement device list differs from its topology"
        )
    peer_support: Dict[Tuple[int, int], bool] = {}
    for raw in benchmark.get("peer_links", ()):
        if not isinstance(raw, dict):
            raise NativeBindingEvidenceError("deployment peer link must be an object")
        peer_support[(int(raw["from_device"]), int(raw["to_device"]))] = bool(
            raw.get("supported", False)
        )
    raw_peer = topology["p2p_access"]
    if not isinstance(raw_peer, dict):
        raise NativeBindingEvidenceError(
            "deployment topology p2p_access must be an object"
        )
    for source in selected:
        for target in selected:
            if source == target:
                continue
            key = "{}->{}".format(source, target)
            if key not in raw_peer:
                raise NativeBindingEvidenceError(
                    "deployment topology lacks peer identity {}".format(key)
                )
            if bool(raw_peer[key]) != peer_support.get((source, target)):
                raise NativeBindingEvidenceError(
                    "deployment topology and transport peer support disagree"
                )
    return fingerprint, records, peer_support


def _canonical_pci(value: str) -> str:
    fields = str(value).lower().split(":")
    if len(fields) != 3:
        return str(value).lower()
    try:
        return "{:x}:{:02x}:{}".format(
            int(fields[0], 16), int(fields[1], 16), fields[2]
        )
    except ValueError:
        return str(value).lower()


def _validate_group_within_deployment(
    group: TopologyEvidence,
    deployment_records: Mapping[int, Mapping[str, Any]],
    peer_support: Mapping[Tuple[int, int], bool],
) -> None:
    for raw in group.identity["devices"]:
        device_id = int(raw["device_id"])
        deployment = deployment_records.get(device_id)
        if deployment is None:
            raise NativeBindingEvidenceError(
                "device group is absent from the measured deployment"
            )
        for field in ("uuid", "name", "compute_capability"):
            if str(raw[field]) != str(deployment[field]):
                raise NativeBindingEvidenceError(
                    "device group identity differs from the measured deployment"
                )
        # CUDA reports allocatable global memory after the driver/ECC reserve,
        # while nvidia-smi reports nominal board memory in MiB.  Bind both to
        # the same exact GiB capacity class instead of pretending the two API
        # quantities are byte-identical.
        gib = 1 << 30
        runtime_capacity_class = (
            int(raw["total_global_memory"]) + gib - 1
        ) // gib
        deployment_capacity_class = (int(deployment["total_global_memory"]) + gib - 1) // gib
        if runtime_capacity_class != deployment_capacity_class:
            raise NativeBindingEvidenceError(
                "device group memory class differs from the measured deployment"
            )
        if _canonical_pci(raw["pci_bus_id"]) != _canonical_pci(
            deployment["pci_bus_id"]
        ):
            raise NativeBindingEvidenceError(
                "device group PCI identity differs from the measured deployment"
            )
    expected_peer = [
        [
            source == target
            or bool(peer_support.get((source, target), False))
            for target in group.devices
        ]
        for source in group.devices
    ]
    if group.identity["peer_access"] != expected_peer:
        raise NativeBindingEvidenceError(
            "device group peer identity differs from the measured deployment"
        )


def _parse_topology_identity(value: Any, context: str) -> Dict[str, Any]:
    payload = _object(value, ("devices", "peer_access"), (), context)
    if not isinstance(payload["devices"], list) or len(payload["devices"]) < 2:
        raise NativeBindingEvidenceError(
            "{} requires at least two device records".format(context)
        )
    devices = []
    required = (
        "device_id",
        "uuid",
        "name",
        "pci_bus_id",
        "compute_capability",
        "total_global_memory",
    )
    for index, item in enumerate(payload["devices"]):
        record = _object(item, required, (), "{} device {}".format(context, index))
        normalized = {
            "device_id": int(record["device_id"]),
            "uuid": str(record["uuid"]),
            "name": str(record["name"]),
            "pci_bus_id": str(record["pci_bus_id"]),
            "compute_capability": str(record["compute_capability"]),
            "total_global_memory": int(record["total_global_memory"]),
        }
        if (
            normalized["device_id"] < 0
            or normalized["total_global_memory"] <= 0
            or any(
                not normalized[key]
                for key in required
                if key not in {"device_id", "total_global_memory"}
            )
        ):
            raise NativeBindingEvidenceError(
                "{} has an invalid device record".format(context)
            )
        devices.append(normalized)
    ids = tuple(item["device_id"] for item in devices)
    if tuple(sorted(ids)) != ids or len(set(ids)) != len(ids):
        raise NativeBindingEvidenceError(
            "{} device IDs must be unique and sorted".format(context)
        )
    peer = payload["peer_access"]
    if (
        not isinstance(peer, list)
        or len(peer) != len(devices)
        or any(not isinstance(row, list) or len(row) != len(devices) for row in peer)
        or any(type(cell) is not bool for row in peer for cell in row)
    ):
        raise NativeBindingEvidenceError(
            "{} has an invalid peer-access matrix".format(context)
        )
    return {"devices": devices, "peer_access": peer}


def _resolve_evidence_file(index_path: Path, relative: Any) -> Path:
    text = str(relative)
    if not text or Path(text).is_absolute():
        raise NativeBindingEvidenceError("raw evidence paths must be relative")
    root = index_path.parent.resolve()
    resolved = (root / text).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise NativeBindingEvidenceError(
            "raw evidence must remain under {}".format(root)
        ) from exc
    if not resolved.is_file():
        raise NativeBindingEvidenceError(
            "raw evidence is not a regular file: {}".format(resolved)
        )
    return resolved


def _read_raw_json(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise NativeBindingEvidenceError(
            "cannot read raw evidence {}: {}".format(path, exc)
        ) from exc
    starts = [0]
    starts.extend(
        offset + 1
        for offset, character in enumerate(text)
        if character == "\n"
    )
    for offset in starts:
        if text[offset:].lstrip().startswith("{"):
            candidate = text[offset:].lstrip()
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise NativeBindingEvidenceError(
        "raw correctness evidence has no JSON object: {}".format(path)
    )


def _canonical_components(value: Any) -> Tuple[Tuple[int, ...], ...]:
    if not isinstance(value, list):
        raise NativeBindingEvidenceError("component evidence must be a list")
    result = []
    for component in value:
        if isinstance(component, str):
            try:
                component = ast.literal_eval(component)
            except (ValueError, SyntaxError) as exc:
                raise NativeBindingEvidenceError(
                    "component evidence contains an invalid set"
                ) from exc
        if not isinstance(component, (list, tuple, set, frozenset)):
            raise NativeBindingEvidenceError("component evidence contains a non-set")
        try:
            result.append(tuple(sorted(int(node) for node in component)))
        except (TypeError, ValueError) as exc:
            raise NativeBindingEvidenceError(
                "component evidence contains a non-integer node"
            ) from exc
    return tuple(sorted(result))


def _validate_partition_result_identity(
    value: Any,
    devices: Tuple[int, ...],
    context: str,
) -> None:
    if not isinstance(value, dict):
        raise NativeBindingEvidenceError(
            "{} has no native partition result".format(context)
        )
    try:
        used = tuple(int(device) for device in value["devices_used"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeBindingEvidenceError(
            "{} has invalid devices_used".format(context)
        ) from exc
    if used != devices:
        raise NativeBindingEvidenceError(
            "{} used a different device group".format(context)
        )
    resident = value.get("resident_device_bytes")
    if (
        not isinstance(resident, list)
        or len(resident) != len(devices)
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for item in resident
        )
    ):
        raise NativeBindingEvidenceError(
            "{} lacks one resident-byte count per device".format(context)
        )
    if value.get("converged") is not True:
        raise NativeBindingEvidenceError(
            "{} did not report convergence".format(context)
        )


def _group_result(case: Mapping[str, Any], function: str) -> Mapping[str, Any]:
    """Read current ``group`` raw records and legacy two-device names."""

    partitioned = case.get("partitioned")
    if isinstance(partitioned, dict):
        value = partitioned.get("group", partitioned.get("dual"))
    else:
        value = case.get("group", case.get("two"))
    if not isinstance(value, dict):
        raise NativeBindingEvidenceError(
            "{} case has no device-group result".format(function)
        )
    return value


def _validate_bfs_frontier_result(
    raw: Any,
    *,
    policy: str,
    devices: Tuple[int, ...],
    thresholds: Mapping[str, float],
    context: str,
) -> Tuple[int, int, int, int]:
    _validate_partition_result_identity(raw, devices, context)
    if str(raw.get("frontier_policy")) != policy:
        raise NativeBindingEvidenceError(
            "{} requested/observed frontier policy differs".format(context)
        )
    if raw.get("adaptive_frontier_available") is not True:
        raise NativeBindingEvidenceError(
            "{} did not expose dense/adaptive frontier execution".format(context)
        )
    observed_thresholds = _object(
        raw.get("frontier_thresholds"),
        _BFS_THRESHOLD_FIELDS,
        (),
        "{} thresholds".format(context),
    )
    if any(
        _frontier_fraction(
            observed_thresholds[name], "{} {}".format(context, name)
        ) != thresholds[name]
        for name in _BFS_THRESHOLD_FIELDS
    ):
        raise NativeBindingEvidenceError(
            "{} thresholds differ from the normalized gate".format(context)
        )
    if raw.get("frontier_threshold_source") != "binding_arguments":
        raise NativeBindingEvidenceError(
            "{} threshold source is not binding_arguments".format(context)
        )
    trace = raw.get("frontier_trace")
    if not isinstance(trace, list) or not trace:
        raise NativeBindingEvidenceError("{} frontier trace is empty".format(context))
    kinds = []
    for index, value in enumerate(trace):
        row = _object(
            value,
            (
                "iteration", "frontier_kind", "active_vertices",
                "estimated_active_edges", "active_vertex_fraction",
                "estimated_active_edge_fraction", "boundary_edge_fraction",
            ),
            (),
            "{} trace row {}".format(context, index),
        )
        if int(row["iteration"]) != index + 1:
            raise NativeBindingEvidenceError(
                "{} trace iterations are not contiguous".format(context)
            )
        kind = str(row["frontier_kind"])
        if kind not in {"sparse", "dense"}:
            raise NativeBindingEvidenceError(
                "{} trace contains an unknown frontier path".format(context)
            )
        kinds.append(kind)
        if int(row["active_vertices"]) < 0:
            raise NativeBindingEvidenceError(
                "{} trace has a negative active count".format(context)
            )
        try:
            active_edges = float(row["estimated_active_edges"])
        except (TypeError, ValueError) as exc:
            raise NativeBindingEvidenceError(
                "{} trace active edges are invalid".format(context)
            ) from exc
        if not math.isfinite(active_edges) or active_edges < 0.0:
            raise NativeBindingEvidenceError(
                "{} trace active edges are invalid".format(context)
            )
        for name in (
            "active_vertex_fraction",
            "estimated_active_edge_fraction",
            "boundary_edge_fraction",
        ):
            _frontier_fraction(row[name], "{} trace {}".format(context, name))
    sparse_rounds = kinds.count("sparse")
    dense_rounds = kinds.count("dense")
    switches = sum(left != right for left, right in zip(kinds, kinds[1:]))
    if (
        int(raw.get("iterations", -1)) != len(trace)
        or int(raw.get("sparse_rounds", -1)) != sparse_rounds
        or int(raw.get("dense_rounds", -1)) != dense_rounds
        or int(raw.get("frontier_switches", -1)) != switches
    ):
        raise NativeBindingEvidenceError(
            "{} frontier counters disagree with its trace".format(context)
        )
    if policy == "sparse" and dense_rounds:
        raise NativeBindingEvidenceError(
            "{} forced sparse execution entered dense mode".format(context)
        )
    if policy == "dense" and sparse_rounds:
        raise NativeBindingEvidenceError(
            "{} forced dense execution entered sparse mode".format(context)
        )
    return len(trace), sparse_rounds, dense_rounds, switches


def _bfs_cpu_reference(case: Mapping[str, Any]) -> Tuple[int, ...]:
    """Recompute the retained CPU oracle from the exact validator graph."""

    vertex_count = case.get("vertex_count")
    source = case.get("source")
    edges = case.get("edges")
    if (
        isinstance(vertex_count, bool)
        or not isinstance(vertex_count, int)
        or vertex_count < 1
        or isinstance(source, bool)
        or not isinstance(source, int)
        or source < 0
        or source >= vertex_count
        or not isinstance(edges, list)
    ):
        raise NativeBindingEvidenceError("BFS CPU reference graph is invalid")
    if case.get("edge_count") != len(edges):
        raise NativeBindingEvidenceError("BFS CPU reference edge count changed")
    adjacency = [[] for _ in range(vertex_count)]
    for index, edge in enumerate(edges):
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in edge)
            or any(value < 0 or value >= vertex_count for value in edge)
        ):
            raise NativeBindingEvidenceError(
                "BFS CPU reference edge {} is invalid".format(index)
            )
        adjacency[edge[0]].append(edge[1])
    distances = [-1] * vertex_count
    distances[source] = 0
    frontier = [source]
    cursor = 0
    while cursor < len(frontier):
        left = frontier[cursor]
        cursor += 1
        for right in adjacency[left]:
            if distances[right] == -1:
                distances[right] = distances[left] + 1
                frontier.append(right)
    return tuple(distances)


def _validate_raw_correctness(
    function: str,
    run: BackendEvidence,
    payload: Mapping[str, Any],
    devices: Tuple[int, ...],
    bfs_frontier_gate: Optional[BfsFrontierGate] = None,
) -> None:
    selected_payload = payload
    backend_runs = payload.get("backend_runs")
    if isinstance(backend_runs, list):
        matches = [
            item for item in backend_runs
            if isinstance(item, dict)
            and str(item.get("requested_backend")) == run.requested_backend
            and str(item.get("runtime", {}).get("communication_backend"))
                == run.observed_backend
        ]
        if len(matches) != 1:
            raise NativeBindingEvidenceError(
                "raw evidence does not uniquely identify the normalized backend run"
            )
        selected_payload = matches[0]
    raw_devices = payload.get("devices")
    if raw_devices is None and isinstance(payload.get("runtime"), dict):
        raw_devices = payload["runtime"].get("device_ids")
    if raw_devices is not None:
        try:
            observed_devices = tuple(int(device) for device in raw_devices)
        except (TypeError, ValueError) as exc:
            raise NativeBindingEvidenceError("raw evidence has invalid devices") from exc
        if devices != observed_devices:
            raise NativeBindingEvidenceError(
                "raw evidence device group differs from its normalized record"
            )
    observed = selected_payload.get("communication_backend")
    if observed is None and isinstance(selected_payload.get("runtime"), dict):
        observed = selected_payload["runtime"].get("communication_backend")
    if str(observed) != run.observed_backend:
        raise NativeBindingEvidenceError(
            "raw evidence backend differs from its normalized record"
        )
    requested = selected_payload.get("requested_backend")
    if requested is not None and str(requested) != run.requested_backend:
        raise NativeBindingEvidenceError(
            "raw requested backend differs from its normalized record"
        )
    embedded_extension = payload.get("extension_sha256")
    if embedded_extension is not None and str(embedded_extension) != run.extension_sha256:
        raise NativeBindingEvidenceError(
            "raw evidence extension SHA-256 differs from its normalized record"
        )
    if "exit_code" in payload and int(payload["exit_code"]) != 0:
        raise NativeBindingEvidenceError("raw correctness command did not succeed")

    raw_cases = selected_payload.get("cases", [])
    names = tuple(str(case.get("name")) for case in raw_cases)
    declared_real_cases = tuple(
        case for case in run.cases
        if case not in {
            "cache-generation-hit-and-invalidation",
            "forced-host-staging-reference",
            "nccl-reduction-reference",
        }
    )
    if declared_real_cases and set(names) != set(declared_real_cases):
        raise NativeBindingEvidenceError(
            "raw correctness cases differ from their normalized record"
        )
    if function == "pagerank":
        errors = []
        for case in raw_cases:
            _validate_partition_result_identity(
                case.get("one"), (devices[0],),
                "PageRank {} one-partition result".format(case.get("name")),
            )
            _validate_partition_result_identity(
                _group_result(case, "PageRank"), devices,
                "PageRank {} device-group result".format(case.get("name")),
            )
            values = case.get("errors", {})
            if not isinstance(values, dict):
                raise NativeBindingEvidenceError("PageRank case has no error summary")
            errors.extend(float(value) for value in values.values())
        if payload.get("max_error_vs_cpu") is not None:
            errors.append(float(payload["max_error_vs_cpu"]))
        if not errors or any(not math.isfinite(value) or value > 1e-6 for value in errors):
            raise NativeBindingEvidenceError("PageRank raw correctness threshold failed")
        if "cache-generation-hit-and-invalidation" in run.cases:
            cache = payload.get("cache_generation", {})
            if (
                cache.get("first") != [False] * len(devices)
                or cache.get("second") != [True] * len(devices)
                or cache.get("new_generation") != [False] * len(devices)
            ):
                raise NativeBindingEvidenceError(
                    "PageRank raw cache-generation case is incomplete"
                )
    elif function == "bfs":
        if bfs_frontier_gate is None:
            raise NativeBindingEvidenceError("BFS raw evidence has no frontier gate")
        if payload.get("schema") != "egmgpu.partitioned-bfs-correctness/v2":
            raise NativeBindingEvidenceError(
                "BFS raw evidence has an unsupported correctness schema"
            )
        if tuple(payload.get("frontier_policies", ())) != (
            bfs_frontier_gate.qualified_policies
        ):
            raise NativeBindingEvidenceError(
                "BFS raw evidence qualified-policy list changed"
            )
        thresholds = bfs_frontier_gate.thresholds
        raw_thresholds = _object(
            payload.get("frontier_thresholds"),
            _BFS_THRESHOLD_FIELDS,
            (),
            "BFS raw threshold vector",
        )
        if any(
            _frontier_fraction(raw_thresholds[name], "BFS raw {}".format(name))
            != thresholds[name]
            for name in _BFS_THRESHOLD_FIELDS
        ) or payload.get("frontier_threshold_source") != "qualifier_cli":
            raise NativeBindingEvidenceError(
                "BFS raw threshold values or provenance changed"
            )
        normalized_observations = {
            item.backend: item for item in bfs_frontier_gate.adaptive_observations
        }
        layered_observation = None
        for case in raw_cases:
            cpu_reference = _bfs_cpu_reference(case)
            try:
                retained_reference = tuple(
                    int(distance) for distance in case["expected_dense_distances"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise NativeBindingEvidenceError(
                    "BFS retained CPU reference is invalid"
                ) from exc
            if retained_reference != cpu_reference:
                raise NativeBindingEvidenceError(
                    "BFS retained CPU reference differs from the exact graph"
                )
            expected = {
                index: int(distance)
                for index, distance in enumerate(cpu_reference)
                if int(distance) >= 0
            }
            single_gpu = case.get("single_gpu_eggpu_dense_distances")
            if not isinstance(single_gpu, list) or tuple(
                int(value) for value in single_gpu
            ) != cpu_reference:
                raise NativeBindingEvidenceError(
                    "BFS existing single-GPU comparison failed"
                )
            partitioned = _object(
                case.get("partitioned"),
                ("single", "group"),
                (),
                "BFS {} partitioned results".format(case.get("name")),
            )
            policy_maps = {}
            for location, expected_devices in (
                ("single", (devices[0],)),
                ("group", devices),
            ):
                policy_map = partitioned[location]
                if (
                    not isinstance(policy_map, dict)
                    or set(policy_map) != set(bfs_frontier_gate.qualified_policies)
                    or len(policy_map) != len(bfs_frontier_gate.qualified_policies)
                ):
                    raise NativeBindingEvidenceError(
                        "BFS {} {} policy set changed".format(case.get("name"), location)
                    )
                policy_maps[location] = policy_map
                for policy in bfs_frontier_gate.qualified_policies:
                    raw = policy_map[policy]
                    summary = _validate_bfs_frontier_result(
                        raw,
                        policy=policy,
                        devices=expected_devices,
                        thresholds=thresholds,
                        context="BFS {} {} {}".format(
                            case.get("name"), location, policy
                        ),
                    )
                    actual = {
                        int(node): int(distance)
                        for node, distance in raw.get("values", {}).items()
                    }
                    if actual != expected:
                        raise NativeBindingEvidenceError(
                            "BFS raw correctness comparison failed"
                        )
                    if location == "group" and raw.get(
                        "communication_backend"
                    ) != run.observed_backend:
                        raise NativeBindingEvidenceError(
                            "BFS group result used an unqualified backend"
                        )
                    if (
                        str(case.get("name")) == "dense-layered-switch"
                        and location == "group"
                        and policy == "adaptive"
                    ):
                        layered_observation = summary
            if str(case.get("name")) == "strict-empty-partition" and int(
                case.get("vertex_count", -1)
            ) >= len(devices):
                raise NativeBindingEvidenceError(
                    "BFS strict-empty-partition case contains no empty partition"
                )
            if str(case.get("name")) == "strict-empty-partition":
                ownership = _object(
                    case.get("partition_owned_vertex_counts"),
                    ("single", "group"),
                    (),
                    "BFS strict-empty-partition ownership",
                )
                try:
                    single_owned = tuple(int(value) for value in ownership["single"])
                    group_owned = tuple(int(value) for value in ownership["group"])
                except (TypeError, ValueError) as exc:
                    raise NativeBindingEvidenceError(
                        "BFS strict-empty-partition ownership is invalid"
                    ) from exc
                if (
                    single_owned != (1,)
                    or len(group_owned) != len(devices)
                    or sum(group_owned) != 1
                    or group_owned.count(0) != len(devices) - 1
                ):
                    raise NativeBindingEvidenceError(
                        "BFS strict-empty-partition was not physically realized"
                    )
        normalized = normalized_observations.get(run.observed_backend)
        if normalized is None or layered_observation != (
            normalized.trace_rounds,
            normalized.sparse_rounds,
            normalized.dense_rounds,
            normalized.frontier_switches,
        ):
            raise NativeBindingEvidenceError(
                "BFS adaptive trace differs from its normalized round counts"
            )
    elif function == "wcc":
        for case in raw_cases:
            expected = _canonical_components(case["expected_components"])
            group = _group_result(case, "WCC")
            _validate_partition_result_identity(
                case["partitioned"].get("single"), (devices[0],),
                "WCC {} one-partition result".format(case.get("name")),
            )
            _validate_partition_result_identity(
                group, devices,
                "WCC {} device-group result".format(case.get("name")),
            )
            actual = _canonical_components(group["components"])
            if actual != expected or group.get("communication_backend") != run.observed_backend:
                raise NativeBindingEvidenceError("WCC raw correctness comparison failed")
    elif function == "sssp":
        for case in raw_cases:
            expected = {
                index: float(distance)
                for index, distance in enumerate(case["expected_dense_distances"])
                if distance is not None
            }
            group = _group_result(case, "SSSP")
            _validate_partition_result_identity(
                case["partitioned"].get("single"), (devices[0],),
                "SSSP {} one-partition result".format(case.get("name")),
            )
            _validate_partition_result_identity(
                group, devices,
                "SSSP {} device-group result".format(case.get("name")),
            )
            actual = {int(node): float(value) for node, value in group["values"].items()}
            if (
                set(actual) != set(expected)
                or any(abs(actual[node] - expected[node]) > 1e-8 for node in expected)
                or group.get("communication_backend") != run.observed_backend
            ):
                raise NativeBindingEvidenceError("SSSP raw correctness comparison failed")


def _validate_raw_identity(
    run: BackendEvidence,
    topology: TopologyEvidence,
) -> None:
    correctness_files = [
        Path(item.path) for item in run.raw_files
        if "correctness" in item.role
    ]
    if len(correctness_files) != 1:
        raise NativeBindingEvidenceError(
            "each backend requires exactly one raw correctness record"
        )
    extension_files = [
        Path(item.path) for item in run.raw_files
        if item.role == "extension identity"
    ]
    for path in extension_files:
        try:
            embedded = path.read_text(encoding="utf-8").split()[0].lower()
        except (OSError, UnicodeError, IndexError) as exc:
            raise NativeBindingEvidenceError(
                "cannot read retained extension identity"
            ) from exc
        if embedded != run.extension_sha256:
            raise NativeBindingEvidenceError(
                "extension identity file differs from normalized evidence"
            )
    identity_files = [
        Path(item.path) for item in run.raw_files
        if item.role == "device identity"
    ]
    for path in identity_files:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        raw = {}
        for line in lines:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 3:
                raise NativeBindingEvidenceError("invalid retained device identity")
            raw[int(fields[0])] = (fields[1], fields[2])
        expected = {
            item["device_id"]: (item["uuid"], item["name"])
            for item in topology.identity["devices"]
        }
        if raw != expected:
            raise NativeBindingEvidenceError(
                "retained device identities differ from the topology fingerprint"
            )
    payload = _read_raw_json(correctness_files[0])
    raw_topology = payload.get("topology")
    if isinstance(raw_topology, dict) and isinstance(raw_topology.get("devices"), list):
        expected = {
            int(item["device_id"]): item
            for item in topology.identity["devices"]
        }
        observed = {
            int(item["device_id"]): item
            for item in raw_topology["devices"]
            if int(item["device_id"]) in expected
        }
        for device_id, identity in expected.items():
            item = observed.get(device_id)
            if item is None or any(
                str(item.get(field)) != str(identity[field])
                for field in (
                    "name", "pci_bus_id", "compute_capability",
                    "total_global_memory",
                )
            ):
                raise NativeBindingEvidenceError(
                    "raw topology differs from the normalized topology identity"
                )
        peer = raw_topology.get("peer_access")
        selected = tuple(identity["device_id"] for identity in topology.identity["devices"])
        if isinstance(peer, list):
            try:
                observed_peer = [
                    [bool(peer[source][target]) for target in selected]
                    for source in selected
                ]
            except (IndexError, TypeError) as exc:
                raise NativeBindingEvidenceError("raw peer matrix is incomplete") from exc
            if observed_peer != topology.identity["peer_access"]:
                raise NativeBindingEvidenceError(
                    "raw peer matrix differs from the topology fingerprint"
                )
    nvidia_smi = payload.get("nvidia_smi")
    visibility = payload.get("cuda_visible_devices")
    if visibility and (
        not isinstance(nvidia_smi, dict)
        or not isinstance(nvidia_smi.get("stdout"), str)
        or not isinstance(raw_topology, dict)
        or not isinstance(raw_topology.get("devices"), list)
    ):
        raise NativeBindingEvidenceError(
            "masked CUDA evidence requires both raw native topology and NVML identity"
        )
    if isinstance(nvidia_smi, dict) and isinstance(nvidia_smi.get("stdout"), str):
        observed = {}
        for line in nvidia_smi["stdout"].splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 3:
                observed[int(fields[0])] = (fields[1], fields[2])
        expected = {
            item["device_id"]: (item["uuid"], item["name"])
            for item in topology.identity["devices"]
        }
        if visibility:
            visible_uuids = str(visibility).split(",")
            if (
                len(set(visible_uuids)) != len(visible_uuids)
                or any(not value.startswith("GPU-") for value in visible_uuids)
            ):
                raise NativeBindingEvidenceError(
                    "masked CUDA evidence requires an explicit unique UUID order"
                )
            # nvidia-smi retains physical indices even after CUDA visibility
            # remapping. Bind each logical native rank to its exact UUID and
            # the preserved physical NVML record. Native PCI/name/capacity
            # identity was independently compared above; no identity is lost.
            nvml_by_uuid = {identity[0]: identity for identity in observed.values()}
            mismatch = any(
                device >= len(visible_uuids)
                or visible_uuids[device] != identity[0]
                or nvml_by_uuid.get(identity[0]) != identity
                for device, identity in expected.items()
            )
        else:
            mismatch = any(observed.get(device) != identity for device, identity in expected.items())
        if mismatch:
            raise NativeBindingEvidenceError(
                "raw GPU identity differs from the topology fingerprint"
            )


def default_partitioned_native_specs() -> Tuple[PartitionedBindingSpec, ...]:
    pagerank_cases = (
        "singleton-empty-partition",
        "cycle-dangling",
        "weighted-dangling",
        "random-u",
        "random-w",
    )
    wcc_cases = (
        "fixed-cross-partition",
        "strict-empty-partition",
        "random-0",
        "random-1",
        "random-2",
        "random-3",
        "random-4",
        "random-5",
    )
    bfs_cases = (
        "fixed-cross-partition-unreachable",
        "strict-empty-partition",
        "dense-layered-switch",
        "random-directed-0",
        "random-directed-1",
        "random-directed-2",
        "random-directed-3",
        "random-directed-4",
        "random-directed-5",
    )
    return (
        PartitionedBindingSpec(
            "pagerank",
            "cpp_egmgpu_partitioned_pagerank",
            PartitionedResultKind.NODE_SCORES,
            (
                BackendRequirement(
                    "peer_to_peer",
                    pagerank_cases + ("cache-generation-hit-and-invalidation",),
                ),
                BackendRequirement("host_staging", pagerank_cases),
                BackendRequirement("nccl", pagerank_cases),
            ),
        ),
        PartitionedBindingSpec(
            "wcc",
            "cpp_egmgpu_partitioned_wcc",
            PartitionedResultKind.COMPONENT_SETS,
            (
                BackendRequirement("peer_to_peer", wcc_cases),
                BackendRequirement("host_staging", wcc_cases),
            ),
        ),
        PartitionedBindingSpec(
            "bfs",
            "cpp_egmgpu_partitioned_bfs",
            PartitionedResultKind.HOP_DISTANCES,
            (
                BackendRequirement("peer_to_peer", bfs_cases),
                BackendRequirement("host_staging", bfs_cases),
            ),
        ),
        PartitionedBindingSpec(
            "sssp",
            "cpp_egmgpu_partitioned_sssp",
            PartitionedResultKind.WEIGHTED_DISTANCES,
            (
                BackendRequirement(
                    "peer_to_peer",
                    (
                        "fixed-cross-partition-unreachable",
                        "strict-empty-partition",
                        "random-positive-0",
                        "random-positive-1",
                        "random-positive-2",
                        "random-positive-3",
                        "random-positive-4",
                        "random-positive-5",
                    ),
                ),
                BackendRequirement(
                    "host_staging",
                    (
                        "fixed-cross-partition-unreachable",
                        "strict-empty-partition",
                        "random-positive-0",
                        "random-positive-1",
                        "random-positive-2",
                        "random-positive-3",
                        "random-positive-4",
                        "random-positive-5",
                    ),
                ),
            ),
        ),
    )


def _spec_map() -> Dict[str, PartitionedBindingSpec]:
    return {item.name: item for item in default_partitioned_native_specs()}


def _requirements_for_group(
    spec: PartitionedBindingSpec,
    device_count: int,
    topology_identity: Optional[Mapping[str, Any]] = None,
) -> Tuple[BackendRequirement, ...]:
    """Return audited transport/case coverage for an exact device-group size."""

    if topology_identity is not None:
        peer = topology_identity.get("peer_access")
        if (not isinstance(peer, list) or len(peer) != device_count
                or any(not isinstance(row, list) or len(row) != device_count for row in peer)
                or any(type(cell) is not bool for row in peer for cell in row)):
            raise NativeBindingEvidenceError("qualification topology requires a complete boolean P2P matrix")
        fully_peer = all(peer[i][j] for i in range(device_count)
                         for j in range(device_count) if i != j)
        if not fully_peer:
            if device_count not in {2, 4, 8}:
                raise NativeBindingEvidenceError("partitioned evidence supports exact 2-, 4-, or 8-device groups")
            # Unsupported peer transports cannot be qualified on this group.
            # Keep the same function/case checks on the realizable host route.
            host = next(item for item in spec.backend_requirements if item.name == "host_staging")
            cases = host.cases if device_count == 2 else (
                host.cases[0], host.cases[1], host.cases[3] if spec.name == "pagerank" else host.cases[2])
            requirements = [BackendRequirement("host_staging", cases)]
            if spec.name == "pagerank":
                requirements.append(BackendRequirement("nccl", cases))
            return tuple(requirements)

    if device_count == 2:
        return spec.backend_requirements
    if device_count not in {4, 8}:
        raise NativeBindingEvidenceError(
            "partitioned evidence supports exact 2-, 4-, or 8-device groups"
        )
    first = spec.backend_requirements[0]
    cases = (
        first.cases[0],
        first.cases[1],
        first.cases[3] if spec.name == "pagerank" else first.cases[2],
    )
    requirements = [BackendRequirement("peer_to_peer", cases)]
    if spec.name == "pagerank":
        requirements.append(BackendRequirement("nccl", cases))
    return tuple(requirements)


def _frontier_fraction(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NativeBindingEvidenceError(
            "{} must be numeric".format(context)
        ) from exc
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise NativeBindingEvidenceError(
            "{} must be finite and in [0, 1]".format(context)
        )
    return result


def _validate_bfs_calibration_payload(
    payload: Any,
    *,
    extension_sha256: str,
    deployment_fingerprint: str,
    device_group_fingerprint: str,
    devices: Tuple[int, ...],
    thresholds: Mapping[str, float],
) -> None:
    calibration = _object(
        payload,
        (
            "schema", "scope", "extension_sha256", "topology_fingerprint",
            "device_group_fingerprint", "devices", "frontier_thresholds",
            "selection_rule", "samples", "passed",
        ),
        (),
        "BFS frontier calibration",
    )
    if calibration["schema"] != BFS_FRONTIER_CALIBRATION_SCHEMA or calibration[
        "scope"
    ] != "performance_calibration_for_auto_frontier_selection":
        raise NativeBindingEvidenceError(
            "BFS frontier calibration has an unsupported schema or scope"
        )
    if _valid_sha256(
        calibration["extension_sha256"], "BFS calibration extension SHA-256"
    ) != extension_sha256:
        raise NativeBindingEvidenceError(
            "BFS frontier calibration belongs to another extension"
        )
    if _valid_sha256(
        calibration["topology_fingerprint"], "BFS calibration topology"
    ) != deployment_fingerprint or _valid_sha256(
        calibration["device_group_fingerprint"], "BFS calibration device group"
    ) != device_group_fingerprint:
        raise NativeBindingEvidenceError(
            "BFS frontier calibration belongs to another topology"
        )
    try:
        calibrated_devices = tuple(int(value) for value in calibration["devices"])
    except (TypeError, ValueError) as exc:
        raise NativeBindingEvidenceError(
            "BFS frontier calibration has invalid devices"
        ) from exc
    if calibrated_devices != devices:
        raise NativeBindingEvidenceError(
            "BFS frontier calibration uses another exact device group"
        )
    calibrated = _object(
        calibration["frontier_thresholds"],
        _BFS_THRESHOLD_FIELDS,
        (),
        "BFS calibrated thresholds",
    )
    if any(
        _frontier_fraction(calibrated[name], "BFS calibrated {}".format(name))
        != thresholds[name]
        for name in _BFS_THRESHOLD_FIELDS
    ):
        raise NativeBindingEvidenceError(
            "BFS calibrated thresholds differ from correctness evidence"
        )
    if calibration["passed"] is not True or not str(
        calibration["selection_rule"]
    ).strip():
        raise NativeBindingEvidenceError("BFS frontier calibration did not pass")
    samples = calibration["samples"]
    if not isinstance(samples, list) or len(samples) < 3:
        raise NativeBindingEvidenceError(
            "BFS frontier calibration requires at least three samples"
        )
    outcomes = set()
    for index, value in enumerate(samples):
        sample = _object(
            value,
            (
                "workload_id", "source", "sparse_seconds", "dense_seconds",
                "selected_policy",
            ),
            (),
            "BFS calibration sample {}".format(index),
        )
        try:
            sparse = float(sample["sparse_seconds"])
            dense = float(sample["dense_seconds"])
        except (TypeError, ValueError) as exc:
            raise NativeBindingEvidenceError(
                "BFS calibration timings must be numeric"
            ) from exc
        if (
            not math.isfinite(sparse)
            or not math.isfinite(dense)
            or sparse <= 0.0
            or dense <= 0.0
        ):
            raise NativeBindingEvidenceError(
                "BFS calibration timings must be finite and positive"
            )
        selected = str(sample["selected_policy"])
        fastest = "sparse" if sparse <= dense else "dense"
        if selected not in {"sparse", "dense"} or selected != fastest:
            raise NativeBindingEvidenceError(
                "BFS calibration sample does not select its faster path"
            )
        outcomes.add(selected)
    if outcomes != {"sparse", "dense"}:
        raise NativeBindingEvidenceError(
            "BFS calibration must observe both crossover outcomes"
        )


def _parse_bfs_frontier_gate(
    value: Any,
    *,
    index_path: Path,
    extension_sha256: str,
    deployment_fingerprint: str,
    device_group_fingerprint: str,
    devices: Tuple[int, ...],
    expected_backends: Tuple[str, ...],
) -> BfsFrontierGate:
    gate = _object(
        value,
        (
            "qualified_policies", "frontier_thresholds", "threshold_source",
            "adaptive_observations", "auto_policy", "calibration_file",
        ),
        (),
        "BFS frontier gate",
    )
    policies = tuple(str(policy) for policy in gate["qualified_policies"])
    if policies != _BFS_FRONTIER_POLICIES:
        raise NativeBindingEvidenceError(
            "BFS evidence must qualify sparse, dense, and adaptive policies"
        )
    raw_thresholds = _object(
        gate["frontier_thresholds"],
        _BFS_THRESHOLD_FIELDS,
        (),
        "BFS frontier thresholds",
    )
    thresholds = {
        name: _frontier_fraction(
            raw_thresholds[name], "BFS frontier {}".format(name)
        )
        for name in _BFS_THRESHOLD_FIELDS
    }
    if (
        thresholds["dense_vertex_exit"] > thresholds["dense_vertex_enter"]
        or thresholds["dense_edge_exit"] > thresholds["dense_edge_enter"]
    ):
        raise NativeBindingEvidenceError(
            "BFS frontier exit thresholds cannot exceed enter thresholds"
        )
    threshold_source = str(gate["threshold_source"])
    if threshold_source not in {
        "qualification_defaults", "performance_calibration"
    }:
        raise NativeBindingEvidenceError(
            "BFS AUTO threshold provenance is ambiguous"
        )
    raw_observations = gate["adaptive_observations"]
    if not isinstance(raw_observations, list):
        raise NativeBindingEvidenceError(
            "BFS adaptive observations must be a list"
        )
    observations = []
    seen_backends = set()
    for index, value in enumerate(raw_observations):
        item = _object(
            value,
            (
                "backend", "case", "trace_rounds", "sparse_rounds",
                "dense_rounds", "frontier_switches",
            ),
            (),
            "BFS adaptive observation {}".format(index),
        )
        backend = str(item["backend"])
        if backend not in expected_backends or backend in seen_backends:
            raise NativeBindingEvidenceError(
                "BFS adaptive observation has an unexpected backend"
            )
        seen_backends.add(backend)
        try:
            trace_rounds = int(item["trace_rounds"])
            sparse_rounds = int(item["sparse_rounds"])
            dense_rounds = int(item["dense_rounds"])
            switches = int(item["frontier_switches"])
        except (TypeError, ValueError) as exc:
            raise NativeBindingEvidenceError(
                "BFS adaptive observation counters must be integers"
            ) from exc
        if (
            str(item["case"]) != "dense-layered-switch"
            or trace_rounds != sparse_rounds + dense_rounds
            or sparse_rounds < 1
            or dense_rounds < 1
            or switches < 1
            or switches >= trace_rounds
        ):
            raise NativeBindingEvidenceError(
                "BFS adaptive observation does not prove a path switch"
            )
        observations.append(
            BfsAdaptiveObservation(
                backend,
                "dense-layered-switch",
                trace_rounds,
                sparse_rounds,
                dense_rounds,
                switches,
            )
        )
    if seen_backends != set(expected_backends):
        raise NativeBindingEvidenceError(
            "BFS adaptive observations do not cover every required backend"
        )

    auto_policy = str(gate["auto_policy"])
    if auto_policy not in {"sparse", "adaptive"}:
        raise NativeBindingEvidenceError(
            "BFS AUTO evidence policy must be sparse or adaptive"
        )
    calibration_file = None
    raw_calibration = gate["calibration_file"]
    if raw_calibration is not None:
        descriptor = _object(
            raw_calibration,
            ("path", "sha256", "role"),
            (),
            "BFS calibration file",
        )
        resolved = _resolve_evidence_file(index_path, descriptor["path"])
        digest = _valid_sha256(
            descriptor["sha256"], "BFS calibration file SHA-256"
        )
        if _sha256_file(resolved) != digest:
            raise NativeBindingEvidenceError(
                "BFS calibration checksum changed: {}".format(resolved)
            )
        role = str(descriptor["role"])
        if role != "performance calibration for AUTO adaptive frontier selection":
            raise NativeBindingEvidenceError("BFS calibration role is ambiguous")
        _validate_bfs_calibration_payload(
            _read_raw_json(resolved),
            extension_sha256=extension_sha256,
            deployment_fingerprint=deployment_fingerprint,
            device_group_fingerprint=device_group_fingerprint,
            devices=devices,
            thresholds=thresholds,
        )
        calibration_file = EvidenceFile(str(resolved), digest, role)
    if (auto_policy == "adaptive") != (calibration_file is not None):
        raise NativeBindingEvidenceError(
            "BFS AUTO adaptive selection requires exact calibrated thresholds"
        )
    expected_threshold_source = (
        "performance_calibration"
        if calibration_file is not None
        else "qualification_defaults"
    )
    if threshold_source != expected_threshold_source:
        raise NativeBindingEvidenceError(
            "BFS AUTO policy and threshold provenance disagree"
        )
    return BfsFrontierGate(
        policies,
        tuple((name, thresholds[name]) for name in _BFS_THRESHOLD_FIELDS),
        threshold_source,
        tuple(observations),
        auto_policy,
        calibration_file,
    )


def load_partitioned_binding_evidence(
    path: Any = DEFAULT_PARTITIONED_EVIDENCE_INDEX,
) -> PartitionedEvidenceIndex:
    """Load and validate a normalized index without probing a GPU."""

    index_path = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeBindingEvidenceError(
            "cannot parse partitioned evidence index {}: {}".format(index_path, exc)
        ) from exc
    root = _object(
        payload,
        (
            "schema",
            "extension_sha256",
            "deployment",
            "topologies",
            "functions",
        ),
        (),
        "index",
    )
    if root["schema"] != PARTITIONED_EVIDENCE_SCHEMA:
        raise NativeBindingEvidenceError(
            "unsupported partitioned evidence schema {!r}".format(root["schema"])
        )
    if not isinstance(root["topologies"], list) or not root["topologies"]:
        raise NativeBindingEvidenceError("evidence index must contain topologies")
    extension_sha256 = _valid_sha256(
        root["extension_sha256"], "index extension SHA-256"
    )

    deployment = _object(
        root["deployment"],
        ("topology_fingerprint", "measurement_file"),
        (),
        "deployment",
    )
    deployment_fingerprint = _valid_sha256(
        deployment["topology_fingerprint"],
        "deployment topology fingerprint",
    )
    deployment_file_value = _object(
        deployment["measurement_file"],
        ("path", "sha256", "role"),
        (),
        "deployment measurement file",
    )
    deployment_path = _resolve_evidence_file(
        index_path, deployment_file_value["path"]
    )
    deployment_sha = _valid_sha256(
        deployment_file_value["sha256"],
        "deployment measurement file SHA-256",
    )
    if _sha256_file(deployment_path) != deployment_sha:
        raise NativeBindingEvidenceError(
            "deployment measurement checksum changed: {}".format(deployment_path)
        )
    deployment_role = str(deployment_file_value["role"])
    if not deployment_role:
        raise NativeBindingEvidenceError("deployment measurement role is empty")
    measured_fingerprint, deployment_records, peer_support = _deployment_identity(
        _read_raw_json(deployment_path)
    )
    if measured_fingerprint != deployment_fingerprint:
        raise NativeBindingEvidenceError(
            "indexed deployment fingerprint differs from its measurement"
        )
    deployment_measurement = EvidenceFile(
        str(deployment_path), deployment_sha, deployment_role
    )

    topologies: Dict[str, TopologyEvidence] = {}
    for index, value in enumerate(root["topologies"]):
        item = _object(
            value,
            ("device_group_fingerprint", "devices", "identity"),
            (),
            "topology {}".format(index),
        )
        identity = _parse_topology_identity(
            item["identity"], "topology {} identity".format(index)
        )
        fingerprint = _valid_sha256(
            item["device_group_fingerprint"],
            "topology {} device-group fingerprint".format(index),
        )
        if fingerprint != _canonical_topology(identity):
            raise NativeBindingEvidenceError(
                "topology {} fingerprint does not match its identity".format(index)
            )
        devices = tuple(int(device) for device in item["devices"])
        identity_devices = tuple(
            record["device_id"] for record in identity["devices"]
        )
        if devices != identity_devices:
            raise NativeBindingEvidenceError(
                "topology {} devices do not match its identity".format(index)
            )
        if fingerprint in topologies:
            raise NativeBindingEvidenceError("duplicate topology fingerprint")
        topology = TopologyEvidence(fingerprint, devices, identity)
        _validate_group_within_deployment(
            topology, deployment_records, peer_support
        )
        topologies[fingerprint] = topology

    specs_by_name = _spec_map()
    if not isinstance(root["functions"], list):
        raise NativeBindingEvidenceError("index functions must be a list")
    grouped_bindings: Dict[str, list] = {}
    index_sha = _sha256_file(index_path)
    for function_index, value in enumerate(root["functions"]):
        context = "function {}".format(function_index)
        item = _object(
            value,
            (
                "function", "binding", "mode", "devices",
                "topology_fingerprint", "device_group_fingerprint",
                "required_backends", "evidence",
            ),
            ("bfs_frontier_gate",), context,
        )
        name = str(item["function"])
        if name not in specs_by_name:
            raise NativeBindingEvidenceError("unknown function {!r}".format(name))
        spec = specs_by_name[name]
        binding = str(item["binding"])
        mode = ExecutionMode.parse(item["mode"])
        devices = tuple(int(device) for device in item["devices"])
        topology_fingerprint = _valid_sha256(
            item["topology_fingerprint"], "{} topology fingerprint".format(context)
        )
        device_group_fingerprint = _valid_sha256(
            item["device_group_fingerprint"],
            "{} device-group fingerprint".format(context),
        )
        required_backends = tuple(str(value) for value in item["required_backends"])
        if binding != spec.binding or mode is not spec.mode:
            raise NativeBindingEvidenceError(
                "{} binding or mode does not match its production spec".format(name)
            )
        if (
            devices != tuple(sorted(devices))
            or len(devices) < 2
            or len(set(devices)) != len(devices)
        ):
            raise NativeBindingEvidenceError(
                "{} devices must contain at least two unique sorted IDs".format(name)
            )
        if topology_fingerprint != deployment_fingerprint:
            raise NativeBindingEvidenceError(
                "{} references a different measured deployment".format(name)
            )
        topology = topologies.get(device_group_fingerprint)
        if topology is None or topology.devices != devices:
            raise NativeBindingEvidenceError(
                "{} references a missing or mismatched topology".format(name)
            )
        group_requirements = _requirements_for_group(spec, len(devices), topology.identity)
        expected_backends = tuple(value.name for value in group_requirements)
        if required_backends != expected_backends:
            raise NativeBindingEvidenceError(
                "{} backend requirements do not match the audited spec".format(name)
            )
        raw_frontier_gate = item.get("bfs_frontier_gate")
        if name == "bfs":
            if raw_frontier_gate is None:
                raise NativeBindingEvidenceError(
                    "BFS evidence is missing its frontier-policy gate"
                )
            bfs_frontier_gate = _parse_bfs_frontier_gate(
                raw_frontier_gate,
                index_path=index_path,
                extension_sha256=extension_sha256,
                deployment_fingerprint=deployment_fingerprint,
                device_group_fingerprint=device_group_fingerprint,
                devices=devices,
                expected_backends=expected_backends,
            )
        else:
            if raw_frontier_gate is not None:
                raise NativeBindingEvidenceError(
                    "{} cannot carry a BFS frontier-policy gate".format(name)
                )
            bfs_frontier_gate = None
        if not isinstance(item["evidence"], list):
            raise NativeBindingEvidenceError("{} evidence must be a list".format(name))
        runs = []
        seen_backends = set()
        for run_index, run_value in enumerate(item["evidence"]):
            run_context = "{} evidence {}".format(name, run_index)
            run = _object(
                run_value,
                (
                    "backend", "requested_backend", "observed_backend", "cases",
                    "correctness", "extension_sha256", "raw_files",
                ),
                (), run_context,
            )
            backend = str(run["backend"])
            if backend not in expected_backends or backend in seen_backends:
                raise NativeBindingEvidenceError(
                    "{} has an unexpected or duplicate backend".format(run_context)
                )
            seen_backends.add(backend)
            cases = tuple(str(case) for case in run["cases"])
            if any(not case for case in cases) or len(set(cases)) != len(cases):
                raise NativeBindingEvidenceError(
                    "{} cases must be non-empty unique names".format(run_context)
                )
            required_cases = next(
                value.cases for value in group_requirements
                if value.name == backend
            )
            missing_cases = sorted(set(required_cases).difference(cases))
            if missing_cases:
                raise NativeBindingEvidenceError(
                    "{} is missing required cases {}".format(
                        run_context, ", ".join(missing_cases)
                    )
                )
            correctness = _object(
                run["correctness"], ("passed", "criterion"), (),
                "{} correctness".format(run_context),
            )
            if type(correctness["passed"]) is not bool:
                raise NativeBindingEvidenceError(
                    "{} correctness.passed must be boolean".format(run_context)
                )
            criterion = str(correctness["criterion"])
            if not criterion:
                raise NativeBindingEvidenceError(
                    "{} correctness criterion is empty".format(run_context)
                )
            if not isinstance(run["raw_files"], list) or not run["raw_files"]:
                raise NativeBindingEvidenceError(
                    "{} must bind at least one raw file".format(run_context)
                )
            files = []
            seen_paths = set()
            for file_index, file_value in enumerate(run["raw_files"]):
                file_context = "{} raw file {}".format(run_context, file_index)
                file_item = _object(
                    file_value, ("path", "sha256", "role"), (), file_context
                )
                resolved = _resolve_evidence_file(index_path, file_item["path"])
                relative = str(resolved.relative_to(index_path.parent.resolve()))
                if relative in seen_paths:
                    raise NativeBindingEvidenceError(
                        "{} repeats a raw file".format(run_context)
                    )
                seen_paths.add(relative)
                expected_sha = _valid_sha256(
                    file_item["sha256"], "{} sha256".format(file_context)
                )
                if _sha256_file(resolved) != expected_sha:
                    raise NativeBindingEvidenceError(
                        "raw evidence checksum changed: {}".format(resolved)
                    )
                role = str(file_item["role"])
                if not role:
                    raise NativeBindingEvidenceError(
                        "{} role is empty".format(file_context)
                    )
                files.append(EvidenceFile(str(resolved), expected_sha, role))
            normalized_run = BackendEvidence(
                backend=backend,
                requested_backend=str(run["requested_backend"]),
                observed_backend=str(run["observed_backend"]),
                cases=cases,
                correctness_passed=correctness["passed"],
                correctness_criterion=criterion,
                extension_sha256=_valid_sha256(
                    run["extension_sha256"],
                    "{} extension SHA-256".format(run_context),
                ),
                raw_files=tuple(files),
            )
            if normalized_run.extension_sha256 != extension_sha256:
                raise NativeBindingEvidenceError(
                    "{} was not produced by the indexed extension".format(run_context)
                )
            correctness_files = [
                Path(file.path) for file in normalized_run.raw_files
                if "correctness" in file.role
            ]
            if len(correctness_files) != 1:
                raise NativeBindingEvidenceError(
                    "{} requires one raw correctness record".format(run_context)
                )
            raw_payload = _read_raw_json(correctness_files[0])
            _validate_raw_correctness(
                name,
                normalized_run,
                raw_payload,
                devices,
                bfs_frontier_gate=bfs_frontier_gate,
            )
            _validate_raw_identity(normalized_run, topology)
            runs.append(normalized_run)
        normalized_evidence = PartitionedBindingEvidence(
            function=name,
            binding=binding,
            mode=mode,
            devices=devices,
            device_group_fingerprint=device_group_fingerprint,
            topology_fingerprint=topology_fingerprint,
            required_backends=required_backends,
            runs=tuple(runs),
            bfs_frontier_gate=bfs_frontier_gate,
            index_path=str(index_path),
            index_sha256=index_sha,
            topology_identity=topology.identity,
        )
        existing_devices = {
            item.devices for item in grouped_bindings.get(name, ())
        }
        if devices in existing_devices:
            raise NativeBindingEvidenceError(
                "duplicate function/device-group record {!r} {}".format(
                    name, devices
                )
            )
        grouped_bindings.setdefault(name, []).append(normalized_evidence)
    missing_functions = sorted(set(specs_by_name).difference(grouped_bindings))
    if missing_functions:
        raise NativeBindingEvidenceError(
            "index is missing binding records {}".format(", ".join(missing_functions))
        )
    binding_groups = {
        name: tuple(sorted(items, key=lambda item: (len(item.devices), item.devices)))
        for name, items in grouped_bindings.items()
    }
    bindings = {name: items[0] for name, items in binding_groups.items()}
    return PartitionedEvidenceIndex(
        path=str(index_path),
        sha256=index_sha,
        extension_sha256=extension_sha256,
        deployment_fingerprint=deployment_fingerprint,
        deployment_measurement=deployment_measurement,
        topologies=topologies,
        binding_groups=binding_groups,
        bindings=bindings,
    )


def _revalidate_evidence(evidence: PartitionedBindingEvidence) -> None:
    index_path = Path(evidence.index_path).resolve(strict=True)
    if _sha256_file(index_path) != evidence.index_sha256:
        raise NativeBindingEvidenceError(
            "partitioned evidence index checksum changed: {}".format(index_path)
        )
    for run in evidence.runs:
        for item in run.raw_files:
            path = Path(item.path).resolve(strict=True)
            if _sha256_file(path) != item.sha256:
                raise NativeBindingEvidenceError(
                    "raw evidence checksum changed: {}".format(path)
                )
    try:
        candidates = load_partitioned_binding_evidence(index_path).binding_groups[
            evidence.function
        ]
        retained = next(
            item for item in candidates if item.devices == evidence.devices
        )
    except (KeyError, StopIteration) as exc:
        raise NativeBindingEvidenceError(
            "partitioned evidence function is absent from its retained index"
        ) from exc
    if retained != evidence:
        raise NativeBindingEvidenceError(
            "injected partitioned evidence differs from its retained index"
        )


def _readiness(
    spec: PartitionedBindingSpec,
    evidence: PartitionedBindingEvidence,
) -> Tuple[NativeBindingReadiness, str]:
    if any(not run.correctness_passed for run in evidence.runs):
        return NativeBindingReadiness.WIREABLE, "at least one correctness run failed"
    observed = {run.backend for run in evidence.runs}
    required = {
        item.name
        for item in _requirements_for_group(spec, len(evidence.devices), evidence.topology_identity)
    }
    missing = sorted(required.difference(observed))
    if missing:
        return (
            NativeBindingReadiness.WIREABLE,
            "missing backend evidence: {}".format(", ".join(missing)),
        )
    return (
        NativeBindingReadiness.AVAILABLE,
        "all required correctness evidence is retained",
    )


def _require_qualified_transport(raw, evidence, extension_sha256):
    """Reject an observed transport outside this exact group's raw evidence."""
    observed_backends = {
        run.observed_backend for run in evidence.runs
        if run.extension_sha256 == extension_sha256 and run.correctness_passed
    }
    if str(raw.get("communication_backend", "")) not in observed_backends:
        raise NativeBindingResultError(
            "partitioned result used a backend not qualified for this extension"
        )


def _validate_evidence_against_spec(
    spec: PartitionedBindingSpec,
    evidence: PartitionedBindingEvidence,
) -> None:
    if (
        evidence.function != spec.name
        or evidence.binding != spec.binding
        or evidence.mode is not spec.mode
    ):
        raise NativeBindingEvidenceError(
            "partitioned evidence does not match {}".format(spec.name)
        )
    if spec.name == "bfs":
        gate = evidence.bfs_frontier_gate
        if gate is None or gate.qualified_policies != _BFS_FRONTIER_POLICIES:
            raise NativeBindingEvidenceError(
                "BFS evidence lacks the complete frontier-policy gate"
            )
        if gate.auto_policy == "adaptive" and not gate.adaptive_auto_qualified:
            raise NativeBindingEvidenceError(
                "BFS AUTO adaptive policy lacks calibrated thresholds"
            )
    elif evidence.bfs_frontier_gate is not None:
        raise NativeBindingEvidenceError(
            "non-BFS evidence cannot carry a frontier-policy gate"
        )
    if evidence.topology_identity is not None:
        if _canonical_topology(evidence.topology_identity) != evidence.device_group_fingerprint:
            raise NativeBindingEvidenceError("partitioned transport scope differs from its topology fingerprint")
    group_requirements = _requirements_for_group(spec, len(evidence.devices), evidence.topology_identity)
    expected_backends = tuple(item.name for item in group_requirements)
    if evidence.required_backends != expected_backends:
        raise NativeBindingEvidenceError(
            "partitioned evidence weakened the required backend coverage"
        )
    if (
        len(evidence.devices) < 2
        or evidence.devices != tuple(sorted(evidence.devices))
        or len(set(evidence.devices)) != len(evidence.devices)
    ):
        raise NativeBindingEvidenceError(
            "partitioned evidence has an invalid device group"
        )
    _valid_sha256(
        evidence.device_group_fingerprint, "device-group fingerprint"
    )
    _valid_sha256(evidence.topology_fingerprint, "topology fingerprint")
    runs = {run.backend: run for run in evidence.runs}
    if len(runs) != len(evidence.runs):
        raise NativeBindingEvidenceError("partitioned evidence repeats a backend")
    for requirement in group_requirements:
        run = runs.get(requirement.name)
        if run is None:
            continue
        missing_cases = sorted(set(requirement.cases).difference(run.cases))
        if missing_cases:
            raise NativeBindingEvidenceError(
                "{} evidence is missing required cases {}".format(
                    requirement.name, ", ".join(missing_cases)
                )
            )
        if not run.requested_backend or not run.observed_backend:
            raise NativeBindingEvidenceError(
                "partitioned evidence must identify requested and observed backends"
            )
        _valid_sha256(run.extension_sha256, "extension SHA-256")
        if not run.raw_files:
            raise NativeBindingEvidenceError(
                "partitioned evidence must bind at least one raw file"
            )


class PartitionedNativeBindingRegistry:
    """Evidence-derived availability for PageRank, WCC, BFS, and SSSP."""

    def __init__(self, binding_specs: Iterable[PartitionedBindingSpec]) -> None:
        self._status: Dict[str, PartitionedBindingStatus] = {}
        self._evidence_groups: Dict[
            str, Tuple[PartitionedBindingEvidence, ...]
        ] = {}
        for spec in binding_specs:
            if spec.name in self._status:
                raise ValueError("duplicate partitioned binding {}".format(spec.name))
            self._status[spec.name] = PartitionedBindingStatus(
                spec, NativeBindingReadiness.WIREABLE
            )
            self._evidence_groups[spec.name] = ()

    def status(self, name: str) -> PartitionedBindingStatus:
        try:
            return self._status[str(name)]
        except KeyError as exc:
            raise KeyError("unknown partitioned binding {!r}".format(name)) from exc

    def record(
        self,
        name: str,
        evidence: PartitionedBindingEvidence,
    ) -> PartitionedBindingStatus:
        current = self.status(name)
        _revalidate_evidence(evidence)
        _validate_evidence_against_spec(current.spec, evidence)
        updated = PartitionedBindingStatus(
            current.spec,
            NativeBindingReadiness.WIREABLE,
            evidence,
            "retained evidence loaded; current extension and topology are not verified",
            (evidence,),
        )
        self._status[name] = updated
        self._evidence_groups[name] = (evidence,)
        return updated

    def record_index(self, evidence_index: PartitionedEvidenceIndex) -> None:
        pending = []
        for name in self._status:
            groups = evidence_index.binding_groups[name]
            for evidence in groups:
                _revalidate_evidence(evidence)
                _validate_evidence_against_spec(self.status(name).spec, evidence)
            evidence = groups[0]
            current = self.status(name)
            pending.append(
                (
                    name,
                    PartitionedBindingStatus(
                        current.spec,
                        NativeBindingReadiness.WIREABLE,
                        evidence,
                        "retained evidence loaded; current extension and topology are not verified",
                        groups,
                    ),
                )
            )
        for name, updated in pending:
            self._status[name] = updated
            self._evidence_groups[name] = evidence_index.binding_groups[name]

    def evidence_groups(
        self, name: str
    ) -> Tuple[PartitionedBindingEvidence, ...]:
        """Return retained exact device groups, shortest group first."""

        self.status(name)
        return self._evidence_groups[name]

    def matrix(self) -> Dict[str, str]:
        return {
            name: item.readiness.value
            for name, item in sorted(self._status.items())
        }

    def install(
        self,
        functions: Mapping[str, Callable[..., Any]],
        bindings: Any,
        *,
        topology_fingerprint: str,
        registrar: Any,
        raw_result_sink: Optional[Callable[[PartitionedRawResult], None]] = None,
        bfs_frontier_policy: str = "auto",
        request_dense_paths: bool = False,
        sssp_policy_routes: Any = None,
        sssp_device_binding: Any = None,
    ) -> Tuple[str, ...]:
        if not functions:
            return ()
        if not topology_fingerprint:
            raise NativeBindingEvidenceError(
                "partitioned registration requires a topology fingerprint"
            )
        extension_sha = _extension_sha256(bindings)
        pending = []
        for name, function in functions.items():
            state = self.status(name)
            if state.evidence is None:
                raise NativeBindingEvidenceError(
                    "{} remains WIREABLE: retained evidence has not been recorded".format(
                        name
                    )
                )
            evidence_groups = self._evidence_groups[name]
            for evidence in evidence_groups:
                _revalidate_evidence(evidence)
                evidence_readiness, detail = _readiness(state.spec, evidence)
                if evidence_readiness is not NativeBindingReadiness.AVAILABLE:
                    raise NativeBindingEvidenceError(
                        "{} remains WIREABLE for {}: {}".format(
                            name, evidence.devices, detail
                        )
                    )
                if evidence.topology_fingerprint != topology_fingerprint:
                    raise NativeBindingEvidenceError(
                        "{} evidence was captured on a different topology".format(name)
                    )
                if extension_sha not in evidence.extension_sha256s:
                    raise NativeBindingEvidenceError(
                        "{} loaded extension SHA-256 has no retained evidence".format(name)
                    )
            executor = PartitionedNativeExecutor(
                state.spec,
                function,
                bindings,
                evidence_groups,
                raw_result_sink=raw_result_sink,
                bfs_frontier_policy=bfs_frontier_policy,
                request_dense_paths=request_dense_paths,
                sssp_policy_routes=sssp_policy_routes if name == "sssp" else None,
                sssp_device_binding=sssp_device_binding if name == "sssp" else None,
            )
            pending.append((name, function, executor))

        register = getattr(registrar, "register_native_executor", None)
        unregister = getattr(registrar, "unregister_native_executor", None)
        if not callable(register):
            if callable(registrar):
                register = registrar
                unregister = getattr(registrar, "unregister", None)
            else:
                raise TypeError("registrar must register native executors")
        if not callable(unregister):
            register_many = getattr(
                registrar, "register_native_executors_atomic", None
            )
            if callable(register_many):
                register_many(
                    tuple(
                        (function, ExecutionMode.PARTITIONED, executor)
                        for _, function, executor in pending
                    )
                )
                for name, _, _ in pending:
                    state = self.status(name)
                    self._status[name] = PartitionedBindingStatus(
                        state.spec,
                        NativeBindingReadiness.AVAILABLE,
                        state.evidence,
                        "current extension and topology match retained correctness evidence",
                        state.evidence_groups,
                    )
                return tuple(name for name, _, _ in pending)
            raise TypeError("atomic partitioned installation requires rollback")
        register_many = getattr(registrar, "register_native_executors_atomic", None)
        if callable(register_many):
            register_many(
                tuple(
                    (function, ExecutionMode.PARTITIONED, executor)
                    for _, function, executor in pending
                )
            )
            for name, _, _ in pending:
                state = self.status(name)
                self._status[name] = PartitionedBindingStatus(
                    state.spec,
                    NativeBindingReadiness.AVAILABLE,
                    state.evidence,
                    "current extension and topology match retained correctness evidence",
                    state.evidence_groups,
                )
            return tuple(name for name, _, _ in pending)
        installed = []
        try:
            for name, function, executor in pending:
                register(function, ExecutionMode.PARTITIONED, executor)
                installed.append((name, function))
        except BaseException:
            for _, function in reversed(installed):
                try:
                    unregister(function, ExecutionMode.PARTITIONED)
                except BaseException:
                    pass
            raise
        for name, _, _ in pending:
            state = self.status(name)
            self._status[name] = PartitionedBindingStatus(
                state.spec,
                NativeBindingReadiness.AVAILABLE,
                state.evidence,
                "current extension and topology match retained correctness evidence",
                state.evidence_groups,
            )
        return tuple(name for name, _, _ in pending)


def default_partitioned_native_registry() -> PartitionedNativeBindingRegistry:
    return PartitionedNativeBindingRegistry(default_partitioned_native_specs())


def _extension_sha256(bindings: Any) -> str:
    injected = getattr(bindings, "__egmgpu_extension_sha256__", None)
    if injected is not None:
        return _valid_sha256(injected, "loaded extension SHA-256")
    path = getattr(bindings, "__file__", None)
    if not path:
        raise NativeBindingEvidenceError(
            "compiled bindings do not expose an extension file identity"
        )
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise NativeBindingEvidenceError("compiled extension path is not a file")
    return _sha256_file(resolved)


def load_partitioned_graph(
    manifest_path: Any,
    *,
    bindings: Any = None,
) -> ImmutableMultiPartitionGraph:
    """Load one immutable native manifest without starting device workers."""

    if bindings is None:
        import cpp_easygraph as bindings
    native_type = getattr(bindings, "_EGMGPUMultiPartitionGraph", None)
    loader = getattr(bindings, "cpp_egmgpu_load_partition_manifest", None)
    if not isinstance(native_type, type) or not callable(loader):
        raise NativeBindingError(
            "compiled extension lacks the multi-partition manifest loader"
        )
    native = loader(str(Path(manifest_path).expanduser().resolve(strict=True)))
    if not isinstance(native, native_type):
        raise NativeBindingError("partition manifest loader returned the wrong type")
    return ImmutableMultiPartitionGraph(native, _GRAPH_TOKEN)


def _finite_nonnegative(raw: Mapping[str, Any], field: str) -> float:
    if field not in raw:
        raise NativeBindingResultError("partitioned result is missing {}".format(field))
    try:
        result = float(raw[field])
    except (TypeError, ValueError) as exc:
        raise NativeBindingResultError("{} must be numeric".format(field)) from exc
    if not math.isfinite(result) or result < 0:
        raise NativeBindingResultError("{} must be finite and non-negative".format(field))
    return result


class PartitionedRawResultAdapter:
    """Validate diagnostics and reconstruct the exact EasyGraph return shape."""

    def adapt(
        self,
        spec: PartitionedBindingSpec,
        raw: Any,
        devices: Sequence[int],
        vertex_count: int,
        source: Optional[int] = None,
    ) -> PartitionedRawResult:
        if not isinstance(raw, Mapping):
            raise NativeBindingResultError(
                "{} returned a non-mapping result".format(spec.binding)
            )
        reserved = tuple(int(device) for device in devices)
        try:
            used = tuple(int(device) for device in raw["devices_used"])
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeBindingResultError("invalid devices_used") from exc
        if used != reserved:
            raise NativeBindingResultError(
                "partitioned result devices differ from the reserved device group"
            )
        backend = str(raw.get("communication_backend", ""))
        if not backend:
            raise NativeBindingResultError("partitioned result has no communication backend")
        if raw.get("converged") is not True:
            raise NativeBindingResultError("partitioned algorithm did not converge")
        if spec.name in {"bfs", "wcc"}:
            unprofiled = raw.get("profile_enabled") is False
        elif spec.name == "sssp":
            unprofiled = raw.get("device_phase_profiling") is False
        else:
            unprofiled = (
                spec.result_kind is PartitionedResultKind.NODE_SCORES
                and raw.get("execution_schedule") == "streamed"
                and raw.get("device_phase_profiling") is False
            )
        if unprofiled and "kernel_seconds" in raw and raw["kernel_seconds"] is None:
            kernel = None
        else:
            kernel = _finite_nonnegative(raw, "kernel_seconds")
        processing = _finite_nonnegative(raw, "device_processing_seconds")

        if spec.result_kind is PartitionedResultKind.NODE_SCORES:
            import numpy as np

            values = raw.get("values_dense")
            if values is None or isinstance(values, (str, bytes, Mapping)):
                raise NativeBindingResultError("PageRank result has no dense score vector")
            try:
                if np.iscomplexobj(values):
                    raise TypeError("complex PageRank score")
                dense = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise NativeBindingResultError("PageRank scores must be numeric") from exc
            if dense.shape != (vertex_count,) or not np.isfinite(dense).all():
                raise NativeBindingResultError(
                    "PageRank score vector does not match the manifest"
                )
            if np.any(dense < 0.0) or (
                dense.size and abs(float(dense.sum()) - 1.0) > 1e-6
            ):
                raise NativeBindingResultError(
                    "PageRank scores are not a normalized probability vector"
                )
            from easygraph.utils.gpu_eggpu_backend import _DenseValueDict

            public_result = _DenseValueDict(range(vertex_count), dense, dtype=float)
        elif spec.result_kind is PartitionedResultKind.COMPONENT_SETS:
            components = raw.get("components")
            if not isinstance(components, (list, tuple)):
                raise NativeBindingResultError("WCC result has no component sequence")
            public_result = []
            covered = set()
            for component in components:
                if not isinstance(component, (set, frozenset, list, tuple)):
                    raise NativeBindingResultError("a WCC component is not a node set")
                try:
                    normalized = {int(node) for node in component}
                except (TypeError, ValueError) as exc:
                    raise NativeBindingResultError("WCC contains a non-integer node") from exc
                if len(normalized) != len(component) or covered.intersection(normalized):
                    raise NativeBindingResultError("WCC components overlap or repeat nodes")
                if any(node < 0 or node >= vertex_count for node in normalized):
                    raise NativeBindingResultError("WCC contains a node outside the manifest")
                covered.update(normalized)
                public_result.append(normalized)
            if covered != set(range(vertex_count)):
                raise NativeBindingResultError("WCC components do not cover every node")
        elif spec.result_kind is PartitionedResultKind.HOP_DISTANCES and "values_dense" in raw:
            import numpy as np
            from easygraph.utils.gpu_eggpu_backend import _DenseBfsDistanceDict

            dense = np.asarray(raw["values_dense"])
            if dense.shape != (vertex_count,) or dense.dtype != np.dtype(np.int64):
                raise NativeBindingResultError("BFS dense distances must be a full int64 vector")
            if np.any(dense < -1):
                raise NativeBindingResultError("BFS dense vector has an invalid unreachable sentinel")
            public_result = _DenseBfsDistanceDict(
                dense, reachable_count=raw.get("reachable_vertex_count")
            )
            if source is None or public_result.get(source) != 0:
                raise NativeBindingResultError("path result does not contain its source at zero")
        elif spec.result_kind is PartitionedResultKind.WEIGHTED_DISTANCES and "values_dense" in raw:
            import numpy as np
            from .dense_results import DenseSsspDistances

            dense = np.asarray(raw["values_dense"])
            if dense.shape != (vertex_count,) or dense.dtype != np.dtype(np.float64):
                raise NativeBindingResultError("SSSP dense distances must be a full FP64 vector")
            if np.isnan(dense).any() or np.any(dense < 0):
                raise NativeBindingResultError("SSSP dense distances must be nonnegative or infinity")
            reachable = raw.get("reachable_vertex_count")
            if not isinstance(reachable, int) or not 0 <= reachable <= vertex_count:
                raise NativeBindingResultError("SSSP dense result has no valid reachable count")
            public_result = DenseSsspDistances(dense, reachable)
            if source is None or public_result.get(source) != 0:
                raise NativeBindingResultError("path result does not contain its source at zero")
        else:
            values = raw.get("values")
            if not isinstance(values, Mapping):
                raise NativeBindingResultError("path result has no node-distance mapping")
            public_result = {}
            for node, distance in values.items():
                try:
                    if isinstance(node, bool):
                        raise TypeError("boolean node")
                    node_id = operator.index(node)
                    if (
                        spec.result_kind is PartitionedResultKind.HOP_DISTANCES
                        and isinstance(distance, bool)
                    ):
                        raise TypeError("boolean distance")
                    numeric = (
                        operator.index(distance)
                        if spec.result_kind is PartitionedResultKind.HOP_DISTANCES
                        else float(distance)
                    )
                except (TypeError, ValueError) as exc:
                    raise NativeBindingResultError(
                        "path result contains an invalid value"
                    ) from exc
                if node_id < 0 or node_id >= vertex_count or numeric < 0:
                    raise NativeBindingResultError("path result lies outside the manifest")
                if (
                    spec.result_kind is PartitionedResultKind.WEIGHTED_DISTANCES
                    and not math.isfinite(numeric)
                ):
                    raise NativeBindingResultError("SSSP distance must be finite")
                public_result[node_id] = numeric
            if source is None or public_result.get(source) != 0:
                raise NativeBindingResultError("path result does not contain its source at zero")

        return PartitionedRawResult(
            function=spec.name,
            binding=spec.binding,
            execution_mode=spec.mode,
            devices_used=used,
            communication_backend=backend,
            kernel_seconds=kernel,
            device_processing_seconds=processing,
            public_result=public_result,
            raw=raw,
        )


class PartitionedNativeExecutor:
    """Strict controller adapter for one synchronous partitioned binding."""

    service_mode = ExecutionMode.PARTITIONED

    def __init__(
        self,
        spec: PartitionedBindingSpec,
        public_function: Callable[..., Any],
        bindings: Any,
        evidence: Any,
        raw_result_sink: Optional[Callable[[PartitionedRawResult], None]] = None,
        bfs_frontier_policy: str = "auto",
        request_dense_paths: bool = False,
        sssp_policy_routes: Any = None,
        sssp_device_binding: Any = None,
    ) -> None:
        if not callable(public_function):
            raise TypeError("public_function must be callable")
        binding = getattr(bindings, spec.binding, None)
        native_type = getattr(bindings, "_EGMGPUMultiPartitionGraph", None)
        if not callable(binding) or not isinstance(native_type, type):
            raise NativeBindingError(
                "compiled extension lacks {} or its native graph type".format(spec.binding)
            )
        if raw_result_sink is not None and not callable(raw_result_sink):
            raise TypeError("raw_result_sink must be callable")
        self.spec = spec
        self.public_function = public_function
        groups = (
            (evidence,)
            if isinstance(evidence, PartitionedBindingEvidence)
            else tuple(evidence)
        )
        if not groups or any(
            not isinstance(item, PartitionedBindingEvidence) for item in groups
        ):
            raise TypeError("partitioned executor requires retained evidence groups")
        if len({item.devices for item in groups}) != len(groups):
            raise NativeBindingEvidenceError("partitioned executor repeats a device group")
        self.evidence = groups[0]
        self._evidence_by_devices = {item.devices: item for item in groups}
        self._extension_sha256 = _extension_sha256(bindings)
        if any(
            self._extension_sha256 not in item.extension_sha256s
            for item in groups
        ):
            raise NativeBindingEvidenceError(
                "loaded extension SHA-256 has no retained evidence for {}".format(
                    spec.name
                )
            )
        self._binding = binding
        self._native_type = native_type
        self._signature = inspect.signature(public_function)
        self._sink = raw_result_sink
        self._adapter = PartitionedRawResultAdapter()
        # Opt in only for a newly qualified binary with dense path bindings.
        # Older binaries keep their existing positional call signatures.
        self._path_output_options = {"dense_output": True} if request_dense_paths else {}
        requested_frontier = str(bfs_frontier_policy).strip().lower()
        if requested_frontier not in {"auto", "sparse", "dense", "adaptive"}:
            raise ValueError(
                "BFS frontier policy must be auto, sparse, dense, or adaptive"
            )
        if self.spec.name != "bfs" and requested_frontier != "auto":
            raise ValueError(
                "a BFS frontier policy cannot be applied to {}".format(spec.name)
            )
        if self.spec.name == "bfs":
            for item in groups:
                gate = item.bfs_frontier_gate
                if gate is None:
                    raise NativeBindingEvidenceError(
                        "BFS executor has no frontier-policy evidence"
                    )
                if requested_frontier != "auto" and requested_frontier not in (
                    gate.qualified_policies
                ):
                    raise NativeBindingEvidenceError(
                        "explicit BFS frontier policy has no retained evidence"
                    )
                if requested_frontier == "adaptive" and not (
                    gate.adaptive_auto_qualified
                ):
                    raise NativeBindingEvidenceError(
                        "explicit adaptive BFS requires calibrated thresholds; "
                        "use explicit sparse/dense or retain calibration evidence"
                    )
        self._requested_bfs_frontier_policy = requested_frontier
        if spec.name != "sssp" and sssp_policy_routes is not None:
            raise ValueError("SSSP policies cannot be applied to another function")
        if sssp_policy_routes is not None and not callable(getattr(bindings, "cpp_egmgpu_sssp_execution_policy", None)):
            raise NativeBindingEvidenceError("Extension lacks explicit SSSP policy support")
        self._sssp_policies = load_sssp_policy_routes(sssp_policy_routes,
            extension_path=bindings.__file__, eligible_groups=self._evidence_by_devices,
            observed_device_binding=sssp_device_binding,
            cuda_topology=bindings.cpp_egmgpu_probe_topology() if sssp_policy_routes is not None else None)


    @property
    def eligible_device_groups(self) -> Tuple[Tuple[int, ...], ...]:
        """Exact retained-evidence groups the scheduler may reserve."""

        return tuple(
            sorted(self._evidence_by_devices, key=lambda group: (len(group), group))
        )

    def bfs_frontier_policy_for(self, devices: Sequence[int]) -> str:
        """Return the exact policy selected for one retained device group."""

        if self.spec.name != "bfs":
            raise NativeBindingError("frontier policy is defined only for BFS")
        group = tuple(int(device) for device in devices)
        evidence = self._evidence_by_devices.get(group)
        if evidence is None or evidence.bfs_frontier_gate is None:
            raise NativeBindingEvidenceError(
                "device group has no BFS frontier-policy evidence"
            )
        if self._requested_bfs_frontier_policy == "auto":
            return evidence.bfs_frontier_gate.auto_policy
        return self._requested_bfs_frontier_policy

    def __call__(
        self,
        context: NativeCallContext,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if context.mode is not self.service_mode:
            raise NativeBindingError("native manifest executor received the wrong service mode")
        if context.graph_snapshot is None:
            raise NativeBindingError("partitioned execution requires a graph snapshot")
        snapshot = context.graph_snapshot
        if not bool(getattr(snapshot, "source_is_immutable", False)):
            raise NativeBindingError("partitioned execution requires an immutable snapshot")
        native = snapshot.require_native_payload()
        if not isinstance(native, self._native_type):
            raise NativeBindingError(
                "partitioned execution accepts only a loaded MultiPartitionGraph"
            )
        devices = tuple(int(device) for device in context.devices)
        group_evidence = self._evidence_by_devices.get(devices)
        if group_evidence is None:
            raise NativeBindingEvidenceError(
                "reserved devices do not match retained partitioned evidence"
            )
        if int(native.partition_count) != len(devices):
            raise NativeBindingError(
                "manifest partition count does not match the reserved device group"
            )
        try:
            bound = self._signature.bind(*args, **kwargs)
        except TypeError as exc:
            raise NativeBindingError(
                "arguments do not match {}: {}".format(self.spec.name, exc)
            ) from exc
        bound.apply_defaults()
        arguments = bound.arguments
        graph_argument = next(iter(arguments.values()), None)
        if (
            not isinstance(graph_argument, ImmutableMultiPartitionGraph)
            or graph_argument._cpp_graph is not native
        ):
            raise NativeBindingError(
                "the public call must use the loaded immutable partition manifest"
            )
        if (
            int(snapshot.generation) != int(native.generation)
            or int(snapshot.vertex_count) != int(native.vertex_count)
            or int(snapshot.edge_count) != int(native.edge_count)
        ):
            raise NativeBindingError("snapshot metadata does not match its native manifest")

        source = None
        if self.spec.name == "pagerank":
            weight = arguments.get("weight")
            expected_weight = str(native.weight_key) if bool(native.weighted) else None
            if weight != expected_weight:
                raise NativeBindingError(
                    "PageRank weight parameter does not match the loaded manifest view"
                )
            max_iterations = arguments.get("max_iter")
            tolerance = arguments.get("tol")
            alpha = float(arguments.get("alpha", 0.85))
            max_iterations = (
                max(1, int(os.environ.get("EASYGRAPH_CPU_PR_MAX_ITER", "200")))
                if max_iterations is None
                else int(max_iterations)
            )
            tolerance = (
                max(0.0, float(os.environ.get("EASYGRAPH_CPU_PR_TOL", "1e-6")))
                if tolerance is None
                else float(tolerance)
            )
            if (
                not math.isfinite(alpha) or not 0.0 < alpha < 1.0
                or max_iterations < 1
                or not math.isfinite(tolerance) or tolerance < 0.0
            ):
                raise NativeBindingError("invalid PageRank convergence parameters")
            raw = self._binding(
                native,
                alpha,
                max_iterations,
                tolerance,
                list(devices),
            )
        elif self.spec.name == "wcc":
            if bool(native.directed):
                raise NativeBindingError("WCC requires an undirected manifest view")
            raw = self._binding(native, 100000, list(devices))
        elif self.spec.name == "bfs":
            if arguments.get("target") is not None:
                raise NativeBindingError("partitioned BFS currently requires full output")
            source = _source_id(arguments.get("source"), int(native.vertex_count))
            gate = group_evidence.bfs_frontier_gate
            if gate is None:  # defensive: constructor and loader reject this.
                raise NativeBindingEvidenceError(
                    "BFS device group has no frontier-policy evidence"
                )
            frontier_policy = self.bfs_frontier_policy_for(devices)
            thresholds = gate.thresholds
            raw = self._binding(
                native,
                source,
                0,
                frontier_policy,
                list(devices),
                **dict(thresholds, **self._path_output_options)
            )
        elif self.spec.name == "sssp":
            if arguments.get("target") is not None:
                raise NativeBindingError("partitioned SSSP currently requires full output")
            if arguments.get("weight", "weight") != str(native.weight_key):
                raise NativeBindingError(
                    "SSSP weight parameter does not match the loaded manifest view"
                )
            if "source" in arguments:
                source_value = arguments.get("source")
            else:
                sources = arguments.get("sources")
                if (
                    isinstance(sources, (str, bytes))
                    or not hasattr(sources, "__len__")
                    or len(sources) != 1
                ):
                    raise NativeBindingError(
                        "partitioned SSSP currently supports exactly one source"
                    )
                source_value = sources[0]
            source = _source_id(source_value, int(native.vertex_count))
            # Policy is fixed at registration for this evidence-authorized
            # route and copied into the native call; no process env mutation.
            raw = invoke_sssp_policy(self._binding, native, source, devices,
                self._sssp_policies.get(devices), self._path_output_options)
        else:  # pragma: no cover - production specs are exhaustive.
            raise AssertionError("unknown partitioned binding")
        _require_qualified_transport(raw, group_evidence, self._extension_sha256)
        if self.spec.name == "bfs":
            _validate_bfs_frontier_result(
                raw,
                policy=frontier_policy,
                devices=devices,
                thresholds=group_evidence.bfs_frontier_gate.thresholds,
                context="production BFS",
            )
        adapted = self._adapter.adapt(
            self.spec, raw, devices, int(native.vertex_count), source=source
        )
        if self.service_mode is ExecutionMode.SINGLE:
            # Native one-part manifest evidence describes the primitive; the
            # public request reserves exactly one owner and has SINGLE mode.
            adapted = replace(adapted, execution_mode=ExecutionMode.SINGLE)
        if self.spec.name == "sssp" and "source" not in arguments:
            # EasyGraph multi_source_dijkstra keeps a source outer dimension,
            # including its supported one-source case.
            adapted = replace(adapted, public_result={source: adapted.public_result})
        if self._sink is not None:
            self._sink(adapted)
        return adapted.public_result



class SingleOwnerSsspExecutor(PartitionedNativeExecutor):
    """A single request on one owner using the qualified one-part SSSP primitive.

    This does not make cooperative mode accept singleton device groups. The
    primitive's retained manifest evidence is reused, while scheduler mode,
    reservation and returned execution metadata are all SINGLE.
    """

    service_mode = ExecutionMode.SINGLE

    def __init__(self, spec, public_function, bindings, evidence, **kwargs):
        if spec.name != "sssp" or kwargs.get("sssp_policy_routes") is None:
            raise NativeBindingEvidenceError(
                "single-owner SSSP requires an explicit qualified SSSP policy")
        groups = ((evidence,) if isinstance(evidence, PartitionedBindingEvidence)
                  else tuple(evidence))
        if not groups or any(len(item.devices) != 1 for item in groups):
            raise NativeBindingEvidenceError(
                "single-owner SSSP evidence must use exactly one device per group")
        super().__init__(spec, public_function, bindings, groups, **kwargs)
        if set(self._sssp_policies) != set(self._evidence_by_devices):
            raise NativeBindingEvidenceError(
                "each single-owner SSSP device group needs an explicit policy")


def _source_id(value: Any, vertex_count: int) -> int:
    if isinstance(value, bool):
        raise NativeBindingError("partitioned source must be an integer global ID")
    try:
        source = operator.index(value)
    except TypeError as exc:
        raise NativeBindingError(
            "partitioned source must be an integer global ID"
        ) from exc
    if source < 0 or source >= vertex_count:
        raise NativeBindingError("partitioned source lies outside the manifest")
    return source


class _PublicAPIRegistrar:
    def register_native_executor(self, function: Any, mode: Any, executor: Any) -> None:
        from .api import register_native_executor

        register_native_executor(function, mode, executor)

    def unregister_native_executor(self, function: Any, mode: Any) -> None:
        from .api import unregister_native_executor

        unregister_native_executor(function, mode)

    def register_native_executors_atomic(self, registrations: Any) -> None:
        from .api import _register_native_executors_atomic

        _register_native_executors_atomic(registrations)


def register_probed_partitioned_bindings(
    functions: Mapping[str, Callable[..., Any]],
    *,
    evidence_index: Any = DEFAULT_PARTITIONED_EVIDENCE_INDEX,
    topology_fingerprint: str,
    bindings: Any = None,
    registrar: Any = None,
    raw_result_sink: Optional[Callable[[PartitionedRawResult], None]] = None,
    bfs_frontier_policy: str = "auto",
    request_dense_paths: bool = False,
    sssp_policy_routes: Any = None,
    sssp_device_binding: Any = None,
) -> Tuple[str, ...]:
    """Explicitly register evidence-qualified partitioned paths.

    Legacy registration is lazy. Explicit SSSP policy registration checks the
    process CUDA logical topology against the supplied observed device binding;
    it does not start native workers.
    ``AUTO`` selection remains unchanged and therefore cannot select these
    cooperative paths without a separately calibrated planner route.
    """

    if bindings is None:
        import cpp_easygraph as bindings
    if registrar is None:
        registrar = _PublicAPIRegistrar()
    evidence = load_partitioned_binding_evidence(evidence_index)
    registry = default_partitioned_native_registry()
    registry.record_index(evidence)
    return registry.install(
        functions,
        bindings,
        topology_fingerprint=topology_fingerprint,
        registrar=registrar,
        raw_result_sink=raw_result_sink,
        bfs_frontier_policy=bfs_frontier_policy,
        request_dense_paths=request_dense_paths,
        sssp_policy_routes=sssp_policy_routes,
        sssp_device_binding=sssp_device_binding,
    )


def register_partitioned_bindings(
    functions: Mapping[str, Callable[..., Any]],
    *,
    evidence_index: Any = DEFAULT_PARTITIONED_EVIDENCE_INDEX,
    topology_fingerprint: str,
    bindings: Any = None,
    registrar: Any = None,
    raw_result_sink: Optional[Callable[[PartitionedRawResult], None]] = None,
    bfs_frontier_policy: str = "auto",
    request_dense_paths: bool = False,
    sssp_policy_routes: Any = None,
    sssp_device_binding: Any = None,
) -> Tuple[str, ...]:
    """Public explicit-registration alias; never called automatically."""

    return register_probed_partitioned_bindings(
        functions,
        evidence_index=evidence_index,
        topology_fingerprint=topology_fingerprint,
        bindings=bindings,
        registrar=registrar,
        raw_result_sink=raw_result_sink,
        bfs_frontier_policy=bfs_frontier_policy,
        request_dense_paths=request_dense_paths,
        sssp_policy_routes=sssp_policy_routes,
        sssp_device_binding=sssp_device_binding,
    )


# Short module-scoped verbs mirror the replicated gate's lifecycle while
# keeping all mutations explicit.
def specs() -> Tuple[PartitionedBindingSpec, ...]:
    return default_partitioned_native_specs()


def status(
    registry: Optional[PartitionedNativeBindingRegistry] = None,
) -> Dict[str, str]:
    return (default_partitioned_native_registry() if registry is None else registry).matrix()


def load(path: Any = DEFAULT_PARTITIONED_EVIDENCE_INDEX) -> PartitionedEvidenceIndex:
    return load_partitioned_binding_evidence(path)


def record(
    registry: PartitionedNativeBindingRegistry,
    evidence_index: PartitionedEvidenceIndex,
) -> PartitionedNativeBindingRegistry:
    registry.record_index(evidence_index)
    return registry


def install(*args: Any, **kwargs: Any) -> Tuple[str, ...]:
    return register_probed_partitioned_bindings(*args, **kwargs)


__all__ = [
    "BackendEvidence",
    "BackendRequirement",
    "BFS_FRONTIER_CALIBRATION_SCHEMA",
    "BfsAdaptiveObservation",
    "BfsFrontierGate",
    "DEFAULT_PARTITIONED_EVIDENCE_INDEX",
    "EvidenceFile",
    "ImmutableMultiPartitionGraph",
    "PARTITIONED_EVIDENCE_SCHEMA",
    "PartitionedBindingEvidence",
    "PartitionedBindingSpec",
    "PartitionedBindingStatus",
    "PartitionedEvidenceIndex",
    "PartitionedNativeBindingRegistry",
    "PartitionedNativeExecutor",
    "PartitionedRawResult",
    "PartitionedRawResultAdapter",
    "PartitionedResultKind",
    "TopologyEvidence",
    "default_partitioned_native_registry",
    "default_partitioned_native_specs",
    "install",
    "load",
    "load_partitioned_binding_evidence",
    "load_partitioned_graph",
    "record",
    "register_partitioned_bindings",
    "register_probed_partitioned_bindings",
    "specs",
    "status",
    "topology_fingerprint",
]
