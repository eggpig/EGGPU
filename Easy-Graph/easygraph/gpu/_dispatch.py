"""CPU-only, workload-level selection of the integrated GPU execution paths.

Selection depends on the requested devices and state semantics. It never reads
graph identifiers, graph sizes, timings, or benchmark results, and never retries
on a different backend after an execution failure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import operator


CORE_ALGORITHMS = ("bfs", "sssp", "wcc", "pr")
_SUPPORTED = frozenset(CORE_ALGORITHMS + ("weighted_pr",))


@dataclass(frozen=True)
class SessionPlan:
    policy: str
    backend: str
    devices: tuple[int, ...]
    state_model: str
    algorithms: tuple[str, ...]
    input_kind: str
    budget_scope: str

    def to_dict(self):
        """Return the fixed decision for experiment and application provenance."""
        return asdict(self)


def plan_session(*, devices=(0,), state_model="static", algorithms=CORE_ALGORITHMS, input_format=None):
    """Select a route before loading data or importing a GPU runtime.

    ``static`` uses the optimized arena on one device and native partitions on
    multiple devices. ``journal`` requires the single-device mutable arena.
    ``snapshots`` requires the native immutable-version/D-Maintain/C-Retain path,
    including when only one device is requested. Algorithm availability is an
    API contract, not a statement of per-input numerical qualification.
    """
    normalized = []
    for device in devices:
        if isinstance(device, bool) or type(device).__name__ in ("bool", "bool_"):
            raise ValueError("Devices must be nonnegative integer logical IDs")
        try:
            value = operator.index(device)
        except TypeError as exc:
            raise ValueError("Devices must be nonnegative integer logical IDs") from exc
        if value < 0:
            raise ValueError("Devices must be nonnegative integer logical IDs")
        normalized.append(value)
    devices = tuple(normalized)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Require nonempty unique devices")
    if state_model not in ("static", "journal", "snapshots"):
        raise ValueError("state_model must be static, journal, or snapshots")
    if isinstance(algorithms, str):
        raise ValueError("algorithms must be a sequence of operator names")
    algorithms = tuple(algorithms)
    if not algorithms or any(name not in _SUPPORTED for name in algorithms):
        raise NotImplementedError("Requested operator is outside the integrated core")
    if len(set(algorithms)) != len(algorithms):
        raise ValueError("Operator requirements must be unique")
    if state_model == "journal" and len(devices) != 1:
        raise NotImplementedError("Journal mutation requires one GPU")
    arena = len(devices) == 1 and state_model != "snapshots"
    if input_format is None:
        input_format = 'coo' if arena else 'partition_manifest'
    if (arena and input_format not in ('coo', 'csr')) or (not arena and input_format != 'partition_manifest'):
        raise ValueError('Arena accepts coo/csr; native requires partition_manifest')
    return SessionPlan(
        policy="capability_v2", backend="arena" if arena else "native",
        devices=devices, state_model=state_model, algorithms=algorithms,
        input_kind=("owned_" + input_format) if arena else "partition_manifest",
        budget_scope=("single-device backing arena including workspace" if arena
                      else "per-device retained state at completed-query boundaries"),
    )
