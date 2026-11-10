"""Atomic, evidence-derived production binding bootstrap.

This is the only public path that gives native routes production/AUTO
eligibility.  It loads all retained evidence from disk, verifies one compiled
extension identity and one measured deployment topology across the complete
16/6/4 capability matrix, stages every executor without mutating the public
registry, and finally publishes all routes in one atomic registration call.
"""

from __future__ import annotations

import hashlib
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Tuple

from .compatibility import _attest_production_executor
from .compatibility import production_compatibility_predicate
from .native_bindings import NativeBindingEvidenceError
from .native_bindings import default_replicated_native_registry
from .native_bindings import default_replicated_native_specs
from .native_bindings import load_native_probe_evidence
from .partitioned_bindings import default_partitioned_native_registry
from .partitioned_bindings import default_partitioned_native_specs
from .partitioned_bindings import load_partitioned_binding_evidence
from .single_bindings import default_single_native_registry
from .single_bindings import default_single_native_specs
from .single_bindings import default_single_public_functions
from .single_bindings import load_single_native_evidence
from .types import ExecutionMode


_SINGLE_NAMES = tuple(spec.name for spec in default_single_native_specs())
_REPLICATED_NAMES = tuple(spec.name for spec in default_replicated_native_specs())
_PARTITIONED_NAMES = tuple(spec.name for spec in default_partitioned_native_specs())


@dataclass(frozen=True)
class ProductionBootstrapReport:
    """Immutable receipt for one successfully published production bundle."""

    extension_sha256: str
    topology_fingerprint: str
    single_evidence_sha256: str
    partitioned_index_sha256: str
    replicated_evidence_sha256: Mapping[str, str]
    registered_routes: Tuple[Tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "replicated_evidence_sha256",
            dict(sorted(self.replicated_evidence_sha256.items())),
        )


class _StagedRegistrar:
    """Collect registrations while exposing no executable route."""

    def __init__(self) -> None:
        self.routes: Dict[Tuple[Callable[..., Any], ExecutionMode], Callable[..., Any]] = {}

    def register_native_executor(self, function, mode, executor) -> None:
        self._stage(((function, mode, executor),))

    def register_native_executors_atomic(self, registrations) -> None:
        self._stage(tuple(registrations))

    def unregister_native_executor(self, function, mode) -> None:
        self.routes.pop((function, ExecutionMode.parse(mode)), None)

    def _stage(self, registrations) -> None:
        pending = []
        seen = set()
        for function, mode, executor in registrations:
            parsed = ExecutionMode.parse(mode)
            if not callable(function) or not callable(executor):
                raise TypeError("staged function and executor must be callable")
            if parsed is ExecutionMode.AUTO:
                raise ValueError("AUTO cannot be staged as a native executor")
            key = (function, parsed)
            if key in seen or key in self.routes:
                raise NativeBindingEvidenceError(
                    "the production bundle contains a duplicate route"
                )
            seen.add(key)
            pending.append((key, executor))
        self.routes.update(pending)


class _PublicAtomicRegistrar:
    def register_native_executors_atomic(self, registrations) -> None:
        from .api import _register_native_executors_atomic

        _register_native_executors_atomic(registrations)


def bootstrap_production_bindings(
    *,
    single_evidence: Any,
    replicated_evidence: Mapping[str, Any],
    partitioned_evidence_index: Any,
    topology_fingerprint: str,
    raw_result_sinks: Optional[Mapping[str, Callable[..., Any]]] = None,
    bfs_frontier_policy: str = "auto",
    sssp_policy_routes: Any = None,
    sssp_device_binding: Any = None,
) -> ProductionBootstrapReport:
    """Validate and atomically publish the complete production route bundle.

    Evidence arguments must be filesystem paths.  Preconstructed evidence or
    status objects are intentionally rejected: callers cannot manufacture an
    ``AVAILABLE`` value and bypass checksum, extension, and topology checks.
    The expected matrix is exactly 16 single/request functions, six replicated
    functions, and four partitioned functions.
    """

    import cpp_easygraph as bindings

    # The public entry point deliberately has no callable, binding-module, or
    # registrar injection hook.  Production attestation is attached only to
    # the exact 16 EasyGraph wrappers exported by this process and is published
    # only through the process-global atomic registrar.  Research tests and
    # plugins use the private implementation below without gaining a public
    # AVAILABLE/AUTO path.
    return _bootstrap_production_bindings(
        single_evidence=single_evidence,
        replicated_evidence=replicated_evidence,
        partitioned_evidence_index=partitioned_evidence_index,
        topology_fingerprint=topology_fingerprint,
        functions=default_single_public_functions(),
        bindings=bindings,
        registrar=_PublicAtomicRegistrar(),
        raw_result_sinks=raw_result_sinks,
        bfs_frontier_policy=bfs_frontier_policy,
        sssp_policy_routes=sssp_policy_routes,
        sssp_device_binding=sssp_device_binding,
    )


def _bootstrap_production_bindings(
    *,
    single_evidence: Any,
    replicated_evidence: Mapping[str, Any],
    partitioned_evidence_index: Any,
    topology_fingerprint: str,
    functions: Mapping[str, Callable[..., Any]],
    bindings: Any,
    registrar: Any,
    raw_result_sinks: Optional[Mapping[str, Callable[..., Any]]] = None,
    bfs_frontier_policy: str = "auto",
    sssp_policy_routes: Any = None,
    sssp_device_binding: Any = None,
) -> ProductionBootstrapReport:
    """Private dependency-injected implementation for CPU contract tests."""

    topology = str(topology_fingerprint).strip()
    if len(topology) != 64 or any(
        character not in "0123456789abcdef" for character in topology.lower()
    ):
        raise NativeBindingEvidenceError(
            "production bootstrap requires a SHA-256 deployment topology fingerprint"
        )
    topology = topology.lower()
    public_functions = dict(functions)
    _require_exact_names("public function", public_functions, _SINGLE_NAMES)
    if any(not callable(function) for function in public_functions.values()):
        raise TypeError("every production public function must be callable")
    if not isinstance(replicated_evidence, Mapping):
        raise TypeError("replicated_evidence must map six function names to paths")
    _require_exact_names("replicated evidence", replicated_evidence, _REPLICATED_NAMES)
    sinks = {} if raw_result_sinks is None else dict(raw_result_sinks)
    frontier_policy = str(bfs_frontier_policy).strip().lower()
    if frontier_policy not in {"auto", "sparse", "dense", "adaptive"}:
        raise ValueError(
            "bfs_frontier_policy must be auto, sparse, dense, or adaptive"
        )
    unknown_sinks = set(sinks).difference({"single", "replicated", "partitioned"})
    if unknown_sinks:
        raise ValueError(
            "unknown raw-result sinks: {}".format(", ".join(sorted(unknown_sinks)))
        )
    if any(not callable(value) for value in sinks.values()):
        raise TypeError("raw-result sinks must be callable")

    single_path = _evidence_path(single_evidence, "single/request evidence")
    partitioned_path = _evidence_path(
        partitioned_evidence_index, "partitioned evidence index"
    )
    replicated_paths = {
        name: _evidence_path(value, "{} replicated evidence".format(name))
        for name, value in replicated_evidence.items()
    }

    extension_sha = _compiled_extension_sha256(bindings)

    # Load every retained source before constructing or staging one executor.
    loaded_single = load_single_native_evidence(single_path)
    loaded_replicated = {
        name: load_native_probe_evidence(path)
        for name, path in replicated_paths.items()
    }
    loaded_partitioned = load_partitioned_binding_evidence(partitioned_path)

    if tuple(sorted(loaded_single.passed_functions)) != tuple(sorted(_SINGLE_NAMES)):
        raise NativeBindingEvidenceError(
            "single/request evidence must pass all 16 production functions"
        )
    if loaded_single.extension_sha256 != extension_sha:
        raise NativeBindingEvidenceError(
            "single/request evidence does not match the loaded extension"
        )
    if loaded_single.topology_fingerprint != topology:
        raise NativeBindingEvidenceError(
            "single/request evidence was captured on a different topology"
        )
    for name, evidence in loaded_replicated.items():
        if evidence.function != name:
            raise NativeBindingEvidenceError(
                "{} retained evidence identifies function {}".format(
                    name, evidence.function
                )
            )
        if not evidence.passed:
            raise NativeBindingEvidenceError(
                "{} replicated evidence did not pass correctness".format(name)
            )
        if evidence.extension_sha256 != extension_sha:
            raise NativeBindingEvidenceError(
                "{} replicated evidence does not match the loaded extension".format(name)
            )
        if evidence.topology_fingerprint != topology:
            raise NativeBindingEvidenceError(
                "{} replicated evidence was captured on a different topology".format(name)
            )
    if loaded_partitioned.extension_sha256 != extension_sha:
        raise NativeBindingEvidenceError(
            "partitioned evidence does not match the loaded extension"
        )
    if loaded_partitioned.deployment_fingerprint != topology:
        raise NativeBindingEvidenceError(
            "partitioned evidence was captured on a different topology"
        )
    _require_exact_names(
        "partitioned evidence",
        loaded_partitioned.binding_groups,
        _PARTITIONED_NAMES,
    )
    for name, groups in loaded_partitioned.binding_groups.items():
        if not groups:
            raise NativeBindingEvidenceError(
                "{} has no partitioned evidence group".format(name)
            )
        if any(extension_sha not in group.extension_sha256s for group in groups):
            raise NativeBindingEvidenceError(
                "{} partitioned evidence mixes extension identities".format(name)
            )
        if any(group.topology_fingerprint != topology for group in groups):
            raise NativeBindingEvidenceError(
                "{} partitioned evidence mixes deployment topologies".format(name)
            )

    staged = _StagedRegistrar()
    single_registry = default_single_native_registry()
    single_registry.record_evidence(loaded_single)
    single_registry.install_available(
        staged,
        public_functions,
        bindings,
        topology,
        raw_result_sink=sinks.get("single"),
    )

    replicated_registry = default_replicated_native_registry()
    for name, evidence in loaded_replicated.items():
        replicated_registry.record_probe(name, evidence)
    replicated_registry.install_available(
        staged,
        {name: public_functions[name] for name in _REPLICATED_NAMES},
        bindings,
        topology,
        raw_result_sink=sinks.get("replicated"),
    )

    partitioned_registry = default_partitioned_native_registry()
    partitioned_registry.record_index(loaded_partitioned)
    partitioned_registry.install(
        {name: public_functions[name] for name in _PARTITIONED_NAMES},
        bindings,
        topology_fingerprint=topology,
        registrar=staged,
        raw_result_sink=sinks.get("partitioned"),
        bfs_frontier_policy=frontier_policy,
        sssp_policy_routes=sssp_policy_routes,
        sssp_device_binding=sssp_device_binding,
    )

    expected = _expected_route_keys(public_functions)
    if set(staged.routes) != expected:
        missing = expected.difference(staged.routes)
        extra = set(staged.routes).difference(expected)
        raise NativeBindingEvidenceError(
            "staged production matrix is incomplete (missing={}, extra={})".format(
                len(missing), len(extra)
            )
        )

    bundle_identity = _bundle_identity(
        extension_sha,
        topology,
        loaded_single.sha256,
        loaded_partitioned.sha256,
        {name: evidence.sha256 for name, evidence in loaded_replicated.items()},
    )
    registrations = []
    route_names = []
    function_names = {function: name for name, function in public_functions.items()}
    for (function, mode), executor in sorted(
        staged.routes.items(),
        key=lambda item: (function_names[item[0][0]], item[0][1].value),
    ):
        name = function_names[function]
        predicate = production_compatibility_predicate(name, mode, function)
        attested = _attest_production_executor(
            executor,
            predicate,
            "{}:{}:{}".format(bundle_identity, name, mode.value),
        )
        registrations.append((function, mode, attested))
        route_names.append((name, mode.value))

    register_many = getattr(registrar, "register_native_executors_atomic", None)
    if not callable(register_many):
        raise TypeError(
            "production bootstrap requires register_native_executors_atomic(); "
            "per-route registration cannot guarantee zero partial publication"
        )
    # This is the sole externally visible mutation.  All loading, checksum,
    # topology, ABI, callable, predicate, and matrix validation has completed.
    register_many(tuple(registrations))

    return ProductionBootstrapReport(
        extension_sha256=extension_sha,
        topology_fingerprint=topology,
        single_evidence_sha256=loaded_single.sha256,
        partitioned_index_sha256=loaded_partitioned.sha256,
        replicated_evidence_sha256={
            name: evidence.sha256 for name, evidence in loaded_replicated.items()
        },
        registered_routes=tuple(route_names),
    )


def _require_exact_names(label: str, mapping: Mapping[str, Any], expected) -> None:
    found = set(str(name) for name in mapping)
    required = set(expected)
    if found != required:
        raise NativeBindingEvidenceError(
            "{} names differ from the production matrix; missing={}, extra={}".format(
                label,
                sorted(required.difference(found)),
                sorted(found.difference(required)),
            )
        )


def _evidence_path(value: Any, label: str) -> Path:
    if not isinstance(value, (str, bytes, os.PathLike)):
        raise TypeError(
            "{} must be a retained filesystem path, not an injected evidence object".format(
                label
            )
        )
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise NativeBindingEvidenceError("{} is not a regular file".format(label))
    return path


def _compiled_extension_sha256(bindings: Any) -> str:
    path_value = getattr(bindings, "__file__", None)
    if not path_value:
        raise NativeBindingEvidenceError(
            "production bindings must expose their compiled extension file"
        )
    path = Path(path_value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise NativeBindingEvidenceError("compiled extension path is not a file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_route_keys(functions):
    expected = set()
    for name in _SINGLE_NAMES:
        expected.add((functions[name], ExecutionMode.SINGLE))
        expected.add((functions[name], ExecutionMode.REQUEST_PARALLEL))
    for name in _REPLICATED_NAMES:
        expected.add((functions[name], ExecutionMode.REPLICATED))
    for name in _PARTITIONED_NAMES:
        expected.add((functions[name], ExecutionMode.PARTITIONED))
    return expected


def _bundle_identity(
    extension_sha,
    topology,
    single_sha,
    partitioned_sha,
    replicated_sha,
):
    digest = hashlib.sha256()
    fields = [extension_sha, topology, single_sha, partitioned_sha]
    fields.extend(
        "{}={}".format(name, replicated_sha[name])
        for name in sorted(replicated_sha)
    )
    digest.update("\n".join(fields).encode("utf-8"))
    return digest.hexdigest()


__all__ = [
    "ProductionBootstrapReport",
    "bootstrap_production_bindings",
]
