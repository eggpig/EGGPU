"""Stable synchronous entry points for the existing EasyGraph API.

The wrappers in this module are installed once, after :mod:`easygraph` has
finished exporting its public functions.  Installation is deliberately
side-effect free: it neither configures EGMGPU nor starts native workers.

Only an explicitly configured, registered route on an immutable native graph
or a generation-tracked EasyGraph graph is redirected.  All other calls
preserve the original EasyGraph/EGGPU path.  Once a call is redirected,
failures are propagated to the caller; falling back after a failed native
submission would execute the analysis twice and hide a production error.
"""

from __future__ import annotations

import inspect
import math

from collections.abc import Sequence
from functools import wraps
from typing import Any
from typing import Dict
from typing import Mapping
from typing import MutableMapping
from typing import Tuple


# Operation names match the evidence-backed single/native binding registry;
# attribute names are the exact objects exported as ``easygraph.<name>``.
DIRECT_PUBLIC_FUNCTIONS: Tuple[Tuple[str, str], ...] = (
    ("pagerank", "pagerank"),
    ("mst", "minimum_spanning_tree"),
    ("lcc", "clustering"),
    ("wcc", "connected_components"),
    ("scc", "strongly_connected_components"),
    ("bfs", "bfs"),
    ("dijkstra", "single_source_dijkstra"),
    ("bellman_ford", "multi_source_bellman_ford"),
    ("sssp", "multi_source_dijkstra"),
    ("kcore", "k_core"),
    ("bc", "betweenness_centrality"),
    ("closeness", "closeness_centrality"),
    ("effective_size", "effective_size"),
    ("efficiency", "efficiency"),
    ("constraint", "constraint"),
    ("hierarchy", "hierarchy"),
)

_WRAPPER_MARKER = "_egmgpu_direct_public_wrapper"
_ORIGINAL_ATTRIBUTE = "_egmgpu_direct_original"
_OPERATION_ATTRIBUTE = "_egmgpu_operation"


def _wrap_public_function(operation: str, original: Any):
    if not callable(original):
        raise TypeError("EasyGraph public attribute {} is not callable".format(operation))
    if bool(getattr(original, _WRAPPER_MARKER, False)):
        return original

    @wraps(original)
    def synchronous_public_entry(*args, **kwargs):
        # Import lazily so installing wrappers during ``easygraph`` import
        # cannot configure or initialize the runtime.
        from . import api

        graph = args[0] if args else kwargs.get("G")
        if _requires_original_call_validation(
            operation, original, graph, args, kwargs
        ):
            # Keep the pre-existing EasyGraph decorators authoritative for
            # invalid graph types.  In particular, SCC on an undirected graph
            # must raise EasyGraphNotImplemented instead of entering the
            # native connected-components dispatcher.
            return original(*args, **kwargs)
        if not api._direct_dispatch_eligible(synchronous_public_entry, args, kwargs):
            return original(*args, **kwargs)
        # Do not catch either submit-time or execution-time failures.  A
        # routed native request is observable and must never be replayed on
        # the legacy path.
        return api._submit_direct_configured(
            synchronous_public_entry, *args, **kwargs
        ).result()

    setattr(synchronous_public_entry, _WRAPPER_MARKER, True)
    setattr(synchronous_public_entry, _ORIGINAL_ATTRIBUTE, original)
    setattr(synchronous_public_entry, _OPERATION_ATTRIBUTE, str(operation))
    return synchronous_public_entry


def _requires_original_call_validation(
    operation: str,
    original: Any,
    graph: Any,
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> bool:
    """Whether existing EasyGraph validation/semantics must run first.

    Production compatibility is checked again against the frozen snapshot in
    the runtime.  This side-effect-free front-door guard handles cases where
    the legacy API has a decorator or a supported option that the native route
    intentionally does not implement, so direct dispatch never changes the
    public exception or silently narrows an argument.
    """

    if graph is None:
        return False
    is_multigraph = getattr(graph, "is_multigraph", None)
    if callable(is_multigraph) and bool(is_multigraph()):
        return True
    directed = False
    if operation in {"scc", "wcc", "mst", "lcc"}:
        is_directed = getattr(graph, "is_directed", None)
        directed = bool(is_directed()) if callable(is_directed) else False
    if operation == "scc" and not directed:
        return True
    if operation in {"wcc", "mst"} and directed:
        return True
    try:
        bound = inspect.signature(original).bind(*args, **dict(kwargs))
        bound.apply_defaults()
        arguments = bound.arguments
    except (TypeError, ValueError):
        # Let the original callable produce its established argument error.
        return True
    if arguments.get("n_workers") is not None:
        return True
    weight = arguments.get("weight")
    if weight is not None and not isinstance(weight, str):
        return True
    if operation == "lcc" and weight is not None:
        return True
    if operation == "mst":
        if bool(arguments.get("ignore_nan", False)):
            return True
        algorithm = arguments.get("algorithm", "kruskal")
        if not isinstance(algorithm, str) or algorithm not in {
            "kruskal",
            "prim",
            "boruvka",
        }:
            return True
    if operation in {"bellman_ford", "sssp", "bc", "closeness"}:
        sources = arguments.get("sources")
        if sources is not None and (
            not isinstance(sources, Sequence)
            or isinstance(sources, (str, bytes))
        ):
            return True
    if operation == "pagerank":
        try:
            alpha = float(arguments.get("alpha", 0.85))
            maximum = arguments.get("max_iter")
            tolerance = arguments.get("tol")
            if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
                return True
            if maximum is not None and int(maximum) < 1:
                return True
            if tolerance is not None and (
                not math.isfinite(float(tolerance)) or float(tolerance) < 0.0
            ):
                return True
        except (TypeError, ValueError, OverflowError):
            return True
    return False


def install_direct_public_api(namespace: MutableMapping[str, Any]) -> Mapping[str, Any]:
    """Install the 16 stable wrappers into an EasyGraph export namespace.

    The operation is idempotent.  Returning the installed mapping makes the
    exact scheduler identities inspectable in CPU tests without importing a
    CUDA extension.
    """

    if not isinstance(namespace, MutableMapping):
        raise TypeError("EasyGraph export namespace must be mutable")
    installed: Dict[str, Any] = {}
    for operation, attribute in DIRECT_PUBLIC_FUNCTIONS:
        if attribute not in namespace:
            raise RuntimeError(
                "EasyGraph did not export required EGMGPU function {!r}".format(
                    attribute
                )
            )
        wrapped = _wrap_public_function(operation, namespace[attribute])
        namespace[attribute] = wrapped
        installed[operation] = wrapped
    return installed


def direct_original(function: Any) -> Any:
    """Return the legacy callable behind a direct public wrapper."""

    return getattr(function, _ORIGINAL_ATTRIBUTE, function)


def is_direct_public_function(function: Any) -> bool:
    """Whether *function* is one of the stable direct-entry wrappers."""

    return bool(getattr(function, _WRAPPER_MARKER, False))


__all__ = [
    "DIRECT_PUBLIC_FUNCTIONS",
    "direct_original",
    "install_direct_public_api",
    "is_direct_public_function",
]
