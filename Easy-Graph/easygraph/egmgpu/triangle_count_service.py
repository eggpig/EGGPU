"""Explicit TC service extension; separate from the attested 16-function bundle.

The host plan is prepared once. Upload/reuse and query execute under the actual
native runtime's SINGLE or PARTITIONED reservation. Registration is a research
extension and does not create AUTO eligibility or a production attestation.
"""
from dataclasses import dataclass, field
from typing import Any

from .types import ExecutionMode


@dataclass(frozen=True, eq=False)
class TriangleCountGraph:
    """Immutable owner of a native exact-TC plan and its declared vertex domain."""
    _plan: Any = field(repr=False, compare=False)
    num_nodes: int
    num_entries: int
    parts: int
    owner_policy: str
    _eggpu_bulk_csr = True
    _mutation_generation = 0
    cflag = 0

    @property
    def num_edges(self): return self.num_entries // 2

    @property
    def _cpp_graph(self): return self

    @property
    def _eggpu_bulk_signature(self):
        return ('triangle-count-plan', id(self._plan), self.num_nodes, self.num_entries,
                self.parts, self.owner_policy)

    def is_directed(self): return False


def prepare_triangle_count(offsets, indices, *, parts=1, owner_policy='edge_work'):
    """Prepare canonical simple symmetric int32 CSR; retain all declared vertices.

    The existing native constructor checks the CSR/undirected-simple contract.
    Input arrays and native preparation are outside subsequent service requests.
    """
    import cpp_easygraph as cpp
    if parts < 1 or owner_policy not in ('range', 'edge_work'):
        raise ValueError('TC requires positive parts and range/edge_work ownership')
    plan = cpp.cpp_egmgpu_prepare_triangle_count(offsets, indices, parts, owner_policy)
    return TriangleCountGraph(plan, len(offsets)-1, len(indices), parts, owner_policy)


class _TriangleCountExecutor:
    def __init__(self, devices, raw_result_sink=None):
        self.eligible_device_groups = (tuple(devices),)
        self._sink = raw_result_sink

    def __call__(self, context, args, kwargs):
        if len(args) != 1 or kwargs:
            raise TypeError('TC service expects one prepared TriangleCountGraph')
        graph = context.graph_snapshot.require_native_payload()
        if not isinstance(graph, TriangleCountGraph) or graph is not args[0]:
            raise TypeError('TC service needs its prepared immutable graph snapshot')
        devices = tuple(context.devices)
        mode = ExecutionMode.SINGLE if graph.parts == 1 else ExecutionMode.PARTITIONED
        if context.mode is not mode or len(devices) != graph.parts:
            raise ValueError('TC owner count, reserved device count and actual service mode differ')
        upload = graph._plan.upload(list(devices), False)
        metrics = graph._plan.query(list(devices), False, False)
        value = metrics['count']
        if type(value) is not int or not 0 <= value < 2**64:
            raise ValueError('Native TC did not return an owning uint64-valued Python integer')
        if self._sink is not None:
            self._sink(dict(value=value, upload=upload, metrics=metrics,
                            devices=list(devices), mode=mode.value))
        return value


def register_triangle_count_bindings(*, devices, raw_result_sink=None):
    """Register one explicit SINGLE/PARTITIONED TC group, leaving bootstrap unchanged.

    Call after configure and before submit. Each process chooses its explicit
    group; this function does not assert that the group has GPU qualification.
    """
    from .api import register_native_executor
    devices = tuple(int(d) for d in devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('TC requires distinct explicit devices')
    mode = ExecutionMode.SINGLE if len(devices) == 1 else ExecutionMode.PARTITIONED
    register_native_executor(triangle_count, mode, _TriangleCountExecutor(devices, raw_result_sink))
    return mode


def triangle_count(graph, *, devices):
    """Return exact global TC through the configured native service, as an owning int.

    Requires prepare_triangle_count and explicit TC binding registration. The
    mode is SINGLE for one owner, PARTITIONED for two or more; AUTO is not used.
    """
    from .api import submit
    if not isinstance(graph, TriangleCountGraph):
        raise TypeError('Use prepare_triangle_count for the TC service input')
    mode = ExecutionMode.SINGLE if graph.parts == 1 else ExecutionMode.PARTITIONED
    return submit(triangle_count, graph, mode=mode, devices=tuple(devices)).result()
