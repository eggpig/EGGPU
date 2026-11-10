"""Fixed-domain, read-only node-map lease for the CPU topology probe.

The supplied mapping is already owned by the observed input HostGraph. Inputs
remain immutable during this lease. This is a local adapter, not a mutation
contract change to FastHostGraph or to an EasyGraph object.
"""
from types import MappingProxyType
import time

from easygraph.gpu._arena.fast_host_graph import FastHostGraph


class ImmutableNodeDomain:
    def __init__(self, nodes, node_to_id):
        started = time.perf_counter()
        self.nodes = tuple(nodes)
        self.n = len(self.nodes)
        if len(node_to_id) != self.n or any(node_to_id[node] != i for i, node in enumerate(self.nodes)):
            raise ValueError('Existing node map must exactly match the fixed label sequence')
        # Keep the observed mapping alive without copying or rebuilding it.
        # MappingProxyType prevents writes through every rebuilt graph's API.
        self.node_to_id = MappingProxyType(node_to_id)
        self.source_mapping_identity = id(node_to_id)
        self.registration_seconds = time.perf_counter() - started


class RetainedNodeMapHostGraph(FastHostGraph):
    """All ordinary FastHostGraph state work, sharing a checked fixed domain."""
    def __init__(self, name, n, src, dst, attributes, node_domain, profile=False):
        if n != node_domain.n:
            raise ValueError('Retained node domain does not match graph size')
        # The base factory still validates endpoints/weights and constructs
        # offsets, directions, edge-order maps and independent mutable weights.
        # Its cheap identity-label placeholder is replaced before publication.
        super().__init__(name, n, src, dst, attributes, nodes=None, profile=profile)
        self.nodes = node_domain.nodes
        self.node_to_id = node_domain.node_to_id
        self.node_domain = node_domain  # Strong lease until the last graph dies.
