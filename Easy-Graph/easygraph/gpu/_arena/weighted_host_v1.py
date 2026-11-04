"""Experimental CSR-segment reduction; frozen V4 kernels and lifecycle unchanged.

The initial DemandState has outgoing.ids == arange(m) by construction. Its
nonempty CSR rows partition values[:m], so source sums need no edge-length
source labels or gathered weight copy. Only this initial immutable state uses
the shortcut. Inherited updates publish ordinary PreparedState instances and
retain their existing update/rebuild semantics.
"""
from time import perf_counter
import numpy as np
from easygraph.gpu._arena.input_hotpath.csr_session_v4 import DemandState
from easygraph.gpu._arena.candidate_v4 import DemandCandidateAdapter


def segmented_sums(row, values):
    """Nonnegative FP64 CSR row sums, including interior/trailing empty rows."""
    result = np.zeros(len(row) - 1, dtype=np.float64)
    nonempty = row[1:] != row[:-1]
    starts = row[:-1][nonempty]
    if starts.size:
        result[nonempty] = np.add.reduceat(values[:int(row[-1])], starts,
                                          dtype=np.float64)
    return result


class SegmentedDemandState(DemandState):
    def __getattr__(self, name):
        if name != 'sums':
            return super().__getattr__(name)
        started = perf_counter()
        value = segmented_sums(self.outgoing.row, self.values)
        self.__dict__[name] = value
        self.host_view_seconds[name] = perf_counter() - started
        return value


class WeightedHostCandidate(DemandCandidateAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Layout-compatible promotion keeps every existing snapshot/reference
        # pointing at the same immutable object. No buffers or answers copied.
        # Only exact initial DemandState objects qualify; updated generations
        # are ordinary PreparedState and use the frozen inherited code.
        for graph in self.session.graphs.values():
            state = graph.r2_state
            if type(state) is DemandState:
                state.__class__ = SegmentedDemandState
