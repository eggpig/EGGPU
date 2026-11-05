"""Single-request provenance for fresh arena host downloads.

Only the internal download producer registers arrays. A public array, a metrics
dictionary, OWNDATA or a copied token is not a certificate. The context keeps a
weak reference, never an owning reference or an answer cache. This private Python
protocol is not a security boundary against code that mutates private modules.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import get_ident
import weakref

import numpy as np


def _protocol():
    active = ContextVar('eggpu_private_arena_download', default=None)

    def eligible(value):
        return (type(value) is np.ndarray and value.dtype == np.dtype(np.float64)
                and value.ndim == 1 and value.flags.owndata
                and value.flags.c_contiguous and value.base is None)

    @contextmanager
    def request(graph_id, algorithm):
        # An opaque, identity-checked capability is minted for exactly this call.
        capability = object()
        state = dict(capability=capability, graph=graph_id, algorithm=algorithm,
                     thread=get_ident(), valid=True, registered=False,
                     consumed=False, reference=None)
        previous = active.set(state)
        try:
            yield capability
        finally:
            state.update(valid=False, reference=None)
            active.reset(previous)

    def register(value):
        state = active.get()
        if state is None or not state['valid'] or state['thread'] != get_ident():
            return
        # Multiple download producers in one request make provenance ambiguous.
        # Fail closed to the ordinary bridge copy without changing computation.
        if state['registered'] or state['consumed'] or not eligible(value):
            state.update(valid=False, reference=None)
            return
        state.update(registered=True, reference=weakref.ref(value))

    def consume(capability, graph_id, algorithm, value):
        state = active.get()
        if (state is None or capability is not state['capability']
                or state['thread'] != get_ident() or not state['valid']
                or state['consumed'] or state['graph'] != graph_id
                or state['algorithm'] != algorithm
                or algorithm not in ('bfs', 'sssp', 'pr', 'weighted_pr', 'wcc')):
            return False
        # Consume even a rejected attempted transfer: never retry a capability.
        state['consumed'] = True
        reference = state['reference']
        state['reference'] = None
        return reference is not None and reference() is value and eligible(value)

    return request, register, consume


_download_request, _register_download, _consume_download = _protocol()
del _protocol
