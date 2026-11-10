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
                     consumed=False, reference=None, canonical_wcc=False)
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

    def certify_wcc_producer(value):
        # Called only by ConnectivityExecutor after its union and complete-root
        # kernels, synchronization and fresh host download have all completed.
        state = active.get()
        if (state is None or not state['valid'] or state['consumed']
                or state['thread'] != get_ident() or state['algorithm'] != 'wcc'):
            return False
        reference = state['reference']
        if reference is None or reference() is not value or not eligible(value):
            return False
        state['canonical_wcc'] = True
        return True

    def consume_canonical_wcc(capability, graph_id, value):
        state = active.get()
        # An absent producer proof leaves the ordinary checked fallback usable.
        if state is None or not state.get('canonical_wcc', False):
            return False
        return consume(capability, graph_id, 'wcc', value)

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

    return request, register, consume, certify_wcc_producer, consume_canonical_wcc


(_download_request, _register_download, _consume_download,
 _certify_wcc_producer, _consume_canonical_wcc_download) = _protocol()
del _protocol
