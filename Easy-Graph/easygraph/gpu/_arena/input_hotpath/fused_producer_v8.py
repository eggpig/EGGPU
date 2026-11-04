"""V7 fast producer with charged fallback for non-native integer byte order.

Only the dispatch contract changes. The native signed-integer branch is the
frozen V7 producer; no caller buffers are adopted or edited.
"""
from time import perf_counter
import numpy as np
from easygraph.gpu._arena.input_hotpath.fused_producer_v7 import _produce as _native_produce
from easygraph.gpu._arena.input_hotpath.owned_csr_v6 import _produce as _fallback


def _produce(nodes, edges, weights):
    started = perf_counter()
    edge_array = np.asarray(edges)
    if edge_array.dtype.kind in 'iu' and not edge_array.dtype.isnative:
        answer = _fallback(nodes, edge_array, weights)
        answer[2].update(producer_total_seconds=perf_counter() - started,
                         byteorder_fallback=True,
                         producer_dispatch='non_native_integer_original_fallback')
        return answer
    answer = _native_produce(nodes, edge_array, weights)
    answer[2]['producer_total_seconds'] = perf_counter() - started
    return answer


def prepare_owned(data):
    prepared, profiles = {}, {}
    for name, item in data.items():
        nodes, certificate, profile = _produce(item['nodes'], item['edges'], item['weights'])
        prepared[name] = dict(nodes=nodes, edges=certificate, weights=None)
        profiles[name] = profile
    return prepared, profiles
