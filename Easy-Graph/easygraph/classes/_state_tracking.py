"""Weak mutation observers for optional graph-analysis sessions.

Observers live outside graph instances so graph copying/serialization retains
its existing contract. The generation identifies one committed mutation. An
attribute notification refines the generic notification for that generation.
"""
from weakref import WeakKeyDictionary, WeakMethod, ref

_observers = WeakKeyDictionary()


def subscribe(graph, callback):
    weak_callback = WeakMethod(callback) if getattr(callback, '__self__', None) is not None else ref(callback)
    _observers.setdefault(graph, set()).add(weak_callback)

    def unsubscribe():
        listeners = _observers.get(graph)
        if listeners is not None:
            listeners.discard(weak_callback)
            if not listeners:
                _observers.pop(graph, None)
    return unsubscribe


def notify(graph, edge_data=None, keys=None):
    if not _observers:
        return
    listeners = _observers.get(graph)
    if not listeners:
        return
    for weak_callback in tuple(listeners):
        callback = weak_callback()
        if callback is None:
            listeners.discard(weak_callback)
        else:
            callback(graph, graph._mutation_generation, edge_data,
                     None if keys is None else frozenset(keys))
