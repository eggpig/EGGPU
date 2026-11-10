"""Preserve custom edge mappings while observing their public mutations."""
import copy
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import MutableMapping

from ._state_tracking import notify


_types = {}
_shared = weakref.WeakValueDictionary()
_INTERNAL = frozenset(('_eg_owners', '_eg_depth', '_eg_restoring', '_eg_factory_original'))


def _slots(cls):
    for base in cls.__mro__:
        names = base.__dict__.get('__slots__', ())
        for name in (names,) if isinstance(names, str) else names:
            if name not in ('__dict__', '__weakref__'):
                if name.startswith('__') and not name.endswith('__'):
                    name = '_' + base.__name__.lstrip('_') + name
                yield name


def _instance_state(mapping):
    state = {k: v for k, v in getattr(mapping, '__dict__', {}).items() if k not in _INTERNAL}
    slots = {name: getattr(mapping, name) for name in _slots(type(mapping)) if hasattr(mapping, name)}
    if isinstance(mapping, defaultdict):
        slots['default_factory'] = mapping.default_factory
    return state, slots


def _raw_setitem(mapping, key, value):
    if isinstance(mapping, OrderedDict):
        OrderedDict.__setitem__(mapping, key, value)
    else:
        dict.__setitem__(mapping, key, value)


def _raw_items(mapping):
    return OrderedDict.items(mapping) if isinstance(mapping, OrderedDict) else dict.items(mapping)


def _empty_mapping(base):
    tracked = _tracked_type(base)
    if issubclass(base, OrderedDict):
        result = OrderedDict.__new__(tracked)
    elif issubclass(base, defaultdict):
        result = defaultdict.__new__(tracked)
    elif issubclass(base, dict):
        result = dict.__new__(tracked)
    else:
        result = object.__new__(tracked)
    result._eg_owners = []
    result._eg_depth = 0
    result._eg_restoring = True
    return result


def _snapshot(mapping):
    # Read actual dictionary contents without triggering __missing__ or custom
    # value transformations. Other MutableMappings expose their logical items.
    return dict(dict.items(mapping)) if isinstance(mapping, dict) else dict(mapping.items())


def _different(a, b):
    if a is b:
        return False
    try:
        return not bool(a == b)
    except (TypeError, ValueError):
        return True


class _FactoryTrackedMixin:
    """Mixin keeps the factory class in the MRO, including its custom methods."""

    def __init__(self, *args, **kwargs):
        # Built-in mapping copy/fromkeys methods may construct type(self)
        # directly instead of using the graph factory or pickle reconstructor.
        self._eg_owners = []
        self._eg_depth = 0
        self._eg_restoring = False
        super().__init__(*args, **kwargs)

    def _has_owner(self, owner):
        return any(ref() is owner for ref in self._eg_owners)

    def _bind_owner(self, owner):
        if self._has_owner(owner):
            return
        weak_self = weakref.ref(self)
        def discard(ref):
            mapping = weak_self()
            if mapping is not None:
                mapping._eg_owners = [r for r in mapping._eg_owners if r is not ref]
        self._eg_owners.append(weakref.ref(owner, discard))

    def _apply(self, method, *args, **kwargs):
        if self._eg_restoring:
            if method == '__setitem__' and isinstance(self, dict):
                return _raw_setitem(self, *args)
            return getattr(super(), method)(*args, **kwargs)
        outer = self._eg_depth == 0
        before = _snapshot(self) if outer else None
        self._eg_depth += 1
        succeeded = False
        try:
            result = getattr(super(), method)(*args, **kwargs)
            succeeded = True
            return result
        finally:
            self._eg_depth -= 1
            if outer:
                after = _snapshot(self)
                base = self._eg_base_type
                missing = getattr(base, '__missing__', None)
                precise = (issubclass(base, dict) and base.get is dict.get
                           and base.__getitem__ is dict.__getitem__
                           and missing in (None, defaultdict.__missing__))
                keys = set(before).symmetric_difference(after)
                keys.update(k for k in before.keys() & after.keys() if _different(before[k], after[k]))
                # An explicit assignment may commit an already-mutated mutable
                # value. A successful write with equal contents conservatively
                # invalidates current keys; rejected writes do not notify.
                if not keys and succeeded and method in ('__setitem__', 'update', '__ior__'):
                    keys.update(after)
                if keys or (succeeded and not precise):
                    for ref in tuple(self._eg_owners):
                        owner = ref()
                        if owner is not None:
                            owner._clear_cache()
                            # Custom lookup can alias keys or derive one value
                            # from another. Retain edge scope while invalidating
                            # all its attributes when key equivalence is opaque.
                            notify(owner, self, keys if precise else None)

    def __setitem__(self, key, value):
        return self._apply('__setitem__', key, value)

    def __delitem__(self, key):
        return self._apply('__delitem__', key)

    def update(self, *args, **kwargs):
        return self._apply('update', *args, **kwargs)

    def clear(self):
        return self._apply('clear')

    def pop(self, key, *args):
        return self._apply('pop', key, *args)

    def popitem(self, *args, **kwargs):
        return self._apply('popitem', *args, **kwargs)

    def setdefault(self, key, default=None):
        return self._apply('setdefault', key, default)

    def __ior__(self, other):
        method = getattr(super(), '__ior__', None)
        if method is None:
            self.update(other)
            return self
        return self._apply('__ior__', other)

    def __setstate__(self, state):
        attrs, slots = state
        self.__dict__.update(attrs)
        for name, value in slots.items():
            object.__setattr__(self, name, value)
        self._eg_restoring = False

    def __reduce_ex__(self, protocol):
        # Reconstruct through a stable module-level function, never pickle a
        # dynamically generated class or weak graph owners. State follows dict
        # items, so raw loading does not replay custom validation/conversion.
        return (_empty_mapping, (self._eg_base_type,), _instance_state(self), None,
                iter(_raw_items(self)) if isinstance(self, dict) else None)

    def __deepcopy__(self, memo):
        result = _empty_mapping(self._eg_base_type)
        memo[id(self)] = result
        if isinstance(self, dict):
            for key, value in _raw_items(self):
                _raw_setitem(result, copy.deepcopy(key, memo), copy.deepcopy(value, memo))
        result.__setstate__(copy.deepcopy(_instance_state(self), memo))
        return result

    def __copy__(self):
        method = getattr(super(), '__copy__', None)
        if method is not None:
            result = method()
            if result is not self and isinstance(result, _FactoryTrackedMixin):
                result._eg_owners = []
                result._eg_depth = 0
                result._eg_restoring = False
                result.__dict__.pop('_eg_factory_original', None)
            return result
        result = _empty_mapping(self._eg_base_type)
        if isinstance(self, dict):
            for key, value in _raw_items(self):
                _raw_setitem(result, key, value)
        result.__setstate__(_instance_state(self))
        return result


def _tracked_type(base):
    if base not in _types:
        _types[base] = type('_Tracked_' + base.__name__, (_FactoryTrackedMixin, base),
                            {'_eg_base_type': base, '__module__': __name__})
    return _types[base]


def bind_factory_mapping(mapping, owner):
    """Track a factory result once, preserving shared-factory edge aliases.

    The returned object remains an instance of the factory's mapping class.
    Its actual dict storage backs JSON/dict operations; this is not a dict
    facade with hidden data. Mutations through the returned public edge mapping
    notify every bound graph. The raw factory seed is retained only to identify
    repeated returns of that same object, and is not a mutation API.
    """
    if not isinstance(mapping, MutableMapping):
        raise TypeError('edge_attr_dict_factory must return a mutable mapping')
    if isinstance(mapping, _FactoryTrackedMixin):
        mapping._bind_owner(owner)
        return mapping
    key = id(mapping)
    tracked = _shared.get(key)
    if tracked is None:
        tracked = _empty_mapping(type(mapping))
        if isinstance(mapping, dict):
            # OrderedDict's iterator order is observable and must be retained.
            for name, value in _raw_items(mapping):
                _raw_setitem(tracked, name, value)
        tracked.__setstate__(_instance_state(mapping))
        tracked._eg_factory_original = mapping
        _shared[key] = tracked
    tracked._bind_owner(owner)
    return tracked
