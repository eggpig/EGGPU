"""Deferred execution of the original root export/bootstrap sequence.

This initializes the real package namespace before importing egmgpu, whose
idempotent public wrapper installer deliberately inspects that namespace.
"""
from importlib import import_module
import sys

_MODULES = ('classes', 'convert', 'datapipe', 'datasets', 'exception', 'experiments',
            'functions', 'ml_metrics', 'model', 'nn', 'readwrite', 'utils')


def load_into(namespace):
    modules = [import_module('easygraph.' + name) for name in _MODULES]
    optional = import_module('easygraph._optional')
    namespace['_HAS_TORCH'] = optional.HAS_TORCH
    namespace['_require_torch'] = optional.require_torch
    namespace['_warn_missing_torch'] = optional.warn_missing_torch
    # The old dotted imports bind the package itself as the public name
    # ``easygraph`` in addition to attaching each submodule to that package.
    namespace['easygraph'] = sys.modules['easygraph']
    for module in modules:
        names = getattr(module, '__all__', None)
        if names is None:
            names = [name for name in vars(module) if not name.startswith('_')]
        for name in names:
            namespace[name] = getattr(module, name)
    namespace['egmgpu'] = import_module('easygraph.egmgpu')
    direct = import_module('easygraph.egmgpu.direct_dispatch')
    direct.install_direct_public_api(namespace)
    optional.warn_missing_torch()
