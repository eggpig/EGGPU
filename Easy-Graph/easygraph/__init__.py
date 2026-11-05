"""EasyGraph public API with on-demand optional application exports.

Importing a subpackage such as ``easygraph.gpu`` does not initialize unrelated
machine-learning, drawing and dataset modules. Root-level graph/functions and
wildcard imports retain the original export/bootstrap behavior on first use.
"""
from importlib import import_module as _import_module
from threading import RLock as _RLock

_legacy_lock = _RLock()
_legacy_loaded = False
_legacy_loading = False
_legacy_module = None
_SUBMODULES = frozenset(('classes', 'convert', 'datapipe', 'datasets', 'exception',
                        'experiments', 'functions', 'ml_metrics', 'model', 'nn',
                        'readwrite', 'utils', 'egmgpu', 'gpu'))



def _load_legacy_exports():
    global _legacy_loaded, _legacy_loading, _legacy_module
    with _legacy_lock:
        if _legacy_loaded:
            return _legacy_module
        if _legacy_loading:
            # A partially imported legacy module may probe the root while
            # Python resolves a submodule. Let normal import handling proceed.
            raise AttributeError('EasyGraph legacy exports are still initializing')
        _legacy_loading = True
        try:
            module = _import_module('._legacy_exports', __name__)
            module.load_into(globals())
            _legacy_module = module
            _legacy_loaded = True
        finally:
            _legacy_loading = False
        return _legacy_module


def __getattr__(name):
    if name in _SUBMODULES:
        return _import_module('.' + name, __name__)
    if name.startswith('__') and name != '__all__':
        raise AttributeError("module 'easygraph' has no attribute {!r}".format(name))
    module = _load_legacy_exports()
    if name == '__all__':
        # Recompute so a subsequently imported public submodule keeps the same
        # wildcard visibility as the original root without a static __all__.
        return [key for key in globals() if not key.startswith('_')]
    if name in globals():
        return globals()[name]
    if not globals()['_HAS_TORCH'] and name in {'Hypergraph', 'BaseHypergraph', 'load_structure'}:
        globals()['_require_torch'](name)
    raise AttributeError("module 'easygraph' has no attribute {!r}".format(name))


def __dir__():
    _load_legacy_exports()
    return sorted(globals())
