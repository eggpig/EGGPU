"""CPU import isolation and retained legacy context behavior; no CUDA calls."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


PRELUDE = '''
import sys, types, importlib.abc
def no_cuda(*args, **kwargs):
    raise AssertionError('CUDA invocation is forbidden in CPU import tests')
cp = types.ModuleType('cupy')
cp.RawModule = cp.RawKernel = no_cuda
sys.modules['cupy'] = cp
'''

BLOCK_LEGACY = '''
class RejectLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'sklearn', 'matplotlib', 'pandas') or fullname in (
            'easygraph.classes', 'easygraph.utils', 'easygraph.functions', 'easygraph._legacy_exports'):
            raise AssertionError('Unexpected legacy import: ' + fullname)
sys.meta_path.insert(0, RejectLegacy())
'''


def run(script):
    source = Path(__file__).resolve().parents[2]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join([str(source), *map(str, sys.path)]))
    result = subprocess.run([sys.executable, '-B', '-c', PRELUDE + script], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('statement', [
    'from easygraph.gpu._arena.public_session_final import PublicExecutor',
    'from easygraph.gpu._arena.public_session_release import PublicExecutor',
    '''
from easygraph.gpu._arena.input_hotpath import owned_csr_v6 as owned
from easygraph.gpu._arena.input_hotpath.confirm_candidate_v1 import ConfirmCandidateAdapter
class ReachedOwnedSession(Exception): pass
def stop_before_allocation(self, *args, **kwargs): raise ReachedOwnedSession
owned.OwnedCSRSession.__init__ = stop_before_allocation
try:
    ConfirmCandidateAdapter({}, budget_bytes=4096)
except ReachedOwnedSession:
    pass
else:
    raise AssertionError('The actual adapter must reach the owned session constructor')
''',
])
def test_actual_arena_construction_import_chain_avoids_legacy(statement):
    run(BLOCK_LEGACY + statement + '''
import easygraph
assert not easygraph._legacy_loaded
assert not any(name in sys.modules for name in
    ('torch', 'sklearn', 'matplotlib', 'pandas', 'easygraph.classes', 'easygraph.utils'))
''')


def test_legacy_context_still_installs_nested_dispatch_and_restores_originals():
    run('''
from easygraph.gpu._arena import public_session_final as public
from types import SimpleNamespace
utils = types.ModuleType('easygraph.utils')
backend = types.ModuleType('easygraph.utils.gpu_eggpu_backend')
originals = {}
for name in public._names:
    def original(graph, *args, _name=name, **kwargs): return ('original', _name)
    originals[name] = original
    setattr(backend, name, original)
utils.gpu_eggpu_backend = backend
sys.modules['easygraph.utils'] = utils
sys.modules['easygraph.utils.gpu_eggpu_backend'] = backend
closed = []
def make(label):
    session = public.EasyGraphSession.__new__(public.EasyGraphSession)
    session.closed = False
    session._context_token = None
    session.endpoint_unsubscribe, session.unsubscribe = [], []
    session.engine = SimpleNamespace(close=lambda: closed.append(label))
    for name in ('pending', 'tables', 'graphs', 'edge_positions', 'labels', 'index', 'names', 'endpoint_depth'):
        setattr(session, name, {})
    session.accepts = lambda *args: True
    session._dispatch = lambda graph, name, kwargs: (label, name)
    return session
a, b = make('outer'), make('inner')
with a:
    assert backend.pagerank(None) == ('outer', 'pagerank')
    with b:
        assert backend.pagerank(None) == ('inner', 'pagerank')
    assert backend.pagerank(None) == ('outer', 'pagerank')
assert closed == ['inner', 'outer']
assert public._users == 0 and public._active.get() is None
assert all(getattr(backend, name) is original for name, original in originals.items())
a.close()
assert closed == ['inner', 'outer']
''')


def test_legacy_constructor_loads_observer_only_when_used():
    run('''
from easygraph.gpu._arena import public_session_final as public
from types import SimpleNamespace
classes = types.ModuleType('easygraph.classes')
classes.__path__ = []
tracking = types.ModuleType('easygraph.classes._state_tracking')
tracking.subscribe = lambda *args: None
sys.modules['easygraph.classes'] = classes
sys.modules['easygraph.classes._state_tracking'] = tracking
calls = []
def typed(inputs, *args, **kwargs):
    calls.append((inputs, args, kwargs))
    return SimpleNamespace(executor=object(), close=lambda: None)
public.TypedSession = typed
public.PublicExecutor = lambda executor: executor
session = public.EasyGraphSession({}, budget_bytes=4096)
assert calls[0][0] == [] and calls[0][1][0] == 4096
session.close()
assert session.closed
''')
