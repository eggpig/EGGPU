"""Normal Python imports, without a replacement easygraph namespace."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('statement', ['from easygraph.gpu import Session',
                                     'from easygraph import gpu; Session = gpu.Session'])
def test_session_import_defers_optional_application_modules(statement):
    source = Path(__file__).resolve().parents[2]
    script = statement + '''
import sys, easygraph
assert Session.__module__ == 'easygraph.gpu.session'
assert not hasattr(easygraph, '__wrapped__')
assert not any(name in sys.modules for name in
               ('torch', 'sklearn', 'matplotlib', 'easygraph._legacy_exports'))
assert not easygraph._legacy_loaded
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
               PYTHONPATH=os.pathsep.join([str(source), *map(str, sys.path)]))
    result = subprocess.run([sys.executable, '-c', script], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
