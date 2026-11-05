"""Consume an explicitly bound independent CPU oracle across runtime versions.

The producer's source/runtime identity remains evidence. The consumer's NumPy
version must not choose another oracle. No computation or cache search occurs.
"""
from pathlib import Path
import hashlib
import json

SCHEMA = 'eggpu.e6.independent-reference-manifest/v1'
FIELDS = {'effective_size', 'constraint', 'closeness', 'bfs', 'wcc', 'candidates', 'winner'}


def binding(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def check(item):
    if binding(item['path']) != item:
        raise ValueError('E6 explicit independent reference binding changed')
    return Path(item['path'])


def load(graph, manifest_path, contract, reused_path):
    import numpy as np
    manifest = json.loads(Path(manifest_path).read_text())
    if (manifest.get('schema') != SCHEMA or manifest.get('protocol') != contract or
            manifest.get('gpu_executed') is not False or
            manifest.get('performance_eligible') is not False):
        raise ValueError('Wrong E6 independent reference manifest or contract')
    producer = manifest['producer']
    check(producer['worker'])
    check(producer['reused'])
    current_reused = binding(reused_path)
    if any(current_reused[k] != producer['reused'][k] for k in ('sha256', 'bytes')):
        raise ValueError('E6 independent oracle formulas changed')
    rows = [r for r in manifest['references'] if r['input_sha256'] == graph['input_sha256']]
    if len(rows) != 1:
        raise ValueError('E6 needs one explicit oracle for this exact input')
    row = rows[0]
    receipt = check(row['receipt'])
    record = json.loads(receipt.read_text())
    expected = dict(input_sha256=graph['input_sha256'], protocol=contract,
        code={k: producer['worker'][k] for k in ('bytes', 'sha256')},
        reused={k: producer['reused'][k] for k in ('bytes', 'sha256')},
        numpy=producer['numpy'], scipy=producer['scipy'])
    if (record.get('identity') != expected or record.get('gpu_executed') is not False or
            record.get('performance_eligible') is not False):
        raise ValueError('Wrong independent oracle producer/source/input/runtime identity')
    blob = check(row['output'])
    if record.get('output') != {k: row['output'][k] for k in ('bytes', 'sha256')}:
        raise ValueError('Independent oracle payload does not match producer receipt')
    with np.load(blob, allow_pickle=False) as z:
        if set(z.files) != FIELDS:
            raise ValueError('Incomplete independent oracle fields')
        values = {name: z[name].copy() for name in z.files}
    for name, value in values.items():
        count = 1 if name == 'winner' else min(32, graph['n']) if name in ('candidates', 'closeness') else graph['n']
        if value.shape != (count,):
            raise ValueError('Wrong independent oracle vector domain')
    return values, row['receipt']
