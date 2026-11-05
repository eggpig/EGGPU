"""Source-bound state qualification, durable full outputs, and real view leases."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import math
import platform
import sys
import tempfile

import state_protocol as c

PROTOCOL = 'state-complete-workflow-completion-v1'
BENCHMARK_FILES = ('state_worker.py', 'state_contract.py', 'state_protocol.py',
                   'state_reference.py', 'state_inputs.py', 'state_controls.py',
                   'state_rebuild.py', 'baselines/hytgraph_qualify.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def source_binding(source, cpp):
    """Bind actual native binary plus all Python/native source, not six proxies."""
    source = Path(source).resolve()
    files = {}
    suffixes = {'.py', '.cpp', '.cc', '.c', '.cu', '.cuh', '.h', '.hpp', '.cmake'}
    for directory in ('easygraph', 'cpp_easygraph', 'gpu_easygraph'):
        for path in sorted((source / directory).rglob('*')):
            if path.is_file() and (path.suffix in suffixes or path.name == 'CMakeLists.txt'):
                if '__pycache__' not in path.parts:
                    files[str(path.relative_to(source))] = c.bind(path)
    for name in ('setup.py', 'pyproject.toml', 'CMakeLists.txt'):
        if (source / name).is_file():
            files[name] = c.bind(source / name)
    for required in ('easygraph/gpu/session.py', 'easygraph/gpu/_snapshot_lease.py'):
        if required not in files:
            raise ValueError('Required current source missing: ' + required)
    import numpy as np
    import scipy
    root = Path(__file__).resolve().parent
    return {'schema': 'state.source.binding.v4', 'files': files,
            'benchmark': {name: c.bind(root / name) for name in BENCHMARK_FILES},
            'dso': c.bind(cpp.__file__), 'python': c.bind(sys.executable),
            'python_version': platform.python_version(),
            'numpy_version': np.__version__, 'scipy_version': scipy.__version__}


def configuration(a, binding, devices, environment, affinity):
    from state_reference import QUALITY
    return {'protocol': PROTOCOL, 'source_binding': binding, 'experiment': a.experiment,
            'arm': a.arm, 'route': c.ARMS[a.arm][0], 'policy': c.ARMS[a.arm][1],
            'old_readers': a.old_readers, 'beta': a.beta, 'order': a.order,
            'devices': devices, 'gpu_uuids': a.gpu_uuids.split(','),
            'environment': environment, 'cpu_affinity': list(affinity), 'quality': QUALITY,
            'input_schema': 'state.inputs.v2', 'functions': list(c.FUNCTIONS),
            'trace_recipe': {'E3': '20 commits + 80 queries', 'E4': '32 visits + 128 queries',
                             'E3R': '16 readonly queries; identical multiset; grouped versus interleaved',
                             'E5': '24 commits + 96 queries', 'E4_once': 'one visit per graph'},
            'rebuild_scope': ('canonical host partitions, file-manifest import, full device state; catalog disabled'
                             if a.arm == 'Rebuild' else 'not_rebuild'),
            'retention_labels': {'D1CviewLRU':'individual-view LRU with workspace fallback',
                                 'D1Clogical':'existing gdsf history demand / owner-referenced bytes; not GreedyDual'},
            'old_reader_semantics': 'resident FP64 completed-query views; no in-flight kernel'}


def validate_arguments(a, inputs, names):
    if a.phase not in ('tiny', 'target', 'formal'):
        raise ValueError('Unknown phase')
    if inputs['qualification_only'] != (a.phase == 'tiny'):
        raise ValueError('Tiny fixtures and real target/formal inputs must remain separate')
    if a.repeat not in range(10) or (a.phase != 'formal' and a.repeat != 0):
        raise ValueError('Formal repeat is 0..9; tiny/target use repeat 0')
    if a.old_readers != 'none' and a.experiment != 'E3':
        raise ValueError('The old-reader axis is defined for E3 only')
    if a.old_readers not in ('none', 'per_update') or not 0 <= a.beta <= 1:
        raise ValueError('Invalid reader intervention or beta')
    allowed = {'E3': ('Rebuild','D0C1','D1C1'), 'E3R': ('D0C1','D1C1'),
               'E4': ('D1Cnone','D1CviewLRU','D1C0','D1Clogical','D1C1'),
               'E5': ('D0C0','D1C0','D0C1','D1C1')}
    if a.arm not in allowed[a.experiment]:
        raise ValueError('Arm is outside its frozen experiment')
    if not math.isfinite(a.reference_cache_gib) or a.reference_cache_gib <= 0:
        raise ValueError('Reference cache budget must be finite and positive')
    if not names or len(set(names)) != len(names) or not set(names) <= set(inputs['datasets']):
        raise ValueError('Invalid dataset pool')
    if a.phase != 'tiny':
        expected = set(c.POOLS[a.experiment])
        if ((a.experiment in ('E3','E3R') and (len(names) != 1 or not set(names) <= expected)) or
                (a.experiment not in ('E3','E3R') and set(names) != expected)):
            raise ValueError('Dataset pool differs from freeze')
    c.trace(a.experiment, names, a.order, a.phase == 'tiny')
    if a.action == 'run' and a.phase in ('target', 'formal') and a.tiny_proof is None:
        raise ValueError('Exact-arm tiny proof is required')
    if a.action == 'run' and a.phase == 'formal' and a.target_proof is None:
        raise ValueError('Exact-input target proof is required for formal runs')


def check_output(item):
    import numpy as np
    path = c.checked(item['blob'])
    with np.load(path, allow_pickle=False) as archive:
        values = archive['values']
    if list(values.shape) != item['shape'] or values.dtype.str != item['dtype']:
        raise ValueError('Output vector domain changed')
    if hashlib.sha256(memoryview(values).cast('B')).hexdigest() != item['vector_sha256']:
        raise ValueError('Output vector bytes changed')
    return values


class OutputStore:
    """Immutable content-addressed lossless full vectors, shared across runs."""
    def __init__(self, directory):
        self.path = Path(directory).resolve()
        self.path.mkdir(parents=True, exist_ok=True)

    def put(self, values):
        import numpy as np
        values = np.asarray(values)
        if values.ndim != 1 or not values.flags.c_contiguous or values.dtype.hasobject:
            raise ValueError('A complete contiguous numeric vector is required')
        raw = hashlib.sha256(memoryview(values).cast('B')).hexdigest()
        domain = {'dtype': values.dtype.str, 'shape': list(values.shape), 'vector_sha256': raw}
        key = digest(domain)
        target = self.path / key[:2] / (key + '.npz')
        target.parent.mkdir(exist_ok=True)
        with (self.path / '.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existed = target.exists()
            if not existed:
                fd, name = tempfile.mkstemp(prefix=key + '.', suffix='.partial', dir=target.parent)
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        np.savez_compressed(stream, values=values)
                        stream.flush()
                        os.fsync(stream.fileno())
                    # Never overwrite an existing raw evidence object.
                    os.link(name, target)
                finally:
                    Path(name).unlink(missing_ok=True)
            item = dict(domain, object_id=key, blob=c.bind(target), lossless=True,
                        format='npz:values', reused_existing_object=existed)
            check_output(item)
            return item


def validate_proof(path, phase, config, inputs_binding, names, budget):
    proof = json.loads(Path(path).read_text())
    if (proof.get('schema') != 'state.complete.workflow.v4' or
            proof.get('status') != 'PASS_STATE_GPU_QUALIFICATION' or
            proof.get('phase') != phase or proof.get('configuration') != config or
            proof.get('configuration_sha256') != digest(config) or
            not proof.get('execution_complete') or proof.get('quality_pass') is not True):
        raise ValueError('Qualification source/configuration/phase mismatch')
    c.checked(proof['inputs'])
    if phase == 'target' and (proof['inputs'] != inputs_binding or
            proof.get('datasets') != names or proof.get('budget') != budget):
        raise ValueError('Target qualification input/pool/physical budget mismatch')
    expected = c.trace(config['experiment'], proof['datasets'], config['order'], phase == 'tiny')
    queries = [x for x in expected if x['kind'] == 'query']
    if len(proof.get('events', [])) != len(expected) or len(proof.get('checks', [])) != len(queries):
        raise ValueError('Qualification trace is incomplete')
    for expected_event, measured in zip(expected, proof['events']):
        if any(measured.get(k) != v for k, v in expected_event.items()):
            raise ValueError('Qualification trace differs')
    for expected_event, check in zip(queries, proof['checks']):
        if check.get('event') != expected_event or check.get('quality', {}).get('passed') is not True:
            raise ValueError('Qualification output quality missing')
        if not check.get('survives_session_close'):
            raise ValueError('Output lifetime is not qualified')
        check_output(check['output'])
    if config['old_readers'] == 'per_update':
        leases = proof.get('leases', [])
        if len(leases) != sum(x['kind'] == 'commit' for x in expected):
            raise ValueError('Qualification reader lifetime incomplete')
        if not all(x.get('released') and (phase != 'tiny' or x.get('old_weights_bitwise_unchanged')) for x in leases):
            raise ValueError('Qualification readonly lease proof missing')
    return c.bind(path)


def owners(inventory):
    return {row['owner_id']: row for device in inventory['per_device'] for row in device['owners']}


def lease_weight_hashes(read):
    import numpy as np
    result = {}
    for row in read['per_view']:
        values = row['values']
        if values.dtype != np.float64 or values.flags.writeable or not values.flags.owndata or values.base is not None:
            raise ValueError('Old weights must be complete owning readonly FP64 arrays')
        if row['owner_id'] in result:
            raise ValueError('Duplicate old view owner')
        result[row['owner_id']] = {'algorithm': row['algorithm'], 'device_id': row['device_id'],
            'count': len(values), 'sha256': hashlib.sha256(memoryview(values).cast('B')).hexdigest()}
    return result


class OldReader:
    def __init__(self, session, snapshot, acquire, device_count, tiny):
        self.session = session
        self.lease = None
        self.closed = False
        before = owners(session.inventory())
        metadata = session._backend.metadata[snapshot]
        try:
            self.lease = acquire(session, snapshot)
            after = owners(session.inventory())
            increased = {key: row for key, row in after.items()
                         if row.get('active_leases', 0) > before.get(key, {}).get('active_leases', 0)}
            if len(increased) != 2 * device_count:
                raise ValueError('Expected one actual SSSP and weighted-PR lease per device')
            identities = []
            for key, row in increased.items():
                if (key not in before or row['graph_id'] != metadata['id'] or row['generation'] != metadata['generation'] or
                        row['kind'] != 'view' or row['algorithm'] not in ('sssp', 'weighted_pr', 'pagerank', 'pr') or
                        row['active_leases'] != before.get(key, {}).get('active_leases', 0) + 1):
                    raise ValueError('Unexpected native lease ownership change')
                identities.append((row['device_id'], 'sssp' if row['algorithm'] == 'sssp' else 'pr'))
            expected = {(rank, algorithm) for rank in range(device_count) for algorithm in ('sssp', 'pr')}
            if set(identities) != expected or len(set(identities)) != len(identities):
                raise ValueError('Old reader must hold exactly one SSSP and one PR view on every device')
            self.before = before
            self.ids = tuple(increased)
            self.tiny = tiny
            self.original = lease_weight_hashes(self.lease.read_weights()) if tiny else None
            if tiny and set(self.original) != set(self.ids):
                raise ValueError('Full old-weight readback does not cover leased views')
            self.receipt = {'snapshot': snapshot, 'native_graph_id': metadata['id'],
                'native_generation': metadata['generation'], 'owner_ids': list(self.ids),
                'before': {k: before[k] for k in self.ids}, 'held': increased,
                'qualification_readback': tiny, 'old_weights': self.original,
                'released': False, 'semantics': 'completed-query resident view, not an in-flight kernel'}
        except BaseException:
            if self.lease is not None:
                self.lease.close()
            raise

    def close(self):
        if self.closed:
            return self.receipt
        try:
            if self.tiny:
                after = lease_weight_hashes(self.lease.read_weights())
                if after != self.original:
                    raise ValueError('Readonly old FP64 view changed across commit/query batch')
                self.receipt['old_weights_bitwise_unchanged'] = True
        finally:
            self.lease.close()
            self.closed = True
        after = owners(self.session.inventory())
        for key in self.ids:
            if key in after and after[key]['active_leases'] != self.before[key]['active_leases']:
                raise ValueError('Explicit old-reader close retained native lease ownership')
        self.receipt.update(released=True, after_release={k: after.get(k) for k in self.ids})
        return self.receipt
