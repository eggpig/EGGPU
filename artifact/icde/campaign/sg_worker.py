"""E1 single-GPU complete-task worker. Only the experiment owner runs --execute-gpu.

A job is one system/graph/operator/repeat. Source-dependent jobs spawn three
fresh Python processes, serially, under one global GPU lease. Hashing, file IO,
CPU references and quality checks are outside both measured intervals.
"""
from pathlib import Path
import argparse
import contextlib
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ALGORITHMS = ('bfs', 'sssp', 'wcc', 'pr')
DATASETS = ('EP', 'OR', 'IN', 'UK', 'LJ', 'US', 'TW')
TINY_DATASETS = ('TINY_D', 'TINY_U')
PROTOCOL = 'e1-host-csr-complete-v1'
PR_ALPHA = .85
PR_TOLERANCE = 1e-9
PR_MAX_ITER = 2000
QUALITY = {'pr_residual_l1': 1e-6, 'pr_mass_error': 1e-6,
           'pr_reference_l1': 1e-5, 'integer_distances_exact': True,
           'wcc_partition_equivalence': True}


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b''):
            h.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def content_binding(path):
    item = binding(path)
    return {k: item[k] for k in ('bytes', 'sha256')}


def load_input(directory, *, verify=True):
    """Read immutable, source-qualified CSR; never write to shared input files."""
    import numpy as np
    directory = Path(directory).resolve()
    receipt_path = directory / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'PASS_CANONICAL_INPUT':
        raise ValueError('Input is not PASS_CANONICAL_INPUT')
    required = {'offsets': 'offsets.bin', 'indices': 'indices.bin',
                'weights': 'weights.bin', 'canonical': 'canonical.json', 'sources': 'sources.json'}
    files = receipt['files']
    for role, name in required.items():
        expected = files[role]
        path = directory / name
        if path.stat().st_size != expected['bytes']:
            raise ValueError('Input byte count changed: ' + role)
        if verify and content_binding(path) != {k: expected[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Input content changed: ' + role)
    # Optional mapping files must also be bound when the preparation receipt includes them.
    for role, item in files.items():
        if role not in required:
            path = directory / Path(item['path']).name
            if verify and content_binding(path) != {k: item[k] for k in ('bytes', 'sha256')}:
                raise ValueError('Auxiliary canonical input changed: ' + role)
    meta = json.loads((directory / 'canonical.json').read_text())
    if (meta.get('format') != 'eggpu-csr-v1' or type(meta.get('directed')) is not bool
            or meta.get('offset_dtype') != 'int32' or meta.get('index_dtype') != 'int32'
            or meta.get('weight_dtype') != 'float64' or meta.get('weight_rule') != 'w16-v1'
            or meta.get('node_labels') != 'zero_based_contiguous'
            or not meta.get('sorted_indices') or meta.get('endianness') != 'little'):
        raise ValueError('Canonical input contract differs')
    n, m = meta['num_nodes'], meta['num_entries']
    if type(n) is not int or type(m) is not int or not 0 < n < 2**31 or not 0 <= m < 2**31:
        raise ValueError('Input dimensions exceed the frozen int32 contract')
    if any(files[role]['bytes'] != size for role, size in
           [('offsets', 4 * (n + 1)), ('indices', 4 * m), ('weights', 8 * m)]):
        raise ValueError('Canonical shape/byte count mismatch')
    sources = json.loads((directory / 'sources.json').read_text())
    chosen = sources.get('primary_sources')
    if (not isinstance(chosen, list) or len(chosen) != 3 or len(set(chosen)) != 3
            or any(type(s) is not int or not 0 <= s < n for s in chosen)
            or sources.get('selection_uses_timing') is not False):
        raise ValueError('Exactly three distinct topology-selected sources are required')
    offsets = np.memmap(directory / 'offsets.bin', dtype='<i4', mode='r', shape=(n+1,))
    indices = np.memmap(directory / 'indices.bin', dtype='<i4', mode='r', shape=(m,)) if m else np.empty(0, np.int32)
    weights = np.memmap(directory / 'weights.bin', dtype='<f8', mode='r', shape=(m,)) if m else np.empty(0, np.float64)
    if offsets[0] != 0 or offsets[-1] != m or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError('Invalid CSR offsets')
    for start in range(0, m, 1 << 20):
        v, w = indices[start:start+(1 << 20)], weights[start:start+(1 << 20)]
        if np.any(v < 0) or np.any(v >= n) or np.any(w < 1) or np.any(w > 16) or np.any(w != np.floor(w)):
            raise ValueError('Invalid endpoint or w16 weight')
    identity = {'receipt': content_binding(receipt_path),
                'files': {role: {k: item[k] for k in ('bytes', 'sha256')}
                          for role, item in files.items()}, 'sources': chosen,
                'n': n, 'm': m, 'directed': meta['directed']}
    return dict(n=n, m=m, directed=meta['directed'], offsets=offsets, indices=indices,
                weights=weights, sources=chosen, identity=identity,
                input_sha256=digest_json(identity), directory=str(directory))


def make_tiny(root):
    """CPU-only asymmetric/reciprocal fixtures, including disconnected and dangling nodes."""
    import numpy as np
    for name, directed in [('TINY_D', True), ('TINY_U', False)]:
        out = Path(root) / 'canonical' / name
        out.mkdir(parents=True, exist_ok=False)
        edges = {(0, 1), (1, 2), (2, 0), (2, 3), (4, 5), (5, 4)}
        if not directed:
            edges |= {(v, u) for u, v in edges}
        edges = sorted(edges)
        offsets = np.r_[0, np.cumsum(np.bincount([u for u, _ in edges], minlength=7))].astype('<i4')
        indices = np.array([v for _, v in edges], dtype='<i4')
        values = []
        for u, v in edges:
            if not directed:
                u, v = min(u, v), max(u, v)
            key = f'eggpu-w16-v1|20261104|{name}|{u}|{v}'.encode()
            values.append(1 + int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % 16)
        offsets.tofile(out / 'offsets.bin'); indices.tofile(out / 'indices.bin')
        np.asarray(values, dtype='<f8').tofile(out / 'weights.bin')
        meta = dict(format='eggpu-csr-v1', generation=1, name=name, abbreviation=name,
                    num_nodes=7, num_entries=len(edges), num_edges=len(edges) if directed else len(edges)//2,
                    directed=directed, weighted=True, offset_dtype='int32', index_dtype='int32',
                    weight_dtype='float64', weight_rule='w16-v1', endianness='little',
                    node_labels='zero_based_contiguous', sorted_indices=True,
                    allow_self_loops=False, duplicate_policy='simple',
                    offsets_path='offsets.bin', indices_path='indices.bin', weights_path='weights.bin')
        save(out / 'canonical.json', meta)
        save(out / 'sources.json', dict(primary_sources=[0, 1, 2], diagnostic_source=6,
                                       selection_uses_timing=False, rule='fixed synthetic qualification fixture'))
        save(out / 'receipt.json', dict(status='PASS_CANONICAL_INPUT', dataset=name,
             synthetic=True, performance_eligible=False,
             files={role: binding(out / filename) for role, filename in
                    [('offsets', 'offsets.bin'), ('indices', 'indices.bin'), ('weights', 'weights.bin'),
                     ('canonical', 'canonical.json'), ('sources', 'sources.json')]}))


def cgroup_memory():
    """Account for visible cgroup v2 ancestor limits, not just host MemAvailable."""
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable'):
            mem[key] = int(value.split()[0]) * 1024
    root = Path('/sys/fs/cgroup')
    relative = next((line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                     if line.startswith('0::')), '/')
    leaf = root / relative.lstrip('/')
    if not leaf.is_dir():
        leaf = root  # cgroup-namespace mount already represents this process's group.
    limits = []
    for path in [leaf, *leaf.parents]:
        if path != root and root not in path.parents:
            continue
        maximum, current = path / 'memory.max', path / 'memory.current'
        if maximum.exists() and current.exists():
            raw = maximum.read_text().strip()
            value = int(current.read_text().strip())
            limits.append({'path': str(path), 'max': raw, 'current': value,
                           'headroom': None if raw == 'max' else max(0, int(raw) - value)})
    mem['cgroup_limits'] = limits
    mem['available_bytes'] = min([mem['MemAvailable']] + [x['headroom'] for x in limits if x['headroom'] is not None])
    if not limits:
        raise RuntimeError('Cannot inspect cgroup v2 memory limits; resource qualification unavailable')
    return mem


def resource_check(args, graph):
    expected = json.loads(args.setup.read_text())['gpus']['output']
    allowed = {row[1].strip() for row in list(csv.reader(io.StringIO(expected)))[1:] if len(row) > 1}
    for probe in range(5):
        text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version',
                                        '--format=csv,noheader,nounits'], text=True)
        gpus = [dict(zip(('index', 'uuid', 'name', 'total_mib', 'used_mib', 'utilization', 'driver'),
                         [field.strip() for field in row])) for row in csv.reader(io.StringIO(text))]
        selected = next((g for g in gpus if g['uuid'] == args.gpu_uuid), None)
        if selected is None or args.gpu_uuid not in allowed:
            raise RuntimeError('GPU UUID does not match the authorized dedicated server setup')
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                            '--format=csv,noheader,nounits'], text=True).strip()
        # All-card exclusion enforces serial experiments even when this job requests one GPU.
        if processes:
            raise RuntimeError('GPU machine has active processes; no processes are terminated')
        if all(int(g['used_mib']) <= 64 and int(g['utilization']) == 0 for g in gpus):
            break
        if probe == 4:
            raise RuntimeError('GPU machine is occupied; no processes are terminated')
        # nvidia-smi may briefly report the previous, already exited child in its
        # utilization sampling window. Recheck idle state, never overlap jobs.
        time.sleep(1)
    if int(selected['total_mib']) - int(selected['used_mib']) < args.budget_mib + 512:
        raise RuntimeError('GPU free memory cannot admit fixed arena budget plus reserve')
    mem = cgroup_memory()
    required = max(int(args.min_host_free_gib * (1 << 30)), 128 * graph['m'] + 128 * graph['n'] + (2 << 30))
    if mem['available_bytes'] < required:
        raise RuntimeError(f'Host/cgroup headroom insufficient: need {required}, have {mem["available_bytes"]}')
    return {'gpus': gpus, 'selected': selected, 'processes': processes, 'host': mem,
            'required_host_bytes': required, 'checked_unix': time.time()}


def source_epoch(args):
    """Bind code and installed binary payloads without importing a GPU backend."""
    files = {'worker': binding(__file__), 'adapter': binding(HERE / 'cugraph_csr_adapter.py')}
    release = args.release.resolve()
    sources = {}
    for base in [release / 'easygraph', release / 'cpp_easygraph']:
        if base.is_dir():
            for path in sorted(base.rglob('*')):
                if path.is_file() and path.suffix in ('.py', '.cu', '.cpp', '.c', '.h', '.hpp', '.cuh'):
                    sources[str(path.relative_to(release))] = content_binding(path)
    if args.system == 'eggpu':
        if not (release / 'easygraph/gpu/session.py').is_file():
            raise ValueError('Release does not contain the frozen Session.create route')
        files['native_dso'] = binding(args.native_dso)
        libraries = sorted(args.arena_library_dir.glob('*.so'))
        if not (args.arena_library_dir / 'libeggpu_native_bfs.so').is_file():
            raise ValueError('A prebuilt bound arena library is required; benchmark cannot build it')
        files['arena_libraries'] = {p.name: binding(p) for p in libraries}
    packages = {}
    prefixes = ('numpy', 'scipy', 'cupy', 'pylibcugraph', 'pylibraft', 'rmm', 'libcugraph', 'libraft', 'cuda-python')
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get('Name', '').lower().replace('_', '-')
        if any(name == prefix or name.startswith(prefix + '-') for prefix in prefixes):
            payload = {}
            for item in distribution.files or []:
                path = Path(distribution.locate_file(item))
                if path.is_file() and ('.so' in path.name or path.suffix in ('.py', '.pyx', '.pxd')
                                       or path.name in ('RECORD', 'METADATA')):
                    payload[str(item)] = content_binding(path)
            packages[name] = {'version': distribution.version, 'files': payload}
    environment = {key: value for key, value in os.environ.items()
                   if key.startswith('EGGPU_') or key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS', 'CUDA_DEVICE_ORDER', 'CUPY_CACHE_DIR', 'NVCC', 'LD_LIBRARY_PATH',
                    'PYTHONPATH', 'CUDA_PATH', 'CUDA_HOME')}
    epoch = dict(protocol=PROTOCOL, system=args.system, files=files, release_sources=sources,
                 packages=packages, python=binding(sys.executable), python_version=platform.python_version(),
                 host=platform.node(), kernel=platform.release(), affinity=sorted(os.sched_getaffinity(0)),
                 environment=environment, gpu_uuid=args.gpu_uuid, setup=content_binding(args.setup),
                 arena_budget_mib=args.budget_mib, pr_tolerance_l1=PR_TOLERANCE,
                 pr_max_iter=PR_MAX_ITER, warmups=1, quality=QUALITY,
                 code_cache_policy='persistent compiled-code cache; fresh process/device graph state per source')
    epoch['observed_gpu_environment'] = subprocess.check_output([
        'nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version',
        '--format=csv,noheader,nounits'], text=True).strip()
    return epoch, digest_json(epoch)


def validate_proof(path, *, kind, system, algorithm, epoch_sha, graph):
    proof = json.loads(Path(path).read_text())
    if (proof.get('status') != 'PASS_QUALIFICATION' or proof.get('phase') != kind
            or proof.get('system') != system or proof.get('algorithm') != algorithm
            or proof.get('epoch_sha256') != epoch_sha or proof.get('quality_pass') is not True):
        raise ValueError('Qualification proof does not match current source/binary/environment/operator')
    if kind == 'tiny':
        if proof.get('dataset') not in TINY_DATASETS or proof.get('directed') != graph['directed']:
            raise ValueError('Tiny qualification must cover the target directedness')
    elif proof.get('input_sha256') != graph['input_sha256']:
        raise ValueError('Target qualification input or three-source selection changed')
    epoch_item = proof.get('epoch', {})
    if not epoch_item or binding(epoch_item['path']) != epoch_item:
        raise ValueError('Qualification epoch receipt changed')
    if digest_json(json.loads(Path(epoch_item['path']).read_text())) != epoch_sha:
        raise ValueError('Qualification epoch content differs')
    expected_count = 3 if algorithm in ('bfs', 'sssp') else 1
    if len(proof.get('sources', [])) != expected_count:
        raise ValueError('Qualification did not cover the complete source block')
    for source in proof['sources']:
        child = source.get('result', {})
        if not child or binding(child['path']) != child:
            raise ValueError('Qualification source receipt changed')
        data = json.loads(Path(child['path']).read_text())
        if data.get('status') != 'PASS' or data.get('quality_pass') is not True:
            raise ValueError('Qualification contains a failed source')
        for label in ('first', 'warmup', 'ready'):
            call = data['calls'][label]
            if not call['quality']['pass'] or binding(call['output']['path']) != call['output']:
                raise ValueError('Qualification complete output or quality evidence changed')
    return binding(path)


def canonical_labels(values):
    import numpy as np
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'iuf' or np.any(~np.isfinite(values)) or np.any(values != np.floor(values)):
        raise ValueError('WCC output must be complete finite integral labels')
    _, inverse = np.unique(values, return_inverse=True)
    minima = np.full(int(inverse.max()) + 1, len(values), dtype=np.int64)
    np.minimum.at(minima, inverse, np.arange(len(values)))
    return minima[inverse]


def graph_matrix(graph, weighted=False):
    import numpy as np
    from scipy.sparse import csr_matrix
    return csr_matrix((graph['weights'] if weighted else np.ones(graph['m'], dtype=np.float64),
                       graph['indices'], graph['offsets']), shape=(graph['n'], graph['n']), copy=False)


def pr_step(graph, ranks, matrix=None):
    import numpy as np
    matrix = graph_matrix(graph) if matrix is None else matrix
    degree = np.diff(graph['offsets'])
    scaled = np.divide(ranks, degree, out=np.zeros_like(ranks), where=degree != 0)
    return PR_ALPHA * (matrix.T @ scaled + float(ranks[degree == 0].sum()) / graph['n']) + (1 - PR_ALPHA) / graph['n']


def reference(graph, algorithm, source):
    import numpy as np
    from scipy.sparse.csgraph import connected_components, dijkstra
    matrix = graph_matrix(graph, weighted=algorithm == 'sssp')
    if algorithm in ('bfs', 'sssp'):
        return dijkstra(matrix, directed=graph['directed'], indices=source,
                        unweighted=algorithm == 'bfs', return_predecessors=False)
    if algorithm == 'wcc':
        return canonical_labels(connected_components(matrix, directed=graph['directed'],
                                connection='weak', return_labels=True)[1])
    ranks = np.full(graph['n'], 1. / graph['n'], dtype=np.float64)
    for iteration in range(10000):
        updated = pr_step(graph, ranks, matrix)
        if np.linalg.norm(updated - ranks, ord=1) <= 1e-12:
            return updated
        ranks = updated
    raise RuntimeError('Independent CPU PageRank reference did not converge')


def reference_cached(graph, algorithm, source, directory):
    import numpy as np
    identity = dict(input_sha256=graph['input_sha256'], algorithm=algorithm, source=source,
                    worker=content_binding(__file__), numpy=np.__version__,
                    scipy=importlib.metadata.version('scipy'), alpha=PR_ALPHA)
    key = digest_json(identity)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with (directory / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        values_path, receipt_path = directory / (key + '.npy'), directory / (key + '.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['identity'] != identity or content_binding(values_path) != receipt['output']:
                raise ValueError('Independent reference cache changed')
        else:
            values = reference(graph, algorithm, source)
            # Never replace an existing blob left by an interrupted or different run.
            with values_path.open('xb') as stream:
                np.save(stream, values, allow_pickle=False)
            receipt = {'identity': identity, 'output': content_binding(values_path),
                       'method': 'SciPy CPU Dijkstra/weak-components or independent FP64 power iteration'}
            save(receipt_path, receipt)
        return np.load(values_path, mmap_mode='r', allow_pickle=False), binding(receipt_path)


def audit_output(values, expected, graph, algorithm):
    import numpy as np
    values = np.asarray(values)
    result = {'complete': values.shape == (graph['n'],), 'dtype': str(values.dtype),
              'owning': bool(values.flags.owndata), 'immutable': not values.flags.writeable}
    if not result['complete'] or np.any(np.isnan(values)) or np.any(np.isneginf(values)):
        result['pass'] = False
        return result
    if algorithm == 'wcc':
        result['partition_equivalent'] = bool(np.array_equal(canonical_labels(values), expected))
        result['pass'] = result['partition_equivalent']
    elif algorithm == 'pr':
        if values.dtype != np.float64 or not np.all(np.isfinite(values)) or np.any(values < 0):
            result['pass'] = False
            return result
        result.update(residual_l1=float(np.abs(pr_step(graph, values) - values).sum()),
                      mass_error=float(abs(values.sum() - 1.)),
                      reference_l1=float(np.abs(values - expected).sum()))
        result['pass'] = bool(values.dtype == np.float64 and np.all(values >= 0) and np.all(np.isfinite(values))
                              and result['residual_l1'] <= QUALITY['pr_residual_l1']
                              and result['mass_error'] <= QUALITY['pr_mass_error']
                              and result['reference_l1'] <= QUALITY['pr_reference_l1'])
    else:
        result['exact_reference_match'] = bool(np.array_equal(values, expected))
        finite = np.isfinite(values) & np.isfinite(expected)
        error = values[finite] - expected[finite]
        result['rmse_finite'] = float(np.sqrt(np.mean(error * error))) if error.size else 0.
        result['unreachable_match'] = bool(np.array_equal(np.isposinf(values), np.isposinf(expected)))
        result['pass'] = result['exact_reference_match']
    result['pass'] = bool(result['pass'] and result['owning'] and result['immutable'])
    return result


def persist_output(directory, values):
    """Deduplicate equal outputs only within this new job; never mutate old arrays."""
    import numpy as np
    h = hashlib.sha256(str(values.dtype).encode() + str(values.shape).encode())
    view = memoryview(values).cast('B')
    for start in range(0, len(view), 16 << 20):
        h.update(view[start:start+(16 << 20)])
    path = Path(directory) / (h.hexdigest() + '.npy')
    if not path.exists():
        with path.open('xb') as stream:
            np.save(stream, values, allow_pickle=False)
    return binding(path)


@contextlib.contextmanager
def complete_timeout(seconds):
    def expired(signum, frame):
        raise TimeoutError('Complete task exceeded frozen timeout')
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def build_backend(args, graph, source):
    """Called inside T_first: every backend-specific layout and upload is charged."""
    import numpy as np
    if args.system == 'eggpu':
        sys.path.insert(0, str(args.release.resolve()))
        sys.path.insert(1, str(args.native_dso.resolve().parent))
        from easygraph.gpu import Session
        session_module = sys.modules[Session.__module__]
        if Path(session_module.__file__).resolve() != (args.release / 'easygraph/gpu/session.py').resolve():
            raise RuntimeError('Imported Session does not match the bound release source')
        native_module = sys.modules.get('cpp_easygraph')
        if native_module is not None and Path(native_module.__file__).resolve() != args.native_dso.resolve():
            raise RuntimeError('Imported native extension differs from the bound DSO')
        # Candidate: common host CSR enters the existing owned-CSR store directly.
        # Validation, registration, copies, preparation and upload remain timed.
        payload = {'nodes': np.arange(graph['n'], dtype=np.int64),
                   'offsets': graph['offsets'], 'indices': graph['indices'],
                   'weights': graph['weights']}
        session = Session.create({'g': payload}, devices=(0,), state_model='static',
                                 algorithms=ALGORITHMS, input_format='csr',
                                 budget_bytes=args.budget_mib << 20,
                                 pr_tolerance_l1=PR_TOLERANCE, pr_max_iter=PR_MAX_ITER)
        params = {'source': source} if args.algorithm in ('bfs', 'sssp') else {}
        return session, lambda: session.query('g', args.algorithm, params).values
    from cugraph_csr_adapter import CuGraphCSR
    prepared = graph
    if args.algorithm == 'wcc' and graph['directed']:
        matrix = graph_matrix(graph)
        weak = matrix.maximum(matrix.T).tocsr()
        weak.sort_indices()
        if weak.nnz >= 2**31:
            raise OverflowError('INDEX_UNSUPPORTED: weak projection exceeds int32; no edges removed')
        prepared = dict(graph, directed=False, m=int(weak.nnz),
                        offsets=weak.indptr.astype(np.int32, copy=False),
                        indices=weak.indices.astype(np.int32, copy=False))
    backend = CuGraphCSR(prepared, args.algorithm, source=source or 0)
    def query():
        output = backend.complete_query()
        if not output.flags.owndata or not output.flags.c_contiguous:
            output = np.array(output, copy=True, order='C')
        output.setflags(write=False)
        return output
    return backend, query


def source_child(args):
    """Private subprocess entry requires an inherited exclusive lock descriptor."""
    import numpy as np
    if args.lease_fd is None or args.lease_fd < 0:
        raise RuntimeError('GPU child requires inherited global lease')
    inherited = os.fstat(args.lease_fd)
    actual = args.lock_file.stat()
    if (inherited.st_dev, inherited.st_ino) != (actual.st_dev, actual.st_ino):
        raise RuntimeError('Inherited descriptor is not the requested global lock')
    # A new open description must fail LOCK_NB while the parent holds the lease.
    with args.lock_file.open('a') as independent:
        try:
            fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('Global lease is not held by the parent')
    graph = load_input(args.input_root / 'canonical' / args.dataset)
    source = graph['sources'][args.source_index] if args.algorithm in ('bfs', 'sssp') else None
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    record = {'status': 'RUNNING', 'source': source, 'source_index': args.source_index,
              'timing_start': 'common immutable host CSR ready; fresh Python backend process',
              'input_sha256': graph['input_sha256'], 'calls': {}}
    save(out / 'result.json', record)
    backend = None
    try:
        # Read every host page before starting the backend-specific interval.
        # Input hashing above also touches all bytes; this checksum prevents a
        # future metadata-only verification change from silently making CSR lazy.
        record['host_ready_checksum'] = [int(graph['offsets'].sum(dtype=np.int64)),
                                         int(graph['indices'].sum(dtype=np.int64)),
                                         float(graph['weights'].sum(dtype=np.float64))]
        outputs = {}
        save(out / 'stage.json', {'active': True, 'label': 'first', 'started_monotonic': time.monotonic()})
        with complete_timeout(args.timeout):
            start = time.perf_counter()
            backend, query = build_backend(args, graph, source)
            outputs['first'] = query()
            first_seconds = time.perf_counter() - start
        save(out / 'stage.json', {'active': False, 'label': 'first'})
        record['calls']['first'] = {'seconds': first_seconds}
        save(out / 'stage.json', {'active': True, 'label': 'warmup', 'started_monotonic': time.monotonic()})
        with complete_timeout(args.timeout):
            outputs['warmup'] = query()
        save(out / 'stage.json', {'active': False, 'label': 'warmup'})
        save(out / 'stage.json', {'active': True, 'label': 'ready', 'started_monotonic': time.monotonic()})
        with complete_timeout(args.timeout):
            start = time.perf_counter()
            outputs['ready'] = query()
            ready_seconds = time.perf_counter() - start
        save(out / 'stage.json', {'active': False, 'label': 'ready'})
        record['calls']['ready'] = {'seconds': ready_seconds}
        record['calls']['warmup'] = {'timed': False, 'count': 1}
        record['backend'] = (backend.dispatch_plan.to_dict() if args.system == 'eggpu' else backend.contract)
        record['backend_input_semantics'] = ('stored directed arcs; reciprocal canonical input already contains both orientations'
            if args.system == 'eggpu' else 'original canonical CSR; directed WCC builds explicit weak CSR inside T_first')
        backend.close(); backend = None
        expected, reference_receipt = reference_cached(graph, args.algorithm, source, args.reference_cache)
        record['reference_receipt'] = reference_receipt
        for label, values in outputs.items():
            if any(np.shares_memory(values, other) for name, other in outputs.items() if name != label):
                raise ValueError('Output buffers alias across calls')
            record['calls'][label]['output'] = persist_output(out, values)
            record['calls'][label]['quality'] = audit_output(values, expected, graph, args.algorithm)
        record['quality_pass'] = all(call['quality']['pass'] for call in record['calls'].values())
        record['status'] = 'PASS' if record['quality_pass'] else 'FAIL_QUALITY'
        record['outputs_survive_close'] = True
    except BaseException as error:
        record.update(status='TIMEOUT' if isinstance(error, TimeoutError) else 'FAILED',
                      error=type(error).__name__ + ': ' + str(error), quality_pass=False)
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as error:
                record['close_error'] = type(error).__name__ + ': ' + str(error)
                record['status'], record['quality_pass'] = 'FAILED_CLOSE', False
        save(out / 'result.json', record)
    return record


def run_child(command, log, lease_fd, child, timeout):
    """Parent watchdog also terminates a hung native CUDA call's owned process group."""
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               pass_fds=(lease_fd,), env=dict(os.environ), start_new_session=True)
    try:
        while process.poll() is None:
            stage_path = child / 'stage.json'
            if stage_path.exists():
                stage = json.loads(stage_path.read_text())
                if stage.get('active') and time.monotonic() - stage['started_monotonic'] > timeout + 1:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    save(child / 'watchdog.json', {'status': 'TIMEOUT', 'stage': stage['label'],
                         'limit_seconds': timeout, 'action': 'killed only this worker-owned process group'})
                    return process.returncode
            time.sleep(.25)
        return process.returncode
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def job(args):
    if args.dataset in TINY_DATASETS and args.phase != 'tiny':
        raise ValueError('Synthetic inputs may only be used for tiny qualification')
    if args.phase == 'tiny' and args.dataset not in TINY_DATASETS:
        raise ValueError('Tiny qualification requires the fixed synthetic fixture')
    if args.repeat < 0 or (args.phase == 'formal' and args.repeat not in range(10)):
        raise ValueError('Formal repeat must be one of 0..9')
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(schema='eggpu.e1.sg.job.v1', protocol=PROTOCOL, experiment='E1',
                  phase=args.phase, system=args.system, dataset=args.dataset, algorithm=args.algorithm,
                  repeat=args.repeat, gpu_count=1, status='RUNNING', quality_pass=False,
                  performance_eligible=False, sources=[], source_processes='fresh process per source',
                  complete_task_timeout_seconds=args.timeout, started_unix=time.time())
    save(args.output / 'result.json', record)
    try:
        if not args.lock_file.parent.is_dir():
            raise ValueError('Global lock parent must already exist; use the machine-wide lock path')
        with args.lock_file.open('a') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            graph = load_input(args.input_root / 'canonical' / args.dataset)
            minimum_output_space = 8 * graph['n'] * (9 if args.algorithm in ('bfs', 'sssp') else 3) + (2 << 30)
            if shutil.disk_usage(args.output).free < minimum_output_space:
                raise RuntimeError('Insufficient disk space for complete retained outputs plus reserve')
            record.update(input=graph['identity'], input_sha256=graph['input_sha256'], directed=graph['directed'])
            # Freeze environment before hashing it and pass exactly that environment
            # to each fresh source process. No entire-environment dump is recorded.
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
            os.environ['EGGPU_ARENA_LIBRARY_DIR'] = str(args.arena_library_dir.resolve())
            epoch, epoch_sha = source_epoch(args)
            record['epoch_sha256'] = epoch_sha
            save(args.output / 'epoch.json', epoch)
            record['epoch'] = binding(args.output / 'epoch.json')
            if args.phase in ('target', 'formal'):
                if args.tiny_proof is None:
                    raise ValueError('Current tiny GPU qualification is required')
                record['tiny_proof'] = validate_proof(args.tiny_proof, kind='tiny', system=args.system,
                    algorithm=args.algorithm, epoch_sha=epoch_sha, graph=graph)
            if args.phase == 'formal':
                if args.target_proof is None:
                    raise ValueError('Current target-input GPU qualification is required')
                record['target_proof'] = validate_proof(args.target_proof, kind='target', system=args.system,
                    algorithm=args.algorithm, epoch_sha=epoch_sha, graph=graph)
            source_count = 3 if args.algorithm in ('bfs', 'sssp') else 1
            for index in range(source_count):
                admission = resource_check(args, graph)
                child = args.output / f'source_{index}'
                command = [sys.executable, str(Path(__file__).resolve()), *forward_options(args),
                           '--source-index', str(index), '--lease-fd', str(lease.fileno()),
                           '--output', str(child)]
                with (args.output / f'source_{index}.log').open('wb') as log:
                    returncode = run_child(command, log, lease.fileno(), child, args.timeout)
                item = json.loads((child / 'result.json').read_text()) if (child / 'result.json').exists() else {
                    'status': 'FAILED_PROCESS', 'quality_pass': False}
                if (child / 'watchdog.json').exists():
                    item['status'], item['quality_pass'] = 'TIMEOUT', False
                record['sources'].append(dict(index=index, source=graph['sources'][index] if source_count == 3 else None,
                    result=binding(child / 'result.json') if (child / 'result.json').exists() else None,
                    returncode=returncode, status=item['status'], admission=admission,
                    quality_pass=item.get('quality_pass', False), calls=item.get('calls', {})))
                save(args.output / 'result.json', record)
                if returncode or item.get('status') != 'PASS':
                    raise RuntimeError(f'Source {index} failed; see its retained output/log')
            # Detect live edits or binary/environment replacement during this job.
            _, final_epoch_sha = source_epoch(args)
            if final_epoch_sha != epoch_sha:
                raise RuntimeError('Source/binary/environment changed during the job')
            record['quality_pass'] = True
            record['T_first_seconds'] = sum(s['calls']['first']['seconds'] for s in record['sources']) / source_count
            record['T_ready_seconds'] = sum(s['calls']['ready']['seconds'] for s in record['sources']) / source_count
            record['source_aggregation'] = 'arithmetic equal-weight mean; this entire job is one repeat'
            record['performance_eligible'] = args.phase == 'formal'
            record['status'] = 'PASS_QUALIFICATION' if args.phase in ('tiny', 'target') else 'PASS'
    except BaseException as error:
        record.update(status='FAILED', error=type(error).__name__ + ': ' + str(error), quality_pass=False)
        raise
    finally:
        record['finished_unix'] = time.time()
        save(args.output / 'result.json', record)
    return record


def forward_options(args):
    values = ['--execute-gpu', '--system', args.system, '--algorithm', args.algorithm,
              '--dataset', args.dataset, '--phase', args.phase, '--repeat', str(args.repeat)]
    for name in ('input_root', 'release', 'arena_library_dir', 'native_dso', 'setup',
                 'gpu_uuid', 'lock_file', 'reference_cache', 'budget_mib', 'min_host_free_gib', 'timeout'):
        values += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    return values


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--make-tiny', type=Path, help='CPU only: create canonical/TINY_D and TINY_U below this new root')
    p.add_argument('--execute-gpu', action='store_true')
    p.add_argument('--system', choices=('eggpu', 'cugraph'))
    p.add_argument('--algorithm', choices=ALGORITHMS)
    p.add_argument('--dataset', choices=DATASETS + TINY_DATASETS)
    p.add_argument('--phase', choices=('tiny', 'diagnostic', 'target', 'formal'), default='diagnostic')
    p.add_argument('--repeat', type=int, default=0)
    for name in ('input-root', 'release', 'arena-library-dir', 'native-dso', 'setup', 'lock-file',
                 'reference-cache', 'output', 'tiny-proof', 'target-proof'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--gpu-uuid')
    p.add_argument('--budget-mib', type=int, default=22000)
    p.add_argument('--min-host-free-gib', type=float, default=8.)
    p.add_argument('--timeout', type=float, default=1800.)
    p.add_argument('--source-index', type=int, help=argparse.SUPPRESS)
    p.add_argument('--lease-fd', type=int, help=argparse.SUPPRESS)
    return p


def main():
    p = parser(); args = p.parse_args()
    if args.make_tiny:
        if args.execute_gpu:
            p.error('--make-tiny is a CPU-only operation')
        make_tiny(args.make_tiny)
        print(json.dumps({'status': 'PASS_TINY_INPUTS_CREATED', 'gpu_executed': False}))
        return
    required = ('system', 'algorithm', 'dataset', 'input_root', 'release', 'arena_library_dir', 'native_dso',
                'setup', 'gpu_uuid', 'lock_file', 'reference_cache', 'output')
    if not args.execute_gpu or any(getattr(args, name) is None for name in required):
        p.error('GPU work requires --execute-gpu and all input, binary, resource and output bindings')
    if args.budget_mib <= 0 or args.min_host_free_gib <= 0 or args.timeout != 1800.:
        p.error('Positive budgets and the frozen 1800-second complete-task timeout are required')
    result = source_child(args) if args.source_index is not None else job(args)
    print(json.dumps({key: result[key] for key in ('status', 'quality_pass')}, allow_nan=False))


if __name__ == '__main__':
    main()
