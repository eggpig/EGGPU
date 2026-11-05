"""Source-bound E6 workflow; no GPU work without --execute-gpu.

Existing structural, selected-source centrality and traversal implementations
are composed into one actual result-dependent task. Root owns GPU execution.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time

import sg_worker as sg
import e6_reused as reused
import e6_cpu_reference

HERE = Path(__file__).resolve().parent
PROTOCOL = 'e6-weak-structural-dependent-complete-v1'
DATASETS = ('EP', 'LJ', 'OR')
TINY = 'E6_TINY_U'
FUNCTIONS = ('effective_size', 'constraint', 'closeness', 'bfs', 'wcc')
GROUPS = {'structural': ('effective_size', 'constraint'), 'closeness': ('closeness',),
          'bfs': ('bfs',), 'wcc': ('wcc',)}
SYSTEMS = ('eggpu', 'igraph', 'cugraph_igraph', 'cugraph_composite')
ABLATIONS = ('none', *GROUPS)
RTOL, ATOL = 1e-8, 1e-10
CONTRACT = {
    'input': 'same source-bound canonical host CSR; construct simple unweighted weak projection inside T_first; retain every canonical vertex including isolates',
    'weights': 'original w16-v1 files remain input-bound; all E6 operators are explicitly unweighted',
    'selection': 'all finite ES/Constraint rows; descending ES, ascending Constraint, canonical ID, after fixed round-to-even 1e-6 quantization; first min(32, finite vertices)',
    'winner': 'maximum exact Wasserman-Faust closeness among actual selected vertices; fixed 1e-6 rounding then minimum canonical ID',
    'closeness': '32 exact sources, not a full all-vertex closeness ranking; r^2/((n-1)*sum_distance), zero for r=0',
    'complete': 'all ES and Constraint values, candidate IDs, every candidate closeness value, winner, full BFS distances and full canonical WCC partition on host',
    'timing': 'T_first starts with common host CSR ready and includes weak projection, imports, graph formats, uploads, all five calls, host materialization, both dependent selections; T_ready retains graph state after one full warmup; close recorded separately',
    'ablation': 'one of four fixed groups delegates its actual work to igraph C/derived CPU; unchanged groups use the same native ABI; no precomputed outputs or omitted functions',
    'mg': 'whole graph replicated on each GPU; ego-vertex shards for structural metrics and selected-source shards for closeness; BFS/WCC on logical GPU0; not partitioned capacity scaling',
    'quality': {'rtol': RTOL, 'atol': ATOL, 'bfs_exact': True, 'wcc_partition_exact': True,
                'nan_mask_exact': True, 'selection_exact': True, 'owning_immutable': True},
    'formal_repetitions': 10, 'timeout_seconds': 1800,
    'task_origin': 'controlled structural screening workflow, not an observed production trace',
}


def make_tiny(root):
    """Forty vertices, two components and four isolates; 32 real candidates for P8."""
    import numpy as np
    out = Path(root)/'canonical'/TINY; out.mkdir(parents=True, exist_ok=False)
    n = 40
    pairs = {(i, (i+1) % 20) for i in range(20)}
    pairs |= {(i, 20+(i-20+1) % 16) for i in range(20, 36)}
    pairs |= {(i, j) for i in range(4, 10) for j in range(i+1, 10)}
    pairs = {tuple(sorted(pair)) for pair in pairs}
    edges = np.asarray(sorted(pairs | {(v, u) for u, v in pairs}), dtype=np.int32)
    offsets = np.r_[0, np.cumsum(np.bincount(edges[:, 0], minlength=n))].astype('<i4')
    weights = []
    for u, v in edges:
        u, v = sorted((int(u), int(v)))
        key = f'eggpu-w16-v1|20261104|{TINY}|{u}|{v}'.encode()
        weights.append(1 + int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % 16)
    offsets.tofile(out/'offsets.bin'); edges[:, 1].astype('<i4').tofile(out/'indices.bin')
    np.asarray(weights, dtype='<f8').tofile(out/'weights.bin')
    sg.save(out/'canonical.json', dict(format='eggpu-csr-v1', generation=1, name=TINY,
        abbreviation=TINY, num_nodes=n, num_entries=len(edges), num_edges=len(pairs),
        directed=False, weighted=True, offset_dtype='int32', index_dtype='int32', weight_dtype='float64',
        weight_rule='w16-v1', endianness='little', node_labels='zero_based_contiguous', sorted_indices=True,
        allow_self_loops=False, duplicate_policy='simple', offsets_path='offsets.bin',
        indices_path='indices.bin', weights_path='weights.bin'))
    sg.save(out/'sources.json', dict(primary_sources=[0, 20, 35], diagnostic_source=39,
        selection_uses_timing=False, rule='fixed qualification fixture; E6 winner still comes from actual computed scores'))
    sg.save(out/'receipt.json', dict(status='PASS_CANONICAL_INPUT', dataset=TINY,
        synthetic=True, performance_eligible=False, files={role: sg.binding(out/name) for role, name in
        [('offsets', 'offsets.bin'), ('indices', 'indices.bin'), ('weights', 'weights.bin'),
         ('canonical', 'canonical.json'), ('sources', 'sources.json')]}))
    return out


def weak_graph(graph):
    """Shared semantic transform; execution belongs inside complete first-call time."""
    import numpy as np
    a = sg.graph_matrix(graph)
    a = a.maximum(a.T).tocsr()
    a.setdiag(0); a.eliminate_zeros(); a.sum_duplicates(); a.sort_indices()
    if a.nnz >= 2**31:
        raise OverflowError('INDEX_UNSUPPORTED: weak projection exceeds int32; no truncation')
    offsets = a.indptr.astype(np.int32, copy=False)
    indices = a.indices.astype(np.int32, copy=False)
    degree = np.diff(offsets)
    rows = np.repeat(np.arange(graph['n'], dtype=np.int64), degree)
    edges = np.column_stack((rows, indices)).astype(np.int64, copy=False)
    return {'n': graph['n'], 'm': int(a.nnz), 'degree': degree, 'offsets': offsets,
            'indices': indices, 'edges': edges, 'undirected': edges[edges[:, 0] < edges[:, 1]],
            'directed': False, 'parent_input_sha256': graph['input_sha256']}


def native_arrays(data):
    """Construct exactly the file-backed CSR views directly in owned native arrays.

    Called during the first timed workflow. Conversion, projection construction,
    native validation and ownership copies remain inside T_first.
    """
    import numpy as np
    n, degree = data['n'], data['degree']
    lower = data['undirected']; forward = lower.copy()
    swap = ((degree[lower[:, 0]] > degree[lower[:, 1]]) |
            ((degree[lower[:, 0]] == degree[lower[:, 1]]) & (lower[:, 0] > lower[:, 1])))
    forward[swap] = forward[swap, ::-1]
    projection = {'unique_edge_count': len(lower)}
    for name, pairs in [('lower', lower), ('upper', lower[:, ::-1]), ('forward', forward)]:
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        offsets = np.r_[0, np.cumsum(np.bincount(pairs[:, 0], minlength=n))]
        projection[name + '_V'] = np.ascontiguousarray(offsets, dtype=np.int32)
        projection[name + '_E'] = np.ascontiguousarray(pairs[:, 1], dtype=np.int32)
    projection['degree'] = np.ascontiguousarray(degree, dtype=np.int32)
    return {'offsets': np.ascontiguousarray(data['offsets'], dtype=np.int32),
            'indices': np.ascontiguousarray(data['indices'], dtype=np.int32),
            'undirected_projection': projection}


class DenseNative(reused.NativeBackend):
    """Owned memory CSR input with the existing dense structural output ABI."""
    def __init__(self, p, j, data):
        super().__init__(p, j, data)
        self.input_data = data

    def build(self):
        if self.graph is None:
            payload = native_arrays(self.input_data)
            self.graph = self.cpp.cpp_graph_from_csr_arrays(
                payload['offsets'], payload['indices'], directed=True,
                validate=True, undirected_projection=payload['undirected_projection'])

    def call(self, function, selected=None):
        if len(self.devices) == 1 or function not in GROUPS['structural']:
            return super().call(function, selected)
        self.build()
        raw = getattr(self.cpp, 'cpp_egmgpu_replicated_' + function)(
            self.graph, None, None, self.devices, True, output_format='dense')
        if list(raw['devices_used']) != self.devices or 'values' in raw:
            raise ValueError('Wrong replicated dense route or device set')
        if len(raw['nodes_dense']) != self.data_n or len(raw['values_dense']) != self.data_n:
            raise ValueError('Incomplete dense structural output')
        import numpy as np
        if not np.array_equal(raw['nodes_dense'], np.arange(self.data_n)):
            raise ValueError('Dense structural result does not use the canonical vertex order')
        return reused.own(raw['values_dense']), dict(mode='replicated_graph_ego_vertex_shards',
            devices_used=list(raw['devices_used']), kernel_seconds=float(raw['kernel_seconds']),
            output_format='dense', public_Session=False)


class CoverageBackend:
    def __init__(self, args, data, directory):
        p = dict(source=str(args.release), native_dso={'path': str(args.native_dso)},
                 datasets={})
        j = {'P': args.gpu_count, 'dataset': args.dataset, 'phase': args.phase}
        self.native = DenseNative(p, j, data); self.native.data_n = data['n']
        if Path(self.native.eg.__file__).resolve() != (args.release/'easygraph/__init__.py').resolve():
            self.native.close()
            raise ValueError('Imported EasyGraph source is not the bound release')
        self.cpu = reused.IgraphBackend(p, j, data) if args.ablate != 'none' else None
        self.fallback = GROUPS.get(args.ablate, ())

    def call(self, function, selected=None):
        delegated = function in self.fallback
        values, metrics = (self.cpu if delegated else self.native).call(function, selected)
        metrics = dict(metrics, coverage_fallback=delegated,
                       native_input_format='owned_in_memory_csr')
        return values, metrics

    def close(self):
        try:
            self.native.close()
        finally:
            if self.cpu is not None:
                self.cpu.close()


def build_backend(args, data, directory):
    if args.system == 'eggpu':
        return CoverageBackend(args, data, directory)
    p = {'fallback_site': str(args.fallback_site or ''), 'composition': 'gpu_composite'}
    j = {'phase': 'qualification' if args.phase in ('tiny', 'target') else args.phase}
    if args.fallback_site:
        sys.path.append(str(args.fallback_site))
    classes = {'igraph': reused.IgraphBackend, 'cugraph_igraph': reused.MixedBackend,
               'cugraph_composite': reused.GPUCompositeBackend}
    return classes[args.system](p, j, data)


def flow(backend, n, clock=time.perf_counter):
    """One causal execution: never substitute reference scores or a preset winner."""
    import numpy as np
    start = clock(); stages = []; outputs = {}
    def call(function, selected=None):
        begin = clock(); values, metrics = backend.call(function, selected); end = clock()
        if not isinstance(values, np.ndarray) or not values.flags.owndata or values.flags.writeable:
            raise ValueError('Every stage must return an immutable owning host ndarray')
        expected = len(selected) if function == 'closeness' else n
        if values.shape != (expected,):
            raise ValueError('Incomplete output for ' + function)
        # Some RAPIDS wrappers provide vertex IDs. Require the full original domain.
        if 'vertex_ids' in metrics:
            ids = np.asarray(metrics.pop('vertex_ids'))
            if ids.shape != (n,) or not np.array_equal(np.sort(ids), np.arange(n)):
                raise ValueError('External backend omitted or duplicated original vertices')
            metrics['complete_vertex_ids_checked'] = True
        stages.append(dict(function=function, start_seconds=begin-start, end_seconds=end-start,
                           complete_seconds=end-begin, metrics=metrics))
        return values
    outputs['effective_size'] = call('effective_size')
    outputs['constraint'] = call('constraint')
    t = clock()
    outputs['candidates'] = reused.own(reused.select_candidates(outputs['effective_size'], outputs['constraint']))
    candidate_seconds = clock() - t
    outputs['closeness'] = call('closeness', outputs['candidates'])
    t = clock(); winner = reused.select_winner(outputs['candidates'], outputs['closeness'])
    outputs['winner'] = reused.own([winner]); winner_seconds = clock() - t
    outputs['bfs'] = call('bfs', winner)
    outputs['wcc'] = call('wcc')
    return outputs, {'complete_seconds': clock()-start, 'stages': stages,
                     'candidate_seconds': candidate_seconds, 'winner_seconds': winner_seconds}


def independent_reference(data):
    """Existing exact SciPy formulas, row-blocked to avoid a full A*A materialization.

    Product planning is topology-only and never changes the numerical task.
    Closeness BFS rows are evaluated one at a time; every candidate is exact.
    """
    import numpy as np
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.csgraph import dijkstra, connected_components
    n, degree = data['n'], data['degree'].astype(np.float64)
    a = csr_matrix((np.ones(data['m']), data['indices'], data['offsets']), shape=(n, n))
    inv = np.divide(1., degree, out=np.zeros(n), where=degree > 0)
    p = diags(inv) @ a
    blocks, _ = reused.product_blocks(data['degree'], data['edges'], 64 << 20)
    effective = np.empty(n); constraint = np.empty(n)
    for lo, hi in blocks:
        ar, pr = a[lo:hi], p[lo:hi]
        twice_triangles = np.asarray((ar @ a).multiply(ar).sum(axis=1)).ravel()
        effective[lo:hi] = degree[lo:hi] - np.divide(twice_triangles, degree[lo:hi],
            out=np.zeros(hi-lo), where=degree[lo:hi] > 0)
        local = (pr + pr @ p).multiply(ar)
        constraint[lo:hi] = np.asarray(local.multiply(local).sum(axis=1)).ravel()
    effective[degree == 0] = np.nan; constraint[degree == 0] = np.nan
    candidates = reused.select_candidates(effective, constraint)
    closeness = np.zeros(len(candidates))
    for i, vertex in enumerate(candidates):
        distance = dijkstra(a, directed=False, indices=int(vertex), unweighted=True)
        finite = np.isfinite(distance); reachable = int(finite.sum()) - 1
        total = float(distance[finite].sum())
        closeness[i] = reachable * reachable / ((n-1) * total) if total > 0 else 0.
    winner = reused.select_winner(candidates, closeness)
    bfs = dijkstra(a, directed=False, indices=winner, unweighted=True)
    labels = connected_components(a, directed=False, return_labels=True)[1]
    return dict(effective_size=effective, constraint=constraint, candidates=candidates,
                closeness=closeness, winner=np.array([winner], dtype=np.int64), bfs=bfs,
                wcc=sg.canonical_labels(labels))


def reference_cached(graph, directory, *, allow_compute=True):
    import numpy as np
    identity = dict(input_sha256=graph['input_sha256'], protocol=CONTRACT,
                    code=sg.content_binding(__file__), reused=sg.content_binding(HERE/'e6_reused.py'),
                    numpy=np.__version__, scipy=importlib.metadata.version('scipy'))
    key = sg.digest_json(identity); directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with (directory/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        blob, receipt = directory/(key+'.npz'), directory/(key+'.json')
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record['identity'] != identity or sg.content_binding(blob) != record['output']:
                raise ValueError('E6 independent reference cache changed')
        else:
            if not allow_compute:
                raise ValueError('Prepare the source-bound CPU E6 reference before target/formal GPU execution')
            values = independent_reference(weak_graph(graph))
            with blob.open('xb') as stream:
                np.savez(stream, **values)
            sg.save(receipt, {'identity': identity, 'output': sg.content_binding(blob),
                'method': 'independent SciPy structural algebra; exact one-source-at-a-time BFS and weak components',
                'gpu_executed': False, 'performance_eligible': False})
        with np.load(blob, allow_pickle=False) as z:
            values = {name: z[name].copy() for name in z.files}
        return values, sg.binding(receipt)


def audit_output(values, expected, n):
    import numpy as np
    record = {'vectors': {}, 'pass': True}
    for name in (*FUNCTIONS, 'candidates', 'winner'):
        actual, ref = np.asarray(values[name]), np.asarray(expected[name])
        result = dict(shape=list(actual.shape), owning=bool(actual.flags.owndata),
                      immutable=not actual.flags.writeable, dtype=str(actual.dtype))
        valid = actual.shape == ref.shape and result['owning'] and result['immutable']
        if valid and name in ('effective_size', 'constraint', 'closeness'):
            finite = np.isfinite(actual) & np.isfinite(ref)
            result['nan_mask_equal'] = bool(np.array_equal(np.isnan(actual), np.isnan(ref)))
            result['finite_mask_equal'] = bool(np.array_equal(np.isfinite(actual), np.isfinite(ref)))
            difference = actual[finite] - ref[finite]
            result['rmse_finite'] = float(np.sqrt(np.mean(difference**2))) if difference.size else 0.
            result['max_absolute_error_finite'] = float(np.max(np.abs(difference))) if difference.size else 0.
            valid = (actual.dtype == np.float64 and result['nan_mask_equal'] and result['finite_mask_equal']
                     and np.allclose(actual, ref, rtol=RTOL, atol=ATOL, equal_nan=True))
        elif valid and name == 'wcc':
            valid = np.array_equal(sg.canonical_labels(actual), ref)
        elif valid:
            valid = np.array_equal(actual, ref)
        result['pass'] = bool(valid); record['vectors'][name] = result
        record['pass'] = bool(record['pass'] and valid)
    actual_candidates = reused.select_candidates(values['effective_size'], values['constraint'])
    actual_winner = reused.select_winner(values['candidates'], values['closeness'])
    membership = values['wcc'] == values['wcc'][int(values['winner'][0])]
    record['actual_candidate_selection'] = bool(np.array_equal(actual_candidates, values['candidates']))
    record['actual_winner_selection'] = int(values['winner'][0]) == actual_winner
    record['bfs_wcc_membership_exact'] = bool(np.array_equal(np.isfinite(values['bfs']), membership))
    record['pass'] = bool(record['pass'] and record['actual_candidate_selection'] and
                          record['actual_winner_selection'] and record['bfs_wcc_membership_exact'])
    return record


def audit_execution(trace, args):
    """Check recorded actual device use and all five unchanged numerical stages."""
    if tuple(stage['function'] for stage in trace['stages']) != FUNCTIONS:
        raise ValueError('Workflow omitted or reordered a numerical stage')
    for stage in trace['stages']:
        function, metrics = stage['function'], stage['metrics']
        if args.system == 'igraph':
            expected = []
        elif args.system == 'cugraph_igraph':
            expected = [] if function in ('constraint', 'closeness') else [0]
        elif args.system == 'cugraph_composite':
            expected = [0]
        elif function in GROUPS.get(args.ablate, ()):
            expected = []
        else:
            expected = (list(range(args.gpu_count)) if args.gpu_count > 1 and function in
                ('effective_size', 'constraint', 'closeness') else [0])
        if metrics.get('devices_used') != expected:
            raise ValueError('Actual stage devices differ from fixed coverage/resource contract: '+function)
        if args.system == 'eggpu' and metrics.get('coverage_fallback') != (function in GROUPS.get(args.ablate, ())):
            raise ValueError('The coverage ablation did not execute the specified fallback')
    return True


def options_valid(args):
    if args.system not in SYSTEMS or args.ablate not in ABLATIONS:
        raise ValueError('Unknown E6 system or ablation')
    if args.system != 'eggpu' and args.ablate != 'none':
        raise ValueError('Coverage ablations apply only to EGGPU')
    if args.gpu_count not in (0, 1, 2, 4, 8):
        raise ValueError('GPU count must be 0/1/2/4/8')
    if (args.system == 'igraph') != (args.gpu_count == 0):
        raise ValueError('Only the full CPU igraph composition has zero GPUs')
    if args.system.startswith('cugraph') and args.gpu_count != 1:
        raise ValueError('External compositions are single-GPU only; no fictional MG support')
    if args.gpu_count > 1 and args.ablate != 'none':
        raise ValueError('Coverage ablations are frozen to SG; MG extension is not a new ablation menu')
    if len(args.gpu_uuids) != args.gpu_count or len(set(args.gpu_uuids)) != args.gpu_count:
        raise ValueError('Explicit distinct physical GPU UUIDs must match the requested count')
    if (args.dataset == TINY) != (args.phase == 'tiny'):
        raise ValueError('Tiny input is only a qualification fixture')
    if args.dataset not in (*DATASETS, TINY):
        raise ValueError('E6 main EP/LJ and fixed OR pressure scope cannot be replaced')
    if args.repeat < 0 or (args.phase == 'formal' and args.repeat not in range(10)):
        raise ValueError('Formal repetitions are 0..9')
    if args.timeout != 1800.:
        raise ValueError('Complete workflow timeout is frozen at 1800 seconds')


def check_native_build(args):
    if args.build_receipt is None:
        raise ValueError('Native E6 requires a build receipt with compiled source/DSO bindings')
    build = json.loads(args.build_receipt.read_text())
    if build.get('status') != 'PASS' or not isinstance(build.get('source_sha256'), dict) or not build['source_sha256']:
        raise ValueError('Expected successful Easy-Graph scripts/gpu/build.py receipt')
    for relative, expected in build['source_sha256'].items():
        if sg.content_binding(args.release/relative)['sha256'] != expected:
            raise ValueError('Compiled source differs from build receipt: '+relative)
    outputs = build.get('outputs', {})
    expected = [sha for path, sha in outputs.items() if Path(path).name == args.native_dso.name]
    if expected != [sg.content_binding(args.native_dso)['sha256']]:
        raise ValueError('Native DSO is absent from or differs from build receipt')
    return sg.content_binding(args.build_receipt)


def epoch(args):
    proxy = SimpleNamespace(**vars(args)); proxy.system = 'eggpu' if args.system == 'eggpu' else 'cugraph'
    proxy.gpu_uuid = ','.join(args.gpu_uuids)
    if args.fallback_site:
        sys.path.append(str(args.fallback_site))
    base, _ = sg.source_epoch(proxy)
    base.update(protocol=PROTOCOL, contract=CONTRACT, system=args.system, ablate=args.ablate,
                gpu_count=args.gpu_count, gpu_uuids=args.gpu_uuids,
                files_e6={name: sg.content_binding(HERE/name) for name in
                          ('e6_worker.py', 'e6_reused.py', 'e6_reuse_provenance.json', 'sg_worker.py', 'e6_cpu_reference.py')})
    base['independent_reference_manifest'] = sg.content_binding(args.reference_manifest)
    base['gpu_kernel_sources'] = {str(p.relative_to(args.release)): sg.content_binding(p)
        for p in sorted((args.release/'gpu_easygraph').rglob('*'))
        if p.is_file() and p.suffix in ('.cu', '.cpp', '.h', '.hpp', '.cuh')}
    if args.system == 'eggpu':
        base['build_receipt'] = check_native_build(args)
    # Public cuGraph and cuDF wrappers are part of the actual composite implementation.
    extra = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get('Name', '').lower().replace('_', '-')
        if name == 'cugraph' or name.startswith('cugraph-') or name == 'cudf' or name.startswith('cudf-'):
            extra[name] = {'version': distribution.version, 'files': {
                str(item): sg.content_binding(Path(distribution.locate_file(item)))
                for item in distribution.files or [] if Path(distribution.locate_file(item)).is_file()
                and ('.so' in str(item) or str(item).endswith('.py'))}}
    base['external_wrapper_packages'] = extra
    # sg_worker only binds its own numerical packages; include the actual CPU fallback.
    import igraph
    package = Path(igraph.__file__).resolve().parent
    base['igraph'] = {'version': igraph.__version__, 'files': {
        str(p.relative_to(package)): sg.content_binding(p) for p in sorted(package.rglob('*'))
        if p.is_file() and (p.suffix == '.py' or '.so' in p.name)}}
    return base, sg.digest_json(base)


def validate_proof(path, kind, args, epoch_sha, graph):
    record = json.loads(Path(path).read_text())
    identity = (record.get('phase'), record.get('system'), record.get('ablate'),
                record.get('gpu_count'), record.get('epoch_sha256'))
    if (record.get('status') != 'PASS_QUALIFICATION' or not record.get('quality_pass') or
        identity != (kind, args.system, args.ablate, args.gpu_count, epoch_sha) or
        record.get('protocol') != PROTOCOL or record.get('contract') != CONTRACT):
        raise ValueError('E6 qualification does not bind the current source/system/coverage/resources')
    if kind == 'tiny' and record.get('dataset') != TINY:
        raise ValueError('Expected the frozen disconnected/isolate tiny input')
    if kind == 'target' and record.get('input_sha256') != graph['input_sha256']:
        raise ValueError('Target qualification input changed')
    for label in ('epoch', 'child'):
        item = record[label]
        if sg.binding(item['path']) != item:
            raise ValueError('Qualification child or epoch file changed')
    if sg.digest_json(json.loads(Path(record['epoch']['path']).read_text())) != epoch_sha:
        raise ValueError('Qualification epoch content mismatch')
    child = json.loads(Path(record['child']['path']).read_text())
    if child.get('status') != 'PASS' or not child.get('quality_pass'):
        raise ValueError('Qualification child failed')
    if child.get('input_sha256') != record.get('input_sha256') or set(child.get('calls', {})) != {'first', 'warmup', 'ready'}:
        raise ValueError('Qualification omitted a full workflow call or changed its input')
    for call in child['calls'].values():
        if not call['quality']['pass'] or set(call.get('outputs', {})) != {*FUNCTIONS, 'candidates', 'winner'}:
            raise ValueError('Qualification has a failed full call')
        audit_execution(call['trace'], args)
        for item in call['outputs'].values():
            if sg.binding(item['path']) != item:
                raise ValueError('Qualification output bytes changed')
    return sg.binding(path)


def admission(args, graph):
    if args.gpu_count:
        records = []
        for gpu in args.gpu_uuids:
            proxy = SimpleNamespace(**vars(args)); proxy.gpu_uuid = gpu
            records.append(sg.resource_check(proxy, graph))
        return records
    memory = sg.cgroup_memory()
    required = max(int(args.min_host_free_gib * (1 << 30)), 128*graph['m'] + 128*graph['n'] + (2 << 30))
    if memory['available_bytes'] < required:
        raise MemoryError('CPU composition lacks actual cgroup/host headroom')
    return {'host': memory, 'required_host_bytes': required}


def child(args):
    import numpy as np
    inherited, actual = os.fstat(args.lease_fd), args.lock_file.stat()
    if (inherited.st_dev, inherited.st_ino) != (actual.st_dev, actual.st_ino):
        raise ValueError('Wrong inherited global lease')
    with args.lock_file.open('a') as other:
        try:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError('Parent does not hold the global experiment lease')
    graph = sg.load_input(args.input_root/'canonical'/args.dataset)
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(status='RUNNING', quality_pass=False, calls={}, input_sha256=graph['input_sha256'])
    backend = None
    try:
        record['host_ready_checksum'] = [int(graph['offsets'].sum(dtype=np.int64)),
            int(graph['indices'].sum(dtype=np.int64)), float(graph['weights'].sum(dtype=np.float64))]
        outputs = {}
        for label in ('first', 'warmup', 'ready'):
            sg.save(args.output/'stage.json', {'active': True, 'label': label, 'started_monotonic': time.monotonic()})
            with sg.complete_timeout(args.timeout):
                start = time.perf_counter()
                if label == 'first':
                    prepared = weak_graph(graph)
                    if np.count_nonzero(prepared['degree']) < 32:
                        raise ValueError('Frozen E6 task requires 32 nonisolated candidates, including the tiny fixture')
                    backend = build_backend(args, prepared, args.output)
                    record['weak_projection'] = {'n': prepared['n'], 'stored_arcs': prepared['m'],
                        'source_vertices_preserved': prepared['n'] == graph['n'], 'weights_used': False}
                outputs[label], trace = flow(backend, graph['n'])
                seconds = time.perf_counter() - start
            audit_execution(trace, args)
            sg.save(args.output/'stage.json', {'active': False, 'label': label})
            record['calls'][label] = dict(seconds=seconds, timed=label != 'warmup', trace=trace)
        close_start = time.perf_counter(); backend.close(); backend = None
        record['close_seconds'] = time.perf_counter() - close_start
        expected, reference = e6_cpu_reference.load(graph, args.reference_manifest, CONTRACT, HERE/'e6_reused.py')
        record['independent_reference'] = reference
        for label, values in outputs.items():
            for name, value in values.items():
                for other_label, other in outputs.items():
                    if other_label != label and np.shares_memory(value, other[name]):
                        raise ValueError('Complete outputs alias across workflow invocations')
            call = record['calls'][label]
            call['quality'] = audit_output(values, expected, graph['n'])
            call['outputs'] = {name: sg.persist_output(args.output, value) for name, value in values.items()}
        record['quality_pass'] = all(c['quality']['pass'] for c in record['calls'].values())
        record['status'] = 'PASS' if record['quality_pass'] else 'FAIL_QUALITY'
        record['outputs_survive_close'] = True
    except BaseException as error:
        record.update(status='TIMEOUT' if isinstance(error, TimeoutError) else 'FAILED',
                      quality_pass=False, error=type(error).__name__+': '+str(error))
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as error:
                record.update(status='FAILED_CLOSE', quality_pass=False, close_error=str(error))
        sg.save(args.output/'result.json', record)
    return record


def forward_options(args):
    values = ['--execute-gpu', '--system', args.system, '--ablate', args.ablate, '--dataset', args.dataset,
              '--phase', args.phase, '--repeat', str(args.repeat), '--gpu-count', str(args.gpu_count),
              '--gpu-uuids', ','.join(args.gpu_uuids)]
    for name in ('input_root', 'release', 'native_dso', 'arena_library_dir', 'setup', 'lock_file',
                 'reference_cache', 'reference_manifest', 'budget_mib', 'min_host_free_gib', 'timeout'):
        values += ['--'+name.replace('_', '-'), str(getattr(args, name))]
    if args.fallback_site:
        values += ['--fallback-site', str(args.fallback_site)]
    if args.build_receipt:
        values += ['--build-receipt', str(args.build_receipt)]
    return values


def job(args):
    options_valid(args)
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(schema='eggpu.e6.job/v1', protocol=PROTOCOL, contract=CONTRACT, phase=args.phase,
        system=args.system, ablate=args.ablate, dataset=args.dataset, repeat=args.repeat,
        display_dataset=args.dataset+'-U' if args.dataset in ('EP', 'LJ') else args.dataset,
        dataset_role='pressure' if args.dataset == 'OR' else 'qualification' if args.dataset == TINY else 'main',
        gpu_count=args.gpu_count, status='RUNNING', quality_pass=False, performance_eligible=False)
    sg.save(args.output/'result.json', record)
    try:
        with args.lock_file.open('a') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            graph = sg.load_input(args.input_root/'canonical'/args.dataset)
            reference_values, record['prepared_cpu_reference'] = e6_cpu_reference.load(
                graph, args.reference_manifest, CONTRACT, HERE/'e6_reused.py')
            del reference_values
            if shutil.disk_usage(args.output).free < 128*graph['m'] + 256*graph['n'] + (2 << 30):
                raise RuntimeError('Insufficient free space for full projections and outputs')
            record.update(input=graph['identity'], input_sha256=graph['input_sha256'])
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu_uuids)
            os.environ['EGGPU_ARENA_LIBRARY_DIR'] = str(args.arena_library_dir.resolve())
            initial, fingerprint = epoch(args)
            sg.save(args.output/'epoch.json', initial)
            record.update(epoch=sg.binding(args.output/'epoch.json'), epoch_sha256=fingerprint)
            if args.phase in ('target', 'formal'):
                if args.tiny_proof is None:
                    raise ValueError('Explicit E6 tiny qualification is required')
                record['tiny_proof'] = validate_proof(args.tiny_proof, 'tiny', args, fingerprint, graph)
            if args.phase == 'formal':
                if args.target_proof is None:
                    raise ValueError('Explicit E6 target qualification is required')
                record['target_proof'] = validate_proof(args.target_proof, 'target', args, fingerprint, graph)
            record['admission'] = admission(args, graph)
            destination = args.output/'workflow'
            command = [sys.executable, str(Path(__file__).resolve()), *forward_options(args),
                       '--lease-fd', str(lease.fileno()), '--output', str(destination)]
            with (args.output/'workflow.log').open('wb') as log:
                returncode = sg.run_child(command, log, lease.fileno(), destination, args.timeout)
            result = json.loads((destination/'result.json').read_text()) if (destination/'result.json').exists() else {}
            record['returncode'] = returncode
            if (destination/'result.json').exists():
                record['child'] = sg.binding(destination/'result.json')
            if (destination/'watchdog.json').exists() or result.get('status') == 'TIMEOUT':
                record['status'] = 'TIMEOUT'
            if returncode or result.get('status') != 'PASS' or not result.get('quality_pass'):
                raise RuntimeError('E6 child failed; retained complete log/output is authoritative')
            if epoch(args)[1] != fingerprint:
                raise RuntimeError('E6 source/binary/environment changed during execution')
            record.update(quality_pass=True, performance_eligible=args.phase == 'formal',
                T_first_seconds=result['calls']['first']['seconds'], T_ready_seconds=result['calls']['ready']['seconds'],
                status='PASS_QUALIFICATION' if args.phase in ('tiny', 'target') else 'PASS')
    except BaseException as error:
        record.update(status=record['status'] if record['status'] == 'TIMEOUT' else 'FAILED',
                      error=type(error).__name__+': '+str(error), quality_pass=False)
        raise
    finally:
        sg.save(args.output/'result.json', record)
    return record


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-gpu', action='store_true')
    p.add_argument('--make-tiny', type=Path, help='CPU-only creation of canonical/E6_TINY_U')
    p.add_argument('--prepare-reference', action='store_true', help='CPU-only independent reference preparation')
    p.add_argument('--system', choices=SYSTEMS, default='eggpu')
    p.add_argument('--ablate', choices=ABLATIONS, default='none')
    p.add_argument('--dataset', choices=(*DATASETS, TINY), default='EP')
    p.add_argument('--phase', choices=('tiny', 'diagnostic', 'target', 'formal'), default='diagnostic')
    p.add_argument('--repeat', type=int, default=0)
    p.add_argument('--gpu-count', type=int, default=1)
    p.add_argument('--gpu-uuids', default='')
    for name in ('input-root', 'release', 'native-dso', 'arena-library-dir', 'setup', 'lock-file',
                 'reference-cache', 'reference-manifest', 'output', 'tiny-proof', 'target-proof', 'fallback-site', 'build-receipt'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--budget-mib', type=int, default=22000)
    p.add_argument('--min-host-free-gib', type=float, default=8.)
    p.add_argument('--timeout', type=float, default=1800.)
    p.add_argument('--lease-fd', type=int, help=argparse.SUPPRESS)
    return p


def main():
    p = parser(); args = p.parse_args()
    args.gpu_uuids = args.gpu_uuids.split(',') if args.gpu_uuids else []
    if args.make_tiny:
        if args.execute_gpu or args.prepare_reference:
            p.error('Tiny fixture creation is a separate CPU-only action')
        print(json.dumps({'status': 'PASS_CPU_TINY_INPUT', 'path': str(make_tiny(args.make_tiny)), 'gpu_executed': False}))
        return
    if args.prepare_reference:
        if args.execute_gpu or args.input_root is None or args.reference_cache is None:
            p.error('Reference preparation is CPU-only and requires input-root/reference-cache')
        graph = sg.load_input(args.input_root/'canonical'/args.dataset)
        _, receipt = reference_cached(graph, args.reference_cache)
        print(json.dumps({'status': 'PASS_CPU_REFERENCE', 'receipt': receipt, 'gpu_executed': False}))
        return
    required = ('input_root', 'release', 'native_dso', 'arena_library_dir', 'setup', 'lock_file', 'reference_cache', 'reference_manifest', 'output')
    if not args.execute_gpu or any(getattr(args, name) is None for name in required):
        p.error('Execution is opt-in and requires all source/input/binary/resource/output paths')
    options_valid(args)
    result = child(args) if args.lease_fd is not None else job(args)
    print(json.dumps({key: result[key] for key in ('status', 'quality_pass')}))


if __name__ == '__main__':
    main()
