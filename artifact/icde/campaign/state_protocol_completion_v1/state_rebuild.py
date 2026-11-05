"""Full host-partition + device-state rebuild control through existing APIs.

This is intentionally the existing file-manifest admission path, not a proposed
fast in-memory importer. Partition construction, new payload writes/reads and
validation are part of the complete application interval. Canonical download,
normalization and common-input verification remain outside it for every arm.
"""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import time


def make_manifest(row, graph_name, version, count, sign, root, *, projection):
    import numpy as np
    from scipy.sparse import csr_matrix
    if projection not in ('weighted', 'bfs', 'wcc'):
        raise ValueError('Unknown rebuild projection')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    n, m = row['n'], row['m']
    off = np.memmap(row['canonical']['offsets']['path'], dtype='<i4', mode='r', shape=(n + 1,))
    ind = np.memmap(row['canonical']['indices']['path'], dtype='<i4', mode='r', shape=(m,))
    meta = json.loads(Path(row['canonical']['metadata']['path']).read_text())
    directed = meta['directed']
    weighted = projection == 'weighted'
    weights = None
    if weighted:
        # Materialize the complete current version from the canonical starting
        # payload, not from an earlier maintained native host snapshot.
        weights = np.fromfile(row['canonical']['weights']['path'], dtype='<f8')
        positions = np.load(row['positions']['path'], allow_pickle=False, mmap_mode='r')
        weights[positions] = 1 + (weights[positions] - 1 + version) % 16
    if projection == 'wcc' and directed:
        matrix = csr_matrix((np.ones(m, dtype=np.uint8), ind, off), shape=(n, n))
        weak = matrix.maximum(matrix.T).tocsr()
        weak.sort_indices()
        if weak.nnz >= 2**31:
            raise OverflowError('Rebuilt weak projection exceeds int32; no edges omitted')
        off, ind, m = weak.indptr.astype('<i4'), weak.indices.astype('<i4'), int(weak.nnz)
        directed = False
    cuts = np.linspace(0, n, count + 1, dtype=np.int64)
    if n < count or not 0 < m < 2**31:
        raise ValueError('Unsupported rebuild partition domain')
    parts = []
    for rank, (first, last) in enumerate(zip(cuts[:-1], cuts[1:])):
        a, b = int(off[first]), int(off[last])
        targets = ind[a:b]
        mask = (targets < first) | (targets >= last)
        halo = np.unique(targets[mask])
        mapping = np.r_[np.arange(first, last, dtype='<u8'), halo.astype('<u8')]
        inverse = np.full(n, -1, dtype='<i4')
        inverse[mapping] = np.arange(len(mapping), dtype='<i4')
        arrays = dict(offsets=(off[first:last + 1] - a).astype('<i4'),
            indices=inverse[targets], local_to_global=mapping,
            owners=np.searchsorted(cuts[1:], mapping, side='right').astype('<i4'))
        if weighted:
            arrays['weights'] = weights[a:b]
        files = {}
        for name, array in arrays.items():
            path = root / f'p{rank}.{name}.bin'
            array.tofile(path)
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 << 20), b''):
                    h.update(block)
            files[name] = dict(path=path.name,
                dtype='uint64' if name == 'local_to_global' else 'float64' if name == 'weights' else 'int32',
                count=len(array), byte_size=int(array.nbytes), sha256=h.hexdigest())
        parts.append(dict(partition_id=rank, device_id=rank,
            owned_vertex_count=int(last - first), halo_vertex_count=len(halo),
            adjacency_count=b - a, cut_edge_count=int(mask.sum()),
            ownership=dict(encoding='range', vertex_begin=int(first), vertex_end=int(last)), files=files))
    identity = graph_name if weighted else graph_name + '__' + projection
    doc = dict(format='egmgpu-partitioned-csr', version=1,
        graph=dict(id=identity, generation=version + 1, vertex_count=n,
            adjacency_count=m, edge_count=m if directed else m // 2,
            directed=directed, weighted=weighted),
        indexing=dict(endianness='little', global_vertex_dtype='uint64', index_dtype='int32', offset_dtype='int32'),
        partitioning=dict(ownership_encoding='range'),
        view=dict(orientation='outgoing' if directed else 'undirected',
            weight_key='controlled_positive_integer' if weighted else None), partitions=parts)
    topology, weight, view = sign(doc)
    doc['graph']['topology_sha256'] = topology
    if weighted:
        doc['graph']['weight_sha256'] = weight
    doc['view']['signature'] = view
    path = root / 'manifest.json'
    path.write_text(json.dumps(doc, indent=2) + '\n')
    return identity, path, dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        identity=identity, generation=version + 1, topology_sha256=topology,
        weight_sha256=weight, view_signature=view,
        payload_bytes=sum(s['byte_size'] for p in parts for s in p['files'].values()),
        files={f"p{p['partition_id']}.{name}": dict(spec) for p in parts for name, spec in p['files'].items()})


def rebuild(session, row, event, partitions, source, directory):
    from state_controls import retire_for_rebuild
    from state_inputs import signer
    from state_protocol import sha
    started = time.perf_counter()
    release = retire_for_rebuild(session)
    construction = time.perf_counter()
    # One immutable per-commit staging directory. It is removed only after all
    # sessions/leases close; source-bound generation receipts are preserved.
    root = Path(directory) / event['snapshot']
    if root.exists():
        raise FileExistsError('Rebuild attempt already exists')
    sign = signer(Path(source))
    selected, receipts = {}, {}
    for projection in ('weighted', 'bfs', 'wcc'):
        identity, path, receipt = make_manifest(row, event['snapshot'], event['version'],
            partitions, sign, root / projection, projection=projection)
        begin_import = time.perf_counter()
        graph = session.register_snapshot(identity, path)
        receipt['host_import_seconds'] = time.perf_counter() - begin_import
        selected[projection] = identity
        receipts[projection] = receipt
    return selected, dict(kind='canonical_host_partition_and_device_rebuild',
        rebuild_generation=event['version'], release=release,
        host_reconstruction_seconds=time.perf_counter() - construction,
        complete_rebuild_prepare_seconds=time.perf_counter() - started,
        projections=receipts,
        host_payload_written_bytes=sum(x['payload_bytes'] for x in receipts.values()),
        source_signer_sha256=sha(Path(source) / 'easygraph/gpu/_manifest_signatures.py'),
        device_construction='Lazy full native query from newly imported host partitions; shared catalog budget is zero',
        old_reader_protection='Leased views stay resident in the same exclusive runtime; no second Session created',
        storage_path='Existing file-manifest API; partition writes/reads/verification charged inside the interval')


def cleanup(directory):
    directory = Path(directory)
    if directory.is_dir():
        shutil.rmtree(directory)
