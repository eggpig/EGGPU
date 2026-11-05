"""Native partition preparation from common CSR, with no GPU imports.

Adapted from the campaign's historical large_panel_v5/csr_contract.py
native_partitions/weak_projection procedures. Global weak CSR may use int64
offsets; every produced native partition must independently fit signed int32.
Round-trip audit is separate from the timed preparation.
"""
from pathlib import Path
import errno
import importlib.util
import json
import os
import shutil

from sg_worker import binding, save

CHUNK = 1 << 20
MAX_LOCAL_ENTRIES = 2**31 - 1
MAX_LOCAL_VERTICES = 2**31 - 1
ORIGIN = 'experiments/rtx4090d_20261031/performance_revision_20261102/large_panel_v5/csr_contract.py'
ORIGIN_SHA256 = '5b04cb81347293bbe64fdc7d6d1f649ee366db92c211b9da5a497b67902ea795'


class IndexUnsupported(ValueError):
    """The requested partition geometry cannot be represented without truncation."""


def load_signer(release):
    path = Path(release) / 'easygraph/gpu/_manifest_signatures.py'
    spec = importlib.util.spec_from_file_location('_campaign_native_manifest_signer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sign


def original_graph(graph, name):
    directory = Path(graph['directory'])
    return dict(graph, name=name, paths={role: directory / (role + '.bin')
                                       for role in ('offsets', 'indices', 'weights')})


def weak_projection(graph, output):
    import numpy as np
    from scipy.sparse import csr_matrix
    outgoing = csr_matrix((np.ones(graph['m'], dtype=np.uint8), graph['indices'], graph['offsets']),
                          shape=(graph['n'], graph['n']), copy=False)
    weak = outgoing.maximum(outgoing.T).tocsr()
    weak.sort_indices()
    # A global weak graph can exceed int32 arc count while each partition fits.
    # Preserve its full int64 offset domain until local rebasing, never truncate.
    row_dtype = '<i8' if weak.nnz > MAX_LOCAL_ENTRIES else '<i4'
    offsets = weak.indptr.astype(row_dtype, copy=False)
    indices = weak.indices.astype('<i4', copy=False)
    folder = Path(output); folder.mkdir(exist_ok=False)
    offsets.tofile(folder / 'offsets.bin'); indices.tofile(folder / 'indices.bin')
    result = dict(name=graph['name'] + '-weak', n=graph['n'], m=int(weak.nnz), offsets=offsets,
                  indices=indices, weights=None, directed=False,
                  paths={'offsets': folder / 'offsets.bin', 'indices': folder / 'indices.bin'})
    info = dict(operation='simple A OR transpose(A); preserve every vertex including isolates',
                original_stored_arcs=graph['m'], weak_stored_arcs=int(weak.nnz),
                global_offset_dtype=str(offsets.dtype), all_arcs_retained=True,
                files={role: binding(path) for role, path in result['paths'].items()})
    save(folder / 'projection.json', info)
    return result, info


def partition_ranges(offsets, n, m, count):
    import numpy as np
    if count not in (1, 2, 4, 8) or count > n:
        raise ValueError('Native P must be 1/2/4/8 and not exceed vertex count')
    cuts = [0]
    for rank in range(1, count):
        cuts.append(max(cuts[-1] + 1, min(n - (count - rank),
                        int(np.searchsorted(offsets, m * rank // count)))))
    cuts.append(n)
    for first, last in zip(cuts, cuts[1:]):
        if int(offsets[last]) - int(offsets[first]) > MAX_LOCAL_ENTRIES:
            raise IndexUnsupported('INDEX_UNSUPPORTED: one native partition exceeds int32 adjacency capacity')
        if last - first > MAX_LOCAL_VERTICES:
            raise IndexUnsupported('INDEX_UNSUPPORTED: owned vertex count exceeds int32 capacity')
    return cuts


def native_partitions(graph, output, count, sign, weighted=False):
    import numpy as np
    n, m = graph['n'], graph['m']
    offsets, indices = graph['offsets'], graph['indices']
    if n > MAX_LOCAL_VERTICES:
        raise IndexUnsupported('INDEX_UNSUPPORTED: canonical vertex domain exceeds this adapter')
    cuts = partition_ranges(offsets, n, m, count)
    folder = Path(output); folder.mkdir(exist_ok=False)
    parts, cross_filesystem_copy_bytes = [], 0
    for rank, (first, last) in enumerate(zip(cuts, cuts[1:])):
        start, stop = int(offsets[first]), int(offsets[last])
        seen = np.zeros(n, dtype=bool); cut_edges = 0
        for pos in range(start, stop, CHUNK):
            targets = indices[pos:min(stop, pos + CHUNK)]
            seen[targets] = True
            cut_edges += int(np.count_nonzero((targets < first) | (targets >= last)))
        seen[first:last] = False
        mapping = np.r_[np.arange(first, last, dtype='<u8'), np.flatnonzero(seen).astype('<u8')]
        if len(mapping) > MAX_LOCAL_VERTICES:
            raise IndexUnsupported('INDEX_UNSUPPORTED: owned plus halo vertices exceed int32 capacity')
        inverse = np.full(n, -1, dtype='<i4')
        inverse[mapping] = np.arange(len(mapping), dtype='<i4')
        files = {}
        def write(role, array, dtype, original=None):
            nonlocal cross_filesystem_copy_bytes
            path = folder / f'p{rank}.{role}.bin'
            if original is not None:
                try:
                    os.link(original, path)
                except OSError as error:
                    if error.errno != errno.EXDEV:
                        raise
                    shutil.copyfile(original, path)
                    cross_filesystem_copy_bytes += path.stat().st_size
            else:
                with path.open('xb') as stream:
                    array.tofile(stream)
            item = binding(path)
            files[role] = {'path': path.name, 'dtype': dtype, 'count': len(array),
                           'byte_size': array.nbytes, 'sha256': item['sha256']}
        local_offsets = (offsets[first:last+1].astype(np.int64) - start).astype('<i4')
        source_offsets = graph['paths']['offsets'] if count == 1 and offsets.dtype == np.dtype('<i4') else None
        write('offsets', local_offsets, 'int32', source_offsets)
        write('local_to_global', mapping, 'uint64')
        write('owners', np.searchsorted(cuts[1:], mapping, side='right').astype('<i4'), 'int32')
        ipath = folder / f'p{rank}.indices.bin'
        if count == 1:
            try:
                os.link(graph['paths']['indices'], ipath)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(graph['paths']['indices'], ipath)
                cross_filesystem_copy_bytes += ipath.stat().st_size
        else:
            with ipath.open('xb') as stream:
                for pos in range(start, stop, CHUNK):
                    inverse[indices[pos:min(stop, pos + CHUNK)]].astype('<i4', copy=False).tofile(stream)
        files['indices'] = {'path': ipath.name, 'dtype': 'int32', 'count': stop-start,
                            'byte_size': 4*(stop-start), 'sha256': binding(ipath)['sha256']}
        if weighted:
            write('weights', graph['weights'][start:stop], 'float64', graph['paths']['weights'] if count == 1 else None)
        parts.append(dict(partition_id=rank, device_id=rank, owned_vertex_count=last-first,
            halo_vertex_count=len(mapping)-(last-first), adjacency_count=stop-start, cut_edge_count=cut_edges,
            ownership=dict(encoding='range', vertex_begin=first, vertex_end=last), files=files))
    doc = dict(format='egmgpu-partitioned-csr', version=1,
        graph=dict(id=graph['name'] + ('-weighted' if weighted else '-unweighted'), generation=1,
                   directed=graph['directed'], weighted=weighted, vertex_count=n,
                   edge_count=m if graph['directed'] else m//2, adjacency_count=m),
        view=dict(orientation='outgoing' if graph['directed'] else 'undirected', weight_key='weight' if weighted else None),
        indexing=dict(offset_dtype='int32', index_dtype='int32', global_vertex_dtype='uint64', endianness='little'),
        partitioning=dict(ownership_encoding='range'), partitions=parts)
    top, weight, signature = sign(doc)
    doc['graph']['topology_sha256'] = top
    if weighted:
        doc['graph']['weight_sha256'] = weight
    doc['view']['signature'] = signature
    path = folder / 'manifest.json'; save(path, doc)
    return dict(binding(path), count=count, cross_filesystem_copy_bytes=cross_filesystem_copy_bytes,
                local_int32_checked=True, global_adjacency_count=m, input_global_offset_dtype=str(offsets.dtype))


def audit_partitions(graph, manifest):
    """Full topology/weight round trip outside measured intervals."""
    import numpy as np
    path = Path(manifest); doc = json.loads(path.read_text())
    n, total = graph['n'], 0
    ownership = np.empty(n, np.int32)
    for part in doc['partitions']:
        a, b = part['ownership']['vertex_begin'], part['ownership']['vertex_end']
        ownership[a:b] = part['partition_id']
    for part in doc['partitions']:
        a, b = part['ownership']['vertex_begin'], part['ownership']['vertex_end']
        start, stop = int(graph['offsets'][a]), int(graph['offsets'][b])
        arrays = {}
        for role, item in part['files'].items():
            source = path.parent / item['path']
            if binding(source)['sha256'] != item['sha256']:
                raise ValueError('Prepared partition bytes changed')
            dtype = {'int32': '<i4', 'uint64': '<u8', 'float64': '<f8'}[item['dtype']]
            arrays[role] = np.memmap(source, mode='r', dtype=dtype, shape=(item['count'],)) if item['count'] else np.empty(0, dtype)
        try:
            np.testing.assert_array_equal(arrays['offsets'], graph['offsets'][a:b+1].astype(np.int64) - start)
            np.testing.assert_array_equal(arrays['local_to_global'][:b-a], np.arange(a, b))
            np.testing.assert_array_equal(arrays['owners'], ownership[arrays['local_to_global']])
            for position in range(0, stop-start, CHUNK):
                end = min(stop-start, position + CHUNK)
                np.testing.assert_array_equal(arrays['local_to_global'][arrays['indices'][position:end]],
                                              graph['indices'][start+position:start+end])
                if 'weights' in arrays:
                    np.testing.assert_array_equal(arrays['weights'][position:end], graph['weights'][start+position:start+end])
            total += stop-start
        finally:
            for array in arrays.values():
                if isinstance(array, np.memmap):
                    array._mmap.close()
    if total != graph['m']:
        raise ValueError('Partition round trip dropped arcs')
    return {'status': 'PASS_FULL_PARTITION_ROUNDTRIP', 'stored_arcs_checked': total,
            'vertices_checked': n, 'weighted': doc['graph']['weighted'], 'timed': False}


def remove_generated_payloads(manifest, projection=None):
    """Remove only this attempt's generated links/files after all outputs pass."""
    path = Path(manifest).resolve(); document = json.loads(path.read_text())
    removed = 0
    for part in document['partitions']:
        for item in part['files'].values():
            payload = path.parent / item['path']
            if payload.parent.resolve() != path.parent or payload.is_symlink():
                raise ValueError('Unexpected generated payload path')
            removed += payload.stat().st_size
            payload.unlink()
    if projection:
        for item in projection['files'].values():
            payload = Path(item['path'])
            if payload.parent.parent.resolve() != path.parent.parent:
                raise ValueError('Weak payload is outside this source attempt')
            removed += payload.stat().st_size
            payload.unlink()
    return dict(generated_payload_bytes_unlinked=removed, retained='manifest, projection receipt, all content hashes, complete outputs',
                canonical_inputs_modified=False, timing='after all timed calls and full independent audit')
