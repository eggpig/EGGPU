"""Prepare the seven frozen graphs, deterministic weights, and topology-only sources.

No GPU imports. Existing topologies are hash checked and shared without mutation.
Source payloads, graph preparation and benchmark timings remain separate.
"""
from pathlib import Path
import argparse
import gzip
import bz2
import hashlib
import heapq
import json
import os
import shutil
import subprocess
import tarfile
import time


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def binding(path):
    path = Path(path).absolute(); h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(16 << 20), b''): h.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def link_checked(source, destination, expected=None):
    item = binding(source)
    if expected is not None and item['sha256'] != expected:
        raise ValueError('Frozen topology SHA differs: ' + str(source))
    os.link(source, destination)
    return item


def topology(args, spec, out):
    import numpy as np
    from scipy.sparse import csr_matrix
    import zipfile
    name = spec['abbreviation']; n = spec['canonical_vertices']
    directory = args.root / 'source' / name
    receipt_path = directory / 'source_receipt.json'
    downloaded = json.loads(receipt_path.read_text())
    if downloaded['status'] != 'PASS_SOURCE_FILES':
        raise ValueError('Original source is not qualified')
    provenance = {'source_receipt': binding(receipt_path), 'source_files': []}
    payloads = {}
    for item in downloaded['files']:
        path = directory / Path(item['path']).name
        observed = binding(path)
        if any(observed[k] != item[k] for k in ('bytes', 'sha256')):
            raise ValueError('Original source payload changed')
        provenance['source_files'].append(observed); payloads[path.name] = path
    if name == 'EP':
        with gzip.open(payloads['soc-Epinions1.txt.gz'], 'rt') as stream:
            edges = np.loadtxt(stream, dtype=np.int64, comments='#', ndmin=2)
        if edges.shape != (spec['source_stored_arcs'], 2):
            raise ValueError('Official Epinions source row count differs')
        labels = np.unique(edges)
        if len(labels) != n or np.any(labels < 0):
            raise ValueError('Official Epinions complete label domain differs')
        canonical = np.searchsorted(labels, edges)
        graph = csr_matrix((np.ones(len(edges), dtype=np.bool_),
                            (canonical[:, 0], canonical[:, 1])), shape=(n, n))
        graph.sum_duplicates(); graph.setdiag(0); graph.eliminate_zeros(); graph.sort_indices()
        graph.indptr.astype('<i4').tofile(out / 'offsets.bin')
        graph.indices.astype('<i4').tofile(out / 'indices.bin')
        np.save(out / 'source_labels.npy', labels.astype('<i8'), allow_pickle=False)
        info = dict(vertices=n, arcs=graph.nnz, raw_arcs=len(edges),
                    normalization='sorted unique official labels; remove loops and ordered duplicates',
                    source_label_count=len(labels), all_published_vertices_accounted_for=True)
    elif name in ('LJ', 'US', 'OR'):
        mode = 'snap' if name == 'LJ' else 'mm'
        command = ([str(args.tools / 'metis_to_csr'), str(n), str(spec['source_logical_edges']), str(out)]
                   if name == 'US' else
                   [str(args.tools / 'edge_list_to_csr'), mode, str(n), str(spec['source_stored_arcs']), str(out)])
        provenance['converter'] = binding(command[0])
        provenance['converter_source'] = binding(Path(__file__).with_name(Path(command[0]).name + '.cpp'))
        with (out / 'topology.log').open('wb') as log:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
            try:
                if name == 'US':
                    with bz2.open(payloads['road_usa.graph.bz2'], 'rb') as stream:
                        shutil.copyfileobj(stream, proc.stdin, 8 << 20)
                    provenance['origin_identity_basis'] = downloaded['identity_basis']
                    provenance['suitesparse_byte_hash_verified'] = False
                elif name == 'LJ':
                    with gzip.open(payloads['soc-LiveJournal1.txt.gz'], 'rb') as stream:
                        shutil.copyfileobj(stream, proc.stdin, 8 << 20)
                else:
                    with zipfile.ZipFile(payloads['soc-orkut.zip']) as archive:
                        members = [item for item in archive.infolist() if item.filename == 'soc-orkut.mtx']
                        if len(members) != 1: raise ValueError('Unexpected original NR Orkut archive member')
                        with archive.open(members[0]) as stream:
                            first = stream.readline()
                            if first.rstrip(b' \t\r\n') != b'%MatrixMarket matrix coordinate pattern symmetric':
                                raise ValueError('Original NR single-percent Matrix Market header differs')
                            proc.stdin.write(b'%' + first)
                            shutil.copyfileobj(stream, proc.stdin, 8 << 20)
                    provenance['header_adapter'] = 'Exactly prepend one percent to the known original NR header; no edge or label changes'
                proc.stdin.close()
                if proc.wait(): raise RuntimeError('Topology conversion failed; see topology.log')
            except BaseException:
                proc.kill(); proc.wait(); raise
        info = json.loads((out / 'conversion.json').read_text())
    elif name in ('IN', 'UK', 'TW'):
        source_name = {'IN': 'indochina-2004', 'UK': 'uk-2002', 'TW': 'twitter-2010'}[name]
        temporary = out / 'converted'
        classpath = str(args.tools / 'classes') + os.pathsep + str(args.tools / 'jars/*')
        with (out / 'topology.log').open('wb') as log:
            subprocess.run([args.java, '-Xmx2g', '-cp', classpath, 'LawToCsr',
                            str(directory / source_name), str(temporary)], check=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT)
        for role in ('offsets.bin', 'indices.bin', 'conversion.json'):
            (temporary / role).replace(out / role)
        temporary.rmdir(); info = json.loads((out / 'conversion.json').read_text())
        provenance['toolchain'] = binding(args.tools / 'toolchain.json')
        provenance['exporter_source'] = binding(Path(__file__).with_name('LawToCsr.java'))
        provenance['java_executable'] = binding(shutil.which(args.java) or args.java)
        for role, expected in spec.get('archived_topology_sha256', {}).items():
            if binding(out / (role + '.bin'))['sha256'] != expected:
                raise ValueError('Fresh LAW export differs from frozen canonical topology: ' + role)
    else:
        raise ValueError('Unknown frozen graph')
    if info['vertices'] != n or info['raw_arcs'] != spec['source_stored_arcs']:
        raise ValueError('Published source dimensions differ')
    save(out / 'conversion.json', info)
    return info, provenance


def validate_and_select(args, spec, out, info):
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n, m = info['vertices'], info['arcs']
    if n != spec['canonical_vertices'] or m >= 2**31: raise ValueError('Canonical dimensions differ')
    if spec.get('canonical_stored_arcs') is not None and m != spec['canonical_stored_arcs']:
        raise ValueError('Previously qualified canonical arc count differs')
    if (out / 'offsets.bin').stat().st_size != 4*(n+1) or (out / 'indices.bin').stat().st_size != 4*m:
        raise ValueError('CSR payload size differs')
    offsets = np.memmap(out / 'offsets.bin', dtype='<i4', mode='r', shape=(n+1,))
    indices = np.memmap(out / 'indices.bin', dtype='<i4', mode='r', shape=(m,))
    if offsets[0] or offsets[-1] != m or np.any(offsets[1:] < offsets[:-1]): raise ValueError('Invalid offsets')
    wide = np.asarray(offsets, dtype=np.int64); previous = None
    for start in range(0, m, 1 << 20):
        stop = min(m, start + (1 << 20)); v = indices[start:stop]
        u = np.searchsorted(wide, np.arange(start, stop, dtype=np.int64), side='right') - 1
        if np.any(v < 0) or np.any(v >= n) or np.any(u == v): raise ValueError('Bad canonical endpoint')
        if np.any((u[1:] == u[:-1]) & (v[1:] <= v[:-1])): raise ValueError('Rows not simple and sorted')
        if previous is not None and (int(u[0]), int(v[0])) <= previous: raise ValueError('Bad row boundary')
        previous = (int(u[-1]), int(v[-1]))
    graph = csr_matrix((np.ones(m, dtype=np.uint8), indices, offsets), shape=(n, n), copy=False)
    if not spec['directed'] and (graph != graph.T).nnz: raise ValueError('Undirected graph is not reciprocal')
    count, labels = connected_components(graph, directed=spec['directed'],
                                         connection='strong' if spec['directed'] else 'weak', return_labels=True)
    sizes = np.bincount(labels); maximum = int(sizes.max()); largest = np.flatnonzero(sizes == maximum)
    # Among tied maximum components, the one containing the smallest canonical ID wins.
    target = int(labels[np.flatnonzero(np.isin(labels, largest))[0]])
    selected = np.flatnonzero(labels == target); fallback = False; supplement = np.empty(0, dtype=np.int64)
    if len(selected) < 3:
        _, weak = connected_components(graph, directed=False, return_labels=True)
        sizes2 = np.bincount(weak); tied = np.flatnonzero(sizes2 == sizes2.max())
        chosen = int(weak[np.flatnonzero(np.isin(weak, tied))[0]])
        supplement = np.flatnonzero((weak == chosen) & (np.diff(offsets) > 0))
        supplement = np.setdiff1d(supplement, selected, assume_unique=True); fallback = True
    prefix = ('eggpu-source-v1|20261104|' + spec['dataset_id'] + '|').encode()
    rank = lambda v: (hashlib.sha256(prefix + str(int(v)).encode()).digest(), int(v))
    sources = [v for _, v in heapq.nsmallest(3, map(rank, selected))]
    if len(sources) < 3:
        sources += [v for _, v in heapq.nsmallest(3-len(sources), map(rank, supplement))]
    diagnostic = min(map(rank, range(n)))[1]
    source_record = {'primary_sources': sources, 'diagnostic_source': diagnostic,
                     'largest_component_vertices': maximum, 'component_count': int(count),
                     'fallback_used': fallback, 'selection_uses_timing': False,
                     'rule': 'DATASET_REGISTRY_20261104.json/source_selection'}
    save(out / 'sources.json', source_record)
    return offsets, indices, source_record


def prepare(args, spec):
    import numpy as np
    out = args.root / 'canonical' / spec['abbreviation']
    if (out / 'receipt.json').exists():
        record = json.loads((out / 'receipt.json').read_text())
        if record['status'] == 'PASS_CANONICAL_INPUT':
            for item in record['files'].values():
                observed=binding(out/Path(item['path']).name)
                if any(observed[k] != item[k] for k in ('bytes','sha256')): raise ValueError('Existing qualified input changed')
            return record
        raise ValueError('Previous failed preparation requires inspection; do not overwrite')
    out.mkdir(parents=True, exist_ok=False)
    record = {'status': 'RUNNING', 'dataset': spec['abbreviation'], 'dataset_id': spec['dataset_id'],
              'gpu_executed': False, 'source_page': spec['source_url']}
    started = time.monotonic(); save(out / 'receipt.json', record)
    try:
        info, provenance = topology(args, spec, out)
        offsets, indices, sources = validate_and_select(args, spec, out, info)
        n, m = info['vertices'], info['arcs']
        if shutil.disk_usage(out).free < 8*m + (6 << 30): raise RuntimeError('Insufficient weight payload disk plus reserve')
        partial = out / 'weights.bin.partial'
        env = dict(os.environ, OMP_NUM_THREADS='16', CUDA_VISIBLE_DEVICES='')
        subprocess.run([str(args.tools / 'weights'), str(out / 'offsets.bin'), str(out / 'indices.bin'),
                        str(partial), str(n), str(m), spec['dataset_id'], str(int(spec['directed']))], check=True, env=env)
        weights = np.memmap(partial, dtype='<f8', mode='r', shape=(m,))
        for start in range(0, m, 1 << 20):
            w = weights[start:start+(1<<20)]
            if np.any(w < 1) or np.any(w > 16) or np.any(w != np.floor(w)): raise ValueError('Bad weight domain')
        sample_positions = np.unique(np.linspace(0, m-1, min(4096,m), dtype=np.int64))
        sample_sources = np.searchsorted(offsets, sample_positions, side='right') - 1
        for j, source in zip(sample_positions, sample_sources):
            u = int(source); v = int(indices[j])
            if not spec['directed']: u, v = min(u,v), max(u,v)
            key = f"eggpu-w16-v1|20261104|{spec['dataset_id']}|{u}|{v}".encode()
            expected = 1 + int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % 16
            if weights[j] != expected: raise ValueError('Independent Python SHA weight check failed')
        del weights; partial.replace(out / 'weights.bin')
        meta = {'format':'eggpu-csr-v1','generation':1,'name':spec['dataset_id'],'abbreviation':spec['abbreviation'],
                'num_nodes':n,'num_edges':m if spec['directed'] else m//2,'num_entries':m,
                'directed':spec['directed'],'weighted':True,'offset_dtype':'int32','index_dtype':'int32',
                'weight_dtype':'float64','endianness':'little','node_labels':'zero_based_contiguous',
                'sorted_indices':True,'allow_self_loops':False,'duplicate_policy':'simple',
                'offsets_path':'offsets.bin','indices_path':'indices.bin','weights_path':'weights.bin',
                'weight_key':'weight','weight_rule':'w16-v1','source':spec['source_url']}
        save(out / 'canonical.json',meta)
        mapping = {'canonical_domain': [0,n-1], 'preserves_all_source_vertices': True,
                   'rule': 'source_labels.npy' if spec['abbreviation']=='EP' else
                           'canonical ID = official one-based ID - 1' if spec['abbreviation']=='US' else
                           'identity in the official contiguous ordering' if spec['abbreviation'] in ('LJ','IN','UK','TW') else
                           'canonical ID = official one-based Matrix Market ID - 1'}
        save(out / 'id_mapping.json', mapping)
        files = {role:binding(out/name) for role,name in [('offsets','offsets.bin'),('indices','indices.bin'),
                 ('weights','weights.bin'),('canonical','canonical.json'),('sources','sources.json'),('id_mapping','id_mapping.json')]}
        if (out/'source_labels.npy').exists(): files['source_labels']=binding(out/'source_labels.npy')
        expected=json.loads(Path(__file__).with_name('expected_canonical.json').read_text())['datasets'].get(spec['abbreviation'])
        if expected:
            for role, wanted in expected['files'].items():
                if any(files[role][k] != wanted[k] for k in ('bytes','sha256')): raise ValueError('Fresh payload differs from qualified reference: '+role)
            record['exact_reference_payload_match']=list(expected['files'])
        record.update(status='PASS_CANONICAL_INPUT',vertices=n,stored_arcs=m,normalization=info,
                      source_bindings=provenance,source_selection=sources,files=files,
                      generator=binding(args.tools/'weights'),generator_source=binding(Path(__file__).with_name('weights.cpp')),
                      preparation_script=binding(__file__),weights_checked='full finite integer domain + 4096 deterministic cross-language SHA checks',
                      seconds=time.monotonic()-started)
    except BaseException as e:
        record.update(status='FAILED',error=type(e).__name__+': '+str(e));raise
    finally: save(out/'receipt.json',record)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--tools',type=Path,required=True);p.add_argument('--java',default='java')
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--datasets',default='EP,OR,IN,UK,LJ,US,TW')
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=''
    toolchain=json.loads((a.tools/'toolchain.json').read_text())
    if toolchain.get('status') != 'PASS_CPU_TOOLCHAIN': raise ValueError('CPU toolchain is not qualified')
    for relative, item in toolchain['files'].items():
        observed=binding(a.tools/relative)
        if any(observed[k] != item[k] for k in ('bytes','sha256')): raise ValueError('CPU toolchain file changed: '+relative)
    registry=json.loads(a.registry.read_text());specs={d['abbreviation']:d for d in registry['datasets']}
    for name in a.datasets.split(','):
        record=prepare(a,specs[name]);print(json.dumps({'dataset':name,'status':record['status'],'arcs':record['stored_arcs']}),flush=True)

if __name__=='__main__':main()
