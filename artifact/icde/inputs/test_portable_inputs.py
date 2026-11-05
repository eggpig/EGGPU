"""CPU tests of the original-format paths; no official payload downloads or GPU."""
from pathlib import Path
from types import SimpleNamespace
import bz2
import gzip
import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import zipfile

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location('portable_' + name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='session')
def tools(tmp_path_factory):
    path = tmp_path_factory.mktemp('portable-cxx')
    for name in ['edge_list_to_csr', 'metis_to_csr']:
        subprocess.run(['g++', '-O2', '-std=c++17', str(HERE / (name + '.cpp')), '-o', str(path / name)], check=True)
    return path


def receipt(prep, root, name, payload, **extra):
    folder = root / 'source' / name; folder.mkdir(parents=True, exist_ok=True)
    path = folder / payload[0]; path.write_bytes(payload[1])
    (folder / 'source_receipt.json').write_text(json.dumps(dict(status='PASS_SOURCE_FILES', files=[prep.binding(path)], **extra)))
    out = root / 'canonical' / name; out.mkdir(parents=True)
    return out


@pytest.mark.parametrize('name,contents,n,raw,expected', [
    ('LJ', b'# Nodes: 5 Edges: 5\n0 1\n0 1\n1 1\n1 2\n2 1\n', 5, 5, [(0, 1), (1, 2), (2, 1)]),
    ('EP', b'# Nodes: 3 Edges: 3\n9 100\n100 9\n101 100\n', 3, 3, [(0, 1), (1, 0), (2, 1)]),
])
def test_snap_paths_keep_official_label_contract(name, contents, n, raw, expected, tools, tmp_path):
    prep = load('prepare_inputs')
    filename = 'soc-LiveJournal1.txt.gz' if name == 'LJ' else 'soc-Epinions1.txt.gz'
    out = receipt(prep, tmp_path, name, (filename, gzip.compress(contents)))
    info, provenance = prep.topology(SimpleNamespace(root=tmp_path, tools=tools),
                                    dict(abbreviation=name, canonical_vertices=n, source_stored_arcs=raw), out)
    offsets = np.fromfile(out / 'offsets.bin', '<i4'); indices = np.fromfile(out / 'indices.bin', '<i4')
    actual = [(u, int(indices[j])) for u in range(n) for j in range(offsets[u], offsets[u+1])]
    assert actual == expected and len(offsets) == n + 1 and info['raw_arcs'] == raw
    if name == 'EP': np.testing.assert_array_equal(np.load(out / 'source_labels.npy'), [9, 100, 101])


@pytest.mark.parametrize('line_end', ['\n', ' \n', ' \r\n'])
def test_nr_exact_header_adapter_and_reciprocal_ids(tools, tmp_path, line_end):
    prep = load('prepare_inputs'); zipped = tmp_path / 'source.zip'
    with zipfile.ZipFile(zipped, 'w') as archive:
        archive.writestr('soc-orkut.mtx', '%MatrixMarket matrix coordinate pattern symmetric' + line_end + '4 4 2\n2 1\n3 2\n')
    out = receipt(prep, tmp_path, 'OR', ('soc-orkut.zip', zipped.read_bytes()))
    info, provenance = prep.topology(SimpleNamespace(root=tmp_path, tools=tools),
                                    dict(abbreviation='OR', canonical_vertices=4, source_stored_arcs=4), out)
    np.testing.assert_array_equal(np.fromfile(out / 'offsets.bin', '<i4'), [0, 1, 3, 4, 4])
    np.testing.assert_array_equal(np.fromfile(out / 'indices.bin', '<i4'), [1, 0, 2, 1])
    assert info['arcs'] == 4 and 'one percent' in provenance['header_adapter']


def test_original_us_source_keeps_zeros_and_no_byte_equivalence_claim(tools, tmp_path):
    prep = load('prepare_inputs')
    out = receipt(prep, tmp_path, 'US', ('road_usa.graph.bz2', bz2.compress(b'4 1\n2\n1\n\n\n')),
                  identity_basis='synthetic fixture')
    info, provenance = prep.topology(SimpleNamespace(root=tmp_path, tools=tools),
                                    dict(abbreviation='US', canonical_vertices=4, source_stored_arcs=2,
                                         source_logical_edges=1), out)
    assert info['vertices'] == 4 and info['arcs'] == 2
    np.testing.assert_array_equal(np.fromfile(out / 'offsets.bin', '<i4'), [0, 1, 2, 2, 2])
    assert provenance['suitesparse_byte_hash_verified'] is False


def test_relocated_pass_receipt_uses_local_hashes(tmp_path):
    prep = load('prepare_inputs')
    original = tmp_path / 'before/canonical/EP'; original.mkdir(parents=True)
    (original / 'offsets.bin').write_bytes(b'payload')
    record = dict(status='PASS_CANONICAL_INPUT', files={'offsets': prep.binding(original / 'offsets.bin')})
    (original / 'receipt.json').write_text(json.dumps(record))
    moved = tmp_path / 'after'; shutil.move(str(tmp_path / 'before'), str(moved))
    assert prep.prepare(SimpleNamespace(root=moved), {'abbreviation': 'EP'}) == record
    (moved / 'canonical/EP/offsets.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        prep.prepare(SimpleNamespace(root=moved), {'abbreviation': 'EP'})


def test_exact_disk_formula_and_unknown_tw_bound():
    preflight = load('preflight')
    source = {'files': [{'bytes': 100}, {'bytes': 3}]}
    known = preflight.budgets(dict(abbreviation='x', canonical_vertices=7,
                                  canonical_stored_arcs=10, source_stored_arcs=11), source)
    assert known['canonical_payload_bytes'] == 4 * 8 + 12 * 10
    assert known['compressed_source_bytes'] == 103 and known['canonical_size_exact']
    unknown = preflight.budgets(dict(abbreviation='x', canonical_vertices=7,
                                    canonical_stored_arcs=None, source_stored_arcs=11), source)
    assert unknown['canonical_payload_bytes'] == 4 * 8 + 12 * 11 and not unknown['canonical_size_exact']


def test_original_payload_hash_rejection(tmp_path):
    fetch = load('fetch'); path = tmp_path / 'payload'; path.write_bytes(b'official')
    spec = {'bytes': 8, 'sha256': hashlib.sha256(b'official').hexdigest()}
    assert fetch.validate_payload(path, spec)['bytes'] == 8
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='SHA256'): fetch.validate_payload(path, spec)


def test_maven_lock_has_fixed_versions_sizes_and_all_urls_probed():
    lock = json.loads((HERE / 'java_dependencies.json').read_text())
    assert len(lock['jars']) == 22 and lock['all_download_urls_probed']
    assert len({i['file'] for i in lock['jars']}) == 22
    for item in lock['jars']:
        assert len(item['sha256']) == 64 and item['bytes'] > 0
        assert item['url'].startswith('https://repo.maven.apache.org/maven2/')
        assert item['url_probe']['status'] == 200 and item['url_probe']['content_length'] == item['bytes']


def test_fetch_idempotence_preserves_bound_source_receipt_bytes(tmp_path, monkeypatch):
    fetch = load('fetch'); folder = tmp_path / 'source/EP'; folder.mkdir(parents=True)
    payload = folder / 'payload'; payload.write_bytes(b'original')
    source = dict(format='fixture', files=[dict(name='payload', bytes=8,
                  sha256=hashlib.sha256(b'original').hexdigest(), url='https://fixture')])
    previous = dict(status='PASS_SOURCE_FILES', source_spec=source, files=[fetch.binding(payload)],
                    started_unix=123, completed_unix=124)
    path = folder / 'source_receipt.json'; path.write_text(json.dumps(previous))
    before = path.read_bytes()
    monkeypatch.setattr(fetch, 'download_file', lambda *a, **k: pytest.fail('cache hit must not download'))
    assert fetch.fetch(tmp_path, 'EP', source) == previous
    assert path.read_bytes() == before
