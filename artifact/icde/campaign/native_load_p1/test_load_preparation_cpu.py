"""Full-byte parity for the P1 identity preparation, without CUDA imports."""
from pathlib import Path
import importlib.util
import json
import os
import tempfile
import unittest

import numpy as np

import native_input as candidate


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'native_parent' / 'native_input.py'
RELEASE = HERE.parents[3] / 'Easy-Graph'
spec = importlib.util.spec_from_file_location('load_parent_native_input', PARENT)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)


def fixture(folder, directed, offset_dtype, mmap, empty=False):
    folder.mkdir()
    edges = [] if empty else [(0, 1), (0, 3), (1, 2), (3, 1), (4, 5), (5, 4)]
    if not directed:
        edges = sorted(set(edges + [(v, u) for u, v in edges]))
    else:
        edges = sorted(set(edges))
    n = 8  # vertices 6 and 7 remain isolated in every fixture
    offsets = np.zeros(n + 1, dtype=offset_dtype)
    for u, _ in edges:
        offsets[u + 1] += 1
    np.cumsum(offsets, out=offsets)
    indices = np.array([v for _, v in edges], dtype='<i4')
    weights = np.array([1 + (min(u, v) + max(u, v)) % 16 for u, v in edges], dtype='<f8')
    arrays = dict(offsets=offsets, indices=indices, weights=weights)
    paths = {}
    for role, array in arrays.items():
        path = folder / (role + '.bin')
        array.tofile(path)
        paths[role] = path
        if mmap and len(array):
            arrays[role] = np.memmap(path, dtype=array.dtype, mode='r', shape=array.shape)
    return dict(name='parity', n=n, m=len(edges), directed=directed, paths=paths, **arrays)


class LoadPreparation(unittest.TestCase):
    def setUp(self):
        scratch = os.environ['EGGPU_TEST_SCRATCH']
        self.temp = tempfile.TemporaryDirectory(prefix='p1_load_', dir=scratch)
        self.base = Path(self.temp.name)
        self.sign = candidate.load_signer(RELEASE)

    def tearDown(self):
        self.temp.cleanup()

    def assert_same(self, graph, count, weighted, label):
        old, new = self.base / (label + '_old'), self.base / (label + '_new')
        parent.native_partitions(graph, old, count, self.sign, weighted=weighted)
        result = candidate.native_partitions(graph, new, count, self.sign, weighted=weighted)
        self.assertEqual(sorted(p.name for p in old.iterdir()), sorted(p.name for p in new.iterdir()))
        for path in old.iterdir():
            self.assertEqual(path.read_bytes(), (new / path.name).read_bytes(), path.name)
        audit = candidate.audit_partitions(graph, result['path'])
        self.assertEqual(audit['status'], 'PASS_FULL_PARTITION_ROUNDTRIP')
        self.assertEqual(audit['stored_arcs_checked'], graph['m'])
        if count == 1:
            part = json.loads((new / 'manifest.json').read_text())['partitions'][0]
            self.assertEqual(part['halo_vertex_count'], 0)
            self.assertEqual(part['cut_edge_count'], 0)
            self.assertEqual((new / 'p0.indices.bin').stat().st_ino, graph['paths']['indices'].stat().st_ino)
        return result

    def test_full_byte_parity_all_partition_counts_and_domains(self):
        cases = 0
        for directed in (False, True):
            for offset_dtype in ('<i4', '<i8'):
                for mmap in (False, True):
                    graph = fixture(self.base / ('input_' + str(cases)), directed, offset_dtype, mmap)
                    for count in (1, 2, 4, 8):
                        for weighted in (False, True):
                            self.assert_same(graph, count, weighted, 'case_' + str(cases))
                            cases += 1
        self.assertEqual(cases, 64)

    def test_empty_arcs_and_isolates(self):
        for i, dtype in enumerate(('<i4', '<i8')):
            graph = fixture(self.base / ('empty_' + str(i)), False, dtype, False, empty=True)
            for count in (1, 2, 4, 8):
                self.assert_same(graph, count, True, f'empty_{i}_{count}')

    def test_p1_does_not_read_adjacency_to_discover_nonexistent_halos(self):
        graph = fixture(self.base / 'no_scan', True, '<i4', True)
        class NoIndexScan:
            def __getitem__(self, key):
                raise AssertionError('P1 preparation scanned adjacency for halos')
        prepared = dict(graph, indices=NoIndexScan())
        result = candidate.native_partitions(prepared, self.base / 'p1_noscan', 1, self.sign, weighted=True)
        self.assertEqual(candidate.audit_partitions(graph, result['path'])['status'], 'PASS_FULL_PARTITION_ROUNDTRIP')

    def test_complete_audit_still_rejects_bad_indices_and_tampered_payloads(self):
        for i, target in enumerate((-1, 8)):
            graph = fixture(self.base / ('bad_' + str(i)), True, '<i4', False)
            graph['indices'][0] = target
            graph['indices'].tofile(graph['paths']['indices'])
            result = candidate.native_partitions(graph, self.base / ('bad_out_' + str(i)), 1, self.sign)
            with self.assertRaises((IndexError, AssertionError)):
                candidate.audit_partitions(graph, result['path'])
        graph = fixture(self.base / 'tamper', True, '<i4', False)
        result = candidate.native_partitions(graph, self.base / 'tamper_out', 1, self.sign)
        with (self.base / 'tamper_out/p0.owners.bin').open('r+b') as stream:
            stream.write(b'\xff\xff\xff\xff')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            candidate.audit_partitions(graph, result['path'])

    def test_index_domain_guards_remain(self):
        graph = fixture(self.base / 'domain', True, '<i8', False)
        graph['offsets'][-1] = 2**31
        graph['m'] = 2**31
        with self.assertRaises(candidate.IndexUnsupported):
            candidate.native_partitions(graph, self.base / 'too_wide', 1, self.sign)
        graph['n'] = 2**31
        with self.assertRaises(candidate.IndexUnsupported):
            candidate.native_partitions(graph, self.base / 'too_many_vertices', 1, self.sign)


if __name__ == '__main__':
    unittest.main(verbosity=2)
