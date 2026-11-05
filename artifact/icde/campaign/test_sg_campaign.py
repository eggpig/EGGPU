"""CPU-only contract tests; no CUDA import or GPU execution."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import fcntl
import gc
import json
import tempfile
import unittest

import numpy as np
import sg_worker as worker
import run_sg
from cugraph_csr_adapter import CuGraphCSR, ordered_output


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        worker.make_tiny(self.root)
        self.graph = worker.load_input(self.root / 'canonical/TINY_D')

    def tearDown(self):
        for value in self.graph.values():
            if isinstance(value, np.memmap):
                value._mmap.close()
        del self.graph
        gc.collect()
        self.temporary.cleanup()

    def test_tiny_directedness_sources_and_weight_domain(self):
        for name, directed in [('TINY_D', True), ('TINY_U', False)]:
            graph = worker.load_input(self.root / 'canonical' / name)
            self.assertEqual(graph['directed'], directed)
            self.assertEqual(graph['sources'], [0, 1, 2])
            self.assertTrue(np.all((graph['weights'] >= 1) & (graph['weights'] <= 16)))
            matrix = worker.graph_matrix(graph)
            self.assertEqual((matrix != matrix.T).nnz == 0, not directed)

    def test_input_tamper_rejected(self):
        path = self.root / 'canonical/TINY_D/weights.bin'
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'content changed'):
            worker.load_input(path.parent)

    def test_three_sources_required_even_when_receipt_is_rebound(self):
        directory = self.root / 'canonical/TINY_D'
        worker.save(directory / 'sources.json', {'primary_sources': [0, 0, 1], 'selection_uses_timing': False})
        receipt = json.loads((directory / 'receipt.json').read_text())
        receipt['files']['sources'] = worker.binding(directory / 'sources.json')
        worker.save(directory / 'receipt.json', receipt)
        with self.assertRaisesRegex(ValueError, 'three distinct'):
            worker.load_input(directory)

    def test_exact_distance_and_partition_audits(self):
        bfs = worker.reference(self.graph, 'bfs', 0).copy()
        np.testing.assert_array_equal(bfs, [0, 1, 2, 3, np.inf, np.inf, np.inf])
        bfs.setflags(write=False)
        self.assertTrue(worker.audit_output(bfs, bfs, self.graph, 'bfs')['pass'])
        wrong = bfs.copy(); wrong[3] = 2; wrong.setflags(write=False)
        self.assertFalse(worker.audit_output(wrong, bfs, self.graph, 'bfs')['pass'])
        expected = worker.reference(self.graph, 'wcc', None)
        labels = np.array([101, 101, 101, 101, -9, -9, 5], dtype=np.int64)
        labels.setflags(write=False)
        self.assertTrue(worker.audit_output(labels, expected, self.graph, 'wcc')['pass'])
        labels = np.array([0, 0, 0, 1, 2, 2, 3]); labels.setflags(write=False)
        self.assertFalse(worker.audit_output(labels, expected, self.graph, 'wcc')['pass'])

    def test_pr_asymmetry_dangling_and_independent_reference(self):
        expected = worker.reference(self.graph, 'pr', None)
        # Independent dense linear solve includes isolated/dangling redistribution.
        matrix = worker.graph_matrix(self.graph).toarray()
        degree = matrix.sum(axis=1)
        transition = np.divide(matrix, degree[:, None], out=np.zeros_like(matrix), where=degree[:, None] != 0)
        transition[degree == 0] = 1 / self.graph['n']
        exact = np.linalg.solve(np.eye(self.graph['n']) - .85 * transition.T,
                                np.full(self.graph['n'], .15 / self.graph['n']))
        np.testing.assert_allclose(expected, exact, atol=1e-12)
        expected.setflags(write=False)
        self.assertTrue(worker.audit_output(expected, exact, self.graph, 'pr')['pass'])
        uniform = np.full(self.graph['n'], 1 / self.graph['n']); uniform.setflags(write=False)
        self.assertFalse(worker.audit_output(uniform, exact, self.graph, 'pr')['pass'])
        inf = expected.copy(); inf[0] = np.inf; inf.setflags(write=False)
        self.assertFalse(worker.audit_output(inf, exact, self.graph, 'pr')['pass'])

    def test_reference_cache_detects_mutation(self):
        cache = self.root / 'refs'
        first, receipt = worker.reference_cached(self.graph, 'sssp', 1, cache)
        second, _ = worker.reference_cached(self.graph, 'sssp', 1, cache)
        np.testing.assert_array_equal(first, second)
        path = next(cache.glob('*.npy'))
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'cache changed'):
            worker.reference_cached(self.graph, 'sssp', 1, cache)

    def test_output_dedup_is_local_and_immutable(self):
        out = self.root / 'outputs'; out.mkdir()
        values = np.arange(7, dtype=float); values.setflags(write=False)
        first = worker.persist_output(out, values)
        second = worker.persist_output(out, values.copy())
        self.assertEqual(first, second)
        self.assertEqual(len(list(out.glob('*.npy'))), 1)

    def test_cpu_fake_backend_exercises_lease_timings_and_output_receipts(self):
        lock_path = self.root / 'global.lock'
        args = SimpleNamespace(lease_fd=None, lock_file=lock_path, input_root=self.root,
            dataset='TINY_D', algorithm='pr', source_index=0, system='cugraph',
            output=self.root / 'cpu_fake_source', timeout=1800., reference_cache=self.root / 'refs')
        backend = SimpleNamespace(contract={'test_double': True}, close=lambda: None)
        def fake_build(a, graph, source):
            expected = worker.reference(graph, 'pr', None)
            def query():
                value = expected.copy(); value.setflags(write=False); return value
            return backend, query
        with lock_path.open('a') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            args.lease_fd = lease.fileno()
            with mock.patch.object(worker, 'build_backend', side_effect=fake_build):
                record = worker.source_child(args)
        self.assertEqual(record['status'], 'PASS')
        self.assertTrue(record['outputs_survive_close'])
        self.assertEqual(set(record['calls']), {'first', 'warmup', 'ready'})
        self.assertGreater(record['calls']['first']['seconds'], 0)
        self.assertEqual(len(list(args.output.glob('*.npy'))), 1)
        # A proof binds not only an affirmative status but every child/output hash.
        epoch = {'test_double': True}
        epoch_path = self.root / 'epoch.json'; worker.save(epoch_path, epoch)
        proof_path = self.root / 'qualification.json'
        proof = dict(status='PASS_QUALIFICATION', phase='tiny', system='cugraph', algorithm='pr',
            epoch_sha256=worker.digest_json(epoch), epoch=worker.binding(epoch_path),
            quality_pass=True, dataset='TINY_D', directed=True,
            sources=[{'result': worker.binding(args.output / 'result.json')}])
        worker.save(proof_path, proof)
        kwargs = dict(kind='tiny', system='cugraph', algorithm='pr',
                      epoch_sha=worker.digest_json(epoch), graph=self.graph)
        worker.validate_proof(proof_path, **kwargs)
        with self.assertRaisesRegex(ValueError, 'directedness'):
            worker.validate_proof(proof_path, **dict(kwargs, graph=dict(self.graph, directed=False)))
        (args.output / 'result.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'source receipt changed'):
            worker.validate_proof(proof_path, **kwargs)

    def test_formal_gate_rejects_missing_and_stale_proofs(self):
        proof = self.root / 'proof.json'
        worker.save(proof, {'status': 'PASS_QUALIFICATION', 'phase': 'target', 'system': 'eggpu',
                           'algorithm': 'bfs', 'epoch_sha256': 'old', 'quality_pass': True})
        with self.assertRaisesRegex(ValueError, 'current source'):
            worker.validate_proof(proof, kind='target', system='eggpu', algorithm='bfs',
                                  epoch_sha='new', graph=self.graph)
        with self.assertRaisesRegex(RuntimeError, 'global lease'):
            worker.source_child(SimpleNamespace(lease_fd=None))

    def test_cugraph_native_output_domain_normalization(self):
        ids = np.array([2, 0, 1], dtype=np.int32)
        raw = np.array([np.iinfo(np.int32).max, 0, 1], dtype=np.int32)
        np.testing.assert_array_equal(ordered_output(ids, raw, 3, 'bfs'), [0, 1, np.inf])
        with self.assertRaisesRegex(ValueError, 'Missing/duplicated'):
            ordered_output(np.array([0, 0, 2]), np.ones(3), 3, 'sssp')
        with self.assertRaisesRegex(ValueError, 'FP64'):
            ordered_output(np.arange(3), np.ones(3, dtype=np.float32), 3, 'pr')

    def test_cugraph_accepts_reciprocal_undirected_for_every_algorithm(self):
        graph = worker.load_input(self.root / 'canonical/TINY_U')
        cp = SimpleNamespace(asarray=np.asarray, ones=np.ones, arange=np.arange,
            float64=np.float64, int32=np.int32,
            cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None)))
        properties = []
        def props(**kwargs):
            properties.append(kwargs); return kwargs
        plc = SimpleNamespace(ResourceHandle=lambda: object(), GraphProperties=props,
                              SGGraph=lambda *a, **kw: object())
        for algorithm in worker.ALGORITHMS:
            backend = CuGraphCSR(graph, algorithm, cp=cp, plc=plc)
            self.assertTrue(properties[-1]['is_symmetric'])
            backend.close()
        with self.assertRaisesRegex(ValueError, 'explicit weak CSR'):
            CuGraphCSR(self.graph, 'wcc', cp=cp, plc=plc)

    def test_serial_plan_retains_ten_repeats_and_balances_system_order(self):
        # Reuse a fixture under EP only to test CPU scheduling, never execute it.
        ep = self.root / 'canonical/EP'; ep.mkdir()
        worker.save(ep / 'canonical.json', {'directed': True})
        args = SimpleNamespace(datasets='EP', systems='eggpu,cugraph', algorithms='bfs',
            phase='formal', repeat=None, proofs=None, output=self.root / 'attempt',
            input_root=self.root, release=self.root, arena_library_dir=self.root, native_dso=self.root/'fake.so',
            setup=self.root/'setup.json', gpu_uuid='GPU-test', lock_file=self.root/'global.lock',
            reference_cache=self.root/'refs', budget_mib=22000, min_host_free_gib=8,
            python_eggpu=Path('/python/eggpu'), python_cugraph=Path('/python/cugraph'))
        with self.assertRaisesRegex(ValueError, 'Missing explicit tiny'):
            run_sg.plan(args)
        args.phase = 'diagnostic'; args.repeat = 1
        jobs = run_sg.plan(args)
        self.assertEqual([x['system'] for x in jobs], ['cugraph', 'eggpu'])
        self.assertTrue(all(x['repeat'] == 1 for x in jobs))
        proofs = {}
        for system in ('eggpu', 'cugraph'):
            proofs[('tiny', system, 'bfs', 'TINY_D')] = self.root/'tiny.json'
            proofs[('target', system, 'bfs', 'EP')] = self.root/'target.json'
        args.phase = 'formal'; args.repeat = None
        with mock.patch.object(run_sg, 'proof_index', return_value=proofs):
            jobs = run_sg.plan(args)
        self.assertEqual(len(jobs), 20)
        self.assertEqual({x['repeat'] for x in jobs}, set(range(10)))
        self.assertEqual(len({x['output'] for x in jobs}), 20)

    def test_resource_gate_rechecks_stale_utilization_without_overlapping(self):
        setup = self.root / 'setup.json'
        worker.save(setup, {'gpus': {'output': 'index, uuid\n0, GPU-test\n'}})
        args = SimpleNamespace(setup=setup, gpu_uuid='GPU-test', budget_mib=22000, min_host_free_gib=8)
        busy = '0, GPU-test, RTX-test, 24564, 0, 12, 580\n'
        idle = '0, GPU-test, RTX-test, 24564, 0, 0, 580\n'
        with mock.patch.object(worker.subprocess, 'check_output', side_effect=[busy, '', idle, '']), \
             mock.patch.object(worker.time, 'sleep') as sleep, \
             mock.patch.object(worker, 'cgroup_memory', return_value={'available_bytes': 64 << 30}):
            record = worker.resource_check(args, self.graph)
        self.assertEqual(record['selected']['utilization'], '0')
        sleep.assert_called_once_with(1)
        with mock.patch.object(worker.subprocess, 'check_output', side_effect=[idle, 'GPU-test, 123, 1000']), \
             mock.patch.object(worker.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'active processes'):
                worker.resource_check(args, self.graph)
            sleep.assert_not_called()
        with mock.patch.object(worker.subprocess, 'check_output', side_effect=[idle, '']), \
             mock.patch.object(worker, 'cgroup_memory', return_value={'available_bytes': 1 << 30}):
            with self.assertRaisesRegex(RuntimeError, 'headroom insufficient'):
                worker.resource_check(args, self.graph)


if __name__ == '__main__':
    unittest.main()
