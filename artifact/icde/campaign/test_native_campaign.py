"""CPU-only native layout, capacity, timing-boundary and qualification tests."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import fcntl
import gc
import json
import tempfile
import unittest

import numpy as np
import native_input
import native_worker
import run_native
import sg_worker as shared

RELEASE = Path(__file__).resolve().parents[3] / 'Easy-Graph'


class NativeCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        native_worker.make_tiny(self.root)
        self.graph = shared.load_input(self.root / 'canonical/TINY_D')
        self.sign = native_input.load_signer(RELEASE)

    def tearDown(self):
        for value in self.graph.values():
            if isinstance(value, np.memmap):
                value._mmap.close()
        del self.graph
        gc.collect()
        self.temporary.cleanup()

    def test_fixture_preserves_isolates_and_supports_all_partition_counts(self):
        self.assertEqual(self.graph['n'], 16)
        self.assertEqual(len(self.graph['offsets']), 17)
        self.assertEqual(self.graph['sources'], [0, 1, 2])
        for count in (1, 2, 4, 8):
            graph = native_input.original_graph(self.graph, 'tiny')
            manifest = native_input.native_partitions(graph, self.root / f'p{count}', count, self.sign, weighted=True)
            audit = native_input.audit_partitions(graph, manifest['path'])
            self.assertEqual(audit['stored_arcs_checked'], 6)
            doc = json.loads(Path(manifest['path']).read_text())
            self.assertEqual(sum(part['owned_vertex_count'] for part in doc['partitions']), 16)
            self.assertEqual([part['device_id'] for part in doc['partitions']], list(range(count)))
            self.assertEqual(self.sign(doc), (doc['graph']['topology_sha256'], doc['graph']['weight_sha256'], doc['view']['signature']))

    def test_weak_projection_retains_every_original_and_reverse_arc(self):
        graph = native_input.original_graph(self.graph, 'tiny')
        weak, info = native_input.weak_projection(graph, self.root / 'weak')
        matrix = shared.graph_matrix(self.graph)
        expected = matrix.maximum(matrix.T)
        np.testing.assert_array_equal(weak['offsets'], expected.indptr)
        np.testing.assert_array_equal(weak['indices'], expected.indices)
        self.assertFalse(weak['directed'])
        self.assertEqual(weak['n'], 16)
        for count in (1, 2, 4, 8):
            manifest = native_input.native_partitions(weak, self.root / f'weak_p{count}', count, self.sign)
            audit = native_input.audit_partitions(weak, manifest['path'])
            self.assertEqual(audit['stored_arcs_checked'], weak['m'])
            self.assertFalse(audit['weighted'])

    def test_global_int64_offsets_are_rebased_without_truncating_local_offsets(self):
        graph = native_input.original_graph(self.graph, 'global-offsets-64')
        graph['offsets'] = graph['offsets'].astype(np.int64)
        # A small capacity models the real signed-int32 boundary without
        # allocating billions of test edges; the identical check guards it.
        with mock.patch.object(native_input, 'MAX_LOCAL_ENTRIES', 4):
            with self.assertRaisesRegex(native_input.IndexUnsupported, 'INDEX_UNSUPPORTED'):
                native_input.native_partitions(graph, self.root / 'unsupported_p1', 1, self.sign)
            self.assertFalse((self.root / 'unsupported_p1').exists())
            manifest = native_input.native_partitions(graph, self.root / 'supported_p2', 2, self.sign)
        doc = json.loads(Path(manifest['path']).read_text())
        self.assertEqual(doc['graph']['adjacency_count'], 6)
        self.assertEqual(sum(p['adjacency_count'] for p in doc['partitions']), 6)
        self.assertTrue(all(p['adjacency_count'] <= 4 for p in doc['partitions']))
        self.assertEqual(manifest['input_global_offset_dtype'], 'int64')
        native_input.audit_partitions(graph, manifest['path'])

    def test_projection_chooses_int64_when_global_capacity_boundary_is_crossed(self):
        graph = native_input.original_graph(self.graph, 'tiny')
        with mock.patch.object(native_input, 'MAX_LOCAL_ENTRIES', 4):
            weak, info = native_input.weak_projection(graph, self.root / 'weak64')
        self.assertEqual(weak['offsets'].dtype, np.dtype('<i8'))
        self.assertEqual(int(weak['offsets'][-1]), weak['m'])
        self.assertTrue(info['all_arcs_retained'])

    def test_tampered_partition_rejected_by_full_audit(self):
        graph = native_input.original_graph(self.graph, 'tiny')
        manifest = native_input.native_partitions(graph, self.root / 'p2', 2, self.sign)
        path = self.root / 'p2/p0.indices.bin'
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            native_input.audit_partitions(graph, manifest['path'])

    def test_generated_cleanup_never_changes_hardlinked_canonical_inputs(self):
        graph = native_input.original_graph(self.graph, 'tiny')
        before = {role: shared.binding(path) for role, path in graph['paths'].items()}
        manifest = native_input.native_partitions(graph, self.root / 'p1', 1, self.sign, weighted=True)
        native_input.audit_partitions(graph, manifest['path'])
        cleanup = native_input.remove_generated_payloads(manifest['path'])
        self.assertFalse(cleanup['canonical_inputs_modified'])
        self.assertTrue(Path(manifest['path']).is_file())
        self.assertEqual(list((self.root / 'p1').glob('*.bin')), [])
        self.assertEqual(before, {role: shared.binding(path) for role, path in graph['paths'].items()})

    def test_cpu_fake_native_exercises_preparation_output_and_lease(self):
        lock_path = self.root / 'global.lock'
        args = SimpleNamespace(lease_fd=None, lock_file=lock_path, input_root=self.root,
            dataset='TINY_D', algorithm='wcc', source_index=0, count=8, keep_prepared=False,
            output=self.root/'fake_source', reference_cache=self.root/'refs')
        sign = self.sign
        class FakeBackend:
            def __init__(self, args, graph, source):
                base = native_input.original_graph(graph, args.dataset)
                prepared, projection = native_input.weak_projection(base, args.output/'weak')
                manifest = native_input.native_partitions(prepared, args.output/'partitions', args.count, sign)
                self.preparation = {'manifest': manifest, 'projection': projection}
                self.audit_graph = prepared
                self.expected = shared.reference(graph, args.algorithm, source)
                self.last_metrics = {'devices_used': list(range(args.count)), 'cpu_test_double': True}
            def query(self):
                value = self.expected.astype(np.float64, copy=True); value.setflags(write=False)
                return value
            def close(self):
                pass
        with lock_path.open('a') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX); args.lease_fd = lease.fileno()
            with mock.patch.object(native_worker, 'NativeBackend', FakeBackend):
                result = native_worker.source_child(args)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['partition_audit']['status'], 'PASS_FULL_PARTITION_ROUNDTRIP')
        self.assertGreater(result['calls']['first']['seconds'], 0)
        self.assertEqual(set(result['calls']), {'first', 'warmup', 'ready'})
        self.assertEqual(result['calls']['first']['native_metrics']['devices_used'], list(range(8)))
        self.assertTrue(result['outputs_survive_close'])
        self.assertEqual(list((args.output/'partitions').glob('*.bin')), [])
        self.assertEqual(list((args.output/'weak').glob('*.bin')), [])
        epoch_path = self.root/'epoch.json'; shared.save(epoch_path, {'cpu_test_double': True})
        digest = shared.digest_json({'cpu_test_double': True})
        proof = dict(status='PASS_QUALIFICATION', implementation='forced_native', gpu_count=8,
            phase='tiny', system='eggpu', algorithm='wcc', dataset='TINY_D', directed=True,
            epoch_sha256=digest, epoch=shared.binding(epoch_path), quality_pass=True,
            sources=[{'result': shared.binding(args.output/'result.json')}])
        proof_path = self.root/'proof.json'; shared.save(proof_path, proof)
        native_worker.validate_proof(proof_path, 'tiny', args, digest, self.graph)
        manifest = Path(result['preparation']['manifest']['path'])
        manifest.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            native_worker.validate_proof(proof_path, 'tiny', args, digest, self.graph)

    def test_e2_scope_and_e1_mg_resource_count_are_explicit(self):
        args = SimpleNamespace(gpu_uuids='GPU-a', count=1, experiment='E2', dataset='EP', phase='diagnostic', repeat=0)
        with self.assertRaisesRegex(ValueError, 'Frozen E2'):
            native_worker.configure(args)
        args.experiment, args.dataset = 'E1-MG', 'EP'
        with self.assertRaisesRegex(ValueError, 'eight GPUs'):
            native_worker.configure(args)
        args.experiment, args.dataset = 'E2', 'US'
        self.assertEqual(native_worker.configure(args).uuids, ['GPU-a'])

    def test_controller_uses_native_p1_and_requires_each_p_qualification(self):
        directory = self.root/'canonical/US'; directory.mkdir()
        shared.save(directory/'canonical.json', {'directed': False})
        args = SimpleNamespace(datasets='US', algorithms='bfs', counts='1,2,4,8',
            gpu_uuids=','.join('GPU-'+str(i) for i in range(8)), experiment='E2', phase='formal',
            repeat=None, proofs=None, input_root=self.root, output=self.root/'attempt',
            python=Path('/python/core'), release=RELEASE, native_dso=self.root/'fake.so', setup=self.root/'setup.json',
            lock_file=self.root/'global.lock', reference_cache=self.root/'refs', budget_mib=22000,
            retained_budget_mib=0, min_host_free_gib=8, keep_prepared=False)
        with self.assertRaisesRegex(ValueError, 'Missing fixed-native tiny'):
            run_native.plan(args)
        proofs = {(phase, 'bfs', name, count): self.root/'proof.json'
                  for phase, name in [('tiny', 'TINY_U'), ('target', 'US')] for count in (1, 2, 4, 8)}
        with mock.patch.object(run_native, 'proof_index', return_value=proofs):
            jobs = run_native.plan(args)
        self.assertEqual(len(jobs), 40)
        self.assertEqual({job['repeat'] for job in jobs}, set(range(10)))
        self.assertEqual([job['gpu_count'] for job in jobs[:8]], [1, 2, 4, 8, 8, 4, 2, 1])
        self.assertTrue(all(job['implementation'] == 'forced_native' for job in jobs))
        self.assertTrue(all('native_worker.py' in job['command'][1] for job in jobs))
        self.assertTrue(all(job['communication'] == 'host_staging' for job in jobs))
        for job in jobs:
            index = job['command'].index('--communication')
            self.assertEqual(job['command'][index+1], 'host_staging')
        args.communication = 'nccl'
        with mock.patch.object(run_native, 'proof_index', return_value=proofs):
            nccl_jobs = run_native.plan(args)
        self.assertTrue(all(job['communication'] == 'nccl' for job in nccl_jobs))

    def test_runtime_dependency_identity_ignores_aslr_but_rejects_missing_libraries(self):
        library = self.root/'libtest.so'; library.write_bytes(b'cpu test fixture')
        outputs = [f'libtest.so => {library} (0x1234)\n', f'libtest.so => {library} (0x5678)\n']
        with mock.patch.object(native_worker.subprocess, 'check_output', side_effect=outputs):
            first = native_worker.runtime_dependencies(self.root/'module.so')
            second = native_worker.runtime_dependencies(self.root/'module.so')
        self.assertEqual(first, second)
        with mock.patch.object(native_worker.subprocess, 'check_output', return_value='libcuda.so => not found\n'):
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                native_worker.runtime_dependencies(self.root/'module.so')


if __name__ == '__main__':
    unittest.main()
