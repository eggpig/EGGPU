"""CPU-only E6 correctness, dependency, coverage and execution-gate contracts."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import fcntl

import numpy as np
import networkx as nx
import e6_worker as e6
import e6_reused as reused
import run_e6
import sg_worker as sg


class E6Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='e6_cpu_')
        cls.root = Path(cls.temporary.name)
        cls.path = e6.make_tiny(cls.root)
        cls.graph = sg.load_input(cls.path)
        cls.data = e6.weak_graph(cls.graph)
        cls.reference = e6.independent_reference(cls.data)

    @classmethod
    def tearDownClass(cls):
        for name in ('offsets', 'indices', 'weights'):
            mapped = cls.graph[name]
            if getattr(mapped, '_mmap', None) is not None:
                mapped._mmap.close()
        cls.temporary.cleanup()

    def values(self):
        return {name: reused.own(value) for name, value in self.reference.items()}

    def args(self, **updates):
        values = dict(system='eggpu', ablate='none', gpu_count=1, gpu_uuids=['gpu-a'],
            dataset=e6.TINY, phase='tiny', repeat=0, timeout=1800., release=Path('/release'),
            native_dso=Path('/build/cpp_easygraph.so'), output=self.root/'out')
        values.update(updates)
        return SimpleNamespace(**values)

    def backend(self):
        values = self.values()
        class FakeBackend:
            def __init__(self): self.calls = []
            def call(self, function, selected=None):
                self.calls.append((function, None if selected is None else np.array(selected)))
                return reused.own(values[function]), {'devices_used': [0], 'coverage_fallback': False}
        return FakeBackend()

    def test_tiny_has_32_actual_candidates_for_eight_gpu_shards(self):
        self.assertEqual(self.graph['n'], 40)
        self.assertEqual(len(self.reference['candidates']), 32)
        self.assertTrue(np.all(self.reference['candidates'] < 36))
        self.assertEqual(int((self.data['degree'] == 0).sum()), 4)
        self.assertEqual(self.data['n'], self.graph['n'])

    def test_exact_reference_against_networkx(self):
        g = nx.Graph(); g.add_nodes_from(range(self.data['n'])); g.add_edges_from(self.data['undirected'])
        effective, constraint = nx.effective_size(g), nx.constraint(g)
        np.testing.assert_allclose(self.reference['effective_size'], [effective[v] for v in g], atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(self.reference['constraint'], [constraint[v] for v in g], atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(self.reference['closeness'],
            [nx.closeness_centrality(g, int(v), wf_improved=True) for v in self.reference['candidates']], atol=1e-12)

    def test_directed_weak_projection_preserves_isolates_and_ignores_weights(self):
        graph = dict(n=5, m=3, offsets=np.array([0, 1, 2, 3, 3, 3]), indices=np.array([1, 2, 0]),
                     weights=np.array([3., 8., 16.]), input_sha256='synthetic', directed=True)
        result = e6.weak_graph(graph)
        self.assertEqual(result['n'], 5); self.assertEqual(result['m'], 6)
        self.assertEqual(result['degree'].tolist(), [2, 2, 2, 0, 0])
        self.assertEqual(set(map(tuple, result['edges'])), {(0,1),(1,0),(1,2),(2,1),(2,0),(0,2)})

    def test_candidate_and_winner_ties_are_predefined(self):
        np.testing.assert_array_equal(reused.select_candidates([2.,2.,2.,np.nan], [.5,.1,.1,np.nan], 2), [1,2])
        self.assertEqual(reused.select_winner(np.array([9,4,8]), np.array([.2,.2,.1])), 4)

    def test_selection_reacts_to_computed_scores(self):
        scores = np.arange(40, dtype=float)+1
        before = reused.select_candidates(scores, np.ones(40)); scores[0] = 1000
        after = reused.select_candidates(scores, np.ones(40))
        self.assertNotIn(0, before); self.assertEqual(after[0], 0)

    def test_flow_passes_actual_dependencies_and_full_outputs(self):
        backend = self.backend(); values, trace = e6.flow(backend, 40)
        self.assertEqual([fn for fn, _ in backend.calls], list(e6.FUNCTIONS))
        np.testing.assert_array_equal(backend.calls[2][1], values['candidates'])
        self.assertEqual(int(backend.calls[3][1]), int(values['winner'][0]))
        self.assertTrue(e6.audit_output(values, self.reference, 40)['pass'])
        self.assertGreaterEqual(trace['complete_seconds'], sum(x['complete_seconds'] for x in trace['stages']))
        self.assertTrue(e6.audit_execution(trace, self.args()))

    def test_mutable_return_rejected_before_selection(self):
        backend = self.backend(); original = backend.call
        backend.call = lambda fn, selected=None: (original(fn, selected)[0].copy(), {})
        with self.assertRaisesRegex(ValueError, 'immutable owning'):
            e6.flow(backend, 40)

    def test_missing_isolate_id_is_not_silent_success(self):
        backend = self.backend(); original = backend.call
        def incomplete(fn, selected=None):
            values, metrics = original(fn, selected)
            if fn == 'effective_size': metrics['vertex_ids'] = np.arange(39)
            return values, metrics
        backend.call = incomplete
        with self.assertRaisesRegex(ValueError, 'omitted or duplicated'):
            e6.flow(backend, 40)

    def test_full_domain_error_not_hidden_by_selected_subset(self):
        values = self.values(); changed = values['constraint'].copy(); changed[35] += .1
        values['constraint'] = reused.own(changed)
        self.assertFalse(e6.audit_output(values, self.reference, 40)['pass'])

    def test_nan_isolate_contract_checked(self):
        values = self.values(); changed = values['effective_size'].copy(); changed[39] = 0
        values['effective_size'] = reused.own(changed)
        self.assertFalse(e6.audit_output(values, self.reference, 40)['pass'])

    def test_bfs_and_component_membership_checked(self):
        values = self.values(); changed = values['bfs'].copy(); changed[39] = 1
        values['bfs'] = reused.own(changed)
        self.assertFalse(e6.audit_output(values, self.reference, 40)['bfs_wcc_membership_exact'])

    def test_output_ownership_independent_across_calls(self):
        backend = self.backend(); first, _ = e6.flow(backend, 40); second, _ = e6.flow(backend, 40)
        for name in first:
            self.assertFalse(np.shares_memory(first[name], second[name]))
            self.assertTrue(first[name].flags.owndata); self.assertFalse(first[name].flags.writeable)

    def test_bulk_projection_keeps_all_vertices_and_triangular_edges(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            manifest = e6.native_manifest(self.data, Path(temporary)/'native')
            meta = json.loads(manifest.read_text()); projection = json.loads(Path(meta['undirected_projection_manifest']).read_text())
            self.assertEqual(meta['num_nodes'], 40)
            offsets = np.fromfile(meta['offsets_path'], dtype='<i4')
            np.testing.assert_array_equal(offsets, self.data['offsets'])
            np.testing.assert_array_equal(np.fromfile(projection['degree_path'], dtype='<i4'), self.data['degree'])
            for stem in ('lower', 'upper', 'forward'):
                v = np.fromfile(projection[stem+'_V_path'], dtype='<i4')
                edges = np.fromfile(projection[stem+'_E_path'], dtype='<i4')
                self.assertEqual(len(v), 41); self.assertEqual(v[-1], self.data['m']//2)
                self.assertEqual(len(edges), self.data['m']//2)

    def test_cpu_reference_cache_bound_and_tamper_detected(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            values, receipt = e6.reference_cached(self.graph, Path(temporary))
            np.testing.assert_array_equal(values['candidates'], self.reference['candidates'])
            item = json.loads(Path(receipt['path']).read_text())
            blob = next(Path(temporary).glob('*.npz')); blob.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'cache changed'):
                e6.reference_cached(self.graph, Path(temporary))

    def test_target_needs_reference_before_gpu_work(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            with patch.object(e6,'independent_reference',side_effect=AssertionError('must not compute')):
                with self.assertRaisesRegex(ValueError,'Prepare the source-bound CPU'):
                    e6.reference_cached(self.graph,Path(temporary),allow_compute=False)

    def test_native_build_source_and_binary_both_bound(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            root=Path(temporary);source=root/'source';source.mkdir()
            code=source/'kernel.cu';code.write_text('test fixture, never compiled')
            binary=root/'cpp_easygraph.so';binary.write_bytes(b'not a loadable shared library')
            receipt=root/'build.json';sg.save(receipt,dict(status='PASS',source_sha256={'kernel.cu':sg.content_binding(code)['sha256']},
                outputs={'native/cpp_easygraph.so':sg.content_binding(binary)['sha256']}))
            args=self.args(release=source,native_dso=binary,build_receipt=receipt)
            self.assertEqual(e6.check_native_build(args),sg.content_binding(receipt))
            code.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Compiled source differs'):e6.check_native_build(args)
            code.write_text('test fixture, never compiled');binary.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Native DSO'):e6.check_native_build(args)

    def test_dense_native_replication_requires_complete_canonical_nodes(self):
        fake = e6.DenseNative.__new__(e6.DenseNative)
        fake.devices=[0,1]; fake.data_n=40; fake.graph=object(); fake.build=lambda: None
        class Cpp:
            def cpp_egmgpu_replicated_effective_size(self, *args, **kwargs):
                self.kwargs=kwargs
                return dict(devices_used=[0,1], nodes_dense=np.arange(40), values_dense=np.ones(40), kernel_seconds=0.)
        fake.cpp=Cpp(); values, metrics=fake.call('effective_size')
        self.assertEqual(fake.cpp.kwargs, {'output_format':'dense'})
        self.assertEqual(metrics['devices_used'], [0,1]); self.assertFalse(values.flags.writeable)

    def test_four_coverage_groups_delegate_real_calls(self):
        for group, functions in e6.GROUPS.items():
            backend=e6.CoverageBackend.__new__(e6.CoverageBackend); calls=[]
            class Impl:
                def __init__(self, name): self.name=name
                def call(self, fn, selected):
                    calls.append((self.name,fn)); return reused.own([0.]), {'devices_used': [] if self.name=='cpu' else [0]}
            backend.native=Impl('native');backend.cpu=Impl('cpu');backend.fallback=functions
            for fn in e6.FUNCTIONS: backend.call(fn)
            self.assertEqual([fn for where,fn in calls if where=='cpu'], list(functions))
            self.assertEqual(len(calls), 5)

    def test_complete_child_times_prep_and_audits_after_close_cpu_fake(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            directory=Path(temporary); lockpath=directory/'global.lock'
            backend=self.backend(); events=[]
            backend.close=lambda:events.append('closed')
            args=self.args(input_root=self.root, lock_file=lockpath, output=directory/'job',
                           reference_cache=directory/'references')
            with lockpath.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);args.lease_fd=lock.fileno()
                with patch.object(e6,'build_backend',return_value=backend), patch.object(e6,'reference_cached',
                    return_value=(self.reference, {'cpu_fixture_reference':True})):
                    result=e6.child(args)
            self.assertEqual(events,['closed']);self.assertEqual(result['status'],'PASS')
            self.assertTrue(result['outputs_survive_close']);self.assertEqual(len(backend.calls),15)
            for call in result['calls'].values():
                self.assertGreaterEqual(call['seconds'],call['trace']['complete_seconds'])
                self.assertEqual(set(call['outputs']),set(self.reference))
                self.assertTrue(call['quality']['pass'])

    def test_mg_execution_trace_rejects_fake_tail_parallelism(self):
        trace={'stages':[{'function':fn,'metrics':{'devices_used':list(range(8)) if fn in
            ('effective_size','constraint','closeness') else [0], 'coverage_fallback':False}} for fn in e6.FUNCTIONS]}
        args=self.args(gpu_count=8, gpu_uuids=[f'gpu-{i}' for i in range(8)])
        self.assertTrue(e6.audit_execution(trace,args))
        trace['stages'][-1]['metrics']['devices_used']=list(range(8))
        with self.assertRaisesRegex(ValueError,'Actual stage devices'):
            e6.audit_execution(trace,args)

    def test_invalid_resource_or_ablation_domains_rejected(self):
        for overrides in (dict(system='cugraph_composite',gpu_count=2,gpu_uuids=['a','b']),
            dict(system='igraph',gpu_count=1), dict(ablate='structural',gpu_count=2,gpu_uuids=['a','b']),
            dict(system='cugraph_igraph',ablate='closeness'), dict(dataset='US'),dict(repeat=10,phase='formal',dataset='EP')):
            with self.assertRaises(ValueError): e6.options_valid(self.args(**overrides))

    def plan_args(self, **overrides):
        args=dict(datasets='EP,LJ,OR', systems=','.join(e6.SYSTEMS), native_gpu_counts='1,2,4,8',
            phase='diagnostic', repeat=0, coverage_ablation=True, gpu_uuids=[f'gpu-{i}' for i in range(8)],
            proofs=None, output=self.root/'planned', python_core=Path('/core/python'), python_cugraph=Path('/rapids/python'),
            input_root=self.root, release=Path('/release'), native_dso=Path('/cpp.so'), arena_library_dir=Path('/arena'),
            setup=Path('/setup'), lock_file=Path('/lock'), reference_cache=Path('/references'), budget_mib=22000,
            min_host_free_gib=8., fallback_site=Path('/fallback'), build_receipt=Path('/build_receipt'))
        args.update(overrides);return SimpleNamespace(**args)

    def test_controller_fixed_menu_and_pressure_label(self):
        with patch('subprocess.run', side_effect=AssertionError('plan must not execute')):
            jobs=run_e6.plan(self.plan_args())
        self.assertEqual(len(jobs),33)
        self.assertEqual({j['dataset_role'] for j in jobs if j['dataset']=='OR'},{'pressure'})
        self.assertTrue(all(j['gpu_count']==1 for j in jobs if j['ablate']!='none'))
        self.assertTrue(all(j['gpu_count']==1 for j in jobs if j['system'].startswith('cugraph')))
        self.assertFalse((self.root/'planned').exists())

    def test_formal_cannot_start_without_qualifications(self):
        with self.assertRaisesRegex(ValueError,'Missing explicit E6 qualification'):
            run_e6.plan(self.plan_args(phase='formal'))

    def test_controller_balances_order_without_measured_selection(self):
        first=run_e6.plan(self.plan_args(repeat=0,datasets='EP'))
        second=run_e6.plan(self.plan_args(repeat=1,datasets='EP'))
        identify=lambda job:(job['system'],job['ablate'],job['gpu_count'])
        self.assertEqual(list(map(identify,first)),list(reversed(list(map(identify,second)))))

    def test_bfs_output_exact_unreachable_semantics(self):
        values=self.values(); changed=values['bfs'].copy()
        changed[np.isposinf(changed)]=np.finfo(np.float64).max;values['bfs']=reused.own(changed)
        self.assertFalse(e6.audit_output(values,self.reference,40)['pass'])

    def test_complete_ids_accepts_permutation_with_isolate(self):
        ids=np.array([3,0,2,1],dtype=np.int32)
        self.assertIs(reused.require_complete_ids(ids,4,np),ids)

    def test_complete_ids_rejects_missing_duplicate_and_invalid(self):
        for ids in ([0,1,2],[0,1,1,3],[-1,0,1,2],[0,1,2,4]):
            with self.assertRaises(ValueError): reused.require_complete_ids(np.asarray(ids),4,np)
        with self.assertRaises(ValueError): reused.require_complete_ids(np.arange(4,dtype=float),4,np)

    def test_wcc_equivalent_partition_labels_accepted(self):
        values=self.values();values['wcc']=reused.own(values['wcc']+1000)
        self.assertTrue(e6.audit_output(values,self.reference,40)['pass'])


if __name__ == '__main__':
    unittest.main()
