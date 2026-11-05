"""CPU evidence tampering/regression fixtures; never invoke a worker or GPU."""
from pathlib import Path
from unittest import mock
import copy
import importlib.util
import json
import os
import sys
import unittest

import inspect_ledger as m

A, PARENT = m.parent_helper()
sys.modules['admit'] = A
spec = importlib.util.spec_from_file_location('_original_admission_test_fixture',
                                            PARENT.with_name('test_admit.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class Ledger(unittest.TestCase):
    def setUp(self):
        require_tmp = Path(os.environ['EGGPU_TEST_TMP']).resolve()
        self.assertTrue(require_tmp.is_dir())
        self.fixture = fixtures.Admission('runTest')
        self.fixture.setUp()
        self.f = f = self.fixture
        self.root = f.root
        self.setup = f.save('setup.json', {'gpu_uuids': ['GPU-FIXTURE']})
        worker = f.manifest['proof_readers']['sg']['path']
        for job in f.qual['jobs'] + f.formal['jobs']:
            job['command'][1] = worker
            job['command'] += ['--setup', self.setup['path'], '--gpu-uuid', 'GPU-FIXTURE']
            job['timeout'] = 1800
        self.sync_plans()
        self.request = dict(schema='eggpu.formal.disposition.inspect.v1', group='sg',
                            manifest=f.b.bind(f.manifest_path), dispositions=[])
        self.request_path = self.root / 'request.json'

    def tearDown(self):
        self.fixture.tearDown()

    def sync_plans(self, group='sg'):
        f = self.f
        f.qbinding = f.save('qualification.json', f.qual)
        f.fbinding = f.save('pending.json', f.formal)
        group_spec = f.manifest['groups'].get(group, f.manifest['groups']['sg'])
        group_spec.update(pending=f.fbinding, qualifications=[f.qbinding],
                          expected_formal_jobs=len(f.formal['jobs']))
        f.manifest['groups'][group] = group_spec
        f.write(f.manifest_path, f.manifest)
        if hasattr(self, 'request'):
            self.request['manifest'] = f.b.bind(f.manifest_path)

    def run_inspect(self):
        self.f.write(self.request_path, self.request)
        return m.inspect(self.request_path)

    def assert_unresolved(self, report, status=None):
        self.assertFalse(report['all_cells_resolved_and_proofs_checked'])
        self.assertFalse(report['formal_admitted'])
        self.assertEqual(report['workers_launched'], 0)
        if status:
            self.assertEqual(report['cells'][0]['status'], status)

    def disposition(self, classification='OOM'):
        f = self.f
        target, tiny = f.qual['jobs'][-1], f.qual['jobs'][0]
        proof = A.load(target['receipt'])
        proof['status'] = 'FAILED'
        f.write(target['receipt'], proof)
        log = self.root / 'failed-native.log'
        log.write_text('original failure: cudaErrorMemoryAllocation\n')
        resources = dict(schema='eggpu.reviewed.failure.resources.v1', cell_id=target['id'],
                         source_sha256=f.manifest['source_sha256'],
                         epoch_sha256=proof['epoch_sha256'], input_sha256=proof['input_sha256'],
                         failure_stage='graph_upload', execution_attempted=True,
                         unrelated_gpu_contention=False, disk_preflight_passed=True,
                         host_preflight_passed=True, gpu_uuids=['GPU-FIXTURE'],
                         evidence={'original_runtime_log': f.b.bind(log)})
        rec = dict(schema='eggpu.formal.cell.disposition.v1', cell_id=target['id'],
                   classification=classification,
                   cell_identity_sha256=m.stage_identity(f.manifest, 'sg', target, f.formal['jobs']),
                   source_sha256=f.manifest['source_sha256'], epoch_sha256=proof['epoch_sha256'],
                   input_sha256=proof['input_sha256'], failure_stage='graph_upload',
                   review=dict(authority='primary_agent_explicit', decision='accept_capability_terminal',
                               rationale='Synthetic CPU test only: device allocation failure.'),
                   error_marker=dict(path=str(log), text='cudaErrorMemoryAllocation'),
                   facts=dict(memory_domain='device', error_class='cudaErrorMemoryAllocation'),
                   evidence=dict(target_receipt=f.b.bind(target['receipt']),
                                 tiny_receipt=f.b.bind(tiny['receipt']), raw_logs=[f.b.bind(log)],
                                 resource_observation=f.save('resources.json', resources), setup=self.setup))
        self.request['dispositions'] = [f.save('disposition.json', rec)]
        return rec

    def save_disposition(self, record):
        self.request['dispositions'] = [self.f.save('disposition.json', record)]

    def replace_marker(self, record, text):
        log = Path(record['evidence']['raw_logs'][0]['path'])
        log.write_text(text + '\n')
        bound = self.f.b.bind(log)
        record['evidence']['raw_logs'] = [bound]
        record['error_marker'] = dict(path=str(log), text=text)
        path = Path(record['evidence']['resource_observation']['path'])
        resource = A.load(path); resource['evidence']['original_runtime_log'] = bound
        self.f.write(path, resource)
        record['evidence']['resource_observation'] = self.f.b.bind(path)

    def test_real_parent_deep_proof_qualifies_cell_but_never_admits(self):
        report = self.run_inspect()
        self.assertEqual(report['counts'], {'QUALIFIED': 1})
        self.assertTrue(report['all_cells_resolved_and_proofs_checked'])
        self.assertFalse(report['formal_admitted'])
        self.assertEqual(report['original_formal_slots'], 10)
        self.assertEqual([s['original_job'] for s in report['cells'][0]['formal_slots']],
                         self.f.formal['jobs'])
        self.assertTrue(all(s['time_seconds'] is None and not s['counts_as_win'] and
                            not s['execution_scheduled'] for s in report['cells'][0]['formal_slots']))
        self.assertNotIn('plan', report)
        self.assertFalse((self.root / 'admitted').exists())

    def test_explicit_reviewed_oom_retains_ten_display_slots_without_timings(self):
        self.disposition()
        report = self.run_inspect()
        self.assertEqual(report['counts'], {'OOM': 1})
        self.assertTrue(report['all_cells_resolved_and_proofs_checked'])
        self.assertEqual(len(report['cells'][0]['formal_slots']), 10)
        self.assertFalse(report['cells'][0]['counts_as_win'])

    def test_generic_failed_receipt_without_review_remains_error(self):
        target = self.f.qual['jobs'][-1]['receipt']
        p = A.load(target); p['status'] = 'FAILED'; self.f.write(target, p)
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_missing_receipt_stays_pending_and_is_not_a_terminal(self):
        Path(self.f.qual['jobs'][-1]['receipt']).unlink()
        self.assert_unresolved(self.run_inspect(), 'PENDING')

    def test_raw_quality_failure_stays_correctness_error(self):
        target = self.f.qual['jobs'][-1]['receipt']
        p = A.load(target); p.update(status='FAIL_QUALITY', quality_pass=False); self.f.write(target, p)
        self.assert_unresolved(self.run_inspect(), 'CORRECTNESS')

    def test_default_false_quality_flag_is_not_misreported_as_numeric_failure(self):
        target = self.f.qual['jobs'][-1]['receipt']
        p = A.load(target); p.update(status='FAILED', quality_pass=False); self.f.write(target, p)
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_measured_wrong_result_cannot_be_reviewed_away_as_oom(self):
        rec = self.disposition()
        target = self.f.qual['jobs'][-1]['receipt']
        p = A.load(target); p['status'] = 'FAIL_QUALITY'; self.f.write(target, p)
        rec['evidence']['target_receipt'] = self.f.b.bind(target); self.save_disposition(rec)
        report = self.run_inspect()
        self.assert_unresolved(report, 'INFRA')
        self.assertIn('correctness failure cannot be waived', report['cells'][0]['reason'])

    def test_missing_duplicate_and_out_of_range_repeats_rejected(self):
        original = copy.deepcopy(self.f.formal)
        for kind in ['missing', 'duplicate', 'out_of_range']:
            self.f.formal = copy.deepcopy(original)
            if kind == 'missing':
                self.f.formal['jobs'].pop()
            else:
                c = self.f.formal['jobs'][-1]['command']
                c[c.index('--repeat')+1] = '0' if kind == 'duplicate' else '10'
            self.sync_plans()
            with self.assertRaisesRegex(ValueError, 'ten repeats'):
                self.run_inspect()

    def test_omitted_or_duplicate_claimed_cells_rejected(self):
        for ids in [[], ['target', 'target'], ['other']]:
            self.request['expected_cell_ids'] = ids
            with self.assertRaisesRegex(ValueError, 'omits/duplicates'):
                self.run_inspect()

    def test_changed_failure_log_rejected_even_same_size(self):
        self.disposition()
        p = self.root / 'failed-native.log'
        p.write_bytes(b'x' * p.stat().st_size)
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_review_for_old_source_epoch_rejected(self):
        rec = self.disposition(); rec['source_sha256'] = 'old_epoch'; self.save_disposition(rec)
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_failed_target_directedness_cannot_differ_from_tiny_or_input(self):
        rec = self.disposition()
        target = self.f.qual['jobs'][-1]['receipt']
        proof = A.load(target); proof['directed'] = True; self.f.write(target, proof)
        rec['evidence']['target_receipt'] = self.f.b.bind(target); self.save_disposition(rec)
        report = self.run_inspect()
        self.assert_unresolved(report, 'INFRA')
        self.assertIn('directedness differs', report['cells'][0]['reason'])

    def test_numeric_pass_status_cannot_skip_frozen_native_protocol(self):
        target = self.f.qual['jobs'][0]
        proof = A.load(target['receipt'])
        proof['status'] = 'PASS_STATIC_NATIVE_NUMERIC_QUALIFICATION'
        self.f.write(target['receipt'], proof)
        auditor = m.Auditor(A, A.Binder(), self.f.manifest, 'native')
        with self.assertRaisesRegex(ValueError, 'timing/protocol epoch differs'):
            auditor.common(target)

    def test_terminal_disposition_cannot_skip_formal_file_bindings(self):
        payload = self.root / 'formal_only_config.json'
        payload.write_text('{"frozen": true}')
        bound = self.f.b.bind(payload)
        for job in self.f.formal['jobs']:
            job['bindings'].append(bound)
        self.sync_plans()
        self.disposition()
        payload.write_text('{"frozen":false}')
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_failed_e4s_target_cannot_skip_formal_source_sequence(self):
        target = self.f.qual['jobs'][-1]
        proof = A.load(target['receipt'])
        proof.update(status='FAILED', source_sequence=[0, 1])
        self.f.write(target['receipt'], proof)
        for job in self.f.formal['jobs']:
            job['target_contract'] = dict(source_sequence=[1, 0], input_sha256=proof['input_sha256'])
        auditor = m.Auditor(A, A.Binder(), self.f.manifest, 'e4s')
        with self.assertRaisesRegex(ValueError, 'source sequence/input differs'):
            auditor.formal_contract(target, self.f.formal['jobs'])

    def test_review_for_other_resource_command_rejected(self):
        self.disposition()
        for job in self.f.qual['jobs'] + self.f.formal['jobs']:
            c = job['command']; c[c.index('--budget-mib')+1] = '21000'
        self.sync_plans()
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_disposition_cannot_waive_success_or_manufacture_pass(self):
        rec = self.disposition(); rec['classification'] = 'QUALIFIED'; self.save_disposition(rec)
        with self.assertRaisesRegex(ValueError, 'manufacture PASS'):
            self.run_inspect()

    def test_disposition_cannot_convert_pending_or_success_to_oom(self):
        rec = self.disposition()
        target = self.f.qual['jobs'][-1]['receipt']
        for status in ['RUNNING', 'PASS_QUALIFICATION']:
            p = A.load(target); p['status'] = status; self.f.write(target, p)
            rec['evidence']['target_receipt'] = self.f.b.bind(target); self.save_disposition(rec)
            self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_classification_does_not_accept_host_memory_as_device_oom(self):
        rec = self.disposition(); rec['facts']['memory_domain'] = 'host'; self.save_disposition(rec)
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_root_explicit_review_is_required(self):
        rec = self.disposition(); rec['review']['authority'] = 'automatic'; self.save_disposition(rec)
        with self.assertRaisesRegex(ValueError, 'primary-agent review'):
            self.run_inspect()

    def test_unreviewed_or_duplicate_disposition_cell_rejected(self):
        rec = self.disposition()
        binding = self.request['dispositions'][0]
        self.request['dispositions'] = [binding, binding]
        with self.assertRaisesRegex(ValueError, 'Duplicate disposition'):
            self.run_inspect()
        rec['cell_id'] = 'outside_matrix'; self.save_disposition(rec)
        with self.assertRaisesRegex(ValueError, 'outside fixed matrix'):
            self.run_inspect()

    def test_pretend_pass_with_corrupt_full_vector_is_not_qualified(self):
        (self.root / 'US/out0first').write_bytes(b'tampered complete vector')
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_current_source_mutation_is_not_bypassed_by_review(self):
        self.disposition()
        (self.f.source / 'easygraph/gpu/session.py').write_text('new source')
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_required_cpu_reference_failure_still_blocks(self):
        self.f.write(self.f.state, {'status': 'WAITING_CPU_REFERENCE_COHORT'})
        report = self.run_inspect()
        self.assert_unresolved(report)
        self.assertEqual(report['cells'][0]['status'], 'QUALIFIED')
        self.assertEqual(report['global_blockers'][0]['stage'], 'unchanged_parent_cpu_gate')

    def test_index_classification_requires_complete_bound_geometry(self):
        rec = self.disposition('INDEX')
        rec['facts'] = dict(required_entries=2147483648, supported_max_entries=2147483647,
                            index_bits=32, index_domain='global_offsets')
        rec['evidence']['index_geometry'] = self.f.save('geometry.json', dict(
            input_sha256=rec['input_sha256'], all_arcs_retained=False, **rec['facts']))
        self.save_disposition(rec)
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_timeout_requires_frozen_deadline_and_owned_cleanup(self):
        rec = self.disposition('TO')
        rec['facts'] = dict(frozen_timeout_seconds=1, observed_elapsed_seconds=2,
                            owned_processes_terminated=True)
        self.save_disposition(rec)
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_explicit_timeout_index_and_platform_proofs_keep_terminal_slots(self):
        cases = ['TO', 'INDEX', 'PLATFORM']
        for kind in cases:
            rec = self.disposition(kind)
            if kind == 'TO':
                self.replace_marker(rec, 'TIMEOUT at frozen controller deadline')
                rec['facts'] = dict(frozen_timeout_seconds=1800, observed_elapsed_seconds=1801,
                                    owned_processes_terminated=True)
            elif kind == 'INDEX':
                self.replace_marker(rec, 'INDEX_UNSUPPORTED global_offsets')
                rec['facts'] = dict(required_entries=2147483648, supported_max_entries=2147483647,
                                    index_bits=32, index_domain='global_offsets')
                rec['evidence']['index_geometry'] = self.f.save('geometry.json', dict(
                    input_sha256=rec['input_sha256'], all_arcs_retained=True, **rec['facts']))
            else:
                self.replace_marker(rec, 'CUDA 217: peer access is not supported')
                rec['facts'] = dict(required_feature='peer_access', native_error_code=217,
                                    author_feature_enabled=True)
            self.save_disposition(rec)
            report = self.run_inspect()
            self.assertEqual(report['counts'], {kind: 1})
            self.assertTrue(report['all_cells_resolved_and_proofs_checked'])
            self.assertEqual(len(report['cells'][0]['formal_slots']), 10)
            self.assertFalse(report['formal_admitted'])

    def test_oom_log_cannot_be_relabelled_index_with_valid_geometry(self):
        rec = self.disposition('INDEX')
        rec['facts'] = dict(required_entries=2147483648, supported_max_entries=2147483647,
                            index_bits=32, index_domain='global_offsets')
        rec['evidence']['index_geometry'] = self.f.save('geometry.json', dict(
            input_sha256=rec['input_sha256'], all_arcs_retained=True, **rec['facts']))
        self.save_disposition(rec)
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_platform_requires_actual_author_peer_access_feature(self):
        rec = self.disposition('PLATFORM')
        rec['facts'] = dict(required_feature='peer_access', native_error_code=217,
                            author_feature_enabled=False)
        self.save_disposition(rec)
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_resource_identity_and_preflight_cannot_be_waived(self):
        rec = self.disposition()
        p = Path(rec['evidence']['resource_observation']['path'])
        d = A.load(p); d['disk_preflight_passed'] = False; self.f.write(p, d)
        rec['evidence']['resource_observation'] = self.f.b.bind(p); self.save_disposition(rec)
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_resource_description_is_not_a_raw_file_binding(self):
        rec = self.disposition()
        p = Path(rec['evidence']['resource_observation']['path'])
        d = A.load(p); d['evidence'] = {'unbound': 'looks fine'}; self.f.write(p, d)
        rec['evidence']['resource_observation'] = self.f.b.bind(p); self.save_disposition(rec)
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_error_marker_must_exist_in_bound_raw_log(self):
        rec = self.disposition(); rec['error_marker']['text'] = 'missing cudaErrorMemoryAllocation'
        self.save_disposition(rec)
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_old_tiny_epoch_blocks_target_even_if_success_shape_valid(self):
        target = self.f.qual['jobs'][-1]['receipt']
        p = A.load(target); p['epoch_sha256'] = 'old'; self.f.write(target, p)
        self.assert_unresolved(self.run_inspect(), 'INFRA')

    def test_e4s_passing_vectors_without_recoverable_sidecar_are_rejected(self):
        f = self.f
        ep = A.load(f.epoch['path']); ep['schema'] = 'e4s.epoch.v1'
        f.epoch = f.save('epoch.json', ep); f.edigest = A.digest(ep)
        for job in f.qual['jobs']:
            p = A.load(job['receipt'])
            p.update(status='PASS_E4S_QUALIFICATION', variant='eggpu', batch=2,
                     epoch=f.epoch, epoch_sha256=f.edigest, source_sequence=[0, 1],
                     checks=[dict(sequence=i, source=i, values=16,
                                  quality=dict(passed=True, exact=True, vertices=16),
                                  output_sha256='a'*64, survives_session_close=True) for i in range(2)],
                     measurements=dict(first_complete_seconds=3, adapter_preparation_seconds=1,
                                       query_batch_seconds=2, all_outputs_retained_through_close=True,
                                       requests=[dict(source=i, sequence=i) for i in range(2)]))
            f.write(job['receipt'], p)
        for job in f.qual['jobs'] + f.formal['jobs']:
            job['command'][1] = f.manifest['proof_readers']['e4s']['path']
            job['command'] += ['--variant', 'eggpu', '--batch', '2']
            job['expected_statuses'] = ['PASS_E4S_QUALIFICATION']
        target = A.load(f.qual['jobs'][-1]['receipt'])
        target['tiny_proof'] = f.b.bind(f.qual['jobs'][0]['receipt'])
        f.write(f.qual['jobs'][-1]['receipt'], target)
        for job in f.formal['jobs']:
            job['target_contract'] = dict(source_sequence=[0, 1], input_sha256=target['input_sha256'])
        f.manifest['policy'] = {'e4s_formal_archive_repair_pending': False}
        self.request['group'] = 'e4s'; self.sync_plans('e4s')
        report = self.run_inspect()
        self.assert_unresolved(report, 'INFRA')
        self.assertIn('sidecar is missing', report['cells'][0]['reason'])

    def test_cli_has_no_admit_or_execute_option(self):
        with mock.patch.object(sys, 'argv', ['inspect_ledger.py', '--admit']):
            with self.assertRaises(SystemExit):
                m.main()

    @unittest.skipUnless(os.environ.get('EGGPU_PRIVATE_FROZEN_PLAN_ROOT'), 'Original private deployment plans are not included in the public tool package')
    def test_actual_frozen_four_group_cardinality_is_preserved(self):
        campaign = Path(os.environ['EGGPU_PRIVATE_FROZEN_PLAN_ROOT'])
        for group, expected in [('sg', 56), ('native', 88), ('sequence', 6), ('e4s', 36)]:
            root = campaign / 'unified_release_d84_v2'
            formal = A.load(root / (group + '_formal/pending.json'))
            plans = [A.load(root / group / 'plan.json')]
            if group == 'native':
                plans.append(A.load(root / 'preflight/plan.json'))
            qualifications, mapping = A.structural_gate(formal, plans)
            counts = {}
            for target in mapping.values():
                counts[target['id']] = counts.get(target['id'], 0) + 1
            self.assertEqual(len(counts), expected)
            self.assertEqual(set(counts.values()), {10})
            self.assertEqual(len(mapping), expected * 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
