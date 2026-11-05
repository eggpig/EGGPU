from pathlib import Path
import copy
import importlib.util
import json
import sys
import unittest

D=Path(__file__).resolve().parent
def module(name):
    s=importlib.util.spec_from_file_location(name,D/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
v8=module('run_campaign_queue_v8');base=module('test_campaign_queue_v7_cpu')


class MigrationTests(unittest.TestCase):
    setUp=base.ResumeTests.setUp
    tearDown=base.ResumeTests.tearDown
    fixture=base.ResumeTests.fixture

    def setup_migration(self):
        old,state=self.fixture(maintenance=True)
        d=json.loads(old.read_text());new=copy.deepcopy(d)
        maint=self.p/'new_cpu_maintenance.py'
        maint.write_text("from pathlib import Path\nimport json,sys\np=Path(sys.argv[1]);p.mkdir();(p/'receipt.json').write_text(json.dumps({'status':'PASS_NEW_MAINT'}))\n")
        spec=new['jobs'][1]['maintenance'];out=self.p/'new_maint1';cmd=[sys.executable,'-B',str(maint),str(out)]
        spec.update(command=cmd,command_sha256=v8.command_digest(cmd),bindings=[v8.bind(Path(sys.executable).resolve()),v8.bind(maint)],receipt=str(out/'receipt.json'),expected_statuses=['PASS_NEW_MAINT'])
        plan=self.p/'migration_plan.json';plan.write_text(json.dumps(new))
        return old,state,plan
    def run_migration(self,old,state,plan):
        return v8.run(plan,self.p/'migrated',self.lock,state,v8.bind(state)['sha256'],v8.bind(old)['sha256'])
    def alter(self,plan,fn):
        d=json.loads(plan.read_text());fn(d);plan.write_text(json.dumps(d))
    def test_real_maintenance_only_resume_preserves_worker_and_prefix(self):
        old,state,plan=self.setup_migration();parent=state.read_bytes();before=json.loads(parent)['completed'];result=self.run_migration(old,state,plan)
        self.assertEqual(self.counter.read_text().splitlines(),['j0','j1']);self.assertEqual(result['completed'][:1],before);self.assertEqual(state.read_bytes(),parent)
        self.assertTrue((self.p/'new_maint1/receipt.json').exists());self.assertFalse((self.p/'maint1').exists())
        ad=json.loads((self.p/'migrated/resume_admission.json').read_text())['maintenance_migration']
        self.assertEqual(ad['old_plan'],v8.bind(old));self.assertEqual(ad['new_plan'],v8.bind(plan));self.assertEqual(len(ad['changed_jobs']),1)
    def test_completed_job_maintenance_cannot_change(self):
        old,state,plan=self.setup_migration();self.alter(plan,lambda d:d['jobs'][0]['maintenance'].update(timeout=16))
        with self.assertRaisesRegex(ValueError,'completed job'):self.run_migration(old,state,plan)
    def test_worker_argv_cannot_change(self):
        old,state,plan=self.setup_migration();self.alter(plan,lambda d:d['jobs'][1]['command'].append('--changed'))
        with self.assertRaisesRegex(ValueError,'worker contract'):self.run_migration(old,state,plan)
    def test_worker_environment_cannot_change(self):
        old,state,plan=self.setup_migration();self.alter(plan,lambda d:d['jobs'][1].update(environment={'CHANGED':'1'}))
        with self.assertRaisesRegex(ValueError,'worker contract'):self.run_migration(old,state,plan)
    def test_top_environment_cannot_change(self):
        old,state,plan=self.setup_migration();self.alter(plan,lambda d:d.update(environment={'CHANGED':'1'}))
        with self.assertRaisesRegex(ValueError,'plan metadata'):self.run_migration(old,state,plan)
    def test_numerically_equal_different_type_worker_field_rejected(self):
        old,state,plan=self.setup_migration();self.alter(plan,lambda d:d['jobs'][1].update(timeout=15.0))
        with self.assertRaisesRegex(ValueError,'worker contract'):self.run_migration(old,state,plan)
    def test_old_plan_sha_must_be_explicit_and_exact(self):
        old,state,plan=self.setup_migration()
        with self.assertRaisesRegex(ValueError,'old plan SHA'):v8.run(plan,self.p/'migrated',self.lock,state,v8.bind(state)['sha256'],'0'*64)
    def test_new_helper_file_change_rejected(self):
        old,state,plan=self.setup_migration();(self.p/'new_cpu_maintenance.py').write_text('raise SystemExit(0)\n')
        with self.assertRaisesRegex(ValueError,'binding changed'):self.run_migration(old,state,plan)
    def test_maintenance_cannot_be_removed(self):
        old,state,plan=self.setup_migration();self.alter(plan,lambda d:d['jobs'][1].pop('maintenance'))
        with self.assertRaisesRegex(ValueError,'add or remove'):self.run_migration(old,state,plan)
    def test_global_gpu_lock_cannot_change(self):
        old,state,plan=self.setup_migration();self.alter(plan,lambda d:d['jobs'][1]['maintenance'].update(gpu_lock=str(self.p/'other.lock')))
        with self.assertRaisesRegex(ValueError,'global GPU lock'):self.run_migration(old,state,plan)
    def test_migration_not_implicit(self):
        old,state,plan=self.setup_migration()
        with self.assertRaisesRegex(ValueError,'unchanged complete original plan'):v8.run(plan,self.p/'migrated',self.lock,state,v8.bind(state)['sha256'])
    def test_migration_without_checkpoint_rejected(self):
        old,state,plan=self.setup_migration()
        with self.assertRaisesRegex(ValueError,'resume state and SHA'):v8.run(plan,self.p/'migrated',self.lock,maintenance_migration_from=v8.bind(old)['sha256'])


if __name__=='__main__':unittest.main(verbosity=2)
