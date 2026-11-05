from pathlib import Path
import fcntl
import subprocess
import sys
import hashlib
import importlib.util
import json
import os
import stat
import tempfile
import unittest
from unittest.mock import patch

D=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('incremental',D/'incremental_completed_output_v1.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
ROOT=Path(os.environ.get('EGGPU_INCREMENTAL_TEST_SCRATCH',str(D.parents[2]/'EGGPU_TMP/EGGPU_ICDE_20261017/artifact_recovery_20261105_v5/incremental_cpu')))
ROOT.mkdir(parents=True,exist_ok=True)


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT);self.p=Path(self.temp.name)
        self.index=self.p/'index.json';self.arrays={};self.sequence=0
    def tearDown(self):
        self.temp.cleanup()
    def array(self,name,data=b'0123456789',mode=0o644):
        p=self.p/name;p.write_bytes(data);p.chmod(mode)
        self.arrays[p]=(hashlib.sha256(data).hexdigest(),len(data));return p
    def run_once(self,apply=True):
        self.sequence+=1
        return m.deduplicate(self.arrays,self.index,self.p/('run'+str(self.sequence)),apply)
    def test_full_sha_once_per_inode_and_reuse_only_readonly(self):
        a=self.array('a.npy');b=self.p/'b.npy';os.link(a,b);self.arrays[b]=self.arrays[a]
        r=self.run_once();self.assertEqual(r['unique_inodes_hashed'],1);self.assertEqual(r['bytes_hashed'],10)
        self.assertEqual(a.stat().st_mode & 0o222,0)
        with patch.object(m,'sha',wraps=m.sha) as digest:
            r=self.run_once()
            self.assertEqual(r['unique_inodes_hashed'],0);self.assertEqual(r['cached_paths'],2)
            self.assertFalse(any(call.args[0] in [a,b] for call in digest.call_args_list))
    def test_equal_new_outputs_share_inode_preserving_every_path(self):
        a=self.array('a.npy');b=self.array('b.npy');r=self.run_once()
        self.assertEqual(r['paths_replaced'],1);self.assertEqual(a.stat().st_ino,b.stat().st_ino)
        self.assertEqual(a.read_bytes(),b.read_bytes());self.assertTrue(a.exists() and b.exists())
        self.assertEqual(self.run_once()['cached_paths'],2)
    def test_multiple_alias_groups_are_replaced_safely(self):
        a=self.array('a.npy');b=self.p/'b.npy';os.link(a,b);self.arrays[b]=self.arrays[a]
        c=self.array('c.npy');d=self.p/'d.npy';os.link(c,d);self.arrays[d]=self.arrays[c]
        r=self.run_once();self.assertEqual(r['paths_replaced'],2);self.assertEqual(len({x.stat().st_ino for x in self.arrays}),1)
    def test_content_tamper_with_restored_mtime_is_detected_by_ctime(self):
        a=self.array('a.npy');self.run_once();stamp=a.stat().st_mtime_ns
        a.chmod(0o644);a.write_bytes(b'XXXXXXXXXX');a.chmod(0o444);os.utime(a,ns=(stamp,stamp))
        with self.assertRaisesRegex(ValueError,'changed during full SHA'):self.run_once()
    def test_writable_unchanged_file_is_not_a_cache_hit(self):
        a=self.array('a.npy');self.run_once();a.chmod(0o644)
        r=self.run_once();self.assertEqual(r['unique_inodes_hashed'],1);self.assertEqual(r['cached_paths'],0)
    def test_new_inode_same_bytes_is_fully_hashed_then_deduped(self):
        self.array('a.npy');self.run_once();self.array('b.npy');r=self.run_once()
        self.assertEqual(r['unique_inodes_hashed'],1);self.assertEqual(r['cached_paths'],1);self.assertEqual(r['paths_replaced'],1)
    def test_external_hardlinks_are_not_changed(self):
        a=self.array('a.npy');external=self.p/'external';os.link(a,external);self.array('b.npy')
        r=self.run_once();self.assertEqual(r['paths_replaced'],0);self.assertIn(str(a),r['skipped_external_hardlinks']);self.assertTrue(a.stat().st_mode & 0o200)
    def test_unknown_bin_role_rejected(self):
        self.array('input.bin')
        with self.assertRaisesRegex(ValueError,'Unapproved'):self.run_once()
    def test_symlink_rejected(self):
        a=self.array('a.npy');b=self.p/'b.npy';b.symlink_to(a);self.arrays[b]=self.arrays[a]
        with self.assertRaisesRegex(ValueError,'Symlink'):self.run_once()
    def test_live_writable_descriptor_rejected(self):
        a=self.array('a.npy')
        with a.open('r+b'):
            with self.assertRaisesRegex(ValueError,'Writable descriptor'):self.run_once()
    def test_dry_run_changes_no_arrays_or_index(self):
        a=self.array('a.npy');b=self.array('b.npy');before=[m.fingerprint(x) for x in (a,b)]
        r=self.run_once(False);self.assertEqual(r['paths_replaced'],0);self.assertFalse(self.index.exists());self.assertEqual(before,[m.fingerprint(x) for x in (a,b)])
    def test_changed_binding_on_same_immutable_path_rejected(self):
        a=self.array('a.npy');self.run_once();self.arrays[a]=('0'*64,10)
        with self.assertRaisesRegex(ValueError,'Immutable path'):self.run_once()
    def test_missing_completed_array_is_not_silently_skipped(self):
        receipt=self.p/'result.json';receipt.write_text(json.dumps({'output':{'path':str(self.p/'missing.npy'),'sha256':'0'*64,'bytes':10}}))
        state=self.p/'state.json';state.write_text(json.dumps({'completed':[{'receipt':m.binding(receipt)}]}))
        with self.assertRaisesRegex(ValueError,'missing'):m.completed_arrays([state])
    def test_failed_completed_jobs_are_preserved_and_eligible(self):
        a=self.array('failed.npy');receipt=self.p/'result.json';receipt.write_text(json.dumps({'status':'FAIL','output':m.binding(a)}))
        state=self.p/'state.json';state.write_text(json.dumps({'completed':[{'outcome':'FAIL','receipt':m.binding(receipt)}]}))
        arrays,proofs=m.completed_arrays([state]);self.assertEqual(arrays,self.arrays);self.assertEqual(len(proofs),1)
    def test_child_json_tamper_rejected(self):
        child=self.p/'child.json';child.write_text('{}');receipt=self.p/'result.json';receipt.write_text(json.dumps({'child':m.binding(child)}));child.write_text('{"changed":true}')
        state=self.p/'state.json';state.write_text(json.dumps({'completed':[{'receipt':m.binding(receipt)}]}))
        with self.assertRaisesRegex(ValueError,'child receipt'):m.completed_arrays([state])

    def cli_fixture(self):
        state=self.p/'empty_state.json';state.write_text(json.dumps({'completed':[]}))
        registry=self.p/'registry.json';registry.write_text(json.dumps({'queue_states':[str(state)]}))
        return [sys.executable,'-B',str(D/'incremental_completed_output_v1.py'),'--registry',str(registry),'--index',str(self.index),'--output',str(self.p/'cli_output'),'--controller-lock',str(self.p/'controller.lock'),'--gpu-lock',str(self.p/'gpu.lock'),'--apply']
    def test_cli_inherited_controller_lock_does_not_deadlock(self):
        command=self.cli_fixture()+['--controller-lock-held']
        with (self.p/'controller.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
        self.assertEqual(r.returncode,0,r.stderr.decode())
    def test_cli_false_inherited_lock_assertion_is_rejected(self):
        r=subprocess.run(self.cli_fixture()+['--controller-lock-held'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
        self.assertNotEqual(r.returncode,0);self.assertIn(b'is not held',r.stderr)
    def test_cli_standalone_acquires_both_locks(self):
        r=subprocess.run(self.cli_fixture(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
        self.assertEqual(r.returncode,0,r.stderr.decode());self.assertTrue(self.index.exists())


if __name__=='__main__':unittest.main(verbosity=2)
