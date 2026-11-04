#!/usr/bin/env python3
"""CPU ownership/closure contracts for actual public controller code; no CUDA.

Only unused import dependencies are stubbed. The controller, receipt merge and
inventory validation execute their real public-source implementations.
"""
from pathlib import Path
import argparse, copy, hashlib, importlib.util, json, sys, types, unittest

PUBLIC=Path(__file__).resolve().parents[3]/'Easy-Graph'


def load_controller(source):
    for name in ('easygraph','easygraph.gpu'):
        module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
    # These symbols are used only by the untouched GPU session classes, which
    # these CPU tests never instantiate or call.
    sys.modules.setdefault('numpy',types.ModuleType('numpy'))
    bridge=types.ModuleType('easygraph.gpu._bridge');bridge.NativePartitionSession=type('NativePartitionSession',(),{})
    bridge.source_parameter=lambda p:p['source'];sys.modules[bridge.__name__]=bridge
    contracts=types.ModuleType('easygraph.gpu.contracts');contracts.QueryContract=type('QueryContract',(),{})
    sys.modules[contracts.__name__]=contracts
    for name in ('_retention','_receipts','_controller'):
        spec=importlib.util.spec_from_file_location('easygraph.gpu.'+name,source/'easygraph/gpu'/f'{name}.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def owner(name,graph,kind='view',parent='',lease=0,device=0):
    return dict(owner_id=name,kind=kind,graph_id=graph,algorithm='sssp',parent_owner_id=parent,
                active_leases=lease,device_id=device,generation=0,partition_id=device,policy='per_worker')


def allocation(name,size,holders,device=0,catalog=True):
    noncatalog=[x for x in holders if x!='catalog']
    return dict(allocation_id=name,bytes=size,owners=list(holders),device_id=device,
                catalog_retained=catalog,catalog_external_references=len(noncatalog) if catalog else 0,
                inventoried_shared_references=len(noncatalog),untracked_shared_references=0,
                catalog_only_idle=catalog and not noncatalog)


def inventory(owners,allocations,device=0):
    row=dict(device_id=device,owners=owners,allocations=allocations,
             managed_resident_bytes=sum(a['bytes'] for a in allocations),counters={})
    return dict(per_device=[row],devices=[device],managed_resident_bytes=row['managed_resident_bytes'],
                has_active_leases=any(o['active_leases'] for o in owners),has_untracked_shared_references=False)


class MemoryOracle:
    """Independent tiny refcount model supplies exact native-style receipts."""
    def __init__(self,state,corrupt=False):self.state=copy.deepcopy(state);self.corrupt=corrupt;self.calls=[]
    def cpp_egmgpu_managed_state_inventory(self,devices):return copy.deepcopy(self.state)
    def cpp_egmgpu_managed_evict(self,owner_ids,devices,release_idle,release_workspaces):
        before=self.state['managed_resident_bytes'];self.calls.append(tuple(owner_ids))
        row=self.state['per_device'][0];doomed=set(owner_ids)
        doomed.update(o['owner_id'] for o in row['owners'] if o['parent_owner_id'] in doomed)
        if any(o['active_leases'] and o['owner_id'] in doomed for o in row['owners']):raise RuntimeError('Oracle rejects active lease')
        row['owners']=[o for o in row['owners'] if o['owner_id'] not in doomed]
        kept=[]
        for a in row['allocations']:
            a['owners']=[o for o in a['owners'] if o not in doomed]
            users=[o for o in a['owners'] if o!='catalog']
            a['inventoried_shared_references']=len(users)
            a['catalog_external_references']=len(users) if a['catalog_retained'] else 0
            a['catalog_only_idle']=a['catalog_retained'] and not users
            if not users and (not a['catalog_retained'] or release_idle):continue
            kept.append(a)
        row['allocations']=kept;row['managed_resident_bytes']=sum(a['bytes'] for a in kept)
        self.state['managed_resident_bytes']=row['managed_resident_bytes']
        after=row['managed_resident_bytes']
        return dict(before_bytes=before,after_bytes=after,released_bytes=before-after+(1 if self.corrupt else 0),after_inventory=copy.deepcopy(self.state))


class ClosureContracts(unittest.TestCase):
    def regular(self):
        return inventory([owner('A','A'),owner('B','B'),owner('catalog','','catalog')],
                         [allocation('a',80,['A','catalog']),allocation('b',80,['B','catalog'])])
    def controllers(self,state,budget=96):
        answer=[]
        for policy in ('closure_lru','closure'):
            oracle=MemoryOracle(state);ctl=M.StateSwitchController(oracle,[0],{0:budget},policy=policy,inventory_mode='recheck',batch_devices=True)
            ctl.clock=3;ctl.last_use={'A':1,'B':2}
            ctl.history.extend([('A','sssp','{}','unbound')]*20+[('B','sssp','{}','unbound')])
            answer.append((ctl,oracle))
        return answer
    def test_same_menu_only_ranking_changes(self):
        pairs=self.controllers(self.regular());menus=[];results=[]
        for ctl,oracle in pairs:
            result=ctl.finish_query('Z','sssp');results.append(result)
            menus.append({tuple(c['closure_owner_ids']):{k:v for k,v in c.items() if k not in ('score','score_kind')}
                          for c in result['decisions'][0]['candidates']})
        self.assertEqual(menus[0],menus[1])
        self.assertEqual(pairs[0][1].calls,[('A',)])
        self.assertEqual(pairs[1][1].calls,[('B',)])
        for result in results:
            self.assertEqual(result['actions'][0]['prediction_error_bytes'],0)
            self.assertEqual(result['after_retention']['managed_resident_bytes'],80)
            self.assertEqual(result['control_counts']['recheck_exports'],1)
    def test_zero_marginal_shared_allocation_requires_joint_release(self):
        state=inventory([owner('A','A'),owner('B','B'),owner('catalog','','catalog')],
                        [allocation('shared',128,['A','B','catalog'])])
        for ctl,oracle in self.controllers(state,budget=64):
            result=ctl.finish_query('Z','sssp')
            self.assertEqual(oracle.calls,[('A','B')]);self.assertEqual(result['after_retention']['managed_resident_bytes'],0)
    def test_parent_workspace_cascade_and_no_double_count(self):
        state=inventory([owner('A','A'),owner('W','A','workspace','A'),owner('catalog','','catalog')],
                        [allocation('shared',80,['A','W','catalog']),allocation('scratch',16,['W'],catalog=False)])
        prediction=M.predict_release(state['per_device'][0],['A'])
        self.assertEqual(prediction['closure_owner_ids'],['A','W'])
        self.assertEqual(prediction['predicted_freed_bytes'],96)
        for ctl,oracle in self.controllers(state,budget=8):
            result=ctl.finish_query('Z','sssp')
            self.assertIn(('A',),oracle.calls)
            self.assertEqual(sum(a['freed_bytes'] for a in result['actions']),96)
            self.assertEqual(result['after_retention']['managed_resident_bytes'],0)
    def test_active_child_lease_blocks_parent_and_all_legal_fallback(self):
        state=inventory([owner('A','A'),owner('W','A','workspace','A',lease=1),owner('B','B'),owner('catalog','','catalog')],
                        [allocation('a',80,['A','W','catalog']),allocation('b',80,['B','catalog'])])
        for ctl,oracle in self.controllers(state,budget=96):
            result=ctl.finish_query('Z','sssp');self.assertEqual(oracle.calls,[('B',)])
            self.assertEqual({o['owner_id'] for o in result['after_retention']['per_device'][0]['owners']},{'A','W','catalog'})
    def test_no_legal_release_fails_closed(self):
        state=inventory([owner('A','A',lease=1),owner('catalog','','catalog')],[allocation('a',80,['A','catalog'])])
        for ctl,oracle in self.controllers(state,budget=16):
            with self.assertRaisesRegex(RuntimeError,'No legal owner release'):ctl.finish_query('Z','sssp')
    def test_corrupt_native_receipt_rejected(self):
        for policy in ('closure_lru','closure'):
            ctl=M.StateSwitchController(MemoryOracle(self.regular(),corrupt=True),[0],{0:96},policy=policy)
            with self.assertRaisesRegex(RuntimeError,'freed bytes disagree'):ctl.finish_query('Z','sssp')
    def test_untracked_sharing_rejected(self):
        state=self.regular();state['per_device'][0]['allocations'][0]['untracked_shared_references']=1
        for ctl,oracle in self.controllers(state):
            with self.assertRaisesRegex(RuntimeError,'Untracked'):ctl.finish_query('Z','sssp')
    def test_idle_catalog_trim_is_identical(self):
        state=inventory([owner('A','A'),owner('catalog','','catalog')],
                        [allocation('a',80,['A','catalog']),allocation('idle',32,['catalog'])])
        for ctl,oracle in self.controllers(state,budget=96):
            r=ctl.finish_query('Z','sssp');self.assertEqual(oracle.calls,[()]);self.assertEqual(r['actions'][0]['reason'],'idle_catalog')
    def test_owner_order_permutation_does_not_change_menu_or_choice(self):
        state=self.regular();other=copy.deepcopy(state);other['per_device'][0]['owners'].reverse();other['per_device'][0]['allocations'].reverse()
        for (left,_),(right,_) in zip(self.controllers(state),self.controllers(other)):
            a=left.finish_query('Z','sssp');b=right.finish_query('Z','sssp')
            self.assertEqual(a['decisions'],b['decisions'])
    def test_explicit_policy_registration_and_legacy_identity(self):
        self.assertIn('closure_lru',M.POLICIES);self.assertIn('lru',M.POLICIES)
        ctl=M.StateSwitchController(MemoryOracle(self.regular()),[0],None,policy='lru')
        self.assertEqual(ctl.finish_query('Z','sssp')['history_stats']['cost_model'],'frozen_v2_owner_recency')


def main():
    global M
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=PUBLIC);p.add_argument('--output',type=Path);a=p.parse_args()
    M=load_controller(a.source);suite=unittest.defaultTestLoader.loadTestsFromTestCase(ClosureContracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(schema='state.strong_controls.cpu/v1',status='PASS_CPU_ONLY' if result.wasSuccessful() else 'FAIL',
                 tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),gpu_executed=False,
                 modified_source={'path':str(a.source/'easygraph/gpu/_controller.py'),'sha256':hashlib.sha256((a.source/'easygraph/gpu/_controller.py').read_bytes()).hexdigest()},
                 files=[{'path':'easygraph/gpu/_controller.py','sha256':hashlib.sha256((a.source/'easygraph/gpu/_controller.py').read_bytes()).hexdigest()}],
                 note='Actual public closure-LRU and C-Retain controller with independent CPU refcount receipts; no GPU, native DSO or formal performance qualification.',
                 native_gpu_qualification='PENDING',formal_performance_eligible=False)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(receipt,indent=2)+'\n')
    return not result.wasSuccessful()


if __name__=='__main__':raise SystemExit(main())
