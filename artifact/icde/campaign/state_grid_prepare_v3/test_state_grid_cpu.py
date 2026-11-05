from pathlib import Path
from types import SimpleNamespace
import gc,importlib.util,json,os,sys,tempfile,unittest
import numpy as np
import prepare_state_grid as grid

CAMPAIGN=Path(__file__).resolve().parent.parent
SOURCE=Path(os.environ.get('EGGPU_TEST_SOURCE',CAMPAIGN.parents[2]/'Easy-Graph'))
FREEZE=Path(os.environ.get('EGGPU_TEST_FREEZE',CAMPAIGN.parent/'provenance/runtime_source_freeze.json'))


class GridTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.canonical=self.root/'canonical'
        for name,n in [('A',31),('B',37)]:
            d=self.canonical/name;d.mkdir(parents=True)
            pairs={(u,(u+1)%(n-3)) for u in range(n-3)}
            arcs=sorted(pairs|{(v,u) for u,v in pairs}) if name=='A' else sorted(pairs);off=np.r_[0,np.cumsum(np.bincount([u for u,v in arcs],minlength=n))].astype('<i4');ind=np.array([v for u,v in arcs],dtype='<i4');weights=np.ones(len(arcs),dtype='<f8')
            for file,arr in [('offsets.bin',off),('indices.bin',ind),('weights.bin',weights)]:arr.tofile(d/file)
            grid.save(d/'canonical.json',dict(num_nodes=n,num_entries=len(arcs),num_edges=len(arcs)//2 if name=='A' else len(arcs),directed=name!='A'))
            grid.save(d/'sources.json',dict(primary_sources=[0,1,2]))
        self.cells=[dict(id=f'A_p{p}_{rho}_{loc}',experiment='E3',datasets=['A'],partitions=p,rho=rho,locality=loc) for p in [1,8] for rho,loc in [(0.,'uniform'),(.2,'uniform'),(.2,'clustered'),(1.,'uniform')]]
        self.cells.append(dict(id='AB',experiment='E4',datasets=['A','B'],partitions=1,rho=.2,locality='uniform'))
        grid.save(self.root/'spec.json',dict(cells=self.cells))
        self.args=SimpleNamespace(source=SOURCE,source_freeze=FREEZE,state_code=Path(os.environ.get('EGGPU_TEST_STATE_CODE',CAMPAIGN)),canonical=self.canonical,output=self.root/'grid',plan=self.root/'plan.json',spec=self.root/'spec.json',reserve_bytes=0,qualification_only=True,execute=True,legacy_controller=CAMPAIGN/'state_grid_prepare_v2/prepare_state_grid.py',position_selector=Path(__file__).parent/'positions.py',run_receipt=self.root/'run.json')
    def tearDown(self):
        gc.collect();self.temp.cleanup()
    def test_shared_bases_positions_and_complete_worker_contract(self):
        r=grid.execute(self.args)
        self.assertEqual(r['status'],'PASS_CPU_SHARED_GRID_PREPARED_GPU_UNQUALIFIED')
        self.assertEqual(len(r['new_bases']),3)
        self.assertEqual(len(r['new_positions']),5)
        import state_inputs,state_worker
        for cell in self.cells:
            d=state_worker.load_inputs(self.args.output/'cells'/cell['id']/'inputs.json')
            for index,name in enumerate(cell['datasets']):
                row=d['datasets'][name];expected=state_inputs.positions(self.canonical/name,cell['rho'],cell['locality'],20261104+index)
                actual=np.load(row['positions']['path'],allow_pickle=False)
                np.testing.assert_array_equal(actual,expected)
                self.assertEqual(row['position_seed'],20261104+index)
        one=grid.load(self.args.output/'cells/A_p1_0.2_uniform/inputs.json')['datasets']['A']
        two=grid.load(self.args.output/'cells/A_p1_0.2_clustered/inputs.json')['datasets']['A']
        eight=grid.load(self.args.output/'cells/A_p8_0.2_uniform/inputs.json')['datasets']['A']
        self.assertEqual(one['manifest'],two['manifest'])
        self.assertEqual(one['positions'],eight['positions'])
        for spec in one['manifest_files']:self.assertFalse(Path(spec['path']).stat().st_mode&0o222)
    def test_reexecution_does_not_rewrite_arrays(self):
        first=grid.execute(self.args)
        files={p:(p.stat().st_ino,p.stat().st_mtime_ns,grid.sha(p)) for p in self.args.output.rglob('*.bin')}
        self.args.plan=self.root/'second_plan.json';self.args.run_receipt=self.root/'second_run.json';second=grid.execute(self.args)
        self.assertFalse(second['new_bases']);self.assertFalse(second['new_positions'])
        for p,before in files.items():self.assertEqual(before,(p.stat().st_ino,p.stat().st_mtime_ns,grid.sha(p)))
    def test_plan_only_does_not_create_payload(self):
        self.args.execute=False;p=grid.execute(self.args)
        self.assertEqual(p['status'],'PLAN_ONLY');self.assertFalse(self.args.output.exists());self.assertEqual(len(p['bases']),3)
    def test_disk_reserve_prevents_partial_allocation(self):
        self.args.reserve_bytes=1<<70
        with self.assertRaisesRegex(RuntimeError,'disk'):grid.execute(self.args)
        self.assertFalse(self.args.output.exists())
    def test_tampered_base_cannot_be_reused(self):
        result=grid.execute(self.args);doc=grid.load(result['new_bases'][0]['path']);f=Path(doc['owned_files'][0]['path']);f.chmod(0o644);f.write_bytes(f.read_bytes()+b'x')
        self.args.plan=self.root/'tamper_plan.json';self.args.run_receipt=self.root/'tamper_run.json'
        with self.assertRaisesRegex(ValueError,'Immutable file changed'):grid.execute(self.args)
    def test_keys_ignore_source_location_but_bind_preparation_bytes(self):
        code,impl=grid.code_identity(self.args.state_code,SOURCE,FREEZE);entry=grid.inspect_input(self.canonical,'A',True)
        original=grid.base_key('A',1,entry,code)
        code['source_signer']['path']='/new/source/signer.py';self.assertEqual(original,grid.base_key('A',1,entry,code))
        code['source_signer']['sha256']='0'*64;self.assertNotEqual(original,grid.base_key('A',1,entry,code))
    def test_initial_pool_order_and_seed_are_explicit(self):
        rows=grid.validate_cells(grid.initial_cells());self.assertEqual(len(rows),9)
        for row in rows:
            if row['experiment'] in ['E4','E5']:self.assertEqual(tuple(row['datasets']),grid.POOLS[row['experiment']])
        bad=dict(rows[0],id='../escape')
        with self.assertRaises(ValueError):grid.validate_cells([bad])
    def test_reduced_reserve_requires_complete_tw_and_margin(self):
        self.args.qualification_only=False;self.args.reserve_bytes=2<<30
        with self.assertRaisesRegex(ValueError,'TW receipt'):grid.reserve_evidence(self.args)
        self.args.reserve_bytes=0
        with self.assertRaisesRegex(ValueError,'2 GiB'):grid.reserve_evidence(self.args)
        self.args.reserve_bytes=29<<30
        self.assertEqual(grid.reserve_evidence(self.args)['mode'],'TW_pending_reserve')
    def test_completed_tw_payload_is_fully_verified(self):
        import shutil
        d=self.canonical/'TW';shutil.copytree(self.canonical/'B',d)
        meta=grid.load(d/'canonical.json');meta['weight_rule']='w16-v1';grid.save(d/'canonical.json',meta)
        files={('canonical' if role=='metadata' else role):grid.bind(d/file) for role,file in grid.ROLES.items()}
        grid.save(d/'receipt.json',dict(status='PASS_CANONICAL_INPUT',dataset='TW',files=files))
        self.args.qualification_only=False;self.args.reserve_bytes=2<<30;self.args.completed_tw_receipt=d/'receipt.json'
        self.assertEqual(grid.reserve_evidence(self.args)['mode'],'TW_complete_full_payload_verified')
        (d/'weights.bin').write_bytes((d/'weights.bin').read_bytes()+b'x')
        with self.assertRaises(ValueError):grid.reserve_evidence(self.args)

    def test_undirected_symmetric_positions_preserved(self):
        _,impl=grid.code_identity(self.args.state_code,SOURCE,FREEZE);d=self.canonical/'A';off=np.fromfile(d/'offsets.bin',dtype='<i4');ind=np.fromfile(d/'indices.bin',dtype='<i4')
        for loc in ['uniform','clustered']:
            pos=set(impl.positions(d,.3,loc,20261104).tolist())
            for k in pos:
                u=int(np.searchsorted(off,k,side='right')-1);v=int(ind[k]);reverse=int(off[v])+int(np.searchsorted(ind[off[v]:off[v+1]],u));self.assertIn(reverse,pos)

    def test_old_bases_reused_without_rewriting_or_changing_prior_metadata(self):
        spec=importlib.util.spec_from_file_location('_legacy_grid_test',self.args.legacy_controller);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        old=legacy.execute(self.args);old_receipt=grid.bind(self.args.output/'receipt.json')
        files={p:(p.stat().st_ino,p.stat().st_mtime_ns,grid.sha(p)) for p in self.args.output.rglob('*.bin')}
        old_cells={p:grid.bind(p) for p in (self.args.output/'cells').rglob('inputs.json')}
        for c in self.cells:c['id']+='_typed'
        grid.save(self.root/'typed_spec.json',dict(cells=self.cells));self.args.spec=self.root/'typed_spec.json'
        self.args.plan=self.root/'typed_plan.json';self.args.run_receipt=self.root/'typed_run.json'
        result=grid.execute(self.args)
        self.assertFalse(result['new_bases']);self.assertEqual(len(result['reused_bases']),3)
        self.assertEqual(old_receipt,grid.bind(self.args.output/'receipt.json'))
        for p,before in old_cells.items():self.assertEqual(before,grid.bind(p))
        for p,before in files.items():self.assertEqual(before,(p.stat().st_ino,p.stat().st_mtime_ns,grid.sha(p)))
        import state_worker
        for cell in self.cells:state_worker.load_inputs(self.args.output/'cells'/cell['id']/'inputs.json')
    def test_selector_tampering_denied_before_payload(self):
        changed=self.root/'changed_selector.py';changed.write_text(self.args.position_selector.read_text()+'\n# changed\n');self.args.position_selector=changed
        with self.assertRaisesRegex(ValueError,'typed position selector'):grid.execute(self.args)
        self.assertFalse(self.args.output.exists())
    def test_existing_run_receipt_is_never_overwritten(self):
        grid.save(self.args.run_receipt,{'status':'retained'})
        with self.assertRaisesRegex(FileExistsError,'unused explicit run receipt'):grid.execute(self.args)
        self.assertFalse(self.args.output.exists())

if __name__=='__main__':unittest.main()

