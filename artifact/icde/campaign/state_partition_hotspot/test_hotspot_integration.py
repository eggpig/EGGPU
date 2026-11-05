from pathlib import Path
import copy,importlib.util,json
import numpy as np
import pytest
import prepare_hotspot_inputs as h


def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x));return h.selector.binding(p)


def fixture(p):
 n=512;indices=np.concatenate([np.r_[np.arange(u,dtype=np.int32),np.arange(u+1,n,dtype=np.int32)] for u in range(n)]);m=len(indices);offsets=np.arange(n+1,dtype=np.int32)*(n-1)
 canonical={}
 for role,data in [('offsets',offsets),('indices',indices),('weights',np.ones(m,dtype='<f8'))]:
  f=p/(role+'.bin');data.tofile(f);canonical[role]=h.selector.binding(f)
 canonical['metadata']=write(p/'canonical.json',dict(num_nodes=n,num_entries=m,directed=False));canonical['sources']=write(p/'sources.json',{'primary_sources':[0,1,2]})
 signer={'path':'/old/signer.py','sha256':'a'*64,'bytes':7};freeze=write(p/'freeze.json',{'source_files':{'easygraph/gpu/_manifest_signatures.py':{k:signer[k] for k in ('bytes','sha256')}}})
 cells=[]
 for name in ['IN','OR']:
  base=write(p/(name+'_base.json'),{'status':'PASS_IMMUTABLE_STATE_BASE','owned_files':list(canonical.values())})
  row=dict(n=n,m=m,canonical=canonical,shared_base=base,edge_cuts=offsets[np.linspace(0,n,9,dtype=np.int64)].tolist(),requested_rho=1e-4)
  cells.append(write(p/(name+'_input.json'),{'schema':'state.inputs.v2','status':'PASS_CPU_INPUTS_GPU_UNQUALIFIED','qualification_only':False,'source_signer':signer,'partitions':8,'cell':dict(experiment='E3',partitions=8,rho=1e-4,locality='uniform',datasets=[name]),'datasets':{name:row}}))
 other=write(p/'other.json',{'cell':{'experiment':'E4'}})
 write(p/'state_grid_full_execute_expansion200_v1/receipt.json',{'status':'PASS_CPU_SHARED_GRID_PREPARED_GPU_UNQUALIFIED','cells':cells+[other]*54})
 return freeze,canonical


def planner():
 path=Path(__file__).parents[1]/'state_final_plan/state_plan.py';spec=importlib.util.spec_from_file_location('p7',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_hotspot_eight_inputs_bind_existing_bases(tmp_path):
 freeze,canonical=fixture(tmp_path);before={x['path']:Path(x['path']).stat().st_ino for x in canonical.values()};r=h.prepare(tmp_path,tmp_path/'out',freeze['path']);assert r['base_bytes_copied']==0 and len(r['inputs'])==8 and not r['infeasible']
 for b in r['inputs']:
  doc=h.load(b['path']);planner().validate_hotspot(doc);assert next(iter(doc['datasets'].values()))['canonical']==canonical
 assert before=={x['path']:Path(x['path']).stat().st_ino for x in canonical.values()}


def test_hotspot_signer_change_rejected(tmp_path):
 freeze,_=fixture(tmp_path);write(Path(freeze['path']),{'source_files':{'easygraph/gpu/_manifest_signatures.py':{'sha256':'b'*64,'bytes':7}}})
 with pytest.raises(ValueError,match='signer'):h.prepare(tmp_path,tmp_path/'out',freeze['path'])


def test_hotspot_changed_positions_rejected(tmp_path):
 freeze,_=fixture(tmp_path);r=h.prepare(tmp_path,tmp_path/'out',freeze['path']);doc=h.load(r['inputs'][0]['path']);pos=Path(doc['datasets']['IN']['positions']['path']);pos.chmod(0o644);pos.write_bytes(b'changed')
 with pytest.raises(ValueError,match='changed'):planner().validate_hotspot(doc)


def test_grid_preserves_main_and_adds_exact_32():
 p=planner();assert len(p.grid())==288;rows=p.hotspot_grid();assert len(rows)==32 and len({r['id'] for r in rows})==32
 assert {r['hotspot_k'] for r in rows}=={1,2,4,8};assert {r['rho'] for r in rows}=={1e-4};assert {r['arm'] for r in rows}=={'D0C1','D1C1'};assert {r['old_readers'] for r in rows}=={'none','per_update'}
