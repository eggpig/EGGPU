"""Twenty actual cyclic-weight batches match the frozen state packet recipe."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,tempfile
from types import SimpleNamespace
import numpy as np
import observe_build as observer
import prepare_inputs as inputs
import worker as adapter
import prepare_plan as planner


def main():
    p=argparse.ArgumentParser();p.add_argument('--scratch',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    source=Path(__file__).with_name('grapin_update_sssp.cu');original=source.read_text()
    assert observer.bind(source)['sha256']==observer.ORIGINAL
    patched=observer.observation_patch(original)
    assert original[:original.index('bool HybridSSSP()')]==patched[:patched.index('bool HybridSSSP()')]
    assert 'post_base_update_workflow_seconds' in patched and len(patched)-len(original)>0
    author=json.loads(source.with_name('grapin_update_receipt.json').read_text())
    commands=observer.build_commands(author,Path(author['source']).parent/'receipt.json',Path('/fixture/isolated'))
    assert len(commands)==2 and '/fixture/isolated/source' in commands[0] and '/fixture/isolated/build' in commands[1]
    assert not any(str(Path(author['source']).parent) in item for command in commands for item in command)
    campaign=Path(__file__).resolve().parents[3];runner=campaign/'first_pass_20261106/state_main_first_pass/runner'
    sys.path.insert(0,str(runner));import state_worker
    with tempfile.TemporaryDirectory(prefix='sssp-packets-',dir=a.scratch) as directory:
        root=Path(directory);inputs.build(root/'build');writer=inputs.UpdateWriter(root/'build/receipt.json')
        off=np.array([0,2,3,3,5,6],dtype='<i4');ind=np.array([1,4,2,0,4,1],dtype='<i4');weights=np.array([1,9,10,15,16,2],dtype='<f8')
        weights.tofile(root/'weights.bin');checks=[]
        for case,positions in enumerate((np.array([],dtype='<i8'),np.array([0,4],dtype='<i8'),np.arange(6,dtype='<i8'))):
            position_path=root/f'positions{case}.npy';np.save(position_path,positions,allow_pickle=False)
            row=dict(m=6,positions={'path':str(position_path)},canonical={'weights':{'path':str(root/'weights.bin')}},edge_cuts=[0,6])
            result=root/f'updates{case}.edges';expected=[]
            for version in range(1,21):
                for kind,current in [('d',version-1),('a',version)]:
                    packets=list(state_worker.packet_for(row,current))
                    for (rank,position,value),p0 in zip(packets,positions):
                        assert rank==0 and position==p0
                        u=int(np.searchsorted(off,p0,side='right')-1)
                        expected.append(f'{kind} {u} {int(ind[p0])} {int(value)}\n')
            nbytes=writer.write(5,6,off,ind,weights,positions,result)
            assert result.read_bytes()==''.join(expected).encode()
            assert nbytes==writer.write(5,6,off,ind,weights,positions)==result.stat().st_size
            checks.append(dict(positions=len(positions),commits=20,bytes=nbytes,sha256=hashlib.sha256(result.read_bytes()).hexdigest()))
        try:writer.write(5,6,off,ind,weights,np.array([4,0],dtype='<i8'))
        except ValueError:pass
        else:raise AssertionError('Unsorted positions accepted')
        class Session:
            def __init__(self,bad=None):self.commits=[];self.queries=[];self.values=[];self.closed=False;self.bad=bad
            def commit_weights(self,base,current,packet,storage):
                self.commits.append((base,current,list(packet),storage));return {'fixture_commit':True}
            def query(self,snapshot,function,params):
                self.queries.append((snapshot,function,params));value=np.arange(5,dtype=np.float64)
                if self.bad=='alias' and self.values:value=self.values[0]
                value.setflags(write=False);self.values.append(value)
                return SimpleNamespace(values=value,seconds=0.,backend_metrics={})
            def close(self):
                self.closed=True
                if self.bad=='close_mutation':self.values[0].setflags(write=True);self.values[0][0]=7
        row.update(n=5,sources=[0,1,2]);spec=dict(dataset='FIXTURE',system='D1C1')
        session=Session();held,timing=adapter.own_workflow(spec,row,lambda:session,state_worker.packet_for,lambda *args:True)
        assert len(held)==21 and len(session.commits)==20 and session.closed
        assert [q[0] for q in session.queries]==[f'FIXTURE@{v}' for v in range(21)]
        assert all(q[1:] == ('sssp',{'source':0}) for q in session.queries)
        assert 0<timing['post_base_update_workflow_seconds']<=timing['prepared_service_seconds']
        for bad in ('alias','close_mutation'):
            session=Session(bad)
            prior,failed=adapter.own_workflow(spec,row,lambda:session,state_worker.packet_for,lambda *args:True)
            assert not failed['execution_complete'] and failed['execution_error'] and failed['prepared_service_seconds'] is None
            assert session.closed and prior
        # CPU-only plan fixtures deliberately contain no executable GPU source.
        campaign_fixture=root/'campaign';state=root/'state';state.mkdir();deployment=campaign_fixture/'state_unified_full_v1/main/deployment.json';deployment.parent.mkdir(parents=True)
        selection=dict(source_tree_sha256=planner.SOURCE,python_invocation=str(Path(sys.executable).absolute()),native_dso={'path':str(root/'fixture.so')},source_root=str(root/'fixture-source'),hash_cache_helper={'path':str(root/'fixture-cache.py')},hash_cache=str(root/'fixture-cache'))
        (state/'selection.json').write_text(json.dumps(selection))
        def make_input(name,rho,count=0):
            path=root/(name+str(rho)+'.json');path.write_text(json.dumps({'datasets':{name:dict(n=5,m=max(6,count),sources=[0,1,2],changed_positions=count)}}));return observer.bind(path)
        tiny=make_input('TINY_A',1e-4,2);coverage=[]
        actual=json.loads(source.with_name('actual_density_metadata.json').read_text())
        actual_counts={r['dataset']:r['metadata']['changed_positions'] for r in actual['rows']}
        for name in ('IN','OR'):
            for rho in planner.RHO:
                binding=make_input(name,rho,actual_counts[name] if rho==1 else 2)
                coverage.append(dict(experiment='E3',partitions=1,datasets=[name],rho=rho,locality='uniform',inputs=binding))
        (state/'coverage.json').write_text(json.dumps({'configurations':coverage}));deployment.write_text(json.dumps(dict(environment={},resources={'1':{'gpu-uuids':'FIXTURE_UUID','cpu-set':','.join(map(str,range(16))),'lock':str(root/'fixture-lock'),'tiny_inputs':tiny}})))
        build=root/'author-build.json';build.write_text(json.dumps({'status':'PASS_OBSERVATION_ONLY_CPU_BUILD'}))
        result=planner.prepare(campaign_fixture,state,build,root/'build/receipt.json',root/'planned');plan=json.loads((root/'planned/plan.json').read_text())
        assert result['jobs']==39 and plan['unavailable']==[]
        assert len([j for j in plan['jobs'] if j['phase']=='tiny'])==3 and len([j for j in plan['jobs'] if j['phase']=='target'])==36
        assert all(len(j['depends_on'])==1 for j in plan['jobs'] if j['phase']=='target')
        coverage[-1]['inputs']=make_input('OR',1.,2**32//20+1);(state/'coverage.json').write_text(json.dumps({'configurations':coverage}))
        planner.prepare(campaign_fixture,state,build,root/'build/receipt.json',root/'overflow_plan');overflow=json.loads((root/'overflow_plan/plan.json').read_text())
        assert len(overflow['jobs'])==38 and len(overflow['unavailable'])==1 and overflow['unavailable'][0]['status']=='AUTHOR_UPDATE_COUNT_UNSUPPORTED'
        import _ctypes;_ctypes.dlclose(writer.library._handle)
    record=dict(status='PASS_CPU_SSSP_COMMON_UPDATE_ADAPTER',gpu_executed=False,native_algorithm_executed=False,
        original_author=observer.bind(source),observer=observer.bind(observer.__file__),formatter=observer.bind(Path(__file__).with_name('update_text.cpp')),
        frozen_packet_source=observer.bind(runner/'state_worker.py'),fixtures=checks,
        checks=['Exact d/a bytes agree with frozen packet_for for all 20 cyclic versions, including zero updates, sparse/full density and 16-to-1 wrap',
            'Count-only formatter matches actual output bytes','Malformed positions rejected','Grapin App/kernel prefix unchanged; observer adds only one completed-base timestamp and one elapsed output field',
            'Actual original receipt build-command schema rebases to isolated source/build directories',
            'CPU fake Session executes 20 exact commits and 21 fixed-source full queries, closes, rejects aliased or teardown-mutated outputs',
            'CPU plan fixtures using bound actual IN/OR max counts preserve36 executable targets plus3 tiny checks; synthetic uint32 over-count explicitly unavailable; all targets depend on one initial same-system check'],
        worker=observer.bind(adapter.__file__),planner=observer.bind(planner.__file__),test=observer.bind(__file__))
    a.receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':main()
