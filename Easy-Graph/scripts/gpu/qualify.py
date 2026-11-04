#!/usr/bin/env python3
"""Small complete-output qualification, not a performance benchmark.

--prepare-only writes manifests/references without touching CUDA. Execution is
explicit, checks a low-load device gate, pins UUIDs, and saves every full output.
Run each device count in a fresh process. Build/CPU receipts do not imply PASS_GPU.
"""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, traceback

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--devices',default='0')
    p.add_argument('--backend',choices=('native','arena'),default='native')
    p.add_argument('--communication',choices=('auto','host_staging','peer_to_peer','nccl'),default='auto',
                   help='Requested native transport; resolved runtime and observed per-query transports are recorded')
    p.add_argument('--native-dso',type=Path)
    p.add_argument('--arena-library-dir',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--max-utilization',type=int,default=10)
    p.add_argument('--min-free-mib',type=int,default=2048)
    p.add_argument('--pagerank-halo-exchange',choices=('peer_gather','scalar_copy','nccl_allgather'),default='peer_gather')
    p.add_argument('--fixture',choices=('mixed','no_halo','partial_halo'),default='mixed')
    a=p.parse_args()
    devices=[int(x) for x in a.devices.split(',')]
    if len(devices) not in (1,2,4,8) or min(devices)<0 or len(set(devices))!=len(devices):p.error('select 1, 2, 4 or 8 distinct nonnegative device indices')
    if a.backend=='arena' and len(devices)!=1:p.error('arena is a single-device executor')
    if not a.prepare_only and a.native_dso is None:p.error('--native-dso is required for GPU execution')
    out=a.output.resolve()
    if out==ROOT or ROOT in out.parents:p.error('output must be outside the source tree')
    out.mkdir(parents=True,exist_ok=False)
    result=dict(status='RUNNING',gpu_execution_attempted=False,backend=a.backend,requested_devices=devices,
                requested_communication=a.communication,requested_halo_exchange=a.pagerank_halo_exchange,fixture=a.fixture,outputs=[],source={})
    source_files=[]
    for folder in ('easygraph/gpu','easygraph/egmgpu','cpp_easygraph','gpu_easygraph','scripts/gpu'):
        source_files.extend(f for f in (ROOT/folder).rglob('*') if f.is_file() and f.suffix in ('.py','.cpp','.cu','.cuh','.h'))
    for f in sorted(source_files):
        result['source'][str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    try:result['git_head']=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    except subprocess.CalledProcessError:result['git_head']=None
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path.insert(0,str(ROOT))
    if a.native_dso:
        a.native_dso=a.native_dso.resolve();sys.path.insert(0,str(a.native_dso.parent))
    try:
        if not a.prepare_only:
            hardware=[]
            for d in devices:
                line=subprocess.check_output(['nvidia-smi','-i',str(d),'--query-gpu=index,uuid,name,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
                fields=[x.strip() for x in line.strip().split(',')]
                if len(fields)!=5 or int(fields[0])!=d:raise RuntimeError('Unexpected physical device response')
                if float(fields[3])<a.min_free_mib or float(fields[4])>a.max_utilization:raise RuntimeError('Device is outside the requested low-load gate')
                hardware.append(dict(index=d,uuid=fields[1],name=fields[2],free_mib=float(fields[3]),utilization=float(fields[4])))
            result['hardware_before']=hardware
            os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES']=','.join(x['uuid'] for x in hardware)
            os.environ.update(EGGPU_COMM_BACKEND=a.communication,EGMGPU_WORKSPACE_POLICY='per_worker',
                 EGMGPU_SHARED_DEVICE_CACHE_BYTES=str(128<<20),EGMGPU_PARTITION_STATE_CACHE_ENTRIES='32',
                 EGMGPU_PAGERANK_DEVICE_METADATA='pull_only',EGMGPU_PAGERANK_HOST_TRANSPOSE='auto',
                 EGMGPU_PAGERANK_HOST_WEIGHT_STORAGE='implicit',EGMGPU_PAGERANK_PULL='vector8' if len(devices)==1 else 'split',
                 EGMGPU_PAGERANK_OUTPUT='pinned',EGMGPU_PAGERANK_SCHEDULE='streamed',
                 EGMGPU_PAGERANK_VIEW='transpose',EGMGPU_PAGERANK_TRANSPOSE_ENGINE='cpu',
                 EGMGPU_PAGERANK_SCALAR_REDUCTION='host',EGMGPU_PAGERANK_HALO_EXCHANGE=a.pagerank_halo_exchange,EGMGPU_PAGERANK_SHARE_CSR='1')
            if a.arena_library_dir:os.environ['EGGPU_ARENA_LIBRARY_DIR']=str(a.arena_library_dir.resolve())
        import numpy as np
        from qualification_inputs import canonical,manifest,reference,check_vector,sha
        from easygraph.gpu._manifest_signatures import sign
        n=127
        if a.fixture=='no_halo':
            cuts=np.linspace(0,n,len(devices)+1,dtype=int)
            pairs={(u,lo+(u-lo+k)%(hi-lo)) for lo,hi in zip(cuts[:-1],cuts[1:]) for u in range(lo,hi) for k in (1,3,7)}
        else:
            limit=n-1 if a.fixture=='mixed' or len(devices)==1 else min(n-1,int(np.linspace(0,n,len(devices)+1,dtype=int)[2]))
            pairs={(u,(u+k)%limit) for u in range(limit) for k in (1,3,7)}
        edges=np.array(sorted(pairs),np.int32)
        weights=((edges[:,0].astype(np.int64)*31+edges[:,1]*17)%16+1).astype(float)
        raw=canonical(n,edges,weights)
        new=dict(raw,weights=raw['weights'].copy());new['weights'][0]+=2
        refs={};paths={};maps={}
        for name,graph in [('A0',raw),('A1',new)]:
            refs[name],_=reference(graph)
            unweighted=dict(graph,weights=np.ones(len(edges)))
            refs[name]['bfs']=reference(unweighted)[0]['sssp']
            np.savez(out/(name+'-reference.npz'),**refs[name])
        for name,weighted in [('A0',True),('U',False)]:
            paths[name],maps[name]=manifest(raw,out/name,name,len(devices),weighted=weighted)
        weak_edges=np.unique(np.vstack((edges,edges[:,::-1])),axis=0)
        weak=canonical(n,weak_edges,np.ones(len(weak_edges)))
        paths['W'],_=manifest(weak,out/'W','W',len(devices),weighted=False)
        wp=Path(paths['W']['manifest']);wd=json.loads(wp.read_text());wd['graph']['directed']=False;wd['view']['orientation']='undirected'
        top,_,view=sign(wd);wd['graph']['topology_sha256']=top;wd['view']['signature']=view;wp.write_text(json.dumps(wd,indent=2)+'\n')
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        _,labels=connected_components(csr_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(n,n)),directed=True,connection='weak')
        wcc=np.array([min(np.flatnonzero(labels==label)) for label in labels],float)
        result['prepared']=dict(n=n,arcs=len(edges),partitions=len(devices),canonical='directed nonnegative integer-weight graph plus one isolated vertex',cpu_references=8)
        if a.prepare_only:
            result['status']='PASS_CPU_PREPARATION';return
        import cpp_easygraph as cpp
        if Path(cpp.__file__).resolve()!=a.native_dso:raise RuntimeError('Loaded a different native extension')
        result['native_dso']=dict(path=str(a.native_dso),sha256=sha(a.native_dso))
        result['device_topology']=cpp.cpp_egmgpu_probe_topology()
        from easygraph.gpu import Session,StaleRequestError
        held=[];expected_outputs=[]
        pr_tolerance=1e-5 if a.backend=='arena' else 1e-8
        result['numerical_check']=dict(distances='exact including unreachable',pagerank_l1_error_below=pr_tolerance)
        def compare(values,ref,alg):
            if alg in ('pr','weighted_pr'):
                error=float(np.abs(values-ref).sum())
                if values.shape!=ref.shape or not np.all(np.isfinite(values)) or error>=pr_tolerance:
                    raise AssertionError((alg,error,pr_tolerance))
                return error
            return check_vector(values,ref,alg)
        def check(session,name,alg,ref,weight_version=None):
            answer=session.query(name,alg,{'source':0} if alg in ('bfs','sssp') else {})
            error=compare(answer.values,ref,alg)
            rank_check=None
            if alg in ('pr','weighted_pr'):
                weights_for_check=np.ones(len(edges)) if alg=='pr' else (new['weights'] if name=='A1' or weight_version=='A1' else raw['weights'])
                strength=np.bincount(edges[:,0],weights=weights_for_check,minlength=n)
                scaled=np.divide(answer.values,strength,out=np.zeros(n),where=strength!=0)
                incoming=np.bincount(edges[:,1],weights=weights_for_check*scaled[edges[:,0]],minlength=n)
                image=(1-.85)/n+.85*(incoming+float(answer.values[strength==0].sum())/n)
                residual=float(np.abs(image-answer.values).sum());mass_error=abs(float(answer.values.sum())-1)
                if residual>1e-9 or mass_error>1e-8 or np.min(answer.values)<-1e-15:raise AssertionError(('global PR contract',residual,mass_error))
                rank_check=dict(global_l1_residual=residual,mass_error=mass_error)
                if a.backend=='arena':
                    contract=answer.contract
                    if contract.max_iterations!=1000 or contract.tolerance_per_vertex!=1e-9/n:
                        raise AssertionError('Arena stopping contract differs from qualification parameters')
                    rank_check['stopping_contract']=contract.to_dict()
                if a.pagerank_halo_exchange=='nccl_allgather':
                    m=answer.backend_metrics;active=len(devices)>1 and m['halo_values_per_iteration']>0
                    expected='nccl_allgather' if active else 'local_skip' if len(devices)==1 else 'global_no_halo_skip'
                    if m['halo_exchange']!=expected or m['nccl_halo_active']!=active or m['host_staging_bytes']!=0:raise AssertionError('Bulk route silently changed')
                    if m['scalar_reduction_backend']!=('local' if len(devices)==1 else 'host'):raise AssertionError('Scalar route changed')
            if not answer.values.flags.owndata or answer.values.flags.writeable:raise AssertionError('Owning immutable output required')
            transport_keys=('communication_backend','runtime_communication_backend','halo_exchange',
                            'scalar_reduction_backend','collective_context_used','requested_halo_exchange','nccl_halo_active','nccl_halo_buffer_bytes_per_device','nccl_halo_stride','halo_values_per_iteration')
            transport={key:answer.backend_metrics[key] for key in transport_keys if key in answer.backend_metrics}
            held.append(answer);expected_outputs.append((ref.copy(),alg));result['outputs'].append(dict(graph=name,algorithm=alg,values=len(answer.values),l1_error=error,numerical_contract=rank_check,observed_transport=transport))
            return answer
        result['gpu_execution_attempted']=True
        if a.backend=='native':
            with Session.native({name:row['manifest'] for name,row in paths.items()},devices=tuple(range(len(devices))),cpp=cpp,route='lease_sparse') as session:
                result['runtime_after_start']=cpp.cpp_egmgpu_runtime_status()
                result['resolved_communication']=result['runtime_after_start']['communication_backend']
                for alg in ('bfs','pr'):check(session,'U',alg,refs['A0'][alg])
                check(session,'W','wcc',wcc)
                for alg in ('sssp','weighted_pr'):check(session,'A0',alg,refs['A0'][alg])
                session.retain_snapshot('A0')
                partition,position=maps['A0'][0]
                session.commit_weights('A0','A1',[(partition,position,float(new['weights'][0]))],out/'commits')
                for alg in ('sssp','weighted_pr'):check(session,'A1',alg,refs['A1'][alg])
                session.release_snapshot('A0')
                for alg in ('sssp','weighted_pr'):check(session,'A0',alg,refs['A0'][alg])
                result['inventory']=session.inventory()
                result['capabilities']=session.capabilities().to_dict()
        else:
            inputs={'A':dict(nodes=np.arange(n),edges=edges,weights=weights)}
            with Session.arena(inputs,budget_bytes=32<<20,device=0,pr_tolerance_l1=1e-9,pr_max_iter=1000) as session:
                for alg in ('bfs','sssp','weighted_pr'):check(session,'A',alg,refs['A0'][alg])
                old=session.request('A','sssp',{'source':0})
                session.update('A',{'set':[[int(edges[0,0]),int(edges[0,1]),float(new['weights'][0])]]})
                try:session.execute(old)
                except StaleRequestError:result['stale_request_rejected']=True
                else:raise AssertionError('Stale semantic request accepted')
                for alg in ('sssp','weighted_pr'):check(session,'A',alg,refs['A1'][alg],weight_version='A1')
                session.update('A',{'set':[[int(edges[0,0]),int(edges[0,1]),float(weights[0])]]})
                check(session,'A','sssp',refs['A0']['sssp'])
                result['capabilities']=session.capabilities().to_dict()
        for i,answer in enumerate(held):
            compare(answer.values,*expected_outputs[i])
            if not answer.values.flags.owndata or answer.values.flags.writeable:raise AssertionError('Output lifetime failed')
            for other in held[:i]:
                if np.shares_memory(answer.values,other.values):raise AssertionError('Outputs alias')
        np.savez(out/'outputs.npz',**{f'output_{i:03}':x.values for i,x in enumerate(held)})
        result.update(status='PASS_GPU',full_outputs=len(held),all_outputs_survive_close=True,output_sha256=sha(out/'outputs.npz'))
    except BaseException:
        result.update(status='FAIL',traceback=traceback.format_exc());raise
    finally:
        (out/'receipt.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        print(json.dumps({k:result.get(k) for k in ('status','backend','gpu_execution_attempted','full_outputs')}))

if __name__=='__main__':main()
