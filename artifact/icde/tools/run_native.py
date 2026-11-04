#!/usr/bin/env python3
"""Execute the same public Session.native API with one or several logical GPUs.

This is an explicit user workload entrypoint, not the qualification release gate.
Input manifests and data are supplied separately. It never downloads datasets.
"""
import argparse,json,os,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
import verify
ROOT=Path(__file__).resolve().parents[3]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-dso',type=Path,required=True)
    p.add_argument('--build-receipt',type=Path,required=True,help='Portable clean-build receipt for this exact package/source and DSO')
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--devices',default='0',help='Logical CUDA IDs after caller CUDA_VISIBLE_DEVICES mapping')
    p.add_argument('--function',choices=('bfs','sssp','wcc','pr'),required=True);p.add_argument('--source',type=int,default=0)
    p.add_argument('--communication',choices=('host_staging','peer_to_peer','nccl','auto'),default='host_staging')
    p.add_argument('--updates',type=Path,help='JSON array of [partition_rank,weight_position,new_value], committed as a new immutable snapshot')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=verify.verify(ROOT)
    devices=[int(x) for x in a.devices.split(',')]
    if len(devices) not in (1,2,4,8) or min(devices)<0 or len(set(devices))!=len(devices):p.error('Choose 1,2,4,8 distinct logical devices')
    if a.updates and a.function!='sssp':p.error('This minimal update example is weighted SSSP only')
    out=a.output.resolve()
    if out.exists() or out==ROOT or ROOT in out.parents:raise ValueError('Absent output directory outside package required')
    dso=a.native_dso.resolve();build=json.loads(a.build_receipt.read_text())
    if (build.get('schema')!='eggpu.portable.clean_build/v2' or build.get('status')!='PASS_BUILD_GPU_UNQUALIFIED'
            or build.get('base_source_sha256')!=m['base_source_sha256']
            or build.get('source_sha256')!=m['current_source_sha256']
            or not any(Path(row['path']).resolve()==dso and row['sha256']==verify.sha(dso) for row in build.get('outputs',[]))):raise ValueError('Require the exact new portable source/build/DSO receipt')
    sys.dont_write_bytecode=True;sys.path[:0]=[str(dso.parent),str(ROOT/'Easy-Graph')]
    # Preserve library defaults for optimizations. Explicit transport is part of
    # the workload request; no per-graph engine/timing selection is performed.
    os.environ['EGGPU_COMM_BACKEND']=a.communication
    import numpy as np
    import cpp_easygraph as cpp
    from easygraph.gpu import Session
    if Path(cpp.__file__).resolve()!=dso:raise ValueError('Different DSO loaded')
    out.mkdir(parents=True);record=dict(schema='eggpu.portable.native_call/v1',status='RUNNING',performance_eligible=False,gpu_executed=False,
        base_source_sha256=m['base_source_sha256'],source_sha256=m['current_source_sha256'],dso_sha256=verify.sha(dso),
        build_receipt_sha256=verify.sha(a.build_receipt),
        manifest_sha256=verify.sha(a.manifest),function=a.function,logical_devices=devices,requested_communication=a.communication,calls=[])
    session=None;held=[]
    try:
        session=Session.native({'A0':str(a.manifest.resolve())},devices=tuple(devices),cpp=cpp,route='lease_sparse',inventory_mode='recheck');record['gpu_executed']=True
        parameters={'source':a.source} if a.function in ('bfs','sssp') else {}
        def query(name):
            begin=time.perf_counter();answer=session.query(name,a.function,parameters);seconds=time.perf_counter()-begin
            if not answer.values.flags.owndata or answer.values.flags.writeable:raise AssertionError('Owning immutable complete output required')
            held.append(answer);np.save(out/(name+'.npy'),answer.values,allow_pickle=False)
            record['calls'].append(dict(snapshot=name,complete_seconds=seconds,metrics=dict(answer.backend_metrics),output_sha256=verify.sha(out/(name+'.npy')),vertices=len(answer.values)))
        query('A0')
        if a.updates:
            session.retain_snapshot('A0');record['commit']=session.commit_weights('A0','A1',json.loads(a.updates.read_text()),storage='native_memory');query('A1');session.release_snapshot('A0')
        record['inventory']=session.inventory();session.close();session=None
        for answer,row in zip(held,record['calls']):np.testing.assert_array_equal(answer.values,np.load(out/(row['snapshot']+'.npy'),allow_pickle=False))
        record.update(status='COMPLETE_USER_WORKLOAD_NOT_QUALIFICATION',outputs_survive_close=True)
    except BaseException:record.update(status='FAIL_USER_WORKLOAD',error=traceback.format_exc());raise
    finally:
        if session is not None:session.close()
        (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status=record['status'],outputs=len(record['calls']),dso_sha256=record['dso_sha256'])))
if __name__=='__main__':main()
