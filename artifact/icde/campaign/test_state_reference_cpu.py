"""Tiny CPU-only cache integrity and full-vector quality tests."""
from pathlib import Path
import argparse, json, tempfile
import numpy as np
import state_protocol as c
import state_worker as worker
import state_reference as ref


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();inputs=worker.load_inputs(args.inputs);checks=[]
    if not inputs['qualification_only']:raise ValueError('CPU unit tests require only tiny fixtures')
    with tempfile.TemporaryDirectory(prefix='state-reference-cpu-') as directory:
        cache=ref.ReferenceCache(directory,max_bytes=128<<20,min_free_bytes=0)
        for name,row in inputs['datasets'].items():
            for function in c.FUNCTIONS:
                for version in (0,1):
                    matrix=ref.matrix_for(row,version,function);source=int(row['sources'][0])
                    truth,_=ref.build_reference(matrix,function,source);values=truth.astype(np.float64,copy=True);values.flags.writeable=False
                    first=worker.output_quality(values,row,version,function,source,cache)
                    builder=ref.build_reference
                    def forbidden(*args,**kwargs):raise AssertionError('Cache hit solved the reference again')
                    ref.build_reference=forbidden
                    try:second=worker.output_quality(values,row,version,function,source,cache)
                    finally:ref.build_reference=builder
                    if not second['reference']['cache_hit']:raise AssertionError('Expected reference hit')
                    wrong=values.copy();wrong[0]=123456;wrong.flags.writeable=False
                    try:worker.output_quality(wrong,row,version,function,source,cache)
                    except (ValueError,AssertionError):pass
                    else:raise AssertionError('Changed output passed full-vector audit')
                    checks.append({'dataset':name,'function':function,'version':version,'cached_solve_avoided':True,'changed_output_rejected':True,'passed':True})
            # Source and mutation version must never alias a reference key.
            keys=[ref.key_document(row,v,'sssp',s)[1] for v,s in ((0,0),(1,0),(0,1))]
            if len(set(keys))!=3:raise AssertionError('Distinct query identities alias')
            checks.append({'dataset':name,'source_and_version_separate':True,'passed':True})
        item=next(Path(directory).glob('*.npz'));metadata=json.loads(item.with_suffix('.json').read_text());item.write_bytes(b'corrupt')
        row=next(row for row in inputs['datasets'].values() if row['n']==metadata['key']['n'])
        key=metadata['key']
        try:cache.get(row,key['version'],key['function'],key['source'] or 0,ref.matrix_for(row,key['version'],key['function']))
        except ValueError as error:
            if 'corruption' not in str(error):raise
        else:raise AssertionError('Corrupt cached bytes accepted')
        checks.append({'corrupt_reference_rejected':True,'passed':True})
    receipt={'status':'PASS_CPU_INDEPENDENT_REFERENCE_CACHE','gpu_executed':False,'tests':len(checks),'checks':checks,
             'files':[c.bind(__file__),c.bind(worker.__file__),c.bind(ref.__file__)],'inputs':c.bind(args.inputs),
             'scope':'Tiny fixtures only; complete output checks and cached CPU solves. No GPU/performance experiment.'}
    c.save(args.output,receipt);print(json.dumps({'status':receipt['status'],'tests':len(checks)}))


if __name__=='__main__':main()
