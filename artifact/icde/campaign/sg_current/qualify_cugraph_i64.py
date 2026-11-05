"""Primary-agent-only GPU capability check of the author's 64/64 WCC API."""
import argparse
import fcntl
import os
from pathlib import Path
import time
from types import SimpleNamespace
import sg_worker as shared


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--execute-gpu', action='store_true')
    for name in ('setup', 'lock-file', 'input-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    a=p.parse_args()
    if not a.execute_gpu: p.error('Explicit --execute-gpu required')
    a.output.mkdir(parents=True, exist_ok=False)
    record=dict(schema='cugraph.i64.capability.v1',status='RUNNING',
                formal_measurements=0, started_unix=time.time(), cases=[])
    try:
        with a.lock_file.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            import numpy as np
            from cugraph_csr_adapter import CuGraphCSR, weak_csr_graph
            graph=shared.load_input(a.input_root/'canonical/TINY_D')
            args=SimpleNamespace(setup=a.setup, gpu_uuid=a.gpu_uuid, min_host_free_gib=8., budget_mib=22000)
            record['admission']=shared.resource_check(args, graph)
            os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid
            for fixture in ('TINY_D','TINY_U'):
                graph=shared.load_input(a.input_root/'canonical'/fixture)
                matrix=shared.graph_matrix(graph)
                weak=matrix.maximum(matrix.T).tocsr(); weak.sort_indices()
                prepared=weak_csr_graph(graph,weak)
                prepared['offsets']=prepared['offsets'].astype(np.int64)
                prepared['indices']=prepared['indices'].astype(np.int64)
                backend=CuGraphCSR(prepared,'wcc')
                values=backend.complete_query(); backend.close(); values.flags.writeable=False
                quality=shared.audit_output(values,shared.reference(graph,'wcc',None),graph,'wcc')
                assert quality['pass'] and backend.contract['topology_dtype']=='int64'
                record['cases'].append(dict(fixture=fixture,quality=quality,
                    output=shared.persist_output(a.output,values),contract=backend.contract))
            record.update(status='PASS_CUGRAPH_I64_CAPABILITY',quality_pass=True)
    except BaseException as e:
        record.update(status='FAIL',error=type(e).__name__+': '+str(e));raise
    finally:
        record['finished_unix']=time.time();shared.save(a.output/'receipt.json',record)


if __name__=='__main__':main()
