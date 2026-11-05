"""Prepare eight small position/manifests against existing IN/OR P8 bases.

Never serializes or copies bases. Keeps the original canonical/shared-base
provenance, adds the explicit new selector proof, and preserves infeasibility.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,sys
import numpy as np
import hotspot_positions as selector


def load(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    path.chmod(0o444)

def checked(item):
    if selector.binding(item['path'])!=item:raise ValueError('Bound input changed: '+item['path'])
    return Path(item['path'])


def prepare(campaign,output,source_freeze):
    campaign=Path(campaign);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    grid=load(campaign/'state_grid_full_execute_expansion200_v1/receipt.json')
    if grid['status']!='PASS_CPU_SHARED_GRID_PREPARED_GPU_UNQUALIFIED' or len(grid['cells'])!=56:raise ValueError('Complete fixed input grid required')
    freeze=load(source_freeze);signer=freeze['source_files']['easygraph/gpu/_manifest_signatures.py']
    selected={}
    for binding in grid['cells']:
        record=load(checked(binding));c=record['cell']
        if c['experiment']=='E3' and c['partitions']==8 and c['rho']==1e-4 and c['locality']=='uniform' and c['datasets'][0] in ('IN','OR'):
            name=c['datasets'][0]
            if name in selected:raise ValueError('Duplicate shared source input')
            if {k:record['source_signer'][k] for k in ('bytes','sha256')}!=signer:raise ValueError('Latest source signer requires new base')
            selected[name]=(binding,record)
    if set(selected)!={'IN','OR'}:raise ValueError('Exact IN/OR P8 shared inputs required')
    receipt=dict(status='RUNNING_CPU_PREPARATION',gpu_executed=False,source_freeze=selector.binding(source_freeze),controller=selector.binding(__file__),selector=selector.binding(selector.__file__),inputs=[],infeasible=[],base_bytes_copied=0)
    for name,(original,record) in selected.items():
        row=record['datasets'][name];base=load(checked(row['shared_base']))
        if base['status']!='PASS_IMMUTABLE_STATE_BASE':raise ValueError('Incomplete shared base')
        for item in base['owned_files']:checked(item)
        for item in row['canonical'].values():checked(item)
        meta=load(row['canonical']['metadata']['path']);n,m=row['n'],row['m']
        offsets=np.memmap(row['canonical']['offsets']['path'],dtype='<i4',mode='r',shape=(n+1,));indices=np.memmap(row['canonical']['indices']['path'],dtype='<i4',mode='r',shape=(m,))
        for k in (1,2,4,8):
            folder=output/(name+'_k'+str(k));folder.mkdir()
            positions,proof=selector.select(offsets,indices,meta['directed'],row['edge_cuts'],np.linspace(0,n,9,dtype=np.int64),k,1e-4,20261104)
            proof.update(base=row['shared_base'],canonical=row['canonical'],selector=receipt['selector'])
            if proof['status']!='PASS_CPU_SELECTION':
                save(folder/'selection.json',proof);receipt['infeasible'].append(selector.binding(folder/'selection.json'));continue
            path=folder/'positions.npy';np.save(path,positions,allow_pickle=False);path.chmod(0o444);proof['positions']=selector.binding(path);save(folder/'selection.json',proof)
            obj=copy.deepcopy(record);cell=obj['cell'];cell.update(id=f'E3_{name}_p8_rho0.0001_partition_hotspot_k{k}',locality='partition_hotspot_k'+str(k),hotspot_k=k)
            r=obj['datasets'][name];r.update(positions=proof['positions'],locality=cell['locality'],actual_rho=proof['actual_rho'],changed_positions=proof['changed_positions'],position_seed=20261104,hotspot_selection=selector.binding(folder/'selection.json'))
            obj['hotspot_preparation']=dict(controller=receipt['controller'],selector=receipt['selector'],original_input=original,source_freeze=receipt['source_freeze'])
            save(folder/'inputs.json',obj);receipt['inputs'].append(selector.binding(folder/'inputs.json'))
        del offsets,indices
    receipt['status']='PASS_EIGHT_HOTSPOT_INPUTS_GPU_UNQUALIFIED' if len(receipt['inputs'])==8 else 'INFEASIBLE_HOTSPOT_CONFIGURATIONS_PRESERVED'
    save(output/'receipt.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-freeze',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.campaign,a.output,a.source_freeze)))
