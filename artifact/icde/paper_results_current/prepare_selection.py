"""Build explicit current grids or bind a preselected receipt index; never find latest runs."""
from pathlib import Path
from collections import Counter
import argparse
import json
import normalize as n
import protocol


def scaffold(experiments):
    rows=[]
    for row in protocol.state_cells()+protocol.e6_cells():
        if row['experiment'] not in experiments:continue
        row=dict(row)
        if row['experiment']=='E6':
            row['id']='E6_'+row['dataset']+'_'+row['system']+'_'+row['ablate']+'_'+row['coverage_backend']+'_p'+str(row['gpu_count'])
            row.update(adapter='e6',role='eggpu' if row['system']=='eggpu' else 'external')
        else:row.update(adapter='state',role='eggpu')
        row['runs']=[];rows.append(row)
    if 'E4S' in experiments:
        for graph in ('US','OR','IN','UK'):
            for batch in (32,128,512):
                for variant in ('eggpu','c2graph_pm','kgraph_sparse' if graph=='US' else 'kgraph_dense'):
                    rows.append(dict(id=f'E4S_{graph}_{variant}_q{batch}',experiment='E4S',adapter='e4s',
                        role='eggpu' if variant=='eggpu' else 'external',dataset=graph,variant=variant,batch=batch,runs=[],recoverable_outputs=[]))
    n.validate_grid(rows,'formal')
    return dict(schema='eggpu.paper.selection.scaffold.v1',status='PENDING_EXPLICIT_BINDINGS_NOT_RESULTS',
                export_protocol='paper-results-protocol-completion-v1',mode='formal',
                expected_counts=dict(Counter(row['experiment'] for row in rows)),expected_repetitions=10,
                required_cells=[row['id'] for row in rows],cells=rows)


def bind_index(plan,index):
    """The index names each selected cell explicitly; it cannot redefine its dimensions."""
    n.need(index.get('status')=='FROZEN_EXPLICIT_RECEIPT_INDEX','Explicit frozen receipt index required')
    supplied={row['id']:row for row in index['cells']}
    n.need(len(supplied)==len(index['cells']) and set(supplied)==set(plan['required_cells']),'Missing/duplicate/extra receipt-index cell')
    selected=[]
    for template in plan['cells']:
        row=dict(template);extra=supplied[row['id']]
        for key,value in extra.items():
            if key in template and key not in ('runs','recoverable_outputs'):
                n.need(value==template[key],'Receipt index attempts to change a predeclared dimension: '+key)
            row[key]=value
        selected.append(row)
    n.validate_grid(selected,'formal')
    return dict(schema='eggpu.paper.selection.v1',status='FROZEN_EXPLICIT_SELECTION',mode='formal',
                export_protocol='paper-results-protocol-completion-v1',cells=selected,required_cells=plan['required_cells'],
                final_eggpu_release=index['final_eggpu_release'],input_registry=index['input_registry'],
                path_remap=index.get('path_remap',[]),attempt_history=index.get('attempt_history',[]),
                synthetic_fixture=index.get('synthetic_fixture') is True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiments',nargs='+',choices=('E3','E4','E3R','E4S','E6'),required=True)
    p.add_argument('--receipt-index',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();document=scaffold(set(args.experiments))
    if args.receipt_index:document=bind_index(document,json.loads(args.receipt_index.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(document,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=document['status'],cells=len(document['cells']),performance_evidence=False)))


if __name__=='__main__':main()
