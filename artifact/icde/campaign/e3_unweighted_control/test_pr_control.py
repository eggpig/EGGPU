from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import json
import numpy as np
import pytest
import pr_control as c


@pytest.fixture
def graph(tmp_path):
    # Directed, asymmetric, one dangling vertex and one isolated vertex.
    offsets=np.array([0,2,3,5,6,6,6],dtype='<i4')
    indices=np.array([1,2,2,0,3,2],dtype='<i4')
    for name,array in [('offsets',offsets),('indices',indices)]:array.tofile(tmp_path/(name+'.bin'))
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'graph':{'topology_sha256':'weighted-topology','weight_sha256':'w0'}}))
    row=dict(n=6,m=6,canonical={key:c.bind(tmp_path/(key+'.bin')) for key in ('offsets','indices')},
             changed_positions=2,requested_rho=1e-4,locality='uniform',manifest=c.bind(manifest),
             projections={'bfs':{'identity':'US@topology'}})
    # Reading weights in the unit oracle would fail this fixture.
    row['canonical']['weights']={'path':str(tmp_path/'SHOULD_NOT_READ.bin')}
    return row


def test_unit_oracle_matches_independent_dense_linear_system(graph):
    step,reference,proof=c.unit_oracle(graph)
    adjacency=np.zeros((6,6))
    for u,v in [(0,1),(0,2),(1,2),(2,0),(2,3),(3,2)]:adjacency[u,v]=1
    transition=np.divide(adjacency,adjacency.sum(1)[:,None],out=np.zeros_like(adjacency),where=adjacency.sum(1)[:,None]>0)
    transition[adjacency.sum(1)==0]=1/6
    exact=np.linalg.solve(np.eye(6)-.85*transition.T,np.full(6,.15/6))
    np.testing.assert_allclose(reference,exact,rtol=0,atol=1e-11)
    assert np.abs(reference-step(reference)).sum()<1e-12
    assert proof['input_weight_payload_read'] is False


def result(values,arm='D1C1',h2d=0,graph_id='US@topology'):
    return SimpleNamespace(values=values,algorithm='pr',graph_id=graph_id,seconds=.001,
        contract=SimpleNamespace(algorithm='pr',weight_projection='unweighted'),
        backend_metrics={'iterations':50,'converged':True,'native_dynamic_patch':{
            'route':c.ARMS[arm],'retirement_policy':'lineage',
            'per_device_delta':{'0':{'weight_full_h2d_bytes':h2d}}},
            'resource_management':{'state_switch_policy':'closure'}})


class FakeSession:
    def __init__(self,reference,arm,reuse=False):
        self.reference=reference;self.arm=arm;self.calls=[];self.reuse=reuse;self.old=None;self.closed=False
    def query(self,name,function,parameters):
        self.calls.append(('query',name,function,parameters))
        values=self.old if self.reuse and self.old is not None else self.reference.copy()
        values.flags.writeable=False;self.old=values
        return result(values,self.arm)
    def commit_weights(self,base,snapshot,updates,storage):
        self.calls.append(('commit',base,snapshot,storage))
        return dict(kind='native_memory_absolute_weight_commit',snapshot_storage=storage,
                    positions=2,topology_sha256='weighted-topology',weight_sha256='w'+snapshot.split('@')[1])
    def inventory(self):return {'per_device':[]}
    def close(self):self.closed=True


@pytest.mark.parametrize('arm',list(c.ARMS))
def test_exact21_new_calls_and20_commits(graph,arm):
    _,reference,_=c.unit_oracle(graph)
    session=FakeSession(reference,arm)
    args=SimpleNamespace(dataset='US',arm=arm,partitions=1)
    held,before,record=c.run_workflow(args,graph,lambda:session,lambda row,version:())
    assert record['execution_complete'] is True
    assert len(held)==21 and len(before)==21 and len(record['events'])==41
    assert session.closed and len(session.calls)==41
    assert sum(x[0]=='query' for x in session.calls)==21
    assert sum(x[0]=='commit' for x in session.calls)==20
    assert all(x[1:3]==('US@topology','pr') for x in session.calls if x[0]=='query')
    assert len({id(values) for _,values in held})==21
    assert record['first_complete_workflow_seconds']>=record['workflow_complete_host_seconds']>=0


def test_reused_output_buffer_rejected(graph):
    _,reference,_=c.unit_oracle(graph);session=FakeSession(reference,'D1C1',reuse=True)
    _,_,record=c.run_workflow(SimpleNamespace(dataset='US',arm='D1C1',partitions=1),graph,
                              lambda:session,lambda row,version:())
    assert record['execution_complete'] is False
    assert 'Output buffer reused' in record['execution_error'] and session.closed


def test_commit_changes_content_but_not_topology():
    receipt=dict(kind='native_memory_absolute_weight_commit',snapshot_storage='native_memory',positions=2,
                 topology_sha256='topo',weight_sha256='new')
    assert c.validate_commit(receipt,'topo','old',2)=='new'
    for update in [{'weight_sha256':'old'},{'topology_sha256':'changed'},{'positions':0},{'snapshot_storage':'files'}]:
        with pytest.raises(ValueError):c.validate_commit(dict(receipt,**update),'topo','old',2)


def test_only_weight_maintenance_transfer_is_zero(graph):
    _,reference,_=c.unit_oracle(graph)
    evidence=c.query_evidence(result(reference.copy()),'US@topology','D1C1',1,1)
    assert set(evidence['weighted_view_transfer_counters'].values())=={0}
    assert evidence['total_cuda_copy_bytes']=='NOT_FULLY_INSTRUMENTED_DO_NOT_INFER_ZERO'
    c.query_evidence(result(reference.copy(),h2d=100),'US@topology','D1C1',1,0)
    with pytest.raises(ValueError,match='Weight-only'):
        c.query_evidence(result(reference.copy(),h2d=100),'US@topology','D1C1',1,1)


@pytest.mark.parametrize('kind',('algorithm','graph','weight_semantics','no_iterations','wrong_route','wrong_policy','missing_device'))
def test_query_rejects_wrong_execution(graph,kind):
    _,reference,_=c.unit_oracle(graph);r=result(reference.copy())
    if kind=='algorithm':r.algorithm='weighted_pr'
    if kind=='graph':r.graph_id='US@20'
    if kind=='weight_semantics':r.contract.weight_projection='input edge weights'
    if kind=='no_iterations':r.backend_metrics['iterations']=0
    if kind=='wrong_route':r.backend_metrics['native_dynamic_patch']['route']='persistent_full'
    if kind=='wrong_policy':r.backend_metrics['resource_management']['state_switch_policy']='closure_lru'
    if kind=='missing_device':r.backend_metrics['native_dynamic_patch']['per_device_delta']={}
    with pytest.raises(ValueError):c.query_evidence(r,'US@topology','D1C1',1,1)


def test_numerical_quality_detects_dependence_on_weights(graph):
    step,reference,_=c.unit_oracle(graph)
    values=reference.copy();values.flags.writeable=False
    assert c.numerical_quality(values,step,reference,values)['passed']
    bad=values.copy();bad[0]+=.01;bad[1]-=.01;bad.flags.writeable=False
    with pytest.raises(ValueError):c.numerical_quality(bad,step,reference,values)
    view=values.view();view.flags.writeable=False
    with pytest.raises(ValueError,match='owning'):c.numerical_quality(view,step,reference,values)


def test_old_weighted_reader_is_explicitly_inapplicable(graph):
    args=SimpleNamespace(old_readers='per_update',arm='D1C1',phase='target',repeat=0,dataset='US',
                         tiny_proof=Path('tiny'),target_proof=None,gpu_uuids='GPU-one')
    inputs={'qualification_only':False,'partitions':1,'datasets':{'US':graph}}
    with pytest.raises(ValueError,match='inapplicable'):c.validate_arguments(args,inputs)
    args.old_readers='none';c.validate_arguments(args,inputs)
    graph['changed_positions']=0
    with pytest.raises(ValueError,match='actually change'):c.validate_arguments(args,inputs)


def test_proof_cannot_substitute_main_e3_or_old_epoch(tmp_path):
    file=tmp_path/'receipt.json'
    for value in [{'schema':'state.complete.workflow.v4'},
                  {'schema':c.PROTOCOL,'phase':'tiny','status':'PASS_UNWEIGHTED_PR_QUALIFICATION',
                   'configuration':{'old_source':True},'configuration_sha256':'old'}]:
        file.write_text(json.dumps(value))
        with pytest.raises(ValueError,match='proof source'):
            c.validate_proof(file,'tiny',{'new_source':True},{},'US',None)


def test_imports_and_cpu_entry_never_execute_gpu():
    assert 'cupy' not in __import__('sys').modules
    with pytest.raises(ValueError,match='owned by the root'):
        c.execute(SimpleNamespace(execute_gpu=False))
