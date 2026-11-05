"""Fresh CPU references, mutation rejection, and exact producer identity."""
from pathlib import Path
import json
import pytest
import bind_references as binder
import e6_worker as worker
import e6_cpu_reference as reference
import sg_worker as shared

@pytest.fixture
def produced(tmp_path):
    graph=shared.load_input(worker.make_tiny(tmp_path/'input'))
    values,proof=worker.reference_cached(graph,tmp_path/'oracle')
    return graph,values,Path(proof['path'])

def test_fresh_oracle_roundtrip(produced,tmp_path):
    graph,expected,path=produced
    manifest=binder.create([path]);target=tmp_path/'manifest.json';target.write_text(json.dumps(manifest))
    values,_=reference.load(graph,target,worker.CONTRACT,Path(worker.__file__).with_name('e6_reused.py'))
    assert worker.audit_output({k:worker.reused.own(v) for k,v in values.items()},expected,graph['n'])['pass']

def test_duplicate_input_rejected(produced):
    with pytest.raises(ValueError,match='Duplicate'):binder.create([produced[2],produced[2]])

def test_mutated_payload_rejected(produced):
    path=produced[2];blob=path.with_suffix('.npz');blob.write_bytes(blob.read_bytes()+b'altered')
    with pytest.raises(ValueError,match='data hash'):binder.create([path])

def test_wrong_producer_rejected(produced):
    path=produced[2];record=json.loads(path.read_text());record['identity']['code']['sha256']='0'*64;path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='producer'):binder.create([path])

def test_gpu_receipt_rejected(produced):
    path=produced[2];record=json.loads(path.read_text());record['gpu_executed']=True;path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='independent CPU'):binder.create([path])

def test_empty_references_rejected():
    with pytest.raises(ValueError,match='Explicit complete'):binder.create([])
