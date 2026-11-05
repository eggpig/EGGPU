"""Real compiled-DSO host import/clone/capability checks. Never starts CUDA."""
from pathlib import Path
import gc,hashlib,importlib.util,json,sys
import numpy as np
import pytest
from test_wide_offsets import fixture,widen,read,rewrite_offsets,save,resign
import cpp_easygraph as cpp

HERE=Path(__file__).resolve().parent
@pytest.fixture(autouse=True)
def exact_candidate_binary():
    assert Path(cpp.__file__).resolve().parent == HERE/'native'

@pytest.mark.parametrize('wide',[False,True])
@pytest.mark.parametrize('ownership',['range','explicit'])
@pytest.mark.parametrize('policy',['full','structural'])
def test_compiled_import_exports_actual_offsets_and_payload_bytes(tmp_path,wide,ownership,policy):
    path=fixture(tmp_path/'g',ownership_encoding=ownership,weighted=True)
    if wide:widen(path)
    doc=read(path);g=cpp.cpp_egmgpu_load_partition_manifest(str(path),policy)
    for i,spec in enumerate(doc['partitions']):
        p=g.partition(i);item=spec['files']['offsets']
        expected=np.fromfile(path.parent/item['path'],dtype='<i8' if wide else '<i4').tolist()
        assert p.offsets==expected and p.offset_dtype==item['dtype']
        assert p.host_storage['offsets']['bytes']==item['byte_size']
        assert p.offsets_sha256==item['sha256']
    assert sum(x['bytes_read'] for p in g.import_profile['partitions'] for x in p['arrays'])==sum(x['byte_size'] for p in doc['partitions'] for x in p['files'].values())

@pytest.mark.parametrize('wide',[False,True])
@pytest.mark.parametrize('policy',['deep','shared'])
def test_compiled_weight_clone_preserves_active_width_and_counts_real_bytes(tmp_path,monkeypatch,wide,policy):
    path=fixture(tmp_path/'g',weighted=True)
    document=read(path)
    for part in document['partitions']:
        item=part['files']['weights']
        raw=np.fromfile(path.parent/item['path'],dtype='<f4' if item['dtype']=='float32' else '<f8').astype('<f8').tobytes()
        (path.parent/item['path']).write_bytes(raw)
        item.update(dtype='float64',byte_size=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    save(path,resign(document))
    if wide:widen(path)
    doc=read(path);g=cpp.cpp_egmgpu_load_partition_manifest(str(path));old=[g.partition(i).offsets for i in range(g.partition_count)]
    monkeypatch.setenv('EGGPU_HOST_SNAPSHOT_STORAGE',policy)
    committed=cpp.cpp_egmgpu_commit_weights_memory(g,'wide-clone',[(0,0,7.25)])
    target=committed['graph']
    assert g.topology_signature==target.topology_signature and g.weight_signature!=target.weight_signature
    assert committed['topology_sha256']==target.topology_signature
    assert committed['weight_sha256']==target.weight_signature
    assert committed['view_signature']==target.view_signature
    topology=sum(x['byte_size'] for p in doc['partitions'] for key,x in p['files'].items() if key!='weights')
    allbytes=sum(x['byte_size'] for p in doc['partitions'] for x in p['files'].values())
    profile=target.native_commit_profile
    if policy=='shared':assert profile['shared_topology_payload_bytes']==topology
    else:assert profile['copied_array_payload_bytes']==allbytes
    for i in range(g.partition_count):
        a,b=g.partition(i),target.partition(i)
        assert a.offsets==b.offsets==old[i] and a.offset_dtype==b.offset_dtype==('int64' if wide else 'int32')
        assert (a.host_storage['offsets']['allocation_id']==b.host_storage['offsets']['allocation_id'])==(policy=='shared')
        assert a.host_storage['offsets']['bytes']==b.host_storage['offsets']['bytes']==doc['partitions'][i]['files']['offsets']['byte_size']
    exported=target.partition(0).offsets
    del a,b,committed,target,g
    gc.collect();assert exported==old[0]

@pytest.mark.parametrize('entry,args',[
 ('cpp_egmgpu_partitioned_bfs',(0,)),('cpp_egmgpu_partitioned_bfs_serial',(0,)),
 ('cpp_egmgpu_partitioned_pagerank',()),('cpp_egmgpu_partitioned_pagerank_serial',()),
 ('cpp_egmgpu_partitioned_sssp',(0,)),('cpp_egmgpu_partitioned_sssp_serial',(0,)),
])
def test_narrow_operator_rejects_wide_before_runtime_or_cache(tmp_path,entry,args):
    path=fixture(tmp_path/'g',weighted=True);widen(path);g=cpp.cpp_egmgpu_load_partition_manifest(str(path))
    for _ in range(2):
        with pytest.raises((ValueError,RuntimeError),match='INDEX_UNSUPPORTED.*int32 offsets'):
            getattr(cpp,entry)(g,*args)

@pytest.mark.parametrize('values',[[0,-1,2],[0,2,1],[0,1,2**31]])
def test_structural_native_import_rejects_invalid_i64_offsets(tmp_path,values):
    path=fixture(tmp_path/'g');widen(path);doc=read(path);count=doc['partitions'][0]['files']['offsets']['count']
    # Keep exact declared length and test native structural validation, with a
    # valid SHA of the invalid complete payload, never a truncated file.
    row=np.arange(count,dtype=np.int64);row[-len(values):]=values
    rewrite_offsets(path,0,row)
    with pytest.raises((ValueError,RuntimeError)):
        cpp.cpp_egmgpu_load_partition_manifest(str(path),'structural')
