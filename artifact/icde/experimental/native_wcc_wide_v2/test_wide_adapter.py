"""Actual partition builder equivalence; scaled index limit exercises wide IO."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'runner'))
import native_input as ni
from easygraph.gpu._manifest_signatures import sign
from easygraph.egmgpu_control.partition_manifest import load_partition_manifest

@pytest.mark.parametrize('count',[1,2,4,8])
def test_actual_roundtrip_for_wide_and_original_narrow_geometry(tmp_path,monkeypatch,count):
    n=16;rows=[];off=[0]
    for v in range(n):
        rows.extend(sorted({(v-1)%n,(v+1)%n}));off.append(len(rows))
    original=tmp_path/'original';original.mkdir()
    offsets=np.array(off,dtype='<i4');indices=np.array(rows,dtype='<i4')
    offsets.tofile(original/'offsets.bin');indices.tofile(original/'indices.bin')
    graph=dict(name='ring16',n=n,m=len(rows),offsets=offsets,indices=indices,weights=None,directed=False,
               paths={'offsets':original/'offsets.bin','indices':original/'indices.bin'})
    normal=ni.native_partitions(graph,tmp_path/'narrow',count,sign,allow_wide_offsets=True)
    assert normal['offset_dtype']=='int32'
    # A reduced test-only capacity exercises identical i64 writer/reader code
    # with a real tiny payload. The separate metadata tests cover 2**31.
    monkeypatch.setattr(ni,'MAX_LOCAL_ENTRIES',1)
    wide=ni.native_partitions(graph,tmp_path/'wide',count,sign,allow_wide_offsets=True)
    assert wide['offset_dtype']=='int64'
    assert ni.audit_partitions(graph,wide['path'])['status']=='PASS_FULL_PARTITION_ROUNDTRIP'
    parsed=load_partition_manifest(wide['path']);assert all(p.offsets.dtype=='int64' for p in parsed.partitions)
    a=json.loads(Path(normal['path']).read_text());b=json.loads(Path(wide['path']).read_text())
    assert a['graph']['adjacency_count']==b['graph']['adjacency_count']
    for pa,pb in zip(a['partitions'],b['partitions']):
        for key in ('indices','local_to_global','owners'):
            assert pa['files'][key]==pb['files'][key]

@pytest.mark.parametrize('edges,expected',[(2**31-1,'int32'),(2**31,'int64'),(2**32+17,'int64')])
def test_exact_metadata_selection_without_large_allocation(edges,expected):
    offsets=np.array([0,edges],dtype='<i8')
    assert ni.choose_offset_dtype(offsets,[0,1],True)==expected
    if expected=='int64':
        with pytest.raises(ni.IndexUnsupported):ni.choose_offset_dtype(offsets,[0,1],False)

def test_same_global_wide_graph_selects_narrow_when_each_partition_fits():
    offsets=np.array([0,2**30,2**31],dtype='<i8')
    assert ni.choose_offset_dtype(offsets,[0,2],True)=='int64'
    assert ni.choose_offset_dtype(offsets,[0,1,2],True)=='int32'
