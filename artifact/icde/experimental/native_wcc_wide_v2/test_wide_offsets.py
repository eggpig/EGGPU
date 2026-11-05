"""Independent CPU boundary tests for the WCC-only wide-offset candidate.

Run with the candidate Easy-Graph first on PYTHONPATH. No GPU backend is
constructed. Large limits are metadata-only: these tests never allocate a
2^31-entry array. Compiled loader/clone/narrow-dispatch tests should reuse
``fixture`` / ``widen`` after the candidate DSO is built, in a separate suite.
"""
from pathlib import Path
from dataclasses import replace
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pytest
from easygraph.egmgpu_control import partition_manifest as pm
from easygraph.gpu import _manifest_signatures as signatures

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'Easy-Graph'
import os
BASE=Path(os.environ['EGGPU_BASE_SOURCE'])


@pytest.fixture(autouse=True)
def candidate_module_identity():
    assert Path(pm.__file__).resolve().is_relative_to(SOURCE.resolve()), 'Use the exact wide candidate PYTHONPATH'


def read(path):return json.loads(Path(path).read_text())


def save(path,document):
    Path(path).write_text(json.dumps(document,indent=2)+'\n')


def resign(document):
    topology,weight,view=signatures.sign(document)
    document['graph']['topology_sha256']=topology
    if weight is not None:document['graph']['weight_sha256']=weight
    document['view']['signature']=view
    return document


def fixture(root,**kwargs):
    return pm.build_tiny_partition_fixture(root,**kwargs)


def widen(path,selected=None,*,signed=True):
    """Turn the small real fixture into a semantically identical explicit i64 view."""
    path=Path(path);document=read(path);old=read(path)
    document['indexing']['offset_dtype']='int64'
    chosen=set(range(len(document['partitions']))) if selected is None else set(selected)
    for index,part in enumerate(document['partitions']):
        if index not in chosen:continue
        item=part['files']['offsets'];values=np.fromfile(path.parent/item['path'],dtype='<i4').astype('<i8')
        payload=values.tobytes();name=item['path'].replace('.i32','.i64')
        if name==item['path']:name+='.i64'
        (path.parent/name).write_bytes(payload)
        item.update(path=name,dtype='int64',byte_size=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    save(path,resign(document) if signed else document)
    return old,document


def rewrite_offsets(path,index,values,*,endian='<'):
    document=read(path);item=document['partitions'][index]['files']['offsets']
    payload=np.asarray(values,dtype=endian+'i8').tobytes();(path.parent/item['path']).write_bytes(payload)
    item.update(byte_size=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    save(path,resign(document))


def metadata_only(adjacency,dtype='int64',owned=1,halo=0):
    def array(name,kind,count):
        size={'int32':4,'int64':8,'uint64':8}[kind]
        return pm.ArrayFile(name,kind,count,count*size,'0'*64)
    return pm.ManifestPartitionMetadata(partition_id=0,device_id=0,ownership_encoding='range',
        owned_vertex_count=owned,halo_vertex_count=halo,adjacency_count=adjacency,cut_edge_count=0,
        offsets=array('offsets',dtype,owned+1),indices=array('indices','int32',adjacency),
        local_to_global=array('global','uint64',owned+halo),owners=array('owners','int32',owned+halo),
        vertex_begin=0,vertex_end=owned)


@pytest.mark.parametrize('adjacency',[2**31-1,2**31,2**31+1,2**32+17])
def test_i64_adjacency_metadata_crosses_signed_i32_without_allocation(adjacency):
    part=metadata_only(adjacency)
    assert part.adjacency_count==adjacency and part.offsets.dtype=='int64'
    assert part.offsets.byte_size==16 and part.indices.count==adjacency


@pytest.mark.parametrize('adjacency',[2**31,2**31+1,2**32+17])
def test_i32_metadata_still_rejects_unrepresentable_terminal_offset(adjacency):
    with pytest.raises(pm.ManifestError):metadata_only(adjacency,'int32')


def test_i32_metadata_boundary_remains_accepted():
    assert metadata_only(2**31-1,'int32').adjacency_count==2**31-1


def test_wide_offsets_do_not_expand_local_vertex_id_domain():
    with pytest.raises(pm.ManifestError):metadata_only(1,owned=2**31,halo=1)


def test_empty_partition_allows_one_i64_zero_offset_metadata():
    part=metadata_only(0,owned=0);assert part.offsets.count==1 and part.indices.count==0
    with pytest.raises(pm.ManifestError):metadata_only(1,owned=0)


@pytest.mark.parametrize('ownership',['range','explicit'])
@pytest.mark.parametrize('weighted',[False,True])
def test_real_i64_files_roundtrip_and_preserve_weight_domain(tmp_path,ownership,weighted):
    path=fixture(tmp_path/'graph',ownership_encoding=ownership,weighted=weighted)
    old,new=widen(path);manifest=pm.load_partition_manifest(path)
    assert all(p.offsets.dtype=='int64' for p in manifest.partitions)
    assert manifest.adjacency_count==old['graph']['adjacency_count']
    assert manifest.topology_sha256!=old['graph']['topology_sha256']
    assert manifest.view_signature!=old['view']['signature']
    assert new['graph'].get('weight_sha256')==old['graph'].get('weight_sha256')
    for part in manifest.partitions:
        original=old['partitions'][part.partition_id]['files']['offsets']
        np.testing.assert_array_equal(list(part.offsets.values(path.parent)),np.fromfile(path.parent/original['path'],dtype='<i4'))
    roundtrip=pm._manifest_document(manifest)
    assert roundtrip['indexing']['offset_dtype']=='int64'
    assert roundtrip==new
    save(path,roundtrip);assert pm.load_partition_manifest(path).view_signature==manifest.view_signature


def test_mixed_partition_width_cannot_hide_under_global_i64(tmp_path):
    path=fixture(tmp_path/'graph');widen(path,selected=[0])
    with pytest.raises(pm.ManifestError):pm.load_partition_manifest(path)


def test_global_i32_cannot_hide_i64_partition(tmp_path):
    path=fixture(tmp_path/'graph');widen(path);document=read(path);document['indexing']['offset_dtype']='int32';save(path,document)
    with pytest.raises(pm.ManifestError):pm.load_partition_manifest(path)


def test_global_i64_cannot_reinterpret_i32_payload(tmp_path):
    path=fixture(tmp_path/'graph');document=read(path);document['indexing']['offset_dtype']='int64';save(path,document)
    with pytest.raises(pm.ManifestError):pm.load_partition_manifest(path)


def test_widening_requires_updated_topology_and_view_signatures(tmp_path):
    path=fixture(tmp_path/'graph');widen(path,signed=False)
    with pytest.raises(pm.ManifestError):pm.load_partition_manifest(path)


def test_stale_payload_hash_rejected(tmp_path):
    path=fixture(tmp_path/'graph');widen(path);document=read(path);item=document['partitions'][0]['files']['offsets']
    file=path.parent/item['path'];payload=bytearray(file.read_bytes());payload[8]^=1;file.write_bytes(payload)
    with pytest.raises(pm.ManifestFileError):pm.load_partition_manifest(path)


@pytest.mark.parametrize('failure',['negative','descending','too_large_terminal','nonzero_start','big_endian_payload'])
def test_signed_correctly_hashed_invalid_i64_contents_are_rejected(tmp_path,failure):
    path=fixture(tmp_path/'graph');widen(path);document=read(path);item=document['partitions'][0]['files']['offsets']
    values=np.fromfile(path.parent/item['path'],dtype='<i8').copy()
    if failure=='negative':values[1]=-1
    elif failure=='descending':values[1]=values[-1]+1
    elif failure=='too_large_terminal':values[-1]=2**31+3
    elif failure=='nonzero_start':values[0]=1
    rewrite_offsets(path,0,values,endian='>' if failure=='big_endian_payload' else '<')
    with pytest.raises(pm.ManifestError):pm.load_partition_manifest(path)


@pytest.mark.parametrize('change',[
    lambda d:d['indexing'].update(offset_dtype='uint64'),
    lambda d:d['indexing'].update(endianness='big'),
    lambda d:d['indexing'].update(index_dtype='int64'),
    lambda d:d['partitions'][0]['files']['offsets'].update(byte_size=4*d['partitions'][0]['files']['offsets']['count']),
])
def test_dtype_and_byte_width_are_exact_not_implicitly_cast(tmp_path,change):
    path=fixture(tmp_path/'graph');widen(path);document=read(path);change(document);save(path,resign(document))
    with pytest.raises(pm.ManifestError):pm.load_partition_manifest(path)


@pytest.fixture(scope='module')
def old_manifest_module():
    name='easygraph.egmgpu_control._independent_d84_manifest'
    loader=importlib.util.spec_from_file_location(name,BASE/'easygraph/egmgpu_control/partition_manifest.py')
    module=importlib.util.module_from_spec(loader);sys.modules[name]=module;loader.loader.exec_module(module)
    return module


@pytest.mark.parametrize('ownership',['range','explicit'])
@pytest.mark.parametrize('weighted',[False,True])
def test_original_i32_fixture_bytes_and_signatures_stay_identical(tmp_path,old_manifest_module,ownership,weighted):
    old=old_manifest_module.build_tiny_partition_fixture(tmp_path/'old',ownership_encoding=ownership,weighted=weighted)
    new=fixture(tmp_path/'new',ownership_encoding=ownership,weighted=weighted)
    assert old.read_bytes()==new.read_bytes()
    old_files={str(p.relative_to(old.parent)):p.read_bytes() for p in old.parent.rglob('*') if p.is_file()}
    new_files={str(p.relative_to(new.parent)):p.read_bytes() for p in new.parent.rglob('*') if p.is_file()}
    assert old_files==new_files
    assert pm.load_partition_manifest(new).view_signature==old_manifest_module.load_partition_manifest(old).view_signature
