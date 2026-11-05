"""Binding schema regression: optional size cannot bypass content identity."""
import hashlib
from pathlib import Path
import pytest
import watch_external_formals_v4 as w
@pytest.mark.parametrize('with_size',[False,True])
def test_exact_full_sha_binding_accepts_both_supported_shapes(tmp_path,with_size):
 p=tmp_path/'payload';p.write_bytes(b'fixture');b=w.bind(p)
 if not with_size:b.pop('bytes')
 assert w.checked(b)==p
 p.write_bytes(b'tamper!')
 with pytest.raises(ValueError):w.checked(b)
@pytest.mark.parametrize('change',['negative','boolean','extra','badsha','relative'])
def test_malformed_bindings_rejected(tmp_path,change):
 p=tmp_path/'payload';p.write_bytes(b'fixture');b=w.bind(p)
 if change=='negative':b['bytes']=-1
 if change=='boolean':b['bytes']=True
 if change=='extra':b['mtime']=1
 if change=='badsha':b['sha256']='a'*63
 if change=='relative':b['path']='payload'
 with pytest.raises(ValueError):w.checked(b)
