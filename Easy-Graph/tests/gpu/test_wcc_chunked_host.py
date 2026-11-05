"""CPU-only WCC validation/certificate tests across bounded scratch blocks."""
import numpy as np
import pytest
from easygraph.gpu import _core as core


def reference(labels):
    roots={}
    for i,label in enumerate(labels):roots.setdefault(int(label),i)
    return np.array([roots[int(label)] for label in labels],dtype=np.float64)


@pytest.mark.parametrize('dtype',[np.int32,np.int64,np.uint32,np.uint64,np.float32,np.float64])
@pytest.mark.parametrize('certify',[False,True])
def test_chunked_groups_match_independent_dictionary(monkeypatch,dtype,certify):
    monkeypatch.setattr(core,'_WCC_CANONICAL_CHUNK',7)
    rng=np.random.default_rng(20261104)
    for n in [0,1,2,7,8,17,33,129]:
        labels=rng.integers(0,n,size=n,dtype=np.int64).astype(dtype) if n else np.array([],dtype=dtype)
        before=labels.copy();result=core.canonical_wcc(labels,n,certify_minimum=certify)
        np.testing.assert_array_equal(result,reference(labels))
        np.testing.assert_array_equal(labels,before)
        assert result.dtype==np.float64 and result.flags.owndata and not np.shares_memory(result,labels)


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf,-1.,19.,.5])
@pytest.mark.parametrize('position',[0,6,7,8,14,18])
def test_invalid_labels_in_every_chunk_are_rejected(monkeypatch,bad,position):
    monkeypatch.setattr(core,'_WCC_CANONICAL_CHUNK',7)
    labels=np.zeros(19,dtype=np.float64)
    labels[0]=1 # makes the certificate fail early, but all later values still need validation
    labels[position]=bad
    with pytest.raises(ValueError,match='exact dense indices'):
        core.canonical_wcc(labels,len(labels),certify_minimum=True)


def test_cross_block_representative_and_fallback(monkeypatch):
    monkeypatch.setattr(core,'_WCC_CANONICAL_CHUNK',3)
    # Every group points at a real member, but that member is not its minimum.
    labels=np.array([6,7,8,6,7,8,6,7,8,9],dtype=np.int64)
    before=core._WCC_CANONICAL_CERTIFICATE_AUDIT.copy()
    np.testing.assert_array_equal(core.canonical_wcc(labels,10,certify_minimum=True),reference(labels))
    assert core._WCC_CANONICAL_CERTIFICATE_AUDIT['fallback']==before['fallback']+1
    roots=reference(labels)
    before=core._WCC_CANONICAL_CERTIFICATE_AUDIT.copy()
    result=core.canonical_wcc(roots,10,certify_minimum=True)
    np.testing.assert_array_equal(result,roots)
    assert core._WCC_CANONICAL_CERTIFICATE_AUDIT['certified']==before['certified']+1
    result[0]=123;assert roots[0]==0


def test_noncontiguous_input_and_maximum_unsigned(monkeypatch):
    monkeypatch.setattr(core,'_WCC_CANONICAL_CHUNK',2)
    labels=np.array([1,0,1,0,5,5,5,0],dtype=np.int64)[::2]
    # use an in-domain label on the strided input
    labels[labels==5]=3
    np.testing.assert_array_equal(core.canonical_wcc(labels,4,certify_minimum=True),reference(labels))
    with pytest.raises(ValueError,match='exact dense indices'):
        core.canonical_wcc(np.array([0,2**64-1],dtype=np.uint64),2,certify_minimum=True)


def test_real_chunk_boundary_with_isolates_and_late_invalid():
    n=core._WCC_CANONICAL_CHUNK+3
    labels=np.arange(n,dtype=np.float64)
    result=core.canonical_wcc(labels,n,certify_minimum=True)
    np.testing.assert_array_equal(result,labels)
    labels[-1]=np.nan
    with pytest.raises(ValueError,match='exact dense indices'):
        core.canonical_wcc(labels,n,certify_minimum=True)


def test_domain_error_precedes_invalid_mode(monkeypatch):
    monkeypatch.setenv('EGGPU_WCC_CANONICAL_CERTIFICATE','invalid')
    with pytest.raises(ValueError,match='exact dense indices'):
        core.canonical_wcc(np.array([np.nan]),1)
    with pytest.raises(ValueError,match='must be 0 or 1'):
        core.canonical_wcc(np.array([0.]),1)
