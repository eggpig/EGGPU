import numpy as np
import pytest
import hotspot_positions as h


def csr(n,pairs):
    pairs=sorted(set(pairs));off=np.r_[0,np.cumsum(np.bincount([u for u,v in pairs],minlength=n))].astype('<i4')
    ind=np.array([v for u,v in pairs],dtype='<i4');return off,ind


@pytest.fixture
def graph():
    n=64;pairs={(u,v) for u in range(n) for v in range(n) if u!=v and u//8==v//8}
    pairs|={(u,v) for u in range(n) for v in range(n) if u%8==v%8 and u//8!=v//8}
    return csr(n,pairs)


@pytest.mark.parametrize('k',[1,2,4,8])
@pytest.mark.parametrize('directed',[True,False])
def test_exact_count_owner_coverage_and_reverse_pairs(graph,k,directed):
    off,ind=graph;cuts=np.linspace(0,len(off)-1,9,dtype=np.int64);rho=.03
    pos,record=h.select(off,ind,directed,off[cuts],cuts,k,rho,chunk=73)
    assert record['status']=='PASS_CPU_SELECTION'
    assert record['affected_owners']==list(range(k))
    count=int(len(ind)*rho);count=count if directed else 2*(count//2)
    assert len(pos)==count==len(set(pos))
    rows=np.searchsorted(off,pos.astype(off.dtype),side='right')-1
    pairs={(int(u),int(ind[p])) for p,u in zip(pos,rows)}
    if not directed:
        assert all((v,u) in pairs and v//8<k for u,v in pairs)
    again,_=h.select(off,ind,directed,off[cuts],cuts,k,rho,chunk=51)
    np.testing.assert_array_equal(pos,again)


def test_insufficient_capacity_returns_no_partial_selection(graph):
    off,ind=graph;cuts=np.linspace(0,64,9,dtype=np.int64)
    pos,record=h.select(off,ind,False,off[cuts],cuts,1,1.)
    assert len(pos)==0 and record['status']=='INFEASIBLE'


def test_no_cross_owner_undirected_updates():
    n=16;off,ind=csr(n,[(0,2),(2,0),(1,3),(3,1)])
    cuts=np.linspace(0,n,9,dtype=np.int64)
    pos,record=h.select(off,ind,False,off[cuts],cuts,1,.5)
    assert len(pos)==0 and record['status']=='INFEASIBLE'
    assert record['eligible_logical_edges']==0


def test_partition_without_edges_is_infeasible():
    n=16;off,ind=csr(n,[(0,1),(1,0)]);cuts=np.linspace(0,n,9,dtype=np.int64)
    pos,record=h.select(off,ind,False,off[cuts],cuts,8,1.)
    assert len(pos)==0 and record['status']=='INFEASIBLE'


def test_invalid_edge_partition_rejected(graph):
    off,ind=graph;cuts=np.linspace(0,64,9,dtype=np.int64);edges=off[cuts].copy();edges[1]+=1
    with pytest.raises(ValueError):h.select(off,ind,False,edges,cuts,1,.01)


def test_broken_reverse_edge_rejected():
    off,ind=csr(16,[(0,1)]);cuts=np.linspace(0,16,9,dtype=np.int64)
    # Force the paired selector to include the missing reverse with two logical edges.
    off,ind=csr(16,[(0,1),(0,1),(0,0)])
    with pytest.raises(ValueError,match='reverse'):
        h.select(off,ind,False,off[cuts],cuts,1,1.)
