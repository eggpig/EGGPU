"""Borrow an immutable transition's dense columns under its original identity.

The source/target object identity binds the precise CSR layout. Logical target
version and GraphRef additionally bind the still-current registration. A miss
rebuilds exactly the frozen Delta; no certificate or snapshot is retained here.
"""
from time import perf_counter
import numpy as np
from easygraph.gpu._arena.round2_topology.state import Delta
from easygraph.gpu._arena.round8_columns.columns import TransitionColumns


def resolve_certificate(store,ref,old,new,changes):
    cert=store.certificates.get(ref)
    record=store._records.get(ref.graph_id)
    if cert is None or record is None:return None
    if not (record.ref==ref==cert.ref and cert.old() is old and cert.new() is new
            and cert.changes is changes and record.snapshot is not None
            and record.snapshot.ref==ref and record.snapshot.host_state is new
            and record.snapshot.version==record.version==cert.target_version
            and 0<=cert.source_version<cert.target_version):return None
    for a,dtype in ((cert.src,np.int32),(cert.dst,np.int32),(cert.values,np.float64),(cert.inserted,np.bool_)):
        if not isinstance(a,np.ndarray) or a.shape!=(len(changes),) or a.dtype!=dtype or a.flags.writeable:return None
    return cert


def prepare_delta(store,ref,old,new,changes):
    started=perf_counter()
    if isinstance(changes,TransitionColumns):
        record=store._records.get(ref.graph_id)
        if not (changes.ref==ref and record is not None and record.ref==ref and
                record.snapshot is not None and record.snapshot.ref==ref and
                record.snapshot.host_state is new and
                record.version==record.snapshot.version==changes.version and
                0<=changes.base_version<changes.version and
                changes.source_state is not None and changes.source_state() is old and
                not changes.has_deletions):
            raise ValueError('Transition does not bind the current target and exact source layout')
    cert=resolve_certificate(store,ref,old,new,changes)
    if cert is None:
        if isinstance(changes,TransitionColumns):
            if changes.ref!=ref or changes.has_deletions:
                raise ValueError('Nondeleting current generation required for a transfer Delta')
            src=np.searchsorted(new.nodes,changes.src).astype(np.int32)
            dst=np.searchsorted(new.nodes,changes.dst).astype(np.int32)
            values=changes.final
            inserted=(changes.flags&1)==0
        else:
            src=np.searchsorted(new.nodes,[c.src for c in changes]).astype(np.int32)
            dst=np.searchsorted(new.nodes,[c.dst for c in changes]).astype(np.int32)
            values=np.asarray([c.final_weight for c in changes],np.float64)
            inserted=np.asarray([c.base_weight is None for c in changes],bool)
        borrowed=values.nbytes if isinstance(changes,TransitionColumns) else 0
        owned=src.nbytes+dst.nbytes+inserted.nbytes+(0 if borrowed else values.nbytes)
        store.certificate_count['packet_misses']+=1
    else:
        src,dst,values,inserted=cert.src,cert.dst,cert.values,cert.inserted
        borrowed=src.nbytes+dst.nbytes+values.nbytes+inserted.nbytes;owned=0
        store.certificate_count['packet_reuses']+=1
    ids=np.arange(len(changes),dtype=np.int32);rows=np.unique(src)
    owned+=ids.nbytes+rows.nbytes
    delta=Delta(src,dst,ids,values,inserted,rows)
    return delta,cert,dict(path='certificate' if cert is not None else 'reconstructed',
        seconds=perf_counter()-started,rows=len(changes),borrowed_payload_bytes=borrowed,
        owned_payload_bytes=owned,retained_extra_payload_bytes=0,
        scope='Logical view payload of returned Delta arrays (structured owner accounted in Store): borrowed existing certificate buffers and new ids/unique-source rows. No retained owner added; temporary Python lists, np.unique workspace, metadata and allocator RSS are separate.')
