"""Version-bound typed journal columns. Hot consumers never iterate rows."""
from dataclasses import dataclass
from collections import Counter
import sys
import weakref
from time import perf_counter
import numpy as np
from easygraph.gpu._arena.round4_switch.journal import CHANGE_DTYPE, EdgeChange, retained_footprint


@dataclass(frozen=True, eq=False)
class TransitionColumns:
    ref: object
    base_version: int
    version: int
    data: np.ndarray
    source_state: object
    src: np.ndarray
    dst: np.ndarray
    base: np.ndarray
    final: np.ndarray
    flags: np.ndarray
    has_deletions: bool
    has_topology: bool
    new_count: int
    source_rows: int

    @classmethod
    def from_journal(cls, journal, source_state=None):
        data = journal.data
        if data.dtype != CHANGE_DTYPE or data.ndim != 1 or data.flags.writeable:
            raise ValueError('Immutable canonical journal columns required')
        src, dst, base, final, flags = (data[k] for k in ('src','dst','base','final','flags'))
        inserted = (flags & 1) == 0
        deleted = bool(np.any((flags & 2) == 0))
        count = int(np.count_nonzero(inserted))
        rows = int(bool(len(src))) + int(np.count_nonzero(src[1:] != src[:-1]))
        return cls(journal.ref, journal.base_version, journal.version, data,
                   weakref.ref(source_state) if source_state is not None else None,
                   src, dst, base, final, flags, deleted, deleted or bool(count), count, rows)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        raise TypeError('Typed transitions require column consumers; use legacy() only for explicit compatibility/oracles')

    def legacy(self):
        return tuple(EdgeChange(int(s),int(d),float(b) if f&1 else None,
                                float(v) if f&2 else None)
                     for s,d,b,v,f in zip(self.src,self.dst,self.base,self.final,self.flags))


def topology(changes):
    return changes.has_topology if isinstance(changes,TransitionColumns) else any(
        c.base_weight is None or c.final_weight is None for c in changes)


def deletion(changes):
    return changes.has_deletions if isinstance(changes,TransitionColumns) else any(c.final_weight is None for c in changes)


def insert_count(changes):
    return changes.new_count if isinstance(changes,TransitionColumns) else sum(c.base_weight is None for c in changes)


def source_count(changes):
    return changes.source_rows if isinstance(changes,TransitionColumns) else len({c.src for c in changes})


class ColumnsCache:
    """One optional descriptor; conservative admission charges its whole owner.

    The cache budget includes metadata and the entire structured owner, even
    while a pending journal also owns it. Store-wide reporting deduplicates it.
    Constant-size metadata traversal replaces per-edge Python admission work.
    """
    def __init__(self,budget,created):
        if isinstance(budget,bool) or not isinstance(budget,(int,np.integer)) or budget<0:
            raise ValueError('Nonnegative integer transition budget required')
        self.budget=int(budget);self.entry=None;self.live_charge=0;self.peak_charge=0
        self.count=Counter();self.created=created

    def clear(self,reason='explicit'):
        if self.entry is not None:
            self.entry=None;self.live_charge=0;self.count[reason]+=1

    def prune(self,journals):
        entry=self.entry
        if entry is not None:
            journal=entry['journal']()
            if journal is None or journals.get(entry['ref']) is not journal:
                self.clear('superseded')

    def get(self,journal,source_state=None):
        if journal is None:return ()
        started=perf_counter();entry=self.entry
        if entry is not None and entry['journal']() is journal and (
                entry['changes'].source_state is None and source_state is None or
                entry['changes'].source_state is not None and entry['changes'].source_state() is source_state):
            self.count['hits']+=1;self.count['lookup_seconds']+=perf_counter()-started
            return entry['changes']
        self.count['lookup_seconds']+=perf_counter()-started;self.count['misses']+=1
        self.clear('replaced');started=perf_counter()
        changes=TransitionColumns.from_journal(journal,source_state)
        self.count['column_preparation_seconds']+=perf_counter()-started
        self.count['prepared_journals']+=1;self.count['prepared_edges']+=len(changes)
        self.created(changes)
        if self.budget:
            started=perf_counter()
            owner=weakref.ref(self)
            def expire(token):
                cache=owner()
                if cache is not None and cache.entry is not None and cache.entry['journal'] is token:
                    cache.clear('weak_expiry')
            entry=dict(journal=weakref.ref(journal,expire),ref=journal.ref,changes=changes)
            charge=retained_footprint(entry)['total_bytes']
            self.count['admission_seconds']+=perf_counter()-started
            self.count['candidate_charge_peak_bytes']=max(self.count['candidate_charge_peak_bytes'],charge)
            if charge<=self.budget:
                self.entry=entry;self.live_charge=charge;self.peak_charge=max(self.peak_charge,charge)
                self.count['admitted']+=1
            else:self.count['budget_rejections']+=1
        return changes

    def stats(self):
        exact=retained_footprint(self.entry) if self.entry is not None else dict(payload_bytes=0,metadata_bytes=0,total_bytes=0,objects=0)
        assert exact['total_bytes']<=self.live_charge<=self.budget
        return dict(self.count,budget_bytes=self.budget,entries=int(self.entry is not None),
                    retained_charge_bytes=self.live_charge,retained_charge_peak_bytes=self.peak_charge,
                    retained_payload_bytes=exact['payload_bytes'],retained_metadata_bytes=exact['metadata_bytes'],
                    scope='Constant-object typed descriptor; conservative admission includes the complete immutable 33-byte-per-edge journal owner plus reachable metadata. Store union deduplicates pending-journal and certificate ownership.')
