"""Bounded preparation of immutable journal changes, shared by peek and flush.

The authority and automatic pressure publication retain their frozen code.
Column decoding changes representation, never transaction semantics. One weak
journal entry is admitted under a metadata-inclusive conservative byte charge.
Exact identity-deduplicated metadata is available from stats().
"""
from collections import Counter
import sys
from time import perf_counter
import weakref
import numpy as np

from easygraph.gpu._arena.round4_switch.journal import EdgeChange, FlushResult, retained_footprint
from easygraph.gpu._arena.round8_host.host_authority import Store
from easygraph.gpu._arena.round8_cold.authority import ColdStore

DEFAULT_CHANGE_BUDGET = 8 << 20


def column_changes(journal):
    if journal is None:
        return ()
    data = journal.data
    return tuple(EdgeChange(src, dst, base if flags & 1 else None,
                            final if flags & 2 else None)
                 for src, dst, base, final, flags in zip(
                     data['src'].tolist(), data['dst'].tolist(),
                     data['base'].tolist(), data['final'].tolist(), data['flags'].tolist()))


def admission_charge(entry):
    """Conservative owned-object charge; sharing can only lower exact bytes.

    Includes each dataclass and its actual dictionary header, an int64-domain
    upper bound for both endpoint integers and float-size bounds for weights,
    tuple/header and entry/ref/version metadata. Repeated scalar identities and
    None weights are conservatively overcharged. Exact deduplicated metadata is
    independently exposed by stats; no ndarray-only budget is claimed.
"""
    changes = entry['changes']
    if not changes:
        return retained_footprint(entry)['total_bytes']
    shallow = dict(entry, changes=())
    total = retained_footprint(shallow)['total_bytes'] - sys.getsizeof(()) + sys.getsizeof(changes)
    # Field-name strings are shared between the split instance dictionaries.
    total += sum(sys.getsizeof(key) for key in changes[0].__dict__)
    # All rows use the frozen EdgeChange schema, signed-int64 labels and
    # optional float64 weights. Their scalar-size bound is computable once;
    # only instance-dictionary headers can vary as Python shares their keys.
    integer_bound = max(sys.getsizeof(-(1 << 63)), sys.getsizeof((1 << 63)-1))
    value_bound = max(sys.getsizeof(0.), sys.getsizeof(None))
    total += len(changes) * (sys.getsizeof(changes[0])+2*integer_bound+2*value_bound)
    total += sum(sys.getsizeof(change.__dict__) for change in changes)
    return total


class ChangeCache:
    def __init__(self, budget):
        if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 0:
            raise ValueError('Nonnegative integer change metadata budget required')
        self.budget = int(budget)
        self.entry = None
        self.live_charge = self.peak_charge = 0
        self.count = Counter()
        owner = weakref.ref(self)
        def expire(token):
            cache = owner()
            if cache is not None and cache.entry is not None and cache.entry['journal'] is token:
                cache.clear('weak_expiry')
        self.expire = expire

    def clear(self, reason='explicit'):
        if self.entry is not None:
            started = perf_counter()
            self.entry = None
            self.live_charge = 0
            self.count[reason] += 1
            self.count['eviction_seconds'] += perf_counter()-started

    def prune(self, journals):
        entry = self.entry
        if entry is not None and journals.get(entry['ref']) is not entry['journal']():
            self.clear('superseded')

    def get(self, journal):
        if journal is None:
            return ()
        started = perf_counter()
        entry = self.entry
        if (entry is not None and entry['journal']() is journal and
                entry['ref'] == journal.ref and entry['version'] == journal.version and
                entry['base_version'] == journal.base_version):
            self.count['hits'] += 1
            self.count['lookup_seconds'] += perf_counter()-started
            return entry['changes']
        self.count['lookup_seconds'] += perf_counter()-started
        self.count['misses'] += 1
        self.clear('replaced')
        started = perf_counter()
        changes = column_changes(journal)
        self.count['decode_seconds'] += perf_counter()-started
        self.count['decoded_journals'] += 1
        self.count['decoded_edges'] += len(changes)
        if not changes or self.budget == 0:
            self.count['uncached_decodes'] += 1
            return changes
        started = perf_counter()
        entry = dict(journal=weakref.ref(journal, self.expire), ref=journal.ref,
                     version=journal.version, base_version=journal.base_version, changes=changes)
        charge = admission_charge(entry)
        self.count['admission_seconds'] += perf_counter()-started
        self.count['candidate_charge_peak_bytes'] = max(self.count['candidate_charge_peak_bytes'], charge)
        if charge <= self.budget:
            self.entry = entry
            self.live_charge = charge
            self.peak_charge = max(self.peak_charge, charge)
            self.count['admitted'] += 1
        else:
            self.count['budget_rejections'] += 1
        return changes

    def stats(self):
        # This explicit observation is O(number of decoded rows), separate from
        # the cheaper conservative admission ledger and never needed for a hit.
        exact = retained_footprint(self.entry) if self.entry is not None else dict(
            payload_bytes=0, metadata_bytes=0, total_bytes=0, objects=0)
        assert exact['payload_bytes'] == 0
        assert exact['total_bytes'] <= self.live_charge <= self.budget
        return dict(self.count, budget_bytes=self.budget, entries=int(self.entry is not None),
                    retained_charge_bytes=self.live_charge, retained_charge_peak_bytes=self.peak_charge,
                    retained_metadata_bytes=exact['metadata_bytes'], retained_payload_bytes=0,
                    controller_shallow_bytes=sys.getsizeof(self)+sys.getsizeof(self.__dict__)+
                        sys.getsizeof(self.count)+sys.getsizeof(self.expire),
                    scope='Budget covers conservative reachable entry, tuple, EdgeChange/dict and Python scalar metadata; '
                          'exact metadata deduplicates identities. Weakrefs are opaque and never traverse journals. '
                          'Fixed controller, decoder transient column lists and allocator/RSS are separate.')


class ChangePreparationMixin:
    def __init__(self, *args, change_budget_bytes=DEFAULT_CHANGE_BUDGET, **kwargs):
        self.change_cache = ChangeCache(change_budget_bytes)
        super().__init__(*args, **kwargs)

    def peek_pending(self, ref):
        with self._lock:
            self._record(ref)
            self.change_cache.prune(self._journals)
            return self.change_cache.get(self._journals.get(ref))

    def flush(self, ref):
        # Same fallible work and publication order as JournalStore.flush.
        with self._lock:
            record = self._record(ref)
            journal = self._journals.get(ref)
            self.change_cache.prune(self._journals)
            if journal is None and record.snapshot is not None and record.snapshot.version == record.version:
                return FlushResult(ref, record.snapshot, record.snapshot, (), True)
            changes = self.change_cache.get(journal)
            edges = self._logical(record, journal)
            plan = self._build(ref, record.nodes, edges, record.snapshot, changes, record.version)
            journals = {key: value for key, value in self._journals.items() if key != ref}
            record.base_edges = edges
            record.base_version = record.version
            record.snapshot = plan.new_snapshot
            self._journals = journals
            self._counters['explicit_flushes'] += 1
            self.change_cache.prune(self._journals)
            return plan

    def commit(self, ref, mutations):
        with self._lock:
            try:
                return super().commit(ref, mutations)
            finally:
                self.change_cache.prune(self._journals)

    def replace(self, ref, *args, **kwargs):
        with self._lock:
            try:
                return super().replace(ref, *args, **kwargs)
            finally:
                self.change_cache.prune(self._journals)

    def stats(self):
        with self._lock:
            self.change_cache.prune(self._journals)
            # The tuple passed to _build is the same tuple stored in the cache.
            # If a certificate concurrently owns it, a joint identity walk
            # counts it once. Do not sum separate cache and certificate tables.
            members = [self.index.entries, self.certificates]
            if self.change_cache.entry is not None:
                members.append(self.change_cache.entry)
            union = retained_footprint(tuple(members))
            union['metadata_bytes'] -= sys.getsizeof(tuple(members))
            union['total_bytes'] -= sys.getsizeof(tuple(members))
            return dict(super().stats(), prepared_changes=self.change_cache.stats(),
                        prepared_host_union=dict(union, scope='Identity-deduplicated index entries, certificates '
                            'and decoded-change cache. Shared EdgeChange tuple/scalars count once; weak owners '
                            'are opaque. Fixed controllers, logical journal authority, CSR and RSS separate.'))


class ChangesStore(ChangePreparationMixin, Store):
    """Decoded changes alone, with the frozen original initialization."""


class PreparedStore(ChangePreparationMixin, ColdStore):
    """Explicit composition of qualified cold preparation and changes reuse."""
