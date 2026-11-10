"""Bounded input-only preparation; the qualified certificate is unchanged.

_input is copied from frozen FastColdStore with only sort/map substitutions.
Stable uint64 ordering applies to validated nonnegative uint32 endpoints.
A transient dense-label table is admitted under explicit table bytes and span
limits. Signed/sparse domains retain the original searchsorted path. All costs
are within register(); this module adds no persistent per-edge preparation.
"""
from time import perf_counter
import weakref
import numpy as np
from easygraph.gpu._arena.round8_cold.authority import ColdPreparation
from easygraph.gpu._arena.round8_cold_fast.authority import FastColdStore, lexicographic_unique
from easygraph.gpu._arena.round7_recovery.host_authority_v2 import IndexedEdgeMap
from easygraph.gpu._arena.round8_host.host_authority import payload

DEFAULT_LOOKUP_BUDGET=8<<20

class BoundedInputStore(FastColdStore):
    def __init__(self,*args,input_lut_budget_bytes=DEFAULT_LOOKUP_BUDGET,packed_input_sort=True,**kwargs):
        if isinstance(input_lut_budget_bytes,bool) or not isinstance(input_lut_budget_bytes,(int,np.integer)) or input_lut_budget_bytes<0:
            raise ValueError('Nonnegative integer temporary lookup-table budget required')
        if not isinstance(packed_input_sort,bool):raise ValueError('Boolean packed sort option required')
        self.input_lut_budget_bytes=int(input_lut_budget_bytes)
        self.packed_input_sort=packed_input_sort
        super().__init__(*args,**kwargs)

    def _temporary(self,stage,*arrays):
        actual=payload(arrays)
        self.cold_count['input_explicit_temporary_live_bytes']=actual
        self.cold_count['input_explicit_temporary_peak_bytes']=max(actual,self.cold_count['input_explicit_temporary_peak_bytes'])
        self.cold_count[stage+'_explicit_temporary_peak_bytes']=max(actual,self.cold_count[stage+'_explicit_temporary_peak_bytes'])

    def _sort_input(self,pairs):
        if self.packed_input_sort and pairs.size and int(pairs.min())>=0 and int(pairs.max())<1<<32:
            keys=pairs[:,0].astype(np.uint64);self._temporary('sort',keys)
            keys<<=np.uint64(32)
            columns=pairs[:,1].astype(np.uint64);self._temporary('sort',keys,columns)
            keys|=columns;del columns
            order=np.argsort(keys,kind='stable');self._temporary('sort',keys,order)
            self.cold_count['packed_stable_sort_calls']+=1
            del keys;self._temporary('sort',order)
            return order
        self.cold_count['lexicographic_fallback_sort_calls']+=1
        order=np.lexsort((pairs[:,1],pairs[:,0]));self._temporary('sort',order)
        return order

    def _map_input(self,nodes,pairs):
        if not len(pairs):
            self.cold_count['empty_endpoint_mapping_calls']+=1
            return np.empty((0,2),np.int32)
        if len(nodes):
            lo,hi=int(nodes[0]),int(nodes[-1]);span=hi-lo+1
            eligible=0<=lo<=hi<1<<32 and span<=8*len(nodes)
            if eligible and span*4<=self.input_lut_budget_bytes:
                if pairs.size and (int(pairs.min())<lo or int(pairs.max())>hi):
                    raise ValueError('Endpoint outside generation node domain')
                table=np.full(span,-1,np.int32)
                self.cold_count['lookup_table_live_bytes']=table.nbytes
                self.cold_count['lookup_table_peak_bytes']=max(table.nbytes,self.cold_count['lookup_table_peak_bytes'])
                self._temporary('mapping',table)
                positions=nodes-lo;values=np.arange(len(nodes),dtype=np.int32)
                self._temporary('mapping',table,positions,values);table[positions]=values
                del positions,values
                mapped=np.empty(pairs.shape,np.int32);self._temporary('mapping',table,mapped)
                for column in (0,1):
                    positions=pairs[:,column]-lo;self._temporary('mapping',table,mapped,positions)
                    values=table[positions];self._temporary('mapping',table,mapped,positions,values)
                    if np.any(values<0):raise ValueError('Endpoint outside generation node domain')
                    mapped[:,column]=values
                    del positions,values
                del table
                self.cold_count['lookup_table_live_bytes']=0
                self._temporary('mapping')
                self.cold_count['bounded_lookup_mapping_calls']+=1
                return mapped
            self.cold_count['lookup_budget_or_span_fallbacks']+=1
        mapped=np.searchsorted(nodes,pairs)
        self._temporary('mapping',mapped)
        if pairs.size and (not len(nodes) or np.any(mapped>=len(nodes)) or np.any(nodes[mapped]!=pairs)):
            raise ValueError('Endpoint outside generation node domain')
        result=mapped.astype(np.int32);self._temporary('mapping',mapped,result)
        self.cold_count['endpoint_searchsorted_calls']+=1
        self._temporary('mapping')
        return result

    def stats(self):
        result=super().stats()
        result['bounded_input']=dict(lookup_table_budget_bytes=self.input_lut_budget_bytes,packed_sort=self.packed_input_sort,
            lookup_table_live_bytes=self.cold_count['lookup_table_live_bytes'],lookup_table_peak_bytes=self.cold_count['lookup_table_peak_bytes'],
            explicit_temporary_peak_bytes=self.cold_count['input_explicit_temporary_peak_bytes'],
            scope='Table budget limits only transient LUT payload. Explicit temporary peak counts unique owned key/order/table/offset/index-output arrays at construction points; excludes common input copies, validation boolean temporaries, NumPy internal workspaces, Python metadata and process RSS. Produced endpoint mapping transfers to registration-local ColdPreparation; no LUT/key retained after register.')
        return result

    def _input(self, nodes, edges, weights):
            typed = (self._cold_frames and isinstance(nodes, np.ndarray) and nodes.ndim == 1 and
                     nodes.dtype.kind in 'iu' and isinstance(edges, np.ndarray) and
                     edges.ndim == 2 and edges.shape[1] == 2 and edges.dtype.kind in 'iu')
            if not typed:
                return super()._input(nodes, edges, weights)
            started = perf_counter()
            try:
                if ((nodes.dtype.kind == 'u' and nodes.size and int(nodes.max()) >= 1 << 63) or
                    (edges.dtype.kind == 'u' and edges.size and int(edges.max()) >= 1 << 63)):
                    raise ValueError('Node label exceeds int64')
                node_array = nodes.astype(np.int64, copy=True)
                proof_started = perf_counter()
                if len(node_array) > 1 and not np.all(node_array[1:] > node_array[:-1]):
                    raise ValueError('Sorted unique node labels required')
                if len(node_array) >= 1 << 31:
                    raise ValueError('Node positions must fit signed int32')
                # n strictly increasing integers with endpoints 0 and n-1 are
                # exactly the dense identity domain; no second arange array needed.
                identity = (not len(node_array) or
                            (node_array[0] == 0 and node_array[-1] == len(node_array)-1))
                self.cold_count['node_property_validation_seconds'] += perf_counter()-proof_started
                labels = tuple(node_array.tolist())
                pairs = edges.astype(np.int64, copy=True)
                if weights is None:
                    values = np.ones(len(pairs), np.float64)
                else:
                    original = weights if isinstance(weights, np.ndarray) else list(weights)
                    if np.iscomplexobj(original):
                        raise ValueError('Finite nonnegative weight required')
                    try:
                        values = np.asarray(original, np.float64)
                    except (TypeError, ValueError, OverflowError) as error:
                        raise ValueError('Finite nonnegative weight required') from error
                if values.ndim != 1 or len(values) != len(pairs):
                    raise ValueError('Initial edge/weight length mismatch')
                if not np.all(np.isfinite(values)) or np.any(values < 0):
                    raise ValueError('Finite nonnegative weight required')
                proof_started = perf_counter()
                canonical = lexicographic_unique(pairs)
                self.cold_count['edge_property_validation_seconds'] += perf_counter()-proof_started
                if canonical:
                    # Sorted input is still caller-owned; the authority takes its
                    # own weight array instead of relying on sort's advanced copy.
                    values = values.copy()
                    self.cold_count['verified_canonical_inputs'] += 1
                elif len(pairs):
                    sort_started = perf_counter()
                    order = self._sort_input(pairs)
                    pairs, values = pairs[order], values[order]
                    del order
                    self._temporary('sort')
                    first = np.r_[True, np.any(pairs[1:] != pairs[:-1], axis=1)]
                    if not first.all():
                        grouped = np.zeros(int(first.sum()), np.float64)
                        with np.errstate(over='ignore'):
                            np.add.at(grouped, np.cumsum(first)-1, values)
                        if not np.all(np.isfinite(grouped)):
                            raise ValueError('Finite nonnegative weight required')
                        pairs, values = pairs[first], grouped
                    self.cold_count['stable_sort_calls'] += 1
                    self.cold_count['stable_sort_seconds'] += perf_counter()-sort_started
                if len(pairs) >= 1 << 31:
                    raise ValueError('Edge positions must fit signed int32')
                mapping_started = perf_counter()
                if identity:
                    if pairs.size and (np.any(pairs < 0) or np.any(pairs >= len(node_array))):
                        raise ValueError('Endpoint outside generation node domain')
                    mapped = pairs.astype(np.int32)
                    self.cold_count['verified_identity_domains'] += 1
                else:
                    mapped = self._map_input(node_array, pairs)
                self.cold_count['endpoint_mapping_seconds'] += perf_counter()-mapping_started
                mapped.flags.writeable = False
                base = IndexedEdgeMap(pairs, values)
                self._cold_frames[-1] = ColdPreparation(weakref.ref(base), labels, node_array, mapped)
                self.cold_count['typed_input_calls'] += 1
                self.cold_count['endpoint_mapping_passes'] += 1
                self.cold_count['raw_edges_validated'] += len(edges)
                self.cold_count['canonical_edges_mapped'] += len(pairs)
                self._cold_observe()
                return labels, frozenset(labels), base
            finally:
                self._temporary('input')
                self.cold_count['lookup_table_live_bytes'] = 0
                self.cold_count['typed_input_seconds'] += perf_counter()-started
