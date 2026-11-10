"""Node-level source sums on the existing host graph and version contract."""
import time
import numpy as np
from easygraph.gpu._arena.gpu_runtime import HostGraph
from easygraph.gpu._arena.fast_host_graph import FastHostGraph
from easygraph.gpu._arena.retained_node_map import RetainedNodeMapHostGraph

class FactorizedGraph(HostGraph):
    @classmethod
    def wrap(cls,graph):
        if isinstance(graph,cls):return graph
        variants={HostGraph:cls,FastHostGraph:FactorizedFastHostGraph,
                  RetainedNodeMapHostGraph:FactorizedRetainedHostGraph}
        if type(graph) not in variants:
            raise TypeError('This qualified adapter requires HostGraph, FastHostGraph or RetainedNodeMapHostGraph')
        result=variants[type(graph)].__new__(variants[type(graph)])
        result.__dict__=graph.__dict__
        result.sum_repair_rows=0
        result.sum_repair_scanned_edges=0
        result.norm_closure_entries=0
        result._registered_keys=set()
        result._active_key_callback=None
        result.input_graph_class=type(graph).__name__
        return result

    def register_keys(self,keys):self._registered_keys.update(keys)

    def set_active_key_callback(self,callback):self._active_key_callback=callback

    def key_is_active(self,key):
        if self._active_key_callback is not None:return self._active_key_callback(key)
        return key in self.arrays or key in self._registered_keys

    def get(self,key):
        self.register_keys([key])
        if key[2] is None and key[1] in ('norm','dangling'):
            start=time.perf_counter()
            if key not in self.arrays:
                degree=np.diff(self.out_row)
                self.arrays[key]=(1.0/degree[self.col] if key[1]=='norm' else np.asarray(degree==0,np.uint8))
                self.view_builds+=1
            self.host_prepare_seconds+=time.perf_counter()-start
            return self.arrays[key]
        if key[1]!='source_sum':return super().get(key)
        start=time.perf_counter()
        if key not in self.arrays:
            attr=key[2]
            # Unit PR has no edge-length all-ones array.
            self.arrays[key]=(np.diff(self.out_row).astype(np.float64) if attr is None else
                              np.bincount(self.col,weights=self.attributes[attr],minlength=self.n))
            self.view_builds+=1
        self.host_prepare_seconds+=time.perf_counter()-start
        return self.arrays[key]

    def dependency(self,key):
        if key[1]!='source_sum':return super().dependency(key)
        return (self.name+':topology',self.name+':mapping')+(() if key[2] is None else (self.name+':attr:'+key[2],))

    def version(self,key):
        if key[1]!='source_sum':return super().version(key)
        return self.generation,0 if key[2] is None else self.versions[key[2]]

    def _sum_rows(self,attr,rows,record=True):
        if attr is None:return np.asarray(self.out_row[rows+1]-self.out_row[rows],np.float64)
        values=self.attributes[attr]
        result=np.empty(len(rows),np.float64)
        for i,u in enumerate(rows):
            positions=self.out_order[self.out_row[u]:self.out_row[u+1]]
            result[i]=values[positions].sum(dtype=np.float64)
            if record:self.sum_repair_scanned_edges+=len(positions)
        if record:self.sum_repair_rows+=len(rows)
        return result

    def update(self,attr,positions,values):
        positions=np.asarray(positions,np.int32);values=np.asarray(values,np.float64)
        if positions.shape!=values.shape or np.any(positions<0) or np.any(positions>=len(self.col)):
            raise ValueError('invalid update batch')
        if not np.all(np.isfinite(values)) or np.any(values<0):raise ValueError('nonnegative finite weights required')
        kinds=('weight','out_weight','norm','dangling','source_sum')
        active={kind for kind in kinds if self.key_is_active(self.key(kind,attr))}
        self.attributes[attr][positions]=values;self.versions[attr]+=1;version=self.versions[attr]
        rows=(np.unique(self.col[positions]) if active.intersection(('norm','dangling','source_sum')) else np.empty(0,np.int32))
        affected=(np.concatenate([self.out_order[self.out_row[u]:self.out_row[u+1]] for u in rows])
                  if 'norm' in active and rows.size else np.empty(0,np.int32))
        self.norm_closure_entries+=len(affected)
        for kind in kinds:
            key=self.key(kind,attr)
            if kind not in active:
                # A later first request builds from current attributes. An old
                # untracked representation cannot silently use an empty delta.
                self.changes.pop(key,None);self.change_floors[key]=version
                continue
            touched=(positions if kind=='weight' else self.in_to_out[positions] if kind=='out_weight'
                     else affected if kind=='norm' else rows)
            self.changes[key].append((version,touched.copy()))
            if key not in self.arrays:continue
            if kind=='out_weight':self.arrays[key][touched]=self.attributes[attr][positions]
            elif kind=='norm':
                for u in rows:
                    edges=self.out_order[self.out_row[u]:self.out_row[u+1]];total=self.attributes[attr][edges].sum(dtype=np.float64)
                    self.arrays[key][edges]=self.attributes[attr][edges]/total if total else 0.
                self.repaired_values+=len(affected)
            elif kind=='dangling':self.arrays[key][rows]=self._sum_rows(attr,rows,record=False)==0
            elif kind=='source_sum':self.arrays[key][rows]=self._sum_rows(attr,rows)

    def repair_values(self,key,positions):
        if key[1]!='source_sum':return super().repair_values(key,positions)
        if key in self.arrays:return self.arrays[key][positions]
        # Recompute complete affected source sums from current raw values.
        # Repeated updates therefore do not accumulate subtract/add drift.
        return self._sum_rows(key[2],positions)


class FactorizedFastHostGraph(FactorizedGraph,FastHostGraph):
    pass


class FactorizedRetainedHostGraph(FactorizedGraph,RetainedNodeMapHostGraph):
    pass
