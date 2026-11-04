"""Identical physical maintenance with a batched CPU packet builder."""
import time
import numpy as np
import cupy as cp
from easygraph.gpu._arena.gpu_runtime import Arena
from easygraph.gpu._arena.round2_topology.state import Delta
from easygraph.gpu._arena.round3_transfer.arena_growth import grow_many, growth_bytes
from easygraph.gpu._arena.round3_transfer.kernels import TransferKernels
from easygraph.gpu._arena.round7_recovery.packet import make_packet

class VectorTransferMixin:
    def _advance(self, gid, old_state, new_state, changes, names):
        """One shared packet and growth operation for this graph's survivors."""
        if getattr(self,'packet_engine','legacy')=='legacy':
            return super()._advance(gid,old_state,new_state,changes,names)
        if not names or not changes:
            return {}, {}
        if any(c.final_weight is None for c in changes):
            self.counters['deletion_refreshes'] += 1
            return {}, {'path': 'deletion_refresh'}
        host_started=time.perf_counter()
        nodes = new_state.nodes
        src = np.searchsorted(nodes, [c.src for c in changes]).astype(np.int32)
        dst = np.searchsorted(nodes, [c.dst for c in changes]).astype(np.int32)
        delta = Delta(src, dst, np.arange(len(changes), dtype=np.int32),
                      np.asarray([c.final_weight for c in changes], np.float64),
                      np.asarray([c.base_weight is None for c in changes], bool), np.unique(src))
        started = time.perf_counter()
        packet = make_packet(old_state.outgoing, old_state.incoming, new_state, delta,
                             {key[1:] for key in names.values()})
        shapes = {self.model.components[name].allocation_id:
                  ((new_state.m,), self.arena.slots[self.model.components[name].allocation_id].array.dtype)
                  for name, key in names.items() if key[1] in ('col', 'out_col', 'weight')}
        reserve = Arena.capacity((packet.nbytes,), np.uint8) if packet.nbytes else 0
        needed = growth_bytes(self.arena, shapes)
        detail = {'packet_bytes': packet.nbytes, 'packet_seconds': time.perf_counter()-started,
                  'migration_d2d_bytes': 0, 'growth_d2d_bytes': 0, 'path': 'incremental',
                  'host_advance_preparation_seconds':time.perf_counter()-host_started,
                  'packet_implementation':'batch_lookup_r7'}
        if self.arena.live + needed + reserve > self.arena.budget:
            self.counters['incremental_capacity_fallbacks'] += 1
            return {}, dict(detail, path='capacity_refresh')
        if self.transfer_kernels is None:
            self.transfer_kernels = TransferKernels()
        log_key = 'r4:transfer_packet'
        try:
            grow = grow_many(self.arena, shapes, reserve_bytes=reserve)
            detail['growth_d2d_bytes'] = grow['packed_growth_d2d_bytes']
            fields = {}
            if packet.nbytes:
                try:
                    array = self.arena.allocate(log_key, (packet.nbytes,), np.uint8)
                except MemoryError:
                    self.arena.compact(staging_key='@native_bfs_output')
                    array = self.arena.allocate(log_key, (packet.nbytes,), np.uint8)
                array.set(packet.data)
                self.counters['h2d_bytes'] += packet.nbytes
                self.counters['maintenance_packet_h2d_bytes'] += packet.nbytes
                slot = self.arena.slots[log_key]
                fields = {key: cp.ndarray(info['shape'], dtype=info['dtype'],
                           memptr=cp.cuda.memory.MemoryPointer(self.arena.memory, slot.offset+info['offset']))
                          for key, info in packet.fields.items()}
            for name, key in names.items():
                kind, attr = key[1:]
                array = self.arena.slots[self.model.components[name].allocation_id].array
                direction = 'out' if kind.startswith('out_') else 'in'
                desc = {p: fields[direction+'_'+p] for p in ('before', 'values', 'rows')
                        if direction+'_'+p in fields}
                if np.any(delta.new):
                    if kind in ('col', 'out_col', 'weight'):
                        detail['migration_d2d_bytes'] += self.transfer_kernels.insert(array, old_state.m, desc, self.arena)
                    elif kind in ('row', 'out_row'):
                        self.transfer_kernels.row(array, desc)
                if kind == 'weight' and 'weight_positions' in fields:
                    self.transfer_kernels.patch(array, fields['weight_positions'], fields['weight_values'])
                elif kind == 'source_sum':
                    prefix = 'degree' if attr is None else 'sum'
                    if prefix+'_positions' in fields:
                        self.transfer_kernels.patch(array, fields[prefix+'_positions'], fields[prefix+'_values'])
            cp.cuda.runtime.deviceSynchronize()
        except Exception:
            # Logical mutations were already committed. Retire affected device
            # state; a retry reconstructs the current authoritative host graph.
            self._retire_graph(gid)
            self.counters['failed_device_maintenance_flushes'] += 1
            raise
        finally:
            if log_key in self.arena.slots:
                self.arena.release(log_key)
        self.counters['maintenance_packets'] += bool(packet.nbytes)
        self.counters['migration_d2d_bytes'] += detail['migration_d2d_bytes']
        self.counters['growth_d2d_bytes'] += detail['growth_d2d_bytes']
        return names, detail
