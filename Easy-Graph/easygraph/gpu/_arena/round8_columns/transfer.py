"""Certificate-first Delta preparation; frozen R8 physical maintenance tail."""
import time
import numpy as np
import cupy as cp
from easygraph.gpu._arena.gpu_runtime import Arena
from easygraph.gpu._arena.round2_topology.state import Delta
from easygraph.gpu._arena.round3_transfer.arena_growth import grow_many, growth_bytes
from easygraph.gpu._arena.round3_transfer.kernels import TransferKernels
from easygraph.gpu._arena.round8_host.packet import make_packet

from easygraph.gpu._arena.round8_columns.delta import prepare_delta
from easygraph.gpu._arena.round8_columns.columns import deletion

class ColumnsTransferMixin:
    def _advance(self, gid, old_state, new_state, changes, names):
        """One shared packet and growth operation for this graph's survivors."""
        if getattr(self,'packet_engine','legacy')=='legacy':
            return super()._advance(gid,old_state,new_state,changes,names)
        if not names or not changes:
            return {}, {}
        if deletion(changes):
            self.counters['deletion_refreshes'] += 1
            return {}, {'path': 'deletion_refresh'}
        host_started=time.perf_counter()
        delta, certificate, evidence = prepare_delta(self.journal, self.refs[gid], old_state, new_state, changes)
        self.counters['certificate_delta_hits' if certificate is not None else 'certificate_delta_fallbacks'] += 1
        self.counters['certificate_delta_borrowed_payload_bytes'] += evidence['borrowed_payload_bytes']
        self.counters['certificate_delta_owned_payload_peak_bytes'] = max(
            self.counters['certificate_delta_owned_payload_peak_bytes'], evidence['owned_payload_bytes'])
        started = time.perf_counter()
        packet = make_packet(old_state.outgoing, old_state.incoming, new_state, delta,
                             {key[1:] for key in names.values()}, index=self.journal.index,
                             certificate=certificate)
        shapes = {self.model.components[name].allocation_id:
                  ((new_state.m,), self.arena.slots[self.model.components[name].allocation_id].array.dtype)
                  for name, key in names.items() if key[1] in ('col', 'out_col', 'weight')}
        reserve = Arena.capacity((packet.nbytes,), np.uint8) if packet.nbytes else 0
        needed = growth_bytes(self.arena, shapes)
        detail = {'packet_bytes': packet.nbytes, 'packet_seconds': time.perf_counter()-started,
                  'migration_d2d_bytes': 0, 'growth_d2d_bytes': 0, 'path': 'incremental',
                  'host_advance_preparation_seconds':time.perf_counter()-host_started,
                  'packet_implementation':'typed_columns_r8',
                  'delta_preparation':evidence}
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
