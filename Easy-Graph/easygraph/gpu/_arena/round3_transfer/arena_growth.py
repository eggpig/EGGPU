"""Expand several packed graph slots in one paid, backward device pass."""
import cupy as cp
from easygraph.gpu._arena.gpu_runtime import Arena,Slot


def growth_bytes(arena,shapes):
    return sum(max(0,Arena.capacity(shape,dtype)-arena.slots[key].capacity)
               for key,(shape,dtype) in shapes.items())


def grow_many(arena,shapes,reserve_bytes=0):
    needed=growth_bytes(arena,shapes)
    if arena.live+needed+reserve_bytes>arena.budget:
        raise MemoryError('Retained state growth and transfer log exceed arena budget')
    if not needed:
        for key,(shape,dtype) in shapes.items():
            slot=arena.slots[key]
            slot.array=cp.ndarray(shape,dtype=dtype,memptr=cp.cuda.memory.MemoryPointer(arena.memory,slot.offset))
        return dict(growth_capacity_bytes=0,packed_growth_d2d_bytes=0,compaction_d2d_bytes=0)
    cp.cuda.runtime.deviceSynchronize()
    scratch_key='@native_bfs_output';scratch=arena.slots[scratch_key]
    before_compact=arena.compaction_bytes
    if needed:arena.compact(staging_key=scratch_key)
    cursor=max(slot.offset+slot.capacity for key,slot in arena.slots.items() if str(key).startswith('@'))
    destinations=[]
    for key,slot in sorted(arena.slots.items(),key=lambda pair:pair[1].offset):
        if str(key).startswith('@'):continue
        shape,dtype=shapes.get(key,(slot.array.shape,slot.array.dtype))
        capacity=max(slot.capacity,Arena.capacity(shape,dtype))
        destinations.append((key,slot,cursor,capacity,shape,dtype));cursor+=capacity
    stream=cp.cuda.get_current_stream().ptr;moved=0
    for key,slot,target,capacity,shape,dtype in reversed(destinations):
        if target!=slot.offset:
            if target<slot.offset:raise AssertionError('Expansion must move toward higher addresses')
            for end in range(slot.capacity,0,-scratch.capacity):
                start=max(0,end-scratch.capacity);count=end-start
                cp.cuda.runtime.memcpyAsync(scratch.array.data.ptr,arena.memory.ptr+slot.offset+start,count,3,stream)
                cp.cuda.runtime.memcpyAsync(arena.memory.ptr+target+start,scratch.array.data.ptr,count,3,stream)
            moved+=2*slot.capacity
        array=cp.ndarray(shape,dtype=dtype,memptr=cp.cuda.memory.MemoryPointer(arena.memory,target))
        arena.slots[key]=Slot(target,capacity,array)
    cp.cuda.runtime.deviceSynchronize()
    arena.live=cursor;arena.peak=max(arena.peak,cursor)
    arena.free=[(cursor,arena.budget-cursor)] if cursor<arena.budget else []
    return dict(growth_capacity_bytes=needed,packed_growth_d2d_bytes=moved,
                compaction_d2d_bytes=arena.compaction_bytes-before_compact)
