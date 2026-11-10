"""R7 batch descriptors with exactly the frozen packet field/byte contract."""
from dataclasses import dataclass
import numpy as np
from easygraph.gpu._arena.round2_topology.state import locate as scalar_locate

def locate(layout,src,dst):
    """Batch rank lookup; scalar row lookup remains efficient for tiny deltas."""
    if len(src)<64:return scalar_locate(layout,src,dst)
    rows=np.repeat(np.arange(len(layout.row)-1,dtype=np.uint64),np.diff(layout.row))
    keys=(rows<<np.uint64(32))|layout.col.astype(np.uint64)
    wanted=(src.astype(np.uint64)<<np.uint64(32))|dst.astype(np.uint64)
    positions=np.searchsorted(keys,wanted)
    present=positions<len(keys)
    if len(keys):present[present]&=keys[positions[present]]==wanted[present]
    return positions.astype(np.int32),present



@dataclass
class Packet:
    data: np.ndarray
    fields: dict
    arrays: dict

    @property
    def nbytes(self):return self.data.nbytes


def insertion_description(layout,src,dst):
    """Before[i] old entries precede inserted item i in new sorted CSR order."""
    order=np.lexsort((dst,src));src=src[order];dst=dst[order]
    before,present=locate(layout,src,dst)
    if present.any():raise ValueError('Insertion descriptor includes an existing edge')
    return dict(before=np.ascontiguousarray(before,np.int32),values=np.ascontiguousarray(dst,np.int32),
                rows=np.ascontiguousarray(src,np.int32))


def make_packet(old_out,old_in,state,delta,kinds):
    arrays={};new=delta.new
    for direction,layout,src,dst in (('out',old_out,delta.src,delta.dst),('in',old_in,delta.dst,delta.src)):
        relevant={'out_row','out_col'} if direction=='out' else {'row','col','weight'}
        if np.any(new) and any(kind in relevant for kind,attr in kinds):
            desc=insertion_description(layout,src[new],dst[new])
            for name,value in desc.items():arrays[direction+'_'+name]=value
    if ('weight','message_count') in kinds and len(delta.ids):
        positions,present=locate(state.incoming,delta.dst,delta.src)
        if not present.all():raise AssertionError('Updated weights missing from new layout')
        arrays['weight_positions']=positions;arrays['weight_values']=delta.values
    for attr,name,value in ((None,'degree',state.degree),('message_count','sum',state.sums)):
        if ('source_sum',attr) in kinds and (attr is not None or np.any(new)) and len(delta.changed_rows):
            arrays[name+'_positions']=np.ascontiguousarray(delta.changed_rows,np.int32)
            arrays[name+'_values']=np.ascontiguousarray(value[delta.changed_rows],np.float64)
    fields={};size=0
    for name,value in arrays.items():
        size=(size+7)//8*8
        value=np.ascontiguousarray(value);arrays[name]=value
        fields[name]=dict(offset=size,shape=value.shape,dtype=value.dtype.str)
        size+=value.nbytes
    data=np.zeros(size,np.uint8)
    for name,value in arrays.items():
        offset=fields[name]['offset'];data[offset:offset+value.nbytes]=value.view(np.uint8)
    return Packet(data,fields,arrays)


def advance_cpu(old,description):
    """Independent executable definition used to qualify insertion descriptors."""
    before=description['before'];m=len(old)
    target=np.arange(m)+np.searchsorted(before,np.arange(m),side='right')
    out=np.empty(m+len(before),old.dtype);out[target]=old
    out[before+np.arange(len(before))]=description['values']
    return out,target
