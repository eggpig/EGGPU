"""Borrowed-arena transfer kernels; no private device allocation."""
import cupy as cp
import numpy as np

SOURCE=r'''
__device__ int upper(const int *p,int n,int x) {
 int l=0,r=n;while(l<r){int c=l+(r-l)/2;if(p[c]<=x)l=c+1;else r=c;}return l;
}
extern "C" __global__ void move_int(int count,int start,int k,const int *before,const int *stage,int *out){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<count){int old=start+j;out[old+upper(before,k,old)]=stage[j];}
}
extern "C" __global__ void move_double(int count,int start,int k,const int *before,const double *stage,double *out){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<count){int old=start+j;out[old+upper(before,k,old)]=stage[j];}
}
extern "C" __global__ void insert_int(int k,const int *before,const int *value,int *out){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<k)out[before[j]+j]=value[j];
}
extern "C" __global__ void advance_row(int n,int k,const int *sources,int *row){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)row[j]+=upper(sources,k,j-1);
}
extern "C" __global__ void patch_double(int k,const int *positions,const double *value,double *out){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<k)out[positions[j]]=value[j];
}
'''


class TransferKernels:
    def __init__(self):
        module=cp.RawModule(code=SOURCE,backend='nvcc',options=('--std=c++17',))
        self.functions={name:module.get_function(name) for name in
                        ('move_int','move_double','insert_int','advance_row','patch_double')}

    def launch(self,name,count,*args):
        if count:self.functions[name](((count+255)//256,),(256,),(np.int32(count),*args))

    def insert(self,array,old_size,description,arena):
        """Reverse chunk order preserves unread input during rightward shifts."""
        before=description['before'];k=len(before)
        if not k:return 0
        scratch=arena.slots['@native_bfs_output']
        item=array.dtype.itemsize;chunk=scratch.capacity//item
        stage=cp.ndarray((chunk,),dtype=array.dtype,memptr=scratch.array.data)
        stream=cp.cuda.get_current_stream().ptr
        for end in range(old_size,0,-chunk):
            start=max(0,end-chunk);count=end-start
            cp.cuda.runtime.memcpyAsync(stage.data.ptr,array.data.ptr+start*item,count*item,3,stream)
            self.launch('move_double' if item==8 else 'move_int',count,np.int32(start),np.int32(k),before,stage,array)
        if item==4:self.launch('insert_int',k,before,description['values'],array)
        return 2*old_size*item

    def row(self,array,description):
        self.launch('advance_row',len(array),np.int32(len(description['rows'])),description['rows'],array)

    def patch(self,array,positions,values):
        self.launch('patch_double',len(positions),positions,values,array)
