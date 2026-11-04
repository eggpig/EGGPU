#include <cuda_runtime.h>
#include <cstddef>
__global__ void dynamic_scatter(double* target,const int* position,const double* values,std::size_t count) {
 const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<count)target[position[i]]=values[i];
}
extern "C" cudaError_t eggpu_dynamic_scatter_f64(double* target,const int* position,const double* values,std::size_t count) {
 if(count)dynamic_scatter<<<unsigned((count+255)/256),256>>>(target,position,values,count);
 return cudaGetLastError();
}
