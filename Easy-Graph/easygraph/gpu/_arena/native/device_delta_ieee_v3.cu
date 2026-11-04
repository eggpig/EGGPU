// Exact nonnegative parallel-delta SSSP, with device-resident control.
// All graph, output and workspace pointers are borrowed from the session Arena.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <climits>
#include <cstddef>
#include <limits>

namespace {
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr int BLOCK = 256;
struct Control { int active, done, rounds, unused; double delta; };
std::size_t padded(std::size_t n) { return (n+255)/256*256; }
int reduce_blocks(int n) { return std::min(4096,std::max(1,(n+255)/256)); }
__device__ double atomic_read(const double* p) {
    return __longlong_as_double(atomicCAS((unsigned long long*)p,0ULL,0ULL));
}
__device__ void nonnegative_min(double* p,double value) {
    // Nonnegative finite weights imply nonnegative distances. Unsigned IEEE754
    // ordering matches numeric ordering, including +inf on overflow.
    atomicMin((unsigned long long*)p,(unsigned long long)__double_as_longlong(value));
}
__global__ void minimum_edges(const int* row,const double* weight,int n,double* edge_min) {
    for(int u=blockIdx.x*blockDim.x+threadIdx.x;u<n;u+=blockDim.x*gridDim.x) {
        double value=INF;
        for(int e=row[u];e<row[u+1];++e) value=fmin(value,weight[e]);
        edge_min[u]=value;
    }
}
__global__ void initialize(int n,const int* sources,int source_index,double* distance,int* unsettled,int* frontier,Control* ctl) {
    int source=sources[source_index];
    for(int u=blockIdx.x*blockDim.x+threadIdx.x;u<n;u+=blockDim.x*gridDim.x) {
        unsettled[u]=1;distance[u]=(u==source)?0.:INF;
    }
    if(blockIdx.x==0 && threadIdx.x==0) {
        frontier[0]=source;ctl->active=1;ctl->done=0;ctl->rounds=0;ctl->delta=INF;
    }
}
__global__ void relax(const int* row,const int* col,const double* weight,const int* frontier,
                      double* distance,const Control* ctl,int width) {
    if(ctl->done) return;
    long long total=(long long)ctl->active*width;
    long long tid=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long stride=(long long)gridDim.x*blockDim.x;
    for(long long i=tid;i<total;i+=stride) {
        int u=frontier[i/width];double du=atomic_read(distance+u);
        if(du>=INF) continue;
        for(int e=row[u]+i%width;e<row[u+1];e+=width)
            nonnegative_min(distance+col[e],du+weight[e]);
    }
}
__global__ void partial_min(const double* distance,const int* unsettled,const double* edge_min,
                            double* partial,int n,const Control* ctl) {
    if(ctl->done) return;
    __shared__ double values[BLOCK];double value=INF;
    for(int u=blockIdx.x*blockDim.x+threadIdx.x;u<n;u+=blockDim.x*gridDim.x)
        if(unsettled[u] && distance[u]<INF) value=fmin(value,distance[u]+edge_min[u]);
    values[threadIdx.x]=value;__syncthreads();
    for(int s=BLOCK/2;s;s>>=1) {if(threadIdx.x<s) values[threadIdx.x]=fmin(values[threadIdx.x],values[threadIdx.x+s]);__syncthreads();}
    if(threadIdx.x==0) partial[blockIdx.x]=values[0];
}
__global__ void finalize_min(const double* partial,int count,Control* ctl) {
    if(ctl->done) return;
    __shared__ double values[BLOCK];double value=INF;
    for(int i=threadIdx.x;i<count;i+=BLOCK) value=fmin(value,partial[i]);
    values[threadIdx.x]=value;__syncthreads();
    for(int s=BLOCK/2;s;s>>=1) {if(threadIdx.x<s) values[threadIdx.x]=fmin(values[threadIdx.x],values[threadIdx.x+s]);__syncthreads();}
    if(threadIdx.x==0) {
        ctl->delta=values[0];ctl->active=0;++ctl->rounds;
        if(values[0]>=INF) ctl->done=1;
    }
}
__global__ void build_frontier(const double* distance,int* unsettled,int* frontier,int n,Control* ctl) {
    if(ctl->done) return;
    int lane=threadIdx.x&31;
    for(int u=blockIdx.x*blockDim.x+threadIdx.x;u<n;u+=blockDim.x*gridDim.x) {
        bool add=unsettled[u] && distance[u]<=ctl->delta && distance[u]<INF;
        unsigned mask=__ballot_sync(__activemask(),add);
        if(add) {
            int leader=__ffs(mask)-1,first=0;
            if(lane==leader) first=atomicAdd(&ctl->active,__popc(mask));
            first=__shfl_sync(mask,first,leader);
            frontier[first+__popc(mask&((1U<<lane)-1))]=u;unsettled[u]=0;
        }
    }
}
}

extern "C" int eggpu_device_delta_ieee_v3(const int* row,const int* col,const double* weight,const int* sources,
     int n,int m,int source_count,int width,int checkpoint,int launch_cap,
     void* workspace,std::size_t workspace_capacity,double* output,cudaStream_t stream,
     std::uint64_t* d2h_bytes,int* rounds,int* queued_rounds,int* checkpoints) {
    *d2h_bytes=0;*rounds=0;*queued_rounds=0;*checkpoints=0;
    if(n<0||m<0||source_count<0||width<1||width>32||(width&(width-1))||checkpoint<1||launch_cap<1)
        return int(cudaErrorInvalidValue);
    if(!n||!source_count) return int(cudaSuccess);
    int reduction=reduce_blocks(n);
    std::size_t need=2*padded(std::size_t(n)*4)+padded(std::size_t(n)*8)+padded(std::size_t(reduction)*8)+256;
    if(!row||!sources||!workspace||!output||(m&&(!col||!weight))||workspace_capacity<need||std::uintptr_t(workspace)%256)
        return int(cudaErrorInvalidValue);
    unsigned char* bytes=(unsigned char*)workspace;
    int* unsettled=(int*)bytes;bytes+=padded(std::size_t(n)*4);
    int* frontier=(int*)bytes;bytes+=padded(std::size_t(n)*4);
    double* edge_min=(double*)bytes;bytes+=padded(std::size_t(n)*8);
    double* partial=(double*)bytes;bytes+=padded(std::size_t(reduction)*8);
    Control* ctl=(Control*)bytes;
    int scan=std::min(65535,std::max(1,(n+BLOCK-1)/BLOCK));
    int expand=std::min(launch_cap,std::max(1,int(((long long)n*width+BLOCK-1)/BLOCK)));
    cudaError_t error;
#define CHECK(expr) do{error=(expr);if(error!=cudaSuccess)return int(error);}while(0)
    minimum_edges<<<scan,BLOCK,0,stream>>>(row,weight,n,edge_min);
    CHECK(cudaGetLastError());
    for(int source=0;source<source_count;++source) {
        double* distance=output+std::size_t(source)*n;
        initialize<<<scan,BLOCK,0,stream>>>(n,sources,source,distance,unsettled,frontier,ctl);
        CHECK(cudaGetLastError());
        Control host;
        for(int first=0;first<n+1;first+=checkpoint) {
            int batch=std::min(checkpoint,n+1-first);
            for(int iteration=0;iteration<batch;++iteration) {
                relax<<<expand,BLOCK,0,stream>>>(row,col,weight,frontier,distance,ctl,width);
                partial_min<<<reduction,BLOCK,0,stream>>>(distance,unsettled,edge_min,partial,n,ctl);
                finalize_min<<<1,BLOCK,0,stream>>>(partial,reduction,ctl);
                build_frontier<<<scan,BLOCK,0,stream>>>(distance,unsettled,frontier,n,ctl);
            }
            CHECK(cudaGetLastError());*queued_rounds+=batch;
            CHECK(cudaMemcpyAsync(&host,ctl,sizeof(Control),cudaMemcpyDeviceToHost,stream));
            CHECK(cudaStreamSynchronize(stream));*d2h_bytes+=sizeof(Control);++*checkpoints;
            if(host.done) break;
        }
        if(!host.done) return int(cudaErrorUnknown);
        *rounds+=host.rounds;
    }
#undef CHECK
    return int(cudaSuccess);
}
