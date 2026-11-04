#include "single_owner_sssp.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <limits>
namespace arena_frontier {

__device__ double minimum(double* p, double value) {
  unsigned long long* address=(unsigned long long*)p;
  unsigned long long old=*address, assumed;
  while(value<__longlong_as_double(old)) {
    assumed=old;
    old=atomicCAS(address,assumed,__double_as_longlong(value));
    if(old==assumed) break;
  }
  return __longlong_as_double(old);
}
__global__ void initialize(const int n, const int* sources,
 const int source_index, double* distances, int* frontier, int* tags, int* ctl) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
    distances[i]=i==sources[source_index] ? 0. : __longlong_as_double(0x7ff0000000000000ULL);
    tags[i]=-1;
  }
  if(blockIdx.x==0 && threadIdx.x==0) {
    frontier[0]=sources[source_index];
    ctl[0]=1; ctl[1]=0; ctl[2]=0; ctl[3]=0;
  }
}
template<int WIDTH> __device__ void traverse(const int* row,const int* col,
 const double* weight, const int* frontier, int* next, double* distances,
 int* tags, int* ctl) {
  if(ctl[2]) return;
  int lane=threadIdx.x&(WIDTH-1);
  int stride=gridDim.x*blockDim.x/WIDTH;
  for(int i=(blockIdx.x*blockDim.x+threadIdx.x)/WIDTH;i<ctl[0];i+=stride) {
    int u=frontier[i];
    double du=0.;
    if(lane==0) du=__longlong_as_double(atomicCAS((unsigned long long*)(distances+u),0ULL,0ULL));
    du=__shfl_sync((0xffffffffU >> (32-WIDTH)) << ((threadIdx.x&31)&~(WIDTH-1)),du,0,WIDTH);
    for(int e=row[u]+lane;e<row[u+1];e+=WIDTH) {
      int v=col[e]; double candidate=du+weight[e];
      double prior=minimum(distances+v,candidate);
      if(candidate<prior && atomicExch(tags+v,ctl[3])!=ctl[3]) {
        unsigned mask=__activemask();
        int leader=__ffs(mask)-1,warp_lane=threadIdx.x&31;
        int first=0;
        if(warp_lane==leader) first=atomicAdd(ctl+1,__popc(mask));
        first=__shfl_sync(mask,first,leader);
        int offset=__popc(mask&((1U<<warp_lane)-1));
        next[first+offset]=v;
      }
    }
  }
}
#define STEP(W) __global__ void step##W(const int* row,const int* col, \
 const double* weight,const int* frontier,int* next,double* distances,int* tags,int* ctl) \
 { traverse<W>(row,col,weight,frontier,next,distances,tags,ctl); }
STEP(1) STEP(2) STEP(4) STEP(8) STEP(16) STEP(32)
__global__ void advance(int* ctl) {
  if(!ctl[2]) {
    ctl[3]++;
    ctl[0]=ctl[1]; ctl[1]=0;
    if(ctl[0]==0) ctl[2]=1;
  }
}

__global__ void degree_bound(const int n,const int* row,int* answer) {
  __shared__ int values[128];
  int m=0;
  for(int i=threadIdx.x;i<n;i+=128) m=max(m,row[i+1]-row[i]);
  values[threadIdx.x]=m;__syncthreads();
  for(int stride=64;stride;stride>>=1) {
    if(threadIdx.x<stride) values[threadIdx.x]=max(values[threadIdx.x],values[threadIdx.x+stride]);
    __syncthreads();
  }
  if(threadIdx.x==0) answer[0]=values[0];
}

}
namespace arena_delta {
// Exact nonnegative parallel-delta SSSP, with device-resident control.
// All graph, output and workspace pointers are borrowed from the session Arena.

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

}
namespace gpu_easygraph { namespace partitioned {
cudaError_t run_single_owner_sssp(const int* row, const int* col,
    const double* weight, int n, int m, int source,
    void* scratch, std::size_t capacity, double* output,
    cudaStream_t stream, runtime::SingleOwnerSsspStats* stats) {
    if (!stats) return cudaErrorInvalidValue;
    *stats = runtime::SingleOwnerSsspStats{};
    const auto need = runtime::single_owner_sssp_workspace_bytes(n < 0 ? UINT64_MAX : n);
    if (n <= 0 || n > std::numeric_limits<int>::max()/2 || m < 0 ||
        m > std::numeric_limits<int>::max()/2 || source < 0 || source >= n || !need || capacity < need ||
        !row || !scratch || !output || (m && (!col || !weight)) ||
        reinterpret_cast<std::uintptr_t>(scratch) % 256) return cudaErrorInvalidValue;
    // This function runs only on a fixed native runtime worker. A per-thread
    // device property value is metadata, never a CUDA allocation or graph cache.
    static thread_local int cached_device = -1;
    static thread_local int cached_sms = 0;
    int device = -1;
    cudaError_t error;
#define OWNER_CHECK(expr) do { error = (expr); if (error != cudaSuccess) return error; } while (0)
    OWNER_CHECK(cudaGetDevice(&device));
    if (cached_device != device) {
        cudaDeviceProp properties;
        OWNER_CHECK(cudaGetDeviceProperties(&properties, device));
        cached_device = device; cached_sms = properties.multiProcessorCount;
    }
    stats->sm_count = cached_sms;
    stats->scratch_capacity_bytes = capacity;
    unsigned char* bytes = static_cast<unsigned char*>(scratch);
    const auto payload = runtime::single_owner_sssp_payload_bytes(n);
    int* source_device = reinterpret_cast<int*>(bytes + payload);
    OWNER_CHECK(cudaMemcpyAsync(source_device, &source, sizeof(int), cudaMemcpyHostToDevice, stream));
    stats->source_h2d_bytes = sizeof(int);
    bool sparse = n >= 256 && static_cast<std::uint64_t>(m) <= 4ULL * n;
    if (sparse) {
        int* degree = reinterpret_cast<int*>(scratch);
        arena_frontier::degree_bound<<<1,128,0,stream>>>(n,row,degree);
        OWNER_CHECK(cudaGetLastError());
        OWNER_CHECK(cudaMemcpyAsync(&stats->maximum_degree,degree,sizeof(int),cudaMemcpyDeviceToHost,stream));
        OWNER_CHECK(cudaStreamSynchronize(stream));
        stats->control_d2h_bytes += sizeof(int);
        ++stats->kernel_launches; ++stats->stream_synchronizations;
        sparse = stats->maximum_degree <= 128;
    }
    stats->sparse_frontier = sparse;
    const int checkpoint = 16;
    const int cap = 8 * cached_sms;
    if (!sparse) {
        std::uint64_t d2h = 0;
        int result = arena_delta::eggpu_device_delta_ieee_v3(row,col,weight,source_device,
            n,m,1,32,checkpoint,cap,scratch,payload,output,stream,&d2h,
            &stats->rounds,&stats->queued_rounds,&stats->checkpoints);
        if (result) return static_cast<cudaError_t>(result);
        stats->control_d2h_bytes += d2h;
        stats->kernel_launches += 2 + 4ULL * stats->queued_rounds;
        stats->stream_synchronizations += stats->checkpoints;
    } else {
        // The exact corrected WarpFrontierSSSPV2 kernels receive original
        // native storage with the same 4-lane/8-blocks-per-SM/16-round policy.
        int* first = reinterpret_cast<int*>(scratch);
        int* second = first + n;
        int* tags = second + n;
        int* control = tags + n;
        const int grid = std::min(cap, std::max(1, static_cast<int>((4LL*n+127)/128)));
        const int init_grid = std::min(cap, std::max(1,(n+127)/128));
        arena_frontier::initialize<<<init_grid,128,0,stream>>>(n,source_device,0,output,first,tags,control);
        OWNER_CHECK(cudaGetLastError());
        int host[4] = {0,0,0,0};
        for (int begin = 0; begin < n+1; begin += checkpoint) {
            const int batch = std::min(checkpoint,n+1-begin);
            for (int iteration = 0; iteration < batch; ++iteration) {
                arena_frontier::step4<<<grid,128,0,stream>>>(row,col,weight,first,second,output,tags,control);
                arena_frontier::advance<<<1,1,0,stream>>>(control);
                std::swap(first,second);
            }
            OWNER_CHECK(cudaGetLastError());
            stats->queued_rounds += batch;
            OWNER_CHECK(cudaMemcpyAsync(host,control,sizeof(host),cudaMemcpyDeviceToHost,stream));
            OWNER_CHECK(cudaStreamSynchronize(stream));
            stats->control_d2h_bytes += sizeof(host); ++stats->checkpoints;
            if (host[2]) break;
        }
        if (!host[2]) return cudaErrorUnknown;
        stats->rounds = host[3];
        stats->kernel_launches += 1 + 2ULL * stats->queued_rounds;
        stats->stream_synchronizations += stats->checkpoints;
    }
    stats->used = true;
#undef OWNER_CHECK
    return cudaSuccess;
}
}}
