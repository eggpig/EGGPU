// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#include <functional>
#include <map>
#include <framework/framework.cuh>
#include <framework/hybrid_policy.h>
#include <framework/clion_cuda.cuh>
#include <utils/cuda_utils.h>
#include "hybrid_sssp_common.h"
#include "../../include/groute/graphs/csr_graph.cuh"

DEFINE_int32(source_node,
             0, "The source node for the SSSP traversal (clamped to [0, nnodes-1])");
DEFINE_bool(sparse,
            false, "use async/push/dd + fusion for high-diameter");
DECLARE_int32(top_ranks);
DECLARE_bool(print_ranks);
DECLARE_string(output);
DECLARE_bool(check);
DECLARE_int32(prio_delta);

namespace hybrid_sssp
{
    template<typename TValue, typename TBuffer, typename TWeight, typename...UnusedData>
    struct SSSP : sepgraph::api::AppBase<TValue, TBuffer, TWeight>
    {
        using sepgraph::api::AppBase<TValue, TBuffer, TWeight>::AccumulateBuffer;
        index_t m_source_node;
        unsigned long long* eggpu_weight_keys=nullptr;
        unsigned* eggpu_weight_values=nullptr;
        unsigned eggpu_weight_capacity=0;
        __device__ __forceinline__ unsigned EggpuEdgeWeight(unsigned src,unsigned dst)const{
          if(dst==UINT32_MAX)return 0;
          unsigned long long key=((static_cast<unsigned long long>(src)<<32)|dst)+1;
          auto hash=key;hash^=hash>>33;hash*=0xff51afd7ed558ccdULL;hash^=hash>>33;hash*=0xc4ceb9fe1a85ec53ULL;hash^=hash>>33;
          unsigned slot=hash&(eggpu_weight_capacity-1);
          for(unsigned i=0;i<eggpu_weight_capacity;++i){auto found=eggpu_weight_keys[slot];if(found==key)return eggpu_weight_values[slot];if(found==0)break;slot=(slot+1)&(eggpu_weight_capacity-1);}
          // v7: a missing positive-weight edge is an invalid state, including Release builds.
          assert(false);asm volatile("trap;");return 0;
        }

        SSSP(index_t source_node) : m_source_node(source_node)
        {
         
        }

        __forceinline__ __device__

        TValue GetInitValue(index_t node) const override
        {
            return static_cast<TValue> (IDENTITY_ELEMENT);
        }

        __forceinline__ __device__

        TBuffer GetInitBuffer(index_t node) const override 
        {
            TBuffer buffer;
            if (node == m_source_node)//source_node = 1
            {
                buffer = 0;
            }
            else
            {
                buffer = IDENTITY_ELEMENT;
            }
            return buffer;
        }

        __forceinline__ __host__ __device__
        TBuffer GetIdentityElement() const override
        {
            return IDENTITY_ELEMENT;
        }

        __forceinline__ __device__
        utils::pair<TBuffer, bool> CombineValueBufferAmend(index_t node,int *type_device,
                                                      TValue *p_value,
                                                      TBuffer *p_buffer) override
        {
           
            return utils::pair<TBuffer, bool>(*p_buffer, true);
        }
        __forceinline__ __device__
        utils::pair<TBuffer, bool> CombineValueBuffer(index_t node,
                                                      TValue *p_value,
                                                      TBuffer *p_buffer) override
        {
            TBuffer buffer = *p_buffer;
            bool schedule = false;
            if (*p_value > buffer)
            {
                *p_value = buffer;
                schedule = true;
            }
            return utils::pair<TBuffer, bool>(buffer, schedule);
        }

        // 3: if new_path==result[dst] then
        // 4: old_p=parent[dst]
        // 5: repeat
        // 6: old_p= atomicCAS(&parent[dst],src)
        // 7: until (new_path!= result[dst]||old_p==parent[dst])
        __forceinline__ __device__
        int AccumulateBuffer(index_t src,
                             index_t dst,
                            //  TValue level, //src_level
                            //  TValue *p_level, //dst_level
                             TWeight weight,
                             TValue *p_parent,
                             TBuffer *p_buffer,    //dst_buffer
                             TBuffer buffer) override   //src_buffer
        {
            // TBuffer old_buffer = *p_buffer;
            TBuffer new_buffer = buffer + weight;
            TBuffer old_buffer=atomicMin(p_buffer, buffer + weight);;
            if(new_buffer< old_buffer){
            
                TValue old_parent = *p_parent;
                do{
                    old_parent = atomicCAS(p_parent,old_parent,src);
                    
                } while (new_buffer==*p_buffer && (*p_parent) !=src);
            }
            return 1;
        }

        __forceinline__ __device__
        int AccumulateBuffer_add(index_t src,
                             index_t dst,
                            //  TValue level, //src_level
                            //  TValue *p_level, //dst_level
                             TWeight weight,
                             TValue *p_parent, //dst_parent
                             TValue *p_buffer,    //dst_value
                             TValue *buffer) override   //src_value
        {
            TValue incoming_value_curr = *buffer;
            if(incoming_value_curr ==UINT32_MAX){
                return 0;
            }
            TValue new_buffer = incoming_value_curr + weight;
            TValue new_parent = src;
            // TValue new_level = level+1;
            TValue old_value;
            old_value = atomicMin(p_buffer, new_buffer);
            TValue old_parent = *p_parent;
        // TValue curr_buffer = *p_buffer;
            // if(new_buffer == curr_buffer){
            do{
                if(new_buffer==old_value){
                    old_parent = atomicCAS(p_parent,old_parent,src);
                }
            } while (new_buffer==old_value && old_parent !=src);
            return 1;
        }

        // __forceinline__ __device__
        // bool reduce()

        __forceinline__ __device__
        int AccumulateBuffer_del(index_t src,
                             index_t dst,
                             TValue *p_parent,
                             TBuffer *p_buffer,    //dst_buffer
                             TValue *p_value) override   //dst_value
        {
            if(*p_parent == src){
                *p_buffer = UINT32_MAX;
                *p_value = UINT32_MAX;
                *p_parent = UINT32_MAX;
                this->m_vcsr_graph.vertices_[dst].deletion = true;
            }
            return 1;
        }

        __forceinline__ __device__

        bool IsActiveNode(index_t node, TBuffer buffer,TValue value) const override
        {
            return buffer < value;
        }
        
        
        __forceinline__ __device__
        TValue sum_value(index_t node, TValue value,TBuffer buffer) const override
        {
            if(value > buffer * 2)
                return TValue(2);

            return TValue(1);
        }


        __forceinline__ __device__

        bool IsHighPriority(TBuffer current_priority, TBuffer buffer) const override
        {
            return current_priority > buffer;
        }
    };
}


/**
 * Δ = cw/d,
    where d is the average degree in the graph, w is the average
    edge weight, and c is the warp width (32 on our GPUs).
 * @return
 */

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
using EggpuClock=std::chrono::steady_clock;
static double eggpu_seconds(EggpuClock::time_point a,EggpuClock::time_point b){return std::chrono::duration<double>(b-a).count();}
struct EggpuWeightEntry { unsigned u,v,w; };
__host__ __device__ inline unsigned long long eggpu_weight_hash(unsigned long long key){key^=key>>33;key*=0xff51afd7ed558ccdULL;key^=key>>33;key*=0xc4ceb9fe1a85ec53ULL;return key^(key>>33);}
__global__ void eggpu_weight_scatter(const EggpuWeightEntry* entries,size_t count,unsigned long long* keys,unsigned* values,unsigned capacity){
 size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;auto entry=entries[i];unsigned long long key=((static_cast<unsigned long long>(entry.u)<<32)|entry.v)+1;unsigned slot=eggpu_weight_hash(key)&(capacity-1);
 for(unsigned probe=0;probe<capacity;++probe){auto old=atomicCAS(keys+slot,0ULL,key);if(old==0ULL || old==key){values[slot]=entry.w;return;}slot=(slot+1)&(capacity-1);}assert(false);
}
struct EggpuWeightTable {
 unsigned long long* keys=nullptr;unsigned* values=nullptr;unsigned capacity=1;EggpuWeightEntry* transfer=nullptr;size_t transfer_capacity=0;
 std::vector<EggpuWeightEntry> initial;std::vector<std::vector<EggpuWeightEntry>> adds;
 void load(){
  unsigned u,v,w;std::ifstream graph(FLAGS_graphfile);while(graph>>u>>v>>w)initial.push_back({u,v,w});
  std::vector<unsigned> counts;unsigned a,d;std::ifstream sizes(FLAGS_update_size);while(sizes>>a>>d)counts.push_back(a);
  std::vector<EggpuWeightEntry> added;std::string kind;std::ifstream updates(FLAGS_updatefile);while(updates>>kind>>u>>v>>w)if(kind=="a")added.push_back({u,v,w});
  size_t offset=0;for(auto count:counts){adds.emplace_back(added.begin()+offset,added.begin()+offset+count);offset+=count;}if(offset!=added.size())throw std::runtime_error("weight update count mismatch");
  while(capacity<2*(initial.size()+added.size()+1))capacity*=2;
  GROUTE_CUDA_CHECK(cudaMalloc(&keys,capacity*sizeof(*keys)));GROUTE_CUDA_CHECK(cudaMemset(keys,0,capacity*sizeof(*keys)));GROUTE_CUDA_CHECK(cudaMalloc(&values,capacity*sizeof(*values)));
  transfer_capacity=initial.size();for(auto& batch:adds)transfer_capacity=std::max(transfer_capacity,batch.size());transfer_capacity=std::max<size_t>(1,transfer_capacity);GROUTE_CUDA_CHECK(cudaMalloc(&transfer,transfer_capacity*sizeof(*transfer)));apply(initial);
 }
 void apply(const std::vector<EggpuWeightEntry>& entries){if(entries.empty())return;GROUTE_CUDA_CHECK(cudaMemcpy(transfer,entries.data(),entries.size()*sizeof(*transfer),cudaMemcpyHostToDevice));eggpu_weight_scatter<<<(entries.size()+255)/256,256>>>(transfer,entries.size(),keys,values,capacity);GROUTE_CUDA_CHECK(cudaDeviceSynchronize());}
 size_t bytes()const{return capacity*(sizeof(*keys)+sizeof(*values))+transfer_capacity*sizeof(*transfer);}
};

bool HybridSSSP(){
 typedef sepgraph::engine::Engine<distance_t,distance_t,distance_t,hybrid_sssp::SSSP,index_t> HybridEngine;
 const char* raw_policy=std::getenv("EGGPU_GRAPIN_CACHE_POLICY");
 const std::string policy=raw_policy?raw_policy:"keep";
 if(policy!="keep" && policy!="full_hot_detach" && policy!="typed_reclaim" && policy!="typed_borrow" && policy!="typed_sublease")
  throw std::runtime_error("unknown transfer policy");
 const char* raw_mode=std::getenv("EGGPU_GRAPIN_MODE");
 const std::string mode=raw_mode?raw_mode:"qualification";
 if(mode!="qualification" && mode!="formal")throw std::runtime_error("unknown execution mode");
 const bool qualification=mode=="qualification";
 const char* raw_parent_mode=std::getenv("EGGPU_GRAPIN_PARENT_MODE");
 const std::string parent_mode=raw_parent_mode?raw_parent_mode:"repair";
 if(parent_mode!="repair")throw std::runtime_error("v11 requires common parent repair");
 const bool parent_repair=parent_mode=="repair";
 const char* raw_size=std::getenv("EGGPU_GRAPIN_B_N");
 const uint32_t B_n=raw_size?uint32_t(std::stoul(raw_size)):256;
 if(B_n==0 || B_n>1000000)throw std::runtime_error("invalid B size");
 const char* raw_H=std::getenv("EGGPU_GRAPIN_H_N");
 if(!raw_H)throw std::runtime_error("explicit H size required");
 const uint32_t H_n=uint32_t(std::stoul(raw_H));
 if(H_n==0 || H_n>1000000)throw std::runtime_error("invalid H size");
 struct AResult {size_t snapshot;bool empty;double compute,output,complete;std::vector<double> values;};
 std::vector<AResult> results;
 std::vector<double> completed_with_parent;
 GROUTE_CUDA_CHECK(cudaFree(nullptr));
 GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
 const auto service_begin=EggpuClock::now();
 HybridEngine engine(sepgraph::policy::AlgoType::TRAVERSAL_SCHEME);engine.LoadGraph();
 index_t source=FLAGS_source_node;
 if(source>=engine.GetGraphDatum().nnodes)throw std::runtime_error("source outside domain");
 sepgraph::common::EngineOptions options;engine.SetOptions(options);engine.InitGraph(source);
 engine.get_update_file();EggpuWeightTable weights;weights.load();
 engine.SetExplicitWeightTable(weights.keys,weights.values,weights.capacity);
 GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
 const auto prepared=EggpuClock::now();
 std::pair<index_t,index_t> begin{0,0};
 const size_t batches=engine.UpdateBatchCount();
 results.reserve(batches+1);completed_with_parent.reserve(batches+1);
 double calls=0;
 for(size_t snapshot=0;snapshot<=batches;++snapshot){
  GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
  const auto a=EggpuClock::now();bool empty=false;
  if(snapshot==0){
   engine.Start(0);engine.compute_hot_vertices_sssp();engine.confirm_candidate_batch();engine.LoadCache();
  } else {
   index_t batch=snapshot-1;empty=engine.UpdateBatchEmpty(batch);
   if(!empty){
    engine.del_edge(begin,batch);weights.apply(weights.adds[batch]);engine.add_edge(begin,batch);
    engine.compute_hot_vertices_sssp();engine.confirm_candidate_batch();
    engine.evication_cache();engine.compact_cache();engine.LoadCache();
   }
  }
  GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
  const auto computed=EggpuClock::now();
  engine.GatherValue();const auto& values=engine.GetGraphDatum().host_value;
  std::vector<double> out(values.size());
  for(size_t i=0;i<values.size();++i)
   out[i]=values[i]==UINT32_MAX?std::numeric_limits<double>::infinity():double(values[i]);
  GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
  const auto complete=EggpuClock::now();
  const double elapsed=eggpu_seconds(a,complete);
  results.push_back({snapshot,empty,eggpu_seconds(a,computed),eggpu_seconds(computed,complete),
                     elapsed,std::move(out)});
  calls+=elapsed;
  engine.EggpuParentCheckpointV6(snapshot,source,parent_repair,qualification);
  completed_with_parent.push_back(eggpu_seconds(a,EggpuClock::now()));
  if(qualification)engine.EggpuCheckPartialCacheV2(snapshot);
  if(policy!="keep" && snapshot<batches){
   const auto cursor=begin;
   engine.EggpuTransferV11(snapshot,policy,B_n,H_n,qualification);
   if(cursor!=begin)throw std::runtime_error("A update cursor changed");
  }
 }
 GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
 const auto service_end=EggpuClock::now();
 // Every owning A and B result survives subsequent A calls; no archive or CPU oracle
 // runs inside the formal continuous service interval.
 double active=eggpu_seconds(service_begin,prepared);
 for(const auto& a:results){
  const auto path=FLAGS_output+"."+std::to_string(a.snapshot)+".f64";
  std::ofstream output(path,std::ios::binary);
  output.write(reinterpret_cast<const char*>(a.values.data()),a.values.size()*sizeof(double));
  if(!output)throw std::runtime_error("A output archive failed");
  active+=a.complete;
  std::cout<<std::setprecision(17)<<"EGGPU_GRAPIN_JSON {\"snapshot\":"<<a.snapshot
   <<",\"nodes\":"<<a.values.size()<<",\"empty_update\":"<<(a.empty?"true":"false")
   <<",\"initial_preparation_seconds\":"<<eggpu_seconds(service_begin,prepared)
   <<",\"update_and_compute_seconds\":"<<a.compute<<",\"full_host_output_seconds\":"<<a.output
   <<",\"update_complete_with_parent_seconds\":"<<completed_with_parent[a.snapshot]
   <<",\"complete_call_seconds\":"<<a.complete<<",\"active_session_seconds\":"<<active<<"}"<<std::endl;
 }
 engine.EggpuArchiveTransfersV11();
 double hooks=0;
 for(const auto& transfer:engine.eggpu_transfers_v11)hooks+=transfer.hook_seconds;
 std::cout<<std::setprecision(17)<<"EGGPU_GRAPIN_SESSION_V11_JSON {\"policy\":"<<std::quoted(policy)
  <<",\"mode\":"<<std::quoted(mode)<<",\"snapshots\":"<<results.size()
  <<",\"B_outputs\":"<<engine.eggpu_transfers_v11.size()
  <<",\"H_outputs\":"<<engine.eggpu_transfers_v11.size()
  <<",\"initial_preparation_seconds\":"<<eggpu_seconds(service_begin,prepared)
  <<",\"parent_mode\":"<<std::quoted(parent_mode)
  <<",\"parent_maintenance_seconds\":"<<engine.eggpu_parent_maintenance_seconds_v6
  <<",\"A_complete_calls_seconds\":"<<calls<<",\"hook_seconds\":"<<hooks
  <<",\"complete_service_wall_seconds\":"<<eggpu_seconds(service_begin,service_end)
  <<",\"active_workflow_wall_seconds\":"<<eggpu_seconds(prepared,service_end)
  <<",\"archive_after_service_interval\":true,\"B_distance_D2H_inside_interval\":true"
  <<",\"H_distance_D2H_inside_interval\":true,\"controlled_H_arrival_after_B_compute\":true"
  <<",\"extra_validation_inside_interval\":"<<(qualification?"true":"false")<<"}"<<std::endl;
 std::cout<<"AWAIT_RELEASE"<<std::endl;std::string release;std::getline(std::cin,release);
 if(release!="Q")throw std::runtime_error("explicit context release required");
 return true;
}
