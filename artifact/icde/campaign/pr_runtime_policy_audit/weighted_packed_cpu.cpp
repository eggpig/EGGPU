#include "partitioned_pagerank_reference.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>
using namespace gpu_easygraph::runtime;
struct Edge { int u,v; double w; };
std::vector<PageRankOutgoingPartition> make(int n,int p,const std::vector<Edge>& edges) {
 std::vector<int> cut(p+1);for(int i=0;i<=p;++i)cut[i]=n*i/p;
 auto owner=[&](int v){return int(std::upper_bound(cut.begin()+1,cut.end(),v)-cut.begin()-1);};
 std::vector<PageRankOutgoingPartition> out(p);
 for(int rank=0;rank<p;++rank){auto& a=out[rank];a.partition_id=rank;a.device_id=rank;a.owned_vertex_count=cut[rank+1]-cut[rank];
  for(int v=cut[rank];v<cut[rank+1];++v)a.local_to_global.push_back(v);
  std::set<int> halo;for(auto e:edges)if(owner(e.u)==rank && owner(e.v)!=rank)halo.insert(e.v);
  for(int v:halo)a.local_to_global.push_back(v);for(auto v:a.local_to_global)a.owners.push_back(owner(v));
  a.outgoing_offsets.push_back(0);
  for(int u=cut[rank];u<cut[rank+1];++u){for(auto e:edges)if(e.u==u){auto i=std::find(a.local_to_global.begin(),a.local_to_global.end(),e.v);assert(i!=a.local_to_global.end());a.outgoing_destinations.push_back(i-a.local_to_global.begin());a.outgoing_weights.push_back(e.w);}a.outgoing_offsets.push_back(a.outgoing_destinations.size());}
 }
 return out;
}
void same_input(const std::vector<PageRankOutgoingPartition>& a,const std::vector<PageRankOutgoingPartition>& b){assert(a.size()==b.size());for(size_t i=0;i<a.size();++i){assert(a[i].outgoing_offsets==b[i].outgoing_offsets);assert(a[i].outgoing_destinations==b[i].outgoing_destinations);assert(a[i].outgoing_weights==b[i].outgoing_weights);assert(a[i].local_to_global==b[i].local_to_global);assert(a[i].owners==b[i].owners);}}
void same(const std::vector<PageRankHostPartition>& a,const std::vector<PageRankHostPartition>& b){assert(a.size()==b.size());for(size_t i=0;i<a.size();++i){assert(a[i].partition_id==b[i].partition_id);assert(a[i].device_id==b[i].device_id);assert(a[i].owned_vertex_count==b[i].owned_vertex_count);assert(a[i].incoming_offsets==b[i].incoming_offsets);assert(a[i].incoming_sources==b[i].incoming_sources);assert(a[i].incoming_weights==b[i].incoming_weights);assert(a[i].local_to_global==b[i].local_to_global);assert(a[i].owners==b[i].owners);assert(a[i].owned_out_weight==b[i].owned_out_weight);}}
std::vector<double> dense(int n,const std::vector<Edge>& edges){std::vector<double> x(n,1./n),s(n);for(auto e:edges)s[e.u]+=e.w;for(int it=0;it<2000;++it){double d=0;for(int u=0;u<n;++u)if(s[u]==0)d+=x[u];std::vector<double> y(n,(1.-.85+.85*d)/n);for(auto e:edges)if(s[e.u]>0)y[e.v]+=.85*x[e.u]*e.w/s[e.u];double err=0;for(int u=0;u<n;++u)err+=std::abs(x[u]-y[u]);x=y;if(err<1e-14)break;}return x;}
int main(){int cases=0,exact_arrays=0,weight_changes=0;double maxerr=0;
 for(int n:{3,17})for(int p:{1,2,4,8}){if(n<p)continue;std::vector<Edge> edges;
  for(int u=0;u<n-1;++u){edges.push_back({u,(u+1)%n,double(1+(u*7)%16)});if(n>3)edges.push_back({u,(u*5+3)%n,double(1+(u*11)%16)});}
  edges.push_back({0,n-1,0.});edges.push_back({0,n-1,3.});
  std::vector<double> prior;
  auto original=make(n,p,edges);auto unchanged=original;
  for(int version=0;version<2;++version){auto data=original;if(version){assert(!data[0].outgoing_weights.empty());data[0].outgoing_weights[0]+=1.;}
   auto saved=data;auto general=build_destination_owned_page_rank(data,n,true);auto implicit=build_destination_owned_page_rank(data,n,false);PackedRouteTransposeTimings time;auto packed=build_destination_owned_page_rank_packed(data,n,false,&time);auto packed_materialized=build_destination_owned_page_rank_packed(data,n,true);same(general,implicit);same(general,packed);same(general,packed_materialized);exact_arrays+=3;
   if(p==1){auto whole=build_whole_contiguous_page_rank(data,n,false);same(general,whole);++exact_arrays;}
   same_input(data,saved);same_input(original,unchanged);validate_page_rank_partitions(packed,n);
   PartitionedPageRankOptions o;o.alpha=.85;o.tolerance=1e-13;o.max_iterations=2000;auto g=partitioned_page_rank_reference(general,n,o),fast=partitioned_page_rank_reference(packed,n,o);assert(g.converged&&fast.converged);assert(g.values==fast.values);
   std::vector<Edge> actual;for(auto a:data)for(size_t u=0;u<a.owned_vertex_count;++u)for(int k=a.outgoing_offsets[u];k<a.outgoing_offsets[u+1];++k)actual.push_back({int(a.local_to_global[u]),int(a.local_to_global[a.outgoing_destinations[k]]),a.outgoing_weights[k]});auto independent=dense(n,actual);
   for(int u=0;u<n;++u){maxerr=std::max(maxerr,std::abs(g.values[u]-independent[u]));assert(std::abs(g.values[u]-independent[u])<1e-11);}
   if(version){assert(prior!=fast.values);++weight_changes;}else prior=fast.values;
   ++cases;
  }
 }
 // Invalid weight must still be rejected by the packed path.
 auto invalid=make(3,2,{{0,1,-1.},{1,2,2.}});bool rejected=false;try{build_destination_owned_page_rank_packed(invalid,3,false);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
 std::cout<<"{\"status\":\"PASS_WEIGHTED_PACKED_CPU_EQUIVALENCE\",\"cases\":"<<cases<<",\"exact_partition_array_comparisons\":"<<exact_arrays<<",\"weight_changed_topology_fixed_cases\":"<<weight_changes<<",\"independent_dense_max_abs_error\":"<<maxerr<<",\"negative_weight_rejected\":true,\"gpu_executed\":false,\"native_dynamic_patch_executed\":false,\"device_pull_only_validated_by_this_test\":false}\n";
}
