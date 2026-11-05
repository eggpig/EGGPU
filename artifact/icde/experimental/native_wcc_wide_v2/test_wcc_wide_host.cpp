#include "gpu_easygraph/runtime/partitioned_wcc_plan.h"
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <iostream>
using namespace gpu_easygraph::runtime;
static std::vector<WccHostPartition> fixture() {
    WccHostPartition a; a.partition_id=0; a.device_id=0; a.owned_vertex_count=3;
    a.offsets={0,1,3,4}; a.indices={1,0,3,3};a.local_to_global={0,1,2,3};a.owners={0,0,0,1};
    WccHostPartition b; b.partition_id=1;b.device_id=1;b.owned_vertex_count=3;
    b.offsets={0,2,2,2};b.indices={3,4};b.local_to_global={3,4,5,1,2};b.owners={1,1,1,0,0};
    WccHostPartition empty;empty.partition_id=2;empty.device_id=2;empty.owned_vertex_count=0;empty.offsets={0};
    return {a,b,empty};
}
static void widen(std::vector<WccHostPartition>& parts) {
    for(auto& p:parts) {p.wide_offsets=true;p.offsets64.assign(p.offsets.begin(),p.offsets.end());p.offsets.clear();}
}
static bool rejected(const std::vector<WccHostPartition>& parts) {
    try {build_partitioned_wcc_plan(parts,6);return false;}catch(const std::invalid_argument&){return true;}
}
int main() {
    auto narrow=fixture(),wide=narrow;widen(wide);
    auto a=build_partitioned_wcc_plan(narrow,6),b=build_partitioned_wcc_plan(wide,6);
    const std::vector<std::uint64_t> expected={0,0,0,0,4,5};
    assert(partitioned_wcc_reference(a)==expected);
    assert(partitioned_wcc_reference(b)==expected);
    assert(a.halo_value_count==b.halo_value_count && a.routes.size()==b.routes.size());
    for(std::size_t i=0;i<a.routes.size();++i) {
        assert(a.routes[i].source_owned_local_ids==b.routes[i].source_owned_local_ids);
        assert(a.routes[i].destination_halo_local_ids==b.routes[i].destination_halo_local_ids);
    }
    auto bad=wide;bad[0].offsets={0};assert(rejected(bad));
    bad=wide;bad[0].offsets64[1]=-1;assert(rejected(bad));
    bad=wide;bad[0].offsets64[2]=INT64_C(2147483648);assert(rejected(bad));
    bad=wide;bad[0].offsets64.back()=INT64_C(4294967313);assert(rejected(bad));
    bad=wide;bad[2].offsets64.clear();assert(rejected(bad));
    bad=wide;bad[2].offsets64={1};assert(rejected(bad));
    std::cout << "PASS_WCC_HOST_NARROW_WIDE_EQUIVALENCE_AND_INVALID_DOMAIN_REJECTION\n";
}
