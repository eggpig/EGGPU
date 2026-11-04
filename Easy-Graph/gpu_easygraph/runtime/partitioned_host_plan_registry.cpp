#include "partitioned_host_plan_registry.h"

namespace gpu_easygraph {
namespace runtime {
namespace {

const std::size_t kHostPlanCacheCapacityPerAlgorithm = 4;

}  // namespace

PartitionedHostPlanCache<PartitionedBfsPlan>&
partitioned_bfs_host_plan_cache() {
    static PartitionedHostPlanCache<PartitionedBfsPlan> cache(
        kHostPlanCacheCapacityPerAlgorithm);
    return cache;
}

PartitionedHostPlanCache<PartitionedWccPlan>&
partitioned_wcc_host_plan_cache() {
    static PartitionedHostPlanCache<PartitionedWccPlan> cache(
        kHostPlanCacheCapacityPerAlgorithm);
    return cache;
}

PartitionedHostPlanCache<PartitionedSsspPlan>&
partitioned_sssp_host_plan_cache() {
    static PartitionedHostPlanCache<PartitionedSsspPlan> cache(
        kHostPlanCacheCapacityPerAlgorithm);
    return cache;
}

PartitionedHostPlanCache<PartitionedEgoPlan>&
partitioned_ego_host_plan_cache() {
    static PartitionedHostPlanCache<PartitionedEgoPlan> cache(
        kHostPlanCacheCapacityPerAlgorithm);
    return cache;
}

PartitionedHostPlanCacheClearResult clear_partitioned_host_plan_caches() {
    PartitionedHostPlanCacheClearResult result;
    result.bfs = partitioned_bfs_host_plan_cache().clear();
    result.wcc = partitioned_wcc_host_plan_cache().clear();
    result.sssp = partitioned_sssp_host_plan_cache().clear();
    result.ego = partitioned_ego_host_plan_cache().clear();
    result.total = result.bfs + result.wcc + result.sssp + result.ego;
    return result;
}

PartitionedHostPlanCacheRegistryStats partitioned_host_plan_cache_stats() {
    PartitionedHostPlanCacheRegistryStats result;
    result.bfs = partitioned_bfs_host_plan_cache().stats();
    result.wcc = partitioned_wcc_host_plan_cache().stats();
    result.sssp = partitioned_sssp_host_plan_cache().stats();
    result.ego = partitioned_ego_host_plan_cache().stats();
    return result;
}

}  // namespace runtime
}  // namespace gpu_easygraph
