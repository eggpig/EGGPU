#pragma once

#include "partitioned_bfs_plan.h"
#include "partitioned_ego_plan.h"
#include "partitioned_host_plan_cache.h"
#include "partitioned_sssp_plan.h"
#include "partitioned_wcc_plan.h"

#include <cstddef>

namespace gpu_easygraph {
namespace runtime {

struct PartitionedHostPlanCacheClearResult {
    std::size_t bfs;
    std::size_t wcc;
    std::size_t sssp;
    std::size_t ego;
    std::size_t total;

    PartitionedHostPlanCacheClearResult()
        : bfs(0), wcc(0), sssp(0), ego(0), total(0) {}
};

struct PartitionedHostPlanCacheRegistryStats {
    PartitionedHostPlanCacheStats bfs;
    PartitionedHostPlanCacheStats wcc;
    PartitionedHostPlanCacheStats sssp;
    PartitionedHostPlanCacheStats ego;
};

PartitionedHostPlanCache<PartitionedBfsPlan>&
partitioned_bfs_host_plan_cache();
PartitionedHostPlanCache<PartitionedWccPlan>&
partitioned_wcc_host_plan_cache();
PartitionedHostPlanCache<PartitionedSsspPlan>&
partitioned_sssp_host_plan_cache();
PartitionedHostPlanCache<PartitionedEgoPlan>&
partitioned_ego_host_plan_cache();

PartitionedHostPlanCacheClearResult clear_partitioned_host_plan_caches();
PartitionedHostPlanCacheRegistryStats partitioned_host_plan_cache_stats();

}  // namespace runtime
}  // namespace gpu_easygraph
