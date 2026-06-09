#pragma once

#include <cstddef>

namespace gpu_easygraph {

struct HostCsrStats {
    int vertices = 0;
    int edges = 0;
    int max_degree = 0;
    int nonzero_degree_vertices = 0;
    double avg_degree = 0.0;
    bool ready = false;
};

bool adaptive_policy_enabled();

HostCsrStats summarize_host_csr(const int* V, int len_V, int len_E);

bool should_use_weighted_frontier_sssp(const HostCsrStats& stats, int len_V, int len_E);

} // namespace gpu_easygraph
