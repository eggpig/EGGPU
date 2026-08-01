#pragma once

#include <cstdint>
#include <vector>

#include "./common/err.h"
#include "./common/device_graph_cache.h"

namespace gpu_easygraph {

int closeness_centrality(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<double>& W,
    const std::vector<int>& sources,
    bool unweighted,
    std::vector<double>& CC,
    double* kernel_seconds = nullptr
);



int betweenness_centrality(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<double>& W,
    const std::vector<int>& sources,
    bool is_directed,
    bool normalized,
    bool endpoints,
    bool unweighted,
    std::vector<double>& BC,
    double* kernel_seconds = nullptr
);



int k_core(
    const std::vector<int>& V,
    const std::vector<int>& E,
    std::vector<int>& KC,
    double* kernel_seconds = nullptr
);

int k_core_into(
    const std::vector<int>& V,
    const std::vector<int>& E,
    int* KC,
    double* kernel_seconds = nullptr
);

int k_core_split_into(
    const std::vector<int>& lower_V,
    const std::vector<int>& lower_E,
    const std::vector<int>& upper_V,
    const std::vector<int>& upper_E,
    const std::vector<int>& degree,
    int* KC,
    double* kernel_seconds = nullptr
);



int sssp_dijkstra(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<double>& W,
    const std::vector<int>& sources,
    int target,
    std::vector<double>& res,
    double* kernel_seconds = nullptr
);

int sssp_unweighted_bfs(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& sources,
    int target,
    std::vector<double>& res,
    double* kernel_seconds = nullptr
);

int sssp_bellman_ford(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<double>& W,
    const std::vector<int>& sources,
    int target,
    std::vector<double>& res,
    double* kernel_seconds = nullptr
);



int pagerank(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<double>& W,
    double alpha,
    int max_iter_num,
    double threshold,
    std::vector<double>& PR,
    double* kernel_seconds = nullptr
);


int mst(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<double>& W,
    std::vector<int>& mst_src,
    std::vector<int>& mst_dst,
    std::vector<double>& mst_weight,
    double* kernel_seconds = nullptr
);

int mst_single_incidence(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<double>& W,
    std::vector<int>& mst_src,
    std::vector<int>& mst_dst,
    std::vector<double>& mst_weight,
    double* kernel_seconds = nullptr
);

int clustering(
    const std::vector<int>& V,
    const std::vector<int>& E,
    bool directed,
    std::vector<double>& CC,
    double* kernel_seconds = nullptr
);

int clustering_forward(
    const std::vector<int>& forward_V,
    const std::vector<int>& forward_E,
    const std::vector<int>& degree,
    std::vector<double>& CC,
    double* kernel_seconds = nullptr
);

int connected_components(
    const std::vector<int>& V,
    const std::vector<int>& E,
    bool directed,
    std::vector<int>& labels,
    double* kernel_seconds = nullptr
);



int constraint(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& in_V,
    const std::vector<int>& in_E,
    const std::vector<int>& row,
    const std::vector<int>& col,
    int num_nodes,
    const std::vector<double>& W,
    bool is_directed,
    std::vector<int>& node_mask,
    std::vector<double>& constraint,
    double* kernel_seconds = nullptr
);



int hierarchy(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& in_V,
    const std::vector<int>& in_E,
    const std::vector<int>& row,
    const std::vector<int>& col,
    int num_nodes,
    const std::vector<double>& W,
    bool is_directed,
    std::vector<int>& node_mask, 
    std::vector<double>& hierarchy,
    double* kernel_seconds = nullptr
);



int effective_size(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& in_V,
    const std::vector<int>& in_E,
    const std::vector<int>& row,
    const std::vector<int>& col,
    int num_nodes,
    const std::vector<double>& W,
    bool is_directed,
    std::vector<int>& node_mask, 
    std::vector<double>& effective_size,
    double* kernel_seconds = nullptr
);

int effective_size_ego_edge_statistics(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& forward_V,
    const std::vector<int>& forward_E,
    const std::vector<int>& degree,
    std::uint64_t graph_id,
    std::vector<double>& result,
    double* kernel_seconds = nullptr
);

int constraint_ego_edge_statistics(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& forward_V,
    const std::vector<int>& forward_E,
    const std::vector<int>& degree,
    std::uint64_t graph_id,
    std::vector<double>& result,
    double* kernel_seconds = nullptr
);

int hierarchy_ego_edge_statistics(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& forward_V,
    const std::vector<int>& forward_E,
    const std::vector<int>& degree,
    std::uint64_t graph_id,
    std::vector<double>& result,
    double* kernel_seconds = nullptr
);

} // namespace gpu_easygraph
