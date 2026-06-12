#include <vector>

#include "./common/err.h"

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

int clustering(
    const std::vector<int>& V,
    const std::vector<int>& E,
    bool directed,
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
    const std::vector<int>& row,
    const std::vector<int>& col,
    int num_nodes,
    const std::vector<double>& W,
    bool is_directed,
    std::vector<int>& node_mask, 
    std::vector<double>& effective_size,
    double* kernel_seconds = nullptr
);

} // namespace gpu_easygraph
