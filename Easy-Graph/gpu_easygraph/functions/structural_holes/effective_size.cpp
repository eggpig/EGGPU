#include <string>
#include <vector>
#include <memory> 
#include <algorithm>

#include "structural_holes/effective_size.cuh"
#include "common.h"

namespace gpu_easygraph {

using std::vector;

static bool all_unit_weights(const vector<double>& W) {
    return std::all_of(W.begin(), W.end(), [](double w) { return w == 1.0; });
}

static void build_in_csr(
    int num_nodes,
    const vector<int>& row,
    const vector<int>& col,
    vector<int>& in_V,
    vector<int>& in_E
) {
    in_V.assign(num_nodes + 1, 0);
    in_E.assign(row.size(), 0);
    for (int dst : col) {
        if (dst >= 0 && dst < num_nodes) in_V[dst + 1]++;
    }
    for (int i = 0; i < num_nodes; ++i) in_V[i + 1] += in_V[i];
    vector<int> cursor = in_V;
    for (size_t i = 0; i < row.size(); ++i) {
        int dst = col[i];
        if (dst >= 0 && dst < num_nodes) in_E[cursor[dst]++] = row[i];
    }
}

int effective_size(
    _IN_ const vector<int>& V,
    _IN_ const vector<int>& E,
    _IN_ const vector<int>& row,
    _IN_ const vector<int>& col,
    _IN_ int num_nodes,
    _IN_ const vector<double>& W,
    _IN_ bool is_directed,
    _IN_ vector<int>& node_mask,  
    _OUT_ vector<double>& effective_size,
    _OUT_ double* kernel_seconds
) {
    int num_edges = row.size();
    vector<int> in_V;
    vector<int> in_E;
    build_in_csr(num_nodes, row, col, in_V, in_E);
    
    effective_size = vector<double>(num_nodes);
    int r = cuda_effective_size(
        V.data(),
        E.data(),
        in_V.data(),
        in_E.data(),
        row.data(),
        col.data(),
        W.data(),
        num_nodes,
        num_edges,
        is_directed,
        all_unit_weights(W),
        node_mask.data(),
        effective_size.data(),
        kernel_seconds
    );

    return r; 
}

} // namespace gpu_easygraph
