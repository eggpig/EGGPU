#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "structural_holes/constraint.cuh"
#include "common.h"

namespace gpu_easygraph {

using std::vector;

static bool all_unit_weights(const vector<double>& W) {
    return std::all_of(W.begin(), W.end(), [](double w) { return w == 1.0; });
}

int constraint(
    _IN_ const vector<int>& V,
    _IN_ const vector<int>& E,
    _IN_ const vector<int>& in_V,
    _IN_ const vector<int>& in_E,
    _IN_ const vector<int>& row,
    _IN_ const vector<int>& col,
    _IN_ int num_nodes,
    _IN_ const vector<double>& W,
    _IN_ bool is_directed,
    _IN_ vector<int>& node_mask,
    _OUT_ vector<double>& constraint,
    _OUT_ double* kernel_seconds
) {
    int num_edges = E.size();
    constraint = vector<double>(num_nodes);
    int r = cuda_constraint(
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
        constraint.data(),
        kernel_seconds
    );

    return r;
}

} // namespace gpu_easygraph
