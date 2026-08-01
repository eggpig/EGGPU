#include <limits>
#include <vector>

#include "path/sssp_dijkstra.cuh"
#include "path/mst.cuh"
#include "common.h"

namespace gpu_easygraph {

using std::vector;

static int decide_warp_size(
    _IN_ int len_V,
    _IN_ int len_E
)
{
    vector<int> warp_size_cand{1, 2, 4, 8, 16, 32};

    if (len_E / len_V < warp_size_cand.front()) {
        return warp_size_cand.front();
    }

    for (int i = 0; i + 1 < warp_size_cand.size(); ++i) {
        if (warp_size_cand[i] <= len_E / len_V
                && len_E / len_V < warp_size_cand[i + 1]) {
            return warp_size_cand[i + 1];
        }
    }
    return warp_size_cand.back();
}



int sssp_dijkstra(
    _IN_ const vector<int>& V,
    _IN_ const vector<int>& E,
    _IN_ const vector<double>& W,
    _IN_ const vector<int>& sources,
    _IN_ int target,
    _OUT_ vector<double>& res,
    _IN_ double* kernel_seconds
)
{
    int len_V = V.size() - 1;
    int len_E = E.size();

    int warp_size = decide_warp_size(len_V, len_E);

    res = vector<double>(sources.size() * len_V);
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;

    int r = cuda_sssp_dijkstra(V.data(), E.data(), W.data(),
            sources.data(), len_V, len_E, sources.size(),
            target, warp_size, res.data(), kernel_seconds);

    double double_inf = std::numeric_limits<double>::infinity();
    for (int i = 0; i < res.size(); ++i) {
        if (res[i] >= EG_DOUBLE_INF) {
            res[i] = double_inf;
        }
    }

    return r;
}

int sssp_unweighted_bfs(
    _IN_ const vector<int>& V,
    _IN_ const vector<int>& E,
    _IN_ const vector<int>& sources,
    _IN_ int target,
    _OUT_ vector<double>& res,
    _IN_ double* kernel_seconds
)
{
    int len_V = V.size() - 1;
    int len_E = E.size();

    res = vector<double>(sources.size() * len_V);
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;

    int r = cuda_sssp_unweighted_bfs(V.data(), E.data(),
            sources.data(), len_V, len_E, sources.size(),
            target, res.data(), kernel_seconds);

    double double_inf = std::numeric_limits<double>::infinity();
    for (int i = 0; i < res.size(); ++i) {
        if (res[i] >= EG_DOUBLE_INF) {
            res[i] = double_inf;
        }
    }

    return r;
}

int sssp_bellman_ford(
    _IN_ const vector<int>& V,
    _IN_ const vector<int>& E,
    _IN_ const vector<double>& W,
    _IN_ const vector<int>& sources,
    _IN_ int target,
    _OUT_ vector<double>& res,
    _IN_ double* kernel_seconds
)
{
    int len_V = V.size() - 1;
    int len_E = E.size();

    res = vector<double>(sources.size() * len_V);
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;

    int r = cuda_sssp_bellman_ford(V.data(), E.data(), W.data(),
            sources.data(), len_V, len_E, sources.size(),
            target, res.data(), kernel_seconds);

    double double_inf = std::numeric_limits<double>::infinity();
    for (int i = 0; i < res.size(); ++i) {
        if (res[i] >= EG_DOUBLE_INF) {
            res[i] = double_inf;
        }
    }

    return r;
}


int mst(
    _IN_ const vector<int>& V,
    _IN_ const vector<int>& E,
    _IN_ const vector<double>& W,
    _OUT_ vector<int>& mst_src,
    _OUT_ vector<int>& mst_dst,
    _OUT_ vector<double>& mst_weight,
    _IN_ double* kernel_seconds
) {
    int len_V = V.size() - 1;
    int len_E = E.size();
    if (len_V <= 0 || len_E <= 0) {
        mst_src.clear();
        mst_dst.clear();
        mst_weight.clear();
        if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
        return EG_GPU_SUCC;
    }
    return cuda_mst(
        V.data(),
        E.data(),
        W.data(),
        len_V,
        len_E,
        mst_src,
        mst_dst,
        mst_weight,
        kernel_seconds
    );
}

int mst_single_incidence(
    _IN_ const vector<int>& V,
    _IN_ const vector<int>& E,
    _IN_ const vector<double>& W,
    _OUT_ vector<int>& mst_src,
    _OUT_ vector<int>& mst_dst,
    _OUT_ vector<double>& mst_weight,
    _IN_ double* kernel_seconds
) {
    const int len_V = static_cast<int>(V.size()) - 1;
    const int len_E = static_cast<int>(E.size());
    if (len_V <= 0 || len_E <= 0) {
        mst_src.clear();
        mst_dst.clear();
        mst_weight.clear();
        if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
        return EG_GPU_SUCC;
    }
    if (!W.empty() && W.size() != E.size()) {
        return EG_GPU_UNKNOW_ERROR;
    }
    return cuda_mst_single_incidence(
        V.data(),
        E.data(),
        W.empty() ? nullptr : W.data(),
        len_V,
        len_E,
        mst_src,
        mst_dst,
        mst_weight,
        kernel_seconds
    );
}

} // namespace gpu_easygraph
