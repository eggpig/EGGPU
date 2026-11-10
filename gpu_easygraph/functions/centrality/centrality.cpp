#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>

#include "centrality/closeness_centrality.cuh"
#include "centrality/betweenness_centrality.cuh"
#include "centrality/pagerank.cuh"
#include "common.h"

namespace gpu_easygraph {

using std::pair;
using std::string;
using std::vector;

static int decide_warp_size (
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

static std::size_t bc_source_chunk_limit()
{
    const char* v = std::getenv("EASYGRAPH_GPU_BC_SOURCE_CHUNK");
    if (v == nullptr) return 0;
    unsigned long long parsed = std::strtoull(v, nullptr, 10);
    return (std::size_t)parsed;
}

static int bc_warp_size_override(int fallback)
{
    const char* v = std::getenv("EASYGRAPH_GPU_BC_WARP_SIZE");
    if (v == nullptr) return fallback;
    char* end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v || *end != '\0') return fallback;
    if (parsed == 1 || parsed == 2 || parsed == 4 ||
        parsed == 8 || parsed == 16 || parsed == 32) {
        return (int)parsed;
    }
    return fallback;
}

static int decide_bc_warp_size(int len_V, int len_E, bool is_directed)
{
    int fallback = decide_warp_size(len_V, len_E);
    if (is_directed && len_V >= 300000) {
        double avg_degree = len_V > 0 ? (double)len_E / (double)len_V : 0.0;
        if (avg_degree <= 8.0) return 2;
        if (avg_degree <= 16.0) return std::min(fallback, 4);
    }
    return fallback;
}



int closeness_centrality(
    _IN_ const std::vector<int>& V,
    _IN_ const std::vector<int>& E,
    _IN_ const std::vector<double>& W,
    _IN_ const std::vector<int>& sources,
    _IN_ bool unweighted,
    _OUT_ std::vector<double>& CC,
    _IN_ double* kernel_seconds
) {
    int len_V = V.size() - 1;
    int len_E = E.size();

    int warp_size = decide_warp_size(len_V, len_E);
    
    CC = vector<double>(sources.size());

    int r = cuda_closeness_centrality(V.data(), E.data(), W.data(), 
            sources.data(), len_V, len_E, sources.size(), 
            warp_size, unweighted, CC.data(), kernel_seconds);
        
    return r;
}



int betweenness_centrality(
    _IN_ const std::vector<int>& V,
    _IN_ const std::vector<int>& E,
    _IN_ const std::vector<double>& W,
    _IN_ const std::vector<int>& sources,
    _IN_ bool is_directed,
    _IN_ bool normalized,
    _IN_ bool endpoints,
    _IN_ bool unweighted,
    _OUT_ std::vector<double>& BC,
    _IN_ double* kernel_seconds
) {
    int len_V = V.size() - 1;
    int len_E = E.size();

    int warp_size = bc_warp_size_override(decide_bc_warp_size(len_V, len_E, is_directed));

    BC = vector<double>(len_V);
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;

    std::size_t chunk_limit = bc_source_chunk_limit();
    if (chunk_limit > 0 && sources.size() > chunk_limit) {
        vector<double> acc(len_V, 0.0);
        for (std::size_t off = 0; off < sources.size(); off += chunk_limit) {
            std::size_t end = std::min(off + chunk_limit, sources.size());
            vector<int> chunk_sources(sources.begin() + off, sources.begin() + end);
            vector<double> part(len_V, 0.0);
            double chunk_kernel_seconds = 0.0;
            int r = cuda_betweenness_centrality(V.data(), E.data(), W.data(),
                    chunk_sources.data(), len_V, len_E, chunk_sources.size(),
                    warp_size, is_directed, normalized, endpoints, unweighted, part.data(),
                    &chunk_kernel_seconds);
            if (r != EG_GPU_SUCC) return r;
            for (int i = 0; i < len_V; ++i) {
                acc[i] += part[i];
            }
            if (kernel_seconds != nullptr) {
                *kernel_seconds += chunk_kernel_seconds;
            }
        }
        BC.swap(acc);
        return EG_GPU_SUCC;
    }

    int r = cuda_betweenness_centrality(V.data(), E.data(), W.data(),
            sources.data(), len_V, len_E, sources.size(),
            warp_size, is_directed, normalized, endpoints, unweighted, BC.data(),
            kernel_seconds);

    return r;
}


int pagerank(
    _IN_ const std::vector<int>& V,
    _IN_ const std::vector<int>& E,
    _IN_ const std::vector<double>& W,
    _IN_ double alpha,
    _IN_ int max_iter_num,
    _IN_ double threshold,
    _OUT_ std::vector<double>& PR,
    _IN_ double* kernel_seconds
) {
    int len_V = V.size() - 1;
    int len_E = E.size();
    if (len_V <= 0) {
        PR.clear();
        if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
        return EG_GPU_SUCC;
    }
    return cuda_pagerank(
        V.data(),
        E.data(),
        W.empty() ? nullptr : W.data(),
        len_V,
        len_E,
        alpha,
        max_iter_num,
        threshold,
        PR,
        kernel_seconds
    );
}

} // namespace gpu_easygraph
