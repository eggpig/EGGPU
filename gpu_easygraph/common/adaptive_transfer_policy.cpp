#include "adaptive_transfer_policy.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>

namespace gpu_easygraph {

namespace {

bool env_bool(const char* name, bool default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) return default_value;
    std::string s(raw);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char)std::toupper(c);
    });
    return s == "1" || s == "TRUE" || s == "ON" || s == "YES" || s == "AUTO";
}

int env_int(const char* name, int default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) return default_value;
    if (std::strcmp(raw, "AUTO") == 0 || std::strcmp(raw, "auto") == 0) {
        return default_value;
    }
    char* end = nullptr;
    long v = std::strtol(raw, &end, 10);
    if (end == raw) return default_value;
    if (v < 0) return 0;
    if (v > 2000000000L) return 2000000000;
    return (int)v;
}

double env_double(const char* name, double default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) return default_value;
    char* end = nullptr;
    double v = std::strtod(raw, &end);
    if (end == raw) return default_value;
    if (v < 0.0) return 0.0;
    return v;
}

bool env_present(const char* name) {
    return std::getenv(name) != nullptr;
}

} // namespace

bool adaptive_policy_enabled() {
    return env_bool("EASYGRAPH_GPU_ADAPTIVE_POLICY", true);
}

HostCsrStats summarize_host_csr(const int* V, int len_V, int len_E) {
    HostCsrStats stats;
    stats.vertices = std::max(0, len_V);
    stats.edges = std::max(0, len_E);
    if (V == nullptr || len_V <= 0) {
        stats.ready = true;
        return stats;
    }

    int max_degree = 0;
    int nonzero = 0;
    for (int u = 0; u < len_V; ++u) {
        int begin = V[u];
        int end = V[u + 1];
        if (begin < 0) begin = 0;
        if (end < begin) end = begin;
        if (end > len_E) end = len_E;
        int degree = end - begin;
        if (degree > max_degree) max_degree = degree;
        if (degree > 0) ++nonzero;
    }
    stats.max_degree = max_degree;
    stats.nonzero_degree_vertices = nonzero;
    stats.avg_degree = len_V > 0 ? (double)len_E / (double)len_V : 0.0;
    stats.ready = true;
    return stats;
}

bool should_use_weighted_frontier_sssp(
    const HostCsrStats& stats,
    int len_V,
    int len_E,
    int len_sources
) {
    if (env_present("EASYGRAPH_GPU_SSSP_FRONTIER")) {
        return env_bool("EASYGRAPH_GPU_SSSP_FRONTIER", true);
    }
    // A single-source request gives the source-parallel CTA path only one
    // block of work. Use the frontier backend so parallelism comes from the
    // active vertices instead. Multi-source calls retain the established
    // size and topology policy below.
    if (len_sources == 1) return true;

    const int min_edges = env_int("EASYGRAPH_GPU_SSSP_FRONTIER_MIN_EDGES", 300000);
    const int min_vertices = env_int("EASYGRAPH_GPU_SSSP_FRONTIER_MIN_VERTICES", 50000);
    bool large_enough = len_E >= min_edges || len_V >= min_vertices;
    if (!large_enough) return false;
    if (!adaptive_policy_enabled()) return true;

    const double max_avg_degree = env_double("EASYGRAPH_GPU_SSSP_FRONTIER_MAX_AVG_DEGREE", 64.0);
    const int max_degree = env_int("EASYGRAPH_GPU_SSSP_FRONTIER_MAX_DEGREE", 2000000000);
    if (stats.ready) {
        if (stats.avg_degree > max_avg_degree) return false;
        if (stats.max_degree > max_degree) return false;
    }
    return true;
}

} // namespace gpu_easygraph
