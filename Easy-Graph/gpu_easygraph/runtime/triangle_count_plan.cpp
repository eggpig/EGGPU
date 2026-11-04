#include "triangle_count_plan.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace gpu_easygraph { namespace runtime {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}
std::atomic<std::uint64_t> next_identity(1);
}

TriangleHostPlan make_triangle_plan(const int* row, std::size_t row_count,
    const int* col, std::size_t arcs, int parts, const std::string& policy) {
    if (parts < 1 || (policy != "range" && policy != "edge_work"))
        throw std::invalid_argument("TC needs positive parts and range or edge_work ownership");
    if (row_count == 0 || row_count - 1 > static_cast<std::size_t>(INT32_MAX) ||
        arcs > static_cast<std::size_t>(INT32_MAX))
        throw std::invalid_argument("TC input CSR exceeds signed int32 domain");
    const int n = static_cast<int>(row_count - 1);
    auto begin = Clock::now();
    if (row[0] != 0 || row[n] != static_cast<int>(arcs))
        throw std::invalid_argument("TC CSR endpoints disagree with array sizes");
    for (int u = 0; u < n; ++u) {
        if (row[u] < 0 || row[u + 1] < row[u] || row[u + 1] > static_cast<int>(arcs))
            throw std::invalid_argument("TC CSR offsets must be monotonic");
        int previous = -1;
        for (int e = row[u]; e < row[u + 1]; ++e) {
            const int v = col[e];
            if (v < 0 || v >= n || v == u || v <= previous)
                throw std::invalid_argument("TC requires sorted simple CSR without self loops");
            previous = v;
        }
    }
    TriangleHostPlan plan;
    plan.identity = next_identity.fetch_add(1);
    plan.vertex_count = n;
    plan.input_arcs = static_cast<int>(arcs);
    plan.owner_policy = policy;
    plan.validation_seconds = elapsed(begin);
    begin = Clock::now();
    plan.offsets.assign(static_cast<std::size_t>(n) + 1, 0);
    // The canonical input contract supplies both directions of each simple edge.
    // This is the LCC degree/ID forward construction, without per-vertex outputs.
    auto forward = [&](int u, int v) {
        const int du = row[u + 1] - row[u], dv = row[v + 1] - row[v];
        return du < dv || (du == dv && u < v);
    };
    for (int u = 0; u < n; ++u) {
        int count = 0;
        for (int e = row[u]; e < row[u + 1]; ++e) if (forward(u, col[e])) ++count;
        plan.offsets[u + 1] = plan.offsets[u] + count;
    }
    if (static_cast<std::size_t>(plan.offsets[n]) * 2 != arcs)
        throw std::invalid_argument("TC forward edge count disagrees with symmetric simple input");
    plan.indices.resize(plan.offsets[n]);
    plan.edge_sources.resize(plan.offsets[n]);
    for (int u = 0; u < n; ++u) {
        int out = plan.offsets[u];
        for (int e = row[u]; e < row[u + 1]; ++e)
            if (forward(u, col[e])) {
                plan.edge_sources[out] = u;
                plan.indices[out++] = col[e];
            }
    }
    plan.orientation_seconds = elapsed(begin);
    begin = Clock::now();
    std::vector<std::uint64_t> prefix(static_cast<std::size_t>(n) + 1, 0);
    unsigned __int128 upper = 0;
    for (int u = 0; u < n; ++u) {
        const std::uint64_t du = plan.offsets[u + 1] - plan.offsets[u];
        upper += static_cast<unsigned __int128>(du) * (du ? du - 1 : 0) / 2;
        std::uint64_t work = 0;
        for (int e = plan.offsets[u]; e < plan.offsets[u + 1]; ++e) {
            const int v = plan.indices[e];
            const std::uint64_t dv = plan.offsets[v + 1] - plan.offsets[v];
            std::uint64_t width = std::max(du, dv), levels = 0;
            while (width) { ++levels; width >>= 1; }
            // Binary-search comparison work proxy; zero-length intersections do no search.
            work += std::min(du, dv) * levels;
        }
        if (work > UINT64_MAX - prefix[u]) throw std::overflow_error("TC work prefix exceeds uint64");
        prefix[u + 1] = prefix[u] + work;
    }
    if (upper > UINT64_MAX) throw std::overflow_error("TC counting upper bound exceeds uint64");
    plan.triangle_upper_bound = static_cast<std::uint64_t>(upper);
    std::vector<int> cuts(static_cast<std::size_t>(parts) + 1, 0);
    cuts[parts] = n;
    for (int rank = 1; rank < parts; ++rank) {
        if (policy == "range" || prefix.back() == 0) {
            cuts[rank] = static_cast<int>(static_cast<std::int64_t>(n) * rank / parts);
        } else {
            const std::uint64_t target = static_cast<std::uint64_t>(
                static_cast<unsigned __int128>(prefix.back()) * rank / parts);
            cuts[rank] = static_cast<int>(std::lower_bound(prefix.begin(), prefix.end(), target) - prefix.begin());
        }
    }
    plan.owners.resize(parts);
    for (int rank = 0; rank < parts; ++rank) {
        plan.owners[rank].begin = cuts[rank];
        plan.owners[rank].end = cuts[rank + 1];
        plan.owners[rank].estimated_scan_work = prefix[cuts[rank + 1]] - prefix[cuts[rank]];
    }
    plan.owner_partition_seconds = elapsed(begin);
    return plan;
}

std::uint64_t triangle_plan_cpu_count(const TriangleHostPlan& p, const TriangleOwnerRange& owner) {
    std::uint64_t result = 0;
    for (int u = owner.begin; u < owner.end; ++u) {
        for (int e = p.offsets[u]; e < p.offsets[u + 1]; ++e) {
            const int v = p.indices[e];
            int i = p.offsets[u], j = p.offsets[v];
            while (i < p.offsets[u + 1] && j < p.offsets[v + 1]) {
                const int a = p.indices[i], b = p.indices[j];
                if (a == b) { ++result; ++i; ++j; }
                else if (a < b) ++i;
                else ++j;
            }
        }
    }
    return result;
}
} }
