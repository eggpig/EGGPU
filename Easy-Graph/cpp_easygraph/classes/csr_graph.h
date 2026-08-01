#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>

#include "../common/common.h"

struct UndirectedProjection {
    // Each unique undirected edge {u, v}, u < v, is stored once in each
    // incidence view.  Keeping the views separate avoids a single 2|E|-entry
    // CSR while preserving both endpoint traversal directions.
    std::vector<int> lower_V;
    std::vector<int> lower_E;
    std::vector<int> upper_V;
    std::vector<int> upper_E;

    // Undirected degree and the acyclic orientation induced by the
    // lexicographic order (degree, node id).
    std::vector<int> degree;
    std::vector<int> forward_V;
    std::vector<int> forward_E;

    // Optional edge weights aligned with lower_E. Each undirected edge is
    // represented once, so weighted algorithms avoid a second symmetric copy.
    std::vector<double> lower_W;
    std::string weight_key;

    std::int64_t unique_edge_count = 0;
};

struct CSRGraph {
    CSRGraph() : cache_id(next_cache_id()) {}
    CSRGraph(const CSRGraph&) = delete;
    CSRGraph& operator=(const CSRGraph&) = delete;
    CSRGraph(CSRGraph&&) = delete;
    CSRGraph& operator=(CSRGraph&&) = delete;

    static std::uint64_t next_cache_id() {
        static std::atomic<std::uint64_t> next_id(1);
        return next_id.fetch_add(1, std::memory_order_relaxed);
    }

    // Identifies one immutable host CSR representation. Graph mutations drop
    // the owning CSRGraph, so a rebuilt representation receives a new ID even
    // if the allocator later reuses the same host addresses.
    const std::uint64_t cache_id;

    std::vector<int> V;
    std::vector<int> E;
    std::vector<double> unweighted_W;
    std::unordered_map<std::string, std::shared_ptr<std::vector<double>>> W_map;

    std::vector<node_t> nodes;
    std::unordered_map<node_t, int> node2idx;

    // Bulk CSR graphs use public node labels 0..N-1 and intentionally avoid
    // Python/C++ hash maps and an 8-byte unit-weight entry for every edge.
    bool contiguous_zero_based = false;
    bool implicit_unit_weights = false;
    // True only when every CSR row is strictly sorted, duplicate-free, and
    // loop-free. Projection-based edge statistics rely on this contract for
    // logarithmic directed-edge membership checks.
    bool simple_sorted_topology = false;
    std::int64_t node_count = 0;

    // Optional derived topology for algorithms that require the simple
    // undirected projection of a directed bulk graph.  The source CSR remains
    // unchanged and retains independent ownership and cache identity.
    std::shared_ptr<UndirectedProjection> undirected_projection;

    // Lazily built transpose of the immutable topology. Functions that need
    // incoming adjacency share this host view instead of rebuilding COO and
    // reverse CSR for every call.
    std::vector<int> reverse_V;
    std::vector<int> reverse_E;
    bool reverse_ready = false;
    std::mutex reverse_mutex;
};
