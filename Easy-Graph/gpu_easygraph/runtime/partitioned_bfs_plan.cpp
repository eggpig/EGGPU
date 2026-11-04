#include "partitioned_bfs_plan.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace gpu_easygraph {
namespace runtime {
namespace {

std::size_t checked_size(std::uint64_t value, const char* context) {
    if (value > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(context);
    }
    return static_cast<std::size_t>(value);
}

std::size_t find_owned_local(
    const BfsHostPartition& partition, std::uint64_t global_id) {
    const std::vector<std::uint64_t>::const_iterator owned_end =
        partition.local_to_global.begin() +
        static_cast<std::ptrdiff_t>(partition.owned_vertex_count);
    const std::vector<std::uint64_t>::const_iterator found = std::lower_bound(
        partition.local_to_global.begin(), owned_end, global_id);
    if (found == owned_end || *found != global_id) {
        throw std::invalid_argument(
            "BFS halo global ID is absent from its declared owner");
    }
    return static_cast<std::size_t>(
        found - partition.local_to_global.begin());
}

}  // namespace

BfsFrontierMetrics::BfsFrontierMetrics()
    : active_vertex_fraction(0.0),
      estimated_active_edge_fraction(0.0),
      boundary_edge_fraction(0.0) {}

BfsFrontierThresholds::BfsFrontierThresholds()
    : dense_vertex_enter(0.08),
      dense_vertex_exit(0.04),
      dense_edge_enter(0.10),
      dense_edge_exit(0.05),
      dense_boundary_enter(0.05) {}

BfsFrontierPolicy parse_bfs_frontier_policy(const std::string& value) {
    if (value == "sparse") return BfsFrontierPolicy::Sparse;
    if (value == "dense") return BfsFrontierPolicy::Dense;
    if (value == "adaptive" || value == "auto") {
        return BfsFrontierPolicy::Adaptive;
    }
    throw std::invalid_argument(
        "partitioned BFS frontier_policy must be 'sparse', 'dense', or "
        "'adaptive'");
}

const char* bfs_frontier_policy_name(BfsFrontierPolicy policy) {
    switch (policy) {
        case BfsFrontierPolicy::Sparse:
            return "sparse";
        case BfsFrontierPolicy::Dense:
            return "dense";
        case BfsFrontierPolicy::Adaptive:
            return "adaptive";
    }
    return "unknown";
}

bool bfs_dense_frontier_available() { return true; }

const char* bfs_frontier_kind_name(BfsFrontierKind kind) {
    switch (kind) {
        case BfsFrontierKind::Sparse:
            return "sparse";
        case BfsFrontierKind::Dense:
            return "dense";
    }
    return "unknown";
}

void validate_bfs_frontier_thresholds(
    const BfsFrontierThresholds& thresholds) {
    const double values[] = {
        thresholds.dense_vertex_enter,
        thresholds.dense_vertex_exit,
        thresholds.dense_edge_enter,
        thresholds.dense_edge_exit,
        thresholds.dense_boundary_enter};
    for (std::size_t index = 0; index < sizeof(values) / sizeof(values[0]);
         ++index) {
        if (!std::isfinite(values[index]) || values[index] < 0.0 ||
            values[index] > 1.0) {
            throw std::invalid_argument(
                "BFS frontier thresholds must be finite values in [0, 1]");
        }
    }
    if (thresholds.dense_vertex_exit > thresholds.dense_vertex_enter ||
        thresholds.dense_edge_exit > thresholds.dense_edge_enter) {
        throw std::invalid_argument(
            "BFS dense exit thresholds cannot exceed enter thresholds");
    }
}

BfsFrontierKind select_bfs_frontier_kind(
    BfsFrontierPolicy policy,
    const BfsFrontierMetrics& metrics,
    const BfsFrontierThresholds& thresholds,
    BfsFrontierKind previous) {
    validate_bfs_frontier_thresholds(thresholds);
    const double values[] = {
        metrics.active_vertex_fraction,
        metrics.estimated_active_edge_fraction,
        metrics.boundary_edge_fraction};
    for (std::size_t index = 0; index < sizeof(values) / sizeof(values[0]);
         ++index) {
        if (!std::isfinite(values[index]) || values[index] < 0.0 ||
            values[index] > 1.0) {
            throw std::invalid_argument(
                "BFS frontier metrics must be finite values in [0, 1]");
        }
    }
    if (policy == BfsFrontierPolicy::Sparse) return BfsFrontierKind::Sparse;
    if (policy == BfsFrontierPolicy::Dense) return BfsFrontierKind::Dense;

    const bool boundary_pressure =
        metrics.boundary_edge_fraction >= thresholds.dense_boundary_enter;
    if (previous == BfsFrontierKind::Dense) {
        const bool remain_dense =
            metrics.active_vertex_fraction >= thresholds.dense_vertex_exit ||
            metrics.estimated_active_edge_fraction >=
                thresholds.dense_edge_exit ||
            (boundary_pressure &&
             metrics.estimated_active_edge_fraction >=
                 thresholds.dense_edge_exit * 0.5);
        return remain_dense ? BfsFrontierKind::Dense
                            : BfsFrontierKind::Sparse;
    }
    const bool enter_dense =
        metrics.active_vertex_fraction >= thresholds.dense_vertex_enter ||
        metrics.estimated_active_edge_fraction >=
            thresholds.dense_edge_enter ||
        (boundary_pressure &&
         metrics.estimated_active_edge_fraction >=
             thresholds.dense_edge_enter * 0.5);
    return enter_dense ? BfsFrontierKind::Dense
                       : BfsFrontierKind::Sparse;
}

std::size_t bfs_bitmap_word_count(std::size_t vertex_count) {
    return vertex_count / 32U + (vertex_count % 32U == 0U ? 0U : 1U);
}

std::vector<std::uint32_t> pack_bfs_owner_bitmap_reference(
    const std::vector<int>& owner_local_ids,
    std::size_t owned_vertex_count) {
    std::vector<std::uint32_t> result(
        bfs_bitmap_word_count(owned_vertex_count), 0U);
    for (std::size_t index = 0; index < owner_local_ids.size(); ++index) {
        const int vertex = owner_local_ids[index];
        if (vertex < 0 ||
            static_cast<std::size_t>(vertex) >= owned_vertex_count) {
            throw std::out_of_range(
                "BFS owner bitmap contains an out-of-range local ID");
        }
        const std::size_t local = static_cast<std::size_t>(vertex);
        result[local / 32U] |=
            static_cast<std::uint32_t>(1U) << (local & 31U);
    }
    return result;
}

std::vector<int> unpack_bfs_owner_bitmap_reference(
    const std::vector<std::uint32_t>& bitmap,
    std::size_t owned_vertex_count) {
    if (bitmap.size() != bfs_bitmap_word_count(owned_vertex_count)) {
        throw std::invalid_argument(
            "BFS owner bitmap word count does not match its owner domain");
    }
    std::vector<int> result;
    for (std::size_t vertex = 0; vertex < owned_vertex_count; ++vertex) {
        if ((bitmap[vertex / 32U] &
             (static_cast<std::uint32_t>(1U) << (vertex & 31U))) != 0U) {
            result.push_back(static_cast<int>(vertex));
        }
    }
    return result;
}

std::size_t BfsOwnerRoute::capacity() const {
    return source_halo_local_ids.size();
}

std::size_t BfsDenseRouteLayout::outgoing_word_count() const {
    return outgoing_word_offsets.empty() ? 0U : outgoing_word_offsets.back();
}

std::size_t BfsDenseRouteLayout::incoming_word_count() const {
    return incoming_word_offsets.empty() ? 0U : incoming_word_offsets.back();
}

BfsDenseRouteLayout build_bfs_dense_route_layout(
    const PartitionedBfsPlan& plan, std::size_t rank) {
    if (rank >= plan.partitions.size()) {
        throw std::out_of_range("BFS dense route rank is out of range");
    }
    BfsDenseRouteLayout result;
    result.outgoing_word_offsets.push_back(0U);
    result.incoming_word_offsets.push_back(0U);
    const std::size_t own_words = bfs_bitmap_word_count(
        plan.partitions[rank].owned_vertex_count);
    for (std::size_t peer = 0; peer < plan.partitions.size(); ++peer) {
        result.outgoing_word_offsets.push_back(
            result.outgoing_word_offsets.back() +
            (peer == rank ? 0U : bfs_bitmap_word_count(
                plan.partitions[peer].owned_vertex_count)));
        result.incoming_word_offsets.push_back(
            result.incoming_word_offsets.back() +
            (peer == rank ? 0U : own_words));
    }
    return result;
}

const BfsOwnerRoute& PartitionedBfsPlan::route(
    std::size_t source_rank, std::size_t destination_rank) const {
    const std::size_t count = partitions.size();
    if (source_rank >= count || destination_rank >= count) {
        throw std::out_of_range("BFS route rank is out of range");
    }
    return routes[source_rank * count + destination_rank];
}

std::size_t PartitionedBfsPlan::owner_rank(
    std::uint64_t global_vertex) const {
    if (global_vertex >= owner_by_global.size()) {
        throw std::out_of_range("BFS source vertex is outside the graph");
    }
    return static_cast<std::size_t>(
        owner_by_global[static_cast<std::size_t>(global_vertex)]);
}

std::size_t PartitionedBfsPlan::owner_local(
    std::uint64_t global_vertex) const {
    const std::size_t rank = owner_rank(global_vertex);
    return find_owned_local(partitions[rank], global_vertex);
}

PartitionedBfsPlan build_partitioned_bfs_plan(
    const std::vector<BfsHostPartition>& partitions,
    std::uint64_t vertex_count) {
    if (vertex_count == 0 || partitions.empty()) {
        throw std::invalid_argument(
            "partitioned BFS requires a nonempty graph and partitions");
    }
    const std::size_t global_count = checked_size(
        vertex_count, "BFS vertex count exceeds host address space");
    std::vector<int> owner(global_count, -1);
    std::vector<int> devices;
    devices.reserve(partitions.size());

    for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
        const BfsHostPartition& partition = partitions[rank];
        if (partition.partition_id != static_cast<int>(rank) ||
            partition.device_id < 0 ||
            std::find(devices.begin(), devices.end(), partition.device_id) !=
                devices.end()) {
            throw std::invalid_argument(
                "BFS partition IDs must be contiguous and devices distinct");
        }
        devices.push_back(partition.device_id);
        const std::size_t local_count = partition.local_to_global.size();
        if (partition.owned_vertex_count >
                static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            partition.owned_vertex_count > local_count ||
            partition.owners.size() != local_count ||
            partition.offsets.size() != partition.owned_vertex_count + 1U ||
            partition.offsets.empty() || partition.offsets.front() != 0 ||
            partition.offsets.back() < 0 ||
            static_cast<std::size_t>(partition.offsets.back()) !=
                partition.indices.size()) {
            throw std::invalid_argument(
                "BFS partition cardinalities are inconsistent");
        }
        if (partition.owned_vertex_count == 0 &&
            (local_count != 0 || !partition.indices.empty() ||
             partition.offsets != std::vector<int>(1U, 0))) {
            throw std::invalid_argument(
                "empty BFS partition must contain only the zero CSR offset");
        }
        for (std::size_t row = 1; row < partition.offsets.size(); ++row) {
            if (partition.offsets[row] < partition.offsets[row - 1U]) {
                throw std::invalid_argument("BFS offsets must be monotone");
            }
        }
        if (!std::is_sorted(
                partition.local_to_global.begin(),
                partition.local_to_global.begin() +
                    static_cast<std::ptrdiff_t>(
                        partition.owned_vertex_count))) {
            throw std::invalid_argument(
                "BFS owned global IDs must be sorted");
        }
        std::uint64_t previous_halo = 0;
        bool have_halo = false;
        for (std::size_t local = 0; local < local_count; ++local) {
            const std::uint64_t global = partition.local_to_global[local];
            const int declared_owner = partition.owners[local];
            if (global >= vertex_count || declared_owner < 0 ||
                static_cast<std::size_t>(declared_owner) >=
                    partitions.size()) {
                throw std::invalid_argument(
                    "BFS local vertex metadata is invalid");
            }
            if (local < partition.owned_vertex_count) {
                if (declared_owner != partition.partition_id ||
                    owner[static_cast<std::size_t>(global)] != -1) {
                    throw std::invalid_argument(
                        "BFS owned vertices are not a disjoint cover");
                }
                owner[static_cast<std::size_t>(global)] =
                    partition.partition_id;
            } else {
                if (declared_owner == partition.partition_id ||
                    (have_halo && global <= previous_halo)) {
                    throw std::invalid_argument(
                        "BFS halo IDs must be remote and strictly increasing");
                }
                previous_halo = global;
                have_halo = true;
            }
        }
        for (std::size_t edge = 0; edge < partition.indices.size(); ++edge) {
            const int target = partition.indices[edge];
            if (target < 0 || static_cast<std::size_t>(target) >= local_count) {
                throw std::invalid_argument(
                    "BFS local edge endpoint is out of range");
            }
            const std::size_t local = static_cast<std::size_t>(target);
            if (partition.owners[local] == partition.partition_id &&
                local >= partition.owned_vertex_count) {
                throw std::invalid_argument(
                    "BFS owner-local edge endpoint is outside the owned prefix");
            }
        }
    }
    if (std::find(owner.begin(), owner.end(), -1) != owner.end()) {
        throw std::invalid_argument(
            "BFS partitions do not own every global vertex");
    }

    PartitionedBfsPlan plan;
    plan.vertex_count = vertex_count;
    plan.partitions = partitions;
    plan.owner_by_global = owner;
    plan.route_capacity = 0;
    plan.boundary_adjacency_count = 0;
    const std::size_t count = partitions.size();
    plan.routes.resize(count * count);
    for (std::size_t source = 0; source < count; ++source) {
        for (std::size_t destination = 0; destination < count;
             ++destination) {
            BfsOwnerRoute& route =
                plan.routes[source * count + destination];
            route.source_rank = static_cast<int>(source);
            route.destination_rank = static_cast<int>(destination);
        }
    }
    for (std::size_t source = 0; source < count; ++source) {
        const BfsHostPartition& partition = partitions[source];
        for (std::size_t edge = 0; edge < partition.indices.size(); ++edge) {
            const std::size_t target = static_cast<std::size_t>(
                partition.indices[edge]);
            if (partition.owners[target] != partition.partition_id) {
                ++plan.boundary_adjacency_count;
            }
        }
        for (std::size_t local = partition.owned_vertex_count;
             local < partition.local_to_global.size(); ++local) {
            const std::uint64_t global = partition.local_to_global[local];
            const std::size_t destination = static_cast<std::size_t>(
                partition.owners[local]);
            if (owner[static_cast<std::size_t>(global)] !=
                static_cast<int>(destination)) {
                throw std::invalid_argument(
                    "BFS halo owner disagrees with global ownership");
            }
            BfsOwnerRoute& route =
                plan.routes[source * count + destination];
            route.source_halo_local_ids.push_back(static_cast<int>(local));
            route.destination_owner_local_ids.push_back(static_cast<int>(
                find_owned_local(partitions[destination], global)));
            ++plan.route_capacity;
        }
    }
    return plan;
}

std::vector<std::int64_t> partitioned_bfs_reference(
    const PartitionedBfsPlan& plan,
    std::uint64_t source) {
    const PartitionedBfsPlan checked = build_partitioned_bfs_plan(
        plan.partitions, plan.vertex_count);
    checked.owner_rank(source);
    const std::size_t count = checked_size(
        checked.vertex_count, "BFS result exceeds host address space");
    std::vector<std::vector<std::uint64_t> > adjacency(count);
    for (std::size_t rank = 0; rank < checked.partitions.size(); ++rank) {
        const BfsHostPartition& partition = checked.partitions[rank];
        for (std::size_t row = 0; row < partition.owned_vertex_count; ++row) {
            const std::uint64_t global_source =
                partition.local_to_global[row];
            for (int edge = partition.offsets[row];
                 edge < partition.offsets[row + 1U]; ++edge) {
                adjacency[static_cast<std::size_t>(global_source)].push_back(
                    partition.local_to_global[static_cast<std::size_t>(
                        partition.indices[static_cast<std::size_t>(edge)])]);
            }
        }
    }

    std::vector<std::int64_t> distances(count, -1);
    std::queue<std::uint64_t> frontier;
    distances[static_cast<std::size_t>(source)] = 0;
    frontier.push(source);
    while (!frontier.empty()) {
        const std::uint64_t vertex = frontier.front();
        frontier.pop();
        const std::int64_t candidate =
            distances[static_cast<std::size_t>(vertex)] + 1;
        const std::vector<std::uint64_t>& row =
            adjacency[static_cast<std::size_t>(vertex)];
        for (std::size_t edge = 0; edge < row.size(); ++edge) {
            const std::size_t target = static_cast<std::size_t>(row[edge]);
            if (distances[target] == -1) {
                distances[target] = candidate;
                frontier.push(row[edge]);
            }
        }
    }
    return distances;
}

std::vector<std::int64_t> merge_partitioned_bfs_owner_distances(
    const PartitionedBfsPlan& plan,
    const std::vector<std::vector<std::int64_t> >& owned_distances,
    std::int64_t unvisited_value) {
    if (owned_distances.size() != plan.partitions.size()) {
        throw std::invalid_argument(
            "BFS owner result rank count does not match the partition plan");
    }
    const std::size_t count = checked_size(
        plan.vertex_count, "BFS result exceeds host address space");
    std::vector<std::int64_t> result(count, -1);
    for (std::size_t rank = 0; rank < plan.partitions.size(); ++rank) {
        const BfsHostPartition& partition = plan.partitions[rank];
        if (owned_distances[rank].size() != partition.owned_vertex_count) {
            throw std::invalid_argument(
                "BFS owner result length does not match its partition");
        }
        for (std::size_t local = 0; local < partition.owned_vertex_count;
             ++local) {
            const std::uint64_t global = partition.local_to_global[local];
            if (global >= plan.vertex_count) {
                throw std::invalid_argument(
                    "BFS owner result references a vertex outside the graph");
            }
            const std::int64_t distance = owned_distances[rank][local];
            result[static_cast<std::size_t>(global)] =
                distance == unvisited_value ? -1 : distance;
        }
    }
    return result;
}

}  // namespace runtime
}  // namespace gpu_easygraph
