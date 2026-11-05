#include "partitioned_wcc_plan.h"
#include <chrono>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace gpu_easygraph {
namespace runtime {

void prepare_wcc_output_gather_index(PartitionedWccPlan& plan) {
    const auto start = std::chrono::steady_clock::now();
    plan.output_source_for_global.resize(static_cast<std::size_t>(plan.vertex_count));
    for (std::size_t rank = 0; rank < plan.partitions.size(); ++rank) {
        const auto& partition = plan.partitions[rank];
        if (rank > UINT32_MAX || partition.owned_vertex_count > UINT32_MAX)
            throw std::overflow_error("WCC output index exceeds packed uint32 owner/local protocol");
        for (std::size_t local = 0; local < partition.owned_vertex_count; ++local)
            plan.output_source_for_global[partition.local_to_global[local]] =
                (static_cast<std::uint64_t>(rank) << 32) | static_cast<std::uint64_t>(local);
    }
    plan.output_index_build_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-start).count();
}
namespace {

std::size_t checked_size(std::uint64_t value, const char* context) {
    if (value > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(context);
    }
    return static_cast<std::size_t>(value);
}

std::size_t find_owned_local(
    const WccHostPartition& partition, std::uint64_t global_id) {
    const std::vector<std::uint64_t>::const_iterator owned_end =
        partition.local_to_global.begin() +
        static_cast<std::ptrdiff_t>(partition.owned_vertex_count);
    const std::vector<std::uint64_t>::const_iterator found = std::lower_bound(
        partition.local_to_global.begin(), owned_end, global_id);
    if (found == owned_end || *found != global_id) {
        throw std::invalid_argument("WCC halo global ID is absent from its owner");
    }
    return static_cast<std::size_t>(
        found - partition.local_to_global.begin());
}

std::uint64_t find_root(
    std::vector<std::uint64_t>* parent, std::uint64_t vertex) {
    std::uint64_t current = vertex;
    while ((*parent)[static_cast<std::size_t>(current)] != current) {
        current = (*parent)[static_cast<std::size_t>(current)];
    }
    while ((*parent)[static_cast<std::size_t>(vertex)] != vertex) {
        const std::uint64_t next = (*parent)[static_cast<std::size_t>(vertex)];
        (*parent)[static_cast<std::size_t>(vertex)] = current;
        vertex = next;
    }
    return current;
}

}  // namespace

std::size_t WccHaloRoute::size() const {
    return source_owned_local_ids.size();
}

const WccHaloRoute& PartitionedWccPlan::route(
    std::size_t source_rank, std::size_t destination_rank) const {
    const std::size_t count = partitions.size();
    if (source_rank >= count || destination_rank >= count) {
        throw std::out_of_range("WCC route rank is out of range");
    }
    return routes[source_rank * count + destination_rank];
}

PartitionedWccPlan build_partitioned_wcc_plan(
    const std::vector<WccHostPartition>& partitions,
    std::uint64_t vertex_count) {
    if (vertex_count == 0 || partitions.empty()) {
        throw std::invalid_argument(
            "partitioned WCC requires a nonempty graph and at least one partition");
    }
    const std::size_t global_count = checked_size(
        vertex_count, "WCC vertex count exceeds host address space");
    std::vector<int> owner(global_count, -1);
    std::vector<int> devices;
    devices.reserve(partitions.size());

    for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
        const WccHostPartition& partition = partitions[rank];
        if ((partition.wide_offsets ? !partition.offsets.empty() : !partition.offsets64.empty()) ||
            partition.partition_id != static_cast<int>(rank) ||
            partition.device_id < 0 ||
            std::find(devices.begin(), devices.end(), partition.device_id) !=
                devices.end()) {
            throw std::invalid_argument(
                "WCC partition IDs must be contiguous and devices distinct");
        }
        devices.push_back(partition.device_id);
        const std::size_t local_count = partition.local_to_global.size();
        if (partition.owned_vertex_count > local_count ||
            partition.owners.size() != local_count ||
            partition.offset_count() != partition.owned_vertex_count + 1U ||
            partition.offset_count() == 0 || partition.offset(0) != 0 ||
            partition.offset(partition.offset_count() - 1) < 0 ||
            static_cast<std::size_t>(partition.offset(partition.offset_count() - 1)) !=
                partition.indices.size()) {
            throw std::invalid_argument("WCC partition cardinalities are inconsistent");
        }
        if (partition.owned_vertex_count == 0 &&
            (local_count != 0 || !partition.indices.empty() ||
             (partition.offset_count() != 1U || partition.offset(0) != 0))) {
            throw std::invalid_argument(
                "empty WCC partition must contain only the zero CSR offset");
        }
        for (std::size_t row = 1; row < partition.offset_count(); ++row) {
            if (partition.offset(row) < partition.offset(row - 1U)) {
                throw std::invalid_argument("WCC offsets must be monotone");
            }
        }
        if (!std::is_sorted(
                partition.local_to_global.begin(),
                partition.local_to_global.begin() +
                    static_cast<std::ptrdiff_t>(partition.owned_vertex_count))) {
            throw std::invalid_argument("WCC owned global IDs must be sorted");
        }
        std::uint64_t previous_halo = 0;
        bool have_halo = false;
        for (std::size_t local = 0; local < local_count; ++local) {
            const std::uint64_t global = partition.local_to_global[local];
            if (global >= vertex_count || partition.owners[local] < 0 ||
                static_cast<std::size_t>(partition.owners[local]) >=
                    partitions.size()) {
                throw std::invalid_argument("WCC local vertex metadata is invalid");
            }
            if (local < partition.owned_vertex_count) {
                if (partition.owners[local] != partition.partition_id ||
                    owner[static_cast<std::size_t>(global)] != -1) {
                    throw std::invalid_argument(
                        "WCC owned vertices are not a disjoint cover");
                }
                owner[static_cast<std::size_t>(global)] = partition.partition_id;
            } else {
                if (partition.owners[local] == partition.partition_id ||
                    (have_halo && global <= previous_halo)) {
                    throw std::invalid_argument(
                        "WCC halo IDs must be remote and strictly increasing");
                }
                previous_halo = global;
                have_halo = true;
            }
        }
        for (std::size_t edge = 0; edge < partition.indices.size(); ++edge) {
            if (partition.indices[edge] < 0 ||
                static_cast<std::size_t>(partition.indices[edge]) >= local_count) {
                throw std::invalid_argument("WCC local edge endpoint is out of range");
            }
        }
    }
    if (std::find(owner.begin(), owner.end(), -1) != owner.end()) {
        throw std::invalid_argument("WCC partitions do not own every global vertex");
    }

    PartitionedWccPlan plan;
    plan.vertex_count = vertex_count;
    plan.partitions = partitions;
    plan.halo_value_count = 0;
    const std::size_t count = partitions.size();
    plan.routes.resize(count * count);
    for (std::size_t source = 0; source < count; ++source) {
        for (std::size_t destination = 0; destination < count; ++destination) {
            WccHaloRoute& route = plan.routes[source * count + destination];
            route.source_rank = static_cast<int>(source);
            route.destination_rank = static_cast<int>(destination);
        }
    }
    for (std::size_t destination = 0; destination < count; ++destination) {
        const WccHostPartition& partition = partitions[destination];
        for (std::size_t local = partition.owned_vertex_count;
             local < partition.local_to_global.size(); ++local) {
            const std::size_t source = static_cast<std::size_t>(
                partition.owners[local]);
            if (owner[static_cast<std::size_t>(
                    partition.local_to_global[local])] !=
                static_cast<int>(source)) {
                throw std::invalid_argument(
                    "WCC halo owner disagrees with global ownership");
            }
            WccHaloRoute& route = plan.routes[source * count + destination];
            const std::size_t owner_local = find_owned_local(
                partitions[source], partition.local_to_global[local]);
            route.source_owned_local_ids.push_back(
                static_cast<int>(owner_local));
            route.destination_halo_local_ids.push_back(
                static_cast<int>(local));
            ++plan.halo_value_count;
        }
    }
    return plan;
}

std::vector<std::uint64_t> partitioned_wcc_reference(
    const PartitionedWccPlan& plan) {
    // Revalidate rather than trusting a caller-constructed plan.
    const PartitionedWccPlan checked = build_partitioned_wcc_plan(
        plan.partitions, plan.vertex_count);
    const std::size_t count = checked_size(
        checked.vertex_count, "WCC result exceeds host address space");
    std::vector<std::uint64_t> parent(count);
    for (std::size_t vertex = 0; vertex < count; ++vertex) {
        parent[vertex] = static_cast<std::uint64_t>(vertex);
    }
    for (std::size_t rank = 0; rank < checked.partitions.size(); ++rank) {
        const WccHostPartition& partition = checked.partitions[rank];
        for (std::size_t row = 0; row < partition.owned_vertex_count; ++row) {
            const std::uint64_t source = partition.local_to_global[row];
            for (std::int64_t edge = partition.offset(row);
                 edge < partition.offset(row + 1U); ++edge) {
                const std::uint64_t destination = partition.local_to_global[
                    static_cast<std::size_t>(partition.indices[
                        static_cast<std::size_t>(edge)])];
                std::uint64_t source_root = find_root(&parent, source);
                std::uint64_t destination_root = find_root(&parent, destination);
                if (source_root != destination_root) {
                    const std::uint64_t low = std::min(
                        source_root, destination_root);
                    const std::uint64_t high = std::max(
                        source_root, destination_root);
                    parent[static_cast<std::size_t>(high)] = low;
                }
            }
        }
    }
    for (std::size_t vertex = 0; vertex < count; ++vertex) {
        parent[vertex] = find_root(&parent, static_cast<std::uint64_t>(vertex));
    }
    return parent;
}

}  // namespace runtime
}  // namespace gpu_easygraph
