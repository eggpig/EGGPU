#include "partitioned_pagerank_reference.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gpu_easygraph {
namespace runtime {
namespace {

double outgoing_weight_at(
    const PageRankOutgoingPartition& partition, std::size_t edge) {
    return partition.outgoing_weights.empty()
        ? 1.0
        : partition.outgoing_weights[edge];
}

}  // namespace

PartitionedPageRankOptions::PartitionedPageRankOptions()
    : alpha(0.85), tolerance(1e-10), max_iterations(200) {}

std::vector<PageRankHostPartition> build_destination_owned_page_rank(
    const std::vector<PageRankOutgoingPartition>& outgoing,
    std::uint64_t vertex_count, bool materialize_unit_weights) {
    if (vertex_count == 0) {
        if (!outgoing.empty()) {
            throw std::invalid_argument(
                "empty PageRank graph cannot contain outgoing partitions");
        }
        return std::vector<PageRankHostPartition>();
    }
    if (outgoing.empty()) {
        throw std::invalid_argument(
            "PageRank pull preparation requires outgoing partitions");
    }

    const bool store_weights = materialize_unit_weights || std::any_of(
        outgoing.begin(), outgoing.end(), [](const PageRankOutgoingPartition& p) {
            return !p.outgoing_weights.empty();
        });
    std::vector<PageRankHostPartition> result(outgoing.size());
    std::uint64_t owned_total = 0;
    // Global IDs already occupy [0, vertex_count). Build the inverse ownership
    // once; every edge can then address its destination row in constant time.
    std::vector<int> global_owner(static_cast<std::size_t>(vertex_count), -1);
    std::vector<std::size_t> global_owned_local(static_cast<std::size_t>(vertex_count));
    std::vector<int> seen_devices;
    for (std::size_t rank = 0; rank < outgoing.size(); ++rank) {
        const PageRankOutgoingPartition& source = outgoing[rank];
        if (source.partition_id != static_cast<int>(rank) ||
            source.device_id < 0 ||
            std::find(seen_devices.begin(), seen_devices.end(), source.device_id) != seen_devices.end() ||
            source.owned_vertex_count > source.local_to_global.size() ||
            source.owners.size() != source.local_to_global.size() ||
            source.outgoing_offsets.size() != source.owned_vertex_count + 1U ||
            source.outgoing_offsets.empty() ||
            source.outgoing_offsets.front() != 0 ||
            source.outgoing_offsets.back() < 0 ||
            static_cast<std::size_t>(source.outgoing_offsets.back()) !=
                source.outgoing_destinations.size() ||
            (!source.outgoing_weights.empty() &&
             source.outgoing_weights.size() !=
                 source.outgoing_destinations.size()) ||
            (source.owned_vertex_count == 0 &&
             (!source.local_to_global.empty() || !source.owners.empty() ||
              !source.outgoing_destinations.empty() ||
              !source.outgoing_weights.empty()))) {
            throw std::invalid_argument(
                "outgoing PageRank partition metadata is inconsistent");
        }
        seen_devices.push_back(source.device_id);
        for (std::size_t row = 1; row < source.outgoing_offsets.size(); ++row) {
            if (source.outgoing_offsets[row] < source.outgoing_offsets[row - 1]) {
                throw std::invalid_argument(
                    "outgoing PageRank offsets must be monotone");
            }
        }
        if (!std::is_sorted(
                source.local_to_global.begin(),
                source.local_to_global.begin() +
                    static_cast<std::ptrdiff_t>(source.owned_vertex_count))) {
            throw std::invalid_argument(
                "outgoing PageRank owned IDs must be ordered");
        }
        for (std::size_t local = 0; local < source.local_to_global.size(); ++local) {
            if (source.local_to_global[local] >= vertex_count ||
                source.owners[local] < 0 ||
                static_cast<std::size_t>(source.owners[local]) >= outgoing.size() ||
                (local < source.owned_vertex_count &&
                 source.owners[local] != source.partition_id)) {
                throw std::invalid_argument(
                    "outgoing PageRank local vertex metadata is invalid");
            }
        }
        for (std::size_t local = 0; local < source.owned_vertex_count; ++local) {
            const std::size_t global = static_cast<std::size_t>(source.local_to_global[local]);
            if (global_owner[global] != -1) {
                throw std::invalid_argument("outgoing PageRank ownership must be disjoint");
            }
            global_owner[global] = static_cast<int>(rank);
            global_owned_local[global] = local;
        }
        owned_total += static_cast<std::uint64_t>(source.owned_vertex_count);

        PageRankHostPartition& destination = result[rank];
        destination.partition_id = source.partition_id;
        destination.device_id = source.device_id;
        destination.owned_vertex_count = source.owned_vertex_count;
        destination.local_to_global.assign(
            source.local_to_global.begin(),
            source.local_to_global.begin() +
                static_cast<std::ptrdiff_t>(source.owned_vertex_count));
        destination.owners.assign(source.owned_vertex_count, source.partition_id);
        destination.owned_out_weight.assign(source.owned_vertex_count, 0.0);
        destination.incoming_offsets.assign(source.owned_vertex_count + 1U, 0);
    }
    if (owned_total != vertex_count) {
        throw std::invalid_argument(
            "outgoing PageRank partitions do not cover all vertices");
    }

    for (const PageRankOutgoingPartition& source : outgoing) {
        for (std::size_t local = 0; local < source.local_to_global.size(); ++local) {
            if (source.owners[local] != global_owner[static_cast<std::size_t>(source.local_to_global[local])]) {
                throw std::invalid_argument("outgoing PageRank halo owner is inconsistent");
            }
        }
    }

    std::vector<std::vector<std::uint64_t> > remote_sources(outgoing.size());
    // First pass computes outgoing denominators, incoming row sizes, and the
    // exact remote-source halo needed by each destination partition.
    for (std::size_t source_rank = 0; source_rank < outgoing.size(); ++source_rank) {
        const PageRankOutgoingPartition& source = outgoing[source_rank];
        std::vector<std::size_t> last_remote_source(outgoing.size(), std::numeric_limits<std::size_t>::max());
        for (std::size_t source_local = 0;
             source_local < source.owned_vertex_count; ++source_local) {
            const std::uint64_t source_global = source.local_to_global[source_local];
            for (int edge_value = source.outgoing_offsets[source_local];
                 edge_value < source.outgoing_offsets[source_local + 1U];
                 ++edge_value) {
                const std::size_t edge = static_cast<std::size_t>(edge_value);
                const int target_local_value = source.outgoing_destinations[edge];
                const double weight = outgoing_weight_at(source, edge);
                if (target_local_value < 0 ||
                    static_cast<std::size_t>(target_local_value) >=
                        source.local_to_global.size() ||
                    !std::isfinite(weight) || weight < 0.0) {
                    throw std::invalid_argument(
                        "outgoing PageRank edge has invalid endpoint or weight");
                }
                const std::size_t target_local =
                    static_cast<std::size_t>(target_local_value);
                const int target_owner = source.owners[target_local];
                const std::uint64_t target_global =
                    source.local_to_global[target_local];
                const std::size_t target_row = global_owned_local[static_cast<std::size_t>(target_global)];
                int& row_edges = result[static_cast<std::size_t>(target_owner)].incoming_offsets[target_row + 1U];
                if (row_edges == std::numeric_limits<int>::max()) {
                    throw std::overflow_error("PageRank incoming row exceeds int32 capacity");
                }
                ++row_edges;
                result[source_rank].owned_out_weight[source_local] += weight;
                if (target_owner != static_cast<int>(source_rank) &&
                    last_remote_source[static_cast<std::size_t>(target_owner)] != source_local) {
                    remote_sources[static_cast<std::size_t>(target_owner)].push_back(source_global);
                    last_remote_source[static_cast<std::size_t>(target_owner)] = source_local;
                }
            }
            if (!std::isfinite(result[source_rank].owned_out_weight[source_local])) {
                throw std::invalid_argument("PageRank outgoing weight sum must be finite");
            }
        }
    }

    std::vector<std::vector<int> > cursors(outgoing.size());
    for (std::size_t rank = 0; rank < result.size(); ++rank) {
        PageRankHostPartition& partition = result[rank];
        for (std::size_t row = 0; row < partition.owned_vertex_count; ++row) {
            if (partition.incoming_offsets[row + 1U] >
                std::numeric_limits<int>::max() - partition.incoming_offsets[row]) {
                throw std::overflow_error("PageRank incoming partition exceeds int32 capacity");
            }
            partition.incoming_offsets[row + 1U] +=
                partition.incoming_offsets[row];
        }
        std::vector<std::uint64_t>& halo = remote_sources[rank];
        std::sort(halo.begin(), halo.end());
        // One insertion per (source, remote owner) already guarantees uniqueness.
        partition.local_to_global.insert(
            partition.local_to_global.end(), halo.begin(), halo.end());
        if (partition.local_to_global.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("PageRank local vertex domain exceeds int32 capacity");
        }
        for (std::size_t index = 0; index < halo.size(); ++index) {
            partition.owners.push_back(global_owner[static_cast<std::size_t>(halo[index])]);
        }
        partition.incoming_sources.resize(
            static_cast<std::size_t>(partition.incoming_offsets.back()));
        if (store_weights)
            partition.incoming_weights.resize(partition.incoming_sources.size());
        cursors[rank] = partition.incoming_offsets;
    }

    // Second pass materializes incoming source local IDs and edge weights.
    for (std::size_t source_rank = 0; source_rank < outgoing.size(); ++source_rank) {
        const PageRankOutgoingPartition& source = outgoing[source_rank];
        std::vector<std::size_t> source_in_owner(outgoing.size());
        for (std::size_t source_local = 0;
             source_local < source.owned_vertex_count; ++source_local) {
            const std::uint64_t source_global = source.local_to_global[source_local];
            std::fill(source_in_owner.begin(), source_in_owner.end(), std::numeric_limits<std::size_t>::max());
            source_in_owner[source_rank] = source_local;
            for (int edge_value = source.outgoing_offsets[source_local];
                 edge_value < source.outgoing_offsets[source_local + 1U];
                 ++edge_value) {
                const std::size_t edge = static_cast<std::size_t>(edge_value);
                const std::size_t target_local = static_cast<std::size_t>(
                    source.outgoing_destinations[edge]);
                const int target_owner = source.owners[target_local];
                PageRankHostPartition& target =
                    result[static_cast<std::size_t>(target_owner)];
                const std::uint64_t target_global =
                    source.local_to_global[target_local];
                const std::size_t target_row = global_owned_local[static_cast<std::size_t>(target_global)];
                std::size_t& source_in_target = source_in_owner[static_cast<std::size_t>(target_owner)];
                if (source_in_target == std::numeric_limits<std::size_t>::max()) {
                    const std::vector<std::uint64_t>::const_iterator halo_begin =
                        target.local_to_global.begin() +
                        static_cast<std::ptrdiff_t>(target.owned_vertex_count);
                    const std::vector<std::uint64_t>::const_iterator found =
                        std::lower_bound(
                            halo_begin, target.local_to_global.cend(), source_global);
                    if (found == target.local_to_global.cend() ||
                        *found != source_global) {
                        throw std::logic_error(
                            "PageRank halo lookup failed during pull construction");
                    }
                    source_in_target = static_cast<std::size_t>(
                        found - target.local_to_global.begin());
                }
                const std::size_t position = static_cast<std::size_t>(
                    cursors[static_cast<std::size_t>(target_owner)][target_row]++);
                target.incoming_sources[position] =
                    static_cast<int>(source_in_target);
                if (store_weights)
                    target.incoming_weights[position] = outgoing_weight_at(source, edge);
            }
        }
    }
    // Input ownership/endpoints/weights were checked once above. Incoming CSR
    // and halo metadata are constructed from those same maps; rescanning every
    // generated edge here would repeat that work.
    return result;
}

std::vector<PageRankHostPartition> make_whole_contiguous_page_rank_metadata(
    const std::vector<PageRankOutgoingPartition>& outgoing, std::uint64_t vertex_count) {
    if (vertex_count == 0) {
        if (!outgoing.empty())
            throw std::invalid_argument("empty PageRank graph cannot contain outgoing partitions");
        return {};
    }
    if (outgoing.size() != 1 ||
        vertex_count > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("whole contiguous transpose requires one int32 vertex domain");
    const auto& source = outgoing.front();
    const std::size_t n = static_cast<std::size_t>(vertex_count);
    const std::size_t m = source.outgoing_destinations.size();
    if (source.partition_id != 0 || source.device_id < 0 ||
        source.owned_vertex_count != n || source.local_to_global.size() != n ||
        source.owners.size() != n || source.outgoing_offsets.size() != n + 1 ||
        source.outgoing_offsets.front() != 0 || source.outgoing_offsets.back() < 0 ||
        static_cast<std::size_t>(source.outgoing_offsets.back()) != m ||
        (!source.outgoing_weights.empty() && source.outgoing_weights.size() != m))
        throw std::invalid_argument("whole contiguous outgoing metadata is inconsistent");
    for (std::size_t row = 0; row < n; ++row) {
        if (source.local_to_global[row] != row || source.owners[row] != 0 ||
            source.outgoing_offsets[row + 1] < source.outgoing_offsets[row])
            throw std::invalid_argument("whole contiguous transpose requires identity IDs and monotone CSR");
    }
    std::vector<PageRankHostPartition> result(1);
    auto& target = result.front();
    target.partition_id = 0;
    target.device_id = source.device_id;
    target.owned_vertex_count = n;
    target.local_to_global = source.local_to_global;
    target.owners = source.owners;
    target.owned_out_weight.resize(n);
    target.incoming_offsets.assign(n + 1, 0);
    return result;
}

std::vector<PageRankHostPartition> build_whole_contiguous_page_rank(
    const std::vector<PageRankOutgoingPartition>& outgoing,
    std::uint64_t vertex_count, bool materialize_unit_weights,
    WholeContiguousTransposeTimings* timings) {
    using Clock = std::chrono::steady_clock;
    const auto begin = timings ? Clock::now() : Clock::time_point{};
    auto result = make_whole_contiguous_page_rank_metadata(outgoing, vertex_count);
    if (result.empty()) return result;
    const auto& source = outgoing.front();
    auto& target = result.front();
    const std::size_t n = static_cast<std::size_t>(vertex_count);
    const std::size_t m = source.outgoing_destinations.size();
    const auto metadata_ready = timings ? Clock::now() : Clock::time_point{};
    const bool weighted = !source.outgoing_weights.empty();
    if (weighted) {
        for (std::size_t row = 0; row < n; ++row) {
            double sum = 0.0;
            for (int edge = source.outgoing_offsets[row];
                 edge < source.outgoing_offsets[row + 1]; ++edge) {
                const int destination = source.outgoing_destinations[edge];
                const double weight = source.outgoing_weights[edge];
                if (destination < 0 || static_cast<std::size_t>(destination) >= n ||
                    !std::isfinite(weight) || weight < 0.0)
                    throw std::invalid_argument("whole contiguous edge has invalid endpoint or weight");
                ++target.incoming_offsets[static_cast<std::size_t>(destination) + 1];
                sum += weight;
            }
            if (!std::isfinite(sum))
                throw std::invalid_argument("PageRank outgoing weight sum must be finite");
            target.owned_out_weight[row] = sum;
        }
    } else {
        for (std::size_t row = 0; row < n; ++row)
            target.owned_out_weight[row] =
                source.outgoing_offsets[row + 1] - source.outgoing_offsets[row];
        for (const int destination : source.outgoing_destinations) {
            if (destination < 0 || static_cast<std::size_t>(destination) >= n)
                throw std::invalid_argument("whole contiguous edge has invalid endpoint");
            ++target.incoming_offsets[static_cast<std::size_t>(destination) + 1];
        }
    }
    const auto count_ready = timings ? Clock::now() : Clock::time_point{};
    // m <= INT_MAX is established by the signed CSR terminal offset.
    // Every partial degree sum is <= m, so these int32 prefix sums cannot overflow.
    for (std::size_t row = 0; row < n; ++row)
        target.incoming_offsets[row + 1] += target.incoming_offsets[row];
    std::vector<int> cursor = target.incoming_offsets;
    target.incoming_sources.resize(m);
    if (weighted)
        target.incoming_weights.resize(m);
    else if (materialize_unit_weights)
        target.incoming_weights.assign(m, 1.0);
    const auto allocation_ready = timings ? Clock::now() : Clock::time_point{};
    if (weighted) {
        for (std::size_t row = 0; row < n; ++row) {
            for (int edge = source.outgoing_offsets[row];
                 edge < source.outgoing_offsets[row + 1]; ++edge) {
                const int destination = source.outgoing_destinations[edge];
                const std::size_t slot = static_cast<std::size_t>(cursor[destination]++);
                target.incoming_sources[slot] = static_cast<int>(row);
                target.incoming_weights[slot] = source.outgoing_weights[edge];
            }
        }
    } else {
        for (std::size_t row = 0; row < n; ++row) {
            for (int edge = source.outgoing_offsets[row];
                 edge < source.outgoing_offsets[row + 1]; ++edge) {
                const int destination = source.outgoing_destinations[edge];
                const std::size_t slot = static_cast<std::size_t>(cursor[destination]++);
                target.incoming_sources[slot] = static_cast<int>(row);
            }
        }
    }
    if (timings) {
        const auto end = Clock::now();
        const auto seconds = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double>(b-a).count();
        };
        timings->metadata_seconds = seconds(begin, metadata_ready);
        timings->degree_count_seconds = seconds(metadata_ready, count_ready);
        timings->prefix_and_allocation_seconds = seconds(count_ready, allocation_ready);
        timings->scatter_seconds = seconds(allocation_ready, end);
        timings->total_seconds = seconds(begin, end);
    }
    return result;
}

std::vector<PageRankHostPartition> build_destination_owned_page_rank_packed(
    const std::vector<PageRankOutgoingPartition>& outgoing,
    std::uint64_t vertex_count, bool materialize_unit_weights,
    PackedRouteTransposeTimings* timings) {
    if (timings) *timings = PackedRouteTransposeTimings{};
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    auto seconds = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double>(b-a).count();
    };
    if (vertex_count == 0) {
        if (!outgoing.empty()) {
            throw std::invalid_argument(
                "empty PageRank graph cannot contain outgoing partitions");
        }
        return std::vector<PageRankHostPartition>();
    }
    if (outgoing.empty()) {
        throw std::invalid_argument(
            "PageRank pull preparation requires outgoing partitions");
    }

    const bool store_weights = materialize_unit_weights || std::any_of(
        outgoing.begin(), outgoing.end(), [](const PageRankOutgoingPartition& p) {
            return !p.outgoing_weights.empty();
        });
    std::vector<PageRankHostPartition> result(outgoing.size());
    std::uint64_t owned_total = 0;
    // Global IDs already occupy [0, vertex_count). Build the inverse ownership
    // once; every edge can then address its destination row in constant time.
    std::vector<int> global_owner(static_cast<std::size_t>(vertex_count), -1);
    std::vector<std::size_t> global_owned_local(static_cast<std::size_t>(vertex_count));
    std::vector<int> seen_devices;
    for (std::size_t rank = 0; rank < outgoing.size(); ++rank) {
        const PageRankOutgoingPartition& source = outgoing[rank];
        if (source.partition_id != static_cast<int>(rank) ||
            source.device_id < 0 ||
            std::find(seen_devices.begin(), seen_devices.end(), source.device_id) != seen_devices.end() ||
            source.owned_vertex_count > source.local_to_global.size() ||
            source.owners.size() != source.local_to_global.size() ||
            source.outgoing_offsets.size() != source.owned_vertex_count + 1U ||
            source.outgoing_offsets.empty() ||
            source.outgoing_offsets.front() != 0 ||
            source.outgoing_offsets.back() < 0 ||
            static_cast<std::size_t>(source.outgoing_offsets.back()) !=
                source.outgoing_destinations.size() ||
            (!source.outgoing_weights.empty() &&
             source.outgoing_weights.size() !=
                 source.outgoing_destinations.size()) ||
            (source.owned_vertex_count == 0 &&
             (!source.local_to_global.empty() || !source.owners.empty() ||
              !source.outgoing_destinations.empty() ||
              !source.outgoing_weights.empty()))) {
            throw std::invalid_argument(
                "outgoing PageRank partition metadata is inconsistent");
        }
        seen_devices.push_back(source.device_id);
        for (std::size_t row = 1; row < source.outgoing_offsets.size(); ++row) {
            if (source.outgoing_offsets[row] < source.outgoing_offsets[row - 1]) {
                throw std::invalid_argument(
                    "outgoing PageRank offsets must be monotone");
            }
        }
        if (!std::is_sorted(
                source.local_to_global.begin(),
                source.local_to_global.begin() +
                    static_cast<std::ptrdiff_t>(source.owned_vertex_count))) {
            throw std::invalid_argument(
                "outgoing PageRank owned IDs must be ordered");
        }
        for (std::size_t local = 0; local < source.local_to_global.size(); ++local) {
            if (source.local_to_global[local] >= vertex_count ||
                source.owners[local] < 0 ||
                static_cast<std::size_t>(source.owners[local]) >= outgoing.size() ||
                (local < source.owned_vertex_count &&
                 source.owners[local] != source.partition_id)) {
                throw std::invalid_argument(
                    "outgoing PageRank local vertex metadata is invalid");
            }
        }
        for (std::size_t local = 0; local < source.owned_vertex_count; ++local) {
            const std::size_t global = static_cast<std::size_t>(source.local_to_global[local]);
            if (global_owner[global] != -1) {
                throw std::invalid_argument("outgoing PageRank ownership must be disjoint");
            }
            global_owner[global] = static_cast<int>(rank);
            global_owned_local[global] = local;
        }
        owned_total += static_cast<std::uint64_t>(source.owned_vertex_count);

        PageRankHostPartition& destination = result[rank];
        destination.partition_id = source.partition_id;
        destination.device_id = source.device_id;
        destination.owned_vertex_count = source.owned_vertex_count;
        destination.local_to_global.assign(
            source.local_to_global.begin(),
            source.local_to_global.begin() +
                static_cast<std::ptrdiff_t>(source.owned_vertex_count));
        destination.owners.assign(source.owned_vertex_count, source.partition_id);
        destination.owned_out_weight.assign(source.owned_vertex_count, 0.0);
        destination.incoming_offsets.assign(source.owned_vertex_count + 1U, 0);
    }
    if (owned_total != vertex_count) {
        throw std::invalid_argument(
            "outgoing PageRank partitions do not cover all vertices");
    }

    for (const PageRankOutgoingPartition& source : outgoing) {
        for (std::size_t local = 0; local < source.local_to_global.size(); ++local) {
            if (source.owners[local] != global_owner[static_cast<std::size_t>(source.local_to_global[local])]) {
                throw std::invalid_argument("outgoing PageRank halo owner is inconsistent");
            }
        }
    }

    const auto metadata_end = Clock::now();
    // One route per source-local vertex, shared by both edge passes. Global
    // inverse ownership already exists; no additional per-edge storage.
    std::vector<std::vector<std::uint64_t> > routes(outgoing.size());
    std::uint64_t route_bytes = 0;
    for (std::size_t rank = 0; rank < outgoing.size(); ++rank) {
        const auto& source = outgoing[rank];
        auto& route = routes[rank];
        route.resize(source.local_to_global.size());
        for (std::size_t local = 0; local < route.size(); ++local) {
            const auto row = global_owned_local[static_cast<std::size_t>(source.local_to_global[local])];
            if (row > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("PageRank packed owner row exceeds uint32 capacity");
            route[local] = (static_cast<std::uint64_t>(source.owners[local]) << 32U) |
                static_cast<std::uint32_t>(row);
        }
        route_bytes += route.capacity() * sizeof(std::uint64_t);
    }
    const std::uint64_t inverse_bytes = global_owner.capacity()*sizeof(int) +
        global_owned_local.capacity()*sizeof(std::size_t);
    const std::uint64_t route_build_peak_bytes = route_bytes + inverse_bytes +
        seen_devices.capacity()*sizeof(int);
    // Routes now contain every owned-local destination row used by both passes.
    std::vector<std::size_t>().swap(global_owned_local);
    const auto route_end = Clock::now();
    std::vector<std::vector<std::uint64_t> > remote_sources(outgoing.size());
    std::uint64_t count_scratch_bytes = 0;
    // First pass computes outgoing denominators, incoming row sizes, and the
    // exact remote-source halo needed by each destination partition.
    for (std::size_t source_rank = 0; source_rank < outgoing.size(); ++source_rank) {
        const PageRankOutgoingPartition& source = outgoing[source_rank];
        const auto& source_routes = routes[source_rank];
        std::vector<std::size_t> last_remote_source(outgoing.size(), std::numeric_limits<std::size_t>::max());
        count_scratch_bytes = std::max(count_scratch_bytes,
            static_cast<std::uint64_t>(last_remote_source.capacity() * sizeof(std::size_t)));
        for (std::size_t source_local = 0;
             source_local < source.owned_vertex_count; ++source_local) {
            const std::uint64_t source_global = source.local_to_global[source_local];
            for (int edge_value = source.outgoing_offsets[source_local];
                 edge_value < source.outgoing_offsets[source_local + 1U];
                 ++edge_value) {
                const std::size_t edge = static_cast<std::size_t>(edge_value);
                const int target_local_value = source.outgoing_destinations[edge];
                const double weight = outgoing_weight_at(source, edge);
                if (target_local_value < 0 ||
                    static_cast<std::size_t>(target_local_value) >=
                        source.local_to_global.size() ||
                    !std::isfinite(weight) || weight < 0.0) {
                    throw std::invalid_argument(
                        "outgoing PageRank edge has invalid endpoint or weight");
                }
                const std::size_t target_local =
                    static_cast<std::size_t>(target_local_value);
                const std::uint64_t route = source_routes[target_local];
                const int target_owner = static_cast<int>(route >> 32U);
                const std::size_t target_row = static_cast<std::uint32_t>(route);
                int& row_edges = result[static_cast<std::size_t>(target_owner)].incoming_offsets[target_row + 1U];
                if (row_edges == std::numeric_limits<int>::max()) {
                    throw std::overflow_error("PageRank incoming row exceeds int32 capacity");
                }
                ++row_edges;
                result[source_rank].owned_out_weight[source_local] += weight;
                if (target_owner != static_cast<int>(source_rank) &&
                    last_remote_source[static_cast<std::size_t>(target_owner)] != source_local) {
                    remote_sources[static_cast<std::size_t>(target_owner)].push_back(source_global);
                    last_remote_source[static_cast<std::size_t>(target_owner)] = source_local;
                }
            }
            if (!std::isfinite(result[source_rank].owned_out_weight[source_local])) {
                throw std::invalid_argument("PageRank outgoing weight sum must be finite");
            }
        }
    }

    std::uint64_t count_peak_bytes = route_bytes + global_owner.capacity()*sizeof(int) +
        seen_devices.capacity()*sizeof(int) + count_scratch_bytes;
    for (const auto& x : remote_sources) count_peak_bytes += x.capacity()*sizeof(std::uint64_t);
    const auto count_end = Clock::now();
    std::vector<std::vector<int> > cursors(outgoing.size());
    for (std::size_t rank = 0; rank < result.size(); ++rank) {
        PageRankHostPartition& partition = result[rank];
        for (std::size_t row = 0; row < partition.owned_vertex_count; ++row) {
            if (partition.incoming_offsets[row + 1U] >
                std::numeric_limits<int>::max() - partition.incoming_offsets[row]) {
                throw std::overflow_error("PageRank incoming partition exceeds int32 capacity");
            }
            partition.incoming_offsets[row + 1U] +=
                partition.incoming_offsets[row];
        }
        std::vector<std::uint64_t>& halo = remote_sources[rank];
        std::sort(halo.begin(), halo.end());
        // One insertion per (source, remote owner) already guarantees uniqueness.
        partition.local_to_global.insert(
            partition.local_to_global.end(), halo.begin(), halo.end());
        if (partition.local_to_global.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("PageRank local vertex domain exceeds int32 capacity");
        }
        for (std::size_t index = 0; index < halo.size(); ++index) {
            partition.owners.push_back(global_owner[static_cast<std::size_t>(halo[index])]);
        }
        partition.incoming_sources.resize(
            static_cast<std::size_t>(partition.incoming_offsets.back()));
        if (store_weights)
            partition.incoming_weights.resize(partition.incoming_sources.size());
        cursors[rank] = partition.incoming_offsets;
    }

    std::uint64_t prefix_peak_bytes = route_bytes + global_owner.capacity()*sizeof(int) +
        seen_devices.capacity()*sizeof(int);
    for (const auto& x : remote_sources) prefix_peak_bytes += x.capacity()*sizeof(std::uint64_t);
    for (const auto& x : cursors) prefix_peak_bytes += x.capacity()*sizeof(int);
    // Incoming halo owner labels are now complete; scatter needs neither inverse.
    std::vector<int>().swap(global_owner);
    const auto prefix_end = Clock::now();
    std::uint64_t scatter_scratch_bytes = 0;
    // Second pass materializes incoming source local IDs and edge weights.
    for (std::size_t source_rank = 0; source_rank < outgoing.size(); ++source_rank) {
        const PageRankOutgoingPartition& source = outgoing[source_rank];
        const auto& source_routes = routes[source_rank];
        std::vector<std::size_t> source_in_owner(outgoing.size());
        scatter_scratch_bytes = std::max(scatter_scratch_bytes,
            static_cast<std::uint64_t>(source_in_owner.capacity() * sizeof(std::size_t)));
        for (std::size_t source_local = 0;
             source_local < source.owned_vertex_count; ++source_local) {
            const std::uint64_t source_global = source.local_to_global[source_local];
            std::fill(source_in_owner.begin(), source_in_owner.end(), std::numeric_limits<std::size_t>::max());
            source_in_owner[source_rank] = source_local;
            for (int edge_value = source.outgoing_offsets[source_local];
                 edge_value < source.outgoing_offsets[source_local + 1U];
                 ++edge_value) {
                const std::size_t edge = static_cast<std::size_t>(edge_value);
                const std::size_t target_local = static_cast<std::size_t>(
                    source.outgoing_destinations[edge]);
                const std::uint64_t route = source_routes[target_local];
                const int target_owner = static_cast<int>(route >> 32U);
                const std::size_t target_row = static_cast<std::uint32_t>(route);
                PageRankHostPartition& target =
                    result[static_cast<std::size_t>(target_owner)];
                std::size_t& source_in_target = source_in_owner[static_cast<std::size_t>(target_owner)];
                if (source_in_target == std::numeric_limits<std::size_t>::max()) {
                    const std::vector<std::uint64_t>::const_iterator halo_begin =
                        target.local_to_global.begin() +
                        static_cast<std::ptrdiff_t>(target.owned_vertex_count);
                    const std::vector<std::uint64_t>::const_iterator found =
                        std::lower_bound(
                            halo_begin, target.local_to_global.cend(), source_global);
                    if (found == target.local_to_global.cend() ||
                        *found != source_global) {
                        throw std::logic_error(
                            "PageRank halo lookup failed during pull construction");
                    }
                    source_in_target = static_cast<std::size_t>(
                        found - target.local_to_global.begin());
                }
                const std::size_t position = static_cast<std::size_t>(
                    cursors[static_cast<std::size_t>(target_owner)][target_row]++);
                target.incoming_sources[position] =
                    static_cast<int>(source_in_target);
                if (store_weights)
                    target.incoming_weights[position] = outgoing_weight_at(source, edge);
            }
        }
    }
    const auto scatter_end = Clock::now();
    if (timings) {
        timings->metadata_seconds = seconds(started, metadata_end);
        timings->route_preparation_seconds = seconds(metadata_end, route_end);
        timings->degree_and_halo_count_seconds = seconds(route_end, count_end);
        timings->prefix_halo_sort_allocation_seconds = seconds(count_end, prefix_end);
        timings->scatter_seconds = seconds(prefix_end, scatter_end);
        timings->inner_total_seconds = seconds(started, scatter_end);
        timings->route_temporary_bytes = route_bytes;
        timings->count_scratch_temporary_bytes = count_scratch_bytes;
        timings->scatter_scratch_temporary_bytes = scatter_scratch_bytes;
        // Payload capacities coexist during scatter; exclude result/input
        // arrays, vector objects/stack, and allocator bookkeeping. The general
        // algorithm has these identical buffers, with identical push order.
        std::uint64_t shared_bytes = global_owner.capacity() * sizeof(int) +
            global_owned_local.capacity() * sizeof(std::size_t) +
            seen_devices.capacity() * sizeof(int) + scatter_scratch_bytes;
        for (const auto& x : remote_sources) shared_bytes += x.capacity() * sizeof(std::uint64_t);
        for (const auto& x : cursors) shared_bytes += x.capacity() * sizeof(int);
        timings->general_scatter_temporary_payload_bytes = shared_bytes + inverse_bytes;
        timings->route_build_peak_temporary_payload_bytes = route_build_peak_bytes;
        timings->prefix_peak_temporary_payload_bytes = prefix_peak_bytes;
        timings->scatter_temporary_payload_bytes = shared_bytes + route_bytes;
        timings->packed_boundary_peak_temporary_payload_bytes = std::max(route_build_peak_bytes,
            std::max(count_peak_bytes, std::max(prefix_peak_bytes, shared_bytes + route_bytes)));
        std::uint64_t max_remote_capacity_bytes = 0;
        for (const auto& x : remote_sources)
            max_remote_capacity_bytes = std::max(max_remote_capacity_bytes,
                static_cast<std::uint64_t>(x.capacity()*sizeof(std::uint64_t)));
        // Only one push_back can reallocate at once. The old buffer is strictly
        // smaller than its final capacity, so +max capacity is a portable bound.
        timings->temporary_payload_upper_bound_bytes = std::max(
            timings->packed_boundary_peak_temporary_payload_bytes,
            count_peak_bytes + max_remote_capacity_bytes);

    }
    // Input ownership/endpoints/weights were checked once above. Incoming CSR
    // and halo metadata are constructed from those same maps; rescanning every
    // generated edge here would repeat that work.
    return result;
}

std::vector<PageRankHostPartition> reuse_symmetric_page_rank_view(
    std::vector<PageRankOutgoingPartition> outgoing, bool materialize_unit_weights) {
    std::vector<PageRankHostPartition> result(outgoing.size());
    for (std::size_t rank = 0; rank < outgoing.size(); ++rank) {
        auto& source = outgoing[rank];
        auto& target = result[rank];
        target.partition_id = source.partition_id;
        target.device_id = source.device_id;
        target.owned_vertex_count = source.owned_vertex_count;
        target.owned_out_weight.resize(source.owned_vertex_count);
        for (std::size_t row = 0; row < source.owned_vertex_count; ++row) {
            double weight = source.outgoing_offsets[row + 1] - source.outgoing_offsets[row];
            if (!source.outgoing_weights.empty()) {
                weight = 0.0;
                for (int edge = source.outgoing_offsets[row]; edge < source.outgoing_offsets[row + 1]; ++edge) {
                    const double edge_weight = source.outgoing_weights[edge];
                    if (!std::isfinite(edge_weight) || edge_weight < 0.0)
                        throw std::invalid_argument("PageRank edge weight must be finite and nonnegative");
                    weight += edge_weight;
                }
                if (!std::isfinite(weight))
                    throw std::invalid_argument("PageRank outgoing weight sum must be finite");
            }
            target.owned_out_weight[row] = weight;
        }
        target.incoming_offsets = std::move(source.outgoing_offsets);
        target.incoming_sources = std::move(source.outgoing_destinations);
        target.local_to_global = std::move(source.local_to_global);
        target.owners = std::move(source.owners);
        target.incoming_weights = std::move(source.outgoing_weights);
        if (materialize_unit_weights && target.incoming_weights.empty())
            target.incoming_weights.assign(target.incoming_sources.size(), 1.0);
    }
    return result;
}

void validate_page_rank_partitions(
    const std::vector<PageRankHostPartition>& partitions,
    std::uint64_t vertex_count) {
    if (vertex_count == 0) {
        if (!partitions.empty()) {
            throw std::invalid_argument(
                "an empty PageRank graph cannot contain partitions");
        }
        return;
    }
    if (partitions.empty()) {
        throw std::invalid_argument(
            "partitioned PageRank requires at least one partition");
    }
    if (vertex_count >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("PageRank result exceeds host address space");
    }

    std::vector<int> owner(static_cast<std::size_t>(vertex_count), -1);
    std::vector<int> seen_partition_ids(partitions.size(), 0);
    std::vector<int> seen_devices;
    seen_devices.reserve(partitions.size());
    for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
        const PageRankHostPartition& partition = partitions[rank];
        if (partition.partition_id < 0 ||
            static_cast<std::size_t>(partition.partition_id) >= partitions.size() ||
            seen_partition_ids[static_cast<std::size_t>(partition.partition_id)]++) {
            throw std::invalid_argument(
                "PageRank partition IDs must be unique and contiguous");
        }
        if (partition.partition_id != static_cast<int>(rank)) {
            throw std::invalid_argument(
                "PageRank partitions must be stored in partition-ID order");
        }
        if (partition.device_id < 0 ||
            std::find(seen_devices.begin(), seen_devices.end(),
                      partition.device_id) != seen_devices.end()) {
            throw std::invalid_argument(
                "PageRank partitions require distinct nonnegative devices");
        }
        seen_devices.push_back(partition.device_id);
        const std::size_t local_count = partition.local_to_global.size();
        if (partition.owned_vertex_count > local_count ||
            partition.owners.size() != local_count ||
            partition.owned_out_weight.size() != partition.owned_vertex_count ||
            (partition.owned_vertex_count == 0 &&
             (!partition.local_to_global.empty() ||
              !partition.incoming_sources.empty() ||
              !partition.incoming_weights.empty()))) {
            throw std::invalid_argument(
                "PageRank partition metadata has inconsistent cardinalities");
        }
        if (partition.incoming_offsets.size() !=
            partition.owned_vertex_count + 1U) {
            throw std::invalid_argument(
                "PageRank incoming CSR needs one row per owned destination");
        }
        if (!std::is_sorted(
                partition.local_to_global.begin(),
                partition.local_to_global.begin() +
                    static_cast<std::ptrdiff_t>(partition.owned_vertex_count))) {
            throw std::invalid_argument(
                "PageRank owned global IDs must be ordered");
        }
        if (partition.incoming_offsets.front() != 0 ||
            partition.incoming_offsets.back() < 0 ||
            static_cast<std::size_t>(partition.incoming_offsets.back()) !=
                partition.incoming_sources.size() ||
            (!partition.incoming_weights.empty() && partition.incoming_sources.size() !=
                partition.incoming_weights.size())) {
            throw std::invalid_argument("PageRank incoming CSR is inconsistent");
        }
        for (std::size_t row = 1;
             row < partition.incoming_offsets.size(); ++row) {
            if (partition.incoming_offsets[row] <
                partition.incoming_offsets[row - 1]) {
                throw std::invalid_argument(
                    "PageRank incoming offsets must be monotone");
            }
        }
        for (std::size_t edge = 0;
             edge < partition.incoming_sources.size(); ++edge) {
            const int source = partition.incoming_sources[edge];
            const double weight = partition.incoming_weights.empty() ? 1.0 : partition.incoming_weights[edge];
            if (source < 0 || static_cast<std::size_t>(source) >= local_count ||
                !std::isfinite(weight) || weight < 0.0) {
                throw std::invalid_argument(
                    "PageRank incoming edge has an invalid source or weight");
            }
        }
        for (std::size_t local = 0; local < local_count; ++local) {
            const std::uint64_t global = partition.local_to_global[local];
            const int declared_owner = partition.owners[local];
            if (global >= vertex_count || declared_owner < 0 ||
                static_cast<std::size_t>(declared_owner) >= partitions.size()) {
                throw std::invalid_argument(
                    "PageRank local vertex has an invalid global ID or owner");
            }
            if (local < partition.owned_vertex_count) {
                if (declared_owner != partition.partition_id ||
                    owner[static_cast<std::size_t>(global)] != -1) {
                    throw std::invalid_argument(
                        "PageRank owned vertices do not form a disjoint cover");
                }
                owner[static_cast<std::size_t>(global)] = partition.partition_id;
                const double denominator = partition.owned_out_weight[local];
                if (!std::isfinite(denominator) || denominator < 0.0) {
                    throw std::invalid_argument(
                        "PageRank outgoing weight must be finite and nonnegative");
                }
            } else if (declared_owner == partition.partition_id) {
                throw std::invalid_argument(
                    "PageRank halo vertices must have a remote owner");
            }
        }
    }
    if (std::find(owner.begin(), owner.end(), -1) != owner.end()) {
        throw std::invalid_argument(
            "PageRank partitions do not own every global vertex");
    }
    for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
        const PageRankHostPartition& partition = partitions[rank];
        for (std::size_t local = 0;
             local < partition.local_to_global.size(); ++local) {
            if (partition.owners[local] !=
                owner[static_cast<std::size_t>(partition.local_to_global[local])]) {
                throw std::invalid_argument(
                    "PageRank halo owner metadata disagrees with global ownership");
            }
        }
    }
}

PartitionedPageRankResult partitioned_page_rank_reference(
    const std::vector<PageRankHostPartition>& partitions,
    std::uint64_t vertex_count,
    const PartitionedPageRankOptions& options) {
    if (!(options.alpha > 0.0 && options.alpha < 1.0) ||
        !std::isfinite(options.alpha) || options.tolerance <= 0.0 ||
        !std::isfinite(options.tolerance) || options.max_iterations <= 0) {
        throw std::invalid_argument("invalid partitioned PageRank parameters");
    }
    validate_page_rank_partitions(partitions, vertex_count);

    PartitionedPageRankResult result;
    result.iterations = 0;
    result.residual = 0.0;
    result.converged = true;
    result.logical_halo_values = 0;
    if (vertex_count == 0) {
        return result;
    }
    const std::size_t count = static_cast<std::size_t>(vertex_count);
    std::vector<double> current(count, 1.0 / static_cast<double>(count));
    std::vector<double> next(count, 0.0);

    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        double dangling = 0.0;
        for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
            const PageRankHostPartition& partition = partitions[rank];
            for (std::size_t local = 0;
                 local < partition.owned_vertex_count; ++local) {
                const std::size_t global = static_cast<std::size_t>(
                    partition.local_to_global[local]);
                if (partition.owned_out_weight[local] == 0.0) {
                    dangling += current[global];
                }
            }
            result.logical_halo_values +=
                partition.local_to_global.size() - partition.owned_vertex_count;
        }
        const double base = (1.0 - options.alpha) /
                static_cast<double>(count) +
            options.alpha * dangling / static_cast<double>(count);
        std::fill(next.begin(), next.end(), base);
        double residual = 0.0;
        for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
            const PageRankHostPartition& partition = partitions[rank];
            for (std::size_t row = 0;
                 row < partition.owned_vertex_count; ++row) {
                double sum = 0.0;
                for (int edge = partition.incoming_offsets[row];
                     edge < partition.incoming_offsets[row + 1]; ++edge) {
                    const std::size_t local_source = static_cast<std::size_t>(
                        partition.incoming_sources[static_cast<std::size_t>(edge)]);
                    const std::size_t global_source = static_cast<std::size_t>(
                        partition.local_to_global[local_source]);
                    const int owner_rank = partition.owners[local_source];
                    const PageRankHostPartition& owner_partition =
                        partitions[static_cast<std::size_t>(owner_rank)];
                    const std::vector<std::uint64_t>::const_iterator found =
                        std::lower_bound(
                            owner_partition.local_to_global.begin(),
                            owner_partition.local_to_global.begin() +
                                static_cast<std::ptrdiff_t>(
                                    owner_partition.owned_vertex_count),
                            static_cast<std::uint64_t>(global_source));
                    if (found == owner_partition.local_to_global.begin() +
                                     static_cast<std::ptrdiff_t>(
                                         owner_partition.owned_vertex_count) ||
                        *found != global_source) {
                        throw std::logic_error(
                            "PageRank owner lookup failed after validation");
                    }
                    const std::size_t owner_local =
                        static_cast<std::size_t>(
                            found - owner_partition.local_to_global.begin());
                    const double denominator =
                        owner_partition.owned_out_weight[owner_local];
                    if (denominator > 0.0) {
                        sum += current[global_source] *
                            (partition.incoming_weights.empty() ? 1.0 :
                             partition.incoming_weights[static_cast<std::size_t>(edge)]) /
                            denominator;
                    }
                }
                const std::size_t global_destination = static_cast<std::size_t>(
                    partition.local_to_global[row]);
                next[global_destination] += options.alpha * sum;
                residual += std::fabs(
                    next[global_destination] - current[global_destination]);
            }
        }
        current.swap(next);
        result.iterations = iteration;
        result.residual = residual;
        if (residual <= static_cast<double>(count) * options.tolerance) {
            result.values.swap(current);
            return result;
        }
    }
    result.values.swap(current);
    result.converged = false;
    return result;
}

}  // namespace runtime
}  // namespace gpu_easygraph
